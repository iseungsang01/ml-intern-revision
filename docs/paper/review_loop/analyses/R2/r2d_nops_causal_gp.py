"""R2-d (REGISTRATION.md): attach a causal-GP (and persistence) arm to the per_shot = 0
artifacts data/.seq_v2_nops_lstm_s{42,1,7,123} and their per_shot = 1 siblings
data/.seq_v2_lstm_s{s}. EARLIER protocol (runner_common.FULL_ENV: W = 4 harness rows,
held-free scoring, no 3 keV cut, no per-file cap).

The row population is re-derived with eval_seq.compare()'s harness calls (no model) and
accepted only if count, shot ids, dt_ms and the PCHIP squared errors reproduce the stored npz.
Run from the repo root:  py docs/paper/review_loop/analyses/R2/r2d_nops_causal_gp.py
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

import r2_common as C

sys.path.insert(0, str(C.CES / "experiments"))
import runner_common as rc  # noqa: E402
from run_b1_gate import AMBIENT_VARS  # noqa: E402

METHODS = ("persistence", "linear", "pchip", "ar_local")
TARGETS = ("CES_TI", "CES_VT")


def pin_earlier_env():
    for v in AMBIENT_VARS + ("CES_MAX_SAMPLES_PER_FILE",):
        os.environ.pop(v, None)
    os.environ.update(rc.FULL_ENV)
    os.environ["CES_DROP_STUCK_TARGETS"] = "1"
    os.environ["CES_PER_SHOT_NORM"] = "0"
    os.environ["CES_TI_SPIKE_CUT_EV"] = "0"


def derive(seed):
    from evaluate import _load_stats, _persistence_from_history, build_clean_val_subset  # noqa: PLC0415
    import baselines_interpolation as B  # noqa: PLC0415
    stats = _load_stats(json.loads((C.DATA / f".sf_iter009_s{seed}" / "metrics.json").read_text(encoding="utf-8")))
    man = json.loads((rc.SPLIT_SRC[seed] / "split_manifest.json").read_text(encoding="utf-8"))
    dataset, idxs, _ = build_clean_val_subset(C.DATA, 4, stats, set(man["test_files"]), 40000, seed)
    path_to_idx = {p: i for i, p in enumerate(dataset.valid_files)}
    time_col = int(dataset._column_slices["time"])
    tcols = [int(c) for c in dataset._column_slices["target"]]
    mean = torch.as_tensor(stats["target"]["mean"], dtype=torch.float32)
    std = torch.as_tensor(stats["target"]["std"], dtype=torch.float32)

    def to_phys(t):
        return (t * std + mean).numpy()

    tgt, keep, shot, row, dts = [], [], [], [], []
    base = {m: [] for m in METHODS}
    for batch in DataLoader(Subset(dataset, idxs), batch_size=512, shuffle=False):
        pers, has = _persistence_from_history(batch["ces_history"])
        files = batch["file"]
        rows = [int(r) for r in batch["row_index"].tolist()]
        b = len(rows)
        tgt.append(to_phys(batch["target"]))
        keep.append(((batch["target_mask"] > 0.5) & has).numpy())
        pp = to_phys(pers)
        arr = {m: np.full((b, 2), np.nan) for m in METHODS}
        dt = np.full((b, 2), np.nan)
        for j in range(b):
            fa = dataset.file_arrays[path_to_idx[files[j]]]
            for t in range(2):
                times, values, tt = B.build_neighbor_set(fa, time_col, tcols[t], rows[j])
                past = times < tt
                if np.any(past):
                    dt[j, t] = tt - float(times[past].max())
                for m in METHODS:
                    arr[m][j, t] = float(pp[j, t]) if m == "persistence" else B.PREDICTORS[m](times, values, tt)
        for m in METHODS:
            base[m].append(arr[m])
        dts.append(dt)
        shot.append(np.array([path_to_idx[f] for f in files], dtype=np.int64))
        row.append(np.array(rows, dtype=np.int64))
    tgt = np.concatenate(tgt)
    keep = np.concatenate(keep)
    base = {m: np.concatenate(v) for m, v in base.items()}
    dts = np.concatenate(dts) * 1000.0
    shot = np.concatenate(shot)
    row = np.concatenate(row)
    out = {}
    for t, name in enumerate(TARGETS):
        valid = keep[:, t].copy()
        for m in METHODS:
            valid &= ~np.isnan(base[m][:, t])
        y = tgt[valid, t]
        d = {"shot": shot[valid], "row": row[valid], "dt_ms": dts[valid, t], "y": y,
             "pchip": (base["pchip"][valid, t] - y) ** 2, "linear": (base["linear"][valid, t] - y) ** 2,
             "persistence": (base["persistence"][valid, t] - y) ** 2}
        gp = np.empty(int(valid.sum()))
        for i, (s_, r_) in enumerate(zip(d["shot"], d["row"])):
            times, values, tt = B.build_neighbor_set(dataset.file_arrays[int(s_)], time_col, tcols[t], int(r_))
            gp[i] = B.predict_gp_causal(times, values, tt)
        if np.isnan(gp).any():
            raise SystemExit(f"FATAL: causal GP returned NaN on {int(np.isnan(gp).sum())} valid rows")
        d["gp_causal"] = (gp - y) ** 2
        out[name] = d
    return out


def main():
    t0 = time.time()
    pin_earlier_env()
    res = {"protocol": "EARLIER protocol (runner_common.FULL_ENV): W=4 harness rows, test cap 40000, held-free "
                       "scoring, no T_i cut, no per-file cap; control stats .sf_iter009_s{s}; splits "
                       "runner_common.SPLIT_SRC. Not the confirmed W=2 protocol.",
           "bootstrap": {"B": C.B_BOOT, "seed": C.SEED_BOOT, "cluster": "shot"}, "splits": {}}
    for s in C.SEEDS:
        rows = derive(s)
        a0 = np.load(C.DATA / f".seq_v2_nops_lstm_s{s}" / "comparison_errors_test.npz")
        a1 = np.load(C.DATA / f".seq_v2_lstm_s{s}" / "comparison_errors_test.npz")
        cell = {}
        for name in TARGETS:
            d = rows[name]
            chk = {"n": int(len(d["shot"])), "n_npz": int(len(a0[f"{name}_shot"]))}
            if chk["n"] != chk["n_npz"] or not np.array_equal(d["shot"], a0[f"{name}_shot"]):
                raise SystemExit(f"FATAL: s{s} {name}: re-derived rows differ from the npz ({chk})")
            chk["max_ddt_ms"] = float(np.max(np.abs(d["dt_ms"] - a0[f"{name}_dt_ms"])))
            chk["pchip_se_bit_identical"] = bool(np.array_equal(d["pchip"].astype(a0[f"{name}_se_pchip"].dtype),
                                                                a0[f"{name}_se_pchip"]))
            chk["pchip_se_max_rel"] = float(np.max(np.abs(d["pchip"] - a0[f"{name}_se_pchip"])
                                                   / np.maximum(a0[f"{name}_se_pchip"], 1e-6)))
            if chk["max_ddt_ms"] > 1e-6 or not np.allclose(d["pchip"], a0[f"{name}_se_pchip"], rtol=1e-5, atol=1e-6):
                raise SystemExit(f"FATAL: s{s} {name}: dt_ms / PCHIP errors not reproduced ({chk})")
            sib = bool(len(a1[f"{name}_shot"]) == chk["n"]
                       and np.array_equal(a1[f"{name}_se_pchip"], a0[f"{name}_se_pchip"]))
            m0 = a0[f"{name}_se_model"].astype(np.float64)
            rec = {"check": chk, "sibling_rows_identical": sib,
                   "rmse": {"per_shot0": float(np.sqrt(m0.mean())),
                            **{k: float(np.sqrt(d[k].mean())) for k in ("gp_causal", "persistence", "pchip", "linear")}},
                   "per_shot0_vs": {k: C.skill(d["shot"], m0, d[k]) for k in ("gp_causal", "persistence", "pchip")}}
            if sib:
                m1 = a1[f"{name}_se_model"].astype(np.float64)
                rec["rmse"]["per_shot1"] = float(np.sqrt(m1.mean()))
                rec["per_shot1_vs"] = {k: C.skill(d["shot"], m1, d[k]) for k in ("gp_causal", "persistence", "pchip")}
                rec["per_shot1_vs_per_shot0"] = C.skill(d["shot"], m1, m0)
            cell[name] = rec
            line = (f"[R2d] s{s} {name}: n={chk['n']} pchip bit-identical={chk['pchip_se_bit_identical']} | per_shot=0 vs "
                    + " ".join(f"{k} {C.fmt(v)}" for k, v in rec["per_shot0_vs"].items()))
            if sib:
                line += (" | per_shot=1 vs gp_causal " + C.fmt(rec["per_shot1_vs"]["gp_causal"])
                         + " | ps1 vs ps0 " + C.fmt(rec["per_shot1_vs_per_shot0"]))
            print(line, flush=True)
            np.savez(C.DATA / f".seq_v2_nops_lstm_s{s}" / f"r2d_added_arms_{name}.npz",
                     shot=d["shot"], row=d["row"], y_true=d["y"], se_gp_causal=d["gp_causal"],
                     se_persistence=d["persistence"])
        res["splits"][f"s{s}"] = cell
        res["minutes"] = round((time.time() - t0) / 60, 1)
        (C.HERE / "r2d_nops_causal_gp.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(f"wrote r2d_nops_causal_gp.json ({res['minutes']} min)")


if __name__ == "__main__":
    main()
