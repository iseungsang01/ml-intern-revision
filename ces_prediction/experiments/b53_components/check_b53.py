"""B.53 checks before any GPU run (PREREGISTRATION_B53.md section 3).  CPU only, no training.

  (1) the extra channels reach the T_i branch and only it: v2hx built on the CPU for each table width;
      the T_i encoder's first-layer input width grows by K, the V_rot branch has v2h's shapes; a forward
      pass on real blocks (seq_data with CES_SEQ_EXTRA_TI) changes the T_i output and leaves the V_rot
      output bit-identical when the extra channels are perturbed; the fast channels reach neither; with
      K = 0, v2hx equals v2h (weights under the same seed, forward); the table reaches the input unscaled;
      train_seq / eval_seq see v2hx through the shared registry dict.
  (2) the feature tables match the grid rows of every file (time equality; seq_data's own matcher).
  (3) the constants of a split come from its training discharges only (recomputed; the values with the
      test or validation discharges added differ; the tables equal (x - mean) / std of the constants).
  (4) with no extra table the runner's env is the env of the B.39 twin, exactly (whole env), and agrees
      with the metrics.json of the twin runs.
  (5) arm F's env differs from B.39's only in CES_SEQ_EPOCHS, CES_SEQ_FIXED_EPOCHS and the output dir.

    py ces_prediction/experiments/b53_components/check_b53.py
Output: docs/paper/b53_checks.md (also printed).  Exit 1 if any check fails.
"""
from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "4"

import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_b53 as R  # noqa: E402
import make_features as MF  # noqa: E402

REPO = HERE.parents[2]
OUT_MD = REPO / "docs" / "paper" / "b53_checks.md"
T0 = time.time()
LINES, FAILS = [], []


def log(m):
    print(f"[{time.time() - T0:6.0f}s] {m}", flush=True)


def put(m=""):
    LINES.append(m)
    print(m, flush=True)


def verdict(name, ok, detail):
    put(f"- **{name}**: {'PASS' if ok else 'FAIL'} -- {detail}")
    if not ok:
        FAILS.append(name)


# ------------------------------------------------------------------------------ (1)
def check_model(grid):
    import torch
    torch.set_num_threads(4)
    import register_v2hx as RV          # puts seq/ first on sys.path
    import seq_data
    from model_seq_v2h import SeqCESLSTMv2Hist
    from model_seq_v2hx import SeqCESLSTMv2HistX
    put("## (1) The extra channels reach the T_i branch and only it")
    put()
    put("| table | K | T_i encoder layer-0 W_ih | V_rot encoder layer-0 W_ih | parameters (v2h 347,970) | "
        "T_i output change when the extras are perturbed | V_rot output change | output change when the fast channels are perturbed |")
    put("|---|---|---|---|---|---|---|---|")
    os.environ.pop("CES_SEQ_EXTRA_VT", None)
    os.environ.pop("CES_SEQ_EXTRA_TI", None)
    torch.manual_seed(0)
    h = SeqCESLSTMv2Hist()
    ok = True
    names = sorted(grid)[:3] + sorted(grid)[-3:]
    stats = seq_data.fit_stats(grid, {"bes": 9, "ecei": 4, "mc": 2}, sorted(grid))
    for combo in R.COMBOS:
        d = R.feat_dir(combo, 42)
        os.environ["CES_SEQ_EXTRA_TI"] = str(d)
        k, exempt, _ = seq_data.extra_ti_meta()
        torch.manual_seed(0)
        m = SeqCESLSTMv2HistX()
        m.eval()
        wi, wv = tuple(m.lstm_ti.weight_ih_l0.shape), tuple(m.lstm_vt.weight_ih_l0.shape)
        same_vt = all(tuple(a.shape) == tuple(b.shape) for a, b in zip(m.lstm_vt.parameters(), h.lstm_vt.parameters()))
        same_rest_ti = all(tuple(getattr(m.lstm_ti, n).shape) == tuple(getattr(h.lstm_ti, n).shape)
                           for n, _ in h.lstm_ti.named_parameters() if n != "weight_ih_l0")
        # real blocks through seq_data with the table set (CES_PER_SHOT_NORM=1 as the twin's env)
        dts = {}
        for nm in names:
            arr = grid[nm]
            extra = seq_data._load_extra_vt(nm, arr[:, 0].astype(np.float64), k, exempt, d)
            arr2 = np.concatenate([arr, extra.astype(np.float32)], axis=1)
            dims = {"bes": 9, "ecei": 4, "mc": 2, "n_fast": 15, "extra_vt": 0, "extra_ti": k}
            for b in seq_data.build_blocks(arr2, dims, stats, per_shot_norm=True):
                x = torch.from_numpy(b["x"]).unsqueeze(0)
                if x.shape[-1] != 22 + k:
                    raise SystemExit(f"FATAL: block width {x.shape[-1]} != 22 + {k}")
                with torch.no_grad():
                    y0 = m(x)
                    xe = x.clone(); xe[..., 22:] += torch.randn_like(xe[..., 22:])
                    ye = m(xe)
                    xf = x.clone(); xf[..., :15] += torch.randn_like(xf[..., :15])
                    yf = m(xf)
                dts.setdefault("ti_e", []).append(float((ye[..., 0] - y0[..., 0]).abs().max()))
                dts.setdefault("vt_e", []).append(float((ye[..., 1] - y0[..., 1]).abs().max()))
                dts.setdefault("f", []).append(float((yf - y0).abs().max()))
        ti_e, vt_e, f = max(dts["ti_e"]), max(dts["vt_e"]), max(dts["f"])
        row_ok = (wi == (640, 7 + k) and wv == (256, 7) and same_vt and same_rest_ti and ti_e > 1e-4
                  and vt_e == 0.0 and f == 0.0 and m.n_params == 347970 + 640 * k)
        ok &= row_ok
        put(f"| {d.name} | {k} | {wi} (v2h (640, 7)) | {wv} (v2h {tuple(h.lstm_vt.weight_ih_l0.shape)}) | {m.n_params:,} | "
            f"{ti_e:.3g} | {vt_e:.3g} | {f:.3g} |")
    put()
    # K = 0: v2hx is v2h
    os.environ.pop("CES_SEQ_EXTRA_TI", None)
    torch.manual_seed(7)
    h0 = SeqCESLSTMv2Hist()
    torch.manual_seed(7)
    x0 = SeqCESLSTMv2HistX()
    sd_h, sd_x = h0.state_dict(), x0.state_dict()
    eq_w = sd_h.keys() == sd_x.keys() and all(torch.equal(sd_h[q], sd_x[q]) for q in sd_h)
    xx = torch.randn(2, 50, 22)
    h0.eval(); x0.eval()
    with torch.no_grad():
        eq_f = torch.equal(h0(xx), x0(xx))
    # the table reaches the input unscaled: load_grid_files with CES_SEQ_EXTRA_TI on CD_s42, compare columns
    os.environ["CES_SEQ_EXTRA_TI"] = str(R.feat_dir("CD", 42))
    os.environ["CES_TI_SPIKE_CUT_EV"] = "3000"
    g2, dims2 = seq_data.load_grid_files(R.DATA, True)
    maxdiff = 0.0
    for nm in sorted(g2):
        z = np.load(R.feat_dir("CD", 42) / f"mus_features_s{nm[1:].split('.')[0]}.npz")
        maxdiff = max(maxdiff, float(np.abs(g2[nm][:, 3 + 15:] - z["feat"]).max()))
    blk = seq_data.build_blocks(g2[sorted(g2)[0]], dims2, stats, per_shot_norm=True)[0]
    z0 = np.load(R.feat_dir("CD", 42) / f"mus_features_s{sorted(g2)[0][1:].split('.')[0]}.npz")
    t_idx = {round(float(t), 4): i for i, t in enumerate(z0["time"])}
    blk_diff = float(np.abs(blk["x"][:, 22:] - z0["feat"][[t_idx[round(float(t), 4)] for t in blk["time"]]]).max())
    os.environ.pop("CES_SEQ_EXTRA_TI", None)
    # registry
    import train_seq
    import eval_seq
    reg = RV.register()
    reg_ok = train_seq.SEQ_MODELS is reg and eval_seq.SEQ_MODELS is reg and reg["v2hx"] is SeqCESLSTMv2HistX
    verdict("(1) routing", ok and eq_w and eq_f and maxdiff == 0.0 and blk_diff == 0.0 and reg_ok and dims2["extra_ti"] == 5,
            f"T_i layer-0 input 7 -> 7 + K for K = 1, 4, 5; V_rot branch shapes equal to v2h; perturbing the extras moves "
            f"T_i and leaves V_rot bit-identical; perturbing BES/ECEI/MC changes nothing; K = 0: weights under one seed "
            f"equal to v2h {eq_w}, forward equal {eq_f}; table columns in load_grid_files vs table max |diff| {maxdiff:g}, "
            f"in the block x[:, 22:] {blk_diff:g} (unscaled, exempt); v2hx in the registry dict of train_seq and eval_seq {reg_ok}")
    put("Code path: seq_data.load_grid_files appends the table after the base columns (seq_data.py 164-167, "
        "_load_extra_vt 76-108, every channel exempt so lines 102-107 scale nothing); build_blocks writes it to "
        "x[:, 22:] (seq_data.py 246-248, 286-287); v2hx reads x[..., 15:22] and x[..., 22:22+K] into lstm_ti and "
        "x[..., 15:22] into lstm_vt (model_seq_v2hx.py forward). v2h itself (model_seq_v2h.py 28-40) passes "
        "n_extra_ti=0 and reads x[..., 15:22] only, so the table would not reach it.")
    put()


# ------------------------------------------------------------------------------ (2)
def check_tables(grid):
    import seq_data
    put("## (2) The feature tables match the grid rows of every file")
    put()
    put("| table | files | rows | time equal to the grid (rows) | seq_data matcher hits | zero rows (non-positive BES) | size on disk (MB) |")
    put("|---|---|---|---|---|---|---|")
    ok = True
    for s in R.SEEDS:
        for combo in R.COMBOS:
            d = R.feat_dir(combo, s)
            meta = json.loads((d / "feature_meta.json").read_text(encoding="utf-8"))
            k = meta["k"]
            rows = eq = hits = zero = 0
            nfiles = 0
            for nm in sorted(grid):
                p = d / f"mus_features_s{nm[1:].split('.')[0]}.npz"
                z = np.load(p)
                t = grid[nm][:, 0].astype(np.float64)
                nfiles += 1
                rows += len(t)
                eq += int(len(z["time"]) == len(t) and np.array_equal(z["time"], t)) * len(t)
                if z["feat"].shape != (len(t), k):
                    ok = False
                zero += int((z["feat"] == 0).all(1).sum())
                table = {round(float(x), 4) for x in z["time"]}
                hits += sum(round(float(x), 4) in table for x in t)
            files_on_disk = len(list(d.glob("mus_features_s*.npz")))
            mb = sum(p.stat().st_size for p in d.glob("*")) / 2 ** 20
            row_ok = nfiles == 641 and files_on_disk == 641 and eq == rows and hits == rows and zero == meta["zero_rows_nonpositive_bes"]
            ok &= row_ok
            put(f"| {d.name} | {files_on_disk} | {rows:,} | {eq:,} | {hits:,} | {zero} | {mb:.1f} |")
    put()
    verdict("(2) grid rows", ok, "every table has 641 files whose times equal the grid times of seq_data.load_grid_files "
            "row for row (100 %; seq_data aborts below 99 %); zero rows are exactly the rows with a non-positive BES channel")
    put()


# ------------------------------------------------------------------------------ (3)
def check_constants(grid, raw):
    put("## (3) The constants of a split come from its training discharges only")
    put()
    put("| split | train / val / test files | overlap train with val or test | rows | max |mean - recomputed| | "
        "max |std - recomputed| | max |table - (x - mean)/std| | mean of c with test added minus train-only | with val added |")
    put("|---|---|---|---|---|---|---|---|---|")
    ok = True
    for s in R.SEEDS:
        man = MF.manifest(s)
        tr, va, te = list(man["train_files"]), list(man["val_files"]), list(man["test_files"])
        overlap = len(set(tr) & (set(va) | set(te)))
        mean, std, n = MF.constants(raw, tr)
        mean_t, std_t, _ = MF.constants(raw, tr + te)
        mean_v, _, _ = MF.constants(raw, tr + va)
        dm = ds = dt = 0.0
        for combo in R.COMBOS:
            d = R.feat_dir(combo, s)
            meta = json.loads((d / "feature_meta.json").read_text(encoding="utf-8"))
            idx = [MF.ALL.index(c) for c in meta["components"]]
            dm = max(dm, max(abs(meta["mean"][c] - mean[i]) for c, i in zip(meta["components"], idx)))
            ds = max(ds, max(abs(meta["std"][c] - std[i]) for c, i in zip(meta["components"], idx)))
            for nm in sorted(raw):
                X, bad = raw[nm]
                Z = (X[:, idx] - mean[idx]) / std[idx]
                Z[bad] = 0.0
                z = np.load(d / f"mus_features_s{nm[1:].split('.')[0]}.npz")["feat"]
                dt = max(dt, float(np.abs(z - Z.astype(np.float32)).max()))
        differs = bool(np.abs(mean_t - mean).max() > 1e-6 and np.abs(mean_v - mean).max() > 1e-6)
        row_ok = overlap == 0 and dm < 1e-12 and ds < 1e-12 and dt == 0.0 and differs and n == json.loads(
            (R.feat_dir("C", s) / "feature_meta.json").read_text(encoding="utf-8"))["constants_rows"]
        ok &= row_ok
        put(f"| {s} | {len(tr)} / {len(va)} / {len(te)} | {overlap} | {n:,} | {dm:.2g} | {ds:.2g} | {dt:.2g} | "
            f"{mean_t[0] - mean[0]:+.4f} | {mean_v[0] - mean[0]:+.4f} |")
    put()
    verdict("(3) constants", ok, "recomputed from the manifest train_files alone they equal the stored constants and "
            "rebuild every table exactly; adding the test (or validation) discharges moves them, so no other discharge entered")
    put()


# ------------------------------------------------------------------------------ (4) and (5)
def check_envs():
    put("## (4) With no extra table the runner reproduces the env of the B.39 twin")
    put()
    ok4, rows4 = True, []
    for s in R.SEEDS:
        d = R.B39.twin_dir(s, s)
        mine, ref = R.c_env(None, s, d), R.B39.split_env("v2h", s, s, d)
        equal = mine == ref
        m = json.loads((d / "metrics.json").read_text(encoding="utf-8"))
        rec_ok = (m["seq_model"] == mine["CES_SEQ_MODEL"] == "v2h" and m["init_seed"] == int(mine["CES_INIT_SEED"]) == s
                  and m["seed"] == int(mine["CES_SEED"]) and m["per_shot_input_norm"] == (mine["CES_PER_SHOT_NORM"] == "1")
                  and m["drop_stuck_targets"] == (mine["CES_DROP_STUCK_TARGETS"] == "1") and m["fixed_epochs"] is False
                  and mine["CES_SEQ_FIXED_EPOCHS"] == "0" and m["extra_ti_channels"] == 0 and "CES_SEQ_EXTRA_TI" not in mine
                  and m["epochs_run"] <= int(mine["CES_SEQ_EPOCHS"]))
        with_table = R.env_diff(R.ces_vars(R.c_env("D", s, d)), R.ces_vars(ref))
        keys = sorted(k for k, _, _ in with_table)
        ok4 &= equal and rec_ok and keys == ["CES_SEQ_EXTRA_TI", "CES_SEQ_MODEL"]
        rows4.append(f"| {s} | {d.name} | {len(mine)} | {equal} | {rec_ok} (epochs run {m['epochs_run']}, best {m['best_epoch']}) | "
                     f"{', '.join(keys)} |")
    put("| split | B.39 twin run | variables in the env | env equal (whole dict) | agrees with the twin's metrics.json | "
        "variables that change when table D is set (same output dir) |")
    put("|---|---|---|---|---|---|")
    for r in rows4:
        put(r)
    put()
    verdict("(4) twin env", ok4, "c_env(None, ...) == run_b39.split_env('v2h', s, s, twin dir) for all four splits; with a "
            "table only CES_SEQ_EXTRA_TI and CES_SEQ_MODEL (v2h -> v2hx) change")
    put()
    put("## (5) Arm F differs from B.39 only in the epoch budget and the output directory")
    put()
    put("| run | B.39 reference | non-CES variables equal | CES variables that differ |")
    put("|---|---|---|---|")
    ok5 = True
    for r in R.plan(["F"]):
        _, label, out, model, s = r[:5]
        mine = r[6]()
        d_ref, ref = R.b39_reference("F", model, s)
        non = {k: v for k, v in mine.items() if not k.startswith("CES_")} == {k: v for k, v in ref.items() if not k.startswith("CES_")}
        diff = R.env_diff(R.ces_vars(mine), R.ces_vars(ref))
        keys = sorted(k for k, _, _ in diff)
        good = (non and keys == ["CES_OUTPUT_DIR", "CES_SEQ_EPOCHS", "CES_SEQ_FIXED_EPOCHS"]
                and mine["CES_SEQ_EPOCHS"] == "40" and mine["CES_SEQ_FIXED_EPOCHS"] == "1")
        ok5 &= good
        put(f"| {out.name} | {d_ref.name} | {non} | " + "; ".join(f"{k}: {a.split(chr(92))[-1]} -> {b.split(chr(92))[-1]}" for k, a, b in diff) + " |")
    put()
    verdict("(5) fixed-budget env", ok5, "for all eight runs exactly CES_SEQ_EPOCHS (80 -> 40), CES_SEQ_FIXED_EPOCHS (0 -> 1) "
            "and CES_OUTPUT_DIR differ; train_seq.py 198-203 / 310-315: with CES_SEQ_FIXED_EPOCHS=1 no early stop and the final weights are saved")
    put()


def main():
    put("# B.53 checks before any GPU run")
    put()
    put(f"Registered: ces_prediction/experiments/PREREGISTRATION_B53.md (section 3). Written by "
        f"ces_prediction/experiments/b53_components/check_b53.py on {time.strftime('%Y-%m-%d %H:%M')}; CPU only, "
        "no network trained. Real data: data/s*.csv (641 files).")
    put()
    grid = MF.load_grid()
    raw = MF.raw_components(grid, MF.components_fn())
    log(f"grid {len(grid)} files")
    check_model(grid)
    check_tables(grid)
    check_constants(grid, raw)
    check_envs()
    put("## Result")
    put()
    put(f"{'All five checks pass.' if not FAILS else 'FAILED: ' + ', '.join(FAILS)}")
    OUT_MD.write_text("\n".join(LINES) + "\n", encoding="utf-8", newline="\n")
    log(f"written {OUT_MD}")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
