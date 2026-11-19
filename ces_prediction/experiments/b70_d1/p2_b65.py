"""B.70 phase 2: B.65's inclusive networks scored with the cut harness on the D1 rows (PREREGISTRATION_B70.md,
Addendum 2, C7).

Each of the 30 networks data/.b65_{v2,v2h}_incl_r{r}_f{k} (trained with the values above 3 keV kept) is run by
eval_seq.build_pred_lookup under run_b39.pooled_env of the same model and fold (cut population, cut harness statistics,
CES_MAX_VAL_SAMPLES=0, CPU, a new output directory holding copies of its metrics.json and weights); its inputs and
outputs use its own training statistics.  The T_i prediction of every D1 row is read from the lookup at the row's
(file, time) key, as eval_seq.compare does; references and rows are those of the D1 re-scoring of the fold.
Path check: the same code applied to the B.39 network of each fold reproduces its D1 squared errors bit for bit.
Readout: analyze_b65.cell (its seed and 20,000 draws) -> statements I1-I4 on the cut population, plus the inclusive
networks over the cut-trained ones on the same rows.

    py ces_prediction/experiments/b70_d1/p2_b65.py [score|readout|all]
Output: data/.b70p2/b65cut/*.npz, data/.b70p2/paper/b65_inclusive_cut.{json,md}
"""
from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
import json  # noqa: E402
import shutil  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(EXP / "b65_inclusive"))
import score_b70 as SC  # noqa: E402  (eval_seq on sys.path; set_env, guards)
import runs_b70 as RB  # noqa: E402

OUT = RB.DATA / ".b70p2" / "b65cut"
PAPER = RB.DATA / ".b70p2" / "paper"
T = "CES_TI"


def b65_dir(model, r, k):
    return RB.DATA / f".b65_{model}_incl_r{r}_f{k}"


_G = {}


def grid_keys():
    if "g" not in _G:
        from seq_data import load_grid_files
        g, _ = load_grid_files(RB.DATA, True, 3000.0)
        valid = [nm for nm in sorted(g) if np.isfinite(g[nm][:, 1:3]).any()]
        _G["g"], _G["valid"] = g, valid
    return _G["g"], _G["valid"]


def predict_rows(run_copy, fam_model, r, k, shot, row):
    """T_i prediction (float32, eV) of the network in run_copy at the D1 rows (shot, row) of fold (r, k), by
    eval_seq.build_pred_lookup under the B.39 pooled env of the model."""
    import torch
    import eval_seq as E
    env = RB.rescore_env("b39", fam_model, r, k, run_copy, 0)
    SC.set_env(env)
    metrics = json.loads((run_copy / "metrics.json").read_text(encoding="utf-8"))
    SC.guards(metrics)
    if metrics["seq_model"] != env["CES_SEQ_MODEL"]:
        raise SystemExit(f"FATAL: {run_copy.name} is {metrics['seq_model']}, env says {env['CES_SEQ_MODEL']}")
    man = json.loads((Path(env["CES_SPLIT_DIR"]) / "split_manifest.json").read_text(encoding="utf-8"))
    lk = E.build_pred_lookup(RB.DATA, sorted(set(man["test_files"])), E._load_stats(metrics), torch.device("cpu"), [], 0)["full"]
    g, valid = grid_keys()
    out = np.zeros(len(shot), np.float32)
    for u in np.unique(shot):
        sel = np.flatnonzero(shot == u)
        nm = valid[int(u)]
        arr = g[nm]
        gi = np.flatnonzero(np.isfinite(arr[:, 1:3]).any(1))
        for j in sel:
            out[j] = lk[nm][float(arr[gi[row[j]], 0])][0]
    return out


def stage_score():
    OUT.mkdir(parents=True, exist_ok=True)
    for r in (2, 3, 4):
        for k in range(5):
            p = OUT / f"r{r}_f{k}.npz"
            if p.exists():
                continue
            z = np.load(RB.new_dir("b39", "v2", r, k) / "comparison_errors_test.npz")
            shot, row, y = z[f"{T}_shot"], z[f"{T}_row"], z[f"{T}_y_true"]
            res = {}
            # path check: the B.39 network through the same code reproduces its D1 squared errors bit for bit
            pc = predict_rows(RB.new_dir("b39", "v2", r, k), "v2", r, k, shot, row)
            if not np.array_equal((pc - y) ** 2, z[f"{T}_se_model"]):
                raise SystemExit(f"FATAL: r{r} f{k}: the row path does not reproduce the D1 errors of the B.39 network")
            for m in ("v2", "v2h"):
                src = b65_dir(m, r, k)
                cp = OUT / f"{src.name.lstrip('.')}"
                (cp / "weights").mkdir(parents=True, exist_ok=True)
                shutil.copy2(src / "metrics.json", cp / "metrics.json")
                shutil.copy2(src / "weights" / "seq_lstm.pth", cp / "weights" / "seq_lstm.pth")
                mm = json.loads((src / "metrics.json").read_text(encoding="utf-8"))
                if mm["seq_model"] != m:
                    raise SystemExit(f"FATAL: {src.name} is not {m}")
                pred = predict_rows(cp, m, r, k, shot, row)
                res[f"se_{m}_incl"] = ((pred - y) ** 2).astype(np.float32)
            np.savez(p, shot=shot, row=row, y=y, path_check_bit_identical=np.array(True), **res)
            print(f"r{r} f{k}: {len(y)} rows; inclusive networks scored with the cut harness", flush=True)


def stage_readout():
    import analyze_b65 as AB
    D = {}
    for r in AB.REPS:
        parts = []
        for k in range(5):
            q = np.load(OUT / f"r{r}_f{k}.npz")
            c39 = np.load(RB.new_dir("b39", "v2", r, k) / "comparison_errors_test.npz")
            t39 = np.load(RB.new_dir("b39", "v2h", r, k) / "comparison_errors_test.npz")
            if not (np.array_equal(q["shot"], c39[f"{T}_shot"]) and np.array_equal(q["row"], c39[f"{T}_row"])):
                raise SystemExit(f"FATAL: r{r} f{k}: rows differ")
            parts.append({"shot": q["shot"], "y": q["y"].astype(np.float64),
                          "c": q["se_v2_incl"].astype(np.float64), "t": q["se_v2h_incl"].astype(np.float64),
                          "c39": c39[f"{T}_se_model"].astype(np.float64), "t39": t39[f"{T}_se_model"].astype(np.float64),
                          **{e: c39[f"{T}_se_{src}"].astype(np.float64) for e, src in AB.SRC.items()}})
        d = {kk: np.concatenate([p[kk] for p in parts]) for kk in parts[0]}
        u, f = np.unique(d["shot"], return_inverse=True)
        d["f"], d["n"] = f, len(u)
        D[r] = d
    AB.EST = ("c", "t", "gp", "hold", "off", "pchip", "c39", "t39")
    AB.LAB = {**AB.LAB, "c39": "network trained on the cut population (B.39)", "t39": "twin trained on the cut population (B.39)"}
    AB.PAIRS = AB.PAIRS + [("c", "c39"), ("t", "t39"), ("c39", "gp"), ("c39", "t39")]
    rng = np.random.default_rng(AB.SEED)
    A = AB.cell(D, lambda d: np.ones(len(d["y"]), bool), rng)
    st = {"I1": ("network over causal GP, MSE skill", A["mse|c|gp"]),
          "I2": ("network over twin, MSE skill", A["mse|c|t"]),
          "I3": ("network over causal GP, Huber skill", A["huber|c|gp"]),
          "I4": ("network over twin, Huber skill", A["huber|c|t"])}
    st = {k: {"text": t + " (inclusive networks, cut population, D1 rows)", "value": v, "holds": bool(v[1] > 0)}
          for k, (t, v) in st.items()}
    out = {"registration": AB.REG + " ; B.70 phase 2 (Addendum 2, C7): cut harness, D1 rows",
           "seed": AB.SEED, "n_boot": AB.NBOOT, "statements": st, "all_rows": A,
           "expectations": {k: "B.65's expectations were for the inclusive population; not applicable" for k in st}}
    PAPER.mkdir(parents=True, exist_ok=True)
    (PAPER / "b65_inclusive_cut.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    (PAPER / "b65_inclusive_cut.md").write_text(AB.render(out, title_pop="cut"), encoding="utf-8", newline="\n")
    for k, v in st.items():
        print(k, v["text"], [round(x, 4) for x in v["value"]], "holds" if v["holds"] else "does not hold", flush=True)


if __name__ == "__main__":
    s = sys.argv[1] if len(sys.argv) > 1 else "all"
    if s in ("score", "all"):
        stage_score()
    if s in ("readout", "all"):
        stage_readout()
    os._exit(0)
