"""B.70 phase 2: the readouts that sit on B.54's row tables -- B.60 (relative error, with its Addendum 1 on the filters),
B.61 parts A and B (absolute and robust scores), B.66 (regimes) -- on the D1 rows (PREREGISTRATION_B70.md, Addendum 2).

Each script's own main / stages run with run_b54 pointed at the D1 row tables (p2_common.b54_d1), the filter caches
at data/.b70p2/b56/, every pooled set = r2-r4 (r2 alone where a script reports its development set), outputs in
data/.b70p2/paper/.  C2: the checks that compared recorded old values are pointed at D1 references:
  B.60  SEC2 (section-2 percentiles of |e| per estimator) and A1_MED (KFX/KF0 medians): recomputed here directly from
        the D1 npz files / the D1 filter caches, independently of the row tables;
  B.61  b39_primary.json and b54_pooled_rows.json of data/.b70p2/paper/ (the D1 B.39 P and the D1 B.54 readout);
  B.66  unchanged (its checks are row-independent or read the D1 B.54 checks).

    py ces_prediction/experiments/b70_d1/p2_rows.py b60|b61|b66
"""
from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
import shutil  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
sys.path.insert(0, str(HERE))
for sub in ("b54_pooled_rows", "b60_relative_error", "b61_absolute_robust", "b66_regimes"):
    sys.path.insert(0, str(EXP / sub))
import p2_common as PC  # noqa: E402

T = "CES_TI"


def d1_percentiles():
    """Mean over r2-r4 of the 50/75/90th percentiles of |e| per estimator, from the D1 npz files directly."""
    src = {"c": ("v2", "se_model"), "t": ("v2h", "se_model"), "gp": ("v2", "se_gp_causal"),
           "hold": ("v2", "se_persistence"), "off": ("v2", "se_gp")}
    out = {}
    for e, (m, key) in src.items():
        per = []
        for r in PC.REPEATS:
            v = np.concatenate([np.load(PC.d1_run_dir(m, r, k) / "comparison_errors_test.npz")[f"{T}_{key}"].astype(np.float64)
                                for k in range(5)])
            per.append([np.percentile(np.sqrt(v), p) for p in (50, 75, 90)])
        out[e] = tuple(round(float(x), 1) for x in np.mean(per, 0))
    return out


def d1_filter_medians():
    out = {}
    for e in ("kfx", "kf0"):
        med = []
        for r in PC.REPEATS:
            est, y = [], []
            for k in range(5):
                z = np.load(PC.P2 / "b56" / f"r{r}_f{k}.npz")
                est.append(z[f"est_{e}"].astype(np.float64))
                y.append(z["y"].astype(np.float64))
            med.append(np.median(np.abs(np.concatenate(est) - np.concatenate(y))))
        out[e] = round(float(np.mean(med)), 1)
    return out


def run_b60():
    import run_b54 as B54
    import run_b60 as B60
    PC.use_d1()
    PC.b54_d1(B54)
    B60.REPEATS = PC.REPEATS
    B60.DEV, B60.CONFIRM = (2,), PC.REPEATS
    B60.B56 = PC.P2 / "b56"
    B60.OUT_JSON, B60.OUT_MD = PC.paper() / "b60_relative_error.json", PC.paper() / "b60_relative_error.md"
    B60.SEC2 = d1_percentiles()
    B60.A1_MED = d1_filter_medians()
    print("D1 references for the checks:", B60.SEC2, B60.A1_MED, flush=True)
    B60.main()
    B60.addendum1()


def run_b61():
    import run_b54 as B54
    import run_b61 as B61
    PC.use_d1()
    PC.b54_d1(B54)
    B61.ALL = B61.CONFIRM = PC.REPEATS
    B61.DEV = (2,)
    B61.B56 = PC.P2 / "b56"
    B61.OUT_JSON, B61.OUT_MD = PC.paper() / "b61_absolute_robust.json", PC.paper() / "b61_absolute_robust.md"
    B61.main()


def run_b66():
    import run_b54 as B54
    import run_b66 as B66
    PC.use_d1()
    PC.b54_d1(B54)
    c = PC.P2 / "b66"
    c.mkdir(parents=True, exist_ok=True)
    if not (c / "detect.npz").exists():
        shutil.copy2(PC.DATA / ".b66" / "detect.npz", c / "detect.npz")       # the detector on every file: row-independent
    B66.CACHE, B66.B54C, B66.B56C = c, PC.P2 / "b54", PC.P2 / "b56"
    B66.REPEATS = B66.KFX_REPEATS = PC.REPEATS
    B66.PAPER = PC.paper()
    B66.OUT_JSON, B66.OUT_MD = B66.PAPER / "b66_regimes.json", B66.PAPER / "b66_regimes.md"
    B66.stage_counts()
    B66.stage_readout()


if __name__ == "__main__":
    {"b60": run_b60, "b61": run_b61, "b66": run_b66}[sys.argv[1]]()
