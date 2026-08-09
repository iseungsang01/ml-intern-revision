"""CES fit-failure sensitivity: does the CES_TI headline survive dropping the
spectral-fit-failure rows (CES_TI above ~3 keV; global p99 = 2,089 eV,
max 14,984 eV -- THESIS_RESULTS §8 'CES fit-failure artifacts')?

Method (evaluation-only; no retraining, no re-scoring): reads the
`{target}_y_true` key added to the genuine npz by
experiments/gp/rerun_compare_gp.py, drops CES_TI samples whose TRUE value
exceeds the threshold from ALL arms simultaneously (paired -- the same rows are
removed from model and baselines alike), and recomputes skill_vs_pchip with the
PR4 shot-clustered bootstrap (B=10,000, seed 12345). Thresholds: 3,000 eV
(primary, the documented fit-failure boundary) and 2,089 eV (global p99,
stricter secondary).

CES_VT has no analogous artifact and is untouched.

Output: data/.fitfail_analysis.json
Usage (repo root):  py ces_prediction/experiments/fitfail/analyze_fitfail.py
"""

import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
CES_DIR = REPO_ROOT / "ces_prediction"
sys.path.insert(0, str(CES_DIR / "experiments"))
from runner_common import BASELINE_OUT  # noqa: E402

SEEDS = (42, 1, 7, 123)
NPZ_NAME = "comparison_errors_test__test_genuine_gp.npz"
THRESHOLDS_EV = (3000.0, 2089.0)
B = 10000
BOOTSTRAP_SEED = 12345


def _per_shot_sums(shot, values):
    shots, inv = np.unique(shot, return_inverse=True)
    sums = np.zeros(len(shots), dtype=np.float64)
    np.add.at(sums, inv, values)
    return shots, sums


def _bootstrap(shot, se_a, se_b, rng):
    _, sa = _per_shot_sums(shot, se_a)
    _, sb = _per_shot_sums(shot, se_b)
    n = len(sa)
    skill_point = (1.0 - sa.sum() / sb.sum()) if sb.sum() > 0 else float("nan")
    idx = rng.integers(0, n, size=(B, n))
    skill = 1.0 - sa[idx].sum(axis=1) / sb[idx].sum(axis=1)
    finite = np.isfinite(skill)
    lo, hi = (np.percentile(skill[finite], [2.5, 97.5]) if finite.any()
              else (float("nan"), float("nan")))
    return {"n_shots": int(n), "skill_point": float(skill_point),
            "skill_ci95": [float(lo), float(hi)], "pass": bool(lo > 0.0)}


def main():
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    out = {"npz": NPZ_NAME, "target": "CES_TI", "thresholds_ev": list(THRESHOLDS_EV),
           "bootstrap_resamples": B, "seed": BOOTSTRAP_SEED, "per_seed": {}}
    for seed in SEEDS:
        path = BASELINE_OUT[seed] / NPZ_NAME
        if not path.exists():
            raise SystemExit(f"FATAL: missing {path} -- run rerun_compare_gp.py first")
        d = np.load(path)
        shot = d["CES_TI_shot"]
        se_m = d["CES_TI_se_model"]
        se_p = d["CES_TI_se_pchip"]
        y = d["CES_TI_y_true"]
        rec = {
            "n_total": int(len(y)),
            "y_true_max_ev": float(y.max()),
            "full": _bootstrap(shot, se_m, se_p, rng),
        }
        for thr in THRESHOLDS_EV:
            keep = y <= thr
            dropped = int((~keep).sum())
            r = _bootstrap(shot[keep], se_m[keep], se_p[keep], rng)
            r["n_dropped"] = dropped
            r["dropped_pct"] = round(100.0 * dropped / len(y), 3)
            r["n_shots_with_drops"] = int(len(np.unique(shot[~keep])))
            rec[f"le_{int(thr)}ev"] = r
        out["per_seed"][str(seed)] = rec

    print(f"\n=== CES_TI fit-failure sensitivity (genuine, vs PCHIP, B={B}) ===")
    print(f"  {'seed':>5} {'full skill':>24}  {'<=3000 eV':>28}  {'<=2089 eV':>28}")
    for seed in SEEDS:
        r = out["per_seed"][str(seed)]
        def fmt(x, extra=""):
            return (f"{x['skill_point']:+.3f} [{x['skill_ci95'][0]:+.3f},"
                    f"{x['skill_ci95'][1]:+.3f}]{' PASS' if x['pass'] else ' n.s.'}{extra}")
        a, b3, b2 = r["full"], r["le_3000ev"], r["le_2089ev"]
        print(f"  {seed:>5} {fmt(a):>24}  {fmt(b3, f' (-{b3['n_dropped']})'):>28}"
              f"  {fmt(b2, f' (-{b2['n_dropped']})'):>28}")

    out_path = REPO_ROOT / "data" / ".fitfail_analysis.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nsaved -> {out_path}")


if __name__ == "__main__":
    main()
