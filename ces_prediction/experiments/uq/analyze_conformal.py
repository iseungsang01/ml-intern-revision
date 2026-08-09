"""Calibrated predictive intervals for the CES nowcaster, via split conformal prediction.

A virtual sensor that emits a number and no error bar cannot be used: a physicist
reading a filled CES point has no way to know whether to trust it, and the fusion
community's default tool for exactly this reason is Gaussian-process regression,
which returns a posterior. Our model returns a point estimate. This closes that gap
without touching the frozen checkpoints.

Why conformal rather than a learned variance head. A heteroscedastic or quantile head
would require retraining, which would move the point predictions and therefore every
skill number in the paper -- the uncertainty work would confound the headline. Split
conformal instead calibrates on a held-out split and gives **distribution-free,
finite-sample marginal coverage** with no assumption about the error distribution and
no change to the predictor. The price is that the interval comes from observable
context rather than from the network's own belief, which we say plainly.

Two variants, because marginal coverage is not enough here:
  global    one quantile for everything. Guaranteed marginally, but §8g shows the
            error scale varies by an order of magnitude with dt, so a single width
            is far too wide at small gaps and too narrow at large ones.
  mondrian  a separate quantile per (dt bin x local-activity) stratum -- the same
            covariates that carry the missingness mechanism in §8i. This buys
            approximate CONDITIONAL coverage, which is what a user actually needs.

Reported per target: PICP (empirical coverage), mean interval half-width, and the
Winkler interval score (width plus a miss penalty; lower is better), for the model
and for the two baselines calibrated the identical way -- so the comparison is
interval quality, not calibration technique.

Assumption, stated: conformal needs calibration and test to be exchangeable. They are
disjoint SHOTS, so shot-level heterogeneity can violate it; we therefore also report
the spread of per-shot coverage, not just the pooled number.

Usage (repo root):  py ces_prediction/experiments/uq/analyze_conformal.py
Writes: data/.uq_conformal.json
"""

import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
SEEDS = (42, 1, 7, 123)
RUN_DIR = {
    42: REPO_ROOT / "data" / ".vt_repro_out",
    1: REPO_ROOT / "data" / ".vt_repro_ms_1",
    7: REPO_ROOT / "data" / ".vt_repro_ms_7",
    123: REPO_ROOT / "data" / ".vt_repro_ms_123",
}
CAL_NPZ = "comparison_errors_val__val_genuine.npz"
TEST_NPZ = "comparison_errors_test__test_genuine_persist.npz"
TARGETS = ("CES_TI", "CES_VT")
ARMS = ("model", "pchip", "persistence")
ALPHA = 0.10                      # 90% intervals
DT_EDGES = (15.0, 25.0, 45.0)
MIN_CAL_PER_STRATUM = 50          # below this, fall back to the global quantile


def stratum(dt_ms, act):
    return np.digitize(dt_ms, DT_EDGES) * 2 + act.astype(int)


N_STRATA = (len(DT_EDGES) + 1) * 2


def conformal_q(scores, alpha=ALPHA):
    """Split-conformal quantile with the finite-sample correction."""
    n = len(scores)
    if n == 0:
        return np.inf
    k = int(np.ceil((n + 1) * (1 - alpha)))
    if k > n:
        return float(np.max(scores))
    return float(np.sort(scores)[k - 1])


def winkler(residual, half_width, alpha=ALPHA):
    """Interval score for a symmetric interval: width + miss penalty. Lower better."""
    width = 2.0 * half_width
    miss = np.maximum(residual - half_width, 0.0)
    return width + (2.0 / alpha) * 2.0 * miss


def load(seed, name, target):
    path = RUN_DIR[seed] / name
    if not path.exists():
        raise SystemExit(
            f"FATAL: missing {path}.\n"
            "Run: py ces_prediction/experiments/uq/rerun_compare_val.py  (calibration set)\n"
            "and  py ces_prediction/experiments/largegap/rerun_compare_persistence.py (test set)"
        )
    z = np.load(path)
    out = {"shot": z[f"{target}_shot"],
           "dt": z[f"{target}_dt_ms"],
           "act": z[f"{target}_is_peak"].astype(bool)}
    for arm in ARMS:
        key = f"{target}_se_{arm}"
        out[arm] = np.sqrt(z[key]) if key in z.files else None
    return out


def evaluate(cal, test, arm, mode):
    r_cal, r_test = cal[arm], test[arm]
    if r_cal is None or r_test is None:
        return None
    if mode == "global":
        hw = np.full(len(r_test), conformal_q(r_cal))
    else:
        s_cal, s_test = stratum(cal["dt"], cal["act"]), stratum(test["dt"], test["act"])
        q_global = conformal_q(r_cal)
        q = np.full(N_STRATA, q_global)
        for s in range(N_STRATA):
            sel = s_cal == s
            # A stratum with too few calibration points cannot support its own
            # quantile; falling back to the global one keeps marginal validity
            # instead of inventing a confident-looking narrow interval.
            if sel.sum() >= MIN_CAL_PER_STRATUM:
                q[s] = conformal_q(r_cal[sel])
        hw = q[s_test]
    covered = r_test <= hw
    shots = test["shot"]
    uniq = np.unique(shots)
    per_shot = np.array([covered[shots == s].mean() for s in uniq])
    return {
        "n_cal": int(len(r_cal)), "n_test": int(len(r_test)),
        "coverage": float(covered.mean()),
        "target_coverage": 1 - ALPHA,
        "half_width_mean": float(np.mean(hw)),
        "half_width_median": float(np.median(hw)),
        "winkler_mean": float(np.mean(winkler(r_test, hw))),
        "per_shot_coverage_p10": float(np.percentile(per_shot, 10)),
        "per_shot_coverage_p90": float(np.percentile(per_shot, 90)),
        "n_shots": int(len(uniq)),
    }


def main():
    results = {"_design": {
        "method": "split conformal on |residual|, calibrated on the validation split",
        "alpha": ALPHA, "dt_edges_ms": list(DT_EDGES),
        "min_cal_per_stratum": MIN_CAL_PER_STRATUM,
        "assumption": "exchangeability between calibration and test shots",
        "note": ("intervals are model-agnostic: identical procedure applied to the "
                 "baselines, so differences are interval quality not calibration"),
    }, "per_seed": {}}

    for seed in SEEDS:
        entry = {}
        for target in TARGETS:
            cal, test = load(seed, CAL_NPZ, target), load(seed, TEST_NPZ, target)
            tgt = {}
            for mode in ("global", "mondrian"):
                tgt[mode] = {arm: evaluate(cal, test, arm, mode) for arm in ARMS}
            # Scale-free readout: how wide is the interval relative to the spread of
            # the quantity itself? This is what makes TI and VT comparable at all.
            spread = float(np.std(test["persistence"])) if test["persistence"] is not None else None
            tgt["persistence_residual_std"] = spread
            entry[target] = tgt
        results["per_seed"][seed] = entry
        print(f"[uq] seed {seed} done")

    out = REPO_ROOT / "data" / ".uq_conformal.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nwrote {out}\n")

    for target in TARGETS:
        print(f"=== {target}: 90% conformal intervals (calibrated on val, evaluated on test)")
        print(f"{'seed':>5} {'mode':>9} {'arm':>12} {'coverage':>9} {'per-shot p10..p90':>19} "
              f"{'half-width':>11} {'Winkler':>11}")
        for seed in SEEDS:
            for mode in ("global", "mondrian"):
                for arm in ARMS:
                    e = results["per_seed"][seed][target][mode][arm]
                    if e is None:
                        continue
                    print(f"{seed:>5} {mode:>9} {arm:>12} {e['coverage']*100:>8.1f}% "
                          f"{e['per_shot_coverage_p10']*100:>8.0f}..{e['per_shot_coverage_p90']*100:<8.0f}"
                          f"{e['half_width_mean']:>11.2f} {e['winkler_mean']:>11.1f}")
        print()


if __name__ == "__main__":
    main()
