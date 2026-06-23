"""Shot-clustered paired bootstrap CI for model-vs-interpolation (PR4 headline gate).

Loads `comparison_errors_{split}.npz` (per-target shot id + squared errors for
model / pchip / linear, written by compare_baselines.py), resamples WHOLE SHOTS
with replacement (within-shot rows are autocorrelated, so the shot is the unit),
and reports the 95% CI of `skill_vs_<baseline> = 1 - MSE_model/MSE_baseline` and of
the paired mean `(SE_model - SE_baseline)`.

Headline gate (PR4): the model "beats" the baseline iff the 95% skill CI lower
bound is > 0 (equivalently the paired-diff CI upper bound < 0). Reported per target.

Env: CES_OUTPUT_DIR (where the npz live). Deterministic (fixed bootstrap seed).
"""

import json
import os
from pathlib import Path

import numpy as np

TARGET_NAMES = ("CES_TI", "CES_VT")
B = 10000
BOOTSTRAP_SEED = 12345


def _per_shot_sums(shot, values):
    shots, inv = np.unique(shot, return_inverse=True)
    sums = np.zeros(len(shots), dtype=np.float64)
    np.add.at(sums, inv, values)
    counts = np.zeros(len(shots), dtype=np.float64)
    np.add.at(counts, inv, 1.0)
    return shots, sums, counts


def _bootstrap(shot, se_model, se_base, rng):
    _, sm, cnt = _per_shot_sums(shot, se_model)
    _, sb, _ = _per_shot_sums(shot, se_base)
    n = len(sm)
    skill_point = (1.0 - sm.sum() / sb.sum()) if sb.sum() > 0 else float("nan")
    diff_point = (sm.sum() - sb.sum()) / cnt.sum()  # mean(SE_model - SE_base); <0 = model better

    idx = rng.integers(0, n, size=(B, n))
    tot_sm = sm[idx].sum(axis=1)
    tot_sb = sb[idx].sum(axis=1)
    tot_cnt = cnt[idx].sum(axis=1)
    # Guard the statistic-of-record: a resample drawing only zero-baseline-error
    # shots would give inf/nan (impossible on real continuous CES errors, but the
    # bootstrap CI must never silently propagate non-finite values into percentiles).
    skill = 1.0 - tot_sm / tot_sb
    diff = (tot_sm - tot_sb) / tot_cnt
    finite = np.isfinite(skill)
    if finite.any():
        skill_lo, skill_hi = np.percentile(skill[finite], [2.5, 97.5])
    else:
        skill_lo, skill_hi = float("nan"), float("nan")
    diff_lo, diff_hi = np.percentile(diff, [2.5, 97.5])
    return {
        "n_shots": int(n),
        "skill_point": float(skill_point),
        "skill_ci95": [float(skill_lo), float(skill_hi)],
        "paired_diff_point": float(diff_point),
        "paired_diff_ci95": [float(diff_lo), float(diff_hi)],
        "pass": bool(skill_lo > 0.0),
    }


def run():
    output_dir = Path(os.getenv("CES_OUTPUT_DIR", Path(__file__).resolve().parent))
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    summary = {"bootstrap_resamples": B, "seed": BOOTSTRAP_SEED, "splits": {}}

    for split in ("test", "val"):
        npz_path = output_dir / f"comparison_errors_{split}.npz"
        if not npz_path.exists():
            continue
        data = np.load(npz_path)
        summary["splits"][split] = {}
        print(f"\n=== {split}: shot-clustered paired bootstrap (B={B}) ===")
        for name in TARGET_NAMES:
            shot = data[f"{name}_shot"]
            se_model = data[f"{name}_se_model"]
            per_base = {}
            for base in ("pchip", "linear"):
                res = _bootstrap(shot, se_model, data[f"{name}_se_{base}"], rng)
                per_base[base] = res
                verdict = "PASS (beats)" if res["pass"] else "n.s."
                print(
                    f"  {name} vs {base:<6}: skill={res['skill_point']:+.4f} "
                    f"CI95=[{res['skill_ci95'][0]:+.4f}, {res['skill_ci95'][1]:+.4f}] "
                    f"{verdict}  (n_shots={res['n_shots']})"
                )
            # Pre-registered gap stratification: is the model significant in the
            # dominant small-gap (<=15ms) regime, vs the heavy-tailed aggregate?
            if f"{name}_dt_ms" in data.files:
                small = data[f"{name}_dt_ms"] <= 15.0
                if int(small.sum()) > 50:
                    r = _bootstrap(shot[small], se_model[small], data[f"{name}_se_pchip"][small], rng)
                    per_base["pchip_dt_le_15ms"] = r
                    v = "PASS (beats)" if r["pass"] else "n.s."
                    print(
                        f"  {name} vs pchip [dt<=15ms]: skill={r['skill_point']:+.4f} "
                        f"CI95=[{r['skill_ci95'][0]:+.4f}, {r['skill_ci95'][1]:+.4f}] {v}  "
                        f"(n={int(small.sum())}, n_shots={r['n_shots']})"
                    )
            summary["splits"][split][name] = per_base

    out_path = output_dir / "bootstrap_summary.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nGate (PR4): PASS iff 95% skill CI lower bound > 0 vs PCHIP.")
    print(f"Saved {out_path}")
    return summary


if __name__ == "__main__":
    run()
