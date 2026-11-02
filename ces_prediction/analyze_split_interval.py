"""Across-split interval for the headline skills (THESIS_RESULTS.md sec. 8bv).

The registered headline rule counts splits whose shot-cluster bootstrap CI excludes 0
("4/4"). This script adds the complementary readout: treat the four frozen splits as
replicates and put one 95 % t-interval on the mean skill, so split-to-split variation
is inside the error bar. It reads only docs/paper/paper_numbers.json (no data, no model).

Also reported: the smallest number of splits n for which the half-width
t_{0.975,n-1} * sd / sqrt(n) would fall below |mean|, holding mean and sd fixed.

Run from the repo root:  python ces_prediction/analyze_split_interval.py
"""

import json
import statistics
from pathlib import Path

from scipy import stats

REPO = Path(__file__).resolve().parents[1]
NUMBERS = REPO / "docs" / "paper" / "paper_numbers.json"
OUT = REPO / "docs" / "paper" / "split_interval.json"
SEEDS = ("42", "1", "7", "123")
REFS = ("pchip", "persistence", "gp_causal", "gp")


def interval(values):
    n = len(values)
    mean = statistics.mean(values)
    sd = statistics.stdev(values)
    half = stats.t.ppf(0.975, n - 1) * sd / n**0.5
    n_needed = next(
        (k for k in range(2, 1001) if stats.t.ppf(0.975, k - 1) * sd / k**0.5 < abs(mean)),
        None,
    )
    return mean, sd, float(mean - half), float(mean + half), n_needed


def main():
    headline = json.loads(NUMBERS.read_text(encoding="utf-8"))["headline"]
    out = {}
    for pop in ("cut", "incl"):
        per_seed = headline[pop]["seq"]["per_seed"]
        for target in ("CES_TI", "CES_VT"):
            for ref in REFS:
                cells = [per_seed[s][target][ref] for s in SEEDS]
                values = [c["skill"] for c in cells]
                mean, sd, lo, hi, n_needed = interval(values)
                key = f"{pop}/{target}/{ref}"
                out[key] = {
                    "per_split": dict(zip(SEEDS, values)),
                    "per_split_pass": sum(bool(c["pass"]) for c in cells),
                    "mean": mean,
                    "sd": sd,
                    "ci95_t": [lo, hi],
                    "excludes_zero": bool(lo > 0 or hi < 0),
                    "n_splits_needed": n_needed,
                }
                print(
                    f"{key:28s} mean {mean:+.3f} sd {sd:.3f} CI [{lo:+.3f}, {hi:+.3f}] "
                    f"{'OUT ' if lo > 0 or hi < 0 else 'span'} per-split {out[key]['per_split_pass']}/4 "
                    f"n_needed {n_needed}"
                )
    OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"wrote {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
