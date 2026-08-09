"""Does the window-sweep conclusion survive when the history is NOT the row right
before the target?

The sweep's headline ("one previous observation is enough; more window does not
help") is dominated by samples whose nearest observed CES sits one 10 ms step back
-- ~98% of them. This script re-reads the SAME scored samples, stratifies them by
`dt` (time from the target back to its nearest observed CES) and reports skill per
stratum per window, so the conclusion can be checked in the regime where the
history is genuinely detached from the target (dt = 20, 30, 50+ ms).

Nothing is retrained: it pools the per-sample squared errors that
compare_baselines.py already wrote to comparison_errors_test.npz for every run.

Seeds are pooled per window (each seed has its own test shots, so shot ids are
namespaced by seed before bootstrapping) and CIs come from the project's standard
shot-clustered paired bootstrap.

Usage (repo root):
  py ces_prediction/experiments/window_sweep/analyze_gap_by_window.py
  py ces_prediction/experiments/window_sweep/analyze_gap_by_window.py --variant kept
Writes data/.wsweep_<variant>_gap_by_window.json
"""

import argparse
import json
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA = REPO_ROOT / "data"
SEEDS = (42, 1, 7, 123)
WINDOWS = (2, 3, 4, 6, 8)
TARGETS = ("CES_TI", "CES_VT")
# dt is "how far back the nearest observed CES is". On a 10 ms grid the first bin is
# the adjacent-history regime; everything above it is a genuinely detached history.
BINS = [(0.0, 15.0), (15.0, 25.0), (25.0, 45.0), (45.0, np.inf)]
BIN_LABELS = ["10ms (직전)", "20ms", "30-40ms", "50ms+"]
B = 4000
BOOT_SEED = 20260804


def _load(prefix, window, seed):
    p = DATA / f".{prefix}_w{window}_s{seed}" / "comparison_errors_test.npz"
    return np.load(p) if p.exists() else None


def _skill(se_model, se_base):
    mb = se_base.mean()
    return float(1.0 - se_model.mean() / mb) if mb > 0 else float("nan")


def _boot_ci(shot, se_model, se_base, rng):
    """Shot-clustered paired bootstrap on the skill score."""
    shots = np.unique(shot)
    index = {s: np.flatnonzero(shot == s) for s in shots}
    out = np.empty(B)
    for b in range(B):
        pick = rng.choice(shots, size=len(shots), replace=True)
        idx = np.concatenate([index[s] for s in pick])
        out[b] = _skill(se_model[idx], se_base[idx])
    lo, hi = np.percentile(out[np.isfinite(out)], [2.5, 97.5])
    return float(lo), float(hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=("hf", "kept"), default="hf")
    args = ap.parse_args()
    prefix = "wsweep_hf" if args.variant == "hf" else "wsweep"
    rng = np.random.default_rng(BOOT_SEED)

    report = {"variant": args.variant, "bins_ms": BIN_LABELS, "bootstrap": B, "per_target": {}}
    for target in TARGETS:
        print(f"\n{'='*78}\n{target}   ({args.variant})")
        header = f"{'W':>3} " + " ".join(f"{lab:>22}" for lab in BIN_LABELS)
        print(header)
        print("-" * len(header))
        rows = {}
        for w in WINDOWS:
            pooled = {"shot": [], "dt": [], "m": [], "p": []}
            for seed in SEEDS:
                d = _load(prefix, w, seed)
                if d is None:
                    continue
                # namespace shot ids so pooled bootstrap never merges two seeds' shots
                pooled["shot"].append(d[f"{target}_shot"] + seed * 100_000)
                pooled["dt"].append(d[f"{target}_dt_ms"])
                pooled["m"].append(d[f"{target}_se_model"])
                pooled["p"].append(d[f"{target}_se_pchip"])
            if not pooled["shot"]:
                continue
            shot = np.concatenate(pooled["shot"])
            dt = np.concatenate(pooled["dt"])
            se_m = np.concatenate(pooled["m"])
            se_p = np.concatenate(pooled["p"])

            cells, cellstr = [], []
            for (lo, hi) in BINS:
                sel = (dt > lo) & (dt <= hi)
                n = int(sel.sum())
                if n < 30:  # too few to say anything; report the count only
                    cells.append({"n": n})
                    cellstr.append(f"n={n:<5} —".rjust(22))
                    continue
                sk = _skill(se_m[sel], se_p[sel])
                ci = _boot_ci(shot[sel], se_m[sel], se_p[sel], rng)
                cells.append({"n": n, "skill": sk, "ci95": ci, "pass": bool(ci[0] > 0)})
                mark = "*" if ci[0] > 0 else " "
                cellstr.append(f"{sk:+.3f}{mark} [{ci[0]:+.2f},{ci[1]:+.2f}] n={n}".rjust(22))
            rows[w] = cells
            print(f"{w:>3} " + " ".join(cellstr))
        report["per_target"][target] = rows
    print("\n* = shot-clustered 95% CI excludes 0")

    out = DATA / f".{prefix}_gap_by_window.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
