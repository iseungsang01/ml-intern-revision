"""Re-score the reach ladder by pooling the four splits, instead of counting them.

The ladder has been read as "how many of the 4 splits had a CI clearing zero". That
statistic throws away nearly everything: each split produces a continuous paired estimate
with an interval, and reducing it to a 0/1 and summing four of them leaves a five-level
integer. §8al measured the consequence — the count reads 4/4 at 40 ms, **3/4 at 50**, 4/4
at 60 and 70, **3/4 at 100**, while the point estimates underneath rise smoothly. The
flicker is in the indicator, not in the effect.

So this stops counting and pools. Two things make that legitimate here:

* **The clusters barely overlap.** Each split tests 96 discharges out of a 301-discharge
  union; 224 of those appear in exactly one split and pairwise overlap is 10–20%.
* **Reuse is handled by construction.** Clustering is on the *physical discharge*, so a
  discharge that appears in two splits is one cluster carrying both its rows rather than
  two independent draws — the same rule `largegap/analyze_largegap.py` already uses, and
  conservative in the right direction.

The result is one estimate and one interval per rung over **301 clusters** instead of four
verdicts over 96, plus the thing the count could never give: a **trend test**. The question
was never "is rung 5 significant" but "does skill rise with context", so the slope of skill
against log context is estimated directly, with the same discharge clustering, by refitting
it inside every bootstrap resample.

Usage (repo root):
  py ces_prediction/experiments/b9_reach/pooled_ladder.py
  py ces_prediction/experiments/b9_reach/pooled_ladder.py --family tcn
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
CES_DIR = REPO_ROOT / "ces_prediction"
DATA = REPO_ROOT / "data"
sys.path.insert(0, str(CES_DIR))
from bootstrap_compare import B, BOOTSTRAP_SEED  # noqa: E402

SEEDS = (42, 1, 7, 123)
TARGETS = ("CES_TI", "CES_VT")
STEP_MS = 10.0
FAMILIES = {
    "lstm": ("v2r{r}", [2, 3, 4, 5, 6, 7, 10, 15, 31, 63]),
    "tcn": ("tcn{r}", [3, 5, 7, 15, 63]),
    "xfmr": ("xfmr{r}", [5, 7, 15, 63]),
    # 2/3/5 added 2026-08-21 (승상님): the "does not convert context" verdict was a 3-rung
    # trend over 70-630 ms; the 20-50 ms region where the other families turn was never
    # bought. These rungs give the trend test leverage where the action is.
    "ssm": ("ssmr{r}", [2, 3, 5, 7, 15, 63]),
}


def load(arm, seed, target):
    """-> (shot, se_model, se_gp_causal) for one run, or None if it was not trained."""
    path = DATA / f".b9_{arm}_s{seed}" / "comparison_errors_test.npz"
    if not path.exists():
        return None
    d = np.load(path, allow_pickle=True)
    shot, sm, sb = (d[f"{target}_shot"], d[f"{target}_se_model"],
                    d[f"{target}_se_gp_causal"])
    ok = np.isfinite(sm) & np.isfinite(sb)
    # Discharges are pooled across splits, so the cluster key must be the PHYSICAL shot
    # and must not be made split-unique -- that is the whole point of the pooling.
    return shot[ok], sm[ok], sb[ok]


def pool(arm, target):
    parts = [load(arm, s, target) for s in SEEDS]
    if any(p is None for p in parts):
        return None
    return tuple(np.concatenate([p[i] for p in parts]) for i in range(3))


def shot_sums(shot, values, keys):
    """Per-discharge sums aligned to `keys`, so every rung shares one cluster index."""
    idx = np.searchsorted(keys, shot)
    out = np.zeros(len(keys))
    np.add.at(out, idx, values)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", nargs="*", default=list(FAMILIES))
    args = ap.parse_args()
    fams = [f for f in args.family if f in FAMILIES] or list(FAMILIES)
    rng = np.random.default_rng(BOOTSTRAP_SEED)

    summary = {"cluster": "physical discharge, pooled over the 4 splits",
               "bootstrap": {"B": B, "seed": BOOTSTRAP_SEED},
               "baseline": "gp_causal", "families": {}}

    for fam in fams:
        pattern, reaches = FAMILIES[fam]
        print("\n" + "=" * 92)
        print(f"{fam}: skill vs causal GP, pooled over 4 splits, clustered on the discharge")
        print("context".rjust(9) + "shots".rjust(7)
              + "  CES_TI skill [95% CI]".ljust(30) + "win  -top10 "
              + " CES_VT skill [95% CI]".ljust(29) + "win  -top10")
        node = {}
        for target in TARGETS:
            # One cluster list per family+target so the trend refit resamples DISCHARGES,
            # not rungs: the same resampled discharge set scores every rung.
            pooled = {r: pool(pattern.format(r=r), target) for r in reaches}
            pooled = {r: v for r, v in pooled.items() if v is not None}
            if not pooled:
                continue
            keys = np.unique(np.concatenate([v[0] for v in pooled.values()]))
            sm = {r: shot_sums(v[0], v[1], keys) for r, v in pooled.items()}
            sb = {r: shot_sums(v[0], v[2], keys) for r, v in pooled.items()}

            draws = rng.integers(0, len(keys), size=(B, len(keys)))
            rows = {}
            boot_skill = {}
            for r in pooled:
                point = 1.0 - sm[r].sum() / sb[r].sum()
                bs = 1.0 - sm[r][draws].sum(axis=1) / sb[r][draws].sum(axis=1)
                bs = bs[np.isfinite(bs)]
                lo, hi = np.percentile(bs, [2.5, 97.5])
                boot_skill[r] = bs
                # Generality, printed next to the interval on purpose. A cluster bootstrap
                # asks "would re-drawing these discharges give the same answer" -- with ~200
                # clusters the answer is yes even when 5% of them carry the whole effect,
                # because a resample almost always contains some of them. These two columns
                # ask the different question the CI cannot: is the win typical?
                per = np.where(sb[r] > 0, 1.0 - sm[r] / np.where(sb[r] > 0, sb[r], 1.0), np.nan)
                seen = sb[r] > 0
                gain = sb[r] - sm[r]
                order = np.argsort(-gain)

                def without(k):
                    keep = np.ones(len(keys), bool)
                    keep[order[:k]] = False
                    return float(1.0 - sm[r][keep].sum() / sb[r][keep].sum())

                rows[r] = {"ms": r * STEP_MS, "n_shots": int(seen.sum()),
                           "n_rows": int(len(pooled[r][0])), "skill": float(point),
                           "ci95": [float(lo), float(hi)], "clears_zero": bool(lo > 0),
                           "win_rate": float(np.nanmean(per[seen] > 0)),
                           "drop_top5": without(5), "drop_top10": without(10),
                           "general": bool(np.nanmean(per[seen] > 0) > 0.6
                                           and without(10) > 0)}

            # Trend: slope of skill on log10(context), refit inside every resample so the
            # interval carries the same discharge clustering as the rungs themselves.
            rs = sorted(boot_skill)
            n_boot = min(len(boot_skill[r]) for r in rs)
            x = np.log10(np.array([r * STEP_MS for r in rs]))
            xc = x - x.mean()
            y = np.stack([boot_skill[r][:n_boot] for r in rs], axis=1)   # (B, n_rungs)
            slopes = (y - y.mean(axis=1, keepdims=True)) @ xc / (xc @ xc)
            point_y = np.array([rows[r]["skill"] for r in rs])
            slope_pt = float(((point_y - point_y.mean()) @ xc) / (xc @ xc))
            s_lo, s_hi = np.percentile(slopes, [2.5, 97.5])
            # Three outcomes, not two: a CI entirely below zero is a real DECLINE, and
            # calling that "not resolved" would hide it.
            verdict = ("rises" if s_lo > 0 else
                       "declines" if s_hi < 0 else "not resolved")
            node[target] = {"rungs": rows,
                            "trend_slope_per_decade": slope_pt,
                            "trend_ci95": [float(s_lo), float(s_hi)],
                            "trend": verdict}

        for r in sorted({r for t in node for r in node[t]["rungs"]}):
            ti = node.get("CES_TI", {}).get("rungs", {}).get(r)
            vt = node.get("CES_VT", {}).get("rungs", {}).get(r)
            if not ti:
                continue
            def cell(c):
                mark = "*" if c["clears_zero"] else " "
                gen = "" if c["general"] else "!"
                return (f"{c['skill']:+.4f} [{c['ci95'][0]:+.3f}, {c['ci95'][1]:+.3f}]{mark}"
                        .ljust(30) + f"{c['win_rate']:.2f}{gen} {c['drop_top10']:+.3f} ")
            print(f"{ti['ms']:>6.0f} ms{ti['n_shots']:>7d}  " + cell(ti)
                  + (cell(vt) if vt else ""))
        for target in node:
            t = node[target]
            mark = t["trend"].upper()
            print(f"  trend {target}: {t['trend_slope_per_decade']:+.4f} skill per decade of "
                  f"context [{t['trend_ci95'][0]:+.4f}, {t['trend_ci95'][1]:+.4f}] -> {mark}")
        summary["families"][fam] = node
        for line in ("(* = pooled 95% CI clears zero.  win = fraction of discharges the model wins.",
                     "  -top10 = pooled skill with the 10 best-contributing discharges removed.",
                     "  ! = NOT shot-general: win rate <= 0.60 or -top10 <= 0. A starred ! means",
                     "      the interval is real and the effect still is not typical -- report both."):
            print("  " + line)

    out = DATA / ".b9_pooled_ladder.json"
    out.write_text(json.dumps(summary, indent=1))
    print(f"\n[pool] wrote {out}")


if __name__ == "__main__":
    main()
