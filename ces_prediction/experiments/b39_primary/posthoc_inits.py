"""B.39 arm A, read after the fact: each initialisation by itself.

The registered readout of S1 and S2 (analyze_b39.py) draws one of the three initialisations and
resamples the test discharges.  This script shows where that interval comes from: per
initialisation, the skill of the causal variant over the causal GP, over the twin of the same
initialisation, and of that twin over the causal GP, each with a discharge-level 95 % interval
(2,000 resamples); the skill of the mean squared error over the three initialisations; and, for
the initialisation with the largest error on a split, how much of its excess squared error over
the causal GP three discharges hold.  Nothing here changes a registered verdict.

Run from the repo root:  py ces_prediction/experiments/b39_primary/posthoc_inits.py
Output: docs/paper/b39_inits.md
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analyze_b39 as A  # noqa: E402
import run_b39 as B  # noqa: E402

OUT = HERE.parents[2] / "docs" / "paper" / "b39_inits.md"
N_BOOT, SEED = 2000, 20261118


def s3(p, lo, hi):
    return f"{p:+.3f} [{lo:+.3f}, {hi:+.3f}]"


def main():
    rng = np.random.default_rng(SEED)
    L = ["# B.39 arm A, each initialisation by itself (after the fact)", "",
         "T_i, cut population, the four file-level splits, test rows of `eval_seq`. Causal variant and "
         "history-only twin of B.39 (epoch cap 80), three initialisations per split. Skill = 1 - sum of squared "
         "errors of A over that of R on the same rows. Intervals: 95 %, 2,000 resamples of the test discharges "
         "(one initialisation at a time; the registered readout of S1 and S2 also draws the initialisation). "
         "Nothing here changes a registered verdict.", "",
         "| split | initialisation | causal variant over causal GP | over the twin of the same initialisation | "
         "that twin over causal GP |", "|---|---|---|---|---|"]
    M, W = [], []
    n_runs = n_pos_gp = n_pos_tw = n_excl_gp = n_excl_tw = 0
    for s in B.SEEDS:
        C, T, G = [], [], None
        for i in B.inits(s):
            c, t = A.load(B.split_run_dir("v2", s, i)), A.load(B.twin_dir(s, i))
            A.same_rows(c, t, f"s{s} i{i}")
            cs, ts, gs = (A.sums(c["shot"], c["model"]), A.sums(t["shot"], t["model"]),
                          A.sums(c["shot"], c["gp_causal"]))
            C.append(cs); T.append(ts); G = gs
            idx = rng.integers(0, len(cs), size=(N_BOOT, len(cs)))
            g = 1 - cs[idx].sum(1) / gs[idx].sum(1)
            w = 1 - cs[idx].sum(1) / ts[idx].sum(1)
            tg = 1 - ts[idx].sum(1) / gs[idx].sum(1)
            pg, pw = 1 - cs.sum() / gs.sum(), 1 - cs.sum() / ts.sum()
            n_runs += 1
            n_pos_gp += pg > 0; n_pos_tw += pw > 0
            n_excl_gp += np.quantile(g, .025) > 0; n_excl_tw += np.quantile(w, .025) > 0
            L.append(f"| {s} | {i} | {s3(pg, *np.quantile(g, [.025, .975]))} | {s3(pw, *np.quantile(w, [.025, .975]))} | "
                     f"{s3(1 - ts.sum() / gs.sum(), *np.quantile(tg, [.025, .975]))} |")
        C, T = np.array(C), np.array(T)
        cm, tm = C.mean(0), T.mean(0)
        idx = rng.integers(0, len(cm), size=(N_BOOT, len(cm)))
        g = 1 - cm[idx].sum(1) / G[idx].sum(1)
        w = 1 - cm[idx].sum(1) / tm[idx].sum(1)
        M.append(f"| {s} | {s3(1 - cm.sum() / G.sum(), *np.quantile(g, [.025, .975]))} | "
                 f"{s3(1 - cm.sum() / tm.sum(), *np.quantile(w, [.025, .975]))} |")
        k = int(np.argmax(C.sum(1)))
        d = C[k] - G
        top = np.sort(d)[-3:].sum()
        W.append(f"| {s} | {B.inits(s)[k]} | {top / d[d > 0].sum():.2f} | {int((d > 0).sum())} of {len(d)} |")
    L += ["", f"Of the {n_runs} trainings the causal variant has a lower squared error than the causal GP in {n_pos_gp} "
              f"and than the twin of the same initialisation in {n_pos_tw}; the discharge-level interval is above zero "
              f"in {n_excl_gp} and in {n_excl_tw}.", "",
          "## Mean squared error over the three initialisations, discharges resampled", "",
          "| split | causal variant over causal GP | over the twin (mean over its three initialisations) |",
          "|---|---|---|"] + M
    L += ["", "## The initialisation with the largest error on each split", "",
          "| split | initialisation | share of its positive excess squared error over the causal GP held by three "
          "discharges | discharges with a positive excess |", "|---|---|---|---|"] + W
    OUT.write_text("\n".join(L) + "\n", encoding="utf-8", newline="\n")
    print("written", OUT.name)


if __name__ == "__main__":
    main()
