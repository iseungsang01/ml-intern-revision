"""B.84 stage 2: the mixture learners (T0all, TXall, T0raw, TXraw) and TXrel per unit (Addendum 1 section 3).

    py ces_prediction/experiments/b84_onefit/fits84.py [--units NAME,...] [--learners T0all,...]
Output: data/.b84/fits/{unit}__{learner}__{rule}.npz (idx = rows of rows.npz, est) and {unit}__{learner}__fit.json
(stacked rows per rule, n_iter, time).  The log carries names, counts and times only.
"""
from __future__ import annotations

import argparse
import gc
import time
import warnings

import common84 as M

import numpy as np  # noqa: E402

warnings.filterwarnings("ignore")


def out_path(u, L, r):
    return M.FITS / f"{u['name']}__{L}__{r}.npz"


def info_path(u, L):
    return M.FITS / f"{u['name']}__{L}__fit.json"


def fit_mixture(ctx, ua, u, L):
    if info_path(u, L).exists():
        return 0
    t0 = time.time()
    Xs, ys, counts, src = M.build_stack(ctx, ua, u, L)
    test_fids = set(int(x) for x in u["test"])
    if any(int(f) in test_fids for f in np.unique(ctx.R["fidx"][src])):
        raise SystemExit(f"FATAL: {u['name']} {L}: a test discharge in the stack")
    m = M.gbr().fit(Xs, ys)
    n_stack = len(ys)
    del Xs, ys, src
    gc.collect()
    t_fit = time.time() - t0
    for r in M.ALL_RULES:
        te = ctx.rows_of(u, r, "test")
        rm = M.RuleMats(ua, r, need_raw=L.endswith("raw"))
        est = rm.G[te] + m.predict(rm.X(L, te))
        M.save_npz(out_path(u, L, r), idx=te, est=est)
        del rm
    M.save_json(info_path(u, L), {"unit": u["name"], "learner": L, "stacked_rows_per_rule": counts,
                                  "stacked_rows": int(n_stack), "n_iter": int(m.n_iter_),
                                  "fit_s": round(t_fit, 1), "total_s": round(time.time() - t0, 1)})
    M.log(f"{u['name']} {L}: stack {n_stack:,} rows, n_iter {m.n_iter_}, fit {t_fit:.0f}s, total {time.time() - t0:.0f}s")
    return 1


def fit_rel(ctx, ua, u):
    n = 0
    for r in M.REL_RULES:
        if out_path(u, "TXrel", r).exists():
            continue
        t0 = time.time()
        fr, te = ctx.rows_of(u, r, "fit"), ctx.rows_of(u, r, "test")
        rm = M.RuleMats(ua, r, need_raw=False)
        m = M.gbr().fit(rm.X("TXrel", fr), rm.target(fr))
        est = rm.G[te] + m.predict(rm.X("TXrel", te))
        M.save_npz(out_path(u, "TXrel", r), idx=te, est=est, fit_rows=np.int64(len(fr)), n_iter=np.int64(m.n_iter_))
        M.log(f"{u['name']} TXrel {r}: fit rows {len(fr):,} ({time.time() - t0:.0f}s)")
        n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--units", default="")
    ap.add_argument("--learners", default=",".join(M.MIX_LEARNERS + ("TXrel",)))
    a = ap.parse_args()
    ctx = M.Ctx()
    us, _, _ = M.units()
    if a.units:
        names = set(a.units.split(","))
        us = [u for u in us if u["name"] in names]
    learners = a.learners.split(",")
    tot = 0
    for u in us:
        ua = M.UnitArrays(ctx, u)
        for L in learners:
            if L == "TXrel":
                tot += fit_rel(ctx, ua, u)
            else:
                tot += fit_mixture(ctx, ua, u, L)
        del ua
        gc.collect()
    M.log(f"done, {tot} fits")


if __name__ == "__main__":
    main()
