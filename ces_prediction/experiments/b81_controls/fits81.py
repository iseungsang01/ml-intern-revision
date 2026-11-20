"""B.81 fit stage (registration section 3, Addendum 1 section 2): TXp and TXo for every unit, TE and TB for V2 and V3;
rules D650 and S; target L5.  The log carries unit names, fit counts and times only.

    py ces_prediction/experiments/b81_controls/fits81.py --part K/N [--units NAME,...]
Output: data/.b81/fits/{unit}__{learner}__{rule}__L5.npz  (idx, est, fit_rows, n_iter; TXo: nonfinite_fit_rows,
nonfinite_test_rows)
"""
from __future__ import annotations

import argparse
import time
import warnings

import common81 as B

import numpy as np  # noqa: E402

warnings.filterwarnings("ignore")


def run_unit(ctx, u):
    todo = [p for p in B.plan(u) if not B.path81(u["name"], *p).exists()]
    if not todo:
        return 0
    t0 = time.time()
    wf, ms_b, ms_e = ctx.unit_arrays(u)
    tgt = ctx.R[f"tg_{B.X}"]
    nfit = 0
    for r in B.RULES:
        tl = [L for L, rr in todo if rr == r]
        if not tl:
            continue
        H10, f8, A, I = ctx.matrices(r, wf, ms_b, ms_e)
        Gr = I["G"]
        fr, te = ctx.rows_of(u, r)
        for L in tl:
            Xf, Xt = B.design_matrices(L, u, r, H10, f8, A, I, wf, fr, te)
            extra = {}
            if L == "TXo":
                extra["nonfinite_fit_rows"] = np.int64((~np.isfinite(Xf).all(1)).sum())
                extra["nonfinite_test_rows"] = np.int64((~np.isfinite(Xt).all(1)).sum())
            est, m = B.fit_estimate(Xf, Xt, tgt, Gr, fr, te)
            nfit += 1
            B.save_npz(B.path81(u["name"], L, r), idx=te, est=est, fit_rows=np.int64(len(fr)), n_iter=np.int64(m.n_iter_),
                       **extra)
        del H10, f8, A
    B.log(f"{u['name']}: {nfit} fits ({time.time() - t0:.0f}s)")
    return nfit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="0/1")
    ap.add_argument("--units", default="")
    a = ap.parse_args()
    k, n = (int(x) for x in a.part.split("/"))
    us, _, _ = B.units()
    if a.units:
        names = set(a.units.split(","))
        us = [u for u in us if u["name"] in names]
    us = sorted(us, key=lambda u: (-len(B.plan(u)), u["name"]))
    mine = us[k::n]
    ctx = B.Ctx()
    B.log(f"part {k}/{n}: {len(mine)} units")
    tot = 0
    for u in mine:
        tot += run_unit(ctx, u)
    B.log(f"part {k}/{n}: done, {tot} fits")


if __name__ == "__main__":
    main()
