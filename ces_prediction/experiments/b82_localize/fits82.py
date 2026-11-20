"""B.82 fit stage (registration sections 5-6, Addendum 1 sections 3-5): part C (TM, TXM, TXs, TBi, TBm, TBo for the units
of V1, V3 and V4a) and part D (T0 and TX with random_state 1 to 10 for V4a, V4b and V1_r2_k0..k4); rules D650 and S;
target L5.  The log carries unit names, fit counts and times only.

    py ces_prediction/experiments/b82_localize/fits82.py --part K/N [--units NAME,...]
Output: data/.b82/fits/{unit}__{learner}__{rule}__L5.npz (idx, est, fit_rows, n_iter; TXs also donor grid rows/files)
  seed refits are stored as learners T0_seed{s}, TX_seed{s}.
"""
from __future__ import annotations

import argparse
import time
import warnings

import common82 as M

import numpy as np  # noqa: E402

warnings.filterwarnings("ignore")


def plan(u):
    out = []
    if u["design"] in ("V1", "V3") or u["name"] == "V4a":
        out += [(L, r) for r in M.RULES for L in M.PART_C_LEARNERS]
    if u["name"] in M.SEED_UNITS:
        out += [(f"{L}_seed{s}", r) for r in M.RULES for L in ("T0", "TX") for s in M.SEEDS]
    return out


def run_unit(ctx, u):
    todo = [p for p in plan(u) if not M.path82(u["name"], *p).exists()]
    if not todo:
        return 0
    t0 = time.time()
    wf, ms_b, ms_e = ctx.unit_arrays(u)
    tgt = ctx.R[f"tg_{M.X}"]
    nfit = 0
    cache = {}
    for r in M.RULES:
        tl = [L for L, rr in todo if rr == r]
        if not tl:
            continue
        H10, f8, A, I = ctx.matrices(r, wf, ms_b, ms_e)
        Gr = I["G"]
        fr, te = ctx.rows_of(u, r)
        cache.clear()
        for L in tl:
            seed = None
            base = L
            if "_seed" in L:
                base, s = L.split("_seed")
                seed = int(s)
            Xf = M.matrix(ctx, base, u, r, H10, f8, A, I, wf, ms_b, ms_e, fr, cache)
            Xt = M.matrix(ctx, base, u, r, H10, f8, A, I, wf, ms_b, ms_e, te, cache)
            extra = {}
            if base == "TXs":
                _, dg_f, df_f = ctx.swap14(u, r, wf, ms_b, ms_e, fr)
                _, dg_t, df_t = ctx.swap14(u, r, wf, ms_b, ms_e, te)
                extra = {"donor_grid_fit": dg_f, "donor_file_fit": df_f, "donor_grid_test": dg_t, "donor_file_test": df_t}
            m = M.gbr(seed).fit(Xf, (tgt - Gr)[fr])
            est = Gr[te] + m.predict(Xt)
            nfit += 1
            M.save_npz(M.path82(u["name"], L, r), idx=te, est=est, fit_rows=np.int64(len(fr)), n_iter=np.int64(m.n_iter_),
                       nonfinite_fit_rows=np.int64((~np.isfinite(Xf).all(1)).sum()),
                       nonfinite_test_rows=np.int64((~np.isfinite(Xt).all(1)).sum()), **extra)
        del H10, f8, A
    M.log(f"{u['name']}: {nfit} fits ({time.time() - t0:.0f}s)")
    return nfit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="0/1")
    ap.add_argument("--units", default="")
    a = ap.parse_args()
    k, n = (int(x) for x in a.part.split("/"))
    us, _, _ = M.units()
    us = [u for u in us if plan(u)]
    if a.units:
        names = set(a.units.split(","))
        us = [u for u in us if u["name"] in names]
    us = sorted(us, key=lambda u: (-len(plan(u)), u["name"]))
    mine = us[k::n]
    ctx = M.Ctx()
    M.log(f"part {k}/{n}: {len(mine)} units")
    tot = 0
    for u in mine:
        tot += run_unit(ctx, u)
    M.log(f"part {k}/{n}: done, {tot} fits")


if __name__ == "__main__":
    main()
