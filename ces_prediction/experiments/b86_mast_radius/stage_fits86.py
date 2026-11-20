"""B.86 stage 3: T0 and TX per band, unit, rule and target (Addendum 1 section 5).

    py ces_prediction/experiments/b86_mast_radius/stage_fits86.py --part K/N
Output: data/.b86/fits/{band}/{unit}__{learner}__{rule}__{target}.npz (idx = rows of rows_{band}.npz, est, fit_rows, n_iter).
Progress lines carry counts only.
"""
from __future__ import annotations

import argparse
import time

import common86 as C

import numpy as np  # noqa: E402

KEYS = (["shot", "y", "L5", "rho", "te_t", "ne_t", "dedge_t", "common_F", "common_L5"]
        + [f"{r}_{k}" for r in C.RULES for k in ("H", "age", "G", "th", "te_a", "ne_a")])


def gbr():
    from sklearn.ensemble import HistGradientBoostingRegressor
    return HistGradientBoostingRegressor(random_state=C.TREE_SEED)


def design_matrix(R, L, r):
    """As b76 stage_fits.Ctx.X_of: T0 = the ten history scalars; TX = those and the nine Thomson scalars."""
    H = C.K.hist_matrix(R, r)
    if L == "T0":
        return H
    if L == "TX":
        return np.column_stack([H, C.K.ts_matrix(R, r)])
    raise KeyError(L)


def fit_estimate(R, L, r, X, fr, te):
    """Fit (target - G) on rows fr, estimate G + prediction on rows te (b76 stage_fits.run_unit)."""
    tgt = R["y"] if X == "F" else R["L5"]
    G = R[f"{r}_G"]
    Xm = design_matrix(R, L, r)
    Xf, Xt = Xm[fr], Xm[te]
    if not (np.isfinite(Xf).all() and np.isfinite(Xt).all()):
        raise SystemExit(f"FATAL: {L} {r} {X}: inputs not finite on the set")
    mdl = gbr().fit(Xf, (tgt - G)[fr])
    return G[te] + mdl.predict(Xt), int(mdl.n_iter_)


def run_unit(R, b, u):
    todo = [p for p in C.plan_of(u) if not C.fit_path(b, u["name"], *p).exists()]
    if not todo:
        return 0
    t0 = time.time()
    fit_s = np.isin(R["shot"], np.array(u["fit"]))
    test_s = np.isin(R["shot"], np.array(u["test"]))
    for L, r, X in todo:
        sm = R[f"common_{X}"]
        fr = np.flatnonzero(fit_s & sm)
        te = np.flatnonzero(test_s & sm)
        est, nit = fit_estimate(R, L, r, X, fr, te)
        C.save_npz(C.fit_path(b, u["name"], L, r, X), idx=te, est=est, fit_rows=np.int64(len(fr)), n_iter=np.int64(nit))
    C.log(f"{b} {u['name']}: {len(todo)} fits ({time.time() - t0:.0f}s)")
    return len(todo)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="0/1")
    a = ap.parse_args()
    k, n = (int(x) for x in a.part.split("/"))
    jobs = []
    for b in C.BAND_NAMES:
        us, _ = C.units(b)
        jobs += [(b, u) for u in us]
    mine = jobs[k::n]
    C.log(f"part {k}/{n}: {len(mine)} units")
    tot = 0
    cur, R = None, None
    for b, u in mine:
        if b != cur:
            R = C.load_band(b, KEYS)
            cur = b
        tot += run_unit(R, b, u)
    C.log(f"part {k}/{n}: done, {tot} fits")


if __name__ == "__main__":
    main()
