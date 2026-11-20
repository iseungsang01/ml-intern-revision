"""B.77 stage 1: per unit the regressions T0, TX, C0, CX, CB, CE and the classifiers PH, PHX, FH, FHX; under V1 also the
placebo CXp and the row t - 1 control CXlag (Addendum 1 sections 3-7).

    py ces_prediction/experiments/b77_correct/fits77.py --part K/N [--units NAME,...]
Output: data/.b77/fits/{unit}__{L}.npz  (idx = rows of rows.npz; est (regressions) or p (classifiers); fit_rows; n_iter)
The log prints row counts and times only.
"""
from __future__ import annotations

import argparse
import time
import warnings

import common77 as M

import numpy as np  # noqa: E402

import stage_fits as SF  # noqa: E402  (B.75: Ctx, gbr)
import stage_trees as ST  # noqa: E402  (B.73: gbc)

warnings.filterwarnings("ignore")
K, C = M.K, M.C


def plan(u):
    out = list(M.REGS) + list(M.CLFS)
    if u["design"] == "V1":
        out += list(M.V1_CTRL)
    return out


def finite(X, what):
    if not np.all(np.isfinite(X)):
        raise SystemExit(f"FATAL: {what}: {int((~np.isfinite(X)).sum())} non-finite inputs")
    return X


class Base:
    def __init__(self):
        self.ctx = SF.Ctx()
        self.R = M.rows()
        self.inset = M.row_set(self.R, self.ctx.S)
        I = K.load_inputs(M.RULE)
        self.F4 = M.frame_scalars(self.R["y"], I)
        self.lab = np.zeros(len(self.R["y"]), np.int64)
        self.lab[self.inset] = M.label(self.R["y"][self.inset], self.R["tg_L5"][self.inset])


def run_unit(B, u, only=None):
    todo = [L for L in plan(u) if (only is None or L in only) and not M.fit_path(u, L).exists()]
    if not todo:
        return 0
    t0 = time.time()
    R = B.R
    UI = M.UnitInputs(B.ctx, u, B.F4, lag="CXlag" in todo)
    fit = np.flatnonzero(np.isin(R["fidx"], u["fit"]) & B.inset)
    te = np.flatnonzero(np.isin(R["fidx"], u["test"]) & B.inset)
    G = UI.G
    resid = R["tg_L5"] - G
    for L in todo:
        if L in M.REGS:
            m = SF.gbr().fit(finite(UI.X(L, fit), L), finite(resid[fit], "target"))
            M.save_npz(M.fit_path(u, L), idx=te, est=G[te] + m.predict(finite(UI.X(L, te), L)),
                       fit_rows=np.int64(len(fit)), n_iter=np.int64(m.n_iter_))
        elif L in M.CLFS:
            m = ST.gbc().fit(finite(UI.X(L, fit), L), B.lab[fit])
            M.save_npz(M.fit_path(u, L), idx=te, p=m.predict_proba(finite(UI.X(L, te), L))[:, 1],
                       fit_rows=np.int64(len(fit)), n_iter=np.int64(m.n_iter_))
        elif L == "CXp":
            rng = M.rng_for(f"placebo|{u['name']}")
            pf = rng.permutation(len(fit))
            pt = rng.permutation(len(te))
            Xf = UI.X("CXp", fit, fast14=UI.fast14(fit)[pf])
            Xt = UI.X("CXp", te, fast14=UI.fast14(te)[pt])
            m = SF.gbr().fit(finite(Xf, L), resid[fit])
            M.save_npz(M.fit_path(u, L), idx=te, est=G[te] + m.predict(finite(Xt, L)),
                       fit_rows=np.int64(len(fit)), n_iter=np.int64(m.n_iter_))
        elif L == "CXlag":
            ok = np.isfinite(UI.lag).all(1)
            fl, tl = fit[ok[fit]], te[ok[te]]
            m = SF.gbr().fit(finite(UI.X("CXlag", fl, fast14=UI.lag[fl]), L), resid[fl])
            M.save_npz(M.fit_path(u, L), idx=tl, est=G[tl] + m.predict(finite(UI.X("CXlag", tl, fast14=UI.lag[tl]), L)),
                       fit_rows=np.int64(len(fl)), n_iter=np.int64(m.n_iter_),
                       dropped_fit=np.int64(len(fit) - len(fl)), dropped_test=np.int64(len(te) - len(tl)))
        else:
            raise KeyError(L)
    M.log(f"{u['name']}: {len(todo)} fits, fit rows {len(fit):,}, test rows {len(te):,} ({time.time() - t0:.0f}s)")
    return len(todo)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="0/1")
    ap.add_argument("--units", default="")
    a = ap.parse_args()
    k, n = (int(x) for x in a.part.split("/"))
    B = Base()
    us, _, _ = M.units(B.R)
    if a.units:
        names = set(a.units.split(","))
        us = [u for u in us if u["name"] in names]
    us = sorted(us, key=lambda u: (-len(plan(u)), u["name"]))
    mine = us[k::n]
    M.log(f"part {k}/{n}: {len(mine)} units; row set {int(B.inset.sum()):,} rows")
    tot = 0
    for u in mine:
        tot += run_unit(B, u)
    M.log(f"part {k}/{n}: done, {tot} fits")


if __name__ == "__main__":
    main()
