"""B.78 stage 2: the boosted regressions per unit, rule and target (registration sections 4-6, Addendum 1 section 5).

    py ces_prediction/experiments/b78_sparse/fits78.py --part K/N [--units NAME,...]
Output: data/.b78/fits/{unit}__{learner}__{rule}__{target}.npz  (idx = rows of rows.npz, est, fit_rows, n_iter)
  learners T0, TX (every unit, rule, target); TB, TE (V1, V4; K1, K10; F, L5);
  TXp (placebo: the fourteen fast scalars permuted over rows) and TXo (the fast scalars of the anchor row), V1, K10, L5.
"""
from __future__ import annotations

import argparse
import time
import warnings

import common78 as M

import numpy as np  # noqa: E402

warnings.filterwarnings("ignore")
C = M.C


def gbr():
    from sklearn.ensemble import HistGradientBoostingRegressor
    return HistGradientBoostingRegressor(random_state=C.TREE_SEED)


def plan(u):
    out = []
    for X in M.TARGETS:
        for r in M.RULES:
            out += [("T0", r, X), ("TX", r, X)]
            if u["design"] in ("V1", "V4") and r in ("K1", "K10"):
                out += [("TB", r, X), ("TE", r, X)]
    if u["design"] == "V1":
        out += [("TXp", "K10", "L5"), ("TXo", "K10", "L5")]
    return out


def path(u, L, r, X):
    return M.FITS / f"{u['name']}__{L}__{r}__{X}.npz"


class Ctx:
    def __init__(self):
        import stage_trees as ST
        self.R = M.rows_lean()
        self.S = M.load_npz(M.CACHE / "sets.npz")
        G = M.grid_lean(("seg", "fast"))
        self.seg = G["seg"]
        self.mr_b, self.mr_e = (M.run_mean10(x, self.seg) for x in M.raw_means(G["fast"]))
        del G
        self.FE = ST.Feats()

    def unit_arrays(self, u):
        wf = self.FE.window(u["fit"])
        bz, ez = M.zmeans(self.FE, u["fit"])
        ms_b, ms_e = M.run_mean10(bz, self.seg), M.run_mean10(ez, self.seg)
        return wf, ms_b, ms_e


def matrices(ctx, rule, wf, ms_b, ms_e):
    R = ctx.R
    I = M.load_inputs(rule)
    H10 = M.history_matrix(R["tstep"], I)
    A = M.anchor_scalars(R["gi"], I["anc_g"], ms_b, ms_e, ctx.mr_b, ctx.mr_e)
    f8 = wf[R["gi"]]
    return H10, f8, A, I


def inputs_of(L, H10, f8, A, rows, wf=None, anc_g=None):
    h = H10[rows]
    if L == "T0":
        return h
    if L in ("TX", "TXp"):
        return np.column_stack([h, f8[rows], A[rows]])
    if L == "TB":
        return np.column_stack([h, f8[rows][:, :4], A[rows][:, :3]])
    if L == "TE":
        return np.column_stack([h, f8[rows][:, 4:], A[rows][:, 3:]])
    if L == "TXo":
        # the fast scalars of the anchor row; the anchor scalars formed with the anchor in place of row t reduce to a1
        # (a2 = 0, a3 = 0 are left out)
        return np.column_stack([h, wf[anc_g[rows]], A[rows][:, [0, 3]]])
    raise KeyError(L)


def run_unit(ctx, u):
    todo = [p for p in plan(u) if not path(u, *p).exists()]
    if not todo:
        return 0
    t0 = time.time()
    R, S = ctx.R, ctx.S
    wf, ms_b, ms_e = ctx.unit_arrays(u)
    fit_d = np.isin(R["fidx"], u["fit"])
    test_d = np.isin(R["fidx"], u["test"])
    nfit = 0
    for r in M.RULES:
        tl = [p for p in todo if p[1] == r]
        if not tl:
            continue
        H10, f8, A, I = matrices(ctx, r, wf, ms_b, ms_e)
        Gr = I["G"]
        for L, _, X in tl:
            tgt = R[f"tg_{X}"]
            sm = S[f"common_{X}"]
            fr = np.flatnonzero(fit_d & sm)
            te = np.flatnonzero(test_d & sm)
            Xf = inputs_of(L, H10, f8, A, fr, wf, I["anc_g"])
            Xt = inputs_of(L, H10, f8, A, te, wf, I["anc_g"])
            extra = {}
            if L == "TXp":
                rng = M.rng_for(f"placebo|{u['name']}")
                Xf[:, 10:] = Xf[rng.permutation(len(fr)), 10:]
                Xt[:, 10:] = Xt[rng.permutation(len(te)), 10:]
            if L == "TXo":
                extra["nonfinite_fit_rows"] = np.int64((~np.isfinite(Xf).all(1)).sum())
                extra["nonfinite_test_rows"] = np.int64((~np.isfinite(Xt).all(1)).sum())
            m = gbr().fit(Xf, (tgt - Gr)[fr])
            nfit += 1
            est = Gr[te] + m.predict(Xt)
            M.save_npz(path(u, L, r, X), idx=te, est=est, fit_rows=np.int64(len(fr)), n_iter=np.int64(m.n_iter_), **extra)
        del H10, f8, A, I
    M.log(f"{u['name']}: {nfit} fits ({time.time() - t0:.0f}s)")
    return nfit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="0/1")
    ap.add_argument("--units", default="")
    a = ap.parse_args()
    k, n = (int(x) for x in a.part.split("/"))
    ctx = Ctx()
    us, _, _ = M.units(M.load_keys(C.CACHE / "rows.npz", ("fidx", "shot", "clus", "r2_k", "r3_k", "r4_k")))
    if a.units:
        names = set(a.units.split(","))
        us = [u for u in us if u["name"] in names]
    cost = {u["name"]: len(plan(u)) for u in us}
    us = sorted(us, key=lambda u: (-cost[u["name"]], u["name"]))
    mine = us[k::n]
    M.log(f"part {k}/{n}: {len(mine)} units")
    tot = 0
    for u in mine:
        tot += run_unit(ctx, u)
    M.log(f"part {k}/{n}: done, {tot} fits")


if __name__ == "__main__":
    main()
