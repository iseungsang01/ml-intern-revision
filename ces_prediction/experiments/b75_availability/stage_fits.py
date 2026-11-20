"""B.75 stage 2: the boosted regressions per unit, rule and target (registration sections 4-6, Addendum 1 section 4).

    py ces_prediction/experiments/b75_availability/stage_fits.py --part K/N [--units NAME,...]
Output: data/.b75/fits/{unit}__{learner}__{rule}__{target}.npz  (idx = rows of rows.npz, est, fit_rows, n_iter)
  learners T0, TX, TX8, TB, TE, X0 (rule "C" = common set); T0a, TXa = the D10 fit applied without refit;
  rule "D10S" = the D10 history fitted and scored on the S set.
"""
from __future__ import annotations

import argparse
import time
import warnings

import common75 as K

import numpy as np  # noqa: E402

warnings.filterwarnings("ignore")
C = K.C


def gbr():
    from sklearn.ensemble import HistGradientBoostingRegressor
    return HistGradientBoostingRegressor(random_state=C.TREE_SEED)


def plan(u):
    """List of (learner, rule, target) outputs of a unit (registration section 6)."""
    out = []
    if u["design"] == "V3c":
        for r in ("D10", "D650"):
            for L in ("T0", "TX"):
                out.append((L, r, "L5"))
        return out
    for X in K.TARGETS:
        for r in K.FIXED + ("S", "D10S"):
            for L in ("T0", "TX"):
                out.append((L, r, X))
        for r in K.FIXED[1:] + ("S",):
            for L in ("T0a", "TXa"):
                out.append((L, r, X))
        if u["design"] in ("V1", "V4"):
            for r in ("D10", "D650", "S"):
                for L in ("TX8", "TB", "TE"):
                    out.append((L, r, X))
        out.append(("X0", "C", X))
    return out


def path(u, L, r, X):
    return K.FITS / f"{u['name']}__{L}__{r}__{X}.npz"


class Ctx:
    def __init__(self):
        import stage_trees as ST
        self.R = K.rows_lean()
        self.S = K.load_npz(K.CACHE / "sets.npz")
        G = K.grid_lean(("seg", "fast"))
        self.seg = G["seg"]
        self.mr_b, self.mr_e = (K.run_mean10(x, self.seg) for x in K.raw_means(G["fast"]))
        del G
        self.FE = ST.Feats()

    def unit_arrays(self, u):
        wf = self.FE.window(u["fit"])
        bz, ez = K.zmeans(self.FE, u["fit"])
        ms_b, ms_e = K.run_mean10(bz, self.seg), K.run_mean10(ez, self.seg)
        return wf[self.R["gi"]], ms_b, ms_e

    def matrices(self, rule, fast8, ms_b, ms_e):
        """rule inputs -> XS (rows -> matrix of a learner, built on demand), G."""
        I = K.load_inputs(rule)
        H10 = K.history_matrix(self.R["tstep"], I)
        A = K.anchor_scalars(self.R["gi"], I["anc_g"], ms_b, ms_e, self.mr_b, self.mr_e)
        return Lazy(H10, fast8, A), I["G"]


class Lazy:
    """Learner inputs (Addendum 1 section 3), built only for the rows asked for."""

    def __init__(self, H10, fast8, A):
        self.H10, self.f8, self.A = H10, fast8, A

    def __getitem__(self, L):
        return _Sel(self, L)


class _Sel:
    def __init__(self, z, L):
        self.z, self.L = z, L

    def __getitem__(self, rows):
        z, L = self.z, self.L
        h = z.H10[rows]
        if L == "T0":
            return h
        if L == "TX8":
            return np.column_stack([h, z.f8[rows]])
        if L == "TX":
            return np.column_stack([h, z.f8[rows], z.A[rows]])
        if L == "TB":
            return np.column_stack([h, z.f8[rows][:, :4], z.A[rows][:, :3]])
        if L == "TE":
            return np.column_stack([h, z.f8[rows][:, 4:], z.A[rows][:, 3:]])
        raise KeyError(L)


def run_unit(ctx, u):
    todo = [p for p in plan(u) if not path(u, *p).exists()]
    if not todo:
        return 0
    t0 = time.time()
    R, S = ctx.R, ctx.S
    fast8, ms_b, ms_e = ctx.unit_arrays(u)
    fit_d = np.isin(R["fidx"], u["fit"])
    test_d = np.isin(R["fidx"], u["test"])
    nfit = 0
    for X in K.TARGETS:
        tl = [p for p in todo if p[2] == X]
        if not tl:
            continue
        tgt = R[f"tg_{X}"]
        need = {(L, r) for L, r, _ in tl}
        # rules whose matrices are needed
        rules_needed = set()
        for L, r in need:
            if r == "C":
                continue
            if r == "D10S":
                rules_needed.add("D10")
            else:
                rules_needed.add(r)
            if L in ("T0a", "TXa"):
                rules_needed.add("D10")
        # applied-without-refit models: D10 on the common set; D10S on the S set
        models = {}

        def fit_store(L, r, XS, Gr, setname, store=True):
            nonlocal nfit
            sm = S[f"{setname}_{X}"]
            fr = fit_d & sm
            te = np.flatnonzero(test_d & sm)
            m = gbr().fit(XS[L][fr], (tgt - Gr)[fr])
            nfit += 1
            if store:
                est = Gr[te] + m.predict(XS[L][te])
                K.save_npz(path(u, L, r, X), idx=te, est=est, fit_rows=np.int64(fr.sum()), n_iter=np.int64(m.n_iter_))
            return m

        def store_applied(La, r, m, XS, Gr, setname):
            sm = S[f"{setname}_{X}"]
            te = np.flatnonzero(test_d & sm)
            L = La[:-1]
            est = Gr[te] + m.predict(XS[L][te])
            K.save_npz(path(u, La, r, X), idx=te, est=est, fit_rows=np.int64(-1), n_iter=np.int64(-1))

        # D10 first (fits and models for the applied rows)
        need_applied_fixed = any(L in ("T0a", "TXa") and r in K.FIXED for L, r in need)
        need_applied_S = any(L in ("T0a", "TXa") and r == "S" for L, r in need)
        if "D10" in rules_needed:
            XS, Gr = ctx.matrices("D10", fast8, ms_b, ms_e)
            for L in ("T0", "TX"):
                if (L, "D10") in need or need_applied_fixed:
                    models[("D10", L)] = fit_store(L, "D10", XS, Gr, "common", store=(L, "D10") in need)
                if (L, "D10S") in need or need_applied_S:
                    models[("D10S", L)] = fit_store(L, "D10S", XS, Gr, "S", store=(L, "D10S") in need)
            for L in ("TX8", "TB", "TE"):
                if (L, "D10") in need:
                    fit_store(L, "D10", XS, Gr, "common")
            del XS
        for r in K.FIXED[1:] + ("S",):
            if r not in rules_needed:
                continue
            setname = "S" if r == "S" else "common"
            XS, Gr = ctx.matrices(r, fast8, ms_b, ms_e)
            for L in ("T0", "TX", "TX8", "TB", "TE"):
                if (L, r) in need:
                    fit_store(L, r, XS, Gr, setname)
            for La in ("T0a", "TXa"):
                if (La, r) in need:
                    m = models[("D10S" if r == "S" else "D10", La[:-1])]
                    store_applied(La, r, m, XS, Gr, setname)
            del XS
        if ("X0", "C") in need:
            sm = S[f"common_{X}"]
            fr = fit_d & sm
            te = np.flatnonzero(test_d & sm)
            Xx = np.column_stack([R["tstep"], fast8])
            med = float(np.median(tgt[fr]))
            m = gbr().fit(Xx[fr], (tgt - med)[fr])
            nfit += 1
            K.save_npz(path(u, "X0", "C", X), idx=te, est=med + m.predict(Xx[te]), fit_rows=np.int64(fr.sum()),
                       n_iter=np.int64(m.n_iter_), median=np.float64(med))
    K.log(f"{u['name']}: {nfit} fits, {len(todo)} outputs ({time.time() - t0:.0f}s)")
    return nfit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="0/1")
    ap.add_argument("--units", default="")
    a = ap.parse_args()
    k, n = (int(x) for x in a.part.split("/"))
    ctx = Ctx()
    us, _, _ = K.units_all(K.load_keys(C.CACHE / "rows.npz", ("fidx", "shot", "clus", "r2_k", "r3_k", "r4_k")))
    if a.units:
        names = set(a.units.split(","))
        us = [u for u in us if u["name"] in names]
    # cost order: V1/V4 units carry 52 fits, V2/V3 34, V3c 4; deal them out round-robin
    cost = {"V1": 52, "V4": 52, "V2": 34, "V3": 34, "V3c": 4}
    us = sorted(us, key=lambda u: -cost[u["design"]])
    mine = us[k::n]
    K.log(f"part {k}/{n}: {len(mine)} units")
    tot = 0
    for u in mine:
        tot += run_unit(ctx, u)
    K.log(f"part {k}/{n}: done, {tot} fits")


if __name__ == "__main__":
    main()
