"""B.76 stage 3: the boosted regressions and the two-coefficient ratio form per unit, rule and target (registration
sections 5-8, Addendum 1 section 6).

    py ces_prediction/experiments/b76_mast_availability/stage_fits.py --part K/N
Output: data/.b76/fits/{unit}__{learner}__{rule}__{target}.npz  (idx = rows of rows.npz, est, fit_rows, n_iter; RTa: alpha,
beta).  Learners on the common set: T0, TX, RTa (every unit, rule, target); TXp (placebo), TXo (Thomson of the anchor
frame only) for M1 units, D100, L5.  On the DA set (M1 and M4 units, D5 and D100): T0d, TXd, TD, TA.
Progress lines carry counts only.
"""
from __future__ import annotations

import argparse
import time
import warnings

import common76 as K

import numpy as np  # noqa: E402

warnings.filterwarnings("ignore")


def gbr():
    from sklearn.ensemble import HistGradientBoostingRegressor
    return HistGradientBoostingRegressor(random_state=K.TREE_SEED)


def plan(u):
    out = []
    for X in K.TARGETS:
        for r in K.RULES:
            for L in ("T0", "TX", "RTa"):
                out.append((L, r, X, "common"))
        if u["design"] in ("M1", "M4"):
            for r in K.DA_RULES:
                for L in ("T0d", "TXd", "TD", "TA"):
                    out.append((L, r, X, "DA"))
    if u["design"] == "M1":
        out += [("TXp", "D100", "L5", "common"), ("TXo", "D100", "L5", "common")]
    return out


def path(u, L, r, X):
    return K.FITS / f"{u['name']}__{L}__{r}__{X}.npz"


class Ctx:
    def __init__(self):
        keys = (["shot", "y", "L5", "rho", "te_t", "ne_t", "dedge_t", "da_t", "dmx_t"]
                + [f"{r}_{k}" for r in K.RULES for k in ("H", "age", "G", "th", "te_a", "ne_a", "da_a", "dmx_a")])
        self.R = K.load_rows(keys)
        self.S = K.load_npz(K.CACHE / "sets.npz")
        self.cache = {}

    def mats(self, r):
        if r not in self.cache:
            R = self.R
            self.cache[r] = {"H": K.hist_matrix(R, r), "T": K.ts_matrix(R, r), "D": K.da_matrix(R, r), "G": R[f"{r}_G"],
                             "Hold": R[f"{r}_H"]}
        return self.cache[r]

    def X_of(self, L, r):
        m = self.mats(r)
        if L in ("T0", "T0d"):
            return m["H"]
        if L in ("TX", "TXd", "TXp"):
            return np.column_stack([m["H"], m["T"]])
        if L == "TD":
            return np.column_stack([m["H"], m["D"]])
        if L == "TA":
            return np.column_stack([m["H"], m["T"], m["D"]])
        if L == "TXo":
            return np.column_stack([m["H"], m["T"][:, 2:4]])
        raise KeyError(L)


def run_unit(ctx, u):
    todo = [p for p in plan(u) if not path(u, *p[:3]).exists()]
    if not todo:
        return 0
    t0 = time.time()
    R, S = ctx.R, ctx.S
    fit_s = np.isin(R["shot"], np.array(u["fit"]))
    test_s = np.isin(R["shot"], np.array(u["test"]))
    nfit = 0
    for L, r, X, setname in todo:
        tgt = R["y"] if X == "F" else R["L5"]
        sm = S[f"{setname}_{X}"]
        fr = np.flatnonzero(fit_s & sm)
        te = np.flatnonzero(test_s & sm)
        m = ctx.mats(r)
        G = m["G"]
        if L == "RTa":
            H = m["Hold"]
            lr = m["T"][:, 6]
            A = np.column_stack([lr[fr], np.ones(len(fr))])
            b = np.log(tgt[fr]) - np.log(H[fr])
            coef = np.linalg.lstsq(A, b, rcond=None)[0]
            est = H[te] * np.exp(coef[0] * lr[te] + coef[1])
            K.save_npz(path(u, L, r, X), idx=te, est=est, fit_rows=np.int64(len(fr)), n_iter=np.int64(-1),
                       alpha=np.float64(coef[0]), beta=np.float64(coef[1]))
            continue
        Xm = ctx.X_of(L, r)
        Xf, Xt = Xm[fr], Xm[te]
        if L == "TXp":
            nh = len(K.HIST_NAMES)
            pf = K.rng_for(f"placebo|{u['name']}|fit").permutation(len(fr))
            pt = K.rng_for(f"placebo|{u['name']}|test").permutation(len(te))
            Xf = Xf.copy()
            Xt = Xt.copy()
            Xf[:, nh:] = Xf[pf, nh:]
            Xt[:, nh:] = Xt[pt, nh:]
        if not (np.isfinite(Xf).all() and np.isfinite(Xt).all()):
            raise SystemExit(f"FATAL: {u['name']} {L} {r} {X}: inputs not finite on the set")
        mdl = gbr().fit(Xf, (tgt - G)[fr])
        nfit += 1
        est = G[te] + mdl.predict(Xt)
        K.save_npz(path(u, L, r, X), idx=te, est=est, fit_rows=np.int64(len(fr)), n_iter=np.int64(mdl.n_iter_))
    K.log(f"{u['name']}: {nfit} tree fits, {len(todo)} outputs ({time.time() - t0:.0f}s)")
    return nfit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="0/1")
    a = ap.parse_args()
    k, n = (int(x) for x in a.part.split("/"))
    ctx = Ctx()
    us, _ = K.units()
    cost = {"M1": 34, "M4": 32, "M2": 24, "M3": 24}
    us = sorted(us, key=lambda u: (-cost[u["design"]], u["name"]))
    mine = us[k::n]
    K.log(f"part {k}/{n}: {len(mine)} units")
    tot = 0
    for u in mine:
        tot += run_unit(ctx, u)
    K.log(f"part {k}/{n}: done, {tot} tree fits")


if __name__ == "__main__":
    main()
