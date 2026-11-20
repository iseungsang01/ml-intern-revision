"""B.79 stage 2: the boosted regressions per unit, rule and target (registration section 4, Addendum 1 section 4).

    py ces_prediction/experiments/b79_beam_inputs/fits79.py --part K/N [--units NAME,...]
Output: data/.b79/fits/{unit}__{learner}__{rule}__{target}.npz  (idx = rows of rows.npz, est, fit_rows, n_iter)
  learners T0, TN, TX, TXN (every unit, rule D10/D650/S, target F/L5); TEN, TBN (V1, V4; D650, S; F, L5);
  TNp (the nineteen beam scalars permuted over rows) and TXNp (the fourteen fast scalars permuted), V1, D650, L5.
The history, fast and anchor inputs are B.75's (b75_availability/stage_fits.Ctx); the beam scalars come from
data/.b79/beam_{rule}.npz.
"""
from __future__ import annotations

import argparse
import time
import warnings

import common79 as M

import numpy as np  # noqa: E402
import stage_fits as SF75  # noqa: E402  (b75_availability/stage_fits.py)

warnings.filterwarnings("ignore")
C = M.C
LEARNERS = ("T0", "TN", "TX", "TXN", "TEN", "TBN", "TNp", "TXNp")


def plan(u):
    out = []
    for X in M.TARGETS:
        for r in M.RULES:
            out += [(L, r, X) for L in ("T0", "TN", "TX", "TXN")]
            if u["design"] in ("V1", "V4") and r in ("D650", "S"):
                out += [("TEN", r, X), ("TBN", r, X)]
    if u["design"] == "V1":
        out += [("TNp", "D650", "L5"), ("TXNp", "D650", "L5")]
    return out


def path(u, L, r, X):
    return M.FITS / f"{u['name']}__{L}__{r}__{X}.npz"


def inputs_of(L, z, B, rows):
    """Learner matrix (Addendum 1 section 4): z = B.75's Lazy (H10, f8, A), B = the nineteen beam scalars."""
    h = z.H10[rows]
    if L == "T0":
        return h
    if L in ("TN", "TNp"):
        return np.column_stack([h, B[rows]])
    if L == "TX":
        return np.column_stack([h, z.f8[rows], z.A[rows]])
    if L in ("TXN", "TXNp"):
        return np.column_stack([h, z.f8[rows], z.A[rows], B[rows]])
    if L == "TEN":
        return np.column_stack([h, B[rows], z.f8[rows][:, 4:], z.A[rows][:, 3:]])
    if L == "TBN":
        return np.column_stack([h, B[rows], z.f8[rows][:, :4], z.A[rows][:, :3]])
    raise KeyError(L)


PERM_COLS = {"TNp": slice(10, 10 + M.NB), "TXNp": slice(10, 24)}


def fit_one(ctx, u, L, rule, X, z, Gr, B, setmask):
    """Fit learner L on the unit's fit rows in setmask, estimate its test rows in setmask.  Returns (te, est, fit rows,
    iterations)."""
    R = ctx.R
    tgt = R[f"tg_{X}"]
    fr = np.flatnonzero(np.isin(R["fidx"], u["fit"]) & setmask)
    te = np.flatnonzero(np.isin(R["fidx"], u["test"]) & setmask)
    Xf, Xt = inputs_of(L, z, B, fr), inputs_of(L, z, B, te)
    if not (np.isfinite(Xf).all() and np.isfinite(Xt).all() and np.isfinite(Gr[fr]).all() and np.isfinite(Gr[te]).all()
            and np.isfinite(tgt[fr]).all()):
        raise SystemExit(f"FATAL: {u['name']} {L} {rule} {X}: a non-finite input on the rows of the set")
    if L in PERM_COLS:
        rng = M.rng_for(f"placebo {L}|{u['name']}")
        cs = PERM_COLS[L]
        Xf[:, cs] = Xf[rng.permutation(len(fr)), cs]
        Xt[:, cs] = Xt[rng.permutation(len(te)), cs]
    m = SF75.gbr().fit(Xf, (tgt - Gr)[fr])
    return te, Gr[te] + m.predict(Xt), len(fr), int(m.n_iter_)


def set_of(sets, rule, X):
    return sets[f"S_{X}" if rule == "S" else f"common_{X}"]


def run_unit(ctx, sets, u):
    todo = [p for p in plan(u) if not path(u, *p).exists()]
    if not todo:
        return 0
    t0 = time.time()
    fast8, ms_b, ms_e = ctx.unit_arrays(u)
    nfit = 0
    for r in M.RULES:
        tl = [p for p in todo if p[1] == r]
        if not tl:
            continue
        z, Gr = ctx.matrices(r, fast8, ms_b, ms_e)
        B = M.load_beam(r)
        for L, _, X in tl:
            te, est, nf, ni = fit_one(ctx, u, L, r, X, z, Gr, B, set_of(sets, r, X))
            M.save_npz(path(u, L, r, X), idx=te, est=est, fit_rows=np.int64(nf), n_iter=np.int64(ni))
            nfit += 1
        del z, Gr, B
    M.log(f"{u['name']}: {nfit} fits ({time.time() - t0:.0f}s)")
    return nfit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="0/1")
    ap.add_argument("--units", default="")
    a = ap.parse_args()
    k, n = (int(x) for x in a.part.split("/"))
    ctx = SF75.Ctx()
    sets = M.load_sets()
    us, _, _ = M.units(M.load_keys(C.CACHE / "rows.npz", ("fidx", "shot", "clus", "r2_k", "r3_k", "r4_k")))
    if a.units:
        names = set(a.units.split(","))
        us = [u for u in us if u["name"] in names]
    cost = {u["name"]: len(plan(u)) for u in us}
    us = sorted(us, key=lambda u: (-cost[u["name"]], u["name"]))
    mine = us[k::n]
    M.log(f"part {k}/{n}: {len(mine)} units, {sum(cost[u['name']] for u in mine)} fits planned")
    tot = 0
    for u in mine:
        tot += run_unit(ctx, sets, u)
    M.log(f"part {k}/{n}: done, {tot} fits")


if __name__ == "__main__":
    main()
