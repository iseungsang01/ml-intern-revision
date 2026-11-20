"""B.81 (PREREGISTRATION_B81.md with Addendum 1): shared paths, constants, the unit context and the learner matrices.

Reuses b75_availability/common75.py (inputs, history and anchor scalars, units; it imports b73_targets/common.py, which
sets OMP/MKL/OPENBLAS threads to 2 and hides CUDA) and b78_sparse/fits78.inputs_of (the matrices of T0, TX, TXp, TB, TE,
TXo).  Caches under data/.b81/.  Run every stage from the repo root with `py`.
"""
from __future__ import annotations

import sys
import zlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "b78_sparse"))
sys.path.insert(0, str(HERE.parent / "b75_availability"))
import common75 as K75  # noqa: E402
import fits78 as F78  # noqa: E402

import numpy as np  # noqa: E402

C = K75.C
CACHE = C.DATA / ".b81"
FITS = CACHE / "fits"
CHK = CACHE / "checks"
LOGS = CACHE / "logs"
B75FITS = K75.FITS
PAPER = C.PAPER
OUT_JSON, OUT_MD, CHECKS_MD = PAPER / "b81_controls.json", PAPER / "b81_controls.md", PAPER / "b81_checks.md"
REG = "ces_prediction/experiments/PREREGISTRATION_B81.md"
ADD = "ces_prediction/experiments/PREREGISTRATION_B81_ADDENDUM1.md"

SEED = 20261181
RULES = ("D650", "S")
SETOF = {"D650": "common_L5", "S": "S_L5"}
X = "L5"
LEARNERS = ("T0", "TX", "TXp", "TXo", "TE", "TB")
log = C.log
save_npz, save_json, save_text, load_npz = C.save_npz, C.save_json, C.save_text, C.load_npz
inputs_of = F78.inputs_of


def rng_for(name):
    return np.random.default_rng([SEED, zlib.crc32(name.encode())])


def units():
    us, tab, fifth = K75.DS.units(K75.load_keys(C.CACHE / "rows.npz", ("fidx", "shot", "clus", "r2_k", "r3_k", "r4_k")))
    return us, tab, fifth


def plan(u):
    """(learner, rule) fitted in this batch for a unit (Addendum 1 section 2)."""
    out = []
    for r in RULES:
        out += [("TXp", r), ("TXo", r)]
        if u["design"] in ("V2", "V3"):
            out += [("TE", r), ("TB", r)]
    return out


def path81(name, L, r):
    return FITS / f"{name}__{L}__{r}__{X}.npz"


def est_path(u, L, r):
    """Where the estimates of a learner are stored (B.75 for T0, TX and for TE, TB of V1 and V4; B.81 otherwise)."""
    if L in ("T0", "TX") or (L in ("TE", "TB") and u["design"] in ("V1", "V4")):
        return B75FITS / f"{u['name']}__{L}__{r}__{X}.npz"
    return path81(u["name"], L, r)


def gbr():
    from sklearn.ensemble import HistGradientBoostingRegressor
    return HistGradientBoostingRegressor(random_state=C.TREE_SEED)


class Ctx:
    def __init__(self):
        import stage_trees as ST
        self.R = K75.rows_lean()
        self.S = load_npz(K75.CACHE / "sets.npz")
        G = K75.grid_lean(("seg", "fast"))
        self.seg = G["seg"]
        self.mr_b, self.mr_e = (K75.run_mean10(x, self.seg) for x in K75.raw_means(G["fast"]))
        del G
        self.FE = ST.Feats()
        self._I = {}

    def unit_arrays(self, u):
        wf = self.FE.window(u["fit"])
        bz, ez = K75.zmeans(self.FE, u["fit"])
        ms_b, ms_e = K75.run_mean10(bz, self.seg), K75.run_mean10(ez, self.seg)
        return wf, ms_b, ms_e

    def inputs(self, rule):
        if rule not in self._I:
            self._I[rule] = K75.load_inputs(rule)
        return self._I[rule]

    def matrices(self, rule, wf, ms_b, ms_e):
        R = self.R
        I = self.inputs(rule)
        H10 = K75.history_matrix(R["tstep"], I)
        A = K75.anchor_scalars(R["gi"], I["anc_g"], ms_b, ms_e, self.mr_b, self.mr_e)
        return H10, wf[R["gi"]], A, I

    def rows_of(self, u, rule):
        sm = self.S[SETOF[rule]]
        fr = np.flatnonzero(np.isin(self.R["fidx"], u["fit"]) & sm)
        te = np.flatnonzero(np.isin(self.R["fidx"], u["test"]) & sm)
        return fr, te


def design_matrices(L, u, rule, H10, f8, A, I, wf, fr, te):
    """Fit and test matrices of learner L for a unit and rule; TXp permuted as Addendum 1 section 2."""
    Lb = "TX" if L == "TXp" else L
    Xf = inputs_of(Lb, H10, f8, A, fr, wf, I["anc_g"])
    Xt = inputs_of(Lb, H10, f8, A, te, wf, I["anc_g"])
    if L == "TXp":
        rng = rng_for(f"placebo|{u['name']}|{rule}")
        Xf[:, 10:] = Xf[rng.permutation(len(fr)), 10:]
        Xt[:, 10:] = Xt[rng.permutation(len(te)), 10:]
    return Xf, Xt


def ecei_constant():
    """Grid rows (data/.b73/grid.npz) in which the four raw ECEI channels have not changed for at least ten rows of the
    same file: lead_checks/stuck_ecei.py `stuck(E, 10)`, the run length computed without a Python loop (same values)."""
    G = K75.grid_lean(("fast", "fidx"))
    E = G["fast"][:, C.NB:C.NB + C.NE].astype(np.float64)
    gf = G["fidx"]
    N = len(gf)
    eq1 = np.zeros(N, bool)
    eq1[1:] = (E[1:] == E[:-1]).all(1) & (gf[1:] == gf[:-1])
    idx = np.arange(N)
    last_false = np.maximum.accumulate(np.where(~eq1, idx, 0))
    run = idx - last_false
    return run >= 10, run


def fit_estimate(Xf, Xt, tgt, Gr, fr, te):
    m = gbr().fit(Xf, (tgt - Gr)[fr])
    return Gr[te] + m.predict(Xt), m
