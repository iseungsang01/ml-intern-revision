"""B.84 (PREREGISTRATION_B84.md with Addendum 1): shared paths, constants, the raw history windows per rule and the
learner matrices of a unit.

Reuses b75_availability/common75.py (inputs, history and anchor scalars, fast means; it imports b73_targets/common.py,
which sets OMP/MKL/OPENBLAS threads to 2 and hides CUDA) and b78_sparse/common78.py (frame index, K10 inputs).
Caches under data/.b84/.  Run every stage from the repo root with `py`.
"""
from __future__ import annotations

import sys
import zlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "b78_sparse"))
sys.path.insert(0, str(HERE.parent / "b75_availability"))
import common75 as K75  # noqa: E402  (sets OMP/MKL/OPENBLAS threads to 2 and hides CUDA)
import common78 as M78  # noqa: E402

import numpy as np  # noqa: E402

C = K75.C
CACHE = C.DATA / ".b84"
RAW = CACHE / "raw"
FITS = CACHE / "fits"
CHK = CACHE / "checks"
LOGS = CACHE / "logs"
PAPER = C.PAPER
OUT_JSON, OUT_MD, CHECKS_MD = PAPER / "b84_onefit.json", PAPER / "b84_onefit.md", PAPER / "b84_checks.md"
REG = "ces_prediction/experiments/PREREGISTRATION_B84.md"
ADD = "ces_prediction/experiments/PREREGISTRATION_B84_ADDENDUM1.md"

SEED = 20261184
X = "L5"
MIX = ("D10", "D50", "D100", "D300", "D650", "D1000", "S")
ALL_RULES = MIX + ("K10",)
UNIT_NAMES = tuple(f"V1_r2_k{k}" for k in range(5)) + tuple(f"V3_q{q}" for q in range(1, 5)) + ("V4a",)
MIX_LEARNERS = ("T0all", "TXall", "T0raw", "TXraw")
REL_RULES = ("D650", "S")
REL_DROP = (10, 14, 18, 21)          # x_t of BES, x_t of ECEI, a1 of BES, a1 of ECEI in TX's 24 columns (0-based)
log = C.log
save_npz, save_json, save_text, load_npz, load_keys = C.save_npz, C.save_json, C.save_text, C.load_npz, K75.load_keys


def rng_for(name):
    return np.random.default_rng([SEED, zlib.crc32(name.encode())])


def set_of(rule, sets75, sets78):
    if rule == "K10":
        return sets78["common_L5"]
    if rule == "S":
        return sets75["S_L5"]
    return sets75["common_L5"]


def load_inputs(rule):
    return M78.load_inputs("K10") if rule == "K10" else K75.load_inputs(rule)


def units():
    us, tab, fifth = K75.DS.units(load_keys(C.CACHE / "rows.npz", ("fidx", "shot", "clus", "r2_k", "r3_k", "r4_k")))
    by = {u["name"]: u for u in us}
    return [by[n] for n in UNIT_NAMES], tab, fifth


# ------------------------------------------------------------------------------------------ raw history windows
def raw_hist_file(tm, ti, seg, rows, rule, fr=None):
    """Raw history of the given file-local rows under `rule` (Addendum 1 section 2): the last up to ten visible values of
    the block of t, newest first, as (value, age in ms), plus the file-local anchor row (or -1).  Visibility exactly as
    common75.file_inputs (fixed delays, S) or common78.file_inputs_k (K10).  Never reads whether row t carries a value."""
    obs = np.flatnonzero(np.isfinite(ti))
    n_all = len(tm)
    newblk = np.concatenate(([True], seg[1:] != seg[:-1]))
    first = np.maximum.accumulate(np.where(newblk, np.arange(n_all), 0))
    if rule == "K10":
        kept = obs[(fr[obs] >= 0) & (fr[obs] % 10 == 0)]
        p = np.searchsorted(kept, rows, side="left")
    else:
        kept = obs
        lim, _ = K75.limit_stamps(tm, seg, rows, rule)
        p = np.searchsorted(tm[obs], lim + K75.TOL, side="right")
    b = np.searchsorted(kept, first[rows], side="left")
    n = len(rows)
    V = np.full((n, 10), np.nan)
    A = np.full((n, 10), np.nan)
    for k in range(10):
        j = p - 1 - k
        ok = j >= b
        idx = kept[j[ok]]
        if np.any(idx >= rows[ok]):
            raise SystemExit(f"FATAL: a visible value at or after its row under {rule}")
        V[ok, k] = ti[idx]
        A[ok, k] = (tm[rows[ok]] - tm[idx]) * 1000.0
    anc = np.full(n, -1, np.int64)
    has = p > b
    anc[has] = kept[p[has] - 1]
    return V, A, anc


def raw_path(rule):
    return RAW / f"{rule}.npz"


def load_raw(rule):
    return load_npz(raw_path(rule))


# ------------------------------------------------------------------------------------------ unit context
class Ctx:
    def __init__(self):
        import stage_trees as ST
        self.R = K75.rows_lean()
        self.sets75 = load_npz(K75.CACHE / "sets.npz")
        self.sets78 = load_npz(M78.CACHE / "sets.npz")
        G = K75.grid_lean(("seg", "fast"))
        self.seg = G["seg"]
        self.mr_b, self.mr_e = (K75.run_mean10(x, self.seg) for x in K75.raw_means(G["fast"]))
        del G
        self.FE = ST.Feats()

    def rows_of(self, u, rule, part):
        R = self.R
        d = np.isin(R["fidx"], u["fit"] if part == "fit" else u["test"])
        return np.flatnonzero(d & set_of(rule, self.sets75, self.sets78))


class UnitArrays:
    """The fast arrays of a unit (standardisation with its fit discharges)."""

    def __init__(self, ctx, u):
        self.ctx = ctx
        self.wf = ctx.FE.window(u["fit"])                       # grid N x 8
        self.bz, self.ez = K75.zmeans(ctx.FE, u["fit"])       # grid N
        self.ms_b, self.ms_e = K75.run_mean10(self.bz, ctx.seg), K75.run_mean10(self.ez, ctx.seg)


class RuleMats:
    """Inputs of every learner under one rule for a unit, built for the rows asked for."""

    def __init__(self, ua, rule, need_raw=True):
        ctx = ua.ctx
        R = ctx.R
        self.ua, self.rule = ua, rule
        I = load_inputs(rule)
        self.G = I["G"]
        self.H = I["H"]
        self.age = I["age"]
        self.anc_g = I["anc_g"]
        self.H10 = K75.history_matrix(R["tstep"], I)
        self.A = K75.anchor_scalars(R["gi"], I["anc_g"], ua.ms_b, ua.ms_e, ctx.mr_b, ctx.mr_e)
        self.tstep = R["tstep"]
        self.gi = R["gi"]
        self.raw = load_raw(rule) if need_raw else None
        del I

    def target(self, rows):
        return (self.ua.ctx.R["tg_L5"] - self.G)[rows]

    def X(self, L, rows):
        h = self.H10[rows]
        if L in ("T0", "T0all"):
            return h
        if L in ("TX", "TXall", "TXrel"):
            gi = self.gi[rows]
            m = np.column_stack([h, self.ua.wf[gi], self.A[rows]])
            if L == "TXrel":
                m = np.delete(m, REL_DROP, axis=1)
            return m
        if L in ("T0raw", "TXraw"):
            V, Ag = self.raw["V"][rows], self.raw["A"][rows]
            H = self.H[rows]
            hist = np.column_stack([self.tstep[rows], self.age[rows], self.G[rows] - H, V - H[:, None], Ag])
            if L == "T0raw":
                return hist
            gi = self.gi[rows]
            fb = np.column_stack([self.ua.bz[gi - k] for k in range(10)])
            fe = np.column_stack([self.ua.ez[gi - k] for k in range(10)])
            anc = self.anc_g[rows]
            return np.column_stack([hist, fb, fe, self.ua.ms_b[anc], self.ua.ms_e[anc]])
        raise KeyError(L)


def gbr():
    from sklearn.ensemble import HistGradientBoostingRegressor
    return HistGradientBoostingRegressor(random_state=C.TREE_SEED)


def build_stack(ctx, ua, u, L, rules=MIX):
    """The mixture stack of learner L for unit u: blocks of the rules in order; returns X, y, per-rule counts and the
    D1 row index of each stacked row."""
    blocks, counts = [], {}
    total = 0
    for r in rules:
        fr = ctx.rows_of(u, r, "fit")
        blocks.append((r, fr))
        counts[r] = int(len(fr))
        total += len(fr)
    ncol = {"T0all": 10, "TXall": 24, "T0raw": 23, "TXraw": 45}[L]
    Xs = np.empty((total, ncol))
    ys = np.empty(total)
    src = np.empty(total, np.int64)
    a = 0
    for r, fr in blocks:
        rm = RuleMats(ua, r, need_raw=L.endswith("raw"))
        Xs[a:a + len(fr)] = rm.X(L, fr)
        ys[a:a + len(fr)] = rm.target(fr)
        src[a:a + len(fr)] = fr
        a += len(fr)
        del rm
    return Xs, ys, counts, src
