"""B.77 (PREREGISTRATION_B77.md with Addendum 1): paths, constants, the row set, MED3, the label and the learner inputs.

The D10 history inputs, the eight fast scalars and the six anchor scalars are B.75's (common75, stage_fits.Ctx), imported
unchanged; the units are B.73's (designs.units).  Caches under data/.b77/, written whole and then renamed.  Run every
stage from the repo root with `py`.
"""
from __future__ import annotations

import sys
import zlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
B75 = HERE.parent / "b75_availability"
sys.path.insert(0, str(B75))
import common75 as K  # noqa: E402  (b73_targets/common: threads 2, CUDA hidden; puts b73_targets first on sys.path)

import numpy as np  # noqa: E402

C = K.C
DS = K.DS
CACHE = C.DATA / ".b77"
FITS = CACHE / "fits"
CHK = CACHE / "checks"
PAPER = C.PAPER
OUT_JSON, OUT_MD, CHECKS_MD = PAPER / "b77_correct.json", PAPER / "b77_correct.md", PAPER / "b77_checks.md"
REG = "ces_prediction/experiments/PREREGISTRATION_B77.md"
ADD = "ces_prediction/experiments/PREREGISTRATION_B77_ADDENDUM1.md"

SEED = 20261177
N_BOOT = 20_000
RULE = "D10"
HI, LO = 1.35, 0.74
QS = (0.05, 0.10, 0.20)
REGS = ("T0", "TX", "C0", "CX", "CB", "CE")
CLFS = ("PH", "PHX", "FH", "FHX")
V1_CTRL = ("CXp", "CXlag")
ROW_KEYS = K.ROW_KEYS + ("r2_k", "r3_k", "r4_k")
log = C.log
save_npz, save_json, save_text, load_npz = C.save_npz, C.save_json, C.save_text, C.load_npz


def rng_for(name):
    return np.random.default_rng([SEED, zlib.crc32(name.encode())])


def rows():
    return K.load_keys(C.CACHE / "rows.npz", ROW_KEYS)


def row_set(R, S):
    """Addendum 1 section 2: complete under D10 (B.75) and L5 defined."""
    return S["complete_D10"] & np.isfinite(R["tg_L5"])


def frame_scalars(y, I):
    """The four scalars of the frame's value (registration order): y - G, y - H, y - m10, |y - G|; m10 = th[:, 0]."""
    G, H, m10 = I["G"], I["H"], I["th"][:, 0]
    return np.column_stack([y - G, y - H, y - m10, np.abs(y - G)])


def prev_measured(ti, seg):
    """For every grid row g: the last row before g with a finite value in the same block, or -1 (block numbers are unique
    across files)."""
    n = len(ti)
    idx = np.where(np.isfinite(ti), np.arange(n), -1)
    upto = np.maximum.accumulate(idx)                     # last measured row at or before g
    prev = np.full(n, -1, np.int64)
    prev[1:] = upto[:-1]
    ok = prev >= 0
    ok[ok] = seg[prev[ok]] == seg[ok]
    return np.where(ok, prev, -1)


def med3(y, anc_g, ti, seg):
    """Median of y, the anchor row's value and the value of the measured row before the anchor row in its block."""
    p2 = prev_measured(ti, seg)[anc_g]
    if np.any(anc_g < 0) or np.any(p2 < 0):
        raise SystemExit("FATAL: MED3 lacks an earlier value on a row of the set")
    return np.median(np.column_stack([y, ti[anc_g], ti[p2]]), axis=1)


def label(y, L5):
    if not np.all(L5 > 0):
        raise SystemExit("FATAL: L5 not above zero on a row of the set")
    r = y / L5
    return ((r > HI) | (r < LO)).astype(np.int64)


class UnitInputs:
    """The learner inputs of one unit over all D1 rows, built on demand for the rows asked for (Addendum 1 section 3)."""

    def __init__(self, ctx, u, F4, lag=False):
        fast8, ms_b, ms_e = ctx.unit_arrays(u)
        self.XS, self.G = ctx.matrices(RULE, fast8, ms_b, ms_e)       # B.75 Lazy(H10, fast8, A), G under D10
        self.F4 = F4
        self.lag = None
        if lag:
            wf = ctx.FE.window(u["fit"])
            _, fastl = ctx.FE.at(wf, ctx.R["gi"])                       # B.73 TXlag: eight scalars of row t - 1
            I = K.load_inputs(RULE)
            Al = K.anchor_scalars(ctx.R["gi"] - 1, I["anc_g"], ms_b, ms_e, ctx.mr_b, ctx.mr_e)
            self.lag = np.column_stack([fastl, Al])

    def fast14(self, rows):
        return np.column_stack([self.XS.f8[rows], self.XS.A[rows]])

    def X(self, L, rows, frame=True, fast14=None):
        """Matrix of learner L on rows; fast14 replaces the fourteen fast scalars (controls); frame=False empties the
        block of the four frame scalars (check 1c)."""
        if L in ("T0", "PH"):
            return self.XS["T0"][rows]
        if L in ("TX", "PHX"):
            return self.XS["TX"][rows]
        h = self.XS.H10[rows]
        f4 = self.F4[rows] if frame else self.F4[rows][:, :0]
        if L in ("C0", "FH"):
            return np.column_stack([h, f4])
        f14 = self.fast14(rows) if fast14 is None else fast14
        if L in ("CX", "FHX", "CXp", "CXlag"):
            return np.column_stack([h, f4, f14])
        if L == "CB":
            return np.column_stack([h, f4, f14[:, :4], f14[:, 8:11]])
        if L == "CE":
            return np.column_stack([h, f4, f14[:, 4:8], f14[:, 11:14]])
        raise KeyError(L)


def units(R):
    us, tab, fifth = DS.units(R)
    return us, tab, fifth


def fit_path(u, L):
    return FITS / f"{u['name']}__{L}.npz"
