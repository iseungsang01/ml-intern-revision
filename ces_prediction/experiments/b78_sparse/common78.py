"""B.78 (PREREGISTRATION_B78.md with Addendum 1): shared paths, constants, the frame index and the per-file input
computation under a rate rule Kk.

Imports b75_availability/common75.py, which imports b73_targets/common.py (threads 2, CUDA hidden) and designs.py.
Caches under data/.b78/, written whole and then renamed.  Run every stage from the repo root with `py`.
"""
from __future__ import annotations

import sys
import zlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "b75_availability"))
import common75 as K75  # noqa: E402  (sets OMP/MKL/OPENBLAS threads to 2 and hides CUDA)

import numpy as np  # noqa: E402

C = K75.C
DS = K75.DS
BI = K75.BI

CACHE = C.DATA / ".b78"
INPUTS = CACHE / "inputs"
FITS = CACHE / "fits"
CHK = CACHE / "checks"
PAPER = C.PAPER
OUT_JSON, OUT_MD, CHECKS_MD = PAPER / "b78_sparse.json", PAPER / "b78_sparse.md", PAPER / "b78_checks.md"
REG = "ces_prediction/experiments/PREREGISTRATION_B78.md"
ADD = "ces_prediction/experiments/PREREGISTRATION_B78_ADDENDUM1.md"

SEED = 20261178
KS = (1, 2, 5, 10, 20, 50)
RULES = tuple(f"K{k}" for k in KS)
KOF = {f"K{k}": k for k in KS}
TARGETS = ("F", "L5")
LAT = 0.010                       # frame spacing (s)
LAT_TOL = 0.003                   # tolerance of a stamp against its multiple of 10 ms (s)
log = C.log
save_npz, save_json, save_text, load_npz = C.save_npz, C.save_json, C.save_text, C.load_npz
rows_lean, grid_lean, load_keys = K75.rows_lean, K75.grid_lean, K75.load_keys


def rng_for(name):
    return np.random.default_rng([SEED, zlib.crc32(name.encode())])


# ------------------------------------------------------------------------------------------ frame index
def frame_index(G):
    """Frame index of every grid row (global arrays): round((s - s0) / 10 ms), s0 = the stamp of the first row of the
    block that is not a file's first row.  A file's first row (t = 0) and last row (t = 30 s) are not grid rows
    (docs/paper/file_rules.md) and get -1; so does a row whose stamp lies more than 3 ms from s0 + 10 ms x index.
    Returns (index, number of rows other than first/last rows that are off the lattice)."""
    tm, seg, grow, fid = G["time"], G["seg"], G["grow"], G["fidx"]
    N = len(tm)
    newblk = np.concatenate(([True], seg[1:] != seg[:-1]))
    start = newblk.copy()
    s0 = np.flatnonzero(newblk & (grow == 0))
    s0 = s0[s0 + 1 < N]
    s0 = s0[seg[s0 + 1] == seg[s0]]               # a block that begins with a file's first row
    start[s0] = False
    start[s0 + 1] = True
    first = np.maximum.accumulate(np.where(start, np.arange(N), 0))
    d = tm - tm[first]
    f = np.rint(d / LAT).astype(np.int64)
    off_lat = np.abs(d - LAT * f) > LAT_TOL
    last_row = np.concatenate((fid[1:] != fid[:-1], [True]))
    not_grid = (grow == 0) | last_row
    f[not_grid | off_lat] = -1
    return f, int((off_lat & ~not_grid).sum())


def scored_mask(fr, gi):
    """D1 rows whose frame index is odd and not a multiple of 5 (registration section 3)."""
    f = fr[gi]
    return (f >= 0) & (f % 2 == 1) & (f % 5 != 0)


def load_frames():
    return load_npz(CACHE / "frames.npz")


# ------------------------------------------------------------------------------------------ inputs under Kk
def file_inputs_k(tm, ti, seg, fr, rows, k):
    """Inputs under rule Kk of the given file-local rows (arrays of one file; fr = frame index of the file's rows).
    Visible: the measured values (finite ti) of the rows of the block of t that lie before t and whose frame index is
    a multiple of k.  Same operations, in the same order, as common75.file_inputs, so that K1 gives D10 bit for bit.
    Never reads whether row t itself carries a value.  Returns H, age (ms), M5, G, th (n x 5), anc (file-local anchor
    row or -1), nvis (visible values in the block)."""
    obs = np.flatnonzero(np.isfinite(ti))
    kept = obs[(fr[obs] >= 0) & (fr[obs] % k == 0)]
    n_all = len(tm)
    newblk = np.concatenate(([True], seg[1:] != seg[:-1]))
    first = np.maximum.accumulate(np.where(newblk, np.arange(n_all), 0))
    n = len(rows)
    H, age, M5, G = (np.full(n, np.nan) for _ in range(4))
    th = np.full((n, 5), np.nan)
    anc = np.full(n, -1, np.int64)
    nvis = np.zeros(n, np.int64)
    p_all = np.searchsorted(kept, rows, side="left")
    b_all = np.searchsorted(kept, first[rows], side="left")
    for q in range(n):
        t = int(rows[q])
        p, b = int(p_all[q]), int(b_all[q])
        nv = p - b
        nvis[q] = max(nv, 0)
        if nv <= 0:
            continue
        last = kept[p - 1]
        if last >= t:
            raise SystemExit(f"FATAL: a visible value at or after row {t} under K{k}")
        H[q] = ti[last]
        age[q] = (tm[t] - tm[last]) * 1000.0
        anc[q] = last
        v = ti[kept[max(p - 10, b):p]]
        M5[q] = np.median(v[-5:])
        if len(v) >= 3:
            th[q] = (v.mean(), v.std(), v[-1] - v.mean(), v[-1] - v[-2], len(v))
        g16 = kept[max(p - 16, b):p]
        G[q] = BI.predict_gp_causal(tm[g16], ti[g16], tm[t])
    return {"H": H, "age": age, "M5": M5, "G": G, "th": th, "anc": anc, "nvis": nvis}


def load_inputs(rule):
    return load_npz(INPUTS / f"{rule}.npz")


history_matrix = K75.history_matrix
anchor_scalars = K75.anchor_scalars
zmeans, raw_means, run_mean10, valid8 = K75.zmeans, K75.raw_means, K75.run_mean10, K75.valid8


def units(R):
    """The B.73 units (V1 r2-r4 x 5, V2 seeds 1-3 x 5, V3 fifths 2-5, V4a, V4b)."""
    return DS.units(R)
