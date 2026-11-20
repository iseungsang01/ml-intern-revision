"""B.79 (PREREGISTRATION_B79.md with Addendum 1): shared paths, constants, the beam columns joined to the grid and the
nineteen beam scalars of a row under an availability rule.

Imports b75_availability/common75.py, which imports b73_targets/common.py (threads 2, CUDA hidden) and designs.py.
Caches under data/.b79/, written whole and then renamed.  Run every stage from the repo root with `py`.
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

import os  # noqa: E402

CACHE = C.DATA / ".b79"
CHK = CACHE / "checks"                              # join, beam facts, validity selection (shared); main-run checks
# Two runs (Addendum 1 section 13): "main" = the beam columns as delivered (the registration read literally; the main
# readout); "refit" = the session lead's validity rule (the thirteen files out of every fit and score; a sensitivity).
VARIANT = os.environ.get("B79_VARIANT", "main")
if VARIANT not in ("main", "refit"):
    raise SystemExit(f"FATAL: B79_VARIANT={VARIANT}")
REFIT_DIR = CACHE / "refit_valid"
VDIR = CACHE if VARIANT == "main" else REFIT_DIR    # beam_{rule}.npz, sets.npz, sets.json, fits/ of the run
FITS = VDIR / "fits"
VCHK = CHK if VARIANT == "main" else REFIT_DIR / "checks"   # repro, causality, fits_vs_b75 of the run
NBI = C.DATA / "nbi"
PAPER = C.PAPER
OUT_JSON, OUT_MD, CHECKS_MD = PAPER / "b79_beam_inputs.json", PAPER / "b79_beam_inputs.md", PAPER / "b79_checks.md"
REG = "ces_prediction/experiments/PREREGISTRATION_B79.md"
ADD = "ces_prediction/experiments/PREREGISTRATION_B79_ADDENDUM1.md"

SEED = 20261179
RULES = ("D10", "D650", "S")
TARGETS = ("F", "L5")
SRC = ("NB1B", "NB1C", "NB2A", "NB2B", "NB2C")      # NB1A is never on (docs/paper/nbi_files.md) and is not used
ON_MW = 0.1                                         # a source is on when its power is above 0.1 MW
JOIN_TOL = 1e-4                                     # largest stamp difference of a joined row (s)
MIN_ROWS = 5                                        # a running mean needs at least five rows
INVALID_HI_MW, INVALID_LO_MW = 5.0, -0.1            # validity rule of a file-source pair (Addendum 1 section 11)
NOTCH_NEAR = 0.095                                  # "within 100 ms after an absent step": s(t) - s(j+1) < 0.095 s
BEAM_NAMES = ("P1(t)", "P1(anchor)", "P1(t)-P1(anchor)", "P2(t)", "P2(anchor)", "P2(t)-P2(anchor)",
              "Pt(t)", "Pt(anchor)", "Pt(t)-Pt(anchor)", "mean10 Pt(t)", "mean50 Pt(t)", "mean10 Pt(anchor)",
              "mean50 Pt(anchor)", "on NB1B(t)", "on NB1C(t)", "on NB2A(t)", "on NB2B(t)", "on NB2C(t)", "V NB1B(t)")
NB = len(BEAM_NAMES)                                # 19
log = C.log
save_npz, save_json, save_text, load_npz, load_keys = C.save_npz, C.save_json, C.save_text, C.load_npz, K75.load_keys
rows_lean, grid_lean = K75.rows_lean, K75.grid_lean


def rng_for(name):
    return np.random.default_rng([SEED, zlib.crc32(name.encode())])


# ------------------------------------------------------------------------------------------ beam columns
def read_nbi(shot):
    """time, the five source powers (raw, NaN where empty) and the NB1B voltage of one NBI file, all rows."""
    import pandas as pd
    df = pd.read_csv(NBI / f"s{shot}_NBI.csv", usecols=["time"] + [s + "_P" for s in SRC] + ["NB1B_V"])
    return (df["time"].to_numpy(np.float64), df[[s + "_P" for s in SRC]].to_numpy(np.float64),
            df["NB1B_V"].to_numpy(np.float64))


def grid_row_mask(grow, fidx):
    """False on a file's first row (t = 0) and last row (t = 30 s), which are not grid rows (docs/paper/file_rules.md)."""
    last = np.concatenate((fidx[1:] != fidx[:-1], [True]))
    return (grow > 0) & ~last


def run_mean(x, seg, ok, K):
    """m(i): mean of x over rows i-k (k = 0..K-1) that lie in the block of i and are grid rows (ok), added in the order
    k = 0..K-1 and divided by their number; NaN with fewer than MIN_ROWS rows.  Ends at row i: no later row enters."""
    N = len(x)
    acc = np.zeros(N)
    cnt = np.zeros(N, np.int64)
    idx = np.arange(N)
    for k in range(K):
        src = idx - k
        m = src >= 0
        m[m] = (seg[src[m]] == seg[m]) & ok[src[m]]
        acc[m] += x[src[m]]
        cnt += m
    with np.errstate(invalid="ignore", divide="ignore"):
        out = acc / cnt
    out[cnt < MIN_ROWS] = np.nan
    return out


def beam_cols(Praw, Vraw, seg, ok):
    """Row-wise beam quantities of a stretch of grid rows (one file, or several whole files): P1, P2, Pt, the running
    means of Pt over ten and fifty rows, the five on flags, the NB1B voltage (zero while NB1B is off).  An empty power
    is zero."""
    Pz = np.where(np.isfinite(Praw), Praw, 0.0)
    P1 = Pz[:, 0] + Pz[:, 1]
    P2 = Pz[:, 2] + Pz[:, 3] + Pz[:, 4]
    Pt = P1 + P2
    on = (Pz > ON_MW).astype(np.float64)
    V = np.where(Pz[:, 0] > ON_MW, Vraw, 0.0)
    return {"P1": P1, "P2": P2, "Pt": Pt, "m10": run_mean(Pt, seg, ok, 10), "m50": run_mean(Pt, seg, ok, 50),
            "on": on, "V": V}


def beam_scalars(cols, rows, anc):
    """The nineteen beam scalars (order BEAM_NAMES) of rows t with anchor rows anc (-1: none; then the anchor scalars are
    NaN).  rows and anc index the arrays of cols."""
    n = len(rows)
    out = np.full((n, NB), np.nan)
    has = anc >= 0
    a = anc[has]
    for j, key in enumerate(("P1", "P2", "Pt")):
        x = cols[key]
        out[:, 3 * j] = x[rows]
        out[has, 3 * j + 1] = x[a]
        out[:, 3 * j + 2] = out[:, 3 * j] - out[:, 3 * j + 1]
    out[:, 9] = cols["m10"][rows]
    out[:, 10] = cols["m50"][rows]
    out[has, 11] = cols["m10"][a]
    out[has, 12] = cols["m50"][a]
    out[:, 13:18] = cols["on"][rows]
    out[:, 18] = cols["V"][rows]
    return out


def invalid_files():
    """Shots whose beam scalars are not valid under the validity rule (data/.b79/checks/validity.json)."""
    import json
    return json.loads((CHK / "validity.json").read_text(encoding="utf-8"))["invalid_files"]


def load_beam(rule):
    return np.load(VDIR / f"beam_{rule}.npz")["B"]


def load_sets():
    return load_npz(VDIR / "sets.npz")


def units(R):
    """The B.73 units (V1 r2-r4 x 5, V2 seeds 1-3 x 5, V3 fifths 2-5, V4a, V4b)."""
    return DS.units(R)


def notch_class(tm, gi, jS_g):
    """Stratum by the time since the last notch (Addendum 1): 0 = within 100 ms after an absent step (s(t) - s(j+1)
    < 0.095 s, j+1 the first row after the most recent absent step that ends at or before t in the block of t); 1 = later
    rows (an absent step earlier in the block, s(t) - s(j+1) >= 0.095 s); 2 = no absent step earlier in the block."""
    cls = np.full(len(gi), 2, np.int64)
    has = jS_g >= 0
    ts = tm[gi[has]] - tm[jS_g[has] + 1]
    if np.any(ts < 0):
        raise SystemExit("FATAL: an absent step ends after its row")
    cls[has] = np.where(ts < NOTCH_NEAR, 0, 1)
    return cls


NOTCH_NAMES = ("within 100 ms after an absent step", "later rows after an absent step", "no absent step earlier in the block")
