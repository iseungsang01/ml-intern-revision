"""B.75 (PREREGISTRATION_B75.md with Addendum 1): shared paths, constants and the per-file input computation under an
availability rule.

Imports b73_targets/common.py (threads, CUDA off, grid, rows, helpers) and designs.py.  Caches under data/.b75/, written
whole and then renamed.  Run every stage from the repo root with `py`.
"""
from __future__ import annotations

import sys
import zlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
B73 = HERE.parent / "b73_targets"
sys.path.insert(0, str(B73))
import common as C  # noqa: E402  (sets OMP/MKL/OPENBLAS threads to 2 and hides CUDA)
import designs as DS  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(1, str(C.REPO / "ces_prediction"))
import baselines_interpolation as BI  # noqa: E402

CACHE = C.DATA / ".b75"
INPUTS = CACHE / "inputs"
FITS = CACHE / "fits"
LOGS = CACHE / "logs"
PAPER = C.PAPER
OUT_JSON, OUT_MD, CHECKS_MD = PAPER / "b75_availability.json", PAPER / "b75_availability.md", PAPER / "b75_checks.md"
REG = "ces_prediction/experiments/PREREGISTRATION_B75.md"
ADD = "ces_prediction/experiments/PREREGISTRATION_B75_ADDENDUM1.md"

SEED = 20261175
TOL = 0.003
DELAYS = {"D10": 0.010, "D50": 0.050, "D100": 0.100, "D300": 0.300, "D650": 0.650, "D1000": 1.000}
FIXED = tuple(DELAYS)
RULES = FIXED + ("S",)
AB_LO, AB_HI = 0.015, 0.045
TARGETS = ("F", "L5")
V3C_SEEDS = (1, 2, 3)
V3C_FIFTHS = (1, 2, 3)          # unit index q of V3_q (fifths 2, 3, 4)
log = C.log
save_npz, save_json, save_text, load_npz = C.save_npz, C.save_json, C.save_text, C.load_npz


def rng_for(name):
    return np.random.default_rng([SEED, zlib.crc32(name.encode())])


def load_keys(path, keys):
    z = np.load(path, allow_pickle=False)
    return {k: z[k] for k in keys}


ROW_KEYS = ("fidx", "grow", "gi", "y", "shot", "clus", "blk", "tstep", "tg_F", "tg_L5")


def rows_lean():
    return load_keys(C.CACHE / "rows.npz", ROW_KEYS)


def grid_lean(keys=("names", "off", "time", "ti", "seg", "fidx", "grow")):
    return load_keys(C.CACHE / "grid.npz", keys)


# ------------------------------------------------------------------------------------------ availability
def s_limit_rows(tm, seg):
    """File-local: for each row t, the index j of the last row before the most recent absent step (j, j+1) of the block
    of t with j + 1 <= t; -1 when there is none."""
    n = len(tm)
    dt = np.diff(tm)
    ab = (seg[1:] == seg[:-1]) & (dt > AB_LO) & (dt < AB_HI)          # absent step between j and j + 1 (index j)
    mark = np.full(n, -1, np.int64)
    j = np.flatnonzero(ab)
    mark[j + 1] = j
    last = np.maximum.accumulate(mark)
    ok = (last >= 0) & (seg[np.maximum(last, 0)] == seg)
    return np.where(ok, last, -1)


def limit_stamps(tm, seg, rows, rule):
    """Visibility limit (stamp) of each row; -inf when nothing is visible (S without an absent step).  Also the S row j
    (or -1) for the age of the limit."""
    if rule in DELAYS:
        return tm[rows] - DELAYS[rule], np.full(len(rows), -1, np.int64)
    jr = s_limit_rows(tm, seg)[rows]
    lim = np.where(jr >= 0, tm[np.maximum(jr, 0)], -np.inf)
    return lim, jr


def file_inputs(tm, ti, seg, rows, rule):
    """Inputs under `rule` of the given file-local rows (arrays of one file).  Returns dict of arrays over rows:
    H, age (ms), M5, G, th (n x 5), anc (file-local anchor row or -1), nvis (visible values in the block), lim (limit stamp),
    jS (S limit row or -1).  Never reads whether row t itself carries a value."""
    obs = np.flatnonzero(np.isfinite(ti))
    tobs = tm[obs]
    lim, jr = limit_stamps(tm, seg, rows, rule)
    # first row of each row's block
    n_all = len(tm)
    newblk = np.concatenate(([True], seg[1:] != seg[:-1]))
    first = np.maximum.accumulate(np.where(newblk, np.arange(n_all), 0))
    n = len(rows)
    H, age, M5, G = (np.full(n, np.nan) for _ in range(4))
    th = np.full((n, 5), np.nan)
    anc = np.full(n, -1, np.int64)
    nvis = np.zeros(n, np.int64)
    p_all = np.searchsorted(tobs, lim + TOL, side="right")
    b_all = np.searchsorted(obs, first[rows], side="left")
    for q in range(n):
        t = int(rows[q])
        p, b = int(p_all[q]), int(b_all[q])
        # visible values must lie before row t (true for every rule; asserted)
        if p > b and obs[p - 1] >= t:
            raise SystemExit(f"FATAL: a visible value at or after row {t} under {rule}")
        nv = p - b
        nvis[q] = max(nv, 0)
        if nv <= 0:
            continue
        last = obs[p - 1]
        H[q] = ti[last]
        age[q] = (tm[t] - tm[last]) * 1000.0
        anc[q] = last
        v = ti[obs[max(p - 10, b):p]]
        M5[q] = np.median(v[-5:])
        if len(v) >= 3:
            th[q] = (v.mean(), v.std(), v[-1] - v.mean(), v[-1] - v[-2], len(v))
        g16 = obs[max(p - 16, b):p]
        G[q] = BI.predict_gp_causal(tm[g16], ti[g16], tm[t])
    return {"H": H, "age": age, "M5": M5, "G": G, "th": th, "anc": anc, "nvis": nvis, "lim": lim, "jS": jr}


def history_matrix(tstep, I):
    """The ten history scalars in B.73's order."""
    return np.column_stack([tstep, I["age"], I["H"], I["G"] - I["H"], np.abs(I["G"] - I["H"]), I["th"]])


# ------------------------------------------------------------------------------------------ fast-channel means
def seq_mean(cols):
    acc = cols[:, 0].copy()
    for c in range(1, cols.shape[1]):
        acc += cols[:, c]
    acc /= cols.shape[1]
    return acc


def zmeans(FE, fit_files):
    """The standardised BES and ECEI array means exactly as stage_trees.Feats.window forms them."""
    mu, sd = FE.stats(fit_files)
    z = (FE.fast.astype(np.float64) - mu) / sd
    bes = z[:, 0].copy()
    for c in range(1, C.NB):
        bes += z[:, c]
    bes /= C.NB
    ecei = z[:, C.NB].copy()
    for c in range(C.NB + 1, C.NB + C.NE):
        ecei += z[:, c]
    ecei /= C.NE
    return bes, ecei


def raw_means(fast):
    x = fast.astype(np.float64)
    return seq_mean(x[:, :C.NB]), seq_mean(x[:, C.NB:C.NB + C.NE])


def run_mean10(x, seg):
    """m(i): mean over rows i-k (k = 0..9) of the block of i, added in the order k = 0..9; NaN with fewer than five."""
    N = len(x)
    acc = np.zeros(N)
    cnt = np.zeros(N, np.int64)
    idx = np.arange(N)
    for k in range(10):
        src = idx - k
        ok = src >= 0
        ok[ok] = seg[src[ok]] == seg[ok]
        acc[ok] += x[src[ok]]
        cnt += ok
    with np.errstate(invalid="ignore", divide="ignore"):
        m = acc / cnt
    m[cnt < 5] = np.nan
    return m


def anchor_scalars(gi, anc_g, mstd_b, mstd_e, mraw_b, mraw_e):
    """Six anchor scalars (a1, a2, a3 of BES, then of ECEI); anc_g = global anchor row or -1."""
    n = len(gi)
    out = np.full((n, 6), np.nan)
    ok = anc_g >= 0
    a = anc_g[ok]
    g = gi[ok]
    for j, (ms, mr) in enumerate(((mstd_b, mraw_b), (mstd_e, mraw_e))):
        out[ok, 3 * j] = ms[a]
        out[ok, 3 * j + 1] = ms[g] - ms[a]
        rt, ra = mr[g], mr[a]
        pos = (rt > 0) & (ra > 0)
        with np.errstate(invalid="ignore", divide="ignore"):
            out[ok, 3 * j + 2] = np.where(pos, np.log(np.where(pos, rt, 1.0) / np.where(pos, ra, 1.0)), np.nan)
    return out


def valid8(G, gi):
    """Rows where the eight fast scalars exist (B.73 window_feats: row t and row t - 10 in the same block)."""
    grow, seg = G["grow"], G["seg"]
    return (grow[gi] >= 10) & (seg[gi] == seg[np.maximum(gi - 10, 0)])


def load_inputs(rule):
    return load_npz(INPUTS / f"{rule}.npz")


def units_all(R):
    """B.73 units plus the V3c units (Addendum 1 section 5)."""
    us, tab, fifth = DS.units(R)
    u = tab["fidx"]
    extra = []
    for q in V3C_FIFTHS:
        base = next(x for x in us if x["name"] == f"V3_q{q}")
        outside = np.sort(u[fifth != q])
        for s in V3C_SEEDS:
            fit = np.sort(np.random.default_rng(s).choice(outside, len(base["fit"]), replace=False))
            extra.append({"name": f"V3c_q{q}_s{s}", "design": "V3c", "sub": s, "fifth": q, "fit": fit, "test": base["test"]})
    return us + extra, tab, fifth
