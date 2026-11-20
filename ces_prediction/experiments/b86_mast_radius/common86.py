"""B.86 (PREREGISTRATION_B86.md with Addendum 1): shared paths, constants, the per-shot row build for the five radius
bands, loaders.

Reuses b76_mast_availability/common76.py by import (prepare_shot, chord_history, thomson_grids, hist_matrix, ts_matrix,
meta_tables, session_of, save/load helpers).  common76.build_shot has the pedestal-top band hard-coded and builds six
rules and the D-alpha inputs; build_shot86 below re-implements only that function with the band union as a parameter,
rules D5 and D100, no D-alpha, and stores the chord series of the block for the references that wait.
Caches under data/.b86/.  Run every stage from the repo root with `py`.
"""
from __future__ import annotations

import json
import sys
import time as _time
import warnings
import zlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
sys.path.insert(0, str(EXP / "b76_mast_availability"))
import common76 as K  # noqa: E402  (sets OMP/MKL/OPENBLAS threads to 2, hides CUDA)

import numpy as np  # noqa: E402

REPO = K.REPO
DATA = K.DATA
CACHE = DATA / ".b86"
SHOTS = CACHE / "shots"
FITS = CACHE / "fits"
CHK = CACHE / "checks"
PAPER = K.PAPER
OUT_JSON, OUT_MD, CHECKS_MD = PAPER / "b86_mast_radius.json", PAPER / "b86_mast_radius.md", PAPER / "b86_checks.md"
REG = "ces_prediction/experiments/PREREGISTRATION_B86.md"
ADD = "ces_prediction/experiments/PREREGISTRATION_B86_ADDENDUM1.md"

SEED = 20261186
TREE_SEED = K.TREE_SEED
N_BOOT = 20_000
STEP = K.STEP
TOL_MS = K.TOL_MS
RULES = ("D5", "D100")
DELAYS = {r: K.DELAYS[r] for r in RULES}
TARGETS = ("F", "L5")
BANDS = {"E": (0.015, 0.05), "B2": (0.05, 0.10), "B3": (0.10, 0.20), "B4": (0.20, 0.35), "B5": (0.35, 0.60)}
BAND_NAMES = tuple(BANDS)
UNION = (0.015, 0.60)
MIN_ROWS = 5000
NF = 5
M1_SEED = 1
ROW_KEYS = (["shot", "c", "craw", "R", "rho", "j", "t", "y", "sig", "prevgood", "te_t", "ne_t", "dedge_t", "dband", "nts",
             "nlas", "L5", "nL5"]
            + [f"{r}_{k}" for r in RULES for k in ("H", "age", "M5", "G", "th", "anc", "nvis", "te_a", "ne_a", "nts_a")])
T0_ = _time.time()
log = K.log
save_npz, save_json, save_text, load_npz = K.save_npz, K.save_json, K.save_text, K.load_npz
warnings.filterwarnings("ignore")


def rng_for(name):
    return np.random.default_rng([SEED, zlib.crc32(name.encode())])


def band_of(dband):
    """Band label index (0..4) of each band distance, -1 outside every band."""
    out = np.full(len(dband), -1, np.int8)
    for k, (lo, hi) in enumerate(BANDS.values()):
        out[(dband >= lo) & (dband < hi)] = k
    return out


# ------------------------------------------------------------------------------------------ per-shot row build
def build_shot86(shot, cat):
    """common76.build_shot with the band union [1.5, 60) cm, rules D5 and D100, no D-alpha; plus the chord series of the
    block (Y, Gm) for the references.  Every expression for a row is that of common76.build_shot."""
    st, P = K.prepare_shot(shot, cat)
    if st != "ok":
        return {"status": np.array(st)}
    d, p, a, chords, ex, (ax, ro) = P["d"], P["p"], P["a"], P["chords"], P["ex"], P["geo"]
    L, G = p["L"], p["G"]
    y = d["T"][a:a + L][:, chords].astype(np.float64)
    sig = d["Te"][a:a + L][:, chords].astype(np.float64)
    Rc = d["R"][chords].astype(np.float64)
    rho = (Rc - ax) / (ro - ax)
    t_a = float(d["time_ms"][a])
    tf = STEP * np.arange(L)
    tf_abs = t_a + tf
    TEw, NEw, dedge, dband, nts, nlas, sysname = K.thomson_grids(ex, Rc, tf_abs, ax)
    band = (dband >= UNION[0]) & (dband < UNION[1])
    ts_t = np.isfinite(TEw) & np.isfinite(NEw)
    cand = G & band & ts_t & np.isfinite(dedge)
    counts = np.zeros((len(BANDS), 4), np.int64)        # per band: good, in band, with Thomson at t, candidate
    for k, (lo, hi) in enumerate(BANDS.values()):
        inb = (dband >= lo) & (dband < hi)
        counts[k] = [int(G.sum()), int((G & inb).sum()), int((G & inb & ts_t).sum()), int((cand & inb).sum())]
    jj, cc = np.nonzero(cand)
    o = np.lexsort((jj, cc))           # by chord, then frame (as common76.build_shot)
    jj, cc = jj[o], cc[o]
    n = len(jj)
    out = {"status": np.array("ok"), "counts": counts, "L": np.int64(L), "t_a": np.float64(t_a),
           "ts_sys": np.array(sysname), "n_chords": np.int64(len(chords)), "Y": y, "Gm": G, "Rc": Rc,
           "shot": np.full(n, shot, np.int32), "c": cc.astype(np.int16), "craw": chords[cc].astype(np.int16),
           "R": Rc[cc], "rho": rho[cc], "j": jj.astype(np.int32), "t": tf_abs[jj], "y": y[jj, cc], "sig": sig[jj, cc],
           "prevgood": np.where(jj >= 1, G[np.maximum(jj - 1, 0), cc], False),
           "te_t": TEw[jj, cc], "ne_t": NEw[jj, cc], "dedge_t": dedge[jj, cc], "dband": dband[jj, cc],
           "nts": nts[jj, cc].astype(np.int16), "nlas": nlas[jj].astype(np.int16)}
    out["band"] = band_of(out["dband"])
    later = np.full((n, 5), np.nan)
    for k in range(1, 6):
        jk = jj + k
        ok = jk < L
        jkc = np.minimum(jk, L - 1)
        gk = ok & G[jkc, cc]
        later[gk, k - 1] = y[jkc[gk], cc[gk]]
    nl = np.isfinite(later).sum(1)
    with np.errstate(all="ignore"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            l5 = np.nanmedian(later, 1)
    out["L5"] = np.where(nl >= 3, l5, np.nan)
    out["nL5"] = nl.astype(np.int8)
    for r, dl in DELAYS.items():
        H, age, M5, Gp, anc, nvis = (np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan),
                                     np.full(n, -1, np.int64), np.zeros(n, np.int64))
        th = np.full((n, 5), np.nan)
        for c in np.unique(cc):
            q = np.flatnonzero(cc == c)
            I = K.chord_history(tf, y[:, c], G[:, c], jj[q], dl)
            H[q], age[q], M5[q], Gp[q], th[q], anc[q], nvis[q] = I["H"], I["age"], I["M5"], I["G"], I["th"], I["anc"], I["nvis"]
        ok = anc >= 0
        ac = np.maximum(anc, 0)
        out[f"{r}_H"], out[f"{r}_age"], out[f"{r}_M5"], out[f"{r}_G"], out[f"{r}_th"] = H, age, M5, Gp, th
        out[f"{r}_anc"], out[f"{r}_nvis"] = anc.astype(np.int32), nvis.astype(np.int32)
        out[f"{r}_te_a"] = np.where(ok, TEw[ac, cc], np.nan)
        out[f"{r}_ne_a"] = np.where(ok, NEw[ac, cc], np.nan)
        out[f"{r}_nts_a"] = np.where(ok, nts[ac, cc], 0).astype(np.int16)
    return out


def complete_mask(R, r):
    """Complete under a rule (B.76 Addendum 1 section 5): ten history scalars and G finite, nine Thomson scalars finite."""
    H = K.hist_matrix(R, r)
    h_ok = np.isfinite(H).all(1) & np.isfinite(R[f"{r}_G"])
    T = K.ts_matrix(R, r)
    return h_ok & np.isfinite(T).all(1)


# ------------------------------------------------------------------------------------------ loaders
def band_path(b):
    return CACHE / f"rows_{b}.npz"


def load_band(b, keys=None):
    z = np.load(band_path(b), allow_pickle=False)
    return {k: z[k] for k in (keys or z.files)}


def units(b):
    u = json.loads((CACHE / f"units_{b}.json").read_text(encoding="utf-8"))
    return u["units"], u["table"]


def sets_info():
    return json.loads((CACHE / "sets.json").read_text(encoding="utf-8"))


def bands_with_statements():
    si = sets_info()
    return [b for b in BAND_NAMES if si["bands"][b]["common_L5"]["rows"] >= MIN_ROWS]


def fit_path(b, unit, L, r, X):
    return FITS / b / f"{unit}__{L}__{r}__{X}.npz"


def plan_of(u):
    """(learner, rule, target) of one unit (Addendum 1 section 6)."""
    out = [(L, r, "L5") for r in RULES for L in ("T0", "TX")]
    if u["design"] == "M1":
        out += [(L, "D5", "F") for L in ("T0", "TX")]
    return out


def checks_passed():
    for nm in ("band_e", "visibility", "references", "rows"):
        p = CHK / f"{nm}.json"
        if not p.exists() or not json.loads(p.read_text(encoding="utf-8")).get("passes"):
            return False, nm
    return True, None
