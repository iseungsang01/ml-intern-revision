"""B.80 (PREREGISTRATION_B80.md with Addendum 1): shared paths, constants, the stored estimates of B.75 and B.79 and
B.75's interval code (class Group) with this batch's seed.

Imports b75_availability/common75.py, which imports b73_targets/common.py (threads 2, CUDA hidden) and designs.py.
No model is fitted.  Caches under data/.b80/, written whole and then renamed.  Run from the repo root with `py`.
"""
from __future__ import annotations

import importlib.util
import sys
import types
import zlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "b75_availability"))
import common75 as K75  # noqa: E402  (sets OMP/MKL/OPENBLAS threads to 2 and hides CUDA)

import numpy as np  # noqa: E402

C = K75.C
DS = K75.DS
CACHE = C.DATA / ".b80"
FITS75 = C.DATA / ".b75" / "fits"
FITS79 = C.DATA / ".b79" / "fits"          # the main run of B.79 (beam columns as delivered); not refit_valid/
PAPER = C.PAPER
OUT_JSON, OUT_MD = PAPER / "b80_ecei_gate.json", PAPER / "b80_ecei_gate.md"
REG = "ces_prediction/experiments/PREREGISTRATION_B80.md"
ADD = "ces_prediction/experiments/PREREGISTRATION_B80_ADDENDUM1.md"
B75_JSON = PAPER / "b75_availability.json"
B75_MD = PAPER / "b75_availability.md"

SEED = 20261180
MIN_ROWS = 50                    # discharges with at least 50 grid rows carry C_E and N_E
EARLY_ROWS = 200                 # C_E early: the first 200 grid rows of the discharge
LOW_CUT, HIGH_CUT = 0.7, 0.9     # classes of C_E (registration section 3)
CLASS_NAMES = ("low", "middle", "high")
NO_CLASS = -1
MIN_STEPS = 3                    # N_E needs at least three absent steps with five rows of the block before them
PRE_ROWS = 5
MIN_SCORED = 50                  # Spearman: discharges with at least 50 scored rows
RULE_OF = {"common": "D650", "S": "S"}
DESIGNS = ("V1", "V2", "V3", "V4a", "V4b")
HALF_DESIGNS = ("V1", "V4a", "V4b")      # TB and TE exist for these only
log = C.log
save_npz, save_json, save_text, load_npz = C.save_npz, C.save_json, C.save_text, C.load_npz


def rng_for(name):
    return np.random.default_rng([SEED, zlib.crc32(name.encode())])


def load_by_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# B.75's interval code, imported unchanged; its draws use this batch's streams [20261180, crc32(name)].
R75 = load_by_path("readout75", K75.HERE / "readout.py")
R75.K = types.SimpleNamespace(**{**vars(K75), "rng_for": rng_for, "SEED": SEED})
Group, subs_of, expectation_met = R75.Group, R75.subs_of, R75.expectation_met
N_BOOT = R75.N
FORMS = R75.FORMS


def class_of(ce):
    """0 low (below 0.7), 1 middle (0.7 to below 0.9), 2 high (0.9 and above), -1 without C_E."""
    ce = np.asarray(ce, np.float64)
    out = np.full(ce.shape, NO_CLASS, np.int64)
    ok = np.isfinite(ce)
    out[ok & (ce < LOW_CUT)] = 0
    out[ok & (ce >= LOW_CUT) & (ce < HIGH_CUT)] = 1
    out[ok & (ce >= HIGH_CUT)] = 2
    return out


def est_path(unit, L, r):
    base = FITS79 if L in ("TN", "TXN") else FITS75
    return base / f"{unit}__{L}__{r}__L5.npz"


def load_est(units, L, r, n):
    """Stored estimates of learner L under rule r, target L5, over the test rows of the given units (NaN elsewhere)."""
    a = np.full(n, np.nan)
    for u in units:
        z = np.load(est_path(u["name"], L, r))
        if np.any(np.isfinite(a[z["idx"]])):
            raise SystemExit(f"FATAL: {u['name']} {L} {r}: a test row estimated twice in one split")
        a[z["idx"]] = z["est"]
    return a


ROW_KEYS = ("fidx", "gi", "y", "shot", "clus", "blk", "tg_L5", "r2_k", "r3_k", "r4_k")


def rows():
    return K75.load_keys(C.CACHE / "rows.npz", ROW_KEYS)


def sets():
    return load_npz(C.DATA / ".b75" / "sets.npz")
