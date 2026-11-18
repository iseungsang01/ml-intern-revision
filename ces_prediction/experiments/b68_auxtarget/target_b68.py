"""B.68 (PREREGISTRATION_B68.md section 3): the auxiliary target, the offline-GP estimate of T_i at every row.

The estimator is that of the offline-GP reference of B.39, unchanged: seq/eval_seq.py scores it as
`baselines_interpolation.PREDICTORS["gp"](*build_neighbor_set(fa, time_col, ti_col, row))` on the harness's file
array `fa` (KSTAR_CES_Dataset: the rows of a discharge with at least one measured target after the fit-failure cut
and the held rule).  The neighbour set is the measured T_i values of the contiguous block of `fa` (time steps below
0.5 s) without the row itself, the 16 nearest on each side; a Matern-3/2 GP with the grid-selected length scale and
noise gives the value; without a later measured value it falls back to the last earlier one; without an earlier one
it gives no value (NaN).

Here it is applied to every row of the full grid of seq_data.load_grid_files (the rows the network is trained on):

  * a row of the harness's array (at least one measured target) is evaluated on that array exactly as eval_seq
    does, so where T_i is measured the row's own value is withheld by build_neighbor_set;
  * a row with no measured target is not in the harness's array; it is inserted into it at its time (its targets
    NaN) and the estimator is evaluated at that position.  The contiguous block is therefore always the harness's
    block rule applied to the harness's rows (plus the row asked about), never the full grid's blocks.

Nothing depends on the fold: the target of a discharge is computed from that discharge's own values only, once, and
cached in data/.b68_aux/ (build_aux_b68.py).  The cache holds physical units (eV); train_b68.py standardizes it with
the target statistics of each run.  numpy only (no torch), so the cache build workers stay light.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SEQ_DIR = HERE.parent / "seq"
CES_DIR = HERE.parents[1]
REPO = HERE.parents[2]
for _p in (str(CES_DIR), str(SEQ_DIR)):      # seq/ ends up first, as train_seq / eval_seq put it
    if _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)

import baselines_interpolation as B  # noqa: E402

DATA = REPO / "data"
CACHE_DIR = DATA / ".b68_aux"
CACHE_NPZ = CACHE_DIR / "aux_ti_offline_gp.npz"
CACHE_META = CACHE_DIR / "meta.json"
SPIKE_CUT_EV = 3000.0          # cut population (CES_TI_SPIKE_CUT_EV of run_b39.pooled_env)
DROP_STUCK = True              # held-free (CES_DROP_STUCK_TARGETS of run_b39.pooled_env)
METHOD = "gp"                  # the offline-GP reference of B.39 (eval_seq methods list)
TIME_COL, TI_COL, VT_COL = 0, 1, 2   # seq_data.load_grid_files layout: [time, CES_TI, CES_VT, fast...]
KIND_TI, KIND_VT_ONLY, KIND_INSERTED = 0, 1, 2
KIND_LABEL = {KIND_TI: "T_i measured (own value withheld)", KIND_VT_ONLY: "V_rot only measured",
              KIND_INSERTED: "no target measured (row inserted into the harness array)"}
SOURCES_HASHED = ("baselines_interpolation.py", "experiments/seq/seq_data.py", "dataset.py")


def harness_rows(arr):
    """Grid rows that form the harness's file array: at least one measured target (after cut and held rule)."""
    return np.flatnonzero(np.isfinite(arr[:, TI_COL]) | np.isfinite(arr[:, VT_COL]))


def _query(arr, lab, base, pos_of, i):
    """(sub-array, position of row i in it) on which the estimator is evaluated for grid row i."""
    j = pos_of.get(int(i))
    if j is not None:
        return base, j
    p = int(np.searchsorted(lab, i))
    return np.insert(base, p, arr[i, :3], axis=0), p


def aux_for_file(arr):
    """arr (n, >=3) float32 grid rows of one discharge -> dict of per-row arrays:
    aux (float64 eV, NaN = no value), kind (int8), past / future (a measured T_i before / after in the block)."""
    arr = np.asarray(arr)
    n = arr.shape[0]
    t = arr[:, TIME_COL].astype(np.float64)
    if n > 1 and np.any(np.diff(t) <= 0):
        raise SystemExit("FATAL: grid times are not strictly increasing")
    lab = harness_rows(arr)
    base = arr[lab, :3]
    pos_of = {int(i): j for j, i in enumerate(lab)}
    aux = np.full(n, np.nan)
    past = np.zeros(n, bool)
    future = np.zeros(n, bool)
    for i in range(n):
        sub, pos = _query(arr, lab, base, pos_of, i)
        times, values, tt = B.build_neighbor_set(sub, TIME_COL, TI_COL, pos)
        past[i] = bool(np.any(times < tt))
        future[i] = bool(np.any(times > tt))
        aux[i] = B.PREDICTORS[METHOD](times, values, tt)
    kind = np.where(np.isfinite(arr[:, TI_COL]), KIND_TI,
                    np.where(np.isfinite(arr[:, VT_COL]), KIND_VT_ONLY, KIND_INSERTED)).astype(np.int8)
    return {"aux": aux, "kind": kind, "past": past, "future": future}


def aux_at(arr, i):
    """The auxiliary target of grid row i of one discharge (the same evaluation as aux_for_file, one row)."""
    arr = np.asarray(arr)
    lab = harness_rows(arr)
    base = arr[lab, :3]
    pos_of = {int(r): j for j, r in enumerate(lab)}
    sub, pos = _query(arr, lab, base, pos_of, i)
    return float(B.PREDICTORS[METHOD](*B.build_neighbor_set(sub, TIME_COL, TI_COL, pos)))


def grid_blocks(times):
    """Row ranges [a, b) of seq_data.build_blocks (split at a time step >= 0.5 s or <= 0; blocks of < 2 rows dropped)."""
    t = np.asarray(times, dtype=np.float64)
    from dataset import STUCK_GAP_SECONDS
    bounds = [0] + [i for i in range(1, len(t)) if (t[i] - t[i - 1]) >= STUCK_GAP_SECONDS or (t[i] - t[i - 1]) <= 0] \
        + [len(t)]
    return [(a, b) for a, b in zip(bounds[:-1], bounds[1:]) if b - a >= 2]


def source_hashes():
    return {s: hashlib.sha256((CES_DIR / s).read_bytes()).hexdigest() for s in SOURCES_HASHED}


def estimator_constants():
    return {"method": METHOD, "max_side": B._GP_MAX_SIDE, "length_scales_s": list(B._GP_LS_GRID),
            "noise_variances": list(B._GP_NOISE_GRID), "jitter": B._GP_JITTER, "block_gap_s": B.GAP_SECONDS}


# ------------------------------------------------------------------------------ cache
def save_cache(results, meta):
    """results: name -> (time float32 (n,), ti float32 (n,), aux_for_file dict)."""
    names = sorted(results)
    off = np.cumsum([0] + [len(results[n][0]) for n in names]).astype(np.int64)
    cat = lambda f: np.concatenate([f(results[n]) for n in names])          # noqa: E731
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = CACHE_NPZ.with_suffix(".tmp.npz")
    np.savez(tmp, names=np.asarray(names), offsets=off, time=cat(lambda r: r[0]), ti=cat(lambda r: r[1]),
             aux=cat(lambda r: r[2]["aux"]), kind=cat(lambda r: r[2]["kind"]), past=cat(lambda r: r[2]["past"]),
             future=cat(lambda r: r[2]["future"]))
    os.replace(tmp, CACHE_NPZ)
    CACHE_META.write_text(json.dumps(meta, indent=1), encoding="utf-8")


def load_cache():
    """name -> {time, ti, aux, kind, past, future}; meta.  Refuses a cache of another data treatment or estimator."""
    if not (CACHE_NPZ.exists() and CACHE_META.exists()):
        raise SystemExit(f"FATAL: the B.68 auxiliary-target cache {CACHE_NPZ} is missing; run "
                         "py ces_prediction/experiments/b68_auxtarget/build_aux_b68.py")
    meta = json.loads(CACHE_META.read_text(encoding="utf-8"))
    if (float(meta["spike_cut_ev"]) != SPIKE_CUT_EV or bool(meta["drop_stuck"]) != DROP_STUCK
            or meta["estimator"] != estimator_constants()):
        raise SystemExit(f"FATAL: {CACHE_META} was built for another treatment or estimator: {meta}")
    z = np.load(CACHE_NPZ)
    off = z["offsets"]
    cols = ("time", "ti", "aux", "kind", "past", "future")
    data = {q: z[q] for q in cols}
    out = {}
    for j, name in enumerate(z["names"]):
        a, b = int(off[j]), int(off[j + 1])
        out[str(name)] = {q: data[q][a:b] for q in cols}
    return out, meta


def verify_against_grid(entry, arr, name):
    """The cache entry of a discharge is on exactly the rows and T_i values of this grid array."""
    if not np.array_equal(entry["time"], arr[:, TIME_COL]):
        raise SystemExit(f"FATAL: {name}: the B.68 cache times differ from the grid rows")
    if not np.array_equal(entry["ti"], arr[:, TI_COL], equal_nan=True):
        raise SystemExit(f"FATAL: {name}: the B.68 cache T_i values differ from the grid (another data treatment?)")
