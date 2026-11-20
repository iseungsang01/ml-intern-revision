"""B.83 (PREREGISTRATION_B83.md with Addendum 1): shared paths, constants and loaders.

Reuses b73_targets/common.py (paths, rows, threads 2, CUDA hidden), b73_targets/designs.py (units) and
b82_localize/boot82.py (draws, weighted quantiles).  Caches under data/.b83/.  Run every stage from the repo root.
"""
from __future__ import annotations

import sys
import zlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
for _p in (EXP / "b82_localize", EXP / "b73_targets"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import common as C  # noqa: E402  (b73_targets/common.py)

import numpy as np  # noqa: E402

CACHE = C.DATA / ".b83"
CHK = CACHE / "checks"
REFS = CACHE / "refs.npz"
ARRAYS = CACHE / "arrays.npz"
PAPER = C.PAPER
OUT_JSON, OUT_MD, CHECKS_MD = PAPER / "b83_interp.json", PAPER / "b83_interp.md", PAPER / "b83_checks.md"
REG = "ces_prediction/experiments/PREREGISTRATION_B83.md"
ADD = "ces_prediction/experiments/PREREGISTRATION_B83_ADDENDUM1.md"
SEED = 20261183
N_BOOT = C.N_BOOT
DESIGNS = ("V1", "V2", "V3", "V4a", "V4b")
DESIGN_LABEL = {"V1": "V1 random (r2-r4)", "V2": "V2 cluster-blocked (assignments 1-3)",
                "V3": "V3 rolling origin (fifths 2-5)", "V4a": "V4a chronological, fit to 31991, score from 32312",
                "V4b": "V4b chronological, fit to 32310, score from 32312"}
EST = ("H", "P", "G", "O", "T0", "TX")
EST_LABEL = {"H": "hold", "P": "PCHIP", "G": "causal GP", "O": "offline GP", "T0": "T0 (history only)",
             "TX": "TX (paper's estimator)"}
REFS_OF_PAIR = ("H", "G", "P", "O")
CHECK_NAMES = ("references", "no_look", "rows", "stored")
log = C.log
save_npz, save_json, save_text, load_npz = C.save_npz, C.save_json, C.save_text, C.load_npz


def rng_for(name):
    return np.random.default_rng([SEED, zlib.crc32(name.encode())])


def stage2_map(r=2):
    """(fidx, grow) -> dataset row of every D1 row, from data/.b70/stage2_r{r}_f{k}.npz."""
    out = []
    for k in range(C.K):
        s = np.load(C.DATA / ".b70" / f"stage2_r{r}_f{k}.npz", allow_pickle=True)
        out.append((s["fidx"].astype(np.int64), s["grow"].astype(np.int64), s["row"].astype(np.int64), np.full(len(s["row"]), k)))
    return tuple(np.concatenate([o[j] for o in out]) for j in range(4))


def load_arrays():
    """The dataset-equivalent arrays (Addendum 1 section 1): {name: float32 (rows, cols)}, and per file the grid row of
    each kept row."""
    z = np.load(ARRAYS, allow_pickle=False)
    names = [str(x) for x in z["names"]]
    off = z["off"]
    arr, grow = z["arr"], z["grow"]
    return names, {n: arr[off[i]:off[i + 1]] for i, n in enumerate(names)}, {n: grow[off[i]:off[i + 1]] for i, n in enumerate(names)}


def reference_of(fa, row):
    """PCHIP and offline GP of one row exactly as score_b70.harness (B.PREDICTORS on build_neighbor_set), plus the flags
    of Addendum 1 section 2.  fa: float32 array with columns time, CES_TI, ..."""
    import baselines_interpolation as B
    times, values, target_time = B.build_neighbor_set(fa, 0, 1, row)
    p = B.PREDICTORS["pchip"](times, values, target_time)
    g = B.PREDICTORS["gp"](times, values, target_time)
    past = times < target_time
    fut = times > target_time
    formed = bool(np.any(past) and np.any(fut))
    gp_fail = False
    two_pts = False
    if formed:
        pi, fi = np.flatnonzero(past), np.flatnonzero(fut)
        pk = pi[np.argsort(target_time - times[pi])[:B._GP_MAX_SIDE]]
        fk = fi[np.argsort(times[fi] - target_time)[:B._GP_MAX_SIDE]]
        idx = np.concatenate([pk, fk])
        if len(idx) >= 2:
            gp_fail = not np.isfinite(B._gp_fit_predict(times[idx], values[idx], target_time))
        two_pts = len(np.unique(times)) == 2
    return p, g, formed, gp_fail, two_pts, int(np.any(past)), int(np.any(fut))


def units():
    import designs as DS
    return DS.units()


def assemble(d, R, us):
    """Subs of design d (Addendum 1 section 3): [{"name", "T0", "TX"}] with full-length arrays of the stored B.73
    estimates fitted to F, and the scored mask (the same in every sub)."""
    n = len(R["y"])
    if d in ("V1", "V2"):
        subs = C.REPEATS if d == "V1" else C.V2_SEEDS
        groups = {s: [u for u in us if u["design"] == d and u["sub"] == s] for s in subs}
    elif d == "V3":
        groups = {"pooled": [u for u in us if u["design"] == "V3"]}
    else:
        groups = {d: [u for u in us if u["name"] == d]}
    out, mask = [], None
    for s, ul in groups.items():
        m = np.zeros(n, bool)
        sub = {"name": str(s)}
        for L in ("T0", "TX"):
            a = np.full(n, np.nan)
            for u in ul:
                z = C.load_npz(C.CACHE / "trees" / f"{u['name']}__{L}__F.npz")
                a[z["idx"]] = z["est"]
                m[z["idx"]] = True
            sub[L] = a
        out.append(sub)
        if mask is None:
            mask = m
        elif not np.array_equal(mask, m):
            raise SystemExit(f"FATAL: {d}: the subs score different rows")
    return out, mask


def estimates(sub, R, refs):
    """The six estimators of one sub, full length."""
    return {"H": R["hold"], "P": refs["P"], "G": R["gp"], "O": refs["O"], "T0": sub["T0"], "TX": sub["TX"]}


def checks_passed():
    import json
    for nm in CHECK_NAMES:
        p = CHK / f"{nm}.json"
        if not p.exists() or not json.loads(p.read_text(encoding="utf-8")).get("passes"):
            return False, nm
    return True, None
