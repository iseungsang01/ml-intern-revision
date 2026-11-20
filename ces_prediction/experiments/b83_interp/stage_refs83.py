"""B.83 stage 1: the dataset-equivalent arrays and the PCHIP / offline-GP value of every D1 row (Addendum 1 sections 1-2).

    py ces_prediction/experiments/b83_interp/stage_refs83.py
Output: data/.b83/arrays.npz, data/.b83/refs.npz, data/.b83/refs_info.json.  No error of any estimator is computed here.
"""
from __future__ import annotations

import os
import sys
import time

import common83 as M

import numpy as np  # noqa: E402

C = M.C
sys.path.insert(0, str(C.REPO / "ces_prediction"))
WORKERS = 4


def pin_env():
    for k in [k for k in os.environ if k.startswith("CES_")]:
        del os.environ[k]
    os.environ.update({"CES_DATA_DIR": str(C.DATA), "CES_DROP_STUCK_TARGETS": "1", "CES_SEQ_NORM_EXPANDING": "0",
                       "CES_SEQ_ZERO_MC": "0", "CES_PER_SHOT_NORM": "0", "CES_TI_SPIKE_CUT_EV": "3000"})


def build_arrays():
    if M.ARRAYS.exists():
        M.log("arrays: cached")
        return
    pin_env()
    sys.path.insert(0, str(C.EXP / "seq"))
    from seq_data import load_grid_files
    M.log("loading seq_data.load_grid_files (held removed, cut 3000 eV)")
    g, _ = load_grid_files(C.DATA, True, C.CUT_EV)
    G = C.grid()
    names = [str(x) for x in G["names"]]
    if sorted(g) != names:
        raise SystemExit("FATAL: the file list differs from the B.73 grid")
    parts, grows, lens = [], [], []
    for i, n in enumerate(names):
        a = g[n]
        a0, a1 = G["off"][i], G["off"][i + 1]
        if len(a) != a1 - a0 or not np.array_equal(a[:, 0].astype(np.float64), G["time"][a0:a1]) or \
                not np.array_equal(a[:, 1].astype(np.float64), G["ti"][a0:a1], equal_nan=True):
            raise SystemExit(f"FATAL: {n}: the loaded grid differs from the B.73 grid")
        keep = np.isfinite(a[:, 1]) | np.isfinite(a[:, 2])          # dataset.py:138, rows with >= 1 target
        parts.append(a[keep, :3].astype(np.float32))
        grows.append(np.flatnonzero(keep))
        lens.append(int(keep.sum()))
    off = np.concatenate(([0], np.cumsum(lens)))
    M.save_npz(M.ARRAYS, names=np.array(names), off=off, arr=np.concatenate(parts), grow=np.concatenate(grows))
    M.log(f"arrays: {len(names)} files, {int(off[-1]):,} rows with a CES value")


def _init():
    from threadpoolctl import threadpool_limits
    globals()["_TP"] = threadpool_limits(1)
    sys.path.insert(0, str(C.REPO / "ces_prediction"))


def _work(job):
    name, fa, rows = job
    out = np.empty((len(rows), 7))
    for j, r in enumerate(rows):
        out[j] = M.reference_of(fa, int(r))
    return name, out


def main():
    build_arrays()
    if M.REFS.exists():
        M.log("refs: cached")
        return
    R = C.rows()
    names, arrs, grows = M.load_arrays()
    fidx, grow = R["fidx"], R["grow"]
    n = len(fidx)
    drow = np.full(n, -1, np.int64)
    for i in np.unique(fidx):
        q = np.flatnonzero(fidx == i)
        gk = grows[names[i]]
        p = np.searchsorted(gk, grow[q])
        if np.any(p >= len(gk)) or not np.array_equal(gk[np.minimum(p, len(gk) - 1)], grow[q]):
            raise SystemExit(f"FATAL: {names[i]}: a D1 row is not a row of the dataset array")
        drow[q] = p
    # the dataset rows must be those of B.70 (stage2 pairs grow with the dataset row)
    sf, sg, sr, _ = M.stage2_map(2)
    key = fidx * 1_000_000 + grow
    skey = sf * 1_000_000 + sg
    o = np.argsort(skey)
    pos = np.searchsorted(skey[o], key)
    if not np.array_equal(skey[o][np.minimum(pos, len(skey) - 1)], key) or not np.array_equal(sr[o][pos], drow):
        raise SystemExit("FATAL: the dataset rows of the rebuilt arrays are not B.70's")
    M.log(f"D1 rows {n:,}: dataset rows equal to B.70's stage2 rows")
    jobs = []
    order = []
    for i in np.unique(fidx):
        q = np.flatnonzero(fidx == i)
        jobs.append((names[i], arrs[names[i]], drow[q]))
        order.append(q)
    t0 = time.time()
    res = np.full((n, 7), np.nan)
    from multiprocessing import Pool
    with Pool(WORKERS, initializer=_init) as pool:
        done = 0
        qmap = {names[int(fidx[q[0]])]: q for q in order}
        for nm, out in pool.imap_unordered(_work, jobs, chunksize=2):
            res[qmap[nm]] = out
            done += 1
            if done % 100 == 0:
                M.log(f"  {done}/{len(jobs)} files ({time.time() - t0:.0f} s)")
    if np.any(np.isnan(res[:, 2])):
        raise SystemExit("FATAL: rows without a computed reference")
    M.save_npz(M.REFS, P=res[:, 0], O=res[:, 1], formed=res[:, 2].astype(bool), gp_fail=res[:, 3].astype(bool),
               two_pts=res[:, 4].astype(bool), has_past=res[:, 5].astype(bool), has_fut=res[:, 6].astype(bool), drow=drow)
    info = {"rows": int(n), "formed": int(res[:, 2].sum()), "not_formed": int(n - res[:, 2].sum()),
            "no_past": int(n - res[:, 5].sum()), "no_future": int(n - res[:, 6].sum()),
            "gp_fit_failed_fell_back_to_hold": int(res[:, 3].sum()), "pchip_two_points_linear": int(res[:, 4].sum()),
            "seconds": round(time.time() - t0, 1), "workers": WORKERS}
    M.save_json(M.CACHE / "refs_info.json", info)
    M.log(f"refs written: {info}")


if __name__ == "__main__":
    main()
