"""B.86 stage 2: the references that wait (PCHIP and offline GP, frame t left out) for every row of the F common set of
every band (Addendum 1 section 6).

    py ces_prediction/experiments/b86_mast_radius/stage_refs86.py
Output: data/.b86/refs_{band}.npz (P, O, formed, gp_fail; NaN / False off the F common set), refs_info.json.
No error of any estimator is computed here.
"""
from __future__ import annotations

import time

import common86 as C

import numpy as np  # noqa: E402

WORKERS = 4


def reference_of(tf_ms, ycol, gcol, j):
    """PCHIP and offline GP at frame j from the good values of the chord at the other block frames.
    Returns (P, O, formed, gp_fail)."""
    B = C.K.BI
    obs = np.flatnonzero(gcol)
    obs = obs[obs != j]
    times = tf_ms[obs] / 1000.0
    values = ycol[obs]
    tt = tf_ms[j] / 1000.0
    p = B.predict_pchip(times, values, tt)
    g = B.predict_gp(times, values, tt)
    past, fut = times < tt, times > tt
    formed = bool(np.any(past) and np.any(fut))
    gp_fail = False
    if formed:
        pi, fi = np.flatnonzero(past), np.flatnonzero(fut)
        pk = pi[np.argsort(tt - times[pi])[:B._GP_MAX_SIDE]]
        fk = fi[np.argsort(times[fi] - tt)[:B._GP_MAX_SIDE]]
        idx = np.concatenate([pk, fk])
        if len(idx) >= 2:
            gp_fail = not np.isfinite(B._gp_fit_predict(times[idx], values[idx], tt))
    return p, g, formed, gp_fail


def _init():
    from threadpoolctl import threadpool_limits
    globals()["_TP"] = threadpool_limits(1)


def _work(job):
    shot, cs, js = job
    z = np.load(C.SHOTS / f"{shot}.npz", allow_pickle=False)
    Y, Gm = z["Y"], z["Gm"]
    tf = C.STEP * np.arange(Y.shape[0])
    out = np.empty((len(cs), 4))
    for i, (c, j) in enumerate(zip(cs, js)):
        out[i] = reference_of(tf, Y[:, c], Gm[:, c], int(j))
    return shot, out


def main():
    from multiprocessing import Pool
    info = {}
    t_all = time.time()
    for b in C.BAND_NAMES:
        p = C.CACHE / f"refs_{b}.npz"
        if p.exists():
            C.log(f"refs {b}: cached")
            continue
        R = C.load_band(b, ["shot", "c", "j", "common_F"])
        rows = np.flatnonzero(R["common_F"])
        sh = R["shot"][rows]
        jobs, qmap = [], {}
        for s in np.unique(sh):
            q = rows[sh == s]
            qmap[int(s)] = q
            jobs.append((int(s), R["c"][q].astype(np.int64), R["j"][q].astype(np.int64)))
        n = len(R["shot"])
        res = np.full((n, 4), np.nan)
        t0 = time.time()
        with Pool(WORKERS, initializer=_init) as pool:
            done = 0
            for s, out in pool.imap_unordered(_work, jobs, chunksize=4):
                res[qmap[s]] = out
                done += 1
                if done % 500 == 0:
                    C.log(f"  {b}: {done}/{len(jobs)} shots ({time.time() - t0:.0f} s)")
        if np.any(np.isnan(res[rows, 2])):
            raise SystemExit("FATAL: rows without a computed reference")
        formed = np.zeros(n, bool)
        formed[rows] = res[rows, 2].astype(bool)
        gpf = np.zeros(n, bool)
        gpf[rows] = res[rows, 3].astype(bool)
        C.save_npz(p, P=res[:, 0], O=res[:, 1], formed=formed, gp_fail=gpf)
        info[b] = {"rows_F_common": int(len(rows)), "formed": int(formed.sum()), "not_formed": int(len(rows) - formed.sum()),
                   "gp_fit_failed_fell_back_to_hold": int(gpf.sum()), "seconds": round(time.time() - t0, 1)}
        C.log(f"refs {b}: {info[b]}")
    if info:
        old = {}
        ip = C.CACHE / "refs_info.json"
        if ip.exists():
            import json
            old = json.loads(ip.read_text(encoding="utf-8"))
        old.update(info)
        C.save_json(ip, old)
    C.log(f"refs done ({time.time() - t_all:.0f} s)")


if __name__ == "__main__":
    main()
