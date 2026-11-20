"""N3 (unregistered, after the fact; feasibility only): does a burst of mid-plane D-alpha light
inside a MAST CXRS exposure go with a one-frame rise of the fitted edge T_i, a rise of the
reported T_i error, and no change of the fitted velocity -- the MAST analogue of B.74 (KSTAR)?

Reads data/mast/act/<shot>.npz (act_ss_temperature, _error, act_ss_velocity, act_time,
act_majorradius) and data/mast/dalpha/<shot>.npz (50 kHz mid-plane D-alpha, HM10 wide view).
Frames are 5 ms; the exposure convention is not in the metadata, so two windows are read:
E0 = [t, t+5) ms and EC = [t-2.5, t+2.5) ms.
Edge channels: chords with R within 6 cm inside the outermost chord that has a finite T_i in at
least 30 % of the frames of the shot (not the Thomson-based pedestal-top band of the project).
Per frame and edge chord: ratio of T_i to the median of the five preceding frames of that chord;
following-level ratio = median of frames t+1..t+5 over that same median; reported-error ratio
likewise; velocity difference (km/s) to the median of the five preceding frames.
Frame class from D-alpha: burst B if the exposure mean exceeds 1.5 x the median of the 50 ms
before the exposure, quiet Q if within [0.9, 1.1].
Statistics: medians per class pooled over chords and frames; 95 % interval by resampling shots
(500 draws). Shots: the first N with both files (argument, default 800). ASCII output.
"""
import glob
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np  # noqa: E402

NMAX = int(sys.argv[1]) if len(sys.argv) > 1 else 800
shots = sorted(int(os.path.basename(p)[:-4]) for p in glob.glob("data/mast/dalpha/*.npz"))
shots = [s for s in shots if os.path.exists("data/mast/act/%d.npz" % s)][:NMAX]


def win_mean(ta, da, lo, hi):
    i0 = np.searchsorted(ta, lo)
    i1 = np.searchsorted(ta, hi)
    return da[i0:i1].mean() if i1 > i0 else np.nan


def per_shot(s, mode):
    z = np.load("data/mast/act/%d.npz" % s)
    a = np.load("data/mast/dalpha/%d.npz" % s)
    t = z["act_time"].astype(float) * 1e3
    R = z["act_majorradius"].astype(float)
    T = z["act_ss_temperature"].astype(float)
    E = z["act_ss_temperature_error"].astype(float)
    V = z["act_ss_velocity"].astype(float)
    ta = a["time"].astype(float) * 1e3
    da = a["dalpha"].astype(float)
    if T.ndim != 2 or T.shape[1] != R.size or E.shape != T.shape or V.shape != T.shape:
        return None
    good = (np.isfinite(T) & (T > 0)).mean(axis=0) >= 0.3
    if not good.any():
        return None
    rout = R[good].max()
    edge = np.where(good & (R >= rout - 0.06))[0]
    out = {"B": [], "Q": []}
    for k in range(5, len(t) - 5):
        lo, hi = (t[k], t[k] + 5) if mode == "E0" else (t[k] - 2.5, t[k] + 2.5)
        m = win_mean(ta, da, lo, hi)
        ref = ta[(ta >= lo - 50) & (ta < lo)]
        if ref.size < 100 or not np.isfinite(m):
            continue
        base = np.median(da[(ta >= lo - 50) & (ta < lo)])
        if base <= 0:
            continue
        r = m / base
        cls = "B" if r > 1.5 else ("Q" if 0.9 <= r <= 1.1 else None)
        if cls is None:
            continue
        for c in edge:
            prev = T[k - 5:k, c]
            prev = prev[np.isfinite(prev) & (prev > 0)]
            if prev.size < 3:
                continue
            mp = np.median(prev)
            tk = T[k, c]
            nxt = T[k + 1:k + 6, c]
            nxt = nxt[np.isfinite(nxt) & (nxt > 0)]
            pe = E[k - 5:k, c]
            pe = pe[np.isfinite(pe) & (pe > 0)]
            pv = V[k - 5:k, c]
            pv = pv[np.isfinite(pv)]
            row = [np.nan] * 5
            row[0] = 1.0 if not (np.isfinite(tk) and tk > 0) else 0.0          # missing value
            if np.isfinite(tk) and tk > 0:
                row[1] = tk / mp                                                   # T ratio
                if pe.size >= 3 and np.isfinite(E[k, c]) and E[k, c] > 0:
                    row[3] = E[k, c] / np.median(pe)                               # error ratio
                if pv.size >= 3 and np.isfinite(V[k, c]):
                    row[4] = (V[k, c] - np.median(pv)) / 1e3                       # dV km/s
            if nxt.size >= 3:
                row[2] = np.median(nxt) / mp                                       # following level
            out[cls].append(row)
    return {k: np.array(v, float).reshape(-1, 5) for k, v in out.items()}


def summarise(mode):
    data = {}
    for s in shots:
        try:
            r = per_shot(s, mode)
        except Exception:  # noqa: BLE001
            r = None
        if r is not None and (len(r["B"]) or len(r["Q"])):
            data[s] = r
    keys = sorted(data)
    names = ["share missing", "T_i ratio (median)", "following level ratio (median)",
             "T_i error ratio (median)", "velocity change km/s (median)"]
    nb = sum(len(data[s]["B"]) for s in keys)
    nq = sum(len(data[s]["Q"]) for s in keys)
    nsb = sum(1 for s in keys if len(data[s]["B"]))
    print("mode %s: shots %d (with burst frames %d), chord-frames B %d, Q %d" % (mode, len(keys), nsb, nb, nq))

    def stat(rows, j):
        v = rows[:, j]
        v = v[np.isfinite(v)]
        if v.size == 0:
            return np.nan
        return v.mean() if j == 0 else np.median(v)

    rng = np.random.default_rng(20261120)
    for j, nm in enumerate(names):
        def both(sel):
            B = np.concatenate([data[s]["B"] for s in sel]) if sel else np.zeros((0, 5))
            Q = np.concatenate([data[s]["Q"] for s in sel]) if sel else np.zeros((0, 5))
            return stat(B, j), stat(Q, j)
        b, q = both(keys)
        bs = []
        for _ in range(500):
            sel = [keys[i] for i in rng.integers(0, len(keys), len(keys))]
            bb, qq = both(sel)
            bs.append(bb - qq)
        lo, hi = np.nanpercentile(bs, [2.5, 97.5])
        print("  %-32s B %.3f  Q %.3f  B-Q %+.3f [%+.3f, %+.3f]" % (nm, b, q, b - q, lo, hi))
    # share of upward one-frame departures (T ratio > 1.35 and following level ratio < 1.1) in B vs all
    Ball = np.concatenate([data[s]["B"] for s in keys])
    Qall = np.concatenate([data[s]["Q"] for s in keys])
    for nm, X in (("B", Ball), ("Q", Qall)):
        okr = np.isfinite(X[:, 1]) & np.isfinite(X[:, 2])
        up = (X[okr, 1] > 1.35) & (X[okr, 2] < 1.1)
        dn = (X[okr, 1] < 0.74) & (X[okr, 2] > 0.9)
        print("  class %s: share of one-frame up-departures %.3f, down-departures %.3f (n %d)" % (nm, up.mean(), dn.mean(), okr.sum()))


for mode in ("E0", "EC"):
    summarise(mode)
