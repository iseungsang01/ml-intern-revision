"""Unregistered look: on quiet stretches (no absent frame within the window), what is the lag structure between
changes of the BES common mode / radial difference / ECEI level and changes of the reported T_i?
A heating-driven (beam-power) relation should make BES lead T_i by the ~40 ms response measured at blips;
a simultaneous plasma co-variation should peak at lag 0.  Real 10 ms files only.  Prints ASCII only.
"""
import sys
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

REPO = Path(r"C:\Users\lss\Documents\GitHub\ml-intern-revision")
sys.path.insert(0, str(REPO / "ces_prediction"))
import analyze_file_rules as F  # noqa: E402

BLK = 3            # rows per block (30 ms)
H = np.array([0.184,0.160,0.142,0.122,0.081,0.059,0.034,0.018,0.009,0.001]); H = H / H.sum()
LAGS = range(-4, 5)
files = sorted(F.DATA.glob("s*.csv"))
per = {q: {l: [] for l in LAGS} for q in ("c", "g", "e", "k")}
null = {q: {l: [] for l in LAGS} for q in ("c", "g", "e", "k")}
rng = np.random.default_rng(3)
nd = 0
for f in files:
    df, *_ = F.load(f)
    t = df["time"].to_numpy(float)
    if len(t) < 200:
        continue
    ti = df["CES_TI"].to_numpy(float).copy()
    ti[ti > 3000] = np.nan
    B = df[[c for c in df.columns if c.startswith("BES_")]].to_numpy(float)
    E = df[[c for c in df.columns if c.startswith("ECEI_")]].to_numpy(float).mean(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        L = np.log(np.clip(B, 1e-4, None))
    c = L.mean(1)
    cols = [c_ for c_ in df.columns if c_.startswith("BES_")]
    col07 = [i for i, n in enumerate(cols) if n.endswith("07")]
    col05 = [i for i, n in enumerate(cols) if n.endswith("05")]
    g = L[:, col07].mean(1) - L[:, col05].mean(1)
    # mask rows within 150 ms after / 30 ms before an absent frame, and segment breaks
    bad = np.zeros(len(t), bool)
    dt = np.diff(t)
    for j in np.flatnonzero(dt > 0.015):
        bad[max(0, j - 3):j + 16] = True
    # contiguous runs of good rows with 10 ms steps
    ok = ~bad
    runs, start = [], None
    for i in range(len(t)):
        if ok[i] and (start is None or abs(t[i] - t[i - 1] - 0.010) < 1e-4):
            if start is None:
                start = i
        else:
            if start is not None and i - start >= 30:
                runs.append((start, i))
            start = i if ok[i] else None
    if start is not None and len(t) - start >= 30:
        runs.append((start, len(t)))
    pairs = {q: {l: ([], []) for l in LAGS} for q in per}
    for a, b in runs:
        n = (b - a) // BLK
        if n < 8:
            continue
        sl = slice(a, a + n * BLK)
        def bm(x):
            x = x[sl].reshape(n, BLK)
            with np.errstate(invalid="ignore"):
                return np.nanmean(x, 1)
        T = bm(ti)
        X = {"c": bm(c), "g": bm(g), "e": bm(E), "k": bm(c)}
        dT = np.diff(T)
        cc = c[a:a + n * BLK]
        kp = np.convolve(cc, H)[:len(cc)]
        kp[:len(H)] = np.nan
        Tk = np.nanmean(kp.reshape(n, BLK), 1)
        for q in X:
            dX = np.diff(X[q])
            dY = np.diff(Tk) if q == "k" else dT
            for l in LAGS:          # l > 0: X change leads T change by l blocks
                if l >= 0:
                    xx, yy = dX[:len(dX) - l], dY[l:]
                else:
                    xx, yy = dX[-l:], dY[:len(dY) + l]
                pairs[q][l][0].extend(xx.tolist())
                pairs[q][l][1].extend(yy.tolist())
    used = False
    for q in pairs:
        for l in LAGS:
            x, y = np.array(pairs[q][l][0]), np.array(pairs[q][l][1])
            m = np.isfinite(x) & np.isfinite(y)
            if m.sum() >= 20 and np.std(x[m]) > 0 and np.std(y[m]) > 0:
                per[q][l].append(spearmanr(x[m], y[m])[0])
                yy = np.roll(y[m], m.sum() // 2)
                null[q][l].append(spearmanr(x[m], yy)[0])
                used = True
    nd += used
print("discharges", nd)
for q, name in (("c", "BES common (log)"), ("g", "BES radial diff 07-05 (log)"), ("e", "ECEI level"), ("k", "PREDICTION: BES common vs BES common passed through the blip response kernel")):
    print(name)
    for l in LAGS:
        v = np.array(per[q][l]); v0 = np.array(null[q][l])
        bs = [np.median(rng.choice(v, len(v))) for _ in range(500)]
        print(f"  lag {l:+d} blk ({30*l:+4d} ms, + = X leads): median rho {np.median(v):+.3f} "
              f"[{np.quantile(bs, .025):+.3f},{np.quantile(bs, .975):+.3f}]  null {np.median(v0):+.3f}  n={len(v)}")
