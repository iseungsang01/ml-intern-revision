"""P07 time-domain check: missing rate vs lag after an input-only density
crash.  Event t0 = row where the per-file z-scored BES mean drops by more than
the 1st percentile of all one-step increments (within a block), with a 10-row
refractory period.  Reports P(T_i missing | lag) and P(V_rot genuine missing |
lag) for lag -5..+15, with the block-wide base rate, plus the fitted
exponential decay of the excess (tau_B in rows) by least squares on lags 0..15.
Descriptive only, no model.  ASCII output."""
import glob, os, json
import numpy as np, pandas as pd
DATA = "C:/Users/lss/Documents/GitHub/ml-intern-revision/data"
GAP = 0.5
LAGS = list(range(-5, 16))

def stuck(values, block):
    m = np.zeros(values.shape[0], bool); obs = np.flatnonzero(~np.isnan(values))
    if obs.size <= 1: return m
    ov, ob = values[obs], block[obs]; dup = (ov[1:] == ov[:-1]) & (ob[1:] == ob[:-1]); m[obs[1:][dup]] = True; return m

T, V, Z, BLK = [], [], [], []
nb = 0
for f in sorted(glob.glob(os.path.join(DATA, "*.csv"))):
    df = pd.read_csv(f); t = df["time"].to_numpy(float)
    blk = np.concatenate(([0], np.cumsum(np.diff(t) >= GAP))) + nb; nb = int(blk.max()) + 1
    bes = df[[c for c in df.columns if c.startswith("BES_")]].to_numpy(float).mean(1)
    z = (bes - bes.mean()) / (bes.std() if bes.std() > 0 else np.nan)
    T.append(df["CES_TI"].to_numpy(float)); V.append(df["CES_VT"].to_numpy(float)); Z.append(z); BLK.append(blk)
T, V, Z, BLK = map(np.concatenate, (T, V, Z, BLK))
missT = np.isnan(T); missV = np.isnan(V) | stuck(V, BLK)
dz = np.full(Z.shape, np.nan); dz[1:] = Z[1:] - Z[:-1]; dz[1:][BLK[1:] != BLK[:-1]] = np.nan
for pct, name in ((1.0, "p1"), (0.5, "p05")):
    thr = np.nanpercentile(dz, pct)
    cand = np.flatnonzero(dz < thr)
    events, last = [], -10**9
    for i in cand:
        if i - last >= 10: events.append(i); last = i
    events = np.array(events)
    print("BES-drop events (%s, thr=%.3f): %d" % (name, thr, len(events)))
    res = {}
    for tgt, miss in (("T_i", missT), ("V_rot_gen", missV)):
        rates = []
        for L in LAGS:
            j = events + L; ok = (j >= 0) & (j < len(miss)); ok[ok] &= BLK[j[ok]] == BLK[events[ok]]
            rates.append(float(miss[j[ok]].mean()))
        base = float(miss.mean())
        # fit excess(L) = A exp(-L/tau) on lags 0..15 (log-linear on positive excess)
        ex = np.array([r - base for r, L in zip(rates, LAGS) if L >= 0]); Ls = np.array([L for L in LAGS if L >= 0])
        pos = ex > 0
        tau = float("nan"); A = float("nan")
        if pos.sum() >= 4:
            p = np.polyfit(Ls[pos], np.log(ex[pos]), 1); tau = -1.0 / p[0]; A = float(np.exp(p[1]))
        res[tgt] = {"rates": rates, "base": base, "tau_rows": tau, "A": A}
        print("  %-10s base=%.4f  " % (tgt, base) + " ".join("L%+d:%.3f" % (L, r) for L, r in zip(LAGS, rates)))
        print("  %-10s excess fit: A=%.4f tau=%.2f rows (%.0f ms)" % (tgt, A, tau, 10 * tau))
    json.dump({"events": int(len(events)), "thr": float(thr), "lags": LAGS, "res": res},
              open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "p07_event_%s.json" % name), "w"), indent=1)
