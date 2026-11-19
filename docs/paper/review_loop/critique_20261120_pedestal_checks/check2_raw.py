# Check 2 (twelve raw-stream discharges): (a) T_i against the BES level averaged over its own exposure, the snapshot,
# and the exposures before / after; (b) beam-off residual per BES column (does the radial log-difference leak the beam?).
import os, sys, json, numpy as np, pandas as pd
from scipy.stats import spearmanr
data = sys.argv[1]
shots = [31097, 31114, 31357, 31359, 31747, 31873, 31902, 31921, 31923, 32027, 32092, 32097]
rows = []; leak = []
for s in shots:
    z = np.load(os.path.join(data, "raw_mus", ".cache", f"s{s}_BES.npz"), allow_pickle=True)
    x = z["x"].astype(np.float64); meta = json.loads(str(z["meta"])); fs = meta["fs_hz"]; t0 = meta["t0"]; n = x.shape[0]
    lvl = x.mean(1)                                   # common level (linear mean of the nine channels)
    cs = np.r_[0.0, np.cumsum(lvl)]
    csc = np.vstack([np.r_[0.0, np.cumsum(x[:, j])] for j in range(9)])
    def wmean(a, b, cum=cs):                          # mean over [a, b) seconds
        i0 = np.clip(np.round((a - t0) * fs).astype(int), 0, n); i1 = np.clip(np.round((b - t0) * fs).astype(int), 0, n)
        out = np.full(a.shape, np.nan); ok = (i1 - i0) > 0.9 * (b - a) * fs
        out[ok] = (cum[i1[ok]] - cum[i0[ok]]) / (i1[ok] - i0[ok]); return out
    df = pd.read_csv(os.path.join(data, f"s{s}.csv")).iloc[1:-1].reset_index(drop=True)
    t = df["time"].to_numpy(); ti = df["CES_TI"].to_numpy().copy(); ti[ti > 3000] = np.nan
    snap = df[[c for c in df.columns if c.startswith("BES_")]].to_numpy().mean(1)
    E = wmean(t - 0.002, t + 0.008); P = wmean(t - 0.012, t - 0.002); N = wmean(t + 0.008, t + 0.018)
    E1 = wmean(t - 0.0005, t + 0.0005)                # 1 ms around the stamp
    dt = np.r_[np.nan, np.diff(t)]
    blip = (dt > 0.015) & (dt < 0.045); since = np.full(t.size, 99); c = 99
    for i in range(t.size):
        if blip[i]: c = 0
        since[i] = c; c = min(c + 1, 99)
    rows.append(pd.DataFrame(dict(shot=s, t=t, ti=ti, snap=snap, E=E, P=P, N=N, E1=E1, dt=dt, since=since)))
    # (b) beam-off residual per channel at single absent frames: level inside [stamp_absent+1ms, +7ms] over the 20 ms before the blip
    for i in np.flatnonzero(blip & (np.abs(dt - 0.020) < 1e-4)):
        ta = t[i] - 0.010                              # stamp of the absent frame
        a0 = np.array([ta + 0.001]); a1 = np.array([ta + 0.007]); b0 = np.array([ta - 0.025]); b1 = np.array([ta - 0.005])
        off = np.array([wmean(a0, a1, csc[j])[0] for j in range(9)]); on = np.array([wmean(b0, b1, csc[j])[0] for j in range(9)])
        if np.all(np.isfinite(off)) and np.all(on > 0) and off.mean() < 0.5 * on.mean():
            leak.append(np.r_[s, off / on, on])
R = pd.concat(rows, ignore_index=True)
print("rows inside raw windows with exposure average:", np.isfinite(R.E).sum(), "discharges", R.shot.nunique())
q = (np.abs(R.dt - 0.010) < 1e-4) & (R.since > 10)
def d(a): 
    v = np.log(a.to_numpy()); out = np.r_[np.nan, np.diff(v)]; return out
dT = d(R.ti); 
same = np.r_[False, R.shot.to_numpy()[1:] == R.shot.to_numpy()[:-1]]
print("(a) quiet rows, frame-to-frame change of ln T_i against the change of ln BES level measured four ways")
for name in ("snap", "E1", "E", "P", "N"):
    dx = d(R[name]); m = q.to_numpy() & same & np.isfinite(dT) & np.isfinite(dx)
    yy = np.clip(dT[m], *np.percentile(dT[m], [0.5, 99.5]))
    sl = np.cov(dx[m], yy)[0, 1] / np.var(dx[m], ddof=1)
    per = [spearmanr(dx[m & (R.shot.to_numpy() == s)], dT[m & (R.shot.to_numpy() == s)]).statistic for s in shots if (m & (R.shot.to_numpy() == s)).sum() > 50]
    print(f"   {name:5s} n={m.sum():5d} spearman={spearmanr(dx[m], dT[m]).statistic:+.3f} slope={sl:+.3f} sd(dln level)={dx[m].std():.4f} per-discharge spearman median={np.median(per):+.3f} (positive in {sum(p > 0 for p in per)}/{len(per)})")
# variance of the snapshot level that survives a 10 ms average
m = q.to_numpy() & same & np.isfinite(d(R.E)) & np.isfinite(d(R.snap))
print("   sd of d ln(snapshot) vs d ln(exposure mean):", d(R.snap)[m].std(), d(R.E)[m].std(), " spearman(dln snap, dln E) =", spearmanr(d(R.snap)[m], d(R.E)[m]).statistic)
# multiple regression: does the exposure mean absorb the snapshot, or the reverse?
m = q.to_numpy() & same & np.isfinite(dT) & np.isfinite(d(R.E)) & np.isfinite(d(R.snap)) & np.isfinite(d(R.N)) & np.isfinite(d(R.P))
A = np.c_[d(R.snap)[m], d(R.E)[m], d(R.P)[m], d(R.N)[m], np.ones(m.sum())]
yy = np.clip(dT[m], *np.percentile(dT[m], [0.5, 99.5]))
A[:, :4] = (A[:, :4] - A[:, :4].mean(0)) / A[:, :4].std(0)
co = np.linalg.lstsq(A, yy, rcond=None)[0]
print("   joint regression of dlnT_i on standardised [snapshot, own exposure, exposure before, exposure after]:", np.round(co[:4], 4), " sd(dlnT clipped)", round(yy.std(), 4))
L = np.array(leak)
print("(b) single absent frames inside raw windows:", len(L))
if len(L):
    f = L[:, 1:10]; names = ["0105", "0106", "0107", "0205", "0206", "0207", "0305", "0306", "0307"]
    print("   beam-off residual (level with beam off / level before), median per channel:", dict(zip(names, np.round(np.median(f, 0), 3))))
    c5 = f[:, [0, 3, 6]].mean(1); c6 = f[:, [1, 4, 7]].mean(1); c7 = f[:, [2, 5, 8]].mean(1)
    print("   by column 05 / 06 / 07: median", np.round([np.median(c5), np.median(c6), np.median(c7)], 3), " median of (col07 - col05):", round(np.median(c7 - c5), 3), " share of events with col07 > col05:", round(np.mean(c7 > c5), 3))
    on = L[:, 10:19]
    print("   level before the blip by column 05/06/07 (median, V):", np.round([np.median(on[:, [0, 3, 6]]), np.median(on[:, [1, 4, 7]]), np.median(on[:, [2, 5, 8]])], 4))
R.to_pickle(os.path.join(sys.argv[2], "raw_rows.pkl"))
