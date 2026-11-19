# Check 3: at which time scale do T_i (line width) and V_rot (line shift) follow the BES level?
# Block means over L consecutive 10 ms rows (no absent frame inside or within 10 rows before), change between adjacent blocks.
import sys, numpy as np
from scipy.stats import spearmanr
d = np.load(sys.argv[1], allow_pickle=True); X, shot = d["X"], d["shot"]
t = X[:, 0]; ti = X[:, 1].copy(); vt = X[:, 2].copy(); bes = X[:, 3:12]
ti[ti > 3000] = np.nan
rng = np.random.default_rng(3)
new_shot = np.r_[True, shot[1:] != shot[:-1]]
dt = np.r_[np.nan, np.diff(t)]; dt[new_shot] = np.nan
pv = np.r_[np.nan, vt[:-1]]; pv[new_shot] = np.nan
vf = vt.copy(); vf[np.isfinite(vt) & (vt == pv)] = np.nan; vf[~(vf > 5)] = np.nan      # fresh, co-current, above 5 km/s
c = np.log(np.clip(bes, 1e-6, None)).mean(1); c[(bes <= 0).any(1)] = np.nan
lt = np.log(ti); lv = np.log(vf)
blip = (dt > 0.015) & (dt < 0.045)
# contiguous runs of 10 ms steps
brk = new_shot | ~(np.abs(dt - 0.010) < 1e-4)
run_id = np.cumsum(brk)
since = np.full(t.size, 99); k = 99
for i in range(t.size):
    if new_shot[i]: k = 99
    if blip[i]: k = 0
    since[i] = k; k = min(k + 1, 99)
def blocks(L):
    out = []   # shot, dlnT, dlnV, dc
    starts = np.flatnonzero(np.r_[True, run_id[1:] != run_id[:-1]]); ends = np.r_[starts[1:], t.size]
    for a, b in zip(starts, ends):
        n = (b - a) // L
        if n < 2: continue
        sl = slice(a, a + n * L)
        ok_run = (since[sl] > 10).reshape(n, L).all(1)
        def bm(x, need):
            y = x[sl].reshape(n, L); cnt = np.isfinite(y).sum(1)
            m = np.where(cnt >= need, np.nanmean(np.where(np.isfinite(y), y, np.nan), axis=1) if True else 0, np.nan)
            return m
        need = max(1, int(np.ceil(0.5 * L)))
        T = bm(lt, need); V = bm(lv, max(1, int(np.ceil(0.3 * L)))); C = bm(c, need)
        good = ok_run[1:] & ok_run[:-1]
        out.append(np.c_[np.full(n - 1, shot[a]), np.where(good, np.diff(T), np.nan), np.where(good, np.diff(V), np.nan), np.where(good, np.diff(C), np.nan)])
    return np.vstack(out)
import warnings; warnings.simplefilter("ignore")
def slope_ci(x, y, g, n=300):
    lo, hi = np.percentile(y, [0.5, 99.5]); y = np.clip(y, lo, hi)
    ug, inv = np.unique(g, return_inverse=True); k = ug.size
    S = np.zeros((k, 5))
    for j, v in enumerate((np.ones_like(x), x, y, x * x, x * y)): S[:, j] = np.bincount(inv, weights=v, minlength=k)
    def sl(s): 
        n_, sx, sy, sxx, sxy = s; return (sxy - sx * sy / n_) / (sxx - sx * sx / n_)
    b = [sl(S[rng.integers(0, k, k)].sum(0)) for _ in range(n)]
    return sl(S.sum(0)), np.percentile(b, [2.5, 97.5])
print("block length (ms) | T_i on BES: n, slope [95%], spearman | V_rot on BES (same blocks as T_i where both defined): n, T_i slope, V slope [95%], spearman V | spearman(T_i, V_rot)")
for L in (1, 2, 5, 10, 20, 50, 100):
    B = blocks(L); g, dT, dV, dC = B[:, 0], B[:, 1], B[:, 2], B[:, 3]
    m = np.isfinite(dT) & np.isfinite(dC)
    s, ci = slope_ci(dC[m], dT[m], g[m]); r = spearmanr(dC[m], dT[m]).statistic
    m2 = m & np.isfinite(dV)
    s1, ci1 = slope_ci(dC[m2], dT[m2], g[m2]); s2, ci2 = slope_ci(dC[m2], dV[m2], g[m2])
    r2 = spearmanr(dC[m2], dV[m2]).statistic; r3 = spearmanr(dT[m2], dV[m2]).statistic
    print(f"{L*10:5d} | n={m.sum():6d} {s:+.3f} [{ci[0]:+.3f},{ci[1]:+.3f}] rho={r:+.3f} | n={m2.sum():6d} shots={np.unique(g[m2]).size} T:{s1:+.3f} [{ci1[0]:+.3f},{ci1[1]:+.3f}] V:{s2:+.3f} [{ci2[0]:+.3f},{ci2[1]:+.3f}] rhoV={r2:+.3f} | rho(T,V)={r3:+.3f}")
