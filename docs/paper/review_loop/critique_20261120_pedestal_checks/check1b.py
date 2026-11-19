# Check 1b: matched rows, thirds by discharge median T_i, lag structure of the T_i - BES coupling.
import os, sys, numpy as np
from scipy.stats import spearmanr
d = np.load(sys.argv[1], allow_pickle=True)
X, shot = d["X"], d["shot"]
t = X[:, 0]; ti = X[:, 1].copy(); vt = X[:, 2].copy(); bes = X[:, 3:12]; ecei = X[:, 12:16]
ti[ti > 3000] = np.nan
rng = np.random.default_rng(1)
new_shot = np.r_[True, shot[1:] != shot[:-1]]
dt = np.r_[np.nan, np.diff(t)]; dt[new_shot] = np.nan
step10 = np.abs(dt - 0.010) < 1e-4
def prev(a):
    p = np.r_[np.nan, a[:-1]]; p[new_shot] = np.nan; return p
rep_v = np.isfinite(vt) & (vt == prev(vt))
vf = vt.copy(); vf[rep_v] = np.nan
c = np.log(np.clip(bes, 1e-6, None)).mean(1); c[(bes <= 0).any(1)] = np.nan
e = np.where((ecei > 0).all(1), np.log(np.clip(ecei, 1e-6, None)).mean(1), np.nan)
dc = c - prev(c); de = e - prev(e); dlnT = np.log(ti) - np.log(prev(ti))
okv = (vf > 5) & (prev(vf) > 5)
dlnV = np.where(okv, np.log(np.where(okv, vf, 1.0)) - np.log(np.where(okv, prev(vf), 1.0)), np.nan)
step_blip = (dt > 0.015) & (dt < 0.045)
since = np.full(t.size, 99); cnt = 99
for i in range(t.size):
    if new_shot[i]: cnt = 99
    if step_blip[i]: cnt = 0
    since[i] = cnt; cnt = min(cnt + 1, 99)
quiet = step10 & (since > 10)
def slope_ci(x, y, g, n=300):
    lo, hi = np.percentile(y, [0.5, 99.5]); y = np.clip(y, lo, hi)
    s = np.cov(x, y)[0, 1] / np.var(x, ddof=1)
    ug = np.unique(g); idx = {u: np.flatnonzero(g == u) for u in ug}; b = []
    for _ in range(n):
        ii = np.concatenate([idx[u] for u in rng.choice(ug, ug.size)])
        b.append(np.cov(x[ii], y[ii])[0, 1] / np.var(x[ii], ddof=1))
    return s, np.percentile(b, [2.5, 97.5])
print("1. Same rows for both: quiet rows where T_i change AND fresh V_rot change are both defined")
m = quiet & np.isfinite(dc) & np.isfinite(dlnT) & np.isfinite(dlnV)
s, ci = slope_ci(dc[m], dlnT[m], shot[m]); print(f"   dlnT_i on dlnBES: n={m.sum()} shots={np.unique(shot[m]).size} slope={s:+.3f} [{ci[0]:+.3f},{ci[1]:+.3f}] spearman={spearmanr(dc[m], dlnT[m]).statistic:+.3f}")
s, ci = slope_ci(dc[m], dlnV[m], shot[m]); print(f"   dlnV   on dlnBES: slope={s:+.3f} [{ci[0]:+.3f},{ci[1]:+.3f}] spearman={spearmanr(dc[m], dlnV[m]).statistic:+.3f}")
print("   robust scale (1.4826 MAD) of dlnT_i, dlnV, dlnBES on these rows:", [round(1.4826*np.median(np.abs(a[m]-np.median(a[m]))),4) for a in (dlnT, dlnV, dc)])
print("   median |V_rot| (km/s) and median T_i (eV) on these rows:", np.median(np.abs(vf[m])), np.median(ti[m]))
print("2. Thirds of the discharges by median T_i (quiet rows): slope of dlnT_i on dlnBES, and of dlnT_i on dlnECEI")
med = {s_: np.nanmedian(ti[shot == s_]) for s_ in np.unique(shot)}
mt = np.array([med[s_] for s_ in shot])
us = np.array([v for v in med.values() if np.isfinite(v)]); q1, q2 = np.percentile(us, [100/3, 200/3])
print("   tercile edges of the discharge medians (eV):", round(q1, 1), round(q2, 1))
for lab, mm in (("cold", mt <= q1), ("middle", (mt > q1) & (mt <= q2)), ("hot", mt > q2)):
    m = quiet & mm & np.isfinite(dc) & np.isfinite(dlnT)
    s, ci = slope_ci(dc[m], dlnT[m], shot[m])
    m2 = quiet & mm & np.isfinite(de) & np.isfinite(dlnT)
    s2, ci2 = slope_ci(de[m2], dlnT[m2], shot[m2])
    print(f"   {lab:6s} BES: n={m.sum()} slope={s:+.3f} [{ci[0]:+.3f},{ci[1]:+.3f}] rho={spearmanr(dc[m], dlnT[m]).statistic:+.3f} | ECEI: n={m2.sum()} slope={s2:+.3f} [{ci2[0]:+.3f},{ci2[1]:+.3f}] rho={spearmanr(de[m2], dlnT[m2]).statistic:+.3f}")
m2 = quiet & np.isfinite(de) & np.isfinite(dlnT)
s2, ci2 = slope_ci(de[m2], dlnT[m2], shot[m2]); print(f"   all    ECEI: n={m2.sum()} slope={s2:+.3f} [{ci2[0]:+.3f},{ci2[1]:+.3f}] rho={spearmanr(de[m2], dlnT[m2]).statistic:+.3f}  robust sd of dlnECEI {1.4826*np.median(np.abs(de[m2]-np.median(de[m2]))):.4f}")
print("3. Lag structure: deviation of ln T_i(t) from the median of the 10 rows before, against the same deviation of ln BES at t+k")
def dev(a, w=10):
    out = np.full(a.size, np.nan)
    # causal: value minus median of the w rows before, all inside 10 ms steps
    ok = np.ones(a.size, bool)
    stack = []
    for k in range(1, w + 1):
        sh = np.r_[np.full(k, np.nan), a[:-k]]
        bad = np.r_[np.ones(k, bool), shot[k:] != shot[:-k]] | np.r_[np.ones(k, bool), np.abs((t[k:] - t[:-k]) - 0.01 * k) > 1e-4]
        sh[bad] = np.nan; stack.append(sh)
    st = np.vstack(stack); cntf = np.isfinite(st).sum(0)
    medv = np.nanmedian(np.where(cntf >= 6, st, np.nan), axis=0) if True else None
    return a - medv
import warnings; warnings.simplefilter("ignore")
dT = dev(np.log(ti)); dC = dev(c)
for k in (-3, -2, -1, 0, 1, 2, 3):
    if k == 0: cs = dC
    elif k > 0:
        cs = np.r_[dC[k:], np.full(k, np.nan)]; bad = np.r_[shot[k:] != shot[:-k], np.ones(k, bool)] | np.r_[np.abs((t[k:] - t[:-k]) - 0.01 * k) > 1e-4, np.ones(k, bool)]; cs[bad] = np.nan
    else:
        kk = -k; cs = np.r_[np.full(kk, np.nan), dC[:-kk]]; bad = np.r_[np.ones(kk, bool), shot[kk:] != shot[:-kk]] | np.r_[np.ones(kk, bool), np.abs((t[kk:] - t[:-kk]) - 0.01 * kk) > 1e-4]; cs[bad] = np.nan
    m = (since > 13) & np.isfinite(dT) & np.isfinite(cs)
    print(f"   BES at t{k:+d}: n={m.sum()} spearman={spearmanr(cs[m], dT[m]).statistic:+.3f}")
