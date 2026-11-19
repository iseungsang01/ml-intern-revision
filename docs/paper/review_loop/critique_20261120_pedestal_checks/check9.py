# Check 9: does the same-frame coupling of T_i with the BES level depend on how much light there is?
# A fit that depends on the signal level predicts a steeper coupling at low light; a moving profile does not predict that.
import sys, numpy as np, warnings
warnings.simplefilter("ignore")
from scipy.stats import spearmanr
d = np.load(sys.argv[1], allow_pickle=True); X, shot = d["X"], d["shot"]
t = X[:, 0]; ti = X[:, 1].copy(); bes = X[:, 3:12]
ti[ti > 3000] = np.nan
N = t.size; P = lambda *a: print(*a, flush=True)
new_shot = np.r_[True, shot[1:] != shot[:-1]]
dt = np.r_[np.nan, np.diff(t)]; dt[new_shot] = np.nan
anyblip = (dt > 0.015) & (dt < 0.045)
since = np.full(N, 99); k = 99
for i in range(N):
    if new_shot[i]: k = 99
    if anyblip[i]: k = 0
    since[i] = k; k = min(k + 1, 99)
def prev(a):
    p = np.r_[np.nan, a[:-1]]; p[new_shot] = np.nan; return p
quiet = (np.abs(dt - 0.010) < 1e-4) & (since > 10)
c = np.where((bes > 0).all(1), np.log(np.clip(bes, 1e-9, None)).mean(1), np.nan)
dT = np.log(ti) - np.log(prev(ti)); dc = c - prev(c)
# light level of the pair relative to the discharge: mean of the two rows' ln level minus the discharge median
lev = 0.5 * (c + prev(c)); rel = np.full(N, np.nan)
for s in np.unique(shot):
    q = shot == s
    if np.isfinite(c[q]).sum() > 30: rel[q] = lev[q] - np.nanmedian(c[q])
rng = np.random.default_rng(9)
def slope_ci(x, y, g, n=400):
    lo, hi = np.percentile(y, [0.5, 99.5]); y = np.clip(y, lo, hi)
    ug, inv = np.unique(g, return_inverse=True); K = ug.size; S = np.zeros((K, 5))
    for j, v in enumerate((np.ones_like(x), x, y, x * x, x * y)): S[:, j] = np.bincount(inv, weights=v, minlength=K)
    def sl(s):
        n_, sx, sy, sxx, sxy = s; return (sxy - sx * sy / n_) / (sxx - sx * sx / n_)
    b = [sl(S[rng.integers(0, K, K)].sum(0)) for _ in range(n)]
    return sl(S.sum(0)), np.percentile(b, [2.5, 97.5])
m = quiet & np.isfinite(dT) & np.isfinite(dc) & np.isfinite(rel)
edges = np.percentile(rel[m], [20, 40, 60, 80])
P("quiet frame pairs: %d ; quintile edges of the light level relative to the discharge median (ln): %s" % (m.sum(), np.round(edges, 3)))
lab = np.digitize(rel, edges)
for q in range(5):
    s = m & (lab == q)
    sl, ci = slope_ci(dc[s], dT[s], shot[s])
    miss = quiet & np.isfinite(dc) & np.isfinite(rel) & (lab == q)
    P("light quintile %d (median relative level %+.3f): n=%6d  slope of dlnT_i on dlnBES %+.3f [%+.3f, %+.3f]  spearman %+.3f  robust sd of dlnT_i %.3f  share of frames without a T_i %.3f" % (
        q + 1, np.median(rel[s]), s.sum(), sl, ci[0], ci[1], spearmanr(dc[s], dT[s]).statistic, 1.4826 * np.median(np.abs(dT[s] - np.median(dT[s]))), np.mean(~np.isfinite(ti[miss]))))
# difference of the slope between the lowest and the highest quintile, bootstrap over discharges
lo_ = m & (lab == 0); hi_ = m & (lab == 4)
ug = np.unique(shot[m]); K = ug.size
def suff(sel):
    inv = np.searchsorted(ug, shot[sel]); x = dc[sel]; y = np.clip(dT[sel], *np.percentile(dT[sel], [0.5, 99.5])); S = np.zeros((K, 5))
    for j, v in enumerate((np.ones_like(x), x, y, x * x, x * y)): S[:, j] = np.bincount(inv, weights=v, minlength=K)
    return S
def sl(s):
    n_, sx, sy, sxx, sxy = s; return (sxy - sx * sy / n_) / (sxx - sx * sx / n_)
SL, SH = suff(lo_), suff(hi_); dif = []
for _ in range(1000):
    p = rng.integers(0, K, K); dif.append(sl(SL[p].sum(0)) - sl(SH[p].sum(0)))
P("slope in the lowest light quintile minus the highest: %+.3f [%+.3f, %+.3f]" % (sl(SL.sum(0)) - sl(SH.sum(0)), *np.percentile(dif, [2.5, 97.5])))
