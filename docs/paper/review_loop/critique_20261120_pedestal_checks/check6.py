# Check 6: does the ECEI level of the same frame follow T_i as BES does, and is that independent of BES?
import sys, numpy as np, warnings
warnings.simplefilter("ignore")
from scipy.stats import spearmanr, rankdata
d = np.load(sys.argv[1], allow_pickle=True); X, shot = d["X"], d["shot"]
t = X[:, 0]; ti = X[:, 1].copy(); bes = X[:, 3:12]; ecei = X[:, 12:16]
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
# per-discharge normalisation by the median level of each channel; channels alive when the median exceeds 0.05 V
en = np.full_like(ecei, np.nan); alive = np.zeros((N, 4), bool)
for s in np.unique(shot):
    q = shot == s; med = np.median(ecei[q], 0); sd = ecei[q].std(0)
    for j in range(4):
        if med[j] > 0.05 and sd[j] > 1e-4:
            en[q, j] = ecei[q, j] / med[j]; alive[q, j] = True
e_mean = np.nanmean(en, 1); e_mean[alive.sum(1) < 2] = np.nan
c = np.where((bes > 0).all(1), np.log(np.clip(bes, 1e-9, None)).mean(1), np.nan)
dT = np.log(ti) - np.log(prev(ti)); dc = c - prev(c); de = e_mean - prev(e_mean)
P("discharges with at least two live ECEI channels:", np.unique(shot[np.isfinite(e_mean)]).size, "of", np.unique(shot).size)
m = quiet & np.isfinite(dT) & np.isfinite(dc) & np.isfinite(de)
P("quiet rows with T_i, BES and ECEI changes: n=%d discharges=%d" % (m.sum(), np.unique(shot[m]).size))
P("robust sd of the frame-to-frame change: ECEI level %.4f (of its median) ; BES %.4f (ln)" % (1.4826 * np.median(np.abs(de[m] - np.median(de[m]))), 1.4826 * np.median(np.abs(dc[m] - np.median(dc[m])))))
r_tb = spearmanr(dT[m], dc[m]).statistic; r_te = spearmanr(dT[m], de[m]).statistic; r_be = spearmanr(dc[m], de[m]).statistic
P("spearman: T_i-BES %+.3f ; T_i-ECEI %+.3f ; BES-ECEI %+.3f" % (r_tb, r_te, r_be))
def partial(a, b, z):
    ra, rb, rz = rankdata(a), rankdata(b), rankdata(z)
    def res(u, v):
        beta = np.polyfit(v, u, 1); return u - np.polyval(beta, v)
    return np.corrcoef(res(ra, rz), res(rb, rz))[0, 1]
P("partial rank correlation: T_i-ECEI given BES %+.3f ; T_i-BES given ECEI %+.3f" % (partial(dT[m], de[m], dc[m]), partial(dT[m], dc[m], de[m])))
for j in range(4):
    dj = en[:, j] - prev(en[:, j]); mj = quiet & np.isfinite(dT) & np.isfinite(dj)
    P("   channel %d: n=%d spearman with T_i %+.3f ; with BES %+.3f" % (j, mj.sum(), spearmanr(dT[mj], dj[mj]).statistic, spearmanr(dc[mj & np.isfinite(dc)], dj[mj & np.isfinite(dc)]).statistic))
# per-discharge distribution of the two rank correlations
rb, re_ = [], []
for s in np.unique(shot[m]):
    q = m & (shot == s)
    if q.sum() >= 100:
        rb.append(spearmanr(dT[q], dc[q]).statistic); re_.append(spearmanr(dT[q], de[q]).statistic)
rb, re_ = np.array(rb), np.array(re_)
P("per discharge (at least 100 rows, n=%d): median spearman T_i-BES %+.3f (share positive %.3f) ; T_i-ECEI %+.3f (share positive %.3f)" % (rb.size, np.median(rb), np.mean(rb > 0), np.median(re_), np.mean(re_ > 0)))
P("   quantiles 10,25,50,75,90: BES %s ; ECEI %s" % (np.round(np.percentile(rb, [10, 25, 50, 75, 90]), 3), np.round(np.percentile(re_, [10, 25, 50, 75, 90]), 3)))
# lagged: T_i change against the ECEI change one frame earlier / later
for lag in (-1, 1):
    de_l = np.roll(de, -lag); bad = np.roll(shot, -lag) != shot; de_l = de_l.copy(); de_l[bad] = np.nan
    dc_l = np.roll(dc, -lag); dc_l = dc_l.copy(); dc_l[bad] = np.nan
    ml = quiet & np.roll(quiet, -lag) & np.isfinite(dT) & np.isfinite(de_l) & np.isfinite(dc_l)
    P("   T_i change at t against the change at t%+d: ECEI %+.3f ; BES %+.3f (n=%d)" % (lag, spearmanr(dT[ml], de_l[ml]).statistic, spearmanr(dT[ml], dc_l[ml]).statistic, ml.sum()))
