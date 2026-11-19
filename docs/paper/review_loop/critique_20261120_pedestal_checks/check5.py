# Check 5: is the BES change that T_i follows uniform across the three radial columns (beam-like) or a tilt (profile-like)?
import sys, numpy as np, warnings
warnings.simplefilter("ignore")
from scipy.stats import spearmanr
d = np.load(sys.argv[1], allow_pickle=True); X, shot = d["X"], d["shot"]
t = X[:, 0]; ti = X[:, 1].copy(); vt = X[:, 2]; bes = X[:, 3:12]
ti[ti > 3000] = np.nan
N = t.size; P = lambda *a: print(*a, flush=True)
new_shot = np.r_[True, shot[1:] != shot[:-1]]
dt = np.r_[np.nan, np.diff(t)]; dt[new_shot] = np.nan
okb = (bes > 0).all(1)
L = np.where(okb[:, None], np.log(np.clip(bes, 1e-9, None)), np.nan)
c5 = L[:, [0, 3, 6]].mean(1); c6 = L[:, [1, 4, 7]].mean(1); c7 = L[:, [2, 5, 8]].mean(1)
c = L.mean(1); g = c7 - c5
anyblip = (dt > 0.015) & (dt < 0.045); blip1 = np.abs(dt - 0.020) < 1e-4
since = np.full(N, 99); k = 99
for i in range(N):
    if new_shot[i]: k = 99
    if anyblip[i]: k = 0
    since[i] = k; k = min(k + 1, 99)
def prev(a):
    p = np.r_[np.nan, a[:-1]]; p[new_shot] = np.nan; return p
quiet = (np.abs(dt - 0.010) < 1e-4) & (since > 10)
dT = np.log(ti) - np.log(prev(ti)); dc = c - prev(c); dg = g - prev(g); d5 = c5 - prev(c5); d7 = c7 - prev(c7)
rng = np.random.default_rng(7)
def ols(Xm, y, grp, n=300):
    lo, hi = np.percentile(y, [0.5, 99.5]); y = np.clip(y, lo, hi)
    A = np.c_[Xm, np.ones(len(y))]; b = np.linalg.lstsq(A, y, rcond=None)[0]
    ug, inv = np.unique(grp, return_inverse=True); kk = ug.size; p = A.shape[1]
    XtX = np.zeros((kk, p, p)); Xty = np.zeros((kk, p))
    for a_ in range(p):
        Xty[:, a_] = np.bincount(inv, weights=A[:, a_] * y, minlength=kk)
        for b_ in range(p): XtX[:, a_, b_] = np.bincount(inv, weights=A[:, a_] * A[:, b_], minlength=kk)
    bs = []
    for _ in range(n):
        pick = rng.integers(0, kk, kk); bs.append(np.linalg.solve(XtX[pick].sum(0), Xty[pick].sum(0)))
    bs = np.array(bs); return b[:-1], np.percentile(bs[:, :-1], [2.5, 97.5], axis=0)

P("1. At the first row after a single absent frame (beam at about two thirds): change of the common mode c and of the tilt g = ln(col 07) - ln(col 05)")
m = blip1 & np.isfinite(dc) & np.isfinite(dg)
P("   events %d: median dc %+.3f, median dg %+.3f, ratio of medians %+.3f ; median d ln col05 %+.3f, col07 %+.3f" % (m.sum(), np.median(dc[m]), np.median(dg[m]), np.median(dg[m]) / np.median(dc[m]), np.median(d5[m]), np.median(d7[m])))
P("   predicted by constant backgrounds of 0.110 (col 05) and 0.151 (col 07) at a level ratio of 0.658: dg = %+.3f" % (np.log((0.849 * 0.6075 + 0.151) / (0.890 * 0.6075 + 0.110))))
P("2. Quiet rows, frame to frame: slope of dg on dc (a pure change of the beam gives about -0.05; a shift of the profile gives the curvature of the profile)")
m = quiet & np.isfinite(dc) & np.isfinite(dg)
b, ci = ols(dc[m][:, None], dg[m], shot[m]); P("   n=%d slope %+.3f [%+.3f, %+.3f] ; spearman %+.3f ; sd(dc) %.3f sd(dg) %.3f ; robust sd dc %.4f dg %.4f" % (m.sum(), b[0], ci[0, 0], ci[1, 0], spearmanr(dc[m], dg[m]).statistic, dc[m].std(), dg[m].std(), 1.4826 * np.median(np.abs(dc[m] - np.median(dc[m]))), 1.4826 * np.median(np.abs(dg[m] - np.median(dg[m])))))
P("3. Quiet rows: joint regression of d ln T_i on [dc, dg] (both in ln units)")
m = quiet & np.isfinite(dc) & np.isfinite(dg) & np.isfinite(dT)
b, ci = ols(np.c_[dc[m], dg[m]], dT[m], shot[m])
P("   n=%d  coefficient of dc %+.3f [%+.3f, %+.3f] ; of dg %+.3f [%+.3f, %+.3f]" % (m.sum(), b[0], ci[0, 0], ci[1, 0], b[1], ci[0, 1], ci[1, 1]))
P("   spearman(dT, dc) %+.3f ; spearman(dT, dg) %+.3f" % (spearmanr(dT[m], dc[m]).statistic, spearmanr(dT[m], dg[m]).statistic))
# residualise
bg = np.polyfit(dc[m], dg[m], 1); g_res = dg[m] - np.polyval(bg, dc[m]); bc = np.polyfit(dg[m], dc[m], 1); c_res = dc[m] - np.polyval(bc, dg[m])
P("   spearman(dT, tilt with the common mode removed) %+.3f ; spearman(dT, common mode with the tilt removed) %+.3f" % (spearmanr(dT[m], g_res).statistic, spearmanr(dT[m], c_res).statistic))
P("4. The same by column: joint regression of d ln T_i on [d ln col05, d ln col06, d ln col07]")
d6 = c6 - prev(c6)
m = quiet & np.isfinite(d5) & np.isfinite(d6) & np.isfinite(d7) & np.isfinite(dT)
b, ci = ols(np.c_[d5[m], d6[m], d7[m]], dT[m], shot[m])
P("   col05 %+.3f [%+.3f, %+.3f] ; col06 %+.3f [%+.3f, %+.3f] ; col07 %+.3f [%+.3f, %+.3f]" % (b[0], ci[0, 0], ci[1, 0], b[1], ci[0, 1], ci[1, 1], b[2], ci[0, 2], ci[1, 2]))
P("   pairwise spearman of the column changes: 05-06 %+.3f, 05-07 %+.3f, 06-07 %+.3f" % (spearmanr(d5[m], d6[m]).statistic, spearmanr(d5[m], d7[m]).statistic, spearmanr(d6[m], d7[m]).statistic))
P("5. Rows with the BES level below a quarter of the discharge median that carry values: V_rot there")
lvl = bes.mean(1); medl = np.zeros(N); medv = np.full(N, np.nan); medt = np.full(N, np.nan)
pvt = prev(vt); vf = vt.copy(); vf[np.isfinite(vt) & (vt == pvt)] = np.nan
for s in np.unique(shot):
    q = shot == s; medl[q] = np.median(lvl[q])
    if np.isfinite(vf[q]).sum() > 20: medv[q] = np.nanmedian(vf[q])
    if np.isfinite(ti[q]).sum() > 20: medt[q] = np.nanmedian(ti[q])
low = lvl < 0.25 * medl
q = low & np.isfinite(vf) & (np.abs(medv) > 5)
P("   fresh V_rot values there: %d ; ratio to the discharge median, quantiles 10,25,50,75,90: %s" % (q.sum(), np.round(np.percentile(vf[q] / medv[q], [10, 25, 50, 75, 90]), 3)))
q2 = low & np.isfinite(ti) & np.isfinite(vf) & (np.abs(medv) > 5)
P("   rows with both: %d ; median T_i ratio %.3f ; median V_rot ratio %.3f" % (q2.sum(), np.median(ti[q2] / medt[q2]), np.median(vf[q2] / medv[q2])))
# where are these rows in the discharge
frac = np.zeros(N)
for s in np.unique(shot):
    q = shot == s; frac[q] = (t[q] - t[q].min()) / max(t[q].max() - t[q].min(), 1e-9)
ql = low & np.isfinite(ti)
P("   position of those T_i rows inside the file (0 = first row, 1 = last): quantiles 10,25,50,75,90: %s ; share within 5 rows after an absent frame: %.3f" % (np.round(np.percentile(frac[ql], [10, 25, 50, 75, 90]), 3), np.mean(since[ql] <= 5)))
