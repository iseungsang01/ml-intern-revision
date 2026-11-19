# Check 1: does the line SHIFT (V_rot) of the same spectrum follow the BES level the way the line WIDTH (T_i) does?
import os, sys, numpy as np
os.environ.setdefault("OMP_NUM_THREADS", "2")
from scipy.stats import spearmanr
d = np.load(sys.argv[1], allow_pickle=True)
X, shot, cols = d["X"], d["shot"], list(d["cols"])
t = X[:, 0]; ti = X[:, 1].copy(); vt = X[:, 2].copy()
bes = X[:, 3:12]
rng = np.random.default_rng(20261120)
ti[ti > 3000] = np.nan                       # cut population
# descriptive: V_rot repeats
new_shot = np.r_[True, shot[1:] != shot[:-1]]
dt = np.r_[np.nan, np.diff(t)]; dt[new_shot] = np.nan
step10 = np.abs(dt - 0.010) < 1e-4
prev_vt = np.r_[np.nan, vt[:-1]]; prev_vt[new_shot] = np.nan
prev_ti = np.r_[np.nan, ti[:-1]]; prev_ti[new_shot] = np.nan
rep_v = np.isfinite(vt) & np.isfinite(prev_vt) & (vt == prev_vt)
rep_t = np.isfinite(ti) & np.isfinite(prev_ti) & (ti == prev_ti)
print("V_rot finite", np.isfinite(vt).sum(), "repeats of previous row", rep_v.sum(), "share", rep_v.sum() / np.isfinite(vt).sum())
print("T_i finite", np.isfinite(ti).sum(), "repeats", rep_t.sum())
# run-length pattern of V_rot repeats
vt_fresh = vt.copy(); vt_fresh[rep_v] = np.nan
c = np.log(np.clip(bes, 1e-6, None)).mean(1)   # common mode of the BES array (log)
bad_bes = (bes <= 0).any(1)
c[bad_bes] = np.nan
prev_c = np.r_[np.nan, c[:-1]]; prev_c[new_shot] = np.nan
dc = c - prev_c
dlnT = np.log(ti) - np.log(prev_ti)
prev_vf = np.r_[np.nan, vt_fresh[:-1]]; prev_vf[new_shot] = np.nan
# rotation change in units of the discharge's own median |V| (fresh values), finite when both rows are fresh
medv = {}
for s in np.unique(shot):
    m = shot == s
    v = vt_fresh[m]; v = v[np.isfinite(v)]
    medv[s] = np.median(np.abs(v)) if v.size >= 20 else np.nan
mv = np.array([medv[s] for s in shot])
dV = (vt_fresh - prev_vf) / mv
dV[(mv < 5) | ~np.isfinite(mv)] = np.nan
# also log ratio where both positive and beyond 5 km/s
ok_v = (vt_fresh > 5) & (prev_vf > 5)
dlnV = np.where(ok_v, np.log(np.where(ok_v, vt_fresh, 1)) - np.log(np.where(ok_v, prev_vf, 1)), np.nan)

def boot_slope(x, y, g, n=400):
    # cluster bootstrap over discharges of the OLS slope of y on x and Spearman rho
    ug = np.unique(g); idx = {u: np.flatnonzero(g == u) for u in ug}
    sl = []; rh = []
    for _ in range(n):
        pick = rng.choice(ug, ug.size)
        ii = np.concatenate([idx[u] for u in pick])
        xx, yy = x[ii], y[ii]
        sl.append(np.cov(xx, yy)[0, 1] / np.var(xx, ddof=1))
    return np.percentile(sl, [2.5, 97.5])

def report(name, mask):
    for lab, y in (("dlnT_i", dlnT), ("dV/|V|med", dV), ("dlnV", dlnV)):
        m = mask & np.isfinite(dc) & np.isfinite(y)
        x, yy, g = dc[m], y[m], shot[m]
        # clip heavy tails of y at the 0.5 and 99.5 percentiles for the slope (robustness shown beside)
        lo, hi = np.percentile(yy, [0.5, 99.5]); yc = np.clip(yy, lo, hi)
        sl = np.cov(x, yc)[0, 1] / np.var(x, ddof=1)
        ci = boot_slope(x, yc, g)
        rho = spearmanr(x, yy).statistic
        print(f"  {name:38s} {lab:10s} n={m.sum():7d} shots={np.unique(g).size:4d} slope={sl:+.3f} [{ci[0]:+.3f},{ci[1]:+.3f}] spearman={rho:+.3f} sd(x)={x.std():.3f} sd(y)={yc.std():.3f}")

# row classes
step_blip = (dt > 0.015) & (dt < 0.045)           # first row after 1-3 absent frames
since = np.full(t.size, 99, dtype=int)             # rows since the last absent-frame step
cnt = 99
for i in range(t.size):
    if new_shot[i]: cnt = 99
    if step_blip[i]: cnt = 0
    since[i] = cnt; cnt = min(cnt + 1, 99)
quiet = step10 & (since > 10)
print("rows: first after absent frame", step_blip.sum(), " quiet 10 ms steps", quiet.sum())
print("A. first row after an absent frame (change from the last row before the blip)")
report("first row after absent frame", step_blip)
print("B. quiet rows (10 ms step, more than 10 rows after any absent frame)")
report("quiet, all", quiet)
big = quiet & (np.abs(dc) > 0.10)
report("quiet, |d ln BES| > 0.10", big)
report("quiet, d ln BES < -0.10", quiet & (dc < -0.10))
report("quiet, d ln BES > +0.10", quiet & (dc > 0.10))
# C. are the width and the shift innovations of the same spectrum correlated at all?
m = quiet & np.isfinite(dlnT) & np.isfinite(dV)
print("C. corr of dlnT_i and dV in quiet rows: n", m.sum(), "spearman", spearmanr(dlnT[m], dV[m]).statistic)
m = quiet & np.isfinite(dlnT) & np.isfinite(dlnV)
print("   corr of dlnT_i and dlnV in quiet rows: n", m.sum(), "spearman", spearmanr(dlnT[m], dlnV[m]).statistic)
# D. medians at the first row after a blip (ratio to the row before the blip)
m = step_blip & np.isfinite(dlnT); print("D. median T_i ratio first row / row before:", np.exp(np.median(dlnT[m])), "n", m.sum())
m = step_blip & np.isfinite(dlnV); print("   median V_rot ratio (both fresh, >5 km/s):", np.exp(np.median(dlnV[m])), "n", m.sum())
m = step_blip & np.isfinite(dc); print("   median BES common-mode ratio:", np.exp(np.median(dc[m])), "n", m.sum())
