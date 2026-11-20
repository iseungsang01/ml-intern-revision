# -*- coding: utf-8 -*-
"""Session lead's checks of B.75, written independently of the agent's code.

(A) Statement values recomputed from the stored per-unit estimates with own pooling.
(B) Unit V4a, rule D650, target L5: every input recomputed from the grid with own code (visibility, history scalars,
    causal GP, fast scalars, anchor scalars), own fits of T0 and TX, compared with the stored estimates.
(C) Controls (unregistered): TX refitted with the fast and anchor scalars permuted over rows (placebo), and with the
    fast scalars taken at the anchor row instead of row t (fast information as old as the history).
    py verify_b75.py <repo root>
"""
import os
import sys
import time
from pathlib import Path

for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[v] = "2"
import numpy as np

ROOT = Path(sys.argv[1])
sys.path.insert(0, str(ROOT / "ces_prediction"))
import baselines_interpolation as BI  # noqa: E402

B73, B75 = ROOT / "data" / ".b73", ROOT / "data" / ".b75"
T0_ = time.time()


def log(m):
    print("[%5.0fs] %s" % (time.time() - T0_, m), flush=True)


R = dict(np.load(B73 / "rows.npz", allow_pickle=False))
S = dict(np.load(B75 / "sets.npz", allow_pickle=False))
y, L5, F = R["y"], R["tg_L5"], R["tg_F"]


def est(unit, learner, rule, tg):
    z = np.load(B75 / "fits" / ("%s__%s__%s__%s.npz" % (unit, learner, rule, tg)))
    return z["idx"], z["est"]


def sse_pair(units, la, lb, rule, tg, target):
    a = b = 0.0
    n = 0
    for u in units:
        ia, ea = est(u, la, rule, tg)
        ib, eb = est(u, lb, rule, tg)
        assert np.array_equal(ia, ib)
        t = target[ia]
        assert np.isfinite(t).all()
        a += ((ea - t) ** 2).sum()
        b += ((eb - t) ** 2).sum()
        n += len(ia)
    return a, b, n


def incr(units, rule, tg="L5", la="TX", lb="T0"):
    a, b, n = sse_pair(units, la, lb, rule, tg, R["tg_" + tg])
    return 1 - a / b, n


print("(A) from the stored estimates, own pooling")
V1 = [["V1_r%d_k%d" % (r, k) for k in range(5)] for r in (2, 3, 4)]
V2 = [["V2_s%d_k%d" % (s, k) for k in range(5)] for s in (1, 2, 3)]
V3 = ["V3_q%d" % q for q in (1, 2, 3, 4)]
for rule in ("D10", "D650", "S"):
    print("  TX over T0, L5, %-5s: V1 %+.3f  V2 %+.3f  V3 %+.3f  V4a %+.3f  V4b %+.3f" % (
        rule, np.mean([incr(u, rule)[0] for u in V1]), np.mean([incr(u, rule)[0] for u in V2]),
        incr(V3, rule)[0], incr(["V4a"], rule)[0], incr(["V4b"], rule)[0]))
print("  TE over T0, L5, D650, V1: %+.3f" % np.mean([incr(u, "D650", la="TE")[0] for u in V1]))
print("  T0 over G, L5, D650: needs G; see part B for V4a")


def vs_own_reading(units, rule):
    a = b = 0.0
    for u in units:
        i, e = est(u, "TX", rule, "L5")
        a += ((e - L5[i]) ** 2).sum()
        b += ((y[i] - L5[i]) ** 2).sum()
    return 1 - a / b, np.sqrt(b)


print("  1 - MSE(TX)/MSE(own reading), L5: D10 V1 %+.3f, D10 V4a %+.3f, S V1 %+.3f" % (
    np.mean([vs_own_reading(u, "D10")[0] for u in V1]), vs_own_reading(["V4a"], "D10")[0],
    np.mean([vs_own_reading(u, "S")[0] for u in V1])))
n_idx = sum(len(est(u, "TX", "D650", "L5")[0]) for u in V1[0])
print("  V1 r2 scored rows under D650 (L5): %d; common_L5 rows: %d" % (n_idx, int(S["common_L5"].sum())))

# ---------------------------------------------------------------------------------------------------- part B
print("(B) own inputs for rule D650, all D1 rows")
G = dict(np.load(B73 / "grid.npz", allow_pickle=False))
print("  grid keys:", sorted(G.keys()))
tm_all, ti_all, seg_all, off = G["time"].astype(np.float64), G["ti"].astype(np.float64), G["seg"], G["off"]
fast_all = G["fast"].astype(np.float64)
shot_file = G["shot"]
gi, fidx = R["gi"], R["fidx"]
n = len(gi)
D, TOL = 0.650, 0.003
H = np.full(n, np.nan)
AGE = np.full(n, np.nan)
GP = np.full(n, np.nan)
TH = np.full((n, 5), np.nan)
ANC = np.full(n, -1, np.int64)
order = np.argsort(gi)
for i in np.unique(fidx):
    a, b = off[i], off[i + 1]
    tm, ti, sg = tm_all[a:b], ti_all[a:b], seg_all[a:b]
    rows_i = np.flatnonzero(fidx == i)
    for q in rows_i:
        g = gi[q] - a
        m = (sg == sg[g]) & np.isfinite(ti) & (tm <= tm[g] - D + TOL)
        vis = np.flatnonzero(m)
        if len(vis) == 0:
            continue
        last = vis[-1]
        H[q] = ti[last]
        AGE[q] = (tm[g] - tm[last]) * 1000.0
        ANC[q] = a + last
        v10 = ti[vis[-10:]]
        if len(v10) >= 3:
            TH[q] = (v10.mean(), v10.std(), v10[-1] - v10.mean(), v10[-1] - v10[-2], len(v10))
        v16 = vis[-16:]
        GP[q] = BI.predict_gp_causal(tm[v16], ti[v16], tm[g])
log("history and causal GP done")
A = dict(np.load(B75 / "inputs" / "D650.npz", allow_pickle=False))
both = np.isfinite(GP) & np.isfinite(A["G"])
print("  rows with a visible value: mine %d, stored %d" % (int(np.isfinite(H).sum()), int(np.isfinite(A["H"]).sum())))
print("  largest |mine - stored|: H %.2e eV, age %.2e ms, G %.2e eV, history scalars %.2e; anchor rows equal: %s" % (
    np.nanmax(np.abs(H - A["H"])), np.nanmax(np.abs(AGE - A["age"])), np.max(np.abs(GP[both] - A["G"][both])),
    np.nanmax(np.abs(TH - A["th"])), bool(np.array_equal(ANC[ANC >= 0], A["anc_g"][ANC >= 0]))))

# fast scalars with the statistics of the fit discharges of V4a (D1 discharges up to shot 31991)
shot_row = R["shot"]
fit_files = np.unique(fidx[shot_row <= 31991])
test_rows_all = shot_row >= 32312
frow = np.isin(G["fidx"], fit_files)
mu, sd = fast_all[frow].mean(0), fast_all[frow].std(0) + 1e-12
z = (fast_all - mu) / sd
NB, NE = 9, 4
bes_z, ece_z = z[:, :NB].mean(1), z[:, NB:NB + NE].mean(1)
bes_r, ece_r = fast_all[:, :NB].mean(1), fast_all[:, NB:NB + NE].mean(1)
N = len(tm_all)


def wfeat(x):
    out = np.full((N, 4), np.nan)
    for i in range(10, N):
        pass
    idx = np.arange(10, N)
    ok = seg_all[idx] == seg_all[idx - 10]
    idx = idx[ok]
    win = np.stack([x[idx - k] for k in range(1, 11)], 1)
    out[idx, 0] = x[idx]
    out[idx, 1] = x[idx] - win.mean(1)
    out[idx, 2] = win.std(1)
    out[idx, 3] = x[idx] - x[idx - 1]
    return out


def runmean(x):
    """mean of x over rows i-9..i of the same block; NaN with fewer than five rows."""
    s = np.zeros(N)
    c = np.zeros(N)
    for k in range(10):
        j = np.arange(N) - k
        ok = (j >= 0) & (seg_all[np.maximum(j, 0)] == seg_all)
        s[ok] += x[j[ok]]
        c[ok] += 1
    return np.where(c >= 5, s / np.maximum(c, 1), np.nan)


WF = np.column_stack([wfeat(bes_z), wfeat(ece_z)])
mz_b, mz_e, mr_b, mr_e = runmean(bes_z), runmean(ece_z), runmean(bes_r), runmean(ece_r)
has = ANC >= 0
anc = np.where(has, ANC, 0)


def anchor3(mz, mr):
    a1 = np.where(has, mz[anc], np.nan)
    a2 = mz[gi] - a1
    with np.errstate(invalid="ignore", divide="ignore"):
        a3 = np.where(has & (mr[gi] > 0) & (mr[anc] > 0), np.log(mr[gi] / mr[anc]), np.nan)
    return np.column_stack([a1, a2, a3])


FAST8 = WF[gi]
ANC6 = np.column_stack([anchor3(mz_b, mr_b), anchor3(mz_e, mr_e)])
H10 = np.column_stack([R["tstep"], AGE, H, GP - H, np.abs(GP - H), TH])
X_T0 = H10
X_TX = np.column_stack([H10, FAST8, ANC6])
common = S["common_L5"]
print("  my D650 inputs finite on the stored common set (L5): %s (%d rows)" % (bool(np.isfinite(X_TX[common]).all()), int(common.sum())))
fit = common & np.isin(fidx, fit_files)
test = common & test_rows_all
i_st, e_t0 = est("V4a", "T0", "D650", "L5")
_, e_tx = est("V4a", "TX", "D650", "L5")
print("  V4a fit rows %d, test rows %d (stored test rows %d, same rows: %s)" % (
    int(fit.sum()), int(test.sum()), len(i_st), bool(np.array_equal(np.flatnonzero(test), np.sort(i_st)))))
from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: E402


def fit_predict(X, rows_fit, rows_test):
    m = HistGradientBoostingRegressor(random_state=20261116).fit(X[rows_fit], (L5 - GP)[rows_fit])
    return GP[rows_test] + m.predict(X[rows_test])


ti_ = np.flatnonzero(test)
t = L5[ti_]
p0 = fit_predict(X_T0, fit, ti_)
px = fit_predict(X_TX, fit, ti_)
sse = lambda p: float(((p - t) ** 2).sum())  # noqa: E731
o = np.argsort(i_st)
s0, sx = e_t0[o], e_tx[o]
print("  V4a D650 L5: RMSE G %.1f, T0 mine %.1f (stored %.1f), TX mine %.1f (stored %.1f), own reading %.1f, hold %.1f" % (
    np.sqrt(sse(GP[ti_]) / len(t)), np.sqrt(sse(p0) / len(t)), np.sqrt(sse(s0) / len(t)), np.sqrt(sse(px) / len(t)),
    np.sqrt(sse(sx) / len(t)), np.sqrt(sse(y[ti_]) / len(t)), np.sqrt(sse(H[ti_]) / len(t))))
print("  V4a D650 L5: TX over T0 mine %+.3f, stored %+.3f; T0 over G mine %+.3f" % (
    1 - sse(px) / sse(p0), 1 - sse(sx) / sse(s0), 1 - sse(p0) / sse(GP[ti_])))
print("  |mine - stored| estimates: T0 median %.2f eV, 99th pct %.2f; TX median %.2f eV, 99th pct %.2f" % (
    np.median(np.abs(p0 - s0)), np.percentile(np.abs(p0 - s0), 99), np.median(np.abs(px - sx)), np.percentile(np.abs(px - sx), 99)))

# ---------------------------------------------------------------------------------------------------- part C
print("(C) controls (unregistered), V4a, D650, L5")
rng = np.random.default_rng(75)
Xp = X_TX.copy()
fi, te = np.flatnonzero(fit), ti_
Xp[fi, 10:] = X_TX[rng.permutation(fi), 10:]
Xp[te, 10:] = X_TX[rng.permutation(te), 10:]
pp = fit_predict(Xp, fit, ti_)
print("  placebo (fast and anchor scalars permuted over rows): TX over T0 %+.3f" % (1 - sse(pp) / sse(p0)))
FAST8_anchor = np.where(has[:, None], WF[anc], np.nan)
X_old = np.column_stack([H10, FAST8_anchor, ANC6[:, [0, 3]]])
ok_old = np.isfinite(X_old).all(1)
fo, to = fit & ok_old, test & ok_old
tio = np.flatnonzero(to)
m0 = HistGradientBoostingRegressor(random_state=20261116).fit(X_T0[fo], (L5 - GP)[fo])
mo = HistGradientBoostingRegressor(random_state=20261116).fit(X_old[fo], (L5 - GP)[fo])
mx = HistGradientBoostingRegressor(random_state=20261116).fit(X_TX[fo], (L5 - GP)[fo])
tt = L5[tio]
e0 = ((GP[tio] + m0.predict(X_T0[tio]) - tt) ** 2).sum()
eo = ((GP[tio] + mo.predict(X_old[tio]) - tt) ** 2).sum()
ex = ((GP[tio] + mx.predict(X_TX[tio]) - tt) ** 2).sum()
print("  fast scalars taken at the anchor row (as old as the history), %d test rows: over T0 %+.3f; present fast scalars on the same rows: %+.3f" % (
    len(tio), 1 - eo / e0, 1 - ex / e0))
log("done")
