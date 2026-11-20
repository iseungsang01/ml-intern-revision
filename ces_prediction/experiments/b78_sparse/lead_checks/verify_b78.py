# -*- coding: utf-8 -*-
"""Session lead's checks of B.78, written independently of the agent's code.
(A) statement values from the stored estimates with own pooling.
(B) rule K10: frame index, kept grid and every history input recomputed for all scored rows with own code, compared
    with the cached inputs; unit V4a refitted (T0 from own inputs; TX with own fast and anchor scalars).
    py verify_b78.py <repo root>"""
import os
import sys
import time
from pathlib import Path
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[v] = "2"
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = Path(sys.argv[1])
sys.path.insert(0, str(ROOT / "ces_prediction"))
import baselines_interpolation as BI  # noqa: E402

R = dict(np.load(ROOT / "data" / ".b73" / "rows.npz", allow_pickle=False))
G = dict(np.load(ROOT / "data" / ".b73" / "grid.npz", allow_pickle=False))
B = ROOT / "data" / ".b78"
S = dict(np.load(B / "sets.npz", allow_pickle=False))
FRS = dict(np.load(B / "frames.npz", allow_pickle=False))
L5, gi, fidx, shot, y = R["tg_L5"], R["gi"], R["fidx"], R["shot"], R["y"]
T0_ = time.time()


def est(u, L, rule, tg="L5"):
    z = np.load(B / "fits" / ("%s__%s__%s__%s.npz" % (u, L, rule, tg)))
    return z["idx"], z["est"]


def sse(units, L, rule):
    s = 0.0
    for u in units:
        i, e = est(u, L, rule)
        s += ((e - L5[i]) ** 2).sum()
    return s


def incr(units, rule, la="TX", lb="T0"):
    return 1 - sse(units, la, rule) / sse(units, lb, rule)


V1 = [["V1_r%d_k%d" % (r, k) for k in range(5)] for r in (2, 3, 4)]
V2 = [["V2_s%d_k%d" % (s, k) for k in range(5)] for s in (1, 2, 3)]
V3 = ["V3_q%d" % q for q in (1, 2, 3, 4)]
print("(A) own pooling of the stored estimates, target L5")
print("  TX over T0 under K10: V1 %+.3f, V2 %+.3f, V3 %+.3f, V4a %+.3f, V4b %+.3f" % (
    np.mean([incr(u, "K10") for u in V1]), np.mean([incr(u, "K10") for u in V2]), incr(V3, "K10"), incr(["V4a"], "K10"), incr(["V4b"], "K10")))
print("  V1 by rule (K1, K2, K5, K10, K20, K50): " + ", ".join("%+.3f" % np.mean([incr(u, k) for u in V1]) for k in ("K1", "K2", "K5", "K10", "K20", "K50")))
print("  K-5: 1 - MSE(TX, K10) / MSE(T0, K5), V1: %+.3f | K-6: MSE(TX, K10) / MSE(TX, K1), V1: %.3f | K-7: TE over T0, V1, K10: %+.3f" % (
    np.mean([1 - sse(u, "TX", "K10") / sse(u, "T0", "K5") for u in V1]), np.mean([sse(u, "TX", "K10") / sse(u, "TX", "K1") for u in V1]),
    np.mean([incr(u, "K10", la="TE") for u in V1])))
n1 = sum(len(est(u, "T0", "K10")[0]) for u in V1[0])
print("  scored rows of one re-partition: %d; common set L5: %d" % (n1, int(S["common_L5"].sum())))

print("(B) own frame index, kept grid and K10 inputs")
tm, ti, seg, off, gfid = G["time"].astype(np.float64), G["ti"].astype(np.float64), G["seg"], G["off"], G["fidx"]
fast = G["fast"].astype(np.float64)
N = len(tm)
first_of_file = np.zeros(N, bool)
first_of_file[off[:-1]] = True
last_of_file = np.zeros(N, bool)
last_of_file[off[1:] - 1] = True
fr = np.full(N, -1, np.int64)
for s_ in np.unique(seg):
    idx = np.flatnonzero((seg == s_) & ~first_of_file & ~last_of_file)
    if len(idx):
        fr[idx] = np.rint((tm[idx] - tm[idx[0]]) / 0.010).astype(np.int64)
print("  frame index equal to the cached one: %s" % bool(np.array_equal(fr, FRS["fr"])))
sc = (fr[gi] % 2 == 1) & (fr[gi] % 5 != 0) & (fr[gi] >= 0)
print("  scored rows: mine %d, cached %d, equal: %s" % (int(sc.sum()), int(FRS["scored"].sum()), bool(np.array_equal(sc, FRS["scored"]))))
K = 10
n = len(gi)
H, AGE, GP = (np.full(n, np.nan) for _ in range(3))
TH = np.full((n, 5), np.nan)
ANC = np.full(n, -1, np.int64)
for i in np.unique(fidx[sc]):
    a, b = off[i], off[i + 1]
    t_, v_, s_, f_ = tm[a:b], ti[a:b], seg[a:b], fr[a:b]
    kept = np.isfinite(v_) & (f_ >= 0) & (f_ % K == 0)
    for q in np.flatnonzero(sc & (fidx == i)):
        g = gi[q] - a
        vis = np.flatnonzero(kept & (s_ == s_[g]) & (np.arange(b - a) < g))
        if len(vis) == 0:
            continue
        last = vis[-1]
        H[q], AGE[q], ANC[q] = v_[last], (t_[g] - t_[last]) * 1000.0, a + last
        v10 = v_[vis[-10:]]
        if len(v10) >= 3:
            TH[q] = (v10.mean(), v10.std(), v10[-1] - v10.mean(), v10[-1] - v10[-2], len(v10))
        v16 = vis[-16:]
        GP[q] = BI.predict_gp_causal(t_[v16], v_[v16], t_[g])
A = dict(np.load(B / "inputs" / "K10.npz", allow_pickle=False))
m = sc
both = m & np.isfinite(GP)
print("  [%4.0fs] K10 on the scored rows: rows with a visible value mine %d, cached %d; largest |mine - cached|: H %.2e eV, age %.2e ms, G %.2e eV, history scalars %.2e; anchor rows equal: %s" % (
    time.time() - T0_, int(np.isfinite(H[m]).sum()), int(np.isfinite(A["H"][m]).sum()), np.nanmax(np.abs(H[m] - A["H"][m])),
    np.nanmax(np.abs(AGE[m] - A["age"][m])), np.max(np.abs(GP[both] - A["G"][both])), np.nanmax(np.abs(TH[m] - A["th"][m])),
    bool(np.array_equal(ANC[m], A["anc_g"][m]))))

fit_files = np.unique(fidx[shot <= 31991])
frow = np.isin(gfid, fit_files)
mu, sd = fast[frow].mean(0), fast[frow].std(0) + 1e-12
z = (fast - mu) / sd
bz, ez, br, er = z[:, :9].mean(1), z[:, 9:13].mean(1), fast[:, :9].mean(1), fast[:, 9:13].mean(1)


def wfeat(x):
    out = np.full((N, 4), np.nan)
    idx = np.arange(10, N)
    idx = idx[seg[idx] == seg[idx - 10]]
    win = np.stack([x[idx - k] for k in range(1, 11)], 1)
    out[idx, 0], out[idx, 1], out[idx, 2], out[idx, 3] = x[idx], x[idx] - win.mean(1), win.std(1), x[idx] - x[idx - 1]
    return out


def runmean(x):
    s, c = np.zeros(N), np.zeros(N)
    for k in range(10):
        j = np.arange(N) - k
        o = (j >= 0) & (seg[np.maximum(j, 0)] == seg)
        s[o] += x[j[o]]
        c[o] += 1
    return np.where(c >= 5, s / np.maximum(c, 1), np.nan)


has = ANC >= 0
anc = np.where(has, ANC, 0)


def a3(mz, mr):
    a1 = np.where(has, mz[anc], np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        lr = np.where(has & (mr[gi] > 0) & (mr[anc] > 0), np.log(mr[gi] / mr[anc]), np.nan)
    return np.column_stack([a1, mz[gi] - a1, lr])


FX = np.column_stack([wfeat(bz)[gi], wfeat(ez)[gi], a3(runmean(bz), runmean(br)), a3(runmean(ez), runmean(er))])
H10 = np.column_stack([R["tstep"], AGE, H, GP - H, np.abs(GP - H), TH])
c = S["common_L5"]
XT, XX = H10, np.column_stack([H10, FX])
print("  my K10 inputs finite on the cached common set: %s (%d rows)" % (bool(np.isfinite(XX[c]).all()), int(c.sum())))
fit, te = c & (shot <= 31991), np.flatnonzero(c & (shot >= 32312))
i_st, e0 = est("V4a", "T0", "K10")
_, ex = est("V4a", "TX", "K10")
o = np.argsort(i_st)
assert np.array_equal(te, i_st[o])
t = L5[te]
p0 = GP[te] + HistGradientBoostingRegressor(random_state=20261116).fit(XT[fit], (L5 - GP)[fit]).predict(XT[te])
px = GP[te] + HistGradientBoostingRegressor(random_state=20261116).fit(XX[fit], (L5 - GP)[fit]).predict(XX[te])
f = lambda p: float(((p - t) ** 2).sum())  # noqa: E731
print("  V4a, K10 (%d test rows): T0 largest |mine - stored| %.2e eV; RMSE hold %.1f, G %.1f, T0 %.1f, TX mine %.1f (stored %.1f), own reading %.1f; TX over T0 mine %+.3f, stored %+.3f" % (
    len(te), np.abs(p0 - e0[o]).max(), np.sqrt(f(H[te]) / len(t)), np.sqrt(f(GP[te]) / len(t)), np.sqrt(f(p0) / len(t)), np.sqrt(f(px) / len(t)),
    np.sqrt(f(ex[o]) / len(t)), np.sqrt(f(y[te]) / len(t)), 1 - f(px) / f(p0), 1 - f(ex[o]) / f(e0[o])))
