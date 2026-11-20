# -*- coding: utf-8 -*-
"""Session lead's checks of B.77, written independently of the agent's code.
(A) statement values from the stored estimates with own pooling; (B) units V4a and V1_r2_k0: C0 and CX refitted from
B.75's cached D10 history inputs, the reported value and own fast and anchor scalars.   py verify_b77.py <repo root>"""
import os
import sys
from pathlib import Path
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[v] = "2"
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = Path(sys.argv[1])
R = dict(np.load(ROOT / "data" / ".b73" / "rows.npz", allow_pickle=False))
G = dict(np.load(ROOT / "data" / ".b73" / "grid.npz", allow_pickle=False))
S = dict(np.load(ROOT / "data" / ".b75" / "sets.npz", allow_pickle=False))
A = dict(np.load(ROOT / "data" / ".b75" / "inputs" / "D10.npz", allow_pickle=False))
y, L5, gi, fidx, shot = R["y"], R["tg_L5"], R["gi"], R["fidx"], R["shot"]
F = ROOT / "data" / ".b77" / "fits"


def est(u, L):
    z = np.load(F / ("%s__%s.npz" % (u, L)))
    return z["idx"], z["est"]


def sk(units, a, b):
    sa = sb = 0.0
    for u in units:
        i, ea = est(u, a) if a != "O" else (est(u, "C0")[0], None)
        ea = y[i] if a == "O" else ea
        j, eb = est(u, b) if b != "O" else (i, y[i])
        assert np.array_equal(i, j)
        sa += ((ea - L5[i]) ** 2).sum()
        sb += ((eb - L5[i]) ** 2).sum()
    return 1 - sa / sb


V1 = [["V1_r%d_k%d" % (r, k) for k in range(5)] for r in (2, 3, 4)]
V2 = [["V2_s%d_k%d" % (s, k) for k in range(5)] for s in (1, 2, 3)]
V3 = ["V3_q%d" % q for q in (1, 2, 3, 4)]
m = lambda us, a, b: np.mean([sk(u, a, b) for u in us])  # noqa: E731
print("(A) own pooling of the stored estimates")
print("  A1 C0 over O, V1 %+.3f | A2 C0 over T0, V1 %+.3f | A3 CX over C0, V1 %+.3f | A4 V2 %+.3f | A5 V3 %+.3f | A6 V4a %+.3f (V4b %+.3f) | A10 CE over C0, V1 %+.3f | placebo CXp over C0, V1 %+.3f"
      % (m(V1, "C0", "O"), m(V1, "C0", "T0"), m(V1, "CX", "C0"), m(V2, "CX", "C0"), sk(V3, "CX", "C0"), sk(["V4a"], "CX", "C0"),
         sk(["V4b"], "CX", "C0"), m(V1, "CE", "C0"), m(V1, "CXp", "C0")))

print("(B) own refit")
tm, seg, fast = G["time"].astype(np.float64), G["seg"], G["fast"].astype(np.float64)
N = len(tm)
rowset = S["complete_D10"] & np.isfinite(L5)
H10 = np.column_stack([R["tstep"], A["age"], A["H"], A["G"] - A["H"], np.abs(A["G"] - A["H"]), A["th"]])
F4 = np.column_stack([y - A["G"], y - A["H"], y - A["th"][:, 0], np.abs(y - A["G"])])
has = A["anc_g"] >= 0
anc = np.where(has, A["anc_g"], 0)


def fast14(fit_files):
    frow = np.isin(G["fidx"], fit_files)
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

    def a3(mz, mr):
        a1 = np.where(has, mz[anc], np.nan)
        with np.errstate(invalid="ignore", divide="ignore"):
            lr = np.where(has & (mr[gi] > 0) & (mr[anc] > 0), np.log(mr[gi] / mr[anc]), np.nan)
        return np.column_stack([a1, mz[gi] - a1, lr])

    return np.column_stack([wfeat(bz)[gi], wfeat(ez)[gi], a3(runmean(bz), runmean(br)), a3(runmean(ez), runmean(er))])


k2 = R["r2_k"]
for name, fit_d, test_d in (("V4a", shot <= 31991, shot >= 32312), ("V1_r2_k0", k2 != 0, k2 == 0)):
    FX = fast14(np.unique(fidx[fit_d]))
    fr, te = rowset & fit_d, np.flatnonzero(rowset & test_d)
    i_st, e0 = est(name, "C0")
    _, ex = est(name, "CX")
    o = np.argsort(i_st)
    assert np.array_equal(te, i_st[o]), "test rows differ"
    X0, XX = np.column_stack([H10, F4]), np.column_stack([H10, F4, FX])
    assert np.isfinite(XX[rowset]).all()
    t = L5[te]
    p0 = A["G"][te] + HistGradientBoostingRegressor(random_state=20261116).fit(X0[fr], (L5 - A["G"])[fr]).predict(X0[te])
    px = A["G"][te] + HistGradientBoostingRegressor(random_state=20261116).fit(XX[fr], (L5 - A["G"])[fr]).predict(XX[te])
    f = lambda p: float(((p - t) ** 2).sum())  # noqa: E731
    print("  %-9s rows %d: C0 largest |mine - stored| %.2e eV; RMSE O %.1f, C0 %.1f, CX mine %.1f (stored %.1f); CX over C0 mine %+.3f, stored %+.3f"
          % (name, len(te), np.abs(p0 - e0[o]).max(), np.sqrt(f(y[te]) / len(t)), np.sqrt(f(p0) / len(t)), np.sqrt(f(px) / len(t)),
             np.sqrt(f(ex[o]) / len(t)), 1 - f(px) / f(p0), 1 - f(ex[o]) / f(e0[o])))
