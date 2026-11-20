# -*- coding: utf-8 -*-
"""Unregistered diagnostic beside B.75 (session lead): is the fast-channel increment under a stale history only the
knowledge of when the beam was last interrupted?  The time since the end of the last absent step (schedule information,
known without any diagnostic) is given to the history estimator and to the estimator with fast inputs; the increment is
read again, and by that time (a stratum on an input).
Uses the agent's cached history inputs (checked against own code in verify_b75.py) and own fast scalars.
    py verify_b75_blip.py <repo root>
"""
import os
import sys
from pathlib import Path

for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[v] = "2"
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = Path(sys.argv[1])
B73, B75 = ROOT / "data" / ".b73", ROOT / "data" / ".b75"
R = dict(np.load(B73 / "rows.npz", allow_pickle=False))
S = dict(np.load(B75 / "sets.npz", allow_pickle=False))
G = dict(np.load(B73 / "grid.npz", allow_pickle=False))
tm, seg, off, gfid = G["time"].astype(np.float64), G["seg"], G["off"], G["fidx"]
fast = G["fast"].astype(np.float64)
gi, fidx, shot, L5 = R["gi"], R["fidx"], R["shot"], R["tg_L5"]
N = len(tm)

# time since the end of the last absent step of the block (ms); none -> 5000
dt = np.diff(tm)
same = seg[1:] == seg[:-1]
absent_end = np.zeros(N, bool)
absent_end[1:] = same & (dt > 0.015) & (dt < 0.045)          # row j+1 ends an absent step
last_end = np.where(absent_end, np.arange(N), -1)
last_end = np.maximum.accumulate(last_end)
ok = (last_end >= 0) & (seg[np.maximum(last_end, 0)] == seg)
tsb_all = np.where(ok, (tm - tm[np.maximum(last_end, 0)]) * 1000.0, 5000.0)
tsb_all = np.minimum(tsb_all, 5000.0)
tsb = tsb_all[gi]
print("D1 rows: share within 100 ms after an absent step %.3f, within 300 ms %.3f, none in the block %.3f" % (
    (tsb <= 100).mean(), (tsb <= 300).mean(), (tsb >= 5000).mean()))

NB, NE = 9, 4


def fast_feats(fit_files, anc_g):
    frow = np.isin(gfid, fit_files)
    mu, sd = fast[frow].mean(0), fast[frow].std(0) + 1e-12
    z = (fast - mu) / sd
    bz, ez = z[:, :NB].mean(1), z[:, NB:NB + NE].mean(1)
    br, er = fast[:, :NB].mean(1), fast[:, NB:NB + NE].mean(1)

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

    has = anc_g >= 0
    anc = np.where(has, anc_g, 0)

    def a3(mz, mr):
        a1 = np.where(has, mz[anc], np.nan)
        with np.errstate(invalid="ignore", divide="ignore"):
            lr = np.where(has & (mr[gi] > 0) & (mr[anc] > 0), np.log(mr[gi] / mr[anc]), np.nan)
        return np.column_stack([a1, mz[gi] - a1, lr])

    return np.column_stack([wfeat(bz)[gi], wfeat(ez)[gi], a3(runmean(bz), runmean(br)), a3(runmean(ez), runmean(er))])


def run(rule, setname, units):
    A = dict(np.load(B75 / "inputs" / (rule + ".npz"), allow_pickle=False))
    H10 = np.column_stack([R["tstep"], A["age"], A["H"], A["G"] - A["H"], np.abs(A["G"] - A["H"]), A["th"]])
    inset = S[setname]
    acc = {k: 0.0 for k in ("T0", "T0s", "TX", "TXs")}
    strat = {"le100": dict(acc), "gt100": dict(acc)}
    for name, fit_d, test_d in units:
        fit_files = np.unique(fidx[fit_d])
        FX = fast_feats(fit_files, A["anc_g"])
        X = {"T0": H10, "T0s": np.column_stack([H10, tsb]), "TX": np.column_stack([H10, FX]),
             "TXs": np.column_stack([H10, FX, tsb])}
        fr, te = inset & fit_d, np.flatnonzero(inset & test_d)
        assert np.isfinite(X["TX"][inset]).all()
        t = L5[te]
        for k in X:
            m = HistGradientBoostingRegressor(random_state=20261116).fit(X[k][fr], (L5 - A["G"])[fr])
            e = (A["G"][te] + m.predict(X[k][te]) - t) ** 2
            acc[k] += e.sum()
            strat["le100"][k] += e[tsb[te] <= 100].sum()
            strat["gt100"][k] += e[tsb[te] > 100].sum()
    f = lambda d, a, b: 1 - d[a] / d[b]  # noqa: E731
    print("  %-5s %-22s TX over T0 %+.3f | with the time since the last absent step in both: %+.3f | that time alone over T0: %+.3f"
          % (rule, units[0][0][:2] + (" r2 (5 folds)" if len(units) > 1 else "a"), f(acc, "TX", "T0"), f(acc, "TXs", "T0s"), f(acc, "T0s", "T0")))
    print("        rows within 100 ms after an absent step: TX over T0 %+.3f, with the time in both %+.3f; later rows: %+.3f, %+.3f"
          % (f(strat["le100"], "TX", "T0"), f(strat["le100"], "TXs", "T0s"), f(strat["gt100"], "TX", "T0"), f(strat["gt100"], "TXs", "T0s")))


k2 = R["r2_k"]
V1 = [("V1_r2_k%d" % k, k2 != k, k2 == k) for k in range(5)]
V4 = [("V4a", shot <= 31991, shot >= 32312)]
for rule, setname in (("D650", "common_L5"), ("S", "S_L5")):
    run(rule, setname, V1)
    run(rule, setname, V4)
