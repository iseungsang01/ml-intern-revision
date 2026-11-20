# -*- coding: utf-8 -*-
"""Session lead's checks of B.79, written independently of the agent's code.
(A) statement values of the main readout (beam columns as delivered) from the stored estimates with own pooling.
(B) the nineteen beam scalars under D650 recomputed for all D1 rows from the NBI files with own code (join by shot and
    stamp), compared with the cached ones as sets of columns; unit V4a, D650, L5: own fits of T0, TN, TX, TXN.
(C) the same unit refitted with the thirteen files whose NBI-2 power column is invalid left out (the lead's treatment).
    py verify_b79.py <repo root>"""
import os
import re
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
B = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "data" / ".b79"      # the cache folder of the as-delivered run
L5, gi, fidx, shot, y = R["tg_L5"], R["gi"], R["fidx"], R["shot"], R["y"]


def est(u, L, rule, tg="L5"):
    z = np.load(B / "fits" / ("%s__%s__%s__%s.npz" % (u, L, rule, tg)))
    return z["idx"], z["est"]


def sse(units, L, rule):
    s = 0.0
    for u in units:
        i, e = est(u, L, rule)
        s += ((e - L5[i]) ** 2).sum()
    return s


def inc(units, rule, a, b):
    return 1 - sse(units, a, rule) / sse(units, b, rule)


V1 = [["V1_r%d_k%d" % (r, k) for k in range(5)] for r in (2, 3, 4)]
V2 = [["V2_s%d_k%d" % (s, k) for k in range(5)] for s in (1, 2, 3)]
V3 = ["V3_q%d" % q for q in (1, 2, 3, 4)]
m3 = lambda g, *a: float(np.mean([inc(u, *a) for u in g]))  # noqa: E731
print("(A) own pooling of the stored estimates (main readout), target L5")
print("  N1 TN over T0, V1, D650 %+.3f | N2 TXN over TN, V1, D650 %+.3f | N3 V2 %+.3f | N4 V3 %+.3f | N5 V4a %+.3f" % (
    m3(V1, "D650", "TN", "T0"), m3(V1, "D650", "TXN", "TN"), m3(V2, "D650", "TXN", "TN"), inc(V3, "D650", "TXN", "TN"), inc(["V4a"], "D650", "TXN", "TN")))
print("  N6 TXN over TN, V1, S %+.3f | N7 V4a, S %+.3f | N8 TN over T0, V4a, S %+.3f | N9 TXN over TX, V1, D650 %+.3f | N10 TXN over TN, V1, D10 %+.3f" % (
    m3(V1, "S", "TXN", "TN"), inc(["V4a"], "S", "TXN", "TN"), inc(["V4a"], "S", "TN", "T0"), m3(V1, "D650", "TXN", "TX"), m3(V1, "D10", "TXN", "TN")))
print("  TN over T0, D650: V2 %+.3f, V3 %+.3f, V4a %+.3f | TXN over TX, D650: V2 %+.3f, V3 %+.3f, V4a %+.3f" % (
    m3(V2, "D650", "TN", "T0"), inc(V3, "D650", "TN", "T0"), inc(["V4a"], "D650", "TN", "T0"),
    m3(V2, "D650", "TXN", "TX"), inc(V3, "D650", "TXN", "TX"), inc(["V4a"], "D650", "TXN", "TX")))

print("(B) own beam scalars under D650")
names = [str(x) for x in G["names"]]
off, tm, seg = G["off"], G["time"].astype(np.float64), G["seg"]
N = len(tm)
COLS = ("NB1B_P", "NB1C_P", "NB2A_P", "NB2B_P", "NB2C_P", "NB1B_V")
P = np.zeros((N, 6))
bad_files = []
worst = 0.0
for i, nm in enumerate(names):
    sh = int(re.search(r"(\d+)", nm).group(1))
    with open(ROOT / "data" / "nbi" / ("s%d_NBI.csv" % sh), encoding="utf-8", errors="replace") as fh:
        hdr = fh.readline().strip().split(",")
        a = np.array([[float(x) if x not in ("", "nan", "NaN") else np.nan for x in ln.rstrip("\n").split(",")] for ln in fh])
    lo, hi = off[i], off[i + 1]
    assert len(a) == hi - lo, "row count differs for shot %d" % sh
    worst = max(worst, float(np.max(np.abs(a[:, 0] - tm[lo:hi]))))
    blk = np.column_stack([a[:, hdr.index(c)] for c in COLS])
    pw = blk[1:-1, :5]
    if np.nanmax(pw) > 5.0 or np.nanmin(pw) < -0.1:
        bad_files.append(sh)
    P[lo:hi] = np.where(np.isfinite(blk), blk, 0.0)
print("  largest stamp difference between a beam row and its grid row: %.1e s; files with an invalid power column: %d %s" % (worst, len(bad_files), bad_files))
p1, p2 = P[:, 0] + P[:, 1], P[:, 2] + P[:, 3] + P[:, 4]
pt = p1 + p2
first = np.zeros(N, bool)
first[off[:-1]] = True
last = np.zeros(N, bool)
last[off[1:] - 1] = True
gridrow = ~first & ~last


def runmean(x, K):
    s, c = np.zeros(N), np.zeros(N)
    for k in range(K):
        j = np.arange(N) - k
        o = (j >= 0) & (seg[np.maximum(j, 0)] == seg) & gridrow[np.maximum(j, 0)]
        s[o] += x[j[o]]
        c[o] += 1
    return np.where(c >= 5, s / np.maximum(c, 1), np.nan)


m10, m50 = runmean(pt, 10), runmean(pt, 50)
A = dict(np.load(ROOT / "data" / ".b75" / "inputs" / "D650.npz", allow_pickle=False))
has = A["anc_g"] >= 0
anc = np.where(has, A["anc_g"], 0)
nanif = lambda v: np.where(has, v, np.nan)  # noqa: E731
mine = np.column_stack(
    [p1[gi], nanif(p1[anc]), p1[gi] - nanif(p1[anc]), p2[gi], nanif(p2[anc]), p2[gi] - nanif(p2[anc]), pt[gi], nanif(pt[anc]), pt[gi] - nanif(pt[anc]),
     m10[gi], m50[gi], nanif(m10[anc]), nanif(m50[anc])] + [(P[gi, k] > 0.1).astype(float) for k in range(5)] + [np.where(P[gi, 0] > 0.1, P[gi, 5], 0.0)])
cached = np.load(B / "beam_D650.npz")["B"]
c = S["common_L5"]
# compare as sets of columns: every cached column must equal one of mine
match = []
for j in range(cached.shape[1]):
    d = [float(np.nanmax(np.abs(cached[c, j] - mine[c, k]))) for k in range(mine.shape[1])]
    match.append(min(d))
print("  cached beam scalars: %d columns; for each, the smallest largest-deviation to one of my 19 columns: max over columns %.2e (rows of the common set: %d; all finite in mine: %s)" % (
    cached.shape[1], max(match), int(c.sum()), bool(np.isfinite(mine[c]).all())))

print("(B, C) own fits, V4a, D650, L5")
H10 = np.column_stack([R["tstep"], A["age"], A["H"], A["G"] - A["H"], np.abs(A["G"] - A["H"]), A["th"]])
fast = G["fast"].astype(np.float64)
fit_files = np.unique(fidx[shot <= 31991])
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


def rm10(x):
    s, cnt = np.zeros(N), np.zeros(N)
    for k in range(10):
        j = np.arange(N) - k
        o = (j >= 0) & (seg[np.maximum(j, 0)] == seg)
        s[o] += x[j[o]]
        cnt[o] += 1
    return np.where(cnt >= 5, s / np.maximum(cnt, 1), np.nan)


def a3(mz, mr):
    a1 = np.where(has, mz[anc], np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        lr = np.where(has & (mr[gi] > 0) & (mr[anc] > 0), np.log(mr[gi] / mr[anc]), np.nan)
    return np.column_stack([a1, mz[gi] - a1, lr])


FX = np.column_stack([wfeat(bz)[gi], wfeat(ez)[gi], a3(rm10(bz), rm10(br)), a3(rm10(ez), rm10(er))])
Gp = A["G"]
X = {"T0": H10, "TN": np.column_stack([H10, mine]), "TX": np.column_stack([H10, FX]), "TXN": np.column_stack([H10, FX, mine])}
bad_rows = np.isin(shot, np.array(bad_files))
for tag, keep in (("columns as delivered", np.ones(len(shot), bool)), ("without the %d files" % len(bad_files), ~bad_rows)):
    fit = c & (shot <= 31991) & keep
    te = np.flatnonzero(c & (shot >= 32312) & keep)
    t = L5[te]
    f = lambda p: float(((p - t) ** 2).sum())  # noqa: E731
    pr = {k: Gp[te] + HistGradientBoostingRegressor(random_state=20261116).fit(X[k][fit], (L5 - Gp)[fit]).predict(X[k][te]) for k in X}
    line = "  %s: fit rows %d, test rows %d; RMSE T0 %.1f, TN %.1f, TX %.1f, TXN %.1f; TN over T0 %+.3f, TXN over TN %+.3f, TX over T0 %+.3f, TXN over TX %+.3f" % (
        tag, int(fit.sum()), len(te), *[np.sqrt(f(pr[k]) / len(t)) for k in ("T0", "TN", "TX", "TXN")],
        1 - f(pr["TN"]) / f(pr["T0"]), 1 - f(pr["TXN"]) / f(pr["TN"]), 1 - f(pr["TX"]) / f(pr["T0"]), 1 - f(pr["TXN"]) / f(pr["TX"]))
    print(line)
    if tag.startswith("columns"):
        o = np.argsort(est("V4a", "T0", "D650")[0])
        st = {k: est("V4a", k, "D650")[1][o] for k in X}
        print("     stored: RMSE T0 %.1f, TN %.1f, TX %.1f, TXN %.1f; TXN over TN %+.3f; largest |mine - stored| T0 %.2e, TN %.2e eV" % (
            *[np.sqrt(f(st[k]) / len(t)) for k in ("T0", "TN", "TX", "TXN")], 1 - f(st["TXN"]) / f(st["TN"]),
            np.abs(pr["T0"] - st["T0"]).max(), np.abs(pr["TN"] - st["TN"]).max()))
