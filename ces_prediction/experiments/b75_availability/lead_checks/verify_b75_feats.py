# -*- coding: utf-8 -*-
"""Are the session lead's fast and anchor scalars (verify_b75.py) the agent's up to rounding?  Unit V4a, rule D650.
Also: the agent's features with the lead's own fit call reproduce the stored TX estimates.   py verify_b75_feats.py <repo root>"""
import sys
from pathlib import Path
ROOT = Path(sys.argv[1])
sys.path.insert(0, str(ROOT / "ces_prediction" / "experiments" / "b75_availability"))
import common75 as K  # noqa: E402
import numpy as np  # noqa: E402
import stage_trees as ST  # noqa: E402
from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: E402

R = dict(np.load(ROOT / "data" / ".b73" / "rows.npz", allow_pickle=False))
G = dict(np.load(ROOT / "data" / ".b73" / "grid.npz", allow_pickle=False))
S = dict(np.load(ROOT / "data" / ".b75" / "sets.npz", allow_pickle=False))
A = dict(np.load(ROOT / "data" / ".b75" / "inputs" / "D650.npz", allow_pickle=False))
gi, fidx, shot, L5 = R["gi"], R["fidx"], R["shot"], R["tg_L5"]
seg, fast = G["seg"], G["fast"].astype(np.float64)
N = len(seg)
fit_files = np.unique(fidx[shot <= 31991])
# agent's
FE = ST.Feats()
wf = FE.window(fit_files)[gi]
bz, ez = K.zmeans(FE, fit_files)
mrb, mre = (K.run_mean10(x, seg) for x in K.raw_means(G["fast"]))
AN = K.anchor_scalars(gi, A["anc_g"], K.run_mean10(bz, seg), K.run_mean10(ez, seg), mrb, mre)
agent = np.column_stack([wf, AN])
# lead's
frow = np.isin(G["fidx"], fit_files)
mu, sd = fast[frow].mean(0), fast[frow].std(0) + 1e-12
z = (fast - mu) / sd
b2, e2 = z[:, :9].mean(1), z[:, 9:13].mean(1)
br, er = fast[:, :9].mean(1), fast[:, 9:13].mean(1)


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


has = A["anc_g"] >= 0
anc = np.where(has, A["anc_g"], 0)


def a3(mz, mr):
    a1 = np.where(has, mz[anc], np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        lr = np.where(has & (mr[gi] > 0) & (mr[anc] > 0), np.log(mr[gi] / mr[anc]), np.nan)
    return np.column_stack([a1, mz[gi] - a1, lr])


lead = np.column_stack([wfeat(b2)[gi], wfeat(e2)[gi], a3(runmean(b2), runmean(br)), a3(runmean(e2), runmean(er))])
c = S["common_L5"]
d = np.abs(agent[c] - lead[c])
names = ["BES level", "BES departure", "BES sd", "BES increment", "ECEI level", "ECEI departure", "ECEI sd", "ECEI increment",
         "BES a1", "BES a2", "BES a3", "ECEI a1", "ECEI a2", "ECEI a3"]
print("largest |agent - lead| on the common set, per fast input (inputs are in standard units; a3 is a log ratio):")
for j, nm in enumerate(names):
    print("   %-16s %.2e   (sd of the input %.3f)" % (nm, d[:, j].max(), np.std(agent[c, j])))
print("largest over all fourteen: %.2e; same undefined entries: %s" % (d.max(), bool(np.array_equal(np.isfinite(agent), np.isfinite(lead)))))
H10 = K.history_matrix(R["tstep"], A)
fit = c & np.isin(fidx, fit_files)
te = np.flatnonzero(c & (shot >= 32312))
z_ = np.load(ROOT / "data" / ".b75" / "fits" / "V4a__TX__D650__L5.npz")
o = np.argsort(z_["idx"])
for tag, F in (("agent's fast inputs", agent), ("lead's fast inputs", lead)):
    X = np.column_stack([H10, F])
    m = HistGradientBoostingRegressor(random_state=20261116).fit(X[fit], (L5 - A["G"])[fit])
    p = A["G"][te] + m.predict(X[te])
    print("own fit call with the %s: iterations %d, largest |estimate - stored| %.2e eV, RMSE %.1f" % (
        tag, m.n_iter_, np.abs(p - z_["est"][o]).max(), np.sqrt(((p - L5[te]) ** 2).mean())))
