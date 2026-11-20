# -*- coding: utf-8 -*-
"""Session lead's checks of B.76 (MAST), written independently of the agent's row builder and readout.
(A) statement values from the stored estimates with own pooling.
(B) 24 random shots: block, good values, history under D100, causal GP, L5 and the Thomson window values at frame t and
    at the anchor frame recomputed from the shot files with own code on the shared loaders; compared with the row table.
(C) design M4 (chronological), rule D100, target L5: own fits of T0 and TX from the row table; unregistered controls:
    Thomson scalars permuted; T_e scalars only; n_e scalars only; without the edge distance; the edge distance only.
    py verify_b76.py <repo root>"""
import json
import os
import sys
from pathlib import Path

for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[v] = "2"
import numpy as np

ROOT = Path(sys.argv[1])
sys.path.insert(0, str(ROOT / "ces_prediction" / "experiments" / "mast"))
sys.path.insert(0, str(ROOT / "ces_prediction"))
import run_gate as rg  # noqa: E402
import run_coupling as rc  # noqa: E402
import explore_ped as ep  # noqa: E402
import baselines_interpolation as BI  # noqa: E402
from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: E402

B = ROOT / "data" / ".b76"
z = np.load(B / "rows.npz", allow_pickle=False)
R = {k: z[k] for k in z.files}
S = dict(np.load(B / "sets.npz", allow_pickle=False))
U = json.loads((B / "units.json").read_text(encoding="utf-8"))
units = U["units"] if "units" in U else [v for v in U.values() if isinstance(v, list)][0]
y, L5, shot = R["y"], R["L5"], R["shot"]


def est(u, L, rule, tg="L5"):
    f = np.load(B / "fits" / ("%s__%s__%s__%s.npz" % (u, L, rule, tg)))
    return f["idx"], f["est"]


def sse(names, L, rule):
    s = 0.0
    for u in names:
        if L in ("O", "H"):
            i = est(u, "T0", rule)[0]
            e = y[i] if L == "O" else R[rule + "_H"][i]
        else:
            i, e = est(u, L, rule)
        s += ((e - L5[i]) ** 2).sum()
    return s


def inc(us, rule, a="TX", b="T0"):
    return 1 - sse(us, a, rule) / sse(us, b, rule)


def mean3(g, *a):
    return float(np.mean([inc(u, *a) for u in g]))


names = [u["name"] for u in units]
M1 = [[n for n in names if n.startswith("M1_s%d_" % s)] for s in (1, 2, 3)]
M2 = [[n for n in names if n.startswith("M2_s%d_" % s)] for s in (1, 2, 3)]
M3 = [n for n in names if n.startswith("M3")]
M4 = [n for n in names if n.startswith("M4")]
print("(A) own pooling of the stored estimates, target L5; units: %d, M3: %s, M4: %s" % (len(names), M3, M4))
print("  TX over T0, D100: M1 %+.3f | M2 %+.3f | M3 %+.3f | M4 %+.3f ;  D5: M1 %+.3f" % (
    mean3(M1, "D100"), mean3(M2, "D100"), inc(M3, "D100"), inc(M4, "D100"), mean3(M1, "D5")))
print("  S1 MSE(T0, D100) / MSE(T0, D5), M1: %.3f | S7 RTa over H, M1, D100: %+.3f | S10 TX over O, M1, D5: %+.3f" % (
    float(np.mean([sse(u, "T0", "D100") / sse(u, "T0", "D5") for u in M1])), mean3(M1, "D100", "RTa", "H"), mean3(M1, "D5", "TX", "O")))
print("  placebo TXp over T0, M1, D100: %+.3f | TXo over T0: %+.3f" % (mean3(M1, "D100", "TXp", "T0"), mean3(M1, "D100", "TXo", "T0")))

print("(B) 24 shots recomputed from the shot files")
REL, FRAC, STEP, TOL = rg.POPS["W"], 0.5, rg.STEP_MS, 0.5
rng = np.random.default_rng(76)
cs = np.unique(shot[S["common_L5"]])
pick = rng.choice(cs, 24, replace=False)
dev = {k: 0.0 for k in ("y", "L5", "H", "age", "G", "th", "te_t", "te_a", "ne_t")}
nrow = 0
for s in pick:
    s = int(s)
    d = rg.load_shot(s)
    a, b = rc.block_of(d, "T_i", REL)
    good = rg.good_masks(d, REL)["T_i"][a:b + 1]
    L = b - a + 1
    Y = d["T"][a:b + 1].astype(np.float64)
    t_abs = float(d["time_ms"][a]) + STEP * np.arange(L)
    tt, RR, te, ne, _ = ep.load_extra(s)["ts"]
    for q in np.flatnonzero(shot == s):
        craw, j = int(R["craw"][q]), int(R["j"][q])
        g, v = good[:, craw], Y[:, craw]
        assert g[j]
        dev["y"] = max(dev["y"], abs(v[j] - y[q]))
        nx = [v[j + k] for k in range(1, 6) if j + k < L and g[j + k]]
        l5 = float(np.median(nx)) if len(nx) >= 3 else np.nan
        if np.isfinite(l5) != np.isfinite(L5[q]):
            raise SystemExit("L5 defined in one build only")
        if np.isfinite(l5):
            dev["L5"] = max(dev["L5"], abs(l5 - L5[q]))
        vis = [i for i in range(0, j) if g[i] and STEP * i <= STEP * j - 100.0 + TOL]
        if not vis:
            assert R["D100_anc"][q] < 0
            continue
        last = vis[-1]
        assert last == R["D100_anc"][q]
        dev["H"] = max(dev["H"], abs(v[last] - R["D100_H"][q]))
        dev["age"] = max(dev["age"], abs(STEP * (j - last) - R["D100_age"][q]))
        v10 = v[vis[-10:]]
        if len(v10) >= 3:
            mine = np.array([v10.mean(), v10.std(), v10[-1] - v10.mean(), v10[-1] - v10[-2], len(v10)])
            dev["th"] = max(dev["th"], float(np.abs(mine - R["D100_th"][q]).max()))
        v16 = np.array(vis[-16:])
        gp = BI.predict_gp_causal(STEP * v16 / 1000.0, v[v16], STEP * j / 1000.0)
        dev["G"] = max(dev["G"], abs(gp - R["D100_G"][q]))
        Rc = np.array([float(d["R"][craw])])
        for fr, key, col in ((j, "te_t", R["te_t"][q]), (last, "te_a", R["D100_te_a"][q])):
            w = np.flatnonzero((tt >= t_abs[fr] - 2.5) & (tt < t_abs[fr] + 2.5))
            vals = np.array([ep.to_chords(Rc, RR[i][np.isfinite(te[i])], te[i][np.isfinite(te[i])])[0] for i in w])
            mine = float(np.nanmean(vals)) if len(vals) and np.isfinite(vals).any() else np.nan
            if np.isfinite(mine) != np.isfinite(col):
                raise SystemExit("Thomson value defined in one build only (%s)" % key)
            if np.isfinite(mine):
                dev[key] = max(dev[key], abs(mine - col))
        w = np.flatnonzero((tt >= t_abs[j] - 2.5) & (tt < t_abs[j] + 2.5))
        vals = np.array([ep.to_chords(Rc, RR[i][np.isfinite(ne[i])], ne[i][np.isfinite(ne[i])])[0] for i in w])
        dev["ne_t"] = max(dev["ne_t"], abs(float(np.nanmean(vals)) / 1e19 - R["ne_t"][q]))
        nrow += 1
print("  rows compared: %d of %d shots; largest |mine - row table|: " % (nrow, len(pick)) + ", ".join("%s %.2e" % kv for kv in dev.items()))

print("(C) own fits, M4, D100, L5")
r = "D100"
Hm = np.column_stack([R[r + "_age"], R[r + "_H"], R[r + "_G"] - R[r + "_H"], np.abs(R[r + "_G"] - R[r + "_H"]), R[r + "_th"], R["rho"]])
te_t, ne_t, te_a, ne_a = R["te_t"], R["ne_t"], R[r + "_te_a"], R[r + "_ne_a"]
with np.errstate(all="ignore"):
    Tm = np.column_stack([te_t, ne_t, te_a, ne_a, te_t - te_a, ne_t - ne_a, np.log(te_t / te_a), np.log(ne_t / ne_a), R["dedge_t"]])
c = S["common_L5"]
u4 = [u for u in units if u["name"] == M4[0]][0]
fit = c & np.isin(shot, np.array(u4["fit"]))
te_idx = np.flatnonzero(c & np.isin(shot, np.array(u4["test"])))
i_st, e0 = est(M4[0], "T0", r)
_, ex_ = est(M4[0], "TX", r)
o = np.argsort(i_st)
assert np.array_equal(te_idx, i_st[o]), "test rows differ"
G_, t = R[r + "_G"], L5[te_idx]


def fp(X):
    mdl = HistGradientBoostingRegressor(random_state=20261116).fit(X[fit], (L5 - G_)[fit])
    return G_[te_idx] + mdl.predict(X[te_idx])


def f(p):
    return float(((p - t) ** 2).sum())


p0, px = fp(Hm), fp(np.column_stack([Hm, Tm]))
print("  fit rows %d, test rows %d (shots fit %d, test %d)" % (int(fit.sum()), len(te_idx), len(u4["fit"]), len(u4["test"])))
print("  largest |mine - stored|: T0 %.2e eV, TX %.2e eV; RMSE hold %.1f, G %.1f, T0 %.1f, TX %.1f, own reading %.1f; TX over T0 mine %+.3f, stored %+.3f" % (
    np.abs(p0 - e0[o]).max(), np.abs(px - ex_[o]).max(), np.sqrt(f(R[r + "_H"][te_idx]) / len(t)), np.sqrt(f(G_[te_idx]) / len(t)),
    np.sqrt(f(p0) / len(t)), np.sqrt(f(px) / len(t)), np.sqrt(f(y[te_idx]) / len(t)), 1 - f(px) / f(p0), 1 - f(ex_[o]) / f(e0[o])))
rng = np.random.default_rng(761)
Tp = Tm.copy()
fi = np.flatnonzero(fit)
Tp[fi] = Tm[rng.permutation(fi)]
Tp[te_idx] = Tm[rng.permutation(te_idx)]
print("  controls (unregistered, no interval), skill over T0: Thomson scalars permuted %+.3f | T_e scalars only %+.3f | n_e scalars only %+.3f | all but the edge distance %+.3f | edge distance only %+.3f" % (
    1 - f(fp(np.column_stack([Hm, Tp]))) / f(p0), 1 - f(fp(np.column_stack([Hm, Tm[:, [0, 2, 4, 6]]]))) / f(p0),
    1 - f(fp(np.column_stack([Hm, Tm[:, [1, 3, 5, 7]]]))) / f(p0), 1 - f(fp(np.column_stack([Hm, Tm[:, :8]]))) / f(p0),
    1 - f(fp(np.column_stack([Hm, Tm[:, [8]]]))) / f(p0)))
