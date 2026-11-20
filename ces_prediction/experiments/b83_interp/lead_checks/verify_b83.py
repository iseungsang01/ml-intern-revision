# -*- coding: utf-8 -*-
"""Session lead's checks of B.83, written independently of the agent's code.

(A) PCHIP recomputed with own code for every D1 row of 60 discharges drawn at random: through all measured values
    of the row's segment (the cut, held-free values of the grid of B.73, also those of rows that are not D1 rows)
    with the row left out, evaluated at the row's stamp (scipy PchipInterpolator), against the batch's cached
    reference.  A first version of this check took only the D1 rows as neighbours and differed on 532 of 19,289
    rows; the reference reads every measured value of the block, as registered.
(B) The statement values on the value of the frame pooled again from the stored estimates of B.73 and the cached
    references: MSE skills (I1 to I10) and the ratios of the median, mean and 90th percentile of the absolute
    error (I11 to I14), with own pooling (per re-partition or assignment, mean of three under V1 and V2).
    py verify_b83.py <repo root>
"""
import sys
from pathlib import Path

import numpy as np
from scipy.interpolate import PchipInterpolator

ROOT = Path(sys.argv[1])
R = dict(np.load(ROOT / "data" / ".b73" / "rows.npz", allow_pickle=False))
G = dict(np.load(ROOT / "data" / ".b73" / "grid.npz", allow_pickle=False))
Z = dict(np.load(ROOT / "data" / ".b83" / "refs.npz", allow_pickle=False))
y, gi, fidx = R["y"], R["gi"], R["fidx"]
t_g, ti_g, seg_g = G["time"].astype(np.float64), G["ti"].astype(np.float64), G["seg"]
P, O, formed = Z["P"], Z["O"], Z["formed"]

# ---- (A) own PCHIP
rng = np.random.default_rng(831)
files = rng.choice(np.unique(fidx), 60, replace=False)
row_of_grid = {int(g): k for k, g in enumerate(gi)}          # D1 row of a grid index
n_cmp, worst, n_exact = 0, 0.0, 0
meas = np.isfinite(ti_g)                                       # every measured value of the grid (cut, held-free)
for f in files:
    rows = np.flatnonzero(fidx == f)
    for sg in np.unique(seg_g[gi[rows]]):
        g_all = np.flatnonzero(seg_g == sg)
        g_meas = g_all[meas[g_all]]
        if len(g_meas) < 3:
            continue
        tt, vv = t_g[g_meas], ti_g[g_meas]
        for j, g in enumerate(g_meas):
            k = row_of_grid.get(int(g))
            if k is None or not formed[k]:
                continue
            keep = np.ones(len(g_meas), bool)
            keep[j] = False
            if not ((tt[keep] < tt[j]).any() and (tt[keep] > tt[j]).any()):
                continue
            mine = float(PchipInterpolator(tt[keep], vv[keep])(tt[j])) if keep.sum() > 2 else float(np.interp(tt[j], tt[keep], vv[keep]))
            d = abs(mine - P[k])
            worst = max(worst, d)
            n_exact += d == 0
            n_cmp += 1
print("(A) own PCHIP on %d rows of 60 discharges: largest |mine - cached| %.3g eV; exactly equal on %d rows" % (n_cmp, worst, n_exact))

# ---- (B) statements
TR = ROOT / "data" / ".b73" / "trees"
GROUPS = {"V1": [["V1_r%d_k%d" % (r, k) for k in range(5)] for r in (2, 3, 4)],
          "V2": [["V2_s%d_k%d" % (a, k) for k in range(5)] for a in (1, 2, 3)],
          "V3": [["V3_q%d" % q for q in (1, 2, 3, 4)]], "V4a": [["V4a"]]}


def part_rows(part, L):
    I, E = [], []
    for u in part:
        z = np.load(TR / ("%s__%s__F.npz" % (u, L)))
        I.append(z["idx"])
        E.append(z["est"])
    return np.concatenate(I), np.concatenate(E)


def measures(design):
    out = {k: [] for k in ("TX/P", "TX/O", "T0/P", "T0/O", "med TX/O", "med TX/P", "mae TX/O", "p90 TX/O")}
    for part in GROUPS[design]:
        i0, e0 = part_rows(part, "T0")
        ix, ex = part_rows(part, "TX")
        assert np.array_equal(i0, ix)
        m = formed[ix]
        i = ix[m]
        a0, ax, ap, ao = (np.abs(e0[m] - y[i]), np.abs(ex[m] - y[i]), np.abs(P[i] - y[i]), np.abs(O[i] - y[i]))
        out["TX/P"].append(1 - (ax ** 2).sum() / (ap ** 2).sum())
        out["TX/O"].append(1 - (ax ** 2).sum() / (ao ** 2).sum())
        out["T0/P"].append(1 - (a0 ** 2).sum() / (ap ** 2).sum())
        out["T0/O"].append(1 - (a0 ** 2).sum() / (ao ** 2).sum())
        out["med TX/O"].append(np.median(ax) / np.median(ao))
        out["med TX/P"].append(np.median(ax) / np.median(ap))
        out["mae TX/O"].append(ax.mean() / ao.mean())
        out["p90 TX/O"].append(np.percentile(ax, 90) / np.percentile(ao, 90))
    return {k: float(np.mean(v)) for k, v in out.items()}, len(i)


print("(B) statement values from the stored estimates and the cached references (target: the value of the frame)")
for d in ("V1", "V2", "V3", "V4a"):
    v, n = measures(d)
    print("   %-3s rows of one part %6d | MSE skill: TX over PCHIP %+.3f, TX over offline GP %+.3f, T0 over PCHIP %+.3f, T0 over offline GP %+.3f" % (
        d, n, v["TX/P"], v["TX/O"], v["T0/P"], v["T0/O"]))
    if d == "V1":
        print("       ratios TX over offline GP: median %.3f, MAE %.3f, 90th percentile %.3f; TX over PCHIP: median %.3f" % (
            v["med TX/O"], v["mae TX/O"], v["p90 TX/O"], v["med TX/P"]))
