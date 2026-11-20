# -*- coding: utf-8 -*-
"""Session lead's checks of B.80, written independently of the agent's code: the coherence of the four ECEI channels
per discharge recomputed from the grid, the class counts, and the class skills and the gated estimator under V1 and
rule S from the stored estimates of B.75.    py verify_b80.py <repo root>"""
import os
import sys
from pathlib import Path

for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[v] = "2"
import numpy as np

ROOT = Path(sys.argv[1])
R = dict(np.load(ROOT / "data" / ".b73" / "rows.npz", allow_pickle=False))
G = dict(np.load(ROOT / "data" / ".b73" / "grid.npz", allow_pickle=False))
L5, fidx, shot = R["tg_L5"], R["fidx"], R["shot"]
off, fast = G["off"], G["fast"].astype(np.float64)

ce = {}
for f in np.unique(fidx):
    a, b = off[f] + 1, off[f + 1] - 1            # without the first and last row of the file
    if b - a < 50:
        continue
    E = fast[a:b, 9:13]
    if (E.std(0) == 0).any() or not np.isfinite(E).all():
        continue
    c = np.corrcoef(E.T)
    ce[int(f)] = float(c[np.triu_indices(4, 1)].mean())
cls = np.full(len(fidx), -1)
cv = np.array([ce.get(int(f), np.nan) for f in fidx])
cls[cv < 0.7] = 0
cls[(cv >= 0.7) & (cv < 0.9)] = 1
cls[cv >= 0.9] = 2
d1 = np.unique(fidx)
dc = np.array([ce.get(int(f), np.nan) for f in d1])
sh1 = np.array([shot[fidx == f][0] for f in d1])
print("D1 discharges with the measure: %d of %d; low / middle / high: %d / %d / %d; median %.3f" % (
    np.isfinite(dc).sum(), len(d1), (dc < 0.7).sum(), ((dc >= 0.7) & (dc < 0.9)).sum(), (dc >= 0.9).sum(), np.nanmedian(dc)))
for nm, lo, hi in (("up to 31991", 0, 31991), ("32002-32310", 32002, 32310), ("from 32312", 32312, 10 ** 9)):
    m = (sh1 >= lo) & (sh1 <= hi) & np.isfinite(dc)
    print("   %-12s low / middle / high: %d / %d / %d; median %.3f; share low %.3f" % (
        nm, (dc[m] < 0.7).sum(), ((dc[m] >= 0.7) & (dc[m] < 0.9)).sum(), (dc[m] >= 0.9).sum(), np.median(dc[m]), (dc[m] < 0.7).mean()))


def est(u, L, rule):
    z = np.load(ROOT / "data" / ".b75" / "fits" / ("%s__%s__%s__L5.npz" % (u, L, rule)))
    return z["idx"], z["est"]


for rule in ("S", "D650"):
    sk = {k: [] for k in ("lo", "mid", "hi", "tg_t0", "tg_tx")}
    for r in (2, 3, 4):
        s0 = np.zeros(3)
        sx = np.zeros(3)
        t0a = txa = tga = 0.0
        for k in range(5):
            i, e0 = est("V1_r%d_k%d" % (r, k), "T0", rule)
            _, ex = est("V1_r%d_k%d" % (r, k), "TX", rule)
            q0, qx = (e0 - L5[i]) ** 2, (ex - L5[i]) ** 2
            for c_ in range(3):
                s0[c_] += q0[cls[i] == c_].sum()
                sx[c_] += qx[cls[i] == c_].sum()
            g = np.where(cls[i] >= 1, qx, q0)          # TG: TX where C_E is 0.7 or above, T0 otherwise
            t0a, txa, tga = t0a + q0.sum(), txa + qx.sum(), tga + g.sum()
        for c_, nm in enumerate(("lo", "mid", "hi")):
            sk[nm].append(1 - sx[c_] / s0[c_])
        sk["tg_t0"].append(1 - tga / t0a)
        sk["tg_tx"].append(1 - tga / txa)
    print("V1, rule %s: TX over T0 by class low / middle / high: %+.3f / %+.3f / %+.3f; high minus low %+.3f; TG over T0 %+.3f; TG over TX %+.3f" % (
        rule, np.mean(sk["lo"]), np.mean(sk["mid"]), np.mean(sk["hi"]), np.mean(sk["hi"]) - np.mean(sk["lo"]), np.mean(sk["tg_t0"]), np.mean(sk["tg_tx"])))
