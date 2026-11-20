# -*- coding: utf-8 -*-
"""Session lead's checks of B.81, written independently of the agent's readout.

(A) The seventeen statement values pooled again from the stored estimates (B.75's T0, TX, TE, TB; B.81's TXp, TXo,
    TE, TB), with own pooling: per re-partition or assignment the squared errors of its five folds are summed, the
    skill is formed, and the skills of the three are averaged (V1, V2); V3 pools its four units; V4 is one unit.
(B) The permuted control: for one unit, the stored TXp estimate must not be a function of the row's own fast inputs.
    Read here only as: the correlation between (TXp - T0) and (TX - T0) over the test rows, which is near zero when
    the fast columns were shuffled and near one if they were not.
    py verify_b81.py <repo root>
"""
import os
import sys
from pathlib import Path

for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[v] = "2"
import numpy as np

ROOT = Path(sys.argv[1])
L5 = np.load(ROOT / "data" / ".b73" / "rows.npz", allow_pickle=False)["tg_L5"]
F75, F81 = ROOT / "data" / ".b75" / "fits", ROOT / "data" / ".b81" / "fits"
GROUPS = {
    "V1": [["V1_r%d_k%d" % (r, k) for k in range(5)] for r in (2, 3, 4)],
    "V2": [["V2_s%d_k%d" % (a, k) for k in range(5)] for a in (1, 2, 3)],
    "V3": [["V3_q%d" % q for q in (1, 2, 3, 4)]],
    "V4a": [["V4a"]],
    "V4b": [["V4b"]],
}


def load(unit, L, rule):
    for d in (F81, F75):
        p = d / ("%s__%s__%s__L5.npz" % (unit, L, rule))
        if p.exists():
            z = np.load(p)
            return z["idx"], z["est"]
    raise FileNotFoundError("%s %s %s" % (unit, L, rule))


def skill(design, A, R, rule):
    out = []
    for part in GROUPS[design]:
        sa = sr = 0.0
        for u in part:
            ia, ea = load(u, A, rule)
            ir, er = load(u, R, rule)
            if not np.array_equal(ia, ir):
                raise SystemExit("rows differ: %s %s %s %s" % (u, A, R, rule))
            sa += ((ea - L5[ia]) ** 2).sum()
            sr += ((er - L5[ir]) ** 2).sum()
        out.append(1 - sa / sr)
    return float(np.mean(out))


ST = [("C1", "V1", "TXp", "T0", "D650"), ("C2", "V1", "TXp", "T0", "S"), ("C3", "V1", "TX", "TXo", "D650"),
      ("C4", "V1", "TX", "TXo", "S"), ("C5", "V2", "TX", "TXo", "D650"), ("C6", "V3", "TX", "TXo", "D650"),
      ("C7", "V4a", "TX", "TXo", "D650"), ("C8", "V2", "TX", "TXo", "S"), ("C9", "V3", "TX", "TXo", "S"),
      ("E1", "V2", "TE", "T0", "D650"), ("E2", "V3", "TE", "T0", "D650"), ("E3", "V2", "TB", "T0", "D650"),
      ("E4", "V3", "TB", "T0", "D650"), ("E5", "V2", "TE", "T0", "S"), ("E6", "V3", "TE", "T0", "S"),
      ("E7", "V2", "TB", "T0", "S"), ("E8", "V3", "TB", "T0", "S")]
print("(A) statement values from the stored estimates, own pooling (target L5)")
print("   " + " | ".join("%s %s %s over %s, %s: %+.3f" % (i, d, a, r, ru, skill(d, a, r, ru)) for i, d, a, r, ru in ST[:5]))
print("   " + " | ".join("%s %s %s over %s, %s: %+.3f" % (i, d, a, r, ru, skill(d, a, r, ru)) for i, d, a, r, ru in ST[5:9]))
print("   " + " | ".join("%s %s %s over %s, %s: %+.3f" % (i, d, a, r, ru, skill(d, a, r, ru)) for i, d, a, r, ru in ST[9:13]))
print("   " + " | ".join("%s %s %s over %s, %s: %+.3f" % (i, d, a, r, ru, skill(d, a, r, ru)) for i, d, a, r, ru in ST[13:]))
print("   TXo over T0: " + "; ".join("%s %s" % (ru, ", ".join("%s %+.3f" % (d, skill(d, "TXo", "T0", ru)) for d in GROUPS)) for ru in ("D650", "S")))
print("   TXp over T0: " + "; ".join("%s %s" % (ru, ", ".join("%s %+.3f" % (d, skill(d, "TXp", "T0", ru)) for d in GROUPS)) for ru in ("D650", "S")))
print("(B) correlation over the test rows between the correction of the permuted fit and that of TX (both minus T0)")
for u in ("V1_r2_k0", "V3_q2", "V4a"):
    for ru in ("D650", "S"):
        i0, e0 = load(u, "T0", ru)
        _, ex = load(u, "TX", ru)
        _, ep = load(u, "TXp", ru)
        _, eo = load(u, "TXo", ru)
        c = np.corrcoef(ep - e0, ex - e0)[0, 1]
        c2 = np.corrcoef(eo - e0, ex - e0)[0, 1]
        print("   %-9s %-4s rows %6d: permuted %+.3f; anchor row %+.3f; sd of the correction (eV): TX %.1f, permuted %.1f, anchor row %.1f" % (
            u, ru, len(i0), c, c2, (ex - e0).std(), (ep - e0).std(), (eo - e0).std()))
