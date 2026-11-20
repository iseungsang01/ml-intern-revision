# -*- coding: utf-8 -*-
"""Session lead's checks of B.84, written independently of the agent's readout: the eleven statement values and the
headline RMSE values pooled again from the stored estimates (B.75 and B.78 per-rule fits, B.84 mixture fits).
    py verify_b84.py <repo root>
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(sys.argv[1])
L5 = np.load(ROOT / "data" / ".b73" / "rows.npz", allow_pickle=False)["tg_L5"]
F75, F78, F84 = (ROOT / "data" / d / "fits" for d in (".b75", ".b78", ".b84"))
UNITS = {"V1": ["V1_r2_k%d" % k for k in range(5)], "V3": ["V3_q%d" % q for q in (1, 2, 3, 4)], "V4a": ["V4a"]}


def load(unit, L, rule):
    cands = [F84 / ("%s__%s__%s.npz" % (unit, L, rule)), F84 / ("%s__%s__%s__L5.npz" % (unit, L, rule)),
             F75 / ("%s__%s__%s__L5.npz" % (unit, L, rule)), F78 / ("%s__%s__%s__L5.npz" % (unit, L, rule))]
    for p in cands:
        if p.exists():
            z = np.load(p)
            return z["idx"], z["est"]
    raise FileNotFoundError("%s %s %s" % (unit, L, rule))


def sse(design, L, rule, rows=None):
    """Pooled squared error and row index over the units of the design."""
    I, E = [], []
    for u in UNITS[design]:
        i, e = load(u, L, rule)
        I.append(i)
        E.append((e - L5[i]) ** 2)
    return np.concatenate(I), np.concatenate(E)


def skill(design, A, R, rule):
    ia, ea = sse(design, A, rule)
    ir, er = sse(design, R, rule)
    if not np.array_equal(ia, ir):
        # estimators stored on different row sets: compare on the common rows, in the order of A
        common = np.intersect1d(ia, ir)
        ma, mr = np.isin(ia, common), np.isin(ir, common)
        oa, orr = np.argsort(ia[ma]), np.argsort(ir[mr])
        return 1 - ea[ma][oa].sum() / er[mr][orr].sum(), len(common)
    return 1 - ea.sum() / er.sum(), len(ia)


def rmse(design, L, rule):
    try:
        _, e = sse(design, L, rule)
        return "%.1f" % np.sqrt(e.mean())
    except FileNotFoundError:
        return "-"


ST = [("U1", "V1", "TXall", "TX", "D650"), ("U2", "V1", "TXall", "TX", "D10"), ("U3", "V1", "TXall", "TX", "S"),
      ("U4", "V1", "TXall", "TXa", "D650"), ("U5", "V1", "TXall", "T0all", "D650"), ("U6", "V1", "TXall", "T0", "K10"),
      ("U7", "V3", "TXall", "TX", "D650"), ("U8", "V1", "TXraw", "TXall", "D650"), ("U9", "V3", "TXrel", "TX", "D650"),
      ("U10", "V4a", "TXrel", "T0", "S"), ("U11", "V1", "TXrel", "TX", "D650")]
print("(A) statement values from the stored estimates, own pooling (target L5; V1 = re-partition r2)")
for i, d, a, r, ru in ST:
    v, n = skill(d, a, r, ru)
    print("   %-3s %-3s %-5s over %-5s under %-4s: %+.3f  (%d rows)" % (i, d, a, r, ru, v, n))
print("(B) RMSE (eV), per-rule TX / applied TXa / TXall / TXraw / TXrel")
for d in ("V1", "V3", "V4a"):
    for ru in ("D10", "D100", "D650", "D1000", "S", "K10"):
        print("   %-3s %-5s %s / %s / %s / %s / %s" % (d, ru, rmse(d, "TX", ru), rmse(d, "TXa", ru), rmse(d, "TXall", ru), rmse(d, "TXraw", ru), rmse(d, "TXrel", ru)))
print("(C) the increment inside the one fit, TXall over T0all, V1")
print("   " + ", ".join("%s %+.3f" % (ru, skill("V1", "TXall", "T0all", ru)[0]) for ru in ("D10", "D50", "D100", "D300", "D650", "D1000", "S", "K10")))
