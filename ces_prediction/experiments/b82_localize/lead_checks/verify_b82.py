# -*- coding: utf-8 -*-
"""Session lead's checks of B.82, written independently of the agent's code.

(A) tau (time since the first row after the most recent absent step) recomputed from the grid stamps with own code,
    its classes, and the increment of TX over T0 by class under V1 and V3 from the stored estimates of B.75.
(B) Robust measures of T0, TX and O for V1, D650 from the stored estimates (own pooling: the measure per
    re-partition on its pooled rows, mean over the three re-partitions).
(C) Specificity controls pooled again from the stored fits of this batch (TM, TXM, TXs, the three BES columns).
(D) Seed spread of the chronological fit.
    py verify_b82.py <repo root>
"""
import os
import sys
from pathlib import Path

for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[v] = "2"
import numpy as np

ROOT = Path(sys.argv[1])
R = dict(np.load(ROOT / "data" / ".b73" / "rows.npz", allow_pickle=False))
G = dict(np.load(ROOT / "data" / ".b73" / "grid.npz", allow_pickle=False))
S = dict(np.load(ROOT / "data" / ".b75" / "sets.npz", allow_pickle=False))
L5, gi, y = R["tg_L5"], R["gi"], R["y"]
t, seg = G["time"].astype(np.float64), G["seg"]
N = len(t)
F75, F81, F82 = (ROOT / "data" / d / "fits" for d in (".b75", ".b81", ".b82"))

# ---- own tau: for every grid row the stamp of the newest "first row after an absent step" at or before it, same segment
dt = np.zeros(N)
dt[1:] = t[1:] - t[:-1]
same = np.zeros(N, bool)
same[1:] = seg[1:] == seg[:-1]
after = same & (dt > 0.015) & (dt < 0.045)            # row i is the first row after an absent step
last = np.full(N, -1, np.int64)
cur = -1
for i in range(N):
    if i == 0 or not same[i]:
        cur = -1
    if after[i]:
        cur = i
    last[i] = cur
tau_g = np.where(last >= 0, (t - t[np.maximum(last, 0)]) * 1000.0, np.nan)
tau = tau_g[gi]
cls = np.where(np.isnan(tau), 3, np.where(tau <= 103, 0, np.where(tau <= 303, 1, 2)))
CN = ("up to 100 ms", "100 to 300 ms", "above 300 ms", "no notch before")
z = np.load(ROOT / "data" / ".b82" / "tau.npz")
key = [k for k in z.files if "tau" in k][0]
mine_vs = np.nanmax(np.abs(np.where(np.isnan(tau), 0, tau) - np.where(np.isnan(z[key]), 0, z[key]))) if z[key].shape == tau.shape else float("nan")
print("(A) tau: largest |mine - cached| over the D1 rows %.3g ms; rows with no notch before: mine %d, cached %d" % (
    mine_vs, np.isnan(tau).sum(), np.isnan(z[key]).sum() if z[key].shape == tau.shape else -1))

GROUPS = {"V1": [["V1_r%d_k%d" % (r, k) for k in range(5)] for r in (2, 3, 4)],
          "V2": [["V2_s%d_k%d" % (a, k) for k in range(5)] for a in (1, 2, 3)],
          "V3": [["V3_q%d" % q for q in (1, 2, 3, 4)]], "V4a": [["V4a"]], "V4b": [["V4b"]]}


def load(unit, L, rule):
    for d in (F82, F81, F75):
        p = d / ("%s__%s__%s__L5.npz" % (unit, L, rule))
        if p.exists():
            zz = np.load(p)
            return zz["idx"], zz["est"]
    raise FileNotFoundError("%s %s %s" % (unit, L, rule))


def parts(design, L, rule):
    """Per re-partition: (idx, est) concatenated over its units."""
    out = []
    for part in GROUPS[design]:
        ii, ee = zip(*(load(u, L, rule) for u in part))
        out.append((np.concatenate(ii), np.concatenate(ee)))
    return out


def skill(design, A, Rf, rule, mask=None):
    v = []
    for (ia, ea), (ir, er) in zip(parts(design, A, rule), parts(design, Rf, rule)):
        assert np.array_equal(ia, ir)
        m = np.ones(len(ia), bool) if mask is None else mask[ia]
        v.append(1 - ((ea - L5[ia]) ** 2)[m].sum() / ((er - L5[ir]) ** 2)[m].sum())
    return float(np.mean(v))


for design in ("V1", "V3"):
    for rule in ("D650", "S"):
        cells = []
        for c in range(4):
            m = cls == c
            n = int(m[parts(design, "T0", rule)[0][0]].sum())
            cells.append("%s %+.3f (%d rows)" % (CN[c], skill(design, "TX", "T0", rule, m), n) if n > 0 else "%s -" % CN[c])
        print("   %s %-4s TX over T0: %s" % (design, rule, "; ".join(cells)))
m_hi, m_lo = cls == 0, cls == 2
print("   A5 V1, S: up to 100 ms minus above 300 ms: %+.3f | D650: %+.3f" % (
    skill("V1", "TX", "T0", "S", m_hi) - skill("V1", "TX", "T0", "S", m_lo),
    skill("V1", "TX", "T0", "D650", m_hi) - skill("V1", "TX", "T0", "D650", m_lo)))

# ---- (B) robust measures, V1, D650
print("(B) V1, D650, own pooling (measure per re-partition, mean of three)")
res = {}
for L in ("T0", "TX", "O"):
    acc = []
    for (i0, e0), (ix, ex) in zip(parts("V1", "T0", "D650"), parts("V1", "TX", "D650")):
        e = {"T0": e0, "TX": ex, "O": y[i0]}[L]
        a = np.abs(e - L5[i0])
        rel = a / L5[i0]
        ok = L5[i0] > 100
        acc.append((a.mean(), np.median(a), np.percentile(a, 90), (rel[ok] <= 0.10).mean(), (rel[ok] <= 0.20).mean()))
    res[L] = np.mean(acc, 0)
    print("   %-2s MAE %.1f eV, median %.1f eV, 90th percentile %.1f eV, within 10 %% %.3f, within 20 %% %.3f" % ((L,) + tuple(res[L])))
print("   ratios TX over T0: MAE %.3f, median %.3f, 90th percentile %.3f; share within 10 %%, TX minus T0: %+.3f" % (
    res["TX"][0] / res["T0"][0], res["TX"][1] / res["T0"][1], res["TX"][2] / res["T0"][2], res["TX"][3] - res["T0"][3]))
for design, rule in (("V3", "D650"), ("V1", "S"), ("V4a", "D650")):
    v = []
    for (i0, e0), (ix, ex) in zip(parts(design, "T0", rule), parts(design, "TX", rule)):
        v.append(np.abs(ex - L5[ix]).mean() / np.abs(e0 - L5[i0]).mean())
    print("   MAE ratio TX over T0, %s %s: %.3f" % (design, rule, np.mean(v)))

# ---- (C) specificity
print("(C) pooled again from the stored fits of this batch")
for design in ("V1", "V3", "V4a"):
    for rule in ("D650", "S"):
        print("   %-3s %-4s over T0: TM %+.3f, TXM %+.3f, TXs %+.3f, TBi %+.3f, TBm %+.3f, TBo %+.3f, TX %+.3f | TX over TM %+.3f | TXM over TX %+.3f" % (
            design, rule, skill(design, "TM", "T0", rule), skill(design, "TXM", "T0", rule), skill(design, "TXs", "T0", rule),
            skill(design, "TBi", "T0", rule), skill(design, "TBm", "T0", rule), skill(design, "TBo", "T0", rule),
            skill(design, "TX", "T0", rule), skill(design, "TX", "TM", rule), skill(design, "TXM", "TX", rule)))

# ---- (D) seeds
print("(D) increment of the chronological fit over ten seeds of the regression")
for u in ("V4a", "V4b"):
    for rule in ("D650", "S"):
        v = []
        for sd in range(1, 11):
            i0, e0 = load(u, "T0_seed%d" % sd, rule)
            ix, ex = load(u, "TX_seed%d" % sd, rule)
            v.append(1 - ((ex - L5[ix]) ** 2).sum() / ((e0 - L5[i0]) ** 2).sum())
        i0, e0 = np.load(F75 / ("%s__T0__%s__L5.npz" % (u, rule)))["idx"], np.load(F75 / ("%s__T0__%s__L5.npz" % (u, rule)))["est"]
        ex = np.load(F75 / ("%s__TX__%s__L5.npz" % (u, rule)))["est"]
        reg = 1 - ((ex - L5[i0]) ** 2).sum() / ((e0 - L5[i0]) ** 2).sum()
        print("   %-3s %-4s registered %+.3f; ten seeds: smallest %+.3f, median %+.3f, largest %+.3f, sd %.3f" % (
            u, rule, reg, min(v), float(np.median(v)), max(v), float(np.std(v))))
