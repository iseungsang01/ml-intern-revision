# -*- coding: utf-8 -*-
"""Unregistered diagnostic (session lead): rows in which the ECEI channels do not move.

Found while checking the example discharge of the paper (31076): its four ECEI channels are exactly constant from the
start of the file to about 2 s.  This script counts such rows over all discharges (input only), by block of shots and
on the row sets of B.75, and reads the stored estimates of B.75 (random design V1, target L5) by whether the ECEI is
constant at the row (a stratum on an input).
    py stuck_ecei.py <repo root>
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
L5, gi, shot, fidx = R["tg_L5"], R["gi"], R["shot"], R["fidx"]
fast, seg, off, gf = G["fast"].astype(np.float64), G["seg"], G["off"], G["fidx"]
N = len(seg)
E, Bc = fast[:, 9:13], fast[:, :9]
same_file = np.zeros(N, bool)
same_file[1:] = gf[1:] == gf[:-1]


def stuck(X, k):
    """Row i: every channel equal to its value k rows earlier and at every row in between (same file)."""
    eq1 = np.zeros(N, bool)
    eq1[1:] = (X[1:] == X[:-1]).all(1) & same_file[1:]
    run = np.zeros(N, np.int64)
    for i in range(1, N):
        run[i] = run[i - 1] + 1 if eq1[i] else 0
    return run >= k, run


stE, runE = stuck(E, 10)          # the ECEI has not moved for at least ten rows (100 ms)
stB, runB = stuck(Bc, 10)
first = np.zeros(N, bool)
first[off[:-1]] = True
last = np.zeros(N, bool)
last[off[1:] - 1] = True
grid = ~first & ~last
shot_g = G["shot"][gf] if G["shot"].shape[0] != N else G["shot"]
BLOCKS = (("up to 31991", 0, 31991), ("32002-32310", 32002, 32310), ("from 32312", 32312, 10 ** 9))
print("(1) grid rows in which the four ECEI channels have not changed for at least ten rows (BES: the nine channels)")
print("   block         grid rows | ECEI constant: share of rows, discharges with any, with more than half | BES constant: share of rows")
for nm, lo, hi in BLOCKS + (("all", 0, 10 ** 9),):
    m = grid & (shot_g >= lo) & (shot_g <= hi)
    files = np.unique(gf[m])
    anyf = sum(stE[grid & (gf == f)].any() for f in files)
    half = sum(stE[grid & (gf == f)].mean() > 0.5 for f in files)
    print("   %-12s %8d | %.3f, %d of %d, %d | %.4f" % (nm, m.sum(), stE[m].mean(), anyf, len(files), half, stB[m].mean()))

anc = np.maximum(dict(np.load(ROOT / "data" / ".b75" / "inputs" / "S.npz", allow_pickle=False))["anc_g"], 0)
anc650 = np.maximum(dict(np.load(ROOT / "data" / ".b75" / "inputs" / "D650.npz", allow_pickle=False))["anc_g"], 0)
blk = np.where(shot <= 31991, 0, np.where(shot <= 32310, 1, 2))
print("(2) scored rows of B.75 with a constant ECEI at row t or at the anchor row")
for setn, an in (("S_L5", anc), ("common_L5", anc650)):
    m = S[setn]
    c_t, c_a = stE[gi], stE[an]
    print("   %-10s rows %6d: constant at row t %.3f; at the anchor row %.3f; either %.3f | by block (either): %s" % (
        setn, m.sum(), c_t[m].mean(), c_a[m].mean(), (c_t | c_a)[m].mean(),
        ", ".join("%s %.3f" % (BLOCKS[b][0], (c_t | c_a)[m & (blk == b)].mean()) for b in range(3))))


def se_of(rule, L):
    out = np.zeros(len(shot))
    cnt = np.zeros(len(shot))
    for r in (2, 3, 4):
        for k in range(5):
            z = np.load(ROOT / "data" / ".b75" / "fits" / ("V1_r%d_k%d__%s__%s__L5.npz" % (r, k, L, rule)))
            out[z["idx"]] += (z["est"] - L5[z["idx"]]) ** 2
            cnt[z["idx"]] += 1
    return out / np.maximum(cnt, 1)


print("(3) TX over T0 (V1, stored estimates, target L5) by whether the ECEI is constant at row t or at the anchor row")
for rule, setn, an in (("S", "S_L5", anc), ("D650", "common_L5", anc650)):
    m = S[setn]
    s0, sx = se_of(rule, "T0"), se_of(rule, "TX")
    either = stE[gi] | stE[an]
    for nm, sel in (("all blocks", np.ones(len(shot), bool)), ("up to 31991", blk == 0), ("32002-32310", blk == 1), ("from 32312", blk == 2)):
        a, b = m & sel & ~either, m & sel & either
        f = lambda q: ("%d rows, %d discharges, %+.3f" % (q.sum(), len(np.unique(shot[q])), 1 - sx[q].sum() / s0[q].sum())) if q.sum() >= 200 else ("%d rows" % q.sum())  # noqa: E731
        print("   rule %-4s %-12s | ECEI moving: %s | ECEI constant: %s" % (rule, nm, f(a), f(b)))
