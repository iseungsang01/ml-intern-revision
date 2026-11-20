# -*- coding: utf-8 -*-
"""Unregistered diagnostic (session lead): why the fast-channel increment under the schedule rule S is not established
on the last block of shots (from 32312).  Random design V1 of B.75 (stored estimates), target L5, S set.

Reads: per block of shots, (1) what the inputs look like (age, level, scatter, beam), (2) how much the level moves
between the anchor and the present and how that movement relates to the change of the ECEI and BES levels (no
estimator), (3) the increment by age, (4) which discharges carry it.
    py last_block.py <repo root>      -> prints; the lead copies the output into docs/paper/b75_last_block.md
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
A = dict(np.load(ROOT / "data" / ".b75" / "inputs" / "S.npz", allow_pickle=False))
D10 = dict(np.load(ROOT / "data" / ".b75" / "inputs" / "D10.npz", allow_pickle=False))
BC = dict(np.load(ROOT / "data" / ".b79" / "beam_cols.npz", allow_pickle=False))
L5, y, gi, shot, clus = R["tg_L5"], R["y"], R["gi"], R["shot"], R["clus"]
seg, fast = G["seg"], G["fast"].astype(np.float64)
N = len(seg)
BLOCKS = (("up to 31991", 0, 31991), ("32002-32310", 32002, 32310), ("from 32312", 32312, 10 ** 9))
blk = np.full(len(shot), -1)
for b, (_, lo, hi) in enumerate(BLOCKS):
    blk[(shot >= lo) & (shot <= hi)] = b
m = S["S_L5"]

# squared errors of T0 and TX under S, mean over the three re-partitions of V1
se = {k: np.zeros(len(shot)) for k in ("T0", "TX")}
cnt = np.zeros(len(shot))
for r in (2, 3, 4):
    for k in range(5):
        for L in ("T0", "TX"):
            z = np.load(ROOT / "data" / ".b75" / "fits" / ("V1_r%d_k%d__%s__S__L5.npz" % (r, k, L)))
            se[L][z["idx"]] += (z["est"] - L5[z["idx"]]) ** 2
        cnt[z["idx"]] += 1
assert np.all(cnt[m] == 3)
for L in se:
    se[L] /= 3.0
se_h = (A["H"] - L5) ** 2
se_o = (y - L5) ** 2
se_hf = (D10["H"] - L5) ** 2


def runmean(x):
    s, c = np.zeros(N), np.zeros(N)
    for k in range(10):
        j = np.arange(N) - k
        o = (j >= 0) & (seg[np.maximum(j, 0)] == seg)
        s[o] += x[j[o]]
        c[o] += 1
    return np.where(c >= 5, s / np.maximum(c, 1), np.nan)


anc = np.maximum(A["anc_g"], 0)
me, mb = runmean(fast[:, 9:13].mean(1)), runmean(fast[:, :9].mean(1))
with np.errstate(all="ignore"):
    dE = np.log(me[gi] / me[anc])           # change of the ECEI level since the anchor (log ratio)
    dB = np.log(mb[gi] / mb[anc])
    dT = np.log(L5 / A["H"])                 # the level against the newest visible value (log ratio)
    dT5 = np.log(L5 / A["M5"])               # the level against the median of the last five visible values
p1, p2 = np.nansum(BC["P"][:, :2], 1), np.nansum(BC["P"][:, 2:], 1)       # an empty power is off

print("(1) inputs on the S set (target L5), by block of the discharge")
print("   block         rows  disch clus | age median (10-90 %) s | L5 median eV | RMSE of the frame's own reading / hold fresh / hold S | P1, P2 median MW | rows with NBI-2 off")
for b, (nm, _, _) in enumerate(BLOCKS):
    q = m & (blk == b)
    a = A["age"][q] / 1000.0
    print("   %-12s %6d %5d %4d | %.2f (%.2f-%.2f) | %6.0f | %5.1f / %5.1f / %5.1f | %.2f, %.2f | %.3f" % (
        nm, q.sum(), len(np.unique(shot[q])), len(np.unique(clus[q])), np.median(a), np.percentile(a, 10), np.percentile(a, 90), np.median(L5[q]),
        np.sqrt(se_o[q].mean()), np.sqrt(se_hf[q].mean()), np.sqrt(se_h[q].mean()), np.median(p1[gi][q]), np.median(p2[gi][q]), (p2[gi][q] < 0.1).mean()))

print("(2) how much the level moves since the newest visible value, and how it goes with the fast levels (no estimator)")
print("   block        | sd of ln(L5 / M5 of the visible values) | Spearman of that with the ECEI change, with the BES change | sd of the ECEI change, of the BES change | relative scatter of the own reading |O/L5 - 1| median")
from scipy.stats import spearmanr  # noqa: E402
for b, (nm, _, _) in enumerate(BLOCKS):
    q = m & (blk == b) & np.isfinite(dE) & np.isfinite(dB) & np.isfinite(dT5)
    print("   %-12s | %.3f | %+.3f, %+.3f | %.3f, %.3f | %.3f" % (
        nm, np.std(dT5[q]), spearmanr(dT5[q], dE[q])[0], spearmanr(dT5[q], dB[q])[0], np.std(dE[q]), np.std(dB[q]), np.median(np.abs(y[q] / L5[q] - 1))))

print("(3) TX over T0 under S by the age of the newest visible value (V1, stored estimates)")
edges = (0, 0.1, 0.3, 1.0, 99)
print("   block        | " + " | ".join("%.1f-%.1f s (rows; T0, TX RMSE; skill)" % (edges[i], edges[i + 1]) for i in range(4)))
for b, (nm, _, _) in enumerate(BLOCKS):
    cells = []
    for i in range(4):
        q = m & (blk == b) & (A["age"] / 1000.0 >= edges[i]) & (A["age"] / 1000.0 < edges[i + 1])
        cells.append("%d; %.0f, %.0f; %+.3f" % (q.sum(), np.sqrt(se["T0"][q].mean()), np.sqrt(se["TX"][q].mean()), 1 - se["TX"][q].sum() / se["T0"][q].sum()) if q.sum() > 50 else "-")
    print("   %-12s | " % nm + " | ".join(cells))

print("(4) discharges of the last block, S set: contribution to SSE(T0) - SSE(TX)")
q = m & (blk == 2)
sh = np.unique(shot[q])
rows = []
for s_ in sh:
    w = q & (shot == s_)
    rows.append((s_, int(w.sum()), se["T0"][w].sum() - se["TX"][w].sum(), np.sqrt(se["T0"][w].mean()), np.sqrt(se["TX"][w].mean()), np.sqrt(se_o[w].mean()),
                 np.median(L5[w]), np.median(p1[gi][w]), np.median(p2[gi][w]), np.median(A["age"][w]) / 1000.0, np.median(np.abs(y[w] / L5[w] - 1)), int(clus[w][0])))
rows.sort(key=lambda r: r[2])
tot = sum(r[2] for r in rows)
t0 = se["T0"][q].sum()
print("   discharges %d; total reduction %.3g eV^2 (skill %+.3f); sum of the negative contributions %.3g, of the positive %.3g" % (
    len(rows), tot, tot / t0, sum(r[2] for r in rows if r[2] < 0), sum(r[2] for r in rows if r[2] > 0)))
print("   without the three discharges with the most negative contribution: skill %+.3f; without the three most positive: %+.3f" % (
    sum(r[2] for r in rows[3:]) / (t0 - sum(se["T0"][q & (shot == r[0])].sum() for r in rows[:3])),
    sum(r[2] for r in rows[:-3]) / (t0 - sum(se["T0"][q & (shot == r[0])].sum() for r in rows[-3:]))))
print("   shot  cluster rows | share of the block's SSE(T0) | RMSE T0, TX, own reading | L5 median | P1, P2 MW | age s | scatter")
for r in rows[:6] + rows[-6:]:
    w = q & (shot == r[0])
    print("   %5d %4d %5d | %.3f | %5.0f %5.0f %5.0f | %5.0f | %.2f %.2f | %.2f | %.3f   contribution %+.3g" % (
        r[0], r[11], r[1], se["T0"][w].sum() / t0, r[3], r[4], r[5], r[6], r[7], r[8], r[9], r[10], r[2]))
# the same summary for the other blocks: how concentrated is the reduction
for b in (0, 1):
    qb = m & (blk == b)
    d = np.array([se["T0"][qb & (shot == s_)].sum() - se["TX"][qb & (shot == s_)].sum() for s_ in np.unique(shot[qb])])
    print("   block %s: discharges %d, share with a negative contribution %.3f (last block: %.3f)" % (
        BLOCKS[b][0], len(d), (d < 0).mean(), np.mean([r[2] < 0 for r in rows])))

print("(5) by the level of the visible history (median of the last five visible values; a stratum on an input), S set")
lev = A["M5"]
LE = (0, 250, 400, 600, 900, 1e9)
LN = ["below 250 eV", "250-400", "400-600", "600-900", "above 900"]
print("   level        | all blocks: rows; Spearman with the ECEI change, with the BES change; scatter of the own reading; TX over T0 | the same for the last block | for the two earlier blocks")
for i in range(5):
    out = []
    for sel in (np.ones(len(shot), bool), blk == 2, blk < 2):
        q = m & sel & (lev >= LE[i]) & (lev < LE[i + 1]) & np.isfinite(dE) & np.isfinite(dB) & np.isfinite(dT5)
        if q.sum() < 200:
            out.append("%d rows" % q.sum())
            continue
        out.append("%d; %+.2f, %+.2f; %.3f; %+.3f" % (q.sum(), spearmanr(dT5[q], dE[q])[0], spearmanr(dT5[q], dB[q])[0],
                                                     np.median(np.abs(y[q] / L5[q] - 1)), 1 - se["TX"][q].sum() / se["T0"][q].sum()))
    print("   %-12s | " % LN[i] + " | ".join(out))
print("   share of S-set rows by level, per block (below 250 / 250-400 / 400-600 / 600-900 / above 900):")
for b, (nm, _, _) in enumerate(BLOCKS):
    q = m & (blk == b)
    print("   %-12s " % nm + " / ".join("%.3f" % ((lev[q] >= LE[i]) & (lev[q] < LE[i + 1])).mean() for i in range(5)))

print("(6) by the state of NBI-2 at row t (a stratum on an input), S set: rows; Spearman with the ECEI change; TX over T0")
p2z = p2[gi]
for nm, sel in (("NBI-2 on", p2z > 0.1), ("NBI-2 off", p2z <= 0.1)):
    out = []
    for sel2 in (np.ones(len(shot), bool), blk == 2, blk < 2):
        q = m & sel & sel2 & np.isfinite(dE) & np.isfinite(dT5)
        out.append("%d; %+.2f; %+.3f" % (q.sum(), spearmanr(dT5[q], dE[q])[0], 1 - se["TX"][q].sum() / se["T0"][q].sum()) if q.sum() >= 200 else "%d rows" % q.sum())
    print("   %-10s | all: %s | last block: %s | earlier blocks: %s" % (nm, *out))

print("(7) the notch as a common event (no estimator): value in the first row after an absent step over the median of the five rows before it, by block of the discharge")
tm, ti_g, fid = G["time"].astype(np.float64), G["ti"].astype(np.float64), G["fidx"]
shot_g = G["shot"][fid] if G["shot"].shape[0] != N else G["shot"]
ece_raw, bes_raw = fast[:, 9:13].mean(1), fast[:, :9].mean(1)
dt = np.diff(tm)
ab = np.flatnonzero((seg[1:] == seg[:-1]) & (dt > 0.015) & (dt < 0.045))
ok = [j for j in ab if j >= 5 and seg[j - 5] == seg[j] and j + 1 < N]
res = {b: {"ti": [], "ece": [], "bes": []} for b in range(3)}
for j in ok:
    sh_ = shot_g[j]
    b = 0 if sh_ <= 31991 else (1 if sh_ <= 32310 else 2)
    for key, x in (("ti", ti_g), ("ece", ece_raw), ("bes", bes_raw)):
        ref = np.nanmedian(x[j - 4:j + 1])
        if np.isfinite(ref) and ref > 0 and np.isfinite(x[j + 1]):
            res[b][key].append(x[j + 1] / ref)
for b, (nm, _, _) in enumerate(BLOCKS):
    r = res[b]
    print("   %-12s notches %5d | T_i %.3f (n %d) | ECEI %.3f (10-90 %%: %.3f-%.3f) | BES %.3f" % (
        nm, len(r["bes"]), np.median(r["ti"]), len(r["ti"]), np.median(r["ece"]), np.percentile(r["ece"], 10), np.percentile(r["ece"], 90), np.median(r["bes"])))

print("(8) the ECEI channels themselves, per discharge (median over discharges of the block), grid rows of blocks of at least 100 rows")
for b, (nm, lo, hi) in enumerate(BLOCKS):
    lev_, cv_, cc_, neg_ = [], [], [], []
    for f in np.unique(fid):
        a_, b_ = G["off"][f], G["off"][f + 1]
        sh_ = shot_g[a_]
        if not (lo <= sh_ <= hi) or b_ - a_ < 120:
            continue
        E = fast[a_ + 1:b_ - 1, 9:13]
        mE = E.mean(1)
        lev_.append(np.median(mE))
        neg_.append((mE <= 0).mean())
        sm = np.convolve(mE, np.ones(10) / 10, mode="valid")
        cv_.append(np.std(np.diff(np.log(np.clip(sm[::10], 1e-9, None)))) if (sm > 0).all() and len(sm) > 50 else np.nan)
        c = np.corrcoef(E.T)
        cc_.append(np.nanmean(c[np.triu_indices(4, 1)]))
    print("   %-12s discharges %3d | raw array mean, median %.4f | share of rows at or below zero %.4f | sd of the 100 ms log change of the smoothed level %.3f | mean correlation between the four channels %.3f" % (
        nm, len(lev_), np.median(lev_), np.mean(neg_), np.nanmedian(cv_), np.nanmedian(cc_)))
