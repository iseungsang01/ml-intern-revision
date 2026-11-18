"""Unregistered look (use-case lens): does the T_i deficit after a beam blip have the time shape of the
ECEI deficit (beam-independent) or of the BES deficit (beam-dependent)?  Real 10 ms files only.
Run from repo root:  py <this file>
"""
import sys
from pathlib import Path
import numpy as np

REPO = Path(r"C:\Users\lss\Documents\GitHub\ml-intern-revision")
sys.path.insert(0, str(REPO / "ces_prediction"))
import analyze_file_rules as F  # noqa: E402

files = sorted(F.DATA.glob("s*.csv"))
K = list(range(1, 11))
ev = {q: [] for q in ("ti", "ecei", "bes")}
ctl = {q: [] for q in ("ti", "ecei", "bes")}
evf, ctf, evn = [], [], []
rng = np.random.default_rng(7)
for fi, f in enumerate(files):
    df, *_ = F.load(f)
    t = df["time"].to_numpy(float)
    if len(t) < 30:
        continue
    ti = df["CES_TI"].to_numpy(float).copy()
    ti[ti > 3000] = np.nan
    ecei = df[[c for c in df.columns if c.startswith("ECEI_")]].to_numpy(float).mean(1)
    bes = df[[c for c in df.columns if c.startswith("BES_")]].to_numpy(float).mean(1)
    dt = np.diff(t)
    groups = F.absent_groups(t)
    absent_j = {j for j, _ in groups}

    def prof(j, first_after):
        pre = slice(j - 5, j + 1)
        if j < 5 or not np.all(np.abs(np.diff(t[pre]) - 0.010) < 1e-4):
            return None
        tip = np.nanmedian(ti[pre]) if np.isfinite(ti[pre]).sum() >= 3 else np.nan
        ep, bp = np.median(ecei[pre]), np.median(bes[pre])
        out = {q: [] for q in ev}
        for k in K:
            i = first_after + k - 1
            want = t[first_after] + 0.010 * (k - 1)
            ok = i < len(t) and abs(t[i] - want) < 1e-3
            out["ti"].append(ti[i] / tip if ok and np.isfinite(tip) and tip > 0 else np.nan)
            out["ecei"].append(ecei[i] / ep if ok and abs(ep) > 1e-9 else np.nan)
            out["bes"].append(bes[i] / bp if ok and abs(bp) > 1e-9 else np.nan)
        return out

    for j, stamps in groups:
        p = prof(j, j + 1)
        if p is None:
            continue
        for q in ev:
            ev[q].append(p[q])
        evf.append(fi)
        evn.append(len(stamps))
    # control pseudo-events: rows with 10 ms steps around, no absent frame within +-200 ms
    cand = np.flatnonzero(np.abs(dt - 0.010) < 1e-4)
    bad = np.zeros(len(t), bool)
    for j in absent_j:
        bad[max(0, j - 20):j + 21] = True
    cand = cand[(cand > 6) & (cand < len(t) - 12) & ~bad[cand]]
    if len(cand):
        for j in rng.choice(cand, size=min(3, len(cand)), replace=False):
            # pretend the step after row j were a blip: first_after = j + 1 (no frame missing)
            p = prof(j, j + 1)
            if p is None:
                continue
            for q in ctl:
                ctl[q].append(p[q])
            ctf.append(fi)

for d in (ev, ctl):
    for q in d:
        d[q] = np.array(d[q], float)
evf, ctf = np.array(evf), np.array(ctf)
print("events", len(evf), "discharges", len(np.unique(evf)), "control", len(ctf))


def stat(d, fidx, sel):
    m = {}
    for q in d:
        m[q] = np.array([np.nanmedian(d[q][sel, k]) for k in range(len(K))])
    return m


def boot(d, fidx, nb=1000, seed=1):
    uf = np.unique(fidx)
    pos = {f: np.flatnonzero(fidx == f) for f in uf}
    r = np.random.default_rng(seed)
    res = []
    for _ in range(nb):
        pick = r.choice(uf, size=len(uf))
        sel = np.concatenate([pos[f] for f in pick])
        res.append(stat(d, fidx, sel))
    return res


full = stat(ev, evf, np.arange(len(evf)))
cfull = stat(ctl, ctf, np.arange(len(ctf)))
B = boot(ev, evf)


def shape(m, q):
    dfc = 1 - m[q]
    return dfc[3:6].mean() / dfc[0]          # mean deficit rows 4-6 / deficit row 1


def ci(x):
    return np.quantile(x, [0.025, 0.975])


print("row  Ti     ECEI   BES   | control Ti ECEI BES")
for k in range(len(K)):
    print(f"+{K[k]:<3} {full['ti'][k]:.3f}  {full['ecei'][k]:.3f}  {full['bes'][k]:.3f} | "
          f"{cfull['ti'][k]:.3f} {cfull['ecei'][k]:.3f} {cfull['bes'][k]:.3f}")
for q in ("ti", "ecei", "bes"):
    b = np.array([shape(m, q) for m in B])
    print(f"shape(rows4-6/row1) {q}: {shape(full, q):.3f} CI {ci(b).round(3)}")
dTE = np.array([shape(m, 'ti') - shape(m, 'ecei') for m in B])
dTB = np.array([shape(m, 'ti') - shape(m, 'bes') for m in B])
print("Ti-ECEI shape diff", round(shape(full, 'ti') - shape(full, 'ecei'), 3), ci(dTE).round(3))
print("Ti-BES shape diff", round(shape(full, 'ti') - shape(full, 'bes'), 3), ci(dTB).round(3))
# proportionality across rows 1..8: least squares a on deficits, relative residual
for q in ("ecei", "bes"):
    rr = []
    for m in [full] + B:
        y, x = 1 - m["ti"][:8], 1 - m[q][:8]
        a = (x @ y) / (x @ x)
        rr.append(np.sum((y - a * x) ** 2) / np.sum(y ** 2))
    rr = np.array(rr)
    print(f"Ti deficit = a*{q} deficit, rows 1-8: unexplained share {rr[0]:.3f} CI {ci(rr[1:]).round(3)}")

# event level: rows 4-6, where BES is back: does T_i deficit track ECEI deficit, BES deficit?
from scipy.stats import spearmanr  # noqa: E402
a = np.nanmean(1 - ev["ti"][:, 3:6], 1)
e = np.nanmean(1 - ev["ecei"][:, 3:6], 1)
bb = np.nanmean(1 - ev["bes"][:, 3:6], 1)
b1 = 1 - ev["bes"][:, 0]
ok = np.isfinite(a) & np.isfinite(e) & np.isfinite(bb) & np.isfinite(b1)
print("event rows4-6: rho(Ti,ECEI)", round(spearmanr(a[ok], e[ok])[0], 3), "rho(Ti,BES)", round(spearmanr(a[ok], bb[ok])[0], 3),
      "rho(Ti_4-6, BES_row1)", round(spearmanr(a[ok], b1[ok])[0], 3), "n", ok.sum())
# partial rank: residualise ranks
from scipy.stats import rankdata  # noqa: E402
ra, re_, rb_ = rankdata(a[ok]), rankdata(e[ok]), rankdata(bb[ok])
def resid(y, x):
    X = np.c_[np.ones_like(x), x]
    return y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
print("partial rho(Ti,ECEI | BES)", round(np.corrcoef(resid(ra, rb_), resid(re_, rb_))[0, 1], 3),
      "partial rho(Ti,BES | ECEI)", round(np.corrcoef(resid(ra, re_), resid(rb_, re_))[0, 1], 3))
# row 1 vs absent-frame count
evn = np.array(evn)
for n in (1, 2, 3):
    s = evn == n
    print("frames", n, "events", s.sum(), "Ti row1", round(np.nanmedian(ev['ti'][s, 0]), 3),
          "ECEI row1", round(np.nanmedian(ev['ecei'][s, 0]), 3), "BES row1", round(np.nanmedian(ev['bes'][s, 0]), 3),
          "Ti rows4-6", round(np.nanmedian(ev['ti'][s, 3:6]), 3), "ECEI rows4-6", round(np.nanmedian(ev['ecei'][s, 3:6]), 3))

# two-regressor decomposition of the event-average deficit over rows 1-8:
# Ti_def(k) = a * ECEI_def(k) + b * BES_def(k); b bounds a signal-level (beam-emission) dependent bias
ab = []
for m in [full] + B:
    y = 1 - m["ti"][:8]
    X = np.c_[1 - m["ecei"][:8], 1 - m["bes"][:8]]
    ab.append(np.linalg.lstsq(X, y, rcond=None)[0])
ab = np.array(ab)
print("a (per ECEI deficit)", ab[0, 0].round(3), ci(ab[1:, 0]).round(3), " b (per BES deficit)", ab[0, 1].round(3), ci(ab[1:, 1]).round(3))
print("BES-attributable part of Ti deficit at row 1:", (ab[0, 1] * (1 - full['bes'][0])).round(4),
      ci(ab[1:, 1] * np.array([1 - m['bes'][0] for m in B])).round(4), " total", round(1 - full['ti'][0], 3))
