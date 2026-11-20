"""What the NBI files are (delivered 2026-11-20): how they relate to the 10 ms shot files, which sources run, what an
absent frame is in them, and whether a CES value is written by the state of the sources.

Reads data/s{shot}.csv and data/nbi/s{shot}_NBI.csv; writes docs/paper/nbi_files.md.  No estimator, no target scoring.
    py ces_prediction/analyze_nbi_files.py
"""
from __future__ import annotations

import os
import re
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("CES_DATA_DIR", REPO / "data"))
NBI = DATA / "nbi"
OUT = REPO / "docs" / "paper" / "nbi_files.md"
SRC = ("NB1A", "NB1B", "NB1C", "NB2A", "NB2B", "NB2C")
ON_MW = 0.1
BLOCKS = (("up to 31991", 0, 31991), ("32002-32310", 32002, 32310), ("from 32312", 32312, 10 ** 9))


def load(path):
    with open(path, encoding="utf-8", errors="replace") as fh:
        hdr = fh.readline().strip().split(",")
        a = np.array([[float(x) if x not in ("", "nan", "NaN") else np.nan for x in ln.rstrip("\n").split(",")] for ln in fh])
    return hdr, a


def q(v, p):
    return float(np.percentile(v, p))


def main():
    ours = sorted(int(re.search(r"s(\d+)\.csv$", f.name).group(1)) for f in DATA.glob("s*.csv"))
    nbi_shots = sorted(int(re.search(r"s(\d+)_NBI\.csv$", f.name).group(1)) for f in NBI.glob("s*_NBI.csv"))
    hdr0 = None
    same_rows = same_time = 0
    rows_tot = 0
    nan_rows = np.zeros(6, int)
    nan_shots = np.zeros(6, int)
    on_rows = np.zeros(6, int)
    on_shots = np.zeros(6, int)
    volt = [[] for _ in SRC]
    after = [[] for _ in SRC]
    before = [[] for _ in SRC]
    ptot_dev = []
    conf = {}
    combo = {}
    small = []
    all_off = [0, 0]
    n_abs = 0
    ti_after = []
    for sh in ours:
        hn, N = load(NBI / f"s{sh}_NBI.csv")
        hc, C = load(DATA / f"s{sh}.csv")
        hdr0 = hdr0 or hn
        if len(N) == len(C):
            same_rows += 1
            same_time += int(np.max(np.abs(N[:, 0] - C[:, hc.index("time")])) < 1e-4)
        N, C = N[1:-1], C[1:-1]                      # the first (t = 0) and last (t = 30 s) rows are not grid rows
        t, ti, vt = C[:, hc.index("time")], C[:, hc.index("CES_TI")], C[:, hc.index("CES_VT")]
        P = np.column_stack([N[:, hn.index(s + "_P")] for s in SRC])
        V = np.column_stack([N[:, hn.index(s + "_V")] for s in SRC])
        rows_tot += len(P)
        nan_rows += np.isnan(P).sum(0)
        nan_shots += np.isnan(P).any(0)
        Pz = np.where(np.isfinite(P), P, 0.0)
        on = Pz > ON_MW
        on_rows += on.sum(0)
        on_shots += on.any(0)
        for k in range(6):
            if on[:, k].any():
                volt[k].append(float(np.nanmedian(V[on[:, k], k])))
        pt = N[:, hn.index("PNB_tot")]
        m = np.isfinite(pt)
        if m.any():
            ptot_dev.append(float(np.max(np.abs(pt[m] - np.where(Pz > 0, Pz, 0).sum(1)[m]))))
        conf[sh] = "+".join(s[2:] for s, o in zip(SRC, on.any(0)) if o) or "none"
        if not on.any():
            small.append((sh, len(t), int(np.isfinite(ti).sum()), float(Pz.max()) if len(Pz) else 0.0))
        anyon = on.any(1)
        all_off[0] += int((~anyon).sum())
        all_off[1] += int((~anyon & np.isfinite(ti)).sum())
        key = on[:, 1].astype(int) * 4 + on[:, 2].astype(int) * 2 + on[:, 3:].any(1).astype(int)
        for k in range(8):
            mk = key == k
            if mk.any():
                c = combo.setdefault(k, [0, 0, 0])
                c[0] += int(mk.sum())
                c[1] += int(np.isfinite(ti[mk]).sum())
                c[2] += int(np.isfinite(vt[mk]).sum())
        dt = np.diff(t)
        ab = np.flatnonzero((dt > 0.015) & (dt < 0.045))
        n_abs += len(ab)
        for j in ab:
            if j + 11 < len(t) and (np.diff(t[j + 1:j + 11]) < 0.015).all():
                for k in range(6):
                    ref = np.median(Pz[j + 6:j + 11, k])
                    if ref > 0.3:
                        after[k].append(Pz[j + 1, k] / ref)
                        before[k].append(Pz[j, k] / ref)
    L = []
    w = L.append
    w("# The NBI files\n")
    w("Delivered 2026-11-20 (`data/nbi/s{shot}_NBI.csv`, not in the repository). Written by `ces_prediction/analyze_nbi_files.py`; "
      "no estimator and no scoring. A source counts as on in a row when its power is above 0.1 MW. The first (t = 0) and last (t = 30 s) rows of a file are left out of every count.\n")
    w("## 1. Files and rows\n")
    w(f"- NBI files: {len(nbi_shots)} (shots {min(nbi_shots)} to {max(nbi_shots)}). Shot files in use: {len(ours)}; with an NBI file: {len(set(ours) & set(nbi_shots))}. "
      f"NBI files of shots without a shot file: {len(set(nbi_shots) - set(ours))}.")
    w(f"- Same number of rows as the shot file: {same_rows} of {len(ours)} shots; same time stamps (within 0.1 ms): {same_time}. "
      f"An NBI file is the shot file's row list with the beam columns; it has no row for an absent frame. Grid rows: {rows_tot:,}.")
    w(f"- Columns: `time`; for each of {', '.join(SRC)}: power `_P` (MW), voltage `_V` (kV), current `_I` (A), energies of the full, half and third component `_E1`, `_E2`, `_E3` (keV, empty while the source is off); `PNB_tot`; `Eb_mean`.")
    w(f"- `PNB_tot` equals the sum of the positive source powers (largest difference over all rows {max(ptot_dev):.4f} MW).\n")
    w("## 2. Sources\n")
    w("| source | rows on (share) | shots with it on | voltage while on, median over shots (10th to 90th percentile), kV | rows with an empty power (shots) |")
    w("|---|---|---|---|---|")
    for k, s in enumerate(SRC):
        v = np.array(volt[k])
        vs = f"{np.median(v):.1f} ({q(v, 10):.1f} to {q(v, 90):.1f})" if len(v) else "-"
        w(f"| {s} | {on_rows[k]:,} ({on_rows[k] / rows_tot:.3f}) | {on_shots[k]} | {vs} | {nan_rows[k]:,} ({nan_shots[k]}) |")
    w("\nAn empty power is read as off. Source combinations over a shot (sources on in at least one row):\n")
    w("| block of shots | shots | most frequent combinations (shots) |")
    w("|---|---|---|")
    for nm, lo, hi in BLOCKS:
        cc = Counter(c for s_, c in conf.items() if lo <= s_ <= hi)
        w(f"| {nm} | {sum(cc.values())} | " + "; ".join(f"{c} ({n})" for c, n in cc.most_common(6)) + " |")
    w(f"\nShots in which no source is above 0.1 MW in any row: {len(small)}; they have {sum(x[1] for x in small)} rows in all, "
      f"{sum(x[2] for x in small)} of them with a T_i value (largest power in them {max(x[3] for x in small):.3f} MW). "
      f"Rows with every source below 0.1 MW, all shots: {all_off[0]}, of them with a T_i value: {all_off[1]}.\n")
    w("## 3. An absent frame is a notch of the NBI-1 sources\n")
    w(f"Absent steps (a time step of 20 to 40 ms inside a discharge): {n_abs:,}. Power of a source in the last row before and in the first row after the step, "
      "relative to its median in rows 5 to 10 after it (steps followed by ten 10 ms rows, source above 0.3 MW there):\n")
    w("| source | steps | last row before, median (10th percentile) | first row after, median (10th percentile) | share of first rows below 0.9 |")
    w("|---|---|---|---|---|")
    for k, s in enumerate(SRC):
        a_, b_ = np.array(after[k]), np.array(before[k])
        if len(a_):
            w(f"| {s} | {len(a_):,} | {np.median(b_):.3f} ({q(b_, 10):.3f}) | {np.median(a_):.3f} ({q(a_, 10):.3f}) | {(a_ < 0.9).mean():.3f} |")
    w("\nThe two NBI-1 sources that run (B and C) are low together in the first row after an absent frame; the NBI-2 sources are not. "
      "The absent frames are the notches of NBI-1, and the power written in the first row after one is a sample taken while the beam returns.\n")
    w("## 4. Is a CES value written, by the state of the sources\n")
    names = {0: "NBI-1 off, NBI-2 off", 1: "NBI-1 off, NBI-2 on", 2: "1C only", 3: "1C and NBI-2", 4: "1B only", 5: "1B and NBI-2", 6: "1B and 1C", 7: "1B, 1C and NBI-2"}
    w("| state | rows | share with a T_i value | share with a V_rot value |")
    w("|---|---|---|---|")
    for k in sorted(combo):
        n, a, b = combo[k]
        w(f"| {names[k]} | {n:,} | {a / n:.3f} | {b / n:.3f} |")
    w("\nValues are written with either NBI-1 source on. The rows with NBI-1 off are few.\n")
    OUT.write_text("\n".join(L) + "\n", encoding="utf-8", newline="\n")
    print("written", OUT)


if __name__ == "__main__":
    main()
