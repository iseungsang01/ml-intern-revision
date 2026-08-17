# -*- coding: utf-8 -*-
"""How much do neighbouring shot numbers resemble each other?

Repeat discharges from one session share plasma setup, diagnostic gain/offset and wall
state, so a split that puts one on each side of a train/test boundary can hand the model
free information. Whether that actually matters is measurable, and this measures it over
all 641 shots rather than asserting it.

Three distances per shot pair, binned by shot-number gap:
  * summary distance   -- z-scored shot-level summary vector (CES levels, BES/ECEI/MC
                          levels and spreads, discharge timing)
  * |dTi|              -- difference of mean CES_TI in eV: what a leaked neighbour would
                          tell the model about the answer
  * calibration distance -- z-scored per-channel BES/ECEI means: the gain/offset channel,
                          which is how a session leaks even when the physics differs

Usage (repo root):  py ces_prediction/experiments/hires_shots/session_similarity.py
Writes: session_similarity.json next to this file.
"""
from __future__ import annotations

import glob
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA = Path(os.environ.get("CES_DATA_DIR", REPO_ROOT / "data"))
HERE = Path(__file__).resolve().parent
GAP_S = 0.5
TI_SPIKE_EV = 3000.0
PAIRS_OF_INTEREST = ((31921, 31923), (31357, 31359))
BINS = ((1, 2), (3, 6), (7, 20), (21, 60), (61, 200), (201, 600), (601, 2000))


def main_block(df):
    t = df.time.to_numpy(float)
    blk = np.concatenate(([0], np.cumsum(np.diff(t) >= GAP_S)))
    return df.loc[blk == int(np.argmax(np.bincount(blk)))].reset_index(drop=True)


def summarise():
    rows = []
    for f in sorted(glob.glob(str(DATA / "s*.csv"))):
        df = main_block(pd.read_csv(f))
        if len(df) < 50:
            continue
        ti = df.CES_TI.to_numpy(float)
        ti = np.where(ti > TI_SPIKE_EV, np.nan, ti)
        if not np.isfinite(ti).any():
            continue
        bes_c = [c for c in df.columns if c.startswith("BES_")]
        ece_c = [c for c in df.columns if c.startswith("ECEI_")]
        bes = df[bes_c].to_numpy(float)
        ece = df[ece_c].to_numpy(float)
        mc = df[[c for c in df.columns if c.startswith("MC")]].to_numpy(float)
        vt = df.CES_VT.to_numpy(float)
        rec = {"shot": int(os.path.basename(f)[1:-4]),
               "ti_mean": np.nanmean(ti), "ti_std": np.nanstd(ti),
               "vt_mean": np.nanmean(vt) if np.isfinite(vt).any() else np.nan,
               "t_start": float(df.time.min()),
               "t_span": float(df.time.max() - df.time.min()),
               "bes_mean": np.nanmean(bes), "bes_std": np.nanstd(bes),
               "ece_mean": np.nanmean(ece), "ece_std": np.nanstd(ece),
               "mc_rms": float(np.sqrt(np.nanmean(mc ** 2)))}
        for j, c in enumerate(bes_c + ece_c):
            rec[f"m_{c}"] = np.nanmean((bes if j < len(bes_c) else ece)
                                       [:, j if j < len(bes_c) else j - len(bes_c)])
        rows.append(rec)
    return pd.DataFrame(rows).dropna(subset=["ti_mean"]).reset_index(drop=True)


def zscore(M):
    M = np.asarray(M, float)
    return np.nan_to_num((M - np.nanmean(M, axis=0)) / np.nanstd(M, axis=0))


def main():
    d = summarise()
    print(f"shots with a usable summary: {len(d)}")
    shots = d.shot.to_numpy()
    iu = np.triu_indices(len(d), 1)
    gap = np.abs(shots[iu[0]] - shots[iu[1]])

    X = zscore(d[["ti_mean", "ti_std", "vt_mean", "bes_mean", "bes_std", "ece_mean",
                  "ece_std", "mc_rms", "t_start", "t_span"]].to_numpy(float))
    C = zscore(d[[c for c in d.columns if c.startswith("m_")]].to_numpy(float))
    dist = np.linalg.norm(X[iu[0]] - X[iu[1]], axis=1)
    cal = np.linalg.norm(C[iu[0]] - C[iu[1]], axis=1)
    ti = d.ti_mean.to_numpy()
    dti = np.abs(ti[iu[0]] - ti[iu[1]])

    out = {"n_shots": int(len(d)), "n_pairs": int(len(gap)), "bins": [], "pairs": {},
           "permutation": {}}
    print("\n gap         pairs   summary-dist   |dTi| [eV]   calibration-dist")
    print(" " + "-" * 70)
    for lo, hi in BINS:
        m = (gap >= lo) & (gap <= hi)
        if m.sum() < 5:
            continue
        row = {"gap_lo": lo, "gap_hi": hi, "n": int(m.sum()),
               "summary_dist": float(np.median(dist[m])),
               "d_ti_ev": float(np.median(dti[m])),
               "calibration_dist": float(np.median(cal[m]))}
        out["bins"].append(row)
        print(f" {lo:4d}-{hi:<6d} {row['n']:7d}   {row['summary_dist']:8.3f}   "
              f"{row['d_ti_ev']:10.1f}   {row['calibration_dist']:12.3f}")
    print(f" {'all':<11s} {len(gap):7d}   {np.median(dist):8.3f}   "
          f"{np.median(dti):10.1f}   {np.median(cal):12.3f}")

    rng = np.random.default_rng(0)
    near = gap <= 2
    for label, vec in (("summary_dist", dist), ("d_ti_ev", dti), ("calibration_dist", cal)):
        obs = float(np.median(vec[near]))
        null = np.array([np.median(rng.choice(vec, size=int(near.sum()), replace=False))
                         for _ in range(4000)])
        p = float((null <= obs).mean())
        out["permutation"][label] = {"adjacent_median": obs,
                                     "random_median": float(np.median(null)),
                                     "p_one_sided": p, "n_adjacent_pairs": int(near.sum())}
        print(f"\n{label}: adjacent(gap<=2) {obs:.3f} vs random {np.median(null):.3f}, "
              f"one-sided p = {p:.4f}")

    print("\n=== the two adjacent pairs in the fetch list ===")
    for a, b in PAIRS_OF_INTEREST:
        ia = int(np.flatnonzero(shots == a)[0])
        ib = int(np.flatnonzero(shots == b)[0])
        ds = float(np.linalg.norm(X[ia] - X[ib]))
        dc = float(np.linalg.norm(C[ia] - C[ib]))
        rec = {"summary_dist": ds, "summary_pct": float((dist < ds).mean() * 100),
               "calibration_dist": dc, "calibration_pct": float((cal < dc).mean() * 100),
               "d_ti_ev": float(abs(ti[ia] - ti[ib]))}
        out["pairs"][f"{a}-{b}"] = rec
        print(f"  #{a} vs #{b}: summary {ds:.3f} (pct {rec['summary_pct']:.1f}), "
              f"calibration {dc:.3f} (pct {rec['calibration_pct']:.1f}), "
              f"|dTi| {rec['d_ti_ev']:.1f} eV")

    (HERE / "session_similarity.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"\nwrote {HERE / 'session_similarity.json'}")


if __name__ == "__main__":
    main()
