"""Regime number on KSTAR: lag-1 autocorrelation of 10 ms increments of the CES series.

Descriptive (not pre-registered). Same definition as the MAST gate readout: per discharge,
Pearson correlation of consecutive increments (y_t - y_{t-1}, y_{t+1} - y_t) over triples of
rows 10 ms apart that are all observed; at least 8 pairs. T_i: values above 3 keV treated as
missing (cut population). V_rot: bit-identical repeats of the previous observation (holds)
treated as missing; values beyond 1000 km/s treated as missing.
Run from the repo root:  py docs/paper/ppcf/kstar_regime.py
"""
import glob
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(os.environ.get("CES_DATA_DIR", "data"))
OUT = Path(__file__).parent / "kstar_regime.json"
SEED, NBOOT, MIN_PAIRS = 20261116, 2000, 8


def held_free(v):
    v = v.copy()
    last = np.nan
    for i in range(len(v)):
        if np.isfinite(v[i]):
            if v[i] == last:
                v[i] = np.nan
            else:
                last = v[i]
    return v


def regime(t, y):
    ok = np.isfinite(y)
    step = np.abs(np.diff(t) - 0.010) < 0.001
    tri = ok[:-2] & ok[1:-1] & ok[2:] & step[:-1] & step[1:]
    if tri.sum() < MIN_PAIRS:
        return np.nan, int(tri.sum())
    d1 = (y[1:-1] - y[:-2])[tri]
    d2 = (y[2:] - y[1:-1])[tri]
    if d1.std() == 0 or d2.std() == 0:
        return np.nan, int(tri.sum())
    return float(np.corrcoef(d1, d2)[0, 1]), int(tri.sum())


def summary(x, rng):
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    boot = np.median(x[rng.integers(0, len(x), (NBOOT, len(x)))], axis=1)
    return {"n_discharges": int(len(x)), "median": float(np.median(x)),
            "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
            "quartiles": [float(q) for q in np.percentile(x, [25, 50, 75])],
            "share_below_0": float((x < 0).mean()), "values": [float(v) for v in x]}


def main():
    files = sorted(glob.glob(str(DATA / "*.csv")))
    ti, vt, npair = [], [], {"T_i": 0, "V_rot": 0}
    for f in files:
        df = pd.read_csv(f, usecols=["time", "CES_TI", "CES_VT"])
        t = df["time"].to_numpy(float)
        y = df["CES_TI"].to_numpy(float).copy()
        y[y > 3000.0] = np.nan
        r, n = regime(t, y)
        ti.append(r); npair["T_i"] += n
        v = held_free(df["CES_VT"].to_numpy(float))
        v[np.abs(v) > 1000.0] = np.nan
        r, n = regime(t, v)
        vt.append(r); npair["V_rot"] += n
    rng = np.random.default_rng(SEED)
    res = {"n_files": len(files), "min_pairs": MIN_PAIRS, "pairs_total": npair,
           "T_i": summary(ti, rng), "V_rot": summary(vt, rng)}
    OUT.write_text(json.dumps(res, indent=1), encoding="utf-8")
    for k in ("T_i", "V_rot"):
        s = res[k]
        print(k, "n", s["n_discharges"], "median %.3f [%.3f, %.3f]" % (s["median"], *s["ci95"]),
              "quartiles", [round(q, 3) for q in s["quartiles"]], "pairs", npair[k])


if __name__ == "__main__":
    main()
