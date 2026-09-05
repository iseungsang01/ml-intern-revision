# -*- coding: utf-8 -*-
"""Is the CES missingness a beam-modulation duty cycle?

NN-CES (J.K. Lee et al., Fusion Eng. Des. 222:115518, 2025) states the mechanism
plainly: KSTAR runs charge exchange spectroscopy off the **main heating beam,
modulated**, rather than off a dedicated diagnostic beam. Background subtraction
therefore needs beam-off frames, and the price is paid in time coverage.

If that is what produces the gaps in our 641 files, the missingness is an
*instrument protocol*, not a random dropout -- and a modulation-free analysis
would attack it at the source rather than downstream. That would make it the
cheapest lever on the list, so it is worth measuring before it is claimed.

The measurement, on the confirmed data treatment's own grid:

  1. run lengths of observed and missing stretches, per target, inside
     contiguous 10 ms blocks (a gap >= 0.5 s starts a new block, as in dataset.py);
  2. the dominant period of the observation indicator, by FFT on the
     mean-removed mask, reported only where a block is long enough to resolve it;
  3. how far Ti and V_rot missingness agree, computed twice -- on the raw NaN
     masks and on the confirmed treatment's masks (3 keV cut applied). One shared
     shutter (beam modulation) should hit both channels on the same frames; a
     per-channel fit failure should not, and the difference between the two
     agreements is exactly what our own cut adds;
  3b. the short-lag autocorrelation of the observation mask, where a fixed duty
     cycle would show as a periodic ripple rather than a smooth decay;
  4. the same three statistics for held (forward-filled) V_rot runs, which are
     the other half of the coverage problem.

Everything here is descriptive. No model, no split, no TEST involvement.

Usage (repo root):  py ces_prediction/analyze_ces_duty_cycle.py
Writes: data/.ces_duty_cycle.json
"""

from __future__ import annotations

import json
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

DATA_DIR = os.environ.get("CES_DATA_DIR") or os.path.join(
    os.path.dirname(HERE), "data")
OUT = os.path.join(os.path.dirname(HERE), "data", ".ces_duty_cycle.json")

BLOCK_GAP_S = 0.5          # dataset.py's contiguous-block rule
SPIKE_CUT_EV = 3000.0      # confirmed protocol's Ti fit-failure cut
MIN_BLOCK = 24             # rows needed before a period is worth estimating


def shot_files(data_dir: str) -> list[str]:
    return sorted(
        os.path.join(data_dir, f)
        for f in os.listdir(data_dir)
        if f.lower().endswith(".csv") and f.lower().startswith("s")
    )


def blocks_of(t: np.ndarray) -> list[tuple[int, int]]:
    """Index ranges of contiguous 10 ms stretches."""
    if len(t) == 0:
        return []
    cuts = np.where(np.diff(t) >= BLOCK_GAP_S)[0] + 1
    edges = [0] + cuts.tolist() + [len(t)]
    return [(edges[i], edges[i + 1]) for i in range(len(edges) - 1)]


def run_lengths(mask: np.ndarray) -> tuple[Counter, Counter]:
    """Run lengths of True and of False stretches."""
    on, off = Counter(), Counter()
    if len(mask) == 0:
        return on, off
    start, cur = 0, bool(mask[0])
    for i in range(1, len(mask) + 1):
        if i == len(mask) or bool(mask[i]) != cur:
            (on if cur else off)[i - start] += 1
            if i < len(mask):
                start, cur = i, bool(mask[i])
    return on, off


def dominant_period(mask: np.ndarray) -> float | None:
    """Period (in 10 ms steps) of the strongest FFT component of the mask.

    Returns None when the block is too short, or the mask is constant, or the
    peak sits at the DC / Nyquist bin where a 'period' is meaningless.
    """
    n = len(mask)
    if n < MIN_BLOCK:
        return None
    x = mask.astype(float)
    if x.std() < 1e-9:
        return None
    x = x - x.mean()
    spec = np.abs(np.fft.rfft(x)) ** 2
    if len(spec) < 3:
        return None
    k = int(np.argmax(spec[1:])) + 1          # skip DC
    if k >= len(spec) - 1:                     # Nyquist: alternating, period 2
        return 2.0
    return float(n) / k


def held_mask(values: np.ndarray, observed: np.ndarray) -> np.ndarray:
    """True where a value merely repeats the previous observed value."""
    out = np.zeros(len(values), dtype=bool)
    prev = np.nan
    for i in range(len(values)):
        if not observed[i]:
            continue
        if np.isfinite(prev) and values[i] == prev:
            out[i] = True
        prev = values[i]
    return out


def summarise(counter: Counter) -> dict:
    if not counter:
        return {"n_runs": 0}
    lengths = np.repeat(list(counter.keys()), list(counter.values())).astype(float)
    return {
        "n_runs": int(lengths.size),
        "mean": round(float(lengths.mean()), 3),
        "median": float(np.median(lengths)),
        "p90": float(np.percentile(lengths, 90)),
        "max": float(lengths.max()),
        "top5": [[int(k), int(v)] for k, v in counter.most_common(5)],
    }


def main() -> None:
    files = shot_files(DATA_DIR)
    if not files:
        raise SystemExit("no shot CSVs under %s -- set CES_DATA_DIR" % DATA_DIR)

    on_ti, off_ti = Counter(), Counter()
    on_vt, off_vt = Counter(), Counter()
    held_vt_runs, held_ti_runs = Counter(), Counter()
    periods_ti, periods_vt = [], []
    rows = both_obs = ti_only = vt_only = neither = 0
    n_blocks = 0
    agree_num = agree_den = raw_agree_num = 0
    acf_ti = np.zeros(11); acf_vt = np.zeros(11); acf_n = 0

    for path in files:
        df = pd.read_csv(path)
        if not {"time", "CES_TI", "CES_VT"} <= set(df.columns):
            continue
        t = df["time"].to_numpy(dtype=float)
        ti = df["CES_TI"].to_numpy(dtype=float)
        vt = df["CES_VT"].to_numpy(dtype=float)

        raw_ti = np.isfinite(ti)                          # instrument as delivered
        obs_ti = raw_ti & (ti <= SPIKE_CUT_EV)            # confirmed treatment
        obs_vt = np.isfinite(vt)
        raw_agree_num += int(np.sum(raw_ti == obs_vt))

        rows += len(t)
        both_obs += int(np.sum(obs_ti & obs_vt))
        ti_only += int(np.sum(obs_ti & ~obs_vt))
        vt_only += int(np.sum(~obs_ti & obs_vt))
        neither += int(np.sum(~obs_ti & ~obs_vt))
        agree_num += int(np.sum(obs_ti == obs_vt))
        agree_den += len(t)

        for a, b in blocks_of(t):
            if b - a < 4:
                continue
            n_blocks += 1
            mi, mv = obs_ti[a:b], obs_vt[a:b]
            oi, fi = run_lengths(mi)
            ov, fv = run_lengths(mv)
            on_ti.update(oi); off_ti.update(fi)
            on_vt.update(ov); off_vt.update(fv)

            p = dominant_period(mi)
            if p is not None:
                periods_ti.append(p)
            p = dominant_period(mv)
            if p is not None:
                periods_vt.append(p)

            if b - a >= MIN_BLOCK:
                acf_n += 1
                for m, acc in ((mi, acf_ti), (mv, acf_vt)):
                    x = m.astype(float) - m.mean()
                    v0 = float(np.dot(x, x))
                    if v0 < 1e-9:
                        continue
                    for lag in range(1, 11):
                        acc[lag] += float(np.dot(x[:-lag], x[lag:])) / v0

            hv = held_mask(vt[a:b], mv)
            hi = held_mask(ti[a:b], mi)
            hv_on, _ = run_lengths(hv)
            hi_on, _ = run_lengths(hi)
            held_vt_runs.update({k: v for k, v in hv_on.items() if k > 0})
            held_ti_runs.update({k: v for k, v in hi_on.items() if k > 0})

    def pstats(ps):
        if not ps:
            return {"n": 0}
        arr = np.array(ps, dtype=float)
        hist = Counter(np.round(arr).astype(int).tolist())
        return {
            "n": int(arr.size),
            "median_steps": float(np.median(arr)),
            "median_ms": float(np.median(arr) * 10.0),
            "iqr_steps": [float(np.percentile(arr, 25)), float(np.percentile(arr, 75))],
            "top5_steps": [[int(k), int(v)] for k, v in hist.most_common(5)],
        }

    res = {
        "files": len(files),
        "rows": rows,
        "grid_ms": 10,
        "spike_cut_eV": SPIKE_CUT_EV,
        "blocks": n_blocks,
        "coverage": {
            "both_observed": both_obs,
            "Ti_only": ti_only,
            "Vrot_only": vt_only,
            "neither": neither,
            "frac_both": round(both_obs / max(rows, 1), 4),
            "frac_Ti_only": round(ti_only / max(rows, 1), 4),
            "frac_Vrot_only": round(vt_only / max(rows, 1), 4),
            "frac_neither": round(neither / max(rows, 1), 4),
            "mask_agreement_after_cut": round(agree_num / max(agree_den, 1), 4),
            "mask_agreement_raw": round(raw_agree_num / max(agree_den, 1), 4),
        },
        "mask_autocorrelation": {
            "blocks": acf_n,
            "Ti": [round(float(v / max(acf_n, 1)), 4) for v in acf_ti[1:]],
            "Vrot": [round(float(v / max(acf_n, 1)), 4) for v in acf_vt[1:]],
        },
        "runs": {
            "Ti_observed": summarise(on_ti),
            "Ti_missing": summarise(off_ti),
            "Vrot_observed": summarise(on_vt),
            "Vrot_missing": summarise(off_vt),
            "Vrot_held": summarise(held_vt_runs),
            "Ti_held": summarise(held_ti_runs),
        },
        "dominant_period": {"Ti": pstats(periods_ti), "Vrot": pstats(periods_vt)},
    }

    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=2)

    c = res["coverage"]
    print("files %d  rows %d  blocks %d" % (res["files"], res["rows"], res["blocks"]))
    print("coverage: both %.3f  Ti only %.3f  Vrot only %.3f  neither %.3f"
          % (c["frac_both"], c["frac_Ti_only"], c["frac_Vrot_only"], c["frac_neither"]))
    print("mask agreement: raw %.3f  after 3 keV cut %.3f"
          % (c["mask_agreement_raw"], c["mask_agreement_after_cut"]))
    a = res["mask_autocorrelation"]
    print("mask ACF lag1..10  Ti  %s" % [round(v, 3) for v in a["Ti"]])
    print("mask ACF lag1..10  Vrot %s" % [round(v, 3) for v in a["Vrot"]])
    for k in ("Ti_observed", "Ti_missing", "Vrot_observed", "Vrot_missing", "Vrot_held"):
        s = res["runs"][k]
        if s["n_runs"]:
            print("%-14s runs %6d  median %4.1f  mean %5.2f  p90 %5.1f  top %s"
                  % (k, s["n_runs"], s["median"], s["mean"], s["p90"], s["top5"][:3]))
    for k in ("Ti", "Vrot"):
        p = res["dominant_period"][k]
        if p["n"]:
            print("%-5s dominant period: median %.1f steps = %.0f ms  top %s"
                  % (k, p["median_steps"], p["median_ms"], p["top5_steps"][:3]))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
