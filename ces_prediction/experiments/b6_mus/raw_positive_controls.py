"""PREREGISTRATION_B6.md sec. 1.4: the raw data must reproduce published spectral facts.

Positive controls before any training -- a check that fails here is an acquisition
problem, not an experiment. The four checks are fixed in the pre-registration; the
detection functions are pure and unit-tested with synthetic signals
(tests/test_b6_features.py), so when the raw data arrives only `raw_loader` is new code.

  1  #31923  BES_0206, 3..6 s: a coherent peak in the 30..70 kHz band (the published
             ~50 kHz WCM, Nucl. Fusion adacfc fig.2)
  2  #32027  BES channel pair around the L/H transition: cross-power in 100..300 kHz
             (PanoMHD fig.7); skipped automatically if the delivered rate cannot see it
  3  #31921  MC envelope jump near 7.3 s (the 100 Hz grid already shows 1.7 -> 13.9)
  4  every channel: windowed raw RMS^2 vs the decimated CSV's per-window mean square
             (for a random-phase snapshot both estimate A^2/2 -> ratio near 1)
"""

from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from mus_features import band_powers, causal_window  # noqa: E402

PEAK_PROMINENCE = 3.0     # band power must exceed the median of the other bands 3x
ENVELOPE_JUMP = 3.0       # late/early MC envelope ratio for #31921
RMS_RATIO_BAND = (0.5, 2.0)


def wcm_peak_present(t, x, fs, t_lo=3.0, t_hi=6.0, band=(30.0, 70.0)):
    """Is there a 30..70 kHz feature well above the neighbouring bands in [t_lo, t_hi]?"""
    sel = (t >= t_lo) & (t <= t_hi)
    if sel.sum() < 1024:
        return False, 0.0
    others = ((3.0, 10.0), (10.0, 30.0), (70.0, 300.0))
    p = band_powers(x[sel], fs, (band, *others))
    ref = np.median(p[1:][p[1:] > 0]) if (p[1:] > 0).any() else 0.0
    ratio = float(p[0] / ref) if ref > 0 else float("inf") if p[0] > 0 else 0.0
    return ratio >= PEAK_PROMINENCE, ratio


def highband_crosspower_present(t1, x1, t2, x2, fs, band=(100.0, 300.0)):
    """Any coherent 100..300 kHz cross-power between two channels (32027 check)."""
    from mus_features import cross_mode
    n = min(len(x1), len(x2))
    if n < 4096 or fs / 2.0 < band[0] * 1e3:
        return None, 0.0        # None = undecidable at this rate; recorded, not failed
    f_hz, _sign, valid = cross_mode(x1[:n], x2[:n], fs,
                                    search_khz=band, coh_min=0.3)
    return bool(valid), f_hz / 1e3


def mc_envelope_jump(t, x, t_break=7.3, half=0.5):
    """RMS(t_break..t_break+half) / RMS(t_break-half..t_break) for #31921."""
    early = (t >= t_break - half) & (t < t_break)
    late = (t >= t_break) & (t < t_break + half)
    if early.sum() < 100 or late.sum() < 100:
        return False, 0.0
    r_e = float(np.sqrt(np.mean(x[early] ** 2)))
    r_l = float(np.sqrt(np.mean(x[late] ** 2)))
    ratio = r_l / r_e if r_e > 0 else float("inf")
    return ratio >= ENVELOPE_JUMP, ratio


def rms_relation(t_raw, x_raw, grid_times, csv_values, n_windows=200):
    """Windowed raw mean-square vs the decimated snapshots' mean square (both ~ A^2/2)."""
    take = np.linspace(0, len(grid_times) - 1, min(n_windows, len(grid_times))).astype(int)
    raw_ms, snap_sq = [], []
    for j in take:
        w = x_raw[causal_window(t_raw, grid_times[j])]
        if len(w) and np.isfinite(csv_values[j]):
            raw_ms.append(float(np.mean(w ** 2)))
            snap_sq.append(float(csv_values[j] ** 2))
    if len(raw_ms) < 30:
        return False, 0.0
    ratio = float(np.mean(snap_sq) / np.mean(raw_ms)) if np.mean(raw_ms) > 0 else 0.0
    return RMS_RATIO_BAND[0] <= ratio <= RMS_RATIO_BAND[1], ratio


def main():
    from raw_loader import load_raw, RAW_READY
    import pandas as pd
    if not RAW_READY:
        raise SystemExit("raw_loader is a stub -- the on-arrival slot is not filled yet.")
    root = Path(__file__).resolve().parents[3]
    results = {}

    (t, x), fs = load_raw(31923, "BES_0206")
    results["wcm_31923"] = wcm_peak_present(t, x, fs)

    (ta, xa), fsa = load_raw(32027, "BES_0101")
    (tb, xb), _ = load_raw(32027, "BES_0102")
    results["crosspower_32027"] = highband_crosspower_present(ta, xa, tb, xb, fsa)

    (tm, xm), _fsm = load_raw(31921, "MC1T03")
    results["envelope_31921"] = mc_envelope_jump(tm, xm)

    df = pd.read_csv(root / "data" / "s31921.csv", usecols=["time", "MC1T03"])
    results["rms_relation_31921"] = rms_relation(
        tm, xm, df["time"].to_numpy(float), df["MC1T03"].to_numpy(float))

    print(results)
    hard = [k for k, v in results.items() if v[0] is False]
    if hard:
        raise SystemExit(f"POSITIVE CONTROL FAILED: {hard} -- acquisition problem, "
                         "record in sec. 8 and do not train (PREREGISTRATION_B6.md 1.4)")
    print("[b6pc] all positive controls pass")


if __name__ == "__main__":
    main()
