"""PREREGISTRATION_B6.md sec. 1.4: the raw data must reproduce published spectral facts.

Positive controls before any training -- a check that fails here is an acquisition
problem, not an experiment. The checks are fixed in the pre-registration; the detection
functions are pure and unit-tested with synthetic signals (tests/test_b6_features.py),
and `raw_loader` is the only code that waited for the delivery.

  1  #31923  BES_0206, 3..6 s: a coherent peak in the 30..70 kHz band (the published
             ~50 kHz WCM, Nucl. Fusion adacfc fig.2)
  2  #32027  BES channel pair across the delivered span: coherent cross-power in
             100..300 kHz (PanoMHD fig.7), scanned window by window with the
             extractor's own cross-spectrum; bands above the delivered Nyquist are
             dropped, never moved
  3  #31921  MC envelope jump near 7.3 s (the 100 Hz grid already shows 1.7 -> 13.9)
  4  every channel of every delivered shot: windowed raw mean square vs the decimated
             CSV's snapshot mean square (for a random-phase snapshot both estimate
             A^2/2 -> ratio near 1)
  5  #32092  Mirnov coils in the ELM-free window around 7.6 s: EHO peaks at ~4 kHz
             (n = 1) and ~8 kHz (n = 2) (NF ae8679 table 1) -- the one published answer
             inside the 3..10 kHz band H1 tries to restore

On-arrival fills (2026-10-26, PREREGISTRATION_B6.md sec. 1.4 slots -- channel names and
windows, never the criteria): the delivery has BES_0105..0307 only, so control 2 uses the
poloidal pair (BES_0106, BES_0206) centred on the literature channel; control 2 passes
when >= 5 % of 10 ms windows carry a valid coherent peak in the band; control 5 uses
7.3..7.9 s and passes when the Welch auto-spectrum of either coil has local peaks within
+-1 kHz of 4 kHz and +-1.5 kHz of 8 kHz, each >= 3x its local background.
"""

from pathlib import Path
import json
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from mus_features import band_powers, causal_window, cross_mode, WINDOW_S  # noqa: E402

PEAK_PROMINENCE = 3.0     # band power must exceed the median of the other bands 3x
ENVELOPE_JUMP = 3.0       # late/early MC envelope ratio for #31921
RMS_RATIO_BAND = (0.5, 2.0)
CROSSPOWER_MIN_FRAC = 0.05
EHO_TARGETS_KHZ = ((4.0, 1.0), (8.0, 1.5))   # (centre, half-width) per NF ae8679 table 1
EHO_PROMINENCE = 3.0
EHO_WINDOW = (7.3, 7.9)                       # ELM-free marker t ~ 7.6 s


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


def wcm_bump_over_background(t, x, fs, t_lo=3.0, t_hi=6.0, seg=8192):
    """Supplementary (reported, not a criterion): Welch PSD in [t_lo, t_hi]; ratio of the
    30..70 kHz mean PSD to the log-log straight-line interpolation between the 10..30 and
    70..Nyquist bands -- a bump on a 1/f background."""
    sel = (t >= t_lo) & (t <= t_hi)
    f, p = welch_psd(x[sel], fs, seg)
    def band_mean(lo, hi):
        m = (f >= lo * 1e3) & (f < hi * 1e3)
        return float(np.exp(np.log(p[m]).mean())), float(np.exp(np.log(f[m]).mean()))
    p_lo, f_lo = band_mean(10.0, 30.0)
    p_hi, f_hi = band_mean(70.0, min(300.0, fs / 2e3))
    p_mid, f_mid = band_mean(30.0, 70.0)
    slope = (np.log(p_hi) - np.log(p_lo)) / (np.log(f_hi) - np.log(f_lo))
    p_interp = np.exp(np.log(p_lo) + slope * (np.log(f_mid) - np.log(f_lo)))
    return float(p_mid / p_interp)


def welch_psd(x, fs, seg):
    """Plain Welch PSD (Hann, 50 % overlap) -- numpy only."""
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean()
    step = seg // 2
    n_seg = max(1, (len(x) - seg) // step + 1)
    w = np.hanning(seg)
    acc = np.zeros(seg // 2 + 1)
    for k in range(n_seg):
        s = x[k * step:k * step + seg]
        if len(s) < seg:
            break
        acc += np.abs(np.fft.rfft(s * w)) ** 2
    acc /= max(n_seg, 1) * np.sum(w ** 2) * fs
    return np.fft.rfftfreq(seg, d=1.0 / fs), acc


def highband_crosspower_scan(t1, x1, t2, x2, fs, band=(100.0, 300.0), window=WINDOW_S):
    """Fraction of 10 ms windows with a valid coherent peak inside `band` (capped at the
    delivered Nyquist), using the extractor's own cross-spectrum. None when the band is
    entirely above Nyquist (undecidable at this rate; recorded, not failed)."""
    nyq_khz = fs / 2.0 / 1e3
    if nyq_khz <= band[0]:
        return None, 0.0, 0
    search = (band[0], min(band[1], nyq_khz))
    n = min(len(x1), len(x2))
    t_rows = np.arange(t1[0] + window, t1[n - 1], window)
    hits = 0
    freqs = []
    for tr in t_rows:
        sl = causal_window(t1[:n], tr, window)
        f_hz, _s, valid = cross_mode(x1[sl], x2[sl], fs, search_khz=search, coh_min=0.3)
        if valid:
            hits += 1
            freqs.append(f_hz / 1e3)
    frac = hits / max(len(t_rows), 1)
    return frac >= CROSSPOWER_MIN_FRAC, float(frac), int(len(t_rows))


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
    """Windowed raw mean-square vs the decimated snapshots' mean square (both ~ A^2/2).

    Only grid rows inside the delivered raw span are compared (rows outside have no
    raw window at all)."""
    inside = np.flatnonzero((grid_times >= t_raw[0] + WINDOW_S) & (grid_times <= t_raw[-1]))
    if len(inside) == 0:
        return False, 0.0
    take = inside[np.linspace(0, len(inside) - 1, min(n_windows, len(inside))).astype(int)]
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


def eho_peaks_present(t, x, fs, t_lo=EHO_WINDOW[0], t_hi=EHO_WINDOW[1], seg=16384):
    """Welch auto-spectrum in [t_lo, t_hi]; for each EHO target, the PSD maximum inside
    centre +- half-width over the median PSD of the 2 kHz-wide flanks on either side."""
    sel = (t >= t_lo) & (t <= t_hi)
    if sel.sum() < seg * 2:
        return False, {}
    f, p = welch_psd(x[sel], fs, seg)
    out = {}
    ok_all = True
    for centre, half in EHO_TARGETS_KHZ:
        inb = (f >= (centre - half) * 1e3) & (f <= (centre + half) * 1e3)
        flank = (((f >= (centre - half - 2.0) * 1e3) & (f < (centre - half) * 1e3)) |
                 ((f > (centre + half) * 1e3) & (f <= (centre + half + 2.0) * 1e3)))
        f_peak = float(f[inb][np.argmax(p[inb])]) / 1e3
        ratio = float(p[inb].max() / np.median(p[flank])) if flank.any() else 0.0
        out[f"{centre:g}kHz"] = {"peak_khz": f_peak, "prominence": ratio}
        ok_all = ok_all and ratio >= EHO_PROMINENCE
    return ok_all, out


def eho_mode_histogram(t1, x1, t2, x2, fs, t_lo=EHO_WINDOW[0], t_hi=EHO_WINDOW[1]):
    """The extractor's own mode feature, window by window, inside the ELM-free stretch:
    share of valid rows and the modal dominant frequency (kHz, 1 kHz bins)."""
    rows = np.arange(t_lo + WINDOW_S, t_hi, WINDOW_S)
    freqs = []
    for tr in rows:
        f_hz, _s, valid = cross_mode(x1[causal_window(t1, tr)], x2[causal_window(t2, tr)], fs)
        if valid:
            freqs.append(f_hz / 1e3)
    if not freqs:
        return {"valid_frac": 0.0, "modal_khz": None, "n_rows": int(len(rows))}
    h, edges = np.histogram(freqs, bins=np.arange(0, 301, 1.0))
    return {"valid_frac": float(len(freqs) / len(rows)),
            "modal_khz": float(edges[int(np.argmax(h))]),
            "median_khz": float(np.median(freqs)), "n_rows": int(len(rows))}


def a(s):
    return str(s).encode("ascii", "replace").decode("ascii")


def main():
    from raw_loader import load_raw, RAW_READY, DIAGS
    sys.path.insert(0, str(HERE.parents[0] / "hires_shots"))
    from folds import TEST, POOL, COMPANIONS
    import pandas as pd
    if not RAW_READY:
        raise SystemExit("raw_loader is a stub -- the on-arrival slot is not filled yet.")
    root = Path(__file__).resolve().parents[3]
    results = {}

    (t, x), fs = load_raw(31923, "BES_0206")
    results["wcm_31923"] = wcm_peak_present(t, x, fs)
    results["wcm_31923_bump_over_1f"] = (None, wcm_bump_over_background(t, x, fs))

    (ta, xa), fsa = load_raw(32027, "BES_0106")
    (tb, xb), _ = load_raw(32027, "BES_0206")
    results["crosspower_32027"] = highband_crosspower_scan(ta, xa, tb, xb, fsa)

    (tm, xm), fsm = load_raw(31921, "MC1T03")
    results["envelope_31921"] = mc_envelope_jump(tm, xm)

    # control 4 over every channel of every delivered shot
    rms_table = {}
    worst = (True, 1.0, "")
    for shot in (*TEST, *POOL, *COMPANIONS):
        df = pd.read_csv(root / "data" / f"s{shot}.csv")
        g = df["time"].to_numpy(float)
        for diag in DIAGS:
            for ch in [c for c in df.columns if c.startswith(diag if diag != "MC" else "MC")
                       and (diag != "MC" or not c.startswith("ECEI"))]:
                if diag == "BES" and not ch.startswith("BES_"):
                    continue
                if diag == "ECEI" and not ch.startswith("ECEI_"):
                    continue
                (tr, xr), _ = load_raw(shot, ch)
                ok, ratio = rms_relation(tr, xr, g, df[ch].to_numpy(float))
                rms_table[f"s{shot}/{ch}"] = (ok, ratio)
                if not ok and (worst[0] or abs(np.log(max(ratio, 1e-12))) > abs(np.log(worst[1]))):
                    worst = (False, ratio, f"s{shot}/{ch}")
    n_ok = sum(v[0] for v in rms_table.values())
    results["rms_relation_all"] = (n_ok == len(rms_table), f"{n_ok}/{len(rms_table)} channels in band")

    (t3, x3), fs3 = load_raw(32092, "MC1T03")
    (t16, x16), _ = load_raw(32092, "MC1T16")
    ok3, d3 = eho_peaks_present(t3, x3, fs3)
    ok16, d16 = eho_peaks_present(t16, x16, fs3)
    results["eho_32092"] = (ok3 or ok16, {"MC1T03": d3, "MC1T16": d16})
    results["eho_32092_mode_feature"] = (None, eho_mode_histogram(t3, x3, t16, x16, fs3))

    for k, v in results.items():
        print(a(f"  {k:28s} pass={v[0]}  {v[1]}"))
    hard = [k for k, v in results.items() if v[0] is False]
    dst = root / "data" / ".b6_positive_controls.json"
    dst.write_text(json.dumps({"results": {k: [v[0], v[1]] for k, v in results.items()},
                               "rms_table": rms_table, "worst_rms": list(worst)},
                              indent=1, default=str), encoding="utf-8")
    print(a(f"[b6pc] wrote {dst}"))
    if hard:
        raise SystemExit(f"POSITIVE CONTROL FAILED: {hard} -- acquisition problem, "
                         "record in sec. 8 and do not train (PREREGISTRATION_B6.md 1.4)")
    print("[b6pc] all positive controls pass")


if __name__ == "__main__":
    main()
