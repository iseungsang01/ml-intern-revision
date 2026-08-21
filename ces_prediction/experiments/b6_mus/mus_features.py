"""B.6 feature extraction: raw microsecond streams -> per-10 ms-row features.

Implements PREREGISTRATION_B6.md sec. 3.1 (MC, the H1 confirmatory features) and
sec. 3.2 (BES/ECEI high-band, the T_i-path exploratory features). Every function here is
pure ((t, x) arrays in, features out) so the whole module is unit-testable with synthetic
signals before a single raw byte arrives -- `tests/test_b6_features.py` does exactly
that. The raw-format-specific part lives in `raw_loader.py` and is the ONE piece that
must wait for delivery.

Implementation constants (fixed BEFORE data arrival; the pre-registration froze the
feature list and band edges, these pin the remaining numerical choices):

  window        each grid row t gets the causal window (t - 10 ms, t] -- never a sample
                at strictly later time than the row it feeds
  band power    one Hann-windowed periodogram per 10 ms window (frequency resolution
                100 Hz); band power = sum of the power spectrum over the band's bins;
                stored as log1p(power)
  cross terms   Welch over 16 non-overlapping Hann sub-segments of the window;
                magnitude-squared coherence |S12|^2 / (S11 * S22); the mode search
                runs over 3..300 kHz
  mode feature  dominant peak of |S12| in the search range -> (log1p(f_kHz),
                sign(cross-phase at the peak), valid); valid requires the peak's
                coherence to clear max(0.5, the alpha = 0.01 family-wise noise
                threshold over the searched bins) -- picking the |S12| maximum is a
                selection over ~hundreds of bins, and with few averages independent
                noise clears a bare 0.5 too often (verified by the unit test);
                invalid rows carry (0, 0, 0)
  z-scoring     RMS and band powers are per-shot z-scored AT LOAD TIME by
                seq_data._load_extra_vt; the (sign, valid) channels are exempt, and the
                log1p frequency channel is exempt too (bounded, and zero-when-invalid
                must stay zero). The extractor itself stores raw values.

MC channel layout (K = 15), written to `mus_features_s{shot}.npz` with a
`feature_meta.json` beside it:

    0      rms coil 1                    1      rms coil 2
    2..6   log1p band power coil 1       7..11  log1p band power coil 2
           (bands 0.1-3 / 3-10 / 10-30 / 30-70 / 70-300 kHz)
    12     log1p mode frequency [kHz]    13     cross-phase sign (-1/0/+1)
    14     valid flag (0/1)
    z-score exempt: 12, 13, 14

Bands whose upper edge exceeds the delivered Nyquist are DROPPED, never moved
(PREREGISTRATION_B6.md sec. 3.1 slot rule); dropping shrinks K and is recorded in the
meta so the loader and the model agree.
"""

from pathlib import Path
import json

import numpy as np

WINDOW_S = 0.01
BANDS_KHZ = ((0.1, 3.0), (3.0, 10.0), (10.0, 30.0), (30.0, 70.0), (70.0, 300.0))
MODE_SEARCH_KHZ = (3.0, 300.0)
COHERENCE_MIN = 0.5
WELCH_SEGMENTS = 16
FAMILY_ALPHA = 0.01
MC_COILS = ("MC1T03", "MC1T16")


def causal_window(t_raw, t_row, width=WINDOW_S):
    """Index slice of samples in (t_row - width, t_row]. Strictly causal."""
    lo = np.searchsorted(t_raw, t_row - width, side="right")
    hi = np.searchsorted(t_raw, t_row, side="right")
    return slice(lo, hi)


def _hann(n):
    return np.hanning(n) if n > 1 else np.ones(n)


def band_powers(x, fs, bands_khz):
    """One Hann periodogram over the window; per band, sum of the power spectrum.

    Returns raw powers (callers log1p). A band with no bin inside it (too-short
    window or too-low fs) returns 0.0 -- the loader's z-scoring tolerates constants.
    """
    n = len(x)
    if n < 4:
        return np.zeros(len(bands_khz))
    w = _hann(n)
    xw = (x - x.mean()) * w
    spec = np.abs(np.fft.rfft(xw)) ** 2 / (np.sum(w ** 2) * fs)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    out = np.zeros(len(bands_khz))
    for i, (lo, hi) in enumerate(bands_khz):
        sel = (freqs >= lo * 1e3) & (freqs < hi * 1e3)
        if sel.any():
            out[i] = float(spec[sel].sum())
    return out


def cross_mode(x1, x2, fs, n_seg=WELCH_SEGMENTS,
               search_khz=MODE_SEARCH_KHZ, coh_min=COHERENCE_MIN):
    """(mode frequency [Hz], cross-phase sign, valid) from two coils' window samples.

    Welch over `n_seg` non-overlapping Hann sub-segments: averaged auto and cross
    spectra, magnitude-squared coherence, and the dominant |S12| peak inside the search
    band. The sign of the cross-phase at the peak is the rotation-direction proxy; a
    peak whose coherence is below `coh_min` returns (0.0, 0, 0).
    """
    n = min(len(x1), len(x2))
    seg = n // n_seg
    if seg < 8:
        return 0.0, 0, 0
    w = _hann(seg)
    s11 = s22 = s12 = 0.0
    for k in range(n_seg):
        a = (x1[k * seg:(k + 1) * seg] - x1[k * seg:(k + 1) * seg].mean()) * w
        b = (x2[k * seg:(k + 1) * seg] - x2[k * seg:(k + 1) * seg].mean()) * w
        fa, fb = np.fft.rfft(a), np.fft.rfft(b)
        s11 = s11 + np.abs(fa) ** 2
        s22 = s22 + np.abs(fb) ** 2
        s12 = s12 + fa * np.conj(fb)
    freqs = np.fft.rfftfreq(seg, d=1.0 / fs)
    sel = (freqs >= search_khz[0] * 1e3) & (freqs <= search_khz[1] * 1e3)
    if not sel.any():
        return 0.0, 0, 0
    mag = np.abs(s12)
    idx = np.flatnonzero(sel)[np.argmax(mag[sel])]
    denom = float(s11[idx] * s22[idx])
    coh = float(mag[idx] ** 2 / denom) if denom > 0 else 0.0
    # The |S12| maximum is a selection over `n_bins` bins; under independent noise
    # P(coherence > c) = (1 - c)^(n_seg - 1) per bin, so the family-wise alpha = 0.01
    # floor is 1 - (alpha / n_bins)^(1/(n_seg - 1)). The pre-registered 0.5 stays as
    # the physical minimum; the floor only ever raises it.
    n_bins = int(sel.sum())
    noise_floor = 1.0 - (FAMILY_ALPHA / n_bins) ** (1.0 / max(n_seg - 1, 1))
    if coh < max(coh_min, noise_floor):
        return 0.0, 0, 0
    phase = float(np.angle(s12[idx]))
    return float(freqs[idx]), int(np.sign(phase)), 1


def usable_bands(fs, bands_khz=BANDS_KHZ):
    """Drop (never move) bands whose upper edge exceeds Nyquist -- the sec. 3.1 slot rule."""
    nyq_khz = fs / 2.0 / 1e3
    return tuple(b for b in bands_khz if b[1] <= nyq_khz)


def mc_features_for_row(w1, w2, fs, bands):
    """One grid row's MC feature vector from the two coils' causal-window samples."""
    n_b = len(bands)
    out = np.zeros(2 + 2 * n_b + 3)             # 2 rms + 2*n_b band powers + 3 mode
    out[0] = float(np.sqrt(np.mean(w1 ** 2))) if len(w1) else 0.0
    out[1] = float(np.sqrt(np.mean(w2 ** 2))) if len(w2) else 0.0
    out[2:2 + n_b] = np.log1p(band_powers(w1, fs, bands))
    out[2 + n_b:2 + 2 * n_b] = np.log1p(band_powers(w2, fs, bands))
    f_hz, sign, valid = cross_mode(w1, w2, fs)
    out[2 + 2 * n_b] = np.log1p(f_hz / 1e3)
    out[2 + 2 * n_b + 1] = sign
    out[2 + 2 * n_b + 2] = valid
    return out


def extract_mc_shot(t1, x1, t2, x2, fs, grid_times):
    """All grid rows of one shot -> (n_rows, K) MC features + the meta dict."""
    bands = usable_bands(fs)
    k = 2 + 2 * len(bands) + 3
    feat = np.zeros((len(grid_times), k), dtype=np.float32)
    for j, t_row in enumerate(grid_times):
        feat[j] = mc_features_for_row(x1[causal_window(t1, t_row)],
                                      x2[causal_window(t2, t_row)], fs, bands)
    meta = {"k": k, "z_exempt_channels": [k - 3, k - 2, k - 1],
            "bands_khz": [list(b) for b in bands], "fs_hz": float(fs),
            "coils": list(MC_COILS), "window_s": WINDOW_S,
            "welch_segments": WELCH_SEGMENTS, "coherence_min": COHERENCE_MIN}
    return feat, meta


def highband_features_for_row(w, fs, bands):
    """Sec. 3.2 exploratory: log1p band powers of one channel's causal window."""
    return np.log1p(band_powers(w, fs, bands))


def write_shot(out_dir, shot, grid_times, feat, meta):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / f"mus_features_s{shot}.npz",
             time=np.asarray(grid_times, dtype=np.float64),
             feat=np.asarray(feat, dtype=np.float32))
    meta_path = out_dir / "feature_meta.json"
    if meta_path.exists():
        old = json.loads(meta_path.read_text(encoding="utf-8"))
        if old != meta:
            raise SystemExit(f"FATAL: {meta_path} disagrees with this shot's meta -- "
                             "one feature dir must hold one consistent layout.")
    else:
        meta_path.write_text(json.dumps(meta, indent=1), encoding="utf-8")


def main():
    """CLI: extract MC features for the frozen twelve from delivered raw data.

    Runs only once `raw_loader.load_raw` is implemented (on-arrival slot). Grid times
    come from the shot CSVs so alignment with the training pipeline is exact.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.path.insert(1, str(Path(__file__).resolve().parents[1] / "hires_shots"))
    from raw_loader import load_raw, RAW_READY  # noqa: E402
    from folds import TEST, POOL, COMPANIONS  # noqa: E402
    import pandas as pd  # noqa: E402

    if not RAW_READY:
        raise SystemExit("raw_loader is a stub -- fill it in when the raw data arrives "
                         "(PREREGISTRATION_B6.md on-arrival slot).")
    root = Path(__file__).resolve().parents[3]
    out_dir = root / "data" / ".b6_features"
    for shot in (*TEST, *POOL, *COMPANIONS):
        grid_times = pd.read_csv(root / "data" / f"s{shot}.csv", usecols=["time"])[
            "time"].to_numpy(float)
        (t1, x1), fs1 = load_raw(shot, MC_COILS[0])
        (t2, x2), fs2 = load_raw(shot, MC_COILS[1])
        if fs1 != fs2:
            raise SystemExit(f"FATAL: coil sampling rates differ for s{shot}: {fs1} vs {fs2}")
        feat, meta = extract_mc_shot(t1, x1, t2, x2, fs1, grid_times)
        write_shot(out_dir, shot, grid_times, feat, meta)
        print(f"[b6f] s{shot}: {feat.shape[0]} rows x {feat.shape[1]} channels")
    print(f"[b6f] wrote {out_dir}")


if __name__ == "__main__":
    main()
