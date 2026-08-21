"""B.6 prebuild verification: the feature extractor and the A1 plumbing, on synthetic
signals -- so every numerical choice is proven BEFORE the first raw byte arrives.
These are unit tests of pure functions (known sine -> known band; known rotation ->
known phase sign), not experiments; no experimental claim ever comes from them."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "ces_prediction" / "experiments" / "b6_mus"))
sys.path.insert(0, str(REPO / "ces_prediction" / "experiments" / "seq"))
sys.path.insert(0, str(REPO / "ces_prediction"))

from mus_features import (BANDS_KHZ, band_powers, causal_window, cross_mode,  # noqa: E402
                          extract_mc_shot, usable_bands, write_shot)

FS = 2_000_000.0


def _sine(freq_hz, n, fs=FS, phase=0.0):
    t = np.arange(n) / fs
    return t, np.sin(2 * np.pi * freq_hz * t + phase)


def test_band_power_lands_in_the_right_band():
    _, x = _sine(20_000.0, 20_000)          # 20 kHz -> band (10, 30]
    p = band_powers(x, FS, BANDS_KHZ)
    assert int(np.argmax(p)) == 2
    assert p[2] > 100 * (p[0] + p[1] + p[3] + p[4])


def test_cross_mode_finds_frequency_and_phase_sign():
    n = 20_000
    _, x1 = _sine(50_000.0, n)
    _, x2 = _sine(50_000.0, n, phase=-np.pi / 4)
    rng = np.random.default_rng(0)
    x1 = x1 + 0.05 * rng.standard_normal(n)
    x2 = x2 + 0.05 * rng.standard_normal(n)
    f, sign, valid = cross_mode(x1, x2, FS)
    assert valid == 1
    assert abs(f - 50_000.0) < 2_000.0
    f2, sign2, valid2 = cross_mode(x2, x1, FS)
    assert valid2 == 1 and sign2 == -sign and sign != 0


def test_cross_mode_incoherent_is_flagged_invalid():
    rng = np.random.default_rng(1)
    f, sign, valid = cross_mode(rng.standard_normal(20_000), rng.standard_normal(20_000), FS)
    assert (f, sign, valid) == (0.0, 0, 0)


def test_causal_window_never_reads_the_future():
    t = np.arange(0, 0.1, 1 / FS)
    x = np.sin(2 * np.pi * 20_000.0 * t)
    row_t = 0.05
    sl = causal_window(t, row_t)
    base = band_powers(x[sl], FS, BANDS_KHZ)
    x_future = x.copy()
    x_future[t > row_t] = 1e6                      # corrupt the future only
    assert np.array_equal(band_powers(x_future[causal_window(t, row_t)], FS, BANDS_KHZ), base)
    assert t[sl][-1] <= row_t and t[sl][0] > row_t - 0.01


def test_usable_bands_drops_never_moves():
    kept = usable_bands(100_000.0)                 # Nyquist 50 kHz
    assert kept == BANDS_KHZ[:3]                   # (0.1,3) (3,10) (10,30) survive


def test_extract_roundtrip_through_the_loader(tmp_path, monkeypatch):
    grid_times = np.round(np.arange(1.0, 1.2, 0.01), 4)
    n = int(0.25 * FS)
    t_raw = 0.95 + np.arange(n) / FS
    _, x1 = _sine(50_000.0, n)
    _, x2 = _sine(50_000.0, n, phase=-np.pi / 3)
    feat, meta = extract_mc_shot(t_raw, x1, t_raw, x2, FS, grid_times)
    assert feat.shape == (len(grid_times), meta["k"]) and meta["k"] == 15
    assert meta["z_exempt_channels"] == [12, 13, 14]
    assert (feat[:, 14] == 1).all()                # coherent pair -> valid everywhere
    write_shot(tmp_path, 99999, grid_times, feat, meta)

    monkeypatch.setenv("CES_SEQ_EXTRA_VT", str(tmp_path))
    import importlib
    import seq_data
    importlib.reload(seq_data)
    k, exempt, d = seq_data.extra_vt_meta()
    assert k == 15 and set(exempt) == {12, 13, 14}
    out = seq_data._load_extra_vt("s99999.csv", grid_times, k, exempt, d)
    assert out.shape == (len(grid_times), 15)
    # scaled channels are z-scored within the shot; exempt channels pass through raw
    assert abs(out[:, 0].mean()) < 1e-9
    assert np.array_equal(out[:, 14], feat[:, 14])
    with pytest.raises(SystemExit):                # wholesale misalignment must be loud
        seq_data._load_extra_vt("s99999.csv", grid_times + 5.0, k, exempt, d)
    monkeypatch.delenv("CES_SEQ_EXTRA_VT")
    importlib.reload(seq_data)


def test_v2_routing_extras_reach_vrot_only():
    import torch
    from model_seq_v2 import SeqCESLSTMv2
    base = SeqCESLSTMv2(n_extra_vt=0)
    assert base.n_params == 357_570                # frozen B.1 checkpoints must reload
    model = SeqCESLSTMv2(n_extra_vt=15)
    model.eval()
    x = torch.randn(1, 40, 22 + 15)
    with torch.no_grad():
        out_a = model(x)
        x2 = x.clone()
        x2[..., 22:] = torch.randn_like(x2[..., 22:])
        out_b = model(x2)
    assert torch.equal(out_a[..., 0], out_b[..., 0])          # T_i blind to the extras
    assert not torch.allclose(out_a[..., 1], out_b[..., 1])   # V_rot sees them


def test_positive_control_detectors_fire_and_stay_quiet():
    from raw_positive_controls import mc_envelope_jump, wcm_peak_present
    t = np.arange(0, 8.0, 1 / 100_000.0)
    rng = np.random.default_rng(2)
    x = 0.1 * rng.standard_normal(len(t))
    x[(t >= 3.0) & (t <= 6.0)] += np.sin(2 * np.pi * 48_000.0 * t[(t >= 3.0) & (t <= 6.0)])
    ok, ratio = wcm_peak_present(t, x, 100_000.0 * 2)          # fs must see 48 kHz
    t = np.arange(0, 8.0, 1 / 200_000.0)
    x = 0.1 * rng.standard_normal(len(t))
    win = (t >= 3.0) & (t <= 6.0)
    x[win] += np.sin(2 * np.pi * 48_000.0 * t[win])
    ok, ratio = wcm_peak_present(t, x, 200_000.0)
    assert ok and ratio > 3.0
    quiet = 0.1 * rng.standard_normal(len(t))
    ok_q, _ = wcm_peak_present(t, quiet, 200_000.0)
    assert not ok_q
    amp = np.where(t >= 7.3, 5.0, 1.0)
    ok_j, r_j = mc_envelope_jump(t, amp * np.sin(2 * np.pi * 5_000.0 * t))
    assert ok_j and r_j > 3.0
