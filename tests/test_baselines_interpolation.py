import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ces_prediction"))

import baselines_interpolation as B  # noqa: E402

TIME, TGT = 0, 1  # column layout for these synthetic arrays


def _arr(times, vals):
    return np.array(list(zip(times, vals)), dtype=np.float32)


def test_linear_recovers_linear_function():
    # target = 2*time + 1, exact at an interior point excluded from neighbors.
    times = [0.0, 0.01, 0.02, 0.03, 0.04]
    vals = [2 * t + 1 for t in times]
    a = _arr(times, vals)
    pred = B.predict("linear", a, TIME, TGT, row_index=2)
    assert pred == pytest.approx(1.04, abs=1e-5)


def test_pchip_no_overshoot_on_monotone_step():
    # Monotone step: PCHIP must stay within neighbor bounds (no spline ringing).
    times = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05]
    vals = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
    a = _arr(times, vals)
    pred = B.predict("pchip", a, TIME, TGT, row_index=2)
    assert -1e-9 <= pred <= 1.0 + 1e-9  # bounded by [min,max] of neighbors


def test_no_leakage_excludes_target_row():
    times = [0.0, 0.01, 0.02, 0.03, 0.04]
    base_vals = [1.0, 1.0, 1.0, 1.0, 1.0]
    a_sentinel = _arr(times, [1.0, 1.0, 999.0, 1.0, 1.0])  # poison the target row
    a_clean = _arr(times, base_vals)

    nt, nv, tt = B.build_neighbor_set(a_sentinel, TIME, TGT, row_index=2)
    assert 999.0 not in nv  # target row never enters the neighbor set
    assert len(nv) == 4

    # Prediction must be identical whether or not the target row is poisoned.
    for method in ("linear", "pchip", "persistence", "ar_local"):
        p_sent = B.predict(method, a_sentinel, TIME, TGT, row_index=2)
        p_clean = B.predict(method, a_clean, TIME, TGT, row_index=2)
        assert p_sent == pytest.approx(p_clean, abs=1e-9)


def test_gap_refusal_excludes_across_half_second_gap():
    # Future point sits across a >=0.5s gap -> not a neighbor -> linear falls back
    # to persistence (last past), NOT influenced by the post-gap value.
    times = [0.0, 0.01, 0.02, 0.70, 0.71]
    vals = [0.0, 0.0, 0.0, 5.0, 5.0]
    a = _arr(times, vals)
    nt, nv, tt = B.build_neighbor_set(a, TIME, TGT, row_index=2)
    assert nt.max() < 0.5  # post-gap neighbors excluded
    pred = B.predict("linear", a, TIME, TGT, row_index=2)
    assert pred == pytest.approx(0.0, abs=1e-9)  # persistence fallback, not 5.0


def test_persistence_is_last_past_observation():
    times = [0.0, 0.01, 0.03]
    vals = [10.0, 20.0, 0.0]  # row 2 is the target (excluded)
    a = _arr(times, vals)
    pred = B.predict("persistence", a, TIME, TGT, row_index=2)
    assert pred == pytest.approx(20.0)


def test_ar_local_extrapolates_recent_slope():
    times = [0.0, 0.01, 0.02]
    vals = [10.0, 12.0, 0.0]  # slope 200/s over last two past points
    a = _arr(times, vals)
    pred = B.predict("ar_local", a, TIME, TGT, row_index=2)
    assert pred == pytest.approx(14.0, abs=1e-4)  # 12 + 200*0.01


def test_neighbor_set_skips_nan_targets():
    times = [0.0, 0.01, 0.02, 0.03]
    vals = [1.0, np.nan, 1.0, 9.0]  # row 1 unobserved, row 3 is target (excluded)
    a = _arr(times, vals)
    nt, nv, tt = B.build_neighbor_set(a, TIME, TGT, row_index=3)
    assert len(nv) == 2  # rows 0 and 2 only (row 1 NaN, row 3 excluded)
    assert not np.isnan(nv).any()
