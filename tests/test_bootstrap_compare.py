"""CPU-only unit tests for bootstrap_compare's internal functions.

These exercise the shot-clustered paired bootstrap (PR4 gate) on SYNTHETIC numpy
data only -- no GPU, no real CSVs, no comparison_errors_*.npz on disk. They pin:

  * _per_shot_sums: correct per-shot grouping (sums + counts), regardless of
    shot-id ordering / non-contiguity.
  * _bootstrap: determinism (same seed -> same CI), the PASS gate when the model
    strictly beats the baseline on every sample across many shots, the n.s.
    (no-difference) case, and that n_shots is reported correctly.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# Match the import convention in the rest of tests/: put ces_prediction on path.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ces_prediction"))

import bootstrap_compare as BC  # noqa: E402


# --------------------------------------------------------------------------- #
# _per_shot_sums
# --------------------------------------------------------------------------- #
def test_per_shot_sums_groups_by_shot():
    # Shots are non-contiguous and unsorted in arrival order; grouping must be by
    # shot id, and np.unique returns shots in ascending order.
    shot = np.array([7, 3, 7, 3, 7], dtype=np.int64)
    values = np.array([1.0, 10.0, 2.0, 20.0, 3.0], dtype=np.float64)
    shots, sums, counts = BC._per_shot_sums(shot, values)

    assert list(shots) == [3, 7]  # ascending unique
    # shot 3: 10 + 20 = 30 (count 2); shot 7: 1 + 2 + 3 = 6 (count 3)
    np.testing.assert_allclose(sums, [30.0, 6.0])
    np.testing.assert_allclose(counts, [2.0, 3.0])
    assert sums.dtype == np.float64
    assert counts.dtype == np.float64


def test_per_shot_sums_single_shot():
    shot = np.array([42, 42, 42], dtype=np.int64)
    values = np.array([0.5, 1.5, 2.0], dtype=np.float64)
    shots, sums, counts = BC._per_shot_sums(shot, values)
    assert list(shots) == [42]
    np.testing.assert_allclose(sums, [4.0])
    np.testing.assert_allclose(counts, [3.0])


def test_per_shot_sums_total_is_preserved():
    rng = np.random.default_rng(1)
    shot = rng.integers(0, 5, size=200)
    values = rng.random(200)
    _, sums, counts = BC._per_shot_sums(shot, values)
    # Grouping must conserve the grand total and the grand count.
    assert sums.sum() == pytest.approx(values.sum())
    assert counts.sum() == pytest.approx(float(values.size))


# --------------------------------------------------------------------------- #
# _bootstrap
# --------------------------------------------------------------------------- #
def _many_shots(n_shots=40, per_shot=25, seed=7):
    """Synthetic (shot, se_model, se_base) where se_model < se_base everywhere."""
    rng = np.random.default_rng(seed)
    shot = np.repeat(np.arange(n_shots), per_shot).astype(np.int64)
    # Strictly positive baseline error; model error strictly smaller per sample.
    se_base = rng.uniform(1.0, 2.0, size=shot.size)
    se_model = se_base * rng.uniform(0.3, 0.6, size=shot.size)  # always < se_base
    return shot, se_model, se_base


def test_bootstrap_is_deterministic_for_same_seed():
    shot, se_model, se_base = _many_shots()
    r1 = BC._bootstrap(shot, se_model, se_base, np.random.default_rng(0))
    r2 = BC._bootstrap(shot, se_model, se_base, np.random.default_rng(0))
    assert r1 == r2  # identical dicts (CI, points, pass, n_shots)


def test_bootstrap_different_seed_changes_ci():
    shot, se_model, se_base = _many_shots()
    r1 = BC._bootstrap(shot, se_model, se_base, np.random.default_rng(0))
    r2 = BC._bootstrap(shot, se_model, se_base, np.random.default_rng(999))
    # Point estimates are seed-independent (computed on the full data)...
    assert r1["skill_point"] == pytest.approx(r2["skill_point"])
    # ...but the resampled CI bounds should differ for a different rng stream.
    assert r1["skill_ci95"] != r2["skill_ci95"]


def test_bootstrap_model_strictly_better_passes():
    shot, se_model, se_base = _many_shots(n_shots=40, per_shot=25)
    res = BC._bootstrap(shot, se_model, se_base, np.random.default_rng(0))

    assert res["skill_point"] > 0.0
    assert res["skill_ci95"][0] > 0.0           # 95% lower bound strictly > 0
    assert res["skill_ci95"][0] <= res["skill_ci95"][1]
    assert res["pass"] is True
    # Paired diff = mean(SE_model - SE_base) must be negative (model better).
    assert res["paired_diff_point"] < 0.0
    assert res["paired_diff_ci95"][1] < 0.0     # diff CI upper bound < 0
    assert res["n_shots"] == 40


def test_bootstrap_equal_errors_is_zero_skill_and_fails():
    rng = np.random.default_rng(3)
    shot = np.repeat(np.arange(30), 20).astype(np.int64)
    se = rng.uniform(0.5, 1.5, size=shot.size)
    # se_model == se_base exactly -> no skill, degenerate (zero-width) CI at 0.
    res = BC._bootstrap(shot, se.copy(), se.copy(), np.random.default_rng(0))

    assert res["skill_point"] == pytest.approx(0.0, abs=1e-12)
    assert res["paired_diff_point"] == pytest.approx(0.0, abs=1e-12)
    # Every resample gives skill exactly 0 -> CI collapses to [0, 0] -> lo not > 0.
    assert res["skill_ci95"][0] == pytest.approx(0.0, abs=1e-12)
    assert res["pass"] is False
    assert res["n_shots"] == 30


def test_bootstrap_reports_correct_n_shots():
    for n_shots in (5, 17, 50):
        shot, se_model, se_base = _many_shots(n_shots=n_shots, per_shot=10)
        res = BC._bootstrap(shot, se_model, se_base, np.random.default_rng(0))
        assert res["n_shots"] == n_shots


def test_bootstrap_noisy_no_real_difference_is_not_significant():
    # Same error distribution for both arms (no systematic edge) -> CI should
    # straddle 0 -> not a PASS. Guards against a gate that fires on noise.
    rng = np.random.default_rng(11)
    shot = np.repeat(np.arange(30), 20).astype(np.int64)
    se_base = rng.uniform(0.5, 1.5, size=shot.size)
    se_model = rng.uniform(0.5, 1.5, size=shot.size)  # independent, same dist
    res = BC._bootstrap(shot, se_model, se_base, np.random.default_rng(0))
    assert res["skill_ci95"][0] <= 0.0  # lower bound not above 0
    assert res["pass"] is False
