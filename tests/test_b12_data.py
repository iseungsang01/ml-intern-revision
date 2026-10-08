"""B.12 offline features must never see a hidden value (real data; skipped without it)."""

import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "ces_prediction" / "experiments" / "b12")]
DATA = Path(os.getenv("CES_DATA_DIR", ROOT / "data"))

pytestmark = pytest.mark.skipif(not any(DATA.glob("s*.csv")), reason="real shot CSVs not available")


@pytest.fixture(scope="module")
def grid():
    from b12_data import load_population
    g, _ = load_population(DATA, 3000.0)
    return dict(list(g.items())[:20])


def test_hidden_values_never_enter_features(grid):
    from b12_data import file_features, draw_hidden, missing_run_lengths, fit_stats
    from seq_data import load_grid_files  # noqa: F401  (path set by b12_data)
    names = list(grid)
    dims = {"bes": 9, "ecei": 4, "mc": 2}
    stats = fit_stats(grid, dims, names)
    runs = missing_run_lengths(grid, names)
    rng = np.random.default_rng(0)
    for n in names:
        arr = grid[n]
        hid = draw_hidden(arr, runs, rng)
        assert not (hid & np.isnan(arr[:, 1:3])).any(), "only observed rows may be hidden"
        poked = arr.copy()
        poked[:, 1:3] = np.where(hid, poked[:, 1:3] + 1e4, poked[:, 1:3])
        a, b = file_features(arr, hid, stats), file_features(poked, hid, stats)
        for ba, bb in zip(a, b):
            np.testing.assert_array_equal(ba["x"], bb["x"])


def test_hide_fraction_close_to_target(grid):
    from b12_data import draw_hidden, missing_run_lengths, HIDE_FRACTION
    runs = missing_run_lengths(grid, list(grid))
    rng = np.random.default_rng(1)
    obs = hid = 0
    for arr in grid.values():
        h = draw_hidden(arr, runs, rng)
        obs += (~np.isnan(arr[:, 1])).sum()
        hid += h[:, 0].sum()
    # per-file goals are rounded, so the pooled share can sit a hair under the target
    assert HIDE_FRACTION - 0.01 <= hid / obs < HIDE_FRACTION + 0.1
