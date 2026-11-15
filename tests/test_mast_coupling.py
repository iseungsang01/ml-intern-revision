"""Unit tests of the lag machinery of the MAST coupling study (no data needed).

The simulated series only check the sign convention and the pooling of the statistic; they are
never a result.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ces_prediction" / "experiments" / "mast"))
import run_coupling as rc  # noqa: E402


def _sums(y_inc, dx_grid, n0):
    rows = len(y_inc)
    ysum = y_inc[:, None]
    y2 = (y_inc ** 2)[:, None]
    cnt = np.ones((rows, 1))
    noise = np.zeros((rows, 1))
    xc, xm = rc.centred(dx_grid, np.isfinite(dx_grid))
    idx = rc.lag_index(rows, 1, n0)
    return rc.lag_sums(ysum, y2, cnt, noise, xc, xm, idx)


def test_positive_lag_means_driver_first():
    rng = np.random.default_rng(0)
    L = 400
    n0 = -rc.PAD
    grid = rng.normal(size=L + 2 * rc.PAD)          # driver increments on the frame grid
    grid[0] = np.nan
    lead = 3                                        # target row j responds to the driver at j - 3
    j = 1 + np.arange(L - 1)
    y = grid[(j - lead) - n0] + 0.1 * rng.normal(size=L - 1)
    y = y - y.mean()
    r = rc.corr(_sums(y, grid, n0))[0]
    assert rc.LAGS[int(np.nanargmax(r))] == lead
    assert r[rc.MAXLAG + lead] > 0.9
    assert abs(r[rc.MAXLAG - lead]) < 0.2


def test_frame_means_window():
    t = np.arange(0.0, 100.0, 0.5)                  # ms
    v = (t >= 50.0).astype(float)
    x = rc.frame_means(t, v, t_a=40.0, n_lo=-2, n_hi=4)
    # windows centred at 30, 35, 40, 45, 50, 55, 60 ms; the step at 50 ms splits the 50 ms window
    assert np.allclose(x[:4], 0.0) and np.allclose(x[5:], 1.0)
    assert 0.4 < x[4] < 0.6
    assert np.isnan(rc.frame_means(t, v, t_a=500.0, n_lo=0, n_hi=1)).all()


def test_signal_fraction_and_bins():
    s = np.zeros((1, 1, rc.NSTAT))
    s[..., 2], s[..., 3] = 4.0, 1.0
    assert np.isclose(rc.sigfrac(s)[0, 0], 0.75)
    cb = rc.rho_bin(np.array([0.1, 0.35, 0.6, 0.8, 0.95, 1.2, -0.1, np.nan]))
    assert cb.tolist() == [0, 1, 2, 3, 4, -1, -1, -1]
    B = rc.membership(cb)
    assert B[:, -1].tolist() == [1, 1, 1, 1, 1, 0, 0, 0]
