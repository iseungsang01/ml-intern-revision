"""B.26: the pre-check filter carries the same merged mean as the B.23 scan, and the mixture median
is right. Hand-built series: arithmetic only, no data."""
import os
import sys

import numpy as np
from scipy.stats import norm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "ces_prediction", "experiments", "seq"))
sys.path.insert(0, os.path.join(ROOT, "ces_prediction", "experiments", "b16_physics"))
sys.path.insert(0, os.path.join(ROOT, "tests"))

import precheck_b26 as P  # noqa: E402
from fit_vrot_jump_b23 import scan_loglik  # noqa: E402
from test_b23_vrot_jump_fit import _series  # noqa: E402


def test_mix_median():
    w, m0, v0, m1, v1 = np.array([0.3, 0.0, 1.0]), np.zeros(3), np.ones(3), np.array([2.0, 5.0, -1.0]), np.full(3, 4.0)
    med = P.mix_median(w, m0, v0, m1, v1)
    cdf = (1 - w) * norm.cdf(med, m0, 1.0) + w * norm.cdf(med, m1, 2.0)
    np.testing.assert_allclose(cdf, 0.5, atol=1e-9)
    np.testing.assert_allclose(med[1], 0.0, atol=1e-9)
    np.testing.assert_allclose(med[2], -1.0, atol=1e-9)


def test_filter_matches_b23_scan():
    carried, obs_prev = _series()
    lam, j, r = 0.03, 0.5 ** 2, 0.06 ** 2
    xs = []
    scan_loglik(carried, obs_prev, np.array([lam]), np.array([j]), r=r, anchors=xs)
    w, em, ed = P.filter_e1(carried, obs_prev, lam, j, r)
    # the next-observation error of the merged mean equals |carried - anchor before the update|
    anchors = np.stack(xs, axis=-1)[0]
    seen = np.zeros(carried.shape[0], bool)
    ref = []
    for t in range(1, carried.shape[1]):
        upd = obs_prev[:, t] > 0.5
        ref.append(np.abs(carried[:, t] - anchors[:, t - 1])[upd & seen])
        seen |= upd
    np.testing.assert_allclose(em, np.concatenate(ref), rtol=1e-12, atol=1e-12)
    assert len(w) == len(em) == len(ed) and np.all((w >= 0) & (w <= 1))
