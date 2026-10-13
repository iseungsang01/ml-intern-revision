"""B.16 structural tests for the physics-structured nowcaster (model_seq_v4k.py).

No data needed. They pin the three properties the pre-registration relies on:
  1. routing -- nothing on the V_rot path reads the 15 fast channels (bit-identical output
     under fast-channel perturbation);
  2. causality -- the prediction at step t does not change when rows > t change (inputs or
     observation masks);
  3. the Kalman cell is the closed-form noisy-OU filter of docs/theory_foundations.md
     sec. 7.4: with every step observed its predictive variance converges to the Riccati
     fixed point, and with no observations the anchor decays by rho per step toward the
     level; at initialisation the whole model equals its Kalman anchor.
"""
import math
import os
import sys

import numpy as np
import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "ces_prediction", "experiments", "seq"))

from model_seq_v4k import KalmanAnchor, SeqCESV4K  # noqa: E402
from seq_data import N_FAST_CHANNELS, N_FEATURES  # noqa: E402
from seq_models import SEQ_MODELS  # noqa: E402


def _batch(B=2, L=40, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(B, L, N_FEATURES, generator=g)
    obs = (torch.rand(B, L, 2, generator=g) < 0.7).float()
    return x, obs


def test_registry_and_shapes():
    torch.manual_seed(0)
    for name in ("v4k", "v4kng"):
        m = SEQ_MODELS[name]()
        assert getattr(m, "needs_obs", False)
        x, obs = _batch()
        y = m(x, None, obs)
        assert y.shape == (2, 40, 2)
        assert torch.isfinite(y).all()
        assert m.n_params < 1_000_000


def test_vrot_path_never_sees_fast_channels():
    torch.manual_seed(1)
    m = SEQ_MODELS["v4k"]()
    # make the residual and gate non-trivial so a leak would show
    for p in m.parameters():
        p.data.add_(0.05 * torch.randn_like(p))
    m.eval()
    x, obs = _batch(seed=1)
    x2 = x.clone()
    x2[..., :N_FAST_CHANNELS] += torch.randn_like(x2[..., :N_FAST_CHANNELS]) * 3.0
    with torch.no_grad():
        y1, y2 = m(x, None, obs), m(x2, None, obs)
    assert torch.equal(y1[..., 1], y2[..., 1]), "V_rot output changed under fast-channel perturbation"
    assert not torch.allclose(y1[..., 0], y2[..., 0]), "T_i output should respond to the fast channels"


def test_causality():
    torch.manual_seed(2)
    m = SEQ_MODELS["v4k"]()
    for p in m.parameters():
        p.data.add_(0.05 * torch.randn_like(p))
    m.eval()
    x, obs = _batch(seed=2)
    t = 17
    x2, obs2 = x.clone(), obs.clone()
    x2[:, t + 1:] = torch.randn_like(x2[:, t + 1:])
    obs2[:, t + 1:] = (torch.rand_like(obs2[:, t + 1:]) < 0.5).float()
    with torch.no_grad():
        y1, y2 = m(x, None, obs), m(x2, None, obs2)
    assert torch.allclose(y1[:, :t + 1], y2[:, :t + 1], atol=1e-6), "future rows leaked into the past"


def test_model_starts_at_the_kalman_anchor():
    torch.manual_seed(3)
    m = SEQ_MODELS["v4kng"]()  # gate pinned to 1, residual head zero-initialised
    m.eval()
    x, obs = _batch(seed=3)
    obs_prev = torch.zeros_like(obs)
    obs_prev[:, 1:] = obs[:, :-1]
    with torch.no_grad():
        y = m(x, None, obs)
        a_ti, _ = m.kalman_ti(x[..., m.col_carried[0]], obs_prev[..., 0])
        a_vt, _ = m.kalman_vt(x[..., m.col_carried[1]], obs_prev[..., 1])
    assert torch.allclose(y[..., 0], a_ti, atol=1e-6)
    assert torch.allclose(y[..., 1], a_vt, atol=1e-6)


def test_kalman_cell_matches_closed_form():
    rho0, ss, sn = 0.884, 0.18, 0.10
    cell = KalmanAnchor(rho0, ss, sn, alpha0=1e-6)  # level frozen at 0 for the closed form
    rho, q, r, _ = (float(v) for v in cell.params())
    # all steps observed: predictive variance -> Riccati fixed point (foundations sec. 7.4)
    P = q
    for _ in range(5000):
        P_minus = rho * rho * P + q * (1 - rho * rho)
        K = P_minus / (P_minus + r)
        P = (1 - K) * P_minus
    p_pred_ss = rho * rho * P + q * (1 - rho * rho)
    L = 200
    carried = torch.zeros(1, L)
    obs_prev = torch.ones(1, L)
    with torch.no_grad():
        _, s = cell(carried, obs_prev)
    assert abs(float(s[0, -1]) ** 2 - p_pred_ss) < 1e-6
    # no observations: anchor decays by rho per step from the last filtered value
    obs_prev2 = torch.zeros(1, L)
    obs_prev2[0, 1] = 1.0
    carried2 = torch.zeros(1, L)
    carried2[0, 1:] = 1.0  # the one observation (value 1 at step 0) is carried forward
    with torch.no_grad():
        a, _ = cell(carried2, obs_prev2)
    k1 = q / (q + r)  # first update from the stationary prior
    expected = [k1 * rho * rho ** k for k in range(L - 1)]
    got = a[0, 1:].tolist()
    assert np.allclose(got, expected, atol=1e-6)


def test_physical_readout_recovers_init():
    m = SEQ_MODELS["v4k"]()
    phys = m.physics(target_std=(385.0, 60.0))
    assert abs(phys["CES_TI"]["rho"] - 0.88) < 1e-4
    assert abs(phys["CES_TI"]["tau_ms"] - (-10.0 / math.log(0.88))) < 1e-3
    assert abs(phys["CES_TI"]["sigma_s"] - 0.35 * 385.0) < 1e-2
    assert abs(phys["CES_VT"]["sigma_n"] - 0.40 * 60.0) < 1e-2
