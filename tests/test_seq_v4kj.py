"""B.18 structural tests for the hold-or-jump anchor (model_seq_v4k.py::JumpAnchor, arm v4kj).

No data needed. They pin what PREREGISTRATION_B18.md relies on: routing and causality are
those of v4k; with the jump hazard off the cell is the static-level Kalman filter (it averages
repeated observations); a small innovation is shrunk toward the history and a large one is
followed; without observations the mean is held and the predictive sd grows.
"""
import math
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "ces_prediction", "experiments", "seq"))

from model_seq_v4k import JumpAnchor  # noqa: E402
from seq_data import N_FAST_CHANNELS, N_FEATURES  # noqa: E402
from seq_models import SEQ_MODELS  # noqa: E402


def _batch(B=2, L=40, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(B, L, N_FEATURES, generator=g)
    obs = (torch.rand(B, L, 2, generator=g) < 0.7).float()
    return x, obs


def _perturbed(seed):
    torch.manual_seed(seed)
    m = SEQ_MODELS["v4kj"]()
    for p in m.parameters():
        p.data.add_(0.05 * torch.randn_like(p))
    return m.eval()


def test_registry_shapes_and_start():
    m = SEQ_MODELS["v4kj"]().eval()
    assert isinstance(m.kalman_ti, JumpAnchor) and not m.gated and m.n_params < 1_000_000
    x, obs = _batch()
    obs_prev = torch.zeros_like(obs)
    obs_prev[:, 1:] = obs[:, :-1]
    with torch.no_grad():
        y = m(x, None, obs)
        a_ti, _ = m.kalman_ti(x[..., m.col_carried[0]], obs_prev[..., 0])
        a_vt, _ = m.kalman_vt(x[..., m.col_carried[1]], obs_prev[..., 1])
    assert y.shape == (2, 40, 2) and torch.isfinite(y).all()
    assert torch.allclose(y[..., 0], a_ti, atol=1e-6) and torch.allclose(y[..., 1], a_vt, atol=1e-6)


def test_vrot_path_never_sees_fast_channels():
    m = _perturbed(1)
    x, obs = _batch(seed=1)
    x2 = x.clone()
    x2[..., :N_FAST_CHANNELS] += torch.randn_like(x2[..., :N_FAST_CHANNELS]) * 3.0
    with torch.no_grad():
        y1, y2 = m(x, None, obs), m(x2, None, obs)
    assert torch.equal(y1[..., 1], y2[..., 1])
    assert not torch.allclose(y1[..., 0], y2[..., 0])


def test_causality():
    m = _perturbed(2)
    x, obs = _batch(seed=2)
    t = 17
    x2, obs2 = x.clone(), obs.clone()
    x2[:, t + 1:] = torch.randn_like(x2[:, t + 1:])
    obs2[:, t + 1:] = (torch.rand_like(obs2[:, t + 1:]) < 0.5).float()
    with torch.no_grad():
        y1, y2 = m(x, None, obs), m(x2, None, obs2)
    assert torch.allclose(y1[:, :t + 1], y2[:, :t + 1], atol=1e-6)


def test_no_hazard_is_the_static_level_filter():
    # lam -> 0, drift -> 0, diffuse prior: the anchor is the running mean of the observations
    cell = JumpAnchor(sigma_n0=0.3, lam0=1e-6, sigma_d0=1e-5, p0=1e6)
    rng = np.random.default_rng(0)
    vals = rng.normal(0.7, 0.3, size=30)
    L = len(vals) + 1
    carried = torch.zeros(1, L)
    obs_prev = torch.zeros(1, L)
    carried[0, 1:] = torch.tensor(vals, dtype=torch.float32)
    obs_prev[0, 1:] = 1.0
    with torch.no_grad():
        a, _ = cell(carried, obs_prev)
    running = np.cumsum(vals) / np.arange(1, len(vals) + 1)
    assert np.allclose(a[0, 1:].numpy(), running, atol=2e-4)


def test_small_innovation_shrunk_large_followed_and_hold():
    cell = JumpAnchor(sigma_n0=0.3, lam0=0.05, sigma_j0=1.0, sigma_d0=0.02)
    L = 40
    obs_prev = torch.ones(1, L)
    with torch.no_grad():
        for jump, check in ((0.15, lambda g: g < 0.5), (4.0, lambda g: g > 0.9)):
            carried = torch.zeros(1, L)
            carried[0, -1] = jump  # 38 observations of 0, then one innovation
            a, _ = cell(carried, obs_prev)
            gain = float(a[0, -1]) / jump
            assert check(gain), (jump, gain)
        # no observations after step 1: mean held, predictive sd grows
        carried = torch.ones(1, L)
        obs_prev2 = torch.zeros(1, L)
        obs_prev2[0, 1] = 1.0
        a, s = cell(carried, obs_prev2)
    assert torch.allclose(a[0, 2:], a[0, 1].expand(L - 2), atol=1e-6)
    assert (s[0, 2:] > s[0, 1:-1]).all()


def test_physical_readout():
    m = SEQ_MODELS["v4kj"]()
    phys = m.physics(target_std=(385.0, 60.0))
    assert abs(phys["CES_TI"]["lam_per_step"] - 0.05) < 1e-4
    assert abs(phys["CES_TI"]["jump_rate_hz"] - (-math.log(0.95) / 0.01)) < 1e-2
    assert abs(phys["CES_TI"]["sigma_n"] - 0.30 * 385.0) < 1e-2
    assert abs(phys["CES_VT"]["sigma_jump"] - 60.0) < 1e-2
