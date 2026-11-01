"""B.23: the TRAIN likelihood scan reproduces the model's anchor recursion. No data needed."""
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "ces_prediction", "experiments", "seq"))
sys.path.insert(0, os.path.join(ROOT, "ces_prediction", "experiments", "b16_physics"))

from fit_vrot_jump_b23 import NU, P0, SIGMA_D, SIGMA_N, log_t, scan_loglik  # noqa: E402
from model_seq_v4k import _jump_scan  # noqa: E402


def _series(seed=3, B=4, L=60):
    g = np.random.default_rng(seed)
    y = np.cumsum(g.standard_t(3, size=(B, L)) * 0.05 + (g.random((B, L)) < 0.03) * g.normal(0, 1, (B, L)), axis=1)
    obs = (g.random((B, L)) < 0.6).astype(float)
    carried, obs_prev = np.zeros((B, L)), np.zeros((B, L))
    for b in range(B):
        last = None
        for t in range(L):
            if last is not None:
                carried[b, t] = last
            if t > 0:
                obs_prev[b, t] = obs[b, t - 1]
            if obs[b, t]:
                last = y[b, t]
    return carried, obs_prev


def test_anchor_matches_model_scan():
    carried, obs_prev = _series()
    lam, j = 0.03, 0.5 ** 2
    xs = []
    scan_loglik(carried, obs_prev, np.array([lam]), np.array([j]), anchors=xs)
    mine = np.stack(xs, axis=-1)[0]
    a, _ = _jump_scan(torch.tensor(carried), torch.tensor(obs_prev), torch.tensor(lam, dtype=torch.float64),
                      torch.tensor(j, dtype=torch.float64), torch.tensor(SIGMA_D ** 2, dtype=torch.float64),
                      torch.tensor(SIGMA_N ** 2, dtype=torch.float64), P0, torch.tensor(NU, dtype=torch.float64), True)
    assert np.allclose(mine, a.numpy(), atol=1e-9)


def test_t_density_normalised_and_vectorised_grid():
    e = np.linspace(-60, 60, 600001)
    assert abs(np.trapezoid(np.exp(log_t(e * e, 0.3, 3.0)), e) - 1) < 1e-3
    carried, obs_prev = _series(5)
    lam, j = np.array([0.01, 0.05, 0.2]), np.array([0.1, 1.0, 2.0])
    joint, n = scan_loglik(carried, obs_prev, lam, j)
    for g in range(3):
        one, n1 = scan_loglik(carried, obs_prev, lam[g:g + 1], j[g:g + 1])
        assert n1 == n and abs(one[0] - joint[g]) < 1e-9


def test_patch_sets_only_vrot_jump(tmp_path, monkeypatch):
    monkeypatch.setenv("CES_TI_SPIKE_CUT_EV", "3000")
    from anchor_readout_b16 import patch_vt_jump
    from seq_models import SEQ_MODELS
    m = SEQ_MODELS["v4kjt"]()
    torch.save(m.state_dict(), tmp_path / "a.pth")
    patch_vt_jump(tmp_path / "a.pth", tmp_path / "b.pth", {"lam_per_step": 0.02, "sigma_j": 0.5})
    m2 = SEQ_MODELS["v4kjt"]()
    m2.load_state_dict(torch.load(tmp_path / "b.pth"))
    lam, j, d, r = (float(v) for v in m2.kalman_vt.params())
    assert abs(lam - 0.02) < 1e-6 and abs(j - 0.25) < 1e-6
    a, b = m.state_dict(), m2.state_dict()
    assert all(torch.equal(a[k], b[k]) for k in a if k not in ("kalman_vt.lam_logit", "kalman_vt.j_raw"))


def test_b24_patch_sets_vrot_noise_too(tmp_path, monkeypatch):
    monkeypatch.setenv("CES_TI_SPIKE_CUT_EV", "3000")
    from anchor_readout_b16 import patch_vt_jump
    from seq_models import SEQ_MODELS
    m = SEQ_MODELS["v4kjt"]()
    torch.save(m.state_dict(), tmp_path / "a.pth")
    patch_vt_jump(tmp_path / "a.pth", tmp_path / "b.pth", {"lam_per_step": 0.08, "sigma_j": 0.35},
                  {"sigma_n": 0.05, "nu": 4.0})
    m2 = SEQ_MODELS["v4kjt"]()
    m2.load_state_dict(torch.load(tmp_path / "b.pth"))
    lam, j, d, r = (float(v) for v in m2.kalman_vt.params())
    assert abs(lam - 0.08) < 1e-6 and abs(j - 0.35 ** 2) < 1e-6 and abs(r - 0.05 ** 2) < 1e-6
    assert abs(float(m2.kalman_vt.nu()) - 4.0) < 1e-5
    keys = ("kalman_vt.lam_logit", "kalman_vt.j_raw", "kalman_vt.r_raw", "kalman_vt.nu_raw")
    a, b = m.state_dict(), m2.state_dict()
    assert all(torch.equal(a[k], b[k]) for k in a if k not in keys)


def test_b24_noise_grid_reproduces_scalar_scan():
    from fit_vrot_jump_b23 import scan_loglik
    carried, obs_prev = _series()
    lam, j = np.full(2, 0.08), np.full(2, 0.35 ** 2)
    r, nu = np.array([0.06 ** 2, 0.05 ** 2])[:, None], np.array([3.0, 4.0])[:, None]
    joint, n = scan_loglik(carried, obs_prev, lam, j, r=r, nu=nu)
    for g in range(2):
        one, _ = scan_loglik(carried, obs_prev, lam[g:g + 1], j[g:g + 1], r=float(r[g, 0]), nu=float(nu[g, 0]))
        assert abs(one[0] - joint[g]) < 1e-9
