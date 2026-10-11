"""B.14 robust training losses (PREREGISTRATION_B14.md sec. 2): shape of the influence function.

No data needed. Checks the three properties the pre-registration relies on:
  * inside the noise scale every loss is the MSE's quadratic (so the bulk is fit exactly as before),
  * Huber's gradient is bounded by delta (bounded influence),
  * the Student-t loss's gradient goes to zero for huge residuals (redescending),
and that the mse kind is the identity r**2 (the frozen backbone's code path).
"""

import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "ces_prediction" / "experiments" / "seq"), str(ROOT / "ces_prediction")]

from train_seq import robust_loss_config, robust_residual_loss  # noqa: E402

STATS = {"target": {"mean": np.array([600.0, 75.0]), "std": np.array([390.0, 73.0])}}


def _cfg(kind, env):
    saved = {k: os.environ.get(k) for k in ("CES_SEQ_LOSS", "CES_SEQ_LOSS_SIGMA_TI_EV",
                                            "CES_SEQ_LOSS_SIGMA_VT_KMS", "CES_SEQ_LOSS_NU")}
    try:
        for k in saved:
            os.environ.pop(k, None)
        os.environ["CES_SEQ_LOSS"] = kind
        os.environ.update(env)
        return robust_loss_config(STATS)
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def test_mse_is_identity_and_needs_no_scale():
    cfg = _cfg("mse", {})
    r = torch.randn(5, 7, 2)
    assert torch.equal(robust_residual_loss(r, cfg), r ** 2)


def test_scale_is_physical_over_run_std():
    cfg = _cfg("huber", {"CES_SEQ_LOSS_SIGMA_TI_EV": "129.9", "CES_SEQ_LOSS_SIGMA_VT_KMS": "16.4"})
    np.testing.assert_allclose(cfg["delta_norm"], [129.9 / 390.0, 16.4 / 73.0])


def test_missing_scale_aborts():
    import pytest
    with pytest.raises(SystemExit):
        _cfg("tnll", {"CES_SEQ_LOSS_SIGMA_TI_EV": "129.9"})


def test_quadratic_inside_scale_bounded_or_redescending_outside():
    env = {"CES_SEQ_LOSS_SIGMA_TI_EV": "129.9", "CES_SEQ_LOSS_SIGMA_VT_KMS": "16.4"}
    huber, tnll = _cfg("huber", env), _cfg("tnll", env)
    d = torch.tensor(huber["delta_norm"])
    small = (torch.rand(1000, 2) * 2 - 1) * d * 0.999
    # Inside delta Huber IS the squared residual, and the Student-t loss matches it to
    # second order (s2 * log1p(r^2 / s2) = r^2 - r^4 / (2 s2) + ...), so the bulk is
    # fit exactly as the MSE control fits it.
    assert torch.allclose(robust_residual_loss(small, huber), small ** 2)
    s2 = 3.0 * d ** 2
    assert torch.allclose(robust_residual_loss(small, tnll), small ** 2, rtol=0.0,
                          atol=0.5 * float((small ** 4 / s2).max()) + 1e-7)
    big = torch.tensor([[10.0, 10.0], [100.0, 100.0], [1000.0, 1000.0]], requires_grad=True)
    g_huber = torch.autograd.grad(robust_residual_loss(big, huber).sum(), big)[0]
    assert torch.allclose(g_huber, (2 * d).expand_as(g_huber))  # bounded influence = MSE's at delta
    big2 = big.detach().clone().requires_grad_(True)
    g_t = torch.autograd.grad(robust_residual_loss(big2, tnll).sum(), big2)[0]
    assert (g_t[2] < g_t[1]).all() and (g_t[1] < g_t[0]).all()  # redescending
    assert (g_t[2] < 0.01).all()
    mse_grad = 2 * big.detach()
    assert (g_huber < mse_grad).all() and (g_t < mse_grad).all()
