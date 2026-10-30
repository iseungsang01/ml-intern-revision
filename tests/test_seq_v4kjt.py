"""B.19 structural tests for the Student-t measurement model of the hold-or-jump anchor (arm v4kjt).

No data needed. They pin what PREREGISTRATION_B19.md relies on: the scale starts at the
sec. 8bl bulk scatter chosen by population; as nu grows the cell becomes the Gaussian B.18
cell; an outlying observation is followed less than under the Gaussian model; v4kj
checkpoints keep their state_dict; routing and causality are those of v4kj.
"""
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "ces_prediction", "experiments", "seq"))

from model_seq_v4k import BULK_SIGMA_N, JumpAnchor  # noqa: E402
from seq_data import N_FAST_CHANNELS, N_FEATURES  # noqa: E402
from seq_models import SEQ_MODELS  # noqa: E402


def _model(monkeypatch, cut_ev):
    monkeypatch.setenv("CES_TI_SPIKE_CUT_EV", cut_ev)
    return SEQ_MODELS["v4kjt"]()


def test_registry_scale_by_population(monkeypatch):
    for cut_ev, key in (("3000", "CES_TI_cut"), ("0", "CES_TI_incl")):
        m = _model(monkeypatch, cut_ev)
        assert m.kalman_ti.student and m.kalman_vt.student and not m.gated
        r_ti = float(m.kalman_ti.params()[3]) ** 0.5
        r_vt = float(m.kalman_vt.params()[3]) ** 0.5
        assert abs(r_ti - BULK_SIGMA_N[key]) < 1e-4 and abs(r_vt - BULK_SIGMA_N["CES_VT"]) < 1e-4
        assert abs(float(m.kalman_ti.nu()) - 3.0) < 1e-5


def test_v4kj_state_dict_unchanged():
    keys = set(SEQ_MODELS["v4kj"]().state_dict())
    assert not any("nu_raw" in k for k in keys)


def test_large_nu_is_the_gaussian_cell():
    g, t = JumpAnchor(0.3), JumpAnchor(0.3, nu0=500.0)
    torch.manual_seed(0)
    c = torch.randn(3, 50)
    o = (torch.rand(3, 50) < 0.7).float()
    with torch.no_grad():
        (a1, s1), (a2, s2) = g(c, o), t(c, o)
    assert float((a1 - a2).abs().max()) < 0.02 and float((s1 - s2).abs().max()) < 0.01


def test_outlier_followed_less_than_gaussian():
    g, t = JumpAnchor(0.3, lam0=0.01), JumpAnchor(0.3, lam0=0.01, nu0=3.0)
    L = 40
    c = torch.zeros(1, L)
    c[0, 30] = 1.0  # one ~3 sigma observation after a flat history
    o = torch.ones(1, L)
    with torch.no_grad():
        assert float(t(c, o)[0][0, 31]) < 0.5 * float(g(c, o)[0][0, 31])


def test_routing_and_causality(monkeypatch):
    m = _model(monkeypatch, "3000")
    torch.manual_seed(3)
    for p in m.parameters():
        p.data.add_(0.05 * torch.randn_like(p))
    m.eval()
    g = torch.Generator().manual_seed(3)
    x = torch.randn(2, 40, N_FEATURES, generator=g)
    obs = (torch.rand(2, 40, 2, generator=g) < 0.7).float()
    x2 = x.clone()
    x2[..., :N_FAST_CHANNELS] += 3.0 * torch.randn_like(x2[..., :N_FAST_CHANNELS])
    x3, obs3 = x.clone(), obs.clone()
    x3[:, 18:] = torch.randn_like(x3[:, 18:])
    obs3[:, 18:] = 1.0 - obs3[:, 18:]
    with torch.no_grad():
        y1, y2, y3 = m(x, None, obs), m(x2, None, obs), m(x3, None, obs3)
    assert torch.equal(y1[..., 1], y2[..., 1])
    assert torch.allclose(y1[:, :18], y3[:, :18], atol=1e-6)


def test_physical_readout_has_nu(monkeypatch):
    m = _model(monkeypatch, "3000")
    phys = m.physics(target_std=(391.0, 73.6))
    assert abs(phys["CES_TI"]["nu"] - 3.0) < 1e-4
    assert abs(phys["CES_TI"]["sigma_n"] - BULK_SIGMA_N["CES_TI_cut"] * 391.0) < 0.1
