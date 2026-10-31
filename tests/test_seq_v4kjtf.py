"""B.21 structural tests for the variance-weighted residual (arm v4kjtf). No data needed."""
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "ces_prediction", "experiments", "seq"))

from model_seq_v4k import FUSION_SIGMA_R2  # noqa: E402
from seq_data import N_FAST_CHANNELS, N_FEATURES  # noqa: E402
from seq_models import SEQ_MODELS  # noqa: E402


def test_init_by_population_and_v4kjt_state_dict(monkeypatch):
    for cut_ev, key in (("3000", "cut"), ("0", "incl")):
        monkeypatch.setenv("CES_TI_SPIKE_CUT_EV", cut_ev)
        m = SEQ_MODELS["v4kjtf"]()
        assert torch.allclose(m.sigma_r2(), torch.full((2,), FUSION_SIGMA_R2[key]), atol=1e-5)
    assert not any("fuse_raw" in k for k in SEQ_MODELS["v4kjt"]().state_dict())


def test_gain_is_variance_ratio(monkeypatch):
    monkeypatch.setenv("CES_TI_SPIKE_CUT_EV", "3000")
    m = SEQ_MODELS["v4kjtf"]()
    s = torch.tensor([0.0, 0.1, 0.52, 3.0])
    k = m.fusion_gain(s, 0)
    assert torch.all(k[1:] > k[:-1]) and float(k[0]) == 0.0 and float(k[-1]) < 1.0
    assert abs(float(m.fusion_gain(torch.tensor(0.27 ** 0.5), 0)) - 0.5) < 1e-4


def test_residual_scaled_and_causal(monkeypatch):
    monkeypatch.setenv("CES_TI_SPIKE_CUT_EV", "3000")
    torch.manual_seed(5)
    f, a = SEQ_MODELS["v4kjtf"](), SEQ_MODELS["v4kjt"]()
    a.load_state_dict({k: v for k, v in f.state_dict().items() if k != "fuse_raw"})
    for mod in (f.head_ti, f.head_vt, a.head_ti, a.head_vt):
        torch.nn.init.constant_(mod[-1].bias, 0.3)
    f.eval(); a.eval()
    g = torch.Generator().manual_seed(1)
    x = torch.randn(2, 30, N_FEATURES, generator=g)
    obs = (torch.rand(2, 30, 2, generator=g) < 0.7).float()
    with torch.no_grad():
        yf, ya = f(x, None, obs), a(x, None, obs)
        kf = f.last_gate
        anchor = ya - 0.3
        assert torch.allclose(yf, anchor + 0.3 * torch.stack(kf, dim=-1), atol=1e-5)
        x2 = x.clone()
        x2[..., :N_FAST_CHANNELS] += 2.0
        assert torch.equal(f(x2, None, obs)[..., 1], yf[..., 1])  # V_rot never sees fast channels
        x3, o3 = x.clone(), obs.clone()
        x3[:, 15:] = torch.randn_like(x3[:, 15:]); o3[:, 15:] = 1 - o3[:, 15:]
        assert torch.allclose(f(x3, None, o3)[:, :15], yf[:, :15], atol=1e-6)
