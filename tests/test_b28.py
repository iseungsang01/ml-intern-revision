"""B.28 (PREREGISTRATION_B28.md): the two-term equation model keeps the contract it was
registered with -- 20 T_i parameters (+1 cross), causal, V_rot path blind to the fast channels,
and at initialisation exactly the EWMA of the past observations."""
import math
import os
import sys

import torch

HERE = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(HERE, "../ces_prediction")))
sys.path.append(os.path.abspath(os.path.join(HERE, "../ces_prediction/experiments/seq")))

from seq_models import SEQ_MODELS  # noqa: E402

DT = math.log1p(0.01)


def _sequence(L=60, seed=0):
    """A synthetic feature block in seq_data's layout, used only to test wiring (no result)."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(1, L, 22, generator=g)
    x[..., 15] = DT
    obs = torch.zeros(L, dtype=torch.bool)
    obs[torch.arange(2, L, 3)] = True          # an observation every third row
    y = torch.randn(L, generator=g)
    carried, stale, has = torch.zeros(L), torch.zeros(L), torch.zeros(L)
    last_v, last_t = 0.0, None
    for i in range(L):
        if last_t is not None:
            carried[i] = last_v
            stale[i] = math.log1p(0.01 * (i - last_t))
            has[i] = 1.0
        if obs[i]:
            last_v, last_t = float(y[i]), i
    x[0, :, 16], x[0, :, 17], x[0, :, 18] = carried, stale, has
    x[0, :, 19:22] = 0.0
    return x, y, obs


def test_b28_parameter_count_and_shape():
    m = SEQ_MODELS["b28"]()
    mx = SEQ_MODELS["b28x"]()
    assert m.n_params_ti == 20 and mx.n_params_ti == 21
    x, _, _ = _sequence()
    out = m(x)
    assert out.shape == (1, 60, 2)
    out.sum().backward()
    assert m.a.grad is not None


def test_b28_ti_starts_at_the_ewma_of_past_observations():
    m = SEQ_MODELS["b28"]()
    x, y, obs = _sequence()
    with torch.no_grad():
        ti = m(x)[0, :, 0]
    k0, kinf, sk = (float(v) for v in m.gains())
    e, first, last_t = None, True, None
    for t in range(60):
        if e is not None:
            assert abs(float(ti[t]) - e) < 1e-5, t
        if obs[t]:
            if first:
                e, first = float(y[t]), False
            else:
                gap = 0.01 * (t - last_t)
                k = kinf + (k0 - kinf) * math.exp(-gap / sk)
                e = e + k * (float(y[t]) - e)
            last_t = t


def test_b28_is_causal_and_vt_is_blind_to_fast_channels():
    m = SEQ_MODELS["b28x"]()
    m.eval()
    x, _, _ = _sequence()
    with torch.no_grad():
        base = m(x)
        x2 = x.clone()
        x2[0, 40:, :15] += 3.0                    # perturb the future fast channels
        later = m(x2)
        assert torch.allclose(base[0, :40], later[0, :40]), "a future input moved a past output"
        assert torch.allclose(base[0, :, 1], later[0, :, 1]), "V_rot read the fast channels"
        x3 = x.clone()
        x3[0, :, :15] += 1.0                      # perturb all fast channels
        assert torch.allclose(base[0, :, 1], m(x3)[0, :, 1])
