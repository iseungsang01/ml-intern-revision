"""B.17 structural tests for the distribution-head seq model (model_seq_v2d.py).

No data needed. They pin what the pre-registration relies on:
  1. the mean path equals the decoded posterior mean (the point harness scores the right thing);
  2. the NLL is the closed-form Gaussian / 2-mixture log-density (checked against
     torch.distributions and the mixture against explicit logsumexp);
  3. routing -- the V_rot distribution is bit-identical under fast-channel perturbation;
  4. causality -- parameters at step t do not change when rows > t change;
  5. at init the predictive sd is one normalized unit (maximally uninformative start);
  6. the CRPS closed forms used by score_b17.py match a Monte Carlo estimate.
"""
import math
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "ces_prediction", "experiments", "seq"))
sys.path.insert(0, os.path.join(ROOT, "ces_prediction", "experiments", "b17_dist"))

from seq_data import N_FAST_CHANNELS, N_FEATURES  # noqa: E402
from seq_models import SEQ_MODELS  # noqa: E402
from score_b17 import crps_mixture  # noqa: E402


def _batch(B=2, L=30, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(B, L, N_FEATURES, generator=g)


def _perturbed(name, seed):
    torch.manual_seed(seed)
    m = SEQ_MODELS[name]()
    for p in m.parameters():
        p.data.add_(0.05 * torch.randn_like(p))
    m.eval()
    return m


def test_registry_shapes_and_mean_path():
    for name, P in (("v2g", 2), ("v2m2", 5)):
        m = _perturbed(name, 0)
        x = _batch()
        with torch.no_grad():
            params = m(x, None, return_dist=True)
            mean = m(x)
        assert params.shape == (2, 30, 2, P)
        assert mean.shape == (2, 30, 2)
        assert torch.allclose(mean, m.mean(params))
        assert m.n_params < 1_000_000
        mu, sd = m.moments(params)
        assert torch.allclose(mu, mean) and (sd > 0).all()


def test_nll_matches_closed_forms():
    torch.manual_seed(5)
    y = torch.randn(2, 30, 2)
    m = _perturbed("v2g", 5)
    with torch.no_grad():
        params = m(_batch(seed=5), None, return_dist=True)
        pi, mu, sigma = m.decode(params)
        ref = -torch.distributions.Normal(mu[..., 0], sigma[..., 0]).log_prob(y)
        assert torch.allclose(m.nll(params, y), ref, atol=1e-5)
    m2 = _perturbed("v2m2", 6)
    with torch.no_grad():
        params = m2(_batch(seed=6), None, return_dist=True)
        pi, mu, sigma = m2.decode(params)
        comp = torch.distributions.Normal(mu, sigma).log_prob(y.unsqueeze(-1))
        ref = -torch.logsumexp(torch.log(pi) + comp, dim=-1)
        assert torch.allclose(m2.nll(params, y), ref, atol=1e-5)
        assert torch.allclose(pi.sum(-1), torch.ones_like(pi[..., 0]))


def test_vrot_distribution_never_sees_fast_channels():
    for name in ("v2g", "v2m2"):
        m = _perturbed(name, 1)
        x = _batch(seed=1)
        x2 = x.clone()
        x2[..., :N_FAST_CHANNELS] += torch.randn_like(x2[..., :N_FAST_CHANNELS]) * 3.0
        with torch.no_grad():
            p1, p2 = m(x, None, return_dist=True), m(x2, None, return_dist=True)
        assert torch.equal(p1[:, :, 1], p2[:, :, 1]), "V_rot parameters changed under fast perturbation"
        assert not torch.allclose(p1[:, :, 0], p2[:, :, 0]), "T_i parameters should respond"


def test_causality_of_parameters():
    m = _perturbed("v2m2", 2)
    x = _batch(seed=2)
    t = 12
    x2 = x.clone()
    x2[:, t + 1:] = torch.randn_like(x2[:, t + 1:])
    with torch.no_grad():
        p1, p2 = m(x, None, return_dist=True), m(x2, None, return_dist=True)
    assert torch.allclose(p1[:, :t + 1], p2[:, :t + 1], atol=1e-6)


def test_init_sd_is_one_unit():
    for name in ("v2g", "v2m2"):
        torch.manual_seed(0)
        m = SEQ_MODELS[name]()
        with torch.no_grad():
            # zero the hidden->output weights so the biases alone set the parameters
            for head in (m.head_ti, m.head_vt):
                head[-1].weight.zero_()
            params = m(_batch(), None, return_dist=True)
            _, mu, sigma = m.decode(params)
        assert torch.allclose(sigma, torch.ones_like(sigma), atol=2e-3)
        if name == "v2m2":
            pi, _, _ = m.decode(params)
            assert torch.allclose(pi, torch.full_like(pi, 0.5))


def test_crps_closed_form_matches_monte_carlo():
    rng = np.random.default_rng(0)
    pi = np.array([[0.3, 0.7], [1.0, 0.0], [0.5, 0.5]])
    mu = np.array([[-1.0, 2.0], [0.4, 0.0], [0.0, 0.0]])
    sigma = np.array([[0.5, 1.5], [0.8, 1.0], [1.0, 1.0]])
    y = np.array([0.3, -0.2, 1.1])
    closed = crps_mixture(y, pi, mu, sigma)
    n = 400_000
    for i in range(3):
        comp = rng.choice(2, size=n, p=pi[i])
        s = rng.normal(mu[i][comp], sigma[i][comp])
        s2 = rng.normal(mu[i][rng.choice(2, size=n, p=pi[i])], sigma[i][rng.choice(2, size=n, p=pi[i])])
        mc = np.abs(s - y[i]).mean() - 0.5 * np.abs(s - s2).mean()
        assert abs(closed[i] - mc) < 0.01, (i, closed[i], mc)
    # the Gaussian case reduces to the textbook formula
    z = (y[1] - mu[1, 0]) / sigma[1, 0]
    from scipy.stats import norm
    g = sigma[1, 0] * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / math.sqrt(math.pi))
    assert abs(closed[1] - g) < 1e-9


def test_t_head_nll_and_moments_match_scipy():
    from scipy.stats import t as st
    m = _perturbed("v2t", 9)
    y = torch.randn(2, 30, 2)
    with torch.no_grad():
        params = m(_batch(seed=9), None, return_dist=True)
        _, mu, sigma = m.decode(params)
        nu = m.nu(params)
        nll = m.nll(params, y).numpy()
        mean, sd = m.moments(params)
    z = ((y.unsqueeze(-1) - mu) / sigma).numpy()[..., 0]
    ref = -(st.logpdf(z, nu.numpy()[..., 0]) - np.log(sigma.numpy()[..., 0]))
    assert np.allclose(nll, ref, atol=1e-5)
    assert (nu > 2.0).all()
    assert np.allclose(sd.numpy(), (sigma[..., 0] * torch.sqrt(nu[..., 0] / (nu[..., 0] - 2))).numpy(), atol=1e-6)
    assert torch.allclose(mean, mu[..., 0])


def test_t_crps_closed_form_matches_monte_carlo():
    from score_b17b import crps_t
    rng = np.random.default_rng(1)
    mu, sigma, nu, y = np.array([0.5, -1.0]), np.array([0.8, 2.0]), np.array([3.0, 6.5]), np.array([1.2, -0.3])
    closed = crps_t(y, mu, sigma, nu)
    n = 400_000
    for i in range(2):
        s = mu[i] + sigma[i] * rng.standard_t(nu[i], size=n)
        s2 = mu[i] + sigma[i] * rng.standard_t(nu[i], size=n)
        mc = np.abs(s - y[i]).mean() - 0.5 * np.abs(s - s2).mean()
        assert abs(closed[i] - mc) < 0.02, (i, closed[i], mc)


def test_v4k_sensitivity_arms():
    """sec. 10 rows: gate initialised open, level-holding anchor, anchor-only readout."""
    import os as _os
    from model_seq_v4k import KalmanAnchor
    torch.manual_seed(0)
    g = torch.Generator().manual_seed(0)
    x = torch.randn(2, 40, N_FEATURES, generator=g)
    obs = (torch.rand(2, 40, 2, generator=g) < 0.7).float()
    m_open = SEQ_MODELS["v4ko"]()
    assert float(m_open.gate_ti.bias) == 2.0 and float(m_open.gate_vt.bias) == 2.0
    # hold_level: with no observations after the first, the anchor stays put instead of relaxing
    cell = KalmanAnchor(0.88, 0.35, 0.30, hold_level=True)
    L = 50
    carried = torch.ones(1, L)
    obs_prev = torch.zeros(1, L)
    obs_prev[0, 1] = 1.0
    with torch.no_grad():
        a, s = cell(carried, obs_prev)
    assert torch.allclose(a[0, 2:], a[0, 2].expand(L - 2), atol=1e-6), "level not held on unobserved steps"
    assert (s[0, 3:] > s[0, 2]).all() or torch.allclose(s[0, 3:], s[0, 2]), "variance must not shrink without observations"
    m_hold = SEQ_MODELS["v4kngh"]()
    assert m_hold.kalman_ti.hold_level and not m_hold.gated
    # anchor-only readout returns exactly the anchors
    _os.environ["CES_SEQ_ANCHOR_ONLY"] = "1"
    try:
        m = SEQ_MODELS["v4kng"]()
        for p in m.parameters():
            p.data.add_(0.05 * torch.randn_like(p))
        m.eval()
        obs_prev2 = torch.zeros_like(obs)
        obs_prev2[:, 1:] = obs[:, :-1]
        with torch.no_grad():
            y = m(x, None, obs)
            a_ti, _ = m.kalman_ti(x[..., m.col_carried[0]], obs_prev2[..., 0])
        assert torch.allclose(y[..., 0], a_ti, atol=1e-6)
    finally:
        _os.environ.pop("CES_SEQ_ANCHOR_ONLY", None)
