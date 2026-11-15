"""Unit tests of the MAST gate arms and statistic (no data needed).

The simulated series here test the code only; none of them is a result.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ces_prediction" / "experiments" / "mast"))
import run_gate as rg  # noqa: E402

STEP = rg.STEP_MS


def _series(seed=0, L=41, C=5):
    rng = np.random.default_rng(seed)
    t = 100.0 + STEP * np.arange(L) + rng.uniform(-0.05, 0.05, L)
    G = rng.random((L, C)) > 0.12
    Y0 = np.where(G, 500.0 + 80.0 * rng.standard_normal((L, C)), 0.0)
    S2 = np.where(G, rng.uniform(5.0, 40.0, (L, C)) ** 2, 0.0)
    return t, G, Y0, S2, rg.hidden_positions(L)


PARAMS = {"tau_ms": 15.0, "q_level": 30.0, "q_slope": 2.0, "p0": 1e9 * 20.0 ** 2,
          "gp_causal": (40.0, 35.0), "gp_offline": (20.0, 50.0)}


def _weights(arm, t, G, S2, hid):
    if arm == "persistence":
        return rg.arm_persistence(t, G, S2, hid)
    if arm == "extrapolation":
        return rg.arm_extrapolation(t, G, S2, hid)
    if arm == "offline_interp":
        return rg.arm_offline_interp(t, G, S2, hid)
    if arm == "running_mean":
        return rg.arm_running_mean(t, G, S2, hid, PARAMS["tau_ms"])
    if arm == "slope_filter":
        return rg.arm_slope_filter(t, G, S2, hid, PARAMS["q_level"], PARAMS["q_slope"])
    if arm == "causal_gp":
        return rg.arm_gp(t, G, S2, hid, *PARAMS["gp_causal"], True)
    return rg.arm_gp(t, G, S2, hid, *PARAMS["gp_offline"], False)


# ---- independent value-based reference implementations (scalar loops); the reference filter
# approximates the diffuse start with a large prior variance p0

def _ref_kalman(t, g, y, s2, j, ql, qs, p0):
    x = np.zeros(2)
    P = np.diag([p0, p0])
    for i in range(j + 1):
        if i > 0:
            dt = (t[i] - t[i - 1]) / STEP
            F = np.array([[1.0, dt], [0.0, 1.0]])
            x = F @ x
            P = F @ P @ F.T + np.diag([ql * dt, qs * dt])
        if i % 2 == 0 and i < j and g[i]:
            K = P[:, 0] / (P[0, 0] + s2[i])
            x = x + K * (y[i] - x[0])
            P = P - np.outer(K, P[0, :])
    return x[0]


def _ref_gp(t, g, y, s2, j, ell, amp, causal):
    L = len(t)
    cand = [j - (2 * k + 1) for k in range(8)] if causal else [j + o for o in (-7, -5, -3, -1, 1, 3, 5, 7)]
    ii = [i for i in cand if 0 <= i < L and g[i]]
    if not ii:
        return 0.0
    ti = t[ii]

    def k(a, b):
        r = np.sqrt(3.0) * np.abs(a[:, None] - b[None, :]) / ell
        return amp ** 2 * (1.0 + r) * np.exp(-r) + rg.CONST_FACTOR * amp ** 2

    K = k(ti, ti) + np.diag(s2[ii])
    ks = k(np.array([t[j]]), ti)[0]
    return float(ks @ np.linalg.solve(K, y[ii]))


def _ref_running_mean(t, g, y, j, tau):
    ii = [i for i in range(0, j, 2) if g[i]]
    w = np.exp(-(t[j] - t[ii]) / tau)
    return float((w * y[ii]).sum() / w.sum())


def test_weights_reproduce_predictions():
    t, G, Y0, S2, hid = _series()
    rows = G[hid] & G[hid - 1] & G[hid - 3] & G[hid + 1]
    assert rows.sum() > 20
    for arm in rg.ARMS:
        W, idx = _weights(arm, t, G, S2, hid)
        pred, varw = rg.apply_weights(W, idx, Y0, S2)
        for h, c in zip(*np.nonzero(rows)):
            j = hid[h]
            g, y, s2 = G[:, c], Y0[:, c], S2[:, c]
            if arm == "persistence":
                ref = y[j - 1]
            elif arm == "extrapolation":
                ref = y[j - 1] + (y[j - 1] - y[j - 3]) * (t[j] - t[j - 1]) / (t[j - 1] - t[j - 3])
            elif arm == "offline_interp":
                ref = 0.5 * (y[j - 1] + y[j + 1])
            elif arm == "running_mean":
                ref = _ref_running_mean(t, g, y, j, PARAMS["tau_ms"])
            elif arm == "slope_filter":
                ref = _ref_kalman(t, g, y, s2, j, PARAMS["q_level"], PARAMS["q_slope"], PARAMS["p0"])
            else:
                ref = _ref_gp(t, g, y, s2, j, *PARAMS["gp_causal" if arm == "causal_gp" else "gp_offline"],
                              arm == "causal_gp")
            # the reference GP solves the ill-conditioned matrix (c = 1e6 amp^2) directly: loose side
            tol = 1e-3 if arm.endswith("gp") else (1e-5 if arm == "slope_filter" else 1e-9)
            assert abs(pred[h, c] - ref) < tol * max(1.0, abs(ref)), (arm, h, c, pred[h, c], ref)
            # the explicit weight vector over the block frames reproduces prediction and variance
            w = np.zeros(len(t))
            for k in range(idx.shape[1]):
                if idx[h, k] >= 0:
                    w[idx[h, k]] += W[h, c, k]
            assert abs(w @ y - pred[h, c]) < 1e-8 * max(1.0, abs(ref))
            assert abs((w ** 2) @ s2 - varw[h, c]) < 1e-8 * max(1.0, varw[h, c])
            assert np.all(w[~g] == 0.0)
            assert np.all(w[1::2] == 0.0)          # hidden (odd) frames are never used
            assert abs(w.sum() - 1.0) < 1e-4, (arm, w.sum())


def test_slope_filter_value_mode_matches_weight_mode():
    t, G, Y0, S2, hid = _series(seed=3)
    W, idx = rg.arm_slope_filter(t, G, S2, hid, 30.0, 2.0)
    pred_w, _ = rg.apply_weights(W, idx, Y0, S2)
    ql = np.array([[30.0, 5.0], [30.0, 1.0]])
    qs = np.array([[2.0, 2.0], [0.1, 0.1]])
    pred_v = rg.arm_slope_filter(t, G, S2, hid, ql, qs, Y0=Y0)
    assert pred_v.shape == (2, 2) + pred_w.shape
    rows = G[hid] & G[hid - 1] & G[hid - 3] & G[hid + 1]
    assert np.allclose(pred_v[0, 0][rows], pred_w[rows], rtol=1e-6, atol=1e-5)
    assert not np.allclose(pred_v[1, 1][rows], pred_w[rows], rtol=1e-3)


def test_causal_arms_use_no_sample_at_or_after_hidden_time():
    t, G, Y0, S2, hid = _series(seed=1)
    for arm in rg.CAUSAL_ARMS:
        W, idx = _weights(arm, t, G, S2, hid)
        late = idx >= hid[:, None]
        assert np.all(W[np.broadcast_to(late[:, None, :], W.shape)] == 0.0), arm
        # perturbing every value at or after the hidden frame leaves the estimate unchanged
        pred0, _ = rg.apply_weights(W, idx, Y0, S2)
        for h in (0, len(hid) // 2, len(hid) - 1):
            Y1 = Y0.copy()
            Y1[hid[h]:] += 1e4
            pred1, _ = rg.apply_weights(W, idx, Y1, S2)
            assert np.array_equal(pred0[h], pred1[h]), arm
    for arm in ("offline_interp", "offline_gp"):
        W, idx = _weights(arm, t, G, S2, hid)
        late = idx > hid[:, None]
        assert np.abs(W[np.broadcast_to(late[:, None, :], W.shape)]).max() > 0.0   # offline arms look ahead


def _median_t_a(rho, n_shots=60, L=60, C=30, seed=0):
    """MA(1) noise with lag-1 correlation rho on a flat signal: n_t = (e_t + b e_(t-1)) / sqrt(1 + b^2)."""
    rng = np.random.default_rng(seed)
    b = 0.0 if rho == 0 else (1 - np.sqrt(1 - 4 * rho ** 2)) / (2 * rho)
    out = []
    for _ in range(n_shots):
        e = rng.standard_normal((L + 1, C))
        noise = (e[1:] + b * e[:-1]) / np.sqrt(1 + b * b)
        out.append(rg.t_a_statistic(noise, np.ones((L, C), bool))[0])
    return float(np.median(out))


def test_t_a_statistic_on_simulated_noise():
    white = _median_t_a(0.0)
    assert abs(white) < 0.08, white
    corr = _median_t_a(0.3)
    assert corr < -0.25, corr           # about -(4/3) rho_1 = -0.4 in the noise-dominated limit
    # a random-walk signal (second-difference variance growing with spacing) does not lower T_a
    rng = np.random.default_rng(5)
    vals = []
    for _ in range(40):
        walk = np.cumsum(0.6 * rng.standard_normal((60, 30)), axis=0)
        vals.append(rg.t_a_statistic(walk + rng.standard_normal((60, 30)), np.ones((60, 30), bool))[0])
    assert np.median(vals) > -0.08


def test_regime_number_white_noise_on_flat_signal():
    rng = np.random.default_rng(2)
    Y = 300.0 + rng.standard_normal((400, 20))
    r = rg.regime_number(Y, np.ones(Y.shape, bool))
    assert abs(r + 0.5) < 0.05, r


def test_gate_shot_list_has_300_distinct_shots():
    shots = rg.read_gate_shots()
    assert len(shots) == 300 and len(set(shots)) == 300
