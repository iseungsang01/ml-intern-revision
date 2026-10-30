"""seq v4k (B.16, PREREGISTRATION_B16.md): the physics-structured nowcaster.

Two findings of the derivation audit (THESIS_RESULTS.md sec. 8ba / 8bb) say what the
MSE backbone does wrong and where:

  * In the quiet bulk it carries its noisy anchor (median error 1.03 x persistence) while
    the optimal causal filter of the noisy-OU model sits at 0.875 x (foundations sec. 7.4).
  * Its gain over interpolation lives only in rows where a pedestal event is visible in
    the fast channels at the same instant (P01: the transient term is 90-114 % of the
    headline); in quiet discharges it loses by the variance it adds (P08: -c_b).

So the output is structured as

    y_hat(t) = a(t) + g(t) * r(t)

  a(t)  one-step prediction of a LEARNABLE scalar OU-Kalman cell per target, fed by the
        strictly-causal carried value and the one-step-lagged observation mask -- i.e.
        E[y_t | observations before t] under dy = -(y-m)/tau dt + noise. Training starts
        exactly here (residual head zero-initialised, gate bias -2).
  r(t)  the seq_v2-style LSTM residual (the anchor and its predictive sd are appended to
        the LSTM inputs so the correction knows how uncertain the anchor is).
  g(t)  an event gate in (0, 1). For T_i it reads the fast-channel increment energy
        (the p_obs proxy of P08) and the LSTM state; for V_rot only the slow state.

Routing is seq_v2's, at the encoder: nothing on the V_rot path reads the 15 fast
channels (unit-tested bit-identical under fast-channel perturbation). `v4kng` is the
same model with the gate pinned to 1 (anchor + ungated residual) -- the arm that isolates
what the gate buys.
"""

import math
import os

import torch
import torch.nn as nn

from seq_data import N_FEATURES, N_FAST_CHANNELS

GRID_DT_S = 0.010  # the 10 ms CES grid; blocks never cross a >= 0.5 s gap


def _inv_sigmoid(p):
    p = min(max(float(p), 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


def _inv_softplus(v):
    v = max(float(v), 1e-8)
    return math.log(math.expm1(v))


class KalmanAnchor(nn.Module):
    """Scalar OU + white-noise Kalman cell with a slowly adapting level, per target.

    Normalized (z-score) units. Parameters (all learnable, initialised from the
    sec. 8 OU fits): rho = exp(-dt/tau) via a logit, process stationary variance
    sigma_s^2 and measurement variance sigma_n^2 via softplus, level gain alpha via a
    logit. State per sequence: x (filtered mean), P (filtered variance), m (level).

    forward(carried, obs_prev) with carried (B, L) = last observed value before each
    step (seq_data's carried feature) and obs_prev (B, L) = 1 where that value is a
    NEW observation made exactly at the previous step (the loss mask shifted by one),
    returns anchor a (B, L) and predictive sd s (B, L). Step t: update with the
    observation at t-1 if there was one, then propagate one grid step. The output at t
    therefore uses observations strictly before t.
    """

    def __init__(self, rho0, sigma_s0, sigma_n0, alpha0=0.02, hold_level=False):
        super().__init__()
        # B.16b (PREREGISTRATION_B16.md sec. 10): hold_level keeps the filtered value on
        # unobserved steps instead of relaxing it toward the slow level m (the variance still
        # grows by the OU law). sec. 8bg: the OU relaxation costs the median on stale rows.
        self.hold_level = bool(hold_level)
        self.rho_logit = nn.Parameter(torch.tensor(_inv_sigmoid(rho0)))
        self.q_raw = nn.Parameter(torch.tensor(_inv_softplus(sigma_s0 ** 2)))
        self.r_raw = nn.Parameter(torch.tensor(_inv_softplus(sigma_n0 ** 2)))
        self.alpha_logit = nn.Parameter(torch.tensor(_inv_sigmoid(alpha0)))

    def params(self):
        rho = torch.sigmoid(self.rho_logit)
        q = nn.functional.softplus(self.q_raw) + 1e-6
        r = nn.functional.softplus(self.r_raw) + 1e-6
        alpha = torch.sigmoid(self.alpha_logit)
        return rho, q, r, alpha

    def physical(self, target_std):
        """tau [ms], sigma_n, sigma_s (physical units), alpha -- for the record."""
        rho, q, r, alpha = (float(v) for v in self.params())
        tau_ms = -GRID_DT_S / math.log(rho) * 1e3
        return {"tau_ms": tau_ms, "rho": rho, "sigma_n": math.sqrt(r) * target_std,
                "sigma_s": math.sqrt(q) * target_std, "alpha": alpha}

    def forward(self, carried, obs_prev):
        rho, q, r, alpha = self.params()
        B, L = carried.shape
        x = torch.zeros(B, dtype=carried.dtype, device=carried.device)
        m = torch.zeros(B, dtype=carried.dtype, device=carried.device)
        P = q.expand(B).clone()  # stationary variance: no information yet
        anchors, sds = [], []
        for t in range(L):
            if t > 0:
                o = obs_prev[:, t]
                v = carried[:, t]
                K = P / (P + r)
                x = x + o * K * (v - x)
                P = P - o * K * P
                m = m + o * alpha * (v - m)
            if self.hold_level:
                # relax toward the level only on the step after an observation; hold otherwise
                o_now = obs_prev[:, t] if t > 0 else torch.zeros_like(x)
                x = x + o_now * (m + rho * (x - m) - x)
            else:
                x = m + rho * (x - m)
            P = rho * rho * P + q * (1.0 - rho * rho)
            anchors.append(x)
            sds.append(torch.sqrt(P))
        return torch.stack(anchors, dim=1), torch.stack(sds, dim=1)


class JumpAnchor(nn.Module):
    """B.18 (PREREGISTRATION_B18.md): hold-or-jump anchor, per target, normalized units.

    Prior on the true value: y_t = y_{t-1} + w_t + J_t, w ~ N(0, q_d) a slow drift, and a
    jump J ~ N(0, sigma_J^2) with probability lam per 10 ms step (zero otherwise).
    Measurement: carried = y + N(0, r). sec. 8bi: the increments are bursty (hold, then
    jump), which the OU relaxation of KalmanAnchor misreads in the median and on stale rows.

    Between observations the mean is held (jumps and drift are zero-mean), the variance
    grows by q_d per step and the jump hazard h = 1 - (1 - lam)^k accumulates. At an
    observation the two hypotheses "no jump since the last update" (prior variance P) and
    "one jump" (P + sigma_J^2) are weighted by prior hazard x innovation likelihood, each is
    Kalman-updated, and the pair is collapsed to its first two moments (GPB1). Small
    innovations are averaged with the history (denoising the quiet bulk); large ones are
    followed (gain -> 1). Same interface and causality as KalmanAnchor.
    """

    def __init__(self, sigma_n0, lam0=0.05, sigma_j0=1.0, sigma_d0=0.05, p0=1.0):
        super().__init__()
        self.lam_logit = nn.Parameter(torch.tensor(_inv_sigmoid(lam0)))
        self.j_raw = nn.Parameter(torch.tensor(_inv_softplus(sigma_j0 ** 2)))
        self.d_raw = nn.Parameter(torch.tensor(_inv_softplus(sigma_d0 ** 2)))
        self.r_raw = nn.Parameter(torch.tensor(_inv_softplus(sigma_n0 ** 2)))
        self.p0 = float(p0)  # prior variance before the first observation (per-shot units)

    def params(self):
        lam = torch.sigmoid(self.lam_logit).clamp(1e-6, 1 - 1e-6)
        j = nn.functional.softplus(self.j_raw) + 1e-6
        d = nn.functional.softplus(self.d_raw) + 1e-8
        r = nn.functional.softplus(self.r_raw) + 1e-6
        return lam, j, d, r

    def physical(self, target_std):
        """jump rate [1/s], jump sd / drift sd per step / noise sd (physical units)."""
        lam, j, d, r = (float(v.detach()) for v in self.params())
        return {"lam_per_step": lam, "jump_rate_hz": -math.log(1.0 - lam) / GRID_DT_S,
                "sigma_jump": math.sqrt(j) * target_std, "sigma_drift_per_step": math.sqrt(d) * target_std,
                "sigma_n": math.sqrt(r) * target_std}

    def forward(self, carried, obs_prev):
        lam, j, d, r = self.params()
        B, L = carried.shape
        x = torch.zeros(B, dtype=carried.dtype, device=carried.device)
        P = torch.full((B,), self.p0, dtype=carried.dtype, device=carried.device)
        h = torch.zeros(B, dtype=carried.dtype, device=carried.device)
        anchors, sds = [], []
        for t in range(L):
            if t > 0:
                o = obs_prev[:, t]
                v = carried[:, t]
                e = v - x
                S0, S1 = P + r, P + j + r
                hh = h.clamp(1e-6, 1 - 1e-6)
                logit = (torch.log(hh) - torch.log1p(-hh) - 0.5 * torch.log(S1 / S0)
                         - 0.5 * e * e * (1.0 / S1 - 1.0 / S0))
                w1 = torch.sigmoid(logit)
                K0, K1 = P / S0, (P + j) / S1
                x0, x1 = x + K0 * e, x + K1 * e
                xm = (1 - w1) * x0 + w1 * x1
                # (1 - K) P written as P r / S: exact when P >> r (diffuse prior)
                Pm = ((1 - w1) * (P * r / S0 + (x0 - xm) ** 2)
                      + w1 * ((P + j) * r / S1 + (x1 - xm) ** 2))
                # select, not blend: P + o (Pm - P) cancels catastrophically when P >> Pm
                upd = o > 0.5
                x = torch.where(upd, xm, x)
                P = torch.where(upd, Pm, P)
                h = torch.where(upd, torch.zeros_like(h), h)
            P = P + d
            h = 1 - (1 - h) * (1 - lam)
            anchors.append(x)
            sds.append(torch.sqrt(P + h * j))
        return torch.stack(anchors, dim=1), torch.stack(sds, dim=1)


class SeqCESV4K(nn.Module):
    needs_obs = True  # forward wants the per-step per-target observation mask

    def __init__(self, n_in=N_FEATURES, n_fast=N_FAST_CHANNELS,
                 hidden_ti=160, layers_ti=2, hidden_vt=64, layers_vt=1,
                 head=64, dropout=0.1, gated=True, gate_bias=-2.0, hold_level=False, anchor="ou",
                 init_ti=(0.88, 0.35, 0.30), init_vt=(0.90, 0.60, 0.40)):
        super().__init__()
        # sec. 10 sensitivity rows: gate_bias (+2 = gate initialised open), hold_level (B.16b).
        # CES_SEQ_ANCHOR_ONLY=1 makes forward() return the bare anchors (r = 0): the pure
        # learned OU-Kalman filter of a trained checkpoint, inference-only readout.
        self.gate_bias = float(gate_bias)
        self.anchor_only = os.getenv("CES_SEQ_ANCHOR_ONLY", "0") == "1"
        self.n_fast = int(n_fast)
        self.n_in = int(n_in)
        self.n_slow = self.n_in - self.n_fast
        if self.n_slow <= 0:
            raise ValueError(f"n_fast={n_fast} leaves no non-fast channels of {n_in}")
        self.gated = bool(gated)
        # seq_data layout: [fast | dt | TI(carried, stale, has) | VT(carried, stale, has)]
        self.col_carried = (self.n_fast + 1, self.n_fast + 4)

        # B.18: anchor="jump" swaps the OU cell for the hold-or-jump cell (same noise sd init)
        if anchor == "jump":
            self.kalman_ti = JumpAnchor(init_ti[2])
            self.kalman_vt = JumpAnchor(init_vt[2])
        elif anchor == "ou":
            self.kalman_ti = KalmanAnchor(*init_ti, hold_level=hold_level)
            self.kalman_vt = KalmanAnchor(*init_vt, hold_level=hold_level)
        else:
            raise ValueError(f"unknown anchor {anchor!r}")

        # T_i branch: full state + both anchors and sds (4 extra channels)
        self.lstm_ti = nn.LSTM(self.n_in + 4, hidden_ti, num_layers=layers_ti, batch_first=True,
                               dropout=dropout if layers_ti > 1 else 0.0)
        self.norm_ti = nn.LayerNorm(hidden_ti)
        self.head_ti = nn.Sequential(nn.Linear(hidden_ti, head), nn.GELU(), nn.Linear(head, 1))
        # gate: LSTM state + fast-increment energy (1 channel) + anchor sd
        self.gate_ti = nn.Linear(hidden_ti + 2, 1)

        # V_rot branch: slow tail + its own anchor and sd only (no fast channels anywhere)
        self.lstm_vt = nn.LSTM(self.n_slow + 2, hidden_vt, num_layers=layers_vt, batch_first=True,
                               dropout=dropout if layers_vt > 1 else 0.0)
        self.norm_vt = nn.LayerNorm(hidden_vt)
        self.head_vt = nn.Sequential(nn.Linear(hidden_vt, head), nn.GELU(), nn.Linear(head, 1))
        self.gate_vt = nn.Linear(hidden_vt + 1, 1)

        # start exactly at the Kalman anchor: zero residual, nearly-closed gate
        for h in (self.head_ti, self.head_vt):
            nn.init.zeros_(h[-1].weight)
            nn.init.zeros_(h[-1].bias)
        for g in (self.gate_ti, self.gate_vt):
            nn.init.zeros_(g.weight)
            nn.init.constant_(g.bias, self.gate_bias)

        n_params = sum(p.numel() for p in self.parameters())
        if n_params >= 1_000_000:
            raise ValueError(f"parameter budget exceeded: {n_params:,} >= 1,000,000")
        self.n_params = n_params

    @staticmethod
    def _run(lstm, norm, x, lengths):
        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False)
            out, _ = lstm(packed)
            out, _ = nn.utils.rnn.pad_packed_sequence(
                out, batch_first=True, total_length=x.shape[1])
        else:
            out, _ = lstm(x)
        return norm(out)

    def physics(self, target_std):
        return {"CES_TI": self.kalman_ti.physical(float(target_std[0])),
                "CES_VT": self.kalman_vt.physical(float(target_std[1]))}

    def forward(self, x, lengths=None, obs=None):
        """x (B, L, n_in), obs (B, L, 2) per-step per-target observation mask -> (B, L, 2)."""
        if obs is None:
            raise ValueError("v4k needs the observation mask (needs_obs)")
        x = x[..., :self.n_in]
        B, L, _ = x.shape
        # observation at t-1 (shifted mask) -> the carried value at t is a fresh measurement
        obs_prev = torch.zeros_like(obs)
        obs_prev[:, 1:, :] = obs[:, :-1, :]

        a_ti, s_ti = self.kalman_ti(x[..., self.col_carried[0]], obs_prev[..., 0])
        a_vt, s_vt = self.kalman_vt(x[..., self.col_carried[1]], obs_prev[..., 1])
        if self.anchor_only:
            self.last_gate = (torch.zeros_like(a_ti), torch.zeros_like(a_vt))
            return torch.stack([a_ti, a_vt], dim=-1)

        # fast-channel increment energy (strictly causal: uses rows t and t-1)
        fast = x[..., :self.n_fast]
        dfast = torch.zeros_like(fast)
        dfast[:, 1:] = fast[:, 1:] - fast[:, :-1]
        u = torch.log1p(dfast.pow(2).mean(dim=-1))  # (B, L)

        xin_ti = torch.cat([x, a_ti.unsqueeze(-1), s_ti.unsqueeze(-1),
                            a_vt.unsqueeze(-1), s_vt.unsqueeze(-1)], dim=-1)
        h_ti = self._run(self.lstm_ti, self.norm_ti, xin_ti, lengths)
        r_ti = self.head_ti(h_ti).squeeze(-1)
        if self.gated:
            g_ti = torch.sigmoid(self.gate_ti(torch.cat(
                [h_ti, u.unsqueeze(-1), s_ti.unsqueeze(-1)], dim=-1))).squeeze(-1)
        else:
            g_ti = torch.ones_like(r_ti)
        y_ti = a_ti + g_ti * r_ti

        xin_vt = torch.cat([x[..., self.n_fast:], a_vt.unsqueeze(-1), s_vt.unsqueeze(-1)], dim=-1)
        h_vt = self._run(self.lstm_vt, self.norm_vt, xin_vt, lengths)
        r_vt = self.head_vt(h_vt).squeeze(-1)
        if self.gated:
            g_vt = torch.sigmoid(self.gate_vt(torch.cat([h_vt, s_vt.unsqueeze(-1)], dim=-1))).squeeze(-1)
        else:
            g_vt = torch.ones_like(r_vt)
        y_vt = a_vt + g_vt * r_vt

        self.last_gate = (g_ti.detach(), g_vt.detach())
        return torch.stack([y_ti, y_vt], dim=-1)
