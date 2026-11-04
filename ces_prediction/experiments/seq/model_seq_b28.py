"""seq b28 (B.28, PREREGISTRATION_B28.md): the two-term equation read off the frozen seq_v2.

The weight/response readout of the frozen seq_v2 T_i branch (docs/b28_weight_physics_report.md)
found that what the LSTM computes at a T_i gap is, to 77-95 % of its persistence-relative skill,

    y_hat(t) = m(t) + sum_c a_c * kappa * tanh(dx_c(t) / kappa) + b0

  m(t)   an exponentially weighted average of the PAST OBSERVED T_i values, updated when a
         new observation arrives with a gain K(s) that depends on the gap s since the
         observation before it (K ~ 0.35 at 10 ms, ~ 0.5 after a 15-55 ms gap): the
         Kalman-like denoising of the noisy anchor (P06, foundations sec. 7.4).
  dx_c   the 10 ms increment of fast channel c (per-shot z units): the fast diagnostics
         enter as increment detectors (impulse sum ~ 0; BES increment index 0.95, ECEI 0.63)
         with a saturating, not threshold-like, response (event-row gradient smaller).
  kappa  one saturation scale shared by the 15 channels; a_c the per-channel gain.

This module IS that equation, made trainable, so that B.28 stage 2 can ask how much of the
backbone's skill the equation carries when trained end to end under the identical protocol.
The T_i branch has 20 parameters (K(s): 3, a_c: 15, kappa: 1, b0: 1); `cross=True` adds one
product term c * mean(dECEI) * mean(dBES) (the one nonlinear event term tried, B.28 stage 2's
single extra variable). The V_rot branch is seq_v2's, verbatim (slow tail only), so the ONE
controlled variable against the backbone is the T_i computation.

Causality is seq_data's: carried / stale / has at step t use observations strictly before t,
and m(t) is updated from the observation made at t-1 (visible at t through carried). The EWMA
scan runs on the CPU as TorchScript like JumpAnchor (kernel-launch bound on the GPU).
"""

import math
from typing import List

import torch
import torch.nn as nn

from seq_data import N_FEATURES, N_FAST_CHANNELS

NEW_OBS_STALE = math.log1p(0.015)   # stale < log1p(15 ms) <=> the previous grid row was observed
N_ECEI = 4
N_BES = 9


def _inv_sigmoid(p):
    p = min(max(float(p), 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


def _inv_softplus(v):
    v = max(float(v), 1e-8)
    return math.log(math.expm1(v))


@torch.jit.script
def _ewma_scan(carried: torch.Tensor, new_obs: torch.Tensor, gain: torch.Tensor) -> torch.Tensor:
    """m_t = m_{t-1} + new_obs_t * k_t * (carried_t - m_{t-1}); the first observation of a
    sequence is taken whole (k = 1). carried / new_obs / gain are (B, L). Returns (B, L)."""
    B, L = carried.shape[0], carried.shape[1]
    m = torch.zeros([B], dtype=carried.dtype, device=carried.device)
    seen = torch.zeros([B], dtype=carried.dtype, device=carried.device)
    out: List[torch.Tensor] = []
    for t in range(L):
        o = new_obs[:, t]
        k = seen * gain[:, t] + (1.0 - seen)
        m = m + o * k * (carried[:, t] - m)
        seen = torch.maximum(seen, o)
        out.append(m)
    return torch.stack(out, dim=1)


class SeqCESB28(nn.Module):
    def __init__(self, n_in=N_FEATURES, n_fast=N_FAST_CHANNELS, hidden_vt=64, layers_vt=1,
                 head=64, dropout=0.1, cross=False,
                 k0=0.35, k_inf=0.60, s_k=0.03, kappa0=1.0):
        super().__init__()
        self.n_in = int(n_in)
        self.n_fast = int(n_fast)
        self.n_slow = self.n_in - self.n_fast
        if self.n_slow <= 0:
            raise ValueError(f"n_fast={n_fast} leaves no non-fast channels of {n_in}")
        self.cross = bool(cross)
        # seq_data layout: [fast | dt | TI(carried, stale, has) | VT(carried, stale, has)]
        self.col_ti_carried = self.n_fast + 1
        self.col_ti_stale = self.n_fast + 2
        self.col_ti_has = self.n_fast + 3

        # --- T_i: the equation (20 parameters, +1 with the cross term) ---
        # K(s) = K_inf + (K_0 - K_inf) exp(-s / s_K): gain on a new observation after a gap s [s]
        self.k0_logit = nn.Parameter(torch.tensor(_inv_sigmoid(k0)))
        self.kinf_logit = nn.Parameter(torch.tensor(_inv_sigmoid(k_inf)))
        self.sk_raw = nn.Parameter(torch.tensor(_inv_softplus(s_k)))
        self.a = nn.Parameter(torch.zeros(self.n_fast))          # start exactly at the EWMA
        self.kappa_raw = nn.Parameter(torch.tensor(_inv_softplus(kappa0)))
        self.b0 = nn.Parameter(torch.zeros(()))
        if self.cross:
            self.c = nn.Parameter(torch.zeros(()))

        # --- V_rot: seq_v2's branch, verbatim (slow tail only) ---
        self.lstm_vt = nn.LSTM(self.n_slow, hidden_vt, num_layers=layers_vt, batch_first=True,
                               dropout=dropout if layers_vt > 1 else 0.0)
        self.norm_vt = nn.LayerNorm(hidden_vt)
        self.head_vt = nn.Sequential(nn.Linear(hidden_vt, head), nn.GELU(), nn.Linear(head, 1))

        self.n_params_ti = sum(p.numel() for n, p in self.named_parameters()
                               if not n.startswith(("lstm_vt", "norm_vt", "head_vt")))
        n_params = sum(p.numel() for p in self.parameters())
        if n_params >= 1_000_000:
            raise ValueError(f"parameter budget exceeded: {n_params:,} >= 1,000,000")
        self.n_params = n_params

    # ---- readable parameters ----
    def gains(self):
        k0 = torch.sigmoid(self.k0_logit)
        kinf = torch.sigmoid(self.kinf_logit)
        sk = nn.functional.softplus(self.sk_raw) + 1e-4
        return k0, kinf, sk

    def kappa(self):
        return nn.functional.softplus(self.kappa_raw) + 1e-3

    def physics(self, target_std):
        """K_0, K_inf, s_K [ms], per-channel gains a_c in eV per 1 sigma (per-shot) step,
        kappa (sigma units), b0 [eV], c (cross) -- for the record (physics_params.json)."""
        k0, kinf, sk = (float(v.detach()) for v in self.gains())
        std_ti = float(target_std[0])
        a = (self.a.detach() * std_ti).tolist()
        out = {"K0": k0, "K_inf": kinf, "s_K_ms": sk * 1e3,
               "a_eV_per_sigma": a, "a_sum_BES_eV": float(sum(a[:N_BES])),
               "a_sum_ECEI_eV": float(sum(a[N_BES:N_BES + N_ECEI])),
               "a_sum_MC_eV": float(sum(a[N_BES + N_ECEI:])),
               "kappa_sigma": float(self.kappa().detach()), "b0_eV": float(self.b0.detach()) * std_ti,
               "n_params_ti": int(self.n_params_ti)}
        if self.cross:
            out["c_eV_per_sigma2"] = float(self.c.detach()) * std_ti
        return out

    # ---- forward ----
    def ti_terms(self, x):
        """(B, L, n_in) -> (m, fast_term) each (B, L), normalized T_i units."""
        carried = x[..., self.col_ti_carried]
        stale = x[..., self.col_ti_stale]
        has = x[..., self.col_ti_has]
        new_obs = ((has > 0.5) & (stale < NEW_OBS_STALE)).to(x.dtype)
        # gap between the observation at t-1 and the one before it = stale at t-1 [s]
        gap = torch.expm1(stale).clamp(0.0, 1.0)
        gap_prev = torch.cat([torch.zeros_like(gap[:, :1]), gap[:, :-1]], dim=1)
        k0, kinf, sk = self.gains()
        gain = kinf + (k0 - kinf) * torch.exp(-gap_prev / sk)
        dev = x.device
        m = _ewma_scan(carried.cpu(), new_obs.cpu(), gain.cpu()).to(dev)

        fast = x[..., :self.n_fast]
        d = torch.cat([torch.zeros_like(fast[:, :1]), fast[:, 1:] - fast[:, :-1]], dim=1)
        kappa = self.kappa()
        g = kappa * torch.tanh(d / kappa)
        term = g @ self.a
        if self.cross:
            term = term + self.c * d[..., N_BES:N_BES + N_ECEI].mean(-1) * d[..., :N_BES].mean(-1)
        return m, term

    def _run_vt(self, x, lengths):
        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False)
            out, _ = self.lstm_vt(packed)
            out, _ = nn.utils.rnn.pad_packed_sequence(
                out, batch_first=True, total_length=x.shape[1])
        else:
            out, _ = self.lstm_vt(x)
        return self.norm_vt(out)

    def forward(self, x, lengths=None):
        """x (B, L, n_in) -> (B, L, 2) normalized [CES_TI, CES_VT]."""
        m, term = self.ti_terms(x)
        y_ti = m + term + self.b0
        h_vt = self._run_vt(x[..., self.n_fast:self.n_in], lengths)
        return torch.cat([y_ti.unsqueeze(-1), self.head_vt(h_vt)], dim=-1)
