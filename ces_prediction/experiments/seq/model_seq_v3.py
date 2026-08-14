"""seq v3 (B.2 exploration): v2 + observation-masked causal attention readout.

iter009's single mechanism that survived ~40 controlled iterations is the
observation-masked attention pool over history observations. seq_v2 carries past
CES only through (carry-forward value, staleness) -- one observation deep; deeper
history must survive inside the recurrent state. v3 transplants the iter009
mechanism into the sequence frame: each step attends over the hidden states at
PAST fresh-observation steps of its own target, giving direct multi-observation
access (the "reach" lever) with the same routing as v2 (the V_rot path attends
over its slow-path states only).

Causality and masking, precisely:
  - a step s is a key iff the target was observed at s-1 (shifted mask) -- h_s is
    the first state whose carry-forward has absorbed that measurement;
  - queries at t attend over keys s <= t-1 only (strict lower-triangular mask),
    so the observation status and value at t never reach the step-t prediction;
  - a step with no available key gets zero attention output (residual keeps the
    v2 pathway) -- the analogue of iter009's all-masked fallback.

The attention output projections are zero-initialized, so training starts at
exactly the v2 function and the attention must earn its way in.
"""

import torch
import torch.nn as nn

from seq_data import N_FEATURES, N_FAST_CHANNELS


class _ObsCausalAttention(nn.Module):
    def __init__(self, hidden, dim):
        super().__init__()
        self.q = nn.Linear(hidden, dim, bias=False)
        self.k = nn.Linear(hidden, dim, bias=False)
        self.v = nn.Linear(hidden, dim, bias=False)
        self.out = nn.Linear(dim, hidden, bias=False)
        nn.init.zeros_(self.out.weight)  # start exactly at the v2 function
        self.scale = float(dim) ** 0.5

    def forward(self, h, obs_key):
        """h (B, L, H); obs_key (B, L) float/bool -- step is an admissible key."""
        scores = torch.matmul(self.q(h), self.k(h).transpose(1, 2)) / self.scale
        L = h.shape[1]
        causal = torch.ones(L, L, dtype=torch.bool, device=h.device).tril(-1)
        mask = causal.unsqueeze(0) & (obs_key > 0.5).unsqueeze(1)  # (B, L, L)
        scores = scores.masked_fill(~mask, float("-inf"))
        has_key = mask.any(dim=-1, keepdim=True)
        attn = torch.nan_to_num(torch.softmax(scores, dim=-1), nan=0.0)
        ctx = torch.matmul(attn, self.v(h)) * has_key
        return self.out(ctx)


class SeqCESLSTMv3(nn.Module):
    needs_obs = True  # forward wants the per-step per-target observation mask

    def __init__(self, n_in=N_FEATURES, n_fast=N_FAST_CHANNELS,
                 hidden_ti=160, layers_ti=2, hidden_vt=64, layers_vt=1,
                 head=64, dropout=0.1, attn_ti=48, attn_vt=32):
        super().__init__()
        self.n_fast = int(n_fast)
        self.n_slow = int(n_in) - self.n_fast
        if self.n_slow <= 0:
            raise ValueError(f"n_fast={n_fast} leaves no non-fast channels of {n_in}")

        self.lstm_ti = nn.LSTM(n_in, hidden_ti, num_layers=layers_ti, batch_first=True,
                               dropout=dropout if layers_ti > 1 else 0.0)
        self.norm_ti = nn.LayerNorm(hidden_ti)
        self.attn_ti = _ObsCausalAttention(hidden_ti, attn_ti)
        self.norm2_ti = nn.LayerNorm(hidden_ti)
        self.head_ti = nn.Sequential(nn.Linear(hidden_ti, head), nn.GELU(), nn.Linear(head, 1))

        self.lstm_vt = nn.LSTM(self.n_slow, hidden_vt, num_layers=layers_vt, batch_first=True,
                               dropout=dropout if layers_vt > 1 else 0.0)
        self.norm_vt = nn.LayerNorm(hidden_vt)
        self.attn_vt = _ObsCausalAttention(hidden_vt, attn_vt)
        self.norm2_vt = nn.LayerNorm(hidden_vt)
        self.head_vt = nn.Sequential(nn.Linear(hidden_vt, head), nn.GELU(), nn.Linear(head, 1))

        n_params = sum(p.numel() for p in self.parameters())
        if n_params >= 1_000_000:
            raise ValueError(f"parameter budget exceeded: {n_params:,} >= 1,000,000")
        self.n_params = n_params

    def _run(self, lstm, norm, x, lengths):
        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False)
            out, _ = lstm(packed)
            out, _ = nn.utils.rnn.pad_packed_sequence(
                out, batch_first=True, total_length=x.shape[1])
        else:
            out, _ = lstm(x)
        return norm(out)

    @staticmethod
    def _shift_obs(obs_t):
        """(B, L) observed-at-s -> admissible-key-at-s (the step AFTER the
        observation, whose carry has absorbed it). Padded steps stay 0."""
        return torch.cat([torch.zeros_like(obs_t[:, :1]), obs_t[:, :-1]], dim=1)

    def forward(self, x, lengths=None, obs=None):
        """x (B, L, n_in); obs (B, L, 2) observed-at-row mask -> (B, L, 2)."""
        if obs is None:
            raise ValueError("SeqCESLSTMv3 requires the per-step observation mask (obs)")
        h_ti = self._run(self.lstm_ti, self.norm_ti, x, lengths)
        h_vt = self._run(self.lstm_vt, self.norm_vt, x[..., self.n_fast:], lengths)
        h_ti = self.norm2_ti(h_ti + self.attn_ti(h_ti, self._shift_obs(obs[..., 0])))
        h_vt = self.norm2_vt(h_vt + self.attn_vt(h_vt, self._shift_obs(obs[..., 1])))
        return torch.cat([self.head_ti(h_ti), self.head_vt(h_vt)], dim=-1)
