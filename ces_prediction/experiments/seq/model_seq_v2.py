"""seq v2: the full-grid framing PLUS the iter009 physics routing.

§8d closed with a specific next step, and this is it. The minimal seq model (v1) shares
one LSTM between both targets, and §8d's third comparison showed that under identical
held-free data it still loses `CES_VT` on 4/4 seeds against iter009 -- because iter009's
V_rot head is *blocked from the fast diagnostics by construction*. That is not a tuning
detail; it is the observation model. The fast diagnostics carry no toroidal-rotation
information at 10 ms (NBI torque unobserved, Mirnov aliased), so letting a shared encoder
mix them into the V_rot pathway can only add variance.

v2 therefore routes exactly as iter009 does:

  T_i   <- full state: fast diagnostics + target history + time
  V_rot <- history + time ONLY (carry-forward value, staleness, has-observation, dt)

Two separate causal LSTMs, because the routing has to hold at the *encoder*, not just at
the head: a shared recurrent state would carry fast-diagnostic information into the V_rot
head no matter how the head is wired. The V_rot encoder is deliberately small -- it sees
7 channels and models a highly autocorrelated quantity.

Parameter budget mirrors the project rule: hard < 1,000,000.
"""

import torch
import torch.nn as nn

from seq_data import N_FEATURES, N_FAST_CHANNELS


class SeqCESLSTMv2(nn.Module):
    def __init__(self, n_in=N_FEATURES, n_fast=N_FAST_CHANNELS,
                 hidden_ti=160, layers_ti=2, hidden_vt=64, layers_vt=1,
                 head=64, dropout=0.1):
        super().__init__()
        self.n_fast = int(n_fast)
        self.n_slow = int(n_in) - self.n_fast   # dt + per-target (carried, staleness, has)
        if self.n_slow <= 0:
            raise ValueError(f"n_fast={n_fast} leaves no non-fast channels of {n_in}")

        self.lstm_ti = nn.LSTM(n_in, hidden_ti, num_layers=layers_ti, batch_first=True,
                               dropout=dropout if layers_ti > 1 else 0.0)
        self.norm_ti = nn.LayerNorm(hidden_ti)
        self.head_ti = nn.Sequential(nn.Linear(hidden_ti, head), nn.GELU(), nn.Linear(head, 1))

        self.lstm_vt = nn.LSTM(self.n_slow, hidden_vt, num_layers=layers_vt, batch_first=True,
                               dropout=dropout if layers_vt > 1 else 0.0)
        self.norm_vt = nn.LayerNorm(hidden_vt)
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

    def forward(self, x, lengths=None):
        """x (B, L, n_in) -> (B, L, 2) normalized [CES_TI, CES_VT].

        The V_rot branch slices off the fast channels; `seq_data` lays the features out
        as [fast | dt | TI(carried, stale, has) | VT(carried, stale, has)], so the slice
        is exactly the non-fast tail.
        """
        h_ti = self._run(self.lstm_ti, self.norm_ti, x, lengths)
        h_vt = self._run(self.lstm_vt, self.norm_vt, x[..., self.n_fast:], lengths)
        return torch.cat([self.head_ti(h_ti), self.head_vt(h_vt)], dim=-1)
