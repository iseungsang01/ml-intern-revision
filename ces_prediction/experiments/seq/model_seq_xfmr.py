"""Causal local-attention transformer with the seq_v2 routing — B.9's third family.

H3 (PREREGISTRATION_B9.md §1) says the sequence-operator *family* is irrelevant once the
reach matches: an LSTM carrying 63 steps of state, a TCN whose receptive field is 63, and
attention over a 63-step causal window should land on the same skill. Two families cannot
test that — a tie between an LSTM and a TCN is also consistent with "convolutions and
recurrence happen to be equivalent here". Attention is the third, and the one whose
inductive bias is least like the other two: no locality prior, no decay, an explicit
content-based lookup over the whole window.

Same routing (`V_rot` never sees the fast channels), same head shape, same output contract
as `seq_v2` and `model_seq_tcn`, so the family is again the only difference.

**Strictly causal, twice over.** The attention mask is lower-triangular AND banded, so
position `t` attends to `[t − band + 1 .. t]` and nothing later. The band is what makes
"reach" mean the same thing here as it does for the other two families; without it the arm
would be a full-history model wearing a reach label.

**The band is per layer; the reach is what they compose to.** Stacking two 63-wide bands
reaches 125 steps, not 63 — the same compounding the TCN gets from its dilations. So the arm
is parameterised by the reach it must match and derives the per-layer band from it
(`band = ceil((reach − 1) / layers) + 1`), exactly as the TCN reaches 63 by composing five
kernel-3 layers rather than by one 63-wide operator.
`tests/test_architecture.py::test_seq_family_is_causal_and_reach_bounded` asserts the
composed reach numerically, which is how the 125-step version was caught.

`lengths` is accepted and ignored: `batched()` right-pads, and a causal mask means trailing
padding can never reach a valid output.
"""

import math

import torch
import torch.nn as nn

from seq_data import N_FEATURES, N_FAST_CHANNELS


def _sinusoidal(length, dim, device, dtype):
    pos = torch.arange(length, device=device, dtype=dtype).unsqueeze(1)
    i = torch.arange(0, dim, 2, device=device, dtype=dtype)
    freq = torch.exp(-math.log(10000.0) * i / dim)
    pe = torch.zeros(length, dim, device=device, dtype=dtype)
    pe[:, 0::2] = torch.sin(pos * freq)
    pe[:, 1::2] = torch.cos(pos * freq)[:, : pe[:, 1::2].shape[1]]
    return pe


class _CausalLocalEncoder(nn.Module):
    """`layers` pre-norm attention blocks, each attending over a `window`-step causal band."""

    def __init__(self, n_in, d_model, layers, band, heads=4, ff_mult=2, dropout=0.1):
        super().__init__()
        self.band = int(band)
        self.d_model = int(d_model)
        self.proj = nn.Linear(n_in, d_model)
        self.attn = nn.ModuleList(
            nn.MultiheadAttention(d_model, heads, dropout=dropout, batch_first=True)
            for _ in range(layers))
        self.ff = nn.ModuleList(
            nn.Sequential(nn.Linear(d_model, ff_mult * d_model), nn.GELU(),
                          nn.Dropout(dropout), nn.Linear(ff_mult * d_model, d_model))
            for _ in range(layers))
        self.n1 = nn.ModuleList(nn.LayerNorm(d_model) for _ in range(layers))
        self.n2 = nn.ModuleList(nn.LayerNorm(d_model) for _ in range(layers))
        self._mask_cache = {}

    def _mask(self, length, device):
        """True = blocked. Allowed iff 0 <= t − s < band (one layer's share of the reach)."""
        key = (length, str(device))
        if key not in self._mask_cache:
            idx = torch.arange(length, device=device)
            delta = idx[:, None] - idx[None, :]
            self._mask_cache[key] = ~((delta >= 0) & (delta < self.band))
        return self._mask_cache[key]

    def forward(self, x):
        h = self.proj(x)
        h = h + _sinusoidal(h.shape[1], self.d_model, h.device, h.dtype)
        mask = self._mask(h.shape[1], h.device)
        for attn, ff, n1, n2 in zip(self.attn, self.ff, self.n1, self.n2):
            z = n1(h)
            h = h + attn(z, z, z, attn_mask=mask, need_weights=False)[0]
            h = h + ff(n2(h))
        return h


class SeqCESXfmr(nn.Module):
    def __init__(self, n_in=N_FEATURES, n_fast=N_FAST_CHANNELS, reach=63, layers=2,
                 d_ti=128, d_vt=32, heads=4, head=64, dropout=0.1):
        super().__init__()
        band = -(-(int(reach) - 1) // int(layers)) + 1      # ceil division
        self.n_fast = int(n_fast)
        self.n_slow = int(n_in) - self.n_fast
        if self.n_slow <= 0:
            raise ValueError(f"n_fast={n_fast} leaves no non-fast channels of {n_in}")

        self.enc_ti = _CausalLocalEncoder(n_in, d_ti, layers, band, heads, dropout=dropout)
        self.head_ti = nn.Sequential(nn.Linear(d_ti, head), nn.GELU(), nn.Linear(head, 1))
        self.enc_vt = _CausalLocalEncoder(self.n_slow, d_vt, layers, band, heads=2,
                                          dropout=dropout)
        self.head_vt = nn.Sequential(nn.Linear(d_vt, head), nn.GELU(), nn.Linear(head, 1))

        self.receptive_field = layers * (band - 1) + 1
        n_params = sum(p.numel() for p in self.parameters())
        if n_params >= 1_000_000:
            raise ValueError(f"parameter budget exceeded: {n_params:,} >= 1,000,000")
        self.n_params = n_params

    def forward(self, x, lengths=None):
        """x (B, L, n_in) -> (B, L, 2) normalized [CES_TI, CES_VT]. `lengths` unused."""
        h_ti = self.enc_ti(x)
        h_vt = self.enc_vt(x[..., self.n_fast:])
        return torch.cat([self.head_ti(h_ti), self.head_vt(h_vt)], dim=-1)
