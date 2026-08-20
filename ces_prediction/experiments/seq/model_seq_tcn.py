"""Dilated causal TCN with the seq_v2 routing — the arm §8ac named and never measured.

§8ac closed with "it does not establish that recurrence is the only way to reach 50 steps:
a dilated causal TCN reaches 63 steps with 5 layers and remains an untested candidate."
This is that candidate, built so the ONLY difference from `seq_v2` is the family of the
sequence operator:

  * same two-branch routing — the `V_rot` branch sees only the non-fast tail
    (dt + per-target carried/staleness/has), never the fast diagnostics (§8ab);
  * same head shape (`Linear → GELU → Linear`);
  * same output contract, `(B, L, 2)` normalized `[CES_TI, CES_VT]`.

Receptive field is `2^(layers+1) − 1` for kernel 3 with dilations 1, 2, 4, …, which is why
B.9's reach ladder sits on 2 / 7 / 15 / 31 / 63: `tcn15` (3 layers) and `tcn63` (5 layers)
land on ladder rungs exactly, so H3/H4 are direct paired comparisons rather than
interpolations (PREREGISTRATION_B9.md §2.2).

**Strictly causal, and it has to be.** Padding is applied on the LEFT only and the tail is
trimmed, so output `t` is a function of inputs `≤ t`. That is what lets `eval_seq`'s
`_forward_truncated` score this model: it gathers a window that may contain rows after `t`,
and reads the output at the row's own index — correct only for a causal operator.
`tests/test_architecture.py::test_seq_family_is_causal` asserts it numerically.

`lengths` is accepted and ignored on purpose: `batched()` right-pads, and a causal operator
cannot see past the current step, so trailing padding never reaches a valid output.
"""

import torch
import torch.nn as nn

from seq_data import N_FEATURES, N_FAST_CHANNELS


class _CausalStack(nn.Module):
    """`layers` residual dilated conv blocks; receptive field 2^(layers+1) − 1 at k = 3."""

    def __init__(self, n_in, hidden, layers, kernel=3, dropout=0.1, dilations=None):
        """`dilations` overrides the doubling schedule; `layers` is then its length.

        The default 1, 2, 4, ... is what makes RF = 2^(layers+1) - 1, and it can only land
        on 3 / 7 / 15 / 31 / 63. A rung between two of those needs a schedule that is not a
        doubling -- RF 5 is dilations (1, 1) -- so the list is exposed rather than derived.
        Note the receptive field stays ODD for any schedule at kernel 3, because
        RF = 1 + (k-1) * sum(d); an even RF would need an even kernel, which would change
        more than the reach.
        """
        super().__init__()
        self.proj = nn.Linear(n_in, hidden)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.pads = []
        schedule = [2 ** i for i in range(layers)] if dilations is None else list(dilations)
        for dilation in schedule:
            self.convs.append(nn.Conv1d(hidden, hidden, kernel, dilation=dilation))
            self.norms.append(nn.LayerNorm(hidden))
            self.pads.append((kernel - 1) * dilation)
        self.drop = nn.Dropout(dropout)
        self.receptive_field = 1 + sum(self.pads)

    def stream_init(self, device, dtype):
        """Per-layer ring buffers holding each layer's own last `pad + 1` inputs.

        Zeros, because the batch path left-pads with zeros — so a stream started at t = 0
        reproduces the batch output exactly rather than merely approximating it.
        """
        return [torch.zeros(1, self.proj.out_features, pad + 1, device=device, dtype=dtype)
                for pad in self.pads]

    def stream_step(self, state, x_t):
        """(1, 1, n_in) + state -> (1, 1, hidden), O(1) in sequence length.

        This is the whole point of measuring the TCN at a 1 ms budget: §8ac argued a
        convolutional stack "recomputes its receptive field every step unless a streaming
        cache is built explicitly". This is that cache, so the family is priced by what it
        costs to deploy rather than by the absence of an implementation. Eval-only —
        dropout is identity here, which is why the buffers replay the batch math exactly.
        """
        h = self.proj(x_t)[:, 0]                               # (1, hidden)
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            buf = torch.roll(state[i], -1, dims=2)
            buf[:, :, -1] = h
            state[i] = buf
            z = nn.functional.gelu(conv(buf))[:, :, 0]         # (1, hidden)
            h = norm(h + z)
        return h.unsqueeze(1)                                  # (1, 1, hidden)

    def forward(self, x):
        h = self.proj(x).transpose(1, 2)                       # (B, hidden, L)
        for conv, norm, pad in zip(self.convs, self.norms, self.pads):
            z = conv(nn.functional.pad(h, (pad, 0)))           # left pad only => causal
            z = self.drop(nn.functional.gelu(z))
            h = norm((h + z).transpose(1, 2)).transpose(1, 2)
        return h.transpose(1, 2)                               # (B, L, hidden)


class SeqCESTCN(nn.Module):
    def __init__(self, n_in=N_FEATURES, n_fast=N_FAST_CHANNELS, layers=5,
                 hidden_ti=128, hidden_vt=48, head=64, dropout=0.1, dilations=None):
        super().__init__()
        self.n_fast = int(n_fast)
        self.n_slow = int(n_in) - self.n_fast
        if self.n_slow <= 0:
            raise ValueError(f"n_fast={n_fast} leaves no non-fast channels of {n_in}")

        self.tcn_ti = _CausalStack(n_in, hidden_ti, layers, dropout=dropout,
                                   dilations=dilations)
        self.head_ti = nn.Sequential(nn.Linear(hidden_ti, head), nn.GELU(), nn.Linear(head, 1))
        self.tcn_vt = _CausalStack(self.n_slow, hidden_vt, layers, dropout=dropout,
                                   dilations=dilations)
        self.head_vt = nn.Sequential(nn.Linear(hidden_vt, head), nn.GELU(), nn.Linear(head, 1))

        self.receptive_field = self.tcn_ti.receptive_field
        n_params = sum(p.numel() for p in self.parameters())
        if n_params >= 1_000_000:
            raise ValueError(f"parameter budget exceeded: {n_params:,} >= 1,000,000")
        self.n_params = n_params

    def forward(self, x, lengths=None):
        """x (B, L, n_in) -> (B, L, 2) normalized [CES_TI, CES_VT]. `lengths` unused."""
        h_ti = self.tcn_ti(x)
        h_vt = self.tcn_vt(x[..., self.n_fast:])
        return torch.cat([self.head_ti(h_ti), self.head_vt(h_vt)], dim=-1)

    def stream_init(self, device=None, dtype=torch.float32):
        device = device or next(self.parameters()).device
        return {"ti": self.tcn_ti.stream_init(device, dtype),
                "vt": self.tcn_vt.stream_init(device, dtype)}

    def stream_step(self, state, x_t):
        """One online step: (1, 1, n_in) -> (1, 1, 2). Equals `forward`'s row t (see tests)."""
        h_ti = self.tcn_ti.stream_step(state["ti"], x_t)
        h_vt = self.tcn_vt.stream_step(state["vt"], x_t[..., self.n_fast:])
        return torch.cat([self.head_ti(h_ti), self.head_vt(h_vt)], dim=-1)
