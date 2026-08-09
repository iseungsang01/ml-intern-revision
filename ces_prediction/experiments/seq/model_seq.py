"""Causal sequence model for the seq-LSTM experiment.

Unidirectional LSTM over the full 10 ms grid; per-step two-target heads output
normalized [CES_TI, CES_VT] at every step. Supervision is loss-side masked to the
observed labels, so the architecture never needs the window/augmentation machinery.
Parameter budget mirrors the project rule: hard < 1,000,000.
"""

import torch
import torch.nn as nn

from seq_data import N_FEATURES


class SeqCESLSTM(nn.Module):
    def __init__(self, n_in=N_FEATURES, hidden=160, layers=2, head=64, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(n_in, hidden, num_layers=layers, batch_first=True,
                            dropout=dropout if layers > 1 else 0.0)
        self.norm = nn.LayerNorm(hidden)
        self.head_ti = nn.Sequential(nn.Linear(hidden, head), nn.GELU(), nn.Linear(head, 1))
        self.head_vt = nn.Sequential(nn.Linear(hidden, head), nn.GELU(), nn.Linear(head, 1))
        n_params = sum(p.numel() for p in self.parameters())
        if n_params >= 1_000_000:
            raise ValueError(f"parameter budget exceeded: {n_params:,} >= 1,000,000")
        self.n_params = n_params

    def forward(self, x, lengths=None):
        """x (B, L, n_in) -> (B, L, 2) normalized [CES_TI, CES_VT]."""
        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False)
            out, _ = self.lstm(packed)
            out, _ = nn.utils.rnn.pad_packed_sequence(
                out, batch_first=True, total_length=x.shape[1])
        else:
            out, _ = self.lstm(x)
        out = self.norm(out)
        return torch.cat([self.head_ti(out), self.head_vt(out)], dim=-1)
