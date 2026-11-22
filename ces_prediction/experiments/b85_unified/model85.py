# Claude authored the implementation
# ces_prediction/experiments/b85_unified/model85.py
import os as _os
_cuda_prior = _os.environ.get('CUDA_VISIBLE_DEVICES', None)

import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[3]

# Restore CUDA env
if _cuda_prior is not None:
    _os.environ['CUDA_VISIBLE_DEVICES'] = _cuda_prior

import torch
import torch.nn as nn
torch.backends.cudnn.enabled = False  # Native ATen CUDA; see B85 addendum 3.
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
sys.path.insert(0, str(ROOT / 'ces_prediction' / 'experiments' / 'seq'))
from model_seq_v2 import SeqCESLSTMv2


class B85Model(nn.Module):
    """Unified CES estimator with two heads reading same LSTM state.
    X variant uses all 22 inputs; H variant zeros first 15 (fast channels).
    """
    def __init__(self, variant='X', n_in=22, n_fast=15, hidden=160,
                 layers=2, head_dim=64, dropout=0.1):
        super().__init__()
        self.variant = variant
        self.n_in = n_in
        self.n_fast = n_fast
        self.hidden = hidden
        self.layers = layers

        backbone = SeqCESLSTMv2(n_in=n_in, n_fast=n_fast, hidden_ti=hidden,
            layers_ti=layers, head=head_dim, dropout=dropout,
            n_extra_vt=0, n_extra_ti=0)
        self.lstm_ti = backbone.lstm_ti
        self.norm_ti = backbone.norm_ti
        self.head_F = backbone.head_ti

        # Head L5 - separate
        self.head_L5 = nn.Sequential(
            nn.Linear(hidden, head_dim),
            nn.GELU(),
            nn.Linear(head_dim, 1),
        )


    def _mask_input(self, x):
        if self.variant == 'H':
            x = x.clone()
            x[..., :self.n_fast] = 0.0
        return x

    def forward(self, x, lengths=None):
        """x: (B, L, 22) -> (B, L, 2) [F, L5]"""
        x = self._mask_input(x)
        B, L, _ = x.shape

        if lengths is not None:
            lengths_cpu = lengths.cpu().to(torch.int64)
            packed = pack_padded_sequence(x, lengths_cpu, batch_first=True, enforce_sorted=False)
            out_packed, _ = self.lstm_ti(packed)
            out, _ = pad_packed_sequence(out_packed, batch_first=True, total_length=L)
        else:
            out, _ = self.lstm_ti(x)

        out = self.norm_ti(out)

        pred_F = self.head_F(out)    # (B, L, 1)
        pred_L5 = self.head_L5(out)  # (B, L, 1)

        return torch.cat([pred_F, pred_L5], dim=-1)  # (B, L, 2)

    def forward_step(self, x, state=None):
        """Single step recurrent: x (B, 1, 22) or (B, 22).
        Returns pred (B, 2), new_state."""
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self._mask_input(x)

        out, new_state = self.lstm_ti(x, state)
        out = self.norm_ti(out)

        pred_F = self.head_F(out).squeeze(-1)    # (B, 1)
        pred_L5 = self.head_L5(out).squeeze(-1)  # (B, 1)

        pred = torch.cat([pred_F, pred_L5], dim=-1)  # (B, 2)
        return pred, new_state

    def param_count(self):
        return sum(p.numel() for p in self.parameters())


def create_model(variant='X', init_seed=1701):
    """Create model with deterministic initialization."""
    torch.manual_seed(init_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(init_seed)
    model = B85Model(variant=variant)
    assert model.param_count() < 1_000_000, f'Model has {model.param_count()} params >= 1M'
    return model


def create_twin_models(init_seed=1701):
    """Create paired X and H models with identical initialization."""
    torch.manual_seed(init_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(init_seed)
    model_x = B85Model(variant='X')

    # Copy exact same initial weights to H
    model_h = B85Model(variant='H')
    model_h.load_state_dict(model_x.state_dict())

    return model_x, model_h
