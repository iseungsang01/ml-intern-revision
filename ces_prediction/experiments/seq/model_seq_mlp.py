"""seq mlp22: a memoryless regression on the same 22 inputs (B.29 arm D).

PREREGISTRATION_B29.md sec. 2. The Diag2Diag-type baseline the review loop asked for: the
prediction at row t is a function of the feature row at t alone -- no recurrent state, no
neighbouring row. The feature row already carries the last observed value of each target and
its staleness (seq_data), so "memoryless" means "no state beyond what the row carries".

Routing is seq_v2's: T_i <- all 22 channels, V_rot <- the 7 slow channels only. The one
controlled variable against the backbone is the removal of the recurrence.
"""

import torch
import torch.nn as nn

from seq_data import N_FEATURES, N_FAST_CHANNELS


class SeqCESMLP(nn.Module):
    def __init__(self, n_in=N_FEATURES, n_fast=N_FAST_CHANNELS, hidden_ti=160, hidden_vt=64,
                 head=64, dropout=0.1):
        super().__init__()
        self.n_base = int(n_in)
        self.n_fast = int(n_fast)
        n_slow = self.n_base - self.n_fast
        self.enc_ti = nn.Sequential(nn.Linear(n_in, hidden_ti), nn.GELU(), nn.Dropout(dropout),
                                    nn.Linear(hidden_ti, hidden_ti), nn.GELU(),
                                    nn.LayerNorm(hidden_ti))
        self.head_ti = nn.Sequential(nn.Linear(hidden_ti, head), nn.GELU(), nn.Linear(head, 1))
        self.enc_vt = nn.Sequential(nn.Linear(n_slow, hidden_vt), nn.GELU(),
                                    nn.LayerNorm(hidden_vt))
        self.head_vt = nn.Sequential(nn.Linear(hidden_vt, head), nn.GELU(), nn.Linear(head, 1))
        self.n_params = sum(p.numel() for p in self.parameters())

    def forward(self, x, lengths=None):
        """x (B, L, 22) -> (B, L, 2) normalized [CES_TI, CES_VT]; every row independently."""
        ti = self.head_ti(self.enc_ti(x[..., :self.n_base]))
        vt = self.head_vt(self.enc_vt(x[..., self.n_fast:self.n_base]))
        return torch.cat([ti, vt], dim=-1)
