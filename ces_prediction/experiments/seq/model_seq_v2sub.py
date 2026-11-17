"""seq v2 with a SUBSET of the fast channels in the T_i encoder (B.39, PREREGISTRATION_B39.md).

  v2e  T_i <- ECEI (4) + the 7 CES-derived channels
  v2b  T_i <- BES (9)  + the 7 CES-derived channels
  v2n  T_i <- BES (9) + ECEI (4) + the 7 CES-derived channels (no Mirnov)

The channels that are left out are removed from the encoder input, not set to zero, so no weight
exists that could read them (as in `model_seq_v2h`). Widths, layers, heads, dropout and the
V_rot branch are seq_v2's; the one controlled variable is the T_i encoder's input set.
The fast block is laid out [BES 9 | ECEI 4 | MC 2] by `seq_data.build_blocks`.
"""

import torch
import torch.nn as nn

from seq_data import N_FEATURES, N_FAST_CHANNELS
from model_seq_v2 import SeqCESLSTMv2

GROUPS = {"bes": tuple(range(0, 9)), "ecei": tuple(range(9, 13)), "mc": (13, 14)}


class SeqCESLSTMv2Sub(SeqCESLSTMv2):
    def __init__(self, groups=("bes", "ecei"), n_in=N_FEATURES, n_fast=N_FAST_CHANNELS,
                 hidden_ti=160, layers_ti=2, dropout=0.1, **kw):
        if n_fast != 15 or n_in != 22:
            raise ValueError("the channel groups are defined on the base 22-channel layout only")
        super().__init__(n_in=n_in, n_fast=n_fast, hidden_ti=hidden_ti, layers_ti=layers_ti,
                         dropout=dropout, n_extra_vt=0, n_extra_ti=0, **kw)
        fast = [c for g in groups for c in GROUPS[g]]
        cols = fast + list(range(self.n_fast, self.n_base))
        self.groups = tuple(groups)
        # not persistent: a checkpoint holds weights only and reloads under the named variant
        self.register_buffer("ti_cols", torch.tensor(cols, dtype=torch.long), persistent=False)
        self.lstm_ti = nn.LSTM(len(cols), hidden_ti, num_layers=layers_ti, batch_first=True,
                               dropout=dropout if layers_ti > 1 else 0.0)
        self.n_params = sum(p.numel() for p in self.parameters())

    def forward(self, x, lengths=None):
        """x (B, L, 22) -> (B, L, 2) normalized [CES_TI, CES_VT]."""
        h_ti = self._run(self.lstm_ti, self.norm_ti, x.index_select(-1, self.ti_cols), lengths)
        h_vt = self._run(self.lstm_vt, self.norm_vt, x[..., self.n_fast:self.n_base], lengths)
        return torch.cat([self.head_ti(h_ti), self.head_vt(h_vt)], dim=-1)
