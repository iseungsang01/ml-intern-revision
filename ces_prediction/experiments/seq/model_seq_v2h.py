"""seq v2h: seq_v2 with the fast diagnostics REMOVED from the T_i encoder (B.29 arm A).

PREREGISTRATION_B29.md sec. 2. The only ablation of the fast channels on record zeroes them
at evaluation time in the W = 2 windowed control (sec. 8ab) -- an out-of-distribution input
for a model trained with them, and not the main model. Here the T_i encoder is *trained*
on the slow state alone:

  T_i   <- dt + per target (carried last observation, staleness, has-observation)   [7]
  V_rot <- the same 7 channels (seq_v2's branch, verbatim)

BES / ECEI / MC never reach either branch: they are sliced off the input, not zeroed, so no
weight exists that could read them. Widths, layers, heads and dropout are seq_v2's; the one
controlled variable against the backbone is the T_i encoder's input set (15 fewer input
columns in its first LSTM layer = 9,600 fewer parameters). The two branches stay separate
encoders, as in seq_v2.
"""

import torch
import torch.nn as nn

from seq_data import N_FEATURES, N_FAST_CHANNELS
from model_seq_v2 import SeqCESLSTMv2


class SeqCESLSTMv2Hist(SeqCESLSTMv2):
    def __init__(self, n_in=N_FEATURES, n_fast=N_FAST_CHANNELS, hidden_ti=160, layers_ti=2,
                 dropout=0.1, **kw):
        # No B.6 extra channels: this arm is defined on the base 22-channel layout only.
        super().__init__(n_in=n_in, n_fast=n_fast, hidden_ti=hidden_ti, layers_ti=layers_ti,
                         dropout=dropout, n_extra_vt=0, n_extra_ti=0, **kw)
        self.lstm_ti = nn.LSTM(self.n_slow, hidden_ti, num_layers=layers_ti, batch_first=True,
                               dropout=dropout if layers_ti > 1 else 0.0)
        self.n_params = sum(p.numel() for p in self.parameters())

    def forward(self, x, lengths=None):
        """x (B, L, 22) -> (B, L, 2) normalized [CES_TI, CES_VT]; x[..., :n_fast] is never read."""
        slow = x[..., self.n_fast:self.n_base]
        h_ti = self._run(self.lstm_ti, self.norm_ti, slow, lengths)
        h_vt = self._run(self.lstm_vt, self.norm_vt, slow, lengths)
        return torch.cat([self.head_ti(h_ti), self.head_vt(h_vt)], dim=-1)
