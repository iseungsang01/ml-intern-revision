"""B.53 arm C (PREREGISTRATION_B53.md section 3): the history-only twin `v2h` with extra inputs to its
T_i branch only.

`v2h` (seq/model_seq_v2h.py) is defined on the base 22-channel layout: it passes n_extra_ti=0 to
seq_v2 and its forward reads x[..., n_fast:n_base] only, so channels appended by CES_SEQ_EXTRA_TI
would never reach it.  This class keeps everything of `v2h` and changes one thing: the first layer
of the T_i encoder reads the 7 slow channels PLUS the K channels that seq_data appends after the base
layout (x[..., 22:22 + K]).  The V_rot branch reads the same 7 slow channels as in `v2h`.

Construction order: `v2h` is built exactly as in the shared code (same RNG draws), and only when
K > 0 is its T_i LSTM replaced by one K columns wider.  With K = 0 the module is `v2h`: same
parameter shapes, same initial weights under the same seed, same forward.

Registered as `v2hx` in the seq registry by train_b53.py / eval_b53.py at run time; no shared file
is edited.
"""

import torch
import torch.nn as nn

from seq_data import extra_ti_meta, extra_vt_meta
from model_seq_v2h import SeqCESLSTMv2Hist


class SeqCESLSTMv2HistX(SeqCESLSTMv2Hist):
    def __init__(self, n_extra_ti=None, hidden_ti=160, layers_ti=2, dropout=0.1, **kw):
        if n_extra_ti is None:
            n_extra_ti = extra_ti_meta()[0]
        if extra_vt_meta()[0]:
            raise SystemExit("FATAL: v2hx is defined without V_rot extras; CES_SEQ_EXTRA_VT must be unset")
        super().__init__(hidden_ti=hidden_ti, layers_ti=layers_ti, dropout=dropout, **kw)
        self.n_extra_ti = int(n_extra_ti)
        if self.n_extra_ti:
            self.lstm_ti = nn.LSTM(self.n_slow + self.n_extra_ti, hidden_ti, num_layers=layers_ti,
                                   batch_first=True, dropout=dropout if layers_ti > 1 else 0.0)
        self.n_params = sum(p.numel() for p in self.parameters())
        if self.n_params >= 1_000_000:
            raise ValueError(f"parameter budget exceeded: {self.n_params:,} >= 1,000,000")

    def forward(self, x, lengths=None):
        """x (B, L, 22 + K) -> (B, L, 2).  T_i <- [slow 7 | extra K]; V_rot <- slow 7; x[..., :n_fast] never read."""
        slow = x[..., self.n_fast:self.n_base]
        x_ti = slow
        if self.n_extra_ti:
            x_ti = torch.cat([slow, x[..., self.n_base:self.n_base + self.n_extra_ti]], dim=-1)
        h_ti = self._run(self.lstm_ti, self.norm_ti, x_ti, lengths)
        h_vt = self._run(self.lstm_vt, self.norm_vt, slow, lengths)
        return torch.cat([self.head_ti(h_ti), self.head_vt(h_vt)], dim=-1)
