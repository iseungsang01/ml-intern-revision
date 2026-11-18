"""seq v2 with per-row source gating in front of the T_i LSTM (B.67, PREREGISTRATION_B67.md section 3).

The 22 inputs of a row (seq_data.build_blocks layout) form four source groups:

    BES          x[..., 0:9]     (9)
    ECEI         x[..., 9:13]    (4)
    Mirnov       x[..., 13:15]   (2)
    CES-derived  x[..., 15:22]   (7: log1p dt | T_i carried, log1p staleness, has | V_rot carried, log1p staleness, has)

Each group s gets its own embedding e_s = LayerNorm(Linear(x_s -> 32)).  Gate logits come from the concatenated
embeddings through Linear(128 -> 32), GELU, Linear(32 -> 4); the gates a = softmax(logits) are per row.

    v2s  (arm S, gated concatenation)  LSTM input = [4 a_1 e_1 | 4 a_2 e_2 | 4 a_3 e_3 | 4 a_4 e_4]   (128)
    v2a  (arm A, attention pooling)    LSTM input = sum_s a_s e_s                                    (32)

Uniform gates (a_s = 1/4) give v2s the plain concatenation of the embeddings.  The gate at row t reads row t only,
so the model stays causal.  Everything else is seq_v2's: the T_i LSTM (2 layers, 160 wide, dropout 0.1 between the
layers), its LayerNorm and head, and the V_rot branch (history channels only, never the fast channels).

The two variants are NOT in seq/seq_models.py (no shared file is edited during the running GPU batch):
`register()` inserts them into seq_models.SEQ_MODELS at run time; train_b67.py / eval_b67.py call it before
train_seq.main() / eval_seq.compare(), which resolve CES_SEQ_MODEL through that same dict object.
"""
from __future__ import annotations

import functools
import sys
from pathlib import Path

import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
SEQ_DIR = HERE.parent / "seq"
CES_DIR = HERE.parents[1]
for _p in (str(CES_DIR), str(SEQ_DIR)):      # seq/ ends up first, as train_seq / eval_seq put it
    if _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)

from seq_data import N_FEATURES, N_FAST_CHANNELS  # noqa: E402
from model_seq_v2 import SeqCESLSTMv2  # noqa: E402

# (name, first channel, end channel) in the base 22-channel layout of seq_data.build_blocks
GROUPS = (("bes", 0, 9), ("ecei", 9, 13), ("mc", 13, 15), ("ces", 15, 22))
SOURCE_LABELS = ("BES", "ECEI", "Mirnov", "CES-derived")
EMB = 32
GATE_HIDDEN = 32
PARAM_BUDGET = 1_000_000


class SeqCESLSTMv2Gate(SeqCESLSTMv2):
    def __init__(self, mode="S", n_in=N_FEATURES, n_fast=N_FAST_CHANNELS, hidden_ti=160, layers_ti=2,
                 dropout=0.1, emb=EMB, gate_hidden=GATE_HIDDEN, **kw):
        if mode not in ("S", "A"):
            raise ValueError(f"mode must be 'S' (gated concatenation) or 'A' (attention pooling), got {mode!r}")
        if n_fast != 15 or n_in != 22:
            raise ValueError("the source groups are defined on the base 22-channel layout only")
        super().__init__(n_in=n_in, n_fast=n_fast, hidden_ti=hidden_ti, layers_ti=layers_ti,
                         dropout=dropout, n_extra_vt=0, n_extra_ti=0, **kw)
        self.mode = mode
        self.n_groups = len(GROUPS)
        self.group_slices = tuple((a, b) for _, a, b in GROUPS)
        self.embed = nn.ModuleList([nn.Sequential(nn.Linear(b - a, emb), nn.LayerNorm(emb)) for _, a, b in GROUPS])
        self.gate_mlp = nn.Sequential(nn.Linear(self.n_groups * emb, gate_hidden), nn.GELU(),
                                      nn.Linear(gate_hidden, self.n_groups))
        lstm_in = self.n_groups * emb if mode == "S" else emb
        self.lstm_ti = nn.LSTM(lstm_in, hidden_ti, num_layers=layers_ti, batch_first=True,
                               dropout=dropout if layers_ti > 1 else 0.0)
        n_params = sum(p.numel() for p in self.parameters())
        if n_params >= PARAM_BUDGET:
            raise ValueError(f"parameter budget exceeded: {n_params:,} >= {PARAM_BUDGET:,}")
        self.n_params = n_params

    def embeddings(self, x):
        """x (..., 22) -> list of the four group embeddings, each (..., emb)."""
        return [f(x[..., a:b]) for f, (a, b) in zip(self.embed, self.group_slices)]

    def gates_and_input(self, x):
        """x (B, L, 22) -> gates (B, L, 4) and the T_i LSTM input (B, L, 128 for S | 32 for A); row-local."""
        e = torch.stack(self.embeddings(x), dim=-2)                       # (B, L, 4, emb)
        a = torch.softmax(self.gate_mlp(e.flatten(-2)), dim=-1)          # (B, L, 4)
        w = a.unsqueeze(-1) * e                                           # a_s e_s
        u = (self.n_groups * w).flatten(-2) if self.mode == "S" else w.sum(-2)
        return a, u

    def gates(self, x):
        """The per-row source gates (B, L, 4), order BES, ECEI, Mirnov, CES-derived."""
        return self.gates_and_input(x[..., :self.n_base])[0]

    def forward_with_gates(self, x, lengths=None):
        """x (B, L, 22) -> ((B, L, 2) normalized [CES_TI, CES_VT], gates (B, L, 4))."""
        a, u = self.gates_and_input(x[..., :self.n_base])
        h_ti = self._run(self.lstm_ti, self.norm_ti, u, lengths)
        h_vt = self._run(self.lstm_vt, self.norm_vt, x[..., self.n_fast:self.n_base], lengths)
        return torch.cat([self.head_ti(h_ti), self.head_vt(h_vt)], dim=-1), a

    def forward(self, x, lengths=None):
        return self.forward_with_gates(x, lengths)[0]


VARIANTS = {
    "v2s": functools.partial(SeqCESLSTMv2Gate, mode="S"),   # arm S, gated concatenation
    "v2a": functools.partial(SeqCESLSTMv2Gate, mode="A"),   # arm A, attention pooling
}


def register():
    """Insert v2s and v2a into seq_models.SEQ_MODELS (the single registry that train_seq and eval_seq read).
    Returns that dict.  Refuses if a name is already taken by another architecture."""
    import seq_models
    for name, factory in VARIANTS.items():
        have = seq_models.SEQ_MODELS.get(name)
        if have is not None and have is not factory:
            raise SystemExit(f"FATAL: {name!r} is already in seq_models.SEQ_MODELS with another architecture")
        seq_models.SEQ_MODELS[name] = factory
    return seq_models.SEQ_MODELS
