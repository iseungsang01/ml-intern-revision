"""B.12 arms (PREREGISTRATION_B12.md §4). Every arm maps per-block features
x (B, L, 32) -> (B, L, 2) normalized [CES_TI, CES_VT]; arms that need the whole shot
also receive `shot_x` (B, S, 32), the concatenated blocks of the block's own file.

- `bilstm`  control: seq_v2 made bidirectional, with seq_v2's routing (the V_rot branch
            never sees the fast diagnostics).
- `np`      bilstm + a shot encoder: attention pooling over every step of the shot
            (context only) -> z, appended to BOTH branches. The single added route is
            shot-level information (§8au: 28% of V_rot variance is between-shot).
- `tm_pre` / `tm_scratch`  TokaMind's Transformer backbone (MAST-pretrained vs random),
            one token per (signal group, step) inside a sliding window. The only
            difference between the two is the initial weights of the backbone and of
            the positional / modality / role embeddings.
"""

from pathlib import Path

import torch
import torch.nn as nn

from b12_data import N_FAST, N_FEATURES

N_SLOW = N_FEATURES - N_FAST  # dt + 2 x 8


class BiLSTMRouted(nn.Module):
    def __init__(self, n_extra=0, hidden_ti=160, hidden_vt=64, head=64, dropout=0.1):
        super().__init__()
        self.lstm_ti = nn.LSTM(N_FEATURES + n_extra, hidden_ti // 2, num_layers=2, batch_first=True,
                               dropout=dropout, bidirectional=True)
        self.lstm_vt = nn.LSTM(N_SLOW + n_extra, hidden_vt // 2, num_layers=1, batch_first=True,
                               bidirectional=True)
        self.norm_ti, self.norm_vt = nn.LayerNorm(hidden_ti), nn.LayerNorm(hidden_vt)
        self.head_ti = nn.Sequential(nn.Linear(hidden_ti, head), nn.GELU(), nn.Linear(head, 1))
        self.head_vt = nn.Sequential(nn.Linear(hidden_vt, head), nn.GELU(), nn.Linear(head, 1))

    @staticmethod
    def _run(lstm, norm, x, lengths):
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=x.shape[1])
        return norm(out)

    def forward(self, x, lengths, extra=None):
        xt, xv = x, x[..., N_FAST:]
        if extra is not None:
            xt, xv = torch.cat([xt, extra], -1), torch.cat([xv, extra], -1)
        h_ti = self._run(self.lstm_ti, self.norm_ti, xt, lengths)
        h_vt = self._run(self.lstm_vt, self.norm_vt, xv, lengths)
        return torch.cat([self.head_ti(h_ti), self.head_vt(h_vt)], -1)


class BiLSTMArm(nn.Module):
    needs_shot = False

    def __init__(self):
        super().__init__()
        self.core = BiLSTMRouted()

    def forward(self, x, lengths, shot_x=None, shot_len=None):
        return self.core(x, lengths)


class ShotEncoderNP(nn.Module):
    """Conditional-NP style: a permutation-invariant summary of the shot's context."""
    needs_shot = True

    def __init__(self, z_dim=32, d=64, heads=4):
        super().__init__()
        self.embed = nn.Sequential(nn.Linear(N_FEATURES, d), nn.GELU(), nn.Linear(d, d))
        self.query = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.pool = nn.MultiheadAttention(d, heads, batch_first=True)
        self.to_z = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, z_dim))
        self.core = BiLSTMRouted(n_extra=z_dim)

    def shot_z(self, shot_x, shot_len):
        S = shot_x.shape[1]
        pad = torch.arange(S, device=shot_x.device)[None, :] >= shot_len.to(shot_x.device)[:, None]
        e = self.embed(shot_x)
        z, _ = self.pool(self.query.expand(len(shot_x), -1, -1), e, e, key_padding_mask=pad)
        return self.to_z(z[:, 0])

    def forward(self, x, lengths, shot_x=None, shot_len=None):
        z = self.shot_z(shot_x, shot_len)
        return self.core(x, lengths, extra=z[:, None, :].expand(-1, x.shape[1], -1))


# ---- TokaMind transplant -----------------------------------------------------------

TM_D, TM_LAYERS, TM_HEADS, TM_FF, TM_DROPOUT = 192, 4, 6, 768, 0.05  # tokamind-base-v2 pretrain.yaml
TM_MAX_POS = 50
TM_TIMESERIES_MOD = 1  # sorted(["profile", "timeseries", "video"]).index("timeseries")
TM_INPUT_ROLE = 0
WINDOW = 48
# token groups over the 32 features: BES 9, ECEI 4, MC 2, then dt + each target's 8 slow channels
GROUPS = [list(range(0, 9)), list(range(9, 13)), list(range(13, 15)),
          [15] + list(range(16, 24)), [15] + list(range(24, 32))]
TI_GROUP, VT_GROUP = 3, 4


class TokaMindArm(nn.Module):
    needs_shot = False
    window = WINDOW

    def __init__(self, pretrained_dir=None):
        super().__init__()
        self.proj = nn.ModuleList(nn.Linear(len(g), TM_D) for g in GROUPS)
        self.group_embed = nn.Embedding(len(GROUPS), TM_D)  # new: our signals are not MAST's
        self.pos_embed = nn.Embedding(TM_MAX_POS + 1, TM_D)
        self.mod_embed = nn.Embedding(3, TM_D)
        self.role_embed = nn.Embedding(3, TM_D)
        layer = nn.TransformerEncoderLayer(TM_D, TM_HEADS, TM_FF, TM_DROPOUT, activation="gelu", batch_first=True)
        self.backbone = nn.Module()
        self.backbone.encoder = nn.TransformerEncoder(layer, TM_LAYERS, enable_nested_tensor=False)
        self.head_ti = nn.Sequential(nn.Linear(TM_D, 64), nn.GELU(), nn.Linear(64, 1))
        self.head_vt = nn.Sequential(nn.Linear(TM_D, 64), nn.GELU(), nn.Linear(64, 1))
        self.pretrained = pretrained_dir is not None
        if self.pretrained:
            self._load(Path(pretrained_dir))

    def _load(self, d):
        bb = torch.load(d / "backbone.pt", map_location="cpu", weights_only=True)
        self.backbone.load_state_dict(bb, strict=True)
        te = torch.load(d / "token_encoder.pt", map_location="cpu", weights_only=True)
        for name in ("pos_embed", "mod_embed", "role_embed"):
            getattr(self, name).weight.data.copy_(te[f"{name}.weight"])

    def pretrained_parameters(self):
        return [*self.backbone.parameters(), *self.pos_embed.parameters(),
                *self.mod_embed.parameters(), *self.role_embed.parameters()]

    def forward_window(self, x):
        """x (B, W, 32) -> (B, W, 2)."""
        B, W, _ = x.shape
        pos = torch.arange(W, device=x.device)
        toks = []
        for g, idx in enumerate(GROUPS):
            t = self.proj[g](x[..., idx]) + self.group_embed.weight[g] + self.pos_embed(pos)
            toks.append(t)
        tok = torch.stack(toks, 2)  # (B, W, G, D)
        tok = tok + self.mod_embed.weight[TM_TIMESERIES_MOD] + self.role_embed.weight[TM_INPUT_ROLE]
        h = self.backbone.encoder(tok.reshape(B, W * len(GROUPS), TM_D)).reshape(B, W, len(GROUPS), TM_D)
        return torch.cat([self.head_ti(h[:, :, TI_GROUP]), self.head_vt(h[:, :, VT_GROUP])], -1)

    def forward(self, x, lengths, shot_x=None, shot_len=None):
        """Whole padded blocks: sliding windows (stride W/4), predictions averaged."""
        B, L, _ = x.shape
        W = self.window
        if L <= W:
            pad = torch.zeros(B, W - L, x.shape[2], device=x.device)
            return self.forward_window(torch.cat([x, pad], 1))[:, :L]
        stride = W // 4
        starts = list(range(0, L - W + 1, stride))
        if starts[-1] != L - W:
            starts.append(L - W)
        out = torch.zeros(B, L, 2, device=x.device)
        cnt = torch.zeros(1, L, 1, device=x.device)
        wins = torch.stack([x[:, s:s + W] for s in starts], 1).reshape(-1, W, x.shape[2])
        preds = []
        for chunk in wins.split(256):
            preds.append(self.forward_window(chunk))
        preds = torch.cat(preds).reshape(B, len(starts), W, 2)
        for k, s in enumerate(starts):
            out[:, s:s + W] += preds[:, k]
            cnt[:, s:s + W] += 1
        return out / cnt


def build(arm, tokamind_dir=None):
    if arm == "bilstm":
        return BiLSTMArm()
    if arm == "np":
        return ShotEncoderNP()
    if arm == "tm_pre":
        return TokaMindArm(pretrained_dir=tokamind_dir)
    if arm == "tm_scratch":
        return TokaMindArm(pretrained_dir=None)
    raise SystemExit(f"unknown arm {arm!r}")


ARMS = ("bilstm", "np", "tm_pre", "tm_scratch")
