# EXPERIMENT: Multi-head additive attention pooling over the CES-history GRU output (the ONE
# controlled change this iteration). The current best already runs TWO target-specific additive-
# attention pools over the bidirectional GRU output (one routed to T_i, one to V_rot). Each of
# those pools is still a *single-head* readout: one scalar score per timestep -> one softmax over
# the window -> one weighted-average summary. This iteration upgrades each target's pool from a
# single additive-attention head to a MULTI-HEAD additive-attention pool: the 2*hidden GRU output
# is split into `num_heads` subspaces, each head learns its own per-timestep score and softmax over
# the window, and each head pools only its own value subspace. The per-head summaries are then
# concatenated back to exactly 2*hidden and fed to the SAME projection as before. Everything else
# is byte-for-byte the current best baseline: the bidirectional GRU (1 layer, hidden 64), the
# target-aware routing (TI = fast diagnostics + history + time, VT = history + time only), all
# sensor/time encoders, every capacity, the per-target proj LayerNorm->Linear->GELU, and -- the
# critical invariant -- the projection input width (2*hidden) and output width (output_dim=64), so
# ti_head and vt_head input shapes are UNCHANGED. The change is isolated to *how each target's
# attention pool summarizes* the window.
#
# Hypothesis: a single additive-attention head can express only ONE softmax weighting over the
# window, forcing each target's history readout into a single compromise temporal emphasis. But the
# interpolation bar the model must beat combines several distinct cues at once -- the nearest
# observed sample BEFORE the masked gap, the nearest observed sample AFTER it, and the local trend
# across them. V_rot in particular is essentially a distance-aware, two-sided interpolation of the
# observed rotation samples flanking the gap, which a single weighting cannot represent jointly
# (it cannot simultaneously place mass on the closest-past and closest-future rows AND track their
# slope). Multiple attention heads let each target's readout form a richer summary -- e.g. one head
# locking onto the nearest pre-gap sample, one onto the nearest post-gap sample, one onto the
# trend -- while staying entirely within the known-good attention-pooling mechanism. This is NOT
# capacity scaling of d_model/feed-forward/depth (the heads partition the existing 2*hidden width,
# adding only two tiny score projections), NOT a skip-path variant, NOT a local-conv extractor (all
# known failed paths), and NOT a window change. It builds directly on attention pooling (the best
# change so far) and Pre-LayerNorm stability.

from pathlib import Path

import torch
import torch.nn as nn


class TimeAwareSensorEncoder(nn.Module):
    """Encode one diagnostic stream together with true irregular-time features."""

    def __init__(
        self,
        sensor_channels,
        time_channels=4,
        ces_history_channels=3,
        hidden_channels=64,
        output_dim=96,
    ):
        super().__init__()
        in_channels = sensor_channels + time_channels + ces_history_channels
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_channels),
            nn.GELU(),
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_channels),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(hidden_channels, output_dim),
            nn.GELU(),
        )

    def forward(self, sensor_values, time_features, ces_history):
        x = torch.cat((sensor_values, time_features, ces_history), dim=-1)
        return self.net(x.permute(0, 2, 1))


class TimeFeatureEncoder(nn.Module):
    def __init__(self, time_channels=4, hidden_channels=32, output_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(time_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(hidden_channels, output_dim),
            nn.GELU(),
        )

    def forward(self, time_features):
        return self.net(time_features.permute(0, 2, 1))


class HistoryEncoder(nn.Module):
    """Dedicated sequential encoder for previous-CES history (+ irregular time).

    The previous-CES values carry the dominant V_rot signal (toroidal rotation is highly
    persistent on the 10 ms grid), so they get their own GRU pathway rather than being
    folded only into the per-sensor CNNs.

    The GRU is **bidirectional**: the masked target timestep sits inside the window, with
    observed CES values on both sides of it, so forward + backward passes both contribute.

    Readout is **target-specific MULTI-HEAD attention pooling over the full GRU output
    sequence**. For each target, a small linear maps each timestep's bidirectional
    representation to ``num_heads`` scores; each head softmax-normalizes across the window and
    pools only its own slice of the 2*hidden value vector. The per-head summaries are
    concatenated back to width 2*hidden (identical to the previous single-head output) and then
    projected by that target's head. One projected summary is consumed by the T_i pathway, the
    other by the V_rot pathway.

    Motivation: attention pooling (single best architecture change so far) lets the encoder
    up-weight the observed rows temporally nearest the masked target -- the same distance-aware,
    two-sided emphasis past+future PCHIP interpolation (the bar) applies. A *single* attention
    head can express only one weighting over the window, but interpolation combines several cues
    at once (nearest pre-gap sample, nearest post-gap sample, local trend). Multiple heads let
    each target's readout represent these jointly, while staying within the known-good attention
    mechanism. The two targets still get independent pools because they use history differently:
    V_rot relies on history almost entirely and wants a clean rotation-trajectory weighting, while
    T_i uses history only to complement the fast diagnostics. Each projected summary keeps width
    ``output_dim``, identical to the previous output, so downstream head shapes are unchanged.
    """

    def __init__(
        self,
        history_channels,
        time_channels=4,
        hidden_dim=64,
        output_dim=64,
        num_heads=4,
    ):
        super().__init__()
        self.gru = nn.GRU(
            input_size=history_channels + time_channels,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.attn_dim = hidden_dim * 2
        if self.attn_dim % num_heads != 0:
            raise ValueError(
                f"attention width {self.attn_dim} must be divisible by num_heads {num_heads}"
            )
        self.num_heads = num_heads
        self.head_dim = self.attn_dim // num_heads
        # Two independent multi-head additive-attention scorers over the per-timestep bidirectional
        # GRU outputs: one specialized for the T_i readout, one for the V_rot readout. Each emits
        # `num_heads` per-timestep scores.
        self.attn_ti = nn.Linear(self.attn_dim, num_heads)
        self.attn_vt = nn.Linear(self.attn_dim, num_heads)
        self.proj_ti = nn.Sequential(
            nn.LayerNorm(self.attn_dim),
            nn.Linear(self.attn_dim, output_dim),
            nn.GELU(),
        )
        self.proj_vt = nn.Sequential(
            nn.LayerNorm(self.attn_dim),
            nn.Linear(self.attn_dim, output_dim),
            nn.GELU(),
        )

    def _attention_pool(self, out, attn):
        # Multi-head additive attention pool. Each head scores every timestep, softmaxes over the
        # window, and pools only its own value subspace; per-head summaries are concatenated back
        # to the full attention width so the downstream projection input is unchanged.
        batch, window, _ = out.shape
        scores = attn(out)  # (batch, window, num_heads)
        weights = torch.softmax(scores, dim=1)  # softmax across the window, per head
        values = out.view(batch, window, self.num_heads, self.head_dim)
        pooled = (weights.unsqueeze(-1) * values).sum(dim=1)  # (batch, num_heads, head_dim)
        return pooled.reshape(batch, self.attn_dim)  # (batch, hidden_dim * 2)

    def forward(self, ces_history, time_features):
        seq = torch.cat((ces_history, time_features), dim=-1)
        out, _ = self.gru(seq)  # (batch, window, hidden_dim * 2)
        # Target-specific multi-head attention pools: emphasize the observed history rows nearest
        # the masked target across several complementary temporal patterns, with a separate set of
        # heads for each target's needs.
        pooled_ti = self._attention_pool(out, self.attn_ti)
        pooled_vt = self._attention_pool(out, self.attn_vt)
        return self.proj_ti(pooled_ti), self.proj_vt(pooled_vt)


class MultimodalCESPredictor(nn.Module):
    """Predict [CES_TI, CES_VT] from BES, ECEI, MC, and irregular time metadata."""

    def __init__(
        self,
        window_size=10,
        bes_channels=9,
        ecei_channels=4,
        mc_channels=2,
        time_channels=4,
        ces_history_channels=3,
        sensor_feature_dim=96,
        history_feature_dim=64,
        time_feature_dim=32,
    ):
        super().__init__()
        self.window_size = window_size
        self.time_channels = time_channels
        self.ces_history_channels = ces_history_channels

        self.bes_extractor = TimeAwareSensorEncoder(
            bes_channels,
            time_channels=time_channels,
            ces_history_channels=ces_history_channels,
            output_dim=sensor_feature_dim,
        )
        self.ecei_extractor = TimeAwareSensorEncoder(
            ecei_channels,
            time_channels=time_channels,
            ces_history_channels=ces_history_channels,
            output_dim=sensor_feature_dim,
        )
        self.mc_extractor = TimeAwareSensorEncoder(
            mc_channels,
            time_channels=time_channels,
            ces_history_channels=ces_history_channels,
            output_dim=sensor_feature_dim,
        )
        self.time_extractor = TimeFeatureEncoder(
            time_channels=time_channels, output_dim=time_feature_dim
        )
        self.history_extractor = HistoryEncoder(
            ces_history_channels,
            time_channels=time_channels,
            hidden_dim=64,
            output_dim=history_feature_dim,
        )

        # T_i uses fast diagnostics + history + time (physics: collisional T_e/n_e coupling).
        ti_in = sensor_feature_dim * 3 + time_feature_dim + history_feature_dim
        self.ti_head = nn.Sequential(
            nn.Linear(ti_in, 160),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(160, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

        # V_rot uses the dedicated history pathway + time only (physics/ablation: fast
        # diagnostics carry ~no toroidal-rotation info at the 10 ms grid).
        vt_in = history_feature_dim + time_feature_dim
        self.vt_head = nn.Sequential(
            nn.Linear(vt_in, 96),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(96, 48),
            nn.GELU(),
            nn.Linear(48, 1),
        )

    @classmethod
    def from_dataset(cls, dataset, **kwargs):
        dims = dataset.feature_dims
        return cls(
            bes_channels=dims["bes"],
            ecei_channels=dims["ecei"],
            mc_channels=dims["mc"],
            time_channels=dims["time"],
            ces_history_channels=dims.get("ces_history", 3),
            **kwargs,
        )

    def _prepare_time_features(self, time_features, reference):
        if time_features is None:
            batch, steps = reference.shape[:2]
            return reference.new_zeros(batch, steps, self.time_channels)

        if time_features.ndim != 3:
            raise ValueError("time_features must have shape (batch, window, channels)")
        if time_features.shape[:2] != reference.shape[:2]:
            raise ValueError("time_features and sensor windows must share batch/window dimensions")

        if time_features.shape[-1] == self.time_channels:
            return time_features
        if time_features.shape[-1] == 1 and self.time_channels == 4:
            lookback = time_features[..., 0]
            delta = torch.diff(lookback, dim=1, prepend=lookback[:, :1]).abs()
            return torch.stack(
                (
                    lookback,
                    delta,
                    torch.log1p(torch.clamp(lookback, min=0.0)),
                    torch.log1p(torch.clamp(delta, min=0.0)),
                ),
                dim=-1,
            )

        raise ValueError(
            f"Expected {self.time_channels} time channels, got {time_features.shape[-1]}"
        )

    def _prepare_ces_history(self, ces_history, reference):
        batch, steps = reference.shape[:2]
        if ces_history is None:
            return reference.new_zeros(batch, steps, self.ces_history_channels)

        if ces_history.ndim != 3:
            raise ValueError("ces_history must have shape (batch, window, channels)")
        if ces_history.shape[:2] != reference.shape[:2]:
            raise ValueError("ces_history and sensor windows must share batch/window dimensions")
        if ces_history.shape[-1] != self.ces_history_channels:
            raise ValueError(
                f"Expected {self.ces_history_channels} CES history channels, "
                f"got {ces_history.shape[-1]}"
            )
        return ces_history

    def forward(self, bes, ecei, mc, time_features=None, ces_history=None):
        time_features = self._prepare_time_features(time_features, bes)
        ces_history = self._prepare_ces_history(ces_history, bes)

        bes_feat = self.bes_extractor(bes, time_features, ces_history)
        ecei_feat = self.ecei_extractor(ecei, time_features, ces_history)
        mc_feat = self.mc_extractor(mc, time_features, ces_history)
        time_feat = self.time_extractor(time_features)
        # Target-specific history summaries: one for the T_i head, one for the V_rot head.
        hist_ti, hist_vt = self.history_extractor(ces_history, time_features)

        ti_in = torch.cat((bes_feat, ecei_feat, mc_feat, time_feat, hist_ti), dim=1)
        vt_in = torch.cat((hist_vt, time_feat), dim=1)

        ti = self.ti_head(ti_in)
        vt = self.vt_head(vt_in)
        return torch.cat((ti, vt), dim=1)


if __name__ == "__main__":
    from dataset import KSTAR_CES_Dataset

    data_dir = Path(__file__).resolve().parents[1] / "data"
    dataset = KSTAR_CES_Dataset(data_dir=data_dir, window_size=10)
    sample = dataset[0]

    model = MultimodalCESPredictor.from_dataset(dataset, window_size=10)
    with torch.no_grad():
        preds = model(
            sample["bes"].unsqueeze(0),
            sample["ecei"].unsqueeze(0),
            sample["mc"].unsqueeze(0),
            sample["time_features"].unsqueeze(0),
            sample["ces_history"].unsqueeze(0),
        )

    print(f"Loaded real CSV sample from {Path(sample['file']).name}:{sample['row_index']}")
    print(f"Feature dims: {dataset.feature_dims}")
    print(f"Output shape: {tuple(preds.shape)}")
