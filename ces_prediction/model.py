# EXPERIMENT: Add a PRE-HEAD LayerNorm on the fused input of each prediction head (T_i and
# V_rot). This is the ONE controlled change this iteration.
#
# Baseline being built on: the CURRENT BEST kept model — three time-aware sensor CNN encoders,
# the time encoder, the 2-layer Pre-LayerNorm Transformer history encoder with learned positional
# encoding, the target-aware routing (T_i = fast diagnostics + history + time; V_rot = history +
# time only), the two target-specific multi-head additive-attention pools that return weighted
# MEAN (+) weighted DISPERSION (std) with the per-target init-0 observed-neighbor attention bias,
# the per-target LayerNorm->Linear->GELU history projections, and both prediction heads. The most
# recent proposal was discarded (training failure / clean skill below best), so model.py was
# restored to this best and we build on it, never on the regression. Relative to that best model
# the ONLY new variable is the head-input LayerNorm described below; the encoder, routing, history
# readout (including the observed-neighbor bias), and head MLP widths are kept byte-identical.
#
# What changes: each prediction head's Sequential now begins with `nn.LayerNorm(in_dim)` applied to
# the concatenated fused feature vector, before the first Linear. For T_i this normalizes the
# heterogeneous concatenation [bes, ecei, mc, time, hist_ti] (five GELU-activated summaries living
# at different scales: three sensor-CNN outputs, the time-CNN output, and the attention-pooled
# history summary); for V_rot it normalizes [hist_vt, time]. Each LayerNorm has the standard
# learnable affine (gamma init 1, beta init 0). Param cost is tiny: 2*(384) + 2*(96) = 960 scalars,
# leaving the model far under the hard <1,000,000 budget.
#
# Hypothesis (grounded in THIS project's documented lessons): the single most reliable win in this
# search was switching the history encoder to Pre-LayerNorm (`norm_first=True`) — normalization
# BEFORE the nonlinear transform is what restored stability and generalization (val loss ~0.88 ->
# ~0.49). The prediction heads, however, still receive a RAW concatenation of summaries produced by
# separate sub-networks at uncalibrated relative scales, fed straight into a Linear+GELU stack with
# no input normalization. That heterogeneity can let one high-variance modality dominate the head's
# early gradients and couples the head's effective learning rate to upstream scale drift. A pre-head
# LayerNorm puts every fused feature on a common, well-conditioned footing each step, which is a
# generalization (clean-skill) lever rather than a capacity lever — and the chronic problem in this
# loop is generalization, not fit (bigger models overfit and score WORSE). It extends the known-good
# Pre-LayerNorm pathway to the fusion stage. This is NOT capacity scaling, NOT a residual/skip
# variant, NOT a local-conv extractor, NOT the previously rolled-back multiplicative fast-diagnostic
# trust gate, NOT an attention-logit change, and NOT a window change — it is a single, minimal,
# clearly-runnable input-normalization change on the two heads.

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
    persistent on the 10 ms grid), so they get their own pathway rather than being folded only
    into the per-sensor CNNs.

    The sequence mixer is a **2-layer Pre-LayerNorm Transformer encoder** (``norm_first=True``)
    over the history window. Self-attention models every timestep-to-timestep relation directly,
    so the masked target position (whose history channels are all zero) can attend straight to its
    nearest observed neighbours on each side of the gap. A **learned positional embedding** is
    added to the projected input so attention is order/distance aware (the window contains observed
    CES on both sides of the masked target, the same two-sided structure past+future PCHIP
    interpolation exploits). The rest of the readout follows.

    Readout is **target-specific MULTI-HEAD additive attention pooling over the full encoder
    output sequence, returning both a weighted MEAN and a weighted DISPERSION (std)**. For each
    target, a small linear maps each timestep's representation to ``num_heads`` scores; each head
    softmax-normalizes across the window. The attention logits also receive a **per-target,
    learned OBSERVED-NEIGHBOR bias** (a single scalar, initialized to 0, times that target's
    per-timestep observed flag), so the pooling can lean toward window rows where the target was
    actually observed — mirroring how interpolation uses only observed neighbours and naturally
    down-weighting the all-zero masked target row — without changing the baseline at init. With
    the resulting weights we pool, per head, both the first moment (mean of its ``d_model`` value
    slice) and the weighted standard deviation of that slice. The per-head mean and std summaries
    are each concatenated back to width ``d_model`` and then concatenated together to width
    ``2 * d_model`` before that target's projection. The mean captures the (two-sided)
    interpolation-like center of the recent history; the std exposes how volatile the recent
    history is around the gap — the quantity that distinguishes smooth bulk from high-local-
    activity (peak) regions, where the model must beat interpolation. One projected summary is
    consumed by the T_i pathway, the other by the V_rot pathway, because the two targets use
    history differently: V_rot relies on history almost entirely and wants a clean
    rotation-trajectory weighting, while T_i uses history only to complement the fast diagnostics.
    Each projected summary keeps width ``output_dim``, identical to the previous output, so
    downstream head shapes are unchanged.
    """

    def __init__(
        self,
        history_channels,
        time_channels=4,
        d_model=192,
        nhead=4,
        num_layers=2,
        dim_feedforward=384,
        output_dim=64,
        num_heads=4,
        max_window=64,
        dropout=0.1,
    ):
        super().__init__()
        self.history_channels = history_channels
        self.input_proj = nn.Linear(history_channels + time_channels, d_model)
        # Learned positional embedding; sliced to the actual window at forward time. Sized to the
        # max searchable window so a WINDOW_SIZE change (if ever introduced) stays in range.
        self.max_window = max_window
        self.pos_emb = nn.Parameter(torch.zeros(max_window, d_model))
        nn.init.normal_(self.pos_emb, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Pre-LayerNorm: the known-good stability setting.
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model),
        )

        self.attn_dim = d_model
        if d_model % num_heads != 0:
            raise ValueError(
                f"attention width {d_model} must be divisible by num_heads {num_heads}"
            )
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        # Two independent multi-head additive-attention scorers over the per-timestep encoder
        # outputs: one specialized for the T_i readout, one for the V_rot readout.
        self.attn_ti = nn.Linear(d_model, num_heads)
        self.attn_vt = nn.Linear(d_model, num_heads)
        # Per-target observed-neighbor attention bias scalars. Initialized to 0 so the pooling
        # starts byte-identical to the dispersion-pooling baseline; the gradient may grow them to
        # lean each target's pool toward its own observed history rows.
        self.attn_flag_bias_ti = nn.Parameter(torch.zeros(1))
        self.attn_flag_bias_vt = nn.Parameter(torch.zeros(1))
        # Projection inputs are 2 * d_model because each readout is the weighted mean (+) weighted
        # std of the per-timestep representations. Projected output width is unchanged.
        self.proj_ti = nn.Sequential(
            nn.LayerNorm(2 * d_model),
            nn.Linear(2 * d_model, output_dim),
            nn.GELU(),
        )
        self.proj_vt = nn.Sequential(
            nn.LayerNorm(2 * d_model),
            nn.Linear(2 * d_model, output_dim),
            nn.GELU(),
        )

    def _attention_pool(self, out, attn, flag_gamma=None, obs_flag=None):
        # Multi-head additive attention pool returning a weighted MEAN and a weighted DISPERSION.
        # Each head scores every timestep and softmaxes over the window; with those weights we pool
        # both the first moment (mean) and the weighted standard deviation of that head's value
        # subspace. Per-head mean and std summaries are each reshaped to the full attention width
        # and concatenated, so the readout exposes both the interpolation-like center and the local
        # variability of the recent history. Output width is 2 * d_model.
        batch, window, _ = out.shape
        scores = attn(out)  # (batch, window, num_heads)
        if obs_flag is not None and flag_gamma is not None:
            # Per-target observed-neighbor bias: a single learned scalar (init 0) times that
            # target's per-timestep observed flag, broadcast across heads. Lets the pool lean
            # toward rows where THIS target was actually observed and away from the all-zero
            # masked target row, mirroring interpolation's use of observed neighbours.
            scores = scores + flag_gamma * obs_flag.unsqueeze(-1)
        weights = torch.softmax(scores, dim=1)  # softmax across the window, per head
        values = out.view(batch, window, self.num_heads, self.head_dim)
        w = weights.unsqueeze(-1)  # (batch, window, num_heads, 1)
        mean = (w * values).sum(dim=1)  # (batch, num_heads, head_dim)
        mean_sq = (w * values * values).sum(dim=1)  # weighted E[v^2]
        var = (mean_sq - mean * mean).clamp_min(0.0)
        std = torch.sqrt(var + 1e-6)  # (batch, num_heads, head_dim)
        mean = mean.reshape(batch, self.attn_dim)  # (batch, d_model)
        std = std.reshape(batch, self.attn_dim)  # (batch, d_model)
        return torch.cat((mean, std), dim=1)  # (batch, 2 * d_model)

    def forward(self, ces_history, time_features):
        seq = torch.cat((ces_history, time_features), dim=-1)
        x = self.input_proj(seq)  # (batch, window, d_model)
        window = x.shape[1]
        if window > self.max_window:
            raise ValueError(
                f"history window {window} exceeds max_window {self.max_window}"
            )
        x = x + self.pos_emb[:window].unsqueeze(0)  # add learned positional encoding
        out = self.encoder(x)  # (batch, window, d_model)
        # Per-target observed flags from the fixed 4-channel ces_history contract layout
        # (channel 2 = CES_TI observed, channel 3 = CES_VT observed). Skipped if absent, which
        # preserves baseline behavior for reduced-channel dry runs.
        if ces_history.shape[-1] >= 4:
            ti_flag = ces_history[..., 2]  # (batch, window)
            vt_flag = ces_history[..., 3]  # (batch, window)
        else:
            ti_flag = None
            vt_flag = None
        # Target-specific multi-head attention pools: emphasize the observed history rows nearest
        # the masked target across several complementary temporal patterns, with a separate set of
        # heads (and observed-neighbor bias) for each target's needs, returning both center (mean)
        # and spread (std).
        pooled_ti = self._attention_pool(out, self.attn_ti, self.attn_flag_bias_ti, ti_flag)
        pooled_vt = self._attention_pool(out, self.attn_vt, self.attn_flag_bias_vt, vt_flag)
        return self.proj_ti(pooled_ti), self.proj_vt(pooled_vt)


class MultimodalCESPredictor(nn.Module):
    """Predict [CES_TI, CES_VT] from BES, ECEI, MC, and irregular time metadata.

    Each prediction head now begins with a **pre-head LayerNorm** over its concatenated fused
    feature vector (the ONE controlled change this iteration). The head inputs are raw GELU-
    activated summaries from separate sub-networks (sensor CNNs, time CNN, attention-pooled
    history) living at uncalibrated relative scales; normalizing them before the head MLP puts
    every modality on a common, well-conditioned footing each step, extending the known-good
    Pre-LayerNorm pathway from the history encoder to the fusion stage. This is a generalization
    (clean-skill) lever, not a capacity lever; head MLP widths and the rest of the network are
    unchanged.
    """

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
            output_dim=history_feature_dim,
        )

        # T_i uses fast diagnostics + history + time (physics: collisional T_e/n_e coupling).
        # A pre-head LayerNorm calibrates the heterogeneous fused summaries before the MLP.
        ti_in = sensor_feature_dim * 3 + time_feature_dim + history_feature_dim
        self.ti_head = nn.Sequential(
            nn.LayerNorm(ti_in),
            nn.Linear(ti_in, 160),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(160, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

        # V_rot uses the dedicated history pathway + time only (physics/ablation: fast
        # diagnostics carry ~no toroidal-rotation info at the 10 ms grid). Same pre-head LayerNorm.
        vt_in = history_feature_dim + time_feature_dim
        self.vt_head = nn.Sequential(
            nn.LayerNorm(vt_in),
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
        # Target-specific history summaries (each = flag-aware attention-weighted mean (+) std):
        # one for the T_i head, one for the V_rot head.
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
