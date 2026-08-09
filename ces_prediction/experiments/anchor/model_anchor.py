# ANCHOR-EXPERIMENT (anchor): interpretable persistence-anchored linear-correction model.
#
# Motivation (external feedback): the current best model is a deep stack (3 sensor CNNs +
# time CNN + 2-layer Transformer history encoder + per-target multi-head attention pooling)
# whose structure is an artifact of architecture search, not of the task — and the history
# window size (4) has no principled justification. This redesign derives the inputs from the
# task definition instead. The 10 ms gap-filling nowcast needs exactly:
#   (1) the most recent OBSERVED CES value per target        -> persistence anchor y1
#   (2) the local CES trend from the two most recent observed points -> slope m
#   (3) the gap length back to that observation              -> g (seconds)
#   (4) the fast-diagnostic fluctuation state near the target -> per-modality mean/std stats
# and combines them as a sum of nameable terms, per target t in {TI, VT}:
#
#   y_hat_t = [ (1-s_t) * y1_t + s_t * mean_obs_t ]      # anchor (learned denoising blend)
#           + beta_t * m_t * g_t                          # history-gradient extrapolation
#           + sum_mod rate_mod_t(stats_mod) * g_t         # fast-diagnostic drift correction
#
# All learned scalars/heads are ZERO-INITIALIZED (alpha_t starts at sigmoid(-4) ~ 0.02), so
# training starts at (approximately) pure persistence — skill_vs_persistence then measures
# exactly what the learned terms add. Each term is directly plottable per sample; the
# per-modality rate heads give BES/ECEI/MC attribution for free. ~1.3k parameters total.
# The window size no longer parameterizes the architecture: it only bounds how far back the
# nearest-observed search can reach (a data-availability cap, not a model choice).
#
# Contract preserved: forward(self, bes, ecei, mc, time_features=None, ces_history=None),
# output normalized [CES_TI, CES_VT] of shape (batch, 2), no denormalization inside.

from pathlib import Path

import torch
import torch.nn as nn

_BIG = 1.0e6  # sentinel lookback for unobserved/padded rows in the nearest-observed search


def _zero_init_rate_head(in_dim, hidden_dim):
    head = nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, 2),
    )
    nn.init.zeros_(head[2].weight)
    nn.init.zeros_(head[2].bias)
    return head


def _masked_stats(x, valid):
    """Per-channel mean and std over the valid window rows. x: (B, W, C), valid: (B, W)."""
    w = valid.unsqueeze(-1)
    denom = w.sum(dim=1).clamp_min(1.0)
    mean = (x * w).sum(dim=1) / denom
    mean_sq = (x * x * w).sum(dim=1) / denom
    var = (mean_sq - mean * mean).clamp_min(0.0)
    std = torch.sqrt(var + 1e-6)
    return torch.cat((mean, std), dim=1)  # (B, 2C)


class MultimodalCESPredictor(nn.Module):
    """Persistence-anchored linear-correction nowcast for [CES_TI, CES_VT]."""

    def __init__(
        self,
        window_size=10,
        bes_channels=9,
        ecei_channels=4,
        mc_channels=2,
        time_channels=4,
        ces_history_channels=3,
        hidden_dim=32,
        gap_cap_seconds=1.0,
        slope_cap=20.0,
    ):
        super().__init__()
        self.window_size = window_size
        self.time_channels = time_channels
        self.ces_history_channels = ces_history_channels
        self.gap_cap_seconds = gap_cap_seconds
        self.slope_cap = slope_cap

        # Per-modality fast-diagnostic rate heads over (mean, std) window statistics.
        self.rate_bes = _zero_init_rate_head(2 * bes_channels, hidden_dim)
        self.rate_ecei = _zero_init_rate_head(2 * ecei_channels, hidden_dim)
        self.rate_mc = _zero_init_rate_head(2 * mc_channels, hidden_dim)

        # Per-target anchor smoothing gate s_t = sigmoid(alpha_t): 0 -> pure nearest-observed
        # persistence, 1 -> flag-weighted mean of the observed history. Init near 0.
        self.alpha = nn.Parameter(torch.full((2,), -4.0))
        # Per-target trust in the two-point history-gradient extrapolation. Init 0.
        self.beta = nn.Parameter(torch.zeros(2))

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

    @staticmethod
    def _nearest_two_observed(vals, flags, lookback):
        """Most recent and second most recent observed (value, gap) per target.

        vals/flags/lookback: (B, W). Rows with flag 0 (unobserved, padded, or the masked
        target row) are pushed to _BIG lookback so they can never win the argmin.
        """
        masked_lb = lookback + (1.0 - flags) * _BIG
        k = min(2, masked_lb.shape[1])
        _, top_idx = (-masked_lb).topk(k, dim=1)
        i1 = top_idx[:, 0:1]
        i2 = top_idx[:, k - 1 : k]
        y1 = vals.gather(1, i1).squeeze(1)
        g1 = masked_lb.gather(1, i1).squeeze(1)
        y2 = vals.gather(1, i2).squeeze(1)
        g2 = masked_lb.gather(1, i2).squeeze(1)
        n_obs = flags.sum(dim=1)
        has1 = (n_obs >= 1).to(vals.dtype)
        has2 = (n_obs >= 2).to(vals.dtype)
        return y1, g1, y2, g2, has1, has2

    def _components(self, bes, ecei, mc, time_features, ces_history):
        lookback = time_features[..., 0]  # seconds back to the target row; 0 at the target

        if ces_history.shape[-1] >= 2:
            hist_vals = ces_history[..., 0:2]
        else:
            hist_vals = ces_history.new_zeros(*ces_history.shape[:2], 2)
        if ces_history.shape[-1] >= 4:
            hist_flags = ces_history[..., 2:4]
        else:
            hist_flags = ces_history.new_zeros(*ces_history.shape[:2], 2)

        anchors, extraps, gaps = [], [], []
        for t in range(2):
            vals, flags = hist_vals[..., t], hist_flags[..., t]
            y1, g1, y2, g2, has1, has2 = self._nearest_two_observed(vals, flags, lookback)
            gap = torch.clamp(g1, max=self.gap_cap_seconds) * has1  # 0 when no anchor exists

            weight_sum = flags.sum(dim=1).clamp_min(1.0)
            obs_mean = (flags * vals).sum(dim=1) / weight_sum
            s = torch.sigmoid(self.alpha[t])
            anchor = ((1.0 - s) * y1 + s * obs_mean) * has1

            slope = (y1 - y2) / torch.clamp(g2 - g1, min=5e-3)
            slope = torch.clamp(slope, min=-self.slope_cap, max=self.slope_cap) * has2
            extrap = self.beta[t] * slope * gap

            anchors.append(anchor)
            extraps.append(extrap)
            gaps.append(gap)

        anchor = torch.stack(anchors, dim=1)  # (B, 2)
        extrap = torch.stack(extraps, dim=1)  # (B, 2)
        gap = torch.stack(gaps, dim=1)  # (B, 2)

        # Valid (non-padding) rows: padding rows are exactly all-zero across every input.
        valid = (
            bes.abs().sum(dim=-1)
            + ecei.abs().sum(dim=-1)
            + mc.abs().sum(dim=-1)
            + time_features.abs().sum(dim=-1)
        ) > 0
        valid = valid.to(bes.dtype)

        rate_bes = self.rate_bes(_masked_stats(bes, valid))
        rate_ecei = self.rate_ecei(_masked_stats(ecei, valid))
        rate_mc = self.rate_mc(_masked_stats(mc, valid))
        delta_bes = rate_bes * gap
        delta_ecei = rate_ecei * gap
        delta_mc = rate_mc * gap

        prediction = anchor + extrap + delta_bes + delta_ecei + delta_mc
        return {
            "prediction": prediction,
            "anchor": anchor,
            "extrap": extrap,
            "delta_bes": delta_bes,
            "delta_ecei": delta_ecei,
            "delta_mc": delta_mc,
            "gap": gap,
        }

    def decompose(self, bes, ecei, mc, time_features=None, ces_history=None):
        """Named additive terms of the prediction, for interpretability figures."""
        time_features = self._prepare_time_features(time_features, bes)
        ces_history = self._prepare_ces_history(ces_history, bes)
        return self._components(bes, ecei, mc, time_features, ces_history)

    def forward(self, bes, ecei, mc, time_features=None, ces_history=None):
        time_features = self._prepare_time_features(time_features, bes)
        ces_history = self._prepare_ces_history(ces_history, bes)
        return self._components(bes, ecei, mc, time_features, ces_history)["prediction"]


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from dataset import KSTAR_CES_Dataset

    data_dir = Path(__file__).resolve().parents[3] / "data"
    dataset = KSTAR_CES_Dataset(data_dir=data_dir, window_size=4)
    sample = dataset[0]

    model = MultimodalCESPredictor.from_dataset(dataset, window_size=4)
    with torch.no_grad():
        preds = model(
            sample["bes"].unsqueeze(0),
            sample["ecei"].unsqueeze(0),
            sample["mc"].unsqueeze(0),
            sample["time_features"].unsqueeze(0),
            sample["ces_history"].unsqueeze(0),
        )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Output shape: {tuple(preds.shape)}, parameters: {n_params}")
