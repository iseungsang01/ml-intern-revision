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
        self.time_extractor = TimeFeatureEncoder(time_channels=time_channels)

        fusion_dim = sensor_feature_dim * 3 + 32
        self.fusion_fc = nn.Sequential(
            nn.Linear(fusion_dim, 160),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(160, 64),
            nn.GELU(),
            nn.Linear(64, 2),
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

        fused = torch.cat((bes_feat, ecei_feat, mc_feat, time_feat), dim=1)
        return self.fusion_fc(fused)


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
