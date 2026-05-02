from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


TIME_COLUMN = "time"
TARGET_COLUMNS = ("CES_TI", "CES_VT")


class KSTAR_CES_Dataset(Dataset):
    """KSTAR multimodal CES dataset backed by the real per-shot CSV files.

    Each sample is a fixed row-count history window ending at a valid CES target
    row. Because row spacing is not physically uniform, every sample also
    returns explicit time features for the true elapsed seconds in that window.
    """

    def __init__(self, data_dir, window_size=10, max_window_span=None):
        self.data_dir = Path(data_dir)
        self.window_size = int(window_size)
        self.max_window_span = max_window_span

        if self.window_size < 2:
            raise ValueError("window_size must be at least 2")

        self.files = sorted(self.data_dir.glob("*.csv"))
        if not self.files:
            raise FileNotFoundError(f"No CSV files found in {self.data_dir}")

        self.bes_cols, self.ecei_cols, self.mc_cols = self._infer_feature_columns()
        self.time_feature_cols = (
            "lookback_seconds",
            "delta_seconds",
            "log1p_lookback_seconds",
            "log1p_delta_seconds",
        )
        self.sample_indices = self._build_index()

    def _infer_feature_columns(self):
        for file_path in self.files:
            columns = pd.read_csv(file_path, nrows=0).columns.tolist()
            missing = [c for c in (TIME_COLUMN, *TARGET_COLUMNS) if c not in columns]
            if missing:
                continue

            bes_cols = [c for c in columns if c.startswith("BES_")]
            ecei_cols = [c for c in columns if c.startswith("ECEI_")]
            mc_cols = [c for c in columns if c.startswith("MC")]
            if bes_cols and ecei_cols and mc_cols:
                return bes_cols, ecei_cols, mc_cols

        raise ValueError("Could not infer BES/ECEI/MC columns from the CSV files")

    def _build_index(self):
        sample_indices = []
        usecols = [TIME_COLUMN, *TARGET_COLUMNS]

        for file_path in self.files:
            try:
                df = pd.read_csv(file_path, usecols=usecols)
            except ValueError:
                continue

            time = pd.to_numeric(df[TIME_COLUMN], errors="coerce").to_numpy()
            targets = df.loc[:, TARGET_COLUMNS].notna().all(axis=1).to_numpy()

            for idx in np.flatnonzero(targets):
                start = idx - self.window_size + 1
                if start < 0:
                    continue

                window_time = time[start : idx + 1]
                if not np.isfinite(window_time).all():
                    continue
                if np.any(np.diff(window_time) <= 0):
                    continue
                if self.max_window_span is not None:
                    span = window_time[-1] - window_time[0]
                    if span > self.max_window_span:
                        continue

                sample_indices.append((str(file_path), int(idx)))

        return sample_indices

    @lru_cache(maxsize=32)
    def _get_file_data(self, file_path):
        return pd.read_csv(file_path)

    def __len__(self):
        return len(self.sample_indices)

    @property
    def feature_dims(self):
        return {
            "bes": len(self.bes_cols),
            "ecei": len(self.ecei_cols),
            "mc": len(self.mc_cols),
            "time": len(self.time_feature_cols),
        }

    def _window_tensor(self, window_df, columns):
        values = window_df.loc[:, columns].apply(pd.to_numeric, errors="coerce")
        values = values.ffill().bfill().fillna(0.0).to_numpy(dtype=np.float32)
        return torch.from_numpy(values)

    def _time_features(self, time_values):
        time_values = time_values.astype(np.float32, copy=False)
        lookback = time_values[-1] - time_values
        delta = np.diff(time_values, prepend=time_values[0])
        delta[0] = 0.0

        features = np.stack(
            [
                lookback,
                delta,
                np.log1p(np.maximum(lookback, 0.0)),
                np.log1p(np.maximum(delta, 0.0)),
            ],
            axis=1,
        ).astype(np.float32)
        return torch.from_numpy(features)

    def __getitem__(self, i):
        file_path, idx = self.sample_indices[i]
        df = self._get_file_data(file_path)

        start = idx - self.window_size + 1
        window_df = df.iloc[start : idx + 1]

        bes = self._window_tensor(window_df, self.bes_cols)
        ecei = self._window_tensor(window_df, self.ecei_cols)
        mc = self._window_tensor(window_df, self.mc_cols)

        time_values = pd.to_numeric(window_df[TIME_COLUMN], errors="coerce").to_numpy()
        time_features = self._time_features(time_values)

        target_values = (
            df.loc[df.index[idx], list(TARGET_COLUMNS)]
            .astype(np.float32)
            .to_numpy(dtype=np.float32)
        )
        target = torch.from_numpy(target_values)

        return {
            "bes": bes,
            "ecei": ecei,
            "mc": mc,
            "time_features": time_features,
            "dt": time_features,
            "target": target,
            "file": file_path,
            "row_index": idx,
        }


if __name__ == "__main__":
    data_dir = Path(__file__).resolve().parents[1] / "data"
    dataset = KSTAR_CES_Dataset(data_dir=data_dir, window_size=10)
    print(f"CSV files: {len(dataset.files)}")
    print(f"Total samples: {len(dataset)}")
    print(f"Feature dims: {dataset.feature_dims}")

    if len(dataset) > 0:
        sample = dataset[0]
        print("BES shape:", tuple(sample["bes"].shape))
        print("ECEI shape:", tuple(sample["ecei"].shape))
        print("MC shape:", tuple(sample["mc"].shape))
        print("Time feature shape:", tuple(sample["time_features"].shape))
        print("Target shape:", tuple(sample["target"].shape))
