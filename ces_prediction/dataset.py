from functools import lru_cache
from itertools import combinations
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

    def __init__(
        self,
        data_dir,
        window_size=10,
        max_window_span=None,
        temporal_subset_augmentation=False,
        min_subset_size=2,
    ):
        self.data_dir = Path(data_dir)
        self.window_size = int(window_size)
        self.max_window_span = max_window_span
        self.temporal_subset_augmentation = bool(temporal_subset_augmentation)
        self.min_subset_size = int(min_subset_size)

        if self.window_size < 2:
            raise ValueError("window_size must be at least 2")
        if self.min_subset_size < 2:
            raise ValueError("min_subset_size must be at least 2")
        if self.min_subset_size > self.window_size:
            raise ValueError("min_subset_size must be less than or equal to window_size")

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
        self.ces_history_cols = ("CES_TI_history", "CES_VT_history", "CES_observed")
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
        # All columns we need for a sample to be valid
        all_feature_cols = [*self.bes_cols, *self.ecei_cols, *self.mc_cols]
        usecols = [TIME_COLUMN, *TARGET_COLUMNS, *all_feature_cols]

        for file_path in self.files:
            try:
                # Load all necessary columns
                df = pd.read_csv(file_path, usecols=usecols)
            except ValueError:
                continue

            df = df.dropna(subset=usecols).reset_index(drop=True)
            if len(df) < self.min_subset_size:
                continue

            time = df[TIME_COLUMN].to_numpy()
            deltas = np.diff(time)
            is_contiguous = deltas < 0.5

            block_start = 0
            for i, contiguous in enumerate(is_contiguous):
                if contiguous:
                    continue
                self._add_block_samples(sample_indices, file_path, block_start, i)
                block_start = i + 1
            self._add_block_samples(sample_indices, file_path, block_start, len(df) - 1)

        return sample_indices

    def _add_block_samples(self, sample_indices, file_path, block_start, block_end):
        block_len = block_end - block_start + 1
        if block_len < self.min_subset_size:
            return

        if not self.temporal_subset_augmentation:
            if block_len < self.window_size:
                return
            for idx in range(block_start + self.window_size - 1, block_end + 1):
                rows = tuple(range(idx - self.window_size + 1, idx + 1))
                sample_indices.append((str(file_path), rows))
            return

        for target_idx in range(block_start + self.min_subset_size - 1, block_end + 1):
            context_start = max(block_start, target_idx - self.window_size + 1)
            history_rows = range(context_start, target_idx)
            max_history = min(self.window_size - 1, target_idx - context_start)
            min_history = self.min_subset_size - 1

            for history_count in range(min_history, max_history + 1):
                for history_combo in combinations(history_rows, history_count):
                    rows = (*history_combo, target_idx)
                    sample_indices.append((str(file_path), rows))

    @lru_cache(maxsize=32)
    def _get_file_data(self, file_path):
        # We need to apply the same dropna logic when retrieving data in __getitem__
        # to ensure the indices match the sample_indices we built.
        all_feature_cols = [*self.bes_cols, *self.ecei_cols, *self.mc_cols]
        usecols = [TIME_COLUMN, *TARGET_COLUMNS, *all_feature_cols]
        df = pd.read_csv(file_path, usecols=usecols)
        return df.dropna(subset=usecols).reset_index(drop=True)

    def __len__(self):
        return len(self.sample_indices)

    @property
    def feature_dims(self):
        return {
            "bes": len(self.bes_cols),
            "ecei": len(self.ecei_cols),
            "mc": len(self.mc_cols),
            "time": len(self.time_feature_cols),
            "ces_history": len(self.ces_history_cols),
        }

    def _pad_tensor(self, tensor):
        if tensor.shape[0] > self.window_size:
            raise ValueError("sample length exceeds window_size")
        if tensor.shape[0] == self.window_size:
            return tensor

        pad_shape = (self.window_size - tensor.shape[0], *tensor.shape[1:])
        padding = torch.zeros(pad_shape, dtype=tensor.dtype)
        return torch.cat((tensor, padding), dim=0)

    def _window_tensor(self, window_df, columns):
        # Data is already cleaned via dropna in _get_file_data, so no need for ffill/bfill
        values = window_df.loc[:, columns].to_numpy(dtype=np.float32)
        return self._pad_tensor(torch.from_numpy(values))

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
        return self._pad_tensor(torch.from_numpy(features))

    def _ces_history(self, window_df):
        values = window_df.loc[:, TARGET_COLUMNS].to_numpy(dtype=np.float32)
        observed = np.ones((len(window_df), 1), dtype=np.float32)

        values[-1, :] = 0.0
        observed[-1, 0] = 0.0

        history = np.concatenate((values, observed), axis=1)
        return self._pad_tensor(torch.from_numpy(history))

    def __getitem__(self, i):
        file_path, row_indices = self.sample_indices[i]
        df = self._get_file_data(file_path)

        window_df = df.iloc[list(row_indices)]

        bes = self._window_tensor(window_df, self.bes_cols)
        ecei = self._window_tensor(window_df, self.ecei_cols)
        mc = self._window_tensor(window_df, self.mc_cols)

        time_values = window_df[TIME_COLUMN].to_numpy()
        time_features = self._time_features(time_values)
        ces_history = self._ces_history(window_df)
        input_mask = torch.zeros(self.window_size, dtype=torch.bool)
        input_mask[: len(row_indices)] = True
        padded_row_indices = torch.full((self.window_size,), -1, dtype=torch.long)
        padded_row_indices[: len(row_indices)] = torch.tensor(row_indices, dtype=torch.long)

        idx = row_indices[-1]
        target_values = (
            df.loc[df.index[idx], list(TARGET_COLUMNS)]
            .to_numpy(dtype=np.float32)
        )
        target = torch.from_numpy(target_values)

        return {
            "bes": bes,
            "ecei": ecei,
            "mc": mc,
            "time_features": time_features,
            "dt": time_features,
            "ces_history": ces_history,
            "input_mask": input_mask,
            "target": target,
            "file": file_path,
            "row_index": idx,
            "row_indices": padded_row_indices,
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
