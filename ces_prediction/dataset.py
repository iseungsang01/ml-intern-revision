import hashlib
import json
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
        normalization_stats=None,
        use_disk_cache=True,
    ):
        self.data_dir = Path(data_dir)
        self.window_size = int(window_size)
        self.max_window_span = max_window_span
        self.temporal_subset_augmentation = bool(temporal_subset_augmentation)
        self.min_subset_size = int(min_subset_size)
        self.normalization_stats = normalization_stats
        self.use_disk_cache = bool(use_disk_cache)

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
        if not self._load_disk_cache():
            self._preload_data()
            self._build_index()
            self._save_disk_cache()

    def _preload_data(self):
        """Pre-load all CSV files as compact numpy arrays."""
        self.data_cache = {}
        self.file_arrays = []
        self.valid_files = []
        usecols = self._required_columns()
        self._column_slices = self._build_column_slices(usecols)
        
        print(f"Pre-loading {len(self.files)} CSV files into memory...")
        for file_path in self.files:
            try:
                df = pd.read_csv(file_path, usecols=usecols)
                clean_df = df.dropna(subset=usecols).reset_index(drop=True)
                if not clean_df.empty:
                    values = clean_df.loc[:, usecols].to_numpy(dtype=np.float32, copy=True)
                    self.data_cache[str(file_path)] = values
                    self.file_arrays.append(values)
                    self.valid_files.append(str(file_path))
            except Exception as e:
                print(f"Warning: Could not load {file_path}: {e}")

    def _build_column_slices(self, usecols):
        positions = {name: i for i, name in enumerate(usecols)}
        return {
            "time": positions[TIME_COLUMN],
            "target": np.asarray([positions[c] for c in TARGET_COLUMNS], dtype=np.int64),
            "bes": np.asarray([positions[c] for c in self.bes_cols], dtype=np.int64),
            "ecei": np.asarray([positions[c] for c in self.ecei_cols], dtype=np.int64),
            "mc": np.asarray([positions[c] for c in self.mc_cols], dtype=np.int64),
        }

    def _cache_path(self):
        cache_dir = self.data_dir / ".ces_cache"
        signature = {
            "version": 2,
            "window_size": self.window_size,
            "temporal_subset_augmentation": self.temporal_subset_augmentation,
            "min_subset_size": self.min_subset_size,
            "columns": self._required_columns(),
            "files": [
                {
                    "name": file_path.name,
                    "size": file_path.stat().st_size,
                    "mtime_ns": file_path.stat().st_mtime_ns,
                }
                for file_path in self.files
            ],
        }
        payload = json.dumps(signature, sort_keys=True).encode("utf-8")
        digest = hashlib.sha256(payload).hexdigest()[:20]
        return cache_dir / f"kstar_ces_dataset_{digest}.npz"

    def _load_disk_cache(self):
        if not self.use_disk_cache:
            return False

        cache_path = self._cache_path()
        if not cache_path.exists():
            return False

        try:
            payload = np.load(cache_path, allow_pickle=False)
            self.valid_files = payload["valid_files"].astype(str).tolist()
            self.file_arrays = [payload[f"file_{i}"] for i in range(len(self.valid_files))]
            self.data_cache = {
                file_path: self.file_arrays[i]
                for i, file_path in enumerate(self.valid_files)
            }
            self.sample_file_indices = payload["sample_file_indices"].astype(np.int32, copy=False)
            self.sample_row_indices = payload["sample_row_indices"].astype(np.int32, copy=False)
            self.sample_lengths = payload["sample_lengths"].astype(np.int16, copy=False)
            self._column_slices = json.loads(payload["column_slices"].item())
            self._column_slices["target"] = np.asarray(self._column_slices["target"], dtype=np.int64)
            self._column_slices["bes"] = np.asarray(self._column_slices["bes"], dtype=np.int64)
            self._column_slices["ecei"] = np.asarray(self._column_slices["ecei"], dtype=np.int64)
            self._column_slices["mc"] = np.asarray(self._column_slices["mc"], dtype=np.int64)
            print(f"Loaded dataset cache: {cache_path}")
            return True
        except Exception as e:
            print(f"Warning: Could not load dataset cache {cache_path}: {e}")
            return False

    def _save_disk_cache(self):
        if not self.use_disk_cache:
            return

        cache_path = self._cache_path()
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {
            "valid_files": np.asarray(self.valid_files, dtype=str),
            "sample_file_indices": self.sample_file_indices,
            "sample_row_indices": self.sample_row_indices,
            "sample_lengths": self.sample_lengths,
            "column_slices": np.asarray(
                json.dumps(
                    {
                        "time": int(self._column_slices["time"]),
                        "target": self._column_slices["target"].tolist(),
                        "bes": self._column_slices["bes"].tolist(),
                        "ecei": self._column_slices["ecei"].tolist(),
                        "mc": self._column_slices["mc"].tolist(),
                    }
                )
            ),
        }
        for i, values in enumerate(self.file_arrays):
            arrays[f"file_{i}"] = values
        np.savez(cache_path, **arrays)
        print(f"Saved dataset cache: {cache_path}")

    def set_normalization_stats(self, normalization_stats):
        self.normalization_stats = normalization_stats

    def _required_columns(self):
        all_feature_cols = [*self.bes_cols, *self.ecei_cols, *self.mc_cols]
        return [TIME_COLUMN, *TARGET_COLUMNS, *all_feature_cols]

    @staticmethod
    def _channel_stats(arrays):
        values = np.concatenate(arrays, axis=0)
        mean = values.mean(axis=0, dtype=np.float64).astype(np.float32)
        std = values.std(axis=0, dtype=np.float64).astype(np.float32)
        std = np.where(std == 0.0, 1.0, std)
        std = np.maximum(std, 1e-6).astype(np.float32)
        return {"mean": mean, "std": std}

    def fit_normalization_stats(self, file_paths=None):
        selected_files = {str(Path(p)) for p in file_paths} if file_paths is not None else None
        groups = {"bes": [], "ecei": [], "mc": [], "target": []}

        for file_path, values in self.data_cache.items():
            if selected_files is not None and file_path not in selected_files:
                continue
            groups["bes"].append(values[:, self._column_slices["bes"]])
            groups["ecei"].append(values[:, self._column_slices["ecei"]])
            groups["mc"].append(values[:, self._column_slices["mc"]])
            groups["target"].append(values[:, self._column_slices["target"]])

        if not groups["target"]:
            raise ValueError("No rows available to fit normalization statistics")

        return {
            group: self._channel_stats(arrays)
            for group, arrays in groups.items()
        }

    def _normalize_array(self, values, group):
        if self.normalization_stats is None:
            return values

        stats = self.normalization_stats[group]
        mean = np.asarray(stats["mean"], dtype=np.float32)
        std = np.asarray(stats["std"], dtype=np.float32)
        return (values - mean) / std

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
        sample_file_indices = []
        sample_rows = []
        sample_lengths = []

        for file_idx, values in enumerate(self.file_arrays):
            if len(values) < self.min_subset_size:
                continue

            time = values[:, self._column_slices["time"]]
            deltas = np.diff(time)
            is_contiguous = deltas < 0.5

            block_start = 0
            for i, contiguous in enumerate(is_contiguous):
                if contiguous:
                    continue
                self._add_block_samples(sample_file_indices, sample_rows, sample_lengths, file_idx, block_start, i)
                block_start = i + 1
            self._add_block_samples(sample_file_indices, sample_rows, sample_lengths, file_idx, block_start, len(values) - 1)

        self.sample_file_indices = np.asarray(sample_file_indices, dtype=np.int32)
        self.sample_row_indices = np.asarray(sample_rows, dtype=np.int32).reshape(-1, self.window_size)
        self.sample_lengths = np.asarray(sample_lengths, dtype=np.int16)

    def _add_sample(self, sample_file_indices, sample_rows, sample_lengths, file_idx, rows):
        padded_rows = np.full((self.window_size,), -1, dtype=np.int32)
        padded_rows[: len(rows)] = rows
        sample_file_indices.append(file_idx)
        sample_rows.append(padded_rows)
        sample_lengths.append(len(rows))

    def _add_block_samples(self, sample_file_indices, sample_rows, sample_lengths, file_idx, block_start, block_end):
        block_len = block_end - block_start + 1
        if block_len < self.min_subset_size:
            return

        if not self.temporal_subset_augmentation:
            if block_len < self.window_size:
                return
            for idx in range(block_start + self.window_size - 1, block_end + 1):
                rows = tuple(range(idx - self.window_size + 1, idx + 1))
                self._add_sample(sample_file_indices, sample_rows, sample_lengths, file_idx, rows)
            return

        for target_idx in range(block_start + self.min_subset_size - 1, block_end + 1):
            context_start = max(block_start, target_idx - self.window_size + 1)
            history_rows = range(context_start, target_idx)
            max_history = min(self.window_size - 1, target_idx - context_start)
            min_history = self.min_subset_size - 1

            for history_count in range(min_history, max_history + 1):
                for history_combo in combinations(history_rows, history_count):
                    rows = (*history_combo, target_idx)
                    self._add_sample(sample_file_indices, sample_rows, sample_lengths, file_idx, rows)

    def _get_file_data(self, file_path):
        """Retrieve data from memory cache."""
        return self.data_cache[str(file_path)]

    def __len__(self):
        return len(self.sample_file_indices)

    @property
    def sample_indices(self):
        return [
            (self.valid_files[int(file_idx)], tuple(row for row in rows if row >= 0))
            for file_idx, rows in zip(self.sample_file_indices, self.sample_row_indices)
        ]

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

    def _window_tensor(self, file_values, row_indices, columns, group):
        values = file_values[np.ix_(row_indices, columns)].astype(np.float32, copy=False)
        values = self._normalize_array(values, group)
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

    def _ces_history(self, file_values, row_indices):
        values = file_values[np.ix_(row_indices, self._column_slices["target"])].astype(np.float32, copy=True)
        values = self._normalize_array(values, "target")
        observed = np.ones((len(row_indices), 1), dtype=np.float32)

        values[-1, :] = 0.0
        observed[-1, 0] = 0.0

        history = np.concatenate((values, observed), axis=1)
        return self._pad_tensor(torch.from_numpy(history))

    def __getitem__(self, i):
        file_idx = int(self.sample_file_indices[i])
        file_path = self.valid_files[file_idx]
        file_values = self.file_arrays[file_idx]
        valid_len = int(self.sample_lengths[i])
        row_indices = self.sample_row_indices[i, :valid_len]

        bes = self._window_tensor(file_values, row_indices, self._column_slices["bes"], "bes")
        ecei = self._window_tensor(file_values, row_indices, self._column_slices["ecei"], "ecei")
        mc = self._window_tensor(file_values, row_indices, self._column_slices["mc"], "mc")

        time_values = file_values[row_indices, self._column_slices["time"]]
        time_features = self._time_features(time_values)
        ces_history = self._ces_history(file_values, row_indices)
        input_mask = torch.zeros(self.window_size, dtype=torch.bool)
        input_mask[:valid_len] = True
        padded_row_indices = torch.from_numpy(self.sample_row_indices[i].astype(np.int64, copy=True))

        idx = row_indices[-1]
        target_values = file_values[idx, self._column_slices["target"]].astype(np.float32, copy=False)
        target_values = self._normalize_array(target_values, "target")
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


DEFAULT_TRAIN_SAMPLE_COUNT = 50000
DEFAULT_VAL_SAMPLE_COUNT = 10000
DEFAULT_SAMPLE_SEED = 42


def select_seeded_random_indices(indices, max_samples, seed):
    if max_samples <= 0 or len(indices) <= max_samples:
        return list(indices)

    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(indices), generator=generator)[:max_samples].tolist()
    return [indices[i] for i in order]


if __name__ == "__main__":
    data_dir = Path(__file__).resolve().parents[1] / "data"
    dataset = KSTAR_CES_Dataset(data_dir=data_dir, window_size=10)
    print(f"CSV files: {len(dataset.files)}")
    print(f"Total samples: {len(dataset)}")
    print(f"Feature dims: {dataset.feature_dims}")
    train_preview = select_seeded_random_indices(
        range(len(dataset)),
        DEFAULT_TRAIN_SAMPLE_COUNT,
        DEFAULT_SAMPLE_SEED + 101,
    )
    train_preview_set = set(train_preview)
    remaining = [i for i in range(len(dataset)) if i not in train_preview_set]
    test_preview = select_seeded_random_indices(
        remaining,
        DEFAULT_VAL_SAMPLE_COUNT,
        DEFAULT_SAMPLE_SEED + 202,
    )
    print(f"Random train samples: {len(train_preview)}")
    print(f"Random test samples: {len(test_preview)}")

    if len(dataset) > 0:
        sample = dataset[0]
        print("BES shape:", tuple(sample["bes"].shape))
        print("ECEI shape:", tuple(sample["ecei"].shape))
        print("MC shape:", tuple(sample["mc"].shape))
        print("Time feature shape:", tuple(sample["time_features"].shape))
        print("Target shape:", tuple(sample["target"].shape))
