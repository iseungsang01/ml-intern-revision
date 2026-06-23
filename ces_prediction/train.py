import csv
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

from dataset import (
    DEFAULT_TRAIN_SAMPLE_COUNT,
    DEFAULT_VAL_SAMPLE_COUNT,
    KSTAR_CES_Dataset,
    select_seeded_random_indices,
)
from model import MultimodalCESPredictor


FIXED_TRAIN_SPLIT_NAME = "fixed_train_split.csv"
FIXED_VAL_SPLIT_NAME = "fixed_val_split.csv"
FIXED_TEST_SPLIT_NAME = "fixed_test_split.csv"


def default_split_dir(root_dir):
    override = os.getenv("CES_SPLIT_DIR")
    if override:
        return Path(override)
    return root_dir / "data" / "splits"


def resolve_cpu_config():
    available = os.cpu_count() or 1
    requested = int(os.getenv("CES_CPU_WORKERS", str(available)))
    cpu_workers = max(1, min(requested, available))

    # Windows uses spawn-based multiprocessing, which has high startup and copy
    # cost for large in-memory datasets. Keep the default single-process there
    # unless the user explicitly opts into DataLoader workers.
    if os.name == "nt":
        default_loader_workers = 0
    else:
        default_loader_workers = 0 if cpu_workers <= 2 else max(1, min(cpu_workers // 2, cpu_workers - 1))
    loader_workers = int(os.getenv("CES_DATALOADER_WORKERS", str(default_loader_workers)))
    loader_workers = max(0, min(loader_workers, max(cpu_workers - 1, 0)))

    torch_threads = int(os.getenv("CES_TORCH_THREADS", str(max(1, cpu_workers - loader_workers))))
    # Cap torch threads at 16 to avoid synchronization overhead on small architectures
    torch_threads = max(1, min(torch_threads, 16, cpu_workers))

    interop_threads = int(os.getenv("CES_TORCH_INTEROP_THREADS", str(min(4, torch_threads))))
    interop_threads = max(1, min(interop_threads, torch_threads))

    return {
        "available": available,
        "cpu_workers": cpu_workers,
        "dataloader_workers": loader_workers,
        "torch_threads": torch_threads,
        "torch_interop_threads": interop_threads,
    }


def split_indices_by_file(dataset, val_fraction=0.2, seed=42):
    """Split by CSV shot file to avoid train/validation leakage across rows."""

    if hasattr(dataset, "sample_file_indices") and hasattr(dataset, "valid_files"):
        sample_file_indices = dataset.sample_file_indices
        files = sorted(np.unique(sample_file_indices).astype(int).tolist())
        file_name = lambda file_id: dataset.valid_files[int(file_id)]
    else:
        files = sorted({file_path for file_path, _ in dataset.sample_indices})
        file_name = str

    if len(files) < 2:
        raise ValueError("Need at least two CSV files with valid samples for validation split")

    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(files), generator=generator).tolist()
    val_count = max(1, int(round(len(files) * val_fraction)))
    val_files = {files[i] for i in order[:val_count]}

    if hasattr(dataset, "sample_file_indices"):
        val_mask = torch.from_numpy(np.isin(dataset.sample_file_indices, list(val_files)))
        all_indices = torch.arange(len(dataset), dtype=torch.long)
        val_indices = all_indices[val_mask].tolist()
        train_indices = all_indices[~val_mask].tolist()
    else:
        train_indices = []
        val_indices = []
        for sample_idx, (file_path, _) in enumerate(dataset.sample_indices):
            if file_path in val_files:
                val_indices.append(sample_idx)
            else:
                train_indices.append(sample_idx)

    if not train_indices or not val_indices:
        raise ValueError("File-level split produced an empty train or validation set")

    train_files = sorted(file_name(file_id) for file_id in files if file_id not in val_files)
    val_files_resolved = sorted(file_name(file_id) for file_id in val_files)

    return train_indices, val_indices, train_files, val_files_resolved


def select_seeded_subset(indices, max_samples, seed):
    return select_seeded_random_indices(indices, max_samples, seed)


def _sample_split_row(dataset, sample_idx):
    file_idx = int(dataset.sample_file_indices[sample_idx])
    file_name = Path(dataset.valid_files[file_idx]).name
    valid_len = int(dataset.sample_lengths[sample_idx])
    row_indices = dataset.sample_row_indices[sample_idx, :valid_len].astype(int).tolist()
    return {
        "sample_index": int(sample_idx),
        "file": file_name,
        "row_index": int(row_indices[-1]),
        "row_indices": json.dumps(row_indices, separators=(",", ":")),
    }


def write_fixed_split_csv(path, dataset, indices):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["sample_index", "file", "row_index", "row_indices"])
        writer.writeheader()
        for sample_idx in indices:
            writer.writerow(_sample_split_row(dataset, sample_idx))


def load_fixed_split_csv(path, dataset):
    indices = []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sample_idx = int(row["sample_index"])
            if sample_idx < 0 or sample_idx >= len(dataset):
                raise ValueError(f"{path} has out-of-range sample_index={sample_idx}")

            expected = _sample_split_row(dataset, sample_idx)
            row_indices = json.loads(row["row_indices"])
            if (
                row["file"] != expected["file"]
                or int(row["row_index"]) != expected["row_index"]
                or row_indices != json.loads(expected["row_indices"])
            ):
                raise ValueError(
                    f"{path} no longer matches the current dataset at sample_index={sample_idx}. "
                    "Delete fixed split CSV files to regenerate them."
                )
            indices.append(sample_idx)

    if not indices:
        raise ValueError(f"{path} is empty")
    return indices


def split_files_from_indices(dataset, indices):
    file_ids = sorted({int(dataset.sample_file_indices[i]) for i in indices})
    return [dataset.valid_files[file_id] for file_id in file_ids]


def load_or_create_fixed_splits(dataset, split_dir, val_fraction, seed, max_train_samples, max_val_samples):
    split_dir.mkdir(parents=True, exist_ok=True)
    train_split_path = split_dir / FIXED_TRAIN_SPLIT_NAME
    val_split_path = split_dir / FIXED_VAL_SPLIT_NAME

    if train_split_path.exists() and val_split_path.exists():
        train_indices = load_fixed_split_csv(train_split_path, dataset)
        val_indices = load_fixed_split_csv(val_split_path, dataset)
        requested_train_count = max_train_samples if max_train_samples > 0 else len(train_indices)
        requested_val_count = max_val_samples if max_val_samples > 0 else len(val_indices)
        if len(train_indices) == requested_train_count and len(val_indices) == requested_val_count:
            train_files = split_files_from_indices(dataset, train_indices)
            val_files = split_files_from_indices(dataset, val_indices)
            print(f"Loaded fixed splits: {train_split_path.name}, {val_split_path.name}")
            return train_indices, val_indices, train_files, val_files

        print(
            "Fixed split size changed "
            f"from train={len(train_indices)}, val={len(val_indices)} "
            f"to train={max_train_samples}, val={max_val_samples}; regenerating CSV files."
        )
        train_split_path.unlink()
        val_split_path.unlink()

    if train_split_path.exists() != val_split_path.exists():
        raise ValueError(
            f"Expected both {FIXED_TRAIN_SPLIT_NAME} and {FIXED_VAL_SPLIT_NAME}, "
            "but only one exists. Delete the existing fixed split CSV or restore the missing one."
        )

    train_indices, val_indices, train_files, val_files = split_indices_by_file(
        dataset, val_fraction=val_fraction, seed=seed
    )
    train_indices = select_seeded_subset(train_indices, max_train_samples, seed + 101)
    val_indices = select_seeded_subset(val_indices, max_val_samples, seed + 202)
    write_fixed_split_csv(train_split_path, dataset, train_indices)
    write_fixed_split_csv(val_split_path, dataset, val_indices)
    train_files = split_files_from_indices(dataset, train_indices)
    val_files = split_files_from_indices(dataset, val_indices)
    print(f"Created fixed splits: {train_split_path.name}, {val_split_path.name}")
    return train_indices, val_indices, train_files, val_files


def split_indices_3way(dataset, val_fraction, test_fraction, seed):
    """File-level train/val/test split. Test files are carved FIRST, then val,
    then train, from one seeded file ordering -- so a fixed held-out test set is
    reserved before any model selection (e.g. AutoML) ever sees it."""
    sample_file_indices = dataset.sample_file_indices
    files = sorted(np.unique(sample_file_indices).astype(int).tolist())
    if len(files) < 3:
        raise ValueError("Need at least three CSV files for a train/val/test split")
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(files), generator=generator).tolist()
    n = len(files)
    test_count = max(1, int(round(n * test_fraction)))
    val_count = max(1, int(round(n * val_fraction)))
    if test_count + val_count >= n:
        raise ValueError("test_fraction + val_fraction too large; no training files left")
    test_fids = [files[i] for i in order[:test_count]]
    val_fids = [files[i] for i in order[test_count:test_count + val_count]]
    train_fids = [files[i] for i in order[test_count + val_count:]]

    all_indices = np.arange(len(dataset))

    def idx_for(fids):
        return all_indices[np.isin(sample_file_indices, fids)].tolist()

    return idx_for(train_fids), idx_for(val_fids), idx_for(test_fids)


def load_or_create_3way_splits(
    dataset, split_dir, val_fraction, test_fraction, seed,
    max_train_samples, max_val_samples, max_test_samples,
):
    split_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "train": split_dir / FIXED_TRAIN_SPLIT_NAME,
        "val": split_dir / FIXED_VAL_SPLIT_NAME,
        "test": split_dir / FIXED_TEST_SPLIT_NAME,
    }
    caps = {"train": max_train_samples, "val": max_val_samples, "test": max_test_samples}
    if all(p.exists() for p in paths.values()):
        loaded = {k: load_fixed_split_csv(p, dataset) for k, p in paths.items()}
        if all(len(loaded[k]) == (caps[k] if caps[k] > 0 else len(loaded[k])) for k in paths):
            print(f"Loaded fixed 3-way splits from {split_dir}")
            return (
                loaded["train"], loaded["val"], loaded["test"],
                split_files_from_indices(dataset, loaded["train"]),
                split_files_from_indices(dataset, loaded["val"]),
                split_files_from_indices(dataset, loaded["test"]),
            )
        for p in paths.values():
            p.unlink()

    tr, va, te = split_indices_3way(dataset, val_fraction, test_fraction, seed)
    tr = select_seeded_subset(tr, max_train_samples, seed + 101)
    va = select_seeded_subset(va, max_val_samples, seed + 202)
    te = select_seeded_subset(te, max_test_samples, seed + 303)
    write_fixed_split_csv(paths["train"], dataset, tr)
    write_fixed_split_csv(paths["val"], dataset, va)
    write_fixed_split_csv(paths["test"], dataset, te)
    print(f"Created fixed 3-way splits in {split_dir}")
    return (
        tr, va, te,
        split_files_from_indices(dataset, tr),
        split_files_from_indices(dataset, va),
        split_files_from_indices(dataset, te),
    )


def split_manifest(train_files, val_files, train_indices, val_indices, seed, val_fraction,
                   test_files=None, test_indices=None):
    manifest = {
        "seed": seed,
        "val_fraction": val_fraction,
        "split_unit": "csv_shot_file",
        "sample_cap_method": "seeded_random_without_replacement_after_file_split",
        "train_file_count": len(train_files),
        "val_file_count": len(val_files),
        "train_sample_count": len(train_indices),
        "val_sample_count": len(val_indices),
        "train_files": [Path(path).name for path in train_files],
        "val_files": [Path(path).name for path in val_files],
    }
    if test_files is not None:
        manifest["test_file_count"] = len(test_files)
        manifest["test_sample_count"] = len(test_indices or [])
        manifest["test_files"] = [Path(path).name for path in test_files]
    return manifest


def normalization_stats_to_jsonable(stats):
    return {
        group: {
            key: value.tolist()
            for key, value in group_stats.items()
        }
        for group, group_stats in stats.items()
    }


VALID_ABLATIONS = ("none", "no_history", "no_fast")


def apply_ablation(ablate, bes, ecei, mc, ces_history):
    """Zero a modality group for controlled ablation (CES_ABLATE).

    ``no_history`` zeroes the previous-CES history -> tests whether the fast
    diagnostics carry information beyond reusing past CES. ``no_fast`` zeroes the
    BES/ECEI/MC sensor values -> history-only. The persistence baseline in
    evaluate.py is always computed from the *real* history, independent of this.
    """
    if ablate == "no_history":
        ces_history = torch.zeros_like(ces_history)
    elif ablate == "no_fast":
        bes = torch.zeros_like(bes)
        ecei = torch.zeros_like(ecei)
        mc = torch.zeros_like(mc)
    return bes, ecei, mc, ces_history


def train():
    root_dir = Path(__file__).resolve().parents[1]
    output_dir = Path(os.getenv("CES_OUTPUT_DIR", Path(__file__).resolve().parent))
    output_dir.mkdir(parents=True, exist_ok=True)

    data_dir = Path(os.getenv("CES_DATA_DIR", root_dir / "data"))
    split_dir = default_split_dir(root_dir)
    window_size = int(os.getenv("CES_WINDOW_SIZE", "4"))
    batch_size = int(os.getenv("CES_BATCH_SIZE", "512"))
    epochs = int(os.getenv("CES_EPOCHS", "10"))
    lr = float(os.getenv("CES_LR", "1e-3"))
    seed = int(os.getenv("CES_SEED", "42"))
    val_fraction = float(os.getenv("CES_VAL_FRACTION", "0.2"))
    max_train_samples = int(os.getenv("CES_MAX_TRAIN_SAMPLES", str(DEFAULT_TRAIN_SAMPLE_COUNT)))
    max_val_samples = int(os.getenv("CES_MAX_VAL_SAMPLES", str(DEFAULT_VAL_SAMPLE_COUNT)))
    temporal_subset_augmentation = os.getenv("CES_TEMPORAL_SUBSETS", "1") == "1"
    min_subset_size = int(os.getenv("CES_MIN_SUBSET_SIZE", "2"))
    test_fraction = float(os.getenv("CES_TEST_FRACTION", "0.0"))
    max_test_samples = int(os.getenv("CES_MAX_TEST_SAMPLES", str(max_val_samples)))
    init_seed = int(os.getenv("CES_INIT_SEED", str(seed)))
    ablate = os.getenv("CES_ABLATE", "none")
    if ablate not in VALID_ABLATIONS:
        raise ValueError(f"CES_ABLATE must be one of {VALID_ABLATIONS}, got {ablate!r}")
    cpu_config = resolve_cpu_config()

    if epochs < 1:
        raise ValueError("CES_EPOCHS must be at least 1")

    torch.manual_seed(init_seed)  # decoupled from split seed (CES_INIT_SEED) for init-stability seeds
    torch.set_num_threads(cpu_config["torch_threads"])
    torch.set_num_interop_threads(cpu_config["torch_interop_threads"])

    print("Initializing dataset from CSV files...")
    full_dataset = KSTAR_CES_Dataset(
        data_dir=data_dir,
        window_size=window_size,
        temporal_subset_augmentation=temporal_subset_augmentation,
        min_subset_size=min_subset_size,
    )
    if len(full_dataset) == 0:
        print("Error: No valid data found.")
        return

    if test_fraction > 0.0:
        (
            train_indices, val_indices, test_indices,
            train_split_files, val_split_files, test_split_files,
        ) = load_or_create_3way_splits(
            full_dataset, split_dir, val_fraction, test_fraction, seed,
            max_train_samples, max_val_samples, max_test_samples,
        )
    else:
        train_indices, val_indices, train_split_files, val_split_files = load_or_create_fixed_splits(
            full_dataset,
            split_dir,
            val_fraction,
            seed,
            max_train_samples,
            max_val_samples,
        )
        test_indices, test_split_files = [], []

    manifest = split_manifest(
        train_split_files,
        val_split_files,
        train_indices,
        val_indices,
        seed,
        val_fraction,
        test_files=test_split_files if test_fraction > 0.0 else None,
        test_indices=test_indices,
    )
    split_path = split_dir / "split_manifest.json"
    with split_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(
        f"File split: train_shots={manifest['train_file_count']}, "
        f"val_shots={manifest['val_file_count']}, seed={seed}"
    )
    if test_fraction > 0.0:
        print(f"Held-out test shots: {manifest['test_file_count']} (never used in training/selection)")
    print(f"Validation shots preview: {', '.join(manifest['val_files'][:10])}")

    if hasattr(full_dataset, "sample_file_indices") and hasattr(full_dataset, "valid_files"):
        train_files = set(train_split_files)
    else:
        train_files = {full_dataset.sample_indices[i][0] for i in train_indices}
    normalization_stats = full_dataset.fit_normalization_stats(train_files)
    full_dataset.set_normalization_stats(normalization_stats)

    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        # Fixed input shapes -> let cuDNN pick the fastest kernels.
        torch.backends.cudnn.benchmark = True

    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": cpu_config["dataloader_workers"],
        "persistent_workers": cpu_config["dataloader_workers"] > 0,
        "pin_memory": device.type == "cuda",
    }
    if cpu_config["dataloader_workers"] > 0:
        loader_kwargs["prefetch_factor"] = 2

    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)

    print(f"Using device: {device}")
    print(
        "CPU config: "
        f"available={cpu_config['available']}, "
        f"budget={cpu_config['cpu_workers']}, "
        f"torch_threads={cpu_config['torch_threads']}, "
        f"interop_threads={cpu_config['torch_interop_threads']}, "
        f"dataloader_workers={cpu_config['dataloader_workers']}"
    )
    print(f"Feature dims: {full_dataset.feature_dims}")
    print(f"Samples: train={len(train_dataset)}, val={len(val_dataset)}")
    if ablate != "none":
        print(f"Ablation: {ablate} (model inputs zeroed; persistence baseline unaffected)")

    model = MultimodalCESPredictor.from_dataset(full_dataset, window_size=window_size).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=2, factor=0.5)
    target_mean = torch.as_tensor(normalization_stats["target"]["mean"], device=device)
    target_std = torch.as_tensor(normalization_stats["target"]["std"], device=device)
    zero_ti_normalized = (0.0 - target_mean[0]) / target_std[0]

    use_amp = device.type == "cuda" and os.getenv("CES_AMP", "0") == "1"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if use_amp else None
    if use_amp:
        print("Mixed-precision (AMP) enabled on CUDA")

    train_loss = float("inf")
    val_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        train_se_sum = 0.0
        train_label_count = 0.0

        for batch in train_loader:
            bes = batch["bes"].to(device, non_blocking=True)
            ecei = batch["ecei"].to(device, non_blocking=True)
            mc = batch["mc"].to(device, non_blocking=True)
            time_features = batch["time_features"].to(device, non_blocking=True)
            ces_history = batch["ces_history"].to(device, non_blocking=True)
            targets = batch["target"].to(device, non_blocking=True)
            target_mask = batch["target_mask"].to(device, non_blocking=True)
            bes, ecei, mc, ces_history = apply_ablation(ablate, bes, ecei, mc, ces_history)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                outputs = model(bes, ecei, mc, time_features, ces_history)

                # Per-target masked MSE: CES_TI and CES_VT are missing independently,
                # so only observed targets contribute to the loss.
                squared_error = (outputs - targets) ** 2 * target_mask
                observed_count = target_mask.sum().clamp(min=1.0)
                loss_mse = squared_error.sum() / observed_count
                penalty_neg_ti = torch.relu(zero_ti_normalized - outputs[:, 0]).mean()
                loss = loss_mse + 0.1 * penalty_neg_ti
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite train loss at epoch={epoch + 1}: "
                    f"loss={loss.item()}, mse={loss_mse.item()}, penalty_neg_ti={penalty_neg_ti.item()}"
                )

            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

            train_se_sum += squared_error.sum().item()
            train_label_count += target_mask.sum().item()

        train_loss = train_se_sum / max(train_label_count, 1.0)

        model.eval()
        val_se_sum = 0.0
        val_label_count = 0.0
        with torch.no_grad():
            for batch in val_loader:
                bes = batch["bes"].to(device, non_blocking=True)
                ecei = batch["ecei"].to(device, non_blocking=True)
                mc = batch["mc"].to(device, non_blocking=True)
                time_features = batch["time_features"].to(device, non_blocking=True)
                ces_history = batch["ces_history"].to(device, non_blocking=True)
                targets = batch["target"].to(device, non_blocking=True)
                target_mask = batch["target_mask"].to(device, non_blocking=True)
                bes, ecei, mc, ces_history = apply_ablation(ablate, bes, ecei, mc, ces_history)

                outputs = model(bes, ecei, mc, time_features, ces_history)
                squared_error = (outputs - targets) ** 2 * target_mask
                if not torch.isfinite(squared_error.sum()):
                    raise FloatingPointError(
                        f"Non-finite validation loss at epoch={epoch + 1}"
                    )
                val_se_sum += squared_error.sum().item()
                val_label_count += target_mask.sum().item()

        val_loss = val_se_sum / max(val_label_count, 1.0)
        if not np.isfinite(val_loss):
            raise FloatingPointError(f"Non-finite validation loss after epoch={epoch + 1}: {val_loss}")
        scheduler.step(val_loss)

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}"
        )

    weights_dir = output_dir / "weights"
    weights_dir.mkdir(exist_ok=True)
    weight_path = weights_dir / "multimodal_ces.pth"
    torch.save(model.state_dict(), weight_path)
    print(f"Training complete. Model saved to {weight_path}")

    metrics = {
        "final_train_loss": train_loss,
        "final_val_loss": val_loss,
        "epochs": epochs,
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "feature_dims": full_dataset.feature_dims,
        "temporal_subset_augmentation": temporal_subset_augmentation,
        "min_subset_size": min_subset_size,
        "label_handling": "per_target_masked_multitask",
        "loss": "masked_mse_per_target",
        "ablation": ablate,
        "split": manifest,
        "normalization": {
            "scope": "train_files_only",
            "method": "per_channel_zscore",
            "stats": normalization_stats_to_jsonable(normalization_stats),
        },
        "cpu_config": cpu_config,
    }
    metrics_path = output_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {metrics_path}")


if __name__ == "__main__":
    train()
