import json
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

from dataset import KSTAR_CES_Dataset
from model import MultimodalCESPredictor

try:
    import wandb
except ImportError:
    wandb = None


def resolve_cpu_config():
    available = os.cpu_count() or 1
    requested = int(os.getenv("CES_CPU_WORKERS", str(available)))
    cpu_workers = max(1, min(requested, available))

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

    files = sorted({file_path for file_path, _ in dataset.sample_indices})
    if len(files) < 2:
        raise ValueError("Need at least two CSV files with valid samples for validation split")

    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(files), generator=generator).tolist()
    val_count = max(1, int(round(len(files) * val_fraction)))
    val_files = {files[i] for i in order[:val_count]}

    train_indices = []
    val_indices = []
    for sample_idx, (file_path, _) in enumerate(dataset.sample_indices):
        if file_path in val_files:
            val_indices.append(sample_idx)
        else:
            train_indices.append(sample_idx)

    if not train_indices or not val_indices:
        raise ValueError("File-level split produced an empty train or validation set")

    return train_indices, val_indices


def normalization_stats_to_jsonable(stats):
    return {
        group: {
            key: value.tolist()
            for key, value in group_stats.items()
        }
        for group, group_stats in stats.items()
    }


def train():
    root_dir = Path(__file__).resolve().parents[1]
    output_dir = Path(__file__).resolve().parent

    data_dir = root_dir / "data"
    window_size = int(os.getenv("CES_WINDOW_SIZE", "4"))
    batch_size = int(os.getenv("CES_BATCH_SIZE", "512"))
    epochs = int(os.getenv("CES_EPOCHS", "10"))
    lr = float(os.getenv("CES_LR", "1e-3"))
    seed = int(os.getenv("CES_SEED", "42"))
    val_fraction = float(os.getenv("CES_VAL_FRACTION", "0.2"))
    max_train_samples = int(os.getenv("CES_MAX_TRAIN_SAMPLES", "0"))
    max_val_samples = int(os.getenv("CES_MAX_VAL_SAMPLES", "0"))
    temporal_subset_augmentation = os.getenv("CES_TEMPORAL_SUBSETS", "1") == "1"
    min_subset_size = int(os.getenv("CES_MIN_SUBSET_SIZE", "2"))
    cpu_config = resolve_cpu_config()

    torch.manual_seed(seed)
    torch.set_num_threads(cpu_config["torch_threads"])
    torch.set_num_interop_threads(cpu_config["torch_interop_threads"])

    if wandb is not None:
        wandb.init(
            project="kstar-ces-prediction",
            mode="disabled",
            config={
                "window_size": window_size,
                "batch_size": batch_size,
                "epochs": epochs,
                "lr": lr,
                "time_features": "lookback/delta seconds + log1p variants",
                "ces_history": "previous selected CES_TI/CES_VT plus observed mask",
                "temporal_subset_augmentation": temporal_subset_augmentation,
                "min_subset_size": min_subset_size,
                "split": "by_csv_file",
                "val_fraction": val_fraction,
                "cpu_config": cpu_config,
            },
        )

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

    train_indices, val_indices = split_indices_by_file(
        full_dataset, val_fraction=val_fraction, seed=seed
    )
    if max_train_samples > 0:
        train_indices = train_indices[:max_train_samples]
    if max_val_samples > 0:
        val_indices = val_indices[:max_val_samples]

    train_files = {full_dataset.sample_indices[i][0] for i in train_indices}
    normalization_stats = full_dataset.fit_normalization_stats(train_files)
    full_dataset.set_normalization_stats(normalization_stats)

    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)

    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": cpu_config["dataloader_workers"],
        "persistent_workers": cpu_config["dataloader_workers"] > 0,
    }
    if cpu_config["dataloader_workers"] > 0:
        loader_kwargs["prefetch_factor"] = 2

    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

    model = MultimodalCESPredictor.from_dataset(full_dataset, window_size=window_size).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=2, factor=0.5)
    target_mean = torch.as_tensor(normalization_stats["target"]["mean"], device=device)
    target_std = torch.as_tensor(normalization_stats["target"]["std"], device=device)
    zero_ti_normalized = (0.0 - target_mean[0]) / target_std[0]

    train_loss = float("inf")
    val_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        train_loss_sum = 0.0

        for batch in train_loader:
            bes = batch["bes"].to(device)
            ecei = batch["ecei"].to(device)
            mc = batch["mc"].to(device)
            time_features = batch["time_features"].to(device)
            ces_history = batch["ces_history"].to(device)
            targets = batch["target"].to(device)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(bes, ecei, mc, time_features, ces_history)

            loss_mse = criterion(outputs, targets)
            penalty_neg_ti = torch.relu(zero_ti_normalized - outputs[:, 0]).mean()
            loss = loss_mse + 0.1 * penalty_neg_ti

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            train_loss_sum += loss.item() * bes.size(0)

        train_loss = train_loss_sum / len(train_loader.dataset)

        model.eval()
        val_loss_sum = 0.0
        with torch.no_grad():
            for batch in val_loader:
                bes = batch["bes"].to(device)
                ecei = batch["ecei"].to(device)
                mc = batch["mc"].to(device)
                time_features = batch["time_features"].to(device)
                ces_history = batch["ces_history"].to(device)
                targets = batch["target"].to(device)

                outputs = model(bes, ecei, mc, time_features, ces_history)
                loss = criterion(outputs, targets)
                val_loss_sum += loss.item() * bes.size(0)

        val_loss = val_loss_sum / len(val_loader.dataset)
        scheduler.step(val_loss)

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}"
        )

        if wandb is not None:
            wandb.log({"train_loss": train_loss, "val_loss": val_loss, "epoch": epoch})

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

    if wandb is not None:
        wandb.finish()


if __name__ == "__main__":
    train()
