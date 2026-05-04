import json
import os
from pathlib import Path

from dataset import DEFAULT_TRAIN_SAMPLE_COUNT, DEFAULT_VAL_SAMPLE_COUNT, KSTAR_CES_Dataset
from train import default_split_dir, load_or_create_fixed_splits, split_manifest


def main():
    root_dir = Path(__file__).resolve().parents[1]
    data_dir = root_dir / "data"
    split_dir = default_split_dir(root_dir)

    seed = int(os.getenv("CES_SEED", "42"))
    val_fraction = float(os.getenv("CES_VAL_FRACTION", "0.2"))
    window_size = int(os.getenv("CES_WINDOW_SIZE", "4"))
    max_train_samples = int(os.getenv("CES_MAX_TRAIN_SAMPLES", str(DEFAULT_TRAIN_SAMPLE_COUNT)))
    max_val_samples = int(os.getenv("CES_MAX_VAL_SAMPLES", str(DEFAULT_VAL_SAMPLE_COUNT)))
    temporal_subset_augmentation = os.getenv("CES_TEMPORAL_SUBSETS", "1") == "1"
    min_subset_size = int(os.getenv("CES_MIN_SUBSET_SIZE", "2"))

    dataset = KSTAR_CES_Dataset(
        data_dir=data_dir,
        window_size=window_size,
        temporal_subset_augmentation=temporal_subset_augmentation,
        min_subset_size=min_subset_size,
    )
    train_indices, val_indices, train_files, val_files = load_or_create_fixed_splits(
        dataset,
        split_dir,
        val_fraction,
        seed,
        max_train_samples,
        max_val_samples,
    )

    manifest = split_manifest(
        train_files,
        val_files,
        train_indices,
        val_indices,
        seed,
        val_fraction,
    )
    split_path = split_dir / "split_manifest.json"
    with split_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Saved split manifest: {split_path}")
    print(
        f"train_shots={manifest['train_file_count']}, "
        f"val_shots={manifest['val_file_count']}, "
        f"train_samples={manifest['train_sample_count']}, "
        f"val_samples={manifest['val_sample_count']}, "
        f"seed={seed}"
    )
    print("Validation shots preview:")
    for name in manifest["val_files"][:25]:
        print(f"- {name}")


if __name__ == "__main__":
    main()
