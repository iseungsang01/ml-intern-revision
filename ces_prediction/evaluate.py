"""Clean evaluation of a trained MultimodalCESPredictor.

This is the *scientific* evaluation, separate from the training loop:

- It builds a **non-augmented** dataset (``temporal_subset_augmentation=False``) so
  every validation target uses its **real most-recent history window**, not the
  counterfactual random subsets that temporal-subset augmentation produces. The
  training loop's val loss is measured on augmented data and is therefore not a
  clean generalization estimate; this script is.
- It restricts to the **validation shot files** recorded in ``split_manifest.json``
  (the same file-level split used for training), so there is no shot leakage.
- It reports errors in **denormalized (physical) CES units, per target**
  (``CES_TI`` and ``CES_VT`` separately), and compares the model against two
  trivial baselines:
    * **persistence** -- carry forward the most-recent *observed* CES value for that
      target within the window (the "just reuse the last measurement" baseline);
    * **mean** -- always predict the train-set mean (the absolute floor, == R^2 ref).

  The headline number is ``skill_vs_persistence = 1 - MSE_model / MSE_persistence``:
  positive means the multimodal model genuinely beats carrying the last CES forward.

Run (PowerShell), pointing at the same output/split dirs the training run used::

    $env:CES_DATA_DIR="C:\\Users\\lss\\Desktop\\원핵 졸논\\data"
    py ces_prediction/evaluate.py

Reads ``CES_OUTPUT_DIR/metrics.json`` (normalization stats), ``CES_OUTPUT_DIR/weights/
multimodal_ces.pth`` (model), and ``CES_SPLIT_DIR/split_manifest.json`` (val files).
Writes ``CES_OUTPUT_DIR/eval_metrics.json``.
"""

import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from dataset import KSTAR_CES_Dataset, select_seeded_random_indices
from model import MultimodalCESPredictor


TARGET_NAMES = ("CES_TI", "CES_VT")


def _load_stats(metrics):
    raw = metrics["normalization"]["stats"]
    return {
        group: {key: np.asarray(value, dtype=np.float32) for key, value in group_stats.items()}
        for group, group_stats in raw.items()
    }


def _persistence_from_history(ces_history):
    """Most-recent observed CES value per target, from the masked history window.

    ces_history is (batch, window, 4) = [TI_norm, VT_norm, TI_observed, VT_observed].
    The target timestep and padding rows have observed == 0, so the highest index
    with observed == 1 is the most-recent observation *before* the target.
    Returns (values (batch, 2) normalized, has_obs (batch, 2) bool).
    """
    values = ces_history[..., :2]
    observed = ces_history[..., 2:] > 0.5
    batch, window = observed.shape[0], observed.shape[1]

    positions = torch.arange(window, device=ces_history.device).view(1, window, 1)
    masked_positions = torch.where(observed, positions, torch.full_like(positions, -1))
    last_idx = masked_positions.max(dim=1).values  # (batch, 2), -1 if never observed
    has_obs = last_idx >= 0

    rows = torch.arange(batch, device=ces_history.device)
    out = torch.zeros(batch, 2, device=ces_history.device)
    for t in range(2):
        gathered = values[rows, last_idx[:, t].clamp(min=0), t]
        out[:, t] = torch.where(has_obs[:, t], gathered, torch.zeros_like(gathered))
    return out, has_obs


VALID_ABLATIONS = ("none", "no_history", "no_fast")


def apply_ablation(ablate, bes, ecei, mc, ces_history):
    """Zero a modality group for ablation (mirrors train.py). The persistence
    baseline is computed from the real history *before* this is applied."""
    if ablate == "no_history":
        ces_history = torch.zeros_like(ces_history)
    elif ablate == "no_fast":
        bes = torch.zeros_like(bes)
        ecei = torch.zeros_like(ecei)
        mc = torch.zeros_like(mc)
    return bes, ecei, mc, ces_history


def build_clean_val_subset(data_dir, window_size, stats, val_files, max_val_samples, seed):
    """Build the clean (non-augmented) dataset and its val sample indices.

    Single source of truth shared by evaluate.py and compare_baselines.py so the
    model is scored on byte-identical samples in both (prevents harness drift).
    `val_files` is the set of validation shot filenames from split_manifest.json.
    """
    dataset = KSTAR_CES_Dataset(
        data_dir=data_dir,
        window_size=window_size,
        temporal_subset_augmentation=False,
    )
    dataset.set_normalization_stats(stats)

    file_names = [Path(p).name for p in dataset.valid_files]
    val_file_ids = {i for i, name in enumerate(file_names) if name in val_files}
    if not val_file_ids:
        raise ValueError("None of the manifest val files are present in the dataset.")
    val_indices = [
        i for i in range(len(dataset))
        if int(dataset.sample_file_indices[i]) in val_file_ids
    ]
    if not val_indices:
        raise ValueError(
            "No clean (full-window) validation samples for the manifest val files. "
            "Try a smaller CES_WINDOW_SIZE."
        )
    val_indices = select_seeded_random_indices(val_indices, max_val_samples, seed + 202)
    return dataset, val_indices, val_file_ids


def evaluate():
    root_dir = Path(__file__).resolve().parents[1]
    data_dir = Path(os.getenv("CES_DATA_DIR", root_dir / "data"))
    output_dir = Path(os.getenv("CES_OUTPUT_DIR", Path(__file__).resolve().parent))
    split_dir = Path(os.getenv("CES_SPLIT_DIR", root_dir / "data" / "splits"))
    window_size = int(os.getenv("CES_WINDOW_SIZE", "4"))
    seed = int(os.getenv("CES_SEED", "42"))
    max_val_samples = int(os.getenv("CES_MAX_VAL_SAMPLES", "40000"))
    batch_size = int(os.getenv("CES_BATCH_SIZE", "512"))
    ablate = os.getenv("CES_ABLATE", "none")
    if ablate not in VALID_ABLATIONS:
        raise ValueError(f"CES_ABLATE must be one of {VALID_ABLATIONS}, got {ablate!r}")

    metrics_path = output_dir / "metrics.json"
    weights_path = output_dir / "weights" / "multimodal_ces.pth"
    manifest_path = split_dir / "split_manifest.json"
    for path in (metrics_path, weights_path, manifest_path):
        if not path.exists():
            raise FileNotFoundError(f"Required artifact missing: {path}. Run training first.")

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    stats = _load_stats(metrics)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    val_files = set(manifest["val_files"])

    print("Building clean (non-augmented) evaluation dataset...")
    dataset, val_indices, val_file_ids = build_clean_val_subset(
        data_dir, window_size, stats, val_files, max_val_samples, seed
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = DataLoader(
        Subset(dataset, val_indices),
        batch_size=batch_size,
        shuffle=False,
        pin_memory=device.type == "cuda",
    )

    model = MultimodalCESPredictor.from_dataset(dataset, window_size=window_size)
    state = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    print(f"Using device: {device}")

    target_mean = torch.as_tensor(stats["target"]["mean"], dtype=torch.float32)
    target_std = torch.as_tensor(stats["target"]["std"], dtype=torch.float32)

    pred_chunks, target_chunks, mask_chunks, persist_chunks, persist_obs_chunks = [], [], [], [], []
    with torch.no_grad():
        for batch in loader:
            bes = batch["bes"].to(device, non_blocking=True)
            ecei = batch["ecei"].to(device, non_blocking=True)
            mc = batch["mc"].to(device, non_blocking=True)
            time_features = batch["time_features"].to(device, non_blocking=True)
            ces_history = batch["ces_history"].to(device, non_blocking=True)
            # Persistence baseline always uses the REAL history, independent of ablation.
            persistence, has_obs = _persistence_from_history(ces_history)
            a_bes, a_ecei, a_mc, a_hist = apply_ablation(ablate, bes, ecei, mc, ces_history)
            outputs = model(a_bes, a_ecei, a_mc, time_features, a_hist)
            pred_chunks.append(outputs.cpu())
            target_chunks.append(batch["target"])
            mask_chunks.append(batch["target_mask"])
            persist_chunks.append(persistence.cpu())
            persist_obs_chunks.append(has_obs.cpu())

    pred = torch.cat(pred_chunks)
    target = torch.cat(target_chunks)
    mask = torch.cat(mask_chunks) > 0.5
    persist = torch.cat(persist_chunks)
    persist_obs = torch.cat(persist_obs_chunks)

    # Denormalize to physical CES units.
    pred_phys = pred * target_std + target_mean
    target_phys = target * target_std + target_mean
    persist_phys = persist * target_std + target_mean

    report = {
        "val_shots": len(val_file_ids),
        "val_samples": len(val_indices),
        "window_size": window_size,
        "history": "real_most_recent_window (non-augmented)",
        "units": "raw CES units as stored in CSV (CES_TI, CES_VT)",
        "ablation": ablate,
        "per_target": {},
    }
    for t, name in enumerate(TARGET_NAMES):
        # Evaluate only where this target is observed AND persistence has a value,
        # so model and persistence are compared on identical samples.
        keep = mask[:, t] & persist_obs[:, t]
        n = int(keep.sum())
        if n == 0:
            report["per_target"][name] = {"n": 0}
            continue
        y = target_phys[keep, t]
        err_model = pred_phys[keep, t] - y
        err_persist = persist_phys[keep, t] - y
        mse_model = float((err_model ** 2).mean())
        mse_persist = float((err_persist ** 2).mean())
        var = float(((y - y.mean()) ** 2).mean())
        report["per_target"][name] = {
            "n": n,
            "persistence_coverage": float(int((mask[:, t] & persist_obs[:, t]).sum()) / max(int(mask[:, t].sum()), 1)),
            "rmse_model": mse_model ** 0.5,
            "mae_model": float(err_model.abs().mean()),
            "rmse_persistence": mse_persist ** 0.5,
            "mae_persistence": float(err_persist.abs().mean()),
            "rmse_mean_baseline": var ** 0.5,
            "r2_vs_mean": 1.0 - mse_model / var if var > 0 else float("nan"),
            "skill_vs_persistence": 1.0 - mse_model / mse_persist if mse_persist > 0 else float("nan"),
        }

    print(
        f"\nClean validation -- {report['val_samples']} samples over "
        f"{report['val_shots']} val shots (window={window_size}, real history, ablation={ablate})\n"
    )
    header = f"{'target':<8} {'n':>7}  {'RMSE_model':>11} {'RMSE_persist':>12} {'RMSE_mean':>10}  {'skill_vs_persist':>16} {'R2_vs_mean':>11}"
    print(header)
    print("-" * len(header))
    for name in TARGET_NAMES:
        r = report["per_target"][name]
        if r.get("n", 0) == 0:
            print(f"{name:<8} {'0':>7}  (no observed samples)")
            continue
        print(
            f"{name:<8} {r['n']:>7}  {r['rmse_model']:>11.4f} {r['rmse_persistence']:>12.4f} "
            f"{r['rmse_mean_baseline']:>10.4f}  {r['skill_vs_persistence']:>16.4f} {r['r2_vs_mean']:>11.4f}"
        )
    print(
        "\nskill_vs_persistence > 0  => model beats carrying the last CES forward.\n"
        "R2_vs_mean              => fraction of variance explained vs predicting the mean."
    )

    eval_path = output_dir / "eval_metrics.json"
    eval_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nEvaluation metrics saved to {eval_path}")
    return report


if __name__ == "__main__":
    evaluate()
