"""WS1 - Gap-length stratified skill analysis.

Computes skill_vs_persistence as a function of dt = time since the most-recent
observed CES value (from the lookback_seconds time feature), per target, on the
clean non-augmented validation split. Reuses evaluate.py's setup and persistence
logic; does NOT modify the AutoML loop or evaluate.py.

Run (PowerShell), pointing at the dirs the model was trained into:

    $env:CES_OUTPUT_DIR="data/.baseline_out"; $env:CES_SPLIT_DIR="data/.baseline_split"
    py ces_prediction/analyze_gap.py

Reads CES_OUTPUT_DIR/metrics.json, CES_OUTPUT_DIR/weights/multimodal_ces.pth,
CES_SPLIT_DIR/split_manifest.json. Writes CES_OUTPUT_DIR/gap_analysis.json.
"""

import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from dataset import KSTAR_CES_Dataset, select_seeded_random_indices
from model import MultimodalCESPredictor
from evaluate import _persistence_from_history, _load_stats

TARGET_NAMES = ("CES_TI", "CES_VT")
# dt bin upper edges in milliseconds; a final open bin (> last edge) is appended.
BIN_EDGES_MS = [15.0, 25.0, 35.0, 55.0, 105.0]


def _skill_row(err_model, err_persist):
    mse_m = float((err_model ** 2).mean())
    mse_p = float((err_persist ** 2).mean())
    return {
        "rmse_model": mse_m ** 0.5,
        "rmse_persist": mse_p ** 0.5,
        "skill_vs_persist": (1.0 - mse_m / mse_p) if mse_p > 0 else float("nan"),
    }


def _dt_since_last_obs(ces_history, time_features):
    """dt (seconds) from the most-recent observed CES to the target, per target.

    ces_history: (b, window, 4) = [TI, VT, TI_obs, VT_obs]; target masked (obs=0).
    time_features[..., 0] = lookback_seconds (time from each position to the target).
    """
    observed = ces_history[..., 2:] > 0.5  # (b, window, 2)
    b, window = observed.shape[0], observed.shape[1]
    positions = torch.arange(window, device=ces_history.device).view(1, window, 1)
    masked_pos = torch.where(observed, positions, torch.full_like(positions, -1))
    last_idx = masked_pos.max(dim=1).values.clamp(min=0)  # (b, 2)
    lookback = time_features[..., 0]  # (b, window) seconds to target
    rows = torch.arange(b, device=ces_history.device)
    return torch.stack([lookback[rows, last_idx[:, t]] for t in range(2)], dim=1)  # (b, 2)


def analyze():
    root_dir = Path(__file__).resolve().parents[1]
    data_dir = Path(os.getenv("CES_DATA_DIR", root_dir / "data"))
    output_dir = Path(os.getenv("CES_OUTPUT_DIR", Path(__file__).resolve().parent))
    split_dir = Path(os.getenv("CES_SPLIT_DIR", root_dir / "data" / "splits"))
    window_size = int(os.getenv("CES_WINDOW_SIZE", "4"))
    seed = int(os.getenv("CES_SEED", "42"))
    max_val_samples = int(os.getenv("CES_MAX_VAL_SAMPLES", "40000"))
    batch_size = int(os.getenv("CES_BATCH_SIZE", "512"))

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
    dataset = KSTAR_CES_Dataset(
        data_dir=data_dir, window_size=window_size, temporal_subset_augmentation=False
    )
    dataset.set_normalization_stats(stats)

    file_names = [Path(p).name for p in dataset.valid_files]
    val_file_ids = {i for i, name in enumerate(file_names) if name in val_files}
    val_indices = [
        i for i in range(len(dataset)) if int(dataset.sample_file_indices[i]) in val_file_ids
    ]
    if not val_indices:
        raise ValueError("No clean validation samples for the manifest val files.")
    val_indices = select_seeded_random_indices(val_indices, max_val_samples, seed + 202)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = DataLoader(
        Subset(dataset, val_indices),
        batch_size=batch_size,
        shuffle=False,
        pin_memory=device.type == "cuda",
    )

    model = MultimodalCESPredictor.from_dataset(dataset, window_size=window_size)
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    model.to(device)
    model.eval()
    print(f"Using device: {device}")

    target_mean = torch.as_tensor(stats["target"]["mean"], dtype=torch.float32)
    target_std = torch.as_tensor(stats["target"]["std"], dtype=torch.float32)

    preds, targets, masks, persists, persist_obss, dts = [], [], [], [], [], []
    with torch.no_grad():
        for batch in loader:
            bes = batch["bes"].to(device, non_blocking=True)
            ecei = batch["ecei"].to(device, non_blocking=True)
            mc = batch["mc"].to(device, non_blocking=True)
            time_features = batch["time_features"].to(device, non_blocking=True)
            ces_history = batch["ces_history"].to(device, non_blocking=True)
            persistence, has_obs = _persistence_from_history(ces_history)
            outputs = model(bes, ecei, mc, time_features, ces_history)
            dt = _dt_since_last_obs(ces_history, time_features)
            preds.append(outputs.cpu())
            targets.append(batch["target"])
            masks.append(batch["target_mask"])
            persists.append(persistence.cpu())
            persist_obss.append(has_obs.cpu())
            dts.append(dt.cpu())

    pred_phys = torch.cat(preds) * target_std + target_mean
    target_phys = torch.cat(targets) * target_std + target_mean
    persist_phys = torch.cat(persists) * target_std + target_mean
    mask = torch.cat(masks) > 0.5
    persist_obs = torch.cat(persist_obss)
    dt_ms = torch.cat(dts) * 1000.0

    edges = list(BIN_EDGES_MS) + [float("inf")]
    report = {
        "val_samples": len(val_indices),
        "window_size": window_size,
        "bin_edges_ms": BIN_EDGES_MS,
        "per_target": {},
    }

    for t, name in enumerate(TARGET_NAMES):
        keep = mask[:, t] & persist_obs[:, t]
        y = target_phys[:, t]
        err_model = pred_phys[:, t] - y
        err_persist = persist_phys[:, t] - y
        d = dt_ms[:, t]

        overall = {"n": int(keep.sum()), **_skill_row(err_model[keep], err_persist[keep])}
        bins, lo = [], 0.0
        for ub in edges:
            sel = keep & (d > lo) if ub == float("inf") else keep & (d > lo) & (d <= ub)
            label = f"({lo:g},inf)" if ub == float("inf") else f"({lo:g},{ub:g}]"
            row = {"bin_ms": label, "n": int(sel.sum())}
            if row["n"] > 0:
                row.update(_skill_row(err_model[sel], err_persist[sel]))
                row["dt_ms_median"] = float(d[sel].median())
            bins.append(row)
            lo = ub

        report["per_target"][name] = {
            "overall": overall,
            "dt_ms_min": float(d[keep].min()) if int(keep.sum()) else None,
            "dt_ms_max": float(d[keep].max()) if int(keep.sum()) else None,
            "bins": bins,
        }

    # Print human-readable tables.
    print(
        f"\nGap-length stratified skill -- {report['val_samples']} clean val samples "
        f"(window={window_size})"
    )
    for name in TARGET_NAMES:
        tr = report["per_target"][name]
        ov = tr["overall"]
        print(f"\n=== {name} ===  (overall n={ov['n']}, skill={ov['skill_vs_persist']:.4f}, "
              f"dt range {tr['dt_ms_min']:.1f}-{tr['dt_ms_max']:.1f} ms)")
        header = f"{'dt bin (ms)':<12} {'n':>7} {'dt_med':>7} {'RMSE_model':>11} {'RMSE_persist':>12} {'skill':>9}"
        print(header)
        print("-" * len(header))
        for b in tr["bins"]:
            if b["n"] == 0:
                print(f"{b['bin_ms']:<12} {b['n']:>7}  (empty)")
                continue
            print(f"{b['bin_ms']:<12} {b['n']:>7} {b['dt_ms_median']:>7.1f} "
                  f"{b['rmse_model']:>11.4f} {b['rmse_persist']:>12.4f} {b['skill_vs_persist']:>9.4f}")

    out_path = output_dir / "gap_analysis.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nGap analysis saved to {out_path}")
    return report


if __name__ == "__main__":
    analyze()
