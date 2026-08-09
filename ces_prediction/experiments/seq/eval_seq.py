"""Score the seq-LSTM on the EXACT compare_baselines.py evaluation population.

This is a fork of ces_prediction/compare_baselines.py with ONE substitution: model
inference. Instead of the per-sample window model, predictions come from a per-file
lookup table built by running the sequence model once over each eval file's full grid
(strictly causal features, drop_stuck matching the eval env, i.e. genuine-only history
-- the same regime the control model's eval used). Everything else -- population
enumeration (build_clean_val_subset), keep mask, baseline math (same
baselines_interpolation calls on the same file arrays), peak flags, npz keys, metrics
schema -- is byte-identical, so bootstrap_compare.py runs unchanged and
paired_model_compare.py's bit-identical se_pchip guard verifies the pairing.
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(1, str(HERE.parents[1]))  # ces_prediction/

from evaluate import _load_stats, _persistence_from_history, build_clean_val_subset  # noqa: E402
import baselines_interpolation as B  # noqa: E402
from analyze_gap import BIN_EDGES_MS  # noqa: E402
import peak_analysis as PK  # noqa: E402

from seq_data import load_grid_files, build_blocks  # noqa: E402
from model_seq import SeqCESLSTM  # noqa: E402
from model_seq_v2 import SeqCESLSTMv2  # noqa: E402

SEQ_MODELS = {"v1": SeqCESLSTM, "v2": SeqCESLSTMv2}

TARGET_NAMES = ("CES_TI", "CES_VT")
HEADLINE_BASELINE = "pchip"  # PR1


def _rmse(err):
    return float(np.sqrt(np.mean(err ** 2)))


def build_pred_lookup(data_dir, eval_names, my_stats, device):
    """name -> {float(time) -> (ti_phys, vt_phys)} via one causal pass per block.

    Features AND denormalization use the seq model's OWN training stats
    (`my_stats`); the table stores physical-unit predictions so the harness's
    control-stats round-trip (see compare()) never touches them.
    """
    drop_stuck = os.getenv("CES_DROP_STUCK_TARGETS", "1") == "1"
    # must match training: the model was fit on per-shot-standardized inputs or not
    per_shot = os.getenv("CES_PER_SHOT_NORM", "0") == "1"
    grid, dims = load_grid_files(data_dir, drop_stuck)
    variant = os.getenv("CES_SEQ_MODEL", "v1")
    if variant not in SEQ_MODELS:
        raise SystemExit(f"CES_SEQ_MODEL must be one of {sorted(SEQ_MODELS)}, got {variant!r}")
    model = SEQ_MODELS[variant]().to(device)
    out_dir = Path(os.environ["CES_OUTPUT_DIR"])
    model.load_state_dict(torch.load(out_dir / "weights" / "seq_lstm.pth", map_location="cpu"))
    model.eval()

    my_mean = torch.as_tensor(my_stats["target"]["mean"], dtype=torch.float32)
    my_std = torch.as_tensor(my_stats["target"]["std"], dtype=torch.float32)
    lookup = {}
    with torch.no_grad():
        for name in eval_names:
            if name not in grid:
                continue
            table = {}
            for blk in build_blocks(grid[name], dims, my_stats,
                                    per_shot_norm=per_shot):
                x = torch.from_numpy(blk["x"]).unsqueeze(0).to(device)
                pred = (model(x)[0].cpu() * my_std + my_mean).numpy()  # (L, 2) physical
                for i, t in enumerate(blk["time"]):
                    key = float(t)
                    if key in table:
                        raise SystemExit(f"FATAL: duplicate time {key} in {name}")
                    table[key] = pred[i]
            lookup[name] = table
    return lookup


def compare():
    root_dir = Path(__file__).resolve().parents[3]
    data_dir = Path(os.getenv("CES_DATA_DIR", root_dir / "data"))
    output_dir = Path(os.environ["CES_OUTPUT_DIR"])
    split_dir = Path(os.environ["CES_SPLIT_DIR"])
    window_size = int(os.getenv("CES_WINDOW_SIZE", "4"))
    seed = int(os.getenv("CES_SEED", "42"))
    max_val_samples = int(os.getenv("CES_MAX_VAL_SAMPLES", "40000"))
    batch_size = int(os.getenv("CES_BATCH_SIZE", "512"))
    split_tag = os.getenv("CES_SPLIT_TAG", "test")

    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    my_stats = _load_stats(metrics)
    # Harness stats come from the CONTROL run so the float32 normalize/denormalize
    # round-trip of targets, history and persistence is bit-identical to the control
    # npz (pairing requires bit-identical se_pchip; the round-trip depends on stats).
    control_metrics_path = Path(os.environ["CES_CONTROL_METRICS"])
    stats = _load_stats(json.loads(control_metrics_path.read_text(encoding="utf-8")))
    manifest = json.loads((split_dir / "split_manifest.json").read_text(encoding="utf-8"))
    eval_files = set(manifest[f"{split_tag}_files"])

    dataset, eval_indices, _ = build_clean_val_subset(
        data_dir, window_size, stats, eval_files, max_val_samples, seed
    )
    path_to_idx = {p: i for i, p in enumerate(dataset.valid_files)}
    time_col = int(dataset._column_slices["time"])
    target_cols = [int(c) for c in dataset._column_slices["target"]]

    methods = ["persistence", "linear", "pchip", "ar_local"]
    if B._HAVE_SKLEARN:
        methods.append("gp")
    else:
        print("[compare] sklearn unavailable -> GP baseline skipped.")

    device = torch.device(os.getenv("CES_SEQ_DEVICE",
                                    "cuda" if torch.cuda.is_available() else "cpu"))
    pred_lookup = build_pred_lookup(data_dir, sorted(eval_files), my_stats, device)

    target_mean = torch.as_tensor(stats["target"]["mean"], dtype=torch.float32)
    target_std = torch.as_tensor(stats["target"]["std"], dtype=torch.float32)

    def to_phys(t):
        return (t * target_std + target_mean).numpy()

    loader = DataLoader(Subset(dataset, eval_indices), batch_size=batch_size, shuffle=False)

    model_chunks, target_chunks, keep_chunks = [], [], []
    base_chunks = {m: [] for m in methods}
    dt_chunks, shot_chunks, row_chunks = [], [], []
    n_future = np.zeros(2, dtype=np.int64)
    n_total = np.zeros(2, dtype=np.int64)

    bes_cols = [int(c) for c in dataset._column_slices["bes"]]
    ecei_cols = [int(c) for c in dataset._column_slices["ecei"]]
    peak_cache = {}

    def _ces_activity_mask(file_idx, target_col):
        key = (file_idx, target_col)
        cached = peak_cache.get(key)
        if cached is None:
            fa = dataset.file_arrays[file_idx]
            cached = PK.detect_peak_rows_input_only(
                fa, time_col, target_col, bes_cols=bes_cols, ecei_cols=ecei_cols
            )["ces_activity"]
            peak_cache[key] = cached
        return cached

    print(f"[compare] split={split_tag}  samples={len(eval_indices)}  methods={methods}  (seq-LSTM lookup)")
    with torch.no_grad():
        for batch in loader:
            hist = batch["ces_history"]
            persistence, has_obs = _persistence_from_history(hist)

            files = batch["file"]
            rows = [int(r) for r in batch["row_index"].tolist()]
            b = len(rows)

            # seq-LSTM predictions via (file, time)-keyed lookup -- the ONE diff.
            # Table values are already physical units (denormalized with my_stats).
            out = np.zeros((b, 2), dtype=np.float32)
            for j in range(b):
                fa = dataset.file_arrays[path_to_idx[files[j]]]
                key = float(fa[rows[j], time_col])
                name = Path(files[j]).name
                try:
                    out[j] = pred_lookup[name][key]
                except KeyError:
                    raise SystemExit(
                        f"FATAL: no seq prediction for {name} @ t={key} -- grid mismatch.")
            model_chunks.append(out)

            target_chunks.append(to_phys(batch["target"]))
            keep_chunks.append(((batch["target_mask"] > 0.5) & has_obs.cpu()).numpy())

            persist_phys = to_phys(persistence.cpu())
            arr = {m: np.full((b, 2), np.nan, dtype=np.float64) for m in methods}
            dt_arr = np.full((b, 2), np.nan, dtype=np.float64)
            for j in range(b):
                fa = dataset.file_arrays[path_to_idx[files[j]]]
                ri = rows[j]
                for t in range(2):
                    tc = target_cols[t]
                    times, values, target_time = B.build_neighbor_set(fa, time_col, tc, ri)
                    past = times < target_time
                    if np.any(past):
                        dt_arr[j, t] = target_time - float(times[past].max())
                    has_future = np.any(times > target_time)
                    n_total[t] += 1
                    if has_future:
                        n_future[t] += 1
                    for m in methods:
                        if m == "persistence":
                            arr[m][j, t] = float(persist_phys[j, t])
                        else:
                            arr[m][j, t] = B.PREDICTORS[m](times, values, target_time)
            for m in methods:
                base_chunks[m].append(arr[m])
            dt_chunks.append(dt_arr)
            shot_chunks.append(np.array([path_to_idx[f] for f in files], dtype=np.int64))
            row_chunks.append(np.array(rows, dtype=np.int64))

    model_phys = np.concatenate(model_chunks)
    target_phys = np.concatenate(target_chunks)
    keep = np.concatenate(keep_chunks)
    base_phys = {m: np.concatenate(base_chunks[m]) for m in methods}
    dt_ms = np.concatenate(dt_chunks) * 1000.0
    shot_ids = np.concatenate(shot_chunks)
    row_ids = np.concatenate(row_chunks)

    report = {
        "split": split_tag,
        "eval_samples": len(eval_indices),
        "window_size": window_size,
        "headline_baseline": HEADLINE_BASELINE,
        "units": "physical CES (raw CSV units)",
        "note": "held-out TEST split (seq-LSTM full-grid masked-loss experiment)",
        "mnar_caveat": "skill measured on observed CES points only (MNAR optimistic bound)",
        "per_target": {},
    }

    boot = {}
    for t, name in enumerate(TARGET_NAMES):
        k = keep[:, t]
        valid = k.copy()
        for m in methods:
            valid &= ~np.isnan(base_phys[m][:, t])
        n = int(valid.sum())

        tc = target_cols[t]
        is_peak = np.zeros(len(shot_ids), dtype=bool)
        for s in range(len(shot_ids)):
            mask_f = _ces_activity_mask(int(shot_ids[s]), tc)
            is_peak[s] = bool(mask_f[int(row_ids[s])])

        y = target_phys[valid, t]
        err_model = model_phys[valid, t] - y
        rmse_model = _rmse(err_model)
        mse_model = float(np.mean(err_model ** 2))
        per_method = {}
        for m in methods:
            eb = base_phys[m][valid, t] - y
            per_method[m] = {"rmse": _rmse(eb), "mse": float(np.mean(eb ** 2))}
        mse_head = per_method[HEADLINE_BASELINE]["mse"]
        skill_head = 1.0 - mse_model / mse_head if mse_head > 0 else float("nan")
        report["per_target"][name] = {
            "n": n,
            "rmse_model": rmse_model,
            "baselines": per_method,
            f"skill_vs_{HEADLINE_BASELINE}": skill_head,
            f"beats_{HEADLINE_BASELINE}": bool(rmse_model < per_method[HEADLINE_BASELINE]["rmse"]),
            "future_neighbor_fraction": float(n_future[t] / max(int(n_total[t]), 1)),
            "peak_input_only_ces_activity_n": int(is_peak[valid].sum()),
        }
        print(f"{name}: n={n} rmse_model={rmse_model:.4f} "
              f"skill_vs_pchip={skill_head:+.4f} "
              f"(pchip rmse={per_method['pchip']['rmse']:.4f})")

        d = dt_ms[valid, t]
        em2 = err_model ** 2
        ep2 = {bl: (base_phys[bl][valid, t] - y) ** 2 for bl in ("pchip", "linear")}
        edges = list(BIN_EDGES_MS) + [float("inf")]
        bins, lo = [], 0.0
        for ub in edges:
            sel = (d > lo) if ub == float("inf") else ((d > lo) & (d <= ub))
            nb = int(sel.sum())
            label = f"({lo:g},inf)" if ub == float("inf") else f"({lo:g},{ub:g}]"
            brow = {"bin_ms": label, "n": nb}
            if nb > 0:
                mm = float(em2[sel].mean())
                brow["rmse_model"] = mm ** 0.5
                for bl in ("pchip", "linear"):
                    mb = float(ep2[bl][sel].mean())
                    brow[f"rmse_{bl}"] = mb ** 0.5
                    brow[f"skill_vs_{bl}"] = (1 - mm / mb) if mb > 0 else float("nan")
            bins.append(brow)
            lo = ub
        report["per_target"][name]["bins"] = bins

        boot[f"{name}_shot"] = shot_ids[valid]
        boot[f"{name}_dt_ms"] = dt_ms[valid, t]
        boot[f"{name}_se_model"] = err_model ** 2
        boot[f"{name}_se_pchip"] = (base_phys["pchip"][valid, t] - y) ** 2
        boot[f"{name}_se_linear"] = (base_phys["linear"][valid, t] - y) ** 2
        boot[f"{name}_is_peak"] = is_peak[valid]

    err_path = output_dir / f"comparison_errors_{split_tag}.npz"
    np.savez(err_path, **boot)
    out_path = output_dir / "comparison_metrics.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"saved {err_path} + {out_path}", flush=True)
    return report


if __name__ == "__main__":
    compare()
    # PyTorch CUDA teardown on Windows can fastfail (0xC0000409) at interpreter
    # exit AFTER all outputs are saved; skip teardown so rc reflects the run.
    os._exit(0)
