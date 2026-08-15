"""Compare the trained nowcasting model against conventional interpolation
baselines on the clean (non-augmented) split, per target, in physical CES units.

This is the harness for the thesis claim: the model (fast diagnostics + past CES)
should beat conventional CES-only interpolation (linear/pchip/ar/gp) that uses
past+future CES, for CES_TI.

Fairness (see `.omc/plans/ces-interpolation-comparison-consensus.md`):
- Reuses `evaluate.build_clean_val_subset` so the model is scored on byte-identical
  samples to its canonical evaluation (single source of truth, no drift).
- Keep mask per target = `target observed & window-persistence available`
  (evaluate.py's exact mask, `_persistence_from_history`). Interpolation is defined
  on every such sample via persistence fallback where no future neighbor exists (PR2).
- persistence baseline uses the window-based last-observed (matches evaluate.py
  `rmse_persistence` exactly -> regression cross-check). linear/pchip/ar/gp read
  past+future observed CES from the raw per-file array (block neighbors, <0.5s gap).
- Headline baseline = PCHIP (PR1). Full ladder reported.

Split selection: CES_SPLIT_TAG = 'val' (default; this is the pre-registered
SELECTION-VAL FALLBACK, optimism-caveated) or 'test' (held-out headline).

Env: CES_OUTPUT_DIR (metrics.json + weights), CES_SPLIT_DIR (split_manifest.json),
CES_DATA_DIR, CES_WINDOW_SIZE, CES_SEED, CES_MAX_VAL_SAMPLES, CES_BATCH_SIZE.
Writes CES_OUTPUT_DIR/comparison_metrics.json.
"""

import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from evaluate import (
    VALID_ABLATIONS,
    _load_stats,
    _persistence_from_history,
    apply_ablation,
    build_clean_val_subset,
)
from model import MultimodalCESPredictor
import baselines_interpolation as B
from analyze_gap import BIN_EDGES_MS
import peak_analysis as PK

TARGET_NAMES = ("CES_TI", "CES_VT")
HEADLINE_BASELINE = "pchip"  # PR1


def _rmse(err):
    return float(np.sqrt(np.mean(err ** 2)))


def compare():
    root_dir = Path(__file__).resolve().parents[1]
    data_dir = Path(os.getenv("CES_DATA_DIR", root_dir / "data"))
    output_dir = Path(os.getenv("CES_OUTPUT_DIR", Path(__file__).resolve().parent))
    split_dir = Path(os.getenv("CES_SPLIT_DIR", root_dir / "data" / "splits"))
    window_size = int(os.getenv("CES_WINDOW_SIZE", "4"))
    seed = int(os.getenv("CES_SEED", "42"))
    max_val_samples = int(os.getenv("CES_MAX_VAL_SAMPLES", "40000"))
    batch_size = int(os.getenv("CES_BATCH_SIZE", "512"))
    split_tag = os.getenv("CES_SPLIT_TAG", "val")
    ablate = os.getenv("CES_ABLATE", "none")
    if ablate not in VALID_ABLATIONS:
        raise ValueError(f"CES_ABLATE must be one of {VALID_ABLATIONS}, got {ablate!r}")

    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    stats = _load_stats(metrics)
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
        # Past-only GP (PREREGISTRATION_W2.md §5): the candidate strongest
        # deployable causal baseline. Same NaN condition as ar_local, so the
        # `valid` population below is unchanged by its presence.
        methods.append("gp_causal")
    else:
        print("[compare] sklearn unavailable -> GP baselines skipped.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultimodalCESPredictor.from_dataset(dataset, window_size=window_size)
    model.load_state_dict(torch.load(output_dir / "weights" / "multimodal_ces.pth", map_location="cpu"))
    model.to(device)
    model.eval()

    target_mean = torch.as_tensor(stats["target"]["mean"], dtype=torch.float32)
    target_std = torch.as_tensor(stats["target"]["std"], dtype=torch.float32)

    def to_phys(t):
        return (t * target_std + target_mean).numpy()

    loader = DataLoader(Subset(dataset, eval_indices), batch_size=batch_size, shuffle=False)

    model_chunks, target_chunks, keep_chunks = [], [], []
    base_chunks = {m: [] for m in methods}
    dt_chunks = []
    shot_chunks = []
    row_chunks = []  # per-sample target row_index (for input-only peak lookup)
    n_future = np.zeros(2, dtype=np.int64)  # how often a future neighbor was usable
    n_total = np.zeros(2, dtype=np.int64)

    # Input-only (Family i-a) CES-activity peak masks, precomputed once per
    # (file, target_col). Headline peak definition: never reads CES[idx]. Cached
    # so the per-sample lookup below is O(1). See peak_analysis.PEAK_PARAMS.
    bes_cols = [int(c) for c in dataset._column_slices["bes"]]
    ecei_cols = [int(c) for c in dataset._column_slices["ecei"]]
    peak_cache = {}  # (file_idx, target_col) -> bool mask over file rows

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

    print(f"[compare] split={split_tag}  samples={len(eval_indices)}  methods={methods}  ablate={ablate}")
    with torch.no_grad():
        for batch in loader:
            bes = batch["bes"].to(device, non_blocking=True)
            ecei = batch["ecei"].to(device, non_blocking=True)
            mc = batch["mc"].to(device, non_blocking=True)
            tf = batch["time_features"].to(device, non_blocking=True)
            hist = batch["ces_history"].to(device, non_blocking=True)

            # Persistence baseline and keep mask always come from the REAL history;
            # the ablation only deprives the model's inputs (mirrors train.py).
            persistence, has_obs = _persistence_from_history(hist)
            m_bes, m_ecei, m_mc, m_hist = apply_ablation(ablate, bes, ecei, mc, hist)
            out = model(m_bes, m_ecei, m_mc, tf, m_hist)

            model_chunks.append(to_phys(out.cpu()))
            target_chunks.append(to_phys(batch["target"]))
            keep_chunks.append(((batch["target_mask"] > 0.5) & has_obs.cpu()).numpy())

            persist_phys = to_phys(persistence.cpu())
            files = batch["file"]
            rows = [int(r) for r in batch["row_index"].tolist()]
            b = len(rows)
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
    row_ids = np.concatenate(row_chunks)  # per-sample target row_index

    report = {
        "split": split_tag,
        "eval_samples": len(eval_indices),
        "window_size": window_size,
        "ablation": ablate,
        "headline_baseline": HEADLINE_BASELINE,
        "units": "physical CES (raw CSV units)",
        "note": ("SELECTION-VAL FALLBACK (optimism caveat): model was AutoML-selected on this val set"
                 if split_tag == "val" else "held-out TEST split"),
        "mnar_caveat": "skill measured on observed CES points only (MNAR optimistic bound)",
        "per_target": {},
    }

    header = f"{'target':<8} {'n':>7} {'RMSE_model':>11} " + " ".join(f"{'RMSE_'+m:>13}" for m in methods) + f" {'skill_vs_'+HEADLINE_BASELINE:>16}"
    print("\n" + header)
    print("-" * len(header))
    boot = {}
    for t, name in enumerate(TARGET_NAMES):
        k = keep[:, t]
        # require all arms defined on the kept samples
        valid = k.copy()
        for m in methods:
            valid &= ~np.isnan(base_phys[m][:, t])
        n = int(valid.sum())

        # Input-only (Family i-a) CES-activity peak flag per sample, aligned to
        # the full sample order (same as model_phys/target_phys/keep/valid). The
        # mask is the headline peak definition and is NEVER derived from CES[idx].
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
        beats_head = bool(rmse_model < per_method[HEADLINE_BASELINE]["rmse"])
        report["per_target"][name] = {
            "n": n,
            "rmse_model": rmse_model,
            "baselines": per_method,
            f"skill_vs_{HEADLINE_BASELINE}": skill_head,
            f"beats_{HEADLINE_BASELINE}": beats_head,
            "future_neighbor_fraction": float(n_future[t] / max(int(n_total[t]), 1)),
            # additive diagnostic only (NOT an AC1-protected key): how many of the
            # `valid` samples fall in the input-only headline peak subset.
            "peak_input_only_ces_activity_n": int(is_peak[valid].sum()),
        }
        row = f"{name:<8} {n:>7} {rmse_model:>11.4f} " + " ".join(f"{per_method[m]['rmse']:>13.4f}" for m in methods) + f" {skill_head:>16.4f}"
        print(row)

        # gap-length stratified: where (in dt = time since last observed CES) does the model win?
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

        # per-sample squared errors + shot id (for shot-clustered bootstrap CI, PR4)
        boot[f"{name}_shot"] = shot_ids[valid]
        boot[f"{name}_dt_ms"] = dt_ms[valid, t]
        boot[f"{name}_se_model"] = err_model ** 2
        boot[f"{name}_se_pchip"] = (base_phys["pchip"][valid, t] - y) ** 2
        boot[f"{name}_se_linear"] = (base_phys["linear"][valid, t] - y) ** 2
        # Additive key: the CAUSAL reference arm. pchip/linear both read a future
        # anchor, so at large dt they solve an easier problem than the model does by
        # construction; persistence is the strongest baseline that is actually
        # available online, and is the honest comparator for the large-gap regime.
        boot[f"{name}_se_persistence"] = (base_phys["persistence"][valid, t] - y) ** 2
        # Additive keys (2026-08-05 audit follow-up): the GP arm's squared errors
        # (only when sklearn is present -- the arm was designed in from the start
        # but never ran) and the physical target values, which enable downstream
        # sensitivity analyses (e.g. CES fit-failure row exclusion) without
        # another re-scoring pass. Both are sliced by the SAME `valid` mask.
        if "gp" in methods:
            boot[f"{name}_se_gp"] = (base_phys["gp"][valid, t] - y) ** 2
        # Additive key (B.1, PREREGISTRATION_W2.md §5): the past-only GP arm --
        # the claim-2 gate is decided by model-vs-gp_causal paired bootstrap.
        if "gp_causal" in methods:
            boot[f"{name}_se_gp_causal"] = (base_phys["gp_causal"][valid, t] - y) ** 2
        boot[f"{name}_y_true"] = y
        # Additive key (2026-08-09): the target's row index inside its own shot file.
        # `_shot` alone cannot address a row, so any downstream analysis that needs a
        # per-row property computed from the raw CSV (e.g. the held/forward-filled flag
        # of §8r) had no way to align to these SEs. Sliced by the SAME `valid` mask.
        boot[f"{name}_row"] = row_ids[valid]
        # ONE additive key: input-only (Family i-a) CES-activity peak flag,
        # sliced by the SAME `valid` mask as the SE arrays so the byte-identical
        # headline SEs can be masked to the peak subset downstream (peak_analysis).
        boot[f"{name}_is_peak"] = is_peak[valid]
        assert len(boot[f"{name}_is_peak"]) == len(boot[f"{name}_se_model"]), (
            f"is_peak/se_model length mismatch for {name}: "
            f"{len(boot[f'{name}_is_peak'])} != {len(boot[f'{name}_se_model'])}"
        )

    print("\nskill_vs_%s > 0  => model beats %s interpolation (lower MSE)." % (HEADLINE_BASELINE, HEADLINE_BASELINE))
    print("future_neighbor_fraction = share of kept samples where interpolation used a real future neighbor (else persistence fallback, PR2).")
    print(f"NOTE: {report['note']}")

    err_path = output_dir / f"comparison_errors_{split_tag}.npz"
    np.savez(err_path, **boot)
    print(f"Per-sample errors (for bootstrap) saved to {err_path}")

    tagged = output_dir / f"comparison_metrics_{split_tag}.json"
    tagged.write_text(json.dumps(report, indent=2), encoding="utf-8")
    # Legacy unsuffixed report: written for TEST, or when nothing exists yet. A
    # val re-score must never overwrite a frozen TEST report (see eval_seq.py).
    out_path = output_dir / "comparison_metrics.json"
    if split_tag == "test" or not out_path.exists():
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nComparison metrics saved to {tagged}")
    return report


if __name__ == "__main__":
    compare()
