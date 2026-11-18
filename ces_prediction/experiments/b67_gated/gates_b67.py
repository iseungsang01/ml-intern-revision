"""B.67 gate readout (PREREGISTRATION_B67.md section 3, 'Gate readout (arm S)'): the four source gates of the
network at every scored test row, aligned to the rows of comparison_errors_test.npz.

Alignment.  eval_seq scores a row (CES_TI_shot j = index into dataset.valid_files, CES_TI_row i = row index into
dataset.file_arrays[j]) with the prediction stored under the key (file name, float(file_arrays[j][i, time])) in a
table built by one causal pass of the network over every block of seq_data.build_blocks (build_pred_lookup).
readout() builds the same table from the same blocks with the gates beside the prediction, and looks every row of
the npz up under the same key.  Check: the T_i prediction found under that key reproduces CES_TI_se_model of the
npz (every row found; 99th percentile of |difference of the absolute errors| below 0.5 eV and the summed squared
errors within 1e-3, the B.59 criteria; in the scoring process itself the squared errors are reproduced bit for
bit), and the gates of every row are finite and sum to one.  A misaligned table fails the first condition
(check_b67.py shows it with the rows shifted by one).

Per row the file also keeps the T_i staleness of the network's input (seconds since the last measured T_i frame,
from channel 17) and the has-observation flag (channel 18), so the row classes can be cross-checked.

Called by eval_b67.py inside the scoring process (arm S), reusing the grid and dataset eval_seq built.  As a
script it fills in a missing gate file of a finished run (CPU by default; the env is that of the run):

    py ces_prediction/experiments/b67_gated/gates_b67.py --arm S --repeat 2 --fold 0 [--device cpu] [--force]
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

import model_seq_v2gate as M   # puts seq/ and ces_prediction/ on sys.path
import torch  # noqa: E402

T = "CES_TI"
OUT_NPZ, OUT_JSON = "b67_gates_test.npz", "b67_gates_test.json"
DT_COL, TI_STALE_COL, TI_HAS_COL = 15, 17, 18            # seq_data.build_blocks: log1p dt | TI carried, log1p staleness, has
TOL_P99_EV, TOL_SKILL, TOL_GATE_SUM = 0.5, 1e-3, 1e-4
REG = "ces_prediction/experiments/PREREGISTRATION_B67.md"


def device_of_env():
    """The device eval_seq.compare uses."""
    return torch.device(os.getenv("CES_SEQ_DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))


def env_settings():
    """What eval_seq.compare / build_pred_lookup read from the environment."""
    if os.getenv("CES_SEQ_EVAL_CTX", "0") != "0" or os.getenv("CES_SEQ_CONTEXTS", "").strip() \
            or os.getenv("CES_SEQ_ENSEMBLE_DIRS", "").strip():
        raise SystemExit("FATAL: the gate readout is defined for whole-block inference of one checkpoint only")
    root = M.CES_DIR.parent
    return {"data_dir": Path(os.getenv("CES_DATA_DIR", root / "data")),
            "split_dir": Path(os.environ["CES_SPLIT_DIR"]),
            "control_metrics": Path(os.environ["CES_CONTROL_METRICS"]),
            "window_size": int(os.getenv("CES_WINDOW_SIZE", "4")),
            "seed": int(os.getenv("CES_SEED", "42")),
            "max_val_samples": int(os.getenv("CES_MAX_VAL_SAMPLES", "40000")),
            "split_tag": os.getenv("CES_SPLIT_TAG", "test"),
            "drop_stuck": os.getenv("CES_DROP_STUCK_TARGETS", "1") == "1",
            "per_shot": os.getenv("CES_PER_SHOT_NORM", "0") == "1"}


def install_captures(eval_seq):
    """Wrap eval_seq's module globals load_grid_files and build_clean_val_subset so that their return values are
    kept (returned unchanged).  check_b67 verifies that eval_seq reaches both only through these globals."""
    cap = {"grid_calls": 0, "dataset_calls": 0}
    lg, bc = eval_seq.load_grid_files, eval_seq.build_clean_val_subset

    def load_grid_files(*a, **k):
        res = lg(*a, **k)
        cap["grid"], cap["grid_args"] = res, (a, k)
        cap["grid_calls"] += 1
        return res

    def build_clean_val_subset(*a, **k):
        res = bc(*a, **k)
        cap["dataset"] = res[0]
        cap["dataset_calls"] += 1
        return res

    eval_seq.load_grid_files = load_grid_files
    eval_seq.build_clean_val_subset = build_clean_val_subset
    return cap


def load_model(run_dir, variant, device):
    """The checkpoint of the run under its named architecture, as eval_seq.build_pred_lookup loads it."""
    M.register()
    import seq_models
    if variant not in seq_models.SEQ_MODELS:
        raise SystemExit(f"FATAL: {variant!r} not in seq_models.SEQ_MODELS")
    model = seq_models.SEQ_MODELS[variant]().to(device)
    model.load_state_dict(torch.load(Path(run_dir) / "weights" / "seq_lstm.pth", map_location="cpu"))
    model.eval()
    return model


def block_tables(model, grid, dims, names, my_stats, device, per_shot):
    """name -> {"index": {float(time): position}, "pred_ti" (n,) float32 physical, "gates" (n, 4) | None,
    "ti_stale_s" (n,), "ti_has" (n,)}: one causal pass per block, the pass of eval_seq.build_pred_lookup."""
    from seq_data import build_blocks
    with_gates = hasattr(model, "forward_with_gates")
    mean = torch.as_tensor(my_stats["target"]["mean"], dtype=torch.float32)
    std = torch.as_tensor(my_stats["target"]["std"], dtype=torch.float32)
    tables = {}
    with torch.no_grad():
        for name in names:
            if name not in grid:
                continue
            keys, pred, gates, stale, has = [], [], [], [], []
            for blk in build_blocks(grid[name], dims, my_stats, per_shot_norm=per_shot):
                x = torch.from_numpy(blk["x"]).unsqueeze(0).to(device)
                if with_gates:
                    out, g = model.forward_with_gates(x)
                    gates.append(g[0].cpu().numpy())
                else:
                    out = model(x)
                phys = (out[0].cpu() * std + mean).numpy()       # (L, 2), as build_pred_lookup
                pred.append(phys[:, 0])
                keys.extend(float(t) for t in blk["time"])
                stale.append(np.expm1(blk["x"][:, TI_STALE_COL].astype(np.float64)))
                has.append(blk["x"][:, TI_HAS_COL] > 0.5)
            index = {}
            for i, kk in enumerate(keys):
                if kk in index:
                    raise SystemExit(f"FATAL: duplicate time {kk} in {name}")
                index[kk] = i
            tables[name] = {"index": index,
                            "pred_ti": np.concatenate(pred) if pred else np.zeros(0, np.float32),
                            "gates": (np.concatenate(gates) if gates else np.zeros((0, len(M.SOURCE_LABELS)), np.float32))
                            if with_gates else None,
                            "ti_stale_s": np.concatenate(stale) if stale else np.zeros(0),
                            "ti_has": np.concatenate(has) if has else np.zeros(0, bool)}
    return tables


def align(tables, dataset, shot, row, offset=0):
    """Look every stored row up under eval_seq's key.  offset != 0 shifts the row index (negative control only)."""
    time_col = int(dataset._column_slices["time"])
    N = len(shot)
    any_gates = any(t["gates"] is not None for t in tables.values())
    out = {"found": np.zeros(N, bool), "pred_ti": np.full(N, np.nan, np.float32),
           "gates": np.full((N, len(M.SOURCE_LABELS)), np.nan, np.float32) if any_gates else None,
           "ti_stale_s": np.full(N, np.nan), "ti_has": np.zeros(N, bool)}
    for j in range(N):
        s, i = int(shot[j]), int(row[j]) + offset
        fa = dataset.file_arrays[s]
        if not 0 <= i < len(fa):
            continue
        tab = tables.get(Path(dataset.valid_files[s]).name)
        if tab is None:
            continue
        q = tab["index"].get(float(fa[i, time_col]))
        if q is None:
            continue
        out["found"][j] = True
        out["pred_ti"][j] = tab["pred_ti"][q]
        out["ti_stale_s"][j] = tab["ti_stale_s"][q]
        out["ti_has"][j] = tab["ti_has"][q]
        if any_gates:
            out["gates"][j] = tab["gates"][q]
    return out


def check(al, z):
    """The squared error of the looked-up prediction against the stored one; gates finite and summing to one."""
    y = z[f"{T}_y_true"]
    se_st = z[f"{T}_se_model"]
    f = al["found"]
    se_re = (al["pred_ti"] - y) ** 2                                  # float32, as eval_seq
    d = np.abs(np.sqrt(se_re[f].astype(np.float64)) - np.sqrt(se_st[f].astype(np.float64)))
    res = {"rows": int(len(f)), "found": int(f.sum()),
           "identical_rows": int(np.sum(se_re[f] == se_st[f])),
           "p50_abs_err_diff_eV": float(np.median(d)) if len(d) else float("nan"),
           "p99_abs_err_diff_eV": float(np.percentile(d, 99)) if len(d) else float("nan"),
           "max_abs_err_diff_eV": float(d.max()) if len(d) else float("nan"),
           "skill_vs_stored": float(1.0 - se_re[f].astype(np.float64).sum() / se_st[f].astype(np.float64).sum())
           if f.any() else float("nan")}
    ok = bool(f.all() and res["p99_abs_err_diff_eV"] < TOL_P99_EV and abs(res["skill_vs_stored"]) < TOL_SKILL)
    if al["gates"] is not None:
        g = al["gates"][f].astype(np.float64)
        res["gates_finite"] = bool(np.isfinite(g).all())
        res["max_abs_gate_sum_minus_one"] = float(np.max(np.abs(g.sum(1) - 1.0))) if len(g) else float("nan")
        res["mean_gate"] = dict(zip(M.SOURCE_LABELS, map(float, g.mean(0)))) if len(g) else {}
        ok &= bool(res["gates_finite"] and res["max_abs_gate_sum_minus_one"] < TOL_GATE_SUM)
    res["passes"] = ok
    return res


def readout(run_dir, variant, device, captured=None, dest=None, strict=True, offset=0, model=None):
    """Gate table of the run's checkpoint aligned to its comparison_errors_test.npz; written to dest when given.
    `captured` (install_captures) reuses the grid and dataset of the scoring process."""
    from evaluate import _load_stats, build_clean_val_subset
    from seq_data import load_grid_files
    run_dir = Path(run_dir)
    S = env_settings()
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    if metrics["seq_model"] != variant:
        raise SystemExit(f"FATAL: {run_dir.name} was trained as {metrics['seq_model']!r}, not {variant!r}")
    my_stats = _load_stats(metrics)
    if captured and "grid" in captured:
        a, k = captured["grid_args"]
        if k or len(a) != 2 or Path(a[0]) != S["data_dir"] or bool(a[1]) != S["drop_stuck"]:
            raise SystemExit(f"FATAL: the captured grid was loaded with {a} {k}, not ({S['data_dir']}, {S['drop_stuck']})")
        grid, dims = captured["grid"]
    else:
        grid, dims = load_grid_files(S["data_dir"], S["drop_stuck"])
    manifest = json.loads((S["split_dir"] / "split_manifest.json").read_text(encoding="utf-8"))
    eval_files = set(manifest[f"{S['split_tag']}_files"])
    if captured and "dataset" in captured:
        dataset = captured["dataset"]
    else:
        stats = _load_stats(json.loads(S["control_metrics"].read_text(encoding="utf-8")))
        dataset, _, _ = build_clean_val_subset(S["data_dir"], S["window_size"], stats, eval_files,
                                               S["max_val_samples"], S["seed"])
    if model is None:
        model = load_model(run_dir, variant, device)
    tables = block_tables(model, grid, dims, sorted(eval_files), my_stats, device, S["per_shot"])
    z = np.load(run_dir / f"comparison_errors_{S['split_tag']}.npz")
    shot, row = z[f"{T}_shot"], z[f"{T}_row"]
    al = align(tables, dataset, shot, row, offset=offset)
    res = check(al, z)
    res.update({"run": run_dir.name, "variant": variant, "device": str(device), "offset": offset,
                "registration": REG, "sources": list(M.SOURCE_LABELS)})
    if dest is not None:
        if al["gates"] is None:
            raise SystemExit(f"FATAL: {variant!r} has no gates to write")
        dest = Path(dest)
        np.savez_compressed(dest / OUT_NPZ, shot=shot, row=row, gates=al["gates"], pred_ti=al["pred_ti"],
                            ti_stale_s=al["ti_stale_s"], ti_has=al["ti_has"], sources=np.array(M.SOURCE_LABELS),
                            check=np.array(json.dumps(res)))
        (dest / OUT_JSON).write_text(json.dumps(res, indent=1), encoding="utf-8")
    if strict and not res["passes"]:
        raise SystemExit(f"FATAL: gate readout of {run_dir.name}: alignment check fails {res}")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["S", "A"], required=True)
    ap.add_argument("--repeat", type=int, required=True)
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--force", action="store_true", help="rewrite an existing gate file")
    args = ap.parse_args()
    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import run_b67 as R
    out = R.run_dir(args.arm, args.repeat, args.fold)
    if (out / OUT_NPZ).exists() and not args.force:
        print(f"[b67] {out.name}: {OUT_NPZ} exists (use --force to rewrite)")
        return
    if not (out / "comparison_errors_test.npz").exists():
        raise SystemExit(f"FATAL: {out.name} has not been scored")
    env = R.arm_env(args.arm, args.repeat, args.fold, out)
    for key in [x for x in os.environ if x.startswith("CES_")]:
        del os.environ[key]
    os.environ.update({x: v for x, v in env.items() if x.startswith("CES_")})
    os.environ["CES_SEQ_DEVICE"] = args.device
    res = readout(out, R.MODEL[args.arm], torch.device(args.device), dest=out, strict=True)
    print(f"[b67] {out.name}: {res['rows']} rows, passes {res['passes']}, p99 |diff| {res['p99_abs_err_diff_eV']:.3g} eV")


if __name__ == "__main__":
    main()
