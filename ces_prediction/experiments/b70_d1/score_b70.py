"""B.70 phase 1 (a), shared-reference scorer (PREREGISTRATION_B70.md Addendum 1, A6).

seq/eval_seq.compare() computes, for every run, (1) the scored rows, targets, hold, PCHIP, linear, local AR, offline
GP and causal GP of every eligible sample, and (2) the network's prediction by a (file, time) lookup into
build_pred_lookup.  Part (1) does not depend on the network: every arm of a fold has the same split, harness
statistics (CES_CONTROL_METRICS), seed, window and data treatment, so it is identical across the nine arms (check (c)).
This scorer runs eval_seq's own code for part (1) ONCE per fold and part (2) once per arm, then writes, per arm, the
same comparison_errors_test.npz / comparison_metrics_test.json / comparison_metrics.json that eval_seq.compare()
writes.  The loop and the post-loop block below are copied from eval_seq.compare() (marked); the functions it calls
(build_clean_val_subset, _persistence_from_history, B.PREDICTORS, build_pred_lookup, peak_analysis) are imported.

Per arm, os.environ is set to that run's env (runs_b70.rescore_env: the runner's env function with
CES_MAX_VAL_SAMPLES, CES_SEQ_DEVICE and CES_OUTPUT_DIR changed); B.59 arm L installs fast_shift.install_eval(1)
around build_pred_lookup (as eval_b59L.py) and restores it; B.67 registers v2s / v2a (as eval_b67.py; its gate
readout is not written).  Validation: on fold r2 f0 every key of every arm's npz is compared with the output of the
arm's own scoring script (runs_b70.execute) -- `--validate`.

    py ces_prediction/experiments/b70_d1/score_b70.py --folds r2f1 r2f2 ... [--root DIR] [--cap 0]
    py ces_prediction/experiments/b70_d1/score_b70.py --validate        # r2 f0 into data/.b70_rescore_check/, compare
"""
from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import argparse  # noqa: E402
import gc  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time as _time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
sys.path.insert(0, str(HERE))
import runs_b70 as RB  # noqa: E402

SEQ_DIR = RB.SEQ_DIR
CES_DIR = SEQ_DIR.parents[1]
for _p in (str(CES_DIR), str(SEQ_DIR)):          # seq/ first, as eval_seq puts it
    if _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)
sys.path.insert(0, str(EXP / "b59_inputs"))
sys.path.insert(0, str(EXP / "b67_gated"))
PREFIXES = ("CES_", "B59_", "B67_", "B68_")
CHECK_ROOT = RB.DATA / ".b70_rescore_check"
T0 = _time.time()


def log(m):
    print(f"[{_time.time() - T0:6.0f}s] {m}", flush=True)


def set_env(env):
    for k in [k for k in os.environ if k.startswith(PREFIXES)]:
        del os.environ[k]
    os.environ.update({k: v for k, v in env.items() if k.startswith(PREFIXES)})


# ------------------------------------------------------------------------------------------ part (1), once per fold
def harness(r, k, cap):
    import torch
    from torch.utils.data import DataLoader, Subset
    import eval_seq as E
    out_v2 = RB.new_dir("b39", "v2", r, k, cap)
    set_env(RB.rescore_env("b39", "v2", r, k, out_v2, cap))
    # ---- copied from eval_seq.compare() (inputs of the harness)
    root_dir = Path(E.__file__).resolve().parents[3]
    data_dir = Path(os.getenv("CES_DATA_DIR", root_dir / "data"))
    split_dir = Path(os.environ["CES_SPLIT_DIR"])
    window_size = int(os.getenv("CES_WINDOW_SIZE", "4"))
    seed = int(os.getenv("CES_SEED", "42"))
    max_val_samples = int(os.getenv("CES_MAX_VAL_SAMPLES", "40000"))
    batch_size = int(os.getenv("CES_BATCH_SIZE", "512"))
    split_tag = os.getenv("CES_SPLIT_TAG", "test")
    control_metrics_path = Path(os.environ["CES_CONTROL_METRICS"])
    stats = E._load_stats(json.loads(control_metrics_path.read_text(encoding="utf-8")))
    manifest = json.loads((split_dir / "split_manifest.json").read_text(encoding="utf-8"))
    eval_files = set(manifest[f"{split_tag}_files"])
    dataset, eval_indices, _ = E.build_clean_val_subset(data_dir, window_size, stats, eval_files, max_val_samples, seed)
    path_to_idx = {p: i for i, p in enumerate(dataset.valid_files)}
    time_col = int(dataset._column_slices["time"])
    target_cols = [int(c) for c in dataset._column_slices["target"]]
    methods = ["persistence", "linear", "pchip", "ar_local"]
    if not E.B._HAVE_SKLEARN:
        raise SystemExit("FATAL: sklearn/scipy GP unavailable; the GP references cannot be scored")
    methods += ["gp", "gp_causal"]
    target_mean = torch.as_tensor(stats["target"]["mean"], dtype=torch.float32)
    target_std = torch.as_tensor(stats["target"]["std"], dtype=torch.float32)

    def to_phys(t):
        return (t * target_std + target_mean).numpy()

    loader = DataLoader(Subset(dataset, eval_indices), batch_size=batch_size, shuffle=False)
    target_chunks, keep_chunks = [], []
    base_chunks = {m: [] for m in methods}
    dt_chunks, shot_chunks, row_chunks, name_chunks, key_chunks = [], [], [], [], []
    n_future = np.zeros(2, dtype=np.int64)
    n_total = np.zeros(2, dtype=np.int64)
    bes_cols = [int(c) for c in dataset._column_slices["bes"]]
    ecei_cols = [int(c) for c in dataset._column_slices["ecei"]]
    log(f"r{r} f{k}: harness on {len(eval_indices)} samples")
    with torch.no_grad():
        for batch in loader:
            hist = batch["ces_history"]
            persistence, has_obs = E._persistence_from_history(hist)
            files = batch["file"]
            rows = [int(rr) for rr in batch["row_index"].tolist()]
            b = len(rows)
            # the (file, time) key of eval_seq's lookup, stored for part (2)
            for j in range(b):
                fa = dataset.file_arrays[path_to_idx[files[j]]]
                key_chunks.append(float(fa[rows[j], time_col]))
                name_chunks.append(Path(files[j]).name)
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
                    times, values, target_time = E.B.build_neighbor_set(fa, time_col, tc, ri)
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
                            arr[m][j, t] = E.B.PREDICTORS[m](times, values, target_time)
            for m in methods:
                base_chunks[m].append(arr[m])
            dt_chunks.append(dt_arr)
            shot_chunks.append(np.array([path_to_idx[f] for f in files], dtype=np.int64))
            row_chunks.append(np.array(rows, dtype=np.int64))
    H = {"target_phys": np.concatenate(target_chunks), "keep": np.concatenate(keep_chunks),
         "base_phys": {m: np.concatenate(base_chunks[m]) for m in methods},
         "dt_ms": np.concatenate(dt_chunks) * 1000.0, "shot_ids": np.concatenate(shot_chunks),
         "row_ids": np.concatenate(row_chunks), "names": name_chunks, "keys": key_chunks, "methods": methods,
         "n_future": n_future, "n_total": n_total, "eval_samples": len(eval_indices), "window_size": window_size,
         "split_tag": split_tag, "eval_files": sorted(eval_files), "data_dir": data_dir}
    # peak flags (input-only CES activity; no fast channel enters "ces_activity"), as in compare()
    peak_cache = {}

    def _ces_activity_mask(file_idx, target_col):
        key = (file_idx, target_col)
        cached = peak_cache.get(key)
        if cached is None:
            fa = dataset.file_arrays[file_idx]
            cached = E.PK.detect_peak_rows_input_only(fa, time_col, target_col, bes_cols=bes_cols,
                                                      ecei_cols=ecei_cols)["ces_activity"]
            peak_cache[key] = cached
        return cached

    H["is_peak"] = {}
    for t in range(2):
        tc = target_cols[t]
        is_peak = np.zeros(len(H["shot_ids"]), dtype=bool)
        for s in range(len(H["shot_ids"])):
            is_peak[s] = bool(_ces_activity_mask(int(H["shot_ids"][s]), tc)[int(H["row_ids"][s])])
        H["is_peak"][t] = is_peak
    del dataset, loader
    gc.collect()
    log(f"r{r} f{k}: harness done")
    return H


# ------------------------------------------------------------------------------------------ part (2), once per arm
def guards(metrics):
    """eval_seq.compare()'s guards between the env and the trained run (copied)."""
    from seq_data import extra_vt_meta, extra_ti_meta, expanding_norm_enabled
    if extra_vt_meta()[0] != int(metrics.get("extra_vt_channels", 0) or 0):
        raise SystemExit("FATAL: CES_SEQ_EXTRA_VT does not match the run")
    if extra_ti_meta()[0] != int(metrics.get("extra_ti_channels", 0) or 0):
        raise SystemExit("FATAL: CES_SEQ_EXTRA_TI does not match the run")
    env_exp = os.getenv("CES_PER_SHOT_NORM", "0") == "1" and expanding_norm_enabled()
    if env_exp != bool(metrics.get("per_shot_norm_expanding", False)):
        raise SystemExit("FATAL: CES_SEQ_NORM_EXPANDING does not match the training treatment.")
    if (os.getenv("CES_SEQ_ZERO_MC", "0") == "1") != bool(metrics.get("zero_mc", False)):
        raise SystemExit("FATAL: CES_SEQ_ZERO_MC does not match the training treatment.")
    if int(os.getenv("CES_SEQ_EVAL_CTX", "0")) != (metrics.get("train_ctx") or 0):
        raise SystemExit("FATAL: CES_SEQ_EVAL_CTX does not match train_ctx")


def lookup(fam, arm, H, out):
    import torch
    import eval_seq as E
    metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    guards(metrics)
    my_stats = E._load_stats(metrics)
    device = torch.device(os.getenv("CES_SEQ_DEVICE", "cpu"))
    if device.type != "cpu":
        raise SystemExit("FATAL: CPU only")
    restore = None
    if fam == "b59" and arm == "L":
        import fast_shift as FS
        S = FS.read_shift()
        trained = json.loads((out / "b59_shift.json").read_text(encoding="utf-8"))["fast_shift_rows"]
        if trained != S:
            raise SystemExit(f"FATAL: trained with a shift of {trained}, scored with {S}")
        import seq_data
        restore = (E.build_blocks, seq_data.build_blocks)
        FS.install_eval(S)
    if fam == "b67":
        import model_seq_v2gate as M
        M.register()
        if metrics["seq_model"] != os.environ["CES_SEQ_MODEL"]:
            raise SystemExit("FATAL: the B.67 run was trained as another variant")
    try:
        lk = E.build_pred_lookup(H["data_dir"], H["eval_files"], my_stats, device, [], 0)
    finally:
        if restore is not None:
            import seq_data
            E.build_blocks, seq_data.build_blocks = restore
    tab = lk["full"]
    model_phys = np.zeros((len(H["names"]), 2), dtype=np.float32)
    for j, (nm, key) in enumerate(zip(H["names"], H["keys"])):
        try:
            model_phys[j] = tab[nm][key]
        except KeyError:
            raise SystemExit(f"FATAL: no seq prediction for {nm} @ t={key} -- grid mismatch.")
    return model_phys


def finish(H, model_phys, out):
    """Copied from eval_seq.compare() after the loop (contexts empty, primary_ctx 0)."""
    import eval_seq as E
    target_phys, keep, base_phys = H["target_phys"], H["keep"], H["base_phys"]
    dt_ms, shot_ids, row_ids, methods = H["dt_ms"], H["shot_ids"], H["row_ids"], H["methods"]
    n_future, n_total = H["n_future"], H["n_total"]
    split_tag = H["split_tag"]
    report = {
        "split": split_tag,
        "eval_samples": H["eval_samples"],
        "window_size": H["window_size"],
        "eval_ctx": None,
        "headline_baseline": E.HEADLINE_BASELINE,
        "units": "physical CES (raw CSV units)",
        "note": "held-out TEST split (seq-LSTM full-grid masked-loss experiment)",
        "mnar_caveat": "skill measured on observed CES points only (MNAR optimistic bound)",
        "per_target": {},
    }
    HB = E.HEADLINE_BASELINE
    boot = {}
    for t, name in enumerate(E.TARGET_NAMES):
        k = keep[:, t]
        valid = k.copy()
        for m in methods:
            valid &= ~np.isnan(base_phys[m][:, t])
        n = int(valid.sum())
        is_peak = H["is_peak"][t]
        y = target_phys[valid, t]
        err_model = model_phys[valid, t] - y
        rmse_model = E._rmse(err_model)
        mse_model = float(np.mean(err_model ** 2))
        per_method = {}
        for m in methods:
            eb = base_phys[m][valid, t] - y
            per_method[m] = {"rmse": E._rmse(eb), "mse": float(np.mean(eb ** 2))}
        mse_head = per_method[HB]["mse"]
        skill_head = 1.0 - mse_model / mse_head if mse_head > 0 else float("nan")
        report["per_target"][name] = {
            "n": n,
            "rmse_model": rmse_model,
            "baselines": per_method,
            f"skill_vs_{HB}": skill_head,
            f"beats_{HB}": bool(rmse_model < per_method[HB]["rmse"]),
            "future_neighbor_fraction": float(n_future[t] / max(int(n_total[t]), 1)),
            "peak_input_only_ces_activity_n": int(is_peak[valid].sum()),
        }
        d = dt_ms[valid, t]
        em2 = err_model ** 2
        ep2 = {bl: (base_phys[bl][valid, t] - y) ** 2 for bl in ("pchip", "linear")}
        edges = list(E.BIN_EDGES_MS) + [float("inf")]
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
        boot[f"{name}_se_persistence"] = (base_phys["persistence"][valid, t] - y) ** 2
        boot[f"{name}_se_gp"] = (base_phys["gp"][valid, t] - y) ** 2
        boot[f"{name}_se_gp_causal"] = (base_phys["gp_causal"][valid, t] - y) ** 2
        boot[f"{name}_y_true"] = y
        boot[f"{name}_is_peak"] = is_peak[valid]
        boot[f"{name}_row"] = row_ids[valid]
    err_path = out / f"comparison_errors_{split_tag}.npz"
    np.savez(err_path, **boot)
    (out / f"comparison_metrics_{split_tag}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    out_path = out / "comparison_metrics.json"
    if split_tag == "test" or not out_path.exists():
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def score_fold(r, k, cap=0, root=RB.OUT_ROOT, families=tuple(RB.FAMILIES)):
    todo = [(fam, arm) for fam in families for arm in RB.FAMILIES[fam]
            if not (RB.new_dir(fam, arm, r, k, cap, root) / "comparison_errors_test.npz").exists()]
    if not todo:
        return f"r{r} f{k}: all arms exist"
    t0 = _time.time()
    H = harness(r, k, cap)
    t_h = _time.time() - t0
    recs = []
    for fam, arm in todo:
        t1 = _time.time()
        out = RB.new_dir(fam, arm, r, k, cap, root)
        RB.prepare(fam, arm, r, k, out)
        env = RB.rescore_env(fam, arm, r, k, out, cap)
        diff = RB.env_diff(fam, arm, r, k, cap)
        if not set(diff) <= set(RB.CHANGED):
            raise SystemExit(f"FATAL: {out.name}: env differs from the stored run's beyond {RB.CHANGED}: {diff}")
        set_env(env)
        rep = finish(H, lookup(fam, arm, H, out), out)
        rec = {"run": out.name, "family": fam, "arm": arm, "repeat": r, "fold": k, "cap": cap,
               "stored": RB.stored_dir(fam, arm, r, k).name, "status": "ok", "scorer": "score_b70 (shared references)",
               "seconds_arm": round(_time.time() - t1, 1), "seconds_harness_of_fold": round(t_h, 1),
               "env_changed": diff, "ti_rows": rep["per_target"]["CES_TI"]["n"]}
        (out / "b70_rescore.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
        recs.append(rec)
        log(f"r{r} f{k} {fam} {arm}: {rep['per_target']['CES_TI']['n']} T_i rows, rmse {rep['per_target']['CES_TI']['rmse_model']:.2f} "
            f"({rec['seconds_arm']} s)")
    log_p = root / "score.json"
    old = json.loads(log_p.read_text(encoding="utf-8")) if log_p.exists() else []
    names = {x["run"] for x in recs}
    log_p.write_text(json.dumps([x for x in old if x["run"] not in names] + recs, indent=1), encoding="utf-8")
    return f"r{r} f{k}: {len(recs)} arms, harness {t_h:.0f} s, total {_time.time() - t0:.0f} s"


def validate():
    """r2 f0: score every arm with this scorer into data/.b70_rescore_check/ and compare every key with the
    output of the arm's own scoring script in data/.b70_rescore/."""
    msg = score_fold(2, 0, 0, CHECK_ROOT)
    log(msg)
    res = {}
    for fam, arms in RB.FAMILIES.items():
        for arm in arms:
            a = np.load(RB.new_dir(fam, arm, 2, 0, 0, CHECK_ROOT) / "comparison_errors_test.npz")
            b = np.load(RB.new_dir(fam, arm, 2, 0, 0) / "comparison_errors_test.npz")
            keys = sorted(set(a.files) | set(b.files))
            bad = [kk for kk in keys if kk not in a.files or kk not in b.files or not np.array_equal(a[kk], b[kk])]
            ja = json.loads((RB.new_dir(fam, arm, 2, 0, 0, CHECK_ROOT) / "comparison_metrics_test.json").read_text(encoding="utf-8"))
            jb = json.loads((RB.new_dir(fam, arm, 2, 0, 0) / "comparison_metrics_test.json").read_text(encoding="utf-8"))
            res[f"{fam} {arm}"] = {"keys": len(keys), "keys_not_bit_identical": bad, "metrics_json_equal": ja == jb}
    ok = all(not v["keys_not_bit_identical"] and v["metrics_json_equal"] for v in res.values())
    (CHECK_ROOT / "validation.json").write_text(json.dumps({"passes": ok, "arms": res}, indent=1), encoding="utf-8")
    log(f"validation against the arms' own scoring scripts on r2 f0: passes {ok}")
    for kk, v in res.items():
        log(f"  {kk}: {v}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", nargs="*", default=[])
    ap.add_argument("--cap", type=int, default=0)
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--families", nargs="*", default=list(RB.FAMILIES))
    args = ap.parse_args()
    if not any(RB.DATA.glob("s*.csv")):
        raise SystemExit("FATAL: no shot CSVs in data/ -- real data required, aborting.")
    if args.validate:
        sys.exit(0 if validate() else 1)
    for f in args.folds:
        r, k = f[1:].split("f")
        log(score_fold(int(r), int(k), args.cap, RB.OUT_ROOT, tuple(args.families)))


if __name__ == "__main__":
    main()
    os._exit(0)
