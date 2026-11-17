"""B.38 runner (PREREGISTRATION_B38.md with amendment 1): out-of-fold networks of the causal variant.

For each file-level split (seeds 42, 1, 7, 123; cut population) the training discharges are
divided into four folds and the causal variant is trained on three of them, with the split's
validation discharges for early stopping.  16 runs, GPU.  The environment is that of the B.29
causal-variant runs (`run_b29.run_env("norm0", "split", "cut", seed, ...)`); only the split
directory and the output directory differ.  Nothing is written outside `data/.b38_*`
(smoke: `data/.smoke_b38_*`); every run is skipped when its artifacts exist.

Usage (repo root):
  py ces_prediction/experiments/b38_stage2/run_b38.py --smoke     # one epoch, 40 files, GPU plumbing
  py ces_prediction/experiments/b38_stage2/run_b38.py             # 16 runs (GPU)
  py ces_prediction/experiments/b38_stage2/analyze_b38.py         # read-outs (CPU)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "b29_causal"))
import run_b29 as B29  # noqa: E402
from run_b29 import DATA, SEQ_DIR, run_step  # noqa: E402

SEEDS = (42, 1, 7, 123)
N_FOLDS = 4
RUNS_OUT = DATA / ".b38_runs.json"


def base_manifest(seed):
    return json.loads((DATA / f".b1_w2cut_split_s{seed}" / "split_manifest.json").read_text(encoding="utf-8"))


def folds(seed):
    train = sorted(base_manifest(seed)["train_files"])
    perm = np.random.default_rng(seed).permutation(len(train))
    return [sorted(train[i] for i in perm[k::N_FOLDS]) for k in range(N_FOLDS)]


def split_dir(seed, k, smoke=False):
    return DATA / (f".smoke_b38_split_s{seed}_f{k}" if smoke else f".b38_split_s{seed}_f{k}")


def run_dir(seed, k, smoke=False):
    return DATA / (f".smoke_b38_norm0_cut_s{seed}_f{k}" if smoke else f".b38_norm0_cut_s{seed}_f{k}")


def write_split(seed, k, smoke):
    man, fs = base_manifest(seed), folds(seed)
    held = set(fs[k])
    out = {"train_files": [f for f in sorted(man["train_files"]) if f not in held],
           "val_files": list(man["val_files"]), "test_files": sorted(held),
           "source": f".b1_w2cut_split_s{seed}", "fold": k, "n_folds": N_FOLDS}
    if set(out["train_files"]) & (set(man["test_files"]) | set(man["val_files"]) | held):
        raise SystemExit("FATAL: fold training files overlap validation, test or the held-out fold")
    d = split_dir(seed, k, smoke)
    d.mkdir(parents=True, exist_ok=True)
    (d / "split_manifest.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    return d


def one_run(seed, k, smoke):
    out = run_dir(seed, k, smoke)
    out.mkdir(parents=True, exist_ok=True)
    env = B29.run_env("norm0", "split", "cut", seed, out, False)
    env["CES_SPLIT_DIR"] = str(write_split(seed, k, smoke))
    if smoke:
        env.update({"CES_SEQ_EPOCHS": "1", "CES_SEQ_MAX_FILES": "40"})
    weights, metrics = out / "weights" / "seq_lstm.pth", out / "metrics.json"
    rec = {"seed": seed, "fold": k, "out": str(out), "status": "ok"}
    t0 = time.time()
    print(f"[b38] === {out.name}", flush=True)
    if not (weights.exists() and metrics.exists()):
        if not run_step([SEQ_DIR / "train_seq.py"], env, out / "run.log", "train", artifacts=(weights, metrics)):
            rec["status"] = "train_failed"
    if metrics.exists():
        m = json.loads(metrics.read_text(encoding="utf-8"))
        rec.update({"epochs_run": m["epochs_run"], "best_epoch": m["best_epoch"], "n_params": m["n_params"],
                    "train_blocks": m["train_blocks"], "best_val_masked_mse": m["best_val_masked_mse"]})
    rec["minutes"] = round((time.time() - t0) / 60, 1)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seeds", nargs="*", type=int, default=list(SEEDS))
    ap.add_argument("--min-free-mb", type=int, default=3000)
    args = ap.parse_args()
    if not any(DATA.glob("s*.csv")):
        raise SystemExit("FATAL: no shot CSVs in data/ -- real data required, aborting.")
    free = B29.gpu_free_mb()
    print(f"[b38] GPU free memory: {free} MiB", flush=True)
    if free is None or free < args.min_free_mb:
        raise SystemExit(f"FATAL: GPU free memory {free} MiB < {args.min_free_mb} MiB -- the GPU is busy or "
                         "absent; B.38 is never run on CPU.")
    runs = [(42, 0)] if args.smoke else [(s, k) for s in args.seeds for k in range(N_FOLDS)]
    t0, records = time.time(), []
    B29.keep_awake(True)
    try:
        for i, (s, k) in enumerate(runs, 1):
            rec = one_run(s, k, args.smoke)
            records.append(rec)
            print(f"[b38] {i}/{len(runs)} {rec}", flush=True)
            if not args.smoke:
                RUNS_OUT.write_text(json.dumps(records, indent=1), encoding="utf-8")
    finally:
        B29.keep_awake(False)
        print(f"[b38] wall time: {(time.time() - t0) / 3600:.2f} h for {len(records)} runs", flush=True)
    bad = [r for r in records if r["status"] != "ok"]
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
