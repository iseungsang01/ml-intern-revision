"""B.39 runner (PREREGISTRATION_B39.md): the causal variant as the primary estimator under a
symmetric protocol, and ablations of the fast-channel groups.

Stages, in the registered order:
  1  four file-level splits, initialisation = split seed: causal variant, ECEI-only, BES-only, no-Mirnov   16 runs
  2  four file-level splits, two further initialisations: causal variant and history-only twin             16 runs
  3  pooled cross-validation, repeats r2..r9 x five folds: causal variant and history-only twin            80 runs

Every run: train_seq -> eval_seq (TEST). The environment is `run_b29.run_env` (confirmed protocol,
every CES_* variable pinned) with the epoch cap raised to 80; the networks that read fast channels
use CES_PER_SHOT_NORM=0 (statistics of the training discharges), the history-only twin keeps the
environment of its existing cap-80 runs (`.b29a1_v2h_cut_s{seed}`), which reads no fast channel.
Nothing is written outside `data/.b39_*` / `data/.b39p_*` (smoke: `data/.smoke_b39_*`); a run
whose artifacts exist is skipped, so a re-launch resumes.

Usage (repo root):
  py ces_prediction/experiments/b39_primary/run_b39.py --smoke
  py ces_prediction/experiments/b39_primary/run_b39.py [--stages 1 2 3]
  py ces_prediction/experiments/b39_primary/analyze_b39.py
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "b29_causal"))
sys.path.insert(0, str(HERE.parent / "b27_kfold"))
import run_b29 as B29  # noqa: E402
import run_b27 as B27  # noqa: E402
from run_b29 import DATA, SEQ_DIR, run_step  # noqa: E402

SEEDS = (42, 1, 7, 123)
EPOCH_CAP = "80"
ABLATIONS = ("v2e", "v2b", "v2n")
REPEATS = tuple(range(2, 10))
RUNS_OUT = DATA / ".b39_runs.json"


def inits(seed):
    i = SEEDS.index(seed)
    return tuple(SEEDS[(i + j) % len(SEEDS)] for j in range(3))


def split_run_dir(model, seed, init, smoke=False):
    return DATA / (f".smoke_b39_{model}_cut_s{seed}_i{init}" if smoke else f".b39_{model}_cut_s{seed}_i{init}")


def pooled_run_dir(model, r, k):
    return DATA / f".b39p_{model}_cut_r{r}_f{k}"


def twin_dir(seed, init):
    """The twin with initialisation = split seed exists (B.29 addendum 1) and is reused."""
    return DATA / f".b29a1_v2h_cut_s{seed}" if init == seed else split_run_dir("v2h", seed, init)


def base_env(model, seed, out):
    arm = "v2h" if model == "v2h" else "norm0"
    return B29.run_env(arm, "split", "cut", seed, out, False, epochs=EPOCH_CAP, model=model)


def split_env(model, seed, init, out, smoke=False):
    env = base_env(model, seed, out)
    env["CES_INIT_SEED"] = str(init)
    if smoke:
        env.update(B29.SMOKE_OVERRIDES)
        env.update({"CES_SEQ_EPOCHS": "1", "CES_SEQ_MAX_FILES": "40"})
    return env


def pooled_env(model, r, k, out):
    split, ctrl = B27.split_dir(r, k), B27.ctrl_dir("cut", r, k) / "metrics.json"
    for needed in (split / "split_manifest.json", ctrl):
        if not needed.exists():
            raise SystemExit(f"FATAL: missing frozen B.27 artifact {needed}")
    env = base_env(model, 42, out)
    env.update({"CES_SEED": str(B27.REPEAT_SEEDS[r]), "CES_INIT_SEED": str(B27.init_seed(r, k)),
                "CES_SPLIT_DIR": str(split), "CES_CONTROL_METRICS": str(ctrl)})
    return env


def execute(out, env, label):
    out.mkdir(parents=True, exist_ok=True)
    weights, metrics = out / "weights" / "seq_lstm.pth", out / "metrics.json"
    npz, cm = out / "comparison_errors_test.npz", out / "comparison_metrics.json"
    rec = {"run": out.name, "status": "ok"}
    t0 = time.time()
    print(f"[b39] === {out.name}", flush=True)
    for name, cmd, artifacts in (("train", [SEQ_DIR / "train_seq.py"], (weights, metrics)),
                                 ("eval(test)", [SEQ_DIR / "eval_seq.py"], (npz, cm))):
        if all(a.exists() for a in artifacts):
            continue
        if not run_step(cmd, env, out / "run.log", name, artifacts=artifacts):
            rec["status"] = f"{name}_failed"
            break
    if metrics.exists():
        m = json.loads(metrics.read_text(encoding="utf-8"))
        rec.update({"epochs_run": m["epochs_run"], "best_epoch": m["best_epoch"], "n_params": m["n_params"],
                    "cap_bound": m["epochs_run"] >= int(env["CES_SEQ_EPOCHS"])})
    rec.update({"label": label, "minutes": round((time.time() - t0) / 60, 1)})
    return rec


def plan(stages):
    runs = []
    if 1 in stages:
        for s in SEEDS:
            for model in ("v2", *ABLATIONS):
                runs.append((f"A s{s} {model} i{s}", split_run_dir(model, s, s), lambda m=model, s=s: split_env(m, s, s, split_run_dir(m, s, s))))
    if 2 in stages:
        for s in SEEDS:
            for init in inits(s)[1:]:
                for model in ("v2", "v2h"):
                    runs.append((f"A s{s} {model} i{init}", split_run_dir(model, s, init),
                                 lambda m=model, s=s, i=init: split_env(m, s, i, split_run_dir(m, s, i))))
    if 3 in stages:
        for r in REPEATS:
            for k in range(B27.K):
                for model in ("v2", "v2h"):
                    runs.append((f"P r{r} f{k} {model}", pooled_run_dir(model, r, k),
                                 lambda m=model, r=r, k=k: pooled_env(m, r, k, pooled_run_dir(m, r, k))))
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="one epoch, 40 files, GPU plumbing (v2e, split 42)")
    ap.add_argument("--stages", nargs="*", type=int, default=[1, 2, 3], choices=[1, 2, 3])
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--min-free-mb", type=int, default=2000)
    args = ap.parse_args()
    if not any(DATA.glob("s*.csv")):
        raise SystemExit("FATAL: no shot CSVs in data/ -- real data required, aborting.")
    for s in SEEDS:
        ref = twin_dir(s, s) / "comparison_errors_test.npz"
        m = json.loads((twin_dir(s, s) / "metrics.json").read_text(encoding="utf-8"))
        if not ref.exists() or m["seq_model"] != "v2h" or m["init_seed"] != s:
            raise SystemExit(f"FATAL: the reused twin {twin_dir(s, s).name} is missing or not the registered run")
    runs = plan(args.stages)
    if args.list:
        for label, out, _ in runs:
            print(label, out.name)
        print(len(runs), "runs")
        return
    free = B29.gpu_free_mb()
    print(f"[b39] GPU free memory: {free} MiB", flush=True)
    if free is None or free < args.min_free_mb:
        raise SystemExit(f"FATAL: GPU free memory {free} MiB < {args.min_free_mb} MiB -- B.39 is never run on CPU.")
    if args.smoke:
        out = split_run_dir("v2e", 42, 42, smoke=True)
        print(execute(out, split_env("v2e", 42, 42, out, smoke=True), "smoke"), flush=True)
        return
    t0, records = time.time(), (json.loads(RUNS_OUT.read_text(encoding="utf-8")) if RUNS_OUT.exists() else [])
    done = {r["run"] for r in records if r["status"] == "ok"}
    B29.keep_awake(True)
    try:
        for i, (label, out, env_fn) in enumerate(runs, 1):
            if out.name in done and (out / "comparison_errors_test.npz").exists():
                continue
            rec = execute(out, env_fn(), label)
            records = [r for r in records if r["run"] != rec["run"]] + [rec]
            RUNS_OUT.write_text(json.dumps(records, indent=1), encoding="utf-8")
            print(f"[b39] {i}/{len(runs)} {rec} ({(time.time() - t0) / 3600:.2f} h)", flush=True)
    finally:
        B29.keep_awake(False)
    bad = [r for r in records if r["status"] != "ok"]
    for r in bad:
        print(f"[b39] FAILED: {r['run']} ({r['status']})", flush=True)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
