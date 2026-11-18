"""B.64 runner (PREREGISTRATION_B64.md): out-of-fold networks for the two-stage estimator under the pooled protocol.

For each outer fold of re-partitions r2, r3, r4 of the pooled cross-validation of B.39 (15 outer folds), the outer
fold's training discharges are divided into four inner folds (seeded by the re-partition seed and the fold); the
network (`v2`) and the history-only twin (`v2h`) are each trained on three inner folds, with the outer fold's
validation discharges for early stopping: 15 x 4 x 2 = 120 GPU runs (train only, as in B.38; the held-out inner fold
is predicted on the CPU by stage2_b64.py).

The environment of every run is `run_b39.pooled_env` of its model and outer fold (cut, cap 80 with early stopping,
the fold's CES_SEED / CES_INIT_SEED / harness statistics; CES_PER_SHOT_NORM as in B.39: 0 for v2, 1 for the twin,
which reads no fast channel); only CES_SPLIT_DIR (the inner split) and CES_OUTPUT_DIR differ.  Nothing is written
outside `data/.b64_*` (smoke: `data/.smoke_b64_*`); a run whose weights and metrics exist is skipped, so a re-launch
resumes.  The batch never starts while a run_b59.py, run_b47.py or run_b53.py process is alive.

Usage (repo root):
  py ces_prediction/experiments/b64_two_stage/run_b64.py --dry-run              # plan, env diff, overlap check
  py ces_prediction/experiments/b64_two_stage/run_b64.py --smoke                # GPU plumbing: 1 epoch, 40 files
  py ces_prediction/experiments/b64_two_stage/run_b64.py [--wait-for-earlier] > data/.b64_batch.log 2>&1
  py ces_prediction/experiments/b64_two_stage/stage2_b64.py all                 # CPU second stage + readouts
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "b39_primary"))
import run_b39 as B39  # noqa: E402  (adds b29_causal and b27_kfold to sys.path)
from run_b39 import B27, B29, DATA, SEQ_DIR, run_step  # noqa: E402

REPEATS = (2, 3, 4)
N_INNER = 4
MODELS = ("v2", "v2h")
RUNS_OUT = DATA / ".b64_runs.json"
SMOKE_RUNS = ((2, 0, 0, "v2"), (2, 0, 0, "v2h"))
POLL_S, STATUS_EVERY_S, GPU_IDLE_MIB, GPU_IDLE_POLLS = 60, 600, 1000, 3
GUARDED = ("run_b59.py", "run_b47.py", "run_b53.py")


# ------------------------------------------------------------------------------ folds / directories / env
def outer_manifest(r, k):
    return json.loads((B27.split_dir(r, k) / "split_manifest.json").read_text(encoding="utf-8"))


def inner_folds(r, k):
    """Four inner folds of the outer fold's training discharges; seed = (re-partition seed, outer fold)."""
    train = sorted(outer_manifest(r, k)["train_files"])
    perm = np.random.default_rng([B27.REPEAT_SEEDS[r], k]).permutation(len(train))
    return [sorted(train[i] for i in perm[j::N_INNER]) for j in range(N_INNER)]


def split_dir(r, k, j, smoke=False):
    return DATA / (f".smoke_b64_split_r{r}_f{k}_i{j}" if smoke else f".b64_split_r{r}_f{k}_i{j}")


def run_dir(model, r, k, j, smoke=False):
    return DATA / (f".smoke_b64_{model}_cut_r{r}_f{k}_i{j}" if smoke else f".b64_{model}_cut_r{r}_f{k}_i{j}")


def inner_manifest(r, k, j):
    man, fs = outer_manifest(r, k), inner_folds(r, k)
    held = set(fs[j])
    return {"train_files": [f for f in sorted(man["train_files"]) if f not in held],
            "val_files": sorted(man["val_files"]), "test_files": sorted(held),
            "source": str(B27.split_dir(r, k).name), "outer_test_files": sorted(man["test_files"]),
            "repeat": r, "outer_fold": k, "inner_fold": j, "n_inner": N_INNER,
            "inner_seed": [int(B27.REPEAT_SEEDS[r]), k]}


def overlap_check(r, k):
    """No file in two of: inner training, inner held-out, validation, test; the inner folds partition the outer
    training discharges.  Returns a dict; raises on any violation."""
    man, fs = outer_manifest(r, k), inner_folds(r, k)
    tr, va, te = set(man["train_files"]), set(man["val_files"]), set(man["test_files"])
    if tr & va or tr & te or va & te:
        raise SystemExit(f"FATAL: r{r} f{k}: the outer manifest itself overlaps")
    allf = [f for x in fs for f in x]
    if len(allf) != len(set(allf)) or set(allf) != tr:
        raise SystemExit(f"FATAL: r{r} f{k}: the inner folds do not partition the outer training discharges")
    for j in range(N_INNER):
        m = inner_manifest(r, k, j)
        itr, iho = set(m["train_files"]), set(m["test_files"])
        for a, b, what in ((itr, iho, "inner training / inner held-out"), (itr, va, "inner training / validation"),
                           (itr, te, "inner training / test"), (iho, va, "inner held-out / validation"),
                           (iho, te, "inner held-out / test")):
            if a & b:
                raise SystemExit(f"FATAL: r{r} f{k} i{j}: {what} share {sorted(a & b)[:3]}")
        if itr | iho != tr:
            raise SystemExit(f"FATAL: r{r} f{k} i{j}: inner training + held-out != outer training")
    return {"outer_train": len(tr), "val": len(va), "test": len(te), "inner_sizes": [len(x) for x in fs]}


def write_split(r, k, j, smoke):
    overlap_check(r, k)
    d = split_dir(r, k, j, smoke)
    d.mkdir(parents=True, exist_ok=True)
    (d / "split_manifest.json").write_text(json.dumps(inner_manifest(r, k, j), indent=1), encoding="utf-8")
    return d


def run_env(model, r, k, j, smoke=False, write=True):
    """run_b39.pooled_env of the model on outer fold (r, k); the inner split and the output directory replace
    CES_SPLIT_DIR and CES_OUTPUT_DIR.  Smoke: one epoch, 40 files (run_b39.split_env's smoke overrides)."""
    out = run_dir(model, r, k, j, smoke)
    env = B39.pooled_env(model, r, k, out)
    env["CES_SPLIT_DIR"] = str(write_split(r, k, j, smoke) if write else split_dir(r, k, j, smoke))
    if smoke:
        env.update(B29.SMOKE_OVERRIDES)
        env.update({"CES_SEQ_EPOCHS": "1", "CES_SEQ_MAX_FILES": "40"})
    return env


def plan(smoke=False):
    if smoke:
        return list(SMOKE_RUNS)
    return [(r, k, j, m) for r in REPEATS for k in range(B27.K) for j in range(N_INNER) for m in MODELS]


def done(out):
    return (out / "weights" / "seq_lstm.pth").exists() and (out / "metrics.json").exists()


# ------------------------------------------------------------------------------ guards (copied from run_b59.py)
def guarded_processes():
    """PIDs and command lines of live python processes whose command line names a guarded runner."""
    ps = ("Get-CimInstance Win32_Process | Where-Object { $_.Name -like 'py*' } | "
          "ForEach-Object { \"$($_.ProcessId)`t$($_.CommandLine)\" }")
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", ps],
                             capture_output=True, text=True, timeout=120).stdout
    except Exception as e:  # noqa: BLE001
        raise SystemExit(f"FATAL: could not list processes ({e}); refusing to start") from e
    me = os.getpid()
    hits = []
    for line in out.splitlines():
        pid, _, cmd = line.partition("\t")
        if any(g in cmd for g in GUARDED) and pid.strip().isdigit() and int(pid) != me:
            hits.append((int(pid), cmd.strip()))
    return hits


def gpu_used_mib():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=30).stdout
        return int(out.strip().splitlines()[0])
    except Exception:  # noqa: BLE001
        return None


def wait_for_earlier():
    t0 = last = time.time()
    idle = 0
    print(f"[b64] waiting for {', '.join(GUARDED)} to end (no such process, GPU memory used < "
          f"{GPU_IDLE_MIB} MiB on {GPU_IDLE_POLLS} consecutive polls)", flush=True)
    while True:
        procs = guarded_processes()
        used = gpu_used_mib()
        idle = idle + 1 if (not procs and used is not None and used < GPU_IDLE_MIB) else 0
        if idle >= GPU_IDLE_POLLS:
            print(f"[b64] earlier batches have ended ({(time.time() - t0) / 3600:.2f} h waited; GPU used {used} MiB)", flush=True)
            return
        if time.time() - last >= STATUS_EVERY_S:
            last = time.time()
            print(f"[b64] {time.strftime('%Y-%m-%d %H:%M')} waiting: guarded processes {len(procs)}, "
                  f"GPU used {used} MiB, idle polls {idle}", flush=True)
        time.sleep(POLL_S)


def refuse_if_earlier_alive():
    procs = guarded_processes()
    if procs:
        raise SystemExit(f"FATAL: {sorted({g for _, c in procs for g in GUARDED if g in c})} alive (pids "
                         f"{[p for p, _ in procs]}); B.64 never runs beside them. Use --wait-for-earlier.")


def preflight(runs, live):
    if not any(DATA.glob("s*.csv")):
        raise SystemExit("FATAL: no shot CSVs in data/ -- real data required, aborting.")
    for r in REPEATS:
        for k in range(B27.K):
            for m in MODELS:
                d = B39.pooled_run_dir(m, r, k)
                if not ((d / "comparison_errors_test.npz").exists() and (d / "weights" / "seq_lstm.pth").exists()):
                    raise SystemExit(f"FATAL: the B.39 outer-fold run {d.name} is missing")
    for (r, k, j, m) in runs:
        env = B39.pooled_env(m, r, k, run_dir(m, r, k, j))
        if not Path(env["CES_CONTROL_METRICS"]).exists():
            raise SystemExit(f"FATAL: missing {env['CES_CONTROL_METRICS']}")
    if live and os.environ.get("CUDA_VISIBLE_DEVICES", None) == "":
        raise SystemExit("FATAL: CUDA_VISIBLE_DEVICES is empty in the shell; the runs would not see the GPU")


# ------------------------------------------------------------------------------ one run
def execute(r, k, j, model, smoke):
    out = run_dir(model, r, k, j, smoke)
    out.mkdir(parents=True, exist_ok=True)
    env = run_env(model, r, k, j, smoke)
    if not Path(env["CES_OUTPUT_DIR"]).name.startswith(".smoke_b64_" if smoke else ".b64_"):
        raise SystemExit("FATAL: output directory outside data/.b64_*")
    weights, metrics = out / "weights" / "seq_lstm.pth", out / "metrics.json"
    rec = {"run": out.name, "model": model, "repeat": r, "outer_fold": k, "inner_fold": j,
           "init": int(env["CES_INIT_SEED"]), "status": "ok"}
    t0 = time.time()
    print(f"[b64] === {out.name}", flush=True)
    if not done(out):
        if not run_step([SEQ_DIR / "train_seq.py"], env, out / "run.log", "train", artifacts=(weights, metrics)):
            rec["status"] = "train_failed"
    if metrics.exists():
        m = json.loads(metrics.read_text(encoding="utf-8"))
        rec.update({"epochs_run": m["epochs_run"], "best_epoch": m["best_epoch"], "n_params": m["n_params"],
                    "train_blocks": m["train_blocks"], "best_val_masked_mse": m["best_val_masked_mse"],
                    "seq_model": m["seq_model"], "cap_bound": m["epochs_run"] >= int(env["CES_SEQ_EPOCHS"])})
        if rec["status"] == "ok" and m["seq_model"] != model:
            rec["status"] = "model_mismatch"
    rec["minutes"] = round((time.time() - t0) / 60, 1)
    return rec


# ------------------------------------------------------------------------------ dry run
def ces_vars(env):
    return {k: v for k, v in env.items() if k.startswith("CES_")}


def env_diff(mine, ref):
    keys = sorted(set(mine) | set(ref))
    return [(k, ref.get(k, "<unset>"), mine.get(k, "<unset>")) for k in keys if mine.get(k) != ref.get(k)]


def dry_run(runs):
    print("--- no-overlap check (inner training / inner held-out / validation / test), every outer fold")
    for r in REPEATS:
        for k in range(B27.K):
            print(f"    r{r} f{k}: {overlap_check(r, k)} -> passes")
    print("\n--- the planned runs")
    for i, (r, k, j, m) in enumerate(runs, 1):
        man = inner_manifest(r, k, j)
        out = run_dir(m, r, k, j)
        print(f"{i:3d} {out.name:28s} | model {m:3s} | r{r} f{k} inner {j} | split {split_dir(r, k, j).name} "
              f"(train {len(man['train_files'])}, val {len(man['val_files'])}, inner held-out {len(man['test_files'])}) | "
              f"init {run_env(m, r, k, j, write=False)['CES_INIT_SEED']} | artifacts exist: {done(out)}")
    print(f"{len(runs)} runs: " + ", ".join(f"{m} {sum(x[3] == m for x in runs)}" for m in MODELS))
    for m in MODELS:
        r, k, j = runs[0][:3]
        mine = ces_vars(run_env(m, r, k, j, write=False))
        ref = ces_vars(B39.pooled_env(m, r, k, B39.pooled_run_dir(m, r, k)))
        print(f"\n--- env of {run_dir(m, r, k, j).name}")
        for key in sorted(mine):
            print(f"    {key}={mine[key]}")
        print(f"  diff against the B.39 outer-fold run {B39.pooled_run_dir(m, r, k).name}:")
        for key, a, b in env_diff(mine, ref):
            print(f"    {key}: {a}  ->  B.64 {b}")
    print("\n--- every run: variables that differ from the B.39 outer-fold run of the same model and fold")
    bad = []
    for (r, k, j, m) in runs:
        d = env_diff(ces_vars(run_env(m, r, k, j, write=False)),
                     ces_vars(B39.pooled_env(m, r, k, B39.pooled_run_dir(m, r, k))))
        keys = sorted(x[0] for x in d)
        if keys != ["CES_OUTPUT_DIR", "CES_SPLIT_DIR"]:
            bad.append((run_dir(m, r, k, j).name, keys))
    print(f"    {len(runs) - len(bad)} of {len(runs)} runs differ in exactly CES_OUTPUT_DIR and CES_SPLIT_DIR" +
          ("" if not bad else f"; OTHERS: {bad}"))
    if bad:
        raise SystemExit("FATAL: an environment differs beyond the split and output directories")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="print the plan, the env diff and the overlap check")
    ap.add_argument("--smoke", action="store_true", help="GPU plumbing: r2 f0 inner 0, v2 and v2h, 1 epoch, 40 files")
    ap.add_argument("--wait-for-earlier", action="store_true",
                    help="wait until no run_b59 / run_b47 / run_b53 process is alive and the GPU is idle")
    ap.add_argument("--min-free-mb", type=int, default=2000)
    args = ap.parse_args()
    runs = plan(args.smoke)
    preflight(runs, live=not args.dry_run)
    if args.dry_run:
        dry_run(plan())
        return
    if args.wait_for_earlier:
        wait_for_earlier()
    refuse_if_earlier_alive()
    free = B29.gpu_free_mb()
    print(f"[b64] GPU free memory: {free} MiB", flush=True)
    if free is None or free < args.min_free_mb:
        raise SystemExit(f"FATAL: GPU free memory {free} MiB < {args.min_free_mb} MiB -- B.64 is never run on CPU.")
    t0 = time.time()
    out_json = DATA / ".smoke_b64_runs.json" if args.smoke else RUNS_OUT
    records = json.loads(out_json.read_text(encoding="utf-8")) if out_json.exists() else []
    B29.keep_awake(True)
    try:
        for i, (r, k, j, m) in enumerate(runs, 1):
            if not args.smoke and done(run_dir(m, r, k, j)) and any(x["run"] == run_dir(m, r, k, j).name and
                                                                    x["status"] == "ok" for x in records):
                continue
            refuse_if_earlier_alive()
            rec = execute(r, k, j, m, args.smoke)
            records = [x for x in records if x["run"] != rec["run"]] + [rec]
            out_json.write_text(json.dumps(records, indent=1), encoding="utf-8")
            print(f"[b64] {i}/{len(runs)} {rec} ({(time.time() - t0) / 3600:.2f} h)", flush=True)
    finally:
        B29.keep_awake(False)
    bad = [x for x in records if x["status"] != "ok"]
    capped = [x["run"] for x in records if x.get("cap_bound")]
    for x in bad:
        print(f"[b64] FAILED: {x['run']} ({x['status']})", flush=True)
    if capped and not args.smoke:
        print(f"[b64] reached the epoch cap (reported): {capped}", flush=True)
    print(f"[b64] wall time {(time.time() - t0) / 3600:.2f} h; next: "
          "py ces_prediction/experiments/b64_two_stage/stage2_b64.py all", flush=True)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
