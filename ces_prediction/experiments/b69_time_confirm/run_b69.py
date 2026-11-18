"""B.69 runner (PREREGISTRATION_B69.md): the two-stage estimator on the temporal split of B.5, a confirmation out of time.

The split is B.5's temporal split in the cut population (`data/.b5_camp_cut_split_s{s}`, s = 42, 1, 7, 123): the test
block is the last 97 discharges by shot number.  Per seed:

  outer  the network (`v2`) and the history-only twin (`v2h`) trained on the seed's training discharges, early stopping
         on its validation discharges, then scored on the test block (train_seq -> eval_seq):        4 x 2 = 8 runs
  inner  the seed's training discharges divided into four inner folds (seed = (split seed, 0), the analogue of B.64's
         (re-partition seed, outer fold)); each model trained on three inner folds with the seed's validation
         discharges for early stopping (train only, as in B.64; the held-out inner fold is predicted on the CPU by
         stage2_b69.py):                                                                             4 x 4 x 2 = 32 runs

Order: the 8 outer runs first (a partial batch still gives the first-stage comparison), then the 32 inner runs.

Environment of every run: `run_b39.pooled_env` of its model (the pooled estimator's env: cut, cap 80 with early stopping,
CES_PER_SHOT_NORM=0 for the network and 1 for the twin, which reads no fast channel; every CES_* variable pinned), taken
on re-partition r2 fold 0 as the reference, with only the split, the seed and the output directory replaced:
CES_SPLIT_DIR (the temporal split, or its inner split), CES_CONTROL_METRICS (the temporal split's harness statistics,
`.b5_camp_cut_win_s{s}`, as B.29 used them; read by eval_seq only), CES_SEED = CES_INIT_SEED = s, CES_OUTPUT_DIR.
Outputs: `data/.b69_{model}_camp_cut_s{s}` (outer), `data/.b69_{model}_camp_cut_s{s}_i{j}` (inner), inner splits
`data/.b69_split_camp_cut_s{s}_i{j}`.  A run whose artifacts exist is skipped (outer: comparison_errors_test.npz;
inner: weights + metrics.json, B.64's rule), so a re-launch resumes.  The batch never runs beside a run_b39, run_b47,
run_b59, run_b64, run_b65, run_b67 or run_b68 process; --wait-for-earlier waits until none is alive and the GPU is idle.

Usage (repo root):
  py ces_prediction/experiments/b69_time_confirm/run_b69.py --dry-run
  py ces_prediction/experiments/b69_time_confirm/run_b69.py --wait-for-earlier > data/.b69_batch.log 2>&1
  py ces_prediction/experiments/b69_time_confirm/stage2_b69.py all        # CPU second stage
  py ces_prediction/experiments/b69_time_confirm/analyze_b69.py [--partial]
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
from run_b39 import B29, DATA, SEQ_DIR, run_step  # noqa: E402

REG = "ces_prediction/experiments/PREREGISTRATION_B69.md"
SEEDS = (42, 1, 7, 123)
MODELS = ("v2", "v2h")
N_INNER = 4
TEST_BLOCK = 97
REF_FOLD = (2, 0)            # run_b39.pooled_env is taken on this fold as the reference env
SWAPPED = ("CES_CONTROL_METRICS", "CES_INIT_SEED", "CES_OUTPUT_DIR", "CES_SEED", "CES_SPLIT_DIR")
RUNS_OUT = DATA / ".b69_runs.json"
POLL_S, STATUS_EVERY_S, GPU_IDLE_MIB, GPU_IDLE_POLLS = 60, 600, 1000, 3
GUARDED = ("run_b39.py", "run_b47.py", "run_b59.py", "run_b64.py", "run_b65.py", "run_b67.py", "run_b68.py")


# ------------------------------------------------------------------------------ split / directories / env
def camp(s):
    """(frozen B.5 backbone run, temporal split dir, harness-statistics metrics.json) of seed s (run_b29.controls)."""
    return B29.controls("camp", "cut", s)


def outer_manifest(s):
    return json.loads((camp(s)[1] / "split_manifest.json").read_text(encoding="utf-8"))


def inner_folds(s):
    """Four inner folds of the seed's training discharges; rng seed = (split seed, 0)."""
    train = sorted(outer_manifest(s)["train_files"])
    perm = np.random.default_rng([s, 0]).permutation(len(train))
    return [sorted(train[i] for i in perm[j::N_INNER]) for j in range(N_INNER)]


def split_dir(s, j):
    return DATA / f".b69_split_camp_cut_s{s}_i{j}"


def run_dir(model, s, j=None):
    return DATA / (f".b69_{model}_camp_cut_s{s}" + ("" if j is None else f"_i{j}"))


def inner_manifest(s, j):
    man, fs = outer_manifest(s), inner_folds(s)
    held = set(fs[j])
    return {"train_files": [f for f in sorted(man["train_files"]) if f not in held],
            "val_files": sorted(man["val_files"]), "test_files": sorted(held),
            "source": camp(s)[1].name, "outer_test_files": sorted(man["test_files"]),
            "seed": s, "inner_fold": j, "n_inner": N_INNER, "inner_seed": [int(s), 0]}


def overlap_check(s):
    """No file in two of: inner training, inner held-out, validation, test; the inner folds partition the training
    discharges.  Returns sizes; raises on any violation."""
    man, fs = outer_manifest(s), inner_folds(s)
    tr, va, te = set(man["train_files"]), set(man["val_files"]), set(man["test_files"])
    if tr & va or tr & te or va & te:
        raise SystemExit(f"FATAL: s{s}: the temporal manifest itself overlaps")
    allf = [f for x in fs for f in x]
    if len(allf) != len(set(allf)) or set(allf) != tr:
        raise SystemExit(f"FATAL: s{s}: the inner folds do not partition the training discharges")
    for j in range(N_INNER):
        m = inner_manifest(s, j)
        itr, iho = set(m["train_files"]), set(m["test_files"])
        for a, b, what in ((itr, iho, "inner training / inner held-out"), (itr, va, "inner training / validation"),
                           (itr, te, "inner training / test"), (iho, va, "inner held-out / validation"),
                           (iho, te, "inner held-out / test")):
            if a & b:
                raise SystemExit(f"FATAL: s{s} i{j}: {what} share {sorted(a & b)[:3]}")
        if itr | iho != tr:
            raise SystemExit(f"FATAL: s{s} i{j}: inner training + held-out != training")
    return {"train": len(tr), "val": len(va), "test": len(te), "inner_sizes": [len(x) for x in fs]}


def write_split(s, j):
    overlap_check(s)
    d = split_dir(s, j)
    d.mkdir(parents=True, exist_ok=True)
    (d / "split_manifest.json").write_text(json.dumps(inner_manifest(s, j), indent=1), encoding="utf-8")
    return d


def ref_env(model, out):
    """The pooled estimator's env of this model (run_b39.pooled_env on REF_FOLD)."""
    return B39.pooled_env(model, *REF_FOLD, out)


def run_env(model, s, j=None, write=True):
    out = run_dir(model, s, j)
    env = ref_env(model, out)
    _, split, cm = camp(s)
    if j is not None:
        split = write_split(s, j) if write else split_dir(s, j)
    env.update({"CES_SPLIT_DIR": str(split), "CES_CONTROL_METRICS": str(cm), "CES_SEED": str(s), "CES_INIT_SEED": str(s)})
    return env


def plan():
    """(kind, model, seed, inner fold or None) in the registered order: outer runs, then inner runs."""
    outer = [("outer", m, s, None) for s in SEEDS for m in MODELS]
    inner = [("inner", m, s, j) for s in SEEDS for j in range(N_INNER) for m in MODELS]
    return outer + inner


def done(kind, out):
    if kind == "outer":
        return (out / "comparison_errors_test.npz").exists() and (out / "comparison_metrics.json").exists()
    return (out / "weights" / "seq_lstm.pth").exists() and (out / "metrics.json").exists()


# ------------------------------------------------------------------------------ guards (copied from run_b67.py)
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
    print(f"[b69] waiting for {', '.join(GUARDED)} to end (no such process, GPU memory used < "
          f"{GPU_IDLE_MIB} MiB on {GPU_IDLE_POLLS} consecutive polls)", flush=True)
    while True:
        procs = guarded_processes()
        used = gpu_used_mib()
        idle = idle + 1 if (not procs and used is not None and used < GPU_IDLE_MIB) else 0
        if idle >= GPU_IDLE_POLLS:
            print(f"[b69] earlier batches have ended ({(time.time() - t0) / 3600:.2f} h waited; GPU used {used} MiB)", flush=True)
            return
        if time.time() - last >= STATUS_EVERY_S:
            last = time.time()
            print(f"[b69] {time.strftime('%Y-%m-%d %H:%M')} waiting: guarded processes {len(procs)}, "
                  f"GPU used {used} MiB, idle polls {idle}", flush=True)
        time.sleep(POLL_S)


def refuse_if_earlier_alive():
    procs = guarded_processes()
    if procs:
        raise SystemExit(f"FATAL: {sorted({g for _, c in procs for g in GUARDED if g in c})} alive (pids "
                         f"{[p for p, _ in procs]}); B.69 never runs beside them. Use --wait-for-earlier.")


def preflight(live):
    if not any(DATA.glob("s*.csv")):
        raise SystemExit("FATAL: no shot CSVs in data/ -- real data required, aborting.")
    tests = set()
    for s in SEEDS:
        _, split, cm = camp(s)
        for needed in (split / "split_manifest.json", cm):
            if not needed.exists():
                raise SystemExit(f"FATAL: missing frozen B.5 artifact {needed}")
        tests.add(tuple(sorted(outer_manifest(s)["test_files"])))
        overlap_check(s)
    if len(tests) != 1 or len(next(iter(tests))) != TEST_BLOCK:
        raise SystemExit(f"FATAL: the test block is not one block of {TEST_BLOCK} discharges for the four seeds")
    for kind, m, s, j in plan():
        env = run_env(m, s, j, write=False)
        if not Path(env["CES_OUTPUT_DIR"]).name.startswith(".b69_"):
            raise SystemExit("FATAL: output directory outside data/.b69_*")
        if env["CES_SEQ_MODEL"] != m or env["CES_SEQ_EPOCHS"] != B39.EPOCH_CAP:
            raise SystemExit(f"FATAL: {run_dir(m, s, j).name}: model {env['CES_SEQ_MODEL']}, cap {env['CES_SEQ_EPOCHS']}")
    if live and os.environ.get("CUDA_VISIBLE_DEVICES", None) == "":
        raise SystemExit("FATAL: CUDA_VISIBLE_DEVICES is empty in the shell; the runs would not see the GPU")


# ------------------------------------------------------------------------------ one run
def execute(kind, model, s, j):
    out = run_dir(model, s, j)
    out.mkdir(parents=True, exist_ok=True)
    env = run_env(model, s, j)
    weights, metrics = out / "weights" / "seq_lstm.pth", out / "metrics.json"
    steps = [("train", [SEQ_DIR / "train_seq.py"], (weights, metrics))]
    if kind == "outer":
        steps.append(("eval(test)", [SEQ_DIR / "eval_seq.py"], (out / "comparison_errors_test.npz",
                                                                out / "comparison_metrics.json")))
    rec = {"run": out.name, "kind": kind, "model": model, "seed": s, "inner_fold": j,
           "init": int(env["CES_INIT_SEED"]), "status": "ok"}
    t0 = time.time()
    print(f"[b69] === {out.name}", flush=True)
    for name, cmd, artifacts in steps:
        if all(a.exists() for a in artifacts):
            continue
        if not run_step(cmd, env, out / "run.log", name, artifacts=artifacts):
            rec["status"] = f"{name}_failed"
            break
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


def dry_run():
    print("--- no-overlap check (inner training / inner held-out / validation / test), every seed")
    for s in SEEDS:
        print(f"    s{s}: {overlap_check(s)} -> passes")
    print("\n--- the planned runs (registered order: outer, then inner)")
    runs = plan()
    for i, (kind, m, s, j) in enumerate(runs, 1):
        out = run_dir(m, s, j)
        if kind == "outer":
            man = outer_manifest(s)
            sizes = f"train {len(man['train_files'])}, val {len(man['val_files'])}, test {len(man['test_files'])}"
            split = camp(s)[1].name
        else:
            man = inner_manifest(s, j)
            sizes = f"train {len(man['train_files'])}, val {len(man['val_files'])}, inner held-out {len(man['test_files'])}"
            split = split_dir(s, j).name
        steps = "train + eval(test)" if kind == "outer" else "train"
        print(f"{i:3d} {kind:5s} {out.name:28s} | model {m:3s} | seed {s:3d} | split {split} ({sizes}) | "
              f"init {run_env(m, s, j, write=False)['CES_INIT_SEED']} | {steps} | artifacts exist: {done(kind, out)}")
    print(f"{len(runs)} runs: outer {sum(r[0] == 'outer' for r in runs)}, inner {sum(r[0] == 'inner' for r in runs)}; "
          + ", ".join(f"{m} {sum(r[1] == m for r in runs)}" for m in MODELS))
    for m in MODELS:
        for kind, j in (("outer", None), ("inner", 0)):
            mine = ces_vars(run_env(m, SEEDS[0], j, write=False))
            ref = ces_vars(ref_env(m, B39.pooled_run_dir(m, *REF_FOLD)))
            print(f"\n--- env of {run_dir(m, SEEDS[0], j).name}")
            for key in sorted(mine):
                print(f"    {key}={mine[key]}")
            print(f"  diff against run_b39.pooled_env({m!r}, r{REF_FOLD[0]}, f{REF_FOLD[1]}) "
                  f"({B39.pooled_run_dir(m, *REF_FOLD).name}):")
            for key, a, b in env_diff(mine, ref):
                print(f"    {key}: {a}  ->  B.69 {b}")
    print("\n--- every run: the variables that differ from run_b39.pooled_env of the same model")
    bad = []
    for kind, m, s, j in runs:
        keys = sorted(x[0] for x in env_diff(ces_vars(run_env(m, s, j, write=False)),
                                             ces_vars(ref_env(m, B39.pooled_run_dir(m, *REF_FOLD)))))
        if not set(keys) <= set(SWAPPED):
            bad.append((run_dir(m, s, j).name, keys))
    print(f"    {len(runs) - len(bad)} of {len(runs)} runs differ only in {', '.join(SWAPPED)}"
          + ("" if not bad else f"; OTHERS: {bad}"))
    if bad:
        raise SystemExit("FATAL: an environment differs beyond the split, the seed and the output directory")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="print the plan, the env diff and the overlap check")
    ap.add_argument("--wait-for-earlier", action="store_true",
                    help="wait until no guarded runner is alive and the GPU is idle")
    ap.add_argument("--min-free-mb", type=int, default=2000)
    args = ap.parse_args()
    preflight(live=not args.dry_run)
    if args.dry_run:
        dry_run()
        return
    if args.wait_for_earlier:
        wait_for_earlier()
    refuse_if_earlier_alive()
    free = B29.gpu_free_mb()
    print(f"[b69] GPU free memory: {free} MiB", flush=True)
    if free is None or free < args.min_free_mb:
        raise SystemExit(f"FATAL: GPU free memory {free} MiB < {args.min_free_mb} MiB -- B.69 is never run on CPU.")
    runs = plan()
    t0 = time.time()
    records = json.loads(RUNS_OUT.read_text(encoding="utf-8")) if RUNS_OUT.exists() else []
    first = True
    B29.keep_awake(True)
    try:
        for i, (kind, m, s, j) in enumerate(runs, 1):
            if done(kind, run_dir(m, s, j)):
                continue
            refuse_if_earlier_alive()
            rec = execute(kind, m, s, j)
            records = [x for x in records if x["run"] != rec["run"]] + [rec]
            RUNS_OUT.write_text(json.dumps(records, indent=1), encoding="utf-8")
            print(f"[b69] {i}/{len(runs)} {rec} ({(time.time() - t0) / 3600:.2f} h)", flush=True)
            if rec["status"] != "ok" and first:
                print("[b69] the first run of the batch failed; the batch stops (see its run.log)", flush=True)
                break
            first = False
    finally:
        B29.keep_awake(False)
    bad = [x for x in records if x["status"] != "ok"]
    capped = [x["run"] for x in records if x.get("cap_bound")]
    for x in bad:
        print(f"[b69] FAILED: {x['run']} ({x['status']})", flush=True)
    if capped:
        print(f"[b69] reached the epoch cap (reported): {capped}", flush=True)
    print(f"[b69] wall time {(time.time() - t0) / 3600:.2f} h; next: "
          "py ces_prediction/experiments/b69_time_confirm/stage2_b69.py all, then analyze_b69.py", flush=True)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
