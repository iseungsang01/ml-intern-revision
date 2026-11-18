"""B.67 runner (PREREGISTRATION_B67.md): weighing the sources before the recurrent layer.  30 GPU runs on the
fifteen folds of re-partitions r2, r3, r4 of the pooled cross-validation of B.39, after B.65.

Arms, in the registered order (fifteen runs each; folds r2 f0 .. r4 f4; one initialisation per fold):
  S  v2s, gated concatenation   -> data/.b67_S_cut_r{r}_f{k}   (scoring also writes b67_gates_test.npz / .json)
  A  v2a, attention pooling     -> data/.b67_A_cut_r{r}_f{k}
Training: train_b67.py (seq/train_seq.py with the two variants registered at run time); scoring: eval_b67.py
(seq/eval_seq.py likewise, then the gate readout for arm S).  model_seq_v2gate.py holds the architecture.

The environment of every run is run_b39.pooled_env of its fold (cut, cap 80 with early stopping,
CES_PER_SHOT_NORM=0, the seeds, split and harness statistics of the fold) with CES_SEQ_MODEL and the output
directory changed: the env of B.59 arm N of the same fold apart from those two (check_b67.py verifies it).
A run whose comparison_errors_test.npz exists (arm S: and b67_gates_test.npz) is skipped.  The batch never runs
beside a run_b39.py, run_b64.py, run_b65.py, run_b47.py or run_b59.py process; --wait-for-earlier waits until no
such process is alive and the GPU is idle.  A failed first run of an arm stops that arm; the batch goes on.

Usage (repo root):
  py ces_prediction/experiments/b67_gated/check_b67.py            # CPU checks before any run
  py ces_prediction/experiments/b67_gated/run_b67.py --dry-run
  py ces_prediction/experiments/b67_gated/run_b67.py --wait-for-earlier > data/.b67_batch.log 2>&1
  py ces_prediction/experiments/b67_gated/analyze_b67.py [--partial]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "b39_primary"))
import run_b39 as B39  # noqa: E402  (adds b29_causal and b27_kfold to sys.path)
from run_b39 import B27, B29, DATA, run_step  # noqa: E402

REPEATS = (2, 3, 4)
ARMS = ("S", "A")
MODEL = {"S": "v2s", "A": "v2a"}
RUNS_OUT = DATA / ".b67_runs.json"
TRAIN, EVAL = HERE / "train_b67.py", HERE / "eval_b67.py"
MODEL_FILE = HERE / "model_seq_v2gate.py"
GATES_NPZ, GATES_JSON = "b67_gates_test.npz", "b67_gates_test.json"
POLL_S, STATUS_EVERY_S, GPU_IDLE_MIB, GPU_IDLE_POLLS = 60, 600, 1000, 3
GUARDED = ("run_b39.py", "run_b64.py", "run_b65.py", "run_b47.py", "run_b59.py")


# ------------------------------------------------------------------------------ directories / env
def run_dir(arm, r, k):
    return DATA / f".b67_{arm}_cut_r{r}_f{k}"


def no_b67(env):
    """B67_* variables are never set by this batch; one left in the shell must not leak into a run."""
    return {k: v for k, v in env.items() if not k.startswith("B67_")}


def arm_env(arm, r, k, out):
    """run_b39.pooled_env of the arm's network on fold (r, k): only CES_SEQ_MODEL and the output directory are
    the arm's own."""
    return no_b67(B39.pooled_env(MODEL[arm], r, k, out))


def estimator_dir(r, k):
    return B39.pooled_run_dir("v2", r, k)


def twin_dir(r, k):
    return B39.pooled_run_dir("v2h", r, k)


def eval_artifacts(arm, out):
    a = (out / "comparison_errors_test.npz", out / "comparison_metrics.json")
    return a + ((out / GATES_NPZ, out / GATES_JSON) if arm == "S" else ())


def plan(arms):
    """(arm, label, out, model, r, k, env_fn) in the registered order."""
    runs = []
    for a in arms:
        for r in REPEATS:
            for k in range(B27.K):
                out = run_dir(a, r, k)
                runs.append((a, f"{a} r{r} f{k} {MODEL[a]}", out, MODEL[a], r, k,
                             lambda a=a, r=r, k=k, o=out: arm_env(a, r, k, o)))
    return runs


def done(arm, out):
    """Scored (arm S: and its gates written)."""
    need = [out / "comparison_errors_test.npz"] + ([out / GATES_NPZ] if arm == "S" else [])
    return all(p.exists() for p in need)


# ------------------------------------------------------------------------------ guards
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
    print(f"[b67] waiting for {', '.join(GUARDED)} to end (no such process, GPU memory used < "
          f"{GPU_IDLE_MIB} MiB on {GPU_IDLE_POLLS} consecutive polls)", flush=True)
    while True:
        procs = guarded_processes()
        used = gpu_used_mib()
        idle = idle + 1 if (not procs and used is not None and used < GPU_IDLE_MIB) else 0
        if idle >= GPU_IDLE_POLLS:
            print(f"[b67] earlier batches have ended ({(time.time() - t0) / 3600:.2f} h waited; GPU used {used} MiB)", flush=True)
            return
        if time.time() - last >= STATUS_EVERY_S:
            last = time.time()
            print(f"[b67] {time.strftime('%Y-%m-%d %H:%M')} waiting: guarded processes {len(procs)}, "
                  f"GPU used {used} MiB, idle polls {idle}", flush=True)
        time.sleep(POLL_S)


def refuse_if_earlier_alive():
    procs = guarded_processes()
    if procs:
        raise SystemExit(f"FATAL: {sorted({g for _, c in procs for g in GUARDED if g in c})} alive (pids "
                         f"{[p for p, _ in procs]}); B.67 never runs beside them. Use --wait-for-earlier.")


def preflight(runs, live):
    if not any(DATA.glob("s*.csv")):
        raise SystemExit("FATAL: no shot CSVs in data/ -- real data required, aborting.")
    for r in REPEATS:
        for k in range(B27.K):
            for d in (estimator_dir(r, k), twin_dir(r, k)):
                if not (d / "comparison_errors_test.npz").exists():
                    raise SystemExit(f"FATAL: the B.39 reference {d.name} is missing")
    for script in (TRAIN, EVAL, MODEL_FILE, HERE / "gates_b67.py"):
        if not script.exists():
            raise SystemExit(f"FATAL: missing {script}")
    for run in runs:
        env = run[6]()
        for needed in (Path(env["CES_SPLIT_DIR"]) / "split_manifest.json", Path(env["CES_CONTROL_METRICS"])):
            if not needed.exists():
                raise SystemExit(f"FATAL: {run[1]}: missing {needed}")
        if not Path(env["CES_OUTPUT_DIR"]).name.startswith(".b67_"):
            raise SystemExit(f"FATAL: {run[1]}: output directory outside data/.b67_*")
        if env["CES_SEQ_MODEL"] != run[3]:
            raise SystemExit(f"FATAL: {run[1]}: CES_SEQ_MODEL={env['CES_SEQ_MODEL']}")
        if live and env.get("CUDA_VISIBLE_DEVICES", None) == "":
            raise SystemExit("FATAL: CUDA_VISIBLE_DEVICES is empty in the shell; the runs would not see the GPU")


# ------------------------------------------------------------------------------ one run
def execute(arm, label, out, model, r, k, env):
    out.mkdir(parents=True, exist_ok=True)
    weights, metrics = out / "weights" / "seq_lstm.pth", out / "metrics.json"
    rec = {"run": out.name, "arm": arm, "model": model, "repeat": r, "fold": k,
           "init": int(env["CES_INIT_SEED"]), "status": "ok"}
    t0 = time.time()
    print(f"[b67] === {out.name}", flush=True)
    for name, cmd, artifacts in (("train", [TRAIN], (weights, metrics)),
                                 ("eval(test)", [EVAL], eval_artifacts(arm, out))):
        if all(a.exists() for a in artifacts):
            continue
        if not run_step(cmd, env, out / "run.log", name, artifacts=artifacts):
            rec["status"] = f"{name}_failed"
            break
    if metrics.exists():
        m = json.loads(metrics.read_text(encoding="utf-8"))
        rec.update({"epochs_run": m["epochs_run"], "best_epoch": m["best_epoch"], "n_params": m["n_params"],
                    "best_val_masked_mse": m["best_val_masked_mse"], "seq_model": m["seq_model"],
                    "cap_bound": m["epochs_run"] >= int(env["CES_SEQ_EPOCHS"])})
        if rec["status"] == "ok" and m["seq_model"] != model:
            rec["status"] = "model_mismatch"
    if arm == "S" and (out / GATES_JSON).exists():
        g = json.loads((out / GATES_JSON).read_text(encoding="utf-8"))
        rec["gate_check"] = {q: g.get(q) for q in ("passes", "rows", "found", "identical_rows", "p99_abs_err_diff_eV",
                                                   "mean_gate")}
        if not g.get("passes") and rec["status"] == "ok":
            rec["status"] = "gate_check_failed"
    rec.update({"label": label, "minutes": round((time.time() - t0) / 60, 1)})
    return rec


# ------------------------------------------------------------------------------ dry run
def ces_vars(env):
    return {k: v for k, v in env.items() if k.startswith("CES_") or k.startswith("B67_")}


def env_diff(mine, ref):
    keys = sorted(set(mine) | set(ref))
    return [(k, ref.get(k, "<unset>"), mine.get(k, "<unset>")) for k in keys if mine.get(k) != ref.get(k)]


def b59_n_env(r, k):
    sys.path.insert(0, str(HERE.parent / "b59_inputs"))
    import run_b59
    return run_b59.arm_env("N", r, k, run_b59.run_dir("N", r, k))


def print_env_diff(runs):
    reps = {}
    for run in runs:
        reps.setdefault(run[0], run)
    for arm, run in reps.items():
        mine = ces_vars(run[6]())
        ref = ces_vars(b59_n_env(run[4], run[5]))
        print(f"\n--- env of {run[1]} ({run[2].name})")
        for k in sorted(mine):
            print(f"    {k}={mine[k]}")
        d = env_diff(mine, ref)
        print(f"  diff against B.59 arm N of the same fold ({len(d)} variables):")
        for k, a, b in d:
            print(f"    {k}: {a}  ->  B.67 {b}")
    print("\n--- every run: CES variables that differ from B.59 arm N of its fold (CES_OUTPUT_DIR omitted)")
    for run in runs:
        d = [x for x in env_diff(ces_vars(run[6]()), ces_vars(b59_n_env(run[4], run[5]))) if x[0] != "CES_OUTPUT_DIR"]
        print(f"    {run[2].name:22s} init {run[6]()['CES_INIT_SEED']:5s} " + ("; ".join(f"{k}={b}" for k, _, b in d) or "(none)"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="print the plan and the env diff; start nothing")
    ap.add_argument("--arms", nargs="*", default=list(ARMS), choices=list(ARMS))
    ap.add_argument("--wait-for-earlier", action="store_true",
                    help="wait until no run_b39 / run_b64 / run_b65 / run_b47 / run_b59 process is alive and the GPU is idle")
    ap.add_argument("--min-free-mb", type=int, default=2000)
    args = ap.parse_args()
    arms = [a for a in ARMS if a in args.arms]
    runs = plan(arms)
    preflight(runs, live=not args.dry_run)
    if args.dry_run:
        for i, run in enumerate(runs, 1):
            arm, label, out, model, r, k = run[:6]
            print(f"{i:2d} arm {arm} | {out.name:22s} | model {model:4s} | repeat {r} fold {k} | "
                  f"init {run[6]()['CES_INIT_SEED']} | train {TRAIN.name} | eval {EVAL.name} | artifacts exist: {done(arm, out)}")
        print(f"{len(runs)} runs: " + ", ".join(f"{a} {sum(x[0] == a for x in runs)}" for a in arms))
        print_env_diff(runs)
        return
    if args.wait_for_earlier:
        wait_for_earlier()
    refuse_if_earlier_alive()
    free = B29.gpu_free_mb()
    print(f"[b67] GPU free memory: {free} MiB", flush=True)
    if free is None or free < args.min_free_mb:
        raise SystemExit(f"FATAL: GPU free memory {free} MiB < {args.min_free_mb} MiB -- B.67 is never run on CPU.")
    t0 = time.time()
    records = json.loads(RUNS_OUT.read_text(encoding="utf-8")) if RUNS_OUT.exists() else []
    stopped, started = set(), set()
    B29.keep_awake(True)
    try:
        for i, run in enumerate(runs, 1):
            arm, out = run[0], run[2]
            if arm in stopped or done(arm, out):
                continue
            refuse_if_earlier_alive()
            first = arm not in started
            started.add(arm)
            rec = execute(*run[:6], run[6]())
            records = [x for x in records if x["run"] != rec["run"]] + [rec]
            RUNS_OUT.write_text(json.dumps(records, indent=1), encoding="utf-8")
            print(f"[b67] {i}/{len(runs)} {rec} ({(time.time() - t0) / 3600:.2f} h)", flush=True)
            if rec["status"] != "ok":
                if first:
                    print(f"[b67] the first run of arm {arm} failed; arm {arm} is stopped, the batch goes on", flush=True)
                    stopped.add(arm)
                else:
                    print(f"[b67] FAILED (logged, the batch goes on): {rec['run']} ({rec['status']})", flush=True)
    finally:
        B29.keep_awake(False)
    bad = [x for x in records if x["status"] != "ok"]
    for x in bad:
        print(f"[b67] FAILED: {x['run']} ({x['status']})", flush=True)
    if stopped:
        print(f"[b67] arms stopped after a failed first run: {sorted(stopped)}", flush=True)
    print(f"[b67] wall time {(time.time() - t0) / 3600:.2f} h; next: py ces_prediction/experiments/b67_gated/analyze_b67.py",
          flush=True)
    sys.exit(1 if (bad or stopped) else 0)


if __name__ == "__main__":
    main()
