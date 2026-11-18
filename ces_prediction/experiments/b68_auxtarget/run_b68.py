"""B.68 runner (PREREGISTRATION_B68.md): an auxiliary training target denoised by the neighbouring values.  15 GPU
runs on the fifteen folds of re-partitions r2, r3, r4 of the pooled cross-validation of B.39, after B.67.

Arm X (folds r2 f0 .. r4 f4, one initialisation per fold) -> data/.b68_X_cut_r{r}_f{k}
  training  train_b68.py: seq/train_seq.py with the auxiliary head on the T_i hidden state and the loss
            masked MSE + 0.5 x masked MSE of the auxiliary head on the offline-GP target (aux_b68.py), installed at
            run time; early stopping on the main heads' validation loss; the checkpoint is plain seq_v2
  scoring   seq/eval_seq.py unchanged (CES_SEQ_MODEL=v2, main heads only)

The environment of every run is run_b39.pooled_env("v2", r, k, out): the env of the pooled network of B.39 on the
fold (cut, cap 80 with early stopping, CES_PER_SHOT_NORM=0, the seeds, split and harness statistics of the fold)
with only the output directory changed; B68_* variables are never set and one left in the shell is removed.  The
auxiliary target is read from data/.b68_aux/ (build_aux_b68.py, CPU, once; the runner refuses to start without it).
A run whose comparison_errors_test.npz exists is skipped.  The batch never runs beside a run_b39.py, run_b64.py,
run_b65.py, run_b67.py, run_b47.py or run_b59.py process; --wait-for-earlier waits until no such process is alive
and the GPU is idle.  A failed first run stops the batch.

Usage (repo root):
  py ces_prediction/experiments/b68_auxtarget/build_aux_b68.py         # CPU, once (the auxiliary-target cache)
  py ces_prediction/experiments/b68_auxtarget/check_b68.py             # CPU checks before any run
  py ces_prediction/experiments/b68_auxtarget/run_b68.py --dry-run
  py ces_prediction/experiments/b68_auxtarget/run_b68.py --wait-for-earlier > data/.b68_batch.log 2>&1
  py ces_prediction/experiments/b68_auxtarget/analyze_b68.py [--partial]
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
from run_b39 import B27, B29, DATA, SEQ_DIR, run_step  # noqa: E402

REPEATS = (2, 3, 4)
ARM, MODEL = "X", "v2"
RUNS_OUT = DATA / ".b68_runs.json"
TRAIN, EVAL = HERE / "train_b68.py", SEQ_DIR / "eval_seq.py"
SIDECAR = "b68_aux.json"
AUX_CACHE = (DATA / ".b68_aux" / "aux_ti_offline_gp.npz", DATA / ".b68_aux" / "meta.json")
POLL_S, STATUS_EVERY_S, GPU_IDLE_MIB, GPU_IDLE_POLLS = 60, 600, 1000, 3
GUARDED = ("run_b39.py", "run_b64.py", "run_b65.py", "run_b67.py", "run_b47.py", "run_b59.py")


# ------------------------------------------------------------------------------ directories / env
def run_dir(r, k):
    return DATA / f".b68_{ARM}_cut_r{r}_f{k}"


def no_b68(env):
    """B68_* variables are never set by this batch; one left in the shell must not leak into a run."""
    return {k: v for k, v in env.items() if not k.startswith("B68_")}


def arm_env(r, k, out):
    """run_b39.pooled_env of the network (v2) on fold (r, k); only the output directory is the run's own."""
    return no_b68(B39.pooled_env(MODEL, r, k, out))


def estimator_dir(r, k):
    return B39.pooled_run_dir("v2", r, k)


def twin_dir(r, k):
    return B39.pooled_run_dir("v2h", r, k)


def plan():
    """(label, out, r, k, env_fn) in the registered order."""
    return [(f"X r{r} f{k} v2+aux", run_dir(r, k), r, k, lambda r=r, k=k: arm_env(r, k, run_dir(r, k)))
            for r in REPEATS for k in range(B27.K)]


def done(out):
    return (out / "comparison_errors_test.npz").exists()


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
    print(f"[b68] waiting for {', '.join(GUARDED)} to end (no such process, GPU memory used < "
          f"{GPU_IDLE_MIB} MiB on {GPU_IDLE_POLLS} consecutive polls)", flush=True)
    while True:
        procs = guarded_processes()
        used = gpu_used_mib()
        idle = idle + 1 if (not procs and used is not None and used < GPU_IDLE_MIB) else 0
        if idle >= GPU_IDLE_POLLS:
            print(f"[b68] earlier batches have ended ({(time.time() - t0) / 3600:.2f} h waited; GPU used {used} MiB)", flush=True)
            return
        if time.time() - last >= STATUS_EVERY_S:
            last = time.time()
            print(f"[b68] {time.strftime('%Y-%m-%d %H:%M')} waiting: guarded processes {len(procs)}, "
                  f"GPU used {used} MiB, idle polls {idle}", flush=True)
        time.sleep(POLL_S)


def refuse_if_earlier_alive():
    procs = guarded_processes()
    if procs:
        raise SystemExit(f"FATAL: {sorted({g for _, c in procs for g in GUARDED if g in c})} alive (pids "
                         f"{[p for p, _ in procs]}); B.68 never runs beside them. Use --wait-for-earlier.")


def preflight(runs, live):
    if not any(DATA.glob("s*.csv")):
        raise SystemExit("FATAL: no shot CSVs in data/ -- real data required, aborting.")
    for p in AUX_CACHE:
        if not p.exists():
            raise SystemExit(f"FATAL: {p} missing; run py ces_prediction/experiments/b68_auxtarget/build_aux_b68.py first")
    for r in REPEATS:
        for k in range(B27.K):
            for d in (estimator_dir(r, k), twin_dir(r, k)):
                if not (d / "comparison_errors_test.npz").exists():
                    raise SystemExit(f"FATAL: the B.39 reference {d.name} is missing")
    for script in (TRAIN, EVAL, HERE / "aux_b68.py", HERE / "target_b68.py"):
        if not script.exists():
            raise SystemExit(f"FATAL: missing {script}")
    for label, out, r, k, env_fn in runs:
        env = env_fn()
        for needed in (Path(env["CES_SPLIT_DIR"]) / "split_manifest.json", Path(env["CES_CONTROL_METRICS"])):
            if not needed.exists():
                raise SystemExit(f"FATAL: {label}: missing {needed}")
        if not Path(env["CES_OUTPUT_DIR"]).name.startswith(".b68_"):
            raise SystemExit(f"FATAL: {label}: output directory outside data/.b68_*")
        if env["CES_SEQ_MODEL"] != MODEL:
            raise SystemExit(f"FATAL: {label}: CES_SEQ_MODEL={env['CES_SEQ_MODEL']}")
        if live and env.get("CUDA_VISIBLE_DEVICES", None) == "":
            raise SystemExit("FATAL: CUDA_VISIBLE_DEVICES is empty in the shell; the runs would not see the GPU")


# ------------------------------------------------------------------------------ one run
def execute(label, out, r, k, env):
    out.mkdir(parents=True, exist_ok=True)
    weights, metrics, side = out / "weights" / "seq_lstm.pth", out / "metrics.json", out / SIDECAR
    rec = {"run": out.name, "arm": ARM, "model": MODEL, "repeat": r, "fold": k, "init": int(env["CES_INIT_SEED"]),
           "status": "ok"}
    t0 = time.time()
    print(f"[b68] === {out.name}", flush=True)
    for name, cmd, artifacts in (("train", [TRAIN], (weights, metrics, side)),
                                 ("eval(test)", [EVAL], (out / "comparison_errors_test.npz", out / "comparison_metrics.json"))):
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
        if rec["status"] == "ok" and m["seq_model"] != MODEL:
            rec["status"] = "model_mismatch"
    if side.exists():
        s = json.loads(side.read_text(encoding="utf-8"))
        rec["aux"] = {"weight": s["aux_weight"], "train_share_rows_with_target": s["train"]["share_rows_with_aux_target"],
                      "val_share_rows_with_target": s["val"]["share_rows_with_aux_target"],
                      "last_val_aux_mse": next((p["aux_masked_mse"] for p in reversed(s["passes"]) if p["phase"] == "val"), None)}
    elif rec["status"] == "ok":
        rec["status"] = "sidecar_missing"
    rec.update({"label": label, "minutes": round((time.time() - t0) / 60, 1)})
    return rec


# ------------------------------------------------------------------------------ dry run
def ces_vars(env):
    return {k: v for k, v in env.items() if k.startswith("CES_") or k.startswith("B68_")}


def env_diff(mine, ref):
    keys = sorted(set(mine) | set(ref))
    return [(k, ref.get(k, "<unset>"), mine.get(k, "<unset>")) for k in keys if mine.get(k) != ref.get(k)]


def b59_n_env(r, k):
    sys.path.insert(0, str(HERE.parent / "b59_inputs"))
    import run_b59
    return run_b59.arm_env("N", r, k, run_b59.run_dir("N", r, k))


def print_env_diff(runs):
    label, out, r, k, env_fn = runs[0]
    mine = ces_vars(env_fn())
    print(f"\n--- env of {label} ({out.name})")
    for key in sorted(mine):
        print(f"    {key}={mine[key]}")
    d = env_diff(mine, ces_vars(b59_n_env(r, k)))
    print(f"  diff against B.59 arm N of the same fold ({len(d)} variables):")
    for key, a, b in d:
        print(f"    {key}: {a}  ->  B.68 {b}")
    print("\n--- every run: CES variables that differ from B.59 arm N of its fold (CES_OUTPUT_DIR omitted)")
    for label, out, r, k, env_fn in runs:
        e = env_fn()
        d = [x for x in env_diff(ces_vars(e), ces_vars(b59_n_env(r, k))) if x[0] != "CES_OUTPUT_DIR"]
        print(f"    {out.name:22s} init {e['CES_INIT_SEED']:5s} " + ("; ".join(f"{q}={b}" for q, _, b in d) or "(none)"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="print the plan and the env diff; start nothing")
    ap.add_argument("--wait-for-earlier", action="store_true",
                    help="wait until no run_b39 / b64 / b65 / b67 / b47 / b59 process is alive and the GPU is idle")
    ap.add_argument("--min-free-mb", type=int, default=2000)
    args = ap.parse_args()
    runs = plan()
    preflight(runs, live=not args.dry_run)
    if args.dry_run:
        for i, (label, out, r, k, env_fn) in enumerate(runs, 1):
            print(f"{i:2d} arm {ARM} | {out.name:22s} | model {MODEL} + auxiliary head | repeat {r} fold {k} | "
                  f"init {env_fn()['CES_INIT_SEED']} | train {TRAIN.name} | eval {EVAL.name} | artifacts exist: {done(out)}")
        print(f"{len(runs)} runs: {ARM} {len(runs)}")
        print_env_diff(runs)
        return
    if args.wait_for_earlier:
        wait_for_earlier()
    refuse_if_earlier_alive()
    free = B29.gpu_free_mb()
    print(f"[b68] GPU free memory: {free} MiB", flush=True)
    if free is None or free < args.min_free_mb:
        raise SystemExit(f"FATAL: GPU free memory {free} MiB < {args.min_free_mb} MiB -- B.68 is never run on CPU.")
    t0 = time.time()
    records = json.loads(RUNS_OUT.read_text(encoding="utf-8")) if RUNS_OUT.exists() else []
    first = True
    B29.keep_awake(True)
    try:
        for i, (label, out, r, k, env_fn) in enumerate(runs, 1):
            if done(out):
                continue
            refuse_if_earlier_alive()
            rec = execute(label, out, r, k, env_fn())
            records = [x for x in records if x["run"] != rec["run"]] + [rec]
            RUNS_OUT.write_text(json.dumps(records, indent=1), encoding="utf-8")
            print(f"[b68] {i}/{len(runs)} {rec} ({(time.time() - t0) / 3600:.2f} h)", flush=True)
            if rec["status"] != "ok":
                if first:
                    print("[b68] the first run failed; the batch is stopped", flush=True)
                    break
                print(f"[b68] FAILED (logged, the batch goes on): {rec['run']} ({rec['status']})", flush=True)
            first = False
    finally:
        B29.keep_awake(False)
    bad = [x for x in records if x["status"] != "ok"]
    for x in bad:
        print(f"[b68] FAILED: {x['run']} ({x['status']})", flush=True)
    print(f"[b68] wall time {(time.time() - t0) / 3600:.2f} h; next: py ces_prediction/experiments/b68_auxtarget/analyze_b68.py",
          flush=True)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
