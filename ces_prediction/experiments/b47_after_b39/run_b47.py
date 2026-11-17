"""B.47 runner (PREREGISTRATION_B47.md): after B.39 -- the ablation at two further initialisations,
the causal variant and its twin trained with a delayed CES history, and the inclusive population.

Arms, in the registered order (64 runs):
  D  v2e, v2b, v2n at the two further initialisations of each split (run_b39.inits(seed)[1:])    24 runs
     -> data/.b47_{model}_cut_s{seed}_i{init}
  L  v2 and v2h, initialisation = split seed, trained through train_delayed.py with
     B47_CES_DELAY = 2 and the L = 0 control of the wrapper                                      16 runs
     -> data/.b47L{delay}_{model}_cut_s{seed}_i{seed}
  I  v2 and v2h, inclusive population (CES_TI_SPIKE_CUT_EV=0, inclusive split and harness
     statistics of B.29), the three initialisations of each split                                 24 runs
     -> data/.b47_{model}_incl_s{seed}_i{init}

Every run: train (arm L: train_delayed.py, otherwise seq/train_seq.py) -> seq/eval_seq.py (TEST).
The environment is run_b39.split_env (cut) or run_b29.run_env(..., "incl", ...) with the B.39 epoch
cap; nothing but the initialisation seed, the model, the population and the output directory
changes (arm L adds B47_CES_DELAY).  A run whose comparison_errors_test.npz exists is skipped
(resume).  The batch never starts while a run_b39.py process is alive.

Usage (repo root):
  py ces_prediction/experiments/b47_after_b39/run_b47.py --dry-run
  py ces_prediction/experiments/b47_after_b39/run_b47.py --wait-for-b39 > data/.b47_batch.log 2>&1
  py ces_prediction/experiments/b47_after_b39/run_b47.py --arms L I
  py ces_prediction/experiments/b47_after_b39/analyze_b47.py
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
from run_b39 import B29, DATA, SEQ_DIR, run_step  # noqa: E402

SEEDS = B39.SEEDS
EPOCH_CAP = B39.EPOCH_CAP
ABLATIONS = B39.ABLATIONS
DELAYS = (2, 0)
ARMS = ("D", "L", "I")
RUNS_OUT = DATA / ".b47_runs.json"
TRAIN_DELAYED = HERE / "train_delayed.py"
POLL_S, STATUS_EVERY_S, GPU_IDLE_MIB, GPU_IDLE_POLLS = 60, 600, 1000, 3


# ------------------------------------------------------------------------------ directories / env
def d_dir(model, seed, init):
    return DATA / f".b47_{model}_cut_s{seed}_i{init}"


def l_dir(model, seed, delay):
    return DATA / f".b47L{delay}_{model}_cut_s{seed}_i{seed}"


def i_dir(model, seed, init):
    return DATA / f".b47_{model}_incl_s{seed}_i{init}"


def no_b47(env):
    """B47_* variables are set only by delayed_env; one left in the shell must not leak into a run."""
    return {k: v for k, v in env.items() if not k.startswith("B47_")}


def cut_env(model, seed, init, out):
    """B.39's env of this model (cut, cap 80, CES_PER_SHOT_NORM=0 for fast-channel networks, the
    twin as in its cap-80 runs) with the initialisation and the output directory set."""
    return no_b47(B39.split_env(model, seed, init, out))


def incl_env(model, seed, init, out):
    """The same with the inclusive population of B.29 (CES_TI_SPIKE_CUT_EV=0, .b1_manifest split,
    .wsweep_hf_w2 harness statistics)."""
    arm = "v2h" if model == "v2h" else "norm0"
    env = B29.run_env(arm, "split", "incl", seed, out, False, epochs=EPOCH_CAP, model=model)
    env["CES_INIT_SEED"] = str(init)
    return no_b47(env)


def delayed_env(model, seed, delay, out):
    env = cut_env(model, seed, seed, out)
    env["B47_CES_DELAY"] = str(delay)
    return env


def b39_reference_env(model, seed, init):
    """The env of the B.39 run of the same model / split / initialisation (for the dry-run diff)."""
    out = B39.twin_dir(seed, init) if model == "v2h" else B39.split_run_dir(model, seed, init)
    return B39.split_env(model, seed, init, out)


def plan(arms):
    """(arm, label, out, model, seed, init, delay, env_fn, train_script) in the registered order."""
    runs = []
    if "D" in arms:
        for s in SEEDS:
            for init in B39.inits(s)[1:]:
                for m in ABLATIONS:
                    out = d_dir(m, s, init)
                    runs.append(("D", f"D s{s} {m} i{init}", out, m, s, init, None,
                                 lambda m=m, s=s, i=init, o=out: cut_env(m, s, i, o), SEQ_DIR / "train_seq.py"))
    if "L" in arms:
        for s in SEEDS:
            for m in ("v2", "v2h"):
                for L in DELAYS:
                    out = l_dir(m, s, L)
                    runs.append(("L", f"L s{s} {m} delay{L}", out, m, s, s, L,
                                 lambda m=m, s=s, L=L, o=out: delayed_env(m, s, L, o), TRAIN_DELAYED))
    if "I" in arms:
        for s in SEEDS:
            for init in B39.inits(s):
                for m in ("v2", "v2h"):
                    out = i_dir(m, s, init)
                    runs.append(("I", f"I s{s} {m} i{init}", out, m, s, init, None,
                                 lambda m=m, s=s, i=init, o=out: incl_env(m, s, i, o), SEQ_DIR / "train_seq.py"))
    return runs


def done(out):
    return (out / "comparison_errors_test.npz").exists()


# ------------------------------------------------------------------------------ guards
def b39_processes():
    """PIDs and command lines of live processes whose command line contains run_b39.py.  The query
    lists python / py processes only, so the PowerShell process that runs it never matches."""
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
        if "run_b39.py" in cmd and pid.strip().isdigit() and int(pid) != me:
            hits.append((int(pid), cmd.strip()))
    return hits


def gpu_used_mib():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=30).stdout
        return int(out.strip().splitlines()[0])
    except Exception:  # noqa: BLE001
        return None


def wait_for_b39():
    t0 = last = time.time()
    idle = 0
    print("[b47] waiting for B.39 to end (no run_b39.py process, GPU memory used < "
          f"{GPU_IDLE_MIB} MiB on {GPU_IDLE_POLLS} consecutive polls)", flush=True)
    while True:
        procs = b39_processes()
        used = gpu_used_mib()
        idle = idle + 1 if (not procs and used is not None and used < GPU_IDLE_MIB) else 0
        if idle >= GPU_IDLE_POLLS:
            print(f"[b47] B.39 has ended ({(time.time() - t0) / 3600:.2f} h waited; GPU used {used} MiB)", flush=True)
            return
        if time.time() - last >= STATUS_EVERY_S:
            last = time.time()
            print(f"[b47] {time.strftime('%Y-%m-%d %H:%M')} waiting: run_b39 processes {len(procs)}, "
                  f"GPU used {used} MiB, idle polls {idle}", flush=True)
        time.sleep(POLL_S)


def refuse_if_b39_alive():
    procs = b39_processes()
    if procs:
        raise SystemExit(f"FATAL: run_b39.py is alive (pids {[p for p, _ in procs]}); B.47 never runs beside B.39. "
                         "Use --wait-for-b39 or re-launch when B.39 has ended.")


def preflight(runs):
    if not any(DATA.glob("s*.csv")):
        raise SystemExit("FATAL: no shot CSVs in data/ -- real data required, aborting.")
    for r in runs:
        env = r[7]()
        for needed in (Path(env["CES_SPLIT_DIR"]) / "split_manifest.json", Path(env["CES_CONTROL_METRICS"])):
            if not needed.exists():
                raise SystemExit(f"FATAL: {r[1]}: missing frozen artifact {needed}")
        if not str(Path(env["CES_OUTPUT_DIR"]).name).startswith(".b47"):
            raise SystemExit(f"FATAL: {r[1]}: output directory outside data/.b47_*")
    if not TRAIN_DELAYED.exists():
        raise SystemExit(f"FATAL: {TRAIN_DELAYED} missing")


# ------------------------------------------------------------------------------ one run
def execute(arm, label, out, model, seed, init, delay, env, train_script):
    out.mkdir(parents=True, exist_ok=True)
    weights, metrics = out / "weights" / "seq_lstm.pth", out / "metrics.json"
    npz, cm = out / "comparison_errors_test.npz", out / "comparison_metrics.json"
    train_art = (weights, metrics) + ((out / "b47_delay.json",) if arm == "L" else ())
    rec = {"run": out.name, "arm": arm, "model": model, "split": seed, "init": init, "delay": delay, "status": "ok"}
    t0 = time.time()
    print(f"[b47] === {out.name}", flush=True)
    for name, cmd, artifacts in (("train", [train_script], train_art), ("eval(test)", [SEQ_DIR / "eval_seq.py"], (npz, cm))):
        if all(a.exists() for a in artifacts):
            continue
        if not run_step(cmd, env, out / "run.log", name, artifacts=artifacts):
            rec["status"] = f"{name}_failed"
            break
    if metrics.exists():
        m = json.loads(metrics.read_text(encoding="utf-8"))
        rec.update({"epochs_run": m["epochs_run"], "best_epoch": m["best_epoch"], "n_params": m["n_params"],
                    "cap_bound": m["epochs_run"] >= int(env["CES_SEQ_EPOCHS"])})
    if arm == "L" and (out / "b47_delay.json").exists():
        rec["delay_recorded"] = json.loads((out / "b47_delay.json").read_text(encoding="utf-8"))["delay_frames"]
        if rec["delay_recorded"] != delay and rec["status"] == "ok":
            rec["status"] = "delay_mismatch"
    rec.update({"label": label, "minutes": round((time.time() - t0) / 60, 1)})
    return rec


# ------------------------------------------------------------------------------ dry run
def ces_vars(env):
    return {k: v for k, v in env.items() if k.startswith("CES_") or k.startswith("B47_")}


def print_env_diff(runs):
    def diff(mine, ref, what):
        keys = sorted(set(mine) | set(ref))
        d = [(k, ref.get(k, "<unset>"), mine.get(k, "<unset>")) for k in keys if mine.get(k) != ref.get(k)]
        print(f"  diff against {what} ({len(d)} variables):")
        for k, a, b in d:
            print(f"    {k}: {a}  ->  B.47 {b}")

    reps = {}
    for r in runs:      # one representative per arm and per kind of network (twin / fast channels)
        reps.setdefault((r[0], r[3] == "v2h"), r)
    for (arm, _), r in reps.items():
        _, label, out, model, seed, init, delay, env_fn, _ = r
        mine = ces_vars(env_fn())
        print(f"\n--- env of {label} ({out.name})")
        for k in sorted(mine):
            print(f"    {k}={mine[k]}")
        diff(mine, ces_vars(b39_reference_env(model, seed, init)), f"the B.39 env of model {model}, split {seed}, init {init}")
        if arm == "I":
            b29 = B29.run_env("v2h" if model == "v2h" else "norm0", "split", "incl", seed,
                              DATA / f".b29_{'v2h' if model == 'v2h' else 'norm0'}_incl_s{seed}", False, model=model)
            diff(mine, ces_vars(b29), f"the B.29 inclusive run of model {model}, split {seed} (cap 30)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="print the plan and the env diff; start nothing")
    ap.add_argument("--arms", nargs="*", default=list(ARMS), choices=list(ARMS))
    ap.add_argument("--wait-for-b39", action="store_true")
    ap.add_argument("--min-free-mb", type=int, default=2000)
    args = ap.parse_args()
    arms = [a for a in ARMS if a in args.arms]
    runs = plan(arms)
    preflight(runs)
    if args.dry_run:
        for i, r in enumerate(runs, 1):
            arm, label, out, model, seed, init, delay = r[:7]
            print(f"{i:2d} arm {arm} | {out.name:28s} | model {model:4s} | split {seed:3d} | init {init:3d} | "
                  f"delay {'-' if delay is None else delay} | train {r[8].name} | artifacts exist: {done(out)}")
        print(f"{len(runs)} runs: " + ", ".join(f"{a} {sum(r[0] == a for r in runs)}" for a in arms))
        print_env_diff(runs)
        return
    if args.wait_for_b39:
        wait_for_b39()
    refuse_if_b39_alive()
    free = B29.gpu_free_mb()
    print(f"[b47] GPU free memory: {free} MiB", flush=True)
    if free is None or free < args.min_free_mb:
        raise SystemExit(f"FATAL: GPU free memory {free} MiB < {args.min_free_mb} MiB -- B.47 is never run on CPU.")
    t0 = time.time()
    records = json.loads(RUNS_OUT.read_text(encoding="utf-8")) if RUNS_OUT.exists() else []
    stopped, started = set(), set()
    B29.keep_awake(True)
    try:
        for i, r in enumerate(runs, 1):
            arm, label, out = r[0], r[1], r[2]
            if arm in stopped:
                continue
            if done(out):
                continue
            refuse_if_b39_alive()
            first = arm not in started
            started.add(arm)
            rec = execute(*r[:7], r[7](), r[8])
            records = [x for x in records if x["run"] != rec["run"]] + [rec]
            RUNS_OUT.write_text(json.dumps(records, indent=1), encoding="utf-8")
            print(f"[b47] {i}/{len(runs)} {rec} ({(time.time() - t0) / 3600:.2f} h)", flush=True)
            if rec["status"] != "ok":
                if first:
                    print(f"[b47] the first run of arm {arm} failed; arm {arm} is stopped, the batch goes on", flush=True)
                    stopped.add(arm)
                else:
                    print(f"[b47] FAILED (logged, the batch goes on): {rec['run']} ({rec['status']})", flush=True)
    finally:
        B29.keep_awake(False)
    bad = [x for x in records if x["status"] != "ok"]
    for x in bad:
        print(f"[b47] FAILED: {x['run']} ({x['status']})", flush=True)
    if stopped:
        print(f"[b47] arms stopped after a failed first run: {sorted(stopped)}", flush=True)
    print(f"[b47] wall time {(time.time() - t0) / 3600:.2f} h; next: py ces_prediction/experiments/b47_after_b39/analyze_b47.py",
          flush=True)
    sys.exit(1 if (bad or stopped) else 0)


if __name__ == "__main__":
    main()
