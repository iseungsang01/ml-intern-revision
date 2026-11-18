"""B.53 runner (PREREGISTRATION_B53.md): networks that read one component of the BES array, and training
under a fixed budget.  20 GPU runs, after B.39 and B.47.

Arms, in the registered order:
  C  the history-only twin with extra inputs to its T_i branch (model v2hx = v2h whose T_i encoder also
     reads CES_SEQ_EXTRA_TI), initialisation = split seed; tables C (c), D (g1 g2 h1 h2), CD (all five)
     from make_features.py, standardized with the training discharges of the split            12 runs
     -> data/.b53_{C|D|CD}_cut_s{seed}_i{seed}   (train_b53.py -> eval_b53.py)
  F  the estimator (causal variant v2, CES_PER_SHOT_NORM=0) and the twin v2h, exactly 40 epochs, final
     weights kept (CES_SEQ_FIXED_EPOCHS=1), initialisation = split seed                          8 runs
     -> data/.b53F_{v2|v2h}_cut_s{seed}_i{seed}  (seq/train_seq.py -> seq/eval_seq.py)

The environment of every run is run_b39.split_env of the network it is paired with (the twin of B.39 for
arm C, the network itself for arm F); arm C adds CES_SEQ_EXTRA_TI and sets CES_SEQ_MODEL=v2hx, arm F
sets CES_SEQ_EPOCHS=40 and CES_SEQ_FIXED_EPOCHS=1; the output directory is the run's.  Nothing else
moves (check_b53.py checks (4) and (5)).  A run whose comparison_errors_test.npz exists is skipped.  The
batch never starts while a run_b39.py or run_b47.py process is alive.

Usage (repo root):
  py ces_prediction/experiments/b53_components/make_features.py            # CPU, once (tables)
  py ces_prediction/experiments/b53_components/check_b53.py                # CPU, checks before any run
  py ces_prediction/experiments/b53_components/run_b53.py --dry-run
  py ces_prediction/experiments/b53_components/run_b53.py --wait-for-b39 > data/.b53_batch.log 2>&1
  py ces_prediction/experiments/b53_components/analyze_b53.py
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
COMBOS = ("C", "D", "CD")
K = {"C": 1, "D": 4, "CD": 5}
F_MODELS = ("v2", "v2h")
F_EPOCHS = "40"
ARMS = ("C", "F")
RUNS_OUT = DATA / ".b53_runs.json"
FEAT_ROOT = DATA / ".b53_feat"
TRAIN_C, EVAL_C = HERE / "train_b53.py", HERE / "eval_b53.py"
POLL_S, STATUS_EVERY_S, GPU_IDLE_MIB, GPU_IDLE_POLLS = 60, 600, 1000, 3
GUARDED = ("run_b39.py", "run_b47.py")


# ------------------------------------------------------------------------------ directories / env
def feat_dir(combo, seed):
    return FEAT_ROOT / f"{combo}_s{seed}"


def c_dir(combo, seed):
    return DATA / f".b53_{combo}_cut_s{seed}_i{seed}"


def f_dir(model, seed):
    return DATA / f".b53F_{model}_cut_s{seed}_i{seed}"


def twin_env(seed, out):
    """The env of the B.39 twin (v2h, cut, cap 80, initialisation = split seed) with the output set."""
    return B39.split_env("v2h", seed, seed, out)


def c_env(combo, seed, out):
    """Arm C: the twin's env plus the feature table of this split; combo None = the twin's env itself."""
    env = twin_env(seed, out)
    if combo is not None:
        env["CES_SEQ_EXTRA_TI"] = str(feat_dir(combo, seed))
        env["CES_SEQ_MODEL"] = "v2hx"
    return env


def f_env(model, seed, out):
    """Arm F: B.39's env of this network (v2: CES_PER_SHOT_NORM=0; v2h: the twin's) with 40 fixed epochs."""
    env = B39.split_env(model, seed, seed, out)
    env["CES_SEQ_EPOCHS"] = F_EPOCHS
    env["CES_SEQ_FIXED_EPOCHS"] = "1"
    return env


def b39_reference(arm, model, seed):
    """(directory, env) of the B.39 run each B.53 run is compared with in the dry-run diff."""
    if arm == "C" or model == "v2h":
        d = B39.twin_dir(seed, seed)
        return d, B39.split_env("v2h", seed, seed, d)
    d = B39.split_run_dir(model, seed, seed)
    return d, B39.split_env(model, seed, seed, d)


def plan(arms):
    """(arm, label, out, model, seed, combo, env_fn, train_script, eval_script) in the registered order."""
    runs = []
    if "C" in arms:
        for s in SEEDS:
            for c in COMBOS:
                out = c_dir(c, s)
                runs.append(("C", f"C s{s} {c}", out, "v2hx", s, c,
                             lambda c=c, s=s, o=out: c_env(c, s, o), TRAIN_C, EVAL_C))
    if "F" in arms:
        for s in SEEDS:
            for m in F_MODELS:
                out = f_dir(m, s)
                runs.append(("F", f"F s{s} {m} 40 epochs", out, m, s, None,
                             lambda m=m, s=s, o=out: f_env(m, s, o), SEQ_DIR / "train_seq.py", SEQ_DIR / "eval_seq.py"))
    return runs


def done(out):
    return (out / "comparison_errors_test.npz").exists()


# ------------------------------------------------------------------------------ guards
def guarded_processes():
    """PIDs and command lines of live python processes whose command line names run_b39.py or run_b47.py."""
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
    print("[b53] waiting for B.39 and B.47 to end (no run_b39.py / run_b47.py process, GPU memory used < "
          f"{GPU_IDLE_MIB} MiB on {GPU_IDLE_POLLS} consecutive polls)", flush=True)
    while True:
        procs = guarded_processes()
        used = gpu_used_mib()
        idle = idle + 1 if (not procs and used is not None and used < GPU_IDLE_MIB) else 0
        if idle >= GPU_IDLE_POLLS:
            print(f"[b53] B.39 and B.47 have ended ({(time.time() - t0) / 3600:.2f} h waited; GPU used {used} MiB)", flush=True)
            return
        if time.time() - last >= STATUS_EVERY_S:
            last = time.time()
            print(f"[b53] {time.strftime('%Y-%m-%d %H:%M')} waiting: guarded processes {len(procs)}, "
                  f"GPU used {used} MiB, idle polls {idle}", flush=True)
        time.sleep(POLL_S)


def refuse_if_earlier_alive():
    procs = guarded_processes()
    if procs:
        raise SystemExit(f"FATAL: {sorted({g for _, c in procs for g in GUARDED if g in c})} alive (pids "
                         f"{[p for p, _ in procs]}); B.53 never runs beside B.39 or B.47. Use --wait-for-b39.")


def preflight(runs, live):
    if not any(DATA.glob("s*.csv")):
        raise SystemExit("FATAL: no shot CSVs in data/ -- real data required, aborting.")
    for s in SEEDS:   # the B.39 runs every B.53 run is read against
        for d in (B39.twin_dir(s, s), B39.split_run_dir("v2", s, s), B39.split_run_dir("v2b", s, s)):
            if not (d / "comparison_errors_test.npz").exists():
                print(f"[b53] note: the B.39 reference {d.name} has no comparison_errors_test.npz yet", flush=True)
    for r in runs:
        env = r[6]()
        for needed in (Path(env["CES_SPLIT_DIR"]) / "split_manifest.json", Path(env["CES_CONTROL_METRICS"]), r[7], r[8]):
            if not Path(needed).exists():
                raise SystemExit(f"FATAL: {r[1]}: missing {needed}")
        if not Path(env["CES_OUTPUT_DIR"]).name.startswith(".b53"):
            raise SystemExit(f"FATAL: {r[1]}: output directory outside data/.b53*")
        if r[0] == "C":
            meta_p = Path(env["CES_SEQ_EXTRA_TI"]) / "feature_meta.json"
            if not meta_p.exists():
                raise SystemExit(f"FATAL: {r[1]}: feature table {meta_p.parent} missing; run make_features.py")
            meta = json.loads(meta_p.read_text(encoding="utf-8"))
            if meta["k"] != K[r[5]] or meta["split_seed"] != r[4] or meta["files"] != 641:
                raise SystemExit(f"FATAL: {r[1]}: feature table {meta_p.parent} does not match (k {meta['k']}, "
                                 f"split {meta['split_seed']}, files {meta['files']})")
        if live and env.get("CUDA_VISIBLE_DEVICES", None) == "":
            raise SystemExit("FATAL: CUDA_VISIBLE_DEVICES is empty in the shell; the runs would not see the GPU")


# ------------------------------------------------------------------------------ one run
def execute(arm, label, out, model, seed, combo, env, train_script, eval_script):
    out.mkdir(parents=True, exist_ok=True)
    weights, metrics = out / "weights" / "seq_lstm.pth", out / "metrics.json"
    npz, cm = out / "comparison_errors_test.npz", out / "comparison_metrics.json"
    train_art = (weights, metrics) + ((out / "b53_features.json",) if arm == "C" else ())
    rec = {"run": out.name, "arm": arm, "model": model, "split": seed, "init": seed, "combo": combo, "status": "ok"}
    t0 = time.time()
    print(f"[b53] === {out.name}", flush=True)
    for name, cmd, artifacts in (("train", [train_script], train_art), ("eval(test)", [eval_script], (npz, cm))):
        if all(a.exists() for a in artifacts):
            continue
        if not run_step(cmd, env, out / "run.log", name, artifacts=artifacts):
            rec["status"] = f"{name}_failed"
            break
    if metrics.exists():
        m = json.loads(metrics.read_text(encoding="utf-8"))
        rec.update({"epochs_run": m["epochs_run"], "best_epoch": m["best_epoch"], "n_params": m["n_params"],
                    "best_val_masked_mse": m["best_val_masked_mse"], "fixed_epochs": m["fixed_epochs"],
                    "extra_ti_channels": m["extra_ti_channels"], "seq_model": m["seq_model"]})
        if arm == "F" and rec["status"] == "ok" and (m["epochs_run"] != int(F_EPOCHS) or not m["fixed_epochs"]):
            rec["status"] = "budget_mismatch"
        if arm == "C" and rec["status"] == "ok" and m["extra_ti_channels"] != K[combo]:
            rec["status"] = "channels_mismatch"
    rec.update({"label": label, "minutes": round((time.time() - t0) / 60, 1)})
    return rec


# ------------------------------------------------------------------------------ dry run
def ces_vars(env):
    return {k: v for k, v in env.items() if k.startswith("CES_")}


def env_diff(mine, ref):
    keys = sorted(set(mine) | set(ref))
    return [(k, ref.get(k, "<unset>"), mine.get(k, "<unset>")) for k in keys if mine.get(k) != ref.get(k)]


def print_env_diff(runs):
    reps = {}
    for r in runs:      # one representative per arm and network
        reps.setdefault((r[0], r[3]), r)
    for (arm, model), r in reps.items():
        mine = ces_vars(r[6]())
        d_ref, e_ref = b39_reference(arm, model, r[4])
        print(f"\n--- env of {r[1]} ({r[2].name})")
        for k in sorted(mine):
            print(f"    {k}={mine[k]}")
        d = env_diff(mine, ces_vars(e_ref))
        print(f"  diff against the B.39 env of {d_ref.name} ({len(d)} variables):")
        for k, a, b in d:
            print(f"    {k}: {a}  ->  B.53 {b}")
    print("\n--- every run: variables that differ from its B.39 reference (CES_OUTPUT_DIR omitted)")
    for r in runs:
        d = [x for x in env_diff(ces_vars(r[6]()), ces_vars(b39_reference(r[0], r[3], r[4])[1])) if x[0] != "CES_OUTPUT_DIR"]
        print(f"    {r[2].name:26s} " + "; ".join(f"{k}={b}" for k, _, b in d))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="print the plan and the env diff; start nothing")
    ap.add_argument("--arms", nargs="*", default=list(ARMS), choices=list(ARMS))
    ap.add_argument("--wait-for-b39", action="store_true", help="wait until neither B.39 nor B.47 runs and the GPU is idle")
    ap.add_argument("--min-free-mb", type=int, default=2000)
    args = ap.parse_args()
    arms = [a for a in ARMS if a in args.arms]
    runs = plan(arms)
    preflight(runs, live=not args.dry_run)
    if args.dry_run:
        for i, r in enumerate(runs, 1):
            arm, label, out, model, seed, combo = r[:6]
            print(f"{i:2d} arm {arm} | {out.name:26s} | model {model:4s} | split {seed:3d} | init {seed:3d} | "
                  f"table {'-' if combo is None else feat_dir(combo, seed).name:7s} | train {r[7].name} | "
                  f"eval {r[8].name} | artifacts exist: {done(out)}")
        print(f"{len(runs)} runs: " + ", ".join(f"{a} {sum(r[0] == a for r in runs)}" for a in arms))
        print_env_diff(runs)
        return
    if args.wait_for_b39:
        wait_for_earlier()
    refuse_if_earlier_alive()
    free = B29.gpu_free_mb()
    print(f"[b53] GPU free memory: {free} MiB", flush=True)
    if free is None or free < args.min_free_mb:
        raise SystemExit(f"FATAL: GPU free memory {free} MiB < {args.min_free_mb} MiB -- B.53 is never run on CPU.")
    t0 = time.time()
    records = json.loads(RUNS_OUT.read_text(encoding="utf-8")) if RUNS_OUT.exists() else []
    stopped, started = set(), set()
    B29.keep_awake(True)
    try:
        for i, r in enumerate(runs, 1):
            arm, label, out = r[0], r[1], r[2]
            if arm in stopped or done(out):
                continue
            refuse_if_earlier_alive()
            first = arm not in started
            started.add(arm)
            rec = execute(*r[:6], r[6](), r[7], r[8])
            records = [x for x in records if x["run"] != rec["run"]] + [rec]
            RUNS_OUT.write_text(json.dumps(records, indent=1), encoding="utf-8")
            print(f"[b53] {i}/{len(runs)} {rec} ({(time.time() - t0) / 3600:.2f} h)", flush=True)
            if rec["status"] != "ok":
                if first:
                    print(f"[b53] the first run of arm {arm} failed; arm {arm} is stopped, the batch goes on", flush=True)
                    stopped.add(arm)
                else:
                    print(f"[b53] FAILED (logged, the batch goes on): {rec['run']} ({rec['status']})", flush=True)
    finally:
        B29.keep_awake(False)
    bad = [x for x in records if x["status"] != "ok"]
    for x in bad:
        print(f"[b53] FAILED: {x['run']} ({x['status']})", flush=True)
    if stopped:
        print(f"[b53] arms stopped after a failed first run: {sorted(stopped)}", flush=True)
    print(f"[b53] wall time {(time.time() - t0) / 3600:.2f} h; next: py ces_prediction/experiments/b53_components/analyze_b53.py",
          flush=True)
    sys.exit(1 if (bad or stopped) else 0)


if __name__ == "__main__":
    main()
