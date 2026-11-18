"""B.65 runner (PREREGISTRATION_B65.md): the network (`v2`) and its history-only twin (`v2h`) on the inclusive
population (values above 3 keV kept: CES_TI_SPIKE_CUT_EV=0) under the pooled protocol of B.39, re-partitions r2, r3,
r4 x five folds = 30 GPU runs (train_seq -> eval_seq on the fold's test discharges), one initialisation per fold as in
B.39.

The environment of every run is `run_b39.pooled_env` of its model and fold with exactly these changes:
  CES_TI_SPIKE_CUT_EV   3000 -> 0                         (the population)
  CES_CONTROL_METRICS   .b27_ctrl_cut_r{r}_f{k} -> .b27_ctrl_incl_r{r}_f{k}   (the harness statistics of the same
                        population: eval_seq rounds targets, history and references through them)
  CES_OUTPUT_DIR        .b39p_* -> .b65_*
Nothing is written outside `data/.b65_*` (smoke: `data/.smoke_b65_*`); a run whose artifacts exist is skipped, so a
re-launch resumes.  The batch never starts while a run_b59.py, run_b47.py, run_b53.py or run_b64.py process is alive.

Usage (repo root):
  py ces_prediction/experiments/b65_inclusive/run_b65.py --dry-run              # plan, env diff, overlap check
  py ces_prediction/experiments/b65_inclusive/run_b65.py --smoke                # GPU plumbing: 1 epoch, 40 files
  py ces_prediction/experiments/b65_inclusive/run_b65.py [--wait-for-earlier] > data/.b65_batch.log 2>&1
  py ces_prediction/experiments/b65_inclusive/refs_b65.py                       # CPU references (no network)
  py ces_prediction/experiments/b65_inclusive/analyze_b65.py                    # statements I1..I4 + readouts
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
MODELS = ("v2", "v2h")
POP = "incl"
CUT_EV = str(int(B27.POPS[POP]))               # "0"
RUNS_OUT = DATA / ".b65_runs.json"
SMOKE_RUNS = ((2, 0, "v2"), (2, 0, "v2h"))
POLL_S, STATUS_EVERY_S, GPU_IDLE_MIB, GPU_IDLE_POLLS = 60, 600, 1000, 3
GUARDED = ("run_b59.py", "run_b47.py", "run_b53.py", "run_b64.py")
ALLOWED_DIFF = ["CES_CONTROL_METRICS", "CES_OUTPUT_DIR", "CES_TI_SPIKE_CUT_EV"]


def run_dir(model, r, k, smoke=False):
    return DATA / (f".smoke_b65_{model}_{POP}_r{r}_f{k}" if smoke else f".b65_{model}_{POP}_r{r}_f{k}")


def ctrl_metrics(r, k):
    return B27.ctrl_dir(POP, r, k) / "metrics.json"


def run_env(model, r, k, smoke=False):
    out = run_dir(model, r, k, smoke)
    env = B39.pooled_env(model, r, k, out)
    env["CES_TI_SPIKE_CUT_EV"] = CUT_EV
    env["CES_CONTROL_METRICS"] = str(ctrl_metrics(r, k))
    if smoke:
        env.update(B29.SMOKE_OVERRIDES)
        env.update({"CES_SEQ_EPOCHS": "1", "CES_SEQ_MAX_FILES": "40"})
    return env


def plan(smoke=False):
    if smoke:
        return list(SMOKE_RUNS)
    return [(r, k, m) for r in REPEATS for k in range(B27.K) for m in MODELS]


def done(out):
    return all((out / a).exists() for a in ("weights/seq_lstm.pth", "metrics.json", "comparison_errors_test.npz",
                                            "comparison_metrics.json"))


def overlap_check(r, k):
    """No file in two of training, validation and test (the B.27 fold manifest, shared by both populations)."""
    m = json.loads((B27.split_dir(r, k) / "split_manifest.json").read_text(encoding="utf-8"))
    tr, va, te = set(m["train_files"]), set(m["val_files"]), set(m["test_files"])
    for a, b, what in ((tr, va, "training / validation"), (tr, te, "training / test"), (va, te, "validation / test")):
        if a & b:
            raise SystemExit(f"FATAL: r{r} f{k}: {what} share {sorted(a & b)[:3]}")
    return {"train": len(tr), "val": len(va), "test": len(te)}


# ------------------------------------------------------------------------------ guards (as run_b64.py)
def guarded_processes():
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
    print(f"[b65] waiting for {', '.join(GUARDED)} to end (no such process, GPU memory used < "
          f"{GPU_IDLE_MIB} MiB on {GPU_IDLE_POLLS} consecutive polls)", flush=True)
    while True:
        procs = guarded_processes()
        used = gpu_used_mib()
        idle = idle + 1 if (not procs and used is not None and used < GPU_IDLE_MIB) else 0
        if idle >= GPU_IDLE_POLLS:
            print(f"[b65] earlier batches have ended ({(time.time() - t0) / 3600:.2f} h waited; GPU used {used} MiB)", flush=True)
            return
        if time.time() - last >= STATUS_EVERY_S:
            last = time.time()
            print(f"[b65] {time.strftime('%Y-%m-%d %H:%M')} waiting: guarded processes {len(procs)}, "
                  f"GPU used {used} MiB, idle polls {idle}", flush=True)
        time.sleep(POLL_S)


def refuse_if_earlier_alive():
    procs = guarded_processes()
    if procs:
        raise SystemExit(f"FATAL: {sorted({g for _, c in procs for g in GUARDED if g in c})} alive (pids "
                         f"{[p for p, _ in procs]}); B.65 never runs beside them. Use --wait-for-earlier.")


def preflight(runs, live):
    if not any(DATA.glob("s*.csv")):
        raise SystemExit("FATAL: no shot CSVs in data/ -- real data required, aborting.")
    for r in REPEATS:
        for k in range(B27.K):
            overlap_check(r, k)
            if not ctrl_metrics(r, k).exists():
                raise SystemExit(f"FATAL: missing harness statistics {ctrl_metrics(r, k)}")
            for m in MODELS:
                d = B39.pooled_run_dir(m, r, k)
                if not (d / "comparison_errors_test.npz").exists():
                    raise SystemExit(f"FATAL: the B.39 run {d.name} (paired cut run) is missing")
    for (r, k, m) in runs:
        env = run_env(m, r, k)
        if env["CES_TI_SPIKE_CUT_EV"] != "0":
            raise SystemExit("FATAL: population is not the inclusive one")
    if live and os.environ.get("CUDA_VISIBLE_DEVICES", None) == "":
        raise SystemExit("FATAL: CUDA_VISIBLE_DEVICES is empty in the shell; the runs would not see the GPU")


# ------------------------------------------------------------------------------ one run
def execute(r, k, model, smoke):
    out = run_dir(model, r, k, smoke)
    out.mkdir(parents=True, exist_ok=True)
    env = run_env(model, r, k, smoke)
    if not Path(env["CES_OUTPUT_DIR"]).name.startswith(".smoke_b65_" if smoke else ".b65_"):
        raise SystemExit("FATAL: output directory outside data/.b65_*")
    weights, metrics = out / "weights" / "seq_lstm.pth", out / "metrics.json"
    npz, cm = out / "comparison_errors_test.npz", out / "comparison_metrics.json"
    rec = {"run": out.name, "model": model, "repeat": r, "fold": k, "init": int(env["CES_INIT_SEED"]), "status": "ok"}
    (out / "b65_env.json").write_text(json.dumps(ces_vars(env), indent=1, sort_keys=True), encoding="utf-8")
    t0 = time.time()
    print(f"[b65] === {out.name}", flush=True)
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


def b39_recorded_check(runs):
    """The B.39 reference env is recomputed (run_b39.pooled_env); check it against the fields every stored B.39 run
    recorded in its metrics.json (the env itself is not stored), so the diff above is a diff against what ran."""
    bad = []
    for (r, k, m) in runs:
        ref = B39.pooled_env(m, r, k, B39.pooled_run_dir(m, r, k))
        rec = json.loads((B39.pooled_run_dir(m, r, k) / "metrics.json").read_text(encoding="utf-8"))
        want = {"seq_model": m, "seed": int(ref["CES_SEED"]), "init_seed": int(ref["CES_INIT_SEED"]),
                "per_shot_input_norm": ref["CES_PER_SHOT_NORM"] == "1", "drop_stuck_targets": ref["CES_DROP_STUCK_TARGETS"] == "1",
                "fixed_epochs": ref["CES_SEQ_FIXED_EPOCHS"] == "1", "loss_kind": ref["CES_SEQ_LOSS"],
                "select_metric": ref["CES_SEQ_SELECT"], "zero_mc": ref["CES_SEQ_ZERO_MC"] == "1",
                "per_shot_norm_expanding": ref["CES_SEQ_NORM_EXPANDING"] == "1"}
        diff = {key: (rec.get(key), v) for key, v in want.items() if rec.get(key) != v}
        if diff:
            bad.append((B39.pooled_run_dir(m, r, k).name, diff))
    if bad:
        raise SystemExit(f"FATAL: the recomputed B.39 env disagrees with what B.39 recorded: {bad[:3]}")
    return (f"{len(runs)} of {len(runs)} B.39 runs recorded the model, split seed, initialisation seed, input "
            "normalisation, held-value drop, fixed-epoch flag, loss, selection metric, Mirnov and expanding-norm flags "
            "of the recomputed env")


def dry_run(runs):
    print("--- no-overlap check (training / validation / test), every fold")
    for r in REPEATS:
        for k in range(B27.K):
            print(f"    r{r} f{k}: {overlap_check(r, k)} -> passes")
    print("\n--- the planned runs")
    for i, (r, k, m) in enumerate(runs, 1):
        out = run_dir(m, r, k)
        env = run_env(m, r, k)
        print(f"{i:3d} {out.name:24s} | model {m:3s} | r{r} f{k} | init {env['CES_INIT_SEED']} | seed {env['CES_SEED']} | "
              f"artifacts exist: {done(out)}")
    print(f"{len(runs)} runs: " + ", ".join(f"{m} {sum(x[2] == m for x in runs)}" for m in MODELS))
    for m in MODELS:
        r, k = runs[0][:2]
        mine = ces_vars(run_env(m, r, k))
        ref = ces_vars(B39.pooled_env(m, r, k, B39.pooled_run_dir(m, r, k)))
        print(f"\n--- env of {run_dir(m, r, k).name}")
        for key in sorted(mine):
            print(f"    {key}={mine[key]}")
        print(f"  diff against the B.39 run {B39.pooled_run_dir(m, r, k).name}:")
        for key, a, b in env_diff(mine, ref):
            print(f"    {key}: {a}  ->  B.65 {b}")
    print("\n--- every run: variables that differ from the B.39 run of the same model and fold")
    bad = []
    for (r, k, m) in runs:
        d = env_diff(ces_vars(run_env(m, r, k)), ces_vars(B39.pooled_env(m, r, k, B39.pooled_run_dir(m, r, k))))
        keys = sorted(x[0] for x in d)
        if keys != ALLOWED_DIFF:
            bad.append((run_dir(m, r, k).name, keys))
    print(f"    {len(runs) - len(bad)} of {len(runs)} runs differ in exactly {', '.join(ALLOWED_DIFF)}" +
          ("" if not bad else f"; OTHERS: {bad}"))
    if bad:
        raise SystemExit("FATAL: an environment differs beyond the population and the directories")
    print("\n--- the recomputed B.39 environment against what the stored B.39 runs recorded (metrics.json)")
    print(f"    {b39_recorded_check(runs)}")
    t = []
    for r in REPEATS:
        for k in range(B27.K):
            for m in MODELS:
                p = B39.pooled_run_dir(m, r, k) / "metrics.json"
                rec = [x for x in json.loads((DATA / ".b39_runs.json").read_text(encoding="utf-8"))
                       if x["run"] == B39.pooled_run_dir(m, r, k).name]
                if rec:
                    t.append(rec[0]["minutes"])
    print(f"\n--- time of the 30 paired B.39 cut runs: {sum(t):.0f} min in total ({len(t)} records), "
          f"mean {sum(t) / max(len(t), 1):.1f} min per run")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="print the plan, the env diff and the overlap check")
    ap.add_argument("--smoke", action="store_true", help="GPU plumbing: r2 f0, v2 and v2h, 1 epoch, 40 files")
    ap.add_argument("--wait-for-earlier", action="store_true",
                    help="wait until no run_b59 / run_b47 / run_b53 / run_b64 process is alive and the GPU is idle")
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
    print(f"[b65] GPU free memory: {free} MiB", flush=True)
    if free is None or free < args.min_free_mb:
        raise SystemExit(f"FATAL: GPU free memory {free} MiB < {args.min_free_mb} MiB -- B.65 is never run on CPU.")
    t0 = time.time()
    out_json = DATA / ".smoke_b65_runs.json" if args.smoke else RUNS_OUT
    records = json.loads(out_json.read_text(encoding="utf-8")) if out_json.exists() else []
    B29.keep_awake(True)
    try:
        for i, (r, k, m) in enumerate(runs, 1):
            if not args.smoke and done(run_dir(m, r, k)) and any(x["run"] == run_dir(m, r, k).name and
                                                                 x["status"] == "ok" for x in records):
                continue
            refuse_if_earlier_alive()
            rec = execute(r, k, m, args.smoke)
            records = [x for x in records if x["run"] != rec["run"]] + [rec]
            out_json.write_text(json.dumps(records, indent=1), encoding="utf-8")
            print(f"[b65] {i}/{len(runs)} {rec} ({(time.time() - t0) / 3600:.2f} h)", flush=True)
    finally:
        B29.keep_awake(False)
    bad = [x for x in records if x["status"] != "ok"]
    capped = [x["run"] for x in records if x.get("cap_bound")]
    for x in bad:
        print(f"[b65] FAILED: {x['run']} ({x['status']})", flush=True)
    if capped and not args.smoke:
        print(f"[b65] reached the epoch cap (reported): {capped}", flush=True)
    print(f"[b65] wall time {(time.time() - t0) / 3600:.2f} h; next: "
          "py ces_prediction/experiments/b65_inclusive/analyze_b65.py", flush=True)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
