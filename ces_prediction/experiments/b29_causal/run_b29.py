"""B.29 runner -- PREREGISTRATION_B29.md: the fast-channel increment on the main model and
causal standardization.

Four arms, each ONE controlled variable against the frozen seq_v2 backbone, scored on the
backbone's exact TEST rows (same split manifests, same harness statistics):

  v2h     history-only seq_v2 (T_i encoder trained without BES / ECEI / MC)      8 runs
  norm0   seq_v2 with CES_PER_SHOT_NORM=0 (train-file statistics)                8 + 8 runs
  expand  seq_v2 with the expanding-window per-discharge standardizer            8 + 8 runs
  mlp22   memoryless MLP on the same 22 inputs                                   8 runs

"8" = split seeds {42, 1, 7, 123} x populations {cut, incl}, file-level splits (controls
`.b1_seqv2_s{s}_i{s}` / `.b5i_seqv2_s{s}`). "+ 8" = the B.5 temporal (campaign) split, four
initialisations x both populations (controls `.b5_camp_{pop}_seq_s{i}`). 48 runs, sequential.

Per run: train_seq -> eval_seq (TEST) -> bootstrap_compare (arm vs PCHIP / linear /
persistence / GP / causal GP) -> paired_model_compare twice (arm vs backbone, backbone vs arm).
Nothing is written outside `data/.b29_*` (smoke: `data/.smoke_b29_*`); no tracked file is
touched; every step is skipped when its artifact exists, so a re-run resumes.

The subprocess env starts from the shell with EVERY `CES_*` variable removed and then pins each
one explicitly (experiments/README.md non-negotiable 1): nothing is inherited.

Usage (repo root):
  py ces_prediction/experiments/b29_causal/run_b29.py --smoke            # CPU plumbing check
  py ces_prediction/experiments/b29_causal/run_b29.py                    # full batch (GPU)
  py ces_prediction/experiments/b29_causal/run_b29.py --arms v2h norm0   # a subset
  py ces_prediction/experiments/b29_causal/analyze_b29.py                # rules + tables (CPU)
"""

import argparse
import ctypes
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
CES_DIR = REPO_ROOT / "ces_prediction"
SEQ_DIR = CES_DIR / "experiments" / "seq"
PAIRED = CES_DIR / "experiments" / "paired_model_compare.py"
DATA = REPO_ROOT / "data"
sys.path.insert(0, str(CES_DIR / "experiments" / "b1_gate"))
from run_b1_gate import GATE_ENV, SMOKE_OVERRIDES, run_step  # noqa: E402

SEEDS = (42, 1, 7, 123)
POPS = {"cut": "3000", "incl": "0"}
EPOCH_CAP = "30"            # the backbone's (train_seq default at B.1 / B.5 time), pinned here

# arm -> the ONE variable it changes. `families`: "split" = four file-level splits,
# "camp" = the B.5 temporal split (four initialisations).
ARMS = {
    "v2h":    {"model": "v2h",   "per_shot": "1", "expanding": "0", "families": ("split",)},
    "norm0":  {"model": "v2",    "per_shot": "0", "expanding": "0", "families": ("split", "camp")},
    "expand": {"model": "v2",    "per_shot": "1", "expanding": "1", "families": ("split", "camp")},
    "mlp22":  {"model": "mlp22", "per_shot": "1", "expanding": "0", "families": ("split",)},
}

# family -> pop -> (frozen backbone run, split dir, harness-stats run), all read-only.
CONTROLS = {
    "split": {
        "cut": (".b1_seqv2_s{s}_i{s}", ".b1_w2cut_split_s{s}", ".b1_w2cut_s{s}"),
        "incl": (".b5i_seqv2_s{s}", ".b1_manifest_s{s}", ".wsweep_hf_w2_s{s}"),
    },
    "camp": {
        "cut": (".b5_camp_cut_seq_s{s}", ".b5_camp_cut_split_s{s}", ".b5_camp_cut_win_s{s}"),
        "incl": (".b5_camp_incl_seq_s{s}", ".b5_camp_incl_split_s{s}", ".b5_camp_incl_win_s{s}"),
    },
}
RUNS_OUT = DATA / ".b29_runs.json"


def controls(family, pop, s):
    backbone, split, stats = (DATA / p.format(s=s) for p in CONTROLS[family][pop])
    return backbone, split, stats / "metrics.json"


def arm_dir(arm, family, pop, s, smoke=False):
    fam = "" if family == "split" else "camp_"
    return DATA / (f".smoke_b29_{arm}_{fam}{pop}_s{s}" if smoke else f".b29_{arm}_{fam}{pop}_s{s}")


def run_env(arm, family, pop, s, out, smoke):
    backbone, split, cm = controls(family, pop, s)
    env = {k: v for k, v in os.environ.items() if not k.startswith("CES_")}
    env.update(GATE_ENV)     # W = 2 harness, held-free, per-file cap 500, sample caps
    if smoke:
        env.update(SMOKE_OVERRIDES)
    a = ARMS[arm]
    env.update({
        # data treatment / population
        "CES_DATA_DIR": str(DATA),
        "CES_TI_SPIKE_CUT_EV": POPS[pop],
        "CES_PER_SHOT_NORM": a["per_shot"],
        "CES_SEQ_NORM_EXPANDING": a["expanding"],
        "CES_SEQ_ZERO_MC": "0",
        # model / optimisation: the backbone's regime
        "CES_SEQ_MODEL": a["model"],
        "CES_SEQ_LOSS": "mse",
        "CES_SEQ_SELECT": "mse",
        "CES_SEQ_EPOCHS": EPOCH_CAP,
        "CES_SEQ_BATCH": "16",
        "CES_SEQ_FIXED_EPOCHS": "0",
        "CES_SEQ_TRAIN_CTX": "0",
        "CES_SEQ_EVAL_CTX": "0",
        "CES_SEQ_DEVICE": "cuda",
        # pairing
        "CES_SEED": str(s),
        "CES_INIT_SEED": str(s),
        "CES_SPLIT_DIR": str(split),
        "CES_CONTROL_METRICS": str(cm),
        "CES_OUTPUT_DIR": str(out),
        "CES_SPLIT_TAG": "test",
    })
    if smoke:
        env.update({"CES_SEQ_EPOCHS": "1", "CES_SEQ_MAX_FILES": "40", "CES_SEQ_DEVICE": "cpu"})
    return env


def one_run(arm, family, pop, s, smoke):
    backbone, split, cm = controls(family, pop, s)
    for needed in (backbone / "comparison_errors_test.npz", split / "split_manifest.json", cm):
        if not needed.exists():
            raise SystemExit(f"FATAL: missing frozen control artifact {needed}")
    out = arm_dir(arm, family, pop, s, smoke)
    out.mkdir(parents=True, exist_ok=True)
    log = out / "run.log"
    env = run_env(arm, family, pop, s, out, smoke)
    rec = {"arm": arm, "family": family, "pop": pop, "seed": s, "out": str(out), "status": "ok"}
    t0 = time.time()
    print(f"[b29] === {out.name}", flush=True)
    weights, metrics = out / "weights" / "seq_lstm.pth", out / "metrics.json"
    npz = out / "comparison_errors_test.npz"
    steps = [
        ("train", [SEQ_DIR / "train_seq.py"], (weights, metrics)),
        ("eval(test)", [SEQ_DIR / "eval_seq.py"], (npz, out / "comparison_metrics.json")),
        ("bootstrap", [CES_DIR / "bootstrap_compare.py"], (out / "bootstrap_summary.json",)),
    ]
    if not smoke:   # smoke caps change the TEST rows, so there is no backbone to pair with
        steps += [
            ("paired-vs-backbone",
             [PAIRED, "--a", npz, "--b", backbone / "comparison_errors_test.npz",
              "--out", out / "paired_vs_backbone.json"], (out / "paired_vs_backbone.json",)),
            ("paired-backbone-vs-arm",
             [PAIRED, "--a", backbone / "comparison_errors_test.npz", "--b", npz,
              "--out", out / "paired_backbone_vs_arm.json"], (out / "paired_backbone_vs_arm.json",)),
        ]
    for label, cmd, artifacts in steps:
        if all(Path(a).exists() for a in artifacts):
            continue
        if not run_step(cmd, env, log, label, artifacts=artifacts):
            rec["status"] = f"{label}_failed"
            break
    if metrics.exists():
        m = json.loads(metrics.read_text(encoding="utf-8"))
        rec.update({"epochs_run": m["epochs_run"], "best_epoch": m["best_epoch"],
                    "cap_bound": m["epochs_run"] >= int(env["CES_SEQ_EPOCHS"]) and not smoke,
                    "n_params": m["n_params"]})
    rec["minutes"] = round((time.time() - t0) / 60, 1)
    return rec


def plan(arms, pops, seeds, families, smoke):
    """Order: arm (A -> B -> C -> D), then family, population, seed."""
    if smoke:
        # every arm on cut s42 (file-level), plus one temporal-split run for the camp path
        runs = [(a, "split", "cut", 42) for a in arms]
        if "expand" in arms:
            runs.append(("expand", "camp", "cut", 42))
        return runs
    return [(a, f, p, s) for a in arms for f in ARMS[a]["families"] if f in families
            for p in pops for s in seeds]


def gpu_free_mb():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=30).stdout
        return int(out.strip().splitlines()[0])
    except Exception:  # noqa: BLE001  (no GPU / no nvidia-smi: reported, decided by the caller)
        return None


def keep_awake(on):
    """Windows: hold the machine out of standby for the batch (B.27 r0 f1 lost 22 min to it)."""
    if sys.platform != "win32":
        return
    ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
    ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | (ES_SYSTEM_REQUIRED if on else 0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="CPU plumbing check into data/.smoke_b29_*")
    ap.add_argument("--resume", action="store_true", help="accepted; every step is artifact-checked anyway")
    ap.add_argument("--arms", nargs="*", default=list(ARMS), choices=list(ARMS))
    ap.add_argument("--pops", nargs="*", default=list(POPS), choices=list(POPS))
    ap.add_argument("--seeds", nargs="*", type=int, default=list(SEEDS))
    ap.add_argument("--families", nargs="*", default=["split", "camp"], choices=["split", "camp"])
    ap.add_argument("--min-free-mb", type=int, default=3000,
                    help="full batch only: abort unless the GPU has this much free memory")
    ap.add_argument("--list", action="store_true", help="print the run plan and exit")
    args = ap.parse_args()

    if not any(DATA.glob("s*.csv")):
        raise SystemExit("FATAL: no shot CSVs in data/ -- real data required, aborting.")
    runs = plan(args.arms, args.pops, args.seeds, args.families, args.smoke)
    if args.list:
        for r in runs:
            print(arm_dir(*r, smoke=args.smoke).name)
        print(f"{len(runs)} runs")
        return
    if not args.smoke:
        free = gpu_free_mb()
        print(f"[b29] GPU free memory: {free} MiB", flush=True)
        if free is None or free < args.min_free_mb:
            raise SystemExit(f"FATAL: GPU free memory {free} MiB < {args.min_free_mb} MiB -- the GPU is "
                             "busy or absent; B.29 is never run on CPU. Re-launch when it is idle.")

    t0 = time.time()
    records = []
    runs_out = DATA / ".smoke_b29_runs.json" if args.smoke else RUNS_OUT
    keep_awake(True)
    try:
        for i, r in enumerate(runs, 1):
            rec = one_run(*r, smoke=args.smoke)
            rec["batch_hours_so_far"] = round((time.time() - t0) / 3600, 2)
            records.append(rec)
            print(f"[b29] {i}/{len(runs)} {rec}", flush=True)
            runs_out.write_text(json.dumps(records, indent=1), encoding="utf-8")
    finally:
        keep_awake(False)
        print(f"[b29] wall time: {(time.time() - t0) / 3600:.2f} h for {len(records)} runs", flush=True)
    bad = [r for r in records if r["status"] != "ok"]
    for r in bad:
        print(f"[b29] FAILED: {r['out']} ({r['status']})", flush=True)
    if not args.smoke and not bad:
        print("[b29] next: py ces_prediction/experiments/b29_causal/analyze_b29.py", flush=True)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
