"""B.1 backbone-gate runner -- PREREGISTRATION_W2.md sections 4-5 (THESIS_RESULTS.md 8v order 2).

Everything runs under the confirmed protocol: W = 2, held-free, CES_TI fit-failure
cut at 3,000 eV, per-file cap 500, per-shot input standardization OFF for the
window family and ON for seq_v2 (part of its definition). No W = 4 artifact is a
control; the file-level split is pinned to the frozen W = 2 sweep manifests
(recovered from each sweep run's metrics.json) with a test-isolation assert.

Stages (sequential, each resumable):
  A. `w2cut` window control family, seeds {42, 1, 7, 123}: train -> compare(test)
     -> bootstrap. The bootstrap now also gates persistence / gp / gp_causal, so
     the claim-2 verdict (model vs causal GP, section 5) falls out of stage A.
  B. seq_v2 16-run grid: split seeds {42, 1, 7, 123} x init seeds {42, 1, 7, 123},
     each paired against its split's stage-A control.
  C. seq_v2 budget-equalized arm (CES_SEQ_FIXED_EPOCHS=1, 10 epochs, final
     weights), one run per split seed.
  D. gate summary -> data/.b1_gate_summary.json, applying the four pre-registered
     conditions of section 4 and the claim-2 rule of section 5.

Usage (repo root):
  py ces_prediction/experiments/b1_gate/run_b1_gate.py --smoke    # plumbing check
  py ces_prediction/experiments/b1_gate/run_b1_gate.py            # full batch
  py ces_prediction/experiments/b1_gate/run_b1_gate.py --resume   # skip finished runs
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
CES_DIR = REPO_ROOT / "ces_prediction"
SEQ_DIR = CES_DIR / "experiments" / "seq"
PAIRED = CES_DIR / "experiments" / "paired_model_compare.py"
DATA = REPO_ROOT / "data"

SEEDS = (42, 1, 7, 123)          # split seeds (frozen sweep manifests exist for these)
INIT_SEEDS = (42, 1, 7, 123)     # init seeds of the seq_v2 grid (16 runs)
SWEEP_OUT = {s: DATA / f".wsweep_hf_w2_s{s}" for s in SEEDS}

# The confirmed protocol (PREREGISTRATION_W2.md section 1). Base = the window
# sweep's FULL_ENV so the stage-A family differs from the frozen W = 2 sweep
# family by exactly one treatment: the spike cut.
GATE_ENV = {
    "CES_TEST_FRACTION": "0.15",
    "CES_WINDOW_SIZE": "2",
    "CES_EPOCHS": "10",
    "CES_BATCH_SIZE": "512",
    "CES_LR": "1e-3",
    "CES_VAL_FRACTION": "0.2",
    "CES_MAX_TRAIN_SAMPLES": "200000",
    "CES_MAX_VAL_SAMPLES": "40000",
    "CES_MAX_TEST_SAMPLES": "48000",
    "CES_TEMPORAL_SUBSETS": "1",
    "CES_MAX_SAMPLES_PER_FILE": "500",
    "CES_DROP_STUCK_TARGETS": "1",
    "CES_TI_SPIKE_CUT_EV": "3000",
    # Windows/MKL: survive console detach during a long batch.
    "FOR_DISABLE_CONSOLE_CTRL_HANDLER": "1",
    "KMP_HANDLE_SIGNALS": "0",
}
SMOKE_OVERRIDES = {
    "CES_EPOCHS": "1",
    "CES_MAX_TRAIN_SAMPLES": "4000",
    "CES_MAX_VAL_SAMPLES": "1200",
    "CES_MAX_TEST_SAMPLES": "1200",
    "CES_MAX_SAMPLES_PER_FILE": "50",
}
# Treatment-relevant vars that must never leak in from the shell.
AMBIENT_VARS = (
    "CES_ABLATE", "CES_FILE_SPLIT_FROM", "CES_MIN_SUBSET_SIZE", "CES_DATA_DIR",
    "CES_PER_SHOT_NORM", "CES_TI_SPIKE_CUT_EV", "CES_DROP_STUCK_TARGETS",
    "CES_SEQ_MODEL", "CES_SEQ_EPOCHS", "CES_SEQ_MAX_FILES", "CES_SEQ_DEVICE",
    "CES_SEQ_FIXED_EPOCHS", "CES_CONTROL_METRICS", "CES_INIT_SEED",
)

BOOTSTRAP_B = 10000
BOOTSTRAP_SEED = 12345


def run_step(cmd, env, log_path, label, artifacts=()):
    """rc != 0 still counts as success if every expected artifact is fresh
    (Windows CUDA teardown can crash the process after the real work is done)."""
    start = time.time()
    with open(log_path, "a", encoding="utf-8") as log:
        log.write(f"\n===== {label}: {' '.join(map(str, cmd))} =====\n")
        log.flush()
        proc = subprocess.run(
            [sys.executable, *map(str, cmd)],
            cwd=REPO_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
        )
    mins = (time.time() - start) / 60.0
    ok = proc.returncode == 0
    if not ok and artifacts and all(
        Path(a).exists() and Path(a).stat().st_mtime >= start - 1 for a in artifacts
    ):
        print(f"[b1]   {label}: rc={proc.returncode} but artifacts fresh -> OK ({mins:.1f} min)", flush=True)
        ok = True
    else:
        print(f"[b1]   {label}: rc={proc.returncode} ({mins:.1f} min)", flush=True)
    return ok


def base_env(smoke):
    env = os.environ.copy()
    for var in AMBIENT_VARS:
        env.pop(var, None)
    env.update(GATE_ENV)
    if smoke:
        env.update(SMOKE_OVERRIDES)
    return env


def names(kind, seed, init=None, smoke=False):
    tag = {"w2cut": f".b1_w2cut_s{seed}",
           "seq": f".b1_seqv2_s{seed}_i{init}",
           "eq": f".b1_seqv2eq_s{seed}"}[kind]
    if smoke:
        tag += "_smoke"
    return DATA / tag


def reconstruct_manifest(seed):
    """Recover the frozen W = 2 sweep split manifest from the sweep run's
    metrics.json (the sweep's split dirs were cleaned up after the batch)."""
    src = SWEEP_OUT[seed] / "metrics.json"
    manifest = json.loads(src.read_text(encoding="utf-8"))["split"]
    for k in ("train_files", "val_files", "test_files"):
        if not manifest.get(k):
            raise SystemExit(f"FATAL: {src} split manifest lacks {k}")
    dst_dir = DATA / f".b1_manifest_s{seed}"
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / "split_manifest.json"
    dst.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return dst, sorted(manifest["test_files"])


def stage_a_one(seed, smoke, resume):
    out_dir = names("w2cut", seed, smoke=smoke)
    split_dir = DATA / (f".b1_w2cut_split_s{seed}" + ("_smoke" if smoke else ""))
    done = out_dir / "bootstrap_summary.json"
    if resume and not smoke and done.exists():
        print(f"[b1] === w2cut s{seed}: complete, skipping (--resume)", flush=True)
        return {"seed": seed, "out_dir": str(out_dir), "status": "ok", "resumed": True}
    out_dir.mkdir(parents=True, exist_ok=True)
    log = out_dir / "run.log"
    manifest_path, src_test_files = reconstruct_manifest(seed)

    env = base_env(smoke)
    env.update({
        "CES_SEED": str(seed),
        "CES_INIT_SEED": str(seed),
        "CES_SPLIT_DIR": str(split_dir),
        "CES_OUTPUT_DIR": str(out_dir),
        "CES_SPLIT_TAG": "test",
        "CES_FILE_SPLIT_FROM": str(manifest_path),
        "CES_PER_SHOT_NORM": "0",   # window family: OFF (PREREGISTRATION_W2.md sec.1)
    })

    print(f"[b1] === w2cut s{seed} -> {out_dir.name}", flush=True)
    record = {"seed": seed, "out_dir": str(out_dir), "split_dir": str(split_dir), "status": "ok"}
    start = time.time()
    try:
        if not run_step([CES_DIR / "train.py"], env, log, "train",
                        artifacts=(out_dir / "metrics.json", out_dir / "weights" / "multimodal_ces.pth")):
            record["status"] = "train_failed"
            return record
        # Test isolation: the pinned split must reproduce the sweep's test list.
        got = sorted(json.loads((split_dir / "split_manifest.json").read_text(encoding="utf-8"))["test_files"])
        if not smoke and got != src_test_files:
            record["status"] = "test_isolation_failed"
            print(f"[b1]   FATAL: test file list moved under the cut for seed {seed}", flush=True)
            return record
        record["test_isolation"] = "ok" if got == src_test_files else "smoke-skip"
        if not run_step([CES_DIR / "compare_baselines.py"], env, log, "compare(test)",
                        artifacts=(out_dir / "comparison_errors_test.npz",
                                   out_dir / "comparison_metrics.json")):
            record["status"] = "compare_failed"
            return record
        if not smoke and not run_step([CES_DIR / "bootstrap_compare.py"], env, log, "bootstrap",
                                      artifacts=(done,)):
            record["status"] = "bootstrap_failed"
            return record
    finally:
        record["minutes"] = round((time.time() - start) / 60.0, 1)
    return record


def stage_bc_one(seed, init, smoke, resume, equalized=False):
    kind = "eq" if equalized else "seq"
    out_dir = names(kind, seed, init=init, smoke=smoke)
    control_out = names("w2cut", seed, smoke=smoke)
    split_dir = DATA / (f".b1_w2cut_split_s{seed}" + ("_smoke" if smoke else ""))
    paired_json = out_dir / "paired_vs_w2cut.json"
    if resume and not smoke and paired_json.exists():
        print(f"[b1] === {out_dir.name}: complete, skipping (--resume)", flush=True)
        return {"seed": seed, "init": init, "out_dir": str(out_dir), "status": "ok", "resumed": True}
    out_dir.mkdir(parents=True, exist_ok=True)
    log = out_dir / "run.log"

    env = base_env(smoke)
    env.update({
        "CES_SEED": str(seed),
        "CES_INIT_SEED": str(init),
        "CES_SPLIT_DIR": str(split_dir),     # read-only: stage-A manifest supplies the split
        "CES_OUTPUT_DIR": str(out_dir),
        "CES_SPLIT_TAG": "test",
        "CES_CONTROL_METRICS": str(control_out / "metrics.json"),
        "CES_SEQ_MODEL": "v2",
        "CES_PER_SHOT_NORM": "1",            # part of seq_v2's definition (sec. 8t)
    })
    if equalized:
        env.update({"CES_SEQ_EPOCHS": "10", "CES_SEQ_FIXED_EPOCHS": "1"})
    if smoke:
        env.update({"CES_SEQ_EPOCHS": "2", "CES_SEQ_MAX_FILES": "40", "CES_SEQ_DEVICE": "cpu"})

    print(f"[b1] === {out_dir.name} (split s{seed}, init {init})", flush=True)
    record = {"seed": seed, "init": init, "equalized": equalized,
              "out_dir": str(out_dir), "status": "ok"}
    start = time.time()
    try:
        if not run_step([SEQ_DIR / "train_seq.py"], env, log, "train_seq",
                        artifacts=(out_dir / "weights" / "seq_lstm.pth", out_dir / "metrics.json")):
            record["status"] = "train_failed"
            return record
        if not run_step([SEQ_DIR / "eval_seq.py"], env, log, "eval_seq(test)",
                        artifacts=(out_dir / "comparison_errors_test.npz",
                                   out_dir / "comparison_metrics.json")):
            record["status"] = "compare_failed"
            return record
        if smoke:
            return record
        if not run_step([PAIRED,
                         "--a", out_dir / "comparison_errors_test.npz",
                         "--b", control_out / "comparison_errors_test.npz",
                         "--out", paired_json],
                        env, log, "paired-vs-w2cut", artifacts=(paired_json,)):
            record["status"] = "paired_failed"
    finally:
        record["minutes"] = round((time.time() - start) / 60.0, 1)
    return record


def _load_paired(out_dir):
    return json.loads((Path(out_dir) / "paired_vs_w2cut.json").read_text(encoding="utf-8"))["targets"]


def stage_d_summary():
    """Apply the pre-registered gate rules (PREREGISTRATION_W2.md sections 4-5)."""
    summary = {"protocol": {"env": GATE_ENV, "seeds": SEEDS, "init_seeds": INIT_SEEDS},
               "window_family": {}, "seq_grid": {}, "equalized": {}, "gate": {}}

    # Stage A ladder + claim-2 gate (model vs gp_causal) per seed.
    gp_causal_pass = {"CES_TI": 0, "CES_VT": 0}
    for seed in SEEDS:
        out_dir = names("w2cut", seed)
        cm = json.loads((out_dir / "comparison_metrics.json").read_text(encoding="utf-8"))
        bs = json.loads((out_dir / "bootstrap_summary.json").read_text(encoding="utf-8"))
        entry = {}
        for t in ("CES_TI", "CES_VT"):
            gates = bs["splits"]["test"][t]
            entry[t] = {
                "skill_vs_pchip": cm["per_target"][t]["skill_vs_pchip"],
                "n": cm["per_target"][t]["n"],
                "gates": {b: {"skill": gates[b]["skill_point"],
                              "ci95": gates[b]["skill_ci95"],
                              "pass": gates[b]["pass"]}
                          for b in ("pchip", "persistence", "gp", "gp_causal") if b in gates},
                "rmse_ladder": {m: cm["per_target"][t]["baselines"][m]["rmse"]
                                for m in cm["per_target"][t]["baselines"]},
            }
            if entry[t]["gates"].get("gp_causal", {}).get("pass"):
                gp_causal_pass[t] += 1
        summary["window_family"][seed] = entry

    # Stage B grid: per-split init-mean sign + pooled run-cluster bootstrap.
    ti_skills, vt_deficits = [], 0
    per_split = {}
    for seed in SEEDS:
        vals = []
        for init in INIT_SEEDS:
            tg = _load_paired(names("seq", seed, init=init))
            vals.append(tg["CES_TI"]["skill_point"])
            ti_skills.append(tg["CES_TI"]["skill_point"])
            if tg["CES_VT"]["b_better"]:
                vt_deficits += 1
        per_split[seed] = {"ti_paired_skills": vals, "init_mean": float(np.mean(vals))}
    summary["seq_grid"]["per_split"] = per_split
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    arr = np.asarray(ti_skills, dtype=np.float64)
    idx = rng.integers(0, len(arr), size=(BOOTSTRAP_B, len(arr)))
    means = arr[idx].mean(axis=1)
    ci = [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]
    summary["seq_grid"]["pooled"] = {
        "n_runs": len(arr), "mean": float(arr.mean()),
        "run_cluster_ci95": ci, "ci_excludes_zero": bool(ci[0] > 0.0 or ci[1] < 0.0),
    }
    summary["seq_grid"]["vt_significant_deficit_runs"] = vt_deficits

    # Stage C equalized arm signs.
    eq_signs = {}
    for seed in SEEDS:
        tg = _load_paired(names("eq", seed))
        eq_signs[seed] = tg["CES_TI"]["skill_point"]
    summary["equalized"] = {"ti_paired_skills": eq_signs,
                            "all_positive": bool(all(v > 0 for v in eq_signs.values()))}

    # Gate conditions (section 4, fixed before the run).
    cond1 = all(per_split[s]["init_mean"] > 0 for s in SEEDS)
    cond2 = ci[0] > 0.0
    cond3 = summary["equalized"]["all_positive"]
    cond4 = vt_deficits == 0
    summary["gate"] = {
        "cond1_split_init_mean_sign_4of4": cond1,
        "cond2_pooled_ci_excludes_zero": cond2,
        "cond3_equalized_sign_holds": cond3,
        "cond4_no_vt_significant_deficit": cond4,
        "backbone": "seq_v2" if (cond1 and cond2 and cond3 and cond4) else "window_w2",
        "claim2_model_vs_gp_causal_pass": gp_causal_pass,
        "claim2_verdict": ("claim 2 holds vs causal GP (4/4)" if gp_causal_pass["CES_TI"] == 4
                           else "restate claim 2 (causal GP not beaten 4/4) and add "
                                "'beat causal GP' to the B.2 objective"),
    }

    out_path = DATA / ".b1_gate_summary.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary["gate"], indent=2))
    print(f"[b1] gate summary saved to {out_path}", flush=True)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="plumbing check: seed 42, tiny caps")
    ap.add_argument("--resume", action="store_true", help="skip runs whose final artifacts exist")
    ap.add_argument("--stages", default="ABCD", help="subset of ABCD to run (default all)")
    args = ap.parse_args()

    if not any(DATA.glob("s*.csv")):
        raise SystemExit("FATAL: no shot CSVs in data/ -- real data required, aborting.")
    for seed in SEEDS:
        if not (SWEEP_OUT[seed] / "metrics.json").exists():
            raise SystemExit(f"FATAL: frozen W=2 sweep run missing: {SWEEP_OUT[seed]}")

    records = []
    batch_start = time.time()
    seeds = (42,) if args.smoke else SEEDS
    inits = (42,) if args.smoke else INIT_SEEDS
    try:
        if "A" in args.stages:
            for seed in seeds:
                records.append(stage_a_one(seed, args.smoke, args.resume))
        if "B" in args.stages:
            for seed in seeds:
                for init in inits:
                    records.append(stage_bc_one(seed, init, args.smoke, args.resume))
        if "C" in args.stages:
            for seed in seeds:
                records.append(stage_bc_one(seed, seed, args.smoke, args.resume, equalized=True))
    finally:
        print(f"[b1] wall time so far: {(time.time() - batch_start) / 3600.0:.2f} h", flush=True)

    bad = [r for r in records if r["status"] != "ok"]
    for r in bad:
        print(f"[b1] FAILED: {r['out_dir']} ({r['status']})", flush=True)
    if args.smoke:
        sys.exit(1 if bad else 0)
    if bad:
        sys.exit(1)
    if "D" in args.stages:
        stage_d_summary()
    sys.exit(0)


if __name__ == "__main__":
    main()
