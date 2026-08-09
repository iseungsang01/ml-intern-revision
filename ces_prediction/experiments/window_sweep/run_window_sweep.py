"""Window sweep runner -- THESIS_RESULTS.md §8f.

One curve answers the "why window=4?" feedback: train the published iter009 model
(GRU + observation-masked multi-head attention) at W in {2, 3, 4, 6, 8} x seeds
{42, 1, 7, 123}, plus a history-0 point (W=1 is impossible in the dataset --
window_size >= 2 -- so history-0 = CES_ABLATE=no_history at W=4, footnoted in the
figure). Metric: each run's own held-out test skill_vs_pchip per target
(compare_baselines.py) + bootstrap pchip PASS.

Deviations from the trusted-baseline env, both deliberate and applied to EVERY
point of the curve identically:
  - CES_MAX_SAMPLES_PER_FILE=500: temporal-subset augmentation explodes
    combinatorially with W, so without a per-shot cap a few long-block shots
    would dominate the 200k train subset more the larger W is (user-requested
    control). 500 ~= 200k / ~400 train files, so total train volume stays at the
    trusted scale.
  - CES_MAX_TEST_SAMPLES=48000 (>= n_test_files * 500): with the per-file cap in
    place the global test cap can never bind, so no shot is ever randomly dropped
    from the manifest test list -> the test file list is identical across W for a
    given seed (asserted in the summary).

Each (W, seed) gets a fresh split dir (fixed splits raise on dataset mismatch
when W changes). The file-level split itself is W-invariant (probed pre-flight).

Usage (repo root):
  py ces_prediction/experiments/window_sweep/run_window_sweep.py --smoke   # extremes, 1 epoch
  py ces_prediction/experiments/window_sweep/run_window_sweep.py          # 24-run batch
  py ces_prediction/experiments/window_sweep/run_window_sweep.py --resume # skip finished runs

Summary: data/.wsweep_summary.json. No git actions, no Slack dependency.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
CES_DIR = REPO_ROOT / "ces_prediction"
WINDOWS = (2, 3, 4, 6, 8)   # history lengths 1, 2, 3, 5, 7
SEEDS = (42, 1, 7, 123)
H0_WINDOW = 4               # history-0 point: no_history ablation at W=4

FULL_ENV = {
    "CES_TEST_FRACTION": "0.15",
    "CES_EPOCHS": "10",
    "CES_BATCH_SIZE": "512",
    "CES_LR": "1e-3",
    "CES_VAL_FRACTION": "0.2",
    "CES_MAX_TRAIN_SAMPLES": "200000",
    "CES_MAX_VAL_SAMPLES": "40000",
    "CES_MAX_TEST_SAMPLES": "48000",
    "CES_TEMPORAL_SUBSETS": "1",
    "CES_MAX_SAMPLES_PER_FILE": "500",
    # Held/forward-filled CES values are dropped at TRAIN time too, not just at
    # eval. §8c established this as the project's training convention: held V_rot
    # (54% of observed values) is an instrument artifact that teaches the model to
    # copy history, and removing it improved CES_VT on 4/4 seeds. It matters more
    # here than anywhere else -- a longer window under held contamination just
    # feeds the model more copies of the same value, which is exactly the
    # persistence signal this sweep is measuring.
    "CES_DROP_STUCK_TARGETS": "1",
}
SMOKE_ENV = {
    **FULL_ENV,
    "CES_EPOCHS": "1",
    "CES_MAX_TRAIN_SAMPLES": "4000",
    "CES_MAX_VAL_SAMPLES": "1200",
    "CES_MAX_TEST_SAMPLES": "1200",
    "CES_MAX_SAMPLES_PER_FILE": "50",
}
# env vars that would silently change the protocol if inherited from the shell
# (CES_DROP_STUCK_TARGETS is NOT here -- FULL_ENV/SMOKE_ENV pin it to 1 on purpose)
AMBIENT_VARS = ("CES_ABLATE", "CES_FILE_SPLIT_FROM",
                "CES_MIN_SUBSET_SIZE", "CES_DATA_DIR")


def run_step(cmd, env, log_path, label, artifacts=()):
    """Run one pipeline step. rc != 0 is still a success if every expected
    artifact exists and was written after the step started (Windows CUDA
    teardown can crash the process after the real work completed)."""
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
        print(f"[runner]   {label}: rc={proc.returncode} but artifacts fresh -> treating as OK ({mins:.1f} min)")
        ok = True
    else:
        print(f"[runner]   {label}: rc={proc.returncode} ({mins:.1f} min)")
    return ok


# Held-free is a different data treatment, so its artifacts live under their own
# prefix; the first (held-kept) sweep stays on disk under `.wsweep_*` for comparison.
PREFIX = "wsweep_hf"


def run_names(window, seed, hist0, smoke):
    tag = f"h0_s{seed}" if hist0 else f"w{window}_s{seed}"
    if smoke:
        tag += "_smoke"
    out_dir = REPO_ROOT / "data" / f".{PREFIX}_{tag}"
    # hist0 shares the W=4 split dir (identical split env -> identical split)
    split_base = f"w{H0_WINDOW}" if hist0 else f"w{window}"
    split_tag = f"{split_base}_s{seed}" + ("_smoke" if smoke else "")
    split_dir = REPO_ROOT / "data" / f".{PREFIX}_split_{split_tag}"
    return tag, out_dir, split_dir


def one_run(window, seed, hist0=False, smoke=False, resume=False):
    tag, out_dir, split_dir = run_names(window, seed, hist0, smoke)
    done_artifacts = (out_dir / "comparison_metrics.json", out_dir / "bootstrap_summary.json")
    record = {
        "window": H0_WINDOW if hist0 else window,
        "history_len": 0 if hist0 else window - 1,
        "hist0": hist0, "seed": seed, "smoke": smoke,
        "out_dir": str(out_dir), "split_dir": str(split_dir), "status": "ok",
    }
    if resume and not smoke and all(p.exists() for p in done_artifacts):
        print(f"[runner] === {tag}: complete, skipping (--resume)")
        record["resumed"] = True
        return record

    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "run.log"

    env = os.environ.copy()
    for var in AMBIENT_VARS:
        env.pop(var, None)
    env.update(SMOKE_ENV if smoke else FULL_ENV)
    env.update({
        "CES_WINDOW_SIZE": str(H0_WINDOW if hist0 else window),
        "CES_SEED": str(seed),
        "CES_INIT_SEED": str(seed),
        "CES_SPLIT_DIR": str(split_dir),
        "CES_OUTPUT_DIR": str(out_dir),
        "CES_SPLIT_TAG": "test",
    })
    if hist0:
        env["CES_ABLATE"] = "no_history"

    print(f"[runner] === {tag} -> {out_dir.name}")
    start = time.time()
    try:
        if not run_step([CES_DIR / "train.py"], env, log_path, "train",
                        artifacts=(out_dir / "metrics.json", out_dir / "weights" / "multimodal_ces.pth")):
            record["status"] = "train_failed"
            return record
        if not run_step([CES_DIR / "compare_baselines.py"], env, log_path, "compare(test)",
                        artifacts=(out_dir / "comparison_metrics.json",
                                   out_dir / "comparison_errors_test.npz")):
            record["status"] = "compare_failed"
            return record
        if not run_step([CES_DIR / "bootstrap_compare.py"], env, log_path, "bootstrap",
                        artifacts=(out_dir / "bootstrap_summary.json",)):
            record["status"] = "bootstrap_failed"
            return record
    finally:
        record["minutes"] = round((time.time() - start) / 60.0, 1)
    return record


def collect_metrics(record):
    """Attach the run's headline numbers from its JSON artifacts."""
    out_dir = Path(record["out_dir"])
    split_dir = Path(record["split_dir"])
    try:
        cm = json.loads((out_dir / "comparison_metrics.json").read_text(encoding="utf-8"))
        record["eval_samples"] = cm["eval_samples"]
        for t in ("CES_TI", "CES_VT"):
            record[f"skill_vs_pchip_{t[-2:]}"] = cm["per_target"][t]["skill_vs_pchip"]
            record[f"n_{t[-2:]}"] = cm["per_target"][t]["n"]
        bs = json.loads((out_dir / "bootstrap_summary.json").read_text(encoding="utf-8"))
        for t in ("CES_TI", "CES_VT"):
            record[f"pchip_pass_{t[-2:]}"] = bs["splits"]["test"][t]["pchip"]["pass"]
        manifest = json.loads((split_dir / "split_manifest.json").read_text(encoding="utf-8"))
        record["test_files"] = sorted(manifest["test_files"])
        record["split_counts"] = {k: manifest[f"{k}_sample_count"] for k in ("train", "val", "test")}
    except (FileNotFoundError, KeyError) as exc:
        record["metrics_missing"] = repr(exc)
    return record


def summarize(records, out_path):
    recs = [collect_metrics(r) for r in records if not r["smoke"]]

    # test-file invariance across W for each seed (the sweep's fairness premise)
    invariance = {}
    for seed in sorted({r["seed"] for r in recs}):
        lists = {}
        for r in recs:
            if r["seed"] == seed and "test_files" in r:
                key = "h0" if r["hist0"] else f"w{r['window']}"
                lists[key] = r["test_files"]
        uniq = {tuple(v) for v in lists.values()}
        invariance[seed] = {
            "identical": len(uniq) <= 1,
            "n_test_files": {k: len(v) for k, v in lists.items()},
        }
        if len(uniq) > 1:
            print(f"[summary] WARNING: test file lists differ across W for seed {seed}!")

    # aggregate per curve point (history length)
    points = {}
    for r in recs:
        points.setdefault(r["history_len"], []).append(r)
    per_point = []
    for hist_len in sorted(points):
        rs = sorted(points[hist_len], key=lambda r: r["seed"])
        entry = {
            "history_len": hist_len,
            "window": rs[0]["window"],
            "hist0": rs[0]["hist0"],
            "seeds": [r["seed"] for r in rs],
            "statuses": [r["status"] for r in rs],
        }
        for t in ("TI", "VT"):
            skills = [r[f"skill_vs_pchip_{t}"] for r in rs if f"skill_vs_pchip_{t}" in r]
            if skills:
                entry[f"skill_{t}_per_seed"] = skills
                entry[f"skill_{t}_mean"] = sum(skills) / len(skills)
                entry[f"skill_{t}_min"] = min(skills)
                entry[f"skill_{t}_max"] = max(skills)
            entry[f"pchip_pass_{t}"] = sum(int(r.get(f"pchip_pass_{t}", False)) for r in rs)
        per_point.append(entry)

    summary = {
        "protocol": {
            "model": "iter009 (GRU + observation-masked multi-head attention)",
            "env": FULL_ENV,
            "metric": "per-run held-out test skill_vs_pchip (compare_baselines.py)",
            "hist0_note": "history-0 point = CES_ABLATE=no_history at W=4 (dataset requires window_size >= 2)",
            "data_treatment": "held-free: CES_DROP_STUCK_TARGETS=1 at train AND eval",
        },
        "test_file_invariance": invariance,
        "per_point": per_point,
        "records": [{k: v for k, v in r.items() if k != "test_files"} for r in recs],
    }
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\n[summary] saved {out_path}")
    hdr = f"{'hist_len':>8} {'W':>3} {'TI mean':>9} {'TI min..max':>17} {'passTI':>6} {'VT mean':>9} {'passVT':>6}"
    print(hdr)
    print("-" * len(hdr))
    for e in per_point:
        ti = e.get("skill_TI_mean")
        vt = e.get("skill_VT_mean")
        ti_rng = (f"{e['skill_TI_min']:+.3f}..{e['skill_TI_max']:+.3f}"
                  if "skill_TI_min" in e else "n/a")
        n = len(e["seeds"])  # actual completed seeds, not the nominal 4
        print(f"{e['history_len']:>8} {e['window']:>3} "
              f"{ti if ti is None else format(ti, '+9.4f')} {ti_rng:>17} {e['pchip_pass_TI']:>5}/{n} "
              f"{vt if vt is None else format(vt, '+9.4f')} {e['pchip_pass_VT']:>5}/{n}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="1-epoch tiny-cap sanity: W=2, W=8, hist0 at seed 42")
    ap.add_argument("--resume", action="store_true",
                    help="skip runs whose comparison+bootstrap artifacts already exist")
    ap.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS), choices=SEEDS)
    args = ap.parse_args()

    if not any((REPO_ROOT / "data").glob("s*.csv")):
        raise SystemExit("FATAL: no shot CSVs in data/ -- real data required, aborting.")
    records = []
    batch_start = time.time()
    try:
        if args.smoke:
            for window, hist0 in ((2, False), (8, False), (H0_WINDOW, True)):
                records.append(one_run(window, 42, hist0=hist0, smoke=True))
        else:
            # W ascending so each window's dataset disk cache is built once and
            # reused by the remaining seeds; hist0 last (reuses the W=4 cache).
            for window in WINDOWS:
                for seed in args.seeds:
                    records.append(one_run(window, seed, smoke=False, resume=args.resume))
            for seed in args.seeds:
                records.append(one_run(H0_WINDOW, seed, hist0=True, smoke=False, resume=args.resume))
    finally:
        print(f"[runner] total wall time: {(time.time() - batch_start) / 3600.0:.2f} h")

    if args.smoke:
        bad = [r for r in records if r["status"] != "ok"]
        for r in records:
            print(f"[smoke] {Path(r['out_dir']).name}: {r['status']} ({r.get('minutes')} min)")
        sys.exit(1 if bad else 0)

    summarize(records, REPO_ROOT / "data" / f".{PREFIX}_summary.json")
    bad = [r for r in records if r["status"] != "ok"]
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
