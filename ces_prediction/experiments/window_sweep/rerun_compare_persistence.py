"""Re-run compare_baselines.py only, over the held-free window sweep, to add the
causal reference arm (`{target}_se_persistence`) to each run's error npz.

Why this exists (THESIS_RESULTS.md §8g). The dt-stratified
read of the sweep shows CES_TI skill at dt > 45 ms is negative-to-zero at every W,
with every CI containing 0. Read against PCHIP that looks like "the model loses in
the large-gap regime" -- but PCHIP gets a *future* anchor, so the larger the gap the
easier its problem becomes and the harder ours does, by construction. And in the
real-time setting PCHIP does not exist at all. The question that actually matters is
therefore: at large gaps, does the model still beat the baselines a real-time system
could use? `compare_baselines.py` computes persistence per sample but never stored
it, so that question was unanswerable from the saved artifacts.

No retraining. Each run is re-scored from its own saved weights + split manifest,
under byte-identical env to the original, so every pre-existing npz key must come
back unchanged -- which this script verifies and refuses to proceed without.

Usage (repo root):
  py ces_prediction/experiments/window_sweep/rerun_compare_persistence.py --check   # 1 run
  py ces_prediction/experiments/window_sweep/rerun_compare_persistence.py           # all 24
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_window_sweep import (  # noqa: E402
    CES_DIR,
    FULL_ENV,
    H0_WINDOW,
    REPO_ROOT,
    SEEDS,
    WINDOWS,
    run_names,
)

NEW_KEYS = ("CES_TI_se_persistence", "CES_VT_se_persistence")


def all_runs():
    """(window, seed, hist0) for every point of the held-free sweep."""
    runs = [(w, s, False) for w in WINDOWS for s in SEEDS]
    runs += [(H0_WINDOW, s, True) for s in SEEDS]
    return runs


def build_env(window, seed, hist0, split_dir, out_dir):
    env = os.environ.copy()
    for var in ("CES_ABLATE", "CES_FILE_SPLIT_FROM", "CES_MIN_SUBSET_SIZE", "CES_DATA_DIR"):
        env.pop(var, None)
    env.update(FULL_ENV)
    env.update({
        "CES_WINDOW_SIZE": str(window),
        "CES_SEED": str(seed),
        "CES_INIT_SEED": str(seed),
        "CES_SPLIT_DIR": str(split_dir),
        "CES_OUTPUT_DIR": str(out_dir),
        "CES_SPLIT_TAG": "test",
    })
    if hist0:
        env["CES_ABLATE"] = "no_history"
    return env


def verify_reproduced(before_path, after_path):
    """Every key that existed before must come back identical; only the new
    persistence keys may appear. Returns (ok, message)."""
    before = np.load(before_path)
    after = np.load(after_path)
    added = sorted(set(after.files) - set(before.files))
    dropped = sorted(set(before.files) - set(after.files))
    if dropped:
        return False, f"keys DROPPED: {dropped}"
    if added != sorted(NEW_KEYS):
        return False, f"unexpected added keys: {added}"
    worst = 0.0
    for k in before.files:
        a, b = before[k], after[k]
        if a.shape != b.shape:
            return False, f"{k}: shape {a.shape} -> {b.shape}"
        if a.dtype.kind in "fc":
            d = float(np.max(np.abs(a - b))) if a.size else 0.0
            worst = max(worst, d)
        elif not np.array_equal(a, b):
            return False, f"{k}: exact-dtype values changed"
    if worst > 0.0:
        return False, f"float drift {worst:.3e} (expected bit-identical)"
    return True, f"{len(before.files)} keys bit-identical, +{len(added)} new"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="one run only (W=4, seed 42)")
    args = ap.parse_args()

    if not any((REPO_ROOT / "data").glob("s*.csv")):
        raise SystemExit("FATAL: no shot CSVs in data/ -- real data required, aborting.")

    runs = [(4, 42, False)] if args.check else all_runs()
    results, batch_start = [], time.time()
    try:
        for window, seed, hist0 in runs:
            tag, out_dir, split_dir = run_names(window, seed, hist0, smoke=False)
            npz = out_dir / "comparison_errors_test.npz"
            rec = {"tag": tag, "window": window, "seed": seed, "hist0": hist0,
                   "out_dir": str(out_dir)}
            if not npz.exists():
                rec["status"] = "missing_npz"
                results.append(rec)
                print(f"[rerun] {tag}: SKIP (no {npz.name})")
                continue

            backup = out_dir / "comparison_errors_test.npz.pre_persist"
            if not backup.exists():
                shutil.copy2(npz, backup)

            env = build_env(window, seed, hist0, split_dir, out_dir)
            log_path = out_dir / "rerun_compare.log"
            start = time.time()
            with open(log_path, "w", encoding="utf-8") as log:
                proc = subprocess.run(
                    [sys.executable, str(CES_DIR / "compare_baselines.py")],
                    cwd=REPO_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                )
            mins = (time.time() - start) / 60.0
            fresh = npz.exists() and npz.stat().st_mtime >= start - 1
            if not fresh:
                rec["status"] = f"compare_failed(rc={proc.returncode})"
                results.append(rec)
                print(f"[rerun] {tag}: FAILED rc={proc.returncode} ({mins:.1f} min)")
                continue

            ok, msg = verify_reproduced(backup, npz)
            rec["status"] = "ok" if ok else "REPRO_MISMATCH"
            rec["verify"] = msg
            rec["minutes"] = round(mins, 2)
            results.append(rec)
            print(f"[rerun] {tag}: {'ok' if ok else 'MISMATCH'} -- {msg} ({mins:.1f} min)")
            if not ok:
                shutil.copy2(backup, npz)
                print(f"[rerun] {tag}: restored original npz; aborting batch")
                break
    finally:
        print(f"[rerun] total wall time: {(time.time() - batch_start) / 60.0:.1f} min")

    out = REPO_ROOT / "data" / (".wsweep_hf_persist_rerun_check.json" if args.check
                                else ".wsweep_hf_persist_rerun.json")
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"[rerun] summary -> {out}")
    bad = [r for r in results if r["status"] != "ok"]
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
