"""Re-score the trusted headline family to add the CAUSAL reference arm
(`{target}_se_persistence`) to each run's per-sample error npz. No retraining.

Why (THESIS_RESULTS.md §8g). THESIS_RESULTS §4.3 shows
`CES_TI` skill going negative in the large-gap bins (−8.98 at Δt > 105 ms), and the
window sweep reproduces it at every W. Read against PCHIP that looks like "the model
collapses in the large-gap regime". But PCHIP reads a *future* anchor: the wider the
gap, the easier its problem gets and the harder ours does, by construction -- and in
the real-time setting PCHIP does not exist at all. The question that decides whether
this is a model weakness or simply interpolation's territory is: at large gaps, does
the model still beat the baselines a real-time system could actually run?
`compare_baselines.py` computed persistence per sample but never stored it, so that
question was unanswerable from the saved artifacts.

Both evaluation treatments are re-scored per seed:
  genuine  -- CES_DROP_STUCK_TARGETS=1, held V_rot excluded (evaluate.py's default)
  stuck0   -- CES_DROP_STUCK_TARGETS=0, held V_rot kept (the historical convention)

Each re-run must reproduce its existing npz bit-for-bit on every pre-existing key;
the script verifies this and aborts if not, so the added key cannot smuggle in a
different population. Originals are never overwritten -- results land in
`comparison_errors_test__{treatment}_persist.npz`.

Usage (repo root):
  py ces_prediction/experiments/largegap/rerun_compare_persistence.py --check
  py ces_prediction/experiments/largegap/rerun_compare_persistence.py
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

REPO_ROOT = Path(__file__).resolve().parents[3]
CES_DIR = REPO_ROOT / "ces_prediction"
sys.path.insert(0, str(CES_DIR / "experiments"))
sys.path.insert(0, str(CES_DIR / "experiments" / "window_sweep"))
from runner_common import BASELINE_OUT, FULL_ENV, SPLIT_SRC  # noqa: E402

SEEDS = (42, 1, 7, 123)
# treatment -> (CES_DROP_STUCK_TARGETS, existing artifact suffix to verify against)
TREATMENTS = {
    "genuine": ("1", "__test_genuine"),
    "stuck0": ("0", "__test_stuck0"),
}
NEW_KEYS = ("CES_TI_se_persistence", "CES_VT_se_persistence")
LIVE_NPZ = "comparison_errors_test.npz"
LIVE_JSON = "comparison_metrics.json"


def verify_reproduced(reference_path, produced_path):
    """Every key of the reference must reappear identically; only the two new
    persistence keys may be added."""
    ref = np.load(reference_path)
    new = np.load(produced_path)
    added = sorted(set(new.files) - set(ref.files))
    dropped = sorted(set(ref.files) - set(new.files))
    if dropped:
        return False, f"keys DROPPED: {dropped}"
    if added != sorted(NEW_KEYS):
        return False, f"unexpected added keys: {added}"
    worst = 0.0
    for k in ref.files:
        a, b = ref[k], new[k]
        if a.shape != b.shape:
            return False, f"{k}: shape {a.shape} -> {b.shape}"
        if a.dtype.kind in "fc":
            if a.size:
                worst = max(worst, float(np.max(np.abs(a - b))))
        elif not np.array_equal(a, b):
            return False, f"{k}: values changed"
    if worst > 0.0:
        return False, f"float drift {worst:.3e} (expected bit-identical)"
    return True, f"{len(ref.files)} keys bit-identical, +{len(added)} new, n={len(ref[ref.files[0]])}"


def run_one(seed, treatment, drop_stuck, suffix):
    out_dir = BASELINE_OUT[seed]
    split_dir = SPLIT_SRC[seed]
    rec = {"seed": seed, "treatment": treatment, "out_dir": str(out_dir)}

    reference = out_dir / f"comparison_errors_test{suffix}.npz"
    if not reference.exists():
        rec["status"] = f"missing reference {reference.name}"
        return rec
    if not (split_dir / "split_manifest.json").exists():
        rec["status"] = f"missing split manifest {split_dir}"
        return rec

    live_npz, live_json = out_dir / LIVE_NPZ, out_dir / LIVE_JSON
    stash = out_dir / ".persist_rerun_stash"
    stash.mkdir(exist_ok=True)
    for p in (live_npz, live_json):
        if p.exists():
            shutil.copy2(p, stash / p.name)

    env = os.environ.copy()
    for var in ("CES_ABLATE", "CES_FILE_SPLIT_FROM", "CES_MIN_SUBSET_SIZE", "CES_DATA_DIR"):
        env.pop(var, None)
    env.update(FULL_ENV)
    env.update({
        "CES_SEED": str(seed),
        "CES_SPLIT_DIR": str(split_dir),
        "CES_OUTPUT_DIR": str(out_dir),
        "CES_SPLIT_TAG": "test",
        "CES_DROP_STUCK_TARGETS": drop_stuck,
    })

    log_path = out_dir / f"rerun_persist_{treatment}.log"
    start = time.time()
    with open(log_path, "w", encoding="utf-8") as log:
        proc = subprocess.run(
            [sys.executable, str(CES_DIR / "compare_baselines.py")],
            cwd=REPO_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
        )
    rec["minutes"] = round((time.time() - start) / 60.0, 2)

    try:
        if not (live_npz.exists() and live_npz.stat().st_mtime >= start - 1):
            rec["status"] = f"compare_failed(rc={proc.returncode}) see {log_path.name}"
            return rec
        ok, msg = verify_reproduced(reference, live_npz)
        rec["verify"] = msg
        if not ok:
            rec["status"] = "REPRO_MISMATCH"
            return rec
        shutil.copy2(live_npz, out_dir / f"comparison_errors_test{suffix}_persist.npz")
        shutil.copy2(live_json, out_dir / f"comparison_metrics{suffix}_persist.json")
        rec["status"] = "ok"
        return rec
    finally:
        # always put the run dir back exactly as we found it
        for p in (live_npz, live_json):
            src = stash / p.name
            if src.exists():
                shutil.copy2(src, p)
        shutil.rmtree(stash, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="seed 42 / genuine only")
    args = ap.parse_args()

    if not any((REPO_ROOT / "data").glob("s*.csv")):
        raise SystemExit("FATAL: no shot CSVs in data/ -- real data required, aborting.")

    jobs = ([(42, "genuine")] if args.check
            else [(s, t) for s in SEEDS for t in TREATMENTS])
    results, batch_start = [], time.time()
    try:
        for seed, treatment in jobs:
            drop_stuck, suffix = TREATMENTS[treatment]
            rec = run_one(seed, treatment, drop_stuck, suffix)
            results.append(rec)
            print(f"[largegap] s{seed}/{treatment}: {rec['status']} -- "
                  f"{rec.get('verify', '')} ({rec.get('minutes', 0)} min)")
            if rec["status"] == "REPRO_MISMATCH":
                print("[largegap] aborting batch: population did not reproduce")
                break
    finally:
        print(f"[largegap] total wall time: {(time.time() - batch_start) / 60.0:.1f} min")

    out = REPO_ROOT / "data" / (".largegap_rerun_check.json" if args.check
                                else ".largegap_rerun.json")
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"[largegap] summary -> {out}")
    sys.exit(1 if [r for r in results if r["status"] != "ok"] else 0)


if __name__ == "__main__":
    main()
