"""Re-score the trusted headline family to add the GP baseline arm
(`{target}_se_gp`) and the physical target values (`{target}_y_true`) to each
run's per-sample error npz. No retraining.

Why (2026-08-05 audit follow-up). `compare_baselines.py` has carried a GP arm
since the harness was written (`if B._HAVE_SKLEARN: methods.append("gp")`), but
sklearn was never installed on this machine, so the community-standard fusion
profile-fitting method (Chilenski 2015; Michoski 2024) was silently skipped and
the defense of its absence stayed rhetorical (Q9 of the defense doc). This run
closes that: the arm is now an exact, deterministic Matern-3/2+white GP
(`baselines_interpolation.predict_gp`, local nearest-16+16 fit, per-sample
grid-ML hyperparameters) scored on the byte-identical headline population.
`{target}_y_true` is stored alongside so downstream sensitivity analyses (e.g.
CES fit-failure row exclusion) need no further re-scoring pass.

The genuine treatment (CES_DROP_STUCK_TARGETS=1, the paper's headline) is
re-scored for all four seeds. Each re-run must reproduce the §8g reference npz
(`comparison_errors_test__test_genuine_persist.npz`) bit-for-bit on every
pre-existing key; the script verifies this and aborts if not, so the added keys
cannot smuggle in a different population. Originals are never overwritten --
results land in `comparison_errors_test__test_genuine_gp.npz`.

Usage (repo root):
  py ces_prediction/experiments/gp/rerun_compare_gp.py --check   # seed 42 only
  py ces_prediction/experiments/gp/rerun_compare_gp.py
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
SUFFIX = "__test_genuine"  # headline treatment only
NEW_KEYS = (
    "CES_TI_se_gp", "CES_VT_se_gp",
    "CES_TI_y_true", "CES_VT_y_true",
)
LIVE_NPZ = "comparison_errors_test.npz"
LIVE_JSON = "comparison_metrics.json"


def verify_reproduced(reference_path, produced_path):
    """Every key of the §8g reference must reappear identically; only the four
    new gp/y_true keys may be added."""
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


def run_one(seed):
    out_dir = BASELINE_OUT[seed]
    split_dir = SPLIT_SRC[seed]
    rec = {"seed": seed, "out_dir": str(out_dir)}

    reference = out_dir / f"comparison_errors_test{SUFFIX}_persist.npz"
    if not reference.exists():
        rec["status"] = f"missing reference {reference.name}"
        return rec
    if not (split_dir / "split_manifest.json").exists():
        rec["status"] = f"missing split manifest {split_dir}"
        return rec

    live_npz, live_json = out_dir / LIVE_NPZ, out_dir / LIVE_JSON
    stash = out_dir / ".gp_rerun_stash"
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
        "CES_DROP_STUCK_TARGETS": "1",
    })

    log_path = out_dir / "rerun_gp_genuine.log"
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
        shutil.copy2(live_npz, out_dir / f"comparison_errors_test{SUFFIX}_gp.npz")
        shutil.copy2(live_json, out_dir / f"comparison_metrics{SUFFIX}_gp.json")
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
    ap.add_argument("--check", action="store_true", help="seed 42 only")
    args = ap.parse_args()

    if not any((REPO_ROOT / "data").glob("s*.csv")):
        raise SystemExit("FATAL: no shot CSVs in data/ -- real data required, aborting.")

    seeds = (42,) if args.check else SEEDS
    results, batch_start = [], time.time()
    try:
        for seed in seeds:
            rec = run_one(seed)
            results.append(rec)
            print(f"[gp] s{seed}: {rec['status']} -- {rec.get('verify', '')} "
                  f"({rec.get('minutes', 0)} min)")
            if rec["status"] == "REPRO_MISMATCH":
                print("[gp] aborting batch: population did not reproduce")
                break
    finally:
        print(f"[gp] total wall time: {(time.time() - batch_start) / 60.0:.1f} min")

    out = REPO_ROOT / "data" / (".gp_rerun_check.json" if args.check else ".gp_rerun.json")
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"[gp] summary -> {out}")
    sys.exit(1 if [r for r in results if r["status"] != "ok"] else 0)


if __name__ == "__main__":
    main()
