"""Re-score the trusted headline family under the HELD-KEPT (`stuck0`) treatment to
add the per-row addressing key (`{target}_row`) each npz was missing. No retraining.

Why (§8r). Two facts about `CES_VT` have never been crossed: 54% of its *observed*
values are held (forward-filled instrument repeats), and the model is n.s. globally
but PASSes at high-activity peaks. A held row's correct answer *is* the previous
value, so no model can beat persistence there -- which means part of the global
`CES_VT` n.s. may be a property of the scored population rather than of the model.
Testing that needs the held flag aligned to the per-sample squared errors, and the
npz files carry `{target}_shot` but no row index, so a row cannot be addressed at
all. This run adds exactly that key.

The held-kept treatment is the one that matters here: under the headline `genuine`
treatment held values are already NaN'd out of scoring, so the crosstab would have
an empty column by construction.

Each re-run must reproduce the §8g reference npz
(`comparison_errors_test__test_stuck0_persist.npz`) bit-for-bit on every pre-existing
key; the script verifies this and aborts if not, so the added keys cannot smuggle in
a different population (the fourth application of the additive-key pattern after
§8g, §8i, §8p). Originals are never overwritten -- results land in
`comparison_errors_test__test_stuck0_held.npz`.

Usage (repo root):
  py ces_prediction/experiments/heldpeak/rerun_compare_stuck0.py --check   # seed 42 only
  py ces_prediction/experiments/heldpeak/rerun_compare_stuck0.py
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
SUFFIX = "__test_stuck0"
TARGETS = ("CES_TI", "CES_VT")
# `_row` is this run's reason to exist; `se_gp` and `y_true` come along because the
# harness gained them after the stuck0 npz files were written (§8p) and they are
# computed unconditionally.
NEW_KEYS = tuple(f"{t}_{s}" for t in TARGETS for s in ("row", "se_gp", "y_true"))
# What the additive-key pattern actually has to prove is that the SCORED POPULATION did
# not move: the same rows, in the same order, with the same baselines. These keys carry
# that and must be bit-identical.
POPULATION_KEYS = tuple(
    f"{t}_{s}" for t in TARGETS
    for s in ("shot", "dt_ms", "is_peak", "se_pchip", "se_linear", "se_persistence")
)
# `se_model` is NOT a property of the population -- it is a float32 CUDA forward pass, and
# on this machine it is not bit-reproducible across sessions (measured 2026-08-09: identical
# weights and identical population, per-sample |err| relative drift median 3e-4, and RMSE
# 372.3162 -> 372.3135, i.e. the 4-decimal published number is unchanged). So we bound the
# drift instead of demanding equality, and the merged artifact KEEPS THE REFERENCE VALUES
# so no published number can shift underneath us.
MODEL_SE_RMSE_TOL = 0.01   # physical CES units on the RMSE, not on individual samples
LIVE_NPZ = "comparison_errors_test.npz"
LIVE_JSON = "comparison_metrics.json"


def verify_reproduced(reference_path, produced_path):
    """The population must be bit-identical; `se_model` may drift within CUDA float noise."""
    ref = np.load(reference_path)
    new = np.load(produced_path)
    added = sorted(set(new.files) - set(ref.files))
    dropped = sorted(set(ref.files) - set(new.files))
    if dropped:
        return False, f"keys DROPPED: {dropped}", None
    if added != sorted(NEW_KEYS):
        return False, f"unexpected added keys: {added}", None
    for k in ref.files:
        a, b = ref[k], new[k]
        if a.shape != b.shape:
            return False, f"{k}: shape {a.shape} -> {b.shape}", None
    for k in POPULATION_KEYS:
        a, b = ref[k], new[k]
        if a.dtype.kind in "fc":
            if a.size and float(np.max(np.abs(a - b))) > 0.0:
                return False, f"{k}: population drifted (expected bit-identical)", None
        elif not np.array_equal(a, b):
            return False, f"{k}: population changed", None
    drift = {}
    for t in TARGETS:
        a, b = ref[f"{t}_se_model"], new[f"{t}_se_model"]
        d = abs(float(np.sqrt(a.mean())) - float(np.sqrt(b.mean())))
        drift[t] = d
        if d > MODEL_SE_RMSE_TOL:
            return False, f"{t}_se_model RMSE moved {d:.4f} > {MODEL_SE_RMSE_TOL}", None
    msg = (f"{len(POPULATION_KEYS)} population keys bit-identical, +{len(added)} new, "
           f"n={len(ref[POPULATION_KEYS[0]])}, se_model RMSE drift "
           + ", ".join(f"{t}={drift[t]:.4f}" for t in TARGETS))
    return True, msg, drift


def merge_reference_with_new_keys(reference_path, produced_path, out_path):
    """Write the merged artifact: every reference key verbatim + the new keys.

    Taking `se_model` from the reference (not the re-run) is deliberate -- the published
    numbers stay exactly what §8g recorded, and the new keys are only addressing metadata.
    """
    ref = np.load(reference_path)
    new = np.load(produced_path)
    merged = {k: ref[k] for k in ref.files}
    for k in NEW_KEYS:
        merged[k] = new[k]
    np.savez(out_path, **merged)


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
    stash = out_dir / ".heldpeak_rerun_stash"
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
        "CES_DROP_STUCK_TARGETS": "0",   # explicit: held KEPT is the point of this run
    })

    log_path = out_dir / "rerun_heldpeak_stuck0.log"
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
        ok, msg, drift = verify_reproduced(reference, live_npz)
        rec["verify"] = msg
        if drift:
            rec["se_model_rmse_drift"] = {t: round(v, 6) for t, v in drift.items()}
        if not ok:
            # keep the produced file so the mismatch can be diagnosed key by key
            shutil.copy2(live_npz, out_dir / f"comparison_errors_test{SUFFIX}_MISMATCH.npz")
            rec["status"] = "REPRO_MISMATCH"
            return rec
        merge_reference_with_new_keys(
            reference, live_npz, out_dir / f"comparison_errors_test{SUFFIX}_held.npz")
        shutil.copy2(live_json, out_dir / f"comparison_metrics{SUFFIX}_held.json")
        rec["status"] = "ok"
        return rec
    finally:
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
            print(f"[heldpeak] s{seed}: {rec['status']} -- {rec.get('verify', '')} "
                  f"({rec.get('minutes', 0)} min)")
            if rec["status"] == "REPRO_MISMATCH":
                print("[heldpeak] aborting batch: population did not reproduce")
                break
    finally:
        print(f"[heldpeak] total wall time: {(time.time() - batch_start) / 60.0:.1f} min")

    out = REPO_ROOT / "data" / (".heldpeak_rerun_check.json" if args.check
                                else ".heldpeak_rerun.json")
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"[heldpeak] summary -> {out}")
    sys.exit(1 if [r for r in results if r["status"] != "ok"] else 0)


if __name__ == "__main__":
    main()
