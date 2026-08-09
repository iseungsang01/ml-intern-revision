"""Produce genuine-treatment VALIDATION error artifacts for conformal calibration.

Conformal prediction needs a calibration set that is exchangeable with the test set:
same model, same data treatment, disjoint shots. The repo has genuine-treatment TEST
npz files for all four seeds but only one VAL npz, and that one is on the held-kept
population -- calibrating on it and evaluating on genuine test would mix two
different target distributions and silently break the coverage guarantee.

This re-scores the validation split of each headline run under the genuine treatment.
No retraining: the saved weights and split manifests are reused, exactly as in
experiments/largegap/rerun_compare_persistence.py, and the run directory is restored
afterwards so no existing artifact is disturbed.

Usage (repo root):  py ces_prediction/experiments/uq/rerun_compare_val.py
Writes: <run_dir>/comparison_errors_val__val_genuine.npz (+ metrics json)
"""

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
CES_DIR = REPO_ROOT / "ces_prediction"
sys.path.insert(0, str(CES_DIR / "experiments"))
from runner_common import BASELINE_OUT, FULL_ENV, SPLIT_SRC  # noqa: E402

SEEDS = (42, 1, 7, 123)
SUFFIX = "__val_genuine"


def run_one(seed):
    out_dir, split_dir = BASELINE_OUT[seed], SPLIT_SRC[seed]
    rec = {"seed": seed, "out_dir": str(out_dir)}
    if not (split_dir / "split_manifest.json").exists():
        rec["status"] = f"missing split manifest {split_dir}"
        return rec

    live = {n: out_dir / n for n in ("comparison_errors_val.npz", "comparison_metrics.json")}
    stash = out_dir / ".uq_val_stash"
    stash.mkdir(exist_ok=True)
    for p in live.values():
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
        "CES_SPLIT_TAG": "val",
        "CES_DROP_STUCK_TARGETS": "1",
    })

    start = time.time()
    log_path = out_dir / "rerun_val_genuine.log"
    with open(log_path, "w", encoding="utf-8") as log:
        proc = subprocess.run([sys.executable, str(CES_DIR / "compare_baselines.py")],
                              cwd=REPO_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    rec["minutes"] = round((time.time() - start) / 60.0, 2)
    npz = live["comparison_errors_val.npz"]
    try:
        if not (npz.exists() and npz.stat().st_mtime >= start - 1):
            rec["status"] = f"compare_failed(rc={proc.returncode}) see {log_path.name}"
            return rec
        shutil.copy2(npz, out_dir / f"comparison_errors_val{SUFFIX}.npz")
        shutil.copy2(live["comparison_metrics.json"], out_dir / f"comparison_metrics{SUFFIX}.json")
        cm = json.loads((out_dir / f"comparison_metrics{SUFFIX}.json").read_text(encoding="utf-8"))
        rec["n"] = {t: cm["per_target"][t]["n"] for t in ("CES_TI", "CES_VT")}
        rec["status"] = "ok"
        return rec
    finally:
        for name, p in live.items():
            src = stash / name
            if src.exists():
                shutil.copy2(src, p)
        shutil.rmtree(stash, ignore_errors=True)


def main():
    if not any((REPO_ROOT / "data").glob("s*.csv")):
        raise SystemExit("FATAL: no shot CSVs in data/ -- real data required, aborting.")
    results = []
    for seed in SEEDS:
        rec = run_one(seed)
        results.append(rec)
        print(f"[uq-val] seed {seed}: {rec['status']} n={rec.get('n')} "
              f"({rec.get('minutes')} min)")
    out = REPO_ROOT / "data" / ".uq_val_rerun.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"[uq-val] summary -> {out}")
    sys.exit(1 if [r for r in results if r["status"] != "ok"] else 0)


if __name__ == "__main__":
    main()
