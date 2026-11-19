"""B.70 phase 1 (a): re-score the 135 stored networks of the r2-r4 folds on the D1 rows (runs_b70.execute), several
CPU processes at a time.  B.39 first (the second stage needs it), then B.59, B.67, B.68.  A run whose new
comparison_errors_test.npz exists is skipped, so a re-launch resumes.

    py ces_prediction/experiments/b70_d1/batch_b70.py [--workers 6] [--folds r2f0 ...] [--families b39 ...]
Log: data/.b70_rescore/batch.json (one record per run).
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import runs_b70 as RB  # noqa: E402

LOG = RB.OUT_ROOT / "batch.json"
_lock = threading.Lock()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--folds", nargs="*", default=None)
    ap.add_argument("--families", nargs="*", default=list(RB.FAMILIES))
    args = ap.parse_args()
    if not any(RB.DATA.glob("s*.csv")):
        raise SystemExit("FATAL: no shot CSVs in data/ -- real data required, aborting.")
    runs = [x for x in RB.plan() if x[0] in args.families
            and (args.folds is None or f"r{x[2]}f{x[3]}" in args.folds)]
    order = {f: i for i, f in enumerate(RB.FAMILIES)}
    runs.sort(key=lambda x: (order[x[0]], x[2], x[3]))
    RB.OUT_ROOT.mkdir(parents=True, exist_ok=True)
    records = json.loads(LOG.read_text(encoding="utf-8")) if LOG.exists() else []
    t0 = time.time()
    print(f"[b70] {len(runs)} runs, {args.workers} workers", flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(RB.execute, *x): x for x in runs}
        for i, fu in enumerate(as_completed(futs), 1):
            x = futs[fu]
            try:
                rec = fu.result()
            except BaseException as e:  # noqa: BLE001
                rec = {"run": RB.new_dir(*x).name, "status": f"error {e!r}"}
            with _lock:
                records = [q for q in records if q["run"] != rec["run"]] + [rec]
                LOG.write_text(json.dumps(records, indent=1), encoding="utf-8")
            print(f"[b70] {i}/{len(runs)} {rec.get('run')} {rec.get('status')} {rec.get('seconds', '')} s "
                  f"({(time.time() - t0) / 3600:.2f} h)", flush=True)
    bad = [q for q in records if q["status"] not in ("ok", "exists")]
    for q in bad:
        print(f"[b70] FAILED: {q['run']} ({q['status']})", flush=True)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
