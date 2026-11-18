"""B.68: build the auxiliary-target cache (target_b68.py) once for every discharge.  CPU only, no torch.

Loads the full grid of every discharge exactly as train_seq does under the env of run_b39.pooled_env (held-free,
CES_TI_SPIKE_CUT_EV=3000; both pinned here, not read from the shell), evaluates the offline-GP estimator of B.39 at
every row (target_b68.aux_for_file), and writes data/.b68_aux/aux_ti_offline_gp.npz + meta.json (build time,
coverage, the estimator's constants, hashes of the sources it depends on).  The target does not depend on the fold.

    py ces_prediction/experiments/b68_auxtarget/build_aux_b68.py [--workers 4] [--force]
"""
from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "1"          # one thread per worker: the GPU batch keeps the other cores

import argparse  # noqa: E402
import multiprocessing as mp  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import target_b68 as T  # noqa: E402


def work(item):
    name, arr3 = item
    t0 = time.time()
    res = T.aux_for_file(arr3)
    return name, res, time.time() - t0


def coverage(results):
    """Shares over all grid rows and over the rows of grid blocks (the rows train_seq trains on)."""
    tot = {"all_rows": {}, "block_rows": {}}
    for scope in tot:
        acc = {k: np.zeros(5, np.int64) for k in ("all", *T.KIND_LABEL)}
        for name, (tm, ti, r) in results.items():
            if scope == "all_rows":
                sel = np.ones(len(tm), bool)
            else:
                sel = np.zeros(len(tm), bool)
                for a, b in T.grid_blocks(tm):
                    sel[a:b] = True
            have = np.isfinite(r["aux"])
            fb = have & r["past"] & ~r["future"]
            for key in acc:
                m = sel if key == "all" else sel & (r["kind"] == key)
                acc[key] += [int(m.sum()), int((m & have).sum()), int((m & fb).sum()), int((m & ~have).sum()),
                             int((m & have & r["future"]).sum())]
        tot[scope] = {("all" if k == "all" else T.KIND_LABEL[k]): {
            "rows": int(v[0]), "with_target": int(v[1]), "share_with_target": float(v[1] / max(v[0], 1)),
            "gp_with_both_sides": int(v[4]), "fallback_last_earlier_value": int(v[2]), "no_value": int(v[3])}
            for k, v in acc.items()}
    return tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--force", action="store_true", help="rebuild even if the cache exists")
    args = ap.parse_args()
    if T.CACHE_NPZ.exists() and T.CACHE_META.exists() and not args.force:
        T.load_cache()
        print(f"[b68] cache exists ({T.CACHE_NPZ}); --force rebuilds it")
        return
    from seq_data import load_grid_files
    t0 = time.time()
    grid, dims = load_grid_files(T.DATA, T.DROP_STUCK, T.SPIKE_CUT_EV)
    t_load = time.time() - t0
    items = sorted(((n, np.ascontiguousarray(a[:, :3])) for n, a in grid.items()), key=lambda x: -len(x[1]))
    rows = sum(len(a) for _, a in items)
    print(f"[b68] {len(items)} discharges, {rows:,} grid rows (load {t_load:.0f} s); {args.workers} workers", flush=True)
    results, per_file_s = {}, {}
    t1 = time.time()
    with mp.get_context("spawn").Pool(args.workers) as pool:
        for j, (name, res, sec) in enumerate(pool.imap_unordered(work, items, chunksize=1), 1):
            arr = grid[name]
            results[name] = (arr[:, T.TIME_COL].copy(), arr[:, T.TI_COL].copy(), res)
            per_file_s[name] = sec
            if j % 64 == 0 or j == len(items):
                print(f"[b68] {j}/{len(items)} discharges ({time.time() - t1:.0f} s)", flush=True)
    t_gp = time.time() - t1
    cov = coverage(results)
    meta = {"registration": "ces_prediction/experiments/PREREGISTRATION_B68.md",
            "built": time.strftime("%Y-%m-%d %H:%M:%S"), "spike_cut_ev": T.SPIKE_CUT_EV, "drop_stuck": T.DROP_STUCK,
            "estimator": T.estimator_constants(), "source_sha256": T.source_hashes(),
            "rule": "baselines_interpolation.PREDICTORS['gp'] on the harness file array (rows with >= 1 measured target); "
                    "a row without a measured target is inserted at its time; the row's own value is never a neighbour",
            "discharges": len(results), "grid_rows": rows, "workers": args.workers,
            "seconds_load": round(t_load, 1), "seconds_estimator": round(t_gp, 1),
            "cpu_seconds_estimator": round(sum(per_file_s.values()), 1), "coverage": cov}
    T.save_cache(results, meta)
    T.load_cache()
    c = cov["block_rows"]
    print(f"[b68] estimator {t_gp:.0f} s wall ({sum(per_file_s.values()):.0f} s CPU) on {args.workers} workers; "
          f"rows of grid blocks {c['all']['rows']:,}, with a target {c['all']['share_with_target']:.4f}; "
          f"T_i-measured rows with a target {c[T.KIND_LABEL[T.KIND_TI]]['share_with_target']:.4f}", flush=True)
    print(f"[b68] written {T.CACHE_NPZ} and {T.CACHE_META}", flush=True)


if __name__ == "__main__":
    main()
