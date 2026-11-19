"""B.70 phase 2: B.55 (sample timing; fast channels one row old) on the D1 rows (PREREGISTRATION_B70.md, Addendum 2).

run_b55's stages nets, checks, readout, netsD, readoutD run unchanged with CACHE = data/.b70p2/b55/, REPEATS = r2-r4,
outputs and the D1 b39_primary.json (its check (d)) in data/.b70p2/paper/, and run_b39.pooled_run_dir on the D1 runs.
The raw-stream caches raw_s{shot}.{npz,json} do not depend on the scored rows and are copied from data/.b55/.
C4: after `nets`, the key after_absent of every run file is replaced by the B.54 definition (rows 1-3 after a 15-45 ms
step with five regular rows and >= 3 finite T_i before; row number k from run_b54.blip_rows_13's walk); B.55's own
marker is kept as after_absent_b55rule.

    py ces_prediction/experiments/b70_d1/p2_b55.py [nets|absent|checks|readout|netsD|readoutD|all]
"""
from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
import json  # noqa: E402
import shutil  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "b55_sample_timing"))
sys.path.insert(0, str(HERE.parent / "b41_events"))
import p2_common as PC  # noqa: E402
import run_b55 as B55  # noqa: E402

CACHE = PC.P2 / "b55"
PRE_ROWS = 5                      # run_b41.PRE_ROWS


def blip_k(t, ti):
    """run_b54.blip_rows_13 (copied), returning the row number k = 1..3 (0 elsewhere): the smallest k of the walk."""
    dt = np.diff(t)
    n = len(t)
    mark = np.zeros(n, np.int8)
    for j in np.flatnonzero((dt > 0.015) & (dt < 0.045)):
        if j < PRE_ROWS or not np.all(np.abs(dt[j - PRE_ROWS:j] - 0.010) < 1e-4):
            continue
        pre = ti[j - PRE_ROWS:j + 1]
        pre = pre[np.isfinite(pre)]
        if len(pre) < 3:
            continue
        for k in range(1, 4):
            r = j + k
            if r >= n or (k > 1 and abs(dt[r - 1] - 0.010) > 1e-4):
                break
            if mark[r] == 0 or mark[r] > k:
                mark[r] = k
    return mark


def setup():
    PC.use_d1()
    PC.write_b39_d1()
    CACHE.mkdir(parents=True, exist_ok=True)
    for shot in B55.RAW_SHOTS:
        for ext in ("npz", "json"):
            if not (CACHE / f"raw_s{shot}.{ext}").exists():
                shutil.copy2(PC.DATA / ".b55" / f"raw_s{shot}.{ext}", CACHE / f"raw_s{shot}.{ext}")
    B55.CACHE = CACHE
    B55.REPEATS = PC.REPEATS
    B55.OUT_JSON, B55.OUT_MD = PC.paper() / "b55_sample_timing.json", PC.paper() / "b55_sample_timing.md"
    B55.B39_JSON = PC.paper() / "b39_primary.json"


def stage_absent():
    """Replace after_absent by the D1 definition; check its rows against the blip13 flags of run_b54 (files.npz)."""
    F = np.load(PC.P2 / "b54" / "files.npz")
    fnames = list(F["names"])
    sys.path.insert(0, str(PC.RB.SEQ_DIR))
    sys.path.insert(1, str(PC.REPO / "ces_prediction"))
    from seq_data import load_grid_files
    grid, _ = load_grid_files(PC.DATA, True, 3000.0)
    cache = {}
    res = {}
    for r in PC.REPEATS:
        for k in range(5):
            p = CACHE / f"run_r{r}_f{k}.npz"
            z = dict(np.load(p, allow_pickle=False))
            if "after_absent_b55rule" in z:
                continue
            names, grow = z["names"], z["grow"]
            new = np.zeros(len(grow), np.int8)
            for nm in np.unique(names):
                if nm not in cache:
                    arr = grid[str(nm)]
                    kk = blip_k(arr[:, 0].astype(np.float64), arr[:, 1].astype(np.float64))
                    if not np.array_equal(kk > 0, F[f"blip13_{fnames.index(str(nm))}"].astype(bool)):
                        raise SystemExit(f"FATAL: {nm}: the row walk differs from run_b54's blip13 flags")
                    cache[nm] = kk
                sel = np.flatnonzero(names == nm)
                new[sel] = cache[nm][grow[sel]]
            z["after_absent_b55rule"] = z["after_absent"]
            z["after_absent"] = new
            np.savez(p, **z)
            res[f"r{r}f{k}"] = {"rows_1_3_d1": int((new > 0).sum()), "rows_1_3_b55rule": int((z["after_absent_b55rule"] > 0).sum())}
            print(f"r{r} f{k}: rows 1-3 after an absent frame, D1 {res[f'r{r}f{k}']['rows_1_3_d1']}, "
                  f"B.55 rule {res[f'r{r}f{k}']['rows_1_3_b55rule']}", flush=True)
    if res:
        (CACHE / "absent_rule.json").write_text(json.dumps(res, indent=1), encoding="utf-8")


def main():
    setup()
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    steps = (("nets", B55.stage_nets), ("absent", stage_absent), ("checks", B55.stage_checks),
             ("readout", B55.stage_readout), ("netsD", B55.stage_nets_d), ("readoutD", B55.stage_readout_d))
    rt_path = CACHE / "runtime.json"
    for name, fn in steps:
        if stage in (name, "all"):
            t = time.time()
            fn()
            rt = json.loads(rt_path.read_text()) if rt_path.exists() else {}
            rt[name] = round(time.time() - t, 1)
            rt_path.write_text(json.dumps(rt))


if __name__ == "__main__":
    main()
