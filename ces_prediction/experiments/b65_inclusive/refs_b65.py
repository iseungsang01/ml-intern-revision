"""B.65 CPU references (PREREGISTRATION_B65.md): hold, causal GP, offline GP and PCHIP on the test rows of
re-partitions r2..r4 (15 folds), computed by the code of B.39 (`eval_seq.compare`, unchanged) with no network: the
model lookup is replaced by NaN, which changes no reference error and no scored row (the valid rows depend only on
the keep mask and the reference methods).

Two populations per fold:
  cut   the B.39 environment (`run_b39.pooled_env` of v2)                         -> check against the stored B.39 errors
  incl  the same with CES_TI_SPIKE_CUT_EV=0 and the inclusive harness statistics  -> the B.65 references
Both: CES_SEQ_DEVICE=cpu and CUDA_VISIBLE_DEVICES='' (references never touch the network; the GPU is not used).
Each output dir data/.b65_refs_{pop}_r{r}_f{k} holds a copy of the B.39 run's metrics.json (read by eval_seq's
guards only) and the eval_seq outputs.  A fold whose npz exists is skipped.

    py ces_prediction/experiments/b65_inclusive/refs_b65.py            # all 30 jobs, then the checks
    py ces_prediction/experiments/b65_inclusive/refs_b65.py check      # checks only
Output of the checks: data/.b65_refs_checks.json
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
T = "CES_TI"
REFS = ("persistence", "gp_causal", "gp", "pchip")


def worker():
    """Run inside the subprocess: eval_seq.compare with the model lookup replaced by NaN."""
    seq = HERE.parent / "seq"
    sys.path.insert(0, str(seq))
    import eval_seq  # noqa: PLC0415

    class _Row:
        def __getitem__(self, key):
            return np.full(2, np.nan, np.float32)

    class _File:
        def __getitem__(self, name):
            return _Row()

    def no_network(data_dir, eval_names, my_stats, device, contexts=(), primary_ctx=0):
        return {c: _File() for c in ["full", *contexts]}

    eval_seq.build_pred_lookup = no_network
    eval_seq.compare()
    os._exit(0)


if len(sys.argv) > 1 and sys.argv[1] == "worker":
    worker()

sys.path.insert(0, str(HERE.parent / "b39_primary"))
sys.path.insert(0, str(HERE))
import run_b39 as B39  # noqa: E402
from run_b39 import B27, DATA  # noqa: E402
import run_b65 as R65  # noqa: E402

CHECKS = DATA / ".b65_refs_checks.json"


def out_dir(pop, r, k):
    return DATA / f".b65_refs_{pop}_r{r}_f{k}"


def job_env(pop, r, k):
    out = out_dir(pop, r, k)
    env = B39.pooled_env("v2", r, k, out)
    if pop == "incl":
        env["CES_TI_SPIKE_CUT_EV"] = R65.CUT_EV
        env["CES_CONTROL_METRICS"] = str(R65.ctrl_metrics(r, k))
    env["CES_SEQ_DEVICE"] = "cpu"
    env["CUDA_VISIBLE_DEVICES"] = ""
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        env[v] = "4"
    return env


def run_all():
    t0 = time.time()
    jobs = [(pop, r, k) for pop in ("cut", "incl") for r in R65.REPEATS for k in range(B27.K)]
    for i, (pop, r, k) in enumerate(jobs, 1):
        out = out_dir(pop, r, k)
        if (out / "comparison_errors_test.npz").exists():
            continue
        out.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(B39.pooled_run_dir("v2", r, k) / "metrics.json", out / "metrics.json")
        t1 = time.time()
        with open(out / "run.log", "w", encoding="utf-8") as log:
            rc = subprocess.run([sys.executable, str(Path(__file__).resolve()), "worker"], env=job_env(pop, r, k),
                                stdout=log, stderr=subprocess.STDOUT).returncode
        ok = rc == 0 and (out / "comparison_errors_test.npz").exists()
        print(f"[refs] {i}/{len(jobs)} {out.name}: rc {rc}, {'ok' if ok else 'FAILED'} ({(time.time() - t1) / 60:.1f} min, "
              f"{(time.time() - t0) / 3600:.2f} h)", flush=True)
        if not ok:
            raise SystemExit(f"FATAL: {out.name} failed (see {out / 'run.log'})")


def load(d):
    z = np.load(d / "comparison_errors_test.npz")
    return {"shot": z[f"{T}_shot"], "row": z[f"{T}_row"], "y": z[f"{T}_y_true"].astype(np.float64),
            **{q: z[f"{T}_se_{q}"].astype(np.float64) for q in REFS}}


def compare(a, b):
    same = len(a["shot"]) == len(b["shot"]) and np.array_equal(a["shot"], b["shot"]) and np.array_equal(a["row"], b["row"])
    o = {"rows": [int(len(a["shot"])), int(len(b["shot"]))], "same_rows": bool(same)}
    if same:
        o["y_bit_equal"] = bool(np.array_equal(a["y"], b["y"]))
        for q in REFS:
            o[f"{q}_bit_equal"] = bool(np.array_equal(a[q], b[q]))
            o[f"{q}_max_abs_diff"] = float(np.max(np.abs(a[q] - b[q])))
    return o


def check():
    res = {"cut_vs_b39": {}, "incl_vs_b27_incl": {}, "incl_rows": {}}
    ok = True
    for r in R65.REPEATS:
        for k in range(B27.K):
            mine = load(out_dir("cut", r, k))
            o = compare(mine, load(B39.pooled_run_dir("v2", r, k)))
            o["passes"] = bool(o["same_rows"] and all(o[f"{q}_bit_equal"] for q in REFS) and o["y_bit_equal"])
            ok &= o["passes"]
            res["cut_vs_b39"][f"r{r}f{k}"] = o
            inc = load(out_dir("incl", r, k))
            b27 = DATA / f".b27_seqv2_incl_r{r}_f{k}"
            if (b27 / "comparison_errors_test.npz").exists():
                res["incl_vs_b27_incl"][f"r{r}f{k}"] = compare(inc, load(b27))
            res["incl_rows"][f"r{r}f{k}"] = {"rows": int(len(inc["y"])), "rows_above_3keV": int((inc["y"] > 3000).sum()),
                                             "rows_cut_population": int(len(mine["y"]))}
    res["passes"] = bool(ok)
    CHECKS.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(f"[refs] causal GP (and hold, offline GP, PCHIP) of the cut population reproduce the stored B.39 errors "
          f"bit for bit on all 15 folds: {ok}", flush=True)
    for f, v in res["incl_vs_b27_incl"].items():
        print(f"[refs] incl {f}: same rows as B.27 incl {v['same_rows']}, " +
              (", ".join(f"{q} {v[f'{q}_bit_equal']}" for q in REFS) if v["same_rows"] else f"rows {v['rows']}"), flush=True)
    tot = {k: sum(v[k] for v in res["incl_rows"].values()) for k in ("rows", "rows_above_3keV", "rows_cut_population")}
    print(f"[refs] inclusive rows r2-r4: {tot}", flush=True)
    if not ok:
        raise SystemExit("FATAL: the cut references do not reproduce B.39")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "check":
        check()
    else:
        run_all()
        check()
