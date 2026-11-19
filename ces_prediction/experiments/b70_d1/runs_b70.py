"""B.70 phase 1 (PREREGISTRATION_B70.md section 3 (a)): the stored networks of the r2-r4 folds and how each is
re-scored on the D1 rows.

Every stored run is re-scored by its own batch's scoring script (seq/eval_seq.py; B.59 arm L: eval_b59L.py; B.67:
eval_b67.py) under its own runner's environment function, with exactly three variables changed:
CES_MAX_VAL_SAMPLES=0 (every eligible row, dataset.select_seeded_random_indices), CES_SEQ_DEVICE=cpu and
CES_OUTPUT_DIR (a new directory under data/.b70_rescore/).  The stored run directory is only read: its metrics.json,
weights and sidecar json are copied into the new directory, which the scoring script then reads and writes.

    py ces_prediction/experiments/b70_d1/runs_b70.py --list
    py ces_prediction/experiments/b70_d1/runs_b70.py --envdiff          # the three changed variables, every run
    py ces_prediction/experiments/b70_d1/runs_b70.py --one v2 2 0 [--cap 40000] [--dest DIR]
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
for _sub in ("b68_auxtarget", "b67_gated", "b59_inputs", "b39_primary"):
    sys.path.insert(0, str(EXP / _sub))
import run_b39 as B39  # noqa: E402  (adds b29_causal and b27_kfold)
import run_b59 as B59  # noqa: E402
import run_b67 as B67  # noqa: E402
import run_b68 as B68  # noqa: E402
from run_b39 import B27, DATA, SEQ_DIR  # noqa: E402

REPEATS = (2, 3, 4)
K = B27.K
OUT_ROOT = DATA / ".b70_rescore"
CHANGED = ("CES_MAX_VAL_SAMPLES", "CES_SEQ_DEVICE", "CES_OUTPUT_DIR")
# family -> arms; arm -> (stored dir fn, env fn(out), scoring script, sidecars copied with metrics.json)
FAMILIES = {
    "b39": ("v2", "v2h"),
    "b59": ("L", "N", "E", "B"),
    "b67": ("S", "A"),
    "b68": ("X",),
}


def stored_dir(fam, arm, r, k):
    if fam == "b39":
        return B39.pooled_run_dir(arm, r, k)
    if fam == "b59":
        return B59.run_dir(arm, r, k)
    if fam == "b67":
        return B67.run_dir(arm, r, k)
    if fam == "b68":
        return B68.run_dir(r, k)
    raise KeyError(fam)


def stored_env(fam, arm, r, k, out):
    """The runner's own env function, unchanged."""
    if fam == "b39":
        return B39.pooled_env(arm, r, k, out)
    if fam == "b59":
        return B59.arm_env(arm, r, k, out)
    if fam == "b67":
        return B67.arm_env(arm, r, k, out)
    if fam == "b68":
        return B68.arm_env(r, k, out)
    raise KeyError(fam)


def script(fam, arm):
    if fam == "b59" and arm == "L":
        return B59.EVAL_L
    if fam == "b67":
        return B67.EVAL
    return SEQ_DIR / "eval_seq.py"


def sidecars(fam, arm):
    if fam == "b59" and arm == "L":
        return ("b59_shift.json",)
    if fam == "b68":
        return ("b68_aux.json",)
    return ()


def new_dir(fam, arm, r, k, cap=0, root=OUT_ROOT):
    tag = "" if cap == 0 else f"_cap{cap}"
    return root / f"{stored_dir(fam, arm, r, k).name.lstrip('.')}{tag}"


def rescore_env(fam, arm, r, k, out, cap=0):
    env = stored_env(fam, arm, r, k, out)
    env["CES_MAX_VAL_SAMPLES"] = str(cap)
    env["CES_SEQ_DEVICE"] = "cpu"
    env["CES_OUTPUT_DIR"] = str(out)
    env["CUDA_VISIBLE_DEVICES"] = ""          # CPU only (the GPU is paused); not a CES_* variable
    env.setdefault("OMP_NUM_THREADS", "2")
    env.setdefault("MKL_NUM_THREADS", "2")
    env["PYTHONUNBUFFERED"] = "1"             # progress lines reach run.log; no effect on the numbers
    return env


def env_diff(fam, arm, r, k, cap=0):
    out = new_dir(fam, arm, r, k, cap)
    a = {kk: v for kk, v in stored_env(fam, arm, r, k, stored_dir(fam, arm, r, k)).items() if kk.startswith(("CES_", "B59_", "B67_", "B68_"))}
    b = {kk: v for kk, v in rescore_env(fam, arm, r, k, out, cap).items() if kk.startswith(("CES_", "B59_", "B67_", "B68_"))}
    return sorted(kk for kk in set(a) | set(b) if a.get(kk) != b.get(kk))


def plan():
    return [(fam, arm, r, k) for fam, arms in FAMILIES.items() for arm in arms for r in REPEATS for k in range(K)]


def prepare(fam, arm, r, k, out):
    """Copy metrics.json, the weights and the sidecar json of the stored run into the new directory."""
    src = stored_dir(fam, arm, r, k)
    for need in (src / "metrics.json", src / "weights" / "seq_lstm.pth", src / "comparison_errors_test.npz"):
        if not need.exists():
            raise SystemExit(f"FATAL: stored artifact missing: {need}")
    if out.resolve() == src.resolve():
        raise SystemExit("FATAL: the new directory is the stored run directory")
    (out / "weights").mkdir(parents=True, exist_ok=True)
    shutil.copy2(src / "metrics.json", out / "metrics.json")
    shutil.copy2(src / "weights" / "seq_lstm.pth", out / "weights" / "seq_lstm.pth")
    for s in sidecars(fam, arm):
        if not (src / s).exists():
            raise SystemExit(f"FATAL: sidecar missing: {src / s}")
        shutil.copy2(src / s, out / s)


def execute(fam, arm, r, k, cap=0, root=OUT_ROOT):
    out = new_dir(fam, arm, r, k, cap, root)
    npz = out / "comparison_errors_test.npz"
    rec = {"run": out.name, "family": fam, "arm": arm, "repeat": r, "fold": k, "cap": cap,
           "stored": stored_dir(fam, arm, r, k).name}
    if npz.exists():
        rec.update(status="exists")
        return rec
    prepare(fam, arm, r, k, out)
    env = rescore_env(fam, arm, r, k, out, cap)
    diff = env_diff(fam, arm, r, k, cap)
    if not set(diff) <= set(CHANGED):
        raise SystemExit(f"FATAL: {out.name}: env differs from the stored run's beyond {CHANGED}: {diff}")
    t0 = time.time()
    with open(out / "run.log", "a", encoding="utf-8") as log:
        log.write(f"\n===== b70 rescore {fam} {arm} r{r} f{k} cap {cap}: {script(fam, arm)} =====\n")
        log.flush()
        proc = subprocess.run([sys.executable, str(script(fam, arm))], cwd=B39.B29.REPO_ROOT, env=env,
                              stdout=log, stderr=subprocess.STDOUT)
    rec.update(status="ok" if (proc.returncode == 0 and npz.exists()) else f"failed rc={proc.returncode}",
               seconds=round(time.time() - t0, 1), env_changed=diff)
    (out / "b70_rescore.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--envdiff", action="store_true")
    ap.add_argument("--one", nargs=4, metavar=("ARM", "R", "K", "FAMILY_OR_AUTO"))
    ap.add_argument("--cap", type=int, default=0)
    args = ap.parse_args()
    if not any(DATA.glob("s*.csv")):
        raise SystemExit("FATAL: no shot CSVs in data/ -- real data required, aborting.")
    if args.list:
        for x in plan():
            print(*x, stored_dir(*x).name, "->", new_dir(*x).name, script(x[0], x[1]).name)
        print(len(plan()), "runs")
    if args.envdiff:
        for x in plan():
            print(*x, env_diff(*x))
    if args.one:
        arm, r, k, fam = args.one
        if fam == "auto":
            fam = next(f for f, arms in FAMILIES.items() if arm in arms)
        print(execute(fam, arm, int(r), int(k), args.cap), flush=True)


if __name__ == "__main__":
    main()
