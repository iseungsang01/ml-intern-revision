"""B.70 phase 2: the filters KF0 / Kc / Kg / KFX of B.56 refitted per D1 item 2.9 (PREREGISTRATION_B70.md, Addendum 2).

The frozen definitions of `b56_kfx/kfx_model.py` (the filter `run`, `loss`, its own `iloc[1:-1]` loader) are executed by
`run_b56.ns()`.  Two changes, both from D1:
  - held repeats removed: a T_i value of the filter grid is set missing (in Y and LY) where seq_data.load_grid_files
    (cut 3000 eV, held removal) has none; filter row j is grid row j + 1 (checked file by file);
  - fitted on the TRAINING discharges of the r2-r4 fold (manifest `train_files`), not on every non-test discharge.
Starts, Nelder-Mead options, the clip 0.2 (and the sensitivity clips 0.1, 0.4) and the fit order are those of
run_b56.fit_unit, copied.  Scored on the D1 rows of the fold (data/.b70_rescore/b39p_v2_cut_r{r}_f{k}), located with
run_b56.map_rows_exact.

    py ces_prediction/experiments/b70_d1/p2_b56_fit.py [r2f0 ...] [--jobs N]
Output: data/.b70p2/b56/r{r}_f{k}.{npz,json} (run_b56's structure) and data/.b70/kf_params_r{r}_f{k}.json (B.71).
"""
from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
from scipy.optimize import minimize  # noqa: E402

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
sys.path.insert(0, str(EXP / "b56_kfx"))
sys.path.insert(0, str(HERE))
import run_b56 as B56  # noqa: E402
import runs_b70 as RB  # noqa: E402

DATA = RB.DATA
OUT = DATA / ".b70p2" / "b56"
PARAMS = DATA / ".b70"
REPEATS, K = (2, 3, 4), 5
T0 = time.time()


def log(m):
    print(f"[{time.time() - T0:7.0f}s] {m}", flush=True)


_HELD = {}


def ns_held():
    """run_b56.ns() with the held repeats removed from Y and LY (once per process)."""
    g = B56.ns()
    if "done" in _HELD:
        return g
    sys.path.insert(0, str(RB.SEQ_DIR))
    sys.path.insert(1, str(EXP.parent))
    from seq_data import load_grid_files
    grid, _ = load_grid_files(DATA, True, 3000.0)
    names = [Path(f).name for f in g["files"]]
    if names != sorted(grid):
        raise SystemExit("FATAL: the filter's file list differs from the grid's")
    removed = 0
    for i, nm in enumerate(names):
        a = grid[nm]
        nk = len(g["grid"][i]["t"])
        if len(a) != nk + 2 or not np.allclose(a[1:-1, 0].astype(np.float64), g["grid"][i]["t"], atol=1e-6):
            raise SystemExit(f"FATAL: {nm}: filter rows are not grid rows 1..n-2")
        tis = a[1:-1, 1].astype(np.float64)
        tik = g["Y"][i, :nk]
        if (np.isfinite(tis) & ~np.isfinite(tik)).any():
            raise SystemExit(f"FATAL: {nm}: a grid value the filter does not have")
        both = np.isfinite(tik) & np.isfinite(tis)
        if both.any() and np.max(np.abs(tik[both] - tis[both])) > 1e-3:
            raise SystemExit(f"FATAL: {nm}: values differ between the filter grid and the seq grid")
        held = np.isfinite(tik) & ~np.isfinite(tis)
        if held.any():
            g["Y"][i, :nk][held] = np.nan
            g["LY"][i, :nk][held] = np.nan
            removed += int(held.sum())
    _HELD["done"] = True
    _HELD["removed"] = removed
    log(f"held repeats removed from the filter grid: {removed}")
    return g


def fit_fold(r, k):
    unit = f"r{r}_f{k}"
    if (OUT / f"{unit}.npz").exists() and (PARAMS / f"kf_params_r{r}_f{k}.json").exists():
        return f"{unit}: cached"
    g = ns_held()
    loss, run, CLIP = g["loss"], g["run"], g["CLIP"]
    names = [Path(f).name for f in g["files"]]
    man = json.loads((B56.DATA / f".b27_split_r{r}_f{k}" / "split_manifest.json").read_text(encoding="utf-8"))
    train_f = np.array(sorted(names.index(n) for n in man["train_files"]))
    ref = B56.load_err(RB.new_dir("b39", "v2", r, k))
    test_f = np.unique(ref["shot"])
    if set(names[i] for i in test_f) - set(man["test_files"]):
        raise SystemExit(f"FATAL: {unit}: a scored discharge is not a test discharge (shot index != file index)")
    t0 = time.time()
    # ---- copied from run_b56.fit_unit (train_f = the fold's training discharges)
    nm = lambda maxiter, xatol: dict(maxiter=maxiter, xatol=xatol, fatol=1e-1)      # noqa: E731
    f0 = minimize(lambda th: loss(np.r_[th, 0, 0], train_f, False),
                  np.array([np.log(0.002), np.log(0.01), np.log(2.0)]), method="Nelder-Mead", options=nm(250, 1e-3))
    th0 = f0.x
    out = {"theta_kf0": th0.tolist(), "train_mse_kf0": float(f0.fun)}
    preds = {"kf0": run(np.r_[th0, 0, 0], None, False)}
    for clip, tag in [(B56.CLIP_MAIN, "")] + [(c, f"_clip{c}") for c in B56.CLIP_SENS]:
        CLIP[0] = clip
        if clip == B56.CLIP_MAIN:
            fc = minimize(lambda b_: loss(np.r_[th0, b_[0], 0.0], train_f, True), np.zeros(1), method="Nelder-Mead", options=nm(200, 1e-4))
            fg = minimize(lambda b_: loss(np.r_[th0, 0.0, b_[0]], train_f, True), np.zeros(1), method="Nelder-Mead", options=nm(200, 1e-4))
            out["b_kc"], out["b_kg"] = float(fc.x[0]), float(fg.x[0])
            out["train_mse_kc"], out["train_mse_kg"] = float(fc.fun), float(fg.fun)
            preds["kc"] = run(np.r_[th0, fc.x[0], 0.0], None, True)
            preds["kg"] = run(np.r_[th0, 0.0, fg.x[0]], None, True)
        fx = minimize(lambda b_: loss(np.r_[th0, b_], train_f, True), np.zeros(2), method="Nelder-Mead", options=nm(300, 1e-4))
        out[f"b_kfx{tag}"] = fx.x.tolist()
        out[f"train_mse_kfx{tag}"] = float(fx.fun)
        preds[f"kfx{tag}"] = run(np.r_[th0, fx.x], None, True)
    CLIP[0] = B56.CLIP_MAIN
    gi = np.full(len(ref["shot"]), -1, np.int64)
    for fi in test_f:
        m = ref["shot"] == fi
        gi[m] = B56.map_rows_exact(int(fi), ref["row"][m], ref["y"][m])
    ok = gi >= 0
    sig = {}
    for k_, P in preds.items():
        v = np.full(len(gi), np.nan)
        v[ok] = P[ref["shot"][ok], gi[ok]]
        sig[k_] = v
    # ---- end of the copy
    out.update({"fit_files": int(len(train_f)), "fit_set": "manifest train_files (D1 item 2.9)",
                "held_repeats_removed_from_filter_grid": _HELD["removed"], "seconds": round(time.time() - t0, 1),
                "rows": int(len(gi)), "rows_unmatched": int((~ok).sum())})
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / f"{unit}.npz", shot=ref["shot"], row=ref["row"], y=ref["y"], gi=gi,
             **{f"est_{k_}": v for k_, v in sig.items()})
    (OUT / f"{unit}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    params = {"KF0": [float(x) for x in th0], "KFX": [float(x) for x in np.r_[th0, out["b_kfx"]]],
              "use_x": {"KF0": False, "KFX": True},
              "order": "kfx_model.run theta: log q, log r, log k[, b_c, b_g]", "clip": B56.CLIP_MAIN,
              "Kc": [float(x) for x in np.r_[th0, out["b_kc"], 0.0]], "Kg": [float(x) for x in np.r_[th0, 0.0, out["b_kg"]]],
              "fold": {"repeat": r, "fold": k}, "fit_set": out["fit_set"], "fit_files": out["fit_files"],
              "held_repeats_removed": True, "train_mse_eV2": {"KF0": out["train_mse_kf0"], "KFX": out["train_mse_kfx"]},
              "registration": "ces_prediction/experiments/PREREGISTRATION_B70.md (D1 item 2.9, Addendum 2)"}
    PARAMS.mkdir(parents=True, exist_ok=True)
    (PARAMS / f"kf_params_r{r}_f{k}.json").write_text(json.dumps(params, indent=1), encoding="utf-8")
    return (f"{unit}: {len(gi)} rows, unmatched {int((~ok).sum())}, KF0 {np.round(th0, 3).tolist()}, "
            f"b {np.round(out['b_kfx'], 3).tolist()} ({out['seconds']} s)")


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    jobs = int(next((a.split("=")[1] for a in sys.argv[1:] if a.startswith("--jobs=")), "1"))
    units = [(int(u[1:].split("f")[0]), int(u.split("f")[1])) for u in args] or [(r, k) for r in REPEATS for k in range(K)]
    if jobs > 1:
        from multiprocessing import Pool
        with Pool(jobs) as p:
            for msg in p.starmap(fit_fold, units):
                log(msg)
    else:
        for r, k in units:
            log(fit_fold(r, k))


if __name__ == "__main__":
    main()
