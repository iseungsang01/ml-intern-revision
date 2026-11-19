"""B.70 phase 1 (b): the B.64 second stage applied to the D1 rows, for the network and for the twin.

B.64 did not save its fitted second-stage models (stage2_b64.fit_fold keeps only the predictions), so each one is
refitted here with stage2_b64's code, model (run_b37.gbr, fixed random_state) and inputs on the unchanged R-fit64 rows
(the out-of-fold rows of the four inner networks plus the outer fold's validation discharges under the B.39 network).
The refit is then applied twice:
  - to the old B.39 rows of the fold, exactly as stage2_b64.fit_fold did (check (d): the stored B.64 predictions must
    be reproduced within 1e-6 eV);
  - to the D1 rows of the fold (the re-scored B.39 runs, data/.b70_rescore/b39p_{v2,v2h}_cut_r{r}_f{k}), with the
    first-stage estimate rebuilt as in fit_fold: y + sign(CPU prediction - y) * |stored error| (the stored error is
    now the D1 re-scoring's own CPU error).
Addendum 1 of B.64 is kept: on a D1 row where any of the network's fifteen second-stage inputs is missing, both arms
keep their first-stage estimate; the boosted model's own handling of missing inputs (the "nan" estimates) is written
beside it.  CPU only; nothing is trained except the boosted second stage (as registered).

    py ces_prediction/experiments/b70_d1/stage2_b70.py [--folds r2f0 ...] [--jobs N]
Output: data/.b70/stage2_r{r}_f{k}.npz (resumable).
"""
from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "4"                         # as stage2_b64 (setdefault "4") when B.64 was run

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time as _time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
sys.path.insert(0, str(EXP / "b64_two_stage"))
sys.path.insert(0, str(EXP / "b45_ablation_rows"))
sys.path.insert(0, str(HERE))
import stage2_b64 as S2  # noqa: E402  (imports run_b37, run_b54, run_b39, run_b64, ...)
import run_b45 as B45  # noqa: E402
import runs_b70 as RB  # noqa: E402

B39, B54, P = S2.B39, S2.B54, S2.P
DATA = S2.R.DATA
CACHE = DATA / ".b70"
REPEATS = (2, 3, 4)
K = 5
T = "CES_TI"
D1_TOL_Y = 0.01                                   # eV: value in the file vs the stored y (run_b54 Y_TOL)
CHECK_D_TOL = 1e-6                                 # eV: refit vs stored B.64 predictions on the old rows
T0 = _time.time()


def log(m):
    print(f"[{_time.time() - T0:6.0f}s] {m}", flush=True)


def d1_dir(arm, r, k):
    return RB.new_dir("b39", arm, r, k)


def load_npz(path):
    z = np.load(path)
    return {"shot": z[f"{T}_shot"].astype(np.int64), "row": z[f"{T}_row"].astype(np.int64),
            "se": z[f"{T}_se_model"].astype(np.float64), "y": z[f"{T}_y_true"].astype(np.float64),
            "hold": z[f"{T}_se_persistence"].astype(np.float64), "gp": z[f"{T}_se_gp_causal"].astype(np.float64),
            "off": z[f"{T}_se_gp"].astype(np.float64), "pchip": z[f"{T}_se_pchip"].astype(np.float64)}


def locate(C, valid, d):
    """Grid file index (in C.names) and grid row of every scored row; checks the value in the file."""
    pos = {nm: j for j, nm in enumerate(C.names)}
    N = len(d["shot"])
    fidx, grow, yf = np.empty(N, np.int64), np.empty(N, np.int64), np.full(N, np.nan)
    for u in np.unique(d["shot"]):
        sel = np.flatnonzero(d["shot"] == u)
        nm = valid[int(u)]
        arr = C.grid[nm]
        gi = B45.compact_index(arr)
        g = gi[d["row"][sel]]
        fidx[sel], grow[sel] = pos[nm], g
        yf[sel] = arr[g, 1].astype(np.float64)
    dy = np.abs(yf - d["y"])
    if not np.all(dy <= D1_TOL_Y):
        raise SystemExit(f"FATAL: {int((~(dy <= D1_TOL_Y)).sum())} scored rows do not carry their value in the file")
    return fidx, grow


def fold(r, k):
    out = CACHE / f"stage2_r{r}_f{k}.npz"
    if out.exists():
        return f"r{r} f{k}: cached"
    C = S2.ctx()
    paths = S2.Paths(False)                      # data/.b64: base.npz (read only)
    base = S2.load_base(paths)
    valid = B54.valid_names(C)
    man = S2.manifest(r, k)
    test, val = S2.idx_of(C, man["test_files"]), S2.idx_of(C, man["val_files"])
    inner = [S2.idx_of(C, f) for f in P.inner_folds(r, k)]
    mu, sd = C.fast_stats(set(test))
    pos = {(int(a), int(b)): j for j, (a, b) in enumerate(zip(base["fidx"], base["row"]))}
    old = B54.load_rows(r, k)
    stored = np.load(paths.cache / f"stage2_r{r}_f{k}.npz")
    d1 = {arm: load_npz(d1_dir(arm, r, k) / "comparison_errors_test.npz") for arm in ("v2", "v2h")}
    if not (np.array_equal(d1["v2"]["shot"], d1["v2h"]["shot"]) and np.array_equal(d1["v2"]["row"], d1["v2h"]["row"])):
        raise SystemExit(f"FATAL: r{r} f{k}: the D1 network and twin were not scored on the same rows")
    fidx, grow = locate(C, valid, d1["v2"])
    y = d1["v2"]["y"]
    jj = np.array([pos.get((int(a), int(b)), -1) for a, b in zip(fidx, grow)])
    if (jj < 0).any():
        raise SystemExit(f"FATAL: r{r} f{k}: {(jj < 0).sum()} D1 rows are not base rows")
    if np.max(np.abs(base["y"][jj] - y)) > 1e-3:
        raise SystemExit(f"FATAL: r{r} f{k}: values of the base rows differ from the D1 values")
    fr_d1 = np.full((len(y), 9), np.nan)
    for i in np.unique(fidx):
        sel = np.flatnonzero(fidx == i)
        fr_d1[sel] = C.feats(int(i), mu, sd)[grow[sel]]
    jo = np.array([pos.get((int(a), int(b)), -1) for a, b in zip(old["fidx"], old["grow"])])
    yo = old["y"].astype(np.float64)
    fr_old = np.full((len(yo), 9), np.nan)
    for i in np.unique(old["fidx"]):
        sel = np.flatnonzero(old["fidx"] == i)
        fr_old[sel] = C.feats(int(i), mu, sd)[old["grow"][sel]]
    res = {"fidx": fidx, "grow": grow, "y": y, "shot": d1["v2"]["shot"], "row": d1["v2"]["row"]}
    info = {}
    for arm in ("v2", "v2h"):
        key = S2.ARM_KEY[arm]
        cols = list(S2.COLS[arm])
        t0 = _time.time()
        outer_net, om = S2.load_net(C, B39.pooled_run_dir(arm, r, k), arm)
        parts = []
        for j in range(P.N_INNER):
            net, _m = S2.load_net(C, P.run_dir(arm, r, k, j), arm)
            parts.append(S2.rows_for(C, base, inner[j], net, mu, sd, S2.COLS[arm]))
        pv = S2.rows_for(C, base, val, outer_net, mu, sd, S2.COLS[arm])
        Xf = np.concatenate([q[0][q[3]] for q in parts + [pv]])
        rf = np.concatenate([(q[1] - q[2])[q[3]] for q in parts + [pv]])
        model = S2.gbr().fit(Xf[:, cols], rf)
        t_fit = _time.time() - t0
        cpu_pred = {}

        def cpu(i):
            if i not in cpu_pred:
                cpu_pred[i] = C.predict(C.grid[C.names[int(i)]], outer_net)[0]
            return cpu_pred[i]

        # ---- check (d): the old rows, exactly as stage2_b64.fit_fold
        c_old_cpu = np.full(len(yo), np.nan)
        for i in np.unique(old["fidx"]):
            sel = np.flatnonzero(old["fidx"] == i)
            c_old_cpu[sel] = cpu(i)[old["grow"][sel]]
        c_old = yo + np.sign(c_old_cpu - yo) * np.sqrt(old[f"se_{key}"])
        X15o = S2.inputs(fr_old, base["age"][jo], base["hold"][jo], c_old, base["gp"][jo])
        oko = np.isfinite(X15o).all(1)
        e2o = c_old.copy()
        e2o[oko] = c_old[oko] + model.predict(X15o[:, cols][oko])
        e2o_nan = c_old + model.predict(X15o[:, cols])
        dev = {"est2": float(np.max(np.abs(e2o - stored[f"est_{key}2"]))),
               "est2nan": float(np.max(np.abs(e2o_nan - stored[f"est_{key}2nan"]))),
               "est1": float(np.max(np.abs(c_old - stored[f"est_{key}"]))),
               "fallback_equal": bool(np.array_equal(~oko, stored[f"fallback_{key}2"]))}
        # ---- the D1 rows
        c_cpu = np.full(len(y), np.nan)
        for i in np.unique(fidx):
            sel = np.flatnonzero(fidx == i)
            c_cpu[sel] = cpu(i)[grow[sel]]
        ae = np.sqrt(d1[arm]["se"])
        dc = np.abs(np.abs(c_cpu - y) - ae)
        dh = np.abs(np.abs(base["hold"][jj] - y) - np.sqrt(d1[arm]["hold"]))
        c = y + np.sign(c_cpu - y) * ae
        X15 = S2.inputs(fr_d1, base["age"][jj], base["hold"][jj], c, base["gp"][jj])
        if not np.isfinite(X15[:, list(S2.HIST_COLS)]).all():
            raise SystemExit(f"FATAL: r{r} f{k} {arm}: a history-side input is missing on a D1 row")
        okt = np.isfinite(X15).all(1)
        t1 = _time.time()
        corr = model.predict(X15[:, cols][okt])
        t_pred = _time.time() - t1
        est2 = c.copy()
        est2[okt] = c[okt] + corr
        res[f"est_{key}"] = c
        res[f"est_{key}2"] = est2
        res[f"est_{key}2nan"] = c + model.predict(X15[:, cols])
        res[f"fallback_{key}2"] = ~okt
        info[arm] = {"fit_rows_out_of_fold": int(sum(q[3].sum() for q in parts)), "fit_rows_validation": int(pv[3].sum()),
                     "fit_rows_stored_b64": None, "check_d_old_rows": int(len(yo)), "check_d_max_dev_eV": dev,
                     "check_d_passes": bool(dev["est2"] <= CHECK_D_TOL and dev["est2nan"] <= CHECK_D_TOL
                                            and dev["fallback_equal"]),
                     "d1_rows": int(len(y)), "d1_rows_first_stage_kept": int((~okt).sum()),
                     "d1_max_dev_cpu_first_stage_eV": float(np.nanmax(dc)),
                     "d1_rows_dev_above_0.5_eV": int((dc > 0.5).sum()),
                     "d1_max_dev_hold_eV": float(np.nanmax(dh)),
                     "outer_epochs_run": om["epochs_run"], "seconds_fit": round(t_fit, 1),
                     "seconds_per_d1_row_second_stage": t_pred / max(int(okt.sum()), 1)}
        sinfo = json.loads(str(stored["info"]))[arm]
        info[arm]["fit_rows_stored_b64"] = [sinfo["fit_rows_out_of_fold"], sinfo["fit_rows_validation"]]
        info[arm]["fit_rows_equal_b64"] = bool(info[arm]["fit_rows_stored_b64"] ==
                                               [info[arm]["fit_rows_out_of_fold"], info[arm]["fit_rows_validation"]])
    res["info"] = np.array(json.dumps(info))
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **res)
    return (f"r{r} f{k}: " + "; ".join(f"{a} check(d) {v['check_d_passes']} (max dev {v['check_d_max_dev_eV']['est2']:.2e} eV), "
                                       f"D1 rows {v['d1_rows']}, kept first stage {v['d1_rows_first_stage_kept']}"
                                       for a, v in info.items()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", nargs="*", default=[f"r{r}f{k}" for r in REPEATS for k in range(K)])
    ap.add_argument("--jobs", type=int, default=1)
    args = ap.parse_args()
    todo = []
    for f in args.folds:
        r, k = f[1:].split("f")
        todo.append((int(r), int(k)))
    if args.jobs > 1:
        from multiprocessing import Pool
        with Pool(args.jobs) as pool:
            for msg in pool.starmap(fold, todo):
                log(msg)
    else:
        for r, k in todo:
            log(fold(r, k))


if __name__ == "__main__":
    main()
