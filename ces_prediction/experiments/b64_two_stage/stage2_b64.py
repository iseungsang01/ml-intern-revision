"""B.64 second stage and readouts (PREREGISTRATION_B64.md).  CPU only; no network is trained here.

The second stage is the model of B.38 (`run_b37.gbr`, `run_b37.inputs`; target = the residual of the first-stage
estimate), fitted per outer fold on the out-of-fold rows of its training discharges (the inner networks of run_b64.py)
together with its validation discharges (the B.39 outer-fold network), and applied once to the B.39 scored rows of
its test discharges.  The network takes all fifteen inputs of B.38; the twin takes the same inputs except the eight
computed from BES or ECEI (B.38 reads no Mirnov input), i.e. the time step, the CES age, hold, and the first-stage
estimate against hold and against the causal GP.

Stages (repo root; each caches under data/.b64/ and is resumable):
    py ces_prediction/experiments/b64_two_stage/stage2_b64.py check     # before any run: B.39 reproduction, input split
    py ces_prediction/experiments/b64_two_stage/stage2_b64.py base      # per-file rows: value, hold, causal GP, age
    py ces_prediction/experiments/b64_two_stage/stage2_b64.py fit       # predictions of every network + second stage
    py ces_prediction/experiments/b64_two_stage/stage2_b64.py readout   # statements T1-T6, reported quantities
    py ces_prediction/experiments/b64_two_stage/stage2_b64.py all       # the four in order
Options: --jobs N (fit: outer folds in parallel processes); --plumb (CPU plumbing before the GPU runs: the B.39
outer-fold network stands in for every inner network, so the "out-of-fold" rows are in-sample; outputs go to
data/.smoke_b64/ only and are never a result).
Output: docs/paper/b64_two_stage.json / .md
"""
from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""          # the GPU belongs to the training batches
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time as _time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
for _sub in ("b61_absolute_robust", "b60_relative_error", "b54_pooled_rows", "b39_primary", "b38_stage2", "b37_plan"):
    sys.path.insert(0, str(EXP / _sub))
sys.path.insert(0, str(HERE))
import run_b37 as M  # noqa: E402
import run_b54 as B54  # noqa: E402
import run_b39 as B39  # noqa: E402
import analyze_b39 as A39  # noqa: E402
import run_b61 as B61  # noqa: E402
import run_b60 as B60  # noqa: E402
import run_b64 as P  # noqa: E402
from run_b37 import R, gbr, inputs  # noqa: E402

REG = "ces_prediction/experiments/PREREGISTRATION_B64.md"
REPEATS = P.REPEATS
K = B39.B27.K
T = "CES_TI"
N_BOOT, N_BOOT_Q = 20_000, 2_000          # sums: 20,000 (analyze_b39.arm_p); weighted quantiles: 2,000 (run_b54.B_RATIO)
SEED_P39 = A39.SEED_POOLED                # 2728: the check reproduces analyze_b39.arm_p draw for draw
SEED = 20261164
REGISTERED_P2 = (0.053, 0.027, 0.076)     # registration sec. 2 / 3: network over twin on r2-r4
CHECK_TOL = 0.002
C_TOL_EV = 0.5                            # eV: recomputed first stage / hold / causal GP vs the stored B.39 errors
C_TOL_SHARE = 0.01                        # at most this share of rows may deviate by more (run_b38's check rule)
INPUT_NAMES = ("BES level", "BES departure from the 10-row mean", "BES 10-row sd", "BES one-row increment",
               "ECEI level", "ECEI departure from the 10-row mean", "ECEI 10-row sd", "ECEI one-row increment",
               "time step (rows of 10 ms)", "age of the last CES value (ms)", "hold",
               "first stage minus hold", "first stage minus causal GP",
               "|first stage minus hold|", "|first stage minus causal GP|")
FAST_COLS = tuple(range(8))
HIST_COLS = tuple(range(8, 15))
COLS = {"v2": FAST_COLS + HIST_COLS, "v2h": HIST_COLS}
ARM_KEY = {"v2": "c", "v2h": "t"}
LAB = {"c2": "two-stage network", "t2": "two-stage twin", "c": "network", "t": "twin", "hold": "hold",
       "gp": "causal GP", "off": "offline GP", "pchip": "PCHIP", "kfx": "KFX"}
T0 = _time.time()


def log(m):
    print(f"[{_time.time() - T0:6.0f}s] {m}", flush=True)


class Paths:
    def __init__(self, plumb):
        self.plumb = plumb
        self.cache = R.DATA / (".smoke_b64" if plumb else ".b64")
        self.out_json = (self.cache / "b64_two_stage_PLUMBING.json") if plumb else (B54.PAPER / "b64_two_stage.json")
        self.out_md = (self.cache / "b64_two_stage_PLUMBING.md") if plumb else (B54.PAPER / "b64_two_stage.md")

    def inner(self, arm, r, k, j):
        return B39.pooled_run_dir(arm, r, k) if self.plumb else P.run_dir(arm, r, k, j)


# ------------------------------------------------------------------------------------------ context
_C = {}


def ctx():
    if "C" not in _C:
        _C["C"] = M.Ctx()                 # pins CES_* (no per-discharge standardization), loads the cut grid
        log(f"run_b37.Ctx loaded: {len(_C['C'].names)} files")
    return _C["C"]


def load_net(C, run, arm):
    m = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
    if m["seq_model"] != arm or m.get("per_shot_norm_expanding") or m.get("zero_mc") \
            or m.get("extra_vt_channels") or m.get("extra_ti_channels") or m.get("train_ctx"):
        raise SystemExit(f"FATAL: {run.name} is not the registered {arm} network")
    if arm == "v2" and bool(m["per_shot_input_norm"]):
        raise SystemExit(f"FATAL: {run.name}: the network must use the statistics of the training discharges")
    model = C.MODELS[arm]()
    model.load_state_dict(C.torch.load(run / "weights" / "seq_lstm.pth", map_location="cpu"))
    model.eval()
    return (model, C.load_stats(m)), m


def manifest(r, k):
    return P.outer_manifest(r, k)


def idx_of(C, files):
    return [C.names.index(n) for n in sorted(files) if n in C.grid]


# ------------------------------------------------------------------------------------------ check (before any run)
def input_split_check(C):
    """Which inputs of run_b37.inputs depend on BES, ECEI, Mirnov: perturb each diagnostic group of one file and see
    which columns move.  Also: the twin's first stage does not move when every fast channel is perturbed."""
    i = next(j for j, nm in enumerate(C.names) if len(C.grid[nm]) > 400)
    nm = C.names[i]
    test = set(idx_of(C, manifest(2, 0)["test_files"]))
    mu, sd = C.fast_stats(test)
    rng = np.random.default_rng(SEED)
    rw = base_rows_of(C, i)
    rows = rw[:, 0].astype(int)
    age, h, g = rw[:, 4], rw[:, 2], rw[:, 3]
    net_c, _ = load_net(C, B39.pooled_run_dir("v2", 2, 0), "v2")
    net_t, _ = load_net(C, B39.pooled_run_dir("v2h", 2, 0), "v2h")
    nb, ne, nf = C.dims["bes"], C.dims["ecei"], C.dims["n_fast"]
    groups = {"BES": slice(3, 3 + nb), "ECEI": slice(3 + nb, 3 + nb + ne), "Mirnov": slice(3 + nb + ne, 3 + nf)}

    def X_of(arr_override):
        keep = C.grid[nm]
        C.grid[nm] = arr_override
        try:
            fr = C.feats(i, mu, sd)[rows]
            ct = C.predict(arr_override, net_t)[0][rows]
            cc = C.predict(arr_override, net_c)[0][rows]
        finally:
            C.grid[nm] = keep
        return inputs(fr, age, h, cc, g), inputs(fr, age, h, ct, g), cc, ct

    base_c, base_t, cc0, ct0 = X_of(C.grid[nm].copy())
    out = {"file": nm, "rows": int(len(rows)), "groups": {}}
    for gname, sl in groups.items():
        a = C.grid[nm].copy()
        a[:, sl] = a[:, sl] + rng.normal(0, 1, a[:, sl].shape).astype(a.dtype) * (np.nanstd(a[:, sl], 0) + 1e-3)
        Xc, Xt, cc, ct = X_of(a)
        moved = lambda X0, X1: [INPUT_NAMES[j] for j in range(X0.shape[1])                       # noqa: E731
                                if not np.allclose(X0[:, j], X1[:, j], equal_nan=True, atol=1e-9)]
        out["groups"][gname] = {"inputs_moved_network_arm": moved(base_c, Xc), "inputs_moved_twin_arm": moved(base_t, Xt),
                                "network_first_stage_moved": bool(not np.allclose(cc0, cc, equal_nan=True)),
                                "twin_first_stage_moved": bool(not np.allclose(ct0, ct, equal_nan=True))}
    fast_moved = sorted({x for gname in ("BES", "ECEI") for x in out["groups"][gname]["inputs_moved_twin_arm"]})
    out["fast_side"] = [INPUT_NAMES[j] for j in FAST_COLS]
    out["history_side"] = [INPUT_NAMES[j] for j in HIST_COLS]
    out["n_inputs"] = int(base_c.shape[1])
    out["passes"] = bool(base_c.shape[1] == len(INPUT_NAMES)
                         and fast_moved == sorted(out["fast_side"])
                         and not out["groups"]["Mirnov"]["inputs_moved_twin_arm"]
                         and not any(out["groups"][gname]["twin_first_stage_moved"] for gname in groups)
                         and all(out["groups"][gname]["network_first_stage_moved"] for gname in groups))
    return out


def reproduction_check():
    """(1) analyze_b39.arm_p restricted to r2-r4 (its own code, REPEATS set on the module object);
    (2) this script's copy of the pooled bootstrap on the B.54 row tables, same seed and draw order."""
    saved = A39.B.REPEATS
    A39.B.REPEATS = REPEATS
    try:
        p = A39.arm_p()
    finally:
        A39.B.REPEATS = saved
    mine = {}
    per = {}
    for r in REPEATS:
        parts = [B54.load_rows(r, k) for k in range(K)]
        shot = np.concatenate([q["shot"] for q in parts])
        u, inv = np.unique(shot, return_inverse=True)
        per[r] = {e: np.bincount(inv, np.concatenate([q[e] for q in parts]), len(u)) for e in ("se_c", "se_t", "se_gp")}
    rng = np.random.default_rng(SEED_P39)
    mine["P1"] = hier({r: per[r]["se_c"] for r in REPEATS}, {r: per[r]["se_gp"] for r in REPEATS}, rng, N_BOOT)[0]
    mine["P2"] = hier({r: per[r]["se_c"] for r in REPEATS}, {r: per[r]["se_t"] for r in REPEATS}, rng, N_BOOT)[0]
    o = {"arm_p_P2_network_over_twin": p["P2_causal_over_twin"], "arm_p_P1_network_over_gp": p["P1_causal_over_gp"],
         "copy_P2": mine["P2"], "copy_P1": mine["P1"], "registered_P2": list(REGISTERED_P2), "tolerance": CHECK_TOL,
         "median_abs_error_eV": p["median_abs_error_eV"]}
    o["arm_p_within_tolerance"] = bool(all(abs(a - b) <= CHECK_TOL for a, b in zip(p["P2_causal_over_twin"], REGISTERED_P2)))
    o["copy_equals_arm_p"] = bool(np.allclose(mine["P2"], p["P2_causal_over_twin"], atol=1e-12)
                                  and np.allclose(mine["P1"], p["P1_causal_over_gp"], atol=1e-12))
    o["passes"] = bool(o["arm_p_within_tolerance"] and o["copy_equals_arm_p"])
    return o


def stage_check(paths):
    C = ctx()
    res = {"registration": REG}
    res["names_match_b54"] = bool(list(B54.load_files()["names"]) == list(C.names))
    res["reproduction"] = reproduction_check()
    log(f"reproduction: arm_p P2 {res['reproduction']['arm_p_P2_network_over_twin']} (registered {REGISTERED_P2}); "
        f"copy equal {res['reproduction']['copy_equals_arm_p']}")
    res["input_split"] = input_split_check(C)
    log(f"input split: passes {res['input_split']['passes']}; " +
        "; ".join(f"{g}: {v['inputs_moved_twin_arm']}" for g, v in res["input_split"]["groups"].items()))
    res["overlap"] = {f"r{r}f{k}": P.overlap_check(r, k) for r in REPEATS for k in range(K)}
    res["passes"] = bool(res["names_match_b54"] and res["reproduction"]["passes"] and res["input_split"]["passes"])
    paths.cache.mkdir(parents=True, exist_ok=True)
    (paths.cache / "check.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    if not res["passes"]:
        raise SystemExit("FATAL: a check before the runs fails (see check.json); nothing is fitted")
    log("checks pass")


# ------------------------------------------------------------------------------------------ base rows
def base_rows_of(C, i):
    """run_b37.Ctx.rows without the network (copied from it): scored T_i rows of file i at the natural cadence:
    row, y, hold, causal GP, age (ms)."""
    arr = C.grid[C.names[i]]
    tm, y = arr[:, 0].astype(np.float64), arr[:, 1].astype(np.float64)
    out = []
    for a, b in R.block_spans(tm, C.GAP):
        obs = np.flatnonzero(~np.isnan(y[a:b])) + a
        for pos in range(1, len(obs)):
            r = obs[pos]
            lo = max(0, pos - R.GP_SIDE)
            gp = C.B.predict_gp_causal(tm[obs[lo:pos]], y[obs[lo:pos]], tm[r])
            out.append((r, y[r], y[obs[pos - 1]], gp, (tm[r] - tm[obs[pos - 1]]) * 1000.0))
    return np.array(out) if out else np.zeros((0, 5))


def stage_base(paths):
    out = paths.cache / "base.npz"
    if out.exists():
        log("base: cached")
        return
    C = ctx()
    parts = []
    for i in range(len(C.names)):
        rw = base_rows_of(C, i)
        parts.append(np.column_stack([np.full(len(rw), i), rw]) if len(rw) else np.zeros((0, 6)))
        if (i + 1) % 50 == 0:
            log(f"base {i + 1}/{len(C.names)}")
    a = np.concatenate(parts)
    paths.cache.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, fidx=a[:, 0].astype(np.int64), row=a[:, 1].astype(np.int64), y=a[:, 2], hold=a[:, 3],
                        gp=a[:, 4], age=a[:, 5], names=np.array(C.names))
    log(f"base: {len(a)} rows written")


def load_base(paths):
    z = np.load(paths.cache / "base.npz")
    b = {k: z[k] for k in z.files}
    if list(b["names"]) != list(ctx().names):
        raise SystemExit("FATAL: base.npz was built on another file list")
    b["by_file"] = {}
    for i in np.unique(b["fidx"]):
        b["by_file"][int(i)] = np.flatnonzero(b["fidx"] == i)
    return b


# ------------------------------------------------------------------------------------------ fit
def rows_for(C, base, files, net, mu, sd):
    """Base rows of the named files under one network: X (15 inputs), y, first stage, finite-input mask, causal GP."""
    X, y, c, g = [], [], [], []
    for i in files:
        sel = base["by_file"].get(i)
        if sel is None:
            continue
        rows = base["row"][sel]
        pred = C.predict(C.grid[C.names[i]], net)[0][rows]
        fr = C.feats(i, mu, sd)[rows]
        X.append(inputs(fr, base["age"][sel], base["hold"][sel], pred, base["gp"][sel]))
        y.append(base["y"][sel]); c.append(pred); g.append(base["gp"][sel])
    X, y, c, g = (np.concatenate(a) for a in (X, y, c, g))
    ok = np.isfinite(X).all(1) & np.isfinite(y) & np.isfinite(c)          # B.38's filter (all fifteen inputs finite)
    return X, y, c, ok, g


def fit_fold(args):
    r, k, plumb = args
    paths = Paths(plumb)
    out = paths.cache / f"stage2_r{r}_f{k}.npz"
    if out.exists():
        return f"r{r} f{k}: cached"
    C = ctx()
    base = load_base(paths)
    man = manifest(r, k)
    test, val = idx_of(C, man["test_files"]), idx_of(C, man["val_files"])
    inner = [idx_of(C, f) for f in P.inner_folds(r, k)]
    mu, sd = C.fast_stats(set(test))
    z = B54.load_rows(r, k)
    pos = {}
    for j, (a, b) in enumerate(zip(base["fidx"], base["row"])):
        pos[(int(a), int(b))] = j
    res = {"rows": int(len(z["shot"]))}
    info = {}
    for arm in ("v2", "v2h"):
        key = ARM_KEY[arm]
        t0 = _time.time()
        outer_net, om = load_net(C, B39.pooled_run_dir(arm, r, k), arm)
        parts, fold_info = [], []
        for j in range(P.N_INNER):
            run = paths.inner(arm, r, k, j)
            net, m = load_net(C, run, arm)
            p = rows_for(C, base, inner[j], net, mu, sd)
            parts.append(p)
            fold_info.append({"inner_fold": j, "run": run.name, "discharges": len(inner[j]), "rows": int(p[3].sum()),
                              "epochs_run": m["epochs_run"], "best_epoch": m["best_epoch"],
                              "skill_over_gp": float(1 - ((p[2] - p[1])[p[3]] ** 2).sum() / ((p[4] - p[1])[p[3]] ** 2).sum())})
        pv = rows_for(C, base, val, outer_net, mu, sd)
        Xf = np.concatenate([q[0][q[3]] for q in parts + [pv]])
        rf = np.concatenate([(q[1] - q[2])[q[3]] for q in parts + [pv]])
        model = gbr().fit(Xf[:, list(COLS[arm])], rf)
        t_fit = _time.time() - t0
        # test: the B.39 scored rows of the outer fold, in the order of the B.54 row table
        jj = np.array([pos.get((int(a), int(b)), -1) for a, b in zip(z["fidx"], z["grow"])])
        if (jj < 0).any():
            raise SystemExit(f"FATAL: r{r} f{k}: {(jj < 0).sum()} B.39 rows are not base rows")
        y = z["y"].astype(np.float64)
        if np.max(np.abs(base["y"][jj] - y)) > 1e-3:
            raise SystemExit(f"FATAL: r{r} f{k}: values of the base rows differ from the B.39 values")
        dh = np.abs(np.abs(base["hold"][jj] - y) - np.sqrt(z["se_hold"]))
        dg = np.abs(np.abs(base["gp"][jj] - y) - np.sqrt(z["se_gp"]))
        c_cpu = np.full(len(y), np.nan)
        for i in np.unique(z["fidx"]):
            sel = np.flatnonzero(z["fidx"] == i)
            c_cpu[sel] = C.predict(C.grid[C.names[int(i)]], outer_net)[0][z["grow"][sel]]
        ae = np.sqrt(z[f"se_{key}"])
        dc = np.abs(np.abs(c_cpu - y) - ae)
        shares = [float(np.mean(~(d <= C_TOL_EV))) for d in (dc, dh, dg)]
        if max(shares) > C_TOL_SHARE:
            raise SystemExit(f"FATAL: r{r} f{k} {arm}: recomputed first stage / hold / GP deviate from B.39 by more "
                             f"than {C_TOL_EV} eV on {shares} of rows")
        c = y + np.sign(c_cpu - y) * ae            # the stored B.39 estimate, its sign from the CPU recomputation
        fr = np.full((len(y), 9), np.nan)
        for i in np.unique(z["fidx"]):
            sel = np.flatnonzero(z["fidx"] == i)
            fr[sel] = C.feats(int(i), mu, sd)[z["grow"][sel]]
        Xt = inputs(fr, base["age"][jj], base["hold"][jj], c, base["gp"][jj])[:, list(COLS[arm])]
        okt = np.isfinite(Xt).all(1)
        t1 = _time.time()
        corr = model.predict(Xt[okt])
        t_pred = _time.time() - t1
        est2 = c.copy()
        est2[okt] = c[okt] + corr
        res[f"est_{key}"] = c
        res[f"est_{key}2"] = est2
        res[f"fallback_{key}2"] = ~okt
        info[arm] = {"fit_rows_out_of_fold": int(sum(q[3].sum() for q in parts)), "fit_rows_validation": int(pv[3].sum()),
                     "fit_rows_dropped_nonfinite": int(sum((~q[3]).sum() for q in parts + [pv])),
                     "test_rows": int(len(y)), "test_rows_first_stage_kept": int((~okt).sum()),
                     "max_dev_first_stage_eV": float(np.nanmax(dc)), "max_dev_hold_eV": float(np.nanmax(dh)),
                     "max_dev_gp_eV": float(np.nanmax(dg)),
                     "rows_dev_above_0.5_eV": {"first_stage": int((dc > C_TOL_EV).sum()), "hold": int((dh > C_TOL_EV).sum()),
                                                "gp": int((dg > C_TOL_EV).sum())}, "inner": fold_info,
                     "outer_epochs_run": om["epochs_run"], "seconds_fit": round(t_fit, 1),
                     "seconds_per_test_row_second_stage": t_pred / max(int(okt.sum()), 1)}
    res["y"] = z["y"]
    res["info"] = np.array(json.dumps(info))
    paths.cache.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **res)
    return f"r{r} f{k}: " + "; ".join(f"{a} fit {v['fit_rows_out_of_fold']}+{v['fit_rows_validation']} rows, "
                                      f"test kept first stage {v['test_rows_first_stage_kept']}" for a, v in info.items())


def stage_fit(paths, jobs):
    missing = []
    for r in REPEATS:
        for k in range(K):
            for arm in ("v2", "v2h"):
                for j in range(P.N_INNER):
                    d = paths.inner(arm, r, k, j)
                    if not P.done(d):
                        missing.append(d.name)
    if missing:
        raise SystemExit(f"FATAL: {len(missing)} inner runs are missing (first: {missing[:3]}); run run_b64.py first")
    todo = [(r, k, paths.plumb) for r in REPEATS for k in range(K)]
    if jobs > 1:
        from multiprocessing import Pool
        with Pool(jobs) as pool:
            for msg in pool.imap_unordered(fit_fold, todo):
                log(msg)
    else:
        for t in todo:
            log(fit_fold(t))


# ------------------------------------------------------------------------------------------ pooled bootstrap
def hier(num, den, rng, nboot, fn=lambda a, b: 1.0 - a / b, reps=REPEATS):
    """analyze_b39.arm_p.hier (copied, with the repeats as an argument): a repeat drawn, then its discharges
    resampled with replacement; point = mean over repeats of the ratio of per-discharge sums."""
    draws = rng.choice(np.array(reps), size=nboot)
    o = np.empty(nboot)
    for r in reps:
        sel = np.flatnonzero(draws == r)
        m, bb = num[r], den[r]
        idx = rng.integers(0, len(m), size=(len(sel), len(m)))
        with np.errstate(invalid="ignore", divide="ignore"):
            o[sel] = fn(m[idx].sum(1), bb[idx].sum(1))
    per = {str(r): float(fn(num[r].sum(), den[r].sum())) for r in reps}
    return [float(np.mean(list(per.values()))), *A39.ci(o)], per


def weights(rng, reps, n_of, nboot):
    draws = rng.choice(np.array(reps), size=nboot)
    W = {}
    for r in reps:
        sel = np.flatnonzero(draws == r)
        idx = rng.integers(0, n_of[r], size=(len(sel), n_of[r]))
        Wr = np.zeros((len(sel), n_of[r]))
        for j in range(len(sel)):
            Wr[j] = np.bincount(idx[j], minlength=n_of[r])
        W[r] = (sel, Wr)
    return W


def cell(D, mask_fn, ests, pairs, rng, nboot=N_BOOT, reps=REPEATS):
    """Every sum-based quantity of a row cell on the same two-level draws: MSE / Huber / trimmed skills of pairs,
    RMSE and MAE (eV) and their differences for pairs, shares within 10 % and 20 % and their differences."""
    W = weights(rng, reps, {r: D[r]["n"] for r in reps}, nboot)
    boot, per = {}, {}

    def quantities(S):
        o = {}
        for e in ests:
            o[f"rmse|{e}"] = np.sqrt(S[f"se|{e}"] / S["cnt"])
            o[f"mae|{e}"] = S[f"ae|{e}"] / S["cnt"]
            for th in ("10", "20"):
                o[f"share{th}|{e}"] = S[f"w{th}|{e}"] / S["cnt"]
        for a, b in pairs:
            o[f"skill_mse|{a}|{b}"] = 1 - S[f"se|{a}"] / S[f"se|{b}"]
            o[f"skill_huber|{a}|{b}"] = 1 - S[f"hub|{a}"] / S[f"hub|{b}"]
            o[f"skill_trim|{a}|{b}"] = 1 - S[f"set|{a}"] / S[f"set|{b}"]
            o[f"rmse_diff_eV|{a}|{b}"] = o[f"rmse|{a}"] - o[f"rmse|{b}"]
            o[f"share20_diff|{a}|{b}"] = o[f"share20|{a}"] - o[f"share20|{b}"]
            o[f"share10_diff|{a}|{b}"] = o[f"share10|{a}"] - o[f"share10|{b}"]
        return o

    for r in reps:
        d = D[r]
        m = mask_fn(r)
        f, n = d["f"][m], d["n"]
        S = {"cnt": np.bincount(f, None, n).astype(float),
             "cnt_trim": np.bincount(f, d["keep_trim"][m].astype(float), n)}
        for e in ests:
            se = d["se"][e][m]
            ae = np.sqrt(se)
            rel = ae / d["y"][m]
            S[f"se|{e}"] = np.bincount(f, se, n)
            S[f"ae|{e}"] = np.bincount(f, ae, n)
            S[f"hub|{e}"] = np.bincount(f, B61.huber(ae), n)
            S[f"set|{e}"] = np.bincount(f, se * d["keep_trim"][m], n)
            S[f"w10|{e}"] = np.bincount(f, (rel <= 0.10).astype(float), n)
            S[f"w20|{e}"] = np.bincount(f, (rel <= 0.20).astype(float), n)
        with np.errstate(invalid="ignore", divide="ignore"):
            per[r] = {kk: float(v) for kk, v in quantities({kk: v.sum() for kk, v in S.items()}).items()}
            sel, Wr = W[r]
            for kk, v in quantities({kk: Wr @ v for kk, v in S.items()}).items():
                boot.setdefault(kk, np.empty(nboot))[sel] = v
    return {kk: [float(np.mean([per[r][kk] for r in reps])), *A39.ci(boot[kk])] for kk in boot}, \
        {kk: {str(r): per[r][kk] for r in reps} for kk in boot}


def medians(D, ests, pairs, rng, nboot=N_BOOT_Q, reps=REPEATS):
    """Median absolute errors (eV) and their ratios under the two-level draws (run_b54.hier_quant, copied with the
    repeats as an argument; weighted quantiles of run_b41.wquant)."""
    W = weights(rng, reps, {r: D[r]["n"] for r in reps}, nboot)
    bq = {e: np.empty(nboot) for e in ests}
    pt = {e: [] for e in ests}
    for r in reps:
        d = D[r]
        sel, Wr = W[r]
        for e in ests:
            ae = np.sqrt(d["se"][e])
            bq[e][sel] = B54.B41.wquant(ae, d["f"], Wr, [0.5])[:, 0]
            pt[e].append(float(np.median(ae)))
    o = {"median_eV|" + e: [float(np.mean(pt[e])), *A39.ci(bq[e])] for e in ests}
    for a, b in pairs:
        o[f"median_ratio|{a}|{b}"] = [float(np.mean([x / y for x, y in zip(pt[a], pt[b])])), *A39.ci(bq[a] / bq[b])]
    return o


# ------------------------------------------------------------------------------------------ readout
def load_data(paths):
    D = {r: B54.repeat_data(r) for r in REPEATS}
    infos, kchk = {}, {}
    for r in REPEATS:
        est = {e: [] for e in ("c", "c2", "t", "t2", "kfx")}
        fb = {"c2": [], "t2": []}
        for k in range(K):
            z = np.load(paths.cache / f"stage2_r{r}_f{k}.npz")
            q = B54.load_rows(r, k)
            if len(z["y"]) != len(q["y"]) or np.max(np.abs(z["y"] - q["y"])) > 1e-6:
                raise SystemExit(f"FATAL: r{r} f{k}: the stage-2 rows are not the B.39 rows")
            for e in ("c", "c2", "t", "t2"):
                est[e].append(z[f"est_{e}"])
            for e in fb:
                fb[e].append(z[f"fallback_{e}"])
            kz = np.load(B61.B56 / f"r{r}_f{k}.npz")
            same = len(kz["shot"]) == len(q["shot"]) and np.array_equal(kz["shot"], q["shot"]) and np.array_equal(kz["row"], q["row"])
            kchk[f"r{r}_f{k}"] = bool(same and np.max(np.abs(kz["y"] - q["y"])) <= 1e-3)
            est["kfx"].append(kz["est_kfx"].astype(np.float64) if same else np.full(len(q["y"]), np.nan))
            infos[f"r{r}_f{k}"] = json.loads(str(z["info"]))
        d = D[r]
        for e, v in est.items():
            d["se"][e + ("_re" if e in ("c", "t") else "")] = (np.concatenate(v) - d["y"]) ** 2
        d["se"]["pchip"] = d["se_pchip"]
        d["fallback"] = {e: np.concatenate(v) for e, v in fb.items()}
        d["keep_trim"] = d["se"]["hold"] < np.quantile(d["se"]["hold"], 1 - B61.TRIM)     # run_b61.load_data
    return D, infos, kchk


def stage_readout(paths):
    D, infos, kchk = load_data(paths)
    chk = json.loads((paths.cache / "check.json").read_text(encoding="utf-8"))
    o = {"registration": REG, "plumbing_only": paths.plumb, "n_boot": N_BOOT, "n_boot_quantiles": N_BOOT_Q,
         "checks": {"before_runs": {k: chk[k] for k in ("reproduction", "input_split", "passes")}, "kfx_rows": kchk}}
    # checks: the first stage of the readout is the stored B.39 estimate (sign recovered), the same errors
    o["checks"]["first_stage_equals_stored"] = {str(r): {e: float(np.max(np.abs(D[r]["se"][e + "_re"] - D[r]["se"][e])))
                                                    for e in ("c", "t")} for r in REPEATS}
    runs = []
    for r in REPEATS:
        for k in range(K):
            for arm, v in infos[f"r{r}_f{k}"].items():
                for x in v["inner"]:
                    runs.append({"run": x["run"], "epochs_run": x["epochs_run"], "best_epoch": x["best_epoch"],
                                 "cap_bound": bool(x["epochs_run"] >= 80)})
    o["checks"]["inner_runs"] = {"runs": len(runs), "cap_bound": [x["run"] for x in runs if x["cap_bound"]]}
    o["fold_info"] = infos
    ests = ("c2", "t2", "c", "t", "hold", "gp", "off", "pchip", "kfx")
    pairs = [("c2", "c"), ("c2", "gp"), ("c", "gp"), ("c2", "t2"), ("t2", "t"), ("c", "t"), ("t2", "gp"),
             *[(a, b) for a in ("c2", "t2", "c") for b in ("hold", "off", "pchip", "kfx")]]
    all_rows = lambda r: np.ones(len(D[r]["y"]), bool)                                  # noqa: E731
    A, Aper = cell(D, all_rows, ests, pairs, B54.rng_for("b64|all"))
    Q = medians(D, ests, [("c2", "hold"), ("c2", "c"), ("c2", "gp"), ("t2", "hold"), ("c", "hold")],
                B54.rng_for("b64|medians"))
    o["all_rows"] = A
    o["all_rows_per_repeat"] = Aper
    o["medians"] = Q
    st = {"T1": {"value": A["skill_mse|c2|c"], "holds": bool(A["skill_mse|c2|c"][1] > 0),
                 "text": "two-stage network over the network: skill interval above zero"},
          "T2": {"value": A["skill_mse|c2|gp"], "network_over_gp": A["skill_mse|c|gp"],
                 "holds": bool(A["skill_mse|c2|gp"][1] > 0 and A["skill_mse|c2|gp"][0] > A["skill_mse|c|gp"][0]),
                 "text": "two-stage network over the causal GP: interval above zero and point above the network's"},
          "T3": {"value": A["skill_mse|c2|t2"], "holds": bool(A["skill_mse|c2|t2"][1] > 0),
                 "text": "two-stage network over the two-stage twin: interval above zero"},
          "T4": {"value": Q["median_ratio|c2|hold"], "holds": bool(Q["median_ratio|c2|hold"][2] < 1),
                 "text": "median absolute error, two-stage network over hold: interval below 1"},
          "T5": {"value": A["share20_diff|c2|gp"], "holds": bool(A["share20_diff|c2|gp"][1] > 0),
                 "text": "share within 20 %, two-stage network minus causal GP: interval above zero"},
          "T6": {"value": A["skill_mse|t2|t"], "holds": bool(A["skill_mse|t2|t"][1] > 0),
                 "text": "two-stage twin over the twin: interval above zero"}}
    o["statements"] = st
    # thirds by median T_i of the discharge (run_b60.thirds_lab)
    F = B54.load_files()
    tl = {r: B60.thirds_lab(D, F, r) for r in REPEATS}
    o["thirds"] = {}
    for j, tn in enumerate(B54.B44.THIRDS):
        q, _ = cell(D, lambda r, j=j: tl[r][D[r]["f"]] == j, ("c2", "t2", "c", "t", "gp", "hold"),
                    [("c2", "c"), ("c2", "gp"), ("c", "gp"), ("c2", "t2"), ("t2", "t")], B54.rng_for(f"b64|third{j}"))
        o["thirds"][tn] = {kk: v for kk, v in q.items() if kk.startswith(("skill_mse", "rmse|", "share20|"))}
    o["fallback_rows_share"] = {e: float(np.mean(np.concatenate([D[r]["fallback"][e] for r in REPEATS]))) for e in ("c2", "t2")}
    o["seconds_per_row_second_stage"] = {a: float(np.mean([v[a]["seconds_per_test_row_second_stage"] for v in infos.values()]))
                                         for a in ("v2", "v2h")}
    o["runtime_s"] = round(_time.time() - T0, 1)
    paths.out_json.write_text(json.dumps(o, indent=1, default=float), encoding="utf-8")
    paths.out_md.write_text(render(o), encoding="utf-8", newline="\n")
    log("written " + str(paths.out_md) + " | " + ", ".join(f"{k} {'holds' if v['holds'] else 'does not hold'}"
                                                          for k, v in st.items()))


def s3(x):
    return f"{x[0]:+.3f} [{x[1]:+.3f}, {x[2]:+.3f}]"


def f3(x):
    return f"{x[0]:.3f} [{x[1]:.3f}, {x[2]:.3f}]"


def e1(x):
    return f"{x[0]:.1f} [{x[1]:.1f}, {x[2]:.1f}]"


def render(o):
    A, Q, st = o["all_rows"], o["medians"], o["statements"]
    L = ["# B.64 — the two-stage estimator under the pooled protocol" + (" (PLUMBING ONLY, NOT A RESULT)" if o["plumbing_only"] else ""), "",
         f"Registered: {REG}. T_i, cut population, re-partitions r2 to r4 of B.39 (15 outer folds); 120 GPU runs of the "
         "out-of-fold networks, the second stage (B.38's model) on a CPU. Rows: the rows scored by B.39. Intervals: 95 %, "
         f"two-level bootstrap (a re-partition drawn, then its discharges), {o['n_boot']:,} draws for sums, "
         f"{o['n_boot_quantiles']:,} for medians.", ""]
    for k, v in st.items():
        val = f3(v["value"]) if k == "T4" else s3(v["value"])
        extra = f" (network alone over the causal GP {s3(v['network_over_gp'])})" if k == "T2" else ""
        L.append(f"- **{k}** ({v['text']}): {val}{extra} -> **{'holds' if v['holds'] else 'does not hold'}**.")
    L += ["", "## Skills (mean-squared error)", "", "| estimator | over hold | over causal GP | over offline GP | over PCHIP | over KFX |",
          "|---|---|---|---|---|---|"]
    for a in ("c2", "t2", "c"):
        cells = [s3(A[f"skill_mse|{a}|{b}"]) if f"skill_mse|{a}|{b}" in A else "" for b in ("hold", "gp", "off", "pchip", "kfx")]
        L.append(f"| {LAB[a]} | " + " | ".join(cells) + " |")
    L += ["", "| pair | MSE skill | Huber skill (200 eV) | 1 %-trimmed skill | RMSE difference (eV) | share within 20 %, difference | within 10 %, difference |",
          "|---|---|---|---|---|---|---|"]
    for a, b in (("c2", "c"), ("c2", "gp"), ("c", "gp"), ("c2", "t2"), ("c", "t"), ("t2", "t"), ("t2", "gp")):
        L.append(f"| {LAB[a]} over {LAB[b]} | {s3(A[f'skill_mse|{a}|{b}'])} | {s3(A[f'skill_huber|{a}|{b}'])} | "
                 f"{s3(A[f'skill_trim|{a}|{b}'])} | {s3(A[f'rmse_diff_eV|{a}|{b}'])} | {s3(A[f'share20_diff|{a}|{b}'])} | "
                 f"{s3(A[f'share10_diff|{a}|{b}'])} |")
    L += ["", "| estimator | RMSE (eV) | mean absolute error (eV) | median absolute error (eV) | share within 20 % | within 10 % |",
          "|---|---|---|---|---|---|"]
    for e in ("c2", "t2", "c", "t", "hold", "gp", "off", "pchip", "kfx"):
        L.append(f"| {LAB[e]} | {e1(A[f'rmse|{e}'])} | {e1(A[f'mae|{e}'])} | {e1(Q[f'median_eV|{e}'])} | "
                 f"{f3(A[f'share20|{e}'])} | {f3(A[f'share10|{e}'])} |")
    L += ["", "Median absolute error ratios: " + "; ".join(f"{LAB[k.split('|')[1]]} over {LAB[k.split('|')[2]]} {f3(v)}"
                                                         for k, v in Q.items() if k.startswith("median_ratio")) + ".", "",
          "## Thirds of the discharges by median T_i", "",
          "| third | two-stage network over network | over causal GP | network over causal GP | two-stage network over two-stage twin | two-stage twin over twin |",
          "|---|---|---|---|---|---|"]
    for tn, q in o["thirds"].items():
        L.append(f"| {tn} | {s3(q['skill_mse|c2|c'])} | {s3(q['skill_mse|c2|gp'])} | {s3(q['skill_mse|c|gp'])} | "
                 f"{s3(q['skill_mse|c2|t2'])} | {s3(q['skill_mse|t2|t'])} |")
    rc = o["checks"]["before_runs"]["reproduction"]
    L += ["", "## Checks", "",
          f"- B.39 network over twin on r2 to r4 from the stored errors: {s3(rc['arm_p_P2_network_over_twin'])} "
          f"(registered +0.053 [+0.027, +0.076], tolerance 0.002) -> {'passes' if rc['arm_p_within_tolerance'] else 'FAILS'}.",
          f"- Inputs: the twin's second stage reads {', '.join(o['checks']['before_runs']['input_split']['history_side'])}; "
          f"the network's also {', '.join(o['checks']['before_runs']['input_split']['fast_side'])}.",
          f"- Inner runs: {o['checks']['inner_runs']['runs']}; reached the epoch cap: {o['checks']['inner_runs']['cap_bound'] or 'none'}.",
          f"- Test rows where an input is missing and the first stage is kept: network {o['fallback_rows_share']['c2']:.4f}, "
          f"twin {o['fallback_rows_share']['t2']:.4f} of rows.",
          f"- Second stage run time per row: network {o['seconds_per_row_second_stage']['v2'] * 1e6:.1f} us, twin "
          f"{o['seconds_per_row_second_stage']['v2h'] * 1e6:.1f} us (batch prediction, CPU)."]
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["check", "base", "fit", "readout", "all"])
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--plumb", action="store_true")
    args = ap.parse_args()
    paths = Paths(args.plumb)
    stages = ["check", "base", "fit", "readout"] if args.stage == "all" else [args.stage]
    for s in stages:
        if s == "check":
            stage_check(paths)
        elif s == "base":
            stage_base(paths)
        elif s == "fit":
            if not (paths.cache / "check.json").exists() or not json.loads((paths.cache / "check.json").read_text())["passes"]:
                raise SystemExit("FATAL: run the check stage first")
            stage_fit(paths, args.jobs)
        else:
            stage_readout(paths)


if __name__ == "__main__":
    main()
