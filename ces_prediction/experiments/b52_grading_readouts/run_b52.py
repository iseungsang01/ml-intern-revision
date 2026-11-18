"""B.52 (PREREGISTRATION_B52.md): readouts asked for by the eleventh independent grading.

D  per discharge: share of test discharges in which the estimator has the lower mean squared error
L  linear filters LH (history) and LF (history + fast channels) fitted on the training discharges
G  skill by the age of the last visible value
V  skill on rows whose reported and last visible values are at or below 2,000 / 2,500 eV
P  percentiles of the absolute error (eV)

CPU only; no network is trained or run (part L fits ridge-penalised least squares).  Rows: k = 1 rows of the B.48
row table (`data/.b48/rows_cut_s{seed}.npz`); E and E_h from the B.49 cache (`data/.b49/s{seed}.npz`).

Run from the repo root:
    py ces_prediction/experiments/b52_grading_readouts/run_b52.py
Output: docs/paper/b52_grading_readouts.json / .md
"""
from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "4"

import json  # noqa: E402
import sys  # noqa: E402
import time as _time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
for sub in ("b30_thinned", "b37_plan", "b45_ablation_rows", "b51_bes_modes", "b49_average"):
    sys.path.insert(0, str(EXP / sub))
import run_b30 as R  # noqa: E402
import run_b37 as R37  # noqa: E402
from run_b45 import ci  # noqa: E402
import run_b49 as B49  # noqa: E402
import run_b39 as B39  # noqa: E402

REG = "ces_prediction/experiments/PREREGISTRATION_B52.md"
ROWS = R.DATA / ".b48"
B49C = R.DATA / ".b49"
CACHE = R.DATA / ".b52"
PAPER = R.REPO / "docs" / "paper"
OUT_JSON = PAPER / "b52_grading_readouts.json"
OUT_MD = PAPER / "b52_grading_readouts.md"
SEEDS = R.SEEDS
SEED, N_BOOT = 20261118, 2000            # bootstrap generator: default_rng(SEED + split) for every row set
EST = ("causal", "hist80", "gp", "hold")
LAB = {"causal": "estimator", "hist80": "twin", "gp": "causal GP", "hold": "hold", "E": "E", "E_h": "E_h",
       "LH": "LH", "LF": "LF"}
Y_TOL, TOL3 = 1e-3, 5e-4
MIN_ROWS_D = 50
LAMBDAS = (0.0, 0.001, 0.01, 0.1, 1.0, 10.0)
PCTS = (50, 75, 90, 95, 99)
THRESH = (2000.0, 2500.0)
T0 = _time.time()


def log(m):
    print(f"[{_time.time() - T0:6.0f}s] {m}", flush=True)


def load_tab(s):
    z = np.load(ROWS / f"rows_cut_s{s}.npz")
    return {c.split("__")[1]: z[c] for c in z.files if c.startswith("CES_TI__")}


def boot(fidx):
    """Discharge index per row -> (inverse, n, W) with W from default_rng(SEED + split) (set by caller)."""
    u, f = np.unique(fidx, return_inverse=True)
    return u, f, len(u)


def make_W(s, n):
    return np.random.default_rng(SEED + s).multinomial(n, np.full(n, 1.0 / n), size=N_BOOT).astype(np.float64)


def skill(se_a, se_b, f, n, W, m=None):
    m = np.ones(len(f), bool) if m is None else m
    a = np.bincount(f[m], se_a[m], n)
    b = np.bincount(f[m], se_b[m], n)
    with np.errstate(invalid="ignore", divide="ignore"):
        return [float(1 - a.sum() / b.sum()), *ci(1 - (W @ a) / (W @ b))]


def wquant(v, inv, W, q):
    """Quantile q of v with row weights W[:, inv] (inverted CDF), per resample (run_b49.wquant)."""
    order = np.argsort(v, kind="stable")
    vs, io = v[order], inv[order]
    out = np.empty(W.shape[0])
    for a in range(0, W.shape[0], 200):
        w = W[a:a + 200][:, io].astype(np.float32)
        c = np.cumsum(w, axis=1)
        k = (c < q * c[:, -1:]).sum(1)
        out[a:a + 200] = vs[np.minimum(k, len(vs) - 1)]
    return out


# ------------------------------------------------------------------------------------------ part L inputs
def dep10(x, seg):
    """x minus the mean of the ten rows before in the same segment (run_b51.dep10, copied unchanged)."""
    n = len(x)
    ok = np.isfinite(x)
    c1 = np.concatenate(([0.0], np.cumsum(np.where(ok, x, 0.0))))
    cn = np.concatenate(([0], np.cumsum(~ok)))
    out = np.full(n, np.nan)
    i = np.arange(10, n)
    i = i[(seg[i] == seg[i - 10]) & (cn[i] - cn[i - 10] == 0) & ok[i]]
    out[i] = x[i] - (c1[i] - c1[i - 10]) / 10.0
    return out


def file_inputs(C, i):
    """Rows of file i that enter part L and their raw (unstandardized) inputs.

    History (7): last three visible T_i strictly before the row in its segment (v1 = most recent), their ages in
    ms, the time step to the row before in units of 10 ms.  Fast (30): the 15 fast channels at the row and their
    departure from the mean of the ten rows before in the segment.  The T_i of the row itself is read only as the
    target (returned separately); the inputs index the T_i column only at visible rows with index < row.
    """
    arr = C.grid[C.names[i]]
    t = arr[:, 0].astype(np.float64)
    ti = arr[:, 1].astype(np.float64)
    seg = C.seg[i]
    n = len(t)
    obs = np.flatnonzero(np.isfinite(ti))
    r = np.arange(n)
    k = np.searchsorted(obs, r, side="left")              # number of visible rows strictly before r
    hist = np.full((n, 7), np.nan)
    okh = k >= 3
    rr = r[okh]
    p = [obs[k[okh] - j] for j in (1, 2, 3)]              # every index < rr by construction
    assert all((pj < rr).all() for pj in p)
    same = seg[p[2]] == seg[rr]                           # p3 in the segment of the row -> p1, p2 too
    rr = rr[same]
    p = [pj[same] for pj in p]
    hist[rr, 0:3] = np.column_stack([ti[pj] for pj in p])
    hist[rr, 3:6] = np.column_stack([(t[rr] - t[pj]) * 1000.0 for pj in p])
    hist[rr, 6] = C.tstep[i][rr]
    X = arr[:, 3:3 + C.nfast].astype(np.float64)
    fast = np.column_stack([X, np.column_stack([dep10(X[:, c], seg) for c in range(C.nfast)])])
    ok = np.isfinite(ti) & np.isfinite(hist).all(1) & np.isfinite(fast).all(1)
    rows = np.flatnonzero(ok)
    return rows, hist[rows], fast[rows], ti[rows]


def design(C, files):
    parts = [(i, *file_inputs(C, i)) for i in files]
    parts = [p for p in parts if len(p[1])]
    fid = np.concatenate([np.full(len(p[1]), p[0]) for p in parts])
    return (fid, np.concatenate([p[1] for p in parts]), np.concatenate([p[2] for p in parts]),
            np.concatenate([p[3] for p in parts]), np.concatenate([p[4] for p in parts]))


def ridge(X, y, lam):
    """Least squares with intercept on standardized X (train mean 0) and y: (X'X/n + lam I) w = X'y/n."""
    n = len(y)
    b0 = float(y.mean())
    yc = y - b0
    if lam == 0.0:
        w = np.linalg.lstsq(X, yc, rcond=None)[0]
    else:
        w = np.linalg.solve(X.T @ X / n + lam * np.eye(X.shape[1]), X.T @ yc / n)
    return b0, w


def part_l_fit(C, s):
    cache = CACHE / f"L_s{s}.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        return json.loads(str(z["info"])), {k: z[k] for k in z.files if k != "info"}
    man = C.manifest(s)
    idx = {nm: j for j, nm in enumerate(C.names)}
    tr = [idx[nm] for nm in sorted(man["train_files"]) if nm in idx]
    va = [idx[nm] for nm in sorted(man["val_files"]) if nm in idx]
    te = [idx[nm] for nm in sorted(man["test_files"]) if nm in idx]
    D = {lab: design(C, fl) for lab, fl in (("train", tr), ("val", va), ("test", te))}
    Xtr = np.column_stack([D["train"][2], D["train"][3]])
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd = np.where(sd > 0, sd, 1.0)
    my, sy = float(D["train"][4].mean()), float(D["train"][4].std())

    def std(d):
        return (np.column_stack([d[2], d[3]]) - mu) / sd, (d[4] - my) / sy

    Z = {lab: std(d) for lab, d in D.items()}
    info = {"train_discharges": len(tr), "val_discharges": len(va), "test_discharges_in_manifest": len(te),
            "rows": {lab: int(len(d[4])) for lab, d in D.items()},
            "discharges_with_rows": {lab: int(len(np.unique(d[0]))) for lab, d in D.items()},
            "target_mean_eV": my, "target_sd_eV": sy, "models": {}}
    out = {"test_fid": D["test"][0], "test_row": D["test"][1], "test_v1": D["test"][2][:, 0], "test_y": D["test"][4]}
    for name, cols in (("LH", np.arange(7)), ("LF", np.arange(37))):
        val_mse = {}
        fits = {}
        for lam in LAMBDAS:
            b0, w = ridge(Z["train"][0][:, cols], Z["train"][1], lam)
            fits[lam] = (b0, w)
            pv = b0 + Z["val"][0][:, cols] @ w
            val_mse[lam] = float(np.mean((pv - Z["val"][1]) ** 2))
        lam = min(LAMBDAS, key=lambda x: (val_mse[x], x))
        b0, w = fits[lam]
        out[name] = (b0 + Z["test"][0][:, cols] @ w) * sy + my
        ptr = (b0 + Z["train"][0][:, cols] @ w) * sy + my
        info["models"][name] = {"inputs": int(len(cols)), "lambda": lam,
                                "val_mse_std": {str(k): v for k, v in val_mse.items()},
                                "val_rmse_eV": float(np.sqrt(val_mse[lam]) * sy),
                                "train_rmse_eV": float(np.sqrt(np.mean((ptr - D["train"][4]) ** 2))),
                                "coef_std": [float(b0), *map(float, w)]}
    CACHE.mkdir(exist_ok=True)
    np.savez_compressed(cache, info=json.dumps(info), **out)
    return info, out


# ------------------------------------------------------------------------------------------ split
def split(C, s, b48, b49, readout=True):
    t0 = _time.time()
    tab = load_tab(s)
    man = C.manifest(s)
    sc = (tab["k"] == 1) & np.all([np.isfinite(tab[e]) for e in (*EST, "y")], axis=0)
    fi, ri = tab["file"][sc].astype(np.int64), tab["row"][sc].astype(np.int64)
    y = tab["y"][sc].astype(np.float64)
    pr = {e: tab[e][sc].astype(np.float64) for e in EST}
    age = tab["age_ms"][sc].astype(np.float64)
    chk = {}
    # ---------- check 1: grid T_i at (file, row) equals y of the table
    yg = np.array([C.grid[C.names[a]][b, 1] for a, b in zip(fi, ri)], dtype=np.float64)
    dy = np.abs(yg - y)
    test_names = set(man["test_files"])
    chk["join_table"] = {"scored_rows": int(len(y)), "max_abs_diff_eV": float(np.nanmax(dy)),
                         "rows_above_tol": int((~(dy <= Y_TOL)).sum()),
                         "files_outside_test": int(sum(C.names[a] not in test_names for a in np.unique(fi)))}
    chk["join_table"]["passes"] = bool(chk["join_table"]["rows_above_tol"] == 0 and chk["join_table"]["files_outside_test"] == 0)
    # ---------- check 2: unstratified skill on every scored row equals B.48 (B.44 overall, k = 1)
    se = {e: (pr[e] - y) ** 2 for e in EST}
    ref48 = b48["b44"]["splits"][str(s)]["overall"]
    chk["b48"] = {r: {"mine": float(1 - se["causal"].sum() / se[r].sum()), "b48": float(ref48[r][0])} for r in ("hist80", "gp", "hold")}
    for r in ("hist80", "gp", "hold"):
        chk["b48"][r]["equal_3dp"] = bool(abs(chk["b48"][r]["mine"] - chk["b48"][r]["b48"]) < TOL3)
    chk["b48"]["passes"] = bool(all(chk["b48"][r]["equal_3dp"] for r in ("hist80", "gp", "hold")))
    # ---------- check 3: E and E_h joined by (file, grid row); y and E1 of B.49
    z = np.load(B49C / f"s{s}.npz")
    ini = b49["splits"][str(s)]["initialisations"]
    E_rows = np.mean([z[f"v2_i{i}"] for i in ini], axis=0)
    Eh_rows = np.mean([z[f"v2h_i{i}"] for i in ini], axis=0)
    shot, crow = z["shot"].astype(np.int64), z["row"].astype(np.int64)
    gfile = np.empty(len(shot), np.int64)
    grow = np.empty(len(shot), np.int64)
    pos = {nm: j for j, nm in enumerate(C.names)}
    for u in np.unique(shot):
        m = shot == u
        nm = C.valid[int(u)]
        gfile[m] = pos[nm]
        grow[m] = R37_compact(C.grid[nm])[crow[m]]
    key = {(int(a), int(b)): j for j, (a, b) in enumerate(zip(fi, ri))}
    jj = np.array([key.get((int(a), int(b)), -1) for a, b in zip(gfile, grow)])
    hasE = jj >= 0
    E = np.full(len(y), np.nan)
    Eh = np.full(len(y), np.nan)
    E[jj[hasE]] = E_rows[hasE]
    Eh[jj[hasE]] = Eh_rows[hasE]
    mE = np.isfinite(E)
    yE_grid = np.array([C.grid[C.names[a]][b, 1] for a, b in zip(gfile[hasE], grow[hasE])], np.float64)
    # Addendum 1: E over the causal GP is checked against B.49 with the causal-GP errors B.49 used (stored by
    # eval_seq, `CES_TI_se_gp_causal` of the estimator's run), on the joined rows (errors of E against y of the table)
    st = B49.load_npz(B39.split_run_dir("v2", s, s))
    if not (np.array_equal(st["shot"], shot) and np.array_equal(st["row"], crow)):
        raise SystemExit(f"FATAL: split {s}: the B.49 cache and the stored npz are not on the same rows")
    gp_tab = pr["gp"][jj[hasE]]
    y_tab = y[jj[hasE]]
    se_E = (E_rows[hasE] - y_tab) ** 2
    e1_stored = float(1 - se_E.sum() / st["gp_causal"][hasE].sum())
    e1_table = float(1 - se_E.sum() / ((gp_tab - y_tab) ** 2).sum())
    e1_b49 = float(b49["splits"][str(s)]["E1_E_over_causal_gp"][0])
    dgp = np.abs(np.sqrt(st["gp_causal"][hasE]) - np.abs(gp_tab - y_tab))
    chk["join_E"] = {"rule": "Addendum 1", "b49_rows": int(len(shot)), "b49_rows_joined": int(hasE.sum()),
                     "scored_rows_with_E": int(mE.sum()), "scored_rows_without_E": int((~mE).sum()),
                     "y_max_abs_diff_eV": float(np.max(np.abs(yE_grid - y_tab))),
                     "y_rows_above_tol": int((~(np.abs(yE_grid - y_tab) <= Y_TOL)).sum()),
                     "y_stored_vs_table_max_abs_diff_eV": float(np.max(np.abs(st["y"][hasE] - y_tab))),
                     "E_over_gp_stored_gp": e1_stored, "E_over_gp_table_gp": e1_table, "E_over_gp_b49": e1_b49,
                     "equal_3dp": bool(abs(e1_stored - e1_b49) < TOL3)}
    chk["join_E"]["passes"] = bool(chk["join_E"]["y_rows_above_tol"] == 0 and chk["join_E"]["b49_rows_joined"] == len(shot)
                                   and chk["join_E"]["equal_3dp"])
    # reported (Addendum 1): rows on which the two computations of the causal GP differ, and where they sit
    bad = np.flatnonzero(dgp > 1e-3)
    where = []
    for k in bad:
        a_, r_ = int(gfile[hasE][k]), int(grow[hasE][k])
        arr = C.grid[C.names[a_]]
        sp = [(u, v) for u, v in R.block_spans(arr[:, 0].astype(np.float64), C.GAP)
              if u <= r_ < v][0]
        nvis = int(np.isfinite(arr[sp[0]:r_, 1]).sum())
        where.append({"file": C.names[a_], "grid_row": r_, "age_ms": float(tab["age_ms"][sc][jj[hasE][k]]),
                      "visible_values_before_in_span": nvis, "row_in_span": r_ - sp[0],
                      "gp_table": float(gp_tab[k]), "gp_unscaled_table": float(tab["gp_unscaled"][sc][jj[hasE][k]]),
                      "abs_err_stored": float(np.sqrt(st["gp_causal"][hasE][k])), "abs_err_table": float(abs(gp_tab[k] - y_tab[k]))})
    gpchk = json.loads((ROWS / "checks_rows.json").read_text(encoding="utf-8"))[str(s)]["b"]["CES_TI"]
    o_gp = {"rows_differ_over_1e-3_eV": int(len(bad)), "rows_differ_over_0.5_eV": int((dgp > 0.5).sum()),
            "max_abs_error_diff_eV": float(dgp.max()),
            "sse_gp_stored_over_table": float(st["gp_causal"][hasE].sum() / ((gp_tab - y_tab) ** 2).sum()),
            "discharges": sorted({w["file"] for w in where}), "rows": where,
            "b48_check_b_gp_share": float(gpchk["gp_causal"]), "b48_check_b_gp_share_abs": float(gpchk["gp_causal_abs"]),
            "b48_check_b_stored_rows": int(gpchk["stored_rows"])}
    o_gp["b48_check_b_rows_not_agreeing"] = int(round((1 - o_gp["b48_check_b_gp_share_abs"]) * o_gp["b48_check_b_stored_rows"]))
    chk["join_E"]["gp_two_computations"] = o_gp
    e1_mine = e1_stored
    log(f"s{s}: checks table {chk['join_table']['passes']} b48 {chk['b48']['passes']} E {chk['join_E']['passes']} "
        f"(E1 {e1_mine:+.4f} vs {e1_b49:+.4f}; {int(mE.sum())} rows with E)")
    # ---------- part L fit (cached), join and leakage check
    info, L = part_l_fit(C, s)
    lkey = {(int(a), int(b)): j for j, (a, b) in enumerate(zip(L["test_fid"], L["test_row"]))}
    lj = np.array([lkey.get((int(a), int(b)), -1) for a, b in zip(fi, ri)])
    mL = lj >= 0
    LH = np.full(len(y), np.nan)
    LF = np.full(len(y), np.nan)
    LH[mL], LF[mL] = L["LH"][lj[mL]], L["LF"][lj[mL]]
    v1 = L["test_v1"][lj[mL]]
    yl = L["test_y"][lj[mL]]
    chk["join_L"] = {"test_rows_of_design": int(len(L["test_y"])), "scored_rows_in_L": int(mL.sum()),
                     "design_rows_not_scored": int(len(L["test_y"]) - mL.sum()),
                     "y_max_abs_diff_eV": float(np.max(np.abs(yl - y[mL]))),
                     "y_rows_above_tol": int((~(np.abs(yl - y[mL]) <= Y_TOL)).sum()),
                     "v1_vs_hold_max_abs_diff_eV": float(np.max(np.abs(v1 - pr["hold"][mL]))),
                     "v1_vs_hold_rows_above_tol": int((~(np.abs(v1 - pr["hold"][mL]) <= Y_TOL)).sum())}
    chk["join_L"]["passes"] = bool(chk["join_L"]["y_rows_above_tol"] == 0 and chk["join_L"]["v1_vs_hold_rows_above_tol"] == 0)
    log(f"s{s}: part L joined {int(mL.sum())} rows; check {chk['join_L']['passes']}; lambdas "
        f"{info['models']['LH']['lambda']} / {info['models']['LF']['lambda']}")
    if not all(c["passes"] for c in chk.values()):
        return {"checks": chk, "stopped": "check(s) fail: " + ", ".join(k for k, c in chk.items() if not c["passes"]),
                "L_info": info}
    if not readout:
        return {"checks": chk, "L_info": info}

    o = {"checks": chk, "rows": int(len(y)), "L_info": info}
    _, f, n = boot(fi)
    W = make_W(s, n)
    o["discharges"] = int(n)
    o["test_discharges_in_manifest"] = int(len(man["test_files"]))
    se["E"], se["E_h"] = (E - y) ** 2, (Eh - y) ** 2
    ab = {k: np.sqrt(v) for k, v in se.items()}

    # ================= part D
    def part_d(est, refs, mrows):
        fsub = fi[mrows]
        u, ff = np.unique(fsub, return_inverse=True)
        cnt = np.bincount(ff, minlength=len(u))
        mse = {e: np.bincount(ff, se[e][mrows], len(u)) / cnt for e in (est, *refs)}
        mae = {}
        for e in (est, *refs):
            mae[e] = np.array([np.median(ab[e][mrows][ff == d]) for d in range(len(u))])
        el = cnt >= MIN_ROWS_D
        ne = int(el.sum())
        Wd = make_W(s, ne)
        r = {"discharges_all": int(len(u)), "discharges_eligible": ne, "rows_eligible": int(cnt[el].sum()),
             "rows_per_eligible_discharge_median": float(np.median(cnt[el]))}
        for ref in refs:
            ind = (mse[est][el] < mse[ref][el]).astype(float)
            indm = (mae[est][el] < mae[ref][el]).astype(float)
            sk_d = 1 - mse[est][el] / mse[ref][el]
            red = (mse[ref] - mse[est]) * cnt                    # per-discharge reduction of the SSE (all discharges)
            top = np.sort(red)[::-1][:10]
            r[ref] = {"share_lower_mse": [float(ind.mean()), *ci(Wd @ ind / ne)],
                      "share_lower_median_abs": [float(indm.mean()), *ci(Wd @ indm / ne)],
                      "ties_mse": int((mse[est][el] == mse[ref][el]).sum()),
                      "skill_per_discharge_quartiles": [float(x) for x in np.quantile(sk_d, [0.25, 0.5, 0.75])],
                      "top10_share_of_reduction": float(top.sum() / red.sum()),
                      "total_reduction_eV2": float(red.sum()),
                      "discharges_with_positive_reduction": int((red > 0).sum())}
            r[ref]["holds"] = bool(r[ref]["share_lower_mse"][1] > 0.5)
        return r

    allr = np.ones(len(y), bool)
    o["D"] = {"estimator": part_d("causal", ("gp", "hist80", "hold"), allr),
              "E": part_d("E", ("gp", "hist80", "E_h", "hold"), mE)}

    # ================= part L
    _, fl, nl = boot(fi[mL])
    Wl = make_W(s, nl)
    seL = {k: v[mL] for k, v in se.items() if k not in ("E", "E_h")}
    seL["LH"], seL["LF"] = (LH[mL] - y[mL]) ** 2, (LF[mL] - y[mL]) ** 2
    sk = lambda a, b: skill(seL[a], seL[b], fl, nl, Wl)                          # noqa: E731
    o["L"] = {"rows": int(mL.sum()), "row_share_of_scored": float(mL.mean()), "discharges": int(nl),
              "estimator_over_LF": sk("causal", "LF"), "LF_over_LH": sk("LF", "LH"),
              "LH_over_gp": sk("LH", "gp"), "LF_over_gp": sk("LF", "gp"),
              "LH_over_hold": sk("LH", "hold"), "LF_over_hold": sk("LF", "hold"),
              "twin_over_LH": sk("hist80", "LH"), "twin_over_LF": sk("hist80", "LF"),
              "estimator_over_gp_same_rows": sk("causal", "gp"), "estimator_over_LH": sk("causal", "LH"),
              "rmse_eV": {k: float(np.sqrt(v.mean())) for k, v in seL.items()},
              "lambda_LH": info["models"]["LH"]["lambda"], "lambda_LF": info["models"]["LF"]["lambda"]}
    mLE = mL & mE
    _, fle, nle = boot(fi[mLE])
    Wle = make_W(s, nle)
    o["L"]["E_over_LF"] = skill(se["E"][mLE], ((LF - y) ** 2)[mLE], fle, nle, Wle)
    o["L"]["E_over_LF_rows"] = int(mLE.sum())
    o["L1"] = bool(o["L"]["estimator_over_LF"][1] > 0)
    o["L2"] = bool(o["L"]["LF_over_LH"][1] > 0)

    # ================= part G
    a10 = np.round(age / 10.0)
    bins = {"10 ms": a10 == 1, "20 to 50 ms": (a10 >= 2) & (a10 <= 5), "above 50 ms": a10 > 5, "20 ms or more": a10 >= 2}
    o["age_unbinned_rows"] = int((a10 < 1).sum())
    o["G"] = {}
    for b, m in bins.items():
        o["G"][b] = {"rows": int(m.sum()), "row_share": float(m.mean()), "discharges": int(len(np.unique(fi[m]))),
                     **({f"over_{r}": skill(se["causal"], se[r], f, n, W, m) for r in ("gp", "hist80", "hold")} if m.any() else {}),
                     **({"twin_over_gp": skill(se["hist80"], se["gp"], f, n, W, m)} if m.any() else {})}
    o["G1"] = bool(o["G"]["20 ms or more"]["over_gp"][1] > 0)

    # ================= part V
    o["V"] = {"all": {f"over_{r}": skill(se["causal"], se[r], f, n, W) for r in ("gp", "hist80", "hold")}}
    for T in THRESH:
        m = (y <= T) & (pr["hold"] <= T)
        o["V"][str(int(T))] = {"rows": int(m.sum()), "row_share": float(m.mean()),
                               "discharges": int(len(np.unique(fi[m]))),
                               **{f"over_{r}": skill(se["causal"], se[r], f, n, W, m) for r in ("gp", "hist80", "hold")}}
    o["V1"] = bool(o["V"]["2000"]["over_gp"][1] > 0)
    o["V2"] = bool(o["V"]["2000"]["over_hist80"][1] > 0)

    # ================= part P
    ref_p = b48["b41"]["G"][str(s)]["abs_error_percentiles_eV"]
    o["P"] = {"scored_rows": {}, "rows_with_E": {}}
    for e in ("causal", "hist80", "gp", "hold"):
        v = ab[e]
        o["P"]["scored_rows"][e] = {str(q): [float(np.percentile(v, q)), *ci(wquant(v, f, W, q / 100))] for q in PCTS}
    _, fE, nE = boot(fi[mE])
    WE = make_W(s, nE)
    for e in ("E", "E_h", "causal", "hist80", "gp", "hold"):
        v = ab[e][mE]
        o["P"]["rows_with_E"][e] = {str(q): [float(np.percentile(v, q)), *ci(wquant(v, fE, WE, q / 100))] for q in PCTS}
    o["P"]["b48_cross_check_max_abs_diff_eV"] = float(max(abs(o["P"]["scored_rows"][e][str(q)][0] - ref_p[e][str(q)])
                                                         for e in ("causal", "hist80", "gp", "hold") for q in PCTS))
    o["runtime_s"] = round(_time.time() - t0, 1)
    dd = o["D"]["estimator"]
    log(f"s{s}: D1 {dd['gp']['share_lower_mse']} D2 {dd['hist80']['share_lower_mse']} L1 {o['L']['estimator_over_LF']} "
        f"L2 {o['L']['LF_over_LH']} G1 {o['G']['20 ms or more']['over_gp']} V1 {o['V']['2000']['over_gp']} "
        f"V2 {o['V']['2000']['over_hist80']}")
    return o


def R37_compact(arr):
    """Grid rows that dataset.py keeps (at least one target finite); B.49 `row` indexes these (run_b49.compact_index)."""
    return np.flatnonzero(np.isfinite(arr[:, 1:3]).any(1))


# ------------------------------------------------------------------------------------------ expectations
def expectations(res):
    S = [str(s) for s in SEEDS]
    v = lambda fn: [fn(res[s]) for s in S]                                      # noqa: E731
    E = []

    def rng_row(desc, vals, lo, hi, fmt="{:+.3f}"):
        E.append({"expectation": desc, "observed": ", ".join(fmt.format(x) for x in vals),
                  "met": bool(all(lo <= x <= hi for x in vals))})

    def cnt_row(desc, vals, lo, hi):
        k = int(sum(vals))
        E.append({"expectation": desc, "observed": f"{k} of 4", "met": bool(lo <= k <= hi)})

    d = lambda r: r["D"]["estimator"]                                            # noqa: E731
    rng_row("D1: share 0.50 to 0.65", v(lambda r: d(r)["gp"]["share_lower_mse"][0]), 0.50, 0.65, "{:.3f}")
    cnt_row("D1: interval above 0.5 on one to three splits", v(lambda r: d(r)["gp"]["holds"]), 1, 3)
    rng_row("D2: share 0.50 to 0.65", v(lambda r: d(r)["hist80"]["share_lower_mse"][0]), 0.50, 0.65, "{:.3f}")
    cnt_row("D2: interval above 0.5 on one to three splits", v(lambda r: d(r)["hist80"]["holds"]), 1, 3)
    rng_row("by the median absolute error (vs causal GP): 0.35 to 0.55",
            v(lambda r: d(r)["gp"]["share_lower_median_abs"][0]), 0.35, 0.55, "{:.3f}")
    rng_row("by the median absolute error (vs twin): 0.35 to 0.55",
            v(lambda r: d(r)["hist80"]["share_lower_median_abs"][0]), 0.35, 0.55, "{:.3f}")
    rng_row("ten discharges with the largest reduction hold 0.5 to 1.2 of the reduction vs the causal GP",
            v(lambda r: d(r)["gp"]["top10_share_of_reduction"]), 0.5, 1.2, "{:.3f}")
    cnt_row("L2 holds", v(lambda r: r["L2"]), 3, 4)
    rng_row("LF over LH +0.01 to +0.05", v(lambda r: r["L"]["LF_over_LH"][0]), 0.01, 0.05)
    cnt_row("L1 holds", v(lambda r: r["L1"]), 3, 4)
    rng_row("estimator over LF +0.03 to +0.10", v(lambda r: r["L"]["estimator_over_LF"][0]), 0.03, 0.10)
    rng_row("LH over the causal GP -0.05 to +0.03", v(lambda r: r["L"]["LH_over_gp"][0]), -0.05, 0.03)
    rng_row("rows of 20 ms or more: 0.02 to 0.04 of the scored rows", v(lambda r: r["G"]["20 ms or more"]["row_share"]),
            0.02, 0.04, "{:.3f}")
    rng_row("skill there over the causal GP +0.03 to +0.25", v(lambda r: r["G"]["20 ms or more"]["over_gp"][0]), 0.03, 0.25)
    cnt_row("G1: interval above zero on two to four splits", v(lambda r: r["G1"]), 2, 4)
    cnt_row("V1 holds", v(lambda r: r["V1"]), 3, 4)
    cnt_row("V2 holds", v(lambda r: r["V2"]), 3, 4)
    rng_row("V1 skill within 0.03 of the value on all rows",
            v(lambda r: r["V"]["2000"]["over_gp"][0] - r["V"]["all"]["over_gp"][0]), -0.03, 0.03)
    rng_row("V2 skill within 0.03 of the value on all rows",
            v(lambda r: r["V"]["2000"]["over_hist80"][0] - r["V"]["all"]["over_hist80"][0]), -0.03, 0.03)
    return E


CHOICES = [
    "Scored rows: the k = 1 rows of the B.48 row table in which the estimator (`causal`), the twin (`hist80`), hold, the "
    "causal GP and y are finite (the population of B.48 / B.50). `file` = index into the sorted file names of the cut "
    "grid (`run_b37.Ctx`), `row` = grid row.",
    "E and E_h: means of the three CPU estimates of the causal variant (resp. the twin) cached by B.49 "
    "(`data/.b49/s{seed}.npz`, initialisations as in b49_average.json); B.49 `shot` indexes the files with at least one "
    "finite target and `row` the compacted rows of that file (`run_b49.compact_index`); mapped to (file, grid row) and "
    "joined to the scored rows. Every quantity involving E uses the scored rows that carry E; the errors of E are taken "
    "against y of the table.",
    "Bootstrap: 2,000 multinomial resamples of the discharges of each row set from `numpy.random.default_rng(20261118 + "
    "split)` (a fresh generator per row set: all scored rows; the part L rows; the rows with E; the eligible discharges "
    "of part D); interval = 2.5 and 97.5 % quantiles (`run_b45.ci`). Skills: 1 - SSE_A / SSE_B on the same rows, "
    "per-discharge sums resampled. Percentile intervals: weighted quantiles under the resamples (`run_b49.wquant`).",
    "Part D: eligible = test discharges with at least 50 scored rows (for E: at least 50 scored rows with E). 'Lower' "
    "is strict. Share interval: the share over resamples of the eligible discharges. Quartiles of the per-discharge "
    "skill 1 - MSE_A / MSE_B over the eligible discharges. Reduction against a reference = SSE_ref - SSE_est per "
    "discharge over all scored discharges (not only the eligible ones); the share of the ten largest reductions is "
    "their sum over the total (it exceeds 1 when the others sum to a negative number). For E, D1 and D2 are given "
    "against the causal GP and the single twin; E against E_h is added.",
    "Part L rows: the T_i of the cut grid (held values removed, values above 3 keV removed) is finite at the row; the "
    "three most recent finite T_i strictly before the row lie in the segment of the row (segments of `run_b37.Ctx`: a "
    "new segment after a time step of at least 0.5 s); the ten rows before lie in the segment (`dep10` of run_b51: "
    "position 10 or later in the segment, all ten finite). Ages in ms from the row time; time step = step to the row "
    "before over 10 ms (`run_b37.Ctx.tstep`). Fast channels = grid columns 3 to 17 (9 BES, 4 ECEI, 2 MC; raw values), "
    "departure = value minus the mean of the ten rows before.",
    "Part L fitting: every input and the target standardized with the mean and standard deviation over the part L rows "
    "of the training discharges of the split (manifest `train_files`). Ridge: minimise mean squared error + lambda * "
    "|w|^2 on the standardized scale (the intercept unpenalised; lambda = 0 by `numpy.linalg.lstsq`); lambda chosen "
    "from {0, 0.001, 0.01, 0.1, 1, 10} by the mean squared error on the part L rows of the validation discharges "
    "(ties to the smaller lambda); the fit on the training rows with that lambda is applied once to the test rows (no "
    "refit on training plus validation).",
    "Part L scoring: the scored rows that are also part L rows (joined by (file, grid row)); checked that y of the "
    "design equals y of the table and that the first history input (the last visible value) equals `hold` of the table "
    "on every joined row (1e-3 eV). Every skill of part L is on those rows; E over LF on those rows that also carry E.",
    "Part G: age of the last visible value rounded to a multiple of 10 ms; bins 10 ms, 20 to 50 ms, above 50 ms; G1 "
    "reads the union of the last two bins ('20 ms or more').",
    "Part V: a scored row enters at threshold T when y <= T and hold <= T (hold is the last visible value).",
    "Part P: percentiles (numpy linear interpolation) of |estimate - y| on the scored rows for the estimator, the twin, "
    "the causal GP and hold, cross-checked against `b48_reread.json` (b41 G); and for all of them with E and E_h on "
    "the scored rows that carry E.",
    "Checks (stop rules): grid T_i = y of the table on every scored row; the unstratified skills of the estimator over "
    "the twin, the causal GP and hold equal B.48 (b44 overall) to three decimals (|difference| < 5e-4, as run_b50); "
    "every B.49 row joins a scored row and the grid T_i there equals y of the table; (Addendum 1) E over the causal GP, "
    "with the causal-GP errors stored by eval_seq (those B.49 used) and the errors of E against y of the table, equals "
    "the E1 point of b49_average.json to three decimals (|difference| < 5e-4); the part L join check above. All checks "
    "run on all four splits before any readout.",
]


# ------------------------------------------------------------------------------------------ render
def s3(x):
    return f"{x[0]:+.3f} [{x[1]:+.3f}, {x[2]:+.3f}]"


def f3(x):
    return f"{x[0]:.3f} [{x[1]:.3f}, {x[2]:.3f}]"


def hs(h):
    return "holds" if h else "does not hold"


def gp_section(res, S):
    L = ["## The two computations of the causal GP (reported, Addendum 1)", "",
         "B.49 used the causal-GP errors stored by `eval_seq`; every readout here uses the causal GP of the B.48 row table. "
         "Rows with E joined; errors against y of the table.", "",
         "| split | E over causal GP: stored GP (check) | table GP (used in the readouts) | B.49 E1 | rows whose absolute GP "
         "error differs by > 1e-3 eV (> 0.5 eV) | largest difference (eV) | SSE GP stored / table | discharges | B.48 check (b): "
         "share of stored rows with the GP within 0.5 eV (rows not agreeing) |", "|---|---|---|---|---|---|---|---|---|"]
    for s in S:
        e = res[s]["checks"]["join_E"]
        g = e["gp_two_computations"]
        L.append(f"| {s} | {e['E_over_gp_stored_gp']:+.5f} | {e['E_over_gp_table_gp']:+.5f} | {e['E_over_gp_b49']:+.5f} | "
                 f"{g['rows_differ_over_1e-3_eV']} ({g['rows_differ_over_0.5_eV']}) | {g['max_abs_error_diff_eV']:.1f} | "
                 f"{g['sse_gp_stored_over_table']:.5f} | {', '.join(g['discharges']) or '-'} | "
                 f"{g['b48_check_b_gp_share_abs']:.5f} ({g['b48_check_b_rows_not_agreeing']}) |")
    L += ["", "The rows that differ:", "", "| split | file | grid row | row in its span | visible values before it in the span | "
          "age (ms) | GP of the table (eV) | gp_unscaled of the table | abs error stored / table (eV) |", "|---|---|---|---|---|---|---|---|---|"]
    for s in S:
        for w in res[s]["checks"]["join_E"]["gp_two_computations"]["rows"]:
            L.append(f"| {s} | {w['file']} | {w['grid_row']} | {w['row_in_span']} | {w['visible_values_before_in_span']} | "
                     f"{w['age_ms']:.0f} | {w['gp_table']:.1f} | {w['gp_unscaled_table']:.1f} | "
                     f"{w['abs_err_stored']:.1f} / {w['abs_err_table']:.1f} |")
    L += ["", GP_CAUSE]
    return L


def render(o):
    S = [str(s) for s in SEEDS]
    res = o["splits"]
    L = ["# B.52 -- readouts asked for by the eleventh independent grading (registered: "
         "ces_prediction/experiments/PREREGISTRATION_B52.md)", "",
         "CPU; no network trained or run (part L fits ridge-penalised least squares). T_i, cut population, the four "
         "file-level splits (42, 1, 7, 123), test discharges. Rows: the k = 1 rows of the B.48 row table in which the "
         "estimator of B.39, the twin, hold and the causal GP are finite. E / E_h: the average of the three "
         "initialisations of the estimator / of the twin (B.49 cache). Skill of A over B = 1 - SSE_A / SSE_B on the same "
         "rows. Intervals: 95 %, 2,000 resamples of the test discharges. A statement holds on a split when its interval "
         "is above zero (D1, D2: above 0.5), and holds when that is so on at least three of four splits.", "",
         "## Checks before the readout", "",
         "| split | scored rows; grid T_i vs y: max abs diff (eV), rows > 1e-3; files outside test | estimator over twin / GP / hold (mine = B.48) | "
         "E (Addendum 1): B.49 rows joined / all; scored rows with E; y max diff (eV); E over GP with the stored GP errors = B.49 | part L: design test rows; joined; y max diff; v1 vs hold max diff (eV), rows > 1e-3 |",
         "|---|---|---|---|---|"]
    for s in S:
        c = res[s]["checks"]
        j, b, e, l_ = c["join_table"], c["b48"], c["join_E"], c["join_L"]
        L.append(f"| {s} | {j['scored_rows']:,}; {j['max_abs_diff_eV']:.1e}, {j['rows_above_tol']}; {j['files_outside_test']} | "
                 + " / ".join(f"{b[r]['mine']:+.4f} = {b[r]['b48']:+.4f}" for r in ("hist80", "gp", "hold")) + " | "
                 f"{e['b49_rows_joined']:,} / {e['b49_rows']:,}; {e['scored_rows_with_E']:,}; {e['y_max_abs_diff_eV']:.1e}; "
                 f"{e['E_over_gp_stored_gp']:+.4f} = {e['E_over_gp_b49']:+.4f} | "
                 f"{l_['test_rows_of_design']:,}; {l_['scored_rows_in_L']:,}; {l_['y_max_abs_diff_eV']:.1e}; "
                 f"{l_['v1_vs_hold_max_abs_diff_eV']:.1e}, {l_['v1_vs_hold_rows_above_tol']} |")
    stopped = [s for s in S if "stopped" in res[s]]
    if stopped:
        L += ["", "Pass / fail per check: " + "; ".join(
            f"split {s}: " + ", ".join(f"{k} {'passes' if c['passes'] else 'FAILS'}" for k, c in res[s]["checks"].items())
            for s in S) + ".", "",
              "**The run stopped: " + "; ".join(f"split {s}: {res[s]['stopped']}" for s in stopped) + ".** Every check was "
              "computed on every split before any readout; no readout (part D, L skills, G, V, P) was computed.", "",
              ] + gp_section(res, S)
        L += ["", "Part L fit (needed for its join check; no skill computed): chosen lambda LH / LF per split: " + "; ".join(
            f"{s}: {res[s]['L_info']['models']['LH']['lambda']:g} / {res[s]['L_info']['models']['LF']['lambda']:g}" for s in S) + ".",
              "", "## Deviations", "", o["deviations"], "", f"Run time: {o['runtime_s']:.0f} s."]
        return "\n".join(L) + "\n"
    L += [""] + gp_section(res, S) + ["", "All checks pass on every split. Leakage: the history inputs index the T_i column only at visible rows "
          "strictly before the row (asserted in the code), and the first history input equals `hold` of the table on "
          "every joined row (column above). Part P cross-check against b48_reread.json (b41 G percentiles of the "
          "estimator, twin, causal GP, hold): largest |difference| " + ", ".join(
              f"{res[s]['P']['b48_cross_check_max_abs_diff_eV']:.2e}" for s in S) + " eV.", "",
          "## Registered statements", "", "Per-split values in the order 42, 1, 7, 123.", ""]
    V = o["verdicts"]
    d = lambda s: res[s]["D"]["estimator"]                                       # noqa: E731
    L.append("- **D1** (share of the test discharges with >= 50 scored rows in which the estimator has a lower MSE than "
             "the causal GP; interval above 0.5): " + "; ".join(f3(d(s)["gp"]["share_lower_mse"]) + f" (n = {d(s)['discharges_eligible']})" for s in S)
             + f"; {V['D1']['splits']} of 4 -> **{hs(V['D1']['holds'])}**.")
    L.append("- **D2** (the same against the twin): " + "; ".join(f3(d(s)["hist80"]["share_lower_mse"]) for s in S)
             + f"; {V['D2']['splits']} of 4 -> **{hs(V['D2']['holds'])}**.")
    L.append("- **L1** (estimator over the linear filter LF): " + "; ".join(s3(res[s]["L"]["estimator_over_LF"]) for s in S)
             + f"; {V['L1']['splits']} of 4 -> **{hs(V['L1']['holds'])}**.")
    L.append("- **L2** (LF over LH): " + "; ".join(s3(res[s]["L"]["LF_over_LH"]) for s in S)
             + f"; {V['L2']['splits']} of 4 -> **{hs(V['L2']['holds'])}**.")
    L.append("- **G1** (rows whose last visible value is 20 ms old or older, estimator over the causal GP): "
             + "; ".join(s3(res[s]["G"]["20 ms or more"]["over_gp"]) for s in S) + f"; {V['G1']['splits']} of 4 -> **{hs(V['G1']['holds'])}**.")
    L.append("- **V1** (T = 2,000 eV, estimator over the causal GP): " + "; ".join(s3(res[s]["V"]["2000"]["over_gp"]) for s in S)
             + f"; {V['V1']['splits']} of 4 -> **{hs(V['V1']['holds'])}**.")
    L.append("- **V2** (T = 2,000 eV, estimator over the twin): " + "; ".join(s3(res[s]["V"]["2000"]["over_hist80"]) for s in S)
             + f"; {V['V2']['splits']} of 4 -> **{hs(V['V2']['holds'])}**.")
    L += ["", "Outcome under section 5 of the registration: " + " ".join(o["outcome"]), "",
          "## Part D -- per discharge", "",
          "Eligible discharges: at least 50 scored rows (for E: at least 50 scored rows with E). Shares with the 95 % "
          "interval over resamples of the eligible discharges; 'reduction' = SSE of the reference minus SSE of the "
          "estimator per discharge, over all scored discharges.", "",
          "| split | estimator | reference | discharges (eligible / all) | share lower MSE | share lower median abs error | "
          "per-discharge skill: quartiles | top ten share of the reduction | discharges with a positive reduction |",
          "|---|---|---|---|---|---|---|---|---|"]
    for s in S:
        for est, refs in (("estimator", ("gp", "hist80", "hold")), ("E", ("gp", "hist80", "E_h", "hold"))):
            r = res[s]["D"][est]
            for ref in refs:
                q = r[ref]["skill_per_discharge_quartiles"]
                L.append(f"| {s} | {est} | {LAB[ref]} | {r['discharges_eligible']} / {r['discharges_all']} | "
                         f"{f3(r[ref]['share_lower_mse'])} | {f3(r[ref]['share_lower_median_abs'])} | "
                         f"{q[0]:+.3f}, {q[1]:+.3f}, {q[2]:+.3f} | {r[ref]['top10_share_of_reduction']:.3f} | "
                         f"{r[ref]['discharges_with_positive_reduction']} |")
    L += ["", "## Part L -- linear filters fitted on the training discharges", "",
          "| split | train / val rows (discharges) | test rows scored (share of scored; discharges) | lambda LH / LF | "
          "estimator over LF (L1) | LF over LH (L2) | LH over causal GP | LF over causal GP | LH over hold | LF over hold |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for s in S:
        lo, inf = res[s]["L"], res[s]["L_info"]
        L.append(f"| {s} | {inf['rows']['train']:,} ({inf['discharges_with_rows']['train']}) / {inf['rows']['val']:,} "
                 f"({inf['discharges_with_rows']['val']}) | {lo['rows']:,} ({lo['row_share_of_scored']:.3f}; {lo['discharges']}) | "
                 f"{lo['lambda_LH']:g} / {lo['lambda_LF']:g} | {s3(lo['estimator_over_LF'])} | {s3(lo['LF_over_LH'])} | "
                 f"{s3(lo['LH_over_gp'])} | {s3(lo['LF_over_gp'])} | {s3(lo['LH_over_hold'])} | {s3(lo['LF_over_hold'])} |")
    L += ["", "| split | twin over LH | twin over LF | E over LF (rows) | estimator over LH | estimator over causal GP, same rows | "
          "RMSE (eV): estimator / twin / GP / hold / LH / LF |", "|---|---|---|---|---|---|---|"]
    for s in S:
        lo = res[s]["L"]
        rm = lo["rmse_eV"]
        L.append(f"| {s} | {s3(lo['twin_over_LH'])} | {s3(lo['twin_over_LF'])} | {s3(lo['E_over_LF'])} ({lo['E_over_LF_rows']:,}) | "
                 f"{s3(lo['estimator_over_LH'])} | {s3(lo['estimator_over_gp_same_rows'])} | "
                 + " / ".join(f"{rm[k]:.1f}" for k in ("causal", "hist80", "gp", "hold", "LH", "LF")) + " |")
    L += ["", "Validation MSE (standardized) by lambda, LH then LF: " + "; ".join(
        f"split {s}: " + " | ".join(", ".join(f"{k}: {v:.5f}" for k, v in res[s]["L_info"]["models"][m]["val_mse_std"].items())
                                    for m in ("LH", "LF")) for s in S) + ".", "",
          "## Part G -- by the age of the last visible value", "",
          "| split | bin | rows (share) | discharges | over causal GP | over twin | over hold | twin over causal GP |",
          "|---|---|---|---|---|---|---|---|"]
    for s in S:
        for b, g in res[s]["G"].items():
            if g["rows"]:
                L.append(f"| {s} | {b} | {g['rows']:,} ({g['row_share']:.4f}) | {g['discharges']} | {s3(g['over_gp'])} | "
                         f"{s3(g['over_hist80'])} | {s3(g['over_hold'])} | {s3(g['twin_over_gp'])} |")
            else:
                L.append(f"| {s} | {b} | 0 | 0 | n/a | n/a | n/a | n/a |")
    L += ["", "## Part V -- value threshold (reported value and last visible value at or below T)", "",
          "| split | T (eV) | rows (share) | discharges | over causal GP | over twin | over hold |", "|---|---|---|---|---|---|---|"]
    for s in S:
        a = res[s]["V"]["all"]
        L.append(f"| {s} | all rows | {res[s]['rows']:,} (1.0000) | {res[s]['discharges']} | {s3(a['over_gp'])} | "
                 f"{s3(a['over_hist80'])} | {s3(a['over_hold'])} |")
        for T in ("2000", "2500"):
            v = res[s]["V"][T]
            L.append(f"| {s} | {T} | {v['rows']:,} ({v['row_share']:.4f}) | {v['discharges']} | {s3(v['over_gp'])} | "
                     f"{s3(v['over_hist80'])} | {s3(v['over_hold'])} |")
    f1 = lambda x: f"{x[0]:.1f} [{x[1]:.1f}, {x[2]:.1f}]"                        # noqa: E731
    L += ["", "## Part P -- percentiles of the absolute error (eV)", "",
          "On every scored row (estimator, twin, causal GP, hold) and, separately, on the scored rows that carry E.", "",
          "| split | rows | estimate | 50th | 75th | 90th | 95th | 99th |", "|---|---|---|---|---|---|---|---|"]
    for s in S:
        for rowset, ests in (("scored_rows", ("causal", "hist80", "gp", "hold")),
                             ("rows_with_E", ("E", "E_h", "causal", "hist80", "gp", "hold"))):
            for e in ests:
                p = res[s]["P"][rowset][e]
                L.append(f"| {s} | {'all scored' if rowset == 'scored_rows' else 'with E'} | {LAB[e]} | "
                         + " | ".join(f1(p[str(q)]) for q in PCTS) + " |")
    L += ["", "## Expectations of the registration (section 6)", "", "| expectation | observed | met |", "|---|---|---|"]
    for e in o["expectations"]:
        L.append(f"| {e['expectation']} | {e['observed']} | {'yes' if e['met'] else 'no'} |")
    L += ["", f"{sum(e['met'] for e in o['expectations'])} of {len(o['expectations'])} expectations met.", "",
          "## Choices made where the registration left room", ""] + [f"- {c}" for c in CHOICES]
    L += ["", "## Deviations", "", o["deviations"], "",
          "## Limits", "", "Stated in advance (section 7): a discharge is one unit whatever its length; the linear filter "
          "is one choice of inputs; the value threshold is applied to the scored rows and the last visible value only (the "
          "estimator was trained with values up to 3 keV in its inputs); one initialisation for parts G and V; the test "
          "rows are those of every earlier readout.", "", f"Run time: {o['runtime_s']:.0f} s."]
    return "\n".join(L) + "\n"


def outcome(V):
    out = []
    out.append("D1 holds: the share is stated in the main result." if V["D1"]["holds"] else
               "D1 does not hold: the main result states the share of discharges with a lower MSE and that the gain of "
               "the mean comes from a minority of discharges; the limit list says so.")
    out.append("L1 holds: the recurrent estimator is separated from the linear filter with the fast channels." if V["L1"]["holds"] else
               "L1 does not hold: the paper states that a linear filter with the fast channels is not separated from the "
               "recurrent estimator, and the estimator is described as one way of reading the fast channels, not as the needed one.")
    if V["L2"]["holds"]:
        out.append("L2 holds: quoted as support for the fast-channel increment from a model with no early stopping and no initialisation.")
    else:
        out.append("L2 does not hold.")
    for k, txt in (("G1", "G1"), ("V1", "V1"), ("V2", "V2")):
        if not V[k]["holds"]:
            out.append(f"{txt} does not hold: one sentence in the main text.")
    return out


DEVIATIONS = ("Addendum 1 of the registration (2026-11-18, committed in f656d60 before any readout was computed): the first "
              "run stopped at the E check on split 42 (E over the causal GP +0.0771 with the causal GP of the B.48 row table "
              "against +0.0780 of B.49; splits 1, 7 and 123 passed). The join itself was exact (every B.49 row joined, y "
              "equal); B.49 had used the causal-GP errors stored by eval_seq, which differ from the GP of the row table on a "
              "few rows. The check is restated: E over the causal GP is compared with B.49 using the stored causal-GP errors "
              "(the run stops if a split differs at the third decimal); every readout uses the causal GP of the row table, as "
              "registered; the rows on which the two computations differ and the skill of E with both are reported. No other "
              "change from the registered definitions, rows, intervals, seeds or rules.")

GP_CAUSE = ("Why they differ (checked within the ten minutes allowed, not a readout): the GP of the table is reproduced "
            "exactly by `baselines_interpolation.predict_gp_causal` on the neighbours of the B.48 machinery (the last 16 "
            "visible values in the span), with the times in float32 or in float64 from the CSV alike, so time precision is "
            "not the cause, and `gp_unscaled` is not involved (it is NaN at k = 1). The differing rows come in runs of "
            "consecutive rows of one discharge (rows 54 to 68 of s31409 in split 42, rows 272 to 287 of s31978 in split 7) "
            "plus one isolated row (s32377, split 42). Each run is 15 to 16 rows long, as long as one visible value stays "
            "among the 16 nearest past values that enter the fit; this points to one neighbour value that differs between "
            "the neighbour set of eval_seq (`build_neighbor_set` on the dataset array) and that of the row table. Which value differs was not identified. It is of the same size as the disagreement "
            "recorded by check (b) of B.48 (column above).")


def main():
    CACHE.mkdir(exist_ok=True)
    b48 = json.loads((PAPER / "b48_reread.json").read_text(encoding="utf-8"))
    b49 = json.loads((PAPER / "b49_average.json").read_text(encoding="utf-8"))
    C = R37.Ctx()
    C.valid = [nm for nm in C.names if np.isfinite(C.grid[nm][:, 1:3]).any()]
    C.nfast = C.nb + C.ne + C.dims["mc"]
    if C.nfast != 15:
        raise SystemExit(f"FATAL: expected 15 fast channels, got {C.nfast} ({C.dims})")
    log(f"grid loaded: {len(C.names)} files, {C.nfast} fast channels")
    # pass 1: every check on every split, no readout; pass 2 only when all pass
    res = {str(s): split(C, s, b48, b49, readout=False) for s in SEEDS}
    if any("stopped" in r for r in res.values()):
        log("STOPPED: " + "; ".join(f"s{s} {r['stopped']}" for s, r in res.items() if "stopped" in r))
    else:
        for s in SEEDS:
            res[str(s)] = split(C, s, b48, b49)
            (CACHE / "partial.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    o = {"registration": REG, "seed": SEED, "n_boot": N_BOOT, "splits": res, "deviations": DEVIATIONS, "choices": CHOICES}
    if len(res) == len(SEEDS) and not any("stopped" in r for r in res.values()):
        V = {k: {"splits": int(sum(res[str(s)][k] if k in ("L1", "L2", "G1", "V1", "V2") else
                                   res[str(s)]["D"]["estimator"][{"D1": "gp", "D2": "hist80"}[k]]["holds"] for s in SEEDS))}
             for k in ("D1", "D2", "L1", "L2", "G1", "V1", "V2")}
        for k in V:
            V[k]["holds"] = bool(V[k]["splits"] >= 3)
        V["D_E"] = {ref: int(sum(res[str(s)]["D"]["E"][ref]["holds"] for s in SEEDS)) for ref in ("gp", "hist80", "E_h")}
        o["verdicts"] = V
        o["outcome"] = outcome(V)
        o["expectations"] = expectations(res)
    o["runtime_s"] = round(_time.time() - T0, 1)
    OUT_JSON.write_text(json.dumps(o, indent=1, default=float), encoding="utf-8")
    if "verdicts" in o or any("stopped" in r for r in res.values()):
        OUT_MD.write_text(render(o), encoding="utf-8", newline="\n")
    log("written " + str(OUT_JSON))


if __name__ == "__main__":
    main()
