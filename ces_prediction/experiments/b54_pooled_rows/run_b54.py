"""B.54 (PREREGISTRATION_B54.md): the row-level readouts of B.48, B.50 and B.52 on the rows of the pooled
cross-validation of B.39 (arm P: repeats r2..r9, five folds each; the estimator `.b39p_v2_cut_r{r}_f{k}` and the
history-only twin `.b39p_v2h_cut_r{r}_f{k}`).

CPU only; nothing is trained except the propensity model of B.37 F (B.50), refitted per fold on the non-test files
of the fold.  No network is run.  Definitions are taken from the earlier runners by import (functions called
unchanged) or, where a runner holds the definition inside a monolithic loop, by copying its lines unchanged
(marked "copied from").

Stages (run from the repo root, in this order; every stage caches under data/.b54/ and is resumable):
    py ces_prediction/experiments/b54_pooled_rows/run_b54.py files     # per-file features (classes, blip rows, median T_i)
    py ces_prediction/experiments/b54_pooled_rows/run_b54.py rows      # locate every stored row in its file
    py ces_prediction/experiments/b54_pooled_rows/run_b54.py prop      # propensity per fold (B.50), joined to the rows
    py ces_prediction/experiments/b54_pooled_rows/run_b54.py checks    # checks (a) to (e); stops if one fails
    py ces_prediction/experiments/b54_pooled_rows/run_b54.py readout   # statements, reported quantities, outputs
Output: docs/paper/b54_pooled_rows.json / .md
"""
from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""          # the GPU belongs to the running training batch
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "4"

import json  # noqa: E402
import sys  # noqa: E402
import time as _time  # noqa: E402
import zlib  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
for sub in ("b30_thinned", "b37_plan", "b41_events", "b44_strata", "b45_ablation_rows", "b50_missing_like",
            "b39_primary", "b32_two_sources"):
    sys.path.insert(0, str(EXP / sub))
import run_b30 as R  # noqa: E402
import run_b37 as R37  # noqa: E402
import run_b41 as B41  # noqa: E402
import run_b44 as B44  # noqa: E402
import run_b45 as B45  # noqa: E402
import run_b50 as B50  # noqa: E402
import run_b39 as B39  # noqa: E402
from run_b39 import B27  # noqa: E402

REG = "ces_prediction/experiments/PREREGISTRATION_B54.md"
DATA = R.DATA
CACHE = DATA / ".b54"
PROP = CACHE / "prop"
PAPER = R.REPO / "docs" / "paper"
OUT_JSON = PAPER / "b54_pooled_rows.json"
OUT_MD = PAPER / "b54_pooled_rows.md"
T = "CES_TI"
REPEATS = B39.REPEATS
K = B27.K
B_SKILL, B_RATIO = 20_000, 2_000
SEED = 20261154
SEED_P39 = 2728                     # analyze_b39.SEED_POOLED (check (a): the B.39 interval itself)
Y_TOL = 0.01                        # eV, check (d)
TOL3 = 5e-4                         # "equal to three decimals"
HOLD_TOL = 0.5                      # eV, hold recomputed from the file vs the stored hold error (reported)
MIN_ROWS_D = 50                     # run_b52.MIN_ROWS_D
EST = ("c", "t", "hold", "gp", "off")    # estimator, twin, hold, causal GP (eval_seq), offline GP (eval_seq)
LAB = {"c": "estimator", "t": "twin", "hold": "hold", "gp": "causal GP", "off": "offline GP"}
T0 = _time.time()


def log(m):
    print(f"[{_time.time() - T0:6.0f}s] {m}", flush=True)


def ci(b):
    b = np.asarray(b, float)
    b = b[np.isfinite(b)]
    return [float(np.quantile(b, 0.025)), float(np.quantile(b, 0.975))]


def rng_for(name):
    return np.random.default_rng([SEED, zlib.crc32(name.encode())])


# ------------------------------------------------------------------------------------------ contexts
class PCtx(R37.Ctx):
    """run_b37.Ctx (the propensity inputs of B.37 F / B.50) whose manifest also serves the pooled folds."""

    def manifest(self, s):
        if isinstance(s, str):
            r, k = s[1:].split("f")
            return json.loads((B27.split_dir(int(r), int(k)) / "split_manifest.json").read_text(encoding="utf-8"))
        return super().manifest(s)


_CTX = {}


def ctx41():
    if "41" not in _CTX:
        _CTX["41"] = B41.Ctx()          # pins the CES_* env, loads the cut grid, after_low_bes
        log(f"run_b41.Ctx loaded: {len(_CTX['41'].names)} files")
    return _CTX["41"]


def ctx37():
    if "37" not in _CTX:
        _CTX["37"] = PCtx()
        log(f"run_b37.Ctx loaded: {len(_CTX['37'].names)} files")
    return _CTX["37"]


def valid_names(C):
    """dataset.valid_files of eval_seq (run_b45.main): grid files with at least one finite target."""
    return [nm for nm in C.names if np.isfinite(C.grid[nm][:, 1:3]).any()]


# ------------------------------------------------------------------------------------------ stage files
def blip_rows_13(C, i):
    """Grid rows 1 to 3 after an absent frame of file i: the event rule and the row walk of
    run_b41.part_b, copied unchanged (rows k = 1..3; a row enters the group once)."""
    t, ti = C.t[i], C.ti[i]
    dt = np.diff(t)
    n = len(t)
    mark = np.zeros(n, bool)
    for j in np.flatnonzero((dt > 0.015) & (dt < 0.045)):
        if j < B41.PRE_ROWS or not np.all(np.abs(dt[j - B41.PRE_ROWS:j] - 0.010) < 1e-4):
            continue
        pre = ti[j - B41.PRE_ROWS:j + 1]
        pre = pre[np.isfinite(pre)]
        if len(pre) < 3:
            continue
        for k in range(1, 4):
            r = j + k
            if r >= n or (k > 1 and abs(dt[r - 1] - 0.010) > 1e-4):
                break
            mark[r] = True
    return mark


def stage_files():
    CACHE.mkdir(parents=True, exist_ok=True)
    out = CACHE / "files.npz"
    if out.exists():
        log("files: cached")
        return
    C = ctx41()
    Cl = B45.Classes(C)
    arrs = {}
    for i, nm in enumerate(C.names):
        cls, cls_alt = Cl.of_file(nm)
        arrs[f"cls_{i}"] = cls
        arrs[f"clsalt_{i}"] = cls_alt
        arrs[f"blip13_{i}"] = blip_rows_13(C, i)
        if (i + 1) % 100 == 0:
            log(f"classes {i + 1}/{len(C.names)}")
    Q, qchk = B44.discharge_quantities(C.grid, C.names, C.dims["bes"])
    arrs["median_ti"] = np.array([Q[i]["median_ti"] for i in range(len(C.names))], float)
    arrs["shot"] = np.array([Q[i]["shot"] for i in range(len(C.names))], float)
    arrs["names"] = np.array(C.names)
    np.savez_compressed(out, **arrs)
    (CACHE / "files_checks.json").write_text(json.dumps({"classes_csv": Cl.csv_check, "b44": qchk}, indent=1))
    log("files: written")


def load_files():
    z = np.load(CACHE / "files.npz")
    return {k: z[k] for k in z.files}


# ------------------------------------------------------------------------------------------ stage rows
def load_run(path):
    z = np.load(path / "comparison_errors_test.npz")
    return {"shot": z[f"{T}_shot"].astype(np.int64), "row": z[f"{T}_row"].astype(np.int64),
            "se": z[f"{T}_se_model"].astype(np.float64), "y": z[f"{T}_y_true"].astype(np.float64),
            "hold": z[f"{T}_se_persistence"].astype(np.float64), "gp": z[f"{T}_se_gp_causal"].astype(np.float64),
            "off": z[f"{T}_se_gp"].astype(np.float64), "pchip": z[f"{T}_se_pchip"].astype(np.float64),
            "dt_ms": z[f"{T}_dt_ms"].astype(np.float64)}


def locate(C, valid, F, d):
    """Grid file, grid row, value in the file, hold (W = 2 hold of run_b45.hold_compacted) and its age for every
    stored row; plus the per-row classes, blip-row flag and plain flag (run_b41.plain_mask)."""
    N = len(d["shot"])
    pos = {nm: j for j, nm in enumerate(C.names)}
    fidx = np.empty(N, np.int64)
    grow = np.empty(N, np.int64)
    yf = np.full(N, np.nan)
    hold = np.full(N, np.nan)
    age = np.full(N, np.nan)
    for u in np.unique(d["shot"]):
        sel = np.flatnonzero(d["shot"] == u)
        nm = valid[int(u)]
        arr = C.grid[nm]
        gi = B45.compact_index(arr)
        rows = d["row"][sel]
        h, _ = B45.hold_compacted(arr, gi, rows)
        g = gi[rows]
        fidx[sel] = pos[nm]
        grow[sel] = g
        yf[sel] = arr[g, 1].astype(np.float64)
        hold[sel] = h
        prev = gi[np.clip(rows - 1, 0, len(gi) - 1)]
        tt = arr[:, 0].astype(np.float64)
        age[sel] = np.where(rows >= 1, (tt[g] - tt[prev]) * 1000.0, np.nan)
    cls = np.array([F[f"cls_{a}"][b] for a, b in zip(fidx, grow)], np.int8)
    cls_alt = np.array([F[f"clsalt_{a}"][b] for a, b in zip(fidx, grow)], np.int8)
    blip = np.array([F[f"blip13_{a}"][b] for a, b in zip(fidx, grow)], bool)
    plain = B41.plain_mask(C, {"y": yf, "hold": hold, "file": fidx, "row": grow})
    return {"fidx": fidx, "grow": grow, "y_file": yf, "hold_val": hold, "age_ms": age, "cls": cls,
            "cls_alt": cls_alt, "blip13": blip, "plain": plain}


def stage_rows():
    C = ctx41()
    valid = valid_names(C)
    F = load_files()
    for r in REPEATS:
        for k in range(K):
            out = CACHE / f"rows_r{r}_f{k}.npz"
            if out.exists():
                continue
            c = load_run(B39.pooled_run_dir("v2", r, k))
            t = load_run(B39.pooled_run_dir("v2h", r, k))
            same = (len(c["shot"]) == len(t["shot"]) and np.array_equal(c["shot"], t["shot"])
                    and np.array_equal(c["row"], t["row"]))
            refs_agree = same and all(np.allclose(c[q], t[q], rtol=1e-6, atol=1e-6, equal_nan=True)
                                      for q in ("y", "hold", "gp", "off", "pchip"))
            loc = locate(C, valid, F, c)
            man = json.loads((B27.split_dir(r, k) / "split_manifest.json").read_text(encoding="utf-8"))
            np.savez_compressed(out, shot=c["shot"], row=c["row"], se_c=c["se"], se_t=t["se"], y=c["y"],
                                se_hold=c["hold"], se_gp=c["gp"], se_off=c["off"], se_pchip=c["pchip"],
                                dt_ms=c["dt_ms"], same_rows=np.array(same), refs_agree=np.array(refs_agree),
                                test_files=np.array(man["test_files"]), **loc)
            log(f"rows r{r} f{k}: {len(c['shot'])} rows, same rows {same}, refs agree {refs_agree}")


def load_rows(r, k):
    z = np.load(CACHE / f"rows_r{r}_f{k}.npz")
    return {kk: z[kk] for kk in z.files}


# ------------------------------------------------------------------------------------------ stage prop
def join_p(C, P, fidx, grow):
    """B.50 join (copied from run_b50.split): p at (file, grid row); window = position >= 10 in the segment."""
    key = {(int(a), int(b)): j for j, (a, b) in enumerate(zip(P["fidx"], P["row"]))}
    jj = np.array([key.get((int(a), int(b)), -1) for a, b in zip(fidx, grow)])
    has = jj >= 0
    pos = np.array([r - np.flatnonzero(C.seg[a] == C.seg[a][r])[0] for a, r in zip(fidx, grow)])
    win = pos >= 10
    p = np.full(len(fidx), np.nan)
    p[has] = P["p"][jj[has]]
    chk = {"rows": int(len(jj)), "with_window": int(win.sum()), "with_window_without_p": int((win & ~has).sum()),
           "without_window_with_p": int((~win & has).sum()), "joined_rows_marked_unusable": int(P["u"][jj[has]].sum())}
    chk["passes"] = bool(chk["with_window_without_p"] == 0 and chk["without_window_with_p"] == 0
                         and chk["joined_rows_marked_unusable"] == 0)
    return p, chk


def stage_prop():
    PROP.mkdir(parents=True, exist_ok=True)
    B50.CACHE = PROP                                  # run_b50.propensity caches p_s{key}.npz here
    C = ctx37()
    keys = ["42"] + [f"r{r}f{k}" for r in REPEATS for k in range(K)]
    for key in keys:
        out = PROP / f"join_{key}.npz"
        if out.exists():
            continue
        t0 = _time.time()
        P = B50.propensity(C, int(key) if key == "42" else key)
        from sklearn.metrics import roc_auc_score
        auc = float(roc_auc_score(P["u"], P["p"]))
        if key == "42":
            tab = B50.load_tab(42)
            sc = (tab["k"] == 1) & np.all([np.isfinite(tab[e]) for e in (*B50.EST, "y")], axis=0)
            fidx, grow = tab["file"][sc].astype(np.int64), tab["row"][sc].astype(np.int64)
        else:
            r, k = key[1:].split("f")
            z = load_rows(int(r), int(k))
            fidx, grow = z["fidx"], z["grow"]
        p, chk = join_p(C, P, fidx, grow)
        chk.update({"auc": auc, "fit_rows": int(P["fit_rows"]), "test_rows": int(len(P["u"]))})
        np.savez_compressed(out, p=p, u_p=P["p"][P["u"] == 1], chk=json.dumps(chk))
        log(f"prop {key}: AUC {auc:.4f}, join {chk['passes']} ({_time.time() - t0:.0f}s)")


def load_prop(key):
    z = np.load(PROP / f"join_{key}.npz")
    return z["p"], z["u_p"], json.loads(str(z["chk"]))


# ------------------------------------------------------------------------------------------ pooled data
def repeat_data(r):
    """The five folds of a repeat concatenated."""
    parts = [load_rows(r, k) for k in range(K)]
    d = {kk: np.concatenate([p[kk] for p in parts]) for kk in parts[0] if parts[0][kk].ndim == 1 and kk != "test_files"}
    pr = [load_prop(f"r{r}f{k}") for k in range(K)]
    d["p"] = np.concatenate([x[0] for x in pr])
    d["u_p"] = np.concatenate([x[1] for x in pr])
    d["fold"] = np.concatenate([np.full(len(p["shot"]), k) for k, p in enumerate(parts)])
    d["se"] = {"c": d["se_c"], "t": d["se_t"], "hold": d["se_hold"], "gp": d["se_gp"], "off": d["se_off"]}
    u, f = np.unique(d["fidx"], return_inverse=True)
    d["u"], d["f"], d["n"] = u, f, len(u)
    return d


# ------------------------------------------------------------------------------------------ two-level bootstrap
def hier(num, den, fn, nboot, rng):
    """Two-level bootstrap of the pooled readout (analyze_b39.arm_p.hier, generalised to any ratio of
    per-discharge sums): a repeat is drawn, then its discharges are resampled with replacement.
    num, den: dict repeat -> per-discharge sums.  Point = mean over repeats."""
    draws = rng.choice(np.array(REPEATS), size=nboot)
    o = np.empty(nboot)
    for r in REPEATS:
        sel = np.flatnonzero(draws == r)
        m, bb = num[r], den[r]
        idx = rng.integers(0, len(m), size=(len(sel), len(m)))
        with np.errstate(invalid="ignore", divide="ignore"):
            o[sel] = fn(m[idx].sum(1), bb[idx].sum(1))
    per = {str(r): float(fn(num[r].sum(), den[r].sum())) for r in REPEATS}
    return [float(np.mean(list(per.values()))), *ci(o)], per


SKILL = lambda a, b: 1.0 - a / b          # noqa: E731
RATIO = lambda a, b: a / b                # noqa: E731


def hier_quant(vals, nboot, rng, qs, pairs):
    """Ratios of weighted quantiles under the same two-level resampling (row weights = discharge multiplicity;
    run_b41.wquant).  vals: dict repeat -> (dict est -> row values, f, n).  pairs: list of (a, b, q)."""
    draws = rng.choice(np.array(REPEATS), size=nboot)
    out = {p: np.empty(nboot) for p in pairs}
    per = {p: {} for p in pairs}
    for r in REPEATS:
        v, f, n = vals[r]
        sel = np.flatnonzero(draws == r)
        idx = rng.integers(0, n, size=(len(sel), n))
        W = np.zeros((len(sel), n))
        for j in range(len(sel)):
            W[j] = np.bincount(idx[j], minlength=n)
        qv = {e: B41.wquant(v[e], f, W, list(qs)) for e in {x for p in pairs for x in p[:2]}}
        pt = {e: np.percentile(v[e], [q * 100 for q in qs]) for e in qv}
        for (a, b, q) in pairs:
            j = list(qs).index(q)
            out[(a, b, q)][sel] = qv[a][:, j] / qv[b][:, j]
            per[(a, b, q)][str(r)] = float(pt[a][j] / pt[b][j])
    return {p: [float(np.mean(list(per[p].values()))), *ci(out[p])] for p in pairs}, per


# ------------------------------------------------------------------------------------------ checks
def stage_checks():
    C = ctx41()
    valid = valid_names(C)
    res = {}
    # (a) P1 and P2 of B.39 from the stored errors (point to three decimals; interval with analyze_b39's procedure)
    b39 = json.loads((PAPER / "b39_primary.json").read_text(encoding="utf-8"))["P"]
    per = {}
    for r in REPEATS:
        parts = [load_rows(r, k) for k in range(K)]
        shot = np.concatenate([p["shot"] for p in parts])
        u, inv = np.unique(shot, return_inverse=True)
        per[r] = {q: np.bincount(inv, np.concatenate([p[q] for p in parts]), len(u)) for q in ("se_c", "se_t", "se_gp")}
    rng = np.random.default_rng(SEED_P39)
    p1, p1r = hier({r: per[r]["se_c"] for r in REPEATS}, {r: per[r]["se_gp"] for r in REPEATS}, SKILL, B_SKILL, rng)
    p2, p2r = hier({r: per[r]["se_c"] for r in REPEATS}, {r: per[r]["se_t"] for r in REPEATS}, SKILL, B_SKILL, rng)
    a = {"P1_mine": p1, "P1_b39": b39["P1_causal_over_gp"], "P2_mine": p2, "P2_b39": b39["P2_causal_over_twin"],
         "P1_per_repeat": p1r, "P2_per_repeat": p2r}
    a["point_equal_3dp"] = bool(abs(p1[0] - a["P1_b39"][0]) < TOL3 and abs(p2[0] - a["P2_b39"][0]) < TOL3)
    a["interval_equal_3dp"] = bool(all(abs(x - y) < TOL3 for x, y in zip(p1 + p2, a["P1_b39"] + a["P2_b39"])))
    a["passes"] = a["point_equal_3dp"]
    res["a"] = a
    log(f"(a) P1 {p1} vs {a['P1_b39']}; P2 {p2} vs {a['P2_b39']}")
    # (b) every discharge in exactly one fold of each repeat; (c) same rows; (d) value in the file
    b, c, d = {}, {}, {}
    allv = set(valid)
    for r in REPEATS:
        cnt = {}
        for k in range(K):
            z = load_rows(r, k)
            test = set(z["test_files"].tolist())
            names = {valid[int(s)] for s in np.unique(z["shot"])}
            for nm in names:
                cnt[nm] = cnt.get(nm, 0) + 1
            outside = sorted(names - test)
            c[f"r{r}f{k}"] = {"same_rows": bool(z["same_rows"]), "reference_errors_agree": bool(z["refs_agree"]),
                              "rows": int(len(z["shot"]))}
            dy = np.abs(z["y_file"] - z["y"])
            hold_dev = np.abs(np.abs(z["hold_val"] - z["y_file"]) - np.sqrt(z["se_hold"]))
            d[f"r{r}f{k}"] = {"rows": int(len(dy)), "max_abs_diff_eV": float(np.nanmax(dy)),
                              "nan_in_file": int(np.isnan(z["y_file"]).sum()),
                              "rows_above_tol": int((~(dy <= Y_TOL)).sum()),
                              "rows_first_or_last_grid_row": int((z["cls"] == 0).sum()),
                              "hold_rows_within_0.5_eV_share": float(np.mean(hold_dev <= HOLD_TOL)),
                              "hold_max_dev_eV": float(np.nanmax(hold_dev)),
                              "test_discharges_with_rows": len(names), "discharges_outside_fold_test_files": outside,
                              "test_files_in_manifest": int(len(test))}
        once = all(v == 1 for v in cnt.values())
        tests = [json.loads((B27.split_dir(r, k) / "split_manifest.json").read_text(encoding="utf-8"))
                 for k in range(K)]
        entries = [nm for m in tests for nm in m["test_files"]]
        b[str(r)] = {"discharges_with_rows": len(cnt), "each_in_exactly_one_fold": bool(once),
                     "valid_files": len(allv), "valid_without_rows": sorted(allv - set(cnt)),
                     "manifest_universe": int(tests[0]["universe"]), "manifest_test_entries": len(entries),
                     "manifest_unique_test_files": len(set(entries)),
                     "universe_files_without_scored_rows": sorted(set(entries) - set(cnt)),
                     "valid_files_outside_universe": sorted(allv - set(entries))}
        # Addendum 1: judged on the fold manifests (639 discharges, each in exactly one test fold) and no discharge
        # with scored rows in two folds; the resampled units are the discharges with scored rows
        b[str(r)]["passes"] = bool(once and len(entries) == 639 and len(set(entries)) == 639
                                   and int(tests[0]["universe"]) == 639 and set(cnt) <= set(entries))
    res["b"] = {"repeats": b, "passes": bool(all(v["passes"] for v in b.values()))}
    res["c"] = {"folds": c, "passes": bool(all(v["same_rows"] and v["reference_errors_agree"] for v in c.values()))}
    res["d"] = {"folds": d, "passes": bool(all(v["rows_above_tol"] == 0 and not v["discharges_outside_fold_test_files"]
                                               and v["rows_first_or_last_grid_row"] == 0 for v in d.values()))}
    log(f"(b) {res['b']['passes']} (c) {res['c']['passes']} (d) {res['d']['passes']}")
    res["e"] = check_e(C, valid)
    log(f"(e) {res['e']['passes']}")
    res["passes"] = bool(all(res[x]["passes"] for x in "abcde"))
    (CACHE / "checks.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    if not res["passes"]:
        raise SystemExit("FATAL: a check fails; the readout is not computed (data/.b54/checks.json)")
    log("all checks pass")


def check_e(C, valid):
    """The flag, class, third, blip-row and risk code applied to split 42 and compared with the recorded numbers."""
    F = load_files()
    b48 = json.loads((PAPER / "b48_reread.json").read_text(encoding="utf-8"))
    b45 = json.loads((PAPER / "b45_ablation_rows.json").read_text(encoding="utf-8"))
    b50 = json.loads((PAPER / "b50_missing_like.json").read_text(encoding="utf-8"))
    b52 = json.loads((PAPER / "b52_grading_readouts.json").read_text(encoding="utf-8"))
    z = np.load(DATA / ".b48" / "rows_cut_s42.npz")
    tab = {cc.split("__")[1]: z[cc] for cc in z.files if cc.startswith("CES_TI__")}
    ok = B41.population(tab)
    fidx, grow = tab["file"][ok].astype(np.int64), tab["row"][ok].astype(np.int64)
    y = tab["y"][ok].astype(np.float64)
    se = {e: (tab[e][ok].astype(np.float64) - y) ** 2 for e in ("causal", "hist80", "gp", "hold")}
    plain = B41.plain_mask(C, {"y": y, "hold": tab["hold"][ok].astype(np.float64), "file": fidx, "row": grow})
    G = b48["b41"]["G"]["42"]
    out = {}
    v = float(1 - plain.mean())
    out["flagged_share"] = {"mine": v, "recorded": G["flagged"]["row_share"][0], "where": "b48_reread.json b41.G.42.flagged"}
    v = float(1 - se["causal"][plain].sum() / se["hist80"][plain].sum())
    out["increment_rows_not_flagged"] = {"mine": v, "recorded": G["plain"]["skill_causal_vs_twin"][0],
                                         "where": "b48_reread.json b41.G.42.plain.skill_causal_vs_twin"}
    # class code on the stored rows that B.45 read (run .b39_v2_cut_s42_i42), through the pooled locate
    d = load_run(DATA / ".b39_v2_cut_s42_i42")
    loc = locate(C, valid, F, d)
    out["quiet_share"] = {"mine": float((loc["cls"] == 6).mean()),
                          "recorded": b45["splits"]["42"]["groups"]["6"]["row_share"][0],
                          "where": "b45_ablation_rows.json splits.42.groups.6.row_share (stored rows of .b39_v2_cut_s42_i42)"}
    # risk code: propensity refitted here for split 42, joined to the B.48 rows, top fifth (run_b50.split)
    p, u_p, pchk = load_prop("42")
    sc = (tab["k"] == 1) & np.all([np.isfinite(tab[e]) for e in (*B50.EST, "y")], axis=0)
    has = np.isfinite(p)
    ys = tab["y"][sc].astype(np.float64)[has]
    ses = {e: (tab[e][sc].astype(np.float64)[has] - ys) ** 2 for e in ("causal", "hist80", "gp")}
    ph = p[has]
    edges = np.quantile(ph, [0.2, 0.4, 0.6, 0.8])
    top = np.searchsorted(edges, ph, side="left") == 4
    out["increment_top_fifth_risk"] = {"mine": float(1 - ses["causal"][top].sum() / ses["hist80"][top].sum()),
                                       "recorded": b50["splits"]["42"]["fifths"][4]["skill"]["hist80"][0],
                                       "where": "b50_missing_like.json splits.42.fifths[4].skill.hist80"}
    for kk, vv in out.items():
        vv["equal_3dp"] = bool(abs(vv["mine"] - vv["recorded"]) < TOL3)
    res = {"registered": out, "passes": bool(all(vv["equal_3dp"] for vv in out.values()))}
    # further comparisons on split 42 (reported beside the registered four)
    ex = {"propensity_auc": {"mine": pchk["auc"], "recorded": b50["splits"]["42"]["checks"]["a"]["auc"]},
          "propensity_join": pchk,
          "skill_top_fifth_risk_over_gp": {"mine": float(1 - ses["causal"][top].sum() / ses["gp"][top].sum()),
                                           "recorded": b50["splits"]["42"]["fifths"][4]["skill"]["gp"][0]}}
    blip = np.array([F[f"blip13_{a}"][b] for a, b in zip(fidx, grow)], bool)
    ex["blip_rows_1_3_increment"] = {"mine": float(1 - se["causal"][blip].sum() / se["hist80"][blip].sum()),
                                     "recorded": b48["b41"]["B"]["42"]["skill_causal_vs_twin_rows_1-3"][0],
                                     "rows_mine": int(blip.sum()), "rows_recorded": b48["b41"]["B"]["42"]["rows_1-3"]}
    uf, f = np.unique(fidx, return_inverse=True)
    sums = {e: np.bincount(f, se[e], len(uf)) for e in se}
    parts = B44.split_thirds(F["median_ti"][uf], F["shot"][uf])
    th = b48["b44"]["splits"]["42"]["strata"]["median_ti"]["thirds"]
    for tn, idx in zip(B44.THIRDS, parts):
        for ref in ("hist80", "gp"):
            ex[f"third_{tn}_over_{ref}"] = {"mine": float(1 - sums["causal"][idx].sum() / sums[ref][idx].sum()),
                                            "recorded": th[tn]["skill_causal_vs"][ref][0]}
    sc_all = (tab["k"] == 1) & np.all([np.isfinite(tab[e]) for e in ("causal", "hist80", "gp", "hold", "y")], axis=0)
    ys2 = tab["y"][sc_all].astype(np.float64)
    a10 = np.round(tab["age_ms"][sc_all].astype(np.float64) / 10.0)
    m = a10 >= 2
    ex["age_20ms_or_more_over_gp"] = {
        "mine": float(1 - ((tab["causal"][sc_all][m] - ys2[m]) ** 2).sum() / ((tab["gp"][sc_all][m] - ys2[m]) ** 2).sum()),
        "recorded": b52["splits"]["42"]["G"]["20 ms or more"]["over_gp"][0]}
    fi2 = tab["file"][sc_all].astype(np.int64)
    sh = per_discharge(fi2, {"c": (tab["causal"][sc_all] - ys2) ** 2, "gp": (tab["gp"][sc_all] - ys2) ** 2})
    ex["share_lower_mse_over_gp"] = {"mine": sh["share"], "recorded": b52["splits"]["42"]["D"]["estimator"]["gp"]["share_lower_mse"][0]}
    ex["top10_share_over_gp"] = {"mine": sh["top10"], "recorded": b52["splits"]["42"]["D"]["estimator"]["gp"]["top10_share_of_reduction"]}
    for kk, vv in ex.items():
        if isinstance(vv, dict) and "mine" in vv:
            vv["equal_3dp"] = bool(abs(vv["mine"] - vv["recorded"]) < TOL3)
    res["further"] = ex
    return res


def per_discharge(fi, se):
    """run_b52.part_d (point values) for the estimator against one reference: eligible = at least 50 rows."""
    u, ff = np.unique(fi, return_inverse=True)
    cnt = np.bincount(ff, minlength=len(u))
    mse = {e: np.bincount(ff, se[e], len(u)) / cnt for e in se}
    el = cnt >= MIN_ROWS_D
    red = (mse["gp"] - mse["c"]) * cnt
    top = np.sort(red)[::-1][:10]
    return {"share": float((mse["c"][el] < mse["gp"][el]).mean()), "top10": float(top.sum() / red.sum())}


# ------------------------------------------------------------------------------------------ readout
def stage_readout():
    chk = json.loads((CACHE / "checks.json").read_text(encoding="utf-8"))
    if not chk["passes"]:
        raise SystemExit("FATAL: checks do not pass; no readout")
    F = load_files()
    D = {r: repeat_data(r) for r in REPEATS}
    log("repeats loaded: " + ", ".join(f"r{r} {len(D[r]['y']):,} rows / {D[r]['n']} discharges" for r in REPEATS))
    out = {"registration": REG, "seed": SEED, "n_boot_skill": B_SKILL, "n_boot_ratio": B_RATIO, "checks": chk}
    st, rep = {}, {}

    def dsum(r, x, m=None):
        d = D[r]
        mm = np.ones(len(x), bool) if m is None else m
        return np.bincount(d["f"][mm], x[mm], d["n"])

    def skill_cell(name, mask_fn, refs=("t", "gp", "hold"), w_fn=None):
        """estimator over each reference on rows mask_fn(r) (weights w_fn(r)); 20,000 draws."""
        o = {}
        for ref in refs:
            num, den = {}, {}
            for r in REPEATS:
                m = mask_fn(r)
                w = np.ones(len(D[r]["y"])) if w_fn is None else w_fn(r)
                num[r] = dsum(r, w * D[r]["se"]["c"], m)
                den[r] = dsum(r, w * D[r]["se"][ref], m)
            o[ref], o[ref + "_per_repeat"] = hier(num, den, SKILL, B_SKILL, rng_for(f"{name}|{ref}"))
        o["rows_per_repeat"] = {str(r): int(mask_fn(r).sum()) for r in REPEATS}
        o["row_share"] = float(np.mean([mask_fn(r).mean() for r in REPEATS]))
        o["discharges_with_rows_mean"] = float(np.mean([len(np.unique(D[r]["f"][mask_fn(r)])) for r in REPEATS]))
        return o

    def share_cell(name, mask_fn, ref, base_fn=None):
        """share of the squared-error reduction of the estimator against ref held by the rows mask_fn(r), over the
        rows base_fn(r) (all rows if None); 2,000 draws."""
        num, den = {}, {}
        for r in REPEATS:
            red = D[r]["se"][ref] - D[r]["se"]["c"]
            base = np.ones(len(red), bool) if base_fn is None else base_fn(r)
            num[r] = dsum(r, red, mask_fn(r) & base)
            den[r] = dsum(r, red, base)
        v, per = hier(num, den, RATIO, B_RATIO, rng_for(f"share|{name}|{ref}"))
        return {"value": v, "per_repeat": per}

    allrows = lambda r: np.ones(len(D[r]["y"]), bool)                                 # noqa: E731

    # ---------------- overall (reported, for reference; equals P1/P2 points)
    rep["overall"] = skill_cell("overall", allrows, refs=("t", "gp", "hold", "off"))
    log("overall done")

    # ---------------- A1 and the median ratios
    vals = {}
    for r in REPEATS:
        vals[r] = ({e: np.sqrt(D[r]["se"][e]) for e in ("c", "t", "gp", "hold")}, D[r]["f"], D[r]["n"])
    pairs = [("c", "gp", 0.9), ("c", "hold", 0.9), ("c", "t", 0.9), ("c", "gp", 0.5), ("c", "hold", 0.5), ("c", "t", 0.5)]
    q, qper = hier_quant(vals, B_RATIO, rng_for("quantiles"), (0.5, 0.9), pairs)
    st["A1"] = {"value": q[("c", "gp", 0.9)], "per_repeat": qper[("c", "gp", 0.9)], "reference": 1.0, "side": "below"}
    rep["p90_ratio"] = {LAB[b]: {"value": q[("c", b, 0.9)], "per_repeat": qper[("c", b, 0.9)]} for b in ("gp", "hold", "t")}
    rep["median_ratio"] = {LAB[b]: {"value": q[("c", b, 0.5)], "per_repeat": qper[("c", b, 0.5)]} for b in ("gp", "hold", "t")}
    rep["abs_error_percentiles_eV_mean_over_repeats"] = {
        LAB[e]: {str(p): float(np.mean([np.percentile(np.sqrt(D[r]["se"][e]), p) for r in REPEATS])) for p in (50, 75, 90, 95, 99)}
        for e in ("c", "t", "gp", "hold", "off")}
    log(f"A1 {st['A1']['value']}")

    # ---------------- B: per discharge (run_b52.part_d on the pooled rows)
    pdisc = {}
    for r in REPEATS:
        d = D[r]
        cnt = np.bincount(d["f"], minlength=d["n"])
        mse = {e: np.bincount(d["f"], d["se"][e], d["n"]) / cnt for e in ("c", "t", "gp", "hold")}
        ab = {e: np.sqrt(d["se"][e]) for e in ("c", "t", "gp", "hold")}
        order = np.argsort(d["f"], kind="stable")
        bounds = np.searchsorted(d["f"][order], np.arange(d["n"] + 1))
        mae = {e: np.array([np.median(ab[e][order[bounds[j]:bounds[j + 1]]]) for j in range(d["n"])]) for e in ab}
        pdisc[r] = {"cnt": cnt, "mse": mse, "mae": mae, "el": cnt >= MIN_ROWS_D}
    for key, ref in (("B1", "gp"), ("B2", "t")):
        num = {r: (pdisc[r]["mse"]["c"] < pdisc[r]["mse"][ref])[pdisc[r]["el"]].astype(float) for r in REPEATS}
        den = {r: np.ones(int(pdisc[r]["el"].sum())) for r in REPEATS}
        v, per = hier(num, den, RATIO, B_RATIO, rng_for(f"{key}"))
        st[key] = {"value": v, "per_repeat": per, "reference": 0.5, "side": "above",
                   "eligible_discharges_per_repeat": {str(r): int(pdisc[r]["el"].sum()) for r in REPEATS},
                   "ties": {str(r): int((pdisc[r]["mse"]["c"] == pdisc[r]["mse"][ref])[pdisc[r]["el"]].sum()) for r in REPEATS}}
        num = {r: (pdisc[r]["mae"]["c"] < pdisc[r]["mae"][ref])[pdisc[r]["el"]].astype(float) for r in REPEATS}
        v, per = hier(num, den, RATIO, B_RATIO, rng_for(f"{key}|median_abs"))
        rep[f"share_lower_median_abs_over_{LAB[ref]}"] = {"value": v, "per_repeat": per}
    num = {r: (pdisc[r]["mse"]["c"] < pdisc[r]["mse"]["hold"])[pdisc[r]["el"]].astype(float) for r in REPEATS}
    den = {r: np.ones(int(pdisc[r]["el"].sum())) for r in REPEATS}
    rep["share_lower_mse_over_hold"] = dict(zip(("value", "per_repeat"), hier(num, den, RATIO, B_RATIO, rng_for("B|hold"))))
    # concentration (per repeat, mean over repeats) and the skill without the ten
    conc = {"top10_share_over_gp": {}, "top_twentieth_share_over_gp": {}, "twentieth_size": {},
            "top10_share_over_twin": {}}
    keep = {}
    for r in REPEATS:
        p_ = pdisc[r]
        red = (p_["mse"]["gp"] - p_["mse"]["c"]) * p_["cnt"]
        o = np.argsort(red)[::-1]
        n20 = int(round(D[r]["n"] / 20))
        conc["top10_share_over_gp"][str(r)] = float(red[o[:10]].sum() / red.sum())
        conc["top_twentieth_share_over_gp"][str(r)] = float(red[o[:n20]].sum() / red.sum())
        conc["twentieth_size"][str(r)] = n20
        redt = (p_["mse"]["t"] - p_["mse"]["c"]) * p_["cnt"]
        conc["top10_share_over_twin"][str(r)] = float(np.sort(redt)[::-1][:10].sum() / redt.sum())
        km = np.ones(D[r]["n"], bool)
        km[o[:10]] = False
        keep[r] = km
    rep["concentration"] = {k: {"mean": float(np.mean(list(v.values()))), "per_repeat": v} for k, v in conc.items()}
    rep["without_top10"] = skill_cell("without_top10", lambda r: keep[r][D[r]["f"]], refs=("gp", "t", "hold"))
    log(f"B1 {st['B1']['value']} B2 {st['B2']['value']}; top10 {rep['concentration']['top10_share_over_gp']['mean']:.3f}; "
        f"without top10 over GP {rep['without_top10']['gp']}")

    # ---------------- C: rows not flagged by the value rules
    plain = lambda r: D[r]["plain"].astype(bool)                                      # noqa: E731
    flagged = lambda r: ~D[r]["plain"].astype(bool)                                   # noqa: E731
    cp = skill_cell("plain", plain, refs=("t", "gp", "hold", "off"))
    cf = skill_cell("flagged", flagged, refs=("t", "gp", "hold", "off"))
    st["C1"] = {"value": cp["t"], "per_repeat": cp["t_per_repeat"], "reference": 0.0, "side": "above"}
    st["C2"] = {"value": cp["gp"], "per_repeat": cp["gp_per_repeat"], "reference": 0.0, "side": "above"}
    s_gp, s_t = share_cell("plain", plain, "gp"), share_cell("plain", plain, "t")
    st["C3"] = {**s_gp, "reference": 0.5, "side": "above"}
    st["C4"] = {**s_t, "reference": 0.5, "side": "above"}
    rep["plain_rows"], rep["flagged_rows"] = cp, cf
    log(f"C1 {cp['t']} C2 {cp['gp']} C3 {s_gp['value']} C4 {s_t['value']}")

    # ---------------- D: thirds of the discharges by median T_i (within a repeat, from all its discharges)
    thirds = {}
    for r in REPEATS:
        u = D[r]["u"]
        v = F["median_ti"][u]
        have = np.flatnonzero(np.isfinite(v))
        parts = [have[p] for p in B44.split_thirds(v[have], F["shot"][u][have])]
        lab = np.full(D[r]["n"], -1)
        for j, idx in enumerate(parts):
            lab[idx] = j
        thirds[r] = {"lab": lab, "edges": [float(v[parts[0]].max()), float(v[parts[1]].max())],
                     "excluded": int((~np.isfinite(v)).sum())}
    rep["thirds_edges_eV"] = {str(r): thirds[r]["edges"] for r in REPEATS}
    rep["thirds_excluded_discharges"] = {str(r): thirds[r]["excluded"] for r in REPEATS}
    dth = {}
    for j, tn in enumerate(B44.THIRDS):
        dth[tn] = skill_cell(f"third_{tn}", lambda r, j=j: thirds[r]["lab"][D[r]["f"]] == j)
    rep["thirds"] = dth
    st["D1"] = {"value": {tn: dth[tn]["gp"] for tn in B44.THIRDS}, "reference": 0.0, "side": "above", "all_thirds": True}
    st["D2"] = {"value": {tn: dth[tn]["t"] for tn in B44.THIRDS}, "reference": 0.0, "side": "above", "all_thirds": True}
    log("D " + "; ".join(f"{tn} gp {dth[tn]['gp']} t {dth[tn]['t']}" for tn in B44.THIRDS))

    # ---------------- E: age of the last value (run_b52 part G bins)
    a10 = {r: np.round(D[r]["age_ms"] / 10.0) for r in REPEATS}
    bins = {"10 ms": lambda r: a10[r] == 1, "20 to 50 ms": lambda r: (a10[r] >= 2) & (a10[r] <= 5),
            "above 50 ms": lambda r: a10[r] > 5, "20 ms or more": lambda r: a10[r] >= 2}
    rep["age"] = {b: skill_cell(f"age|{b}", fn) for b, fn in bins.items()}
    rep["age_unbinned_rows"] = {str(r): int((a10[r] < 1).sum()) for r in REPEATS}
    st["E1"] = {"value": rep["age"]["20 ms or more"]["gp"], "per_repeat": rep["age"]["20 ms or more"]["gp_per_repeat"],
                "reference": 0.0, "side": "above"}
    st["E2"] = {"value": rep["age"]["above 50 ms"]["gp"], "per_repeat": rep["age"]["above 50 ms"]["gp_per_repeat"],
                "reference": 0.0, "side": "above"}
    log(f"E1 {st['E1']['value']} E2 {st['E2']['value']}")

    # ---------------- F: risk of losing the value (B.50)
    win = {r: np.isfinite(D[r]["p"]) for r in REPEATS}
    fifth, wts, finfo = {}, {}, {}
    for r in REPEATS:
        p = D[r]["p"]
        ph = p[win[r]]
        edges = np.quantile(ph, [0.2, 0.4, 0.6, 0.8])
        q5 = np.full(len(p), -1)
        q5[win[r]] = np.searchsorted(edges, ph, side="left")
        fifth[r] = q5
        w = np.zeros(len(p))
        ww = ph / (1 - ph)
        clip = float(np.quantile(ww, 0.99))
        w[win[r]] = np.minimum(ww, clip)
        wts[r] = w
        qu = np.searchsorted(edges, D[r]["u_p"], side="left")
        wv = w[win[r]]
        finfo[str(r)] = {"edges": edges.tolist(), "clip_w": clip, "rows_clipped": int((ww > clip).sum()),
                         "rows_with_window": int(win[r].sum()), "rows_without_window": int((~win[r]).sum()),
                         "effective_rows": float(wv.sum() ** 2 / (wv ** 2).sum()),
                         "effective_share": float(wv.sum() ** 2 / (wv ** 2).sum() / len(wv)),
                         "share_of_unusable_in_top_fifth": float((qu == 4).mean()), "unusable_test_rows": int(len(qu)),
                         "top_fifth_median_age_ms": float(np.median(D[r]["age_ms"][q5 == 4])),
                         "top_fifth_median_ti_eV": float(np.median(D[r]["y"][q5 == 4]))}
    rep["risk_info"] = finfo
    rep["effective_rows"] = {"mean": float(np.mean([v["effective_rows"] for v in finfo.values()])),
                             "share_mean": float(np.mean([v["effective_share"] for v in finfo.values()]))}
    ftop = skill_cell("risk_top_fifth", lambda r: fifth[r] == 4)
    fw = skill_cell("risk_weighted", lambda r: win[r], w_fn=lambda r: wts[r])
    fwin = skill_cell("risk_window_unweighted", lambda r: win[r])
    rep["risk_top_fifth"], rep["risk_weighted"], rep["risk_window_rows_unweighted"] = ftop, fw, fwin
    st["F1"] = {"value": ftop["gp"], "per_repeat": ftop["gp_per_repeat"], "reference": 0.0, "side": "above"}
    st["F2"] = {"value": ftop["t"], "per_repeat": ftop["t_per_repeat"], "reference": 0.0, "side": "above"}
    st["F3"] = {"value": fw["gp"], "per_repeat": fw["gp_per_repeat"], "reference": 0.0, "side": "above"}
    st["F4"] = {"value": fw["t"], "per_repeat": fw["t_per_repeat"], "reference": 0.0, "side": "above"}
    log(f"F1 {ftop['gp']} F2 {ftop['t']} F3 {fw['gp']} F4 {fw['t']}")

    # ---------------- G: rows 1 to 3 after an absent frame
    g13 = skill_cell("blip13", lambda r: D[r]["blip13"].astype(bool))
    rep["blip_rows_1_3"] = g13
    st["G1"] = {"value": g13["t"], "per_repeat": g13["t_per_repeat"], "reference": 0.0, "side": "above"}
    log(f"G1 {g13['t']}")

    # ---------------- H: quiet rows (run_b45 class 6) and the other classes
    rep["classes"] = {}
    for c_ in range(1, 8):
        rep["classes"][str(c_)] = skill_cell(f"class{c_}", lambda r, c_=c_: D[r]["cls"] == c_,
                                             refs=("t", "gp", "hold") + (("off",) if c_ == 6 else ()))
    quiet = lambda r: D[r]["cls"] == 6                                                # noqa: E731
    st["H1"] = {"value": rep["classes"]["6"]["t"], "per_repeat": rep["classes"]["6"]["t_per_repeat"],
                "reference": 0.0, "side": "above"}
    st["H2"] = {**share_cell("quiet", quiet, "t"), "reference": 0.5, "side": "above"}
    rep["quiet_share_of_reduction_vs_gp"] = share_cell("quiet", quiet, "gp")
    log(f"H1 {st['H1']['value']} H2 {st['H2']['value']}")

    # ---------------- verdicts
    for k_, s_ in st.items():
        if s_.get("all_thirds"):
            s_["per_third_holds"] = {tn: bool(v[1] > s_["reference"]) for tn, v in s_["value"].items()}
            s_["holds"] = bool(all(s_["per_third_holds"].values()))
        elif s_["side"] == "above":
            s_["holds"] = bool(s_["value"][1] > s_["reference"])
        else:
            s_["holds"] = bool(s_["value"][2] < s_["reference"])
    out["statements"] = st
    out["reported"] = rep
    out["expectations"] = expectations(st, rep)
    out["choices"] = CHOICES
    out["runtime_readout_s"] = round(_time.time() - T0, 1)
    rt = CACHE / "runtime.json"
    out["runtime_stages_s"] = json.loads(rt.read_text()) if rt.exists() else {}
    OUT_JSON.write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    OUT_MD.write_text(render(out), encoding="utf-8", newline="\n")
    log("written " + str(OUT_MD))


# ------------------------------------------------------------------------------------------ expectations
EXPECT = [
    ("A1", "ratio 0.92 to 0.95, holds", lambda st, rp: st["A1"]["value"][0], 0.92, 0.95, True),
    ("A1-median", "median ratio to the causal GP 0.99 to 1.05", lambda st, rp: rp["median_ratio"]["causal GP"]["value"][0], 0.99, 1.05, None),
    ("B1", "0.62 to 0.70, holds", lambda st, rp: st["B1"]["value"][0], 0.62, 0.70, True),
    ("B2", "0.58 to 0.68, holds", lambda st, rp: st["B2"]["value"][0], 0.58, 0.68, True),
    ("top10", "share of the ten discharges with the largest reduction 0.25 to 0.45",
     lambda st, rp: rp["concentration"]["top10_share_over_gp"]["mean"], 0.25, 0.45, None),
    ("without10", "skill over the causal GP without them +0.04 to +0.07, interval above zero",
     lambda st, rp: rp["without_top10"]["gp"][0], 0.04, 0.07, "without10"),
    ("C1", "+0.03 to +0.07, holds", lambda st, rp: st["C1"]["value"][0], 0.03, 0.07, True),
    ("C2", "+0.03 to +0.07, holds", lambda st, rp: st["C2"]["value"][0], 0.03, 0.07, True),
    ("C3", "share 0.35 to 0.55, does not hold", lambda st, rp: st["C3"]["value"][0], 0.35, 0.55, False),
    ("C4", "0.60 to 0.80, holds", lambda st, rp: st["C4"]["value"][0], 0.60, 0.80, True),
    ("D1", "+0.05 to +0.12 in each third, holds", "D1", 0.05, 0.12, True),
    ("D2", "+0.02 to +0.08, holds in the hottest and one other third, the statement does not hold", "D2", 0.02, 0.08, False),
    ("E1", "+0.06 to +0.11, holds", lambda st, rp: st["E1"]["value"][0], 0.06, 0.11, True),
    ("E2", "-0.03 to +0.08, does not hold", lambda st, rp: st["E2"]["value"][0], -0.03, 0.08, False),
    ("F1", "+0.07 to +0.12, holds", lambda st, rp: st["F1"]["value"][0], 0.07, 0.12, True),
    ("F2", "+0.04 to +0.08, holds", lambda st, rp: st["F2"]["value"][0], 0.04, 0.08, True),
    ("F3", "+0.03 to +0.10, holds", lambda st, rp: st["F3"]["value"][0], 0.03, 0.10, True),
    ("F4", "+0.01 to +0.06, does not hold", lambda st, rp: st["F4"]["value"][0], 0.01, 0.06, False),
    ("G1", "+0.15 to +0.25, holds", lambda st, rp: st["G1"]["value"][0], 0.15, 0.25, True),
    ("H1", "+0.02 to +0.06, holds", lambda st, rp: st["H1"]["value"][0], 0.02, 0.06, True),
    ("H2", "0.55 to 0.80, holds", lambda st, rp: st["H2"]["value"][0], 0.55, 0.80, True),
]


def expectations(st, rp):
    E = []
    for key, text, fn, lo, hi, verdict in EXPECT:
        if fn in ("D1", "D2"):
            vals = {tn: st[fn]["value"][tn][0] for tn in B44.THIRDS}
            inside = all(lo <= v <= hi for v in vals.values())
            obs = ", ".join(f"{tn} {v:+.3f}" for tn, v in vals.items())
            vok = st[fn]["holds"] == verdict
            if fn == "D2":
                ph = st["D2"]["per_third_holds"]
                pattern = ph["high"] and sum(ph.values()) == 2
                obs += f"; holds in: {', '.join(t for t, h in ph.items() if h) or 'none'}"
                vok = vok and pattern
            E.append({"key": key, "expectation": text, "observed": obs + f"; statement {'holds' if st[fn]['holds'] else 'does not hold'}",
                      "range_met": bool(inside), "verdict_met": bool(vok)})
            continue
        v = fn(st, rp)
        if verdict == "without10":
            ok = rp["without_top10"]["gp"][1] > 0
            obs = f"{v:+.3f} [{rp['without_top10']['gp'][1]:+.3f}, {rp['without_top10']['gp'][2]:+.3f}]"
            E.append({"key": key, "expectation": text, "observed": obs, "range_met": bool(lo <= v <= hi), "verdict_met": bool(ok)})
            continue
        obs = f"{v:+.3f}"
        vm = None
        if verdict is not None:
            vm = bool(st[key]["holds"] == verdict)
            obs += f"; {'holds' if st[key]['holds'] else 'does not hold'}"
        E.append({"key": key, "expectation": text, "observed": obs, "range_met": bool(lo <= v <= hi), "verdict_met": vm})
    return E


CHOICES = [
    "Rows: every T_i row of `comparison_errors_test.npz` of the 80 pooled runs; the five folds of a repeat are "
    "concatenated. The squared errors of the estimator, the twin, hold (`se_persistence`), the causal GP "
    "(`se_gp_causal`, the eval_seq computation) and the offline GP (`se_gp`) are the stored ones.",
    "Location: `CES_TI_shot` indexes `dataset.valid_files` of eval_seq, i.e. the sorted grid files with at least one "
    "finite target (the convention of `run_b45.main`; it is a global file index, not an index within the fold); "
    "`CES_TI_row` is mapped to the grid row with `run_b45.compact_index`. The last visible value and its age: "
    "`run_b45.hold_compacted` (T_i of the previous kept row in the same block) and the time between the two grid rows.",
    "Discharge unit of every interval: the data file (grid file index).",
    "Value-rule flags: `run_b41.plain_mask` (the four rules of sensitivity_b30 as re-read by B.48) called on the "
    "located rows with y = the value in the file and hold = the value located above. The registration names "
    "`b42_delay/run_b42.py`; the flags read by B.48 (statements G1, G2 there) are those of `run_b41.plain_mask`, and "
    "B.42 contributes the definition of the last available value (`hold` = last earlier measured value of the block), "
    "which is what is located here.",
    "Classes: `run_b45.Classes.of_file` unchanged (class 6 = quiet; group 7 = departure not defined, in no "
    "statement); shares of the reduction are over all rows, group 7 included (as run_b45).",
    "Rows 1 to 3 after an absent frame: the event rule and row walk of `run_b41.part_b`, copied unchanged; a row "
    "enters once.",
    "Thirds: `run_b44.discharge_quantities` (median measured T_i of the file) and `run_b44.split_thirds`, within each "
    "repeat over its discharges.",
    "Risk: `run_b50.propensity` called unchanged per fold (its cache pointed at data/.b54/prop) through a "
    "`run_b37.Ctx` whose manifest returns the fold manifest. It fits on every file that is not a test file of the "
    "fold (train and validation files), as run_b50 does on the four splits; the registration's 'training discharges' "
    "is read as that. Rows without a ten-row window (rows 2 to 10 of a segment) have no p and are outside F (B.50 "
    "Addendum 1). Fifth edges, the 99th-percentile clip of the odds and the effective number of rows are computed "
    "within a repeat over its rows with a p (as within a split in B.50); the p of each row comes from the model of "
    "its own fold.",
    "Per-discharge shares: `run_b52.part_d` (at least 50 scored rows; mean squared error per discharge; ties count as "
    "not lower). Concentration: per repeat over all its discharges (as part_d), mean over repeats; the twentieth = "
    "round(discharges / 20) = 32 discharges. 'Those ten removed': the ten discharges with the largest reduction "
    "against the causal GP of each repeat are removed for both skills (over the causal GP and over the twin); the "
    "interval resamples the remaining discharges.",
    "Age bins: `run_b52` part G (age rounded to 10 ms: 20 ms or more = rounded age >= 2; older than 50 ms = > 5).",
    "Intervals: the two-level bootstrap of `analyze_b39.arm_p` (`hier`: a repeat drawn uniformly, then its "
    "discharges resampled with replacement), generalised to any ratio of per-discharge sums; 20,000 draws for "
    "skills, 2,000 for percentile ratios and shares (the per-discharge shares of B1, B2 included). Percentile ratios "
    "use `run_b41.wquant` with the discharge multiplicities as row weights. Point = mean over the eight repeats of "
    "the per-repeat value. One generator per quantity: numpy default_rng([20261154, crc32(name)]).",
    "A statement holds when its interval excludes the reference on the stated side; D1 and D2 when that is so in "
    "each of the three thirds.",
    "Check (a): P1 and P2 recomputed from the cached stored errors with analyze_b39's procedure and its generator "
    "(default_rng(2728), P1 then P2); the point must equal to three decimals (the interval is reported beside it).",
    "Check (b) (Addendum 1): by the fold manifests each of the 639 discharges is in exactly one test fold of every "
    "repeat, and no discharge has scored rows in two folds of a repeat (from the stored discharge indices); the "
    "resampled units are the discharges with scored rows (633 to 635 per repeat). Check (c): stored discharge and row indices identical between the "
    "estimator's and the twin's run of a fold, and their stored reference errors equal. Check (d): the grid T_i at "
    "the located row equals `CES_TI_y_true` within 0.01 eV on every row, no row is the first or last row of its "
    "file, every discharge is a test file of its fold; the hold located in the file is compared with "
    "sqrt(`se_persistence`) and reported.",
    "Check (e): flags on the k = 1 rows of the B.48 table of split 42 (population of run_b41) against "
    "b48_reread.json (flagged share; increment over the twin in the rows not flagged); classes through the pooled "
    "location code on the stored rows of `.b39_v2_cut_s42_i42`, which are the rows B.45 read (share of the quiet "
    "rows against b45_ablation_rows.json; no quiet share is recorded on the B.48 table rows); risk: propensity "
    "refitted here for split 42 and the top fifth of the B.48 rows against b50_missing_like.json (increment over the "
    "twin). Third, blip-row, age and per-discharge code are compared on split 42 as well and reported.",
]


# ------------------------------------------------------------------------------------------ render
def s3(x):
    return f"{x[0]:+.3f} [{x[1]:+.3f}, {x[2]:+.3f}]"


def f3(x):
    return f"{x[0]:.3f} [{x[1]:.3f}, {x[2]:.3f}]"


def hs(h):
    return "holds" if h else "does not hold"


def excl(x, ref):
    return f"excl. {ref:g}" if not (x[1] <= ref <= x[2]) else f"incl. {ref:g}"


STEXT = {
    "A1": "90th percentile of the absolute error, estimator over causal GP; interval below 1",
    "B1": "share of discharges (>= 50 rows) where the estimator has the lower MSE than the causal GP; above 0.5",
    "B2": "the same against the twin; above 0.5",
    "C1": "rows not flagged by the value rules: estimator over the twin; above 0",
    "C2": "rows not flagged: estimator over the causal GP; above 0",
    "C3": "rows not flagged hold more than half of the reduction against the causal GP; above 0.5",
    "C4": "the same against the twin; above 0.5",
    "D1": "estimator over the causal GP, interval above 0 in each third by median T_i",
    "D2": "the same over the twin",
    "E1": "last value 20 ms old or older: estimator over the causal GP; above 0",
    "E2": "last value older than 50 ms: over the causal GP; above 0",
    "F1": "top fifth by risk of losing the value: over the causal GP; above 0",
    "F2": "top fifth by risk: over the twin; above 0",
    "F3": "rows weighted by the odds of that risk: over the causal GP; above 0",
    "F4": "odds-weighted: over the twin; above 0",
    "G1": "rows 1 to 3 after an absent frame: over the twin; above 0",
    "H1": "quiet rows: over the twin; above 0",
    "H2": "quiet rows hold more than half of the reduction against the twin; above 0.5",
}
GROUPS = [("A", ["A1"]), ("B", ["B1", "B2"]), ("C", ["C1", "C2", "C3", "C4"]), ("D", ["D1", "D2"]),
          ("E", ["E1", "E2"]), ("F", ["F1", "F2", "F3", "F4"]), ("G", ["G1"]), ("H", ["H1", "H2"])]
CLAB = {"1": "1 after an absent frame", "2": "2 low beam emission", "3": "3 upward departure",
        "4": "4 downward departure", "5": "5 after a departure", "6": "6 quiet", "7": "7 departure not defined"}


def render(o):
    st, rp, ck = o["statements"], o["reported"], o["checks"]
    fm = lambda k, x: (f3(x) if k in ("A1", "B1", "B2", "C3", "C4", "H2") else s3(x))   # noqa: E731
    L = [f"# B.54 -- the row-level readouts on the rows of the pooled cross-validation (registered: {REG})", "",
         "CPU, nothing trained (except the propensity model of B.37 F, refitted per fold), no network run. T_i, cut "
         "population, the 80 pooled runs of B.39 arm P (repeats r2 to r9, five folds each): the estimator "
         "`.b39p_v2_cut_r{r}_f{k}` and the history-only twin `.b39p_v2h_cut_r{r}_f{k}`; the five folds of a repeat "
         "concatenated (639 discharges partitioned per repeat; the resampled units are the discharges with scored "
         "rows, 633 to 635 per repeat, Addendum 1). Skill of the estimator over R = 1 - SSE_est / SSE_R on the same "
         "rows. Intervals: 95 %, two-level bootstrap of the pooled readout (a repeat drawn, then its discharges "
         "resampled); 20,000 draws for skills, 2,000 for percentile ratios and shares. Point = mean over the eight "
         "repeats. A statement holds if its interval excludes the reference on the stated side.", "",
         "## Checks before the readout", ""]
    a = ck["a"]
    L += [f"- **(a)** P1 and P2 of B.39 from the stored errors: P1 {s3(a['P1_mine'])} vs recorded {s3(a['P1_b39'])}; "
          f"P2 {s3(a['P2_mine'])} vs recorded {s3(a['P2_b39'])}. Points equal to three decimals: "
          f"{'yes' if a['point_equal_3dp'] else 'NO'}; intervals equal to three decimals: "
          f"{'yes' if a['interval_equal_3dp'] else 'no'} -> **{'passes' if a['passes'] else 'FAILS'}**."]
    b = ck["b"]["repeats"]
    nomiss = sorted({x for v in b.values() for x in v["universe_files_without_scored_rows"]})
    L += ["- **(b)** (as fixed by Addendum 1) by the fold manifests, test entries / distinct test files / universe per "
          "repeat: " + "; ".join(f"r{r} {v['manifest_test_entries']}/{v['manifest_unique_test_files']}/{v['manifest_universe']}"
                                for r, v in b.items())
          + ", i.e. each of the 639 discharges in exactly one test fold. Discharges with scored rows (the resampled "
          "units), each in exactly one fold: "
          + "; ".join(f"r{r} {v['discharges_with_rows']} ({'yes' if v['each_in_exactly_one_fold'] else 'NO'})" for r, v in b.items())
          + f". The {len(nomiss)} discharges of the 639 without scored rows in at least one repeat: "
          + "; ".join(f"{x.replace('.csv', '')} (no rows in " + ", ".join(f"r{r}" for r, v in b.items()
                                                                     if x in v["universe_files_without_scored_rows"]) + ")"
                      for x in nomiss)
          + ". Files with a target outside the 639: "
          + ", ".join(x.replace(".csv", "") for x in next(iter(b.values()))["valid_files_outside_universe"])
          + f" -> **{'passes' if ck['b']['passes'] else 'FAILS'}**."]
    c = ck["c"]["folds"]
    L += [f"- **(c)** estimator and twin scored on identical rows with equal stored reference errors in "
          f"{sum(v['same_rows'] and v['reference_errors_agree'] for v in c.values())} of {len(c)} folds "
          f"(rows per fold {min(v['rows'] for v in c.values()):,} to {max(v['rows'] for v in c.values()):,}) -> "
          f"**{'passes' if ck['c']['passes'] else 'FAILS'}**."]
    d = ck["d"]["folds"]
    L += [f"- **(d)** value in the data file at the located row against `CES_TI_y_true`: largest absolute difference "
          f"{max(v['max_abs_diff_eV'] for v in d.values()):.2e} eV over {sum(v['rows'] for v in d.values()):,} rows; rows "
          f"above 0.01 eV {sum(v['rows_above_tol'] for v in d.values())}; rows on a first or last file row "
          f"{sum(v['rows_first_or_last_grid_row'] for v in d.values())}; discharges outside their fold's test files "
          f"{sum(len(v['discharges_outside_fold_test_files']) for v in d.values())} -> **{'passes' if ck['d']['passes'] else 'FAILS'}**. "
          f"Reported: the hold located in the file reproduces sqrt(se_persistence) within 0.5 eV on a share "
          f"{min(v['hold_rows_within_0.5_eV_share'] for v in d.values()):.5f} to "
          f"{max(v['hold_rows_within_0.5_eV_share'] for v in d.values()):.5f} of the rows of a fold (largest deviation "
          f"{max(v['hold_max_dev_eV'] for v in d.values()):.2e} eV)."]
    e = ck["e"]
    L += ["- **(e)** code applied to split 42, against the recorded numbers:", "",
          "| quantity | this code | recorded | where | equal to three decimals |", "|---|---|---|---|---|"]
    for k, v in e["registered"].items():
        L.append(f"| {k} | {v['mine']:.4f} | {v['recorded']:.4f} | {v['where']} | {'yes' if v['equal_3dp'] else 'NO'} |")
    L += ["", f"-> **{'passes' if e['passes'] else 'FAILS'}**. Further comparisons on split 42 (reported):", "",
          "| quantity | this code | recorded | equal to three decimals |", "|---|---|---|---|"]
    for k, v in e["further"].items():
        if isinstance(v, dict) and "mine" in v:
            L.append(f"| {k} | {v['mine']:.4f} | {v['recorded']:.4f} | {'yes' if v['equal_3dp'] else 'no'} |")
    pj = e["further"]["propensity_join"]
    L += ["", f"Propensity join on the B.48 rows of split 42: {pj['with_window']:,} of {pj['rows']:,} rows with a window, "
          f"{pj['with_window_without_p']} of them without p, {pj['without_window_with_p']} without a window but with p.", "",
          "All five checks pass." if ck["passes"] else "**A check FAILS; the statements below were not read.**", ""]

    L += ["## Statements", ""]
    for g, keys in GROUPS:
        L += [f"### {g}", "", "| statement | point [95 % interval] | reference | verdict | per repeat (r2 ... r9) |", "|---|---|---|---|---|"]
        for k in keys:
            s = st[k]
            if s.get("all_thirds"):
                for tn, v in s["value"].items():
                    L.append(f"| {k}, {tn} third: {STEXT[k]} | {s3(v)} ({excl(v, 0)}) | 0 | "
                             f"{'interval above 0' if s['per_third_holds'][tn] else 'interval not above 0'} | |")
                L.append(f"| **{k}** (all three thirds) | | | **{hs(s['holds'])}** | |")
                continue
            per = ", ".join(f"{v:.3f}" if k in ("A1", "B1", "B2", "C3", "C4", "H2") else f"{v:+.3f}" for v in s["per_repeat"].values())
            L.append(f"| **{k}**: {STEXT[k]} | {fm(k, s['value'])} ({excl(s['value'], s['reference'])}) | {s['reference']:g} | "
                     f"**{hs(s['holds'])}** | {per} |")
        if g == "B":
            L += ["", "Discharges with at least 50 scored rows (the units of B1 and B2) per repeat: "
                  + ", ".join(f"r{r} {v}" for r, v in st["B1"]["eligible_discharges_per_repeat"].items())
                  + "; resampled discharges (with scored rows) per repeat for every other statement: "
                  + ", ".join(f"r{r} {v['discharges_with_rows']}" for r, v in ck["b"]["repeats"].items()) + "."]
        L.append("")
    L += ["Statements that hold: " + ", ".join(k for k, s in st.items() if s["holds"]) + ". Statements that do not hold: "
          + ", ".join(k for k, s in st.items() if not s["holds"]) + ".", ""]

    # ---------- reported
    L += ["## Reported quantities", "", "### Overall (all pooled rows)", "",
          "| over | skill of the estimator |", "|---|---|"]
    for ref in ("t", "gp", "hold", "off"):
        L.append(f"| {LAB[ref]} | {s3(rp['overall'][ref])} |")
    L += ["", "### Distribution of the absolute error", "",
          "| estimator over | ratio of the 90th percentiles | ratio of the medians |", "|---|---|---|"]
    for lab in ("causal GP", "hold", "twin"):
        L.append(f"| {lab} | {f3(rp['p90_ratio'][lab]['value'])} ({excl(rp['p90_ratio'][lab]['value'], 1)}) | "
                 f"{f3(rp['median_ratio'][lab]['value'])} ({excl(rp['median_ratio'][lab]['value'], 1)}) |")
    L += ["", "Percentiles of the absolute error (eV, mean over repeats):", "",
          "| estimator | p50 | p75 | p90 | p95 | p99 |", "|---|---|---|---|---|---|"]
    for lab, v in rp["abs_error_percentiles_eV_mean_over_repeats"].items():
        L.append(f"| {lab} | " + " | ".join(f"{v[str(p)]:.1f}" for p in (50, 75, 90, 95, 99)) + " |")
    L += ["", "### Per discharge", "",
          f"- Share of discharges with the lower MSE than hold: {f3(rp['share_lower_mse_over_hold']['value'])}.",
          f"- Share of discharges by the median absolute error (lower than the causal GP): "
          f"{f3(rp['share_lower_median_abs_over_causal GP']['value'])}; lower than the twin: "
          f"{f3(rp['share_lower_median_abs_over_twin']['value'])}.",
          f"- Eligible discharges (>= 50 scored rows) per repeat: "
          + ", ".join(str(v) for v in st["B1"]["eligible_discharges_per_repeat"].values()) + ".",
          f"- Share of the total squared-error reduction against the causal GP held by the ten discharges with the "
          f"largest reduction: {rp['concentration']['top10_share_over_gp']['mean']:.3f} (per repeat "
          + ", ".join(f"{v:.3f}" for v in rp['concentration']['top10_share_over_gp']['per_repeat'].values())
          + f"); by the twentieth ({int(next(iter(rp["concentration"]["twentieth_size"].values())))} discharges): "
          f"{rp['concentration']['top_twentieth_share_over_gp']['mean']:.3f} (per repeat "
          + ", ".join(f"{v:.3f}" for v in rp['concentration']['top_twentieth_share_over_gp']['per_repeat'].values())
          + f"). Against the twin, the ten with the largest reduction against the twin: "
          f"{rp['concentration']['top10_share_over_twin']['mean']:.3f}.",
          f"- With those ten removed (lower bound; the removal selects on the outcome): over the causal GP "
          f"{s3(rp['without_top10']['gp'])} ({excl(rp['without_top10']['gp'], 0)}); over the twin "
          f"{s3(rp['without_top10']['t'])} ({excl(rp['without_top10']['t'], 0)}); over hold {s3(rp['without_top10']['hold'])}.", ""]

    def cell_row(lab, c, refs=("t", "gp", "hold")):
        return (f"| {lab} | {np.mean(list(c['rows_per_repeat'].values())):,.0f} | {c['row_share']:.4f} | "
                + " | ".join(f"{s3(c[r])}" if r in c else "" for r in refs) + " |")

    L += ["### Skills by class of rows (every skill also against hold)", "",
          "| rows | rows per repeat (mean) | share of rows | over twin | over causal GP | over hold | over offline GP |",
          "|---|---|---|---|---|---|---|"]
    L.append(cell_row("not flagged by the value rules", rp["plain_rows"], ("t", "gp", "hold", "off")))
    L.append(cell_row("flagged", rp["flagged_rows"], ("t", "gp", "hold", "off")))
    for tn in B44.THIRDS:
        L.append(cell_row(f"{tn} third by median T_i", rp["thirds"][tn]))
    for bn, c in rp["age"].items():
        L.append(cell_row(f"age of the last value {bn}", c))
    L.append(cell_row("top fifth by risk", rp["risk_top_fifth"]))
    L.append(cell_row("rows with a ten-row window, weighted by the odds", rp["risk_weighted"]))
    L.append(cell_row("rows with a ten-row window, unweighted", rp["risk_window_rows_unweighted"]))
    L.append(cell_row("rows 1 to 3 after an absent frame", rp["blip_rows_1_3"]))
    for cn, c in rp["classes"].items():
        L.append(cell_row(f"class {CLAB[cn]}", c, ("t", "gp", "hold", "off")))
    L.append(cell_row("all but the ten discharges with the largest reduction", rp["without_top10"]))
    L += ["", f"Shares of the reduction: rows not flagged hold {f3(st['C3']['value'])} of the reduction against the causal "
          f"GP and {f3(st['C4']['value'])} against the twin; quiet rows hold {f3(st['H2']['value'])} against the twin and "
          f"{f3(rp['quiet_share_of_reduction_vs_gp']['value'])} against the causal GP. A share above 1 means the other "
          "rows have a net increase.", "",
          "Thirds: edges of median T_i (eV) " + "; ".join(f"r{r} {v[0]:.0f}, {v[1]:.0f}" for r, v in rp["thirds_edges_eV"].items())
          + "; discharges without a median: " + ", ".join(str(v) for v in rp["thirds_excluded_discharges"].values()) + ".", "",
          f"Odds weighting: effective number of rows {rp['effective_rows']['mean']:,.0f} (share of the rows with a "
          f"window {rp['effective_rows']['share_mean']:.3f}, mean over repeats). Per repeat: "
          + "; ".join(f"r{r} {v['effective_rows']:,.0f} ({v['effective_share']:.3f}), clip {v['clip_w']:.3f}, unusable rows in "
                      f"the top fifth {v['share_of_unusable_in_top_fifth']:.3f}, rows without a window {v['rows_without_window']:,}"
                      for r, v in rp["risk_info"].items()) + ".", "",
          "Rows whose age could not be binned (rounded age below 10 ms) per repeat: "
          + ", ".join(str(v) for v in rp["age_unbinned_rows"].values()) + ".", ""]

    # ---------- expectations
    L += ["## Expectations of the registration (section 6) beside the findings", "",
          "| expectation | observed (point; verdict) | range met | verdict met |", "|---|---|---|---|"]
    for x in o["expectations"]:
        vm = "" if x["verdict_met"] is None else ("yes" if x["verdict_met"] else "no")
        L.append(f"| {x['key']}: {x['expectation']} | {x['observed']} | {'yes' if x['range_met'] else 'no'} | {vm} |")
    L += ["", "## Addendum 1 and deviations", "",
          "The first run stopped at check (b), read as 639 discharges with scored rows in every repeat (635, 634, 633 "
          "were found); checks (a), (c), (d), (e) had passed and no statement had been read. Addendum 1 of the "
          "registration (committed before any statement was read) fixes check (b) on the fold manifests plus no "
          "discharge with scored rows in two folds, and records the implementation choices below as the registered "
          "ones. No other deviation.", "",
          "## Choices made where the registration left room", ""] + [f"- {c}" for c in o["choices"]]
    L += ["", "## Limits", "", "The pooled folds re-partition the discharges on which the recurrent design was adopted; no "
          "untouched hold-out exists. One training per fold: the two-level interval covers the difference between "
          "re-partitions (in which the initialisation changes too), not a spread between initialisations on a fixed "
          "fold. The classes are defined from BES and the value rules are thresholds. Removing the discharges with the "
          "largest reduction selects on the outcome.", "",
          f"Run time: stages {o['runtime_stages_s']}; readout {o['runtime_readout_s']:.0f} s."]
    return "\n".join(L) + "\n"


# ------------------------------------------------------------------------------------------ main
def main():
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    CACHE.mkdir(parents=True, exist_ok=True)
    rt_path = CACHE / "runtime.json"
    for name, fn in (("files", stage_files), ("rows", stage_rows), ("prop", stage_prop), ("checks", stage_checks),
                     ("readout", stage_readout)):
        if stage in (name, "all"):
            t = _time.time()
            fn()
            rt = json.loads(rt_path.read_text()) if rt_path.exists() else {}
            rt[name] = round(rt.get(name, 0) + _time.time() - t, 1)     # summed over resumed pieces
            rt_path.write_text(json.dumps(rt))
    log("done")


if __name__ == "__main__":
    main()
