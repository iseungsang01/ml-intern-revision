"""B.56 (PREREGISTRATION_B56.md with Addendum 1): a robust state-space filter with the beam-emission
array as input.  CPU only; no network is trained or run.

The specification is the development script `kfx_model.py` (byte-identical copy of the look's
`check_kalman_x_clip.py`); only its definition part (the source before the line
`rng = np.random.default_rng(3)`: data loading, the filter `run`, the loss, the row mapping `map_rows`)
is executed here.  The gain fit with a fixed clip is that of `kfx_components.py` (byte-identical copy of
`check_kfx_components.py`), reimplemented in `fit_unit` with the same starts, iteration caps and
tolerances.  Both SHA-256 are written into the output.

Units (each cached under data/.b56/, so a call resumes):
  pooled r{r}_f{k}  r = 2..9, k = 0..4   (r2 = development set, r3..r9 = confirmation set)
  split s{s}        s = 42, 1, 7, 123    (training discharges = every discharge not in the split's test set)
  camp              the split by time of B.29 (test rows of data/.b29_norm0_camp_cut_s{i})

Usage (repo root):
  py ces_prediction/experiments/b56_kfx/run_b56.py fit [unit ...]     # fit + cache (default: all missing)
  py ces_prediction/experiments/b56_kfx/run_b56.py checks             # checks (a), (b), (c)
  py ces_prediction/experiments/b56_kfx/run_b56.py analyze            # statements -> docs/paper/b56_kfx.{json,md}
"""
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
DATA = ROOT / "data"
CACHE = DATA / ".b56"
OUT_JSON = ROOT / "docs" / "paper" / "b56_kfx.json"
OUT_MD = ROOT / "docs" / "paper" / "b56_kfx.md"
MODEL_SRC = HERE / "kfx_model.py"
COMP_SRC = HERE / "kfx_components.py"
SEEDS = (42, 1, 7, 123)
REPEATS = tuple(range(2, 10))
CONFIRM = tuple(range(3, 10))
CLIP_MAIN = 0.2
CLIP_SENS = (0.1, 0.4)
B_SKILL = 20000
B_MED = 2000
SEED_BOOT = 56
T0 = time.time()


def log(m):
    print(f"[{time.time() - T0:7.0f}s] {m}", flush=True)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


# ------------------------------------------------------------------ the frozen definitions
_NS = None


def ns():
    """Execute the definition part of kfx_model.py once (loads the 641 files)."""
    global _NS
    if _NS is None:
        src = MODEL_SRC.read_text(encoding="utf-8")
        marker = "rng = np.random.default_rng(3)"
        if src.count(marker) != 1:
            raise SystemExit("FATAL: marker line not found exactly once in kfx_model.py")
        defs = src.split(marker)[0]
        saved = sys.argv
        sys.argv = [str(MODEL_SRC), "2"]          # R is read at top level; unused by the definitions
        cwd = os.getcwd()
        os.chdir(ROOT)                            # the definitions glob data/s*.csv relative to the repo root
        try:
            g = {"__file__": str(MODEL_SRC), "__name__": "kfx_model"}
            exec(compile(defs, str(MODEL_SRC), "exec"), g)
        finally:
            sys.argv = saved
            os.chdir(cwd)
        _NS = g
        log(f"definitions of kfx_model.py executed: {g['NF']} files, TMAX {g['TMAX']}")
    return _NS


# ------------------------------------------------------------------ B.39 helpers (copied, not imported)
def inits(seed):
    i = SEEDS.index(seed)
    return tuple(SEEDS[(i + j) % len(SEEDS)] for j in range(3))


def split_run_dir(model, seed, init):
    return DATA / f".b39_{model}_cut_s{seed}_i{init}"


def twin_dir(seed, init):
    return DATA / f".b29a1_v2h_cut_s{seed}" if init == seed else split_run_dir("v2h", seed, init)


def load_err(d):
    z = np.load(Path(d) / "comparison_errors_test.npz")
    t = "CES_TI"
    return {"shot": z[f"{t}_shot"].astype(np.int64), "row": z[f"{t}_row"].astype(np.int64),
            "y": z[f"{t}_y_true"].astype(np.float64), "dt": z[f"{t}_dt_ms"].astype(np.float64),
            "A": z[f"{t}_se_model"].astype(np.float64), "G": z[f"{t}_se_gp_causal"].astype(np.float64),
            "H": z[f"{t}_se_persistence"].astype(np.float64), "O": z[f"{t}_se_gp"].astype(np.float64)}


def unit_sources(unit):
    """-> (estimator run dirs, twin run dirs) of a unit; the scored rows are those of the first estimator dir."""
    if unit.startswith("r"):
        r, k = unit[1:].split("_f")
        return [DATA / f".b39p_v2_cut_r{r}_f{k}"], [DATA / f".b39p_v2h_cut_r{r}_f{k}"]
    if unit.startswith("s"):
        s = int(unit[1:])
        return [split_run_dir("v2", s, i) for i in inits(s)], [twin_dir(s, i) for i in inits(s)]
    if unit == "camp":
        return [DATA / f".b29_norm0_camp_cut_s{i}" for i in SEEDS], []
    raise SystemExit(f"FATAL: unknown unit {unit}")


ALL_UNITS = [f"r{r}_f{k}" for r in REPEATS for k in range(5)] + [f"s{s}" for s in SEEDS] + ["camp"]


# ------------------------------------------------------------------ row location (deviation, declared)
def map_rows_exact(fi, rows, yv, tol=3e-4):
    """As the frozen map_rows (order-preserving subsequence match of the stored T_i values against the grid),
    but with a tolerance of 0.0003 eV instead of 0.01 eV (the stored value is the file value to float32
    precision: at B.54's independent location of the pooled rows the largest difference is 0.00024 eV; two
    distinct file values of one discharge can differ by 0.0005 eV), and
    where several consecutive grid values lie within the tolerance the closest one is taken.  The frozen first-within-tolerance rule put 45 of the ~1.5 million pooled rows
    one or more grid rows early where two neighbouring values differ by < 0.01 eV (found by the cross-check
    against B.54's independent location of the same rows)."""
    g = ns()
    ti = g["grid"][fi]["ti"]
    cand = np.flatnonzero(np.isfinite(ti))
    cv = ti[cand]
    order = np.argsort(rows)
    gi = np.full(len(rows), -1)
    p = 0
    for o in order:
        while p < len(cand) and abs(cv[p] - yv[o]) > tol:
            p += 1
        if p >= len(cand):
            continue
        best = p
        q_ = p + 1
        while q_ < len(cand) and abs(cv[q_] - yv[o]) <= tol:
            if abs(cv[q_] - yv[o]) < abs(cv[best] - yv[o]):
                best = q_
            q_ += 1
        gi[o] = cand[best]
        p = best + 1
    return gi


def theta_of(p, which):
    th0 = np.array(p["theta_kf0"])
    if which == "kf0":
        return np.r_[th0, 0, 0], False, CLIP_MAIN
    if which == "kc":
        return np.r_[th0, p["b_kc"], 0.0], True, CLIP_MAIN
    if which == "kg":
        return np.r_[th0, 0.0, p["b_kg"]], True, CLIP_MAIN
    if which == "kfx":
        return np.r_[th0, p["b_kfx"]], True, CLIP_MAIN
    c = float(which.split("clip")[1])
    return np.r_[th0, p[f"b_kfx_clip{c}"]], True, c


def relocate(unit):
    """Recompute the filter outputs from the cached fitted parameters and locate the scored rows exactly."""
    g = ns()
    z = dict(np.load(CACHE / f"{unit}.npz"))
    if "gi_frozen_map" in z:
        z["gi"] = z["gi_frozen_map"]
    p = json.loads((CACHE / f"{unit}.json").read_text(encoding="utf-8"))
    gi = np.full(len(z["shot"]), -1, np.int64)
    for fi in np.unique(z["shot"]):
        m = z["shot"] == fi
        gi[m] = map_rows_exact(int(fi), z["row"][m], z["y"][m])
    ok = gi >= 0
    out = {"shot": z["shot"], "row": z["row"], "y": z["y"], "gi": gi, "gi_frozen_map": z["gi"]}
    for which in ("kf0", "kc", "kg", "kfx", "kfx_clip0.1", "kfx_clip0.4"):
        th, use_x, clip = theta_of(p, which)
        g["CLIP"][0] = clip
        P = g["run"](th, None, use_x)
        v = np.full(len(gi), np.nan)
        v[ok] = P[z["shot"][ok], gi[ok]]
        out[f"est_{which}"] = v
    g["CLIP"][0] = CLIP_MAIN
    np.savez(CACHE / f"{unit}.npz", **out)
    log(f"{unit}: relocated, rows moved {int((gi != z['gi']).sum())}, unmatched {int((~ok).sum())}")


# ------------------------------------------------------------------ fit (kfx_components.py, reimplemented)
def fit_unit(unit):
    g = ns()
    loss, run, CLIP, NF = g["loss"], g["run"], g["CLIP"], g["NF"]
    est_dirs, twin_dirs = unit_sources(unit)
    ref = load_err(est_dirs[0])
    test_f = np.unique(ref["shot"])
    train_f = np.setdiff1d(np.arange(NF), test_f)
    nm = lambda maxiter, xatol: dict(maxiter=maxiter, xatol=xatol, fatol=1e-1)      # noqa: E731
    f0 = minimize(lambda th: loss(np.r_[th, 0, 0], train_f, False),
                  np.array([np.log(0.002), np.log(0.01), np.log(2.0)]), method="Nelder-Mead", options=nm(250, 1e-3))
    th0 = f0.x
    out = {"theta_kf0": th0.tolist(), "train_mse_kf0": float(f0.fun)}
    preds = {"kf0": run(np.r_[th0, 0, 0], None, False)}
    for clip, tag in [(CLIP_MAIN, "")] + [(c, f"_clip{c}") for c in CLIP_SENS]:
        CLIP[0] = clip
        if clip == CLIP_MAIN:
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
    CLIP[0] = CLIP_MAIN
    # scored rows -> grid rows
    gi = np.full(len(ref["shot"]), -1, np.int64)
    for fi in test_f:
        m = ref["shot"] == fi
        gi[m] = map_rows_exact(int(fi), ref["row"][m], ref["y"][m])
    ok = gi >= 0
    sig = {}
    for k_, P in preds.items():
        v = np.full(len(gi), np.nan)
        v[ok] = P[ref["shot"][ok], gi[ok]]
        sig[k_] = v
    CACHE.mkdir(exist_ok=True)
    np.savez(CACHE / f"{unit}.npz", shot=ref["shot"], row=ref["row"], y=ref["y"], gi=gi,
             **{f"est_{k_}": v for k_, v in sig.items()})
    (CACHE / f"{unit}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    log(f"{unit}: rows {len(gi)}, unmatched {int((~ok).sum())}, KF0 {np.round(th0, 3).tolist()}, "
        f"b_c|b_g {np.round(out['b_kfx'], 3).tolist()}, Kc {out['b_kc']:.3f}, Kg {out['b_kg']:.3f}")


# ------------------------------------------------------------------ checks
def check_c():
    """(c): the output for row t is unchanged when the CES values from t on are altered."""
    g = ns()
    rng = np.random.default_rng(7)
    theta = np.r_[np.log(0.014), np.log(0.051), np.log(2.86), 0.48, 0.45]
    g["CLIP"][0] = CLIP_MAIN
    base = g["run"](theta, None, True)
    LY0 = g["LY"].copy()
    worst, n = 0.0, 0
    try:
        for _ in range(30):
            fi = int(rng.integers(g["NF"]))
            obs = np.flatnonzero(np.isfinite(LY0[fi]))
            if len(obs) < 20:
                continue
            t = int(obs[rng.integers(5, len(obs) - 5)])
            LY = LY0.copy()
            LY[fi, t:] = np.where(np.isfinite(LY0[fi, t:]), LY0[fi, t:] + rng.normal(0, 1.0, LY0.shape[1] - t), np.nan)
            g["LY"][:] = LY
            alt = g["run"](theta, None, True)
            worst = max(worst, float(np.nanmax(np.abs(alt[fi, :t + 1] - base[fi, :t + 1]))))
            moved = float(np.nanmax(np.abs(alt[fi, t + 1:] - base[fi, t + 1:])))   # control: later rows do move
            n += int(moved > 0)
    finally:
        g["LY"][:] = LY0
    return {"perturbations": 30, "max_abs_change_rows_up_to_t_eV": worst, "perturbations_where_later_rows_moved": n,
            "passes": bool(worst == 0.0 and n > 0)}


def load_unit(unit):
    z = np.load(CACHE / f"{unit}.npz")
    d = {k: z[k] for k in z.files}
    est_dirs, twin_dirs = unit_sources(unit)
    E = [load_err(x) for x in est_dirs]
    T = [load_err(x) for x in twin_dirs]
    for x in E[1:] + T:
        if not (np.array_equal(x["shot"], E[0]["shot"]) and np.array_equal(x["row"], E[0]["row"])):
            raise SystemExit(f"FATAL: {unit}: stored runs not scored on identical rows")
    if not (np.array_equal(d["shot"], E[0]["shot"]) and np.array_equal(d["row"], E[0]["row"])):
        raise SystemExit(f"FATAL: {unit}: cache rows differ from the stored rows")
    y = E[0]["y"]
    out = {"unit": unit, "shot": d["shot"], "row": d["row"], "y": y, "gi": d["gi"], "dt": E[0]["dt"],
           "G": E[0]["G"], "H": E[0]["H"], "O": E[0]["O"],
           "A": np.array([x["A"] for x in E]), "Tw": np.array([x["A"] for x in T]) if T else None}
    for k_ in ("kf0", "kc", "kg", "kfx", "kfx_clip0.1", "kfx_clip0.4"):
        out[k_] = (d[f"est_{k_}"] - y) ** 2
    out["signed"] = {k_: d[f"est_{k_}"] for k_ in ("kf0", "kfx")}
    return out


def checks():
    res = {"sha256": {"kfx_model.py": sha(MODEL_SRC), "kfx_components.py": sha(COMP_SRC)}}
    # (b) identical rows: every stored scored row located, the grid value equals the stored one,
    # every filter output finite; cross-check of the location against B.54's independent grid rows.
    g = ns()
    b = {"units": {}, "passes": True}
    for u in ALL_UNITS:
        z = np.load(CACHE / f"{u}.npz")
        ok = z["gi"] >= 0
        vals = np.full(len(ok), np.nan)
        vals[ok] = g["Y"][z["shot"][ok], z["gi"][ok]]
        dmax = float(np.nanmax(np.abs(vals - z["y"]))) if ok.any() else np.inf
        fin = all(np.isfinite(z[f"est_{k}"]).all() for k in ("kf0", "kc", "kg", "kfx"))
        rec = {"rows_stored": int(len(ok)), "rows_located": int(ok.sum()), "max_abs_value_diff_eV": dmax,
               "all_filter_outputs_finite": bool(fin)}
        if u.startswith("r"):
            p54 = DATA / ".b54" / f"rows_{u}.npz"
            if p54.exists():
                q = np.load(p54)
                if np.array_equal(q["shot"], z["shot"]) and np.array_equal(q["row"], z["row"]):
                    off = q["grow"] - z["gi"]
                    rec["b54_grid_row_minus_mine_unique"] = sorted(set(np.unique(off).tolist()))[:5]
        good = rec["rows_located"] == rec["rows_stored"] and dmax <= 0.01 and fin
        rec["rows_moved_from_frozen_map"] = int((z["gi"] != z["gi_frozen_map"]).sum()) if "gi_frozen_map" in z.files else None
        if "b54_grid_row_minus_mine_unique" in rec:
            good &= len(rec["b54_grid_row_minus_mine_unique"]) == 1
        rec["passes"] = bool(good)
        b["passes"] &= bool(good)
        b["units"][u] = rec
    res["b"] = b
    res["c"] = check_c()
    # (a) r2 point estimates of section 2
    D = [load_unit(f"r2_f{k}") for k in range(5)]
    cat = lambda key: np.concatenate([d[key] if d[key].ndim == 1 else d[key][0] for d in D])   # noqa: E731
    S = {k: cat(k) for k in ("A", "Tw", "G", "H", "kf0", "kc", "kg", "kfx")}
    sk = lambda a, r: float(1 - S[a].sum() / S[r].sum())                                       # noqa: E731
    quiet = np.concatenate([quiet_mask(d) for d in D])
    mine = {"kf0_over_gp": sk("kf0", "G"), "kfx_over_kf0": sk("kfx", "kf0"), "net_over_kfx": sk("A", "kfx"),
            "twin_over_kf0": sk("Tw", "kf0"), "kg_over_kf0": sk("kg", "kf0"), "kc_over_kf0": sk("kc", "kf0"),
            "kfx_over_kf0_quiet": float(1 - S["kfx"][quiet].sum() / S["kf0"][quiet].sum())}
    look = {"kf0_over_gp": 0.002, "kfx_over_kf0": 0.049, "net_over_kfx": 0.040, "twin_over_kf0": 0.044,
            "kg_over_kf0": 0.032, "kc_over_kf0": 0.042, "kfx_over_kf0_quiet": 0.045}
    med_m = {k: float(np.median(np.sqrt(S[k]))) for k in ("kfx", "kf0", "G", "H", "A")}
    med_l = {"kfx": 41.8, "kf0": 43.3, "G": 44.1, "H": 45.3, "A": 47.1}
    a = {"skills": {k: {"look": look[k], "batch": mine[k], "same_to_3_decimals": round(mine[k], 3) == look[k]} for k in look},
         "medians_eV": {k: {"look": med_l[k], "batch": med_m[k], "same_to_1_decimal": round(med_m[k], 1) == med_l[k]} for k in med_l}}
    a["passes"] = bool(all(v["same_to_3_decimals"] for v in a["skills"].values()) and
                       all(v["same_to_1_decimal"] for v in a["medians_eV"].values()))
    res["a"] = a
    res["all_pass"] = bool(a["passes"] and b["passes"] and res["c"]["passes"])
    (CACHE / "checks.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    log(f"check (a) {a['passes']}  (b) {b['passes']}  (c) {res['c']['passes']}")
    for k, v in a["skills"].items():
        log(f"  (a) {k}: look {v['look']:+.3f} batch {v['batch']:+.4f}")
    for k, v in a["medians_eV"].items():
        log(f"  (a) median {k}: look {v['look']} batch {v['batch']:.2f}")
    log(f"  (c) {res['c']}")
    return res


# ------------------------------------------------------------------ row classes
_TIME = {}


def grid_time(fi):
    if fi not in _TIME:
        g = ns()
        _TIME[fi] = g["T"][fi]
    return _TIME[fi]


def quiet_mask(d):
    """True for rows NOT within the ten grid rows that start at the first row after an absent frame
    (a step of 15 ms to 0.5 s between grid rows), as in the look (check_kfx_strata.py)."""
    out = np.ones(len(d["gi"]), bool)
    for fi in np.unique(d["shot"]):
        t = grid_time(int(fi))
        step = np.diff(t, prepend=np.nan)
        blip = (step >= 0.015) & (step < 0.5)
        near = np.zeros(len(t), bool)
        for j in np.flatnonzero(blip):
            near[j:j + 10] = True
        m = d["shot"] == fi
        out[m] = ~near[d["gi"][m]]
    return out


# ------------------------------------------------------------------ statistics
def q(b, lo=2.5, hi=97.5):
    b = np.asarray(b, float)
    return [float(np.percentile(b, lo)), float(np.percentile(b, hi))]


class Pooled:
    """Per re-partition: per-discharge sums of squared errors; two-level bootstrap (a re-partition drawn,
    then its discharges resampled), as analyze_b39.arm_p.hier.  The point is the mean over re-partitions."""

    def __init__(self, reps, mask_fn=None):
        self.reps = list(reps)
        self.per, self.rows = {}, {}
        for r in self.reps:
            D = [load_unit(f"r{r}_f{k}") for k in range(5)]
            cols = {}
            for key in ("A", "Tw", "G", "H", "O", "kf0", "kc", "kg", "kfx", "kfx_clip0.1", "kfx_clip0.4"):
                cols[key] = np.concatenate([d[key] if d[key].ndim == 1 else d[key][0] for d in D])
            shot = np.concatenate([d["shot"] for d in D])
            msk = np.concatenate([mask_fn(d, r) for d in D]) if mask_fn else np.ones(len(shot), bool)
            uf, idx = np.unique(shot[msk], return_inverse=True)
            self.per[r] = {k: np.bincount(idx, v[msk], len(uf)) for k, v in cols.items()}
            self.rows[r] = {k: v[msk] for k, v in cols.items()}
            self.rows[r]["_idx"] = idx
            self.rows[r]["_n"] = len(uf)

    def skill(self, a, b, rng, nboot=B_SKILL, lo=2.5, hi=97.5):
        draws = rng.choice(np.array(self.reps), size=nboot)
        o = np.empty(nboot)
        for r in self.reps:
            sel = np.flatnonzero(draws == r)
            m, bb = self.per[r][a], self.per[r][b]
            for c0 in range(0, len(sel), 2000):
                ss = sel[c0:c0 + 2000]
                idx = rng.integers(0, len(m), size=(len(ss), len(m)))
                o[ss] = 1.0 - m[idx].sum(1) / bb[idx].sum(1)
        per = {str(r): float(1 - self.per[r][a].sum() / self.per[r][b].sum()) for r in self.reps}
        return [float(np.mean(list(per.values()))), *q(o, lo, hi)], per

    def median(self, a):
        return float(np.mean([np.median(np.sqrt(self.rows[r][a])) for r in self.reps]))

    def med_ratio(self, a, b, rng, nboot=B_MED):
        def wmed(v, w):
            o = np.argsort(v)
            cw = np.cumsum(w[o])
            return v[o][np.searchsorted(cw, 0.5 * cw[-1])]
        pts = {}
        for r in self.reps:
            pts[r] = float(np.median(np.sqrt(self.rows[r][a])) / np.median(np.sqrt(self.rows[r][b])))
        draws = rng.choice(np.array(self.reps), size=nboot)
        o = np.empty(nboot)
        for r in self.reps:
            va, vb, idx, n = np.sqrt(self.rows[r][a]), np.sqrt(self.rows[r][b]), self.rows[r]["_idx"], self.rows[r]["_n"]
            for j in np.flatnonzero(draws == r):
                w = rng.multinomial(n, np.full(n, 1.0 / n)).astype(float)[idx]
                o[j] = wmed(va, w) / wmed(vb, w)
        return [float(np.mean(list(pts.values()))), *q(o)], {str(r): v for r, v in pts.items()}


def analyze():
    chk = json.loads((CACHE / "checks.json").read_text(encoding="utf-8"))
    if not chk["all_pass"]:
        raise SystemExit("FATAL: a check failed; no statement is read")
    rng = np.random.default_rng(SEED_BOOT)
    res = {"sha256": chk["sha256"], "checks": chk, "clip": CLIP_MAIN}

    def block(reps, mask_fn=None, full=True):
        P = Pooled(reps, mask_fn)
        o = {}
        pairs = [("kfx", "kf0"), ("kg", "kf0"), ("A", "kfx"), ("Tw", "kf0"), ("kfx", "Tw"), ("kc", "kf0"),
                 ("kfx", "kc"), ("kfx", "kg"), ("kfx", "G"), ("kfx", "O"), ("A", "G"), ("A", "Tw"), ("kf0", "G")]
        if full:
            pairs += [("kfx_clip0.1", "kf0"), ("kfx_clip0.4", "kf0")]
        for a, b in pairs:
            o[f"{a}_over_{b}"], o[f"{a}_over_{b}_per_repeat"] = P.skill(a, b, rng)
        if full:
            o["kf0_over_G_90"], _ = P.skill("kf0", "G", rng, lo=5, hi=95)
            o["median_ratio_kfx_over_hold"], o["median_ratio_per_repeat"] = P.med_ratio("kfx", "H", rng)
        o["median_abs_eV"] = {k: P.median(k) for k in ("A", "Tw", "G", "H", "O", "kf0", "kc", "kg", "kfx")}
        o["rows"] = {str(r): int(len(P.rows[r]["A"])) for r in reps}
        o["discharges"] = {str(r): int(P.rows[r]["_n"]) for r in reps}
        return o

    # confirmation set and development set
    log("pooled r3-r9 ...")
    res["confirm"] = block(CONFIRM)
    log("quiet rows r3-r9 ...")
    res["confirm_quiet"] = block(CONFIRM, lambda d, r: quiet_mask(d), full=False)

    def plain(d, r):
        """Rows not flagged by the value rules: B.54's row table of the same fold (B.48 definition)."""
        z = np.load(DATA / ".b54" / f"rows_{d['unit']}.npz")
        if not (np.array_equal(z["shot"], d["shot"]) and np.array_equal(z["row"], d["row"])):
            raise SystemExit(f"FATAL: B.54 row table of {d['unit']} not aligned")
        return z["plain"].astype(bool)

    log("plain rows r3-r9 ...")
    res["confirm_plain"] = block(CONFIRM, plain, full=False)
    # thirds by the median T_i of a discharge, within a re-partition (B.54 / B.44 definition)
    F = np.load(DATA / ".b54" / "files.npz")
    lab = {}
    for r in CONFIRM:
        u = np.unique(np.concatenate([np.load(CACHE / f"r{r}_f{k}.npz")["shot"] for k in range(5)]))
        v = F["median_ti"][u]
        have = np.flatnonzero(np.isfinite(v))
        o = np.lexsort((F["shot"][u][have], v[have]))              # B44.split_thirds: sort by (value, shot)
        parts = [have[p] for p in np.array_split(o, 3)]
        L = np.full(641, -1)
        for j, p in enumerate(parts):
            L[u[p]] = j
        lab[r] = L
    for j, tn in enumerate(("coldest", "middle", "hottest")):
        log(f"third {tn} ...")
        res[f"confirm_third_{tn}"] = block(CONFIRM, lambda d, r, j=j: lab[r][d["shot"]] == j, full=False)
    log("development set r2 ...")
    res["dev_r2"] = block((2,))
    # r2 intervals by discharge resampling (one re-partition: no first level)
    # (block((2,)) already draws only r2 at the first level, so its intervals are discharge-only)

    # four splits
    res["splits"] = {}
    for s in SEEDS:
        log(f"split {s} ...")
        d = load_unit(f"s{s}")
        uf, idx = np.unique(d["shot"], return_inverse=True)
        n = len(uf)
        Ssum = lambda v: np.bincount(idx, v, n)                                    # noqa: E731
        sums = {k: Ssum(d[k]) for k in ("G", "H", "O", "kf0", "kc", "kg", "kfx")}
        Ai = np.array([Ssum(v) for v in d["A"]])
        Ti = np.array([Ssum(v) for v in d["Tw"]])
        qm = quiet_mask(d)
        idxq = np.unique(d["shot"][qm], return_inverse=True)[1]
        nq = len(np.unique(d["shot"][qm]))
        sq = {k: np.bincount(idxq, d[k][qm], nq) for k in ("kf0", "kfx")}

        def det(a, b, lo=2.5, hi=97.5, S_=sums, nn=n):
            I = rng.integers(0, nn, size=(B_SKILL, nn))
            bs = 1 - S_[a][I].sum(1) / S_[b][I].sum(1)
            return [float(1 - S_[a].sum() / S_[b].sum()), *q(bs, lo, hi)]

        def two(Am, bvec):
            """B.39 interval: an initialisation drawn, discharges resampled (bvec deterministic or (inits, n))."""
            Bm = bvec if bvec.ndim == 2 else bvec[None, :]
            I = rng.integers(0, n, size=(B_SKILL, n))
            ia, ib = rng.integers(0, Am.shape[0], B_SKILL), rng.integers(0, Bm.shape[0], B_SKILL)
            a = np.take_along_axis(Am[ia], I, 1).sum(1)
            b = np.take_along_axis(Bm[ib], I, 1).sum(1)
            return [float(1 - Am.sum(1).mean() / Bm.sum(1).mean()), *q(1 - a / b)]

        o = {"KX1": det("kfx", "kf0"), "KX2": det("kg", "kf0"), "KX3": two(Ai, sums["kfx"]),
             "KX3_init_seed_only": det_from(Ai[0], sums["kfx"], rng, n),
             "KX4_90": det("kf0", "G", 5, 95), "KX6": det("kfx", "kf0", S_=sq, nn=nq),
             "twin_over_kf0": two(Ti, sums["kf0"]), "kfx_over_G": det("kfx", "G"), "kc_over_kf0": det("kc", "kf0")}
        va, vb = np.sqrt(d["kfx"]), np.sqrt(d["H"])
        rr = []
        for _ in range(B_MED):
            w = rng.multinomial(n, np.full(n, 1.0 / n)).astype(float)[idx]
            rr.append(wmedian(va, w) / wmedian(vb, w))
        o["KX5"] = [float(np.median(va) / np.median(vb)), *q(rr)]
        o["median_abs_eV"] = {"kfx": float(np.median(va)), "kf0": float(np.median(np.sqrt(d["kf0"]))),
                              "G": float(np.median(np.sqrt(d["G"]))), "H": float(np.median(vb)),
                              "A_mean_over_inits": float(np.mean([np.median(np.sqrt(x)) for x in d["A"]]))}
        o["rows"], o["discharges"] = int(len(idx)), int(n)
        o["params"] = json.loads((CACHE / f"s{s}.json").read_text(encoding="utf-8"))
        res["splits"][str(s)] = o
    # split by time (B.29 campaign split; causal variant cap 30, four initialisations; no twin)
    log("split by time ...")
    d = load_unit("camp")
    uf, idx = np.unique(d["shot"], return_inverse=True)
    n = len(uf)
    sums = {k: np.bincount(idx, d[k], n) for k in ("G", "H", "kf0", "kc", "kg", "kfx")}
    Ai = np.array([np.bincount(idx, v, n) for v in d["A"]])
    I = rng.integers(0, n, size=(B_SKILL, n))

    def detc(a, b):
        return [float(1 - sums[a].sum() / sums[b].sum()), *q(1 - sums[a][I].sum(1) / sums[b][I].sum(1))]
    ia = rng.integers(0, Ai.shape[0], B_SKILL)
    net = [float(1 - Ai.sum(1).mean() / sums["kfx"].sum()),
           *q(1 - np.take_along_axis(Ai[ia], I, 1).sum(1) / sums["kfx"][I].sum(1))]
    res["time_split"] = {"kfx_over_kf0": detc("kfx", "kf0"), "kg_over_kf0": detc("kg", "kf0"), "kf0_over_G": detc("kf0", "G"),
                         "kfx_over_G": detc("kfx", "G"), "net_over_kfx_init_drawn": net,
                         "median_abs_eV": {"kfx": float(np.median(np.sqrt(d["kfx"]))), "kf0": float(np.median(np.sqrt(d["kf0"]))),
                                           "G": float(np.median(np.sqrt(d["G"]))), "H": float(np.median(np.sqrt(d["H"]))),
                                           "A_mean_over_inits": float(np.mean([np.median(np.sqrt(x)) for x in d["A"]]))},
                         "rows": int(len(idx)), "discharges": int(n),
                         "params": json.loads((CACHE / "camp.json").read_text(encoding="utf-8"))}
    # parameters of every fold
    res["params"] = {u: json.loads((CACHE / f"{u}.json").read_text(encoding="utf-8")) for u in ALL_UNITS if u.startswith("r")}
    # statements
    C = res["confirm"]
    st = {"KX1": {"value": C["kfx_over_kf0"], "rule": "interval above 0", "holds": C["kfx_over_kf0"][1] > 0},
          "KX2": {"value": C["kg_over_kf0"], "rule": "interval above 0", "holds": C["kg_over_kf0"][1] > 0},
          "KX3": {"value": C["A_over_kfx"], "rule": "interval above 0", "holds": C["A_over_kfx"][1] > 0},
          "KX4": {"value": C["kf0_over_G_90"], "rule": "90 % interval inside [-0.02, +0.02]",
                  "holds": C["kf0_over_G_90"][1] > -0.02 and C["kf0_over_G_90"][2] < 0.02},
          "KX5": {"value": C["median_ratio_kfx_over_hold"], "rule": "interval below 1", "holds": C["median_ratio_kfx_over_hold"][2] < 1},
          "KX6": {"value": res["confirm_quiet"]["kfx_over_kf0"], "rule": "interval above 0",
                  "holds": res["confirm_quiet"]["kfx_over_kf0"][1] > 0}}
    for k in st:
        st[k]["holds"] = bool(st[k]["holds"])
    sp = res["splits"]
    st_s = {"KX1": sum(sp[str(s)]["KX1"][1] > 0 for s in SEEDS), "KX2": sum(sp[str(s)]["KX2"][1] > 0 for s in SEEDS),
            "KX3": sum(sp[str(s)]["KX3"][1] > 0 for s in SEEDS),
            "KX4": sum(sp[str(s)]["KX4_90"][1] > -0.02 and sp[str(s)]["KX4_90"][2] < 0.02 for s in SEEDS),
            "KX5": sum(sp[str(s)]["KX5"][2] < 1 for s in SEEDS), "KX6": sum(sp[str(s)]["KX6"][1] > 0 for s in SEEDS)}
    res["statements"] = st
    res["statements_four_splits"] = {k: {"of_four": int(v), "holds_3_of_4": bool(v >= 3)} for k, v in st_s.items()}
    OUT_JSON.write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    OUT_MD.write_text(render(res), encoding="utf-8", newline="\n")
    log(f"written {OUT_JSON} and {OUT_MD}")
    for k, v in st.items():
        log(f"{k}: {v['value'][0]:+.4f} [{v['value'][1]:+.4f}, {v['value'][2]:+.4f}] -> {'holds' if v['holds'] else 'does not hold'}")


def wmedian(v, w):
    o = np.argsort(v)
    cw = np.cumsum(w[o])
    return v[o][np.searchsorted(cw, 0.5 * cw[-1])]


def det_from(a_sum, b_sum, rng, n):
    I = rng.integers(0, n, size=(B_SKILL, n))
    return [float(1 - a_sum.sum() / b_sum.sum()), *q(1 - a_sum[I].sum(1) / b_sum[I].sum(1))]


# ------------------------------------------------------------------ report
def s3(x):
    return f"{x[0]:+.3f} [{x[1]:+.3f}, {x[2]:+.3f}]"


def r3(x):
    return f"{x[0]:.3f} [{x[1]:.3f}, {x[2]:.3f}]"


EXPECT = {"KX1": ("+0.03 to +0.06, holds", 0.03, 0.06, True), "KX2": ("+0.02 to +0.04, holds", 0.02, 0.04, True),
          "KX3": ("+0.02 to +0.06, holds", 0.02, 0.06, True), "KX4": ("-0.01 to +0.01, holds", -0.01, 0.01, True),
          "KX5": ("ratio 0.90 to 0.95, holds", 0.90, 0.95, True), "KX6": ("+0.03 to +0.055, holds", 0.03, 0.055, True)}


def render(res):
    C, st = res["confirm"], res["statements"]
    L = ["# B.56: a robust state-space filter with the beam-emission array as input", "",
         "Registration: `ces_prediction/experiments/PREREGISTRATION_B56.md` with Addendum 1. Runner: "
         "`ces_prediction/experiments/b56_kfx/run_b56.py`. CPU only; no network trained or run. T_i, cut population.", "",
         "Frozen specification (byte-identical copies of the development scripts):", "",
         f"- `kfx_model.py` SHA-256 `{res['sha256']['kfx_model.py']}` (only its definition part is executed)",
         f"- `kfx_components.py` SHA-256 `{res['sha256']['kfx_components.py']}` (the gain fit with a fixed clip, reimplemented in `fit_unit`)", "",
         "Declared forking paths (repeated from the registration and its addendum): (1) the agent saw the test error of a filter with an "
         "unclipped drive once on r2 (worse than KF0, -0.07 [-0.29, +0.03]) and then added the clip; (2) a joint fit of all five "
         "parameters was tried on r2 and abandoned because it degenerated in three of five folds. r2 is the development set and is shown "
         "beside the confirmation set (r3 to r9), not pooled into it. The look chose the clip per fold from {0.02, 0.05, 0.1, 0.2, none} "
         "(every r2 fold chose 0.2); the batch fixes it at 0.2.", "",
         "Fit in two stages: KF0 (q, r, k; Nelder-Mead from log 0.002, log 0.01, log 2, at most 250 iterations), then the gains of Kc, Kg "
         "(at most 200 iterations, as in `kfx_components.py`) and KFX (at most 300) from zero with q, r, k held. Training discharges: "
         "every discharge not in the test fold (on the splits: not in the split's test set).", "",
         "## Checks", ""]
    ch = res["checks"]
    L.append(f"- (a) r2 point estimates of section 2: {'pass' if ch['a']['passes'] else 'FAIL'}")
    for k, v in ch["a"]["skills"].items():
        L.append(f"  - {k}: look {v['look']:+.3f}, batch {v['batch']:+.4f}")
    for k, v in ch["a"]["medians_eV"].items():
        L.append(f"  - median {k}: look {v['look']} eV, batch {v['batch']:.2f} eV")
    nb = ch["b"]["units"]
    L.append(f"- (b) identical rows: {'pass' if ch['b']['passes'] else 'FAIL'}; {sum(v['rows_stored'] for v in nb.values())} stored rows "
             f"over {len(nb)} units, all located, max |grid value - stored value| {max(v['max_abs_value_diff_eV'] for v in nb.values()):.4f} eV; "
             "the located grid rows agree with B.54's independent location of the same rows on every pooled fold (B.54's grid "
             "row minus this batch's is 1 on every row: B.54 counts the first file row, this batch does not).")
    L.append(f"- Deviation in the location of the scored rows (declared): the frozen `map_rows` of `kfx_model.py` matches the stored "
             f"values with a tolerance of 0.01 eV and takes the first grid value within it; where two values of a discharge differ by "
             f"less than 0.01 eV it put a row one or more grid rows early ({sum((v.get('rows_moved_from_frozen_map') or 0) for v in nb.values())} "
             "rows of 1,694,675, all on the pooled folds, found by the cross-check against B.54). The batch locates rows with "
             "`map_rows_exact` (tolerance 0.0003 eV, closest value): the filter and its fit are unchanged (the fit does not use the "
             "location); only which grid row's output is scored changes. Check (a) gives the same numbers with either location.")
    c = ch["c"]
    L.append(f"- (c) causality: {'pass' if c['passes'] else 'FAIL'}; {c['perturbations']} perturbations of the CES values from row t on, "
             f"max change of the outputs up to row t {c['max_abs_change_rows_up_to_t_eV']} eV; later rows moved in {c['perturbations_where_later_rows_moved']}.")
    L += ["", "## Statements (re-partitions r3 to r9, two-level bootstrap, 20,000 draws; medians 2,000)", "",
          "Point estimates are the mean over the seven re-partitions of the skill of each (as in B.39 arm P); the median ratio is "
          "the mean over re-partitions of the ratio of the row medians, its interval from weighted medians under the same two-level "
          "resampling. Rows not flagged by the value rules: the `plain` column of B.54's row table of the same fold. Thirds: the "
          "discharges of a re-partition split by their median T_i (`data/.b54/files.npz`, ties by shot, as `run_b44.split_thirds`). "
          "Rows after an absent frame: the ten grid rows starting at the first row after a step of 15 ms to 0.5 s.", "",
          "| | Statement | Value [95 %] | Verdict | Expected | Inside the expected range |", "|---|---|---|---|---|---|"]
    for k, v in st.items():
        e = EXPECT[k]
        val = r3(v["value"]) if k == "KX5" else s3(v["value"])
        inside = e[1] <= v["value"][0] <= e[2]
        rule = v["rule"]
        L.append(f"| {k} | {rule} | {val} | {'holds' if v['holds'] else 'does not hold'} | {e[0]} | {'yes' if inside else 'no'}; "
                 f"verdict {'as' if v['holds'] == e[3] else 'not as'} expected |")
    L += ["", "KX1 KFX over KF0; KX2 Kg (radial difference only) over KF0; KX3 the network over KFX on the same rows; KX4 KF0 over the "
          "causal GP; KX5 median absolute error of KFX over that of hold; KX6 KFX over KF0 in rows not within ten rows after an absent frame.", "",
          "## Reported (r3 to r9)", "", "| Skill | All rows | Rows not after an absent frame | Rows not flagged by the value rules |", "|---|---|---|---|"]
    names = [("Tw_over_kf0", "twin over KF0"), ("kfx_over_Tw", "KFX over twin"), ("kc_over_kf0", "Kc over KF0"), ("kfx_over_kc", "KFX over Kc"),
             ("kfx_over_kg", "KFX over Kg"), ("kfx_over_G", "KFX over causal GP"), ("kfx_over_O", "KFX over offline GP"),
             ("A_over_G", "network over causal GP"), ("A_over_Tw", "network over twin"), ("kfx_over_kf0", "KFX over KF0"),
             ("kg_over_kf0", "Kg over KF0"), ("A_over_kfx", "network over KFX"), ("kf0_over_G", "KF0 over causal GP")]
    for key, nm in names:
        L.append(f"| {nm} | {s3(C[key])} | {s3(res['confirm_quiet'][key])} | {s3(res['confirm_plain'][key])} |")
    L += ["", f"Clip sensitivity (refits, reported only): KFX over KF0 at clip 0.1 {s3(C['kfx_clip0.1_over_kf0'])}, at 0.2 {s3(C['kfx_over_kf0'])}, "
          f"at 0.4 {s3(C['kfx_clip0.4_over_kf0'])}.", "",
          "Median absolute error (eV, mean over re-partitions): " +
          ", ".join(f"{k} {v:.1f}" for k, v in C["median_abs_eV"].items()) + " (A network, Tw twin, G causal GP, H hold, O offline GP).", "",
          "### Thirds of the discharges by median T_i (within a re-partition)", "",
          "| Third | KFX over KF0 | Kg over KF0 | network over KFX | twin over KF0 | KFX over causal GP | median KFX / KF0 / hold (eV) |", "|---|---|---|---|---|---|---|"]
    for tn in ("coldest", "middle", "hottest"):
        X = res[f"confirm_third_{tn}"]
        m = X["median_abs_eV"]
        L.append(f"| {tn} | {s3(X['kfx_over_kf0'])} | {s3(X['kg_over_kf0'])} | {s3(X['A_over_kfx'])} | {s3(X['Tw_over_kf0'])} | "
                 f"{s3(X['kfx_over_G'])} | {m['kfx']:.1f} / {m['kf0']:.1f} / {m['H']:.1f} |")
    D2 = res["dev_r2"]
    L += ["", "## Development set r2 (not pooled into the confirmation; intervals: its discharges resampled, 20,000 draws)", "",
          ", ".join(f"{nm} {s3(D2[key])}" for key, nm in names) + ".", "",
          f"Median ratio KFX / hold {r3(D2['median_ratio_kfx_over_hold'])}; medians: " + ", ".join(f"{k} {v:.1f}" for k, v in D2["median_abs_eV"].items()) + " eV.", "",
          "## Four file-level splits (secondary; statements read as at least three of four)", "",
          "KX3 and the twin use the three initialisations of B.39 with the registered interval of the splits (an initialisation drawn, then the "
          "test discharges resampled); the filters carry no initialisation (discharges resampled, 20,000 draws).", "",
          "| Split | KX1 | KX2 | KX3 | KX4 (90 %) | KX5 | KX6 | twin over KF0 | KFX over causal GP |", "|---|---|---|---|---|---|---|---|---|"]
    for s in SEEDS:
        o = res["splits"][str(s)]
        L.append(f"| {s} | {s3(o['KX1'])} | {s3(o['KX2'])} | {s3(o['KX3'])} | {s3(o['KX4_90'])} | {r3(o['KX5'])} | {s3(o['KX6'])} | "
                 f"{s3(o['twin_over_kf0'])} | {s3(o['kfx_over_G'])} |")
    L.append("")
    L.append("Four-split verdicts: " + "; ".join(f"{k} {v['of_four']}/4 ({'holds' if v['holds_3_of_4'] else 'does not hold'})"
                                                for k, v in res["statements_four_splits"].items()) + ". Expected: KX1 four of four; KX3 two to four of four.")
    L.append("")
    L.append("Network over KFX with the initialisation equal to the split seed only: " +
             ", ".join(f"{s} {s3(res['splits'][str(s)]['KX3_init_seed_only'])}" for s in SEEDS) + ".")
    ts = res["time_split"]
    L += ["", "## Split by time (B.29 campaign split; reported)", "",
          f"{ts['discharges']} test discharges, {ts['rows']} rows. KFX over KF0 {s3(ts['kfx_over_kf0'])}; Kg over KF0 {s3(ts['kg_over_kf0'])}; "
          f"KF0 over causal GP {s3(ts['kf0_over_G'])}; KFX over causal GP {s3(ts['kfx_over_G'])}; the causal variant of B.29 (epoch cap 30, four "
          f"initialisations, one drawn per resample) over KFX {s3(ts['net_over_kfx_init_drawn'])}. No twin exists on this split. Medians: " +
          ", ".join(f"{k} {v:.1f}" for k, v in ts["median_abs_eV"].items()) + " eV.", "",
          "## Fitted parameters", "", "| Unit | log q | log r | log k | KFX b_c | KFX b_g | Kc b_c | Kg b_g | KFX b at clip 0.1 | at 0.4 |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    allp = dict(res["params"])
    allp.update({f"s{s}": res["splits"][str(s)]["params"] for s in SEEDS})
    allp["camp"] = ts["params"]
    for u, p in allp.items():
        t0 = p["theta_kf0"]
        L.append(f"| {u} | {t0[0]:.3f} | {t0[1]:.3f} | {t0[2]:.3f} | {p['b_kfx'][0]:.3f} | {p['b_kfx'][1]:.3f} | {p['b_kc']:.3f} | {p['b_kg']:.3f} | "
                 f"{p['b_kfx_clip0.1'][0]:.3f}, {p['b_kfx_clip0.1'][1]:.3f} | {p['b_kfx_clip0.4'][0]:.3f}, {p['b_kfx_clip0.4'][1]:.3f} |")
    bc = [p["b_kfx"][0] for u, p in res["params"].items() if int(u[1]) in CONFIRM]
    bg = [p["b_kfx"][1] for u, p in res["params"].items() if int(u[1]) in CONFIRM]
    L += ["", f"Gains of KFX on the confirmation folds: b_c {min(bc):.3f} to {max(bc):.3f} (expected 0.4 to 0.6), "
          f"b_g {min(bg):.3f} to {max(bg):.3f} (expected 0.4 to 0.5).", ""]
    return "\n".join(L)


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "fit"
    if cmd == "fit":
        units = sys.argv[2:] or [u for u in ALL_UNITS if not (CACHE / f"{u}.npz").exists()]
        for u in units:
            fit_unit(u)
    elif cmd == "relocate":
        for u in ALL_UNITS:
            relocate(u)
    elif cmd == "checks":
        checks()
    elif cmd == "analyze":
        analyze()
    else:
        raise SystemExit(f"unknown command {cmd}")


if __name__ == "__main__":
    main()
