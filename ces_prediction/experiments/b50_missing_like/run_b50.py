"""B.50 (PREREGISTRATION_B50.md): the skill in scored rows that resemble the rows without a usable T_i.

CPU only; nothing is trained except the propensity model of B.37 F (HistGradientBoostingClassifier, unchanged).
Propensity p = run_b37.part_f's full model (same inputs, same seed, fitted on the non-test files of the split),
evaluated at every grid row of the test files with a ten-row window, joined to the k = 1 rows of the B.48 row
table (`data/.b48/rows_cut_s{seed}.npz`) by (file, grid row).

Run from the repo root:
    py ces_prediction/experiments/b50_missing_like/run_b50.py
Cache: data/.b50/ (propensity per split).  Output: docs/paper/b50_missing_like.json / .md
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
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
for sub in ("b30_thinned", "b37_plan", "b45_ablation_rows"):
    sys.path.insert(0, str(EXP / sub))
import run_b30 as R  # noqa: E402
import run_b37 as R37  # noqa: E402
from run_b45 import ci  # noqa: E402

REG = "ces_prediction/experiments/PREREGISTRATION_B50.md"
ROWS = R.DATA / ".b48"
CACHE = R.DATA / ".b50"
PAPER = R.REPO / "docs" / "paper"
OUT_JSON = PAPER / "b50_missing_like.json"
OUT_MD = PAPER / "b50_missing_like.md"
SEED, N_BOOT = 20261118, 2000            # bootstrap generator: default_rng(SEED + split), as run_b45
EST = ("causal", "hist80", "gp", "hold")
REFS = ("gp", "hist80", "hold")
LAB = {"gp": "causal GP", "hist80": "twin", "hold": "hold", "causal": "estimator"}
AUC_TOL, Y_TOL, B48_TOL = 0.005, 1e-3, 5e-4
T0 = _time.time()


def log(m):
    print(f"[{_time.time() - T0:6.0f}s] {m}", flush=True)


def load_tab(s):
    z = np.load(ROWS / f"rows_cut_s{s}.npz")
    return {c.split("__")[1]: z[c] for c in z.files if c.startswith("CES_TI__")}


# ------------------------------------------------------------------------------------------ propensity
def propensity(C, s):
    """run_b37.part_f, full model only, lines copied unchanged; additionally keeps (file, grid row) of every
    test row so that p can be joined to the row table."""
    p_cache = CACHE / f"p_s{s}.npz"
    if p_cache.exists():
        z = np.load(p_cache)
        return {k: z[k] for k in z.files}
    test = set(C.names.index(n) for n in C.manifest(s)["test_files"] if n in C.grid)
    mu, sd = C.fast_stats(test)
    parts = {True: [], False: []}
    for i in range(len(C.names)):
        y = C.grid[C.names[i]][:, 1].astype(np.float64)
        u = np.isnan(y).astype(float)
        n = len(u)
        fr = C.feats(i, mu, sd)
        since = np.zeros(n)
        for j in range(1, n):
            since[j] = 0 if u[j - 1] == 0 else min(50, since[j - 1] + 1)
        ok = np.flatnonzero(np.isfinite(fr).all(1))
        hist = np.column_stack([*(u[ok - k] for k in range(1, 6)), fr[ok, 8], since[ok]])
        parts[i in test].append((np.column_stack([hist, fr[ok, :8]]), u[ok], np.full(len(ok), i), ok))
    Xf, yf = np.concatenate([p[0] for p in parts[False]]), np.concatenate([p[1] for p in parts[False]])
    Xt, yt = np.concatenate([p[0] for p in parts[True]]), np.concatenate([p[1] for p in parts[True]])
    ft, rt = np.concatenate([p[2] for p in parts[True]]), np.concatenate([p[3] for p in parts[True]])
    pf = HistGradientBoostingClassifier(random_state=R37.SEED).fit(Xf, yf).predict_proba(Xt)[:, 1]
    out = {"p": pf, "u": yt, "fidx": ft.astype(np.int64), "row": rt.astype(np.int64),
           "fit_rows": np.array(len(yf)), "test_files": np.array(sorted(test))}
    np.savez_compressed(p_cache, **out)
    return out


# ------------------------------------------------------------------------------------------ readout
def cells(se, f, n, W, mask, w=None):
    """Skill of the estimator over each reference on the rows `mask` (row weights w), discharge resamples W."""
    ww = np.ones(len(f)) if w is None else w
    o = {}
    a = np.bincount(f[mask], (ww * se["causal"])[mask], n)
    for r in REFS:
        b = np.bincount(f[mask], (ww * se[r])[mask], n)
        o[r] = [float(1 - a.sum() / b.sum()), *ci(1 - (W @ a) / (W @ b))]
    return o


def split(C, s, b37, b48):
    t = _time.time()
    P = propensity(C, s)
    auc = float(roc_auc_score(P["u"], P["p"]))
    auc_b37 = float(b37[str(s)]["auc_full"])
    chk = {"a": {"auc": auc, "auc_b37": auc_b37, "diff": auc - auc_b37, "test_rows": int(len(P["u"])),
                 "test_rows_b37": int(b37[str(s)]["test_rows"]), "passes": bool(abs(auc - auc_b37) <= AUC_TOL)}}
    log(f"s{s}: propensity done ({_time.time() - t:.0f}s); AUC {auc:.4f} vs B.37 {auc_b37:.4f}")
    if not chk["a"]["passes"]:
        return {"checks": chk, "stopped": "check (a) fails"}

    tab = load_tab(s)
    sc = (tab["k"] == 1) & np.all([np.isfinite(tab[e]) for e in (*EST, "y")], axis=0)
    fi, ri = tab["file"][sc].astype(np.int64), tab["row"][sc].astype(np.int64)
    # join check: grid T_i at (file, row) equals y of the table on every scored row
    yg = np.array([C.grid[C.names[a]][b, 1] for a, b in zip(fi, ri)], dtype=np.float64)
    dy = np.abs(yg - tab["y"][sc].astype(np.float64))
    chk["join"] = {"scored_rows": int(sc.sum()), "nan_in_grid": int(np.isnan(yg).sum()),
                   "max_abs_diff_eV": float(np.nanmax(dy)), "rows_above_tol": int((~(dy <= Y_TOL)).sum()),
                   "files_outside_test": int((~np.isin(fi, P["test_files"])).sum())}
    chk["join"]["passes"] = bool(chk["join"]["rows_above_tol"] == 0 and chk["join"]["files_outside_test"] == 0)
    log(f"s{s}: join {chk['join']}")
    if not chk["join"]["passes"]:
        return {"checks": chk, "stopped": "join check fails"}
    key = {(int(a), int(b)): j for j, (a, b) in enumerate(zip(P["fidx"], P["row"]))}
    jj = np.array([key.get((int(a), int(b)), -1) for a, b in zip(fi, ri)])
    has = jj >= 0
    # check (b), Addendum 1: every scored row with ten rows before it in its segment receives a p; the scored rows
    # without such a window (rows 2 to 10 of a segment) are outside the population and are counted
    pos = np.array([r - np.flatnonzero(C.seg[a] == C.seg[a][r])[0] for a, r in zip(fi, ri)])
    win = pos >= 10
    chk["b"] = {"rule": "Addendum 1", "scored_rows": int(len(jj)), "with_window": int(win.sum()),
                "with_window_and_p": int((win & has).sum()), "with_window_without_p": int((win & ~has).sum()),
                "without_window_with_p": int((~win & has).sum()),
                "outside_population": int((~win).sum()), "outside_share": float((~win).mean()),
                "outside_discharges": int(len(np.unique(fi[~win]))),
                "share_with_p_old_rule": float(has.mean())}
    chk["b"]["passes"] = bool(chk["b"]["with_window_without_p"] == 0 and chk["b"]["without_window_with_p"] == 0)
    log(f"s{s}: check (b) {chk['b']}")
    if not chk["b"]["passes"]:
        return {"checks": chk, "stopped": "check (b) fails"}
    # the joined rows must be usable rows in the propensity table
    chk["b"]["joined_rows_marked_unusable"] = int(P["u"][jj[has]].sum())

    sel = np.flatnonzero(sc)[has]
    p = P["p"][jj[has]].astype(np.float64)
    y = tab["y"][sel].astype(np.float64)
    se = {e: (tab[e][sel].astype(np.float64) - y) ** 2 for e in EST}
    age = tab["age_ms"][sel].astype(np.float64)
    uf, f = np.unique(tab["file"][sel].astype(np.int64), return_inverse=True)
    n = len(uf)
    W = np.random.default_rng(SEED + s).multinomial(n, np.full(n, 1.0 / n), size=N_BOOT).astype(np.float64)
    allr = np.ones(len(y), bool)

    # cross-check: unstratified skill against docs/paper/b48_reread.json (B.44 overall at k = 1, same rows)
    full_sc = np.all([np.isfinite(tab[e]) for e in (*EST, "y")], axis=0) & (tab["k"] == 1)
    yy = tab["y"][full_sc].astype(np.float64)
    pt_all = {r: float(1 - ((tab["causal"][full_sc] - yy) ** 2).sum() / ((tab[r][full_sc] - yy) ** 2).sum())
              for r in REFS}
    ref48 = b48["b44"]["splits"][str(s)]["overall"]
    chk["b48"] = {r: {"mine": pt_all[r], "b48": float(ref48[r][0]), "equal_3dp": bool(abs(pt_all[r] - ref48[r][0]) < B48_TOL)}
                  for r in REFS}
    chk["b48"]["passes"] = bool(all(chk["b48"][r]["equal_3dp"] for r in REFS))
    pt_pop = {r: float(1 - se["causal"].sum() / se[r].sum()) for r in REFS}
    chk["b48_population"] = {r: {"population": pt_pop[r], "b48": float(ref48[r][0]), "diff": pt_pop[r] - float(ref48[r][0])}
                             for r in REFS}
    log(f"s{s}: B.48 cross-check {chk['b48']['passes']}")
    if not chk["b48"]["passes"]:
        return {"checks": chk, "stopped": "cross-check against B.48 fails"}

    o = {"checks": chk, "rows": int(len(y)), "discharges": n, "overall_joined": cells(se, f, n, W, allr)}
    # fifths of p among the scored rows: edges = 20/40/60/80 % quantiles; fifth j holds e_{j-1} < p <= e_j
    edges = np.quantile(p, [0.2, 0.4, 0.6, 0.8])
    q = np.searchsorted(edges, p, side="left")
    unus = P["u"] == 1
    pu = P["p"][unus]
    qu = np.searchsorted(edges, pu, side="left")
    med_u = float(np.median(pu))
    o["edges"] = edges.tolist()
    o["unusable_test_rows"] = int(unus.sum())
    o["median_p_unusable"] = med_u
    o["median_p_scored"] = float(np.median(p))
    o["fifths"] = []
    for j in range(5):
        m = q == j
        o["fifths"].append({
            "rows": int(m.sum()), "discharges": int(len(np.unique(f[m]))),
            "p_range": [float(p[m].min()), float(p[m].max())],
            "share_of_unusable": float((qu == j).mean()),
            "skill": cells(se, f, n, W, m),
            "median_age_ms": float(np.median(age[m])), "median_ti_eV": float(np.median(y[m])),
            "rmse": {e: float(np.sqrt(se[e][m].mean())) for e in EST}})
    m = p > med_u
    o["above_median_unusable"] = {
        "rows": int(m.sum()), "row_share": float(m.mean()), "discharges": int(len(np.unique(f[m]))),
        "skill": cells(se, f, n, W, m) if m.any() else None,
        "median_age_ms": float(np.median(age[m])) if m.any() else None,
        "median_ti_eV": float(np.median(y[m])) if m.any() else None,
        "rmse": {e: float(np.sqrt(se[e][m].mean())) for e in EST} if m.any() else None}
    w = p / (1 - p)
    clip = float(np.quantile(w, 0.99))
    w = np.minimum(w, clip)
    o["weighted"] = {"clip_w": clip, "rows_clipped": int((p / (1 - p) > clip).sum()),
                     "effective_rows": float(w.sum() ** 2 / (w ** 2).sum()),
                     "effective_share": float(w.sum() ** 2 / (w ** 2).sum() / len(w)),
                     "skill": cells(se, f, n, W, allr, w)}
    top = o["fifths"][4]["skill"]
    o["P1"] = bool(top["gp"][1] > 0)
    o["P2"] = bool(top["hist80"][1] > 0)
    o["P3"] = bool(o["weighted"]["skill"]["gp"][1] > 0)
    o["P4"] = bool(o["weighted"]["skill"]["hist80"][1] > 0)
    o["runtime_s"] = round(_time.time() - t, 1)
    log(f"s{s}: P1 {top['gp']} P2 {top['hist80']} P3 {o['weighted']['skill']['gp']} "
        f"P4 {o['weighted']['skill']['hist80']} ESS {o['weighted']['effective_share']:.3f}")
    return o


# ------------------------------------------------------------------------------------------ expectations
def expectations(res):
    S = [str(s) for s in R.SEEDS]
    v = lambda fn: [fn(res[s]) for s in S]                                      # noqa: E731
    E = []

    def rng_row(desc, vals, lo, hi):
        E.append({"expectation": desc, "observed": ", ".join(f"{x:+.3f}" for x in vals),
                  "met": bool(all(lo <= x <= hi for x in vals))})

    E.append({"expectation": "P1 holds", "observed": f"{sum(v(lambda r: r['P1']))} of 4",
              "met": bool(sum(v(lambda r: r["P1"])) >= 3)})
    rng_row("top fifth, skill over the causal GP +0.03 to +0.15", v(lambda r: r["fifths"][4]["skill"]["gp"][0]), 0.03, 0.15)
    rng_row("top fifth, increment over the twin -0.02 to +0.08", v(lambda r: r["fifths"][4]["skill"]["hist80"][0]), -0.02, 0.08)
    n2 = sum(v(lambda r: r["P2"]))
    E.append({"expectation": "P2: interval above zero on one to three splits", "observed": f"{n2} of 4",
              "met": bool(1 <= n2 <= 3)})
    rng_row("odds-weighted skill over the causal GP +0.03 to +0.15", v(lambda r: r["weighted"]["skill"]["gp"][0]), 0.03, 0.15)
    E.append({"expectation": "P3 holds", "observed": f"{sum(v(lambda r: r['P3']))} of 4",
              "met": bool(sum(v(lambda r: r["P3"])) >= 3)})
    rng_row("odds-weighted increment over the twin 0.00 to +0.08", v(lambda r: r["weighted"]["skill"]["hist80"][0]), 0.0, 0.08)
    rng_row("top fifth holds 0.70 to 0.95 of the unusable test rows", v(lambda r: r["fifths"][4]["share_of_unusable"]), 0.70, 0.95)
    rng_row("effective number of rows 3 to 15 % of the scored rows", v(lambda r: r["weighted"]["effective_share"]), 0.03, 0.15)
    return E


CHOICES = [
    "Propensity: the lines of `run_b37.part_f` that build the inputs and fit the full model are copied unchanged "
    "(same feature construction through `run_b37.Ctx.feats`, same target `isnan(T_i)` on the cut, held-free grid, "
    "`HistGradientBoostingClassifier(random_state=20261116)`, fitted on the rows of the non-test files); the only "
    "addition is that the grid row of every test row is kept. The history-only model of B.37 F is not fitted (it "
    "is not part of B.50).",
    "Join: a scored row of the B.48 table (k = 1; causal, twin, hold, causal GP and y finite) is matched to the "
    "propensity table by (file index into the sorted grid, grid row). Before the readout, the grid T_i at that "
    "(file, row) is compared with `y` of the table on every scored row (tolerance 1e-3 eV); the run stops if one "
    "row differs.",
    "Fifths: edges are the 20, 40, 60 and 80 % quantiles (linear interpolation) of p over the scored rows with a p; "
    "fifth j holds the rows with e_(j-1) < p <= e_j. The same edges place the unusable test rows into fifths.",
    "Unusable test rows: rows of the test files with a ten-row window whose T_i is not usable (the target of the "
    "propensity model, i.e. NaN after the 3 keV cut and the removal of held values).",
    "Weights: w = p / (1 - p) clipped at the 99th percentile of w over the scored rows of the split.",
    "Bootstrap: 2,000 multinomial resamples of the test discharges from `numpy.random.default_rng(20261118 + split)` "
    "(as `run_b45`), one set per split shared by every cell; per-discharge sums of (weighted) squared errors; "
    "interval = 2.5 and 97.5 % quantiles (`run_b45.ci`). A statement holds on a split when the lower end is above zero.",
    "Cross-check before the readout: the unstratified skills (estimator over the twin, the causal GP and hold) on "
    "every scored k = 1 row equal the B.48 values (`docs/paper/b48_reread.json`, B.44 overall) to three decimals "
    "(stop rule); the same skills on the population (rows with a ten-row window) are reported beside them.",
    "Position in segment: segments as in `run_b37.Ctx` (a new segment after a time step of at least 0.5 s); a row "
    "has a ten-row window when it is at position 10 or later (0-based) in its segment, as `run_b33.window_feats` "
    "requires.",
]


# ------------------------------------------------------------------------------------------ render
def s3(x):
    return f"{x[0]:+.3f} [{x[1]:+.3f}, {x[2]:+.3f}]"


def hs(h):
    return "holds" if h else "does not hold"


def render(o):
    S = [str(s) for s in R.SEEDS]
    res = o["splits"]
    L = ["# B.50 -- the skill in scored rows that resemble the rows without a usable T_i (registered: "
         "ces_prediction/experiments/PREREGISTRATION_B50.md)", "",
         "CPU, no network trained or run. T_i, cut population, the four file-level splits (42, 1, 7, 123), test "
         "discharges. Rows: the k = 1 rows of the B.48 row table (`data/.b48/rows_cut_s{seed}.npz`) in which the "
         "estimator of B.39 (`causal`), the history-only twin (`hist80`), hold and the causal GP are finite. Propensity "
         "p: the full model of B.37 F (probability that T_i is not usable in the row, from the history flags and the "
         "fast-channel window), fitted on the files that are not test files of the split. Skill of the estimator over "
         "R = 1 - SSE_est / SSE_R on the same rows (weighted: 1 - sum w*se_est / sum w*se_R). Intervals: 95 %, 2,000 "
         "resamples of the test discharges; strata and weights fixed on the full test set. A statement holds on a "
         "split when the lower end of its interval is above zero, and holds when that is so on at least three of "
         "four splits.", "",
         "## Checks before the readout", "",
         "| split | (a) AUC of p, test files (mine / B.37 F) | (a) | join: scored rows, grid T_i vs y max abs diff (eV), rows > 1e-3 | "
         "(b) scored rows with a ten-row window: with a p / all (without p) | (b) scored rows without the window (outside the population; share; discharges) | (b) | "
         "unstratified skill on every scored row vs B.48 (twin / GP / hold, mine = B.48) |",
         "|---|---|---|---|---|---|---|---|"]
    for s in S:
        c = res[s]["checks"]
        a, j, b, x = c["a"], c.get("join"), c.get("b"), c.get("b48")
        L.append(f"| {s} | {a['auc']:.4f} / {a['auc_b37']:.4f} ({a['diff']:+.4f}) | {'passes' if a['passes'] else 'FAILS'} | "
                 + (f"{j['scored_rows']:,}, {j['max_abs_diff_eV']:.1e}, {j['rows_above_tol']}" if j else "n/a") + " | "
                 + (f"{b['with_window_and_p']:,} / {b['with_window']:,} ({b['with_window_without_p']})" if b else "n/a") + " | "
                 + (f"{b['outside_population']:,} ({b['outside_share']:.4f}; {b['outside_discharges']})" if b else "n/a")
                 + f" | {('passes' if b['passes'] else 'FAILS') if b else 'n/a'} | "
                 + (" / ".join(f"{x[r]['mine']:+.4f} = {x[r]['b48']:+.4f}" for r in ("hist80", "gp", "hold")) if x else "n/a")
                 + " |")
    stopped = [s for s in S if "stopped" in res[s]]
    if stopped:
        L += ["", "**The run stopped: " + "; ".join(f"split {s}: {res[s]['stopped']}" for s in stopped) + ".** "
              "No readout (strata, weights, skills, statements) was computed.", ""]
        if False:
            L += ["Where the scored rows without a p sit (diagnosis of check (b), not a readout): "
                  + "; ".join(f"split {s}: {res[s]['checks']['b']['dropped']} rows in "
                              f"{res[s]['checks']['b']['dropped_discharges']} discharges, share in the first ten rows of "
                              f"their segment {res[s]['checks']['b']['dropped_share_in_first_ten_rows']:.3f} (largest "
                              f"position {res[s]['checks']['b']['dropped_position_in_segment_max']})" for s in S)
                  + ". The propensity model of B.37 F needs the ten rows before a row (`run_b33.window_feats`), so the "
                  "scored rows in rows 1 to 9 of a segment have no p. These rows are excluded by construction of the "
                  "registered propensity, not by a failed join (the join check passes on every row).", ""]
        L += ["## Deviations", "", o["deviations"], "", f"Run time: {o['runtime_s']:.0f} s."]
        return "\n".join(L) + "\n"
    L += ["", "All checks pass on every split. Scored rows marked unusable in the propensity table (must be 0): "
          + ", ".join(str(res[s]["checks"]["b"]["joined_rows_marked_unusable"]) for s in S) + ".", "",
          "Unstratified skill (point values) on the population of this readout (scored rows from the eleventh row "
          "of a segment on) against the B.48 value on every scored row (twin / causal GP / hold; population minus "
          "B.48): " + "; ".join(f"split {s}: " + " / ".join(
              f"{res[s]['checks']['b48_population'][r]['population']:+.4f} vs {res[s]['checks']['b48_population'][r]['b48']:+.4f} "
              f"({res[s]['checks']['b48_population'][r]['diff']:+.4f})" for r in ("hist80", "gp", "hold")) for s in S) + ".", "",
          "## Registered statements", "", "Per-split values in the order 42, 1, 7, 123.", ""]
    V = o["verdicts"]
    L.append("- **P1** (top fifth of p, skill over the causal GP, interval above zero): "
             + "; ".join(s3(res[s]["fifths"][4]["skill"]["gp"]) for s in S) + f"; {V['P1']['splits']} of 4 -> **{hs(V['P1']['holds'])}**.")
    L.append("- **P2** (top fifth of p, skill over the twin, interval above zero): "
             + "; ".join(s3(res[s]["fifths"][4]["skill"]["hist80"]) for s in S) + f"; {V['P2']['splits']} of 4 -> **{hs(V['P2']['holds'])}**.")
    L.append("- **P3** (rows weighted by the odds of being unusable, skill over the causal GP, interval above zero): "
             + "; ".join(s3(res[s]["weighted"]["skill"]["gp"]) for s in S) + f"; {V['P3']['splits']} of 4 -> **{hs(V['P3']['holds'])}**.")
    L.append("- **P4** (the same over the twin): "
             + "; ".join(s3(res[s]["weighted"]["skill"]["hist80"]) for s in S) + f"; {V['P4']['splits']} of 4 -> **{hs(V['P4']['holds'])}**.")
    L += ["", f"Outcome under section 5 of the registration: {o['outcome']}", "",
          "## Reported", "",
          "### Skill by fifth of p", "",
          "| split | fifth | p range | rows | discharges | share of the unusable test rows | over causal GP | over twin | over hold |",
          "|---|---|---|---|---|---|---|---|---|"]
    for s in S:
        for j, q in enumerate(res[s]["fifths"]):
            L.append(f"| {s} | {j + 1} | {q['p_range'][0]:.4f} to {q['p_range'][1]:.4f} | {q['rows']:,} | {q['discharges']} | "
                     f"{q['share_of_unusable']:.3f} | " + " | ".join(s3(q["skill"][r]) for r in REFS) + " |")
        a = res[s]["above_median_unusable"]
        L.append(f"| {s} | p > median p of the unusable rows ({res[s]['median_p_unusable']:.4f}) | | {a['rows']:,} "
                 f"(share {a['row_share']:.4f}) | {a['discharges']} | | "
                 + (" | ".join(s3(a["skill"][r]) for r in REFS) if a["skill"] else "n/a | n/a | n/a") + " |")
    L += ["", "### Rows of each fifth", "",
          "| split | fifth | median age of the last visible value (ms) | median T_i (eV) | RMSE estimator | RMSE twin | RMSE causal GP | RMSE hold |",
          "|---|---|---|---|---|---|---|---|"]
    for s in S:
        for j, q in enumerate(res[s]["fifths"]):
            L.append(f"| {s} | {j + 1} | {q['median_age_ms']:.1f} | {q['median_ti_eV']:.1f} | "
                     + " | ".join(f"{q['rmse'][e]:.1f}" for e in EST) + " |")
        a = res[s]["above_median_unusable"]
        if a["rmse"]:
            L.append(f"| {s} | above median p of unusable | {a['median_age_ms']:.1f} | {a['median_ti_eV']:.1f} | "
                     + " | ".join(f"{a['rmse'][e]:.1f}" for e in EST) + " |")
    L += ["", "### Odds-weighted population", "",
          "| split | scored rows | unusable test rows | median p scored / unusable | clip of w | rows clipped | effective rows (share) | over causal GP | over twin | over hold | unweighted, same rows: over causal GP / twin |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for s in S:
        r_, wt = res[s], res[s]["weighted"]
        L.append(f"| {s} | {r_['rows']:,} | {r_['unusable_test_rows']:,} | {r_['median_p_scored']:.4f} / {r_['median_p_unusable']:.4f} | "
                 f"{wt['clip_w']:.3f} | {wt['rows_clipped']} | {wt['effective_rows']:.0f} ({wt['effective_share']:.3f}) | "
                 + " | ".join(s3(wt["skill"][r]) for r in REFS) + f" | {s3(r_['overall_joined']['gp'])} / {s3(r_['overall_joined']['hist80'])} |")
    L += ["", "## Expectations of the registration (section 6)", "", "| expectation | observed | met |", "|---|---|---|"]
    for e in o["expectations"]:
        L.append(f"| {e['expectation']} | {e['observed']} | {'yes' if e['met'] else 'no'} |")
    L += ["", f"{sum(e['met'] for e in o['expectations'])} of {len(o['expectations'])} expectations met.", "",
          "## Choices made where the registration left room", ""] + [f"- {c}" for c in CHOICES]
    L += ["", "## Deviations", "", o["deviations"], "",
          "## Limits", "", "Resemblance is measured in the inputs of the propensity model only; the unusable rows may "
          "differ from the scored rows in ways no input records (a larger fit error, for one). The weights concentrate "
          "on few rows. One initialisation (the split seed). The test rows are those of every earlier readout. "
          "(Addendum 1) The first rows after a break of at least 0.5 s (rows 2 to 10 of a segment, which have no "
          "ten-row window) are not covered.", "",
          f"Run time: {o['runtime_s']:.0f} s."]
    return "\n".join(L) + "\n"


def outcome(V):
    if V["P1"]["holds"] and V["P2"]["holds"] and V["P3"]["holds"] and V["P4"]["holds"]:
        return "P1 to P4 hold."
    if V["P1"]["holds"] and V["P3"]["holds"]:
        return ("P1 and P3 hold, P2 or P4 does not: in rows resembling the unusable ones the estimator keeps its "
                "margin over the causal GP while the fast-channel increment is not established there.")
    return "P1 or P3 does not hold: stated as a limit of the claim."


DEVIATIONS = ("Addendum 1 of the registration (2026-11-18, committed in f276233 before any readout was computed): "
              "the first run stopped at check (b) on all four splits (0.9759, 0.9778, 0.9768, 0.9778 of the scored rows "
              "received a p, against the threshold of 0.98); every row without a p was row 2 to 10 of its segment, "
              "where the inputs of B.37 F have no ten-row window. Check (b) is restated: every scored row with ten rows "
              "before it in its segment receives a p (the run stops otherwise); the scored rows without such a window "
              "are outside the population and are counted. No other change from the registered definitions, rows, "
              "intervals, seeds or rules. The cross-check against B.48 is made on every scored row (the same row set "
              "as B.48, a stop rule); the unstratified skill on the population of this readout is reported beside it.")


def main():
    CACHE.mkdir(exist_ok=True)
    b37 = json.loads((PAPER / "b37_plan.json").read_text(encoding="utf-8"))["F"]
    b48 = json.loads((PAPER / "b48_reread.json").read_text(encoding="utf-8"))
    C = R37.Ctx()
    log(f"grid loaded: {len(C.names)} files")
    res = {}
    for s in R.SEEDS:
        res[str(s)] = split(C, s, b37, b48)
        (CACHE / "partial.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    o = {"registration": REG, "seed": SEED, "n_boot": N_BOOT, "splits": res, "deviations": DEVIATIONS,
         "choices": CHOICES}
    if not any("stopped" in r for r in res.values()):
        V = {k: {"splits": int(sum(res[str(s)][k] for s in R.SEEDS))} for k in ("P1", "P2", "P3", "P4")}
        for k in V:
            V[k]["holds"] = bool(V[k]["splits"] >= 3)
        o["verdicts"] = V
        o["outcome"] = outcome(V)
        o["expectations"] = expectations(res)
    o["runtime_s"] = round(_time.time() - T0, 1)
    OUT_JSON.write_text(json.dumps(o, indent=1), encoding="utf-8")
    OUT_MD.write_text(render(o), encoding="utf-8", newline="\n")
    log("written " + str(OUT_MD))


if __name__ == "__main__":
    main()
