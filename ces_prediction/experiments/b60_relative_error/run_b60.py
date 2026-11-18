"""B.60 (PREREGISTRATION_B60.md): the error as a share of the reported value -- how many frames are within 10 % and
within 20 % -- on the rows of the pooled cross-validation of B.39 (the row tables of B.54).

CPU only; no network is run.  Imports the row tables, the thirds/flag/blip definitions (as cached by B.54) and the
generator convention from b54_pooled_rows/run_b54.py; run_b54 must have run its stages files, rows, prop, checks.

    py ces_prediction/experiments/b60_relative_error/run_b60.py
Output: docs/paper/b60_relative_error.json / .md
"""
from __future__ import annotations

import json
import sys
import time as _time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "b54_pooled_rows"))
import run_b54 as B54  # noqa: E402

B41, B44 = B54.B41, B54.B44
REG = "ces_prediction/experiments/PREREGISTRATION_B60.md"
OUT_JSON = B54.PAPER / "b60_relative_error.json"
OUT_MD = B54.PAPER / "b60_relative_error.md"
REPEATS = B54.REPEATS
N_BOOT = 2000
EST = ("c", "t", "gp", "hold", "off")
LAB = B54.LAB
THRESH = (0.10, 0.20)
SEC2 = {"c": (46.0, 100.6, 205.9), "gp": (44.1, 104.0, 217.6), "hold": (45.4, 111.4, 254.7),
        "off": (37.5, 91.5, 198.0), "t": (46.1, 103.8, 216.0)}
LOW_EV = 50.0
T0 = _time.time()


def log(m):
    print(f"[{_time.time() - T0:6.0f}s] {m}", flush=True)


def thirds_lab(D, F, r):
    """Thirds of the discharges of a repeat by median T_i: the lines of run_b54.stage_readout (run_b44 code)."""
    u = D[r]["u"]
    v = F["median_ti"][u]
    have = np.flatnonzero(np.isfinite(v))
    parts = [have[p] for p in B44.split_thirds(v[have], F["shot"][u][have])]
    lab = np.full(D[r]["n"], -1)
    for j, idx in enumerate(parts):
        lab[idx] = j
    return lab


def cell(D, rel, mask_fn, rng):
    """Shares within 10 % / 20 % and median relative error for every estimator on rows mask_fn(r), with the
    two-level resampling of run_b54.hier (a repeat drawn, then its discharges); every quantity and every difference
    is computed on the same draw.  Point = mean over repeats."""
    draws = rng.choice(np.array(REPEATS), size=N_BOOT)
    bs = {}
    pts = {}
    for r in REPEATS:
        d = D[r]
        m = mask_fn(r)
        f, n = d["f"], d["n"]
        sel = np.flatnonzero(draws == r)
        idx = rng.integers(0, n, size=(len(sel), n))
        W = np.zeros((len(sel), n))
        for j in range(len(sel)):
            W[j] = np.bincount(idx[j], minlength=n)
        cnt = np.bincount(f[m], None, n).astype(float)
        wc = W @ cnt
        p_r = {}
        for e in EST:
            x = rel[r][e]
            for th in THRESH:
                key = f"{e}|{th:.2f}"
                num = np.bincount(f[m], (x <= th)[m].astype(float), n)
                bs.setdefault(key, np.empty(N_BOOT))[sel] = (W @ num) / wc
                p_r[key] = float(num.sum() / cnt.sum())
            key = f"{e}|median"
            Wm = W[:, np.unique(f[m])] if False else W
            # weighted median over the rows of the cell (run_b41.wquant; rows outside the cell are dropped)
            fm = f[m]
            bs.setdefault(key, np.empty(N_BOOT))[sel] = B41.wquant(x[m], fm, Wm, [0.5])[:, 0]
            p_r[key] = float(np.median(x[m]))
        pts[r] = p_r
        pts[r]["rows"] = int(m.sum())
        pts[r]["discharges"] = int(len(np.unique(f[m])))
    out = {"rows_per_repeat": {str(r): pts[r]["rows"] for r in REPEATS},
           "discharges_per_repeat": {str(r): pts[r]["discharges"] for r in REPEATS},
           "row_share": float(np.mean([mask_fn(r).mean() for r in REPEATS])), "value": {}, "diff": {}}
    keys = [f"{e}|{q}" for e in EST for q in ("0.10", "0.20", "median")]
    for key in keys:
        out["value"][key] = [float(np.mean([pts[r][key] for r in REPEATS])), *B54.ci(bs[key])]
        out["value"][key + "|per_repeat"] = {str(r): pts[r][key] for r in REPEATS}
    for ref in ("gp", "hold", "t", "off"):
        for q in ("0.10", "0.20", "median"):
            a, b = f"c|{q}", f"{ref}|{q}"
            out["diff"][f"c-{ref}|{q}"] = [float(np.mean([pts[r][a] - pts[r][b] for r in REPEATS])),
                                          *B54.ci(bs[a] - bs[b])]
            out["diff"][f"c-{ref}|{q}|per_repeat"] = {str(r): pts[r][a] - pts[r][b] for r in REPEATS}
    return out


def main():
    F = B54.load_files()
    D = {r: B54.repeat_data(r) for r in REPEATS}
    log("repeats loaded")
    # ---------------- check: section 2 percentiles of the absolute error, to one decimal
    chk = {}
    for e, ref in SEC2.items():
        mine = [float(np.mean([np.percentile(np.sqrt(D[r]["se"][e]), p) for r in REPEATS])) for p in (50, 75, 90)]
        chk[LAB[e]] = {"mine": mine, "registered": list(ref),
                       "equal_1dp": bool(all(round(a, 1) == b for a, b in zip(mine, ref)))}
    passes = all(v["equal_1dp"] for v in chk.values())
    log(f"check: {passes} " + "; ".join(f"{k} {[round(x, 2) for x in v['mine']]}" for k, v in chk.items()))
    if not passes:
        OUT_JSON.write_text(json.dumps({"check": chk, "stopped": True}, indent=1), encoding="utf-8")
        raise SystemExit("FATAL: the check fails; no statement is read")
    # ---------------- relative error (definition of the registration, unchanged)
    rel = {r: {e: np.sqrt(D[r]["se"][e]) / D[r]["y"] for e in EST} for r in REPEATS}
    ally = np.concatenate([D[r]["y"] for r in REPEATS])
    low = {"rows_all_repeats": int(len(ally)), "rows_y_at_or_below_0": int((ally <= 0).sum()),
           "rows_y_below_50_eV": int((ally < LOW_EV).sum()), "rows_y_below_10_eV": int((ally < 10).sum()),
           "min_y_eV": float(ally.min()),
           "share_y_below_50_eV": float((ally < LOW_EV).mean()),
           "within_20pct_in_rows_below_50_eV": {LAB[e]: float(np.mean(np.concatenate(
               [(rel[r][e] <= 0.2)[D[r]["y"] < LOW_EV] for r in REPEATS]))) for e in EST}}
    log(f"low values: {low}")
    tl = {r: thirds_lab(D, F, r) for r in REPEATS}
    cells = {"all rows": lambda r: np.ones(len(D[r]["y"]), bool)}
    for j, tn in enumerate(B44.THIRDS):
        cells[f"{tn} third by median T_i"] = (lambda r, j=j: tl[r][D[r]["f"]] == j)
    cells["rows 1 to 3 after an absent frame"] = lambda r: D[r]["blip13"].astype(bool)
    cells["rows not flagged by the value rules"] = lambda r: D[r]["plain"].astype(bool)
    cells["after the fact: rows with reported value >= 50 eV"] = lambda r: D[r]["y"] >= LOW_EV
    res = {}
    for name, fn in cells.items():
        res[name] = cell(D, rel, fn, B54.rng_for(f"b60|{name}"))
        v = res[name]["value"]
        log(f"{name}: within 20 % est {v['c|0.20'][0]:.3f} gp {v['gp|0.20'][0]:.3f} hold {v['hold|0.20'][0]:.3f}; "
            f"within 10 % est {v['c|0.10'][0]:.3f} gp {v['gp|0.10'][0]:.3f}")
    A = res["all rows"]["diff"]
    st = {"T20": {"value": A["c-gp|0.20"], "reference": 0.0, "holds": bool(A["c-gp|0.20"][1] > 0),
                  "text": "share within 20 %, estimator minus causal GP; interval above 0"},
          "T20h": {"value": A["c-hold|0.20"], "reference": 0.0, "holds": bool(A["c-hold|0.20"][1] > 0),
                   "text": "share within 20 %, estimator minus hold; interval above 0"},
          "T10": {"value": A["c-gp|0.10"], "reference": -0.02, "holds": bool(A["c-gp|0.10"][1] > -0.02),
                  "text": "share within 10 %, estimator minus causal GP; interval above -0.02"}}
    V = res["all rows"]["value"]
    E = []

    def ex(text, val, lo, hi, verdict=None, key=None):
        E.append({"expectation": text, "observed": val if isinstance(val, str) else f"{val:+.4f}",
                  "range_met": None if lo is None else bool(lo <= (val if not isinstance(val, str) else 0) <= hi),
                  "verdict": None if key is None else ("holds" if st[key]["holds"] else "does not hold"),
                  "verdict_met": None if verdict is None else bool(st[key]["holds"] == verdict)})

    ex("within 20 %: estimator 0.74 to 0.80", V["c|0.20"][0], 0.74, 0.80)
    ex("within 20 %: causal GP lower by 0.003 to 0.02 (T20 holds or fails by a small margin)", A["c-gp|0.20"][0], 0.003, 0.02, None, "T20")
    ex("within 20 %: hold lower by 0.015 to 0.04; T20h holds", A["c-hold|0.20"][0], 0.015, 0.04, True, "T20h")
    w10 = [V[f"{e}|0.10"][0] for e in EST]
    E.append({"expectation": "within 10 %: all five between 0.48 and 0.58",
              "observed": ", ".join(f"{LAB[e]} {V[f'{e}|0.10'][0]:.4f}" for e in EST),
              "range_met": bool(all(0.48 <= x <= 0.58 for x in w10)), "verdict": None, "verdict_met": None})
    ex("within 10 %: estimator below the causal GP by 0.00 to 0.02 (difference -0.02 to 0); T10 holds",
       A["c-gp|0.10"][0], -0.02, 0.0, True, "T10")
    ex("median relative error (estimator) 0.085 to 0.10", V["c|median"][0], 0.085, 0.10)
    E.append({"expectation": "median relative error 0.085 to 0.10 (all five, read alongside)",
              "observed": ", ".join(f"{LAB[e]} {V[f'{e}|median'][0]:.4f}" for e in EST),
              "range_met": bool(all(0.085 <= V[f"{e}|median"][0] <= 0.10 for e in EST)), "verdict": None, "verdict_met": None})
    out = {"registration": REG, "n_boot": N_BOOT, "seed_convention": "run_b54.rng_for('b60|<cell>')",
           "check": chk, "low_reported_values": low, "statements": st, "cells": res, "expectations": E,
           "runtime_s": round(_time.time() - T0, 1)}
    OUT_JSON.write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    OUT_MD.write_text(render(out), encoding="utf-8", newline="\n")
    log("written " + str(OUT_MD))


def f3(x):
    return f"{x[0]:.3f} [{x[1]:.3f}, {x[2]:.3f}]"


def s3(x):
    return f"{x[0]:+.3f} [{x[1]:+.3f}, {x[2]:+.3f}]"


def render(o):
    st, C, low = o["statements"], o["cells"], o["low_reported_values"]
    L = [f"# B.60 -- the error as a share of the reported value (registered: {o['registration']})", "",
         "CPU, no network run. T_i, cut population, the rows of the pooled cross-validation of B.39 (row tables of B.54: "
         "80 runs, repeats r2 to r9, five folds each, folds of a repeat concatenated). Relative error of a row = absolute "
         "error / reported value of the row. Estimators: the estimator (`.b39p_v2_cut`), the history-only twin, the causal "
         "GP, hold and the offline GP (stored errors of eval_seq). Intervals: 95 %, 2,000 draws of the two-level bootstrap "
         "of B.54 (a repeat drawn, then its discharges resampled); differences between estimators within a draw. Point = "
         "mean over the eight repeats. Medians are weighted medians over the resampled rows (`run_b41.wquant`).", "",
         "## Check before the readout", "",
         "Percentiles of the absolute error (eV, mean over repeats) against section 2 of the registration, to one decimal:", "",
         "| estimator | p50 / p75 / p90 here | registered | equal |", "|---|---|---|---|"]
    for k, v in o["check"].items():
        L.append(f"| {k} | {' / '.join(f'{x:.2f}' for x in v['mine'])} | {' / '.join(f'{x:.1f}' for x in v['registered'])} | "
                 f"{'yes' if v['equal_1dp'] else 'NO'} |")
    L += ["", "The check passes.", "",
          "## Rows with a small reported value", "",
          f"Of {low['rows_all_repeats']:,} scored rows (all repeats), {low['rows_y_at_or_below_0']} have a reported value at "
          f"or below zero, {low['rows_y_below_50_eV']:,} ({low['share_y_below_50_eV']:.4f}) are below 50 eV and "
          f"{low['rows_y_below_10_eV']:,} below 10 eV; the smallest is {low['min_y_eV']:.4f} eV. The definition is applied "
          "unchanged: these rows have large relative errors and count as outside 10 % and 20 % unless the estimate is as "
          "close in relative terms. Share of them within 20 %: "
          + ", ".join(f"{k} {v:.3f}" for k, v in low["within_20pct_in_rows_below_50_eV"].items())
          + ". The cell 'rows with reported value >= 50 eV' below is an after-the-fact sensitivity readout, not registered.", "",
          "## Statements", "", "| statement | point [95 % interval] | reference | verdict |", "|---|---|---|---|"]
    for k, s in st.items():
        L.append(f"| **{k}**: {s['text']} | {s3(s['value'])} | {s['reference']:+.2f} | **{'holds' if s['holds'] else 'does not hold'}** |")
    L += ["", "## Shares and medians, all rows and strata", ""]
    for name, c in C.items():
        v, d = c["value"], c["diff"]
        L += [f"### {name}", "",
              f"Rows per repeat (mean) {np.mean(list(c['rows_per_repeat'].values())):,.0f} (share of the scored rows "
              f"{c['row_share']:.4f}); discharges with rows per repeat {min(c['discharges_per_repeat'].values())} to "
              f"{max(c['discharges_per_repeat'].values())}.", "",
              "| estimator | within 10 % | within 20 % | median relative error |", "|---|---|---|---|"]
        for e in EST:
            L.append(f"| {LAB[e]} | {f3(v[f'{e}|0.10'])} | {f3(v[f'{e}|0.20'])} | {f3(v[f'{e}|median'])} |")
        L += ["", "| estimator minus | within 10 % | within 20 % | median relative error |", "|---|---|---|---|"]
        for ref in ("gp", "hold", "t", "off"):
            L.append(f"| {LAB[ref]} | {s3(d[f'c-{ref}|0.10'])} | {s3(d[f'c-{ref}|0.20'])} | {s3(d[f'c-{ref}|median'])} |")
        L.append("")
    L += ["## Expectations of the registration (section 6) beside the findings", "",
          "| expectation | observed | range met | verdict | verdict met |", "|---|---|---|---|---|"]
    for e in o["expectations"]:
        yn = lambda x: "" if x is None else ("yes" if x else "no")          # noqa: E731
        L.append(f"| {e['expectation']} | {e['observed']} | {yn(e['range_met'])} | {e['verdict'] or ''} | {yn(e['verdict_met'])} |")
    L += ["", "## Choices made where the registration left room", "",
          "- Rows, thirds (run_b44 code within a repeat), rows 1 to 3 after an absent frame (run_b41.part_b row walk) and "
          "rows not flagged (run_b41.plain_mask) are those of B.54 (data/.b54 caches).",
          "- Shares are ratios of per-discharge sums (rows within the threshold over rows) under discharge-multiplicity "
          "weights; in a stratum, every discharge of the repeat is resampled and contributes only its rows in the stratum "
          "(as B.54's cells).",
          "- 'Median relative error 0.085 to 0.10' in section 6 is read for the estimator; the five medians are listed beside it.",
          "- 'Causal GP lower by 0.003 to 0.02' is read as estimator minus causal GP in [+0.003, +0.02]; 'the estimator below "
          "the causal GP by 0.00 to 0.02' as estimator minus causal GP in [-0.02, 0].",
          "- Generators: numpy default_rng([20261154, crc32('b60|' + cell name)]) (run_b54.rng_for).", "",
          "## Limits", "", "The reported value carries its own scatter, so a share within 10 % of it is bounded by that scatter "
          "and is not a share within 10 % of the ion temperature. Low values of T_i make the relative error large for a "
          "fixed error in eV. The comparison with published real-time analyses is not like for like: their reference is an "
          "offline analysis of the same spectra, ours is the reported value of the frame itself.", "",
          f"Run time: {o['runtime_s']:.0f} s."]
    return "\n".join(L) + "\n"


# ================================================================================ Addendum 1: the filters of B.56
B56 = B54.DATA / ".b56"
A1_EST = ("kfx", "kf0", "c", "t", "gp", "hold", "off")
A1_LAB = {**LAB, "kfx": "KFX", "kf0": "KF0"}
A1_PAIRS = [("kfx", "gp"), ("kfx", "kf0"), ("kfx", "hold"), ("kfx", "t"), ("kfx", "c"), ("kfx", "off"),
            ("kf0", "gp"), ("c", "gp")]
CONFIRM, DEV = tuple(range(3, 10)), (2,)
A1_MED = {"kfx": 41.8, "kf0": 43.3}
A1_MARK = "## Addendum 1: the filters of B.56"


def cell_x(D, rel, mask_fn, rng, reps):
    """B.60's cell (same resampling, same quantities) over the repeats `reps`, the estimators A1_EST and the
    differences A1_PAIRS, each difference within a draw."""
    draws = rng.choice(np.array(reps), size=N_BOOT)
    bs, pts = {}, {}
    for r in reps:
        d = D[r]
        m = mask_fn(r)
        f, n = d["f"], d["n"]
        sel = np.flatnonzero(draws == r)
        idx = rng.integers(0, n, size=(len(sel), n))
        W = np.zeros((len(sel), n))
        for j in range(len(sel)):
            W[j] = np.bincount(idx[j], minlength=n)
        cnt = np.bincount(f[m], None, n).astype(float)
        wc = W @ cnt
        p_r = {}
        for e in A1_EST:
            x = rel[r][e]
            for th in THRESH:
                key = f"{e}|{th:.2f}"
                num = np.bincount(f[m], (x <= th)[m].astype(float), n)
                bs.setdefault(key, np.empty(N_BOOT))[sel] = (W @ num) / wc
                p_r[key] = float(num.sum() / cnt.sum())
            key = f"{e}|median"
            bs.setdefault(key, np.empty(N_BOOT))[sel] = B41.wquant(x[m], f[m], W, [0.5])[:, 0]
            p_r[key] = float(np.median(x[m]))
        p_r["rows"], p_r["discharges"] = int(m.sum()), int(len(np.unique(f[m])))
        pts[r] = p_r
    out = {"rows_per_repeat": {str(r): pts[r]["rows"] for r in reps},
           "discharges_per_repeat": {str(r): pts[r]["discharges"] for r in reps},
           "row_share": float(np.mean([mask_fn(r).mean() for r in reps])), "value": {}, "diff": {}}
    for e in A1_EST:
        for q in ("0.10", "0.20", "median"):
            key = f"{e}|{q}"
            out["value"][key] = [float(np.mean([pts[r][key] for r in reps])), *B54.ci(bs[key])]
            out["value"][key + "|per_repeat"] = {str(r): pts[r][key] for r in reps}
    for a, b in A1_PAIRS:
        for q in ("0.10", "0.20", "median"):
            ka, kb = f"{a}|{q}", f"{b}|{q}"
            out["diff"][f"{a}-{b}|{q}"] = [float(np.mean([pts[r][ka] - pts[r][kb] for r in reps])), *B54.ci(bs[ka] - bs[kb])]
            out["diff"][f"{a}-{b}|{q}|per_repeat"] = {str(r): pts[r][ka] - pts[r][kb] for r in reps}
    return out


def addendum1():
    t0 = _time.time()
    F = B54.load_files()
    D = {r: B54.repeat_data(r) for r in (*DEV, *CONFIRM)}
    log("repeats loaded")
    # ---------------- check: rows identical to the B.54 tables; medians of KFX and KF0 (r3 to r9) to one decimal
    rows_chk = {}
    for r in D:
        est = {"kfx": [], "kf0": []}
        for k in range(B54.K):
            z = np.load(B56 / f"r{r}_f{k}.npz")
            q = B54.load_rows(r, k)
            same = (len(z["shot"]) == len(q["shot"]) and np.array_equal(z["shot"], q["shot"])
                    and np.array_equal(z["row"], q["row"]))
            dy = float(np.max(np.abs(z["y"] - q["y"]))) if same else np.inf
            rows_chk[f"r{r}_f{k}"] = {"rows_b56": int(len(z["shot"])), "rows_b54": int(len(q["shot"])),
                                      "same_rows": bool(same), "max_abs_value_diff_eV": dy,
                                      "filter_outputs_finite": bool(np.isfinite(z["est_kfx"]).all() and np.isfinite(z["est_kf0"]).all()),
                                      "passes": bool(same and dy <= 1e-3)}
            for e in est:
                est[e].append(z[f"est_{e}"].astype(np.float64))
        for e in est:
            D[r]["se"][e] = (np.concatenate(est[e]) - D[r]["y"]) ** 2
    med = {e: {"mine": float(np.mean([np.median(np.sqrt(D[r]["se"][e])) for r in CONFIRM])), "recorded": v}
           for e, v in A1_MED.items()}
    for v in med.values():
        v["equal_1dp"] = bool(round(v["mine"], 1) == v["recorded"])
    chk = {"rows": rows_chk, "rows_pass": bool(all(v["passes"] for v in rows_chk.values())), "medians_eV_r3_r9": med}
    chk["passes"] = bool(chk["rows_pass"] and all(v["equal_1dp"] for v in med.values()))
    log(f"addendum check: rows {chk['rows_pass']}; medians " + "; ".join(f"{k} {v['mine']:.3f} vs {v['recorded']}" for k, v in med.items()))
    o = json.loads(OUT_JSON.read_text(encoding="utf-8"))
    if not chk["passes"]:
        o["addendum1"] = {"check": chk, "stopped": True}
        OUT_JSON.write_text(json.dumps(o, indent=1, default=float), encoding="utf-8")
        raise SystemExit("FATAL: the check of Addendum 1 fails; no statement is read")
    rel = {r: {e: np.sqrt(D[r]["se"][e]) / D[r]["y"] for e in A1_EST} for r in D}
    ally = np.concatenate([D[r]["y"] for r in CONFIRM])
    lowm = {r: D[r]["y"] < LOW_EV for r in CONFIRM}
    low = {"rows_r3_r9": int(len(ally)), "rows_y_at_or_below_0": int((ally <= 0).sum()),
           "rows_y_below_50_eV": int((ally < LOW_EV).sum()), "share_y_below_50_eV": float((ally < LOW_EV).mean()),
           "within_20pct_in_rows_below_50_eV": {A1_LAB[e]: float(np.mean(np.concatenate([(rel[r][e] <= 0.2)[lowm[r]] for r in CONFIRM])))
                                                for e in A1_EST},
           "within_10pct_in_rows_below_50_eV": {A1_LAB[e]: float(np.mean(np.concatenate([(rel[r][e] <= 0.1)[lowm[r]] for r in CONFIRM])))
                                                for e in A1_EST},
           "median_relative_error_in_rows_below_50_eV": {A1_LAB[e]: float(np.median(np.concatenate([rel[r][e][lowm[r]] for r in CONFIRM])))
                                                         for e in A1_EST}}
    tl = {r: thirds_lab(D, F, r) for r in D}
    cells = {"all rows": lambda r: np.ones(len(D[r]["y"]), bool)}
    for j, tn in enumerate(B44.THIRDS):
        cells[f"{tn} third by median T_i"] = (lambda r, j=j: tl[r][D[r]["f"]] == j)
    cells["rows 1 to 3 after an absent frame"] = lambda r: D[r]["blip13"].astype(bool)
    cells["rows not flagged by the value rules"] = lambda r: D[r]["plain"].astype(bool)
    cells["rows with reported value below 50 eV"] = lambda r: D[r]["y"] < LOW_EV
    cells["rows with reported value >= 50 eV"] = lambda r: D[r]["y"] >= LOW_EV
    res = {}
    for name, fn in cells.items():
        res[name] = cell_x(D, rel, fn, B54.rng_for(f"b60a1|{name}"), CONFIRM)
        v = res[name]["value"]
        log(f"{name}: within 20 % KFX {v['kfx|0.20'][0]:.3f} KF0 {v['kf0|0.20'][0]:.3f} GP {v['gp|0.20'][0]:.3f}; "
            f"within 10 % KFX {v['kfx|0.10'][0]:.3f} GP {v['gp|0.10'][0]:.3f}")
    dev = {"all rows": cell_x(D, rel, cells["all rows"], B54.rng_for("b60a1|r2|all rows"), DEV),
           "low third by median T_i": cell_x(D, rel, cells["low third by median T_i"], B54.rng_for("b60a1|r2|low third"), DEV)}
    A, Cd = res["all rows"]["diff"], res["low third by median T_i"]["diff"]
    st = {"F20": {"value": A["kfx-gp|0.20"], "reference": 0.0, "holds": bool(A["kfx-gp|0.20"][1] > 0),
                  "text": "share within 20 %, KFX minus causal GP; interval above 0"},
          "F10": {"value": A["kfx-gp|0.10"], "reference": 0.0, "holds": bool(A["kfx-gp|0.10"][1] > 0),
                  "text": "share within 10 %, KFX minus causal GP; interval above 0"},
          "F20x": {"value": A["kfx-kf0|0.20"], "reference": 0.0, "holds": bool(A["kfx-kf0|0.20"][1] > 0),
                   "text": "share within 20 %, KFX minus KF0; interval above 0"},
          "F20c": {"value": Cd["kfx-gp|0.20"], "reference": -0.02, "holds": bool(Cd["kfx-gp|0.20"][1] > -0.02),
                   "text": "coldest third, share within 20 %, KFX minus causal GP; interval above -0.02"}}
    V = res["all rows"]["value"]
    E = []
    for text, val, lo, hi, key, want in (
            ("KFX within 20 % 0.775 to 0.80", V["kfx|0.20"][0], 0.775, 0.80, None, None),
            ("KFX above the causal GP within 20 % by +0.003 to +0.02 (F20 holds or fails by a small margin)", A["kfx-gp|0.20"][0], 0.003, 0.02, "F20", None),
            ("KFX within 10 % 0.575 to 0.61", V["kfx|0.10"][0], 0.575, 0.61, None, None),
            ("KFX above the causal GP within 10 % by +0.005 to +0.03; F10 holds", A["kfx-gp|0.10"][0], 0.005, 0.03, "F10", True),
            ("F20x +0.003 to +0.015, holds", A["kfx-kf0|0.20"][0], 0.003, 0.015, "F20x", True),
            ("F20c -0.01 to +0.01, holds", Cd["kfx-gp|0.20"][0], -0.01, 0.01, "F20c", True)):
        E.append({"expectation": text, "observed": f"{val:+.4f}", "range_met": bool(lo <= val <= hi),
                  "verdict": None if key is None else ("holds" if st[key]["holds"] else "does not hold"),
                  "verdict_met": None if want is None else bool(st[key]["holds"] == want)})
    a1 = {"note": "Addendum 1 of PREREGISTRATION_B60.md, registered after the statements of B.60 and the skills of B.56 "
                  "were read, before any relative error of a filter was computed. Confirmation set r3 to r9; r2 (the "
                  "development set of the filters) reported apart.",
          "check": chk, "low_reported_values_r3_r9": low, "statements": st, "cells_r3_r9": res, "r2_development": dev,
          "expectations": E, "runtime_s": round(_time.time() - t0, 1)}
    o["addendum1"] = a1
    OUT_JSON.write_text(json.dumps(o, indent=1, default=float), encoding="utf-8")
    md = OUT_MD.read_text(encoding="utf-8")
    if A1_MARK in md:
        md = md[:md.index(A1_MARK)].rstrip("\n") + "\n"
    OUT_MD.write_text(md + "\n" + render_a1(a1), encoding="utf-8", newline="\n")
    log("addendum 1 written")


def render_a1(a):
    st, C, low, ck = a["statements"], a["cells_r3_r9"], a["low_reported_values_r3_r9"], a["check"]
    lab = A1_LAB
    L = [A1_MARK, "",
         "Registered after the statements of B.60 and the skills of B.56 were read (Addendum 1 of the registration), "
         "before any relative error of a filter was computed. KF0 and KFX: the five-parameter robust filters of B.56 on "
         "log T_i (KF0 without fast channels, KFX with them; signed one-step outputs from `data/.b56/r{r}_f{k}.npz`). "
         "Rows: the scored rows of re-partitions r3 to r9 (the confirmation set); r2, the development set of the "
         "filters, is reported apart and not pooled. Every estimator, the recurrent estimator, the twin, the causal "
         "GP, hold and the offline GP included, is read on these same rows. Definitions, interval (2,000 draws, a "
         "re-partition then its discharges, differences within a draw) and strata are those of B.60 above.", "",
         "### Check before the readout", "",
         f"- Rows: the B.56 cache and the B.54 row tables hold the same stored rows (discharge and row index) with the "
         f"same reported values in {sum(v['passes'] for v in ck['rows'].values())} of {len(ck['rows'])} folds of r2 to r9 "
         f"(largest value difference {max(v['max_abs_value_diff_eV'] for v in ck['rows'].values()):.1e} eV; rows per fold "
         f"identical: {'yes' if all(v['rows_b56'] == v['rows_b54'] for v in ck['rows'].values()) else 'NO'}).",
         "- Median absolute error, mean over r3 to r9 (eV): "
         + "; ".join(f"{lab[k]} {v['mine']:.2f} against recorded {v['recorded']:.1f} ({'equal' if v['equal_1dp'] else 'NOT equal'})"
                     for k, v in ck["medians_eV_r3_r9"].items()) + ".",
         f"- The check {'passes' if ck['passes'] else 'FAILS'}.", "",
         "### Statements (r3 to r9)", "", "| statement | point [95 % interval] | reference | verdict |", "|---|---|---|---|"]
    for k, s in st.items():
        L.append(f"| **{k}**: {s['text']} | {s3(s['value'])} | {s['reference']:+.2f} | **{'holds' if s['holds'] else 'does not hold'}** |")
    L += ["", "### Shares and medians by stratum (r3 to r9)", ""]
    for name, c in C.items():
        v, d = c["value"], c["diff"]
        L += [f"#### {name}" + (" (after the fact)" if "50 eV" in name else ""), "",
              f"Rows per repeat (mean) {np.mean(list(c['rows_per_repeat'].values())):,.0f} (share {c['row_share']:.4f}); "
              f"discharges with rows per repeat {min(c['discharges_per_repeat'].values())} to {max(c['discharges_per_repeat'].values())}.", "",
              "| estimator | within 10 % | within 20 % | median relative error |", "|---|---|---|---|"]
        for e in A1_EST:
            L.append(f"| {lab[e]} | {f3(v[f'{e}|0.10'])} | {f3(v[f'{e}|0.20'])} | {f3(v[f'{e}|median'])} |")
        L += ["", "| difference | within 10 % | within 20 % | median relative error |", "|---|---|---|---|"]
        for x, y in A1_PAIRS:
            L.append(f"| {lab[x]} minus {lab[y]} | {s3(d[f'{x}-{y}|0.10'])} | {s3(d[f'{x}-{y}|0.20'])} | {s3(d[f'{x}-{y}|median'])} |")
        L.append("")
    L += ["### Rows with a reported value below 50 eV (r3 to r9)", "",
          f"{low['rows_y_below_50_eV']:,} of {low['rows_r3_r9']:,} rows ({low['share_y_below_50_eV']:.4f}); "
          f"{low['rows_y_at_or_below_0']} at or below zero. Definition applied unchanged.", "",
          "| estimator | within 10 % | within 20 % | median relative error |", "|---|---|---|---|"]
    for e in A1_EST:
        L.append(f"| {lab[e]} | {low['within_10pct_in_rows_below_50_eV'][lab[e]]:.4f} | "
                 f"{low['within_20pct_in_rows_below_50_eV'][lab[e]]:.4f} | {low['median_relative_error_in_rows_below_50_eV'][lab[e]]:.3f} |")
    L += ["", "### r2, the development set (reported apart)", "",
          "| cell | KFX within 20 % | causal GP within 20 % | KFX minus causal GP, 20 % | KFX minus causal GP, 10 % | KFX minus KF0, 20 % |",
          "|---|---|---|---|---|---|"]
    for name, c in a["r2_development"].items():
        v, d = c["value"], c["diff"]
        L.append(f"| {name} | {f3(v['kfx|0.20'])} | {f3(v['gp|0.20'])} | {s3(d['kfx-gp|0.20'])} | {s3(d['kfx-gp|0.10'])} | "
                 f"{s3(d['kfx-kf0|0.20'])} |")
    L += ["", "(r2 alone: the interval resamples its discharges only.)", "",
          "### Expectations of Addendum 1 beside the findings", "",
          "| expectation | observed | range met | verdict | verdict met |", "|---|---|---|---|---|"]
    yn = lambda x: "" if x is None else ("yes" if x else "no")                  # noqa: E731
    for e in a["expectations"]:
        L.append(f"| {e['expectation']} | {e['observed']} | {yn(e['range_met'])} | {e['verdict'] or ''} | {yn(e['verdict_met'])} |")
    L += ["", "Choices: the strata are those of B.60 (B.54 definitions: thirds within a re-partition by median T_i, rows "
          "1 to 3 after an absent frame by the row walk of run_b41.part_b, rows not flagged by run_b41.plain_mask), not "
          "the ten-row after-blip class of B.56. The cell of rows below 50 eV and the cell at or above 50 eV are "
          "reported (the first is listed in the addendum; the second is after the fact). Generators: "
          "run_b54.rng_for('b60a1|' + cell name).", "",
          f"Run time of the addendum stage: {a['runtime_s']:.0f} s."]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    if sys.argv[1:] == ["addendum1"]:
        addendum1()
    else:
        main()
