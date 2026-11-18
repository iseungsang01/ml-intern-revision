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


if __name__ == "__main__":
    main()
