"""B.83: docs/paper/b83_interp.md from the readout JSON (called by readout83.py; also runnable alone).

    py ces_prediction/experiments/b83_interp/render83.py
"""
from __future__ import annotations

import json

import common83 as M

EV = {"RMSE", "MAE", "median absolute error", "p75 absolute error", "p90 absolute error"}


def f(v, ms):
    if ms in EV:
        return f"{v:.1f}"
    return f"{v:.3f}"


def cell(o, ms, sign=False):
    fm = (lambda x: f"{x:+.3f}") if sign else (lambda x: f(x, ms))
    return (f"{fm(o['point'])}<br>c [{fm(o['cluster'][0])}, {fm(o['cluster'][1])}]"
            f"<br>d [{fm(o['discharge'][0])}, {fm(o['discharge'][1])}]")


def render(res):
    L = ["# B.83 — the nowcast against interpolation that waits for the next value (target F)", "",
         f"Registered: `{res['registration']}`; implementation choices `{res['addendum']}`. CPU only; no estimator refitted. "
         "Target: the reported value of row t (F) on the D1 rows. PCHIP and the offline GP read measured values after row t "
         "(row t itself left out) and are not available in real time; hold, the causal GP, T0 and TX are causal. "
         "Rows: per design, the scored rows of B.73 on which both references can be formed (a measured value before and "
         "after row t in B.70's block); every estimator of a table is scored on the same rows. Skill of A over R = "
         "1 - SSE(A)/SSE(R); a ratio is A over R. 95 % intervals, 20,000 draws: c = shot clusters resampled (verdicts), "
         "d = discharges resampled; under V1 (V2) a re-partition (assignment) is drawn first. Points: mean over "
         "re-partitions (V1) or assignments (V2). Checks: `docs/paper/b83_checks.md` (all four passed: "
         f"{all(res['checks'].values())}).", "",
         "## Statements (verdict on the cluster interval)", "",
         "| id | statement | value | cluster interval | discharge interval | verdict | registered expectation |",
         "|---|---|---|---|---|---|---|"]
    for sid, s in res["statements"].items():
        v = s["value"]
        sk = s["measure"] == "MSE skill"
        fm = (lambda x: f"{x:+.3f}") if sk else (lambda x: f"{x:.3f}")
        L.append(f"| {sid} | {s['text']} ({s['measure']}) | {fm(v['point'])} | [{fm(v['cluster'][0])}, {fm(v['cluster'][1])}] | "
                 f"[{fm(v['discharge'][0])}, {fm(v['discharge'][1])}] | {'holds' if s['holds'] else 'does not hold'} | "
                 f"{s['expected']} |")
    L += ["", "Rules: " + "; ".join(sorted({f"{s['judged']}" for s in res["statements"].values()})) + ".", "",
          "## Counts", "",
          "| design | scored rows | reference rows | rows left out (no reference) | discharges | clusters | rows with the value above 100 eV |",
          "|---|---|---|---|---|---|---|"]
    for d, c in res["counts"].items():
        L.append(f"| {M.DESIGN_LABEL[d]} | {c['scored_rows']:,} | {c['reference_rows']:,} | "
                 f"{c['rows_left_out_missing_reference']:,} | {c['discharges']} | {c['clusters']} | {c['rows_value_above_100_eV']:,} |")
    ri = res["refs_info"]
    L += ["", f"Of the {ri['rows']:,} D1 rows, {ri['not_formed']:,} have no measured value after row t in B.70's block "
              f"(no value before: {ri['no_past']}). Offline GP fits that fell back to hold: {ri['gp_fit_failed_fell_back_to_hold']}; "
              f"PCHIP rows interpolated linearly (two neighbour times): {ri['pchip_two_points_linear']:,}.", ""]
    names = res["estimator_names"]
    for d in res["estimators"]:
        E = res["estimators"][d]
        L += [f"## {M.DESIGN_LABEL[d]}", "", f"### Estimators ({res['counts'][d]['reference_rows']:,} rows)", "",
              "| estimator | " + " | ".join(res["measures"]) + " |", "|---|" + "---|" * len(res["measures"])]
        for k, v in E.items():
            L.append(f"| {names[k]} | " + " | ".join(cell(v[ms], ms) for ms in res["measures"]) + " |")
        L += ["", "Absolute errors in eV; relative error and shares as fractions.", "", "### Pairs", "",
              "| A over R | " + " | ".join(res["pair_measures"]) + " |", "|---|" + "---|" * len(res["pair_measures"])]
        for k, v in res["pairs"][d].items():
            L.append(f"| {k} | " + " | ".join(cell(v[pm], pm, sign=pm == "MSE skill" or pm.endswith("difference"))
                                             for pm in res["pair_measures"]) + " |")
        A = res["all_rows"][d]
        L += ["", f"On all scored rows of the design ({A['counts']['rows']:,} rows, {A['counts']['discharges']} discharges, "
                  f"{A['counts']['clusters']} clusters; no reference involved): RMSE (eV) "
                  + ", ".join(f"{names[k]} {v['point']:.1f}" for k, v in A["RMSE"].items()) + "; MSE skill "
                  + "; ".join(f"{k} {v['MSE skill']['point']:+.3f} c [{v['MSE skill']['cluster'][0]:+.3f}, "
                              f"{v['MSE skill']['cluster'][1]:+.3f}] d [{v['MSE skill']['discharge'][0]:+.3f}, "
                              f"{v['MSE skill']['discharge'][1]:+.3f}]" for k, v in A["pairs"].items()) + ".", ""]
    L += [f"Readout time {res['runtime_readout_s']:.0f} s.", ""]
    return "\n".join(L)


if __name__ == "__main__":
    r = json.loads(M.OUT_JSON.read_text(encoding="utf-8"))
    M.save_text(M.OUT_MD, render(r))
