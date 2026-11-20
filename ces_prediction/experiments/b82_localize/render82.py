"""B.82: docs/paper/b82_localize.md from the readout dictionary (readout82.py).  No number is computed here."""
from __future__ import annotations

DES = ("V1", "V2", "V3", "V4a", "V4b")
DLAB = {"V1": "V1 random (r2-r4)", "V2": "V2 cluster-blocked (3 assignments)", "V3": "V3 rolling origin (fifths 2-5 pooled)",
        "V4a": "V4a chronological: fit to 31991, score from 32312", "V4b": "V4b chronological: fit to 32310, score from 32312"}
RLAB = {"D10": "D10 (previous frame visible)", "D650": "D650 (history 0.65 s old)", "S": "S (beam-modulation schedule)"}
TAU = ("up to 100 ms", "above 100 and up to 300 ms", "above 300 ms", "no notch before")
CC = ("lowest third", "middle third", "highest third", "c not finite")


def s3(x):
    return f"{x:+.3f}"


def iv(v):
    return f"[{v[0]:+.3f}, {v[1]:+.3f}]"


def cellf(v):
    if not v:
        return "-"
    return f"{s3(v['point'])}<br>c {iv(v['cluster'])}<br>d {iv(v['discharge'])}"


def f1(v):
    return f"{v['point']:.1f}<br>c [{v['cluster'][0]:.1f}, {v['cluster'][1]:.1f}]"


def f3(v):
    return f"{v['point']:.3f}<br>c [{v['cluster'][0]:.3f}, {v['cluster'][1]:.3f}]"


def rows_of(c):
    r = c["rows"]
    return f"{r:,}" if isinstance(r, int) else f"{r:,.0f} (mean of splits)"


def disc_of(c):
    d = c["discharges"]
    return str(d) if isinstance(d, int) else "-".join(str(x) for x in (min(d), max(d)))


def render(o):
    L = ["# B.82 — where the fast-channel increment arises, robust error measures, specificity, and the seed of the fit", "",
         f"Registered: `{o['registration']}`; implementation choices in `{o['addendum']}`, written before any fit, check or "
         "readout. CPU only; boosted regressions of the kind of B.75, no network. Target L5 (the median of the measured values "
         "among the next five rows). Rows, inputs, units, regression and target transform of B.75; rules D10 and D650 on "
         "B.75's common set, rule S on B.75's S set. Exploratory on used data (registration section 1).", "",
         "Estimators. H: the newest visible value. G: the causal GP. T0: the ten history scalars. TX: T0's inputs, the eight "
         "fast scalars of row t and the six anchor scalars. TE (TB): T0's inputs and the ECEI (BES) half of TX's fast inputs. "
         "TN, TXN: T0 and TX with the nineteen beam scalars (B.79, columns as delivered). O: the frame's own reading. New here: "
         "TM (T0's inputs and seven Mirnov scalars), TXM (TX's inputs and the seven Mirnov scalars), TXs (TX with the fourteen "
         "fast and anchor columns taken from the row with the same frame index in a donor discharge of the same side), TBi, TBm, "
         "TBo (T0's inputs and the seven BES scalars of one radial column: 05, 06, 07).", "",
         "Conventions. Skill of A over R = 1 - MSE(A) / MSE(R) on identical rows. Each value: point; 95 % interval with shot "
         "clusters resampled (c); with discharges resampled (d); 20,000 draws (V1: a re-partition drawn first; V2: an "
         "assignment drawn first; V3, V4: the scored units). Verdicts on the cluster interval. Points under V1 and V2: mean "
         "over the three re-partitions or assignments. RMSE and absolute errors in eV. tau: time since the first row after the "
         "most recent absent step; c = |a2(BES)| + |a2(ECEI)|, thirds cut at the unit's fit rows. Within a stratified cell "
         "the clusters (discharges) with at least one row of the cell are resampled, and every class and every difference of "
         "classes is formed on the same draws. Checks: `docs/paper/b82_checks.md` (all passed before this readout: "
         + ", ".join(f"{k} {v}" for k, v in o["checks"].items()) + ").", ""]
    # ---- statements
    L += ["## Statements (target L5, cluster interval)", "",
          "| Id | statement | point | cluster interval | discharge interval | verdict | registered expectation | as expected |",
          "|---|---|---|---|---|---|---|---|"]
    for k, s in o["statements"].items():
        v = s["value"]
        verdict = "holds" if s["holds"] else "does not hold"
        met = o["expectations"][k]["verdict_as_expected"]
        met = "n/a (uncertain)" if met is None else ("yes" if met else "no")
        L.append(f"| {k} | {s['text']} | {s3(v['point'])} | {iv(v['cluster'])} | {iv(v['discharge'])} | {verdict} | "
                 f"{s['expected']} | {met} |")
    L += ["", "Judgement: A1-A8, B5, C2, C3, C7: lower end above zero; B6: lower end above 0.5; B1-B4, B7, B8: upper end "
          "below one; C1, C4, C5, C6: lower end at most zero.", ""]
    # ---- part A
    L += ["## Part A. The increment by time since the last notch (tau) and by the change of the fast level since the anchor row (c)", ""]
    A = o["part_A"]
    for d in DES:
        for r in ("D650", "S"):
            for block, cell in A[d][r].items():
                head = f"### {DLAB[d]}, {RLAB[r]}" + ("" if block == "all" else f", {block} only")
                L += [head, "", f"Rows {cell['counts']['rows']:,}, discharges {cell['counts']['discharges']}, clusters "
                      f"{cell['counts']['clusters']}. Pooled TX/T0 {s3(cell['pooled_points']['TX/T0'])}.", ""]
                for sname, classes, dname in (("tau", TAU, "up to 100 ms minus above 300 ms"),
                                              ("c", CC, "highest third minus lowest third")):
                    L += [f"| {'tau' if sname == 'tau' else 'c'} class | rows | discharges | TX/T0 | TE/T0 | TB/T0 | TXN/TN | RMSE T0 / TX / TE / TB / TN / TXN |",
                          "|---|---|---|---|---|---|---|---|"]
                    for cn in classes:
                        c = cell[sname][cn]
                        if c.get("empty"):
                            L.append(f"| {cn} | 0 | - | - | - | - | - | - |")
                            continue
                        rm = c["rmse_eV"]
                        L.append(f"| {cn} | {rows_of(c['counts'])} | {disc_of(c['counts'])} | " +
                                 " | ".join(cellf(c.get(x)) for x in ("TX/T0", "TE/T0", "TB/T0", "TXN/TN")) + " | " +
                                 " / ".join(f"{rm[x]:.1f}" for x in ("T0", "TX", "TE", "TB", "TN", "TXN")) + " |")
                    if dname in cell[sname]:
                        dd = cell[sname][dname]
                        L.append(f"| {dname} | | | " + " | ".join(cellf(dd.get(x)) for x in ("TX/T0", "TE/T0", "TB/T0", "TXN/TN"))
                                 + " | |")
                    L.append("")
    # ---- part B
    L += ["## Part B. Robust error measures", "",
          "Per estimator: MAE, median, 75th and 90th percentile of the absolute error (eV), median relative error over rows "
          "with L5 above 100 eV, shares of rows within 10 % and 20 % of L5. Ratios TX over T0; share differences TX minus T0. "
          "Per discharge (at least 50 scored rows): skill of TX over T0.", ""]
    B = o["part_B"]
    MEAS = ("MAE", "median absolute error", "p75 absolute error", "p90 absolute error",
            "median relative error (L5 above 100 eV)", "share within 10 %", "share within 20 %")
    for d in DES:
        for r in ("D10", "D650", "S"):
            c = B[d][r]
            cn = c["counts"]
            L += [f"### {DLAB[d]}, {RLAB[r]}", "",
                  f"Rows {cn['rows']:,} (L5 above 100 eV: {cn['rows_with_L5_above_100_eV']:,}), discharges {cn['discharges']} "
                  f"({cn['discharges_with_at_least_50_rows']} with at least 50 rows), clusters {cn['clusters']}.", "",
                  "| estimator | RMSE | MAE | median AE | p75 AE | p90 AE | median rel. error | within 10 % | within 20 % |",
                  "|---|---|---|---|---|---|---|---|---|"]
            for e in ("H", "G", "T0", "TX", "O"):
                m = c["estimators"][e]
                L.append(f"| {e} | {c['rmse_eV'][e]:.1f} | " + " | ".join(f1(m[x]) for x in MEAS[:4]) + " | " +
                         " | ".join(f3(m[x]) for x in MEAS[4:]) + " |")
            t = c["TX over T0"]
            L += ["", "| TX over T0 | MAE ratio | median AE ratio | p75 AE ratio | p90 AE ratio | median rel. error ratio | "
                  "within 10 %, difference | within 20 %, difference |", "|---|---|---|---|---|---|---|---|",
                  "| | " + " | ".join(cellf(t[x]) for x in ("MAE ratio", "median absolute error ratio", "p75 absolute error ratio",
                                                            "p90 absolute error ratio", "median relative error ratio",
                                                            "share within 10 % difference", "share within 20 % difference")) + " |"]
            p = c["per discharge"]
            L += ["", "| per discharge | share with positive skill | lower quartile | median | upper quartile |",
                  "|---|---|---|---|---|",
                  "| skill of TX over T0 | " + " | ".join(cellf(p[x]) for x in ("share of discharges with positive skill",
                                                                                 "lower quartile of the skill", "median of the skill",
                                                                                 "upper quartile of the skill")) + " |", ""]
    w = o.get("wq_direct_check")
    if w:
        L += ["Weighted quantiles against the quantiles of the explicitly resampled rows (V1, D650, TX, first re-partition, 20 "
              "draws per form): " + "; ".join(f"{k}: largest difference {v['largest_abs_difference']:.3g}" for k, v in w.items())
              + ".", ""]
    # ---- part C
    L += ["## Part C. Specificity: Mirnov in place of and beside the fast diagnostics, fast inputs of another discharge, the BES radial columns", ""]
    Cc = o["part_C"]
    for r in ("D650", "S"):
        L += [f"### {RLAB[r]}", "", "| design | TM/T0 | TXM/T0 | TXs/T0 | TBi/T0 (05) | TBm/T0 (06) | TBo/T0 (07) | TX/T0 | TX/TM | TXM/TX |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for d in ("V1", "V3", "V4a"):
            m = Cc[d][r]
            L.append(f"| {DLAB[d]} | " + " | ".join(cellf(m[x]) for x in ("TM/T0", "TXM/T0", "TXs/T0", "TBi/T0", "TBm/T0",
                                                                         "TBo/T0", "TX/T0", "TX/TM", "TXM/TX")) + " |")
        L.append("")
    L += ["| design | rule | rows | discharges | clusters | RMSE T0 | TX | TM | TXM | TXs | TBi | TBm | TBo | TXs test rows without a donor row |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for d in ("V1", "V3", "V4a"):
        for r in ("D650", "S"):
            m = Cc[d][r]
            cn = m["counts"]
            L.append(f"| {d} | {r} | {cn['rows']:,} | {cn['discharges']} | {cn['clusters']} | " +
                     " | ".join(f"{m['rmse_eV'][x]:.1f}" for x in ("T0", "TX", "TM", "TXM", "TXs", "TBi", "TBm", "TBo")) +
                     f" | {m['missing_inputs']['TXs']['test_rows_without_a_donor_row']:,} |")
    L.append("")
    # ---- part D
    L += ["## Part D. The increment TX over T0 over the seed of the regression (random_state 1 to 10)", "",
          "| cell | rule | rows | registered seed | min | median | max | sd | sign changes among the ten |",
          "|---|---|---|---|---|---|---|---|---|"]
    for r in ("D650", "S"):
        for k, v in o["part_D"][r].items():
            L.append(f"| {k} | {r} | {v.get('rows', '')} | {s3(v['registered_seed_20261116'])} | {s3(v['min'])} | {s3(v['median'])} | "
                     f"{s3(v['max'])} | {v['sd_ddof1']:.3f} | {'yes' if v['sign_changes_among_the_ten_seeds'] else 'no'} |")
    L.append("")
    L += ["Per seed, V1 r2 pooled over its five units:", "", "| rule | " + " | ".join(str(s) for s in range(1, 11)) + " |",
          "|---|" + "---|" * 10]
    for r in ("D650", "S"):
        v = o["part_D"][r]["V1 r2 pooled (five units)"]["per_seed"]
        L.append(f"| {r} | " + " | ".join(s3(v[str(s)]) for s in range(1, 11)) + " |")
    L.append("")
    # ---- part E
    E = o["part_E"]
    b = E["beam"]

    def p3(x, fmt="{:.0f}"):
        return " / ".join(fmt.format(x[k]) for k in ("10", "50", "90"))
    L += ["## Part E. The data set (D1 discharges; file facts, no estimator)", "",
          "| quantity | value |", "|---|---|",
          f"| discharges | {E['d1_discharges']} |",
          f"| grid rows (first and last row of a file left out) | {E['grid_rows_of_d1_discharges']:,} |",
          f"| grid rows per discharge, 10th / 50th / 90th percentile | {p3(E['grid_rows_per_discharge'])} |",
          f"| absent steps per discharge, 10th / 50th / 90th percentile | {p3(E['absent_steps_per_discharge'])} (total {E['absent_steps_total']:,}; "
          f"discharges without one {E['discharges_without_absent_step']}) |",
          f"| D1 rows (reported T_i in the cut, held-free grid) | {E['d1_rows']:,} |",
          f"| reported T_i over D1 rows, 10th / 50th / 90th percentile (eV) | {p3(E['reported_Ti_eV_over_d1_rows'])} |",
          f"| L5 over D1 rows with L5, 10th / 50th / 90th percentile (eV) | {p3(E['L5_eV_over_d1_rows'])} ({E['d1_rows_with_L5']:,} rows) |",
          f"| share of grid rows without a usable T_i | {E['share_of_grid_rows_without_usable_Ti']:.4f} ({E['grid_rows_without_usable_Ti']:,} rows) |",
          f"| total beam power over grid rows with a source on, 10th / 50th / 90th percentile (MW; the 13 files left out) | "
          f"{p3(b['total_power_MW_over_grid_rows_with_a_source_on_outside_13_files'], '{:.2f}')} ({b['rows_of_that_line']:,} rows) |"]
    sh = b["share_of_grid_rows_by_sources_on_all_d1_discharges"]
    sh2 = b["share_of_grid_rows_by_sources_on_outside_13_files"]
    L.append("| share of grid rows with 0 / 1 / 2 / 3 / 4 / 5 sources on, all D1 discharges | " +
             " / ".join(f"{sh[str(k)]:.3f}" for k in range(6)) + " |")
    L.append("| the same without the 13 files | " + " / ".join(f"{sh2[str(k)]:.3f}" for k in range(6)) + " |")
    L += ["", f"D1 discharges among the 13 files with invalid NBI-2 power columns: {b['d1_discharges_among_them']}. Plasma current, "
          "field and density are not in the files and are not reported.", "",
          f"Readout run time {o['runtime_readout_s']} s.", ""]
    return "\n".join(L)
