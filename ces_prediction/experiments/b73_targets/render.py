"""B.73: docs/paper/b73_targets.md from the readout dictionary (readout.py)."""
from __future__ import annotations

import common as C

INC_COLS = ("TX/T0", "KX/K0", "TB/T0", "TE/T0", "TXlag/T0", "c/t as built", "c2/t2 as built", "c/t gain-fitted",
            "c2/t2 gain-fitted")
INC_HEAD = {"TX/T0": "trees TX over T0", "KX/K0": "filters KX over K0", "TB/T0": "TB over T0 (BES only)",
            "TE/T0": "TE over T0 (ECEI only)", "TXlag/T0": "TXlag over T0 (one row old)",
            "c/t as built": "network over twin, as built", "c2/t2 as built": "two-stage over two-stage twin, as built",
            "c/t gain-fitted": "network over twin, gain fitted", "c2/t2 gain-fitted": "two-stage over two-stage twin, gain fitted"}
DES = ("V1", "V2", "V3", "V4a", "V4b")
DLAB = {"V1": "V1 random (r2-r4)", "V2": "V2 cluster-blocked (3 assignments)", "V3": "V3 rolling origin (fifths 2-5 pooled)",
        "V4a": "V4 chronological: fit to 31991, score from 32312", "V4b": "V4 chronological: fit to 32310, score from 32312"}
ELAB = {"G": "G causal GP", "H": "H hold", "M5": "M5 median of last five", "c": "network (first stage), as built",
        "t": "history-only twin, as built", "c2": "two-stage estimator, as built", "t2": "two-stage twin, as built"}


def s3(x):
    return f"{x:+.3f}"


def iv(v):
    return f"[{v[0]:+.3f}, {v[1]:+.3f}]"


def cellf(v, pct=True):
    p = f"{s3(v['point'])}" + (f" ({100 * v['point']:+.1f} %)" if pct else "")
    return f"{p}<br>d {iv(v['discharge'])}<br>c {iv(v['cluster'])}"


def elab(e):
    if e in ELAB:
        return ELAB[e]
    if e.startswith("cL["):
        return "network, gain fitted " + e[2:]
    if e.startswith("c2L["):
        return "two-stage, gain fitted " + e[3:]
    return e


def render(o):
    L = ["# B.73 — what the fast channels add: by target, by validation design, without a network", "",
         f"Registered: `{o['registration']}`, with Addendum 1 of the session lead (strata on the inputs, Q11-Q13) and Addendum 2 "
         "of the implementing agent (implementation choices), both written before any readout. CPU only; no network "
         "trained or run. T_i, cut population, the D1 rows of B.70 (222,672 rows, 635 discharges). Exploratory on used data "
         "(registration section 1): the reviewers had read forms of several of these numbers on the same discharges.", "",
         "Conventions. Skill of A over B = 1 - SSE(A) / SSE(B) on identical rows; MSE reduction in % = 100 x skill. "
         "Increment = the estimator with fast inputs over the same estimator without them. Each comparison: point, then "
         "95 % intervals with discharges resampled (d) and with shot clusters resampled (c), 20,000 draws "
         "(V1: a re-partition drawn first; V2: an assignment drawn first; V3, V4: the scored units). "
         "Verdicts on the cluster form. Point under V1 and V2 = mean over the three re-partitions or assignments. "
         "Checks: `docs/paper/b73_checks.md` (all pass).", "",
         "Order of work. The strata of Addendum 1 (`strata_inputs.py`) were computed before the first run of this readout. "
         "Before them, these results existed: checks (a), (b), (c) and (e), and the filter parameters printed in the fit "
         "log. No increment, skill, AUC or statement had been computed.", ""]
    # ---- statements
    L += ["## Statements (cluster interval)", "",
          "| Id | statement | value: point, d interval, c interval | verdict | registered expectation | note |", "|---|---|---|---|---|---|"]
    for k, s in o["statements"].items():
        v = s["value"]
        if "point" in v:
            val = f"{s3(v['point'])}; d {iv(v['discharge'])}; c {iv(v['cluster'])}"
        else:
            val = "<br>".join(f"{kk}: {s3(vv['point'])}; d {iv(vv['discharge'])}" + (f"; c {iv(vv['cluster'])}" if "cluster" in vv else "")
                              for kk, vv in v.items())
        L.append(f"| {k} | {s['text']} | {val} | "
                 f"**{'holds' if s['holds'] else 'does not hold'}** | {s['expected']} | {s.get('note', '')} |")
    L += ["", "Q1 is a ratio of MSE reductions (L5 over F); Q9 a difference of skills (F minus L5); Q10 a difference of areas "
              "under the ROC curve. Q7 reads 'not below zero' as 'the interval does not lie entirely below zero' (Addendum 2 "
              "section 8). Q13 is judged on discharge intervals (Addendum 1).", ""]
    # ---- main table
    L += ["## The fast-channel increment, design by target", "",
          "Rows = design; columns = learner; one panel per target. Each cell: MSE skill (MSE reduction in %), d interval, "
          "c interval. Network-based rows exist under V1 only; TB, TE, TXlag under V1 and V4 only; filters for F and L5 only. "
          "Each fitted estimator is fitted to the target it is scored on (as-built network rows are not fitted to any target).", ""]
    for X in C.TARGETS:
        des = [d for d in DES if f"{d}|{X}" in o["increments"]]
        cols = [c for c in INC_COLS if any(c in o["increments"][f"{d}|{X}"]["cells"] for d in des)]
        L += [f"### Target {X}", "", "| design | rows / discharges / clusters | " + " | ".join(INC_HEAD[c] for c in cols) + " |",
              "|---|---|" + "---|" * len(cols)]
        for d in des:
            I = o["increments"][f"{d}|{X}"]
            L.append(f"| {DLAB[d]} | {I['rows']:,} / {I['discharges']} / {I['clusters']} | " +
                     " | ".join(cellf(I["cells"][c]) if c in I["cells"] else "-" for c in cols) + " |")
        L.append("")
    # ---- cross-target
    L += ["## Against F and against L5 on identical rows (the L5 rows)", "",
          "Each learner fitted to F scored against F, and fitted to L5 scored against L5, on the rows where L5 is defined. "
          "Ratio = MSE reduction against L5 / that against F; difference = skill against F minus skill against L5.", "",
          "| design | learner | against F | against L5 | ratio L5 / F | difference F - L5 |", "|---|---|---|---|---|---|"]
    for d in DES:
        xt = o["cross_target_L5_rows"][d]
        for lab, v in xt["cells"].items():
            L.append(f"| {DLAB[d]} | {lab} | {cellf(v['against_F'])} | {cellf(v['against_L5'])} | "
                     f"{cellf(v['ratio_L5_over_F'], False)} | {cellf(v['difference_F_minus_L5'], False)} |")
    L.append("")
    # ---- per estimator
    L += ["## Every estimator: skill over G and over H, RMSE, median absolute error", "",
          "Points of RMSE and median absolute error: mean over re-partitions (V1) or assignments (V2). Estimator names: "
          "T0, TX, TB, TE, TXlag [target fitted to]; K0, KX filters [target fitted to]; G causal GP; H hold; M5 median of "
          "the last five values; c, t, c2, t2 the stored network-based estimates (V1).", ""]
    for d in DES:
        for X in C.TARGETS:
            key = f"{d}|{X}"
            if key not in o["per_estimator"]:
                continue
            P = o["per_estimator"][key]
            L += [f"### {DLAB[d]}, target {X}", "", "| estimator | skill over G | skill over H | RMSE (eV) | median abs. error (eV) |",
                  "|---|---|---|---|---|"]
            for e, v in P.items():
                if "rmse_eV" not in v:
                    continue
                g = cellf(v["over_G"], False) if "over_G" in v else "-"
                h = cellf(v["over_H"], False) if "over_H" in v else "-"
                L.append(f"| {elab(e)} | {g} | {h} | {v['rmse_eV']:.1f} | {v['median_abs_eV']:.1f} |")
            for e in ("TX vs c2 gain-fitted", "TX vs c2 as built", "T0 vs t2 as built"):
                if e in P:
                    L.append(f"| skill: {e.replace(' vs ', ' over ')} | {cellf(P[e], False)} | | | |")
            L.append("")
    # ---- strata
    L += ["## V1 by chronological block and by fifth of the BES level of the row", "",
          "Strata by a predictor or by shot number only. No stratum is defined by the scored value or by later values. "
          f"BES level = raw mean of the nine BES channels at row t; quintile edges {', '.join(f'{x:.3g}' for x in o['extra']['bes_level_quintile_edges'])}.", ""]
    for X in C.TARGETS:
        keys = [k for k in o["strata"] if k.endswith(f"|{X}")]
        cols = [c for c in INC_COLS + ("T0/G",) if c in o["strata"][keys[0]]["cells"]]
        L += [f"### Target {X}", "", "| stratum | rows / discharges / clusters | " + " | ".join(INC_HEAD.get(c, "T0 over G") for c in cols) + " |",
              "|---|---|" + "---|" * len(cols)]
        for k in keys:
            S = o["strata"][k]
            L.append(f"| {k.split('|')[0]} | {S['rows']:,} / {S['discharges']} / {S['clusters']} | " +
                     " | ".join(cellf(S["cells"][c]) for c in cols) + " |")
        L.append("")
    L += lead_sections(o)
    # ---- gains
    L += ["## Fitted gains", "", "λ (V1; least-squares slope through the origin of (target - history-only estimate) on the "
          "correction, fitted on the other four folds): mean [smallest, largest] over the 15 folds.", "",
          "| pair | " + " | ".join(C.TARGETS) + " |", "|---|" + "---|" * len(C.TARGETS)]
    for a, lab in (("c", "network over twin (c - t)"), ("c2", "two-stage over two-stage twin (c2 - t2)")):
        L.append(f"| {lab} | " + " | ".join(
            f"{o['lambda']['summary'][f'{a}|{X}']['mean']:.3f} [{o['lambda']['summary'][f'{a}|{X}']['min']:.3f}, {o['lambda']['summary'][f'{a}|{X}']['max']:.3f}]"
            for X in C.TARGETS) + " |")
    L += ["", "Filter gains of KX (b_c: common-mode BES drive; b_g: radial-difference drive), mean [smallest, largest] over the "
              "units of a design:", "", "| design | units | b_c fitted to F | b_g fitted to F | b_c fitted to L5 | b_g fitted to L5 |",
          "|---|---|---|---|---|---|"]
    fs = o["filter_gains"]["summary"]
    for d in ("V1", "V2", "V3", "V4"):
        f_, l_ = fs[f"{d}|F"], fs[f"{d}|L5"]
        L.append(f"| {d} | {f_['units']} | {f_['b_c'][0]:.3f} [{f_['b_c'][1]:.3f}, {f_['b_c'][2]:.3f}] | "
                 f"{f_['b_g'][0]:.3f} [{f_['b_g'][1]:.3f}, {f_['b_g'][2]:.3f}] | {l_['b_c'][0]:.3f} [{l_['b_c'][1]:.3f}, {l_['b_c'][2]:.3f}] | "
                 f"{l_['b_g'][0]:.3f} [{l_['b_g'][1]:.3f}, {l_['b_g'][2]:.3f}] |")
    L.append("")
    # ---- usability
    U = o["usability"]
    lk = U.get("label_kinds") or {}
    L += ["## Usability of the frame (classification)", "",
          f"Label: grid row without a usable T_i, among grid rows of D1 discharges with at least one measured value in the ten "
          f"rows before. Positives {lk.get('positives', '-'):,}: missing in the file {lk.get('missing_in_file', '-'):,}, above "
          f"3 keV {lk.get('above_3keV', '-'):,}, held repeat {lk.get('held_repeat', '-'):,}. Rows without the fast scalars are "
          f"left out of both classifiers (example units: {U['excluded_rows_without_fast_scalars_example_units']}).", "",
          "| design | rows | positives | AUC history | AUC history + fast | difference |", "|---|---|---|---|---|---|"]
    for d in ("V1", "V4a", "V4b"):
        u = U[d]
        L.append(f"| {DLAB[d]} | {u['rows']:,} | {u['positives']:,} | {u['H']['point']:.4f}<br>d [{u['H']['discharge'][0]:.4f}, {u['H']['discharge'][1]:.4f}]<br>c [{u['H']['cluster'][0]:.4f}, {u['H']['cluster'][1]:.4f}] | "
                 f"{u['HX']['point']:.4f}<br>d [{u['HX']['discharge'][0]:.4f}, {u['HX']['discharge'][1]:.4f}]<br>c [{u['HX']['cluster'][0]:.4f}, {u['HX']['cluster'][1]:.4f}] | "
                 f"{u['diff']['point']:+.4f}<br>d [{u['diff']['discharge'][0]:+.4f}, {u['diff']['discharge'][1]:+.4f}]<br>c [{u['diff']['cluster'][0]:+.4f}, {u['diff']['cluster'][1]:+.4f}] |")
    L.append("")
    # ---- V3 per fifth
    L += ["## V3: each scored fifth (points)", "", "| unit (fifth scored) | target | rows | TX over T0 | KX over K0 | T0 over G |", "|---|---|---|---|---|---|"]
    for k, v in o["extra"]["V3_per_fifth"].items():
        un, X = k.split("|")
        L.append(f"| {un} (fifth {int(un[-1]) + 1}) | {X} | {v['rows']:,} | {s3(v['TX/T0'])} | {s3(v['KX/K0'])} | {s3(v['T0/G'])} |")
    L.append("")
    # ---- designs and counts
    ds = o["designs"]
    ri = o["rows_info"]
    L += ["## Designs and counts", "",
          f"- Shot clusters (consecutive delivered shot numbers at most 3 apart): 86 with D1 discharges (89 over all files); "
          f"size 1 to {ds['cluster_sizes_d1_discharges']['max']} discharges, median {ds['cluster_sizes_d1_discharges']['median']:.0f}; "
          f"largest five {ds['cluster_sizes_d1_discharges']['largest_five']}.",
          f"- V2 fold sizes (discharges): seed 1 {ds['v2_seed1_fold_discharges']}, seed 2 {ds['v2_seed2_fold_discharges']}, seed 3 {ds['v2_seed3_fold_discharges']}.",
          "- V3 fifths: " + "; ".join(f"{f['shots'][0]}-{f['shots'][1]} ({f['discharges']})" for f in ds["v3_fifths"]) + ".",
          f"- V4: fit to 31991 {ds['v4']['fit_a']} discharges; fit to 32310 {ds['v4']['fit_b']}; scored {ds['v4']['test']}.",
          "- D1 rows with the target defined: " + ", ".join(f"{k} {v:,}" for k, v in ri["target_rows"].items()) + ".",
          f"- Rows without a complete input set (they keep G in every tree estimator): {100 * ri['incomplete_share_all_d1_rows']:.2f} % of the D1 rows.",
          f"- Readout run time {o['runtime_readout_s']:.0f} s.", ""]
    cc = o["extra"]["V4_concentration_TX_over_T0"]
    L += ["## Remarks (facts, no verdicts)", "",
          "- Under V4 the tree increment against L5 exceeds that against F (the opposite of V1). Concentration of TX over T0 "
          "over the scored discharges: " + "; ".join(
              f"{k}: {v['discharges']} discharges, the five largest reductions hold {v['share_top5']:.2f} (shots {v['top5_shots']}), "
              f"{v['discharges_with_negative_reduction']} discharges negative" for k, v in cc.items()) + ".",
          "- The boosted trees respond to rounding-level changes of their inputs: between two implementations whose fast "
          "scalars differed by at most about 1e-7, TX over T0 on a re-partition moved by up to 0.002 (`b73_checks.md`).",
          "- Of the 95 V4 test discharges, 83 have a row with L5 defined.", ""]
    L += ["## Limits (registration section 9, unchanged)", "",
          "- Exploratory on used data. The following level is built from reported values; a bias of the analysis that lasts "
          "50 ms or more is in the target too. Shot clusters are a proxy for sessions. V3 confounds 'earlier only' with "
          "'fewer fit discharges' for the early fifths; V4 is one block of 95 discharges. The network-based rows exist under V1 "
          "only; the gain λ of a fold is fitted on rows whose networks had this fold's discharges in training (one scalar). "
          "Whether the part that follows the fast channels is the fit, a displacement of the edge profile or the ion "
          "temperature is not decided by any readout here.", ""]
    return "\n".join(L)


def lead_sections(o):
    A = o["lead_addendum"]
    L = ["## Addendum 1 (session lead): strata on the inputs, the rotation, rows without beam emission", "",
         "Implemented as in Addendum 2 section 12. The thresholds were set by a reviewer after looking at the data and "
         "are taken over unchanged. Each estimator keeps its fit on all fit rows of its target and is only scored by "
         "class. Share of the F-reduction = (SSE(B) - SSE(A)) on the class's F rows over the same on all F rows.", ""]
    pcols = ("TX/T0", "KX/K0", "c2/t2 as built", "c/t as built", "c2/t2 gain-fitted", "c/t gain-fitted")
    for scheme, title in (("A1", "A1: by the BES sample relative to its recent level (r_B)"),
                          ("A2", "A2: by the position of the row relative to absent frames")):
        for d, od in A[scheme].items():
            cols = [c for c in pcols if c in next(iter(od.values()))]
            L += [f"### {title}, {DLAB[d]}", "",
                  "| class | share of rows (F rows) | rows F / L5 | " + " | ".join(
                      f"{INC_HEAD.get(c, c)}: against F | against L5 | share of the F-reduction" for c in cols) + " |",
                  "|---|---|---|" + "---|---|---|" * len(cols)]
            for lab, v in od.items():
                cells = []
                for c in cols:
                    e = v.get(c, {})
                    cells += [cellf(e["against_F"]) if "against_F" in e else "-",
                              cellf(e["against_L5"]) if "against_L5" in e else "-",
                              cellf(e["share_of_F_reduction"], False) if "share_of_F_reduction" in e else "-"]
                L.append(f"| {lab} | {v['share_of_rows']:.3f} | {v['rows_F']:,} / {v['rows_L5']:,} | " + " | ".join(cells) + " |")
            L.append("")
    a3 = A["A3"]
    L += ["### A3: T_i, fresh V_rot and the ECEI level against the BES sample (no estimator)", "",
          f"Regular rows with r_B defined: {a3['rows_regular_with_r_B']:,} (r_B below 0.40: {a3['below_0.40']:,}, not in the table). "
          f"Delivered rotation values equal to the previous row's: {a3['rotation_values_equal_to_previous_row_share']:.3f}.", "",
          "| r_B class | rows with T_i ratio | median T_i ratio | rows with fresh V_rot ratio | median V_rot ratio | median ECEI ratio |",
          "|---|---|---|---|---|---|"]
    f = lambda x: "-" if x is None else f"{x:.3f}"   # noqa: E731
    for t in a3["table"]:
        L.append(f"| {t['class']} | {t['rows_T_i']:,} | {f(t['median_T_i_ratio'])} | {t['rows_V_rot']:,} | "
                 f"{f(t['median_V_rot_ratio'])} | {f(t['median_ECEI_ratio'])} |")
    L += ["", "| slope of the log ratio on log r_B | rows / discharges | T_i | V_rot |", "|---|---|---|---|"]
    for k, v in a3["slopes"].items():
        L.append(f"| {k} | {v['rows']:,} / {v['discharges']} | {v['T_i']['point']:.3f} [{v['T_i']['discharge'][0]:.3f}, {v['T_i']['discharge'][1]:.3f}] | "
                 f"{v['V_rot']['point']:+.3f} [{v['V_rot']['discharge'][0]:+.3f}, {v['V_rot']['discharge'][1]:+.3f}] |")
    a4 = A["A4"]
    L += ["", "### A4: rows exposed without beam emission", "",
          f"Rows whose nine BES channels are all below 0.4 of their median over the ten rows before: {a4['rows_exposed_without_beam']:,} "
          f"(of {a4['rows_examined']:,} rows examined) in {a4['discharges']} discharges; {a4['with_T_i_value']:,} carry a T_i value "
          f"(cut, held-free grid); {a4['D1_rows']:,} are D1 rows.", ""]
    if "A4_sensitivity" in A:
        L += ["| target | rows scored (removed) | TX over T0, refitted without those rows | TX over T0, fits of the main table, same rows |",
              "|---|---|---|---|"]
        for X, v in A["A4_sensitivity"].items():
            L.append(f"| {X} | {v['rows']:,} ({v['rows_removed']}) | {cellf(v['refitted without them'])} | {cellf(v['fits as in the main table'])} |")
        L.append("")
    return L
