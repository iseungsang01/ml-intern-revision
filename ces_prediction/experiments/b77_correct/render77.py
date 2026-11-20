"""B.77: docs/paper/b77_correct.md from the readout dictionary (readout77.py)."""
from __future__ import annotations

DES = ("V1", "V2", "V3", "V4a", "V4b")
DLAB = {"V1": "V1 random (r2-r4)", "V2": "V2 cluster-blocked (3 assignments)", "V3": "V3 rolling origin (fifths 2-5 pooled)",
        "V4a": "V4 chronological: fit to 31991, score from 32312", "V4b": "V4 chronological: fit to 32310, score from 32312"}
EST = ("O", "MED3", "G", "H", "T0", "TX", "C0", "CX", "CB", "CE")
CLFS = ("PH", "PHX", "FH", "FHX")


def s3(x):
    return f"{x:+.3f}"


def iv(v):
    return f"[{v[0]:+.3f}, {v[1]:+.3f}]"


def cellf(v):
    return f"{s3(v['point'])}<br>d {iv(v['discharge'])}<br>c {iv(v['cluster'])}"


def ratf(v, nd=3):
    f = f"{{:.{nd}f}}"
    return (f"{f.format(v['point'])}<br>d [{f.format(v['discharge'][0])}, {f.format(v['discharge'][1])}]"
            f"<br>c [{f.format(v['cluster'][0])}, {f.format(v['cluster'][1])}]")


def mse_f(v):
    return f"{v['point']:.0f}<br>d [{v['discharge'][0]:.0f}, {v['discharge'][1]:.0f}]<br>c [{v['cluster'][0]:.0f}, {v['cluster'][1]:.0f}]"


def render(o):
    rs = o["row_set"]
    L = ["# B.77 — the frame's own reading, corrected and flagged with the fast samples of the same frame", "",
         f"Registered: `{o['registration']}`; implementation choices in `{o['addendum']}`, written before any fit and any "
         "readout. CPU only; no network trained or run. T_i, cut population; rows: the D1 rows of B.73 complete under rule D10 "
         f"of B.75 with the following level L5 defined ({rs['rows']:,} of {rs['d1_rows']:,} D1 rows; {rs['discharges']} "
         f"discharges; {rs['clusters']} shot clusters). Exploratory on used data (registration section 1): rows, target, "
         "designs and fast inputs are those of B.73 and B.75; the estimators that read the frame's own value are new.", "",
         "Estimators (all scored against L5, the median of the measured values among rows t+1 to t+5): O the reported value "
         "of frame t; MED3 the median of it and the two newest earlier measured values; G the causal GP and H the hold under "
         "D10 (beside, no statement uses them); T0, TX B.75's estimators under D10 refitted on these rows (history; history "
         "plus fourteen fast scalars); C0 = G + boosted regression on the ten history scalars and four scalars of the frame's "
         "value; CX C0's inputs plus the fourteen fast scalars; CB, CE C0's inputs plus the BES half or the ECEI half.", "",
         "Conventions. Skill of A over B = 1 - SSE(A) / SSE(B) on identical rows; MSE reduction in % = 100 x skill. Each "
         "comparison: point, then 95 % intervals with discharges resampled (d) and with shot clusters resampled (c), 20,000 "
         "draws (V1: a re-partition drawn first; V2: an assignment drawn first; V3, V4: the scored units). Verdicts on the "
         "cluster form. Points under V1 and V2 = mean over the three re-partitions or assignments. RMSE in eV, MSE in eV^2. "
         "Checks: `docs/paper/b77_checks.md`.", "",
         "Order of work. Before this readout, these results existed: the checks of registration section 9 (and the added "
         "ones of Addendum 1 section 11) and the row counts of Addendum 1 section 1. No MSE, skill, AUC, screening value or "
         "statement of this batch had been computed.", ""]
    # ---- statements
    L += ["## Statements (cluster interval)", "",
          "| Id | statement | value: point; d interval; c interval | verdict | registered expectation | expectation met | note |",
          "|---|---|---|---|---|---|---|"]
    for k, s in o["statements"].items():
        v = s["value"]
        if "point" in v:
            val = f"{v['point']:+.3f}; d {iv(v['discharge'])}; c {iv(v['cluster'])}" if k != "A9" else \
                f"{v['point']:.3f}; d [{v['discharge'][0]:.3f}, {v['discharge'][1]:.3f}]; c [{v['cluster'][0]:.3f}, {v['cluster'][1]:.3f}]"
        else:
            val = "<br>".join(f"{kk}: {vv['point']:+.3f}; d {iv(vv['discharge'])}; c {iv(vv['cluster'])}" for kk, vv in v.items())
        e = o["expectations"][k]
        met = {True: "yes", False: "no", None: "n/a (uncertain)"}[e["verdict_as_expected"]]
        if e["point_in_expected_range"] is not None:
            met += f"; point in the stated range: {'yes' if e['point_in_expected_range'] else 'no'}"
        L.append(f"| {k} | {s['text']} | {val} | **{'holds' if s['holds'] else 'does not hold'}** | {s['expected']} | {met} | {s.get('note', '')} |")
    L += ["", "A1, A2, A10 and A3-A6 are MSE skills; A7 is a difference of skills; A8 differences of areas under the ROC curve; "
              "A9 a ratio of MSEs (judged on its upper end below 1); A11 holds when the lower end is at most zero. 'V4' without "
              "a qualifier is V4a; V4b is in the note.", ""]
    # ---- RMSE
    L += ["## RMSE against L5 by design (eV)", "",
          "| design | rows | " + " | ".join(EST) + " |", "|---|---|" + "---|" * len(EST)]
    for d in DES:
        a = o["designs"][d]["all"]
        L.append(f"| {DLAB[d]} | {a['counts']['rows']:,} | " + " | ".join(f"{a['rmse_eV'][e]:.1f}" for e in EST) + " |")
    # ---- skills
    cols = ("C0/O", "C0/T0", "TX/T0", "CX/C0", "CB/C0", "CE/C0", "CX/C0 minus TX/T0")
    L += ["", "## Skills by design", "", "| design | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    for d in DES:
        a = o["designs"][d]["all"]
        L.append(f"| {DLAB[d]} | " + " | ".join(cellf(a[c]) for c in cols) + " |")
    cols = tuple(f"{e}/O" for e in EST[1:])
    L += ["", "## Skill over the frame's own reading O", "", "| design | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    for d in DES:
        a = o["designs"][d]["all"]
        L.append(f"| {DLAB[d]} | " + " | ".join(cellf(a[c]) for c in cols) + " |")
    # ---- V1 strata
    v1 = o["designs"]["V1"]
    cols = ("CX/C0", "CB/C0", "CE/C0", "TX/T0", "C0/T0", "C0/O", "CX/C0 minus TX/T0")
    L += ["", "## V1 by stratum on an input (each estimator keeps its fit)", "",
          "Chronological block of the test discharge; class of r_B, the BES sample relative to its recent median "
          "(B.73 Addendum 1). The class 'r_B undefined' is reported beside.", "",
          "| stratum | rows | discharges | clusters | RMSE O / C0 / CX (eV) | " + " | ".join(cols) + " |",
          "|---|---|---|---|---|" + "---|" * len(cols)]
    for s, a in v1.items():
        if s == "all":
            continue
        c = a["counts"]
        L.append(f"| {s} | {c['rows']:,} | {c['discharges']} | {c['clusters']} | {a['rmse_eV']['O']:.1f} / {a['rmse_eV']['C0']:.1f} / "
                 f"{a['rmse_eV']['CX']:.1f} | " + " | ".join(cellf(a[k]) for k in cols) + " |")
    # ---- controls
    a = v1["all"]
    lg = o["lag_control"]
    L += ["", "## Controls (V1)", "",
          "| control | rows | RMSE (eV) | quantity | value |", "|---|---|---|---|---|",
          f"| placebo CXp: the fourteen fast scalars permuted over rows (within fit rows and within test rows), CX refitted | "
          f"{a['counts']['rows']:,} | CXp {a['rmse_eV']['CXp']:.1f}, C0 {a['rmse_eV']['C0']:.1f}, CX {a['rmse_eV']['CX']:.1f} | "
          f"CXp over C0 | {cellf(a['CXp/C0 (placebo)'])} |",
          f"| | | | CX over CXp | {cellf(a['CX/CXp'])} |",
          f"| row t - 1: CXlag, the fast scalars of row t - 1 in place of those of row t | {lg['counts']['rows']:,} "
          f"({lg['rows_of_set_without_lag_inputs']:,} rows of the set without them) | CXlag {lg['rmse_eV']['CXlag']:.1f}, "
          f"C0 {lg['rmse_eV']['C0']:.1f}, CX {lg['rmse_eV']['CX']:.1f} | CXlag over C0 | {cellf(lg['CXlag/C0'])} |",
          f"| | | | CX over C0 (same rows) | {cellf(lg['CX/C0'])} |",
          f"| | | | CXlag over CX | {cellf(lg['CXlag/CX'])} |",
          f"| | | | (CX over C0) - (CXlag over C0) | {cellf(lg['CX/C0 minus CXlag/C0'])} |"]
    # ---- AUC
    L += ["", "## Flags: area under the ROC curve", "",
          f"Label: y / L5 above {1.35} or below {0.74}. Positives on the row set: {rs['label_positives']:,} of {rs['rows']:,} "
          f"({100 * rs['label_share']:.1f} %; above 1.35: {rs['label_above_1.35']:,}; below 0.74: {rs['label_below_0.74']:,}). "
          "PH: history; PHX: history + fourteen fast scalars; FH: history + four scalars of the frame's value; FHX: FH + "
          "fourteen fast scalars. V1, V2: AUC over all rows of a re-partition or assignment (cross-fitted), mean over the three.", "",
          "| design | rows (positives) | " + " | ".join(CLFS) + " | PHX - PH | FHX - FH | FH - PH | FHX - PHX |",
          "|---|---|" + "---|" * 8]
    for d in DES:
        x = o["auc"][d]
        L.append(f"| {DLAB[d]} | {x['rows']:,} ({x['positives']:,}) | " + " | ".join(ratf(x[c], 4) for c in CLFS) + " | "
                 + " | ".join(cellf(x[k]) for k in ("PHX minus PH", "FHX minus FH", "FH minus PH", "FHX minus PHX")) + " |")
    # ---- screening
    L += ["", "## Screening: MSE of O against L5 on the kept rows (eV^2)", "",
          "In each test unit the rows ranked by the classifier's probability; the q highest dropped (Addendum 1 section 6). "
          "Each cell: MSE, then MSE over no screening.", ""]
    for d in DES:
        sc = o["designs"][d]["all"]["screening"]
        L += [f"### {DLAB[d]}", "", f"No screening: {mse_f(sc['no screening']['mse_eV2'])}", "",
              "| classifier | q = 5 % | q = 10 % | q = 20 % |", "|---|---|---|---|"]
        for c in ("FHX", "FH", "PHX", "PH"):
            L.append(f"| {c} | " + " | ".join(f"{mse_f(sc[f'{c}|q{qq}']['mse_eV2'])}<br>ratio {ratf(sc[f'{c}|q{qq}']['over no screening'])}"
                                              for qq in (5, 10, 20)) + " |")
        L += ["", "| ratio of MSEs on kept rows | q = 5 % | q = 10 % | q = 20 % |", "|---|---|---|---|"]
        for a_, b_ in (("FHX", "FH"), ("PHX", "PH"), ("FH", "PH"), ("FHX", "PHX")):
            L.append(f"| {a_} over {b_} | " + " | ".join(ratf(sc[f"{a_} over {b_}|q{qq}"]) for qq in (5, 10, 20)) + " |")
        L.append("")
    # ---- counts
    L += ["## Row counts", "", "| design | scored rows | discharges | clusters | fit rows per unit (min-max) | units |", "|---|---|---|---|---|---|"]
    for d in DES:
        c = o["counts"][d]
        fr = c["fit_rows_per_unit"]["C0"]
        L.append(f"| {DLAB[d]} | {c['scored']['rows']:,} | {c['scored']['discharges']} | {c['scored']['clusters']} | "
                 f"{fr['min']:,}-{fr['max']:,} | {len(c['units'])} |")
    ch = o["checks"]
    L += ["", "## Checks (details in `docs/paper/b77_checks.md`)", "",
          f"- 1a: T0 and TX against B.75's D10 L5 estimates on the rows both have: median absolute deviation "
          f"{ch['compare']['T0']['median_abs_dev_eV']:.2f} / {ch['compare']['TX']['median_abs_dev_eV']:.2f} eV, largest "
          f"{ch['compare']['T0']['max_abs_dev_eV']:.1f} / {ch['compare']['TX']['max_abs_dev_eV']:.1f} eV (row sets differ; reported).",
          f"- 1b: this code on B.75's rows reproduces B.75 bit for bit: {ch['b75']['passes']}.",
          f"- 1c: C0 without the four frame scalars equals T0 bit for bit: {ch['frame']['passes']}.",
          f"- 2: no input or estimate of C0 or CX changes when values of rows after t change: {ch['leakage']['passes_leakage']}; "
          f"positive control: {ch['leakage']['passes_positive_control']}.",
          f"- 3: inputs and target: {ch['data']['passes']}.", "",
          f"Readout run time {o['runtime_readout_s']:.0f} s.", ""]
    return "\n".join(L)
