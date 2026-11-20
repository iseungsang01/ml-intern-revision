"""B.75: docs/paper/b75_availability.md from the readout dictionary (readout.py)."""
from __future__ import annotations

DES = ("V1", "V2", "V3", "V4a", "V4b")
DLAB = {"V1": "V1 random (r2-r4)", "V2": "V2 cluster-blocked (3 assignments)", "V3": "V3 rolling origin (fifths 2-5 pooled)",
        "V4a": "V4 chronological: fit to 31991, score from 32312", "V4b": "V4 chronological: fit to 32310, score from 32312"}
FIXED = ("D10", "D50", "D100", "D300", "D650", "D1000")
RLAB = {"D10": "D10", "D50": "D50", "D100": "D100", "D300": "D300", "D650": "D650", "D1000": "D1000", "S": "S (schedule)",
        "D10S": "D10 history on the S set"}


def s3(x):
    return f"{x:+.3f}"


def iv(v):
    return f"[{v[0]:+.3f}, {v[1]:+.3f}]"


def cellf(v, pct=False):
    p = s3(v["point"]) + (f" ({100 * v['point']:+.1f} %)" if pct else "")
    return f"{p}<br>d {iv(v['discharge'])}<br>c {iv(v['cluster'])}"


def ratf(v):
    return f"{v['point']:.3f}<br>d [{v['discharge'][0]:.3f}, {v['discharge'][1]:.3f}]<br>c [{v['cluster'][0]:.3f}, {v['cluster'][1]:.3f}]"


def blocks(o, d, X):
    c = o["main"][f"{d}|{X}|common"]["all"]
    s = o["main"][f"{d}|{X}|S"]["all"]
    rows = [(r, c[r], c["counts"]) for r in FIXED] + [(r, s[r], s["counts"]) for r in ("S", "D10S")]
    return rows


def render(o):
    si = o["sets"]
    L = ["# B.75 — the estimate when the CES history is as old as it really is", "",
         f"Registered: `{o['registration']}`; implementation choices in `{o['addendum']}`, written before any readout. "
         "CPU only; no network trained or run. T_i, cut population, the D1 rows of B.70 as cached by B.73 "
         f"({si['d1_rows']:,} rows, 635 discharges). Exploratory on used data (registration section 1): the rows, targets, "
         "designs and learners are those of B.73; the availability rules and the anchor inputs are new.", "",
         "**Rule S assumes zero analysis time**: a value becomes visible as soon as the next beam-off frame has been taken. "
         "This favours every estimator under S.", "",
         "Conventions. Skill of A over B = 1 - SSE(A) / SSE(B) on identical rows; MSE reduction in % = 100 x skill. "
         "Increment = TX over T0 under the same rule on the same rows. Each comparison: point, then 95 % intervals with "
         "discharges resampled (d) and with shot clusters resampled (c), 20,000 draws (V1: a re-partition drawn first; "
         "V2: an assignment drawn first; V3, V4: the scored units). Verdicts on the cluster form. Points under V1 and V2 = "
         "mean over the three re-partitions or assignments. Fixed-delay rules (D10 ... D1000) are fitted and scored on the "
         "common set, rule S and the D10 history beside it on the S set. RMSE in eV. Checks: `docs/paper/b75_checks.md`.", "",
         "Order of work. Before this readout, these results existed: the five checks of registration section 9 and the "
         "counts of the row sets. No MSE, skill or statement under any rule had been computed.", ""]
    # ---- statements
    L += ["## Statements (cluster interval)", "",
          "| Id | statement | value: point; d interval; c interval | verdict | registered expectation | expectation met | note |",
          "|---|---|---|---|---|---|---|"]
    for k, s in o["statements"].items():
        v = s["value"]
        if "point" in v:
            val = f"{s3(v['point'])}; d {iv(v['discharge'])}; c {iv(v['cluster'])}"
        else:
            val = "<br>".join(f"{kk}: {s3(vv['point'])}; d {iv(vv['discharge'])}; c {iv(vv['cluster'])}" for kk, vv in v.items())
        e = o["expectations"][k]
        met = {True: "yes", False: "no", None: "n/a (uncertain)"}[e["verdict_as_expected"]]
        if e["point_in_expected_range"] is not None:
            met += f"; point in the stated range: {'yes' if e['point_in_expected_range'] else 'no'}"
        L.append(f"| {k} | {s['text']} | {val} | **{'holds' if s['holds'] else 'does not hold'}** | {s['expected']} | {met} | {s.get('note', '')} |")
    L += ["", "R1 is a ratio of MSEs (judged on its lower end above 1.5); R3 and R13 are differences of skills; R11 and R12 are "
              "the skill of TX over the frame's own reading O. 'V4' without a qualifier is V4a (Addendum 1 section 5); V4b "
              "is in the note.", ""]
    # ---- RMSE by rule
    L += ["## RMSE by availability rule", "",
          "Rows: the rule; fixed delays on the common set, S and the D10 history beside it on the S set. H hold of the newest "
          "visible value; M5 median of the last five visible values; G causal GP of the last up to 16 visible values; T0 "
          "G + boosted correction on the history scalars; TX8 with the eight fast scalars of B.73; TX with the six anchor "
          "scalars added; X0 no CES history (common set only); O the frame's own reading (against L5 only). "
          "Points; RMSE under V1 and V2 = mean over the three splits.", ""]
    for X in ("L5", "F"):
        for d in DES:
            cols = ["H", "M5", "G", "T0", "TX8", "TX", "X0"] + (["O"] if X == "L5" else [])
            L += [f"### {DLAB[d]}, target {X}", "", "| rule | rows / discharges / clusters | " + " | ".join(cols) + " |",
                  "|---|---|" + "---|" * len(cols)]
            for r, b, cnt in blocks(o, d, X):
                rm = b["rmse_eV"]
                L.append(f"| {RLAB[r]} | {cnt['rows']:,} / {cnt['discharges']} / {cnt['clusters']} | " +
                         " | ".join(f"{rm[c]:.1f}" if c in rm else "-" for c in cols) + " |")
            L.append("")
    # ---- increments by rule and design
    for X in ("L5", "F"):
        L += [f"## The fast-channel increment TX over T0 by rule and design, target {X}", "",
              "Cells: skill, d interval, c interval.", "",
              "| design | " + " | ".join(RLAB[r] for r in FIXED + ("S", "D10S")) + " |", "|---|" + "---|" * 8]
        for d in DES:
            L.append(f"| {DLAB[d]} | " + " | ".join(cellf(b["TX/T0"]) for r, b, _ in blocks(o, d, X)) + " |")
        L.append("")
        L += [f"### TX8, TB, TE over T0 (V1, V4), target {X}", "",
              "| design | rule | TX8 over T0 (eight fast scalars) | TX over T0 (fourteen) | TB over T0 (BES half) | TE over T0 (ECEI half) |",
              "|---|---|---|---|---|---|"]
        for d in ("V1", "V4a", "V4b"):
            for r, b, _ in blocks(o, d, X):
                if "TX8/T0" in b:
                    L.append(f"| {DLAB[d]} | {RLAB[r]} | {cellf(b['TX8/T0'])} | {cellf(b['TX/T0'])} | {cellf(b['TB/T0'])} | {cellf(b['TE/T0'])} |")
        L.append("")
    # ---- history estimator over G and H; over O
    for X in ("L5", "F"):
        L += [f"## T0 over G and over H, target {X}", "", "| design | " + " | ".join(RLAB[r] for r in FIXED + ("S", "D10S")) + " |",
              "|---|" + "---|" * 8]
        for d in DES:
            L.append(f"| {DLAB[d]}: T0 over G | " + " | ".join(cellf(b["T0/G"]) for r, b, _ in blocks(o, d, X)) + " |")
            L.append(f"| {DLAB[d]}: T0 over H | " + " | ".join(cellf(b["T0/H"]) for r, b, _ in blocks(o, d, X)) + " |")
        L.append("")
    L += ["## Against the frame's own reading O (target L5)", "",
          "Skill over O = 1 - MSE(estimator) / MSE(O); above zero means closer to the following level than the frame's own "
          "reported value.", "", "| design | estimator | " + " | ".join(RLAB[r] for r in FIXED + ("S", "D10S")) + " |",
          "|---|---|" + "---|" * 8]
    for d in DES:
        for lab in ("H", "G", "T0", "TX"):
            L.append(f"| {DLAB[d]} | {lab} over O | " + " | ".join(cellf(b[f"{lab}/O"]) for r, b, _ in blocks(o, d, "L5")) + " |")
    L.append("")
    # ---- staleness and recovered share
    for X in ("L5", "F"):
        L += [f"## Cost of staleness and recovered share, target {X}", "",
              "Cost of staleness = MSE under the rule / MSE under D10 (S: over the D10 history fitted on the S set). Recovered "
              "share = (MSE(T0) - MSE(TX)) / (MSE(T0) - MSE(T0 under D10)) under the same rule and rows.", "",
              "| design | rule | cost of staleness, T0 | cost of staleness, TX | recovered share |", "|---|---|---|---|---|"]
        for d in DES:
            for r, b, _ in blocks(o, d, X):
                if "staleness T0" in b:
                    L.append(f"| {DLAB[d]} | {RLAB[r]} | {ratf(b['staleness T0'])} | {ratf(b['staleness TX'])} | {ratf(b['recovered share'])} |")
        L.append("")
    # ---- applied without refit
    for X in ("L5", "F"):
        L += [f"## Applied without refit (the D10 fit on the inputs of the rule) beside refitted, target {X}", "",
              "Fixed delays: the D10 fit on the common set; S: the D10 history fitted on the S set. Skill of the refitted "
              "estimator over the one applied without refit; cost of staleness of the applied one against the refitted D10.", "",
              "| design | rule | RMSE T0 applied / refitted | RMSE TX applied / refitted | T0 refitted over applied | TX refitted over applied | TX over T0, applied | cost of staleness T0 applied | cost of staleness TX applied |",
              "|---|---|---|---|---|---|---|---|---|"]
        for d in DES:
            for r, b, _ in blocks(o, d, X):
                if "applied T0: refit over applied" in b:
                    rm = b["rmse_eV"]
                    L.append(f"| {DLAB[d]} | {RLAB[r]} | {rm['T0a']:.1f} / {rm['T0']:.1f} | {rm['TXa']:.1f} / {rm['TX']:.1f} | "
                             f"{cellf(b['applied T0: refit over applied'])} | {cellf(b['applied TX: refit over applied'])} | "
                             f"{cellf(b['applied TXa/T0a'])} | {ratf(b['staleness T0a'])} | {ratf(b['staleness TXa'])} |")
        L.append("")
    # ---- V1 by block
    for X in ("L5", "F"):
        L += [f"## V1 by chronological block of the test discharge, target {X}", "",
              "| block | rule | rows / discharges / clusters | TX over T0 | TX8 over T0 | TB over T0 | TE over T0 | T0 over G |",
              "|---|---|---|---|---|---|---|---|"]
        for kind, rules in (("common", FIXED), ("S", ("S", "D10S"))):
            G = o["main"][f"V1|{X}|{kind}"]
            for st in [k for k in G if k.startswith("block")]:
                for r in rules:
                    b = G[st][r]
                    cnt = G[st]["counts"]
                    f = lambda k: cellf(b[k]) if k in b else "-"   # noqa: E731
                    L.append(f"| {st[6:]} | {RLAB[r]} | {cnt['rows']:,} / {cnt['discharges']} / {cnt['clusters']} | {f('TX/T0')} | "
                             f"{f('TX8/T0')} | {f('TB/T0')} | {f('TE/T0')} | {f('T0/G')} |")
        L.append("")
    # ---- R3 detail
    L += ["## Difference of increments, D650 minus D10 (common set)", "", "| design | target | TX over T0 under D650 minus under D10 |", "|---|---|---|"]
    for d in DES:
        for X in ("L5", "F"):
            L.append(f"| {DLAB[d]} | {X} | {cellf(o['main'][f'{d}|{X}|common']['all']['D650 minus D10: TX/T0'])} |")
    L.append("")
    # ---- V3 / V3c
    V = o["v3c"]
    L += ["## V3 beside V3c (the control for the size of the fit), target L5, common set", "",
          "V3: fitted on the earlier fifths only. V3c: fitted on the same number of discharges drawn at random from all "
          "discharges outside the test fifth (seeds 1-3; a seed drawn first in each interval draw). Fifth 5 has no control.", "",
          "| fifth(s) scored | rows / discharges / clusters | rule | V3: TX over T0 | V3c: TX over T0 | V3 minus V3c | RMSE T0 V3 / V3c | RMSE TX V3 / V3c |",
          "|---|---|---|---|---|---|---|---|"]
    for k, v in V.items():
        for r in ("D10", "D650"):
            c3 = cellf(v[f"V3 TX/T0 {r}"])
            cc = cellf(v[f"V3c TX/T0 {r}"]) if f"V3c TX/T0 {r}" in v else "-"
            dd = cellf(v[f"V3 minus V3c TX/T0 {r}"]) if f"V3 minus V3c TX/T0 {r}" in v else "-"
            r0 = f"{v[f'V3 T0 RMSE {r}']:.1f} / " + (f"{v[f'V3c T0 RMSE {r}']:.1f}" if f"V3c T0 RMSE {r}" in v else "-")
            r1 = f"{v[f'V3 TX RMSE {r}']:.1f} / " + (f"{v[f'V3c TX RMSE {r}']:.1f}" if f"V3c TX RMSE {r}" in v else "-")
            L.append(f"| {k} | {v['rows']:,} / {v['discharges']} / {v['clusters']} | {r} | {c3} | {cc} | {dd} | {r0} | {r1} |")
    L.append("")
    # ---- ages
    A = o["age_under_S"]
    L += ["## Age under rule S on the S set", "",
          "Age of the newest visible value, and age of the visibility limit (stamp of the row minus the stamp of the last row "
          "before the most recent absent step); in s.", "",
          "| target set | rows | quantity | 10 % | 25 % | median | 75 % | 90 % | 0-0.1 s | 0.1-0.3 s | 0.3-1 s | above 1 s |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for X, a in A.items():
        for key, lab in (("newest_value_age_s", "newest visible value"), ("limit_age_s", "visibility limit")):
            qq, bb = a[key]["quantiles"], a[key]["bands"]
            L.append(f"| {X} | {a['rows']:,} | {lab} | " + " | ".join(f"{qq[p]:.3f}" for p in ("0.1", "0.25", "0.5", "0.75", "0.9")) +
                     " | " + " | ".join(f"{bb[b]:.3f}" for b in ("0-0.1 s", "0.1-0.3 s", "0.3-1 s", "above 1 s")) + " |")
    L.append("")
    # ---- timing
    T = o["timing"]
    L += ["## Computation time", "",
          f"Per row: `predict_gp_causal` on the row's visible values plus one `predict` call of a fitted TX model on that row; "
          f"one thread; {T['rows']:,} random rows of the common set, rule {T['rule']}; model {T['model']}. "
          f"Median {T['median_ms']:.2f} ms, 99th percentile {T['p99_ms']:.2f} ms (mean {T['mean_ms']:.2f}, largest {T['max_ms']:.2f}). "
          f"Machine: {T['cpu']}, {T['logical_cpus']} logical processors; {T['note']}.", ""]
    # ---- counts
    L += ["## Counts", "",
          f"- Common set: F {si['common_F']:,}, L5 {si['common_L5']:,} rows. S set: F {si['S_F']:,}, L5 {si['S_L5']:,} rows. "
          f"D1 rows without a visible value under S (no absent step earlier in the block): {si['no_visible_value_under_S']:,} "
          f"({100 * si['no_visible_value_under_S'] / si['d1_rows']:.1f} % of the D1 rows). Rows complete under S but not under D10: "
          f"{si['complete_S_not_D10']:,}.",
          "- Rows complete under each rule: " + ", ".join(f"{r} {x['complete']:,}" for r, x in si["per_rule"].items()) +
          "; dropped by a3 alone (a raw array mean not above zero): " + ", ".join(f"{r} {x['dropped_by_a3_alone']:,}" for r, x in si["per_rule"].items()) + ".",
          "", "| design | common F | common L5 | S set F | S set L5 | fit rows per unit, T0 L5: D10 / D650 / S / D10 on S |", "|---|---|---|---|---|---|"]
    for d in DES:
        c = o["counts"][d]
        f = lambda k: f"{c[k]['rows']:,} / {c[k]['discharges']} / {c[k]['clusters']}"   # noqa: E731
        fr = c["fit_rows_per_unit_L5"]
        g = lambda k: f"{fr[k]['min']:,}-{fr[k]['max']:,}"   # noqa: E731
        L.append(f"| {DLAB[d]} | {f('common|F')} | {f('common|L5')} | {f('S|F')} | {f('S|L5')} | "
                 f"{g('T0|D10|L5')} / {g('T0|D650|L5')} / {g('T0|S|L5')} / {g('T0|D10S|L5')} |")
    L += ["", "Cells: rows / discharges / clusters scored.", "", f"Readout run time {o['runtime_readout_s']:.0f} s.", ""]
    L += ["## Limits (registration section 11, unchanged)", "",
          "- Exploratory on used data. The anchor scalars are new inputs chosen before any readout; TX8 beside TX shows what "
          "they do.",
          "- Rule S assumes the beam-modulation analysis with zero computation time. A reader of the frame's own spectrum "
          "would give another availability; no such values exist for these discharges.",
          "- The history values are the offline product in every rule.",
          "- The following level is built from reported values; a bias of the analysis that lasts 50 ms or more is in the "
          "target too. R11 is about scatter around the level, not about a systematic error of the diagnostic.",
          "- The fixed-delay rules drop the first second of every block (the common set).",
          "- One estimator per rule is fitted. An estimator fitted once for all ages is not part of this batch; the rows "
          "'applied without refit' bound it from one side.",
          "- V4 is one block of about 95 discharges; V3c is informative for fifths 2 and 3 only.", ""]
    return "\n".join(L)


if __name__ == "__main__":
    # re-render the markdown from the readout JSON (no number is recomputed)
    import json
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import common75 as K
    K.save_text(K.OUT_MD, render(json.loads(K.OUT_JSON.read_text(encoding="utf-8"))))
    print("written", K.OUT_MD)
