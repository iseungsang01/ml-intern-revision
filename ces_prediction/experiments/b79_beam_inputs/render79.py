"""B.79: docs/paper/b79_beam_inputs.md from the readout dictionary (readout79.py).  Cell formats are B.75's
(b75_availability/render.py: s3, iv, cellf, ratf)."""
from __future__ import annotations

import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location("render75", Path(__file__).resolve().parents[1] / "b75_availability" / "render.py")
R75 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(R75)
s3, iv, cellf, ratf = R75.s3, R75.iv, R75.cellf, R75.ratf

DES = ("V1", "V2", "V3", "V4a", "V4b")
DLAB = {"V1": "V1 random (r2-r4)", "V2": "V2 cluster-blocked (3 assignments)", "V3": "V3 rolling origin (fifths 2-5 pooled)",
        "V4a": "V4 chronological: fit to 31991, score from 32312", "V4b": "V4 chronological: fit to 32310, score from 32312"}
RLAB = {"D10": "D10", "D650": "D650", "S": "S (schedule)"}
KIND = {"D10": "common", "D650": "common", "S": "S"}
NOTCH = ("within 100 ms after an absent step", "later rows after an absent step", "no absent step earlier in the block")


def cnt(c):
    return f"{c['rows']:,} / {c['discharges']} / {c['clusters']}"


def ratc(v):
    p = f"{v['point']:.3f}<br>d [{v['discharge'][0]:.3f}, {v['discharge'][1]:.3f}]<br>c [{v['cluster'][0]:.3f}, {v['cluster'][1]:.3f}]"
    nf = v.get("draws_not_finite", {})
    if any(nf.values()):
        p += f"<br>(draws not finite: d {nf['discharge']}, c {nf['cluster']})"
    return p


def blk(o, d, X, r):
    return o["main"][f"{d}|{X}|{KIND[r]}"]["all"][r]


def render(o):
    si = o["sets"]
    L = ["# B.79 — the beam waveforms as inputs: what the fast diagnostics add beyond what the actuators already say", "",
         f"Registered: `{o['registration']}`; implementation choices in `{o['addendum']}`, written before any fit and any "
         "readout. CPU only; no network trained or run. T_i, cut population, the D1 rows of B.70 as cached by B.73 "
         f"({si['d1_rows']:,} rows, 635 discharges). Exploratory on used data (registration section 1): rows, targets, "
         "designs, learners, rules and fast inputs are B.75's; the nineteen beam scalars are new.", "",
         "Learners (all fitted to target - G under the rule): T0 the ten history scalars; TN T0 and the nineteen beam "
         "scalars; TX T0 and the fourteen fast scalars of B.75; TXN TX and the nineteen beam scalars; TEN, TBN TN and the "
         "ECEI or BES half of the fast scalars; TNp TN with the beam scalars permuted over rows; TXNp TXN with the fast "
         "scalars permuted. G causal GP of the visible history; O the frame's own reading (against L5 only).", "",
         "Conventions. Skill of A over B = 1 - SSE(A) / SSE(B) on identical rows. Each comparison: point, then 95 % "
         "intervals with discharges resampled (d) and with shot clusters resampled (c), 20,000 draws (V1: a re-partition "
         "drawn first; V2: an assignment drawn first; V3, V4: the scored units). Verdicts on the cluster form. Points under "
         "V1 and V2 = mean over the three re-partitions or assignments. D10 and D650 are fitted and scored on the common "
         "set, S on the S set; the sets equal B.75's. RMSE in eV. Checks: `docs/paper/b79_checks.md`.", "",
         "The beam columns are used as delivered. In thirteen early files (shots 30801-30909) the NBI-2 power columns are on "
         "another scale (up to 144 MW; Addendum 1 section 7); a scoring-only sensitivity without those files is at the end.", "",
         "Order of work. Before this readout, these results existed: the checks of registration section 9, the comparison "
         "of this batch's T0 and TX files with B.75's, and the counts of the row sets. No MSE, skill or statement with a "
         "beam input had been computed.", ""]
    # ---- statements
    L += ["## Statements (target L5, cluster interval)", "",
          "| Id | statement | value: point; d interval; c interval | verdict | registered expectation | expectation met | note |",
          "|---|---|---|---|---|---|---|"]
    for k, s in o["statements"].items():
        v = s["value"]
        val = f"{s3(v['point'])}; d {iv(v['discharge'])}; c {iv(v['cluster'])}"
        e = o["expectations"][k]
        met = {True: "yes", False: "no", None: "n/a (uncertain)"}[e["verdict_as_expected"]]
        if e["point_in_expected_range"] is not None:
            met += f"; point in the stated range: {'yes' if e['point_in_expected_range'] else 'no'}"
        L.append(f"| {k} | {s['text']} | {val} | **{'holds' if s['holds'] else 'does not hold'}** | {s['expected']} | {met} | {s.get('note', '')} |")
    L += ["", "'V4' without a qualifier is V4a (fit to 31991); V4b is in the note.", ""]
    # ---- RMSE
    cols = ("G", "T0", "TN", "TX", "TXN", "TEN", "TBN")
    for X in ("L5", "F"):
        L += [f"## RMSE by design and rule, target {X}", "",
              "Points; under V1 and V2 the mean over the three splits." + (" O = the frame's own reading." if X == "L5" else ""), "",
              "| design | rule | rows / discharges / clusters | " + " | ".join(cols) + (" | O |" if X == "L5" else " |"),
              "|---|---|---|" + "---|" * (len(cols) + (1 if X == "L5" else 0))]
        for d in DES:
            for r in ("D10", "D650", "S"):
                b = blk(o, d, X, r)
                rm = b["rmse_eV"]
                c = o["main"][f"{d}|{X}|{KIND[r]}"]["all"]["counts"]
                L.append(f"| {DLAB[d]} | {RLAB[r]} | {cnt(c)} | " + " | ".join(f"{rm[x]:.1f}" if x in rm else "-" for x in cols) +
                         (f" | {rm['O']:.1f} |" if X == "L5" else " |"))
        L.append("")
    # ---- increments
    inc = (("TN/T0", "TN over T0 (beam over history)"), ("TXN/TN", "TXN over TN (fast over history and beam)"),
           ("TX/T0", "TX over T0 (fast over history, B.75)"), ("TXN/TX", "TXN over TX (beam over history and fast)"),
           ("TXN/T0", "TXN over T0 (both)"))
    for X in ("L5", "F"):
        L += [f"## Increments by design and rule, target {X}", "", "Cells: skill, d interval, c interval.", "",
              "| design | rule | " + " | ".join(lab for _, lab in inc) + " |", "|---|---|" + "---|" * len(inc)]
        for d in DES:
            for r in ("D10", "D650", "S"):
                b = blk(o, d, X, r)
                L.append(f"| {DLAB[d]} | {RLAB[r]} | " + " | ".join(cellf(b[k]) for k, _ in inc) + " |")
        L.append("")
    # ---- ratio
    for X in ("L5", "F"):
        L += [f"## The ratio of the two MSE reductions, target {X}", "",
              "Registered form: (skill of TXN over TN) / (skill of TX over T0), draw by draw on the same draws. Beside it the "
              "absolute form (SSE(TN) - SSE(TXN)) / (SSE(T0) - SSE(TX)). Points: mean over splits of the per-split ratio.", "",
              "| design | rule | (TXN over TN) / (TX over T0) | absolute form | TX over T0 (the denominator's skill) |", "|---|---|---|---|---|"]
        for d in DES:
            for r in ("D10", "D650", "S"):
                b = blk(o, d, X, r)
                L.append(f"| {DLAB[d]} | {RLAB[r]} | {ratc(b['ratio of skills (TXN/TN)/(TX/T0)'])} | "
                         f"{ratc(b['ratio of absolute reductions'])} | {cellf(b['TX/T0'])} |")
        L.append("")
    # ---- halves
    L += ["## The halves: TEN and TBN over TN (V1, V4)", "",
          "TEN = TN with the ECEI half of the fast scalars (four of row t, three anchor scalars); TBN with the BES half.", "",
          "| design | target | rule | TEN over TN (ECEI) | TBN over TN (BES) | TXN over TN (both) | RMSE TN / TEN / TBN / TXN |",
          "|---|---|---|---|---|---|---|"]
    for X in ("L5", "F"):
        for d in ("V1", "V4a", "V4b"):
            for r in ("D650", "S"):
                b = blk(o, d, X, r)
                rm = b["rmse_eV"]
                L.append(f"| {DLAB[d]} | {X} | {RLAB[r]} | {cellf(b['TEN/TN'])} | {cellf(b['TBN/TN'])} | {cellf(b['TXN/TN'])} | "
                         f"{rm['TN']:.1f} / {rm['TEN']:.1f} / {rm['TBN']:.1f} / {rm['TXN']:.1f} |")
    L.append("")
    # ---- notch strata
    L += ["## By time since the last notch (V1; a stratum on an input)", "",
          "Time since the last notch = s(t) - s(j+1), j+1 the first row after the most recent absent step (15-45 ms) that "
          "ends at or before t in the block of t. Within 100 ms: below 0.095 s (the first row after the step and the next "
          "nine). The third stratum holds the rows with no absent step earlier in the block; under S it is empty.", ""]
    for X in ("L5", "F"):
        L += [f"### Target {X}", "",
              "| set, rule | stratum | rows / discharges / clusters | RMSE T0 / TN / TX / TXN | TXN over TN | TX over T0 | TN over T0 | TXN over TX |",
              "|---|---|---|---|---|---|---|---|"]
        for kind, r in (("common", "D650"), ("S", "S")):
            G = o["main"][f"V1|{X}|{kind}"]
            for st in NOTCH:
                if st not in G:
                    L.append(f"| {kind}, {RLAB[r]} | {st} | 0 | - | - | - | - | - |")
                    continue
                b = G[st][r]
                rm = b["rmse_eV"]
                L.append(f"| {kind}, {RLAB[r]} | {st} | {cnt(G[st]['counts'])} | {rm['T0']:.1f} / {rm['TN']:.1f} / {rm['TX']:.1f} / {rm['TXN']:.1f} | "
                         f"{cellf(b['TXN/TN'])} | {cellf(b['TX/T0'])} | {cellf(b['TN/T0'])} | {cellf(b['TXN/TX'])} |")
        nc = o["notch_counts"]
        L += ["", "Rows per stratum (all V1 scored rows of the set): " + "; ".join(
            f"{k}: " + ", ".join(f"{kk} {vv:,}" for kk, vv in v.items()) for k, v in nc.items() if k.startswith(f"V1|{X}|")) + ".", ""]
    # ---- V1 by block
    binc = (("TN/T0", "TN over T0"), ("TXN/TN", "TXN over TN"), ("TX/T0", "TX over T0"), ("TXN/TX", "TXN over TX"),
            ("TEN/TN", "TEN over TN"), ("TBN/TN", "TBN over TN"))
    for X in ("L5", "F"):
        L += [f"## V1 by chronological block of the test discharge, target {X}", "",
              "| block | rule | rows / discharges / clusters | " + " | ".join(lab for _, lab in binc) + " |", "|---|---|---|" + "---|" * len(binc)]
        for kind, rules in (("common", ("D10", "D650")), ("S", ("S",))):
            G = o["main"][f"V1|{X}|{kind}"]
            for st in [k for k in G if k.startswith("block")]:
                for r in rules:
                    b = G[st][r]
                    L.append(f"| {st[6:]} | {RLAB[r]} | {cnt(G[st]['counts'])} | " +
                             " | ".join(cellf(b[k]) if k in b else "-" for k, _ in binc) + " |")
        L.append("")
    # ---- controls
    b = blk(o, "V1", "L5", "D650")
    rm = b["rmse_eV"]
    L += ["## Controls (V1, D650, L5)", "",
          "TNp: TN refitted with the nineteen beam scalars permuted jointly over rows (within the fit rows and within the test "
          "rows). TXNp: TXN refitted with the fourteen fast scalars permuted jointly in the same way.", "",
          "| quantity | value |", "|---|---|",
          f"| TNp over T0 (N11) | {cellf(b['TNp/T0'])} |", f"| TN over TNp | {cellf(b['TN/TNp'])} |",
          f"| TXNp over TN | {cellf(b['TXNp/TN'])} |", f"| TXN over TXNp | {cellf(b['TXN/TXNp'])} |",
          f"| RMSE T0 / TNp / TN / TXNp / TXN | {rm['T0']:.1f} / {rm['TNp']:.1f} / {rm['TN']:.1f} / {rm['TXNp']:.1f} / {rm['TXN']:.1f} |", ""]
    # ---- sensitivity
    L += ["## Sensitivity: the statements' quantities without the thirteen files (not used for a verdict)", "",
          "Scored rows outside the thirteen files whose NBI-2 power columns are on another scale; fits unchanged. Under V3 and "
          "V4 those files are never scored, and the value is the main one.", "",
          "| Id | value: point; d interval; c interval | rows / discharges / clusters | main value (all scored rows) |", "|---|---|---|---|"]
    for k, x in o["sensitivity_outside_13_files"].items():
        v = x["value"]
        mv = o["statements"][k]["value"]
        L.append(f"| {k} | {s3(v['point'])}; d {iv(v['discharge'])}; c {iv(v['cluster'])}" + ("" if x["scored_rows_of_the_13_files"] else " (same rows as main)") +
                 f" | {cnt(x['counts'])} | {s3(mv['point'])}; c {iv(mv['cluster'])} |")
    L.append("")
    # ---- counts
    L += ["## Counts", "",
          f"- Common set: F {si['common_F']['b79']:,}, L5 {si['common_L5']['b79']:,} rows; S set: F {si['S_F']['b79']:,}, "
          f"L5 {si['S_L5']['b79']:,} rows (equal to B.75's sets; no row lacks a beam scalar). D1 rows of the thirteen files: "
          f"{si['d1_rows_of_the_13_files']:,} (common L5 {si['common_L5_rows_of_the_13_files']:,}, S L5 {si['S_L5_rows_of_the_13_files']:,}).", "",
          "| design | common F | common L5 | S set F | S set L5 | fit rows per unit, TXN L5: D10 / D650 / S |", "|---|---|---|---|---|---|"]
    for d in DES:
        c = o["counts"][d]
        fr = c["fit_rows_per_unit_L5"]
        g = lambda k: f"{fr[k]['min']:,}-{fr[k]['max']:,}"   # noqa: E731
        L.append(f"| {DLAB[d]} | {cnt(c['common|F'])} | {cnt(c['common|L5'])} | {cnt(c['S|F'])} | {cnt(c['S|L5'])} | "
                 f"{g('TXN|D10|L5')} / {g('TXN|D650|L5')} / {g('TXN|S|L5')} |")
    L += ["", "Cells: rows / discharges / clusters scored.", "", f"Readout run time {o['runtime_readout_s']:.0f} s.", ""]
    L += ["## Limits (registration section 8, unchanged)", "",
          "- Exploratory on used data.",
          "- The beam columns are samples on the 10 ms rows; the notch itself has no row. The power in the first row after a "
          "notch is a sample of the returning beam and carries the same event as the BES sample of that row.",
          "- Only the powers and one voltage are used; currents and energy fractions are not. No beam-deposition or heating "
          "calculation is made.",
          "- The beam waveforms are what was delivered, not what was programmed; in real time the programmed waveform is known earlier.",
          "- One estimator per rule; the level target is built from reported values.", ""]
    return "\n".join(L)


def render_refit(o):
    """The refit without the thirteen files (Addendum 1 sections 11 to 13), appended after the main readout."""
    F = o["refit_without_13_files"]
    rf = F["readout"]
    L = ["", "## Sensitivity: refit without the thirteen files (the session lead's validity rule; not judged as the registered statements)", "",
         "Order of events (Addendum 1 sections 11 to 13). The session lead's instruction to leave the thirteen files out of every "
         "fit and score was sent before the fits and did not reach the implementing agent before the main readout above had been "
         "run and read. The main readout uses the beam columns as delivered, as the registration reads, and its statement table "
         "is the judged one. This refit was run afterwards, with every number of the main readout known.", "",
         f"Rule: {F['rule']}. Files selected: {', '.join(str(s) for s in F['invalid_files'])} (exactly the thirteen named "
         f"beforehand: {F['selects_exactly_the_expected_files']}). Every learner of the batch is fitted again with the rows of "
         "those files removed from fit and score; rules, targets, designs, learners, intervals (seed 20261179, cluster and "
         "discharge forms) as in the main readout. Checks: `docs/paper/b79_checks.md`, section 'Refit without the thirteen "
         "files'. Per-unit estimates: `data/.b79/refit_valid/fits/` (main run: `data/.b79/fits/`).", "",
         "### Statements N1 to N11: main beside the refit (target L5)", "",
         "| Id | statement | main: point; c interval; verdict | refit: point; d interval; c interval | refit: the statement's cluster-interval rule met |",
         "|---|---|---|---|---|"]
    for k, s in o["statements"].items():
        mv, rv = s["value"], rf["statements"][k]["value"]
        L.append(f"| {k} | {s['text']} | {s3(mv['point'])}; c {iv(mv['cluster'])}; **{'holds' if s['holds'] else 'does not hold'}** | "
                 f"{s3(rv['point'])}; d {iv(rv['discharge'])}; c {iv(rv['cluster'])} | {'yes' if rf['statements'][k]['holds'] else 'no'} |")
    inc = (("TN/T0", "TN over T0"), ("TXN/TN", "TXN over TN"), ("TX/T0", "TX over T0"), ("TXN/TX", "TXN over TX"))
    for X in ("L5", "F"):
        L += ["", f"### Skills of the refit by design and rule, target {X}", "",
              "Cells: refit skill, d interval, c interval; then the main point.", "",
              "| design | rule | rows / discharges / clusters | " + " | ".join(lab for _, lab in inc) + " |", "|---|---|---|" + "---|" * len(inc)]
        for d in DES:
            for r in ("D10", "D650", "S"):
                b = rf["main"][f"{d}|{X}|{KIND[r]}"]["all"][r]
                bm = blk(o, d, X, r)
                c = rf["main"][f"{d}|{X}|{KIND[r]}"]["all"]["counts"]
                L.append(f"| {DLAB[d]} | {RLAB[r]} | {cnt(c)} | " +
                         " | ".join(cellf(b[q]) + f"<br>main {s3(bm[q]['point'])}" for q, _ in inc) + " |")
    L += ["", "### Rows and discharges removed", "",
          "| set | main rows (discharges) | refit rows (discharges) | rows removed | discharges removed |", "|---|---|---|---|---|"]
    for k, c in F["removed_per_set"].items():
        L.append(f"| {k.replace('_', ', ')} | {c['main_rows']:,} ({c['main_discharges']}) | {c['refit_rows']:,} ({c['refit_discharges']}) | "
                 f"{c['rows_removed']:,} | {c['discharges_removed']} |")
    L += ["", "Per unit (target L5): fit and test discharges with a row in the set, and rows, removed from the main run's "
              "count (main count in brackets).", "",
          "| unit | common set: fit discharges / rows | common set: test discharges / rows | S set: fit discharges / rows | S set: test discharges / rows |",
          "|---|---|---|---|---|"]
    for un, x in F["removed_per_unit"].items():
        f = lambda k, s: (f"{x[k][s + '_discharges_removed']} / {x[k][s + '_rows_removed']:,} "   # noqa: E731
                          f"({x[k][s + '_discharges_main']} / {x[k][s + '_rows_main']:,})")
        L.append(f"| {un} | {f('common_L5', 'fit')} | {f('common_L5', 'test')} | {f('S_L5', 'fit')} | {f('S_L5', 'test')} |")
    L += ["", "The refit's full readout (RMSE, every skill, the ratio, the halves, the strata, the controls) is under the key "
              "`refit_without_13_files.readout` of the JSON.", ""]
    return "\n".join(L)


if __name__ == "__main__":
    # re-render the markdown from the readout JSON (no number is recomputed)
    import json
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import common79 as M
    M.save_text(M.OUT_MD, render(json.loads(M.OUT_JSON.read_text(encoding="utf-8"))))
    print("written", M.OUT_MD)
