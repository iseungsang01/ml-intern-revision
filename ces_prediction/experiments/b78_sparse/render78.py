"""B.78: docs/paper/b78_sparse.md from the readout dictionary (readout78.py).  No number is computed here."""
from __future__ import annotations

DES = ("V1", "V2", "V3", "V4a", "V4b")
DLAB = {"V1": "V1 random (r2-r4)", "V2": "V2 cluster-blocked (3 assignments)", "V3": "V3 rolling origin (fifths 2-5 pooled)",
        "V4a": "V4 chronological: fit to 31991, score from 32312", "V4b": "V4 chronological: fit to 32310, score from 32312"}
RULES = ("K1", "K2", "K5", "K10", "K20", "K50")
RLAB = {"K1": "K1 (every frame)", "K2": "K2 (20 ms)", "K5": "K5 (50 ms)", "K10": "K10 (100 ms)", "K20": "K20 (200 ms)",
        "K50": "K50 (500 ms)"}
POS = ("third 1", "third 2", "third 3", "beyond one spacing")


def s3(x):
    return f"{x:+.3f}"


def iv(v):
    return f"[{v[0]:+.3f}, {v[1]:+.3f}]"


def cellf(v):
    return f"{s3(v['point'])}<br>d {iv(v['discharge'])}<br>c {iv(v['cluster'])}"


def ratf(v):
    return f"{v['point']:.3f}<br>d [{v['discharge'][0]:.3f}, {v['discharge'][1]:.3f}]<br>c [{v['cluster'][0]:.3f}, {v['cluster'][1]:.3f}]"


def kp(x):
    return f"{x:.2f}" if isinstance(x, float) else x


def render(o):
    si = o["sets"]
    fr = o["frames"]
    L = ["# B.78 — a CES value every k-th frame: the level in between, with and without the fast diagnostics", "",
         f"Registered: `{o['registration']}`; implementation choices in `{o['addendum']}`, written before any readout. "
         "CPU only; no network trained or run. T_i, cut population, the D1 rows of B.70 as cached by B.73 "
         f"({si['d1_rows']:,} rows, 635 discharges). Exploratory on used data (registration section 1): rows, targets, designs, "
         "learners and the fast and anchor inputs are those of B.73 and B.75; the rate rules are new.", "",
         "**The rate is lowered after the fact.** Under Kk only the measured values whose frame index is a multiple of k are "
         "visible; the withheld frames were taken with the beam on, and the targets are built from all measured values.", "",
         f"Scored rows: the D1 rows whose frame index is odd and not a multiple of 5 ({fr['scored_rows']:,} rows); they lie on no "
         "kept grid for k = 2, 5, 10, 20, 50, so the same rows are scored under every rule.", "",
         "Conventions. Skill of A over B = 1 - SSE(A) / SSE(B) on identical rows. Increment = TX over T0 under the same rule on "
         "the same rows. Each comparison: point, then 95 % intervals with discharges resampled (d) and with shot clusters "
         "resampled (c), 20,000 draws (V1: a re-partition drawn first; V2: an assignment drawn first; V3, V4: the scored units). "
         "Verdicts on the cluster form. Points under V1 and V2 = mean over the three re-partitions or assignments. Every fit and "
         "score uses the common set (scored rows complete under all six rules, target defined). RMSE in eV. Checks: "
         "`docs/paper/b78_checks.md`.", "",
         "Order of work. Before this readout, these results existed: the checks of registration section 10 and the counts of "
         "the row sets. No MSE, skill or statement under any rule had been computed.", ""]
    # ---- statements
    L += ["## Statements (target L5, cluster interval)", "",
          "| Id | statement | value: point; d interval; c interval | verdict | registered expectation | expectation met | note |",
          "|---|---|---|---|---|---|---|"]
    for k, s in o["statements"].items():
        v = s["value"]
        fmt = (lambda x: f"{x:.3f}") if k == "K-6" else s3
        ivf = (lambda a: f"[{a[0]:.3f}, {a[1]:.3f}]") if k == "K-6" else iv
        val = f"{fmt(v['point'])}; d {ivf(v['discharge'])}; c {ivf(v['cluster'])}"
        e = o["expectations"][k]
        met = {True: "yes", False: "no", None: "n/a (uncertain)"}[e["verdict_as_expected"]]
        if e["point_in_expected_range"] is not None:
            met += f"; point in the stated range: {'yes' if e['point_in_expected_range'] else 'no'}"
        L.append(f"| {k} | {s['text']} | {val} | **{'holds' if s['holds'] else 'does not hold'}** | {s['expected']} | {met} | {s.get('note', '')} |")
    L += ["", "K-5 is the skill of TX under K10 over T0 under K5; K-6 is a ratio of MSEs (judged on its upper end below 2); "
              "K-8 holds when the lower end of the placebo's interval is at most zero. 'V4' without a qualifier is V4a; V4b "
              "is in the note.", ""]
    # ---- RMSE by rule
    L += ["## RMSE by rate rule", "",
          "H newest visible value; M5 median of the last five visible values; G causal GP of the last up to 16 visible values; "
          "T0 G + boosted correction on the ten history scalars; TX with the eight fast scalars of row t and the six anchor "
          "scalars added; TB, TE the BES and the ECEI half (V1, V4; K1, K10); O the frame's own reading (against L5 only). "
          "Points; under V1 and V2 the mean over the three splits.", ""]
    for X in ("L5", "F"):
        for d in DES:
            G = o["main"][f"{d}|{X}"]["all"]
            cnt = G["counts"]
            cols = ["H", "M5", "G", "T0", "TX", "TB", "TE"] + (["O"] if X == "L5" else [])
            L += [f"### {DLAB[d]}, target {X} ({cnt['rows']:,} rows / {cnt['discharges']} discharges / {cnt['clusters']} clusters)", "",
                  "| rule | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
            for r in RULES:
                rm = G[r]["rmse_eV"]
                L.append(f"| {RLAB[r]} | " + " | ".join(f"{rm[c]:.1f}" if c in rm else "-" for c in cols) + " |")
            L.append("")
    # ---- increments
    for X in ("L5", "F"):
        L += [f"## The fast-channel increment TX over T0 by rule and design, target {X}", "",
              "Cells: skill, d interval, c interval.", "", "| design | " + " | ".join(RLAB[r] for r in RULES) + " |",
              "|---|" + "---|" * len(RULES)]
        for d in DES:
            G = o["main"][f"{d}|{X}"]["all"]
            L.append(f"| {DLAB[d]} | " + " | ".join(cellf(G[r]["TX/T0"]) for r in RULES) + " |")
        L.append("")
        L += [f"### TB and TE over T0 (V1, V4), target {X}", "",
              "| design | rule | TX over T0 | TB over T0 (BES half) | TE over T0 (ECEI half) |", "|---|---|---|---|---|"]
        for d in ("V1", "V4a", "V4b"):
            G = o["main"][f"{d}|{X}"]["all"]
            for r in ("K1", "K10"):
                L.append(f"| {DLAB[d]} | {RLAB[r]} | {cellf(G[r]['TX/T0'])} | {cellf(G[r]['TB/T0'])} | {cellf(G[r]['TE/T0'])} |")
        L.append("")
    # ---- T0 over G and H
    for X in ("L5", "F"):
        L += [f"## T0 over G and over H, target {X}", "", "| design | " + " | ".join(RLAB[r] for r in RULES) + " |",
              "|---|" + "---|" * len(RULES)]
        for d in DES:
            G = o["main"][f"{d}|{X}"]["all"]
            L.append(f"| {DLAB[d]}: T0 over G | " + " | ".join(cellf(G[r]["T0/G"]) for r in RULES) + " |")
            L.append(f"| {DLAB[d]}: T0 over H | " + " | ".join(cellf(G[r]["T0/H"]) for r in RULES) + " |")
        L.append("")
    # ---- cost and recovered share
    for X in ("L5", "F"):
        L += [f"## Cost of the lower rate and recovered share, target {X}", "",
              "Cost = MSE under the rule / MSE under K1 (same rows). Recovered share = (MSE(T0) - MSE(TX)) / (MSE(T0) - "
              "MSE(T0 under K1)) under the same rule.", "",
              "| design | rule | cost, T0 | cost, TX | recovered share | TX over T0 of the next denser rule |", "|---|---|---|---|---|---|"]
        for d in DES:
            G = o["main"][f"{d}|{X}"]["all"]
            for r in RULES[1:]:
                b = G[r]
                nd = b["TX over T0 of the next denser rule"]
                L.append(f"| {DLAB[d]} | {RLAB[r]} | {ratf(b['cost T0'])} | {ratf(b['cost TX'])} | {ratf(b['recovered share'])} | "
                         f"over T0 under {nd['denser_rule']}: {cellf(nd)} |")
        L.append("")
    # ---- equivalent rate
    for X in ("L5", "F"):
        L += [f"## Equivalent rate, target {X}", "",
              "k' = the rate at which T0 would have the MSE of TX under k (linear interpolation of T0's MSE in ln k between "
              "the rules); factor = k / k'. 'below 1': TX under k has an MSE at most that of T0 under K1. Intervals draw by "
              "draw; shares of censored draws in the cluster form.", "",
              "| design | k | k' point | factor point | k' d | k' c | factor c | cluster draws below 1 / above 50 | T0 curve monotone (point; share of cluster draws) |",
              "|---|---|---|---|---|---|---|---|---|"]
        for d in DES:
            G = o["main"][f"{d}|{X}"]["all"]
            for r in RULES:
                e = G[r]["equivalent rate"]
                p, dd, cc = e["point"], e["discharge"], e["cluster"]
                L.append(f"| {DLAB[d]} | {r[1:]} | {kp(p['k_prime'])} | {kp(p['factor_k_over_k_prime'])} | "
                         f"[{kp(dd['k_prime'][0])}, {kp(dd['k_prime'][1])}] | [{kp(cc['k_prime'][0])}, {kp(cc['k_prime'][1])}] | "
                         f"[{kp(cc['factor_k_over_k_prime'][0])}, {kp(cc['factor_k_over_k_prime'][1])}] | "
                         f"{cc['share_below_1']:.3f} / {cc['share_above_50']:.3f} | {p['T0_curve_monotone']}; {cc['share_T0_curve_monotone']:.3f} |")
        L.append("")
    # ---- against O
    L += ["## Against the frame's own reading O (target L5)", "",
          "Skill over O = 1 - MSE(estimator) / MSE(O); above zero means closer to the following level than the frame's own "
          "reported value.", "", "| design | estimator | " + " | ".join(RLAB[r] for r in RULES) + " |", "|---|---|" + "---|" * len(RULES)]
    for d in DES:
        G = o["main"][f"{d}|L5"]["all"]
        for lab in ("T0", "TX"):
            L.append(f"| {DLAB[d]} | {lab} over O | " + " | ".join(cellf(G[r][f"{lab}/O"]) for r in RULES) + " |")
    L.append("")
    # ---- controls
    G = o["main"]["V1|L5"]["all"]["K10"]
    rm = G["rmse_eV"]
    nf = o["anchor_row_control_nonfinite_rows"]
    L += ["## Controls (V1, K10, L5)", "",
          "TXp: TX refitted with the fourteen fast and anchor scalars permuted over rows (within the fit rows and within the "
          "test rows). TXo: TX refitted with the eight fast scalars of the anchor row (the row of the newest visible value) "
          "in place of row t, and a1 of BES and ECEI.", "",
          "| quantity | value |", "|---|---|",
          f"| RMSE T0 / TX / TXp / TXo | {rm['T0']:.1f} / {rm['TX']:.1f} / {rm['TXp']:.1f} / {rm['TXo']:.1f} |",
          f"| TXp over T0 | {cellf(G['TXp/T0'])} |",
          f"| TXo over T0 | {cellf(G['TXo/T0'])} |",
          f"| TX over TXo | {cellf(G['TX/TXo'])} |", "",
          f"Rows of TXo whose anchor-row fast scalars are not finite: at most {nf['fit_rows_per_unit_max']:,} fit rows and "
          f"{nf['test_rows_per_unit_max']:,} test rows per unit (left to the regression's handling of missing values).", ""]
    # ---- position strata
    for X in ("L5", "F"):
        L += [f"## By position between two kept values, target {X}", "",
              "uf = frame index of row t minus that of the row of the newest visible value. Thirds of the kept spacing k: "
              "3 uf < k; k <= 3 uf < 2k; 2k <= 3 uf < 3k; 'beyond one spacing': uf >= k (the newest kept frame before t carries "
              "no visible value). Empty strata are empty by the scored-row rule (Addendum 1 section 6).", "",
              "| design | rule | position | rows / discharges / clusters | RMSE T0 | RMSE TX | TX over T0 |", "|---|---|---|---|---|---|---|"]
        for d in ("V1", "V4a", "V4b"):
            B = o["main"][f"{d}|{X}"]
            for r in RULES[1:]:
                for p in POS:
                    key = f"{r}|{p}"
                    if key in B["positions"]:
                        s = B["positions"][key]
                        c = s["counts"]
                        L.append(f"| {DLAB[d]} | {RLAB[r]} | {p} | {c['rows']:,} / {c['discharges']} / {c['clusters']} | "
                                 f"{s['rmse_eV']['T0']:.1f} | {s['rmse_eV']['TX']:.1f} | {cellf(s['TX/T0'])} |")
                    else:
                        L.append(f"| {DLAB[d]} | {RLAB[r]} | {p} | 0 | - | - | - |")
        L.append("")
    # ---- ages
    A = o["age_on_common_set"]
    L += ["## Age of the newest visible value on the common set", "",
          "| target set | rule | rows | age 10 % (ms) | median | 90 % | third 1 | third 2 | third 3 | beyond one spacing |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for X, a in A.items():
        for r in RULES:
            x = a[r]
            qq, sh = x["age_ms_quantiles"], x.get("share_by_position")
            L.append(f"| {X} | {r} | {x['rows']:,} | {qq['0.1']:.0f} | {qq['0.5']:.0f} | {qq['0.9']:.0f} | " +
                     " | ".join(f"{sh[p]:.3f}" if sh else "-" for p in POS) + " |")
    L.append("")
    # ---- counts
    L += ["## Counts", "",
          f"- D1 rows {si['d1_rows']:,}; scored rows {si['scored_rows']:,}; scored rows with the eight fast scalars {si['scored_valid8_rows']:,}; "
          f"complete under all six rules {si['complete_under_all_six']:,}; common set F {si['common_F']:,}, L5 {si['common_L5']:,} "
          f"(scored rows with the target: F {si['scored_with_target_F']:,}, L5 {si['scored_with_target_L5']:,}).",
          "- Per rule (scored rows): " + "; ".join(
              f"{r}: visible {x['visible']:,}, fewer than three {x['fewer_than_three_visible']:,}, complete {x['complete']:,}, "
              f"dropped by a3 alone {x['dropped_by_a3_alone']:,}" for r, x in si["per_rule"].items()) + ".", "",
          "| design | common F | common L5 | fit rows per unit F | fit rows per unit L5 |", "|---|---|---|---|---|"]
    for d in DES:
        c = o["counts"][d]
        f = lambda k: f"{c[k]['rows']:,} / {c[k]['discharges']} / {c[k]['clusters']}"   # noqa: E731
        fr_ = c["fit_rows_per_unit"]
        L.append(f"| {DLAB[d]} | {f('F')} | {f('L5')} | {fr_['F']['min']:,}-{fr_['F']['max']:,} | {fr_['L5']['min']:,}-{fr_['L5']['max']:,} |")
    L += ["", "Cells: rows / discharges / clusters scored.", "", f"Readout run time {o['runtime_readout_s']:.0f} s.", ""]
    L += ["## Limits (registration section 9, unchanged)", "",
          "- Exploratory on used data.",
          "- The rate is lowered by us after the fact. The withheld frames were taken with the beam on; an operation that takes "
          "fewer CES frames would change the beam schedule and with it the plasma between the frames.",
          "- The kept values are the offline product.",
          "- One estimator per rule; the phase of the kept grid is one (frame index a multiple of k).",
          "- The following level is built from reported values.", ""]
    return "\n".join(L)


if __name__ == "__main__":
    # re-render the markdown from the readout JSON (no number is recomputed)
    import json
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import common78 as M
    M.save_text(M.OUT_MD, render(json.loads(M.OUT_JSON.read_text(encoding="utf-8"))))
    print("written", M.OUT_MD)
