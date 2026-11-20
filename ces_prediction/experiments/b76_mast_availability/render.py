"""B.76: docs/paper/b76_mast_availability.md from the readout dictionary (readout.py).  No number is computed here."""
from __future__ import annotations

DES = ("M1", "M2", "M3", "M4")
DLAB = {"M1": "M1 random (3 re-partitions)", "M2": "M2 session-blocked (3 assignments)",
        "M3": "M3 rolling origin (fifths 2-5 pooled)", "M4": "M4 chronological (fit fifths 1-4, score fifth 5)"}
RULES = ("D5", "D15", "D25", "D50", "D100", "D150")


def s3(x):
    return f"{x:+.3f}"


def iv(v, signed=True):
    return f"[{v[0]:+.3f}, {v[1]:+.3f}]" if signed else f"[{v[0]:.3f}, {v[1]:.3f}]"


def cellf(v):
    return f"{s3(v['point'])}<br>s {iv(v['shot'])}<br>S {iv(v['session'])}"


def ratf(v):
    return f"{v['point']:.3f}<br>s {iv(v['shot'], False)}<br>S {iv(v['session'], False)}"


def cnt(c):
    return f"{c['rows']:,} / {c['shots']:,} / {c['sessions']}"


def render(o):
    si, sh = o["sets"], o["shots"]
    M = o["main"]
    L = ["# B.76 — MAST: the estimate under a stale charge-exchange history, with Thomson scattering of the present at the same radius", "",
         f"Registered: `{o['registration']}`; implementation choices in `{o['addendum']}`, written before any readout. CPU only; "
         "no network trained or run. Public FAIR-MAST files on disk; T_i of the charge-exchange chords at the pedestal top "
         "(1.5 to 5 cm inside the edge of the Thomson density profile). Exploratory on used data (registration section 1): "
         "both halves of the shot list were read in the earlier MAST batches with a one-frame history; the availability "
         "rules, the anchor inputs, the level target and the validation designs are new on MAST.", "",
         "Conventions. Skill of A over B = 1 - SSE(A) / SSE(B) on identical rows. Increment = TX over T0 under the same rule on "
         "the same rows. Each comparison: point, then 95 % intervals with shots resampled (s) and with sessions (calendar days) "
         f"resampled (S), {o['n_boot']:,} draws (M1: a re-partition drawn first; M2: an assignment drawn first; M3, M4: the "
         "scored units). Verdicts on the session form. Points under M1 and M2 = mean over the three re-partitions or "
         "assignments; RMSE there = mean of the per-split RMSE. Every rule is fitted and scored on the common set of its "
         "target (rows complete under all six rules), so that the rules differ only in the availability of the history. "
         "RMSE in eV. Checks: `docs/paper/b76_checks.md`.", "",
         "Order of work. Before this readout, these results existed: the four checks of registration section 12 and the "
         "counts of the shots and row sets. No MSE, skill or statement under any rule had been computed.", ""]
    # ---- statements
    L += ["## Statements (session interval)", "",
          "| Id | statement | value (point) | shot interval | session interval | verdict | registered expectation | expectation met | note |",
          "|---|---|---|---|---|---|---|---|---|"]
    for k, s in o["statements"].items():
        v = s["value"]
        signed = k != "S1"
        e = o["expectations"][k]
        met = {True: "yes", False: "no", None: "n/a (uncertain)"}[e["verdict_as_expected"]]
        if e["point_in_expected_range"] is not None:
            met += f"; point in the stated range: {'yes' if e['point_in_expected_range'] else 'no'}"
        pt = s3(v["point"]) if signed else f"{v['point']:.3f}"
        L.append(f"| {k} | {s['text']} | {pt} | {iv(v['shot'], signed)} | {iv(v['session'], signed)} | "
                 f"**{'holds' if s['holds'] else 'does not hold'}** | {s['expected']} | {met} | {s.get('note', '')} |")
    L += ["", "S1 is a ratio of MSEs (judged on its lower end above 1.5); S3 is a difference of skills; S9 holds when the lower "
              "end is at or below zero; S10 is the skill of TX over the frame's own reading O. Target L5 throughout.", ""]
    # ---- RMSE by rule
    L += ["## RMSE by availability rule", "",
          "H hold of the newest visible value; M5 median of the last five visible values; G causal GP of the last up to 16 "
          "visible values; T0 G + boosted correction on the history scalars; TX with the Thomson scalars added; RT = H x "
          "T_e(t) / T_e(anchor) (no fit); RTa the two-coefficient form; O the frame's own reading (against L5 only). Cells "
          "under 'rows / shots / sessions': scored rows of the common set.", ""]
    for X in ("L5", "F"):
        for d in DES:
            g = M[f"{d}|{X}|common"]["all"]
            cols = ["H", "M5", "G", "T0", "TX", "RT", "RTa"] + (["O"] if X == "L5" else [])
            L += [f"### {DLAB[d]}, target {X} ({cnt(g['counts'])} rows / shots / sessions)", "",
                  "| rule | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
            for r in RULES:
                rm = g[r]["rmse_eV"]
                L.append(f"| {r} | " + " | ".join(f"{rm[c]:.1f}" for c in cols) + " |")
            L.append("")
    # ---- increments
    for X in ("L5", "F"):
        L += [f"## The increment TX over T0 by rule and design, target {X}", "", "Cells: skill, s interval, S interval.", "",
              "| design | " + " | ".join(RULES) + " |", "|---|" + "---|" * len(RULES)]
        for d in DES:
            g = M[f"{d}|{X}|common"]["all"]
            L.append(f"| {DLAB[d]} | " + " | ".join(cellf(g[r]["TX/T0"]) for r in RULES) + " |")
        L += ["", f"Difference of increments, rule minus D5, target {X}:", "", "| design | " + " | ".join(RULES[1:]) + " |",
              "|---|" + "---|" * (len(RULES) - 1)]
        for d in DES:
            g = M[f"{d}|{X}|common"]["all"]
            L.append(f"| {DLAB[d]} | " + " | ".join(cellf(g[r]["TX/T0 minus under D5"]) for r in RULES[1:]) + " |")
        L.append("")
    # ---- T0 over G, H; ratio forms
    for X in ("L5", "F"):
        L += [f"## T0 over G and over H; RT and RTa over H and over T0, target {X}", "",
              "| design | comparison | " + " | ".join(RULES) + " |", "|---|---|" + "---|" * len(RULES)]
        for d in DES:
            g = M[f"{d}|{X}|common"]["all"]
            for k, lab in (("T0/G", "T0 over G"), ("T0/H", "T0 over H"), ("RT/H", "RT over H"), ("RT/T0", "RT over T0"),
                           ("RTa/H", "RTa over H"), ("RTa/T0", "RTa over T0")):
                L.append(f"| {DLAB[d]} | {lab} | " + " | ".join(cellf(g[r][k]) for r in RULES) + " |")
        L.append("")
    # ---- against O
    L += ["## Against the frame's own reading O (target L5)", "",
          "Skill over O = 1 - MSE(estimator) / MSE(O); above zero means closer to the following level than the frame's own "
          "reported value.", "", "| design | estimator | " + " | ".join(RULES) + " |", "|---|---|" + "---|" * len(RULES)]
    for d in DES:
        g = M[f"{d}|L5|common"]["all"]
        for lab in ("H", "G", "T0", "TX"):
            L.append(f"| {DLAB[d]} | {lab} over O | " + " | ".join(cellf(g[r][f"{lab}/O"]) for r in RULES) + " |")
    L.append("")
    # ---- staleness and recovered share
    for X in ("L5", "F"):
        L += [f"## Cost of staleness and recovered share, target {X}", "",
              "Cost of staleness = MSE under the rule / MSE under D5. Recovered share = (MSE(T0) - MSE(TX)) / (MSE(T0) - "
              "MSE(T0 under D5)) under the same rule and rows. Skill of TX under the rule over T0 under D5 beside them.", "",
              "| design | rule | cost of staleness, T0 | cost of staleness, TX | recovered share | TX over T0 under D5 |",
              "|---|---|---|---|---|---|"]
        for d in DES:
            g = M[f"{d}|{X}|common"]["all"]
            for r in RULES[1:]:
                b = g[r]
                L.append(f"| {DLAB[d]} | {r} | {ratf(b['staleness T0'])} | {ratf(b['staleness TX'])} | "
                         f"{ratf(b['recovered share'])} | {cellf(b['TX/T0(D5)'])} |")
        L.append("")
    # ---- by fifth
    for d, lab in (("M1", "M1, by fifth of the shot order of the test shot (registered readout)"),
                   ("M3", "M3, by scored fifth (added readout, Addendum 1 section 7)")):
        for X in ("L5", "F"):
            G = M[f"{d}|{X}|common"]
            L += [f"## {lab}, target {X}: TX over T0", "", "| fifth | rows / shots / sessions | " + " | ".join(RULES) + " |",
                  "|---|---|" + "---|" * len(RULES)]
            for st in [k for k in G if k.startswith("fifth")]:
                L.append(f"| {st[6:]} | {cnt(G[st]['counts'])} | " + " | ".join(cellf(G[st][r]["TX/T0"]) for r in RULES) + " |")
            L.append("")
    fifths = o["sets"]["designs"]["fifths"]
    L += ["Fifths of the shot order: " + "; ".join(f"{k + 1}: shots {f['shots'][0]}-{f['shots'][1]} ({f['n']} shots, "
                                                     f"{f['sessions']} sessions)" for k, f in enumerate(fifths)) + ".", ""]
    # ---- D-alpha
    for X in ("L5", "F"):
        L += [f"## D-alpha (DA set: common-set rows with D-alpha under D5 and D100), target {X}", "",
              "T0 and TX refitted on the DA rows (T0d, TXd). TD = T0's inputs and D-alpha; TA = with Thomson and D-alpha.", "",
              "| design | rows / shots / sessions | rule | RMSE H / G / T0d / TXd / TD / TA | TD over T0d | TA over T0d | TXd over T0d | TA over TXd |",
              "|---|---|---|---|---|---|---|---|"]
        for d in ("M1", "M4"):
            g = M[f"{d}|{X}|DA"]["all"]
            for r in ("D5", "D100"):
                rm = g[r]["rmse_eV"]
                L.append(f"| {DLAB[d]} | {cnt(g['counts'])} | {r} | " + " / ".join(f"{rm[c]:.1f}" for c in ("H", "G", "T0d", "TXd", "TD", "TA"))
                         + f" | {cellf(g[r]['TD/T0'])} | {cellf(g[r]['TA/T0'])} | {cellf(g[r]['TX/T0'])} | {cellf(g[r]['TA/TX'])} |")
        L.append("")
    # ---- controls
    c = M["M1|L5|common"]["all"]["controls"]
    rm = c["rmse_eV"]
    L += ["## Controls (M1, D100, target L5, common set)", "",
          "Placebo TXp: the nine Thomson scalars permuted over rows as one block, within the fit rows and within the test rows, "
          "TX refitted. TXo: the history scalars and T_e, n_e of the anchor frame only. T0 under D5 on the same rows.", "",
          f"RMSE (eV): T0 under D5 {rm['T0 D5']:.1f}; T0 under D100 {rm['T0 D100']:.1f}; TX under D100 {rm['TX D100']:.1f}; "
          f"TXp {rm['TXp D100']:.1f}; TXo {rm['TXo D100']:.1f}.", "",
          "| comparison | skill |", "|---|---|",
          f"| placebo TXp over T0 | {cellf(c['TXp/T0'])} |", f"| TXo (Thomson as old as the history) over T0 | {cellf(c['TXo/T0'])} |",
          f"| TX over TXo | {cellf(c['TX/TXo'])} |", f"| TX under D100 over T0 under D5 | {cellf(c['TX(D100)/T0(D5)'])} |",
          f"| T0 under D100 over T0 under D5 | {cellf(c['T0(D100)/T0(D5)'])} |", ""]
    # ---- alpha beta
    L += ["## Fitted alpha and beta of RTa", "",
          "ln(estimate) = ln H + alpha ln(T_e(t) / T_e(anchor)) + beta, least squares on the fit rows. Mean over the units "
          "[smallest, largest].", "", "| design | target | " + " | ".join(RULES) + " |", "|---|---|" + "---|" * len(RULES)]
    ab = o["rta_coefficients"]
    for d in DES:
        for X in ("L5", "F"):
            L.append(f"| {DLAB[d]} | {X} | " + " | ".join(
                f"alpha {ab[f'{d}|{X}|{r}']['alpha']['mean']:.3f} [{ab[f'{d}|{X}|{r}']['alpha']['min']:.3f}, {ab[f'{d}|{X}|{r}']['alpha']['max']:.3f}]"
                f"<br>beta {ab[f'{d}|{X}|{r}']['beta']['mean']:+.4f} [{ab[f'{d}|{X}|{r}']['beta']['min']:+.4f}, {ab[f'{d}|{X}|{r}']['beta']['max']:+.4f}]"
                for r in RULES) + " |")
    L.append("")
    # ---- counts
    st = sh["shot_status"]
    L += ["## Counts", "",
          f"- Shots: {sh['listed_shots']:,} listed; status " + ", ".join(f"{k} {v:,}" for k, v in st.items()) + f". Shots that "
          f"pass every filter: {sh['shots_ok']:,}.",
          f"- Rows (good chord frames) of those shots: {sh['rows_good']:,} ({sh['shots_good']:,} shots); in the pedestal-top band "
          f"at frame t: {sh['rows_good_in_band']:,} ({sh['shots_band']:,} shots); with Thomson at frame t: "
          f"{sh['rows_band_with_thomson_t']:,} ({sh['shots_band_ts']:,} shots); candidates (also the input edge distance): "
          f"{sh['rows_candidate']:,} rows ({si['candidate_shots']:,} shots).",
          "", "| rule | a value visible | history scalars and G finite | Thomson at the anchor | non-positive ratio | complete (rows / shots) | complete with D-alpha |",
          "|---|---|---|---|---|---|---|"]
    for r, x in si["per_rule"].items():
        L.append(f"| {r} | {x['visible']:,} | {x['history_and_G_complete']:,} | {x['thomson_at_anchor']:,} | {x['nonpositive_ratio']:,} | "
                 f"{x['complete']:,} / {x['complete_shots']:,} | {x['complete_with_dalpha']:,} |")
    L += ["", f"Complete under all six rules: {si['complete_all_rules']:,} rows, {si['complete_all_rules_shots']:,} shots. Common set "
              f"F {si['common_F']['rows']:,} rows / {si['common_F']['shots']:,} shots; L5 {si['common_L5']['rows']:,} / "
              f"{si['common_L5']['shots']:,}. DA set F {si['DA_F']['rows']:,} / {si['DA_F']['shots']:,}; L5 "
              f"{si['DA_L5']['rows']:,} / {si['DA_L5']['shots']:,}.", ""]
    dz = si["designs"]
    L += [f"Design shots {dz['shots']:,} in {dz['sessions']} sessions (shots per session: smallest {dz['session_shots']['min']}, "
          f"median {dz['session_shots']['median']:.0f}, largest {dz['session_shots']['max']}); M2 shots per fold by assignment: "
          + "; ".join(f"seed {k}: {v}" for k, v in dz["m2_fold_shots"].items()) + f"; sessions with shots on both sides of a "
          f"fifth boundary: {dz['sessions_straddling_a_fifth_boundary']}.", "",
          "| design | common F | common L5 | DA F | DA L5 | fit rows per unit, T0 L5 D5 / D100 |", "|---|---|---|---|---|---|"]
    fr = o["fit_rows"]
    for d in DES:
        g = lambda k: cnt(M[k]["all"]["counts"]) if k in M else "-"   # noqa: E731
        f5, f100 = fr[f"{d}|T0|D5|L5"], fr[f"{d}|T0|D100|L5"]
        L.append(f"| {DLAB[d]} | {g(f'{d}|F|common')} | {g(f'{d}|L5|common')} | {g(f'{d}|F|DA')} | {g(f'{d}|L5|DA')} | "
                 f"{f5['min']:,}-{f5['max']:,} / {f100['min']:,}-{f100['max']:,} |")
    L += ["", "Cells: rows / shots / sessions scored.", "", f"Readout run time {o['runtime_readout_s']:.0f} s.", ""]
    L += ["## Limits (registration section 11, unchanged)", "",
          "- Exploratory on used data (registration section 1).",
          "- MAST has no beam-modulation latency; the delays are imposed by us. The headline delay of 100 ms is a choice, fixed "
          "beforehand; MAST discharges are short, so the common set drops the first 150 ms and more of every chord's record.",
          "- Thomson scattering on MAST has one to two samples per frame and reported errors of its own; RT divides two of them.",
          "- The pedestal-top band is defined from the Thomson density profile, the same instrument that supplies the inputs of TX.",
          "- The following level is built from reported values; an error of the analysis that lasts 25 ms or more is in the target too.",
          "- One estimator per rule; no estimator fitted once for all ages.", ""]
    return "\n".join(L)


if __name__ == "__main__":
    import json
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import common76 as K
    K.save_text(K.OUT_MD, render(json.loads(K.OUT_JSON.read_text(encoding="utf-8"))))
    print("written", K.OUT_MD)
