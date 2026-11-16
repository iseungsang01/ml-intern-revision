"""Check that every number printed in main_ppcf.tex occurs in a source of record.

A token is a decimal number, or an integer of three or more digits, as printed in the body
of main_ppcf.tex (signs kept when printed). It passes if the same token occurs in one of
the sources below, or in DERIVED with a stated derivation. This catches transcription
errors; it does not check that a number is attached to the right sentence (that was read
by hand against the sources).

Sources: docs/paper/short/main_short.tex (KSTAR numbers, traced there to the frozen
artifacts), docs/paper/mast_gate.md, docs/paper/mast_gate_addendum1.md, the review-loop and
MAST records at the end of THESIS_RESULTS.md, PREREGISTRATION_MAST_GATE.md,
docs/paper/ppcf/kstar_regime.json (rounded to three decimals), kstar_level.json, the two
figure sources computed for this paper (fig_missing.json, fig_gap.json), docs/paper/mast_gate_addendum2.md,
docs/paper/b29_causal.md, PREREGISTRATION_B29.md, docs/paper/mast_edge.md (+ .json for\ncounts), docs/paper/b29_addendum1.md, PREREGISTRATION_B29_ADDENDUM1.md, docs/paper/mast_coupling.md,
PREREGISTRATION_MAST_COUPLING.md and fig_trace.json.
Run from the repo root:  py docs/paper/ppcf/check_numbers.py
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).parent

# token -> derivation (numbers that are computed from, not printed in, the sources)
DERIVED = {
    "0.29": "median MAST block 57 frames x 5 ms = 0.285 s",
    "0.3": "MAST beam block of about 0.3 s (57 frames x 5 ms)",
    "+0.26": "KSTAR running mean vs persistence, 1 - (169.18/197.22)^2 = 0.264 (b28_decomposition.json, min over 4 cut splits)",
    "+0.34": "same, max over 4 cut splits = 0.340",
    "+0.31": "KSTAR causal GP vs persistence, min over 4 cut splits = 0.306 (b28_decomposition.json RMSEs)",
    "+0.35": "same, max over 4 cut splits = 0.346",
    "10{,}461": "10,463 scored V_rot rows minus the 2 rows dropped by the 1,000 km/s rule (R2-c record)",
    "7{,}970": "FAIR-MAST level-1 shots listing the act source (open_data.md); 7,957 after the 13 pilot shots",
    "0.2": "MAST V_i test MSE 53.808 (causal GP) vs 53.925 (slope filter): 0.22 % apart (addendum record)",
    "1.5": "gate threshold 1.5 (registration) / last tuning step -1.48 % (addendum md)",
    "0.006": "offline GP vs two-point interpolation: 0.544 - 0.538 = 0.006 (T_i), 0.626 - 0.622 = 0.004 (V_i)",
    "2.3": "reported V_i error ratio 2.314 rounded",
    "0.455": "median of a chi-square variable with one degree of freedom (registration)",
    "2.2": "mean-to-median ratio of chi-square(1), 1 / 0.455 (registration)",
    "0.010": "10 ms",
    "14708": "first shot of campaign M6 in the FAIR-MAST catalogue (mast_hardware_timeline.md)",
    "30473": "last shot of campaign M9 in the FAIR-MAST catalogue (mast_hardware_timeline.md)",
    "2006": "campaign M6 starts 2006-01 (catalogue time stamps)",
    "2013": "campaign M9 ends 2013-09 (catalogue time stamps)",
    "0.80": "nominal normalized radius of the KSTAR chord, (2.198 - 1.8) / 0.5 = 0.796 (kstar_level.json)",
    "0.8": "same, rounded",
    "1.8": "nominal KSTAR major radius, m",
    "0.5": "nominal KSTAR minor radius, m / white-noise regime number -0.5 / loose fit rule 0.5",
    "0.7": "bin edge of the normalized radius (registration of the radial breakdown)",
    "0.9": "bin edge of the normalized radius",
    "1.0": "bin edge of the normalized radius",
    "+0.05": "registered expectation for the cut fast-channel increment, lower end (PREREGISTRATION_B29.md)",
    "+0.14": "same, upper end",
    "0.047": "smallest median relative T_i error of the three inner bins (addendum 2 md: 0.047)",
    "7,957": "act shots after excluding the 13 pilot shots (select_shots.py output; experiments/README.md mast row)",
    "0.93": "normalized radius range of the edge mode in Heo et al. 2025 (rho_N about 0.93 to 0.97, sec. 3 of that paper)",
    "0.97": "same",
    "0.95": "bin edge of the normalized radius (PREREGISTRATION_MAST_EDGE.md)",
    "0.6": "length of the example-trace window in seconds (fig_trace.py, WINDOW_S)",
    "0.82": "figure width fraction",
    "99.9": "share 0.9990 of the stored rows as a percentage (B.30 k = 1 check, THESIS_RESULTS)",
    "0.06": "fall of the network-vs-hold skill from k = 1 to k = 5, 0.060 to 0.072 (B.30 record), rounded",
    "0.07": "same, upper end rounded",
    "0.49": "network step, median 0.487 ms over five sessions (THESIS_RESULTS latency ladder, 5 sessions, CPU, batch 1, 1 thread)",
    "1.83": "same row, maximum 99th percentile 1.829 ms",
    "0.64": "causal GP per-row refit, median 0.638 ms (same table)",
    "2.28": "same row, maximum 99th percentile 2.277 ms",
    "0.007": "persistence, median 0.007 ms (same table)",
    "1.77": "major radius quoted in Heo et al. 2025, sec. 2 (R0 about 1.77 m)",
    "0.45": "minor radius quoted in Heo et al. 2025, sec. 2 (a about 0.45 m); (2.198 - 1.77) / 0.45 = 0.951",
}


def norm(s):
    s = s.replace("\u2212", "-").replace("{,}", ",").replace("\\,", "")
    return s


def tokens(s):
    return set(re.findall(r"(?<![\w.])[+-]?\d[\d,]*(?:\.\d+)?", s))


def strip_tex(s):
    s = s.split("\\begin{document}", 1)[1]
    s = re.sub(r"(?m)^%.*$", "", s)
    s = re.sub(r"\\(label|ref|cite|citep|includegraphics|url|bibliography|bibliographystyle)(\[[^\]]*\])?\{[^}]*\}", "", s)
    s = re.sub(r"\\(begin|end)\{[^}]*\}(\{[^}]*\})?", "", s)
    s = s.replace("{,}", ",").replace("--", " to ")
    return s.replace("$", " ").replace("\\,", " ")


def main():
    tex = strip_tex((HERE / "main_ppcf.tex").read_text(encoding="utf-8"))
    thesis = (ROOT / "THESIS_RESULTS.md").read_text(encoding="utf-8")
    thesis = thesis[thesis.index("### Review-loop round 1 analysis A3"):]
    src = "\n".join([
        (ROOT / "docs/paper/short/main_short.tex").read_text(encoding="utf-8").replace("$", " "),
        (ROOT / "docs/paper/mast_gate.md").read_text(encoding="utf-8"),
        (ROOT / "docs/paper/mast_gate_addendum1.md").read_text(encoding="utf-8"),
        (ROOT / "ces_prediction/experiments/PREREGISTRATION_MAST_GATE.md").read_text(encoding="utf-8"),
        (ROOT / "docs/paper/mast_gate_addendum2.md").read_text(encoding="utf-8"),
        (ROOT / "docs/paper/b29_causal.md").read_text(encoding="utf-8"),
        (ROOT / "ces_prediction/experiments/PREREGISTRATION_B29.md").read_text(encoding="utf-8"),
        (ROOT / "docs/paper/mast_edge.md").read_text(encoding="utf-8"),
        (ROOT / "docs/paper/b29_addendum1.md").read_text(encoding="utf-8"),
        (ROOT / "ces_prediction/experiments/PREREGISTRATION_B29_ADDENDUM1.md").read_text(encoding="utf-8"),
        (ROOT / "docs/paper/mast_coupling.md").read_text(encoding="utf-8"),
        (ROOT / "ces_prediction/experiments/PREREGISTRATION_MAST_COUPLING.md").read_text(encoding="utf-8"),
        (ROOT / "docs/paper/b30_posthoc.md").read_text(encoding="utf-8"),
        (ROOT / "docs/paper/b31_bursts.md").read_text(encoding="utf-8"),
        (ROOT / "ces_prediction/experiments/PREREGISTRATION_B31.md").read_text(encoding="utf-8"),
        thesis,
    ])
    kr = json.loads((HERE / "kstar_regime.json").read_text(encoding="utf-8"))
    extra = []
    for k in ("T_i", "V_rot"):
        r = kr[k]
        extra += ["%+.3f" % v for v in [r["median"], *r["ci95"], *r["quartiles"]]]
        extra.append(str(r["n_discharges"]))
    # figure sources computed for this paper (fig_missing.py, fig_gap.py)
    fm = json.loads((HERE / "fig_missing.json").read_text(encoding="utf-8"))
    extra += [f"{fm['n_rows']:,}"]
    for k in ("CES_TI", "CES_VT"):
        r = fm[k]
        extra += ["%.1f" % r["missing_pct"], "%.1f" % r["no_independent_pct"], "%.1f" % r["independent_pct"],
                  "%.1f" % (100 * r["gap_le_50ms_share"]), f"{r['n_gaps']:,}"]
    kl = json.loads((HERE / "kstar_level.json").read_text(encoding="utf-8"))
    extra += ["%.0f" % kl["median_of_medians_eV"], "%.0f" % kl["p10_eV"], f"{kl['p90_eV']:,.0f}",
              str(kl["n_discharges"]), "%.2f" % kl["nominal_rho"]]
    fgi = json.loads((HERE / "fig_gap.json").read_text(encoding="utf-8"))["incl"]
    for name, c in fgi["CES_TI"].items():
        for sk in c["skill"].values():
            extra += ["%+.3f" % v for v in [sk["median"], *sk["ci95"]]]
    fg = json.loads((HERE / "fig_gap.json").read_text(encoding="utf-8"))["cut"]
    for t in ("CES_TI", "CES_VT"):
        for name, c in fg[t].items():
            extra += [str(v) for v in c["rows_per_repeat"]]
            extra += ["%.1f" % v for v in c["rmse"].values()]
            for sk in c["skill"].values():
                extra += ["%+.3f" % v for v in [sk["median"], *sk["ci95"]]]
        a, b = fg[t]["all"]["rows_per_repeat"], fg[t]["10"]["rows_per_repeat"]
        extra += ["%.1f" % (100 * b[0] / a[0]), "%.1f" % (100 * b[1] / a[1])]
    # example traces (fig_trace.py): scored rows of the two discharges
    ft = json.loads((HERE / "fig_trace.json").read_text(encoding="utf-8"))
    typ = ft["typical_discharge"]["file"]
    extra += [str(ft[k]["scored_rows"]) for k in ("s31921.csv", "s31923.csv", typ)]
    extra += ["%.1f" % ft[typ][k]["rmse_stored"] for k in ("model", "causal_gp", "persistence")]
    # MAST confirmation set: counts printed with a thousands separator in the paper
    me = json.loads((ROOT / "docs/paper/mast_edge.json").read_text(encoding="utf-8"))
    cnt = me["counts"]["Not Checked"]
    extra += [f"{cnt[k]:,}" for k in ("with_file", "ok")] + [f"{v:,}" for v in me["counts"]["campaigns"].values()]
    extra.append(f"{me['primary']['pre_rule']['shots_with_W_block']:,}")
    for cell in ("T_i|W", "T_i|G"):
        for bn in me["primary"]["cells"][cell]["bins"].values():
            extra.append(f"{bn['regime_number']['n_shots']:,}")
    for bn in me["primary"]["pre_rule"]["bins"].values():
        extra.append(f"{bn['shots']:,}")
    src = norm(src) + "\n" + " ".join(extra)
    have = tokens(src)
    have_abs = {t.lstrip("+-") for t in have}
    derived = {norm(k).replace(",", ",") for k in DERIVED}

    bad = []
    for t in sorted(tokens(norm(tex))):
        body = t.lstrip("+-")
        if "." not in body and len(body.replace(",", "")) < 3:
            continue
        if t in have or t in derived or norm(t) in {norm(k) for k in DERIVED}:
            continue
        if t[0] not in "+-" and body in have_abs:
            continue
        # a signed token whose magnitude is printed unsigned in a source table cell
        if t[0] == "+" and body in have_abs:
            continue
        bad.append(t)
    n = len([t for t in tokens(norm(tex)) if "." in t or len(t.lstrip("+-").replace(",", "")) >= 3])
    print("tokens checked:", n, " not found in any source:", len(bad))
    for t in bad:
        print("  MISSING", t)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
