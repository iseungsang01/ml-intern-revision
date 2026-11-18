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
PREREGISTRATION_MAST_COUPLING.md, fig_trace.json and docs/paper/file_rules.md (how the 10 ms files
relate to the raw deliveries; ces_prediction/analyze_file_rules.py), and the md files of
the later batches (B.39 to B.60, among them b53_components.md, b56_kfx.md, b57_yardsticks.md and
b60_relative_error.md; the floor of B.57 and the lower end of T10 of B.60 are taken from their
json, as make_tables.py rounds them).
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
    "529": "wavelength in nm of the C VI n = 8 to 7 line observed by the KSTAR CES (Ko et al. 2010)",
    "5.5": "robust white component over the level, lowest of the six ratios of b40_noise.md (0.055)",
    "9.5": "same, highest (0.095)",
    "1.9": "mean BES departure of group D1 minus D2, 4.39 - 2.48 (b32_two_sources.md)",
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
    "413": "train files of the temporal split manifest (data/.b5_camp_cut_split_s42: 413 / 128 / 97); 416 files in the campaign-ordered list",
    "0.96": "normalized radius printed for BES_0X06-0Y06 in Heo et al. 2025, Fig. 7 (discharge 31921)",
    "0.94": "normalized radius of the edge CES T_i in Heo et al. 2025, sec. 3 (T_i,edge at rho_N about 0.94)",
    "6.65": "interval 6.65-7.00 s of discharge 31921 in Heo et al. 2025, Figs 7 and 8",
    "7.00": "same interval",
    "0.65": "65 rows x 10 ms",
    "0.45": "minor radius quoted in Heo et al. 2025, sec. 2 (a about 0.45 m); (2.198 - 1.77) / 0.45 = 0.951",
    "529": "C VI (n = 8-7) line at 529.05 nm observed by the KSTAR CES (Ko et al. 2010; analyze_foundations.py)",
    # published device and diagnostic facts (Table S plasma, section 2): literature, not measured here
    "529.05": "C VI line wavelength, Lee2026NeuralNetworkbased / Jang2021DevelopmentUltrafast (lit_kstar_facts.md)",
    "2.25": "LCFS near R = 2.25 m in H-mode, Ko2015IonTemperature sec. 3",
    "2.19": "CES channels nearest the pedestal top at R ~ 219-220 cm, Kim2020PedestalElectron sec. IV",
    "2.20": "same",
    "2.192": "CES channel 16 at R = 2.192 m 'near the pedestal top', Lee2026NeuralNetworkbased Figs 5-6",
    "3.5": "KSTAR B_T up to 3.5 T, Nam2026OverviewKstar sec. 1",
    "0.6": "I_p 0.5-0.6 MA of published dataset discharges (Heo2025, Kim2024, Lee2026Spatiotemporal)",
    "1.4": "line-averaged n_e 1.0-1.4e19 m^-3 of 31921/31923, Heo2025 sec. 2",
    "2.18": "edge CES channels span R ~ 2.18-2.25 m, read from Fig. 1(c) of Lee2026SpatiotemporalStructure",
    "2.24": "midplane separatrix R ~ 2.24 m, read from Fig. 1(c) of Lee2026SpatiotemporalStructure",
    "2021": "year of the discharges in Kim2024KineticProfile (27973, 28019)",
    "3.84": "median CESNN rotation uncertainty 3.84 km/s, Kim2026AutomatedOutlier sec. 3",
    "0.080": "median CESNN T_i uncertainty 0.080 keV, Kim2026AutomatedOutlier sec. 3",
    "1.1": "beam-power drop of at least 1.1 MW at the blips of 31923, read from Fig. 2(b) of Heo2025",
    # channel geometry (S10, Table S7): lit_kstar_geometry.md
    "2.12": "BES view of 31923 spans R ~ 2.12-2.34 m, read from Fig. 1 of Heo2025",
    "2.34": "same",
    "2.16": "BES columns 5-7 at R ~ 2.16-2.21 m if the column index increases outward (from the 2.12-2.34 m box)",
    "2.21": "same",
    "2.6": "ECEI channel frequency f = LO + 2.6 + 0.9 (n_r - 1) GHz (fluctana kstarecei.py; syndia; user guide)",
    "0.9": "same formula, 0.9 GHz channel spacing",
    # blip facts and an earlier inter-ELM profile (lit_kstar_facts.md, third pass)
    "2014": "year of discharge 10350 in Ko2016CharacteristicsToroidal",
    "2.23": "T_i pedestal top at R ~ 2.22-2.23 m, read from Fig. 2 of Ko2016CharacteristicsToroidal",
    "2.26": "CES error bars grow beyond R ~ 2.26 m, read from Fig. 2 of Ko2016CharacteristicsToroidal",
    "2.7": "2.7 MW net beam power, 1.0 MW by NB1-B, Kim2018EnhancedFast sec. II",
    # facility user guide, CES channel table (slide 63, column 2022 campaign on) and Kwak2019
    "2192": "CES channel 17 at R = 2192 mm, KFE2026DiagnosticsDataUserGuide slide 63",
    "2198": "CES channel 18 at R = 2198 mm (the chord of the files), same table",
    "2203": "CES channel 19 at R = 2203 mm, same table",
    "2269": "outermost CES channel at R = 2269 mm, same table",
    "2017": "year of the discharge of Fig. 3(d) of Kwak2019MeasurementFast",
    # the beam-heating and optical-depth estimates are in docs/paper/b62_estimates.md (a source below)
    "10.5":"median beam-off duration of one-absent-frame blips in the raw streams (b61_blip_dose.md: 10.5 ms)",
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
    # main text, supplementary material, and the tables and macros generated from B.39
    gen = re.sub(r"(?m)^%.*$", "", (HERE / "b39_generated.tex").read_text(encoding="utf-8"))
    gen = gen.replace("{,}", ",").replace("$", " ").replace("\\,", " ")
    tex = "\n".join([strip_tex((HERE / "main_ppcf.tex").read_text(encoding="utf-8")),
                     strip_tex((HERE / "supp_ppcf.tex").read_text(encoding="utf-8")), gen])
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
        (ROOT / "docs/paper/b30_sensitivity.md").read_text(encoding="utf-8"),
        (ROOT / "docs/paper/file_rules.md").read_text(encoding="utf-8"),
        (ROOT / "docs/paper/mast_ped_confirmation.md").read_text(encoding="utf-8"),
        (ROOT / "ces_prediction/experiments/PREREGISTRATION_MAST_PED.md").read_text(encoding="utf-8"),
        (ROOT / "ces_prediction/experiments/PREREGISTRATION_B31.md").read_text(encoding="utf-8"),
        (ROOT / "docs/paper/fast_departure.md").read_text(encoding="utf-8"),
        (ROOT / "docs/paper/b32_two_sources.md").read_text(encoding="utf-8"),
        (ROOT / "docs/paper/b32_posthoc.md").read_text(encoding="utf-8"),
        (ROOT / "docs/paper/b32_posthoc_model.md").read_text(encoding="utf-8"),
        (ROOT / "ces_prediction/experiments/PREREGISTRATION_B32.md").read_text(encoding="utf-8"),
        (ROOT / "docs/paper/b34_error_model.md").read_text(encoding="utf-8"),
        (ROOT / "docs/paper/b36_layer.md").read_text(encoding="utf-8"),
        (ROOT / "docs/paper/b37_plan.md").read_text(encoding="utf-8"),
        (ROOT / "docs/paper/b37_addendum1.md").read_text(encoding="utf-8"),
        (ROOT / "docs/paper/b38_stage2.md").read_text(encoding="utf-8"),
        (ROOT / "docs/paper/b40_noise.md").read_text(encoding="utf-8"),
        (ROOT / "docs/paper/b41_events.md").read_text(encoding="utf-8"),
        (ROOT / "docs/paper/b30_thinned.md").read_text(encoding="utf-8"),
        (ROOT / "docs/paper/b35_blip_response.md").read_text(encoding="utf-8"),
        (ROOT / "docs/paper/b39_primary.md").read_text(encoding="utf-8"),
        (ROOT / "docs/paper/b39_inits.md").read_text(encoding="utf-8"),
        (ROOT / "docs/paper/b49_average.md").read_text(encoding="utf-8"),
        (ROOT / "docs/paper/b48_reread.md").read_text(encoding="utf-8"),
        (ROOT / "docs/paper/b50_missing_like.md").read_text(encoding="utf-8"),
        (ROOT / "docs/paper/b51_bes_modes.md").read_text(encoding="utf-8"),
        (ROOT / "docs/paper/b52_grading_readouts.md").read_text(encoding="utf-8"),
        (ROOT / "docs/paper/b54_pooled_rows.md").read_text(encoding="utf-8"),
        (ROOT / "docs/paper/b55_sample_timing.md").read_text(encoding="utf-8"),
        (ROOT / "docs/paper/b58_blip_calibration.md").read_text(encoding="utf-8"),
        (ROOT / "docs/paper/b53_components.md").read_text(encoding="utf-8"),
        (ROOT / "docs/paper/b56_kfx.md").read_text(encoding="utf-8"),
        (ROOT / "docs/paper/b57_yardsticks.md").read_text(encoding="utf-8"),
        (ROOT / "docs/paper/b60_relative_error.md").read_text(encoding="utf-8"),
        (ROOT / "docs/paper/b61_absolute_robust.md").read_text(encoding="utf-8"),
        (ROOT / "docs/paper/b61_blip_dose.md").read_text(encoding="utf-8"),
        (ROOT / "docs/paper/b62_beam_return.md").read_text(encoding="utf-8"),
        (ROOT / "docs/paper/b62_estimates.md").read_text(encoding="utf-8"),
        (ROOT / "docs/paper/b42_delay.md").read_text(encoding="utf-8"),
        (ROOT / "docs/paper/b43_target_checks.md").read_text(encoding="utf-8"),
        (ROOT / "docs/paper/b44_strata.md").read_text(encoding="utf-8"),
        (ROOT / "docs/paper/b45_ablation_rows.md").read_text(encoding="utf-8"),
        (ROOT / "docs/paper/b46_beam_off.md").read_text(encoding="utf-8"),
        (ROOT / "ces_prediction/experiments/PREREGISTRATION_B34.md").read_text(encoding="utf-8"),
        (ROOT / "ces_prediction/experiments/PREREGISTRATION_B37.md").read_text(encoding="utf-8"),
        thesis,
    ])
    kr = json.loads((HERE / "kstar_regime.json").read_text(encoding="utf-8"))
    extra = []
    # B.39: root-mean-square errors and row counts are printed in the paper from the json
    b39 = json.loads((ROOT / "docs/paper/b39_primary.json").read_text(encoding="utf-8"))
    for a in b39["A"].values():
        extra += ["%.1f" % v for v in a["rmse_eV"].values()] + ["%.1f" % v for v in a["median_abs_error_eV"].values()]
        extra += [f"{a['rows']:,}", "%.3f" % a["spread_causal_over_gp"]]
    if b39["P"].get("complete"):
        extra += ["%.1f" % v for v in b39["P"]["median_abs_error_eV"].values()]
        for k in ("P1_fold_level_nadeau_bengio", "P2_fold_level_nadeau_bengio"):
            extra += ["%+.3f" % v for v in b39["P"][k]]
        for v in b39["P"]["causal_over"].values():
            extra += ["%+.3f" % x for x in v]
    # B.57 and B.60: numbers that the paper prints rounded differently from the md (make_tables.py)
    b57 = json.loads((ROOT / "docs/paper/b57_yardsticks.json").read_text(encoding="utf-8"))
    extra += ["%.1f" % v for v in b57["all"]["rms"]]
    b60 = json.loads((ROOT / "docs/paper/b60_relative_error.json").read_text(encoding="utf-8"))
    extra.append("%+.4f" % b60["statements"]["T10"]["value"][1])
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
    # example traces of the main text (fig_trace_causal.py): the causal variant of B.39
    fc = json.loads((HERE / "fig_trace_causal.json").read_text(encoding="utf-8"))
    for k in ("s31921.csv", "s31923.csv", fc["typical_discharge"]["file"]):
        extra.append(str(fc[k]["scored_rows"]))
        extra += ["%.1f" % fc[k][e]["rmse_stored"] for e in ("model", "causal_gp", "persistence")]
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
