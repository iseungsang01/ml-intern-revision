"""Check that every number printed in main_ppcf.tex occurs in a source of record.

A token is a decimal number, or an integer of three or more digits, as printed in the body
of main_ppcf.tex (signs kept when printed). It passes if the same token occurs in one of
the sources below, or in DERIVED with a stated derivation. This catches transcription
errors; it does not check that a number is attached to the right sentence (that was read
by hand against the sources).

Sources: docs/paper/short/main_short.tex (KSTAR numbers, traced there to the frozen
artifacts), docs/paper/mast_gate.md, docs/paper/mast_gate_addendum1.md, the review-loop and
MAST records at the end of THESIS_RESULTS.md, PREREGISTRATION_MAST_GATE.md,
docs/paper/ppcf/kstar_regime.json (rounded to three decimals), and the two figure
sources computed for this paper, docs/paper/ppcf/fig_missing.json and fig_gap.json.
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
    "7,957": "act shots after excluding the 13 pilot shots (select_shots.py output; experiments/README.md mast row)",
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
    fg = json.loads((HERE / "fig_gap.json").read_text(encoding="utf-8"))["cut"]
    for t in ("CES_TI", "CES_VT"):
        for name, c in fg[t].items():
            extra += [str(v) for v in c["rows_per_repeat"]]
            extra += ["%.1f" % v for v in c["rmse"].values()]
            for sk in c["skill"].values():
                extra += ["%+.3f" % v for v in [sk["median"], *sk["ci95"]]]
        a, b = fg[t]["all"]["rows_per_repeat"], fg[t]["10"]["rows_per_repeat"]
        extra += ["%.1f" % (100 * b[0] / a[0]), "%.1f" % (100 * b[1] / a[1])]
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
