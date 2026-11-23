"""Check every number of the two records in use against a source of record.

    py ces_prediction/check_records.py

Records checked: PROJECT_KNOWLEDGE.md and PAPER_DESIGN.md.  A token is a decimal number or an integer
of three or more digits (sign kept).  It passes if the same token occurs in

  tier 1  the manuscript (docs/paper/ppcf/main_ppcf.tex and supp_ppcf.tex), the analysis outputs listed in SOURCES
          (file rules, the result files of B.70 onward, the critiques and gradings), the figure data of
          the manuscript, or DERIVED;
  tier 2  the experiment log THESIS_RESULTS.md.  Tokens that pass on tier 2 alone are listed: a match
          there is weaker evidence.

This catches transcription errors.  It does not check that a number is attached to the right
sentence; that is read by hand.
"""
from __future__ import annotations

import io
import json
import re
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
RECORDS = ("PROJECT_KNOWLEDGE.md", "PAPER_DESIGN.md")
SOURCES = (
    "docs/paper/ppcf/main_ppcf.tex",
    "docs/paper/ppcf/supp_ppcf.tex",
    "docs/paper/ppcf/refs.bib",
    "docs/paper/file_rules.md",
    "docs/paper/dataset_ledger.md",
    "docs/paper/b70_d1.md",
    "docs/paper/b70_checks.md",
    "docs/paper/b70_d1_phase2.md",
    "docs/paper/b71_scenarios.md",
    "docs/paper/b71_checks.md",
    "docs/paper/b72_methods.md",
    "docs/paper/b74_exposure.md",
    "docs/paper/b73_targets.md",
    "docs/paper/b73_checks.md",
    "docs/paper/b75_availability.md",
    "docs/paper/b75_checks.md",
    "docs/paper/b75_lead_checks.md",
    "docs/paper/b75_last_block.md",
    "docs/paper/b75_stuck_ecei.md",
    "docs/paper/b81_controls.md",
    "docs/paper/b81_lead_checks.md",
    "docs/paper/b82_localize.md",
    "docs/paper/b82_lead_checks.md",
    "docs/paper/review_loop/grade16_20261120.md",
    "docs/paper/review_loop/grade17_20261120.md",
    "docs/paper/review_loop/grade18_20261121.md",
    "docs/paper/b83_interp.md",
    "docs/paper/b83_lead_checks.md",
    "docs/paper/b84_onefit.md",
    "docs/paper/b84_lead_checks.md",
    "docs/paper/b86_mast_radius.md",
    "docs/paper/b86_checks.md",
    "docs/paper/b86_lead_checks.md",
    "docs/paper/b87_latency.md",
    "docs/paper/b87_lead_checks.md",
    "docs/paper/b97_lean.md",
    "docs/paper/b97_lean_checks.md",
    "docs/paper/b80_ecei_gate.md",
    "docs/paper/b80_lead_checks.md",
    "docs/paper/b76_mast_availability.md",
    "docs/paper/b76_checks.md",
    "docs/paper/b76_lead_checks.md",
    "docs/paper/b77_correct.md",
    "docs/paper/b77_checks.md",
    "docs/paper/b77_lead_checks.md",
    "docs/paper/b78_sparse.md",
    "docs/paper/b78_checks.md",
    "docs/paper/b78_lead_checks.md",
    "docs/paper/nbi_files.md",
    "docs/paper/b79_beam_inputs.md",
    "docs/paper/b79_checks.md",
    "docs/paper/b79_lead_checks.md",
    "docs/paper/prior_work_missing_ti.md",
    "docs/paper/review_loop/critique_20261120_novelty.md",
    "docs/paper/validation_methods.md",
    "docs/paper/review_loop/critique_20261120_referee.md",
    "docs/paper/review_loop/critique_20261120_pedestal.md",
    "docs/paper/review_loop/critique_20261120_stats.md",
    "docs/paper/review_loop/critique_20261120_cxrs.md",
)
DERIVED = {
    "2022": "the campaign year",
    "30801": "first shot of the data set",
    "32751": "last shot of the data set",
}


def norm(s):
    for a, b in (("−", "-"), (" ", " "), ("{,}", ","), ("\\,", " "), ("$", " "), ("–", " to "),
                 ("…", " to ")):
        s = s.replace(a, b)
    return s


def tokens(s):
    out = set()
    for t in re.findall(r"(?<![\w.])[+-]?\d[\d,]*(?:\.\d+)?", norm(s)):
        t = t.rstrip(",")
        body = t.lstrip("+-")
        if "." in body or len(body.replace(",", "")) >= 3:
            out.add(t)
    return out


def strip_paths(s):
    """Drop file paths, code spans and dates: they are not results."""
    s = re.sub(r"`[^`]*`", " ", s)
    s = re.sub(r"(?m)^#+ +\d+(\.\d+)*\.? ", "# ", s)      # heading numbers
    s = re.sub(r"\d+(\.\d+)?절", " ", s)                   # references to a section
    s = re.sub(r"\b20\d\d-\d\d-\d\d\b", " ", s)
    return s


def have(texts):
    h = tokens("\n".join(texts))
    return h, {t.lstrip("+-") for t in h}


def found(t, h, habs):
    body = t.lstrip("+-")
    return t in h or (t[0] not in "+-" and body in habs) or (t[0] == "+" and body in habs)


def main():
    t1 = [(ROOT / p).read_text(encoding="utf-8") for p in SOURCES]
    fm = json.loads((ROOT / "docs/paper/ppcf/fig_missing.json").read_text(encoding="utf-8"))
    extra = [f"{fm['n_rows']:,}"]
    for k in ("CES_TI", "CES_VT"):
        extra += ["%.1f" % fm[k][q] for q in ("missing_pct", "held_pct_of_grid", "independent_pct", "held_pct_of_observed")]
        extra += ["%.1f" % (100 * fm[k]["gap_le_50ms_share"]), f"{fm[k]['n_gaps']:,}"]
    kl = json.loads((ROOT / "docs/paper/ppcf/kstar_level.json").read_text(encoding="utf-8"))
    extra += ["%.0f" % kl["median_of_medians_eV"], "%.0f" % kl["p10_eV"], f"{kl['p90_eV']:,.0f}", str(kl["n_discharges"])]
    kr = json.loads((ROOT / "docs/paper/ppcf/kstar_regime.json").read_text(encoding="utf-8"))
    for k in ("T_i", "V_rot"):
        extra += ["%+.3f" % v for v in [kr[k]["median"], *kr[k]["ci95"]]] + [str(kr[k]["n_discharges"])]
    h1, a1 = have(t1 + [" ".join(extra)])
    h2, a2 = have([(ROOT / "THESIS_RESULTS.md").read_text(encoding="utf-8"),
                   (ROOT / "ces_prediction/experiments/README.md").read_text(encoding="utf-8")])
    bad = 0
    for rec in RECORDS:
        toks = sorted(tokens(strip_paths((ROOT / rec).read_text(encoding="utf-8"))))
        missing = [t for t in toks if not (found(t, h1, a1) or t in DERIVED or t.lstrip("+-") in DERIVED
                                           or found(t, h2, a2))]
        weak = [t for t in toks if not (found(t, h1, a1) or t in DERIVED or t.lstrip("+-") in DERIVED)
                and found(t, h2, a2)]
        print(f"{rec}: {len(toks)} tokens, {len(missing)} without a source, {len(weak)} found in the log only")
        for t in missing:
            print("  MISSING", t)
        if weak:
            print("  log only:", ", ".join(weak))
        bad += len(missing)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
