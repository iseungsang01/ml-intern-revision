"""Check every number of the two records in use against a source of record.

    py ces_prediction/check_records.py

Records checked: PROJECT_KNOWLEDGE.md and PAPER_DESIGN.md.  A token is a decimal number or an integer
of three or more digits (sign kept).  It passes if the same token occurs in

  tier 1  the manuscript (docs/paper/ppcf/main_ppcf.tex and supp_ppcf.tex), the analysis outputs listed in SOURCES
          (file rules, B.29 / B.30 / B.31, MAST), the figure data of the manuscript, or DERIVED;
  tier 2  the experiment log THESIS_RESULTS.md (design-choice numbers of earlier batches live only
          there).  Tokens that pass on tier 2 alone are listed, because the log holds almost every
          number that was ever produced and a match there is weaker evidence.

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
    "docs/paper/ppcf/main_ppcf.tex", "docs/paper/ppcf/supp_ppcf.tex", "docs/paper/ppcf/refs.bib", "docs/paper/file_rules.md",
    "docs/paper/b29_causal.md", "docs/paper/b29_addendum1.md", "docs/paper/b30_thinned.md",
    "docs/paper/b30_posthoc.md", "docs/paper/b30_sensitivity.md", "docs/paper/b30_stack.md",
    "docs/paper/b30_stack_offline.md", "docs/paper/b31_bursts.md", "docs/paper/mast_ped_confirmation.md",
    "docs/paper/mast_edge.md", "docs/paper/mast_gate.md", "docs/paper/mast_gate_addendum1.md",
    "docs/paper/fast_departure.md", "docs/paper/b32_two_sources.md", "docs/paper/b32_posthoc.md",
    "docs/paper/b32_posthoc_model.md",
    "docs/paper/b33_clusters.md",
    "docs/paper/b34_error_model.md", "docs/paper/b35_blip_response.md",
    "docs/paper/b36_layer.md", "docs/paper/b37_plan.md",
    "docs/paper/b37_addendum1.md",
    "docs/paper/b38_stage2.md",
    "docs/paper/b41_events.md",
    "docs/paper/b40_noise.md",
    "docs/paper/b44_strata.md",
    "docs/paper/b43_target_checks.md",
    "docs/paper/b42_delay.md",
    "docs/paper/b39_primary.md",
    "docs/paper/b39_inits.md",
    "docs/paper/b49_average.md",
    "docs/paper/b48_reread.md",
    "docs/paper/b50_missing_like.md",
    "docs/paper/b51_bes_modes.md",
    "docs/paper/b45_ablation_rows.md",
    "docs/paper/b46_beam_off.md",
)
DERIVED = {
    "2022": "the campaign year",
    "0.218": "seq_v2 lean step, median ms (latency ladder of the log, section 8aj)",
    "1.37": "offline GP median 2.357 ms / causal GP 1.723 ms in the same sessions (bench_offline.py record)",
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
