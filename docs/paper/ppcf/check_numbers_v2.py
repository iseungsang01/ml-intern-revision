"""Check that every number printed in the rewritten manuscript occurs in a source of record.

A token is a decimal number, or an integer of three or more digits (thousands separators allowed), as
printed in the body of the manuscript (signs kept when printed; "$-$", "\\textminus" and U+2212 read as a
minus sign). A token passes if it occurs in
  1. gen_v2_tokens.json (every number make_tables_v2.py printed, each traced to a JSON key or md line),
  2. the "printed_tokens" of a figure json figv2_*.json (fig_v2.py),
  3. one of the source md files listed in SOURCES_MD (the batch result files and the lead's checks),
  4. DERIVED, with the derivation stated.
An unsigned token also passes when its magnitude is listed with a sign, and a "+" token when its magnitude
is listed unsigned. This catches transcription errors; it does not check that a number is attached to the
right sentence.

Usage (repo root):
  py docs/paper/ppcf/check_numbers_v2.py                 # main_ppcf.tex and supp_ppcf.tex
  py docs/paper/ppcf/check_numbers_v2.py --self          # self-test: gen_v2.tex against gen_v2_tokens.json only
  py docs/paper/ppcf/check_numbers_v2.py a.tex b.tex     # any files
Exit code 1 when a token is not found.
"""
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PAPER = ROOT / "docs" / "paper"

SOURCES_MD = [
    "b73_targets.md", "b73_checks.md", "b74_exposure.md", "b75_availability.md", "b75_checks.md",
    "b75_lead_checks.md", "b75_last_block.md", "b76_mast_availability.md", "b76_checks.md",
    "b76_lead_checks.md", "b77_correct.md", "b77_checks.md", "b77_lead_checks.md", "b78_sparse.md",
    "b78_checks.md", "b78_lead_checks.md", "b79_beam_inputs.md", "b79_checks.md", "b79_lead_checks.md",
    "b80_ecei_gate.md", "b80_lead_checks.md", "b75_stuck_ecei.md", "b81_controls.md", "b81_lead_checks.md", "b81_checks.md", "b82_localize.md", "b82_lead_checks.md", "b82_checks.md", "nbi_files.md", "b71_scenarios.md",
]

# token -> derivation (numbers computed from, not printed in, the sources). Add with a stated derivation.
DERIVED = {
    "0.65": "median age of the newest deliverable value, 0.650 s (b75 age_under_S.L5 quantile 0.5)",
    "650": "the same in ms (rule D650)",
    "2.198": "major radius of the CES chord in m (provider; PAPER_DESIGN.md sec. 2)",
    "0.94": "normalized radius of the chord in the literature discharges, lower end (PAPER_DESIGN.md sec. 2)",
    "0.96": "same, upper end",
    "1.5": "edge of the MAST pedestal-top band, 1.5 cm inside the density edge (PAPER_DESIGN.md sec. 2)",
    "0.010": "10 ms frame in s",
    "0.005": "5 ms MAST frame in s",
    "20261116": "random_state of HistGradientBoostingRegressor (PAPER_DESIGN.md sec. 6)",
    "20,000": "bootstrap draws (n_boot of every batch JSON)",
    "3,000": "cut of T_i at 3 keV, in eV (PAPER_DESIGN.md sec. 3)",
    "529": "wavelength in nm of the C VI line observed by the KSTAR CES",
    "2022": "campaign year of the discharges (shots 30801 to 32751)",
}


def norm(s):
    s = s.replace("\u2212", "-").replace("$-$", "-").replace("\\textminus", "-").replace("{,}", ",")
    return s.replace("\\,", "")


TOK = re.compile(r"(?<![\w.])[+-]?\d[\d,]*(?:\.\d+)?")


def tokens(s):
    out = set()
    for t in TOK.findall(s):
        t = t.rstrip(",")
        body = t.lstrip("+-")
        if "." not in body and len(body.replace(",", "")) < 3:
            continue
        out.add(t)
    return out


# lengths in TeX arguments or options ({2pt}, [width=0.9\linewidth], =3mm); a length typed in prose
# ("1.5 cm inside") is printed text and stays checked
UNIT = re.compile(r"(?:(?<=[=\[{,])\s*[+-]?\d*\.?\d+\s*(?:pt|cm|mm|em|ex|in|bp)\b)"
                  r"|(?:[+-]?\d*\.?\d+\s*\\(?:textwidth|linewidth|columnwidth|textheight))")


def strip_line(s):
    """Remove from one line of TeX what is not printed text: comments, labels, references, file names,
    lengths, table geometry. Macro names are letters only in TeX, so a name never yields a token; a number
    typed in the text (outside a macro) stays and is checked."""
    s = re.sub(r"(?<!\\)%.*$", "", s)
    s = re.sub(r"\\(label|ref|eqref|autoref|cref|Cref|cite|citep|citet|citealp|includegraphics|url|href|input|include|"
               r"bibliography|bibliographystyle|newcommand|renewcommand|providecommand)\*?(\[[^\]]*\])?\{[^}]*\}", " ", s)
    s = re.sub(r"\\(begin|end)\{[^}]*\}(\[[^\]]*\])?(\{[^}]*\})?", " ", s)
    s = re.sub(r"\\(setlength|addtolength)\{\\[A-Za-z]+\}\{[^}]*\}", " ", s)
    s = re.sub(r"\\(hspace|vspace|rule|raisebox|scalebox|resizebox)\*?\{[^}]*\}", " ", s)
    s = re.sub(r"\\parbox\[[^\]]*\]\{[^}]*\}", " ", s)
    s = re.sub(r"\\cmidrule\([^)]*\)\{[^}]*\}", " ", s)
    s = re.sub(r"\\multicolumn\{\d+\}\{[^}]*\}", " ", s)
    s = re.sub(r"\\\\\[[^\]]*\]", " ", s)
    s = UNIT.sub(" ", s)
    s = s.replace("--", " to ")
    return norm(s).replace("$", " ")


def strip_tex(s, body_only=True):
    """Whole-text form (self-test)."""
    return "\n".join(strip_line(l) for l in body_lines(s, body_only))


def body_lines(s, body_only=True):
    lines = s.splitlines()
    if body_only:
        for k, l in enumerate(lines):
            if l.lstrip().startswith("\\begin{document}"):
                return [""] * (k + 1) + lines[k + 1:]
    return lines


def load_sources():
    reg = json.loads((HERE / "gen_v2_tokens.json").read_text(encoding="utf-8"))["tokens"]
    have = set(reg)
    figs = sorted(HERE.glob("figv2_*.json"))
    for f in figs:
        have |= set(json.loads(f.read_text(encoding="utf-8")).get("printed_tokens", []))
    md = []
    for name in SOURCES_MD:
        p = PAPER / name
        if p.exists():
            md.append(norm(p.read_text(encoding="utf-8")))
    have_md = tokens("\n".join(md))
    return set(reg), have, have_md, figs


def passes(t, pool):
    body = t.lstrip("+-")
    if t in pool:
        return True
    pool_abs = {x.lstrip("+-") for x in pool}
    if t[0] not in "+-" and body in pool_abs:
        return True
    if t[0] == "+" and body in pool:
        return True
    return False


# sources accepted only for the file named in the key (its numbers were traced under the earlier checker)
EXTRA_SOURCES = {
    "supp_data_fragment.tex": ["docs/paper/ppcf/earlier_line/supp_ppcf_ea2e1a6.tex", "docs/paper/file_rules.md",
                               "docs/paper/nbi_files.md"],
}


def check_file(f, pool, body_only=True):
    pool = set(pool)
    for src in EXTRA_SOURCES.get(f.name, []):
        pool |= tokens(norm((ROOT / src).read_text(encoding="utf-8")).replace("$", " "))
    seen, bad = {}, []
    for n, l in enumerate(body_lines(f.read_text(encoding="utf-8"), body_only), 1):
        for t in tokens(strip_line(l)):
            seen.setdefault(t, n)
            if not passes(t, pool):
                bad.append((n, t))
    return seen, bad


def main(argv):
    reg, have, have_md, figs = load_sources()
    derived = {norm(k) for k in DERIVED}
    if argv and argv[0] == "--self":
        print("self-test: every token of gen_v2.tex must be in gen_v2_tokens.json (no md fallback)")
        seen, bad = check_file(HERE / "gen_v2.tex", reg, body_only=False)
        files = [HERE / "gen_v2.tex"]
        results = [(files[0], seen, bad)]
    else:
        files = [Path(a) if Path(a).is_absolute() else ROOT / a for a in argv] or [
            HERE / "main_ppcf.tex", HERE / "supp_ppcf.tex", HERE / "supp_data_fragment.tex"]
        files = [f for f in files if f.exists()]
        pool = have | have_md | derived
        results = []
        for f in files:
            seen, bad = check_file(f, pool, body_only=(f.name != "supp_data_fragment.tex"))
            results.append((f, seen, bad))
    print("figure jsons:", len(figs), " registered tokens:", len(reg))
    nbad = 0
    for f, seen, bad in results:
        extra = " (extra sources: " + ", ".join(EXTRA_SOURCES[f.name]) + ")" if f.name in EXTRA_SOURCES else ""
        print(f"{f.name}: tokens checked {len(seen)}, not found {len(bad)}{extra}")
        for n, t in bad:
            print(f"  MISSING {t}  (line {n})")
        nbad += len(bad)
    return 1 if nbad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
