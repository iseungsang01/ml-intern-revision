"""Tables and number macros of the rewritten manuscript (PAPER_DESIGN.md, 2026-11-20 line).

Writes, next to this file:
  gen_v2.tex          \\newcommand{\\TabXxx}{...} table bodies, \\newcommand{\\CapXxx}{...} caption stubs
                      (target, availability rule, design, interval), and one-number macros for the prose.
  gen_v2_tokens.json  every numeric token printed in gen_v2.tex, with the file and key path it came from
                      and the transformation (e.g. "100 x skill, one decimal").

Nothing is typed by hand: every number comes from a result JSON under docs/paper/, or, where no JSON
exists, is parsed from the md file named in its provenance (b75_last_block.md, b75_lead_checks.md,
nbi_files.md). No analysis is done here; numbers are formatted only.

Conventions (PAPER_DESIGN.md sections 7 and 8):
  * comparisons are printed as the reduction of the MSE in per cent = 100 x skill, one decimal, with the
    95 % interval from the shot-cluster bootstrap (MAST: session bootstrap); the point is set in bold
    when that interval excludes zero;
  * RMSE in eV, one decimal; ratios with three decimals as in the sources;
  * statement tables print the registered quantity as registered (signed, three decimals).

Run from the repo root:  py docs/paper/ppcf/make_tables_v2.py
"""
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PAPER = ROOT / "docs" / "paper"

_cache = {}


def J(name):
    if name not in _cache:
        _cache[name] = json.loads((PAPER / name).read_text(encoding="utf-8"))
    return _cache[name]


def src_of(name):
    return "docs/paper/" + name


# ----------------------------------------------------------------------------------------------
# token registry
# ----------------------------------------------------------------------------------------------
TOKENS = {}


def reg(tok, src, path, tr):
    TOKENS.setdefault(tok, [])
    ent = {"source": src, "path": path, "transform": tr}
    if ent not in TOKENS[tok] and len(TOKENS[tok]) < 12:
        TOKENS[tok].append(ent)


TOK_RE = re.compile(r"(?<![\w.])[+-]?\d[\d,]*(?:\.\d+)?")


def tex_sign(s):
    return "$-$" + s[1:] if s.startswith("-") else s


def fnum(v, fmt, src, path, tr, scale=1.0):
    """Format v*scale with fmt, register the printed token, return the TeX string."""
    s = fmt % (v * scale)
    if s.lstrip("+-").strip("0.") == "" and s[0] in "+-":     # -0.0 / +0.000 -> 0.0 / 0.000
        s = s[1:] if s[0] == "-" else s
    reg(s, src, path, tr)
    return tex_sign(s)


def cnt(n, src, path, tr="count"):
    s = f"{int(n):,}"
    reg(s, src, path, tr)
    return s.replace(",", "{,}")


def iv_key(cell):
    for k in ("cluster", "session"):
        if k in cell:
            return k
    for k in ("discharge", "shot"):
        if k in cell:
            return k
    raise KeyError(list(cell))


def excl0(lo, hi):
    return (lo > 0 and hi > 0) or (lo < 0 and hi < 0)


def pct_parts(cell, src, path, bold=True):
    """(point, '[lo, hi]') of 100 x skill with the cluster (session) interval."""
    k = iv_key(cell)
    lo, hi = cell[k]
    tr = f"100 x skill, one decimal ({k} interval)"
    pt = fnum(cell["point"], "%.1f", src, path + ".point", "100 x skill, one decimal", 100)
    a = fnum(lo, "%.1f", src, f"{path}.{k}[0]", tr, 100)
    b = fnum(hi, "%.1f", src, f"{path}.{k}[1]", tr, 100)
    if bold and excl0(lo, hi):
        pt = r"\textbf{" + pt + "}"
    return pt, f"[{a}, {b}]"


def rat_parts(cell, src, path, nd=3):
    k = iv_key(cell)
    lo, hi = cell[k]
    f = "%." + str(nd) + "f"
    tr = f"ratio, {nd} decimals"
    pt = fnum(cell["point"], f, src, path + ".point", tr)
    a = fnum(lo, f, src, f"{path}.{k}[0]", tr + f" ({k} interval)")
    b = fnum(hi, f, src, f"{path}.{k}[1]", tr + f" ({k} interval)")
    return pt, f"[{a}, {b}]"


def inline(parts):
    return parts[0] + "~" + parts[1]


def stack(parts):
    return r"\ci{" + parts[0] + "}{" + parts[1] + "}"


def pct(cell, src, path, mode="inline", bold=True):
    if cell is None:
        return "--"
    p = pct_parts(cell, src, path, bold)
    return inline(p) if mode == "inline" else stack(p)


def rat(cell, src, path, mode="inline", nd=3):
    if cell is None:
        return "--"
    p = rat_parts(cell, src, path, nd)
    return inline(p) if mode == "inline" else stack(p)


def ev(v, src, path):
    if v is None:
        return "--"
    return fnum(v, "%.1f", src, path, "RMSE in eV, one decimal")


def g(d, *keys):
    for k in keys:
        if d is None or k not in d:
            return None
        d = d[k]
    return d


SPECIALS = [("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("$", r"\$"), ("#", r"\#"),
            ("_", r"\_"), ("{", r"\{"), ("}", r"\}"), ("~", r"\textasciitilde{}"), ("^", r"\textasciicircum{}")]
UNI = {"−": "-", "–": "--", "—": "---", "≤": r"$\le$", "≥": r"$\ge$", "±": r"$\pm$",
       "×": r"$\times$", "≈": r"$\approx$", "μ": r"$\mu$", "µ": r"$\mu$", "²": r"$^2$",
       "→": r"$\to$", "‘": "`", "’": "'", "“": "``", "”": "''", "α": r"$\alpha$",
       "β": r"$\beta$", "ρ": r"$\rho$", "é": r"\'e", "′": "'"}


def esc(s, src, path):
    """Escape free text for LaTeX and register every numeric token in it (verbatim)."""
    s = str(s)
    for t in TOK_RE.findall(s.replace("−", "-")):
        reg(t, src, path, "verbatim from text")
    out = []
    for ch in s:
        if ch in UNI:
            out.append(UNI[ch])
            continue
        done = False
        for a, b in SPECIALS:
            if ch == a:
                out.append(b)
                done = True
                break
        if done:
            continue
        if ch == "<":
            out.append("$<$")
        elif ch == ">":
            out.append("$>$")
        elif ord(ch) > 127:
            out.append("?")
        else:
            out.append(ch)
    return "".join(out)


OUT = []          # lines of gen_v2.tex
README = []       # (kind, name, what, source, map) for gen_v2_README.md


def emit(name, body, cap, what, source, where, kind="table"):
    OUT.append(f"% ---- {name}: {what}")
    OUT.append("\\newcommand{\\Tab" + name + "}{" + body + "}")
    OUT.append("\\newcommand{\\Cap" + name + "}{" + cap + "}")
    OUT.append("")
    README.append((kind, "\\Tab" + name, what, source, where))


MACROS = []


def macro(name, val, what, source, where=""):
    assert re.fullmatch(r"[A-Za-z]+", name), name
    MACROS.append("\\newcommand{\\" + name + "}{" + val + "}")
    README.append(("macro", "\\" + name, what, source, where))


def tabular(colspec, head, rows, size="footnotesize", sep="4pt"):
    lines = [r"{\%s\setlength{\tabcolsep}{%s}\begin{tabular}{%s}\toprule" % (size, sep, colspec)]
    lines += head
    lines.append(r"\midrule")
    for r in rows:
        if r == "MID":
            lines.append(r"\midrule")
        elif isinstance(r, str):
            lines.append(r)
        else:
            lines.append(" & ".join(r) + r" \\")
    lines.append(r"\bottomrule\end{tabular}}")
    return "\n".join(lines)


# ----------------------------------------------------------------------------------------------
# names
# ----------------------------------------------------------------------------------------------
B75, B76, B78, B79, B80 = ("b75_availability.json", "b76_mast_availability.json", "b78_sparse.json",
                           "b79_beam_inputs.json", "b80_ecei_gate.json")
B73, B74, B77 = "b73_targets.json", "b74_exposure.json", "b77_correct.json"
B81 = "b81_controls.json"


def shot_tok(s, src, path):
    reg(str(s), src, path, "shot number from a block or design key")
    return str(s)


def design_label(d, src=B75):
    p = "main.V1|L5|common keys"
    if d == "V1":
        return "V1 random"
    if d == "V2":
        return "V2 shot clusters held out"
    if d == "V3":
        return "V3 rolling origin"
    if d == "V4a":
        return "V4 chronological, fit to " + shot_tok(31991, src_of(B75), "main.V1|L5|S key 'block up to 31991'")
    if d == "V4b":
        return "V4 chronological, fit to " + shot_tok(32310, src_of(B75), "main.V1|L5|S key 'block 32002-32310'")
    return d


DES = ["V1", "V2", "V3", "V4a", "V4b"]
MDES = ["M1", "M2", "M3", "M4"]
MLAB = {"M1": "M1 random", "M2": "M2 sessions held out", "M3": "M3 rolling origin", "M4": "M4 chronological"}
RULES = ["D10", "D50", "D100", "D300", "D650", "D1000"]
MRULES = ["D5", "D15", "D25", "D50", "D100", "D150"]
KRULES = ["K1", "K2", "K5", "K10", "K20", "K50"]

PRE = [
    "% generated by docs/paper/ppcf/make_tables_v2.py -- do not edit by hand",
    "% every number below is listed with its source in gen_v2_tokens.json",
    r"\providecommand{\ci}[2]{\begin{tabular}[t]{@{}c@{}}#1\\[-1.5pt]{\scriptsize #2}\end{tabular}}",
    r"\providecommand{\tcell}[1]{\begin{tabular}[t]{@{}l@{}}#1\end{tabular}}",
    "",
]


def bldate(blk):
    """'block up to 31991' -> 'up to 31991' with the shot tokens registered."""
    for s in re.findall(r"\d{5}", blk):
        shot_tok(s, src_of(B75), f"block key '{blk}'")
    return blk.replace("block ", "").replace("-", "--")


# ----------------------------------------------------------------------------------------------
# 1. rows and sets
# ----------------------------------------------------------------------------------------------
def tab_sets():
    b75, b73, b76, b78, b77 = J(B75), J(B73), J(B76), J(B78), J(B77)
    s = src_of(B75)
    rows = []

    def r3(label, use, c, src, path, unit3="clusters"):
        return [label, use, cnt(c["rows"], src, path + ".rows"),
                cnt(c.get("discharges", c.get("shots")), src, path + (".discharges" if "discharges" in c else ".shots")),
                cnt(c.get("clusters", c.get("sessions")), src, path + (".clusters" if "clusters" in c else ".sessions"))]

    rows.append([r"D1 rows (cut population)", "definition of the data",
                 cnt(b75["sets"]["d1_rows"], s, "sets.d1_rows"),
                 cnt(b75["designs"]["d1_discharges"], s, "designs.d1_discharges"),
                 cnt(b75["designs"]["clusters_d1"], s, "designs.clusters_d1")])
    rows.append(r3("rows with L5, D10 visible", "previous frame visible (B.73)",
                   b73["increments"]["V1|L5"], src_of(B73), "increments.V1|L5"))
    rows.append(r3("common set, L5", "rules D10 to D1000 (B.75, B.79)",
                   b75["main"]["V1|L5|common"]["all"]["counts"], s, "main.V1|L5|common.all.counts"))
    rows.append(r3("S set, L5", "beam-modulation schedule (B.75, B.79, B.80)",
                   b75["main"]["V1|L5|S"]["all"]["counts"], s, "main.V1|L5|S.all.counts"))
    rows.append(r3("K set, L5", "a value every k-th frame (B.78)",
                   b78["counts"]["V1"]["L5"], src_of(B78), "counts.V1.L5"))
    rows.append(r3("rows with a reading, L5", "correcting the frame's reading (B.77)",
                   b77["row_set"], src_of(B77), "row_set"))
    rows.append("MID")
    rows.append(r3("MAST common set, L5", "rules D5 to D150 (B.76)",
                   b76["main"]["M1|L5|common"]["all"]["counts"], src_of(B76), "main.M1|L5|common.all.counts"))
    rows.append(r3("MAST D-alpha set, L5", "D-alpha rows (B.76)",
                   b76["main"]["M1|L5|DA"]["all"]["counts"], src_of(B76), "main.M1|L5|DA.all.counts"))
    head = [r"set & used for & rows & discharges (shots) & clusters (sessions) \\"]
    body = tabular("@{}llrrr@{}", head, rows, "footnotesize", "4pt")
    cap = (r"Row sets. KSTAR: D1 rows of the cut population; a set used to compare availability rules holds the "
           r"rows whose inputs are complete under every rule compared. Clusters are runs of shot numbers with "
           r"steps of at most three; MAST sessions are experiment dates. Target L5 throughout.")
    emit("Sets", body, cap, "row sets, discharges and clusters (KSTAR), shots and sessions (MAST)",
         "b75 sets/designs/main.*.counts; b73 increments.V1|L5; b78 counts.V1.L5; b77 row_set; b76 main.M1|L5|*.counts",
         "sec. 2/3 (data and method)")


def tab_scored():
    b75, b78, b76, b79 = J(B75), J(B78), J(B76), J(B79)
    rows = []

    def trip(c, src, path):
        return (cnt(c["rows"], src, path + ".rows") + " / " + cnt(c["discharges"], src, path + ".discharges")
                + " / " + cnt(c["clusters"], src, path + ".clusters"))

    for d in DES:
        rows.append([design_label(d), trip(b75["counts"][d]["common|L5"], src_of(B75), f"counts.{d}.common|L5"),
                     trip(b75["counts"][d]["S|L5"], src_of(B75), f"counts.{d}.S|L5"),
                     trip(b78["counts"][d]["L5"], src_of(B78), f"counts.{d}.L5")])
    rows.append("MID")
    rows.append([r"\multicolumn{4}{@{}l}{MAST (rows / shots / sessions), common set of rules D5 to D150}\\[1pt]"])
    rows[-1] = r"\multicolumn{4}{@{}l}{\textit{MAST: rows / shots / sessions, common set of rules D5 to D150}} \\"
    for d in MDES:
        c = b76["main"][f"{d}|L5|common"]["all"]["counts"]
        p = f"main.{d}|L5|common.all.counts"
        rows.append([MLAB[d], cnt(c["rows"], src_of(B76), p + ".rows") + " / " + cnt(c["shots"], src_of(B76), p + ".shots")
                     + " / " + cnt(c["sessions"], src_of(B76), p + ".sessions"), "--", "--"])
    head = [r"design & common set (D rules) & S set & K set \\"]
    body = tabular("@{}lrrr@{}", head, rows, "footnotesize", "5pt")
    cap = (r"Scored rows per design (rows / discharges / shot clusters), target L5. V1 and V2 score every "
           r"discharge once per re-partition or assignment; V3 scores fifths two to five; V4 scores the shots from "
           r"32312 on. The beam inputs of B.79 drop no row of these sets.")
    reg("32312", src_of(B75), "main.V1|L5|S key 'block from 32312'", "shot number from a block key")
    emit("Scored", body, cap, "scored rows/discharges/clusters per design and set",
         "b75 counts.*; b78 counts.*.L5; b76 main.M*|L5|common.all.counts", "sec. 3 / supplement")


# ----------------------------------------------------------------------------------------------
# 2. RMSE by availability rule
# ----------------------------------------------------------------------------------------------
RULE_LAB = {"D10": "D10", "D10S": r"D10 (S rows)", "S": "S"}


def rule_tok(r, src, path):
    m = re.fullmatch(r"[DK](\d+)S?", r)
    if m and len(m.group(1)) >= 3:
        reg(m.group(1), src, path, "delay of the rule name in ms")
    return RULE_LAB.get(r, r)


def tab_rules_v1():
    b = J(B75)
    s = src_of(B75)
    rows = []
    m = b["main"]["V1|L5|common"]["all"]
    for r in RULES:
        c = m[r]
        p = f"main.V1|L5|common.all.{r}"
        rm = c["rmse_eV"]
        rows.append([rule_tok(r, s, p)] + [ev(rm[k], s, f"{p}.rmse_eV.{k}") for k in ("H", "G", "T0", "TX", "O")]
                    + [rat(c.get("staleness T0"), s, p + ".staleness T0", "stack") if r != "D10" else "1",
                       pct(c["TX/T0"], s, p + ".TX/T0", "stack"),
                       rat(c.get("recovered share"), s, p + ".recovered share", "stack") if r != "D10" else "--"])
    rows.append("MID")
    ms = b["main"]["V1|L5|S"]["all"]
    for r in ("D10S", "S"):
        c = ms[r]
        p = f"main.V1|L5|S.all.{r}"
        rm = c["rmse_eV"]
        rows.append([rule_tok(r, s, p)] + [ev(rm[k], s, f"{p}.rmse_eV.{k}") for k in ("H", "G", "T0", "TX", "O")]
                    + [rat(c.get("staleness T0"), s, p + ".staleness T0", "stack") if r == "S" else "1",
                       pct(c["TX/T0"], s, p + ".TX/T0", "stack"),
                       rat(c.get("recovered share"), s, p + ".recovered share", "stack") if r == "S" else "--"])
    head = [r" & \multicolumn{5}{c}{RMSE (eV)} & cost of & fast-channel & recovered \\",
            r"\cmidrule(lr){2-6}",
            r"rule & hold & causal GP & T0 & TX & O & staleness & increment (\%) & share \\"]
    body = tabular("@{}lrrrrrccc@{}", head, rows, "footnotesize", "4.5pt")
    cc = m["counts"]
    cs = ms["counts"]
    cap = (r"Error of the level estimate by availability rule, KSTAR, random design V1, target L5. Upper block: "
           r"fixed delays on the common set (" + cnt(cc["rows"], s, "main.V1|L5|common.all.counts.rows")
           + " rows, " + cnt(cc["discharges"], s, "main.V1|L5|common.all.counts.discharges") + " discharges, "
           + cnt(cc["clusters"], s, "main.V1|L5|common.all.counts.clusters") + r" clusters); lower block: the "
           r"beam-modulation schedule S and the previous frame on the same rows (" + cnt(cs["rows"], s, "main.V1|L5|S.all.counts.rows")
           + " rows, " + cnt(cs["discharges"], s, "main.V1|L5|S.all.counts.discharges") + " discharges, "
           + cnt(cs["clusters"], s, "main.V1|L5|S.all.counts.clusters") + r" clusters). O is the frame's own "
           r"reading, which exists only once the frame has been analysed; S assumes an analysis time of zero. "
           r"Cost of staleness = MSE(T0, rule) / MSE(T0, D10) on the same rows; increment = reduction of the MSE "
           r"of TX against T0; recovered share = (MSE(T0) $-$ MSE(TX)) / (MSE(T0) $-$ MSE(T0, D10)). "
           r"Intervals: 95 \%, shot clusters resampled; bold: the interval excludes zero.")
    emit("RulesVone", body, cap, "RMSE by availability rule (V1), cost of staleness, increment, recovered share",
         "b75 main.V1|L5|common.all.{D10..D1000}, main.V1|L5|S.all.{S,D10S}", "4.1 (main)")


def tab_rules_designs():
    b = J(B75)
    s = src_of(B75)
    rows = []
    for d in ["V2", "V3", "V4a", "V4b"]:
        rows.append(r"\multicolumn{8}{@{}l}{\textit{" + design_label(d) + r"}} \\")
        m = b["main"][f"{d}|L5|common"]["all"]
        for r in ["D10", "D100", "D300", "D650", "D1000"]:
            c = m[r]
            p = f"main.{d}|L5|common.all.{r}"
            rm = c["rmse_eV"]
            rows.append([rule_tok(r, s, p)] + [ev(rm[k], s, f"{p}.rmse_eV.{k}") for k in ("H", "T0", "TX", "O")]
                        + [rat(c.get("staleness T0"), s, p + ".staleness T0") if r != "D10" else "1",
                           pct(c["TX/T0"], s, p + ".TX/T0"),
                           rat(c.get("recovered share"), s, p + ".recovered share") if r != "D10" else "--"])
        c = b["main"][f"{d}|L5|S"]["all"]["S"]
        p = f"main.{d}|L5|S.all.S"
        rm = c["rmse_eV"]
        rows.append(["S"] + [ev(rm[k], s, f"{p}.rmse_eV.{k}") for k in ("H", "T0", "TX", "O")]
                    + [rat(c.get("staleness T0"), s, p + ".staleness T0"), pct(c["TX/T0"], s, p + ".TX/T0"),
                       rat(c.get("recovered share"), s, p + ".recovered share")])
        if d != "V4b":
            rows.append("MID")
    head = [r" & \multicolumn{4}{c}{RMSE (eV)} & & & \\", r"\cmidrule(lr){2-5}",
            r"rule & hold & T0 & TX & O & cost of staleness & increment (\%) & recovered share \\"]
    body = tabular("@{}lrrrrccc@{}", head, rows, "footnotesize", "4pt")
    cap = (r"As table \ref{tab:rulesvone} for the other three designs (target L5): fixed delays on the common "
           r"set, S on the S set. The cost of staleness of S is taken against D10 on the S rows. Intervals: 95 \%, "
           r"shot clusters resampled; bold: the interval excludes zero.")
    emit("RulesDesigns", body, cap, "RMSE by rule for V2, V3, V4 (both fits), with cost, increment, recovered share",
         "b75 main.{V2,V3,V4a,V4b}|L5|{common,S}.all.*", "supplement (4.1)")


# ----------------------------------------------------------------------------------------------
# 3. increment by design
# ----------------------------------------------------------------------------------------------
def tab_increment():
    b = J(B75)
    s = src_of(B75)
    rows = []
    for d in DES:
        m = b["main"][f"{d}|L5|common"]["all"]
        ms = b["main"][f"{d}|L5|S"]["all"]["S"]
        rows.append([design_label(d), pct(m["D10"]["TX/T0"], s, f"main.{d}|L5|common.all.D10.TX/T0"),
                     pct(m["D650"]["TX/T0"], s, f"main.{d}|L5|common.all.D650.TX/T0"),
                     pct(ms["TX/T0"], s, f"main.{d}|L5|S.all.S.TX/T0")])
    rows.append("MID")
    rows.append(r"\textit{MAST} & \textit{D5} & \textit{D100} & \\")
    rows.append(r"\midrule")
    m6 = J(B76)
    for d in MDES:
        m = m6["main"][f"{d}|L5|common"]["all"]
        rows.append([MLAB[d], pct(m["D5"]["TX/T0"], src_of(B76), f"main.{d}|L5|common.all.D5.TX/T0"),
                     pct(m["D100"]["TX/T0"], src_of(B76), f"main.{d}|L5|common.all.D100.TX/T0"), ""])
    head = [r"\textit{KSTAR} & \textit{D10} & \textit{D650} & \textit{S} \\"]
    body = tabular("@{}lccc@{}", head, rows, "footnotesize", "6pt")
    cap = (r"Fast-channel increment: reduction of the MSE of the level estimate (target L5) in per cent, TX "
           r"against T0 on the same rows, by validation design and availability rule. KSTAR: D10 and D650 on the "
           r"common set, S on the S set; fast inputs are BES and ECEI. MAST: CXRS history with the Thomson "
           r"scalars of the same radius as fast inputs; delays D5 (previous frame) and D100. Intervals: 95 \%, "
           r"KSTAR shot clusters and MAST sessions resampled; bold: the interval excludes zero.")
    emit("Increment", body, cap, "fast-channel increment by design and rule, KSTAR D10/D650/S and MAST D5/D100",
         "b75 main.*|L5|common.all.{D10,D650}.TX/T0, main.*|L5|S.all.S.TX/T0; b76 main.M*|L5|common.all.{D5,D100}.TX/T0",
         "4.3 (main); 4.4 for the MAST rows")


def tab_increment_ladder():
    b = J(B75)
    s = src_of(B75)
    rows = []
    for d in DES:
        m = b["main"][f"{d}|L5|common"]["all"]
        rows.append([design_label(d).replace(", fit", ",\\\\ fit") if False else design_label(d)]
                    + [pct(m[r]["TX/T0"], s, f"main.{d}|L5|common.all.{r}.TX/T0", "stack") for r in RULES])
    rows.append("MID")
    m6 = J(B76)
    rows.append(r"\textit{MAST} & \textit{D5} & \textit{D15} & \textit{D25} & \textit{D50} & \textit{D100} & \textit{D150} \\")
    rows.append(r"\midrule")
    for d in MDES:
        m = m6["main"][f"{d}|L5|common"]["all"]
        rows.append([MLAB[d]] + [pct(m[r]["TX/T0"], src_of(B76), f"main.{d}|L5|common.all.{r}.TX/T0", "stack")
                                 for r in MRULES])
    head = [r"\textit{KSTAR} & " + " & ".join(r"\textit{" + rule_tok(r, s, "main.V1|L5|common.all key " + r) + "}"
                                               for r in RULES) + r" \\"]
    body = tabular("@{}lcccccc@{}", head, rows, "footnotesize", "3.5pt")
    cap = (r"Fast-channel increment (reduction of the MSE of TX against T0 in per cent, target L5) along the "
           r"ladder of delays, every design. KSTAR on the common set, MAST on its common set. Intervals: 95 \%, "
           r"KSTAR shot clusters and MAST sessions resampled; bold: the interval excludes zero.")
    emit("IncrementLadder", body, cap, "increment along the whole delay ladder, every design, KSTAR and MAST",
         "b75 main.*|L5|common.all.D*.TX/T0; b76 main.M*|L5|common.all.D*.TX/T0", "supplement (4.1, 4.3, 4.4)")


def tab_fifths():
    b = J(B75)
    s = src_of(B75)
    fif = b["designs"]["v3_fifths"]
    rows = []
    for key in ("fifth 2", "fifth 3", "fifth 4", "fifth 5", "fifths 2+3"):
        c = b["v3c"][key]
        p = f"v3c.{key}"
        if key.startswith("fifth "):
            i = int(key.split()[1]) - 1
            lab = (key.split()[1] + " (" + shot_tok(fif[i]["shots"][0], s, f"designs.v3_fifths[{i}].shots[0]") + "--"
                   + shot_tok(fif[i]["shots"][1], s, f"designs.v3_fifths[{i}].shots[1]") + ")")
        else:
            lab = "2 and 3"
        row = [lab, cnt(c["rows"], s, p + ".rows") + " / " + cnt(c["discharges"], s, p + ".discharges")
               + " / " + cnt(c["clusters"], s, p + ".clusters")]
        for r in ("D10", "D650"):
            for k in ("V3 TX/T0 ", "V3c TX/T0 ", "V3 minus V3c TX/T0 "):
                row.append(pct(c.get(k + r), s, p + "." + k + r, "stack"))
        rows.append(row)
    head = [r" & & \multicolumn{3}{c}{D10} & \multicolumn{3}{c}{D650} \\",
            r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}",
            r"fifth (shots) & rows / disch. / clusters & V3 & V3c & V3 $-$ V3c & V3 & V3c & V3 $-$ V3c \\"]
    body = tabular("@{}lrcccccc@{}", head, rows, "footnotesize", "3.2pt")
    cap = (r"Rolling-origin design by scored fifth of the shot list (target L5, common set): fast-channel "
           r"increment (reduction of the MSE of TX against T0 in per cent) under V3, under the size-matched "
           r"control V3c (fitted on as many discharges as V3, drawn at random from all discharges outside the test fifth), and their "
           r"difference in percentage points. Dashes: no V3c for the last fifth. Intervals: 95 \%, shot clusters "
           r"resampled; bold: the interval excludes zero.")
    emit("Fifths", body, cap, "V3 by scored fifth and the size-matched control V3c (D10, D650)",
         "b75 v3c.*; designs.v3_fifths", "4.3 (main or supplement)")


# ----------------------------------------------------------------------------------------------
# 4. which diagnostic, beyond the beam waveforms; controls; refit
# ----------------------------------------------------------------------------------------------
def tab_which():
    b5, b9, b1 = J(B75), J(B79), J(B81)
    rows = []
    for d in DES:
        for r in ("D650", "S"):
            k5 = f"main.{d}|L5|common.all.D650" if r == "D650" else f"main.{d}|L5|S.all.S"
            c5 = b5["main"][f"{d}|L5|common"]["all"]["D650"] if r == "D650" else b5["main"][f"{d}|L5|S"]["all"]["S"]
            k9 = f"main.{d}|L5|common.all.D650" if r == "D650" else f"main.{d}|L5|S.all.S"
            c9 = b9["main"][f"{d}|L5|common"]["all"]["D650"] if r == "D650" else b9["main"][f"{d}|L5|S"]["all"]["S"]
            c1 = b1["main"][d][r]
            k1 = f"main.{d}.{r}"
            rows.append([design_label(d) if r == "D650" else "", r,
                         # B.75 where it registered the cell (V1, V4a, V4b); B.81 only for V2, V3
                         (pct(c5["TE/T0"], src_of(B75), k5 + ".TE/T0", "stack") if "TE/T0" in c5
                          else pct(c1["TE/T0"], src_of(B81), k1 + ".TE/T0", "stack")),
                         (pct(c5["TB/T0"], src_of(B75), k5 + ".TB/T0", "stack") if "TB/T0" in c5
                          else pct(c1["TB/T0"], src_of(B81), k1 + ".TB/T0", "stack")),
                         pct(c5["TX/T0"], src_of(B75), k5 + ".TX/T0", "stack"),
                         pct(c9["TN/T0"], src_of(B79), k9 + ".TN/T0", "stack"),
                         pct(c9["TXN/TN"], src_of(B79), k9 + ".TXN/TN", "stack"),
                         pct(c9["TXN/TX"], src_of(B79), k9 + ".TXN/TX", "stack")])
        if d != "V4b":
            rows.append("MID")
    head = [r" & & ECEI only & BES only & both & beam alone & both, beam & beam, both \\",
            r"design & rule & TE/T0 & TB/T0 & TX/T0 & TN/T0 & TXN/TN & TXN/TX \\"]
    body = tabular("@{}llcccccc@{}", head, rows, "footnotesize", "3.2pt")
    cap = (r"Which diagnostic, and beyond the beam waveforms: reduction of the MSE in per cent (target L5), "
           r"history 0.65~s old (D650, common set) and beam-modulation schedule (S, S set). TE and TB are T0 "
           r"with the ECEI or the BES half of the fast inputs (V1 and V4 from B.75; V2 and V3 fitted in B.81); TN is "
           r"T0 with the 19 beam scalars; TXN is TX with them. ``Both, beam'': the fast diagnostics after the beam "
           r"waveforms are given; ``beam, both'': the beam waveforms after the fast diagnostics. Beam columns as "
           r"delivered (registered readout). Intervals: 95 \%, shot clusters resampled; bold: the interval excludes zero.")
    reg("0.65", src_of(B75), "age_under_S.L5.newest_value_age_s.quantiles.0.5", "median age in s, two decimals")
    reg("19", src_of(B79), "beam_scalars (list length)", "count")
    emit("Which", body, cap, "ECEI only, BES only, both; beam alone; both given beam; beam given both (D650, S, per design)",
         "b75 main.{V1,V4a,V4b}.{D650,S}.{TE,TB}/T0 and main.*.{D650,S}.TX/T0; b81 main.{V2,V3}.{D650,S}.{TE,TB}/T0; b79 main.*.{D650,S}.{TN/T0,TXN/TN,TXN/TX}", "4.2 (main)")


def lead_controls_b75():
    """Parse the unregistered controls of b75_lead_checks.md (part C)."""
    p = PAPER / "b75_lead_checks.md"
    t = p.read_text(encoding="utf-8")
    m1 = re.search(r"placebo \(fast and anchor scalars permuted over rows\): TX over T0 ([+-]\d\.\d+)", t)
    m2 = re.search(r"fast scalars taken at the anchor row .*?, (\d+) test rows: over T0 ([+-]\d\.\d+); present fast scalars on the same rows: ([+-]\d\.\d+)", t)
    out = {"placebo": float(m1.group(1)), "anchor_rows": int(m2.group(1)), "anchor": float(m2.group(2)),
           "present": float(m2.group(3))}
    return out, "docs/paper/b75_lead_checks.md"


def pct_pt_only(v, src, path):
    return fnum(v, "%.1f", src, path, "100 x skill, one decimal (parsed from md, no interval)", 100)


def tab_controls():
    rows = []
    b1 = J(B81)
    for d in DES:
        for r in ("D650", "S"):
            c = b1["main"][d][r]
            p = f"main.{d}.{r}"
            rows.append(["B.81" if (d == "V1" and r == "D650") else "", design_label(d) if r == "D650" else "", r,
                         "TX vs T0", pct(c["TXp/T0"], src_of(B81), p + ".TXp/T0", "stack"),
                         pct(c["TXo/T0"], src_of(B81), p + ".TXo/T0", "stack"),
                         pct(c["TX/TXo"], src_of(B81), p + ".TX/TXo", "stack")])
    rows.append("MID")
    b8 = J(B78)
    c = b8["main"]["V1|L5"]["all"]["K10"]
    p = "main.V1|L5.all.K10"
    rows.append(["B.78", design_label("V1"), "K10", "TX vs T0", pct(c["TXp/T0"], src_of(B78), p + ".TXp/T0", "stack"),
                 pct(c["TXo/T0"], src_of(B78), p + ".TXo/T0", "stack"), pct(c["TX/TXo"], src_of(B78), p + ".TX/TXo", "stack")])
    b9 = J(B79)
    c = b9["main"]["V1|L5|common"]["all"]["D650"]
    p = "main.V1|L5|common.all.D650"
    rows.append(["B.79", design_label("V1"), "D650", "TXN vs TN",
                 pct(c["TXNp/TN"], src_of(B79), p + ".TXNp/TN", "stack"), "--", "--"])
    rows.append(["B.79", design_label("V1"), "D650", "TN vs T0 (beam)",
                 pct(c["TNp/T0"], src_of(B79), p + ".TNp/T0", "stack"), "--", "--"])
    b7 = J(B77)
    c = b7["designs"]["V1"]["all"]
    rows.append(["B.77", design_label("V1"), "D10", "CX vs C0",
                 pct(c["CXp/C0 (placebo)"], src_of(B77), "designs.V1.all.CXp/C0 (placebo)", "stack"), "--", "--"])
    b6 = J(B76)
    c = b6["main"]["M1|L5|common"]["all"]["controls"]
    p = "main.M1|L5|common.all.controls"
    rows.append(["B.76 (MAST)", MLAB["M1"], "D100", "TX vs T0", pct(c["TXp/T0"], src_of(B76), p + ".TXp/T0", "stack"),
                 pct(c["TXo/T0"], src_of(B76), p + ".TXo/T0", "stack"), pct(c["TX/TXo"], src_of(B76), p + ".TX/TXo", "stack")])
    head = [r" & & & & fast inputs & fast inputs at & present over \\",
            r"batch & design & rule & comparison & permuted & the anchor row & anchor row \\"]
    body = tabular("@{}llllccc@{}", head, rows, "footnotesize", "3pt")
    cap = (r"The two controls beside the fast inputs (target L5), reduction of the MSE in per cent. Upper block "
           r"(B.81): every design under the history 0.65~s old (D650, common set) and the beam-modulation schedule "
           r"(S, S set). Lower block: the controls registered with other batches. ``Permuted'': the fast inputs "
           r"(for B.79 TN: the beam scalars) shuffled over rows and the estimator refitted; the reduction against "
           r"the same reference should be zero. ``At the anchor row'': the fast inputs taken at the row of the "
           r"newest visible CES value instead of row $t$; ``present over anchor row'': TX against that estimator. "
           r"Intervals: 95 \%, shot clusters (MAST: sessions) resampled; bold: the interval excludes zero.")
    emit("Controls", body, cap, "permuted-input and anchor-row controls: B.81 per design (D650, S); B.78, B.79, B.77, B.76",
         "b81 main.*.{D650,S}.{TXp/T0,TXo/T0,TX/TXo}; b78 main.V1|L5.all.K10; b79 main.V1|L5|common.all.D650.{TXNp/TN,TNp/T0}; "
         "b77 designs.V1.all.CXp/C0; b76 main.M1|L5|common.all.controls", "4.2 (main) / beside every table of fast inputs")


def stat_val(v, src, path):
    """Format a registered statement value as registered: signed three decimals with the interval."""
    if isinstance(v, dict) and "point" in v and not isinstance(v["point"], dict):
        k = None
        for kk in ("cluster", "session", "discharge", "shot"):
            if kk in v:
                k = kk
                break
        pt = fnum(v["point"], "%+.3f", src, path + ".point", "as registered, signed three decimals")
        if k is None:
            return pt
        a = fnum(v[k][0], "%+.3f", src, f"{path}.{k}[0]", f"as registered, {k} interval")
        b = fnum(v[k][1], "%+.3f", src, f"{path}.{k}[1]", f"as registered, {k} interval")
        return f"{pt} [{a}, {b}]"
    if isinstance(v, dict):
        parts = []
        for kk, vv in v.items():
            if isinstance(vv, dict) and "point" in vv:
                parts.append(esc(kk, src, path + " key") + ": " + stat_val(vv, src, f"{path}.{kk}"))
        return "; ".join(parts)
    return esc(v, src, path)


def tab_refit():
    b9 = J(B79)
    s = src_of(B79)
    sb = b9["refit_without_13_files"]["statements_beside_main"]
    rows = []
    for k, v in sb.items():
        p = f"refit_without_13_files.statements_beside_main.{k}"
        txt = b9["statements"][k]["text"]
        rows.append([k, r"\parbox[t]{4.5cm}{\raggedright " + esc(txt, s, f"statements.{k}.text") + r"\strut}",
                     stat_val(v["main"]["value"], s, p + ".main.value"),
                     "holds" if v["main"]["holds"] else "does not hold",
                     stat_val(v["refit"]["value"], s, p + ".refit.value"),
                     "holds" if v["refit"].get("cluster_interval_rule_met") else "does not hold"])
    head = [r" & & \multicolumn{2}{c}{columns as delivered (registered)} & \multicolumn{2}{c}{without the 13 files (refit)} \\",
            r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}",
            r"id & statement & value & verdict & value & rule met \\"]
    body = tabular("@{}llllll@{}", head, rows, "footnotesize", "3pt")
    n13 = len(b9["refit_without_13_files"]["invalid_files"])
    reg(str(n13), s, "refit_without_13_files.invalid_files (length)", "count")
    cap = (r"B.79 statements with the beam columns as delivered (the registered readout) and refitted without "
           r"the 13 shot files whose NBI-2 power columns are invalid (run after the registered readout had been "
           r"read; a sensitivity, not judged). Skill as registered (fraction, signed), 95 \% interval with shot "
           r"clusters resampled. Target L5.")
    emit("Refit", body, cap, "B.79 main readout vs refit without the 13 files with invalid NBI-2 power columns",
         "b79 refit_without_13_files.statements_beside_main.*", "supplement (4.2)")


# ----------------------------------------------------------------------------------------------
# 5. MAST
# ----------------------------------------------------------------------------------------------
def tab_mast_rmse():
    b = J(B76)
    s = src_of(B76)
    rows = []
    for d in ("M1", "M4"):
        rows.append(r"\multicolumn{9}{@{}l}{\textit{" + MLAB[d] + r"}} \\")
        m = b["main"][f"{d}|L5|common"]["all"]
        for r in MRULES:
            c = m[r]
            p = f"main.{d}|L5|common.all.{r}"
            rm = c["rmse_eV"]
            al = b["rta_coefficients"][f"{d}|L5|{r}"]["alpha"]["mean"]
            rows.append([rule_tok(r, s, p)] + [ev(rm[k], s, f"{p}.rmse_eV.{k}") for k in ("H", "G", "T0", "TX", "O", "RT", "RTa")]
                        + [fnum(al, "%.3f", s, f"rta_coefficients.{d}|L5|{r}.alpha.mean", "coefficient, three decimals")])
        if d == "M1":
            rows.append("MID")
    head = [r" & \multicolumn{7}{c}{RMSE (eV)} & \\", r"\cmidrule(lr){2-8}",
            r"rule & hold & causal GP & T0 & TX & O & RT & RTa & $\alpha$ of RTa \\"]
    body = tabular("@{}lrrrrrrrr@{}", head, rows, "footnotesize", "4.5pt")
    cap = (r"MAST, pedestal-top band, target L5 (median of the good values of the next five 5~ms frames): RMSE "
           r"by delay of the CXRS history, random design M1 and chronological design M4, common set. TX adds the "
           r"Thomson scalars of the same radius (present and anchor frame). RT multiplies the hold value by the "
           r"ratio of the Thomson $T_e$ (no coefficient); RTa fits $\ln T = \ln H + \alpha \ln(T_e\ \mathrm{ratio}) + \beta$ "
           r"($\alpha$: mean over fit units). The frames are not hidden on MAST: the delay is imposed.")
    emit("MastRmse", body, cap, "MAST RMSE by delay (M1, M4): hold, GP, T0, TX, O, closed ratio form RT, RTa and its alpha",
         "b76 main.{M1,M4}|L5|common.all.D*.rmse_eV; rta_coefficients.*.alpha.mean", "4.4 (main)")


def tab_mast_cost():
    b = J(B76)
    s = src_of(B76)
    rows = []
    for d in MDES:
        m = b["main"][f"{d}|L5|common"]["all"]
        first = True
        for r in ("D25", "D50", "D100", "D150"):
            c = m[r]
            p = f"main.{d}|L5|common.all.{r}"
            rows.append([MLAB[d] if first else "", rule_tok(r, s, p),
                         rat(c["staleness T0"], s, p + ".staleness T0", "stack"),
                         pct(c["TX/T0"], s, p + ".TX/T0", "stack"),
                         rat(c["recovered share"], s, p + ".recovered share", "stack"),
                         pct(c["TX/O"], s, p + ".TX/O", "stack")])
            first = False
        if d != "M4":
            rows.append("MID")
    head = [r"design & rule & cost of staleness & increment (\%) & recovered share & TX vs O (\%) \\"]
    body = tabular("@{}llcccc@{}", head, rows, "footnotesize", "5pt")
    cap = (r"MAST, target L5, common set: cost of staleness of T0 (MSE against the previous frame D5), Thomson "
           r"increment (reduction of the MSE of TX against T0), recovered share, and TX against the frame's own "
           r"reading O, by design and delay. Intervals: 95 \%, sessions resampled; bold: the interval excludes zero.")
    emit("MastCost", body, cap, "MAST cost of staleness, increment, recovered share, TX vs O by design and delay",
         "b76 main.M*|L5|common.all.D{25,50,100,150}.{staleness T0,TX/T0,recovered share,TX/O}", "4.4 / supplement")


def tab_mast_dalpha():
    b = J(B76)
    s = src_of(B76)
    rows = []
    for d in ("M1", "M4"):
        m = b["main"][f"{d}|L5|DA"]["all"]
        cc = m["counts"]
        for r in ("D5", "D100"):
            c = m[r]
            p = f"main.{d}|L5|DA.all.{r}"
            rows.append([MLAB[d] if r == "D5" else "", r,
                         pct(c["TD/T0"], s, p + ".TD/T0", "stack"), pct(c["TA/T0"], s, p + ".TA/T0", "stack"),
                         pct(c["TX/T0"], s, p + ".TX/T0", "stack"), pct(c["TA/TX"], s, p + ".TA/TX", "stack")])
        rows.append([r"\multicolumn{6}{@{}l}{\scriptsize rows " + cnt(cc["rows"], s, f"main.{d}|L5|DA.all.counts.rows")
                     + ", shots " + cnt(cc["shots"], s, f"main.{d}|L5|DA.all.counts.shots") + ", sessions "
                     + cnt(cc["sessions"], s, f"main.{d}|L5|DA.all.counts.sessions") + r"}"])
        rows[-1] = rows[-1][0] + r" \\"
        if d == "M1":
            rows.append("MID")
    head = [r" & & D-alpha & D-alpha + Thomson & Thomson & D-alpha over Thomson \\",
            r"design & rule & TD vs T0 & TA vs T0 & TX vs T0 & TA vs TX \\"]
    body = tabular("@{}llcccc@{}", head, rows, "footnotesize", "3.5pt")
    cap = (r"MAST, rows with a D-alpha signal, target L5: reduction of the MSE in per cent. The D-alpha scale is "
           r"set over the whole shot block and is therefore not causal within a shot. Intervals: 95 \%, sessions "
           r"resampled; bold: the interval excludes zero.")
    emit("MastDalpha", body, cap, "MAST D-alpha inputs (TD, TA) against T0 and TX, M1 and M4",
         "b76 main.{M1,M4}|L5|DA.all.{D5,D100}", "4.4 / supplement")


# ----------------------------------------------------------------------------------------------
# 6. sparse CES
# ----------------------------------------------------------------------------------------------
def kprime(er, src, path):
    pt = er["point"]["k_prime"]
    if isinstance(pt, str):
        return esc(pt, src, path + ".point.k_prime")
    lo, hi = er["cluster"]["k_prime"]
    a = fnum(pt, "%.2f", src, path + ".point.k_prime", "frames, two decimals")
    b = fnum(lo, "%.2f", src, path + ".cluster.k_prime[0]", "frames, two decimals (cluster interval)")
    c = fnum(hi, "%.2f", src, path + ".cluster.k_prime[1]", "frames, two decimals (cluster interval)")
    return r"\ci{" + a + "}{[" + b + ", " + c + "]}"


def tab_sparse():
    b = J(B78)
    s = src_of(B78)
    m = b["main"]["V1|L5"]["all"]
    rows = []
    for r in KRULES:
        c = m[r]
        p = f"main.V1|L5.all.{r}"
        k = int(r[1:])
        sp = str(10 * k)
        if len(sp) >= 3:
            reg(sp, s, f"main.V1|L5.all key {r}", "spacing in ms = 10 ms x k of the rule name")
        rm = c["rmse_eV"]
        er = c["equivalent rate"]
        fac = er["point"]["factor_k_over_k_prime"]
        rows.append([r, sp, ev(rm["T0"], s, p + ".rmse_eV.T0"), ev(rm["TX"], s, p + ".rmse_eV.TX"),
                     pct(c["TX/T0"], s, p + ".TX/T0", "stack"),
                     rat(c["cost T0"], s, p + ".cost T0", "stack") if "cost T0" in c else "1",
                     rat(c["recovered share"], s, p + ".recovered share", "stack") if "recovered share" in c else "--",
                     kprime(er, s, p + ".equivalent rate") if r != "K1" else "--",
                     (fnum(fac, "%.2f", s, p + ".equivalent rate.point.factor_k_over_k_prime", "ratio, two decimals")
                      if not isinstance(fac, str) else esc(fac, s, p + ".equivalent rate.point.factor_k_over_k_prime"))
                     if r != "K1" else "--"])
    head = [r" & spacing & \multicolumn{2}{c}{RMSE (eV)} & fast-channel & cost of & recovered & equivalent & spacing \\",
            r"\cmidrule(lr){3-4}",
            r"rule & (ms) & T0 & TX & increment (\%) & sparseness & share & $k'$ (frames) & ratio $k/k'$ \\"]
    body = tabular("@{}lrrrccccc@{}", head, rows, "footnotesize", "4pt")
    cc = m["counts"]
    cap = (r"A CES value only every $k$-th frame (KSTAR, random design V1, target L5): error of the history-only "
           r"estimator T0 and of TX on the frames that no thinning keeps (" + cnt(cc["rows"], s, "main.V1|L5.all.counts.rows")
           + " rows, " + cnt(cc["discharges"], s, "main.V1|L5.all.counts.discharges") + " discharges, "
           + cnt(cc["clusters"], s, "main.V1|L5.all.counts.clusters") + r" clusters; RMSE of the frame's own "
           r"reading " + ev(m["K1"]["rmse_eV"]["O"], s, "main.V1|L5.all.K1.rmse_eV.O") + r"~eV). Cost of sparseness = "
           r"MSE(T0, K$k$) / MSE(T0, K1). Equivalent spacing $k'$: the spacing at which T0 without fast channels "
           r"has the MSE of TX at spacing $k$. Intervals: 95 \%, shot clusters resampled; bold: the interval excludes zero.")
    emit("Sparse", body, cap, "RMSE of T0 and TX by spacing of the CES values (V1), increment, cost, recovered share, equivalent spacing",
         "b78 main.V1|L5.all.K*", "4.5 (main)")


def tab_sparse_designs():
    b = J(B78)
    s = src_of(B78)
    rows = []
    for d in DES:
        m = b["main"][f"{d}|L5"]["all"]
        rows.append([design_label(d)] + [pct(m[r]["TX/T0"], s, f"main.{d}|L5.all.{r}.TX/T0", "stack") for r in KRULES])
    head = [r"design & " + " & ".join(KRULES) + r" \\"]
    body = tabular("@{}lcccccc@{}", head, rows, "footnotesize", "4pt")
    cap = (r"Fast-channel increment (reduction of the MSE of TX against T0 in per cent, target L5) when the CES "
           r"values are kept only every $k$-th frame, every design. Intervals: 95 \%, shot clusters resampled; bold: "
           r"the interval excludes zero.")
    emit("SparseDesigns", body, cap, "increment by spacing K1..K50 and design", "b78 main.*|L5.all.K*.TX/T0",
         "4.5 (main or supplement)")


# ----------------------------------------------------------------------------------------------
# 7. where it does not hold
# ----------------------------------------------------------------------------------------------
BLOCKS = ["block up to 31991", "block 32002-32310", "block from 32312", "all"]


def tab_blocks():
    b5, b9 = J(B75), J(B79)
    rows = []
    for blk in BLOCKS:
        c5 = b5["main"]["V1|L5|S"][blk]
        c9 = b9["main"]["V1|L5|S"][blk]
        p5 = f"main.V1|L5|S.{blk}"
        cc = c5["counts"]
        rm = c5["S"]["rmse_eV"]
        lab = "all shots" if blk == "all" else bldate(blk)
        if blk == "all":
            rows.append("MID")
        rows.append([lab, cnt(cc["rows"], src_of(B75), p5 + ".counts.rows") + " / "
                     + cnt(cc["discharges"], src_of(B75), p5 + ".counts.discharges") + " / "
                     + cnt(cc["clusters"], src_of(B75), p5 + ".counts.clusters"),
                     ev(rm["T0"], src_of(B75), p5 + ".S.rmse_eV.T0"), ev(rm["TX"], src_of(B75), p5 + ".S.rmse_eV.TX"),
                     ev(rm["O"], src_of(B75), p5 + ".S.rmse_eV.O"),
                     pct(c5["S"]["TX/T0"], src_of(B75), p5 + ".S.TX/T0", "stack"),
                     pct(c5["S"]["TE/T0"], src_of(B75), p5 + ".S.TE/T0", "stack"),
                     pct(c9["S"]["TXN/TN"], src_of(B79), f"main.V1|L5|S.{blk}.S.TXN/TN", "stack")])
    rows.append("MID")
    for d in ("V4a", "V4b"):
        c5 = b5["main"][f"{d}|L5|S"]["all"]
        c9 = b9["main"][f"{d}|L5|S"]["all"]
        p5 = f"main.{d}|L5|S.all"
        cc = c5["counts"]
        rm = c5["S"]["rmse_eV"]
        rows.append([design_label(d), cnt(cc["rows"], src_of(B75), p5 + ".counts.rows") + " / "
                     + cnt(cc["discharges"], src_of(B75), p5 + ".counts.discharges") + " / "
                     + cnt(cc["clusters"], src_of(B75), p5 + ".counts.clusters"),
                     ev(rm["T0"], src_of(B75), p5 + ".S.rmse_eV.T0"), ev(rm["TX"], src_of(B75), p5 + ".S.rmse_eV.TX"),
                     ev(rm["O"], src_of(B75), p5 + ".S.rmse_eV.O"),
                     pct(c5["S"]["TX/T0"], src_of(B75), p5 + ".S.TX/T0", "stack"),
                     pct(c5["S"].get("TE/T0"), src_of(B75), p5 + ".S.TE/T0", "stack"),
                     pct(c9["S"]["TXN/TN"], src_of(B79), f"main.{d}|L5|S.all.S.TXN/TN", "stack")])
    head = [r" & rows / discharges & \multicolumn{3}{c}{RMSE (eV)} & TX vs T0 & TE vs T0 & TXN vs TN \\",
            r"\cmidrule(lr){3-5}",
            r"shots & / clusters & T0 & TX & O & (\%) & (\%) & (\%) \\"]
    body = tabular("@{}llrrrccc@{}", head, rows, "footnotesize", "3.5pt")
    cap = (r"Under the beam-modulation schedule S (target L5), by block of shots. Upper block: the random design "
           r"V1, scored rows grouped by the block of the discharge; lower block: the chronological design, which "
           r"scores the shots from 32312 on. TE: ECEI half only; TXN vs TN: the fast diagnostics after the beam "
           r"scalars (beam columns as delivered). Intervals: 95 \%, shot clusters resampled; bold: the interval excludes zero.")
    emit("Blocks", body, cap, "increment under S by block of shots (V1) and under V4 (both fits), with and without beam inputs",
         "b75 main.V1|L5|S.<block>.S, main.V4*|L5|S.all.S; b79 main.*|L5|S.*.S.TXN/TN", "4.6 (main)")


def parse_last_block():
    t = (PAPER / "b75_last_block.md").read_text(encoding="utf-8")
    code = t.split("```")[1]
    lines = code.splitlines()
    res = {}
    keymap = {"up to 31991": "block up to 31991", "32002-32310": "block 32002-32310", "from 32312": "block from 32312"}
    sec = None
    for ln in lines:
        m = re.match(r"\((\d)\)", ln.strip())
        if m:
            sec = int(m.group(1))
            continue
        s = ln.strip()
        for short, blk in keymap.items():
            if not s.startswith(short):
                continue
            d = res.setdefault(blk, {})
            rest = s[len(short):]
            if sec == 1 and "|" in rest:
                mm = re.search(r"(\d+)\s+(\d+)\s+(\d+)\s*\|\s*([\d.]+) \(([\d.]+)-([\d.]+)\)\s*\|\s*(\d+)\s*\|\s*([\d.]+) / ([\d.]+) / ([\d.]+)", rest)
                d.update(rows=mm.group(1), disch=mm.group(2), clus=mm.group(3), age=mm.group(4), l5=mm.group(7),
                         rmse_own=mm.group(8), rmse_hold_fresh=mm.group(9), rmse_hold_S=mm.group(10))
            elif sec == 2 and "|" in rest:
                f = [x.strip() for x in rest.split("|")]
                sp = f[2].split(",")
                d.update(sd_ln=f[1], sp_ecei=sp[0].strip(), sp_bes=sp[1].strip(), scatter=f[4])
            elif sec == 7:
                mm = re.search(r"notches\s+(\d+) \| T_i ([\d.]+) \(n (\d+)\) \| ECEI ([\d.]+) \(10-90 %: ([\d.]+)-([\d.]+)\) \| BES ([\d.]+)", rest)
                d.update(notches=mm.group(1), n_ti=mm.group(2), ti_n=mm.group(3), n_ecei=mm.group(4),
                         n_ecei_lo=mm.group(5), n_ecei_hi=mm.group(6), n_bes=mm.group(7))
            elif sec == 8:
                mm = re.search(r"discharges\s+(\d+) \| raw array mean, median ([\d.]+) .*? mean correlation between the four channels ([\d.]+)", rest)
                d.update(ch_disch=mm.group(1), raw=mm.group(2), corr=mm.group(3))
    return res


def mdtok(s, path, src="docs/paper/b75_last_block.md"):
    reg(s, src, path, "verbatim from the md (unregistered diagnostic, no interval)")
    return tex_sign(s)


def tab_block_descr():
    lb = parse_last_block()
    rows = []
    for blk in BLOCKS[:3]:
        d = lb[blk]
        pth = f"code block, {blk}"
        rows.append([bldate(blk), mdtok(d["age"], pth + ", (1) age median"), mdtok(d["l5"], pth + ", (1) L5 median"),
                     mdtok(d["sp_ecei"], pth + ", (2) Spearman ECEI"), mdtok(d["sp_bes"], pth + ", (2) Spearman BES"),
                     mdtok(d["n_ti"], pth + ", (7) T_i notch"), mdtok(d["n_bes"], pth + ", (7) BES notch"),
                     mdtok(d["n_ecei"], pth + ", (7) ECEI notch"), mdtok(d["corr"], pth + ", (8) channel correlation"),
                     mdtok(d["scatter"], pth + ", (2) relative scatter")])
    head = [r" & age of & L5 & \multicolumn{2}{c}{Spearman with the} & \multicolumn{3}{c}{first row after a notch} & ECEI & scatter \\",
            r"\cmidrule(lr){4-5}\cmidrule(lr){6-8}",
            r"shots & history (s) & median (eV) & ECEI change & BES change & $T_i$ & BES & ECEI & channels & of O \\"]
    body = tabular("@{}lrrrrrrrrr@{}", head, rows, "footnotesize", "3.5pt")
    cap = (r"Description of the three blocks of shots on the S set, read after the fact (unregistered "
           r"description; no estimator, no interval). Age: median age of the newest visible value under S. "
           r"Spearman: rank correlation of the change of the ECEI or BES level since the newest visible value with "
           r"the change of the $T_i$ level. Notch: value in the first row after an absent frame over the median of "
           r"the five rows before. ECEI channels: mean correlation between the four ECEI channels (median over "
           r"discharges). Scatter of O: median of $|O/L5 - 1|$.")
    emit("BlockDescr", body, cap, "last-block diagnostic: age, Spearman ECEI/BES, notch response T_i/BES/ECEI, channel coherence",
         "docs/paper/b75_last_block.md code block sections (1), (2), (7), (8) (parsed)", "4.6 (main)")
    return lb


def tab_gate():
    b = J(B80)
    s = src_of(B80)
    rows = []
    for rule in ("S", "D650"):
        cells = b["cells"][f"V1|{rule}"]
        for cl in ("class low", "class middle", "class high", "all"):
            c = cells[cl]
            p = f"cells.V1|{rule}.{cl}"
            cc = c["counts"]
            rows.append([rule if cl == "class low" else "", cl.replace("class ", ""),
                         cnt(cc["rows"], s, p + ".counts.rows") + " / " + cnt(cc["discharges"], s, p + ".counts.discharges"),
                         pct(c["TX/T0"], s, p + ".TX/T0", "stack"), pct(c["TE/T0"], s, p + ".TE/T0", "stack"),
                         pct(c["TB/T0"], s, p + ".TB/T0", "stack"), pct(c["TXN/TN"], s, p + ".TXN/TN", "stack"),
                         pct(c["TG/T0"], s, p + ".TG/T0", "stack")])
        c = cells["low+high"]
        p = f"cells.V1|{rule}.low+high"
        rows.append(["", "high $-$ low", "", pct(c["high minus low: TX/T0"], s, p + ".high minus low: TX/T0", "stack"),
                     pct(c["high minus low: TE/T0"], s, p + ".high minus low: TE/T0", "stack"),
                     pct(c["high minus low: TB/T0"], s, p + ".high minus low: TB/T0", "stack"),
                     pct(c["high minus low: TXN/TN"], s, p + ".high minus low: TXN/TN", "stack"), ""])
        if rule == "S":
            rows.append("MID")
    head = [r"rule & class of $C_E$ & rows / discharges & TX vs T0 & TE vs T0 & TB vs T0 & TXN vs TN & TG vs T0 \\"]
    body = tabular("@{}llrccccc@{}", head, rows, "footnotesize", "3.5pt")
    cl = b["classes"]
    for k in ("low", "high"):
        esc(cl[k], s, f"classes.{k}")
    cap = (r"Coherence of the ECEI channels per discharge, $C_E$ (mean correlation between the four channels, "
           r"from inputs only): low below 0.7, middle, high from 0.9. Random design V1, target L5, reduction of the "
           r"MSE in per cent; rows with the estimates stored by B.75 and B.79. TG uses the fast inputs only where "
           r"$C_E \ge 0.7$ (T0 elsewhere). ``High $-$ low'': difference of the reductions, in percentage points. "
           r"Intervals: 95 \%, shot clusters resampled; bold: the interval excludes zero.")
    emit("Gate", body, cap, "B.80: increment by class of ECEI channel coherence (S and D650, V1), gated estimator TG",
         "b80 cells.V1|{S,D650}.{class low,class middle,class high,all,low+high}", "4.6 (main or supplement)")


def tab_gate_blocks():
    b = J(B80)
    s = src_of(B80)
    rows = []
    for blk in ["block up to 31991", "block 32002-32310", "block from 32312", "all blocks"]:
        d = b["distribution"]["D1 discharges"][blk]
        p = f"distribution.D1 discharges.{blk}"
        pc = d["per class"]
        lab = "all shots" if blk == "all blocks" else bldate(blk)
        if blk == "all blocks":
            rows.append("MID")
        cl = blk.replace("all blocks", "all")
        last = b["cells"]["V1|S"]
        rows.append([lab, cnt(d["discharges"], s, p + ".discharges"), cnt(d["with C_E"], s, p + ".with C_E"),
                     cnt(pc["low"], s, p + ".per class.low"), cnt(pc["middle"], s, p + ".per class.middle"),
                     cnt(pc["high"], s, p + ".per class.high"),
                     fnum(d["share low among those with C_E"], "%.3f", s, p + ".share low among those with C_E", "share, three decimals")])
    head = [r"shots & discharges & with $C_E$ & low & middle & high & share low \\"]
    body = tabular("@{}lrrrrrr@{}", head, rows, "footnotesize", "5pt")
    cap = (r"Discharges of D1 by class of the ECEI channel coherence $C_E$ and block of shots (B.80; input only).")
    emit("GateBlocks", body, cap, "B.80 class counts by block of shots and share of the low class",
         "b80 distribution.D1 discharges.*", "4.6 / supplement")


def tab_correct():
    b = J(B77)
    s = src_of(B77)
    rows = []
    for d in DES:
        c = b["designs"][d]["all"]
        p = f"designs.{d}.all"
        rm = c["rmse_eV"]
        rows.append([design_label(d)] + [ev(rm[k], s, f"{p}.rmse_eV.{k}") for k in ("O", "MED3", "T0", "TX", "C0", "CX")]
                    + [pct(c["TX/T0"], s, p + ".TX/T0", "stack"), pct(c["CX/C0"], s, p + ".CX/C0", "stack")])
    head = [r" & \multicolumn{6}{c}{RMSE (eV)} & TX vs T0 & CX vs C0 \\", r"\cmidrule(lr){2-7}",
            r"design & O & MED3 & T0 & TX & C0 & CX & (\%) & (\%) \\"]
    body = tabular("@{}lrrrrrrcc@{}", head, rows, "footnotesize", "4pt")
    cc = b["row_set"]
    cap = (r"Correcting the frame's own reading (B.77), previous frame visible (D10), target L5. O: the reading; "
           r"MED3: median of the reading and the two values before; C0: causal GP and boosted regression on the "
           r"history and on the reading; CX: C0 with the fourteen fast scalars; T0, TX do not read the frame. "
           + cnt(cc["rows"], s, "row_set.rows") + " rows, " + cnt(cc["discharges"], s, "row_set.discharges")
           + " discharges, " + cnt(cc["clusters"], s, "row_set.clusters") + r" clusters (V1, V2); V3 and V4 score "
           r"their own subsets. Intervals: 95 \%, shot clusters resampled; bold: the interval excludes zero.")
    emit("Correct", body, cap, "B.77: correcting the frame's reading, RMSE and fast increments by design",
         "b77 designs.*.all", "4.6 (main) / supplement")


# ----------------------------------------------------------------------------------------------
# 8. previous frame visible (B.73)
# ----------------------------------------------------------------------------------------------
def tab_dten():
    b = J(B73)
    s = src_of(B73)
    rows = []
    for tgt in ("L5", "F"):
        rows.append(r"\multicolumn{7}{@{}l}{\textit{target " + tgt + r"}} \\")
        for d in DES:
            inc = b["increments"][f"{d}|{tgt}"]
            cl = inc["cells"]
            p = f"increments.{d}|{tgt}"
            rows.append([design_label(d), cnt(inc["rows"], s, p + ".rows") + " / " + cnt(inc["discharges"], s, p + ".discharges")]
                        + [pct(cl.get(k), s, f"{p}.cells.{k}", "stack") for k in ("TX/T0", "KX/K0", "TE/T0", "TB/T0", "TXlag/T0")])
        if tgt == "L5":
            rows.append("MID")
    head = [r"design & rows / discharges & TX vs T0 & KX vs K0 & TE vs T0 & TB vs T0 & TXlag vs T0 \\"]
    body = tabular("@{}lrccccc@{}", head, rows, "footnotesize", "3.5pt")
    cap = (r"Previous frame visible (D10; B.73): reduction of the MSE in per cent by the fast inputs, by design and "
           r"target, on all D1 rows where the target is defined. K0, KX: five-parameter filters without and with "
           r"fast inputs; TXlag: fast inputs one row old. Dashes: not computed for that design. Intervals: 95 \%, "
           r"shot clusters resampled; bold: the interval excludes zero.")
    emit("Dten", body, cap, "B.73: D10 increments by design and target (trees, filters, ECEI only, BES only, lagged)",
         "b73 increments.*|{L5,F}.cells", "4.6 (main: the D10 column) / supplement")


NET = [("H", "hold"), ("G", "causal GP"), ("M5", "median of last five"), ("K0[{t}]", "filter K0"),
       ("KX[{t}]", "filter KX"), ("T0[{t}]", "T0"), ("TX[{t}]", "TX"), ("TE[{t}]", "TE (ECEI only)"),
       ("TB[{t}]", "TB (BES only)"), ("TXlag[{t}]", "TXlag"), ("t", "network twin, as built"),
       ("c", "network, as built"), ("t2", "two-stage twin, as built"), ("c2", "two-stage estimator, as built"),
       ("cL[{t}]", "network, gain fitted"), ("c2L[{t}]", "two-stage, gain fitted")]


def tab_network():
    b = J(B73)
    s = src_of(B73)
    rows = []
    for key, lab in NET:
        row = [lab]
        for t in ("F", "L5"):
            pe = b["per_estimator"][f"V1|{t}"]
            k = key.format(t=t)
            c = pe.get(k)
            p = f"per_estimator.V1|{t}.{k}"
            row.append(ev(c["rmse_eV"], s, p + ".rmse_eV") if c else "--")
            row.append(pct(c.get("over_G"), s, p + ".over_G", "stack") if c and "over_G" in c else "--")
        rows.append(row)
    rows.append("MID")
    for key, lab in (("TX vs c2 gain-fitted", "TX vs two-stage, gain fitted"), ("TX vs c2 as built", "TX vs two-stage, as built"),
                     ("T0 vs t2 as built", "T0 vs two-stage twin, as built")):
        row = [lab]
        for t in ("F", "L5"):
            c = b["per_estimator"][f"V1|{t}"].get(key)
            row += ["", pct(c, s, f"per_estimator.V1|{t}.{key}", "stack") if c else "--"]
        rows.append(row)
    head = [r" & \multicolumn{2}{c}{target F} & \multicolumn{2}{c}{target L5} \\",
            r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
            r"estimator & RMSE (eV) & vs causal GP (\%) & RMSE (eV) & vs causal GP (\%) \\"]
    body = tabular("@{}lrcrc@{}", head, rows, "footnotesize", "5pt")
    cap = (r"Previous frame visible (D10; B.73), random design V1, all D1 rows with the target defined: every "
           r"estimator, including the recurrent network (two-branch LSTM) and the two-stage estimator, as built "
           r"(trained on the reading of the frame) and with gains fitted to the target. Reduction of the MSE against "
           r"the causal GP; last rows: TX and T0 against the network-based estimators. Points: mean over three "
           r"re-partitions. Intervals: 95 \%, shot clusters resampled; bold: the interval excludes zero.")
    emit("Network", body, cap, "B.73: every estimator incl. the recurrent network and two-stage estimator, targets F and L5",
         "b73 per_estimator.V1|{F,L5}", "supplement (previous frame visible)")


# ----------------------------------------------------------------------------------------------
# 9. statement tables
# ----------------------------------------------------------------------------------------------
def tab_statements(batch, fname, title):
    b = J(fname)
    s = src_of(fname)
    rows = []
    for k, v in b["statements"].items():
        p = f"statements.{k}"
        if fname == B74:
            txt = v["statistic"]
            parts = []
            if "point" in v:
                pt = fnum(v["point"], "%+.3f", s, p + ".point", "as registered, signed three decimals")
                lo = fnum(v["lo"], "%+.3f", s, p + ".lo", "as registered (interval)")
                hi = fnum(v["hi"], "%+.3f", s, p + ".hi", "as registered (interval)")
                parts.append(f"{pt} [{lo}, {hi}]")
            for kk in ("diff_minus10", "diff_plus10"):
                if kk in v:
                    c = v[kk]
                    parts.append(esc(kk, s, p) + ": " + fnum(c["point"], "%+.3f", s, f"{p}.{kk}.point", "as registered")
                                 + " [" + fnum(c["lo"], "%+.3f", s, f"{p}.{kk}.lo", "as registered") + ", "
                                 + fnum(c["hi"], "%+.3f", s, f"{p}.{kk}.hi", "as registered") + "]")
            val = "; ".join(parts)
            exp = v.get("expectation", "")
        else:
            txt = v["text"]
            val = stat_val(v["value"], s, p + ".value")
            exp = v.get("expected", "")
        verdict = "holds" if v["holds"] else "does not hold"
        rows.append([esc(k, s, p), r"\parbox[t]{6.6cm}{\raggedright " + esc(txt, s, p + ".text") + r"\strut}",
                     r"\parbox[t]{3.7cm}{\raggedright " + val + r"\strut}", verdict,
                     r"\parbox[t]{2.2cm}{\raggedright " + esc(exp, s, p + ".expected") + r"\strut}"])
    head = [r"id & statement & value [95 \% interval] & verdict & expected \\"]
    body = tabular("@{}lllll@{}", head, rows, "scriptsize", "3pt")
    iv = "sessions" if fname == B76 else ("discharges, as registered" if fname == B74 else "shot clusters")
    cap = (title + r": the registered statements with the value as registered (skills as fractions, signed), the "
           r"95 \% interval (" + iv + r" resampled), the verdict by the registered rule and the registered expectation.")
    nm = "Stat" + batch
    emit(nm, body, cap, f"registered statements of {title}", f"{fname} statements.*", "supplement (statement tables)")


# ----------------------------------------------------------------------------------------------
# 10. computation time, NBI tables, estimator inputs
# ----------------------------------------------------------------------------------------------
def tab_timing():
    b = J(B75)
    s = src_of(B75)
    t = b["timing"]
    rows = [[esc(t["rule"], s, "timing.rule"), cnt(t["rows"], s, "timing.rows"),
             fnum(t["median_ms"], "%.2f", s, "timing.median_ms", "ms, two decimals"),
             fnum(t["p99_ms"], "%.2f", s, "timing.p99_ms", "ms, two decimals"),
             fnum(t["mean_ms"], "%.2f", s, "timing.mean_ms", "ms, two decimals"),
             fnum(t["max_ms"], "%.2f", s, "timing.max_ms", "ms, two decimals"), str(t["threads"])]]
    head = [r"rule & rows & median (ms) & 99th percentile (ms) & mean (ms) & largest (ms) & threads \\"]
    body = tabular("@{}lrrrrrr@{}", head, rows, "footnotesize", "5pt")
    cap = (r"Computation time per row of the causal GP and the TX prediction together (one unit of V1, rule D650), "
           r"one CPU thread, one row at a time, on a shared machine with other jobs running. Only the estimator's "
           r"computation is timed; the analysis of the CES frames is not.")
    emit("Timing", body, cap, "computation time per row (GP + TX)", "b75 timing.*", "4.1 / sec. 5")


def md_table(text, header_start):
    """Rows of the first markdown table whose header line starts with header_start."""
    lines = text.splitlines()
    for i, ln in enumerate(lines):
        if ln.startswith("| " + header_start):
            out = []
            for l2 in lines[i + 2:]:
                if not l2.startswith("|"):
                    break
                out.append([c.strip() for c in l2.strip().strip("|").split("|")])
            return out
    raise KeyError(header_start)


def tab_nbi():
    p = PAPER / "nbi_files.md"
    t = p.read_text(encoding="utf-8")
    s = "docs/paper/nbi_files.md"
    src_rows = md_table(t, "source | rows on")
    rows = []
    for r in src_rows:
        rows.append([esc(r[0], s, "sec. 2 table"), esc(r[1], s, f"sec. 2 table, {r[0]}, rows on"),
                     esc(r[2], s, f"sec. 2 table, {r[0]}, shots"), esc(r[3], s, f"sec. 2 table, {r[0]}, voltage")])
    head = [r"source & rows on (share) & shots with it on & voltage while on, kV (10th to 90th percentile) \\"]
    body = tabular("@{}llll@{}", head, rows, "footnotesize", "5pt")
    cap = (r"NBI sources in the beam files of the 641 shot files: a source counts as on in a row when its power "
           r"is above 0.1~MW (file facts, no estimator).")
    reg("641", s, "sec. 1", "verbatim from the md")
    reg("0.1", s, "header", "verbatim from the md")
    emit("NbiSources", body, cap, "NBI sources: rows on, shots, voltage", "docs/paper/nbi_files.md sec. 2 (parsed)",
         "supplement (sec. 2 / 4.2)")
    n_rows = md_table(t, "source | steps")
    rows = []
    for r in n_rows:
        rows.append([esc(r[0], s, "sec. 3 table")] + [esc(x, s, f"sec. 3 table, {r[0]}") for x in r[1:]])
    head = [r"source & absent steps & last row before & first row after & share of first rows below 0.9 \\"]
    body = tabular("@{}lllll@{}", head, rows, "footnotesize", "5pt")
    cap = (r"An absent frame is a notch of the NBI-1 sources: power of a source in the last row before and the "
           r"first row after an absent step (a 20--40~ms step inside a discharge), over its median in rows 5 to 10 "
           r"after it; median (10th percentile). Steps followed by ten 10~ms rows with the source above 0.3~MW.")
    reg("0.3", s, "sec. 3 text", "verbatim from the md")
    emit("NbiNotch", body, cap, "power of each NBI source before/after an absent frame", "docs/paper/nbi_files.md sec. 3 (parsed)",
         "sec. 2 (main or supplement)")
    st_rows = md_table(t, "state | rows")
    rows = [[esc(r[0], s, "sec. 4 table"), esc(r[1], s, f"sec. 4 table, {r[0]}, rows"),
             esc(r[2], s, f"sec. 4 table, {r[0]}, share T_i")] for r in st_rows]
    head = [r"beam state & rows & share with a $T_i$ value \\"]
    body = tabular("@{}lrr@{}", head, rows, "footnotesize", "6pt")
    cap = r"Is a CES $T_i$ value written, by the state of the NBI sources (file facts)."
    emit("NbiStates", body, cap, "share of rows with a T_i value by beam state", "docs/paper/nbi_files.md sec. 4 (parsed)",
         "supplement (sec. 2)")


def tab_inputs():
    b9 = J(B79)
    s = src_of(B79)
    bs = b9["beam_scalars"]
    nbeam = str(len(bs))
    reg(nbeam, s, "beam_scalars (list length)", "count")
    # the groups named in the cell below, checked against the list (9 + 4 + 5 + 1)
    assert sum(1 for x in bs if x.startswith(("P1(", "P2(", "Pt("))) == 9
    assert sum(1 for x in bs if x.startswith("mean")) == 4
    assert sum(1 for x in bs if x.startswith("on ")) == 5 and sum(1 for x in bs if x.startswith("V ")) == 1
    rows = [
        ["H (hold)", "newest visible CES value", "baseline"],
        ["G (causal GP)", r"last 16 visible values; Mat\'ern-3/2, hyperparameters by marginal likelihood per sample", "CES-only baseline"],
        ["T0", r"G + boosted regression on 10 history scalars (time step, age of the newest value, H, $G-H$, $|G-H|$, mean, sd, $H$ $-$ mean, last increment and number of the last ten values)", "history only"],
        ["TX", r"T0 inputs + 8 fast scalars (level of the BES and ECEI array means, departure from the mean of the ten rows before, its sd, one-row increment) + 6 anchor scalars (fast level at the anchor row, change since, log ratio of raw levels)", "the estimator"],
        ["TE, TB", "T0 inputs + the ECEI half or the BES half", "which diagnostic"],
        ["TN, TXN", "T0 or TX inputs + " + nbeam + " beam scalars: power of NBI-1, of NBI-2 and their total at row $t$, "
         "at the anchor row and the difference (9); running means of the total over 10 and 50 rows at row $t$ and "
         "at the anchor row (4); on/off flags of the five sources that are ever on (5); voltage of source NB1B (1)",
         "comparison rows"],
        ["O", "the frame's own reading (scored against L5 only)", "reference"],
    ]
    reg("16", "PAPER_DESIGN.md", "sec. 6 table", "definition")
    rows = [[r[0], r"\parbox[t]{10.2cm}{\raggedright " + r[1] + r"\strut}", r[2]] for r in rows]
    head = [r"name & reads & role \\"]
    body = tabular("@{}lll@{}", head, rows, "footnotesize", "4pt")
    cap = (r"Estimators and their inputs (KSTAR). Fast samples and beam waveforms are visible up to row $t$ under "
           r"every rule; CES values only as the availability rule allows. Boosted regression: "
           r"HistGradientBoostingRegressor, default settings, one per rule and target.")
    emit("Inputs", body, cap, "estimators and their inputs (definitions from PAPER_DESIGN.md sec. 6; beam scalar names from b79)",
         "PAPER_DESIGN.md sec. 6 (text); b79 beam_scalars", "sec. 3 (main)")


# ----------------------------------------------------------------------------------------------
# 10b. round 2: as-fitted estimators, anchor scalars, bursts, constant ECEI, against the reading
# ----------------------------------------------------------------------------------------------
def rule_cell(b, d, r):
    if r == "S":
        return b["main"][f"{d}|L5|S"]["all"]["S"], f"main.{d}|L5|S.all.S"
    return b["main"][f"{d}|L5|common"]["all"][r], f"main.{d}|L5|common.all.{r}"


def tab_as_fitted(designs=("V1", "V2", "V3"), name="AsFitted"):
    b = J(B75)
    s = src_of(B75)
    rows = []
    for d in designs:
        rows.append(r"\multicolumn{9}{@{}l}{\textit{" + design_label(d) + r"}} \\")
        for r in ["D50", "D100", "D300", "D650", "D1000", "S"]:
            c, p = rule_cell(b, d, r)
            if "T0a" not in c["rmse_eV"]:
                continue
            rm = c["rmse_eV"]
            rows.append([rule_tok(r, s, p), ev(rm["T0a"], s, p + ".rmse_eV.T0a"), ev(rm["TXa"], s, p + ".rmse_eV.TXa"),
                         ev(rm["T0"], s, p + ".rmse_eV.T0"), ev(rm["TX"], s, p + ".rmse_eV.TX"),
                         pct(c["applied T0: refit over applied"], s, p + ".applied T0: refit over applied", "stack"),
                         pct(c["applied TX: refit over applied"], s, p + ".applied TX: refit over applied", "stack"),
                         pct(c["applied TXa/T0a"], s, p + ".applied TXa/T0a", "stack"),
                         pct(c["TX/T0"], s, p + ".TX/T0", "stack")])
        if d != designs[-1]:
            rows.append("MID")
    head = [r" & \multicolumn{2}{c}{applied (fitted at D10)} & \multicolumn{2}{c}{refitted under the rule} & "
            r"\multicolumn{2}{c}{refitted vs applied (\%)} & \multicolumn{2}{c}{fast-channel increment (\%)} \\",
            r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}",
            r"rule & T0 & TX & T0 & TX & T0 & TX & applied & refitted \\"]
    body = tabular("@{}lrrrrcccc@{}", head, rows, "footnotesize", "3.2pt")
    cap = (r"Estimators fitted with the previous frame visible and then given the stale history (``applied'') "
           r"against the same estimators fitted under the rule (``refitted''), target L5: RMSE in eV, the "
           r"reduction of the MSE by refitting, and the fast-channel increment (TX against T0) of each pair. "
           r"Fixed delays on the common set, S on the S set. Intervals: 95 \%, shot clusters resampled; bold: the "
           r"interval excludes zero.")
    if name != "AsFitted":
        cap = r"As table \ref{tab:asfitted} for the chronological design V4 (fit to 31991 and to 32310), target L5."
        emit(name, body, cap, "the same for V4a, V4b", "b75 main.V4*.*", "supplement")
        return
    emit("AsFitted", body, cap, "estimators fitted at D10 and applied to a stale history vs refitted under the rule",
         "b75 main.*.{D50..D1000,S}.{rmse_eV.T0a/TXa, applied T0/TX: refit over applied, applied TXa/T0a}",
         "sec. 3 (method sentence) / supplement")
    c, p = rule_cell(b, "V1", "D650")
    pct_macro("RefitOverAppliedTXDsixfiftyVone", c["applied TX: refit over applied"], s,
              p + ".applied TX: refit over applied", "refitted TX vs applied TX, D650, V1", "sec. 3")
    pct_macro("RefitOverAppliedTZeroDsixfiftyVone", c["applied T0: refit over applied"], s,
              p + ".applied T0: refit over applied", "refitted T0 vs applied T0, D650, V1", "sec. 3")
    for e, nm in (("T0a", "RmseAppliedTZeroDsixfiftyVone"), ("TXa", "RmseAppliedTXDsixfiftyVone"),
                  ("T0", "RmseRefitTZeroDsixfiftyVone"), ("TX", "RmseRefitTXDsixfiftyVone")):
        macro(nm, ev(c["rmse_eV"][e], s, p + ".rmse_eV." + e), f"RMSE (eV) of {e}, D650, V1 (applied = fitted at D10)",
              s + " " + p, "sec. 3")


def tab_anchor_scalars():
    b = J(B75)
    s = src_of(B75)
    rows = []
    for d in DES:
        first = True
        for r in ("D10", "D650", "S"):
            c, p = rule_cell(b, d, r)
            if "TX8/T0" not in c:
                continue
            rm = c["rmse_eV"]
            # TX over TX8 is not stored; it is not computed here (no new analysis)
            rows.append([design_label(d) if first else "", r, ev(rm["T0"], s, p + ".rmse_eV.T0"),
                         ev(rm["TX8"], s, p + ".rmse_eV.TX8"), ev(rm["TX"], s, p + ".rmse_eV.TX"),
                         pct(c["TX8/T0"], s, p + ".TX8/T0", "stack"), pct(c["TX/T0"], s, p + ".TX/T0", "stack")])
            first = False
        if first is False and d != "V4b":
            rows.append("MID")
    head = [r" & & \multicolumn{3}{c}{RMSE (eV)} & TX8 vs T0 & TX vs T0 \\", r"\cmidrule(lr){3-5}",
            r"design & rule & T0 & TX8 & TX & (\%) & (\%) \\"]
    body = tabular("@{}llrrrcc@{}", head, rows, "footnotesize", "5pt")
    cap = (r"The six anchor scalars (fast level at the row of the newest visible CES value, its change since, "
           r"the log ratio of the raw levels), target L5: TX8 is TX without them. Reduction of the MSE against "
           r"T0 in per cent. Computed for V1 and V4 only. D10 and D650 on the common set, S on the S set. "
           r"Intervals: 95 \%, shot clusters resampled; bold: the interval excludes zero.")
    emit("AnchorScalars", body, cap, "TX8 (no anchor scalars) and TX against T0, V1 and V4, D10/D650/S",
         "b75 main.{V1,V4a,V4b}.{D10,D650,S}.{TX8/T0,TX/T0,rmse_eV}", "4.2 / supplement")


def tab_bursts():
    b = J(B74)
    s = src_of(B74)
    A, B, E = b["A"]["W10"], b["B"]["W10"], b["E"]["W10"]

    def c3(cell, path, fmt="%.3f"):
        if cell is None:
            return "--"
        pt = fnum(cell["point"], fmt, s, path + ".point", "as registered, three decimals")
        lo = fnum(cell["lo"], fmt, s, path + ".lo", "as registered (discharges resampled)")
        hi = fnum(cell["hi"], fmt, s, path + ".hi", "as registered (discharges resampled)")
        return r"\ci{" + pt + "}{[" + lo + ", " + hi + "]}"

    rows = [["rows", cnt(A["B"]["n"], s, "A.W10.B.n"), cnt(A["P"]["n"], s, "A.W10.P.n"), cnt(A["Q"]["n"], s, "A.W10.Q.n"), ""],
            ["share without a usable $T_i$", c3(A["B"], "A.W10.B"), c3(A["P"], "A.W10.P"), c3(A["Q"], "A.W10.Q"),
             c3(A["B_minus_Q"], "A.W10.B_minus_Q", "%+.3f")]]
    for k, lab in (("ti_ratio", r"median $T_i$ ratio"), ("l5_ratio", "median ratio of the following level"),
                   ("vt_ratio", "median ratio of the rotation"), ("ec_ratio", "median ECEI ratio")):
        q = B[k]
        rows.append([lab, c3(q["B"], f"B.W10.{k}.B"), c3(q["P"], f"B.W10.{k}.P"), c3(q["Q"], f"B.W10.{k}.Q"),
                     c3(q["B_minus_Q"], f"B.W10.{k}.B_minus_Q", "%+.3f")])
    rows.append("MID")
    up = E["up"]
    rows.append([r"\multicolumn{5}{@{}l}{\textit{one-frame upward excursions (" + cnt(up["departures"], s, "E.W10.up.departures")
                 + " excursions, " + cnt(up["departures_in_B"], s, "E.W10.up.departures_in_B")
                 + r" with a burst inside the exposure)}} \\"])
    rows[-1] = rows[-1][0]
    rows.append(["share of burst rows among the excursions", c3(up["share_B"], "E.W10.up.share_B"), "", "", ""])
    rows.append(["share of burst rows among all rows with a value", c3(E["share_B_all_rows"], "E.W10.share_B_all_rows"), "", "", ""])
    rows.append(["ratio of the two shares", c3(up["share_ratio"], "E.W10.up.share_ratio", "%.2f"), "", "", ""])
    head = [r" & burst inside & burst just & no burst & inside minus \\",
            r"quantity & the exposure (B) & before (P) & (Q) & none (B $-$ Q) \\"]
    body = tabular("@{}lcccc@{}", head, rows, "footnotesize", "4pt")
    nd = len(b["discharges_in_statements"])
    reg(str(nd), s, "discharges_in_statements (length)", "count")
    cap = (r"Edge bursts inside the exposure (B.74; raw BES stream of the " + str(nd) + r" discharges with a raw "
           r"delivery, no estimator). A 10~ms row is in class B when a burst onset falls in the window from 2~ms "
           r"before to 8~ms after its stamp, P when none does but one falls in the 10~ms before, Q when none falls "
           r"inside or in the 30~ms before. Ratio: value of the row over the median of the ten preceding regular "
           r"rows. Intervals: 95 \%, the discharges resampled (as registered).")
    emit("Bursts", body, cap, "B.74 in readable form: rows with/without a burst inside the exposure, ratios, excursions",
         "b74 A.W10, B.W10.{ti,l5,vt,ec}_ratio, E.W10", "supplement (sec. 2)")


def parse_stuck():
    t = (PAPER / "b75_stuck_ecei.md").read_text(encoding="utf-8")
    code = t.split("```")[1]
    out = {"blocks": {}, "inc": {}}
    for ln in code.splitlines():
        m = re.match(r"\s+(up to 31991|32002-32310|from 32312|all)\s+(\d+) \| ([\d.]+), (\d+) of (\d+), (\d+) \| ([\d.]+)", ln)
        if m:
            out["blocks"][m.group(1)] = dict(rows=m.group(2), share=m.group(3), dis_any=m.group(4), dis=m.group(5),
                                             dis_half=m.group(6))
            continue
        m = re.match(r"\s+rule (S|D650)\s+(all blocks|up to 31991|32002-32310|from 32312)\s+\| ECEI moving: (\d+) rows, (\d+) discharges, ([+-][\d.]+) \| ECEI constant: (\d+) rows(?:, (\d+) discharges, ([+-][\d.]+))?", ln)
        if m:
            out["inc"][(m.group(1), m.group(2))] = dict(mv_rows=m.group(3), mv_dis=m.group(4), mv=m.group(5),
                                                        c_rows=m.group(6), c_dis=m.group(7), c=m.group(8))
    return out


def tab_stuck():
    st = parse_stuck()
    src = "docs/paper/b75_stuck_ecei.md"

    def t(v, path):
        if v is None:
            return "--"
        reg(v, src, path, "verbatim from the md (unregistered description, no interval)")
        return tex_sign(v)

    def pc(v, path):
        if v is None:
            return "--"
        return fnum(float(v), "%.1f", src, path, "100 x skill, one decimal (parsed from md, no interval)", 100)

    rows = []
    for blk in ("up to 31991", "32002-32310", "from 32312", "all"):
        d = st["blocks"][blk]
        k = "all blocks" if blk == "all" else blk
        p = f"code block, {blk}"
        if blk == "all":
            rows.append("MID")
        lab = "all shots" if blk == "all" else blk.replace("-", "--")
        for sh in re.findall(r"\d{5}", blk):
            reg(sh, src, p, "shot number from a block name")
        r = [lab, cnt(int(d["rows"]), src, p + ", grid rows", "count parsed from the md"), t(d["share"], p + ", share constant"),
             t(d["dis_any"], p + ", discharges with any") + " of " + t(d["dis"], p + ", discharges"),
             t(d["dis_half"], p + ", more than half")]
        for rule in ("S", "D650"):
            q = st["inc"][(rule, k)]
            r += [pc(q["mv"], f"{p}, rule {rule}, ECEI moving"), pc(q["c"], f"{p}, rule {rule}, ECEI constant")]
        rows.append(r)
    head = [r" & grid & share with & discharges & more than & \multicolumn{2}{c}{S: TX vs T0 (\%)} & \multicolumn{2}{c}{D650: TX vs T0 (\%)} \\",
            r"\cmidrule(lr){6-7}\cmidrule(lr){8-9}",
            r"shots & rows & ECEI constant & with any & half & moving & constant & moving & constant \\"]
    body = tabular("@{}lrrrrrrrr@{}", head, rows, "footnotesize", "3.5pt")
    cap = (r"Rows in which the four ECEI channels are constant (unchanged for at least ten rows, 100~ms), by block "
           r"of shots, and the fast-channel increment of the random design V1 (target L5) on rows where the ECEI "
           r"moves and where it is constant at row $t$ or at the anchor row. These rows were kept in every fit and "
           r"score. Unregistered description from the stored estimates; no interval. Dash: no value in the "
           r"source.")
    emit("StuckEcei", body, cap, "constant-ECEI rows by block and the increment on moving/constant rows (unregistered)",
         "docs/paper/b75_stuck_ecei.md code block (1), (3) (parsed)", "sec. 2 / supplement")
    a = st["blocks"]["all"]
    macro("StuckShare", t(a["share"], "code block (1), all, share"), "share of grid rows with a constant ECEI", src, "sec. 2")
    macro("NStuckDischarges", t(a["dis_any"], "code block (1), all, discharges with any"),
          "discharges with any constant-ECEI row", src, "sec. 2")
    macro("NStuckDischargesHalf", t(a["dis_half"], "code block (1), all, more than half"),
          "discharges with a constant ECEI in more than half of the rows", src, "sec. 2")


def tab_vs_reading():
    b = J(B75)
    s = src_of(B75)
    rows = []
    for d in DES:
        r_ = [design_label(d)]
        for r in ("D650", "S"):
            c, p = rule_cell(b, d, r)
            rm = c["rmse_eV"]
            r_ += [ev(rm["TX"], s, p + ".rmse_eV.TX"), ev(rm["O"], s, p + ".rmse_eV.O"), pct(c["TX/O"], s, p + ".TX/O", "stack")]
            if r == "S":
                pct_macro("TXvsOS" + DNAME[d], c["TX/O"], s, p + ".TX/O", f"TX against the frame's own reading under S, {d}", "4.1/4.6")
        rows.append(r_)
    head = [r" & \multicolumn{3}{c}{D650 (common set)} & \multicolumn{3}{c}{S (S set)} \\",
            r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
            r"design & TX (eV) & O (eV) & TX vs O (\%) & TX (eV) & O (eV) & TX vs O (\%) \\"]
    body = tabular("@{}lrrcrrc@{}", head, rows, "footnotesize", "4.5pt")
    cap = (r"The estimator against the frame's own reading O, target L5: RMSE and the reduction of the MSE of TX "
           r"against O (negative: TX is farther from the following level than O). O exists only once the frame "
           r"has been analysed; under S it is the value that the analysis has not yet delivered. Intervals: 95 \%, "
           r"shot clusters resampled; bold: the interval excludes zero.")
    emit("VsReading", body, cap, "TX against the frame's own reading O, D650 and S, per design",
         "b75 main.*.{D650,S}.{TX/O, rmse_eV.TX/O}", "4.1 / 4.6")


# ----------------------------------------------------------------------------------------------
# 10c. compact statement tables (summary; the statements that do not hold)
# ----------------------------------------------------------------------------------------------
STAT_BATCHES = [("B.73", B73, "previous frame visible: targets, designs, network rows"),
                ("B.74", B74, "edge bursts inside the exposure"),
                ("B.75", B75, "stale CES history: delays and schedule"),
                ("B.76", B76, "MAST: stale CXRS history, present Thomson"),
                ("B.77", B77, "correcting the frame's own reading"),
                ("B.78", B78, "a CES value every k-th frame"),
                ("B.79", B79, "beam waveforms as inputs"),
                ("B.80", B80, "coherence of the ECEI channels"),
                ("B.81", B81, "controls; ECEI and BES halves under V2, V3")]


def expectation(v):
    return str(v.get("expected", v.get("expectation", "")))


def definite(e):
    """A registered expectation is definite when it states a verdict ('holds ...' or 'does not hold ...');
    'uncertain ...' is not definite. A conditional one ('holds if J2 holds') counts as definite."""
    e = e.strip().lower()
    return e.startswith("holds") or e.startswith("does not hold")


def met(e, holds):
    return e.strip().lower().startswith("does not hold") == (not holds)


def tab_stat_summary():
    rows = []
    tot = [0, 0, 0, 0, 0]
    for lab, f, what in STAT_BATCHES:
        st = J(f)["statements"]
        s_ = src_of(f)
        n = len(st)
        h = sum(1 for v in st.values() if v["holds"])
        d = [v for v in st.values() if definite(expectation(v))]
        m = sum(1 for v in d if met(expectation(v), v["holds"]))
        vals = [n, h, n - h, len(d), m]
        tot = [a + b for a, b in zip(tot, vals)]
        rows.append([lab, esc(what, "make_tables_v2.py", "STAT_BATCHES text")]
                    + [cnt(x, s_, "statements (counted)", "count of statements") if x >= 100 else str(x) for x in vals])
    rows.append("MID")
    rows.append(["all", ""] + [str(x) for x in tot])
    head = [r" & & regis- & & do not & definite & expectation \\",
            r"batch & question & tered & hold & hold & expectation & met \\"]
    body = tabular("@{}llrrrrr@{}", head, rows, "footnotesize", "4.5pt")
    cap = (r"The registered statements of batches B.73 to B.81, counted from the result files: how many were "
           r"registered, how many hold and do not hold by the registered rule, how many had a definite registered "
           r"expectation (a stated verdict, not ``uncertain'') and how many of those the outcome met. The statements "
           r"that do not hold are listed in table \ref{tab:statnotheld}; the complete lists, with the statements that "
           r"hold, are in the result files \texttt{docs/paper/b7x\_*.md} and \texttt{b8x\_*.md} of the repository.")
    emit("StatSummary", body, cap, "count of registered statements per batch B.73-B.81: hold, do not hold, expectations met",
         "statements blocks of b73..b81 JSON (counted)", "supplement (replaces the nine statement tables)")
    return tot


def stat_rows(batches):
    rows = []
    for lab, f, what in batches:
        st = J(f)["statements"]
        s_ = src_of(f)
        first = True
        for k, v in st.items():
            if v["holds"]:
                continue
            p = f"statements.{k}"
            if f == B74:
                txt = v["statistic"]
                parts = []
                if "point" in v:
                    parts.append(fnum(v["point"], "%+.3f", s_, p + ".point", "as registered, signed three decimals")
                                 + " [" + fnum(v["lo"], "%+.3f", s_, p + ".lo", "as registered (interval)") + ", "
                                 + fnum(v["hi"], "%+.3f", s_, p + ".hi", "as registered (interval)") + "]")
                for kk in ("diff_minus10", "diff_plus10"):
                    if kk in v:
                        c = v[kk]
                        parts.append(esc(kk.replace("_", " "), s_, p) + ": "
                                     + fnum(c["point"], "%+.3f", s_, f"{p}.{kk}.point", "as registered") + " ["
                                     + fnum(c["lo"], "%+.3f", s_, f"{p}.{kk}.lo", "as registered") + ", "
                                     + fnum(c["hi"], "%+.3f", s_, f"{p}.{kk}.hi", "as registered") + "]")
                val = "; ".join(parts)
            else:
                txt = v["text"]
                val = stat_val(v["value"], s_, p + ".value")
            rows.append([lab if first else "", esc(k, s_, p),
                         r"\parbox[t]{7.0cm}{\raggedright " + esc(txt, s_, p + ".text") + r"\strut}",
                         r"\parbox[t]{3.6cm}{\raggedright " + val + r"\strut}",
                         r"\parbox[t]{2.3cm}{\raggedright " + esc(expectation(v), s_, p + ".expected") + r"\strut}"])
            first = False
    return rows


def tab_stat_not_held():
    head = [r"batch & id & statement (as registered) & value [95 \% interval] & expected \\"]
    parts = [("StatNotHeld", STAT_BATCHES[:5]), ("StatNotHeldB", STAT_BATCHES[5:])]
    n = []
    for name, batches in parts:
        rows = stat_rows(batches)
        n.append(len(rows))
        body = tabular("@{}lllll@{}", head, rows, "footnotesize", "3pt")
        if name == "StatNotHeld":
            cap = (r"The registered statements of B.73 to B.81 that do not hold (" + batches[0][0] + " to "
                   + batches[-1][0] + r"). Values as registered: skills are signed fractions, not per cent; 95 \% "
                   r"intervals with shot clusters resampled (B.74: discharges; B.76: sessions). The complete lists, "
                   r"with the statements that hold, are in the result files \texttt{docs/paper/b7x\_*.md} and "
                   r"\texttt{b8x\_*.md} of the repository.")
        else:
            cap = (r"Continuation of table \ref{tab:statnotheld} (" + batches[0][0] + " to " + batches[-1][0] + ").")
        emit(name, body, cap, "registered statements that do not hold (" + batches[0][0] + "-" + batches[-1][0] + ")",
             "statements blocks of the batch JSON (holds = false)", "supplement")
    return n


# ----------------------------------------------------------------------------------------------
# 11. prose macros
# ----------------------------------------------------------------------------------------------
DNAME = {"V1": "Vone", "V2": "Vtwo", "V3": "Vthree", "V4a": "Vfoura", "V4b": "Vfourb",
         "M1": "Mone", "M2": "Mtwo", "M3": "Mthree", "M4": "Mfour"}
RNAME = {"D10": "Dten", "D650": "Dsixfifty", "S": "S", "D5": "Dfive", "D100": "Dhundred", "K10": "Kten"}


def pct_macro(name, cell, src, path, what, where):
    p = pct_parts(cell, src, path, bold=False)
    macro(name, p[0] + "~" + p[1], what + " (per cent, interval)", f"{src} {path}", where)
    macro(name + "Pt", p[0], what + " (per cent, point)", f"{src} {path}", where)


def rat_macro(name, cell, src, path, what, where, nd=3):
    p = rat_parts(cell, src, path, nd)
    macro(name, p[0] + "~" + p[1], what + " (with interval)", f"{src} {path}", where)
    macro(name + "Pt", p[0], what + " (point)", f"{src} {path}", where)


def macros():
    b5, b6, b8, b9, b0 = J(B75), J(B76), J(B78), J(B79), J(B80)
    s5, s6, s8, s9 = src_of(B75), src_of(B76), src_of(B78), src_of(B79)
    for d in DES:
        m = b5["main"][f"{d}|L5|common"]["all"]
        for r in ("D10", "D650"):
            pct_macro("Inc" + RNAME[r] + DNAME[d], m[r]["TX/T0"], s5, f"main.{d}|L5|common.all.{r}.TX/T0",
                      f"fast-channel increment, {r}, {d}, L5", "4.1/4.3/abstract")
        pct_macro("IncS" + DNAME[d], b5["main"][f"{d}|L5|S"]["all"]["S"]["TX/T0"], s5, f"main.{d}|L5|S.all.S.TX/T0",
                  f"fast-channel increment, S, {d}, L5", "4.1/4.6")
        for r in ("D650", "S"):
            c9 = b9["main"][f"{d}|L5|common"]["all"]["D650"] if r == "D650" else b9["main"][f"{d}|L5|S"]["all"]["S"]
            k9 = f"main.{d}|L5|common.all.D650" if r == "D650" else f"main.{d}|L5|S.all.S"
            pct_macro("IncBeam" + RNAME[r] + DNAME[d], c9["TXN/TN"], s9, k9 + ".TXN/TN",
                      f"fast diagnostics after the beam scalars (TXN vs TN), {r}, {d}", "4.2/abstract")
            pct_macro("BeamAlone" + RNAME[r] + DNAME[d], c9["TN/T0"], s9, k9 + ".TN/T0",
                      f"beam scalars alone (TN vs T0), {r}, {d}", "4.2")
        pct_macro("IncKten" + DNAME[d], b8["main"][f"{d}|L5"]["all"]["K10"]["TX/T0"], s8, f"main.{d}|L5.all.K10.TX/T0",
                  f"fast-channel increment, K10, {d}", "4.5")
    for d in ("V1", "V4a"):
        for r in ("D650", "S"):
            c5 = b5["main"][f"{d}|L5|common"]["all"]["D650"] if r == "D650" else b5["main"][f"{d}|L5|S"]["all"]["S"]
            k5 = f"main.{d}|L5|common.all.D650" if r == "D650" else f"main.{d}|L5|S.all.S"
            pct_macro("EceiOnly" + RNAME[r] + DNAME[d], c5["TE/T0"], s5, k5 + ".TE/T0", f"ECEI only (TE vs T0), {r}, {d}", "4.2/abstract")
            pct_macro("BesOnly" + RNAME[r] + DNAME[d], c5["TB/T0"], s5, k5 + ".TB/T0", f"BES only (TB vs T0), {r}, {d}", "4.2")
    b1 = J(B81)
    for d in DES:
        for r in ("D650", "S"):
            c = b1["main"][d][r]
            k = f"main.{d}.{r}"
            pct_macro("CtrlPerm" + RNAME[r] + DNAME[d], c["TXp/T0"], src_of(B81), k + ".TXp/T0",
                      f"control: fast inputs permuted (TXp vs T0), {r}, {d}", "4.2")
            pct_macro("CtrlAnchor" + RNAME[r] + DNAME[d], c["TXo/T0"], src_of(B81), k + ".TXo/T0",
                      f"control: fast inputs at the anchor row (TXo vs T0), {r}, {d}", "4.2")
            pct_macro("PresentOverAnchor" + RNAME[r] + DNAME[d], c["TX/TXo"], src_of(B81), k + ".TX/TXo",
                      f"present fast inputs over anchor-row inputs (TX vs TXo), {r}, {d}", "4.2")
    for d in ("V2", "V3"):
        for r in ("D650", "S"):
            c = b1["main"][d][r]
            k = f"main.{d}.{r}"
            pct_macro("EceiOnly" + RNAME[r] + DNAME[d], c["TE/T0"], src_of(B81), k + ".TE/T0", f"ECEI only (TE vs T0), {r}, {d}", "4.2")
            pct_macro("BesOnly" + RNAME[r] + DNAME[d], c["TB/T0"], src_of(B81), k + ".TB/T0", f"BES only (TB vs T0), {r}, {d}", "4.2")
    for d in MDES:
        m = b6["main"][f"{d}|L5|common"]["all"]
        for r in ("D5", "D100"):
            pct_macro("IncMast" + RNAME[r] + DNAME[d], m[r]["TX/T0"], s6, f"main.{d}|L5|common.all.{r}.TX/T0",
                      f"MAST Thomson increment, {r}, {d}", "4.4/abstract")
    m1 = b5["main"]["V1|L5|common"]["all"]["D650"]
    rat_macro("CostDsixfiftyVone", m1["staleness T0"], s5, "main.V1|L5|common.all.D650.staleness T0",
              "cost of staleness of T0, D650, V1 (R1)", "4.1/abstract")
    rat_macro("CostSVone", b5["main"]["V1|L5|S"]["all"]["S"]["staleness T0"], s5, "main.V1|L5|S.all.S.staleness T0",
              "cost of staleness of T0, S, V1", "4.1")
    rat_macro("RecDsixfiftyVone", m1["recovered share"], s5, "main.V1|L5|common.all.D650.recovered share",
              "recovered share, D650, V1", "4.1")
    rat_macro("RecSVone", b5["main"]["V1|L5|S"]["all"]["S"]["recovered share"], s5, "main.V1|L5|S.all.S.recovered share",
              "recovered share, S, V1", "4.1")
    for d in ("V3", "V4a"):
        rat_macro("RecDsixfifty" + DNAME[d], b5["main"][f"{d}|L5|common"]["all"]["D650"]["recovered share"], s5,
                  f"main.{d}|L5|common.all.D650.recovered share", f"recovered share, D650, {d}", "4.1/4.3")
    mm = b6["main"]["M1|L5|common"]["all"]["D100"]
    rat_macro("CostMastDhundredMone", mm["staleness T0"], s6, "main.M1|L5|common.all.D100.staleness T0",
              "MAST cost of staleness, D100, M1 (S1)", "4.4")
    rat_macro("RecMastDhundredMone", mm["recovered share"], s6, "main.M1|L5|common.all.D100.recovered share",
              "MAST recovered share, D100, M1", "4.4")
    k10 = b8["main"]["V1|L5"]["all"]["K10"]
    er = k10["equivalent rate"]
    a = fnum(er["point"]["k_prime"], "%.2f", s8, "main.V1|L5.all.K10.equivalent rate.point.k_prime", "frames, two decimals")
    lo = fnum(er["cluster"]["k_prime"][0], "%.2f", s8, "main.V1|L5.all.K10.equivalent rate.cluster.k_prime[0]", "frames, two decimals")
    hi = fnum(er["cluster"]["k_prime"][1], "%.2f", s8, "main.V1|L5.all.K10.equivalent rate.cluster.k_prime[1]", "frames, two decimals")
    macro("EqRateKten", f"{a}~[{lo}, {hi}]", "equivalent spacing k' at K10, V1 (frames, interval)", s8 + " main.V1|L5.all.K10.equivalent rate", "4.5/abstract")
    macro("EqRateKtenPt", a, "equivalent spacing k' at K10, V1 (frames)", s8, "4.5")
    macro("EqRateKtenMs", fnum(10 * er["point"]["k_prime"], "%.1f", s8, "main.V1|L5.all.K10.equivalent rate.point.k_prime",
                               "10 ms x k', one decimal"), "equivalent spacing at K10 in ms (10 x k')", s8, "4.5")
    macro("EqRateKtenFactor", fnum(er["point"]["factor_k_over_k_prime"], "%.2f", s8,
                                   "main.V1|L5.all.K10.equivalent rate.point.factor_k_over_k_prime", "ratio, two decimals"),
          "spacing ratio k/k' at K10", s8, "4.5")
    q = b5["age_under_S"]["L5"]["newest_value_age_s"]["quantiles"]
    macro("AgeMedianS", fnum(q["0.5"], "%.2f", s5, "age_under_S.L5.newest_value_age_s.quantiles.0.5", "s, two decimals"),
          "median age (s) of the newest value the analysis could deliver (S set, L5)", s5, "1/2/abstract")
    macro("AgeQuartLoS", fnum(q["0.25"], "%.2f", s5, "age_under_S.L5.newest_value_age_s.quantiles.0.25", "s, two decimals"),
          "25th percentile of that age (s)", s5, "2")
    macro("AgeQuartHiS", fnum(q["0.75"], "%.2f", s5, "age_under_S.L5.newest_value_age_s.quantiles.0.75", "s, two decimals"),
          "75th percentile of that age (s)", s5, "2")
    sS = b5["main"]["V1|L5|S"]["all"]["S"]
    macro("RmseTXSVone", ev(sS["rmse_eV"]["TX"], s5, "main.V1|L5|S.all.S.rmse_eV.TX"), "RMSE of TX under S, V1 (eV)", s5, "4.1/4.6")
    macro("RmseOSVone", ev(sS["rmse_eV"]["O"], s5, "main.V1|L5|S.all.S.rmse_eV.O"), "RMSE of the frame's own reading on the S set, V1 (eV)", s5, "4.1/4.6")
    macro("RmseTZeroSVone", ev(sS["rmse_eV"]["T0"], s5, "main.V1|L5|S.all.S.rmse_eV.T0"), "RMSE of T0 under S, V1 (eV)", s5, "4.1")
    t = b5["timing"]
    macro("TimeMedian", fnum(t["median_ms"], "%.2f", s5, "timing.median_ms", "ms, two decimals"), "computation time per row, median (ms)", s5, "4.1/abstract")
    macro("TimePnn", fnum(t["p99_ms"], "%.2f", s5, "timing.p99_ms", "ms, two decimals"), "computation time per row, 99th percentile (ms)", s5, "4.1")
    # counts
    macro("NDoneRows", cnt(b5["sets"]["d1_rows"], s5, "sets.d1_rows"), "D1 rows", s5, "2/3")
    macro("NDoneDischarges", cnt(b5["designs"]["d1_discharges"], s5, "designs.d1_discharges"), "D1 discharges", s5, "2/3")
    macro("NDoneClusters", cnt(b5["designs"]["clusters_d1"], s5, "designs.clusters_d1"), "D1 shot clusters", s5, "2/3")
    nb = (PAPER / "nbi_files.md").read_text(encoding="utf-8")
    mg = re.search(r"Grid rows: ([\d,]+)", nb)
    reg(mg.group(1), "docs/paper/nbi_files.md", "sec. 1 'Grid rows'", "verbatim from the md")
    macro("NGridRows", mg.group(1).replace(",", "{,}"), "10 ms grid rows without the first and last rows of the files",
          "docs/paper/nbi_files.md sec. 1", "2")
    macro("NFiles", cnt(J(B73)["rows_info"]["files"], src_of(B73), "rows_info.files"), "shot files", src_of(B73), "2")
    cc = b5["main"]["V1|L5|common"]["all"]["counts"]
    macro("NCommonRows", cnt(cc["rows"], s5, "main.V1|L5|common.all.counts.rows"), "common set rows (L5)", s5, "3")
    macro("NCommonDischarges", cnt(cc["discharges"], s5, "main.V1|L5|common.all.counts.discharges"), "common set discharges", s5, "3")
    macro("NCommonClusters", cnt(cc["clusters"], s5, "main.V1|L5|common.all.counts.clusters"), "common set clusters", s5, "3")
    cs = b5["main"]["V1|L5|S"]["all"]["counts"]
    macro("NSRows", cnt(cs["rows"], s5, "main.V1|L5|S.all.counts.rows"), "S set rows (L5)", s5, "3")
    macro("NSDischarges", cnt(cs["discharges"], s5, "main.V1|L5|S.all.counts.discharges"), "S set discharges", s5, "3")
    macro("NSClusters", cnt(cs["clusters"], s5, "main.V1|L5|S.all.counts.clusters"), "S set clusters", s5, "3")
    ck = b8["counts"]["V1"]["L5"]
    macro("NKRows", cnt(ck["rows"], s8, "counts.V1.L5.rows"), "K set rows (L5)", s8, "4.5")
    macro("NKDischarges", cnt(ck["discharges"], s8, "counts.V1.L5.discharges"), "K set discharges", s8, "4.5")
    cm = b6["main"]["M1|L5|common"]["all"]["counts"]
    macro("NMastRows", cnt(cm["rows"], s6, "main.M1|L5|common.all.counts.rows"), "MAST common rows (L5)", s6, "4.4")
    macro("NMastShots", cnt(cm["shots"], s6, "main.M1|L5|common.all.counts.shots"), "MAST shots", s6, "4.4")
    macro("NMastSessions", cnt(cm["sessions"], s6, "main.M1|L5|common.all.counts.sessions"), "MAST sessions", s6, "4.4")
    lb = b5["main"]["V1|L5|S"]["block from 32312"]["counts"]
    macro("NLastBlockDischarges", cnt(lb["discharges"], s5, "main.V1|L5|S.block from 32312.counts.discharges"),
          "discharges of the last block on the S set", s5, "4.6")
    macro("NLastBlockRows", cnt(lb["rows"], s5, "main.V1|L5|S.block from 32312.counts.rows"), "rows of the last block on the S set", s5, "4.6")
    pct_macro("IncSLastBlockVone", b5["main"]["V1|L5|S"]["block from 32312"]["S"]["TX/T0"], s5,
              "main.V1|L5|S.block from 32312.S.TX/T0", "increment under S on the last block, V1", "4.6")
    pct_macro("IncBeamSLastBlockVone", b9["main"]["V1|L5|S"]["block from 32312"]["S"]["TXN/TN"], s9,
              "main.V1|L5|S.block from 32312.S.TXN/TN", "TXN vs TN under S on the last block, V1", "4.6")
    sp = b0["spearman"]["all blocks"]
    p = (fnum(sp["point"], "%+.3f", src_of(B80), "spearman.all blocks.point", "signed three decimals"),
         fnum(sp["cluster"][0], "%+.3f", src_of(B80), "spearman.all blocks.cluster[0]", "signed three decimals"),
         fnum(sp["cluster"][1], "%+.3f", src_of(B80), "spearman.all blocks.cluster[1]", "signed three decimals"))
    macro("GateSpearman", f"{p[0]}~[{p[1]}, {p[2]}]", "rank correlation of per-discharge skill (V1, S) with C_E",
          src_of(B80) + " spearman.all blocks", "4.6")
    macro("NGateSpearmanDischarges", cnt(sp["discharges"], src_of(B80), "spearman.all blocks.discharges"),
          "discharges of that rank correlation", src_of(B80), "4.6")


# ----------------------------------------------------------------------------------------------
README_HEAD = """# Data layer of the rewritten manuscript (tables, macros, figures)

Generated hand-off for the writer of `main_ppcf.tex` / `supp_ppcf.tex` (line of PAPER_DESIGN.md, 2026-11-20).
This file is written by `make_tables_v2.py`; do not edit it by hand.

## How to use

- `\\input{gen_v2.tex}` in the preamble (after `booktabs`). Every table is a macro `\\TabXxx` (a complete
  `tabular`, already sized at `\\footnotesize` or `\\scriptsize` to fit the 16 cm text width) with a caption stub
  `\\CapXxx` that names target, availability rule, design and interval. Use it as
  `\\begin{table}\\centering\\caption{\\CapXxx}\\label{tab:xxx}\\TabXxx\\end{table}`; edit the stub in the
  manuscript if the prose needs it (keep its numbers; they are registered tokens).
- `gen_v2.tex` defines `\\ci{point}{[lo, hi]}` (point above its interval) and `\\tcell`.
- Comparisons are printed as the reduction of the MSE in per cent (100 x skill, one decimal) with the 95 %
  interval from the shot-cluster bootstrap (MAST: session bootstrap); the point is bold when that interval
  excludes zero. RMSE in eV with one decimal; ratios with three decimals. Statement tables print the
  registered value as registered (skill as a signed fraction, three decimals).
- Design names: V1 random, V2 shot clusters held out, V3 rolling origin, V4 chronological fit to 31991 (V4a)
  and fit to 32310 (V4b); MAST M1 random, M2 sessions held out, M3 rolling origin, M4 chronological.
- Regenerate: `py docs/paper/ppcf/make_tables_v2.py` then `py docs/paper/ppcf/fig_v2.py` (needs the caches
  `data/.b73`, `data/.b75`, `data/.b79` for figures 1 and 7 only). Compile test: `latexmk -pdf test_v2.tex`
  in this folder. Number check: `py docs/paper/ppcf/check_numbers_v2.py` (manuscript) or `--self`.
- Provenance of every printed number: `gen_v2_tokens.json` (token -> JSON file, key path, transformation).

"""

README_FIGS = """## Figures (`fig_v2.py`; `figv2_<name>.pdf/.png/.json`, the json holds every plotted number and `printed_tokens`)

Canvas widths are final: 16 cm for full-width figures, 10 cm for `figv2_sparse`; fonts 8 to 9 pt at that size. Include them at their natural size (`width=\\textwidth` for 16 cm, `width=10cm` for the sparse figure) so nothing is scaled.

| figure | shows | source | paper map |
|---|---|---|---|
| `figv2_problem` | (a) the example discharge: reported T_i, absent frames (vertical lines), NBI-1 power (NB1B + NB1C), BES and ECEI array means over their discharge medians, grey band where the four ECEI channels are constant (rows kept); (b) distribution over the S-set rows (L5) of the age of the newest CES value the beam-modulation analysis could deliver (median 0.65 s; last bin holds ages above 2 s) | caches `data/.b73/grid.npz`, `data/.b79/beam_cols.npz`, `data/.b75/inputs/S.npz` (`age`, ms) on `S_L5` of `data/.b75/sets.npz` | sec. 2 |
| `figv2_rules` | schematic of the availability rules on a 10 ms frame axis: fast samples to t; D10; Dd (d = 60 ms drawn); S (before the latest notch); Kk (k = 5 drawn) | none (drawn) | sec. 3 |
| `figv2_ladder` | RMSE against the delay of the history: (a) KSTAR V1 (hold, causal GP, T0, TX; O as dashed line, named in the legend; rule S as open diamonds at the median age 0.65 s, S set), (b) MAST M1 | b75 `main.V1|L5|common.all.D*.rmse_eV`, `main.V1|L5|S.all.S.rmse_eV`; b76 `main.M1|L5|common.all.D*.rmse_eV` | 4.1, 4.4 |
| `figv2_designs` | forest plot of the fast-channel increment (%, cluster/session interval; filled = excludes zero): KSTAR D10, D650, S x V1..V4b; MAST D5, D100 x M1..M4 | b75, b76 `TX/T0` | 4.3, 4.4 |
| `figv2_which` | (a) D650, (b) S: per design TE/T0, TB/T0 (V1, V4 from B.75; V2, V3 from B.81), TX/T0 (B.75), TN/T0, TXN/TN; (c) the B.81 controls under D650 per design (fast inputs permuted, fast inputs at the anchor row, present over anchor row) and MAST M1 D100 (B.76) | b81, b75, b79, b76 | 4.2 |
| `figv2_sparse` | RMSE of T0 and TX against the spacing of the CES values (V1), O as dashed line, equivalent spacing of TX at K10 (k' = 4.67 frames = 46.7 ms, cluster interval as error bar) | b78 `main.V1|L5.all.K*` | 4.5 |
| `figv2_trace` | the example discharge in three 2 s windows (centred at the 25/50/75 % quantiles of its scored times): reported values, following level L5, T0 and TX under rule S (stored estimates), absent frames | caches `data/.b75/fits/V1_r2_k*__{T0,TX}__S__L5.npz`, `data/.b73/rows.npz`, `grid.npz` | 4.1 / 4.6 |
| `figv2_trace_q25` | the same for the 25th-percentile discharge (second example) | same | supplement |
| `figv2_limits` | (a) increment under S by block of shots, TX vs T0 and TXN vs TN (V1); (b) first row after a notch over the level before, T_i, BES, ECEI, by block (after the fact, no interval) | b75, b79 `main.V1|L5|S.<block>.S`; `b75_last_block.md` section (7) | 4.6 |

**Example discharge (selection rule, no hand picking).** Among the test discharges of unit group V1,
re-partition r2 (folds k0 to k4), with at least 500 rows in the S set (target L5) and at least three absent
steps (15 ms < time step < 45 ms inside a segment of the grid), the one whose skill of TX over T0 under S
(stored B.75 estimates, its own scored rows) is the median (lower median for an even count). Second example:
the 25th percentile. The chosen shots, the number of candidates and the skills are in `figv2_problem.json`
(`example`).

"""

README_MISSING = """## Missing from the result files (not computed here; listed for the session lead)

1. (resolved by B.81) TE and TB under V2 and V3 and the controls per design under D650 and S. The earlier
   one-unit unregistered B.75 control (`b75_lead_checks.md` part C: permuted +0.013, anchor row -0.128) is
   superseded by B.81 and no longer printed.
2. No anchor-row control for B.79 (beam inputs); none for MAST under rules other than D100.
3. (none)
4. TX against TX8 (the anchor scalars given the other inputs) is not stored in b75; `\\TabAnchorScalars`
   shows TX8 vs T0 and TX vs T0 side by side instead. TX8 exists for V1 and V4 only.
5. Computation time is measured for rule D650 only (one unit, shared machine); none for S or for MAST.
6. The age distribution of the newest deliverable value exists in the JSON as quantiles and bands only; the
   histogram of `figv2_problem` is computed from the stored ages in the cache (no estimator).
7. B.71 (replay when updates stop) and B.74 (bursts) have no table beyond the B.74 statement table; B.71's
   statements are not tabulated (the paper map puts it in the supplement; ask if wanted).

"""


def write_readme():
    lines = [README_HEAD, "## Tables (`gen_v2.tex`)\n", "| macro | shows | source (file and key) | paper map |",
             "|---|---|---|---|"]
    for k, n, w, s, m in README:
        if k == "table":
            lines.append(f"| `{n}` (caption `\\Cap{n[4:]}`) | {w} | {s.replace('|', '/')} | {m} |")
    lines += ["", "## Prose macros (`gen_v2.tex`, end of file)", "",
              "Naming: `Inc<Rule><Design>` = fast-channel increment (TX vs T0), `IncBeam` = TXN vs TN, `BeamAlone` = "
              "TN vs T0, `EceiOnly`/`BesOnly` = TE/TB vs T0, `IncMast`, `IncKten`, `Cost`, `Rec` (recovered share), "
              "`N...` counts. Rule: Dten, Dsixfifty, S, Dfive, Dhundred, Kten. Design: Vone, Vtwo, Vthree, Vfoura "
              "(fit to 31991), Vfourb (fit to 32310), Mone..Mfour. A macro gives `point~[lo, hi]` (per cent for "
              "skills); the same name with `Pt` gives the point only. The writer adds `\\%` and the words.", "",
              "| macro | value is | source |", "|---|---|---|"]
    for k, n, w, s, m in README:
        if k == "macro":
            lines.append(f"| `{n}` | {w} | {s.replace('|', '/')} |")
    lines += ["", README_FIGS, README_MISSING]
    (HERE / "gen_v2_README.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    tab_sets()
    tab_scored()
    tab_inputs()
    tab_rules_v1()
    tab_rules_designs()
    tab_increment()
    tab_increment_ladder()
    tab_fifths()
    tab_which()
    tab_controls()
    tab_refit()
    tab_mast_rmse()
    tab_mast_cost()
    tab_mast_dalpha()
    tab_sparse()
    tab_sparse_designs()
    tab_blocks()
    tab_block_descr()
    tab_gate()
    tab_gate_blocks()
    tab_correct()
    tab_dten()
    tab_network()
    tab_timing()
    tab_as_fitted()
    tab_as_fitted(("V4a", "V4b"), "AsFittedChrono")
    tab_anchor_scalars()
    tab_bursts()
    tab_stuck()
    tab_vs_reading()
    tab_nbi()
    for batch, f, title in (("Bseventythree", B73, "B.73 (previous frame visible: targets, designs, network rows)"),
                            ("Bseventyfour", B74, "B.74 (bursts inside the exposure)"),
                            ("Bseventyfive", B75, "B.75 (stale CES history)"),
                            ("Bseventysix", B76, "B.76 (MAST, stale CXRS history)"),
                            ("Bseventyseven", B77, "B.77 (correcting the frame's reading)"),
                            ("Bseventyeight", B78, "B.78 (a CES value every k-th frame)"),
                            ("Bseventynine", B79, "B.79 (beam waveforms as inputs)"),
                            ("Beighty", B80, "B.80 (coherence of the ECEI channels)"),
                            ("Beightyone", B81, "B.81 (controls of the fast inputs; ECEI and BES halves under V2, V3)")):
        tab_statements(batch, f, title)
    tot = tab_stat_summary()
    nnh = tab_stat_not_held()
    print("statement totals (registered, hold, do not hold, definite, met):", tot, " not-held rows per table:", nnh)
    macros()
    text = "\n".join(PRE + OUT + ["% ---- prose macros"] + MACROS) + "\n"
    (HERE / "gen_v2.tex").write_text(text, encoding="utf-8")
    (HERE / "gen_v2_tokens.json").write_text(json.dumps(
        {"generator": "docs/paper/ppcf/make_tables_v2.py", "n_tokens": len(TOKENS), "tokens": TOKENS},
        indent=1, ensure_ascii=False), encoding="utf-8")
    (HERE / "gen_v2_index.json").write_text(json.dumps(
        [{"kind": k, "name": n, "what": w, "source": s, "paper_map": m} for k, n, w, s, m in README],
        indent=1, ensure_ascii=False), encoding="utf-8")
    write_readme()
    # every table in a float, for test_v2.tex
    allt = ["% generated by make_tables_v2.py: every table of gen_v2.tex in a float (for test_v2.tex)"]
    for i, (k, n, w, s, m) in enumerate(r for r in README if r[0] == "table"):
        nm = n[len("\\Tab"):]
        allt += [r"\begin{table}[htbp]\centering", "\\Cap" and f"\\caption{{\\Cap{nm}}}\\label{{tab:{nm.lower()}}}",
                 f"\\Tab{nm}", r"\end{table}"]
        if i % 3 == 2:
            allt.append(r"\clearpage")
    (HERE / "gen_v2_alltables.tex").write_text("\n".join(allt) + "\n", encoding="utf-8")
    print("tables:", sum(1 for r in README if r[0] == "table"), " macros:", len(MACROS), " tokens:", len(TOKENS))


if __name__ == "__main__":
    main()
