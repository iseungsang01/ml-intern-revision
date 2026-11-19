# -*- coding: utf-8 -*-
"""A0 conference poster (84.1 x 118.9 cm, portrait, printed 1:1) for the PPCF manuscript
docs/paper/ppcf/main_ppcf.tex.  English only, laid out for plasma physicists.

  py docs/presentation/make_figures_poster_ppcf.py          # figures first (figures/poster_ppcf/)
  py docs/presentation/build_poster_ppcf.py                 # -> KSTAR_CES_poster_PPCF.pptx + .png
  py docs/presentation/check_poster_numbers.py docs/presentation/KSTAR_CES_poster_PPCF.pptx
  py docs/presentation/build_poster_ppcf.py --anon --out X.pptx --png Y.png   # author line anonymised

* Three columns of 25.3 cm.  Sections are numbered, with a navy rule under each heading; bold
  lead-in words; takeaway boxes with a light fill and a thick navy bar on the left; booktabs
  tables; italic captions; a check-mark list in the conclusion; numbered limitations with
  arrow remedies.
* Result numbers are in tables only.  In every table a bold point estimate means that its 95 %
  interval excludes 0 (1 for a ratio); emphasis is by row shading.  Sentences keep only numbers
  that describe the data (discharges, rows, R, channel counts, 10 ms, 3 keV).
* Every sentence condenses the manuscript; the section -> manuscript-line map is in
  docs/presentation/poster_ppcf_sources.md.
* Text boxes are sized from the Segoe UI metrics (PIL); lines also break after a hyphen, as in
  PowerPoint.  With FIT = 1.0 the predicted line count matched PowerPoint's for every text box.
  The body size is the largest of BODY_STEPS at which all three columns fit; the height left in
  a column is spread over its gaps so that the three columns end level.
* All shapes are native PowerPoint shapes and text boxes; the PNG is exported through
  PowerPoint COM (Slides(1).Export), which also reports any text box whose text PowerPoint lays out
  taller than the box.  PowerPoint is quit only if no other document is open.
"""
import argparse
import io
import re
import sys
from pathlib import Path

from PIL import Image, ImageFont
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Cm, Pt

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures" / "poster_ppcf"

ap = argparse.ArgumentParser()
ap.add_argument("--anon", action="store_true", help="anonymous author line (needs --out)")
ap.add_argument("--out", type=Path)
ap.add_argument("--png", type=Path)
ap.add_argument("--png-px", type=int, default=2400)
ap.add_argument("--no-png", action="store_true")
args = ap.parse_args()
if args.anon and args.out is None:
    sys.exit("--anon needs --out (the anonymous copy never overwrites the poster)")
OUT = args.out or HERE / "KSTAR_CES_poster_PPCF.pptx"
PNG = args.png or OUT.with_suffix(".png")

# ====================================================================================== text
TITLE_LINES = ("Causal nowcasting of the charge-exchange ion temperature",
               "on an edge chord of KSTAR with co-located fast diagnostics")
SUBTITLE = ("A recurrent network with a boosted second stage, and a five-parameter filter, "
            "against references that read the CES history alone")
AUTHOR = "Presenter · Affiliation" if args.anon else \
    "**Seungsang Lee** · Department of Nuclear Engineering, Seoul National University"
DATA_LINE = "KSTAR 2022 campaign data provided by the KSTAR team"

S1 = [
    "**Problem:** charge-exchange spectroscopy (CES) reports the temperature of C VI ions once per 10 ms "
    "frame [1]. On the KSTAR edge chord selected as the pedestal top the signal is weak, and values are "
    "missing more often when the beam emission is low: not completely at random.",
    "**Need:** monitoring or control needs an estimate formed when the frame is taken, without the next value; "
    "the simplest reference holds the last value.",
    "**Available:** beam-emission spectroscopy (BES) and ECEI view the same location without gaps; neither "
    "measures T_i.",
]
S1_BOX = ("Question: what does the BES value of the present frame add to the CES history, when the reported "
          "temperature of that frame is estimated causally?")

S2_DATA = ("**Data:** 641 discharge files of the KSTAR 2022 campaign, 245,925 rows on a 10 ms grid (639 "
           "discharges enter the splits). A row holds 9 BES, 4 ECEI and 2 Mirnov channels and the CES values "
           "of one chord at R = 2.198 m (Fig. 1).")
FIG1_CAP = ("Fig. 1. Positions on the major radius. Top: schematic T_i shape (orientation only; no profile is in "
            "the files), published pedestal tops of other discharges, separatrix of one 2022 discharge read "
            "from a published figure. Bottom: the channels of the files (hatched: inferred). The Mirnov coils "
            "are at the wall.")
S2_ROW = ("**What a row holds:** raw streams of twelve discharges show that each fast-channel value is a "
          "single raw sample at the row's stamp, not an average over the CES frame. A step of 20 to 40 ms "
          "between rows is a beam blip, whose frames are left out (Fig. 2).")
FIG2_CAP = ("Fig. 2. A row of the 10 ms files (schematic). Its CES value is the frame whose exposure starts "
            "within about 2 ms of the stamp; a causal estimate for row t uses the fast samples up to t and the "
            "CES values before t.")
S2_TARGET = ("**Target:** the C VI temperature that the KSTAR CES analysis reports; whether it equals the "
             "main-ion temperature is not tested. Values above 3 keV, taken as failed fits [5], are treated as "
             "missing; held values (repeats to the last digit) are removed.")
FIG3_CAP = ("Fig. 3. Reported T_i and three causal estimates in about 0.65 s of three test discharges of one "
            "split. Shaded: rows without a value; dotted: absent frames. (a, b) Many missing values and large "
            "jumps; (c) a typical discharge. Network: first stage, one initialisation.")

FIG4_CAP = ("Fig. 4. (a) The network (first stage; rotation branch not drawn) reads the CES history as the last "
            "value, its age and an observed flag [4]; the second stage, a boosted regression [3] fitted only on "
            "out-of-fold errors of non-test discharges, adds its estimate of the network's error. (b) The filter: "
            "c, BES common mode; g, radial difference; a large innovation v of variance S counts less. KF0: no "
            "drive, KFX: both.")
TAB1_CAP = ("Table 1. The references. Causal: CES values of earlier frames, training statistics, fast samples "
            "up to the stamp of the row.")
TAB1_HEAD = ("Reference", "What it reads")
TAB1_ROWS = [
    ("**hold**", "the last observed value"),
    ("**causal GP**", "main reference: Matérn-3/2 GP on the 16 latest CES values; hyper-parameters by the "
     "marginal likelihood at each row"),
    ("**offline GP**", "as the causal GP, with neighbours on both sides: uses the next value, not causal"),
    ("**history-only\ncontrol**", "the same network trained without the fast channels; the difference is the "
     "fast-channel increment (two-stage control: with a second stage on the CES history)"),
]
SKILL_NOTE = "skill = 1 − MSE(estimator) / MSE(reference); above 0: lower error than the reference"
S3_SCORE = ("**Scoring:** errors in eV on withheld measured rows, the same for every estimator; shot-level "
            "5-fold cross-validation on 639 discharges, repeated and pooled; 95 % intervals by a two-level "
            "bootstrap. **Registration:** statements and decision rules were committed to the author's public "
            "repository before each run.")

TAB2_CAP = ("Table 2. Values above 3 keV excluded. Network skills: eight re-partitions (r2–r9); the rest: "
            "three (r2–r4, 15 folds); one initialisation per network. Within 20 %: share of frames within 20 % "
            "of the value.")
TAB2_HEAD = ("Skill over", "Network (first stage)", "Two-stage estimator")
TAB2_ROWS = [
    ("hold", "**+0.375** {[+0.352, +0.394]}", "**+0.436** {[+0.418, +0.452]}"),
    ("causal GP", "**+0.089** {[+0.064, +0.112]}", "**+0.176** {[+0.155, +0.196]}"),
    ("offline GP (next value)", "**−0.037** {[−0.068, −0.008]}", "**+0.063** {[+0.038, +0.086]}"),
    ("control (increment)", "**+0.048** {[+0.022, +0.073]}", "**+0.097** {[+0.077, +0.116]}"),
    "Errors on the rows of r2–r4",
    ("RMSE (eV)", "184.2 {[174.3, 194.4]}", "175.4 {[165.9, 185.3]}"),
    ("RMSE − causal GP (eV)", "**−9.1** {[−11.5, −6.6]}", "**−17.8** {[−20.2, −15.4]}"),
    ("median |error| ÷ hold", "1.010 {[0.972, 1.062]}", "**0.909** {[0.889, 0.931]}"),
    ("within 20 % − causal GP", "−0.002 {[−0.013, +0.007]}", "**+0.021** {[+0.018, +0.025]}"),
]
TAB2_FILL = (1, 2)
TAB_BOLD_NOTE = "Bold: the 95 % interval excludes 0 (1 for a ratio)."
S4_ITEMS = [
    "**Network:** lower error than every causal reference; its increment has an interval above 0 also at the "
    "fold level, under a Huber loss and with the largest errors trimmed.",
    "**Two-stage estimator:** all six registered statements met; over the offline GP, which waits for the "
    "next value, its interval is above 0, that of the network alone below 0.",
]

TAB3_CAP = ("Table 3. Skill over the history-only control (filter: over KF0, the filter without the drive). "
            "Bold as in Table 2.")
TAB3_HEAD = ("Inputs", "Skill over the control")
TAB3_ROWS = [
    "Trained network, inputs changed at test time (r2–r9)",
    ("BES channels one frame old", "+0.000 {[−0.024, +0.022]}"),
    "Networks retrained with changed inputs (r2–r4)",
    ("all fast channels one row old", "−0.006 {[−0.025, +0.016]}"),
    ("BES and CES history only", "**+0.044** {[+0.020, +0.066]}"),
    ("ECEI and CES history only", "−0.011 {[−0.092, +0.022]}"),
    ("Mirnov removed", "**+0.060** {[+0.023, +0.085]}"),
    ("first-stage network over the BES-only one", "+0.009 {[−0.019, +0.031]}"),
    "Five-parameter filter (r3–r9)",
    ("KFX (BES drive) over KF0 (no drive)", "**+0.049** {[+0.040, +0.059]}"),
]
TAB3_FILL = (1, 4, 9)
S5_ITEMS = [
    "**Present frame, BES:** with only its BES channels one frame old the trained network loses its increment; "
    "a network on BES and the CES history alone has it, one on ECEI has none, and Mirnov is not needed.",
    "**Beam or plasma:** BES is light of the heating beam; the filter's BES drive gives an increment of the "
    "same size. With no beam power in the files, a change of the beam is not excluded.",
]

FIG5_CAP = ("Fig. 5. Event means after beam blips of one absent frame, as ratios to the six rows before "
            "(shaded: 95 % interval; dotted: control rows). The T_i deficit follows the ECEI level, which uses "
            "neither the beam nor the spectrometer, not BES, which recovers within two rows.")
TAB4_CAP = "Table 4. First stage, pooled (r2–r9)."
TAB4_HEAD = ("Readout", "Value")
TAB4_ROWS = [
    "Increment: network over its control",
    ("rows 1–3 after an absent frame", "**+0.151** {[+0.093, +0.215]}"),
    ("coldest third of the discharges by median T_i", "+0.007 {[−0.023, +0.031]}"),
    ("middle third", "**+0.042** {[+0.009, +0.074]}"),
    ("hottest third", "**+0.092** {[+0.040, +0.137]}"),
    ("without the 64 discharges with the largest gain", "+0.027 {[−0.004, +0.051]}"),
    "Fit or plasma, and the scatter floor",
    ("ρ1: correction still present in the next frame", "0.810 {[0.753, 0.866]}"),
    ("floor s, root mean square (eV)", "135.7 {[126.4, 145.7]}"),
    ("network: mean squared error over the floor", "1.331 {[1.246, 1.427]}"),
]
TAB4_FILL = (1, 2, 7)
TAB4_NOTE = ("Bold: the interval of a skill excludes 0. ρ1: coefficient of the network-minus-control correction "
             "one frame later over that in its own frame; it can exceed 1.")
S6_ITEMS = [
    "**Where:** the increment is largest in the three rows after an absent frame, where the control sees the "
    "longer time step but not the depth of the fall; it is established in the middle and hottest thirds of the "
    "discharges, not in the coldest.",
    "**Fit or plasma:** the next frame is fitted independently, and most of the fast-channel correction lasts "
    "into it. Between two scored frames, the floor set by the scatter of the reported value accounts for most "
    "of the network's error.",
]

S7_BOX = ("On the KSTAR edge CES chord selected as the pedestal top, the beam-emission value of the present "
          "frame carries information about the reported ion temperature of that frame beyond what the CES "
          "history gives.")
S7_CHECKS = [
    "**Two estimators of different kinds agree:** the network against its control, the filter with and "
    "without a BES drive.",
    "**Present frame:** with only its BES channels one frame old, the trained network loses its increment.",
    "**Two-stage estimator:** the increment and the skill over the offline GP have intervals above 0, the "
    "ratio of the median error to that of hold an interval below 1.",
    "**The correction lasts:** most of it is still there in the next, independently fitted frame.",
]
S7_CAVEAT = ("**For the network alone,** the gain is small against the frame-to-frame scatter of the reported "
             "value, is not established in the coldest third of the discharges by temperature, rests on the "
             "tenth of them with the largest gain, and was not established on single file-level splits.")

S8_ITEMS = [
    ("**Cut population.** With the values above 3 keV kept (Table 5) the margin over the causal GP remains, "
     "the increment does not: every statement is made on the cut population.",
     "→ per-frame fit uncertainties, to tell failed fits from plasma"),
    ("**One chord, one campaign, one device;** the C VI impurity temperature; no beam power in the files.",
     "→ the beam power, an equilibrium mapping, further chords"),
    ("**No untouched hold-out discharges.**",
     "→ temporal confirmation on the last 97 discharges (running)"),
    ("**Real time is a prospect:** no real-time use of BES or CES on KSTAR is published; use needs BES of the "
     "present frame and a CES value at most two frames late.",
     "→ a CES analysis that needs no beam-off frame [2]"),
]
TAB5_CAP = "Table 5. Values above 3 keV kept in training and scoring (r2–r4)."
TAB5_HEAD = ("Network over", "Skill")
TAB5_ROWS = [("causal GP", "**+0.109** {[+0.070, +0.146]}"),
             ("history-only control", "−0.002 {[−0.017, +0.010]}")]

REFS = ("[1] W. H. Ko et al., Rev. Sci. Instrum. 81, 10D740 (2010).   [2] J. K. Lee et al., Fusion Eng. Des. "
        "222, 115518 (2026).   [3] J. H. Friedman, Ann. Stat. 29, 1189 (2001).   [4] Z. Che et al., Sci. Rep. "
        "8, 6085 (2018).   [5] M. Kim et al., Nucl. Fusion 64, 106052 (2024).")
ACK = "The KSTAR data were provided by the KSTAR team. There was no dedicated funding."

# ====================================================================================== geometry and type
W_CM, H_CM = 84.1, 118.9
CW = 25.3                                  # column width = width of every figure (make_figures_poster_ppcf.py)
M = 2.0                                    # side margin
GUT = (W_CM - 2 * M - 3 * CW) / 2          # 2.1 cm, with a thin rule in the middle
XS = [M + i * (CW + GUT) for i in range(3)]
FONT, FONT_SB, FONT_SYM = "Segoe UI", "Segoe UI Semibold", "Segoe UI Symbol"


def rgb(h):
    return RGBColor(int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


NAVY, INK, BODY, MUTED = rgb("1C2E5A"), rgb("15181E"), rgb("22262E"), rgb("5A6475")
TINT, RULE, WHITE, SUBTLE = rgb("E8ECF4"), rgb("B7BFCC"), rgb("FFFFFF"), rgb("C8D2E6")
SHADE = rgb("E3E9F3")
ACCENT = rgb("5B7DB8")

_FONTS = {}


def _f(bold, italic):
    key = (bold, italic)
    if key not in _FONTS:
        name = {(False, False): "segoeui.ttf", (True, False): "segoeuib.ttf", (False, True): "segoeuii.ttf",
                (True, True): "segoeuiz.ttf", ("sb", False): "seguisb.ttf"}[key]
        _FONTS[key] = ImageFont.truetype(str(Path(r"C:\Windows\Fonts") / name), 200)
    return _FONTS[key]


PT2CM = 2.54 / 72.0
NB = "\u00a0"
FIT = 1.0                                # share of the box width filled when wrapping (PowerPoint safety)
TOK = re.compile(r"(\*\*|\{|\}|T_i|ρ1)")
WARN = []


def glue(s):
    """Keep intervals, ranges, numbers with units and short labels on one line."""
    s = re.sub(r"\[[^\]]*\]", lambda m: m.group(0).replace(" ", NB), s)
    s = re.sub(r"(?<=\d) (?=%)", NB, s)
    s = re.sub(r"(?<=\d) (?=(?:ms|eV|keV|m|cm|s)(?![A-Za-z]))", NB, s)
    s = re.sub(r"(?<=[=÷−]) ", NB, s)
    s = re.sub(r" (?=[=÷])", NB, s)
    return s


def parse(s, bold=False):
    """'**bold**', '{muted}', T_i and ρ1 (subscript) -> pieces (text, bold, muted, shift)."""
    out, b, mu = [], bold, False
    for part in TOK.split(glue(s)):
        if not part:
            continue
        if part == "**":
            b = not b
        elif part == "{":
            mu = True
        elif part == "}":
            mu = False
        elif part in ("T_i", "ρ1"):
            out += [(part[0], b, mu, 0), (part[-1], b, mu, -1)]
        else:
            out.append((part, b, mu, 0))
    return out


def _w(s, pt, bold, italic=False):
    return _f(bold, italic).getlength(s) / 200.0 * pt * PT2CM


def wrap(s, pt, width, bold=False, italic=False):
    """Word wrap ('\\n' forces a break; a line may also break after a hyphen inside a word, as PowerPoint
    does); returns the width (cm) of each line."""
    words, cur, forced, glued = [], 0.0, False, False
    for txt, b, mu, sh in parse(s, bold):
        size = pt * (0.7 if sh else 1.0)
        txt = re.sub(r"(?<=[A-Za-z]-)(?=[A-Za-z])", "​", txt)
        for atom in re.split("( |\n|​)", txt):
            if atom == "​":                              # a break opportunity after a hyphen
                words.append((cur, forced, glued))
                cur, forced, glued = 0.0, False, True
                continue
            if atom == "":
                continue
            if atom in (" ", "\n"):
                words.append((cur, forced, glued))
                cur, forced, glued = 0.0, atom == "\n", False
            else:
                cur += _w(atom, size, b, italic)
    words.append((cur, forced, glued))
    space = _w(" ", pt, bold, italic)
    lines, x = [], 0.0
    for w, brk, glue_ in words:
        sp = 0.0 if glue_ else space
        if brk:
            lines.append(x)
            x = w
        elif x > 0 and x + sp + w > width * FIT:
            lines.append(x)
            x = w
        else:
            x = x + (sp if x > 0 else 0) + w
    lines.append(x)
    if max(lines) > width * FIT + 1e-6:
        WARN.append("word wider than its box (%.1f > %.1f cm): %s" % (max(lines), width, s[:40]))
    return lines


def text_h(paras, pt, width, pitch, after=0.0, bold=False, italic=False):
    paras = [paras] if isinstance(paras, str) else paras
    n = sum(len(wrap(s, pt, width, bold, italic)) for s in paras)
    return (n * pitch + after * (len(paras) - 1)) * PT2CM


def text_w(s, pt, bold=False):
    return max(wrap(s, pt, 1e9, bold))


# ====================================================================================== shapes
prs = Presentation()
prs.slide_width, prs.slide_height = Cm(W_CM), Cm(H_CM)
slide = prs.slides.add_slide(prs.slide_layouts[6])
DRY = False


def _font(run, size, color, bold=False, italic=False, shift=0, face=FONT):
    f = run.font
    f.size = Pt(size)
    f.bold = bold
    f.italic = italic
    f.color.rgb = color
    f.name = face
    rpr = run._r.get_or_add_rPr()
    rpr.set("lang", "en-US")
    latin = rpr.find(qn("a:latin"))
    for tag in ("a:cs", "a:ea"):
        latin.addnext(rpr.makeelement(qn(tag), {"typeface": face}))
    if shift:
        rpr.set("baseline", "-25000")


def _fill(tf, paras, size, color, pitch, after, align, bold, italic=False, face=FONT):
    for i, s in enumerate(paras):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.line_spacing = Pt(pitch)
        p.space_after = Pt(after if i < len(paras) - 1 else 0)
        for txt, b, mu, sh in parse(s, bold):
            for k, piece in enumerate(txt.split("\n")):
                if k:
                    p.add_line_break()
                if piece:
                    r = p.add_run()
                    r.text = piece
                    fc = FONT_SYM if piece in ("✓",) else face
                    _font(r, size, MUTED if mu else color, b, italic, sh, fc)


def text(x, y, w, paras, size, color=BODY, pitch=None, after=0.0, align=PP_ALIGN.LEFT, bold=False,
         italic=False, name=None, face=FONT, check=True):
    paras = [paras] if isinstance(paras, str) else paras
    pitch = pitch or round(size * 1.3, 1)
    need = text_h(paras, size, w, pitch, after, bold, italic)
    if DRY:
        return need
    if check:
        for s in paras:
            ln = wrap(s, size, w, bold, italic)
            if len(ln) > 1 and ln[-1] < 0.16 * w:
                WARN.append("short last line (%.1f of %.1f cm): %s" % (ln[-1], w, s[:50]))
    tb = slide.shapes.add_textbox(Cm(x), Cm(y), Cm(w), Cm(need + 0.05))
    if name:
        tb.name = name
    tf = tb.text_frame
    tf.word_wrap = True
    tf.auto_size = None
    tf.vertical_anchor = MSO_ANCHOR.TOP
    for side in ("margin_left", "margin_right", "margin_top", "margin_bottom"):
        setattr(tf, side, 0)
    _fill(tf, paras, size, color, pitch, after, align, bold, italic, face)
    return need


def box(x, y, w, h, fill=WHITE, line=None, lw=1.0, name=None, shape=MSO_SHAPE.RECTANGLE):
    if DRY:
        return None
    s = slide.shapes.add_shape(shape, Cm(x), Cm(y), Cm(w), Cm(h))
    if fill is None:
        s.fill.background()
    else:
        s.fill.solid()
        s.fill.fore_color.rgb = fill
    if line is None:
        s.line.fill.background()
    else:
        s.line.color.rgb = line
        s.line.width = Pt(lw)
    s.shadow.inherit = False
    if name:
        s.name = name
    return s


def rule(x1, y1, x2, y2, color=INK, w=1.0, name=None):
    if DRY:
        return None
    c = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Cm(x1), Cm(y1), Cm(x2), Cm(y2))
    st = c._element.find(qn("p:style"))
    if st is not None:
        c._element.remove(st)
    c.line.color.rgb = color
    c.line.width = Pt(w)
    if name:
        c.name = name
    return c


def badge(x, y, d, n, size, name):
    b = box(x, y, d, d, fill=NAVY, shape=MSO_SHAPE.OVAL, name=name)
    if b is None:
        return
    tf = b.text_frame
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    for side in ("margin_left", "margin_right", "margin_top", "margin_bottom"):
        setattr(tf, side, 0)
    _fill(tf, [str(n)], size, WHITE, size * 1.1, 0, PP_ALIGN.CENTER, True)


def fig_size(name):
    im = Image.open(FIG / name)
    dpi = im.info["dpi"][0]
    return im.size[0] / dpi * 2.54, im.size[1] / dpi * 2.54


def pic(name, x, y, alt):
    pw, ph = fig_size(name)
    if abs(pw - CW) > 0.02:
        WARN.append("figure %s is %.2f cm wide, not one column" % (name, pw))
    if DRY:
        return ph
    p = slide.shapes.add_picture(str(FIG / name), Cm(x), Cm(y), Cm(pw), Cm(ph))
    p.name = "fig." + name.replace(".png", "")
    p._element.nvPicPr.cNvPr.set("descr", alt)
    return ph


def measure(fn, *a, **k):
    global DRY
    old, DRY = DRY, True
    try:
        return fn(*a, **k)
    finally:
        DRY = old


# ====================================================================================== type sizes
def set_sizes(b):
    global BODY_PT, BODY_PITCH, TAKE_PT, TAKE_PITCH, TAB_PT, TAB_PITCH, CAP_PT, CAP_PITCH, HEAD_PT, PARA_AFTER
    global GROUP_PT
    BODY_PT, BODY_PITCH = b, round(b * 1.3, 1)
    TAKE_PT = round(b * 1.08, 1)
    TAKE_PITCH = round(TAKE_PT * 1.3, 1)
    TAB_PT = max(20.0, round(b * 0.84, 1))
    TAB_PITCH = round(TAB_PT * 1.22, 1)
    GROUP_PT = TAB_PT
    CAP_PT = max(18.0, round(b * 0.75, 1))
    CAP_PITCH = round(CAP_PT * 1.28, 1)
    HEAD_PT = round(b * 1.75, 1)
    PARA_AFTER = round(b * 0.45, 1)


BODY_STEPS = (28.0, 27.5, 27.0, 26.5, 26.0, 25.5, 25.0, 24.5, 24.0)
GB, GS = 0.6, 1.4                # minimum gap between blocks, between sections (cm)
CAP_GAP = 0.3


def body(x, y, w, paras, **kw):
    kw.setdefault("after", PARA_AFTER)
    return text(x, y, w, paras, BODY_PT, pitch=BODY_PITCH, **kw)


def caption(x, y, w, s, name=None):
    return text(x, y, w, s, CAP_PT, color=MUTED, pitch=CAP_PITCH, italic=True, align=PP_ALIGN.CENTER, name=name)


def heading(x, y, w, n, title, key):
    nw = HEAD_PT * 0.95 * PT2CM
    text(x, y, nw, str(n), HEAD_PT, color=NAVY, bold=True, pitch=HEAD_PT * 1.15, name=f"{key}.num", check=False)
    text(x + nw, y, w - nw, title, HEAD_PT, color=NAVY, bold=True, pitch=HEAD_PT * 1.15, name=f"{key}.title")
    ry = y + HEAD_PT * 1.15 * PT2CM + 0.25
    rule(x, ry, x + w, ry, color=NAVY, w=2.25, name=f"{key}.rule")
    return ry - y + 0.65


def takeaway(x, y, w, s, key):
    bar, padx, pady = 0.42, 0.7, 0.55
    tw = w - bar - 2 * padx
    th = text_h(s, TAKE_PT, tw, TAKE_PITCH, bold=True)
    h = th + 2 * pady
    box(x, y, w, h, fill=TINT, name=f"{key}.box")
    box(x, y, bar, h, fill=NAVY, name=f"{key}.bar")
    text(x + bar + padx, y + pady, tw, s, TAKE_PT, color=NAVY, pitch=TAKE_PITCH, bold=True, name=f"{key}.text")
    return h


def items(x, y, w, paras, glyph, key, gap=None, color=NAVY):
    gap = PARA_AFTER * PT2CM if gap is None else gap
    indent = BODY_PT * 0.95 * PT2CM
    yy = y
    for i, s in enumerate(paras):
        text(x, yy, indent, glyph, BODY_PT, color=color, pitch=BODY_PITCH, bold=True, name=f"{key}.g{i}",
             check=False)
        yy += text(x + indent, yy, w - indent, s, BODY_PT, pitch=BODY_PITCH, name=f"{key}.t{i}")
        if i < len(paras) - 1:
            yy += gap
    return yy - y


def booktabs(x, y, cols, header, rows, key, aligns=None, fills=(), padx=0.18, pady=0.105, head_bold=True):
    """Thick top and bottom rules, thin rule under the header; a str row is an italic group label.
    Row indices in fills get a light shade."""
    pt, pitch = TAB_PT, TAB_PITCH
    aligns = aligns or [PP_ALIGN.LEFT] + [PP_ALIGN.CENTER] * (len(cols) - 1)
    xs = [x + sum(cols[:j]) for j in range(len(cols))]
    W = sum(cols)
    yy = y
    rule(x, yy, x + W, yy, w=2.0, name=f"{key}.top")
    yy += 0.1

    def row(cells, tag, fill=False, bold=False, color=INK):
        nonlocal yy
        hs = [text_h(c, pt, cols[j] - 2 * padx, pitch, bold=bold) for j, c in enumerate(cells)]
        rh = max(hs)
        if fill:
            box(x, yy, W, rh + 2 * pady, fill=SHADE, name=f"{key}.{tag}.fill")
        for j, c in enumerate(cells):
            text(xs[j] + padx, yy + pady + (rh - hs[j]) / 2, cols[j] - 2 * padx, c, pt, color=color, pitch=pitch,
                 align=aligns[j], bold=bold, name=f"{key}.{tag}c{j}", check=False)
        yy += rh + 2 * pady

    row(header, "h", bold=head_bold, color=NAVY)
    rule(x, yy, x + W, yy, w=1.0, name=f"{key}.mid")
    yy += 0.08
    for i, cells in enumerate(rows):
        if isinstance(cells, str):
            if i > 0:
                yy += 0.08
            gh = text_h(cells, GROUP_PT, W - 2 * padx, pitch, italic=True)
            text(x + padx, yy + pady * 0.7, W - 2 * padx, cells, GROUP_PT, color=NAVY, pitch=pitch, italic=True,
                 name=f"{key}.g{i}", check=False)
            yy += gh + pady * 1.2
            continue
        row(cells, f"r{i}", fill=i in fills)
    yy += 0.06
    rule(x, yy, x + W, yy, w=2.0, name=f"{key}.bottom")
    return yy - y


def table_block(x, y, w, cap, cols, head, rows, key, fills=(), note=None, aligns=None):
    ch = caption(x, y, w, cap, name=f"{key}.caption")
    yy = y + ch + CAP_GAP
    yy += booktabs(x, yy, cols, head, rows, key, fills=fills, aligns=aligns)
    if note:
        yy += 0.2
        yy += text(x, yy, w, note, CAP_PT, color=MUTED, pitch=CAP_PITCH, italic=True, align=PP_ALIGN.CENTER,
                   name=f"{key}.note")
    return yy - y


def value_col(rows, extra=0.5):
    vals = [c for r in rows if not isinstance(r, str) for c in r[1:]]
    return max(text_w(v, TAB_PT) for v in vals) / FIT + 2 * 0.18 + extra


# ====================================================================================== blocks
def figure(name, cap, alt, key):
    def blk(x, y, w):
        ph = pic(name, x, y, alt)
        ch = caption(x, y + ph + CAP_GAP, w, cap, name=f"{key}.caption")
        return ph + CAP_GAP + ch
    return blk


def s1_text(x, y, w):
    return body(x, y, w, S1, name="s1.text")


def s1_box(x, y, w):
    return takeaway(x, y, w, S1_BOX, "s1.take")


def s2_data(x, y, w):
    return body(x, y, w, S2_DATA, name="s2.data")


def s2_row(x, y, w):
    return body(x, y, w, S2_ROW, name="s2.row")


def s2_target(x, y, w):
    return body(x, y, w, S2_TARGET, name="s2.target")


def s3_refs(x, y, w):
    c0 = max(text_w(part, TAB_PT) for r in TAB1_ROWS for part in r[0].split(chr(10))) / FIT + 0.6
    h = table_block(x, y, w, TAB1_CAP, (c0, w - c0), TAB1_HEAD, TAB1_ROWS, "tab1",
                    aligns=(PP_ALIGN.LEFT, PP_ALIGN.LEFT))
    h += 0.25
    h += text(x, y + h, w, SKILL_NOTE, CAP_PT, color=MUTED, pitch=CAP_PITCH, align=PP_ALIGN.CENTER,
              name="s3.skill.note")
    return h


def s3_score(x, y, w):
    return body(x, y, w, S3_SCORE, name="s3.score")


def s4_table(x, y, w):
    c0 = max(text_w(r[0], TAB_PT) for r in TAB2_ROWS if not isinstance(r, str)) / FIT + 0.6
    cv = (w - c0) / 2
    return table_block(x, y, w, TAB2_CAP, (c0, cv, cv), TAB2_HEAD, TAB2_ROWS, "tab2", fills=TAB2_FILL,
                       note=TAB_BOLD_NOTE)


def s4_items(x, y, w):
    return items(x, y, w, S4_ITEMS, "•", "s4.b")


def s5_table(x, y, w):
    cv = value_col(TAB3_ROWS)
    return table_block(x, y, w, TAB3_CAP, (w - cv, cv), TAB3_HEAD, TAB3_ROWS, "tab3", fills=TAB3_FILL,
                       aligns=(PP_ALIGN.LEFT, PP_ALIGN.CENTER))


def s5_items(x, y, w):
    return items(x, y, w, S5_ITEMS, "•", "s5.b")


def s6_table(x, y, w):
    cv = value_col(TAB4_ROWS)
    return table_block(x, y, w, TAB4_CAP, (w - cv, cv), TAB4_HEAD, TAB4_ROWS, "tab4", fills=TAB4_FILL,
                       note=TAB4_NOTE, aligns=(PP_ALIGN.LEFT, PP_ALIGN.CENTER))


def s6_items(x, y, w):
    return items(x, y, w, S6_ITEMS, "•", "s6.b")


def s7_box(x, y, w):
    return takeaway(x, y, w, S7_BOX, "s7.take")


def s7_checks(x, y, w):
    return items(x, y, w, S7_CHECKS, "✓", "s7.c")


def s7_caveat(x, y, w):
    return body(x, y, w, S7_CAVEAT, name="s7.caveat")


def s8_limits(x, y, w):
    bd = BODY_PT * 1.25 * PT2CM
    ix = x + bd + 0.45
    iw = x + w - ix
    yy = y
    for i, (head, fix) in enumerate(S8_ITEMS):
        badge(x, yy + 0.05, bd, i + 1, BODY_PT * 0.85, f"s8.item{i}.badge")
        yy += text(ix, yy, iw, head, BODY_PT, pitch=BODY_PITCH, name=f"s8.item{i}.text") + 0.08
        yy += text(ix, yy, iw, fix, BODY_PT, color=NAVY, pitch=BODY_PITCH, name=f"s8.item{i}.fix")
        if i == 0:
            yy += 0.45
            cv = value_col(TAB5_ROWS)
            yy += table_block(ix, yy, iw, TAB5_CAP, (iw - cv, cv), TAB5_HEAD, TAB5_ROWS, "tab5",
                              aligns=(PP_ALIGN.LEFT, PP_ALIGN.CENTER))
        if i < len(S8_ITEMS) - 1:
            yy += 0.55
    return yy - y


FIG_ALT = {
    "geometry": "Positions on the major radius: the CES chord at R = 2.198 m inside the band of published "
                "pedestal tops, the BES and ECEI columns of the files around it",
    "rows": "Schematic of a row: one raw fast-channel sample at the stamp, the CES frame starting at the stamp, "
            "a beam blip leaving out one frame",
    "schematic": "Block diagram of the two-stage estimator and of the five-parameter filter",
    "trace": "Reported T_i and the hold, causal GP and network estimates in three test discharges",
    "blips": "Ratios of BES level, ECEI level and reported T_i in the ten rows after a beam blip",
}

SECTIONS = {
    1: ("Motivation", "s1", [s1_text, s1_box]),
    2: ("Diagnostics and data", "s2",
        [s2_data, figure("poster_geometry.png", FIG1_CAP, FIG_ALT["geometry"], "fig1"), s2_row,
         figure("poster_rows.png", FIG2_CAP, FIG_ALT["rows"], "fig2"), s2_target,
         figure("poster_trace.png", FIG3_CAP, FIG_ALT["trace"], "fig3")]),
    3: ("Estimators and scoring", "s3",
        [figure("poster_schematic.png", FIG4_CAP, FIG_ALT["schematic"], "fig4"), s3_refs, s3_score]),
    4: ("Main result", "s4", [s4_table, s4_items]),
    5: ("What carries the information", "s5", [s5_table, s5_items]),
    6: ("Where the gain is; fit or plasma", "s6",
        [figure("poster_blips.png", FIG5_CAP, FIG_ALT["blips"], "fig5"), s6_table, s6_items]),
    7: ("Conclusion", "s7", [s7_box, s7_checks, s7_caveat]),
    8: ("Limitations and outlook", "s8", [s8_limits]),
}
COLUMNS = [(1, 2), (3, 4, 5), (6, 7, 8)]


def col_content(x, secs):
    hh = sum(measure(heading, x, 0, CW, n, SECTIONS[n][0], SECTIONS[n][1]) for n in secs)
    hb = [[measure(b, x, 0, CW) for b in SECTIONS[n][2]] for n in secs]
    return hh + sum(map(sum, hb)), sum(len(b) - 1 for b in hb), len(secs) - 1


def draw_column(x, secs, y0, y1, label):
    content, nb, ns = col_content(x, secs)
    free = (y1 - y0) - content - nb * GB - ns * GS
    share = free / (nb + 2 * ns) if (nb + 2 * ns) else 0.0
    print("%s column: content %.1f cm, free %.1f cm -> +%.2f cm per block gap" % (label, content, free, share))
    y = y0
    for i, n in enumerate(secs):
        title, key, blks = SECTIONS[n]
        y += heading(x, y, CW, n, title, key)
        for j, blk in enumerate(blks):
            y += blk(x, y, CW)
            if j < len(blks) - 1:
                y += GB + share
        if i < ns:
            y += GS + 2 * share
    return free, y


# ====================================================================================== title band
def title_band():
    tw = W_CM - 2 * M
    tpt = 80.0
    lines = TITLE_LINES
    widest = max(_w(t, tpt, "sb") for t in lines)
    if widest > tw * 0.985:
        WARN.append("title line %.1f cm wider than the band text width %.1f cm" % (widest, tw))
    n_t = len(lines)
    t_h = n_t * tpt * 1.08 * PT2CM
    spt = 36.0
    while len(wrap(SUBTITLE, spt, tw)) > 1 and spt > 34:
        spt -= 1
    s_h = text_h(SUBTITLE, spt, tw, spt * 1.25)
    apt, dpt = 34.0, 26.0
    band_h = 1.4 + t_h + 0.5 + s_h + 0.65 + apt * 1.25 * PT2CM + 0.2 + dpt * 1.25 * PT2CM + 1.15
    box(0, 0, W_CM, band_h, fill=NAVY, name="title.band")
    box(0, band_h - 0.3, W_CM, 0.3, fill=ACCENT, name="title.accent")
    y = 1.4
    if not DRY:
        tb = slide.shapes.add_textbox(Cm(M), Cm(y), Cm(tw), Cm(t_h + 0.1))
        tb.name = "title"
        tf = tb.text_frame
        tf.word_wrap = True
        for side in ("margin_left", "margin_right", "margin_top", "margin_bottom"):
            setattr(tf, side, 0)
        _fill(tf, ["\n".join(lines)], tpt, WHITE, tpt * 1.08, 0, PP_ALIGN.CENTER, False, face=FONT_SB)
    y += t_h + 0.5
    y += text(M, y, tw, SUBTITLE, spt, color=SUBTLE, pitch=spt * 1.25, align=PP_ALIGN.CENTER, name="subtitle",
              check=False) + 0.65
    y += text(M, y, tw, AUTHOR, apt, color=WHITE, pitch=apt * 1.25, align=PP_ALIGN.CENTER, name="author",
              check=False) + 0.2
    text(M, y, tw, DATA_LINE, dpt, color=SUBTLE, pitch=dpt * 1.25, align=PP_ALIGN.CENTER, name="data.line",
         check=False)
    print("title %d pt (%d lines, widest %.1f cm of %.1f), subtitle %d pt, band %.1f cm"
          % (tpt, n_t, widest, tw, spt, band_h))
    return band_h


# ====================================================================================== draw
band_h = title_band()
Y0 = band_h + 1.35
REF_PT = 18.0
foot_h = text_h(["**References**  " + REFS, "**Acknowledgement**  " + ACK], REF_PT, W_CM - 2 * M, REF_PT * 1.3, 4)
FOOT_Y = H_CM - 1.3 - foot_h - 0.45
Y1 = FOOT_Y - 0.9

best = None
for b_ in BODY_STEPS:
    set_sizes(b_)
    frees = []
    for x, secs in zip(XS, COLUMNS):
        c, nb, ns = col_content(x, secs)
        frees.append((Y1 - Y0) - c - nb * GB - ns * GS)
    print("body %.1f pt: free per column %s" % (b_, ", ".join("%.1f" % f for f in frees)))
    if min(frees) >= 0:
        best = b_
        break
if best is None:
    best = BODY_STEPS[-1]
    WARN.append("columns overfull even at the smallest body size")
set_sizes(best)
print("body %.1f pt, tables %.1f, captions %.1f, takeaway %.1f, headings %.1f" %
      (BODY_PT, TAB_PT, CAP_PT, TAKE_PT, HEAD_PT))

ends = []
for k, (x, secs) in enumerate(zip(XS, COLUMNS)):
    free, yend = draw_column(x, secs, Y0, Y1, "col%d" % (k + 1))
    ends.append(yend)
for k in range(2):
    xm = XS[k] + CW + GUT / 2
    rule(xm, Y0 + 0.2, xm, Y1, color=RULE, w=1.25, name=f"column.divider{k}")

rule(M, FOOT_Y, W_CM - M, FOOT_Y, color=RULE, w=1.25, name="footer.rule")
text(M, FOOT_Y + 0.45, W_CM - 2 * M, ["**References**  " + REFS, "**Acknowledgement**  " + ACK], REF_PT,
     color=MUTED, pitch=REF_PT * 1.3, after=4, name="footer.text", check=False)
print("column ends: %s (target %.2f)" % (", ".join("%.2f" % e for e in ends), Y1))
for w_ in dict.fromkeys(WARN):
    print("  WARN", w_)

prs.core_properties.title = "Causal nowcasting of the CES ion temperature on a KSTAR edge chord (A0 poster)"
prs.core_properties.author = "" if args.anon else "Seungsang Lee"
OUT.parent.mkdir(parents=True, exist_ok=True)
prs.save(OUT)
print("saved", OUT)


def export_png(pptx, png, px):
    """Export the slide and, from PowerPoint's own layout, compare every text's height with its box:
    text taller than its box would overflow; a box much taller than its text means the metrics drifted."""
    import win32com.client
    app = win32com.client.Dispatch("PowerPoint.Application")
    pres = app.Presentations.Open(str(Path(pptx).resolve()), True, False, False)
    over, slack = [], []
    try:
        pres.Slides(1).Export(str(Path(png).resolve()), "PNG", px, round(px * H_CM / W_CM))
        for sh in pres.Slides(1).Shapes:
            if sh.HasTextFrame and sh.TextFrame.HasText:
                d = sh.TextFrame.TextRange.BoundHeight - sh.Height
                if d > 2.0:
                    over.append((sh.Name, round(d, 1)))
                elif d < -12.0:
                    slack.append((sh.Name, round(-d, 1)))
    finally:
        pres.Close()
        if app.Presentations.Count == 0:
            app.Quit()
    print("PowerPoint layout: %d text(s) taller than their box, %d box(es) taller than their text by > 12 pt"
          % (len(over), len(slack)))
    for o in over + slack:
        print("  ", o)


if not args.no_png:
    export_png(OUT, PNG, args.png_px)
    print("png", PNG)
