# -*- coding: utf-8 -*-
"""KSTAR CES nowcasting 포스터 (세로 45 x 60 cm; 200 % 로 인쇄하면 90 x 120 cm).

  py docs/presentation/make_figures_poster.py               # 그림 먼저 (figures/poster_20261119/)
  py docs/presentation/build_poster.py                      # -> KSTAR_CES_포스터_초안.pptx + .png
  py docs/presentation/check_poster_numbers.py              # 포스터의 모든 숫자를 원천과 대조
  py docs/presentation/build_poster.py --anon --out X.pptx  # 저자 줄을 "발표자 · 소속"으로 바꾼 사본

* 틀: 남색 제목 띠, 그 아래 핵심 수치 네 칸, 두 단(1–3 | 4–7)과 가운데 가는 세로선. 구역은 상자 없이
  번호 + 굵은 남색 제목 + 단 너비의 선으로 나누고, 요점은 왼쪽에 굵은 남색 막대가 있는 연한 상자,
  표는 booktabs(위아래 굵은 선, 머리 아래 가는 선), 그림·표 설명은 회색 기울임.
* 내용과 수치는 2026-11-19 포스터 명세(두 단계 estimator, B.64·B.65)를 그대로 옮긴다.
  그림 두 장은 make_figures_poster.py 가 포스터에 놓이는 실제 크기(그림 안 글자 10 pt 이상)로 그린다.
* 글상자 높이는 Pretendard 실제 글자 폭으로 줄 수를 세어 정하고(줄 간격은 pt 고정), 두 단은 한 번
  재어 본 뒤 남는 높이를 간격에 나눠 아래 끝을 맞춘다.
* 도형은 전부 PowerPoint 기본 도형과 글상자이다 — 받아서 그대로 고칠 수 있다.
* PNG 미리보기는 PowerPoint COM(Slides(1).Export)으로 내보낸다. 열린 문서가 없을 때만 Quit 한다.
"""
import argparse
import io
import os
import re
import sys
from pathlib import Path

from PIL import Image, ImageFont
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.dml import MSO_LINE
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Cm, Pt

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures" / "poster_20261119"      # make_figures_poster.py 가 포스터 크기로 그린 그림

ap = argparse.ArgumentParser()
ap.add_argument("--anon", action="store_true", help='저자 줄을 "발표자 · 소속"으로 (웹 미리보기용, --out 필요)')
ap.add_argument("--out", type=Path, help="pptx 경로 (기본: docs/presentation/KSTAR_CES_포스터_초안.pptx)")
ap.add_argument("--png", type=Path, help="PNG 경로 (기본: pptx 와 같은 이름)")
ap.add_argument("--png-px", type=int, default=1800, help="PNG 가로 픽셀")
ap.add_argument("--no-png", action="store_true")
args = ap.parse_args()
if args.anon and args.out is None:
    sys.exit("--anon needs --out (the anonymous copy never overwrites the poster)")
OUT = args.out or HERE / "KSTAR_CES_포스터_초안.pptx"
PNG = args.png or OUT.with_suffix(".png")
AUTHOR = "발표자 · 소속" if args.anon else "**이승상**  ·  서울대학교 원자핵공학과"

W_CM, H_CM = 45.0, 60.0
FONT, FONT_XB = "Pretendard", "Pretendard ExtraBold"


def rgb(h):
    return RGBColor(int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


NAVY, INK, BODY, MUTED = rgb("1B2F5E"), rgb("15181E"), rgb("23272F"), rgb("5E6878")
TINT, RULE, WHITE, SUBTLE = rgb("E9EDF4"), rgb("B9C1CE"), rgb("FFFFFF"), rgb("C9D3E6")
BLUE, TINT_B, PANEL = rgb("2A78D6"), rgb("E3EDFA"), rgb("F1F3F6")
FRAME, PREV, BEAM = rgb("E6E9EE"), rgb("D3DAE5"), rgb("9DB7D8")

# ------------------------------------------------------------------ 글자 폭 계측 (Pretendard)
_uf = Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft" / "Windows" / "Fonts"


def _load(name):
    p = _uf / name
    return ImageFont.truetype(str(p if p.exists() else Path(r"C:\Windows\Fonts\malgun.ttf")), 200)


F_REG, F_BOLD, F_XB = _load("Pretendard-Regular.otf"), _load("Pretendard-Bold.otf"), _load("Pretendard-ExtraBold.otf")
PT2CM = 2.54 / 72.0
NB = "\u00a0"
SMALL_F = 0.80          # {작은 회색} 글자의 크기 비 (10 pt 아래로는 내리지 않는다)
FIT = 0.995             # 한 줄에 채우는 폭의 비. PowerPoint 의 실제 폭은 PIL 계측의 0.996 배(시험으로 확인)
FRAGILE = []            # 줄바꿈이 폭 끝에 걸린 글(PowerPoint 와 갈릴 수 있음)


def small_pt(pt):
    return max(10.0, pt * SMALL_F)
TOK = re.compile(r"(\*\*|\{|\}|T\u0302?_i|\^\(1\))")


def glue(s):
    """구간 [a, b], 범위 a … b, 숫자와 단위를 한 줄에 묶는다(줄바꿈 없는 공백)."""
    s = re.sub(r"\[[^\]]*\]", lambda m: m.group(0).replace(" ", NB), s)
    s = s.replace(" … ", NB + "…" + NB)
    s = re.sub(r"(?<=\d) (?=%)", NB, s)
    return re.sub(r"(?<=\d) (?=(?:ms|eV|keV|kHz|MHz|µs|m)(?![A-Za-z]))", NB, s)


def parse(s, bold=False):
    """'**굵게**', '{작은 회색}', 'T_i'(아래 첨자), '^(1)'(위 첨자) 표기를 (글자, 굵기, 작음, 첨자) 조각으로."""
    out, b, sm = [], bold, False
    for part in TOK.split(glue(s)):
        if not part:
            continue
        if part == "**":
            b = not b
        elif part == "{":
            sm = True
        elif part == "}":
            sm = False
        elif part.endswith("_i"):
            out += [(part[:-2], b, sm, 0), ("i", b, sm, -1)]
        elif part == "^(1)":
            out.append(("(1)", b, sm, 1))
        else:
            out.append((part, b, sm, 0))
    return out


def _w(s, pt, bold, face=None):
    f = F_XB if face == FONT_XB else (F_BOLD if bold else F_REG)
    return f.getlength(s) / 200.0 * pt * PT2CM


def wrap(s, pt, width, bold=False, face=None):
    """낱말 단위 줄바꿈(한글 낱말을 자르지 않음; '\n' 은 강제 줄바꿈). 줄마다 차지한 폭(cm)을 돌려준다."""
    words, cur, forced = [], 0.0, False
    for txt, b, sm, sh in parse(s, bold):
        size = (small_pt(pt) if sm else pt) * (0.66 if sh else 1.0)
        for atom in re.split(r"( |\n)", txt):
            if atom == "":
                continue
            if atom in (" ", "\n"):
                words.append((cur, forced))
                cur, forced = 0.0, atom == "\n"
            else:
                cur += _w(atom, size, b, face)
    words.append((cur, forced))
    space = _w(" ", pt, bold, face)
    lines, x = [], 0.0
    for w, brk in words:
        if brk:
            lines.append(x)
            x = w
        elif x > 0 and x + space + w > width * FIT:
            if x + space + w <= width / 0.99:
                FRAGILE.append(s)
            lines.append(x)
            x = w
        else:
            x = x + (space if x > 0 else 0) + w
    lines.append(x)
    return lines


def text_h(paras, pt, width, pitch, after=0.0, bold=False, face=None):
    paras = [paras] if isinstance(paras, str) else paras
    return (sum(len(wrap(s, pt, width, bold, face)) for s in paras) * pitch + after * (len(paras) - 1)) * PT2CM


def text_w(s, pt, bold=False, face=None):
    return wrap(s, pt, 1e9, bold, face)[0]


# ------------------------------------------------------------------ 기본 도형 (DRY 이면 재기만 한다)
prs = Presentation()
prs.slide_width, prs.slide_height = Cm(W_CM), Cm(H_CM)
slide = prs.slides.add_slide(prs.slide_layouts[6])
DRY = False
WARN = []


def _font(run, size, color, bold=False, italic=False, shift=0, face=FONT):
    f = run.font
    f.size = Pt(size)
    f.bold = bold
    f.italic = italic
    f.color.rgb = color
    f.name = face
    rpr = run._r.get_or_add_rPr()
    rpr.set("lang", "ko-KR")       # 한글 낱말을 자르지 않는 줄바꿈은 lang = ko-KR 과 latinLnBrk = 0 이 함께 있어야 듣는다
    rpr.set("altLang", "en-US")
    latin = rpr.find(qn("a:latin"))
    for tag in ("a:cs", "a:ea"):
        latin.addnext(rpr.makeelement(qn(tag), {"typeface": face}))
    if shift:
        rpr.set("baseline", "-25000" if shift < 0 else "30000")


def _fill_paras(tf, paras, size, color, pitch, after, align, bold, italic=False, face=FONT):
    for i, s in enumerate(paras):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.line_spacing = Pt(pitch)
        p.space_after = Pt(after if i < len(paras) - 1 else 0)
        p._p.get_or_add_pPr().set("latinLnBrk", "0")
        for txt, b, sm, sh in parse(s, bold):
            for k, piece in enumerate(txt.split("\n")):
                if k:
                    p.add_line_break()
                if piece:
                    r = p.add_run()
                    r.text = piece
                    _font(r, small_pt(size) if sm else size, MUTED if sm else color, b, italic, sh, face)


def text(x, y, w, paras, size, color=BODY, pitch=None, after=0.0, align=PP_ALIGN.LEFT, bold=False, italic=False,
         face=FONT, name=None, check=True):
    """글상자. 높이는 줄 수로 계산해 돌려준다(cm)."""
    paras = [paras] if isinstance(paras, str) else paras
    pitch = pitch or round(size * 1.4, 1)
    need = text_h(paras, size, w, pitch, after, bold, face)
    if DRY:
        return need
    for s in paras:
        FRAGILE.clear()
        ln = wrap(s, size, w, bold, face)
        if FRAGILE:
            WARN.append("break at the box edge (%.1f cm box): %s" % (w, s[:40]))
        if check and len(ln) > 1 and ln[-1] < 0.12 * w:
            WARN.append("short last line (%.1f cm of %.1f): %s" % (ln[-1], w, s[:40]))
    tb = slide.shapes.add_textbox(Cm(x), Cm(y), Cm(w), Cm(need))
    if name:
        tb.name = name
    tf = tb.text_frame
    tf.word_wrap = True
    tf.auto_size = None
    tf.vertical_anchor = MSO_ANCHOR.TOP
    for side in ("margin_left", "margin_right", "margin_top", "margin_bottom"):
        setattr(tf, side, 0)
    _fill_paras(tf, paras, size, color, pitch, after, align, bold, italic, face)
    return need


def box(x, y, w, h, fill=WHITE, line=None, lw=0.75, radius=0.28, name=None, dash=False,
        shape=MSO_SHAPE.RECTANGLE):
    if DRY:
        return None
    s = slide.shapes.add_shape(shape, Cm(x), Cm(y), Cm(w), Cm(h))
    if shape == MSO_SHAPE.ROUNDED_RECTANGLE:
        s.adjustments[0] = min(0.5, radius / min(w, h))
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
        if dash:
            s.line.dash_style = MSO_LINE.DASH
    s.shadow.inherit = False
    if name:
        s.name = name
    return s


def label_in(shape, paras, size, color=INK, bold=False, pitch=None, align=PP_ALIGN.CENTER, pad=0.12, face=FONT):
    """도형 안 글자."""
    if shape is None:
        return
    paras = [paras] if isinstance(paras, str) else paras
    tf = shape.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    for side in ("margin_left", "margin_right", "margin_top", "margin_bottom"):
        setattr(tf, side, Cm(pad))
    _fill_paras(tf, paras, size, color, pitch or round(size * 1.3, 1), 0, align, bold, face=face)


def line(x1, y1, x2, y2, color=INK, w=1.0, head=False, dash=False, name=None):
    if DRY:
        return None
    c = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Cm(x1), Cm(y1), Cm(x2), Cm(y2))
    st = c._element.find(qn("p:style"))      # 기본 선 스타일(그림자 포함)을 떼고 색과 굵기만 준다
    if st is not None:
        c._element.remove(st)
    c.line.color.rgb = color
    c.line.width = Pt(w)
    if dash:
        c.line.dash_style = MSO_LINE.DASH
    if head:
        ln = c.line._get_or_add_ln()
        ln.append(ln.makeelement(qn("a:tailEnd"), {"type": "triangle", "w": "med", "len": "med"}))
    if name:
        c.name = name
    return c


def badge(x, y, d, n, size, name):
    b = box(x, y, d, d, fill=NAVY, shape=MSO_SHAPE.OVAL, name=name)
    label_in(b, str(n), size, color=WHITE, bold=True, pad=0.0)


def fig_size(name):
    """그림은 포스터 위 실제 크기로 그려져 있다: 픽셀 / dpi 로 폭과 높이(cm)."""
    im = Image.open(FIG / name)
    dpi = im.info["dpi"][0]
    return im.size[0] / dpi * 2.54, im.size[1] / dpi * 2.54


def pic(name, x, y, alt):
    """그림을 제 크기로 놓고 높이를 돌려준다."""
    pw, ph = fig_size(name)
    if DRY:
        return ph
    p = slide.shapes.add_picture(str(FIG / name), Cm(x), Cm(y), Cm(pw), Cm(ph))
    p.name = "fig." + name.replace(".png", "")
    p._element.nvPicPr.cNvPr.set("descr", alt)
    return ph


# ------------------------------------------------------------------ 격자와 글자 크기
M = 1.5                       # 바깥 여백
GUT = 1.6                     # 단 사이 (가운데 세로선)
COL_W = (W_CM - 2 * M - GUT) / 2
XL, XR = M, M + COL_W + GUT
BAND_H = 6.4
STRIP_Y, STRIP_H = 7.0, 3.6
Y0, Y1 = 11.5, 57.15          # 두 단의 위아래
FOOT_Y = 57.6

HEAD_PT = 25.0
BODY_PT, BODY_PITCH = 13.2, 18.6
TAKE_PT, TAKE_PITCH = 15.5, 21.0
TAB_PT, TAB_PITCH = 12.6, 17.0
CAP_PT, CAP_PITCH = 11.2, 15.0
DIA_PT = 11.0                 # 도형 안 글자
PARA_AFTER = 7.0              # 문단 사이, pt
GB, GS = 0.45, 1.0            # 블록 사이, 구역 사이 (cm, 남는 높이를 더 받는다)


def body(x, y, w, paras, **kw):
    kw.setdefault("after", PARA_AFTER)
    return text(x, y, w, paras, BODY_PT, pitch=BODY_PITCH, **kw)


def caption(x, y, w, s, align=PP_ALIGN.CENTER, name=None):
    return text(x, y, w, s, CAP_PT, color=MUTED, pitch=CAP_PITCH, italic=True, align=align, name=name)


def heading(x, y, w, n, title, key):
    text(x, y, 1.3, str(n), HEAD_PT, color=NAVY, bold=True, pitch=HEAD_PT * 1.2, name=f"{key}.num")
    text(x + 1.3, y, w - 1.3, title, HEAD_PT, color=NAVY, bold=True, pitch=HEAD_PT * 1.2, name=f"{key}.title")
    ry = y + HEAD_PT * 1.2 * PT2CM + 0.2
    line(x, ry, x + w, ry, color=NAVY, w=1.75, name=f"{key}.rule")
    return ry - y + 0.5


def takeaway(x, y, w, s, key, pt=TAKE_PT, pitch=TAKE_PITCH):
    bar, padx, pady = 0.24, 0.55, 0.4
    tw = w - bar - 2 * padx
    th = text_h(s, pt, tw, pitch, bold=True)
    h = th + 2 * pady
    box(x, y, w, h, fill=TINT, name=f"{key}.box")
    box(x, y, bar, h, fill=NAVY, name=f"{key}.bar")
    text(x + bar + padx, y + pady, tw, s, pt, color=NAVY, pitch=pitch, bold=True, name=f"{key}.text")
    return h


def items(x, y, w, paras, glyph, key, gap=0.22, color=NAVY, indent=0.55, pt=None, pitch=None, glyph_pt=None):
    """글머리 목록: 기호 + 들여 쓴 문단. 높이를 돌려준다."""
    pt, pitch = pt or BODY_PT, pitch or BODY_PITCH
    yy = y
    for i, s in enumerate(paras):
        text(x, yy, indent, glyph, glyph_pt or pt, color=color, pitch=pitch, bold=True, name=f"{key}.g{i}", check=False)
        yy += text(x + indent, yy, w - indent, s, pt, pitch=pitch, name=f"{key}.t{i}")
        if i < len(paras) - 1:
            yy += gap
    return yy - y


def booktabs(x, y, cols, header, rows, key, aligns=None, mid_rules=(), pt=TAB_PT, pitch=TAB_PITCH, padx=0.16,
             pady=0.11):
    """위아래 굵은 선, 머리 아래 가는 선. rows 의 칸은 '**굵게**', '{작은 회색}' 표기를 쓴다."""
    aligns = aligns or [PP_ALIGN.LEFT] + [PP_ALIGN.CENTER] * (len(cols) - 1)
    xs = [x + sum(cols[:j]) for j in range(len(cols))]
    W = sum(cols)
    yy = y
    line(x, yy, x + W, yy, w=1.6, name=f"{key}.top")
    yy += 0.12

    def row(cells, tag):
        nonlocal yy
        hs = [text_h(c, pt, cols[j] - 2 * padx, pitch) for j, c in enumerate(cells)]
        rh = max(hs)
        for j, c in enumerate(cells):
            text(xs[j] + padx, yy + pady + (rh - hs[j]) / 2, cols[j] - 2 * padx, c, pt, color=INK, pitch=pitch,
                 align=aligns[j], name=f"{key}.{tag}c{j}", check=False)
        yy += rh + 2 * pady

    row(header, "h")
    yy += 0.02
    line(x, yy, x + W, yy, w=0.75, name=f"{key}.mid")
    yy += 0.08
    for i, cells in enumerate(rows):
        if i in mid_rules:
            yy += 0.04
            line(x, yy, x + W, yy, w=0.75, name=f"{key}.mid{i}")
            yy += 0.08
        row(cells, f"r{i}")
    yy += 0.04
    line(x, yy, x + W, yy, w=1.6, name=f"{key}.bottom")
    return yy - y


def run_column(x, w, sections, y0=Y0, y1=Y1, label=""):
    """sections: [(번호, 제목, key, [블록 함수(x, y, w) -> h])]. 한 번 재고, 남는 높이를 간격에 나눠 그린다."""
    global DRY
    DRY = True
    hh = [heading(x, 0, w, n, t, k) for n, t, k, _ in sections]
    hs = [[blk(x, 0, w) for blk in blks] for _, _, _, blks in sections]
    DRY = False
    content = sum(hh) + sum(map(sum, hs))
    nb = sum(len(b) - 1 for b in hs)
    ns = len(sections) - 1
    free = (y1 - y0) - content - nb * GB - ns * GS
    share = free / (nb + 2 * ns) if (nb + 2 * ns) else 0.0
    print("%s column: content %.2f cm, free %.2f cm -> +%.2f cm per block gap, +%.2f per section gap"
          % (label, content, free, share, 2 * share))
    y = y0
    for i, (n, t, k, blks) in enumerate(sections):
        y += heading(x, y, w, n, t, k)
        for j, blk in enumerate(blks):
            y += blk(x, y, w)
            if j < len(blks) - 1:
                y += GB + share
        if i < ns:
            y += GS + 2 * share
    return free


# ================================================================== 제목 띠
box(0, 0, W_CM, BAND_H, fill=NAVY, name="title.band")
text(M, 1.05, W_CM - 2 * M, "KSTAR pedestal top 이온 온도의 10 ms nowcasting", 46, color=WHITE, bold=True,
     pitch=54, align=PP_ALIGN.CENTER, name="title")
text(M, 3.15, W_CM - 2 * M, "같은 위치의 빔 방출 신호(BES)를 CES 이력과 함께 읽는 두 단계 estimator", 21,
     color=SUBTLE, pitch=27, align=PP_ALIGN.CENTER, name="subtitle")
text(M, 4.6, W_CM - 2 * M, AUTHOR, 17, color=WHITE, pitch=22, align=PP_ALIGN.CENTER, name="author")
text(W_CM - M - 12.5, 5.35, 12.5, "자료: KSTAR 2022 캠페인 639개 방전 (KSTAR team 제공)", 11, color=SUBTLE,
     pitch=14, align=PP_ALIGN.RIGHT, name="data.line")

# ================================================================== 핵심 수치 네 칸
TILES = [("+0.176", "", "[+0.155, +0.196]", "CES 이력만 쓰는 인과 GP 대비 skill"),
         ("+0.063", "", "[+0.038, +0.086]", "다음 값까지 쓰는 offline GP 대비 skill"),
         ("+0.097", "", "[+0.077, +0.116]", "빠른 채널의 증분 (같은 단계 쌍둥이 대비)"),
         ("175.4", " eV", "인과 GP 193.3 eV", "RMSE")]
TG = 0.6
TW = (W_CM - 2 * M - 3 * TG) / 4
for i, (big, unit, sub, lab) in enumerate(TILES):
    tx = M + i * (TW + TG)
    box(tx, STRIP_Y, TW, STRIP_H, fill=TINT, name=f"key{i}.tile")
    box(tx, STRIP_Y, TW, 0.16, fill=NAVY, name=f"key{i}.bar")
    if not DRY:
        tb = slide.shapes.add_textbox(Cm(tx), Cm(STRIP_Y + 0.45), Cm(TW), Cm(1.45))
        tb.name = f"key{i}.number"
        tf = tb.text_frame
        tf.word_wrap = True
        tf.auto_size = None
        for side in ("margin_left", "margin_right", "margin_top", "margin_bottom"):
            setattr(tf, side, 0)
        p = tf.paragraphs[0]
        p.alignment = PP_ALIGN.CENTER
        p.line_spacing = Pt(40)
        r = p.add_run()
        r.text = big.replace("-", "\u2212")
        _font(r, 36, NAVY, face=FONT_XB)
        if unit:
            r = p.add_run()
            r.text = unit
            _font(r, 22, NAVY, bold=True)
    text(tx, STRIP_Y + 1.98, TW, glue(sub), 12.5, color=MUTED, pitch=15, align=PP_ALIGN.CENTER, name=f"key{i}.interval")
    text(tx, STRIP_Y + 2.62, TW, lab, 13, color=INK, bold=True, pitch=16, align=PP_ALIGN.CENTER, name=f"key{i}.label")


# ================================================================== 왼쪽 단: 1 서론 · 2 자료와 채점 · 3 두 단계 estimator
S1_PARAS = [
    "**문제:** CES는 이온 온도 T_i를 10 ms마다 한 값 잰다. pedestal top 측정선에서는 신호가 약해 T_i 값의 7.7 %가 "
    "비어 있고, 빠짐은 무작위가 아니다. BES 수준이 낮은 5분의 1에서 18.0 %, 높은 5분의 1에서 3.4 %가 빈다(그림 1).",
    "**제약:** 감시·제어는 지금 값을 써야 하므로, 다음 값을 기다리는 보간은 쓸 수 없다.",
    "**착안:** 같은 위치를 BES 9채널과 ECEI 4채널이 빈틈 없이 본다. T_i를 재지는 않지만, 지금 프레임의 플라스마 변화를 담는다.",
]
FIG1_W = fig_size("poster_missing.png")[0]


def s1_intro(x, y, w):
    tw = w - FIG1_W - 0.6
    th = body(x, y, tw, S1_PARAS, name="s1.text")
    fh = pic("poster_missing.png", x + tw + 0.6, y,
             "BES 수준의 5분위별 T_i 결측 비율: 낮은 5분의 1 18.0 %, 전체 7.7 %, 높은 5분의 1 3.4 %")
    ch = caption(x + tw + 0.6, y + fh + 0.12, FIG1_W, "그림 1. BES 수준의 5분위별 T_i 결측 비율 (10 ms 격자 245,925행).",
                 name="s1.fig1.caption")
    return max(th, fh + 0.12 + ch)


def s1_question(x, y, w):
    return takeaway(x, y, w, "질문: 다음 CES 값 없이, CES 이력만 쓸 때보다\n지금 프레임의 T_i를 더 정확히 추정할 수 있는가?",
                    "s1.take")


def s2_data(x, y, w):
    return body(x, y, w, "**자료:** KSTAR 2022 캠페인 639개 방전, 10 ms 격자 245,925행. 측정선 R = 2.198 m(자료 제공자가 고른 "
                         "pedestal top). 한 행에는 같은 위치의 BES 9, ECEI 4, Mirnov 2채널 값과 CES 값이 있다.", name="s2.data")


def s2_timeline(x, y, w):
    """그림 2: 한 행이 담는 것 (CES 프레임, 추정 대상 프레임, blip 으로 빠진 프레임, 빔, 시간 축)."""
    lab_w = 1.35
    sx0 = x + lab_w
    sw = (w - lab_w - 0.7) / 5
    snap_h = 0.55
    fy = y + snap_h + 0.75              # 프레임 줄
    fh = 1.5
    by_ = fy + fh + 0.25                # 빔 막대
    ay = by_ + 0.65                     # 시간 축
    h_fig = ay + 0.2 + 0.5 - y
    cap = ("그림 2. 한 행이 담는 것. 빠른 진단은 시각 t의 원시 표본 하나이고, CES 값은 t에서 노출이 시작되는 프레임이다. "
           "빔이 약 10 ms 꺼지는 blip의 프레임에는 값이 없다.")
    if DRY:
        return h_fig + 0.2 + text_h(cap, CAP_PT, w, CAP_PITCH)
    slots = [("CES 프레임", FRAME, None, False, BODY),
             ("직전 프레임\n(hold가 쓰는 값)", PREV, None, False, INK),
             ("**추정 대상 프레임**\n(t에서 노출 시작)", WHITE, NAVY, False, INK),
             ("blip: 프레임 없음\n(빠진 행)", None, MUTED, True, MUTED),
             ("CES 프레임", FRAME, None, False, BODY)]
    text(x, fy + fh / 2 - 0.3, lab_w, "CES", DIA_PT, color=MUTED, bold=True, pitch=14, name="sch.label.ces")
    for i, (lab, fill, ln, dash, col) in enumerate(slots):
        b = box(sx0 + i * sw + 0.06, fy, sw - 0.12, fh, fill=fill, line=ln, lw=2.0 if ln is NAVY else 1.0,
                radius=0.12, dash=dash, shape=MSO_SHAPE.ROUNDED_RECTANGLE, name=f"sch.frame{i}")
        label_in(b, lab.split("\n"), 10.5, color=col, pitch=13.5, pad=0.06)
    text(x, by_ - 0.12, lab_w, "빔", DIA_PT, color=MUTED, bold=True, pitch=14, name="sch.label.beam")
    box(sx0 + 0.06, by_, 3 * sw - 0.12, 0.32, fill=BEAM, radius=0.05, shape=MSO_SHAPE.ROUNDED_RECTANGLE,
        name="sch.beam.on1")
    box(sx0 + 4 * sw + 0.06, by_, sw - 0.12, 0.32, fill=BEAM, radius=0.05, shape=MSO_SHAPE.ROUNDED_RECTANGLE,
        name="sch.beam.on2")
    text(sx0 + 3 * sw, by_ - 0.07, sw, "꺼짐 (약 10 ms)", 10.0, color=MUTED, pitch=13, align=PP_ALIGN.CENTER,
         name="sch.beam.off")
    line(sx0 - 0.1, ay, sx0 + 5 * sw + 0.55, ay, color=MUTED, w=1.25, head=True, name="sch.axis")
    for i, lab in enumerate(("t − 20 ms", "t − 10", "t", "t + 10", "t + 20", "t + 30")):
        line(sx0 + i * sw, ay - 0.12, sx0 + i * sw, ay + 0.12, color=MUTED, w=1.25, name=f"sch.tick{i}")
        text(sx0 + i * sw - 1.4, ay + 0.2, 2.8, lab, 10.5, color=INK if lab == "t" else MUTED, bold=(lab == "t"),
             pitch=13, align=PP_ALIGN.CENTER, name=f"sch.ticklabel{i}")
    mx = sx0 + 2 * sw                   # 시각 t 의 빠른 진단 표본
    text(mx - 6.5, y, 13.0, "빠른 진단(BES·ECEI·Mirnov) 표본: 시각 t의 순간값", 11.5, color=NAVY, bold=True, pitch=15,
         align=PP_ALIGN.CENTER, name="sch.snapshot.label")
    line(mx, y + snap_h + 0.05, mx, fy - 0.05, color=BLUE, w=2.5, head=True, name="sch.snapshot.arrow")
    caption(x, y + h_fig + 0.2, w, cap, name="s2.fig2.caption")
    return h_fig + 0.2 + text_h(cap, CAP_PT, w, CAP_PITCH)


def s2_skill(x, y, w):
    ew, eh = 13.2, 1.25
    b = box(x + (w - ew) / 2, y, ew, eh, fill=TINT, name="s2.eq")
    label_in(b, "skill = 1 − MSE(estimator) / MSE(기준)", 15.5, color=INK, bold=True, pitch=19)
    nh = text(x, y + eh + 0.15, w, "0보다 크면 estimator의 평균 제곱 오차가 작다.", CAP_PT, color=MUTED, pitch=CAP_PITCH,
              align=PP_ALIGN.CENTER, name="s2.eq.note")
    return eh + 0.15 + nh


BASE_ROWS = [("**hold**", "직전 값 유지"),
             ("**인과 GP**", "CES 이력만 맞춘 Gaussian process"),
             ("**offline GP**", "다음 값까지 사용, 실시간으로는 쓸 수 없는 참고 기준"),
             ("**쌍둥이**", "같은 모델에서 빠른 채널만 뺌, 그 차이가 “빠른 채널의 증분”")]


def s2_base(x, y, w):
    return booktabs(x, y, (3.6, w - 3.6), ("기준", "내용"), BASE_ROWS, "tab.base",
                    aligns=(PP_ALIGN.LEFT, PP_ALIGN.LEFT))


def s2_read(x, y, w):
    return body(x, y, w, "**판독:** 방전을 다섯 묶음으로 나눠 번갈아 시험하고 그 나눔을 바꿔 합친다(pooled). 구간은 방전 단위 "
                         "bootstrap 95 %. 3 keV 넘는 값(대부분 fit 실패)은 뺀다. 진술과 판정 규칙은 실행 전에 등록했다.",
                name="s2.read")


def s3_pipeline(x, y, w):
    """그림 3: 두 단계 estimator (편집 가능한 도형). 첫 단계 상자는 두 입력의 높이를 다 덮는다."""
    iw, ih, igap = 5.75, 2.05, 0.4
    s1w, s2w, o1w, o2w = 3.55, 4.0, 2.05, 1.7
    a = (w - iw - s1w - o1w - s2w - o2w) / 5          # 화살표 길이 다섯
    full = 2 * ih + igap
    cy = y + full / 2
    s2h = 2.55
    x_s1 = x + iw + a
    x_o1 = x_s1 + s1w + a
    x_s2 = x_o1 + o1w + a
    x_o2 = x_s2 + s2w + a
    lab_y = cy + s2h / 2 + 0.7
    cap = "그림 3. 두 단계 estimator. 쌍둥이는 빠른 채널만 뺀 같은 구조이고, 그 둘째 단계는 CES 이력만 읽는다."
    h_fig = max(full, lab_y + 0.5 - y)
    if DRY:
        return h_fig + 0.25 + text_h(cap, CAP_PT, w, CAP_PITCH)
    nodes = [(x, y, iw, ih, PANEL, RULE, ["**빠른 채널 15**", "BES 9 · ECEI 4 · Mirnov 2", "행 t의 값"], "arch.fast"),
             (x, y + ih + igap, iw, ih, PANEL, RULE, ["**CES 이력 7**", "마지막 값 · 경과 시간 · 관측 여부", "행 t 이전"],
              "arch.ces"),
             (x_s1, y, s1w, full, TINT_B, BLUE, ["**첫 단계: LSTM**", "2층 · 160 유닛", "격자의 모든 행"], "arch.stage1"),
             (x_s2, cy - s2h / 2, s2w, s2h, TINT_B, BLUE, ["**둘째 단계:**", "**gradient boosting**", "첫 단계의 오차 추정"],
              "arch.stage2")]
    for (bx, by, bw, bh, fill, ln, lab, nm) in nodes:
        b = box(bx, by, bw, bh, fill=fill, line=ln, lw=1.5 if ln is BLUE else 1.0, radius=0.22,
                shape=MSO_SHAPE.ROUNDED_RECTANGLE, name=nm)
        label_in(b, lab, DIA_PT, color=INK, pitch=14.5, pad=0.1)
    for (ox, ow, lab, nm) in ((x_o1, o1w, "T̂_i^(1)(t)", "arch.out1"), (x_o2, o2w, "T̂_i(t)", "arch.out2")):
        b = box(ox, cy - 0.6, ow, 1.2, fill=WHITE, line=NAVY, lw=1.5, shape=MSO_SHAPE.OVAL, name=nm)
        label_in(b, lab, 13.5, color=NAVY, bold=True, pitch=16, pad=0.02)
    for k, yy in enumerate((y + ih / 2, y + ih + igap + ih / 2)):
        line(x + iw + 0.06, yy, x_s1 - 0.06, yy, color=NAVY, w=1.75, head=True, name=f"arch.in{k}")
    line(x_s1 + s1w + 0.06, cy, x_o1 - 0.06, cy, color=NAVY, w=1.75, head=True, name="arch.a1")
    line(x_o1 + o1w + 0.06, cy, x_s2 - 0.06, cy, color=NAVY, w=1.75, head=True, name="arch.a2")
    line(x_s2 + s2w + 0.06, cy, x_o2 - 0.06, cy, color=NAVY, w=1.75, head=True, name="arch.a3")
    mx2 = x_s2 + s2w / 2
    line(mx2, lab_y - 0.05, mx2, cy + s2h / 2 + 0.06, color=BLUE, w=1.75, head=True, dash=True, name="arch.fit")
    lw_ = 8.4
    text(min(mx2 - lw_ / 2, x + w - lw_), lab_y, lw_, "학습 방전의 fold 밖 예측으로 맞춤", DIA_PT, color=BLUE, pitch=14,
         align=PP_ALIGN.CENTER, name="arch.fit.label")
    caption(x, y + h_fig + 0.25, w, cap, name="s3.fig3.caption")
    return h_fig + 0.25 + text_h(cap, CAP_PT, w, CAP_PITCH)


S3_ITEMS = [
    "**모든 행을 순서대로:** 값이 빈 프레임도 격자에 남기고, 값이 있는 행에서만 손실을 센다.",
    "**잔차 보정:** 입력 15개(BES·ECEI 창 8, CES 이력과 첫 단계 출력 7, Mirnov 없음). 학습 방전에서 만든 fold 밖 예측으로만 맞춘다.",
    "**계산:** CPU 한 스레드에서 한 행에 신경망 0.49 ms, 인과 GP 0.64 ms, 둘째 단계 1.257 ms. 10 ms 프레임 안이다.",
]


def s3_panel(x, y, w):
    pad, bd, gap = 0.45, 0.72, 0.5
    tw = w - 2 * pad - bd - 0.4
    hs = [text_h(s, BODY_PT, tw, BODY_PITCH) for s in S3_ITEMS]
    h = 2 * pad + sum(hs) + gap * (len(hs) - 1)
    box(x, y, w, h, fill=WHITE, line=RULE, lw=1.0, radius=0.35, shape=MSO_SHAPE.ROUNDED_RECTANGLE, name="s3.panel")
    yy = y + pad
    for i, (s, hh) in enumerate(zip(S3_ITEMS, hs)):
        badge(x + pad, yy + (min(hh, 2 * BODY_PITCH * PT2CM) - bd) / 2, bd, i + 1, 13, f"s3.item{i}.badge")
        text(x + pad + bd + 0.4, yy, tw, s, BODY_PT, pitch=BODY_PITCH, name=f"s3.item{i}.text")
        yy += hh
        if i < len(hs) - 1:
            line(x + pad, yy + gap / 2, x + w - pad, yy + gap / 2, color=rgb("E1E5EB"), w=0.75, name=f"s3.sep{i}")
            yy += gap
    return h


# ================================================================== 오른쪽 단: 4 결과 · 5 정보 · 6 결론 · 7 한계
T1_ROWS = [
    ("hold", "+0.378 {[+0.358, +0.397]}", "**+0.436** {[+0.418, +0.452]}"),
    ("PCHIP 보간 (다음 값 사용)", "+0.205 {[+0.173, +0.234]}", "**+0.279** {[+0.251, +0.304]}"),
    ("인과 GP", "+0.092 {[+0.068, +0.114]}", "**+0.176** {[+0.155, +0.196]}"),
    ("offline GP (다음 값 사용)", "−0.033 {[−0.062, −0.005]}", "**+0.063** {[+0.038, +0.086]}"),
    ("같은 단계의 쌍둥이 (증분)", "+0.053 {[+0.028, +0.076]}", "**+0.097** {[+0.077, +0.116]}"),
    ("RMSE (eV)", "184.2", "**175.4**"),
    ("중앙값 절대오차 (eV)", "45.8", "**41.2**"),
]


def s4_table(x, y, w):
    ch = caption(x, y, w, "표 1. 기준 대비 skill (pooled, 세 번의 나눔 15 fold). 0보다 크면 오차가 더 낮다.", name="s4.tab1.caption")
    yy = y + ch + 0.15
    c0 = 6.9
    th = booktabs(x, yy, (c0, (w - c0) / 2, (w - c0) / 2), ("기준", "신경망 혼자", "**두 단계 estimator**"), T1_ROWS,
                  "tab1", mid_rules=(5,))
    yy += th + 0.15
    nh = caption(x, yy, w, "인과 GP: RMSE 193.3 eV, 중앙값 44.1 eV. hold: 중앙값 45.4 eV.", name="s4.tab1.note")
    return yy + nh - y


S4_BULLETS = [
    "**인과 GP 대비:** +0.176 [+0.155, +0.196], 둘째 단계가 신경망 혼자의 +0.092를 높인다",
    "**다음 값을 쓰는 offline GP 대비:** +0.063 [+0.038, +0.086], 신경망 혼자는 −0.033",
    "**중앙값 오차:** hold의 0.909 [0.889, 0.931]배",
    "**빠른 채널의 증분:** 둘째 단계 뒤 +0.053 → +0.097",
]


def s4_bullets(x, y, w):
    return items(x, y, w, S4_BULLETS, "•", "s4.b", gap=0.16)


T2_ROWS = [("빠른 채널 전부 (기준 모델)", "+0.053 {[+0.028, +0.076]}"),
           ("빠른 채널을 한 행(10 ms) 늦게", "−0.006 {[−0.025, +0.016]}"),
           ("BES + CES 이력", "+0.044 {[+0.020, +0.066]}"),
           ("ECEI + CES 이력", "−0.011 {[−0.092, +0.022]}"),
           ("Mirnov만 뺌", "+0.060 {[+0.023, +0.085]}")]
T5_PT = 11.6
_IND = NB * round(text_w("방전 온도 삼등분: ", T5_PT) / text_w(NB * 10, T5_PT) * 10)   # '중간', '뜨거운'을 '차가운' 밑에
T3_ROWS = [("빔 blip 뒤 1–3행", "+0.151 {[+0.093, +0.215]}"),
           ("BES 버스트 사이 구간", "+0.050 {[+0.027, +0.075]}"),
           ("방전 온도 삼등분: 차가운", "+0.007 {[−0.023, +0.031]}"),
           (_IND + "중간", "+0.042 {[+0.009, +0.074]}"),
           (_IND + "뜨거운", "+0.092 {[+0.040, +0.137]}")]


def s5_tables(x, y, w):
    gap = 0.7
    tw = (w - gap) / 2
    cv = 4.45
    out = 0.0
    for j, (cap, head, rows, key) in enumerate(
            (("표 2. 입력을 바꿔 다시 학습한 신경망의 쌍둥이 대비 skill.", ("신경망이 읽는 것", "쌍둥이 대비"), T2_ROWS, "tab2"),
             ("표 3. 증분이 있는 곳 (쌍둥이 대비).", ("행 · 방전", "쌍둥이 대비"), T3_ROWS, "tab3"))):
        tx = x + j * (tw + gap)
        ch = caption(tx, y, tw, cap, name=f"s5.{key}.caption")
        out = max(out, ch)
    yy = y + out + 0.15
    hs = []
    for j, (head, rows, key) in enumerate(((("신경망이 읽는 것", "쌍둥이 대비"), T2_ROWS, "tab2"),
                                          (("행 · 방전", "쌍둥이 대비"), T3_ROWS, "tab3"))):
        tx = x + j * (tw + gap)
        hs.append(booktabs(tx, yy, (tw - cv, cv), head, rows, key, pt=T5_PT, pitch=15.6, padx=0.1,
                           aligns=(PP_ALIGN.LEFT, PP_ALIGN.RIGHT)))
    return out + 0.15 + max(hs)


S5_BULLETS = [
    "**지금 프레임의 BES:** 한 행 늦게 넣으면 증분이 없다. 10 ms 앞선 전조가 아니다.",
    "**Mirnov는 필요 없다.** ECEI만으로는 증분이 없다.",
    "**빔 blip 직후에 가장 크고,** 가장 차가운 3분의 1에는 없다.",
]


def s5_bullets(x, y, w):
    return items(x, y, w, S5_BULLETS, "•", "s5.b", gap=0.16)


def s6_take(x, y, w):
    return takeaway(x, y, w, "같은 위치 BES의 지금 프레임 변화는 pedestal top CES T_i에 대해 CES 이력 너머의 정보를 준다. "
                             "두 단계 estimator는 인과 GP 대비 +0.176, 다음 값을 쓰는 offline GP 대비로도 +0.063이다"
                             "(3 keV 컷 모집단).", "s6.take")


S6_CHECKS = [
    "**fit의 산물이 아니다:** 빠른 채널이 만든 보정의 0.810 [0.753, 0.866]이\n따로 fit되는 다음 프레임에도 남는다",
    "**오차의 바닥:** 보고값의 프레임별 산포 하한은 RMS 135.7 eV, 남은 오차의 대부분이 이 바닥이다",
    "**10 ms 프레임 안의 계산**, 진술은 실행 전에 등록",
]


def s6_checks(x, y, w):
    return items(x, y, w, S6_CHECKS, "✓", "s6.c", gap=0.2)


S7_ITEMS = [
    ("**3 keV 넘는 값(fit 실패)을 남기면 증분이 사라진다:** 쌍둥이 대비 −0.002 [−0.017, +0.010], 인과 GP 대비는 "
     "+0.109 [+0.070, +0.146]으로 남는다",
     "→ 가설 넷을 가린다: 소수의 큰 값이 손실을 지배 · 정규화 폭 · fit 실패와 빠른 채널의 결합 · 인과 GP 이력"),
    ("**손대지 않은 검증 방전이 없다**", "→ 캠페인 마지막 97개 방전으로 시간순 확증 (진행 중)"),
    ("**측정선 하나, 캠페인 하나, 장치 하나.** 재는 것은 탄소 불순물 온도이고, KSTAR에서 BES·CES의 실시간 사용은 "
     "보고되지 않았다", "→ 실시간 적용은 전망"),
]
FIG4 = "poster_inclusive.png"
FIG4_CAP = "그림 4. 3 keV 컷 모집단과 포함 모집단의 skill (신경망)."


def s7_limits(x, y, w):
    """번호 항목과 "→" 처방. 그림 4는 ①의 오른쪽에 두고, 그림 아래로 내려간 항목은 단 너비를 다 쓴다."""
    bd, gap, fgap = 0.72, 0.42, 0.5
    fw, fh = fig_size(FIG4)
    cw = fw + fgap - 0.1                      # 설명은 그림 왼쪽 틈까지 써서 한 줄에
    ch = text_h(FIG4_CAP, CAP_PT, cw, CAP_PITCH)
    fig_bottom = y + fh + 0.1 + ch
    yy = y
    for i, (head, fix) in enumerate(S7_ITEMS):
        narrow = yy < fig_bottom - 0.6               # 그림 옆이면 좁게, 그림 끝에 가까우면 그 아래로 내려 넓게
        if not narrow:
            yy = max(yy, fig_bottom + 0.3)
        iw = (w - fw - fgap if narrow else w) - bd - 0.32
        ix = x + bd + 0.32
        badge(x, yy + 0.06, bd, i + 1, 13, f"s7.item{i}.badge")
        yy += text(ix, yy, iw, head, BODY_PT, pitch=BODY_PITCH, name=f"s7.item{i}.text") + 0.06
        yy += text(ix, yy, iw, fix, BODY_PT, color=NAVY, pitch=BODY_PITCH, name=f"s7.item{i}.fix")
        if i < len(S7_ITEMS) - 1:
            yy += gap
    pic(FIG4, x + w - fw, y, "3 keV 컷 모집단과 포함 모집단의 신경망 skill: 인과 GP 대비 +0.092와 +0.109, "
                              "쌍둥이 대비 +0.053과 -0.002")
    caption(x + w - cw, y + fh + 0.1, cw, FIG4_CAP, name="s7.fig4.caption")
    return max(yy, fig_bottom) - y


LEFT = [(1, "서론", "s1", [s1_intro, s1_question]),
        (2, "자료와 채점", "s2", [s2_data, s2_timeline, s2_skill, s2_base, s2_read]),
        (3, "두 단계 estimator", "s3", [s3_pipeline, s3_panel])]
RIGHT = [(4, "결과", "s4", [s4_table, s4_bullets]),
         (5, "정보는 어디에 있나 (첫 단계 판독)", "s5", [s5_tables, s5_bullets]),
         (6, "결론", "s6", [s6_take, s6_checks]),
         (7, "한계와 향후 연구", "s7", [s7_limits])]

free_l = run_column(XL, COL_W, LEFT, label="left")
free_r = run_column(XR, COL_W, RIGHT, label="right")
line(M + COL_W + GUT / 2, Y0 + 0.1, M + COL_W + GUT / 2, Y1, color=RULE, w=1.0, name="column.divider")

# ================================================================== 아래: 참고문헌 · 사사
line(M, FOOT_Y, W_CM - M, FOOT_Y, color=RULE, w=1.0, name="footer.rule")
REFS = ["**참고문헌**  [1] W. H. Ko et al., Rev. Sci. Instrum. 81, 10D740 (2010).   [2] J. K. Lee et al., Fusion Eng. Des. "
        "222, 115518 (2026).   [3] C. Heo et al., Nucl. Fusion 65, 036003 (2025).   [4] J. H. Friedman, Ann. Stat. 29, "
        "1189 (2001).",
        "**사사**  KSTAR 자료는 KSTAR team이 제공하였다. 별도의 연구비 지원은 없었다."]
fh_ = text(M, FOOT_Y + 0.3, W_CM - 2 * M, REFS, 10.5, color=MUTED, pitch=14, after=3, name="footer.text")
print("footer ends %.2f cm above the sheet edge" % (H_CM - (FOOT_Y + 0.3 + fh_)))
for w_ in WARN:
    print("  WARN", w_)

prs.core_properties.title = "KSTAR pedestal top 이온 온도의 10 ms nowcasting (포스터)"
prs.core_properties.author = "" if args.anon else "이승상"
OUT.parent.mkdir(parents=True, exist_ok=True)
prs.save(OUT)
print("saved", OUT)


def export_png(pptx, png, px):
    """PowerPoint COM 으로 첫 슬라이드를 PNG 로. 창 없이 열고, 열린 문서가 없을 때만 Quit."""
    import win32com.client
    app = win32com.client.Dispatch("PowerPoint.Application")
    pres = app.Presentations.Open(str(Path(pptx).resolve()), True, False, False)  # ReadOnly, Untitled, WithWindow
    try:
        pres.Slides(1).Export(str(Path(png).resolve()), "PNG", px, round(px * H_CM / W_CM))
    finally:
        pres.Close()
        if app.Presentations.Count == 0:
            app.Quit()


if not args.no_png:
    export_png(OUT, PNG, args.png_px)
    print("png", PNG)
