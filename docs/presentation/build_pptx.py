# -*- coding: utf-8 -*-
"""Build the 1-hour KSTAR CES nowcasting thesis presentation (Korean).

Output: docs/presentation/KSTAR_CES_발표자료.pptx  (45 slides, ~60분 학위논문 발표)

확정 프로토콜(2026-08-16, THESIS_RESULTS.md §8v–§8ab) 기준으로 전면 재작성:
W=2 · held-free(학습·평가) · 파일당 500 · 두 공동 1차 모집단(컷 / 포함) · 인과 GP 기준선.
주 모델은 전체격자 인과 시퀀스 나우캐스터 ``seq_v2``(357,570 파라미터)이고, 옛 주 모델
(윈도 GRU + 관측마스킹 attention, 201,258)은 **W=2 윈도 대조군**이다. 모든 수치는
docs/paper/outline_ko_v2.tex = docs/paper/main_ko.tex = paper_numbers.json에서 왔다.
W=4 시대 수치와 progression 서사는 전부 제거했다.

레이아웃 헬퍼(slide/box/text/header/footer/bullets/card/add_image_fit/table/divider)와
팔레트는 build_pptx_20min.py · build_pptx_flow.py가 import하므로 시그니처를 바꾸지 않는다.

Figures are read from docs/presentation/figures/ (run make_figures.py first).

Usage (from repo root):
    py docs/presentation/build_pptx.py
    py docs/presentation/preview_pptx.py docs/presentation/KSTAR_CES_발표자료.pptx
"""
import os
import sys

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE, MSO_CONNECTOR
from pptx.oxml.ns import qn

HERE = os.path.dirname(__file__)
FIG = os.path.join(HERE, "figures")
if os.path.abspath(HERE) not in sys.path:      # so the build-time fit audit can
    sys.path.insert(0, os.path.abspath(HERE))  # import preview_pptx's metrics

# ---- palette -------------------------------------------------------------
NAVY = RGBColor(0x13, 0x33, 0x5F)
BLUE = RGBColor(0x2B, 0x6C, 0xB0)
TEAL = RGBColor(0x1B, 0x9E, 0x8A)
ORANGE = RGBColor(0xE8, 0x74, 0x3B)
GREEN = RGBColor(0x2E, 0x9E, 0x5B)
RED = RGBColor(0xC0, 0x39, 0x2B)
GRAY = RGBColor(0x5B, 0x66, 0x70)
LGRAY = RGBColor(0xE8, 0xEC, 0xF1)
MGRAY = RGBColor(0x9A, 0xA5, 0xB1)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
DARK = RGBColor(0x22, 0x2B, 0x35)
CARDBG = RGBColor(0xF4, 0xF7, 0xFA)

FONT = "Malgun Gothic"
MONO = "Consolas"

EMU_W, EMU_H = Inches(13.333), Inches(7.5)

prs = Presentation()
prs.slide_width = EMU_W
prs.slide_height = EMU_H
BLANK = prs.slide_layouts[6]

_pageno = {"n": 0}


# ---- low-level helpers ---------------------------------------------------
def _set_fill(shape, color):
    shape.fill.solid()
    shape.fill.fore_color.rgb = color
    shape.line.fill.background()


def slide():
    s = prs.slides.add_slide(BLANK)
    bg = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, EMU_W, EMU_H)
    _set_fill(bg, WHITE)
    bg.shadow.inherit = False
    return s


def box(s, x, y, w, h, fill=None, line=None, line_w=1.0, shape=MSO_SHAPE.RECTANGLE,
        shadow=False, round_=False):
    sp = s.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE if round_ else shape, x, y, w, h)
    if fill is None:
        sp.fill.background()
    else:
        sp.fill.solid()
        sp.fill.fore_color.rgb = fill
    if line is None:
        sp.line.fill.background()
    else:
        sp.line.color.rgb = line
        sp.line.width = Pt(line_w)
    sp.shadow.inherit = False
    if shadow:
        sp.shadow.inherit = False
    return sp


def text(s, x, y, w, h, runs, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP,
         space_after=4, line_spacing=1.06, wrap=True):
    """runs: list of paragraphs; each paragraph is a list of (txt, size, color, bold, italic, font)."""
    tb = s.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame
    tf.word_wrap = wrap
    tf.vertical_anchor = anchor
    tf.margin_left = Pt(2)
    tf.margin_right = Pt(2)
    tf.margin_top = Pt(1)
    tf.margin_bottom = Pt(1)
    for i, para in enumerate(runs):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.space_after = Pt(space_after)
        p.space_before = Pt(0)
        p.line_spacing = line_spacing
        if isinstance(para, dict):  # paragraph-level options
            opts = para
            segs = opts["segs"]
            if "align" in opts:
                p.alignment = opts["align"]
            if "space_after" in opts:
                p.space_after = Pt(opts["space_after"])
            if "space_before" in opts:
                p.space_before = Pt(opts["space_before"])
            if "level" in opts:
                p.level = opts["level"]
        else:
            segs = para
        for seg in segs:
            txt, size, color, bold, italic, font = (seg + (None,) * 6)[:6]
            r = p.add_run()
            r.text = txt
            r.font.size = Pt(size)
            r.font.color.rgb = color if color is not None else DARK
            r.font.bold = bool(bold)
            r.font.italic = bool(italic)
            r.font.name = font if font else FONT
    return tb


def header(s, kicker, title, accent=NAVY):
    box(s, Inches(0), Inches(0), EMU_W, Inches(1.28), fill=WHITE)
    box(s, Inches(0.0), Inches(0.0), Inches(0.22), Inches(1.28), fill=accent)
    text(s, Inches(0.55), Inches(0.20), Inches(12.3), Inches(0.34),
         [[(kicker, 13, accent, True, False, None)]])
    text(s, Inches(0.52), Inches(0.50), Inches(12.4), Inches(0.72),
         [[(title, 27, NAVY, True, False, None)]])
    box(s, Inches(0.55), Inches(1.18), Inches(12.25), Pt(2), fill=LGRAY)
    footer(s)


def footer(s):
    _pageno["n"] += 1
    text(s, Inches(0.55), Inches(7.06), Inches(8.0), Inches(0.3),
         [[("KSTAR CES Nowcasting — 다중진단 기반 CES 결측 구간 예측", 9, MGRAY, False, False, None)]])
    text(s, Inches(11.6), Inches(7.06), Inches(1.2), Inches(0.3),
         [[(str(_pageno["n"]), 10, MGRAY, False, False, None)]], align=PP_ALIGN.RIGHT)


def bullets(s, x, y, w, h, items, size=15, gap=7, color=DARK):
    """items: list of (text, level, color_override, bold)."""
    paras = []
    for it in items:
        txt, lvl, col, bold = (list(it) + [None, None])[:4]
        lvl = lvl or 0
        mark = "●  " if lvl == 0 else ("–  " if lvl == 1 else "·  ")
        sz = size if lvl == 0 else (size - 2 if lvl == 1 else size - 3)
        paras.append({"segs": [(mark + txt, sz,
                                col if col else (NAVY if lvl == 0 else color),
                                bold if bold is not None else (lvl == 0), False, None)],
                      "level": lvl, "space_after": gap})
    return text(s, x, y, w, h, paras, line_spacing=1.08)


def card(s, x, y, w, h, title, lines, accent=BLUE, title_size=14, body_size=12,
         body_color=DARK):
    c = box(s, x, y, w, h, fill=CARDBG, round_=True)
    box(s, x, y, Inches(0.10), h, fill=accent)
    text(s, x + Inches(0.26), y + Inches(0.12), w - Inches(0.4), Inches(0.4),
         [[(title, title_size, accent, True, False, None)]])
    paras = [[(ln, body_size, body_color, False, False, None)] for ln in lines]
    text(s, x + Inches(0.26), y + Inches(0.56), w - Inches(0.42), h - Inches(0.7),
         paras, line_spacing=1.1, space_after=3)
    return c


def add_image_fit(s, path, x, y, w, h):
    """Add image scaled to fit inside (w,h) box, centered."""
    from PIL import Image
    try:
        iw, ih = Image.open(path).size
    except Exception:
        pic = s.shapes.add_picture(path, x, y, width=w)
        return pic
    ar = iw / ih
    bw, bh = w, h
    if bw / bh > ar:
        nh = bh
        nw = int(bh * ar)
    else:
        nw = bw
        nh = int(bw / ar)
    nx = x + (bw - nw) // 2
    ny = y + (bh - nh) // 2
    return s.shapes.add_picture(path, nx, ny, width=nw, height=nh)


def table(s, x, y, col_w, head, rows, row_h=Inches(0.44), head_h=Inches(0.44),
          head_fill=NAVY, head_color=WHITE, size=13, head_size=12.5,
          zebra=LGRAY, emphasis=None, emphasis_fill=None, label_align_left=True):
    """Simple grid table.

    col_w   : list of column widths (Emu)
    head    : list of header cell strings
    rows    : list of rows; each cell is str or (txt, color, bold, font)
    emphasis: set of row indices drawn with emphasis_fill + bold
    """
    total_w = sum(col_w)
    box(s, x, y, total_w, head_h, fill=head_fill)
    cx = x
    for j, h in enumerate(head):
        text(s, cx + Inches(0.10), y + Inches(0.07), col_w[j] - Inches(0.16),
             head_h - Inches(0.1),
             [[(h, head_size, head_color, True, False, None)]],
             align=PP_ALIGN.LEFT if (j == 0 and label_align_left) else PP_ALIGN.CENTER)
        cx += col_w[j]
    emphasis = emphasis or set()
    yy = y + head_h
    for i, row in enumerate(rows):
        if i in emphasis and emphasis_fill is not None:
            box(s, x, yy, total_w, row_h, fill=emphasis_fill)
        elif zebra is not None and i % 2 == 1:
            box(s, x, yy, total_w, row_h, fill=CARDBG)
        cx = x
        for j, cell in enumerate(row):
            if isinstance(cell, str):
                txt, col, bold, font = cell, DARK, (i in emphasis), None
            else:
                txt, col, bold, font = (tuple(cell) + (None,) * 4)[:4]
                col = col if col is not None else DARK
                bold = bold if bold is not None else (i in emphasis)
            text(s, cx + Inches(0.10), yy + Inches(0.08), col_w[j] - Inches(0.16),
                 row_h - Inches(0.1),
                 [[(txt, size, col, bold, False, font)]],
                 align=PP_ALIGN.LEFT if (j == 0 and label_align_left) else PP_ALIGN.CENTER)
            cx += col_w[j]
        box(s, x, yy + row_h - Pt(0.75), total_w, Pt(0.75), fill=LGRAY)
        yy += row_h
    return yy


# ============================ SLIDES ======================================

# --- 1. Title -------------------------------------------------------------
def s_title():
    s = slide()
    box(s, 0, 0, EMU_W, EMU_H, fill=NAVY)
    box(s, 0, Inches(5.6), EMU_W, Inches(1.9), fill=RGBColor(0x0E, 0x26, 0x47))
    box(s, Inches(0.9), Inches(1.7), Inches(2.2), Pt(4), fill=ORANGE)
    text(s, Inches(0.9), Inches(1.9), Inches(11.6), Inches(0.5),
         [[("학위논문 발표 · 약 60분", 16, RGBColor(0x9C, 0xC0, 0xE8), True, False, None)]])
    text(s, Inches(0.88), Inches(2.45), Inches(11.7), Inches(2.0),
         [[("KSTAR 다중 진단 기반 인과(causal) 나우캐스팅:", 30, WHITE, True, False, None)],
          [("희소 CES 신호(Tᵢ · V_rot)의 결측 구간 복원", 34, WHITE, True, False, None)]],
         line_spacing=1.12)
    text(s, Inches(0.9), Inches(4.30), Inches(11.5), Inches(1.1),
         [[("빠른 진단(BES · ECEI · Mirnov)과 과거 CES 이력만으로 ", 16, LGRAY, False, False, None),
           ("이온온도 Tᵢ · 토로이달 회전 V_rot", 16, ORANGE, True, False, None),
           ("를 복원하고,", 16, LGRAY, False, False, None)],
          [("미래까지 보는 오프라인 보간과 배치 가능한 최강 기준선(인과 GP)을 상대로 통계적으로 검증",
            16, LGRAY, False, False, None)]],
         line_spacing=1.2)
    text(s, Inches(0.9), Inches(5.95), Inches(11.5), Inches(1.0),
         [[("이승상  (Seungsang Lee)", 17, WHITE, True, False, None)],
          [("서울대학교 · 원자핵공학  |  확정 프로토콜: W=2 · held-free · 두 모집단 공동 1차 · 백본 seq_v2 (357,570 파라미터)",
            13, MGRAY, False, False, None)]],
         line_spacing=1.25)
    return s


# --- 2. Agenda ------------------------------------------------------------
def s_agenda():
    s = slide()
    header(s, "Contents", "발표 목차")
    items = [
        ("1", "연구 배경 & 문제 정의", "CES는 왜 비는가 · 두 모집단이 필요한 이유", ORANGE),
        ("2", "접근법 — 의도적으로 어려운 평가 bar", "미래를 보는 보간 + 최강 인과 기준선(인과 GP)", BLUE),
        ("3", "데이터 & 파이프라인", "held 전면 제거 · 두 프레이밍 · 누수 삼중 차단", TEAL),
        ("4", "모델", "전체격자 인과 시퀀스 백본 seq_v2 + W=2 윈도 대조군", NAVY),
        ("5", "평가 방법론", "사전등록 · shot 군집 bootstrap · TEST 동결", BLUE),
        ("6", "결과", "Tᵢ 4/4+4/4 · V_rot 동률 · 스트레스 2종 생존", ORANGE),
        ("7", "결론 · 한계 · 향후 연구", "상한은 추정기가 아니라 정보 — 데이터 레버 3종", GRAY),
    ]
    y = 1.55
    for i, (num, t, sub, col) in enumerate(items):
        col_x = 0.7 if i < 4 else 6.95
        yy = Inches(y + (i % 4) * 1.30)
        box(s, Inches(col_x), yy, Inches(5.7), Inches(1.12), fill=CARDBG, round_=True)
        nb = box(s, Inches(col_x + 0.12), yy + Inches(0.16), Inches(0.8), Inches(0.8),
                 fill=col, round_=True)
        text(s, Inches(col_x + 0.12), yy + Inches(0.16), Inches(0.8), Inches(0.8),
             [[(num, 26, WHITE, True, False, None)]], align=PP_ALIGN.CENTER,
             anchor=MSO_ANCHOR.MIDDLE)
        text(s, Inches(col_x + 1.05), yy + Inches(0.16), Inches(4.5), Inches(0.45),
             [[(t, 15.5, NAVY, True, False, None)]])
        text(s, Inches(col_x + 1.05), yy + Inches(0.62), Inches(4.55), Inches(0.45),
             [[(sub, 11.5, GRAY, False, False, None)]])
    return s


# --- Section divider ------------------------------------------------------
def divider(num, title, sub):
    s = slide()
    box(s, 0, 0, EMU_W, EMU_H, fill=NAVY)
    box(s, Inches(0.0), Inches(2.7), Inches(3.6), Inches(2.1), fill=RGBColor(0x0E, 0x26, 0x47))
    text(s, Inches(0.55), Inches(2.55), Inches(3.2), Inches(2.4),
         [[(num, 120, ORANGE, True, False, None)]], align=PP_ALIGN.CENTER,
         anchor=MSO_ANCHOR.MIDDLE)
    box(s, Inches(4.0), Inches(3.0), Pt(3), Inches(1.6), fill=ORANGE)
    text(s, Inches(4.4), Inches(3.05), Inches(8.4), Inches(1.0),
         [[(title, 36, WHITE, True, False, None)]], anchor=MSO_ANCHOR.MIDDLE)
    text(s, Inches(4.42), Inches(4.05), Inches(8.4), Inches(0.8),
         [[(sub, 16, LGRAY, False, False, None)]])
    return s


# --- 4. Background: KSTAR + diagnostics -----------------------------------
def s_diagnostics():
    s = slide()
    header(s, "1. 연구 배경", "진단 장치: 무엇을 측정하나")
    text(s, Inches(0.55), Inches(1.45), Inches(12.3), Inches(0.6),
         [[("KSTAR pedestal-top 플라즈마에서 ", 15, DARK, False, False, None),
           ("CES", 15, ORANGE, True, False, None),
           ("는 핵심 물리량을 주지만 느리고 자주 빈다. ", 15, DARK, False, False, None),
           ("빠른 진단(BES·ECEI·Mirnov)은 같은 10 ms 격자에서 결측 없이 측정된다.", 15, NAVY, True, False, None)]])
    cards = [
        ("CES  (타겟)", ORANGE,
         ["Charge Exchange Spectroscopy",
          "Tᵢ (이온온도), V_rot (토로이달 회전)",
          "광자 적분 필요 → 느림 · 자주 결측",
          "→ 본 연구가 채우려는 대상"]),
        ("BES  (9 ch)", BLUE,
         ["Beam Emission Spectroscopy",
          "밀도요동 nₑ 의 공간 구조",
          "10 ms 격자에서 항상 측정",
          "→ Tᵢ 단서 (충돌 e–i 결합)"]),
        ("ECEI  (4 ch)", TEAL,
         ["Electron Cyclotron Emission Imaging",
          "전자온도 Tₑ 2D 영상",
          "10 ms 격자에서 항상 측정",
          "→ Tᵢ 단서 (Tₑ ↔ Tᵢ 결합)"]),
        ("Mirnov coil  (2 ch)", GRAY,
         ["자기요동(MHD mode) dB/dt",
          "kHz 신호를 100 Hz로 데시메이트",
          "lag-1 자기상관 -0.009 (BES +0.568)",
          "→ 모드 회전 정보가 상류에서 소실"]),
    ]
    x0 = 0.55
    for i, (t, col, lines) in enumerate(cards):
        card(s, Inches(x0 + i * 3.12), Inches(2.2), Inches(2.95), Inches(3.5),
             t, lines, accent=col, title_size=14.5, body_size=11.5)
    text(s, Inches(0.55), Inches(5.95), Inches(12.3), Inches(0.9),
         [[("핵심 비대칭(미리보기): ", 14, NAVY, True, False, None),
           ("빠른 진단은 물리적으로 Tᵢ 정보는 운반하지만 V_rot 정보는 거의 운반하지 않는다 "
            "(NBI 토크 미관측 + Mirnov 앨리어싱). 이 가설이 결과에서 그대로 확인된다.",
            14, DARK, False, False, None)]], line_spacing=1.15)
    return s


# --- 5. CES missing problem ----------------------------------------------
def s_problem():
    s = slide()
    header(s, "1. 연구 배경", "문제: CES는 왜, 얼마나 비는가")
    bullets(s, Inches(0.55), Inches(1.5), Inches(6.3), Inches(4.6), [
        ("CES는 충분한 신호대잡음비를 위해 광자를 오래 적분해야 함", 0),
        ("노출·신호품질 문제로 특정 시점 측정이 자주 누락됨", 1),
        ("같은 10 ms 격자에서 Tᵢ 8.2%, V_rot 23.9%가 완전 결측(NaN)", 0),
        ("V_rot는 여기에 held(직전값 복사) 41.1%가 더해져 실질 65.0% 무정보", 1, RED, True),
        ("두 타겟은 서로 독립적으로 결측 → 타겟별 처리 필요", 1),
        ("결측은 물리적으로 흥미로운 순간에 몰린다 (저신호·ELM·천이)", 0),
        ("무작위가 아닌 결측(MNAR) — 관측점 skill은 낙관적 상한", 1, ORANGE, True),
        ("빠른 진단(BES·ECEI·Mirnov)은 같은 격자에서 결측 없음", 0),
        ("\"항상 있는 빠른 진단\"으로 \"자주 비는 CES\"를 채운다", 1, ORANGE, True),
    ])
    box(s, Inches(7.1), Inches(1.55), Inches(5.7), Inches(2.5), fill=CARDBG, round_=True)
    text(s, Inches(7.35), Inches(1.72), Inches(5.3), Inches(0.5),
         [[("데이터 기반 가상 센서의 강점", 16, NAVY, True, False, None)]])
    bullets(s, Inches(7.35), Inches(2.25), Inches(5.25), Inches(1.8), [
        ("결측 시점의 Tᵢ·V_rot를 데이터로 추정 (gap-filling)", 0, TEAL, True),
        ("강한 역산 가정 불필요 — 축대칭 수준만 가정", 0, TEAL, True),
        ("끊김 없는 CES 가용성 → pedestal 물리 분석·실시간 활용", 0, TEAL, True),
    ], size=13, gap=8)
    add_image_fit(s, os.path.join(FIG, "fig_missing.png"),
                  Inches(7.0), Inches(4.2), Inches(6.0), Inches(2.6))
    return s


# --- 5b. Missingness ledger (measured, all 641 shots) --------------------
def s_missing_table():
    s = slide()
    header(s, "1. 연구 배경", "결측 실측 집계: NaN 결측 + held(같은 값 padding)", accent=RED)
    text(s, Inches(0.55), Inches(1.36), Inches(12.3), Inches(0.62),
         [[("‘V_rot 결측 24%’는 NaN만 센 값이다. ", 14.5, RED, True, False, None),
           ("직전 관측값을 그대로 복사한 held 행을 합치면 실질 무정보는 ", 14.5, DARK, False, False, None),
           ("65.0%", 14.5, RED, True, False, None),
           (" — 전체 641 shot · 247,207 행 전수 집계.", 14.5, DARK, False, False, None)]],
         line_spacing=1.14)
    cw = [Inches(5.4), Inches(3.4), Inches(3.4)]
    rows = [
        ["전체 10 ms 격자 행 (641 shot)",
         ("247,207", GRAY, False, MONO), ("247,207", GRAY, False, MONO)],
        ["① 완전 결측 — 값이 비어 있음 (NaN)",
         ("20,216  (8.2%)", DARK, False, MONO), ("59,107  (23.9%)", BLUE, True, MONO)],
        ["② held / padding — 직전 관측값과 bit-identical",
         ("1  (0.0%)", DARK, False, MONO), ("101,604  (41.1%)", RED, True, MONO)],
        [("실질 무정보  ① + ②", NAVY, True, None),
         ("20,217  (8.2%)", NAVY, True, MONO), ("160,711  (65.0%)", RED, True, MONO)],
        ["독립 관측 — 실제 정보가 있는 행",
         ("226,990  (91.8%)", GREEN, True, MONO), ("86,496  (35.0%)", DARK, True, MONO)],
        ["관측값(non-NaN) 중 held 비율",
         ("0.0%", GREEN, True, MONO), ("54.0%", RED, True, MONO)],
        ["held이 존재하는 shot 파일 수",
         ("1 / 641", DARK, False, MONO), ("499 / 641", RED, True, MONO)],
        ["연속 held 구간 길이 — 중앙값 / 최대 (행)",
         ("2 / 2", DARK, False, MONO), ("10 / 1,214", RED, True, MONO)],
    ]
    table(s, Inches(0.55), Inches(2.04), cw, ["구분", "CES_TI", "CES_VT"], rows,
          row_h=Inches(0.42), head_h=Inches(0.42), emphasis={3},
          emphasis_fill=RGBColor(0xFD, 0xEC, 0xE8))
    box(s, Inches(0.55), Inches(5.90), Inches(12.25), Inches(1.06),
        fill=CARDBG, round_=True)
    text(s, Inches(0.8), Inches(5.98), Inches(11.75), Inches(0.94),
         [[("왜 중요한가: ", 12, NAVY, True, False, None),
           ("held 행은 baseline(persistence·보간)이 오차 ≈0으로 맞히는 ‘공짜 정답’이다.",
            12, DARK, False, False, None)],
          [("확정 프로토콜: ", 12, NAVY, True, False, None),
           ("지도 타겟·이력 입력·정규화 통계·모든 기준선의 보간 앵커에서 전부 제거 — 진짜 측정만 채점한다.",
            12, DARK, False, False, None)],
          [("판정 기준: ", 12, NAVY, True, False, None),
           ("연속 블록 안에서 직전 관측값과 부동소수점까지 동일한 행. CES_TI는 226,991행 중 단 1행이다.",
            12, DARK, False, False, None)]],
         line_spacing=1.12, space_after=2)
    return s


# --- 5c. Fitting failures -> the two co-primary populations --------------
def s_two_populations():
    s = slide()
    header(s, "1. 연구 배경", "데이터 품질 감사: Tᵢ 피팅 실패와 두 공동 1차 모집단", accent=RED)
    text(s, Inches(0.55), Inches(1.38), Inches(12.3), Inches(0.6),
         [[("관측 Tᵢ의 p99 = 2,089 eV · p99.9 = 9,601 eV · 최대 14,984 eV. ", 14.5, DARK, False, False, None),
           ("이 먼 꼬리는 플라즈마가 아니라 실패한 스펙트럼 피팅이다.", 14.5, RED, True, False, None)]],
         line_spacing=1.14)
    card(s, Inches(0.55), Inches(2.02), Inches(6.0), Inches(2.05),
         "실측 — >3 keV는 무엇인가", [
             "1,197행(0.53%) = 951 run / 274 방전",
             "run의 85%는 단일 행, 5행 이상 지속은 2%",
             "run 정점 = 관측 이웃 평균의 13× (IQR 6–26×)",
             "어떤 방법으로도 예측 불가 + 보간 앵커를 오염시킴",
         ], accent=RED, title_size=14.5, body_size=12.5)
    card(s, Inches(6.8), Inches(2.02), Inches(6.0), Inches(2.05),
         "대응 — 두 대응 모두 비판 가능하다", [
             "제거하면: “어려운 행을 없앴다”는 비판",
             "유지하면: 스파이크 앵커가 오프라인 기준선을 핸디캡",
             "→ 두 공동 1차 모집단을 사전등록:",
             "   컷(적재 시 결측 처리, 전 arm 동일) / 포함(컷 없음)",
         ], accent=BLUE, title_size=14.5, body_size=12.5)
    box(s, Inches(0.55), Inches(4.35), Inches(12.25), Inches(0.72), fill=NAVY, round_=True)
    text(s, Inches(0.85), Inches(4.46), Inches(11.7), Inches(0.6),
         [[("규칙: ", 14, ORANGE, True, False, None),
           ("무조건부 주장은 두 모집단 모두에서 성립할 때만 한다. 한쪽에서만 성립하면 모집단을 명시해 보고한다.",
            14, WHITE, False, False, None)]])
    bullets(s, Inches(0.55), Inches(5.35), Inches(12.3), Inches(1.5), [
        ("문턱은 무관하다: 2.5 / 3 / 4 keV 재학습 → Tᵢ +0.230 / +0.236 / +0.232, PR4 4/4 전부", 0, GREEN, True),
        ("값 컷은 일방향 프록시 — 하향 dip 4,965행은 손대지 않고, ≥2× 상향 이상치의 19%만 제거", 0),
        ("V_rot 스파이크(>1,000 km/s 119행 / 16 방전, 101행은 한 방전 한 블록)는 컷 없이 두되 SSE 비중을 병기", 0),
    ], size=13, gap=6)
    return s


# --- 6. Research question / core idea ------------------------------------
def s_idea():
    s = slide()
    header(s, "1. 연구 배경", "연구 질문 & 핵심 아이디어")
    box(s, Inches(0.7), Inches(1.55), Inches(11.9), Inches(1.4), fill=NAVY, round_=True)
    text(s, Inches(1.0), Inches(1.72), Inches(11.4), Inches(1.1),
         [[("연구 질문", 13, ORANGE, True, False, None)],
          [("CES가 결측된 10 ms 시점에서, 동시각 빠른 진단(BES·ECEI·Mirnov) + 과거 CES 이력만으로 ",
            16.5, WHITE, False, False, None)],
          [("CES 자체의 시간 보간이 복원할 수 없는 정보를 회복할 수 있는가?",
            16.5, WHITE, True, False, None)]], line_spacing=1.18)
    cards = [
        ("가상 센서 (Virtual Sensor)", BLUE,
         ["빠른 진단으로부터 CES를 데이터 기반 추정",
          "역산(inverse mapping) 가정 없이",
          "결측·고장 시점을 온라인으로 메움"]),
        ("Gap-filling / Nowcasting", TEAL,
         ["미래를 예보하는 forecasting 아님",
          "현재 시점의 빈 값을 채움 (nowcast)",
          "초해상(super-resolution)과도 구분"]),
        ("정직한 검증", ORANGE,
         ["진짜 결측은 참값이 없음 → 직접검증 불가",
          "관측된 CES를 가린 뒤 복원 정확도 측정",
          "두 모집단(컷/포함) 각각에서 타겟별로"]),
    ]
    for i, (t, col, lines) in enumerate(cards):
        card(s, Inches(0.55 + i * 4.13), Inches(3.35), Inches(3.95), Inches(2.15),
             t, lines, accent=col, title_size=15, body_size=12.5)
    text(s, Inches(0.7), Inches(5.90), Inches(11.9), Inches(0.9),
         [[("결론의 성격: ", 13.5, NAVY, True, False, None),
           ("masking 검증에서 baseline을 이기면 결측 구간에서도 잘 복원할 것으로 ", 13.5, DARK, False, False, None),
           ("추정", 13.5, ORANGE, True, False, None),
           ("한다. 결측이 무작위라는 보장이 없어(MNAR) 결측 지점 정확도를 단정하지 않고, 대신 결측 분포로 재가중해 얼마나 살아남는지를 측정한다.",
            13.5, DARK, False, False, None)]], line_spacing=1.15)
    return s


# --- 7. Approach: the hard bar -------------------------------------------
def s_bar():
    s = slide()
    header(s, "2. 접근법", "의도적으로 어려운 평가 bar: 미래를 보는 보간 + 인과 GP")
    text(s, Inches(0.55), Inches(1.42), Inches(12.3), Inches(0.62),
         [[("모델을 ", 15, DARK, False, False, None),
           ("오프라인 CES-only 보간(선형 · PCHIP · 국소 AR · GP)", 15, NAVY, True, False, None),
           ("과 비교한다 — 이 보간들은 타겟 주변의 과거+미래 CES를 모두 사용한다.",
            15, DARK, False, False, None)]])
    box(s, Inches(0.55), Inches(2.10), Inches(6.0), Inches(2.55), fill=CARDBG, round_=True)
    box(s, Inches(0.55), Inches(2.10), Inches(0.12), Inches(2.55), fill=ORANGE)
    text(s, Inches(0.8), Inches(2.27), Inches(5.6), Inches(0.5),
         [[("우리 모델 (causal)", 16, ORANGE, True, False, None)]])
    bullets(s, Inches(0.8), Inches(2.80), Inches(5.5), Inches(1.8), [
        ("타겟 시점까지의 빠른 진단 (BES·ECEI·Mirnov)", 0),
        ("과거 CES 이력 — 도달거리는 세그먼트 전체", 0),
        ("미래 CES는 전혀 보지 않음", 0, RED, True),
    ], size=13.5, gap=9)
    box(s, Inches(6.8), Inches(2.10), Inches(6.0), Inches(2.55), fill=CARDBG, round_=True)
    box(s, Inches(6.8), Inches(2.10), Inches(0.12), Inches(2.55), fill=BLUE)
    text(s, Inches(7.05), Inches(2.27), Inches(5.6), Inches(0.5),
         [[("보간 baseline (오프라인)", 16, BLUE, True, False, None)]])
    bullets(s, Inches(7.05), Inches(2.80), Inches(5.5), Inches(1.8), [
        ("타겟 양쪽의 과거 + 미래 CES 이웃 사용", 0),
        ("PCHIP(단조 3차) = 사전등록 headline 기준선", 0),
        ("세그먼트 경계는 넘지 않음 → persistence 폴백", 0),
    ], size=13.5, gap=9)
    box(s, Inches(0.55), Inches(4.85), Inches(12.25), Inches(1.02), fill=CARDBG, round_=True)
    box(s, Inches(0.55), Inches(4.85), Inches(0.12), Inches(1.02), fill=TEAL)
    text(s, Inches(0.85), Inches(4.94), Inches(11.7), Inches(0.9),
         [[("추가된 팔 — 인과 GP: 배치 가능한 가장 강한 경쟁자", 13.5, TEAL, True, False, None)],
          [("같은 GP를 과거 이웃 16개로 제한(NaN 조건 동일 → 모집단 불변). seed 42·컷에서 Tᵢ RMSE 164.3 vs "
            "persistence 197.2 — “배치 가능한 모든 인과 방법을 이긴다”는 이 기준선으로 판정한다.",
            12.5, DARK, False, False, None)]], line_spacing=1.12)
    box(s, Inches(0.55), Inches(6.03), Inches(12.25), Inches(0.85), fill=NAVY, round_=True)
    text(s, Inches(0.85), Inches(6.12), Inches(11.7), Inches(0.72),
         [[("왜 이 정보 비대칭이 핵심인가", 12.5, ORANGE, True, False, None)],
          [("미래까지 보는 보간을 인과 모델이 이긴다면, 빠른 진단이 시간 보간으로는 얻을 수 없는 CES 정보를 "
            "운반한다는 강력한 증거다.", 14, WHITE, False, False, None)]], line_spacing=1.10, space_after=2)
    return s


# --- 8. Validation strategy ----------------------------------------------
def s_validation():
    s = slide()
    header(s, "2. 접근법", "검증 전략과 정직한 caveat")
    cards = [
        ("Masking 복원 검증", BLUE,
         ["관측된 CES 값을 가린 뒤 모델이 복원",
          "물리 단위 역정규화 후 타겟별 채점",
          "전 arm이 동일한 (file, row) 집합·동일 마스크",
          "모델 입력에서 타겟 시점은 완전 마스킹(누수 차단)"]),
        ("Observed-only 측정", TEAL,
         ["관측된 CES 지점에서만 skill 측정",
          "진짜 결측은 참값이 없어 직접 검증 불가",
          "관측 지점이 결측 지점보다 쉬울 수 있음",
          "PR2 폴백률: Tᵢ 0.3–0.4% · V_rot 40–44%"]),
        ("MNAR — 낙관적 상한", ORANGE,
         ["CES 결측은 무작위 아님 (MNAR)",
          "저 SNR · ELM · 천이에서 drop-out",
          "→ observed-only skill = 낙관적 bound",
          "→ 결측 분포로 재가중해 정량화 (결과 ⑤)"]),
    ]
    for i, (t, col, lines) in enumerate(cards):
        card(s, Inches(0.55 + i * 4.13), Inches(1.6), Inches(3.95), Inches(3.0),
             t, lines, accent=col, title_size=15, body_size=12.5)
    box(s, Inches(0.7), Inches(4.95), Inches(11.9), Inches(1.6), fill=CARDBG, round_=True)
    text(s, Inches(0.95), Inches(5.12), Inches(11.4), Inches(1.4),
         [[("Murphy(1988) skill 정의", 14, NAVY, True, False, None)],
          [("skill_vs_pchip = 1 - MSE_model / MSE_pchip", 17, BLUE, True, False, MONO)],
          [("> 0 이면 모델이 보간보다 우수 · = 0 이면 동률. 유의 = shot 군집 paired bootstrap 95% CI가 0을 제외(PR4), "
            "‘n/4’ = 유의한 분할 수.", 13, DARK, False, False, None)]], line_spacing=1.2)
    return s


# --- 9. Data collection ---------------------------------------------------
def s_data():
    s = slide()
    header(s, "3. 데이터 & 파이프라인", "데이터 구성과 세그먼트 구조")
    bullets(s, Inches(0.55), Inches(1.5), Inches(6.6), Inches(4.9), [
        ("641개 KSTAR 방전 (shot 30801–32751) · 10 ms 공통 격자", 0, NAVY, True),
        ("제공 측 선정 기준: 하드웨어 일관성 · H-mode ELM 억제(RMP) 구간", 1),
        ("총 247,207행 (파일당 중앙값 339행)", 1),
        ("행당 채널: BES 9 · ECEI 4 · Mirnov 2 · time · CES_TI · CES_VT", 0),
        ("세그먼트는 ≥0.5 s 간극에서 분리 — 이봉 delta 분포의 골", 0),
        ("(0.1, 0.5) s 구간에 delta 82개뿐 · 세그먼트 안 스텝의 99.4%가 10 ms", 1),
        ("전형적 파일 = 주 세그먼트 1개 (중앙값 301행 ≈ 3.0 s, 10–90분위 1.3–7.0 s)", 0),
        ("2개인 파일 28개 · 10행 넘는 게 없는 파일 20개 · 고립 단일행 1,279개", 1),
        ("어떤 arm도(보간이든 모델 입력이든) 세그먼트 간극을 넘지 않는다", 0, ORANGE, True),
    ], size=14, gap=7)
    box(s, Inches(7.35), Inches(1.5), Inches(5.45), Inches(4.9), fill=NAVY, round_=True)
    text(s, Inches(7.6), Inches(1.66), Inches(5.0), Inches(0.5),
         [[("TEST 규모 (선택이 절대 읽지 않음)", 15, ORANGE, True, False, None)]])
    rows = [
        ("기준 분할", "seed 42", ""),
        ("Tᵢ (컷)", "32,589 행", "96 방전"),
        ("V_rot (컷)", "10,463 행", "60 방전"),
        ("Tᵢ (포함)", "32,721 행", "96 방전"),
        ("V_rot (포함)", "10,461 행", "60 방전"),
        ("4 분할 Tᵢ", "32.6–35.9k 행", "96 방전"),
        ("4 분할 V_rot", "10.5–14.5k 행", "60–66 방전"),
    ]
    yy = 2.24
    for name, val, shots in rows:
        text(s, Inches(7.6), Inches(yy), Inches(1.85), Inches(0.4),
             [[(name, 13, WHITE, True, False, None)]])
        text(s, Inches(9.45), Inches(yy), Inches(1.9), Inches(0.4),
             [[(val, 12.5, ORANGE, True, False, None)]])
        text(s, Inches(11.4), Inches(yy), Inches(1.35), Inches(0.4),
             [[(shots, 11.5, LGRAY, False, False, None)]])
        yy += 0.48
    text(s, Inches(7.6), Inches(yy + 0.08), Inches(5.0), Inches(0.9),
         [[("모든 수치는 단일 collector가 얼린 run 디렉터리에서 읽는다 — 본문·표·그림이 어긋날 수 없다.",
            11.5, LGRAY, False, True, None)]], line_spacing=1.15)
    return s


# --- 10. No fake data + contract -----------------------------------------
def s_contract():
    s = slide()
    header(s, "3. 데이터 & 파이프라인", "No-Fake-Data 원칙과 데이터 계약(contract)")
    cards1 = [
        ("① 가짜 라벨 금지", ORANGE,
         ["학습 행을 만들려고 타겟을 대체(impute)하지 않음",
          "윈도: 진단 입력 완전 + 타겟 ≥1개 관측된 행만 사용",
          "시퀀스: 라벨 없는 행은 맥락으로만 기여",
          "어느 프레이밍도 타겟 행 자신의 값을 읽지 않음"]),
        ("② 타겟별 masked loss", BLUE,
         ["L = Σ m·(예측 - 실측)² / Σ m   (m = 타겟별 관측 마스크)",
          "한쪽 타겟만 관측된 행도 그 타겟은 학습에 기여",
          "두-타겟-필수 필터는 라벨 행의 ≈28%를 조용히 버림",
          "→ 이를 제거한 것은 순수한 데이터 이득"]),
    ]
    for i, (t, col, lines) in enumerate(cards1):
        card(s, Inches(0.55 + i * 6.2), Inches(1.55), Inches(6.0), Inches(2.4),
             t, lines, accent=col, title_size=15, body_size=12.5)
    cards2 = [
        ("③ 누수 삼중 차단", TEAL,
         ["파일(shot) 단위 분할 — 인접 행 자기상관 차단",
          "학습 파일 전용 정규화 (희소 타겟은 NaN-인지)",
          "  시퀀스 모델은 여기에 shot별 입력 표준화를 더함",
          "타겟 시점의 값·관측 flag는 입력에 결코 들어가지 않음"]),
        ("④ held 전면 제거", NAVY,
         ["관측 V_rot의 54%가 계측기 유지값 (진짜 측정 아님)",
          "지도 타겟 · 이력/이월 입력 · 정규화 통계 ·",
          "모든 기준선의 보간 앵커에서 동일하게 제거",
          "→ 어떤 arm도 forward-fill로 공짜 점수를 못 받음"]),
    ]
    for i, (t, col, lines) in enumerate(cards2):
        card(s, Inches(0.55 + i * 6.2), Inches(4.15), Inches(6.0), Inches(2.4),
             t, lines, accent=col, title_size=15, body_size=12.5)
    return s


# --- 11. Split + normalization -------------------------------------------
def s_split():
    s = slide()
    header(s, "3. 데이터 & 파이프라인", "누수 방지 split + 학습셋 전용 정규화")
    cards = [
        ("File-level split (행 누수 방지)", BLUE,
         ["행 단위가 아니라 CSV(shot) 파일 단위로 분할",
          "인접 행 상관 때문에 행 단위 분할은 train/val 누수",
          "고정 split을 디스크에 pin (fixed_*_split.csv)",
          "재로딩 시 데이터 불일치하면 예외 발생"]),
        ("Train-file-only 정규화", TEAL,
         ["BES·ECEI·MC·타겟 per-channel z-score",
          "통계는 학습 파일에서만 추정 (val/test 누수 차단)",
          "타겟 통계는 NaN-aware (관측값만) — CES는 sparse",
          "시퀀스 모델은 추가로 각 방전의 빠른 진단을",
          "  그 방전 자신의 통계로 표준화 (캠페인 전이의 열쇠)"]),
    ]
    for i, (t, col, lines) in enumerate(cards):
        card(s, Inches(0.55 + i * 6.2), Inches(1.55), Inches(6.0), Inches(2.7),
             t, lines, accent=col, title_size=15, body_size=12.5)
    box(s, Inches(0.55), Inches(4.5), Inches(12.25), Inches(2.0), fill=CARDBG, round_=True)
    text(s, Inches(0.8), Inches(4.65), Inches(11.8), Inches(0.5),
         [[("타겟 시점 마스킹 — 누수 차단의 핵심", 15, NAVY, True, False, None)]])
    bullets(s, Inches(0.8), Inches(5.15), Inches(11.8), Inches(1.3), [
        ("윈도 ces_history (batch, W=2, 4): 이전 정규화 Tᵢ · 이전 정규화 V_rot · 타겟별 관측 flag 2개", 0),
        ("Tᵢ·V_rot는 독립적으로 결측(8.2% / 23.9%) → 관측을 타겟별로 추적", 1),
        ("타겟 시점은 값·flag 모두 0으로 완전 마스킹 → 자기 정답 누수 차단 (시퀀스도 동일)", 1, RED, True),
    ], size=13, gap=7)
    return s


# --- 11b. The two framings ------------------------------------------------
def s_samples():
    s = slide()
    header(s, "3. 데이터 & 파이프라인", "학습 예제의 두 프레이밍: 윈도(대조군) vs 전체격자 시퀀스(주 모델)")
    cards = [
        ("① 연속 세그먼트", BLUE,
         ["time delta ≥ 0.5 s = 세그먼트 경계",
          "전형 파일 = 주 세그먼트 1개 (중앙값 301행)",
          "모델 입력·보간 모두 경계를 넘지 않음"]),
        ("② 윈도 프레이밍 (대조군, W=2)", TEAL,
         ["타겟 t 앞 W=2행: bes(2,9)·ecei(2,4)·mc(2,2)",
          "  + time_features(2,4) + ces_history(2,4)",
          "타겟 [Tᵢ, V_rot] + 타겟별 마스크 m ∈ {0,1}²",
          "파일당 샘플 상한 500 (사전등록)"]),
        ("③ 전체격자 시퀀스 (주 모델)", ORANGE,
         ["세그먼트의 입력-완전 행 전부를 맥락으로 유지",
          "  (라벨 유무와 무관 — 희소성은 loss로)",
          "스텝당 22채널 = z-score 빠른 채널 15 +",
          "  log1p(Δt) + 타겟별(이월값·신선도·flag) 3×2"]),
        ("④ 시간 특징 4채널 (윈도)", NAVY,
         ["lookback 초 · 행간 delta 초 · 각각의 log1p",
          "불규칙 관측 패턴을 명시적으로 노출",
          "과거 CES 값의 신뢰도는 10 ms 전인지",
          "  200 ms 전인지에 강하게 의존한다"]),
    ]
    for i, (t, col, lines) in enumerate(cards):
        r, c = divmod(i, 2)
        card(s, Inches(0.55 + c * 6.2), Inches(1.5 + r * 2.5), Inches(6.0), Inches(2.3),
             t, lines, accent=col, title_size=14.5, body_size=12.5)
    text(s, Inches(0.55), Inches(6.55), Inches(12.3), Inches(0.6),
         [[("두 프레이밍의 결정적 차이 = 도달거리(reach): ", 12.5, NAVY, True, False, None),
           ("윈도는 과거 W-1개 관측, 시퀀스는 세그먼트 전체. 이 차이가 백본 관문(결과 ③)에서 측정된다.",
            12.5, GRAY, False, False, None)]])
    return s


# --- 12. Data quality: held values ---------------------------------------
def s_stuck():
    s = slide()
    header(s, "3. 데이터 & 파이프라인", "데이터 품질 감사: held / forward-fill된 CES_VT", accent=RED)
    text(s, Inches(0.55), Inches(1.45), Inches(12.3), Inches(0.9),
         [[("감사 발견: ", 15, RED, True, False, None),
           ("관측된 CES_VT 값의 54%가 계측기 유지값", 15, RED, True, False, None),
           ("(같은 연속 블록 안에서 직전 관측과 bit-identical, 최대 1,214행 연속). "
            "V_rot의 고유 측정 주기가 행 주기보다 느려 값이 carry-forward된 것 — 독립적인 측정이 아니다.",
            14.5, DARK, False, False, None)]], line_spacing=1.16)
    bullets(s, Inches(0.55), Inches(2.55), Inches(6.4), Inches(4.0), [
        ("641개 중 499개 파일이 영향 · CES_TI는 226,991행 중 1행", 0),
        ("오탐 통로 없음: V_rot는 소수점 5자리, 값 간 최소 간격 4e-5", 1),
        ("확정 프로토콜 = 어디서나 제거", 0, RED, True),
        ("지도 타겟 · 이력/이월 입력과 그 관측 flag · 정규화 통계", 1),
        ("모든 기준선의 보간 앵커 — 이 논문의 모든 수치는 진짜 측정만 사용", 1),
        ("held는 평가뿐 아니라 학습도 오염시킨다 (짝지은 재학습으로 확인)", 0, NAVY, True),
        ("forward-fill 계단은 '이력을 복사하는 것이 최적'이라고 가르침", 1),
        ("→ 민감도 한 줄이 아니라 프로토콜로 삼은 이유", 1, NAVY, True),
    ], size=13.5, gap=8)
    box(s, Inches(7.1), Inches(2.55), Inches(5.7), Inches(3.9), fill=CARDBG, round_=True)
    box(s, Inches(7.1), Inches(2.55), Inches(0.12), Inches(3.9), fill=ORANGE)
    text(s, Inches(7.35), Inches(2.70), Inches(5.3), Inches(0.5),
         [[("held 제거가 남기는 대가 — PR2 폴백", 14, ORANGE, True, False, None)]])
    bullets(s, Inches(7.35), Inches(3.22), Inches(5.2), Inches(3.1), [
        ("보간은 모델이 채점되는 모든 곳에서 예측해야 함", 0),
        ("미래 이웃이 없으면 persistence로 후퇴", 1),
        ("폴백률: Tᵢ 채점 행의 0.3–0.4%", 0, GREEN, True),
        ("V_rot 채점 행의 40–44%", 0, RED, True),
        ("→ V_rot의 'vs PCHIP'은 5분의 2가 'vs persistence'", 1, RED, True),
        ("사전등록이 폴백률 보고를 의무화한 이유", 0),
    ], size=12.5, gap=8)
    return s


# --- 13. Architecture: the sequence backbone ------------------------------
def s_arch():
    s = slide()
    header(s, "4. 모델", "주 모델: 전체격자 인과 시퀀스 나우캐스터 seq_v2")
    add_image_fit(s, os.path.join(FIG, "fig_architecture_seq.png"),
                  Inches(0.45), Inches(1.38), Inches(12.45), Inches(5.15))
    text(s, Inches(0.55), Inches(6.62), Inches(12.3), Inches(0.42),
         [[("구조적 라우팅: ", 12.5, NAVY, True, False, None),
           ("Tᵢ 분기(2층 160)는 22채널 전체 상태를, V_rot 분기(1층 64)는 비-빠른 7채널만 읽는다 · 총 357,570 파라미터.",
            12.5, DARK, False, False, None)]], line_spacing=1.12)
    return s


# --- 14. Architecture detail ---------------------------------------------
def s_arch_detail():
    s = slide()
    header(s, "4. 모델", "핵심 설계 결정과 근거")
    cards = [
        ("라우팅은 head가 아니라 인코더에서", BLUE,
         ["순환 상태를 공유하면 head를 어떻게 배선해도",
          "  빠른 진단 정보가 V_rot로 샌다",
          "seq_v2는 분기 자체를 분리 → 빠른 15채널을",
          "  전부 섭동해도 V_rot 출력이 bit-identical"]),
        ("도달거리 = 세그먼트 전체", ORANGE,
         ["W는 더 이상 하이퍼파라미터가 아니다",
          "라벨 없는 행도 맥락으로 유지 (고속 진단은 조밀)",
          "윈도 대조군 대비 pooled Tᵢ +0.081",
          "  (16 run · CI [+0.067, +0.096] · 16/16 양수)"]),
        ("희소성은 loss가 처리한다", TEAL,
         ["세그먼트의 모든 라벨 행에 대한 타겟별 masked MSE",
          "LayerNorm + 작은 GELU head로 마무리",
          "학습: AdamW 1e-3 · batch 16 세그먼트",
          "  val masked MSE 조기 종료(patience 6, 상한 30)"]),
        ("해석 가능한 사다리 칸 b3k8", NAVY,
         ["seq_v2의 프레이밍·라우팅을 유지하고 head만 교체",
          "예측 = 이월값 + Σ w·z + b   (z ∈ [-1, 1], K = 8 / 4)",
          "readout 0 초기화 → 학습이 정확히 persistence에서 출발",
          "21,498 파라미터 = 백본의 6% (결과 ⑨)"]),
    ]
    for i, (t, col, lines) in enumerate(cards):
        r, c = divmod(i, 2)
        card(s, Inches(0.55 + c * 6.2), Inches(1.5 + r * 2.5), Inches(6.0), Inches(2.3),
             t, lines, accent=col, title_size=14.5, body_size=12)
    text(s, Inches(0.55), Inches(6.55), Inches(12.3), Inches(0.6),
         [[("탐색이 가르친 것: ", 12, NAVY, True, False, None),
           ("윈도 계열 두 라운드에서 살아남은 메커니즘은 attention pooling 하나뿐이었고, 확정 프로토콜에서도 같은 교훈이 "
            "두 번 반복된다 — attention 후보는 비유의, 폭 스윕은 평평. 모델 절이 짧고 평가 절이 긴 이유다.",
            12, GRAY, False, False, None)]], line_spacing=1.12)
    return s


# --- 14b. The paired window control --------------------------------------
def s_arch_window():
    s = slide()
    header(s, "4. 모델", "짝지은 대조군: W=2 윈도 모델 (관측 마스킹 attention pooling)")
    add_image_fit(s, os.path.join(FIG, "fig_architecture.png"),
                  Inches(0.45), Inches(1.35), Inches(8.15), Inches(4.55))
    box(s, Inches(8.75), Inches(1.42), Inches(4.05), Inches(4.55), fill=CARDBG, round_=True)
    box(s, Inches(8.75), Inches(1.42), Inches(0.10), Inches(4.55), fill=TEAL)
    text(s, Inches(9.0), Inches(1.55), Inches(3.6), Inches(0.5),
         [[("대조군의 역할", 14.5, TEAL, True, False, None)]])
    bullets(s, Inches(9.0), Inches(2.05), Inches(3.6), Inches(3.8), [
        ("옛 주 모델 (201,258 파라미터)", 0),
        ("모달리티별 time-aware 1-D CNN", 1),
        ("이력은 양방향 GRU(64)", 1),
        ("관측 마스킹 attention pooling", 0, ORANGE, True),
        ("해당 타겟이 실제 관측된 행에만 질량 허용", 1),
        ("보간의 귀납 편향을 파라미터 0으로 이식", 1),
        ("모든 것이 백본과 동일하게 고정", 0, NAVY, True),
        ("데이터 계약 · held 처리 · 분할 · 채점 모집단", 1),
        ("→ 두 모델 비교는 행 단위로 paired", 1, NAVY, True),
    ], size=12, gap=6)
    text(s, Inches(0.55), Inches(6.05), Inches(8.1), Inches(0.9),
         [[("물리 기반 head 라우팅: ", 12, NAVY, True, False, None),
           ("Tᵢ head = 빠른진단+시간+이력 / V_rot head = 이력+시간. 이 구조는 ~40회 keep/discard 통제 실험의 결과이며, "
            "지금은 (i) 백본 관문의 비교 대상, (ii) 절제 실험의 무대, (iii) 캠페인 붕괴의 재현자로 남는다.",
            12, DARK, False, False, None)]], line_spacing=1.14)
    return s


# --- 14c. Training configuration ------------------------------------------
def s_training():
    s = slide()
    header(s, "4. 모델", "어떻게 학습시켰나 — 손실 함수와 최적화 설정")
    card(s, Inches(0.55), Inches(1.5), Inches(6.0), Inches(3.05),
         "손실 — 타겟별 masked MSE",
         ["L = Σ m·(예측 - 실측)² / Σ m   (Tᵢ · V_rot 각각 계산)",
          "관측된 타겟만 손실에 기여 · 결측 타겟은 마스킹으로 제외",
          "Tᵢ·V_rot 독립 결측(8.2% / 23.9%) — 한쪽만 있어도 학습",
          "시퀀스: 세그먼트의 모든 라벨 행에 대해 계산",
          "  (라벨 없는 행은 맥락으로만 기여)",
          "출력은 정규화 단위 — 역정규화는 평가에서만"],
         accent=BLUE, title_size=14.5, body_size=12.5)
    card(s, Inches(6.75), Inches(1.5), Inches(6.0), Inches(3.05),
         "최적화 설정",
         ["백본 seq_v2: AdamW 1e-3 · batch 16 세그먼트",
          "val masked MSE 조기 종료(patience 6, 상한 30 epoch)",
          "  확정 실행은 14–25 epoch에서 종료",
          "윈도 대조군: 같은 데이터 계약 · 파일당 샘플 상한 500",
          "분할 seed와 초기화 seed를 분리 (분할 4 × 초기화 4)",
          "loss가 비유한이면 즉시 실패(fail-loud) — 조용한 NaN 금지"],
         accent=TEAL, title_size=14.5, body_size=12.5)
    box(s, Inches(0.55), Inches(4.80), Inches(12.25), Inches(1.6), fill=CARDBG, round_=True)
    text(s, Inches(0.8), Inches(4.95), Inches(11.8), Inches(0.5),
         [[("아키텍처는 어디서 왔나 — 깨끗한 val skill을 게이트로 쓴 keep/discard 통제 루프", 14, NAVY, True, False, None)]])
    bullets(s, Inches(0.8), Inches(5.43), Inches(11.8), Inches(0.9), [
        ("반복마다 구조 변경은 딱 하나 → 처음부터 재학습 → 증강 없는 검증셋의 skill_vs_pchip으로 채점", 0),
        ("증강 val loss는 보간이 이미 강한 곳에서 평활화를 보상하므로 쓰지 않는다 · TEST는 전 과정 봉인", 1, NAVY, True),
    ], size=12.5, gap=6)
    return s


# --- 15. Evaluation methodology: split + prereg --------------------------
def s_methodology():
    s = slide()
    header(s, "5. 평가 방법론", "선택 편향 없는 3-way split + 사전등록")
    cards = [
        ("3-way split · TEST 동결", BLUE,
         ["TEST는 아키텍처 탐색 시작 전 예약, 선택 중 절대 안 봄",
          "모델 선택은 val에서만 → 헤드라인에 winner's curse 없음",
          "전 arm이 동일한 (file, row) 집합 · 동일 마스크로 채점",
          "짝지은 비교 전에 모집단 키가 bit-identical임을 검증",
          "보간은 타겟 자신의 값을 제외하고 이웃만 읽음(누수 없음)"]),
        ("사전등록 (PR1–PR4 + 확정 프로토콜)", ORANGE,
         ["PR1 헤드라인 기준선 = PCHIP (사다리 전체도 함께 보고)",
          "PR2 보간은 모든 채점 지점에서 예측, 폴백률 보고 의무",
          "PR3 TEST 하한 ≥15 방전 · ≥3,000 Tᵢ 샘플 (충족)",
          "PR4 shot 군집 bootstrap 95% CI가 0을 제외 = PASS",
          "추가: held-free · W=2 · 파일당 500 · 두 모집단 ·",
          "     TEST 채점 전 모델 결정 규칙 커밋 · 문턱 민감도"]),
    ]
    for i, (t, col, lines) in enumerate(cards):
        card(s, Inches(0.55 + i * 6.2), Inches(1.5), Inches(6.0), Inches(2.5),
             t, lines, accent=col, title_size=15, body_size=12)
    cw = [Inches(3.05), Inches(5.4), Inches(3.8)]
    rows = [
        ["Persistence", "마지막 관측 CES", ("인과", GRAY, True, None)],
        ["AR (국소)", "과거 CES만", ("인과", GRAY, True, None)],
        ["인과 GP", "과거 CES 이웃 16개 (NaN 조건 동일)", ("인과 · 최강 배치 기준선", TEAL, True, None)],
        ["선형 / PCHIP*", "타겟 양쪽의 과거 + 미래 CES 이웃", ("오프라인", BLUE, True, None)],
        ["GP (오프라인)", "Matérn-3/2 + 백색잡음, 이웃 16+16", ("오프라인 · 최강 평활기", BLUE, True, None)],
    ]
    table(s, Inches(0.55), Inches(4.15), cw, ["기준선 사다리", "정보 접근", "분류"], rows,
          row_h=Inches(0.40), head_h=Inches(0.40), size=12.5, head_size=12.5)
    text(s, Inches(0.55), Inches(6.60), Inches(12.3), Inches(0.42),
         [[("* PR1 헤드라인. ", 11.5, GRAY, True, False, None),
           ("보간·모델 모두 세그먼트 경계를 넘지 않고, 경계 밖 이웃이 필요하면 보간은 persistence 값을 대신 예측한다(모집단 축소 없음).",
            11.5, GRAY, False, False, None)]])
    return s


# --- 16. Bootstrap --------------------------------------------------------
def s_bootstrap():
    s = slide()
    header(s, "5. 평가 방법론", "Shot 군집 paired bootstrap — 왜 shot이 단위인가")
    bullets(s, Inches(0.55), Inches(1.55), Inches(7.0), Inches(3.6), [
        ("한 방전(shot) 내 인접 CES 행은 강하게 상관됨", 0),
        ("개별 샘플을 독립으로 보면 불확실성을 크게 과소평가", 1, RED, True),
        ("PR4 검정: 샘플별 짝지은 오차 (SE_model - SE_pchip)를", 0),
        ("shot 단위로 묶고, shot 전체를 복원추출 (B = 10,000, 고정 seed)", 1),
        ("skill 95% CI가 모델에 유리한 방향으로 0을 제외 = PASS", 0, GREEN, True),
        ("→ CI가 within-shot 가짜 복제가 아닌 진짜 shot-to-shot 일반화 반영", 0),
        ("모델 대 모델 비교도 같은 행 위에서 같은 paired bootstrap", 0, NAVY, True),
        ("백본 vs 윈도 대조군 · 사다리 칸 vs 백본", 1),
    ], size=14, gap=10)
    box(s, Inches(7.85), Inches(1.6), Inches(4.95), Inches(4.5), fill=NAVY, round_=True)
    text(s, Inches(8.1), Inches(1.8), Inches(4.5), Inches(0.5),
         [[("재현 단위 = shot", 16, ORANGE, True, False, None)]])
    text(s, Inches(8.1), Inches(2.45), Inches(4.5), Inches(3.5),
         [[("개별 행", 13, LGRAY, True, False, None)],
          [("→ 자기상관, 가짜 복제 (X)", 12.5, MGRAY, False, False, None)],
          [("", 6, WHITE, False, False, None)],
          [("shot 전체", 13, WHITE, True, False, None)],
          [("→ 독립 복제의 단위 (O)", 12.5, RGBColor(0x9D, 0xE8, 0xCD), False, False, None)],
          [("", 6, WHITE, False, False, None)],
          [("Tᵢ ≈ 96 방전 · V_rot 60–66 방전", 13, WHITE, True, False, None)],
          [("이 방전 수가 검정력의 상한이며", 12, LGRAY, False, False, None)],
          [("모든 유의성 판정의 구속 조건이다", 12, LGRAY, False, False, None)]],
         line_spacing=1.18, space_after=3)
    return s


# --- 17. Pre-registered model selection protocol -------------------------
def s_res_protocol():
    s = slide()
    header(s, "5. 평가 방법론", "모델 선택 프로토콜 — 규칙을 수치보다 먼저 적는다", accent=TEAL)
    text(s, Inches(0.55), Inches(1.4), Inches(12.3), Inches(0.55),
         [[("모든 모델 결정은 검증 데이터 위에서, 또는 ", 14, DARK, False, False, None),
           ("해당 TEST 채점 이전에 문서로 확정된 결정 규칙", 14, TEAL, True, False, None),
           (" 아래에서 이루어졌고, TEST는 결정마다 한 번만 채점되었다.", 14, DARK, False, False, None)]])
    card(s, Inches(0.55), Inches(2.05), Inches(6.0), Inches(2.5),
         "백본 관문 — 4조건을 먼저 고정하고 그다음 충족",
         ["① 4개 분할 전부에서 부호 유지",
          "② 통합 실행 군집 CI가 0을 제외",
          "③ 예산 균등화(고정 10 epoch)에서도 부호 유지",
          "④ V_rot 손실 없음",
          "→ 충족되어 시퀀스 프레이밍을 백본으로 채택 (결과 ③)"],
         accent=NAVY, title_size=14.5, body_size=12.5)
    card(s, Inches(6.75), Inches(2.05), Inches(6.0), Inches(2.5),
         "유일한 아키텍처 후보 — 미승격",
         ["seq_v2 + 관측 마스킹 인과 attention (0-초기화 사영)",
          "백본 대비 4/4 분할 양수:",
          "  +0.009 / +0.013 / +0.033 / +0.020",
          "그러나 유의는 1/4 → 사전 기준 ≥3/4 미달 → 미승격",
          "val에선 2/2 유의였다 — 승격 bar를 TEST에 두는 이유"],
         accent=ORANGE, title_size=14.5, body_size=12.5)
    box(s, Inches(0.55), Inches(4.75), Inches(12.25), Inches(1.95), fill=CARDBG, round_=True)
    text(s, Inches(0.8), Inches(4.88), Inches(11.8), Inches(0.5),
         [[("나머지 결정 규칙도 같은 방식으로", 14, NAVY, True, False, None)]])
    bullets(s, Inches(0.8), Inches(5.36), Inches(11.8), Inches(1.3), [
        ("윈도 계열: 한 번에 하나만 바꾸고 깨끗한 val skill로 keep/discard · 이력 길이는 24-run 스윕(plateau 최소) · held는 감사로 제거", 0),
        ("사다리 칸·폭 스윕: 두 갈래 판정과 서술적 읽기(상한 / 무릎)를 TEST 채점 전에 문서화", 0),
        ("스윕 위에서 백본을 재선정하는 것은 구성상 금지", 1, RED, True),
    ], size=12.5, gap=6)
    return s


# --- 18. Result 1: RMSE ladder -------------------------------------------
def s_res_ladder():
    s = slide()
    header(s, "6. 결과 ①", "인과 기준선을 압도하고, 오프라인 평활기와 동률")
    add_image_fit(s, os.path.join(FIG, "fig_rmse_ladder.png"),
                  Inches(0.55), Inches(1.42), Inches(12.25), Inches(4.35))
    box(s, Inches(0.55), Inches(5.85), Inches(12.25), Inches(1.1), fill=CARDBG, round_=True)
    text(s, Inches(0.85), Inches(5.95), Inches(11.7), Inches(0.95),
         [[("백본이 두 타겟 모두에서 최저 RMSE. ", 13, NAVY, True, False, None),
           ("인과 GP보다 Tᵢ 4% · V_rot 18% 낮고, 미래를 쓰는 오프라인 GP(153.8)와는 동률(157.8)이다.",
            13, DARK, False, False, None)],
          [("포함 모집단: ", 12.5, NAVY, True, False, None),
           ("seq_v2 363.0 / 23.7 · PCHIP 412.4 / 30.2 · 인과 GP 394.6 / 28.8 · persistence 478.0 / 33.4 — "
            "스파이크가 Tᵢ RMSE를 두 배 이상 키우지만 순서는 불변.", 12.5, DARK, False, False, None)]],
         line_spacing=1.14)
    return s


# --- 19. Result 2: headline forest ---------------------------------------
def s_res_forest():
    s = slide()
    header(s, "6. 결과 ②", "헤드라인 — Tᵢ는 두 모집단 모두에서 4/4")
    add_image_fit(s, os.path.join(FIG, "fig_forest.png"),
                  Inches(0.55), Inches(1.38), Inches(12.25), Inches(3.85))
    cards = [
        ("CES_TI — 무조건부 PASS", GREEN,
         ["컷 +0.174 / +0.248 / +0.257 / +0.264 (평균 +0.236) · 4/4",
          "포함 +0.225 / +0.238 / +0.292 / +0.316 (평균 +0.268) · 4/4",
          "8개 셀 전부 인과 GP(+0.08~+0.17)·persistence(+0.36~+0.46)도 승",
          "오프라인 GP와는 동률(-0.05~+0.11, 1/8 유의) — 상한을 명시"]),
        ("CES_VT — 동률 보고", GRAY,
         ["점추정은 8/8 양수지만 PR4는 컷 1/4 · 포함 2/4 (잡음 수준)",
          "vs persistence 3/4 양쪽(+0.30~+0.50) · vs 인과 GP 2/4",
          "포함 수치가 높은 이유: 스파이크가 보간 앵커를 오염시켜",
          "  모든 arm이 PCHIP 대비 더 좋아 보인다 → 결과 ⑨에서 분해"]),
    ]
    for i, (t, col, lines) in enumerate(cards):
        card(s, Inches(0.55 + i * 6.2), Inches(5.22), Inches(6.0), Inches(1.72),
             t, lines, accent=col, title_size=13.5, body_size=10.5)
    return s


# --- 20. Result 3: the B.1 backbone gate ---------------------------------
def s_res_gate():
    s = slide()
    header(s, "6. 결과 ③", "전체격자 프레이밍 vs 윈도 대조군 — 백본 관문 (B.1)")
    add_image_fit(s, os.path.join(FIG, "fig_gate_b1.png"),
                  Inches(0.45), Inches(1.40), Inches(8.05), Inches(4.3))
    box(s, Inches(8.65), Inches(1.42), Inches(4.15), Inches(4.35), fill=CARDBG, round_=True)
    box(s, Inches(8.65), Inches(1.42), Inches(0.10), Inches(4.35), fill=NAVY)
    text(s, Inches(8.9), Inches(1.55), Inches(3.7), Inches(0.5),
         [[("16 run paired 결과 (컷)", 14.5, NAVY, True, False, None)]])
    bullets(s, Inches(8.9), Inches(2.05), Inches(3.7), Inches(3.6), [
        ("Tᵢ 16/16 양수 · 13/16 유의", 0, GREEN, True),
        ("분할별 초기화 평균", 0),
        ("+0.129 / +0.059 / +0.078 / +0.058", 1),
        ("pooled +0.081", 0, GREEN, True),
        ("run 군집 CI [+0.067, +0.096]", 1),
        ("예산 균등화에서도 4/4 부호 유지", 0),
        ("+0.063 / +0.033 / +0.045 / +0.030", 1),
        ("V_rot 유의 열세 0/16 (우세 8/16)", 0, NAVY, True),
    ], size=12, gap=7)
    box(s, Inches(0.55), Inches(5.85), Inches(12.25), Inches(1.1), fill=CARDBG, round_=True)
    text(s, Inches(0.85), Inches(5.95), Inches(11.7), Inches(0.95),
         [[("확증 4 분할에서 같은 비교(seq - 윈도): ", 12.5, NAVY, True, False, None),
           ("컷 +0.130 / +0.058 / +0.062 / +0.044, 포함 +0.053 / +0.024 / +0.047 / +0.029 (8/8 양수, 각 2/4 유의).",
            12.5, DARK, False, False, None)],
          [("무엇을 사는가: ", 12.5, ORANGE, True, False, None),
           ("윈도 대조군은 인과 GP와 동률(1/4)이지만 시퀀스 백본은 4/4+4/4 — 세그먼트 과거 전체로의 도달거리가 "
            "최강 배치 기준선을 이기게 한다. 비용은 음수다(윈도 조립·조합 증강이 없어 학습비 1/10).",
            12.5, DARK, False, False, None)]], line_spacing=1.14)
    return s


# --- 21. Result 4: gap-stratified ----------------------------------------
def s_res_gap():
    s = slide()
    header(s, "6. 결과 ④", "간극 영역 — 비인접 시점에서도 이긴다 (4 분할 통합)")
    text(s, Inches(0.55), Inches(1.38), Inches(12.3), Inches(0.55),
         [[("분할별로는 넓은 Δt 층의 표본이 수십 개뿐이라 4개 test 분할을 합치고 방전 단위로 군집화했다. "
            "PCHIP은 미래 앵커를 읽고 모델은 읽지 않는다.", 13, DARK, False, False, None)]], line_spacing=1.12)
    cw = [Inches(2.35), Inches(2.45), Inches(1.2), Inches(3.15), Inches(3.1)]
    rows = [
        [("Tᵢ  ≤ 15 ms", DARK, True, None), ("134,546 / 135,317", GRAY, False, MONO), ("301", GRAY, False, MONO),
         ("+0.239 [+0.197, +0.274]", GREEN, True, MONO), ("+0.299 [+0.244, +0.347]", GREEN, True, MONO)],
        [("Tᵢ  > 15 ms", DARK, True, None), ("3,422 / 3,334", GRAY, False, MONO), ("265 / 263", GRAY, False, MONO),
         ("+0.268 [+0.187, +0.337]", GREEN, True, MONO), ("+0.206 [+0.108, +0.290]", GREEN, True, MONO)],
        [("Tᵢ  > 45 ms", DARK, True, None), ("460 / 429", GRAY, False, MONO), ("104 / 101", GRAY, False, MONO),
         ("+0.267 [+0.092, +0.414]", GREEN, True, MONO), ("-0.004 [-0.304, +0.246]", RED, True, MONO)],
        [("V_rot  ≤ 15 ms", DARK, True, None), ("51,689", GRAY, False, MONO), ("197", GRAY, False, MONO),
         ("+0.233 [+0.020, +0.318]", GREEN, True, MONO), ("+0.233 [+0.020, +0.317]", GREEN, True, MONO)],
        [("V_rot  > 15 ms", DARK, True, None), ("466 / 456", GRAY, False, MONO), ("130", GRAY, False, MONO),
         ("+0.418 [+0.104, +0.680]", GREEN, True, MONO), ("+0.432 [+0.128, +0.696]", GREEN, True, MONO)],
        [("V_rot  > 45 ms", DARK, True, None), ("14", GRAY, False, MONO), ("7", GRAY, False, MONO),
         ("미채점", MGRAY, False, None), ("미채점", MGRAY, False, None)],
    ]
    table(s, Inches(0.55), Inches(2.02), cw,
          ["Δt 구간", "n (컷 / 포함)", "방전", "컷: vs PCHIP (95% CI)", "포함: vs PCHIP (95% CI)"],
          rows, row_h=Inches(0.46), head_h=Inches(0.44), size=11.5, head_size=12)
    box(s, Inches(0.55), Inches(5.42), Inches(12.25), Inches(1.45), fill=CARDBG, round_=True)
    text(s, Inches(0.85), Inches(5.54), Inches(11.7), Inches(1.3),
         [[("세 가지 읽기", 13, NAVY, True, False, None)],
          [("① Tᵢ 우위는 인접 이력에 국한되지 않는다 — >15 ms에서 두 모집단 모두 PASS (persistence 대비 +0.40 / +0.43).",
            12.5, DARK, False, False, None)],
          [("② >45 ms는 컷에서 승·포함에서 동률 — 429행/101 방전은 스파이크 앵커 몇 행이 층을 지배할 수 있는 규모다.",
            12.5, DARK, False, False, None)],
          [("③ V_rot는 보간이 가장 어려운 >15 ms에서 두 모집단 모두 PCHIP를 이긴다 — 논문의 유일한 V_rot 무조건부 양성.",
            12.5, TEAL, True, False, None)]], line_spacing=1.12, space_after=2)
    return s


# --- 22. Result 5: MNAR reweighting ---------------------------------------
def s_stress():
    s = slide()
    header(s, "6. 결과 ⑤", "스트레스 ① — 실제 결측점으로 재가중 (MNAR)", accent=ORANGE)
    text(s, Inches(0.55), Inches(1.36), Inches(12.3), Inches(0.55),
         [[("층 = Δt(15/25/45 ms) × 입력만의 활동 flag. 결측 행의 층 분포로 채점 지점을 재가중(30 미만 층은 기각, "
            "가중 격자에도 컷 적용). 도달 범위: 결측 Tᵢ의 ", 12.5, DARK, False, False, None),
           ("54–68%", 12.5, NAVY, True, False, None),
           (", 결측 V_rot의 ", 12.5, DARK, False, False, None),
           ("4–6%", 12.5, RED, True, False, None),
           ("만 in-domain → 재가중 V_rot은 결론 없음.", 12.5, DARK, False, False, None)]], line_spacing=1.12)
    cw = [Inches(1.25), Inches(1.0), Inches(1.7), Inches(4.15), Inches(4.15)]
    rows = [
        [("컷", NAVY, True, None), "42", ("+0.174", GRAY, False, MONO),
         ("+0.140 [-0.071, +0.290]", GRAY, False, MONO), ("+0.398 [+0.278, +0.518]", GREEN, True, MONO)],
        [("컷", NAVY, True, None), "1", ("+0.248", GRAY, False, MONO),
         ("+0.164 [+0.062, +0.250]", GREEN, True, MONO), ("+0.366 [+0.299, +0.448]", GREEN, True, MONO)],
        [("컷", NAVY, True, None), "7", ("+0.257", GRAY, False, MONO),
         ("+0.203 [-0.024, +0.346]", GRAY, False, MONO), ("+0.310 [+0.227, +0.398]", GREEN, True, MONO)],
        [("컷", NAVY, True, None), "123", ("+0.264", GRAY, False, MONO),
         ("+0.283 [+0.069, +0.392]", GREEN, True, MONO), ("+0.381 [+0.246, +0.464]", GREEN, True, MONO)],
        [("포함", ORANGE, True, None), "42", ("+0.225", GRAY, False, MONO),
         ("+0.140 [+0.033, +0.250]", GREEN, True, MONO), ("+0.443 [+0.141, +0.623]", GREEN, True, MONO)],
        [("포함", ORANGE, True, None), "1", ("+0.238", GRAY, False, MONO),
         ("+0.217 [+0.086, +0.319]", GREEN, True, MONO), ("+0.383 [+0.273, +0.536]", GREEN, True, MONO)],
        [("포함", ORANGE, True, None), "7", ("+0.292", GRAY, False, MONO),
         ("+0.167 [+0.067, +0.265]", GREEN, True, MONO), ("+0.283 [+0.172, +0.380]", GREEN, True, MONO)],
        [("포함", ORANGE, True, None), "123", ("+0.316", GRAY, False, MONO),
         ("+0.221 [+0.050, +0.320]", GREEN, True, MONO), ("+0.337 [+0.164, +0.455]", GREEN, True, MONO)],
    ]
    table(s, Inches(0.55), Inches(2.02), cw,
          ["모집단", "분할", "비가중 vs PCHIP", "결측 정합: vs PCHIP", "결측 정합: vs persistence"],
          rows, row_h=Inches(0.36), head_h=Inches(0.40), size=11.5, head_size=11.5)
    box(s, Inches(0.55), Inches(5.42), Inches(12.25), Inches(1.45), fill=CARDBG, round_=True)
    text(s, Inches(0.85), Inches(5.54), Inches(11.7), Inches(1.3),
         [[("Tᵢ 우위는 온라인 시스템의 실제 상대(persistence)에 대해 두 모집단 4/4 생존", 13, GREEN, True, False, None),
           ("  (+0.28~+0.44, MNAR 보정 비용 최대 0.12).", 13, DARK, False, False, None)],
          [("PCHIP 대비 점추정은 +0.14~+0.28로 유지되지만 고정 가중 CI가 컷 2개 분할에서 0을 지난다 (컷 2/4 · 포함 4/4).",
            12.5, DARK, False, False, None)],
          [("→ 진술: 실제 결측·in-domain 시점에서 나우캐스터는 모든 인과 CES-only 방법보다 유의하게 낫고, "
            "오프라인 보간보다는 모집단 조건부로 낫다.", 12.5, NAVY, True, False, None)]],
         line_spacing=1.12, space_after=2)
    return s


# --- 23. Result 6: campaign (time) split ---------------------------------
def s_res_campaign():
    s = slide()
    header(s, "6. 결과 ⑥", "스트레스 ② — 캠페인(시간) 분할: 윈도는 붕괴, 백본은 생존", accent=ORANGE)
    add_image_fit(s, os.path.join(FIG, "fig_campaign.png"),
                  Inches(0.45), Inches(1.36), Inches(7.85), Inches(3.25))
    box(s, Inches(8.45), Inches(1.36), Inches(4.35), Inches(3.3), fill=CARDBG, round_=True)
    box(s, Inches(8.45), Inches(1.36), Inches(0.10), Inches(3.3), fill=ORANGE)
    text(s, Inches(8.7), Inches(1.47), Inches(3.9), Inches(0.5),
         [[("설계와 수치 (Tᵢ vs PCHIP)", 13.5, ORANGE, True, False, None)]])
    bullets(s, Inches(8.7), Inches(1.94), Inches(3.9), Inches(2.7), [
        ("train 416 (30801–31991)", 0),
        ("val 128 (32002–32310) · test 97 (32312–32751)", 1),
        ("초기화 seed 4개 (분할 4개가 아님)", 1, RED, True),
        ("윈도 OFF: 컷 2/4 · 포함 0/4 · 인과 GP 0/4", 0, RED, True),
        ("윈도 ON(shot별 표준화, 컷): 4/4", 0, TEAL, True),
        ("seq_v2: 컷 +0.187/+0.174/+0.181/+0.177", 0, GREEN, True),
        ("포함 +0.173/+0.202/+0.198/+0.184 → 4/4+4/4", 1, GREEN, True),
    ], size=11.5, gap=6)
    cw = [Inches(4.35), Inches(4.0), Inches(3.9)]
    rows = [
        [("무작위 분할 · 관측점 (결과 ②)", DARK, True, None),
         ("4/4 · 4/4,  +0.17~+0.32", GREEN, True, None), ("4/4 · 4/4 vs 인과 GP,  +0.08~+0.17", GREEN, True, None)],
        [("결측점 재가중 (결과 ⑤)", DARK, True, None),
         ("2/4 · 4/4,  +0.14~+0.28", ORANGE, True, None), ("4/4 · 4/4 vs persistence,  +0.28~+0.44", GREEN, True, None)],
        [("캠페인 시간 분할 (이 슬라이드)", DARK, True, None),
         ("4/4 · 4/4,  +0.17~+0.20", GREEN, True, None), ("4/4 · 4/4 vs 인과 GP,  +0.11~+0.16", GREEN, True, None)],
    ]
    table(s, Inches(0.55), Inches(4.80), cw,
          ["Tᵢ 평가 (컷 · 포함)", "vs PCHIP (오프라인)", "vs 인과 기준선"], rows,
          row_h=Inches(0.44), head_h=Inches(0.42), size=12, head_size=12)
    text(s, Inches(0.55), Inches(6.52), Inches(12.3), Inches(0.6),
         [[("원인은 단언이 아니라 측정: ", 12, NAVY, True, False, None),
           ("train→test 드리프트가 BES 1.22σ · ECEI 0.53σ인데 타겟은 0.115σ. seq-윈도 마진은 8/8 유의, "
            "V_rot도 persistence 대비 seq 4/4 양쪽(윈도 0/4). 남는 주의: 한 시간 블록 위의 초기화 4개, 컷 run 2/4가 상한 종료.",
            12, GRAY, False, False, None)]], line_spacing=1.12)
    return s


# --- 24. Result 7: asymmetry + ablation ----------------------------------
def s_res_asym():
    s = slide()
    header(s, "6. 결과 ⑦", "Tᵢ ↔ V_rot 정보 비대칭 — 본 연구의 과학적 발견")
    add_image_fit(s, os.path.join(FIG, "fig_ablation.png"),
                  Inches(0.45), Inches(1.42), Inches(7.4), Inches(5.2))
    box(s, Inches(7.95), Inches(1.5), Inches(4.85), Inches(5.05), fill=CARDBG, round_=True)
    text(s, Inches(8.2), Inches(1.64), Inches(4.4), Inches(0.5),
         [[("무엇을 지우면 무엇이 사라지나", 14.5, NAVY, True, False, None)]])
    bullets(s, Inches(8.2), Inches(2.15), Inches(4.4), Inches(4.3), [
        ("이력은 두 타겟 모두 필수", 0, NAVY, True),
        ("no_history: Tᵢ -2.11 / -1.16, V_rot -2.89 / -3.51", 1),
        ("Tᵢ: 컷 모집단의 마진은 빠른 진단 정보", 0, ORANGE, True),
        ("no_fast -0.125 (paired 4/4 유의 감소)", 1),
        ("물리 채널 = 충돌 e–i 결합 (ECEI Tₑ · BES nₑ)", 1),
        ("포함 모집단엔 스파이크-강건성 성분이 섞임", 0, RED, True),
        ("이력-전용도 PCHIP를 +0.15~+0.23 이김", 1),
        ("빠른 채널이 더하는 건 0.03–0.09", 1),
        ("V_rot: 정보는 전부 CES 이력", 0, BLUE, True),
        ("빠른 채널 0으로 만들면 출력 bit-identical (8/8)", 1),
        ("NBI 토크 미관측 + Mirnov 100 Hz 앨리어싱", 1),
        ("V_rot의 비-승리는 실패가 아니라 발견", 0, NAVY, True),
    ], size=11.5, gap=5)
    return s


# --- 25. Result 8: window sweep ------------------------------------------
def s_window_sweep():
    s = slide()
    header(s, "6. 결과 ⑧", "이력은 얼마나 필요한가 — 관측 하나 (윈도 스윕, W=2의 근거)", accent=TEAL)
    text(s, Inches(0.55), Inches(1.36), Inches(12.3), Inches(0.56),
         [[("W ∈ {2,3,4,6,8} × seed 4개 + history-0 = ", 13, DARK, False, False, None),
           ("독립 run 24회", 13, NAVY, True, False, None),
           (". held-free 학습·평가, 파일당 500, 컷 없음(이 스윕의 얼린 W=2 run이 포함 모집단 대조군). "
            "각 run은 자기 자신의 held-out TEST skill_vs_pchip으로 채점.", 13, DARK, False, False, None)]],
         line_spacing=1.12)
    add_image_fit(s, os.path.join(FIG, "fig_window_sweep.png"),
                  Inches(0.45), Inches(1.98), Inches(7.75), Inches(4.15))
    box(s, Inches(8.35), Inches(1.98), Inches(4.45), Inches(4.15), fill=CARDBG, round_=True)
    text(s, Inches(8.6), Inches(2.12), Inches(4.0), Inches(0.5),
         [[("곡선이 말하는 세 가지", 15, NAVY, True, False, None)]])
    bullets(s, Inches(8.6), Inches(2.66), Inches(4.0), Inches(3.4), [
        ("history가 없으면 무너진다", 0, RED, True),
        ("Tᵢ -0.026 (0/4) · V_rot -0.783", 1),
        ("과거 관측 1개가 전부를 만든다", 0, GREEN, True),
        ("W=2: Tᵢ +0.238 (4/4) · V_rot +0.206", 1),
        ("이후 평평 (Tᵢ 0.190–0.246)", 1),
        ("점 내부 seed 산포 0.07–0.16 > 곡선 전체", 1),
        ("→ plateau 최소 W = 2", 0, TEAL, True),
        ("넓은 W의 유일한 논거는 커버리지인데", 1),
        ("그건 시퀀스 프레이밍의 논거다", 1),
    ], size=12, gap=7)
    text(s, Inches(0.55), Inches(6.22), Inches(12.3), Inches(0.62),
         [[("결론  ", 12.5, TEAL, True, False, None),
           ("W=3 +0.246(4/4) · W=4 +0.221(3/4) · W=6 +0.190(3/4) · W=8 +0.216(4/4) — 곡선은 평평하다. "
            "이력을 늘려 얻는 것은 skill이 아니라 >15 ms 채점 커버리지(456→1,958)이고, 그것은 도달거리 = 세그먼트 전체인 "
            "시퀀스 프레이밍이 W 없이 해결한다.", 12.5, DARK, False, False, None)]], line_spacing=1.15)
    return s


# --- 26. Result 9: complexity ladder + width sweep ------------------------
def s_res_scaling():
    s = slide()
    header(s, "6. 결과 ⑨", "복잡도 사다리와 크기 축 — 상한은 추정기가 아니라 정보")
    add_image_fit(s, os.path.join(FIG, "fig_ladder_scaling.png"),
                  Inches(0.45), Inches(1.36), Inches(7.55), Inches(3.35))
    box(s, Inches(8.15), Inches(1.36), Inches(4.65), Inches(3.35), fill=CARDBG, round_=True)
    box(s, Inches(8.15), Inches(1.36), Inches(0.10), Inches(3.35), fill=NAVY)
    text(s, Inches(8.4), Inches(1.47), Inches(4.2), Inches(0.5),
         [[("Tᵢ skill (컷 / 포함)", 13.5, NAVY, True, False, None)]])
    rowsx = [
        ("Persistence", "0", "-0.264 / -0.288", GRAY),
        ("Anchor+Δ", "1,258", "-0.261 / -0.287", GRAY),
        ("b3k8", "21,498", "+0.237 / +0.126", GREEN),
        ("윈도 대조군", "201,258", "+0.173 / +0.238", DARK),
        ("seq_v2 백본", "357,570", "+0.236 / +0.268", NAVY),
    ]
    yy = 2.02
    for name, par, val, col in rowsx:
        text(s, Inches(8.4), Inches(yy), Inches(1.65), Inches(0.36),
             [[(name, 12, col, True, False, None)]])
        text(s, Inches(10.0), Inches(yy), Inches(1.0), Inches(0.36),
             [[(par, 11.5, GRAY, False, False, MONO)]], align=PP_ALIGN.RIGHT)
        text(s, Inches(11.05), Inches(yy), Inches(1.65), Inches(0.36),
             [[(val, 11.5, col, True, False, MONO)]], align=PP_ALIGN.RIGHT)
        yy += 0.42
    text(s, Inches(8.4), Inches(yy + 0.06), Inches(4.2), Inches(0.8),
         [[("b3 - anchor: 컷 +0.35~+0.42 · 포함 +0.29~+0.34 (양쪽 4/4 유의)",
            11, DARK, False, False, None)]], line_spacing=1.12)
    card(s, Inches(0.55), Inches(4.85), Inches(6.0), Inches(2.05),
         "컷: 백본의 Tᵢ skill 전부가 유계 수 8개 + persistence로 압축", [
             "b3 - seq_v2 평균 +0.002 (CI 전부 0 포함) · PR4 4/4 · 인과 GP 4/4",
             "probe: latent은 직전 Tᵢ(R² 0.47–0.75)와 ECEI Tₑ(0.31–0.48)를",
             "  분산 부호화 · 보정은 예측 분산의 25–39%",
             "포함에선 -0.194 (4/4 유의): 유계 보정으로는 스파이크 앵커를",
             "  못 살린다 — ≈1% 행이 모든 arm SSE의 70–83%",
         ], accent=TEAL, title_size=13, body_size=11.5)
    card(s, Inches(6.8), Inches(4.85), Inches(6.0), Inches(2.05),
         "크기는 무효 — 크기 축을 닫는다", [
             "Tᵢ 인코더 폭 24 / 40 / 80 / 160 / 260",
             "  = 34k / 49k / 114k / 358k / 879k 파라미터",
             "skill +0.230 / +0.236 / +0.235 / +0.236 / +0.230 (컷)",
             "160 대비 ±0.008 · 최대 폭 유의 우세 1/4 · V_rot 불변",
             "→ 남은 분산은 모델 크기가 아니라 분할 분산이다",
         ], accent=ORANGE, title_size=13, body_size=11.5)
    return s


# --- 27. Result 10: peak --------------------------------------------------
def s_res_peak():
    s = slide()
    header(s, "6. 결과 ⑩", "우위는 고변동(peak) 구간에 집중된다")
    add_image_fit(s, os.path.join(FIG, "fig_peak.png"),
                  Inches(0.5), Inches(1.45), Inches(7.4), Inches(5.0))
    box(s, Inches(8.0), Inches(1.55), Inches(4.8), Inches(4.85), fill=CARDBG, round_=True)
    text(s, Inches(8.25), Inches(1.70), Inches(4.35), Inches(0.5),
         [[("어디서 가치를 버는가", 15, NAVY, True, False, None)]])
    bullets(s, Inches(8.25), Inches(2.25), Inches(4.35), Inches(4.1), [
        ("peak = 입력 기반 고-국소활동 이웃", 0),
        ("타겟 행 자체는 제외 — 보수적 영역 프록시", 1),
        ("Tᵢ peak: 컷 +0.45~+0.61 · 포함 +0.62~+0.72", 0, TEAL, True),
        ("8/8 셀 PASS — 무조건부", 1, GREEN, True),
        ("Tᵢ 본류: +0.09~+0.20(컷 4/4) · +0.06~+0.19(포함 2/4)", 1),
        ("V_rot peak: +0.54~+0.79 (8/8 양수, PASS 각 2/4)", 0, BLUE, True),
        ("persistence 대비는 +0.75~+0.86, 8/8 PASS", 1),
        ("본류는 ≈0 (-0.07~+0.15, 0/8)", 1),
        ("보간은 매끄러운 본류에서 이미 최적", 0, NAVY, True),
        ("비대칭은 전역이 아니라 지역적이다", 1, NAVY, True),
    ], size=12, gap=7)
    return s


# --- 28. Result 11: transient case study ---------------------------------
def s_res_transient():
    s = slide()
    header(s, "6. 결과 ⑪", "급변 구간을 눈으로 — held-out TEST shot #31815")
    add_image_fit(s, os.path.join(FIG, "fig_transient_seq_31815.png"),
                  Inches(0.45), Inches(1.40), Inches(7.15), Inches(5.4))
    box(s, Inches(7.75), Inches(1.55), Inches(5.05), Inches(4.85), fill=CARDBG, round_=True)
    text(s, Inches(8.0), Inches(1.70), Inches(4.6), Inches(0.5),
         [[("한 shot에서 실제로 벌어지는 일", 15, NAVY, True, False, None)]])
    bullets(s, Inches(8.0), Inches(2.25), Inches(4.6), Inches(4.1), [
        ("빠른 진단이 급변을 먼저 본다", 0, NAVY, True),
        ("BES 급락(빨간 점선)이 CES crash와 정렬", 1),
        ("PCHIP는 스파이크마다 overshoot", 0, GRAY, True),
        ("과거+미래를 다 보는 오프라인 보간인데도", 1),
        ("모델은 세그먼트 과거 + 빠른 진단만 쓰는 causal", 0, GREEN, True),
        ("Tᵢ: RMSE 199.2 vs PCHIP 262.3 → skill +0.42", 0, ORANGE, True),
        ("n = 395 실측점 (genuine, 컷 후)", 1),
        ("V_rot: 17.3 vs 19.5 → skill +0.21 (n = 149)", 0, BLUE, True),
        ("V_rot 관측이 끊기면 이력만으로 완만히 감쇠", 1),
        ("우위는 gap·peak에 집중 (결과 ④⑩과 일관)", 0, GRAY, True),
    ], size=12, gap=6)
    return s


# --- 29. Result 12: deployment measured -----------------------------------
def s_deploy():
    s = slide()
    header(s, "6. 결과 ⑫", "배치 가능성 — 주장이 아니라 측정")
    card(s, Inches(0.55), Inches(1.5), Inches(6.0), Inches(2.65),
         "지연 — 상태 유지 1-step은 여유 있게 CPU에 든다", [
             "온라인은 은닉 상태를 격자 따라 이월 → 새 행마다 1-step",
             "seq_v2 CPU 중앙값 1.05 ms / p99 1.61 ms (예산의 16%, p95 1.35)",
             "GPU 1.21 / 2.31 ms — 이 크기에선 배치 1에서 이득 없음",
             "세그먼트 재실행: 100행 2.9 / 5.6 ms · 300행 6.4 / 8.9 ms",
             "  (35–47k 행/s · GPU 47–63k)",
             "윈도 대조군 배치 1은 더 느리고 꼬리가 무겁다 (W=2 3.8 / 18.9 ms)",
         ], accent=TEAL, title_size=14, body_size=11.5)
    card(s, Inches(6.8), Inches(1.5), Inches(6.0), Inches(2.65),
         "불확실성 — split conformal (재학습 없음, α = 0.10)", [
             "해당 run 자신의 val에서 보정 · 예측기는 전혀 건드리지 않음",
             "같은 절차를 PCHIP·persistence에도 적용한 공정 비교",
             "모델 구간이 32/32 셀에서 두 기준선을 Winkler 점수로 이김",
             "  Tᵢ 1,272 vs 1,554 vs 1,727(컷) · 2,290 / 2,851 / 3,120(포함)",
             "  V_rot 150 vs 164 vs 179",
             "Mondrian 보정은 Tᵢ 팔을 4–5% 조이고 판정은 불변",
         ], accent=BLUE, title_size=14, body_size=11.5)
    card(s, Inches(0.55), Inches(4.35), Inches(12.25), Inches(2.2),
         "실무 지침과 정직한 실패", [
             "→ 상태 유지형 나우캐스터를 제어 계산기의 CPU에서 돌려라. 10 ms 예산의 80% 이상이 획득과 제어에 남는다.",
             "정직한 실패 ①: coverage는 marginal이지 조건부가 아니다 — Tᵢ 0.87–0.92 (목표 0.90), V_rot 0.91–0.94, shot별로는 넓게 흩어진다.",
             "정직한 실패 ②: 유휴 재실행 두 번의 절대값이 최대 2× 차이 났다(다른 실행 seq_v2 스텝 0.51 / 0.99 ms) — 노트북 전원 상태. 순서는 불변.",
             "포함 모집단에선 모델 Tᵢ 구간이 PCHIP보다 넓은데도(반폭 224–255 vs 211–241 eV) 점수가 더 좋다 — 스파이크가 miss 페널티를 키우고 모델이 덜 놓친다.",
         ], accent=ORANGE, title_size=14, body_size=12)
    return s


# --- 30. Conclusion -------------------------------------------------------
def s_conclusion():
    s = slide()
    header(s, "7. 결론", "정직한 결론 (4가지)")
    items = [
        ("1", "Tᵢ: 미래를 쓰는 보간을 두 모집단 모두에서 유의하게 능가", BLUE,
         "skill_vs_pchip 컷 +0.17~+0.26 · 포함 +0.23~+0.32, 4개 독립 분할 전부 shot 군집 95% CI가 0 제외(4/4+4/4). 인과 GP는 8/8 셀에서 이기고, 최강 오프라인 평활기(GP)와는 동률 — 상한도 함께 보고한다."),
        ("2", "배치 주장이 이제 두 스트레스를 모두 견딘다", GREEN,
         "실제 결측 in-domain 시점에서 인과 방법 대비 8/8 생존, 캠페인 경계 너머 PCHIP·인과 GP 대비 4/4+4/4. 대체된 윈도 대조군은 오프라인 우위를 완전히 잃었고(2/4·0/4) 그 차이는 측정됐다(빠른 진단 5–11× 드리프트)."),
        ("3", "V_rot: 전역 동률 — skill은 천이 구간에만 있다", GRAY,
         "PCHIP 대비 1/4·2/4(잡음 수준)이지만 >15 ms 간극과 peak 층에서는 두 모집단 모두 이긴다. 빠른 채널을 0으로 해도 출력이 bit-identical — Tᵢ↔V_rot 정보 비대칭은 물리로 예측되고 절제로 확인됐다."),
        ("4", "상한은 추정기가 아니라 정보 — 남은 레버는 데이터다", ORANGE,
         "21,498 파라미터 b3k8이 컷에서 백본과 동급(+0.002)이고 폭 34k→879k는 평평(±0.01) → 크기 축은 닫혔다. 남은 셋: CES 피팅 품질 메타데이터 · 원본 kHz Mirnov · NBI 토크 채널."),
    ]
    yy = 1.55
    for num, t, col, body in items:
        box(s, Inches(0.6), Inches(yy), Inches(12.2), Inches(1.18), fill=CARDBG, round_=True)
        box(s, Inches(0.72), Inches(yy + 0.17), Inches(0.84), Inches(0.84), fill=col, round_=True)
        text(s, Inches(0.72), Inches(yy + 0.17), Inches(0.84), Inches(0.84),
             [[(num, 26, WHITE, True, False, None)]], align=PP_ALIGN.CENTER,
             anchor=MSO_ANCHOR.MIDDLE)
        text(s, Inches(1.75), Inches(yy + 0.11), Inches(10.9), Inches(0.42),
             [[(t, 15.5, NAVY, True, False, None)]])
        text(s, Inches(1.75), Inches(yy + 0.53), Inches(10.9), Inches(0.62),
             [[(body, 11.5, DARK, False, False, None)]], line_spacing=1.08)
        yy += 1.32
    return s


# --- 31. Follow-up: can Mirnov be salvaged for V_rot? --------------------
def s_mirnov():
    s = slide()
    header(s, "7. 추가 검증", "\"Mirnov를 더 잘 쓰면 V_rot이 되지 않나?\" — 정보는 전처리가 파괴했다")
    # left panel of fig_mirnov only (lag-1 autocorrelation); the right panel is a
    # W=4-era paired experiment and is cropped away.
    pic = s.shapes.add_picture(os.path.join(FIG, "fig_mirnov.png"),
                               Inches(0.75), Inches(1.45), Inches(4.74), Inches(3.6))
    pic.crop_right = 0.5   # frame aspect matches the (a) panel of the source figure
    box(s, Inches(5.95), Inches(1.45), Inches(6.85), Inches(3.6), fill=CARDBG, round_=True)
    box(s, Inches(5.95), Inches(1.45), Inches(0.10), Inches(3.6), fill=RED)
    text(s, Inches(6.2), Inches(1.58), Inches(6.4), Inches(0.5),
         [[("진단은 실측이다", 14.5, RED, True, False, None)]])
    bullets(s, Inches(6.2), Inches(2.08), Inches(6.4), Inches(2.9), [
        ("같은 10 ms 격자, 연속 블록 내 lag-1 자기상관", 0),
        ("BES +0.568 · ECEI +0.572 vs Mirnov -0.009", 1, RED, True),
        ("블록의 82%가 |r| < 0.1 → 이 격자 위에서 백색잡음", 1),
        ("kHz dB/dt를 안티앨리어싱 없이 100 Hz로 데시메이트한 서명", 0),
        ("연속 표본의 상대 위상이 무작위가 된다", 1),
        ("우리 진단 집합에서 회전의 유일한 대리(모드 회전 주파수)가", 0, NAVY, True),
        ("모델보다 상류에서 버려졌다", 1, NAVY, True),
    ], size=12, gap=7)
    card(s, Inches(0.55), Inches(5.20), Inches(3.90), Inches(1.62), "① 시도한 것", [
        "적분 · PCHIP 적분 · |MC| · 이동 RMS 파생 특징",
        "학습에서 개선 없음 — 이미 잃은 정보는",
        "하류에서 복구되지 않는다",
    ], accent=ORANGE, title_size=13.5, body_size=11)
    card(s, Inches(4.72), Inches(5.20), Inches(3.90), Inches(1.62), "② 해야 할 것", [
        "모델 변경이 아니라 전처리 변경:",
        "원본 kHz 스트림에서 윈도 RMS · 대역 파워 ·",
        "모드 수 · 모드 회전 주파수를 계산",
    ], accent=TEAL, title_size=13.5, body_size=11)
    card(s, Inches(8.89), Inches(5.20), Inches(3.90), Inches(1.62), "③ 어떻게", [
        "V_rot 분기로 라우팅 (사전등록된",
        "pilot-then-expand 규칙 아래)",
        "V_rot 최상위 실험 · 아카이브 데이터로 검정 가능",
    ], accent=NAVY, title_size=13.5, body_size=11)
    return s


# --- 32. Expected rebuttal: does Te proxy the NBI torque? ----------------
def s_te_nbi():
    s = slide()
    header(s, "7. 추가 검증",
           "\"Tₑ가 NBI 가열을 대리하니 V_rot도 담기지 않나?\" — 가설과 기각", accent=RED)
    text(s, Inches(0.55), Inches(1.36), Inches(12.3), Inches(0.60),
         [[("이론적으로 충분히 가능한 경로다. ", 14, RED, True, False, None),
           ("NBI가 들어가면 전자가 가열되므로 Tₑ가 NBI 주입량을 간접 반영하고, "
            "그렇다면 ECEI가 회전 정보를 간접적으로 운반할 수 있다 — 데이터로 직접 검정했다.",
            14, DARK, False, False, None)]], line_spacing=1.14)
    box(s, Inches(0.55), Inches(2.00), Inches(5.05), Inches(3.62), fill=CARDBG, round_=True)
    box(s, Inches(0.55), Inches(2.00), Inches(0.10), Inches(3.62), fill=ORANGE)
    text(s, Inches(0.80), Inches(2.13), Inches(4.6), Inches(0.4),
         [[("가설의 인과 사슬", 14.5, ORANGE, True, False, None)]])
    bullets(s, Inches(0.80), Inches(2.62), Inches(4.6), Inches(2.9), [
        ("NBI 주입 → 전자 가열 → Tₑ 상승", 0),
        ("∴ Tₑ는 NBI power의 간접 대리 변수", 0),
        ("power가 크면 토크도 크다면", 0),
        ("→ ECEI가 V_rot 정보를 운반해야 함", 1, NAVY, True),
        ("데이터에 NBI 컬럼이 없으므로", 0),
        ("Tₑ 대리변수로 경로 존재 여부를 검정", 1),
    ], size=12.5, gap=9)
    text(s, Inches(5.85), Inches(2.03), Inches(7.0), Inches(0.4),
         [[("shot 간 상관 (538 shot, ECEI 채널 평균 = Tₑ 대리)",
            13, NAVY, True, False, None)]])
    cwn = [Inches(3.30), Inches(1.85), Inches(1.80)]
    nrows = [
        [("Tₑ ~ CES_TI", DARK, True, None), ("+0.353", GREEN, True, MONO), ("3e-17", GREEN, True, MONO)],
        [("Tₑ ~ CES_VT", DARK, True, None), ("+0.024", RED, True, MONO), ("0.58", RED, True, MONO)],
        [("Tₑ ~ |CES_VT|", DARK, False, None), ("+0.001", RED, False, MONO), ("0.98", RED, False, MONO)],
        [("Tₑ 변동성 ~ |CES_VT|", DARK, False, None), ("-0.026", GRAY, False, MONO), ("0.55", GRAY, False, MONO)],
        [("BES 변동성 ~ |CES_VT|", DARK, False, None), ("-0.059", GRAY, False, MONO), ("0.17", GRAY, False, MONO)],
    ]
    table(s, Inches(5.85), Inches(2.48), cwn, ["관계", "Pearson r", "p"], nrows,
          row_h=Inches(0.38), head_h=Inches(0.36), size=12, head_size=11.5,
          emphasis_fill=None)
    text(s, Inches(5.85), Inches(4.86), Inches(6.95), Inches(0.80),
         [[("shot 내부에서도 같다: ", 12, NAVY, True, False, None),
           ("Tₑ~CES_TI는 블록 평균 r = +0.246으로 부호가 일관되게 양수인데, "
            "Tₑ~CES_VT는 +0.006으로 부호조차 무작위다 (|r|>0.3 블록: 42.7% vs 14.8%).",
            12, DARK, False, False, None)]], line_spacing=1.14)
    box(s, Inches(0.55), Inches(5.72), Inches(12.25), Inches(1.16), fill=CARDBG, round_=True)
    box(s, Inches(0.55), Inches(5.72), Inches(0.12), Inches(1.16), fill=RED)
    text(s, Inches(0.85), Inches(5.81), Inches(11.75), Inches(1.02),
         [[("결론 — 경로의 전반부는 참이고, 후반부에서 끊긴다.", 12, RED, True, False, None)],
          [("Tₑ가 가열 수준의 대리로 작동하는 건 사실이다 (Tᵢ와 r = +0.35). 그런데 바로 그 Tₑ가 "
            "V_rot과는 무관하다 (r = +0.02, p = 0.58). 끊기는 지점은 ",
            12, DARK, False, False, None),
           ("power ≠ torque", 12, RED, True, False, None),
           (" — 토크는 빔 에너지·접선 반경·주입 기하에 의존해 power와 분리된다. NBI 토크 확보는 모델링이 아니라 "
            "데이터 획득 과제이며, 문헌에 양성 대조(DIII-D 전방전 시뮬레이터)가 있다.", 12, DARK, False, False, None)]],
         line_spacing=1.12)
    return s


# --- 33. Limitations + future --------------------------------------------
def s_limits():
    s = slide()
    header(s, "7. 한계 & 향후 연구", "한계와 다음 단계")
    box(s, Inches(0.55), Inches(1.5), Inches(6.0), Inches(4.95), fill=CARDBG, round_=True)
    box(s, Inches(0.55), Inches(1.5), Inches(0.12), Inches(4.95), fill=RED)
    text(s, Inches(0.8), Inches(1.64), Inches(5.5), Inches(0.5),
         [[("한계 — 논문이 먼저 인정하는 것", 15.5, RED, True, False, None)]])
    bullets(s, Inches(0.8), Inches(2.18), Inches(5.55), Inches(4.2), [
        ("검정력: test 방전 96(Tᵢ) / 60–66(V_rot)이 모든 유의성의 구속조건", 0),
        ("포함 모집단에선 ≈1% 행이 SSE의 70–83%를 담는다", 1),
        ("MNAR 낙관: 재가중은 인과 비교엔 양쪽, 오프라인 비교엔 모집단 조건부", 0),
        ("도달 범위도 Tᵢ 54–68% · V_rot 4–6%뿐", 1),
        ("오프라인 주장의 상한은 GP 동률(1/8 유의)", 0),
        ("값 컷은 일방향 프록시 — V_rot 스파이크는 컷되지 않은 채 남는다", 0),
        ("캠페인은 한 시간 블록(초기화 4개), 컷 run 2/4가 상한 종료,", 0),
        ("shot별 표준화는 오프라인 형태 — 인과 EWMA 형태는 미측정", 1),
        ("지표 비대칭 · 단일 장치 · 두 순환 계열 · conformal은 marginal", 0),
        ("지연은 네트워크만, 전원 상태에 따라 2× 변동", 1),
        ("범위: 페데스탈 상단 프레이밍, 이벤트-위상 분석은 후속", 0),
    ], size=11.5, gap=7)
    box(s, Inches(6.8), Inches(1.5), Inches(6.0), Inches(4.95), fill=CARDBG, round_=True)
    box(s, Inches(6.8), Inches(1.5), Inches(0.12), Inches(4.95), fill=TEAL)
    text(s, Inches(7.05), Inches(1.64), Inches(5.5), Inches(0.5),
         [[("향후 — 남은 레버는 전부 데이터다", 15.5, TEAL, True, False, None)]])
    bullets(s, Inches(7.05), Inches(2.18), Inches(5.55), Inches(4.2), [
        ("음성 결과는 그것을 뒤집을 측정을 지목할 때만 결론이 된다", 0, NAVY, True),
        ("① CES 피팅 품질 메타데이터 (χ²·신호 수준)", 0, ORANGE, True),
        ("값 컷을 품질 컷으로 대체하면 두 모집단이 하나로 합쳐진다", 1),
        ("V_rot 스파이크도 같은 규칙으로 처리", 1),
        ("② 원본 kHz Mirnov 특징 — V_rot 최상위 레버", 0, ORANGE, True),
        ("윈도 RMS · 대역 파워 · 모드 수 · 모드 회전 주파수", 1),
        ("아카이브 데이터로 검정 가능 (pilot→expand 사전등록)", 1),
        ("③ NBI 토크 채널 확보 — 회전의 원인 변수", 0, ORANGE, True),
        ("모델링이 아니라 데이터 획득 과제 · 양성 대조 존재", 1),
        ("용량·긴 윈도는 이미 배제됐다 → 크기 축은 닫혔다", 0, GRAY, True),
    ], size=11.5, gap=7)
    return s


# --- 34. Takeaways / closing ---------------------------------------------
def s_closing():
    s = slide()
    box(s, 0, 0, EMU_W, EMU_H, fill=NAVY)
    box(s, Inches(0.9), Inches(0.95), Inches(2.2), Pt(4), fill=ORANGE)
    text(s, Inches(0.9), Inches(1.15), Inches(11.5), Inches(0.6),
         [[("한 장 요약 — Key Takeaways", 26, WHITE, True, False, None)]])
    points = [
        ("항상 있는 빠른 진단으로 자주 비는 CES를 채우는, 엄격히 인과적인 가상 센서", ORANGE),
        ("Tᵢ는 미래를 쓰는 PCHIP를 두 모집단 4/4+4/4로 능가 (+0.17~+0.32), 인과 GP는 8/8", GREEN),
        ("두 스트레스 생존: 결측 재가중 인과 대비 8/8 · 캠페인 분할 4/4+4/4 (윈도 대조군은 붕괴)", BLUE),
        ("V_rot는 전역 동률 — 빠른 진단에 회전 정보가 없다는 Tᵢ↔V_rot 비대칭 (발견)", TEAL),
        ("가치는 peak·간극에 집중 · CPU 1-step p99 1.61 ms · conformal 32/32 셀 우위", ORANGE),
        ("상한은 추정기가 아니라 정보: 21k = 백본(컷), 폭 26× 평평 → 다음은 데이터 레버 3종", BLUE),
    ]
    yy = 2.1
    for t, col in points:
        box(s, Inches(0.95), Inches(yy + 0.05), Inches(0.28), Inches(0.28), fill=col)
        text(s, Inches(1.45), Inches(yy - 0.04), Inches(11.0), Inches(0.55),
             [[(t, 15, WHITE, False, False, None)]], line_spacing=1.1)
        yy += 0.62
    box(s, Inches(0.9), Inches(6.15), Inches(11.5), Pt(2), fill=RGBColor(0x2A, 0x47, 0x6E))
    text(s, Inches(0.9), Inches(6.35), Inches(11.5), Inches(0.7),
         [[("감사합니다.  ", 20, WHITE, True, False, None),
           ("Q & A", 20, ORANGE, True, False, None)]])
    return s


# ======================= build ===========================================
def _fit_report():
    """Re-run preview_pptx's layout math at build time — FIT WARNING must be 0.

    Uses the *same* tokenizer, font metrics and overflow rule as
    ``preview_pptx.py`` (dpi 110), so a clean build implies a clean preview.
    """
    try:
        from PIL import Image, ImageDraw
        import preview_pptx as PV
    except Exception as exc:  # pragma: no cover - QC only
        return ["fit audit unavailable (%s)" % exc]
    scale, emu = 110, 914400
    W = int(prs.slide_width / emu * scale)
    H = int(prs.slide_height / emu * scale)
    warns = []
    for idx, sl in enumerate(prs.slides, start=1):
        draw = ImageDraw.Draw(Image.new("RGB", (W, H), (255, 255, 255)))
        for sh in sl.shapes:
            if sh.left is None:
                continue
            if sh.has_text_frame and sh.text_frame.text.strip():
                if PV.draw_text_frame(draw, sh, scale):
                    warns.append("slide %d: text overflows its box @ (%.2f, %.2f) -> %r"
                                 % (idx, sh.left / emu, sh.top / emu,
                                    sh.text_frame.text[:44]))
            x, y = sh.left / emu * scale, sh.top / emu * scale
            w, h = sh.width / emu * scale, sh.height / emu * scale
            if x < -1 or y < -1 or x + w > W + 1 or y + h > H + 1:
                warns.append("slide %d: shape outside slide @ (%.2f, %.2f)"
                             % (idx, sh.left / emu, sh.top / emu))
    return warns


def build():
    try:  # slide text is Korean; the console may be cp949
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    s_title()
    s_agenda()
    divider("1", "연구 배경 & 문제 정의", "CES는 왜 비는가 · 두 모집단이 필요한 이유")
    s_diagnostics()
    s_problem()
    s_missing_table()
    s_two_populations()
    s_idea()
    divider("2", "접근법", "미래를 보는 보간 + 배치 가능한 최강 기준선(인과 GP)")
    s_bar()
    s_validation()
    divider("3", "데이터 & 파이프라인", "held 전면 제거 · 두 프레이밍 · 누수 삼중 차단")
    s_data()
    s_contract()
    s_split()
    s_samples()
    s_stuck()
    divider("4", "모델", "전체격자 인과 시퀀스 백본 seq_v2 + W=2 윈도 대조군")
    s_arch()
    s_arch_detail()
    s_arch_window()
    s_training()
    divider("5", "평가 방법론", "사전등록 · shot 군집 bootstrap · TEST 동결")
    s_methodology()
    s_bootstrap()
    s_res_protocol()
    divider("6", "결과", "Tᵢ 4/4+4/4 · V_rot 동률 · 스트레스 2종 생존")
    s_res_ladder()
    s_res_forest()
    s_res_gate()
    s_res_gap()
    s_stress()
    s_res_campaign()
    s_res_asym()
    s_window_sweep()
    s_res_scaling()
    s_res_peak()
    s_res_transient()
    s_deploy()
    divider("7", "결론 · 한계 · 향후 연구", "무엇을 주장하고 무엇을 인정하는가")
    s_conclusion()
    s_mirnov()
    s_te_nbi()
    s_limits()
    s_closing()

    warns = _fit_report()
    out = os.path.join(HERE, "KSTAR_CES_발표자료.pptx")
    prs.save(out)
    print("SAVED:", out, "| slides:", len(prs.slides._sldIdLst))
    for w in warns:
        print("  FIT WARNING:", w)
    print("FIT WARNING count:", len(warns))


if __name__ == "__main__":
    build()
