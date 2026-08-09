# -*- coding: utf-8 -*-
"""Build the 1-hour KSTAR CES nowcasting thesis presentation (Korean).

Output: docs/presentation/KSTAR_CES_발표자료.pptx
Figures are read from docs/presentation/figures/ (run make_figures.py first).
"""
import os
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE, MSO_CONNECTOR
from pptx.oxml.ns import qn

HERE = os.path.dirname(__file__)
FIG = os.path.join(HERE, "figures")

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
         [[("학위논문 발표 · 약 60분", 16, RGBColor(0x9CC0E8 >> 16 & 255 and 0x9C, 0xC0, 0xE8), True, False, None)]])
    text(s, Inches(0.88), Inches(2.45), Inches(11.7), Inches(2.0),
         [[("KSTAR 다중 진단 기반 Multimodal 신경망을 활용한", 30, WHITE, True, False, None)],
          [("CES 결측 구간 예측 (Gap-filling / Nowcasting)", 34, WHITE, True, False, None)]],
         line_spacing=1.12)
    text(s, Inches(0.9), Inches(4.35), Inches(11.5), Inches(1.0),
         [[("빠른 진단(BES · ECEI · Mirnov)과 과거 CES 이력으로 ", 16, LGRAY, False, False, None),
           ("이온온도 Tᵢ · 토로이달 회전 V_rot", 16, ORANGE, True, False, None),
           ("를 복원하고,", 16, LGRAY, False, False, None)],
          [("미래까지 보는 오프라인 보간(interpolation)을 이기는가를 통계적으로 검증", 16, LGRAY, False, False, None)]],
         line_spacing=1.2)
    text(s, Inches(0.9), Inches(5.95), Inches(11.5), Inches(1.0),
         [[("이승상  (Seungsang Lee)", 17, WHITE, True, False, None)],
          [("서울대학교 · 원자핵공학  |  KSTAR Charge Exchange Spectroscopy", 13, MGRAY, False, False, None)]],
         line_spacing=1.25)
    return s


# --- 2. Agenda ------------------------------------------------------------
def s_agenda():
    s = slide()
    header(s, "Contents", "발표 목차")
    items = [
        ("1", "연구 배경 & 문제 정의", "CES는 왜 자주 비는가 — 가상 센서의 필요성", ORANGE),
        ("2", "접근법 — 의도적으로 어려운 평가 bar", "미래를 보는 오프라인 보간을 causal 모델로 이기기", BLUE),
        ("3", "데이터 & 전처리 파이프라인", "No-Fake-Data · per-target 마스킹 · 누수 방지 split", TEAL),
        ("4", "Multimodal 모델 아키텍처", "진단별 인코더 + 이력 인코더 + target별 head", NAVY),
        ("5", "평가 방법론 (통계적 엄밀성)", "3-way split · shot-clustered paired bootstrap", BLUE),
        ("6", "결과", "causal 압도 · Tᵢ 보간 유의 · Tᵢ↔V_rot 비대칭", ORANGE),
        ("7", "결론 · 한계 · 향후 연구", "정직한 결론과 통계적 검정력 한계", GRAY),
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
           ("빠른 진단(BES·ECEI·MC)은 항상 조밀하게 측정된다.", 15, NAVY, True, False, None)]])
    cards = [
        ("CES  (타겟)", ORANGE,
         ["Charge Exchange Spectroscopy",
          "Tᵢ (이온온도), V_rot (토로이달 회전)",
          "광자 수집 필요 → 느림 · 자주 결측",
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
         ["자기요동(MHD mode) 신호",
          "10 ms(100 Hz)로 샘플 → kHz 모드 aliasing",
          "회전 정보가 접혀(alias) 사라짐",
          "→ V_rot 직접 정보 거의 없음"]),
    ]
    x0 = 0.55
    for i, (t, col, lines) in enumerate(cards):
        card(s, Inches(x0 + i * 3.12), Inches(2.2), Inches(2.95), Inches(3.5),
             t, lines, accent=col, title_size=14.5, body_size=11.5)
    text(s, Inches(0.55), Inches(5.95), Inches(12.3), Inches(0.9),
         [[("핵심 비대칭(미리보기): ", 14, NAVY, True, False, None),
           ("빠른 진단은 물리적으로 Tᵢ 정보는 운반하지만 V_rot 정보는 거의 운반하지 않는다 "
            "(NBI 토크 미관측 + Mirnov aliasing). 이 가설이 결과에서 그대로 확인된다.",
            14, DARK, False, False, None)]], line_spacing=1.15)
    return s


# --- 5. CES missing problem ----------------------------------------------
def s_problem():
    s = slide()
    header(s, "1. 연구 배경", "문제: CES는 왜, 얼마나 비는가")
    bullets(s, Inches(0.55), Inches(1.5), Inches(6.3), Inches(4.6), [
        ("CES는 충분한 신호대잡음비(SNR)를 위해 광자를 오래 수집해야 함", 0),
        ("노출시간 · 신호품질 문제로 특정 시점 측정이 자주 누락됨", 1),
        ("같은 10 ms 격자에서 Tᵢ 8.2%, V_rot 23.9%가 완전 결측(NaN)", 0),
        ("V_rot는 여기에 held(직전값 복사) 41.1%가 더해져 실질 65.0% 무정보", 1, RED, True),
        ("두 타겟은 서로 독립적으로 결측 → target별 처리 필요", 1),
        ("빠른 진단(BES·ECEI·MC)은 같은 격자에서 100% 조밀", 0),
        ("\"항상 있는 빠른 진단\"으로 \"자주 비는 CES\"를 채운다", 1, ORANGE, True),
        ("기존 대안 UFCES의 한계", 0),
        ("제한된 파장 채널 + 측정값→Tᵢ/V_rot 역산에 강한 물리 가정 필요", 1),
    ])
    box(s, Inches(7.1), Inches(1.55), Inches(5.7), Inches(2.5), fill=CARDBG, round_=True)
    text(s, Inches(7.35), Inches(1.72), Inches(5.3), Inches(0.5),
         [[("Multimodal AI 도입의 강점", 16, NAVY, True, False, None)]])
    bullets(s, Inches(7.35), Inches(2.25), Inches(5.25), Inches(1.8), [
        ("결측 시점의 Tᵢ·V_rot를 데이터 기반으로 추정 (gap-filling)", 0, TEAL, True),
        ("강한 역산 가정 불필요 — 축대칭(axisymmetry) 수준만 가정", 0, TEAL, True),
        ("끊김 없는 연속 CES 가용성 → pedestal 물리 분석에 활용", 0, TEAL, True),
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
    text(s, Inches(0.8), Inches(5.99), Inches(11.75), Inches(0.9),
         [[("왜 중요한가: ", 12.5, NAVY, True, False, None),
           ("held 행은 baseline(persistence·보간)이 오차 ≈0으로 맞히는 ‘공짜 정답’이라 V_rot RMSE를 "
            "35~55% 낮춰 보이게 한다 → 평가는 genuine 관측만 사용, 학습은 유지.",
            12.5, DARK, False, False, None)],
          [("판정 기준: ", 12.5, NAVY, True, False, None),
           ("연속 시간블록 안에서 직전 관측값과 부동소수점까지 동일한 행. CES_TI는 641 shot 전체에서 "
            "단 1행 — V_rot 고유의 계측 특성이다.", 12.5, DARK, False, False, None)]],
         line_spacing=1.12)
    return s


# --- 6. Research question / core idea ------------------------------------
def s_idea():
    s = slide()
    header(s, "1. 연구 배경", "연구 질문 & 핵심 아이디어")
    box(s, Inches(0.7), Inches(1.55), Inches(11.9), Inches(1.4), fill=NAVY, round_=True)
    text(s, Inches(1.0), Inches(1.72), Inches(11.4), Inches(1.1),
         [[("연구 질문", 13, ORANGE, True, False, None)],
          [("CES가 결측된 10 ms 시점에서, 동시각 빠른 진단(BES·ECEI·MC) + 과거 CES 이력만으로 ",
            16.5, WHITE, False, False, None)],
          [("CES의 시간 보간(interpolation)이 복원할 수 없는 정보를 회복할 수 있는가?",
            16.5, WHITE, True, False, None)]], line_spacing=1.18)
    cards = [
        ("가상 센서 (Virtual Sensor)", BLUE,
         ["빠른 진단으로부터 CES를 데이터 기반 추정",
          "역산(inverse mapping) 가정 없이",
          "결측·고장 시점을 메움"]),
        ("Gap-filling / Nowcasting", TEAL,
         ["미래를 예보하는 forecasting 아님",
          "현재 시점의 빈 값을 채움 (nowcast)",
          "초해상(super-resolution)과도 구분"]),
        ("정직한 검증", ORANGE,
         ["진짜 결측은 참값이 없음 → 직접검증 불가",
          "관측된 CES를 가린 뒤 복원 정확도 측정",
          "persistence·mean·보간과 target별 비교"]),
    ]
    for i, (t, col, lines) in enumerate(cards):
        card(s, Inches(0.55 + i * 4.13), Inches(3.25), Inches(3.95), Inches(2.45),
             t, lines, accent=col, title_size=15, body_size=12.5)
    text(s, Inches(0.7), Inches(6.0), Inches(11.9), Inches(0.9),
         [[("결론의 성격: ", 13.5, NAVY, True, False, None),
           ("masking 검증에서 baseline을 이기면 결측 구간에서도 잘 복원할 것으로 ", 13.5, DARK, False, False, None),
           ("추정", 13.5, ORANGE, True, False, None),
           ("한다. 결측이 무작위라는 보장이 없어(MNAR) 결측 지점 정확도를 단정하지 않는 — 정직한 추정.",
            13.5, DARK, False, False, None)]], line_spacing=1.15)
    return s


# --- 7. Approach: the hard bar -------------------------------------------
def s_bar():
    s = slide()
    header(s, "2. 접근법", "의도적으로 어려운 평가 bar: 미래를 보는 보간")
    text(s, Inches(0.55), Inches(1.45), Inches(12.3), Inches(0.55),
         [[("모델을 ", 15, DARK, False, False, None),
           ("오프라인 CES-only 보간(linear · PCHIP · local AR)", 15, NAVY, True, False, None),
           ("과 비교한다 — 이 보간들은 타겟 주변의 과거+미래 CES를 모두 사용한다.",
            15, DARK, False, False, None)]])
    # two access boxes
    box(s, Inches(0.55), Inches(2.15), Inches(6.0), Inches(2.55), fill=CARDBG, round_=True)
    box(s, Inches(0.55), Inches(2.15), Inches(0.12), Inches(2.55), fill=ORANGE)
    text(s, Inches(0.8), Inches(2.32), Inches(5.6), Inches(0.5),
         [[("우리 모델 (causal)", 16, ORANGE, True, False, None)]])
    bullets(s, Inches(0.8), Inches(2.85), Inches(5.5), Inches(1.8), [
        ("타겟 시점의 빠른 진단 (BES·ECEI·MC)", 0),
        ("과거 CES 이력 (window = 4)", 0),
        ("미래 CES는 전혀 보지 않음", 0, RED, True),
    ], size=13.5, gap=9)
    box(s, Inches(6.8), Inches(2.15), Inches(6.0), Inches(2.55), fill=CARDBG, round_=True)
    box(s, Inches(6.8), Inches(2.15), Inches(0.12), Inches(2.55), fill=BLUE)
    text(s, Inches(7.05), Inches(2.32), Inches(5.6), Inches(0.5),
         [[("보간 baseline (오프라인)", 16, BLUE, True, False, None)]])
    bullets(s, Inches(7.05), Inches(2.85), Inches(5.5), Inches(1.8), [
        ("타겟 양쪽의 과거 + 미래 CES 이웃 사용", 0),
        ("PCHIP(단조 3차) = pre-registered headline 기준선", 0),
        ("0.5 s 이상 gap은 보간 거부 → persistence fallback", 0),
    ], size=13.5, gap=9)
    box(s, Inches(0.7), Inches(5.0), Inches(11.9), Inches(1.55), fill=NAVY, round_=True)
    text(s, Inches(1.0), Inches(5.10), Inches(11.4), Inches(1.42),
         [[("왜 이 정보 비대칭이 핵심인가", 13.5, ORANGE, True, False, None)],
          [("미래까지 보는 보간을 causal(과거만 보는) 모델이 이긴다면, 그것은 빠른 진단이 "
            "시간 보간으로는 얻을 수 없는 CES 정보를 운반한다는 강력한 증거다.",
            15.5, WHITE, False, False, None)],
          [("미래 보간을 이기는 모델은 자명히 모든 causal baseline(persistence·AR)도 이긴다.",
            13.5, LGRAY, False, True, None)]], line_spacing=1.10, space_after=3)
    return s


# --- 8. Validation strategy ----------------------------------------------
def s_validation():
    s = slide()
    header(s, "2. 접근법", "검증 전략과 정직한 caveat")
    cards = [
        ("Masking 복원 검증", BLUE,
         ["관측된 CES 값을 가린 뒤 모델이 복원",
          "per-target RMSE/MAE",
          "persistence 대비 skill, mean 대비 R²",
          "ces_history에서 타겟 시점은 완전 마스킹(누수 방지)"]),
        ("Observed-only 측정", TEAL,
         ["관측된 CES 지점에서만 skill 측정",
          "진짜 결측은 참값이 없어 직접 검증 불가",
          "관측 지점이 결측 지점보다 쉬울 수 있음"]),
        ("MNAR — 낙관적 상한", ORANGE,
         ["CES 결측은 무작위 아님 (MNAR)",
          "저 SNR · ELM · 천이에서 drop-out",
          "→ observed-only skill = 낙관적 bound",
          "결측 지점 정확도를 단정하지 않음"]),
    ]
    for i, (t, col, lines) in enumerate(cards):
        card(s, Inches(0.55 + i * 4.13), Inches(1.6), Inches(3.95), Inches(3.0),
             t, lines, accent=col, title_size=15, body_size=12.5)
    box(s, Inches(0.7), Inches(4.95), Inches(11.9), Inches(1.6), fill=CARDBG, round_=True)
    text(s, Inches(0.95), Inches(5.12), Inches(11.4), Inches(1.4),
         [[("Murphy(1988) skill 정의", 14, NAVY, True, False, None)],
          [("skill_vs_pchip = 1 − MSE_model / MSE_pchip", 17, BLUE, True, False, MONO)],
          [("> 0 이면 모델이 보간보다 우수 · = 0 이면 동률. 모든 오차는 물리 단위로 역정규화하여 target별 계산.",
            13, DARK, False, False, None)]], line_spacing=1.2)
    return s


# --- 9. Data collection ---------------------------------------------------
def s_data():
    s = slide()
    header(s, "3. 데이터 & 파이프라인", "데이터 구성 기준")
    bullets(s, Inches(0.55), Inches(1.55), Inches(6.4), Inches(4.8), [
        ("플라즈마 상태 타겟: H-mode ELM suppression (RMP 인가) — 수집 계획 기준", 0),
        ("실제 파일: shot당 고정 30 s 구간 · 10 ms 격자 · 파일당 연속 블록 1–8개", 1),
        ("샷 번호: 실제 #30801 ~ #32751 (계획: #24000 ~ #33000 우선)", 0),
        ("하드웨어 이력 고려 (’17 MicroTCA, ’20 UFCES, ’23 W-divertor)", 1),
        ("총 641개 shot CSV 파일 (저장소에는 미포함, gitignore)", 0, NAVY, True),
        ("진단 채널: BES 9 · ECEI 4 · Mirnov 2 · time 4 · ces_history 4", 1),
        ("타겟: CES_TI, CES_VT (정규화된 2-벡터)", 1),
    ])
    box(s, Inches(7.2), Inches(1.55), Inches(5.6), Inches(4.9), fill=NAVY, round_=True)
    text(s, Inches(7.45), Inches(1.72), Inches(5.2), Inches(0.5),
         [[("샘플 1개의 구성 (window = 4)", 15, ORANGE, True, False, None)]])
    rows = [
        ("bes", "(4, 9)", "밀도요동 공간구조"),
        ("ecei", "(4, 4)", "전자온도 2D"),
        ("mc", "(4, 2)", "자기요동"),
        ("time_features", "(4, 4)", "불규칙 시간 인코딩"),
        ("ces_history", "(4, 4)", "이전 Tᵢ·V_rot + 관측 flag"),
        ("target", "(2,)", "정규화 [Tᵢ, V_rot]"),
        ("target_mask", "(2,)", "관측된 타겟만 손실 반영"),
    ]
    yy = 2.3
    for name, shp, desc in rows:
        text(s, Inches(7.45), Inches(yy), Inches(2.3), Inches(0.4),
             [[(name, 13, WHITE, True, False, MONO)]])
        text(s, Inches(9.5), Inches(yy), Inches(1.2), Inches(0.4),
             [[(shp, 12.5, ORANGE, True, False, MONO)]])
        text(s, Inches(10.55), Inches(yy), Inches(2.1), Inches(0.4),
             [[(desc, 11, LGRAY, False, False, None)]])
        yy += 0.56
    return s


# --- 10. No fake data + contract -----------------------------------------
def s_contract():
    s = slide()
    header(s, "3. 데이터 & 파이프라인", "No-Fake-Data 원칙과 데이터 계약(contract)")
    cards1 = [
        ("① No Fake Data", ORANGE,
         ["선형보간 등으로 '가짜 데이터'를 만들지 않음",
          "진단 입력이 온전하고, 타겟 중 적어도 하나가",
          "관측된 시점의 샘플만 사용",
          "한쪽 타겟만 관측된 행도 버리지 않음"]),
        ("② per-target masked MSE", BLUE,
         ["각 샘플의 target_mask로 관측된 타겟만 손실 반영",
          "결측 타겟은 참값 없이 마스킹되어 제외",
          "기존 코드는 둘 다 관측해야만 사용 →",
          "라벨 행의 ≈28%를 조용히 버리던 문제 수정"]),
    ]
    for i, (t, col, lines) in enumerate(cards1):
        card(s, Inches(0.55 + i * 6.2), Inches(1.55), Inches(6.0), Inches(2.4),
             t, lines, accent=col, title_size=15, body_size=12.5)
    cards2 = [
        ("③ 불규칙 시계열 대응", TEAL,
         ["결측 제거로 시간 간격이 불규칙해짐",
          "연속성 의존 LSTM 대신 1D CNN 지역패턴 추출",
          "time_features 4채널: lookback·delta·log1p×2"]),
        ("④ Late-Fusion Multimodal", NAVY,
         ["섣불리 평균내어 공간특성 잃지 않음",
          "진단별 전용 인코더 → 마지막에 융합(concat)",
          "→ 두 정규화 CES 타겟 예측"]),
    ]
    for i, (t, col, lines) in enumerate(cards2):
        card(s, Inches(0.55 + i * 6.2), Inches(4.15), Inches(6.0), Inches(2.3),
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
          "model.py 내부에서는 절대 역정규화하지 않음"]),
    ]
    for i, (t, col, lines) in enumerate(cards):
        card(s, Inches(0.55 + i * 6.2), Inches(1.55), Inches(6.0), Inches(2.7),
             t, lines, accent=col, title_size=15, body_size=12.5)
    box(s, Inches(0.55), Inches(4.5), Inches(12.25), Inches(2.0), fill=CARDBG, round_=True)
    text(s, Inches(0.8), Inches(4.65), Inches(11.8), Inches(0.5),
         [[("ces_history (batch, 4, 4) — 누수 차단의 핵심", 15, NAVY, True, False, None)]])
    bullets(s, Inches(0.8), Inches(5.15), Inches(11.8), Inches(1.3), [
        ("4채널: 이전 정규화 Tᵢ · 이전 정규화 V_rot · Tᵢ 관측flag · V_rot 관측flag", 0),
        ("Tᵢ·V_rot는 독립적으로 결측(≈8% / ≈24%) → 관측을 target별로 추적", 1),
        ("타겟 시점(target timestep)은 값·flag 모두 0으로 완전 마스킹 → 자기 정답 누수 차단", 1, RED, True),
    ], size=13, gap=7)
    return s


# --- 11b. Sample construction + augmentation ------------------------------
def s_samples():
    s = slide()
    header(s, "3. 데이터 & 파이프라인", "학습 샘플 구성: 연속 블록 → 윈도(W=4) → temporal subset 증강")
    cards = [
        ("① 연속 블록 분할", BLUE,
         ["time delta ≥ 0.5 s = 세그먼트 경계",
          "641/641 shot에 존재 (파일당 ~2개, 간극 중앙값 6.3 s)",
          "모델 윈도·보간 모두 경계를 넘지 않음"]),
        ("② 샘플 정의 (window W=4)", TEAL,
         ["연속 4행 윈도, 마지막 행 = 예측 타겟 시점",
          "입력(BES·ECEI·MC) 완전 + 타겟 ≥1개 관측 시 채택",
          "타겟 시점 이력은 값·flag 모두 0 (누수 차단)"]),
        ("③ Temporal subset 증강", ORANGE,
         ["블록 내 이전 행들의 부분집합(combinations) 열거",
          "같은 타겟을 이력 2·3개짜리 변형으로도 학습",
          "다양한 이력 밀도·간격 노출 → 결측 패턴에 강건"]),
        ("④ 샘플 캡 — 조합 폭발 제어", NAVY,
         ["seeded 랜덤 캡: train 200,000 / val 40,000",
          "고정 split CSV로 디스크에 pin → 완전 재현",
          "데이터 불일치 시 로드 거부(예외) — 조용한 drift 차단"]),
    ]
    for i, (t, col, lines) in enumerate(cards):
        r, c = divmod(i, 2)
        card(s, Inches(0.55 + c * 6.2), Inches(1.55 + r * 2.5), Inches(6.0), Inches(2.3),
             t, lines, accent=col, title_size=14.5, body_size=12.5)
    text(s, Inches(0.55), Inches(6.65), Inches(12.3), Inches(0.5),
         [[("한 샘플 = ", 12.5, NAVY, True, False, None),
           ("진단 3종(BES 9ch · ECEI 4ch · MC 2ch) + time_features 4ch + ces_history 4ch "
            "→ 정규화 [Tᵢ, V_rot] 예측. 캡 외에는 어떤 라벨 행도 조용히 버리지 않는다.",
            12.5, GRAY, False, False, None)]])
    return s


# --- 12. Data quality: stuck values --------------------------------------
def s_stuck():
    s = slide()
    header(s, "3. 데이터 & 파이프라인", "데이터 품질 보정: held/stuck CES_VT", accent=RED)
    text(s, Inches(0.55), Inches(1.45), Inches(12.3), Inches(0.9),
         [[("후반 audit 발견: ", 15, RED, True, False, None),
           ("관측된 CES_VT 값의 약 54%가 held/forward-fill", 15, RED, True, False, None),
           ("(직전 관측값과 bit-identical, 같은 시간블록 내 최대 1214행 연속). "
            "V_rot의 native cadence가 행 cadence보다 느려 값이 carry-forward된 것 — 진짜 측정이 아님.",
            14.5, DARK, False, False, None)]], line_spacing=1.16)
    bullets(s, Inches(0.55), Inches(2.55), Inches(6.3), Inches(4.0), [
        ("CES_TI는 사실상 영향 없음 (held 0.0%)", 0, GREEN, True),
        ("499 / 641 shot 파일이 영향받음", 0),
        ("평가: held 전 구간 채점 제외 (genuine-only가 headline)", 0),
        ("학습도 오염 — 초기 '무해' 판정(단일 seed)을 4-seed paired가 뒤집음", 0, RED, True),
        ("held 제거 학습이 V_rot 4/4 개선 (평균 +0.039, 3/4 유의)", 1, RED, True),
        ("→ 현행 규약: 학습·평가 모두 제거 (CES_DROP_STUCK_TARGETS=1)", 1, NAVY, True),
        ("CES_TI는 genuine-only 평가에서도 4 seed 모두 PASS (강건)", 0, GREEN, True),
    ], size=13.5, gap=8)
    box(s, Inches(7.1), Inches(2.55), Inches(5.7), Inches(3.9), fill=CARDBG, round_=True)
    text(s, Inches(7.35), Inches(2.7), Inches(5.3), Inches(0.5),
         [[("V_rot 물리 RMSE는 held 값에 의해 deflated", 14, NAVY, True, False, None)]])
    # mini table
    hdr = ["seed", "보고(stuck포함)", "genuine RMSE"]
    rows = [["42", "22.5", "35.0"], ["1", "24.7", "34.8"],
            ["7", "30.0", "43.5"], ["123", "32.9", "46.5"]]
    yy = 3.25
    text(s, Inches(7.35), Inches(yy), Inches(1.3), Inches(0.4), [[(hdr[0], 12, GRAY, True, False, None)]])
    text(s, Inches(8.5), Inches(yy), Inches(2.2), Inches(0.4), [[(hdr[1], 12, GRAY, True, False, None)]])
    text(s, Inches(10.8), Inches(yy), Inches(1.9), Inches(0.4), [[(hdr[2], 12, GRAY, True, False, None)]])
    yy += 0.45
    for r in rows:
        text(s, Inches(7.35), Inches(yy), Inches(1.3), Inches(0.4), [[(r[0], 13, DARK, False, False, MONO)]])
        text(s, Inches(8.5), Inches(yy), Inches(2.2), Inches(0.4), [[(r[1], 13, MGRAY, False, False, MONO)]])
        text(s, Inches(10.8), Inches(yy), Inches(1.9), Inches(0.4), [[(r[2], 13, RED, True, False, MONO)]])
        yy += 0.45
    text(s, Inches(7.35), Inches(yy + 0.05), Inches(5.2), Inches(0.8),
         [[("held 타겟은 baseline 오차가 ≈0 → 보고 RMSE를 35~55% 끌어내림. "
            "결론(Tᵢ↔V_rot 비대칭)은 그대로 유지.", 11.5, GRAY, False, True, None)]], line_spacing=1.1)
    return s


# --- 13. Architecture diagram --------------------------------------------
def s_arch():
    s = slide()
    header(s, "4. 모델 아키텍처", "Multimodal Late-Fusion + target별 routing")
    add_image_fit(s, os.path.join(FIG, "fig_architecture.png"),
                  Inches(0.45), Inches(1.38), Inches(12.45), Inches(5.15))
    text(s, Inches(0.55), Inches(6.62), Inches(12.3), Inches(0.5),
         [[("물리 기반 routing: ", 12.5, NAVY, True, False, None),
           ("Tᵢ = 빠른진단+이력+시간 / V_rot = 이력+시간만. 총 파라미터 201,258개(≈0.20 M) — "
            "capacity가 아니라 일반화가 관건. 출력은 정규화 단위, 역정규화는 평가에서만.",
            12.5, DARK, False, False, None)]])
    return s


# --- 14. Architecture detail ---------------------------------------------
def s_arch_detail():
    s = slide()
    header(s, "4. 모델 아키텍처", "핵심 설계 결정과 근거")
    cards = [
        ("진단별 time-aware CNN", BLUE,
         ["각 진단 + 시간 + 이력을 함께 1D Conv",
          "불규칙 시계열에 강건 (LSTM 회피)",
          "공간 채널 구조 보존 후 융합"]),
        ("History Encoder — 양방향 GRU", ORANGE,
         ["1층 bidirectional GRU (hidden 64)",
          "마스킹된 타겟 시점이 window 안에 있어",
          "양쪽 관측 이웃을 forward+backward로 취합",
          "→ '학습된 보간'과 유사한 동작"]),
        ("관측-마스크 attention pooling", TEAL,
         ["타겟별 독립 multi-head(4) additive attention",
          "관측 flag=1인 행에만 softmax 허용",
          "→ '관측만 본다'는 보간의 귀납 편향 주입",
          "관측 0개 행은 전체 window로 fallback"]),
        ("target별 head + LayerNorm proj", NAVY,
         ["이력 요약을 타겟별 LN→Linear→GELU projection",
          "Tᵢ head 384→160→64→1 / V_rot head 96→96→48→1",
          "capacity가 아닌 일반화 lever (총 0.20 M)"]),
    ]
    for i, (t, col, lines) in enumerate(cards):
        r, c = divmod(i, 2)
        card(s, Inches(0.55 + c * 6.2), Inches(1.55 + r * 2.5), Inches(6.0), Inches(2.3),
             t, lines, accent=col, title_size=14.5, body_size=12)
    text(s, Inches(0.55), Inches(6.65), Inches(12.3), Inches(0.5),
         [[("설계 교훈 (탐색에서 확인): ", 12, NAVY, True, False, None),
           ("Pre-LayerNorm(안정성) + attention pooling이 가장 신뢰성 있는 개선. 무리한 capacity 확대는 overfit으로 오히려 악화.",
            12, GRAY, False, False, None)]])
    return s


# --- 14b. Training configuration ------------------------------------------
def s_training():
    s = slide()
    header(s, "4. 모델 아키텍처", "어떻게 학습시켰나 — 손실 함수와 최적화 설정")
    card(s, Inches(0.55), Inches(1.55), Inches(6.0), Inches(3.4),
         "손실 함수 — per-target masked MSE + 물리 페널티",
         ["L = Σ mask·(ŷ−y)² / Σ mask + 0.1·ReLU(z₀ − ŷ_Tᵢ)",
          "target_mask로 관측된 타겟만 손실에 기여",
          "Tᵢ·V_rot 독립 결측(≈8% / ≈24%) — 한쪽만 있어도 학습",
          "ReLU 항 = '이온온도 < 0 keV 불가' soft 물리 제약",
          "  (z₀ = 물리 0을 정규화 단위로 변환한 값)",
          "출력은 정규화 단위 — 역정규화는 평가에서만"],
         accent=BLUE, title_size=14.5, body_size=12.5)
    card(s, Inches(6.75), Inches(1.55), Inches(6.0), Inches(3.4),
         "최적화 설정 (train.py)",
         ["AdamW — lr 1e-3, weight decay 1e-4",
          "batch 512 · 10 epochs · 단일 GPU (선택적 AMP)",
          "ReduceLROnPlateau(patience 2, ×0.5) — val masked MSE",
          "gradient clipping max-norm 5.0",
          "split seed와 init seed 분리 — 초기화 안정성 실험",
          "loss가 비유한이면 즉시 실패(fail-loud) — 조용한 NaN 금지"],
         accent=TEAL, title_size=14.5, body_size=12.5)
    box(s, Inches(0.55), Inches(5.15), Inches(12.25), Inches(1.6), fill=CARDBG, round_=True)
    text(s, Inches(0.8), Inches(5.3), Inches(11.8), Inches(0.5),
         [[("모델 선택 절차 — keep/discard 통제 루프 (~40회 반복)", 14, NAVY, True, False, None)]])
    bullets(s, Inches(0.8), Inches(5.78), Inches(11.8), Inches(0.9), [
        ("반복마다 구조 변경은 딱 하나 → 처음부터 재학습 → 증강 없는 검증셋에서 보간 대비 skill로 채점", 0),
        ("최고 기록을 넘으면 채택, 못 넘으면 직전 최고 모델로 복원 — 후퇴 위에 쌓지 않음 · TEST는 전 과정 봉인", 1, NAVY, True),
    ], size=12.5, gap=6)
    return s


# --- 15. Evaluation methodology: split + prereg --------------------------
def s_methodology():
    s = slide()
    header(s, "5. 평가 방법론", "선택 편향 없는 3-way split + 사전등록")
    cards = [
        ("3-way split (train/val/test)", BLUE,
         ["TEST는 아키텍처 탐색 시작 전 예약, 선택 과정에서 절대 안 봄",
          "모델 선택은 val에서만 → 헤드라인은 winner's-curse 없음",
          "TEST(genuine, seed 42): 33,693 샘플 / 96 shot",
          "관측 n: Tᵢ 32,787 (96 shot) · V_rot 10,729 (61 shot)",
          "held 포함 시 V_rot 27,437 — 민감도 확인용으로만 보고"]),
        ("사전등록 (PR1–PR4)", ORANGE,
         ["PR1 best-interpolation 규칙: headline = PCHIP",
          "PR2 평가 모집단: future-neighbor 없으면 persistence",
          "PR3 TEST floor: ≥15 shot & ≥3,000 Tᵢ 샘플 (충족)",
          "PR4 부트스트랩 구성: 95% CI가 0 제외 = PASS"]),
    ]
    for i, (t, col, lines) in enumerate(cards):
        card(s, Inches(0.55 + i * 6.2), Inches(1.55), Inches(6.0), Inches(2.95),
             t, lines, accent=col, title_size=15, body_size=12)
    box(s, Inches(0.55), Inches(4.70), Inches(6.55), Inches(2.05), fill=CARDBG, round_=True)
    text(s, Inches(0.8), Inches(4.84), Inches(6.05), Inches(1.85),
         [[("공정성 보장", 14, NAVY, True, False, None)],
          [("모든 arm(모델+모든 baseline)을 동일한 (file, row) 샘플 집합 · 동일 per-target "
            "keep mask로 채점 — 어떤 arm도 thinning 되지 않음.", 12.5, DARK, False, False, None)],
          [("보간은 타겟 자신의 값을 제외하고 이웃만 읽음(누수 없음). 0.5 s+ 세그먼트 경계(641/641 shot, "
            "간극 중앙값 6.3 s)는 넘지 않고 persistence 값으로 예측 — 커버리지 축소 없음.",
            12, GRAY, False, False, None)]], line_spacing=1.16)
    # baseline-choice robustness: swap the headline baseline for the stronger one
    box(s, Inches(7.30), Inches(4.70), Inches(5.50), Inches(2.25), fill=CARDBG, round_=True)
    text(s, Inches(7.52), Inches(4.80), Inches(5.1), Inches(0.4),
         [[("‘왜 PCHIP를 골랐나?’ — baseline을 바꿔도 결론 불변", 12.5, NAVY, True, False, None)]])
    mcols = [7.52, 8.45, 10.60]
    mhead = ["seed", "vs PCHIP", "vs linear (더 강함)"]
    mrows = [
        [("42", GRAY, False), ("+0.179 PASS", DARK, False), ("+0.148 n.s.", GRAY, True)],
        [("1", GRAY, False), ("+0.197 PASS", DARK, False), ("+0.167 PASS", GREEN, True)],
        [("7", GRAY, False), ("+0.280 PASS", DARK, False), ("+0.259 PASS", GREEN, True)],
        [("123", GRAY, False), ("+0.263 PASS", DARK, False), ("+0.234 PASS", GREEN, True)],
    ]
    yy = 5.16
    for hcell, xh in zip(mhead, mcols):
        text(s, Inches(xh), Inches(yy), Inches(2.0), Inches(0.35),
             [[(hcell, 10.5, GRAY, True, False, None)]])
    yy += 0.34
    for r in mrows:
        for (val, col, bold), xh in zip(r, mcols):
            text(s, Inches(xh), Inches(yy), Inches(2.0), Inches(0.35),
                 [[(val, 10.5, col, bold, False, MONO)]])
        yy += 0.31
    text(s, Inches(7.52), Inches(yy + 0.03), Inches(5.1), Inches(0.30),
         [[("genuine 기준 3/4 PASS(시드 42 n.s.도 명시) · held 포함 4/4.",
            10, GRAY, False, True, None)]], line_spacing=1.0)
    return s


# --- 16. Bootstrap --------------------------------------------------------
def s_bootstrap():
    s = slide()
    header(s, "5. 평가 방법론", "Shot-clustered paired bootstrap — 왜 shot이 단위인가")
    bullets(s, Inches(0.55), Inches(1.55), Inches(7.0), Inches(3.6), [
        ("한 방전(shot) 내 인접 CES 행은 강하게 상관됨", 0),
        ("개별 샘플을 독립으로 보면 불확실성을 크게 과소평가", 1, RED, True),
        ("PR4 검정: per-sample paired error (SE_model − SE_pchip)를", 0),
        ("shot 단위로 묶고, shot 전체를 복원추출(B=10,000, seed 12345)", 1),
        ("skill 95% CI가 모델에 유리한 방향으로 0을 제외 = PASS", 0, GREEN, True),
        ("→ CI가 within-shot 가짜 복제가 아닌 진짜 shot-to-shot 일반화 반영", 0),
        ("split-seed 변동은 2차 안정성 점검일 뿐, headline CI에 풀링하지 않음", 1),
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
          [("Tᵢ ≈ 96 shot · V_rot ≈ 91 shot", 13, WHITE, True, False, None)],
          [("이 적은 shot 수가 모든 유의성", 12, LGRAY, False, False, None)],
          [("판정의 구속 조건 (검정력 한계)", 12, LGRAY, False, False, None)]],
         line_spacing=1.18, space_after=3)
    return s


# --- 17. Result 1: RMSE ladder -------------------------------------------
def s_res_ladder():
    s = slide()
    header(s, "6. 결과 ①", "causal baseline을 큰 폭으로 압도 (강건한 결과)")
    add_image_fit(s, os.path.join(FIG, "fig_rmse_ladder.png"),
                  Inches(0.55), Inches(1.45), Inches(12.25), Inches(4.4))
    box(s, Inches(0.7), Inches(5.95), Inches(11.9), Inches(0.95), fill=CARDBG, round_=True)
    text(s, Inches(0.95), Inches(6.06), Inches(11.5), Inches(0.8),
         [[("두 타겟 모두 모델 RMSE가 사다리에서 최저. ", 13.5, NAVY, True, False, None),
           ("persistence·AR(causal) 대비 큰 마진으로 우수 — 온라인/실시간 환경에서 모델이 명확한 승자. "
            "보간(PCHIP/Linear)보다도 point estimate는 앞섬.", 13.5, DARK, False, False, None)]],
         line_spacing=1.16)
    return s


# --- 18. Result 2: headline forest ---------------------------------------
def s_res_forest():
    s = slide()
    header(s, "6. 결과 ②", "보간 대비 headline — 최종 모델, 4 seed")
    add_image_fit(s, os.path.join(FIG, "fig_forest.png"),
                  Inches(0.55), Inches(1.4), Inches(12.25), Inches(3.9))
    cards = [
        ("CES_TI — PASS (강건)", GREEN,
         ["4개 독립 split 모두 CI > 0 (1/7/123은 선택 밖 복제)",
          "genuine +0.18 ~ +0.28 · held 포함 평가에서도 4/4",
          "사후 GP 팔(최강 오프라인, PCHIP에 4/4 승)과는 동률",
          "— 사전등록 headline 유지 · 인과(배치) 주장 무영향"]),
        ("CES_VT — 동률 보고", GRAY,
         ["PASS는 seed 1 하나뿐 (1/4 = 잡음이 낼 수 있는 수준)",
          "point estimate는 4 seed 모두 +지만 승리 주장 안 함",
          "→ Tᵢ↔V_rot 비대칭 (다음 슬라이드)"]),
    ]
    for i, (t, col, lines) in enumerate(cards):
        card(s, Inches(0.55 + i * 6.2), Inches(5.28), Inches(6.0), Inches(1.68),
             t, lines, accent=col, title_size=13.5, body_size=10.5)
    return s


# --- 19. Result 3: progression -------------------------------------------
def s_res_prog():
    s = slide()
    header(s, "6. 결과 ③", "정직한 진전: n.s. → 통계적으로 유의")
    add_image_fit(s, os.path.join(FIG, "fig_progression.png"),
                  Inches(0.7), Inches(1.5), Inches(7.2), Inches(5.0))
    box(s, Inches(8.1), Inches(1.65), Inches(4.7), Inches(4.8), fill=CARDBG, round_=True)
    text(s, Inches(8.35), Inches(1.82), Inches(4.3), Inches(0.5),
         [[("무엇이 달라졌나", 15, NAVY, True, False, None)]])
    bullets(s, Inches(8.35), Inches(2.4), Inches(4.3), Inches(4.0), [
        ("기존 baseline(iter2)은 단일 split에서 +0.088, n.s.", 0, GRAY, True),
        ("아키텍처 선택 게이트를 val loss → clean skill_vs_pchip으로 교체", 0),
        ("→ GRU 이력인코더 + target별 multi-head attention head", 1),
        ("최종(iter5): +0.19~+0.28, 4 seed 모두 PASS", 0, GREEN, True),
        ("왜 clean skill인가: 증강 val loss는 쉬운 구간이 복제돼 평활화를 보상", 0, ORANGE, True),
        ("→ 평활한 예측은 보간이 이미 잘하는 영역. 보고 지표로 직접 채점해야", 1),
        ("   보간이 약한 곳(peak·gap)에서 이기는 모델이 선택됨", 1),
    ], size=12.5, gap=9)
    return s


# --- 20. Result 4: gap-stratified ----------------------------------------
def s_res_gap():
    """Gap-stratified, 4 splits POOLED, vs both bars — THESIS_RESULTS.md §8g
    (data/.largegap_analysis.json; genuine eval, physical-shot-clustered CI)."""
    s = slide()
    header(s, "6. 결과 ④", "Gap별 분석 — 4분할 통합: 비인접 영역에서도 이긴다")
    text(s, Inches(0.55), Inches(1.42), Inches(12.3), Inches(0.55),
         [[("seed별로는 넓은 Δt 구간 표본이 수십 개뿐 → 4개 test 분할을 합치고 물리적 shot 단위로 "
            "군집화해 처음으로 CI를 붙였다. PCHIP은 미래 앵커를 읽고 persistence는 읽지 않는다 (Tᵢ, genuine).",
            13, DARK, False, False, None)]], line_spacing=1.12)
    headers = ["Δt bin", "n", "vs PCHIP (미래 사용)", "vs persistence (인과)"]
    rows = [
        ["≤ 15 ms", "134,629", "+0.262 PASS", "+0.407 PASS", GREEN],
        ["(15, 25] ms", "2,987", "+0.183 PASS", "+0.384 PASS", GREEN],
        ["(25, 35] ms", "784", "+0.402 PASS", "+0.564 PASS", GREEN],
        ["(55, 105] ms", "163", "+0.282 PASS", "+0.287 PASS", GREEN],
        ["> 105 ms", "167", "−0.542 (PCHIP 승)", "+0.266 n.s.", RED],
        ["전체 > 15 ms", "4,496", "+0.191 PASS", "+0.388 PASS", GREEN],
        ["전체 > 45 ms", "435", "−0.057 n.s.", "+0.271 PASS", ORANGE],
    ]
    x0 = [0.7, 2.9, 4.9, 8.7]
    w0 = [2.0, 1.8, 3.6, 3.6]
    yy = 2.05
    box(s, Inches(0.55), Inches(yy - 0.05), Inches(12.25), Inches(0.46), fill=NAVY)
    for h, x, w in zip(headers, x0, w0):
        text(s, Inches(x), Inches(yy), Inches(w), Inches(0.4),
             [[(h, 12.5, WHITE, True, False, None)]])
    yy += 0.5
    for i, r in enumerate(rows):
        col = r[-1]
        bg = WHITE if (i % 2 == 0) else CARDBG
        box(s, Inches(0.55), Inches(yy - 0.05), Inches(12.25), Inches(0.46), fill=bg)
        for j, (val, x, w) in enumerate(zip(r[:-1], x0, w0)):
            c = DARK if j == 0 else (col if j in (2, 3) else GRAY)
            b = (j == 0) or (j in (2, 3))
            text(s, Inches(x), Inches(yy), Inches(w), Inches(0.38),
                 [[(val, 12.5, c, b, False, MONO if j > 0 else None)]])
        yy += 0.48
    box(s, Inches(0.7), Inches(5.65), Inches(11.9), Inches(1.2), fill=CARDBG, round_=True)
    text(s, Inches(0.95), Inches(5.74), Inches(11.5), Inches(1.05),
         [[("세 가지 읽기: ", 13, NAVY, True, False, None),
           ("① Tᵢ 우위는 인접 이력에 국한되지 않는다 — Δt>15 ms 전체에서 미래를 쓰는 PCHIP에도 유의 승. "
            "② >105 ms는 양측 보간의 영역(실시간엔 없는 것) — 인과 대비로는 열세 없음. "
            "③ V_rot도 인과 대비로는 전 구간 유의 승 (≤15 ms +0.368 · >15 ms +0.309).",
            12.5, DARK, False, False, None)]], line_spacing=1.15)
    return s


# --- 21. Result 5: asymmetry + ablation ----------------------------------
def s_res_asym():
    s = slide()
    header(s, "6. 결과 ⑤", "Tᵢ ↔ V_rot 비대칭 — 본 연구의 과학적 발견")
    add_image_fit(s, os.path.join(FIG, "fig_ablation.png"),
                  Inches(0.55), Inches(1.45), Inches(7.2), Inches(5.2))
    box(s, Inches(7.95), Inches(1.55), Inches(4.85), Inches(5.0), fill=CARDBG, round_=True)
    text(s, Inches(8.2), Inches(1.72), Inches(4.4), Inches(0.5),
         [[("물리적 해석", 15, NAVY, True, False, None)]])
    bullets(s, Inches(8.2), Inches(2.3), Inches(4.4), Inches(4.2), [
        ("Tᵢ: 빠른 진단이 진짜 정보 운반", 0, ORANGE, True),
        ("충돌 e–i 결합 (t_ei ∝ Tₑ^1.5/nₑ) → ECEI(Tₑ)+BES(nₑ)", 1),
        ("fast-only도 persistence 능가 (+0.372)", 1),
        ("V_rot: 정보는 거의 전적으로 과거 CES 이력", 0, BLUE, True),
        ("토로이달 회전은 미관측 NBI 토크가 주도", 1),
        ("Mirnov 100 Hz 순간샘플 → kHz 모드 소실 (lag-1 r=−0.01)", 1),
        ("fast-only V_rot = −0.64 (persistence보다 나쁨)", 1, RED, True),
        ("‘Tₑ가 NBI 가열을 대리한다’ 가설도 기각", 0, RED, True),
        ("shot간 Tₑ~Tᵢ r=+0.35 (p~1e−17) — 경로는 실재", 1),
        ("그러나 Tₑ~V_rot r=+0.02 (p=0.58) — 전달 안 됨", 1),
        ("V_rot의 비-승리는 실패가 아니라 발견", 0, NAVY, True),
    ], size=12, gap=5)
    return s


# --- 22. Result 6: peak ---------------------------------------------------
def s_res_peak():
    s = slide()
    header(s, "6. 결과 ⑥", "모델의 우위는 '고변동(peak) 구간'에 집중")
    add_image_fit(s, os.path.join(FIG, "fig_peak.png"),
                  Inches(0.7), Inches(1.5), Inches(7.0), Inches(5.0))
    box(s, Inches(7.9), Inches(1.6), Inches(4.9), Inches(4.85), fill=CARDBG, round_=True)
    text(s, Inches(8.15), Inches(1.77), Inches(4.5), Inches(0.5),
         [[("어디서 가치를 버는가", 15, NAVY, True, False, None)]])
    bullets(s, Inches(8.15), Inches(2.35), Inches(4.5), Inches(4.0), [
        ("peak = 입력 기반 고-국소활동 이웃 (타겟 행 제외)", 0),
        ("pointwise 극값이 아니라 보수적 영역 proxy", 1),
        ("Tᵢ: global +0.272 → peak +0.702 (PASS)", 0, TEAL, True),
        ("보간은 매끄러운 bulk에서 거의 최적", 1),
        ("V_rot: global +0.131 → peak +0.438 (PASS)", 0, TEAL, True),
        ("global은 약하지만 peak에서는 통과 — 비대칭은 regional", 1),
        ("Tᵢ ablation: peak에서 빠른진단 제거 시 큰 손해 (유의)", 0, NAVY, True),
    ], size=12.5, gap=8)
    return s


# --- 22b. Result 7: transient case study ---------------------------------
def s_res_transient():
    s = slide()
    header(s, "6. 결과 ⑦", "급변 구간을 눈으로 — held-out shot #31815")
    add_image_fit(s, os.path.join(FIG, "fig_transient_31815.png"),
                  Inches(0.45), Inches(1.42), Inches(6.95), Inches(5.5))
    box(s, Inches(7.65), Inches(1.6), Inches(5.15), Inches(4.85), fill=CARDBG, round_=True)
    text(s, Inches(7.9), Inches(1.77), Inches(4.7), Inches(0.5),
         [[("한 shot에서 실제로 벌어지는 일", 15, NAVY, True, False, None)]])
    bullets(s, Inches(7.9), Inches(2.35), Inches(4.7), Inches(4.0), [
        ("빠른 진단이 급변을 먼저 본다", 0, NAVY, True),
        ("BES 급락(빨간 점선)이 CES crash와 정렬", 1),
        ("PCHIP는 스파이크마다 overshoot", 0, GRAY, True),
        ("과거+미래를 다 보는 오프라인 보간인데도", 1),
        ("모델은 과거 CES + 빠른 진단만 쓰는 causal", 0, GREEN, True),
        ("Tᵢ: RMSE 210 vs PCHIP 263 → skill +0.36", 0, ORANGE, True),
        ("V_rot: skill +0.20 — 이 shot은 두 타겟 모두 승리", 0, BLUE, True),
        ("peak 구간 Tᵢ skill +0.63", 1),
        ("단, 보간을 이기는 건 test 43/89 shot", 0, GRAY, True),
        ("우위는 gap·peak에 집중 (결과 ④⑥와 일관)", 1),
    ], size=12.5, gap=6)
    return s


# --- 22c. Result 8: two stress tests --------------------------------------
def s_stress():
    """MNAR reweighting (§8i) + campaign time split (§8n) — which claim survives.
    Numbers: data/.mnar_analysis.json, data/.campaign_summary.json."""
    s = slide()
    header(s, "6. 결과 ⑨", "스트레스 테스트 2종 — 어느 주장이 살아남는가", accent=ORANGE)
    yy = table(
        s, Inches(0.55), Inches(1.5),
        [Inches(4.6), Inches(3.9), Inches(3.75)],
        ["평가", "vs PCHIP (오프라인·미래)", "vs persistence (인과)"],
        [
            ["무작위 분할 · 관측 지점 (headline)", ("+0.18~+0.28 · 4/4 PASS", GREEN, True), "+0.35~+0.42"],
            ["진짜 결측 지점으로 재가중 (MNAR)", ("+0.06~+0.21 · 1/4", RED, False), ("+0.29 · 4/4 PASS", GREEN, True)],
            ["캠페인 시간 분할 (미래 방전)", ("−0.15~+0.05 · 0/4", RED, False), ("+0.12~+0.28 (V_rot 4/4)", GREEN, True)],
        ],
        row_h=Inches(0.6), size=13)
    bullets(s, Inches(0.55), yy + Inches(0.28), Inches(12.2), Inches(2.9), [
        ("오프라인 보간 대비 우위는 무작위 분할·관측 모집단의 성질 — 배치 주장은 인과 우위로 좁힌다", 0, ORANGE, True),
        ("MNAR 재가중 부산물: W=4에서 진짜 결측 Tᵢ의 54.1% · V_rot의 4.8%만 도메인 내 (커버리지 한계 → reach 확장 지목)", 0),
        ("캠페인 손실의 원인을 측정: 드리프트 BES 1.22σ · ECEI 0.53σ vs CES 타겟 0.115σ — 보간 대비 우위를 사는 고속 진단 경로가 5~11배 더 이동", 0),
        ("→ 지목된 수리: 고속 진단의 shot별(인과) 표준화 — 누수 없음 · 실행 시점 가용 · 절충은 다음 통제 실험", 1),
        ("V_rot조차 캠페인 분할에서 persistence를 4/4 유의 승 — '실패한 타겟'이 아니라 '의미 있는 비교가 인과뿐인 타겟'", 0),
    ], size=13, gap=7)
    return s


# --- 22d. Result 9: deployment measured ------------------------------------
def s_deploy():
    """Latency (§8l), conformal UQ (§8m), complexity ladder (§8k)."""
    s = slide()
    header(s, "6. 결과 ⑩", "배치 가능성 — 주장이 아니라 측정")
    card(s, Inches(0.55), Inches(1.5), Inches(6.0), Inches(2.45),
         "지연시간 — CPU에서 10 ms 예산 안에 든다", [
             "batch-1 p99 = 6.4 ms (W=4) · 중앙값 2.8 ms — 주기의 64%",
             "반직관: CUDA는 batch 1에서 ~8× 느림 (p99 43–72 ms)",
             "0.2M 파라미터로는 커널 실행 오버헤드를 못 상쇄",
             "→ 실무 지침: 제어 컴퓨터의 CPU에서 돌려라",
             "(측정 2방식 — 호출별·amortized — 일치)"],
         accent=TEAL, body_size=12)
    card(s, Inches(6.8), Inches(1.5), Inches(6.0), Inches(2.45),
         "불확실성 — split conformal (재학습 없음)", [
             "val에서 캘리브레이션 · 예측기 불변 · 분포 무가정 (α=0.10)",
             "동일 절차를 PCHIP·persistence에도 적용한 공정 비교에서",
             "모델 구간이 8/8 시드·타겟 조합 Winkler 점수 승 (~0.80×)",
             "정직한 실패: 커버리지는 주변적 — shot별 50~100% 산포",
             "(shot-조건부 보정은 현 shot 수로 불가함을 명시)"],
         accent=BLUE, body_size=12)
    card(s, Inches(0.55), Inches(4.2), Inches(12.25), Inches(2.35),
         "복잡도 사다리 — '너무 복잡하다'에 측정으로 답한다 (§8k)", [
             "완전 해석 가능한 앵커+Δ 모델(1,258 파라미터 = 0.6%; 항별 분해 가능; persistence에서 정확히 출발)을 같은 4분할에 올리면:",
             "persistence −0.272  →  앵커+Δ −0.113  →  전체 모델 +0.234 (Tᵢ 평균)  — 해석 가능한 형태가 Tᵢ 마진의 31.5%(V_rot 7%)를 회수",
             "→ 남은 20만 파라미터가 사는 것은 비선형·비국소 구조라고 정량적으로 말할 수 있다 (28~36%로 4분할 일관).",
         ], accent=ORANGE, body_size=12.5)
    return s


# --- 23. Conclusion -------------------------------------------------------
def s_conclusion():
    s = slide()
    header(s, "7. 결론", "정직한 결론 (4가지)")
    items = [
        ("1", "CES_TI: 미래 보간도 유의하게 능가 (관측 모집단 headline)", BLUE,
         "genuine skill_vs_pchip +0.18~+0.28, 4개 독립 분할 모두 shot-clustered 95% CI가 0 제외. held 포함/제외 두 평가 모집단 모두에서 생존, 비인접 영역(Δt>15 ms)까지 포함."),
        ("2", "배치 주장은 인과 우위 — 스트레스 테스트 2종을 생존하는 유일한 주장", GREEN,
         "진짜 결측 재가중(+0.29, 4/4)과 캠페인 시간 분할(+0.22; V_rot 4/4)을 통과. 오프라인 보간 대비 우위는 어느 쪽도 통과 못함(1/4, 0/4) — 온라인 가상 센서의 경쟁 상대는 persistence다."),
        ("3", "CES_VT: 보간과 동률 (PASS 1/4 = 잡음 수준)", GRAY,
         "point estimate는 4 seed 모두 +지만 승리 주장 안 함. 단 인과 대안은 캠페인 분할에서도 4/4 유의하게 이김 — 회전은 실패한 타겟이 아니라 의미 있는 비교가 인과뿐인 타겟."),
        ("4", "Tᵢ ↔ V_rot 비대칭 = 과학적 기여", ORANGE,
         "빠른 진단은 10 ms에서 Tᵢ 정보는 운반하나(fast-only +0.37) V_rot 정보는 거의 없음(fast-only −0.64; NBI 토크 미관측 + Mirnov aliasing 실측). 물리로 예측되고 ablation으로 확인됨."),
    ]
    yy = 1.55
    for num, t, col, body in items:
        box(s, Inches(0.6), Inches(yy), Inches(12.2), Inches(1.18), fill=CARDBG, round_=True)
        box(s, Inches(0.72), Inches(yy + 0.17), Inches(0.84), Inches(0.84), fill=col, round_=True)
        text(s, Inches(0.72), Inches(yy + 0.17), Inches(0.84), Inches(0.84),
             [[(num, 26, WHITE, True, False, None)]], align=PP_ALIGN.CENTER,
             anchor=MSO_ANCHOR.MIDDLE)
        text(s, Inches(1.75), Inches(yy + 0.13), Inches(10.9), Inches(0.45),
             [[(t, 16, NAVY, True, False, None)]])
        text(s, Inches(1.75), Inches(yy + 0.58), Inches(10.9), Inches(0.55),
             [[(body, 12.5, DARK, False, False, None)]], line_spacing=1.08)
        yy += 1.32
    return s


# --- 24b. Follow-up: can Mirnov be salvaged for V_rot? --------------------
def s_mirnov():
    s = slide()
    header(s, "7. 추가 검증", "\"Mirnov를 더 잘 쓰면 V_rot이 되지 않나?\" — 검증된 음성 결과")
    add_image_fit(s, os.path.join(FIG, "fig_mirnov.png"),
                  Inches(0.65), Inches(1.40), Inches(12.0), Inches(3.50))
    card(s, Inches(0.55), Inches(5.08), Inches(3.90), Inches(1.74), "① 진단 (실측)", [
        "MC = 100 Hz 격자의 순간 dB/dt 스냅샷",
        "lag-1 r: MC −0.01 vs BES +0.57 · ECEI +0.57",
        "→ kHz 모드가 무작위 위상으로 접혀 소실",
    ], accent=RED, title_size=14, body_size=10.5)
    card(s, Inches(4.67), Inches(5.08), Inches(3.90), Inches(1.74), "② 시도와 검정", [
        "적분·PCHIP 적분·|MC|·이동 RMS 인과 파생",
        "셔플 대조군: 적분 n.s. · |MC|↔Tᵢ 천이 유의",
        "학습 개선 없음 (Tᵢ 평균 −0.074, 4/4 음수)",
    ], accent=ORANGE, title_size=14, body_size=10.5)
    card(s, Inches(8.79), Inches(5.08), Inches(3.90), Inches(1.74), "③ 함의", [
        "부수: V_rot이 분할 시드 따라 −0.15~+0.86 요동",
        "→ 소규모 학습에선 V_rot 단일 수치 측정 불가",
        "레버는 NBI 토크 확보 · 원본 kHz Mirnov",
    ], accent=TEAL, title_size=14, body_size=10.5)
    return s


# --- 24b. Expected rebuttal: does Te proxy the NBI torque? ---------------
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
        [("Tₑ ~ CES_TI", DARK, True, None), ("+0.353", GREEN, True, MONO), ("2.9e−17", GREEN, True, MONO)],
        [("Tₑ ~ CES_VT", DARK, True, None), ("+0.024", RED, True, MONO), ("0.58", RED, True, MONO)],
        [("Tₑ ~ |CES_VT|", DARK, False, None), ("+0.001", RED, False, MONO), ("0.98", RED, False, MONO)],
        [("Tₑ 변동성 ~ |CES_VT|", DARK, False, None), ("−0.026", GRAY, False, MONO), ("0.55", GRAY, False, MONO)],
        [("BES 변동성 ~ |CES_VT|", DARK, False, None), ("−0.059", GRAY, False, MONO), ("0.17", GRAY, False, MONO)],
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
           (" — 토크는 빔 에너지·접선 반경·주입 각도에 의존해 power와 분리되고, 회전은 "
            "운동량 수송과 경계 제동(NTV·오차장·벽 마찰)에 지배된다.", 12, DARK, False, False, None)]],
         line_spacing=1.12)
    return s


# --- 25. Limitations + future --------------------------------------------
def s_window_sweep():
    """Window sweep (2026-08-04, 24 runs) — answers "why window = 4?".
    Numbers: THESIS_RESULTS.md §8f / data/.wsweep_summary.json."""
    s = slide()
    header(s, "6. 결과 ⑧", "window 민감도 sweep — 'W=4여야 하는가'에 답하다", accent=TEAL)
    text(s, Inches(0.55), Inches(1.36), Inches(12.3), Inches(0.56),
         [[("W ∈ {2,3,4,6,8} × seed 4개 + history-0 = ", 13, DARK, False, False, None),
           ("독립 run 24회", 13, NAVY, True, False, None),
           (". 발표 모델(iter009) 고정, W만 변화 — 매 run 자체의 held-out TEST "
            "skill_vs_pchip. held 제거 학습·평가, seed별 test shot 96개는 모든 W에서 동일(검증됨).",
            13, DARK, False, False, None)]], line_spacing=1.12)
    add_image_fit(s, os.path.join(FIG, "fig_window_sweep.png"),
                  Inches(0.45), Inches(1.98), Inches(7.75), Inches(4.15))
    box(s, Inches(8.35), Inches(1.98), Inches(4.45), Inches(4.15), fill=CARDBG, round_=True)
    text(s, Inches(8.6), Inches(2.12), Inches(4.0), Inches(0.5),
         [[("곡선이 말하는 세 가지", 15, NAVY, True, False, None)]])
    bullets(s, Inches(8.6), Inches(2.66), Inches(4.0), Inches(3.4), [
        ("history가 없으면 무너진다", 0, RED, True),
        ("Tᵢ −0.026 (보간에 짐) · V_rot −0.78", 1),
        ("→ 빠른 진단만으론 보간에 못 미친다", 1),
        ("과거 관측 1개가 전부를 만든다", 0, GREEN, True),
        ("W=2에서 Tᵢ +0.238 · V_rot +0.206", 1),
        ("두 타겟 모두 이후 평탄 (추세 없음)", 1),
        ("구간 폭 < seed 산포 0.07~0.16", 1),
        ("W>2의 근거는 skill 아닌 커버리지", 0, BLUE, True),
        ("긴 gap 샘플 채점 수 4~10배 (V_rot)", 1),
    ], size=12, gap=7)
    text(s, Inches(0.55), Inches(6.22), Inches(12.3), Inches(0.62),
         [[("결론  ", 12.5, TEAL, True, False, None),
           ("W=4는 근거 없이 쓰던 기본값이었고 두 타겟 모두 W=2/3보다 낮다 → 이 곡선으로 W=2를 선택. ",
            12.5, DARK, False, False, None),
           ("1차 실험의 'V_rot은 긴 window가 필요하다'는 held 오염이 짧은 window를 "
            "벌해서 생긴 착시였다.", 12.5, GRAY, False, True, None)]],
         line_spacing=1.15)
    return s


def s_limits():
    s = slide()
    header(s, "7. 한계 & 향후 연구", "한계와 다음 단계")
    box(s, Inches(0.55), Inches(1.55), Inches(6.0), Inches(4.9), fill=CARDBG, round_=True)
    box(s, Inches(0.55), Inches(1.55), Inches(0.12), Inches(4.9), fill=RED)
    text(s, Inches(0.8), Inches(1.72), Inches(5.5), Inches(0.5),
         [[("한계", 16, RED, True, False, None)]])
    bullets(s, Inches(0.8), Inches(2.3), Inches(5.55), Inches(4.0), [
        ("통계적 검정력: test shot ≈ 96(Tᵢ)/61(V_rot genuine) — 모든 유의성의 구속조건", 0),
        ("MNAR: 관측 지점 채점은 낙관 상한 — 재가중으로 정량화(결과 ⑧): 인과 우위만 생존", 0),
        ("커버리지: W=4에서 진짜 결측 Tᵢ 54.1% · V_rot 4.8%만 도메인 내", 0),
        ("캠페인 전이: 시간 분할에서 오프라인 보간 우위 소멸(0/4) — 원인 측정, 수리 미실행", 0),
        ("불확실성 구간은 주변적으로만 보정 (shot별 50~100%)", 0),
        ("metric 비대칭: 보간은 full-shot 이웃, 모델은 W=4 · >105 ms는 양측 보간 우세", 0),
        ("단일 장치·단일 모델 계열 (window·CT 인코더 민감도는 특성화 완료)", 0),
    ], size=12, gap=8)
    box(s, Inches(6.8), Inches(1.55), Inches(6.0), Inches(4.9), fill=CARDBG, round_=True)
    box(s, Inches(6.8), Inches(1.55), Inches(0.12), Inches(4.9), fill=TEAL)
    text(s, Inches(7.05), Inches(1.72), Inches(5.5), Inches(0.5),
         [[("향후 연구", 16, TEAL, True, False, None)]])
    bullets(s, Inches(7.05), Inches(2.3), Inches(5.55), Inches(4.0), [
        ("음성 결과는 뒤집을 측정을 지목할 때만 보고 — 세 레버 전부 자체 측정으로 지목", 0, NAVY, True),
        ("① 이력 reach 확장: 슬롯 2~3개 유지 + 더 넓은 span — 커버리지(54.1%/4.8%) 직접 개선", 0),
        ("② 원 kHz Mirnov window 특징(RMS·대역 파워·모드 번호) — V_rot 최대 레버", 0),
        ("   (100 Hz 무필터 데시메이션이 파괴한 정보의 상류 복원)", 1),
        ("③ NBI 토크 데이터 확보 — 회전의 원인 변수 (액추에이터 입력 시 회전 예측 가능: 양성 대조군 존재)", 0),
        ("④ 고속 진단 shot별 표준화 — 캠페인 전이 수리 (통제 실험 설계 완료)", 0),
        ("다중 캠페인 확장 → 검정력 직접 보강", 0),
    ], size=12, gap=8)
    return s


# --- 26. Takeaways / closing ---------------------------------------------
def s_closing():
    s = slide()
    box(s, 0, 0, EMU_W, EMU_H, fill=NAVY)
    box(s, Inches(0.9), Inches(0.95), Inches(2.2), Pt(4), fill=ORANGE)
    text(s, Inches(0.9), Inches(1.15), Inches(11.5), Inches(0.6),
         [[("한 장 요약 — Key Takeaways", 26, WHITE, True, False, None)]])
    points = [
        ("항상 있는 빠른 진단으로 자주 비는 CES를 채우는 데이터 기반 가상 센서", ORANGE),
        ("CES_TI는 미래까지 보는 보간을 4 seed 모두 유의하게 능가 (genuine +0.18~+0.28)", GREEN),
        ("배치 주장은 인과 우위 — 결측 재가중·캠페인 분할을 생존하는 유일한 주장 (+0.29/+0.22)", BLUE),
        ("CES_VT는 보간과 동률 — 빠른 진단에 회전 정보가 없다는 Tᵢ↔V_rot 비대칭 (발견)", TEAL),
        ("가치는 고변동(peak) 구간에 집중 · CPU p99 6.4 ms로 실시간 · conformal 구간 8/8 승", ORANGE),
        ("데이터 품질 감사(held 54% → 학습·평가 제거 규약) + 선택 게이트 교체가 유의를 만듦", BLUE),
    ]
    yy = 2.1
    for t, col in points:
        box(s, Inches(0.95), Inches(yy + 0.05), Inches(0.28), Inches(0.28), fill=col)
        text(s, Inches(1.45), Inches(yy - 0.04), Inches(11.0), Inches(0.55),
             [[(t, 15.5, WHITE, False, False, None)]], line_spacing=1.1)
        yy += 0.62
    box(s, Inches(0.9), Inches(6.15), Inches(11.5), Pt(2), fill=RGBColor(0x2A, 0x47, 0x6E))
    text(s, Inches(0.9), Inches(6.35), Inches(11.5), Inches(0.7),
         [[("감사합니다.  ", 20, WHITE, True, False, None),
           ("Q & A", 20, ORANGE, True, False, None)]])
    return s


# ======================= build ===========================================
def build():
    s_title()
    s_agenda()
    divider("1", "연구 배경 & 문제 정의", "CES는 왜 자주 비는가 — 가상 센서의 필요성")
    s_diagnostics()
    s_problem()
    s_missing_table()
    s_idea()
    divider("2", "접근법", "의도적으로 어려운 평가 bar 설정")
    s_bar()
    s_validation()
    divider("3", "데이터 & 파이프라인", "No-Fake-Data · 마스킹 · 누수 방지")
    s_data()
    s_contract()
    s_split()
    s_samples()
    s_stuck()
    divider("4", "모델 아키텍처", "Multimodal Late-Fusion + target별 routing")
    s_arch()
    s_arch_detail()
    s_training()
    divider("5", "평가 방법론", "통계적 엄밀성: split · 사전등록 · bootstrap")
    s_methodology()
    s_bootstrap()
    divider("6", "결과", "causal 압도 · Tᵢ 보간 유의 · 비대칭")
    s_res_ladder()
    s_res_forest()
    s_res_prog()
    s_res_gap()
    s_res_asym()
    s_res_peak()
    s_res_transient()
    s_window_sweep()
    s_stress()
    s_deploy()
    divider("7", "결론 · 한계 · 향후 연구", "정직한 결론과 다음 단계")
    s_conclusion()
    s_mirnov()
    s_te_nbi()
    s_limits()
    s_closing()

    out = os.path.join(HERE, "KSTAR_CES_발표자료.pptx")
    prs.save(out)
    print("SAVED:", out, "| slides:", len(prs.slides._sldIdLst))


if __name__ == "__main__":
    build()
