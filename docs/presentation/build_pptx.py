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
        ("7", "AutoML 자율 연구 루프", "Claude 기반 keep/discard autoresearch", TEAL),
        ("8", "결론 · 한계 · 향후 연구", "정직한 결론과 통계적 검정력 한계", GRAY),
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
        ("같은 10 ms 격자에서 Tᵢ ≈ 8%, V_rot ≈ 24% 결측", 0),
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
    text(s, Inches(1.0), Inches(5.18), Inches(11.4), Inches(1.3),
         [[("왜 이 정보 비대칭이 핵심인가", 13.5, ORANGE, True, False, None)],
          [("미래까지 보는 보간을 causal(과거만 보는) 모델이 이긴다면, 그것은 빠른 진단이 "
            "시간 보간으로는 얻을 수 없는 CES 정보를 운반한다는 강력한 증거다.",
            15.5, WHITE, False, False, None)],
          [("미래 보간을 이기는 모델은 자명히 모든 causal baseline(persistence·AR)도 이긴다.",
            13.5, LGRAY, False, True, None)]], line_spacing=1.16)
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
        ("플라즈마 상태 타겟: H-mode ELM suppression (RMP 인가)", 0),
        ("ELM suppression 유지 중 Dα가 크게 튀는 구간 중심 ~100 ms로 절단", 1),
        ("샷 번호: #24000 ~ #33000 우선 선정", 0),
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
        ("dataset이 이를 탐지·마스킹 가능 (CES_DROP_STUCK_TARGETS)", 0),
        ("학습은 오염 아님 — held 값 마스킹해도 genuine 성능 향상 없음", 0),
        ("→ 학습은 유지(=0), 평가만 genuine 측정값으로(=1)", 1, NAVY, True),
        ("CES_TI는 genuine-only 평가에서도 4 seed 모두 PASS (강건)", 0, GREEN, True),
    ], size=13.5, gap=9)
    box(s, Inches(7.1), Inches(2.55), Inches(5.7), Inches(3.9), fill=CARDBG, round_=True)
    text(s, Inches(7.35), Inches(2.7), Inches(5.3), Inches(0.5),
         [[("V_rot 물리 RMSE는 held 값에 의해 deflated", 14, NAVY, True, False, None)]])
    # mini table
    hdr = ["seed", "보고(stuck포함)", "genuine RMSE"]
    rows = [["42", "22.4", "35.0"], ["1", "24.6", "34.6"],
            ["7", "29.7", "43.2"], ["123", "32.6", "46.2"]]
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

    def node(x, y, w, h, title, sub, col):
        b = box(s, Inches(x), Inches(y), Inches(w), Inches(h), fill=col, round_=True)
        text(s, Inches(x), Inches(y + 0.06), Inches(w), Inches(h - 0.1),
             [[(title, 12.5, WHITE, True, False, None)],
              [(sub, 9.5, RGBColor(0xEA, 0xF0, 0xF7), False, False, None)]],
             align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE, line_spacing=1.0, space_after=1)
        return b

    def arrow(x1, y1, x2, y2, col=MGRAY):
        cn = s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1),
                                    Inches(x2), Inches(y2))
        cn.line.color.rgb = col
        cn.line.width = Pt(2.0)
        cn.shadow.inherit = False
        le = cn.line._get_or_add_ln()
        tail = le.makeelement(qn('a:tailEnd'),
                              {'type': 'triangle', 'w': 'med', 'len': 'med'})
        le.append(tail)
        return cn

    # inputs
    inputs = [("BES (4×9)", "밀도요동", BLUE, 1.45),
              ("ECEI (4×4)", "전자온도", TEAL, 2.45),
              ("MC (4×2)", "자기요동", GRAY, 3.45),
              ("time (4×4)", "불규칙시간", NAVY, 4.45),
              ("ces_history (4×4)", "이전 CES + flag", ORANGE, 5.45)]
    for t, sub, col, y in inputs:
        node(0.55, y, 2.0, 0.78, t, sub, col)

    # encoders
    encs = [("BES Enc", "time-aware CNN", BLUE, 1.45),
            ("ECEI Enc", "time-aware CNN", TEAL, 2.45),
            ("MC Enc", "time-aware CNN", GRAY, 3.45),
            ("Time Enc", "1D CNN", NAVY, 4.45),
            ("History Enc", "Pre-LN Transformer\n+ multi-head attn pool", ORANGE, 5.45)]
    for (t, sub, col, y) in encs:
        node(3.15, y, 2.2, 0.78, t, sub, col)
        arrow(2.55, y + 0.39, 3.15, y + 0.39)

    # fusion / routing
    node(6.05, 1.7, 2.5, 1.25, "Tᵢ 융합", "bes+ecei+mc\n+time+hist_ti", ORANGE)
    node(6.05, 4.55, 2.5, 1.25, "V_rot 융합", "hist_vt + time\n(빠른진단 제외)", BLUE)
    # arrows to fusion
    for y in [1.45, 2.45, 3.45, 4.45]:
        arrow(5.35, y + 0.39, 6.05, 2.32, col=RGBColor(0xCF, 0xB0, 0x90))
    arrow(5.35, 5.84, 6.05, 5.17, col=ORANGE)  # hist_vt -> vt
    arrow(5.35, 5.45, 6.05, 2.6, col=ORANGE)   # hist_ti -> ti

    # heads
    node(9.0, 1.85, 2.0, 0.95, "Tᵢ Head", "preLN→MLP", ORANGE)
    node(9.0, 4.7, 2.0, 0.95, "V_rot Head", "preLN→MLP", BLUE)
    arrow(8.55, 2.32, 9.0, 2.32, col=ORANGE)
    arrow(8.55, 5.17, 9.0, 5.17, col=BLUE)

    # output
    node(11.4, 3.25, 1.6, 1.1, "[Tᵢ, V_rot]", "정규화 (B,2)", NAVY)
    arrow(11.0, 2.32, 11.9, 3.25, col=NAVY)
    arrow(11.0, 5.17, 11.9, 4.35, col=NAVY)

    # labels
    text(s, Inches(0.55), Inches(1.18), Inches(2.0), Inches(0.3),
         [[("입력", 11, GRAY, True, False, None)]], align=PP_ALIGN.CENTER)
    text(s, Inches(3.15), Inches(1.18), Inches(2.2), Inches(0.3),
         [[("진단별 인코더", 11, GRAY, True, False, None)]], align=PP_ALIGN.CENTER)
    text(s, Inches(6.05), Inches(1.18), Inches(2.5), Inches(0.3),
         [[("target별 융합(routing)", 11, GRAY, True, False, None)]], align=PP_ALIGN.CENTER)
    text(s, Inches(9.0), Inches(1.18), Inches(2.0), Inches(0.3),
         [[("예측 head", 11, GRAY, True, False, None)]], align=PP_ALIGN.CENTER)
    text(s, Inches(0.55), Inches(6.55), Inches(12.3), Inches(0.5),
         [[("물리 기반 routing: ", 12.5, NAVY, True, False, None),
           ("Tᵢ = 빠른진단+이력+시간 / V_rot = 이력+시간만. 전체 파라미터 < 1,000,000 (capacity가 아니라 일반화가 관건).",
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
        ("History Encoder", ORANGE,
         ["2-layer Pre-LayerNorm Transformer",
          "학습된 positional embedding",
          "self-attention이 masked 타겟을 양쪽",
          "관측 이웃에 직접 attend (보간과 유사)"]),
        ("Multi-head attention pooling", TEAL,
         ["가중 평균(center) + 가중 분산(std) 함께",
          "std가 '고변동 구간'을 노출 → peak 구분",
          "target별 관측-이웃 bias (init 0)"]),
        ("target별 head + pre-head LN", NAVY,
         ["Tᵢ/V_rot 분리 head, 이종 융합특징 정규화",
          "Pre-LN 경로를 융합단까지 확장",
          "capacity가 아닌 일반화 lever"]),
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


# --- 15. Evaluation methodology: split + prereg --------------------------
def s_methodology():
    s = slide()
    header(s, "5. 평가 방법론", "선택 편향 없는 3-way split + 사전등록")
    cards = [
        ("3-way split (train/val/test)", BLUE,
         ["TEST는 AutoML 시작 전 예약, 탐색루프는 절대 안 봄",
          "모델 선택은 val에서만 → 헤드라인은 winner's-curse 없음",
          "TEST: 34,644 샘플 / 96 shot",
          "관측 후 n: Tᵢ 32,716 · V_rot 27,437"]),
        ("사전등록 (PR1–PR4)", ORANGE,
         ["PR1 best-interpolation 규칙: headline = PCHIP",
          "PR2 평가 모집단: future-neighbor 없으면 persistence",
          "PR3 TEST floor: ≥15 shot & ≥3,000 Tᵢ 샘플 (충족)",
          "PR4 부트스트랩 구성: 95% CI가 0 제외 = PASS"]),
    ]
    for i, (t, col, lines) in enumerate(cards):
        card(s, Inches(0.55 + i * 6.2), Inches(1.55), Inches(6.0), Inches(2.95),
             t, lines, accent=col, title_size=15, body_size=12)
    box(s, Inches(0.7), Inches(4.8), Inches(11.9), Inches(1.75), fill=CARDBG, round_=True)
    text(s, Inches(0.95), Inches(4.95), Inches(11.4), Inches(1.5),
         [[("공정성 보장", 14, NAVY, True, False, None)],
          [("모든 arm(모델+모든 baseline)을 동일한 (file, row) 샘플 집합 · 동일 per-target keep mask로 채점. "
            "어떤 arm도 상대적으로 thinning 되지 않음.", 13.5, DARK, False, False, None)],
          [("보간은 타겟 자신의 값(row_index)을 제외하고 이웃만 읽음(누수 없음), 0.5 s 이상 gap은 보간 거부.",
            13, GRAY, False, False, None)]], line_spacing=1.18)
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
         ["4개 독립 test split(seed 42/1/7/123) 모두 CI > 0",
          "skill_vs_pchip = +0.20 ~ +0.30",
          "(1/7/123은 아키텍처 선택에 한 번도 안 쓰임)"]),
        ("CES_VT — n.s.", GRAY,
         ["4 seed 모두 CI가 0을 포함 (유의하지 않음)",
          "point estimate는 +지만 통계적 미지지",
          "→ Tᵢ↔V_rot 비대칭 (다음 슬라이드)"]),
    ]
    for i, (t, col, lines) in enumerate(cards):
        card(s, Inches(0.55 + i * 6.2), Inches(5.35), Inches(6.0), Inches(1.55),
             t, lines, accent=col, title_size=13.5, body_size=11.5)
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
        ("AutoML 루프가 val skill_vs_pchip로 모델을 선택", 0),
        ("→ GRU 이력인코더 + target별 multi-head attention head", 1),
        ("최종(iter5): +0.20~+0.30, 4 seed 모두 PASS", 0, GREEN, True),
        ("핵심: val loss가 아니라 clean skill로 선택한 것이 결정적", 0, ORANGE, True),
    ], size=13, gap=11)
    return s


# --- 20. Result 4: gap-stratified ----------------------------------------
def s_res_gap():
    s = slide()
    header(s, "6. 결과 ④", "Gap별 분석: 모델의 우위는 small-gap에 집중")
    text(s, Inches(0.55), Inches(1.45), Inches(12.3), Inches(0.5),
         [[("실데이터는 압도적으로 small-gap: Δt ≤ 15 ms가 Tᵢ 31,966/32,716 · V_rot 26,938/27,437 — "
            "이 구간이 nowcasting이 실제로 필요한 곳.", 13.5, DARK, False, False, None)]])
    # table
    headers = ["Δt bin", "n (Tᵢ)", "skill Tᵢ", "n (V_rot)", "skill V_rot"]
    rows = [
        ["(0, 15] ms", "31,966", "+0.080", "26,938", "+0.234", GREEN],
        ["(15, 25] ms", "520", "+0.003", "357", "+0.029", GRAY],
        ["(25, 35] ms", "140", "+0.401", "97", "−0.123", GRAY],
        ["(35, 55] ms", "33", "−0.823", "14", "+0.870", GRAY],
        ["(105, ∞) ms", "45", "−6.92", "30", "−2783", RED],
    ]
    x0 = [0.7, 3.0, 5.0, 7.3, 9.6]
    w0 = [2.2, 1.9, 2.2, 1.9, 2.5]
    yy = 2.15
    box(s, Inches(0.55), Inches(yy - 0.05), Inches(12.25), Inches(0.5), fill=NAVY)
    for h, x, w in zip(headers, x0, w0):
        text(s, Inches(x), Inches(yy), Inches(w), Inches(0.4),
             [[(h, 13, WHITE, True, False, None)]])
    yy += 0.55
    for r in rows:
        col = r[-1]
        bg = WHITE if (rows.index(r) % 2 == 0) else CARDBG
        box(s, Inches(0.55), Inches(yy - 0.05), Inches(12.25), Inches(0.5), fill=bg)
        for j, (val, x, w) in enumerate(zip(r[:-1], x0, w0)):
            c = DARK if j == 0 else (col if j in (2, 4) else GRAY)
            b = (j == 0) or (j in (2, 4))
            text(s, Inches(x), Inches(yy), Inches(w), Inches(0.4),
                 [[(val, 13, c, b, False, MONO if j > 0 else None)]])
        yy += 0.52
    box(s, Inches(0.7), Inches(5.55), Inches(11.9), Inches(1.15), fill=CARDBG, round_=True)
    text(s, Inches(0.95), Inches(5.66), Inches(11.5), Inches(1.0),
         [[("읽는 법: ", 13, NAVY, True, False, None),
           ("잘 통제된(표본 충분한) 유일한 비교인 small-gap이 모델 우세. Δt가 커지면 표본 수십 개 미만 "
            "bin들이 부호를 뒤집고(미래 anchor를 가진 PCHIP이 압도), CI가 없으므로 과대해석 금물.",
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
        ("fast-only도 persistence 능가 (+0.162)", 1),
        ("V_rot: 정보는 거의 전적으로 과거 CES 이력", 0, BLUE, True),
        ("토로이달 회전은 미관측 NBI 토크가 주도", 1),
        ("Mirnov은 100 Hz로 aliasing → 회전 정보 소실", 1),
        ("fast-only V_rot = −3.31 (평균보다 나쁨)", 1, RED, True),
        ("V_rot의 비-승리는 실패가 아니라 발견", 0, NAVY, True),
    ], size=12.5, gap=7)
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
        ("Tᵢ: global +0.515 → peak +0.855 (PASS)", 0, TEAL, True),
        ("보간은 매끄러운 bulk에서 거의 최적", 1),
        ("V_rot: global +0.241 → peak +0.691 (PASS)", 0, TEAL, True),
        ("global은 약하지만 peak에서는 통과 — 비대칭은 regional", 1),
        ("Tᵢ ablation: peak에서 빠른진단 제거 시 큰 손해 (유의)", 0, NAVY, True),
    ], size=12.5, gap=8)
    return s


# --- 23. AutoML loop ------------------------------------------------------
def s_automl():
    s = slide()
    header(s, "7. AutoML 자율 연구 루프", "Claude 기반 keep/discard autoresearch")
    text(s, Inches(0.55), Inches(1.4), Inches(12.3), Inches(0.5),
         [[("단일 모델 구현에 그치지 않고, ", 14, DARK, False, False, None),
           ("LLM이 연구자(Researcher) 역할", 14, ORANGE, True, False, None),
           ("을 맡아 model.py를 통제된 실험으로 개선 (Karpathy의 autoresearch에서 영감).",
            14, DARK, False, False, None)]])

    def step(x, title, sub, col):
        box(s, Inches(x), Inches(2.1), Inches(2.25), Inches(1.5), fill=col, round_=True)
        text(s, Inches(x), Inches(2.25), Inches(2.25), Inches(1.3),
             [[(title, 14, WHITE, True, False, None)],
              [(sub, 10.5, RGBColor(0xEC, 0xF1, 0xF7), False, False, None)]],
             align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE, line_spacing=1.05, space_after=2)

    steps = [("① Evaluation", "smoke→train→\nclean evaluate", BLUE),
             ("② Briefing", "HANDOFF 갱신\nplateau 감지", TEAL),
             ("③ Research", "Claude가 1개\n통제변수 제안", ORANGE),
             ("④ Keep/Discard", "best 갱신 or\nmodel.py 롤백", NAVY)]
    xs = [0.7, 3.35, 6.0, 8.65]
    for x, (t, sub, col) in zip(xs, steps):
        step(x, t, sub, col)
    for x in xs[:-1]:
        cn = s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x + 2.25), Inches(2.85),
                                    Inches(x + 2.65), Inches(2.85))
        cn.line.color.rgb = MGRAY
        cn.line.width = Pt(2.5)
        cn.shadow.inherit = False
    box(s, Inches(11.1), Inches(2.1), Inches(1.7), Inches(1.5), fill=CARDBG, round_=True,
        line=MGRAY, line_w=1)
    text(s, Inches(11.1), Inches(2.25), Inches(1.7), Inches(1.3),
         [[("⟳ 반복", 14, NAVY, True, False, None)],
          [("best 위에만\n쌓임", 10.5, GRAY, False, False, None)]],
         align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE, line_spacing=1.05, space_after=2)
    cards = [
        ("핵심 규칙: keep/discard", ORANGE,
         ["고정 예산으로 학습 → clean non-aug val skill로 채점",
          "best 갱신하면 KEEP, 아니면 DISCARD + 자동 롤백",
          "→ 회귀(regression) 위에 절대 쌓지 않음"]),
        ("Researcher = Claude (opus)", BLUE,
         ["program.md = 편집가능한 agent 'skill'",
          "DATA_CONTRACT + PROJECT_KNOWLEDGE 주입",
          "출력은 contract 보존한 raw model.py"]),
        ("선택 게이트 = clean skill", TEAL,
         ["augmented val loss 아님 (신뢰 불가)",
          "clean mean skill_vs_pchip이 게이트",
          "test split은 선택에 절대 미사용"]),
    ]
    for i, (t, col, lines) in enumerate(cards):
        card(s, Inches(0.55 + i * 4.13), Inches(3.95), Inches(3.95), Inches(2.6),
             t, lines, accent=col, title_size=13.5, body_size=11.5)
    return s


# --- 24. Conclusion -------------------------------------------------------
def s_conclusion():
    s = slide()
    header(s, "8. 결론", "정직한 결론 (4가지)")
    items = [
        ("1", "causal baseline을 결정적으로 압도", GREEN,
         "persistence·AR 대비 두 타겟 모두 큰 마진 (Tᵢ 369 vs 487/1006). 강건하고 방어 가능한 결과 — 온라인/실시간에서 명확한 승자."),
        ("2", "CES_TI: 미래 보간도 유의하게 능가", BLUE,
         "최종 모델·4 seed에서 skill_vs_pchip +0.20~+0.30, shot-clustered 95% CI가 매번 0을 제외 (PASS). genuine-only 평가에서도 강건."),
        ("3", "CES_VT: 보간과 동률 (n.s.)", GRAY,
         "point estimate는 +지만 통계적으로 미지지. 검정력 한계(≈91 shot)와 heavy-tailed 오차가 구속. 과대주장하지 않음."),
        ("4", "Tᵢ ↔ V_rot 비대칭 = 과학적 기여", ORANGE,
         "빠른 진단은 10 ms에서 Tᵢ 정보는 운반하나 V_rot 정보는 거의 없음 (NBI 토크 미관측 + Mirnov aliasing). 물리적으로 예측되고 ablation으로 확인됨."),
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


# --- 25. Limitations + future --------------------------------------------
def s_limits():
    s = slide()
    header(s, "8. 한계 & 향후 연구", "한계와 다음 단계")
    box(s, Inches(0.55), Inches(1.55), Inches(6.0), Inches(4.9), fill=CARDBG, round_=True)
    box(s, Inches(0.55), Inches(1.55), Inches(0.12), Inches(4.9), fill=RED)
    text(s, Inches(0.8), Inches(1.72), Inches(5.5), Inches(0.5),
         [[("한계", 16, RED, True, False, None)]])
    bullets(s, Inches(0.8), Inches(2.3), Inches(5.55), Inches(4.0), [
        ("통계적 검정력: test shot ≈ 96(Tᵢ)/91(V_rot) — 모든 유의성의 구속조건", 0),
        ("heavy-tailed 오차: 소수 방전이 부트스트랩 분산을 지배", 0),
        ("MNAR 낙관적 상한: 관측 지점에서만 측정", 0),
        ("metric 비대칭: 보간은 full-shot 이웃, 모델은 window=4", 0),
        ("단일 아키텍처·단일 window(4): 민감도 미특성화", 0),
        ("큰 Δt bin은 표본 수십 개·CI 없음 → 비신뢰", 0),
    ], size=12.5, gap=9)
    box(s, Inches(6.8), Inches(1.55), Inches(6.0), Inches(4.9), fill=CARDBG, round_=True)
    box(s, Inches(6.8), Inches(1.55), Inches(0.12), Inches(4.9), fill=TEAL)
    text(s, Inches(7.05), Inches(1.72), Inches(5.5), Inches(0.5),
         [[("향후 연구", 16, TEAL, True, False, None)]])
    bullets(s, Inches(7.05), Inches(2.3), Inches(5.55), Inches(4.0), [
        ("bracket-distance 층화: 가장 가까운 미래 anchor까지의 거리로 분석", 0, NAVY, True),
        ("보간이 가장 어려운 sub-population에서 powered한 유의 win 발굴 기대", 1),
        ("더 많은 test shot / 다중 캠페인 → 검정력 직접 보강", 0),
        ("peak-weighted loss: 고변동 샘플 upweight (이연된 실험)", 0),
        ("window 크기·아키텍처 민감도 체계적 sweep", 0),
        ("V_rot 보강: 이력을 더 효과적으로 활용하는 구조", 0),
    ], size=12.5, gap=9)
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
        ("미래까지 보는 오프라인 보간을 causal 모델로 이기는 의도적으로 어려운 bar", BLUE),
        ("CES_TI는 4 seed 모두 보간을 통계적으로 유의하게 능가 (shot-clustered CI)", GREEN),
        ("CES_VT는 동률(n.s.) — 빠른 진단에 회전 정보가 없다는 Tᵢ↔V_rot 비대칭", TEAL),
        ("모델의 가치는 고변동(peak) 구간에 집중 · causal baseline은 압도적으로 능가", ORANGE),
        ("Claude 기반 keep/discard autoresearch로 n.s. → 유의로 개선", BLUE),
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
    s_idea()
    divider("2", "접근법", "의도적으로 어려운 평가 bar 설정")
    s_bar()
    s_validation()
    divider("3", "데이터 & 파이프라인", "No-Fake-Data · 마스킹 · 누수 방지")
    s_data()
    s_contract()
    s_split()
    s_stuck()
    divider("4", "모델 아키텍처", "Multimodal Late-Fusion + target별 routing")
    s_arch()
    s_arch_detail()
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
    divider("7", "AutoML 자율 연구 루프", "Claude 기반 keep/discard autoresearch")
    s_automl()
    divider("8", "결론 · 한계 · 향후 연구", "정직한 결론과 다음 단계")
    s_conclusion()
    s_limits()
    s_closing()

    out = os.path.join(HERE, "KSTAR_CES_발표자료.pptx")
    prs.save(out)
    print("SAVED:", out, "| slides:", len(prs.slides._sldIdLst))


if __name__ == "__main__":
    build()
