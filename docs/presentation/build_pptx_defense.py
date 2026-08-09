# -*- coding: utf-8 -*-
"""Build the research-audit / defense deck (Korean).

Output: docs/presentation/KSTAR_CES_종합방어.pptx

Companion of docs/연구방어_종합문서.md (2026-08-05 전면 감사). Standalone on
purpose: importing build_pptx.py would rebuild the 1-hour deck as a side effect,
so the palette/layout helpers are copied here instead.

Numeric figures are taken from docs/paper/figures/ (regenerated from
paper_numbers.json by make_figures_en.py, so they cannot drift from the paper);
non-numeric diagrams come from docs/presentation/figures/.
"""
import os
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

HERE = os.path.dirname(__file__)
FIG = os.path.join(HERE, "figures")
FIG_P = os.path.join(HERE, "..", "paper", "figures")

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
        round_=False):
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
    return sp


def text(s, x, y, w, h, runs, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP,
         space_after=4, line_spacing=1.06, wrap=True):
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
        if isinstance(para, dict):
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
         [[(title, 26, NAVY, True, False, None)]])
    box(s, Inches(0.55), Inches(1.18), Inches(12.25), Pt(2), fill=LGRAY)
    footer(s)


def footer(s):
    _pageno["n"] += 1
    text(s, Inches(0.55), Inches(7.06), Inches(9.0), Inches(0.3),
         [[("KSTAR CES Nowcasting — 연구 종합 정리·방어 (2026-08-05 감사)", 9, MGRAY, False, False, None)]])
    text(s, Inches(11.6), Inches(7.06), Inches(1.2), Inches(0.3),
         [[(str(_pageno["n"]), 10, MGRAY, False, False, None)]], align=PP_ALIGN.RIGHT)


def bullets(s, x, y, w, h, items, size=15, gap=7, color=DARK):
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
    from PIL import Image
    try:
        iw, ih = Image.open(path).size
    except Exception:
        return s.shapes.add_picture(path, x, y, width=w)
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
s = slide()
box(s, 0, 0, EMU_W, EMU_H, fill=NAVY)
text(s, Inches(1.0), Inches(2.0), Inches(11.3), Inches(1.7),
     [[("KSTAR 희소 CES 신호의 멀티모달 나우캐스팅", 34, WHITE, True, False, None)],
      [("연구 종합 정리 · 방어 문서", 26, RGBColor(0xBF, 0xD3, 0xEE), True, False, None)]],
     space_after=10)
text(s, Inches(1.0), Inches(4.1), Inches(11.3), Inches(1.8),
     [[("주장 · novelty 검증 · 관련연구 대비 · 데이터/학습 · 전처리 · 통계 · 기준선 · 해석성 · 결론 일관성 · 활용 · 예상질문 — 11개 축 전면 감사 결과", 15, RGBColor(0xBF, 0xD3, 0xEE), False, False, None)],
      [("", 8, WHITE, False, False, None)],
      [("이승상 · 서울대학교 원자핵공학과 · 2026-08-05", 14, WHITE, False, False, None)],
      [("근거: THESIS_RESULTS.md · docs/paper/main_ko.tex · paper_numbers.json(동결 산출물 자동 수집) · docs/paper/NOVELTY.md", 11, RGBColor(0x8F, 0xA8, 0xC8), False, False, None)]],
     space_after=6)
footer(s)

# --- 2. What we claim ------------------------------------------------------
s = slide()
header(s, "① 주장", "우리가 주장하는 것 — 범위가 다른 4개의 문장")
yy = table(
    s, Inches(0.55), Inches(1.45),
    [Inches(0.5), Inches(6.4), Inches(3.1), Inches(2.7)],
    ["", "주장", "적용 범위", "증거"],
    [
        ["C1", ("Tᵢ: 미래를 쓰는 보간(PCHIP)을 유의하게 이긴다 (+0.18~+0.28)", NAVY, True),
         "무작위 분할 · 관측 지점", ("4/4 분할 PASS, 두 모집단", GREEN, True)],
        ["C2", ("두 타겟 모두 모든 인과 기준선을 이긴다 — 배치 주장", NAVY, True),
         "진짜 결측 지점 · 미래 캠페인", ("스트레스 2종 생존", GREEN, True)],
        ["C3", "V_rot: 오프라인 보간과는 동률 (n.s.)", "관측 지점", ("1/4 PASS = 잡음 수준", GRAY, False)],
        ["C4", ("Tᵢ↔V_rot 정보 비대칭은 물리에서 온 발견", NAVY, True),
         "진단 정보 내용", ("예측→ablation→데이터 검증", GREEN, False)],
    ],
    row_h=Inches(0.62), size=13.5)
card(s, Inches(0.55), yy + Inches(0.28), Inches(12.25), Inches(1.55),
     "과대주장 방지 장치 — 두 주장의 분리", [
         "\"미래를 쓰는 보간을 이긴다\"(C1)는 관측 모집단의 주장 — 더 어렵고 유익한 기준이라 headline 유지.",
         "\"모든 인과 방법을 이긴다\"(C2)만이 진짜 결측 재가중(4/4 +0.29)과 캠페인 시간분할(V_rot 4/4 +0.22)을 생존.",
         "온라인 가상 센서의 경쟁 상대는 미래를 읽는 보간이 아니라 persistence — 배치 주장은 C2로 좁힌다.",
     ], accent=ORANGE, body_size=12.5)

# --- 3. Position in the literature -----------------------------------------
s = slide()
header(s, "② 선행 지형과 위치", "활발한 계보 위에 서서, 세 축으로 확장한다")
bullets(s, Inches(0.55), Inches(1.45), Inches(12.2), Inches(1.6), [
    ("진단 재구성·예측은 활발한 분야 — NN-CES 스펙트럼 분석('93 JET~), 교차 진단 추론, 시간 조밀화 계열이 존재하며 전부 인용", 0, None, True),
    ("우리의 주장은 \"이 계보를 3축으로 확장한다\" — \"선행이 없다\"가 아니다 (부재-프레임은 반례 하나에 무너지고, 확장-프레임은 계보를 인정할수록 강해짐)", 1, ORANGE, True),
    ("위치 설정의 근거 = 적대적 탐색 2회 (1차 25+쿼리·OpenAlex / 2차 18쿼리·원문 12건) → 세 확장의 결합을 다룬 연구는 미확인", 1),
    ("논문 문구: \"계열의 프로그램을 이어받아 세 방향으로 확장 … 자연스러운 다음 단계로 제시\" — 관례적 한정(\"우리가 아는 한\")은 한 문장뿐", 1, GRAY),
], size=13.5, gap=5)
cw = Inches(3.95)
for i, (t, lines, acc) in enumerate([
    ("확장 ① — 타겟", ["전자 채널(Thomson/ECE의 Tₑ·nₑ)에서", "희소 이온 채널(CES Tᵢ+V_rot)로", "(선행에서 CES가 등장하면", "어디까지나 입력이었다)"], BLUE),
    ("확장 ② — 매핑", ["동시각·기억 없는 매핑에서", "타겟 자신의 불규칙 과거 이력에", "조건화된 인과 추정으로", "— 온라인 센서가 요구하는 형태"], TEAL),
    ("확장 ③ — 평가", ["가정된 재구성 가능성에서", "사전등록 검정으로: 현직 보간을", "통계 bar로 세움 (V_rot 승리를", "부정한, 음성 가능한 프로토콜)"], ORANGE),
]):
    card(s, Inches(0.55) + i * (cw + Inches(0.2)), Inches(3.35), cw, Inches(2.0),
         t, lines, accent=acc, body_size=12.5)
text(s, Inches(0.55), Inches(5.6), Inches(12.2), Inches(1.2),
     [[("Tᵢ↔V_rot 비대칭은 확장 ③의 검정이 산출한 중심 발견. 가장 가까운 계열과의 논문별 대비는 다음 장.", 13.5, DARK, False, False, None)]])

# --- 4. Related work table -------------------------------------------------
s = slide()
header(s, "③ 관련 연구", "각 논문은 무엇을 보고(입력), 무엇을 예측했나(타겟)")
table(
    s, Inches(0.55), Inches(1.42),
    [Inches(2.9), Inches(5.45), Inches(3.9)],
    ["논문", "입력 → 타겟(예측 결과)", "우리와 다른 점"],
    [
        [("Diag2Diag (NC 2025)", NAVY, True), ("간섭계·ECE·자기·CER(입력!)·MSE → Thomson Tₑ/nₑ", DARK, True), "이온 채널 예측 없음 · memory-less"],
        ["COMPASS 초해상 (2026)", "SXR·AXUV 등 고속 복사+자기 → Thomson Tₑ/nₑ 초해상", "전자 채널 · 비인과 동시각"],
        ["EAST 결측 Tₑ (NF 2025)", "타 진단 시계열(외생 회귀) → 결측 ECE형 Tₑ", "Tₑ 대상 · 인과성 기준 없음"],
        ["FusionMAE (HL-3)", "비마스킹 88ch 신호(윈도) → 마스킹된 임의 채널", "일반 dropout · 윈도 내 비인과"],
        ["RTCAKENN (2024) — 최대 위협", "자기·MSE·간섭계·ECE → Tₑ·nₑ·불순물 Tᵢ·회전 프로파일", "요동 진단·타겟 이력·보간 비교 없음"],
        ["EAST XCS (NF 2024)", "XCS Ar 스펙트럼(동시각) → Tᵢ·V_rot 프로파일", "같은 물리 채널 · 순간 매핑"],
        ["KSTAR ML 프로파일 (2026)", "Thomson+TCI+CES(입력) → Tₑ·nₑ·Tᵢ 공간 프로파일", "공간 피팅 · 회전은 향후 과제"],
        [("본 연구", TEAL, True), ("BES·ECEI·Mirnov + 과거 CES 이력(미래 X) → 결측 시점 CES Tᵢ·V_rot", TEAL, True), ("이온 타겟 · 인과 이력 · 사전등록 검정", TEAL, True)],
    ],
    row_h=Inches(0.48), size=11.5, head_size=11.5,
    emphasis={7}, emphasis_fill=RGBColor(0xE2, 0xF3, 0xEF))
card(s, Inches(0.55), Inches(5.78), Inches(12.25), Inches(1.26),
     "읽는 법 — 계보의 다음 단계로서의 3축 확장", [
         "시간 조밀화 계열의 타겟은 전부 전자 채널(CES/CER은 입력) · 이온 Tᵢ·회전을 내는 연구(RTCAKENN·XCS)는 다른 문제를 푼다.",
         "→ \"희소 CES 이온 측정 자체를 타겟으로 한 인과적 시간 결측 채움\"이 우리가 채우는 칸 (이온 타겟 · 인과 이력 · 사전등록 검정).",
     ], accent=GREEN, body_size=11.5)

# --- 5. Dataset ------------------------------------------------------------
s = slide()
header(s, "④ 데이터셋", "641개 방전 · 10 ms 격자 · 결측의 실태 (전수 조사)")
add_image_fit(s, os.path.join(FIG_P, "fig_missing.png"),
              Inches(6.9), Inches(1.5), Inches(5.9), Inches(4.1))
bullets(s, Inches(0.55), Inches(1.5), Inches(6.1), Inches(4.3), [
    ("KSTAR shot #30801–32751, 641개 CSV · 247,207행", 0),
    ("H-mode ELM 억제(RMP) 구간, D_α 활동 주변 ~100 ms 절단", 1),
    ("BES 9ch · ECEI 4ch · Mirnov 2ch · time · CES_TI · CES_VT", 1),
    ("결측(NaN): Tᵢ 8.2% / V_rot 23.9% — 서로 독립", 0),
    ("V_rot는 추가로 41.1%가 held(직전 값 복사) → 실질 무정보 65.0%", 1, RED),
    ("샘플 = 진단 3종 윈도 + 시간특징 4ch + CES 이력 4ch(값 2+관측 flag 2)", 0),
    ("타겟 시점은 값·flag 완전 마스킹 — 자기 정답 열람 차단", 1),
    ("연속 블록(Δt<0.5 s) 내 temporal-subset 증강 + seeded 표본 상한", 1),
], size=14, gap=6)
card(s, Inches(0.55), Inches(5.85), Inches(12.25), Inches(1.0),
     "행 유지 규칙 (No Fake Data)", [
         "가짜 라벨 대체 금지 — 입력이 완전하고 타겟 ≥1개 실측인 행만 학습. 타겟별 mask 손실로 전환해 \"둘 다 관측\" 필터가 버리던 라벨 행 ≈28%를 회복.",
     ], accent=BLUE, body_size=12.5)

# --- 6. Preprocessing ------------------------------------------------------
s = slide()
header(s, "⑤ 전처리", "누수 방지 3겹 + 데이터 품질 발견 (held)")
cw = Inches(3.95)
for i, (t, lines, acc) in enumerate([
    ("파일(shot) 단위 분할", ["인접 행 자기상관 → 행 단위 분할은 누수", "seeded 3-way split, 디스크 고정", "재로드 시 불일치 검사(예외 발생)"], BLUE),
    ("train-file-only 정규화", ["z-score 통계를 학습 파일에서만 추정", "타겟 통계는 NaN-aware", "(캠페인 이동 시 비용은 §8n에서 실측)"], TEAL),
    ("타겟 시점 완전 마스킹", ["ces_history에서 타겟 행의", "값·관측 flag 모두 0", "보간도 타겟 자신 값 미사용 · ≥0.5 s 불통과"], NAVY),
]):
    card(s, Inches(0.55) + i * (cw + Inches(0.2)), Inches(1.5), cw, Inches(1.75),
         t, lines, accent=acc, body_size=12)
card(s, Inches(0.55), Inches(3.5), Inches(12.25), Inches(1.9),
     "held(유지값) — 관측된 V_rot의 54%가 forward-fill 복사본이었다 (499/641 파일, 최대 1,214행 연속)", [
         "평가: 전 구간 채점 제외(genuine-only가 headline) — V_rot RMSE 22~33 → 35~46으로 정직하게 상승.",
         "학습: 초기 단일-seed A/B \"무해\" 판정을 4-seed paired 재학습이 뒤집음 — held 제거 학습이 V_rot 4/4 개선(평균 +0.039, 3/4 유의).",
         "재확인: window sweep에서 held 제거 이득이 W에 단조 감소(+0.088→+0.006) — held는 짧은 윈도를 벌하고 있었다. 현행 규약: 학습·평가 모두 제거.",
     ], accent=RED, body_size=12.5)
card(s, Inches(0.55), Inches(5.6), Inches(12.25), Inches(1.25),
     "알려진 전처리 결함 (정직 공개)", [
         "Mirnov: kHz dB/dt를 무필터 100 Hz 데시메이션 → 같은 격자에서 lag-1 자기상관 −0.009(BES +0.57, ECEI +0.57 대비) = 백색잡음 실측.",
         "모드 회전 주파수(유일한 회전 대리)가 모델 상류에서 파괴 — 수리는 원 kHz 스트림의 window별 RMS/대역 파워/모드 번호.",
     ], accent=ORANGE, body_size=12.5)

# --- 7. Architecture -------------------------------------------------------
s = slide()
header(s, "④ 모델", "출판 아키텍처 — 201,258 파라미터 후기 융합 (model_iter009)")
add_image_fit(s, os.path.join(FIG, "fig_architecture.png"),
              Inches(0.55), Inches(1.45), Inches(7.4), Inches(5.3))
bullets(s, Inches(8.2), Inches(1.5), Inches(4.7), Inches(5.3), [
    ("진단별 time-aware 1D CNN 인코더 ×3 + 시간특징 CNN — 후기 융합", 0),
    ("이력: 1-layer 양방향 GRU(64) → 타겟별 관측 마스킹 멀티헤드 어텐션", 0),
    ("관측 flag=0인 시점을 softmax 전 −∞ → 어텐션 질량은 실측점에만", 1, RED),
    ("= 보간의 귀납 편향을 파라미터 0개로 주입 — 유일한 비표준 요소, n.s.→유의 전환의 주역", 1),
    ("물리 기반 라우팅: Tᵢ 헤드=고속+시간+이력 / V_rot 헤드=이력+시간만", 0),
    ("윈도는 과거 전용 — BiGRU 양방향성은 이력 내부 문맥화일 뿐 미래 접근 없음", 1, GRAY),
    ("model.py는 model_iter009.py를 재수출 — 기본 import 경로가 곧 출판 아키텍처(SHA-256 고정)", 1, GRAY),
], size=13.5, gap=6)

# --- 8. Training + lineage -------------------------------------------------
s = slide()
header(s, "④ 학습", "학습 절차와 참고한 모델·논문 계보")
bullets(s, Inches(0.55), Inches(1.45), Inches(6.3), Inches(3.3), [
    ("손실: 타겟별 마스킹 MSE + 0.1·relu(정규화-0 이하 Tᵢ 벌점)", 0),
    ("AdamW lr 1e-3 (wd 1e-4) · batch 512 · 10 epoch", 1),
    ("ReduceLROnPlateau(p2, ×0.5) · grad-clip 5.0 · 고정 seed", 1),
    ("데이터 규약: held 제거(CES_DROP_STUCK_TARGETS=1) 명시 고정", 0),
    ("아키텍처 도달: keep/discard 통제 탐색 ~40회", 0),
    ("후보 = 통제 변수 1개 변경 · 채점 = 깨끗한 val skill", 1),
    ("게이트를 val loss→clean skill로 바꾼 것이 n.s.(+0.088)→유의의 결정타", 1, ORANGE),
], size=13.5, gap=6)
table(
    s, Inches(7.1), Inches(1.45),
    [Inches(3.0), Inches(2.75)],
    ["차용한 표준 재료", "참고"],
    [
        ["관측 마스크 순환", "GRU-D (Che 2018)"],
        ["불규칙 시간 어텐션", "mTAN (Shukla 2021)"],
        ["어텐션 풀링", "ABMIL (Ilse 2018)"],
        ["멀티태스크 부분 라벨", "Caruana 1997"],
        ["후기 융합", "Ngiam 2011 외"],
        ["나우캐스팅 프레이밍", "ConvLSTM·DGMR(기상)"],
        ["기준선·지표", "PCHIP · Murphy · IAR"],
    ],
    row_h=Inches(0.42), size=12)
text(s, Inches(0.55), Inches(5.1), Inches(12.2), Inches(1.6),
     [[("새로움은 개별 재료가 아니라 \"물리적으로 불리한 평가 프로토콜 아래의 누수 방지 조합\"에 있다 — 논문 관련연구 절에 명시.", 13.5, NAVY, True, False, None)],
      [("재현성: 분할 디스크 고정 · 동결 실행 디렉터리 · collect_paper_numbers.py → paper_numbers.json → 그림/표 자동 파이프라인 (수기 전사 금지).", 12.5, GRAY, False, False, None)]],
     space_after=6)

# --- 9. Statistics ---------------------------------------------------------
s = slide()
header(s, "⑥ 통계 검증", "\"통계적으로 검증했다\"의 정확한 의미")
card(s, Inches(0.55), Inches(1.45), Inches(12.25), Inches(1.28),
     "한 문장 정의", ["TEST를 보기 전에 고정한 규칙(사전등록) 아래, shot을 재표본 단위로 한 paired bootstrap 95% CI가 0을 제외하는지를,",
                  "모델 선택이 접근하지 않은 4개 독립 분할에서 반복 확인했다."],
     accent=NAVY, body_size=12.5)
cw = Inches(2.95)
for i, (t, lines) in enumerate([
    ("PR1 — 기준선", ["headline = PCHIP", "(사전 선택)", "사다리 전체 병행 보고"]),
    ("PR2 — 모집단", ["보간은 모든 채점 지점에서", "예측(없으면 persistence 후퇴)", "모집단 솎아내기 금지"]),
    ("PR3 — 규모 하한", ["TEST ≥15 shot,", "≥3,000 Tᵢ 표본", "→ 실제 96 shot 충족"]),
    ("PR4 — 유의성", ["shot-군집 bootstrap", "95% CI가 0 제외 = PASS", "B=10,000 · 고정 seed"]),
]):
    card(s, Inches(0.55) + i * (cw + Inches(0.15)), Inches(2.88), cw, Inches(1.6),
         t, lines, accent=BLUE, body_size=12, title_size=13)
bullets(s, Inches(0.55), Inches(4.72), Inches(12.2), Inches(2.1), [
    ("왜 shot 군집인가 — 방전 내 인접 행은 강한 자기상관: 표본 3만 개를 독립 취급하면 CI가 가짜로 좁아진다", 0),
    ("SE_model−SE_pchip을 shot으로 집계 → shot 전체를 복원추출: CI가 shot-to-shot 일반화를 반영, 유효 n≈96의 검정력 비용을 정직하게 지불", 1),
    ("3-way split — TEST는 탐색 전 예약·미열람 · 시드 1/7/123은 선택 밖 재현(winner's curse 차단)", 0),
    ("통과 못한 것도 그대로 보고 — V_rot 1/4=동률 판정, >105 ms PCHIP 승, 캠페인 0/4, linear 대비 시드 42 n.s.", 0, RED),
], size=13.5, gap=6)

# --- 10. Baselines ---------------------------------------------------------
s = slide()
header(s, "⑦ 기준선", "정보 접근이 다른 5개 팔의 사다리 — 전부 같은 표본·같은 mask")
table(
    s, Inches(0.55), Inches(1.42),
    [Inches(2.7), Inches(2.9), Inches(3.4), Inches(3.2)],
    ["팔", "정보 접근", "역할", "근거"],
    [
        ["Persistence", "마지막 관측값", "인과 최소 기준 (실시간의 현직)", "Murphy 1988 관습"],
        ["AR (국소·인과)", "과거 CES만", "인과 시계열 모형 대표", "IAR (Eyheramendy 2018)"],
        ["Linear", "과거+미래 CES", "오프라인 사실상 표준", "Press 2007"],
        [("PCHIP (headline)", NAVY, True), "과거+미래 CES", "단조 3차 — 급변에서 무진동", "Fritsch–Carlson 1980"],
        [("모델 (nowcaster)", TEAL, True), "고속 진단+과거 CES", "검정 대상 — 유일하게 미래 미접근", "—"],
    ],
    row_h=Inches(0.5), size=13)
card(s, Inches(0.55), Inches(4.55), Inches(6.0), Inches(2.15),
     "왜 PCHIP인가 · linear 교차확인", [
         "자연 스플라인은 급변에서 Gibbs 진동 → 단조 3차를 사전 선택(PR1).",
         "사후: 이 데이터에선 linear가 근소 우세(Δt≤15 ms가 97.7%).",
         "→ 더 강한 linear로 바꿔도 결론 유지: hold 4/4, genuine 3/4 PASS",
         "   (시드 42 n.s.까지 명시) — \"기준선 유리 선택\" 반박 차단.",
     ], accent=BLUE, body_size=12.5)
card(s, Inches(6.8), Inches(4.55), Inches(6.0), Inches(2.15),
     "GP 팔 — 사후 실측 완료 (§8p)", [
         "GP가 최강 오프라인 팔: PCHIP에 Tᵢ 4/4 유의 승(+0.21~+0.28)",
         "— 잡음 앵커 통과 대신 평균해 내는 평활기.",
         "모델은 GP와 동률(유의 승 1/4·패 0/4) — 논문에 그대로 보고.",
         "사전등록 headline(PCHIP) 유지 · 배치(인과) 주장 무영향.",
     ], accent=GREEN, body_size=12)

# --- 11. Headline results --------------------------------------------------
s = slide()
header(s, "결과 I", "Headline — Tᵢ는 4개 독립 분할 전부에서 미래를 쓰는 보간을 이긴다")
add_image_fit(s, os.path.join(FIG_P, "fig_forest.png"),
              Inches(0.55), Inches(1.5), Inches(6.9), Inches(5.2))
bullets(s, Inches(7.7), Inches(1.55), Inches(5.1), Inches(5.2), [
    ("Tᵢ genuine: +0.179 / +0.197 / +0.280 / +0.263 — 4/4 PASS", 0, GREEN),
    ("held 포함 모집단에서도 4/4 PASS (+0.19~+0.28) — 데이터 처리에 불변", 1),
    ("V_rot: 점추정 전부 양수지만 PASS는 시드 1 하나 → 동률 보고", 0),
    ("RMSE 사다리(시드 42): 모델 407.0 < linear 441.0 < PCHIP 449.3 < pers. 504.3 < AR 2425.9 (Tᵢ)", 0),
    ("인과 기준선 압도가 가장 강건한 결과 — V_rot도 35.0 < 44.4(pers.)", 1),
    ("정직한 진행: 초기 iter2는 +0.088 n.s. — 선택 게이트 교체가 유의를 만듦(과정 공개)", 0, GRAY),
], size=13.5, gap=7)

# --- 12. Gap + window ------------------------------------------------------
s = slide()
header(s, "결과 II", "간극 층화 · window sweep — 정보 한계의 지도")
add_image_fit(s, os.path.join(FIG, "fig_window_sweep.png"),
              Inches(0.55), Inches(1.5), Inches(6.4), Inches(4.0))
bullets(s, Inches(7.2), Inches(1.55), Inches(5.6), Inches(4.2), [
    ("4분할 통합으로 비인접 영역 첫 측정: Δt>15 ms에서도 Tᵢ가 PCHIP에 +0.191 유의 승", 0, GREEN),
    (">105 ms는 양측 보간의 영역(−0.542) — 단 실시간은 그것을 가질 수 없고, 인과 대비는 >45 ms에서 +0.271 유의 승", 1),
    ("V_rot도 인과 대비 전 구간 유의 승 (≤15 ms +0.368, >15 ms +0.309)", 1),
    ("window sweep 24-run: 관측 1개가 사실상 전부 — history-0은 붕괴(Tᵢ −0.03, V_rot −0.78), 이후 곡선 평탄", 0),
    ("plateau 규칙은 W=2 반환 — \"왜 W=4인가\"에 곡선과 선택 규칙으로 답", 1),
    ("W>2의 유일한 논리는 커버리지: 긴 간극 표본을 4~10배 더 예측", 1),
], size=13.5, gap=7)
text(s, Inches(0.55), Inches(5.75), Inches(6.4), Inches(1.0),
     [[("용량 확장·복잡한 skip·추가 conv 추출기 전부 개선 없음 — 용량·아키텍처가 아니라 정보가 한계라는 진단의 근거.", 12, GRAY, False, False, None)]])

# --- 13. Stress tests ------------------------------------------------------
s = slide()
header(s, "결과 III", "스트레스 테스트 2종 — 어느 주장이 살아남는가", accent=ORANGE)
yy = table(
    s, Inches(0.55), Inches(1.5),
    [Inches(4.6), Inches(3.9), Inches(3.75)],
    ["평가", "vs PCHIP (오프라인·미래 사용)", "vs persistence (인과)"],
    [
        ["무작위 분할 · 관측 지점 (headline)", ("+0.18~+0.28 · 4/4 PASS", GREEN, True), "+0.35~+0.42"],
        ["진짜 결측 지점으로 재가중 (MNAR)", ("+0.06~+0.21 · 1/4", RED, False), ("+0.29 · 4/4 PASS", GREEN, True)],
        ["캠페인 시간 분할 (미래 방전)", ("−0.15~+0.05 · 0/4", RED, False), ("+0.12~+0.28 (V_rot 4/4 PASS)", GREEN, True)],
    ],
    row_h=Inches(0.62), size=13.5)
bullets(s, Inches(0.55), yy + Inches(0.3), Inches(12.2), Inches(2.9), [
    ("결론: 오프라인 보간 대비 우위는 무작위 분할·관측 모집단의 성질 — 배치 주장은 인과 우위로 좁힌다", 0, ORANGE, True),
    ("MNAR 재가중의 부산물 — 적용 범위 사실: W=4에서 진짜 결측 Tᵢ의 54.1%, V_rot의 4.8%만 도메인 내 (커버리지 한계, reach 확장 지목)", 0),
    ("캠페인 손실의 원인을 측정: BES 1.22σ · ECEI 0.53σ 드리프트 vs CES 타겟 0.115σ — 고속 진단 경로가 5~11배 더 이동", 0),
    ("→ 지목된 수리: 고속 진단의 shot별(인과) 표준화 — 누수 없음 · 실행 시점 가용 · 절충은 다음 통제 실험", 1),
    ("V_rot조차 캠페인 분할에서 persistence를 4/4 유의하게 이김 — \"실패한 타겟\"이 아니라 \"의미 있는 비교가 인과뿐인 타겟\"", 0),
], size=13.5, gap=7)

# --- 14. Physics interpretability ------------------------------------------
s = slide()
header(s, "⑧ 해석 가능성", "모델은 물리학적으로 해석 가능한가 — 4겹의 답")
add_image_fit(s, os.path.join(FIG_P, "fig_ablation.png"),
              Inches(0.55), Inches(1.5), Inches(5.6), Inches(3.6))
bullets(s, Inches(6.45), Inches(1.5), Inches(6.35), Inches(3.7), [
    ("① 구조가 물리를 코딩 — V_rot 헤드의 고속 진단 배제(라우팅), 관측 마스킹 어텐션(실측점만)", 0),
    ("② 비대칭의 기전 검증 — Tᵢ: e–i 충돌 결합(fast-only +0.37, Te~Tᵢ r=+0.353) / V_rot: 토크 미관측+aliasing(fast-only −0.64, Te~V_rot r=+0.02)", 0),
    ("③ 복잡도의 가격 측정 — 1,258-파라미터 완전 해석 모델이 Tᵢ 마진의 31.5%(V_rot 7%) 회수 → 나머지는 비선형·비국소 구조", 0),
    ("④ 우위의 위치 특정 — peak에 집중(Tᵢ +0.27→+0.70, V_rot +0.13→+0.44) · #31815 사례로 시각 확인", 0),
], size=13, gap=7)
card(s, Inches(0.55), Inches(5.35), Inches(12.25), Inches(1.4),
     "한계 인정", [
         "어텐션 가중치는 열람 가능하나 채널별 인과 귀속(attribution)은 미수행 — 인코더 내부는 학습된 표현.",
         "그 부족분을 \"해석 가능한 1,258 파라미터가 회수하는 비율\"로 정량화한 것이 우리의 대응 (반박이 아니라 측정).",
     ], accent=GRAY, body_size=12.5)

# --- 15. Conclusion consistency --------------------------------------------
s = slide()
header(s, "⑨ 결론 일관성", "주장 → 증거 → 한계 → 다음 단계가 한 사슬로 닫힌다")
table(
    s, Inches(0.55), Inches(1.45),
    [Inches(3.6), Inches(3.6), Inches(2.6), Inches(2.5)],
    ["결론", "근거", "남는 한계", "지목된 다음 단계"],
    [
        ["Tᵢ, 미래 보간을 이김", "4-seed·2모집단·linear 교차", "MNAR 낙관성", "→ 재가중으로 정량화 완료"],
        ["배치 주장 = 인과 우위", "재가중 4/4 · 캠페인 생존", "오프라인 우위 소멸", "shot별 표준화 실험"],
        ["V_rot 동률·비대칭 발견", "ablation·Te 프로브·MC 자기상관", "회전 원천 부재", "NBI 토크 + 원 Mirnov"],
        ["커버리지 54.1%/4.8%", "MNAR 도메인 산정", "W=4의 도달 범위", "reach 확장(슬롯≠span)"],
        ["복잡도는 값을 함", "1,258-파라미터 사다리", "인코더 내부", "다른 함수 형태"],
        ["실시간 가능", "CPU p99 6.4 ms · conformal 8/8", "커버리지 주변적", "shot-조건부 보정"],
    ],
    row_h=Inches(0.52), size=12)
card(s, Inches(0.55), Inches(5.35), Inches(12.25), Inches(1.5),
     "일관성을 지키는 편집 규칙 2개 (문서화됨)", [
         "① 음성 결과는 그것을 뒤집을 측정을 지목할 때만 보고 — \"정보가 부족하다\"는 종결 금지 (§8j).",
         "② 모든 숫자는 동결 산출물에서 자동 수집 — §8h 감사의 재발 방지 장치 = paper_numbers.json 파이프라인.",
     ], accent=NAVY, body_size=12)

# --- 16. Applications ------------------------------------------------------
s = slide()
header(s, "⑩ 활용", "이 연구는 앞으로 어떻게 쓰이는가")
cw = Inches(5.95)
card(s, Inches(0.55), Inches(1.5), cw, Inches(1.55), "1. 온라인 가상 CES 센서", [
    "CPU p99 6.4 ms < 10 ms 주기 (실측) + conformal 구간 동반.",
    "결측 Tᵢ를 실시간으로 채워 pedestal 분석·제어에 끊김 없는 입력 —",
    "경쟁 상대(persistence) 대비 우위가 결측 지점·미래 캠페인에서 생존.",
], accent=TEAL, body_size=12.5)
card(s, Inches(6.85), Inches(1.5), cw, Inches(1.55), "2. 오프라인 재분석", [
    "아카이브 방전의 CES 간극 고품질 채움 —",
    "관측 모집단에선 미래를 쓰는 보간보다 우수,",
    "과도 현상(ELM·천이) 구간에서 우위 최대(peak).",
], accent=BLUE, body_size=12.5)
card(s, Inches(0.55), Inches(3.25), cw, Inches(1.55), "3. 진단 정보 내용 감사 방법론", [
    "\"이 진단 조합이 저 물리량을 나르는가\"를 검정하는",
    "절차(ablation+사전등록+shot-군집 CI) 자체가 재사용 가능 —",
    "Mirnov 결함·NBI 토크 부재를 데이터로 고발한 것이 시범.",
], accent=ORANGE, body_size=12)
card(s, Inches(6.85), Inches(3.25), cw, Inches(1.55), "4. 평가 프로토콜의 이식", [
    "\"인과 모델 vs 미래 쓰는 현직 보간 + shot-군집 paired bootstrap",
    "+ 사전등록\" — 다른 희소 진단(Thomson·XICS)의 ML 채움 연구가",
    "그대로 가져다 쓸 수 있는 bar.",
], accent=NAVY, body_size=12.5)
card(s, Inches(0.55), Inches(5.0), Inches(12.25), Inches(1.7),
     "5. 명시된 확장 로드맵 (전부 아카이브 데이터로 실행 가능 · 비용 낮은 순)", [
         "이력 reach 확장(슬롯 2–3개 유지, span 확대 — 커버리지 54.1%/4.8% 직접 개선)  →  원 kHz Mirnov window 특징(V_rot 최대 레버)",
         "→  NBI 토크 데이터 확보(회전의 원인 변수)  →  고속 진단 shot별 표준화(캠페인 전이 수리)  →  타 장치(EAST·DIII-D) 이식 검증.",
     ], accent=GREEN, body_size=12.5)

# --- 17. Q&A defense I ------------------------------------------------------
s = slide()
header(s, "⑪ 예상 질문", "방어 가능성 평가 I — 통계·평가 설계", accent=RED)
table(
    s, Inches(0.55), Inches(1.45),
    [Inches(4.6), Inches(6.35), Inches(1.3)],
    ["질문", "답변 요지", "방어"],
    [
        ["관측 지점만 평가 — 진짜 결측에선? (MNAR)", "재가중으로 정량화: 인과 우위 4/4 생존(+0.29), 오프라인은 1/4 → 주장을 인과로 한정", ("●●●", GREEN, True)],
        ["PCHIP이 최선? linear가 더 강한데", "사전등록 사실 + linear 대비 결과(4/4·3/4, 시드42 n.s.)까지 공개", ("●●●", GREEN, True)],
        ["shot 96개 — 검정력 부족?", "인정: 올바른 독립 단위의 비용. V_rot 미해결 원인으로 명시, 다중 캠페인이 해결책", ("●●○", ORANGE, True)],
        ["탐색이 test에 과적합?", "test 사전 예약·미열람, 시드 1/7/123은 선택 밖 재현", ("●●●", GREEN, True)],
        ["미래 쓰는 보간과의 비교가 의미 있나?", "의도적 불리함 — 이기면 시간 구조 밖 정보의 강한 증거. 실시간 주장은 별도로 인과 비교", ("●●●", GREEN, True)],
        ["window=4 근거는?", "24-run sweep으로 우리가 먼저 답: 관측 1개가 전부, plateau 규칙은 W=2, W>2 논리는 커버리지", ("●●●", GREEN, True)],
        ["불확실성은 신뢰 가능?", "conformal 8/8 Winkler 승 — 단 주변적 커버리지(shot별 50~100%)를 스스로 보고", ("●●○", ORANGE, True)],
        ["데이터 54%가 가짜(held)라며?", "우리가 발견·전수 정량화·규약화. 초기 오판을 4-seed paired로 뒤집은 과정도 공개", ("●●●", GREEN, True)],
    ],
    row_h=Inches(0.6), size=12)

# --- 18. Q&A defense II -----------------------------------------------------
s = slide()
header(s, "⑪ 예상 질문", "방어 가능성 평가 II — 물리·일반화·novelty", accent=RED)
table(
    s, Inches(0.55), Inches(1.45),
    [Inches(4.6), Inches(6.35), Inches(1.3)],
    ["질문", "답변 요지", "방어"],
    [
        ["V_rot은 결국 실패 아닌가?", "물리로 사전 예측된 비대칭 · 인과 대비 캠페인 4/4 승 · peak 유의 승", ("●●●", GREEN, True)],
        ["모델이 복잡해서 못 믿겠다", "1,258-파라미터 모델이 31.5%만 회수 — 복잡도의 값을 정량 진술", ("●●●", GREEN, True)],
        ["실시간이 실제로 되나?", "CPU p99 6.4 ms 실측(GPU 8× 느림) · 조립 지연은 별도 인정", ("●●●", GREEN, True)],
        ["Diag2Diag·RTCAKENN과 차이는?", "4축 차별 + RTCAKENN 방어 3종 + 적대 탐색 2회(STANDS ×3)", ("●●●", GREEN, True)],
        ["타 장치·미래 캠페인 일반화?", "캠페인 0/4 자진 보고 + 원인 측정 + 수리 지목 · 타 장치 한계 명시", ("●●○", ORANGE, True)],
        ["Mirnov가 잡음이면 왜 입력에?", "기여 ~0 확인 · 파생 4종 무효 · 원인 실측 · 원 스트림 수리 지목", ("●●○", ORANGE, True)],
        ["GP 기준선은 왜 없나?", "실측 완료: GP=최강 오프라인 팔, 모델과 동률 — 그대로 보고", ("●●●", GREEN, True)],
        ["Tᵢ>3 keV 이상값(fit 실패)?", "민감도 실측: 제거 시 skill ~2배·4/4 유지 — headline은 보수적", ("●●●", GREEN, True)],
        ["반경 방향·ELM 위상은?", "데이터 선택에서 상속 — 한계 명시, 향후 과제", ("●○○", RED, True)],
    ],
    row_h=Inches(0.55), size=12)

# --- 19. Audit findings -----------------------------------------------------
s = slide()
header(s, "⑫ 감사 결과", "이번 루프에서 발견·수정된 것 / 남은 것")
table(
    s, Inches(0.55), Inches(1.45),
    [Inches(7.3), Inches(4.95)],
    ["발견", "조치"],
    [
        ["연구설명_목차별정리.md가 구버전 결론 유지 (held 무해 판정·구 아키텍처·구 headline·2026-08 실험 7종 미반영)", ("전면 재작성 완료", GREEN, True)],
        ["기존 발표덱 4종이 논문 대비 구세대 (held 포함 headline · 스트레스 테스트 미반영)", ("4종 전부 재빌드 완료 (figures는 paper_numbers.json 판독)", GREEN, True)],
        ["문헌 검증이 2026-07-03 시점 고정", ("재검증 완료: STANDS ×3 · 논문에 인용 3건 반영", GREEN, True)],
        ["논문·THESIS_RESULTS·paper_numbers.json 수치 대조", ("불일치 없음 확인", GREEN, True)],
        ["GP 기준선 부재 + fit-failure 영향 미측정", ("둘 다 실측 완료: GP=최강 팔·모델 동률 / 제거 시 skill ~2배 (§8p·§8q)", GREEN, True)],
    ],
    row_h=Inches(0.62), size=12.5)
card(s, Inches(0.55), Inches(5.35), Inches(12.25), Inches(1.4),
     "잔여 미해결 (다음 루프 후보)", [
         "캠페인 전이 수리(shot별 표준화) 통제 실험 — 정규화 계약 변경이라 승인 필요 · 인과 과거-전용 GP 팔(동률 해소책).",
         "상세는 docs/연구방어_종합문서.md §12. (GP 팔·fit-failure 민감도는 실행 완료 — §8p·§8q.)",
     ], accent=ORANGE, body_size=12.5)

# --- 20. One-page summary ---------------------------------------------------
s = slide()
box(s, 0, 0, EMU_W, EMU_H, fill=NAVY)
text(s, Inches(0.9), Inches(0.7), Inches(11.5), Inches(0.6),
     [[("한 장 요약", 28, WHITE, True, False, None)]])
for i, (t, body, acc) in enumerate([
    ("주장 (분리된 2개)", "Tᵢ는 미래를 쓰는 보간을 4/4 분할에서 유의하게 이긴다(관측 모집단). 배치 주장은 \"모든 인과 방법을 이긴다\" — 진짜 결측·미래 캠페인을 생존하는 것은 이것뿐(+0.29/+0.22).", TEAL),
    ("발견", "Tᵢ↔V_rot 정보 비대칭 — 고속 진단은 10 ms에서 Tᵢ 정보를 나르지만(e–i 결합) 회전 정보는 없다(NBI 토크 미관측·Mirnov aliasing 실측). V_rot 동률은 실패가 아니라 물리 발견.", ORANGE),
    ("방법의 기여", "사전등록 + shot-군집 paired bootstrap + \"현직 보간을 이겨라\" bar + 데이터 품질 감사(held 54%) + 관측 마스킹 어텐션 + 복잡도 사다리(31.5%) — 전부 이식 가능.", BLUE),
    ("정직성", "V_rot 동률·캠페인 0/4·>105 ms 열세·커버리지 54.1%/4.8%·주변적 커버리지 — 전부 본문 보고, 각각 뒤집을 측정을 지목(reach 확장·원 Mirnov·NBI 토크·shot별 표준화).", GREEN),
]):
    y0 = Inches(1.55) + i * Inches(1.32)
    box(s, Inches(0.9), y0, Inches(0.12), Inches(1.12), fill=acc)
    text(s, Inches(1.25), y0, Inches(11.2), Inches(1.2),
         [[(t, 15, acc, True, False, None)],
          [(body, 13, WHITE, False, False, None)]], space_after=3, line_spacing=1.12)
footer(s)

# ===================== SPEAKER NOTES (one per slide, build order) ==========
# 각 슬라이드의 근거·수치·예상질문 대비를 발표자 노트로 주입한다. 본문은 요약,
# 상세는 전부 여기. 원본: docs/연구방어_종합문서.md + THESIS_RESULTS.md §8.
NOTES = [
# 1. 표지
"""이 덱은 docs/연구방어_종합문서.md(11축 전면 감사, 2026-08-05)의 발표판입니다.
모든 수치의 출처: THESIS_RESULTS.md §8(실험 원장) · docs/paper/paper_numbers.json(동결
산출물 자동 수집) · data/.gp_analysis.json · data/.fitfail_analysis.json ·
docs/paper/NOVELTY.md(문헌 검증 2회).

시간이 2분뿐이면: 2번(주장) → 13번(스트레스 테스트) → 20번(요약)만 보여줍니다.
심사 대비 핵심 슬라이드는 17·18번(예상질문 17문항 방어표)입니다.""",

# 2. 주장 C1-C4
"""[정확한 수치]
C1 (Tᵢ vs 사전등록 보간): genuine skill_vs_pchip = +0.179 [+0.007,+0.302] /
+0.197 [+0.091,+0.276] / +0.280 [+0.192,+0.347] / +0.263 [+0.109,+0.348] — 4/4 PASS.
held 포함 모집단에서도 +0.19~+0.28 4/4. 더 강한 linear 대비 genuine 3/4 PASS
(시드 42만 +0.148 n.s. — 이것도 논문에 명시).
C2 (인과 우위·배치): 진짜 결측 재가중 후 vs persistence +0.292/+0.308/+0.293/+0.293
(4/4 PASS, MNAR 보정 비용 0.06~0.12뿐) · 캠페인 시간 분할에서 Tᵢ 3/4(평균 +0.222),
V_rot 4/4(평균 +0.216).
C3 (V_rot): 점추정 +0.100~+0.203 전부 양수지만 PASS는 시드 1(+0.162)뿐 = 잡음 수준.
C4 (비대칭): fast-only Tᵢ +0.372(승) vs fast-only V_rot −0.642(평균 예측보다 나쁨).

[C1의 정밀화 — 반드시 이 형태로 말할 것]
"미래를 쓰는 보간을 이긴다" = "사전등록된 보간(PCHIP·linear)을 이긴다". 사후 실측된
최강 오프라인 팔(GP)과는 동률(유의 승 1/4·패 0/4)이며 이는 논문에 그대로 보고됨.
배치 주장(C2)은 무영향 — 미래 앵커 없는 GP는 온라인에 존재하지 않는다.

[왜 두 주장을 분리하는가]
오프라인 우위는 무작위 분할·관측 모집단의 성질(스트레스 2종 모두 탈락: 1/4, 0/4).
인과 우위만 결측 지점·미래 캠페인을 생존. 온라인 가상 센서의 실제 경쟁 상대는
persistence다. 이 둘을 뭉개는 것이 과대판매의 주된 경로.
출처: THESIS §8h·§8i·§8n·§8p / 방어문서 §1.""",

# 3. 선행 지형과 위치 (3축 확장)
"""[표현 원칙 — 승상님 2026-08-05]
"관련 논문이 없다"를 앞세우지 않는다. 부재-프레임은 반례 하나에 무너지고 심사자에게
문헌 이해 부족으로 읽힌다. 확장-프레임은 계보를 인정할수록 강해진다.
말하는 순서: ① 계보 인정(NN-CES '93 JET~, 교차 진단 추론, 시간 조밀화) →
② 3축 확장 → ③ 관례적 한정 한 문장("우리가 아는 한 이 결합은 아직 다뤄지지 않았으며,
계열의 자연스러운 다음 단계로 제시한다" — 논문 관련연구 절 마지막 문장 그대로).

[3축 확장의 정확한 내용]
① 타겟: Thomson/ECE 전자 물리량(Tₑ·nₑ) → 희소 CES 이온 물리량(Tᵢ·V_rot).
  선행에서 CES/CER이 등장하면 어디까지나 입력이었다.
② 매핑: 동시각·기억 없는 매핑(Diag2Diag는 명시적 memory-less) → 타겟 자신의 불규칙
  과거 이력+관측 플래그에 조건화된 인과 추정(미래 절대 안 읽음).
③ 평가: 재구성 가능성을 가정 → 현직 방법(타겟 채널의 미래-사용 보간)을 통계 bar로
  세우는 사전등록 검정. 이 프로토콜이 음성 결과를 낼 수 있다는 것 자체가 기여
  (실제로 V_rot 승리를 부정했다). Tᵢ↔V_rot 비대칭은 이 검정의 산출물.

[검증 방법]
1차 2026-07-03: 병렬 4에이전트, 25+ 쿼리, OpenAlex 서지 검증 → BORDERLINE-POSITIVE.
2차 2026-08-05: 18개 고유 쿼리(한국어 포함) + 원문 12건 → 판정 유지, 서지 정정 1건,
신규 인용 3건(Kim NF 2024, Char arXiv 2024, Jung 차별 문구).
출처: NOVELTY.md(상단 표현 규칙 포함) / 방어문서 §2.""",

# 4. 관련 연구 입력→타겟 표
"""[정식 서지 + 입력→타겟 상세]
· Diag2Diag = Jalalvand et al., "Multimodal super-resolution: discovering hidden
  physics and its application to fusion plasmas", Nat. Commun. 16, 8506 (2025), DIII-D.
  입력: 간섭계·ECE·자기·CER·MSE(전부 동시각). 타겟: Thomson Tₑ/nₑ 하나.
  원문 인용: "all the included diagnostics are aligned with the TS sampling time
  steps" — CER은 입력이지 초해상 대상이 아님. 모델은 명시적 memory-less.
  ※ 주의: 이 논문 PDF 추출 경로로 읽으면 환각이 남 — HTML 본문으로 확인할 것.
· Imríšek et al., PPCF 68, 065049 (2026), COMPASS. SXR·AXUV+간섭계·분광·자기 →
  Thomson Tₑ/nₑ μs급 시간 초해상.
· Wang et al., "Time series extrinsic regression...", NF 65, 076008 (2025), EAST.
  타 진단 시계열 → 결측 ECE형 Tₑ (고장 복구 프레이밍).
· FusionMAE = Yang et al., Comm. Phys. (2026), HL-3. 88ch×1 kHz 비마스킹 → 마스킹
  채널 (Transformer MAE, 10 ms 윈도).
· RTCAKENN = Shousha et al., NF 64, 026006 (2024), DIII-D. 자기·MSE·간섭계·ECE →
  Tₑ·nₑ·불순물 Tᵢ·회전 프로파일(실시간). CER/TS 소실에 dropout 강건.
  [최대 위협 — 방어 3종] (a) 입력이 BES/ECEI/Mirnov 요동 진단이 아님;
  (b) 과거 타겟 이력 채널 없음(불규칙 CES 이력+관측 플래그 구성은 조사 범위에서 유일);
  (c) 보간 기준선과 통계 비교 없음.
· Lin et al., NF 64, 106061 (2024), EAST. XCS(X선 결정 분광기 — XICS 아님) Ar XVII
  101픽셀 스펙트럼 → Tᵢ·V_rot 프로파일 (Voigt 피팅 ~10배 가속). 같은 물리 채널의
  다른 계기·순간 매핑 → 소스 진단 가용성 상속.
· Jung, Kim & Kang, JKPS 88, 1079 (2026), KSTAR. Thomson 31ch+TCI 5ch+CES 32ch(입력)
  → Tₑ·nₑ·Tᵢ EPED 공간 프로파일. 회전은 저자들이 명시한 향후 과제 — 동일 장치에서
  회전 미착수를 공식 문헌이 인정(우리 차별점 강화).
계보(피팅 가속): Bishop & Roach & von Hellermann PPCF 35, 765 (1993) → Svensson &
von Hellermann PPCF 41, 315 (1999) → Chai et al. PST 21, 105103 (2019).
출처: NOVELTY.md 2026-08-05 절 / 방어문서 §3.2.""",

# 5. 데이터셋
"""[데이터]
KSTAR shot #30801–32751 연속 641개 CSV, 247,207행, 공통 10 ms 격자. H-mode RMP ELM
억제 구간을 D_α 활동 중심 ~100 ms(파일은 shot당 ~30 s 구간)로 절단. 하드웨어 이력
('17 MicroTCA, '20 UFCES, '23 W-divertor)을 고려한 캠페인 범위 #24000–33000에서 선정.
열: BES_* 9ch(밀도요동) · ECEI_* 4ch(전자온도) · MC* 2ch(자기요동) · time ·
CES_TI · CES_VT — 접두사로 자동 인식.

[결측 실태 — 전수 조사]
NaN: Tᵢ 8.2% / V_rot 23.9%, 서로 독립(한쪽만 있는 행이 흔함).
V_rot는 추가로 41.1%가 held(직전 관측의 bit-동일 복사) → 실질 무정보 65.0%.
관측된 V_rot 중 54.0%가 held. Tᵢ의 held는 641 shot 전체에서 1행(0.0%).
"V_rot 결측 24%"만 말하면 실태를 과소평가 — 반드시 65%와 함께.

[샘플 구성 (W=4)]
bes(4,9)+ecei(4,4)+mc(4,2)+time_features(4,4)+ces_history(4,4) → target(2)+mask(2).
time_features = lookback초·delta초·각각의 log1p — 불규칙 간격을 명시 입력으로.
ces_history = 이전 정규화 Tᵢ·V_rot + 타겟별 관측 flag 2개. 타겟 시점은 값·flag 완전
마스킹. 연속 블록(행간 Δt<0.5 s) 안에서 temporal-subset 증강(조합 폭발 → seeded 상한:
train 20만; window 비교 실험은 shot당 500 추가).
행 유지: 진단 입력 완전 + 타겟 ≥1개 실측 (구 "둘 다 관측" 필터는 라벨 행 ≈28%를
조용히 폐기 — per-target mask로 회복).
출처: 방어문서 §4.1·§4.2 / 논문 §데이터.""",

# 6. 전처리
"""[누수 방지 3겹]
① 파일(shot) 단위 3-way 분할, seeded, 디스크 고정 + 재로드 검증(불일치 시 예외).
② train-file-only 채널별 z-score(타겟 통계는 NaN-aware). — 캠페인 분할이 이 선택의
  비용을 실측(§8n): 무작위 분할에선 올바른 누수 방지, 시간 이동에선 깨지는 지점.
③ ces_history의 타겟 시점 완전 마스킹(값+flag). 보간 기준선도 타겟 자신 값 불독,
  ≥0.5 s 세그먼트 경계 불통과(persistence 후퇴, 모집단 축소 없음).

[held 스토리 — 신뢰를 얻는 지점, 방어적으로 말하지 말 것]
발견: 관측된 V_rot의 54%가 forward-fill(499/641 파일, 최대 1,214행 연속).
평가: 전 구간 제외(genuine-only가 headline) → V_rot RMSE 22~33 → 35~46으로 정직 상승.
학습: 초기 단일-seed A/B는 "무해" → 4-seed paired가 뒤집음: held 제거 학습이 V_rot
4/4 개선(평균 +0.039, 3/4 유의). forward-fill이 "이력 복사가 최적"이라고 가르치고
있었다. 재확인: window sweep에서 held 제거 이득이 W에 단조 감소(+0.088→+0.006) —
held는 짧은 윈도를 벌하고 있었다. 현행 규약: 학습·평가 모두 제거(CES_DROP_STUCK_TARGETS=1).

[Mirnov 결함 — 정직 공개 + 수리 지목]
같은 100 Hz 격자에서 블록 내 lag-1 자기상관: BES +0.568 / ECEI +0.572 / MC −0.009
(블록 82%가 |r|<0.1) = kHz dB/dt를 무필터 데시메이션한 백색잡음. 파생 특징 4종
(적분·PCHIP적분·|MC|·이동RMS) 전부 무효(4-seed paired). 수리는 상류: 원 kHz 스트림의
window별 RMS·대역 파워·모드 번호.
출처: THESIS §1·§8b.2·§8c·§8f / 방어문서 §5.""",

# 7. 모델
"""[아키텍처 상세 — 201,258 파라미터]
· 진단별 time-aware 1D CNN 인코더 ×3 (Conv–BN–GELU ×2 → GAP → 96-d 사영) + 시간
  특징 소형 CNN. 순환 대신 CNN인 이유: 결측 제거로 시간축이 불규칙 — 등간격 가정 회피.
· 이력 인코더: 1-layer 양방향 GRU(hidden 64) → 타겟별 독립 관측 마스킹 멀티헤드(4)
  가산 어텐션 풀링 2개. softmax 이전에 해당 타겟의 관측 flag=0인 시점(완전 마스킹된
  타겟 행 + 미관측 이력 행)을 −∞로 밀어냄 → 어텐션 질량은 실측점에만.
  이것이 "관측점만 쓴다"는 보간의 귀납 편향을 파라미터 0개로 주입한, 이 모델의
  유일한 비표준 구성요소이자 n.s.(+0.088) → 유의 전환의 주역.
· 양방향성 주의: 윈도가 엄격히 과거 전용이므로 미래 접근이 아님 — 이력 구간 내부의
  앞뒤 문맥화일 뿐. 마스킹된 타겟이 윈도 중앙에 있어 forward+backward가 양쪽 관측
  이웃을 취합 → 구조적으로 '학습된 보간'처럼 동작.
· 물리 기반 라우팅: Tᵢ 헤드 = 고속+시간+이력(72k) / V_rot 헤드 = 이력+시간만(14k,
  고속 진단 의도적 배제 — ablation 근거). 출력은 정규화 단위(역정규화는 평가만).

[체크포인트 함정 — 질문 대비]
논문 숫자를 만든 아키텍처는 model_iter009.py 하나뿐이고 SHA-256으로 고정돼 있다.
한때 model.py가 AutoML이 Transformer로 재작성한 별개 파일이라 저장 체크포인트 45개를
하나도 못 불러왔고, 채점할 때마다 iter009를 주입해야 했다 — 2026-08-09에 model.py가
iter009를 재수출하도록 배선을 정리해 해소. 변형 아키텍처는 CES_MODEL_FILE로 지정한다.
재학습으로 양 타겟 재현 확인(2026-07-14 .vt_repro_* 계열).
출처: PROJECT_KNOWLEDGE "Checkpoint Provenance" / 방어문서 §4.4.""",

# 8. 학습
"""[학습 절차]
손실 = 타겟별 마스킹 MSE Σm(ŷ−y)²/Σm + 0.1·relu(정규화0 − ŷ_Tᵢ) 물리 soft 벌점.
AdamW(lr 1e-3, wd 1e-4) · batch 512 · 10 epoch · ReduceLROnPlateau(patience 2, ×0.5)
· grad-clip 5.0 · 고정 seed · loss 비유한 시 즉시 실패(fail-loud).
데이터 규약: held 제거를 모든 실행에서 명시 고정 — 기본값 상속이 §8f의 잘못된 결론
(V_rot은 긴 윈도 필요)을 한 번 만든 적이 있음.

[아키텍처 도달 경로 — keep/discard 통제 탐색 ~40회]
후보 = 데이터 계약 보존 + 통제 변수 1개 변경 → 완전 학습 → 깨끗한 비증강 검증
skill_vs_pchip으로 채점 → 최고 갱신 시 KEEP, 아니면 롤백. test는 전 과정 봉인.
게이트 교체가 결정타: 증강 val loss는 이력 촘촘한 쉬운 구간이 복제돼 '평활한 예측'을
보상 — 평활한 영역은 보간이 이미 잘하는 곳. clean skill로 직접 채점하면 선택 기준이
보고 지표와 일치, 보간이 약한 곳(고변동·gap)에서 이기는 구조가 선택됨.

[계보 — "무엇을 참고했나" 질문 대비]
GRU-D(Che 2018, 관측 마스크 순환) · mTAN(Shukla 2021) · Latent-ODE(Rubanova 2019,
시험 후 기각) · ABMIL(Ilse 2018, 어텐션 풀링) · Caruana 1997(부분 라벨 멀티태스크) ·
Ngiam 2011(후기 융합) · Xiong 2020(Pre-LN) · ConvLSTM/DGMR(나우캐스팅 용법).
새로움은 재료가 아니라 "물리적으로 불리한 평가 프로토콜 아래의 누수 방지 조합".
재현성: collect_paper_numbers.py → paper_numbers.json → 그림/표 (수기 전사 금지).
출처: 방어문서 §4.3·§4.5·§4.6.""",

# 9. 통계 검증
"""[한 문장 정의를 풀어서]
"TEST 숫자를 보기 전에 규칙을 문서로 고정했고(PR1–4), 방전(shot)을 재표본 단위로
하는 paired bootstrap 95% CI가 0을 제외하는지를, 모델 선택이 접근한 적 없는 4개
독립 분할에서 반복 확인했다."

[왜 shot이 단위인가 — 가장 자주 나올 질문]
한 방전 내 인접 CES 행은 강하게 자기상관 → 표본 3만 개를 독립 취급하면 CI가
터무니없이 좁아짐(가짜 복제). SE_model−SE_pchip을 shot으로 집계해 shot 전체를
복원추출(B=10,000, 고정 seed 12345). 유효 n이 표본 수가 아니라 shot 수(~96)가 되는
검정력 비용을 정직하게 지불 — 이것이 V_rot 미해결의 구속 조건.
paired인 이유: shot 간 난이도 분산이 소거돼 비교가 예리해짐.

[PR 세부]
PR1 headline=PCHIP(사전 선택; 사다리 전체 병행). PR2 보간은 모든 채점 지점에서 예측
(미래 이웃 없으면 persistence 후퇴 — 모집단 솎아내기 금지). PR3 TEST ≥15 shot &
≥3,000 Tᵢ 표본(실제 96). PR4 PASS ⟺ 95% CI 하한 > 0.
선택 편향 차단: test는 탐색 전 예약·미열람, 시드 1/7/123은 선택 밖 재현.

[통과 못한 것 목록 — 먼저 말하면 신뢰를 얻음]
V_rot 1/4(동률 판정) · linear 대비 시드 42 n.s. · Δt>105 ms에서 PCHIP 유의 승 ·
캠페인 분할 vs PCHIP 0/4 · 모델 vs GP 동률(1/4).
출처: 방어문서 §6 / THESIS §3.""",

# 10. 기준선
"""[사다리 — 정보 접근·근거]
persistence(마지막 관측; Murphy 1988 skill 분모 관습) · AR 국소(과거만; IAR —
Eyheramendy 2018) · linear(과거+미래; Press 2007) · PCHIP(과거+미래 단조 3차;
Fritsch–Carlson 1980) · 모델(고속+과거 CES — 유일하게 미래 미접근).
PCHIP 사전 선택 이유: 자연 스플라인은 급변에서 Gibbs형 진동(Kvernadze 1998).
linear 교차확인(genuine): +0.148 n.s.(42) / +0.167 / +0.259 / +0.234 PASS —
"기준선을 유리하게 골랐다"는 반박을 시드 42 n.s.까지 공개하며 차단.
공정성: 전 팔 동일 (file,row)·동일 per-target mask, 보간은 타겟 자신 값 불독,
≥0.5 s 간극 불통과, Δt≤15 ms가 채점의 97.7%.

[GP 팔 — 2026-08-05 사후 실측, §8p]
구현: 정확한(exact) Matern-3/2+백색잡음 GP, 최근접 과거16+미래16 국소 적합, 이웃집합
내 표준화, 초모수는 격자(길이척도 5–80 ms × 잡음 1e-4–1e-1)에서 표본별 주변우도 최대
선택 — 완전 결정론적, 0.94 ms/fit. (sklearn 원안은 블록 전체 적합 38 ms/fit = 실행
불가로 판명되어 교체.) 무누수·PR2 fallback 동일. NaN 조건이 ar_local과 동일 → 모집단
불변이 구조적으로 보장, 4-seed 재채점 전부 기존 14키 bit-identical.
결과: GP vs PCHIP(Tᵢ) +0.242/+0.216/+0.206/+0.278 — 4/4 PASS, 최강 오프라인 팔.
model vs GP(Tᵢ) −0.083/−0.024/+0.094*/−0.021 — 유의 승 1/4·패 0/4 = 동률.
V_rot: GP vs PCHIP 3/4 PASS, model vs GP 전부 n.s.
기전: PCHIP은 잡음 앵커를 정확히 통과, GP는 잡음을 평균해 냄(linear>PCHIP의 연장).
말할 것: "기준선이 약했다는 비판을 측정으로 선제 차단했고, 결과를 그대로 보고한다."
출처: data/.gp_analysis.json / THESIS §8p / 방어문서 §7.""",

# 11. 결과 I headline
"""[정확한 수치 (genuine)]
Tᵢ skill_vs_pchip: 42 +0.179 [+0.007,+0.302] / 1 +0.197 [+0.091,+0.276] /
7 +0.280 [+0.192,+0.347] / 123 +0.263 [+0.109,+0.348] — 전부 PASS.
V_rot: +0.203 n.s. / +0.162 PASS / +0.100 n.s. / +0.183 n.s.
RMSE 사다리(시드 42, genuine): Tᵢ 모델 407.0 < linear 441.0 < PCHIP 449.3 <
persistence 504.3 < AR 2425.9 (eV). V_rot 35.0 < 38.4 < 39.2 < 44.4 < 91.5 (원 CSV 물리 단위).
채점 모집단: 33,693 샘플/96 shot — Tᵢ n=32,787, V_rot n=10,729(61 shot; held 제외).

[정직한 진행 스토리]
초기 iter2는 단일 분할 +0.088 [−0.221,+0.323] n.s. → 선택 게이트를 val loss에서
clean skill로 교체 → iter009가 4/4 유의. n.s. 결과도 버전과 함께 보고.

[fit-failure 민감도 — §8q, "이상값이 결과 부풀리지 않았나" 대비]
Tᵢ>3 keV(분광 적합실패, 행의 0.4~0.6%)를 모든 팔에서 동일 제거하면 4/4 PASS 유지 +
skill이 약 2배: +0.362/+0.470/+0.589/+0.538. 스파이크는 어느 팔에도 예측 불가 →
MSE 비율을 1로 끌어당김. 즉 아티팩트는 headline을 끌어내리는 쪽 — 사전등록 모집단을
유지한 headline은 보수적. (더 엄격한 p99=2,089 eV 컷은 3/4 PASS, 시드 42 경계.)
출처: paper_numbers.json / data/.fitfail_analysis.json / THESIS §8q.""",

# 12. 결과 II gap·window
"""[간극 층화 — 4분할 통합, 물리적 shot 군집, genuine (§8g)]
Tᵢ vs PCHIP | vs persistence:
≤15 ms(n=134,629): +0.262 PASS | +0.407 PASS
(15,25]: +0.183 PASS | +0.384 PASS · (25,35]: +0.402 PASS | +0.564 PASS
(55,105]: +0.282 PASS | +0.287 PASS · >105 ms(n=167): −0.542(PCHIP 승) | +0.266 n.s.
전체 >15 ms(n=4,496): +0.191 PASS | +0.388 PASS · 전체 >45 ms: −0.057 n.s. | +0.271 PASS
V_rot vs persistence: ≤15 ms +0.368 · >15 ms +0.309 — 전 구간 유의 승.
말하기: "비인접 영역에서도 미래를 쓰는 PCHIP을 이긴다 — 이전 초안은 검정력이 없어
못 하던 주장. >105 ms는 양측 보간의 영역이고 실시간엔 그것이 없다."

[window sweep — 24 run + history-0 ×4 (§8f)]
history-0: Tᵢ −0.026(보간 이하)·V_rot −0.783(PCHIP MSE의 1.8배) — 고속 진단만으로는
보간 동률에도 못 미침. 관측 1개(W=2): Tᵢ +0.238 4/4·V_rot +0.206 — 즉시 최대치.
이후 평탄(Tᵢ 0.190–0.246, V_rot 0.190–0.206; 한 점 안의 seed 산포 0.07–0.16이 곡선
전체 폭보다 넓음). plateau 규칙 → W=2. W=4는 경험적 정당성 없음(우리가 먼저 보고).
W>2의 유일한 논리 = 커버리지: Δt>15 ms 채점 456→1,958, >45 ms 14→135 (W=2→8).
검증된 통제 3종: 분할 window-불변(seed당 동일 96 test shot), shot당 상한 500,
채점 모집단 1.8%만 감소. held-free 필수(held는 짧은 윈도를 벌함).
출처: THESIS §8f·§8g / data/.largegap_analysis.json.""",

# 13. 결과 III 스트레스
"""[스트레스 1 — 진짜 결측 지점 재가중 (MNAR, §8i)]
방법: 관측·결측 양쪽에서 계산 가능한 공변량(마지막 진짜 관측 후 Δt × 입력 전용 국소
활동도)으로 사후 층화, <30표본 층 폐기, 층 커버리지 95–100% 보고. 가정(층 내 교환
가능성)은 명시.
적용 범위 사실: W=4에서 진짜 결측 Tᵢ의 54.1%·V_rot의 4.8%만 도메인 내(윈도 안에
관측값 존재) — 정확도가 아니라 커버리지 한계 → reach 확장 지목.
결과: Tᵢ vs persistence +0.292/+0.308/+0.293/+0.293 — 4/4 PASS, 낙관 편향 0.06~0.12.
vs PCHIP은 1/4만 유의 — 결측 지점은 큰 Δt에 있고 거기가 양측 앵커의 최대 이점 지점.

[스트레스 2 — 캠페인 시간 분할 (§8n)]
641 방전을 shot 번호로 정렬해 시간 절단: train 416 [30801,31991] / val 128 / test 97
[32312,32751] — 어떤 test도 train보다 앞서지 않음. 분할 고정, 초기화 시드만 4개
(재현은 초기화에 대한 것임을 그대로 밝힘).
Tᵢ vs PCHIP 0/4(−0.148~+0.051) | vs persistence 3/4 PASS(평균 +0.222).
V_rot vs persistence 4/4 PASS(평균 +0.216) — 회전은 "실패한 타겟"이 아니라
"의미 있는 비교가 인과뿐인 타겟".
원인 실측: train→test 드리프트 z = BES 1.218σ(스케일 0.75) / ECEI 0.529σ(0.62) /
MC 0.007(제로평균 잡음) / CES 타겟 0.115σ(1.06). 보간 대비 우위를 사는 고속 진단
경로가 5~11배 더 이동 — train-file-only 정규화가 깨지는 지점.
수리 지목: 고속 진단의 shot별(인과) 표준화 — 누수 없음·실행 시점 가용. 절대 수준
정보(BES↔밀도) 상실이 Tᵢ를 해칠 수 있어 통제 실험 대상(가정 아님).
출처: THESIS §8i·§8n / data/.mnar_analysis.json·.campaign_*.""",

# 14. 해석 가능성
"""[4겹 답의 상세]
① 구조가 물리를 코딩: V_rot 헤드에서 고속 진단 배제(라우팅), 관측 마스킹 어텐션
  (실측점만) — 어텐션 질량 위치는 직접 열람 가능.
② 비대칭의 기전 검증:
  Tᵢ 경로 = 충돌성 e–i 결합, t_ei ∝ Tₑ^1.5/nₑ → ECEI(Tₑ)+BES(nₑ)가 정보 운반.
  검증: fast-only +0.372(persistence 승) · shot 간 Tₑ~Tᵢ r=+0.353(p=2.9e−17) ·
  peak에서 fast 제거 시 Tᵢ 유의 붕괴(paired diff CI≫0).
  V_rot 경로 부재 = NBI 토크 미관측 + Mirnov aliasing(lag-1 −0.009 실측).
  검증: fast-only −0.642(평균 예측보다 나쁨) · Tₑ~V_rot r=+0.024(p=0.58 — 파워는
  토크가 아니다; 토크는 빔 에너지·접선 반경·기하 의존) · peak에서 fast 제거해도 불변.
  양성 대조군(Char et al. arXiv:2404.12416): NBI 액추에이터 입력 시 회전 예측 가능
  → "토크가 들어오면 학습 가능"이라는 반증 가능한 형태.
③ 복잡도의 가격(§8k): 완전 해석 가능한 앵커+Δ 모델(1,258 파라미터 = 0.6%; 앵커+
  기울기×갭+진단별 변화율×갭; 항별 분해; persistence에서 정확히 출발) —
  persistence −0.272 → 앵커 −0.113 → 전체 +0.234 (Tᵢ 평균). 회수율 31.5%(28~36%
  4분할 일관) / V_rot 7%. 남은 것은 비선형·비국소 구조라고 정량 진술.
④ 위치 특정: peak(입력 전용 proxy) Tᵢ +0.27→+0.702 [+0.503,+0.851] ·
  V_rot +0.13→+0.438 [+0.068,+0.726] (val, 낙관 주의). #31815 사례: 모델은 과도 추적,
  PCHIP은 overshoot (Tᵢ +0.36, peak +0.63).
한계 인정: 채널별 인과 귀속은 미수행 — 부족분을 사다리로 정량화한 것이 우리의 대응.
출처: THESIS §5·§8b·§8k / 방어문서 §8.""",

# 15. 결론 일관성
"""[사슬의 각 행 근거]
행1: 4-seed genuine+held 이중 보고 + linear 교차(시드 42 n.s. 공개) + §8g 비인접 확장.
행2: §8i 재가중 4/4 + §8n 캠페인 — 이 두 독립 스트레스가 같은 답(인과만 생존).
행3: ablation 2계열 + Tₑ 프로브 + MC 자기상관 → NBI·원 Mirnov 지목.
행4: §8i 도메인 산정(54.1%/4.8%) → reach≠depth 확장 지목(§8f가 W는 커버리지만 삼을
  보임 — dataset.py의 미사용 max_window_span이 바로 그 자리).
행5: §8k 사다리 31.5%/7% → 다른 함수 형태 지목.
행6: §8l p99 6.4 ms + §8m conformal 8/8 → shot-조건부 보정은 현 shot 수로 불가 명시.

[편집 규칙 2개의 배경 — 왜 생겼나]
§8j: 초안이 "정보가 부족하다"를 종결 주장으로 쓰다가 교정됨(승상님) — 음성 결과는
그것을 뒤집을 측정을 지목할 때만 보고.
§8h: 감사에서 논문이 다른 아키텍처를 설명하고 구세대 체크포인트 숫자를 인용하던 것을
적발 — 원인은 수기 전사. collect_paper_numbers.py → paper_numbers.json → 그림/표
파이프라인으로 재발 차단(genuine·stuck0 두 처리를 모두 수록해 혼용 방지).
+ 추가 규칙(2026-08-05): novelty는 부재-프레임 금지, 확장-프레임으로(NOVELTY.md 상단).
출처: 방어문서 §9 / PROJECT_KNOWLEDGE Framing Rules.""",

# 16. 활용
"""[각 항목의 근거]
① 온라인 가상 센서: CPU batch-1 p99 6.4 ms(W=4; 중앙값 2.8 ms) — 10 ms 주기의 64%.
  반직관: CUDA는 ~8× 느림(p99 43–72 ms; 0.2M 파라미터로는 커널 오버헤드 상쇄 불가;
  호출별·amortized 두 측정 일치). 지침: 제어 컴퓨터의 CPU에서.
  conformal 구간 동반(α=0.10): 동일 절차를 PCHIP·persistence에도 적용한 공정 비교에서
  모델이 8/8 Winkler 승(persistence의 ~0.80×) · Mondrian(Δt×활동도 층화)이 전부 개선.
  정직한 실패: 커버리지는 주변적 — shot별 50~100% 산포, shot-조건부 보정은 shot 수 부족.
② 오프라인 재분석: 관측 모집단에서 사전등록 보간보다 우수(GP와는 동률), 과도 구간
  (peak)에서 이점 최대.
③ 진단 정보 감사 방법론: ablation+사전등록+shot-군집 CI로 "이 진단이 저 물리량을
  나르는가"를 검정 — Mirnov 전처리 결함·NBI 부재를 데이터로 고발한 것이 시범.
④ 평가 프로토콜 이식: "인과 모델 vs 미래 쓰는 현직 보간 + shot-군집 paired bootstrap
  + 사전등록" — Thomson·XICS 등 다른 희소 진단의 ML 채움 연구가 그대로 사용 가능.
⑤ 로드맵(비용 낮은 순, 전부 아카이브 데이터로 실행 가능): 이력 reach 확장 →
  원 kHz Mirnov window 특징(V_rot 최대 레버) → NBI 토크 확보(원인 변수) →
  shot별 표준화(캠페인 수리) → 타 장치(EAST·DIII-D) 이식.
출처: THESIS §8l·§8m / 방어문서 §10.""",

# 17. Q&A I
"""[답변 스크립트 보강]
Q1 MNAR: "직접 검증은 불가능하니 재가중으로 정량화했습니다. 도메인 내 결측 지점에서
  인과 우위는 4/4 +0.29로 살아남고 보정 비용은 0.06~0.12뿐입니다. 오프라인 보간
  대비는 1/4만 살아남아 — 그래서 배치 주장을 인과로 좁혔습니다."
Q2 PCHIP/linear: "linear가 근소하게 더 강합니다(Δt≤15 ms가 97.7%라 3차의 여지가
  없음). PR1로 사전등록했고, linear 대비도 genuine 3/4 PASS — 시드 42 n.s.까지
  논문에 명시했습니다." (+ GP는 Q9에서: 최강 팔이고 동률로 보고)
Q3 검정력: "맞습니다 — shot이 올바른 독립 단위라 지불한 비용이고, 모든 판정의 구속
  조건이라고 논문 한계 첫 줄에 썼습니다. V_rot 미해결의 원인이며 다중 캠페인 확장이
  해결책입니다."
Q4 과적합: "test는 탐색 시작 전 예약·전 과정 미열람, 시드 1/7/123은 아키텍처 선택에
  한 번도 쓰이지 않은 선택-밖 재현입니다. 게이트도 val의 clean skill입니다."
Q5 미래 보간 비교 의미: "의도적 불리함입니다. 이기면 고속 진단이 시간 구조 밖 정보를
  나른다는 강한 증거이고, 실시간 주장은 별도로 인과 기준선과 비교합니다(분리)."
Q6 W=4: "24-run sweep으로 우리가 먼저 답했습니다 — 관측 1개가 전부, plateau 규칙은
  W=2, W>2의 논리는 skill이 아니라 커버리지(길간극 채점 4~10배)."
Q7 불확실성: "conformal이 8/8 Winkler 승, 단 보장은 주변 커버리지 — shot별
  50~100% 산포를 스스로 보고했고 수리(shot-조건부)가 왜 지금 불가한지도 명시."
Q8 held: "우리가 발견했고(54%, 499/641 파일), 전수 정량화·평가 제외·학습 제거
  규약화까지 했습니다. 초기 오판을 4-seed paired로 뒤집은 과정도 공개합니다."
출처: 방어문서 §11 Q1~Q8·Q16.""",

# 18. Q&A II
"""[답변 스크립트 보강]
Q V_rot 실패?: "동률로 보고하되 — 물리로 사전 예측된 비대칭이고(fast-only −0.64),
  인과 대비로는 캠페인 분할에서도 4/4 유의 승, peak에서도 유의 승입니다. 실패한
  타겟이 아니라 의미 있는 비교가 인과뿐인 타겟입니다."
Q 복잡도: "1,258-파라미터 완전 해석 모델이 Tᵢ 마진의 31.5%만 회수함을 측정했습니다
  — 복잡도가 사는 것(비선형·비국소 구조)을 정량적으로 말할 수 있습니다."
Q 실시간: "CPU p99 6.4 ms 실측 — GPU가 오히려 8× 느린 것도 두 방식으로 검증.
  특징 조립·수집 지연은 모델 밖이라 별도임을 인정합니다."
Q novelty (확장 프레임으로!): "그 계열(Diag2Diag 등)의 프로그램을 이어받아 3축으로
  확장합니다 — 전자→이온 타겟, 동시각→인과 이력, 가정→사전등록 검정. Diag2Diag는
  원문 확인 결과 CER이 입력이고 타겟은 Thomson뿐입니다. RTCAKENN에는 방어 3종."
Q 일반화: "캠페인 시간 분할을 직접 수행해 오프라인 우위 소멸(0/4)을 자진 보고했고,
  원인(BES 1.22σ 드리프트)을 측정했으며 수리(shot별 표준화)를 지목했습니다.
  타 장치는 미검증 한계로 명시합니다."
Q Mirnov: "기여 ~0(무해)을 ablation으로, 파생 4종 무효를 4-seed paired로 확인.
  원인은 무필터 100 Hz 데시메이션(lag-1 −0.009 실측) — 수리는 원 kHz 스트림."
Q GP: "실측했습니다 — GP가 최강 오프라인 팔(PCHIP에 4/4 승)이고 모델은 동률(승 1/4·
  패 0/4). 사전등록 headline은 유지되고 배치 주장은 무영향. 그대로 보고했습니다."
Q fit-failure: "제거 민감도를 돌렸더니 방향이 반대 — 4/4 유지에 skill이 약 2배로
  뜁니다. headline은 이 아티팩트에 대해 보수적입니다."
Q 반경 방향: "페데스탈-상단 채널 선택에서 상속된 프레이밍 — 한계 절 명시, ELM 위상
  분석과 함께 향후 과제입니다." (유일하게 ●○○ — 순순히 인정할 것)
출처: 방어문서 §11 Q4·Q7~Q17.""",

# 19. 감사 결과
"""[각 행의 상세]
1 목차별정리: "held는 학습 무오염"(§8c에서 뒤집힌 결론)·Transformer 아키텍처(숫자를
  만든 적 없는 model.py 설명)·유지값 포함 headline — 3중 구버전을 전면 재작성.
2 발표덱 4종: make_figures.py를 paper_numbers.json 판독으로 전환(수기 숫자 금지 규칙을
  발표 그림에도 강제), 1시간 덱 40장(스트레스·배치 신설), 20분·흐름·1pager 재빌드,
  1pager의 글리프 tofu(matplotlib에 ᵢ/ₑ/−/≈ 없음)도 수리 — 전 덱 preview QC 0경고.
3 문헌 재검증: 18쿼리+원문 12건 — 주장 3개 STANDS, Shousha 연도 정정, 신규 인용
  3건(Kim/Char/Jung 문구), 정적 검사 통과.
4 수치 대조: 논문↔THESIS↔paper_numbers.json 불일치 0건.
5 GP 팔(§8p): 최강 오프라인 팔 판명·모델 동률 — 논문 headline·한계·결론 반영.
6 fit-failure(§8q): 제거 시 skill ~2배·4/4 유지 — headline 보수적.
7 novelty 프레임: 부재→확장 재설계(논문·문서·덱·규칙 4곳).

[잔여 — 다음 결정]
shot별(인과) 표준화 통제 실험: 정규화 계약 변경이라 승인 필요. 설계 완료 —
env 1곳 변경, 4-seed, 무작위+캠페인 두 분할 규칙 채점, GPU 학습 8회 규모.
인과 과거-전용 GP 팔: GP 동률의 지목된 해소책, 평가 전용으로 실행 가능.
출처: 방어문서 §12 / THESIS §8o~§8q.""",

# 20. 한 장 요약
"""[마무리 대사 예시]
"관측된 지점에서는 사전등록된 미래-사용 보간을 4개 분할 전부에서 유의하게 이기고,
최강 오프라인 평활기(GP)와는 동률입니다. 배치에 관한 주장은 더 좁고 더 튼튼합니다 —
진짜 결측 지점으로 재가중해도, 캠페인을 시간으로 갈라도 살아남는 것은 인과 기준선
대비 우위뿐이고, 그것이 실시간 가상 센서가 실제로 경쟁하는 상대입니다. V_rot의
동률은 실패가 아니라 물리로 예측되고 ablation으로 확인된 발견이며, 그 발견이 다음
데이터 과제(NBI 토크·원 Mirnov)를 지목합니다. 작동하지 않는 지점들도 전부 본문에
있고, 각각 그것을 뒤집을 측정과 함께입니다."

[숫자 최종 점검 목록]
genuine Tᵢ +0.179/+0.197/+0.280/+0.263 · V_rot PASS 1/4(+0.162) ·
재가중 +0.29 4/4 · 캠페인 +0.222(3/4)/+0.216(4/4) · GP 동률(승 1/4·패 0/4) ·
fit-fail 제거 시 +0.36~+0.59 · peak +0.702/+0.438 · 사다리 31.5%/7% ·
CPU p99 6.4 ms · conformal 8/8 · 커버리지 54.1%/4.8% · held 54%/499파일.
전부 THESIS_RESULTS.md §8 및 동결 산출물에서 재확인 가능.""",
]

assert len(NOTES) == len(prs.slides._sldIdLst), (
    f"NOTES({len(NOTES)}) != slides({len(prs.slides._sldIdLst)})")
for _s, _n in zip(prs.slides, NOTES):
    _s.notes_slide.notes_text_frame.text = _n.strip("\n")

OUT = os.path.join(HERE, "KSTAR_CES_종합방어.pptx")
prs.save(OUT)
print("saved:", OUT, "slides:", len(prs.slides._sldIdLst), "| notes on all slides")
