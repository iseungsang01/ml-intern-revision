# -*- coding: utf-8 -*-
"""KSTAR CES nowcasting 포스터 초안 (세로 45 x 60 cm; 200 % 로 인쇄하면 90 x 120 cm).

  py docs/presentation/make_figures_poster.py     # 그림 먼저
  py docs/presentation/build_poster.py            # -> docs/presentation/KSTAR_CES_포스터_초안.pptx
  py docs/presentation/check_poster_numbers.py    # 포스터의 모든 숫자를 원천과 대조

* 배치는 2018년 R&E 포스터의 틀을 따른다: 위에 제목 띠, 두 단, 회색 머리띠가 달린 네 구역
  (Ⅰ 연구 목표 · Ⅱ 연구 과정 및 방법 · Ⅲ 연구 결과 및 결론 · Ⅳ 고찰), 결과는 "그림 → 화살표 → 해석".
* 내용과 수치는 PPCF 원고(docs/paper/ppcf/main_ppcf.tex)와 docs/paper/file_rules.md 의 것만 쓴다.
* 문체: 초록체(서술형 종결). 결과는 점추정 + 95 % 구간 + 0 제외 여부. 용어는 흔한 영어 용어.
* 글상자 높이는 Pretendard 실제 글자 폭으로 줄 수를 세어 정한다(줄 간격은 pt 고정).
* 도형은 전부 PowerPoint 기본 도형과 글상자이다 — 받아서 그대로 고칠 수 있다.
"""
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
FIG = HERE / "figures" / "poster"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "KSTAR_CES_포스터_초안.pptx"

W_CM, H_CM = 45.0, 60.0
FONT = "Pretendard"


def rgb(h):
    return RGBColor(int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


NAVY, INK, BODY, MUTED = rgb("13335F"), rgb("121418"), rgb("3A3F47"), rgb("6B7280")
RULE, HEAD, PANEL, WHITE = rgb("C9CDD3"), rgb("DFE4EB"), rgb("F1F3F6"), rgb("FFFFFF")
ACCENT = rgb("E8743B")                         # 화살표 (원본 포스터의 주황 화살표)
TINT_B, TINT_O = rgb("E6EEFA"), rgb("FCE9DF")   # 구조도: T_i 경로(파랑) · CES 이력/V_rot 경로(주황)
LINE_B, LINE_O = rgb("2A78D6"), rgb("EB6834")
SUBTLE = rgb("D9E0EA")                          # 제목 띠 안의 보조 글자
BEAM = rgb("9DB7D8")

# ------------------------------------------------------------------ 글자 폭 계측 (Pretendard)
_uf = Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft" / "Windows" / "Fonts"


def _load(name):
    p = _uf / name
    return ImageFont.truetype(str(p if p.exists() else Path(r"C:\Windows\Fonts\malgun.ttf")), 200)


F_REG, F_BOLD = _load("Pretendard-Regular.otf"), _load("Pretendard-Bold.otf")
TOK = re.compile(r"(\*\*|T_i|V_rot|V_i)")
PT2CM = 2.54 / 72.0


NB = "\u00a0"


def glue(s):
    """구간 [a, b], 범위 a … b, 숫자와 단위를 한 줄에 묶는다(줄바꿈 없는 공백)."""
    s = re.sub(r"\[[^\]]*\]", lambda m: m.group(0).replace(" ", NB), s)
    s = s.replace(" … ", NB + "…" + NB)
    s = re.sub(r"(?<=\d) (?=%)", NB, s)
    return re.sub(r"(?<=\d) (?=(?:ms|eV|keV|kHz|MHz|µs)(?![A-Za-z]))", NB, s)


def parse(s, bold=False):
    """'**굵게**', 'T_i'(아래 첨자) 표기를 (글자, 굵기, 첨자) 조각으로."""
    s = glue(s)
    out, b = [], bold
    for part in TOK.split(s):
        if not part:
            continue
        if part == "**":
            b = not b
        elif part in ("T_i", "V_rot", "V_i"):
            out += [(part[0], b, False), (part[2:], b, True)]
        else:
            out.append((part, b, False))
    return out


def _w(s, pt, bold):
    return (F_BOLD if bold else F_REG).getlength(s) / 200.0 * pt * PT2CM


def n_lines(s, pt, width, bold=False):
    """낱말 단위 줄바꿈(한글 낱말을 자르지 않음)으로 줄 수를 센다."""
    words, cur = [], 0.0
    for txt, b, sub in parse(s, bold):
        for atom in re.split(r"( )", txt):
            if atom == "":
                continue
            if atom == " ":
                words.append(cur)
                cur = 0.0
            else:
                cur += _w(atom, pt * (0.66 if sub else 1.0), b)
    words.append(cur)
    space = _w(" ", pt, bold)
    lines, x = 1, 0.0
    for w in words:
        if x > 0 and x + space + w > width * 0.985:
            lines, x = lines + 1, w
        else:
            x = x + (space if x > 0 else 0) + w
    return lines


def text_h(paras, pt, width, pitch, after=5.0, bold=False):
    paras = [paras] if isinstance(paras, str) else paras
    return (sum(n_lines(s, pt, width, bold) for s in paras) * pitch + after * (len(paras) - 1)) * PT2CM


# ------------------------------------------------------------------ 기본 도형
prs = Presentation()
prs.slide_width, prs.slide_height = Cm(W_CM), Cm(H_CM)
slide = prs.slides.add_slide(prs.slide_layouts[6])


def _font(run, size, color, bold=False, sub=False):
    f = run.font
    f.size = Pt(size)
    f.bold = bold
    f.color.rgb = color
    f.name = FONT
    rpr = run._r.get_or_add_rPr()
    rpr.set("lang", "ko-KR")       # 한글 낱말을 자르지 않는 줄바꿈은 lang = ko-KR 과 latinLnBrk = 0 이 함께 있어야 듣는다
    rpr.set("altLang", "en-US")
    latin = rpr.find(qn("a:latin"))
    for tag in ("a:cs", "a:ea"):
        latin.addnext(rpr.makeelement(qn(tag), {"typeface": FONT}))
    if sub:
        rpr.set("baseline", "-25000")


def _fill_paras(tf, paras, size, color, pitch, after, align, bold):
    for i, s in enumerate(paras):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.line_spacing = Pt(pitch)
        p.space_after = Pt(after if i < len(paras) - 1 else 0)
        p._p.get_or_add_pPr().set("latinLnBrk", "0")
        for txt, b, sub in parse(s, bold):
            r = p.add_run()
            r.text = txt
            _font(r, size, color, b, sub)


def text(x, y, w, paras, size=11.0, color=BODY, pitch=None, after=5.0, align=PP_ALIGN.LEFT, bold=False, name=None):
    """글상자. 높이는 줄 수로 계산해 돌려준다(cm)."""
    paras = [paras] if isinstance(paras, str) else paras
    pitch = pitch or round(size * 1.38, 1)
    need = text_h(paras, size, w, pitch, after, bold)
    tb = slide.shapes.add_textbox(Cm(x), Cm(y), Cm(w), Cm(need))
    if name:
        tb.name = name
    tf = tb.text_frame
    tf.word_wrap = True
    tf.auto_size = None
    tf.vertical_anchor = MSO_ANCHOR.TOP
    for side in ("margin_left", "margin_right", "margin_top", "margin_bottom"):
        setattr(tf, side, 0)
    _fill_paras(tf, paras, size, color, pitch, after, align, bold)
    return need


def box(x, y, w, h, fill=WHITE, line=None, lw=0.75, radius=0.28, name=None, dash=False,
        shape=MSO_SHAPE.ROUNDED_RECTANGLE):
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


def label_in(shape, paras, size, color=INK, bold=False, pitch=None, align=PP_ALIGN.CENTER, pad=0.12):
    """도형 안 글자."""
    paras = [paras] if isinstance(paras, str) else paras
    tf = shape.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    for side in ("margin_left", "margin_right", "margin_top", "margin_bottom"):
        setattr(tf, side, Cm(pad))
    _fill_paras(tf, paras, size, color, pitch or round(size * 1.3, 1), 0, align, bold)


def arrow(x, y, w, h, kind="down", color=ACCENT, name=None):
    shp = {"down": MSO_SHAPE.DOWN_ARROW, "right": MSO_SHAPE.RIGHT_ARROW}[kind]
    return box(x, y, w, h, fill=color, shape=shp, name=name or f"arrow.{kind}")


def line_arrow(x1, y1, x2, y2, color=MUTED, w=1.25, head=True, name=None):
    c = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Cm(x1), Cm(y1), Cm(x2), Cm(y2))
    st = c._element.find(qn("p:style"))      # 기본 선 스타일(그림자 포함)을 떼고 색과 굵기만 준다
    if st is not None:
        c._element.remove(st)
    c.line.color.rgb = color
    c.line.width = Pt(w)
    if head:
        ln = c.line._get_or_add_ln()
        ln.append(ln.makeelement(qn("a:tailEnd"), {"type": "triangle", "w": "med", "len": "med"}))
    if name:
        c.name = name
    return c


def pic_h(name, w):
    iw, ih = Image.open(FIG / name).size
    return w * ih / iw


def pic(name, x, y, w, alt):
    """그림을 폭 w 에 맞춰 놓고 높이를 돌려준다."""
    ph = pic_h(name, w)
    p = slide.shapes.add_picture(str(FIG / name), Cm(x), Cm(y), Cm(w), Cm(ph))
    p.name = "fig." + name.replace(".png", "")
    p._element.nvPicPr.cNvPr.set("descr", alt)
    return ph


def section(x, y, w, h, title, key):
    box(x, y, w, h, fill=PANEL, radius=0.45, name=f"{key}.panel")
    hd = box(x, y, w, HEAD_H, fill=HEAD, radius=0.45, name=f"{key}.header")
    label_in(hd, title, 21, color=NAVY, bold=True, align=PP_ALIGN.LEFT, pad=0.55)


def card(x, y, w, h, name):
    return box(x, y, w, h, fill=WHITE, line=RULE, lw=0.6, radius=0.3, name=name)


def subhead(x, y, w, s, name=None):
    return text(x, y, w, s, size=12.5, color=NAVY, bold=True, pitch=17.0, name=name)


# ------------------------------------------------------------------ 격자
M = 1.3
COL_W = 20.85
XL, XR = M, W_CM - M - COL_W
TOP, BOTTOM = 7.55, 58.85
HEAD_H = 1.25
PAD = 0.38            # 구역 안쪽 여백
GAP = 0.36            # 카드 사이
CP = 0.36             # 카드 안쪽 여백
CARD_W = COL_W - 2 * PAD
INNER = CARD_W - 2 * CP
BODY_PT, SMALL_PT, CAP_PT = 11.0, 10.3, 8.8
SP = 14.2             # 본문(작은 글자) 줄 간격, pt
SUB_H = 17.0 * PT2CM  # 소제목 한 줄 높이

# ------------------------------------------------------------------ 제목 띠
box(M, 1.2, W_CM - 2 * M, 5.5, fill=NAVY, radius=0.5, name="title.band")
text(M + 1.0, 1.85, W_CM - 2 * M - 2.0, "KSTAR pedestal top 측정선의 CES 이온 온도 nowcasting", size=33, color=WHITE,
     bold=True, pitch=40, name="title")
text(M + 1.0, 3.65, W_CM - 2 * M - 2.0, "같은 위치의 빠른 진단(BES·ECEI)을 함께 읽는 인과 estimator와 hold·보간의 비교",
     size=16.5, color=SUBTLE, pitch=21, name="subtitle")
text(M + 1.0, 5.15, 20.0, "이승상  ·  서울대학교 원자핵공학과", size=14, color=WHITE, bold=True, pitch=18, name="author")
text(W_CM - M - 1.0 - 20.0, 5.25, 20.0, "자료: KSTAR 2022 캠페인 641 방전(KSTAR team 제공) · 공개 FAIR-MAST",
     size=11, color=SUBTLE, pitch=15, align=PP_ALIGN.RIGHT, name="data.line")

# ================================================================== Ⅰ. 연구 목표
goal_paras = [
    "전하 교환 분광(CES)은 이온 온도 T_i를 10 ms마다 측정한다. pedestal top 측정선에서는 신호가 약하여 T_i 표본의 **7.7 %**가 비어 있었다.",
    "지금 값을 써야 하는 감시·제어는 다음 표본을 기다릴 수 없으므로 보간을 쓸 수 없다. 가장 단순한 대안은 직전 값을 유지하는 hold이며, 다음 측정 없이 지금 값을 추정하는 일을 nowcasting이라 한다.",
    "본 연구는 다음 CES 표본을 쓰지 않는 estimator 가운데 어느 것이 hold와 보간보다 오차가 낮은지, 한 프레임(10 ms) 안에 계산되는지, 표본이 드물어질 때 어떻게 되는지를 물었다.",
]
FLOW = [("문제", "pedestal top의 CES는 약하고 자주 빈다"),
        ("제약", "지금 값이 필요하여 다음 표본을 못 쓴다"),
        ("질문", "hold·보간보다 오차가 낮은가, 10 ms 안에 도는가"),
        ("방법", "CES 이력과 BES·ECEI를 읽는 순환 신경망"),
        ("검증", "641 방전, 등록 시험, 재생 실험, 두 번째 장치")]
FLOW_H = 2.4
y = TOP
H1 = HEAD_H + PAD + CP + text_h(goal_paras, BODY_PT, INNER, 15.2) + 0.4 + FLOW_H + CP + PAD
section(XL, y, COL_W, H1, "Ⅰ. 연구 목표", "sec1")
cy = y + HEAD_H + PAD
card(XL + PAD, cy, CARD_W, H1 - HEAD_H - 2 * PAD, "sec1.card")
ty = cy + CP
ty += text(XL + PAD + CP, ty, INNER, goal_paras, size=BODY_PT, pitch=15.2, name="sec1.text") + 0.4
bw, aw = 3.38, 0.62
for i, (k, v) in enumerate(FLOW):
    bx = XL + PAD + CP + i * (bw + aw)
    b = box(bx, ty, bw, FLOW_H, fill=PANEL, radius=0.22, name=f"sec1.flow{i + 1}")
    label_in(b, ["**" + k + "**", v], 9.6, color=INK, pitch=12.6, pad=0.16)
    if i < len(FLOW) - 1:
        arrow(bx + bw + 0.11, ty + FLOW_H / 2 - 0.26, aw - 0.22, 0.52, "right", name=f"sec1.flowarrow{i + 1}")

# ================================================================== Ⅱ. 연구 과정 및 방법
y2 = y + H1 + 0.7
H2 = BOTTOM - y2
section(XL, y2, COL_W, H2, "Ⅱ. 연구 과정 및 방법", "sec2")
cx = XL + PAD
ix = cx + CP
cy = y2 + HEAD_H + PAD

# ---- 1. 자료
A_paras = [
    "KSTAR 2022 캠페인의 방전 641개(10 ms 격자 245,925행)를 사용하였다. 한 행은 같은 위치의 BES(빔 방출 분광) 9채널, ECEI(전자 사이클로트론 방출 영상) 4채널, Mirnov 코일 2채널과 한 측정선(R = 2.198 m)의 CES T_i, V_rot을 담는다.",
    "T_i는 행의 7.7 %, 회전 속도 V_rot은 23.5 %가 결측이었다. V_rot 관측의 54 %는 직전 값이 끝자리까지 반복된 유지값이어서 측정으로 세지 않고 제거하였다.",
]
imgA_h = pic_h("poster_missing.png", INNER)
hA = CP + SUB_H + 0.12 + text_h(A_paras, SMALL_PT, INNER, SP) + 0.2 + imgA_h + CP
B_paras = [
    "12개 방전의 원시 신호(0.5–2 MHz)를 10 ms 파일과 대조하여 한 행의 뜻을 확인하였다. 빠른 진단의 값은 그 행 시각의 원시 표본 하나였고(BES 6,502행 중 6,502행 일치) 평균값이 아니었다.",
    "CES 값은 시각 표지에서 시작하는 10 ms 노출의 값이었다. 빔이 약 10 ms 꺼지는 blip의 프레임은 파일에 없었으며, 원시 자료 구간의 45건이 모두 일치하였다. 따라서 estimator는 추정 대상 프레임이 시작되는 시각까지의 정보만 읽는다.",
]
SCH_H = 4.1
hB = CP + SUB_H + 0.12 + text_h(B_paras, SMALL_PT, INNER, SP) + 0.3 + SCH_H + CP
K_paras = [
    "빔이 약 10 ms 꺼진 blip 1,394건의 직후에 T_i는 직전의 **0.819 [0.796, 0.838]**, 빔을 쓰지 않는 ECEI 세기는 0.926 [0.910, 0.939]이었다.",
    "둘 다 약 90 ms에 회복하였고, 빔 방출은 넷째 행에 0.954로 돌아왔다.",
    "blip 사이의 CES 값은 빔 세기가 아니라 플라스마의 변화를 따랐다.",
]
K_IMG_W = 9.4
kt_w = INNER - K_IMG_W - 0.5
k_img_h = pic_h("poster_blip.png", K_IMG_W)
k_body = max(k_img_h, text_h(K_paras, SMALL_PT, kt_w, SP, 4))
hK = CP + SUB_H + 0.12 + k_body + CP
# 설계 선택의 흐름: 무엇을 골랐고, 무엇과 견줬고, 무엇을 쟀는가 (PROJECT_KNOWLEDGE.md 3절)
WHY_HEAD = ("선택", "견준 대안", "측정한 것")
WHY_ROWS = [
    ("모든 행을 순서대로 읽는 순환 모델", "과거 W개 행만 읽는 윈도 모델 (W = 2–8)",
     "윈도 모델은 W = 2에서 이미 평평하였다. 순환 모델의 짝지은 T_i skill은 16쌍 모두 양이었고 pooled **+0.081 [+0.067, +0.096]**이었다."),
    ("순환 연산자 LSTM", "합성곱(TCN), attention",
     "같은 문맥 길이에서 세 계열이 0.023 안이었다. GRU·단순 RNN과는 견주지 않았다."),
    ("357,570 parameters (2층, 160 유닛)", "폭 34k–879k parameters",
     "PCHIP 대비 skill이 +0.230 … +0.236으로 평평하였다. 깊이와 더 큰 크기는 재지 않았다."),
    ("한 스텝 0.49 ms", "10 ms 프레임",
     "계산 시간은 크기의 제약이 아니다. 인과 GP와 섞으면 skill이 **+0.021 … +0.037** 더 높아(네 분할 모두 0 제외) 구조는 더 나아질 수 있다."),
]
WHY_COLS = (4.5, 4.3, INNER - 8.8 - 0.3)
WHY_PT, WHY_PITCH = 9.6, 12.8


def why_table(x, y, draw=True):
    """머리줄 + 행마다 세 칸. 높이를 돌려준다."""
    xs = (x, x + WHY_COLS[0] + 0.15, x + WHY_COLS[0] + WHY_COLS[1] + 0.3)
    yy = y
    if draw:
        for j, hname in enumerate(WHY_HEAD):
            text(xs[j], yy, WHY_COLS[j], hname, size=9.0, color=MUTED, bold=True, pitch=12, name=f"why.head{j}")
    yy += 12 * PT2CM + 0.1
    if draw:
        line_arrow(x, yy, x + INNER, yy, color=RULE, w=0.9, head=False, name="why.rule0")
    yy += 0.1
    for i, row in enumerate(WHY_ROWS):
        hs = [text_h(c, WHY_PT, WHY_COLS[j], WHY_PITCH) for j, c in enumerate(row)]
        if draw:
            for j, c in enumerate(row):
                text(xs[j], yy, WHY_COLS[j], c, size=WHY_PT, color=INK if j == 0 else BODY, pitch=WHY_PITCH,
                     bold=(j == 0), name=f"why.r{i}c{j}")
        yy += max(hs) + 0.1
        if draw:
            line_arrow(x, yy, x + INNER, yy, color=RULE, w=0.6, head=False, name=f"why.rule{i + 1}")
        yy += 0.1
    return yy - y


BH, R2 = 1.8, 2.2
ARCH_H = R2 + BH
hC = CP + SUB_H + 0.15 + ARCH_H + 0.3 + why_table(0, 0, draw=False) + CP - 0.1
D_paras = [
    "측정된 값을 가리고 추정한 뒤 가린 값과 비교하였다. skill이 0보다 크면 기준선보다 평균 제곱 오차(MSE)가 작다. 구간은 방전 단위 bootstrap의 95 % 구간이며, 구간이 0을 포함하지 않을 때만 차이로 읽었다.",
    "기준선은 hold(직전 값 유지), 인과 GP(CES 이력에만 맞춘 Gaussian process), 그리고 다음 표본까지 쓰는 보간(PCHIP, GP)이다. 보간은 실시간으로 쓸 수 없는 참고 기준이다. 주요 진술은 결과를 보기 전에 등록하였다.",
]
FORM_H = 0.9
hD = CP + SUB_H + 0.12 + FORM_H + 0.2 + text_h(D_paras, SMALL_PT, INNER, SP) + CP
left_over = (BOTTOM - PAD) - (cy + hA + hB + hK + hC + hD + 4 * GAP)
GAP2 = GAP + max(0.0, left_over) / 4
print("left column: spare height %.2f cm -> card gap %.2f cm" % (left_over, GAP2))
card(cx, cy, CARD_W, hA, "sec2.cardA")
ty = cy + CP
ty += subhead(ix, ty, INNER, "1. 자료", "sec2.A.head") + 0.12
ty += text(ix, ty, INNER, A_paras, size=SMALL_PT, pitch=SP, name="sec2.A.text") + 0.2
pic("poster_missing.png", ix, ty, INNER, "10 ms 격자 행 가운데 측정, 유지값, 결측의 비율: T_i 92.3 / 0 / 7.7 %, V_rot 35.2 / 41.3 / 23.5 %")
cy += hA + GAP2

# ---- 2. 한 행이 담는 것
card(cx, cy, CARD_W, hB, "sec2.cardB")
ty = cy + CP
ty += subhead(ix, ty, INNER, "2. 한 행이 담는 것 (원시 1 µs 자료와 대조)", "sec2.B.head") + 0.12
ty += text(ix, ty, INNER, B_paras, size=SMALL_PT, pitch=SP, name="sec2.B.text") + 0.3
sx0, sw = ix + 1.25, 3.5
fy = ty + 1.25                       # frame row
slots = [("CES 프레임", PANEL, None, False, BODY),
         ("직전 프레임\n(hold가 쓰는 값)", HEAD, None, False, INK),
         ("**추정 대상 프레임**\n(시각 표지 t에서 노출 시작)", WHITE, NAVY, False, INK),
         ("blip: 프레임 없음\n(빠진 행)", None, MUTED, True, MUTED),
         ("CES 프레임", PANEL, None, False, BODY)]
text(ix, fy + 0.4, 1.15, "CES", size=9.2, color=MUTED, bold=True, pitch=12, name="sch.label.ces")
for i, (lab, fill, ln, dash, col) in enumerate(slots):
    b = box(sx0 + i * sw + 0.05, fy, sw - 0.10, 1.25, fill=fill, line=ln, lw=1.5 if ln is NAVY else 0.9, radius=0.12,
            dash=dash, name=f"sch.frame{i}")
    label_in(b, lab.split("\n"), 8.6, color=col, pitch=11.4, pad=0.06)
by_ = fy + 1.45                      # beam bar
text(ix, by_ - 0.06, 1.15, "빔", size=9.2, color=MUTED, bold=True, pitch=12, name="sch.label.beam")
box(sx0 + 0.05, by_, 3 * sw - 0.10, 0.3, fill=BEAM, radius=0.05, name="sch.beam.on1")
box(sx0 + 4 * sw + 0.05, by_, sw - 0.10, 0.3, fill=BEAM, radius=0.05, name="sch.beam.on2")
text(sx0 + 3 * sw, by_ - 0.04, sw, "꺼짐 (약 10 ms)", size=8.2, color=MUTED, pitch=10.5, align=PP_ALIGN.CENTER,
     name="sch.beam.off")
ay = by_ + 0.55                      # time axis
line_arrow(sx0 - 0.1, ay, sx0 + 5 * sw + 0.45, ay, color=MUTED, w=1.0, name="sch.axis")
for i, lab in enumerate(("t − 20 ms", "t − 10", "t", "t + 10", "t + 20", "t + 30")):
    line_arrow(sx0 + i * sw, ay - 0.1, sx0 + i * sw, ay + 0.1, color=MUTED, w=1.0, head=False, name=f"sch.tick{i}")
    text(sx0 + i * sw - 1.2, ay + 0.16, 2.4, lab, size=8.4, color=INK if lab == "t" else MUTED, bold=(lab == "t"),
         pitch=11, align=PP_ALIGN.CENTER, name=f"sch.ticklabel{i}")
mx = sx0 + 2 * sw                    # fast-channel snapshot at t
text(mx - 5.2, ty, 10.4, "빠른 진단(BES·ECEI·Mirnov) 표본: 시각 t의 순간값", size=9.0, color=INK, bold=True, pitch=12,
     align=PP_ALIGN.CENTER, name="sch.snapshot.label")
arrow(mx - 0.2, ty + 0.48, 0.4, 0.62, "down", name="sch.snapshot.arrow")
cy += hB + GAP2

# ---- 3. 자료의 점검: 빔 blip 뒤의 응답
card(cx, cy, CARD_W, hK, "sec2.cardK")
ty = cy + CP
ty += subhead(ix, ty, INNER, "3. 자료의 점검: 빔 blip 뒤의 응답", "sec2.K.head") + 0.12
pic("poster_blip.png", ix, ty + (k_body - k_img_h) / 2, K_IMG_W,
    "빔 blip 뒤 CES T_i, ECEI 세기, 빔 방출의 blip 직전 대비 비. 첫 행에서 T_i 0.819, ECEI 0.926, BES 0.656")
text(ix + K_IMG_W + 0.5, ty + (k_body - text_h(K_paras, SMALL_PT, kt_w, SP, 4)) / 2, kt_w, K_paras, size=SMALL_PT,
     pitch=SP, after=4, name="sec2.K.text")
cy += hK + GAP2

# ---- 4. 비교한 estimator
card(cx, cy, CARD_W, hC, "sec2.cardC")
ty = cy + CP
ty += subhead(ix, ty, INNER, "4. 모델 설계와 선택 이유", "sec2.C.head") + 0.15
c1, c2, c3 = ix, ix + 7.05, ix + 13.75
w1, w2, w3 = 6.0, 5.65, INNER - 13.75
nodes = [
    (c1, ty, w1, TINT_B, None, ["**빠른 진단 15채널**", "BES 9 · ECEI 4 · Mirnov 2", "행 t의 순간값"], "arch.fast"),
    (c1, ty + R2, w1, TINT_O, None, ["**CES 이력 7채널**", "직전까지의 관측값 · 경과 시간", "관측 여부 (행 t 이전만)"], "arch.ces"),
    (c2, ty, w2, TINT_B, LINE_B, ["**T_i 가지**", "LSTM 2층 · 160 유닛", "22채널을 읽음"], "arch.ti"),
    (c2, ty + R2, w2, TINT_O, LINE_O, ["**V_rot 가지**", "LSTM 1층 · 64 유닛", "CES 이력 7채널만 읽음"], "arch.vrot"),
    (c3, ty, w3, WHITE, LINE_B, ["**행 t의 T_i 추정**"], "arch.out.ti"),
    (c3, ty + R2, w3, WHITE, LINE_O, ["**행 t의 V_rot 추정**"], "arch.out.vrot"),
]
for (bx, by, bw_, fill, ln, lab, nm) in nodes:
    b = box(bx, by, bw_, BH, fill=fill, line=ln, lw=1.2, radius=0.2, name=nm)
    label_in(b, lab, 9.2, color=INK, pitch=12.2, pad=0.08)
line_arrow(c1 + w1 + 0.05, ty + BH / 2, c2 - 0.08, ty + BH / 2, color=LINE_B, w=1.5, name="arch.a1")
line_arrow(c1 + w1 + 0.05, ty + R2 + BH / 2 - 0.35, c2 - 0.08, ty + BH / 2 + 0.5, color=LINE_O, w=1.5, name="arch.a2")
line_arrow(c1 + w1 + 0.05, ty + R2 + BH / 2 + 0.2, c2 - 0.08, ty + R2 + BH / 2 + 0.2, color=LINE_O, w=1.5, name="arch.a3")
line_arrow(c2 + w2 + 0.05, ty + BH / 2, c3 - 0.08, ty + BH / 2, color=LINE_B, w=1.5, name="arch.a4")
line_arrow(c2 + w2 + 0.05, ty + R2 + BH / 2, c3 - 0.08, ty + R2 + BH / 2, color=LINE_O, w=1.5, name="arch.a5")
ty += ARCH_H + 0.3
why_table(ix, ty)
cy += hC + GAP2

# ---- 5. 채점
card(cx, cy, CARD_W, hD, "sec2.cardD")
ty = cy + CP
ty += subhead(ix, ty, INNER, "5. 채점과 기준선", "sec2.D.head") + 0.12
f = box(ix + 3.4, ty, INNER - 6.8, FORM_H, fill=PANEL, radius=0.2, name="sec2.D.formula")
label_in(f, "skill = 1 − MSE(estimator) / MSE(기준선)", 12.5, color=INK, bold=True, pitch=16)
ty += FORM_H + 0.2
text(ix, ty, INNER, D_paras, size=SMALL_PT, pitch=SP, name="sec2.D.text")
print("left column: card 5 ends %.2f cm above the panel edge" % ((BOTTOM - PAD) - (cy + hD)))

# ================================================================== Ⅲ. 연구 결과 및 결론
rx = XR + PAD
rix = rx + CP
TW = (CARD_W - GAP) / 2             # two-up card width
TIN = TW - 2 * CP                    # two-up inner width
TITLE_H = 0.66
ARROW_H = 0.5


def result_title(x, y, w, n, s, key):
    b = box(x, y + 0.02, 0.56, 0.56, fill=NAVY, shape=MSO_SHAPE.OVAL, name=f"{key}.badge")
    label_in(b, str(n), 10.5, color=WHITE, bold=True, pitch=13, pad=0.0)
    text(x + 0.74, y + 0.03, w - 0.74, s, size=11.5, color=INK, bold=True, pitch=15, name=f"{key}.title")


CAPH = 12.0 * PT2CM


def two_up(y, items):
    """items: (번호, 제목, 그림, 대체 글, 해석 문단들, 그림 아래 한 줄). 그림 · 화살표 · 해석 줄을 맞춘다."""
    img_hs = [pic_h(it[2], TIN) for it in items]
    txt_hs = [text_h(it[4], SMALL_PT, TIN, SP, 4) for it in items]
    ih_, th_ = max(img_hs), max(txt_hs)
    h = CP + TITLE_H + 0.12 + ih_ + 0.06 + CAPH + 0.14 + ARROW_H + 0.14 + th_ + CP
    for j, (n, title, img, alt, paras, cap) in enumerate(items):
        x = rx + j * (TW + GAP)
        card(x, y, TW, h, f"res{n}.card")
        result_title(x + CP, y + CP, TIN, n, title, f"res{n}")
        iy = y + CP + TITLE_H + 0.12
        pic(img, x + CP, iy + (ih_ - img_hs[j]) / 2, TIN, alt)
        text(x + CP, iy + ih_ + 0.06, TIN, cap, size=CAP_PT, color=MUTED, pitch=12, align=PP_ALIGN.CENTER,
             name=f"res{n}.caption")
        ay_ = iy + ih_ + 0.06 + CAPH + 0.14
        arrow(x + TW / 2 - 0.36, ay_, 0.72, ARROW_H, "down", name=f"res{n}.arrow")
        text(x + CP, ay_ + ARROW_H + 0.14, TIN, paras, size=SMALL_PT, pitch=SP, after=4, name=f"res{n}.text")
    return h


def wide(y, n, title, img, alt, paras, img_w):
    img_h = pic_h(img, img_w)
    tw = INNER - img_w - 1.05
    th_ = text_h(paras, SMALL_PT, tw, SP, 4)
    body = max(img_h, th_)
    h = CP + TITLE_H + 0.12 + body + CP
    card(rx, y, CARD_W, h, f"res{n}.card")
    result_title(rix, y + CP, INNER, n, title, f"res{n}")
    iy = y + CP + TITLE_H + 0.12
    pic(img, rix, iy + (body - img_h) / 2, img_w, alt)
    arrow(rix + img_w + 0.2, iy + body / 2 - 0.36, 0.62, 0.72, "right", name=f"res{n}.arrow")
    text(rix + img_w + 1.05, iy + max(0.0, (body - th_) / 2), tw, paras, size=SMALL_PT, pitch=SP, after=4,
         name=f"res{n}.text")
    return h


R1 = (1, "기준선 대비 T_i 오차 (KSTAR)", "poster_skill_kstar.png",
      "KSTAR T_i: 순환 신경망의 skill 과 95 % 구간. hold +0.391, 인과 GP +0.113, PCHIP 보간 +0.221, GP 보간 -0.010",
      ["다음 표본을 쓰지 않는 estimator 가운데 순환 신경망의 평균 제곱 오차가 가장 낮았다. 인과 GP 대비 skill은 **+0.113 [+0.085, +0.132]**이었고, 다음 표본까지 쓰는 GP 보간과는 구간이 0을 포함하였다.",
       "이 수치는 방전 전체의 통계로 입력을 표준화한 것이다. 학습 방전의 통계만 쓰는 인과 변형은 인과 GP 대비 +0.068 … +0.141이었다(네 분할 모두 0 제외)."],
      "639 방전을 여덟 번 다시 나눈 추정 · 3 keV 초과 값 제외")
R2 = (2, "두 번째 장치: MAST pedestal top", "poster_skill_mast.png",
      "MAST pedestal top T_i: 등록 확증의 skill 과 95 % 구간. 모든 입력 대 hold +0.148, Thomson 추가 +0.047",
      ["공개 MAST 자료 1,651 방전의 등록 시험에서, 숨긴 프레임에 대한 인과 추정의 hold 대비 skill은 **+0.148 [+0.122, +0.177]**이었다.",
       "같은 반경의 Thomson 산란을 더한 증분은 +0.047 [+0.033, +0.068]이었고, 다음 프레임을 쓰는 offline 보간에는 미치지 못하였다."],
      "확증용 1,651 방전 · pedestal top 74,060 프레임")
R3 = (3, "결측 구간을 지나는 추정의 예", "fig_trace.png",
      "세 방전의 T_i 시간 흔적: 측정값, hold, 인과 GP, 순환 신경망",
      ["회색 띠는 T_i가 비어 있는 행이고, 범례의 persistence는 hold이다. (a), (b)는 문헌에 보고된 두 방전으로 측정이 드물고 값의 변화가 크다. (c)는 전형적인 방전이다.",
       "방전 #31921에서 RMSE는 순환 신경망 244.2 eV, hold 282.8 eV, 인과 GP 283.4 eV였다.",
       "전형적인 방전 (c)에서는 순환 신경망 63.8 eV, 인과 GP 63.6 eV, hold 80.7 eV였다."])
R4 = (4, "CES가 드물게 보일 때 (재생 실험)", "fig_thin.png",
      "CES 표본 간격에 따른 hold 대비 skill 과 빠른 진단의 몫",
      ["측정 표본을 숨겨 50 ms마다 하나만 보이게 하였다. 고정한 순환 신경망(범례의 network)의 인과 GP 대비 skill은 **+0.108 … +0.129**(네 분할 모두 95 % 구간이 0 제외)이었다.",
       "보이는 표본 사이의 PCHIP 보간 대비 +0.062 … +0.155(네 분할 중 셋), 빠른 진단의 몫은 +0.076 … +0.156(네 분할 모두 0 제외)이었다."])
R5 = (5, "가장자리 버스트 구간 (원시 500 kHz BES)", "fig_bursts.png",
      "두 방전의 BES 세기와 버스트 표시, 측정된 T_i 와 hold, 인과 추정",
      ["원시 BES에서 가장자리 버스트(ELM과 부합)를 검출하였다. T_i의 결측과 큰 변화는 버스트가 이어지는 구간에 있었다.",
       "그 구간은 채점 행의 0.115였고, 빠른 진단이 준 오차 감소의 **0.604 [0.056, 0.786]**이 그 구간에 있었다. 구간이 넓어 집중은 확정되지 않았다."])

WIDE_W = 11.6
ry = TOP + HEAD_H + PAD
ry += two_up(ry, [R1, R2]) + GAP
ry += wide(ry, *R3, img_w=WIDE_W) + GAP
ry += wide(ry, *R4, img_w=WIDE_W) + GAP
ry += wide(ry, *R5, img_w=WIDE_W)
H3 = (ry + PAD) - TOP
section(XR, TOP, COL_W, H3, "Ⅲ. 연구 결과 및 결론", "sec3")

# ================================================================== Ⅳ. 고찰
y4 = TOP + H3 + 0.7
H4 = BOTTOM - y4
section(XR, y4, COL_W, H4, "Ⅳ. 고찰", "sec4")
card(rx, y4 + HEAD_H + PAD, CARD_W, H4 - HEAD_H - 2 * PAD, "sec4.card")
E_paras = [
    "이득은 평균 제곱 오차에 있었고, 중앙값 오차는 hold보다 낮지 않았다. 제곱 오차는 오차가 큰 행이 좌우한다.",
    "빠른 진단을 뺀 같은 신경망과 견준 skill은 +0.064 … +0.117이었다(3 keV 초과 값 제외, 네 분할 모두 0 제외). 그 값을 포함하면 +0.016 … +0.024로 줄었다.",
    "CES 값에는 행별 fit 오차가 없어, 버스트 직후의 100–300 eV가 온도인지 약한 신호의 fit인지는 가르지 못하였다. blip 뒤의 응답은 값이 플라스마를 따른다는 점검 하나이다.",
    "측정선은 하나이며 정규화 반경은 평형으로 확정되지 않았다(문헌의 두 방전에서 0.94–0.96). 회전 V_rot은 같은 빠른 진단과의 연관이 작아(|r| ≤ 0.044) 자기 이력으로 추정하였고, hold 대비 skill은 +0.265 [+0.198, +0.326]이었다.",
    "다음 단계는 CES 결측을 넣은 학습과, 인과 변형을 주 estimator로 둔 재검증이다. 버스트 직후의 값을 가르려면 빔 출력과 행별 fit 오차가 있어야 한다.",
]
REFS = ["**참고문헌**  [1] W. H. Ko et al., Rev. Sci. Instrum. 81, 10D740 (2010).  [2] J. K. Lee et al., Fusion Eng. Des. 222, 115518 (2026).  "
        "[3] C. Heo et al., Nucl. Fusion 65, 036003 (2025).  [4] S. Jackson et al., SoftwareX 27, 101869 (2024).",
        "**사사**  KSTAR 자료는 KSTAR team이 제공하였다. 별도의 연구비 지원은 없었다."]
need4 = text_h(E_paras, SMALL_PT, INNER, SP, 4) + 0.3 + text_h(REFS, CAP_PT, INNER, 12.2, 3)
free4 = (H4 - HEAD_H - 2 * PAD - 2 * CP) - need4
ty = y4 + HEAD_H + PAD + CP + max(0.0, free4) / 2
ty += text(rix, ty, INNER, E_paras, size=SMALL_PT, pitch=SP, after=4, name="sec4.text") + 0.3
text(rix, ty, INNER, REFS, size=CAP_PT, color=MUTED, pitch=12.2, after=3, name="sec4.refs")
print("right column: section III %.2f cm, section IV %.2f cm, free in IV %.2f cm" % (H3, H4, free4))


# ------------------------------------------------------------------ z-order: panels behind cards, headers on panels
tree = slide.shapes._spTree
for key in ("sec4", "sec3", "sec2", "sec1"):
    pn = [s._element for s in slide.shapes if s.name == f"{key}.panel"][0]
    hd = [s._element for s in slide.shapes if s.name == f"{key}.header"][0]
    tree.remove(pn)
    tree.remove(hd)
    tree.insert(2, hd)
    tree.insert(2, pn)

prs.core_properties.title = "KSTAR pedestal top 측정선의 CES 이온 온도 nowcasting (포스터 초안)"
prs.core_properties.author = "이승상"
prs.save(OUT)
print("saved", OUT)
