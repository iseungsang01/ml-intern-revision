"""KSTAR CES 나우캐스팅 발표(대본 포함). 핵플실 발표 템플릿(1조, 수정_5) 위에 짓는다.

  py docs/presentation/build_pptx_talk.py      -> docs/presentation/KSTAR_CES_발표_v2.pptx

* 템플릿: 핵플실발표자료_1조_김하빈, 이승상_수정_5.pptx 에서는 배경 모양(위쪽 세 색 띠, 남색 제목 띠, 표지 남색 블록)만 쓴다.
  글꼴(Pretendard · IBM Plex Mono)·색·배치는 v1 그대로.
* 본문 배치 문법은 Cellfie 기말발표 v5.17 에서 왔다: 큰 숫자 · 짧은 설명 · 출처 한 줄 · 회색 카드.
* 내용: THESIS_RESULTS.md §0 (2026-11-11)과 8쪽 논문 docs/paper/short/main_short_ko.tex 의 수치만 쓴다.
* 발표자 노트 = 쓰임·시간 / 핵심 / 대본(발표 말투) / 숫자와 출처 / 예상 질문.
* 용어: 파라미터, estimator 등 흔한 영어 용어. 결과는 점추정 + 95 % 구간 + 0 제외 여부.
"""
import io
import sys
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.dml import MSO_LINE
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Inches, Pt

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

HERE = Path(__file__).resolve().parent
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "KSTAR_CES_발표_v2.pptx"
TEMPLATE = Path(r"D:\승상 자료\대학교\4학년\2학기\핵플실\실험 3 (필라멘트)\발표준비\핵플실발표자료_1조_김하빈, 이승상_수정_5.pptx")

# 템플릿에서는 배경 모양(세 색 띠, 남색 제목 띠, 표지 남색 블록)만 쓴다. 글꼴·색·배치는 v1 그대로.
SANS, SANS_B, MONO = "Pretendard", "Pretendard", "IBM Plex Mono"
EA = "Pretendard"
INK = RGBColor(0x12, 0x14, 0x18)
BODY = RGBColor(0x3A, 0x3F, 0x47)
MUTED = RGBColor(0x6B, 0x72, 0x80)
RULE = RGBColor(0xC9, 0xCD, 0xD3)
GRAY = RGBColor(0xA4, 0xA9, 0xB1)
CARD = RGBColor(0xF1, 0xF1, 0xF4)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
TI = RGBColor(0x2F, 0x5E, 0x9E)    # 이온 온도
VR = RGBColor(0xB9, 0x56, 0x3C)    # 회전
TI_BG = RGBColor(0xE9, 0xF0, 0xF8)
VR_BG = RGBColor(0xF8, 0xEC, 0xE7)

prs = Presentation(TEMPLATE)
_ids = prs.slides._sldIdLst
for _sid in list(_ids):                # 템플릿의 기존 슬라이드는 지우고 마스터·레이아웃만 쓴다
    prs.part.drop_rel(_sid.rId)
    _ids.remove(_sid)
W, H = prs.slide_width, prs.slide_height      # 13.333 × 7.5 in
S = 2 / 3                                      # 설계 좌표(20 in 폭) → 템플릿 좌표
FS = 0.70                                      # 글자 크기 배율
DY = Inches(0.55)                              # 남색 제목 띠 아래로 내리는 양


def I(v):
    return Inches(v * S)


def P(v):
    return Pt(v * FS)


L = I(1.3)
CW = W - 2 * L
LAY = {l.name: l for l in prs.slide_layouts}


# ---------------------------------------------------------------- 기본 도형 (Cellfie build_v4.py 규칙)

def _font(run, face, size, color, bold=False, spacing=None):
    f = run.font
    f.size = P(size)
    f.bold = bold
    f.color.rgb = color
    rpr = run._r.get_or_add_rPr()
    for tag in ("a:latin", "a:ea", "a:cs"):
        el = rpr.find(qn(tag))
        if el is None:
            el = rpr.makeelement(qn(tag), {})
            rpr.append(el)
        el.set("typeface", SANS if (tag == "a:ea" and face == MONO) else face)
    if spacing is not None:
        rpr.set("spc", str(spacing))


def text(slide, x, y, w, h, lines, size=20, face=SANS, color=BODY, bold=False, align=PP_ALIGN.LEFT,
         anchor=MSO_ANCHOR.TOP, spacing=None, line=1.25, after=0):
    tb = slide.shapes.add_textbox(x, y + DY, w, h)
    tf = tb.text_frame
    tf.word_wrap = True
    tf.auto_size = None
    tf.vertical_anchor = anchor
    for side in ("margin_left", "margin_right", "margin_top", "margin_bottom"):
        setattr(tf, side, 0)
    paras = lines if isinstance(lines, list) else [(s, {}) for s in str(lines).split("\n")]
    for i, (s, opt) in enumerate(paras):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = opt.get("align", align)
        p.line_spacing = opt.get("line", line)
        p.space_after = Pt(opt.get("after", after))
        for seg, so in (s if isinstance(s, list) else [(s, {})]):
            r = p.add_run()
            r.text = seg
            o = {**opt, **so}
            _font(r, o.get("face", face), o.get("size", size), o.get("color", color), o.get("bold", bold),
                  o.get("spacing", spacing))
    return tb


def B(s, **k):
    return (s, {"bold": True, "face": SANS_B, "color": k.pop("color", INK), **k})


def rule(slide, x, y, w, color=RULE, weight=1.0, dash=False):
    ln = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, x, y + DY, x + w, y + DY)
    ln.line.color.rgb = color
    ln.line.width = Pt(weight)
    if dash:
        ln.line.dash_style = MSO_LINE.DASH
    return ln


def box(slide, x, y, w, h, fill=CARD, radius=0.06, line=None, dash=False, shape=MSO_SHAPE.ROUNDED_RECTANGLE):
    shp = slide.shapes.add_shape(shape, x, y + DY, w, h)
    if shape == MSO_SHAPE.ROUNDED_RECTANGLE:
        shp.adjustments[0] = radius
    if fill is None:
        shp.fill.background()
    else:
        shp.fill.solid()
        shp.fill.fore_color.rgb = fill
    if line is None:
        shp.line.fill.background()
    else:
        shp.line.color.rgb = line
        shp.line.width = Pt(1.5)
        if dash:
            shp.line.dash_style = MSO_LINE.DASH
    shp.shadow.inherit = False
    return shp


def dot(slide, cx, cy, d, color, line=None):
    return box(slide, cx - d // 2, cy - d // 2, d, d, fill=color, shape=MSO_SHAPE.OVAL, line=line)


def eyebrow(slide, x, y, w, s, color=MUTED, size=15):
    return text(slide, x, y, w, I(0.35), s, size=size, face=MONO, color=color, spacing=60)


CUR = {"sec": None}
SECTIONS = [  # (번호, 이름, 한 줄 설명, 시간)
    ("01", "문제", "CES 10 ms 격자의 빈 칸과, 지금 그 칸을 채우는 방식의 한계", "0:30~2:30"),
    ("02", "방법", "두 갈래로 나눈 LSTM과, 보간 대비로 재는 채점 방식", "2:30~4:30"),
    ("03", "결과 1 · 이온 온도", "과거만 보고도 보간보다 오차가 작고, 그 계산이 식으로 적힘", "4:30~8:00"),
    ("04", "결과 2 · 회전", "회전도 오차가 작지만, 이득은 과거 CES에서만 옴", "8:00~11:00"),
    ("05", "정리", "두 주장, 한계, 그리고 다음에 잴 것", "11:00~12:00"),
]
PAGE = {"n": 0}


def title(slide, s, y=None, size=None):
    tf = slide.shapes.title.text_frame
    tf.text = s
    n = len(s)
    pt = 28 if n <= 24 else 24 if n <= 30 else 21
    for r in tf.paragraphs[0].runs:
        _font(r, SANS_B, pt / FS, WHITE, bold=True)
        r._r.get_or_add_rPr().set("cap", "none")


def source(slide, s, y=I(9.6)):
    text(slide, L, y, CW, I(0.4), "Source: " + s, size=14, face=MONO, color=MUTED)


def chrome(slide, label=True):
    text(slide, L, I(10.15), I(8), I(0.3), "KSTAR CES NOWCASTING", size=14, face=MONO, color=MUTED, spacing=60)
    s = f"{CUR['sec'][0]} · {CUR['sec'][1]}  ·  {PAGE['n']:02d}" if (label and CUR["sec"]) else f"{PAGE['n']:02d}"
    text(slide, W - L - I(8), I(10.15), I(8), I(0.3), s, size=14, face=MONO, color=MUTED, align=PP_ALIGN.RIGHT)


def stat_col(slide, x, y, w, label, big, desc, big_size=66, big_color=INK, rule_top=True, desc_size=22,
             rule_color=BODY):
    if rule_top:
        rule(slide, x, y, w, color=rule_color, weight=1.25)
    eyebrow(slide, x, y + I(0.32), w, label)
    text(slide, x, y + I(0.8), w, I(1.3), big, size=big_size, face=SANS_B, color=big_color, bold=True, line=1.0)
    text(slide, x, y + I(0.8) + P(big_size * 1.3), w, I(1.8), desc, size=desc_size, color=BODY, line=1.4)


def cols(n, gap=I(0.8), x0=None, width=None):
    x0 = L if x0 is None else x0
    width = CW if width is None else width
    w = int((width - gap * (n - 1)) / n)
    return [(x0 + i * (w + gap), w) for i in range(n)]


def row_list(slide, x, y, w, rows, head_size=26, gap=I(1.3), desc_size=20, head_color=INK):
    rule(slide, x, y, w)
    for i, (head, desc) in enumerate(rows):
        yy = y + i * gap
        text(slide, x, yy + I(0.22), w, I(0.5), head, size=head_size, face=SANS_B, color=head_color, bold=True)
        text(slide, x, yy + I(0.72), w, I(0.6), desc, size=desc_size, color=BODY)
        rule(slide, x, yy + gap, w)


def arrow(slide, x, y, w=I(0.5), color=GRAY, size=30):
    text(slide, x, y, w, I(0.6), "→", size=size, color=color, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)


def flow(slide, x, y, w, h, steps, gap=I(0.45), hot=(), hot_fill=TI_BG, hot_color=TI):
    n = len(steps)
    bw = int((w - gap * (n - 1)) / n)
    for i, (no, head, desc) in enumerate(steps):
        bx = x + i * (bw + gap)
        on = i in hot
        box(slide, bx, y, bw, h, fill=hot_fill if on else CARD, radius=0.08)
        text(slide, bx + I(0.3), y + I(0.3), bw - I(0.6), I(0.4), no, size=14, face=MONO,
             color=hot_color if on else MUTED, spacing=40)
        text(slide, bx + I(0.3), y + I(0.72), bw - I(0.6), I(0.9), head, size=24, face=SANS_B, bold=True, color=INK,
             line=1.2)
        text(slide, bx + I(0.3), y + I(1.5), bw - I(0.6), h - I(1.6), desc, size=18, color=BODY, line=1.4)
        if i < n - 1:
            arrow(slide, bx + bw, y + h // 2 - I(0.3), gap, size=24)


def band(slide, y, head, body, fill=CARD, head_color=INK, h=I(0.75)):
    box(slide, L, y, CW, h, fill=fill, radius=0.08)
    text(slide, L + I(0.4), y, CW - I(0.8), h, [([B(head + "  ", color=head_color), (body, {})], {})],
         size=22, color=BODY, anchor=MSO_ANCHOR.MIDDLE)


def notes(slide, use, key, say, facts="", qa=""):
    tf = slide.notes_slide.notes_text_frame
    parts = [f"[쓰임] {use}", f"[핵심] {key}", "[대본]\n" + say.strip()]
    if facts:
        parts.append("[숫자와 출처]\n" + facts.strip())
    if qa:
        parts.append("[예상 질문]\n" + qa.strip())
    tf.text = "\n\n".join(parts)


def new_slide(layout="제목만"):
    PAGE["n"] += 1
    return prs.slides.add_slide(LAY[layout])


def drop_placeholders(slide, keep=()):
    for ph in list(slide.placeholders):
        if ph.placeholder_format.idx not in keep:
            ph._element.getparent().remove(ph._element)


def divider(k, say):
    no, name, desc, when = SECTIONS[k]
    CUR["sec"] = (no, name)
    s = new_slide()
    drop_placeholders(s)
    eyebrow(s, L, I(3.7), I(8), f"SECTION {no}")
    text(s, L, I(4.15), CW, I(1.5), name, size=72, face=SANS_B, bold=True, color=INK, line=1.05)
    text(s, L, I(5.75), I(15), I(1.2), desc, size=26, color=MUTED, line=1.4)
    chrome(s)
    notes(s, f"구획 전환 · 3초 · 구획 시간 {when}", f"{name}: {desc}", say)
    return s


# ---------------------------------------------------------------- 장

def s_cover():
    CUR["sec"] = None
    PAGE["n"] += 1
    s = prs.slides.add_slide(LAY["제목 슬라이드"])
    drop_placeholders(s, keep=(0, 1))
    tf = s.shapes.title.text_frame
    tf.text = "KSTAR CES 나우캐스팅"
    p2 = tf.add_paragraph()
    p2.text = "10 ms마다 비는 이온 온도와 회전을 채우기"
    for p, pt, bold in ((tf.paragraphs[0], 40, True), (p2, 28, False)):
        for r in p.runs:
            _font(r, SANS_B, pt / FS, INK, bold=bold)
            r._r.get_or_add_rPr().set("cap", "none")
    sub = s.placeholders[1].text_frame
    sub.text = "그 순간까지의 정보만으로, 미래를 쓰는 보간보다 덜 틀릴 수 있을까"
    for r in sub.paragraphs[0].runs:
        _font(r, SANS, 18 / FS, BODY)
    tb = s.shapes.add_textbox(Inches(0.49), Inches(5.55), Inches(12.35), Inches(0.9))
    tf = tb.text_frame
    tf.word_wrap = True
    for i, (line, pt) in enumerate((("SEOUL NATIONAL UNIVERSITY · NUCLEAR ENGINEERING", 14),
                                    ("이승상", 18))):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = PP_ALIGN.CENTER
        r = p.add_run()
        r.text = line
        _font(r, MONO if i == 0 else SANS, pt / FS, WHITE)
    notes(s, "발표 0:00~0:30", "빈 칸을 미래 없이 채울 수 있는가, 하나의 질문",
          say="""안녕하십니까, 이승상입니다.

오늘 발표는 질문 하나로 시작합니다. 토카막에서 이온 온도와 회전을 재는 장비가 10 ms마다 값을 내는데, 그중 빈 칸이 꽤 많습니다.

그 칸을, 그 순간까지 들어온 정보만으로 채울 수 있을까요?

"그 순간까지"가 중요한 이유는 간단합니다. 실시간으로 쓰려면, 그 뒤의 측정은 아직 없거든요.

결론부터 말씀드리면, 채울 수 있습니다. 다만 이온 온도와 회전은 채워지는 이유가 서로 다릅니다. 그 차이가 오늘 발표의 뼈대입니다.""",
          facts="자료: KSTAR 2022 캠페인 641개 방전, CES 10 ms 격자.",
          qa="나우캐스팅이 예보와 무엇이 다른가 → 미래 값을 맞히는 게 아니라, 측정이 빈 '지금' 값을 추정합니다.")


def s_summary():
    CUR["sec"] = None
    s = new_slide()
    title(s, "한눈에 보기")
    data = [
        ("PROBLEM", "65 %", "회전 값의 65 %는\n새 측정이 없는 칸"),
        ("ION TEMPERATURE", "+0.22", "미래를 쓰는 보간 대비 skill\n95 % 구간 [+0.19, +0.25]"),
        ("ROTATION", "+0.31", "보간 대비 skill [+0.07, +0.70]\n이득은 과거 CES에서만"),
        ("MODEL", "20개", "파라미터 20개 식이\nLSTM 이득의 72–78 %"),
    ]
    colors = [INK, TI, VR, INK]
    for ((x, w), (lab, big, desc)), c in zip(zip(cols(4, gap=I(0.6)), data), colors):
        stat_col(s, x, I(3.4), w, lab, big, desc, big_size=56, desc_size=20, big_color=c)
    source(s, "THESIS_RESULTS.md §0 (2026-11-11) · B.27c 두 단계 부트스트랩, 새 분할 8벌, 컷 기준 · B.28 4단계")
    chrome(s, label=False)
    notes(s, "발표 0:30~1:00", "문제 하나, 주장 둘, 식 하나",
          say="""오늘 내용을 네 칸으로 먼저 보여 드리겠습니다.

왼쪽부터 문제입니다. 회전 값은 열 칸 중 여섯 칸 넘게 새 측정이 없습니다.

두 번째 칸, 이온 온도입니다. 과거만 보는 모델이, 앞뒤 측정을 다 쓰는 보간보다 오차가 작습니다.

세 번째 칸, 회전입니다. 회전도 오차가 작습니다. 그런데 그 이득은 과거 CES 값에서만 나옵니다.

마지막 칸이 제가 가장 보여 드리고 싶은 부분입니다. 35만 개 넘는 파라미터를 가진 신경망이 하는 일의 약 4분의 3이, 파라미터 20개짜리 식 하나로 적힙니다.

그럼 문제부터 보겠습니다.""",
          facts="+0.22 = +0.221 [+0.188, +0.248] (B.27c, 컷). +0.31 = +0.313 [+0.072, +0.701]. 72–78 % = B.28 4단계, 컷 4/4 분할, 미리 정한 기준 70 %.")


def s_agenda():
    CUR["sec"] = None
    s = new_slide()
    title(s, "목차")
    rule(s, L, I(2.55), CW, color=BODY, weight=1.25)
    for i, (no, name, desc, when) in enumerate(SECTIONS):
        y = I(2.55) + i * I(0.95)
        text(s, L, y + I(0.32), I(0.8), I(0.5), no, size=16, face=MONO, color=MUTED)
        text(s, L + I(0.9), y + I(0.24), I(5.6), I(0.6), name, size=26, face=SANS_B, bold=True, color=INK)
        text(s, L + I(6.6), y + I(0.3), I(8.6), I(0.6), desc, size=19, color=BODY)
        text(s, W - L - I(2.4), y + I(0.32), I(2.4), I(0.5), when, size=15, face=MONO, color=MUTED, align=PP_ALIGN.RIGHT)
        rule(s, L, y + I(0.95), CW)
    source(s, "발표 12분 · 질의응답 별도 · 수치는 THESIS_RESULTS.md §0과 8쪽 논문(docs/paper/short)")
    chrome(s, label=False)
    notes(s, "3초", "다섯 구획", say="순서는 이렇습니다. 문제, 방법, 그리고 결과 두 개, 마지막에 정리입니다.")


def s_gap():
    s = new_slide()
    title(s, "CES는 10 ms마다 재지만, 빈 칸이 많음")
    data = [
        ("T_i · MISSING", "8.2 %", "이온 온도 값이\n비어 있는 칸", TI),
        ("V_rot · MISSING", "23.9 %", "회전 값이\n비어 있는 칸", VR),
        ("V_rot · NO NEW INFO", "65 %", "빈 칸에, 직전 값을 그대로\n복사한 칸 41.1 %를 더한 것", VR),
    ]
    for (x, w), (lab, big, desc, c) in zip(cols(3), data):
        stat_col(s, x, I(3.2), w, lab, big, desc, big_size=72, big_color=c)
    band(s, I(7.9), "왜 비나", "빔이 꺼지거나 빛 알갱이(광자)가 모자란 순간에 CES 분석이 값을 내지 못함")
    source(s, "THESIS_RESULTS.md §0 · KSTAR 2022 캠페인 641개 방전 · 두 값은 서로 따로 빠짐")
    chrome(s)
    notes(s, "발표 1:00~1:40", "회전은 열 칸 중 여섯 칸 반이 새 정보가 없다",
          say="""CES라는 장비가 이온 온도와 회전을 잽니다. 10 ms마다 값 하나씩입니다.

그런데 생각보다 많이 비어 있습니다. 이온 온도는 100칸 중 8칸이 비어 있고, 회전은 24칸이 비어 있습니다.

회전은 여기서 끝이 아닙니다. 기기가 새로 재지 못하고 직전 값을 그대로 복사해 넣은 칸이 41 %나 됩니다. 둘을 합치면, 회전은 열 칸 중 여섯 칸 반에 새 정보가 없습니다.

왜 비냐면, 빔이 꺼지거나, 빛 알갱이가 모자라서 분석이 값을 내지 못하는 순간이 있거든요.

그럼 지금은 이 빈 칸을 어떻게 채우고 있을까요?""",
          facts="8.2 % / 23.9 % / 41.1 % / 65.0 % (§0, 데이터 감사). 두 타깃은 서로 독립적으로 빠짐.",
          qa="복사된 값은 어떻게 찾았나 → 직전 값과 비트 단위로 같은 값을 계측기 유지(held)로 보고 학습과 채점에서 뺐습니다.")


def s_interp():
    s = new_slide()
    title(s, "지금 방식은 앞뒤 측정을 잇는 보간, 실시간에는 뒤쪽 측정이 없음")
    x0, y0, step, d = I(2.0), I(4.1), I(2.6), I(0.8)
    labels = ["t − 30 ms", "t − 20 ms", "t − 10 ms", "t (빈 칸)", "t + 10 ms", "t + 20 ms"]
    for i, lab in enumerate(labels):
        cx = x0 + i * step + d // 2
        if i == 3:
            dot(s, cx, y0, d, WHITE, line=TI)
            text(s, cx - I(0.5), y0 - I(0.4), I(1.0), I(0.8), "?", size=34, face=SANS_B, bold=True, color=TI,
                 align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
        elif i > 3:
            dot(s, cx, y0, d, WHITE, line=GRAY)
        else:
            dot(s, cx, y0, d, INK)
        text(s, cx - I(1.2), y0 + I(0.6), I(2.4), I(0.5), lab, size=16, face=MONO,
             color=TI if i == 3 else MUTED, align=PP_ALIGN.CENTER)
    rule(s, L, I(5.6), CW)
    for (x, w), (lab, head, desc, c) in zip(cols(2), [
        ("INTERPOLATION · 지금", "보간 (PCHIP · 선형)", "빈 칸 앞뒤의 측정을 이어 채운다.\n뒤쪽 측정이 필요해서 오프라인에서만 쓸 수 있다.", MUTED),
        ("NOWCAST · 이 연구", "나우캐스트", "빈 칸 앞의 측정과, 빈 칸 없이 들어오는\n빠른 진단만으로 채운다. 실시간에 쓸 수 있다.", TI),
    ]):
        eyebrow(s, x, I(5.9), w, lab, color=c)
        text(s, x, I(6.35), w, I(0.6), head, size=30, face=SANS_B, bold=True, color=INK)
        text(s, x, I(7.05), w, I(1.6), desc, size=22, color=BODY, line=1.45)
    source(s, "보간은 미래를 쓰므로 일부러 불리한 기준으로 둠 · 실시간에 쓸 수 있는 가장 강한 비교 대상은 과거만 쓰는 GP")
    chrome(s)
    notes(s, "발표 1:40~2:30", "보간은 미래를 읽는다. 실시간에는 그 미래가 없다",
          say="""지금은 보간으로 채웁니다. 빈 칸 앞의 측정과 뒤의 측정을 선으로 잇는 방식입니다.

화면 위쪽을 봐 주십시오. 물음표 칸을 채우려면, 보간은 오른쪽 회색 점, 즉 10 ms 뒤의 측정을 씁니다.

그런데 실시간에는 그 점이 아직 없습니다. 그래서 보간은 실험이 끝난 뒤에만 쓸 수 있습니다.

저희는 반대로 했습니다. 왼쪽 측정과, 빈 칸 없이 계속 들어오는 빠른 진단만 씁니다.

여기서 저희가 일부러 기준을 불리하게 잡았습니다. 미래를 보는 보간과 비교하는 겁니다. 미래를 안 보고도 그보다 덜 틀린다면, 빠른 진단에 진짜 정보가 있다는 뜻이니까요.""",
          facts="비교 사다리: 직전 값 유지(persistence), 과거만 쓰는 GP, 선형 보간, PCHIP, 미래도 쓰는 GP.",
          qa="왜 PCHIP 을 기준으로 했나 → 실험 전에 정했고, 급한 변화에서 튀지 않는 보간이라서입니다. 선형 보간이 이 자료에선 조금 더 강한데, 선형 대비로도 이온 온도는 네 분할 모두 구간이 0을 뺍니다.")


def s_question():
    s = new_slide()
    title(s, "질문: 과거만 보고도 미래를 쓰는 보간보다 덜 틀릴 수 있는가")
    for (x, w), (lab, head, rows, fill) in zip(cols(2, gap=I(0.6)), [
        ("CAN USE", "쓸 수 있는 것", ["빠른 진단 15채널: BES · ECEI · Mirnov", "지금까지의 CES 측정", "마지막 측정 뒤 지난 시간"], TI_BG),
        ("CANNOT USE", "쓸 수 없는 것", ["빈 칸 뒤의 CES 측정", "같은 실험의 앞으로 올 값", "시험용 방전의 정답"], CARD),
    ]):
        box(s, x, I(2.9), w, I(4.6), fill=fill, radius=0.06)
        eyebrow(s, x + I(0.5), I(3.3), w - I(1.0), lab, color=TI if fill == TI_BG else MUTED)
        text(s, x + I(0.5), I(3.8), w - I(1.0), I(0.7), head, size=32, face=SANS_B, bold=True, color=INK)
        rule(s, x + I(0.5), I(4.7), w - I(1.0))
        text(s, x + I(0.5), I(4.95), w - I(1.0), I(2.4), "\n".join(rows), size=24, color=BODY, line=1.7)
    band(s, I(7.9), "판정", "보간보다 오차 제곱이 작고, 그 차이의 95 % 구간이 0을 빼야 '작다'고 말함")
    source(s, "결과 문장 규칙: 점추정 + 95 % 구간 + 0을 빼는지")
    chrome(s)
    notes(s, "발표 2:30~2:50", "입력의 경계와 판정 규칙",
          say="""정리하면 질문은 이겁니다. 과거만 보고도, 미래를 쓰는 보간보다 덜 틀릴 수 있는가.

왼쪽이 모델이 쓸 수 있는 것입니다. 빠른 진단 15채널, 지금까지의 CES 측정, 그리고 마지막 측정 뒤 얼마나 지났는지입니다.

오른쪽은 쓸 수 없는 것입니다. 빈 칸 뒤의 측정은 절대 보지 않습니다.

판정 규칙도 미리 정했습니다. 차이의 95 % 구간이 0을 빼야만 "작다"고 말합니다. 구간이 0을 포함하면, 차이를 판정할 수 없다고 적습니다.""")


def s_model():
    s = new_slide()
    title(s, "모델: 이온 온도 갈래만 빠른 진단을 봄")
    flow(s, L, I(2.75), I(11.2), I(2.5), [
        ("INPUT", "빠른 진단 · CES", "15채널 + 과거 CES + 지난 시간"),
        ("T_i BRANCH", "LSTM 2층", "160 유닛"),
        ("OUTPUT", "이온 온도", "10 ms마다 한 값"),
    ], hot=(1,), hot_fill=TI_BG, hot_color=TI)
    flow(s, L, I(5.65), I(11.2), I(2.5), [
        ("INPUT", "과거 CES만", "빠른 진단을 넣지 않음"),
        ("V_rot BRANCH", "LSTM 1층", "64 유닛"),
        ("OUTPUT", "회전", "10 ms마다 한 값"),
    ], hot=(1,), hot_fill=VR_BG, hot_color=VR)
    x, w = L + I(12.0), CW - I(12.0)
    for i, (lab, big, desc, c) in enumerate([
        ("PARAMETERS", "357,570", "학습되는 파라미터 수", INK),
        ("LATENCY", "0.97 ms", "CPU 한 스텝, 10 ms 주기 안", INK),
        ("WHY NO FAST INPUT", "−0.006", "BES와 회전 변화율의 상관", VR),
    ]):
        stat_col(s, x, I(2.75) + i * I(2.15), w, lab, big, desc, big_size=36, desc_size=17, big_color=c)
    source(s, "seq_v2 · 방전 구간 전체를 10 ms 격자로 읽는 인과 LSTM · ECEI와 회전 변화율 상관 −0.003 (§8ar)")
    chrome(s)
    notes(s, "발표 2:50~3:50", "두 갈래 구조와, 회전 갈래에서 빠른 진단을 뺀 근거",
          say="""모델은 LSTM입니다. 시간 순서대로 읽으면서 기억을 이어 가는 신경망입니다.

갈래를 둘로 나눴습니다. 위쪽 파란 갈래가 이온 온도입니다. 빠른 진단과 과거 CES를 모두 봅니다.

아래쪽 갈래가 회전입니다. 여기에는 빠른 진단을 일부러 넣지 않았습니다.

왜 뺐을까요? 오른쪽 맨 아래 숫자입니다. 빠른 진단과 회전이 변하는 속도의 상관이 거의 0입니다. 같은 방법으로 재면 이온 온도 쪽은 분명히 양수가 나오거든요. 그러니까 이건 방법이 틀려서가 아니라, 정말로 회전 정보가 없는 겁니다.

모델 크기는 파라미터 35만 개 정도이고, CPU에서 한 스텝에 1 ms가 안 걸립니다. 10 ms 주기 안에 충분히 들어갑니다.""",
          facts="입력 22개 = 빠른 진단 15 + 시간 간격 1 + 타깃별(이월 값, 경과 시간, 관측 여부)×2. 지연 중앙값 0.97 ms, p99 1.49 ms.\n"
                "운동량 식 감사: r(BES, dV/dt) = −0.006, r(ECEI, dV/dt) = −0.003 vs 에너지 식 r(BES, dT/dt) = +0.070, r(ECEI, dT/dt) = +0.078 (§8ar).",
          qa="더 큰 모델이나 다른 구조는? → 폭을 26배 범위로 바꿔도 이온 온도 skill 평균은 +0.230~+0.236으로 그대로였고, 다른 구조와 앙상블도 결론을 바꾸지 못했습니다.")


def s_score():
    s = new_slide()
    title(s, "채점: 보간보다 오차 제곱이 얼마나 줄었나")
    box(s, L, I(2.75), CW, I(1.5), fill=CARD, radius=0.08)
    text(s, L, I(2.75), CW, I(1.5), [([B("skill  =  1  −  "), ("모델 오차 제곱의 합  ÷  보간 오차 제곱의 합", {"color": INK})], {})],
         size=36, color=INK, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    for (x, w), (lab, big, desc, c) in zip(cols(3), [
        ("SKILL = 0", "0", "보간과 오차가 같다", INK),
        ("SKILL = +0.2", "+0.2", "오차 제곱이 보간보다\n20 % 작다", TI),
        ("SKILL < 0", "음수", "보간보다 오차가 크다", MUTED),
    ]):
        stat_col(s, x, I(4.8), w, lab, big, desc, big_size=60, big_color=c)
    band(s, I(8.2), "공정하게", "방전 단위로 학습·시험을 나누고, 시험 방전은 모델을 고를 때 보지 않음. 판정 기준은 실험 전에 저장소에 기록")
    source(s, "구간: 시험 방전을 통째로 다시 뽑는 부트스트랩 10,000회의 95 % 구간")
    chrome(s)
    notes(s, "발표 3:50~4:30", "skill 하나만 기억하면 된다",
          say="""결과를 읽으려면 숫자 하나만 기억하시면 됩니다. skill입니다.

화면 위의 식처럼, 모델 오차를 보간 오차와 비교한 값입니다. 0이면 보간과 같고, 플러스 0.2면 오차 제곱이 20 % 작다는 뜻입니다. 음수면 보간보다 못한 겁니다.

공정성도 챙겼습니다. 같은 방전 안의 10 ms 행들은 서로 닮아 있어서, 행 단위로 나누면 답을 미리 보는 셈이 됩니다. 그래서 방전 통째로 학습용과 시험용을 나눴습니다.

그리고 판정 기준은 실험을 돌리기 전에 먼저 적어 두었습니다.

이제 결과입니다. 먼저 이온 온도입니다.""",
          facts="평가 방식: 방전 단위 paired bootstrap 10,000회. 재현: 새로 나눈 분할 8벌에 대한 두 단계 부트스트랩(학습 반복 → 방전).",
          qa="왜 평균 ± 표준편차가 아니라 구간인가 → 분할이 넷뿐이라, 분할별 값과 구간, 그리고 몇 개가 0을 뺐는지를 적습니다(Weissgerber 2015, Agarwal 2021).")


def s_claim1():
    s = new_slide()
    title(s, "이온 온도는 과거만 보고도 보간보다 오차가 작음")
    x1, w1 = L, I(7.4)
    rule(s, x1, I(2.9), w1, color=TI, weight=2.0)
    eyebrow(s, x1, I(3.2), w1, "SKILL vs PCHIP · 95 % CI")
    text(s, x1, I(3.7), w1, I(1.9), "+0.221", size=120, face=SANS_B, bold=True, color=TI, line=1.0)
    text(s, x1, I(5.5), w1, I(0.7), "[+0.188, +0.248]", size=34, face=SANS_B, bold=True, color=INK)
    text(s, x1, I(6.3), w1, I(1.4), "미래를 쓰는 PCHIP 보간 대비.\n구간이 0을 뺀다.", size=22, color=BODY, line=1.45)
    x2, w2 = L + I(8.4), CW - I(8.4)
    eyebrow(s, x2, I(2.95), w2, "OTHER BASELINES · SAME RUNS")
    row_list(s, x2, I(3.4), w2, [
        ("직전 값 유지   +0.391", "[+0.369, +0.408] · 0을 뺀다"),
        ("과거만 쓰는 GP   +0.113", "[+0.085, +0.132] · 0을 뺀다 · 실시간 최강 비교 대상"),
        ("미래도 쓰는 GP   −0.010", "[−0.043, +0.014] · 0을 포함, 차이를 판정할 수 없음"),
    ], head_size=26, gap=I(1.35))
    source(s, "B.27c (§8by) · 새로 나눈 분할 8벌, 두 단계 부트스트랩 · 컷 기준 · 처음 고정한 네 분할 값은 +0.17~+0.26, 4/4 구간이 0을 뺌")
    chrome(s)
    notes(s, "발표 4:30~5:40", "보간 대비 +0.22, 미래도 쓰는 GP와는 구분 불가",
          say="""첫 번째 주장입니다. 이온 온도는, 과거만 보고도 보간보다 오차가 작습니다.

왼쪽 큰 숫자를 봐 주십시오. 플러스 0.22입니다. 미래를 쓰는 보간보다 오차 제곱이 약 22 % 작다는 뜻입니다. 그리고 그 아래 구간이 0을 빼니까, 우연이 아닙니다.

오른쪽은 다른 비교 대상입니다. 직전 값을 그대로 쓰는 방법보다는 훨씬 낫고, 실시간에 쓸 수 있는 가장 강한 방법인 과거만 쓰는 GP보다도 낫습니다.

마지막 줄은 솔직하게 말씀드리겠습니다. 미래까지 쓰는 GP와는 차이가 없습니다. 구간이 0을 포함하거든요. 저희 주장은 미래 없이 그 수준까지 왔다는 것이지, 그것보다 낫다는 게 아닙니다.

그럼 왜 이게 되는 걸까요?""",
          facts="B.27c(§8by): vs PCHIP +0.221 [+0.188, +0.248], persistence +0.391 [+0.369, +0.408], 인과 GP +0.113 [+0.085, +0.132], 오프라인 GP −0.010 [−0.043, +0.014].\n"
                "네 고정 분할: 컷 +0.17~+0.26, 포함 +0.23~+0.32, 8칸 모두 0 제외. 선형 보간 대비 컷 +0.136~+0.226, 포함 +0.209~+0.288, 4/4.",
          qa="컷과 포함은? → 이온 온도 3 keV를 넘는 0.53 %의 값은 스펙트럼 맞춤 실패로 보입니다. 뺀 자료(컷)와 둔 자료(포함) 둘 다에서 같은 결론일 때만 주장합니다.\n"
             "+0.22와 +0.17~+0.26의 차이는? → 앞은 평균 skill의 95 % 구간, 뒤는 고정한 네 분할 각각의 값의 범위입니다.")


def s_why1():
    s = new_slide()
    title(s, "온도가 변하는 순간은 빠른 진단에 같이 보임")
    for (x, w), (lab, big, desc, c) in zip(cols(2, gap=I(1.0)), [
        ("CORRELATION · d T_i / dt", "+0.070 / +0.078", "BES / ECEI와\n이온 온도 변화율의 상관.\n회전은 −0.006 / −0.003", TI),
        ("ABLATION · CUT, 4/4 SPLITS", "−0.25 ~ −0.43", "빠른 진단을 지웠을 때\n줄어드는 skill.\n보간 아래로 떨어진다", INK),
    ]):
        stat_col(s, x, I(2.9), w, lab, big, desc, big_size=64, big_color=c, desc_size=24)
    band(s, I(7.9), "기전", "가장자리에서 온도가 무너졌다 회복하는 사건이 BES·ECEI에 먼저 비치고, 모델은 그 흔적을 읽음",
         fill=TI_BG, head_color=TI)
    source(s, "§8ar 운동량·에너지 식 항별 감사 · §8ab 비교용 윈도 모델 절제 · 포함 기준에서는 과거 CES만으로도 +0.15~+0.23")
    chrome(s)
    notes(s, "발표 5:40~6:30", "여유는 빠른 진단에서 온다",
          say="""이유는 빠른 진단에 있습니다.

플라즈마 가장자리에서 온도가 갑자기 무너졌다가 회복하는 사건이 생기면, 그 흔적이 BES와 ECEI에 먼저 찍힙니다. CES가 비어 있어도, 모델은 온도가 방금 변했다는 걸 압니다.

왼쪽 숫자가 그 증거입니다. 빠른 진단과 이온 온도 변화율 사이에 분명한 상관이 있습니다. 같은 방법으로 회전을 재면 거의 0이 나옵니다.

오른쪽은 반대로 확인한 겁니다. 빠른 진단을 지워 봤습니다. 그러면 skill이 크게 떨어져서 보간보다 못해집니다. 네 분할 모두 그랬습니다.

즉, 보간 대비 여유는 빠른 진단에서 옵니다. 그럼 모델은 이 정보를 어떻게 쓰고 있을까요?""",
          facts="절제는 W=2 비교용 윈도 모델, 컷 모집단에서 측정(짝지은 차이 −0.25~−0.43, 4/4 유의). 포함 모집단에서는 이력만으로 +0.15~+0.23이 남음(스파이크 강건성).",
          qa="포함 기준에서는? → 맞춤 실패 값이 섞이면 과거 CES만으로도 보간 대비 여유가 남습니다. 튄 값을 덜 따라가는 효과입니다. 빠른 진단의 몫을 가르는 건 컷 기준입니다.")


def s_equation():
    s = new_slide()
    title(s, "신경망이 하는 일의 약 3/4이 파라미터 20개 식으로 적힘")
    box(s, L, I(2.6), CW, I(1.2), fill=TI_BG, radius=0.08)
    text(s, L, I(2.6), CW, I(1.2), "지금 온도  ≈  과거 측정의 가중 평균  +  Σ 계수 × ( 빠른 진단 − 그 진단의 최근 100 ms 평균 )",
         size=28, face=SANS_B, bold=True, color=INK, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    flow(s, L, I(4.2), CW, I(2.6), [
        ("01 · 3 PARAMETERS", "가중 평균", "측정이 오래 비면 새 측정을 더 믿는다.\n믿는 정도 0.28 → 0.51, 약 42 ms"),
        ("02 · 16 PARAMETERS", "변화에 반응", "빠른 진단이 평소에서 벗어난 만큼 더한다.\nBES +50, ECEI +48 eV/σ"),
        ("03 · 1 PARAMETER", "튐은 무시", "평소보다 3σ 넘게 튀면\n그 채널은 반영하지 않는다"),
    ], hot=(0, 1, 2), hot_fill=CARD, hot_color=TI)
    for (x, w), (lab, big, desc) in zip(cols(2, gap=I(1.0)), [
        ("SHARE OF LSTM GAIN · 4/4 SPLITS", "72–78 %", "LSTM의 보간 대비 이득 중 식이 내는 몫 (미리 정한 기준 70 %)"),
        ("PARAMETERS", "357,570 → 20", "나머지 22–28 %는 LSTM의 비선형 사건 처리"),
    ]):
        stat_col(s, x, I(7.15), w, lab, big, desc, big_size=40, desc_size=18, rule_top=True)
    chrome(s)
    notes(s, "발표 6:30~8:00", "LSTM은 형태를 찾았고, 식은 그 형태를 20개 숫자로 적는다",
          say="""여기가 제가 가장 보여 드리고 싶은 장입니다.

신경망은 보통 블랙박스라고 하죠. 저희는 학습이 끝난 LSTM의 입력과 출력 관계를 직접 재 봤습니다. 그랬더니 대부분이 위쪽 식 하나로 적혔습니다.

세 조각입니다. 첫째, 과거 측정의 가중 평균입니다. 측정이 오래 비어 있었으면 새 측정을 더 믿습니다. 칼만 필터와 같은 구조입니다.

둘째, 빠른 진단이 평소보다 갑자기 변하면, 그만큼 온도를 올리거나 내립니다. 여기서 "최근 100 ms 평균"은 저희가 정한 값이 아닙니다. LSTM의 반응을 재 보니 그 모양이 나왔습니다.

셋째, 3σ 넘게 튀는 값은 무시합니다.

이 식을 처음부터 다시 학습시키면, LSTM이 보간보다 나아진 몫의 72에서 78 %가 나옵니다. 실험 전에 정한 기준이 70 %였는데, 네 분할 모두 넘었습니다.

즉, 35만 개 파라미터 중 대부분은 이 형태를 찾는 데 쓰였고, 형태를 알면 20개로 4분의 3이 나옵니다.

이제 회전입니다.""",
          facts="B.28 4단계(§8bz-4): K₀ ≈ 0.28, K∞ ≈ 0.51, s_K ≈ 42 ms, θ ≈ 3.0σ, ΣBES ≈ +50, ΣECEI ≈ +48 eV/σ (컷, 4분할 평균). "
                "파라미터 20 = K 3 + 채널 계수 15 + θ 1 + 상수 1. 고정한 구조 상수(기준선 10스텝, 차단 폭 0.5)는 LSTM의 반응에서 읽음. "
                "처음 형태(10 ms 증분)는 50–66 %로 기준 미달, 형태를 고친 뒤 통과. persistence 대비로는 87–90 %. "
                "LSTM 망각 게이트 기하평균 0.465–0.482, 기억 약 1.3스텝.",
          qa="그럼 LSTM은 왜 필요했나 → 식의 형태(100 ms 기준선, 3σ 차단)를 정할 근거가 LSTM의 반응이었습니다. 처음 짐작한 형태는 기준을 넘지 못했습니다.")


def s_claim2():
    s = new_slide()
    title(s, "회전도 오차가 작지만, 이득은 과거 CES에서만 옴")
    x1, w1 = L, I(7.4)
    rule(s, x1, I(2.9), w1, color=VR, weight=2.0)
    eyebrow(s, x1, I(3.2), w1, "SKILL vs PCHIP · 95 % CI")
    text(s, x1, I(3.7), w1, I(1.9), "+0.313", size=120, face=SANS_B, bold=True, color=VR, line=1.0)
    text(s, x1, I(5.5), w1, I(0.7), "[+0.072, +0.701]", size=34, face=SANS_B, bold=True, color=INK)
    text(s, x1, I(6.3), w1, I(1.4), "평균 제곱 오차, 구간이 0을 뺀다.\n구간이 넓다.", size=22, color=BODY, line=1.45)
    x2, w2 = L + I(8.4), CW - I(8.4)
    eyebrow(s, x2, I(2.95), w2, "NO 10 ms ROTATION INFO IN FAST DIAGNOSTICS")
    row_list(s, x2, I(3.4), w2, [
        ("−0.006 / −0.003", "BES / ECEI와 회전 변화율의 상관"),
        ("24 / 24 코일", "100 Hz 자기 코일 값 = 원시 표본 한 점, 평균이 아님"),
        ("−0.128  [−0.544, −0.105]", "원시 2 MHz 자기 코일 특징을 넣으면 skill이 준다"),
        ("0.34 %", "가열 빔(NBI)이 움직이는 느린 대역의 몫"),
    ], head_size=24, gap=I(1.12), desc_size=18, head_color=VR)
    source(s, "B.27c (§8by) 컷 · 포함 +0.319 [+0.046, +0.683] · 원시 자료 §8bd: 컷, 시험 방전 3개 · NBI §8bd 부록")
    chrome(s)
    notes(s, "발표 8:00~9:30", "회전은 평균 오차가 작다. 그러나 빠른 진단에는 10 ms 회전 정보가 없다",
          say="""두 번째 주장, 회전입니다.

회전도 보간보다 오차가 작습니다. 플러스 0.31이고, 구간이 0을 뺍니다. 다만 구간이 넓다는 점은 같이 말씀드립니다.

그런데 이온 온도와 다른 점이 있습니다. 이 이득은 과거 CES 값에서만 나옵니다.

오른쪽이 그 근거입니다. 첫 줄, 빠른 진단과 회전 변화율의 상관이 거의 0입니다.

혹시 자기 코일 신호를 더 잘 쓰면 되지 않을까요? 저희도 그렇게 생각해서 원시 자료를 따로 받았습니다. 그랬더니 100 Hz 값은 평균이 아니라 원시 신호에서 한 점을 찍은 값이었습니다. 원시 신호로 특징을 만들어 넣어 봤더니, 오히려 skill이 떨어졌습니다.

그럼 가열 빔은요? 빔은 켜고 끄는 계단 입력이라, 10 ms 변화에는 0.34 %밖에 영향을 주지 못합니다. 100 ms 이상 앞을 내다볼 때의 레버입니다.

정리하면, 회전의 10 ms 변화를 알려 주는 신호가 지금 입력에는 없습니다.""",
          facts="B.27c: 컷 +0.313 [+0.072, +0.701], 포함 +0.319 [+0.046, +0.683]. 겹 단위 겹침 보정 구간 [−0.02, +0.41]은 0 포함 → 주장은 등록된 두 단계 판정으로 세움. 네 고정 분할 관문은 컷 1/4, 포함 2/4.\n"
                "원시 Mirnov(B.6, §8bd): 24/24 코일, 특징 15개, 학습 방전 6개, 시험 방전 3개, 컷 −0.128 [−0.544, −0.105], 포함 −0.164 [−0.473, −0.133], 0/3 방전 개선. "
                "모드 주파수와 회전의 방전 내 상관 |ρ| ≤ 0.11 (7개 중 6개). NBI: 300 ms 이동평균 대역 = 10 ms 증분 분산의 0.34 % (20개 방전).",
          qa="NBI 신호가 있으면 회전이 좋아지나 → 10 ms 나우캐스트에는 아닙니다. 방전 간 수준이나 100 ms 이상 예보에 쓰입니다.")


def s_median():
    s = new_slide()
    title(s, "평범한 순간의 회전 오차는 측정 잡음이 바닥")
    # 지어낸 숫자 예시 표
    x1, w1 = L, I(8.6)
    eyebrow(s, x1, I(2.75), w1, "MADE-UP EXAMPLE · NOT DATA")
    heads = ["방법", "95행 오차", "5행 오차", "중앙값", "평균 제곱"]
    widths = [I(2.4), I(1.5), I(1.5), I(1.4), I(1.8)]
    rows = [heads, ["직전 값 유지", "1", "20", "1", "약 21"], ["모델", "1", "10", "1", "약 6"]]
    yy = I(3.2)
    rule(s, x1, yy, w1, color=BODY, weight=1.25)
    for r, row in enumerate(rows):
        xx = x1
        for c, (cell, cw) in enumerate(zip(row, widths)):
            text(s, xx, yy + I(0.18), cw, I(0.5), cell, size=19 if r == 0 else 24,
                 face=MONO if r == 0 else (SANS_B if c in (3, 4) else SANS), bold=(r > 0 and c in (3, 4)),
                 color=MUTED if r == 0 else INK, align=PP_ALIGN.LEFT if c == 0 else PP_ALIGN.RIGHT)
            xx += cw
        yy += I(0.85)
        rule(s, x1, yy, w1)
    text(s, x1, yy + I(0.3), w1, I(1.6), "중앙값은 같은데 평균 제곱은 줄어든다.\n'평균 오차가 준다'는 '모든 순간에 더 정확하다'가 아니다.",
         size=21, color=BODY, line=1.5)
    x2, w2 = L + I(9.6), CW - I(9.6)
    for i, (lab, big, desc, c) in enumerate([
        ("PHYSICS FILTER · V_rot", "0.94–1.00×", "직전 값 유지 대비, 3/4 분할에서 1 포함", VR),
        ("LSTM · V_rot MEDIAN", "4.3 vs 3.7", "km/s, LSTM 대 직전 값 유지", INK),
        ("PHYSICS FILTER · T_i", "0.91×", "이온 온도는 평범한 순간에도 더 정확", TI),
    ]):
        stat_col(s, x2, I(2.75) + i * I(2.15), w2, lab, big, desc, big_size=40, desc_size=18, big_color=c)
    source(s, "§8bp 이중 판독 · §8ax · 온도/속도 흔들림 비가 8/8 칸에서 가우스 선 맞춤의 루트 2 → 광자 잡음")
    chrome(s)
    notes(s, "발표 9:30~11:00", "평균과 중앙값은 다른 질문이다. 회전의 중앙값은 광자 잡음 바닥",
          say="""그럼 회전은 평소에도 더 정확할까요? 여기서 평균과 중앙값을 구분해야 합니다.

왼쪽 표는 지어낸 숫자입니다. 100번 중 95번은 두 방법 모두 1만큼 틀리고, 나머지 5번에서 직전 값 유지는 20, 모델은 10만큼 틀린다고 해 보겠습니다. 중앙값은 둘 다 1입니다. 그런데 평균 제곱 오차는 21에서 6으로 줄어듭니다.

즉 평균 오차가 준다는 건, 크게 틀리는 순간에 덜 틀린다는 뜻이지, 모든 순간에 더 정확하다는 뜻이 아닙니다.

회전이 정확히 이 모양입니다. 오른쪽 위를 봐 주십시오. 평범한 순간을 맡는 물리 필터가 직전 값 유지와 거의 같습니다. LSTM은 오히려 조금 더 틀립니다.

왜 그럴까요? 10 ms 동안의 회전 변화 대부분이 측정에 쓰인 빛 알갱이 수가 유한해서 생기는 잡음이기 때문입니다. 더 내려갈 여지가 없는 바닥입니다.

반대로 맨 아래 이온 온도는, 평범한 순간에도 직전 값보다 정확합니다. 두 물리량의 차이가 여기서도 보입니다.""",
          facts="물리 필터 = 유지-또는-점프 과정 + Student-t 측정 잡음(학습 본류 척도). V_rot 0.938–1.003 분할별, T_i 0.906–0.920 (8/8). "
                "LSTM V_rot 중앙값 행 4.3 vs persistence 3.7 km/s (컷). 한 estimator가 평균과 중앙값을 함께 낮추지 못함: 물리 필터의 평균 제곱 오차 skill은 LSTM 대비 −0.06~−0.52.",
          qa="중앙값을 왜 따로 재나 → 평균 제곱 오차는 상위 1 % 행이 이온 온도 제곱 오차의 48 %(컷)를 차지할 만큼 소수 행이 좌우합니다. 평범한 순간의 정확도는 중앙값으로만 보입니다.")


def s_summary_end():
    CUR["sec"] = ("05", "정리")
    s = new_slide()
    title(s, "정리: 물리량마다 주장 하나씩")
    for (x, w), (lab, head, body, c, fill) in zip(cols(2, gap=I(0.6)), [
        ("CLAIM 1 · ION TEMPERATURE", "과거만 보고도 보간보다 오차가 작음",
         "보간 대비 +0.221 [+0.188, +0.248]\n여유는 빠른 진단 BES·ECEI에서 옴\n그 계산은 파라미터 20개 식으로 3/4", TI, TI_BG),
        ("CLAIM 2 · ROTATION", "평균 오차도 작지만, 과거 CES에서만",
         "보간 대비 +0.313 [+0.072, +0.701]\n빠른 진단에 10 ms 회전 정보가 없음\n평범한 순간은 광자 잡음이 바닥", VR, VR_BG),
    ]):
        box(s, x, I(2.75), w, I(4.4), fill=fill, radius=0.06)
        eyebrow(s, x + I(0.5), I(3.15), w - I(1.0), lab, color=c)
        text(s, x + I(0.5), I(3.65), w - I(1.0), I(1.2), head, size=30, face=SANS_B, bold=True, color=INK, line=1.25)
        rule(s, x + I(0.5), I(4.95), w - I(1.0))
        text(s, x + I(0.5), I(5.2), w - I(1.0), I(1.9), body, size=22, color=BODY, line=1.6)
    band(s, I(7.55), "한계", "한 장치·한 해(KSTAR 2022) 자료 · 미래도 쓰는 GP와는 구분 불가 · 노출마다의 측정 오차 값이 파일에 없음")
    band(s, I(8.5), "다음", "다른 캠페인 자료로 전이 검정 · CES 분석의 노출별 불확도를 받으면 회전 중앙값 바닥 아래를 시험", fill=WHITE)
    source(s, "THESIS_RESULTS.md §0 · 8쪽 논문 docs/paper/short/main_short_ko.tex")
    chrome(s)
    notes(s, "발표 11:00~12:00", "두 주장, 세 한계, 다음에 잴 것",
          say="""정리하겠습니다.

첫째, 이온 온도입니다. 과거만 보고도 보간보다 오차가 작습니다. 그 여유는 빠른 진단에서 오고, 신경망이 하는 계산의 4분의 3이 파라미터 20개짜리 식으로 적힙니다.

둘째, 회전입니다. 평균 오차는 역시 작습니다. 다만 그 이득은 과거 CES에서만 나오고, 평범한 순간은 측정 잡음이 바닥입니다.

한계도 분명히 말씀드립니다. 자료는 한 장치의 한 해 실험입니다. 미래까지 쓰는 가장 강한 방법과는 차이를 판정할 수 없습니다. 그리고 노출마다의 측정 오차 값이 파일에 없어서, 회전의 바닥 아래로는 내려갈 수 없었습니다.

그래서 다음에 잴 것도 정해져 있습니다. 다른 해의 자료로 같은 결론이 나오는지, 그리고 CES 분석에서 노출별 불확도를 받으면 회전의 바닥을 넘을 수 있는지입니다.

들어 주셔서 감사합니다.""",
          qa="이게 새로운 방법인가 → 모델은 표준 LSTM입니다. 결측 시각을 인과적으로 채우는 이 조합의 선례는 찾지 못했다고만 말하고, 없다고 하지 않습니다.\n"
             "실시간 제어에 바로 쓸 수 있나 → 계산 시간은 10 ms 안에 들어갑니다. 다만 다른 캠페인 자료로 검정하기 전입니다.")


# ---------------------------------------------------------------- 조립

s_cover()
s_summary()
s_agenda()
divider(0, "먼저 문제입니다.")
s_gap()
s_interp()
s_question()
divider(1, "다음은 방법입니다.")
s_model()
s_score()
divider(2, "결과입니다. 이온 온도부터 보겠습니다.")
s_claim1()
s_why1()
s_equation()
divider(3, "두 번째 결과, 회전입니다.")
s_claim2()
s_median()
s_summary_end()

prs.save(OUT)
print("saved", len(prs.slides), "slides")
