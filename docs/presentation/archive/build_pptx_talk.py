"""KSTAR CES 나우캐스팅 발표(대본 포함). 핵플실 발표 템플릿(1조, 수정_5) 위에 짓는다.

  py docs/presentation/build_pptx_talk.py      -> docs/presentation/기말발표/KSTAR_CES_발표_v4.pptx

* 템플릿: 핵플실발표자료_1조_김하빈, 이승상_수정_5.pptx 에서는 배경 모양(위쪽 세 색 띠, 남색 제목 띠, 표지 남색 블록)만 쓴다.
  글꼴(Pretendard · IBM Plex Mono)·색·배치는 v1 그대로.
* 본문 배치 문법은 Cellfie 기말발표 v5.17 에서 왔다: 큰 숫자 · 짧은 설명 · 출처 한 줄 · 회색 카드.
* 내용: THESIS_RESULTS.md §0 (2026-11-11)과 8쪽 논문 docs/paper/short/main_short_ko.tex 의 수치만 쓴다.
* 발표자 노트 = 쓰임·시간 / 핵심 / 대본(발표 말투) / 숫자와 출처 / 예상 질문.
* 용어: 파라미터, estimator 등 흔한 영어 용어. 결과는 점추정 + 95 % 구간 + 0 제외 여부.
"""
import io
import re
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
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "기말발표" / "KSTAR_CES_발표_v4.pptx"
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
# 글씨는 Cellfie 처럼 검정·회색만. 두 물리량도 색으로 가르지 않는다.
TI = INK
VR = INK
TI_BG = CARD
VR_BG = CARD

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
SECTIONS = [  # (번호, 이름, 한 줄 설명, 시간) -- 8쪽 논문의 절 순서
    ("01", "서론", "CES 10 ms 격자의 빈 칸, 그리고 그 칸을 미래 없이 채울 수 있는가", "약 2분"),
    ("02", "자료", "KSTAR 2022 캠페인 641개 방전과 두 가지 자료 감사", "약 2분"),
    ("03", "모델", "두 보존식이 정한 두 갈래 LSTM, 학습, 그리고 그것이 계산하는 식", "약 6분"),
    ("04", "평가 방법", "보간 대비 skill, 방전 단위 분할과 재현", "약 2분"),
    ("05", "결과", "이온 온도와 회전, 두 주장과 그 근거", "약 9분"),
    ("06", "결론", "두 주장, 한계, 다음에 잴 것", "약 1분"),
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
    """출처 줄은 화면에 두지 않는다(발표자 노트의 '숫자와 출처'에 있다)."""
    return None


def chrome(slide, label=True):
    """아래 바닥 줄(덱 이름 · 쪽 번호)은 두지 않는다."""
    return None

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


def _dur(use):
    m = re.search(r"발표 (\d+):(\d+)~(\d+):(\d+)", use)
    if not m:
        return use
    a, b, c, d = map(int, m.groups())
    sec = (c * 60 + d) - (a * 60 + b)
    return re.sub(r"발표 \d+:\d+~\d+:\d+", f"약 {sec}초", use)


def notes(slide, use, key, say, facts="", qa=""):
    use = _dur(use)
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

분할과 구간을 어떻게 잡았는지 한 장 더 보겠습니다.""",
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

즉, 보간 대비 여유는 빠른 진단에서 옵니다. 그럼 이 결과가 조건을 바꿔도 남는지 보겠습니다.""",
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

그럼 모델을 더 키우면 어떻게 될까요? 다음 장입니다.""",
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
    CUR["sec"] = ("06", "결론")
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



# ---------------------------------------------------------------- 그림 · 표

FIGS = HERE.parents[1] / "docs" / "paper" / "figures"
ASSET = HERE / "figures" / "talk"


def fig_png(name):
    """논문 그림(pdf 또는 png)을 발표용 png 로. pdf 는 PyMuPDF 로 200 dpi 렌더."""
    ASSET.mkdir(parents=True, exist_ok=True)
    src_png, src_pdf = FIGS / f"{name}.png", FIGS / f"{name}.pdf"
    if src_png.exists():
        return src_png
    out = ASSET / f"{name}.png"
    if not out.exists() or out.stat().st_mtime < src_pdf.stat().st_mtime:
        import fitz
        doc = fitz.open(src_pdf)
        doc[0].get_pixmap(dpi=200).save(out)
    return out


def pic(slide, path, x, y, w, h):
    """상자 (x, y, w, h) 안에 비율을 지켜 가운데 놓는다."""
    from PIL import Image
    iw, ih = Image.open(path).size
    k = min(w / iw, h / ih)
    pw, ph = int(iw * k), int(ih * k)
    slide.shapes.add_picture(str(path), x + (w - pw) // 2, y + DY + (h - ph) // 2, pw, ph)


def table(slide, x, y, widths, rows, size=20, head_size=17, row_h=I(0.72), bold_cols=(), align=None):
    """Cellfie 표: 굵은 윗줄, 고정폭 머리줄, 행마다 가는 줄. 글씨는 검정·회색."""
    w = sum(widths)
    rule(slide, x, y, w, color=BODY, weight=1.25)
    yy = y
    for r, row in enumerate(rows):
        xx = x
        for c, (cell, cw) in enumerate(zip(row, widths)):
            al = (align[c] if align else (PP_ALIGN.LEFT if c == 0 else PP_ALIGN.LEFT))
            bold = r > 0 and (c in bold_cols or (isinstance(cell, str) and cell.startswith("**")))
            cell = cell[2:] if isinstance(cell, str) and cell.startswith("**") else cell
            text(slide, xx + I(0.12), yy + I(0.16), cw - I(0.24), row_h - I(0.1), cell,
                 size=head_size if r == 0 else size, face=MONO if r == 0 else (SANS_B if bold else SANS),
                 bold=bold, color=MUTED if r == 0 else INK, align=al, line=1.2)
            xx += cw
        yy += row_h
        rule(slide, x, yy, w)
    return yy


# ---------------------------------------------------------------- 02 자료

def s_data():
    s = new_slide()
    title(s, "자료: KSTAR 2022 캠페인 641개 방전, 10 ms 격자")
    row_list(s, L, I(2.75), I(7.6), [
        ("641개 방전 · 247,207행", "shot 30801–32751, 10 ms 격자로 정렬"),
        ("빠른 진단 15채널", "BES 9 · ECEI 4 · Mirnov 코일 2"),
        ("CES 타겟 2개", "이온 온도 T_i, 토로이달 회전 V_rot"),
        ("연속 세그먼트", "0.5 s 넘는 공백에서 나누고, 어떤 방법도 공백을 넘지 않음"),
    ], head_size=28, gap=I(1.55), desc_size=21)
    box(s, L + I(8.4), I(2.75), CW - I(8.4), I(6.6), fill=CARD, radius=0.04)
    pic(s, fig_png("fig_missing"), L + I(8.7), I(3.0), CW - I(9.0), I(5.4))
    text(s, L + I(8.7), I(8.55), CW - I(9.0), I(0.7), "CES 결측: T_i 8.2 %, V_rot 23.9 %, 서로 따로 빠짐",
         size=19, color=MUTED)
    notes(s, "약 60초", "자료의 크기와 구성",
          say="""자료부터 보겠습니다. KSTAR 2022년 캠페인의 방전 641개입니다. 10 ms 격자로 맞추면 약 25만 행이 됩니다.

각 행에는 빠른 진단 15채널이 들어 있습니다. BES 9개, ECEI 4개, 자기 코일 2개입니다. 그리고 맞혀야 할 값, CES의 이온 온도와 회전이 있습니다.

오른쪽 그림이 CES가 비는 모습입니다. 이온 온도와 회전은 서로 따로 빠집니다. 그래서 둘을 한 모델로 학습할 때, 한쪽만 있는 행도 버리지 않고 씁니다. 이건 모델 절에서 다시 말씀드리겠습니다.

자료를 쓰기 전에 두 가지를 먼저 점검했습니다.""",
          facts="8쪽 논문 §자료: 641개 파일, 247,207행, BES 9 · ECEI 4 · Mirnov 2, 세그먼트는 0.5 s 이상 공백에서 나눔.")


def s_audit():
    s = new_slide()
    title(s, "자료 감사: 복사된 회전 값과 맞춤 실패한 온도 값")
    for (x, w), (lab, big, desc, after) in zip(cols(2, gap=I(0.6)), [
        ("AUDIT 1 · HELD V_rot", "54 %", "관측된 회전 값 가운데\n직전 값을 비트 그대로 반복한 값\n최대 1,214행 연속 · 641개 중 499개 파일",
         "학습·채점·기준선 어디에서도 빼고 쓴다"),
        ("AUDIT 2 · FIT FAILURE", "0.53 %", "3 keV를 넘는 이온 온도 1,197행\n스펙트럼 맞춤 실패, 런의 85 %가 한 행",
         "뺀 자료(컷)와 둔 자료(포함) 둘 다에서 판정한다"),
    ]):
        box(s, x, I(2.75), w, I(5.1), fill=CARD, radius=0.05)
        eyebrow(s, x + I(0.5), I(3.15), w - I(1.0), lab)
        text(s, x + I(0.5), I(3.65), w - I(1.0), I(1.2), big, size=64, face=SANS_B, bold=True, color=INK, line=1.0)
        text(s, x + I(0.5), I(4.95), w - I(1.0), I(1.9), desc, size=21, color=BODY, line=1.45)
        rule(s, x + I(0.5), I(6.85), w - I(1.0))
        text(s, x + I(0.5), I(7.05), w - I(1.0), I(0.6), after, size=21, face=SANS_B, bold=True, color=INK)
    band(s, I(8.3), "문턱 민감도", "2.5–4 keV로 바꿔도 평균 T_i skill +0.230 · +0.232 (3 keV에서 +0.236)")
    notes(s, "약 60초", "공짜 점수가 생기는 두 구멍을 막았다",
          say="""첫째, 회전 값의 절반 이상이 복사된 값이었습니다. 관측된 회전 값의 54 %가 직전 값을 비트 그대로 반복합니다. 길게는 1,214행 연속입니다.

이걸 그냥 두면 어떻게 될까요? 직전 값을 그대로 쓰는 방법이 공짜로 정답을 맞히게 됩니다. 그래서 학습에서도, 채점에서도, 비교 대상에서도 전부 뺐습니다.

둘째, 이온 온도 중 0.53 %가 3 keV를 넘습니다. 스펙트럼 맞춤이 실패한 값으로 보입니다. 그런데 파일만 보고는 확정할 수 없거든요. 그래서 뺀 자료와 둔 자료, 두 가지로 모두 채점합니다. 두 쪽에서 결론이 같을 때만 주장합니다.

문턱을 2.5에서 4 keV로 바꿔도 결과는 거의 그대로였습니다.""",
          facts="유지값: 관측 V_rot의 54 % (격자 기준 41.1 %), T_i 유지값은 226,991행 중 1행. 맞춤 실패 1,197행(0.53 %), 274개 방전의 951개 런.",
          qa="컷과 포함 중 어느 쪽이 진짜인가 → 정할 수 없어서 둘 다 1차로 둡니다. 포함 쪽 수치가 더 높게 나오는데, 튄 값이 보간의 앵커를 오염시키기 때문입니다.")


# ---------------------------------------------------------------- 03 모델

def s_physics():
    s = new_slide()
    title(s, "왜 두 갈래인가: 두 보존식은 변화율을 묶음")
    box(s, L, I(2.75), CW, I(2.3), fill=CARD, radius=0.05)
    text(s, L + I(0.6), I(3.0), CW - I(1.2), I(1.9), [
        ([("에너지     ", {"face": MONO, "size": 20, "color": MUTED}),
          ("∂t(3/2 nᵢTᵢ) + ∇·qᵢ  =  Q_ei (Tₑ − Tᵢ) + Sᵢ", {})], {}),
        ([("운동량     ", {"face": MONO, "size": 20, "color": MUTED}),
          ("∂t(nᵢmᵢ⟨R²⟩ω_φ) + ∇·Π_φ  =  T_NBI + T_NTV + T_int + …", {})], {"after": 6}),
    ], size=30, face=SANS_B, bold=True, color=INK, line=1.7)
    table(s, L, I(5.55), [I(5.2), I(4.2), I(4.2)], [
        ["빠른 진단", "d T_i / dt", "d V_rot / dt"],
        ["BES (밀도 요동)", "**+0.070", "−0.006"],
        ["ECEI (전자 온도)", "**+0.078", "−0.003"],
    ], size=24, row_h=I(0.85))
    text(s, L + I(14.2), I(5.75), CW - I(14.2), I(2.4),
         "두 식 모두 값이 아니라\n변화율을 묶는다.\n그래서 변화율과의\n상관을 잰다.", size=21, color=BODY, line=1.5)
    band(s, I(8.6), "결론", "같은 측정이 에너지식의 결합은 잡으므로, 회전 쪽 0은 방법의 한계가 아님 → 회전 갈래에서 빠른 진단을 뺀다")
    notes(s, "약 70초", "라우팅은 취향이 아니라 두 보존식을 항별로 읽은 결과",
          say="""모델 설명은 왜 이렇게 만들었는지부터 시작하겠습니다.

이온 온도와 회전은 각각 보존식을 따릅니다. 위가 에너지, 아래가 운동량입니다. 여기서 중요한 건, 두 식 모두 값이 아니라 변하는 속도를 묶는다는 점입니다.

그래서 저희는 빠른 진단이 온도나 회전 값과 상관이 있는지가 아니라, 변하는 속도와 상관이 있는지를 쟀습니다.

표를 봐 주십시오. 이온 온도의 변화율과는 BES, ECEI 모두 양의 상관이 있습니다. 회전의 변화율과는 거의 0입니다.

같은 방법이 한쪽에서는 신호를 잡으니까, 회전 쪽 0은 방법이 둔해서가 아닙니다. 정말로 정보가 없는 겁니다. 그래서 회전 갈래에는 빠른 진단을 넣지 않았습니다.""",
          facts="값 수준 상관: T_i +0.341 / +0.311, V_rot +0.027 / +0.005. 641개 방전 전수, 624개 연속 블록 (§8ar).")


def s_torque():
    s = new_slide()
    title(s, "운동량식 항별 감사: 우리 입력이 회전 변화에 닿는 길이 없음")
    table(s, L, I(2.75), [I(4.6), I(7.6), I(5.2)], [
        ["운동량식의 항", "우리 자료에 있는가", "판정"],
        ["∂L/∂t", "연속한 관측 쌍", "측정됨"],
        ["T_NBI (빔 토크)", "0-D 채널 없음 · 계단 입력, 10 ms 증분 분산의 0.34 %", "10 ms 레버 아님"],
        ["T_NTV ~ δB²", "Mirnov 100 Hz 한 점 표본 · 원시 2 MHz 특징 검정", "음성으로 종결"],
        ["∇·Π_φ (난류 유속)", "BES 밀도 요동", "측정됨, r ≈ 0"],
        ["T_int (잔여 응력)", "∇Tᵢ 필요, 스칼라만 보유", "도달 불가"],
    ], size=22, row_h=I(0.95))
    band(s, I(8.85), "T_i 쪽 길", "충돌성 전자–이온 결합 Q_ei(Tₑ − Tᵢ): ECEI가 Tₑ를, BES가 nₑ 구조를 공급한다")
    notes(s, "약 50초", "회전 변화에 닿는 항이 하나도 없다",
          say="""운동량식의 항을 하나씩 저희 자료에 대 봤습니다.

빔 토크는 채널이 없습니다. 있더라도 빔은 켜고 끄는 계단이라, 10 ms 변화에는 0.34 %밖에 기여하지 못합니다.

자기 섭동 항은 Mirnov로 봐야 하는데, 100 Hz 값은 한 점 표본이었고, 원시 신호로 다시 해 봐도 음성이었습니다.

난류 운동량 유속은 BES로 볼 수 있는데, 상관이 0입니다. 마지막 잔여 응력은 온도 기울기 분포가 필요한데 저희에겐 스칼라 값뿐입니다.

정리하면, 저희 입력이 회전 변화에 닿는 길이 없습니다. 반대로 이온 온도는 전자와 이온이 충돌로 에너지를 주고받는 항이 있어서, ECEI와 BES가 거기에 닿습니다.""",
          facts="표는 8쪽 논문 tab:torque를 최신 기록(§8bd, NBI 부록)으로 채운 것.")


def s_scales():
    s = new_slide()
    title(s, "시간 척도: 10 ms 격자가 담을 수 있는 것과 없는 것")
    for (x, w), (lab, big, desc) in zip(cols(3), [
        ("NYQUIST · Δt = 10 ms", "50 Hz", "격자가 담을 수 있는 한계.\n모드 회전(약 4 · 8 kHz)은\n담을 수 없다"),
        ("MIRNOV LAG-1", "−0.009", "100 Hz Mirnov의 자기상관.\n같은 격자에서 BES는 +0.568"),
        ("τ_eq NEAR T_i MEDIAN", "8–59 ms", "전자–이온 평형 시간.\n측정한 문맥 포화 약 50 ms와\n같은 자릿수"),
    ]):
        stat_col(s, x, I(2.75), w, lab, big, desc, big_size=64)
    band(s, I(8.3), "자기상관 1/e 시간", "T_i 159 ms · BES 161 · ECEI 147로 서로 맞음. V_rot는 이 자료에서 잴 수 없음")
    notes(s, "약 50초", "격자 간격이 Mirnov를 막고, 평형 시간이 문맥 길이를 정한다",
          say="""시간 척도를 보면 두 가지가 설명됩니다.

첫째, 10 ms 격자는 50 Hz까지만 담을 수 있습니다. 그런데 회전과 관련된 모드는 수 kHz에서 돕니다. 그러니 100 Hz로 찍은 자기 코일 값은 그 회전을 담을 수 없습니다. 실제로 자기상관이 거의 0입니다. 같은 격자에서 BES는 0.57이거든요.

둘째, 전자와 이온이 온도를 맞추는 데 걸리는 시간이 8에서 59 ms입니다. 모델이 과거를 얼마나 보면 충분한지 쟀더니 50 ms에서 포화했습니다. 같은 자릿수입니다.

즉, 모델이 보는 시간 길이도 물리와 맞아떨어집니다.""",
          facts="τ_eq는 Braginskii τ_e로부터 T_e 0.5–3 keV, n_e 2–5×10¹⁹ m⁻³에서 8.3–305 ms, 관측 T_i 중앙값 593 eV 근방에서 8–59 ms. 자릿수 수준의 정합성이며 n_e, T_e의 측정은 아님.")


def s_arch():
    s = new_slide()
    title(s, "구조: 입력 22채널, 두 갈래 LSTM")
    box(s, L, I(2.65), CW, I(5.9), fill=CARD, radius=0.03)
    pic(s, fig_png("fig_arch_seq_v2_ko"), L + I(0.3), I(2.8), CW - I(0.6), I(5.6))
    text(s, L, I(8.8), CW, I(1.2),
         "T_i 갈래: 22채널 전부, 2층 160 유닛  ·  V_rot 갈래: 느린 7채널, 1층 64 유닛  ·  끝은 LayerNorm + Linear–GELU–Linear  ·  357,570 파라미터",
         size=20, color=BODY, line=1.4)
    notes(s, "약 60초", "22채널이 두 LSTM으로, 회전 갈래로 가는 빠른 진단 경로는 없다",
          say="""구조 그림입니다. 10 ms마다 한 행씩, 22개 채널이 들어옵니다.

빠른 진단 15채널은 방전마다 그 방전 자신의 평균과 분산으로 맞춥니다. 거기에 지난 시간 간격, 그리고 타깃마다 마지막으로 관측된 값, 그 뒤로 얼마나 지났는지, 관측이 있었는지가 붙습니다.

이온 온도 갈래는 22개를 전부 읽습니다. 회전 갈래는 빠른 진단을 뺀 7개만 읽습니다. 그림의 붉은 점선이 없는 길입니다.

갈래가 입력 단계에서부터 나뉘기 때문에, 빠른 진단 정보가 회전 쪽으로 새어 들어갈 길이 구조적으로 없습니다.

그리고 고정된 창이 아니라, 방전 구간 전체를 처음부터 읽어 내려갑니다.""",
          facts="입력 22 = 빠른 진단 15(방전별 z) + log(1+Δt) + 타겟별(이월 값, log(1+경과 시간), 관측 플래그)×2.")


def s_lstm():
    s = new_slide()
    title(s, "LSTM 셀: 기억 길이는 forget 게이트가 정함")
    box(s, L, I(2.65), I(10.2), I(6.9), fill=CARD, radius=0.03)
    pic(s, fig_png("fig_lstm_cell_ko"), L + I(0.25), I(2.8), I(9.7), I(6.6))
    x, w = L + I(11.0), CW - I(11.0)
    for i, (lab, big, desc) in enumerate([
        ("FORGET GATE · 실측", "0.465–0.482", "동결 가중치에서 잰 기하평균"),
        ("MEMORY", "1.3 스텝", "약 13 ms. τ > 5 스텝 유닛은 320개 중 0"),
        ("MIRNOV INPUT COLUMN", "0.58 → 0.97", "학습 중 forget 게이트 행에서 자람 (BES 0.59 → 0.71)"),
    ]):
        stat_col(s, x, I(2.65) + i * I(2.3), w, lab, big, desc, big_size=40, desc_size=18)
    notes(s, "약 60초", "c_t = f·c_(t−1) + i·g, 기억 길이는 f가 정한다",
          say="""LSTM 셀 하나를 확대한 그림입니다.

핵심은 한 줄입니다. 기억 c는 이전 기억에 f를 곱한 것과, 새로 들어온 정보를 더한 겁니다. 그래서 f가 1에 가까우면 오래 기억하고, 0에 가까우면 금방 잊습니다.

학습이 끝난 모델에서 f를 직접 재 봤습니다. 평균이 0.47 정도입니다. 기억이 1.3 스텝, 13 ms 정도밖에 이어지지 않는다는 뜻입니다.

이게 뒤에서 보여 드릴 식과 이어집니다. LSTM이 실제로 하는 일은, 짧은 기억으로 최근 측정을 가중 평균하는 데 가깝다는 겁니다.""",
          facts="그림: T_i 갈래 1층, 160 유닛. 보라색 = 동결 가중치 실측값, 붉은색 = 학습 중 Mirnov 입력 열이 자라는 자리.")


def s_loss():
    s = new_slide()
    title(s, "학습: 타겟별 마스크 손실, 라벨 행을 버리지 않음")
    box(s, L, I(2.75), CW, I(1.5), fill=CARD, radius=0.06)
    text(s, L, I(2.75), CW, I(1.5), "L  =  Σ m (ŷ − y)²  ÷  Σ m   +   0.1 · mean ReLU(z₀ − ŷ_Ti)",
         size=32, face=SANS_B, bold=True, color=INK, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    for (x, w), (lab, big, desc) in zip(cols(3), [
        ("MASK m", "28 %", "두 타겟을 모두 요구하면\n버려지는 라벨 행.\n타겟별 마스크로 살린다"),
        ("MEASUREMENT FLOOR", "46.4–129.9 eV", "타겟 자신의 잡음 σ_meas.\n신경망 RMSE 157.8 eV와\n같은 자릿수"),
        ("BATCH", "16", "경사 잡음 척도\n중앙값 33.6 블록의 절반 근처"),
    ]):
        stat_col(s, x, I(4.75), w, lab, big, desc, big_size=48, desc_size=20)
    band(s, I(8.7), "최적화", "AdamW 10⁻³ · weight decay 10⁻⁴ · clipping 1.0 · 최대 30 epoch 조기 종료 · 정규화는 학습 파일로만")
    notes(s, "약 60초", "평균 제곱 오차를 줄이도록 학습한다. 그래서 평균 estimator다",
          say="""학습은 위 식으로 합니다. 관측이 있는 칸에서만 오차 제곱을 더합니다. m이 그 표시입니다.

이렇게 하면 이온 온도만 있는 행, 회전만 있는 행도 버리지 않습니다. 두 값이 다 있어야 쓰는 방식이면 라벨 행의 28 %를 버리게 되거든요.

두 번째 숫자는 측정 잡음입니다. 정답 자체가 46에서 130 eV 정도 흔들립니다. 모델 오차 158 eV와 같은 자릿수입니다. 완벽한 모델도 이 잡음은 못 넘습니다.

하나 기억해 주십시오. 평균 제곱 오차를 줄이도록 학습했기 때문에, 이 모델은 평균을 맞히는 estimator입니다. 이게 뒤에서 평균과 중앙값을 나눠 읽는 이유가 됩니다.""",
          facts="둘째 항은 정규화 영점 z₀ 아래의 T_i 예측을 억제(λ = 0.1). 4차 차분 제곱질량의 46.6 %(V_rot 65.6 %)가 상위 1 % 한 스텝 변화에 있음 → 손실은 꼬리 통계. 배치 16 점검: 첫 epoch 이후 그래디언트 에너지의 86–98 %가 잡음.")


def s_readout():
    s = new_slide()
    title(s, "학습된 신경망 읽기: 미리 등록한 네 단계")
    flow(s, L, I(2.75), CW, I(3.6), [
        ("STAGE 1 · FROZEN ROWS", "이동 평균 하나", "persistence 대비 이득의\n73–82 %. 선형 항을 더하면\n77–85 %"),
        ("STAGE 2 · TRAINED", "증분 형태 식", "PCHIP 대비 이득의\n50–66 %. 기준 70 % 미달"),
        ("STAGE 3 · JACOBIAN", "반응 모양 측정", "100 ms 고역 통과.\n2σ까지 평탄, 3σ 넘으면 꺼짐"),
        ("STAGE 4 · CORRECTED", "형태를 고친 식", "72–78 %.\n기준 통과 (4/4 분할)"),
    ], hot=(3,))
    band(s, I(7.0), "가중치 크기는 쓰지 않음", "첫 층 열 노름은 15채널에서 평평하고 순위가 분할마다 달라, 중요도를 말해 주지 않음")
    band(s, I(8.0), "입출력 반응으로 읽음", "출력을 각 입력으로 미분해 반응 모양을 재고, 그 모양으로 식을 다시 세움", fill=WHITE)
    notes(s, "약 70초", "가중치가 아니라 반응을 읽었다. 네 단계 모두 실행 전에 기준을 등록",
          say="""그럼 학습된 신경망이 실제로 무엇을 계산하는지 어떻게 알까요?

처음엔 가중치 크기를 봤습니다. 그런데 이건 아무것도 알려 주지 않았습니다. 채널마다 평평하고, 분할을 바꾸면 순위가 바뀌었습니다.

그래서 입력과 출력의 관계를 직접 쟀습니다. 네 단계로, 매 단계 통과 기준을 미리 적어 두고 했습니다.

1단계, 학습된 모델 출력 옆에 단순 이동 평균 하나를 놓아 보니 이득의 4분의 3이 설명됐습니다.

2단계, 그걸 식으로 처음부터 학습시켰더니 기준에 못 미쳤습니다. 50에서 66 %입니다.

3단계, 그래서 모델의 반응 모양을 미분으로 쟀습니다. 빠른 진단에 대해 "최근 100 ms 평균에서 얼마나 벗어났나"를 보고, 3σ 넘게 튀면 무시하고 있었습니다.

4단계, 그 모양으로 식을 고쳤더니 기준을 넘었습니다. 다음 장이 그 식입니다.""",
          facts="B.28 1–4단계 (§8bz–§8bz-4). 동결 체크포인트 30벌. 1단계 문턱 0.75, 2·4단계 문턱 0.70. 2단계 교차항은 0.001 미만.")


def s_size():
    s = new_slide()
    title(s, "크기와 구조: 더 키워도 바뀌지 않음")
    for (x, w), (lab, big, desc) in zip(cols(4, gap=I(0.6)), [
        ("SMALL MODEL", "+0.002", "21,498 파라미터 해석 가능 모델의\n신경망 대비 T_i skill\n(컷, 모든 구간 0 포함)"),
        ("WIDTH × 26", "+0.23", "T_i 인코더 폭을 바꿔도\n평균 skill +0.230~+0.236"),
        ("ENSEMBLE × 4", "+0.020", "초기화 네 개 평균의\n짝지은 차이"),
        ("CONTEXT", "50 ms", "문맥 포화. 인과 GP 대비\n+0.104 [+0.079, +0.128]"),
    ]):
        stat_col(s, x, I(2.75), w, lab, big, desc, big_size=52, desc_size=19)
    band(s, I(7.6), "격자 전체 vs 창", "전체 격자 LSTM이 W = 2 창 모델보다 16/16 실행에서 양수, 합산 +0.081 [+0.067, +0.096]")
    band(s, I(8.6), "비용", "상태를 이어 가는 한 스텝이 CPU에서 중앙값 0.97 ms, p99 1.49 ms", fill=WHITE)
    notes(s, "약 50초", "남은 한계는 모델 용량이 아니라 입력 정보",
          say="""모델을 더 키우면 나아질까요? 아닙니다.

파라미터 2만 개짜리 작은 모델도, 컷 기준으로 신경망과 차이가 없습니다. 폭을 26배 범위로 바꿔도 skill은 그대로입니다. 초기화 네 개를 평균해도 0.02 정도입니다.

과거를 보는 길이도 50 ms면 충분했습니다. 더 보여 줘도 늘지 않습니다.

반대로 창 몇 칸만 보는 모델보다는, 방전 전체를 이어서 읽는 지금 구조가 16번 중 16번 모두 나았습니다.

즉, 남은 한계는 모델 크기가 아니라 입력에 담긴 정보입니다.""",
          facts="해석 가능 칸 b3k8: 포함 모집단에서는 −0.194. 폭 스윕 34k–879k 파라미터. 앙상블 +0.020은 사전등록 0.02 띠 위.")


# ---------------------------------------------------------------- 04 평가 방법

def s_protocol():
    s = new_slide()
    title(s, "평가 절차: 방전 단위 분할과 재현")
    row_list(s, L, I(2.75), CW, [
        ("네 분할 · 시드 42, 1, 7, 123", "시험 방전은 구조 탐색 전에 떼어 둠. T_i 32,589행 / 96방전, V_rot 10,463행 / 60방전 (컷)"),
        ("방전 단위 부트스트랩 10,000회", "같은 방전의 행들은 닮아 있으므로, 방전을 통째로 다시 뽑아 95 % 구간을 낸다"),
        ("반복 5겹 교차검증 · 100회 실행", "639개 방전을 반복마다 한 번씩 시험. 겹 단위는 Nadeau–Bengio 겹침 보정 구간"),
        ("두 단계 부트스트랩", "반복 하나를 뽑고 그 반복의 방전을 다시 뽑음. 등록 뒤 실행한 반복 8회, 판정 기준"),
        ("사전등록", "가설·판정 규칙·예측 수치를 실행 전에 저장소에 커밋. 제3자 등록소는 없음"),
    ], head_size=26, gap=I(1.38), desc_size=20)
    notes(s, "약 60초", "분할이 넷뿐이라 재현을 따로 돌렸다",
          say="""평가 절차입니다.

먼저 방전 단위로 네 번 나눴습니다. 시험 방전은 모델을 고르기 전에 떼어 두고 한 번도 보지 않았습니다.

구간은 방전을 통째로 다시 뽑아서 냅니다. 같은 방전 안의 행들은 서로 닮아 있어서, 행 단위로 뽑으면 구간이 실제보다 좁아지거든요.

그런데 네 분할은 학습 방전을 대부분 공유합니다. 그래서 따로 재현을 돌렸습니다. 방전 639개를 다섯 겹으로 나눠 열 번 반복했고, 모두 100번 학습했습니다.

여기서 나온 결과를 두 단계로 다시 뽑아 구간을 냅니다. 학습을 다시 하는 우연과, 어떤 방전이 시험에 걸리는 우연을 둘 다 담습니다. 판정 기준은 이 구간으로, 실행 전에 정해 두었습니다.""",
          facts="교차 재표집은 분산을 과대추정하므로 구간은 보수적(Owen 2007, Gallas 2009). 분할 4개뿐이라 평균 ± 표준편차 대신 분할별 값·범위·0 제외 수를 적음(Weissgerber 2015, Agarwal 2021).")


# ---------------------------------------------------------------- 05 결과

def s_ladder():
    s = new_slide()
    title(s, "RMSE 사다리: 미래를 안 보는 방법 가운데 가장 낮음")
    table(s, L, I(2.7), [I(5.0), I(6.4), I(3.0), I(3.0)], [
        ["방법", "쓰는 정보", "T_i (eV)", "V_rot (km/s)"],
        ["**LSTM (이 연구)", "빠른 진단 + 과거 CES, 구간 전체", "**157.8", "**23.6"],
        ["창 모델 (W = 2)", "빠른 진단 + 과거 CES 2행", "169.2", "26.1"],
        ["과거만 쓰는 GP", "과거 CES 16개", "164.3", "28.8"],
        ["직전 값 유지", "마지막 관측 CES", "197.2", "33.4"],
        ["국소 AR", "과거 CES", "472.2", "51.0"],
        ["미래도 쓰는 GP", "과거 + 미래 CES", "153.8", "24.7"],
        ["선형 보간", "과거 + 미래 CES", "169.8", "29.0"],
        ["PCHIP", "과거 + 미래 CES", "173.6", "30.2"],
    ], size=22, row_h=I(0.74),
       align=[PP_ALIGN.LEFT, PP_ALIGN.LEFT, PP_ALIGN.RIGHT, PP_ALIGN.RIGHT])
    text(s, L, I(9.55), CW, I(0.5), "시드 42 시험 분할, 컷, 진짜 측정만 (T_i 32,589행 · V_rot 10,463행). 아래 세 줄은 미래를 읽는다.",
         size=18, color=MUTED)
    notes(s, "약 50초", "인과 방법 중 최저, 미래도 쓰는 GP만 T_i에서 조금 낮다",
          say="""결과입니다. 먼저 물리 단위의 오차입니다.

맨 위가 저희 모델입니다. 이온 온도 158 eV, 회전 23.6 km/s입니다.

그 아래 네 줄이 미래를 보지 않는 방법들인데, 모두 저희보다 큽니다. 과거만 쓰는 GP보다 이온 온도는 4 %, 회전은 18 % 낮습니다.

아래 세 줄은 미래를 읽는 방법입니다. 보간 두 가지보다도 저희가 낮습니다. 미래까지 쓰는 GP만 이온 온도에서 조금 더 낮습니다. 이 차이는 뒤에서 구간으로 다시 보겠습니다.""",
          facts="포함 모집단: LSTM 363.0 / 23.7, PCHIP 412.4 / 30.2, 인과 GP 394.6 / 28.8, persistence 478.0 / 33.4.")


def s_headline():
    s = new_slide()
    title(s, "비교 대상별 skill: 네 분할, 반복 교차검증, 두 단계 부트스트랩")
    table(s, L, I(2.6), [I(1.6), I(3.4), I(4.4), I(4.0), I(4.0)], [
        ["타깃", "대비", "네 분할 (0 제외 수)", "겹 단위 구간", "두 단계 구간"],
        ["T_i", "PCHIP", "+0.174~+0.264 (4/4)", "**[+0.186, +0.253]", "**[+0.188, +0.248]"],
        ["", "직전 값 유지", "+0.360~+0.415 (4/4)", "**[+0.368, +0.410]", "**[+0.369, +0.408]"],
        ["", "과거만 쓰는 GP", "+0.078~+0.138 (4/4)", "**[+0.084, +0.138]", "**[+0.085, +0.132]"],
        ["", "미래도 쓰는 GP", "−0.05~+0.11 (7/8 0 포함)", "[−0.044, +0.020]", "[−0.043, +0.014]"],
        ["V_rot", "PCHIP", "+0.135~+0.390 (1/4)", "[−0.021, +0.405]", "**[+0.072, +0.701]"],
        ["", "직전 값 유지", "+0.30~+0.50 (3/4)", "**[+0.190, +0.601]", "**[+0.277, +0.763]"],
        ["", "과거만 쓰는 GP", "+0.020~+0.331 (2/4)", "[−0.060, +0.512]", "**[+0.045, +0.701]"],
        ["", "미래도 쓰는 GP", "−0.07~+0.09 (7/8 0 포함)", "[−0.234, +0.312]", "[−0.119, +0.635]"],
    ], size=19, head_size=16, row_h=I(0.74))
    text(s, L, I(9.4), CW, I(0.6), "컷 기준. 굵은 글씨 = 95 % 구간이 0을 뺌. 미래도 쓰는 GP의 네 분할 칸은 두 모집단을 합친 것.",
         size=18, color=MUTED)
    notes(s, "약 70초", "T_i는 세 방식 모두 0을 뺀다. V_rot는 판정 기준인 두 단계 구간에서 0을 뺀다",
          say="""두 주장을 한 표로 모았습니다. 오른쪽 끝 열이 판정 기준인 두 단계 구간입니다. 굵은 글씨가 0을 뺀 칸입니다.

위 네 줄, 이온 온도는 어느 방식으로 재도 미래도 쓰는 GP를 빼면 모두 0을 뺍니다.

아래 네 줄, 회전은 사정이 다릅니다. 네 분할로만 보면 PCHIP 대비 한 분할에서만 0을 뺍니다. 겹 단위로 봐도 0을 포함합니다.

그래서 회전 주장은 실험 전에 판정 기준으로 정해 둔 두 단계 구간으로만 세웁니다. 거기서는 0을 뺍니다. 그리고 구간이 넓다는 걸 같이 말씀드립니다.

미래도 쓰는 GP와는 두 타깃 모두 차이를 판정할 수 없습니다.""",
          facts="8쪽 논문 tab:headline (컷 열). 포함: T_i PCHIP 두 단계 [+0.192, +0.294], V_rot [+0.046, +0.683]. 겹 단위 구간이 넓은 이유: 방전 80개짜리 겹의 비율이 큰 오차 방전 몇 개로 정해짐.")


def s_stress():
    s = new_slide()
    title(s, "두 스트레스 테스트: 진짜 결측 시각과 캠페인 이동")
    for (x, w), (lab, big, desc) in zip(cols(2, gap=I(1.0)), [
        ("MNAR · 진짜 결측 시각으로 재가중", "+0.28 ~ +0.44", "직전 값 유지 대비 T_i skill, 8/8 셀에서 0 제외\nPCHIP 대비 +0.14~+0.28 (컷 2/4, 포함 4/4)"),
        ("CAMPAIGN · 시간 순서로 분할", "+0.17 ~ +0.20", "PCHIP 대비, 두 모집단 4/4 초기화에서 0 제외\n과거만 쓰는 GP 대비 +0.11~+0.16"),
    ]):
        stat_col(s, x, I(2.75), w, lab, big, desc, big_size=60, desc_size=21)
    band(s, I(7.0), "Δt > 15 ms", "마지막 관측에서 먼 시각에서도 PCHIP 대비 +0.268(컷) · +0.206(포함), 구간 0 제외")
    band(s, I(8.0), "창 모델은 무너짐", "같은 시간 분할에서 W = 2 창 모델은 −0.001~+0.091 (컷 2/4, 포함 0/4)", fill=WHITE)
    band(s, I(8.9), "이유", "빠른 진단이 두 기간 사이 1.22σ(BES) · 0.53σ(ECEI) 이동, 타겟은 0.115σ → 방전별 표준화가 흡수", fill=WHITE)
    notes(s, "약 70초", "관측된 시각만이 아니라, 진짜 빈 시각과 다른 시기에도 남는다",
          say="""채점은 측정이 있는 시각에서만 할 수 있습니다. 그런데 실제로 채워야 하는 건 측정이 빈 시각이죠. 둘이 다르면 어떡할까요?

왼쪽이 그 검사입니다. 채점 행을, 진짜 빈 시각과 비슷한 조건으로 다시 가중했습니다. 직전 값 유지보다는 여전히 확실히 낫습니다. 보간 대비로는 컷 기준 일부 분할에서만 0을 뺍니다.

오른쪽은 시기를 바꾼 검사입니다. 시간 순서로 앞쪽 방전으로 학습하고 가장 늦은 방전으로 시험했습니다. 저희 모델은 그대로 유지됐습니다.

창 몇 칸만 보는 모델은 여기서 무너졌습니다. 이유도 쟀습니다. 빠른 진단 신호가 시기에 따라 크게 이동하는데, 저희 모델은 방전마다 자기 기준으로 맞추기 때문에 그 이동을 흡수합니다.""",
          facts="MNAR 재가중은 결측 T_i 행의 54–68 %(도메인 안)에 닿음. 결측 V_rot 행은 4–6 %만 도메인 안이라 판정 없음. 시간 분할: 학습 416, 검증 128, 시험 가장 늦은 97 방전.")


def s_ablation():
    s = new_slide()
    title(s, "절제: 빠른 진단을 지우면 T_i만 무너짐")
    table(s, L, I(2.75), [I(4.2), I(2.0), I(5.6), I(5.6)], [
        ["입력", "타깃", "컷", "포함"],
        ["전체", "T_i", "+0.173", "+0.238"],
        ["빠른 진단 제거", "T_i", "**−0.125 · −0.25~−0.43 (4/4)", "+0.201 · −0.03~−0.09 (2/4)"],
        ["이력 제거", "T_i", "−2.11", "−1.16"],
        ["전체", "V_rot", "+0.213", "+0.206"],
        ["빠른 진단 제거", "V_rot", "+0.213 · 변화 0 (출력 동일)", "+0.206 · 변화 0 (출력 동일)"],
        ["이력 제거", "V_rot", "−2.89", "−3.51"],
    ], size=21, row_h=I(0.85))
    band(s, I(9.0), "읽는 법", "회전 출력이 그대로인 건 구조의 결과이고, 근거는 변화율 상관과 원시 자료 검정. PCHIP 대비 skill, 4분할 평균")
    notes(s, "약 50초", "이력은 둘 다에 필수, 빠른 진단은 T_i의 여유",
          say="""비교용 창 모델에서 입력을 하나씩 지워 봤습니다.

과거 이력을 지우면 두 타깃 모두 완전히 무너집니다. 이력은 둘 다에게 필수입니다.

빠른 진단을 지우면 이야기가 갈립니다. 이온 온도는 컷 기준으로 보간보다 못해집니다. 네 분할 모두 유의했습니다. 회전은 출력이 그대로입니다.

다만 회전이 그대로인 건 애초에 빠른 진단을 안 넣었으니 당연한 결과입니다. 회전에 정보가 없다는 근거는 앞에서 본 변화율 상관과, 다음 장의 원시 자료 검정입니다.""",
          facts="포함 모집단에서는 이력만으로 PCHIP 대비 +0.15~+0.23이 남음(스파이크 강건성). 그래서 빠른 진단의 몫을 가르는 건 컷.")


def s_routes():
    s = new_slide()
    title(s, "회전 정보로 가는 두 길: 원시 자기 코일과 가열 빔")
    x1, w1 = L, I(9.4)
    x2, w2 = L + I(10.2), CW - I(10.2)
    eyebrow(s, x1, I(2.65), w1, "MIRNOV · 원시 2 MHz, 방전 12개")
    row_list(s, x1, I(3.1), w1, [
        ("24 / 24 코일", "100 Hz 값 = 원시 표본 한 점, 시차 0"),
        ("−0.128  [−0.544, −0.105]", "등록 특징 15개를 넣은 skill 변화 (컷) · 포함 −0.164"),
        ("0 / 3 방전", "학습 6 · 시험 3, 나아진 시험 방전 없음"),
        ("4.39 · 8.91 kHz", "#32092 가장자리 조화 진동, 코일은 정상"),
        ("|ρ| ≤ 0.11", "모드 주파수와 회전의 방전 내 상관, 7개 중 6개"),
    ], head_size=24, gap=I(1.22), desc_size=19)
    eyebrow(s, x2, I(2.65), w2, "NBI · 가열 빔")
    row_list(s, x2, I(3.1), w2, [
        ("+0.353 / +0.024", "방전 간 Tₑ 대리 지표와 T_i / V_rot의 상관"),
        ("0.34 %", "10 ms V_rot 변화 중 300 ms 대역 몫 · T_i 0.25 %"),
        ("100 ms 이상", "토크 채널이 의미를 갖는 예보 길이"),
    ], head_size=24, gap=I(1.5), desc_size=19)
    notes(s, "약 80초", "자기 코일도 빔도 10 ms 회전 변화에는 닿지 않는다",
          say="""회전 정보를 얻을 수 있었던 길이 두 개 있었습니다. 둘 다 직접 검정했습니다.

왼쪽, 자기 코일입니다. 원시 2 MHz 신호를 따로 받았습니다. 먼저 확인한 건, 저희 자료의 100 Hz 값이 평균이 아니라 원시 신호의 한 점이라는 겁니다. 코일 24개 모두 그랬습니다.

그래서 원시 신호로 특징 15개를 만들어 회전 갈래에 넣었습니다. 결과는 오히려 나빠졌습니다. 시험 방전 3개 중 나아진 게 하나도 없었습니다.

코일이 고장 난 건 아닙니다. 문헌에 나온 진동이 4.39, 8.91 kHz에 그대로 보입니다. 다만 그 모드 주파수가 회전과 상관이 없었습니다.

오른쪽, 가열 빔입니다. 빔은 회전의 수준을 정하지만, 켜고 끄는 계단 입력입니다. 10 ms 변화에는 0.34 %만 기여합니다. 그래서 100 ms 이상 앞을 내다볼 때 쓰는 레버입니다.""",
          facts="B.6 (§8bd): 36 CSV, BES · ECEI 500 kHz, MC 2 MHz. 특징 = RMS 포락선, 대역 파워, 교차 스펙트럼 봉우리. #31359만 +0.51. NBI 0.34 %는 20개 방전 IQR 0.21–0.94 %.",
          qa="다른 특징을 쓰면 되지 않나 → 기각 범위는 등록한 특징 15개와 학습 방전 6개입니다. 여러 코일로 모드 번호를 가리는 특징이나 원시 방전 수십 개는 아직 검정하지 않았습니다.")


def s_retain():
    s = new_slide()
    title(s, "식의 skill 유지: 처음부터 학습해도 약 3/4")
    for (x, w), (lab, big, desc) in zip(cols(3), [
        ("PCHIP-RELATIVE · CUT 4/4", "72–78 %", "LSTM의 보간 대비 이득 중\n식이 내는 몫 (등록 기준 70 %)"),
        ("PERSISTENCE-RELATIVE", "87–90 %", "같은 식,\n직전 값 유지 대비"),
        ("vs STAGE 2 FORM", "+0.034~+0.042", "형태를 고치기 전 식보다\n짝지은 차이, 4/4 구간 0 위"),
    ]):
        stat_col(s, x, I(2.75), w, lab, big, desc, big_size=44)
    band(s, I(7.4), "남은 22–28 %", "신경망의 비선형 사건 처리. 빠른 진단 이력의 어떤 저차 함수로도 재현되지 않음")
    band(s, I(8.4), "포함 모집단", "0.43–0.81 유지. BES 이득이 맞춤 실패 스파이크 아래에서 부호를 바꿔, 물리 값은 컷에서만 읽음", fill=WHITE)
    notes(s, "약 40초", "식은 신경망 이득의 약 4분의 3",
          say="""모델 절에서 보여 드린 식의 성적입니다.

파라미터 20개짜리 식을 처음부터 학습시키면, LSTM이 보간보다 나아진 몫의 72에서 78 %가 나옵니다. 미리 정한 기준 70 %를 네 분할 모두 넘었습니다.

남은 4분의 1은 신경망만 하는 비선형 사건 처리입니다. 어떤 단순한 식으로도 재현되지 않았습니다.

포함 기준에서는 식의 값이 흔들립니다. 그래서 식의 물리 값은 컷 기준에서만 읽습니다.""")


def s_dual():
    s = new_slide()
    title(s, "이중 판독: 평균은 신경망, 중앙값은 물리 필터")
    table(s, L, I(2.75), [I(7.4), I(5.0), I(5.0)], [
        ["지표", "T_i", "V_rot"],
        ["신경망 중앙값 오차 vs 직전 값 유지", "47.2 vs 46.1 eV (같음)", "4.3 vs 3.7 km/s (더 큼)"],
        ["PCHIP 중앙값 오차", "39.4 eV (17 % 낮음)", "—"],
        ["물리 필터 중앙값 ÷ 직전 값 유지", "**0.906–0.920 (8/8 0 제외)", "0.94–1.00 (3/4 분할 1 포함)"],
        ["물리 필터의 신경망 대비 MSE skill", "−0.06 ~ −0.52 (두 타깃)", "사건 행에서 늦음"],
        ["상위 1 % 행의 제곱오차 몫", "48 % (컷) · 83 % (포함)", "—"],
    ], size=21, row_h=I(0.9))
    band(s, I(8.4), "보정된 분포", "Student-t head 50 % / 90 % 포함률 0.48–0.51 / 0.91–0.92 (컷). 포함은 혼합 head 0.53–0.55 / 0.90–0.91")
    notes(s, "약 60초", "한 estimator가 평균과 중앙값을 함께 낮추지 못한다",
          say="""평균과 중앙값을 표 하나로 정리했습니다.

신경망은 평범한 순간에 직전 값을 거의 그대로 옮깁니다. 이온 온도는 직전 값 유지와 같고, 회전은 조금 더 틀립니다.

평범한 순간을 맡는 건 물리 필터입니다. 값은 머물다가 가끔 뛰고, 측정 잡음은 꼬리가 두껍다고 보는 단순한 모델입니다. 이 필터가 이온 온도에서는 직전 값보다 9 % 정도 낫습니다. 회전에서는 직전 값과 같습니다. 광자 잡음이 바닥이니까요.

그런데 이 필터는 사건이 있는 순간에 늦어서 평균 오차가 큽니다. 한 방법으로 둘 다 낮추지는 못했습니다. 그래서 평균은 신경망에서, 중앙값은 필터에서 따로 읽습니다.

마지막 줄은 불확도입니다. 신경망이 값 하나가 아니라 분포를 내게 했고, 말한 구간이 실제로 그만큼 맞았습니다.""",
          facts="물리 필터 = 유지-또는-점프 사전 + Student-t(ν₀ = 3) 측정 잡음, 10 ms 본류 산포 척도, CES 이력 위에서 인과적. persistence 대비 T_i 중앙값 skill +0.08~+0.09 (8/8). V_rot 백본 대비 중앙값 skill +0.09~+0.27.")


# ---------------------------------------------------------------- 조립

s_cover()
s_summary()
s_agenda()
divider(0, "먼저 서론입니다.")
s_gap()
s_interp()
s_question()
divider(1, "다음은 자료입니다.")
s_data()
s_audit()
divider(2, "이제 모델입니다. 왜 이렇게 만들었는지부터 말씀드리겠습니다.")
s_physics()
s_torque()
s_scales()
s_model()
s_arch()
s_lstm()
s_loss()
s_readout()
s_equation()
s_size()
divider(3, "다음은 평가 방법입니다.")
s_score()
s_protocol()
divider(4, "결과입니다.")
s_ladder()
s_claim1()
s_why1()
s_stress()
s_claim2()
s_ablation()
s_routes()
s_headline()
s_retain()
s_median()
s_dual()
s_summary_end()

prs.save(OUT)
print("saved", len(prs.slides), "slides")
