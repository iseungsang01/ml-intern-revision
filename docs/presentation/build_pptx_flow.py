# -*- coding: utf-8 -*-
"""Build the **paper-writing digest** deck — "논문에 들어가는 것만".

Output: docs/presentation/KSTAR_CES_연구흐름.pptx

This is a third, different-purpose deck. The other two are *presentation* decks:

    build_pptx.py        -> 38 slides, ~60 min thesis defense   (결과를 설득하는 덱)
    build_pptx_20min.py  -> 24 slides, 20 min seminar           (결과를 압축한 덱)
    build_pptx_flow.py   -> this file, 18 slides                (논문을 쓰기 위한 덱)

Where the result decks answer "이 결과를 믿어도 되는가", this one answers "논문의 이
절에 무엇을 쓰고, 어느 수치를 인용하며, 근거는 어디에 있는가". Every slide maps to a
section of `docs/paper/main_ko.tex`, and its notes carry that section's `\\label`.

**Deliberately excluded** (2026-08-09 재편): 날짜 타임라인, 연구 질문이 바뀐 경위
(초해상 → gap-filling), AutoML 탐색 경위, 재현성/체크포인트 함정, 운영용 다음-작업
우선순위. 논문에 한 줄도 들어가지 않는 과정 서사이므로 이 덱에서 전부 잘라냈다.
그 기록은 THESIS_RESULTS.md §8과 PROJECT_KNOWLEDGE.md에만 남긴다. 음성 결과는
버리지 않되, 논문이 쓰는 형태 — §9 "남은 개선 여지"의 레버와 §10 한계 — 로만 싣는다.

Palette, layout helpers and figures are reused from build_pptx.py so all three decks
look like one family. 모든 수치는 `docs/paper/main_ko.tex`(= `paper_numbers.json`
동결 산출물)에서 그대로 옮겼다.

Usage (from repo root):
    python docs/presentation/build_pptx_flow.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from build_pptx import (  # noqa: E402  (path bootstrap must run first)
    prs, slide, box, text, header, card, add_image_fit, table,
    NAVY, BLUE, TEAL, ORANGE, GREEN, RED, GRAY, LGRAY, MGRAY, WHITE, DARK, CARDBG,
    EMU_W, EMU_H, FIG, FONT,
)
from preview_pptx import load_font, _TOKEN  # noqa: E402  (same metrics as the QC renderer)
from PIL import Image, ImageDraw  # noqa: E402
from pptx.util import Inches, Pt  # noqa: E402
from pptx.dml.color import RGBColor  # noqa: E402
from pptx.enum.text import PP_ALIGN  # noqa: E402

OUT = os.path.join(HERE, "KSTAR_CES_연구흐름.pptx")

_MEASURE = ImageDraw.Draw(Image.new("RGB", (8, 8)))
_MARGIN_IN = 2 * (2 / 72.0)   # text() sets 2pt left + 2pt right margins
_WARNED = []


def _n_lines(txt, avail_in, size_pt, bold=False):
    """Line count after wrapping, using preview_pptx's exact tokenizer and metrics."""
    font = load_font(FONT, bold, size_pt)          # px == pt  ->  lengths are in points
    avail_px = max(avail_in, 0.1) * 72.0
    n, cur = 1, 0.0
    for tok in _TOKEN.findall(txt):
        tw = _MEASURE.textlength(tok, font=font)
        if tok.isspace():
            if cur + tw > avail_px:
                n += 1
                cur = 0.0
                continue
            cur += tw
            continue
        if cur + tw > avail_px and cur > 0:
            n += 1
            cur = 0.0
        cur += tw
    return n


def _block_h(lines, avail_in, size_pt, line_spacing, space_after_pt, bold=False):
    lh = size_pt * 1.24 * line_spacing / 72.0
    total = 0.0
    for ln in lines:
        total += _n_lines(ln, avail_in, size_pt, bold) * lh + space_after_pt / 72.0
    return total


def note(s, txt):
    s.notes_slide.notes_text_frame.text = txt.strip("\n")
    return s


def chip(s, x, y, w, h, label, fill, color=WHITE, size=11):
    box(s, x, y, w, h, fill=fill, round_=True)
    text(s, x, y + Inches(0.04), w, h,
         [[(label, size, color, True, False, None)]], align=PP_ALIGN.CENTER)


def _in(v):
    """Accept either inches (float) or Emu (what Inches() returns)."""
    return v / 914400.0 if isinstance(v, int) and v > 1000 else float(v)


def fcard(s, x, y, w, h, title, lines, accent=BLUE, body_size=11.5, tag=""):
    """card() that shrinks the type until the wrapped text actually fits the box.

    Layout QC is not optional here: this deck is dense by design, so every card is
    measured with the same font metrics `preview_pptx.py` uses to flag overflow.
    A card that cannot fit even at 9 pt is reported at build time and must be cut.
    """
    x, y, w, h = _in(x), _in(y), _in(w), _in(h)
    body_avail = w - 0.42 - _MARGIN_IN
    title_avail = w - 0.40 - _MARGIN_IN
    body_room = h - 0.70

    title_size = 14.0
    while title_size > 10.5 and _n_lines(title, title_avail, title_size, bold=True) > 1:
        title_size -= 0.5
    if _n_lines(title, title_avail, title_size, bold=True) > 1:
        _WARNED.append(f"title too long: {title!r} ({tag})")

    size = body_size
    while size > 10.0 and _block_h(lines, body_avail, size, 1.1, 1) > body_room:
        size -= 0.5
    need = _block_h(lines, body_avail, size, 1.1, 1)
    if need > body_room:
        lh = size * 1.24 * 1.1 / 72.0
        cut = int((need - body_room) / lh) + 1
        wrapped = [i for i, ln in enumerate(lines)
                   if _n_lines(ln, body_avail, size) > 1]
        _WARNED.append(
            f"{title!r}: cut ~{cut} line(s) at {size} pt "
            f"(wrapped rows {wrapped})")

    # same look as build_pptx.card(), but with tighter paragraph spacing so a
    # reference deck can carry more lines per card without overflowing it
    c = box(s, Inches(x), Inches(y), Inches(w), Inches(h), fill=CARDBG, round_=True)
    box(s, Inches(x), Inches(y), Inches(0.10), Inches(h), fill=accent)
    text(s, Inches(x + 0.26), Inches(y + 0.12), Inches(w - 0.40), Inches(0.4),
         [[(title, title_size, accent, True, False, None)]])
    text(s, Inches(x + 0.26), Inches(y + 0.56), Inches(w - 0.42), Inches(h - 0.70),
         [[(ln, size, DARK, False, False, None)] for ln in lines],
         line_spacing=1.1, space_after=1)
    return c


def band(s, y, lines, fill=NAVY, x=0.55, w=12.23, h=0.78, pad=0.3, tag=""):
    """Full-width conclusion band. lines: list of paragraph run-lists (auto-shrunk)."""
    avail = w - 2 * pad - _MARGIN_IN
    flat = ["".join(seg[0] for seg in para) for para in lines]
    base = max(seg[1] for para in lines for seg in para)

    sz = base
    while sz > 9.5 and _block_h(flat, avail, sz, 1.2, 3) > h - 0.2:
        sz -= 0.5
    if _block_h(flat, avail, sz, 1.2, 3) > h - 0.2:
        _WARNED.append(f"band overflows at {sz} pt ({tag}): {flat[0][:40]!r}")
    ratio = sz / base

    box(s, Inches(x), Inches(y), Inches(w), Inches(h), fill=fill, round_=True)
    text(s, Inches(x + pad), Inches(y + 0.10), Inches(w - 2 * pad), Inches(h - 0.2),
         [[(seg[0], round(seg[1] * ratio, 1)) + tuple(seg[2:]) for seg in para]
          for para in lines],
         line_spacing=1.2, space_after=3)


# =============================== SLIDES ===================================

# --- 1. Title -------------------------------------------------------------
def f_title():
    s = slide()
    box(s, 0, 0, EMU_W, EMU_H, fill=NAVY)
    box(s, 0, Inches(5.5), EMU_W, Inches(2.0), fill=RGBColor(0x0E, 0x26, 0x47))
    box(s, Inches(0.9), Inches(1.75), Inches(2.2), Pt(4), fill=TEAL)
    text(s, Inches(0.9), Inches(1.95), Inches(11.6), Inches(0.5),
         [[("논문 집필용 정리 · Paper-Writing Digest", 16,
            RGBColor(0x8F, 0xD6, 0xCB), True, False, None)]])
    text(s, Inches(0.88), Inches(2.5), Inches(11.7), Inches(2.0),
         [[("논문에 들어가는 것만:", 30, WHITE, True, False, None)],
          [("확정된 주장 · 인용할 수치 · 근거의 위치", 34, WHITE, True, False, None)]],
         line_spacing=1.12)
    text(s, Inches(0.9), Inches(4.35), Inches(11.5), Inches(1.0),
         [[("슬라이드 한 장 = ", 16, LGRAY, False, False, None),
           ("main_ko.tex의 한 절", 16, ORANGE, True, False, None),
           (". 각 장의 노트에 그 절의 \\label과 인용 시 주의가 적혀 있습니다.",
            16, LGRAY, False, False, None)],
          [("연구가 걸어온 경로·기각된 곁가지·재현성 이슈는 논문에 쓰지 않으므로 이 덱에서 제외했습니다",
            16, LGRAY, False, False, None)]],
         line_spacing=1.2)
    text(s, Inches(0.9), Inches(5.9), Inches(11.5), Inches(1.1),
         [[("이승상  (Seungsang Lee)", 17, WHITE, True, False, None)],
          [("서울대학교 · 원자핵공학  |  2026-08-09 기준", 13, MGRAY, False, False, None)],
          [("출처: docs/paper/main_ko.tex · docs/paper/paper_numbers.json · THESIS_RESULTS.md §8",
            11, MGRAY, False, False, None)]],
         line_spacing=1.25)
    return note(s, """
용도: 논문을 쓰는 동안 옆에 두는 참조판.
수치는 전부 main_ko.tex(= 동결 산출물 paper_numbers.json)에서 그대로 옮겼으므로
여기서 인용하면 논문 본문과 어긋날 수 없다.

이 덱에서 의도적으로 제외한 것 — 날짜 타임라인, 연구 질문이 바뀐 경위(초해상 → gap-filling),
AutoML 탐색 경위, 체크포인트/재현성 함정, 운영용 다음-작업 우선순위.
논문에 한 줄도 들어가지 않는 과정 서사다. 기록은 THESIS_RESULTS.md §8과 PROJECT_KNOWLEDGE.md에 있다.
""")


# --- 2. Paper map ---------------------------------------------------------
def f_map():
    s = slide()
    header(s, "Map", "논문 골격 — 절 ↔ 확정한 것 ↔ 이 덱의 슬라이드", accent=NAVY)

    col_w = [Inches(2.30), Inches(5.05), Inches(2.60), Inches(2.28)]
    table(s, Inches(0.55), Inches(1.45), col_w,
          ["논문 절", "그 절이 확정한 것", "근거 (표·그림)", "이 덱"],
          [["§1–2 서론·관련 연구", "계보 인정 후 3축 확장으로 novelty 진술", "NOVELTY.md", "3"],
           ["§3 데이터·문제 설정", "641 shot · 10 ms 격자 · 유지값 54% 감사", "fig_missing", "4–5"],
           ["§4 모델", "201,258 파라미터 · 관측 마스킹 어텐션", "fig_architecture", "6"],
           ["§5 평가 방법론", "사전등록 PR1–PR4 · shot 군집 bootstrap", "—", "7"],
           ["§6 결과 (9개 소절)", "인과 압도 · Tᵢ 4/4 · 간극 · MNAR · 캠페인 · 비대칭 · window · peak",
            "표 1–8 · fig_forest", "8–15"],
           ["§7 선택 프로토콜", "게이트를 val loss → clean skill로", "fig_progression", "9"],
           ["§8 배치 가능한가", "CPU p99 6.4 ms · conformal 8/8 승", "—", "16"],
           ["§9 남은 개선 여지", "레버 3종: reach · 원 kHz Mirnov · NBI 토크", "—", "17"],
           ["§10–11 한계·결론", "두 주장(오프라인 vs 인과)의 분리", "표 stress", "18"]],
          row_h=Inches(0.53), size=11.5, head_size=12)

    band(s, 6.16,
         [[("이 덱을 쓰는 법 — ", 13, TEAL, True, False, None),
           ("논문의 한 절을 쓸 차례가 되면 대응 슬라이드를 열고 카드의 수치를 그대로 인용한다. "
            "수치를 고쳐야 하면 먼저 collect_paper_numbers.py를 돌려 paper_numbers.json을 갱신한다.",
            13, WHITE, False, False, None)]])
    return note(s, """
main_ko.tex label 목록: sec:data, sec:stuck, sec:model, sec:eval, sec:results, sec:headline,
sec:gap, sec:mnar, sec:campaign, sec:asym, sec:window, sec:ladder, sec:peak, sec:selection,
sec:deploy, sec:headroom, sec:limits, sec:conclusion.

그림 파이프라인: collect_paper_numbers.py -> docs/paper/paper_numbers.json -> make_figures_en.py
(논문) / make_figures.py (덱). 수치를 그림 스크립트에 손으로 박아 넣지 말 것 — 그 오류가 §8h다.
""")


# --- 3. The two claims ----------------------------------------------------
def f_two_claims():
    s = slide()
    header(s, "논문의 중심 논리", "두 개의 주장을 절대 뭉개지 말 것", accent=RED)

    col_w = [Inches(4.55), Inches(3.95), Inches(3.73)]
    table(s, Inches(0.55), Inches(1.45), col_w,
          ["평가", "PCHIP 대비 (오프라인 · 미래 사용)", "persistence 대비 (인과)"],
          [["무작위 분할 · 관측 지점 (헤드라인)",
            ("+0.18~+0.28,  4/4", GREEN, True, None), "+0.35~+0.42"],
           ["결측 지점으로 재가중 (§6.4)",
            ("+0.06~+0.21,  1/4", RED, True, None),
            ("+0.29,  4/4", GREEN, True, None)],
           ["시간 캠페인 분할 (§6.5)",
            ("−0.15~+0.05,  0/4", RED, True, None), "+0.12~+0.28"]],
          row_h=Inches(0.56), size=12.5)

    fcard(s, 0.55, 3.62, 6.03, 2.05,
          "본문에 쓸 문장 형태",
          ["· 헤드라인(관측 모집단 한정):",
           "  “사전등록된 보간들을 이긴다”",
           "· 배치 주장: “진짜 결측·도메인 내 시점에서",
           "  어떤 인과적 CES 전용 방법보다 유의하게 낫다”",
           "· 두 주장은 상대가 다르다. 뭉개는 것이 이 결과가",
           "  과대 판매되는 주된 경로다(§11이 명시)."],
          accent=RED, body_size=11.5)

    fcard(s, 6.75, 3.62, 6.03, 2.05,
          "novelty는 부재가 아니라 확장으로 (§1–2)",
          ["① 계보 인정 — NN 기반 CES/CER 피팅(JET ’93~),",
           "   교차진단 추론, 시간 밀도화(COMPASS 초해상)",
           "② 3축 확장 — 전자→희소 이온 / 동시각·무기억→",
           "   인과 이력 / 재구성 가정→사전등록 타겟별 검정",
           "③ 한정은 한 문장만 — “우리가 아는 한 이 조합은",
           "   아직 다뤄지지 않았다”"],
          accent=BLUE, body_size=11.5)

    band(s, 5.85,
         [[("한 문장 — ", 13.5, TEAL, True, False, None),
           ("온라인 가상 센서는 미래를 읽는 보간이 아니라 persistence와 경쟁한다. "
            "그 비교에서 이 나우캐스터는 중요한 지점에서 작동한다.", 13.5, WHITE, False, False, None)]])
    return note(s, """
main_ko.tex 표 tab:stress(§sec:campaign), §sec:conclusion 2문단, docs/paper/NOVELTY.md.
PROJECT_KNOWLEDGE "Framing Rules"(승상님 2026-08-05):
· 두 주장을 분리하라 — "미래를 쓰는 보간을 이긴다"는 관측 모집단에 대한 진술,
  "모든 인과 방법을 이긴다"는 진짜 결측으로 재가중해도 살아남는 진술.
· novelty는 부재로 진술하지 말 것 — 반례 하나로 무너진다. 인용을 많이 할수록 확장 프레임이 강해진다.
· 헤드라인은 "사전등록된 보간들을 이긴다"로 읽어야 한다(§10: 사후 GP 팔과는 동률).
""")


# --- 4. Data & problem setup ---------------------------------------------
def f_data_setup():
    s = slide()
    header(s, "§3  sec:data", "데이터와 문제 설정 — 논문 §3.1–3.3", accent=TEAL)

    fcard(s, 0.55, 1.45, 4.0, 2.55,
          "데이터셋 (§3.1)",
          ["· 641개 KSTAR 방전",
           "  shot 30801–32751",
           "· H-mode ELM 억제(RMP) 구간,",
           "  Dα 활동 부근 ~100 ms 절단",
           "· 공통 10 ms 격자:",
           "  BES 9ch + ECEI 4ch + MC 2ch",
           "  + time + [CES_TI, CES_VT]",
           "· 결측은 타겟별로 독립:",
           "  Tᵢ ≈ 8% · V_rot ≈ 24%"],
          accent=TEAL, body_size=11.5)

    fcard(s, 4.72, 1.45, 4.0, 2.55,
          "샘플 구성 (§3.2)",
          ["W = 4, 타겟 행이 윈도의 마지막",
           "· bes (W,9) · ecei (W,4)",
           "· mc (W,2) · time_features (W,4)",
           "· ces_history (W,4)",
           "· 타겟 (2) + 관측 마스크 m (2)",
           "시간 특징 = lookback·delta 초와",
           "각 log1p. 과거 CES의 신뢰도는",
           "10 ms 전인지 200 ms 전인지에",
           "강하게 의존하기 때문."],
          accent=BLUE, body_size=11.5)

    fcard(s, 8.89, 1.45, 3.89, 2.55,
          "설계 원칙 3종 (§3.3)",
          ["· 가짜 라벨 금지 — impute 없음.",
           "  입력이 완전하고 CES 중 최소",
           "  하나가 관측된 행만 유지",
           "· 타겟별 마스킹 손실 —",
           "  ℒ = Σₖ mₖ(ŷₖ−yₖ)² / Σₖ mₖ",
           "  (옛 필터는 라벨 28%를 폐기)",
           "· 누수 방지 3중 — 파일 단위",
           "  분할 · 학습 파일 전용 z-score",
           "  · 타겟 시점 완전 마스킹"],
          accent=ORANGE, body_size=11.5)

    add_image_fit(s, os.path.join(FIG, "fig_missing.png"),
                  Inches(0.55), Inches(4.15), Inches(7.3), Inches(2.55))

    fcard(s, 8.05, 4.15, 4.73, 2.55,
          "이 그림이 논문에서 하는 일",
          ["fig_missing = 논문 그림 1.",
           "",
           "Tᵢ와 V_rot이 독립적으로 결측된다는",
           "사실이 “행 필터링이 아니라 타겟별",
           "마스킹”이라는 설계 논거를 그림 하나로",
           "세운다.",
           "",
           "이 그림이 없으면 §3.3의 마스킹 손실",
           "항목은 설계 취향처럼 읽힌다."],
          accent=NAVY, body_size=11.5)
    return note(s, """
main_ko.tex §sec:data (324–370행), 그림 fig:missing.

인용 시 주의
· 결측률은 '행 기준·타겟별 독립'이며 두 타겟이 함께 결측되는 비율이 아니다.
· 타겟 시점 마스킹은 값과 관측 플래그를 '모두' 0으로 만든다 — 플래그만 남기면 누수다.
· ces_history 4채널 = 직전 정규화 Tᵢ, 직전 정규화 V_rot, Tᵢ 관측 플래그, V_rot 관측 플래그.
""")


# --- 5. Held (stuck) audit ------------------------------------------------
def f_stuck():
    s = slide()
    header(s, "§3.4  sec:stuck", "데이터 품질 감사 — 유지(forward-fill)된 V_rot", accent=ORANGE)

    fcard(s, 0.55, 1.45, 3.95, 2.35,
          "① 무엇을 발견했나",
          ["관측된 V_rot 값의 54%가",
           "계기 유지값 — 같은 연속 블록",
           "안에서 직전 관측과 비트 단위",
           "동일한 반복.",
           "최대 1,214행 연속,",
           "641개 중 499개 파일 영향.",
           "Tᵢ는 영향 없음(0.0%)."],
          accent=ORANGE, body_size=11.5)

    fcard(s, 4.72, 1.45, 3.95, 2.35,
          "② 평가에서 제외한 결과",
          ["유지값은 전 구간에서 채점에서",
           "제외한다.",
           "물리 단위 V_rot RMSE가",
           "35–55% 커진다(22–33 → 35–46).",
           "유지 타겟의 기준선 오차가",
           "0에 가까웠기 때문 — 보정은",
           "우리에게 불리한 방향이다."],
          accent=RED, body_size=11.5)

    fcard(s, 8.89, 1.45, 3.89, 2.35,
          "③ 학습도 오염시킨다",
          ["학습에서도 제거하면(타겟·이력",
           "값과 플래그·정규화 통계·보간",
           "앵커 전부) 진짜 측정 V_rot이",
           "4개 분할 전부에서 개선:",
           "평균 +0.039 (MSE ≈4% 감소),",
           "3/4 CI가 0 제외, 역방향 없음.",
           "Tᵢ는 무영향(+0.004)."],
          accent=GREEN, body_size=11.5)

    add_image_fit(s, os.path.join(FIG, "fig_stuckfree_paired.png"),
                  Inches(0.55), Inches(3.95), Inches(7.0), Inches(2.55))

    fcard(s, 7.75, 3.95, 5.03, 2.72,
          "본문에 함께 쓸 두 문장",
          ["· 한 번도 시험된 적 없는 축에서 같은 효과가",
           "  재현된다: 유지값 제거 이득이 W = 2,3,4,6,8에서",
           "  +0.088 → +0.048 → +0.035 → +0.003 → +0.006로",
           "  단조 감소. 유지값은 긴 window에 상을 준 게 아니라",
           "  짧은 window를 벌하고 있었다.",
           "· 헤드라인은 이 처리에 불변이다: 재학습해도 Tᵢ는",
           "  4/4 PASS(+0.184 / +0.291 / +0.238 / +0.221),",
           "  V_rot 점추정은 전부 상승하나 3/4은 여전히 n.s."],
          accent=NAVY, body_size=11)
    return note(s, """
main_ko.tex §sec:stuck (372–413행). THESIS_RESULTS.md §1, §8c.

이 절의 핵심 문장: V_rot이 자기 과거에 의존하는 것은 실제 물리인 '동시에' 부분적으로 계측
아티팩트이며, 정직한 서술은 두 몫을 모두 정량화해야 한다.

보정하지 않으면 이 아티팩트 하나가 "V_rot은 긴 이력이 필요하다"는 겉보기 추세(+0.118 → +0.202)를
만들어내고, 보정하면 사라진다 — §6.7 window sweep이 그 대조군이다.
""")


# --- 6. Model -------------------------------------------------------------
def f_model():
    s = slide()
    header(s, "§4  sec:model", "모델 — 201,258 파라미터, 병목은 용량이 아니라 정보", accent=BLUE)

    add_image_fit(s, os.path.join(FIG, "fig_architecture.png"),
                  Inches(0.55), Inches(1.45), Inches(6.6), Inches(5.05))

    fcard(s, 7.35, 1.45, 5.43, 1.55,
          "진단별 인코더",
          ["모달리티별(BES/ECEI/MC) 전용 시간 인지 1-D CNN",
           "(Conv–BN–GELU 2블록, 전역 평균 풀링, 96차원).",
           "시간 특징 전용 소형 CNN 하나 추가.",
           "결측 제거로 샘플링이 불규칙해 순환 구조는 부적합."],
          accent=BLUE, body_size=11)

    fcard(s, 7.35, 3.06, 5.43, 2.05,
          "이력 인코더 — 관측 마스킹 어텐션",
          ["양방향 GRU(64) → 타겟별로 독립인 두 멀티헤드",
           "가산 어텐션 readout.",
           "softmax 이전에 그 타겟의 관측 플래그가 0인 시점을",
           "−∞로 밀어, 어텐션 질량이 실제 측정된 행에만 놓인다.",
           "= 보간의 귀납 편향을 파라미터 비용 0으로 주입.",
           "윈도는 과거 전용 — 양방향성이 미래를 주지 않는다."],
          accent=TEAL, body_size=11)

    fcard(s, 7.35, 5.13, 5.43, 1.57,
          "타겟별 라우팅 · 용량은 레버가 아니다",
          ["Tᵢ 헤드 = 고속+시간+이력(72k) / V_rot 헤드 = 이력+",
           "시간만(14k, 고속 차단 — §6.6이 근거). 출력은 정규화 단위.",
           "~40회 통제 반복에서 용량 확대·복잡 스킵·추가 conv는",
           "전부 무효였다(정규화 위치만이 안정성 전환점)."],
          accent=GRAY, body_size=11)
    return note(s, """
main_ko.tex §sec:model (416–469행).

논문이 아키텍처 기여로 내세우는 지점은 하나다 — "일반적이지 않은 유일한 구성요소는 이 풀링
마스크". 마스킹되지 않은 어텐션 풀은 0으로 채워진 타겟 행과 측정값 없는 이력 행에 가중치를
쓸 수 있어 readout을 희석시키고, 그 억제를 플래그 채널로 암묵 학습하게 떠넘긴다.
하드 마스킹은 같은 편향을 파라미터 비용 0으로 공급한다 — 이것이 §7 선택 프로토콜이 마지막으로
채택한 변경이다.

탐색 경위 자체는 논문에 쓰지 않는다. 쓰는 것은 '용량은 레버가 아니었다'는 음성 결과뿐이다.
""")


# --- 7. Evaluation protocol ----------------------------------------------
def f_eval():
    s = slide()
    header(s, "§5  sec:eval", "평가 방법론 — 스스로에게 불리하게 세운 bar", accent=NAVY)

    fcard(s, 0.55, 1.45, 6.03, 2.35,
          "지표와 채점 모집단",
          ["· skill_PCHIP = 1 − MSE_model / MSE_PCHIP",
           "  (Murphy skill), 물리 단위로 역정규화 후 타겟별.",
           "· 모든 팔이 동일한 (파일, 행) 집합에서 동일한",
           "  타겟별 유지 마스크로 채점된다.",
           "· 보간은 타겟 자신의 값을 제외(누수 없음).",
           "· 세그먼트 경계를 넘지 않으며, 경계 너머가 필요한",
           "  시점은 persistence로 예측(PR2, 커버리지 불변)."],
          accent=NAVY, body_size=11.5)

    fcard(s, 6.75, 1.45, 6.03, 2.35,
          "사전등록 PR1–PR4",
          ["PR1  헤드라인 상대는 PCHIP (사다리도 함께 보고)",
           "PR2  보간은 모델이 채점되는 모든 곳에서 예측,",
           "        미래 이웃이 없으면 persistence로 후퇴",
           "PR3  test 최소 규모 ≥15 shot, ≥3,000 Tᵢ 샘플",
           "PR4  유의 = shot 군집 bootstrap 95% CI가 0 제외",
           "test 셋은 탐색 이전에 예약되었고 선택 중 한 번도",
           "읽지 않았다 → 헤드라인에 winner’s curse가 없다."],
          accent=GREEN, body_size=11.5)

    fcard(s, 0.55, 3.95, 6.03, 2.15,
          "Shot 군집 paired bootstrap",
          ["한 방전 안의 인접 행은 강하게 상관 → 샘플을",
           "독립으로 보면 불확실성이 크게 과소평가된다.",
           "SE_model − SE_PCHIP를 shot 단위로 집계하고",
           "shot 전체를 복원추출(10,000회, 고정 시드).",
           "대가: 유효 표본이 shot 수(≈96) — 이것이 모든",
           "유의성 판정의 검정력을 제한한다(§10 첫 항목)."],
          accent=BLUE, body_size=11.5)

    fcard(s, 6.75, 3.95, 6.03, 2.15,
          "두 채점 모집단 — 항상 함께 명시",
          ["· genuine(진짜 측정만) = 유지값 제외. 헤드라인.",
           "  시드 42 test: 33,693 샘플 / Tᵢ n = 32,787(96 shot)",
           "  / 진짜 V_rot n = 10,729(61 shot)",
           "· stuck0(유지값 포함) = 민감도 확인 전용",
           "  (V_rot n이 27,437로 부푼다)",
           "하나를 인용하며 다른 하나를 주장하는 것이 감사에서",
           "실제로 발견된 오류다(§8h)."],
          accent=ORANGE, body_size=11.5)

    band(s, 6.22,
         [[("MNAR 예고 — ", 12.5, TEAL, True, False, None),
           ("결측은 무작위가 아니므로 관측 시점 skill은 낙관적 추정이다. "
            "명시하고 멈추면 핵심 수치에 크기를 모르는 보정항이 남으므로, §6.4에서 재가중해 "
            "얼마가 살아남는지 보고한다.", 12.5, WHITE, False, False, None)]],
         h=0.72)
    return note(s, """
main_ko.tex §sec:eval (472–524행).

세그먼트 사실(본문에 있음): 641개 shot 전부가 ≥0.5 s 간극으로 나뉜 복수 측정 세그먼트를
가진다 — 파일당 중앙값 2개, 간극 중앙값 6.3 s.

검정력 한계(shot ≈96)를 방법론 절에서 미리 인정하고 §10에서 다시 받는 구조를 유지할 것.
""")


# --- 8. Result 1: causal ladder ------------------------------------------
def f_res_causal():
    s = slide()
    header(s, "§6.1  결과 ①", "모델은 모든 인과 기준선을 압도한다 — 가장 강건한 결과", accent=GREEN)

    col_w = [Inches(3.05), Inches(3.10), Inches(1.75), Inches(1.75)]
    table(s, Inches(0.55), Inches(1.45), col_w,
          ["팔", "정보 접근", "Tᵢ RMSE", "V_rot RMSE"],
          [[("모델 (나우캐스터)", NAVY, True, None), "고속 진단 + 과거 CES",
            ("407.0", GREEN, True, None), ("35.0", GREEN, True, None)],
           ["선형 보간", "과거 + 미래 CES", "441.0", "38.4"],
           ["PCHIP 보간", "과거 + 미래 CES", "449.3", "39.2"],
           ["Persistence", "마지막 관측 CES", "504.3", "44.4"],
           ["AR (국소, 인과)", "과거 CES만", "2425.9", "91.5"]],
          row_h=Inches(0.50), size=12)

    add_image_fit(s, os.path.join(FIG, "fig_rmse_ladder.png"),
                  Inches(0.55), Inches(4.35), Inches(6.0), Inches(2.3))

    fcard(s, 6.75, 1.45, 6.03, 2.35,
          "본문에 쓸 논지",
          ["· 모델이 두 타겟 모두에서 최저 RMSE이며, 실시간에",
           "  실제 사용 가능한 모든 것(persistence·과거 전용 AR)을",
           "  큰 차이로 이긴다.",
           "· 미래 CES가 정의상 없는 온라인 환경에서 명백한 승자.",
           "· 이것이 논문에서 가장 강건한 결과 — 스트레스 테스트",
           "  두 종을 모두 통과하는 주장의 뿌리(§6.4·§6.5).",
           "· 표와 그림은 같은 동결 산출물에서 생성된다."],
          accent=GREEN, body_size=11.5)

    fcard(s, 6.75, 3.95, 6.03, 2.7,
          "캡션에 넣을 주의 세 가지",
          ["· V_rot RMSE가 ≈23이 아니라 ≈35인 이유: 유지값 포함",
           "  모집단에서는 유지 타겟의 기준선 오차가 0에 가깝다",
           "  (§3.4). 이 표는 진짜 측정 기준이다.",
           "· AR의 2425.9는 오타가 아니다 — 국소 선형 외삽은 희소·",
           "  불규칙 격자에서 발산한다. 사다리에 남기는 이유는",
           "  ‘과거만 쓰는 단순 외삽’이 왜 대안이 못 되는지를 보이기",
           "  위함이다.",
           "· 어느 팔이 미래를 읽는지를 표 안에 명시할 것(정보 접근 열)."],
          accent=NAVY, body_size=11.5)
    return note(s, """
main_ko.tex §sec:results 첫 소절, 표 tab:ladder, 그림 fig:ladder (530–566행).
시드 42 held-out test 분할, 진짜 측정 평가 기준.
""")


# --- 9. Result 2: headline ------------------------------------------------
def f_res_headline():
    s = slide()
    header(s, "§6.2  sec:headline", "헤드라인 — Tᵢ는 4개 독립 분할에서 미래를 쓰는 보간을 이긴다",
           accent=GREEN)
    add_image_fit(s, os.path.join(FIG, "fig_forest.png"),
                  Inches(0.55), Inches(1.45), Inches(6.6), Inches(3.55))

    fcard(s, 7.35, 1.45, 5.43, 2.05,
          "인용할 수치 (genuine TEST)",
          ["· Tᵢ  +0.179 [+0.007, +0.302] · +0.197 [+0.091, +0.276]",
           "  +0.280 [+0.192, +0.347] · +0.263 [+0.109, +0.348]",
           "  → 4/4 PASS (유지값 포함에서도 4/4)",
           "· 선형 보간 상대로도 3/4 생존 (시드 42만 +0.148 n.s.)",
           "· V_rot  +0.203 / +0.162 / +0.100 / +0.183 —",
           "  점추정 전부 양수, PASS는 시드 1뿐 → 동률로 보고"],
          accent=GREEN, body_size=11)

    fcard(s, 7.35, 3.58, 5.43, 1.42,
          "사후 GP 팔 — 오프라인 주장의 상한",
          ["GP는 Tᵢ 4개 분할 전부에서 PCHIP을 유의하게 이기는",
           "(+0.21~+0.28) 가장 강한 오프라인 평활기이고,",
           "모델은 그것과 동률(유의 승 1/4, 유의 패 0/4)."],
          accent=ORANGE, body_size=11)

    fcard(s, 0.55, 5.15, 6.6, 1.5,
          "§7과 함께 쓰는 ‘정직한 진행’ 문단",
          ["초기 기준 모델은 유의하지 않았다(+0.088, CI [−0.221, +0.323]).",
           "선택 게이트를 검증 손실 → 깨끗한 보간 대비 skill로 바꾼 것이",
           "최종 모델을 낳았고 4개 분할 모두에서 유의하다(fig_progression)."],
          accent=BLUE, body_size=11)

    fcard(s, 7.35, 5.15, 5.43, 1.5,
          "동률을 해소할 지목된 측정",
          ["· 더 많은 test shot — 시드 7은 이미 해소한다",
           "· 아직 실행하지 않은 인과(과거 전용) GP 팔",
           "배치 주장은 무관 — 미래 앵커 없는 GP는 온라인에 없다."],
          accent=NAVY, body_size=11)
    return note(s, """
main_ko.tex §sec:headline (568–658행), 표 tab:headline, 그림 fig:forest / fig:progression.

인용 시 주의
· 시드 1/7/123은 모델 선택이 전혀 건드리지 않은 진짜 복제다.
· 유지값 포함 Tᵢ: +0.257 / +0.194 / +0.263 / +0.280 (4/4).
· 선형 보간 대비 genuine: 시드 42 +0.148(n.s.), 나머지 +0.167 / +0.259 / +0.234(PASS).
  사전등록된 PCHIP 비교만 인용하지 말고 이 사실을 명시할 것.
· V_rot은 점추정이 4/4 양수인데도 n.s.로 보고한다 — 이 보고 기준 자체를 본문에 쓴다.
· GP 점추정 범위는 −0.08~+0.09이고 V_rot은 전부 n.s.
""")


# --- 10. Result 3: gap stratification ------------------------------------
def f_res_gap():
    s = slide()
    header(s, "§6.3  sec:gap", "간극 층화 — 큰 간극에서 올바른 상대는 persistence다", accent=BLUE)

    col_w = [Inches(2.35), Inches(1.30), Inches(1.15), Inches(3.75), Inches(3.68)]
    table(s, Inches(0.55), Inches(1.45), col_w,
          ["Δt (4 분할 통합)", "n", "shot", "PCHIP 대비 (미래 사용)", "persistence 대비 (인과)"],
          [[("Tᵢ  ≤ 15 ms", NAVY, True, None), "134,629", "298",
            ("+0.262 [+0.20, +0.31]", GREEN, True, None),
            ("+0.407 [+0.36, +0.45]", GREEN, True, None)],
           [("Tᵢ  전체 > 15 ms", NAVY, True, None), "4,496", "272",
            ("+0.191 [+0.10, +0.28]", GREEN, True, None),
            ("+0.388 [+0.32, +0.46]", GREEN, True, None)],
           [("Tᵢ  전체 > 45 ms", NAVY, True, None), "435", "105",
            "−0.057 [−0.45, +0.21]  n.s.",
            ("+0.271 [+0.12, +0.41]", GREEN, True, None)],
           [("Tᵢ  > 105 ms", NAVY, True, None), "167", "62",
            ("−0.542 [−2.07, −0.08]", RED, True, None), "+0.266 [−0.00, +0.46]"],
           [("V_rot  ≤ 15 ms", NAVY, True, None), "51,457", "195",
            "+0.209 [−0.02, +0.29]  n.s.",
            ("+0.368 [+0.17, +0.45]", GREEN, True, None)],
           [("V_rot  전체 > 15 ms", NAVY, True, None), "1,756", "161",
            "+0.027 [−0.20, +0.24]  n.s.",
            ("+0.309 [+0.13, +0.47]", GREEN, True, None)]],
          row_h=Inches(0.48), size=11)

    fcard(s, 0.55, 4.75, 4.0, 1.9,
          "① 우위는 인접 이력에 국한되지 않는다",
          ["4개 분할을 통합하고 bootstrap을",
           "방전 단위로 군집화해 처음으로 넓은",
           "구간에 CI를 붙였다. Δt > 15 ms 전체",
           "+0.191, (15,25]·(25,35]·(55,105]가",
           "각각 단독 유의."],
          accent=GREEN, body_size=11)

    fcard(s, 4.72, 4.75, 4.0, 1.95,
          "② 가장 넓은 간극은 보간의 영역",
          ["105 ms 초과에서 모델은 PCHIP보다",
           "유의하게 나쁘다(−0.542). 그대로 보고.",
           "올바른 진술: “모델이 큰 간극에서",
           "실패한다”가 아니라 “큰 간극은 양측",
           "보간의 영역이고, 그것은 실시간이",
           "가질 수 없는 바로 그것이다”."],
          accent=ORANGE, body_size=11)

    fcard(s, 8.89, 4.75, 3.89, 1.9,
          "③ V_rot은 동률이되 인과는 전부 승",
          ["§6.2의 전역 동률은 미래를 쓰는",
           "방법과의 동률이다.",
           "persistence 대비로는 작은 간극",
           "(+0.368)에서도, 15 ms를 넘어서도",
           "(+0.309) 유의하게 우수하다."],
          accent=TEAL, body_size=11)
    return note(s, """
main_ko.tex §sec:gap (660–726행), 표 tab:gap.

인용 시 주의
· 실제 타겟의 ≥96%가 Δt ≤ 15 ms 영역에 있다.
· Δt가 커질수록 PCHIP의 문제는 쉬워지고(양쪽 실측을 잇기만 하면 된다) 우리 문제는 어려워진다.
  이 비대칭을 본문에 명시해야 −0.542가 정직하게 읽힌다.
· 시드별로는 넓은 구간 표본이 수십 개뿐이라 이전 초안은 결론을 지탱할 수 없었다 —
  4개 분할 통합 + shot 군집화가 이 절을 가능하게 한 방법론적 변경이다.
""")


# --- 11. Result 4: MNAR ---------------------------------------------------
def f_res_mnar():
    s = slide()
    header(s, "§6.4  sec:mnar", "실제로 결측인 지점에서 얼마나 살아남는가", accent=ORANGE)

    box(s, Inches(0.55), Inches(1.42), Inches(12.23), Inches(1.00),
        fill=RGBColor(0xFF, 0xF3, 0xE6), round_=True)
    box(s, Inches(0.55), Inches(1.42), Inches(0.12), Inches(1.00), fill=ORANGE)
    text(s, Inches(0.85), Inches(1.51), Inches(11.6), Inches(0.9),
         [[("먼저 적용 범위 사실 — ", 13, ORANGE, True, False, None),
           ("표본이 예측 가능하려면 그 타겟의 관측값이 모델 window 안에 있어야 한다. "
            "진짜 결측 행 중 도메인 내인 것은 ", 13, DARK, False, False, None),
           ("Tᵢ 54.1% · V_rot 4.8%", 13, RED, True, False, None),
           ("뿐이다.", 13, DARK, False, False, None)],
          [("정확도 한계가 아니라 커버리지 한계이며 구조적으로 고칠 수 있다(§9 레버 1).",
            11.5, GRAY, False, False, None)]],
         line_spacing=1.2)

    col_w = [Inches(3.30), Inches(1.20), Inches(2.30), Inches(3.55), Inches(1.88)]
    table(s, Inches(0.55), Inches(2.50), col_w,
          ["비교", "시드", "관측 가중", "결측 정합 (95% CI)", "낙관 편향"],
          [[("Tᵢ vs persistence", NAVY, True, None), "42", "+0.349",
            ("+0.292 [+0.162, +0.415]", GREEN, True, None), "+0.056"],
           ["", "1", "+0.380", ("+0.308 [+0.143, +0.429]", GREEN, True, None), "+0.072"],
           ["", "7", "+0.405", ("+0.293 [+0.192, +0.441]", GREEN, True, None), "+0.112"],
           ["", "123", "+0.415", ("+0.293 [+0.155, +0.392]", GREEN, True, None), "+0.121"],
           [("Tᵢ vs PCHIP", NAVY, True, None), "42", "+0.146", "+0.061 [−0.075, +0.267]", "+0.084"],
           ["", "1", "+0.188", "+0.132 [−0.064, +0.274]", "+0.056"],
           ["", "7", "+0.280", ("+0.211 [+0.050, +0.374]", GREEN, True, None), "+0.069"],
           ["", "123", "+0.264", "+0.175 [−0.025, +0.290]", "+0.090"]],
          row_h=Inches(0.33), size=10.5)

    fcard(s, 0.55, 5.66, 6.03, 1.24,
          "방법 — 한 문단으로",
          ["Δt × 국소 활동도로 사후 층화해 재가중. 활동 플래그는",
           "자기 행을 제외한 이웃에서 계산되므로 결측 행에서도 정의된다."],
          accent=BLUE, body_size=11)

    fcard(s, 6.75, 5.66, 6.03, 1.24,
          "결론 문장 (그대로 인용 가능)",
          ["진짜 결측·도메인 내 시점에서 이 나우캐스터는 어떤 인과적 CES 전용",
           "방법보다 유의하게 낫고(4/4, +0.29), 오프라인 우위는 미입증(1/4)."],
          accent=GREEN, body_size=11)
    return note(s, """
main_ko.tex §sec:mnar (728–799행), 표 tab:mnar.

· MNAR 보정 비용은 0.06–0.12에 불과하다(인과 비교에 대해).
· 층 커버리지 95–100%, 채점 표본 30개 미만 층은 버린다. 가정(층 안에서 결측·관측 행이 교환 가능)은
  숨기지 않고 명시한다. 대조 확인: 활동도 비율이 동결 산출물의 플래그와 Tᵢ 0.004 / V_rot 0.009 이내 일치.
· 두 비교가 갈라지는 것이 이 절에서 가장 쓸모 있는 결과 — §6.3과 기계적으로 일관된다
  (결측 지점은 더 큰 Δt에 있고, 그곳이 양측 앵커가 가장 크게 돕는 곳).
· V_rot은 기반이 얇아(도메인 내 4.8%, persistence 대비 2/4) 회전에 대한 배치 결론은 내지 않는다.
""")


# --- 12. Result 5: campaign shift + repair -------------------------------
def f_res_campaign():
    s = slide()
    header(s, "§6.5  sec:campaign", "캠페인 이동 — 실패를 측정하고, 지목한 수리를 실행했다",
           accent=RED)

    fcard(s, 0.55, 1.45, 4.0, 2.35,
          "① 설계 — 시간으로 자른 분할",
          ["shot 번호로 정렬해 엄격히 절단:",
           "train 416 [30801, 31991]",
           "val 128 [32002, 32310]",
           "test 97 [32312, 32751]",
           "어떤 test shot도 어떤 train shot",
           "보다 앞서지 않는다. 바뀐 변수는",
           "분할 규칙뿐이고, 네 실행은",
           "초기화 시드만 다르다."],
          accent=NAVY, body_size=11)

    fcard(s, 4.72, 1.45, 4.0, 2.35,
          "② 결과 — 오프라인 주장이 죽는다",
          ["PCHIP 대비: +0.051 / +0.044 /",
           "  −0.148 / −0.018 → 평균 −0.018,",
           "  0/4 PASS",
           "persistence 대비: +0.275 / +0.270",
           "  / +0.123 / +0.223 → 평균 +0.222",
           "test 구간은 모든 팔에게 더 어렵지만",
           "skill이 그것을 정규화한다 — 즉",
           "보간 ‘대비’ 손실이다."],
          accent=RED, body_size=11)

    fcard(s, 8.89, 1.45, 3.89, 2.35,
          "③ 원인 — 주장이 아니라 측정",
          ["train→test 드리프트(정규화 단위,",
           "중앙값):",
           "· BES 1.22 σ (스케일비 0.75)",
           "· ECEI 0.53 σ (0.62)",
           "· CES 타겟 0.115 σ (1.06)",
           "고속 진단이 예측 대상보다 5–11배",
           "더 이동한다. 이력 경로는 온전해",
           "persistence 마진은 유지된다."],
          accent=ORANGE, body_size=11)

    fcard(s, 0.55, 3.85, 6.03, 2.35,
          "④ 수리 — 고속 진단의 shot별 표준화 (§8s)",
          ["각 방전의 고속 진단을 그 방전 자신의 평균·분산으로",
           "적재 시점에 표준화(타겟 불변 → 기준선도 불변).",
           "· 캠페인 분할: 짝지은 Tᵢ가 4/4 초기화에서 +0.10~+0.29",
           "  개선(평균 +0.155), 95% CI 모두 0 제외.",
           "  PCHIP 관문 0/4 → 2/4 (+0.18, +0.14, +0.19, +0.08)",
           "· V_rot 불변(−0.019) — 예상된 음성 대조군",
           "· 헤드라인 분할 비용: 어느 시드도 유의한 손실 없음",
           "  (평균 −0.036, 최악 −0.127이며 CI 상한 +0.008)"],
          accent=GREEN, body_size=11)

    fcard(s, 6.75, 3.85, 6.03, 2.35,
          "⑤ ‘공짜’가 아니라 ‘측정 가능한 비용 없음’",
          ["· 수리는 기제도 인과적으로 확정한다: 캠페인 손실이",
           "  물리의 변화였다면 입력의 수준 이동을 제거하는 것으로",
           "  회복될 수 없었다.",
           "· 정직한 주의: shot별 표준화는 절대 수준 정보도 버리고",
           "  그것이 Tᵢ 정보를 나를 수 있다(BES 높음 ↔ 밀도 높음).",
           "  그래서 가정이 아니라 통제 실험으로 확인했다.",
           "· 남은 미검증: 배치 가능한 형태(인과 running / EWMA).",
           "  shot별 표준화는 이 계열의 오프라인 상한이다."],
          accent=NAVY, body_size=11)
    return note(s, """
main_ko.tex §sec:campaign (801–891행), 표 tab:campaign / tab:stress. THESIS_RESULTS §8n, §8s.

· 분할이 고정이므로 네 실행은 '초기화' 시드만 다르다 — 이를 "4개 시드"로 제시하지 말고 그대로 밝힌다.
· test 구간의 PCHIP RMSE는 601(무작위 분할 449).
· V_rot이 안 변하는 것이 오히려 기제 확인이다: V_rot 헤드는 구조상 고속 진단이 차단돼 있다.
· 이 절의 서사 구조 자체가 기여다 — 실패를 보고하고, 원인을 측정하고, 지목한 수리를 실행해 되돌렸다.
""")


# --- 13. Result 6: asymmetry ---------------------------------------------
def f_res_asym():
    s = slide()
    header(s, "§6.6  sec:asym", "Tᵢ ↔ V_rot 정보 비대칭 — 본 연구의 과학적 발견", accent=ORANGE)

    col_w = [Inches(3.5), Inches(1.85), Inches(1.85)]
    table(s, Inches(0.55), Inches(1.45), col_w,
          ["입력 모달리티 절제 (val, vs persistence)", "Tᵢ skill", "V_rot skill"],
          [["전체 (이력 + 고속 진단 + 시간)", "+0.43", "+0.30"],
           ["이력만 (no_fast)", "+0.46", "+0.29"],
           ["고속 진단만 (no_history)",
            ("+0.37", GREEN, True, None), ("−0.64", RED, True, None)]],
          emphasis={2}, emphasis_fill=RGBColor(0xFF, 0xF3, 0xE6), size=12.5)

    text(s, Inches(0.55), Inches(3.35), Inches(6.6), Inches(1.0),
         [[("읽는 법: ", 12.5, NAVY, True, False, None),
           ("고속 진단만 주면 Tᵢ는 여전히 persistence를 크게 이기지만(+0.37 = 전체 +0.43의 대부분), "
            "V_rot은 −0.64이고 평균 예측 대비 R² = −0.06 — ", 12.5, DARK, False, False, None),
           ("입력을 무시하고 평균을 내놓는 것보다 엄격히 나쁘다.", 12.5, ORANGE, True, False, None)]],
         line_spacing=1.25)

    add_image_fit(s, os.path.join(FIG, "fig_ablation.png"),
                  Inches(0.55), Inches(4.35), Inches(6.0), Inches(2.3))

    fcard(s, 6.75, 1.45, 6.03, 2.35,
          "물리적 근거 — 절제 이전에 예측되었다",
          ["· Tᵢ: 충돌 전자–이온 결합 (t_ei ∝ Tₑ^{3/2}/nₑ).",
           "  ECEI가 Tₑ를, BES가 nₑ 구조를 공급한다.",
           "· V_rot: 토로이달 회전은 우리 입력이 관측하지 못하는",
           "  운동량 소스 — 외부 NBI 토크와 고유 회전 구동 —",
           "  이 지배한다.",
           "· Mirnov는 100 Hz로 샘플되어 회전을 대리할 수 있었던",
           "  kHz 모드 회전 주파수가 앨리어싱으로 사라진다."],
          accent=ORANGE, body_size=11.5)

    fcard(s, 6.75, 3.95, 6.03, 2.7,
          "이 절이 논문에서 하는 세 가지 일",
          ["· V_rot의 비-승리를 모델 실패가 아니라 진단 정보 내용에",
           "  관한 발견으로 재정의한다.",
           "· §4의 타겟별 라우팅(V_rot 헤드에서 고속 진단 차단)을",
           "  사후 합리화가 아닌 근거 기반 설계로 만든다.",
           "· §9 레버 2·3(원 kHz Mirnov, NBI 토크)의 출발점을 놓는다.",
           "인용 시 주의: 검증 분할 절제이며, 동일 학습 예산 · 한 번에",
           "한 모달리티 그룹만 0 · persistence는 항상 실제 이력에서",
           "계산한다."],
          accent=NAVY, body_size=11.5)
    return note(s, """
main_ko.tex §sec:asym (893–942행), 표 tab:ablation, 그림 fig:ablation.
이 비대칭은 물리로 먼저 예측되었고 절제로 확인되었다 — 순서를 본문에서 그대로 밝힐 것.
""")


# --- 14. Result 7: window + complexity ladder ----------------------------
def f_res_window():
    s = slide()
    header(s, "§6.7–6.8", "이력은 관측 하나면 충분하고, 복잡도가 사는 것은 정량화된다",
           accent=GREEN)

    col_w = [Inches(1.55), Inches(1.20), Inches(1.30), Inches(0.85), Inches(1.30), Inches(0.85)]
    table(s, Inches(0.55), Inches(1.45), col_w,
          ["이력 관측 수", "W", "Tᵢ skill", "PASS", "V_rot", "PASS"],
          [["0 (no_history)", "4", ("−0.026", RED, True, None), "0/4",
            ("−0.783", RED, True, None), "0/4"],
           ["1", "2", ("+0.238", GREEN, True, None), "4/4",
            ("+0.206", GREEN, True, None), "0/4"],
           ["2", "3", ("+0.246", GREEN, True, None), "4/4", "+0.203", "1/4"],
           ["3", "4 (기본)", "+0.221", "3/4", "+0.190", "1/4"],
           ["5", "6", "+0.190", "3/4", "+0.205", "1/4"],
           ["7", "8", "+0.216", "4/4", "+0.204", "2/4"]],
          row_h=Inches(0.42), size=11)

    add_image_fit(s, os.path.join(FIG, "fig_window_sweep.png"),
                  Inches(7.0), Inches(1.45), Inches(5.6), Inches(2.75))

    fcard(s, 0.55, 4.33, 6.03, 2.35,
          "선택 규칙과 그 답 — W = 2",
          ["· 이력을 완전히 제거하면 Tᵢ는 PCHIP 아래로(−0.026),",
           "  V_rot은 −0.78. 마진 전체가 고속 진단과 과거 CES의",
           "  결합에서 나온다.",
           "· 단 하나의 과거 관측이 두 타겟을 동시에 최대치로 올리고",
           "  이후 곡선은 평탄(Tᵢ 0.190–0.246, V_rot 폭 0.016).",
           "  점 하나 안의 seed 산포 0.07–0.16이 곡선보다 넓다.",
           "· 넓은 window의 유일한 근거는 skill이 아니라 커버리지:",
           "  W=2→8에서 Δt>15 ms 456→1,958, >45 ms 14→135."],
          accent=GREEN, body_size=11)

    fcard(s, 6.75, 4.33, 6.03, 2.35,
          "복잡도 사다리 (§6.8) — 앵커+Δ, 1,258 파라미터",
          ["앵커(최근접 관측을 학습된 가중으로 평균 쪽 혼합)",
           "+ 기울기(최근 2점 × 갭) + 진단별 변화율의 합.",
           "학습 항이 전부 0 초기화 → persistence에서 정확히",
           "출발하므로 도달 skill은 학습이 더한 값 그 자체다.",
           "· Tᵢ: −0.272 → −0.113 → +0.234 ⇒ 격차의 31.5% 회수",
           "· V_rot: 7.0%만 회수 — 그 신호는 국소 기울기가 아니다",
           "· paired 비교는 앵커를 4/4 분할에서 기각한다.",
           "  부족분은 모델이 아니라 문제에 대한 정보다."],
          accent=BLUE, body_size=11)
    return note(s, """
main_ko.tex §sec:window (944–1008행) 표 tab:window, §sec:ladder (1010–1069행) 표 tab:ladder2.

인용 시 주의
· 24개 독립 실행(W ∈ {2,3,4,6,8} × seed {42,1,7,123}) + history-0 ×4. 전 구간 유지값 제거 학습.
· paired가 아닌 독립 실행이므로 곡선 위 차이는 전부 seed 잡음 안 — 주장은 순위가 아니라 '효과의 부재'다.
· shot당 표본 상한 때문에 절대 skill을 헤드라인 계열과 직접 비교할 수 없다(곡선 내부 비교용).
· 통제 검증: 24개 실행 전부 seed당 동일한 96개 test shot을 평가하고, 채점 표본은 W=2→8에서 1.8%만 줄었다.
· 앵커+Δ는 decompose()로 항별 기여를 반환한다 — "앵커 + 기울기 + BES 기반 변화율"로 읽을 수 있다.
""")


# --- 15. Result 8: peak ---------------------------------------------------
def f_res_peak():
    s = slide()
    header(s, "§6.9  sec:peak", "우위는 고변동 국소 구간에 집중된다 — 그리고 V_rot은 분해된다",
           accent=TEAL)
    add_image_fit(s, os.path.join(FIG, "fig_peak.png"),
                  Inches(0.55), Inches(1.45), Inches(6.3), Inches(3.3))

    fcard(s, 7.05, 1.45, 5.73, 1.75,
          "수치 (검증 분할, shot 군집 CI)",
          ["· Tᵢ  전역 +0.27 → 피크 +0.70 [+0.50, +0.85] PASS",
           "  (124개 shot의 4,764행)",
           "· V_rot  전역 +0.13 → 피크 +0.44 [+0.07, +0.73] PASS",
           "· 피크 한정 절제: Tᵢ는 유의하게 나빠지고(멀티모달을",
           "  실제로 사용) V_rot은 불변(이력 기반)"],
          accent=TEAL, body_size=11)

    fcard(s, 7.05, 3.30, 5.73, 1.45,
          "피크 선정이 순환 논리가 아닌 이유",
          ["타겟 행을 제외하고 계산한 입력 측 활동 대리변수(이웃 괄호",
           "기울기 / 국소 CES 이웃 분산)로 나눈다 — 타겟 자신의 값을",
           "보지 않는다. 단 검증 분할이므로 헤드라인으로 쓰지 않는다."],
          accent=GRAY, body_size=11)

    fcard(s, 0.55, 4.9, 12.23, 1.75,
          "V_rot 결과의 분해 (§8r) — 전역 동률은 부호가 다른 세 영역의 평균이다",
          ["· 예상과 반대인 사실: 피크가 벌크보다 유지값이 더 많다(4개 분할 전부 68/62/71/73% 대 58/51/48/46%).",
           "  forward-fill 계단(평평–평평–도약) 자체가 큰 국소 기울기여서 활동 검출기가 계측기 패턴을 부분 검출한다.",
           "· 진짜 측정된 고활동 행: PCHIP 대비 +0.55~+0.63 · 인과 기준선 대비 +0.75~+0.82로 4/4 유의.",
           "  진짜 측정된 조용한 벌크: 보간과 동률(≈0). 유지값 행: −48 ~ −411 — 실패가 아니라 구조적이다.",
           "· Tᵢ는 유지값이 사실상 없는 깨끗한 대조군으로 피크 집중을 4/4 재현한다(+0.59~+0.68, 전부 유의)."],
          accent=ORANGE, body_size=11)
    return note(s, """
main_ko.tex §sec:peak (1071–1121행) + 유지값 교차 문단, THESIS_RESULTS §8r.

· 채점을 진짜 측정 행으로 한정하면 모든 분할이 상승하지만(+0.154→+0.201, +0.109→+0.126,
  +0.065→+0.098, +0.127→+0.170) 사전등록 관문에는 못 미친다 ⇒ 유지값 희석은 실재하나
  V_rot 동률의 전부는 아니고, 나머지는 검정력이다.
· 유지값 행에서 지는 것은 구조적이다 — PCHIP은 정의상 직전 값인 그 값을 정확히 통과하므로
  어떤 인과 방법도 그곳에서는 이길 수 없다.
· 이를 더 날카롭게 할 지목된 측정: 유지값을 배제한 활동 검출기 — 아직 실행하지 않았다.
""")


# --- 16. Deployability ----------------------------------------------------
def f_deploy():
    s = slide()
    header(s, "§8  sec:deploy", "배치 가능한가 — 지연과 불확실성, 둘 다 예상 밖의 답", accent=BLUE)

    fcard(s, 0.55, 1.45, 6.03, 2.5,
          "지연: CPU에서는 들어가고, GPU가 틀린 장치다",
          ["· 측정: 순전파만(특징 조립 제외), warmup 후 1,000회,",
           "  batch 1.",
           "· CPU W=4: p99 6.4 ms, 중앙값 2.8 ms (W=2는 p99 8.7 ms)",
           "  → 10 ms 격자 한 주기의 64–87%",
           "· 같은 기계에서 CUDA batch-1은 약 8× 느리다",
           "  (중앙값 21 ms, p99 43–72 ms). 20만 파라미터로는",
           "  커널 실행 오버헤드를 상쇄할 것이 없다.",
           "· 실무 지침은 기본값의 반대: 제어 컴퓨터의 CPU에서 돌려라."],
          accent=BLUE, body_size=11)

    fcard(s, 6.75, 1.45, 6.03, 2.5,
          "불확실성: 모델을 건드리지 않는 분포 무가정 구간",
          ["· 분산/분위 헤드는 재학습이 필요하고 점 예측을 움직여",
           "  위의 모든 수치를 교란한다 → split conformal.",
           "  val에서 캘리브레이션, 예측기는 전혀 바꾸지 않는다.",
           "· 두 변형: 단일 분위(global), Δt × 활동도 층별(Mondrian).",
           "  두 기준선에도 동일 절차를 적용하므로 비교되는 것은",
           "  캘리브레이션 기법이 아니라 구간 품질이다.",
           "· α = 0.10에서 모델 구간이 모든 시드·타겟·변형에서 두",
           "  기준선을 이긴다(각 8/8). Winkler로 persistence의 ≈0.80배."],
          accent=TEAL, body_size=11)

    fcard(s, 0.55, 4.1, 6.03, 1.6,
          "비대칭은 불확실성에도 약하게 나타난다",
          ["커버리지를 맞춘 상태에서 모델은 구간을 Tᵢ에 대해",
           "persistence 폭의 0.884배로 줄이지만 V_rot에 대해서는",
           "0.937배에 그친다 — 점 예측과 같은 방향이되 크기는",
           "훨씬 작다. 모델은 자기가 모르는 것을 부분적으로 안다."],
          accent=ORANGE, body_size=11)

    fcard(s, 6.75, 4.1, 6.03, 1.6,
          "정직한 실패 — 주변 커버리지지 조건부가 아니다",
          ["Tᵢ는 4개 중 2개 분할에서 목표 90%에 미달(87.0–88.9%),",
           "shot별 커버리지는 10분위 ≈50–68%에서 90분위 100%까지",
           "퍼진다. 캘리브레이션과 test가 서로 다른 방전이고 shot",
           "수준 이동이 교환 가능성을 깬다."],
          accent=RED, body_size=11)

    band(s, 5.9,
         [[("이 절을 쓰는 이유 — ", 13, TEAL, True, False, None),
           ("skill 점수와 쓸 수 있는 계측 사이에는 두 가지가 놓여 있다. 측정 주기 안에서 돌아야 하고, "
            "얼마나 믿어야 하는지를 말해야 한다. 둘 다 이 문헌에서 보고된 적이 없다.",
            13, WHITE, False, False, None)]])
    return note(s, """
main_ko.tex §sec:deploy (1146–1197행). PROJECT_KNOWLEDGE "Deployment Facts".

· 지연은 호출별(장치를 사이에 유휴로 두는 10 ms 루프의 현실적 모형)과 연속 amortized 두 방식으로
  측정했고 둘이 일치하므로 GPU 유휴 아티팩트가 아니다.
· 대량 재처리도 여기서는 CPU가 유리했지만(batch 512에서 48k 대 24k samples/s) 노트북 GPU +
  매우 작은 신경망에 특정한 결과이므로 일반화하지 않는다는 단서를 함께 쓸 것.
· Mondrian은 평균 구간을 넓히면서도 구간 점수를 모든 곳에서 개선한다 — Δt와 활동도가 필요하다고
  말하는 곳에 폭을 배치한다는 뜻이다.
· 조건부 커버리지를 맞추려면 shot 조건부 캘리브레이션이 필요한데 현재 shot 수로는 지탱되지 않는다.
""")


# --- 17. Headroom ---------------------------------------------------------
def f_headroom():
    s = slide()
    header(s, "§9  sec:headroom", "남은 개선 여지 — 음성 결과가 지목하는 레버 3종", accent=ORANGE)

    band(s, 1.42,
         [[("이 절의 규칙 — ", 12.5, TEAL, True, False, None),
           ("음성 결과는 “어떤 변경이 문제를 움직이는가”를 말할 때만 실린다. 용량은 배제됐고(§4), "
            "더 긴 이력도 배제됐다(§6.7). 남는 셋은 모두 추정기가 아니라 측정 또는 커버리지에 대한 변경이다.",
            12.5, WHITE, False, False, None)]],
         h=0.72)

    fcard(s, 0.55, 2.30, 3.95, 3.76,
          "1. 깊이가 아니라 도달 범위",
          ["진짜 결측 행 중 W=4 window 안에",
           "관측값을 가진 것은 Tᵢ 54.1% ·",
           "V_rot 4.8%뿐(§6.4).",
           "",
           "올바른 대응은 더 긴 연속 window가",
           "아니다 — skill을 전혀 사지 못한다.",
           "더 넓은 도달 범위다: W=2→8에서",
           "총 채점 수는 그대로인데 Δt>15 ms는",
           "456→1,958, >45 ms는 14→135.",
           "",
           "→ 슬롯은 2–3개로 유지하되 더 넓은",
           "span에서 뽑는다. 실제 문제 중",
           "다루는 비율을 직접 키우는 유일한",
           "변경이다."],
          accent=BLUE, body_size=11)

    fcard(s, 4.72, 2.30, 3.95, 3.76,
          "2. Mirnov 정보는 파괴됐다",
          ["같은 10 ms 격자에서 블록 내 lag-1",
           "자기상관: BES +0.568 ·",
           "ECEI +0.572 · Mirnov −0.009,",
           "블록의 82%가 |r| < 0.1.",
           "즉 자기 채널은 이 격자 위에서",
           "백색잡음이다.",
           "원인: kHz dB/dt를 안티앨리어싱",

           "없이 100 Hz로 데시메이션 →",
           "상대 위상이 무작위. 회전을",
           "대리할 수 있었던 모드 회전",
           "주파수가 상류에서 폐기됐다.",
           "",
           "→ 해법은 전처리 변경: 원 kHz",
           "시계열의 window별 RMS·대역",
           "파워·모드 번호."],
          accent=TEAL, body_size=11)

    fcard(s, 8.89, 2.30, 3.89, 3.76,
          "3. 구동기 채널이 아예 없다",
          ["토로이달 회전은 주입 토크가",
           "결정하는데 이 데이터셋에 토크",
           "신호가 없다.",
           "",
           "shot 간 ECE 유래 Tₑ 대리는",
           "· Tᵢ와 r = +0.353 (p = 3×10⁻¹⁷)",
           "· V_rot와 r = +0.024 (p = 0.58)",
           "shot 내에서도 Tᵢ 상관은 부호가",
           "일관되나 V_rot은 무작위다.",
           "즉 파워는 토크가 아니다.",
           "",
           "→ NBI 토크(또는 빔별 파워·기하)",
           "확보는 모델링이 아니라 데이터의",
           "문제. 양성 대조군도 문헌에 있다",
           "(DIII-D 전체-방전 시뮬레이터)."],
          accent=ORANGE, body_size=11)

    band(s, 6.14,
         [[("맺는 문장 — ", 12.5, TEAL, True, False, None),
           ("이 중 어느 것도 현재 결과가 천장에 도달했다는 진술이 아니다. 시도 비용이 낮은 순으로 "
            "정렬돼 있고, 셋 모두 아카이브된 KSTAR 데이터에서 실행 가능하다.",
            12.5, WHITE, False, False, None)]],
         h=0.72)
    return note(s, """
main_ko.tex §sec:headroom (1200–1250행).

PROJECT_KNOWLEDGE "Framing Rules"(승상님 2026-08-05): 음성 결과는 그것을 뒤집을 측정을
함께 지목할 때만 결론이 된다. "정보가 부족하다"는 결론이 아니라 변명이다.

· 레버 1은 skill이 이력 길이에 평탄하다는 사실 덕분에 거의 공짜인 적용 가능성 확장이다.
· 레버 2가 우리가 시도한 모든 MC 파생 특징(적분, PCHIP 적분, |MC|, 이동 RMS)이 실패한 이유다 —
  이미 잃은 정보는 하류에서 복원되지 않는다.
· 레버 3의 양성 대조군: Char et al. 2024, DIII-D 순환 전체-방전 시뮬레이터가 빔 액추에이터를
  입력으로 받아 회전 전개를 예측한다 — 이 채널은 측정만 되면 학습 가능하다.
""")


# --- 18. Limits + conclusion ---------------------------------------------
def f_limits_conclusion():
    s = slide()
    header(s, "§10–11", "한계와 결론 — 무엇을 인정하고 무엇을 주장하는가", accent=NAVY)

    fcard(s, 0.55, 1.45, 6.03, 5.2,
          "§10 한계 — 논문이 먼저 인정하는 것",
          ["· 통계적 검정력: 재현 단위는 shot이고 test에 ≈96(Tᵢ) /",
           "  91(V_rot)개뿐이며 shot별 오차 차이는 꼬리가 두껍다.",
           "· MNAR 낙관성: skill은 관측 시점에서만 측정된다.",
           "· 오프라인 주장의 상한은 GP 동률 — “미래를 쓰는 보간을",
           "  이긴다”는 “사전등록된 보간들을 이긴다”로 읽어야 한다.",
           "· CES 적합실패 아티팩트는 헤드라인을 부풀리는 게 아니라",
           "  끌어내린다: Tᵢ > 3 keV 값(행의 0.4–0.6%)을 모든 팔에서",
           "  동일하게 제거하면 4/4 PASS를 유지한 채 skill이 약 2배",
           "  (+0.18~+0.28 → +0.36~+0.59) ⇒ 헤드라인은 보수적이다.",
           "· 지표 비대칭: 보간은 shot 전체 이웃을, 모델은 W=4 이력만",
           "  본다(의도적으로 불리하나 해석을 복잡하게 한다).",
           "· 단일 장치·단일 계열. 우리 헤드라인에 대해서도 보고한다 —",
           "  전체 격자 시퀀스 변형(동일 라우팅 + 유지값 제거 + shot별",
           "  표준화)이 4개 분할 전부에서 더 높은 Tᵢ skill(평균 +0.045,",
           "  1/4 유의), 학습 비용 ≈1/10. 사전등록이 모델을 미리",
           "  지명하므로 헤드라인을 다시 쓰지는 않는다.",
           "· 캠페인 수리의 배치 가능한 형태(인과 running)는 미검증.",
           "· 불확실성은 주변적으로만 캘리브레이션된다(shot별 50–100%).",
           "· 가장 넓은 간극은 여전히 62 shot의 167 표본이다.",
           "· 범위: 페데스탈-상단 프레이밍은 데이터 선택에서 상속됐고",
           "  반경 의존성·이벤트 위상 분석은 수행하지 않았다."],
          accent=RED, body_size=11)

    fcard(s, 6.75, 1.45, 6.03, 2.95,
          "§11 결론 — 세 문단의 뼈대",
          ["① 관측 모집단: 이온온도에 대해 미래를 쓰는 PCHIP을",
           "  유의하고 재현된 skill(+0.18~+0.28, 4개 독립 분할)로",
           "  이기고 — 가장 강한 오프라인 평활기 GP와는 동률 —",
           "  두 타겟 모두에서 모든 인과 기준선을 큰 차이로 이긴다.",
           "  검정력이 없어 아무 말도 못 하던 Δt > 15 ms를 포함해서.",
           "② 배치 주장은 더 좁고 더 잘 뒷받침된다: 두 스트레스",
           "  테스트가 같은 답을 준다 — 인과 대비는 둘 다 통과",
           "  (+0.29, +0.22), 오프라인 대비는 어느 것도 통과 못 한다",
           "  (1/4, 0/4). MNAR 보정 비용은 0.06–0.12.",
           "③ 작동하지 않는 지점과 그 이유도 함께 보고하며, 각각은",
           "  구체적이고 검증 가능한 변경을 지목한다."],
          accent=GREEN, body_size=11)

    fcard(s, 6.75, 4.55, 6.03, 2.1,
          "결론 문단이 명시하는 기여 4가지",
          ["① 보정된 V_rot 평가 — 관측값의 54%가 계기 유지값이며",
           "  평가뿐 아니라 학습도 오염시킨다는 것을 4-seed로 입증",
           "② 관측 마스킹 어텐션 풀링 — 보간의 귀납 편향을 무비용 주입",
           "③ 복잡도 사다리 — 불투명성의 가격을 “해석 가능한 1,258",
           "  파라미터가 회수하는 Tᵢ 마진의 31.5%”로 매긴다",
           "④ 깨끗한 보간 대비 skill을 게이트로 삼는 선택 프로토콜"],
          accent=NAVY, body_size=11)
    return note(s, """
main_ko.tex §sec:limits (1253–1299행), §sec:conclusion (1302–1338행).

재현성·코드 공개·데이터 공개 문단은 이미 논문에 있다(1341–1360행) — 별도 슬라이드로 만들지 않는다.
코드 공개에 남은 TODO 하나: 투고본에 대한 아카이브 DOI(Zenodo) 발급 후 인용 추가.
""")


def build():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    f_title()
    f_map()
    f_two_claims()
    f_data_setup()
    f_stuck()
    f_model()
    f_eval()
    f_res_causal()
    f_res_headline()
    f_res_gap()
    f_res_mnar()
    f_res_campaign()
    f_res_asym()
    f_res_window()
    f_res_peak()
    f_deploy()
    f_headroom()
    f_limits_conclusion()
    prs.save(OUT)
    print(f"wrote {OUT}  ({len(prs.slides.__iter__.__self__._sldIdLst)} slides)")
    for w in _WARNED:
        print("  FIT WARNING:", w)
    if not _WARNED:
        print("  layout: every card and band fits at its chosen type size")


if __name__ == "__main__":
    build()
