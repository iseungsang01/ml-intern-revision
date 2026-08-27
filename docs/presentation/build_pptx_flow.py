# -*- coding: utf-8 -*-
"""Build the **paper-writing digest** deck — "논문에 들어가는 것만" (abstract register).

Output: docs/presentation/KSTAR_CES_연구흐름.pptx

This is a third, different-purpose deck. The other two are *presentation* decks:

    build_pptx.py        -> 1-hour thesis defense                (결과를 설득하는 덱)
    build_pptx_20min.py  -> 20 min seminar                       (결과를 압축한 덱)
    build_pptx_flow.py   -> this file, 23 slides                 (논문을 쓰기 위한 덱)

Where the result decks answer "이 결과를 믿어도 되는가", this one answers "논문의 이
절에 무엇을 쓰고, 어느 수치를 인용하며, 근거는 어디에 있는가". **슬라이드 한 장 =
`docs/paper/main_ko.tex`의 한 절**이고, 각 장의 노트에 그 절의 `\\label`과 인용 시
유의점이 적혀 있다.

2026-08-27 재작성. (1) 승상님 지시에 따라 모든 카드·밴드·표·노트를 논문 초록 문체
(서술형 종결, 객관·비인칭)로 통일하였다. 명령형 메모("~하지 말 것")는 규칙을 진술하는
서술문("~하지 않는다")으로 바꾸었다. (2) `main_ko.tex`는 2026-08-16 이후 개정되지 않았으므로
§3–§11 슬라이드는 논문과 일치한다. 2026-08-16 이후의 기록(B.9 도달 범위·계열·비용·승패
방전, §8ac–§8an; μs shot 동결 §8ao; 양자 가지 종결 §8ap; 프레이밍 §9)은 **논문에 추가할 절**로
슬라이드 한 장(f_b9, "§6.12 (추가 예정)")에 정리하였고, §9 개선 여지·§10 한계 슬라이드에 그
함의를 반영하였다.

**Deliberately excluded**: 날짜 타임라인, 연구 질문이 바뀐 경위(초해상 → gap-filling),
AutoML 탐색 경위, 재현성/체크포인트 함정, 운영용 다음-작업 우선순위. 논문에 한 줄도
들어가지 않는 과정 서사이므로 이 덱에서 전부 제외한다. 그 기록은 THESIS_RESULTS.md
§8과 PROJECT_KNOWLEDGE.md에만 남긴다. 음성 결과는 버리지 않되, 논문이 쓰는 형태 —
§9 "남은 개선 여지"의 레버와 §10 한계 — 로만 싣는다.

Palette, layout helpers and figures are reused from build_pptx.py so all decks look
like one family. 모든 수치는 `docs/paper/main_ko.tex`(= `paper_numbers.json` 동결
산출물)와 THESIS_RESULTS.md §8ac–§8ap의 표에서 그대로 옮겼다.

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
    A card that cannot fit even at 10 pt is reported at build time and must be cut.
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
PAPERFIG = os.path.join(HERE, "..", "paper", "figures")


# --- 1. Title -------------------------------------------------------------
def f_title():
    s = slide()
    box(s, 0, 0, EMU_W, EMU_H, fill=NAVY)
    box(s, 0, Inches(5.5), EMU_W, Inches(2.0), fill=RGBColor(0x0E, 0x26, 0x47))
    box(s, Inches(0.9), Inches(1.55), Inches(2.2), Pt(4), fill=TEAL)
    text(s, Inches(0.9), Inches(1.75), Inches(11.6), Inches(0.5),
         [[("논문 집필용 정리 · Paper-Writing Digest", 16,
            RGBColor(0x8F, 0xD6, 0xCB), True, False, None)]])
    text(s, Inches(0.88), Inches(2.30), Inches(11.7), Inches(2.0),
         [[("논문에 들어가는 것만:", 30, WHITE, True, False, None)],
          [("확정된 주장 · 인용할 수치 · 근거의 위치", 34, WHITE, True, False, None)]],
         line_spacing=1.12)
    text(s, Inches(0.9), Inches(4.12), Inches(11.5), Inches(1.3),
         [[("슬라이드 한 장은 ", 16, LGRAY, False, False, None),
           ("main_ko.tex의 한 절", 16, ORANGE, True, False, None),
           ("에 대응하며, 각 장의 노트에 그 절의 \\label과 인용 시 유의점이 적혀 있다.",
            16, LGRAY, False, False, None)],
          [("확정 프로토콜은 W=2 · held-free(학습·평가) · 파일당 500 · 두 모집단 공동 1차 · 백본 seq_v2이다.",
            16, RGBColor(0x8F, 0xD6, 0xCB), True, False, None)],
          [("연구가 걸어온 경로·기각된 곁가지·재현성 이슈는 논문에 쓰지 않으므로 이 덱에서 제외되었다. B.9(§8ac–§8an)는 논문에 추가할 절로 정리하였다.",
            15, LGRAY, False, False, None)]],
         line_spacing=1.2)
    text(s, Inches(0.9), Inches(5.9), Inches(11.5), Inches(1.1),
         [[("이승상  (Seungsang Lee)", 17, WHITE, True, False, None)],
          [("서울대학교 · 원자핵공학  |  2026-08-27 기준 (main_ko.tex 2026-08-16판 + B.9 추가 예정 절)",
            13, MGRAY, False, False, None)],
          [("출처: docs/paper/main_ko.tex · paper_numbers.json · THESIS_RESULTS.md §8v–§8ap · §9",
            11, MGRAY, False, False, None)]],
         line_spacing=1.25)
    return note(s, """
용도: 논문을 쓰는 동안 옆에 두는 참조판이다. 수치는 전부 main_ko.tex(= 동결 산출물 paper_numbers.json)와
THESIS_RESULTS.md §8ac–§8ap의 표에서 그대로 옮겼으므로 여기서 인용하면 본문과 어긋나지 않는다.

폐기된 이전 판(W=4 시대) 수치는 인용하지 않는다: iter2→iter9 progression, seq +0.045, anchor+Δ가 마진의
31.5% 회수, MNAR 1/4, 캠페인 0/4, held 포함/제외 이중 보고, CPU p99 6.4 ms, 윈도 W=2 p99 18.9 ms(§8ac
오염 판정). 대응 그림(fig_progression, fig_seq_paired, fig_stuckfree_paired, fig_window_sweep_heldkept,
W=4 fig_transient_*)도 쓰지 않는다.

이 덱에서 의도적으로 제외한 것은 날짜 타임라인, 연구 질문이 바뀐 경위(초해상 → gap-filling), AutoML 탐색
경위, 체크포인트/재현성 함정, 운영용 다음-작업 우선순위이다. 논문에 한 줄도 들어가지 않는 과정 서사이며,
기록은 THESIS_RESULTS.md §8과 PROJECT_KNOWLEDGE.md에 있다.
""")


# --- 2. Paper map ---------------------------------------------------------
def f_map():
    s = slide()
    header(s, "Map", "논문 골격 — 절 ↔ 확정한 것 ↔ 이 덱의 슬라이드", accent=NAVY)

    col_w = [Inches(2.30), Inches(4.75), Inches(2.95), Inches(2.23)]
    table(s, Inches(0.55), Inches(1.42), col_w,
          ["논문 절", "그 절이 확정한 것", "근거 (label · 표 · 그림)", "이 덱"],
          [["§1–2 서론·관련 연구", "인과 전체격자 프레이밍 · 기여 8항 · 계보 위의 확장",
            "sec:intro · sec:related", "3"],
           ["§3.1–3.3 데이터·프레이밍", "641 shot · 10 ms · 247,207행 · 두 프레이밍",
            "sec:data · fig:missing", "4"],
           ["§3.4–3.5 품질 감사 2종", "유지값 54% 제거 · Tᵢ>3 keV로 두 공동 1차 모집단",
            "sec:stuck · sec:spikes", "5"],
           ["§4 모델", "seq_v2 357,570 · 윈도 대조군 201,258 · b3k8 21,498",
            "sec:model · fig:architecture", "6"],
           ["§5 평가 방법론", "PR1–PR4 + 두 모집단 + 인과 GP · TEST 동결",
            "sec:eval", "7"],
           ["§6.1–6.3 결과 ①", "RMSE 사다리 · headline 4/4+4/4 · 백본 관문 +0.081",
            "tab:ladder · headline · gate", "8–10"],
           ["§6.4–6.6 결과 ②", "간극 >15 ms · MNAR 재가중 · 캠페인 시간 분할",
            "sec:gap · mnar · campaign", "11–13"],
           ["§6.7–6.9 결과 ③", "Tᵢ↔V_rot 비대칭 · 윈도 스윕(W=2) · 사다리+폭 스윕",
            "sec:asym · window · ladder", "14–16"],
           ["§6.10–6.11 · §7", "peak 집중 · 컷 문턱 무관 · TEST 전 결정 규칙",
            "sec:peak · cutsens · selection", "17–19"],
           [("§6.12 (추가 예정) B.9", NAVY, True, None), "문맥 50 ms 포화 · 계열 동률 · 연산자 비용 · 승패 방전",
            ("THESIS §8ac–§8an · fig_context", NAVY, True, None), ("20", NAVY, True, None)],
           ["§8–§11", "배치(지연·conformal) · 레버 3종 · 한계 · 결론",
            "sec:deploy · headroom · limits", "21–23"]],
          row_h=Inches(0.38), size=11, head_size=11.5)

    band(s, 6.12,
         [[("이 덱을 쓰는 법 — ", 13, TEAL, True, False, None),
           ("논문의 한 절을 쓸 차례가 되면 대응 슬라이드를 열고 카드의 수치를 그대로 인용한다. "
            "수치를 고쳐야 하면 먼저 collect_paper_numbers.py를 돌려 paper_numbers.json을 갱신한 뒤 "
            "본문·그림·덱을 함께 재생성한다. 손으로 박아 넣은 수치가 §8h의 오류였다.",
            13, WHITE, False, False, None)]])
    return note(s, """
main_ko.tex label 목록(전수): sec:intro, sec:related, sec:data, sec:framing, sec:stuck, sec:spikes,
sec:model, sec:eval, sec:results, sec:headline, sec:gate, sec:gap, sec:mnar, sec:campaign, sec:asym,
sec:window, sec:ladder, sec:peak, sec:cutsens, sec:selection, sec:deploy, sec:headroom, sec:limits,
sec:conclusion. 표: tab:ladder, tab:headline, tab:gap, tab:mnar, tab:campaign, tab:stress, tab:ablation,
tab:window, tab:ladder2. 그림: fig:missing, fig:ladder, fig:forest, fig:campaign, fig:ablation,
fig:ladder_scaling, fig:peak.

§6.1(RMSE 사다리)에는 소절 label이 없으므로 §sec:results 첫 소절 + tab:ladder / fig:ladder로 인용한다.

§6.12는 아직 본문에 없다. 2026-08-16 이후의 B.9 기록을 담을 절이며, 20번 슬라이드가 그 초안 골격이다.
그림은 docs/paper/figures/fig_context_family_ladder.png(이미 생성됨, 본문 미참조)이다.

그림 파이프라인: collect_paper_numbers.py → docs/paper/paper_numbers.json → make_figures_en.py(논문) /
make_figures.py(덱)이다. B.9 수치는 data/.b9_*.json에서 읽으며 collector v3에 통합하는 것이 후속이다.
""")


# --- 3. The two claims + the two populations ------------------------------
def f_two_claims():
    s = slide()
    header(s, "논문의 중심 논리", "두 개의 주장, 두 개의 모집단은 서로 뭉개지 않는다",
           accent=RED)

    col_w = [Inches(3.95), Inches(4.20), Inches(4.08)]
    table(s, Inches(0.55), Inches(1.45), col_w,
          ["평가 (seq_v2, Tᵢ)", "PCHIP 대비 (오프라인 · 미래 사용)", "인과 기준선 대비"],
          [["무작위 분할 · 관측 지점 (§6.2)",
            ("4/4 / 4/4,  +0.17~+0.32", GREEN, True, None),
            ("인과 GP 4/4 / 4/4,  +0.08~+0.17", GREEN, True, None)],
           ["결측 지점으로 재가중 (§6.5)",
            ("2/4 / 4/4,  +0.14~+0.28", ORANGE, True, None),
            ("persistence 4/4 / 4/4,  +0.28~+0.44", GREEN, True, None)],
           ["시간 캠페인 분할 (§6.6)",
            ("4/4 / 4/4,  +0.17~+0.20", GREEN, True, None),
            ("인과 GP 4/4 / 4/4,  +0.11~+0.16", GREEN, True, None)]],
          row_h=Inches(0.54), size=12)

    fcard(s, 0.55, 3.60, 6.03, 2.25,
          "두 모집단 규칙 (§3.5 sec:spikes)",
          ["· 컷은 Tᵢ > 3 keV(1,197행, 0.53%)를 적재 시점에 결측 처리한",
           "  모집단이며 전 arm에 동일하다(지도·이력·정규화·앵커).",
           "· 포함은 컷 없음이다. 둘은 공동 1차이며 사전등록되었다.",
           "· 무조건부 주장은 두 모집단에서 모두 성립할 때만 하며,",
           "  한쪽만 성립하면 모집단을 명시해 보고한다.",
           "· 표기 규약: 이 덱의 모든 수치는 “컷 / 포함” 순서이다.",
           "· 문턱 2.5–4 keV는 무의미하며(§6.11) 두 모집단이 본질이다."],
          accent=ORANGE, body_size=11.5)

    fcard(s, 6.75, 3.60, 6.03, 2.25,
          "본문에 쓰는 문장 형태",
          ["· headline(관측 모집단): “사전등록된 보간들과 배치 가능한",
           "  모든 인과 방법을 이기고, 미래를 쓰며 표본별로 튜닝된",
           "  평활기(오프라인 GP)와는 대등하다.”",
           "· 배치 주장: “진짜 결측이면서 도메인 안인 시점에서 CES 단독의",
           "  어떤 인과 방법보다 유의하게 낫다.”",
           "· 두 주장은 상대가 다르다. 둘을 뭉개는 것이 이 결과가 과대",
           "  판매되는 주된 경로이며 §10이 이를 명시한다."],
          accent=RED, body_size=11.5)

    band(s, 6.02,
         [[("한 문장 — ", 13.5, TEAL, True, False, None),
           ("온라인 가상 센서는 미래를 읽는 보간이 아니라 persistence와 인과 평활기와 경쟁한다. "
            "그 비교로 보면 이 나우캐스터는 중요한 지점에서 작동한다.",
            13.5, WHITE, False, False, None)]], h=0.72)
    return note(s, """
main_ko.tex 표 \\label{tab:stress}(§sec:campaign 끝), \\label{sec:spikes}, \\label{sec:conclusion} 2문단.

인용 시 유의점
· tab:stress는 seq_v2 · Tᵢ에 대한 표이다. 윈도 대조군 판본(이전 초안이 보고하였던 것)에서는 오프라인 우위가
  두 스트레스 테스트를 모두 견디지 못하였다. 그 사실을 본문에서 그대로 밝힌다.
· "2/4 / 4/4"는 컷 2/4, 포함 4/4를 뜻한다. 표기 순서를 덱·본문·그림에서 뒤집지 않는다.
· 무조건부 Tᵢ 주장은 headline(4/4+4/4) · 간극 >15 ms · peak(8/8) · 캠페인(4/4+4/4)이다. 무조건부 V_rot
  주장은 간극 >15 ms(+0.418 / +0.432) 하나이다.
· B.9는 세 번째 문장을 더한다: 약 50 ms의 연속 인과 문맥이 인과 GP 대비 우위를 전형적으로 만들며(승률
  0.52 → 0.66), 계열은 skill이 아니라 비용을 정한다(20번 슬라이드).
""")


# --- 4. Data & problem setup (sec:data / sec:framing) ---------------------
def f_data_setup():
    s = slide()
    header(s, "§3.1–3.3  sec:data · sec:framing", "데이터와 문제 설정 — 641파일 실측 사양",
           accent=TEAL)

    fcard(s, 0.55, 1.45, 4.0, 2.55,
          "데이터셋 (§3.1)",
          ["· 641 KSTAR 방전, shot 30801–32751이다",
           "  (제공 측 선정: 하드웨어 일관성 ·",
           "  H-mode ELM 억제(RMP) 구간).",
           "· 공통 10 ms 격자, 총 247,207행이다",
           "  (파일당 중앙값 339행).",
           "· 행당 BES 9 · ECEI 4 · Mirnov 2",
           "  + time + [CES_TI, CES_VT]이다.",
           "· 세그먼트는 0.5 s 간극에서 분리된다",
           "  (안쪽 스텝의 99.4%가 10 ms)."],
          accent=TEAL, body_size=11.5)

    fcard(s, 4.72, 1.45, 4.0, 2.55,
          "세그먼트 구조 · 결측 (§3.1)",
          ["· 전형적 파일은 주 세그먼트 1개이다",
           "  (중앙값 301행 ≈ 3.0 s, 10–90분위",
           "  1.3–7.0 s; 2개인 파일 28).",
           "  고립 단일행은 전체 1,279개(대개 t=0)이다.",
           "· Tᵢ 결측 8.2% · V_rot 결측 23.9%이며",
           "  서로 독립이다.",
           "· V_rot held 41.1%로 독립 관측은 35.0%이고,",
           "  관측 V_rot의 54.0%가 held이다.",
           "· 격자의 65.0%에 독립 V_rot 정보가 없다."],
          accent=ORANGE, body_size=11.5)

    fcard(s, 8.89, 1.45, 3.89, 2.55,
          "두 프레이밍 (§3.2)",
          ["· 윈도(대조군), W=2: bes (W,9) ·",
           "  ecei (W,4) · mc (W,2) ·",
           "  time_features (W,4) ·",
           "  ces_history (W,4) + 타깃별 마스크이다.",
           "· 전체격자 시퀀스(주 모델): 세그먼트의",
           "  입력-완전 행 전부를 맥락으로 쓴다.",
           "  스텝당 22채널 = z-score 고속 15",
           "  + log(1+Δt) + 타깃별 3(이월값 ·",
           "  신선도 · 과거관측 flag)이다."],
          accent=BLUE, body_size=11.5)

    add_image_fit(s, os.path.join(FIG, "fig_missing.png"),
                  Inches(0.55), Inches(4.15), Inches(7.3), Inches(2.5))

    fcard(s, 8.05, 4.12, 4.73, 2.55,
          "설계 원칙 3종 (§3.3)",
          ["· 가짜 라벨 금지 — impute가 없다. 윈도에서는 입력이",
           "  완전하고 CES 중 최소 하나가 관측된 행만 유지하며,",
           "  시퀀스에서 라벨 없는 행은 맥락으로만 기여한다.",
           "· 타깃별 마스킹 손실 —",
           "  ℒ = Σₖ mₖ(ŷₖ-yₖ)² / Σₖ mₖ 이다.",
           "  두-타깃-필수 필터는 라벨 행의 ≈28%를 버렸을 것이다.",
           "· 누수 방지 3중 — 파일 단위 분할 · 학습 파일 전용",
           "  z-score(시퀀스는 + shot별 고속 표준화) ·",
           "  타깃 시점 완전 마스킹이다."],
          accent=NAVY, body_size=11)
    return note(s, """
main_ko.tex \\label{sec:data} · \\label{sec:framing}, 그림 \\label{fig:missing}.

인용 시 유의점
· 결측률은 '행 기준·타깃별 독립'이며 두 타깃이 함께 결측되는 비율이 아니다.
· 0.5 s 임계값은 이봉 delta 분포의 골이다. 약 247k개 delta 중 (0.1, 0.5) s 구간은 82개뿐이다.
· 옛 초고의 "캠페인 #24000–#33000 · ~100 ms 절편"은 현재 데이터와 맞지 않아 폐기되었다. 이 슬라이드의
  사양이 641파일 실측이다.
· 타깃 시점 마스킹은 값과 관측 플래그를 '모두' 0으로 만든다. 플래그만 남기면 누수이다.
· ces_history 4채널은 직전 정규화 Tᵢ, 직전 정규화 V_rot, Tᵢ 관측 플래그, V_rot 관측 플래그이다.
· 시퀀스 22채널은 고속 15(BES 9 + ECEI 4 + MC 2) + log(1+Δt_prev) + 타깃 2개 × 3채널이다.
· 현재 격자는 99.46% 균일(Δt = 0.01 s)이므로 "불규칙 샘플링을 다룬다"는 주장은 이 데이터로 뒷받침되지
  않는다(§9.5). 다중 속도 획득(B.6)이 그것을 실재하게 한다.
""")


# --- 5. Two data-quality audits (sec:stuck / sec:spikes) ------------------
def f_audits():
    s = slide()
    header(s, "§3.4–3.5  sec:stuck · sec:spikes",
           "데이터 품질 감사 2종 — 유지값, 그리고 두 모집단이 생긴 이유", accent=ORANGE)

    fcard(s, 0.55, 1.45, 6.03, 4.35,
          "감사 1: 유지(forward-fill)된 V_rot  §3.4",
          ["· 관측된 V_rot 값의 54%가 계측기 유지값이다. 같은 연속",
           "  블록 안에서 직전 관측과 비트 단위로 동일한 반복이며,",
           "  최대 1,214행 연속, 641개 중 499개 파일이 영향을 받는다.",
           "· V_rot의 고유 측정 주기가 행 주기보다 느려서 생기며",
           "  독립적인 측정이 아니다.",
           "· 위양성 통로는 없다. V_rot은 소수점 다섯 자리까지",
           "  기록되고 서로 다른 값의 최소 간격이 4×10⁻⁵이다.",
           "  Tᵢ는 영향이 없다(226,991행 중 1행).",
           "",
           "확정 프로토콜: 유지값은 어디서나 제거된다",
           "  ① 지도 타깃  ② 이력·이월 입력과 그 관측 플래그",
           "  ③ 정규화 통계  ④ 모든 기준선의 보간 앵커",
           "따라서 어떤 팔도 forward-fill로 점수를 받을 수 없으며,",
           "본 논문의 모든 수치는 진짜 측정만 사용한다."],
          accent=ORANGE, body_size=11.5)

    fcard(s, 6.75, 1.45, 6.03, 4.35,
          "감사 2: Tᵢ 피팅 실패와 두 모집단  §3.5",
          ["· 관측 Tᵢ의 p99 = 2,089 eV, p99.9 = 9,601 eV,",
           "  최댓값 14,984 eV이며 먼 꼬리는 실패한 스펙트럼 피팅이다.",
           "· >3 keV가 1,197행(0.53%) = 274 방전의 951개 run이다.",
           "  run의 85%가 단일 행, 5행 이상은 2%이며,",
           "  run 정점 중앙값은 관측 이웃 평균의 13×(IQR 6–26×)이다.",
           "· 두 대응 모두 방어 가능하고 각각 비판 가능하다.",
           "  제거는 “어려운 행을 없앴다”, 유지는 스파이크 앵커가",
           "  오프라인 기준선을 오염시킨다.",
           "따라서 두 개의 공동 1차 모집단(컷 / 포함)을 사전등록하였다.",
           "",
           "· 값 컷은 한쪽 방향 프록시이다. 하향 급락(양쪽 이웃의",
           "  ½ 이하) 4,965행은 건드리지 않고, ≥2× 상향 이상치의",
           "  19%만 제거한다.",
           "· V_rot 스파이크(>1,000 km/s 119행 / 16 방전, 101행은",
           "  한 방전의 한 블록)는 컷하지 않고 SSE 비중을 병기한다."],
          accent=RED, body_size=11.5)

    band(s, 5.95,
         [[("하나로 고르지 못하는 이유 — ", 13, TEAL, True, False, None),
           ("컷 없이는 모든 팔이 PCHIP 대비 더 좋아 보인다(백본 +0.268 대 +0.236). "
            "스파이크가 보간 앵커를 오염시키기 때문이며, 학습된 모델은 그것을 할인할 수 있지만 보간은 못 한다. "
            "§6.7이 그 성분을 정량화하므로 두 모집단을 항상 함께 보고한다.",
            13, WHITE, False, False, None)]], h=0.85)
    return note(s, """
main_ko.tex \\label{sec:stuck}(§3.4) · \\label{sec:spikes}(§3.5). THESIS_RESULTS.md §8w, §8y, §8ab.

인용 시 유의점
· 유지값 제거는 민감도 한 줄이 아니라 '프로토콜'이다. 유지값은 평가뿐 아니라 학습도 오염시킨다
  (forward-fill 계단은 "이력을 복사하는 것이 거의 최적"이라고 모델에 가르친다). 짝지은 재학습 근거는
  프로젝트 기록에 있고 논문 본문은 프로토콜만 진술한다.
· 두 모집단은 '공동 1차'이다. p100 단일 headline으로 합치지 않는다(2026-08-16 결정).
· 컷은 로드 시점에 적용되고 전 arm이 비트 단위로 같은 모집단 키를 쓰는지 검증한다.
· V_rot 프로토콜은 불변이다. 컷/점프 규칙도 재학습도 없고, anchored 비교마다 스파이크 행의 제곱오차 비중만
  병기한다(2026-08-16 결정).
""")


# --- 6. Model (sec:model) -------------------------------------------------
def f_model():
    s = slide()
    header(s, "§4  sec:model", "모델 — 병목은 용량이 아니라 정보이다", accent=BLUE)

    add_image_fit(s, os.path.join(FIG, "fig_architecture_seq.png"),
                  Inches(0.55), Inches(1.45), Inches(5.40), Inches(5.20))

    fcard(s, 6.10, 1.45, 6.68, 2.55,
          "주 모델: 전체격자 시퀀스 백본 seq_v2 (357,570)",
          ["22채널 격자 시퀀스 위의 서로 독립인 인과 LSTM 2개이다.",
           "· Tᵢ 분기(2층, 160)는 전체 상태를 읽는다: 고속 진단 · 두 타깃의",
           "  이월값 · 신선도 플래그 · Δt.",
           "· V_rot 분기(1층, 64)는 비-고속 7채널만 읽는다: Δt와 타깃별",
           "  이월값 · 신선도 · 관측 플래그.",
           "· LayerNorm + 작은 GELU 헤드이며, 손실은 세그먼트의 모든",
           "  라벨 행에 대한 타깃별 마스킹 MSE이다.",
           "· AdamW 10⁻³, batch 16 세그먼트, 조기 종료(patience 6,",
           "  상한 30; 확정 실행은 14–25 epoch에서 종료)이다."],
          accent=BLUE, body_size=11)

    fcard(s, 6.10, 4.10, 6.68, 2.55,
          "프레이밍의 두 함의 · 짝지은 대조군 · 사다리 칸",
          ["① 도달 범위는 세그먼트 전체이다(고정 윈도가 아니다).",
           "   §6.3에서 이 모델을 윈도 변형과 가르는 양이며, B.9는 그중",
           "   약 50 ms가 실제로 쓰임을 측정하였다(§6.12 추가 예정).",
           "② 라우팅이 인코더 수준에서 성립한다. 고속 15채널을 전부",
           "   교란해도 V_rot 출력이 비트 단위로 동일하다(§6.7).",
           "· 윈도 대조군(201,258): 시간 인지 1-D CNN(W=2) + 양방향",
           "  GRU(64) + 타깃별 관측 마스킹 어텐션(파라미터 비용 0)이다.",
           "· b3k8(21,498): 헤드를 정확 분해 ŷ = 이월값 + Σ wₖzₖ + b",
           "  로 교체한다(readout 0 초기화 → persistence에서 출발)."],
          accent=TEAL, body_size=11)
    return note(s, """
main_ko.tex \\label{sec:model}(§4). 그림은 fig_architecture_seq.png(주 모델 seq_v2 도식)이다.
윈도 대조군 도식은 fig_architecture.png이며 필요하면 §6.3 슬라이드에서 함께 쓴다.

인용 시 유의점
· 주 모델은 seq_v2이다. 옛 주 모델(윈도 GRU + 관측마스킹 attention, 201,258, iter009)은 'W=2 윈도
  대조군'이며 그렇게만 부른다.
· b3k8의 z ∈ [-1,1]^K는 소형 인과 GRU(64/32)의 tanh 잠재이고 K=8(Tᵢ) / 4(V_rot)이며, readout이 0
  초기화라 학습이 정확히 persistence에서 출발한다. 파라미터는 백본의 6%이다.
· 탐색이 가르쳐 준 것(본문에 쓰는 유일한 탐색 서사): 윈도 계열 두 라운드의 통제 반복에서 살아남은
  메커니즘은 어텐션 풀링 하나뿐이고 용량 확장·스킵·추가 추출기는 한 번도 돕지 않았다. 확정 프로토콜에서
  같은 교훈이 반복된다(어텐션 후보 비유의 §7, 폭 스윕 평평 §6.9, 계열 동률 §6.12). 그래서 모델 절은 짧고
  평가 절은 길다.
· B.9 이후 모델 절에 추가할 문장: 아키텍처는 같은 문맥에서 skill로 구분되지 않으므로 비용(디스패치 연산자
  수)으로 선택된다(§9.3). 순환은 도달 범위에 O(1)이며 이것이 백본이 LSTM인 이유이다.
""")


# --- 7. Evaluation protocol (sec:eval) ------------------------------------
def f_eval():
    s = slide()
    header(s, "§5  sec:eval", "평가 방법론 — 모델에 불리하게 세운 기준", accent=NAVY)

    fcard(s, 0.55, 1.45, 6.03, 2.35,
          "지표와 채점 모집단",
          ["· skill = 1 - MSE_model / MSE_PCHIP(Murphy)이며,",
           "  물리 CES 단위로 역정규화한 뒤 타깃별로 계산한다.",
           "· 모든 팔이 동일한 (파일, 행) 집합에서 동일한 타깃별",
           "  마스크로 채점되고, 짝지은 비교 이전에 모집단 키가",
           "  비트 단위로 동일함을 검증한다.",
           "· 보간은 타깃 자신의 값을 제외한다(누수 없음).",
           "· 세그먼트 경계를 넘지 않으며, 경계 밖 이웃이 필요하면",
           "  보간이 persistence 값을 예측한다(PR2, 모집단 불변)."],
          accent=NAVY, body_size=11.5)

    fcard(s, 6.75, 1.45, 6.03, 2.35,
          "기준선 사다리 — 인과 GP가 새로 들어왔다",
          ["· 인과: persistence · 국소 과거 전용 AR · 인과 GP",
           "  (같은 GP를 과거 이웃 16개로 제한, NaN 조건 동일 →",
           "  채점 모집단이 움직이지 않는다)이다.",
           "· 오프라인(미래 사용): 선형 · PCHIP · GP",
           "  (Matérn-3/2 + 백색잡음, 최근접 16+16 국소 적합,",
           "  표본별 주변우도 격자 선택)이다.",
           "· 인과 GP가 배치 가능한 가장 강한 기준선이다",
           "  (시드 42 Tᵢ RMSE 164.3 대 persistence 197.2, 컷)."],
          accent=TEAL, body_size=11.5)

    fcard(s, 0.55, 3.90, 6.03, 2.75,
          "사전등록 PR1–PR4 + 확정 프로토콜이 더한 것",
          ["PR1  headline 상대는 PCHIP이며 사다리 전체도 함께 보고한다.",
           "PR2  보간은 모델이 채점되는 모든 곳에서 예측하며, 미래 이웃이",
           "        없으면 persistence로 후퇴하고 후퇴율을 보고한다. Tᵢ",
           "        0.3–0.4%, V_rot 40–44%이다(“vs PCHIP”의 2/5가 사실상",
           "        “vs persistence”이다).",
           "PR3  test 최소 규모는 ≥15 shot, ≥3,000 관측 Tᵢ이다(전부 충족).",
           "PR4  유의는 shot 군집 bootstrap 95% CI가 0을 제외함이다.",
           "＋ held-free · W=2 · 파일당 500 · 두 모집단 · 문턱 민감도 ·",
           "     모든 결정 규칙을 TEST 채점 이전에 문서로 확정한다."],
          accent=GREEN, body_size=11)

    fcard(s, 6.75, 3.90, 6.03, 2.75,
          "3-way 분할, TEST 동결, shot 군집 paired bootstrap",
          ["· 시드 42 test(컷): 관측 Tᵢ n = 32,589 / 96 shot,",
           "  진짜 V_rot n = 10,463 / 60 shot(포함 32,721 / 10,461)이다.",
           "· 4개 분할(42·1·7·123): Tᵢ 32.6–35.9k행 / 96 shot,",
           "  V_rot 10.5–14.5k행 / 60–66 shot이다.",
           "· 짝지은 SE 차이를 shot 단위로 집계하고 shot 전체를 복원",
           "  추출한다(10,000회, 고정 시드). 모델 대 모델 비교도 동일하다.",
           "· 대가: 유효 표본은 shot 수이며 전체 검정력의 상한이다.",
           "· MNAR 예고: 결측은 무작위가 아니므로(저신호·ELM·천이에서",
           "  탈락) 관측 지점 skill은 낙관적 추정이다. §6.5에서 실제",
           "  결측 지점으로 재가중해 얼마가 살아남는지 보고한다."],
          accent=BLUE, body_size=11)
    return note(s, """
main_ko.tex \\label{sec:eval}(§5).

인용 시 유의점
· "test 셋은 어떤 아키텍처 탐색 이전에 예약되었고 모델 선택 과정에서 절대 읽지 않았다"가 headline에
  winner's curse가 없다는 근거이며 §7이 그 이행을 문서화한다.
· V_rot의 PR2 폴백률 40–44%는 반드시 함께 인용한다. V_rot "vs PCHIP" 수치의 2/5는 사실상 "vs persistence"이다.
· 검정력 한계(shot ≈96 / 60–66)를 방법론 절에서 미리 인정하고 §10 첫 항목에서 다시 받는 구조를 유지한다.
· 모든 수치는 단일 수집 스크립트가 동결된 실행 디렉터리에서 읽으므로 본문·표·그림이 어긋나지 않는다.
· B.9가 더한 방법: 통합 재채점(4 분할의 행별 제곱오차를 합쳐 물리적 방전을 군집으로 bootstrap, 301 방전)과
  문맥 로그에 대한 skill 기울기 검정(각 재표본 안에서 재적합)이다(§8am). 단일 체크포인트의 배치 주장은
  여전히 시간 분할이 담당한다.
""")


# --- 8. Result 6.1: RMSE ladder ------------------------------------------
def f_res_ladder():
    s = slide()
    header(s, "§6.1  tab:ladder · fig:ladder",
           "나우캐스터는 모든 인과 기준선을 압도한다 — 가장 강건한 결과이다", accent=GREEN)

    col_w = [Inches(2.00), Inches(2.95), Inches(1.40), Inches(1.45)]
    table(s, Inches(0.55), Inches(1.45), col_w,
          ["팔", "정보 접근", "Tᵢ RMSE (eV)", "V_rot (km/s)"],
          [[("seq_v2 (나우캐스터)", NAVY, True, None), "고속 진단 + 과거 CES, 세그먼트 전체",
            ("157.8", GREEN, True, None), ("23.6", GREEN, True, None)],
           ["윈도 대조군 (W=2)", "고속 진단 + 과거 CES 2행", "169.2", "26.1"],
           ["인과 GP", "과거 CES 이웃 16개", "164.3", "28.8"],
           ["Persistence", "마지막 관측 CES", "197.2", "33.4"],
           ["AR (국소, 인과)", "과거 CES만", "472.2", "51.0"],
           [("GP (오프라인)", GRAY, True, None), "과거 + 미래 CES", "153.8", "24.7"],
           [("선형 보간", GRAY, True, None), "과거 + 미래 CES", "169.8", "29.0"],
           [("PCHIP 보간", GRAY, True, None), "과거 + 미래 CES", "173.6", "30.2"]],
          row_h=Inches(0.38), size=11, head_size=11.5)

    add_image_fit(s, os.path.join(FIG, "fig_rmse_ladder.png"),
                  Inches(8.50), Inches(1.45), Inches(4.28), Inches(2.85))

    fcard(s, 8.50, 4.40, 4.28, 2.25,
          "포함 모집단 — 순서는 불변이다",
          ["· seq_v2 363.0 / 23.7",
           "· PCHIP 412.4 / 30.2",
           "· 인과 GP 394.6 / 28.8",
           "· persistence 478.0 / 33.4",
           "피팅 실패 스파이크가 모든 Tᵢ RMSE를",
           "두 배 이상으로 키우지만 사다리의",
           "순서는 바꾸지 않는다."],
          accent=ORANGE, body_size=11)

    fcard(s, 0.55, 5.00, 7.80, 1.65,
          "본문에 쓰는 논지",
          ["· seq_v2는 미래를 읽지 않는 모든 방법 중 두 타깃 최저 RMSE이고, 인과 GP를 4%(Tᵢ)·18%(V_rot) 이긴다.",
           "· 유일한 동률 팔은 잡음 앵커를 통과하는 대신 평균해 내는 오프라인 GP이다(153.8 대 157.8).",
           "· 미래 CES가 정의상 없는 온라인 환경에서 시퀀스 나우캐스터는 명백한 승자이며 가장 강건한 결과이다.",
           "· 표와 그림은 같은 동결 산출물에서 생성된다. 두 스트레스 테스트를 통과하는 주장의 뿌리이다(§6.5·§6.6)."],
          accent=GREEN, body_size=11)
    return note(s, """
main_ko.tex §sec:results 첫 소절(소절 label 없음), 표 \\label{tab:ladder}, 그림 \\label{fig:ladder}.

인용 시 유의점
· 표는 시드 42 held-out test · 컷 모집단 · 진짜 측정만(n = 32,589 Tᵢ / 10,463 V_rot)이다.
· 어느 팔이 미래를 읽는지를 표 안에 명시한다(정보 접근 열). 아래 세 행이 미래를 읽는다.
· AR의 472.2는 오타가 아니다. 국소 선형 외삽은 희소·불규칙 격자에서 발산한다. 사다리에 남기는 이유는
  '과거만 쓰는 단순 외삽'이 왜 대안이 못 되는지를 보이기 위함이다.
· 표와 그림은 같은 동결 산출물에서 생성된다.
""")


# --- 9. Result 6.2: headline (sec:headline) ------------------------------
def f_res_headline():
    s = slide()
    header(s, "§6.2  sec:headline",
           "Headline — Tᵢ는 두 모집단 모두 4/4로 미래를 읽는 보간을 이긴다",
           accent=GREEN)

    fcard(s, 0.55, 1.45, 7.40, 2.95,
          "인용할 수치 — Tᵢ, held-out TEST (컷 / 포함)",
          ["· 42     +0.174 [+0.097, +0.236]   /   +0.225 [+0.109, +0.293]",
           "· 1       +0.248 [+0.188, +0.295]   /   +0.238 [+0.153, +0.302]",
           "· 7       +0.257 [+0.199, +0.302]   /   +0.292 [+0.232, +0.344]",
           "· 123   +0.264 [+0.188, +0.320]   /   +0.316 [+0.186, +0.392]",
           "컷 4/4 · 포함 4/4 PASS이다. 8개 셀 전부가 인과 GP와",
           "persistence(+0.36~+0.46)도 이긴다.",
           "vs 인과 GP(컷 / 포함): +0.078/+0.154 · +0.133/+0.169 ·",
           "+0.138/+0.123 · +0.105/+0.149이다.",
           "두 모집단 모두에서 성립하므로 논문의 무조건부 주장이다."],
          accent=GREEN, body_size=11)

    add_image_fit(s, os.path.join(FIG, "fig_forest.png"),
                  Inches(0.55), Inches(4.55), Inches(7.40), Inches(2.10))

    fcard(s, 8.15, 1.45, 4.63, 2.35,
          "V_rot — 승리가 아니라 동률이다",
          ["컷    +0.390* / +0.183 / +0.135 / +0.305",
           "        → PR4 1/4",
           "포함  +0.384* / +0.195* / +0.132 / +0.304",
           "        → PR4 2/4",
           "vs persistence 양쪽 3/4(+0.30~+0.50),",
           "vs 인과 GP 2/4이다. 4개 중 1–2개 유의는",
           "잡음이 만들 수 있는 수준이므로 회전",
           "채널의 승리는 주장하지 않는다."],
          accent=ORANGE, body_size=11)

    fcard(s, 8.15, 3.95, 4.63, 2.70,
          "포함이 더 높은 이유 · 오프라인 상한",
          ["· 컷이 없으면 모든 팔이 PCHIP 대비 더",
           "  좋은 점수를 받는다(백본 평균 +0.268 대",
           "  +0.236, 대조군 +0.238 대 +0.173).",
           "  스파이크가 보간 앵커를 오염시키며,",
           "  모델은 할인하고 보간은 못 한다(§6.7).",
           "· 오프라인 GP와는 동률(-0.05~+0.11,",
           "  8개 중 1개 유의)이며 오프라인 주장의 상한이다.",
           "headline은 “사전등록된 보간들을 이긴다”로",
           "읽는다."],
          accent=NAVY, body_size=11)
    return note(s, """
main_ko.tex \\label{sec:headline}(§6.2), 표 \\label{tab:headline}, 그림 \\label{fig:forest}.

인용 시 유의점
· 시드 1/7/123은 어떤 선택 단계에도 쓰이지 않은 진짜 복제이다.
· 표기 순서는 언제나 "컷 / 포함"이며 한쪽만 인용하지 않는다.
· V_rot은 점추정이 8/8 양수인데도 동률로 보고한다. 이 보고 기준 자체를 본문에 쓴다. PASS 셀은 컷 시드 42,
  포함 시드 42와 1(1은 [+0.000, +0.286]로 경계에 있다)이다.
· "미래를 쓰는 보간을 이긴다"는 §10에서 "사전등록된 보간들을 이긴다"로 한정된다(GP 동률).
· V_rot의 PR2 폴백률 40–44%(§5)를 함께 기억한다. B.9(§8al §4·§8an)는 V_rot의 동률이 검정력 부족이 아니라
  소수 방전에 집중된 우위임을 보였으며, 그 문장은 §6.12·§10에 들어간다.
""")


# --- 10. Result 6.3: backbone gate (sec:gate) ----------------------------
def f_res_gate():
    s = slide()
    header(s, "§6.3  sec:gate",
           "백본 관문 — 전체격자 프레이밍이 산 것은 '도달 범위'이다", accent=BLUE)

    add_image_fit(s, os.path.join(FIG, "fig_gate_b1.png"),
                  Inches(0.55), Inches(1.45), Inches(6.40), Inches(3.55))

    fcard(s, 7.15, 1.45, 5.63, 3.55,
          "사전등록 4조건 관문 (컷 모집단, 한 번만 채점)",
          ["분할 시드 × 초기화 시드의 4×4 격자를 돌리고 각 실행을",
           "그 분할의 W=2 윈도 대조군과 행 단위로 짝지었다.",
           "· 짝지은 Tᵢ skill은 16/16 실행 양수 · 13/16 개별 유의이다.",
           "· 분할별 초기화 평균은 +0.129 / +0.059 / +0.078 / +0.058이다",
           "  (초기화 산포가 분할 산포보다 훨씬 작다).",
           "· 통합 평균 +0.081, 실행 군집 95% CI [+0.067, +0.096]이다.",
           "· 예산 균등화(고정 10 epoch, 최종 가중치, val 선택 없음)",
           "  에서도 4/4 부호가 유지된다: +0.063 / +0.033 / +0.045 / +0.030.",
           "  이 이득은 학습 예산이 아니라 아키텍처에서 온다.",
           "· V_rot 유의 열세 0/16(유의 우세 8/16)이다.",
           "네 조건이 모두 성립하였다. 관문은 채점 이전에 고정되었다(§7)."],
          accent=BLUE, body_size=11)

    fcard(s, 0.55, 5.15, 12.23, 1.50,
          "확증 4 분할, 그리고 프레이밍이 '무엇을' 사는가",
          ["· 확증 4 분할의 같은 비교(seq_v2 − 대조군): 컷 +0.130 / +0.058 / +0.062 / +0.044, 포함 +0.053 / +0.024 / +0.047 / +0.029으로 8/8 양수, 각 2/4 유의이다.",
           "· 윈도 대조군은 인과 GP와 동률(컷 1/4)인데 시퀀스 백본은 두 모집단 모두 4/4이다. 세그먼트 과거 전체로의 도달 범위가 최강 배치 기준선을 넘어서게 한다.",
           "· B.9(§8af)는 이 +0.081을 도달 범위 −0.065와 구조 ≈ −0.016으로 분해하였다. 비용은 음수이며 학습비가 윈도 계열의 1/10이다."],
          accent=GREEN, body_size=11)
    return note(s, """
main_ko.tex \\label{sec:gate}(§6.3). 그림은 fig_gate_b1.png(B.1 관문 16 run)이다. THESIS_RESULTS.md §8x.

인용 시 유의점
· 관문은 '컷 모집단'에서 채점한다. 내부 모델 선택이 수행되는 모집단이기 때문이며 그 사실을 밝힌다.
· pooled +0.081의 CI는 shot 군집이 아니라 'run 군집'이다(16 run). 표기를 섞지 않는다.
· 예산 균등화 팔이 이 이득이 학습 예산 효과가 아니라 아키텍처에서 온다는 증거이다.
· 이 절은 §7 선택 프로토콜과 짝을 이룬다. 관문 4조건은 채점 이전에 문서로 고정되었다.
· B.9의 분해(§8af): 20 ms 도달 범위에서 학습한 seq_v2는 전체 블록 대비 −0.065(4/4)이며, 윈도 대조군 대비
  +0.081의 4/5가 도달 범위, 1/5가 구조이다. 이 문장은 §6.3 끝 또는 §6.12에 들어간다.
""")


# --- 11. Result 6.4: gap strata (sec:gap) --------------------------------
def f_res_gap():
    s = slide()
    header(s, "§6.4  sec:gap", "skill이 사는 곳 ① — 간극 영역 (4 분할 통합)", accent=BLUE)

    col_w = [Inches(2.10), Inches(1.95), Inches(1.15), Inches(3.50), Inches(3.53)]
    table(s, Inches(0.55), Inches(1.45), col_w,
          ["Δt", "n (컷 / 포함)", "shot", "컷: PCHIP 대비", "포함: PCHIP 대비"],
          [[("Tᵢ  ≤ 15 ms", NAVY, True, None), "134,546 / 135,317", "301",
            ("+0.239 [+0.197, +0.274]", GREEN, True, None),
            ("+0.299 [+0.244, +0.347]", GREEN, True, None)],
           [("Tᵢ  > 15 ms", NAVY, True, None), "3,422 / 3,334", "265 / 263",
            ("+0.268 [+0.187, +0.337]", GREEN, True, None),
            ("+0.206 [+0.108, +0.290]", GREEN, True, None)],
           [("Tᵢ  > 45 ms", NAVY, True, None), "460 / 429", "104 / 101",
            ("+0.267 [+0.092, +0.414]", GREEN, True, None),
            ("-0.004 [-0.304, +0.246]", ORANGE, True, None)],
           [("V_rot  ≤ 15 ms", NAVY, True, None), "51,689", "197",
            ("+0.233 [+0.020, +0.318]", GREEN, True, None),
            ("+0.233 [+0.020, +0.317]", GREEN, True, None)],
           [("V_rot  > 15 ms", NAVY, True, None), "466 / 456", "130",
            ("+0.418 [+0.104, +0.680]", GREEN, True, None),
            ("+0.432 [+0.128, +0.696]", GREEN, True, None)],
           [("V_rot  > 45 ms", NAVY, True, None), "14", "7", "미채점 (n < 50)", "미채점 (n < 50)"]],
          row_h=Inches(0.44), size=11, head_size=11.5)

    fcard(s, 0.55, 4.62, 3.95, 2.03,
          "① 인접 이력에 국한되지 않는다",
          ["채점 타깃의 ≥96%가 Δt ≤ 15 ms에",
           "있지만, 15 ms를 넘어서도 백본은",
           "두 모집단 모두에서 미래를 쓰는",
           "PCHIP을 이긴다(+0.268 / +0.206).",
           "같은 층에서 persistence 대비는",
           "+0.40과 +0.43이다."],
          accent=GREEN, body_size=11)

    fcard(s, 4.72, 4.62, 3.95, 2.03,
          "② >45 ms는 모집단 조건부이다",
          ["컷에서는 여전히 이기지만(+0.267)",
           "포함에서는 동률이다(-0.004,",
           "CI [-0.30, +0.25]).",
           "101 shot의 429행은 소수의 스파이크",
           "앵커 행이 한 계층을 지배할 수 있는",
           "규모이므로 동률로 보고한다."],
          accent=ORANGE, body_size=11)

    fcard(s, 8.89, 4.62, 3.89, 2.03,
          "③ 논문 유일의 무조건부 V_rot 양성",
          ["전역 V_rot 동률은 '인접 영역에서의'",
           "동률이다. Δt > 15 ms에서는 백본이",
           "두 모집단 모두 PCHIP을 이긴다",
           "(+0.418 / +0.432, 130 shot).",
           "≤ 15 ms에서는 persistence를 이긴다",
           "(+0.39). 가장 넓은 구간은 14행뿐이다."],
          accent=TEAL, body_size=11)
    return note(s, """
main_ko.tex \\label{sec:gap}(§6.4), 표 \\label{tab:gap}.

인용 시 유의점
· Δt가 커질수록 PCHIP의 과제는 쉬워지고(간극 양쪽의 실제 관측을 잇기만 하면 된다) 본 모델의 과제는
  어려워진다. 이 비대칭을 본문에 명시해야 이 층의 승리가 정직하게 읽힌다.
· 분할별로는 넓은 층에 수백 개 표본뿐이라 4개 test 분할을 통합하고 bootstrap을 물리적 방전 단위로
  군집하였다. 이 절을 가능하게 한 방법론적 변경이며 §8am의 통합 재채점과 같은 규칙이다.
· 표본 50개 미만 층은 채점하지 않는다(V_rot > 45 ms = 14행 / 7 shot).
""")


# --- 12. Result 6.5: MNAR reweighting (sec:mnar) -------------------------
def f_res_mnar():
    s = slide()
    header(s, "§6.5  sec:mnar", "스트레스 ① — 실제로 결측인 지점에서 얼마가 살아남는가",
           accent=ORANGE)

    box(s, Inches(0.55), Inches(1.42), Inches(12.23), Inches(1.02),
        fill=RGBColor(0xFF, 0xF3, 0xE6), round_=True)
    box(s, Inches(0.55), Inches(1.42), Inches(0.12), Inches(1.02), fill=ORANGE)
    text(s, Inches(0.85), Inches(1.48), Inches(11.6), Inches(0.96),
         [[("적용 범위 결과 — ", 13, ORANGE, True, False, None),
           ("도메인은 “진짜 관측이 두 행 이내”(W=2)이다. 진짜 결측 행 중 도메인 안은 ",
            13, DARK, False, False, None),
           ("Tᵢ 54–68% · V_rot 4–6%", 13, RED, True, False, None),
           ("뿐이다.", 13, DARK, False, False, None)],
          [("살아남은 층의 커버리지는 Tᵢ 0.99 · V_rot 0.73–0.76이다. 재가중된 V_rot은 결측 질량의 20분의 1에 대한 답이므로 결론을 내지 않는다.",
            11.5, GRAY, False, False, None)]],
         line_spacing=1.2)

    col_w = [Inches(1.05), Inches(0.80), Inches(1.55), Inches(3.60),
             Inches(1.55), Inches(3.68)]
    table(s, Inches(0.55), Inches(2.55), col_w,
          ["모집단", "분할", "vs PCHIP 무가중", "vs PCHIP 결측 정합 (95% CI)",
           "vs persist. 무가중", "vs persist. 결측 정합 (95% CI)"],
          [[("컷", NAVY, True, None), "42", "+0.174", "+0.140 [-0.071, +0.290]",
            "+0.360", ("+0.398 [+0.278, +0.518]", GREEN, True, None)],
           ["", "1", "+0.248", ("+0.164 [+0.062, +0.250]", GREEN, True, None),
            "+0.406", ("+0.366 [+0.299, +0.448]", GREEN, True, None)],
           ["", "7", "+0.257", "+0.203 [-0.024, +0.346]",
            "+0.403", ("+0.310 [+0.227, +0.398]", GREEN, True, None)],
           ["", "123", "+0.264", ("+0.283 [+0.069, +0.392]", GREEN, True, None),
            "+0.415", ("+0.381 [+0.246, +0.464]", GREEN, True, None)],
           [("포함", NAVY, True, None), "42", "+0.225",
            ("+0.140 [+0.033, +0.250]", GREEN, True, None),
            "+0.423", ("+0.443 [+0.141, +0.623]", GREEN, True, None)],
           ["", "1", "+0.238", ("+0.217 [+0.086, +0.319]", GREEN, True, None),
            "+0.436", ("+0.383 [+0.273, +0.536]", GREEN, True, None)],
           ["", "7", "+0.292", ("+0.167 [+0.067, +0.265]", GREEN, True, None),
            "+0.406", ("+0.283 [+0.172, +0.380]", GREEN, True, None)],
           ["", "123", "+0.316", ("+0.221 [+0.050, +0.320]", GREEN, True, None),
            "+0.459", ("+0.337 [+0.164, +0.455]", GREEN, True, None)]],
          row_h=Inches(0.33), head_h=Inches(0.38), size=10.5, head_size=10.5)

    band(s, 5.65,
         [[("방법 — ", 12.5, TEAL, True, False, None),
           ("층은 Δt(15/25/45 ms) × 입력 전용 활동 플래그(자기 행 제외 → 결측 행에서도 정의)이다. "
            "30 미만 층은 기각하고, 커버리지를 병기하며, 가중 격자에도 컷을 적용한다.", 12.5, WHITE, False, False, None)],
          [("결론 문장 — ", 12.5, TEAL, True, False, None),
           ("“진짜 결측·도메인 내 시점에서 나우캐스터는 두 모집단 모두 CES 단독의 어떤 인과 방법보다 "
            "유의하게 낫고(4/4+4/4, +0.28~+0.44), 오프라인 보간보다는 모집단 조건부(컷 2/4, 포함 4/4)로 낫다.”",
            12.5, WHITE, False, False, None)]],
         h=1.00)
    return note(s, """
main_ko.tex \\label{sec:mnar}(§6.5), 표 \\label{tab:mnar}.

인용 시 유의점
· MNAR 보정이 앗아가는 skill은 인과 비교에 대해 많아야 0.12이고 어떤 분할에서는 전혀 없다.
· PCHIP 대비 점추정은 +0.14~+0.28로 유지되지만 고정 가중치의 넓은 CI가 컷 2개 분할에서 0을 지난다.
  결측 지점은 더 큰 Δt에 놓이는데 그곳이 바로 양쪽 앵커가 가장 크게 돕고 재가중 bootstrap이 가장 얇은
  곳이며 §6.4와 기계적으로 일관된다.
· 한 층 안에서 결측 행과 관측 행이 교환 가능하다는 가정은 명시한다.
· 윈도 대조군도 같은 방식으로 거동한다(PCHIP 대비 2/4와 4/4, persistence 대비 4/4).
· V_rot은 도메인 도달이 4–6%뿐이므로 회전에 대한 배치 결론은 내지 않는다.
""")


# --- 13. Result 6.6: campaign shift (sec:campaign) -----------------------
def f_res_campaign():
    s = slide()
    header(s, "§6.6  sec:campaign",
           "스트레스 ② — 캠페인(시간) 분할: 대조군은 붕괴하고 백본은 견딘다", accent=RED)

    col_w = [Inches(0.95), Inches(2.55), Inches(3.55), Inches(0.70),
             Inches(1.55), Inches(2.93)]
    table(s, Inches(0.55), Inches(1.45), col_w,
          ["모집단", "팔", "Tᵢ vs PCHIP (초기화 42 / 1 / 7 / 123)", "PASS",
           "vs 인과 GP", "seq_v2 − 대조군"],
          [[("컷", NAVY, True, None), "윈도 대조군 (OFF)",
            "+0.027 / +0.091 / -0.001 / +0.061", ("2/4", RED, True, None), "0/4", "—"],
           ["", "대조군, shot별 표준화 (ON)",
            "+0.103 / +0.107 / +0.094 / +0.107", ("4/4", GREEN, True, None), "—", "—"],
           ["", ("seq_v2", NAVY, True, None),
            ("+0.187 / +0.174 / +0.181 / +0.177", GREEN, True, None),
            ("4/4", GREEN, True, None), "4/4 (+0.11~+0.12)",
            ("+0.164 / +0.091 / +0.182 / +0.124", GREEN, True, None)],
           [("포함", NAVY, True, None), "윈도 대조군 (OFF)",
            "+0.014 / +0.047 / +0.055 / +0.089", ("0/4", RED, True, None), "0/4", "—"],
           ["", ("seq_v2", NAVY, True, None),
            ("+0.173 / +0.202 / +0.198 / +0.184", GREEN, True, None),
            ("4/4", GREEN, True, None), "4/4 (+0.13~+0.16)",
            ("+0.161 / +0.163 / +0.151 / +0.104", GREEN, True, None)]],
          row_h=Inches(0.40), size=10.5, head_size=10.5)

    add_image_fit(s, os.path.join(FIG, "fig_campaign.png"),
                  Inches(0.55), Inches(3.95), Inches(4.40), Inches(2.70))

    fcard(s, 5.15, 3.95, 3.75, 2.70,
          "설계와 측정된 원인",
          ["shot 번호로 시간 분할하였다:",
           "· train 416 [30801, 31991]",
           "· val 128 [32002, 32310]",
           "· test 97 [32312, 32751]",
           "어떤 test shot도 어떤 train shot보다",
           "앞서지 않는다. 네 실행은 초기화 시드만",
           "다르다(분할 4개가 아니다).",
           "train→test 드리프트(중앙값)는 BES 1.22 σ ·",
           "ECEI 0.53 σ 대 타깃 0.115 σ로 5–11×이다."],
          accent=ORANGE, body_size=11)

    fcard(s, 9.05, 3.95, 3.73, 2.70,
          "명시할 것 세 가지",
          ["① 윈도의 오프라인 우위는 이동을 못",
           "   견딘다(2/4 · 0/4, 인과 GP 0/4).",
           "② 지목되었던 수리가 작동하였다. shot별",
           "   표준화로 컷 관문 2/4→4/4,",
           "   paired +0.078/+0.018/+0.095/+0.049,",
           "   V_rot 불변(-0.003~+0.008)이다.",
           "③ 시퀀스는 붕괴하지 않는다. 대조군",
           "   대비 8/8이며 V_rot은 persistence를",
           "   양쪽 4/4로 이긴다(대조군 0/4)."],
          accent=GREEN, body_size=11)
    return note(s, """
main_ko.tex \\label{sec:campaign}(§6.6), 표 \\label{tab:campaign} · \\label{tab:stress}, 그림 \\label{fig:campaign}.

인용 시 유의점
· 분할이 고정이므로 네 실행은 '초기화' 시드만 다르다. "4개 시드"나 "4개 분할"로 제시하지 않는다.
· 학습 파일 전용 정규화는 무작위 분할에서는 올바른 누수 방지 선택이지만 캠페인 이동에서 바로 그것이
  깨진다. shot별 표준화는 그 방전 자신의 데이터만 쓰므로 누수가 없다.
· seq_v2가 전이되는 이유는 둘이다: 정의상 shot별 표준화 + 도달 범위(§6.3).
· 대조군 대비 V_rot 마진도 8/8 유의이다(+0.06~+0.18).
· 남는 단서: 캠페인 4회는 하나의 시간 test 블록 위의 초기화들이며, 컷 모집단 seq_v2 실행 4개 중 2개는
  30-epoch 상한에서 멈추었다. 이 두 단서를 §10에서 다시 받는다.
· 이 절의 서사 구조 자체가 기여이다. 실패를 보고하고, 원인을 측정하고, 지목한 수리를 실행하였다. 새 열이
  원래부터 거기 있었던 것처럼 제시하지 않는다.
""")


# --- 14. Result 6.7: information asymmetry (sec:asym) --------------------
def f_res_asym():
    s = slide()
    header(s, "§6.7  sec:asym",
           "Tᵢ ↔ V_rot 정보 비대칭 — 본 연구의 과학적 발견", accent=ORANGE)

    col_w = [Inches(2.95), Inches(0.75), Inches(0.95), Inches(2.90),
             Inches(0.95), Inches(3.73)]
    table(s, Inches(0.55), Inches(1.45), col_w,
          ["입력 (평가 시 절제, 재학습 없음)", "타깃", "컷", "컷: paired (42/1/7/123)",
           "포함", "포함: paired (42/1/7/123)"],
          [[("전체 (이력 + 고속 + 시간)", NAVY, True, None), "Tᵢ", "+0.173", "—",
            "+0.238", "—"],
           ["이력 + 시간만 (고속 없음)", "Tᵢ", ("-0.125", RED, True, None),
            ("-0.25* / -0.38* / -0.42* / -0.43*", RED, True, None),
            "+0.201", "-0.03* / -0.04 / -0.03 / -0.09*"],
           ["고속 + 시간만 (이력 없음)", "Tᵢ", ("-2.11", RED, True, None),
            ("-4.6* / -1.8* / -2.3* / -1.9*", RED, True, None),
            ("-1.16", RED, True, None), ("-1.5* / -3.4* / -1.2* / -1.1*", RED, True, None)],
           [("전체", NAVY, True, None), "V_rot", "+0.213", "—", "+0.206", "—"],
           ["이력 + 시간만 (고속 없음)", "V_rot", "+0.213",
            ("+0.000 ×4 (비트 동일)", GREEN, True, None), "+0.206",
            ("+0.000 ×4 (비트 동일)", GREEN, True, None)],
           ["고속 + 시간만 (이력 없음)", "V_rot", ("-2.89", RED, True, None),
            ("-5.4* / -6.3* / -1.8* / -2.3*", RED, True, None),
            ("-3.51", RED, True, None), ("-6.9* / -7.8* / -1.9* / -2.2*", RED, True, None)]],
          row_h=Inches(0.42), size=10.5, head_size=10.5)

    add_image_fit(s, os.path.join(FIG, "fig_ablation.png"),
                  Inches(0.55), Inches(4.50), Inches(4.90), Inches(2.15))

    fcard(s, 5.65, 4.50, 3.55, 2.15,
          "읽는 법",
          ["· 이력은 두 타깃 모두에 필수이며,",
           "  제거하면 -1에서 -4까지 떨어진다.",
           "· Tᵢ, 컷: 마진은 고속 진단 정보이다.",
           "  고속을 0으로 하면 보간 아래로",
           "  간다(-0.10~-0.18). 물리 채널은",
           "  충돌성 전자–이온 결합(ECEI Tₑ, BES nₑ)이다.",
           "· V_rot: 정보는 전적으로 CES 이력이다."],
          accent=ORANGE, body_size=11)

    fcard(s, 9.40, 4.50, 3.38, 2.15,
          "포함 마진의 스파이크 성분",
          ["포함 모집단에서는 이력 전용 모델도",
           "PCHIP을 +0.15~+0.23으로 이기고,",
           "고속이 더하는 것은 0.03–0.09뿐이다",
           "(2/4 유의). 보간 앵커에 스파이크가",
           "끼어 있고 학습된 모델은 그것을",
           "할인하기 때문이며, 고속 기여를 분리하는",
           "쪽은 컷 모집단이다."],
          accent=NAVY, body_size=11)
    return note(s, """
main_ko.tex \\label{sec:asym}(§6.7), 표 \\label{tab:ablation}, 그림 \\label{fig:ablation}.

인용 시 유의점
· 절제는 '윈도 대조군'에 대해 '평가 시점'에 수행한다. 재학습 없음, 한 번에 한 모달리티 그룹만 0,
  persistence와 보간은 언제나 실제 이력에서 계산, TEST 두 모집단이다.
· seq_v2와 b3k8의 V_rot 분기도 같은 교란 시험을 통과한다(비트 동일). 라우팅이 인코더 수준이라는 §4의
  진술이 여기서 측정으로 확인된다.
· 이 비대칭은 절제 이전에 물리로부터 예측되었고 절제로 확인되었다. 순서를 본문에서 그대로 밝힌다.
· V_rot의 비승리는 모델 실패가 아니라 진단 정보량에 관한 발견이며 §9 레버 2·3의 출발점이다.
· B.9(§8an)가 세 번째 독립 증거를 더한다: 방전 단위 승패에서 Tₑ 수준은 V_rot 승률을 예측하지 못하며
  (ρ = -0.031), 승리 방전은 뜨거운 방전이 아니라 회전이 움직이는 방전이다. §6.12에 넣는다.
""")


# --- 15. Result 6.8: window sweep (sec:window) ---------------------------
def f_res_window():
    s = slide()
    header(s, "§6.8  sec:window", "이력은 얼마나 필요한가 — 관측 하나 (W=2 선택 근거)",
           accent=GREEN)

    col_w = [Inches(1.65), Inches(1.15), Inches(1.25), Inches(0.80),
             Inches(1.25), Inches(0.80)]
    table(s, Inches(0.55), Inches(1.45), col_w,
          ["이력 관측 수", "W", "Tᵢ skill", "PASS", "V_rot skill", "PASS"],
          [["0 (history-0)", "4", ("-0.026", RED, True, None), "0/4",
            ("-0.783", RED, True, None), "0/4"],
           ["1", ("2 (확정)", NAVY, True, None), ("+0.238", GREEN, True, None),
            ("4/4", GREEN, True, None), ("+0.206", GREEN, True, None), "0/4"],
           ["2", "3", ("+0.246", GREEN, True, None), ("4/4", GREEN, True, None),
            "+0.203", "1/4"],
           ["3", "4 (구 기본값)", "+0.221", "3/4", "+0.190", "1/4"],
           ["5", "6", "+0.190", "3/4", "+0.205", "1/4"],
           ["7", "8", "+0.216", ("4/4", GREEN, True, None), "+0.204", "2/4"]],
          row_h=Inches(0.42), size=11, head_size=11.5)

    add_image_fit(s, os.path.join(FIG, "fig_window_sweep.png"),
                  Inches(7.60), Inches(1.45), Inches(5.18), Inches(2.94))

    fcard(s, 0.55, 4.50, 6.03, 2.15,
          "선택 규칙과 그 답 — W = 2",
          ["· 24회 독립 실행(W ∈ {2,3,4,6,8} × 시드 4개) + history-0 ×4이다.",
           "  held-free · 파일당 500 · 컷 없음(동결 W=2 실행이 §6.2",
           "  포함 모집단의 윈도 대조군이다).",
           "· 이력을 제거하면 Tᵢ는 PCHIP 아래로(-0.026), V_rot은 -0.78이다.",
           "· 단 하나의 과거 관측이 두 타깃을 plateau로 올리고 곡선은",
           "  평평하다(Tᵢ 0.190–0.246, V_rot 0.190–0.206). 한 지점 안의",
           "  시드 산포 0.07–0.16이 곡선 전체보다 넓으므로 W = 2이다."],
          accent=GREEN, body_size=11)

    fcard(s, 6.75, 4.50, 6.03, 2.15,
          "두 자원 — 관측 수와 연속 문맥은 다른 양이다 (§9.2)",
          ["· 정확도를 근거로 더 긴 윈도의 비용을 치를 이유는 데이터 안에",
           "  없다. 구 기본값 W=4는 두 타깃 모두 W=2/W=3 아래이다.",
           "· 과거 CES 관측은 하나면 충분하다(이 절). 연속 빠른 진단",
           "  문맥은 약 50 ms가 필요하다(§6.12, B.9).",
           "· 윈도 계열은 구성상 두 번째 자원에 닿지 못한다(평균 풀링이",
           "  순서를 버리고, 시간 부분집합 증강이 비연속이며, 라벨 없는",
           "  행을 버린다). 이것이 §6.8이 평평하였던 이유를 소급 설명한다."],
          accent=BLUE, body_size=11)
    return note(s, """
main_ko.tex \\label{sec:window}(§6.8), 표 \\label{tab:window}. 그림은 fig_window_sweep.png(held-free 스윕)이다.
fig_window_sweep_heldkept.png는 폐기되었으며 쓰지 않는다.

인용 시 유의점
· paired가 아닌 독립 실행이므로 곡선 위 차이는 전부 시드 잡음 안이다. 주장은 순위가 아니라 '효과의 부재'이다.
· 파일당 상한 500이 시간 부분집합 증강(W=2의 240k 샘플 → W=8의 30.1M)이 소수의 긴 블록 방전에 지배되는
  것을 막는다.
· shot당 표본 상한 때문에 이 곡선의 절대 skill을 headline 계열과 직접 비교하지 않는다(곡선 내부 비교용).
· 이 스윕은 스파이크 컷보다 앞선다. 컷 없음이며, 동결된 W=2 실행이 §6.2 포함 모집단의 윈도 대조군이다.
· §9.2의 "두 자원" 프레이밍: §6.8(관측 수)과 §6.12(연속 문맥)는 충돌하지 않는다. 윈도 iter009 W=2의
  +0.041(1/4)과 seq_v2 2스텝 절단의 +0.055(2/4)가 같은 굶주림의 같은 결과이다. 전체격자 재프레이밍은
  모델링 선호가 아니라 두 번째 자원에 닿는 유일한 방법이었다.
""")


# --- 16. Result 6.9: complexity ladder + width sweep (sec:ladder) --------
def f_res_ladder2():
    s = slide()
    header(s, "§6.9  sec:ladder",
           "복잡도는 무엇을 사고, 크기는 돕는가 — 사다리와 폭 스윕", accent=BLUE)

    col_w = [Inches(2.95), Inches(1.35), Inches(1.30), Inches(1.30)]
    table(s, Inches(0.55), Inches(1.45), col_w,
          ["팔", "파라미터", "Tᵢ 컷", "Tᵢ 포함"],
          [["Persistence", "0", ("-0.264", RED, True, None), ("-0.288", RED, True, None)],
           ["앵커+Δ (명명된 항)", "1,258", ("-0.261", RED, True, None),
            ("-0.287", RED, True, None)],
           [("b3k8", NAVY, True, None), ("21,498", NAVY, True, None),
            ("+0.237", GREEN, True, None), ("+0.126", ORANGE, True, None)],
           ["윈도 대조군", "201,258", "+0.173", "+0.238"],
           [("seq_v2 백본", NAVY, True, None), ("357,570", NAVY, True, None),
            ("+0.236", GREEN, True, None), ("+0.268", GREEN, True, None)]],
          row_h=Inches(0.42), size=11, head_size=11.5)

    add_image_fit(s, os.path.join(FIG, "fig_ladder_scaling.png"),
                  Inches(7.30), Inches(1.45), Inches(5.48), Inches(2.95))

    fcard(s, 0.55, 4.10, 6.53, 2.55,
          "사전등록된 두 조건, 그 판정, 그리고 probe",
          ["· 조건 ① b3k8 − 앵커+Δ: 컷 +0.35~+0.42 4/4* / 포함",
           "  +0.29~+0.34 4/4*이며 V_rot 손실 없이 4/4 승으로 충족(양쪽)이다.",
           "· 조건 ② b3k8 − seq_v2: 컷 평균 +0.002(모든 CI가 0 포함) /",
           "  포함 평균 -0.194(4/4*)이므로 “백본 -0.05 이내”는 컷 조건부이다.",
           "· 컷에서는 백본의 Tᵢ skill 전부가 유계 수 8개 + persistence로",
           "  압축된다: 짝지은 -0.009 / -0.005 / +0.026 / -0.004,",
           "  PCHIP 대비 PR4 4/4, 인과 GP 4/4이다.",
           "· 선형 probe: Tᵢ 잠재는 직전 관측 Tᵢ(R² 0.47–0.75)와 ECEI Tₑ",
           "  대리(0.31–0.48)를 분산 부호화하며 활동 0.09–0.13은 거의 안 담는다."],
          accent=GREEN, body_size=11)

    fcard(s, 7.30, 4.50, 5.48, 2.15,
          "포함에서는 -0.16~-0.21 · 크기 축은 닫혔다",
          ["· 컷이 없으면 백본 대비 -0.16~-0.21(4/4 유의), 3/4 분할에서",
           "  윈도 대조군 아래이다. 유계 보정이 스파이크 이월값을 못 살리기",
           "  때문이며, persistence 오차 >2 keV 행은 포함 test의 0.6–1.3%",
           "  인데 b3k8 Tᵢ 제곱오차의 73–83%를 담는다(다른 팔 70–83%).",
           "· 폭 24→260(34k / 49k / 114k / 358k / 879k): Tᵢ +0.230 /",
           "  +0.236 / +0.235 / +0.236 / +0.230(컷), 358k 대비 ±0.008,",
           "  최대 폭 유의 우세 1/4, V_rot 불변(+0.250~+0.254)이다."],
          accent=ORANGE, body_size=11)
    return note(s, """
main_ko.tex \\label{sec:ladder}(§6.9), 표 \\label{tab:ladder2}, 그림 \\label{fig:ladder_scaling}.

인용 시 유의점
· 앵커+Δ(1,258)는 W=2에서 persistence로 붕괴한다. 기울기 항이 관측된 이력 행 두 개를 요구하는데 결코
  발화하지 않는다. 옛 초고의 "앵커+Δ가 마진의 31.5%를 회수" 문장은 폐기되었다.
· b3k8의 학습된 보정은 예측 분산의 25–39%를 설명한다.
· 사다리 칸 조건은 두 모집단 모두에서 성립하지만(앵커 대비 4/4) '백본 허용' 조건은 컷 조건부이며 본문도
  그렇게 진술한다.
· 남는 분산은 용량이 아니라 분할 분산이다(시드 42의 +0.14~+0.17 대 시드 123의 +0.23~+0.28).
· 결론 문장: 상한은 추정기가 아니라 정보이다. {100 Hz BES/ECEI/MC + CES 이력 + 시간}에 든 Tᵢ 정보는
  ~50k 파라미터의 인과 순환 상태로 소진된다. B.9(§8ai)는 이를 아래로 확장한다: 10k 파라미터 아래에서는
  합성곱이 크기를 맞춘 순환 arm보다 +0.027~+0.040 나으며, 계열은 상한이 아니라 상한에 이르는 데 필요한
  파라미터 수를 정한다.
""")


# --- 17. Result 6.10: peak stratification (sec:peak) ---------------------
def f_res_peak():
    s = slide()
    header(s, "§6.10  sec:peak", "skill이 사는 곳 ② — 우위는 고변동 국소 구간에 집중된다",
           accent=TEAL)

    add_image_fit(s, os.path.join(FIG, "fig_peak.png"),
                  Inches(0.55), Inches(1.45), Inches(6.30), Inches(5.20))

    fcard(s, 7.05, 1.45, 5.73, 1.85,
          "Tᵢ — 무조건부 진술",
          ["· peak: 컷 +0.45~+0.61 / 포함 +0.62~+0.72로 8/8 PASS이다",
           "  (미래 이웃을 가진 보간을 상대로).",
           "· bulk: 컷 +0.09~+0.20(4/4) / 포함 +0.06~+0.19(2/4)이다.",
           "“매끄러운 bulk에서는 보간이 거의 최적이고 모델의 가치는",
           "활동 구간에 있다”는 무조건부 진술이다."],
          accent=TEAL, body_size=11)

    fcard(s, 7.05, 3.40, 5.73, 1.85,
          "V_rot — 비대칭은 '지역적'이다",
          ["· peak: +0.54~+0.79(8/8 점추정 양수, 각 모집단 PASS 2/4;",
           "  persistence 대비 +0.75~+0.86에 8/8 PASS)이다.",
           "· bulk: ≈0(-0.07~+0.15, PASS 0/8)이다.",
           "전역은 동률이지만 매끄러운 과거+미래 보간이 가장 나쁜",
           "고활동 구간에서는 이력 기반 예측기도 가치를 더한다."],
          accent=ORANGE, body_size=11)

    fcard(s, 7.05, 5.30, 5.73, 1.35,
          "peak 선정이 순환 논리가 아닌 이유",
          ["층은 타깃 행을 제외하고 계산한 입력 측 활동 대리(이웃 괄호",
           "기울기 · 국소 CES 이웃 분산)로 나눈다. 규모는 분할당",
           "peak Tᵢ 4.1–4.6k행 · V_rot 2.4–2.9k행이며 채점은 TEST이다."],
          accent=GRAY, body_size=11)
    return note(s, """
main_ko.tex \\label{sec:peak}(§6.10), 그림 \\label{fig:peak}.

인용 시 유의점
· peak/bulk 분할은 TEST 위에서, seq_v2에 대해 수행한다(옛 판의 '검증 분할' 단서는 더 이상 없다).
· V_rot의 PASS 2/4는 각 모집단에서의 수이다. 점추정은 8/8 양수이다.
· Tᵢ의 8/8 PASS가 이 절을 무조건부 진술로 만든다. 두 모집단 모두에서 성립하기 때문이다.
· B.9(§8al §4·§8an)와의 관계: 행 단위 peak 비율은 방전 단위 승패를 예측하지 못하며(고-peak 절반 0.435 vs
  저-peak 절반 0.528), 승패를 예측하는 변수는 방전 내 타깃의 산포이다. 두 양은 다르므로 이 절의 결과는
  행 단위 PCHIP 대비 결과로 한정하여 인용한다.
""")


# --- 18. Result 6.11: cut-threshold sensitivity (sec:cutsens) ------------
def f_res_cutsens():
    s = slide()
    header(s, "§6.11  sec:cutsens", "컷 임계값 민감도 — 문턱은 무관하고, 두 모집단이 본질이다",
           accent=ORANGE)

    col_w = [Inches(2.30), Inches(2.55), Inches(2.55), Inches(2.35), Inches(2.48)]
    table(s, Inches(0.55), Inches(1.45), col_w,
          ["임계값 (재학습)", "Tᵢ skill (4 분할 평균)", "PCHIP 대비 PR4",
           "인과 GP 대비", "V_rot skill"],
          [["2.5 keV", "+0.230", ("4/4", GREEN, True, None),
            ("4/4", GREEN, True, None), "+0.252"],
           [("3 keV (확정 프로토콜)", NAVY, True, None),
            ("+0.236", GREEN, True, None), ("4/4", GREEN, True, None),
            ("4/4", GREEN, True, None), "+0.253"],
           ["4 keV", "+0.232", ("4/4", GREEN, True, None),
            ("4/4", GREEN, True, None), "+0.257"]],
          row_h=Inches(0.46), size=12, head_size=11.5)

    fcard(s, 0.55, 3.45, 3.95, 2.35,
          "① 수행한 것",
          ["스파이크 임계값을 2.5와 4 keV로",
           "두고 백본을 재학습하였다. 각각 4개",
           "분할이고 각 팔은 자기 모집단에서",
           "채점된다.",
           "모든 임계값에서 PCHIP 대비 PR4",
           "4/4, 인과 GP 대비 4/4이다. 물리적으로",
           "방어할 만한 범위 안에서 임계값은",
           "무의미하다."],
          accent=BLUE, body_size=11)

    fcard(s, 4.72, 3.45, 3.95, 2.35,
          "② 논문이 말하는 것",
          ["중요한 것은 꼬리 안의 '어디에서'",
           "자르느냐가 아니라 '두 모집단'이다.",
           "컷/포함을 언제나 함께 보고하고,",
           "무조건부 주장은 둘 다 성립할 때만",
           "조건 없이 진술한다.",
           "한쪽에서만 성립하는 결과는 모집단을",
           "명시해 보고한다(예: 간극 >45 ms,",
           "b3k8의 백본 허용치)."],
          accent=GREEN, body_size=11)

    fcard(s, 8.89, 3.45, 3.89, 2.35,
          "③ V_rot 스파이크는 컷하지 않는다",
          ["V_rot에도 피팅 실패 스파이크가",
           "있다. 16 shot에서 1,000 km/s를",
           "넘는 119행이며 그중 101행은 한 방전의",
           "한 블록이다.",
           "프로토콜은 V_rot을 컷하지 않고,",
           "persistence 기반 V_rot 비교마다",
           "그 행들이 담는 제곱오차 비중을",
           "함께 보고한다(§6.9와 같은 규칙)."],
          accent=ORANGE, body_size=11)

    band(s, 5.95,
         [[("결정 기록 — ", 12.5, TEAL, True, False, None),
           ("08-16 ① 두 모집단 공동 1차 유지(p100 단일 headline 없음) ② V_rot 프로토콜 불변(컷·점프 규칙 없음, 재학습 없음, anchored 비교엔 스파이크 행 SSE 비중 병기) · "
            "08-21 ③ B.6 μs shot 집합 동결(test 4 / pool 6 / companion 2) · 08-24 ④ 양자 가지 음성 종결.",
            12.5, WHITE, False, False, None)]], h=0.80)
    return note(s, """
main_ko.tex \\label{sec:cutsens}(§6.11).

인용 시 유의점
· 이 절은 §3.5(sec:spikes)의 "임계값은 2.5–4 keV 범위에서 무의미하다"는 문장의 근거이며 두 절을 상호
  참조로 묶는다.
· 각 임계값 팔은 '자기 모집단'에서 채점된다. 서로 다른 모집단의 skill을 직접 비교하는 것이 아니다.
· 값 기준 컷이 프록시라는 한계는 §10에 남고, 그것을 결말짓는 것은 §9 레버 1(CES 피팅 품질 메타데이터)이다.
""")


# --- 19. Model selection protocol (sec:selection) ------------------------
def f_selection():
    s = slide()
    header(s, "§7  sec:selection",
           "모델 선택 프로토콜 — 결정 규칙은 결정할 수치보다 먼저 적혔다",
           accent=NAVY)

    fcard(s, 0.55, 1.45, 6.03, 2.30,
          "윈도 계열 — 통제된 실험의 연속",
          ["· 각 실험은 데이터 계약을 보존하는 '하나의' 변경이었고,",
           "  깨끗한 비증강 검증 skill로만 채점하였으며, 지금까지의",
           "  최고 점수를 개선할 때만 유지하였다.",
           "· 증강된 검증 손실은 쓰지 않았다. 보간이 이미 강한",
           "  바로 그곳에서 평활화를 보상하기 때문이다.",
           "· 이후 이력 길이는 §6.8의 스윕으로 정해졌고(W=2),",
           "  유지값은 §3.4의 감사에 따라 학습에서 제거되었다."],
          accent=BLUE, body_size=11.5)

    fcard(s, 6.75, 1.45, 6.03, 2.30,
          "백본 관문 — 먼저 고정하고, 그 다음 충족",
          ["시퀀스 프레이밍은 네 조건이 먼저 고정된 뒤에야 채택되었다:",
           "  ① 4/4 분할에서 부호 유지",
           "  ② 통합 실행 군집 CI가 0을 제외",
           "  ③ 예산 균등화에서도 부호 유지",
           "  ④ V_rot 손실 없음",
           "네 조건은 §6.3에서 모두 성립하였다(16/16 양수, 13/16 유의,",
           "pooled +0.081 [+0.067, +0.096], 균등화 4/4, V_rot 0/16)."],
          accent=GREEN, body_size=11.5)

    fcard(s, 0.55, 3.87, 6.03, 2.30,
          "유일한 후속 후보는 승격되지 않았다",
          ["같은 규칙 아래 이후 탐색된 단 하나의 아키텍처 후보는 각",
           "타깃 자신의 과거 관측 스텝에 대한 관측 마스킹 인과 어텐션을",
           "0으로 초기화된 사영과 함께 seq_v2에 추가한 것이다.",
           "· 4/4 분할에서 양수: +0.009 / +0.013 / +0.033 / +0.020",
           "· 사전 확정 기준(≥3/4 유의)에 대해 1/4에서만 유의 → 미승격",
           "검증 이득은 탐색 분할에서 2/2 유의였다. 선택 분할 결과의",
           "통상적 낙관이며 승격 기준을 TEST에 두는 이유이다."],
          accent=ORANGE, body_size=11)

    fcard(s, 6.75, 3.87, 6.03, 2.30,
          "사다리 칸·폭 스윕·B.9도 같은 규율 아래",
          ["· 두 갈래 판정(사다리 칸이 앵커를 이길 것 · 백본 -0.05",
           "  이내)과 서술적 독법(천장 / 무릎)을 어떤 TEST 채점",
           "  이전에 적어 두었다.",
           "· 구성상 이 스윕 위에서 백본을 재선택하는 것은 허용되지",
           "  않았다. 스윕은 서술이지 선택이 아니다.",
           "· B.9는 PREREGISTRATION_B9.md(H1–H6, §3.2·§3.4·§4)로",
           "  사전등록되었고, H1 규칙의 명세 오류는 사후 수정 없이 기록되었다."],
          accent=TEAL, body_size=11)

    band(s, 6.28,
         [[("이 절이 논문에서 하는 일 — ", 12.5, TEAL, True, False, None),
           ("headline에 winner’s curse가 없다는 주장을 '절차'로 뒷받침한다. "
            "TEST는 결정마다 한 번만 채점되었고, 승격되지 않은 후보까지 그대로 보고한다는 사실이 그 증거이다.",
            12.5, WHITE, False, False, None)]], h=0.66)
    return note(s, """
main_ko.tex \\label{sec:selection}(§7).

인용 시 유의점
· "본 논문의 모든 모델 결정은 검증 데이터 위에서, 또는 해당 TEST 채점 이전에 문서로 확정된 결정 규칙
  아래에서 이루어졌고, TEST는 결정마다 한 번만 채점되었다"가 이 절의 첫 문장이다.
· 승격되지 않은 어텐션 후보를 '싣는 것' 자체가 기여의 일부이다. val 2/2 유의 → TEST 1/4 유의는 선택 분할
  낙관의 교과서적 사례이다. §8ak는 attention 계열이 70 ms에서 LSTM보다 -0.023 뒤짐을 더하였다.
· 옛 판의 "게이트를 val loss → clean skill로 바꾼 것이 최종 모델을 낳았다(fig_progression)"는 W=4 시대
  서사이며 폐기되었다. 진행(progression) 그림도 삭제되었다.
· B.9의 H1은 효과 크기 경계와 유의 계수를 disjunction으로 묶어 작은 강건 효과(-0.065, 4/4)가 '약한 의존'
  가설을 기각하게 하였다. 문서는 편집되지 않았고 결함은 §8af §3에 기록되었다. 이것도 절차의 증거로 인용한다.
""")


# --- 20. §6.12 (to be added): B.9 context / family / cost / wins ----------
def f_b9():
    s = slide()
    header(s, "§6.12 (추가 예정)  B.9",
           "문맥·구조·비용 — 논문에 추가할 절의 골격과 인용할 수치 (§8ac–§8an)", accent=TEAL)

    add_image_fit(s, os.path.join(PAPERFIG, "fig_context_family_ladder.png"),
                  Inches(0.55), Inches(1.42), Inches(5.0), Inches(3.5))

    fcard(s, 5.75, 1.42, 7.03, 2.35,
          "① 문맥 — 포화 약 50 ms, 문맥이 사는 것은 전형성이다 (§8af·§8al·§8am)",
          ["· seq_v2를 2·3·4·5·6·7·10·15·31·63 스텝에서 학습·채점하였다. 전체 대비 결손은",
           "  20 ms -0.066 → 50 ms -0.017 → 70 ms +0.002이며 §3.4 규칙은 50 ms를 반환한다.",
           "· 301 방전 통합: 20 ms +0.057 [+0.027, +0.085] → 630 ms +0.143 [+0.118, +0.168].",
           "  승률은 0.52 → 0.66(70 ms)에서 평평해지고, 문맥 10배당 skill +0.050 [+0.036, +0.064]이다.",
           "· §8ac의 절단 사다리(500 ms)는 cold start였다. warm-up 비중 87%(§8ae·§8af).",
           "· V_rot는 20 ms에서만 -0.013(2/4)이며, 네 계열 전부에서 문맥이 길수록 나빠진다."],
          accent=TEAL, body_size=10.5)

    fcard(s, 5.75, 3.87, 7.03, 1.85,
          "② 계열 — 같은 문맥에서 동률, 비용은 연산자 수 (§8ag·§8ak·§8ai·§8aj)",
          ["· 같은 도달 범위의 LSTM 대비 tcn3/7/15/63 -0.004/-0.004/+0.014/-0.016, xfmr7/15/63",
           "  -0.023*/+0.002/-0.019이며 '차이'는 xfmr7 하나이다. 최대 계열 효과 0.023 < 문맥 효과 0.060.",
           "· 10k 아래: 합성곱이 크기 맞춘 순환보다 +0.027~+0.040. tcn2k(1,808)가 인과 GP 4/4.",
           "· 비용 t ≈ N_ops × 2–3 µs: 순환 111(O(1)) · 합성곱 +48/층(O(log R)) · attention 473."],
          accent=NAVY, body_size=10.5)

    fcard(s, 0.55, 5.02, 5.0, 1.65,
          "③ 승패 방전 (§8al §4·§8an)",
          ["방전 단위 승률은 Tᵢ 0.695 / V_rot 0.481이다.",
           "11개 공변량 중 타깃 산포만 승패를 예측한다(ρ +0.401 / +0.281).",
           "3분위 승률은 Tᵢ 42/83/85%, V_rot 34/48/55%이다.",
           "V_rot 잔차는 구동 변수(NBI 토크) 부재의 세 번째 증거이다."],
          accent=ORANGE, body_size=10.5)

    band(s, 5.82,
         [[("본문에 쓰는 lead claim (§9) — ", 12, TEAL, True, False, None),
           ("약 50 ms의 연속 인과 문맥이 최강 배치 기준선에 대한 우위를 평균이 아니라 전형적으로 만들며, 그 문맥을 어떤 계열로 넘느냐는 skill이 아니라 비용을 정한다. "
            "\"N ms가 있어야 인과 GP를 이긴다\"는 쓰지 않는다(20 ms에서도 이긴다).",
            12, WHITE, False, False, None)]], x=5.75, w=7.03, h=0.85, pad=0.22)
    return note(s, """
이 절은 main_ko.tex에 아직 없다. THESIS_RESULTS.md §8ac–§8an과 §9 프레이밍을 담을 절이며, 그림은
docs/paper/figures/fig_context_family_ladder.png(이미 생성)이다. 수치 원천은 data/.b9_reach_ladder.json ·
.b9_family.json · .b9_minimal_family.json · .b9_pooled_ladder.json · .b9_op_counts.json · .b9_latency*.json ·
shot_covariates.py 산출물이며, collector v3에 통합한 뒤 인용한다.

절의 골격(§9 순서)
1. 질문: 다중 센서 관행(합성곱) 대비 순환의 정당성, 윈도 계열이 충분하다면 무한 순환 상태의 필요성.
   두 질문은 "학습된 모델이 실제로 몇 스텝의 연속 과거를 쓰는가"로 환원된다.
2. 절단 사다리와 그 교정: §8ac(500 ms) → §8ae(warm-up ≥ 84%) → §8af(트레인-앳-리치, 70 ms) → §8al
   (밀집, 50 ms). 세 교정을 순서대로 싣는다. 측정은 옳았고 읽기가 틀렸다는 문장을 그대로 쓴다.
3. 통합 재채점(§8am): 4/4 계수의 요동을 이유로 문턱 추정기에서 퇴출하고, 301 방전 통합 CI + 추세 검정으로
   대체한다. 승률·-top10 열을 CI 옆에 병기하는 규칙을 명시한다(V_rot는 좁은 CI가 곧 전형적 효과가 아님의
   증거이다: 197 방전 +0.121, 승률 0.46, -top10 -0.022).
4. 계열(§8ag·§8ak·§8ai): 같은 문맥 paired, 두 한정(10k 아래 합성곱 우세, 70 ms attention 열세), SSM 부록.
5. 비용(§8ah·§8aj): 연산자 수 법칙, 계열별 O(1)/O(log R)/O(1)·큰 상수, 10 ms 비구속, 1 ms 보류(21.84배).
6. 승패 방전(§8an): 타깃 산포 단일 변수, 3분위 표, V_rot 잔차 = 구동 변수 부재의 세 번째 증거.
7. 쓰지 않는 문장(§9.6): "모든 오프라인 방법을 이긴다", "윈도 프레이밍은 정보로 기각된다", "1 ms 배치 판정",
   "문맥은 길수록 좋다"(630 ms 유계가 무한보다 낫다).

§6.8과의 관계는 §9.2 "두 자원"으로 쓴다. 관측 수(하나면 충분)와 연속 문맥(약 50 ms)은 다른 양이다.
""")


# --- 21. Deployability (sec:deploy) --------------------------------------
def f_deploy():
    s = slide()
    header(s, "§8  sec:deploy", "배치 가능한가 — 지연과 불확실성을 둘 다 측정하였다", accent=BLUE)

    fcard(s, 0.55, 1.45, 6.03, 2.45,
          "지연: 상태 유지 1-스텝은 여유를 두고 CPU에 들어간다",
          ["· 온라인에서는 은닉 상태가 격자를 따라 이월되고 새 행마다",
           "  배치 1의 순환 스텝 하나가 든다. 이것이 중요한 수치이다.",
           "· seq_v2 스텝: CPU 중앙값 1.05 ms / p99 1.61 ms(격자 주기의",
           "  16%, p95 1.35), GPU 1.21 / 2.31 ms이며 이 크기에서 GPU는",
           "  배치 1에서 아무것도 사주지 않는다.",
           "· 세그먼트 재실행: 100행 2.9 / 5.6 ms, 300행 6.4 / 8.9 ms이다.",
           "· 같은 세션에서 인과 GP p99 2.34 ms, 윈도 W=2 4.46 ms(44.6%)이며",
           "  순서 seq_v2 < 인과 GP < 윈도 W=2 < W=4는 세션과 무관하다(§8ac)."],
          accent=BLUE, body_size=11)

    fcard(s, 6.75, 1.45, 6.03, 2.45,
          "불확실성: 모델을 건드리지 않는 분포 무가정 구간",
          ["· 분산·분위 헤드는 재학습이 필요하고 위의 모든 수치를",
           "  움직이므로 split conformal을 쓴다. 해당 실행 자신의 검증",
           "  분할에서 보정하고 예측기는 아무것도 바꾸지 않는다.",
           "· 변형 둘: 단일 분위(global), Δt 구간별(Mondrian)이다. 동일",
           "  절차를 두 기준선에도 적용하므로 비교되는 것은 구간 품질이다.",
           "· α = 0.10, TEST, 두 모집단: 모델 구간이 32/32 셀에서 두",
           "  기준선을 이긴다. Winkler는 Tᵢ 1,272 / 1,554(PCHIP) /",
           "  1,727(pers.) 컷, 포함 2,290 / 2,851 / 3,120이다."],
          accent=TEAL, body_size=11)

    fcard(s, 0.55, 4.00, 6.03, 1.85,
          "포함 모집단의 역설 — 더 넓은데 더 좋다",
          ["모델의 Tᵢ 구간은 PCHIP의 것보다 실제로 넓은데(반폭 224–255",
           "대 211–241 eV) 그럼에도 점수는 더 좋다. 스파이크가 빗나감",
           "벌점을 부풀리는데 모델이 덜 빗나가기 때문이다. Mondrian은",
           "모든 Tᵢ 팔을 ≈4–5% 좁히고 판정은 바꾸지 않는다."],
          accent=ORANGE, body_size=11)

    fcard(s, 6.75, 4.00, 6.03, 1.85,
          "인정하는 한계 — 커버리지는 주변적이고 지연 절댓값은 기계 종속이다",
          ["Tᵢ 커버리지는 목표 0.90에 대해 0.87–0.92(모든 팔에서 한 분할이",
           "미달), V_rot은 0.91–0.94이며 shot별 커버리지는 넓게 흩어진다.",
           "5세션 지연 프로토콜의 p99 산포가 21.84배에 이르러 1 ms 판정은",
           "보류되었고, 밀리초 대신 순서와 연산자 수(2–3 µs/op)를 주장한다."],
          accent=RED, body_size=11)

    band(s, 5.95,
         [[("실무 지침 — ", 13, TEAL, True, False, None),
           ("상태 유지형 나우캐스터를 제어 계산기의 CPU에서 실행하면 10 ms 예산의 80% 이상이 획득과 제어에 남는다. "
            "skill 점수와 쓸 만한 계측기 사이에 놓인 두 가지, 주기 안에서 도는가와 얼마나 믿어도 되는가를 이 절이 답한다.",
            13, WHITE, False, False, None)]], h=0.72)
    return note(s, """
main_ko.tex \\label{sec:deploy}(§8).

인용 시 유의점
· 지연은 유휴 노트북급 머신 한 대에서 워밍업 후 1,000회 호출, 네트워크 순전파만(특징 조립은 모델 밖) 잰
  값이다. 세션 간 절댓값은 2×(§8ab), 4.2×(§8ac), 21.84×(§8aj)까지 달라졌으며 순서만 불변이었다.
· 옛 판의 "CPU W=4 p99 6.4 ms · CUDA가 8× 느리다"와 "윈도 W=2 p99 18.9 ms"는 폐기되었다. 후자는 §8ac가
  같은 세션 재측정(4.455 ms)으로 오염을 판정하였다. 본문의 배치 문단은 이 값으로 교체한다.
· 32/32는 '모집단 × 타깃 × 변형 × 분할' 셀 수이다(2 × 2 × 2 × 4).
· conformal 구간 우위는 더 나은 점추정에서 따라 나오는 것이 아니라 별개의 성질이며 그렇게 진술한다.
· B.9 이후 추가할 문장: 비용 모델은 t ≈ N_ops × 2–3 µs이고, 순환은 도달 범위에 O(1)이며, 10 ms 예산은
  어느 arm에도 구속 조건이 아니다(§8ah). 1 ms 판정은 조용한 기계의 5세션 재실행을 기다린다.
""")


# --- 22. Headroom (sec:headroom) -----------------------------------------
def f_headroom():
    s = slide()
    header(s, "§9  sec:headroom", "남은 개선 여지 — 음성 결과가 지목하는 레버 3종 (전부 데이터)",
           accent=ORANGE)

    band(s, 1.42,
         [[("이 절의 규칙 — ", 12.5, TEAL, True, False, None),
           ("음성 결과는 “어떤 변경이 이를 움직이는가”를 말할 때만 실린다. 용량은 배제되었고(§6.9), "
            "더 긴 윈도도 배제되었으며(§6.8), 도달 범위를 사는 프레이밍은 채택되었고(§6.3), 문맥은 약 50 ms에서 포화하며 계열은 동률이다(§6.12). "
            "남은 것은 데이터이고, 각 레버는 이미 이 데이터셋 안의 증거로 지목된다.",
            12.5, WHITE, False, False, None)]], h=0.84)

    fcard(s, 0.55, 2.40, 3.95, 3.70,
          "1. CES 피팅 품질 메타데이터",
          ["두 모집단 보고가 존재하는 이유는",
           "값 기준 컷이 “피팅이 실패했다”의",
           "대리 지표이기 때문이다.",
           "어떤 방법도 예측할 수 없는 단일",
           "표본 상향 사건을 제거하지만 한쪽",
           "방향뿐이고(하향 급락은 손대지",
           "않는다) ≥2× 상향 이상치의 19%만",
           "잡는다(§3.5).",
           "표본별 피팅 χ²이나 신호 수준이",
           "있다면 품질 컷이 모든 팔에서",
           "값 컷을 대체하거나 동반할 수 있고,",
           "두 모집단이 하나로 합쳐진다.",
           "V_rot 자신의 스파이크(>1,000 km/s",
           "119행)도 같은 규칙으로 처리된다."],
          accent=BLUE, body_size=11)

    fcard(s, 4.72, 2.40, 3.95, 3.70,
          "2. Mirnov 정보는 전처리가 파괴하였다",
          ["같은 10 ms 격자, 연속 블록 내 lag-1",
           "자기상관은 BES +0.568 · ECEI +0.572 ·",
           "Mirnov -0.009(블록의 82%가 |r| < 0.1)",
           "이다. 즉 이 격자 '위에서' 자기 채널은",
           "백색잡음이다. kHz dB/dt를 안티",
           "앨리어싱 없이 100 Hz로 데시메이션한",
           "결과이며 연속 표본의 상대 위상이",
           "무작위가 된다.",
           "해법은 모델이 아니라 전처리이다.",
           "원시 kHz 시계열에서 윈도별 RMS ·",
           "대역 파워 · 모드 수 · 모드 회전",
           "주파수를 뽑아 V_rot 분기로 라우팅한다.",
           "B.6 shot 집합은 동결되었고(§8ao),",
           "예측은 변동 3분위 승률의 선행 개선이다."],
          accent=TEAL, body_size=11)

    fcard(s, 8.89, 2.40, 3.89, 3.70,
          "3. 액추에이터 채널이 없다",
          ["토로이달 회전은 주입된 토크가",
           "결정하는데 이 데이터셋에는 토크",
           "신호가 존재하지 않는다.",
           "데이터가 그 단절을 보여준다. shot",
           "사이 ECE 유래 Tₑ 대리는",
           "· Tᵢ와 r = +0.353 (p = 3×10⁻¹⁷)",
           "· V_rot과 r = +0.024 (p = 0.58)",
           "shot 내부에서도 Tᵢ 상관은 부호가",
           "일관되나 V_rot은 무작위이며, 방전",
           "단위 Tₑ 수준도 V_rot 승패를 예측하지",
           "못한다(§8an). 즉 파워는 토크가 아니다.",
           "NBI 토크(또는 빔별 파워·기하) 확보는",
           "모델링이 아니라 데이터 획득 과제이며,",
           "양성 대조는 DIII-D 전방전 시뮬레이터이다."],
          accent=ORANGE, body_size=11)

    band(s, 6.20,
         [[("맺는 문장 — ", 12.5, TEAL, True, False, None),
           ("이 중 어느 것도 현재 결과가 천장에 있다는 진술이 아니다. 다음의 측정 가능한 이득이 있는 자리를 "
            "값싸게 시도할 수 있는 순서로 늘어놓은 것이며, 셋 모두 아카이브된 KSTAR 데이터에서 실행 가능하다. 양자 가지는 하드웨어 검증 후 음성으로 종결되었다(§8ap).",
            12.5, WHITE, False, False, None)]], h=0.70)
    return note(s, """
main_ko.tex \\label{sec:headroom}(§9). 보조 그림이 필요하면 fig_mirnov.png(자기상관 사실, 유효)이다.

PROJECT_KNOWLEDGE "Framing Rules"(승상님 2026-08-05): 음성 결과는 그것을 뒤집을 측정을 함께 지목할 때만
결론이 된다. "정보가 부족하다"는 결론이 아니다.

· 레버 2가 시도한 모든 파생 MC 특징(적분, PCHIP 적분, |MC|, 이동 RMS)이 실패한 이유이다. 이미 잃은 정보는
  하류에서 복구할 수 없다.
· 레버 2는 V_rot에 대해 지목할 수 있는 가장 가치 높은 실험이며 아카이브 데이터로 검정 가능하다. B.6의 shot
  집합(test 31921·31873·31114·31902 / pool 31097·31359·31747·32027·32092·32097 / companion 31923·31357)은
  2026-08-21 동결되었고, #32092(EHO n = 1, ~4/~8 kHz)가 다섯 번째 양성 대조이다. §8an의 예측: V_rot 승률은
  변동 3분위에서 먼저 올라야 하며, 균일하게 또는 조용한 3분위에서 오르면 메커니즘이 틀린 것이다.
· 옛 판의 레버 1("깊이가 아니라 도달 범위 — W=2→8 커버리지")은 §6.3에서 '채택된 프레이밍'이므로 개선
  여지가 아니며 목록에서 제외되었다. B.9는 문맥·계열 축도 닫았다.
· 양자 가지(§8ap): VQC는 무잡음 시뮬레이터에서도 persistence에 지고(471.5 vs 449.6 eV), 예측당 22.9 s이며,
  Forte의 탈분극 λ = 0.661 ± 0.040이다. 어떤 수치도 백본 옆에 인용되지 않으며 논문에는 한 문장으로만 남긴다.
""")


# --- 23. Limits + conclusion (sec:limits / sec:conclusion) ---------------
def f_limits_conclusion():
    s = slide()
    header(s, "§10–11  sec:limits · sec:conclusion",
           "한계와 결론 — 무엇을 인정하고 무엇을 주장하는가", accent=NAVY)

    fcard(s, 0.55, 1.45, 6.03, 5.20,
          "§10 한계 — 논문이 먼저 인정하는 것",
          ["· 통계적 검정력: 재현 단위는 shot이고 분할당 test shot이",
           "  96(Tᵢ) / 60–66(V_rot)뿐이며 shot별 제곱오차 차이는 꼬리가",
           "  두껍다. 포함 모집단에서는 행의 ≈1%가 모든 팔 Tᵢ 제곱오차의",
           "  70–83%를 담는다.",
           "· MNAR 낙관: skill은 관측 지점에서만 측정된다. 재가중은 인과",
           "  비교에 대해서는 양쪽, 오프라인 비교에 대해서는 모집단 조건부",
           "  유의성으로 상한을 정하며 결측 Tᵢ 54–68% · V_rot 4–6%만 포괄한다.",
           "· 오프라인 주장의 상한은 GP 동률(8셀 중 1개 유의)이다.",
           "  “미래를 쓰는 보간을 이긴다”는 “사전등록된 보간들을 이긴다”를",
           "  뜻하며, 다만 배치 가능한 최강 방법인 인과 GP로는 확장된다.",
           "· 값 기준 컷은 대리 지표이고 V_rot 스파이크는 컷되지 않는다.",
           "· 캠페인 전이는 하나의 시간 블록에 기댄다(4개 분할이 아니라",
           "  4개 초기화, 컷 실행 2/4가 epoch 상한 종료). shot별 표준화는",
           "  오프라인 형태이며 인과 러닝/EWMA 판본은 측정되지 않았다.",
           "· B.9: 통합 재채점은 방법의 기대 skill이지 단일 체크포인트의",
           "  주장이 아니고, 승패 공변량 분석은 탐색적이며, 1 ms 판정은",
           "  세션 산포 21.84배로 보류되었다. V_rot 우위는 shot-general이 아니다.",
           "· 단일 장치 · 단일 진단 집합 · 불확실성은 주변적 보정 · 지연은",
           "  네트워크만 · 격자는 99.46% 균일(불규칙 샘플링 주장 없음).",
           "· 범위: 페데스탈 상단 프레이밍은 데이터 선정에서 물려받았고,",
           "  반경 의존성·사건 위상(ELM, L–H) 분석은 수행하지 않았다."],
          accent=RED, body_size=10.5)

    fcard(s, 6.75, 1.45, 6.03, 2.95,
          "§11 결론 — 네 문단의 뼈대",
          ["① 관측 모집단: Tᵢ에 대해 미래를 쓰는 PCHIP을 유의하고 재현",
           "  가능한 skill로 이기고(4개 독립 분할 × 두 모집단, +0.17~+0.32),",
           "  가장 강한 오프라인 평활기와 동률이며, 두 타깃 모두에서 모든",
           "  인과 기준선을 이긴다(인과 GP 8개 셀 전부, Δt > 15 ms 포함).",
           "② 배치 주장은 두 스트레스 모두의 지지를 받는다: 진짜 결측·",
           "  도메인 내에서 인과 대비 8/8 생존, 캠페인 경계 너머 PCHIP·",
           "  인과 GP 대비 4/4+4/4이다. 그 차이는 측정이다.",
           "③ 약 50 ms의 연속 인과 문맥이 우위를 전형적으로 만들며(승률",
           "  0.52 → 0.66), 계열은 skill이 아니라 비용을 정한다(추가 예정).",
           "④ 작동하지 않는 곳과 그 이유도 함께 보고하며, 각각은 구체",
           "  적이고 검정 가능한 변경을 지목한다. 크기·문맥·계열 축은 닫혔다."],
          accent=GREEN, body_size=10.5)

    fcard(s, 6.75, 4.48, 6.03, 2.17,
          "결론이 명시하는 기여",
          ["① 교정된 V_rot 평가 — 관측값의 54%가 계측기 유지값이며",
           "  학습·채점·앵커 전 구간에서 제거하였다.",
           "② 스펙트럼 피팅 실패의 두-모집단 처리와, 어느 한쪽만으로는",
           "  왜 충분하지 않은지를 보이는 측정이다.",
           "③ 최강 배치 기준선을 넘어서게 하는 도달 범위의 전체격자 프레이밍과,",
           "  그 도달 범위가 실제로 약 50 ms임을 측정한 사다리이다.",
           "④ 모든 결정 규칙이 결정할 수치보다 먼저 적힌 test 동결 프로토콜이다."],
          accent=NAVY, body_size=10.5)
    return note(s, """
main_ko.tex \\label{sec:limits}(§10), \\label{sec:conclusion}(§11).

· 재현성 · 코드 가용성 · 데이터 가용성 문단은 이미 논문에 있으며 별도 슬라이드로 만들지 않는다. 코드 공개에
  남은 TODO 하나는 투고본에 대한 아카이브 DOI(Zenodo) 발급 후 인용 추가이다.
· 결론 2문단의 마지막 문장을 그대로 인용한다: "온라인 가상 센서는 미래를 읽는 보간이 아니라 persistence와
  인과 평활기와 경쟁하며, 그 비교로 보면 나우캐스터는 중요한 지점에서 작동한다."
· 결론 ③은 B.9 이후 추가되는 문단이다. §9의 lead claim을 그대로 쓰고, 두 한정(10k 아래 합성곱 우세,
  70 ms attention 열세)은 한 문장으로 붙인다.
· 결론이 나열하는 '안 되는 것': 회전은 전역 동률이며 우위는 회전이 변하는 방전에 집중(§8an), 값 컷은 한쪽
  방향 대리 지표, 재가중은 결측 Tᵢ 54–68%와 결측 V_rot의 20분의 1에 닿음, 예측 구간은 주변적 보정, 45 ms를
  넘어서면 포함 모집단은 동률, 1 ms 판정은 보류.
· §9.6 "쓰지 않는 문장": "모든 오프라인 방법을 이긴다"(GP 동률), "윈도 프레이밍은 정보로 기각된다"(87%가
  warm-up), "1 ms 배치 판정", "문맥은 길수록 좋다"(630 ms 유계가 무한보다 낫다), "V_rot는 검정력 부족".
""")


def build():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    f_title()            # 1
    f_map()              # 2
    f_two_claims()       # 3
    f_data_setup()       # 4
    f_audits()           # 5
    f_model()            # 6
    f_eval()             # 7
    f_res_ladder()       # 8   §6.1
    f_res_headline()     # 9   §6.2  sec:headline
    f_res_gate()         # 10  §6.3  sec:gate
    f_res_gap()          # 11  §6.4  sec:gap
    f_res_mnar()         # 12  §6.5  sec:mnar
    f_res_campaign()     # 13  §6.6  sec:campaign
    f_res_asym()         # 14  §6.7  sec:asym
    f_res_window()       # 15  §6.8  sec:window
    f_res_ladder2()      # 16  §6.9  sec:ladder
    f_res_peak()         # 17  §6.10 sec:peak
    f_res_cutsens()      # 18  §6.11 sec:cutsens
    f_selection()        # 19  §7    sec:selection
    f_b9()               # 20  §6.12 (추가 예정) B.9
    f_deploy()           # 21  §8    sec:deploy
    f_headroom()         # 22  §9    sec:headroom
    f_limits_conclusion()  # 23  §10–11
    prs.save(OUT)
    print(f"wrote {OUT}  ({len(prs.slides._sldIdLst)} slides)")
    for w in _WARNED:
        print("  FIT WARNING:", w)
    if not _WARNED:
        print("  layout: every card and band fits at its chosen type size")


if __name__ == "__main__":
    build()
