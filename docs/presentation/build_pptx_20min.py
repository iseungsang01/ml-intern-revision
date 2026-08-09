# -*- coding: utf-8 -*-
"""Build the 20-minute KSTAR CES nowcasting talk (Korean, 원자핵공학과 대학원 내부 발표).

Output: docs/presentation/KSTAR_CES_발표자료_20분.pptx

This is the short sibling of ``build_pptx.py`` (32 slides, ~60 min thesis defense).
It reuses that module's palette, layout helpers and the slides that already fit a
short talk, and replaces the rest with merged/compressed versions:

    1h deck                                   -> 20min deck
    s_diagnostics + s_problem                 -> t_background
    s_data + s_contract + s_split             -> t_pipeline
    s_methodology + s_bootstrap               -> t_eval
    s_stuck                                   -> t_audit      (compressed)
    s_res_transient                           -> t_transient  (compressed)
    s_agenda                                  -> t_overview   (message-first)
    (dropped) s_arch_detail, s_res_gap, s_mirnov, 5 of 7 dividers

Every slide carries speaker notes with a running clock so the deck can actually be
delivered in 20 minutes. Figures come from docs/presentation/figures/
(run make_figures.py / make_figure_architecture.py first).

Usage (from repo root):
    python docs/presentation/build_pptx_20min.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import build_pptx as B  # noqa: E402  (path bootstrap must run first)
from build_pptx import (  # noqa: E402
    prs, slide, box, text, header, bullets, card, add_image_fit, divider,
    NAVY, BLUE, TEAL, ORANGE, GREEN, RED, GRAY, LGRAY, MGRAY, WHITE, DARK, CARDBG,
    MONO, EMU_W, EMU_H, FIG,
)
from pptx.util import Inches, Pt  # noqa: E402
from pptx.dml.color import RGBColor  # noqa: E402
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR  # noqa: E402


def note(s, txt):
    """Attach speaker notes to a slide."""
    s.notes_slide.notes_text_frame.text = txt.strip("\n")
    return s


def kicker(s, txt):
    """Rewrite the header kicker (small accent line above the slide title).

    Slides reused from the 1-hour deck carry that deck's section numbering
    ("6. 결과 ①", "7. 결론", …); this renumbers them for the 20-minute running order.
    """
    for sh in s.shapes:
        if not sh.has_text_frame or sh.top is None:
            continue
        if abs(sh.top - Inches(0.20)) < Inches(0.03) and abs(sh.left - Inches(0.55)) < Inches(0.03):
            runs = sh.text_frame.paragraphs[0].runs
            if runs:
                runs[0].text = txt
                for extra in runs[1:]:
                    extra.text = ""
            return s
    raise RuntimeError(f"kicker textbox not found (wanted to set {txt!r})")


# ---- post-edit helpers for slides reused from the 1-hour deck -------------
# The 중간 보고 deck (2026-07-30, hand-edited in PowerPoint from this deck's own
# output) trimmed and retitled several reused slides. Rather than fork those
# slide builders, we rebuild them and apply the same edits here, so build_pptx.py
# keeps producing the 1-hour deck unchanged.

def _text_shapes(s):
    return [sh for sh in s.shapes if sh.has_text_frame]


def drop_shape(s, needle):
    """Delete the whole textbox whose text contains `needle`."""
    for sh in _text_shapes(s):
        if needle in sh.text_frame.text:
            sh._element.getparent().remove(sh._element)
            return s
    raise RuntimeError(f"drop_shape: no shape containing {needle!r}")


def drop_para(s, needle):
    """Delete the single paragraph containing `needle`."""
    for sh in _text_shapes(s):
        for p in sh.text_frame.paragraphs:
            if needle in p.text:
                p._p.getparent().remove(p._p)
                return s
    raise RuntimeError(f"drop_para: no paragraph containing {needle!r}")


def set_para(s, needle, new_text):
    """Rewrite the paragraph containing `needle`, keeping its first run's format."""
    for sh in _text_shapes(s):
        for p in sh.text_frame.paragraphs:
            if needle in p.text:
                runs = p.runs
                if not runs:
                    continue
                runs[0].text = new_text
                for extra in runs[1:]:
                    extra.text = ""
                return s
    raise RuntimeError(f"set_para: no paragraph containing {needle!r}")


# ========================= NEW / COMPRESSED SLIDES ========================

# --- 1. Title -------------------------------------------------------------
def t_title():
    s = slide()
    box(s, 0, 0, EMU_W, EMU_H, fill=NAVY)
    box(s, 0, Inches(5.6), EMU_W, Inches(1.9), fill=RGBColor(0x0E, 0x26, 0x47))
    box(s, Inches(0.9), Inches(1.7), Inches(2.2), Pt(4), fill=ORANGE)
    text(s, Inches(0.9), Inches(1.9), Inches(11.6), Inches(0.5),
         [[("원자핵공학과 대학원 내부 발표 · 20분", 16, RGBColor(0x9C, 0xC0, 0xE8), True, False, None)]])
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
    return note(s, """
⏱ 00:00–00:30  (30초)

"안녕하십니까, 이승상입니다. 오늘은 KSTAR의 빠른 진단들로 CES 결측 구간을 채우는
연구를 20분간 말씀드리겠습니다."

핵심 한 문장(여기서 미리 던지기):
  "느리고 자주 비는 CES를, 항상 측정되는 빠른 진단으로 채울 수 있는지 —
   그것도 미래까지 보는 오프라인 보간을 이길 수 있는지 통계적으로 검증했습니다."

※ 용어 정리를 여기서 한 번: forecasting(예보)이 아니라 nowcasting(현재 시점 결측 채우기).
""")


# --- 2. Message-first overview -------------------------------------------
def t_overview():
    s = slide()
    header(s, "Overview", "결론")
    # Bodies are lists of lines: the 중간 보고 edit broke each message onto its own
    # line instead of one em-dash-joined sentence.
    msgs = [
        ("1", "빠른 진단은 Tᵢ 정보를 실제로 운반한다", GREEN,
         ["과거만 보는 causal 모델이, 미래까지 보는 오프라인 보간(PCHIP)을 CES_TI에서 "
          "통계적으로 유의하게 능가",
          "- 독립 test split 4개 모두 PASS (genuine skill +0.18 ~ +0.28; held 포함 평가에서도 4/4)."]),
        ("2", "V_rot에서는 통계적으로 무의미", ORANGE,
         ["Tᵢ ↔ V_rot 비대칭. 회전을 구동하는 NBI 토크가 데이터에 없고 Mirnov는 100 Hz에서 "
          "aliasing → 빠른 진단에 회전 정보가 거의 없다.",
          "물리로 예측되고 ablation으로 확인."]),
        ("3", "모델의 가치는 '보간이 약한 곳'에 집중된다", BLUE,
         ["매끄러운 bulk 구간은 보간이 이미 거의 최적. 우위는 고변동(peak)·급변 구간에 몰려 있다 "
          "(Tᵢ peak skill +0.70, V_rot +0.44, 둘 다 PASS)."]),
    ]
    yy = 1.72
    for num, t, col, body in msgs:
        box(s, Inches(0.6), Inches(yy), Inches(12.2), Inches(1.45), fill=CARDBG, round_=True)
        box(s, Inches(0.72), Inches(yy + 0.26), Inches(0.9), Inches(0.9), fill=col, round_=True)
        text(s, Inches(0.72), Inches(yy + 0.26), Inches(0.9), Inches(0.9),
             [[(num, 28, WHITE, True, False, None)]], align=PP_ALIGN.CENTER,
             anchor=MSO_ANCHOR.MIDDLE)
        text(s, Inches(1.82), Inches(yy + 0.18), Inches(10.8), Inches(0.45),
             [[(t, 17, NAVY, True, False, None)]])
        text(s, Inches(1.82), Inches(yy + 0.68), Inches(10.8), Inches(0.72),
             [[(line, 12.5, DARK, False, False, None)] for line in body], line_spacing=1.12)
        yy += 1.60
    return note(s, """
⏱ 00:30–01:30  (60초)

이 슬라이드가 발표 전체의 요약입니다. 여기서 결론을 먼저 다 말하고 들어갑니다.

"결론부터 말씀드리면 세 가지입니다.
 (1) 빠른 진단은 이온온도 정보를 실제로 운반합니다. 과거만 보는 저희 모델이,
     미래까지 다 보는 오프라인 보간을 Tᵢ에서 통계적으로 유의하게 이겼습니다.
 (2) 반면 회전 속도에서는 이기지 못했습니다. 이건 실패가 아니라 물리적으로 예측된
     비대칭이고, 저는 이걸 이 연구의 과학적 발견이라고 봅니다.
 (3) 모델의 가치는 전 구간에 고르게 있는 게 아니라, 보간이 약한 급변 구간에 몰려 있습니다."

※ 시간이 밀리면 이 슬라이드는 30초로 줄이고 뒤에서 회수.
""")


# --- 3. Background: CES problem + diagnostics ----------------------------
def t_background():
    s = slide()
    header(s, "1. 배경 & 문제", "CES는 왜, 얼마나 비는가")
    text(s, Inches(0.55), Inches(1.40), Inches(12.3), Inches(0.68),
         [[("CES는 pedestal 물리의 핵심량 Tᵢ · V_rot을 주지만 ", 14.5, DARK, False, False, None),
           ("광자 수집이 필요해 느리고 자주 빈다", 14.5, ORANGE, True, False, None),
           (". 반면 빠른 진단은 같은 10 ms 격자에서 항상 조밀하다.", 14.5, DARK, False, False, None)]])
    # Layout note: the 중간 보고 edit slid both cards left to free the right edge
    # for a screenshot of a raw shot CSV, so the missingness is shown as data and
    # not only as percentages.
    LX, RX = 0.16, 6.10
    # left: diagnostics table
    box(s, Inches(LX), Inches(2.05), Inches(6.5), Inches(3.48), fill=CARDBG, round_=True)
    text(s, Inches(LX + 0.25), Inches(2.18), Inches(6.0), Inches(0.4),
         [[("진단 구성 (동일한 10 ms 격자)", 15, NAVY, True, False, None)]])
    rows = [
        ("CES", "타겟", "Tᵢ, V_rot — 느림 · 결측", ORANGE),
        ("BES  9 ch", "입력", "밀도요동 nₑ 공간구조 → Tᵢ 단서", BLUE),
        ("ECEI  4 ch", "입력", "전자온도 Tₑ 2D → Tᵢ 단서", TEAL),
        ("Mirnov  2 ch", "입력", "자기요동 — lag-1 r=−0.01 (정보 소실)", GRAY),
    ]
    yy = 2.68
    for name, kind, desc, col in rows:
        box(s, Inches(LX + 0.25), Inches(yy), Inches(0.09), Inches(0.5), fill=col)
        text(s, Inches(LX + 0.45), Inches(yy + 0.02), Inches(1.75), Inches(0.4),
             [[(name, 13.5, col, True, False, None)]])
        text(s, Inches(LX + 2.17), Inches(yy + 0.04), Inches(0.75), Inches(0.4),
             [[(kind, 11.5, MGRAY, False, False, None)]])
        text(s, Inches(LX + 2.95), Inches(yy + 0.03), Inches(3.4), Inches(0.4),
             [[(desc, 12, DARK, False, False, None)]])
        yy += 0.64
    text(s, Inches(LX + 0.25), Inches(5.11), Inches(6.0), Inches(0.35),
         [[("→ 빠른 진단은 항상 100% 측정된다", 12.5, NAVY, True, False, None)]])
    # right: missing stats
    box(s, Inches(RX), Inches(2.05), Inches(5.5), Inches(3.48), fill=NAVY, round_=True)
    text(s, Inches(RX + 0.25), Inches(2.18), Inches(5.0), Inches(0.4),
         [[("결측 실태 (641 shot)", 15, ORANGE, True, False, None)]])
    for i, (lab, pct, col) in enumerate([("CES_TI 완전결측(NaN)", "8.2 %", RGBColor(0x9D, 0xE8, 0xCD)),
                                         ("CES_VT 완전결측(NaN)", "23.9 %", ORANGE)]):
        xx = RX + 0.25 + i * 2.6
        text(s, Inches(xx), Inches(2.66), Inches(2.5), Inches(0.4),
             [[(lab, 12, LGRAY, False, False, None)]])
        text(s, Inches(xx), Inches(2.97), Inches(2.5), Inches(0.55),
             [[(pct, 28, col, True, False, None)]])
    box(s, Inches(RX + 0.25), Inches(3.55), Inches(5.0), Inches(0.80),
        fill=RGBColor(0x8E, 0x2B, 0x22), round_=True)
    text(s, Inches(RX + 0.45), Inches(3.63), Inches(4.65), Inches(0.68),
         [[("+ held(같은 값이 지속적으로 유지) CES_VT 41.1 %",
            12.5, RGBColor(0xFF, 0xD5, 0xCE), True, False, None)],
          [("→ V_rot 실질 무정보 65.0 %", 14, WHITE, True, False, None)]],
         line_spacing=1.12, space_after=0)
    # right edge: raw CSV screenshot — blanks in CES_TI, a value repeating in
    # CES_VT (held), and the fast diagnostics filled on every row.
    add_image_fit(s, os.path.join(FIG, "fig_raw_csv_missing.png"),
                  Inches(11.48), Inches(2.06), Inches(1.77), Inches(3.35))
    text(s, Inches(RX + 0.25), Inches(4.44), Inches(5.0), Inches(1.05),
         [[("· 두 타겟이 서로 ", 12.5, WHITE, False, False, None),
           ("독립적으로", 12.5, ORANGE, True, False, None),
           (" 결측 → target별 처리 필요", 12.5, WHITE, False, False, None)],
          [("· 기존 대안 UFCES는 채널 제한 + 역산에 강한 물리 가정 필요",
            12.5, WHITE, False, False, None)],
          [("· 데이터 기반 접근은 축대칭 수준만 가정", 12.5, WHITE, False, False, None)]],
         line_spacing=1.2, space_after=5)
    # bottom band: the asymmetry preview
    box(s, Inches(0.55), Inches(5.70), Inches(12.25), Inches(1.05), fill=CARDBG, round_=True)
    box(s, Inches(0.55), Inches(5.70), Inches(0.12), Inches(1.05), fill=ORANGE)
    text(s, Inches(0.85), Inches(5.82), Inches(11.8), Inches(0.9),
         [[("핵심 비대칭 (미리보기)", 13, ORANGE, True, False, None)],
          [("빠른 진단은 물리적으로 Tᵢ 정보는 운반하지만 V_rot 정보는 거의 운반하지 않는다 — "
            "회전을 구동하는 NBI 토크가 데이터에 없고, Mirnov는 kHz dB/dt를 100 Hz로 순간샘플해 "
            "위상이 무작위화된다(lag-1 r = −0.01 vs BES +0.57). 이 가설이 결과에서 그대로 확인된다.",
            13.5, DARK, False, False, None)]], line_spacing=1.16)
    return note(s, """
⏱ 01:30–02:50  (80초)

"CES는 이온온도와 토로이달 회전이라는, pedestal 물리에서 가장 중요한 두 양을 줍니다.
 그런데 충분한 신호대잡음비를 얻으려면 광자를 오래 모아야 해서 느리고, 자주 빕니다.
 같은 10 ms 격자에서 Tᵢ는 8.2%, 회전은 23.9%가 값 자체가 비어 있습니다.
 그런데 회전은 여기서 끝이 아닙니다. 값이 채워져 있는 행 중에서도 54%가 직전 값을
 그대로 복사한 held 값입니다 — 진짜 측정이 아닙니다. NaN과 held를 합치면 회전은
 전체 행의 65%가 실질적으로 정보가 없습니다. Tᵢ는 held가 641 shot 전체에서 단 1행,
 사실상 0%입니다. 이 비대칭이 뒤에서 결과로 그대로 나타납니다.
 그리고 이 둘은 서로 독립적으로 빕니다 — 한쪽만 관측된 행이 많다는 뜻입니다.

 반대로 BES, ECEI, Mirnov 같은 빠른 진단은 같은 격자에서 100% 다 있습니다.
 그래서 질문은 자연스럽습니다: '항상 있는 것'으로 '자주 비는 것'을 채울 수 있는가."

물리적 연결고리 (질문 대비):
  · ECEI(Tₑ) + BES(nₑ) → 충돌 e–i 결합 τ_ei ∝ Tₑ^1.5/nₑ 을 통해 Tᵢ와 물리적으로 연결
  · 회전은 주로 NBI 토크가 구동 → 그 토크가 데이터에 없음 = 근본적 정보 부재

아래 밴드(비대칭 미리보기)를 짚으면서 "이 예측이 뒤에서 그대로 확인됩니다"로 넘어갈 것.
""")


# --- 6. Data & pipeline (merged) -----------------------------------------
def t_pipeline():
    s = slide()
    header(s, "3. 데이터 & 모델", "641 shot · No-Fake-Data · 누수 차단")
    # 중간 보고 edit: the collection-plan lines were cut — at 20 min the shot count
    # and the channel list are what the audience needs.
    bullets(s, Inches(0.55), Inches(1.5), Inches(6.5), Inches(2.6), [
        ("샷 #30801–#32751, 하드웨어 이력 고려 → 총 641 shot CSV", 0, NAVY, True),
        ("TEST(genuine, seed 42): 33,693 샘플 / 96 shot — Tᵢ 32,787 · V_rot 10,729 (held 포함 27,437은 민감도)", 1),
        ("입력 채널: BES 9 · ECEI 4 · Mirnov 2 · time 4 · ces_history 4", 0),
        ("타겟: 정규화된 [CES_TI, CES_VT] 2-벡터", 1),
    ], size=13.5, gap=8)
    # sample tensor table
    box(s, Inches(7.3), Inches(1.5), Inches(5.5), Inches(2.6), fill=NAVY, round_=True)
    text(s, Inches(7.55), Inches(1.62), Inches(5.0), Inches(0.4),
         [[("샘플 1개의 구성 (window = 4)", 14, ORANGE, True, False, None)]])
    rows = [
        ("bes / ecei / mc", "(4,9)(4,4)(4,2)", "빠른 진단"),
        ("time_features", "(4, 4)", "시간 인코딩"),
        ("ces_history", "(4, 4)", "이전 CES + flag"),
        ("target / mask", "(2,) / (2,)", "타겟 + 마스크"),
    ]
    yy = 2.12
    for name, shp, desc in rows:
        text(s, Inches(7.55), Inches(yy), Inches(2.15), Inches(0.4),
             [[(name, 11.5, WHITE, True, False, MONO)]])
        text(s, Inches(9.62), Inches(yy), Inches(1.55), Inches(0.4),
             [[(shp, 11, ORANGE, True, False, MONO)]])
        text(s, Inches(11.2), Inches(yy), Inches(1.5), Inches(0.4),
             [[(desc, 10.5, LGRAY, False, False, None)]])
        yy += 0.48
    # three principle cards
    cards = [
        ("① No Fake Data", ORANGE,
         ["보간으로 '가짜 라벨'을 만들지 않음",
          "진단 입력이 온전하고 타겟 중 최소 하나가",
          "관측된 시점만 샘플로 사용"]),
        ("② per-target masked MSE", BLUE,
         ["target_mask로 관측된 타겟만 손실 반영",
          "한쪽만 관측된 행도 그 타겟은 학습에 기여"]),
        ("③ 누수 차단", TEAL,
         ["행이 아닌 shot 파일 단위 split (인접 행 상관)",
          "정규화 통계는 train 파일에서만 추정",
          "ces_history의 타겟 시점은 값·flag 완전 마스킹"]),
    ]
    for i, (t, col, lines) in enumerate(cards):
        card(s, Inches(0.55 + i * 4.13), Inches(4.3), Inches(3.95), Inches(2.15),
             t, lines, accent=col, title_size=14, body_size=11.5)
    text(s, Inches(0.55), Inches(6.58), Inches(12.3), Inches(0.42),
         [[("결측 제거로 시간 간격이 불규칙해지므로 ", 12, DARK, False, False, None),
           ("연속성에 의존하는 LSTM 대신 1D CNN + 시간 인코딩", 12, NAVY, True, False, None),
           ("을 사용한다.", 12, DARK, False, False, None)]])
    return note(s, """
⏱ 05:45–07:00  (75초)

"데이터 수집 계획은 RMP로 ELM을 억제한 H-mode 구간을 노렸고, 실제 파일은
 shot당 30초 구간 CSV 641개입니다(#30801–#32751).
 test는 96 shot, 3만 3천여 샘플입니다(genuine 기준).

 전처리에서 세 가지를 특히 신경 썼습니다.
 첫째, 가짜 데이터를 만들지 않았습니다. 결측을 보간으로 메워서 학습에 쓰면
 '보간을 이기는가'라는 질문 자체가 무의미해집니다.
 둘째, 두 타겟이 독립적으로 비기 때문에 손실을 타겟별로 마스킹했습니다.
 이걸 안 하면 한쪽만 관측된 행 — 라벨의 약 28% — 을 통째로 버리게 됩니다.
 셋째, 누수 차단입니다. 인접 행은 강하게 상관되므로 행 단위가 아니라 shot 파일 단위로
 나눴고, 정규화 통계도 train 파일에서만 뽑았습니다.
 특히 이력 텐서에서 '지금 맞히려는 시점'은 값과 관측 flag를 모두 0으로 지웠습니다 —
 안 그러면 정답을 그대로 보게 됩니다."

※ 질문 대비: window=4는 직전 4스텝. 시간 간격이 불규칙하므로 lookback/delta와 그 log1p를
   4채널 time feature로 명시적으로 넣어 준다.
※ 질문 대비: V_rot '관측' 27,437은 held(직전값 유지 반복) 포함 집계 — bit-identical 반복을
   제거한 genuine V_rot은 10,729행. Tᵢ의 held는 1행뿐이라 사실상 전부 genuine.
""")


# --- 7b. Sample construction (detailed) ------------------------------------
def t_samples():
    s = slide()
    header(s, "3. 데이터 & 모델", "학습 샘플을 어떻게 만들었나 — 블록 → 윈도 → 증강")
    cards = [
        ("① 연속 블록 분할", BLUE,
         ["time 간격 ≥ 0.5 s = 측정 세그먼트 경계",
          "641/641 shot에 존재 (파일당 ~2개, 간극 중앙값 6.3 s)",
          "모델 윈도·보간 기준선 모두 경계를 넘지 않음",
          "→ 양쪽에 동일한 정보 경계 (공정 비교)"]),
        ("② 윈도 샘플 (W = 4)", TEAL,
         ["연속 4행, 마지막 행 = 맞혀야 할 타겟 시점",
          "채택: 3개 진단 입력 완전 + Tᵢ/V_rot 중 ≥1개 관측",
          "타겟 시점 이력은 값·관측 flag 모두 0으로 마스킹",
          "→ 자기 정답을 절대 볼 수 없음 (누수 차단)"]),
        ("③ Temporal subset 증강", ORANGE,
         ["타겟 이전 이력 행들의 부분집합을 전부 열거",
          "같은 타겟을 '이력 2개·3개' 버전으로도 별도 학습",
          "→ 배치가 다양한 이력 밀도·간격을 경험",
          "→ 실전의 불규칙한 결측 패턴에 강건"]),
        ("④ 규모 — 조합 폭발 제어", GREEN,
         ["부분집합 열거는 조합 폭발 → 시드 고정 랜덤 캡",
          "train 200,000 / val 40,000 샘플",
          "split·캡을 CSV로 디스크에 고정(pin) → 완전 재현",
          "재로딩 시 데이터와 불일치하면 예외 (조용한 drift 차단)"]),
    ]
    for i, (t, col, lines) in enumerate(cards):
        r, c = divmod(i, 2)
        card(s, Inches(0.55 + c * 6.2), Inches(1.5 + r * 2.55), Inches(6.0), Inches(2.4),
             t, lines, accent=col, title_size=14, body_size=12)
    return note(s, """
⏱ 08:30–09:30  (60초)

"모델에 들어가는 샘플이 어떻게 만들어지는지부터 보겠습니다.

 각 shot 파일은 0.5초 이상 간격으로 끊어진 측정 세그먼트 두세 개로 이루어져
 있습니다. 윈도는 세그먼트 안에서만 만들고, 보간 기준선에도 똑같은 경계를
 적용했습니다. 정보 조건을 양쪽에 동일하게 맞춘 겁니다.

 샘플 하나는 연속 4행이고 마지막 행이 맞혀야 할 시점입니다. 이 시점의 CES
 이력은 값과 관측 플래그를 모두 0으로 지워서, 모델이 자기 정답을 볼 수 있는
 경로를 원천 차단했습니다.

 그리고 증강입니다. 실전에서는 직전 이력이 2개일 수도 4개일 수도 있으니, 같은
 타겟을 이력 부분집합 버전으로도 만들어 학습시켰습니다. 조합이 폭발하기 때문에
 시드 고정 캡으로 train 20만, val 4만 샘플을 쓰고, split은 디스크에 고정해서
 완전히 재현 가능합니다."

※ 질문 대비: 증강은 train에만 영향 — 평가·채점은 증강 없는 원본 샘플로만 한다.
""")


# --- 7c. Training + model selection (plain language) -----------------------
def t_training():
    s = slide()
    header(s, "3. 데이터 & 모델", "어떻게 학습시켰나 — 손실 · 최적화 · 아키텍처 선택")
    cards = [
        ("손실 함수 – 물리적 페널티 고려", ORANGE,
         ["L = 관측 타겟만의 MSE + 0.1 × ReLU(z₀ − ŷ_Tᵢ)",
          "Tᵢ·V_rot은 독립적으로 결측",
          "→ 한쪽만 관측된 행도 그 타겟은 학습에 기여",
          "페널티 항 = '이온온도 < 0 keV 불가' 물리 제약",
          "  (z₀ = 물리 0을 정규화 단위로 옮긴 값)",
          "학습·출력은 정규화 단위 — 물리 단위 환산은 평가에서만"]),
        ("최적화", TEAL,
         ["AdamW — lr 10⁻³, weight decay 10⁻⁴",
          "batch 512 · 10 epochs · 단일 GPU",
          "LR 스케줄: val 손실이 2 에폭 정체하면 ×0.5",
          "gradient clipping (max-norm 5.0)",
          "split seed와 초기화 seed 분리",
          "→ 결과가 초기화 운이 아님을 별도 점검"]),
    ]
    for i, (t, col, lines) in enumerate(cards):
        card(s, Inches(0.55 + i * 6.2), Inches(1.5), Inches(6.0), Inches(2.95),
             t, lines, accent=col, title_size=14, body_size=12)
    box(s, Inches(0.55), Inches(4.65), Inches(12.25), Inches(2.0), fill=NAVY, round_=True)
    text(s, Inches(0.85), Inches(4.82), Inches(11.7), Inches(0.45),
         [[("아키텍처 선택 — 40여 회의 keep/discard 통제 실험", 15, ORANGE, True, False, None)]])
    text(s, Inches(0.85), Inches(5.32), Inches(11.7), Inches(1.25),
         [[("반복마다 구조 변경은 딱 하나 → 처음부터 재학습 → 증강 없는 검증셋에서 보간 대비 skill로 채점. "
            "최고 기록을 넘으면 채택, 못 넘으면 직전 최고 모델로 복원 — 후퇴 위에 쌓지 않는다.",
            12.5, WHITE, False, False, None)],
          [("채점 기준을 val loss → skill로 바꾼 것이 n.s.(+0.088) → 유의(+0.20~0.30) 진전의 핵심. "
            "TEST는 이 40여 회 전 과정에서 봉인.",
            12.5, LGRAY, False, False, None)]], line_spacing=1.25)
    return note(s, """
⏱ 09:30–10:30  (60초)

"학습 자체는 평범합니다.

 손실은 관측된 타겟만 반영하는 마스킹 MSE입니다. 두 타겟이 독립적으로 비기
 때문에, 한쪽만 관측된 행도 그 타겟은 학습에 씁니다. 여기에 이온온도가 0 아래로
 내려가지 못하게 하는 물리 페널티를 하나 더했습니다.
 최적화는 AdamW, 배치 512, 10 에폭 — 특별한 트릭이 없습니다.

 중요한 건 아키텍처를 고른 절차입니다. 한 번에 구조를 하나만 바꿔서 처음부터
 다시 학습하고, 증강 없는 검증셋에서 보간 대비 skill로 채점했습니다. 최고 기록을
 넘으면 채택하고, 못 넘으면 직전 최고 모델로 되돌렸습니다. 이걸 40여 번 반복했고,
 그 사이 test 세트는 한 번도 열지 않았습니다. 최종 모델은 그렇게 살아남은
 GRU 이력 인코더 + 타겟별 attention 구조, 파라미터 20만 개입니다."

※ 질문 대비: z₀ = 물리 0 keV를 정규화 단위로 옮긴 값. 페널티 가중 0.1은 고정(튜닝 안 함).
※ 질문 대비: '증강 val loss로 고르면 안 되나?' → 쉬운 구간이 복제돼 평활한 예측을
   보상한다(보간이 이미 잘하는 영역). 그래서 채점은 증강 없는 skill로만 했다.
""")


# --- 8. Evaluation methodology (merged) ----------------------------------
def t_eval():
    s = slide()
    header(s, "4. 평가 방법론", "이 숫자를 믿어도 되는 이유 — 두 가지 장치")
    cards = [
        ("장치 ① 채점 규칙을 결과 보기 전에 고정", BLUE,
         ["test 96 shot은 모델을 만들기 전에 봉인",
          "→ 선택이 끝날 때까지 한 번도 열지 않음",
          "모델 선택은 전부 검증셋에서만",
          "결과를 보기 전에 문서로 못박은 것들:",
          "  · 비교 상대는 PCHIP 보간으로 확정 (사후 교체 금지)",
          "  · 보간이 예측 못 하는 지점은 직전값으로 채점 (제외 금지)",
          "  · '이겼다' = 95% 신뢰구간이 0을 제외할 때만"]),
        ("장치 ② 신뢰구간은 행이 아니라 방전 단위", ORANGE,
         ["10 ms 간격의 인접 측정은 거의 복사본",
          "→ 34,644행 ≠ 34,644개의 독립 증거",
          "행을 독립으로 세면 신뢰구간이 너무 좁아져",
          "  '유의하게 이겼다'가 너무 쉽게 나옴 (거짓 확신)",
          "해법: 오차를 shot으로 묶어 96개 shot을",
          "  통째로 복원추출 × 10,000회 (paired bootstrap)",
          "→ '새로운 방전에서도 이길까?'에 답하는 CI"]),
    ]
    for i, (t, col, lines) in enumerate(cards):
        card(s, Inches(0.55 + i * 6.2), Inches(1.5), Inches(6.0), Inches(3.1),
             t, lines, accent=col, title_size=14.5, body_size=12)
    # skill definition band
    box(s, Inches(0.55), Inches(4.75), Inches(7.6), Inches(1.85), fill=CARDBG, round_=True)
    text(s, Inches(0.85), Inches(4.9), Inches(7.1), Inches(1.6),
         [[("Murphy (1988) skill score", 13.5, NAVY, True, False, None)],
          [("skill = 1 − MSE_model / MSE_baseline", 18, BLUE, True, False, None)],
          [("> 0 이면 모델 우수, = 0 이면 동률. 모든 오차는 물리 단위로 역정규화하여 타겟별로 계산.",
            12.5, DARK, False, False, None)]], line_spacing=1.22)
    # fairness box
    box(s, Inches(8.45), Inches(4.75), Inches(4.35), Inches(1.85), fill=NAVY, round_=True)
    text(s, Inches(8.7), Inches(4.9), Inches(3.9), Inches(1.6),
         [[("공정한 비교", 13.5, ORANGE, True, False, None)],
          [("모델과 모든 기준선을 같은 샘플 · 같은 관측 마스크로 채점 — "
            "누구도 유리한 표본을 받지 않음.",
            12, WHITE, False, False, None)],
          [("유효 표본 = shot 수 (Tᵢ 96 · V_rot 91)", 11.5, LGRAY, False, True, None)]], line_spacing=1.2)
    return note(s, """
⏱ 10:30–12:00  (90초)  ★ 통계 질문이 나오는 슬라이드 — 여기서 신뢰를 확보

"결과를 보여드리기 전에, 이 숫자를 믿어도 되는 이유를 먼저 말씀드리겠습니다.
 장치는 두 가지입니다.

 첫째, 채점 규칙을 결과 보기 전에 고정했습니다. test 96개 방전은 모델을 만들기
 전에 떼어놓고 선택이 끝날 때까지 한 번도 열지 않았습니다. 그리고 어떤 보간과
 비교할지, 보간이 예측 못 하는 지점은 어떻게 채점할지, '이겼다'를 뭐라고 정의할지를
 전부 미리 문서로 못박았습니다. 잘 나온 숫자를 사후에 고를 수 있는 길을 다 막은 겁니다.

 둘째, 신뢰구간을 행이 아니라 방전 단위로 계산했습니다. 한 방전 안의 10 ms 간격
 측정들은 거의 복사본입니다. 그걸 3만 4천 개의 독립 증거처럼 세면 확신이
 과장됩니다. 그래서 96개 방전을 통째로 재추출하는 bootstrap을 만 번 돌렸습니다.
 우리에게 불리한 계산이지만, '새로운 방전에서도 이길까'라는 진짜 질문에 답하는
 방식입니다.

 점수는 1 빼기 오차 비율입니다. 0보다 크면 보간보다 정확하다는 뜻이고,
 신뢰구간이 0을 제외할 때만 이겼다고 인정합니다."

※ 예상 질문 "왜 PCHIP인가?" → 단조성을 보존해 overshoot이 적은 보수적(=강한) 기준선.
   게다가 타겟 양쪽의 과거+미래를 다 본다. 즉 우리에게 불리하게 설계된 기준선.
※ 예상 질문 "최소 표본 조건은?" → test 최소 15 shot·관측 3,000행 이상을 사전 조건으로
   걸었고 실제 96 shot·3만 행 이상으로 충족.
""")


# --- 12b. Evaluation methodology, part 2: pre-registration + bootstrap ----
def t_eval2():
    """The mechanics behind the two 장치 of t_eval: the 3-way split with its
    pre-registered rules (PR1–PR4) and the shot-clustered paired bootstrap."""
    s = slide()
    header(s, "4. 평가 방법론", "선택 편향 차단 + shot 단위 부트스트랩")
    card(s, Inches(0.55), Inches(1.5), Inches(6.0), Inches(3.15),
         "3-way split + 사전등록 (PR1–PR4)", [
             "TEST는 아키텍처 탐색 시작 전 예약 — 선택 중 절대 열지 않음",
             "모델 선택은 val에서만 → 헤드라인에 winner's curse 없음",
             "PR1 headline baseline = PCHIP (단조 3차 보간)",
             "PR2 future-neighbor 없으면 persistence로 평가",
             "PR3 TEST floor: ≥15 shot & ≥3,000 Tᵢ 샘플 (충족)",
             "PR4 95% CI가 0을 제외하면 PASS",
         ], accent=BLUE, title_size=14.5, body_size=12)
    card(s, Inches(6.75), Inches(1.5), Inches(6.0), Inches(3.15),
         "Shot-clustered paired bootstrap", [
             "한 shot 안의 인접 CES 행은 강하게 상관됨",
             "샘플을 독립으로 보면 불확실성을 크게 과소평가",
             "per-sample paired error (SE_model − SE_pchip)를",
             "  shot 단위로 묶어 shot 전체를 복원추출 (B = 10,000)",
             "→ CI가 within-shot 가짜 복제가 아닌",
             "  진짜 shot-to-shot 일반화를 반영",
         ], accent=ORANGE, title_size=14.5, body_size=12)
    box(s, Inches(0.55), Inches(4.8), Inches(7.6), Inches(1.8), fill=CARDBG, round_=True)
    text(s, Inches(0.85), Inches(4.95), Inches(7.1), Inches(1.55),
         [[("Murphy (1988) skill score", 13.5, NAVY, True, False, None)],
          [("skill = 1 − MSE_model / MSE_baseline", 18, BLUE, True, False, None)],
          [("> 0 이면 모델 우수, = 0 이면 동률. 모든 오차는 물리 단위로 역정규화하여 타겟별로 계산.",
            12.5, DARK, False, False, None)]], line_spacing=1.22)
    box(s, Inches(8.45), Inches(4.8), Inches(4.35), Inches(1.8), fill=NAVY, round_=True)
    text(s, Inches(8.7), Inches(4.95), Inches(3.9), Inches(1.55),
         [[("공정성 보장", 13.5, ORANGE, True, False, None)],
          [("모델과 모든 baseline을 동일한 (file, row) 샘플 집합 · 동일 per-target mask로 "
            "채점 — 어떤 arm도 thinning 되지 않음.", 12, WHITE, False, False, None)],
          [("검정력 한계: Tᵢ 96 · V_rot 91 shot", 11.5, LGRAY, False, True, None)]], line_spacing=1.2)
    return note(s, """
⏱ 12:00–12:40  (40초)  — 앞 슬라이드의 '장치 두 가지'를 기계적으로 어떻게 구현했는지

"앞에서 말씀드린 두 장치를 구체적으로 어떻게 구현했는지만 짚고 넘어가겠습니다.

 왼쪽이 사전등록입니다. 3-way split에서 TEST는 아키텍처 탐색을 시작하기 전에
 떼어놨고, 모델 선택은 전부 val에서만 했습니다. 그래서 헤드라인 숫자에
 winner's curse가 없습니다. PR1부터 PR4까지가 결과를 보기 전에 못박은 규칙입니다.

 오른쪽이 부트스트랩입니다. 핵심은 오차를 행이 아니라 shot 단위로 묶어서
 shot 통째로 복원추출한다는 겁니다. 그래야 신뢰구간이 '같은 방전 안의 복제'가 아니라
 '새로운 방전으로의 일반화'를 반영합니다.

 점수는 Murphy skill score이고, 모든 arm을 동일한 샘플 집합과 동일한 마스크로
 채점하기 때문에 어느 쪽도 유리한 표본을 받지 않습니다."

※ 시간이 밀리면 이 슬라이드는 통째로 스킵 가능 — 앞 슬라이드가 이미 요지를 담고 있음.
""")


# --- 13. Transient case study (compressed) -------------------------------
def t_transient():
    s = slide()
    header(s, "5. 결과 ⑤", "급변 구간을 눈으로 — held-out shot #31815")
    add_image_fit(s, os.path.join(FIG, "fig_transient_31815.png"),
                  Inches(0.45), Inches(1.42), Inches(7.35), Inches(5.4))
    box(s, Inches(8.0), Inches(1.6), Inches(4.8), Inches(4.9), fill=CARDBG, round_=True)
    text(s, Inches(8.25), Inches(1.76), Inches(4.35), Inches(0.5),
         [[("한 shot에서 실제로 벌어지는 일", 15, NAVY, True, False, None)]])
    bullets(s, Inches(8.25), Inches(2.32), Inches(4.35), Inches(4.0), [
        ("빠른 진단이 급변을 먼저 본다", 0, NAVY, True),
        ("BES 급락(빨간 점선)이 CES crash와 정렬", 1),
        ("PCHIP는 스파이크마다 overshoot", 0, GRAY, True),
        ("과거+미래를 다 보는 오프라인 보간인데도", 1),
        ("Tᵢ: RMSE 210 vs PCHIP 263 → skill +0.36", 0, ORANGE, True),
        ("peak 구간만 보면 Tᵢ skill +0.63", 1),
        ("V_rot: +0.20 — 이 shot은 두 타겟 모두 승리", 0, BLUE, True),
        ("단, 보간을 이기는 건 test 89 shot 중 43개", 0, GRAY, True),
        ("→ 우위는 gap·peak 구간에 집중된다", 1, NAVY, True),
    ], size=12.5, gap=7)
    return note(s, """
⏱ 16:25–17:05  (40초)

"통계만 보면 감이 잘 안 오니, 실제 held-out shot 하나를 보여드리겠습니다. #31815입니다.

 위가 이온온도입니다. 검은 점이 실측, 파란 선이 저희 모델, 회색이 PCHIP 보간입니다.
 빨간 점선은 BES가 급락한 시점입니다. 보시면 빠른 진단의 급락이 CES crash와 정렬되어
 있습니다 — 빠른 진단이 급변을 먼저 봅니다.
 PCHIP은 스파이크마다 overshoot합니다. 미래까지 다 보고도 그렇습니다.
 이 shot에서 Tᵢ RMSE는 210 대 263, skill로 +0.36이고 peak 구간만 보면 +0.63입니다.

 다만 정직하게 말씀드리면, 이런 shot이 전부는 아닙니다. test 89개 shot 중 43개에서
 보간을 이깁니다. 즉 우위는 고르게 퍼져 있는 게 아니라 급변 구간에 집중되어 있습니다 —
 이건 앞의 peak 분석과 정확히 일치하는 그림입니다."

※ 시간이 밀리면 이 슬라이드를 30초로 압축하거나 건너뛸 것 (결과 ④까지로도 논지는 성립).
""")


# --- 13b. Window sweep: is W = 4 justified? ------------------------------
def t_window_sweep():
    """Answers the 'why window = 4?' feedback with the 24-run sweep curve.
    Numbers: THESIS_RESULTS.md §8f / data/.wsweep_summary.json."""
    s = slide()
    header(s, "5. 결과 ⑦", "window 민감도 sweep — 'W=4여야 하는가'에 답하다", accent=TEAL)
    text(s, Inches(0.55), Inches(1.36), Inches(12.3), Inches(0.56),
         [[("W ∈ {2,3,4,6,8} × seed 4개 + history-0 = ", 13, DARK, False, False, None),
           ("독립 run 24회", 13, NAVY, True, False, None),
           (". 발표 모델(iter009) 고정, W만 변화 — 매 run 자체의 held-out TEST skill_vs_pchip. "
            "held 제거 학습·평가, seed별 test shot 96개는 모든 W에서 동일(검증됨).",
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
           ("1차 실험의 'V_rot은 긴 window가 필요하다'는 held 오염이 짧은 window를 벌해서 생긴 착시였다.",
            12.5, GRAY, False, True, None)]], line_spacing=1.15)
    return note(s, """
⏱ 17:05–17:45  (40초)  ★ "왜 window가 4인가" 피드백에 대한 답

"발표 때 받은 지적 중 하나가 'window를 4로 쓴 근거가 뭐냐'였습니다.
 솔직히 처음에는 경험적으로 놓은 값이었습니다. 그래서 곡선으로 답을 만들었습니다.

 W를 2, 3, 4, 6, 8로 바꿔가며 seed 4개씩, 그리고 history를 아예 없앤 조건까지
 총 24번을 독립적으로 학습·평가했습니다. 모델은 발표한 것 그대로 고정하고 W만 바꿨습니다.
 held 값은 학습·평가 양쪽에서 전부 제외했습니다.

 세 가지가 보입니다.
 첫째, 왼쪽 끝 — 과거 CES를 아예 빼면 이온온도가 보간보다 나빠지고(−0.026)
 회전은 보간의 1.8배 오차가 납니다. 빠른 진단만으로는 보간 수준에도 못 미칩니다.

 둘째, 과거 관측 딱 하나가 전부를 만듭니다. W=2에서 이온온도 +0.238, 회전 +0.206으로
 두 타겟 모두 최대치에 올라서고, 그 뒤로는 평탄합니다. 구간 간 차이보다 seed 산포가
 더 크니까요. 그러니까 W=4는 필연이 아니고, 두 타겟 모두 W=2·3보다 낮습니다.

 셋째, 그럼에도 window를 넓힐 이유가 하나 있는데 정확도가 아니라 커버리지입니다.
 채점하려면 윈도 안에 관측이 하나는 있어야 하는데, W=2는 직전 행에 그 타겟이 없으면
 예측 자체를 못 냅니다. 긴 gap 샘플을 W=8은 4배에서 10배 더 많이 커버합니다."

※ 질문 대비: "왜 W=1이 없나?" → dataset이 window≥2를 요구. history-0는 W=4에서
   history 채널만 0으로 만든 ablation이고, 학습부터 그렇게 했다(평가만 가린 게 아님).
※ 질문 대비: "표본이 W마다 다르지 않나?" → 평가 표본은 W=2→8에서 1.8%만 감소하고
   test shot 96개는 동일. shot별 샘플 상한(500)으로 긴 shot 지배도 차단.
※ 질문 대비: "held를 남긴 채로 하면?" → 실제로 먼저 그렇게 돌렸고 'V_rot은 긴 window가
   필요하다'는 결과가 나왔다. held 제거 이득이 window가 짧을수록 크기 때문에(+0.088→+0.006)
   생긴 착시였다 — held가 긴 window를 돕는 게 아니라 짧은 window를 벌하고 있었다.
""")


# --- 14. Data-quality audit (compressed) ---------------------------------
def t_audit():
    s = slide()
    header(s, "5. 결과 ⑥", "정직성 점검: held/stuck CES_VT 감사", accent=RED)
    text(s, Inches(0.55), Inches(1.42), Inches(12.3), Inches(0.85),
         [[("후반 audit 결과 ", 14.5, DARK, False, False, None),
           ("관측된 CES_VT의 약 54%가 held/forward-fill", 14.5, RED, True, False, None),
           ("이었다 — 직전 관측값과 bit-identical(같은 시간블록 내 최대 1214행 연속). "
            "V_rot의 native cadence가 행 cadence보다 느려 값이 carry-forward된 것으로, 진짜 측정이 아니다. "
            "CES_TI는 held 0.0%로 사실상 무관.", 14, DARK, False, False, None)]], line_spacing=1.15)
    bullets(s, Inches(0.55), Inches(2.42), Inches(6.4), Inches(3.6), [
        ("641개 중 499 shot 파일이 영향받음", 0),
        ("평가: held 전 구간 채점 제외 — genuine-only가 headline", 0),
        ("학습도 오염 — 초기 '무해' 판정(단일 seed)을 4-seed paired가 뒤집음", 0, RED, True),
        ("→ held 제거 학습이 V_rot 4/4 개선 (평균 +0.039, 3/4 유의)", 1, RED, True),
        ("→ 현행 규약: 학습·평가 모두 제거 (CES_DROP_STUCK_TARGETS=1)", 1, NAVY, True),
        ("CES_TI는 genuine-only 평가에서도 4 seed 모두 PASS", 0, GREEN, True),
        ("Tᵢ↔V_rot 비대칭 결론은 그대로 유지 (genuine에서 VT는 3/4 seed n.s.)", 0),
    ], size=12.5, gap=8)
    # mini table
    box(s, Inches(7.2), Inches(2.42), Inches(5.6), Inches(3.95), fill=CARDBG, round_=True)
    text(s, Inches(7.45), Inches(2.56), Inches(5.2), Inches(0.4),
         [[("V_rot RMSE: 보고값 vs genuine", 14, NAVY, True, False, None)]])
    hdr = ["seed", "보고(stuck 포함)", "genuine"]
    xs = [7.45, 8.85, 11.2]
    ws = [1.3, 2.3, 1.5]
    yy = 3.06
    for h, x, w in zip(hdr, xs, ws):
        text(s, Inches(x), Inches(yy), Inches(w), Inches(0.4),
             [[(h, 12, GRAY, True, False, None)]])
    yy += 0.45
    for r in [["42", "22.5", "35.0"], ["1", "24.7", "34.8"],
              ["7", "30.0", "43.5"], ["123", "32.9", "46.5"]]:
        text(s, Inches(xs[0]), Inches(yy), Inches(ws[0]), Inches(0.4),
             [[(r[0], 13, DARK, False, False, MONO)]])
        text(s, Inches(xs[1]), Inches(yy), Inches(ws[1]), Inches(0.4),
             [[(r[1], 13, MGRAY, False, False, MONO)]])
        text(s, Inches(xs[2]), Inches(yy), Inches(ws[2]), Inches(0.4),
             [[(r[2], 13, RED, True, False, MONO)]])
        yy += 0.5
    text(s, Inches(7.45), Inches(yy + 0.06), Inches(5.1), Inches(0.7),
         [[("held 타겟은 baseline 오차가 ≈0이라 보고 RMSE를 끌어내린다. "
            "숫자는 정정했고 결론은 바뀌지 않았다.", 11.5, GRAY, False, True, None)]], line_spacing=1.12)
    return note(s, """
⏱ 17:05–17:45  (40초)

"연구 후반에 데이터 감사를 하다가 하나 발견한 게 있어서 정직하게 공유드립니다.

 관측된 것으로 표시된 CES_VT 값의 약 54%가 실은 직전 값의 forward-fill이었습니다.
 직전 관측값과 비트 단위로 동일하고, 심한 경우 1214행이 연속으로 같은 값입니다.
 회전 진단의 실제 cadence가 행 cadence보다 느려서 값이 carry-forward된 것으로 보입니다.
 CES_TI는 이 현상이 0%라 무관합니다.

 두 가지를 확인했습니다. 첫째, 평가가 영향을 받습니다. held 값은 baseline 오차가
 거의 0이라 보고 RMSE를 35~55% 끌어내립니다. 표의 오른쪽이 진짜 값입니다.
 둘째, 학습도 오염됩니다. 처음엔 단일 seed A/B로 '무해하다'고 봤는데, 4-seed paired
 재학습이 이를 뒤집었습니다 — held를 학습에서도 빼면 V_rot이 4개 분할 전부에서
 좋아집니다. forward-fill이 '이력 복사가 최적'이라고 모델을 가르치고 있던 겁니다.

 그래서 지금 규약은 학습·평가 모두 held 제거이고, Tᵢ는 여전히 4개 seed 모두 PASS,
 비대칭 결론도 그대로입니다."

※ 이 슬라이드는 내부 발표에서 신뢰를 크게 얻는 지점. 방어적으로 말하지 말고
   "찾아서 고쳤고 결론은 견뎠다"는 톤으로.
""")


# --- 17. Limitations + future (compressed) -------------------------------
def t_limits():
    s = slide()
    header(s, "6. 한계 & 향후", "무엇을 주장하지 않는가, 다음은 무엇인가")
    box(s, Inches(0.55), Inches(1.5), Inches(6.0), Inches(4.0), fill=CARDBG, round_=True)
    box(s, Inches(0.55), Inches(1.5), Inches(0.12), Inches(4.0), fill=RED)
    text(s, Inches(0.8), Inches(1.64), Inches(5.5), Inches(0.5),
         [[("한계", 16, RED, True, False, None)]])
    bullets(s, Inches(0.8), Inches(2.2), Inches(5.55), Inches(3.2), [
        ("검정력: test shot Tᵢ 96 · V_rot 61개(genuine)뿐", 0),
        ("모든 유의성 판정의 근본 구속조건", 1),
        ("MNAR: 관측 지점 채점은 낙관 상한 → 재가중으로 정량화", 0),
        ("인과 우위만 결측 지점에서 생존 (+0.29, 4/4)", 1, NAVY, True),
        ("캠페인 시간 분할: 오프라인 보간 우위 소멸(0/4) — 원인 측정", 0),
        ("커버리지: W=4에서 결측 Tᵢ 54.1% · V_rot 4.8%만 도메인 내", 0),
        ("metric 비대칭: 보간은 full-shot 이웃, 모델은 window = 4", 0),
    ], size=12, gap=8)
    box(s, Inches(6.8), Inches(1.5), Inches(6.0), Inches(4.0), fill=CARDBG, round_=True)
    box(s, Inches(6.8), Inches(1.5), Inches(0.12), Inches(4.0), fill=TEAL)
    text(s, Inches(7.05), Inches(1.64), Inches(5.5), Inches(0.5),
         [[("향후 연구", 16, TEAL, True, False, None)]])
    bullets(s, Inches(7.05), Inches(2.2), Inches(5.55), Inches(3.2), [
        ("① 이력 reach 확장 (슬롯 유지·span 확대) — 커버리지 직접 개선", 0, NAVY, True),
        ("② 원본 kHz Mirnov window 특징 — V_rot 최대 레버", 0, NAVY, True),
        ("100 Hz 무필터 데시메이션이 파괴한 정보의 상류 복원", 1),
        ("③ NBI 토크 신호 확보 — 회전의 원인 변수", 0, NAVY, True),
        ("④ 고속 진단 shot별 표준화 — 캠페인 전이 수리", 0),
        ("더 많은 test shot · 다중 캠페인 → 검정력 직접 보강", 0),
    ], size=12, gap=8)
    text(s, Inches(0.55), Inches(5.72), Inches(12.3), Inches(0.6),
         [[("검증된 음성 결과: ", 12.5, RED, True, False, None),
           ("Mirnov를 적분·PCHIP적분·|MC|·이동RMS로 재가공해도 학습 성능 개선 없음 "
            "(CES_TI 평균 −0.074, 4/4 음수). 재가공이 아니라 미관측 물리량 확보가 레버다.",
            12.5, DARK, False, False, None)]])
    return note(s, """
⏱ 18:50–19:20  (30초)  — 빠르게 읽되 첫 항목은 반드시 언급

"한계를 분명히 말씀드리면, 가장 큰 제약은 검정력입니다. test shot이 90여 개뿐이고,
 이게 모든 유의성 판정을 구속합니다. 그리고 채점은 관측된 지점에서만 하므로
 이 수치는 낙관적 상한입니다 — 진짜 결측 구간은 참값이 없으니까요.

 향후 가장 확실한 레버는 NBI 토크 신호를 데이터에 넣는 것입니다.
 참고로 Mirnov를 여러 방식으로 재가공해서 회전 정보를 살려보려는 시도는 이미 했고,
 4개 seed 전부에서 성능이 오히려 떨어졌습니다. 재가공이 아니라 미관측 물리량을
 확보하는 게 답이라는 뜻입니다."
""")


# --- 17b. Reframing experiment: full-grid seq-LSTM + masked loss ----------
def t_stuckfree():
    s = slide()
    header(s, "7. 진행 중인 후속 연구",
           "held 제거 학습: 데이터셋 구성과 결과", accent=RED)
    text(s, Inches(0.55), Inches(1.36), Inches(12.3), Inches(0.60),
         [[("가설(피드백): ", 13.5, RED, True, False, None),
           ("‘V_rot은 강하게 지속적’이라는 물리 가정 자체가 held 값(관측의 54%)이 만든 "
            "인공 지속성 위에서 학습된 순환일 수 있다 — held를 빼고 다시 학습해 검증.",
            13.5, DARK, False, False, None)]], line_spacing=1.12)
    # left: how the dataset was built
    box(s, Inches(0.55), Inches(1.98), Inches(5.75), Inches(4.55), fill=CARDBG, round_=True)
    box(s, Inches(0.55), Inches(1.98), Inches(0.10), Inches(4.55), fill=RED)
    text(s, Inches(0.82), Inches(2.10), Inches(5.3), Inches(0.4),
         [[("데이터셋 구성 (통제 변수 1개)", 14.5, NAVY, True, False, None)]])
    bullets(s, Inches(0.82), Inches(2.58), Inches(5.35), Inches(3.85), [
        ("held 판정", 0, RED, True),
        ("같은 연속 시간블록(Δt<0.5s)에서 직전 관측값과 bit-identical한 값", 1),
        ("로드 시점 NaN 처리 → 4곳에서 동시 제거", 0, NAVY, True),
        ("학습 target (채점 제외) · 이력 채널 값+관측 flag", 1),
        ("정규화 통계 · 보간 baseline의 anchor", 1),
        ("파일 분할은 대조군 manifest에 고정(pinning)", 0),
        ("같은 train/val/test shot — test 누출 원천 차단", 1),
        ("그 외 전부 동일: 아키텍처(iter009)·epochs·평가", 0),
        ("평가는 양쪽 다 genuine 관측만, 행 단위 동일 모집단(paired)", 1),
    ], size=12, gap=6)
    # right: forest plot
    B.add_image_fit(s, os.path.join(FIG, "fig_stuckfree_paired.png"),
                    Inches(6.50), Inches(1.92), Inches(6.35), Inches(4.68))
    box(s, Inches(0.55), Inches(6.60), Inches(12.25), Inches(0.46), fill=CARDBG, round_=True)
    text(s, Inches(0.85), Inches(6.68), Inches(11.8), Inches(0.34),
         [[("V_rot 4/4 seed 양수 · 3/4 유의 (평균 +0.039) · Tᵢ 손해 없음", 12.5, GREEN, True, False, None),
           ("  →  held는 무해한 prior가 아닌 오염원 · 학습 기본값을 held-free로 변경",
            12.5, DARK, False, False, None)]])
    return note(s, """
⏱ 19:20–19:40  (20초)

"오늘 회의에서 나온 가설을 바로 검증했습니다. V_rot이 강하게 지속적이라는
 가정 자체가, 직전 값을 복사한 held 인공물 위에서 학습된 순환일 수 있다는
 지적이었죠. 그래서 held를 로드 시점에 NaN 처리해서 — 학습 target, 이력 입력,
 정규화 통계, 보간 anchor 네 곳에서 동시에 빼고 — 나머지는 전부 동일하게
 다시 학습했습니다. 결과는 회전이 4개 seed 전부 좋아지고 3개는 통계적으로
 유의합니다. 이온온도는 손해가 없고요. held는 무해한 사전분포가 아니라
 학습을 오염시키는 인공물이었다는 게 확정됐습니다."
""")


def t_seq():
    """Seq-reframing result. The 중간 보고 edit replaced the forest-plot figure with
    the per-seed paired table — the numbers read faster than the plot at 20 min."""
    s = slide()
    header(s, "7. 진행 중인 후속 연구",
           "재프레이밍 실험: 전체 격자 LSTM + 결측은 loss에서만 마스킹")
    text(s, Inches(0.55), Inches(1.40), Inches(12.3), Inches(0.42),
         [[("결측 행을 버리지 않고 전 행을 causal LSTM에 넣은 뒤 관측된 값에만 채점 — "
            "4-seed paired vs 최종모델 iter009 (shot-clustered bootstrap).",
            14, DARK, False, False, None)]], line_spacing=1.12)
    # (seed, TI number, TI verdict, TI colour, VT number, VT verdict, VT colour).
    # Numbers get MONO so the columns line up; the Korean verdict must NOT — MONO
    # has no Hangul glyphs and would render as tofu.
    seq_rows = [
        ("42", "+0.078  [-0.010, +0.162]", "n.s.", DARK,
         "-0.103  [-0.231, -0.021]", "유의하게 나쁨", RED),
        ("1", "+0.030  [-0.015, +0.086]", "n.s.", DARK,
         "-0.048  [-0.106, -0.003]", "유의하게 나쁨", RED),
        ("7", "+0.011  [-0.038, +0.075]", "n.s.", DARK,
         "-0.061  [-0.202, +0.051]", "n.s.", DARK),
        ("123", "+0.049  [+0.012, +0.080]", "유의", GREEN,
         "+0.017  [-0.015, +0.035]", "n.s.", DARK),
    ]
    cols = [(0.85, 1.4, "seed"), (2.45, 4.9, "paired CES_TI (seq vs iter009)"),
            (7.55, 4.9, "paired CES_VT")]
    box(s, Inches(0.55), Inches(2.02), Inches(12.25), Inches(0.52), fill=NAVY, round_=True)
    for x, w, label in cols:
        text(s, Inches(x), Inches(2.12), Inches(w), Inches(0.36),
             [[(label, 13, WHITE, True, False, None)]])
    yy = 2.68
    for seed, ti, ti_v, ti_col, vt, vt_v, vt_col in seq_rows:
        box(s, Inches(0.55), Inches(yy), Inches(12.25), Inches(0.72), fill=CARDBG, round_=True)
        text(s, Inches(0.85), Inches(yy + 0.18), Inches(1.4), Inches(0.4),
             [[(seed, 14, NAVY, True, False, None)]])
        for x, num, verdict, col in ((2.45, ti, ti_v, ti_col), (7.55, vt, vt_v, vt_col)):
            text(s, Inches(x), Inches(yy + 0.18), Inches(4.9), Inches(0.4),
                 [[(num + "   ", 13.5, col, col is not DARK, False, MONO),
                   (verdict, 13, col, col is not DARK, False, None)]])
        yy += 0.82
    box(s, Inches(0.55), Inches(6.42), Inches(12.25), Inches(0.52), fill=CARDBG, round_=True)
    text(s, Inches(0.85), Inches(6.50), Inches(11.8), Inches(0.4),
         [[("Tᵢ: 4/4 seed 양수(평균 +0.042) · 자체 PR4 4/4 PASS", 13, GREEN, True, False, None),
           ("   |   ", 13, GRAY, False, False, None),
           ("V_rot 격차는 §8t seq v2에서 해소 — 라우팅을 인코더에서 차단 + held-free + shot별 표준화 → 유의 열세 4/4 → 0/4",
            12.5, DARK, False, False, None)]])
    return note(s, """
⏱ 19:40–20:00  (20초, 시간 있을 때만)

"피드백로 진행한 재프레이밍 실험입니다 — 결측 행을 버리는 대신 전체 격자를
 causal LSTM에 넣고 loss에서만 마스킹합니다. 이온온도는 4개 seed 전부에서
 최종 모델보다 좋아졌고 자체 PR4 게이트도 전부 통과했습니다.
 회전은 나빠졌는데, 분리 실험으로 원인이 둘 다 확정됐습니다 — held 인공물과
 회전 head의 빠른진단 차단(라우팅) 부재입니다. 그 둘을 모두 넣은 seq v2에서
 유의 열세가 4개 seed 전부에서 사라졌고, 이온온도는 4개 split 전부에서 최고입니다.
 학습 비용은 1/10입니다."

※ 시간 부족 시: "결측을 버리지 않고 loss에서 마스킹하는 재프레이밍으로 Tᵢ는 최종
   모델을 4/4에서 능가했고, 회전 격차도 라우팅을 인코더에서 차단해 닫혔습니다."
""")


# --- 18. Closing ----------------------------------------------------------
def t_closing():
    s = slide()
    box(s, 0, 0, EMU_W, EMU_H, fill=NAVY)
    box(s, Inches(0.9), Inches(0.95), Inches(2.2), Pt(4), fill=ORANGE)
    text(s, Inches(0.9), Inches(1.15), Inches(11.5), Inches(0.6),
         [[("한 장 요약 — Key Takeaways", 26, WHITE, True, False, None)]])
    points = [
        ("항상 있는 빠른 진단으로 자주 비는 CES를 채우는 데이터 기반 가상 센서", ORANGE),
        ("미래까지 보는 오프라인 보간을 causal 모델로 이기는 — 의도적으로 어려운 bar", BLUE),
        ("CES_TI는 4 seed 모두 보간을 통계적으로 유의하게 능가 (shot-clustered CI)", GREEN),
        ("CES_VT는 동률(n.s.) — 빠른 진단에 회전 정보가 없다는 Tᵢ↔V_rot 비대칭", TEAL),
        ("모델의 가치는 고변동(peak)·급변 구간에 집중 · causal baseline은 압도적 능가", ORANGE),
    ]
    yy = 2.25
    for t, col in points:
        box(s, Inches(0.95), Inches(yy + 0.05), Inches(0.28), Inches(0.28), fill=col)
        text(s, Inches(1.45), Inches(yy - 0.06), Inches(11.0), Inches(0.6),
             [[(t, 16, WHITE, False, False, None)]], line_spacing=1.1)
        yy += 0.68
    box(s, Inches(0.9), Inches(5.95), Inches(11.5), Pt(2), fill=RGBColor(0x2A, 0x47, 0x6E))
    text(s, Inches(0.9), Inches(6.2), Inches(11.5), Inches(0.7),
         [[("감사합니다.  ", 22, WHITE, True, False, None),
           ("Q & A", 22, ORANGE, True, False, None)]])
    return note(s, """
⏱ 19:40–20:00  (20초) + Q&A

"정리하면, 항상 있는 빠른 진단으로 자주 비는 CES를 채우는 가상 센서를 만들었고,
 미래까지 보는 보간이라는 일부러 어려운 기준선을 세워 검증했습니다.
 이온온도는 통계적으로 유의하게 이겼고, 회전은 이기지 못했으며 그 이유가 물리적으로
 설명됩니다. 후속 실험은 앞 슬라이드대로 진행 중입니다. 감사합니다."

── 예상 질문 대비 ──────────────────────────────
Q. 왜 forecasting이 아니라 nowcasting인가?
 → 미래를 예보하는 게 아니라 이미 지나간 시점의 빈 값을 채우는 문제. 오프라인 물리 분석용.

Q. 진짜 결측 구간에서도 이만큼 잘하나?
 → 단정하지 않는다. 결측이 무작위가 아니므로(MNAR, 저 SNR·ELM·천이에서 drop-out)
    관측 지점 성능은 낙관적 상한. 그래서 "추정한다"고만 말한다.

Q. 보간이 미래를 보는데 이기는 게 가능한가?
 → 가능하고, 그게 논지의 핵심. 빠른 진단이 시간 보간으로 얻을 수 없는 정보를 운반하기 때문.
    특히 보간이 구조적으로 실패하는 급변 구간에서 이긴다.

Q. 모델 크기/과적합은?
 → 파라미터 0.20 M. capacity가 아니라 일반화가 관건이었고, 무리한 확대는 오히려 악화됐다.

Q. V_rot을 살리려면?
 → NBI 토크 신호 확보가 1순위. 원본 kHz Mirnov도 후보. 파생 특징 재가공은 이미 음성 확인.

Q. seed에 따라 결과가 흔들리지 않나?
 → CES_TI는 4개 독립 split 전부 PASS로 강건. CES_VT는 split seed에 따라 크게 요동하며,
    이것 자체가 "현 규모에서 V_rot 단일 수치는 측정 불가"라는 근거.
""")


# ============================== build =====================================
def build():
    # ① 도입
    t_title()
    t_overview()
    t_background()
    s_idea = kicker(B.s_idea(), "1. 배경 & 문제")
    # 중간 보고 edits: spell out *why* nowcasting is the right framing, and split
    # the MNAR caveat onto its own line so it reads as a stated limitation.
    set_para(s_idea, "초해상(super-resolution)과도 구분", "초해상(super-resolution)과도 구분")
    _tf = [sh for sh in s_idea.shapes
           if sh.has_text_frame and "초해상" in sh.text_frame.text][0].text_frame
    _p = _tf.add_paragraph()
    _r = _p.add_run()
    _r.text = ("여기서 nowcasting(과거 + 현재 데이터로 현재 예측) 하는 이유는 CES가 찍히기 "
               "바로 직전의 고해상도 데이터(ECEI, BES, MC) + CES 옛날 데이터 이용해서 "
               "CES 현재 값 알아낼 수 있다고 생각했음")
    _r.font.size = Pt(11)
    _r.font.color.rgb = DARK
    set_para(s_idea, "결론의 성격",
             "결론의 성격: masking 검증에서 baseline을 이기면 결측 구간에서도 잘 복원할 것으로 추정한다.")
    _tf2 = [sh for sh in s_idea.shapes
            if sh.has_text_frame and "결론의 성격" in sh.text_frame.text][0].text_frame
    _p2 = _tf2.add_paragraph()
    _r2 = _p2.add_run()
    _r2.text = ("다만, 결측이 무작위라는 보장이 없어(MNAR) 결측 지점 정확도를 단정 불가 "
                "라는 한계를 지니고 있음.")
    _r2.font.size = Pt(12.5)
    _r2.font.color.rgb = DARK
    note(s_idea, """
⏱ 02:50–04:00  (70초)

"그래서 연구 질문은 이렇습니다. CES가 비어 있는 10 ms 시점에서, 같은 시각의 빠른 진단과
 과거 CES 이력만으로, 시간 보간이 복원할 수 없는 정보를 회복할 수 있는가.

 세 가지를 짚고 넘어가겠습니다.
 첫째, 이건 가상 센서입니다. 물리 역산 가정 없이 데이터로 추정합니다.
 둘째, forecasting이 아니라 nowcasting입니다. 미래를 예보하는 게 아니라 지금 비어 있는
 값을 채웁니다. 초해상과도 다릅니다.
 셋째, 검증 방식입니다. 진짜 결측 지점은 참값이 없어서 직접 검증이 불가능합니다.
 그래서 관측된 CES를 일부러 가린 뒤 얼마나 복원하는지로 채점합니다.

 다만 결측이 무작위라는 보장이 없어서 — 저 SNR이나 급변 구간에서 잘 빠지니까요 —
 결측 지점의 정확도를 단정하지 않고 '추정한다'고만 말씀드립니다."
""")
    s_bar = B.s_bar()
    # 중간 보고 edits: drop the fallback detail (covered by PR2 on the new
    # methodology slide) and stop calling the baseline "오프라인" twice.
    set_para(s_bar, "모델을 오프라인 CES-only 보간",
             "모델을 CES-only 보간(linear · PCHIP · local AR)과 비교한다")
    set_para(s_bar, "보간 baseline (오프라인)", "보간 baseline")
    drop_para(s_bar, "0.5 s 이상 gap은 보간 거부")
    note(s_bar, """
⏱ 04:00–05:45  (105초)  ★ 이 발표에서 가장 중요한 슬라이드

"평가 기준선을 어떻게 잡았는지가 이 연구의 핵심입니다.

 흔한 방식은 persistence, 즉 직전 값 유지와 비교하는 겁니다. 그건 너무 쉽습니다.
 저는 일부러 훨씬 어려운 상대를 골랐습니다 — CES만 쓰는 오프라인 보간,
 구체적으로 linear, PCHIP, local AR입니다.

 여기서 정보 비대칭을 보십시오. 오른쪽 보간은 맞히려는 시점의 앞뒤, 즉 과거와 미래
 CES를 모두 씁니다. 반면 저희 모델은 그 시점의 빠른 진단과 과거 이력 4스텝만 쓰고
 미래는 전혀 보지 않습니다. causal입니다.

 그래서 만약 미래를 보는 보간을 과거만 보는 모델이 이긴다면, 그건 '운이 좋아서'가
 아닙니다. 빠른 진단이 시간 보간으로는 절대 얻을 수 없는 CES 정보를 실제로 운반한다는
 강력한 증거가 됩니다. 그리고 미래 보간을 이기는 모델은 persistence 같은 causal
 기준선은 자명하게 이깁니다."

※ 여기서 청중이 "왜 굳이 불리하게?"라고 생각할 수 있으니 마지막 문장을 강조.
""")

    t_pipeline()
    note(kicker(B.s_arch(), "3. 데이터 & 모델"), """
⏱ 07:00–08:30  (90초)

"모델은 late-fusion multimodal 구조입니다.

 왼쪽부터, 진단별로 전용 인코더를 둡니다. 섣불리 평균 내면 공간 채널 구조가 사라지기
 때문에, BES·ECEI·Mirnov를 각각 시간 정보와 함께 1D convolution으로 인코딩하고
 마지막에 합칩니다.

 가운데가 이력 인코더입니다. 양방향 GRU를 씁니다. 여기가 재미있는 부분인데,
 맞히려는 시점이 window 한가운데에서 마스킹되어 있기 때문에, 양방향 GRU가
 앞뒤 관측 이웃을 모두 취합하게 됩니다. 즉 구조적으로 '학습된 보간'처럼 동작합니다.

 그 위에 관측 마스크 attention pooling을 얹었습니다. 타겟별로 독립적인 multi-head
 additive attention인데, 관측 flag가 1인 행에만 softmax를 허용합니다.
 '관측된 것만 본다'는 보간의 귀납 편향을 명시적으로 주입한 겁니다.

 마지막으로 타겟별 head가 갈라지는데, 물리 기반 routing이 들어갑니다.
 Tᵢ head는 빠른 진단 + 이력 + 시간을 다 받고, V_rot head는 이력과 시간만 받습니다.
 전체 파라미터는 20만 개로 작습니다 — capacity가 아니라 일반화가 관건이었습니다."

※ 질문 대비: LSTM 대신 CNN을 쓴 이유는 결측 제거로 시간축이 불규칙해졌기 때문.
※ 질문 대비: 아키텍처 선택 게이트를 val loss가 아니라 clean skill로 바꾼 게 결정적이었다
   (증강된 val loss는 쉬운 구간이 복제돼 '평활한 예측'을 보상 → 보간이 이미 잘하는 영역).
""")
    t_samples()
    t_training()
    t_eval()
    t_eval2()

    divider("5", "결과", "causal 압도 · Tᵢ 보간 유의 · Tᵢ↔V_rot 비대칭")
    s_ladder = kicker(B.s_res_ladder(), "5. 결과 ①")
    # 중간 보고 edit: the local-AR/linear extrapolation formula, overlaid on the
    # ladder so the weakest baseline's definition is visible while reading it.
    add_image_fit(s_ladder, os.path.join(FIG, "fig_ar_formula.png"),
                  Inches(5.82), Inches(4.51), Inches(2.06), Inches(0.54))
    note(s_ladder, """
⏱ 12:00–12:45  (45초)

"먼저 가장 강건한 결과입니다. RMSE 사다리인데, 낮을수록 좋습니다.

 두 타겟 모두 저희 모델이 사다리 맨 아래, 즉 오차가 가장 작습니다.
 persistence나 local AR 같은 causal 기준선은 큰 마진으로 이깁니다.
 이건 논쟁의 여지가 없는 결과이고, 실시간이나 온라인 상황 — 미래 값을 애초에 쓸 수
 없는 상황 — 에서는 모델이 명확한 승자라는 뜻입니다.

 보간과 비교해도 point estimate로는 앞섭니다. 다만 point estimate가 앞선다고 해서
 '유의하게 이겼다'고 말하면 안 됩니다. 그래서 다음 슬라이드에서 신뢰구간을 봅니다."
""")
    note(kicker(B.s_res_forest(), "5. 결과 ②"), """
⏱ 12:45–14:15  (90초)  ★ 헤드라인 결과

"이게 헤드라인 결과입니다. forest plot이고, 가로축이 PCHIP 보간 대비 skill입니다.
 점이 추정치, 가로 막대가 shot-clustered 95% 신뢰구간입니다.
 막대가 0선을 넘지 않으면 통계적으로 유의하게 이긴 겁니다.

 위쪽 CES_TI를 보시면, 4개의 독립적인 test split — seed 42, 1, 7, 123 — 전부에서
 신뢰구간이 0을 제외합니다. genuine 기준 skill은 +0.18에서 +0.28이고,
 held 포함 모집단에서도 4개 모두 PASS라 데이터 처리 방식에 불변입니다.
 특히 seed 1, 7, 123은 아키텍처 선택 과정에서 단 한 번도 쓰지 않은 split입니다.
 즉 우연히 잘 맞은 split을 고른 게 아닙니다.

 아래쪽 CES_VT는 다릅니다. PASS는 seed 1 하나뿐인데, 4개 중 1개 유의는 잡음이
 만들어낼 수 있는 수준입니다. 추정치는 전부 양수지만 저는 이걸 '이겼다'고 쓰지 않고
 동률로 보고합니다. 이 차이가 왜 생기는지가 다음 슬라이드입니다."

※ 질문 대비: "왜 4개 seed인가?" → split seed에 따른 안정성 점검. 단, headline CI에
   풀링하지 않고 각각 독립적으로 판정한다(PR4).
""")
    note(kicker(B.s_res_asym(), "5. 결과 ③"), """
⏱ 14:15–15:35  (80초)  ★ 과학적 발견

"이 비대칭이 왜 생기는지를 ablation으로 확인했습니다. 왼쪽 그림입니다.

 Tᵢ부터 보시면, 빠른 진단만 주고 CES 이력을 완전히 뺀 fast-only 모델도
 persistence를 +0.372로 이깁니다. 즉 빠른 진단 자체가 이온온도 정보를 진짜로
 운반한다는 뜻입니다. 물리적으로도 자연스럽습니다 — 충돌 e-i 결합을 통해
 ECEI가 주는 Tₑ와 BES가 주는 nₑ가 Tᵢ와 연결되니까요.

 회전은 정반대입니다. fast-only V_rot은 −0.64입니다. persistence보다 나쁩니다.
 회전 정보는 거의 전적으로 과거 CES 이력에서 옵니다.
 이유는 두 가지입니다. 첫째, 토로이달 회전을 구동하는 건 주로 NBI 토크인데
 그 신호가 데이터에 없습니다. 둘째, Mirnov는 kHz로 진동하는 dB/dt를 100 Hz로
 순간샘플한 것이라 샘플마다 위상이 무작위가 됩니다. 실제로 같은 격자에서 BES와
 ECEI는 lag-1 자기상관이 +0.57인데 Mirnov만 -0.01, 즉 백색잡음입니다.
 MHD 모드 정보가 모델에 닿기 전에 이미 소실된 겁니다.

 ※ 질문 대비: "Tₑ가 NBI 가열을 대리하니 회전 정보도 담기지 않나?" — 검정했습니다.
 shot 간 Tₑ~Tᵢ는 r=+0.35 (p~1e-17)로 경로가 실재하지만, Tₑ~V_rot은 r=+0.02 (p=0.58)로
 전달되지 않습니다. power와 torque가 다르기 때문입니다.

 그래서 저는 V_rot에서 이기지 못한 걸 실패로 보지 않습니다.
 물리적으로 예측되고 데이터로 확인된 발견입니다."
""")
    note(kicker(B.s_res_peak(), "5. 결과 ④"), """
⏱ 15:35–16:25  (50초)

"그럼 모델은 어디서 이득을 버는가. peak 분석입니다.

 여기서 peak은 예측값이 아니라 입력 기준으로 정의합니다 — 타겟 행을 제외한 이웃에서
 국소 활동도가 높은 구간입니다. 정답을 엿보지 않는 보수적인 정의입니다.

 CES_TI는 전체 +0.272에서 peak 구간 +0.702로 뜁니다.
 CES_VT도 전체로는 +0.131로 약하지만 peak에서는 +0.438이고, 둘 다 PASS입니다.

 해석은 명확합니다. 매끄러운 bulk 구간에서는 보간이 이미 거의 최적입니다.
 거기서는 이길 여지가 별로 없습니다. 반대로 급변 구간에서는 보간이 구조적으로
 실패하고, 바로 거기서 빠른 진단이 값어치를 합니다.
 즉 V_rot의 열세도 전역적인 게 아니라 국소적입니다."
""")
    t_transient()
    t_audit()
    t_window_sweep()

    note(kicker(B.s_conclusion(), "6. 결론"), """
⏱ 17:45–18:50  (65초)

"결론 네 가지입니다.

 하나, CES_TI는 미래까지 보는 오프라인 보간을 통계적으로 유의하게 능가합니다.
 4개 독립 split 전부, held 포함/제외 두 평가 모집단 모두에서요.
 둘, 배치에 관한 주장은 더 좁게 합니다. 진짜 결측 지점으로 재가중해도, 캠페인을
 시간으로 갈라도 살아남는 것은 인과 기준선 대비 우위뿐입니다 — +0.29와 +0.22.
 온라인 가상 센서가 실제로 경쟁하는 상대는 persistence이고, 그 비교에서 이깁니다.
 셋, CES_VT는 보간과 동률입니다. 다만 인과 대안은 캠페인 분할에서도 4개 전부
 유의하게 이깁니다 — 실패한 타겟이 아니라 의미 있는 비교가 인과뿐인 타겟입니다.
 넷, 그 Tᵢ와 V_rot의 비대칭이 이 연구의 과학적 기여입니다.
 빠른 진단은 10 ms 격자에서 이온온도 정보는 운반하지만 회전 정보는 거의 운반하지
 않는다 — 물리로 예측하고 ablation으로 확인했습니다."
""")
    t_limits()
    # t_ongoing() — the continuous-time batch it previewed finished, was rejected (§8e),
    # and its code was removed on 2026-08-09. t_seq() carries the follow-up story now.
    # t_stuckfree() — dropped in the 중간 보고 running order (held 감사 결과는 결과 ⑥에
    # 이미 나오고, 20분 안에 후속 연구 슬라이드 3장은 과했다). 1시간 덱에는 그대로 있다.
    t_seq()
    t_closing()

    out = os.path.join(HERE, "KSTAR_CES_발표자료_20분.pptx")
    prs.save(out)
    print("SAVED:", out, "| slides:", len(prs.slides._sldIdLst))


if __name__ == "__main__":
    build()
