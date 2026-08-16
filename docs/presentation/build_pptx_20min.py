# -*- coding: utf-8 -*-
"""Build the 20-minute KSTAR CES nowcasting talk (Korean, 원자핵공학과 대학원 내부 발표).

Output: docs/presentation/KSTAR_CES_발표자료_20분.pptx  (23 slides)

확정 프로토콜(2026-08-16, THESIS_RESULTS.md §8v–§8ab) 기준으로 전면 재작성:
W=2 · held-free(학습·평가) · 파일당 500 · 두 공동 1차 모집단(컷 / 포함) · 인과 GP 기준선.
주 모델은 전체격자 인과 시퀀스 나우캐스터 ``seq_v2``(357,570 파라미터), 옛 주 모델
(윈도 GRU + 관측마스킹 attention, 201,258)은 W=2 윈도 대조군이다. 모든 수치는
docs/paper/outline_ko_v2.tex = main_ko.tex = paper_numbers.json에서 왔다. W=4 시대의
수치·서사(progression, held 이중 보고, 연속시간 모델, seq 재프레이밍 실험)는 전부 제거했다.

이 덱은 ``build_pptx.py``(1시간 학위논문 발표)의 팔레트·레이아웃 헬퍼를 그대로 쓰고,
짧은 발표에 그대로 들어가는 슬라이드는 재사용하며(kicker()로 장 번호만 다시 매김),
나머지는 압축·병합본으로 새로 쓴다:

    1h deck                                          -> 20min deck
    s_diagnostics + s_problem + s_idea               -> t_background
    s_data + s_contract + s_split + s_stuck          -> t_pipeline
    s_bootstrap + s_validation + s_res_protocol      -> t_eval2
    s_stress + s_res_campaign                        -> t_stress   (결과 ④)
    s_limits + 개선 여지 3종 + 2026-08-16 결정        -> t_limits
    s_agenda                                         -> t_overview (message-first)
    (제외) s_arch_detail · s_res_gap · s_window_sweep · s_deploy ·
           s_mirnov · s_te_nbi · 모든 divider

모든 슬라이드에 러닝 클록이 붙은 발표자 노트가 있다(총 19:30 + Q&A).
그림은 docs/presentation/figures/ 에서 읽는다 (make_figures.py 먼저 실행).

Usage (from repo root):
    py docs/presentation/build_pptx_20min.py
    py docs/presentation/preview_pptx.py "docs/presentation/KSTAR_CES_발표자료_20분.pptx" --out docs/presentation/.preview20
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import build_pptx as B  # noqa: E402  (path bootstrap must run first)
from build_pptx import (  # noqa: E402
    prs, slide, box, text, header, bullets, card, add_image_fit, table,
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
# Cross-references inside reused slides point at the 1-hour deck's result
# numbering; rather than fork those slide builders we rebuild them and patch the
# offending paragraph here, so build_pptx.py keeps producing the 1-hour deck
# unchanged.

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


def set_run(s, needle, new_text):
    """Rewrite only the run containing `needle`, leaving the paragraph's other runs
    (and their colours/weights) alone — used to shorten a line that wraps."""
    for sh in _text_shapes(s):
        for p in sh.text_frame.paragraphs:
            for r in p.runs:
                if needle in r.text:
                    r.text = new_text
                    return s
    raise RuntimeError(f"set_run: no run containing {needle!r}")


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
    text(s, Inches(0.88), Inches(2.45), Inches(11.7), Inches(1.9),
         [[("KSTAR 다중 진단 기반 인과(causal) 나우캐스팅:", 30, WHITE, True, False, None)],
          [("희소 CES 신호(Tᵢ · V_rot)의 결측 구간 복원", 34, WHITE, True, False, None)]],
         line_spacing=1.12)
    text(s, Inches(0.9), Inches(4.30), Inches(11.5), Inches(1.25),
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
    return note(s, """
⏱ 00:00–00:30  (30초)

"안녕하십니까, 이승상입니다. KSTAR의 빠른 진단으로 CES 결측 구간을 채우는 연구를
 20분간 말씀드리겠습니다."

핵심 한 문장(여기서 미리 던지기):
  "느리고 자주 비는 CES를, 항상 측정되는 빠른 진단으로 채울 수 있는지 —
   그것도 미래까지 보는 오프라인 보간을 이길 수 있는지 통계적으로 검증했습니다."

※ 용어: forecasting(예보)이 아니라 nowcasting(현재 시점 결측 채우기).
※ 재실험(B.1–B.5) 후 확정 프로토콜 기준 수치다. 옛 W=4 발표 수치는 전부 폐기.
""")


# --- 2. Message-first overview -------------------------------------------
def t_overview():
    s = slide()
    header(s, "Overview", "결론부터 — 세 문장")
    msgs = [
        ("1", "빠른 진단은 Tᵢ 정보를 실제로 운반한다", GREEN,
         ["과거만 보는 인과 모델이, 미래까지 보는 오프라인 보간(PCHIP)을 Tᵢ에서 유의하게 능가",
          "컷 +0.17~+0.26 · 포함 +0.23~+0.32 — 두 모집단 × 4개 독립 분할 전부 PASS (4/4 + 4/4)",
          "배치 가능한 최강 인과 기준선(인과 GP)도 8개 셀 전부에서 이긴다"]),
        ("2", "V_rot는 전역 동률 — 물리로 예측된 비대칭", ORANGE,
         ["PR4 통과는 컷 1/4 · 포함 2/4(잡음 수준) → 회전 승리를 주장하지 않는다",
          "빠른 채널을 전부 0으로 만들어도 출력이 bit-identical — 회전 정보가 애초에 없다",
          "원인: 회전을 구동하는 NBI 토크 미관측 + Mirnov kHz 신호의 100 Hz 앨리어싱"]),
        ("3", "상한은 추정기가 아니라 정보다", BLUE,
         ["우위는 고변동(peak)·간극 구간에 집중 (Tᵢ peak +0.45~+0.72, 8/8 PASS)",
          "21,498 파라미터 모델이 컷에서 백본과 동급(+0.002), 폭 34k→879k는 평평",
          "→ 남은 레버는 모델 크기가 아니라 데이터 (피팅 메타 · kHz Mirnov · NBI 토크)"]),
    ]
    yy = 1.55
    for num, t, col, body in msgs:
        box(s, Inches(0.6), Inches(yy), Inches(12.2), Inches(1.62), fill=CARDBG, round_=True)
        box(s, Inches(0.72), Inches(yy + 0.30), Inches(0.9), Inches(0.9), fill=col, round_=True)
        text(s, Inches(0.72), Inches(yy + 0.30), Inches(0.9), Inches(0.9),
             [[(num, 28, WHITE, True, False, None)]], align=PP_ALIGN.CENTER,
             anchor=MSO_ANCHOR.MIDDLE)
        text(s, Inches(1.82), Inches(yy + 0.16), Inches(10.8), Inches(0.45),
             [[(t, 17, NAVY, True, False, None)]])
        text(s, Inches(1.82), Inches(yy + 0.62), Inches(10.8), Inches(0.95),
             [[(line, 12, DARK, False, False, None)] for line in body], line_spacing=1.12)
        yy += 1.75
    return note(s, """
⏱ 00:30–01:20  (50초)

이 슬라이드가 발표 전체의 요약입니다. 결론을 먼저 다 말하고 들어갑니다.

"결론부터 말씀드리면 세 가지입니다.
 (1) 빠른 진단은 이온온도 정보를 실제로 운반합니다. 과거만 보는 저희 모델이,
     미래까지 다 보는 오프라인 보간을 Tᵢ에서 유의하게 이겼습니다. 데이터 처리
     방식을 둘로 나눠 검증했는데(스파이크 컷/포함), 양쪽 4개 분할 전부 통과입니다.
 (2) 회전 속도는 전역적으로 동률입니다. 실패가 아니라 물리적으로 예측된 비대칭이고,
     저는 이걸 이 연구의 과학적 발견으로 봅니다.
 (3) 그리고 성능 상한을 정하는 건 모델이 아니라 데이터입니다. 2만 파라미터 모델이
     35만 파라미터 백본과 같고, 폭을 26배 키워도 평평합니다."

※ 시간이 밀리면 이 슬라이드를 30초로 줄이고 뒤에서 회수.
""")


# --- 3. Background: diagnostics + the missingness problem + the question ---
def t_background():
    s = slide()
    header(s, "1. 배경 & 문제", "CES는 왜, 얼마나 비는가 — 그리고 무엇으로 채우나")
    text(s, Inches(0.55), Inches(1.38), Inches(12.3), Inches(0.60),
         [[("CES는 pedestal 물리의 핵심량 Tᵢ · V_rot를 주지만 ", 14, DARK, False, False, None),
           ("광자 적분이 필요해 느리고 자주 빈다", 14, ORANGE, True, False, None),
           (". 반면 빠른 진단은 같은 10 ms 격자에서 결측 없이 측정된다.", 14, DARK, False, False, None)]])
    # left: what each diagnostic measures
    card(s, Inches(0.55), Inches(1.98), Inches(6.0), Inches(3.35),
         "진단 구성 — 같은 10 ms 격자", [
             "CES (타겟) — Tᵢ · V_rot, 느리고 자주 결측",
             "BES 9 ch — 밀도요동 nₑ 공간구조 → Tᵢ 단서",
             "ECEI 4 ch — 전자온도 Tₑ 2D 영상 → Tᵢ 단서",
             "   (물리 경로: 충돌 e–i 결합)",
             "Mirnov 2 ch — kHz dB/dt를 100 Hz로 데시메이트",
             "   lag-1 자기상관 -0.009 (BES +0.568)",
             "→ 빠른 진단은 격자에서 항상 100% 측정된다",
         ], accent=BLUE, title_size=15, body_size=12.5)
    # right: the measured missingness
    box(s, Inches(6.75), Inches(1.98), Inches(4.55), Inches(3.35), fill=NAVY, round_=True)
    text(s, Inches(7.0), Inches(2.10), Inches(4.1), Inches(0.4),
         [[("결측 실태 (641 방전 · 247,207행)", 14.5, ORANGE, True, False, None)]])
    for i, (lab, pct, col) in enumerate([("CES_TI 완전결측(NaN)", "8.2 %", RGBColor(0x9D, 0xE8, 0xCD)),
                                         ("CES_VT 완전결측(NaN)", "23.9 %", ORANGE)]):
        xx = 7.0 + i * 2.05
        text(s, Inches(xx), Inches(2.58), Inches(2.0), Inches(0.34),
             [[(lab, 11, LGRAY, False, False, None)]])
        text(s, Inches(xx), Inches(2.86), Inches(2.0), Inches(0.56),
             [[(pct, 26, col, True, False, None)]])
    box(s, Inches(7.0), Inches(3.44), Inches(4.05), Inches(0.78),
        fill=RGBColor(0x8E, 0x2B, 0x22), round_=True)
    text(s, Inches(7.18), Inches(3.52), Inches(3.75), Inches(0.66),
         [[("+ held(직전값 복사) CES_VT 41.1 %", 12, RGBColor(0xFF, 0xD5, 0xCE), True, False, None)],
          [("→ V_rot 실질 무정보 65.0 %", 13.5, WHITE, True, False, None)]],
         line_spacing=1.12, space_after=0)
    text(s, Inches(7.0), Inches(4.32), Inches(4.05), Inches(0.95),
         [[("· 두 타겟이 ", 11.5, WHITE, False, False, None),
           ("독립적으로", 11.5, ORANGE, True, False, None),
           (" 결측 → 타겟별 처리 필요", 11.5, WHITE, False, False, None)],
          [("· 결측은 저 SNR · ELM · 천이에 몰린다", 11.5, WHITE, False, False, None)],
          [("  → MNAR: 관측점 skill은 낙관적 상한", 11.5, LGRAY, False, False, None)]],
         line_spacing=1.15, space_after=2)
    # right edge: raw shot CSV — blanks in CES_TI, a repeating value in CES_VT
    add_image_fit(s, os.path.join(FIG, "fig_raw_csv_missing.png"),
                  Inches(11.45), Inches(1.99), Inches(1.75), Inches(3.3))
    # bottom: the research question
    box(s, Inches(0.55), Inches(5.45), Inches(12.25), Inches(1.42), fill=NAVY, round_=True)
    text(s, Inches(0.85), Inches(5.56), Inches(11.7), Inches(1.25),
         [[("연구 질문", 13, ORANGE, True, False, None)],
          [("CES가 결측된 10 ms 시점에서, 동시각 빠른 진단 + 과거 CES 이력만으로 "
            "CES 자체의 시간 보간이 복원할 수 없는 정보를 회복할 수 있는가?", 15, WHITE, True, False, None)],
          [("핵심 비대칭(미리보기): 빠른 진단은 Tᵢ 정보는 운반하지만 V_rot 정보는 거의 운반하지 않는다 — 결과에서 그대로 확인된다.",
            12, LGRAY, False, False, None)]], line_spacing=1.14, space_after=3)
    return note(s, """
⏱ 01:20–02:20  (60초)

"CES는 이온온도와 토로이달 회전이라는, pedestal 물리에서 가장 중요한 두 양을 줍니다.
 그런데 충분한 신호대잡음비를 얻으려면 광자를 오래 모아야 해서 느리고, 자주 빕니다.
 같은 10 ms 격자에서 Tᵢ는 8.2%, 회전은 23.9%가 값 자체가 비어 있습니다.
 회전은 여기서 끝이 아닙니다. 값이 채워진 행 중에서도 54%가 직전 값을 그대로 복사한
 held 값입니다 — 진짜 측정이 아닙니다. NaN과 held를 합치면 회전은 전체 행의 65%가
 실질적으로 정보가 없습니다. Tᵢ의 held는 22만 6천 행 중 단 1행, 사실상 0%입니다.
 그리고 두 타겟은 서로 독립적으로 빕니다 — 한쪽만 관측된 행이 많다는 뜻입니다.

 반대로 BES, ECEI, Mirnov 같은 빠른 진단은 같은 격자에서 100% 다 있습니다.
 그래서 질문은 자연스럽습니다: 항상 있는 것으로 자주 비는 것을 채울 수 있는가.
 정확히는, CES 자신의 시간 보간이 복원할 수 없는 정보를 회복할 수 있는가입니다."

물리적 연결고리 (질문 대비):
  · ECEI(Tₑ) + BES(nₑ) → 충돌 e–i 결합을 통해 Tᵢ와 물리적으로 연결
  · 회전은 주로 NBI 토크가 구동 → 그 토크가 데이터에 없음 = 근본적 정보 부재
※ 결측이 무작위가 아니라는 점(MNAR)은 여기서 한 번 심어두고, 결과 ④에서 정량화한다.
""")


# --- 8. Data pipeline + contract (merged) --------------------------------
def t_pipeline():
    s = slide()
    header(s, "3. 데이터 & 모델", "데이터 계약 — No-Fake-Data · held 전면 제거 · 누수 삼중 차단")
    text(s, Inches(0.55), Inches(1.36), Inches(12.3), Inches(0.72),
         [[("641 방전(shot 30801–32751) · 10 ms 공통 격자 · 총 247,207행. 세그먼트는 ≥0.5 s 간극에서 분리되고 "
            "전형적 파일은 주 세그먼트 1개(중앙값 301행 ≈ 3.0 s) — 모델 입력도 보간도 이 경계를 넘지 않는다. ",
            13, DARK, False, False, None),
           ("TEST(seed 42, 컷): Tᵢ 32,589행 / 96 방전 · V_rot 10,463행 / 60 방전 — 선택이 끝날 때까지 봉인.",
            13, NAVY, True, False, None)]], line_spacing=1.14)
    cards = [
        ("① 가짜 라벨 금지 (No Fake Data)", ORANGE,
         ["학습 행을 만들려고 타겟을 대체(impute)하지 않음",
          "윈도: 진단 입력 완전 + 타겟 ≥1개 관측된 행만 사용",
          "시퀀스: 라벨 없는 행은 맥락으로만 기여",
          "어느 프레이밍도 타겟 행 자신의 값을 읽지 않는다"]),
        ("② 타겟별 masked loss", BLUE,
         ["L = Σ m·(예측 - 실측)² / Σ m   (m = 타겟별 관측 마스크)",
          "한쪽 타겟만 관측된 행도 그 타겟은 학습에 기여",
          "두-타겟-필수 필터는 라벨 행의 ≈28%를 조용히 버림",
          "→ 이를 제거한 것은 순수한 데이터 이득"]),
        ("③ 누수 삼중 차단", TEAL,
         ["파일(shot) 단위 분할 — 인접 행 자기상관 차단",
          "학습 파일 전용 정규화 (희소 타겟은 NaN-인지)",
          "시퀀스 모델은 여기에 shot별 입력 표준화를 더함",
          "타겟 시점의 값·관측 flag는 입력에 결코 안 들어감"]),
        ("④ held 전면 제거", NAVY,
         ["관측 V_rot의 54%가 계측기 유지값 (499/641 파일)",
          "지도 타겟·이력 입력·정규화 통계·보간 앵커에서 동일 제거",
          "대가 = PR2 폴백률 Tᵢ 0.3–0.4% · V_rot 40–44%",
          "→ 어떤 arm도 forward-fill로 공짜 점수를 못 받음"]),
    ]
    for i, (t, col, lines) in enumerate(cards):
        r, c = divmod(i, 2)
        card(s, Inches(0.55 + c * 6.2), Inches(2.20 + r * 2.35), Inches(6.0), Inches(2.2),
             t, lines, accent=col, title_size=14.5, body_size=12)
    return note(s, """
⏱ 05:00–05:50  (50초)

"데이터는 641개 방전, 10 ms 격자로 24만 7천 행입니다. 파일 하나는 대개 3초짜리
 측정 세그먼트 하나이고, 모델도 보간도 그 경계를 넘지 않습니다 — 정보 조건을
 양쪽에 똑같이 맞춘 겁니다.

 데이터 계약은 네 가지입니다.
 첫째, 가짜 라벨을 만들지 않았습니다. 결측을 보간으로 메워 학습에 쓰면
 '보간을 이기는가'라는 질문 자체가 무의미해집니다.
 둘째, 두 타겟이 독립적으로 비기 때문에 손실을 타겟별로 마스킹했습니다.
 이걸 안 하면 한쪽만 관측된 행 — 라벨의 약 28% — 을 통째로 버리게 됩니다.
 셋째, 누수를 세 곳에서 막았습니다. 행이 아니라 shot 파일 단위 분할, 학습 파일
 전용 정규화, 그리고 맞히려는 시점의 값과 관측 플래그를 입력에서 완전히 지우는 것.
 넷째, held 값을 전부 뺐습니다. 지도 타겟에서도, 이력 입력에서도, 정규화 통계에서도,
 그리고 모든 기준선의 보간 앵커에서도요. 대가가 있습니다 — 회전의 채점 행 40~44%는
 미래 이웃이 없어서 보간이 persistence로 후퇴합니다. 그 폴백률까지 사전등록으로
 보고하게 해뒀습니다."

※ 질문 대비: TEST는 아키텍처 탐색 시작 전에 예약했고 선택 중엔 열지 않았다.
""")


# --- 12. Evaluation methodology, part 2 ----------------------------------
def t_eval2():
    s = slide()
    header(s, "4. 평가 방법론", "shot 군집 paired bootstrap · 두 모집단 규칙 · 모델 선택")
    bullets(s, Inches(0.55), Inches(1.5), Inches(6.5), Inches(3.0), [
        ("한 방전 안의 인접 CES 행은 강하게 상관됨", 0),
        ("개별 샘플을 독립으로 보면 불확실성을 크게 과소평가", 1, RED, True),
        ("PR4 검정: 샘플별 짝지은 오차 (SE_model - SE_pchip)를", 0),
        ("shot 단위로 묶고 shot 전체를 복원추출 (B = 10,000)", 1),
        ("95% CI가 0을 제외하면 PASS", 0, GREEN, True),
        ("→ '새로운 방전에서도 이길까'에 답하는 CI", 1),
        ("유효 표본 = 방전 수 (Tᵢ ≈96 · V_rot 60–66) = 검정력 상한", 0),
        ("모델 대 모델 비교도 같은 행 위에서 같은 paired bootstrap", 0, NAVY, True),
    ], size=13, gap=7)
    box(s, Inches(7.35), Inches(1.5), Inches(5.45), Inches(3.0), fill=NAVY, round_=True)
    text(s, Inches(7.6), Inches(1.62), Inches(5.0), Inches(1.35),
         [[("Murphy (1988) skill score", 13.5, ORANGE, True, False, None)],
          [("skill = 1 - MSE_model / MSE_baseline", 15, WHITE, True, False, MONO)],
          [("> 0 모델 우수 · = 0 동률. 오차는 물리 단위로 역정규화해 타겟별로 계산.",
            12, LGRAY, False, False, None)]], line_spacing=1.2)
    box(s, Inches(7.6), Inches(3.02), Inches(4.95), Pt(2), fill=RGBColor(0x2A, 0x47, 0x6E))
    text(s, Inches(7.6), Inches(3.16), Inches(5.0), Inches(1.25),
         [[("두 공동 1차 모집단", 13.5, ORANGE, True, False, None)],
          [("컷 = Tᵢ > 3 keV(피팅 실패)를 결측 처리 · 포함 = 컷 없음. 전 arm에 동일 적용.",
            12, WHITE, False, False, None)],
          [("무조건부 주장은 두 모집단 모두에서 성립할 때만 한다.", 12.5, WHITE, True, False, None)]],
         line_spacing=1.2)
    box(s, Inches(0.55), Inches(4.70), Inches(12.25), Inches(2.0), fill=CARDBG, round_=True)
    text(s, Inches(0.85), Inches(4.82), Inches(11.7), Inches(0.45),
         [[("모델 선택 프로토콜 — 규칙을 수치보다 먼저 적는다", 14.5, NAVY, True, False, None)]])
    bullets(s, Inches(0.85), Inches(5.30), Inches(11.7), Inches(1.35), [
        ("백본 관문: 4조건(4 분할 부호 유지 · 통합 CI 0 제외 · 예산 균등화 · V_rot 손실 없음)을 먼저 고정하고 그다음 충족", 0),
        ("유일한 아키텍처 후보(seq_v2 + 관측마스킹 인과 attention): 4/4 양수(+0.009/+0.013/+0.033/+0.020)지만 유의 1/4 → 미승격", 0),
        ("val에선 2/2 유의였다 — 승격 bar를 TEST에 두는 이유. 스윕 위에서 백본을 재선정하는 것은 구성상 금지.", 1, RED, True),
        ("사다리 칸·폭 스윕의 판정 규칙도 TEST 채점 전에 문서화 — TEST는 결정마다 단 한 번만 채점된다", 0),
    ], size=12.5, gap=6)
    return note(s, """
⏱ 09:00–09:50  (50초)

"신뢰구간을 행이 아니라 방전 단위로 계산했습니다. 한 방전 안의 10 ms 간격 측정들은
 거의 복사본입니다. 그걸 3만 개의 독립 증거처럼 세면 확신이 과장됩니다. 그래서
 짝지은 오차를 shot으로 묶고, shot을 통째로 만 번 재추출했습니다. 우리에게 불리한
 계산이지만 '새로운 방전에서도 이길까'라는 진짜 질문에 답하는 방식입니다.
 그래서 검정력의 상한은 샘플 수가 아니라 방전 수입니다 — Tᵢ 96개, 회전 60~66개.

 오른쪽 아래가 두 모집단입니다. Tᵢ 관측값의 0.53%가 3 keV를 넘는데, 이건 플라즈마가
 아니라 실패한 스펙트럼 피팅입니다. 빼면 '어려운 행을 없앴다', 두면 '스파이크가
 보간 앵커를 오염시킨다' — 어느 쪽도 비판을 피할 수 없어서 둘 다 공동 1차로 사전등록했고,
 무조건부 주장은 양쪽에서 성립할 때만 합니다.

 아래는 모델 선택입니다. 백본을 바꾸는 관문의 네 조건을 먼저 못박고 그다음 충족시켰고,
 그 뒤 유일한 아키텍처 후보는 4개 분할 전부 양수였는데도 유의가 1개뿐이라 승격하지
 않았습니다. val에서는 2/2 유의였다는 게, 승격 기준을 TEST에 두는 이유입니다."

※ 시간이 밀리면 아래 밴드(모델 선택)만 한 줄로 요약하고 넘어갈 것.
""")


# --- 17. Result 4: two stress tests (MNAR + campaign) --------------------
def t_stress():
    s = slide()
    header(s, "5. 결과 ④", "스트레스 2종 — 실제 결측점 재가중(MNAR)과 캠페인(시간) 분할",
           accent=ORANGE)
    text(s, Inches(0.55), Inches(1.36), Inches(12.3), Inches(0.5),
         [[("관측점 skill은 낙관적 상한이고, 무작위 분할은 시간 이동을 검사하지 않는다. "
            "배치 주장을 가르는 두 스트레스를 사전에 정해두고 통과 여부를 본다.",
            13, DARK, False, False, None)]], line_spacing=1.12)
    card(s, Inches(0.55), Inches(1.95), Inches(6.0), Inches(2.5),
         "① 실제 결측점으로 재가중 (MNAR)", [
             "층 = Δt(15/25/45 ms) × 입력만의 활동 flag,",
             "  결측 행의 층 분포로 채점 지점을 재가중",
             "도달: 결측 Tᵢ의 54–68%가 in-domain · V_rot은 4–6%",
             "  → 재가중 V_rot은 결론 없음(결측 질량의 1/20)",
             "Tᵢ vs persistence: 컷 4/4 · 포함 4/4 (+0.28~+0.44)",
             "Tᵢ vs PCHIP: 컷 2/4 · 포함 4/4 (점추정 +0.14~+0.28)",
         ], accent=ORANGE, title_size=14.5, body_size=12)
    card(s, Inches(6.8), Inches(1.95), Inches(6.0), Inches(2.5),
         "② 캠페인(시간) 분할 — shot 번호로 자름", [
             "train 416 (30801–31991) / val 128 (32002–32310) /",
             "  test 97 (32312–32751) · 초기화 seed 4개",
             "윈도 대조군: 컷 2/4 · 포함 0/4 · 인과 GP 0/4 (붕괴)",
             "seq_v2 컷 +0.187/+0.174/+0.181/+0.177 → 4/4",
             "seq_v2 포함 +0.173/+0.202/+0.198/+0.184 → 4/4",
             "원인은 측정: 드리프트 BES 1.22σ·ECEI 0.53σ vs 타겟 0.115σ",
         ], accent=BLUE, title_size=14.5, body_size=12)
    cw = [Inches(4.35), Inches(4.0), Inches(3.9)]
    rows = [
        [("무작위 분할 · 관측점 (결과 ②)", DARK, True, None),
         ("4/4 · 4/4,  +0.17~+0.32", GREEN, True, None),
         ("4/4 · 4/4 vs 인과 GP,  +0.08~+0.17", GREEN, True, None)],
        [("결측점 재가중 (MNAR)", DARK, True, None),
         ("2/4 · 4/4,  +0.14~+0.28", ORANGE, True, None),
         ("4/4 · 4/4 vs persistence,  +0.28~+0.44", GREEN, True, None)],
        [("캠페인 시간 분할", DARK, True, None),
         ("4/4 · 4/4,  +0.17~+0.20", GREEN, True, None),
         ("4/4 · 4/4 vs 인과 GP,  +0.11~+0.16", GREEN, True, None)],
    ]
    table(s, Inches(0.55), Inches(4.78), cw,
          ["Tᵢ 평가 (컷 · 포함)", "vs PCHIP (오프라인)", "vs 인과 기준선"], rows,
          row_h=Inches(0.44), head_h=Inches(0.42), size=12, head_size=12)
    text(s, Inches(0.55), Inches(6.54), Inches(12.3), Inches(0.40),
         [[("진술: ", 12, NAVY, True, False, None),
           ("실제 결측 시점에서 나우캐스터는 모든 인과 CES-only 방법보다 유의하게 낫고, "
            "오프라인 보간보다는 모집단 조건부로 낫다.", 12, DARK, False, False, None)]],
         line_spacing=1.12)
    return note(s, """
⏱ 12:30–13:40  (70초)  ★ 배치 주장을 가르는 슬라이드

"관측된 지점에서만 채점하면 낙관적입니다 — 결측은 하필 어려운 순간에 몰리니까요.
 그래서 결측 행의 층 분포로 채점 지점을 재가중했습니다. 왼쪽입니다.
 온라인 시스템이 실제로 경쟁하는 상대인 persistence 대비로는 두 모집단 4개 분할 전부
 살아남습니다. 오프라인 보간 대비로는 점추정은 유지되지만 컷 모집단에서 2개 분할의
 신뢰구간이 0을 지납니다 — 그래서 모집단 조건부라고 정직하게 씁니다.
 한 가지 한계는 명확히 말씀드립니다. 재가중이 도달하는 범위가 Tᵢ는 결측의 54~68%인데
 회전은 4~6%뿐입니다. 그래서 재가중 회전은 결론을 내지 않습니다.

 오른쪽은 시간 분할입니다. 방전 번호로 잘라서 과거로 학습하고 미래를 맞힙니다.
 옛 윈도 모델은 여기서 오프라인 우위를 완전히 잃었습니다 — 컷 2/4, 포함 0/4.
 그런데 시퀀스 백본은 4/4에 4/4로 버팁니다. 원인은 추측이 아니라 측정했습니다:
 빠른 진단의 드리프트가 1.22σ인데 타겟은 0.115σ입니다. 학습 파일 전용 정규화가
 무작위 분할에선 옳지만 캠페인 이동에서 깨지는 겁니다. 백본은 정의상 shot별
 표준화를 하기 때문에 그 함정을 피합니다."

※ 남는 주의(먼저 말할 것): 캠페인은 한 시간 블록 위의 초기화 4개이지 분할 4개가 아니다.
""")


# --- 22. Limitations + future + the 2026-08-16 decision ------------------
def t_limits():
    s = slide()
    header(s, "7. 한계 & 향후", "무엇을 인정하고, 다음에 무엇을 하는가")
    box(s, Inches(0.55), Inches(1.5), Inches(6.0), Inches(3.9), fill=CARDBG, round_=True)
    box(s, Inches(0.55), Inches(1.5), Inches(0.12), Inches(3.9), fill=RED)
    text(s, Inches(0.8), Inches(1.62), Inches(5.5), Inches(0.45),
         [[("한계 — 논문이 먼저 인정하는 것", 15, RED, True, False, None)]])
    bullets(s, Inches(0.8), Inches(2.12), Inches(5.55), Inches(3.1), [
        ("검정력: test 방전 96(Tᵢ) / 60–66(V_rot)이 모든 유의성의 구속조건", 0),
        ("포함 모집단에선 ≈1% 행이 SSE의 70–83%를 담는다", 1),
        ("MNAR 낙관: 재가중 도달은 Tᵢ 54–68% · V_rot 4–6%뿐", 0),
        ("오프라인 주장의 상한은 GP 동률 (1/8 유의)", 0),
        ("값 컷은 일방향 프록시 — V_rot 스파이크는 남는다", 0),
        ("캠페인은 한 시간 블록 위 초기화 4개 · 컷 run 2/4 상한 종료", 0),
        ("conformal은 marginal coverage · 지연은 네트워크만 측정", 0),
        ("단일 장치 · 두 순환 계열 · 지표 비대칭", 1),
        ("범위: 페데스탈 상단 프레이밍 — 이벤트-위상 분석은 후속", 0),
    ], size=12, gap=7)
    box(s, Inches(6.8), Inches(1.5), Inches(6.0), Inches(3.9), fill=CARDBG, round_=True)
    box(s, Inches(6.8), Inches(1.5), Inches(0.12), Inches(3.9), fill=TEAL)
    text(s, Inches(7.05), Inches(1.62), Inches(5.5), Inches(0.45),
         [[("향후 — 남은 레버는 전부 데이터다", 15, TEAL, True, False, None)]])
    bullets(s, Inches(7.05), Inches(2.12), Inches(5.55), Inches(3.1), [
        ("음성 결과는 그것을 뒤집을 측정을 지목할 때만 결론이 된다", 0, NAVY, True),
        ("① CES 피팅 품질 메타데이터 (χ² · 신호 수준)", 0, ORANGE, True),
        ("값 컷을 품질 컷으로 대체하면 두 모집단이 하나로 합쳐진다", 1),
        ("② 원본 kHz Mirnov 특징 — V_rot 최상위 레버", 0, ORANGE, True),
        ("윈도 RMS · 대역 파워 · 모드 수 · 모드 회전 주파수", 1),
        ("lag-1 자기상관 BES +0.568 / ECEI +0.572 vs Mirnov -0.009", 1),
        ("아카이브 데이터로 검정 가능 (파일럿→확대 사전등록)", 1),
        ("③ NBI 토크 채널 확보 — 회전의 원인 변수", 0, ORANGE, True),
        ("Tₑ~Tᵢ r = +0.353 vs Tₑ~V_rot r = +0.024 → power ≠ torque", 1),
        ("크기 축은 닫혔다: 21k = 백본(컷) · 폭 26× 평평", 0, GRAY, True),
    ], size=12, gap=7)
    box(s, Inches(0.55), Inches(5.56), Inches(12.25), Inches(0.94), fill=NAVY, round_=True)
    text(s, Inches(0.85), Inches(5.65), Inches(11.7), Inches(0.82),
         [[("결정 기록 (2026-08-16)", 12.5, ORANGE, True, False, None)],
          [("① 두 모집단 공동 1차 유지 ② V_rot 프로토콜 불변(재학습 없음, anchored 비교엔 SSE 비중 병기) "
            "③ B.6 kHz Mirnov 미도착 — 대기.", 12.5, WHITE, False, False, None)]],
         line_spacing=1.14, space_after=2)
    return note(s, """
⏱ 18:10–19:00  (50초)

"한계를 분명히 말씀드립니다. 가장 큰 제약은 검정력입니다 — test 방전이 Tᵢ 96개,
 회전은 60~66개뿐이고, 이게 모든 유의성 판정을 구속합니다. 채점은 관측 지점에서
 하므로 낙관적 상한이고, 재가중이 닿는 범위도 Tᵢ 54~68%, 회전은 4~6%뿐입니다.
 오프라인 주장의 상한도 명시합니다 — 가장 강한 오프라인 평활기인 GP와는 동률입니다.

 향후는 전부 데이터 쪽입니다. 저희 원칙이 '음성 결과는 그것을 뒤집을 측정을 지목할
 때만 결론이 된다'인데, 셋을 지목했습니다. 하나, CES 피팅 품질 메타데이터를 받으면
 값 컷을 품질 컷으로 바꿔 두 모집단을 하나로 합칠 수 있습니다. 둘, 원본 kHz Mirnov에서
 모드 회전 주파수를 뽑는 것 — 회전의 최상위 레버입니다. 셋, NBI 토크 채널입니다.
 모델링이 아니라 데이터 획득 과제이고, 아카이브 데이터로 검정할 수 있습니다.
 모델 크기 축은 이미 닫혔습니다."

※ 질문 대비: Mirnov 파생 특징 재가공(적분·|MC|·이동 RMS)은 이미 시도했고 개선이 없었다.
   이미 잃은 정보는 하류에서 복구되지 않는다 — 그래서 전처리(원본 kHz)로 올라가야 한다.
""")


# ============================== build =====================================
def build():
    try:  # slide text is Korean; the console may be cp949
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    # ---- 도입 (1–2) -----------------------------------------------------
    t_title()
    t_overview()

    # ---- 1. 배경 & 문제 (3–5) -------------------------------------------
    t_background()
    note(kicker(B.s_missing_table(), "1. 배경 & 문제"), """
⏱ 02:20–03:00  (40초)

"결측을 퍼센트가 아니라 전수 집계로 보여드립니다. 641 shot, 24만 7천 행 전부입니다.

 위 두 줄이 핵심입니다. 값이 비어 있는 NaN은 Tᵢ 8.2%, 회전 23.9%입니다.
 그런데 그 아래, 직전 관측값과 비트 단위로 똑같은 held 행이 회전은 41.1%입니다.
 심하면 1,214행이 연속으로 같은 값입니다. 회전 진단의 실제 측정 주기가 행 주기보다
 느려서 값이 carry-forward된 겁니다 — 독립적인 측정이 아닙니다.
 둘을 합치면 회전은 65%가 실질 무정보이고, 진짜 정보가 있는 행은 35%뿐입니다.
 Tᵢ는 held가 22만 6,991행 중 1행이라 사실상 무관합니다.

 왜 중요하냐면, held 행은 persistence나 보간이 오차 거의 0으로 맞히는 '공짜 정답'입니다.
 그래서 확정 프로토콜에서는 학습 타겟에서도, 이력 입력에서도, 정규화 통계에서도,
 모든 기준선의 보간 앵커에서도 전부 뺐습니다. 진짜 측정만 채점합니다."

※ 이 슬라이드는 신뢰를 크게 얻는 지점 — 방어적으로 말하지 말고 "찾아서 고쳤다"는 톤으로.
""")
    note(kicker(B.s_two_populations(), "1. 배경 & 문제"), """
⏱ 03:00–03:45  (45초)

"두 번째 감사입니다. 관측된 Tᵢ의 p99가 2,089 eV인데 최댓값은 14,984 eV입니다.
 이 먼 꼬리는 플라즈마가 아니라 실패한 스펙트럼 피팅입니다. 3 keV를 넘는 행이
 1,197행, 0.53%인데, 85%가 단일 행이고 정점이 이웃 평균의 13배입니다.
 어떤 방법으로도 예측할 수 없고, 보간의 앵커를 오염시킵니다.

 여기서 정직한 문제가 생깁니다. 빼면 '어려운 행을 없앴다'는 비판을 받고,
 두면 스파이크 앵커가 오프라인 기준선에 핸디캡을 줍니다. 어느 쪽도 안전하지 않습니다.
 그래서 두 대응을 모두 공동 1차 모집단으로 사전등록했습니다 — 컷과 포함.
 규칙은 하나입니다: 무조건부 주장은 두 모집단 모두에서 성립할 때만 한다.
 문턱이 자의적이라는 지적도 검사했는데, 2.5·3·4 keV 전부 결과가 같습니다."

※ 질문 대비: 값 컷은 일방향 프록시다 — 하향 dip 4,965행은 손대지 않았고, 상향 이상치의
   19%만 제거된다. 회전 스파이크는 컷하지 않고 SSE 비중을 병기한다.
""")

    # ---- 2. 접근법 (6) ---------------------------------------------------
    s_bar = B.s_bar()
    # the 1-hour wording wraps a lone period onto a second line at this width
    set_run(s_bar, "과 비교한다 — 이 보간들은",
            "과 비교한다 — 타겟 주변의 과거+미래 CES를 모두 사용한다.")
    note(s_bar, """
⏱ 03:45–05:00  (75초)  ★ 이 발표에서 가장 중요한 슬라이드

"평가 기준선을 어떻게 잡았는지가 이 연구의 핵심입니다.

 흔한 방식은 persistence, 즉 직전 값 유지와 비교하는 겁니다. 그건 너무 쉽습니다.
 그래서 일부러 훨씬 어려운 상대를 골랐습니다 — CES만 쓰는 오프라인 보간,
 선형·PCHIP·국소 AR·GP입니다.

 여기서 정보 비대칭을 보십시오. 오른쪽 보간은 맞히려는 시점의 과거와 미래 CES를
 모두 씁니다. 반면 저희 모델은 그 시점까지의 빠른 진단과 세그먼트 과거 이력만 쓰고
 미래는 전혀 보지 않습니다. 엄격히 causal입니다.
 그래서 미래를 보는 보간을 과거만 보는 모델이 이긴다면, 그건 운이 아니라 빠른 진단이
 시간 보간으로는 얻을 수 없는 CES 정보를 실제로 운반한다는 강력한 증거가 됩니다.

 이번 개정에서 팔을 하나 더 넣었습니다 — 인과 GP입니다. 같은 GP를 과거 이웃 16개로만
 제한한 건데, 실제로 배치할 수 있는 방법 중 가장 강한 경쟁자입니다. persistence는
 이기기 쉬우니까, '배치 가능한 모든 인과 방법을 이긴다'는 주장을 persistence가 아니라
 이 인과 GP로 판정합니다."

※ 청중이 "왜 굳이 불리하게?"라고 생각할 수 있으니 마지막 두 문장을 강조.
""")

    # ---- 3. 데이터 & 모델 (7–10) ----------------------------------------
    t_pipeline()
    note(kicker(B.s_samples(), "3. 데이터 & 모델"), """
⏱ 05:50–06:35  (45초)

"학습 예제를 만드는 방식이 두 가지입니다. 이 대조가 결과 ③의 핵심입니다.

 하나는 윈도 프레이밍입니다. 맞히려는 시점 앞 두 행을 잘라서 텐서로 만듭니다.
 이게 옛 주 모델의 방식이고 지금은 대조군입니다.
 다른 하나는 전체 격자 시퀀스입니다. 세그먼트 안에서 입력이 온전한 행은 라벨이
 없어도 전부 맥락으로 유지하고, 희소성은 loss 마스킹으로 처리합니다.

 결정적 차이는 도달거리입니다. 윈도는 과거 관측 W-1개만 보고, 시퀀스는 세그먼트
 전체를 봅니다. 그리고 시퀀스에서는 W가 더 이상 하이퍼파라미터가 아닙니다."

※ 질문 대비: 윈도의 시간 특징 4채널(lookback·간격·각 log1p)은 불규칙한 관측 간격을
   명시적으로 노출하기 위한 것. 과거 값의 신뢰도는 10 ms 전인지 200 ms 전인지에 달렸다.
""")
    note(kicker(B.s_arch(), "3. 데이터 & 모델"), """
⏱ 06:35–07:35  (60초)

"주 모델입니다. 22채널 격자 시퀀스 위에 독립적인 인과 LSTM 두 개를 얹었습니다.

 위쪽 Tᵢ 분기는 2층 160으로 전체 상태를 다 읽습니다 — 빠른 진단, 두 타겟의 이월값,
 신선도, 시간 간격까지요.
 아래쪽 V_rot 분기는 1층 64이고, 빠른 진단이 아닌 7채널만 읽습니다.
 여기가 중요합니다. 라우팅을 head가 아니라 인코더에서 했습니다. 순환 상태를 공유하면
 head를 어떻게 배선해도 빠른 진단 정보가 회전 쪽으로 샙니다. 분기 자체를 분리했기
 때문에, 빠른 채널 15개를 전부 섭동해도 회전 출력이 비트 단위로 동일합니다.
 이게 뒤에서 나오는 비대칭 결론을 구조적으로 보증합니다.

 희소성은 loss가 처리합니다 — 세그먼트의 라벨 있는 모든 행에 대한 타겟별 masked MSE.
 전체 35만 7천 파라미터이고, 학습은 AdamW에 조기 종료로 14~25 에폭에서 끝납니다."

※ 질문 대비: 라벨 없는 행을 왜 남기나? → 빠른 진단은 그 행에서도 조밀하게 관측된다.
   버리면 인과 문맥이 끊긴다.
""")
    note(kicker(B.s_arch_window(), "3. 데이터 & 모델"), """
⏱ 07:35–08:10  (35초)

"짝지은 대조군이 옛 주 모델입니다. 진단별 시간 인지 1D CNN에 양방향 GRU 이력
 인코더, 그 위에 관측 마스킹 attention pooling을 얹은 20만 파라미터 모델입니다.
 attention이 해당 타겟이 실제 관측된 행에만 질량을 허용합니다 — 보간의 귀납 편향을
 파라미터 0개로 이식한 겁니다. 이 구조는 40여 회 keep/discard 통제 실험의 산물입니다.

 지금 이 모델의 역할은 셋입니다. 백본 관문의 비교 대상, 절제 실험의 무대,
 그리고 캠페인 붕괴의 재현자입니다. 데이터 계약·held 처리·분할·채점 모집단이
 백본과 완전히 같아서 행 단위로 짝지어 비교할 수 있습니다."

※ 시간이 밀리면 30초로 줄이고 "옛 주 모델이 지금은 대조군"만 전달.
""")

    # ---- 4. 평가 방법론 (11–12) ------------------------------------------
    note(kicker(B.s_methodology(), "4. 평가 방법론"), """
⏱ 08:10–09:00  (50초)  ★ 통계 질문이 나오는 구간 — 여기서 신뢰를 확보

"결과를 보여드리기 전에, 이 숫자를 믿어도 되는 이유를 말씀드립니다.

 왼쪽이 TEST 동결입니다. test 방전은 아키텍처 탐색을 시작하기 전에 떼어놓고
 선택이 끝날 때까지 한 번도 열지 않았습니다. 모델 선택은 전부 val에서만 했습니다.
 그래서 헤드라인 숫자에 winner's curse가 없습니다.

 오른쪽이 사전등록입니다. 결과를 보기 전에 문서로 못박은 것들입니다.
 비교 상대는 PCHIP으로 확정, 보간이 예측 못 하는 지점은 persistence로 채점하되
 폴백률을 반드시 보고, test 최소 규모, 그리고 '이겼다'는 신뢰구간이 0을 제외할 때만.
 여기에 이번 개정에서 held-free, W=2, 파일당 500, 두 모집단, TEST 채점 전 결정 규칙
 커밋이 추가됐습니다.

 아래가 기준선 사다리입니다. 위 세 개가 인과, 아래 세 개가 미래를 보는 오프라인입니다.
 인과 GP가 배치 가능한 최강 상대입니다."

※ 예상 질문 "왜 PCHIP인가?" → 단조성을 보존해 overshoot이 적은 보수적(=강한) 기준선이고,
   타겟 양쪽의 과거+미래를 다 본다. 즉 우리에게 불리하게 설계된 기준선.
""")
    t_eval2()

    # ---- 5. 결과 ①~⑧ (13–20) -------------------------------------------
    note(kicker(B.s_res_ladder(), "5. 결과 ①"), """
⏱ 09:50–10:30  (40초)

"먼저 가장 단순한 결과입니다. RMSE 사다리이고, 낮을수록 좋습니다.

 두 타겟 모두 백본이 맨 아래, 즉 오차가 가장 작습니다. persistence나 국소 AR 같은
 인과 기준선은 큰 마진으로 이깁니다. 그리고 배치 가능한 최강 상대인 인과 GP보다도
 Tᵢ는 4%, 회전은 18% 낮습니다.
 미래까지 보는 오프라인 GP와는 153.8 대 157.8로 사실상 동률입니다 — 이게 오프라인
 주장의 상한이고, 저희는 이걸 숨기지 않고 같이 보고합니다.

 포함 모집단에서는 스파이크 때문에 Tᵢ RMSE가 두 배 이상 커지지만 순서는 그대로입니다."

※ 다음 슬라이드로 넘기는 말: "point estimate가 앞선다고 이겼다고 말하면 안 됩니다.
   그래서 신뢰구간을 봅니다."
""")
    s_forest = kicker(B.s_res_forest(), "5. 결과 ②")
    # cross-reference fix: the scaling slide is 결과 ⑨ in the 1-hour deck, ⑥ here.
    set_para(s_forest, "결과 ⑨에서 분해",
             "  모든 arm이 PCHIP 대비 더 좋아 보인다 → 결과 ⑥에서 분해")
    note(s_forest, """
⏱ 10:30–11:40  (70초)  ★ 헤드라인 결과

"이게 헤드라인입니다. forest plot이고 가로축이 PCHIP 대비 skill입니다.
 점이 추정치, 가로 막대가 shot 군집 95% 신뢰구간입니다. 막대가 0선을 넘지 않으면
 통계적으로 유의하게 이긴 겁니다.

 위쪽 Tᵢ를 보시면, 4개의 독립 분할 전부에서 신뢰구간이 0을 제외합니다.
 컷 모집단에서 +0.17에서 +0.26, 포함 모집단에서 +0.23에서 +0.32 — 4/4에 4/4입니다.
 데이터 처리 방식에 관계없이 성립한다는 뜻이고, 그래서 이건 무조건부 주장입니다.
 8개 셀 전부에서 인과 GP와 persistence도 이깁니다.

 아래쪽 회전은 다릅니다. 점추정은 8개 셀 전부 양수지만, 유의는 컷 1개, 포함 2개뿐입니다.
 4개 중 1~2개는 잡음이 만들 수 있는 수준이라 저는 이걸 '이겼다'고 쓰지 않고 동률로
 보고합니다.

 한 가지 더 정직하게 말씀드리면, 포함 모집단 수치가 더 높아 보이는 건 모델이 더 잘해서가
 아니라 스파이크가 보간 앵커를 오염시켜 모든 arm이 좋아 보이기 때문입니다.
 그 성분을 결과 ⑥에서 분해합니다."
""")
    note(kicker(B.s_res_gate(), "5. 결과 ③"), """
⏱ 11:40–12:30  (50초)

"주 모델을 왜 바꿨는지가 이 슬라이드입니다. 백본 관문입니다.

 조건 네 개를 먼저 못박았습니다 — 4개 분할 전부 부호 유지, 통합 신뢰구간이 0 제외,
 예산을 균등하게 맞춰도 부호 유지, 그리고 회전에 손해가 없을 것.
 분할 4개 × 초기화 4개, 총 16번 학습해서 각 run을 자기 분할의 윈도 대조군과 짝지었습니다.
 Tᵢ는 16/16 전부 양수, 13개가 유의합니다. 통합하면 +0.081이고 신뢰구간이 +0.067에서
 +0.096입니다. 학습 예산을 고정해도 4개 분할 부호가 유지되고, 회전은 유의한 열세가
 하나도 없습니다.

 무엇을 산 건가. 윈도 대조군은 인과 GP와 동률(1/4)인데 시퀀스 백본은 4/4에 4/4입니다.
 세그먼트 과거 전체로의 도달거리가 최강 배치 기준선을 이기게 만든 겁니다.
 그리고 비용은 오히려 음수입니다 — 윈도 조립과 조합 증강이 없어서 학습비가 1/10입니다."
""")
    t_stress()
    note(kicker(B.s_res_asym(), "5. 결과 ⑤"), """
⏱ 13:40–14:40  (60초)  ★ 과학적 발견

"비대칭이 왜 생기는지를 절제로 확인했습니다. 평가 시점에 modality를 지우는 실험입니다.

 먼저 공통점. 이력을 지우면 두 타겟 모두 무너집니다 — -1에서 -4까지 갑니다.

 차이는 빠른 진단입니다. Tᵢ는 컷 모집단에서 빠른 채널을 0으로 만들면 보간 아래로
 떨어집니다(-0.125). 즉 보간을 이기는 마진 자체가 빠른 진단이 운반한 정보입니다.
 물리적으로도 자연스럽습니다 — ECEI의 Tₑ와 BES의 nₑ가 충돌 e–i 결합으로 Tᵢ와 연결되니까요.

 회전은 정반대입니다. 빠른 채널을 전부 0으로 만들어도 출력이 비트 단위로 동일합니다.
 8개 셀 전부에서요. 회전 정보는 100% CES 이력에서 옵니다.
 이유는 두 가지입니다. 회전을 구동하는 NBI 토크가 데이터에 없고, Mirnov는 kHz로
 진동하는 dB/dt를 100 Hz로 순간샘플해서 위상이 무작위가 됩니다 — lag-1 자기상관이
 BES는 +0.568인데 Mirnov는 -0.009, 즉 이 격자 위에서는 백색잡음입니다.

 그리고 정직하게 하나 더: 포함 모집단에서는 이력만 쓰는 모델도 PCHIP을 +0.15~+0.23
 이깁니다. 그 마진에는 스파이크 강건성 성분이 섞여 있다는 뜻이고, 빠른 진단의 기여를
 분리해서 보려면 컷 모집단을 봐야 합니다. 그래서 두 모집단을 함께 보고합니다."

※ 질문 대비: "Tₑ가 NBI 가열을 대리하니 회전 정보도 담기지 않나?" — 검정했다.
   Tₑ~Tᵢ는 r=+0.353(p=3e-17)로 경로가 실재하지만 Tₑ~V_rot은 r=+0.024(p=0.58).
   power와 torque가 다르기 때문이다.
""")
    note(kicker(B.s_res_scaling(), "5. 결과 ⑥"), """
⏱ 14:40–15:30  (50초)

"그럼 모델을 더 키우면 되지 않느냐. 두 실험으로 닫았습니다.

 왼쪽이 복잡도 사다리입니다. persistence에서 시작해 파라미터를 늘려갑니다.
 재미있는 건 b3k8입니다 — persistence 예측에 유계 latent 8개의 선형 보정을 더한
 2만 1천 파라미터 모델인데, 컷 모집단에서 35만 파라미터 백본과 동급입니다.
 짝지은 차이가 평균 +0.002이고 신뢰구간이 전부 0을 포함합니다.
 즉 컷 조건에서 백본의 Tᵢ skill 전부가 숫자 8개로 압축됩니다.
 그 latent이 뭘 담고 있는지도 probe로 확인했습니다 — 직전 Tᵢ와 ECEI의 Tₑ입니다.
 다만 포함 모집단에서는 -0.194로 백본에 집니다. 유계 보정으로는 스파이크 앵커를
 못 살리기 때문입니다. 1%도 안 되는 행이 모든 arm SSE의 70~83%를 차지합니다.

 오른쪽이 크기 축입니다. Tᵢ 인코더 폭을 24에서 260까지, 파라미터로 3만 4천에서
 87만 9천까지 26배를 키웠는데 skill이 +0.230에서 +0.236 사이에서 평평합니다.
 남은 분산은 모델 크기가 아니라 분할 분산입니다. 크기 축은 닫혔습니다."
""")
    note(kicker(B.s_res_peak(), "5. 결과 ⑦"), """
⏱ 15:30–16:15  (45초)

"그럼 모델은 어디서 이득을 버는가. peak 분석입니다.

 여기서 peak은 예측값이 아니라 입력 기준으로 정의합니다 — 타겟 행을 제외한 이웃에서
 국소 활동도가 높은 구간입니다. 정답을 엿보지 않는 보수적인 정의입니다.

 Tᵢ는 peak 구간에서 컷 +0.45~+0.61, 포함 +0.62~+0.72로 8개 셀 전부 PASS입니다.
 본류에서는 +0.09~+0.20으로 훨씬 작습니다.
 회전은 더 극적입니다. 전역으로는 동률인데 peak 구간에서는 +0.54에서 +0.79이고
 persistence 대비로는 8/8 PASS입니다. 본류는 사실상 0입니다.

 해석은 명확합니다. 매끄러운 본류에서는 보간이 이미 거의 최적이라 이길 여지가 없습니다.
 반대로 급변 구간에서는 보간이 구조적으로 실패하고, 거기서 빠른 진단이 값어치를 합니다.
 즉 회전의 열세도 전역적인 게 아니라 지역적입니다."
""")
    s_tr = kicker(B.s_res_transient(), "5. 결과 ⑧")
    # cross-reference fix: the 1-hour deck points at its 결과 ④(간극)·⑩(peak).
    set_para(s_tr, "결과 ④⑩", "●  우위는 gap·peak에 집중 (결과 ⑦과 일관)")
    note(s_tr, """
⏱ 16:15–16:55  (40초)

"통계만 보면 감이 안 오니 실제 held-out TEST shot 하나를 보여드리겠습니다. #31815입니다.

 위가 이온온도입니다. 검은 점이 실측, 파란 선이 모델, 회색이 PCHIP입니다.
 빨간 점선은 BES가 급락한 시점인데, 빠른 진단의 급락이 CES crash와 정렬되어 있습니다 —
 빠른 진단이 급변을 먼저 봅니다. PCHIP은 스파이크마다 overshoot합니다. 미래까지
 다 보고도 그렇습니다.
 이 shot에서 Tᵢ RMSE는 199.2 대 262.3, skill로 +0.42입니다. 실측점 395개 기준입니다.
 회전도 이 shot에서는 +0.21로 이깁니다. 관측이 끊기면 이력만으로 완만히 감쇠합니다.

 다만 정직하게 말씀드리면 이런 shot이 전부는 아닙니다. 우위는 고르게 퍼져 있는 게
 아니라 간극과 급변 구간에 집중되어 있습니다 — 앞의 peak 분석과 정확히 같은 그림입니다."

※ 시간이 밀리면 이 슬라이드는 건너뛰어도 논지는 성립한다.
""")

    # ---- 6~7. 결론 · 한계 · 요약 (21–23) --------------------------------
    note(kicker(B.s_conclusion(), "6. 결론"), """
⏱ 16:55–18:10  (75초)

"정직한 결론 네 가지입니다.

 하나. Tᵢ는 미래를 쓰는 보간을 두 모집단 모두에서 유의하게 능가합니다. 4개 독립 분할
 전부이고, 배치 가능한 최강 인과 방법인 인과 GP는 8개 셀 전부에서 이깁니다.
 상한도 같이 보고합니다 — 최강 오프라인 평활기와는 동률입니다.

 둘. 배치 주장이 이제 두 스트레스를 다 견딥니다. 실제 결측 지점으로 재가중해도 인과
 방법 대비 8/8, 캠페인 경계를 넘어서도 4/4에 4/4입니다. 대체된 윈도 대조군은 여기서
 오프라인 우위를 완전히 잃었고, 그 차이의 원인을 드리프트로 측정했습니다.
 이전 발표에서 '캠페인 분할에서 우위가 사라진다'고 말씀드렸던 부분이 이번에 해결됐습니다.

 셋. 회전은 전역 동률입니다. 다만 15 ms 넘는 간극과 peak 층에서는 두 모집단 모두
 이깁니다. skill이 없는 게 아니라 천이 구간에만 있는 겁니다.

 넷. 상한은 추정기가 아니라 정보입니다. 2만 1천 파라미터가 컷에서 백본과 같고 폭은
 평평합니다. 그래서 다음 단계는 더 큰 모델이 아니라 데이터입니다."
""")
    t_limits()
    note(B.s_closing(), """
⏱ 19:00–19:30  (30초) + Q&A

"정리하면, 항상 있는 빠른 진단으로 자주 비는 CES를 채우는 엄격히 인과적인 가상 센서를
 만들었고, 미래까지 보는 보간이라는 일부러 어려운 기준선으로 검증했습니다.
 이온온도는 두 모집단 모두에서 유의하게 이겼고 두 스트레스도 견뎠습니다. 회전은
 전역 동률이며 그 이유가 물리로 설명됩니다. 감사합니다."

── 예상 질문 대비 ──────────────────────────────
Q. 왜 forecasting이 아니라 nowcasting인가?
 → 미래를 예보하는 게 아니라 이미 지나간 시점의 빈 값을 채우는 문제. 오프라인 물리 분석용.

Q. 진짜 결측 구간에서도 이만큼 잘하나?
 → 단정하지 않는다. MNAR이므로 관측점 성능은 낙관적 상한. 결측 분포로 재가중하면
    인과 대비는 8/8 생존, 오프라인 대비는 모집단 조건부(컷 2/4·포함 4/4).

Q. 보간이 미래를 보는데 이기는 게 가능한가?
 → 가능하고 그게 논지의 핵심. 빠른 진단이 시간 보간으로 얻을 수 없는 정보를 운반하기 때문.
    특히 보간이 구조적으로 실패하는 급변 구간에서 이긴다.

Q. 모델 크기/과적합은?
 → 백본 357,570 파라미터. 폭 34k→879k 스윕이 평평하고 21k 모델이 컷에서 동급이므로
    문제는 capacity가 아니라 정보량이다.

Q. V_rot을 살리려면?
 → NBI 토크 채널 확보가 1순위, 원본 kHz Mirnov(모드 회전 주파수)가 그다음.
    Mirnov 파생 특징 재가공은 이미 음성으로 확인됐다.

Q. 실시간에 쓸 수 있나?
 → 상태 유지 1-step 추론이 CPU 중앙값 1.05 ms / p99 1.61 ms로 10 ms 예산의 16%.
    conformal 예측구간도 32/32 셀에서 두 기준선을 이긴다.

Q. seed에 따라 흔들리지 않나?
 → Tᵢ는 4개 독립 분할 × 두 모집단 전부 PASS로 강건. V_rot은 분할에 따라 크게 요동하며,
    그것 자체가 "현 규모에서 V_rot 단일 수치는 측정 불가"라는 근거다.
""")

    warns = B._fit_report()
    out = os.path.join(HERE, "KSTAR_CES_발표자료_20분.pptx")
    prs.save(out)
    print("SAVED:", out, "| slides:", len(prs.slides._sldIdLst))
    for w in warns:
        print("  FIT WARNING:", w)
    print("FIT WARNING count:", len(warns))


if __name__ == "__main__":
    build()
