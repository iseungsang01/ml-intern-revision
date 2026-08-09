# -*- coding: utf-8 -*-
"""Build the **research-trajectory** deck — "지금까지 어떻게 흘러왔고, 지금 어디에 있는가".

Output: docs/presentation/KSTAR_CES_연구흐름.pptx

This is a third, different-purpose deck. The other two are *result* decks:

    build_pptx.py        -> 38 slides, ~60 min thesis defense   (결과를 설득하는 덱)
    build_pptx_20min.py  -> 24 slides, 20 min seminar           (결과를 압축한 덱)
    build_pptx_flow.py   -> this file, ~15 slides               (연구 경로를 보여주는 덱)

Where the result decks answer "이 결과를 믿어도 되는가", this one answers "이 연구는
어떤 경로로 여기까지 왔고, 무엇이 기각됐으며, 바로 다음에 무엇을 하는가". It is meant for
지도교수 보고 / 랩 미팅 / 본인 정리 — the negative results and the traps are first-class
content here, not footnotes.

Palette, layout helpers and figures are reused from build_pptx.py so all three decks
look like one family. Every number traces to THESIS_RESULTS.md (the regenerated
2026-07-14 checkpoint family) or PROJECT_KNOWLEDGE.md; section refs are in the notes.

Usage (from repo root):
    python docs/presentation/build_pptx_flow.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from build_pptx import (  # noqa: E402  (path bootstrap must run first)
    prs, slide, box, text, header, bullets, card, add_image_fit, table,
    NAVY, BLUE, TEAL, ORANGE, GREEN, RED, GRAY, LGRAY, MGRAY, WHITE, DARK, CARDBG,
    MONO, EMU_W, EMU_H, FIG,
)
from pptx.util import Inches, Pt  # noqa: E402
from pptx.dml.color import RGBColor  # noqa: E402
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR  # noqa: E402
from pptx.enum.shapes import MSO_SHAPE  # noqa: E402

OUT = os.path.join(HERE, "KSTAR_CES_연구흐름.pptx")


def note(s, txt):
    s.notes_slide.notes_text_frame.text = txt.strip("\n")
    return s


def chip(s, x, y, w, h, label, fill, color=WHITE, size=11):
    box(s, x, y, w, h, fill=fill, round_=True)
    text(s, x, y + Inches(0.04), w, h,
         [[(label, size, color, True, False, None)]], align=PP_ALIGN.CENTER)


# =============================== SLIDES ===================================

# --- 1. Title -------------------------------------------------------------
def f_title():
    s = slide()
    box(s, 0, 0, EMU_W, EMU_H, fill=NAVY)
    box(s, 0, Inches(5.5), EMU_W, Inches(2.0), fill=RGBColor(0x0E, 0x26, 0x47))
    box(s, Inches(0.9), Inches(1.75), Inches(2.2), Pt(4), fill=TEAL)
    text(s, Inches(0.9), Inches(1.95), Inches(11.6), Inches(0.5),
         [[("연구 진행 흐름 · Research Trajectory", 16, RGBColor(0x8F, 0xD6, 0xCB), True, False, None)]])
    text(s, Inches(0.88), Inches(2.5), Inches(11.7), Inches(2.0),
         [[("무엇을 했고, 무엇이 기각됐고,", 30, WHITE, True, False, None)],
          [("지금 어디에 있는가", 34, WHITE, True, False, None)]],
         line_spacing=1.12)
    text(s, Inches(0.9), Inches(4.35), Inches(11.5), Inches(1.0),
         [[("KSTAR CES nowcasting — 확정된 결과 · 기각된 경로 · 다음 실험을 ", 16, LGRAY, False, False, None),
           ("한 번에", 16, ORANGE, True, False, None),
           (" 보는 덱", 16, LGRAY, False, False, None)],
          [("결과 발표용 덱(60분 / 20분)과 달리, 음성 결과와 함정이 본문입니다", 16, LGRAY, False, False, None)]],
         line_spacing=1.2)
    text(s, Inches(0.9), Inches(5.9), Inches(11.5), Inches(1.1),
         [[("이승상  (Seungsang Lee)", 17, WHITE, True, False, None)],
          [("서울대학교 · 원자핵공학  |  2026-08-04 기준", 13, MGRAY, False, False, None)],
          [("출처: THESIS_RESULTS.md · PROJECT_KNOWLEDGE.md · docs/설명성_피드백_실험계획.md",
            11, MGRAY, False, False, None)]],
         line_spacing=1.25)
    return note(s, """
이 덱의 목적: 결과를 설득하는 게 아니라 연구가 걸어온 경로를 보여주는 것.
확정된 것 / 기각된 것 / 지금 막힌 것 / 다음 할 일을 한 번에 파악할 수 있게 구성했습니다.
""")


# --- 2. Status at a glance ------------------------------------------------
def f_status():
    s = slide()
    header(s, "Status", "지금 어디에 있나 — 한 장 요약", accent=TEAL)

    card(s, Inches(0.55), Inches(1.55), Inches(3.95), Inches(3.42),
         "① 확정 (논문 본문)",
         ["· Tᵢ: 미래까지 보는 보간(PCHIP)을",
          "  4개 독립 split에서 모두 유의하게 능가",
          "  (genuine +0.18…+0.28, 4/4 PASS)",
          "",
          "· V_rot: 보간과 동률 (PASS 1/4 = 잡음)",
          "  → Tᵢ↔V_rot 비대칭이 본 연구의 발견",
          "",
          "· 배치 주장은 인과 우위로 좁힘 —",
          "  결측 재가중·캠페인 분할 두 스트레스",
          "  테스트를 생존 (+0.29 / +0.22)"],
         accent=GREEN, body_size=11.5)

    card(s, Inches(4.72), Inches(1.55), Inches(3.95), Inches(3.42),
         "② 최근 라운드에서 배운 것",
         ["· 레버는 아키텍처가 아니라",
          "  데이터 처리 + 문제 프레이밍",
          "",
          "· held(forward-fill) 제거 학습이",
          "  V_rot을 4/4 seed 개선 → 새 기본값",
          "",
          "· W=4는 근거 없음 — W=2에서 plateau,",
          "  과거 관측 1개면 충분"],
         accent=BLUE, body_size=11.5)

    card(s, Inches(8.89), Inches(1.55), Inches(3.89), Inches(3.42),
         "③ 다음 작업 (우선순위)",
         ["· 고속 진단 shot별(인과) 표준화 —",
          "  캠페인 전이 수리 (BES 1.22σ 드리프트",
          "  실측, 통제 실험 설계 완료)",
          "",
          "· 이력 reach 확장 (슬롯≠span) —",
          "  커버리지 54.1%/4.8% 직접 개선",
          "",
          "· 원 kHz Mirnov window 특징 (V_rot 레버)",
          "· NBI 토크 데이터 확보 (원인 변수)"],
         accent=ORANGE, body_size=11.5)

    box(s, Inches(0.55), Inches(5.16), Inches(12.23), Inches(1.02), fill=NAVY, round_=True)
    text(s, Inches(0.85), Inches(5.24), Inches(11.6), Inches(0.90),
         [[("한 문장: ", 13.5, TEAL, True, False, None),
           ("Tᵢ 결과는 4개 split에서 재현되어 확정됐고, 배치 주장은 두 스트레스 테스트로 ",
            13.5, WHITE, False, False, None),
           ("'인과 우위'로 좁혀 확정", 13.5, ORANGE, True, False, None),
           ("됐다. 다음 레버는 아키텍처가 아니라 데이터다.", 13.5, WHITE, False, False, None)],
          [("남은 리스크: 캠페인 전이(오프라인 우위 소멸 — 수리 실험 대기) · V_rot 근본 한계 = 미관측 NBI 토크",
            11, MGRAY, False, False, None)]],
         line_spacing=1.15, space_after=3)
    return note(s, """
왼쪽부터: 확정 / 배운 것 / 막힌 것. 발표에서 이 한 장만 봐도 현재 상태가 전달되어야 합니다.
③의 "막힌 지점"은 실패가 아니라 다음 실험의 정의입니다 — 판정 기준까지 이미 정해져 있습니다.
""")


# --- 3. Timeline ----------------------------------------------------------
def f_timeline():
    s = slide()
    header(s, "Timeline", "연구 경로 — 6개 분기점", accent=NAVY)

    y0 = Inches(2.05)
    box(s, Inches(0.7), y0 + Inches(0.62), Inches(11.95), Pt(3), fill=LGRAY)

    stops = [
        ("06-22", "입력 ablation", "fast 진단은 Tᵢ 정보를 나르고\nV_rot 정보는 거의 없다", ORANGE),
        ("06-23", "peak 분석", "모델의 우위는 고변동\n구간에 집중된다", TEAL),
        ("07-14", "체크포인트 정합", "저장 가중치가 표류 —\n소스에서 재현해 통일", GRAY),
        ("07-30", "3연발 실험", "held 제거 = KEEP\nseq = Tᵢ만 · CT = 전부 기각", BLUE),
        ("08-04", "window sweep", "과거 관측 1개면 충분,\nW=4는 근거 없음", GREEN),
        ("08-05", "스트레스 2종+감사", "재가중·캠페인 분할 —\n배치 주장은 인과 우위로", RED),
    ]
    x = 0.7
    w = 1.93
    for i, (date, title, body, col) in enumerate(stops):
        cx = Inches(x)
        dot_x = cx + Inches(w / 2) - Inches(0.09)
        box(s, dot_x, y0 + Inches(0.50), Inches(0.18), Inches(0.18),
            fill=col, shape=MSO_SHAPE.OVAL)
        # date chip above the line
        chip(s, cx + Inches(w / 2) - Inches(0.42), y0 + Inches(0.02),
             Inches(0.84), Inches(0.30), date, col, size=10.5)
        # card below the line
        b = box(s, cx, y0 + Inches(0.92), Inches(w), Inches(1.72),
                fill=CARDBG, round_=True)
        box(s, cx, y0 + Inches(0.92), Inches(w), Inches(0.06), fill=col)
        text(s, cx + Inches(0.12), y0 + Inches(1.06), Inches(w - 0.24), Inches(0.34),
             [[(title, 12, col, True, False, None)]])
        text(s, cx + Inches(0.12), y0 + Inches(1.42), Inches(w - 0.24), Inches(1.1),
             [[(ln, 10, DARK, False, False, None)] for ln in body.split("\n")],
             line_spacing=1.12, space_after=1)
        x += w + 0.06

    box(s, Inches(0.7), Inches(5.35), Inches(11.95), Inches(1.35), fill=CARDBG, round_=True)
    box(s, Inches(0.7), Inches(5.35), Inches(0.10), Inches(1.35), fill=NAVY)
    text(s, Inches(0.95), Inches(5.5), Inches(11.5), Inches(1.1),
         [[("경로가 말해주는 것 — ", 13.5, NAVY, True, False, None),
           ("연구의 무게 중심이 '모델을 어떻게 바꿀까'에서 '데이터와 평가가 정직한가'로 옮겨갔다.",
            13.5, DARK, False, False, None)],
          [("6~7월 초의 AutoML 아키텍처 탐색(18 iteration)은 검증 손실을 안정적으로 낮추지 못했고, "
            "실제로 지표를 움직인 것은 held 제거(데이터)와 전체격자 재프레이밍(문제 정의)이었다.",
            12, GRAY, False, False, None)]],
         line_spacing=1.3)
    return note(s, """
타임라인은 날짜보다 '무엇이 바뀌었나'가 핵심입니다.
07-14의 체크포인트 정합은 결과가 아니라 신뢰성 작업 — 이걸 안 했으면 이후 모든 비교가 무의미했습니다.
""")


# --- 4. How the question evolved -----------------------------------------
def f_question():
    s = slide()
    header(s, "Framing", "연구 질문은 어떻게 바뀌었나 — 세 번의 전환", accent=BLUE)

    rows = [
        ("문제 정의",
         "고해상도 CES를 복원한다 (super-resolution)",
         "10 ms 격자에서 비어 있는 CES를 과거만 보고 채운다 (causal gap-filling / nowcasting)",
         "CES는 애초에 저해상도가 아니라 '자주 비는' 진단이다. 해상도 문제가 아니라 결측 문제.",
         ORANGE),
        ("평가 기준선",
         "persistence(직전값)를 이기면 성공",
         "미래까지 보는 오프라인 보간(PCHIP)을 이겨야 성공",
         "persistence는 너무 쉬운 상대. 일부러 불리한 bar를 세워야 주장이 방어된다.",
         BLUE),
        ("개선 레버",
         "아키텍처를 더 크게 · 더 복잡하게",
         "데이터 처리(held 제거)와 문제 프레이밍(전체 격자)",
         "18 iteration 아키텍처 탐색은 안정적 개선 실패. 데이터·프레이밍 변경은 4/4 seed 개선.",
         GREEN),
    ]
    y = 1.62
    for label, before, after, why, col in rows:
        box(s, Inches(0.55), Inches(y), Inches(12.23), Inches(1.62), fill=CARDBG, round_=True)
        box(s, Inches(0.55), Inches(y), Inches(0.10), Inches(1.62), fill=col)
        text(s, Inches(0.78), Inches(y + 0.14), Inches(1.7), Inches(0.4),
             [[(label, 13, col, True, False, None)]])
        text(s, Inches(2.55), Inches(y + 0.12), Inches(3.95), Inches(0.9),
             [[("이전", 10, MGRAY, True, False, None)],
              [(before, 12.5, GRAY, False, True, None)]], line_spacing=1.15)
        text(s, Inches(6.62), Inches(y + 0.26), Inches(0.4), Inches(0.52),
             [[("→", 20, col, True, False, None)]], align=PP_ALIGN.CENTER,
             space_after=0)
        text(s, Inches(7.15), Inches(y + 0.12), Inches(5.5), Inches(0.9),
             [[("현재", 10, col, True, False, None)],
              [(after, 12.5, DARK, True, False, None)]], line_spacing=1.15)
        text(s, Inches(2.55), Inches(y + 1.08), Inches(10.1), Inches(0.45),
             [[("왜: ", 11, col, True, False, None),
               (why, 11, GRAY, False, False, None)]])
        y += 1.72

    return note(s, """
세 전환 모두 외부 피드백 또는 자체 검증에서 나왔습니다.
특히 두 번째(평가 기준선 격상)가 이 연구의 방어력을 만든 결정입니다 — 이기기 쉬운 상대를 버렸습니다.
""")


# --- 5. Confirmed 1: headline --------------------------------------------
def f_confirmed_head():
    s = slide()
    header(s, "확정 ①", "헤드라인 — 4개 독립 split에서 재현된 Tᵢ 우위", accent=GREEN)
    add_image_fit(s, os.path.join(FIG, "fig_forest.png"),
                  Inches(0.55), Inches(1.5), Inches(7.55), Inches(4.95))

    card(s, Inches(8.35), Inches(1.5), Inches(4.43), Inches(2.35),
         "무엇이 확정됐나 (genuine 평가)",
         ["· CES_TI: +0.179 / +0.197 / +0.280 / +0.263",
          "  4개 held-out TEST split 모두 PR4 PASS",
          "  (held 포함 +0.19…+0.28도 4/4;",
          "   seed 1/7/123은 선택 밖 진짜 복제)",
          "· CES_VT: +0.203 / +0.162 / +0.100 / +0.183",
          "  점추정 전부 양수, PASS는 seed 1뿐 → 동률"],
         accent=GREEN, body_size=10.5)

    card(s, Inches(8.35), Inches(4.02), Inches(4.43), Inches(2.43),
         "이 결과가 강한 이유",
         ["· 상대가 미래를 보는 보간이다. 실시간에서는",
          "  쓸 수 없는 상대를 과거만 보는 모델이 이겼다.",
          "· RMSE 사다리 전 구간 1위 (genuine, Tᵢ eV):",
          "  모델 407.0 < 선형 441.0 < PCHIP 449.3",
          "  < persistence 504.3 < AR 2425.9"],
         accent=NAVY, body_size=10.5)
    return note(s, """
THESIS_RESULTS.md §4.1, §4.2.
핵심은 '4개 split 재현' — 단일 split 결과였다면 주장하지 않았을 것입니다.
CES_VT는 점추정이 4/4 양수인데도 n.s.로 보고합니다. 이게 이 프로젝트의 보고 기준입니다.
""")


# --- 6. Confirmed 2: asymmetry -------------------------------------------
def f_confirmed_asym():
    s = slide()
    header(s, "확정 ②", "Tᵢ ↔ V_rot 비대칭 — 본 연구의 과학적 발견", accent=ORANGE)

    col_w = [Inches(3.5), Inches(1.85), Inches(1.85)]
    table(s, Inches(0.55), Inches(1.6), col_w,
          ["입력 ablation (val, vs persistence)", "CES_TI", "CES_VT"],
          [["Full (이력 + 빠른진단 + 시간)", "+0.428", "+0.296"],
           ["no_fast (이력만)", "+0.458", "+0.295"],
           ["no_history (빠른진단만)",
            ("+0.372", GREEN, True, None), ("−0.642", RED, True, None)]],
          emphasis={2}, emphasis_fill=RGBColor(0xFF, 0xF3, 0xE6), size=12.5)

    text(s, Inches(0.55), Inches(3.55), Inches(7.3), Inches(0.9),
         [[("읽는 법: ", 12.5, NAVY, True, False, None),
           ("빠른 진단만 주면 Tᵢ는 여전히 persistence를 이기지만(+0.372), "
            "V_rot은 persistence보다 나빠진다(−0.642). ", 12.5, DARK, False, False, None),
           ("V_rot 정보는 사실상 전부 과거 CES 이력에서 온다.", 12.5, ORANGE, True, False, None)]],
         line_spacing=1.25)

    card(s, Inches(0.55), Inches(4.5), Inches(6.0), Inches(2.0),
         "물리적 근거 — 예측되고 확인된 비대칭",
         ["· Tᵢ: 충돌 e–i 결합 (τ_ei ∝ Tₑ^{3/2}/nₑ)",
          "  → ECEI(Tₑ) + BES(nₑ)가 Tᵢ 정보를 나른다",
          "· V_rot: 주로 NBI 토크가 결정하는데 NBI는",
          "  데이터에 없다. Mirnov는 10 ms(100 Hz)로",
          "  샘플링돼 kHz 모드 회전이 aliasing으로 소실."],
         accent=ORANGE, body_size=11.5)

    card(s, Inches(6.78), Inches(4.5), Inches(6.0), Inches(2.0),
         "이것이 왜 '실패'가 아닌가",
         ["· 물리에서 먼저 예측되고 ablation이 확인했다.",
          "· 문헌 검토 2회에서도 같은 비대칭이 지지됐다.",
          "· V_rot n.s.는 모델 부족이 아니라 측정되지 않은",
          "  입력의 한계다.",
          "→ 다음 레버는 NBI 토크 확보이지 더 큰 모델이 아니다."],
         accent=NAVY, body_size=11.5)
    return note(s, """
THESIS_RESULTS.md §5.2, §8b.2, §8b.3.
"Mirnov를 더 잘 쓰면 되지 않나" / "Tₑ가 NBI를 대리하지 않나" 두 반론은 각각 실험으로 기각했습니다(슬라이드 12).
""")


# --- 7. Confirmed 3: peak -------------------------------------------------
def f_confirmed_peak():
    s = slide()
    header(s, "확정 ③", "모델의 가치는 '고변동 구간'에 집중된다", accent=TEAL)
    add_image_fit(s, os.path.join(FIG, "fig_peak.png"),
                  Inches(0.55), Inches(1.5), Inches(7.4), Inches(4.9))

    card(s, Inches(8.25), Inches(1.5), Inches(4.53), Inches(2.5),
         "수치 (val split)",
         ["· CES_TI  전체 +0.272 → peak +0.702",
          "   95% CI [+0.503, +0.851]  PASS",
          "· CES_VT  전체 +0.131 → peak +0.438",
          "   95% CI [+0.068, +0.726]  PASS",
          "",
          "peak = 타깃 자신의 값을 배제한 입력 기반",
          "프록시로 고른 '주변이 요동치는' 구간."],
         accent=TEAL, body_size=11.5)

    card(s, Inches(8.25), Inches(4.17), Inches(4.53), Inches(2.28),
         "해석 — 그리고 남은 함정",
         ["· 보간은 매끄러운 구간에서 이미 최적에 가깝다.",
          "  모델의 값어치는 급변 구간에 있다.",
          "· 전체적으로 n.s.인 V_rot조차 peak에서는",
          "  PASS → 비대칭은 '지역적'이다.",
          "· 단, val split이고 관측점만 채점(MNAR",
          "  낙관 상한)이므로 헤드라인으로 쓰지 않는다."],
         accent=GRAY, body_size=11.5)
    return note(s, """
THESIS_RESULTS.md §5.1.
peak 선정이 순환 논리가 아니라는 점이 중요합니다 — 타깃 행의 값을 보지 않고 이웃만으로 고릅니다.
""")


# --- 8. How the method got harder ----------------------------------------
def f_method():
    s = slide()
    header(s, "Rigor", "평가가 단단해진 과정 — 스스로에게 불리하게", accent=BLUE)

    steps = [
        ("①", "선택 게이트 교체", "val loss → clean skill",
         "AutoML 유지/폐기 기준을 검증손실에서 깨끗한 skill로 바꾸자\nn.s.(+0.088) → 유의(+0.257)로 이동", BLUE),
        ("②", "3-way split + 사전등록", "선택과 보고의 분리",
         "선택은 val에서만. TEST 수치를 보기 전에 판정 규칙(PR1–PR4)을\n문서에 고정", TEAL),
        ("③", "shot-clustered bootstrap", "재표집 단위 = shot",
         "한 shot 안의 행은 자기상관 → 행 단위 CI는 과신.\nshot을 단위로 페어드 부트스트랩(B=10,000)", NAVY),
        ("④", "4-seed paired bar", "단일 run 금지",
         "V_rot 관련 주장은 4개 split 페어드 비교를 통과해야 인정.\n단일 seed 결과는 증거로 쓰지 않는다", GREEN),
    ]
    y = 1.55
    for num, title, tag, body, col in steps:
        box(s, Inches(0.55), Inches(y), Inches(7.9), Inches(1.18), fill=CARDBG, round_=True)
        box(s, Inches(0.55), Inches(y), Inches(0.10), Inches(1.18), fill=col)
        text(s, Inches(0.78), Inches(y + 0.30), Inches(0.5), Inches(0.5),
             [[(num, 19, col, True, False, None)]])
        text(s, Inches(1.35), Inches(y + 0.13), Inches(4.0), Inches(0.4),
             [[(title, 13.5, col, True, False, None)]])
        text(s, Inches(5.05), Inches(y + 0.15), Inches(3.2), Inches(0.35),
             [[(tag, 11, MGRAY, True, False, None)]], align=PP_ALIGN.RIGHT)
        text(s, Inches(1.35), Inches(y + 0.50), Inches(6.9), Inches(0.6),
             [[(ln, 11.5, DARK, False, False, None)] for ln in body.split("\n")],
             line_spacing=1.12, space_after=1)
        y += 1.27

    add_image_fit(s, os.path.join(FIG, "fig_progression.png"),
                  Inches(8.65), Inches(1.75), Inches(4.15), Inches(3.6))
    box(s, Inches(8.65), Inches(5.5), Inches(4.13), Inches(1.2), fill=NAVY, round_=True)
    text(s, Inches(8.88), Inches(5.62), Inches(3.7), Inches(1.0),
         [[("정직한 진전", 12.5, TEAL, True, False, None)],
          [("iter2 +0.088 [−0.221, +0.323] n.s.", 11, WHITE, False, False, None)],
          [("→ 최종 +0.257 [+0.118, +0.360] PASS", 11, WHITE, True, False, None)]],
         line_spacing=1.2)
    return note(s, """
THESIS_RESULTS.md §3, §6.
①이 이 프로젝트에서 가장 큰 단일 개선입니다 — 모델이 아니라 '무엇을 보고 고르는가'를 바꿨습니다.
""")


# --- 9. Data honesty round ------------------------------------------------
def f_data():
    s = slide()
    header(s, "Data", "데이터를 정직하게 만든 세 번의 수정", accent=TEAL)

    card(s, Inches(0.55), Inches(1.55), Inches(3.95), Inches(2.3),
         "① per-target 마스킹",
         ["기존: Tᵢ·V_rot 둘 다 관측된 행만 학습",
          "→ 라벨 행의 약 28%를 조용히 폐기",
          "",
          "수정: 행별 target_mask + per-target",
          "masked MSE. 하나만 관측돼도 그 타깃은",
          "학습에 기여한다."],
         accent=TEAL, body_size=11.5)

    card(s, Inches(4.72), Inches(1.55), Inches(3.95), Inches(2.3),
         "② held(stuck) 감사",
         ["관측된 V_rot의 54%가 직전값 복사",
          "(장비의 forward-fill padding)",
          "",
          "이 값들이 학습 타깃과 이력 입력에",
          "함께 들어가 '이력 복사가 최적'이라고",
          "가르치고 있었다 → 순환 논리."],
         accent=ORANGE, body_size=11.5)

    card(s, Inches(8.89), Inches(1.55), Inches(3.89), Inches(2.3),
         "③ held-free 학습 (KEEP)",
         ["held를 학습·평가에서 모두 제거하고",
          "4-seed 페어드 비교:",
          "",
          "V_rot 4/4 양수 · 3/4 CI가 0 제외,",
          "평균 +0.039 (MSE 약 4% 감소).",
          "Tᵢ는 영향 없음(평균 +0.004)."],
         accent=GREEN, body_size=11.5)

    add_image_fit(s, os.path.join(FIG, "fig_stuckfree_paired.png"),
                  Inches(0.55), Inches(4.0), Inches(7.3), Inches(2.5))

    card(s, Inches(8.05), Inches(4.0), Inches(4.73), Inches(2.78),
         "무엇이 바뀌었나",
         ["· 학습 기본값 변경: CES_DROP_STUCK_TARGETS=1",
          "  (V_rot 관련 모든 새 학습에 적용)",
          "· persistence 이야기의 부분적 탈-순환화:",
          "  V_rot의 이력 의존은 진짜 물리이면서",
          "  동시에 장비 padding으로 부풀려져 있었다.",
          "· 교훈: 데이터 처리를 기본값에 맡기지 말 것 —",
          "  모든 새 run에서 명시적으로 고정."],
         accent=NAVY, body_size=11.5)
    return note(s, """
THESIS_RESULTS.md §1, §8c.
②→③이 이 프로젝트에서 '데이터가 레버'라는 걸 처음 증명한 지점입니다.
""")


# --- 10. The 07-30 triple round ------------------------------------------
def f_round0730():
    s = slide()
    header(s, "2026-07-30", "같은 날 돌린 두 실험 — 레버는 데이터와 프레이밍이었다", accent=BLUE)

    col_w = [Inches(2.35), Inches(3.55), Inches(4.05), Inches(2.28)]
    table(s, Inches(0.55), Inches(1.55), col_w,
          ["실험", "가설", "결과 (4-seed paired)", "판정"],
          [[("held-free 학습", NAVY, True, None),
            "held가 학습을 오염시킨다",
            "V_rot 4/4 양수 · 3/4 유의, 평균 +0.039",
            ("KEEP", GREEN, True, None)],
           [("seq 재프레이밍", NAVY, True, None),
            "전체 격자 LSTM + loss 마스킹으로 윈도·증강을 없앤다",
            "Tᵢ 4/4 개선(평균 +0.042) · V_rot은 held-free 대조에서 4/4 악화",
            ("부분 KEEP", ORANGE, True, None)]],
          row_h=Inches(0.86), size=11.5)

    box(s, Inches(0.55), Inches(4.55), Inches(6.05), Inches(1.95), fill=CARDBG, round_=True)
    box(s, Inches(0.55), Inches(4.55), Inches(0.10), Inches(1.95), fill=BLUE)
    text(s, Inches(0.8), Inches(4.68), Inches(5.6), Inches(1.7),
         [[("seq의 V_rot 격차는 완전히 분해됐다", 12.5, BLUE, True, False, None)],
          [("(1) held 오염 — held-free로 학습하면 유의한 악화가 전부 사라짐",
            11.5, DARK, False, False, None)],
          [("(2) 라우팅 부재 — 동일 held-free 조건에서도 공유 인코더 LSTM은 "
            "V_rot을 4/4 잃는다. iter009의 V_rot head 설계(빠른진단 차단 + 관측마스킹 어텐션)가 "
            "그 우위의 원천임을 역으로 확인.", 11.5, DARK, False, False, None)]],
         line_spacing=1.22)

    box(s, Inches(6.75), Inches(4.55), Inches(6.03), Inches(1.95), fill=NAVY, round_=True)
    text(s, Inches(7.0), Inches(4.68), Inches(5.6), Inches(1.7),
         [[("이 라운드의 한 줄 교훈", 12.5, TEAL, True, False, None)],
          [("데이터 처리(held 제거)와 문제 프레이밍(전체 격자)은 지표를 움직였고, "
            "앞선 라운드의 아키텍처 미세변형은 움직이지 않았다.", 11.5, WHITE, False, False, None)],
          [("→ 다음 통제 변경도 '더 좋은 인코더'가 아니라 데이터·프레이밍·평가 쪽에서 찾는다.",
            11.5, ORANGE, True, False, None)]],
         line_spacing=1.22)
    return note(s, """
THESIS_RESULTS.md §8c / §8d.
seq v2(전체격자 프레이밍 + iter009의 V_rot 라우팅, held-free)가 이 라운드가 지목한 다음 후보입니다.
""")


# --- 11. Window sweep -----------------------------------------------------
def f_window():
    s = slide()
    header(s, "2026-08-04", "window sweep — 'W=4여야 하는가'에 답하다", accent=GREEN)
    add_image_fit(s, os.path.join(FIG, "fig_window_sweep.png"),
                  Inches(0.55), Inches(1.5), Inches(7.5), Inches(4.95))

    card(s, Inches(8.3), Inches(1.5), Inches(4.48), Inches(2.92),
         "24 run이 말한 것",
         ["· 과거 관측 1개가 전부다. history-0이면",
          "  Tᵢ −0.026(보간 이하), V_rot −0.783.",
          "  하나만 주면 둘 다 즉시 최대치.",
          "· W=2…8에서 평평하다 (Tᵢ 0.190–0.246,",
          "  V_rot 0.190–0.206). seed 산포가",
          "  곡선 전체보다 넓다 → W=4는 근거 없음.",
          "· W>2의 이점은 skill이 아니라 커버리지.",
          "  (긴 gap 표본 채점 수 4~10배)"],
         accent=GREEN, body_size=11)

    box(s, Inches(8.3), Inches(4.58), Inches(4.48), Inches(1.92), fill=RGBColor(0xFF, 0xF3, 0xE6),
        round_=True)
    box(s, Inches(8.3), Inches(4.58), Inches(0.10), Inches(1.92), fill=RED)
    text(s, Inches(8.55), Inches(4.66), Inches(4.05), Inches(1.80),
         [[("함정 — 첫 패스는 틀렸다", 12, RED, True, False, None)],
          [("held 처리가 기본값에 맡겨져(§8c 위반) 'V_rot은 긴 이력 필요' 결론이 나왔다.",
            10.5, DARK, False, False, None)],
          [("실제론 held가 짧은 윈도를 벌준 것 — 빼면 W=2가 따라잡고 기울기가 사라진다.",
            10.5, DARK, False, False, None)],
          [("교훈: 윈도 길이 비례 결론은 held-free로 재확인.",
            10.5, RED, True, False, None)]],
         line_spacing=1.12, space_after=3)
    return note(s, """
THESIS_RESULTS.md §8f / §8f-R.
이 슬라이드는 결과와 '자기 실수 정정'을 함께 보여줍니다. 외부 피드백("왜 4인가")에 답하려다
데이터 처리 기본값 함정을 하나 더 찾은 라운드입니다.
""")


# --- 12. Rejected paths ---------------------------------------------------
def f_rejected():
    s = slide()
    header(s, "Negative results", "기각된 경로 — 다시 시도하지 않기 위한 목록", accent=RED)

    col_w = [Inches(3.1), Inches(4.5), Inches(4.63)]
    table(s, Inches(0.55), Inches(1.5), col_w,
          ["시도", "무엇을 기대했나", "왜 기각했나"],
          [["AutoML 아키텍처 탐색 (18 iter)",
            "구조를 바꾸면 검증손실이 내려간다",
            "안정적 하강 추세 없음. 용량 확대·복잡 skip·local conv 모두 무효"],
           ["Mirnov 재가공 특징",
            "적분·pchip·burst 특징이 V_rot을 살린다",
            "4-seed 페어드에서 전부 무효. MC는 무작위 위상 dB/dt 스냅샷"],
           ["Tₑ → NBI 대리 가설",
            "Tₑ가 NBI 가열을 대리하니 토크도 담긴다",
            "데이터 수준에서 기각. 가열과 토크는 분리된다"],
           ["W > 2 (넓은 윈도)",
            "긴 이력이 V_rot에 도움",
            "held 오염 artifact였다. held-free에서 기울기 소멸"],
           ["seq 단독 (공유 인코더)",
            "전체 격자면 라우팅 없이도 된다",
            "Tᵢ는 개선, V_rot은 4/4 유의 악화 → 라우팅은 필수"]],
          row_h=Inches(0.62), size=11.5)

    box(s, Inches(0.55), Inches(5.72), Inches(12.23), Inches(0.95), fill=NAVY, round_=True)
    text(s, Inches(0.85), Inches(5.85), Inches(11.6), Inches(0.75),
         [[("이 목록이 자산인 이유 — ", 13, TEAL, True, False, None),
           ("다섯 경로 모두 사전등록된 통제 실험으로 닫혔고, 각각 '무엇이 달라야 재시도할 수 있는지'가 함께 기록돼 있다. "
            "덕분에 남은 탐색 공간이 좁고 명확하다.", 13, WHITE, False, False, None)]],
         line_spacing=1.25)
    return note(s, """
PROJECT_KNOWLEDGE.md "Avoid Repeating These Paths" + THESIS_RESULTS.md §8b/§8f.
질문이 나오면: 각 행은 단일 run이 아니라 4-seed 페어드 판정으로 닫았습니다.
""")


# --- 13. Reproducibility traps -------------------------------------------
def f_traps():
    s = slide()
    header(s, "Traps", "재현성 함정 4종 — 시간을 가장 많이 쓴 곳", accent=GRAY)

    items = [
        ("체크포인트 표류", RED,
         ["· model.py는 AutoML이 Transformer로 재작성돼",
          "  저장된 45개 가중치를 하나도 못 불러온다.",
          "· 논문 아키텍처는 아카이브된 model_iter009.py뿐.",
          "· .improve_final_out 가중치는 기록 지표 미재현",
          "  (V_rot +0.056 vs 기록 +0.161).",
          "· 해결: 소스 재학습 → .vt_repro_* 패밀리로 통일."]),
        ("데이터 처리 기본값", ORANGE,
         ["· held 처리를 명시하지 않으면 train.py 기본값 0이",
          "  조용히 선택된다.",
          "· window sweep 첫 패스의 잘못된 결론이 여기서 나왔다.",
          "· 해결: 모든 run에서 CES_DROP_STUCK_TARGETS를 명시."]),
        ("split manifest 파괴", BLUE,
         ["· train.py는 CES_SPLIT_DIR/split_manifest.json을",
          "  다시 쓴다. CES_TEST_FRACTION=0(기본)이면",
          "  3-way split의 test_files를 지워버린다.",
          "· 해결: manifest 백업 또는 복사본을 가리키게 할 것."]),
        ("Windows/MKL 콘솔 핸들러", GRAY,
         ["· 부모 콘솔이 닫히면 MKL이 학습을 중단시킨다",
          "  (forrtl 200, exit 3221225786).",
          "· 해결: FOR_DISABLE_CONSOLE_CTRL_HANDLER=1 +",
          "  KMP_HANDLE_SIGNALS=0, detached 실행 + --resume."]),
    ]
    pos = [(0.55, 1.55), (6.75, 1.55), (0.55, 4.15), (6.75, 4.15)]
    for (title, col, lines), (x, y) in zip(items, pos):
        card(s, Inches(x), Inches(y), Inches(6.03), Inches(2.42), title, lines,
             accent=col, body_size=11)
    return note(s, """
PROJECT_KNOWLEDGE.md "Checkpoint / Architecture Provenance" + §8f Windows note.
발표에서는 빠르게 넘기되, "왜 결과 재생성에 시간이 걸렸나" 질문이 나오면 이 장으로 답합니다.
""")


# --- 14. Next -------------------------------------------------------------
def f_next():
    s = slide()
    header(s, "Next", "바로 다음 작업 — 우선순위와 판정 기준", accent=ORANGE)

    box(s, Inches(0.55), Inches(1.5), Inches(12.23), Inches(2.35), fill=CARDBG, round_=True)
    box(s, Inches(0.55), Inches(1.5), Inches(0.12), Inches(2.35), fill=RED)
    text(s, Inches(0.85), Inches(1.62), Inches(11.6), Inches(0.45),
         [[("1순위 ", 15, RED, True, False, None),
           ("고속 진단의 shot별(인과) 표준화 — 캠페인 전이 수리", 15, NAVY, True, False, None),
           ("   — 정규화 env 1곳 변경 · 4-seed · 두 분할 규칙 채점", 12, GRAY, False, False, None)]])
    text(s, Inches(0.85), Inches(2.1), Inches(5.8), Inches(1.6),
         [[("왜", 12, RED, True, False, None)],
          [("캠페인 시간 분할에서 오프라인 보간 대비 우위가 소멸(0/4). 원인은 실측됨 — "
            "train→test 드리프트가 BES 1.22σ·ECEI 0.53σ인데 CES 타겟은 0.115σ.",
            11.5, DARK, False, False, None)],
          [("train-file-only 정규화(무작위 분할에선 올바른 선택)가 캠페인 이동에서 깨지는 "
            "바로 그 지점이다.", 11.5, DARK, False, False, None)]],
         line_spacing=1.18)
    text(s, Inches(6.95), Inches(2.1), Inches(5.6), Inches(1.6),
         [[("할 일 · 판정", 12, RED, True, False, None)],
          [("shot 자신의 데이터만 쓰는 표준화(누수 없음·실행 시점 가용)로 교체 후 "
            "무작위·캠페인 두 분할 규칙에서 4-seed 채점.", 11.5, DARK, False, False, None)],
          [("판정: 캠페인 분할 vs PCHIP 회복 여부 + 무작위 분할 skill 비용. 절대 수준 정보"
            "(BES↔밀도) 상실이 Tᵢ를 해칠 수 있음 — 가정 아닌 측정 대상.", 11.5, ORANGE, True, False, None)]],
         line_spacing=1.18)

    card(s, Inches(0.55), Inches(4.02), Inches(6.03), Inches(2.62),
         "2순위 — 이력 reach 확장 (슬롯≠span)",
         ["동기: 진짜 결측 중 도메인 내가 Tᵢ 54.1% ·",
          "V_rot 4.8%뿐 (W=4). 커버리지가 binding limit.",
          "",
          "· 슬롯은 2~3개로 두고 더 넓은 span에서 뽑는다",
          "  (dataset.py의 미사용 max_window_span 자리)",
          "· sweep 근거: W는 skill을 안 사고 커버리지만 산다",
          "· 완전 무작위 금지 — 기준선 정의가 바뀌면 비교가 깨짐"],
         accent=BLUE, body_size=11)

    card(s, Inches(6.75), Inches(4.02), Inches(6.03), Inches(2.62),
         "3순위 — 데이터 확보 2종 (V_rot의 진짜 레버)",
         ["· 원 kHz Mirnov의 window별 RMS·대역 파워·모드 번호",
          "  — 무필터 100 Hz 데시메이션이 파괴한 정보의 상류 복원",
          "· NBI 토크(빔별 파워·기하) — 회전의 원인 변수.",
          "  액추에이터 입력 시 회전 예측 가능(양성 대조군 문헌)",
          "",
          "완료(이번 라운드): 큰 gap causal 재평가 · 앵커+Δ 사다리",
          "· MNAR 재가중 · 캠페인 분할 · conformal · latency"],
         accent=TEAL, body_size=10.5)
    return note(s, """
THESIS_RESULTS.md §8n(캠페인)·§8i(MNAR 커버리지)·§8j(세 레버). 이전 판의 1순위(큰 gap causal
재평가)와 3순위(앵커+Δ 배치)는 2026-08-05에 완료 — §8g·§8k로 기록됨.
""")


# --- 15. Closing ----------------------------------------------------------
def f_closing():
    s = slide()
    box(s, 0, 0, EMU_W, EMU_H, fill=NAVY)
    box(s, Inches(0.75), Inches(0.62), Inches(2.2), Pt(4), fill=TEAL)
    text(s, Inches(0.75), Inches(0.85), Inches(11.8), Inches(0.9),
         [[("연구 흐름 한 장 요약", 26, WHITE, True, False, None)]])

    flow = [
        ("문제를 다시 정의", "초해상 → 10 ms causal gap-filling", TEAL),
        ("기준선을 격상", "persistence → 미래를 보는 보간(PCHIP)", BLUE),
        ("Tᵢ 확정 · V_rot 진단", "4/4 PASS  |  비대칭 = 미관측 NBI 토크", GREEN),
        ("레버를 재조준", "아키텍처 → 데이터(held) · 프레이밍(전체격자)", ORANGE),
        ("배치 주장을 좁혀 확정", "결측 재가중·캠페인 분할 → 인과 우위만 생존 ← 지금", RED),
    ]
    y = 2.0
    for i, (title, sub, col) in enumerate(flow):
        box(s, Inches(0.75), Inches(y), Inches(11.85), Inches(0.78),
            fill=RGBColor(0x1A, 0x3D, 0x6E), round_=True)
        box(s, Inches(0.75), Inches(y), Inches(0.09), Inches(0.78), fill=col)
        chip(s, Inches(1.0), Inches(y + 0.2), Inches(0.38), Inches(0.38), str(i + 1), col, size=12)
        text(s, Inches(1.6), Inches(y + 0.16), Inches(4.4), Inches(0.5),
             [[(title, 14.5, WHITE, True, False, None)]])
        text(s, Inches(6.1), Inches(y + 0.18), Inches(6.3), Inches(0.5),
             [[(sub, 12.5, LGRAY, False, False, None)]])
        if i < len(flow) - 1:
            text(s, Inches(1.06), Inches(y + 0.62), Inches(0.3), Inches(0.32),
                 [[("↓", 12, MGRAY, True, False, None)]], align=PP_ALIGN.CENTER)
        y += 0.93

    box(s, Inches(0.75), Inches(6.68), Inches(11.85), Inches(0.55),
        fill=RGBColor(0x0E, 0x26, 0x47), round_=True)
    text(s, Inches(1.0), Inches(6.76), Inches(11.4), Inches(0.42),
         [[("결론: 모델을 더 키우는 단계는 끝났다. 남은 것은 ", 12.5, LGRAY, False, False, None),
           ("평가가 공정한지 확인하고, 측정되지 않은 입력(NBI)을 확보하는 것", 12.5, ORANGE, True, False, None),
           (".", 12.5, LGRAY, False, False, None)]],
         align=PP_ALIGN.CENTER)
    return note(s, """
마지막 장은 '지금 5단계 중 어디인지'를 못박습니다. 질문이 들어오면 슬라이드 2(현황)와 14(다음 작업)로 돌아갑니다.
""")


def build():
    f_title()
    f_status()
    f_timeline()
    f_question()
    f_confirmed_head()
    f_confirmed_asym()
    f_confirmed_peak()
    f_method()
    f_data()
    f_round0730()
    f_window()
    f_rejected()
    f_traps()
    f_next()
    f_closing()
    prs.save(OUT)
    print(f"wrote {OUT}  ({len(prs.slides.__iter__.__self__._sldIdLst)} slides)")


if __name__ == "__main__":
    build()
