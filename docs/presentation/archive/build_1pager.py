# -*- coding: utf-8 -*-
"""Build a single-page A4 one-pager (PDF + PNG) summarizing the KSTAR CES project.

2026-08-27 edition (abstract register). Every sentence is a declarative statement in the
register of a paper abstract, per 승상님's 2026-08-27 instruction. Content covers the
confirmed protocol (W = 2 · held-free · two co-primary populations · seq_v2 · causal GP)
AND the post-08-16 record: the reach ladder / family / operator-cost / win-loss analyses
(THESIS_RESULTS.md §8ac–§8an), the frozen μs shot set (§8ao) and the closed quantum arm
(§8ap). Numbers below are transcribed from docs/paper/paper_numbers.json (= main_ko.tex)
and from the §8ac–§8ap tables; the forest figure is docs/presentation/figures/fig_forest.png.

Output:
  docs/presentation/KSTAR_CES_1pager.pdf
  docs/presentation/KSTAR_CES_1pager.png
Run make_figures.py first (embeds fig_forest.png).
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import FancyBboxPatch

for cand in ["Malgun Gothic", "NanumGothic", "AppleGothic"]:
    try:
        matplotlib.font_manager.findfont(cand, fallback_to_default=False)
        plt.rcParams["font.family"] = cand
        break
    except Exception:
        continue
plt.rcParams["axes.unicode_minus"] = False

HERE = os.path.dirname(__file__)
FIG = os.path.join(HERE, "figures")

NAVY = "#13335f"
NAVY2 = "#0e2647"
BLUE = "#2b6cb0"
TEAL = "#1b9e8a"
ORANGE = "#e8743b"
GREEN = "#2e9e5b"
RED = "#c0392b"
GRAY = "#5b6670"
MGRAY = "#9aa5b1"
LGRAY = "#e8ecf1"
CARD = "#f4f7fa"
WHITE = "#ffffff"
DARK = "#222b35"

fig = plt.figure(figsize=(8.27, 11.69))  # A4 portrait
fig.patch.set_facecolor(WHITE)


def rect(x, y, w, h, fc, ec="none", lw=0, round_=False, pad=0.012, alpha=1.0):
    style = f"round,pad=0,rounding_size={pad}" if round_ else "square,pad=0"
    p = FancyBboxPatch((x, y), w, h, boxstyle=style, mutation_aspect=1.45,
                       fc=fc, ec=ec, lw=lw, transform=fig.transFigure,
                       clip_on=False, alpha=alpha)
    fig.patches.append(p)
    return p


def txt(x, y, s, size, color=DARK, weight="normal", style="normal", ha="left",
        va="top", font=None, spacing=None):
    fig.text(x, y, s, fontsize=size, color=color, fontweight=weight,
             fontstyle=style, ha=ha, va=va, wrap=False,
             family=font if font else plt.rcParams["font.family"],
             linespacing=spacing if spacing else 1.15)


# ============================ HEADER =====================================
rect(0, 0.928, 1, 0.072, NAVY)
rect(0, 0.928, 0.32, 0.072, NAVY2)
txt(0.045, 0.987, "KSTAR 다중 진단 기반 인과 시퀀스 나우캐스터에 의한", 14.5, WHITE, "bold")
txt(0.045, 0.963, "CES 결측 구간 예측 (Gap-filling / Nowcasting)", 16.5, WHITE, "bold")
txt(0.955, 0.985, "1-PAGE 요약 · 2026-08-27", 10.5, ORANGE, "bold", ha="right")
txt(0.955, 0.962, "이승상 · 서울대 원자핵공학", 9, LGRAY, ha="right")
txt(0.955, 0.944, "확정 프로토콜: W=2 · held-free · 두 모집단(컷/포함) · 인과 GP · B.9 포함", 8, MGRAY, ha="right")

# ============================ ABSTRACT ===================================
rect(0.045, 0.856, 0.91, 0.064, CARD, round_=True, pad=0.008)
rect(0.045, 0.856, 0.012, 0.064, ORANGE, round_=True, pad=0.004)
txt(0.072, 0.913, "초록", 8.3, ORANGE, "bold")
txt(0.072, 0.898,
    "본 연구는 항상 조밀한 빠른 진단(BES·ECEI·Mirnov)과 과거 CES 이력만으로 CES가 빈 10 ms 시점의 $T_i$·V_rot를 복원하는",
    8.6, DARK)
txt(0.072, 0.884,
    "인과 모델을 제안하고, 미래를 읽는 오프라인 보간(PCHIP)과 배치 가능한 최강 인과 기준선(인과 GP)을 상대로 사전등록 프로토콜",
    8.6, DARK)
txt(0.072, 0.870,
    "아래 검증하였다. $T_i$는 두 모집단·4 분할 전부에서 유의하게 이겼고, V_rot는 동률이었으며, 약 50 ms의 문맥에서 skill이 포화하였다.",
    8.6, DARK)

# ============================ PROBLEM / IDEA =============================
ytop = 0.846
txt(0.045, ytop, "■ 문제와 핵심 아이디어", 11, NAVY, "bold")
prob = [
    ("CES는 SNR 확보를 위해 광자를 오래 수집하므로 자주 결측된다(641 shot, 247,207행 10 ms 격자).", GRAY),
    ("NaN 결측은 $T_i$ 8.2%, V_rot 23.9%(서로 독립)이며, V_rot는 held(직전값 복사) 41.1%가 더해져 실질 무정보 65.0%이다.", RED),
    ("$T_i$ > 3 keV(0.53%)는 피팅 실패이므로 컷/포함 두 모집단을 공동 1차로 보고하고, 무조건부 주장은 둘 다 성립해야 한다.", GRAY),
    ("빠른 진단은 격자 100%에서 조밀하므로 격자 전체를 읽는 인과 시퀀스 모델(seq_v2, 358k)로 결측을 채운다.", ORANGE),
    ("V_rot 분기는 빠른 진단을 인코더 수준에서 차단한다(라우팅). 윈도 GRU+attention(W=2)은 paired 대조군이다.", BLUE),
]
yy = ytop - 0.022
for s, c in prob:
    txt(0.052, yy, "•", 9.5, c, "bold")
    txt(0.069, yy, s, 8.4, c, "bold" if c != GRAY else "normal")
    yy -= 0.0200

# ============================ KEY RESULT STATS ===========================
txt(0.045, 0.726, "■ 핵심 결과 (held-out TEST, 4 분할 × 두 모집단 · B.9 통합 301 방전)", 11, NAVY, "bold")
stats = [
    ("$T_i$ vs PCHIP", "4/4 + 4/4", "컷 +0.17~+0.26 · 포함 +0.23~+0.32\n95% CI > 0 (shot 군집 bootstrap)", GREEN),
    ("$T_i$ vs 인과 GP", "8/8", "+0.08~+0.17로 최강 배치 기준선을 이겼다\n오프라인 GP와는 동률이다", ORANGE),
    ("V_rot", "동률", "vs PCHIP 1/4·2/4 (잡음 수준)\n방전 단위 승률 0.48 (소수 방전 집중)", GRAY),
    ("문맥 포화", "약 50 ms", "승리 방전 비율 0.52 → 0.66\n세 계열은 0.023 이내 동률", TEAL),
]
sw, sgap = 0.218, 0.0125
sx = 0.045
sy = 0.650
for name, big, sub, col in stats:
    rect(sx, sy, sw, 0.058, CARD, round_=True, pad=0.008)
    rect(sx, sy + 0.042, sw, 0.016, col, round_=True, pad=0.005)
    txt(sx + sw / 2, sy + 0.050, name, 8.3, WHITE, "bold", ha="center", va="center")
    txt(sx + sw / 2, sy + 0.030, big, 13.0, col, "bold", ha="center")
    txt(sx + sw / 2, sy + 0.015, sub, 7.0, GRAY, ha="center", spacing=1.12)
    sx += sw + sgap

# ============================ METHOD =====================================
txt(0.045, 0.630, "■ 방법", 11, NAVY, "bold")
meth = [
    ("데이터", "641 shot(30801–32751). held는 학습·평가에서 제거하고, $T_i$ > 3 keV 컷/포함 두 모집단, file-level split, train 파일 전용 z-score를 적용하였다."),
    ("모델", "seq_v2: 격자 시퀀스 22채널 위의 인과 LSTM 2분기($T_i$ 2×160 / V_rot 1×64, 비-빠른 7채널만), 타깃별 masked MSE, shot별 표준화(357,570)."),
    ("대조군", "W=2 윈도 모델(GRU+관측마스킹 attention, 201,258)을 같은 행에서 paired 비교하였다. B.1 관문 16 run은 16/16 양수, pooled +0.081이다."),
    ("평가", "물리 단위 skill vs PCHIP, 기준선 6종(+인과 GP), shot 군집 paired bootstrap(B=10k), PR1–4, TEST 동결로 평가하였다."),
    ("B.9", "각 도달 범위에서 학습·채점한 seq_v2 사다리, 세 계열의 같은 문맥 paired 비교, 연산자 수 기반 비용 모델, 방전 단위 승패 회귀를 수행하였다."),
]
yy = 0.599
for tag, body in meth:
    rect(0.045, yy - 0.0035, 0.075, 0.016, BLUE, round_=True, pad=0.004)
    txt(0.0825, yy + 0.0045, tag, 8.2, WHITE, "bold", ha="center", va="center")
    txt(0.130, yy + 0.0045, body, 7.6, DARK, va="center")
    yy -= 0.0205

# ============================ RESULTS FIGURE =============================
txt(0.045, 0.482, "■ headline — seq_v2 vs PCHIP, 두 모집단, shot 군집 95% CI", 11, NAVY, "bold")
try:
    img = mpimg.imread(os.path.join(FIG, "fig_forest.png"))
    ax = fig.add_axes([0.045, 0.262, 0.91, 0.208])
    ax.imshow(img)
    ax.axis("off")
except Exception as e:
    txt(0.5, 0.35, f"[forest figure missing: {e}]", 9, RED, ha="center")

# ============================ ASYMMETRY / PHYSICS =======================
txt(0.045, 0.245, "■ 과학적 발견: $T_i$ ↔ V_rot 비대칭과 그 메커니즘", 11, NAVY, "bold")
rect(0.045, 0.163, 0.445, 0.072, CARD, round_=True, pad=0.008)
rect(0.045, 0.163, 0.012, 0.072, ORANGE, round_=True, pad=0.004)
txt(0.072, 0.227, "$T_i$ — 빠른 진단이 정보를 운반한다 (컷 모집단)", 9.2, ORANGE, "bold")
txt(0.072, 0.211, "충돌 e-i 결합($t_{ei} \\propto T_e^{1.5}/n_e$)으로 ECEI($T_e$)·BES($n_e$)가 단서를 운반한다.", 8.0, DARK)
txt(0.072, 0.197, "빠른 채널을 제거하면 보간 아래로 떨어진다(-0.10~-0.18; paired 4/4).", 8.0, DARK)
txt(0.072, 0.181, "방전 단위 승률 0.695이며 변동 3분위 방전에서 85%를 이긴다.", 8.0, GREEN, "bold")

rect(0.510, 0.163, 0.445, 0.072, CARD, round_=True, pad=0.008)
rect(0.510, 0.163, 0.012, 0.072, BLUE, round_=True, pad=0.004)
txt(0.537, 0.227, "V_rot — 정보는 전적으로 과거 이력에서 온다", 9.2, BLUE, "bold")
txt(0.537, 0.211, "빠른 채널을 0으로 두어도 출력이 bit-identical이다(구조적 라우팅, 8/8).", 8.0, DARK)
txt(0.537, 0.197, "NBI 토크는 미관측이고 Mirnov는 100 Hz aliasing으로 백색잡음(r = -0.009)이다.", 8.0, DARK)
txt(0.537, 0.181, "승률 조용한 방전 34% · 변동 방전 55%로 구동 변수의 부재가 원인이다.", 8.0, RED, "bold")

# ============================ CONCLUSION ================================
rect(0.045, 0.052, 0.91, 0.100, NAVY, round_=True, pad=0.008)
txt(0.072, 0.142, "결론", 9.5, ORANGE, "bold")
concl = [
    "① $T_i$는 미래를 읽는 PCHIP를 두 모집단 모두 4/4 이기고 인과 GP도 8/8 이겼다(무조건부). 오프라인 GP와는 동률이다.",
    "② 배치 주장은 두 스트레스를 견뎠다: 결측 재가중 vs 인과 8/8(vs PCHIP 2/4·4/4), 캠페인 시간 분할 4/4+4/4(윈도 대조군은 붕괴).",
    "③ 약 50 ms의 연속 인과 문맥이 우위를 전형적으로 만들며, 세 계열은 동률이므로 아키텍처는 비용(연산자 수, 2–3 µs/op)으로 정한다.",
    "④ V_rot는 동률이며 우위는 회전이 변하는 방전에 집중된다. 정보 부재는 물리(NBI·Mirnov)이며 검정력 문제가 아니다.",
    "⑤ 21k 해석가능 모델 = 백본(컷), 폭 26배 평평, 1.8k 합성곱도 4/4이므로 상한은 정보에 있다: 피팅 메타·kHz Mirnov·NBI 토크.",
]
yy = 0.124
for s in concl:
    txt(0.072, yy, s, 7.7, WHITE, va="center")
    yy -= 0.0165

# ============================ FOOTER ====================================
txt(0.045, 0.030, "출처: paper_numbers.json · THESIS_RESULTS.md §8ab, §8ac–§8ap", 7.0, MGRAY)
txt(0.955, 0.030, "한계: MNAR 도달 $T_i$ 54–68%/V_rot 4–6% · test shot 96/60–66 · 캠페인 1 블록 · 1 ms 보류",
    7.0, MGRAY, ha="right")

out_pdf = os.path.join(HERE, "KSTAR_CES_1pager.pdf")
out_png = os.path.join(HERE, "KSTAR_CES_1pager.png")
fig.savefig(out_pdf, facecolor=WHITE)
fig.savefig(out_png, dpi=170, facecolor=WHITE)
plt.close(fig)
print("SAVED:", out_pdf)
print("SAVED:", out_png)
