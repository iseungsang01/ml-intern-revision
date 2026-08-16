# -*- coding: utf-8 -*-
"""Build a single-page A4 one-pager (PDF + PNG) summarizing the KSTAR CES project.

Confirmed-protocol edition (2026-08-16, THESIS_RESULTS.md §8ab): W = 2 · held-free · two
co-primary populations (cut / inclusive) · main model seq_v2 · causal-GP baseline. Every
number below is transcribed from docs/paper/paper_numbers.json (= main.tex / main_ko.tex);
the forest figure is docs/presentation/figures/fig_forest.png (make_figures.py).

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
txt(0.045, 0.987, "KSTAR 다중 진단 기반 인과 시퀀스 나우캐스터로", 14.5, WHITE, "bold")
txt(0.045, 0.963, "CES 결측 구간 예측 (Gap-filling / Nowcasting)", 16.5, WHITE, "bold")
txt(0.955, 0.985, "1-PAGE 요약 · 2026-08-16", 10.5, ORANGE, "bold", ha="right")
txt(0.955, 0.962, "이승상 · 서울대 원자핵공학", 9, LGRAY, ha="right")
txt(0.955, 0.944, "확정 프로토콜: W=2 · held-free · 두 모집단(컷/포함) · 인과 GP", 8, MGRAY, ha="right")

# ============================ ONE-LINE THESIS ============================
rect(0.045, 0.870, 0.91, 0.050, CARD, round_=True, pad=0.008)
rect(0.045, 0.870, 0.012, 0.050, ORANGE, round_=True, pad=0.004)
txt(0.072, 0.909, "한 줄 요약", 8.3, ORANGE, "bold")
txt(0.072, 0.892,
    "항상 조밀한 빠른 진단(BES·ECEI·Mirnov)과 과거 CES 이력만으로 CES가 빈 10 ms 시점의 $T_i$·V_rot를 복원하고,",
    9.0, DARK)
txt(0.072, 0.876,
    "미래까지 보는 오프라인 보간과 최강 인과 기준선(인과 GP)을 causal 모델로 이기는가를 사전등록 프로토콜로 검증한다.",
    9.0, DARK)

# ============================ PROBLEM / IDEA =============================
ytop = 0.858
txt(0.045, ytop, "■ 문제 & 핵심 아이디어", 11, NAVY, "bold")
prob = [
    ("CES는 SNR 확보를 위해 광자를 오래 수집 → 자주 결측 (641 shot, 247,207행 10 ms 격자)", GRAY),
    ("NaN 결측: $T_i$ 8.2%, V_rot 23.9% (서로 독립) · V_rot는 held(직전값 복사) 41.1% 추가 → 실질 무정보 65.0%", RED),
    ("$T_i$ >3 keV(0.53%)는 피팅 실패 → 컷/포함 두 모집단을 공동 1차로 보고, 무조건부 주장은 둘 다 성립해야 함", GRAY),
    ("빠른 진단은 격자 100%에서 조밀 → 격자 전체를 읽는 인과 시퀀스 모델(seq_v2, 358k)로 결측 채움", ORANGE),
    ("V_rot 분기는 빠른 진단을 인코더 수준에서 차단(라우팅) · 윈도 GRU+attention(W=2)은 paired 대조군", BLUE),
]
yy = ytop - 0.022
for s, c in prob:
    txt(0.052, yy, "•", 9.5, c, "bold")
    txt(0.069, yy, s, 8.6, c, "bold" if c != GRAY else "normal")
    yy -= 0.0205

# ============================ KEY RESULT STATS ===========================
txt(0.045, 0.735, "■ 핵심 결과 (held-out TEST, 4 분할 × 두 모집단)", 11, NAVY, "bold")
stats = [
    ("$T_i$ vs PCHIP", "4/4 + 4/4", "컷 +0.17~+0.26 · 포함 +0.23~+0.32\n95% CI > 0 (shot 군집 bootstrap)", GREEN),
    ("$T_i$ vs 인과 GP", "8/8", "+0.08~+0.17 · 최강 배치 기준선도 이김\n오프라인 GP와는 동률", ORANGE),
    ("V_rot", "동률", "vs PCHIP 1/4·2/4 (잡음 수준)\nvs persistence 3/4 양쪽", GRAY),
    ("캠페인 분할", "4/4 + 4/4", "시간 순 test에서 vs PCHIP·인과 GP\n윈도 대조군은 2/4·0/4로 붕괴", TEAL),
]
sw, sgap = 0.218, 0.0125
sx = 0.045
sy = 0.658
for name, big, sub, col in stats:
    rect(sx, sy, sw, 0.058, CARD, round_=True, pad=0.008)
    rect(sx, sy + 0.046, sw, 0.012, col, round_=True, pad=0.005)
    txt(sx + sw / 2, sy + 0.0495, name, 8.5, WHITE, "bold", ha="center")
    txt(sx + sw / 2, sy + 0.032, big, 13.5, col, "bold", ha="center")
    txt(sx + sw / 2, sy + 0.016, sub, 7.1, GRAY, ha="center", spacing=1.12)
    sx += sw + sgap

# ============================ METHOD =====================================
txt(0.045, 0.638, "■ 방법", 11, NAVY, "bold")
meth = [
    ("데이터", "641 shot(30801–32751) · held는 학습·평가 모두 제거 · $T_i$>3 keV 컷/포함 두 모집단 · file-level split · train 파일 전용 z-score"),
    ("모델", "seq_v2: 격자 시퀀스 22채널, 인과 LSTM 2분기($T_i$ 2×160 / V_rot 1×64, 비-빠른 7채널만), 타겟별 masked MSE, shot별 표준화 (357,570)"),
    ("대조군", "W=2 윈도 모델(GRU+관측마스킹 attention, 201,258) — 같은 행에서 paired; B.1 관문 16 run 16/16 양수, pooled +0.081"),
    ("평가", "물리단위 skill vs PCHIP · 기준선 6종(+인과 GP) · shot-clustered paired bootstrap(B=10k) · PR1–4 · TEST 동결"),
    ("사다리", "b3k8(21k, persistence + 8 latent 선형 보정) = 백본 (컷, +0.002) · 폭 34k→879k 평평 → 정보가 상한"),
]
yy = 0.607
for tag, body in meth:
    rect(0.045, yy - 0.0035, 0.075, 0.016, BLUE, round_=True, pad=0.004)
    txt(0.0825, yy + 0.0045, tag, 8.2, WHITE, "bold", ha="center", va="center")
    txt(0.130, yy + 0.0045, body, 7.9, DARK, va="center")
    yy -= 0.0205

# ============================ RESULTS FIGURE =============================
txt(0.045, 0.490, "■ headline — seq_v2 vs PCHIP, 두 모집단, shot-clustered 95% CI", 11, NAVY, "bold")
try:
    img = mpimg.imread(os.path.join(FIG, "fig_forest.png"))
    ax = fig.add_axes([0.045, 0.262, 0.91, 0.215])
    ax.imshow(img)
    ax.axis("off")
except Exception as e:
    txt(0.5, 0.35, f"[forest figure missing: {e}]", 9, RED, ha="center")

# ============================ ASYMMETRY / PHYSICS =======================
txt(0.045, 0.245, "■ 과학적 발견: $T_i$ ↔ V_rot 비대칭 (평가 시 modality 절제)", 11, NAVY, "bold")
rect(0.045, 0.163, 0.445, 0.072, CARD, round_=True, pad=0.008)
rect(0.045, 0.163, 0.012, 0.072, ORANGE, round_=True, pad=0.004)
txt(0.072, 0.227, "$T_i$ — 빠른 진단이 정보 운반 (컷 모집단)", 9.2, ORANGE, "bold")
txt(0.072, 0.211, "충돌 e-i 결합($t_{ei} \\propto T_e^{1.5}/n_e$)로 ECEI($T_e$)+BES($n_e$)가 단서 운반", 8.0, DARK)
txt(0.072, 0.197, "빠른 채널 제거 시 보간 아래로(-0.10~-0.18; paired -0.25~-0.43 4/4*)", 8.0, DARK)
txt(0.072, 0.181, "포함 모집단에선 이력-전용도 +0.15~+0.23 → 두 모집단 보고의 근거", 8.0, GREEN, "bold")

rect(0.510, 0.163, 0.445, 0.072, CARD, round_=True, pad=0.008)
rect(0.510, 0.163, 0.012, 0.072, BLUE, round_=True, pad=0.004)
txt(0.537, 0.227, "V_rot — 정보는 전적으로 과거 이력", 9.2, BLUE, "bold")
txt(0.537, 0.211, "빠른 채널 0으로 만들어도 출력 bit-identical (라우팅은 구조적, 8/8)", 8.0, DARK)
txt(0.537, 0.197, "NBI 토크 미관측 · Mirnov 100 Hz aliasing(lag-1 r = -0.009)", 8.0, DARK)
txt(0.537, 0.181, "이력 제거 시 -2.9~-3.5 · skill은 peak 구간(+0.54~+0.79)에만", 8.0, RED, "bold")

# ============================ CONCLUSION ================================
rect(0.045, 0.052, 0.91, 0.100, NAVY, round_=True, pad=0.008)
txt(0.072, 0.142, "정직한 결론", 9.5, ORANGE, "bold")
concl = [
    "① $T_i$: 미래를 보는 PCHIP를 두 모집단 모두 4/4 이기고 인과 GP도 8/8 — 무조건부 주장. 오프라인 GP와는 동률.",
    "② 배치 주장은 두 스트레스를 견딤: 결측 재가중 vs 인과 8/8(vs PCHIP 2/4·4/4), 캠페인 시간 분할 4/4+4/4 (윈도 대조군은 붕괴).",
    "③ V_rot는 보간과 동률(1/4·2/4) — 정보 부재는 물리(NBI·Mirnov)이며 라우팅으로 구조화. 과대주장 안 함.",
    "④ 21k 해석가능 모델 = 백본(컷) · 폭 26× 스윕 평평 → 상한은 추정기가 아니라 데이터: 피팅 메타·kHz Mirnov·NBI 토크.",
    "⑤ 배치: seq_v2 상태 유지 1-step CPU 1.05 ms 중앙값 / p99 1.61 ms · conformal 구간 32/32 셀 우위(coverage marginal).",
]
yy = 0.124
for s in concl:
    txt(0.072, yy, s, 8.0, WHITE, va="center")
    yy -= 0.0165

# ============================ FOOTER ====================================
txt(0.045, 0.030, "수치 출처: docs/paper/paper_numbers.json · THESIS_RESULTS.md §8ab", 7.2, MGRAY)
txt(0.955, 0.030, "한계: MNAR 도달 $T_i$ 54–68%/V_rot 4–6% · test shot 96/60–66 · 캠페인 1 블록",
    7.2, MGRAY, ha="right")

out_pdf = os.path.join(HERE, "KSTAR_CES_1pager.pdf")
out_png = os.path.join(HERE, "KSTAR_CES_1pager.png")
fig.savefig(out_pdf, facecolor=WHITE)
fig.savefig(out_png, dpi=170, facecolor=WHITE)
plt.close(fig)
print("SAVED:", out_pdf)
print("SAVED:", out_png)
