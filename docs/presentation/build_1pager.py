# -*- coding: utf-8 -*-
"""Build a single-page A4 one-pager (PDF + PNG) summarizing the KSTAR CES project.

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
txt(0.045, 0.987, "KSTAR 다중 진단 기반 Multimodal 신경망을 활용한", 14.5, WHITE, "bold")
txt(0.045, 0.963, "CES 결측 구간 예측 (Gap-filling / Nowcasting)", 16.5, WHITE, "bold")
txt(0.955, 0.985, "1-PAGE 요약", 10.5, ORANGE, "bold", ha="right")
txt(0.955, 0.962, "이승상 · 서울대 원자핵공학", 9, LGRAY, ha="right")
txt(0.955, 0.944, "KSTAR Charge Exchange Spectroscopy", 8, MGRAY, ha="right")

# ============================ ONE-LINE THESIS ============================
rect(0.045, 0.870, 0.91, 0.050, CARD, round_=True, pad=0.008)
rect(0.045, 0.870, 0.012, 0.050, ORANGE, round_=True, pad=0.004)
txt(0.072, 0.909, "한 줄 요약", 8.3, ORANGE, "bold")
txt(0.072, 0.892,
    "항상 조밀한 빠른 진단(BES·ECEI·Mirnov)과 과거 CES 이력으로 CES 결측 10 ms 시점의 Tᵢ·V_rot를 복원하고,",
    9.0, DARK)
txt(0.072, 0.876,
    "미래까지 보는 오프라인 보간(interpolation)을 causal 모델로 이기는가를 통계적으로 검증한다.",
    9.0, DARK)

# ============================ PROBLEM / IDEA (2 cols) =====================
ytop = 0.858
txt(0.045, ytop, "■ 문제 & 핵심 아이디어", 11, NAVY, "bold")
# left column text
prob = [
    ("CES는 SNR 확보를 위해 광자를 오래 수집 → 자주 결측", GRAY),
    ("같은 10 ms 격자에서 Tᵢ ≈ 8%, V_rot ≈ 24% 결측 (독립적)", GRAY),
    ("빠른 진단(BES·ECEI·MC)은 항상 100% 조밀하게 측정됨", GRAY),
    ("→ \"항상 있는 빠른 진단\"으로 \"자주 비는 CES\"를 채움", ORANGE),
    ("강한 역산 가정 없는 데이터 기반 가상 센서 (축대칭 수준만 가정)", BLUE),
]
yy = ytop - 0.022
for s, c in prob:
    txt(0.052, yy, "•", 9.5, c, "bold")
    txt(0.069, yy, s, 9.0, c, "bold" if c != GRAY else "normal")
    yy -= 0.0205

# ============================ KEY RESULT STATS ===========================
txt(0.045, 0.715, "■ 핵심 결과 (held-out TEST, 4 seed)", 11, NAVY, "bold")
stats = [
    ("CES_TI", "+0.20~+0.30", "skill_vs_pchip · 4 seed 모두\n95% CI > 0 → PASS (강건)", GREEN),
    ("CES_VT", "n.s.", "보간과 동률 (point est. +)\nTᵢ↔V_rot 비대칭", GRAY),
    ("vs causal", "압도", "persistence·AR 대비 큰 마진\nTᵢ 369 vs 487 / 1006", ORANGE),
    ("Peak 구간", "+0.86 / +0.69", "고변동 구간 skill (Tᵢ/V_rot)\n둘 다 PASS", TEAL),
]
sw, sgap = 0.218, 0.0125
sx = 0.045
sy = 0.638
for name, big, sub, col in stats:
    rect(sx, sy, sw, 0.058, CARD, round_=True, pad=0.008)
    rect(sx, sy + 0.046, sw, 0.012, col, round_=True, pad=0.005)
    txt(sx + sw / 2, sy + 0.0495, name, 8.5, WHITE, "bold", ha="center")
    txt(sx + sw / 2, sy + 0.032, big, 13.5, col, "bold", ha="center")
    txt(sx + sw / 2, sy + 0.016, sub, 7.3, GRAY, ha="center", spacing=1.12)
    sx += sw + sgap

# ============================ METHOD =====================================
txt(0.045, 0.612, "■ 방법", 11, NAVY, "bold")
meth = [
    ("데이터", "H-mode ELM-suppression, #24000–33000, 641 shot CSV · No-Fake-Data · per-target masked MSE"),
    ("전처리", "file-level split(행 누수 차단) · train-file-only z-score(NaN-aware) · 타겟 시점 완전 마스킹(누수 차단)"),
    ("모델", "진단별 time-aware CNN + Pre-LN Transformer 이력 인코더 + target별 multi-head attention head (<1M params)"),
    ("평가", "3-way split(test는 선택에 미사용) · 물리단위 per-target RMSE · shot-clustered paired bootstrap(B=10k)"),
    ("탐색", "Claude 기반 keep/discard autoresearch — clean skill로 채점, 회귀는 자동 롤백 (n.s.→유의로 개선)"),
]
yy = 0.590
for tag, body in meth:
    rect(0.045, yy - 0.0035, 0.075, 0.016, BLUE, round_=True, pad=0.004)
    txt(0.0825, yy + 0.0045, tag, 8.2, WHITE, "bold", ha="center", va="center")
    txt(0.130, yy + 0.0045, body, 8.6, DARK, va="center")
    yy -= 0.0212

# ============================ RESULTS FIGURE =============================
txt(0.045, 0.470, "■ headline — 보간(PCHIP) 대비 skill, shot-clustered 95% CI", 11, NAVY, "bold")
try:
    img = mpimg.imread(os.path.join(FIG, "fig_forest.png"))
    ax = fig.add_axes([0.045, 0.250, 0.91, 0.205])
    ax.imshow(img)
    ax.axis("off")
except Exception as e:
    txt(0.5, 0.35, f"[forest figure missing: {e}]", 9, RED, ha="center")

# ============================ ASYMMETRY / PHYSICS =======================
txt(0.045, 0.232, "■ 과학적 발견: Tᵢ ↔ V_rot 비대칭", 11, NAVY, "bold")
rect(0.045, 0.150, 0.445, 0.072, CARD, round_=True, pad=0.008)
rect(0.045, 0.150, 0.012, 0.072, ORANGE, round_=True, pad=0.004)
txt(0.072, 0.214, "Tᵢ — 빠른 진단이 정보 운반", 9.2, ORANGE, "bold")
txt(0.072, 0.198, "충돌 e–i 결합(t_ei ∝ Tₑ^1.5/nₑ)로 ECEI(Tₑ)+BES(nₑ)가", 8.2, DARK)
txt(0.072, 0.184, "Tᵢ 단서 운반 · fast-only도 persistence 능가(+0.16)", 8.2, DARK)
txt(0.072, 0.168, "peak에서 빠른진단 제거 시 큰 손해(유의) → 실제 사용 확인", 8.2, GREEN, "bold")

rect(0.510, 0.150, 0.445, 0.072, CARD, round_=True, pad=0.008)
rect(0.510, 0.150, 0.012, 0.072, BLUE, round_=True, pad=0.004)
txt(0.537, 0.214, "V_rot — 정보는 거의 전적으로 과거 이력", 9.2, BLUE, "bold")
txt(0.537, 0.198, "토로이달 회전은 미관측 NBI 토크가 주도 ·", 8.2, DARK)
txt(0.537, 0.184, "Mirnov은 100 Hz로 aliasing → 회전 정보 소실", 8.2, DARK)
txt(0.537, 0.168, "fast-only V_rot = -3.31 (평균예측보다 나쁨) → 비-승리는 발견", 8.2, RED, "bold")

# ============================ CONCLUSION ================================
rect(0.045, 0.052, 0.91, 0.088, NAVY, round_=True, pad=0.008)
txt(0.072, 0.128, "정직한 결론", 9.5, ORANGE, "bold")
concl = [
    "① causal baseline(persistence·AR)을 큰 마진으로 압도 — 온라인/실시간에서 명확한 승자 (강건).",
    "② CES_TI는 미래까지 보는 오프라인 보간도 통계적으로 유의하게 능가 (4 seed 모두 PASS, genuine-only도 강건).",
    "③ CES_VT는 보간과 동률(n.s.) — 검정력 한계(≈91 shot)와 heavy-tail이 구속, 과대주장 안 함.",
    "④ 모델의 가치는 고변동(peak) 구간에 집중 · Tᵢ↔V_rot 비대칭은 물리로 예측되고 ablation으로 확인됨.",
]
yy = 0.110
for s in concl:
    txt(0.072, yy, s, 8.5, WHITE, va="center")
    yy -= 0.0175

# ============================ FOOTER ====================================
txt(0.045, 0.030, "KSTAR CES Nowcasting · 다중진단 기반 CES 결측 구간 예측", 7.5, MGRAY)
txt(0.955, 0.030, "한계: observed-only(MNAR 낙관적 상한) · window=4 · 단일 아키텍처 · test shot ≈ 96",
    7.5, MGRAY, ha="right")

out_pdf = os.path.join(HERE, "KSTAR_CES_1pager.pdf")
out_png = os.path.join(HERE, "KSTAR_CES_1pager.png")
fig.savefig(out_pdf, facecolor=WHITE)
fig.savefig(out_png, dpi=170, facecolor=WHITE)
plt.close(fig)
print("SAVED:", out_pdf)
print("SAVED:", out_png)
