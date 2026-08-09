# -*- coding: utf-8 -*-
"""Generate the final-model architecture diagram for the KSTAR CES nowcasting talk.

Draws the *thesis-final* model (ces_prediction/model_iter009.py):
per-diagnostic time-aware CNN encoders, a bidirectional-GRU history encoder with
observation-masked multi-head attention pooling, and physics-informed per-target
routing (the V_rot head never sees the fast diagnostics).

Design: light boxes (white fill + colored border), two tinted target lanes, the
history encoder centered between the lanes so its two summaries fan out
symmetrically — no crossing arrows, minimal text. The total parameter count is
measured from the archived iter009 module at build time (201,258 ≈ 0.20 M).

Output: docs/presentation/figures/fig_architecture.png
"""
import importlib.util
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

for cand in ["Malgun Gothic", "Malgun Gothic Semilight", "NanumGothic", "AppleGothic"]:
    try:
        matplotlib.font_manager.findfont(cand, fallback_to_default=False)
        plt.rcParams["font.family"] = cand
        break
    except Exception:
        continue
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["savefig.dpi"] = 220
plt.rcParams["figure.dpi"] = 220

NAVY = "#13335f"
BLUE = "#2b6cb0"
TEAL = "#1b9e8a"
ORANGE = "#e8743b"
GRAY = "#5b6670"
MGRAY = "#8d99a6"
ARROW = "#b9c4d0"
DARK = "#222b35"
TI_BG = "#fdf3ec"   # pale orange lane
VT_BG = "#edf3fa"   # pale blue lane

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "figures")
os.makedirs(OUT, exist_ok=True)

MODEL_PY = os.path.join(HERE, "..", "..", "ces_prediction", "model_iter009.py")


def count_params():
    spec = importlib.util.spec_from_file_location("m_iter009", MODEL_PY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    m = mod.MultimodalCESPredictor(window_size=4, bes_channels=9, ecei_channels=4,
                                   mc_channels=2, time_channels=4,
                                   ces_history_channels=4)
    return sum(p.numel() for p in m.parameters())


# ---- drawing helpers -----------------------------------------------------
def chip(ax, x, y, w, h, color, title, sub=None, alpha=0.13):
    """Input chip: soft tinted fill, colored title, gray subtitle."""
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.012,rounding_size=0.09",
                                fc=color, ec="none", alpha=alpha, zorder=3))
    cy = y + h / 2
    if sub:
        ax.text(x + w / 2, cy + 0.115, title, ha="center", va="center",
                fontsize=11, color=color, fontweight="bold", zorder=4)
        ax.text(x + w / 2, cy - 0.135, sub, ha="center", va="center",
                fontsize=8.4, color=GRAY, zorder=4)
    else:
        ax.text(x + w / 2, cy, title, ha="center", va="center",
                fontsize=11, color=color, fontweight="bold", zorder=4)


def block(ax, x, y, w, h, color, title, subs=(), fill="white", title_size=12,
          sub_size=8.8, lw=1.7):
    """Encoder/head block: white fill, colored border, colored title."""
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.012,rounding_size=0.07",
                                fc=fill, ec=color, lw=lw, zorder=3))
    n = len(subs)
    cy = y + h / 2 + 0.125 * n / 2
    ax.text(x + w / 2, cy, title, ha="center", va="center", fontsize=title_size,
            color=color, fontweight="bold", zorder=4)
    for i, ln in enumerate(subs):
        ax.text(x + w / 2, cy - 0.185 - 0.155 * i, ln, ha="center", va="center",
                fontsize=sub_size, color=GRAY, zorder=4)


def arrow(ax, x1, y1, x2, y2, color=ARROW, lw=1.5, rad=0.0, ls="-", z=2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=11, lw=lw, color=color,
                                 linestyle=ls, zorder=z, shrinkA=2, shrinkB=2,
                                 connectionstyle=f"arc3,rad={rad}"))


def build():
    total = count_params()
    fig, ax = plt.subplots(figsize=(13.4, 7.1))
    ax.set_xlim(0, 13.4)
    ax.set_ylim(0, 7.1)
    ax.axis("off")

    # ---- target lanes -----------------------------------------------------
    ax.add_patch(FancyBboxPatch((5.70, 3.95), 7.35, 2.80,
                                boxstyle="round,pad=0.02,rounding_size=0.12",
                                fc=TI_BG, ec="none", zorder=1))
    ax.add_patch(FancyBboxPatch((5.70, 0.35), 7.35, 2.80,
                                boxstyle="round,pad=0.02,rounding_size=0.12",
                                fc=VT_BG, ec="none", zorder=1))
    ax.text(6.00, 6.42, "$T_i$ 경로", fontsize=13.5, color=ORANGE,
            fontweight="bold", zorder=4)
    ax.text(7.35, 6.42, "빠른진단 + 시간 + 이력", fontsize=9.5, color=MGRAY,
            va="center", zorder=4)
    ax.text(6.00, 2.83, "V_rot 경로", fontsize=13.5, color=BLUE,
            fontweight="bold", zorder=4)
    ax.text(7.55, 2.83, "이력 + 시간만  ·  빠른진단 미사용 (물리 기반 라우팅)",
            fontsize=9.5, color=MGRAY, va="center", zorder=4)

    # ---- column headers ---------------------------------------------------
    for x, t in [(0.40, "입 력"), (3.05, "진단별 인코더"), (6.00, "타겟별 융합 · 예측"),
                 (11.85, "출 력")]:
        ax.text(x, 6.90, t, fontsize=10.5, color=MGRAY, fontweight="bold")
    ax.text(3.05, 6.66, "입력 = 시점별 [신호 + time + history] 채널 결합",
            fontsize=8.2, color=MGRAY, zorder=4)

    # ---- inputs (chips) ---------------------------------------------------
    CX, CW, CH = 0.40, 1.90, 0.62
    chip(ax, CX, 5.90, CW, CH, BLUE, "BES  4×9", r"밀도요동 $\tilde{n}_e$")
    chip(ax, CX, 5.13, CW, CH, TEAL, "ECEI  4×4", r"전자온도 $T_e$")
    chip(ax, CX, 4.36, CW, CH, GRAY, "MC  4×2", "자기요동")
    chip(ax, CX, 3.24, CW, CH, ORANGE, "ces_history  4×4", "과거 CES + 관측 flag")
    chip(ax, CX, 2.10, CW, CH, NAVY, "time  4×4", "lookback · Δt · log1p")
    ax.text(CX + CW / 2, 3.02, "타겟 시점은 완전 마스킹", ha="center",
            fontsize=8.0, color=MGRAY, style="italic", zorder=4)

    # ---- encoders ---------------------------------------------------------
    EX, EW = 3.05, 2.25
    block(ax, EX, 5.90, EW, CH, BLUE, "BES Enc", ["time-aware CNN → 96"])
    block(ax, EX, 5.13, EW, CH, TEAL, "ECEI Enc", ["time-aware CNN → 96"])
    block(ax, EX, 4.36, EW, CH, GRAY, "MC Enc", ["time-aware CNN → 96"])
    block(ax, EX, 2.93, EW, 1.24, ORANGE, "History Enc",
          ["양방향 GRU  h=64", "관측-마스크 attention pool"], title_size=12.5)
    block(ax, EX, 2.10, EW, CH, NAVY, "Time Enc", ["1D CNN → 32"])
    for yc in (6.21, 5.44, 4.67, 3.55, 2.41):
        arrow(ax, CX + CW, yc, EX, yc)

    # ---- fusion nodes + heads --------------------------------------------
    block(ax, 6.00, 4.75, 1.55, 0.85, ORANGE, "concat", ["384-d"], fill=TI_BG, lw=1.4)
    block(ax, 6.00, 1.45, 1.55, 0.85, BLUE, "concat", ["96-d"], fill=VT_BG, lw=1.4)
    block(ax, 8.45, 4.62, 2.45, 1.10, ORANGE, "$T_i$  Head",
          ["MLP  384 → 160 → 64 → 1"], title_size=13)
    block(ax, 8.45, 1.32, 2.45, 1.10, BLUE, "V_rot  Head",
          ["MLP  96 → 96 → 48 → 1"], title_size=13)
    arrow(ax, 7.55, 5.17, 8.45, 5.17, color=ORANGE, lw=1.8)
    arrow(ax, 7.55, 1.87, 8.45, 1.87, color=BLUE, lw=1.8)

    # fast diagnostics -> Ti concat only (gentle fan)
    for yc in (6.21, 5.44, 4.67):
        arrow(ax, EX + EW, yc, 6.00, 5.30, rad=-0.10)

    # history -> both lanes, symmetric fan (the story)
    arrow(ax, EX + EW, 3.80, 6.00, 4.98, color=ORANGE, lw=2.0, rad=-0.22)
    arrow(ax, EX + EW, 3.30, 6.00, 2.12, color=BLUE, lw=2.0, rad=0.22)
    ax.text(6.10, 4.35, "hist$_{T_i}$ 64", fontsize=8.6, color=ORANGE,
            ha="left", zorder=4)
    ax.text(6.10, 2.50, "hist$_{V\\!rot}$ 64", fontsize=8.6, color=BLUE,
            ha="left", zorder=4)

    # time -> both concats (thin, dashed)
    arrow(ax, EX + EW, 2.41, 6.20, 4.75, rad=-0.30, ls=(0, (3, 3)), lw=1.2)
    arrow(ax, EX + EW, 2.35, 6.20, 1.45, rad=0.18, ls=(0, (3, 3)), lw=1.2)

    # ---- output -----------------------------------------------------------
    ax.add_patch(FancyBboxPatch((11.55, 3.05), 1.50, 1.00,
                                boxstyle="round,pad=0.012,rounding_size=0.08",
                                fc=NAVY, ec="none", zorder=3))
    ax.text(12.30, 3.72, "[$T_i$, V_rot]", ha="center", va="center",
            fontsize=12.5, color="white", fontweight="bold", zorder=4)
    ax.text(12.30, 3.38, "정규화 출력 (B, 2)", ha="center", va="center",
            fontsize=8.6, color="#cdd8e4", zorder=4)
    arrow(ax, 10.90, 5.17, 12.10, 4.05, color=ORANGE, lw=1.8, rad=-0.20)
    arrow(ax, 10.90, 1.87, 12.10, 3.05, color=BLUE, lw=1.8, rad=0.20)

    # ---- footer -----------------------------------------------------------
    ax.text(0.40, 0.55, f"총 파라미터  {total:,}개  (~0.2 M)", fontsize=11,
            color=NAVY, fontweight="bold")
    ax.text(0.40, 0.22,
            "window = 4  ·  attention pool은 관측된 행에만 softmax 허용  ·  "
            "구조는 ~40회 통제실험(keep/discard) 탐색의 결과",
            fontsize=9.0, color=MGRAY)

    path = os.path.join(OUT, "fig_architecture.png")
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", path, "| total params:", total)


if __name__ == "__main__":
    build()
