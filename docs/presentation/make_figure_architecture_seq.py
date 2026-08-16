# -*- coding: utf-8 -*-
"""Architecture diagram of the adopted backbone (seq_v2) for the talk.

Draws ces_prediction/experiments/seq/model_seq_v2.py: the full-grid 22-channel causal
sequence, two independent causal LSTM branches -- the T_i branch reads the full state,
the V_rot branch reads ONLY the 7 non-fast channels (routing at the encoder, verified
bit-identical under fast-channel perturbation) -- LayerNorm + GELU heads, per-row masked
loss. Same visual family as make_figure_architecture.py (which now draws the W = 2
windowed control). The parameter count is measured from the module at build time.

Output: docs/presentation/figures/fig_architecture_seq.png
"""
import os
import sys

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
TI_BG = "#fdf3ec"
VT_BG = "#edf3fa"

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "figures")
os.makedirs(OUT, exist_ok=True)
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "ces_prediction"))
sys.path.insert(0, os.path.join(ROOT, "ces_prediction", "experiments", "seq"))


def count_params():
    from model_seq_v2 import SeqCESLSTMv2  # noqa: E402
    return SeqCESLSTMv2().n_params


def chip(ax, x, y, w, h, color, title, sub=None, alpha=0.13):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.09",
                                fc=color, ec="none", alpha=alpha, zorder=3))
    cy = y + h / 2
    if sub:
        ax.text(x + w / 2, cy + 0.115, title, ha="center", va="center", fontsize=11, color=color, fontweight="bold", zorder=4)
        ax.text(x + w / 2, cy - 0.135, sub, ha="center", va="center", fontsize=8.4, color=GRAY, zorder=4)
    else:
        ax.text(x + w / 2, cy, title, ha="center", va="center", fontsize=11, color=color, fontweight="bold", zorder=4)


def block(ax, x, y, w, h, color, title, subs=(), fill="white", title_size=12, sub_size=8.8, lw=1.7):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.07",
                                fc=fill, ec=color, lw=lw, zorder=3))
    n = len(subs)
    cy = y + h / 2 + 0.125 * n / 2
    ax.text(x + w / 2, cy, title, ha="center", va="center", fontsize=title_size, color=color, fontweight="bold", zorder=4)
    for i, ln in enumerate(subs):
        ax.text(x + w / 2, cy - 0.185 - 0.155 * i, ln, ha="center", va="center", fontsize=sub_size, color=GRAY, zorder=4)


def arrow(ax, x1, y1, x2, y2, color=ARROW, lw=1.5, rad=0.0, ls="-", z=2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=11, lw=lw, color=color,
                                 linestyle=ls, zorder=z, shrinkA=2, shrinkB=2, connectionstyle=f"arc3,rad={rad}"))


def build():
    total = count_params()
    fig, ax = plt.subplots(figsize=(13.4, 7.1))
    ax.set_xlim(0, 13.4)
    ax.set_ylim(0, 7.1)
    ax.axis("off")

    # lanes
    ax.add_patch(FancyBboxPatch((5.70, 3.95), 7.35, 2.80, boxstyle="round,pad=0.02,rounding_size=0.12", fc=TI_BG, ec="none", zorder=1))
    ax.add_patch(FancyBboxPatch((5.70, 0.35), 7.35, 2.80, boxstyle="round,pad=0.02,rounding_size=0.12", fc=VT_BG, ec="none", zorder=1))
    ax.text(6.00, 6.42, "$T_i$ 분기", fontsize=13.5, color=ORANGE, fontweight="bold", zorder=4)
    ax.text(7.25, 6.42, "22채널 전체 상태 (빠른진단 + 이월값 + 신선도 + Δt)", fontsize=9.5, color=MGRAY, va="center", zorder=4)
    ax.text(6.00, 2.83, "V_rot 분기", fontsize=13.5, color=BLUE, fontweight="bold", zorder=4)
    ax.text(7.55, 2.83, "비-빠른 7채널만 · 인코더 수준 라우팅 (빠른진단 섭동 → 출력 bit-identical)", fontsize=9.0, color=MGRAY, va="center", zorder=4)

    for x, t in [(0.40, "입 력  (격자 시퀀스, 스텝당 22채널)"), (6.00, "인과 LSTM 분기 · 예측"), (11.85, "출 력")]:
        ax.text(x, 6.90, t, fontsize=10.5, color=MGRAY, fontweight="bold")
    ax.text(0.40, 6.66, "세그먼트의 모든 행(라벨 없어도)이 맥락 · 타겟 행 자체 값은 입력에 없음", fontsize=8.2, color=MGRAY, zorder=4)

    # input chips (feature layout: [fast 15 | dt | TI(carried, stale, has) | VT(carried, stale, has)])
    CX, CW, CH = 0.40, 2.30, 0.62
    chip(ax, CX, 5.80, CW, CH, BLUE, "BES  L×9", r"밀도요동 $\tilde{n}_e$ · shot별 z-score")
    chip(ax, CX, 5.05, CW, CH, TEAL, "ECEI  L×4", r"전자온도 $T_e$ · shot별 z-score")
    chip(ax, CX, 4.30, CW, CH, GRAY, "MC  L×2", "자기요동 · shot별 z-score")
    chip(ax, CX, 2.95, CW, CH, ORANGE, "T_i 이월값·신선도·flag  L×3", "직전 관측 T_i (z), log1p 경과, 관측 여부")
    chip(ax, CX, 2.20, CW, CH, BLUE, "V_rot 이월값·신선도·flag  L×3", "직전 관측 V_rot (z), log1p 경과, 관측 여부")
    chip(ax, CX, 1.45, CW, CH, NAVY, "Δt  L×1", "log1p(행간 간격)")
    ax.add_patch(FancyBboxPatch((CX - 0.08, 4.18), CW + 0.16, 2.35, boxstyle="round,pad=0.01,rounding_size=0.06",
                                fc="none", ec=MGRAY, lw=0.9, ls=(0, (3, 3)), zorder=2))
    ax.text(CX + CW / 2, 3.98, "빠른 채널 15  (V_rot 분기에는 도달하지 않음)", ha="center", fontsize=8.0, color=MGRAY, style="italic", zorder=4)
    ax.add_patch(FancyBboxPatch((CX - 0.08, 1.33), CW + 0.16, 2.35, boxstyle="round,pad=0.01,rounding_size=0.06",
                                fc="none", ec=MGRAY, lw=0.9, ls=(0, (3, 3)), zorder=2))
    ax.text(CX + CW / 2, 1.13, "비-빠른 채널 7  (두 분기 모두)", ha="center", fontsize=8.0, color=MGRAY, style="italic", zorder=4)

    # branches
    block(ax, 6.00, 4.55, 2.55, 1.30, ORANGE, "causal LSTM ×2", ["hidden 160 · dropout 0.1", "→ LayerNorm"], title_size=12.5)
    block(ax, 6.00, 1.25, 2.55, 1.30, BLUE, "causal LSTM ×1", ["hidden 64", "→ LayerNorm"], title_size=12.5)
    block(ax, 9.05, 4.70, 1.85, 1.00, ORANGE, "$T_i$  Head", ["Linear 160→64 · GELU · →1"], title_size=12.5)
    block(ax, 9.05, 1.40, 1.85, 1.00, BLUE, "V_rot  Head", ["Linear 64→64 · GELU · →1"], title_size=12.5)
    arrow(ax, 8.55, 5.20, 9.05, 5.20, color=ORANGE, lw=1.8)
    arrow(ax, 8.55, 1.90, 9.05, 1.90, color=BLUE, lw=1.8)

    # inputs -> branches: fast (3 chips) -> T_i only; non-fast (3 chips) -> both
    for yc in (6.11, 5.36, 4.61):
        arrow(ax, CX + CW, yc, 6.00, 5.45, rad=-0.10)
    for yc in (3.26, 2.51, 1.76):
        arrow(ax, CX + CW, yc, 6.00, 4.95, color=ORANGE, lw=1.4, rad=-0.18)
        arrow(ax, CX + CW, yc, 6.00, 1.90, color=BLUE, lw=1.4, rad=0.10)

    # outputs
    ax.add_patch(FancyBboxPatch((11.55, 3.35), 1.50, 1.00, boxstyle="round,pad=0.012,rounding_size=0.08", fc=NAVY, ec="none", zorder=3))
    ax.text(12.30, 4.02, "[$T_i$, V_rot]", ha="center", va="center", fontsize=12.5, color="white", fontweight="bold", zorder=4)
    ax.text(12.30, 3.68, "행마다 정규화 출력 (L, 2)", ha="center", va="center", fontsize=8.6, color="#cdd8e4", zorder=4)
    arrow(ax, 10.90, 5.20, 12.10, 4.35, color=ORANGE, lw=1.8, rad=-0.20)
    arrow(ax, 10.90, 1.90, 12.10, 3.35, color=BLUE, lw=1.8, rad=0.20)
    ax.text(9.05, 1.12, "loss = 타겟별 masked MSE (관측된 행만 지도)", fontsize=8.6, color=NAVY, ha="left", va="top", zorder=4)

    ax.text(0.40, 0.72, f"총 파라미터  {total:,}개  (~0.36 M)  ·  W는 하이퍼파라미터가 아님(도달거리 = 세그먼트 전체)",
            fontsize=11, color=NAVY, fontweight="bold")
    ax.text(0.40, 0.38, "학습: AdamW 1e-3 · batch 16 세그먼트 · val masked MSE patience 6 · held-free · 컷/포함 모집단별 · 온라인은 은닉상태 이월 1-step",
            fontsize=9.0, color=MGRAY)
    ax.text(0.40, 0.10, "B.1 관문: 윈도 대조군(W=2) 대비 paired T_i +0.081 pooled (16/16 양수, CI [+0.067, +0.096]) · V_rot 유의 열세 0/16 → 백본 채택 (THESIS §8x)",
            fontsize=8.6, color=MGRAY)

    path = os.path.join(OUT, "fig_architecture_seq.png")
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", path, "| total params:", total)


if __name__ == "__main__":
    build()
