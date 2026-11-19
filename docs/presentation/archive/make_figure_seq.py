# -*- coding: utf-8 -*-
"""seq-LSTM 재프레이밍 실험 forest plot (paired vs iter009, 4 seeds).

Numbers: data/.seq_lstm_s*/paired_vs_iter009.json (THESIS_RESULTS.md §8d row 1).
Output: docs/presentation/figures/fig_seq_paired.png
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

for cand in ["Malgun Gothic", "NanumGothic", "AppleGothic"]:
    try:
        matplotlib.font_manager.findfont(cand, fallback_to_default=False)
        plt.rcParams["font.family"] = cand
        break
    except Exception:
        continue
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["savefig.dpi"] = 200

NAVY = "#13335f"
GREEN = "#2e9e5b"
RED = "#c0392b"
GRAY = "#7b8794"
LGRAY = "#dfe4ea"

# (seed, point, lo, hi) — paired skill_A_vs_B, A = seq-LSTM, B = iter009
TI = [(42, 0.078, -0.010, 0.162), (1, 0.030, -0.015, 0.086),
      (7, 0.011, -0.038, 0.075), (123, 0.049, 0.012, 0.080)]
VT = [(42, -0.103, -0.231, -0.021), (1, -0.048, -0.106, -0.003),
      (7, -0.061, -0.202, 0.051), (123, 0.017, -0.015, 0.035)]


def verdict(lo, hi):
    if lo > 0:
        return GREEN, "유의 개선"
    if hi < 0:
        return RED, "유의 악화"
    return GRAY, "n.s."


def panel(ax, rows, title):
    ax.axvline(0.0, color=NAVY, lw=1.2, ls="--", alpha=0.7)
    for i, (seed, pt, lo, hi) in enumerate(rows):
        y = len(rows) - 1 - i
        col, label = verdict(lo, hi)
        ax.plot([lo, hi], [y, y], color=col, lw=2.6, solid_capstyle="round",
                zorder=2)
        for x in (lo, hi):
            ax.plot([x, x], [y - 0.10, y + 0.10], color=col, lw=2.0, zorder=2)
        ax.plot(pt, y, "o", ms=9, color=col, mec="white", mew=1.4, zorder=3)
        ax.annotate(f"{pt:+.3f}  {label}", (pt, y), textcoords="offset points",
                    xytext=(0, 11), ha="center", fontsize=10.5, color=col,
                    fontweight="bold")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"seed {r[0]}" for r in reversed(rows)], fontsize=11.5)
    ax.set_ylim(-0.55, len(rows) - 0.15)
    ax.set_xlim(-0.27, 0.20)
    ax.set_title(title, fontsize=13.5, fontweight="bold", color=NAVY, pad=10)
    ax.grid(axis="x", color=LGRAY, lw=0.8)
    ax.set_axisbelow(True)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(axis="x", labelsize=10)
    ax.set_xlabel("paired skill (seq-LSTM vs 최종모델 iter009)  —  0보다 크면 seq가 우세",
                  fontsize=10.5, color=GRAY)


fig, axes = plt.subplots(1, 2, figsize=(11.6, 3.6), sharey=False)
panel(axes[0], TI, "CES_TI  (이온온도)")
panel(axes[1], VT, "CES_VT  (토로이달 회전)")
fig.suptitle("4-seed paired, shot-clustered bootstrap 95% CI — T_i는 4/4 양수, V_rot은 2/4 유의 악화",
             fontsize=12.5, fontweight="bold", color=NAVY, y=1.04)
fig.tight_layout()
out = os.path.join(os.path.dirname(__file__), "figures", "fig_seq_paired.png")
fig.savefig(out, bbox_inches="tight", facecolor="#ffffff")
print("wrote", out)
