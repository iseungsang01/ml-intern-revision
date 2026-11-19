# -*- coding: utf-8 -*-
"""held 제거 학습(stuck-free) 실험 forest plot (paired vs held-kept iter009, 4 seeds).

Numbers: data/.sf_iter009_s*/paired_vs_heldkept_iter009.json (THESIS_RESULTS.md §8c).
Output: docs/presentation/figures/fig_stuckfree_paired.png
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

# (seed, point, lo, hi) — paired skill, A = held-free 학습, B = held-kept 학습 (동일 iter009)
VT = [(42, 0.030, 0.001, 0.052), (1, 0.047, 0.012, 0.081),
      (7, 0.048, -0.015, 0.152), (123, 0.032, 0.006, 0.057)]
TI = [(42, 0.072, 0.012, 0.137), (1, -0.016, -0.071, 0.046),
      (7, 0.016, -0.025, 0.067), (123, -0.057, -0.163, 0.012)]


def verdict(lo, hi):
    if lo > 0:
        return GREEN, "유의 개선"
    if hi < 0:
        return RED, "유의 악화"
    return GRAY, "n.s."


def panel(ax, rows, title, xlabel=False):
    ax.axvline(0.0, color=NAVY, lw=1.2, ls="--", alpha=0.7)
    for i, (seed, pt, lo, hi) in enumerate(rows):
        y = len(rows) - 1 - i
        col, label = verdict(lo, hi)
        ax.plot([lo, hi], [y, y], color=col, lw=2.6, solid_capstyle="round", zorder=2)
        for x in (lo, hi):
            ax.plot([x, x], [y - 0.10, y + 0.10], color=col, lw=2.0, zorder=2)
        ax.plot(pt, y, "o", ms=9, color=col, mec="white", mew=1.4, zorder=3)
        ax.annotate(f"{pt:+.3f}  {label}", (pt, y), textcoords="offset points",
                    xytext=(0, 10), ha="center", fontsize=10.5, color=col,
                    fontweight="bold")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"seed {r[0]}" for r in reversed(rows)], fontsize=11)
    ax.set_ylim(-0.5, len(rows) - 0.10)
    ax.set_xlim(-0.19, 0.19)
    ax.set_title(title, fontsize=13, fontweight="bold", color=NAVY, pad=8, loc="left")
    ax.grid(axis="x", color=LGRAY, lw=0.8)
    ax.set_axisbelow(True)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(axis="x", labelsize=9.5)
    if xlabel:
        ax.set_xlabel("paired skill (held-free 학습 vs held-kept 학습)  —  0보다 크면 held 제거가 우세",
                      fontsize=10.5, color=GRAY)


fig, axes = plt.subplots(2, 1, figsize=(7.4, 5.6))
panel(axes[0], VT, "CES_VT  (토로이달 회전) — 4/4 양수 · 3/4 유의")
panel(axes[1], TI, "CES_TI  (이온온도) — 손해 없음", xlabel=True)
fig.suptitle("동일 모델(iter009)·동일 split, held 포함 여부만 변경 — 4-seed paired 95% CI",
             fontsize=12, fontweight="bold", color=NAVY, y=0.995)
fig.tight_layout(rect=[0, 0, 1, 0.97])
out = os.path.join(os.path.dirname(__file__), "figures", "fig_stuckfree_paired.png")
fig.savefig(out, bbox_inches="tight", facecolor="#ffffff")
print("wrote", out)
