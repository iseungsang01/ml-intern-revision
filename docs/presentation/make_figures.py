# -*- coding: utf-8 -*-
"""Generate all presentation figures for the KSTAR CES nowcasting talk.

All numbers are read from / consistent with the frozen artifacts:
  data/.improve_final_out, data/.ms_out_{1,7,123}, data/.final_out
and the project docs (THESIS_RESULTS.md, PROJECT_KNOWLEDGE.md).
Output: docs/presentation/figures/*.png
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# Korean-capable font (Windows)
for cand in ["Malgun Gothic", "Malgun Gothic Semilight", "NanumGothic", "AppleGothic"]:
    try:
        matplotlib.font_manager.findfont(cand, fallback_to_default=False)
        plt.rcParams["font.family"] = cand
        break
    except Exception:
        continue
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["savefig.dpi"] = 200
plt.rcParams["figure.dpi"] = 200

NAVY = "#13335f"
BLUE = "#2b6cb0"
TEAL = "#1b9e8a"
ORANGE = "#e8743b"
GREEN = "#2e9e5b"
RED = "#c0392b"
GRAY = "#7b8794"
LGRAY = "#dfe4ea"
BG = "#ffffff"

OUT = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(OUT, exist_ok=True)


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print("wrote", name)


# ----------------------------------------------------------------------------
# 1) Headline forest plot: per-seed skill_vs_pchip with shot-clustered 95% CI
# ----------------------------------------------------------------------------
def fig_forest():
    seeds = ["seed 42", "seed 1", "seed 7", "seed 123"]
    ti = [0.270, 0.200, 0.271, 0.295]
    ti_lo = [0.144, 0.088, 0.171, 0.155]
    ti_hi = [0.364, 0.283, 0.346, 0.373]
    vt = [0.161, 0.117, 0.078, 0.144]
    vt_lo = [-0.382, -0.055, -0.596, -0.420]
    vt_hi = [0.334, 0.226, 0.279, 0.240]

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    for ax, vals, lo, hi, title, pass_all in [
        (axes[0], ti, ti_lo, ti_hi, "CES_TI  (이온 온도)", True),
        (axes[1], vt, vt_lo, vt_hi, "CES_VT  (토로이달 회전)", False),
    ]:
        y = np.arange(len(seeds))[::-1]
        for i, (v, l, h) in enumerate(zip(vals, lo, hi)):
            passed = l > 0
            c = GREEN if passed else GRAY
            ax.plot([l, h], [y[i], y[i]], color=c, lw=3, solid_capstyle="round", zorder=2)
            ax.scatter([v], [y[i]], color=c, s=90, zorder=3, edgecolor="white", linewidth=1.2)
            tag = "PASS" if passed else "n.s."
            ax.text(h + 0.03, y[i], f"+{v:.2f}  {tag}", va="center", ha="left",
                    fontsize=10.5, color=c, fontweight="bold")
        ax.axvline(0, color=RED, lw=1.6, ls="--", zorder=1)
        ax.set_yticks(y)
        ax.set_yticklabels(seeds, fontsize=11)
        ax.set_xlim(-0.75, 0.65)
        ax.set_xlabel("skill_vs_pchip  (= 1 - MSE_model / MSE_pchip)", fontsize=10)
        ax.set_title(title, fontsize=13, fontweight="bold", color=NAVY, pad=8)
        ax.grid(axis="x", color=LGRAY, lw=0.8)
        for s in ["top", "right", "left"]:
            ax.spines[s].set_visible(False)
    axes[0].text(0.02, 1.0, "4개 독립 test split 모두 95% CI > 0  →  PASS",
                 transform=axes[0].transAxes, fontsize=10, color=GREEN, fontweight="bold")
    axes[1].text(0.02, 1.0, "4개 split 모두 CI가 0을 포함  →  n.s.",
                 transform=axes[1].transAxes, fontsize=10, color=GRAY, fontweight="bold")
    fig.suptitle("보간(PCHIP) 대비 skill — shot-clustered 95% CI (B=10,000)",
                 fontsize=14, fontweight="bold", color=NAVY, y=1.04)
    fig.tight_layout()
    save(fig, "fig_forest.png")


# ----------------------------------------------------------------------------
# 2) RMSE ladder (physical units) — final model vs baseline ladder
# ----------------------------------------------------------------------------
def fig_rmse_ladder():
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))
    data = {
        "CES_TI": dict(
            names=["AR (past only)", "Persistence", "PCHIP*", "Linear", "Model (nowcaster)"],
            vals=[1005.66, 487.31, 431.81, 422.66, 368.86],
            unit="물리단위 RMSE", title="CES_TI  (n=32,716)"),
        "CES_VT": dict(
            names=["AR (past only)", "Persistence", "PCHIP*", "Linear", "Model (nowcaster)"],
            vals=[57.23, 27.77, 24.49, 24.01, 22.44],
            unit="물리단위 RMSE", title="CES_VT  (n=27,437)"),
    }
    for ax, (tgt, d) in zip(axes, data.items()):
        colors = [GRAY, GRAY, BLUE, GRAY, ORANGE]
        y = np.arange(len(d["names"]))
        bars = ax.barh(y, d["vals"], color=colors, edgecolor="white", height=0.62)
        ax.set_yticks(y)
        ax.set_yticklabels(d["names"], fontsize=10)
        for b, v in zip(bars, d["vals"]):
            ax.text(b.get_width() * 1.01, b.get_y() + b.get_height() / 2,
                    f"{v:.0f}" if v > 100 else f"{v:.1f}", va="center", ha="left",
                    fontsize=10, color=NAVY, fontweight="bold")
        ax.set_title(d["title"], fontsize=12.5, fontweight="bold", color=NAVY)
        ax.set_xlabel(d["unit"], fontsize=10)
        ax.set_xlim(0, max(d["vals"]) * 1.18)
        for s in ["top", "right"]:
            ax.spines[s].set_visible(False)
        ax.grid(axis="x", color=LGRAY, lw=0.7)
    axes[0].text(0.0, -0.27, "Model = 빠른진단+과거CES (causal) · PCHIP/Linear = 과거+미래 보간(오프라인) · *PCHIP=headline 기준선",
                 transform=axes[0].transAxes, fontsize=8.6, color=GRAY)
    fig.suptitle("모든 baseline보다 낮은 RMSE — 최종 모델(seed 42), 물리 단위",
                 fontsize=14, fontweight="bold", color=NAVY, y=1.02)
    fig.tight_layout()
    save(fig, "fig_rmse_ladder.png")


# ----------------------------------------------------------------------------
# 3) Progression: n.s. baseline (iter2) -> significant (iter5)  [CES_TI]
# ----------------------------------------------------------------------------
def fig_progression():
    fig, ax = plt.subplots(figsize=(8.6, 3.7))
    labels = ["기존 baseline\n(iter2, GRU)", "최종 모델\n(iter5, +multi-head attn)"]
    vals = [0.088, 0.270]
    lo = [-0.221, 0.144]
    hi = [0.323, 0.364]
    x = [0, 1]
    colors = [GRAY, GREEN]
    for i in x:
        ax.plot([x[i], x[i]], [lo[i], hi[i]], color=colors[i], lw=4, solid_capstyle="round")
        ax.scatter([x[i]], [vals[i]], color=colors[i], s=160, zorder=3,
                   edgecolor="white", linewidth=1.5)
        tag = "PASS" if lo[i] > 0 else "n.s."
        ax.text(x[i], hi[i] + 0.03, f"+{vals[i]:.3f}\n{tag}", ha="center", va="bottom",
                fontsize=11, color=colors[i], fontweight="bold")
    ax.axhline(0, color=RED, lw=1.6, ls="--")
    ax.annotate("", xy=(0.92, 0.27), xytext=(0.08, 0.088),
                arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=2.2))
    ax.text(0.5, 0.05, "val skill_vs_pchip 로\n모델 선택", ha="center", fontsize=9.5,
            color=ORANGE, style="italic")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_xlim(-0.45, 1.5)
    ax.set_ylim(-0.4, 0.5)
    ax.set_ylabel("CES_TI  skill_vs_pchip", fontsize=11)
    ax.set_title("정직한 진전: n.s. → 통계적으로 유의 (held-out test)", fontsize=13,
                 fontweight="bold", color=NAVY)
    ax.grid(axis="y", color=LGRAY, lw=0.7)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    save(fig, "fig_progression.png")


# ----------------------------------------------------------------------------
# 4) Input-modality ablation: where does each target's signal come from?
# ----------------------------------------------------------------------------
def fig_ablation():
    fig, ax = plt.subplots(figsize=(9.4, 4.6))
    groups = ["Full\n(history+fast+time)", "no_fast\n(history only)", "no_history\n(fast only)"]
    ti = [0.359, 0.393, 0.162]
    vt = [0.180, 0.234, -3.31]
    ymin = -1.15
    x = np.arange(len(groups))
    w = 0.36
    b1 = ax.bar(x - w / 2, ti, w, label="CES_TI", color=ORANGE, edgecolor="white")
    b2 = ax.bar(x + w / 2, [min(v, 0.6) if v > 0 else max(v, ymin + 0.02) for v in vt],
                w, label="CES_VT", color=BLUE, edgecolor="white")
    ax.axhline(0, color="#333", lw=1.0)
    for bars, vals in ((b1, ti), (b2, vt)):
        for b, v in zip(bars, vals):
            if v < ymin:  # clipped bar: label just above the floor, in red
                ax.text(b.get_x() + b.get_width() / 2, ymin + 0.06, f"{v:+.2f}",
                        ha="center", va="bottom", fontsize=10, fontweight="bold", color=RED)
            else:
                ax.text(b.get_x() + b.get_width() / 2, v + 0.03, f"{v:+.2f}",
                        ha="center", va="bottom", fontsize=9.5, fontweight="bold", color=NAVY)
    ax.set_xticks(x)
    ax.set_xticklabels(groups, fontsize=10.5)
    ax.set_ylabel("skill_vs_persistence", fontsize=11)
    ax.set_ylim(ymin, 0.62)
    ax.set_title("입력 모달리티 ablation — 신호는 어디서 오는가?", fontsize=13,
                 fontweight="bold", color=NAVY, pad=24)
    ax.text(0.5, 1.03, "no_history(fast only)에서 CES_VT = -3.31  →  V_rot 정보는 거의 전적으로 과거 CES 이력에서 옴",
            transform=ax.transAxes, ha="center", fontsize=9.3, color=RED, fontweight="bold")
    ax.legend(fontsize=10, loc="lower left")
    ax.grid(axis="y", color=LGRAY, lw=0.7)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.annotate("막대가 축 아래로 잘림\n(평균예측보다 나쁨)", xy=(2.18, ymin + 0.10),
                xytext=(1.30, -0.55), fontsize=8.6, color=RED,
                arrowprops=dict(arrowstyle="-|>", color=RED, lw=1.4))
    fig.tight_layout()
    save(fig, "fig_ablation.png")


# ----------------------------------------------------------------------------
# 5) Peak (high-variability) finding: global vs peak skill
# ----------------------------------------------------------------------------
def fig_peak():
    fig, ax = plt.subplots(figsize=(8.6, 4.0))
    groups = ["CES_TI", "CES_VT"]
    glob = [0.515, 0.241]
    peak = [0.855, 0.691]
    peak_lo = [0.658, 0.107]
    peak_hi = [0.933, 0.871]
    x = np.arange(len(groups))
    w = 0.34
    ax.bar(x - w / 2, glob, w, label="전체(global)", color=LGRAY, edgecolor="white")
    pb = ax.bar(x + w / 2, peak, w, label="고변동 구간(peak)", color=TEAL, edgecolor="white")
    err_lo = [peak[i] - peak_lo[i] for i in range(2)]
    err_hi = [peak_hi[i] - peak[i] for i in range(2)]
    ax.errorbar(x + w / 2, peak, yerr=[err_lo, err_hi], fmt="none",
                ecolor=NAVY, elinewidth=1.6, capsize=5)
    for i in range(2):
        ax.text(x[i] - w / 2, glob[i] + 0.02, f"+{glob[i]:.2f}", ha="center",
                fontsize=10, color=GRAY, fontweight="bold")
        ax.text(x[i] + w / 2, peak_hi[i] + 0.02, f"+{peak[i]:.2f}", ha="center",
                fontsize=10.5, color=TEAL, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(groups, fontsize=12)
    ax.set_ylabel("skill_vs_pchip", fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.set_title("모델의 우위는 '고변동(peak) 구간'에 집중된다", fontsize=13,
                 fontweight="bold", color=NAVY)
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(axis="y", color=LGRAY, lw=0.7)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.text(0.5, -0.13, "보간은 매끄러운 bulk에서 거의 최적 · 모델의 진짜 가치는 활발한(active) 구간 · CES_VT도 peak에서는 PASS",
            transform=ax.transAxes, ha="center", fontsize=8.8, color=GRAY)
    fig.tight_layout()
    save(fig, "fig_peak.png")


# ----------------------------------------------------------------------------
# 6) Missingness / data-quality infographic
# ----------------------------------------------------------------------------
def fig_missing():
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.3))

    def donut(ax, frac, color, title, sub):
        ax.pie([frac, 1 - frac], colors=[color, LGRAY], startangle=90,
               counterclock=False, wedgeprops=dict(width=0.42, edgecolor="white"))
        ax.text(0, 0.12, f"{int(round(frac*100))}%", ha="center", va="center",
                fontsize=22, fontweight="bold", color=color)
        ax.text(0, -0.22, sub, ha="center", va="center", fontsize=9, color=GRAY)
        ax.set_title(title, fontsize=12, fontweight="bold", color=NAVY)

    donut(axes[0], 0.08, ORANGE, "CES_TI 결측", "10ms 격자에서\n약 8% 누락")
    donut(axes[1], 0.24, BLUE, "CES_VT 결측", "10ms 격자에서\n약 24% 누락")
    donut(axes[2], 0.54, RED, "CES_VT held/stuck", "관측값의 ~54%가\nforward-fill (가짜 측정)")
    fig.suptitle("CES는 자주 비고(독립적 결측), V_rot 값의 절반 이상이 held 값 — 데이터 품질 보정 필요",
                 fontsize=12.5, fontweight="bold", color=NAVY, y=1.06)
    fig.tight_layout()
    save(fig, "fig_missing.png")


if __name__ == "__main__":
    fig_forest()
    fig_rmse_ladder()
    fig_progression()
    fig_ablation()
    fig_peak()
    fig_missing()
    print("ALL FIGURES DONE ->", OUT)
