# -*- coding: utf-8 -*-
"""Generate all presentation figures for the KSTAR CES nowcasting talk (Korean).

All run-dependent numbers are read from docs/paper/paper_numbers.json, which
`ces_prediction/collect_paper_numbers.py` regenerates from the frozen artifacts
under data/. Never hard-code a run number here (PROJECT_KNOWLEDGE.md
"Numbers Must Come From Artifacts"). The only literals kept are the §1.1
missingness census facts (population statistics of the raw CSVs, not model runs).

Headline population: genuine (held/forward-filled V_rot excluded) — same as the
paper. The progression figure uses the hold-inclusive population because iter2
was only ever scored there (comparing across populations is not a comparison).

Output: docs/presentation/figures/*.png
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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

HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "figures")
os.makedirs(OUT, exist_ok=True)

with open(os.path.join(HERE, "..", "paper", "paper_numbers.json"), encoding="utf-8") as f:
    NUM = json.load(f)

SEEDS = ["42", "1", "7", "123"]


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print("wrote", name)


# ----------------------------------------------------------------------------
# 1) Headline forest plot: per-seed skill_vs_pchip with shot-clustered 95% CI
#    (genuine population — the paper's headline)
# ----------------------------------------------------------------------------
def fig_forest():
    per_seed = NUM["headline"]["genuine"]["per_seed"]

    def series(tgt):
        vals, lo, hi = [], [], []
        for s in SEEDS:
            e = per_seed[s][tgt]["ci_vs_pchip"]
            vals.append(e["skill"])
            lo.append(e["ci95"][0])
            hi.append(e["ci95"][1])
        return vals, lo, hi

    ti, ti_lo, ti_hi = series("CES_TI")
    vt, vt_lo, vt_hi = series("CES_VT")
    seeds = [f"seed {s}" for s in SEEDS]

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    for ax, vals, lo, hi, title in [
        (axes[0], ti, ti_lo, ti_hi, "CES_TI  (이온 온도)"),
        (axes[1], vt, vt_lo, vt_hi, "CES_VT  (토로이달 회전)"),
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
        ax.set_xlim(-0.88, 0.65)
        ax.set_xlabel("skill_vs_pchip  (= 1 - MSE_model / MSE_pchip)", fontsize=10)
        ax.set_title(title, fontsize=13, fontweight="bold", color=NAVY, pad=8)
        ax.grid(axis="x", color=LGRAY, lw=0.8)
        for s in ["top", "right", "left"]:
            ax.spines[s].set_visible(False)
    axes[0].text(0.02, 1.0, "4개 독립 test split 모두 95% CI > 0  →  PASS",
                 transform=axes[0].transAxes, fontsize=10, color=GREEN, fontweight="bold")
    axes[1].text(0.02, 1.0, "PASS는 seed 1 하나 (1/4 = 잡음 수준)  →  동률 보고",
                 transform=axes[1].transAxes, fontsize=10, color=GRAY, fontweight="bold")
    fig.suptitle("보간(PCHIP) 대비 skill — 진짜 측정(genuine) 평가, shot-clustered 95% CI (B=10,000)",
                 fontsize=14, fontweight="bold", color=NAVY, y=1.04)
    fig.tight_layout()
    save(fig, "fig_forest.png")


# ----------------------------------------------------------------------------
# 2) RMSE ladder (physical units) — final model vs baseline ladder (genuine, s42)
# ----------------------------------------------------------------------------
def fig_rmse_ladder():
    s42 = NUM["headline"]["genuine"]["per_seed"]["42"]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))
    data = {}
    for tgt in ("CES_TI", "CES_VT"):
        e = s42[tgt]
        r = e["rmse"]
        data[tgt] = dict(
            names=["AR (past only)", "Persistence", "PCHIP*", "Linear", "Model (nowcaster)"],
            vals=[r["ar_local"], r["persistence"], r["pchip"], r["linear"], e["rmse_model"]],
            unit="물리단위 RMSE", title=f"{tgt}  (n={e['n']:,}, genuine)")
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
    fig.suptitle("모든 baseline보다 낮은 RMSE — 최종 모델(seed 42), 물리 단위, 진짜 측정 평가",
                 fontsize=14, fontweight="bold", color=NAVY, y=1.02)
    fig.tight_layout()
    save(fig, "fig_rmse_ladder.png")


# ----------------------------------------------------------------------------
# 3) Progression: n.s. baseline (iter2) -> significant (iter009)  [CES_TI]
#    Hold-inclusive population on BOTH sides: iter2 was only ever scored there.
# ----------------------------------------------------------------------------
def fig_progression():
    i2 = NUM["before_baseline_iter2"]["CES_TI"]["ci_test"]
    fin = NUM["headline"]["stuck0"]["per_seed"]["42"]["CES_TI"]["ci_vs_pchip"]
    fig, ax = plt.subplots(figsize=(8.6, 3.7))
    labels = ["기존 baseline\n(iter2, GRU)", "최종 모델\n(iter009, +관측마스킹 attn)"]
    vals = [i2["skill"], fin["skill"]]
    lo = [i2["ci95"][0], fin["ci95"][0]]
    hi = [i2["ci95"][1], fin["ci95"][1]]
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
    ax.annotate("", xy=(0.92, vals[1]), xytext=(0.08, vals[0]),
                arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=2.2))
    ax.text(0.5, 0.05, "val skill_vs_pchip 로\n모델 선택", ha="center", fontsize=9.5,
            color=ORANGE, style="italic")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_xlim(-0.45, 1.5)
    ax.set_ylim(-0.4, 0.5)
    ax.set_ylabel("CES_TI  skill_vs_pchip", fontsize=11)
    ax.set_title("정직한 진전: n.s. → 통계적으로 유의 (held-out test, 동일 모집단)", fontsize=13,
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
    ab = NUM["ablation_val_vs_persistence"]

    def sk(arm, tgt):
        return ab[arm]["per_target"][tgt]["skill_vs_persistence"]

    arms = ["full", "no_fast", "no_history"]
    ti = [sk(a, "CES_TI") for a in arms]
    vt = [sk(a, "CES_VT") for a in arms]

    fig, ax = plt.subplots(figsize=(9.4, 4.6))
    groups = ["Full\n(history+fast+time)", "no_fast\n(history only)", "no_history\n(fast only)"]
    ymin = -0.85
    x = np.arange(len(groups))
    w = 0.36
    b1 = ax.bar(x - w / 2, ti, w, label="CES_TI", color=ORANGE, edgecolor="white")
    b2 = ax.bar(x + w / 2, vt, w, label="CES_VT", color=BLUE, edgecolor="white")
    ax.axhline(0, color="#333", lw=1.0)
    for bars, vals in ((b1, ti), (b2, vt)):
        for b, v in zip(bars, vals):
            va, off, col = ("bottom", 0.03, NAVY) if v >= 0 else ("top", -0.03, RED)
            ax.text(b.get_x() + b.get_width() / 2, v + off, f"{v:+.2f}",
                    ha="center", va=va, fontsize=9.5, fontweight="bold", color=col)
    ax.set_xticks(x)
    ax.set_xticklabels(groups, fontsize=10.5)
    ax.set_ylabel("skill_vs_persistence", fontsize=11)
    ax.set_ylim(ymin, 0.62)
    ax.set_title("입력 모달리티 ablation — 신호는 어디서 오는가?", fontsize=13,
                 fontweight="bold", color=NAVY, pad=24)
    ax.text(0.5, 1.03, f"no_history(fast only)에서 CES_VT = {vt[2]:+.2f} (persistence보다 나쁨)  ->  V_rot 정보는 과거 CES 이력에서 옴",
            transform=ax.transAxes, ha="center", fontsize=9.3, color=RED, fontweight="bold")
    ax.legend(fontsize=10, loc="lower left")
    ax.grid(axis="y", color=LGRAY, lw=0.7)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.annotate(f"fast-only CES_TI는 여전히 {ti[2]:+.2f}\n(빠른 진단이 T_i 정보를 운반)", xy=(2.0 - 0.18, ti[2]),
                xytext=(1.15, -0.42), fontsize=8.6, color=ORANGE,
                arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=1.4))
    fig.tight_layout()
    save(fig, "fig_ablation.png")


# ----------------------------------------------------------------------------
# 5) Peak (high-variability) finding: global vs peak skill (validation, s42)
# ----------------------------------------------------------------------------
def fig_peak():
    p = NUM["peak_seed42"]["per_target"]
    groups = ["CES_TI", "CES_VT"]
    glob = [p[t]["global_skill_vs_pchip"] for t in groups]
    peak = [p[t]["peak_skill_vs_pchip"] for t in groups]
    peak_lo = [p[t]["peak_skill_ci95"][0] for t in groups]
    peak_hi = [p[t]["peak_skill_ci95"][1] for t in groups]

    fig, ax = plt.subplots(figsize=(8.6, 4.0))
    x = np.arange(len(groups))
    w = 0.34
    ax.bar(x - w / 2, glob, w, label="전체(global)", color=LGRAY, edgecolor="white")
    ax.bar(x + w / 2, peak, w, label="고변동 구간(peak)", color=TEAL, edgecolor="white")
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
    ax.text(0.5, -0.13, "보간은 매끄러운 bulk에서 거의 최적 · 모델의 진짜 가치는 활발한(active) 구간 · CES_VT도 peak에서는 PASS (val, 낙관 주의)",
            transform=ax.transAxes, ha="center", fontsize=8.8, color=GRAY)
    fig.tight_layout()
    save(fig, "fig_peak.png")


# ----------------------------------------------------------------------------
# 6) Missingness / data-quality infographic  (§1.1 census facts, not run numbers)
# ----------------------------------------------------------------------------
def fig_missing():
    """결측 실측 (641 shot / 247,207 행 전수 집계).

    NaN 결측만 세면 V_rot는 23.9%지만, 직전 관측값을 그대로 복사한 held 행
    41.1%를 합치면 실질 무정보는 65.0%다. 이 그림은 그 합산을 보여준다.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.4),
                             gridspec_kw={"width_ratios": [2.15, 1]})

    # --- (a) stacked composition bars ------------------------------------
    ax = axes[0]
    labels = ["CES_TI", "CES_VT"]
    nan_pct = [8.2, 23.9]
    held_pct = [0.0, 41.1]
    obs_pct = [91.8, 35.0]
    ypos = [1, 0]
    h = 0.46
    for i, y in enumerate(ypos):
        ax.barh(y, nan_pct[i], h, color=BLUE, edgecolor="white")
        ax.barh(y, held_pct[i], h, left=nan_pct[i], color=RED, edgecolor="white")
        ax.barh(y, obs_pct[i], h, left=nan_pct[i] + held_pct[i],
                color=GREEN, edgecolor="white")
        if nan_pct[i] > 5:
            ax.text(nan_pct[i] / 2, y, f"{nan_pct[i]:.1f}%", ha="center",
                    va="center", fontsize=10.5, color="white", fontweight="bold")
        if held_pct[i] > 5:
            ax.text(nan_pct[i] + held_pct[i] / 2, y, f"{held_pct[i]:.1f}%",
                    ha="center", va="center", fontsize=10.5, color="white",
                    fontweight="bold")
        ax.text(nan_pct[i] + held_pct[i] + obs_pct[i] / 2, y, f"{obs_pct[i]:.1f}%",
                ha="center", va="center", fontsize=10.5, color="white",
                fontweight="bold")
        eff = nan_pct[i] + held_pct[i]
        ax.text(101.5, y, f"실질 무정보 {eff:.1f}%", va="center", fontsize=10.5,
                color=RED if eff > 50 else GRAY, fontweight="bold")
    ax.axvline(65.0, color=RED, lw=1.1, ls="--", alpha=0.65)
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels, fontsize=12, fontweight="bold")
    ax.set_xlim(0, 128)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xticklabels(["0", "25", "50", "75", "100%"], fontsize=9.5)
    ax.set_xlabel("전체 10 ms 격자 행 대비 비율", fontsize=10)
    ax.set_title("결측 구성: NaN 결측 + held(같은 값 padding)", fontsize=12,
                 fontweight="bold", color=NAVY)
    for sp in ["top", "right", "left"]:
        ax.spines[sp].set_visible(False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (BLUE, RED, GREEN)]
    ax.legend(handles, ["① NaN 결측", "② held / padding", "독립 관측"],
              fontsize=9.5, loc="upper center", bbox_to_anchor=(0.42, -0.20),
              ncol=3, frameon=False)

    # --- (b) held share of observed values --------------------------------
    ax = axes[1]
    ax.pie([0.54, 0.46], colors=[RED, LGRAY], startangle=90, counterclock=False,
           wedgeprops=dict(width=0.42, edgecolor="white"))
    ax.text(0, 0.12, "54%", ha="center", va="center", fontsize=22,
            fontweight="bold", color=RED)
    ax.text(0, -1.45, "관측된 V_rot 값 중\nforward-fill (가짜 측정)", ha="center",
            va="top", fontsize=9, color=GRAY)
    ax.set_title("CES_VT held/stuck 비율", fontsize=12, fontweight="bold", color=NAVY)

    fig.suptitle("V_rot는 NaN 23.9%가 아니라 실질 65.0%가 무정보 — 641 shot / 247,207 행 전수 집계",
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
