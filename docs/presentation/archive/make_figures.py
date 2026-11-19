# -*- coding: utf-8 -*-
"""Generate all presentation figures for the KSTAR CES nowcasting talk (Korean).

All run-dependent numbers are read from docs/paper/paper_numbers.json, which
`ces_prediction/collect_paper_numbers.py` regenerates from the frozen artifacts
under data/. Never hard-code a run number here (PROJECT_KNOWLEDGE.md
"Numbers Must Come From Artifacts").

Protocol behind every panel (2026-08-16, THESIS_RESULTS.md §8ab): W = 2, held-free,
per-file cap 500, TWO co-primary populations (컷 = CES_TI > 3 keV 결측 처리 / 포함 =
컷 없음), shot-clustered paired bootstrap (10,000). The adopted model is the seq_v2
backbone; the W = 2 window family is the paired control. This file is the Korean
twin of docs/paper/make_figures_en.py.

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
POP_COLOR = {"cut": NAVY, "incl": ORANGE}
POP_LABEL = {"cut": "컷 모집단 (T_i > 3 keV 결측 처리)", "incl": "포함 모집단 (컷 없음)"}

HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "figures")
os.makedirs(OUT, exist_ok=True)

with open(os.path.join(HERE, "..", "paper", "paper_numbers.json"), encoding="utf-8") as f:
    N = json.load(f)

SEEDS = ["42", "1", "7", "123"]
POPS = ("cut", "incl")
T = ("CES_TI", "CES_VT")
TLABEL = {"CES_TI": "CES_TI  (이온 온도 T_i)", "CES_VT": "CES_VT  (토로이달 회전 V_rot)"}
SKILL_PCHIP = "skill_vs_pchip  (= 1 - MSE_model / MSE_PCHIP)"


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print("wrote", name)


def _clean(ax, left=True):
    for s in ["top", "right"] + ([] if left else ["left"]):
        ax.spines[s].set_visible(False)


# ---------------------------------------------------------------- 1) headline forest
def fig_forest():
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    for ax, t in zip(axes, T):
        y0 = np.arange(len(SEEDS))[::-1] * 1.0
        for pop in POPS:
            per = N["headline"][pop]["seq"]["per_seed"]
            off = 0.18 if pop == "cut" else -0.18
            for i, s in enumerate(SEEDS):
                e = per[s][t]["pchip"]
                v, (lo, hi), passed = e["skill"], e["ci"], e["pass"]
                c = POP_COLOR[pop]
                yy = y0[i] + off
                ax.plot([lo, hi], [yy, yy], color=c, lw=2.6, solid_capstyle="round", zorder=2, alpha=1.0 if passed else 0.45)
                ax.scatter([v], [yy], color=c if passed else "white", edgecolor=c, s=70, zorder=3, linewidth=1.4)
                ax.text(hi + 0.02, yy, f"{v:+.2f}" + ("" if passed else "  n.s."), va="center", ha="left", fontsize=8.8, color=c)
        ax.axvline(0, color=RED, lw=1.4, ls="--", zorder=1)
        ax.set_yticks(y0)
        ax.set_yticklabels([f"분할 {s}" for s in SEEDS], fontsize=10.5)
        ax.set_xlim(-0.75, 0.75)
        ax.set_xlabel(SKILL_PCHIP, fontsize=9.5)
        ax.set_ylim(-0.7, len(SEEDS) - 0.3)
        ax.set_title(TLABEL[t], fontsize=12.5, fontweight="bold", color=NAVY, pad=22)
        ax.grid(axis="x", color=LGRAY, lw=0.8)
        _clean(ax, left=False)
        pr = {pop: N["headline"][pop]["seq"]["pr4_pass"][t]["pchip"] for pop in POPS}
        ax.text(0.0, 1.015, f"PR4 PASS: 컷 {pr['cut']}/4  ·  포함 {pr['incl']}/4",
                transform=ax.transAxes, fontsize=9.5, color=GREEN if min(pr.values()) == 4 else GRAY, fontweight="bold")
    h = [plt.Line2D([], [], color=POP_COLOR[p], lw=3, label=POP_LABEL[p]) for p in POPS]
    fig.legend(handles=h, loc="lower center", ncol=2, fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("seq_v2 백본 vs 미래를 쓰는 PCHIP — 4개 독립 test 분할, 두 모집단 (shot-clustered 95% CI, B = 10,000; W = 2, held-free)",
                 fontsize=12, fontweight="bold", color=NAVY, y=1.04)
    fig.tight_layout()
    save(fig, "fig_forest.png")


# ---------------------------------------------------------------- 2) RMSE ladder (seed 42, cut)
def fig_rmse_ladder():
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    lad = N["headline"]["cut"]["rmse_ladder"]["42"]
    order = [("ar_local", "AR (국소, 과거만)"), ("persistence", "Persistence"), ("gp_causal", "인과 GP (과거만)"),
             ("pchip", "PCHIP* (과거+미래)"), ("linear", "선형 보간 (과거+미래)"), ("gp", "GP (과거+미래)")]
    for ax, t in zip(axes, T):
        e = lad[t]
        names = [n for _, n in order] + ["윈도 대조군 (W = 2)", "seq_v2 백본"]
        vals = [e["rmse_baselines"][k] for k, _ in order] + [e["rmse_model_window"], e["rmse_model_seq_v2"]]
        colors = [GRAY, GRAY, BLUE, LGRAY, LGRAY, LGRAY, TEAL, NAVY]
        y = np.arange(len(names))[::-1]
        bars = ax.barh(y, vals, color=colors, edgecolor="white", height=0.66)
        ax.set_yticks(y)
        ax.set_yticklabels(names, fontsize=9.6)
        for b, v in zip(bars, vals):
            ax.text(v + max(vals) * 0.012, b.get_y() + b.get_height() / 2, f"{v:.1f}", va="center", fontsize=9, color=NAVY)
        ax.set_xlim(0, max(vals) * 1.2)
        unit = "eV" if t == "CES_TI" else "km/s"
        ax.set_title(f"{TLABEL[t]}  —  RMSE ({unit}), n = {e['n']:,}", fontsize=11, fontweight="bold", color=NAVY)
        ax.grid(axis="x", color=LGRAY, lw=0.7)
        _clean(ax)
    fig.suptitle("물리 단위 RMSE 사다리 — 기준 test 분할(seed 42), 컷 모집단, 진짜 측정만. 회색 = 인과, 연회색 = 미래 사용; 백본이 두 타겟 모두 최저",
                 fontsize=10.5, fontweight="bold", color=NAVY, y=1.03)
    fig.tight_layout()
    save(fig, "fig_rmse_ladder.png")


# ---------------------------------------------------------------- 3) complexity ladder + width scaling
def fig_ladder_scaling():
    fig, ax = plt.subplots(figsize=(10.5, 4.6))
    L = N["ladder"]
    B4 = N["scaling_b4"]["widths"]
    for pop in POPS:
        c = POP_COLOR[pop]
        r = L[pop]["skill_vs_pchip"]
        pts = [(1258, r["anchor"]["CES_TI"]["mean"], "anchor+Δ"), (21498, r["b3"]["CES_TI"]["mean"], "b3k8"),
               (201258, r["win"]["CES_TI"]["mean"], "윈도 대조군")]
        for x, yv, lab in pts:
            ax.scatter([x], [yv], color=c if pop == "cut" else "white", edgecolor=c, s=95, zorder=4, linewidth=1.6,
                       marker="s" if lab == "윈도 대조군" else "o")
            if pop == "cut":
                ax.annotate(lab, (x, yv), textcoords="offset points", xytext=(0, 9), ha="center", fontsize=8.8, color=c)
        ax.axhline(r["persistence"]["CES_TI"]["mean"], color=c, lw=1.0, ls=":", alpha=0.8)
        ax.text(1.05e6, r["persistence"]["CES_TI"]["mean"] + 0.012, f"persistence ({'컷' if pop == 'cut' else '포함'})", fontsize=8, color=c, ha="right")
    ws = sorted(B4.keys(), key=lambda k: int(k))
    xs = [B4[w]["params"] for w in ws]
    ys = [B4[w]["skill_TI_mean"] for w in ws]
    ax.plot(xs, ys, color=NAVY, lw=2.2, zorder=3)
    for w, x, yv in zip(ws, xs, ys):
        for s in SEEDS:
            ax.scatter([x], [B4[w]["skill_TI_per_seed"][s]], color=NAVY, s=14, alpha=0.35, zorder=2)
        ax.scatter([x], [yv], color=NAVY, s=60, zorder=5, marker="D")
        ax.annotate(f"seq_v2\n폭 {w}", (x, yv), textcoords="offset points", xytext=(0, -26), ha="center", fontsize=7.8, color=NAVY)
    ax.scatter([357570], [L["incl"]["skill_vs_pchip"]["seq"]["CES_TI"]["mean"]], color="white", edgecolor=ORANGE, s=70, marker="D", linewidth=1.6, zorder=5)
    ax.set_xscale("log")
    ax.set_xlim(8e2, 1.2e6)
    ax.set_ylim(-0.4, 0.42)
    ax.axhline(0, color=RED, lw=1.2, ls="--")
    ax.set_xlabel("학습 파라미터 수 (log)", fontsize=10.5)
    ax.set_ylabel("TEST T_i skill vs PCHIP (4 분할 평균)", fontsize=10.5)
    ax.set_title("복잡도 사다리 + 폭 스윕: T_i skill은 21k→879k에서 평평(컷); 해석가능 단은 컷에서만 백본과 동급",
                 fontsize=10.5, fontweight="bold", color=NAVY)
    h = [plt.Line2D([], [], marker="o", color=NAVY, ls="", label="컷 모집단"),
         plt.Line2D([], [], marker="o", color="white", markeredgecolor=ORANGE, ls="", label="포함 모집단"),
         plt.Line2D([], [], marker="D", color=NAVY, ls="-", label="seq_v2 폭 스윕 (B.4, 컷)")]
    ax.legend(handles=h, fontsize=8.8, loc="lower right", frameon=False)
    ax.grid(color=LGRAY, lw=0.7)
    _clean(ax)
    fig.tight_layout()
    save(fig, "fig_ladder_scaling.png")


# ---------------------------------------------------------------- 4) ablation (window family, eval-time)
def fig_ablation():
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6), sharey=False)
    A = N["ablation_window_eval"]
    H = N["headline"]
    arms = ["full", "no_fast", "no_history"]
    labels = ["전체\n(이력 +\n빠른 진단 + 시간)", "no_fast\n(이력 +\n시간만)", "no_history\n(빠른 진단 +\n시간만)"]
    x = np.arange(len(arms)) * 1.25
    w = 0.42
    for ax, t in zip(axes, T):
        for k, pop in enumerate(POPS):
            vals = []
            for arm in arms:
                if arm == "full":
                    v = [H[pop]["win"]["per_seed"][s][t]["pchip"]["skill"] for s in SEEDS]
                else:
                    v = [A[pop][arm][s][t]["skill_vs_pchip"] for s in SEEDS]
                vals.append(v)
            means = [float(np.mean(v)) for v in vals]
            xx = x + (k - 0.5) * w
            ax.bar(xx, [max(m, -1.0) for m in means], w, color=POP_COLOR[pop], alpha=0.85, edgecolor="white", label=POP_LABEL[pop])
            for xi, v, m in zip(xx, vals, means):
                ax.scatter([xi] * len(v), [max(vv, -1.0) for vv in v], color="white", edgecolor=NAVY, s=16, zorder=4, linewidth=0.8)
                if m >= 0:
                    ax.text(xi, max(max(v), m) + 0.035, f"{m:+.2f}", ha="center", va="bottom", fontsize=8.6, color=POP_COLOR[pop], fontweight="bold")
                else:
                    ax.text(xi, max(m, -1.0) - 0.035, f"{m:+.2f}", ha="center", va="top", fontsize=8.6, color=POP_COLOR[pop], fontweight="bold")
        ax.axhline(0, color=RED, lw=1.2, ls="--")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9.0)
        ax.set_ylim(-1.2, 0.5)
        ax.set_ylabel("skill vs PCHIP (TEST, 4 분할 평균; 점 = 분할)", fontsize=9)
        ax.set_title(TLABEL[t], fontsize=12, fontweight="bold", color=NAVY)
        ax.grid(axis="y", color=LGRAY, lw=0.7)
        _clean(ax)
    hnd, lab = axes[0].get_legend_handles_labels()
    fig.legend(hnd, lab, loc="lower center", ncol=2, fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.06))
    fig.text(0.5, -0.11, "no_fast는 V_rot 출력을 bit-identical로 남긴다(라우팅은 구조적); no_history는 두 타겟 모두 붕괴(막대·점은 -1에서 잘림)",
             ha="center", fontsize=8.8, color=GRAY)
    fig.suptitle("W = 2 윈도 대조군의 평가 시 modality 절제 (입력 0 처리, 재학습 없음)",
                 fontsize=12, fontweight="bold", color=NAVY, y=1.02)
    fig.tight_layout()
    save(fig, "fig_ablation.png")


# ---------------------------------------------------------------- 5) peak strata
def fig_peak():
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    P = N["peak"]
    for ax, t in zip(axes, T):
        x = np.arange(2)
        w = 0.36
        for k, pop in enumerate(POPS):
            ents = P[pop]["seq"][t]
            means, pts, npass = [], [], []
            for lab in ("non_peak", "peak"):
                v = [e["pchip"]["skill"] for e in ents[lab] if "pchip" in e]
                means.append(float(np.mean(v))); pts.append(v)
                npass.append(sum(int(e["pchip"]["pass"]) for e in ents[lab] if "pchip" in e))
            xx = x + (k - 0.5) * w
            ax.bar(xx, means, w, color=POP_COLOR[pop], alpha=0.85, edgecolor="white", label=POP_LABEL[pop])
            for xi, v, m, npz in zip(xx, pts, means, npass):
                ax.scatter([xi] * len(v), v, color="white", edgecolor=NAVY, s=16, zorder=4, linewidth=0.8)
                ax.text(xi, max(v) + 0.03, f"{m:+.2f}\nPASS {npz}/4", ha="center", va="bottom", fontsize=8.4, color=POP_COLOR[pop], fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(["본류 (non-peak)", "고변동 (peak)"], fontsize=10.5)
        ax.set_ylim(-0.2, 1.05)
        ax.axhline(0, color=RED, lw=1.2, ls="--")
        ax.set_ylabel("skill vs PCHIP (TEST, 4 분할 평균; 점 = 분할)", fontsize=9)
        ax.set_title(TLABEL[t], fontsize=12, fontweight="bold", color=NAVY)
        ax.grid(axis="y", color=LGRAY, lw=0.7)
        _clean(ax)
    axes[0].legend(fontsize=8.4, loc="upper left", frameon=False)
    fig.suptitle("백본의 우위는 보간이 가장 약한 곳에 집중: peak 층화 TEST skill, 두 모집단",
                 fontsize=12, fontweight="bold", color=NAVY, y=1.02)
    fig.tight_layout()
    save(fig, "fig_peak.png")


# ---------------------------------------------------------------- 6) campaign split
def fig_campaign():
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))
    C = N["campaign"]
    for ax, t in zip(axes, T):
        arms = [("win", "윈도 OFF"), ("winps", "윈도 ON\n(shot별 표준화)"), ("seq", "seq_v2\n백본")]
        x = np.arange(len(arms))
        w = 0.36
        for k, pop in enumerate(POPS):
            per = C[pop]["per_init"]
            xx = x + (k - 0.5) * w
            for xi, (arm, _) in zip(xx, arms):
                if arm not in per["42"]:
                    continue
                v = [per[s][arm][t]["pchip"]["skill"] for s in SEEDS]
                npass = sum(int(per[s][arm][t]["pchip"]["pass"]) for s in SEEDS)
                m = float(np.mean(v))
                ax.bar([xi], [m], w, color=POP_COLOR[pop], alpha=0.85, edgecolor="white")
                ax.scatter([xi] * len(v), v, color="white", edgecolor=NAVY, s=16, zorder=4, linewidth=0.8)
                ax.text(xi, max(max(v), 0) + 0.02, f"{m:+.2f}\n{npass}/4", ha="center", va="bottom", fontsize=8.4, color=POP_COLOR[pop], fontweight="bold")
        ax.axhline(0, color=RED, lw=1.2, ls="--")
        ax.set_xticks(x)
        ax.set_xticklabels([l for _, l in arms], fontsize=9.6)
        ax.set_ylim(-0.15, 0.45)
        ax.set_ylabel("시간 분할 TEST 블록의 skill vs PCHIP (초기화 4개)", fontsize=9)
        ax.set_title(TLABEL[t], fontsize=12, fontweight="bold", color=NAVY)
        ax.grid(axis="y", color=LGRAY, lw=0.7)
        _clean(ax)
    h = [plt.Rectangle((0, 0), 1, 1, color=POP_COLOR[p], alpha=0.85, label=POP_LABEL[p]) for p in POPS]
    axes[0].legend(handles=h, fontsize=8.4, loc="upper left", frameon=False)
    m = C["cut"]["manifest"]
    fig.suptitle(f"캠페인(시간) 분할 — train shot {m['train'][0]}–{m['train'][1]}, test {m['test'][0]}–{m['test'][1]}: "
                 "윈도 모델의 오프라인 우위는 붕괴, 백본은 생존 (라벨 = 평균, PR4 PASS 수)",
                 fontsize=10.5, fontweight="bold", color=NAVY, y=1.03)
    fig.tight_layout()
    save(fig, "fig_campaign.png")


# ---------------------------------------------------------------- 7) missingness census
def fig_missing():
    D = N["data_ledger"]["targets"]
    nan_pct = [100 * D["CES_TI"]["missing_frac"], 100 * D["CES_VT"]["missing_frac"]]
    held_pct = [100 * D["CES_TI"]["held_frac_of_rows"], 100 * D["CES_VT"]["held_frac_of_rows"]]
    obs_pct = [100 * D["CES_TI"]["independent_frac_of_rows"], 100 * D["CES_VT"]["independent_frac_of_rows"]]
    held_obs = D["CES_VT"]["held_frac_of_observed"]
    n_files = N["data_ledger"]["n_files"]
    n_rows = N["data_ledger"]["n_rows"]

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.4), gridspec_kw={"width_ratios": [2.15, 1]})
    ax = axes[0]
    labels = ["CES_TI", "CES_VT"]
    ypos = [1, 0]
    h = 0.46
    for i, y in enumerate(ypos):
        ax.barh(y, nan_pct[i], h, color=BLUE, edgecolor="white")
        ax.barh(y, held_pct[i], h, left=nan_pct[i], color=RED, edgecolor="white")
        ax.barh(y, obs_pct[i], h, left=nan_pct[i] + held_pct[i], color=GREEN, edgecolor="white")
        if nan_pct[i] > 5:
            ax.text(nan_pct[i] / 2, y, f"{nan_pct[i]:.1f}%", ha="center", va="center", fontsize=10.5, color="white", fontweight="bold")
        if held_pct[i] > 5:
            ax.text(nan_pct[i] + held_pct[i] / 2, y, f"{held_pct[i]:.1f}%", ha="center", va="center", fontsize=10.5, color="white", fontweight="bold")
        ax.text(nan_pct[i] + held_pct[i] + obs_pct[i] / 2, y, f"{obs_pct[i]:.1f}%", ha="center", va="center", fontsize=10.5, color="white", fontweight="bold")
        eff = nan_pct[i] + held_pct[i]
        ax.text(101.5, y, f"실질 무정보 {eff:.1f}%", va="center", fontsize=10.5, color=RED if eff > 50 else GRAY, fontweight="bold")
    ax.axvline(nan_pct[1] + held_pct[1], color=RED, lw=1.1, ls="--", alpha=0.65)
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels, fontsize=12, fontweight="bold")
    ax.set_xlim(0, 128)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xticklabels(["0", "25", "50", "75", "100%"], fontsize=9.5)
    ax.set_xlabel("전체 10 ms 격자 행 대비 비율", fontsize=10)
    ax.set_title("결측 구성: NaN 결측 + held(같은 값 padding)", fontsize=12, fontweight="bold", color=NAVY)
    for sp in ["top", "right", "left"]:
        ax.spines[sp].set_visible(False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (BLUE, RED, GREEN)]
    ax.legend(handles, ["① NaN 결측", "② held / padding", "독립 관측"], fontsize=9.5, loc="upper center",
              bbox_to_anchor=(0.42, -0.20), ncol=3, frameon=False)
    ax = axes[1]
    ax.pie([held_obs, 1 - held_obs], colors=[RED, LGRAY], startangle=90, counterclock=False,
           wedgeprops=dict(width=0.42, edgecolor="white"))
    ax.text(0, 0.12, f"{held_obs*100:.0f}%", ha="center", va="center", fontsize=22, fontweight="bold", color=RED)
    ax.text(0, -1.45, "관측된 V_rot 값 중\nforward-fill (가짜 측정)", ha="center", va="top", fontsize=9, color=GRAY)
    ax.set_title("CES_VT held/stuck 비율", fontsize=12, fontweight="bold", color=NAVY)
    fig.suptitle(f"V_rot는 NaN {nan_pct[1]:.1f}%가 아니라 실질 {nan_pct[1]+held_pct[1]:.1f}%가 무정보 — {n_files} shot / {n_rows:,} 행 전수 집계",
                 fontsize=12.5, fontweight="bold", color=NAVY, y=1.06)
    fig.tight_layout()
    save(fig, "fig_missing.png")


# ---------------------------------------------------------------- 8) B.1 backbone gate (16-run grid)
def fig_gate_b1():
    G = N["gate_b1"]
    fig, ax = plt.subplots(figsize=(10.5, 4.2))
    y = 0
    yt, yl = [], []
    for s in SEEDS:
        vals = G["per_split_paired_TI"][s]
        for j, v in enumerate(vals):
            ax.scatter([v], [y], color=NAVY, s=34, zorder=3, alpha=0.85)
            y += 1
        m = G["per_split_init_mean_TI"][s]
        ax.plot([m, m], [y - len(vals) - 0.4, y - 0.6], color=ORANGE, lw=2.2, zorder=4)
        ax.text(m + 0.006, y - 0.5 - len(vals) / 2, f"분할 {s} 평균 {m:+.3f}", va="center", fontsize=9, color=ORANGE, fontweight="bold")
        yt.append(y - 0.5 - len(vals) / 2); yl.append(f"분할 {s}\n(초기화 4개)")
        y += 0.8
    ax.axvline(0, color=RED, lw=1.4, ls="--")
    lo, hi = G["pooled_ci95"]
    ax.axvspan(lo, hi, color=GREEN, alpha=0.12, zorder=1)
    ax.axvline(G["pooled_mean_TI"], color=GREEN, lw=2.0, zorder=2)
    ax.text(G["pooled_mean_TI"] + 0.006, y + 0.2, f"pooled {G['pooled_mean_TI']:+.3f}  CI [{lo:+.3f}, {hi:+.3f}]  ·  {G['n_positive']}/16 양수",
            fontsize=9.5, color=GREEN, fontweight="bold")
    ax.set_yticks(yt); ax.set_yticklabels(yl, fontsize=9.5)
    ax.set_ylim(-1, y + 1.2)
    ax.set_xlim(-0.02, 0.17)
    ax.set_xlabel("paired T_i skill (seq_v2 - W=2 윈도 대조군, 같은 행)", fontsize=10)
    ax.set_title("B.1 백본 관문 — 분할 4 × 초기화 4 = 16 run: 16/16 양수, pooled CI가 0 배제, 예산 균등화에서도 4/4 부호 유지 → 백본 = seq_v2",
                 fontsize=10.5, fontweight="bold", color=NAVY)
    ax.grid(axis="x", color=LGRAY, lw=0.7)
    _clean(ax, left=False)
    fig.tight_layout()
    save(fig, "fig_gate_b1.png")


if __name__ == "__main__":
    fig_forest()
    fig_rmse_ladder()
    fig_ladder_scaling()
    fig_ablation()
    fig_peak()
    fig_campaign()
    fig_missing()
    fig_gate_b1()
    for stale in ("fig_progression.png",):
        p = os.path.join(OUT, stale)
        if os.path.exists(p):
            os.remove(p)
            print("removed stale", stale)
    print("ALL FIGURES DONE ->", OUT)
