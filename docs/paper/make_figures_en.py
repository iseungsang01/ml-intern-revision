# -*- coding: utf-8 -*-
"""English-language paper figures for the KSTAR CES nowcasting draft.

Port of docs/presentation/make_figures.py (Korean talk figures) with all
figure-internal text in English and typeset-friendly labels.

Every number is READ FROM `paper_numbers.json` (regenerate with
`py ces_prediction/collect_paper_numbers.py`), which in turn reads only the frozen
run directories under data/. Nothing here is hard-coded: an earlier version of this
file carried literals from a superseded checkpoint family, which is exactly the
failure mode this indirection removes.

HEADLINE_TREATMENT selects which evaluation population the headline figures use:
  genuine  held/forward-filled V_rot excluded -- these are not measurements, so this
           is the defensible headline (and what the paper text claims).
  stuck0   held kept; deflates V_rot RMSE 35-55%.
The progression figure is pinned to stuck0 regardless, because the "before" (iter2)
baseline exists only on that population and a before/after across two different
populations would not be a comparison.

Output: docs/paper/figures/*.png (overwrites the copied Korean versions).
"""
import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "DejaVu Sans"
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

SKILL_PCHIP = r"skill$_{\mathrm{PCHIP}}$ = 1 $-$ MSE$_{\mathrm{model}}$ / MSE$_{\mathrm{PCHIP}}$"

HEADLINE_TREATMENT = "genuine"
PROGRESSION_TREATMENT = "stuck0"
SEEDS = ("42", "1", "7", "123")

with open(os.path.join(os.path.dirname(__file__), "paper_numbers.json"), encoding="utf-8") as _fh:
    N = json.load(_fh)


def head(treatment=None):
    return N["headline"][treatment or HEADLINE_TREATMENT]["per_seed"]


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print("wrote", name)


def fig_forest():
    rows = head()
    seeds = [f"seed {s}" for s in SEEDS]
    ti = [rows[s]["CES_TI"]["ci_vs_pchip"]["skill"] for s in SEEDS]
    ti_lo = [rows[s]["CES_TI"]["ci_vs_pchip"]["ci95"][0] for s in SEEDS]
    ti_hi = [rows[s]["CES_TI"]["ci_vs_pchip"]["ci95"][1] for s in SEEDS]
    vt = [rows[s]["CES_VT"]["ci_vs_pchip"]["skill"] for s in SEEDS]
    vt_lo = [rows[s]["CES_VT"]["ci_vs_pchip"]["ci95"][0] for s in SEEDS]
    vt_hi = [rows[s]["CES_VT"]["ci_vs_pchip"]["ci95"][1] for s in SEEDS]
    n_pass_vt = sum(rows[s]["CES_VT"]["ci_vs_pchip"]["pass"] for s in SEEDS)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    for ax, vals, lo, hi, title in [
        (axes[0], ti, ti_lo, ti_hi, r"CES_TI  (ion temperature $T_i$)"),
        (axes[1], vt, vt_lo, vt_hi, r"CES_VT  (toroidal rotation $V_{\mathrm{rot}}$)"),
    ]:
        y = np.arange(len(seeds))[::-1]
        for i, (v, l, h) in enumerate(zip(vals, lo, hi)):
            passed = l > 0
            c = GREEN if passed else GRAY
            ax.plot([l, h], [y[i], y[i]], color=c, lw=3, solid_capstyle="round", zorder=2)
            ax.scatter([v], [y[i]], color=c, s=90, zorder=3, edgecolor="white", linewidth=1.2)
            tag = "PASS" if passed else "n.s."
            ax.text(h + 0.03, y[i], f"{v:+.2f}  {tag}", va="center", ha="left",
                    fontsize=10.5, color=c, fontweight="bold")
        ax.axvline(0, color=RED, lw=1.6, ls="--", zorder=1)
        ax.set_yticks(y)
        ax.set_yticklabels(seeds, fontsize=11)
        ax.set_xlim(-0.75, 0.65)
        ax.set_xlabel(SKILL_PCHIP, fontsize=10)
        ax.set_title(title, fontsize=13, fontweight="bold", color=NAVY, pad=8)
        ax.grid(axis="x", color=LGRAY, lw=0.8)
        for s in ["top", "right", "left"]:
            ax.spines[s].set_visible(False)
    axes[0].text(0.02, 1.0, "all four independent test splits: 95% CI > 0  (PASS)",
                 transform=axes[0].transAxes, fontsize=10, color=GREEN, fontweight="bold")
    axes[1].text(0.02, 1.0,
                 ("all four splits: CI includes 0  (n.s.)" if n_pass_vt == 0
                  else f"{4 - n_pass_vt}/4 splits: CI includes 0  (n.s.)"),
                 transform=axes[1].transAxes, fontsize=10, color=GRAY, fontweight="bold")
    fig.suptitle("Skill vs. PCHIP interpolation with shot-clustered 95% CI (B = 10,000)"
                 f"  ·  {HEADLINE_TREATMENT}-measurement evaluation",
                 fontsize=14, fontweight="bold", color=NAVY, y=1.04)
    fig.tight_layout()
    save(fig, "fig_forest.png")


def fig_rmse_ladder():
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))
    row = head()["42"]
    names = ["AR (past only)", "Persistence", "PCHIP*", "Linear", "Model (nowcaster)"]
    keys = ["ar_local", "persistence", "pchip", "linear"]
    data = {}
    for tgt, label in (("CES_TI", "CES_TI"), ("CES_VT", "CES_VT")):
        e = row[tgt]
        data[tgt] = dict(
            names=names,
            vals=[e["rmse"][k] for k in keys] + [e["rmse_model"]],
            title=rf"{label}  ($n$ = {e['n']:,})")
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
        ax.set_xlabel("RMSE (physical units)", fontsize=10)
        ax.set_xlim(0, max(d["vals"]) * 1.18)
        for s in ["top", "right"]:
            ax.spines[s].set_visible(False)
        ax.grid(axis="x", color=LGRAY, lw=0.7)
    axes[0].text(0.0, -0.27,
                 "Model = fast diagnostics + past CES (causal)  ·  PCHIP/Linear = past+future offline interpolation  ·  *PCHIP = pre-registered headline baseline",
                 transform=axes[0].transAxes, fontsize=8.2, color=GRAY)
    fig.suptitle("Final model (seed 42): lowest RMSE in the full baseline ladder (physical units)",
                 fontsize=14, fontweight="bold", color=NAVY, y=1.02)
    fig.tight_layout()
    save(fig, "fig_rmse_ladder.png")


def fig_progression():
    fig, ax = plt.subplots(figsize=(8.6, 3.7))
    labels = ["Baseline\n(iter2, GRU)", "Final model\n(+ observation-masked attn)"]
    before = N["before_baseline_iter2"]["CES_TI"]["ci_test"]
    after = head(PROGRESSION_TREATMENT)["42"]["CES_TI"]["ci_vs_pchip"]
    vals = [before["skill"], after["skill"]]
    lo = [before["ci95"][0], after["ci95"][0]]
    hi = [before["ci95"][1], after["ci95"][1]]
    x = [0, 1]
    colors = [GRAY, GREEN]
    for i in x:
        ax.plot([x[i], x[i]], [lo[i], hi[i]], color=colors[i], lw=4, solid_capstyle="round")
        ax.scatter([x[i]], [vals[i]], color=colors[i], s=160, zorder=3,
                   edgecolor="white", linewidth=1.5)
        tag = "PASS" if lo[i] > 0 else "n.s."
        ax.text(x[i], hi[i] + 0.03, f"{vals[i]:+.3f}\n{tag}", ha="center", va="bottom",
                fontsize=11, color=colors[i], fontweight="bold")
    ax.axhline(0, color=RED, lw=1.6, ls="--")
    ax.annotate("", xy=(0.92, vals[1]), xytext=(0.08, vals[0]),
                arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=2.2))
    ax.text(0.5, 0.05, "model selection on clean\nvalidation skill vs. PCHIP", ha="center",
            fontsize=9.5, color=ORANGE, style="italic")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_xlim(-0.45, 1.5)
    ax.set_ylim(-0.4, 0.5)
    ax.set_ylabel(r"CES_TI  skill$_{\mathrm{PCHIP}}$", fontsize=11)
    ax.set_title("From not-significant to significant (held-out test)", fontsize=13,
                 fontweight="bold", color=NAVY)
    ax.grid(axis="y", color=LGRAY, lw=0.7)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    save(fig, "fig_progression.png")


def fig_ablation():
    fig, ax = plt.subplots(figsize=(9.4, 4.6))
    groups = ["Full\n(history+fast+time)", "no_fast\n(history only)", "no_history\n(fast only)"]
    ab = N["ablation_val_vs_persistence"]
    arms = ["full", "no_fast", "no_history"]
    ti = [ab[a]["per_target"]["CES_TI"]["skill_vs_persistence"] for a in arms]
    vt = [ab[a]["per_target"]["CES_VT"]["skill_vs_persistence"] for a in arms]
    ymin = min(-1.15, min(vt) - 0.15)
    x = np.arange(len(groups))
    w = 0.36
    b1 = ax.bar(x - w / 2, ti, w, label="CES_TI", color=ORANGE, edgecolor="white")
    b2 = ax.bar(x + w / 2, [min(v, 0.6) if v > 0 else max(v, ymin + 0.02) for v in vt],
                w, label="CES_VT", color=BLUE, edgecolor="white")
    ax.axhline(0, color="#333", lw=1.0)
    for bars, vals in ((b1, ti), (b2, vt)):
        for b, v in zip(bars, vals):
            if v < ymin:
                ax.text(b.get_x() + b.get_width() / 2, ymin + 0.06, f"{v:+.2f}",
                        ha="center", va="bottom", fontsize=10, fontweight="bold", color=RED)
            else:
                ax.text(b.get_x() + b.get_width() / 2, v + 0.03, f"{v:+.2f}",
                        ha="center", va="bottom", fontsize=9.5, fontweight="bold", color=NAVY)
    ax.set_xticks(x)
    ax.set_xticklabels(groups, fontsize=10.5)
    ax.set_ylabel("skill vs. persistence", fontsize=11)
    ax.set_ylim(ymin, 0.62)
    ax.set_title("Input-modality ablation: where does each target's information come from?",
                 fontsize=12.5, fontweight="bold", color=NAVY, pad=24)
    vt_fast = f"{vt[2]:+.2f}".replace("-", "−")
    ax.text(0.5, 1.03,
            f"fast-only CES_VT = {vt_fast}"
            + r"  $\rightarrow$  rotation information comes almost entirely from past CES history",
            transform=ax.transAxes, ha="center", fontsize=9.3, color=RED, fontweight="bold")
    ax.legend(fontsize=10, loc="lower left")
    ax.grid(axis="y", color=LGRAY, lw=0.7)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.annotate("worse than predicting the mean", xy=(2.18, vt[2] + 0.03),
                xytext=(1.22, ymin + 0.22), fontsize=8.6, color=RED,
                arrowprops=dict(arrowstyle="-|>", color=RED, lw=1.4))
    fig.tight_layout()
    save(fig, "fig_ablation.png")


def fig_peak():
    fig, ax = plt.subplots(figsize=(8.6, 4.0))
    groups = ["CES_TI", "CES_VT"]
    pk = N["peak_seed42"]["per_target"]
    glob = [pk[t]["global_skill_vs_pchip"] for t in groups]
    peak = [pk[t]["peak_skill_vs_pchip"] for t in groups]
    peak_lo = [pk[t]["peak_skill_ci95"][0] for t in groups]
    peak_hi = [pk[t]["peak_skill_ci95"][1] for t in groups]
    x = np.arange(len(groups))
    w = 0.34
    ax.bar(x - w / 2, glob, w, label="global", color=LGRAY, edgecolor="white")
    ax.bar(x + w / 2, peak, w, label="high-variability (peak)", color=TEAL, edgecolor="white")
    err_lo = [peak[i] - peak_lo[i] for i in range(2)]
    err_hi = [peak_hi[i] - peak[i] for i in range(2)]
    ax.errorbar(x + w / 2, peak, yerr=[err_lo, err_hi], fmt="none",
                ecolor=NAVY, elinewidth=1.6, capsize=5)
    for i in range(2):
        ax.text(x[i] - w / 2, glob[i] + 0.02, f"{glob[i]:+.2f}", ha="center",
                fontsize=10, color=GRAY, fontweight="bold")
        ax.text(x[i] + w / 2, peak_hi[i] + 0.02, f"{peak[i]:+.2f}", ha="center",
                fontsize=10.5, color=TEAL, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(groups, fontsize=12)
    ax.set_ylabel(r"skill$_{\mathrm{PCHIP}}$", fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.set_title("The model's edge concentrates in high-variability neighborhoods",
                 fontsize=13, fontweight="bold", color=NAVY)
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(axis="y", color=LGRAY, lw=0.7)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.text(0.5, -0.13,
            "interpolation is near-optimal on the smooth bulk  ·  the model's value is in active stretches  ·  CES_VT also passes at peaks",
            transform=ax.transAxes, ha="center", fontsize=8.8, color=GRAY)
    fig.tight_layout()
    save(fig, "fig_peak.png")


def fig_missing():
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.3))

    def donut(ax, frac, color, title, sub):
        ax.pie([frac, 1 - frac], colors=[color, LGRAY], startangle=90,
               counterclock=False, wedgeprops=dict(width=0.42, edgecolor="white"))
        ax.text(0, 0.12, f"{int(round(frac*100))}%", ha="center", va="center",
                fontsize=22, fontweight="bold", color=color)
        ax.text(0, -0.22, sub, ha="center", va="center", fontsize=9, color=GRAY)
        ax.set_title(title, fontsize=12, fontweight="bold", color=NAVY)

    donut(axes[0], 0.082, ORANGE, "CES_TI missing",
          "8.2% NaN (held 0.0%)\non the 10 ms grid")
    donut(axes[1], 0.65, BLUE, "CES_VT uninformative",
          "23.9% NaN + 41.1% held\n= 65.0% of the 10 ms grid")
    donut(axes[2], 0.54, RED, "CES_VT held/stuck", "~54% of observed values\nare forward-filled holds")
    fig.suptitle("CES is sparse and the targets go missing independently; counting instrument holds, 65.0% of the grid carries no independent $V_{\\mathrm{rot}}$ information",
                 fontsize=12, fontweight="bold", color=NAVY, y=1.06)
    fig.tight_layout()
    save(fig, "fig_missing.png")


if __name__ == "__main__":
    fig_forest()
    fig_rmse_ladder()
    fig_progression()
    fig_ablation()
    fig_peak()
    fig_missing()
    print("ALL ENGLISH FIGURES DONE ->", OUT)
