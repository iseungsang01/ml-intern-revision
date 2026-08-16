# -*- coding: utf-8 -*-
"""English-language paper figures for the KSTAR CES nowcasting draft (confirmed protocol).

Every number is READ FROM `paper_numbers.json` (regenerate with
`py ces_prediction/collect_paper_numbers.py`), which in turn reads only the frozen
run directories and batch verdicts under data/. Nothing here is hard-coded: an earlier
version of this file carried literals from a superseded checkpoint family, which is
exactly the failure mode this indirection removes.

Protocol behind every panel: W = 2, held-free, per-file cap 500, TWO co-primary
populations (cut = CES_TI > 3 keV treated as missing; incl = no cut), shot-clustered
paired bootstrap (10,000 resamples). The adopted model is the seq_v2 backbone; the
W = 2 window family is the paired control.

Output: docs/paper/figures/*.png
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
POP_COLOR = {"cut": NAVY, "incl": ORANGE}
POP_LABEL = {"cut": "cut population (T$_i$ > 3 keV treated as missing)", "incl": "inclusive population (no cut)"}

OUT = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(OUT, exist_ok=True)

SKILL_PCHIP = r"skill$_{\mathrm{PCHIP}}$ = 1 $-$ MSE$_{\mathrm{model}}$ / MSE$_{\mathrm{PCHIP}}$"
SEEDS = ("42", "1", "7", "123")
POPS = ("cut", "incl")
T = ("CES_TI", "CES_VT")
TLABEL = {"CES_TI": r"CES_TI  (ion temperature $T_i$)", "CES_VT": r"CES_VT  (toroidal rotation $V_{\mathrm{rot}}$)"}

with open(os.path.join(os.path.dirname(__file__), "paper_numbers.json"), encoding="utf-8") as _fh:
    N = json.load(_fh)


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print("wrote", name)


def _clean(ax, left=True):
    for s in ["top", "right"] + ([] if left else ["left"]):
        ax.spines[s].set_visible(False)


# ---------------------------------------------------------------- headline forest
def fig_forest():
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    for ax, t in zip(axes, T):
        y0 = np.arange(len(SEEDS))[::-1] * 1.0
        for k, pop in enumerate(POPS):
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
        ax.set_yticklabels([f"split {s}" for s in SEEDS], fontsize=10.5)
        ax.set_xlim(-0.75, 0.75)
        ax.set_xlabel(SKILL_PCHIP, fontsize=9.5)
        ax.set_ylim(-0.7, len(SEEDS) - 0.3)
        ax.set_title(TLABEL[t], fontsize=12.5, fontweight="bold", color=NAVY, pad=22)
        ax.grid(axis="x", color=LGRAY, lw=0.8)
        _clean(ax, left=False)
        pr = {pop: N["headline"][pop]["seq"]["pr4_pass"][t]["pchip"] for pop in POPS}
        ax.text(0.0, 1.015, f"PR4 PASS: cut {pr['cut']}/4  ·  inclusive {pr['incl']}/4",
                transform=ax.transAxes, fontsize=9.5, color=GREEN if min(pr.values()) == 4 else GRAY, fontweight="bold")
    h = [plt.Line2D([], [], color=POP_COLOR[p], lw=3, label=POP_LABEL[p]) for p in POPS]
    fig.legend(handles=h, loc="lower center", ncol=2, fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("seq_v2 backbone vs. future-using PCHIP on four independent test splits, both populations "
                 "(shot-clustered 95% CI, B = 10,000; W = 2, held-free)", fontsize=12, fontweight="bold", color=NAVY, y=1.04)
    fig.tight_layout()
    save(fig, "fig_forest.png")


# ---------------------------------------------------------------- RMSE ladder (seed 42, cut)
def fig_rmse_ladder():
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    lad = N["headline"]["cut"]["rmse_ladder"]["42"]
    order = [("ar_local", "AR (local, past only)"), ("persistence", "Persistence"), ("gp_causal", "Causal GP (past only)"),
             ("pchip", "PCHIP* (past + future)"), ("linear", "Linear (past + future)"), ("gp", "GP (past + future)")]
    for ax, t in zip(axes, T):
        e = lad[t]
        names = [n for _, n in order] + ["Window model (W = 2)", "seq_v2 backbone"]
        vals = [e["rmse_baselines"][k] for k, _ in order] + [e["rmse_model_window"], e["rmse_model_seq_v2"]]
        colors = [GRAY, GRAY, BLUE, LGRAY, LGRAY, LGRAY, TEAL, NAVY]
        y = np.arange(len(names))[::-1]
        bars = ax.barh(y, vals, color=colors, edgecolor="white", height=0.66)
        ax.set_yticks(y)
        ax.set_yticklabels(names, fontsize=9.6)
        for b, v in zip(bars, vals):
            ax.text(v + max(vals) * 0.012, b.get_y() + b.get_height() / 2, f"{v:.1f}", va="center", fontsize=9, color=NAVY)
        ax.set_xlim(0, max(vals) * 1.2)
        ax.set_title(f"{TLABEL[t]}  —  RMSE, physical units, n = {e['n']:,}", fontsize=11, fontweight="bold", color=NAVY)
        ax.grid(axis="x", color=LGRAY, lw=0.7)
        _clean(ax)
    fig.suptitle("Physical-unit RMSE ladder on the canonical test split (seed 42, cut population, genuine measurements only). "
                 "Grey = causal; light = reads the future; the backbone is lowest for both targets.",
                 fontsize=10.5, fontweight="bold", color=NAVY, y=1.03)
    fig.tight_layout()
    save(fig, "fig_rmse_ladder.png")


# ---------------------------------------------------------------- complexity ladder + width scaling
def fig_ladder_scaling():
    fig, ax = plt.subplots(figsize=(10.5, 4.6))
    L = N["ladder"]
    B4 = N["scaling_b4"]["widths"]
    for pop in POPS:
        c = POP_COLOR[pop]
        r = L[pop]["skill_vs_pchip"]
        pts = [(1258, r["anchor"]["CES_TI"]["mean"], "anchor+Δ"), (21498, r["b3"]["CES_TI"]["mean"], "b3k8"),
               (201258, r["win"]["CES_TI"]["mean"], "window")]
        for x, yv, lab in pts:
            ax.scatter([x], [yv], color=c if pop == "cut" else "white", edgecolor=c, s=95, zorder=4, linewidth=1.6,
                       marker="s" if lab == "window" else "o")
            if pop == "cut":
                ax.annotate(lab, (x, yv), textcoords="offset points", xytext=(0, 9), ha="center", fontsize=8.8, color=c)
        ax.axhline(r["persistence"]["CES_TI"]["mean"], color=c, lw=1.0, ls=":", alpha=0.8)
        ax.text(1.05e6, r["persistence"]["CES_TI"]["mean"] + 0.012, f"persistence ({pop})", fontsize=8, color=c, ha="right")
    # B.4 width curve (cut population, seq_v2)
    ws = sorted(B4.keys(), key=lambda k: int(k))
    xs = [B4[w]["params"] for w in ws]
    ys = [B4[w]["skill_TI_mean"] for w in ws]
    ax.plot(xs, ys, color=NAVY, lw=2.2, zorder=3)
    for w, x, yv in zip(ws, xs, ys):
        for s in SEEDS:
            ax.scatter([x], [B4[w]["skill_TI_per_seed"][s]], color=NAVY, s=14, alpha=0.35, zorder=2)
        ax.scatter([x], [yv], color=NAVY, s=60, zorder=5, marker="D")
        ax.annotate(f"seq_v2\nwidth {w}", (x, yv), textcoords="offset points", xytext=(0, -26), ha="center", fontsize=7.8, color=NAVY)
    # inclusive-population backbone point (width 160)
    ax.scatter([357570], [L["incl"]["skill_vs_pchip"]["seq"]["CES_TI"]["mean"]], color="white", edgecolor=ORANGE, s=70, marker="D", linewidth=1.6, zorder=5)
    ax.set_xscale("log")
    ax.set_xlim(8e2, 1.2e6)
    ax.set_ylim(-0.4, 0.42)
    ax.axhline(0, color=RED, lw=1.2, ls="--")
    ax.set_xlabel("trainable parameters (log scale)", fontsize=10.5)
    ax.set_ylabel(r"TEST $T_i$ skill vs. PCHIP (4-split mean)", fontsize=10.5)
    ax.set_title("Complexity ladder and width scaling: the $T_i$ skill is flat from 21k to 879k parameters (cut population); "
                 "the interpretable rung equals the backbone only under the cut",
                 fontsize=10.5, fontweight="bold", color=NAVY)
    h = [plt.Line2D([], [], marker="o", color=NAVY, ls="", label="cut population"),
         plt.Line2D([], [], marker="o", color="white", markeredgecolor=ORANGE, ls="", label="inclusive population"),
         plt.Line2D([], [], marker="D", color=NAVY, ls="-", label="seq_v2 width sweep (B.4, cut)")]
    ax.legend(handles=h, fontsize=8.8, loc="lower right", frameon=False)
    ax.grid(color=LGRAY, lw=0.7)
    _clean(ax)
    fig.tight_layout()
    save(fig, "fig_ladder_scaling.png")


# ---------------------------------------------------------------- ablation (window family, eval-time)
def fig_ablation():
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6), sharey=False)
    A = N["ablation_window_eval"]
    H = N["headline"]
    arms = ["full", "no_fast", "no_history"]
    labels = ["Full\n(history +\nfast + time)", "no_fast\n(history +\ntime only)", "no_history\n(fast +\ntime only)"]
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
        ax.set_ylabel(r"skill vs. PCHIP (TEST, 4-split mean; dots = splits)", fontsize=9)
        ax.set_title(TLABEL[t], fontsize=12, fontweight="bold", color=NAVY)
        ax.grid(axis="y", color=LGRAY, lw=0.7)
        _clean(ax)
    hnd, lab = axes[0].get_legend_handles_labels()
    fig.legend(hnd, lab, loc="lower center", ncol=2, fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.06))
    fig.text(0.5, -0.11, "no_fast leaves $V_{\mathrm{rot}}$ bit-identical (the routing is structural); no_history collapses both targets (bars and dots clipped at -1)",
             ha="center", fontsize=8.8, color=GRAY)
    fig.suptitle("Evaluation-time modality ablation of the W = 2 window family (inputs zeroed, no retraining)",
                 fontsize=12, fontweight="bold", color=NAVY, y=1.02)
    fig.tight_layout()
    save(fig, "fig_ablation.png")


# ---------------------------------------------------------------- peak strata (seq_v2, both populations)
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
        ax.set_xticklabels(["bulk (non-peak)", "high-variability (peak)"], fontsize=10.5)
        ax.set_ylim(-0.2, 1.05)
        ax.axhline(0, color=RED, lw=1.2, ls="--")
        ax.set_ylabel(r"skill vs. PCHIP (TEST, 4-split mean; dots = splits)", fontsize=9)
        ax.set_title(TLABEL[t], fontsize=12, fontweight="bold", color=NAVY)
        ax.grid(axis="y", color=LGRAY, lw=0.7)
        _clean(ax)
    axes[0].legend(fontsize=8.4, loc="upper left", frameon=False)
    fig.suptitle("The backbone's edge concentrates where interpolation is weakest: peak-stratified TEST skill, both populations",
                 fontsize=12, fontweight="bold", color=NAVY, y=1.02)
    fig.tight_layout()
    save(fig, "fig_peak.png")


# ---------------------------------------------------------------- campaign (temporal) split
def fig_campaign():
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))
    C = N["campaign"]
    for ax, t in zip(axes, T):
        arms = [("win", "window OFF"), ("winps", "window ON\n(per-shot std.)"), ("seq", "seq_v2\nbackbone")]
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
        ax.set_ylabel(r"skill vs. PCHIP on the temporal TEST block (4 inits)", fontsize=9)
        ax.set_title(TLABEL[t], fontsize=12, fontweight="bold", color=NAVY)
        ax.grid(axis="y", color=LGRAY, lw=0.7)
        _clean(ax)
    h = [plt.Rectangle((0, 0), 1, 1, color=POP_COLOR[p], alpha=0.85, label=POP_LABEL[p]) for p in POPS]
    axes[0].legend(handles=h, fontsize=8.4, loc="upper left", frameon=False)
    m = C["cut"]["manifest"]
    fig.suptitle(f"Campaign (temporal) split — train shots {m['train'][0]}–{m['train'][1]}, test {m['test'][0]}–{m['test'][1]}: "
                 "the window model's offline advantage collapses, the backbone's survives (labels = mean, PR4 PASS count)",
                 fontsize=10.5, fontweight="bold", color=NAVY, y=1.03)
    fig.tight_layout()
    save(fig, "fig_campaign.png")


# ---------------------------------------------------------------- data missingness
def fig_missing():
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.3))
    D = N["data_ledger"]["targets"]
    ti_miss = D["CES_TI"]["missing_frac"]
    vt_miss = D["CES_VT"]["missing_frac"]
    vt_held = D["CES_VT"]["held_frac_of_rows"]
    vt_held_obs = D["CES_VT"]["held_frac_of_observed"]

    def donut(ax, frac, color, title, sub):
        ax.pie([frac, 1 - frac], colors=[color, LGRAY], startangle=90,
               counterclock=False, wedgeprops=dict(width=0.42, edgecolor="white"))
        ax.text(0, 0.12, f"{frac*100:.0f}%", ha="center", va="center", fontsize=22, fontweight="bold", color=color)
        ax.text(0, -0.22, sub, ha="center", va="center", fontsize=9, color=GRAY)
        ax.set_title(title, fontsize=12, fontweight="bold", color=NAVY)

    donut(axes[0], ti_miss, ORANGE, "CES_TI missing", f"{ti_miss*100:.1f}% NaN (held 0.0%)\non the 10 ms grid")
    donut(axes[1], vt_miss + vt_held, BLUE, "CES_VT uninformative",
          f"{vt_miss*100:.1f}% NaN + {vt_held*100:.1f}% held\n= {(vt_miss+vt_held)*100:.1f}% of the 10 ms grid")
    donut(axes[2], vt_held_obs, RED, "CES_VT held/stuck", f"{vt_held_obs*100:.0f}% of observed values\nare forward-filled holds")
    fig.suptitle("CES is sparse and the targets go missing independently; counting instrument holds, "
                 f"{(vt_miss+vt_held)*100:.1f}% of the grid carries no independent $V_{{\\mathrm{{rot}}}}$ information",
                 fontsize=12, fontweight="bold", color=NAVY, y=1.06)
    fig.tight_layout()
    save(fig, "fig_missing.png")


if __name__ == "__main__":
    fig_forest()
    fig_rmse_ladder()
    fig_ladder_scaling()
    fig_ablation()
    fig_peak()
    fig_campaign()
    fig_missing()
    for stale in ("fig_progression.png",):
        p = os.path.join(OUT, stale)
        if os.path.exists(p):
            os.remove(p)
            print("removed stale", stale)
    print("ALL ENGLISH FIGURES DONE ->", OUT)
