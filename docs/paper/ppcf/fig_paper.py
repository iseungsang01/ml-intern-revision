"""Figures of the main text of main_ppcf.tex (except the example traces, fig_trace_causal.py).

    py docs/paper/ppcf/fig_paper.py   # -> docs/paper/ppcf/fig_{geometry,schematic,skill,where,blips,use}.pdf/.png

Every plotted number is read from a JSON output of the repository (b39_primary, b54_pooled_rows,
b56_kfx, b57_yardsticks, b58_blip_calibration, b62_beam_return, b42_delay); the geometry figure
draws positions given in the text of Supplementary section S10 and its Table S7 (literature,
typed below and marked as published or inferred there).  Sizes are set for a 16 cm text width
and printed 1:1 (fonts 8 to 9 pt).  A filled marker means that the 95 % interval excludes the
reference (0 for skills, 1 for ratios); a hollow one that it includes it.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402

HERE = Path(__file__).resolve().parent
PAPER = HERE.parent
W = 6.3  # inches, text width of the main text

NET, TWIN, GP, HOLD, OFF, FILT, FILT2 = "#c62828", "#2e7d32", "#1f77b4", "#7f7f7f", "#6a3d9a", "#e07b00", "#f2b25c"
INK, MUTED, GRID = "#111111", "#666666", "#dddddd"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.labelsize": 8.5,
                     "xtick.labelsize": 8, "ytick.labelsize": 8.5, "legend.fontsize": 8, "pdf.fonttype": 42,
                     "axes.spines.top": False, "axes.spines.right": False})


def J(name):
    return json.loads((PAPER / name).read_text(encoding="utf-8"))


def excl(v, ref):
    return v[1] > ref or v[2] < ref


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(HERE / f"{name}.{ext}", dpi=200, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print("written", name)


def irows(ax, rows, ref=0.0, xlim=None, fmt="{:+.3f}", label_x=None):
    """rows: (label, [point, lo, hi], colour) from top to bottom; None as a row leaves a gap."""
    n = len(rows)
    ticks, labels = [], []
    for i, r in enumerate(rows):
        if r is None:
            continue
        lab, v, c = r
        y = n - 1 - i
        ax.plot([v[1], v[2]], [y, y], color=c, lw=1.8, solid_capstyle="butt", zorder=2)
        ax.plot(v[0], y, "o", ms=5.5, mec=c, mew=1.4, mfc=c if excl(v, ref) else "white", zorder=3)
        if label_x is not None:
            ax.text(label_x, y, fmt.format(v[0]).replace("-", "−"), va="center", ha="left", fontsize=8,
                    color=INK)
        ticks.append(y)
        labels.append(lab)
    ax.axvline(ref, color=MUTED, lw=0.8, zorder=1)
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels)
    ax.set_ylim(-0.7, n - 0.3)
    if xlim:
        ax.set_xlim(*xlim)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.grid(axis="x", color=GRID, lw=0.6, zorder=0)


def fill_legend(fig, y, ref="zero"):
    h = [Line2D([], [], marker="o", ls="-", color=INK, mfc=INK, ms=5, lw=1.4, label=f"95% interval excludes {ref}"),
         Line2D([], [], marker="o", ls="-", color=INK, mfc="white", ms=5, lw=1.4, label=f"95% interval includes {ref}")]
    fig.legend(handles=h, loc="lower center", bbox_to_anchor=(0.5, y), ncol=2, frameon=False)


# ------------------------------------------------------------------------------------------------
def fig_geometry():
    """Positions along the major radius (Supplementary S10 and Table S7)."""
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(W, 3.3), sharex=True,
                                 gridspec_kw={"height_ratios": [1.1, 1.5], "hspace": 0.1})
    R = np.linspace(2.08, 2.35, 500)
    # schematic pedestal shape only: no measured profile of these discharges is in the files
    ped = 0.42 * (1 - np.tanh((R - 2.222) / 0.009)) / 2
    prof = 0.06 + ped + 0.28 * np.clip((2.21 - R) / 0.13, 0, None)
    a1.axvspan(2.19, 2.23, color="#f3e3c3", zorder=0)
    a1.plot(R, prof, color=MUTED, lw=1.4, ls="--")
    a1.text(2.187, 1.0, "band: published pedestal tops,\nother discharges and campaigns", ha="right", va="top",
            fontsize=7.4,
            color="#7a5a12")
    a1.axvline(2.24, color=INK, lw=1.0, ls=":")
    a1.text(2.244, 0.98, "midplane separatrix,\none 2022 discharge\n(read from a figure)", fontsize=7.4,
            va="top", ha="left")
    a1.text(2.345, 0.08, "schematic $T_i$ shape,\nnot measured", fontsize=7.4, color=MUTED, ha="right",
            va="bottom")
    a1.axvline(2.198, color=NET, lw=1.6)
    a1.text(2.195, 0.04, "CES chord, $R$ = 2.198 m", color=NET, fontsize=7.8, ha="right", va="bottom",
            fontweight="bold")
    a1.set_ylim(0, 1.05)
    a1.set_yticks([])
    a1.spines["left"].set_visible(False)

    rows = [  # label, (lo, hi) or points, published (True) or inferred (False), colour
        ("CES channels\n(user guide)", [2.192, 2.198, 2.203], True, NET),
        ("BES view, 31923\n(read from a figure)", (2.12, 2.34), True, FILT2),
        ("BES columns 5–7\nof the files", (2.16, 2.21), False, FILT),
        ("ECEI column 3 (from a figure;\n±2–3 cm between discharges)", (2.165, 2.215), True, GP),
    ]
    for i, (lab, x, pub, c) in enumerate(rows):
        y = len(rows) - 1 - i
        if isinstance(x, list):
            for xx in x:
                a2.plot(xx, y, "|", ms=11, mew=2.2 if xx == 2.198 else 1.2, color=c)
        else:
            a2.add_patch(Rectangle((x[0], y - 0.22), x[1] - x[0], 0.44, fc=c if pub else "white", ec=c,
                                   hatch=None if pub else "////", lw=1.2, alpha=0.85 if pub else 1.0))
    a2.set_yticks([len(rows) - 1 - i for i in range(len(rows))])
    a2.set_yticklabels([r[0] for r in rows], fontsize=7.6)
    a2.tick_params(axis="y", length=0)
    a2.axvline(2.198, color=NET, lw=1.0, alpha=0.6)
    a2.axvline(2.24, color=INK, lw=1.0, ls=":")
    a2.set_ylim(-0.6, len(rows) - 0.4)
    a2.spines["left"].set_visible(False)
    a2.set_xlim(2.08, 2.35)
    a2.set_xlabel("major radius $R$ (m); the Mirnov coils are at the wall, outside the frame")
    h = [Rectangle((0, 0), 1, 1, fc=MUTED, ec=MUTED, label="published position"),
         Rectangle((0, 0), 1, 1, fc="white", ec=MUTED, hatch="////", label="inferred position")]
    a2.legend(handles=h, loc="upper center", frameon=False, ncol=2, bbox_to_anchor=(0.5, -0.32))
    save(fig, "fig_geometry")


# ------------------------------------------------------------------------------------------------
def box(ax, x, y, w, h, text, fc, ec, fs=7.8, bold=False, ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01,rounding_size=0.02", fc=fc, ec=ec,
                                lw=1.1, ls=ls))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal")


def arrow(ax, x0, y0, x1, y1, c=INK, ls="-"):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=c, lw=1.0, ls=ls, shrinkA=1, shrinkB=1))


def fig_schematic():
    fig = plt.figure(figsize=(W, 3.3))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, 3.3)
    ax.axis("off")
    ax.text(0.02, 3.27, "(a) recurrent estimator and its history-only twin", fontsize=8.5, va="top",
            fontweight="bold")
    box(ax, 0.02, 2.35, 2.5, 0.55, "fast channels at row $t$, one sample each:\n9 BES, 4 ECEI, 2 Mirnov",
        "#fff1df", FILT)
    box(ax, 0.02, 1.65, 2.5, 0.55, "CES history (rows before $t$): last value,\nits age, observed flag; time step",
        "#e8f0fa", GP)
    box(ax, 2.95, 1.9, 1.3, 0.75, "LSTM, 2 layers,\n160 units, run over\nevery grid row", "#fbe9e9", NET)
    box(ax, 4.6, 2.1, 0.6, 0.35, r"$\hat T_i(t)$", "white", NET, fs=9)
    arrow(ax, 2.52, 2.62, 2.95, 2.42, FILT)
    arrow(ax, 2.52, 1.92, 2.95, 2.12, GP)
    arrow(ax, 4.25, 2.27, 4.6, 2.27, NET)
    ax.text(4.45, 1.95, "twin: the same network\nwithout the fast channels;\nincrement = skill of the\n"
                        "estimator over its twin", fontsize=7.4, va="top")
    ax.plot([0.02, W - 0.02], [1.45, 1.45], color=GRID, lw=0.8)
    ax.text(0.02, 1.4, r"(b) five-parameter robust filter on $\log T_i$", fontsize=8.5, va="top",
            fontweight="bold")
    box(ax, 0.02, 0.5, 2.35, 0.6, r"predict: $x_t = x_{t-1} + b_c\,\Delta c_t + b_g\,\Delta g_t$" "\n"
        "(random walk with variance $q$)", "#fff1df", FILT)
    box(ax, 2.8, 0.5, 2.0, 0.6, "update with a CES value,\n" r"weight $\min(1, k\sqrt{S}/|v|)$, variance $r$",
        "#e8f0fa", GP)
    arrow(ax, 2.37, 0.8, 2.8, 0.8)
    ax.annotate("", xy=(1.2, 0.5), xytext=(3.8, 0.5),
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.0, connectionstyle="arc3,rad=-0.22"))
    ax.text(2.5, 0.02, "next row", ha="center", va="bottom", fontsize=7.2, color=MUTED)
    ax.text(4.95, 1.15, "$c$: mean log of the nine BES\nchannels (common mode)\n"
                        "$g$: column 7 minus column 5\n(radial difference)\n"
                        "KF0: no drive; KFX: both\ngains; Kg: $b_g$ only", fontsize=7.2, va="top", ha="left")
    save(fig, "fig_schematic")


# ------------------------------------------------------------------------------------------------
def fig_skill():
    P = J("b39_primary.json")["P"]
    C = J("b56_kfx.json")["confirm"]
    Y = J("b57_yardsticks.json")["all"]
    rows = [("over hold", P["causal_over"]["persistence"], NET),
            ("over causal GP", P["P1_causal_over_gp"], NET),
            ("over its twin (increment)", P["P2_causal_over_twin"], NET),
            ("over offline GP (uses next value)", P["causal_over"]["gp"], NET),
            None,
            ("KFX over KF0 (increment)", C["kfx_over_kf0"], FILT),
            ("Kg (radial difference) over KF0", C["kg_over_kf0"], FILT),
            ("twin over KF0", C["Tw_over_kf0"], TWIN),
            ("KFX over causal GP", C["kfx_over_G"], FILT),
            ("recurrent estimator over KFX", C["A_over_kfx"], NET)]
    fig, (a, b) = plt.subplots(1, 2, figsize=(W - 0.95, 2.9), gridspec_kw={"width_ratios": [1.35, 1.0], "wspace": 0.95})
    irows(a, rows, 0.0, (-0.09, 0.42))
    a.text(-0.09, 9.55, "recurrent estimator (r2–r9)", fontsize=7.6, color=NET, va="bottom")
    a.text(-0.09, 4.55, "filter family (r3–r9)", fontsize=7.6, color=FILT, va="bottom")
    a.set_xlabel("mean-squared-error skill")
    a.set_title("(a) skill with 95% interval", fontsize=8.5, loc="left", fontweight="bold", pad=12)

    est = [("hold", "H", HOLD), ("causal GP", "G", GP), ("KF0", "K0", FILT2), ("twin", "Tw", TWIN),
           ("KFX", "KX", FILT), ("recurrent est.", "A", NET), ("offline GP", "O", OFF)]
    n = len(est)
    for i, (lab, k, c) in enumerate(est):
        y = n - 1 - i
        v = Y["mse_over_s2_" + k]
        b.barh(y, 1.0, height=0.6, color="#d9d9d9", zorder=2)
        b.barh(y, v[0] - 1.0, left=1.0, height=0.6, color=c, alpha=0.85, zorder=2)
        b.plot([v[1], v[2]], [y, y], color=INK, lw=0.9, zorder=3)
        b.text(2.45, y, f"{v[0]:.2f}", va="center", ha="left", fontsize=8)
    b.axvline(1.0, color=INK, lw=0.9)
    b.set_yticks(range(n))
    b.set_yticklabels([e[0] for e in reversed(est)])
    b.set_xlim(0, 2.45)
    b.set_ylim(-0.6, n - 0.4)
    b.tick_params(axis="y", length=0)
    b.spines["left"].set_visible(False)
    b.set_xlabel("mean squared error / floor $s^2$")
    b.set_title("(b) error against the floor", fontsize=8.5, loc="left", fontweight="bold", pad=12)
    b.text(0.5, -0.95, "floor", ha="center", fontsize=7.4, color=MUTED, va="top")
    fig.subplots_adjust(bottom=0.2)
    fill_legend(fig, -0.04)
    save(fig, "fig_skill")


# ------------------------------------------------------------------------------------------------
def fig_where():
    R = J("b54_pooled_rows.json")["reported"]
    K = J("b56_kfx.json")
    th = [("coldest", "low", "confirm_third_coldest"), ("middle", "middle", "confirm_third_middle"),
          ("hottest", "high", "confirm_third_hottest")]
    fig, (a, b) = plt.subplots(1, 2, figsize=(W, 2.6), gridspec_kw={"width_ratios": [1.2, 1.0], "wspace": 0.9})
    series = [("recurrent estimator over causal GP", lambda t: R["thirds"][t[1]]["gp"], GP, -0.22),
              ("recurrent estimator over its twin", lambda t: R["thirds"][t[1]]["t"], NET, 0.0),
              ("KFX over KF0 (filter increment)", lambda t: K[t[2]]["kfx_over_kf0"], FILT, 0.22)]
    for lab, f, c, dx in series:
        for i, t in enumerate(th):
            v = f(t)
            a.plot([i + dx, i + dx], [v[1], v[2]], color=c, lw=1.8)
            a.plot(i + dx, v[0], "o", ms=5.5, mec=c, mew=1.4, mfc=c if excl(v, 0) else "white",
                   label=lab if i == 0 else None)
    a.axhline(0, color=MUTED, lw=0.8)
    a.set_xticks(range(3))
    a.set_xticklabels([t[0] for t in th])
    a.set_xlim(-0.5, 2.5)
    a.set_ylabel("skill")
    a.set_xlabel("third of the discharges by median $T_i$\n(edges 336 and 667 eV)")
    a.grid(axis="y", color=GRID, lw=0.6)
    a.legend(handles=[Line2D([], [], marker="o", ls="-", color=c, mfc=c, ms=5, lw=1.4, label=lab)
                       for lab, _, c, _ in series], loc="upper left", frameon=False, fontsize=7.2,
             handlelength=1.2)
    a.set_ylim(-0.04, 0.27)
    a.set_title("(a) by temperature level", fontsize=8.5, loc="left", fontweight="bold")

    rows = [("not flagged", R["plain_rows"]["t"], NET),
            ("flagged by value rules", R["flagged_rows"]["t"], NET),
            ("1–3 after a beam blip", R["blip_rows_1_3"]["t"], NET)]
    irows(b, rows, 0.0, (-0.04, 0.24))
    b.set_xticks([0, 0.1, 0.2])
    b.set_xlabel("skill over the twin")
    b.set_title("(b) by row class", fontsize=8.5, loc="left", fontweight="bold")
    fig.subplots_adjust(bottom=0.27)
    fill_legend(fig, -0.06)
    save(fig, "fig_where")


# ------------------------------------------------------------------------------------------------
def fig_blips():
    S = J("b58_blip_calibration.json")["S"]["all"]
    E = J("b62_beam_return.json")["E"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(W, 2.6), gridspec_kw={"wspace": 0.32})
    pr, ci, ctl = S["profiles"], S["profiles_ci95"], S["control_profiles"]
    x = np.arange(1, len(pr["ti"]) + 1)
    for k, lab, c in (("bes", "BES level", HOLD), ("ecei", "ECEI level", FILT), ("ti", "reported $T_i$", GP)):
        lo = [p[0] for p in ci[k]]
        hi = [p[1] for p in ci[k]]
        a.fill_between(x, lo, hi, color=c, alpha=0.2, lw=0)
        a.plot(x, pr[k], "-o", color=c, lw=1.6, ms=3, label=lab)
        a.plot(x, ctl[k], ":", color=c, lw=1.0)
    a.axhline(1, color=GRID, lw=0.8)
    a.set_xticks(x)
    a.set_xlim(0.6, len(x) + 0.4)
    a.set_ylim(0.62, 1.06)
    a.set_xlabel("row after the absent frame (10 ms rows)")
    a.set_ylabel("ratio to the six rows before")
    a.legend(loc="lower right", frameon=False, fontsize=7.6)
    a.text(len(x) + 0.3, 1.035, "dotted: controls", ha="right", fontsize=7.2, color=MUTED)
    a.set_title(f"(a) 10 ms files, {S['events']:,} blips", fontsize=8.5, loc="left",
                fontweight="bold")

    for k, lab, c in (("BES", "BES level", HOLD), ("ECEI", "ECEI level", FILT)):
        t = E[k]["profile_return_aligned_ms"]
        m = E[k]["profile_return_aligned_median"]
        cc = E[k]["profile_return_aligned_ci95"]
        b.fill_between(t, [p[0] for p in cc], [p[1] for p in cc], color=c, alpha=0.2, lw=0)
        b.plot(t, m, color=c, lw=1.4, label=lab)
    b.axvspan(-10.5, 0, color="#e6e6e6", zorder=0)
    b.text(-5.2, 0.05, "beam off\n(median)", ha="center", fontsize=7.2, color=MUTED)
    b.axhline(1, color=GRID, lw=0.8)
    b.set_xlabel("time from the return of the beam (ms)")
    b.set_ylabel("ratio to the 20 ms before beam-off")
    b.set_ylim(0, 1.12)
    b.legend(loc="lower right", frameon=False, fontsize=7.6)
    b.set_title(f"(b) raw streams, {E['ecei_readout_interruptions']} interruptions", fontsize=8.5,
                loc="left", fontweight="bold")
    save(fig, "fig_blips")


# ------------------------------------------------------------------------------------------------
def fig_use():
    Y = J("b57_yardsticks.json")["all"]
    D = J("b42_delay.json")
    fig, (a, b) = plt.subplots(1, 2, figsize=(W - 0.3, 2.7), gridspec_kw={"width_ratios": [1.0, 1.0], "wspace": 0.42})
    rows = [("network $-$ twin, $\\rho_1$", Y["net_twin_rho1"], NET),
            ("network $-$ twin, $\\rho_3$", Y["net_twin_rho3"], NET),
            ("KFX $-$ KF0, $\\rho_1$", Y["kfx_kf0_rho1"], FILT),
            ("KFX $-$ KF0, $\\rho_3$", Y["kfx_kf0_rho3"], FILT),
            ("twin $-$ KF0, $\\rho_1$", Y["twin_kf0_rho1"], TWIN)]
    irows(a, rows, 1.0, (0.55, 1.3), fmt="{:.3f}")
    a.set_xticks([0.6, 0.8, 1.0, 1.2])
    a.axvline(0.0, color=MUTED, lw=0)
    a.set_xlabel("$\\rho_h$: share of a correction still\npresent $h$ frames later")
    a.set_title("(a) persistence", fontsize=8.5, loc="left", fontweight="bold", pad=8)
    seeds = ("42", "1", "7", "123")
    delays = D["delays_frames"]
    for key, c, lab in (("causal_vs_gp", GP, "over causal GP"), ("causal_vs_twin", NET, "over its twin")):
        for j, s_ in enumerate(seeds):
            vals = [D["splits"][s_][str(d)]["skill"][key] for d in delays]
            b.plot(delays, [v[0] for v in vals], "-", color=c, lw=0.9, alpha=0.7, label=lab if j == 0 else None)
            for d, v in zip(delays, vals):
                b.plot(d, v[0], "o", ms=4, mec=c, mew=1.0, mfc=c if excl(v, 0) else "white")
    b.axhline(0, color=MUTED, lw=0.8)
    b.set_xticks(delays)
    b.set_xticklabels([str(d) for d in delays])
    b.set_xlabel("delay of the CES value (frames)\n(last value 10, 20, 30, 50 ms old)")
    b.set_ylabel("skill of the network")
    b.set_ylim(0, 0.24)
    b.grid(axis="y", color=GRID, lw=0.6)
    b.legend(loc="upper left", frameon=False, fontsize=7.4, ncol=2, handlelength=1.2, columnspacing=0.8)
    b.set_title("(b) late CES value, four splits", fontsize=8.5, loc="left", fontweight="bold", pad=8)
    fig.subplots_adjust(bottom=0.3)
    fill_legend(fig, -0.04, ref="the reference")
    save(fig, "fig_use")

# ------------------------------------------------------------------------------------------------
def fig_rows():
    """What a row of the 10 ms files holds (schematic; Supplementary S10, docs/paper/file_rules.md)."""
    fig = plt.figure(figsize=(W, 2.45))
    ax = fig.add_axes([0.2, 0.17, 0.78, 0.8])
    ax.set_xlim(-52, 30)
    ax.set_xticks([-50, -40, -30, -20, -10, 0, 10])
    ax.spines["bottom"].set_bounds(-52, 10)
    ax.set_ylim(-0.6, 3.6)
    for sp in ("left", "right", "top"):
        ax.spines[sp].set_visible(False)
    ax.set_yticks([3, 2, 1, 0])
    ax.set_yticklabels(["heating beam", "CES frames\n(10 ms exposures)", "fast channels\n(0.5\u20132 MHz)",
                        "rows of the file"], fontsize=8)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("time relative to the stamp of row $t$ (ms)")
    # beam: off for one frame (a blip)
    ax.plot([-52, -30, -30, -20, -20, 10], [3.15, 3.15, 2.85, 2.85, 3.15, 3.15], color=INK, lw=1.2)
    ax.text(-25, 3.0, "blip", ha="center", va="center", fontsize=7.2)
    # CES frames
    starts = [-50, -40, -30, -20, -10, 0]
    for x0 in starts:
        absent = x0 == -30
        target = x0 == 0
        fc = "white" if absent else ("#fbe9e9" if target else "#e8f0fa")
        ec = MUTED if absent else (NET if target else GP)
        ax.add_patch(Rectangle((x0 + 0.4, 1.78), 9.2, 0.44, fc=fc, ec=ec, lw=1.1, ls="--" if absent else "-"))
    ax.text(-25.0, 2.0, "absent", ha="center", va="center", fontsize=7.0, color=MUTED)
    ax.text(5.0, 2.0, "frame $t$", ha="center", va="center", fontsize=7.4, color=NET)
    ax.text(11.0, 2.0, "frame $t$: estimated;\nearlier frames: CES history", ha="left", va="center", fontsize=7.0,
            color=INK)
    # fast channel: a continuous schematic signal and its single samples at the stamps
    tt = np.linspace(-52, 10, 600)
    sig = 1.0 + 0.12 * np.sin(tt / 2.3) + 0.06 * np.sin(tt / 0.7)
    sig = np.where((tt > -30) & (tt < -20), sig - 0.28, sig)
    ax.plot(tt, sig, color=FILT2, lw=0.9)
    stamps = [-50, -40, -20, -10, 0]
    for x0 in stamps:
        y0 = float(np.interp(x0 + 0.5, tt, sig))
        ax.plot(x0 + 0.5, y0, "o", ms=4.5, color=FILT if x0 == 0 else FILT2, mec=FILT)
        ax.plot([x0 + 0.5, x0 + 0.5], [0.12, y0 - 0.08], color=GRID, lw=0.8)
    ax.text(11.0, 1.05, "one raw sample per row,\ntaken at its stamp", ha="left", va="center", fontsize=7.0,
            color=INK)
    # rows
    for x0 in stamps:
        ax.plot(x0 + 0.5, 0, "s", ms=5, color=NET if x0 == 0 else INK)
    ax.annotate("", xy=(-19.5, -0.3), xytext=(-39.5, -0.3),
                arrowprops=dict(arrowstyle="<->", color=MUTED, lw=0.9))
    ax.text(-29.5, -0.42, "20 ms step = one absent frame", ha="center", va="top", fontsize=7.0, color=MUTED)
    ax.text(11.0, 0.0, "row $t$: estimate\nformed at the stamp", ha="left", va="center", fontsize=7.0,
            color=INK)
    save(fig, "fig_rows")



def main():
    for f in (fig_geometry, fig_rows, fig_schematic, fig_skill, fig_where, fig_blips, fig_use):
        f()


if __name__ == "__main__":
    main()
