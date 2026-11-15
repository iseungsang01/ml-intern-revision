"""Figure 2: the two-branch recurrent estimator (schematic, no data).

Run from the repo root:  py docs/paper/ppcf/fig_arch.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

HERE = Path(__file__).resolve().parent
BLUE, ORANGE, GREY = "#4c72b0", "#dd8452", "#555555"
VR = "$V_" + chr(92) + "mathrm{rot}$"


def box(ax, x, y, w, h, text, fc, ec, fs=8.0, tc="black"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=fc, ec=ec, lw=1.0))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=tc,
            linespacing=1.25)


def arrow(ax, p, q, color=GREY, ls="-", lw=1.1):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=9, color=color,
                                 lw=lw, linestyle=ls, shrinkA=0, shrinkB=0))


def main():
    plt.rcParams.update({"font.family": "serif", "font.size": 8, "mathtext.fontset": "stix"})
    fig, ax = plt.subplots(figsize=(7.0, 3.0))
    ax.set_xlim(0, 14.2)
    ax.set_ylim(0, 6)
    ax.axis("off")

    # inputs
    box(ax, 0.1, 3.55, 4.4, 1.75,
        "15 fast channels at row $t$\n9 BES, 4 ECEI, 2 Mirnov\n(no gaps, 10 ms grid)", "#eef3fa", BLUE)
    box(ax, 0.1, 0.65, 4.4, 2.2,
        "7 CES-derived channels\ntime step; per target:\nlast observed value,\nstaleness, observed flag\n(rows before $t$ only)",
        "#fdf1e7", ORANGE)

    # branches
    box(ax, 5.7, 3.55, 4.1, 1.75, "$T_i$ branch\nLSTM, 2 layers, 160 units\nreads 22 channels", "#eef3fa", BLUE)
    box(ax, 5.7, 0.9, 4.1, 1.7, VR + " branch\nLSTM, 1 layer, 64 units\nreads 7 channels", "#fdf1e7", ORANGE)

    # outputs
    box(ax, 11.0, 3.85, 3.0, 1.15, "estimate of $T_i$\nat row $t$", "white", BLUE)
    box(ax, 11.0, 1.2, 3.0, 1.15, "estimate of " + VR + "\nat row $t$", "white", ORANGE)

    arrow(ax, (4.5, 4.42), (5.7, 4.42), BLUE)
    arrow(ax, (4.5, 2.45), (5.7, 3.8), ORANGE)
    arrow(ax, (4.5, 1.6), (5.7, 1.6), ORANGE)
    arrow(ax, (9.8, 4.42), (11.0, 4.42), BLUE)
    arrow(ax, (9.8, 1.77), (11.0, 1.77), ORANGE)

    # the path that does not exist
    arrow(ax, (4.5, 3.75), (5.7, 2.4), "#999999", ls=(0, (3, 3)), lw=1.0)
    ax.text(5.22, 2.94, r"$\times$", ha="center", va="center", fontsize=13, color="#b22222")
    ax.text(7.1, 5.72, "no path from the fast channels to the " + VR + " branch (dashed)",
            ha="center", va="center", fontsize=7.5, color=GREY)

    ax.text(7.1, 0.25, "each LSTM runs forward over the rows of a contiguous segment; "
            "357,570 parameters in total", ha="center", va="center", fontsize=7.5, color=GREY)
    fig.savefig(HERE / "fig_arch.pdf", bbox_inches="tight", pad_inches=0.03)
    fig.savefig(HERE / "fig_arch_preview.png", dpi=170, bbox_inches="tight", pad_inches=0.03)


if __name__ == "__main__":
    main()
