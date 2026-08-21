"""The context-vs-family ladder as one figure — the graph the window question becomes.

Reads ONLY the frozen pooled-ladder artifact (data/.b9_pooled_ladder.json, sec. 8am):
per family (recurrent / dilated conv / attention / SSM), pooled `CES_TI` skill against
the causal GP over 301 physical discharges with 95% cluster-bootstrap CIs, plus the
fraction of discharges won. Two panels, one x-axis (contiguous causal context, ms):

  (a) pooled skill — positive at EVERY context, saturating at ~50 ms
  (b) win rate — what context actually buys: the win becoming typical (0.52 -> 0.66)

Colors are the Okabe-Ito CVD-safe set (validated); identity is triple-encoded
(color + marker + direct label). Output: docs/paper/figures/fig_context_family_ladder.png

Usage (repo root):  py ces_prediction/experiments/b9_reach/fig_pooled_ladder.py
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[3]
SRC = REPO / "data" / ".b9_pooled_ladder.json"
OUT = REPO / "docs" / "paper" / "figures" / "fig_context_family_ladder.png"

FAMILIES = (  # fixed order, fixed colors (Okabe-Ito), fixed markers
    ("lstm", "Recurrent (LSTM)", "#0072B2", "o"),
    ("tcn", "Dilated conv (TCN)", "#D55E00", "s"),
    ("xfmr", "Attention", "#009E73", "^"),
    ("ssm", "State-space (SSM)", "#CC79A7", "D"),
)


def series(node):
    rungs = sorted(node["rungs"].values(), key=lambda r: r["ms"])
    ms = [r["ms"] for r in rungs]
    skill = [r["skill"] for r in rungs]
    lo = [r["ci95"][0] for r in rungs]
    hi = [r["ci95"][1] for r in rungs]
    win = [r["win_rate"] for r in rungs]
    return ms, skill, lo, hi, win


def main():
    d = json.loads(SRC.read_text(encoding="utf-8"))
    fams = d["families"]

    fig, (ax_s, ax_w) = plt.subplots(
        2, 1, figsize=(7.2, 6.4), sharex=True,
        gridspec_kw={"height_ratios": [1.5, 1.0], "hspace": 0.12})
    for ax in (ax_s, ax_w):
        ax.set_xscale("log")
        ax.grid(True, which="both", color="#e6e6e2", linewidth=0.7)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    for key, label, color, marker in FAMILIES:
        if key not in fams:
            continue
        ms, skill, lo, hi, win = series(fams[key]["CES_TI"])
        ax_s.fill_between(ms, lo, hi, color=color, alpha=0.13, linewidth=0)
        ax_s.plot(ms, skill, color=color, marker=marker, markersize=5,
                  linewidth=2.0, label=label)
        ax_s.annotate(label, xy=(ms[-1], skill[-1]), xytext=(6, 0),
                      textcoords="offset points", va="center",
                      fontsize=8.5, color="#333333")
        ax_w.plot(ms, win, color=color, marker=marker, markersize=4.5, linewidth=2.0)

    ax_s.axhline(0.0, color="#999999", linewidth=1.0)
    ax_w.axhline(0.5, color="#999999", linewidth=1.0, linestyle=":")
    ax_w.annotate("coin flip", xy=(650, 0.5), xytext=(0, 4),
                  textcoords="offset points", ha="right", fontsize=8, color="#777777")
    for ax in (ax_s, ax_w):
        ax.axvline(50, color="#555555", linewidth=1.0, linestyle="--")
    ax_s.annotate("saturation (~50 ms)", xy=(50, ax_s.get_ylim()[1]),
                  xytext=(5, -2), textcoords="offset points",
                  fontsize=8.5, color="#555555", va="top")

    ax_s.set_ylabel("CES $T_i$ skill vs causal GP\n(pooled, 301 discharges, 95% CI)")
    ax_w.set_ylabel("Fraction of\ndischarges won")
    ax_w.set_xlabel("Contiguous causal context (ms)")
    ax_s.legend(loc="lower right", fontsize=8.5, frameon=False)
    ax_s.set_title("Context, not architecture, decides the T$_i$ nowcast "
                   "— and what it buys is typicality", fontsize=10.5, pad=10)
    ax_w.set_xticks([20, 30, 50, 70, 150, 310, 630])
    ax_w.set_xticklabels(["20", "30", "50", "70", "150", "310", "630"])
    ax_s.set_xlim(17, 1080)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=220, bbox_inches="tight")
    print(f"[fig] wrote {OUT}")


if __name__ == "__main__":
    main()
