"""B.41 figure: T_i, ECEI, BES and the two networks against the time since a beam blip and since an
edge-burst onset, each as a ratio to its level before the event.

Reads docs/paper/b41_events_fig.json (written by run_b41.py).  Run from the repo root:
    py ces_prediction/experiments/b41_events/fig_events.py
Output: docs/paper/b41_events_fig.pdf / .png
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
FIG = REPO / "docs" / "paper" / "b41_events_fig"
# Okabe-Ito; the measurement is the black reference line
STYLE = {
    "T_i": ("measured $T_i$", "#000000", "-", 1.4),
    "ECEI": ("ECEI level", "#0072B2", "--", 1.0),
    "BES": ("BES level", "#E69F00", "--", 1.0),
    "causal": ("causal variant", "#D55E00", "-", 1.1),
    "twin": ("history-only twin", "#009E73", "-", 1.1),
}


def band(ax, x, d, key, label=None, median=False):
    lab, col, ls, lw = STYLE[key]
    if median:
        d = d["after_the_fact_median"]
    m, lo, hi = (np.array(d[k], float) for k in ("median" if median else "mean", "lo", "hi"))
    ax.fill_between(x, lo, hi, color=col, alpha=0.13, lw=0)
    ax.plot(x, m, ls=ls, color=col, lw=lw, label=label or lab)


def main():
    d = json.loads((FIG.with_suffix(".json")).read_text(encoding="utf-8"))
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                         "font.family": "DejaVu Sans"})
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True, constrained_layout=True)

    b = d["blip"]
    x = np.array(b["rows_k"], float)
    ax = axes[0]
    for key, src in (("ECEI", "ECEI"), ("BES", "BES"), ("twin", "hist80"), ("causal", "causal"), ("T_i", "y")):
        band(ax, x, b[src], key)
    ax.axvspan(0, 1, color="0.88", lw=0)
    ax.set_xticks(np.arange(-5, 16, 5))
    ax.set_xticks(np.arange(-5, 16), minor=True)
    ax.set_xlabel("rows after the last row before the blip (10 ms each)")
    ax.set_ylabel("ratio to the level before the event")
    ax.set_title("(a) beam blip (means), test discharges\nof four splits; shaded: absent frame(s)",
                 fontsize=8, loc="left")

    u = d["burst"]
    xb = np.array(u["bins_ms"]) + 5.0
    ax = axes[1]
    for key in ("ECEI", "BES", "T_i"):
        band(ax, xb, u[key], key, median=True)
    band(ax, xb, u["twin"], "twin", median=True)
    band(ax, xb, u["causal"], "causal", median=True)
    m = np.array(u["measured_same_rows"]["after_the_fact_median"]["median"], float)
    ax.plot(xb, m, ls=":", color="#000000", lw=1.0, label="measured $T_i$, rows of the networks")
    ax.axvline(0, color="0.6", lw=0.6)
    ax.set_xlabel("row stamp minus burst onset (ms; 10 ms bins)")
    ax.set_title(f"(b) edge-burst onset (medians), {u['events']} events;\n"
                 f"networks on test discharges, {u['test_events']} events", fontsize=8, loc="left")
    for ax in axes:
        ax.axhline(1, color="0.6", lw=0.6)
        ax.grid(axis="y", color="0.9", lw=0.5)
    h, lab = axes[1].get_legend_handles_labels()
    fig.legend(h, lab, loc="outside lower center", ncol=3, frameon=False, fontsize=7.5)
    fig.savefig(FIG.with_suffix(".pdf"))
    fig.savefig(FIG.with_suffix(".png"), dpi=200)
    print("written", FIG.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
