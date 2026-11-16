"""Figure for the thinned replay (B.30): skill over hold against the interval between visible CES
samples, and the fast-channel share per split.  Reads docs/paper/b30_thinned.json and
docs/paper/b30_posthoc.json; writes fig_thin.pdf / .png.  Run from the repo root:
    py docs/paper/ppcf/fig_thin.py
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).parent
ROOT = HERE.parents[2]
SEEDS = (42, 1, 7, 123)
KS = (1, 2, 3, 5, 10)
X = np.array([10 * k for k in KS])
# Okabe-Ito colours, distinguishable without relying on hue alone (markers and line styles differ)
LINES = (("pchip_vs_hold", "PCHIP of the visible samples (not causal)", "#7f7f7f", "s", "--"),
         ("gp_vs_hold", "causal GP", "#0072B2", "D", "-"),
         ("hist80_vs_hold", "network without fast channels", "#E69F00", "v", "-"),
         ("causal_vs_hold", "causal variant", "#009E73", "^", "-"),
         ("bb_vs_hold", "network", "#D55E00", "o", "-"))


def main():
    thin = json.loads((ROOT / "docs/paper/b30_thinned.json").read_text(encoding="utf-8"))
    post = json.loads((ROOT / "docs/paper/b30_posthoc.json").read_text(encoding="utf-8"))
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                         "legend.frameon": False, "pdf.fonttype": 42})
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 2.9), gridspec_kw={"width_ratios": [1.15, 1.0]})

    for key, lab, col, mk, ls in LINES:
        v = np.array([[thin["cells"][f"cut|s{s}|CES_TI"]["by_k"][str(k)]["skill"][key][0] for k in KS]
                      for s in SEEDS])
        m = v.mean(0)
        a.errorbar(X, m, yerr=[m - v.min(0), v.max(0) - m], color=col, marker=mk, ms=4, lw=1.2,
                   ls=ls, capsize=2, label=lab)
    a.set_xscale("log")
    a.set_xticks(X)
    a.set_xticklabels([str(x) for x in X])
    a.minorticks_off()
    a.set_xlabel("interval between visible CES samples (ms)")
    a.set_ylabel(r"$T_i$ skill over hold")
    a.set_ylim(0.12, 0.52)
    a.legend(fontsize=6.5, loc="upper right")
    a.set_title("(a)", loc="left", fontsize=8)

    off = np.linspace(-0.06, 0.06, len(SEEDS))
    for j, s in enumerate(SEEDS):
        for src, key, col, mk, lab in (
                (thin["cells"][f"cut|s{s}|CES_TI"]["by_k"], "bb80_vs_hist80", "#D55E00", "o", "network"),
                (post["causal_share"][f"cut|s{s}"], "causal_vs_hist", "#009E73", "^", "causal variant")):
            v = np.array([(src[str(k)]["skill"][key] if "skill" in src[str(k)] else src[str(k)][key])
                          for k in KS])
            xs = X * 10 ** (off[j] + (0.012 if mk == "^" else -0.012))
            b.errorbar(xs, v[:, 0], yerr=[v[:, 0] - v[:, 1], v[:, 2] - v[:, 0]], color=col, marker=mk,
                       ms=3.2, lw=0, elinewidth=0.8, capsize=1.5, label=lab if j == 0 else None)
    b.axhline(0, color="k", lw=0.6)
    b.set_xscale("log")
    b.set_xticks(X)
    b.set_xticklabels([str(x) for x in X])
    b.minorticks_off()
    b.set_xlabel("interval between visible CES samples (ms)")
    b.set_ylabel(r"fast-channel share of the $T_i$ skill")
    b.legend(fontsize=6.5, loc="upper left")
    b.set_title("(b)", loc="left", fontsize=8)
    fig.tight_layout()
    fig.savefig(HERE / "fig_thin.pdf")
    fig.savefig(HERE / "fig_thin.png", dpi=200)
    print("written fig_thin.pdf / .png")


if __name__ == "__main__":
    main()
