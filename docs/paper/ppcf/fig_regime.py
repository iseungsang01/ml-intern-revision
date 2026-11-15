"""Figure: the regime number on KSTAR and, on MAST, as a function of normalized radius.

(a) regime number (lag-1 autocorrelation of 10 ms increments) per discharge: KSTAR histogram
    from kstar_regime.json; MAST median, quartiles and 95 % interval over all scored chords
    from mast_gate.json.
(b) MAST T_i and V_i regime number by bin of normalized radius (mast_gate_addendum2.json,
    stratum f; post hoc breakdown), with the KSTAR T_i value and the bin that contains the
    nominal radius of the KSTAR chord.
(c) MAST share of samples failing the fit rule and median relative T_i error by the same
    bins, before the fit rule is applied (same source).
Run from the repo root:  py docs/paper/ppcf/fig_regime.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent
PAPER = HERE.parent
kr = json.loads((HERE / "kstar_regime.json").read_text(encoding="utf-8"))
g1 = json.loads((PAPER / "mast_gate.json").read_text(encoding="utf-8"))
sf = json.loads((PAPER / "mast_gate_addendum2.json").read_text(encoding="utf-8"))["stratum_f"]
BINS = sf["bins"]
X = np.arange(len(BINS))

fig, ax = plt.subplots(1, 3, figsize=(11.5, 3.6), gridspec_kw={"width_ratios": [1.15, 1, 1]})

a = ax[0]
bins = np.linspace(-1, 1, 41)
a.hist(kr["T_i"]["values"], bins=bins, color="#4c72b0", alpha=0.75,
       label="KSTAR $T_i$ (%d discharges)" % kr["T_i"]["n_discharges"])
a.hist(kr["V_rot"]["values"], bins=bins, color="#dd8452", alpha=0.6,
       label=r"KSTAR $V_\mathrm{rot}$ (%d discharges)" % kr["V_rot"]["n_discharges"])
ymax = a.get_ylim()[1]
for j, (key, col, lab) in enumerate([("T_i", "#1b3a6b", "MAST $T_i$"), ("V_i", "#8c3b12", "MAST $V_i$")]):
    r = g1["regime"][key]
    q = r["regime_number_quartiles"]
    m = r["regime_number_median_over_TEST_shots"]
    y = ymax * (0.80 - 0.13 * j)
    a.plot([q[0], q[2]], [y, y], color=col, lw=5, alpha=0.35, solid_capstyle="butt")
    a.errorbar(m["point"], y, xerr=[[m["point"] - m["lo"]], [m["hi"] - m["point"]]],
               fmt="o", color=col, capsize=3, ms=5, label=lab + ", all radii (%d shots)" % r["n_shots"])
a.axvline(-0.5, color="k", ls=":", lw=1)
a.text(-0.52, ymax * 0.97, "white noise\non flat signal", ha="right", va="top", fontsize=7)
a.axvline(0, color="grey", lw=0.6)
a.set_xlabel("lag-1 autocorrelation of 10 ms increments")
a.set_ylabel("KSTAR discharges")
a.set_title("(a) regime number per discharge", fontsize=10)
a.legend(fontsize=6.5, loc="upper right", bbox_to_anchor=(1.0, 0.62))


def series(key, off, col, mk, lab):
    first = True
    for i, bn in enumerate(BINS):
        r = sf["readouts"][key]["bins"][bn]["regime_number"]
        if r["lo"] is None:
            b.plot(i + off, r["point"], mk, color=col, ms=5, mfc="none")
            continue
        b.errorbar(i + off, r["point"], yerr=[[r["point"] - r["lo"]], [r["hi"] - r["point"]]],
                   fmt=mk, color=col, capsize=3, ms=5, label=lab if first else None)
        first = False


b = ax[1]
b.axvspan(2.5, 3.5, color="#cccccc", alpha=0.45, lw=0)
series("T_i|G", -0.2, "#1b3a6b", "o", "MAST $T_i$, relative error < 0.1")
series("T_i|W", 0.0, "#5b86c9", "s", "MAST $T_i$, relative error < 0.5")
series("V_i|G", 0.2, "#8c3b12", "^", "MAST $V_i$, relative error < 0.1")
kt = kr["T_i"]
b.axhline(kt["median"], color="#4c72b0", ls="--", lw=1.2)
b.text(-0.4, kt["median"] + 0.02, "KSTAR $T_i$ (one chord)", color="#4c72b0", fontsize=7, va="bottom")
b.axhline(0, color="k", lw=0.8)
b.axhline(-0.5, color="k", ls=":", lw=1)
b.set_xticks(X)
b.set_xticklabels(BINS, fontsize=8)
b.set_xlim(-0.5, len(BINS) - 0.5)
b.set_ylim(-0.6, 0.95)
b.set_xlabel(r"normalized radius $\rho$ (MAST, low-field side)")
b.set_ylabel("regime number")
b.set_title("(b) MAST regime number by radius", fontsize=10)
b.legend(fontsize=6.5, loc="upper left")

c = ax[2]
c.axvspan(2.5, 3.5, color="#cccccc", alpha=0.45, lw=0)
for off, fld, col, mk, lab in [(-0.1, "share_fail01", "#2a7f3f", "o", "fit-rule failures, relative error >= 0.1"),
                               (0.1, "share_fail05", "#86c08f", "s", "fit-rule failures, relative error >= 0.5")]:
    pts = [sf["pre_rule"]["bins"][bn][fld] for bn in BINS]
    c.errorbar(X + off, [p["point"] for p in pts],
               yerr=[[p["point"] - p["lo"] for p in pts], [p["hi"] - p["point"] for p in pts]],
               fmt=mk + "-", color=col, capsize=3, ms=5, lw=1, label=lab)
pts = [sf["pre_rule"]["bins"][bn]["median_rel_Ti_error"] for bn in BINS]
c.errorbar(X, [p["point"] for p in pts],
           yerr=[[p["point"] - p["lo"] for p in pts], [p["hi"] - p["point"] for p in pts]],
           fmt="^--", color="#7a4fa3", capsize=3, ms=5, lw=1, label="median relative $T_i$ error")
c.set_xticks(X)
c.set_xticklabels(BINS, fontsize=8)
c.set_xlim(-0.5, len(BINS) - 0.5)
c.set_ylim(0, 0.8)
c.set_xlabel(r"normalized radius $\rho$ (MAST, low-field side)")
c.set_ylabel("share of samples / relative error")
c.set_title("(c) MAST measurement quality by radius", fontsize=10)
c.legend(fontsize=6.5, loc="upper left")

fig.tight_layout()
fig.savefig(HERE / "fig_regime.pdf")
fig.savefig(HERE / "fig_regime.png", dpi=170)
print("wrote fig_regime.pdf")
