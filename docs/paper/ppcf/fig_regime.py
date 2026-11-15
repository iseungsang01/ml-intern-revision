"""Figure: contrast of regimes between KSTAR and MAST (real data and frozen artifacts only).

(a) regime number (lag-1 autocorrelation of 10 ms increments) per discharge: KSTAR histogram
    from kstar_regime.json; MAST median, quartiles and 95 % interval from mast_gate.json.
(b) MSE skill against persistence of CES-only estimators for T_i: KSTAR four file-level
    splits (cut population; b28_decomposition.json RMSEs) and MAST TEST (population G;
    mast_gate_addendum1.json, retuned arms) with 95 % intervals.
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
g2 = json.loads((PAPER / "mast_gate_addendum1.json").read_text(encoding="utf-8"))["retuned"]
dec = json.loads((PAPER / "b28_decomposition.json").read_text(encoding="utf-8"))["a_decomposition"]

fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.5))

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
               fmt="o", color=col, capsize=3, ms=5, label=lab + " (%d shots; median, 95 %% CI, quartiles)" % r["n_shots"])
a.axvline(-0.5, color="k", ls=":", lw=1)
a.text(-0.52, ymax * 0.97, "white noise\non flat signal", ha="right", va="top", fontsize=7)
a.axvline(0, color="grey", lw=0.6)
a.set_xlabel("lag-1 autocorrelation of 10 ms increments (per discharge)")
a.set_ylabel("KSTAR discharges")
a.set_title("(a) regime number", fontsize=10)
a.legend(fontsize=6.5, loc="upper right", bbox_to_anchor=(1.0, 0.62))

b = ax[1]
arms = [("running mean", "ewma", "running_mean"), ("causal GP", "gp_causal", "causal_gp"),
        ("offline GP", "gp", "offline_gp")]
cut = [d for d in dec if d["pop"] == "cut"]
for i, (lab, kk, mk) in enumerate(arms):
    ks = [1 - (d["rmse_eV"][kk] / d["rmse_eV"]["persistence"]) ** 2 for d in cut]
    b.plot([i - 0.17] * len(ks), ks, "o", color="#4c72b0", ms=5, alpha=0.85,
           label="KSTAR $T_i$, four splits (cut)" if i == 0 else None)
    s = g2["readouts"]["T_i|G"]["arms"][mk]["skill_vs_persistence"]
    b.errorbar(i + 0.17, s["point"], yerr=[[s["point"] - s["lo"]], [s["hi"] - s["point"]]],
               fmt="s", color="#1b3a6b", capsize=3, ms=5,
               label="MAST $T_i$, TEST (95 % CI)" if i == 0 else None)
    s = g2["readouts"]["V_i|G"]["arms"][mk]["skill_vs_persistence"]
    b.errorbar(i + 0.32, s["point"], yerr=[[s["point"] - s["lo"]], [s["hi"] - s["point"]]],
               fmt="^", color="#8c3b12", capsize=3, ms=5,
               label="MAST $V_i$, TEST (95 % CI)" if i == 0 else None)
b.axhline(0, color="k", lw=0.8)
b.set_xticks(range(len(arms)))
b.set_xticklabels([x[0] for x in arms])
b.set_ylabel("MSE skill against persistence")
b.set_title("(b) CES-only estimators", fontsize=10)
b.legend(fontsize=7, loc="upper left")

fig.tight_layout()
fig.savefig(HERE / "fig_regime.pdf")
fig.savefig(HERE / "fig_regime.png", dpi=200)
print("wrote fig_regime.pdf")
for lab, kk, mk in arms:
    print(lab, [round(1 - (d["rmse_eV"][kk] / d["rmse_eV"]["persistence"]) ** 2, 3) for d in cut])
