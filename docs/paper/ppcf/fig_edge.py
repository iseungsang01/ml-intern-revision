"""MAST confirmation set: measurement quality and regime number against normalized radius.

Source: docs/paper/mast_edge.json (3,686 discharges that took no part in forming the
hypothesis) and docs/paper/ppcf/kstar_regime.json (KSTAR reference lines).
Run from the repo root:  py docs/paper/ppcf/fig_edge.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
edge = json.loads((ROOT / "docs" / "paper" / "mast_edge.json").read_text(encoding="utf-8"))
kstar = json.loads((HERE / "kstar_regime.json").read_text(encoding="utf-8"))

BINS = ["0-0.3", "0.3-0.5", "0.5-0.7", "0.7-0.8", "0.8-0.9", "0.9-0.95", "0.95-1"]
X = np.array([0.15, 0.40, 0.60, 0.75, 0.85, 0.925, 0.975])
P = edge["primary"]


def series(getter):
    pts = [getter(b) for b in BINS]
    p = np.array([q["point"] for q in pts])
    lo = np.array([q["lo"] for q in pts])
    hi = np.array([q["hi"] for q in pts])
    return p, p - lo, hi - p


def kstar_value(target):
    for key in ("T_i", "Ti", "CES_TI") if target == "ti" else ("V_rot", "Vrot", "CES_VT"):
        if key in kstar:
            v = kstar[key]
            for name in ("median", "point", "regime_number"):
                if isinstance(v, dict) and name in v:
                    q = v[name]
                    return q["point"] if isinstance(q, dict) else q
    raise SystemExit("FATAL: KSTAR regime number not found in kstar_regime.json")


plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
fig, (a, b) = plt.subplots(1, 2, figsize=(7.4, 3.5))

for ax in (a, b):
    ax.axvspan(0.8, 0.9, color="0.90", lw=0)
    ax.axvspan(0.9, 1.0, color="0.82", lw=0)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel(r"normalized radius $\rho$ (MAST)")

p, lo, hi = series(lambda k: P["pre_rule"]["bins"][k]["fail_share_0p1"])
a.errorbar(X, p, yerr=[lo, hi], fmt="o-", color="#d62728", ms=5, lw=1.4, capsize=2,
           label="share of failed fits")
p, lo, hi = series(lambda k: P["pre_rule"]["bins"][k]["median_rel_Ti_error"])
a.errorbar(X, p, yerr=[lo, hi], fmt="s--", color="#1f77b4", ms=4.5, lw=1.2, capsize=2,
           label=r"median relative $T_i$ error")
a.set_ylabel("fraction")
a.set_ylim(0, 0.8)
a.legend(frameon=False, fontsize=8.5, loc="upper left")
a.set_title("(a) measurement quality", loc="left", fontsize=10)

p, lo, hi = series(lambda k: P["cells"]["T_i|W"]["bins"][k]["regime_number"])
b.errorbar(X, p, yerr=[lo, hi], fmt="o-", color="#d62728", ms=5, lw=1.4, capsize=2,
           label=r"$T_i$, loose fit rule")
p, lo, hi = series(lambda k: P["cells"]["T_i|G"]["bins"][k]["regime_number"])
b.errorbar(X, p, yerr=[lo, hi], fmt="o:", color="#d62728", ms=5, lw=1.1, capsize=2, mfc="white",
           label=r"$T_i$, strict fit rule")
p, lo, hi = series(lambda k: P["cells"]["V_i|W"]["bins"][k]["regime_number"])
b.errorbar(X, p, yerr=[lo, hi], fmt="s--", color="#1f77b4", ms=4.5, lw=1.2, capsize=2,
           label=r"$V_i$, loose fit rule")
b.axhline(0, color="k", lw=0.6)
b.axhline(kstar_value("ti"), color="#d62728", lw=1.0, ls=(0, (6, 3)))
b.text(0.02, kstar_value("ti") + 0.015, r"KSTAR chord, $T_i$", color="#d62728", fontsize=8.5)
b.axhline(-0.5, color="0.4", lw=0.8, ls=":")
b.text(0.02, -0.5 + 0.012, "white noise on a flat signal", color="0.3", fontsize=8)
b.set_ylabel("regime number")
b.set_ylim(-0.56, 0.5)
b.legend(frameon=False, fontsize=8.5, loc="center left", bbox_to_anchor=(0.0, 0.36))
b.set_title("(b) regime number", loc="left", fontsize=10)

fig.tight_layout()
fig.savefig(HERE / "fig_edge.pdf")
fig.savefig(HERE / "fig_edge.png", dpi=170)
print("wrote fig_edge.pdf / .png")
