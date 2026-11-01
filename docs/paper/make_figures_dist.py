# -*- coding: utf-8 -*-
"""Row-level paper figures for the predictive-distribution and per-shot results.

Unlike make_figures_en.py (which reads paper_numbers.json), these panels need the frozen
per-row TEST dumps, so they read them directly. No model is trained or re-scored here.

  fig_pit.png          PIT histograms on the frozen TEST rows, 4 splits pooled. The Gaussian
                       head (B.17 `v2g`) is compared with the arm the B.17b registration
                       named: cut = Student-t (`v2t`), inclusive = two-component mixture
                       (`v2m2`) (THESIS_RESULTS sec. 8bf, 8bj).
  fig_interval.png     One held-out TEST discharge (#31815, split 42, cut population): CES T_i
                       and V_rot with the named cut arm's posterior mean and central 50 % / 90 %
                       predictive intervals.
  fig_event_quiet.png  The same discharge against the quietest TEST discharge of split 42 (rule
                       below): CES T_i with the seq_v2 backbone, then the absolute error of the
                       backbone, PCHIP and the causal GP on the same rows.

Quiet-shot rule: among split-42 cut-population TEST shots with >= 200 scored T_i rows, the one
with the smallest within-discharge spread (s.d.) of the measured T_i -- the covariate sec. 8an
found to decide where the model wins (rho +0.40 with the per-shot skill vs the causal GP).

Usage (repo root):  py docs/paper/make_figures_dist.py
Output: docs/paper/figures/fig_pit.png, fig_interval.png, fig_event_quiet.png
"""
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t as student_t

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data"
sys.path.insert(0, str(REPO / "ces_prediction" / "experiments" / "b17_dist"))
sys.path.insert(0, str(REPO / "ces_prediction"))
from score_b17 import cdf_mixture, moments  # noqa: E402
from score_b17b import arm_scores  # noqa: E402

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["savefig.dpi"] = 200
plt.rcParams["figure.dpi"] = 200

NAVY, BLUE, ORANGE, GREEN, RED, GRAY = "#13335f", "#2b6cb0", "#e8743b", "#2e9e5b", "#c0392b", "#7b8794"
OUT = Path(__file__).resolve().parent / "figures"
SEEDS = (42, 1, 7, 123)
T = ("CES_TI", "CES_VT")
TLAB = {"CES_TI": r"$T_i$", "CES_VT": r"$V_{\mathrm{rot}}$"}
UNIT = {"CES_TI": "eV", "CES_VT": "km/s"}
NAMED = {"cut": (".b17b", "v2t", "Student-$t$ head (named, cut)"),
         "incl": (".b17b", "v2m2", "mixture head (named, inclusive)")}
DEMO_SHOT, DEMO_SEED = "s31815.csv", 42


def save(fig, name):
    fig.savefig(OUT / name, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", name)


def clean(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def dump(prefix, arm, pop, s):
    return np.load(DATA / f"{prefix}_{arm}_{pop}_s{s}" / "dist_test.npz")


def names():
    return [p.name for p in sorted(DATA.glob("s*.csv"))]


def quantile(z, t, q):
    """Per-row predictive quantile of a t or mixture dump."""
    pi, mu, sg = z[f"{t}_pi"], z[f"{t}_mu"], z[f"{t}_sigma"]
    if f"{t}_nu" in z.files:
        return mu[:, 0] + sg[:, 0] * student_t.ppf(q, z[f"{t}_nu"][:, 0])
    lo = (mu - 12 * sg).min(axis=1)
    hi = (mu + 12 * sg).max(axis=1)
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        below = cdf_mixture(mid, pi, mu, sg) < q
        lo, hi = np.where(below, mid, lo), np.where(below, hi, mid)
    return 0.5 * (lo + hi)


def fig_pit():
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 6.6), sharex=True)
    bins = np.linspace(0, 1, 21)
    for j, pop in enumerate(("cut", "incl")):
        prefix, arm, label = NAMED[pop]
        for i, t in enumerate(T):
            ax = axes[i, j]
            pg = np.concatenate([arm_scores(dump(".b17", "v2g", pop, s), t)["pit"] for s in SEEDS])
            pn = np.concatenate([arm_scores(dump(prefix, arm, pop, s), t)["pit"] for s in SEEDS])
            ax.hist(pn, bins=bins, density=True, color=NAVY if pop == "cut" else ORANGE, alpha=0.75, label=label)
            ax.hist(pg, bins=bins, density=True, histtype="step", color=GRAY, lw=1.6, label="Gaussian head (B.17)")
            ax.axhline(1.0, color="black", lw=0.8, ls="--")
            cov = lambda p: (np.mean((p >= .25) & (p <= .75)), np.mean((p >= .05) & (p <= .95)))
            cn, cg = cov(pn), cov(pg)
            ax.text(0.5, 1.0, f"50 % / 90 % coverage: named {cn[0]:.2f} / {cn[1]:.2f},  Gaussian {cg[0]:.2f} / {cg[1]:.2f}",
                    transform=ax.transAxes, ha="center", va="top", fontsize=7.6, color="black")
            ax.set_title(f"{TLAB[t]}, {'cut' if pop == 'cut' else 'inclusive'} population (4 splits, n = {len(pn):,})", fontsize=9.5)
            ax.set_ylim(0, 2.6 if t == "CES_TI" else 4.2)
            clean(ax)
            if i == 1:
                ax.set_xlabel("PIT = predictive CDF at the observed value")
            if j == 0:
                ax.set_ylabel("density")
            if i == 0:
                ax.legend(loc="lower center", fontsize=7.5, frameon=False)
    fig.suptitle("Calibration on the frozen TEST rows: a flat PIT histogram means calibrated", fontsize=11)
    fig.tight_layout()
    save(fig, "fig_pit.png")


def shot_rows(z, t, shot_idx):
    m = z[f"{t}_shot"].astype(int) == shot_idx
    order = np.argsort(z[f"{t}_row"][m])
    return np.flatnonzero(m)[order]


def csv_time(name):
    import csv
    with open(DATA / name, encoding="utf-8") as fh:
        rd = csv.reader(fh)
        head = next(rd)
        k = head.index("time")
        return np.array([float(r[k]) for r in rd])


def fig_interval():
    z = dump(".b17b", "v2t", "cut", DEMO_SEED)
    idx = names().index(DEMO_SHOT)
    tm = csv_time(DEMO_SHOT)
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 6.0), sharex=True)
    for ax, t in zip(axes, T):
        r = shot_rows(z, t, idx)
        x = tm[z[f"{t}_row"][r]]
        y = z[f"{t}_y"][r]
        q = {p: quantile(z, t, p)[r] for p in (0.05, 0.25, 0.75, 0.95)}
        mean = z[f"{t}_mu"][r, 0]
        ax.fill_between(x, q[0.05], q[0.95], color=BLUE, alpha=0.18, lw=0, label="90 % interval")
        ax.fill_between(x, q[0.25], q[0.75], color=BLUE, alpha=0.38, lw=0, label="50 % interval")
        ax.plot(x, mean, color=NAVY, lw=1.0, label="posterior mean")
        ax.plot(x, y, "o", ms=2.2, color=ORANGE, label="CES measurement (held-out)")
        inside = np.mean((y >= q[0.05]) & (y <= q[0.95]))
        ax.set_ylabel(f"{TLAB[t]} [{UNIT[t]}]")
        if t == "CES_TI":
            ax.set_ylim(0, None)
        ax.set_title(f"{TLAB[t]}: {len(r)} scored rows, {100 * inside:.0f} % inside the 90 % interval", fontsize=9.5)
        clean(ax)
    axes[0].legend(loc="upper right", fontsize=7.5, frameon=False, ncol=2)
    axes[1].set_xlabel("time [s]")
    fig.suptitle(f"KSTAR #{DEMO_SHOT[1:6]} (held-out TEST, split {DEMO_SEED}, cut population): Student-$t$ head", fontsize=11)
    fig.tight_layout()
    save(fig, "fig_interval.png")


def spread(y):
    return float(np.std(y))


def fig_event_quiet():
    z = dump(".b17b", "v2t", "cut", DEMO_SEED)
    c = np.load(DATA / f".b17b_v2t_cut_s{DEMO_SEED}" / "comparison_errors_test.npz")
    t = "CES_TI"
    if not ((c[f"{t}_shot"] == z[f"{t}_shot"]).all() and (c[f"{t}_row"] == z[f"{t}_row"]).all()):
        raise SystemExit("FATAL: distribution dump and error npz rows differ")
    nm = names()
    shots = np.unique(z[f"{t}_shot"].astype(int))
    share = {}
    for s in shots:
        r = shot_rows(z, t, s)
        if len(r) >= 200:
            share[s] = spread(z[f"{t}_y"][r])
    quiet = min(share, key=share.get)
    order = sorted(share, key=share.get, reverse=True)
    rank = {k: i + 1 for i, k in enumerate(order)}
    event = nm.index(DEMO_SHOT)
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 6.2), sharex="col", gridspec_kw={"height_ratios": [1.3, 1]})
    for j, s in enumerate((event, quiet)):
        r = shot_rows(z, t, s)
        tm = csv_time(nm[s])
        x = tm[z[f"{t}_row"][r]]
        y = z[f"{t}_y"][r]
        ax = axes[0, j]
        ax.plot(x, y, "o", ms=2.0, color=ORANGE, label="CES $T_i$ (held-out)")
        ax.plot(x, z[f"{t}_backbone_pred"][r], color=NAVY, lw=1.0, label="seq_v2 nowcast")
        kind = "case-study discharge" if j == 0 else "quietest discharge"
        ax.set_title(f"#{nm[s][1:6]} ({kind}): $T_i$ spread {spread(y):.0f} eV, rank {rank[s]} of {len(share)}",
                     fontsize=9.5)
        ax.set_ylabel(r"$T_i$ [eV]")
        clean(ax)
        ax2 = axes[1, j]
        k = 5
        sm = lambda v: np.convolve(v, np.ones(k) / k, mode="same")
        err = {"seq_v2": np.sqrt(z[f"{t}_se_control"][r]), "PCHIP (acausal)": np.sqrt(c[f"{t}_se_pchip"][r]),
               "causal GP": np.sqrt(c[f"{t}_se_gp_causal"][r])}
        for (lab, e), col in zip(err.items(), (NAVY, GRAY, GREEN)):
            ax2.plot(x, sm(e), color=col, lw=1.1, label=lab)
        mse = {lab: float(np.mean(e ** 2)) for lab, e in err.items()}
        ax2.text(0.99, 0.95, "MSE skill vs PCHIP {:+.2f}, vs causal GP {:+.2f}".format(
            1 - mse["seq_v2"] / mse["PCHIP (acausal)"], 1 - mse["seq_v2"] / mse["causal GP"]),
            transform=ax2.transAxes, ha="right", va="top", fontsize=7.8)
        ax2.set_ylabel("|error| [eV], 5-row mean")
        ax2.set_xlabel("time [s]")
        clean(ax2)
    axes[0, 0].legend(loc="upper left", fontsize=7.5, frameon=False)
    axes[1, 0].legend(loc="upper left", fontsize=7.5, frameon=False)
    fig.suptitle("Where the $T_i$ skill comes from: a discharge whose $T_i$ moves and the quietest one (split 42 TEST, cut)",
                 fontsize=11)
    fig.tight_layout()
    save(fig, "fig_event_quiet.png")
    print(f"quiet shot = {nm[quiet]} spread {share[quiet]:.1f} eV; event {DEMO_SHOT} spread {share[event]:.1f} eV rank {rank[event]}/{len(share)}")


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    fig_pit()
    fig_interval()
    fig_event_quiet()
