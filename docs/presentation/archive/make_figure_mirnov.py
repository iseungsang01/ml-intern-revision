# -*- coding: utf-8 -*-
"""Generate the Mirnov-salvage figure for the KSTAR CES nowcasting talk.

Two panels telling one story:
  (a) MC is aliased white noise at the 10 ms grid  -- lag-1 autocorrelation of
      every channel group, measured over the real shot CSVs.
  (b) Deriving extra MC features does not help     -- 4-seed paired difference
      (burst arm minus baseline) in skill_vs_pchip.

Panel (a) is recomputed from the real data at build time (no hard-coded numbers).
Panel (b) reads the frozen 4-seed experiment result recorded in
PROJECT memory / this script's RESULTS table (see docstring of
make_mc_features.py in the experiment scratchpad).

Output: docs/presentation/figures/fig_mirnov.png
"""
import glob
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

for cand in ["Malgun Gothic", "Malgun Gothic Semilight", "NanumGothic", "AppleGothic"]:
    try:
        matplotlib.font_manager.findfont(cand, fallback_to_default=False)
        plt.rcParams["font.family"] = cand
        break
    except Exception:
        continue
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["savefig.dpi"] = 200
plt.rcParams["figure.dpi"] = 200

NAVY = "#13335f"
BLUE = "#2b6cb0"
TEAL = "#1b9e8a"
ORANGE = "#e8743b"
RED = "#c0392b"
GRAY = "#7b8794"
LGRAY = "#dfe4ea"
BG = "#ffffff"

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "figures")
DATA = os.environ.get("CES_DATA_DIR", os.path.join(HERE, "..", "..", "data"))
os.makedirs(OUT, exist_ok=True)

GAP = 0.5
N_FILES = 150

# 4-seed paired experiment: skill_vs_pchip, val split, window 4,
# 40k train / 8k val, 12 epochs, TEST reserved and never opened.
SEEDS = [42, 1, 7, 123]
TI_BASE = [+0.3946, +0.1811, +0.1213, +0.3290]
TI_BURST = [+0.3326, +0.0205, +0.0782, +0.3002]
VT_BASE = [-0.1505, +0.3609, +0.8615, +0.1205]
VT_BURST = [-0.0051, +0.3360, +0.8496, +0.1334]


def measure_lag1():
    """Median lag-1 autocorrelation per channel group over real shot CSVs."""
    groups = {"BES\n(n_e)": [], "ECEI\n(T_e)": [], "CES_TI": [], "CES_VT": [], "MC\n(dB/dt)": []}
    prefix = {"BES\n(n_e)": "BES_", "ECEI\n(T_e)": "ECEI_", "MC\n(dB/dt)": "MC"}
    files = sorted(glob.glob(os.path.join(DATA, "*.csv")))[:N_FILES]
    if not files:
        raise SystemExit(f"no CSVs under {DATA}; set CES_DATA_DIR")
    for f in files:
        df = pd.read_csv(f)
        t = df["time"].to_numpy(float)
        ok = np.diff(t) < GAP
        if ok.sum() < 50:
            continue
        for label, pre in prefix.items():
            for c in [c for c in df.columns if c.startswith(pre)]:
                v = df[c].to_numpy(float)
                a, b = v[:-1][ok], v[1:][ok]
                m = np.isfinite(a) & np.isfinite(b)
                if m.sum() > 50 and np.std(a[m]) > 0 and np.std(b[m]) > 0:
                    groups[label].append(np.corrcoef(a[m], b[m])[0, 1])
        for c in ["CES_TI", "CES_VT"]:
            v = df[c].to_numpy(float)
            a, b = v[:-1][ok], v[1:][ok]
            m = np.isfinite(a) & np.isfinite(b)
            if m.sum() > 50 and np.std(a[m]) > 0 and np.std(b[m]) > 0:
                groups[c].append(np.corrcoef(a[m], b[m])[0, 1])
    return {k: float(np.median(v)) for k, v in groups.items() if v}


def build():
    acf = measure_lag1()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.0, 5.0),
                                   gridspec_kw={"width_ratios": [1.0, 1.15]})

    # ---- panel (a): aliasing evidence ------------------------------------
    labels = list(acf.keys())
    vals = [acf[k] for k in labels]
    cols = [BLUE if v > 0.3 else RED for v in vals]
    bars = ax1.bar(range(len(labels)), vals, color=cols, width=0.62,
                   edgecolor="white", linewidth=1.2)
    ax1.axhline(0, color=GRAY, lw=1.0)
    ax1.axhspan(-0.081, 0.081, color=LGRAY, alpha=0.85, zorder=0)
    ax1.text(len(labels) - 0.45, 0.11, "백색잡음 신뢰대 ±0.08",
             ha="right", va="bottom", fontsize=10.5, color=GRAY)
    for i, (b, v) in enumerate(zip(bars, vals)):
        ax1.text(b.get_x() + b.get_width() / 2, v + (0.035 if v >= 0 else -0.05),
                 f"{v:+.2f}", ha="center",
                 va="bottom" if v >= 0 else "top",
                 fontsize=12, fontweight="bold", color=NAVY if v > 0.3 else RED)
    ax1.set_xticks(range(len(labels)))
    ax1.set_xticklabels(labels, fontsize=11.5)
    ax1.set_ylabel("연속 두 샘플의 상관 (lag-1)", fontsize=12)
    ax1.set_ylim(-0.15, 1.03)
    ax1.set_title("(a) Mirnov만 시간 구조가 없다 → 10 ms에서 aliasing",
                  fontsize=13.5, color=NAVY, fontweight="bold", pad=12)
    ax1.spines[["top", "right"]].set_visible(False)
    ax1.grid(axis="y", alpha=0.25, lw=0.7)

    # ---- panel (b): 4-seed paired experiment -----------------------------
    ti_d = np.array(TI_BURST) - np.array(TI_BASE)
    vt_d = np.array(VT_BURST) - np.array(VT_BASE)
    x = np.arange(len(SEEDS))
    w = 0.36
    ax2.bar(x - w / 2, ti_d, width=w, color=ORANGE, label="CES_TI",
            edgecolor="white", linewidth=1.1)
    ax2.bar(x + w / 2, vt_d, width=w, color=BLUE, label="CES_VT",
            edgecolor="white", linewidth=1.1)
    ax2.axhline(0, color=NAVY, lw=1.4)
    ax2.axhline(ti_d.mean(), color=ORANGE, lw=1.5, ls="--", alpha=0.8)
    ax2.text(len(SEEDS) - 0.42, ti_d.mean() - 0.012,
             f"CES_TI 평균 {ti_d.mean():+.3f} (4/4 음수)",
             ha="right", va="top", fontsize=10.5, color=ORANGE, fontweight="bold")
    for xi, (a, b) in enumerate(zip(ti_d, vt_d)):
        ax2.text(xi - w / 2, a + (0.006 if a >= 0 else -0.008), f"{a:+.3f}",
                 ha="center", va="bottom" if a >= 0 else "top", fontsize=9.5, color=NAVY)
        ax2.text(xi + w / 2, b + (0.006 if b >= 0 else -0.008), f"{b:+.3f}",
                 ha="center", va="bottom" if b >= 0 else "top", fontsize=9.5, color=NAVY)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"seed {s}" for s in SEEDS], fontsize=11.5)
    ax2.set_ylabel("skill_vs_pchip 차이  (파생 MC 추가 - baseline)", fontsize=12)
    ax2.set_title("(b) MC 파생 피처를 넣어도 개선 없음 (4시드 짝지은 비교)",
                  fontsize=13.5, color=NAVY, fontweight="bold", pad=12)
    ax2.legend(fontsize=11, frameon=False, loc="upper left")
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.grid(axis="y", alpha=0.25, lw=0.7)

    fig.tight_layout()
    path = os.path.join(OUT, "fig_mirnov.png")
    fig.savefig(path, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print("wrote", path)
    print("lag-1 medians:", {k.replace(chr(10), " "): round(v, 3) for k, v in acf.items()})


if __name__ == "__main__":
    build()
