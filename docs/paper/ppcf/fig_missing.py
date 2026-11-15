"""Figure 1: CES missingness on the KSTAR 10 ms grid, computed from the 641 discharge files.

(a) share of grid rows per target: measured (independent), held (bit-identical repeat of
the previous observation), missing.  (b) distribution of gap lengths: runs of consecutive
rows without a value, inside contiguous segments (breaks at time steps >= 0.5 s).
Run from the repo root:  py docs/paper/ppcf/fig_missing.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
GAP = 0.5
TARGETS = (("CES_TI", r"$T_i$"), ("CES_VT", r"$V_\mathrm{rot}$"))


def runs(mask, seg):
    """Lengths of runs of True inside each segment."""
    out = []
    n = 0
    for i in range(len(mask)):
        if i > 0 and seg[i] != seg[i - 1]:
            if n:
                out.append(n)
            n = 0
        if mask[i]:
            n += 1
        elif n:
            out.append(n)
            n = 0
    if n:
        out.append(n)
    return out


def main():
    files = sorted((ROOT / "data").glob("*.csv"))
    if len(files) != 641:
        raise SystemExit(f"FATAL: expected 641 CSV files, found {len(files)}")
    tot = 0
    cnt = {c: {"missing": 0, "held": 0, "observed": 0} for c, _ in TARGETS}
    gaps = {c: [] for c, _ in TARGETS}
    for f in files:
        df = pd.read_csv(f, usecols=["time", "CES_TI", "CES_VT"])
        t = df["time"].to_numpy(float)
        seg = np.concatenate(([0], np.cumsum(~(np.diff(t) < GAP)))) if len(t) > 1 else np.zeros(len(t), int)
        tot += len(t)
        for c, _ in TARGETS:
            v = df[c].to_numpy(float)
            miss = np.isnan(v)
            obs = np.flatnonzero(~miss)
            held = np.zeros(len(v), bool)
            if len(obs) > 1:
                same = (v[obs[1:]] == v[obs[:-1]]) & (seg[obs[1:]] == seg[obs[:-1]])
                held[obs[1:][same]] = True
            cnt[c]["missing"] += int(miss.sum())
            cnt[c]["held"] += int(held.sum())
            cnt[c]["observed"] += int((~miss).sum())
            gaps[c] += runs(miss, seg)
    res = {"n_files": len(files), "n_rows": tot}
    for c, _ in TARGETS:
        m, h, o = cnt[c]["missing"], cnt[c]["held"], cnt[c]["observed"]
        g = np.array(gaps[c])
        res[c] = {"missing_pct": 100 * m / tot, "held_pct_of_grid": 100 * h / tot,
                  "held_pct_of_observed": 100 * h / o, "independent_pct": 100 * (o - h) / tot,
                  "no_independent_pct": 100 * (m + h) / tot, "n_gaps": int(len(g)),
                  "gap_median_ms": float(np.median(g) * 10), "gap_single_row_share": float((g == 1).mean()),
                  "gap_le_50ms_share": float((g <= 5).mean()), "gap_max_ms": float(g.max() * 10)}
    (HERE / "fig_missing.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res, indent=1))

    plt.rcParams.update({"font.family": "serif", "font.size": 9, "mathtext.fontset": "stix",
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 2.6), gridspec_kw={"width_ratios": [1.0, 1.25]})
    col = {"independent": "#4c72b0", "held": "#9ecae1", "missing": "#c44e52"}
    for i, (c, lab) in enumerate(TARGETS):
        r = res[c]
        left = 0.0
        for key, val in (("independent", r["independent_pct"]), ("held", r["held_pct_of_grid"]),
                         ("missing", r["missing_pct"])):
            a.barh(i, val, left=left, color=col[key], height=0.55, edgecolor="white", linewidth=0.6)
            if val >= 12:
                a.text(left + val / 2, i, f"{val:.1f}%", ha="center", va="center", fontsize=8,
                       color="white" if key != "held" else "black")
            elif val >= 4:
                a.text(left + val + 1.2, i, f"{val:.1f}%", ha="left", va="center", fontsize=8)
            left += val
    a.set_yticks([0, 1])
    a.set_yticklabels([lab for _, lab in TARGETS])
    a.invert_yaxis()
    a.set_xlim(0, 112)
    a.set_xticks([0, 20, 40, 60, 80, 100])
    a.set_xlabel("share of 10 ms grid rows (%)")
    hnd = [plt.Rectangle((0, 0), 1, 1, color=col[k]) for k in ("independent", "held", "missing")]
    a.legend(hnd, ["measured", "held repeat", "missing"], ncol=3, frameon=False, fontsize=7.5,
             loc="lower center", bbox_to_anchor=(0.5, 1.0), handlelength=1.2, columnspacing=1.0)
    a.text(-0.16, 1.12, "(a)", transform=a.transAxes, fontsize=10)

    edges = np.array([1, 2, 3, 4, 6, 11, 21, 51, 10 ** 6])
    labels = ["10", "20", "30", "40-50", "60-100", "110-200", "210-500", ">500"]
    w = 0.38
    for i, (c, lab) in enumerate(TARGETS):
        g = np.array(gaps[c])
        h = np.array([((g >= edges[j]) & (g < edges[j + 1])).sum() for j in range(len(labels))])
        b.bar(np.arange(len(labels)) + (i - 0.5) * w, 100 * h / h.sum(), w,
              color=("#c44e52", "#dd8452")[i], label=f"{lab} ({len(g):,} gaps)")
    b.set_xticks(np.arange(len(labels)))
    b.set_xticklabels(labels, rotation=30, ha="right")
    b.set_xlabel("gap length (ms)")
    b.set_ylabel("share of gaps (%)")
    b.set_yscale("log")
    b.legend(frameon=False, fontsize=7.5)
    b.text(-0.17, 1.12, "(b)", transform=b.transAxes, fontsize=10)
    fig.tight_layout()
    fig.savefig(HERE / "fig_missing.pdf")
    fig.savefig(HERE / "fig_missing_preview.png", dpi=170)


if __name__ == "__main__":
    main()
