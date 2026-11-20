"""Figures of the rewritten manuscript (PAPER_DESIGN.md section 9): figv2_<name>.{pdf,png,json}.

Every plotted number is written to figv2_<name>.json, so a figure can be checked without the caches.
Result numbers come from the batch JSONs under docs/paper/ (and two md files parsed by make_tables_v2.py);
figures 1 (problem) and 7 (trace) read stored quantities from the git-ignored caches data/.b73, data/.b75,
data/.b79 (no estimator is fitted here).

Selection rule of the example discharge (no hand picking): among the test discharges of unit group V1,
re-partition r2, with at least 500 rows in the S set (target L5) and at least three absent steps (a time
step of more than 15 ms and less than 45 ms inside a segment of the 10 ms grid, i.e. a 20-40 ms step), the
one whose skill of TX over T0 under rule S (stored estimates of B.75, its own rows) is the median; with an
even number of candidates the lower median. The 25th-percentile discharge is the second example.

Run from the repo root:  py docs/paper/ppcf/fig_v2.py
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PAPER = ROOT / "docs" / "paper"
sys.path.insert(0, str(HERE))
import make_tables_v2 as mt  # noqa: E402  (md parsers only)

CM = 1 / 2.54
W = 16 * CM
# Okabe-Ito, a colour-blind-safe categorical palette, fixed order
C = {"H": "#999999", "G": "#56B4E9", "T0": "#E69F00", "TX": "#0072B2", "O": "#000000",
     "TE": "#009E73", "TB": "#CC79A7", "TN": "#D55E00", "TXN": "#0072B2", "ctl": "#666666", "ti": "#000000"}
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 8.5,
    "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "lines.linewidth": 1.2, "axes.spines.top": False,
    "axes.spines.right": False, "legend.frameon": False, "savefig.dpi": 300, "pdf.fonttype": 42,
})


def J(name):
    return json.loads((PAPER / name).read_text(encoding="utf-8"))


B75, B76, B78, B79 = (J("b75_availability.json"), J("b76_mast_availability.json"), J("b78_sparse.json"),
                      J("b79_beam_inputs.json"))


def save(fig, name, data, tokens):
    data = dict(data)
    data["printed_tokens"] = sorted(set(tokens))
    fig.savefig(HERE / f"figv2_{name}.pdf")
    fig.savefig(HERE / f"figv2_{name}.png", dpi=200)
    (HERE / f"figv2_{name}.json").write_text(json.dumps(data, indent=1, default=float), encoding="utf-8")
    plt.close(fig)
    print("wrote figv2_" + name)


def ci(cell):
    k = "cluster" if "cluster" in cell else "session"
    return 100 * cell["point"], 100 * cell[k][0], 100 * cell[k][1]


# ----------------------------------------------------------------------------------------------
# caches and the example discharge
# ----------------------------------------------------------------------------------------------
_C = {}


def caches():
    if not _C:
        d = ROOT / "data"
        _C["rows"] = np.load(d / ".b73/rows.npz")
        _C["grid"] = np.load(d / ".b73/grid.npz")
        _C["sets"] = np.load(d / ".b75/sets.npz")
        _C["S"] = np.load(d / ".b75/inputs/S.npz")
        _C["beam"] = np.load(d / ".b79/beam_cols.npz")
    return _C


def absent_steps(grid, f):
    a, b = grid["off"][f], grid["off"][f + 1]
    t, sg = grid["time"][a:b], grid["seg"][a:b]
    dt = np.diff(t)
    same = sg[1:] == sg[:-1]
    k = np.where(same & (dt > 0.015) & (dt < 0.045))[0]
    return t[k], t[k + 1]          # last row before, first row after


def select_example():
    c = caches()
    rows, grid, sets = c["rows"], c["grid"], c["sets"]
    est = {}
    for k in range(5):
        for e in ("T0", "TX"):
            z = np.load(ROOT / f"data/.b75/fits/V1_r2_k{k}__{e}__S__L5.npz")
            est.setdefault(e, []).append((z["idx"], z["est"]))
    idx = np.concatenate([a for a, _ in est["T0"]])
    t0 = np.concatenate([b for _, b in est["T0"]])
    idx_x = np.concatenate([a for a, _ in est["TX"]])
    tx = np.concatenate([b for _, b in est["TX"]])
    assert np.array_equal(idx, idx_x)
    y = rows["tg_L5"][idx]
    shot = rows["shot"][idx]
    fidx = rows["fidx"][idx]
    nS = {}
    S_L5 = sets["S_L5"]
    for s_, n in zip(*np.unique(rows["shot"][S_L5], return_counts=True)):
        nS[int(s_)] = int(n)
    cand = []
    for s_ in np.unique(shot):
        m = shot == s_
        f = int(fidx[m][0])
        nabs = len(absent_steps(grid, f)[0])
        if nS.get(int(s_), 0) < 500 or nabs < 3:
            continue
        sk = 1 - np.sum((tx[m] - y[m]) ** 2) / np.sum((t0[m] - y[m]) ** 2)
        cand.append((sk, int(s_), f, int(m.sum()), nabs))
    cand.sort()
    med = cand[(len(cand) - 1) // 2]
    q25 = cand[int(np.floor(0.25 * (len(cand) - 1)))]
    info = {"rule": ("test discharges of V1 re-partition r2 (folds k0-k4), >= 500 rows in the S set (L5), "
                     ">= 3 absent steps (15 ms < step < 45 ms inside a segment); median of the skill of TX over T0 "
                     "under S on the discharge's scored rows (lower median for an even count); second example = "
                     "25th percentile (index floor(0.25 (n - 1)) of the sorted skills)"),
            "candidates": len(cand),
            "median": {"skill": med[0], "shot": med[1], "rows": med[3], "absent_steps": med[4]},
            "q25": {"skill": q25[0], "shot": q25[1], "rows": q25[3], "absent_steps": q25[4]}}
    sel = {"idx": idx, "t0": t0, "tx": tx}
    return info, med, q25, sel


# ----------------------------------------------------------------------------------------------
# 1. problem
# ----------------------------------------------------------------------------------------------
def fig_problem(info, med):
    c = caches()
    grid, beam, sets = c["grid"], c["beam"], c["sets"]
    f = med[2]
    a, b = grid["off"][f] + 1, grid["off"][f + 1] - 1          # drop the first and last rows of the file
    t = grid["time"][a:b]
    ti = grid["ti"][a:b]
    p1 = np.nansum(beam["P"][a:b, 0:2], axis=1)
    bes = grid["fast"][a:b, 0:9].mean(axis=1).astype(float)
    ece = grid["fast"][a:b, 9:13].mean(axis=1).astype(float)
    nb, na = absent_steps(grid, f)
    age = c["S"]["age"][sets["S_L5"]] / 1000.0
    q = np.percentile(age, [10, 25, 50, 75, 90])
    edges = np.arange(0, 2.05, 0.05)
    hist, _ = np.histogram(np.clip(age, 0, 2.0 - 1e-9), bins=edges)

    fig = plt.figure(figsize=(W, 10.5 * CM), layout="constrained")
    gs = fig.add_gridspec(3, 2, width_ratios=[2.3, 1])
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
    ax3 = fig.add_subplot(gs[2, 0], sharex=ax1)
    ax4 = fig.add_subplot(gs[:, 1])
    ax1.plot(t, ti, ".", ms=1.6, color=C["ti"], label="reported $T_i$")
    ax1.set_ylabel("$T_i$ (eV)")
    for ax in (ax1, ax2, ax3):
        for x in na:
            ax.axvline(x, color="#F0B48C", lw=0.6, zorder=0)
    ax2.plot(t, p1, color="#555555", lw=0.8)
    ax2.set_ylabel("NBI-1 (MW)")
    medb, mede = np.nanmedian(bes), np.nanmedian(ece)
    ax3.plot(t, bes / medb, color=C["TB"], lw=0.7, label="BES array mean")
    ax3.plot(t, ece / mede, color=C["TE"], lw=0.7, label="ECEI array mean")
    # rows in runs of at least ten rows in which all four ECEI channels are unchanged (b75_stuck_ecei.md rule)
    e4 = grid["fast"][a:b, 9:13]
    same = np.r_[False, np.all(np.diff(e4, axis=0) == 0, axis=1)]
    const = np.zeros(len(t), bool)
    i = 0
    while i < len(t):
        if same[i]:
            j = i
            while j < len(t) and same[j]:
                j += 1
            if j - (i - 1) >= 10:
                const[i - 1:j] = True
            i = j
        else:
            i += 1
    spans = []
    i = 0
    while i < len(t):
        if const[i]:
            j = i
            while j + 1 < len(t) and const[j + 1]:
                j += 1
            spans.append((float(t[i]), float(t[j])))
            i = j + 1
        else:
            i += 1
    for k_, (x0, x1) in enumerate(spans):
        ax3.axvspan(x0, x1, color="#DDDDDD", lw=0, zorder=0, label="ECEI constant (kept)" if k_ == 0 else None)
    ax3.legend(loc="upper center", ncol=3, bbox_to_anchor=(0.5, -0.38))
    ax3.set_ylabel("relative level")
    ax3.set_xlabel("time (s)")
    for ax in (ax1, ax2):
        plt.setp(ax.get_xticklabels(), visible=False)
    ax1.text(0.0, 1.04, f"(a) discharge {med[1]}; vertical lines: absent frames (beam notches)",
             transform=ax1.transAxes, fontsize=8)
    ax4.bar(edges[:-1], hist / hist.sum(), width=0.05, align="edge", color="#0072B2", edgecolor="white", lw=0.3)
    ax4.axvline(q[2], color="black", lw=0.8, ls="--")
    ax4.text(q[2] + 0.03, ax4.get_ylim()[1] * 0.92, f"median {q[2]:.2f} s", fontsize=8)
    ax4.set_xlabel("age of the value (s)")
    fig.align_ylabels([ax1, ax2, ax3])
    ax4.set_ylabel("share of scored rows")
    ax4.text(0.0, 1.015, "(b) rule S: age of newest value", transform=ax4.transAxes, fontsize=8)
    data = {"example": info, "shot": med[1], "time_s": t.tolist(), "ti_eV": ti.tolist(), "nbi1_MW": p1.tolist(),
            "bes_mean_over_median": (bes / medb).tolist(), "ecei_mean_over_median": (ece / mede).tolist(),
            "absent_step_first_row_after_s": na.tolist(),
            "ecei_constant_spans_s": spans,
            "ecei_constant_rule": "runs of at least ten grid rows in which all four ECEI channels are unchanged (docs/paper/b75_stuck_ecei.md); rows kept",
            "age_hist": {"edges_s": edges.tolist(), "share": (hist / hist.sum()).tolist(),
                         "last_bin_holds_ages_above_s": 2.0},
            "age_quantiles_s": dict(zip(["0.1", "0.25", "0.5", "0.75", "0.9"], q.tolist())),
            "rows": int(len(age)), "source": "data/.b73/grid.npz, data/.b79/beam_cols.npz, data/.b75/inputs/S.npz (age, ms) on S_L5 of data/.b75/sets.npz"}
    toks = [str(med[1]), f"{q[2]:.2f}", f"{len(age):,}", str(info["candidates"]), str(med[3])]
    save(fig, "problem", data, toks)


# ----------------------------------------------------------------------------------------------
# 2. rules (schematic, no data)
# ----------------------------------------------------------------------------------------------
def fig_rules():
    fig, ax = plt.subplots(figsize=(W, 6.5 * CM), layout="constrained")
    n = 30                       # frames shown, index 0..29, row t = 29
    tt = 29
    notch = [7, 18, 25]          # absent frames (schematic)
    rules = [("fast samples (BES, ECEI)", lambda i: i <= tt, "fast"),
             ("D10: previous frame", lambda i: i <= tt - 1, "ces"),
             ("D$d$: older than $t-d$ ($d$ = 60 ms)", lambda i: i <= tt - 6, "ces"),
             ("S: before the latest notch", lambda i: i < notch[-1], "ces"),
             ("K$k$: every $k$-th frame ($k$ = 5)", lambda i: i <= tt - 1 and i % 5 == 0, "ces")]
    for r, (lab, vis, kind) in enumerate(rules):
        y = len(rules) - 1 - r
        for i in range(n):
            if kind == "ces" and i in notch:
                ax.plot(i, y, marker="x", color="#D55E00", ms=4, mew=0.8)
                continue
            if i == tt and kind == "ces":
                ax.add_patch(Rectangle((i - 0.35, y - 0.25), 0.7, 0.5, fill=False, ec="#999999", lw=0.6, ls=":"))
                continue
            v = vis(i)
            col = C["TE"] if kind == "fast" else C["TX"]
            ax.plot(i, y, marker="s" if kind == "ces" else "|", ms=4.5 if kind == "ces" else 7,
                    mfc=col if v else "white", mec=col if v else "#BBBBBB", color=col if v else "#BBBBBB", mew=0.8)
        ax.text(-1.0, y, lab, ha="right", va="center", fontsize=8)
    ax.axvline(tt, color="black", lw=0.6, ls="--")
    ax.text(tt, len(rules) - 0.35, "row $t$", ha="center", fontsize=8)
    ax.set_xlim(-0.8, n + 0.5)
    ax.set_ylim(-0.7, len(rules) - 0.1)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xticks(range(0, n, 5))
    ax.set_xticklabels([f"{10 * (i - tt)}".replace("-", "−") for i in range(0, n, 5)])
    ax.set_xlabel("time relative to row $t$ (ms), 10 ms frames")
    handles = [Line2D([], [], marker="s", ls="", mfc=C["TX"], mec=C["TX"], ms=4.5, label="CES value visible"),
               Line2D([], [], marker="s", ls="", mfc="white", mec="#BBBBBB", ms=4.5, label="not visible"),
               Line2D([], [], marker="x", ls="", color="#D55E00", ms=4, label="absent frame (notch)"),
               Line2D([], [], marker="s", ls="", mfc="white", mec="#999999", ms=4.5, label="frame of row $t$ (scored)")]
    fig.legend(handles=handles, loc="outside lower center", ncol=4, fontsize=7.5)
    save(fig, "rules", {"schematic": True, "frames": n, "row_t": tt, "notches": notch, "d_ms": 60, "k": 5}, [])


# ----------------------------------------------------------------------------------------------
# 3. ladder
# ----------------------------------------------------------------------------------------------
def fig_ladder():
    m = B75["main"]["V1|L5|common"]["all"]
    rules = ["D10", "D50", "D100", "D300", "D650", "D1000"]
    x = [int(r[1:]) / 1000 for r in rules]
    sS = B75["main"]["V1|L5|S"]["all"]["S"]["rmse_eV"]
    agemed = B75["age_under_S"]["L5"]["newest_value_age_s"]["quantiles"]["0.5"]
    mm = B76["main"]["M1|L5|common"]["all"]
    mr = ["D5", "D15", "D25", "D50", "D100", "D150"]
    mx = [int(r[1:]) / 1000 for r in mr]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(W, 7.0 * CM), layout="constrained", gridspec_kw={"width_ratios": [1.25, 1]})
    data = {"kstar": {"delay_s": x, "rmse_eV": {}}, "mast": {"delay_s": mx, "rmse_eV": {}}}
    toks = []
    lab = {"H": "hold", "G": "causal GP", "T0": "T0 (history only)", "TX": "TX (with fast diagnostics)"}
    for e in ("H", "G", "T0", "TX"):
        v = [m[r]["rmse_eV"][e] for r in rules]
        a1.plot(x, v, marker="o", ms=3.5, color=C[e], label=lab[e])
        a1.plot([agemed], [sS[e]], marker="D", ms=5, ls="", mfc="white", mec=C[e], mew=1.2)
        data["kstar"]["rmse_eV"][e] = v
        data["kstar"]["rmse_eV"]["S_" + e] = sS[e]
        mv = [mm[r]["rmse_eV"][e] for r in mr]
        a2.plot(mx, mv, marker="o", ms=3.5, color=C[e], label=lab[e])
        data["mast"]["rmse_eV"][e] = mv
        toks += ["%.1f" % z for z in v + mv + [sS[e]]]
    o = m["D10"]["rmse_eV"]["O"]
    a1.axhline(o, color=C["O"], lw=0.8, ls="--")
    a1.plot([agemed], [sS["O"]], marker="D", ms=5, ls="", mfc=C["O"], mec=C["O"])
    mo = mm["D5"]["rmse_eV"]["O"]
    a2.axhline(mo, color=C["O"], lw=0.8, ls="--")
    a2.text(0.0052, mo + 4, "O, the frame's own reading", fontsize=8)
    data["kstar"]["O_eV"], data["kstar"]["S_O_eV"], data["kstar"]["S_x_s"] = o, sS["O"], agemed
    data["mast"]["O_eV"] = mo
    toks += ["%.1f" % o, "%.1f" % sS["O"], "%.1f" % mo, "%.2f" % agemed]
    for ax in (a1, a2):
        ax.set_xscale("log")
        ax.set_ylabel("RMSE of the level estimate (eV)")
        ax.grid(axis="y", lw=0.3, color="#DDDDDD")
    a1.set_xlabel("delay $d$ of the CES history (s)")
    a2.set_xlabel("delay $d$ of the CXRS history (s)")
    a1.set_xticks(x)
    a1.set_xticklabels([f"{z:g}" for z in x])
    a2.set_xticks(mx)
    a2.set_xticklabels([f"{z:g}" for z in mx])
    a1.text(0.0, 1.03, "(a) KSTAR, random design V1, target L5", transform=a1.transAxes, fontsize=8)
    a2.text(0.0, 1.03, "(b) MAST, random design M1, target L5", transform=a2.transAxes, fontsize=8)
    h = a1.get_legend_handles_labels()
    extra = Line2D([], [], marker="D", ls="", mfc="white", mec="black", ms=5,
                   label="rule S (S set), at the median age %.2f s" % agemed)
    oline = Line2D([], [], color=C["O"], lw=0.8, ls="--", label="O, the frame's own reading")
    a1.legend(h[0] + [oline, extra], h[1] + [oline.get_label(), extra.get_label()], loc="upper left", fontsize=7.5)
    data["source"] = "b75 main.V1|L5|common.all.D*.rmse_eV, main.V1|L5|S.all.S.rmse_eV; b76 main.M1|L5|common.all.D*.rmse_eV"
    save(fig, "ladder", data, toks)


# ----------------------------------------------------------------------------------------------
# 4. designs (forest)
# ----------------------------------------------------------------------------------------------
DL = {"V1": "V1 random", "V2": "V2 clusters held out", "V3": "V3 rolling origin", "V4a": "V4 fit to 31991",
      "V4b": "V4 fit to 32310", "M1": "M1 random", "M2": "M2 sessions held out", "M3": "M3 rolling origin",
      "M4": "M4 chronological"}


def forest(ax, items, colors=None):
    """items: list of (label, point, lo, hi, colour or None, marker). Top to bottom."""
    n = len(items)
    for i, it in enumerate(items):
        y = n - 1 - i
        if it is None:
            continue
        lab, p, lo, hi, col, mk = it
        col = col or "black"
        if lo is not None:
            ax.plot([lo, hi], [y, y], color=col, lw=1.0)
        filled = lo is not None and (lo > 0 or hi < 0)
        ax.plot(p, y, marker=mk, ms=4, color=col, mfc=col if filled else "white", mec=col, mew=1.0, ls="")
    ax.set_yticks([n - 1 - i for i, it in enumerate(items) if it is not None])
    ax.set_yticklabels([it[0] for it in items if it is not None])
    ax.axvline(0, color="black", lw=0.6)
    ax.set_ylim(-0.7, n - 0.3)
    ax.grid(axis="x", lw=0.3, color="#DDDDDD")


def fig_designs():
    items_k, items_m, data, toks = [], [], {"kstar": {}, "mast": {}}, []
    rc = {"D10": "#999999", "D650": "#0072B2", "S": "#D55E00", "D5": "#999999", "D100": "#0072B2"}
    for r in ("D10", "D650", "S"):
        for d in ("V1", "V2", "V3", "V4a", "V4b"):
            cell = (B75["main"][f"{d}|L5|common"]["all"][r]["TX/T0"] if r != "S"
                    else B75["main"][f"{d}|L5|S"]["all"]["S"]["TX/T0"])
            p, lo, hi = ci(cell)
            items_k.append((f"{r}  {DL[d]}", p, lo, hi, rc[r], "o"))
            data["kstar"][f"{r}|{d}"] = [p, lo, hi]
            toks += ["%.1f" % z for z in (p, lo, hi)]
        items_k.append(None)
    items_k = items_k[:-1]
    for r in ("D5", "D100"):
        for d in ("M1", "M2", "M3", "M4"):
            p, lo, hi = ci(B76["main"][f"{d}|L5|common"]["all"][r]["TX/T0"])
            items_m.append((f"{r}  {DL[d]}", p, lo, hi, rc[r], "s"))
            data["mast"][f"{r}|{d}"] = [p, lo, hi]
            toks += ["%.1f" % z for z in (p, lo, hi)]
        items_m.append(None)
    items_m = items_m[:-1]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(W, 9.0 * CM), layout="constrained")
    forest(a1, items_k)
    forest(a2, items_m)
    for ax in (a1, a2):
        ax.set_xlabel("MSE reduction (%)")
    a1.set_title("(a) KSTAR: BES and ECEI", loc="left", fontsize=8.5)
    a2.set_title("(b) MAST: Thomson", loc="left", fontsize=8.5)
    data["interval"] = "95 %, KSTAR shot clusters, MAST sessions resampled; filled marker: interval excludes zero"
    data["source"] = "b75 main.*|L5|{common,S}.all.{D10,D650,S}.TX/T0; b76 main.M*|L5|common.all.{D5,D100}.TX/T0"
    save(fig, "designs", data, toks)


# ----------------------------------------------------------------------------------------------
# 5. which diagnostic
# ----------------------------------------------------------------------------------------------
def fig_which():
    data, toks = {}, []
    B81 = J("b81_controls.json")
    est = [("TE/T0", "ECEI only (TE vs T0)", C["TE"], "o", "b81"), ("TB/T0", "BES only (TB vs T0)", C["TB"], "o", "b81"),
           ("TX/T0", "both (TX vs T0)", C["TX"], "o", "b75"), ("TN/T0", "beam alone (TN vs T0)", C["TN"], "^", "b79"),
           ("TXN/TN", "both, beam given (TXN vs TN)", "#000000", "D", "b79")]
    designs = ["V1", "V2", "V3", "V4a", "V4b"]
    fig, axes = plt.subplots(1, 3, figsize=(W, 9.0 * CM), layout="constrained",
                             gridspec_kw={"width_ratios": [1, 1, 0.9]})

    def group(ax, j, i, k, n, cell, col, mk, keyname):
        p, lo, hi = ci(cell)
        y = (n - 1 - j) * (k + 1.5) + (k - 1 - i)
        ax.plot([lo, hi], [y, y], color=col, lw=0.9)
        filled = lo > 0 or hi < 0
        ax.plot(p, y, marker=mk, ms=3.6, color=col, mfc=col if filled else "white", mec=col, ls="")
        data[keyname] = [p, lo, hi]
        toks.extend("%.1f" % z for z in (p, lo, hi))

    for ax, r in zip(axes[:2], ("D650", "S")):
        k = len(est)
        for j, d in enumerate(designs):
            c5 = B75["main"][f"{d}|L5|common"]["all"]["D650"] if r == "D650" else B75["main"][f"{d}|L5|S"]["all"]["S"]
            c9 = B79["main"][f"{d}|L5|common"]["all"]["D650"] if r == "D650" else B79["main"][f"{d}|L5|S"]["all"]["S"]
            c1 = B81["main"][d][r]
            for i, (key, lab, col, mk, src) in enumerate(est):
                # TE, TB: B.75 where it registered them (V1, V4a, V4b), B.81 only for V2 and V3
                cell = c5[key] if (src == "b81" and key in c5) else {"b75": c5, "b79": c9, "b81": c1}[src][key]
                group(ax, j, i, k, len(designs), cell, col, mk, f"{r}|{d}|{key}")
        yt = [(len(designs) - 1 - j) * (k + 1.5) + (k - 1) / 2 for j in range(len(designs))]
        ax.set_yticks(yt)
        ax.set_yticklabels([DL[d] for d in designs] if r == "D650" else [])
        ax.axvline(0, color="black", lw=0.6)
        ax.grid(axis="x", lw=0.3, color="#DDDDDD")
        ax.set_xlabel("MSE reduction (%)")
        ax.set_title("(a) rule D650" if r == "D650" else "(b) rule S",
                     loc="left", fontsize=8)
    toks.append("0.65")
    # (c) controls of B.81 under D650, by design, and MAST M1 D100 (B.76)
    ax = axes[2]
    ctl = [("TXp/T0", "fast inputs permuted (TXp vs T0)", "#999999", "o"),
           ("TXo/T0", "fast inputs at the anchor row (TXo vs T0)", "#E69F00", "s"),
           ("TX/TXo", "present over anchor row (TX vs TXo)", "#000000", "D")]
    k = len(ctl)
    groups = [(d, B81["main"][d]["D650"]) for d in designs] + [("M1", B76["main"]["M1|L5|common"]["all"]["controls"])]
    for j, (d, cell) in enumerate(groups):
        for i, (key, lab, col, mk) in enumerate(ctl):
            group(ax, j, i, k, len(groups), cell[key], col, mk, f"control|D650|{d}|{key}" if d != "M1" else f"control|MAST M1 D100|{key}")
    yt = [(len(groups) - 1 - j) * (k + 1.5) + (k - 1) / 2 for j in range(len(groups))]
    ax.set_yticks(yt)
    ax.set_yticklabels(["V1", "V2", "V3", "V4 (31991)", "V4 (32310)", "MAST M1, D100"])
    ax.yaxis.tick_right()
    ax.axvline(0, color="black", lw=0.6)
    ax.grid(axis="x", lw=0.3, color="#DDDDDD")
    ax.set_xlabel("MSE reduction (%)")
    ax.set_title("(c) controls, D650", loc="left", fontsize=8)
    hs = [Line2D([], [], marker=mk, color=col, ls="-", lw=0.9, ms=3.6, label=lab) for key, lab, col, mk, src in est]
    hs += [Line2D([], [], marker=mk, color=col, ls="-", lw=0.9, ms=3.6, label=lab) for key, lab, col, mk in ctl]
    fig.legend(handles=hs, loc="outside lower center", ncol=3)
    data["note"] = ("filled marker: 95 % cluster (MAST session) interval excludes zero; target L5; TE, TB of V1, V4 from B.75, "
                    "of V2, V3 from B.81; controls from B.81 (KSTAR, D650) and B.76 (MAST M1, D100)")
    data["source"] = ("b75 main.{V1,V4a,V4b}.{D650,S}.{TE,TB}/T0 and main.*.{D650,S}.TX/T0; b81 main.{V2,V3}.{D650,S}.{TE,TB}/T0 "
                      "and main.*.D650.{TXp/T0,TXo/T0,TX/TXo}; "
                      "b79 main.*.{D650,S}.{TN/T0,TXN/TN}; b76 main.M1|L5|common.all.controls")
    save(fig, "which", data, toks)


# ----------------------------------------------------------------------------------------------
# 6. sparse
# ----------------------------------------------------------------------------------------------
def fig_sparse():
    m = B78["main"]["V1|L5"]["all"]
    ks = ["K1", "K2", "K5", "K10", "K20", "K50"]
    x = [10 * int(k[1:]) for k in ks]
    t0 = [m[k]["rmse_eV"]["T0"] for k in ks]
    tx = [m[k]["rmse_eV"]["TX"] for k in ks]
    o = m["K1"]["rmse_eV"]["O"]
    kp = m["K10"]["equivalent rate"]["point"]["k_prime"]
    kplo, kphi = m["K10"]["equivalent rate"]["cluster"]["k_prime"]
    fig, ax = plt.subplots(figsize=(10 * CM, 7.0 * CM), layout="constrained")
    ax.plot(x, t0, marker="o", ms=3.5, color=C["T0"], label="T0 (history only)")
    ax.plot(x, tx, marker="o", ms=3.5, color=C["TX"], label="TX (with fast diagnostics)")
    ax.axhline(o, color=C["O"], lw=0.8, ls="--")
    ax.text(10.5, o + 2, "O, the frame's own reading", fontsize=8)
    y10 = m["K10"]["rmse_eV"]["TX"]
    ax.plot([10 * kp, 100], [y10, y10], color="#555555", lw=0.7, ls=":")
    ax.plot([10 * kp, 10 * kp], [ax.get_ylim()[0], y10], color="#555555", lw=0.7, ls=":")
    ax.errorbar([10 * kp], [y10], xerr=[[10 * (kp - kplo)], [10 * (kphi - kp)]], fmt="none", ecolor="#555555",
                elinewidth=1.0, capsize=2)
    ax.annotate("equivalent spacing for TX at K10:\n$k'$ = %.2f frames (%.1f ms)" % (kp, 10 * kp),
                xy=(10 * kp, y10), xytext=(110, 123), fontsize=8,
                arrowprops={"arrowstyle": "-", "lw": 0.5, "color": "#555555"})
    ax.set_xscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([str(z) for z in x])
    ax.set_xlabel("spacing of the CES values (ms)")
    ax.set_ylabel("RMSE of the level estimate (eV)")
    ax.grid(axis="y", lw=0.3, color="#DDDDDD")
    ax.legend(loc="upper left")
    data = {"spacing_ms": x, "rmse_T0_eV": t0, "rmse_TX_eV": tx, "O_eV": o, "K10_equivalent_k_prime": kp,
            "K10_equivalent_k_prime_cluster_interval": [kplo, kphi], "K10_equivalent_ms": 10 * kp,
            "source": "b78 main.V1|L5.all.K*.rmse_eV, K10.equivalent rate"}
    toks = ["%.1f" % z for z in t0 + tx + [o, 10 * kp]] + ["%.2f" % kp, "%.2f" % kplo, "%.2f" % kphi]
    save(fig, "sparse", data, toks)


# ----------------------------------------------------------------------------------------------
# 7. trace
# ----------------------------------------------------------------------------------------------
def fig_trace(info, ex, sel, name="trace"):
    c = caches()
    rows, grid = c["rows"], c["grid"]
    shot, f = ex[1], ex[2]
    m = rows["shot"][sel["idx"]] == shot
    ridx = sel["idx"][m]
    order = np.argsort(rows["gi"][ridx])
    ridx = ridx[order]
    t0 = sel["t0"][m][order]
    tx = sel["tx"][m][order]
    tr = grid["time"][rows["gi"][ridx]]
    l5 = rows["tg_L5"][ridx]
    a, b = grid["off"][f] + 1, grid["off"][f + 1] - 1
    tg, tig = grid["time"][a:b], grid["ti"][a:b]
    nb, na = absent_steps(grid, f)
    qs = np.quantile(tr, [0.25, 0.5, 0.75])
    half = 1.0
    fig, axes = plt.subplots(1, 3, figsize=(W, 6.8 * CM), sharey=True, layout="constrained")
    data = {"example": info, "shot": int(shot), "window_rule": "three 2 s windows centred at the 25, 50 and 75 % "
            "quantiles of the times of the discharge's scored rows", "windows": []}
    toks = [str(shot)]
    for ax, qc in zip(axes, qs):
        lo, hi = qc - half, qc + half
        mg = (tg >= lo) & (tg <= hi)
        mr = (tr >= lo) & (tr <= hi)
        ax.plot(tg[mg], tig[mg], ".", ms=2.2, color="#AAAAAA", label="reported $T_i$")
        ax.plot(tr[mr], l5[mr], color="black", lw=1.0, label="following level L5")
        ax.plot(tr[mr], t0[mr], color=C["T0"], lw=0.9, label="T0 (history only), rule S")
        ax.plot(tr[mr], tx[mr], color=C["TX"], lw=0.9, label="TX (with fast diagnostics), rule S")
        for x in na[(na >= lo) & (na <= hi)]:
            ax.axvline(x, color="#F0B48C", lw=0.6, zorder=0)
        ax.set_xlim(lo, hi)
        ax.set_xlabel("time (s)")
        ax.grid(axis="y", lw=0.3, color="#DDDDDD")
        sse0 = float(np.nansum((t0[mr] - l5[mr]) ** 2))
        ssex = float(np.nansum((tx[mr] - l5[mr]) ** 2))
        data["windows"].append({"t_lo_s": lo, "t_hi_s": hi, "scored_rows": int(mr.sum()),
                                "time_s": tr[mr].tolist(), "L5_eV": l5[mr].tolist(), "T0_eV": t0[mr].tolist(),
                                "TX_eV": tx[mr].tolist(), "grid_time_s": tg[mg].tolist(), "grid_ti_eV": tig[mg].tolist(),
                                "absent_step_first_row_after_s": na[(na >= lo) & (na <= hi)].tolist(),
                                "rmse_T0_eV": float(np.sqrt(sse0 / max(mr.sum(), 1))),
                                "rmse_TX_eV": float(np.sqrt(ssex / max(mr.sum(), 1)))})
    axes[0].set_ylabel("$T_i$ (eV)")
    allm = np.isfinite(l5)
    data["discharge_rows"] = int(len(ridx))
    data["discharge_rmse_T0_eV"] = float(np.sqrt(np.mean((t0[allm] - l5[allm]) ** 2)))
    data["discharge_rmse_TX_eV"] = float(np.sqrt(np.mean((tx[allm] - l5[allm]) ** 2)))
    data["discharge_skill_TX_over_T0"] = ex[0]
    toks += [str(len(ridx)), "%.1f" % data["discharge_rmse_T0_eV"], "%.1f" % data["discharge_rmse_TX_eV"],
             "%+.3f" % ex[0], "%.1f" % (100 * ex[0])]
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="outside lower center", ncol=2)
    fig.suptitle(f"discharge {shot}, random design V1 (r2), target L5; vertical lines: absent frames", fontsize=8.5, x=0.02, ha="left")
    data["source"] = "data/.b75/fits/V1_r2_k*__{T0,TX}__S__L5.npz (idx into data/.b73/rows.npz), data/.b73/grid.npz"
    save(fig, name, data, toks)


# ----------------------------------------------------------------------------------------------
# 8. limits
# ----------------------------------------------------------------------------------------------
def fig_limits():
    blocks = ["block up to 31991", "block 32002-32310", "block from 32312"]
    labs = ["up to 31991", "32002-32310", "from 32312"]
    lb = mt.parse_last_block()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(W, 6.8 * CM), layout="constrained")
    data, toks = {"increment": {}, "notch": {}}, []
    for i, blk in enumerate(blocks):
        for j, (src, key, col, mk, lab) in enumerate(
                ((B75, "TX/T0", C["TX"], "o", "TX vs T0"), (B79, "TXN/TN", "#000000", "D", "TXN vs TN (beam given)"))):
            p, lo, hi = ci(src["main"]["V1|L5|S"][blk]["S"][key])
            x = i + (j - 0.5) * 0.25
            a1.plot([x, x], [lo, hi], color=col, lw=1.0)
            filled = lo > 0 or hi < 0
            a1.plot(x, p, marker=mk, ms=4, color=col, mfc=col if filled else "white", mec=col, ls="",
                    label=lab if i == 0 else None)
            data["increment"][f"{blk}|{key}"] = [p, lo, hi]
            toks += ["%.1f" % z for z in (p, lo, hi)]
    a1.axhline(0, color="black", lw=0.6)
    a1.set_xticks(range(3))
    a1.set_xticklabels(labs)
    a1.set_xlabel("block of shots")
    a1.set_ylabel("reduction of the MSE under S (%)")
    a1.legend(loc="lower left")
    a1.grid(axis="y", lw=0.3, color="#DDDDDD")
    a1.text(0.0, 1.03, "(a) random design V1, target L5, rule S", transform=a1.transAxes, fontsize=8)
    for j, (key, lab, col, mk) in enumerate((("n_ti", "$T_i$", "#000000", "o"), ("n_bes", "BES", C["TB"], "s"),
                                             ("n_ecei", "ECEI", C["TE"], "^"))):
        v = [float(lb[b][key]) for b in blocks]
        xs = np.arange(3) + (j - 1) * 0.2
        a2.plot(xs, v, marker=mk, ls="", ms=4.5, color=col, label=lab)
        data["notch"][key] = v
        toks += [lb[b][key] for b in blocks]
    a2.axhline(1, color="black", lw=0.6, ls="--")
    a2.set_xticks(range(3))
    a2.set_xticklabels(labs)
    a2.set_xlabel("block of shots")
    a2.set_ylabel("first row after a notch / level before")
    a2.legend(loc="upper left")
    a2.grid(axis="y", lw=0.3, color="#DDDDDD")
    a2.text(0.0, 1.03, "(b) response to a beam notch (after the fact, no interval)", transform=a2.transAxes, fontsize=8)
    data["source"] = ("b75 main.V1|L5|S.<block>.S.TX/T0; b79 main.V1|L5|S.<block>.S.TXN/TN; "
                      "docs/paper/b75_last_block.md code block section (7)")
    save(fig, "limits", data, toks)


# ----------------------------------------------------------------------------------------------
# 9. time since the last notch (B.82 part A)
# ----------------------------------------------------------------------------------------------
def fig_tau():
    A = J("b82_localize.json")["part_A"]
    classes = ["up to 100 ms", "above 100 and up to 300 ms", "above 300 ms", "no notch before"]
    labels = ["up to\n100 ms", "100 to\n300 ms", "above\n300 ms", "no notch\nbefore"]
    designs = [("V1", "V1 random", "#0072B2", "o"), ("V2", "V2 clusters held out", "#E69F00", "s"),
               ("V3", "V3 rolling origin", "#009E73", "^"), ("V4a", "V4 fit to 31991", "#CC79A7", "D")]
    fig, axes = plt.subplots(1, 2, figsize=(W, 6.3 * CM), sharey=True, layout="constrained")
    data, toks = {}, []
    for ax, r in zip(axes, ("D650", "S")):
        for j, (d, lab, col, mk) in enumerate(designs):
            t = A[d][r]["all"]["tau"]
            for i, k in enumerate(classes):
                cell = t.get(k, {}).get("TX/T0")
                if cell is None:
                    continue
                p, lo, hi = ci(cell)
                x = i + (j - 1.5) * 0.16
                ax.plot([x, x], [lo, hi], color=col, lw=1.0)
                filled = lo > 0 or hi < 0
                ax.plot(x, p, marker=mk, ms=4, color=col, mfc=col if filled else "white", mec=col, ls="",
                        label=lab if i == 0 else None)
                data[f"{r}|{d}|{k}"] = [p, lo, hi]
                toks += ["%.1f" % z for z in (p, lo, hi)]
        ax.axhline(0, color="black", lw=0.6)
        ax.set_xticks(range(4))
        ax.set_xticklabels(labels)
        ax.set_xlim(-0.5, 3.5)
        ax.grid(axis="y", lw=0.3, color="#DDDDDD")
        ax.set_xlabel("time since the last notch")
        ax.set_title("(a) rule D650" if r == "D650" else "(b) rule S", loc="left", fontsize=8.5)
    axes[0].set_ylabel("MSE reduction (%)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="outside right center", fontsize=7.5)
    data["note"] = ("TX vs T0, target L5, 95 % cluster interval, filled marker: interval excludes zero; "
                    "under S every row has a notch before it (no marker in that class)")
    data["source"] = "b82 part_A.{V1,V2,V3,V4a}.{D650,S}.all.tau.*.TX/T0"
    save(fig, "tau", data, toks)


def main():
    info, med, q25, sel = select_example()
    print("example discharge (median):", med[1], "skill %.3f" % med[0], "candidates", info["candidates"],
          "| 25th percentile:", q25[1], "skill %.3f" % q25[0])
    fig_problem(info, med)
    fig_rules()
    fig_ladder()
    fig_designs()
    fig_which()
    fig_sparse()
    fig_trace(info, med, sel, "trace")
    fig_trace(info, q25, sel, "trace_q25")
    fig_limits()
    fig_tau()


if __name__ == "__main__":
    main()
