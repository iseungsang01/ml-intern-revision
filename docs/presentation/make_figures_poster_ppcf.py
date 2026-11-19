# -*- coding: utf-8 -*-
"""Figures of the A0 PPCF poster, drawn at their placed size (one poster column wide, printed 1:1).

  py docs/presentation/make_figures_poster_ppcf.py              # all figures, English
  py docs/presentation/make_figures_poster_ppcf.py --lang ko    # the same figures with Korean text (*_ko.png)
  py docs/presentation/make_figures_poster_ppcf.py --trace      # also recompute the example traces (CPU)
  py docs/presentation/make_figures_poster_ppcf.py --only where   # the optional thirds figure (not on the poster)
  -> docs/presentation/figures/poster_ppcf/poster_*.png  (Korean: poster_*_ko.png)

Why these are re-drawn and not re-exported.  The manuscript figures (docs/paper/ppcf/fig_paper.py,
fig_trace_causal.py) are laid out for a 16 cm text width with 7 to 9 pt type set as literals inside
each drawing function.  A poster column is 25.3 cm, so re-exporting them at a higher dpi would put
their text at 11 to 14 pt, below the 18 pt the poster needs; a font size cannot be changed from
outside without editing those scripts.  So each drawing is copied here and laid out again for the
column width with 18 pt or larger type.  The data are the same: the JSON outputs read by
fig_paper.py, through its own loader J() and its colours; the positions of the geometry figure are
the ones typed in fig_paper.fig_geometry; the example traces are recomputed by the code of
fig_trace_causal.py (same run, same discharges, same windows, same row-level check against the
stored errors; CPU only) and cached in figures/poster_ppcf/poster_trace_cache.npz.

All words drawn in a figure are in TX (English and Korean); the drawing code is shared.  The Korean
figures use Pretendard (the poster's Korean font); formulas stay in the mathtext font of the English.
Every text artist of every figure is checked to be at least MIN_PT at the placed size, and every
label drawn inside a box is checked to fit inside it.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PPCF = ROOT / "docs" / "paper" / "ppcf"
OUT = HERE / "figures" / "poster_ppcf"
sys.path.insert(0, str(PPCF))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.font_manager as fm  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402

import fig_paper as FP  # noqa: E402  (manuscript figure script: loader J, colours, interval rule)

CW_CM = 25.3                 # poster column width (cm); build_poster_ppcf.py uses the same value
CW = CW_CM / 2.54            # inches
DPI = 250
MIN_PT = 18.0
NET, TWIN, GP, HOLD, OFF, FILT, FILT2, TWO = FP.NET, FP.TWIN, FP.GP, FP.HOLD, FP.OFF, FP.FILT, FP.FILT2, FP.TWO
INK, MUTED, GRID = "#15181e", "#5e6878", "#d9dde3"

plt.rcParams.update({
    "font.family": "Segoe UI", "font.size": 19, "axes.labelsize": 19, "xtick.labelsize": 18,
    "ytick.labelsize": 18, "legend.fontsize": 18, "axes.titlesize": 20, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "mathtext.fontset": "custom", "mathtext.rm": "Segoe UI",
    "mathtext.it": "Segoe UI:italic", "mathtext.bf": "Segoe UI:bold", "mathtext.sf": "Segoe UI",
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 1.4, "axes.edgecolor": INK,
    "xtick.major.width": 1.4, "ytick.major.width": 1.4, "xtick.major.size": 6, "ytick.major.size": 6,
    "xtick.color": INK, "ytick.color": INK, "axes.labelcolor": INK, "text.color": INK,
    "lines.linewidth": 2.4, "legend.frameon": False, "legend.handlelength": 1.6,
    "figure.constrained_layout.h_pad": 0.06, "figure.constrained_layout.w_pad": 0.06,
})

# ------------------------------------------------------------------------------------------------ words
TX = {
    "en": {
        "suffix": "",
        # geometry
        "g_band": "published pedestal tops\n(other discharges)",
        "g_sep": "midplane separatrix,\none 2022 discharge",
        "g_shape": "schematic $T_i$ shape",
        "g_chord": "CES chord, R = 2.198 m",
        "g_rows": ("CES channels", "BES view, #31923", "BES columns 5–7", "ECEI column 3"),
        "g_notes": ("chord and its\nneighbours", None, "inferred from\nchannel names", "±2–3 cm between\ndischarges"),
        "g_xlabel": "major radius R (m)",
        # rows
        "r_lanes": ("beam", "CES frames", "fast channels", "file rows"),
        "r_xlabel": "time relative to the stamp of row t (ms)",
        "r_blip": "blip",
        "r_absent": "absent",
        "r_frames": "frame t: estimated;\nearlier: CES history",
        "r_sample": "one raw sample per\nrow, at its stamp",
        "r_step": "20 ms step = one absent frame",
        "r_row": "estimate of row t\nformed at its stamp",
        # schematic
        "s_a": "(a) two-stage estimator",
        "s_w": (3.02, 1.78, 1.86), "s_fs": 18.5,       # widths of the input, network, second-stage boxes (in)
        "s_fast": ["fast channels at row t:", "9 BES · 4 ECEI · 2 Mirnov"],
        "s_ces": ["CES history before t:", "last value · age · flag"],
        "s_net": ["first stage:", "recurrent", "network", "(LSTM) over", "every row"],
        "s_two": ["second stage:", "boosted trees", "estimate the", "network's", "error"],
        "s_summ": "summaries of BES and ECEI over the last ten rows, CES history",
        "s_b": r"(b) five-parameter robust filter on $x = \log T_i$",
        "s_pred": [r"predict: $x_t = x_{t-1} + b_c\,\Delta c_t + b_g\,\Delta g_t$", "random walk, variance q"],
        "s_upd": ["update with a CES value:", r"weight $\min(1, k\sqrt{S}/|v|)$, variance r"],
        "s_next": "next row",
        # trace
        "t_blip": "absent frame (beam blip)",
        "t_hold": "hold",
        "t_gp": "causal GP (CES history only)",
        "t_net": "network (first stage)",
        "t_rep": r"reported $T_i$",
        "t_title": "({p}) #{shot}: {miss} of {n} rows without $T_i$; reported {lo:.0f} to {hi:.0f} eV",
        "t_ylabel": r"$T_i$ (eV)",
        "t_xlabel": "time (s)",
        # blips
        "b_lab": ("BES level", "ECEI level", "reported $T_i$"),
        "b_xlabel": "row after the absent frame (10 ms rows)",
        "b_ylabel": "ratio to the six rows before",
        "b_dotted": "dotted: control rows without a blip",
        "b_title": "{n:,} blips of one absent frame",
        # where
        "w_thirds": ("coldest", "middle", "hottest"),
        "w_series": ("network over causal GP", "network over its control", "KFX over KF0 (filter)"),
        "w_ylabel": "skill",
        "w_xlabel": "third of the discharges by median $T_i$",
    },
    "ko": {
        "suffix": "_ko",
        "g_band": "발표된 pedestal top\n(다른 방전)",
        "g_sep": "중앙면 separatrix,\n2022년 방전 하나",
        "g_shape": "$T_i$ 모양 (개념도)",
        "g_chord": "CES 측정선, R = 2.198 m",
        "g_rows": ("CES 채널", "BES 시야, #31923", "BES 5–7열", "ECEI 3열"),
        "g_notes": ("측정선과 이웃 채널", None, "채널 이름에서 추정", "방전마다 ±2–3 cm"),
        "g_xlabel": "대반경 R (m)",
        "r_lanes": ("빔", "CES 프레임", "빠른 채널", "파일의 행"),
        "r_xlabel": "행 t의 시각 기준 시간 (ms)",
        "r_blip": "blip",
        "r_absent": "없음",
        "r_frames": "프레임 t: 추정 대상\n이전 프레임: CES 이력",
        "r_sample": "행마다 원시 표본 하나\n(행의 시각에서)",
        "r_step": "20 ms 간격 = 빠진 프레임 하나",
        "r_row": "행 t의 추정은\n그 시각에 만든다",
        "s_a": "(a) 두 단계 estimator",
        "s_w": (3.38, 1.56, 1.72), "s_fs": 18.0,
        "s_fast": ["행 t의 빠른 채널:", "BES 9 · ECEI 4 · Mirnov 2"],
        "s_ces": ["t 이전의 CES 이력:", "마지막 값 · 경과 시간 · 관측 여부"],
        "s_net": ["첫 단계:", "순환 신경망", "(LSTM),", "모든 행을", "차례로"],
        "s_two": ["둘째 단계:", "gradient", "boosting이", "신경망의", "오차를 추정"],
        "s_summ": "최근 열 행의 BES · ECEI 요약값과 CES 이력",
        "s_b": r"(b) $x = \log T_i$의 5-parameter robust filter",
        "s_pred": [r"예측: $x_t = x_{t-1} + b_c\,\Delta c_t + b_g\,\Delta g_t$", "random walk, 분산 q"],
        "s_upd": ["CES 값으로 갱신:", r"가중치 $\min(1, k\sqrt{S}/|v|)$, 분산 r"],
        "s_next": "다음 행",
        "t_blip": "빠진 프레임 (빔 blip)",
        "t_hold": "hold",
        "t_gp": "인과 GP (CES 이력만)",
        "t_net": "신경망 (첫 단계)",
        "t_rep": r"보고된 $T_i$",
        "t_title": "({p}) #{shot}: {n}행 가운데 {miss}행에 $T_i$ 없음, 보고값 {lo:.0f}–{hi:.0f} eV",
        "t_ylabel": r"$T_i$ (eV)",
        "t_xlabel": "시간 (s)",
        "b_lab": ("BES 수준", "ECEI 수준", "보고된 $T_i$"),
        "b_xlabel": "빠진 프레임 뒤의 행 (10 ms 간격)",
        "b_ylabel": "앞선 여섯 행 대비 비",
        "b_dotted": "점선: blip이 없는 대조 행",
        "b_title": "프레임 하나가 빠진 빔 blip {n:,}건",
        "w_thirds": ("가장 차가운", "중간", "가장 뜨거운"),
        "w_series": ("인과 GP 대비 신경망", "비교군 대비 신경망", "KF0 대비 KFX (filter)"),
        "w_ylabel": "skill",
        "w_xlabel": "중앙값 $T_i$로 나눈 방전의 3분의 1",
    },
}
L = TX["en"]
BOXED = []                   # (text artist, box width in inches) of labels drawn inside boxes


def set_lang(lang):
    """Words of one language; the Korean figures also switch the text font to Pretendard."""
    global L
    L = TX[lang]
    if lang == "ko":
        user = Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft" / "Windows" / "Fonts"
        for f in ("Pretendard-Regular.otf", "Pretendard-SemiBold.otf", "Pretendard-Bold.otf"):
            if (user / f).exists():
                fm.fontManager.addfont(str(user / f))
        if "Pretendard" not in {f.name for f in fm.fontManager.ttflist}:
            raise SystemExit("FATAL: Pretendard is not installed")
        plt.rcParams["font.family"] = "Pretendard"


def J(name):
    return FP.J(name)


def excl(v, ref=0.0):
    return FP.excl(v, ref)


def check_fonts(fig, name):
    """Every visible text of the figure must be at least MIN_PT (mathtext sub/superscripts excepted)."""
    small = []
    for t in fig.findobj(matplotlib.text.Text):
        if t.get_visible() and t.get_text().strip() and t.get_fontsize() < MIN_PT - 1e-6:
            small.append((t.get_text()[:30], t.get_fontsize()))
    if small:
        raise SystemExit(f"FATAL: {name}: text below {MIN_PT} pt: {small}")


def check_boxed(fig, name):
    """Every label drawn inside a box must be narrower than the box (0.08 in of padding on each side)."""
    r = fig.canvas.get_renderer()
    wide = []
    for t, bw in BOXED:
        w = t.get_window_extent(r).width / fig.dpi
        if w > bw - 0.16:
            wide.append((t.get_text()[:30], round(w, 2), round(bw, 2)))
    BOXED.clear()
    if wide:
        raise SystemExit(f"FATAL: {name}: label wider than its box: {wide}")


def save(fig, name):
    check_fonts(fig, name)
    check_boxed(fig, name)
    OUT.mkdir(parents=True, exist_ok=True)
    w, h = fig.get_size_inches()
    if abs(w - CW) > 1e-6:
        raise SystemExit(f"FATAL: {name} is {w:.3f} in wide, not one column ({CW:.3f} in)")
    name = name + L["suffix"]
    fig.savefig(OUT / f"{name}.png", dpi=DPI)
    plt.close(fig)
    print("written %s.png  %.1f x %.1f cm" % (name, w * 2.54, h * 2.54))


# ------------------------------------------------------------------------------------------------ 1
def geometry():
    """fig_paper.fig_geometry, re-laid out: positions along R (Supplementary S10 and Table S7)."""
    fig = plt.figure(figsize=(CW, 5.45), layout="constrained")
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.75])
    a1 = fig.add_subplot(gs[0])
    a2 = fig.add_subplot(gs[1], sharex=a1)
    R = np.linspace(2.08, 2.35, 500)
    ped = 0.42 * (1 - np.tanh((R - 2.222) / 0.009)) / 2          # schematic shape only, as in fig_paper
    prof = 0.06 + ped + 0.28 * np.clip((2.21 - R) / 0.13, 0, None)
    a1.axvspan(2.19, 2.23, color="#f3e3c3", zorder=0)
    a1.plot(R, prof, color=MUTED, lw=2.4, ls="--")
    a1.axvline(2.24, color=INK, lw=1.8, ls=":")
    a1.axvline(2.198, color=NET, lw=3.0)
    a1.text(2.186, 1.30, L["g_band"], ha="right", va="top", fontsize=18, color="#7a5a12")
    a1.text(2.245, 1.30, L["g_sep"], ha="left", va="top", fontsize=18)
    a1.text(2.349, 0.10, L["g_shape"], fontsize=18, color=MUTED, ha="right", va="bottom")
    a1.text(2.194, 0.03, L["g_chord"], color=NET, fontsize=19, ha="right", va="bottom", fontweight="bold")
    a1.set_ylim(0, 1.32)
    a1.set_yticks([])
    a1.spines["left"].set_visible(False)
    a1.tick_params(axis="x", labelbottom=False, length=0)

    pos = [([2.192, 2.198, 2.203], True, NET), ((2.12, 2.34), True, FILT2), ((2.16, 2.21), False, FILT),
           ((2.165, 2.215), True, GP)]
    rows = [(lab, x, pub, c, note) for lab, note, (x, pub, c) in zip(L["g_rows"], L["g_notes"], pos)]
    for i, (lab, x, pub, c, note) in enumerate(rows):
        y = len(rows) - 1 - i
        if isinstance(x, list):
            for xx in x:
                a2.plot(xx, y, "|", ms=26, mew=4.0 if xx == 2.198 else 2.2, color=c)
        else:
            a2.add_patch(Rectangle((x[0], y - 0.27), x[1] - x[0], 0.54, fc=c if pub else "white", ec=c,
                                   hatch=None if pub else "////", lw=2.0, alpha=0.85 if pub else 1.0))
        if note:
            a2.text(2.247, y, note, va="center", ha="left", fontsize=18, color=INK, linespacing=1.1)
    a2.set_yticks([len(rows) - 1 - i for i in range(len(rows))])
    a2.set_yticklabels([r[0] for r in rows])
    a2.tick_params(axis="y", length=0)
    a2.axvline(2.198, color=NET, lw=1.6, alpha=0.6)
    a2.axvline(2.24, color=INK, lw=1.8, ls=":")
    a2.set_ylim(-0.6, len(rows) - 0.4)
    a2.spines["left"].set_visible(False)
    a2.set_xlim(2.08, 2.35)
    a2.set_xticks([2.10, 2.15, 2.20, 2.25, 2.30, 2.35])
    a2.set_xlabel(L["g_xlabel"])
    save(fig, "poster_geometry")


# ------------------------------------------------------------------------------------------------ 2
def rows_schematic():
    """fig_paper.fig_rows, re-laid out: what a row of the 10 ms files holds (schematic)."""
    fig = plt.figure(figsize=(CW, 4.6))
    ax = fig.add_axes([0.158, 0.2, 0.835, 0.79])
    ax.set_xlim(-52, 41)
    ax.set_xticks([-50, -40, -30, -20, -10, 0, 10])
    ax.spines["bottom"].set_bounds(-52, 10)
    ax.set_ylim(-1.12, 3.55)
    for sp in ("left", "right", "top"):
        ax.spines[sp].set_visible(False)
    ax.set_yticks([3, 2, 1, 0])
    ax.set_yticklabels(list(L["r_lanes"]))
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel(L["r_xlabel"])
    ax.plot([-52, -30, -30, -20, -20, 10], [3.25, 3.25, 2.78, 2.78, 3.25, 3.25], color=INK, lw=2.4)
    ax.text(-25, 3.06, L["r_blip"], ha="center", va="center", fontsize=18)
    for x0 in (-50, -40, -30, -20, -10, 0):
        absent, target = x0 == -30, x0 == 0
        fc = "white" if absent else ("#fbe9e9" if target else "#e8f0fa")
        ec = MUTED if absent else (NET if target else GP)
        ax.add_patch(Rectangle((x0 + 0.45, 1.72), 9.1, 0.56, fc=fc, ec=ec, lw=2.0, ls="--" if absent else "-"))
    ax.text(-25.0, 2.0, L["r_absent"], ha="center", va="center", fontsize=18, color=MUTED)
    ax.text(5.0, 2.0, "t", ha="center", va="center", fontsize=20, color=NET, fontweight="bold",
            fontstyle="italic")
    ax.text(11.5, 2.0, L["r_frames"], ha="left", va="center", fontsize=18)
    tt = np.linspace(-52, 10, 600)
    sig = 1.0 + 0.12 * np.sin(tt / 2.3) + 0.06 * np.sin(tt / 0.7)
    sig = np.where((tt > -30) & (tt < -20), sig - 0.28, sig)
    ax.plot(tt, sig, color=FILT2, lw=1.8)
    stamps = [-50, -40, -20, -10, 0]
    for x0 in stamps:
        y0 = float(np.interp(x0 + 0.5, tt, sig))
        ax.plot(x0 + 0.5, y0, "o", ms=10, color=FILT if x0 == 0 else FILT2, mec=FILT, mew=1.6)
        ax.plot([x0 + 0.5, x0 + 0.5], [0.14, y0 - 0.1], color=GRID, lw=1.6)
    ax.text(11.5, 1.0, L["r_sample"], ha="left", va="center", fontsize=18)
    for x0 in stamps:
        ax.plot(x0 + 0.5, 0, "s", ms=12, color=NET if x0 == 0 else INK)
    ax.annotate("", xy=(-19.5, -0.36), xytext=(-39.5, -0.36),
                arrowprops=dict(arrowstyle="<->", color=MUTED, lw=1.8, mutation_scale=22))
    ax.text(-29.5, -0.55, L["r_step"], ha="center", va="top", fontsize=18, color=MUTED)
    ax.text(11.5, 0.0, L["r_row"], ha="left", va="center", fontsize=18)
    r = fig.canvas.get_renderer()
    right = max(t.get_window_extent(r).x1 for t in ax.texts) / fig.dpi
    if right > CW - 0.02:
        raise SystemExit(f"FATAL: rows: a label runs off the figure ({right:.2f} > {CW:.2f} in)")
    save(fig, "poster_rows")


# ------------------------------------------------------------------------------------------------ 3
def _box(ax, x, y, w, h, lines, fc, ec, fs=18.5, bold_first=False, ls="-", lw=2.2, align="center"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.09", fc=fc, ec=ec, lw=lw,
                                ls=ls))
    if isinstance(lines, str):
        lines = [lines]
    n = len(lines)
    pitch = fs * 1.32 / 72.0
    y0 = y + h / 2 + (n - 1) * pitch / 2
    xx = x + w / 2 if align == "center" else x + 0.14
    if n * pitch > h:
        raise SystemExit(f"FATAL: {n} lines do not fit a box {h:.2f} in high: {lines[0]}")
    for i, s in enumerate(lines):
        t = ax.text(xx, y0 - i * pitch, s, ha=align, va="center", fontsize=fs,
                    fontweight="bold" if (bold_first and i == 0) else "normal")
        BOXED.append((t, w))


def _arrow(ax, x0, y0, x1, y1, c=INK, ls="-", lw=2.2, rad=0.0):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=24, color=c, lw=lw, ls=ls,
                                 shrinkA=0, shrinkB=0, connectionstyle=f"arc3,rad={rad}"))


def schematic():
    """fig_paper.fig_schematic, re-laid out: (a) two-stage estimator, (b) five-parameter filter.
    Positions are measured down from the top (d) and turned into axes coordinates at the end."""
    H = 5.86
    fig = plt.figure(figsize=(CW, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, CW)
    ax.set_ylim(0, H)
    ax.axis("off")
    Y = lambda d: H - d
    x0 = 0.04
    ax.text(x0, Y(0.04), L["s_a"], fontsize=20, fontweight="bold", va="top")
    t0 = 0.52                                    # top of the input boxes
    (iw, nw, sw), fs = L["s_w"], L["s_fs"]
    ih, gap = 0.93, 0.16
    _box(ax, x0, Y(t0 + ih), iw, ih, L["s_fast"], "#fff1df", FILT, fs=fs)
    _box(ax, x0, Y(t0 + 2 * ih + gap), iw, ih, L["s_ces"], "#e8f0fa", GP, fs=fs)
    cy = Y(t0 + ih + gap / 2)
    nx, nh = x0 + iw + 0.42, 2.02
    _box(ax, nx, cy - nh / 2, nw, nh, L["s_net"], "#fbe9e9", NET, fs=fs, bold_first=True)
    _arrow(ax, x0 + iw + 0.02, Y(t0 + ih / 2), nx - 0.03, cy + 0.35, FILT)
    _arrow(ax, x0 + iw + 0.02, Y(t0 + 1.5 * ih + gap), nx - 0.03, cy - 0.35, GP)
    ox, ow = nx + nw + 0.34, 0.92
    ax.add_patch(FancyBboxPatch((ox, cy - 0.33), ow, 0.66, boxstyle="round,pad=0,rounding_size=0.33", fc="white",
                                ec=NET, lw=2.2))
    ax.text(ox + ow / 2, cy, r"$\hat{T}_i^{(1)}$", ha="center", va="center", fontsize=21)
    _arrow(ax, nx + nw + 0.02, cy, ox - 0.03, cy, NET)
    sx = ox + ow + 0.34
    _box(ax, sx, cy - nh / 2, sw, nh, L["s_two"], "#f6e6ea", TWO, fs=fs, bold_first=True)
    _arrow(ax, ox + ow + 0.02, cy, sx - 0.03, cy, NET)
    tx = sx + sw + 0.32
    tw = CW - tx - 0.05
    ax.add_patch(FancyBboxPatch((tx, cy - 0.33), tw, 0.66, boxstyle="round,pad=0,rounding_size=0.33", fc="white",
                                ec=TWO, lw=2.6))
    ax.text(tx + tw / 2, cy, r"$\hat{T}_i$", ha="center", va="center", fontsize=21, fontweight="bold")
    _arrow(ax, sx + sw + 0.02, cy, tx - 0.03, cy, TWO)
    # the second stage also reads summaries of the inputs and the CES history
    d_by = t0 + 2 * ih + gap + 0.34
    dash = (0, (4, 3))
    ax.plot([x0 + 0.5, x0 + 0.5], [Y(t0 + 2 * ih + gap) - 0.02, Y(d_by)], color=TWO, lw=2.0, ls=dash)
    ax.plot([x0 + 0.5, sx + sw / 2], [Y(d_by), Y(d_by)], color=TWO, lw=2.0, ls=dash)
    _arrow(ax, sx + sw / 2, Y(d_by), sx + sw / 2, cy - nh / 2 - 0.02, TWO, ls=dash, lw=2.0)
    ax.text(x0 + 0.62, Y(d_by + 0.07), L["s_summ"], fontsize=18, va="top", color=TWO)
    # (b) filter
    d_rule = d_by + 0.56
    ax.plot([x0, CW - 0.04], [Y(d_rule), Y(d_rule)], color=GRID, lw=1.6)
    ax.text(x0, Y(d_rule + 0.12), L["s_b"], fontsize=20, fontweight="bold", va="top")
    d_f = d_rule + 0.62
    fh = 0.98
    fy = Y(d_f + fh)
    pw = 4.55
    _box(ax, x0, fy, pw, fh, L["s_pred"], "#fff1df", FILT, fs=18.5)
    ux = x0 + pw + 0.5
    uw = CW - ux - 0.04
    _box(ax, ux, fy, uw, fh, L["s_upd"], "#e8f0fa", GP, fs=18.5)
    _arrow(ax, x0 + pw + 0.02, fy + fh / 2, ux - 0.03, fy + fh / 2)
    _arrow(ax, ux + uw * 0.45, fy - 0.03, x0 + pw * 0.55, fy - 0.03, rad=-0.16)
    ax.text(x0 + pw + 0.25, fy - 0.50, L["s_next"], ha="center", va="top", fontsize=18, color=MUTED)
    if fy - 0.50 - 0.3 < 0.0:
        raise SystemExit("FATAL: schematic runs off the figure")
    save(fig, "poster_schematic")


# ------------------------------------------------------------------------------------------------ 4
TRACE_CACHE = OUT / "poster_trace_cache.npz"
TRACE_REPORT = OUT / "poster_trace_check.json"


def trace_compute():
    """The computation of fig_trace_causal.main (same run, discharges, windows, check), CPU only."""
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import fig_trace_causal as FTC  # sets the protocol env of the causal run, CPU device
    import torch
    FT = FTC.FT
    torch.set_num_threads(4)
    RUN = FTC.RUN
    metrics = json.loads((RUN / "metrics.json").read_text(encoding="utf-8"))
    if metrics["seq_model"] != "v2" or bool(metrics["per_shot_input_norm"]):
        raise SystemExit("FATAL: the run is not the causal variant")
    stats = FT._load_stats(metrics)
    manifest = json.loads((FT.SPLIT / "split_manifest.json").read_text(encoding="utf-8"))
    grid, _ = FT.load_grid_files(ROOT / "data", True)
    valid_names = sorted(grid)
    (typ_sc, typ_name), n_typ, _ = FT.typical_shot(grid, manifest["test_files"])
    shots = list(FT.SHOTS) + [typ_name]
    with torch.no_grad():
        lookup = FT.build_pred_lookup(ROOT / "data", shots, stats, torch.device("cpu"))["full"]
    npz = np.load(RUN / "comparison_errors_test.npz")
    report, ok, arrays = {"run": RUN.name, "typical_discharge": typ_name}, True, {}
    for k, name in enumerate(shots):
        time, ti, model, pers, gp, seg_id = FT.estimates(grid[name], lookup[name])
        c = FT.check(name, time, ti, model, pers, gp, npz, valid_names)
        report[name] = c
        for label in ("persistence", "causal_gp"):
            ok &= c[label]["share_within_1e-3"] >= 0.999
        ok &= c["model"]["max_abs_dev_of_error_eV"] <= 0.5
        ok &= c["max_value_mismatch_eV"] < 0.05
        lo, hi, _ = FT.pick_window(time, ti, seg_id)
        for key, arr in (("time", time), ("ti", ti), ("model", model), ("pers", pers), ("gp", gp)):
            arrays[f"{k}_{key}"] = arr[lo:hi]
        arrays[f"{k}_name"] = np.array(name)
    report["check_passed"] = bool(ok)
    OUT.mkdir(parents=True, exist_ok=True)
    TRACE_REPORT.write_text(json.dumps(report, indent=1), encoding="utf-8")
    if not ok:
        raise SystemExit("FATAL: recomputed estimates do not match the stored errors; no figure.")
    np.savez(TRACE_CACHE, n=len(shots), **arrays)
    print("trace cache written; check passed")


def trace():
    """fig_trace_causal, re-laid out for the column: three discharges, 0.65 s each."""
    if not TRACE_CACHE.exists():
        trace_compute()
    rep = json.loads(TRACE_REPORT.read_text(encoding="utf-8"))
    if not rep["check_passed"]:
        raise SystemExit("FATAL: trace check failed")
    D = np.load(TRACE_CACHE)
    ref = json.loads((PPCF / "fig_trace_causal.json").read_text(encoding="utf-8"))
    n = int(D["n"])
    fig, axes = plt.subplots(n, 1, figsize=(CW, 8.55), layout="constrained")
    for k, ax in enumerate(axes):
        t, ti, model, pers, gp = (D[f"{k}_{x}"] for x in ("time", "ti", "model", "pers", "gp"))
        name = str(D[f"{k}_name"])
        obs = ~np.isnan(ti)
        # same windows as the manuscript figure
        if [round(float(t[0]), 3), round(float(t[-1]), 3)] != [round(v, 3) for v in ref[name]["window_s"]]:
            raise SystemExit(f"FATAL: window of {name} differs from fig_trace_causal.json")
        for r in np.flatnonzero(~obs):
            ax.axvspan(t[r] - 0.005, t[r] + 0.005, color="#e9ebef", lw=0, zorder=0)
        dt = np.diff(t)
        for j, b in enumerate(np.flatnonzero((dt > 0.015) & (dt < 0.045))):
            ax.axvline(0.5 * (t[b] + t[b + 1]), color=TWIN, lw=2.0, ls=":",
                       label=L["t_blip"] if j == 0 else None)
        ax.plot(t, pers, color="0.55", lw=2.0, drawstyle="steps-post", label=L["t_hold"])
        ax.plot(t, gp, color=GP, lw=2.2, ls="--", label=L["t_gp"])
        ax.plot(t, model, color=NET, lw=2.8, label=L["t_net"])
        ax.plot(t[obs], ti[obs], "o", ms=7.5, mfc="white", mec="k", mew=1.4, label=L["t_rep"], zorder=5)
        vv = ti[obs]
        ax.set_title(L["t_title"].format(p="abc"[k], shot=name[1:6], miss=int((~obs).sum()), n=len(t),
                                         lo=vv.min(), hi=vv.max()), fontsize=19, fontweight="normal", pad=6)
        ax.set_ylabel(L["t_ylabel"])
        ax.margins(x=0.01)
        ax.locator_params(axis="y", nbins=4)
        ax.tick_params(axis="x", pad=3)
    axes[-1].set_xlabel(L["t_xlabel"])
    hs, ls = [], []
    for ax in axes:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in ls:
                hs.append(h)
                ls.append(l)
    order = [ls.index(x) for x in (L["t_rep"], L["t_net"], L["t_gp"], L["t_hold"], L["t_blip"]) if x in ls]
    fig.legend([hs[i] for i in order], [ls[i] for i in order], ncol=2, loc="outside upper center",
               columnspacing=1.4, handlelength=1.8)
    save(fig, "poster_trace")


# ------------------------------------------------------------------------------------------------ 5
def blips():
    """fig_paper.fig_blips panel (a), re-laid out: event means after one absent frame (10 ms files)."""
    S = J("b58_blip_calibration.json")["S"]["all"]
    fig, a = plt.subplots(figsize=(CW, 4.1), layout="constrained")
    pr, ci, ctl = S["profiles"], S["profiles_ci95"], S["control_profiles"]
    x = np.arange(1, len(pr["ti"]) + 1)
    for (k, c), lab in zip((("bes", HOLD), ("ecei", FILT), ("ti", GP)), L["b_lab"]):
        a.fill_between(x, [p[0] for p in ci[k]], [p[1] for p in ci[k]], color=c, alpha=0.22, lw=0)
        a.plot(x, pr[k], "-o", color=c, lw=2.8, ms=8, label=lab)
        a.plot(x, ctl[k], ":", color=c, lw=2.0)
    a.axhline(1, color=GRID, lw=1.4)
    a.set_xticks(x)
    a.set_xlim(0.6, len(x) + 0.4)
    a.set_ylim(0.62, 1.07)
    a.set_xlabel(L["b_xlabel"])
    a.set_ylabel(L["b_ylabel"])
    a.legend(loc="lower right", ncol=1)
    a.text(len(x) + 0.3, 1.045, L["b_dotted"], ha="right", fontsize=18, color=MUTED)
    a.set_title(L["b_title"].format(n=S["events"]), fontsize=19, fontweight="normal")
    save(fig, "poster_blips")


# ------------------------------------------------------------------------------------------------ 6
def where():
    """fig_paper.fig_where panel (a), re-laid out: skill by thirds of the discharges by median T_i."""
    R = J("b54_pooled_rows.json")["reported"]
    K = J("b56_kfx.json")
    th = [(lab, key, conf) for lab, (key, conf) in
          zip(L["w_thirds"], (("low", "confirm_third_coldest"), ("middle", "confirm_third_middle"),
                              ("high", "confirm_third_hottest")))]
    fig, a = plt.subplots(figsize=(CW, 4.7), layout="constrained")
    series = [(L["w_series"][0], lambda t: R["thirds"][t[1]]["gp"], GP, -0.22),
              (L["w_series"][1], lambda t: R["thirds"][t[1]]["t"], NET, 0.0),
              (L["w_series"][2], lambda t: K[t[2]]["kfx_over_kf0"], FILT, 0.22)]
    for lab, f, c, dx in series:
        for i, t in enumerate(th):
            v = f(t)
            a.plot([i + dx, i + dx], [v[1], v[2]], color=c, lw=3.0)
            a.plot(i + dx, v[0], "o", ms=12, mec=c, mew=2.4, mfc=c if excl(v) else "white")
    a.axhline(0, color=MUTED, lw=1.4)
    a.set_xticks(range(3))
    a.set_xticklabels([t[0] for t in th])
    a.set_xlim(-0.5, 2.5)
    a.set_ylabel(L["w_ylabel"])
    a.set_xlabel(L["w_xlabel"])
    a.grid(axis="y", color=GRID, lw=1.0)
    a.set_ylim(-0.04, 0.2)
    a.legend(handles=[Line2D([], [], marker="o", ls="-", color=c, mfc=c, ms=11, lw=2.6, label=lab)
                      for lab, _, c, _ in series], loc="upper left", ncol=1)
    save(fig, "poster_where")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", choices=("en", "ko"), default="en")
    ap.add_argument("--trace", action="store_true", help="recompute the example traces (CPU inference)")
    ap.add_argument("--only", nargs="*", help="names of figures to draw")
    a = ap.parse_args()
    set_lang(a.lang)
    if a.trace:
        trace_compute()
    figs = {"geometry": geometry, "rows": rows_schematic, "schematic": schematic, "trace": trace, "blips": blips,
            "where": where}
    for k, f in figs.items():
        # 'where' (skill by temperature third) is drawn only on request: the poster gives those numbers in
        # its Table 4 and shows the blip figure instead
        if (not a.only and k != "where") or (a.only and k in a.only):
            f()


if __name__ == "__main__":
    main()
