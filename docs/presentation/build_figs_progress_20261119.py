# -*- coding: utf-8 -*-
"""Figures of the progress slides added to the talk deck on 2026-11-19 (after B.64 and B.65).

    py docs/presentation/build_figs_progress_20261119.py   # -> docs/presentation/figs_20261119/p{1,2}_*.png

Every number is read from docs/paper/b64_two_stage.json or docs/paper/b65_inclusive.json and asserted equal,
to the printed precision, to the number written on the slides.  House style of build_figs_20261119.py.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / "docs" / "paper"
OUT = Path(__file__).resolve().parent / "figs_20261119"

BLUE, INK, INK2, MUTED, GRID, SURFACE = "#2a78d6", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fbfbf8"
ORANGE = "#eb6834"
FS, FS_TICK, FS_SMALL = 15, 13.5, 13

plt.rcParams.update({"font.family": ["Malgun Gothic", "DejaVu Sans"], "axes.unicode_minus": True,
                     "axes.edgecolor": GRID, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE})


def J(name):
    return json.loads((PAPER / name).read_text(encoding="utf-8"))


def iv(v):
    return f"{v[0]:+.3f} [{v[1]:+.3f}, {v[2]:+.3f}]".replace("-", "−")


def check(v, spec):
    assert iv(v) == spec, (iv(v), spec)
    return v


def forest(ax, rows, colors, names, xlim):
    """rows: list of (label, [v_a, v_b]); one marker per series, filled when the interval excludes 0."""
    n = len(rows)
    for i, (label, vals) in enumerate(rows):
        y0 = n - 1 - i
        for j, v in enumerate(vals):
            y = y0 + (0.17 if j == 0 else -0.17)
            excl = v[1] > 0 or v[2] < 0
            ax.plot([v[1], v[2]], [y, y], color=colors[j], lw=2.4, solid_capstyle="butt")
            ax.plot(v[0], y, "o", ms=9, color=colors[j], mfc=colors[j] if excl else SURFACE, mew=2)
            ax.text(v[2] + 0.006, y, iv(v), va="center", ha="left", fontsize=FS_SMALL, color=colors[j])
    ax.axvline(0, color=MUTED, lw=1)
    ax.set_yticks(range(n))
    ax.set_yticklabels([r[0] for r in rows][::-1], fontsize=FS)
    ax.set_xlim(*xlim)
    ax.set_ylim(-0.6, n - 0.4)
    ax.tick_params(axis="x", labelsize=FS_TICK, colors=INK2)
    ax.tick_params(axis="y", length=0, colors=INK)
    ax.set_xlabel("skill (평균 제곱 오차), 95 % 구간", fontsize=FS, color=INK2)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    fig = ax.figure
    fig.tight_layout()
    fig.subplots_adjust(top=0.87)
    x = ax.get_position().x0
    for j, nm in enumerate(names):
        t = fig.text(x, 0.94, "● " + nm, color=colors[j], fontsize=FS, ha="left", va="center")
        fig.canvas.draw()
        x = fig.transFigure.inverted().transform(t.get_window_extent())[1][0] + 0.04


def fig_two_stage():
    a = J("b64_two_stage.json")["all_rows"]
    k = "skill_mse|"
    rows = [
        ("인과 GP 대비", [check(a[k + "c|gp"], "+0.092 [+0.068, +0.114]"), check(a[k + "c2|gp"], "+0.176 [+0.155, +0.196]")]),
        ("같은 단계의 쌍둥이 대비", [check(a[k + "c|t"], "+0.053 [+0.028, +0.076]"), check(a[k + "c2|t2"], "+0.097 [+0.077, +0.116]")]),
        ("offline GP 대비", [check(a[k + "c|off"], "−0.033 [−0.062, −0.005]"), check(a[k + "c2|off"], "+0.063 [+0.038, +0.086]")]),
    ]
    check(a[k + "c2|c"], "+0.093 [+0.078, +0.107]")
    fig, ax = plt.subplots(figsize=(8.2, 5.0), dpi=200)
    forest(ax, rows, [MUTED, BLUE], ["신경망 혼자", "두 단계 estimator"], (-0.09, 0.36))
    fig.savefig(OUT / "p1_two_stage.png")
    plt.close(fig)


def fig_inclusive():
    a = J("b64_two_stage.json")["all_rows"]
    st = J("b65_inclusive.json")["statements"]
    rows = [
        ("인과 GP 대비", [check(a["skill_mse|c|gp"], "+0.092 [+0.068, +0.114]"), check(st["I1"]["value"], "+0.109 [+0.070, +0.146]")]),
        ("쌍둥이 대비\n(빠른 채널 증분)", [check(a["skill_mse|c|t"], "+0.053 [+0.028, +0.076]"), check(st["I2"]["value"], "−0.002 [−0.017, +0.010]")]),
    ]
    check(st["I3"]["value"], "+0.037 [+0.003, +0.070]")
    check(st["I4"]["value"], "−0.012 [−0.038, +0.006]")
    fig, ax = plt.subplots(figsize=(8.2, 4.4), dpi=200)
    forest(ax, rows, [BLUE, ORANGE], ["3 keV 컷 모집단", "3 keV 넘는 값 포함"], (-0.05, 0.23))
    fig.savefig(OUT / "p2_inclusive.png")
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    fig_two_stage()
    fig_inclusive()
    print("figures written")
