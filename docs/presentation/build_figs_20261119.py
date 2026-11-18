# -*- coding: utf-8 -*-
"""Figures of the 10-slide talk of 2026-11-19 (one per slide 2-9).

    py docs/presentation/build_figs_20261119.py   # -> docs/presentation/figs_20261119/s{2..9}_*.{png,pdf}

Every number that a JSON output of the repository holds is read from that JSON and asserted equal, to the
printed precision, to the number written in the talk's content specification; numbers no JSON holds are typed
from the specification and marked TYPED below.  House style of docs/paper/ppcf/fig_story.py: one hue, a filled
marker means the 95 % interval excludes the reference and a hollow one that it includes it, intervals as
horizontal bars, a thin reference line, direct labels, Malgun Gothic, no titles inside the image.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / "docs" / "paper"
OUT = Path(__file__).resolve().parent / "figs_20261119"

BLUE, INK, INK2, MUTED, GRID, SURFACE = "#2a78d6", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fbfbf8"
ORANGE, AQUA = "#eb6834", "#1baf7a"
PALE = "#d6e4f5"
FS, FS_TICK, FS_SMALL = 15, 13.5, 13

plt.rcParams.update({"font.family": ["Malgun Gothic", "DejaVu Sans"], "axes.unicode_minus": True, "pdf.fonttype": 42,
                     "axes.edgecolor": GRID, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE})


def J(name):
    return json.loads((PAPER / name).read_text(encoding="utf-8"))


def f3(x, signed=True):
    s = f"{x:+.3f}" if signed else f"{x:.3f}"
    return s.replace("-", "−")


def iv(v, signed=True):
    return f"{f3(v[0], signed)} [{f3(v[1], signed)}, {f3(v[2], signed)}]"


def check(v, spec, signed=True):
    """Assert that a [point, lo, hi] triple prints exactly as the specification writes it."""
    got = iv(v, signed)
    assert got == spec, (got, spec)
    return v


def check1(x, spec, nd=3):
    got = f"{x:.{nd}f}"
    assert got == spec, (got, spec)
    return x


def excludes(v, ref):
    return v[1] > ref or v[2] < ref


def style_axes(ax, xgrid=True):
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(axis="x", labelsize=FS_TICK, colors=INK2, length=3, color=GRID)
    ax.tick_params(axis="y", length=0, labelsize=FS, colors=INK)
    if xgrid:
        ax.grid(axis="x", color=GRID, lw=0.7, zorder=0)


def interval_rows(ax, rows, ref, xlim, emph=(), value_labels=True, groups=()):
    """rows: list of (label, [point, lo, hi]); top to bottom."""
    n = len(rows)
    for i, (lab, v) in enumerate(rows):
        y = n - 1 - i
        strong = lab in emph
        c = BLUE
        ax.plot([v[1], v[2]], [y, y], color=c, lw=3.2 if strong else 2.2, solid_capstyle="round", zorder=2)
        ax.plot(v[0], y, "o", ms=11 if strong else 9, mec=c, mew=2.2, mfc=c if excludes(v, ref) else SURFACE, zorder=3)
        if value_labels:
            ax.text(xlim[1], y, " " + f3(v[0]), va="center", ha="left", fontsize=FS_TICK + 0.5, color=INK,
                    fontweight="bold" if strong else "normal")
    for g in groups:
        ax.axhline(n - 1 - g + 0.5, color=GRID, lw=1.0, zorder=1)
    ax.axvline(ref, color=MUTED, lw=1.1, zorder=1)
    ax.set_yticks(range(n))
    labels = [r[0] for r in reversed(rows)]
    ax.set_yticklabels(labels)
    for t in ax.get_yticklabels():
        if t.get_text() in emph:
            t.set_fontweight("bold")
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_xlim(*xlim)
    style_axes(ax)


def legend_fill(fig, y=0.015, x=0.02, ref_word="0"):
    h = [Line2D([], [], marker="o", ls="-", color=BLUE, mfc=BLUE, mec=BLUE, ms=9, lw=2.2,
                label=f"95 % 구간이 {ref_word}을 제외"),
         Line2D([], [], marker="o", ls="-", color=BLUE, mfc=SURFACE, mec=BLUE, mew=2.2, ms=9, lw=2.2,
                label=f"95 % 구간이 {ref_word}을 포함")]
    fig.legend(handles=h, loc="lower left", bbox_to_anchor=(x, y), ncol=2, frameon=False, fontsize=FS_SMALL,
               labelcolor=INK2, handlelength=2.0, columnspacing=1.6)


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=220, facecolor=SURFACE, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    print("written", name)


# ---------------------------------------------------------------- slide 2: missingness
def fig_s2():
    miss = J("ppcf/fig_missing.json")["CES_TI"]["missing_pct"]
    assert f"{miss:.1f}" == "7.7"
    # TYPED from the specification (PROJECT_KNOWLEDGE.md section 2; no JSON holds the BES-quintile shares)
    low_q, high_q = 18.0, 3.4
    rows = [("BES 수준이 낮은\n5분의 1", low_q), ("전체 프레임", miss), ("BES 수준이 높은\n5분의 1", high_q)]
    fig, ax = plt.subplots(figsize=(6.0, 3.9))
    for i, (lab, v) in enumerate(rows):
        y = len(rows) - 1 - i
        ax.barh(y, v, height=0.56, color=BLUE if i != 1 else MUTED, zorder=2)
        ax.text(v + 0.35, y, f"{v:.1f} %", va="center", ha="left", fontsize=FS + 1, color=INK, fontweight="bold")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in reversed(rows)])
    ax.set_xlim(0, 21.5)
    ax.set_xlabel("$T_i$ 값이 없는 프레임의 비율 (%)", fontsize=FS_TICK, color=INK2)
    style_axes(ax)
    ax.spines["left"].set_visible(True)
    ax.spines["left"].set_color(MUTED)
    fig.subplots_adjust(left=0.30, right=0.97, top=0.97, bottom=0.17)
    save(fig, "s2_missing")


# ---------------------------------------------------------------- slide 3: who reads what
def fig_s3():
    cols = ["CES 이력", "BES", "ECEI", "Mirnov", "다음\nCES 값"]
    rows = [("순환 estimator", (1, 1, 1, 1, 0)),
            ("쌍둥이", (1, 0, 0, 0, 0)),
            ("5-parameter filter", (1, 2, 0, 0, 0)),
            ("hold", (3, 0, 0, 0, 0)),
            ("인과 GP", (1, 0, 0, 0, 0)),
            ("offline GP", (1, 0, 0, 0, 1))]
    fig, ax = plt.subplots(figsize=(6.6, 4.9))
    ax.set_xlim(-3.1, 5.0)
    ax.set_ylim(-1.35, 7.75)
    ax.axis("off")
    nrow = len(rows)
    # column bands: 'up to now' vs 'future'
    ax.add_patch(FancyBboxPatch((-0.42, -0.45), 3.84, nrow - 0.1, boxstyle="round,pad=0,rounding_size=0.12",
                                fc="#eef3fa", ec="none", zorder=0))
    ax.add_patch(FancyBboxPatch((3.58, -0.45), 0.84, nrow - 0.1, boxstyle="round,pad=0,rounding_size=0.12",
                                fc="#f3f1ea", ec="none", zorder=0))
    ax.text(1.5, nrow + 1.45, "지금까지 들어온 값", ha="center", va="center", fontsize=FS_SMALL, color=INK2)
    ax.text(4.0, nrow + 1.45, "미래 값", ha="center", va="center", fontsize=FS_SMALL, color=INK2)
    for j, c in enumerate(cols):
        ax.text(j, nrow + 0.1, c, ha="center", va="bottom", fontsize=FS_TICK, color=INK, fontweight="bold")
    for i, (lab, cells) in enumerate(rows):
        y = nrow - 1 - i
        ax.text(-0.75, y, lab, ha="right", va="center", fontsize=FS, color=INK,
                fontweight="bold" if i == 0 else "normal")
        for j, k in enumerate(cells):
            if k == 1:
                ax.plot(j, y, "o", ms=15, color=BLUE, zorder=3)
            elif k == 2:  # BES changes only (two inputs)
                ax.plot(j, y, "o", ms=15, mfc=SURFACE, mec=BLUE, mew=2.4, zorder=3)
            elif k == 3:  # last value only
                ax.plot(j, y, "o", ms=8, color=BLUE, zorder=3)
            else:
                ax.plot(j, y, "o", ms=5, color=GRID, zorder=2)
        if i in (1, 2):
            ax.axhline(y + 0.5, xmin=0.0, xmax=1.0, color=GRID, lw=0.9, zorder=1)
    ax.axhline(nrow - 3 - 0.5, color=GRID, lw=0.9, zorder=1)
    # legend line under the matrix
    ax.plot(-2.9, -0.95, "o", ms=11, color=BLUE)
    ax.text(-2.65, -0.95, "읽음", va="center", fontsize=FS_SMALL, color=INK2)
    ax.plot(-1.55, -0.95, "o", ms=11, mfc=SURFACE, mec=BLUE, mew=2.2)
    ax.text(-1.3, -0.95, "BES 변화 둘", va="center", fontsize=FS_SMALL, color=INK2)
    ax.plot(0.75, -0.95, "o", ms=7, color=BLUE)
    ax.text(0.95, -0.95, "마지막 값만", va="center", fontsize=FS_SMALL, color=INK2)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.02)
    save(fig, "s3_inputs")


# ---------------------------------------------------------------- slide 4: pooled skill
def fig_s4():
    P = J("b39_primary.json")["P"]
    assert P["complete"]
    rows = [("hold", check(P["causal_over"]["persistence"], "+0.375 [+0.352, +0.394]")),
            ("인과 GP", check(P["P1_causal_over_gp"], "+0.089 [+0.064, +0.112]")),
            ("쌍둥이", check(P["P2_causal_over_twin"], "+0.048 [+0.022, +0.073]")),
            ("offline GP", check(P["causal_over"]["gp"], "−0.037 [−0.068, −0.008]"))]
    fig, ax = plt.subplots(figsize=(6.0, 3.9))
    interval_rows(ax, rows, 0.0, (-0.10, 0.42))
    ax.set_xlabel("순환 estimator의 skill (0보다 크면 오차가 더 낮음)", fontsize=FS_TICK, color=INK2)
    ax.text(0.003, -0.55, "기준 0", fontsize=FS_SMALL, color=MUTED, ha="left", va="bottom")
    fig.subplots_adjust(left=0.22, right=0.86, top=0.97, bottom=0.30)
    legend_fill(fig)
    save(fig, "s4_pooled_skill")


# ---------------------------------------------------------------- slide 5: MSE over the floor
def fig_s5():
    A = J("b57_yardsticks.json")["all"]
    assert f"{A['rms'][0]:.1f} [{A['rms'][1]:.1f}, {A['rms'][2]:.1f}]" == "135.7 [126.4, 145.7]"
    check(A["removable_A_over_G"], "+0.298 [+0.191, +0.408]")
    check(A["removable_A_over_Tw"], "+0.202 [+0.105, +0.303]")
    check(A["gap_closure_A"], "0.734 [0.529, 0.936]", signed=False)
    assert f"{A['gap_closure_Tw'][0]:.3f}" == "0.354"
    rows = [("hold", check1(A["mse_over_s2_H"][0], "2.123")),
            ("인과 GP", check1(A["mse_over_s2_G"][0], "1.472")),
            ("쌍둥이", check1(A["mse_over_s2_Tw"][0], "1.415")),
            ("순환 estimator", check1(A["mse_over_s2_A"][0], "1.331")),
            ("offline GP", check1(A["mse_over_s2_O"][0], "1.279"))]
    fig, ax = plt.subplots(figsize=(6.0, 4.1))
    n = len(rows)
    for i, (lab, v) in enumerate(rows):
        y = n - 1 - i
        strong = lab == "순환 estimator"
        ax.barh(y, 1.0, height=0.58, color="#cfcdc4", zorder=2)
        ax.barh(y, v - 1.0, left=1.0, height=0.58, color=BLUE, zorder=2, alpha=1.0 if strong else 0.55)
        ax.text(v + 0.03, y, f"{v:.3f}", va="center", ha="left", fontsize=FS, color=INK,
                fontweight="bold" if strong else "normal")
    ax.axvline(1.0, color=INK2, lw=1.2, zorder=3)
    ax.text(0.5, n - 0.32, "바닥\n(보고값의 산포)", ha="center", va="bottom", fontsize=FS_SMALL, color=INK2)
    ax.text(1.06, n - 0.32, "줄일 수 있는 몫", ha="left", va="bottom", fontsize=FS_SMALL, color=BLUE)
    ax.set_yticks(range(n))
    ax.set_yticklabels([r[0] for r in reversed(rows)])
    for t in ax.get_yticklabels():
        if t.get_text() == "순환 estimator":
            t.set_fontweight("bold")
    ax.set_xlim(0, 2.45)
    ax.set_ylim(-0.6, n + 0.55)
    ax.set_xticks([0, 0.5, 1.0, 1.5, 2.0])
    ax.set_xlabel("평균 제곱 오차 ÷ 바닥", fontsize=FS_TICK, color=INK2)
    style_axes(ax, xgrid=False)
    fig.subplots_adjust(left=0.27, right=0.97, top=0.98, bottom=0.15)
    save(fig, "s5_floor")


# ---------------------------------------------------------------- slide 6: five-parameter filter
def fig_s6():
    C = J("b56_kfx.json")["confirm"]
    rows = [("신경망 − 쌍둥이", check(C["A_over_Tw"], "+0.049 [+0.023, +0.073]")),
            ("filter: BES 넣음 − 안 넣음", check(C["kfx_over_kf0"], "+0.049 [+0.040, +0.059]")),
            ("filter: 반경 차이만 넣음", check(C["kg_over_kf0"], "+0.032 [+0.025, +0.039]"))]
    check(C["median_ratio_kfx_over_hold"], "0.921 [0.906, 0.940]", signed=False)
    check(C["A_over_kfx"], "+0.036 [+0.012, +0.058]")
    m = C["median_abs_eV"]
    med = [("filter", m["kfx"], "41.8"), ("인과 GP", m["G"], "44.1"), ("hold", m["H"], "45.4"),
           ("순환 estimator", m["A"], "45.8")]
    for _, v, s in med:
        assert f"{v:.1f}" == s, (v, s)
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(6.4, 5.4), gridspec_kw={"height_ratios": [3, 3.4], "hspace": 0.75})
    interval_rows(a1, rows, 0.0, (-0.01, 0.085))
    a1.set_xlabel("빠른 채널의 증분 (skill)", fontsize=FS_TICK, color=INK2)
    n = len(med)
    for i, (lab, v, _) in enumerate(med):
        y = n - 1 - i
        a2.barh(y, v, height=0.56, color=BLUE if i == 0 else "#a9c6ea", zorder=2)
        a2.text(v + 0.15, y, f"{v:.1f} eV", va="center", ha="left", fontsize=FS_TICK + 0.5, color=INK,
                fontweight="bold" if i == 0 else "normal")
    a2.set_yticks(range(n))
    a2.set_yticklabels([r[0] for r in reversed(med)])
    a2.set_xlim(36, 50)
    a2.set_ylim(-0.6, n - 0.4)
    a2.set_xlabel("중앙값 절대오차 (eV, 가로축은 36 eV부터)", fontsize=FS_TICK, color=INK2)
    style_axes(a2)
    fig.subplots_adjust(left=0.36, right=0.88, top=0.98, bottom=0.17)
    legend_fill(fig, y=0.0)
    save(fig, "s6_filter")


# ---------------------------------------------------------------- slide 7: which input carries it
def fig_s7():
    D = J("b55_sample_timing.json")
    pd_ = D["partD"]["all_discharges"]
    rows = [("그대로", check(pd_["skill_file_over_twin"], "+0.048 [+0.022, +0.073]")),
            ("Mirnov를 평균값으로", check(pd_["variants"]["MCmean"]["skill_over_twin"], "+0.053 [+0.028, +0.078]")),
            ("Mirnov만 한 프레임 전", check(pd_["variants"]["Gmc"]["skill_over_twin"], "+0.047 [+0.021, +0.072]")),
            ("ECEI만 한 프레임 전", check(pd_["variants"]["Gecei"]["skill_over_twin"], "+0.029 [+0.006, +0.050]")),
            ("BES만 한 프레임 전", check(pd_["variants"]["Gbes"]["skill_over_twin"], "+0.000 [−0.024, +0.022]")),
            ("빠른 채널 전부 한 프레임 전", check(D["partG"]["variants"]["G"]["skill_over_twin"],
                                         "−0.027 [−0.050, −0.008]"))]
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    interval_rows(ax, rows, 0.0, (-0.065, 0.095), emph=("BES만 한 프레임 전",))
    ax.set_xlabel("쌍둥이 대비 증분 (skill; 학습은 그대로, 입력만 바꿈)", fontsize=FS_TICK, color=INK2)
    fig.subplots_adjust(left=0.40, right=0.88, top=0.98, bottom=0.24)
    legend_fill(fig)
    save(fig, "s7_inputs")


# ---------------------------------------------------------------- slide 8: where the gain is
def fig_s8():
    R = J("b54_pooled_rows.json")["reported"]
    rows = [("평범한 행", check(R["plain_rows"]["t"], "+0.092 [+0.062, +0.125]")),
            ("값 규칙에 걸린 행", check(R["flagged_rows"]["t"], "+0.012 [−0.016, +0.036]")),
            ("빔 blip 뒤 1–3행", check(R["blip_rows_1_3"]["t"], "+0.151 [+0.093, +0.215]")),
            ("가장 차가운 3분의 1", check(R["thirds"]["low"]["t"], "+0.007 [−0.023, +0.031]")),
            ("가운데 3분의 1", check(R["thirds"]["middle"]["t"], "+0.042 [+0.009, +0.074]")),
            ("가장 뜨거운 3분의 1", check(R["thirds"]["high"]["t"], "+0.092 [+0.040, +0.137]"))]
    ratios = [("90번째 백분위수", check(R["p90_ratio"]["causal GP"]["value"], "0.946 [0.928, 0.965]", signed=False)),
              ("중앙값", check(R["median_ratio"]["causal GP"]["value"], "1.042 [1.007, 1.082]", signed=False))]
    E = J("b60_relative_error.json")
    cell = E["cells"]["all rows"]["value"]
    check1(cell["c|0.20"][0], "0.770")
    check1(cell["gp|0.20"][0], "0.772")
    check1(cell["hold|0.20"][0], "0.749")
    A1 = E["addendum1"]["cells_r3_r9"]["all rows"]
    check1(A1["value"]["kfx|0.20"][0], "0.783")
    check(A1["diff"]["kfx-gp|0.20"], "+0.011 [+0.008, +0.014]")

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(6.4, 5.6), gridspec_kw={"height_ratios": [6, 2.3], "hspace": 0.62})
    interval_rows(a1, rows, 0.0, (-0.05, 0.24), groups=(2, 3))
    a1.set_xlabel("쌍둥이 대비 증분 (skill)", fontsize=FS_TICK, color=INK2)
    for i, (lab, v) in enumerate(ratios):
        y = 1 - i
        a2.plot([v[1], v[2]], [y, y], color=BLUE, lw=2.2, solid_capstyle="round", zorder=2)
        a2.plot(v[0], y, "o", ms=9, mec=BLUE, mew=2.2, mfc=BLUE if excludes(v, 1.0) else SURFACE, zorder=3)
        a2.text(1.10, y, " " + f3(v[0], False), va="center", ha="left", fontsize=FS_TICK + 0.5, color=INK)
    a2.axvline(1.0, color=MUTED, lw=1.1)
    a2.set_yticks([0, 1])
    a2.set_yticklabels([r[0] for r in reversed(ratios)])
    a2.set_ylim(-0.6, 1.6)
    a2.set_xlim(0.90, 1.10)
    a2.set_xlabel("절대오차의 비 (순환 estimator ÷ 인과 GP)", fontsize=FS_TICK, color=INK2)
    style_axes(a2)
    fig.subplots_adjust(left=0.34, right=0.88, top=0.98, bottom=0.17)
    legend_fill(fig, y=0.0, ref_word="기준값")
    save(fig, "s8_where")


# ---------------------------------------------------------------- slide 9: recovery after a blip
def fig_s9():
    S = J("b58_blip_calibration.json")["S"]["all"]
    assert f"{S['unexplained_ecei']['point']:.3f} [{S['unexplained_ecei']['ci95'][0]:.3f}, " \
           f"{S['unexplained_ecei']['ci95'][1]:.3f}]" == "0.009 [0.005, 0.032]"
    assert f"{S['unexplained_bes']['point']:.3f} [{S['unexplained_bes']['ci95'][0]:.3f}, " \
           f"{S['unexplained_bes']['ci95'][1]:.3f}]" == "0.333 [0.286, 0.376]"
    Y = J("b57_yardsticks.json")["all"]
    check(Y["net_twin_rho1"], "0.810 [0.753, 0.866]", signed=False)
    check1(Y["twin_kf0_rho1"][0], "1.122")
    pr, ci, ctl = S["profiles"], S["profiles_ci95"], S["control_profiles"]
    x = list(range(1, len(pr["ti"]) + 1))
    series = [("bes", "BES 수준", MUTED, "-"), ("ecei", "ECEI 수준", ORANGE, "-"), ("ti", "보고된 $T_i$", BLUE, "-")]
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    ax.axhline(1.0, color=GRID, lw=1.0, zorder=0)
    for k, lab, c, ls in series:
        lo = [p[0] for p in ci[k]]
        hi = [p[1] for p in ci[k]]
        ax.fill_between(x, lo, hi, color=c, alpha=0.18, lw=0, zorder=1)
        ax.plot(x, pr[k], ls, color=c, lw=2.6 if k == "ti" else 2.2, marker="o", ms=6, zorder=3)
        ax.plot(x, ctl[k], ":", color=c, lw=1.4, zorder=2)
    ax.text(1.25, pr["bes"][0], "BES 수준", va="center", ha="left", fontsize=FS, color=INK2)
    ax.text(1.3, pr["ecei"][0] + 0.025, "ECEI 수준", va="bottom", ha="left", fontsize=FS, color="#b8471a")
    ax.text(3.0, 0.835, "보고된 $T_i$", va="top", ha="left", fontsize=FS, color=BLUE, fontweight="bold")
    ax.text(10, 1.045, "점선: 대조(빠진 프레임 없음)", ha="right", va="bottom", fontsize=FS_SMALL, color=INK2)
    ax.set_xticks(x)
    ax.set_xlim(0.6, 10.4)
    ax.set_ylim(0.6, 1.08)
    ax.set_xlabel("값이 빠진 프레임 뒤의 행 (10 ms 프레임)", fontsize=FS_TICK, color=INK2)
    ax.set_ylabel("빠지기 전 여섯 행의 중앙값에 대한 비", fontsize=FS_TICK, color=INK2)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=FS_TICK, colors=INK2, color=GRID)
    ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
    fig.subplots_adjust(left=0.15, right=0.98, top=0.97, bottom=0.14)
    save(fig, "s9_blip_recovery")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for f in (fig_s2, fig_s3, fig_s4, fig_s5, fig_s6, fig_s7, fig_s8, fig_s9):
        f()


if __name__ == "__main__":
    main()
