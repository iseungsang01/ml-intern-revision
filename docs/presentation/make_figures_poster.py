# -*- coding: utf-8 -*-
"""포스터용 그림 (docs/presentation/figures/poster_20261119/).

  py docs/presentation/make_figures_poster.py

슬라이드 그림(figs_20261119/s2_missing.png, p2_inclusive.png)과 같은 내용을 포스터에 놓이는 실제 크기로
다시 그린다. 슬라이드용을 줄여 넣으면 그림 안 글자가 6–8 pt 로 작아지기 때문이다. 그림 안 글자는 10 pt 이상,
글꼴은 포스터와 같은 Pretendard, 바탕은 흰색. 크기를 정확히 지키려고 bbox_inches 를 쓰지 않는다.

수치는 기록 파일에서 읽고, 포스터 명세에 적힌 값과 같은지 단언한다(손 전사 없음).
  - poster_missing.png    BES 수준 5분위별 T_i 결측 비율: 전체는 docs/paper/ppcf/fig_missing.json,
                          5분위 값은 PROJECT_KNOWLEDGE.md 2절의 문장에서 읽는다(그 값을 담은 JSON 은 없다).
  - poster_inclusive.png  3 keV 컷 모집단과 포함 모집단의 신경망 skill: docs/paper/b64_two_stage.json,
                          docs/paper/b65_inclusive.json
"""
import json
import os
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager as fm  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PAPER = ROOT / "docs" / "paper"
OUT = HERE / "figures" / "poster_20261119"
CM = 1 / 2.54
DPI = 300

BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#8a8a86"
INK, INK2, MUTED, GRID = "#15181e", "#3d424b", "#6b7280", "#e3e5e8"
FS = 10.5          # 그림 안 글자 (포스터 위 실제 크기)
FS_TICK = 10.0

_user_fonts = Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft" / "Windows" / "Fonts"
for _f in ("Pretendard-Regular.otf", "Pretendard-SemiBold.otf", "Pretendard-Bold.otf"):
    if (_user_fonts / _f).exists():
        fm.fontManager.addfont(str(_user_fonts / _f))
FONT = "Pretendard" if "Pretendard" in {f.name for f in fm.fontManager.ttflist} else "Malgun Gothic"
plt.rcParams.update({
    "font.family": FONT, "font.size": FS, "axes.unicode_minus": True, "mathtext.fontset": "custom",
    "mathtext.it": f"{FONT}:italic", "mathtext.rm": FONT, "axes.edgecolor": GRID, "axes.linewidth": 0.8,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK,
    "xtick.labelsize": FS_TICK, "ytick.labelsize": FS, "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
})


def J(name):
    return json.loads((PAPER / name).read_text(encoding="utf-8"))


def iv(v):
    return f"{v[0]:+.3f} [{v[1]:+.3f}, {v[2]:+.3f}]".replace("-", "−")


def check(v, spec):
    assert iv(v) == spec, (iv(v), spec)
    return v


def fig_missing():
    total = J("ppcf/fig_missing.json")["CES_TI"]["missing_pct"]
    pk = (ROOT / "PROJECT_KNOWLEDGE.md").read_text(encoding="utf-8")
    m = re.search(r"최하위 오분위에서 T_i 결측 ([\d.]+) %.*?최상위 ([\d.]+) %", pk)
    low, high = float(m.group(1)), float(m.group(2))
    assert ("%.1f %.1f %.1f" % (low, total, high)) == "18.0 7.7 3.4"
    rows = [("BES 수준이 낮은\n5분의 1", low, BLUE), ("전체 프레임", total, GREY), ("BES 수준이 높은\n5분의 1", high, BLUE)]
    fig = plt.figure(figsize=(8.8 * CM, 5.6 * CM))
    ax = fig.add_axes((0.315, 0.228, 0.655, 0.752))
    for i, (lab, v, c) in enumerate(rows):
        y = len(rows) - 1 - i
        ax.barh(y, v, height=0.58, color=c, zorder=2)
        ax.text(v + 0.4, y, f"{v:.1f} %", va="center", ha="left", fontsize=FS + 0.5, color=INK, fontweight="bold")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in reversed(rows)], linespacing=1.15)
    ax.tick_params(axis="y", length=0, pad=5)
    ax.set_xlim(0, 22.5)
    ax.set_xticks([0, 5, 10, 15, 20])
    ax.grid(axis="x", color=GRID, lw=0.8, zorder=0)
    ax.spines["left"].set_color(MUTED)
    ax.spines["bottom"].set_color(GRID)
    ax.set_xlabel("$T_i$ 값이 없는 프레임의 비율 (%)", fontsize=FS_TICK, color=INK2, labelpad=3)
    fig.savefig(OUT / "poster_missing.png", dpi=DPI)
    plt.close(fig)
    print("written poster_missing.png")


def fig_inclusive():
    a = J("b64_two_stage.json")["all_rows"]
    st = J("b65_inclusive.json")["statements"]
    rows = [("인과 GP 대비", [check(a["skill_mse|c|gp"], "+0.092 [+0.068, +0.114]"),
                            check(st["I1"]["value"], "+0.109 [+0.070, +0.146]")]),
            ("쌍둥이 대비\n(빠른 채널 증분)", [check(a["skill_mse|c|t"], "+0.053 [+0.028, +0.076]"),
                                       check(st["I2"]["value"], "−0.002 [−0.017, +0.010]")])]
    colors = (BLUE, ORANGE)
    fig = plt.figure(figsize=(8.6 * CM, 5.1 * CM))
    ax = fig.add_axes((0.31, 0.255, 0.665, 0.585))
    for i, (_, vals) in enumerate(rows):
        y0 = len(rows) - 1 - i
        for j, v in enumerate(vals):
            y = y0 + (0.2 if j == 0 else -0.2)
            excl = v[1] > 0 or v[2] < 0
            ax.plot([v[1], v[2]], [y, y], color=colors[j], lw=2.0, solid_capstyle="butt", zorder=2)
            ax.plot(v[0], y, "o", ms=6.5, color=colors[j], mfc=colors[j] if excl else "white", mew=1.6, zorder=3)
            ax.text(v[2] + 0.008, y, f"{v[0]:+.3f}".replace("-", "−"), va="center", ha="left", fontsize=FS_TICK,
                    color=colors[j])
    ax.axvline(0, color=MUTED, lw=0.9, zorder=1)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows][::-1], linespacing=1.15)
    ax.tick_params(axis="y", length=0, pad=5)
    ax.set_xlim(-0.04, 0.215)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xticks([0.0, 0.05, 0.10, 0.15, 0.20])
    ax.set_xticklabels(["0", "0.05", "0.10", "0.15", "0.20"])
    ax.grid(axis="x", color=GRID, lw=0.8, zorder=0)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.set_xlabel("skill (평균 제곱 오차), 95 % 구간", fontsize=FS_TICK, color=INK2, labelpad=3)
    x = 0.04
    for j, nm in enumerate(("3 keV 컷 모집단", "3 keV 넘는 값 포함")):
        t = fig.text(x, 0.925, "● " + nm, color=colors[j], fontsize=FS, ha="left", va="center")
        fig.canvas.draw()
        x = fig.transFigure.inverted().transform(t.get_window_extent())[1][0] + 0.06
    fig.savefig(OUT / "poster_inclusive.png", dpi=DPI)
    plt.close(fig)
    print("written poster_inclusive.png")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    fig_missing()
    fig_inclusive()
