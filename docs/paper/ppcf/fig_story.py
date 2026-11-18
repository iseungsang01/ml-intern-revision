"""One-page summary of the statements of the main text: every comparison on the four splits, with its
95 % interval and whether the interval excludes the reference.  For talks and for the overview of the paper.

Reads docs/paper/b39_primary.json (B.39 arm A) and docs/paper/b49_average.json (B.49); nothing is typed by hand.

    py docs/paper/ppcf/fig_story.py              # Korean labels  -> docs/paper/story_fig_ko.{png,pdf}
    py docs/paper/ppcf/fig_story.py --lang en    # English labels -> docs/paper/story_fig_en.{png,pdf}

Also writes docs/paper/story_fig.json (the plotted numbers) and docs/paper/story_table.md (the same as a table).
Encoding: one hue; a filled marker means that the interval excludes the reference, a hollow one that it
includes it (fill, not colour, carries the verdict).  The four markers of a row are the four splits, top to bottom.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

HERE = Path(__file__).resolve().parent
PAPER = HERE.parent
SEEDS = ("42", "1", "7", "123")
BLUE, INK, INK2, MUTED, GRID, SURFACE = "#2a78d6", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#ffffff"

T = {
    "ko": {
        "font": "Malgun Gothic",
        "a": "(a) CES 이력만 쓰는 추정과 견줌",
        "b": "(b) 어느 진단이 증분을 나르는가 (초기화 하나)",
        "c": "(c) 이득은 오차 분포의 어디에 있는가 (초기화 셋 평균)",
        "xs": "평균 제곱 오차 skill (0보다 크면 오차가 더 낮음)",
        "xr": "절대오차의 비: 초기화 셋 평균 / causal GP (1보다 작으면 오차가 더 낮음)",
        "S1": "학습 한 번 대 causal GP\n(등록 진술 S1)", "S2": "학습 한 번 대 쌍둥이\n(등록 진술 S2)",
        "E1": "초기화 셋 평균 대 causal GP\n(E1, S1·S2 실패 뒤 등록)", "E2": "초기화 셋 평균 대 쌍둥이 평균\n(E2, 같음)",
        "D2": "BES만 읽음 대 쌍둥이", "D1": "ECEI만 읽음 대 쌍둥이", "NM": "Mirnov 제외 대 쌍둥이",
        "FB": "모든 빠른 채널 대 BES만 읽음",
        "med": "중앙값의 비", "p90": "90번째 백분위수의 비",
        "in": "구간이 기준값을 제외", "out": "구간이 기준값을 포함",
        "note": "한 행의 점 넷은 위에서부터 분할 1–4, 가로선은 95 % 구간",
        "hold": "학습 한 번 대 hold", "Ehold": "초기화 셋 평균 대 hold",
        "th": ("비교", "분할 1", "분할 2", "분할 3", "분할 4", "기준값 제외"),
        "title": "# 주 결과 요약표 (자동 생성)",
        "lead": ("`docs/paper/ppcf/fig_story.py`가 `docs/paper/b39_primary.json`과 `docs/paper/b49_average.json`에서 만든다. "
                 "손으로 고치지 않는다. 그림은 `docs/paper/story_fig_ko.png`. skill은 1 − MSE(A)/MSE(R), 대괄호는 95 % 구간. "
                 "T_i, cut 모집단, 네 가지 파일 단위 분할."),
        "skill": "## 평균 제곱 오차 skill (기준값 0)", "ratio": "## 절대오차의 비 (기준값 1)",
    },
    "en": {
        "font": "DejaVu Sans",
        "a": "(a) Against estimators that read the CES history only",
        "b": "(b) Which diagnostic carries the increment (one initialisation)",
        "c": "(c) Where in the error distribution the gain is (average of three initialisations)",
        "xs": "mean-squared-error skill (above 0: lower error)",
        "xr": "ratio of absolute errors: average of three initialisations / causal GP (below 1: lower error)",
        "S1": "one training over the causal GP\n(registered S1)", "S2": "one training over its twin\n(registered S2)",
        "E1": "average of three over the causal GP\n(E1, registered after S1 and S2 failed)",
        "E2": "average of three over the twin average\n(E2, the same)",
        "D2": "BES only over the twin", "D1": "ECEI only over the twin", "NM": "no Mirnov over the twin",
        "FB": "all fast channels over BES only",
        "med": "ratio of the medians", "p90": "ratio of the 90th percentiles",
        "in": "interval excludes the reference", "out": "interval includes the reference",
        "note": "the four markers of a row are splits 1 to 4, top to bottom; bars are 95 % intervals",
        "hold": "one training over hold", "Ehold": "average of three over hold",
        "th": ("Comparison", "Split 1", "Split 2", "Split 3", "Split 4", "Reference excluded"),
        "title": "# Summary table of the main results (generated)",
        "lead": ("Written by `docs/paper/ppcf/fig_story.py` from `docs/paper/b39_primary.json` and "
                 "`docs/paper/b49_average.json`; not edited by hand. Figure: `docs/paper/story_fig_en.png`. "
                 "Skill is 1 - MSE(A)/MSE(R); brackets are 95 % intervals. T_i, cut population, four file-level splits."),
        "skill": "## Mean-squared-error skill (reference 0)", "ratio": "## Ratio of absolute errors (reference 1)",
    },
}


def load():
    A = json.loads((PAPER / "b39_primary.json").read_text(encoding="utf-8"))["A"]
    AV = json.loads((PAPER / "b49_average.json").read_text(encoding="utf-8"))["splits"]
    a, av = [A[s] for s in SEEDS], [AV[s] for s in SEEDS]
    assert all(x["three_initialisations"] for x in a)
    return {
        "hold": [x["causal_over"]["persistence"] for x in a],
        "S1": [x["S1_causal_over_gp"] for x in a], "S2": [x["S2_causal_over_twin"] for x in a],
        "Ehold": [x["E_over"]["persistence"] for x in av],
        "E1": [x["E1_E_over_causal_gp"] for x in av], "E2": [x["E2_E_over_twin_average"] for x in av],
        "D2": [x["D2_bes_over_twin"] for x in a], "D1": [x["D1_ecei_over_twin"] for x in a],
        "NM": [x["nomirnov_over_twin"] for x in a], "FB": [x["full_over_bes"] for x in a],
        "med": [x["ratio_E_to_causal_gp"]["median"] for x in av], "p90": [x["ratio_E_to_causal_gp"]["p90"] for x in av],
    }


def excludes(v, ref):
    return v[1] > ref or v[2] < ref


def panel(ax, D, keys, t, ref, xlim):
    off = (0.27, 0.09, -0.09, -0.27)
    for i, k in enumerate(keys):
        y0 = len(keys) - 1 - i
        for v, o in zip(D[k], off):
            ax.plot([v[1], v[2]], [y0 + o, y0 + o], color=BLUE, lw=1.6, solid_capstyle="round", zorder=2)
            ax.plot(v[0], y0 + o, "o", ms=6.5, mec=BLUE, mew=1.6, mfc=BLUE if excludes(v, ref) else SURFACE, zorder=3)
        n = sum(excludes(v, ref) for v in D[k])
        ax.text(xlim[1], y0, f"  {n}/4", va="center", ha="left", fontsize=10.5, color=INK, fontweight="bold")
        if i:
            ax.axhline(y0 + 0.5, color=GRID, lw=0.8, zorder=1)
    ax.axvline(ref, color=MUTED, lw=1.0, zorder=1)
    ax.set_yticks(range(len(keys)))
    ax.set_yticklabels([t[k] for k in reversed(keys)], fontsize=9.5, color=INK)
    ax.set_ylim(-0.55, len(keys) - 0.45)
    ax.set_xlim(*xlim)
    ax.tick_params(axis="x", labelsize=9, colors=INK2, length=3, color=GRID)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color=GRID, lw=0.6, zorder=0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)


def fmt(v, signed=True):
    f = (lambda x: f"{x:+.3f}".replace("-", "−")) if signed else (lambda x: f"{x:.3f}")
    return f"{f(v[0])} [{f(v[1])}, {f(v[2])}]"


def table(D, t):
    L = [t["title"], "", t["lead"], "", t["skill"], "",
         "| " + " | ".join(t["th"]) + " |", "|---|---|---|---|---|---|"]
    for k in ("hold", "S1", "S2", "Ehold", "E1", "E2", "D2", "D1", "NM", "FB"):
        L.append("| " + t[k].replace("\n", " ") + " | " + " | ".join(fmt(v) for v in D[k])
                 + f" | {sum(excludes(v, 0) for v in D[k])}/4 |")
    L += ["", t["ratio"], "", "| " + " | ".join(t["th"]) + " |", "|---|---|---|---|---|---|"]
    for k in ("med", "p90"):
        L.append("| " + t[k] + " | " + " | ".join(fmt(v, False) for v in D[k])
                 + f" | {sum(excludes(v, 1) for v in D[k])}/4 |")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", default="ko", choices=("ko", "en"))
    lang = ap.parse_args().lang
    t, D = T[lang], load()
    plt.rcParams.update({"font.family": t["font"], "axes.unicode_minus": False, "pdf.fonttype": 42})
    fig, axes = plt.subplots(3, 1, figsize=(7.6, 7.4), gridspec_kw={"height_ratios": [4, 4, 2.2], "hspace": 0.62})
    fig.patch.set_facecolor(SURFACE)
    panel(axes[0], D, ("S1", "S2", "E1", "E2"), t, 0.0, (-0.10, 0.22))
    panel(axes[1], D, ("D2", "D1", "NM", "FB"), t, 0.0, (-0.10, 0.22))
    panel(axes[2], D, ("med", "p90"), t, 1.0, (0.85, 1.20))
    for ax, k, xl in zip(axes, "abc", (t["xs"], t["xs"], t["xr"])):
        ax.set_title(t[k], loc="left", fontsize=11, color=INK, fontweight="bold", pad=7)
        ax.set_xlabel(xl, fontsize=9, color=INK2)
        ax.set_facecolor(SURFACE)
    h = [Line2D([], [], marker="o", ls="-", color=BLUE, mfc=BLUE, mec=BLUE, ms=6.5, lw=1.6, label=t["in"]),
         Line2D([], [], marker="o", ls="-", color=BLUE, mfc=SURFACE, mec=BLUE, mew=1.6, ms=6.5, lw=1.6, label=t["out"])]
    fig.legend(handles=h, loc="upper right", bbox_to_anchor=(0.985, 0.995), ncol=2, frameon=False, fontsize=9,
               labelcolor=INK2, handlelength=2.2, columnspacing=1.4)
    fig.text(0.012, 0.012, t["note"], fontsize=8.5, color=INK2, ha="left", va="bottom")
    fig.subplots_adjust(left=0.36, right=0.93, top=0.93, bottom=0.09)
    for ext in ("png", "pdf"):
        fig.savefig(PAPER / f"story_fig_{lang}.{ext}", dpi=220, facecolor=SURFACE)
    (PAPER / "story_fig.json").write_text(json.dumps({"splits": SEEDS, "values": D}, indent=1), encoding="utf-8")
    if lang == "ko":
        (PAPER / "story_table.md").write_text(table(D, t), encoding="utf-8", newline="\n")
    print("written story_fig_%s.png/pdf" % lang)


if __name__ == "__main__":
    main()
