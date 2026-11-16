# -*- coding: utf-8 -*-
"""포스터용 그림 (docs/presentation/figures/poster/).

  py docs/presentation/make_figures_poster.py

1. 논문 그림(docs/paper/ppcf/fig_*.pdf)을 300 dpi PNG 로 옮긴다 — 원본은 건드리지 않는다.
2. 포스터에만 쓰는 세 장을 새로 그린다. 수치는 전부 기록 파일에서 읽는다(손 전사 없음).
   - poster_skill_kstar.png   KSTAR T_i: 순환 신경망의 skill 과 95 % 구간 (docs/paper/b27c_hier.json)
   - poster_skill_mast.png    MAST pedestal top: 등록 확증의 skill 과 95 % 구간
                              (docs/paper/mast_ped_confirmation.json)
   - poster_blip.png          빔 blip 뒤 T_i · ECEI · BES 의 응답 (docs/paper/file_rules.json)
   - poster_missing.png       격자 행의 측정 · 유지값 · 결측 비율 (docs/paper/ppcf/fig_missing.json)

그림 규칙: 한 축(이중 축 없음), 얇은 선(2 px)과 흰 테를 두른 점, 실선 눈금, 계열 색은 고정 순서
(파랑 2a78d6 · 주황 eb6834 · 청록 1baf7a — 색각 이상 분리 검증 통과), 글자는 검정·회색.
크기는 포스터에 놓이는 실제 크기(인치)로 만들어 글자가 그대로 9 pt 안팎이 되게 한다.
"""
import json
import os
from pathlib import Path

import fitz
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager as fm  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / "figures" / "poster"
PAPER = ROOT / "docs" / "paper"

S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"
INK, SEC, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"

# Pretendard 는 사용자 글꼴 폴더에 있다. 없으면 맑은 고딕.
_user_fonts = Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft" / "Windows" / "Fonts"
for _f in ("Pretendard-Regular.otf", "Pretendard-Medium.otf", "Pretendard-SemiBold.otf", "Pretendard-Bold.otf"):
    if (_user_fonts / _f).exists():
        fm.fontManager.addfont(str(_user_fonts / _f))
_names = {f.name for f in fm.fontManager.ttflist}
FONT = "Pretendard" if "Pretendard" in _names else "Malgun Gothic"
plt.rcParams.update({
    "font.family": FONT, "font.size": 9, "axes.unicode_minus": False, "mathtext.fontset": "dejavusans",
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.labelcolor": SEC, "xtick.color": MUTED,
    "ytick.color": MUTED, "xtick.labelcolor": SEC, "ytick.labelcolor": SEC, "axes.spines.top": False,
    "axes.spines.right": False, "figure.facecolor": "white", "savefig.facecolor": "white",
})


def paper_figures():
    for name in ("fig_trace", "fig_thin", "fig_bursts"):
        doc = fitz.open(PAPER / "ppcf" / f"{name}.pdf")
        doc[0].get_pixmap(dpi=300).save(OUT / f"{name}.png")
        print("copied", name)


def interval_chart(rows, path, xlim, size, xlabel, note=None):
    """rows: (label, sublabel, value, lo, hi). 점 + 95 % 구간, 한 계열 한 색, 0 기준선."""
    fig, ax = plt.subplots(figsize=size)
    n = len(rows)
    span = xlim[1] - xlim[0]
    ax.axvline(0, color=AXIS, lw=0.9, zorder=1)
    for i, (lab, sub, v, lo, hi) in enumerate(rows):
        y = n - 1 - i
        ax.plot([lo, hi], [y, y], color=S1, lw=1.8, solid_capstyle="round", zorder=2)
        ax.plot([v], [y], marker="o", ms=6.8, mfc=S1, mec="white", mew=1.4, zorder=3)
        # name column (left), value column (right): text in ink, never in the series colour
        ax.text(xlim[0] - 0.03 * span, y + 0.14, lab, ha="right", va="center", color=INK, fontsize=9,
                clip_on=False)
        ax.text(xlim[0] - 0.03 * span, y - 0.25, sub, ha="right", va="center", color=MUTED, fontsize=7.4,
                clip_on=False)
        ax.text(xlim[1] + 0.05 * span, y + 0.14, "%+.3f" % v, ha="left", va="center", color=INK, fontsize=9,
                clip_on=False)
        ax.text(xlim[1] + 0.05 * span, y - 0.25, "[%+.3f, %+.3f]" % (lo, hi), ha="left", va="center",
                color=SEC, fontsize=7.4, clip_on=False)
    ax.set_xlim(*xlim)
    ax.set_ylim(-0.65, n - 0.35)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.xaxis.grid(True, color=GRID, lw=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", length=0, labelsize=8)
    ax.set_xlabel(xlabel, fontsize=8.4)
    if note:
        fig.text(0.995, 0.012, note, ha="right", va="bottom", color=MUTED, fontsize=7.2)
    return fig, ax


def skill_kstar():
    d = json.loads((PAPER / "b27c_hier.json").read_text(encoding="utf-8"))

    def g(k):
        c = d[f"cut/CES_TI/{k}"]["primary_r2_r9"]
        return c["median"], c["ci95"][0], c["ci95"][1]

    rows = [("hold", "직전 값 유지", *g("persistence")),
            ("인과 GP", "CES 이력만, 실시간 가능", *g("gp_causal")),
            ("PCHIP 보간", "다음 표본 사용, 실시간 불가", *g("pchip")),
            ("GP 보간", "다음 표본 사용, 실시간 불가", *g("gp"))]
    fig, ax = interval_chart(rows, None, (-0.08, 0.46), (3.62, 2.02), "순환 신경망의 $T_i$ skill")
    ax.set_xticks([0, 0.2, 0.4])
    fig.subplots_adjust(left=0.365, right=0.745, top=0.985, bottom=0.215)
    fig.savefig(OUT / "poster_skill_kstar.png", dpi=400)
    plt.close(fig)
    print("poster_skill_kstar", [(r[0], round(r[2], 3), round(r[3], 3), round(r[4], 3)) for r in rows])


def skill_mast():
    m = json.loads((PAPER / "mast_ped_confirmation.json").read_text(encoding="utf-8"))
    s = m["targets"]["T_i"]["cells"]["pedestal top"]["skill"]
    rows = [("CES 이력만", "hold 대비", *s["gb_ces_vs_hold"]),
            ("모든 입력", "hold 대비", *s["gb_all_vs_hold"]),
            ("Thomson 추가", "CES 이력에 더한 증분", *s["gb_ts_vs_gb_ces"]),
            ("D-alpha 추가", "CES 이력에 더한 증분", *s["gb_da_vs_gb_ces"]),
            ("모든 입력", "offline 보간 대비", *s["gb_all_vs_interp"])]
    fig, ax = interval_chart(rows, None, (-0.32, 0.24), (3.62, 2.36), "$T_i$ skill")
    ax.set_xticks([-0.2, 0, 0.2])
    fig.subplots_adjust(left=0.365, right=0.745, top=0.985, bottom=0.185)
    fig.savefig(OUT / "poster_skill_mast.png", dpi=400)
    plt.close(fig)
    print("poster_skill_mast", [(r[0], r[1], round(r[2], 3), round(r[3], 3), round(r[4], 3)) for r in rows])


def blip():
    r = json.loads((PAPER / "file_rules.json").read_text(encoding="utf-8"))["response"]["rows"]
    ks = sorted(int(k) for k in r)
    x = [10 * k for k in ks]
    fig, ax = plt.subplots(figsize=(3.62, 2.2))
    ax.axvspan(-1.7, 8.8, color="#ededeb", lw=0, zorder=0)
    ax.axhline(1.0, color=AXIS, lw=0.9, zorder=1)
    series = (("bes", "빔 방출 (BES)", S3), ("ecei", "ECEI 세기", S2), ("ti", "CES $T_i$", S1))
    for key, lab, col in series:
        for seg in ([k for k in ks if k < 0], [k for k in ks if k > 0]):
            xs = [10 * k for k in seg]
            med = [r[str(k)][key]["median"] for k in seg]
            lo = [r[str(k)][key]["ci95"][0] for k in seg]
            hi = [r[str(k)][key]["ci95"][1] for k in seg]
            ax.fill_between(xs, lo, hi, color=col, alpha=0.16, lw=0, zorder=2)
            ax.plot(xs, med, color=col, lw=1.7, zorder=3, label=lab if seg[0] < 0 else None,
                    solid_capstyle="round")
        k1 = r["1"][key]["median"]
        ax.plot([10], [k1], marker="o", ms=6.0, mfc=col, mec="white", mew=1.3, zorder=4)
    # direct labels (ink, not the series colour): the value of the first row after the blip,
    # written left of its point, in the empty region before the blip
    for key, name in (("ecei", "ECEI 세기"), ("ti", "CES $T_i$"), ("bes", "빔 방출 (BES)")):
        v = r["1"][key]["median"]
        ax.text(-6, v, "%s  %.3f" % (name, v), color=INK, fontsize=8.2, va="center", ha="right")
    ax.text(3.5, 1.022, "빔 꺼짐\n(빠진 행)", color=SEC, fontsize=7.2, ha="center", va="bottom", linespacing=1.12)
    ax.set_xlim(-62, 152)
    ax.set_ylim(0.61, 1.085)
    ax.set_xticks([-50, 0, 50, 100, 150])
    ax.set_yticks([0.7, 0.8, 0.9, 1.0])
    ax.yaxis.grid(True, color=GRID, lw=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(length=0, labelsize=8)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("빠진 행의 시각으로부터 (ms)", fontsize=8.4)
    ax.set_ylabel("blip 직전 대비 비 (중앙값, 띠 = 95 % 구간)", fontsize=8.0)
    h, l = ax.get_legend_handles_labels()
    order = [2, 1, 0]
    ax.legend([h[i] for i in order], [l[i] for i in order], loc="lower right", bbox_to_anchor=(1.0, 0.16),
              frameon=False, fontsize=7.6, handlelength=1.4, labelcolor=SEC, borderaxespad=0.2)
    fig.subplots_adjust(left=0.155, right=0.965, top=0.975, bottom=0.175)
    fig.savefig(OUT / "poster_blip.png", dpi=400)
    plt.close(fig)
    print("poster_blip rows", ks[0], "..", ks[-1], "font", FONT)


def missing():
    """두 물리량의 격자 행을 측정 · 유지값 · 결측으로 나눈 가로 누적 막대 (한 축, 조각 사이 흰 틈)."""
    m = json.loads((PAPER / "ppcf" / "fig_missing.json").read_text(encoding="utf-8"))
    GREY = "#b4b2aa"
    parts = (("independent_pct", "측정", S1, "white"), ("held_pct_of_grid", "유지값(직전 값의 반복)", S2, "white"),
             ("missing_pct", "결측", GREY, INK))
    fig, ax = plt.subplots(figsize=(7.5, 1.32))
    for row, (key, lab) in enumerate((("CES_TI", "$T_i$"), ("CES_VT", "$V_\\mathrm{rot}$"))):
        y, left = 1 - row, 0.0
        for k, name, col, tc in parts:
            v = m[key][k]
            if v < 0.05:
                continue
            ax.barh(y, v, left=left, height=0.56, color=col, edgecolor="white", linewidth=2.0, zorder=2,
                    label=name if row == 1 else None)
            if v >= 9:
                ax.text(left + v / 2, y, "%.1f %%" % v, ha="center", va="center", color=tc, fontsize=9)
            else:
                ax.text(left + v + 1.0, y, "%.1f %%" % v, ha="left", va="center", color=INK, fontsize=9)
            left += v
        ax.text(-1.5, y, lab, ha="right", va="center", color=INK, fontsize=10.5)
    ax.set_xlim(0, 108)
    ax.set_ylim(-0.45, 1.45)
    ax.set_yticks([])
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.tick_params(length=0, labelsize=8)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("10 ms 격자 행 가운데 비율 (%%), 전체 %s행" % format(m["n_rows"], ","), fontsize=8.4)
    h, l = ax.get_legend_handles_labels()
    ax.legend(h, l, loc="lower center", bbox_to_anchor=(0.46, 0.99), ncol=3, frameon=False, fontsize=8.2,
              handlelength=1.1, columnspacing=1.6, labelcolor=SEC, borderaxespad=0.0)
    fig.subplots_adjust(left=0.06, right=0.995, top=0.83, bottom=0.31)
    fig.savefig(OUT / "poster_missing.png", dpi=400)
    plt.close(fig)
    print("poster_missing", {k: [round(m[k][q], 1) for q, *_ in parts] for k in ("CES_TI", "CES_VT")})


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    paper_figures()
    skill_kstar()
    skill_mast()
    blip()
    missing()
