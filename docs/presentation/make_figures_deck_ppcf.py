# -*- coding: utf-8 -*-
"""Figures of the PowerPoint talk deck that are not the poster's.

    py docs/presentation/make_figures_deck_ppcf.py

Inputs kept out of the public repository (data/deck_inputs/, gitignored):
  provided_abcd_positions.png   the channel-position plot the author received ("ABCD positions, need corrections
                                using syndia"); our channels are marked on it here, nothing else is redrawn
  provided_mirnov_array.png     the KSTAR Mirnov toroidal-array layout the author received; the file's two coils
                                are marked on it here
  channel_positions_abcd.json   R, z of every channel, digitized from the first plot (axis ticks fitted, residual
                                < 0.04 mm); used only to place the marks
  mirnov_toroidal_angles.json   angle of each of the 20 coils, digitized from the second plot
and the KFE overview slides `Current status of the KSTAR diagnostics` (J. Ko, KFE, 28 Nov 2024), page 15, cropped.

Outputs: the three marked or cropped pictures go to data/deck_inputs/ (not committed: they are other people's
figures); the discharge overview, drawn from the 639 data files, goes to figures/deck_ppcf/ (committed).
"""
import csv
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
INP = ROOT / "data" / "deck_inputs"
OUT_FIG = HERE / "figures" / "deck_ppcf"
KFE_PDF = Path(r"C:\Users\lss\Desktop\원핵 졸논\1. 데이터, 서론\서론 참고문헌 정리\실험 장치 관련\Current status of the KSTAR diagnostics.pdf")

OUR_BES = [f"BES_{r:02d}{c:02d}" for r in (1, 2, 3) for c in (5, 6, 7)]
OUR_ECEI = [f"ECEI_GT{v}03" for v in (11, 12, 13, 14)]
FONT = r"C:\Windows\Fonts\malgunbd.ttf"


def font(size):
    return ImageFont.truetype(FONT, size)


def kfe_layout():
    import fitz
    doc = fitz.open(str(KFE_PDF))
    pix = doc[14].get_pixmap(dpi=220)
    im = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    # content between the title bar and the footer band
    im = im.crop((0, int(0.14 * im.height), im.width, int(0.955 * im.height)))
    im.save(INP / "kfe_layout_p15.png")
    print("kfe layout", im.size)


def abcd_marked():
    im = Image.open(INP / "provided_abcd_positions.png").convert("RGB")
    pos = json.load(open(INP / "channel_positions_abcd.json", encoding="utf-8"))
    # axis calibration of the plot (tick pixels -> values), as used for the digitization
    xt, xv = np.array([354.0, 620.0, 887.0, 1418.0, 1683.0, 2216.0]), np.array([2.170, 2.175, 2.180, 2.190, 2.195, 2.205])
    yt, yv = np.array([248.0, 421.0, 593.0, 766.0]), np.array([0.04, 0.02, 0.00, -0.02])
    ax, bx = np.polyfit(xv, xt, 1)
    ay, by = np.polyfit(yv, yt, 1)
    to_px = lambda R, z: (ax * R + bx, ay * z + by)
    d = ImageDraw.Draw(im)
    for name in OUR_BES + OUR_ECEI:
        R, z = pos[name]
        x, y = to_px(R, z)
        col = (200, 30, 30) if name.startswith("BES") else (20, 90, 170)
        d.ellipse((x - 22, y - 22, x + 22, y + 22), outline=col, width=5)
    for R, w, lab in ((2.198, 6, "CES ch18, R = 2.198 m"), (2.192, 2, None), (2.203, 2, None)):
        x, _ = to_px(R, 0)
        for y0 in range(92, 876, 22):
            d.line((x, y0, x, y0 + 12), fill=(20, 20, 20) if lab else (120, 120, 120), width=w)
        if lab:
            d.rectangle((x - 530, 820, x - 12, 868), fill=(255, 255, 255))
            d.text((x - 524, 822), lab, font=font(38), fill=(20, 20, 20))
    im = im.crop((60, 40, im.width, 965))
    im.save(INP / "abcd_marked.png")
    print("abcd marked", im.size)


def mirnov_marked():
    from scipy import ndimage
    im = Image.open(INP / "provided_mirnov_array.png").convert("RGB")
    a = np.asarray(im).astype(int)
    blobs, n = ndimage.label((a[..., 0] > 180) & (a[..., 2] > 180) & (a[..., 1] < 90))
    cs = [np.argwhere(blobs == i).mean(axis=0) for i in range(1, n + 1) if (blobs == i).sum() > 150]
    cy, cx = np.mean(cs, axis=0)
    # number the 20 coil markers clockwise from the top, as the picture labels them
    cs.sort(key=lambda c: (90 - math.degrees(math.atan2(-(c[0] - cy), c[1] - cx))) % 360)
    d = ImageDraw.Draw(im)
    for k, label, (lx, ly) in ((3, "MC1T03", (700, 215)), (16, "MC1T16", (25, 385))):
        y, x = cs[k - 1]
        d.ellipse((x - 26, y - 26, x + 26, y + 26), outline=(255, 40, 40), width=7)
        d.rectangle((lx - 6, ly - 4, lx + 214, ly + 48), fill=(255, 255, 255), outline=(255, 40, 40), width=3)
        d.text((lx, ly), label, font=font(38), fill=(200, 20, 20))
        ex = lx + (0 if lx > x else 214)
        d.line((ex, ly + 22, x + (26 if lx > x else -26) * (1 if lx > x else 1), y), fill=(255, 40, 40), width=4)
    im = im.crop((0, 18, 1015, im.height))
    im.save(INP / "mirnov_marked.png")
    print("mirnov marked", im.size)


def discharge_table():
    regimes = json.load(open(ROOT / "docs" / "paper" / "b66_regimes.json", encoding="utf-8"))["rate_per_discharge"]
    rows = []
    for fname, info in regimes.items():
        with open(ROOT / "data" / fname, encoding="utf-8") as fh:
            rd = list(csv.reader(fh))
        h, body = rd[0], rd[2:-1]  # the first and the last row of a file are not grid rows
        iti, ivt = h.index("CES_TI"), h.index("CES_VT")
        ib = [i for i, c in enumerate(h) if c.startswith("BES")]
        ti = [float(r[iti]) for r in body if r[iti].strip()]
        vt, prev = [], None
        for r in body:
            v = r[ivt].strip()
            if v:
                if v != prev:
                    vt.append(float(v))
                prev = v
        cut = [t for t in ti if t <= 3000.0]
        bes = np.median([np.mean([float(r[i]) for i in ib]) for r in body])
        rows.append(dict(shot=int(fname[1:6]), ti=np.median(cut) if cut else np.nan,
                         vt=np.median(vt) if vt else np.nan, miss=100.0 * (1 - len(ti) / len(body)),
                         bes=float(bes), rate=info.get("rate_per_s"), cls=info.get("class")))
    return rows


def overview(rows):
    plt.rcParams.update({"font.family": ["Malgun Gothic", "DejaVu Sans"], "axes.unicode_minus": False,
                         "font.size": 13})
    shot = np.array([r["shot"] for r in rows]); ti = np.array([r["ti"] for r in rows])
    vt = np.array([r["vt"] for r in rows]); miss = np.array([r["miss"] for r in rows])
    bes = np.array([r["bes"] for r in rows])
    rate = np.array([np.nan if r["rate"] is None else r["rate"] for r in rows], dtype=float)
    test = shot >= 32312  # the last 97 discharges: the temporal test block
    cmap = plt.get_cmap("plasma")
    vmax = 2.0
    fig = plt.figure(figsize=(17.0, 5.6), dpi=200)
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 0.62], wspace=0.32)
    panels = [(shot, ti, "방전 번호", "Ti 중앙값 (eV)"), (vt, ti, "Vrot 중앙값 (km/s)", "Ti 중앙값 (eV)"),
              (bes, miss, "BES 수준 중앙값 (파일 값)", "Ti 결측 비율 (%)")]
    for k, (x, y, xl, yl) in enumerate(panels):
        ax = fig.add_subplot(gs[0, k])
        for sel, mk in ((~test, "o"), (test, "s")):
            und = sel & np.isnan(rate)
            ok = sel & ~np.isnan(rate)
            cols = cmap(np.clip(rate[ok], 0, vmax) / vmax)
            if mk == "o":
                ax.scatter(x[ok], y[ok], color=cols, s=26, marker=mk, linewidths=0, alpha=0.9)
            else:
                ax.scatter(x[ok], y[ok], facecolors="none", edgecolors=cols, s=30, marker=mk, linewidths=1.3)
            ax.scatter(x[und], y[und], s=26, marker=mk, facecolors="#b8b8b8" if mk == "o" else "none",
                       edgecolors="#8a8a8a", linewidths=1.1)
        if k == 1:
            out = int(np.sum((x < -100) | (x > 250)))
            ax.set_xlim(-100, 250)
            ax.text(0.97, 0.03, f"축 밖 {out}개", transform=ax.transAxes, ha="right", va="bottom", fontsize=11, color="#444444")
        ax.grid(True, color="#d9d9d9", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        if k == 0:
            ax.axvline(32312, color="#444444", linestyle="--", linewidth=1.0)
            ax.text(32300, np.nanmax(ti) * 0.98, "시간순 시험 블록 →", ha="right", va="top", fontsize=11, color="#444444")
    lax = fig.add_subplot(gs[0, 3])
    lax.axis("off")
    handles = [Line2D([], [], marker="o", linestyle="", color=cmap(0.55), markersize=8, label="앞쪽 방전 (학습·검증)"),
               Line2D([], [], marker="s", linestyle="", markerfacecolor="none", markeredgecolor=cmap(0.55),
                      markeredgewidth=1.4, markersize=8, label="마지막 97개 (시험 블록)"),
               Line2D([], [], marker="o", linestyle="", color="#b8b8b8", markersize=8, label="버스트 빈도 판정 불가")]
    lax.legend(handles=handles, loc="upper left", frameon=False, fontsize=12, title="<방전 639개>", title_fontsize=12)
    cax = lax.inset_axes([0.08, 0.06, 0.12, 0.5])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(0, vmax), cmap=cmap), cax=cax)
    cb.set_label("BES 버스트 빈도 (/s)", fontsize=12)
    OUT_FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG / "dataset_overview_ko.png", bbox_inches="tight", facecolor="white")
    print("overview", len(rows), "discharges,", int(test.sum()), "in the test block")


if __name__ == "__main__":
    kfe_layout()
    abcd_marked()
    mirnov_marked()
    overview(discharge_table())
