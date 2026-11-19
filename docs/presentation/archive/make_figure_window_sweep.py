# -*- coding: utf-8 -*-
"""Window-sweep figure: history length (W-1) vs held-out test skill_vs_pchip.

Answers the "why window=4?" feedback with one curve per target: per-seed points
(4 seeds), the seed mean, the PCHIP-parity baseline y=0, and the adopted W=4
marked. history-0 is the no_history ablation at W=4 (dataset requires W >= 2),
footnoted on the axis.

Two sweeps exist and they are NOT interchangeable:
  --variant hf    (default) held-free training, artifacts data/.wsweep_hf_*  -> the result
  --variant kept            held-kept training, artifacts data/.wsweep_*     -> reference only

The held-kept sweep trained with forward-filled ("held") CES values in the targets and
history channels, which inflates the apparent value of a long window for CES_VT -- held
values are copies of the previous reading, so more window = more copies. It is kept only
to document that distortion (THESIS_RESULTS.md §8f).

Data source per variant: the batch summary JSON if present, else the run artifacts
directly (real runs only -- missing points are simply not drawn).
Output: figures/fig_window_sweep.png (hf) / fig_window_sweep_heldkept.png (kept).
"""
import argparse
import json
import os
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

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
ORANGE = "#e8743b"
GREEN = "#2e9e5b"
RED = "#c0392b"
GRAY = "#7b8794"
LGRAY = "#dfe4ea"
BG = "#ffffff"

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data"
OUT = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(OUT, exist_ok=True)

SEEDS = (42, 1, 7, 123)
# W=4 is the value train.py has defaulted to, with no recorded rationale -- it is the
# thing this sweep exists to re-decide, not a defended choice. W=2 is what the curve
# selects (both targets at plateau, smallest window).
DEFAULT_HIST = 3   # W = 4, the incumbent default
SELECTED_HIST = 1  # W = 2, what the sweep selects


def _load_run(out_dir):
    try:
        cm = json.loads((out_dir / "comparison_metrics.json").read_text(encoding="utf-8"))
        bs = json.loads((out_dir / "bootstrap_summary.json").read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    rec = {}
    for t in ("CES_TI", "CES_VT"):
        rec[f"skill_{t[-2:]}"] = cm["per_target"][t]["skill_vs_pchip"]
        rec[f"pass_{t[-2:]}"] = bool(bs["splits"]["test"][t]["pchip"]["pass"])
    return rec


VARIANTS = {
    # variant -> (artifact prefix, output figure, title suffix)
    "hf": ("wsweep_hf", "fig_window_sweep.png", ""),
    "kept": ("wsweep", "fig_window_sweep_heldkept.png", "  [참고: held 포함 학습]"),
}


def load_points(prefix):
    """{history_len: {seed: rec}} from the summary JSON or the run dirs."""
    points = {}
    summary_path = DATA / f".{prefix}_summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        for r in summary["records"]:
            if r.get("status") == "ok" and "skill_vs_pchip_TI" in r:
                points.setdefault(r["history_len"], {})[r["seed"]] = {
                    "skill_TI": r["skill_vs_pchip_TI"], "skill_VT": r["skill_vs_pchip_VT"],
                    "pass_TI": bool(r.get("pchip_pass_TI")), "pass_VT": bool(r.get("pchip_pass_VT")),
                }
        return points
    for d in sorted(DATA.glob(f".{prefix}_*")):
        m = re.fullmatch(rf"\.{re.escape(prefix)}_(?:w(\d+)|h0)_s(\d+)", d.name)
        if not m:
            continue
        hist_len = int(m.group(1)) - 1 if m.group(1) else 0
        rec = _load_run(d)
        if rec:
            points.setdefault(hist_len, {})[int(m.group(2))] = rec
    return points


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=sorted(VARIANTS), default="hf",
                    help="hf = held-free (the result), kept = held-kept (reference only)")
    args = ap.parse_args()
    prefix, out_name, title_suffix = VARIANTS[args.variant]

    points = load_points(prefix)
    if not points:
        raise SystemExit(f"no {args.variant} window-sweep runs found "
                         f"(looked for data/.{prefix}_*) -- run run_window_sweep.py first")
    hist_lens = sorted(points)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), sharex=True)
    panels = [("TI", "CES_TI  (이온 온도)", axes[0]), ("VT", "CES_VT  (토로이달 회전)", axes[1])]
    for key, title, ax in panels:
        per_point = {h: [points[h][s][f"skill_{key}"] for s in SEEDS if s in points[h]]
                     for h in hist_lens}
        means = {h: float(np.mean(v)) for h, v in per_point.items()}

        # Scale to the plateau (history >= 1). The history-0 ablation can collapse far
        # below it (CES_VT ~ -0.85) and would otherwise flatten the whole curve into a
        # line; it is drawn clipped at the axis floor with its true value labelled.
        plateau = [v for h in hist_lens if h > 0 for v in per_point[h]]
        lo, hi = min(plateau), max(plateau)
        pad = 0.18 * (hi - lo)
        ylo, yhi = min(lo - pad, -0.06), hi + pad
        ax.set_ylim(ylo, yhi)

        for h in hist_lens:
            vals = np.array(per_point[h])
            inside = vals >= ylo
            ax.scatter(np.full(inside.sum(), h), vals[inside], s=34, color=BLUE,
                       alpha=0.35, edgecolor="none", zorder=3)
            if (~inside).any():  # clipped seeds -> floor markers
                ax.scatter(np.full((~inside).sum(), h), np.full((~inside).sum(), ylo),
                           s=42, color=BLUE, alpha=0.35, marker="v", edgecolor="none",
                           zorder=3, clip_on=False)
            n_pass = sum(points[h][s][f"pass_{key}"] for s in SEEDS if s in points[h])
            n = len(vals)
            ax.annotate(f"{n_pass}/{n}", (h, 1.02), xycoords=("data", "axes fraction"),
                        ha="center", fontsize=8.5,
                        color=GREEN if n_pass == n else GRAY)

        my = np.array([means[h] for h in hist_lens])
        drawn = np.clip(my, ylo, None)
        ax.plot(hist_lens, drawn, color=BLUE, lw=2, marker="o", markersize=7,
                markeredgecolor="white", markeredgewidth=1.2, zorder=4, label="4-seed 평균")
        for h, m in means.items():
            if m < ylo:  # mean itself off-scale: label the true value at the floor
                ax.annotate(f"{m:+.2f}", (h, ylo), xytext=(0, 10),
                            textcoords="offset points", ha="center", fontsize=9.5,
                            color=BLUE, fontweight="bold")

        if SELECTED_HIST in hist_lens:
            m = means[SELECTED_HIST]
            ax.scatter([SELECTED_HIST], [m], s=170, facecolor="none",
                       edgecolor=GREEN, linewidth=2.6, zorder=5)
            ax.annotate("선택 W=2", (SELECTED_HIST, m), xytext=(0, 16),
                        textcoords="offset points", ha="center",
                        fontsize=10.5, color=GREEN, fontweight="bold")
        if DEFAULT_HIST in hist_lens:
            m = means[DEFAULT_HIST]
            ax.scatter([DEFAULT_HIST], [m], s=150, facecolor="none",
                       edgecolor=GRAY, linewidth=2.0, linestyle="--", zorder=5)
            ax.annotate("기존 기본값 W=4", (DEFAULT_HIST, m), xytext=(0, -22),
                        textcoords="offset points", ha="center",
                        fontsize=10, color=GRAY)
        ax.axhline(0, color=RED, lw=1.4, ls="--", zorder=1)
        ax.text(hist_lens[-1], 0.006, "PCHIP 동률", va="bottom", ha="right",
                fontsize=9, color=RED)
        ax.set_xticks(hist_lens)
        ax.set_xticklabels(["0†" if h == 0 else str(h) for h in hist_lens], fontsize=10.5)
        ax.set_xlabel("history 관측 수 (= window W - 1)", fontsize=10.5)
        ax.set_title(title, fontsize=13, fontweight="bold", color=NAVY, pad=18)
        ax.grid(axis="y", color=LGRAY, lw=0.8)
        for s in ["top", "right"]:
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("test skill_vs_pchip  (= 1 - MSE_model / MSE_pchip)", fontsize=10.5)
    axes[0].legend(loc="lower right", fontsize=9.5, frameon=False)
    fig.suptitle("Window sweep — held-out test skill vs history 길이 "
                 "(iter009 모델, seed 4개, 상단: bootstrap PASS 수)" + title_suffix,
                 fontsize=13.5, fontweight="bold",
                 color=GRAY if args.variant == "kept" else NAVY, y=1.06)
    treatment = ("held 포함 학습 — 참고용 대조군(본 결과 아님)" if args.variant == "kept"
                 else "held-free 학습·평가 (CES_DROP_STUCK_TARGETS=1)")
    fig.text(0.01, -0.04,
             "† history-0 = W=4에서 no_history ablation (dataset은 W≥2 필요). 각 점은 동일 프로토콜의 독립 run: "
             f"iter009, 10 epochs, shot별 캡 500, train 캡 200k, seed별 동일 test shot 96개, {treatment}.\n"
             "   y축은 history≥1 구간(plateau)에 맞춤 — 축 아래로 벗어난 값은 바닥에 ▽로 표시하고 평균값을 함께 적었다.",
             fontsize=8.5, color=GRAY)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, out_name), bbox_inches="tight", facecolor=BG)
    print(f"wrote {out_name} ({args.variant}) with points:",
          {h: sorted(points[h]) for h in hist_lens})


if __name__ == "__main__":
    main()
