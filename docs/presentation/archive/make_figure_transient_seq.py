# -*- coding: utf-8 -*-
"""Transient/event figure for the talk -- adopted backbone (seq_v2) edition.

Same story as make_figure_transient.py (MC burst -> BES/ECEI collapse -> CES crash, with
the truth overlaid by offline PCHIP vs the causal nowcast), but the nowcaster is the
adopted **seq_v2 backbone** (THESIS_RESULTS.md §8x/§8ab) run under the confirmed
protocol: full-grid causal sequence, held-free, spike-cut population
(CES_TI_SPIKE_CUT_EV = 3000), per-shot input standardization (the backbone's definition).

Only held-out shots of the backbone's split may be plotted (the script refuses training
shots). Defaults: the B.1 backbone for split 42 (`data/.b1_seqv2_s42_i42`), whose split
manifest is `data/.b1_w2cut_split_s42`; #31815 is a TEST shot of that split.

Run from the repo root::

    py docs/presentation/make_figure_transient_seq.py            # default shots
    py docs/presentation/make_figure_transient_seq.py 31815 30842

Env: CES_OUTPUT_DIR (backbone run dir), CES_SPLIT_DIR (split manifest dir),
     CES_DATA_DIR (shot CSVs), CES_TI_SPIKE_CUT_EV (default 3000).
Writes: docs/presentation/figures/fig_transient_seq_<shot>.png
"""
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ces_prediction"))
sys.path.insert(0, str(ROOT / "ces_prediction" / "experiments" / "seq"))
import baselines_interpolation as B  # noqa: E402
from evaluate import _load_stats  # noqa: E402
from seq_data import load_grid_files, build_blocks  # noqa: E402
from model_seq_v2 import SeqCESLSTMv2  # noqa: E402

for _cand in ("Malgun Gothic", "NanumGothic", "AppleGothic"):
    try:
        matplotlib.font_manager.findfont(_cand, fallback_to_default=False)
        plt.rcParams["font.family"] = _cand
        break
    except Exception:
        continue
plt.rcParams["axes.unicode_minus"] = False

NAVY, BLUE, TEAL, ORANGE, GREEN, RED, GRAY, LGRAY = (
    "#13335f", "#2b6cb0", "#1b9e8a", "#e8743b", "#2e9e5b", "#c0392b", "#7b8794", "#dfe4ea",
)
OUT = Path(__file__).resolve().parent / "figures"
DATA_DIR = Path(os.getenv("CES_DATA_DIR", ROOT / "data"))
OUTPUT_DIR = Path(os.getenv("CES_OUTPUT_DIR", ROOT / "data" / ".b1_seqv2_s42_i42"))
SPLIT_DIR = Path(os.getenv("CES_SPLIT_DIR", ROOT / "data" / ".b1_w2cut_split_s42"))
CUT_EV = float(os.getenv("CES_TI_SPIKE_CUT_EV", "3000"))
DEFAULT_SHOTS = ["31815", "30842"]
TARGETS = [("CES_TI", 0, ORANGE, r"$T_i$ [eV]"), ("CES_VT", 1, BLUE, r"$V_\phi$ [km/s]")]


def robust_z(x):
    med = np.median(x)
    mad = np.median(np.abs(x - med)) * 1.4826 + 1e-9
    return (x - med) / mad


def predict_shot(shot, stats, per_shot):
    """seq_v2 causal pass over the shot's grid + PCHIP on the same (cut, held-free) population."""
    grid, dims = load_grid_files(DATA_DIR, drop_stuck_targets=True, ti_spike_cut_ev=CUT_EV)
    name = f"s{shot}.csv"
    if name not in grid:
        raise SystemExit(f"shot {shot} not in {DATA_DIR}")
    arr = grid[name]  # (rows, [time, TI, VT, fast...]) with the cut + held rule applied
    metrics = json.loads((OUTPUT_DIR / "metrics.json").read_text(encoding="utf-8"))
    hidden = metrics.get("hidden_ti", 160)
    model = SeqCESLSTMv2(hidden_ti=int(hidden)) if hidden != 160 else SeqCESLSTMv2()
    model.load_state_dict(torch.load(OUTPUT_DIR / "weights" / "seq_lstm.pth", map_location="cpu"))
    model.eval()
    t_mean = stats["target"]["mean"].astype(np.float64)
    t_std = stats["target"]["std"].astype(np.float64)
    time_col, tcols = 0, [1, 2]
    times, preds = [], []
    with torch.no_grad():
        for blk in build_blocks(arr, dims, stats, per_shot_norm=per_shot):
            x = torch.from_numpy(blk["x"]).unsqueeze(0)
            p = model(x)[0].numpy() * t_std + t_mean
            times.extend(blk["time"].tolist())
            preds.extend(p.tolist())
    times = np.asarray(times)
    preds = np.asarray(preds)
    # row index of every predicted time
    row_of = {float(t): i for i, t in enumerate(arr[:, time_col])}
    rows = np.asarray([row_of[float(t)] for t in times])
    truth = arr[rows][:, tcols].astype(float)  # NaN where missing / held / cut
    pch = np.asarray([[B.predict("pchip", arr, time_col, tc, int(ri)) for tc in tcols] for ri in rows])
    return dict(times=times, rows=rows, pred=preds, truth=truth, pchip=pch, arr=arr,
                bes_cols=list(range(3, 3 + dims["bes"])),
                ecei_cols=list(range(3 + dims["bes"], 3 + dims["bes"] + dims["ecei"])),
                mc_cols=list(range(3 + dims["bes"] + dims["ecei"], 3 + dims["n_fast"])))


def largest_block(times, gap=0.5):
    breaks = np.where(np.diff(times) >= gap)[0]
    edges = np.r_[0, breaks + 1, len(times)]
    return max(zip(edges[:-1], edges[1:]), key=lambda e: e[1] - e[0])


def make_figure(shot, r, split_label):
    fa = r["arr"]
    t_all = fa[:, 0]
    a, b = largest_block(t_all)
    in_block = (r["times"] >= t_all[a]) & (r["times"] <= t_all[b - 1])
    t = t_all[a:b]
    mc = fa[a:b][:, r["mc_cols"]]
    bes = fa[a:b][:, r["bes_cols"]].mean(axis=1)
    ecei = fa[a:b][:, r["ecei_cols"]].mean(axis=1)
    events = t[1:][robust_z(np.diff(bes)) < -6]

    fig, ax = plt.subplots(4, 1, figsize=(11.5, 9.0), sharex=True,
                           gridspec_kw=dict(height_ratios=[0.85, 1.0, 1.5, 1.5], hspace=0.10))
    ax[0].plot(t, mc[:, 0], lw=0.7, color=TEAL)
    ax[0].plot(t, mc[:, 1], lw=0.7, color=GRAY, alpha=0.65)
    ax[0].set_ylabel("MC1T [a.u.]", fontsize=9.5)
    ax[0].text(0.006, 0.86, "미르노프 코일 — 자기 요동 / MHD 버스트", transform=ax[0].transAxes, fontsize=8.8, color=NAVY)
    ax[1].plot(t, bes, lw=1.0, color=BLUE)
    ax[1].set_ylabel("BES", fontsize=9.5, color=BLUE)
    ax[1].tick_params(axis="y", colors=BLUE)
    axe = ax[1].twinx()
    axe.plot(t, ecei, lw=1.0, color=RED, alpha=0.75)
    axe.set_ylabel("ECEI", fontsize=9.5, color=RED)
    axe.tick_params(axis="y", colors=RED)
    axe.spines["top"].set_visible(False)
    ax[1].text(0.006, 0.86, "빠른 진단 — 급락이 곧 급변 이벤트", transform=ax[1].transAxes, fontsize=8.8, color=NAVY)

    for name, k, col, unit in TARGETS:
        A = ax[2 + k]
        tt = r["times"][in_block]
        truth = r["truth"][in_block, k]
        pred = r["pred"][in_block, k]
        pch = r["pchip"][in_block, k]
        A.plot(tt, pch, "--", lw=1.2, color=GRAY, label="PCHIP 보간 (오프라인, 과거+미래)", zorder=2)
        A.plot(tt, pred, "-", lw=1.5, color=GREEN, label="seq_v2 nowcast (causal)", zorder=3)
        A.plot(tt, truth, ".", ms=4.5, color=col, label=f"{name} 실측 (genuine, 컷 후)", zorder=5)
        ok = np.isfinite(truth) & np.isfinite(pch)
        rm_model = float(np.sqrt(np.mean((pred[ok] - truth[ok]) ** 2)))
        rm_pchip = float(np.sqrt(np.mean((pch[ok] - truth[ok]) ** 2)))
        skill = 1.0 - (rm_model ** 2) / (rm_pchip ** 2) if rm_pchip > 0 else float("nan")
        A.text(0.006, 0.06,
               f"이 shot RMSE — 모델 {rm_model:.1f}  vs  PCHIP {rm_pchip:.1f}   →  skill {skill:+.2f}   (n={int(ok.sum())} 실측점)",
               transform=A.transAxes, fontsize=9, color=NAVY, fontweight="bold")
        A.set_ylabel(unit, fontsize=10.5)
        A.legend(fontsize=8.0, loc="upper right", ncol=3, framealpha=0.92)
    for A in ax:
        for e in events:
            A.axvline(e, color=RED, lw=0.8, ls="--", alpha=0.35, zorder=1)
        A.grid(alpha=0.25, color=LGRAY)
        for s in ("top", "right"):
            A.spines[s].set_visible(False)
    ax[3].set_xlabel("time [s]", fontsize=10.5)
    ax[3].set_xlim(t[0], t[-1])
    ax[0].set_title(f"KSTAR #{shot} ({split_label}) — 급변 구간에서 보간은 무너지고 seq_v2 백본은 따라간다",
                    fontsize=13, fontweight="bold", color=NAVY, pad=10)
    fig.text(0.5, 0.045,
             "빨간 점선 = BES 급락(급변 이벤트) · PCHIP는 과거+미래를 모두 보는 오프라인 보간 · seq_v2는 격자 전체의 과거 + 빠른 진단만 쓰는 causal nowcaster "
             "(W=2 · held-free · 컷 모집단, B.1 백본 split 42)",
             ha="center", fontsize=8.6, color=GRAY)
    OUT.mkdir(exist_ok=True)
    path = OUT / f"fig_transient_seq_{shot}.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {path}")
    return path


def main():
    shots = sys.argv[1:] or DEFAULT_SHOTS
    metrics = json.loads((OUTPUT_DIR / "metrics.json").read_text(encoding="utf-8"))
    stats = _load_stats(metrics)
    per_shot = bool(metrics.get("per_shot_input_norm", True))
    manifest = json.loads((SPLIT_DIR / "split_manifest.json").read_text(encoding="utf-8"))
    val_files = set(manifest.get("val_files", []))
    test_files = set(manifest.get("test_files", []))
    for shot in shots:
        name = f"s{shot}.csv"
        if name in test_files:
            label = "held-out TEST"
        elif name in val_files:
            label = "held-out VAL"
        else:
            raise SystemExit(f"Refusing to plot #{shot}: it is a TRAINING shot in {SPLIT_DIR}.")
        make_figure(shot, predict_shot(shot, stats, per_shot), label)


if __name__ == "__main__":
    main()
