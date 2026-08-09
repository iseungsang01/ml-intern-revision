# -*- coding: utf-8 -*-
"""Transient/event figure for the KSTAR CES nowcasting talk.

Story (the KSTAR-#31189-style multi-panel trace, but built from the diagnostics we
actually have):

    MC1T (magnetic fluctuation)  ->  BES/ECEI collapse  ->  CES_TI / CES_VT crash

and, on the two CES panels, the truth is overlaid with what offline PCHIP
interpolation would have said versus what the causal nowcasting model says. The
transients are exactly where interpolation fails and the model wins.

Only **held-out** shots (val/test files of data/.final_split) may be plotted --
plotting model predictions on a training shot would be dishonest. The script
refuses to run on a train shot.

Run from the repo root::

    py docs/presentation/make_figure_transient.py                # default shots
    py docs/presentation/make_figure_transient.py 31815 30842    # pick shots

Reads:  CES_OUTPUT_DIR (default data/.improve_final_out) -> metrics.json + weights/
        CES_SPLIT_DIR   (default data/.improve_split/w4) -> split_manifest.json
        CES_MODEL_PY    (default the archived thesis architecture, see below)
        CES_DATA_DIR    (default data/)                  -> shot CSVs
Writes: docs/presentation/figures/fig_transient_<shot>.png

NOTE on weights + architecture. Two traps here, both verified 2026-07-14:

1. ``ces_prediction/model.py`` has been rewritten by the AutoML loop (it is now a Transformer
   history encoder) and can NO LONGER load any saved checkpoint. The architecture behind the
   thesis result is the archived GRU + masked multi-head-attention snapshot pinned below.
2. The weights in ``data/.improve_final_out/weights/`` are NOT the checkpoint that produced
   the recorded thesis metrics -- they reproduce CES_TI but give CES_VT skill +0.056 instead
   of the recorded +0.161. Retraining the pinned architecture on the canonical split
   reproduces BOTH targets (CES_TI +0.257 CI [+0.118,+0.361] PASS; CES_VT +0.154
   CI [-0.413,+0.336] n.s.), so ``data/.vt_repro_out`` is the trustworthy checkpoint and is
   the default here. Do not silently point this back at ``.improve_final_out``.
"""
import importlib.util
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ces_prediction"))

THESIS_MODEL_PY = Path(os.getenv(
    "CES_MODEL_PY",
    ROOT / "ces_prediction" / "model_iter009.py",
))
_spec = importlib.util.spec_from_file_location("thesis_model", THESIS_MODEL_PY)
_thesis_model = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_thesis_model)
MultimodalCESPredictor = _thesis_model.MultimodalCESPredictor

import baselines_interpolation as B  # noqa: E402
from dataset import KSTAR_CES_Dataset  # noqa: E402
from evaluate import _load_stats  # noqa: E402

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
OUTPUT_DIR = Path(os.getenv("CES_OUTPUT_DIR", ROOT / "data" / ".vt_repro_out"))
SPLIT_DIR = Path(os.getenv("CES_SPLIT_DIR", ROOT / "data" / ".improve_split" / "w4"))
WINDOW = int(os.getenv("CES_WINDOW_SIZE", "4"))

# Physically implausible CES values are spectral-fit failures, not transients.
# (Global CES_TI p99 = 2089 eV with a max of 14984 eV; CES_VT p99 = 206 km/s.)
TI_BAND = (20.0, 3000.0)
VT_BAND = (-60.0, 300.0)

DEFAULT_SHOTS = ["31815", "30842", "32167"]
TARGETS = [
    ("CES_TI", 0, ORANGE, r"$T_i$ [eV]", TI_BAND),
    ("CES_VT", 1, BLUE, r"$V_\phi$ [km/s]", VT_BAND),
]


def largest_block(times, gap=0.5):
    """Row span of the longest contiguous block (dt < gap), mirroring dataset.py."""
    breaks = np.where(np.diff(times) >= gap)[0]
    edges = np.r_[0, breaks + 1, len(times)]
    return max(zip(edges[:-1], edges[1:]), key=lambda e: e[1] - e[0])


def predict_shot(shot, stats):
    """Run the trained model + the PCHIP baseline over every clean sample of one shot.

    Builds the dataset over a temp dir holding only this shot's CSV so it is fast;
    normalization stats still come from the train-file-only stats in metrics.json.
    """
    with tempfile.TemporaryDirectory() as tmp:
        shutil.copy(DATA_DIR / f"s{shot}.csv", Path(tmp) / f"s{shot}.csv")
        # drop_stuck_targets=False matches the convention of the recorded thesis
        # comparison (comparison_metrics.json, 34,644 test samples). Held (forward-filled)
        # CES values are therefore kept -- they are drawn as hollow markers, never as
        # if they were genuine measurements.
        ds = KSTAR_CES_Dataset(
            data_dir=tmp,
            window_size=WINDOW,
            temporal_subset_augmentation=False,
            drop_stuck_targets=False,
        )
        ds.set_normalization_stats(stats)

        model = MultimodalCESPredictor.from_dataset(ds, window_size=WINDOW)
        model.load_state_dict(torch.load(OUTPUT_DIR / "weights" / "multimodal_ces.pth",
                                         map_location="cpu"))
        model.eval()

        t_mean = np.asarray(stats["target"]["mean"], dtype=np.float32)
        t_std = np.asarray(stats["target"]["std"], dtype=np.float32)
        time_col = int(ds._column_slices["time"])
        target_cols = [int(c) for c in ds._column_slices["target"]]
        file_array = ds.file_arrays[0]

        rows, preds, truths, pchips = [], [], [], []
        with torch.no_grad():
            for i in range(len(ds)):
                s = ds[i]
                out = model(
                    s["bes"].unsqueeze(0), s["ecei"].unsqueeze(0), s["mc"].unsqueeze(0),
                    s["time_features"].unsqueeze(0), s["ces_history"].unsqueeze(0),
                )[0].numpy()
                ri = int(s["row_index"])
                rows.append(ri)
                preds.append(out * t_std + t_mean)
                truths.append(s["target"].numpy() * t_std + t_mean)
                # target_mask==0 marks an unobserved target; keep NaN there
                m = s["target_mask"].numpy()
                truths[-1] = np.where(m > 0.5, truths[-1], np.nan)
                pchips.append([
                    B.predict("pchip", file_array, time_col, tc, ri) for tc in target_cols
                ])

        # held = value repeats the previous observed value == forward-filled, not a
        # fresh measurement (~54% of observed CES_VT). Marked hollow in the figure.
        held_all = np.zeros((file_array.shape[0], len(target_cols)), dtype=bool)
        for k, tc in enumerate(target_cols):
            v = file_array[:, tc]
            idx = np.flatnonzero(np.isfinite(v))
            if len(idx) > 1:
                held_all[idx[1:], k] = v[idx[1:]] == v[idx[:-1]]

        rows = np.asarray(rows)
        return dict(
            times=file_array[rows, time_col],
            rows=rows,
            held=held_all[rows],
            pred=np.asarray(preds, dtype=float),
            truth=np.asarray(truths, dtype=float),
            pchip=np.asarray(pchips, dtype=float),
            file_array=file_array,
            time_col=time_col,
            bes_cols=[int(c) for c in ds._column_slices["bes"]],
            ecei_cols=[int(c) for c in ds._column_slices["ecei"]],
            mc_cols=[int(c) for c in ds._column_slices["mc"]],
        )


def robust_z(x):
    med = np.median(x)
    mad = np.median(np.abs(x - med)) * 1.4826 + 1e-9
    return (x - med) / mad


def make_figure(shot, stats, split_label):
    r = predict_shot(shot, stats)
    fa, tc = r["file_array"], r["time_col"]
    t_all = fa[:, tc]
    a, b = largest_block(t_all)
    in_block = (r["times"] >= t_all[a]) & (r["times"] <= t_all[b - 1])

    t = t_all[a:b]
    mc = fa[a:b, r["mc_cols"]]
    bes = fa[a:b, r["bes_cols"]].mean(axis=1)
    ecei = fa[a:b, r["ecei_cols"]].mean(axis=1)
    events = t[1:][robust_z(np.diff(bes)) < -6]  # BES collapse = transient marker

    fig, ax = plt.subplots(
        4, 1, figsize=(11.5, 9.0), sharex=True,
        gridspec_kw=dict(height_ratios=[0.85, 1.0, 1.5, 1.5], hspace=0.10),
    )

    ax[0].plot(t, mc[:, 0], lw=0.7, color=TEAL)
    ax[0].plot(t, mc[:, 1], lw=0.7, color=GRAY, alpha=0.65)
    ax[0].set_ylabel("MC1T [a.u.]", fontsize=9.5)
    ax[0].text(0.006, 0.86, "미르노프 코일 — 자기 요동 / MHD 버스트",
               transform=ax[0].transAxes, fontsize=8.8, color=NAVY)

    ax[1].plot(t, bes, lw=1.0, color=BLUE)
    ax[1].set_ylabel("BES", fontsize=9.5, color=BLUE)
    ax[1].tick_params(axis="y", colors=BLUE)
    axe = ax[1].twinx()
    axe.plot(t, ecei, lw=1.0, color=RED, alpha=0.75)
    axe.set_ylabel("ECEI", fontsize=9.5, color=RED)
    axe.tick_params(axis="y", colors=RED)
    axe.spines["top"].set_visible(False)
    ax[1].text(0.006, 0.86, "빠른 진단 — 급락이 곧 급변 이벤트",
               transform=ax[1].transAxes, fontsize=8.8, color=NAVY)

    for name, k, col, unit, (lo, hi) in TARGETS:
        A = ax[2 + k]
        tt = r["times"][in_block]
        truth = r["truth"][in_block, k].copy()
        pred = r["pred"][in_block, k].copy()
        pch = r["pchip"][in_block, k].copy()
        held = r["held"][in_block, k]
        bad = ~np.isfinite(truth) | (truth < lo) | (truth > hi)
        truth[bad] = np.nan  # never draw fit-failure artifacts as if they were physics

        A.plot(tt, pch, "--", lw=1.2, color=GRAY, label="PCHIP 보간 (오프라인, 과거+미래)", zorder=2)
        A.plot(tt, pred, "-", lw=1.5, color=GREEN, label="모델 nowcast (causal)", zorder=3)
        fresh = np.where(held, np.nan, truth)
        A.plot(tt, fresh, ".", ms=4.5, color=col, label=f"{name} 실측", zorder=5)
        if held.any():
            A.plot(tt, np.where(held, truth, np.nan), "o", ms=3.2, mfc="none",
                   mec=GRAY, mew=0.7, label="held (forward-fill)", zorder=4)

        # Panel skill is scored on genuine measurements only (held values are not data).
        ok = np.isfinite(fresh)
        rm_model = float(np.sqrt(np.mean((pred[ok] - fresh[ok]) ** 2)))
        rm_pchip = float(np.sqrt(np.mean((pch[ok] - fresh[ok]) ** 2)))
        skill = 1.0 - (rm_model ** 2) / (rm_pchip ** 2) if rm_pchip > 0 else float("nan")
        A.text(0.006, 0.06,
               f"이 shot RMSE — 모델 {rm_model:.1f}  vs  PCHIP {rm_pchip:.1f}   →  skill {skill:+.2f}"
               f"   (n={int(ok.sum())} 실측점)",
               transform=A.transAxes, fontsize=9, color=NAVY, fontweight="bold")
        A.set_ylabel(unit, fontsize=10.5)
        A.legend(fontsize=8.0, loc="upper right", ncol=4, framealpha=0.92)

    for A in ax:
        for e in events:
            A.axvline(e, color=RED, lw=0.8, ls="--", alpha=0.35, zorder=1)
        A.grid(alpha=0.25, color=LGRAY)
        for s in ("top", "right"):
            A.spines[s].set_visible(False)
    ax[3].set_xlabel("time [s]", fontsize=10.5)
    ax[3].set_xlim(t[0], t[-1])

    ax[0].set_title(
        f"KSTAR #{shot} ({split_label}) — 급변 구간에서 보간은 무너지고 모델은 따라간다",
        fontsize=13, fontweight="bold", color=NAVY, pad=10,
    )
    fig.text(0.5, 0.045,
             "빨간 점선 = BES 급락(급변 이벤트) · PCHIP는 과거+미래를 모두 보는 오프라인 보간인데도 "
             "급변에서 빗나감 · 모델은 과거 CES + 빠른 진단만 쓰는 causal nowcaster",
             ha="center", fontsize=8.8, color=GRAY)

    OUT.mkdir(exist_ok=True)
    path = OUT / f"fig_transient_{shot}.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {path}")
    return path


def main():
    shots = sys.argv[1:] or DEFAULT_SHOTS
    metrics = json.loads((OUTPUT_DIR / "metrics.json").read_text(encoding="utf-8"))
    stats = _load_stats(metrics)
    manifest = json.loads((SPLIT_DIR / "split_manifest.json").read_text(encoding="utf-8"))
    val_files = set(manifest["val_files"])
    test_files = set(manifest.get("test_files", []))

    for shot in shots:
        name = f"s{shot}.csv"
        if name in test_files:
            label = "held-out TEST"
        elif name in val_files:
            label = "held-out VAL"
        else:
            raise SystemExit(
                f"Refusing to plot #{shot}: it is a TRAINING shot in {SPLIT_DIR}. "
                "Model predictions on a training shot are not honest evidence."
            )
        make_figure(shot, stats, label)


if __name__ == "__main__":
    main()
