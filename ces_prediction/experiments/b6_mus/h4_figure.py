"""B.6 H4 (PREREGISTRATION_B6.md sec. 7): published transitions, three-tier figures.

Qualitative only -- no statistic is computed here. For each discharge in the fixed list,
one figure with (top) the raw Mirnov spectrogram 0..50 kHz from the delivered 2 MHz
stream, (middle) CES_TI observations with the A0 and A1 refit predictions, (bottom)
CES_VT observations with the same, and the pre-registered time markers drawn as
vertical lines. Only discharges the refits never trained on are drawn: the TEST shots
with published markers (#31921, #31873) and the two companions (#31923, #31357).

Predictions come from the batch's own inference path (`eval_seq.build_pred_lookup`)
with the batch env, so the curves are the scored model, not a re-implementation.

Usage (repo root, after run_b6.py):
  py ces_prediction/experiments/b6_mus/h4_figure.py [--pop cut] [--seed 42]
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(1, str(HERE.parents[0] / "seq"))
sys.path.insert(2, str(HERE.parents[1]))          # ces_prediction/
sys.path.insert(3, str(HERE.parents[0] / "hires_shots"))
sys.path.insert(4, str(HERE.parents[0] / "b1_gate"))

DATA = REPO_ROOT / "data"
OUT = DATA / ".b6_h4"

# sec. 7 fixed markers (seconds); only never-trained discharges are drawn
MARKERS = {
    31921: {"FIRE": 5.40, "H-mode": 8.05},
    31873: {"ELM-suppression optimisation trigger": 4.5},
    31923: {"L -> FIRE": 6.1},
    31357: {"ETB (ERMP on)": 5.5},
}


def set_batch_env(pop, arm, out_dir):
    from run_b6 import base_env, STAGE_DATA, FEATURE_DIR, POPULATIONS
    env = base_env(arm, POPULATIONS[pop], FEATURE_DIR)
    env["CES_OUTPUT_DIR"] = str(out_dir)
    env["CES_DATA_DIR"] = str(STAGE_DATA)
    for k, v in env.items():
        os.environ[k] = v
    if arm == "a0":
        os.environ.pop("CES_SEQ_EXTRA_VT", None)
    return Path(env["CES_DATA_DIR"])


def predictions(pop, arm, seed, shots):
    import torch
    out_dir = DATA / f".b6_{pop}_{arm}_refit_s{seed}"
    data_dir = set_batch_env(pop, arm, out_dir)
    import importlib
    import seq_data
    importlib.reload(seq_data)                 # extra_vt_meta reads the env at import time
    import eval_seq
    importlib.reload(eval_seq)
    from evaluate import _load_stats
    metrics = json.loads((out_dir / "metrics.json").read_text(encoding="utf-8"))
    my_stats = _load_stats(metrics)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    names = [f"s{s}.csv" for s in shots]
    lookups = eval_seq.build_pred_lookup(data_dir, names, my_stats, device)
    out = {}
    for name in names:
        tab = lookups["full"].get(name, {})
        t = np.array(sorted(tab))
        p = np.array([tab[k] for k in t]) if len(t) else np.zeros((0, 2))
        out[int(name[1:-4])] = (t, p)
    return out


def spectrogram(shot, coil="MC1T03", f_max_khz=50.0, seg=8192, hop=16384):
    from raw_loader import load_raw
    (t, x), fs = load_raw(shot, coil)
    w = np.hanning(seg)
    S, T = [], []
    for k in range(0, len(x) - seg, hop):
        s = x[k:k + seg]
        S.append(np.abs(np.fft.rfft((s - s.mean()) * w)) ** 2)
        T.append(t[k + seg // 2])
    f = np.fft.rfftfreq(seg, d=1.0 / fs)
    m = f <= f_max_khz * 1e3
    return np.array(T), f[m] / 1e3, np.log10(np.array(S)[:, m].T + 1e-20)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    ap = argparse.ArgumentParser()
    ap.add_argument("--pop", default="cut")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    shots = list(MARKERS)
    pred = {arm: predictions(args.pop, arm, args.seed, shots) for arm in ("a0", "a1")}

    for shot in shots:
        df = pd.read_csv(DATA / ".b6_data" / f"s{shot}.csv", usecols=["time", "CES_TI", "CES_VT"])
        T, F, S = spectrogram(shot)
        fig, ax = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
        ax[0].pcolormesh(T, F, S, shading="auto")
        ax[0].set_ylabel("MC1T03 [kHz]")
        ax[0].set_title(f"#{shot}  raw Mirnov spectrogram (2 MHz) / CES observations / "
                        f"A0 vs A1 refit (pop={args.pop}, init {args.seed})")
        for i, (col, lab) in enumerate((("CES_TI", "T_i [eV]"), ("CES_VT", "V_rot [km/s]"))):
            a = ax[i + 1]
            obs = df[np.isfinite(df[col])]
            a.plot(obs["time"], obs[col], "k.", ms=3, label="CES observed")
            for arm, c in (("a0", "C0"), ("a1", "C3")):
                t, p = pred[arm][shot]
                if len(t):
                    a.plot(t, p[:, i], c, lw=0.8, alpha=0.9, label=f"{arm.upper()} refit")
            a.set_ylabel(lab)
            a.legend(loc="upper left", fontsize=8)
        for lab, tm in MARKERS[shot].items():
            for a in ax:
                a.axvline(tm, color="w" if a is ax[0] else "gray", ls="--", lw=0.9)
            ax[1].text(tm, ax[1].get_ylim()[1], lab, fontsize=8, va="top", ha="left")
        ax[2].set_xlabel("time [s]")
        lo = max(T[0], df["time"].min())
        ax[2].set_xlim(lo, T[-1])
        fig.tight_layout()
        dst = OUT / f"h4_s{shot}_{args.pop}_s{args.seed}.png"
        fig.savefig(dst, dpi=110)
        plt.close(fig)
        print(f"[h4] wrote {dst}")


if __name__ == "__main__":
    main()
