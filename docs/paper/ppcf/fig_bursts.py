"""Case figure for B.31: raw 500 kHz BES level, bursts and beam blips above the CES ion temperature
and its causal estimate, for the two discharges of this data set with a published edge analysis
(31921, 31923).  Windows start 0.2 s before the first burst-active row of the late phase.

CPU inference with the frozen train-file-standardized network of split 1 (both discharges are
TEST discharges of that split).  Run from the repo root:
    py docs/paper/ppcf/fig_bursts.py
Writes fig_bursts.pdf / .png and fig_bursts.json (window limits, counts).
"""
import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).parent
ROOT = HERE.parents[2]
RUN = ROOT / "data" / ".b29_norm0_cut_s1"
SHOTS = ((31921, 7.20, 8.60), (31923, 5.90, 7.98))

for k in [k for k in os.environ if k.startswith("CES_")]:
    del os.environ[k]
os.environ.update({"CES_DATA_DIR": str(ROOT / "data"), "CES_DROP_STUCK_TARGETS": "1",
                   "CES_TI_SPIKE_CUT_EV": "3000", "CES_PER_SHOT_NORM": "0", "CES_SEQ_MODEL": "v2",
                   "CES_SEQ_NORM_EXPANDING": "0", "CES_SEQ_ZERO_MC": "0", "CES_OUTPUT_DIR": str(RUN)})
sys.path.insert(0, str(ROOT / "ces_prediction" / "experiments" / "seq"))
sys.path.insert(1, str(ROOT / "ces_prediction"))
sys.path.insert(2, str(ROOT / "ces_prediction" / "experiments" / "b31_bursts"))
import torch  # noqa: E402
from evaluate import _load_stats  # noqa: E402
from eval_seq import build_pred_lookup  # noqa: E402
from seq_data import load_grid_files  # noqa: E402
import run_b31 as B31  # noqa: E402


def main():
    stats = _load_stats(json.loads((RUN / "metrics.json").read_text(encoding="utf-8")))
    grid, _ = load_grid_files(ROOT / "data", True)
    names = [f"s{s}.csv" for s, _, _ in SHOTS]
    with torch.no_grad():
        look = build_pred_lookup(ROOT / "data", names, stats, torch.device("cpu"))["full"]
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                         "legend.frameon": False, "pdf.fonttype": 42})
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.3), sharex="col",
                             gridspec_kw={"height_ratios": [1.0, 1.5]})
    info = {}
    for j, (shot, lo, hi) in enumerate(SHOTS):
        raw = B31.bursts_and_blips(shot)
        arr = grid[f"s{shot}.csv"]
        t = arr[:, 0].astype(float)
        ti = arr[:, 1].astype(float)
        est = np.array([look[f"s{shot}.csv"][float(x)][0] if float(x) in look[f"s{shot}.csv"] else np.nan
                        for x in arr[:, 0]])
        hold = np.full(len(t), np.nan)
        last = np.nan
        for i in range(len(t)):
            hold[i] = last
            if np.isfinite(ti[i]):
                last = ti[i]
        w = (t >= lo) & (t <= hi)
        t1, lv = raw["level_1ms"]
        m1 = (t1 >= lo) & (t1 <= hi)
        a, b = axes[0, j], axes[1, j]
        a.plot(t1[m1], lv[m1], color="#555555", lw=0.6)
        ev = raw["bursts"][(raw["bursts"] >= lo) & (raw["bursts"] <= hi)]
        a.plot(ev, np.full(len(ev), 2.55), "|", color="#D55E00", ms=6, mew=0.9)
        for x in raw["blips"][(raw["blips"] >= lo) & (raw["blips"] <= hi)]:
            a.axvspan(x - 0.0005, x + 0.0005, color="#0072B2", alpha=0.35, lw=0)
        a.set_ylim(0, 2.8)
        a.set_ylabel("BES level / median")
        a.set_title(f"({'ab'[j]}) discharge {shot}", loc="left", fontsize=8)
        miss = w & ~np.isfinite(ti)
        for x in t[miss]:
            b.axvspan(x - 0.005, x + 0.005, color="#000000", alpha=0.07, lw=0)
        b.plot(t[w], hold[w], color="#999999", lw=0.8, ls="--", label="hold")
        b.plot(t[w], est[w], color="#009E73", lw=1.1, label="causal estimate")
        b.plot(t[w], ti[w], "o", color="#000000", ms=2.2, label=r"measured $T_i$")
        b.set_ylim(0, 3100)
        b.set_ylabel(r"$T_i$ (eV)")
        b.set_xlabel("time (s)")
        b.set_xlim(lo, hi)
        if j == 0:
            b.legend(fontsize=6.5, loc="upper left", ncol=3)
        info[str(shot)] = {"window_s": [lo, hi], "bursts_in_window": int(len(ev)),
                           "rows_in_window": int(w.sum()), "rows_missing": int(miss.sum()),
                           "blips_ms_in_window": int(((raw["blips"] >= lo) & (raw["blips"] <= hi)).sum())}
    fig.tight_layout()
    fig.savefig(HERE / "fig_bursts.pdf")
    fig.savefig(HERE / "fig_bursts.png", dpi=200)
    (HERE / "fig_bursts.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
    print(json.dumps(info))


if __name__ == "__main__":
    main()
