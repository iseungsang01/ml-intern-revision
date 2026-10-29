"""Heating-power step vs the CES labels of the same discharge (descriptive figure, no model).

Source of the heating trace: Fig. 6a (top panel, "P_tot [10 MW]") of Kim et al., "Tailoring
tokamak error fields to control plasma instabilities and transport", Nat. Commun. 15 (2024),
doi 10.1038/s41467-024-45454-1 (CC BY, PMC10858871), which plots shots 31357 (blue) and
31359 (red) -- two of our discharges. The curve is digitised from the published PNG
(1987 x 1018): x ticks 2/4/6/8/10 s at px 174.5/330/485/640/795.5, y = 0 at the frame bottom
(px 202) and 0.8 at px 101. Per pixel column the red (or blue) pixels in the P_tot band are
read as [min, max] value, so the diagnostic beam notches appear as a band. Resolution is one
pixel = 12.9 ms and 0.08 MW; the trace is TOTAL external heating (NBI + ECH), not NBI alone.

What the figure is for (sec. 8bd addendum): the heating input is a step function held
constant between steps, so its effect lives in the slow band of V_rot. The overlay shows the
macro-scale response of the edge CES V_rot (R = 2.198 m) to the power step and, below it, the
10 ms increments that the step does not touch.

Usage (repo root):
  py ces_prediction/experiments/b6_mus/nbi_step_overlay.py --png <Fig6 png> [--shot 31359]
Outputs: ces_prediction/experiments/b6_mus/ptot_fig6a_digitized.csv, docs/paper/figures/fig_nbi_step_31359.png
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
DATA = REPO / "data"
CSV_OUT = HERE / "ptot_fig6a_digitized.csv"
FIG_OUT = REPO / "docs" / "paper" / "figures"

X_TICKS = {2.0: 174.5, 4.0: 330.0, 6.0: 485.0, 8.0: 640.0, 10.0: 795.5}
Y0_PX, Y08_PX = 202.0, 101.0           # value 0 and 0.8 (x 10 MW)
PANEL_Y = (84, 200)                     # inside the top panel frame
PANEL_X = (60, 833)
PTOT_BAND_MIN_Y = 135                   # pixels below this row are the P_tot trace (I_p sits above)


def px_to_time(x):
    xs, ts = np.array(list(X_TICKS.values())), np.array(list(X_TICKS.keys()))
    slope = np.polyfit(xs, ts, 1)
    return np.polyval(slope, x)


def px_to_mw(y):
    return (Y0_PX - np.asarray(y, dtype=float)) / (Y0_PX - Y08_PX) * 0.8 * 10.0


def digitise(png):
    from PIL import Image
    im = np.asarray(Image.open(png).convert("RGB")).astype(int)
    masks = {31359: (im[:, :, 0] > 180) & (im[:, :, 1] < 90) & (im[:, :, 2] < 90),
             31357: (im[:, :, 2] > 150) & (im[:, :, 0] < 90) & (im[:, :, 1] < 90)}
    rows = []
    for shot, m in masks.items():
        for x in range(*PANEL_X):
            ys = np.where(m[PANEL_Y[0]:PANEL_Y[1], x])[0] + PANEL_Y[0]
            ys = ys[ys >= PTOT_BAND_MIN_Y]
            if len(ys) == 0:
                continue
            rows.append({"shot": shot, "time_s": float(px_to_time(x)),
                         "ptot_mw_max": float(px_to_mw(ys.min())), "ptot_mw_min": float(px_to_mw(ys.max()))})
    df = pd.DataFrame(rows).sort_values(["shot", "time_s"])
    df.to_csv(CSV_OUT, index=False)
    return df


def labels(shot):
    d = pd.read_csv(DATA / f"s{shot}.csv", usecols=["time", "CES_TI", "CES_VT"]).iloc[1:-1]
    out = {"time": d["time"].to_numpy(float)}
    for c in ("CES_TI", "CES_VT"):
        y = d[c].to_numpy(float).copy()
        held = np.zeros_like(y, bool)
        held[1:] = y[1:] == y[:-1]      # forward-filled rows are not measurements
        y[held] = np.nan
        out[c] = y
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--png", required=True)
    ap.add_argument("--shot", type=int, default=31359)
    args = ap.parse_args()
    df = digitise(args.png)
    p = df[df.shot == args.shot]
    lab = labels(args.shot)

    # the power step: largest jump of the column-wise minimum after 3 s
    t, pmin, pmax = p.time_s.to_numpy(), p.ptot_mw_min.to_numpy(), p.ptot_mw_max.to_numpy()
    level = np.minimum(pmax, np.maximum(pmin, np.median(pmax)))  # robust level ignoring notches
    smooth = pd.Series(pmax).rolling(9, center=True, min_periods=1).median().to_numpy()
    # the step-UP inside the flat-top: largest positive jump after 3 s and before the ramp-down
    t_end = float(t[smooth > 1.0].max())
    sel = (t[1:] > 3.0) & (t[1:] < t_end - 0.3)
    j = np.argmax(np.diff(smooth) * sel)
    t_step = float(t[j + 1])
    before = (lab["time"] > t_step - 0.6) & (lab["time"] < t_step - 0.1)
    after = (lab["time"] > t_step + 0.2) & (lab["time"] < t_step + 0.7)
    stats = {"shot": args.shot, "t_step_s": t_step,
             "ptot_before_mw": float(np.nanmedian(smooth[(t > t_step - 0.6) & (t < t_step - 0.1)])),
             "ptot_after_mw": float(np.nanmedian(smooth[(t > t_step + 0.2) & (t < t_step + 0.7)]))}
    for c in ("CES_VT", "CES_TI"):
        y = lab[c]
        stats[f"{c}_before_median"] = float(np.nanmedian(y[before]))
        stats[f"{c}_after_median"] = float(np.nanmedian(y[after]))
        stats[f"{c}_n_before_after"] = [int(np.isfinite(y[before]).sum()), int(np.isfinite(y[after]).sum())]
        inc = np.diff(y[np.isfinite(y)])
        stats[f"{c}_increment_sd"] = float(np.nanstd(inc))
    print(json.dumps(stats, indent=1))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(4, 1, figsize=(8, 8.5), sharex=True,
                           gridspec_kw={"height_ratios": [1.2, 1, 1, 1]})
    ax[0].fill_between(t, pmin, pmax, color="tab:red", alpha=0.35, step="mid", label="digitised band (notches)")
    ax[0].plot(t, smooth, color="tab:red", lw=1.5, label="P_tot (Fig. 6a, Kim et al. 2024)")
    ax[0].axvline(t_step, color="k", ls="--", lw=1)
    ax[0].set_ylabel("P_tot [MW]")
    ax[0].set_title(f"KSTAR #{args.shot}: total heating power (published trace) vs CES labels (this dataset)")
    ax[0].legend(loc="lower right", fontsize=8)
    for k, (c, unit) in enumerate((("CES_VT", "V_rot [km/s]"), ("CES_TI", "T_i [eV]"))):
        y = lab[c]
        ok = np.isfinite(y)
        ax[1 + k].plot(lab["time"][ok], y[ok], ".", ms=2.5, color="tab:blue" if c == "CES_VT" else "tab:orange")
        ax[1 + k].axvline(t_step, color="k", ls="--", lw=1)
        ax[1 + k].set_ylabel(unit)
    y = lab["CES_VT"]
    ok = np.isfinite(y)
    tt, yy = lab["time"][ok], y[ok]
    inc = np.diff(yy)
    gap_ok = np.diff(tt) < 0.015
    ax[3].plot(tt[1:][gap_ok], inc[gap_ok], ".", ms=2.5, color="tab:gray")
    ax[3].axvline(t_step, color="k", ls="--", lw=1)
    ax[3].axhline(0, color="k", lw=0.5)
    ax[3].set_ylabel("dV_rot / 10 ms [km/s]")
    ax[3].set_xlabel("time [s]")
    lo = max(1.0, float(np.nanmin(lab["time"])) - 0.5)
    hi = float(np.nanmax(lab["time"])) + 0.3
    ax[3].set_xlim(lo, hi)
    fig.tight_layout()
    FIG_OUT.mkdir(parents=True, exist_ok=True)
    out = FIG_OUT / f"fig_nbi_step_{args.shot}.png"
    fig.savefig(out, dpi=160)
    (HERE / f"nbi_step_{args.shot}.json").write_text(json.dumps(stats, indent=1), encoding="utf-8")
    print(f"wrote {out} and {CSV_OUT}")


if __name__ == "__main__":
    main()
