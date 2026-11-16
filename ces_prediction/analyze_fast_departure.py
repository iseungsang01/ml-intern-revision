"""Two physical sources of a large fast-channel excursion, read from the 10 ms files (descriptive;
CPU; no model).

The trained network responds to the departure of a fast channel from its mean over the preceding
ten rows, linearly up to about two standard deviations and not at all beyond three (the kernel read
in the manuscript).  Two things can move a fast channel that far within one frame: the heating beam
(BES is light of beam atoms and follows the beam within a millisecond, while the ion temperature
answers through an energy balance, later and by less) and an edge burst.  This script measures, in
all 641 files,

 1. how far the BES and ECEI channels depart from their 100 ms mean (per-discharge standard
    deviations) in the rows after a beam blip and in all other rows, and
 2. a response curve without any model: the change of the measured T_i from its last observed value
    against the mean BES departure of the row, in rows that are not within ten rows of a blip.

Intervals: 2,000 resamples of discharges.  Run from the repo root:
    py ces_prediction/analyze_fast_departure.py
Output: docs/paper/fast_departure.json / .md
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("CES_DATA_DIR", REPO / "data"))
OUT_JSON = REPO / "docs" / "paper" / "fast_departure.json"
OUT_MD = REPO / "docs" / "paper" / "fast_departure.md"
SEG_BREAK_S, N_BOOT, BOOT_SEED = 0.5, 2000, 20261116
CLASSES = ("first row after a blip", "rows 2 to 4 after a blip", "rows 5 to 10 after a blip", "all other rows")
EDGES = (-9.0, -3.0, -2.0, -1.0, -0.5, -0.2, 0.2, 0.5, 1.0, 2.0, 3.0, 9.0)


def main():
    files = sorted(DATA.glob("s*.csv"))
    if len(files) != 641:
        raise SystemExit(f"FATAL: expected 641 CSV files in {DATA}, found {len(files)}")
    nb = len(EDGES) - 1
    cnt = {d: np.zeros((len(files), len(CLASSES), 4)) for d in ("BES", "ECEI")}   # n, |d|>2, |d|>3, d<-3
    med = {d: [[] for _ in CLASSES] for d in ("BES", "ECEI")}
    rs = np.zeros((len(files), nb, 3))                                             # n, sum dT, sum d
    for fi, f in enumerate(files):
        df = pd.read_csv(f).iloc[1:-1].reset_index(drop=True)      # first and last row are not grid rows
        if len(df) < 12:
            continue
        t = df["time"].to_numpy(float)
        ti = df["CES_TI"].to_numpy(float)
        n = len(t)
        dt = np.diff(t)
        seg = np.concatenate(([0], np.cumsum(dt >= SEG_BREAK_S)))
        after = np.zeros(n, int)
        for j in np.flatnonzero((dt > 0.015) & (dt < 0.045)):
            for k in range(1, 11):
                i = j + k
                if i < n and seg[i] == seg[j + 1] and after[i] == 0:
                    after[i] = k
        cls = np.where(after == 1, 0, np.where((after >= 2) & (after <= 4), 1, np.where(after >= 5, 2, 3)))
        dep = {}
        for diag in ("BES", "ECEI"):
            X = df[[c for c in df.columns if c.startswith(diag + "_")]].to_numpy(float)
            z = (X - X.mean(0)) / (X.std(0) + 1e-12)
            cs = np.vstack([np.zeros((1, z.shape[1])), np.cumsum(z, 0)])
            d = np.full_like(z, np.nan)
            for i in range(10, n):
                if seg[i] == seg[i - 10]:
                    d[i] = z[i] - (cs[i] - cs[i - 10]) / 10.0
            dep[diag] = d
            ok = np.isfinite(d[:, 0])
            for c in range(len(CLASSES)):
                v = d[ok & (cls == c)].ravel()
                cnt[diag][fi, c] = (len(v), np.sum(np.abs(v) > 2), np.sum(np.abs(v) > 3), np.sum(v < -3))
                if len(v):
                    med[diag][c].append(v)
        # response curve: rows not within ten rows after a blip, T_i and its last observed value below 3 keV
        last = np.full(n, np.nan)
        cur = np.nan
        for i in range(n):
            last[i] = cur
            if np.isfinite(ti[i]):
                cur = ti[i]
            if i + 1 < n and seg[i + 1] != seg[i]:
                cur = np.nan
        dm = dep["BES"].mean(1)
        m = np.isfinite(dm) & np.isfinite(ti) & np.isfinite(last) & (ti <= 3000) & (last <= 3000) & (after == 0)
        b = np.searchsorted(EDGES, dm[m], side="right") - 1
        okb = (b >= 0) & (b < nb)
        np.add.at(rs[fi, :, 0], b[okb], 1.0)
        np.add.at(rs[fi, :, 1], b[okb], (ti[m] - last[m])[okb])
        np.add.at(rs[fi, :, 2], b[okb], dm[m][okb])
    rng = np.random.default_rng(BOOT_SEED)
    W = rng.multinomial(len(files), np.full(len(files), 1.0 / len(files)), size=N_BOOT).astype(float)

    def ratio(num, den):
        p = float(num.sum() / den.sum())
        bsmp = (W @ num) / (W @ den)
        return [p, float(np.quantile(bsmp, 0.025)), float(np.quantile(bsmp, 0.975))]

    res = {"note": "descriptive; all 641 files; departure = standardized channel minus its mean over the preceding ten rows",
           "classes": {}, "response": []}
    for c, name in enumerate(CLASSES):
        cell = {}
        for diag in ("BES", "ECEI"):
            a = cnt[diag][:, c]
            cell[diag] = {"channel_rows": int(a[:, 0].sum()),
                          "median_departure_sd": float(np.median(np.concatenate(med[diag][c]))),
                          "share_beyond_2sd": ratio(a[:, 1], a[:, 0]), "share_beyond_3sd": ratio(a[:, 2], a[:, 0]),
                          "share_below_minus_3sd": ratio(a[:, 3], a[:, 0])}
        res["classes"][name] = cell
    beyond = cnt["BES"][:, :, 2].sum(0)
    values = cnt["BES"][:, :, 0].sum(0)
    res["bes_beyond_3sd"] = {"channel_values": int(values.sum()), "beyond_3sd": int(beyond.sum()),
                             "of_which_first_row_after_a_blip": int(beyond[0]),
                             "share_in_first_row_after_a_blip": ratio(cnt["BES"][:, 0, 2], cnt["BES"][:, :, 2].sum(1)),
                             "share_of_channel_values_in_that_row": float(values[0] / values.sum())}
    for b in range(nb):
        if rs[:, b, 0].sum() < 50:
            continue
        res["response"].append({"departure_from": EDGES[b], "to": EDGES[b + 1], "rows": int(rs[:, b, 0].sum()),
                                "mean_departure_sd": float(rs[:, b, 2].sum() / rs[:, b, 0].sum()),
                                "mean_ti_change_eV": ratio(rs[:, b, 1], rs[:, b, 0])})
    OUT_JSON.write_text(json.dumps(res, indent=1), encoding="utf-8")
    L = ["# Departure of the fast channels from their 100 ms mean, and the T_i change that goes with it", "",
         "Descriptive, all 641 files (`ces_prediction/analyze_fast_departure.py`); no model. A channel is "
         "standardized per discharge; its departure is its value minus its mean over the preceding ten rows of the "
         "same segment. Intervals: 95 %, 2,000 resamples of discharges.", "",
         "## 1. How far the channels depart, by row class", "",
         "| rows | BES: median departure (sd) | BES: share beyond 2 sd | BES: share beyond 3 sd | BES: share below -3 sd | "
         "ECEI: median departure (sd) | ECEI: share beyond 3 sd |", "|---|---|---|---|---|---|---|"]

    def c3(x):
        return f"{x[0]:.3f} [{x[1]:.3f}, {x[2]:.3f}]"

    for name in CLASSES:
        bq, eq = res["classes"][name]["BES"], res["classes"][name]["ECEI"]
        L.append(f"| {name} ({bq['channel_rows']:,} BES channel values) | {bq['median_departure_sd']:+.2f} | "
                 f"{c3(bq['share_beyond_2sd'])} | {c3(bq['share_beyond_3sd'])} | {c3(bq['share_below_minus_3sd'])} | "
                 f"{eq['median_departure_sd']:+.2f} | {c3(eq['share_beyond_3sd'])} |")
    bb = res["bes_beyond_3sd"]
    L += ["", f"Of the {bb['beyond_3sd']:,} BES channel values beyond three standard deviations, "
          f"{bb['of_which_first_row_after_a_blip']:,} are in the first row after a blip: a share of "
          f"{c3(bb['share_in_first_row_after_a_blip'])}, in rows that hold "
          f"{bb['share_of_channel_values_in_that_row']:.4f} of the {bb['channel_values']:,} channel values."]
    L += ["", "## 2. Response curve without a model (rows not within ten rows after a blip)", "",
          "Change of the measured T_i from its last observed value, by the mean departure of the nine BES channels "
          "of the row. Values above 3 keV are left out.", "",
          "| mean BES departure (sd) | rows | mean departure in the bin (sd) | mean T_i change (eV) |", "|---|---|---|---|"]
    for r in res["response"]:
        x = r["mean_ti_change_eV"]
        L.append(f"| {r['departure_from']:+.1f} to {r['to']:+.1f} | {r['rows']:,} | {r['mean_departure_sd']:+.2f} | "
                 f"{x[0]:+.1f} [{x[1]:+.1f}, {x[2]:+.1f}] |")
    OUT_MD.write_text("\n".join(L) + "\n", encoding="utf-8", newline="\n")
    print("written", OUT_MD)


if __name__ == "__main__":
    main()
