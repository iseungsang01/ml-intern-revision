"""B.32, read after the fact (NOT registered): what the trained estimators do in the first row
after a beam blip, by the BES departure of that row.

The registered readout (b32_two_sources.md, part A) measured the T_i change in the first row after
a blip in three groups of the mean BES departure d.  Here the frozen B.30 row tables (natural
cadence, cut population, four file-level splits) are read on the same kind of row: the measured
change from the last visible sample, and the change each estimator predicts.

CPU; nothing is retrained.  Run from the repo root:
    py ces_prediction/experiments/b32_two_sources/posthoc_b32_model.py
Output: docs/paper/b32_posthoc_model.json / .md
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "b30_thinned"))
import run_b32 as R  # noqa: E402
import run_b30 as R30  # noqa: E402

OUT_JSON = R.REPO / "docs" / "paper" / "b32_posthoc_model.json"
OUT_MD = R.REPO / "docs" / "paper" / "b32_posthoc_model.md"
ESTS = ("hist80", "causal", "bb80", "gp")
LABEL = {"hist80": "history-only twin", "causal": "causal variant", "bb80": "whole-discharge variant",
         "gp": "causal GP"}
GROUPS = (("D1 (d <= -3)", lambda d: d <= -3), ("D2 (-3 < d <= -2)", lambda d: (d > -3) & (d <= -2)),
          ("D3 (d > -2)", lambda d: d > -2))


def ci(b):
    b = b[np.isfinite(b)]
    return [float(np.quantile(b, 0.025)), float(np.quantile(b, 0.975))]


def main():
    R30.pin_env()
    _t, _B, _ls, load_grid_files, _bb, _gap, _m = R30.imports()
    grid, _dims = load_grid_files(R30.DATA, True, R30.POPS["cut"])
    names = sorted(grid)
    dep = {}
    for i, nm in enumerate(names):
        stem = nm[:-4] if nm.endswith(".csv") else nm
        k = R.kstar_file(R30.DATA / f"{stem}.csv")
        flag = np.full(grid[nm].shape[0], np.nan)
        if k is not None and k["first"]:
            dmap = {round(float(k["t"][j + 1]), 4): float(k["d"][j + 1]) for j in k["first"]}
            for r, tt in enumerate(grid[nm][:, 0]):
                v = dmap.get(round(float(tt), 4))
                if v is not None:
                    flag[r] = v
        dep[i] = flag
    rng = np.random.default_rng(R.SEED + 2)
    res = {"note": "read after the fact; frozen B.30 row tables, cut population, natural cadence", "splits": {}}
    for s in R30.SEEDS:
        z = np.load(R30.ROWS_DIR / f"rows_cut_s{s}.npz")
        tab = {c.split("__")[1]: z[c] for c in z.files if c.startswith("CES_TI__")}
        d = np.array([dep[int(f)][int(r)] for f, r in zip(tab["file"], tab["row"])])
        fin = (tab["k"] == 1) & np.isfinite(d) & np.all([np.isfinite(tab[e]) for e in ("hold", *ESTS)], axis=0)
        y, h = tab["y"].astype(float), tab["hold"].astype(float)
        cell = {}
        for name, fn in GROUPS:
            m = fin & fn(d)
            if m.sum() < 5:
                cell[name] = {"rows": int(m.sum())}
                continue
            f = tab["file"][m]
            uf, inv = np.unique(f, return_inverse=True)
            W = rng.multinomial(len(uf), np.full(len(uf), 1.0 / len(uf)), size=R.N_BOOT).astype(float)[:, inv]
            den = W.sum(1)
            meas = y[m] - h[m]
            g = {"rows": int(m.sum()), "discharges": int(len(uf)), "mean_departure_sd": float(d[m].mean()),
                 "measured_change_eV": float(meas.mean()), "measured_change_ci95": ci((W * meas).sum(1) / den),
                 "hold_rmse_eV": float(np.sqrt((meas ** 2).mean()))}
            for e in ESTS:
                p = tab[e].astype(float)[m]
                pred = p - h[m]
                err = p - y[m]
                g[e] = {"predicted_change_eV": float(pred.mean()), "predicted_change_ci95": ci((W * pred).sum(1) / den),
                        "mean_error_eV": float(err.mean()), "mean_error_ci95": ci((W * err).sum(1) / den),
                        "rmse_eV": float(np.sqrt((err ** 2).mean()))}
            sse_c, sse_t = (tab["causal"].astype(float)[m] - y[m]) ** 2, (tab["hist80"].astype(float)[m] - y[m]) ** 2
            with np.errstate(invalid="ignore", divide="ignore"):
                b = 1.0 - (W * sse_c).sum(1) / (W * sse_t).sum(1)
            g["causal_vs_twin_skill"] = float(1.0 - sse_c.sum() / sse_t.sum())
            g["causal_vs_twin_skill_ci95"] = ci(b)
            cell[name] = g
        res["splits"][str(s)] = cell
        print(s, "done", flush=True)
    OUT_JSON.write_text(json.dumps(res, indent=1), encoding="utf-8")
    L = ["# B.32, read after the fact: the estimators in the first row after a beam blip (not registered)", "",
         "Frozen B.30 row tables, T_i, cut population, natural cadence, test discharges of the four file-level "
         "splits. Rows: the first row after an absent frame with six contiguous rows before it. d = mean departure "
         "of the nine BES channels from their 100 ms mean (per-discharge standard deviations). Change = value "
         "minus the last visible sample (hold). Intervals: 95 %, 2,000 resamples of discharges.", ""]
    for name, _fn in GROUPS:
        L += [f"## {name}", "",
              "| split | rows | mean d (sd) | measured change (eV) | predicted: history-only twin | predicted: causal variant | "
              "predicted: whole-discharge variant | predicted: causal GP | causal variant vs twin (skill) |",
              "|---|---|---|---|---|---|---|---|---|"]
        for s in R30.SEEDS:
            g = res["splits"][str(s)][name]
            if "measured_change_eV" not in g:
                L.append(f"| {s} | {g['rows']} | | | | | | | |")
                continue

            def pc(e):
                x = g[e]
                return f"{x['predicted_change_eV']:+.1f} [{x['predicted_change_ci95'][0]:+.1f}, {x['predicted_change_ci95'][1]:+.1f}]"

            L.append(f"| {s} | {g['rows']} | {g['mean_departure_sd']:+.2f} | {g['measured_change_eV']:+.1f} "
                     f"[{g['measured_change_ci95'][0]:+.1f}, {g['measured_change_ci95'][1]:+.1f}] | {pc('hist80')} | "
                     f"{pc('causal')} | {pc('bb80')} | {pc('gp')} | {g['causal_vs_twin_skill']:+.3f} "
                     f"[{g['causal_vs_twin_skill_ci95'][0]:+.3f}, {g['causal_vs_twin_skill_ci95'][1]:+.3f}] |")
        L.append("")
    OUT_MD.write_text("\n".join(L) + "\n", encoding="utf-8", newline="\n")
    print("written", OUT_MD)


if __name__ == "__main__":
    main()
