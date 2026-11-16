"""Post hoc readouts of the B.30 row tables (not registered; labelled as such wherever quoted).

1. Fast-channel share of the causal variant: the history-only twin reads no fast channel, so its
   inputs do not depend on how the fast channels are standardized; `causal` vs the history-only
   twin on the same rows is therefore the paired increment of the train-file-standardized network.
2. Where the fast-channel share sits: skill of the network over its history-only twin by quintile
   of the beam-emission level of the row (mean of the BES channels divided by its discharge median)
   and by quintile of its change from the previous row. A share confined to the low-level quintile
   would point at the beam state rather than at the plasma.

CPU, reads data/.b30/rows_*.npz and the shot files.  Run from the repo root:
    py ces_prediction/experiments/b30_thinned/posthoc_b30.py
Output: docs/paper/b30_posthoc.json / .md
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_b30 as R  # noqa: E402

OUT_JSON = R.REPO / "docs" / "paper" / "b30_posthoc.json"
OUT_MD = R.REPO / "docs" / "paper" / "b30_posthoc.md"
HIST = {"cut": "hist80", "incl": "hist30"}
NET = {"cut": "bb80", "incl": "bb"}


def main():
    R.pin_env()
    _torch, _B, _ls, load_grid_files, _bb, _gap, _models = R.imports()
    rng = np.random.default_rng(R.BOOT_SEED)
    cache = {}

    def weights(n):
        if n not in cache:
            cache[n] = rng.multinomial(n, np.full(n, 1.0 / n), size=R.N_BOOT).astype(np.float64)
        return cache[n]

    res = {"note": "post hoc readouts of the B.30 row tables; not registered", "causal_share": {},
           "bes": {}}
    for pop in R.POPS:
        grid, dims = load_grid_files(R.DATA, True, R.POPS[pop])
        names = sorted(grid)
        nb = dims["bes"]
        level, change = {}, {}
        for i, nm in enumerate(names):
            b = grid[nm][:, 3:3 + nb].astype(np.float64).mean(1)
            med = np.median(b)
            sc = med if med > 0 else np.nan
            level[i] = b / sc
            d = np.full(len(b), np.nan)
            d[1:] = np.abs(np.diff(b)) / sc
            change[i] = d
        for s in R.SEEDS:
            z = np.load(R.ROWS_DIR / f"rows_{pop}_s{s}.npz")
            tab = {c.split("__")[1]: z[c] for c in z.files if c.startswith("CES_TI__")}
            ests = ["hold", "gp", "pchip", *R.NETS[pop]]
            fin = tab["has_future"] & np.all([np.isfinite(tab[e]) for e in ests], axis=0)
            se = {e: (tab[e] - tab["y"]).astype(np.float64) ** 2 for e in ests}
            key = f"{pop}|s{s}"
            res["causal_share"][key] = {}
            for k in R.KS:
                sel = fin & (tab["k"] == k)
                f = tab["file"][sel]
                res["causal_share"][key][str(k)] = {
                    "causal_vs_hist": R.skill_ci(se["causal"][sel], se[HIST[pop]][sel], f, weights),
                    "net_vs_hist": R.skill_ci(se[NET[pop]][sel], se[HIST[pop]][sel], f, weights),
                    "causal_vs_pchip": R.skill_ci(se["causal"][sel], se["pchip"][sel], f, weights),
                    "causal_vs_gp": R.skill_ci(se["causal"][sel], se["gp"][sel], f, weights),
                    "causal_vs_hold": R.skill_ci(se["causal"][sel], se["hold"][sel], f, weights)}
            # beam-emission level and change of each row (k = 1 and k = 5)
            lv = np.array([level[int(f)][int(r)] for f, r in zip(tab["file"], tab["row"])])
            ch = np.array([change[int(f)][int(r)] for f, r in zip(tab["file"], tab["row"])])
            res["bes"][key] = {}
            for k in (1, 5):
                sel = fin & (tab["k"] == k) & np.isfinite(lv) & np.isfinite(ch)
                out = {}
                for lab, x in (("level", lv), ("change", ch)):
                    q = np.quantile(x[sel], [0.2, 0.4, 0.6, 0.8])
                    b = np.searchsorted(q, x, side="right")
                    gain_all = (se[HIST[pop]][sel] - se[NET[pop]][sel]).sum()
                    cells = []
                    for j in range(5):
                        ss = sel & (b == j)
                        f = tab["file"][ss]
                        cells.append({
                            "rows": int(ss.sum()),
                            "x_median": float(np.median(x[ss])),
                            "rmse_hold": float(np.sqrt(se["hold"][ss].mean())),
                            "net_vs_hist": R.skill_ci(se[NET[pop]][ss], se[HIST[pop]][ss], f, weights),
                            "net_vs_hold": R.skill_ci(se[NET[pop]][ss], se["hold"][ss], f, weights),
                            "share_of_gain": float((se[HIST[pop]][ss] - se[NET[pop]][ss]).sum() / gain_all)})
                    out[lab] = cells
                res["bes"][key][str(k)] = out
            print(f"{key} done", flush=True)
    OUT_JSON.write_text(json.dumps(res, indent=1), encoding="utf-8")
    OUT_MD.write_text(render(res), encoding="utf-8", newline="\n")
    print("written", OUT_MD)


def f3(x):
    return f"{x[0]:+.3f} [{x[1]:+.3f}, {x[2]:+.3f}]" + ("" if x[1] <= 0 <= x[2] else " *")


def render(res):
    L = ["# B.30 row tables: post hoc readouts (not registered)", "",
         "Skill a vs b = 1 - SSE_a / SSE_b, 95 % interval from 2,000 discharge-level bootstrap "
         "resamples; `*` = the interval excludes 0. T_i only.", "",
         "## 1. Fast-channel share of the causal (train-file-standardized) network", "",
         "The history-only twin reads no fast channel, so its inputs are the same under either "
         "standardization of the fast channels; `causal` vs that twin on the same rows is the paired "
         "increment. The causal network was trained under the 30-epoch cap; the twin under the "
         "80-epoch cap (cut) or the 30-epoch cap (inclusive).", "",
         "| population | split | k | causal vs history-only | network vs history-only | causal vs offline PCHIP | causal vs causal GP | causal vs hold |", "|---|---|---|---|---|---|---|---|"]
    for key, d in res["causal_share"].items():
        pop, s = key.split("|")
        for k, c in d.items():
            L.append(f"| {pop} | {s[1:]} | {k} | {f3(c['causal_vs_hist'])} | {f3(c['net_vs_hist'])} | "
                     f"{f3(c['causal_vs_pchip'])} | {f3(c['causal_vs_gp'])} | {f3(c['causal_vs_hold'])} |")
    L += ["", "## 2. Fast-channel share by the beam-emission level and change of the row", "",
          "Level = mean of the BES channels at the row divided by its median over the discharge; "
          "change = absolute difference of that mean from the previous row, same normalization. "
          "Quintiles of the scored rows of each split. `share of gain` = part of the summed "
          "squared-error reduction (history-only minus network) that falls in the quintile.", ""]
    for key, d in res["bes"].items():
        for k, out in d.items():
            for lab, cells in out.items():
                L += [f"### {key}, k = {k}, by BES {lab}", "",
                      "| quintile | rows | median of the variable | hold RMSE (eV) | network vs history-only | network vs hold | share of gain |",
                      "|---|---|---|---|---|---|---|"]
                for j, c in enumerate(cells):
                    L.append(f"| {j + 1} | {c['rows']} | {c['x_median']:.3f} | {c['rmse_hold']:.1f} | "
                             f"{f3(c['net_vs_hist'])} | {f3(c['net_vs_hold'])} | {c['share_of_gain']:+.2f} |")
                L.append("")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
