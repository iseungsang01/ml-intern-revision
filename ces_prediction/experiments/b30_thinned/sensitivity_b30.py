"""Post hoc sensitivity of the B.30 readouts to suspicious rows (not registered).

Asked for by the seventh independent grading of the PPCF draft: does the fast-channel share stay
when (a) the value cut is tighter (2.5 keV) or looser (4 keV, inclusive population), (b) very low
values are left out, (c) rows with a jump of more than 1 keV from the last visible sample are left
out, (d) rows within 30 ms after a row whose BES level is below half of the discharge median (a
beam blip seen in the 10 ms snapshot) are left out.  Added 2026-11-16: (e) the ten rows after an
absent frame, which is where a beam blip sits in the files.

CPU, reads data/.b30/rows_*.npz and the shot files.  Run from the repo root:
    py ces_prediction/experiments/b30_thinned/sensitivity_b30.py
Output: docs/paper/b30_sensitivity.json / .md
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_b30 as R  # noqa: E402

OUT_JSON = R.REPO / "docs" / "paper" / "b30_sensitivity.json"
OUT_MD = R.REPO / "docs" / "paper" / "b30_sensitivity.md"
LOW_EV, JUMP_EV, BLIP_S = 150.0, 1000.0, 0.030
# A beam blip leaves one to three absent frames in the 10 ms files (a step of 20 to 40 ms in the
# time column: 1,414 such steps; all 45 inside the raw deliveries coincide with a blip in the raw
# BES).  T_i stays below its level before the blip for about 90 ms and the ECEI level with it
# (ces_prediction/analyze_file_rules.py), so the ten rows after the absent frame are set apart.
GAP_S = 0.100
PAIRS = {"cut": [("bb80", "hist80"), ("causal", "hist80"), ("bb", "gp"), ("causal", "gp"), ("bb", "hold")],
         "incl": [("bb", "hist30"), ("causal", "hist30"), ("bb", "gp"), ("causal", "gp"), ("bb", "hold")]}


def main():
    R.pin_env()
    _t, _B, _ls, load_grid_files, _bb, _gap, _m = R.imports()
    rng = np.random.default_rng(R.BOOT_SEED)
    cache = {}

    def weights(n):
        if n not in cache:
            cache[n] = rng.multinomial(n, np.full(n, 1.0 / n), size=R.N_BOOT).astype(np.float64)
        return cache[n]

    res = {"note": "post hoc sensitivity of the B.30 row tables; not registered",
           "low_eV": LOW_EV, "jump_eV": JUMP_EV, "blip_s": BLIP_S, "cells": {}}
    for pop in R.POPS:
        grid, dims = load_grid_files(R.DATA, True, R.POPS[pop])
        names = sorted(grid)
        after_blip = {}
        after_gap = {}
        for i, nm in enumerate(names):
            arr = grid[nm]
            t = arr[:, 0].astype(np.float64)
            b = arr[:, 3:3 + dims["bes"]].astype(np.float64).mean(1)
            low = np.flatnonzero(b < 0.5 * np.median(b))
            flag = np.zeros(len(t), bool)
            for j in low:
                flag |= (t >= t[j]) & (t <= t[j] + BLIP_S)
            after_blip[i] = flag
            dt = np.diff(t)
            g = np.flatnonzero((dt > 0.015) & (dt < 0.045))
            flag2 = np.zeros(len(t), bool)
            for j in g:
                flag2 |= (t >= t[j + 1]) & (t < t[j + 1] + GAP_S - 1e-6)
            after_gap[i] = flag2
        for s in R.SEEDS:
            z = np.load(R.ROWS_DIR / f"rows_{pop}_s{s}.npz")
            tab = {c.split("__")[1]: z[c] for c in z.files if c.startswith("CES_TI__")}
            ests = ["hold", "gp", "pchip", *R.NETS[pop]]
            fin = tab["has_future"] & np.all([np.isfinite(tab[e]) for e in ests], axis=0)
            se = {e: (tab[e] - tab["y"]).astype(np.float64) ** 2 for e in ests}
            y, h = tab["y"].astype(np.float64), tab["hold"].astype(np.float64)
            blip = np.array([after_blip[int(f)][int(r)] for f, r in zip(tab["file"], tab["row"])])
            gap = np.array([after_gap[int(f)][int(r)] for f, r in zip(tab["file"], tab["row"])])
            sets = {"all": np.ones(len(y), bool),
                    "not within 100 ms after a beam blip (absent frame)": ~gap,
                    "only within 100 ms after a beam blip (absent frame)": gap,
                    "values <= 2.5 keV": (y <= 2500) & (h <= 2500),
                    "values >= 150 eV": (y >= LOW_EV) & (h >= LOW_EV),
                    "jump <= 1 keV": np.abs(y - h) <= JUMP_EV,
                    "not within 30 ms after a low-BES row": ~blip}
            sets["all four exclusions"] = (sets["values <= 2.5 keV"] & sets["values >= 150 eV"]
                                           & sets["jump <= 1 keV"] & ~blip)
            sets["all four exclusions and no row within 100 ms after a blip"] = sets["all four exclusions"] & ~gap
            if pop == "incl":
                sets["values <= 4 keV"] = (y <= 4000) & (h <= 4000)
            for k in (1, 5):
                for name, m in sets.items():
                    sel = fin & (tab["k"] == k) & m
                    f = tab["file"][sel]
                    cell = {"rows": int(sel.sum()),
                            "share_of_rows": float(sel.sum() / max(1, (fin & (tab["k"] == k)).sum()))}
                    for a, b in PAIRS[pop]:
                        cell[f"{a}_vs_{b}"] = R.skill_ci(se[a][sel], se[b][sel], f, weights)
                    res["cells"][f"{pop}|s{s}|k{k}|{name}"] = cell
            print(pop, s, "done", flush=True)
    OUT_JSON.write_text(json.dumps(res, indent=1), encoding="utf-8")
    OUT_MD.write_text(render(res), encoding="utf-8", newline="\n")
    print("written", OUT_MD)


def render(res):
    L = ["# B.30 row tables: sensitivity to suspicious rows (post hoc, not registered)", "",
         "T_i, skill a vs b = 1 - SSE_a / SSE_b. Each cell: range of the four file-level splits and the "
         "number of splits whose 95 % interval (2,000 discharge-level resamples) excludes 0. A row is "
         "left out when the stated condition fails for the scored value or for the last visible "
         "sample.", ""]
    for pop in ("cut", "incl"):
        for k in (1, 5):
            names = sorted({key.split("|")[3] for key in res["cells"] if key.startswith(f"{pop}|") and f"|k{k}|" in key},
                           key=lambda x: (x != "all", x))
            pairs = [f"{a}_vs_{b}" for a, b in PAIRS[pop]]
            L += [f"## {pop}, a CES sample visible every {10 * k} ms", "",
                  "| rows kept | share of rows | " + " | ".join(p.replace("_vs_", " vs ") for p in pairs) + " |",
                  "|" + "---|" * (2 + len(pairs))]
            for name in names:
                cells = [res["cells"][f"{pop}|s{s}|k{k}|{name}"] for s in R.SEEDS]
                sh = [c["share_of_rows"] for c in cells]
                row = f"| {name} | {min(sh):.3f} to {max(sh):.3f} | "
                out = []
                for p in pairs:
                    xs = [c[p] for c in cells]
                    out.append(f"{min(x[0] for x in xs):+.3f} to {max(x[0] for x in xs):+.3f} "
                               f"({sum(x[1] > 0 for x in xs)}/4)")
                L.append(row + " | ".join(out) + " |")
            L.append("")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
