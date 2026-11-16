"""Does a combination of the causal estimators already in hand have a lower error than the offline
GP, which reads the next CES sample?  (post hoc, not registered)

The author's point (2026-11-16): a gain of a few hundredths in skill is enough to pass a reference
the network could not be told apart from.  The pooled interval of the network over the offline GP
is -0.010 [-0.043, +0.014]; `stack_b30.py` found that mixing the network with the causal GP gains
+0.021 to +0.037 over the network alone.  This script puts the two on the same rows: the offline GP
(Matern-3/2, up to 16 neighbours on each side, `baselines_interpolation.predict_gp`) is computed on
the stored B.30 rows (T_i, cut population, natural cadence, rows with a visible sample on both
sides), and each causal estimator and each two-fold stack is scored against it.

CPU; a few minutes per split.  Run from the repo root:
    py ces_prediction/experiments/b30_thinned/stack_offline_b30.py
Output: docs/paper/b30_stack_offline.json / .md
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_b30 as R  # noqa: E402

OUT_JSON = R.REPO / "docs" / "paper" / "b30_stack_offline.json"
OUT_MD = R.REPO / "docs" / "paper" / "b30_stack_offline.md"
STACKS = {"causal stack (causal variant + causal GP + hold + history-only network)": ("causal", "gp", "hold", "hist80"),
          "headline stack (network + causal GP + hold)": ("bb80", "gp", "hold")}
SINGLES = {"causal variant": "causal", "network": "bb80", "causal GP": "gp"}


def main():
    R.pin_env()
    _torch, B, _ls, load_grid_files, _bb, GAP, _m = R.imports()
    grid, _dims = load_grid_files(R.DATA, True, R.POPS["cut"])
    names = sorted(grid)
    rng = np.random.default_rng(R.BOOT_SEED)
    cache = {}

    def weights(n):
        if n not in cache:
            cache[n] = rng.multinomial(n, np.full(n, 1.0 / n), size=R.N_BOOT).astype(np.float64)
        return cache[n]

    res = {"note": "post hoc, not registered; T_i, cut, k = 1, rows with a visible sample on both sides",
           "cells": {}}
    for s in R.SEEDS:
        z = np.load(R.ROWS_DIR / f"rows_cut_s{s}.npz")
        tab = {c.split("__")[1]: z[c] for c in z.files if c.startswith("CES_TI__")}
        cols = sorted({e for m in STACKS.values() for e in m} | set(SINGLES.values()))
        sel = tab["has_future"] & (tab["k"] == 1) & np.all([np.isfinite(tab[c]) for c in cols], axis=0)
        f, r, y = tab["file"][sel], tab["row"][sel], tab["y"][sel].astype(np.float64)
        off = np.full(len(y), np.nan)
        for fi in np.unique(f):
            arr = grid[names[int(fi)]]
            tm = arr[:, 0].astype(np.float64)
            yy = arr[:, 1].astype(np.float64)
            spans = R.block_spans(tm, GAP)
            span_of = np.full(len(tm), -1)
            for j, (a, b) in enumerate(spans):
                span_of[a:b] = j
            for i in np.flatnonzero(f == fi):
                ri = int(r[i])
                a, b = spans[span_of[ri]]
                obs = np.flatnonzero(~np.isnan(yy[a:b])) + a
                obs = obs[obs != ri]
                off[i] = B.predict_gp(tm[obs], yy[obs], tm[ri])
        ok = np.isfinite(off)
        se_off = (off - y) ** 2
        uf = np.unique(f)
        half = np.isin(f, uf[::2])
        cell = {"rows": int(ok.sum()), "discharges": int(len(uf)), "rmse_offline_gp_eV": float(np.sqrt(se_off[ok].mean())),
                "vs_offline_gp": {}}
        for name, key in SINGLES.items():
            se = (tab[key][sel].astype(np.float64) - y) ** 2
            cell["vs_offline_gp"][name] = R.skill_ci(se[ok], se_off[ok], f[ok], weights)
        for name, members in STACKS.items():
            X = np.stack([tab[m][sel].astype(np.float64) for m in members], 1)
            pred = np.empty(len(y))
            for a_, b_ in ((half, ~half), (~half, half)):
                w, *_ = np.linalg.lstsq(X[a_], y[a_], rcond=None)
                pred[b_] = X[b_] @ w
            se = (pred - y) ** 2
            cell["vs_offline_gp"][name] = R.skill_ci(se[ok], se_off[ok], f[ok], weights)
        res["cells"][f"s{s}"] = cell
        print(s, "done", {k: [round(v, 3) for v in x] for k, x in cell["vs_offline_gp"].items()}, flush=True)
    OUT_JSON.write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    L = ["# B.30 rows: the causal estimators and their combinations against the offline GP (post hoc, not registered)", "",
         "T_i, cut population, a CES sample visible every 10 ms, rows with a visible sample on both sides. "
         "Skill = 1 - SSE(estimator) / SSE(offline GP); the offline GP reads up to 16 samples on each side of the row. "
         "95 % interval over 2,000 resamples of discharges. Stack weights are fitted on one half of the test "
         "discharges and applied to the other half, and the reverse.", "",
         "| estimator | " + " | ".join(f"split {s}" for s in R.SEEDS) + " | intervals excluding 0 (above / below) |",
         "|---|" + "---|" * (len(R.SEEDS) + 1)]
    for name in [*SINGLES, *STACKS]:
        xs = [res["cells"][f"s{s}"]["vs_offline_gp"][name] for s in R.SEEDS]
        L.append(f"| {name} | " + " | ".join(f"{x[0]:+.3f} [{x[1]:+.3f}, {x[2]:+.3f}]" for x in xs)
                 + f" | {sum(x[1] > 0 for x in xs)} / {sum(x[2] < 0 for x in xs)} |")
    L += ["", "RMSE of the offline GP on these rows (eV): " + ", ".join(
        f"{res['cells'][f's{s}']['rmse_offline_gp_eV']:.1f}" for s in R.SEEDS) + "."]
    OUT_MD.write_text("\n".join(L) + "\n", encoding="utf-8", newline="\n")
    print("written", OUT_MD)


if __name__ == "__main__":
    main()
