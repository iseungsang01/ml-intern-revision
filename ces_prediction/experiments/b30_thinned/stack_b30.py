"""Is the recurrent estimator the best estimator we already have?  (post hoc, not registered)

Asked by the author (2026-11-16): "our model may not be the best one".  A direct, model-free check
on the stored B.30 row tables: combine estimators that are already scored on the same rows and see
whether the combination has a lower error than the network alone.  If it has, a better estimator
exists without any new training; if it has not, the other estimators carry nothing the network
does not already use.

T_i, natural cadence (k = 1), every measured test row with a visible sample before and after.
Two-fold over discharges inside each test split: least-squares weights (no intercept) fitted on one
half, applied to the other, and the reverse; skill of the stack over its reference with the B.30
discharge-level bootstrap.

  causal stack:   causal variant + causal GP + hold + history-only network   vs the causal variant
  headline stack: network (80-epoch) + causal GP + hold                      vs the network
  offline stack:  network + PCHIP of the two-sided samples (not causal)      vs the network
                  (how much the NEXT sample would add to the network)

CPU, reads data/.b30/rows_*.npz.  Run from the repo root:
    py ces_prediction/experiments/b30_thinned/stack_b30.py
Output: docs/paper/b30_stack.json / .md
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_b30 as R  # noqa: E402

OUT_JSON = R.REPO / "docs" / "paper" / "b30_stack.json"
OUT_MD = R.REPO / "docs" / "paper" / "b30_stack.md"
STACKS = {"causal": (("causal", "gp", "hold", "hist80"), "causal"),
          "headline": (("bb80", "gp", "hold"), "bb80"),
          "offline": (("bb80", "pchip"), "bb80")}
NAMES = {"causal": "causal variant", "gp": "causal GP", "hold": "hold", "hist80": "history-only network",
         "bb80": "network", "pchip": "PCHIP (two-sided)"}


def main():
    rng = np.random.default_rng(R.BOOT_SEED)
    cache = {}

    def weights(n):
        if n not in cache:
            cache[n] = rng.multinomial(n, np.full(n, 1.0 / n), size=R.N_BOOT).astype(np.float64)
        return cache[n]

    res = {"note": "post hoc, not registered; T_i, k = 1, two-fold over test discharges", "cells": {}}
    for pop in ("cut", "incl"):
        for s in R.SEEDS:
            z = np.load(R.ROWS_DIR / f"rows_{pop}_s{s}.npz")
            tab = {c.split("__")[1]: z[c] for c in z.files if c.startswith("CES_TI__")}
            cols = sorted({e for members, _ in STACKS.values() for e in members})
            if any(c not in tab for c in cols):
                continue
            sel = tab["has_future"] & (tab["k"] == 1) & np.all([np.isfinite(tab[c]) for c in cols], axis=0)
            y = tab["y"][sel].astype(np.float64)
            f = tab["file"][sel]
            uf = np.unique(f)
            half = np.isin(f, uf[::2])
            for name, (members, ref) in STACKS.items():
                X = np.stack([tab[m][sel].astype(np.float64) for m in members], 1)
                pred = np.empty(len(y))
                ws = []
                for a, b in ((half, ~half), (~half, half)):
                    w, *_ = np.linalg.lstsq(X[a], y[a], rcond=None)
                    pred[b] = X[b] @ w
                    ws.append(w)
                se_stack = (pred - y) ** 2
                se_ref = (tab[ref][sel].astype(np.float64) - y) ** 2
                sk = R.skill_ci(se_stack, se_ref, f, weights)
                res["cells"][f"{pop}|s{s}|{name}"] = {
                    "rows": int(sel.sum()), "discharges": int(len(uf)), "members": list(members), "reference": ref,
                    "weights_fold_mean": [float(v) for v in np.mean(ws, 0)],
                    "rmse_reference_eV": float(np.sqrt(se_ref.mean())), "rmse_stack_eV": float(np.sqrt(se_stack.mean())),
                    "skill": sk}
            print(pop, s, "done", flush=True)
    OUT_JSON.write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    OUT_MD.write_text(render(res), encoding="utf-8", newline="\n")
    print("written", OUT_MD)


def render(res):
    L = ["# B.30 row tables: does a combination of the stored estimators have a lower error than the network? "
         "(post hoc, not registered)", "",
         "T_i, a CES sample visible every 10 ms. Least-squares weights fitted on one half of the test discharges "
         "and applied to the other half, and the reverse. Skill = 1 - SSE(stack) / SSE(reference), 95 % interval "
         "over 2,000 resamples of discharges.", ""]
    for pop in ("cut", "incl"):
        keys = [k for k in res["cells"] if k.startswith(pop + "|")]
        if not keys:
            continue
        L += [f"## {pop}", "", "| stack | split | weights (mean of the two folds) | RMSE reference -> stack (eV) | "
              "skill of the stack over its reference |", "|---|---|---|---|---|"]
        for name, (members, ref) in STACKS.items():
            for s in R.SEEDS:
                c = res["cells"].get(f"{pop}|s{s}|{name}")
                if not c:
                    continue
                w = ", ".join(f"{NAMES[m]} {v:+.2f}" for m, v in zip(c["members"], c["weights_fold_mean"]))
                sk = c["skill"]
                L.append(f"| {name} (vs {NAMES[ref]}) | {s} | {w} | {c['rmse_reference_eV']:.1f} -> "
                         f"{c['rmse_stack_eV']:.1f} | {sk[0]:+.3f} [{sk[1]:+.3f}, {sk[2]:+.3f}] |")
        L.append("")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
