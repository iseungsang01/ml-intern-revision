"""B.36 (PREREGISTRATION_B36.md): a layer on top of the causal variant -- can its signed error be
corrected from what is known at the row?

CPU only; no network is trained.  Run from the repo root:
    py ces_prediction/experiments/b36_layer/run_b36.py
Output: docs/paper/b36_layer.json / .md
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "b30_thinned"))
sys.path.insert(0, str(HERE.parent / "b33_clusters"))
sys.path.insert(0, str(HERE.parent / "b34_error_model"))
import run_b30 as R  # noqa: E402
from run_b33 import window_feats  # noqa: E402
from run_b34 import ci, wq  # noqa: E402

OUT_JSON = R.REPO / "docs" / "paper" / "b36_layer.json"
OUT_MD = R.REPO / "docs" / "paper" / "b36_layer.md"
SEED, N_BOOT = 20261116, 1000


def gbr():
    return HistGradientBoostingRegressor(random_state=SEED)


def main():
    R.pin_env()
    _t, _B, _ls, load_grid_files, _bb, _gap, _m = R.imports()
    grid, dims = load_grid_files(R.DATA, True, R.POPS["cut"])
    names = sorted(grid)
    nb, ne = dims["bes"], dims["ecei"]
    shot = np.array([int(re.search(r"(\d+)", nm).group(1)) for nm in names])
    seg, tstep = {}, {}
    for i, nm in enumerate(names):
        dt = np.diff(grid[nm][:, 0].astype(np.float64))
        seg[i] = np.concatenate(([0], np.cumsum(dt >= 0.5)))
        tstep[i] = np.concatenate(([np.nan], dt / 0.010))
    rng = np.random.default_rng(SEED)
    res = {"registration": "ces_prediction/experiments/PREREGISTRATION_B36.md", "splits": {}}
    for s in R.SEEDS:
        z = np.load(R.ROWS_DIR / f"rows_cut_s{s}.npz")
        tab = {c.split("__")[1]: z[c] for c in z.files if c.startswith("CES_TI__")}
        test = set(np.unique(tab["file"]).tolist())
        X = np.concatenate([grid[names[i]][:, 3:3 + nb + ne] for i in range(len(names)) if i not in test]).astype(np.float64)
        mu, sd = X.mean(0), X.std(0) + 1e-12
        F = {}
        for i in test:
            a = (grid[names[i]][:, 3:3 + nb + ne].astype(np.float64) - mu) / sd
            F[i] = np.column_stack([window_feats(a[:, :nb].mean(1), seg[i]), window_feats(a[:, nb:].mean(1), seg[i]), tstep[i]])
        fr = np.array([F[int(f)][int(r)] for f, r in zip(tab["file"], tab["row"])])
        ok = ((tab["k"] == 1) & np.all([np.isfinite(tab[e]) for e in ("hold", "gp", "causal", "hist80", "y")], axis=0)
              & np.isfinite(fr).all(1))
        y, c, h, g, tw = (tab[e][ok].astype(np.float64) for e in ("y", "causal", "hold", "gp", "hist80"))
        XA = np.column_stack([fr[ok], tab["age_ms"][ok], h, c - h, c - g, np.abs(c - h), np.abs(c - g)])
        XB = np.column_stack([XA, tw - h])
        S = np.column_stack([np.ones(len(y)), c, g, h, tw])
        uf, f = np.unique(tab["file"][ok].astype(int), return_inverse=True)
        even = (shot[uf] % 2 == 0)[f]
        n = len(uf)
        r = y - c
        pa, pb, size, lin = (np.zeros(len(y)) for _ in range(4))
        for tr, te in ((even, ~even), (~even, even)):
            pa[te] = gbr().fit(XA[tr], r[tr]).predict(XA[te])
            pb[te] = gbr().fit(XB[tr], r[tr]).predict(XB[te])
            size[te] = gbr().fit(XA[tr], np.log(np.abs(r[tr]) + 1.0)).predict(XA[te])
            lin[te] = S[te] @ np.linalg.lstsq(S[tr], y[tr], rcond=None)[0]
        top = np.zeros(len(y), bool)
        for te in (even, ~even):
            top[te] = size[te] >= np.quantile(size[te], 0.90)
        W = rng.multinomial(n, np.full(n, 1.0 / n), size=N_BOOT).astype(float)
        se = {"causal": r ** 2, "A": (r - pa) ** 2, "B": (r - pb) ** 2, "gated": (r - np.where(top, pa, 0.0)) ** 2,
              "stack": (y - lin) ** 2}

        def skill(a, b):
            sa, sb = np.bincount(f, se[a], n), np.bincount(f, se[b], n)
            return [float(1 - sa.sum() / sb.sum()), *ci(1 - (W @ sa) / (W @ sb))]

        Wr = W[:, f]
        med = wq(np.abs(r - pa), Wr) / wq(np.abs(r), Wr)
        cb = []
        idx = [np.flatnonzero(f == i) for i in range(n)]
        for _ in range(300):
            q = np.concatenate([idx[i] for i in rng.integers(0, n, n)])
            q = q[top[q]]
            cb.append(np.corrcoef(pa[q], r[q])[0, 1] if len(q) > 10 else np.nan)
        out = {"rows": int(len(y)), "L1_layerA_over_causal": skill("A", "causal"), "L2_layerB_over_causal": skill("B", "causal"),
               "gated_over_causal": skill("gated", "causal"), "stack_over_causal": skill("stack", "causal"),
               "layerA_over_stack": skill("A", "stack"), "layerB_over_stack": skill("B", "stack"),
               "median_layerA_over_causal": [float(np.median(np.abs(r - pa)) / np.median(np.abs(r))), *ci(med)],
               "corr_top_tenth": [float(np.corrcoef(pa[top], r[top])[0, 1]), *ci(np.array(cb))],
               "rmse_causal_eV": float(np.sqrt((r ** 2).mean()))}
        out["L1_holds"] = bool(out["L1_layerA_over_causal"][1] > 0)
        out["L2_holds"] = bool(out["L2_layerB_over_causal"][1] > 0)
        res["splits"][str(s)] = out
        print(s, "done", out["L1_layerA_over_causal"], out["L2_layerB_over_causal"], flush=True)
    res["verdict"] = {k: int(sum(res["splits"][str(s)][f"{k}_holds"] for s in R.SEEDS)) for k in ("L1", "L2")}
    OUT_JSON.write_text(json.dumps(res, indent=1), encoding="utf-8")
    s3 = lambda x: f"{x[0]:+.3f} [{x[1]:+.3f}, {x[2]:+.3f}]"                   # noqa: E731
    f3 = lambda x: f"{x[0]:.3f} [{x[1]:.3f}, {x[2]:.3f}]"                      # noqa: E731
    v = res["verdict"]
    L = ["# B.36 — a layer on top of the causal variant (registered: PREREGISTRATION_B36.md)", "",
         "CPU, no network trained. Frozen B.30 row tables, T_i, cut population, natural cadence. A gradient-boosted model "
         "of the signed residual on the inputs known at the row, fitted on one parity fold of the test discharges and "
         "applied to the other. Skill = 1 - MSE ratio. Intervals: 95 %, 1,000 resamples of discharges.", "",
         f"- **L1** (layer A, 15 inputs, over the causal variant): {v['L1']} of 4 splits -> {'holds' if v['L1'] >= 3 else 'does not hold'}.",
         f"- **L2** (layer B, with the history-only twin added, over the causal variant): {v['L2']} of 4 splits -> "
         f"{'holds' if v['L2'] >= 3 else 'does not hold'}.", "",
         "| split | rows | L1: layer A over causal | L2: layer B over causal | gated layer over causal | linear stack over causal | "
         "layer A over the stack | layer B over the stack | median error, layer A over causal | correlation of predicted and actual "
         "residual in the top tenth |", "|---|---|---|---|---|---|---|---|---|---|"]
    for s in R.SEEDS:
        o = res["splits"][str(s)]
        L.append(f"| {s} | {o['rows']:,} | {s3(o['L1_layerA_over_causal'])} | {s3(o['L2_layerB_over_causal'])} | "
                 f"{s3(o['gated_over_causal'])} | {s3(o['stack_over_causal'])} | {s3(o['layerA_over_stack'])} | "
                 f"{s3(o['layerB_over_stack'])} | {f3(o['median_layerA_over_causal'])} | {s3(o['corr_top_tenth'])} |")
    OUT_MD.write_text("\n".join(L) + "\n", encoding="utf-8", newline="\n")
    print("written", OUT_MD, v)


if __name__ == "__main__":
    main()
