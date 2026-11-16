"""B.34 (PREREGISTRATION_B34.md): can the estimator know in advance which rows it will get wrong?

CPU only; no network is trained.  Reads the frozen B.30 row tables and the shot files.
Run from the repo root:
    py ces_prediction/experiments/b34_error_model/run_b34.py
Output: docs/paper/b34_error_model.json / .md
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "b30_thinned"))
sys.path.insert(0, str(HERE.parent / "b33_clusters"))
import run_b30 as R  # noqa: E402
from run_b33 import FEATS, window_feats  # noqa: E402

OUT_JSON = R.REPO / "docs" / "paper" / "b34_error_model.json"
OUT_MD = R.REPO / "docs" / "paper" / "b34_error_model.md"
SEED, N_BOOT, CHUNK = 20261116, 1000, 100
NAMES = (*FEATS, "age of the last CES sample", "hold value", "causal - hold", "causal - GP",
         "|causal - hold|", "|causal - GP|")


def ci(b):
    b = np.asarray(b, float)
    b = b[np.isfinite(b)]
    return [float(np.quantile(b, 0.025)), float(np.quantile(b, 0.975))]


def wq(v, w, q=0.5):
    """Weighted quantile of v (n,) for every row of w (reps, n)."""
    o = np.argsort(v, kind="stable")
    vs = v[o]
    out = np.empty(w.shape[0])
    for a in range(0, w.shape[0], CHUNK):
        c = np.cumsum(w[a:a + CHUNK][:, o], 1)
        out[a:a + CHUNK] = vs[(c >= q * c[:, -1:]).argmax(1)]
    return out


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
    res = {"registration": "ces_prediction/experiments/PREREGISTRATION_B34.md", "inputs": list(NAMES), "splits": {}}
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
        ok = (tab["k"] == 1) & np.all([np.isfinite(tab[e]) for e in ("hold", "gp", "causal", "y")], axis=0) & np.isfinite(fr).all(1)
        y, c, h, g = (tab[e][ok].astype(np.float64) for e in ("y", "causal", "hold", "gp"))
        Xf = np.column_stack([fr[ok], tab["age_ms"][ok], h, c - h, c - g, np.abs(c - h), np.abs(c - g)])
        uf, f = np.unique(tab["file"][ok].astype(int), return_inverse=True)
        even = (shot[uf] % 2 == 0)[f]
        n = len(uf)
        ec, eh = np.abs(c - y), np.abs(h - y)
        size, size_nofast, sw, half, const = (np.zeros(len(y)) for _ in range(5))
        imp = np.zeros(Xf.shape[1])
        for tr, te in ((even, ~even), (~even, even)):
            m = gbr().fit(Xf[tr], np.log(ec[tr] + 1.0))
            size[te] = np.exp(m.predict(Xf[te]))
            s_tr = np.exp(m.predict(Xf[tr]))
            half[te] = np.quantile(ec[tr] / s_tr, 0.90) * size[te]
            const[te] = np.quantile(ec[tr], 0.90)
            imp += permutation_importance(m, Xf[te], np.log(ec[te] + 1.0), n_repeats=3, random_state=SEED).importances_mean / 2
            size_nofast[te] = np.exp(gbr().fit(Xf[tr][:, 9:], np.log(ec[tr] + 1.0)).predict(Xf[te][:, 9:]))
            sw[te] = gbr().fit(Xf[tr], eh[tr] - ec[tr]).predict(Xf[te])
        use = sw > 0
        es = np.where(use, ec, eh)
        W = rng.multinomial(n, np.full(n, 1.0 / n), size=N_BOOT).astype(float)
        Wr = W[:, f]

        def sums(v):
            return W @ np.bincount(f, v, n)

        def share(sz):
            top = sz >= np.quantile(sz, 0.90)
            return [float((ec[top] ** 2).sum() / (ec ** 2).sum()), *ci(sums(ec ** 2 * top) / sums(ec ** 2))]

        cov = (ec <= half).astype(float)
        ratio = half / const
        med_c, med_h, med_s = wq(ec, Wr), wq(eh, Wr), wq(es, Wr)
        skill_sc = 1 - sums(es ** 2) / sums(ec ** 2)
        skill_sh = 1 - sums(es ** 2) / sums(eh ** 2)
        out = {"rows": int(len(y)), "discharges": int(n),
               "E1_share_top_tenth": share(size), "E1_share_without_fast_channels": share(size_nofast),
               "share_top_tenth_by_abs_causal_minus_hold": share(np.abs(c - h)),
               "E2_coverage": [float(cov.mean()), *ci(sums(cov) / sums(np.ones(len(y))))],
               "E2_width_ratio": [float(np.median(ratio)), *ci(wq(ratio, Wr))],
               "median_half_width_eV": float(np.median(half)), "constant_half_width_eV": float(np.median(const)),
               "E3_median_switch_over_causal": [float(np.median(es) / np.median(ec)), *ci(med_s / med_c)],
               "E3_median_switch_over_hold": [float(np.median(es) / np.median(eh)), *ci(med_s / med_h)],
               "E3_skill_switch_over_causal": [float(1 - (es ** 2).sum() / (ec ** 2).sum()), *ci(skill_sc)],
               "skill_switch_over_hold": [float(1 - (es ** 2).sum() / (eh ** 2).sum()), *ci(skill_sh)],
               "share_rows_using_causal": float(use.mean()),
               "median_abs_error_eV": {"causal": float(np.median(ec)), "hold": float(np.median(eh)), "switch": float(np.median(es))},
               "importance": {NAMES[j]: float(imp[j]) for j in np.argsort(-imp)[:6]}}
        out["E1_holds"] = bool(out["E1_share_top_tenth"][1] > 0.30)
        out["E2_holds"] = bool(0.87 <= out["E2_coverage"][0] <= 0.93 and out["E2_width_ratio"][2] < 1)
        out["E3_holds"] = bool(out["E3_median_switch_over_causal"][2] < 1 and out["E3_median_switch_over_hold"][2] < 1
                               and out["E3_skill_switch_over_causal"][1] > -0.02)
        res["splits"][str(s)] = out
        print(s, "done", {k: out[k] for k in ("E1_holds", "E2_holds", "E3_holds")}, flush=True)
    res["verdict"] = {k: int(sum(res["splits"][str(s)][f"{k}_holds"] for s in R.SEEDS)) for k in ("E1", "E2", "E3")}
    OUT_JSON.write_text(json.dumps(res, indent=1), encoding="utf-8")
    f3 = lambda x: f"{x[0]:.3f} [{x[1]:.3f}, {x[2]:.3f}]"                      # noqa: E731
    s3 = lambda x: f"{x[0]:+.3f} [{x[1]:+.3f}, {x[2]:+.3f}]"                   # noqa: E731
    v = res["verdict"]
    L = ["# B.34 — predicting the size of the error (registered: PREREGISTRATION_B34.md)", "",
         "CPU, no network trained. Frozen B.30 row tables, T_i, cut population, natural cadence. Gradient-boosted "
         "models on 15 inputs known at the row, fitted on one parity fold of the test discharges and applied to the "
         "other. Intervals: 95 %, 1,000 resamples of discharges.", "",
         f"- **E1** (the tenth of rows with the largest predicted size holds more than 0.30 of the squared error): {v['E1']} of 4 splits "
         f"-> {'holds' if v['E1'] >= 3 else 'does not hold'}.",
         f"- **E2** (the adaptive 90 % interval covers 0.87 to 0.93 and is shorter than a constant one): {v['E2']} of 4 splits "
         f"-> {'holds' if v['E2'] >= 3 else 'does not hold'}.",
         f"- **E3** (a switch between hold and the causal variant lowers the median error against both without losing the mean): "
         f"{v['E3']} of 4 splits -> {'holds' if v['E3'] >= 3 else 'does not hold'}.", "",
         "| split | rows | E1: share of squared error in the top tenth | same, without the fast-channel inputs | same, ranked by abs(causal - hold) alone | "
         "E2: coverage | E2: median half-width over constant | median half-width (eV) | constant half-width (eV) |", "|---|---|---|---|---|---|---|---|---|"]
    for s in R.SEEDS:
        o = res["splits"][str(s)]
        L.append(f"| {s} | {o['rows']:,} | {f3(o['E1_share_top_tenth'])} | {f3(o['E1_share_without_fast_channels'])} | "
                 f"{f3(o['share_top_tenth_by_abs_causal_minus_hold'])} | {f3(o['E2_coverage'])} | {f3(o['E2_width_ratio'])} | "
                 f"{o['median_half_width_eV']:.1f} | {o['constant_half_width_eV']:.1f} |")
    L += ["", "| split | E3: median error, switch over causal | switch over hold | skill of the switch over causal | over hold | "
          "rows using the causal variant | median abs error: causal / hold / switch (eV) |", "|---|---|---|---|---|---|---|"]
    for s in R.SEEDS:
        o = res["splits"][str(s)]
        m = o["median_abs_error_eV"]
        L.append(f"| {s} | {f3(o['E3_median_switch_over_causal'])} | {f3(o['E3_median_switch_over_hold'])} | "
                 f"{s3(o['E3_skill_switch_over_causal'])} | {s3(o['skill_switch_over_hold'])} | {o['share_rows_using_causal']:.3f} | "
                 f"{m['causal']:.1f} / {m['hold']:.1f} / {m['switch']:.1f} |")
    L += ["", "Inputs the error model uses most (permutation importance, drop of R2 on the applied fold):", ""]
    for s in R.SEEDS:
        L.append(f"- split {s}: " + "; ".join(f"{k} {x:.3f}" for k, x in res["splits"][str(s)]["importance"].items()))
    OUT_MD.write_text("\n".join(L) + "\n", encoding="utf-8", newline="\n")
    print("written", OUT_MD, v)


if __name__ == "__main__":
    main()
