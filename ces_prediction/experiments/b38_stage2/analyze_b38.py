"""B.38 read-outs (PREREGISTRATION_B38.md with amendment 1): the second stage fitted on
cross-fitted predictions of the training discharges.

CPU; no network is trained here.  Needs the 16 runs of run_b38.py.  Run from the repo root:
    py ces_prediction/experiments/b38_stage2/analyze_b38.py
Output: docs/paper/b38_stage2.json / .md
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "b37_plan"))
import run_b37 as M  # noqa: E402
import run_b38 as B  # noqa: E402
from run_b37 import R, ci, gbr, inputs, wq  # noqa: E402

OUT_JSON = R.REPO / "docs" / "paper" / "b38_stage2.json"
OUT_MD = R.REPO / "docs" / "paper" / "b38_stage2.md"


def load_net(C, run):
    m = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
    if m["seq_model"] != "v2" or bool(m["per_shot_input_norm"]) or m.get("per_shot_norm_expanding") or m.get("zero_mc"):
        raise SystemExit(f"FATAL: {run.name} is not the causal variant")
    stats = C.load_stats(m)
    model = C.MODELS["v2"]()
    model.load_state_dict(C.torch.load(run / "weights" / "seq_lstm.pth", map_location="cpu"))
    model.eval()
    return (model, stats), m


def collect(C, files, net, mu, sd):
    """Rows of the named files under one network: inputs, y, causal, hold, gp."""
    X, y, c, h, g = [], [], [], [], []
    for name in files:
        if name not in C.grid:
            continue
        i = C.names.index(name)
        rw = C.rows(i, net)
        if not len(rw):
            continue
        fr = C.feats(i, mu, sd)[rw[:, 0].astype(int)]
        ok = np.isfinite(fr).all(1) & np.isfinite(rw).all(1)
        X.append(inputs(fr[ok], rw[ok, 5], rw[ok, 3], rw[ok, 2], rw[ok, 4]))
        y.append(rw[ok, 1]); c.append(rw[ok, 2]); h.append(rw[ok, 3]); g.append(rw[ok, 4])
    return tuple(np.concatenate(a) for a in (X, y, c, h, g))


def main():
    C = M.Ctx()
    rng = np.random.default_rng(M.SEED + 20)
    res = {"registration": "ces_prediction/experiments/PREREGISTRATION_B38.md (amendment 1)", "splits": {}}
    runs = json.loads((R.DATA / ".b38_runs.json").read_text(encoding="utf-8")) if (R.DATA / ".b38_runs.json").exists() else []
    res["runs"] = [{k: r.get(k) for k in ("seed", "fold", "epochs_run", "best_epoch", "minutes", "status")} for r in runs]
    for s in R.SEEDS:
        man, net = C.manifest(s), C.net(s)
        z = np.load(R.ROWS_DIR / f"rows_cut_s{s}.npz")
        tab = {c.split("__")[1]: z[c] for c in z.files if c.startswith("CES_TI__")}
        test = set(np.unique(tab["file"]).tolist())
        mu, sd = C.fast_stats(test)
        k1 = tab["k"] == 1
        dev = cnt = 0
        for i in sorted(test)[:5]:
            rw = C.rows(i, net)
            m = k1 & (tab["file"] == i)
            lut = {int(r): j for j, r in enumerate(rw[:, 0])}
            for r, cv, hv, gv in zip(tab["row"][m], tab["causal"][m], tab["hold"][m], tab["gp"][m]):
                j = lut.get(int(r))
                cnt += 1
                if j is None or max(abs(rw[j, 2] - cv), abs(rw[j, 3] - hv), abs(rw[j, 4] - gv)) > 0.5:
                    dev += 1
        if cnt == 0 or dev / cnt > 0.01:
            raise SystemExit(f"FATAL: check failed on split {s}: {dev} of {cnt} rows deviate")
        Xv, yv, cv, hv, gv = collect(C, sorted(man["val_files"]), net, mu, sd)
        fs = B.folds(s)
        parts, fold_info = [], []
        for k in range(B.N_FOLDS):
            fnet, fm = load_net(C, B.run_dir(s, k))
            p = collect(C, fs[k], fnet, mu, sd)
            parts.append(p)
            fold_info.append({"fold": k, "discharges": len(fs[k]), "rows": int(len(p[1])),
                              "epochs_run": fm["epochs_run"], "best_epoch": fm["best_epoch"],
                              "skill_over_hold": float(1 - ((p[2] - p[1]) ** 2).sum() / ((p[3] - p[1]) ** 2).sum()),
                              "skill_over_gp": float(1 - ((p[2] - p[1]) ** 2).sum() / ((p[4] - p[1]) ** 2).sum())})
            print(s, "fold", k, fold_info[-1], flush=True)
        Xo, yo, co, ho, go = (np.concatenate([p[j] for p in parts]) for j in range(5))
        F = {i: C.feats(i, mu, sd) for i in test}
        fr = np.array([F[int(f)][int(r)] for f, r in zip(tab["file"], tab["row"])])
        ok = k1 & np.all([np.isfinite(tab[e]) for e in ("hold", "gp", "causal", "y")], axis=0) & np.isfinite(fr).all(1)
        y, c, h, g = (tab[e][ok].astype(np.float64) for e in ("y", "causal", "hold", "gp"))
        Xt = inputs(fr[ok], tab["age_ms"][ok], h, c, g)
        St = np.column_stack([np.ones(len(y)), c, g, h])
        uf, f = np.unique(tab["file"][ok].astype(int), return_inverse=True)
        n = len(uf)
        W = rng.multinomial(n, np.full(n, 1.0 / n), size=M.N_BOOT).astype(float)
        Wr = W[:, f]
        sets = {"val": (Xv, yv, cv, hv, gv), "oof": (Xo, yo, co, ho, go),
                "both": tuple(np.concatenate([a, b]) for a, b in zip((Xo, yo, co, ho, go), (Xv, yv, cv, hv, gv)))}
        se = {"causal": (c - y) ** 2, "hold": (h - y) ** 2, "gp": (g - y) ** 2}
        pred = {}
        for name, (X_, y_, c_, h_, g_) in sets.items():
            pred[name] = c + gbr().fit(X_, y_ - c_).predict(Xt)
            se[f"layer_{name}"] = (pred[name] - y) ** 2
            w = np.linalg.lstsq(np.column_stack([np.ones(len(y_)), c_, g_, h_]), y_, rcond=None)[0]
            se[f"stack_{name}"] = (St @ w - y) ** 2

        def skill(a, b):
            sa, sb = np.bincount(f, se[a], n), np.bincount(f, se[b], n)
            return [float(1 - sa.sum() / sb.sum()), *ci(1 - (W @ sa) / (W @ sb))]

        Xb, yb, cb = sets["both"][0], sets["both"][1], sets["both"][2]
        ecb, ec = np.abs(cb - yb), np.abs(c - y)
        msize = gbr().fit(Xb, np.log(ecb + 1.0))
        size = np.exp(msize.predict(Xt))
        const = np.quantile(ecb, 0.90)
        half = np.quantile(ecb / np.exp(msize.predict(Xb)), 0.90) * size
        hb = rng.random(len(yb)) < 0.5
        oofs = np.zeros(len(yb))
        for tr, te in ((hb, ~hb), (~hb, hb)):
            oofs[te] = np.exp(gbr().fit(Xb[tr], np.log(ecb[tr] + 1.0)).predict(Xb[te]))
        half2 = np.quantile(ecb / oofs, 0.90) * size
        one = np.ones(len(y))
        sums = lambda v: W @ np.bincount(f, v, n)                              # noqa: E731

        def cov(hw):
            cc = (ec <= hw).astype(float)
            return [float(cc.mean()), *ci(sums(cc) / sums(one))]

        el = np.abs(pred["both"] - y)
        o = {"validation_rows": int(len(yv)), "out_of_fold_rows": int(len(yo)), "test_rows": int(len(y)),
             "check_rows": cnt, "check_deviating": dev, "folds": fold_info,
             "full_network_test_skill_over_hold": float(1 - se["causal"].sum() / se["hold"].sum()),
             "full_network_test_skill_over_gp": float(1 - se["causal"].sum() / se["gp"].sum()),
             "G1_both_over_val": skill("layer_both", "layer_val"),
             "G2_both_over_stack": skill("layer_both", "stack_both"),
             "G3_coverage": cov(half), "G3_width_ratio": [float(np.median(half) / const), *ci(wq(half, Wr) / const)],
             "coverage_scale_out_of_fold": cov(half2), "width_ratio_scale_out_of_fold": float(np.median(half2) / const),
             "median_half_width_eV": float(np.median(half)), "constant_half_width_eV": float(const),
             "layer_both_over_causal": skill("layer_both", "causal"), "layer_both_over_gp": skill("layer_both", "gp"),
             "layer_both_over_hold": skill("layer_both", "hold"),
             "layer_val_over_causal": skill("layer_val", "causal"), "layer_oof_over_causal": skill("layer_oof", "causal"),
             "stack_both_over_causal": skill("stack_both", "causal"),
             "median_layer_over_causal": [float(np.median(el) / np.median(ec)), *ci(wq(el, Wr) / wq(ec, Wr))],
             "median_layer_over_hold": [float(np.median(el) / np.median(np.abs(h - y))),
                                        *ci(wq(el, Wr) / wq(np.abs(h - y), Wr))],
             "rmse_eV": {"hold": float(np.sqrt(se["hold"].mean())), "gp": float(np.sqrt(se["gp"].mean())),
                         "causal": float(np.sqrt(se["causal"].mean())), "layer": float(np.sqrt(se["layer_both"].mean()))},
             "median_abs_error_eV": {"hold": float(np.median(np.abs(h - y))), "causal": float(np.median(ec)),
                                     "layer": float(np.median(el))}}
        o["G1_holds"] = bool(o["G1_both_over_val"][1] > 0)
        o["G2_holds"] = bool(o["G2_both_over_stack"][1] > 0)
        o["G3_holds"] = bool(0.87 <= o["G3_coverage"][0] <= 0.93 and o["G3_width_ratio"][2] < 1)
        res["splits"][str(s)] = o
        print(s, "done", o["G1_both_over_val"], o["layer_both_over_causal"], flush=True)
        OUT_JSON.write_text(json.dumps(res, indent=1), encoding="utf-8")
    res["verdict"] = {k: int(sum(res["splits"][str(s)][f"{k}_holds"] for s in R.SEEDS)) for k in ("G1", "G2", "G3")}
    OUT_JSON.write_text(json.dumps(res, indent=1), encoding="utf-8")
    s3 = lambda x: f"{x[0]:+.3f} [{x[1]:+.3f}, {x[2]:+.3f}]"                   # noqa: E731
    f3 = lambda x: f"{x[0]:.3f} [{x[1]:.3f}, {x[2]:.3f}]"                      # noqa: E731
    v = res["verdict"]
    S = lambda s: res["splits"][str(s)]                                         # noqa: E731
    L = ["# B.38 — the second stage fitted on cross-fitted predictions of the training discharges", "",
         "Registered: PREREGISTRATION_B38.md with amendment 1. 16 GPU runs (the causal variant trained on three of four "
         "folds of the training discharges of each split, epoch cap 30); read-outs on a CPU. T_i, cut population, natural "
         "cadence. The correction is applied once to the test rows of the frozen row table. Intervals: 95 %, 1,000 "
         "resamples of test discharges.", "",
         f"- **G1** (the correction fitted on training and validation discharges over the one fitted on validation "
         f"discharges alone): {v['G1']} of 4 -> {'holds' if v['G1'] >= 3 else 'does not hold'}.",
         f"- **G2** (over the linear stack fitted on the same rows): {v['G2']} of 4 -> {'holds' if v['G2'] >= 3 else 'does not hold'}.",
         f"- **G3** (the adaptive 90 % interval covers 0.87 to 0.93 and is shorter than a constant one): {v['G3']} of 4 -> "
         f"{'holds' if v['G3'] >= 3 else 'does not hold'}.", "",
         "| split | rows: out-of-fold / validation / test | G1: both over validation only | G2: both over the stack | "
         "G3: coverage | G3: median half-width over constant | reported: coverage with an out-of-fold scale (width ratio) |",
         "|---|---|---|---|---|---|---|"]
    for s in R.SEEDS:
        o = S(s)
        L.append(f"| {s} | {o['out_of_fold_rows']:,} / {o['validation_rows']:,} / {o['test_rows']:,} | {s3(o['G1_both_over_val'])} | "
                 f"{s3(o['G2_both_over_stack'])} | {f3(o['G3_coverage'])} | {f3(o['G3_width_ratio'])} | "
                 f"{f3(o['coverage_scale_out_of_fold'])} ({o['width_ratio_scale_out_of_fold']:.3f}) |")
    L += ["", "## Reported", "",
          "| split | correction (both) over the causal variant | over the causal GP | over hold | fitted on validation only, over causal | "
          "fitted on out-of-fold rows only, over causal | linear stack (both) over causal | median error, correction over causal | "
          "over hold |", "|---|---|---|---|---|---|---|---|---|"]
    for s in R.SEEDS:
        o = S(s)
        L.append(f"| {s} | {s3(o['layer_both_over_causal'])} | {s3(o['layer_both_over_gp'])} | {s3(o['layer_both_over_hold'])} | "
                 f"{s3(o['layer_val_over_causal'])} | {s3(o['layer_oof_over_causal'])} | {s3(o['stack_both_over_causal'])} | "
                 f"{f3(o['median_layer_over_causal'])} | {f3(o['median_layer_over_hold'])} |")
    L += ["", "| split | RMSE (eV): hold / causal GP / causal variant / corrected | median absolute error (eV): hold / causal variant / corrected | "
          "half-widths (eV): adaptive median / constant |", "|---|---|---|---|"]
    for s in R.SEEDS:
        o = S(s)
        r, m = o["rmse_eV"], o["median_abs_error_eV"]
        L.append(f"| {s} | {r['hold']:.1f} / {r['gp']:.1f} / {r['causal']:.1f} / {r['layer']:.1f} | "
                 f"{m['hold']:.1f} / {m['causal']:.1f} / {m['layer']:.1f} | {o['median_half_width_eV']:.1f} / {o['constant_half_width_eV']:.1f} |")
    L += ["", "## The out-of-fold networks", "",
          "Skill over hold and over the causal GP of each out-of-fold network on its held-out fold of training discharges, "
          "beside that of the full network on the test discharges.", "",
          "| split | full network on test: over hold / over GP | fold 0 | fold 1 | fold 2 | fold 3 | epochs run (best) per fold |", "|---|---|---|---|---|---|---|"]
    for s in R.SEEDS:
        o = S(s)
        L.append(f"| {s} | {o['full_network_test_skill_over_hold']:+.3f} / {o['full_network_test_skill_over_gp']:+.3f} | " +
                 " | ".join(f"{x['skill_over_hold']:+.3f} / {x['skill_over_gp']:+.3f}" for x in o["folds"]) + " | " +
                 ", ".join(f"{x['epochs_run']} ({x['best_epoch']})" for x in o["folds"]) + " |")
    L += ["", "Check before each readout (five test files recomputed against the row table): " +
          ", ".join(f"{S(s)['check_deviating']} of {S(s)['check_rows']:,} rows deviate" for s in R.SEEDS) + "."]
    OUT_MD.write_text("\n".join(L) + "\n", encoding="utf-8", newline="\n")
    print("written", OUT_MD, v)


if __name__ == "__main__":
    main()
