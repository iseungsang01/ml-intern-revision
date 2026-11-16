"""B.37 addendum 1 (PREREGISTRATION_B37_ADDENDUM1.md): the second stage against a linear stack,
in the inclusive population, and its step time.

CPU only; no network is trained.  Run from the repo root:
    py ces_prediction/experiments/b37_plan/run_b37a.py
Output: docs/paper/b37_addendum1.json / .md
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_b37 as M  # noqa: E402
from run_b37 import R, ci, gbr, inputs, wq  # noqa: E402

OUT_JSON = R.REPO / "docs" / "paper" / "b37_addendum1.json"
OUT_MD = R.REPO / "docs" / "paper" / "b37_addendum1.md"
POP = {"cut": (".b29_norm0_cut_s{s}", ".b1_w2cut_split_s{s}"), "incl": (".b29_norm0_incl_s{s}", ".b1_manifest_s{s}")}


class Ctx(M.Ctx):
    def __init__(self, pop):
        super().__init__()
        self.pop = pop
        if pop != "cut":
            _t, _B, _ls, load_grid_files, _bb, _g, _m = R.imports()
            self.grid, self.dims = load_grid_files(R.DATA, True, R.POPS[pop])
            assert sorted(self.grid) == self.names

    def manifest(self, s):
        return json.loads((R.DATA / POP[self.pop][1].format(s=s) / "split_manifest.json").read_text(encoding="utf-8"))

    def net(self, s):
        run = R.DATA / POP[self.pop][0].format(s=s)
        m = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
        if m["seq_model"] != "v2" or bool(m["per_shot_input_norm"]):
            raise SystemExit(f"FATAL: {run.name} is not the causal variant")
        stats = self.load_stats(m)
        model = self.MODELS["v2"]()
        model.load_state_dict(self.torch.load(run / "weights" / "seq_lstm.pth", map_location="cpu"))
        model.eval()
        return model, stats


def one(C, s, rng, timing):
    man, net = C.manifest(s), C.net(s)
    z = np.load(R.ROWS_DIR / f"rows_{C.pop}_s{s}.npz")
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
        raise SystemExit(f"FATAL: check failed on {C.pop} split {s}: {dev} of {cnt} rows deviate")
    Xv, yv, cv, hv, gv = [], [], [], [], []
    for n_ in sorted(man["val_files"]):
        if n_ not in C.grid:
            continue
        i = C.names.index(n_)
        rw = C.rows(i, net)
        if not len(rw):
            continue
        fr = C.feats(i, mu, sd)[rw[:, 0].astype(int)]
        ok = np.isfinite(fr).all(1) & np.isfinite(rw).all(1)
        Xv.append(inputs(fr[ok], rw[ok, 5], rw[ok, 3], rw[ok, 2], rw[ok, 4]))
        yv.append(rw[ok, 1]); cv.append(rw[ok, 2]); hv.append(rw[ok, 3]); gv.append(rw[ok, 4])
    Xv, yv, cv, hv, gv = (np.concatenate(a) for a in (Xv, yv, cv, hv, gv))
    F = {i: C.feats(i, mu, sd) for i in test}
    fr = np.array([F[int(f)][int(r)] for f, r in zip(tab["file"], tab["row"])])
    ok = k1 & np.all([np.isfinite(tab[e]) for e in ("hold", "gp", "causal", "y")], axis=0) & np.isfinite(fr).all(1)
    y, c, h, g = (tab[e][ok].astype(np.float64) for e in ("y", "causal", "hold", "gp"))
    Xt = inputs(fr[ok], tab["age_ms"][ok], h, c, g)
    uf, f = np.unique(tab["file"][ok].astype(int), return_inverse=True)
    n = len(uf)
    layer = gbr().fit(Xv, yv - cv)
    pa = c + layer.predict(Xt)
    Sv, St = np.column_stack([np.ones(len(yv)), cv, gv, hv]), np.column_stack([np.ones(len(y)), c, g, h])
    w = np.linalg.lstsq(Sv, yv, rcond=None)[0]
    ps = St @ w
    pb = ps + gbr().fit(np.column_stack([Xv, Sv @ w - cv]), yv - Sv @ w).predict(np.column_stack([Xt, ps - c]))
    W = rng.multinomial(n, np.full(n, 1.0 / n), size=M.N_BOOT).astype(float)
    Wr = W[:, f]
    se = {"causal": (c - y) ** 2, "layer": (pa - y) ** 2, "stack": (ps - y) ** 2, "layer_on_stack": (pb - y) ** 2}

    def skill(a, b):
        sa, sb = np.bincount(f, se[a], n), np.bincount(f, se[b], n)
        return [float(1 - sa.sum() / sb.sum()), *ci(1 - (W @ sa) / (W @ sb))]

    o = {"validation_rows": int(len(yv)), "test_rows": int(len(y)), "check_rows": cnt, "check_deviating": dev,
         "stack_weights_intercept_causal_gp_hold": [float(x) for x in w],
         "layer_over_causal": skill("layer", "causal"), "stack_over_causal": skill("stack", "causal"),
         "layer_over_stack": skill("layer", "stack"), "layer_on_stack_over_stack": skill("layer_on_stack", "stack"),
         "layer_on_stack_over_causal": skill("layer_on_stack", "causal"),
         "median_layer_over_causal": [float(np.median(np.abs(pa - y)) / np.median(np.abs(c - y))),
                                      *ci(wq(np.abs(pa - y), Wr) / wq(np.abs(c - y), Wr))]}
    if timing:
        size = gbr().fit(Xv, np.log(np.abs(yv - cv) + 1.0))
        tm = {}
        for name, mdl in (("layer", layer), ("error model", size)):
            meds, p99 = [], []
            for _ in range(5):
                for j in range(200):
                    mdl.predict(Xt[j:j + 1])
                d = np.empty(2000)
                for j in range(2000):
                    t0 = time.perf_counter()
                    mdl.predict(Xt[j:j + 1])
                    d[j] = (time.perf_counter() - t0) * 1000.0
                meds.append(float(np.median(d))); p99.append(float(np.quantile(d, 0.99)))
            tm[name] = {"median_ms": float(np.median(meds)), "max_p99_ms": float(max(p99))}
        o["step_time"] = tm
    return o


def main():
    res = {"registration": "ces_prediction/experiments/PREREGISTRATION_B37_ADDENDUM1.md"}
    for pi, pop in enumerate(("cut", "incl")):
        C = Ctx(pop)
        rng = np.random.default_rng(M.SEED + 10 + pi)
        res[pop] = {}
        for s in R.SEEDS:
            res[pop][str(s)] = one(C, s, rng, timing=(pop == "cut" and s == R.SEEDS[0]))
            print(pop, s, res[pop][str(s)]["layer_over_causal"], res[pop][str(s)]["layer_over_stack"], flush=True)
    res["A1_splits"] = sum(res["cut"][str(s)]["layer_over_stack"][1] > 0 for s in R.SEEDS)
    res["A2_splits"] = sum(res["incl"][str(s)]["layer_over_causal"][1] > 0 for s in R.SEEDS)
    OUT_JSON.write_text(json.dumps(res, indent=1), encoding="utf-8")
    s3 = lambda x: f"{x[0]:+.3f} [{x[1]:+.3f}, {x[2]:+.3f}]"                   # noqa: E731
    f3 = lambda x: f"{x[0]:.3f} [{x[1]:.3f}, {x[2]:.3f}]"                      # noqa: E731
    L = ["# B.37 addendum 1 — the second stage against a linear stack, in the inclusive population, and its step time", "",
         "Registered: PREREGISTRATION_B37_ADDENDUM1.md. CPU, no network trained. Models fitted on the validation discharges "
         "of each split and applied once to its test rows. T_i, natural cadence. Intervals: 95 %, 1,000 resamples of test discharges.", "",
         f"- **A1** (the layer over a linear stack of the causal variant, the causal GP and hold; cut): {res['A1_splits']} of 4 -> "
         f"{'holds' if res['A1_splits'] >= 3 else 'does not hold'}.",
         f"- **A2** (the layer over the causal variant with values above 3 keV kept): {res['A2_splits']} of 4 -> "
         f"{'holds' if res['A2_splits'] >= 3 else 'does not hold'}.", ""]
    for pop, lab in (("cut", "Cut population"), ("incl", "Inclusive population")):
        L += [f"## {lab}", "", "| split | test rows | layer over causal | stack over causal | layer over the stack | "
              "layer fitted on the stack, over the stack | same, over causal | median error, layer over causal |", "|---|---|---|---|---|---|---|---|"]
        for s in R.SEEDS:
            o = res[pop][str(s)]
            L.append(f"| {s} | {o['test_rows']:,} | {s3(o['layer_over_causal'])} | {s3(o['stack_over_causal'])} | "
                     f"{s3(o['layer_over_stack'])} | {s3(o['layer_on_stack_over_stack'])} | {s3(o['layer_on_stack_over_causal'])} | "
                     f"{f3(o['median_layer_over_causal'])} |")
        L.append("")
    tm = res["cut"][str(R.SEEDS[0])]["step_time"]
    L += ["## A4. Step time (one row at a time, one thread, five repetitions of 2,000 calls)", ""]
    for k, v in tm.items():
        L.append(f"- {k}: median {v['median_ms']:.3f} ms, largest 99th percentile {v['max_p99_ms']:.3f} ms.")
    OUT_MD.write_text("\n".join(L) + "\n", encoding="utf-8", newline="\n")
    print("written", OUT_MD, res["A1_splits"], res["A2_splits"])


if __name__ == "__main__":
    main()
