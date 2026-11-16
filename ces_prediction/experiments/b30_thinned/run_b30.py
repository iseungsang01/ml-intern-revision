"""B.30 thinned-CES replay (PREREGISTRATION_B30.md): frozen estimators fill the 10 ms grid when a
CES sample is visible only every k-th frame, scored against the withheld measurements.

CPU inference only, nothing is trained and no frozen artifact is written. Run from the repo root:
    py ces_prediction/experiments/b30_thinned/run_b30.py rows POP SEED    # row table of one split
    py ces_prediction/experiments/b30_thinned/run_b30.py analyze          # json + md from the tables
POP is cut or incl, SEED one of 42, 1, 7, 123. Row tables go to data/.b30/ (gitignored).
"""
from __future__ import annotations

import json
import os
import sys
import time as _time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
CES_DIR = REPO / "ces_prediction"
SEQ_DIR = CES_DIR / "experiments" / "seq"
DATA = REPO / "data"
ROWS_DIR = DATA / ".b30"
OUT_JSON = REPO / "docs" / "paper" / "b30_thinned.json"
OUT_MD = REPO / "docs" / "paper" / "b30_thinned.md"

SEEDS = (42, 1, 7, 123)
POPS = {"cut": 3000.0, "incl": 0.0}
KS = (1, 2, 3, 5, 10)
# pop -> (backbone run with the stored errors, split dir); run_b29.CONTROLS, file-level family
CONTROLS = {"cut": (".b1_seqv2_s{s}_i{s}", ".b1_w2cut_split_s{s}"),
            "incl": (".b5i_seqv2_s{s}", ".b1_manifest_s{s}")}
# pop -> name -> (run dir, model variant, per-discharge standardization)
NETS = {
    "cut": {"bb": (".b1_seqv2_s{s}_i{s}", "v2", True), "bb80": (".b29a1_v2_cut_s{s}", "v2", True),
            "hist80": (".b29a1_v2h_cut_s{s}", "v2h", True),
            "causal": (".b29_norm0_cut_s{s}", "v2", False)},
    "incl": {"bb": (".b5i_seqv2_s{s}", "v2", True), "hist30": (".b29_v2h_incl_s{s}", "v2h", True),
             "causal": (".b29_norm0_incl_s{s}", "v2", False)},
}
TARGETS = ("CES_TI", "CES_VT")
GP_SIDE = 16
N_BOOT = 2000
BOOT_SEED = 20261116
CHECK_REL = 1e-3
CHECK_ABS = {"CES_TI": 0.5, "CES_VT": 0.5}     # eV, km/s (amendment 1 of the registration)
CHECK_SHARE = 0.99


def pin_env():
    """Every CES_* variable this batch depends on is set here; inherited ones are dropped."""
    for k in [k for k in os.environ if k.startswith("CES_")]:
        del os.environ[k]
    os.environ.update({"CES_DATA_DIR": str(DATA), "CES_DROP_STUCK_TARGETS": "1",
                       "CES_SEQ_NORM_EXPANDING": "0", "CES_SEQ_ZERO_MC": "0",
                       "CES_PER_SHOT_NORM": "0"})   # the mode is passed per network, never read


def imports():
    sys.path.insert(0, str(SEQ_DIR))
    sys.path.insert(1, str(CES_DIR))
    import torch
    import baselines_interpolation as B
    from evaluate import _load_stats
    from seq_data import load_grid_files, build_blocks
    from dataset import STUCK_GAP_SECONDS
    from seq_models import SEQ_MODELS
    torch.set_num_threads(1)
    return torch, B, _load_stats, load_grid_files, build_blocks, STUCK_GAP_SECONDS, SEQ_MODELS


def block_spans(time, gap):
    """(a, b) spans of seq_data.build_blocks (blocks shorter than 2 rows are skipped there too)."""
    bounds = [0]
    for i in range(1, len(time)):
        d = time[i] - time[i - 1]
        if d >= gap or d <= 0:
            bounds.append(i)
    bounds.append(len(time))
    return [(a, b) for a, b in zip(bounds[:-1], bounds[1:]) if b - a >= 2]


def visible_rows(n, spans, k, o):
    vis = np.zeros(n, dtype=bool)
    for a, b in spans:
        vis[a:b] = ((np.arange(b - a) - o) % k) == 0
    return vis


def match_stored(y_mine, se_mine, y_ref, se_ref):
    """Share of stored rows whose squared error is reproduced (fig_trace.check matching)."""
    order = np.argsort(y_mine)
    ys = y_mine[order]
    pos = np.clip(np.searchsorted(ys, y_ref), 1, len(ys) - 1)
    left, right = order[pos - 1], order[pos]
    row = np.where(np.abs(y_mine[left] - y_ref) <= np.abs(y_mine[right] - y_ref), left, right)
    rel = np.abs(se_mine[row] - se_ref) / np.maximum(se_ref, 1.0)
    dev = np.abs(np.sqrt(se_mine[row]) - np.sqrt(se_ref))      # absolute-error deviation
    return rel <= CHECK_REL, dev


def rows(pop, seed):
    pin_env()
    torch, B, _load_stats, load_grid_files, build_blocks, GAP, SEQ_MODELS = imports()
    t0 = _time.time()
    base_grid = tuple(B._GP_LS_GRID)
    bb_dir, split_dir = (DATA / p.format(s=seed) for p in CONTROLS[pop])
    manifest = json.loads((split_dir / "split_manifest.json").read_text(encoding="utf-8"))
    grid, dims = load_grid_files(DATA, True, POPS[pop])
    valid_names = sorted(grid)
    test = [n for n in sorted(manifest["test_files"]) if n in grid]
    nets = {}
    for name, (d, variant, per_shot) in NETS[pop].items():
        run = DATA / d.format(s=seed)
        m = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
        if m["seq_model"] != variant or bool(m["per_shot_input_norm"]) != per_shot \
                or m.get("per_shot_norm_expanding") or m.get("zero_mc") \
                or m.get("extra_vt_channels") or m.get("extra_ti_channels") or m.get("train_ctx"):
            raise SystemExit(f"FATAL: {run.name} is not the registered {name} treatment")
        stats = _load_stats(m)
        model = SEQ_MODELS[variant]()
        model.load_state_dict(torch.load(run / "weights" / "seq_lstm.pth", map_location="cpu"))
        model.eval()
        nets[name] = (model, stats, per_shot,
                      torch.as_tensor(stats["target"]["mean"], dtype=torch.float32),
                      torch.as_tensor(stats["target"]["std"], dtype=torch.float32))
    print(f"[{_time.time() - t0:5.0f}s] {pop} s{seed}: {len(test)} TEST files, nets {list(nets)}",
          flush=True)

    cols = ["file", "k", "o", "row", "age_ms", "has_future", "y", "hold", "gp", "gp_unscaled",
            "pchip", *nets]
    acc = {t: {c: [] for c in cols} for t in TARGETS}
    stored = np.load(bb_dir / "comparison_errors_test.npz")
    with torch.no_grad():
        for k in KS:
            for o in range(k):
                for name in test:
                    arr = grid[name]
                    fidx = valid_names.index(name)
                    tm = arr[:, 0].astype(np.float64)
                    tgt = arr[:, 1:3].astype(np.float64)
                    n = len(tm)
                    spans = block_spans(tm, GAP)
                    vis = visible_rows(n, spans, k, o)
                    arr_h = arr.copy()
                    arr_h[~vis, 1:3] = np.nan
                    pred = {}
                    for nm, (model, stats, per_shot, mean, std) in nets.items():
                        out = np.full((n, 2), np.nan)
                        blks = build_blocks(arr_h, dims, stats, per_shot_norm=per_shot,
                                            expanding=False)
                        if len(blks) != len(spans):
                            raise SystemExit("FATAL: block spans do not match build_blocks")
                        for blk, (a, b) in zip(blks, spans):
                            x = torch.from_numpy(blk["x"]).unsqueeze(0)
                            out[a:b] = (model(x)[0] * std + mean).numpy()
                        pred[nm] = out
                    for ti, t in enumerate(TARGETS):
                        y = tgt[:, ti]
                        A = acc[t]
                        for a, b in spans:
                            obs = np.flatnonzero(~np.isnan(y[a:b])) + a
                            vo = obs[vis[obs]]
                            if vo.size == 0:
                                continue
                            tv, vv = tm[vo], y[vo]
                            for r in obs:
                                pos = int(np.searchsorted(vo, r))
                                if pos == 0:
                                    continue
                                fut0 = pos + 1 if (pos < vo.size and vo[pos] == r) else pos
                                has_future = fut0 < vo.size
                                tr = tm[r]
                                lo = max(0, pos - GP_SIDE)
                                B._GP_LS_GRID = tuple(ls * k for ls in base_grid)
                                gp = B.predict_gp_causal(tv[lo:pos], vv[lo:pos], tr)
                                gp0 = np.nan
                                if k == 5:
                                    B._GP_LS_GRID = base_grid
                                    gp0 = B.predict_gp_causal(tv[lo:pos], vv[lo:pos], tr)
                                pch = np.nan
                                if has_future:
                                    idx = [*range(max(0, pos - 4), pos),
                                           *range(fut0, min(vo.size, fut0 + 4))]
                                    pch = B.predict_pchip(tv[idx], vv[idx], tr)
                                A["file"].append(fidx)
                                A["k"].append(k)
                                A["o"].append(o)
                                A["row"].append(r)
                                A["age_ms"].append((tr - tv[pos - 1]) * 1000.0)
                                A["has_future"].append(has_future)
                                A["y"].append(y[r])
                                A["hold"].append(vv[pos - 1])
                                A["gp"].append(gp)
                                A["gp_unscaled"].append(gp0)
                                A["pchip"].append(pch)
                                for nm in nets:
                                    A[nm].append(pred[nm][r, ti])
                B._GP_LS_GRID = base_grid
                print(f"[{_time.time() - t0:5.0f}s] {pop} s{seed}: k {k} phase {o} done, "
                      f"{len(acc['CES_TI']['y'])} T_i rows so far", flush=True)
            if k == 1:
                # registered check: stored squared errors reproduced on the stored scored rows
                res = {}
                for t in TARGETS:
                    A = {c: np.asarray(acc[t][c], dtype=np.float64) for c in ("file", "y", "hold",
                                                                              "gp", "bb")}
                    labs = ("model", "persistence", "gp_causal")
                    ok = {lab: [] for lab in labs}
                    dv = {lab: [] for lab in labs}
                    sh = stored[f"{t}_shot"]
                    for f in np.unique(sh):
                        sel, mine = sh == f, A["file"] == f
                        if mine.sum() < 2:
                            for lab in labs:
                                ok[lab].append(np.zeros(sel.sum(), bool))
                                dv[lab].append(np.full(sel.sum(), np.inf))
                            continue
                        yr = stored[f"{t}_y_true"][sel].astype(np.float64)
                        for lab, est in (("model", "bb"), ("persistence", "hold"),
                                         ("gp_causal", "gp")):
                            se = (A[est][mine] - A["y"][mine]) ** 2
                            o_, d_ = match_stored(A["y"][mine], se, yr,
                                                  stored[f"{t}_se_{lab}"][sel].astype(np.float64))
                            ok[lab].append(o_)
                            dv[lab].append(d_)
                    res[t] = {"stored_rows": int(len(sh))}
                    for lab in labs:
                        d_ = np.concatenate(dv[lab])
                        res[t][lab] = float(np.concatenate(ok[lab]).mean())
                        res[t][lab + "_abs"] = float((d_ <= CHECK_ABS[t]).mean())
                        res[t][lab + "_dev_median"] = float(np.median(d_))
                        res[t][lab + "_dev_p99"] = float(np.quantile(d_, 0.99))
                print(f"k = 1 check {pop} s{seed}: {json.dumps(res)}", flush=True)
                ROWS_DIR.mkdir(exist_ok=True)
                (ROWS_DIR / f"check_{pop}_s{seed}.json").write_text(json.dumps(res, indent=1))
                if min(res["CES_TI"][lab + "_abs"] for lab in ("model", "persistence", "gp_causal")) < CHECK_SHARE:
                    raise SystemExit("FATAL: k = 1 does not reproduce the stored errors; batch stops")
    ROWS_DIR.mkdir(exist_ok=True)
    out = {}
    for t in TARGETS:
        for c in cols:
            dt = np.int32 if c in ("file", "k", "o", "row") else bool if c == "has_future" else np.float32
            out[f"{t}__{c}"] = np.asarray(acc[t][c], dtype=dt)
    np.savez_compressed(ROWS_DIR / f"rows_{pop}_s{seed}.npz", **out)
    print(f"[{_time.time() - t0:5.0f}s] {pop} s{seed}: saved", flush=True)


# --------------------------------------------------------------------------- analysis

PAIRS = {
    "cut": [("bb", "hold"), ("bb", "gp"), ("bb", "pchip"), ("bb80", "hist80"), ("bb", "hist80"),
            ("hist80", "hold"), ("gp", "hold"), ("pchip", "hold"), ("causal", "gp"),
            ("causal", "hold"), ("bb", "causal")],
    "incl": [("bb", "hold"), ("bb", "gp"), ("bb", "pchip"), ("bb", "hist30"), ("hist30", "hold"),
             ("gp", "hold"), ("pchip", "hold"), ("causal", "gp"), ("causal", "hold"),
             ("bb", "causal")],
}


def skill_ci(se_a, se_b, files, rng_w):
    """1 - SSE_a / SSE_b with a discharge-level bootstrap interval."""
    uf, inv = np.unique(files, return_inverse=True)
    sa = np.bincount(inv, se_a, len(uf))
    sb = np.bincount(inv, se_b, len(uf))
    W = rng_w(len(uf))
    bs = 1.0 - (W @ sa) / (W @ sb)
    lo, hi = np.quantile(bs, [0.025, 0.975])
    return [float(1.0 - sa.sum() / sb.sum()), float(lo), float(hi)]


def cell(tab, sel, pop, weights):
    ests = [c for c in ("hold", "gp", "pchip", *NETS[pop])]
    se = {e: (tab[e][sel] - tab["y"][sel]).astype(np.float64) ** 2 for e in ests}
    files = tab["file"][sel]
    out = {"rows": int(sel.sum()), "discharges": int(len(np.unique(files))),
           "rmse": {e: float(np.sqrt(se[e].mean())) for e in ests},
           "median_abs": {e: float(np.median(np.sqrt(se[e]))) for e in ests},
           "skill": {f"{a}_vs_{b}": skill_ci(se[a], se[b], files, weights) for a, b in PAIRS[pop]}}
    thr = np.quantile(se["hold"], 0.99)
    keep = se["hold"] <= thr
    out["skill_trim1"] = {f"{a}_vs_{b}": skill_ci(se[a][keep], se[b][keep], files[keep], weights)
                          for a, b in PAIRS[pop]}
    return out


def analyze():
    rng = np.random.default_rng(BOOT_SEED)
    cache = {}

    def weights(n):
        if n not in cache:
            cache[n] = rng.multinomial(n, np.full(n, 1.0 / n), size=N_BOOT).astype(np.float64)
        return cache[n]

    res = {"registration": "ces_prediction/experiments/PREREGISTRATION_B30.md", "ks": list(KS),
           "check": {}, "cells": {}}
    for pop in POPS:
        for s in SEEDS:
            p = ROWS_DIR / f"rows_{pop}_s{s}.npz"
            if not p.exists():
                raise SystemExit(f"missing {p.name}")
            z = np.load(p)
            res["check"][f"{pop}_s{s}"] = json.loads((ROWS_DIR / f"check_{pop}_s{s}.json").read_text())
            for t in TARGETS:
                tab = {c.split("__")[1]: z[c] for c in z.files if c.startswith(t + "__")}
                ests = ["hold", "gp", "pchip", *NETS[pop]]
                fin = tab["has_future"] & np.all([np.isfinite(tab[e]) for e in ests], axis=0)
                key = f"{pop}|s{s}|{t}"
                res["cells"][key] = {"by_k": {}, "by_age_k10": {}, "gp_unscaled_k5": None}
                for k in KS:
                    res["cells"][key]["by_k"][str(k)] = cell(tab, fin & (tab["k"] == k), pop, weights)
                k10 = fin & (tab["k"] == 10)
                age = np.rint(tab["age_ms"] / 10.0).astype(int)
                for a in range(1, 11):
                    sel = k10 & (age == a)
                    if sel.sum() >= 200:
                        res["cells"][key]["by_age_k10"][str(a * 10)] = cell(tab, sel, pop, weights)
                k5 = fin & (tab["k"] == 5) & np.isfinite(tab["gp_unscaled"])
                se_bb = (tab["bb"][k5] - tab["y"][k5]).astype(np.float64) ** 2
                se_g0 = (tab["gp_unscaled"][k5] - tab["y"][k5]).astype(np.float64) ** 2
                res["cells"][key]["gp_unscaled_k5"] = {
                    "rmse": float(np.sqrt(se_g0.mean())),
                    "bb_vs_gp_unscaled": skill_ci(se_bb, se_g0, tab["file"][k5], weights)}
    res["verdicts"] = verdicts(res)
    OUT_JSON.write_text(json.dumps(res, indent=1), encoding="utf-8")
    OUT_MD.write_text(render(res), encoding="utf-8", newline="\n")
    print(json.dumps(res["verdicts"], indent=1))


def verdicts(res):
    def sk(s, k, pair, t="CES_TI", pop="cut"):
        return res["cells"][f"{pop}|s{s}|{t}"]["by_k"][str(k)]["skill"][pair]

    v = {}
    a = [sk(s, 5, "bb80_vs_hist80") for s in SEEDS]
    b = [sk(s, 1, "bb80_vs_hist80") for s in SEEDS]
    above = sum(x[1] > 0 for x in a)
    grows = sum(x[0] > y[0] for x, y in zip(a, b))
    v["H30-1"] = {"above_0_at_k5": above, "k5_point_above_k1": grows,
                  "holds": bool(above >= 3 and grows >= 3), "k5": a, "k1": b}
    a = [sk(s, 5, "bb_vs_hold") for s in SEEDS]
    b = [sk(s, 1, "bb_vs_hold") for s in SEEDS]
    kept = sum(x[0] >= y[0] - 0.05 for x, y in zip(a, b))
    v["H30-2"] = {"not_below_k1_minus_0.05": kept, "holds": bool(kept >= 3), "k5": a, "k1": b}
    for h, pair in (("H30-3", "bb_vs_gp"), ("H30-4", "causal_vs_gp"), ("H30-5", "bb_vs_pchip")):
        a = [sk(s, 5, pair) for s in SEEDS]
        n = sum(x[1] > 0 for x in a)
        v[h] = {"above_0_at_k5": n, "holds": bool(n >= 3), "k5": a}
    return v


def f3(x):
    return f"{x[0]:+.3f} [{x[1]:+.3f}, {x[2]:+.3f}]" + ("" if x[1] <= 0 <= x[2] else " *")


def render(res):
    L = ["# B.30 thinned-CES replay (registration PREREGISTRATION_B30.md)", "",
         "Frozen estimators; a CES sample is visible only every k-th frame (k x 10 ms), every phase; "
         "all rows scored against the withheld measurement. Skill a vs b = 1 - SSE_a / SSE_b, 95 % "
         "interval from 2,000 discharge-level bootstrap resamples; `*` = the interval excludes 0. "
         "hold = last visible sample; gp = causal GP on the visible past (length-scale grid x k); "
         "pchip = offline interpolation of the visible samples (not causal); bb = the paper's network; "
         "bb80 / hist80 = the network and its history-only twin under the 80-epoch cap; hist30 = "
         "history-only twin under the 30-epoch cap; causal = train-file-standardized network.", ""]
    L += ["## k = 1 check against the stored errors", "",
          "Share of the stored scored rows whose absolute error is reproduced within 0.5 eV (T_i) / "
          "0.5 km/s (V_rot) (amendment 1), the 99th percentile of the deviation, and the share under "
          "the first rule (squared error within 1e-3 relative; CPU inference against stored CUDA "
          "inference).", "",
          "| split | T_i network | p99 dev (eV) | first rule | T_i hold | T_i causal GP | V_rot network | p99 dev (km/s) | V_rot hold | V_rot causal GP |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for key, c in res["check"].items():
        ti, vt = c["CES_TI"], c["CES_VT"]
        L.append(f"| {key} | {ti['model_abs']:.4f} | {ti['model_dev_p99']:.3f} | {ti['model']:.4f} | "
                 f"{ti['persistence_abs']:.4f} | {ti['gp_causal_abs']:.4f} | {vt['model_abs']:.4f} | "
                 f"{vt['model_dev_p99']:.3f} | {vt['persistence_abs']:.4f} | {vt['gp_causal_abs']:.4f} |")
    L += ["", "## Registered statements (T_i, cut)", ""]
    for h, d in res["verdicts"].items():
        L.append(f"- **{h}**: {'holds' if d['holds'] else 'does not hold'} -- "
                 + ", ".join(f"{k} = {d[k]}" for k in d if k not in ("holds", "k5", "k1"))
                 + "; k = 5: " + "; ".join(f3(x) for x in d["k5"])
                 + ("; k = 1: " + "; ".join(f3(x) for x in d["k1"]) if "k1" in d else ""))
    for t in TARGETS:
        for pop in POPS:
            pairs = [f"{a}_vs_{b}" for a, b in PAIRS[pop]]
            ests = ["hold", "gp", "pchip", *NETS[pop]]
            L += ["", f"## {t}, {pop}: by cadence", "",
                  "| split | k | rows | " + " | ".join(f"rmse {e}" for e in ests) + " | "
                  + " | ".join(p.replace("_vs_", " vs ") for p in pairs) + " |",
                  "|" + "---|" * (3 + len(ests) + len(pairs))]
            for s in SEEDS:
                for k in KS:
                    c = res["cells"][f"{pop}|s{s}|{t}"]["by_k"][str(k)]
                    L.append(f"| {s} | {k} | {c['rows']} | "
                             + " | ".join(f"{c['rmse'][e]:.1f}" for e in ests) + " | "
                             + " | ".join(f3(c["skill"][p]) for p in pairs) + " |")
            L += ["", f"### {t}, {pop}: top 1 % of the hold squared error removed (co-reported)", "",
                  "| split | k | " + " | ".join(p.replace("_vs_", " vs ") for p in pairs) + " |",
                  "|" + "---|" * (2 + len(pairs))]
            for s in SEEDS:
                for k in KS:
                    c = res["cells"][f"{pop}|s{s}|{t}"]["by_k"][str(k)]
                    L.append(f"| {s} | {k} | " + " | ".join(f3(c["skill_trim1"][p]) for p in pairs) + " |")
            L += ["", f"### {t}, {pop}: median absolute error by cadence", "",
                  "| split | k | " + " | ".join(ests) + " |", "|" + "---|" * (2 + len(ests))]
            for s in SEEDS:
                for k in KS:
                    c = res["cells"][f"{pop}|s{s}|{t}"]["by_k"][str(k)]
                    L.append(f"| {s} | {k} | " + " | ".join(f"{c['median_abs'][e]:.1f}" for e in ests) + " |")
            L += ["", f"### {t}, {pop}: by age of the last visible sample under k = 10", "",
                  "| split | age (ms) | rows | " + " | ".join(f"rmse {e}" for e in ests) + " | "
                  + " | ".join(p.replace("_vs_", " vs ") for p in pairs[:5]) + " |",
                  "|" + "---|" * (3 + len(ests) + 5)]
            for s in SEEDS:
                for a, c in res["cells"][f"{pop}|s{s}|{t}"]["by_age_k10"].items():
                    L.append(f"| {s} | {a} | {c['rows']} | "
                             + " | ".join(f"{c['rmse'][e]:.1f}" for e in ests) + " | "
                             + " | ".join(f3(c["skill"][p]) for p in pairs[:5]) + " |")
            L += ["", f"### {t}, {pop}: causal GP with the unscaled grid at k = 5", ""]
            for s in SEEDS:
                g = res["cells"][f"{pop}|s{s}|{t}"]["gp_unscaled_k5"]
                L.append(f"- split {s}: rmse {g['rmse']:.1f}; bb vs it {f3(g['bb_vs_gp_unscaled'])}")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    if sys.argv[1] == "rows":
        rows(sys.argv[2], int(sys.argv[3]))
    else:
        analyze()
