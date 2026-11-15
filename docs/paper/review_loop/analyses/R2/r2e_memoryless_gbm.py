"""R2-e (REGISTRATION.md): memoryless gradient boosting on the row inputs.

GBM-A: 20 features (15 fast + dt + staleness/has of both targets), no CES value.
GBM-B: all 22 per-step channels (adds the last observed CES values), increment target.
sklearn HistGradientBoostingRegressor, hyperparameters on VAL files only, scored on the
backbone's frozen TEST rows. CPU.
Run from the repo root:  py docs/paper/review_loop/analyses/R2/r2e_memoryless_gbm.py [--smoke]
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

import r2_common as C
import r2ac_vrot_running_mean as RA
from evaluate import _load_stats
from seq_data import N_FAST_CHANNELS, build_blocks, load_grid_files

TARGETS = ("CES_TI", "CES_VT")
BASE = N_FAST_CHANNELS + 1                       # first slow-triple column
CARRIED = (BASE, BASE + 3)
HAS = (BASE + 2, BASE + 5)
COLS_A = [c for c in range(22) if c not in CARRIED]
COLS_B = list(range(22))
LEAVES = (15, 63)
MIN_LEAF = (20, 200)
MAX_ITER = 600
LR = 0.1


def tables(grid, dims, stats, names, per_shot):
    X, Y, M, T, F = [], [], [], [], []
    for i, nm in enumerate(names):
        for b in build_blocks(grid[nm], dims, stats, per_shot_norm=per_shot):
            X.append(b["x"]); Y.append(b["y"]); M.append(b["mask"]); T.append(b["time"])
            F.append(np.full(len(b["time"]), i))
    return np.concatenate(X), np.concatenate(Y), np.concatenate(M), np.concatenate(T), np.concatenate(F)


def fit_select(Xtr, ytr, Xva, yva, seed, n_iter):
    best = None
    for leaves in LEAVES:
        for min_leaf in MIN_LEAF:
            g = HistGradientBoostingRegressor(loss="squared_error", learning_rate=LR, max_iter=n_iter,
                                              max_leaf_nodes=leaves, min_samples_leaf=min_leaf,
                                              early_stopping=False, random_state=seed)
            g.fit(Xtr, ytr)
            mse = np.array([float(np.mean((p - yva) ** 2)) for p in g.staged_predict(Xva)])
            j = int(np.argmin(mse))
            if best is None or mse[j] < best["val_mse"]:
                best = {"val_mse": float(mse[j]), "n_trees": j + 1, "max_leaf_nodes": leaves,
                        "min_samples_leaf": min_leaf, "model": g}
    return best


def staged_at(g, X, n_trees):
    for i, p in enumerate(g.staged_predict(X)):
        if i + 1 == n_trees:
            return p
    raise RuntimeError("n_trees beyond the fitted ensemble")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="cut s42 only, 40 trees (plumbing check, not a result)")
    args = ap.parse_args()
    n_iter = 40 if args.smoke else MAX_ITER
    out_path = C.HERE / ("r2e_memoryless_gbm_smoke.json" if args.smoke else "r2e_memoryless_gbm.json")
    t0 = time.time()
    out = {"protocol": {"library": "sklearn HistGradientBoostingRegressor", "lr": LR, "max_iter": n_iter,
                        "grid": {"max_leaf_nodes": LEAVES, "min_samples_leaf": MIN_LEAF,
                                 "fast_scaling": ["per_shot", "global"]},
                        "selection": "VAL-file MSE (normalized target units), trees from staged predictions",
                        "B": C.B_BOOT, "seed_boot": C.SEED_BOOT}, "cells": []}
    for pop in (("cut",) if args.smoke else tuple(C.POPS)):
        ds = C.build_dataset(pop)
        tc = int(ds._column_slices["time"])
        vc = int(ds._column_slices["target"][1])
        grid, dims = load_grid_files(C.DATA, drop_stuck_targets=True, ti_spike_cut_ev=float(C.POPS[pop]["cut"]))
        amax_cache = {}
        for s in ((42,) if args.smoke else C.SEEDS):
            man = C.manifest(pop, s)
            stats = _load_stats(json.loads((C.DATA / C.POPS[pop]["run"].format(s=s) / "metrics.json").read_text(encoding="utf-8")))
            mean = stats["target"]["mean"].astype(np.float64)
            std = stats["target"]["std"].astype(np.float64)
            sets = {k: [f for f in man[f"{k}_files"] if f in grid] for k in ("train", "val", "test")}
            data = {sc: {k: tables(grid, dims, stats, sets[k], sc == "per_shot") for k in sets}
                    for sc in ("per_shot", "global")}
            # (file, time) -> row of the TEST table (identical for both scalings)
            _, _, _, Tt, Ft = data["per_shot"]["test"]
            lut = {(sets["test"][int(f)], float(t)): i for i, (f, t) in enumerate(zip(Ft, Tt))}
            cell = {"pop": pop, "seed": s, "minutes_start": round((time.time() - t0) / 60, 1), "targets": {}}
            for t, name in enumerate(TARGETS):
                d = C.canonical(pop, s, name)
                names = [Path(ds.valid_files[int(i)]).name for i in d["shot"]]
                keys = [float(ds.file_arrays[int(i)][int(r), tc]) for i, r in zip(d["shot"], d["row"])]
                ridx = np.array([lut[(n, k)] for n, k in zip(names, keys)])
                y_grid = data["per_shot"]["test"][1][ridx, t] * std[t] + mean[t]
                if float(np.max(np.abs(y_grid - d["y"]))) > 1e-2 * max(1.0, float(std[t]) / 100):
                    raise SystemExit(f"FATAL: {pop} s{s} {name}: grid y does not match the frozen y_true")
                keep_rule = None
                if name == "CES_VT":
                    am = np.empty(len(ridx))
                    for i, (sh, r) in enumerate(zip(d["shot"], d["row"])):
                        sh = int(sh)
                        if sh not in amax_cache:
                            amax_cache[sh] = RA.file_tables(ds.file_arrays[sh], tc, vc)["amax"]
                        am[i] = amax_cache[sh][int(r)]
                    keep_rule = am <= RA.T_RULE
                rec = {"n_rows": int(len(ridx))}
                for var, cols in (("A", COLS_A), ("B", COLS_B)):
                    best = None
                    for sc in ("per_shot", "global"):
                        Xtr, Ytr, Mtr, _, _ = data[sc]["train"]
                        Xva, Yva, Mva, _, _ = data[sc]["val"]
                        ktr = (Mtr[:, t] > 0) & (Xtr[:, HAS[t]] > 0.5)
                        kva = (Mva[:, t] > 0) & (Xva[:, HAS[t]] > 0.5)
                        off_tr = Xtr[ktr, CARRIED[t]] if var == "B" else 0.0
                        off_va = Xva[kva, CARRIED[t]] if var == "B" else 0.0
                        b = fit_select(Xtr[ktr][:, cols], Ytr[ktr, t] - off_tr, Xva[kva][:, cols],
                                       Yva[kva, t] - off_va, s, n_iter)
                        b["scaling"] = sc
                        b["n_train"] = int(ktr.sum()); b["n_val"] = int(kva.sum())
                        b["val_mse_persistence"] = float(np.mean((Yva[kva, t] - Xva[kva, CARRIED[t]]) ** 2))
                        if best is None or b["val_mse"] < best["val_mse"]:
                            best = b
                    Xte = data[best["scaling"]]["test"][0][ridx]
                    p = staged_at(best["model"], Xte[:, cols], best["n_trees"])
                    if var == "B":
                        p = p + Xte[:, CARRIED[t]]
                    se = (p * std[t] + mean[t] - d["y"]) ** 2
                    r = {"hp": {k: best[k] for k in ("scaling", "max_leaf_nodes", "min_samples_leaf", "n_trees",
                                                     "n_train", "n_val", "val_mse", "val_mse_persistence")},
                         "val_skill_vs_persistence": 1.0 - best["val_mse"] / best["val_mse_persistence"],
                         "rmse": float(np.sqrt(se.mean())), "rmse_backbone": float(np.sqrt(d["model"].mean())),
                         "gbm_vs": {k: C.skill(d["shot"], se, d[k]) for k in ("persistence", "pchip", "gp_causal")},
                         "backbone_vs_gbm": C.skill(d["shot"], d["model"], se)}
                    if keep_rule is not None:
                        k = keep_rule
                        r["F1000"] = {"n_dropped": int((~k).sum()),
                                      "gbm_vs": {q: C.skill(d["shot"][k], se[k], d[q][k])
                                                 for q in ("persistence", "pchip", "gp_causal")},
                                      "backbone_vs_gbm": C.skill(d["shot"][k], d["model"][k], se[k])}
                    rec[f"GBM_{var}"] = r
                    print(f"[R2e] {pop} s{s} {name} GBM-{var}: hp {r['hp']['scaling']}/{r['hp']['max_leaf_nodes']}/"
                          f"{r['hp']['min_samples_leaf']}/{r['hp']['n_trees']} val skill/pers {r['val_skill_vs_persistence']:+.3f} | "
                          f"TEST rmse {r['rmse']:.1f} (bb {r['rmse_backbone']:.1f}) vs pers {C.fmt(r['gbm_vs']['persistence'])} "
                          f"pchip {C.fmt(r['gbm_vs']['pchip'])} gpc {C.fmt(r['gbm_vs']['gp_causal'])} | bb vs gbm "
                          f"{C.fmt(r['backbone_vs_gbm'])}"
                          + (f" | F1000 bb vs gbm {C.fmt(r['F1000']['backbone_vs_gbm'])} gbm vs pers "
                             f"{C.fmt(r['F1000']['gbm_vs']['persistence'])}" if keep_rule is not None else ""),
                          flush=True)
                cell["targets"][name] = rec
            cell["minutes_end"] = round((time.time() - t0) / 60, 1)
            out["cells"].append(cell)
            out_path.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"wrote {out_path.name} ({round((time.time() - t0) / 60, 1)} min)")


if __name__ == "__main__":
    main()
