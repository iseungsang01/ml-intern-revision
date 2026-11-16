"""MAST pedestal top, confirmation half (PREREGISTRATION_MAST_PED.md): models trained once on the
discovery row tables, scored once on the even-numbered shots.

CPU only.  Run from the repo root:
    py ces_prediction/experiments/mast/confirm_ped.py build   # confirmation row tables
    py ces_prediction/experiments/mast/confirm_ped.py score   # train on discovery, score, write
Output: docs/paper/mast_ped_confirmation.json / .md
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_gate as rg  # noqa: E402
import run_coupling as rc  # noqa: E402
import run_edge as re_  # noqa: E402
import explore_fast as ef  # noqa: E402
import explore_ped as ep  # noqa: E402

OUT = rg.ROOT / "data" / "mast"
OUT_JSON = rg.ROOT / "docs" / "paper" / "mast_ped_confirmation.json"
OUT_MD = rg.ROOT / "docs" / "paper" / "mast_ped_confirmation.md"
BOOT_SEED, N_BOOT, N_BOOT_MED = 20261116, 2000, 400
MODELS = (("gb_ces", ()), ("gb_ts", ("ts",)), ("gb_da", ("da",)),
          ("gb_all", ("da", "ts", "dens", "lock")), ("gb_all_no_ts", ("da", "dens", "lock")))


def build():
    t0 = time.time()
    listed = [int(x) for x in rc.SHOTS_TXT.read_text().split()]
    conf = [s for s in listed if s % 2 == 0 and (ep.EXTRA / f"{s}.npz").exists()]
    cat = pd.read_parquet(rg.META, columns=["shot_id", "radii_magnetic__radius_efit",
                                            "radii_router_radius_efit",
                                            "radii_router_radius_d_alpha"]).set_index("shot_id")
    add1 = json.loads(re_.ADD1_JSON.read_text(encoding="utf-8"))

    def geo_of(s):
        if s not in cat.index:
            return None
        ax = float(cat.loc[s, "radii_magnetic__radius_efit"])
        for col in ("radii_router_radius_efit", "radii_router_radius_d_alpha"):
            ro = float(cat.loc[s, col])
            if np.isfinite(ax) and np.isfinite(ro) and ro > ax:
                return ax, ro
        return None

    print(f"{len(conf)} confirmation shots with an extra file", flush=True)
    for tg in rg.TARGETS:
        params = re_.params_from(add1, tg)
        parts = []
        for i, s in enumerate(conf):
            geo = geo_of(s)
            r = ef.shot_rows(s, geo, params, tg, ep.POP, ep.FRAC)
            if r is not None:
                parts.append(ep.add_extra(r, s, geo[0]))
            if (i + 1) % 400 == 0:
                print(f"[{time.time() - t0:5.0f}s] {tg} {i + 1}/{len(conf)}", flush=True)
        tab = {k: np.concatenate([q[k] for q in parts]) for k in parts[0]}
        np.savez(OUT / f".ped_conf_rows_{tg}.npz", **tab)
        print(f"{tg}: {len(parts)} shots, {len(tab['y'])} rows", flush=True)   # counts only


def score():
    from sklearn.ensemble import HistGradientBoostingRegressor
    rng = np.random.default_rng(BOOT_SEED)
    res = {"registration": "ces_prediction/experiments/PREREGISTRATION_MAST_PED.md", "targets": {}}
    for tg in rg.TARGETS:
        zd = np.load(OUT / f".ped_rows_{tg}.npz")
        d = {k: zd[k] for k in zd.files}
        zc = np.load(OUT / f".ped_conf_rows_{tg}.npz")
        c = {k: zc[k] for k in zc.files}
        scale = float(np.median(np.sqrt(d["s0"])))
        target = np.clip((d["y"] - d["y1"]).astype(np.float64), -20 * scale, 20 * scale)
        resid = (c["y"] - c["y1"]).astype(np.float64)
        preds = {"hold": np.zeros(len(resid)),
                 "interp": (0.5 * (c["y1"] + c["yn"]) - c["y1"]).astype(np.float64)}
        for name, groups in MODELS:
            m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.06, max_leaf_nodes=31,
                                              min_samples_leaf=200, l2_regularization=1.0,
                                              random_state=ep.SEED)
            m.fit(ep.group_features(d, groups), target)
            preds[name] = m.predict(ep.group_features(c, groups))
            print(f"  {tg} {name} trained and applied", flush=True)
        se = {k: (resid - v) ** 2 for k, v in preds.items()}
        shots, inv = np.unique(c["shot"], return_inverse=True)
        ns = len(shots)
        W = rng.multinomial(ns, np.full(ns, 1.0 / ns), size=N_BOOT).astype(float)
        has_ts = np.isfinite(c["te0"]) & np.isfinite(c["te1"])
        dd = c["d_edge"]
        cells = {"pedestal top": has_ts & (dd >= 0.015) & (dd < 0.05),
                 "core (> 20 cm inside)": has_ts & (dd >= 0.20),
                 "pedestal top, H-mode rating >= 0.5": has_ts & (dd >= 0.015) & (dd < 0.05) & (c["hm0"] >= 0.5),
                 "all rows": np.ones(len(resid), bool), "Thomson available": has_ts}

        def sums(sel, a):
            return np.bincount(inv[sel], se[a][sel], ns)

        def skill(sel, a, b):
            sa, sb = sums(sel, a), sums(sel, b)
            bs = 1.0 - (W @ sa) / (W @ sb)
            return [float(1.0 - sa.sum() / sb.sum()), float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))]

        def med_ratio(sel, a, b):
            idx = np.flatnonzero(sel)
            order = np.argsort(inv[idx], kind="stable")
            idx = idx[order]
            cnt = np.bincount(inv[idx], minlength=ns)
            start = np.concatenate(([0], np.cumsum(cnt)))
            ea, eb = np.sqrt(se[a][idx]), np.sqrt(se[b][idx])
            pt = float(np.median(ea) / np.median(eb))
            bs = []
            for _ in range(N_BOOT_MED):
                pick = rng.integers(0, ns, ns)
                take = np.concatenate([np.arange(start[s], start[s + 1]) for s in pick if cnt[s]])
                bs.append(np.median(ea[take]) / np.median(eb[take]))
            return [pt, float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))]

        out = {"rows": int(len(resid)), "shots": int(ns), "cells": {}}
        pairs = (("gb_all", "hold"), ("gb_ces", "hold"), ("gb_ts", "gb_ces"), ("gb_da", "gb_ces"),
                 ("gb_all", "gb_all_no_ts"), ("gb_all", "gb_ces"), ("gb_all", "interp"), ("interp", "hold"))
        for name, sel in cells.items():
            out["cells"][name] = {"rows": int(sel.sum()), "shots": int(len(np.unique(inv[sel]))),
                                  "rmse_hold": float(np.sqrt(se["hold"][sel].mean())),
                                  "skill": {f"{a}_vs_{b}": skill(sel, a, b) for a, b in pairs}}
        top, core = cells["pedestal top"], cells["core (> 20 cm inside)"]
        out["cells"]["pedestal top"]["median_ratio_gb_all_vs_hold"] = med_ratio(top, "gb_all", "hold")
        out["cells"]["pedestal top"]["median_abs_hold"] = float(np.median(np.sqrt(se["hold"][top])))
        # P4: difference of the Thomson increments, core minus pedestal top, same resamples
        a1, b1 = sums(core, "gb_ts"), sums(core, "gb_ces")
        a2, b2 = sums(top, "gb_ts"), sums(top, "gb_ces")
        diff = (1 - (W @ a1) / (W @ b1)) - (1 - (W @ a2) / (W @ b2))
        out["thomson_increment_core_minus_top"] = [
            float((1 - a1.sum() / b1.sum()) - (1 - a2.sum() / b2.sum())),
            float(np.quantile(diff, 0.025)), float(np.quantile(diff, 0.975))]
        res["targets"][tg] = out
    t = res["targets"]["T_i"]
    top = t["cells"]["pedestal top"]
    res["verdicts"] = {
        "P1": {"value": top["skill"]["gb_all_vs_hold"], "holds": top["skill"]["gb_all_vs_hold"][1] > 0},
        "P2": {"value": top["skill"]["gb_ts_vs_gb_ces"], "holds": top["skill"]["gb_ts_vs_gb_ces"][1] > 0},
        "P3": {"value": top["skill"]["gb_da_vs_gb_ces"], "holds": top["skill"]["gb_da_vs_gb_ces"][1] > 0},
        "P4": {"value": t["thomson_increment_core_minus_top"],
               "holds": t["thomson_increment_core_minus_top"][1] > 0},
        "P5": {"value": top["median_ratio_gb_all_vs_hold"], "holds": top["median_ratio_gb_all_vs_hold"][2] < 1},
        "P6": {"value": top["skill"]["gb_all_vs_interp"], "holds": top["skill"]["gb_all_vs_interp"][2] < 0},
    }
    OUT_JSON.write_text(json.dumps(res, indent=1), encoding="utf-8")
    OUT_MD.write_text(render(res), encoding="utf-8", newline="\n")
    print(json.dumps(res["verdicts"], indent=1))


def f3(x, signed=True):
    s = "+" if signed else ""
    return f"{x[0]:{s}.3f} [{x[1]:{s}.3f}, {x[2]:{s}.3f}]"


def render(res):
    L = ["# MAST pedestal top, confirmation half (registration PREREGISTRATION_MAST_PED.md)", "",
         "Models trained once on the discovery half (odd-numbered shots) and applied to the "
         "even-numbered shots. Skill a vs b = 1 - SSE_a / SSE_b, 95 % interval from 2,000 shot-level "
         "bootstrap resamples. hold = last visible frame; interp = mean of the two neighbouring "
         "visible frames (not causal); gb_ces = trees on the own charge-exchange past; gb_ts = plus "
         "Thomson T_e / n_e at the chord radius; gb_da = plus D-alpha; gb_all = plus Thomson, D-alpha, "
         "line density, saddle-coil signal and H-mode rating.", "", "## Registered statements (T_i)", ""]
    for k, v in res["verdicts"].items():
        L.append(f"- **{k}**: {'holds' if v['holds'] else 'does not hold'} -- {f3(v['value'], signed=(k != 'P5'))}")
    for tg, t in res["targets"].items():
        L += ["", f"## {tg}: {t['rows']} rows, {t['shots']} confirmation shots", "",
              "| cell | rows | shots | hold RMSE | " + " | ".join(
                  p.replace("_vs_", " vs ") for p in next(iter(t["cells"].values()))["skill"]) + " |",
              "|" + "---|" * (4 + len(next(iter(t["cells"].values()))["skill"]))]
        for name, c in t["cells"].items():
            L.append(f"| {name} | {c['rows']} | {c['shots']} | {c['rmse_hold']:.1f} | "
                     + " | ".join(f3(v) for v in c["skill"].values()) + " |")
        top = t["cells"]["pedestal top"]
        L += ["", f"Pedestal top: median absolute error of hold {top['median_abs_hold']:.2f}; ratio of the "
                  f"all-input model to it {f3(top['median_ratio_gb_all_vs_hold'], signed=False)}. Thomson "
                  f"increment, core minus pedestal top: {f3(t['thomson_increment_core_minus_top'])}."]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    {"build": build, "score": score}[sys.argv[1]]()
