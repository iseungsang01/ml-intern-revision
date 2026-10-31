"""B.22 verdict (PREREGISTRATION_B22.md sec. 2) -- dual readout over frozen B.19 scoring, no training, no rescoring.

Input: data/.b19_rescore.json (keys anchort_vs_model, anchort_vs_persistence, anchort_vs_pchip, v4kjt_vs_model).
Median estimator = pure t anchor; mean estimator = v4kjt where its MSE guard holds (sec. 8bm), else backbone v2.
Output: data/.b22_verdict.json.  Usage (repo root):  py ces_prediction/experiments/b16_physics/verdict_b22.py
"""

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from verdict_b16s import DATA, EPS, POPS, SEEDS, TARGETS  # noqa: E402
from verdict_b18 import BAR, m1_rows, median_ratio  # noqa: E402


def per_split(res, t, key, metric):
    return [res["per_seed"][str(s)][t][key]["all"][metric] for s in SEEDS]


def main():
    res_all = json.loads((DATA / ".b19_rescore.json").read_text(encoding="utf-8"))["results"]
    out = {}
    for pop in POPS:
        res = res_all[pop]
        node = {}
        for t in TARGETS:
            vb = m1_rows(res, t, "anchort_vs_model")
            vp = per_split(res, t, "anchort_vs_persistence", "m1_medae")
            node[t] = {
                "m1_vs_backbone": vb,
                "m1_vs_persistence": [{"point": r["skill_point"], "ci": r["skill_ci95"]} for r in vp],
                "median_over_persistence_split": [res["per_seed"][str(s)][t]["anchort_vs_persistence"]["all"]
                                                  ["m5_quantile_ratio"]["50"] for s in SEEDS],
                "median_over_persistence_pooled": median_ratio(res, t, "anchort_vs_persistence"),
                "mse_vs_backbone": [r["skill_point"] for r in per_split(res, t, "anchort_vs_model", "mse")],
                "mse_vs_pchip": [r["skill_point"] for r in per_split(res, t, "anchort_vs_pchip", "mse")],
            }
            n = node[t]
            n["H22-a"] = sum(1 for r in vb["per_seed"] if r["point"] >= BAR and r["ci"][0] > 0) >= 3
            n["H22-a_splits"] = sum(1 for r in vb["per_seed"] if r["point"] >= BAR and r["ci"][0] > 0)
            n["H22-b_splits"] = sum(1 for r in n["m1_vs_persistence"] if r["point"] > 0 and r["ci"][0] > 0)
            n["H22-b"] = n["H22-b_splits"] >= 3
        # mean estimator (restated from sec. 8bm, not tested): v4kjt if its T_i MSE guard holds, else v2
        pts = [r["skill_point"] for r in per_split(res, "CES_TI", "v4kjt_vs_model", "mse")]
        sig = sum(1 for r in per_split(res, "CES_TI", "v4kjt_vs_model", "mse") if r["skill_ci95"][1] < 0)
        guard = sig <= 1 and float(np.mean(pts)) >= -EPS
        node["mean_estimator"] = {"arm": "v4kjt" if guard else "v2", "v4kjt_mse_points": pts,
                                  "v4kjt_sig_worse": sig, "v4kjt_mse_mean": float(np.mean(pts))}
        out[pop] = node
    full = {(p, t): out[p][t]["H22-a"] and out[p][t]["H22-b"] for p in POPS for t in TARGETS}
    rules = {"rule1_unconditional": all(full.values()),
             "supported_cells": [f"{p}/{t}" for (p, t), v in full.items() if v],
             "rule3_vrot_beats_backbone_ties_persistence": [p for p in POPS if out[p]["CES_VT"]["H22-a"]
                                                           and not out[p]["CES_VT"]["H22-b"]]}
    out["rules"] = rules
    (DATA / ".b22_verdict.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    for pop in POPS:
        print(f"\n===== {pop} =====  mean estimator: {out[pop]['mean_estimator']}")
        for t in TARGETS:
            n = out[pop][t]
            print(f"{t} anchor M1 vs backbone " + " ".join(f"{r['point']:+.3f}{'*' if r['point'] >= BAR and r['ci'][0] > 0 else ' '}"
                                                     for r in n["m1_vs_backbone"]["per_seed"])
                  + f" | H22-a {n['H22-a_splits']}/4")
            print(f"{t} anchor M1 vs persistence " + " ".join(f"{r['point']:+.3f}[{r['ci'][0]:+.3f},{r['ci'][1]:+.3f}]"
                                                         for r in n["m1_vs_persistence"]) + f" | H22-b {n['H22-b_splits']}/4")
            print(f"{t} median/persistence split " + " ".join(f"{v:.3f}" for v in n["median_over_persistence_split"])
                  + f" pooled {n['median_over_persistence_pooled']:.3f}")
            print(f"{t} anchor MSE vs backbone " + " ".join(f"{v:+.3f}" for v in n["mse_vs_backbone"])
                  + " | vs PCHIP " + " ".join(f"{v:+.3f}" for v in n["mse_vs_pchip"]))
    print(json.dumps(rules, indent=1))
    print(f"\nwrote {DATA / '.b22_verdict.json'}")


if __name__ == "__main__":
    main()
