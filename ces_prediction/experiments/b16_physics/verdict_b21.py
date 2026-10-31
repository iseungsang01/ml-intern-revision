"""B.21 verdict (PREREGISTRATION_B21.md sec. 4) over frozen artifacts -- no training.

Inputs (all must exist):
  data/.b21_summary.json   run_b16.py --arms v4kjtf --summary-name .b21_summary.json
  data/.b21_rescore.json   rescore_b13.py --extra v4kjtf=.b16_v4kjtf_{pop}_s{s} --out data/.b21_rescore.json
  data/.b16_{v4kjtf,v4kjt}_{pop}_s{seed}/comparison_errors_test.npz, physics_params.json

Arm-vs-arm (v4kjtf - v4kjt) uses the rescore_b13 cluster bootstrap (B = 10^4).
Output: data/.b21_verdict.json.  Usage (repo root):  py ces_prediction/experiments/b16_physics/verdict_b21.py
"""

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from verdict_b16s import DATA, EPS, POPS, SEEDS, TARGETS, BOOTSTRAP_SEED, arms_pair, brief  # noqa: E402
from verdict_b18 import BAR, m1_rows, median_ratio  # noqa: E402

FUSION_SIGMA_R2 = {"cut": 0.27, "incl": 0.54}  # model_seq_v4k.FUSION_SIGMA_R2 (registered init)


def main():
    summary = json.loads((DATA / ".b21_summary.json").read_text(encoding="utf-8"))
    rescore = json.loads((DATA / ".b21_rescore.json").read_text(encoding="utf-8"))["results"]
    for pop in POPS:
        if sorted(summary["arms"]["v4kjtf"][pop]) != sorted(str(s) for s in SEEDS):
            raise SystemExit(f"FATAL: v4kjtf/{pop} is not 4/4 complete in .b21_summary.json")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    out = {}
    for pop in POPS:
        res = rescore[pop]
        rows = summary["arms"]["v4kjtf"][pop]
        node = {}
        for t in TARGETS:
            d = arms_pair("v4kjtf", "v4kjt", pop, t, rng)
            node[t] = {
                "m1_vs_backbone": m1_rows(res, t, "v4kjtf_vs_model"),
                "median_over_persistence": median_ratio(res, t, "v4kjtf_vs_persistence"),
                "mse_paired_vs_backbone": {"points": [rows[str(s)][t]["paired_vs_control"] for s in SEEDS],
                                           "sig_worse": sum(rows[str(s)][t]["control_better"] for s in SEEDS)},
                "fusion_minus_additive": {"pooled": brief(d["pooled"]),
                                              "per_seed": {s: brief(v) for s, v in d["per_seed"].items()}},
            }
        ti, vt = node["CES_TI"], node["CES_VT"]
        mse = ti["mse_paired_vs_backbone"]
        guard = mse["sig_worse"] <= 1 and float(np.mean(mse["points"])) >= -EPS
        b = ti["fusion_minus_additive"]["pooled"]
        bb_hits = sum(1 for r in ti["m1_vs_backbone"]["per_seed"] if r["point"] >= BAR and r["ci"][0] > 0)
        vt_hits = sum(1 for r in vt["m1_vs_backbone"]["per_seed"] if r["point"] >= BAR and r["ci"][0] > 0)
        h_a = ti["median_over_persistence"] <= 0.95 and vt["median_over_persistence"] <= 1.00
        h_b = b["m1"] >= EPS and b["m1_ci"][0] > 0
        node["hypotheses"] = {
            "H21-a": {"pass": h_a, "ti_median": ti["median_over_persistence"], "vrot_median": vt["median_over_persistence"]},
            "H21-b": {"pass": h_b, "fusion_minus_additive_m1": b["m1"], "ci": b["m1_ci"], "mse_diff": b["mse"]},
            "H21-c": {"pass": bb_hits >= 3 and guard, "splits": bb_hits},
            "H21-d": {"pass": vt_hits >= 3 and vt["mse_paired_vs_backbone"]["sig_worse"] == 0, "splits": vt_hits},
            "H21-e": {s: {"epochs": rows[str(s)]["epochs"],
                          "sigma_r_norm": {t: rows[str(s)]["physics"][t]["sigma_r_norm"] for t in TARGETS},
                          "sigma_r_over_init": {t: rows[str(s)]["physics"][t]["sigma_r_norm"] ** 2 / FUSION_SIGMA_R2[pop]
                                                for t in TARGETS}} for s in SEEDS},
            "mse_guard": guard, "mse_mean": float(np.mean(mse["points"])),
        }
        out[pop] = node
    rules = {}
    for pop in POPS:
        h = out[pop]["hypotheses"]
        rules[pop] = {"replaces_combination": h["H21-b"]["pass"] and h["mse_guard"],
                      "rule1": h["H21-c"]["pass"],
                      "trade_is_objective": not h["H21-b"]["pass"],
                      "trade_is_structural": h["H21-a"]["pass"] and not h["mse_guard"]}
    lab = lambda k: ("unconditional" if rules["cut"][k] and rules["incl"][k] else "cut-conditional" if rules["cut"][k]
                     else "inclusive-conditional" if rules["incl"][k] else "not supported")
    rules["replaces_combination"] = lab("replaces_combination")
    rules["physics_rung_b16_rule1"] = lab("rule1")
    rules["next_is_dual_readout"] = any(rules[p]["trade_is_objective"] or rules[p]["trade_is_structural"] for p in POPS)
    out["rules"] = rules
    (DATA / ".b21_verdict.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    for pop in POPS:
        print(f"\n===== {pop} =====")
        for t in TARGETS:
            n = out[pop][t]
            m = n["m1_vs_backbone"]
            p = n["fusion_minus_additive"]["pooled"]
            print(f"{t} M1 vs backbone " + " ".join(f"{r['point']:+.3f}{'*' if r['pass'] else ' '}" for r in m["per_seed"])
                  + f" | pooled {m['pooled']['skill_point']:+.3f} | median/persistence {n['median_over_persistence']:.3f}"
                  + f" | fusion-additive M1 {p['m1']:+.3f} [{p['m1_ci'][0]:+.3f},{p['m1_ci'][1]:+.3f}] MSE {p['mse']:+.3f}")
            print(f"{t} MSE paired vs backbone {n['mse_paired_vs_backbone']}")
        print(json.dumps(out[pop]["hypotheses"], indent=1, default=float))
    print(json.dumps(out["rules"], indent=1))
    print(f"\nwrote {DATA / '.b21_verdict.json'}")


if __name__ == "__main__":
    main()
