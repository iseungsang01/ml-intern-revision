"""B.20 verdict (PREREGISTRATION_B20.md sec. 4) over frozen artifacts -- no training.

Inputs (all must exist):
  data/.b20_summary.json   run_b16.py --arms v4kjt --select medae --summary-name .b20_summary.json
  data/.b20_rescore.json   rescore_b13.py --extra v4kjtm=.b20_v4kjt_{pop}_s{s} --out data/.b20_rescore.json
  data/.b19_summary.json   frozen B.19 (val-MSE selection) epochs, for H20-e
  data/.b{20,16}_v4kjt_{pop}_s{seed}/comparison_errors_test.npz

Arm-vs-arm (medae - MSE selection) uses the rescore_b13 cluster bootstrap (B = 10^4).
Output: data/.b20_verdict.json.  Usage (repo root):  py ces_prediction/experiments/b16_physics/verdict_b20.py
"""

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from verdict_b16s import DATA, EPS, POPS, SEEDS, TARGETS, BOOTSTRAP_SEED, arms_pair, brief  # noqa: E402
from verdict_b18 import BAR, m1_rows, median_ratio  # noqa: E402


def main():
    summary = json.loads((DATA / ".b20_summary.json").read_text(encoding="utf-8"))
    frozen = json.loads((DATA / ".b19_summary.json").read_text(encoding="utf-8"))
    rescore = json.loads((DATA / ".b20_rescore.json").read_text(encoding="utf-8"))["results"]
    if summary.get("select") != "medae":
        raise SystemExit("FATAL: .b20_summary.json was not written with --select medae")
    for pop in POPS:
        if sorted(summary["arms"]["v4kjt"][pop]) != sorted(str(s) for s in SEEDS):
            raise SystemExit(f"FATAL: v4kjt/{pop} is not 4/4 complete in .b20_summary.json")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    out = {}
    for pop in POPS:
        res = rescore[pop]
        rows = summary["arms"]["v4kjt"][pop]
        node = {}
        for t in TARGETS:
            d = arms_pair(".b20_v4kjt", ".b16_v4kjt", pop, t, rng)
            node[t] = {
                "m1_vs_backbone": m1_rows(res, t, "v4kjtm_vs_model"),
                "median_over_persistence": median_ratio(res, t, "v4kjtm_vs_persistence"),
                "mse_paired_vs_backbone": {"points": [rows[str(s)][t]["paired_vs_control"] for s in SEEDS],
                                           "sig_worse": sum(rows[str(s)][t]["control_better"] for s in SEEDS)},
                "medae_minus_mse_selection": {"pooled": brief(d["pooled"]),
                                              "per_seed": {s: brief(v) for s, v in d["per_seed"].items()}},
            }
        ti, vt = node["CES_TI"], node["CES_VT"]
        mse = ti["mse_paired_vs_backbone"]
        guard = mse["sig_worse"] <= 1 and float(np.mean(mse["points"])) >= -EPS
        b = ti["medae_minus_mse_selection"]["pooled"]
        bb_hits = sum(1 for r in ti["m1_vs_backbone"]["per_seed"] if r["point"] >= BAR and r["ci"][0] > 0)
        vt_hits = sum(1 for r in vt["m1_vs_backbone"]["per_seed"] if r["point"] >= BAR and r["ci"][0] > 0)
        h_a = ti["median_over_persistence"] <= 0.95 and vt["median_over_persistence"] <= 1.00
        h_b = b["m1"] >= EPS and b["m1_ci"][0] > 0
        node["hypotheses"] = {
            "H20-a": {"pass": h_a, "ti_median": ti["median_over_persistence"], "vrot_median": vt["median_over_persistence"]},
            "H20-b": {"pass": h_b, "medae_minus_mse_m1": b["m1"], "ci": b["m1_ci"], "mse_diff": b["mse"]},
            "H20-c": {"pass": bb_hits >= 3 and guard, "splits": bb_hits},
            "H20-d": {"pass": vt_hits >= 3 and vt["mse_paired_vs_backbone"]["sig_worse"] == 0, "splits": vt_hits},
            "H20-e": {s: {"epochs_medae": rows[str(s)]["epochs"],
                          "epochs_mse_b19": frozen["arms"]["v4kjt"][pop][str(s)]["epochs"]} for s in SEEDS},
            "mse_guard": guard, "mse_mean": float(np.mean(mse["points"])),
        }
        out[pop] = node
    rules = {}
    for pop in POPS:
        h = out[pop]["hypotheses"]
        moved = h["H20-a"]["pass"] or h["H20-b"]["pass"]
        rules[pop] = {"rule1": h["H20-c"]["pass"],
                      "selection_was_cause": h["H20-a"]["pass"] and h["H20-b"]["pass"] and h["mse_guard"],
                      "structural_trade": moved and not h["mse_guard"],
                      "selection_not_cause": not moved}
    rules["physics_rung"] = ("unconditional" if rules["cut"]["rule1"] and rules["incl"]["rule1"] else
                             "cut-conditional" if rules["cut"]["rule1"] else
                             "inclusive-conditional (selection rule differs by population)" if rules["incl"]["rule1"] else
                             "not met")
    rules["next_is_residual_routing"] = any(rules[p]["structural_trade"] or rules[p]["selection_not_cause"] for p in POPS)
    out["rules"] = rules
    (DATA / ".b20_verdict.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    for pop in POPS:
        print(f"\n===== {pop} =====")
        for t in TARGETS:
            n = out[pop][t]
            m = n["m1_vs_backbone"]
            p = n["medae_minus_mse_selection"]["pooled"]
            print(f"{t} M1 vs backbone " + " ".join(f"{r['point']:+.3f}{'*' if r['pass'] else ' '}" for r in m["per_seed"])
                  + f" | pooled {m['pooled']['skill_point']:+.3f} | median/persistence {n['median_over_persistence']:.3f}"
                  + f" | medae-mse M1 {p['m1']:+.3f} [{p['m1_ci'][0]:+.3f},{p['m1_ci'][1]:+.3f}] MSE {p['mse']:+.3f}")
            print(f"{t} MSE paired vs backbone {n['mse_paired_vs_backbone']}")
        print(json.dumps(out[pop]["hypotheses"], indent=1, default=float))
    print(json.dumps(out["rules"], indent=1))
    print(f"\nwrote {DATA / '.b20_verdict.json'}")


if __name__ == "__main__":
    main()
