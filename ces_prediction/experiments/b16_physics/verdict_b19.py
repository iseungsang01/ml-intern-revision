"""B.19 verdict (PREREGISTRATION_B19.md sec. 4) over frozen artifacts -- no training.

Inputs (all must exist):
  data/.b19_summary.json        run_b16.py --arms v4kjt --summary-name .b19_summary.json (paired MSE vs backbone)
  data/.b19_rescore.json        rescore_b13.py --extra v4kjt=.b16_v4kjt_{pop}_s{s} anchort=.b19_anchor_{pop}_s{s}
                                --out data/.b19_rescore.json
  data/.b18_rescore.json        frozen B.18 rescore (pure hold-or-jump anchor medians)
  data/.b16_{v4kjt,v4kj}_{pop}_s{seed}/, data/.b1{9,8}_anchor_{pop}_s{seed}/comparison_errors_test.npz

Arm-vs-arm comparisons (v4kjt - v4kj; pure t anchor - pure jump anchor) use the same cluster
bootstrap as rescore_b13 (B = 10^4). Output: data/.b19_verdict.json.
Usage (repo root):  py ces_prediction/experiments/b16_physics/verdict_b19.py
"""

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from verdict_b16s import DATA, EPS, POPS, SEEDS, TARGETS, BOOTSTRAP_SEED, arms_pair, brief  # noqa: E402
from verdict_b18 import BAR, m1_rows, median_ratio  # noqa: E402


def anchor_reading(v):
    return ("noise scale" if v <= 1.02 else "scale not the cause" if v >= 1.05 else "partial")


def main():
    summary = json.loads((DATA / ".b19_summary.json").read_text(encoding="utf-8"))
    rescore = json.loads((DATA / ".b19_rescore.json").read_text(encoding="utf-8"))["results"]
    frozen = json.loads((DATA / ".b18_rescore.json").read_text(encoding="utf-8"))["results"]
    for pop in POPS:
        if sorted(summary["arms"]["v4kjt"][pop]) != sorted(str(s) for s in SEEDS):
            raise SystemExit(f"FATAL: v4kjt/{pop} is not 4/4 complete in .b19_summary.json")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    out = {}
    for pop in POPS:
        res = rescore[pop]
        rows = summary["arms"]["v4kjt"][pop]
        node = {}
        for t in TARGETS:
            va = arms_pair("v4kjt", "v4kj", pop, t, rng)
            vp = arms_pair(".b19_anchor", ".b18_anchor", pop, t, rng)
            node[t] = {
                "m1_vs_backbone": m1_rows(res, t, "v4kjt_vs_model"),
                "median_over_persistence": {"v4kjt": median_ratio(res, t, "v4kjt_vs_persistence"),
                                            "pure_t_anchor": median_ratio(res, t, "anchort_vs_persistence"),
                                            "pure_jump_anchor_b18": median_ratio(frozen[pop], t, "anchorj_vs_persistence")},
                "pure_t_anchor_mse_vs_pchip": res["pooled"][t]["anchort_vs_pchip"]["mse"]["skill_point"],
                "mse_paired_vs_backbone": {"points": [rows[str(s)][t]["paired_vs_control"] for s in SEEDS],
                                           "sig_worse": sum(rows[str(s)][t]["control_better"] for s in SEEDS)},
                "v4kjt_minus_v4kj": {"pooled": brief(va["pooled"]), "per_seed": {s: brief(v) for s, v in va["per_seed"].items()}},
                "pure_t_minus_pure_jump": {"pooled": brief(vp["pooled"]), "per_seed": {s: brief(v) for s, v in vp["per_seed"].items()}},
            }
        ti, vt = node["CES_TI"], node["CES_VT"]
        mse = ti["mse_paired_vs_backbone"]
        guard = mse["sig_worse"] <= 1 and float(np.mean(mse["points"])) >= -EPS
        pa = ti["pure_t_minus_pure_jump"]["pooled"]
        vr = vt["median_over_persistence"]["pure_t_anchor"]
        b = ti["v4kjt_minus_v4kj"]["pooled"]
        h_b = b["m1"] >= EPS and b["m1_ci"][0] > 0 and guard
        bb_hits = sum(1 for r in ti["m1_vs_backbone"]["per_seed"] if r["point"] >= BAR and r["ci"][0] > 0)
        vt_hits = sum(1 for r in vt["m1_vs_backbone"]["per_seed"] if r["point"] >= BAR and r["ci"][0] > 0)
        node["hypotheses"] = {
            "H19-a": {"vrot_pure_t_anchor_median_over_persistence": vr, "reading": anchor_reading(vr),
                      "ti_pure_t_minus_pure_jump_m1": pa["m1"], "ci": pa["m1_ci"],
                      "ti_anchor_gain_pass": pa["m1"] >= EPS and pa["m1_ci"][0] > 0},
            "H19-b": {"pass": h_b, "v4kjt_minus_v4kj_m1": b["m1"], "ci": b["m1_ci"], "mse_guard": guard},
            "H19-c": {"pass": bb_hits >= 3 and guard, "splits": bb_hits},
            "H19-d": {"pass": vt_hits >= 3 and vt["mse_paired_vs_backbone"]["sig_worse"] == 0, "splits": vt_hits},
            "H19-e": {s: {t: rows[str(s)]["physics"][t] for t in TARGETS} for s in SEEDS},
            "mse_guard": guard,
        }
        out[pop] = node
    rules = {}
    for pop in POPS:
        h = out[pop]["hypotheses"]
        rules[pop] = {"mechanism": h["H19-a"]["reading"] == "noise scale",
                      "replaces_measurement_model": h["H19-b"]["pass"],
                      "meets_b16_rule1": h["H19-c"]["pass"]}
    lab = lambda k: ("unconditional" if rules["cut"][k] and rules["incl"][k] else "cut-conditional" if rules["cut"][k]
                     else "inclusive-conditional" if rules["incl"][k] else "not supported")
    rules["mechanism_confirmed"] = rules["cut"]["mechanism"] and rules["incl"]["mechanism"]
    rules["replaces_measurement_model"] = lab("replaces_measurement_model")
    rules["physics_rung_b16_rule1"] = lab("meets_b16_rule1")
    rules["next_is_residual_selection"] = rules["mechanism_confirmed"] and rules["replaces_measurement_model"] == "not supported"
    out["rules"] = rules
    (DATA / ".b19_verdict.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    for pop in POPS:
        print(f"\n===== {pop} =====")
        for t in TARGETS:
            n = out[pop][t]
            m = n["m1_vs_backbone"]
            mp = n["median_over_persistence"]
            print(f"{t} M1 vs backbone " + " ".join(f"{r['point']:+.3f}{'*' if r['pass'] else ' '}" for r in m["per_seed"])
                  + f" | pooled {m['pooled']['skill_point']:+.3f} | median/persistence v4kjt {mp['v4kjt']:.3f}"
                  + f" pure t {mp['pure_t_anchor']:.3f} (pure jump {mp['pure_jump_anchor_b18']:.3f})"
                  + f" | pure t MSE vs PCHIP {n['pure_t_anchor_mse_vs_pchip']:+.3f}")
            for key in ("v4kjt_minus_v4kj", "pure_t_minus_pure_jump"):
                p = n[key]["pooled"]
                print(f"{t} {key:22s} pooled M1 {p['m1']:+.3f} [{p['m1_ci'][0]:+.3f},{p['m1_ci'][1]:+.3f}] n={p['n']} MSE {p['mse']:+.3f}")
            print(f"{t} MSE paired vs backbone {n['mse_paired_vs_backbone']}")
        print(json.dumps({k: v for k, v in out[pop]["hypotheses"].items() if k != "H19-e"}, indent=1, default=float))
    print(json.dumps(out["rules"], indent=1))
    print(f"\nwrote {DATA / '.b19_verdict.json'}")


if __name__ == "__main__":
    main()
