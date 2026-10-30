"""B.18 verdict (PREREGISTRATION_B18.md sec. 4) over frozen artifacts -- no training.

Inputs (all must exist):
  data/.b18_summary.json        run_b16.py --arms v4kj --summary-name .b18_summary.json (paired MSE vs backbone)
  data/.b18_rescore.json        rescore_b13.py --extra v4kj=.b16_v4kj_{pop}_s{s} anchorj=.b18_anchor_{pop}_s{s}
                                --out data/.b18_rescore.json
  data/.b16_{v4kj,v4kng}_{pop}_s{seed}/comparison_errors_test.npz

Arm-vs-arm comparisons (v4kj - v4kng, all rows and the stale stratum dt > 15 ms) use the same
cluster bootstrap as rescore_b13 (B = 10^4). Output: data/.b18_verdict.json.
Usage (repo root):  py ces_prediction/experiments/b16_physics/verdict_b18.py
"""

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from verdict_b16s import DATA, EPS, POPS, SEEDS, TARGETS, BOOTSTRAP_SEED, arms_pair, brief  # noqa: E402

BAR = 0.05  # B.16 rule-1 bar on M1 vs the backbone


def m1_rows(res, t, key):
    per = [res["per_seed"][str(s)][t][key]["all"]["m1_medae"] for s in SEEDS]
    return {"per_seed": [{"point": p["skill_point"], "ci": p["skill_ci95"], "pass": p["pass"]} for p in per],
            "pooled": res["pooled"][t][key]["m1_medae"]}


def median_ratio(res, t, key):
    return res["pooled"][t][key]["m5_quantile_ratio"]["50"]


def main():
    summary = json.loads((DATA / ".b18_summary.json").read_text(encoding="utf-8"))
    rescore = json.loads((DATA / ".b18_rescore.json").read_text(encoding="utf-8"))["results"]
    for pop in POPS:
        if sorted(summary["arms"]["v4kj"][pop]) != sorted(str(s) for s in SEEDS):
            raise SystemExit(f"FATAL: v4kj/{pop} is not 4/4 complete in .b18_summary.json")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    stale = lambda A: A["dt"] > 15.0
    out = {}
    for pop in POPS:
        res = rescore[pop]
        rows = summary["arms"]["v4kj"][pop]
        node = {}
        for t in TARGETS:
            va = arms_pair("v4kj", "v4kng", pop, t, rng)
            vs = arms_pair("v4kj", "v4kng", pop, t, rng, stale)
            node[t] = {
                "m1_vs_backbone": m1_rows(res, t, "v4kj_vs_model"),
                "median_over_persistence": {"v4kj": median_ratio(res, t, "v4kj_vs_persistence"),
                                            "pure_jump_anchor": median_ratio(res, t, "anchorj_vs_persistence")},
                "mse_paired_vs_backbone": {"points": [rows[str(s)][t]["paired_vs_control"] for s in SEEDS],
                                           "sig_worse": sum(rows[str(s)][t]["control_better"] for s in SEEDS)},
                "v4kj_minus_v4kng_all": {"pooled": brief(va["pooled"]), "per_seed": {s: brief(v) for s, v in va["per_seed"].items()}},
                "v4kj_minus_v4kng_stale": {"pooled": brief(vs["pooled"]), "per_seed": {s: brief(v) for s, v in vs["per_seed"].items()}},
            }
        ti, vt = node["CES_TI"], node["CES_VT"]
        mse = ti["mse_paired_vs_backbone"]
        guard = mse["sig_worse"] <= 1 and float(np.mean(mse["points"])) >= -EPS
        a = ti["v4kj_minus_v4kng_all"]["pooled"]
        h_a = a["m1"] >= EPS and a["m1_ci"][0] > 0
        bb_hits = sum(1 for r in ti["m1_vs_backbone"]["per_seed"] if r["point"] >= BAR and r["ci"][0] > 0)
        st = ti["v4kj_minus_v4kng_stale"]
        st_hits = sum(1 for s in SEEDS if st["per_seed"][str(s)]["m1"] >= BAR and st["per_seed"][str(s)]["m1_ci"][0] > 0)
        h_b = st_hits >= 3 and a["m1"] >= -EPS and guard
        c = vt["median_over_persistence"]["pure_jump_anchor"]
        c_read = ("P02 band reached under the hold-or-jump prior" if c <= 1.01 else
                  "hold-or-jump prior cannot reach the P02 band either" if c >= 1.05 else "between bands")
        vt_hits = sum(1 for r in vt["m1_vs_backbone"]["per_seed"] if r["point"] >= BAR and r["ci"][0] > 0)
        h_d = vt_hits >= 3 and vt["mse_paired_vs_backbone"]["sig_worse"] == 0
        node["hypotheses"] = {
            "H18-a": {"pass": h_a, "v4kj_minus_v4kng_m1": a["m1"], "ci": a["m1_ci"],
                      "backbone_bar_splits": bb_hits, "backbone_bar_pass": bb_hits >= 3},
            "H18-b": {"pass": h_b, "stale_splits_ge_bar": st_hits, "overall_vs_v4kng": a["m1"], "mse_guard": guard},
            "H18-c": {"vrot_pure_anchor_median_over_persistence": c, "reading": c_read,
                      "ti_pure_anchor_median_over_persistence": ti["median_over_persistence"]["pure_jump_anchor"]},
            "H18-d": {"pass": h_d, "splits": vt_hits},
            "mse_guard": guard,
        }
        out[pop] = node
    rules = {}
    for pop in POPS:
        h = out[pop]["hypotheses"]
        rules[pop] = {"replaces_ou_anchor": h["H18-a"]["pass"] and h["mse_guard"],
                      "meets_b16_rule1": h["H18-a"]["backbone_bar_pass"] and h["mse_guard"]}
    lab = lambda k: ("unconditional" if rules["cut"][k] and rules["incl"][k] else "cut-conditional" if rules["cut"][k]
                     else "inclusive-conditional" if rules["incl"][k] else "not supported")
    rules["replaces_ou_anchor"] = lab("replaces_ou_anchor")
    rules["physics_rung_b16_rule1"] = lab("meets_b16_rule1")
    out["rules"] = rules
    (DATA / ".b18_verdict.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    for pop in POPS:
        print(f"\n===== {pop} =====")
        for t in TARGETS:
            n = out[pop][t]
            m = n["m1_vs_backbone"]
            print(f"{t} M1 vs backbone " + " ".join(f"{r['point']:+.3f}{'*' if r['pass'] else ' '}" for r in m["per_seed"])
                  + f" | pooled {m['pooled']['skill_point']:+.3f} | median/persistence v4kj {n['median_over_persistence']['v4kj']:.3f}"
                  + f" pure anchor {n['median_over_persistence']['pure_jump_anchor']:.3f}")
            for key in ("v4kj_minus_v4kng_all", "v4kj_minus_v4kng_stale"):
                p = n[key]["pooled"]
                print(f"{t} {key:24s} pooled M1 {p['m1']:+.3f} [{p['m1_ci'][0]:+.3f},{p['m1_ci'][1]:+.3f}] n={p['n']} MSE {p['mse']:+.3f}")
            print(f"{t} MSE paired vs backbone {n['mse_paired_vs_backbone']}")
        print(json.dumps(out[pop]["hypotheses"], indent=1, default=float))
    print(json.dumps(out["rules"], indent=1))
    print(f"\nwrote {DATA / '.b18_verdict.json'}")


if __name__ == "__main__":
    main()
