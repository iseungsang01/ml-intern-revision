"""B.16 sec. 10 readings (PREREGISTRATION_B16.md sec. 10) over frozen artifacts -- no training.

(a) gate initialised open:   v4ko  - v4kng  paired M1 (T_i, pooled over 4 splits)
(c) level-holding anchor:    v4kngh - v4kng paired M1 on the stale stratum (dt > 15 ms) per split
                             and pooled, overall M1, and paired MSE significance per split
(b) pure anchor:             median AE / persistence from data/.b16_anchor_rescore.json

Arm-vs-arm comparisons use rescore_b13.compare (same cluster bootstrap, B = 10^4) on rows pinned
by (shot, dt). Output: data/.b16s_verdict.json.
Usage (repo root):  py ces_prediction/experiments/b16_physics/verdict_b16s.py
"""

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
DATA = REPO / "data"
sys.path.insert(0, str(HERE.parent / "b13_robust"))
sys.path.insert(0, str(REPO / "ces_prediction"))
from rescore_b13 import POPS as B13_POPS, SEEDS, TARGETS, B_FULL, compare, load_arm, shot_names  # noqa: E402
from bootstrap_compare import BOOTSTRAP_SEED  # noqa: E402

POPS = ("cut", "incl")
EPS = 0.02


def arms_pair(a, b, pop, t, rng, stratum=None):
    names = shot_names()
    per, parts = {}, []
    for s in SEEDS:
        test_files = set(json.loads((DATA / B13_POPS[pop]["manifest"].format(s=s)).read_text(encoding="utf-8"))["test_files"])
        A = load_arm(DATA / f".b16_{a}_{pop}_s{s}" / "comparison_errors_test.npz", names, test_files)[t]
        Bm = load_arm(DATA / f".b16_{b}_{pop}_s{s}" / "comparison_errors_test.npz", names, test_files)[t]
        if not (np.array_equal(A["shot"], Bm["shot"]) and np.array_equal(A["dt"], Bm["dt"], equal_nan=True)):
            raise SystemExit(f"FATAL: {pop} s{s} {t}: {a} and {b} not row-paired")
        m = np.ones(len(A["shot"]), bool) if stratum is None else stratum(A)
        per[str(s)] = compare(A["shot"][m], A["se_model"][m], Bm["se_model"][m], B13_POPS[pop]["delta"][t], rng, B_FULL)
        parts.append((A["shot"][m], A["se_model"][m], Bm["se_model"][m]))
    pooled = compare(np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts]),
                     np.concatenate([p[2] for p in parts]), B13_POPS[pop]["delta"][t], rng, B_FULL)
    return {"per_seed": per, "pooled": pooled}


def brief(r):
    return {"m1": r["m1_medae"]["skill_point"], "m1_ci": r["m1_medae"]["skill_ci95"], "m1_pass": r["m1_medae"]["pass"],
            "mse": r["mse"]["skill_point"], "mse_ci": r["mse"]["skill_ci95"], "n": r["n"]}


def main():
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    stale = lambda A: A["dt"] > 15.0
    anchor = json.loads((DATA / ".b16_anchor_rescore.json").read_text(encoding="utf-8"))["results"]
    out = {}
    for pop in POPS:
        node = {}
        for t in TARGETS:
            ga = arms_pair("v4ko", "v4kng", pop, t, rng)
            ha = arms_pair("v4kngh", "v4kng", pop, t, rng)
            hs = arms_pair("v4kngh", "v4kng", pop, t, rng, stale)
            node[t] = {
                "a_v4ko_minus_v4kng": {"pooled": brief(ga["pooled"]), "per_seed": {s: brief(v) for s, v in ga["per_seed"].items()}},
                "c_v4kngh_minus_v4kng_all": {"pooled": brief(ha["pooled"]), "per_seed": {s: brief(v) for s, v in ha["per_seed"].items()}},
                "c_v4kngh_minus_v4kng_stale": {"pooled": brief(hs["pooled"]), "per_seed": {s: brief(v) for s, v in hs["per_seed"].items()}},
                "b_anchor_median_over_persistence": anchor[pop]["pooled"][t]["anchor_vs_persistence"]["m5_quantile_ratio"]["50"],
            }
        # registered readings (T_i for (a) and (c); V_rot for (b))
        a = node["CES_TI"]["a_v4ko_minus_v4kng"]["pooled"]
        a_read = ("gate pays but the init blocked it" if a["m1"] >= EPS and a["m1_ci"][0] > 0 else
                  "gate harmful even open" if a["m1"] <= -EPS and a["m1_ci"][1] < 0 else "gate neutral")
        st = node["CES_TI"]["c_v4kngh_minus_v4kng_stale"]
        st_hits = sum(1 for s in SEEDS if st["per_seed"][str(s)]["m1"] >= 0.05 and st["per_seed"][str(s)]["m1_ci"][0] > 0)
        allr = node["CES_TI"]["c_v4kngh_minus_v4kng_all"]
        mse_worse = sum(1 for s in SEEDS if allr["per_seed"][str(s)]["mse_ci"][1] < 0)
        c_pass = st_hits >= 3 and allr["pooled"]["m1"] >= -EPS and mse_worse <= 1
        b = node["CES_VT"]["b_anchor_median_over_persistence"]
        b_read = ("P02 bound reachable, residual head spoils it" if 0.92 <= b <= 1.01 else
                  "learned OU filter cannot reach the P02 bound" if b >= 1.05 else "between bands")
        node["readings"] = {"a": a_read, "c": {"pass": c_pass, "stale_splits_ge_0.05": st_hits, "overall_m1": allr["pooled"]["m1"],
                                               "mse_sig_worse_vs_v4kng": mse_worse},
                            "b": {"vrot_median_over_persistence": b, "reading": b_read}}
        out[pop] = node
    (DATA / ".b16s_verdict.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    for pop in POPS:
        print(f"\n===== {pop} =====")
        for t in TARGETS:
            n = out[pop][t]
            for key in ("a_v4ko_minus_v4kng", "c_v4kngh_minus_v4kng_all", "c_v4kngh_minus_v4kng_stale"):
                p = n[key]["pooled"]
                per = " ".join(f"{n[key]['per_seed'][str(s)]['m1']:+.3f}{'*' if n[key]['per_seed'][str(s)]['m1_pass'] else ' '}" for s in SEEDS)
                mse = " ".join(f"{n[key]['per_seed'][str(s)]['mse']:+.3f}" for s in SEEDS)
                print(f"{t} {key:28s} M1 {per} | pooled {p['m1']:+.3f} [{p['m1_ci'][0]:+.3f},{p['m1_ci'][1]:+.3f}] n={p['n']} | MSE {mse} pooled {p['mse']:+.3f}")
            print(f"{t} (b) anchor median / persistence {n['b_anchor_median_over_persistence']:.3f}")
        print(json.dumps(out[pop]["readings"], indent=1))
    print(f"\nwrote {DATA / '.b16s_verdict.json'}")


if __name__ == "__main__":
    main()
