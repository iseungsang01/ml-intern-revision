"""B.24 verdict (PREREGISTRATION_B24.md sec. 4) over frozen artifacts -- no training.

Inputs (all must exist):
  data/.b24_vrot_noise_fit.json  fit_vrot_noise_b24.py (TRAIN only)
  data/.b24_rescore.json         rescore_b13.py --extra anchorf=.b24_anchor_{pop}_s{s} --out data/.b24_rescore.json
  data/.b{24,19,23}_anchor_{pop}_s{seed}/comparison_errors_test.npz
B.24-vs-B.23 pure anchor uses the rescore_b13 cluster bootstrap (B = 10^4).
Output: data/.b24_verdict.json.  Usage (repo root):  py ces_prediction/experiments/b16_physics/verdict_b24.py
"""

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from verdict_b16s import DATA, EPS, POPS, SEEDS, BOOTSTRAP_SEED, arms_pair, brief  # noqa: E402
from verdict_b18 import BAR, m1_rows, median_ratio  # noqa: E402

T = "CES_VT"


def main():
    fit = json.loads((DATA / ".b24_vrot_noise_fit.json").read_text(encoding="utf-8"))["fits"]
    res_all = json.loads((DATA / ".b24_rescore.json").read_text(encoding="utf-8"))["results"]
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    out = {}
    for pop in POPS:
        # control: the T_i cell is untouched, so its TEST errors must equal B.19's bit for bit
        ti_identical = all(np.array_equal(
            np.load(DATA / f".b24_anchor_{pop}_s{s}" / "comparison_errors_test.npz")["CES_TI_se_model"],
            np.load(DATA / f".b19_anchor_{pop}_s{s}" / "comparison_errors_test.npz")["CES_TI_se_model"]) for s in SEEDS)
        res = res_all[pop]
        vb = m1_rows(res, T, "anchorf_vs_model")
        vp = [res["per_seed"][str(s)][T]["anchorf_vs_persistence"]["all"]["m1_medae"] for s in SEEDS]
        d = arms_pair(".b24_anchor", ".b23_anchor", pop, T, rng)
        b = brief(d["pooled"])
        a_splits = sum(1 for r in vp if r["skill_point"] > 0 and r["skill_ci95"][0] > 0)
        c_splits = sum(1 for r in vb["per_seed"] if r["point"] >= BAR and r["ci"][0] > 0)
        out[pop] = {
            "ti_control_bit_identical": ti_identical,
            "fit": {str(s): {k: fit[f"{pop}_s{s}"][k] for k in ("sigma_n", "sigma_n_kms", "nu", "lam_per_step", "sigma_j",
                                                               "loglik_per_obs", "base_loglik_per_obs", "at_grid_edge")}
                    for s in SEEDS},
            "m1_vs_persistence": [{"point": r["skill_point"], "ci": r["skill_ci95"]} for r in vp],
            "median_over_persistence_split": [res["per_seed"][str(s)][T]["anchorf_vs_persistence"]["all"]
                                              ["m5_quantile_ratio"]["50"] for s in SEEDS],
            "median_over_persistence_pooled": median_ratio(res, T, "anchorf_vs_persistence"),
            "m1_vs_backbone": vb,
            "mse_vs_backbone": [res["per_seed"][str(s)][T]["anchorf_vs_model"]["all"]["mse"]["skill_point"] for s in SEEDS],
            "b24_minus_b23": {"pooled": b, "per_seed": {s: brief(v) for s, v in d["per_seed"].items()}},
            "H24-a": {"pass": a_splits >= 3, "splits": a_splits},
            "H24-b": {"pass": b["m1"] >= EPS and b["m1_ci"][0] > 0, "m1": b["m1"], "ci": b["m1_ci"]},
            "H24-c": {"pass": c_splits >= 3, "splits": c_splits},
        }
    if not all(out[p]["ti_control_bit_identical"] for p in POPS):
        raise SystemExit("FATAL: T_i errors differ from B.19 -- the patch touched more than the V_rot cell")
    lab = lambda k: ("unconditional" if out["cut"][k]["pass"] and out["incl"][k]["pass"] else "cut-conditional"
                     if out["cut"][k]["pass"] else "inclusive-conditional" if out["incl"][k]["pass"] else "not supported")
    out["rules"] = {
        "vrot_physics_median_estimator": lab("H24-a") if all(out[p]["H24-c"]["pass"] for p in POPS) else "not supported",
        "noise_fit_beats_b23": lab("H24-b"),
        "persistence_median_optimal_for_adaptive_t_filter": (not any(out[p]["H24-a"]["pass"] for p in POPS)
                                                              and not any(out[p]["H24-b"]["pass"] for p in POPS)),
    }
    (DATA / ".b24_verdict.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    for pop in POPS:
        n = out[pop]
        print(f"\n===== {pop} =====  T_i control identical: {n['ti_control_bit_identical']}")
        for s, f in n["fit"].items():
            print(f"  s{s}: sigma_n {f['sigma_n']:.3f} ({f['sigma_n_kms']:.2f} km/s) nu {f['nu']:.1f} "
                  f"ll/obs {f['loglik_per_obs']:+.4f} vs B.23 base {f['base_loglik_per_obs']:+.4f} edge={f['at_grid_edge']}")
        print("V_rot M1 vs persistence " + " ".join(f"{r['point']:+.3f}[{r['ci'][0]:+.3f},{r['ci'][1]:+.3f}]"
                                                  for r in n["m1_vs_persistence"]) + f" | H24-a {n['H24-a']['splits']}/4")
        print("V_rot median/persistence " + " ".join(f"{v:.3f}" for v in n["median_over_persistence_split"])
              + f" pooled {n['median_over_persistence_pooled']:.3f}")
        print("V_rot M1 vs backbone " + " ".join(f"{r['point']:+.3f}" for r in n["m1_vs_backbone"]["per_seed"])
              + f" | H24-c {n['H24-c']['splits']}/4 | MSE vs backbone " + " ".join(f"{v:+.3f}" for v in n["mse_vs_backbone"]))
        p = n["b24_minus_b23"]["pooled"]
        print(f"B.24 - B.23 pooled M1 {p['m1']:+.3f} [{p['m1_ci'][0]:+.3f},{p['m1_ci'][1]:+.3f}] MSE {p['mse']:+.3f}"
              f" | H24-b {n['H24-b']['pass']}")
    print(json.dumps(out["rules"], indent=1))
    print(f"\nwrote {DATA / '.b24_verdict.json'}")


if __name__ == "__main__":
    main()
