"""B.16 verdict (PREREGISTRATION_B16.md sec. 3-4) over frozen artifacts -- no training.

Inputs (all must exist):
  data/.b16_summary.json          runner summary (paired MSE vs backbone, physics, per-split winrate)
  data/.b16_rescore.json          rescore_b13.py --extra v4k=... v4kng=... --out data/.b16_rescore.json
  data/.b16_{arm}_{pop}_s{seed}/comparison_errors_test.npz   and the backbone npz per split

Computes what the two inputs do not already hold: the quiet-tercile discharge win rate vs the
causal GP pooled over the four splits (H16-b, sec. 8an definition: per discharge >= 20 rows,
tercile of the truth's spread, pooled before the tercile cut) and the v4k - v4kng paired M1
(H16-e, same cluster bootstrap as rescore_b13). Then applies H16-a..f and decision rules 1-4
verbatim and writes data/.b16_verdict.json.

Usage (repo root):  py ces_prediction/experiments/b16_physics/verdict_b16.py
"""

import json
import math
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

ARMS = ("v4k", "v4kng")
POPS = ("cut", "incl")
GRID_DT_S = 0.010
# KalmanAnchor init (model_seq_v4k.py): (rho0, sigma_s0, sigma_n0) in normalized units
INIT = {"CES_TI": (0.88, 0.35, 0.30), "CES_VT": (0.90, 0.60, 0.40)}
MIN_ROWS = 20


def arm_dir(arm, pop, s):
    return DATA / f".b16_{arm}_{pop}_s{s}"


def backbone_dir(pop, s):
    return DATA / B13_POPS[pop]["model"].format(s=s)


def pooled_quiet_winrate(dirs, target):
    """sec. 8an readout: per-discharge win vs causal GP, terciles of the truth's spread, pooled."""
    rows = []
    for d in dirs:
        z = np.load(d / "comparison_errors_test.npz")
        shot, y = z[f"{target}_shot"], z[f"{target}_y_true"].astype(float)
        sm, sg = z[f"{target}_se_model"].astype(float), z[f"{target}_se_gp_causal"].astype(float)
        for s in np.unique(shot):
            k = shot == s
            if k.sum() < MIN_ROWS:
                continue
            rows.append((float(np.std(y[k])), float(sm[k].mean() < sg[k].mean())))
    rows.sort(key=lambda r: r[0])
    n = len(rows)
    terc = [rows[: n // 3], rows[n // 3: 2 * n // 3], rows[2 * n // 3:]]
    return {"n_shots": n, "win_rate_quiet_mid_variable": [float(np.mean([r[1] for r in g])) for g in terc],
            "win_rate_all": float(np.mean([r[1] for r in rows]))}


def arm_vs_arm_m1(pop, target, rng):
    """v4k vs v4kng paired per split and pooled (rescore_b13.compare: MSE, M1..M5)."""
    names = shot_names()
    per, parts = {}, []
    for s in SEEDS:
        manifest = json.loads((DATA / B13_POPS[pop]["manifest"].format(s=s)).read_text(encoding="utf-8"))
        test_files = set(manifest["test_files"])
        a = load_arm(arm_dir("v4k", pop, s) / "comparison_errors_test.npz", names, test_files)[target]
        b = load_arm(arm_dir("v4kng", pop, s) / "comparison_errors_test.npz", names, test_files)[target]
        if not (np.array_equal(a["shot"], b["shot"]) and np.array_equal(a["dt"], b["dt"], equal_nan=True)):
            raise SystemExit(f"FATAL: {pop} s{s} {target}: v4k and v4kng are not row-paired")
        per[str(s)] = compare(a["shot"], a["se_model"], b["se_model"], B13_POPS[pop]["delta"][target], rng, B_FULL)
        parts.append((a["shot"], a["se_model"], b["se_model"]))
    pooled = compare(np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts]),
                     np.concatenate([p[2] for p in parts]), B13_POPS[pop]["delta"][target], rng, B_FULL)
    return {"per_seed": per, "pooled": pooled}


def main():
    summary = json.loads((DATA / ".b16_summary.json").read_text(encoding="utf-8"))
    rescore = json.loads((DATA / ".b16_rescore.json").read_text(encoding="utf-8"))
    for arm in ARMS:
        for pop in POPS:
            if sorted(summary["arms"][arm][pop]) != sorted(str(s) for s in SEEDS):
                raise SystemExit(f"FATAL: {arm}/{pop} is not 4/4 complete in .b16_summary.json")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    out = {"inputs": {"summary": ".b16_summary.json", "rescore": ".b16_rescore.json"}, "pops": {}}

    for pop in POPS:
        res = rescore["results"][pop]
        node = {"targets": {}, "quiet_winrate": {}, "gate_share": {}}
        # --- quiet-tercile win rate vs causal GP, pooled over the 4 splits (H16-b)
        for t in TARGETS:
            node["quiet_winrate"][t] = {
                "backbone": pooled_quiet_winrate([backbone_dir(pop, s) for s in SEEDS], t),
                **{arm: pooled_quiet_winrate([arm_dir(arm, pop, s) for s in SEEDS], t) for arm in ARMS}}
        # --- v4k - v4kng (H16-e)
        for t in TARGETS:
            node["gate_share"][t] = arm_vs_arm_m1(pop, t, rng)
        # --- per target readouts
        for t in TARGETS:
            tn = {}
            for arm in ARMS:
                rows = summary["arms"][arm][pop]
                per = [res["per_seed"][str(s)][t][f"{arm}_vs_model"]["all"] for s in SEEDS]
                pooled = res["pooled"][t][f"{arm}_vs_model"]
                pers = res["pooled"][t][f"{arm}_vs_persistence"]
                tn[arm] = {
                    "m1_vs_backbone": [{"seed": s, "point": p["m1_medae"]["skill_point"],
                                        "ci": p["m1_medae"]["skill_ci95"], "pass": p["m1_medae"]["pass"]}
                                       for s, p in zip(SEEDS, per)],
                    "m1_pooled": pooled["m1_medae"],
                    "mse_paired_vs_backbone": [{"seed": s, "point": rows[str(s)][t]["paired_vs_control"],
                                                "ci": rows[str(s)][t]["ci95"],
                                                "arm_better": rows[str(s)][t]["arm_better"],
                                                "control_better": rows[str(s)][t]["control_better"]}
                                               for s in SEEDS],
                    "mse_pooled_mean": float(np.mean([rows[str(s)][t]["paired_vs_control"] for s in SEEDS])),
                    "median_ae_over_persistence": pers["m5_quantile_ratio"]["50"],
                    "backbone_median_ae_over_persistence": res["pooled"][t]["model_vs_persistence"]["m5_quantile_ratio"]["50"],
                    "skill_vs_pchip": [rows[str(s)][t]["skill_vs_pchip"] for s in SEEDS],
                    "backbone_skill_vs_pchip": [res["per_seed"][str(s)][t]["model_vs_pchip"]["all"]["mse"]["skill_point"]
                                                for s in SEEDS],
                    "epochs": [rows[str(s)]["epochs"] for s in SEEDS],
                    "physics": [rows[str(s)]["physics"][t] for s in SEEDS],
                }
            node["targets"][t] = tn
        # --- hypotheses
        ti, vt = node["targets"]["CES_TI"], node["targets"]["CES_VT"]
        h = {}
        a_hits = sum(1 for r in ti["v4k"]["m1_vs_backbone"] if r["point"] >= 0.05 and r["ci"][0] > 0)
        h["H16-a"] = {"pass": a_hits >= 3, "splits_m1_ge_0.05_and_ci_gt_0": a_hits,
                      "m1_points": [r["point"] for r in ti["v4k"]["m1_vs_backbone"]]}
        qw = node["quiet_winrate"]["CES_TI"]
        dq = qw["v4k"]["win_rate_quiet_mid_variable"][0] - qw["backbone"]["win_rate_quiet_mid_variable"][0]
        h["H16-b"] = {"pass": dq >= 0.05, "quiet_winrate_v4k": qw["v4k"]["win_rate_quiet_mid_variable"][0],
                      "quiet_winrate_backbone": qw["backbone"]["win_rate_quiet_mid_variable"][0], "delta": dq}
        h["H16-c"] = {}
        for arm in ARMS:
            worse = sum(r["control_better"] for r in ti[arm]["mse_paired_vs_backbone"])
            mean = ti[arm]["mse_pooled_mean"]
            h["H16-c"][arm] = {"pass": worse <= 1 and mean >= -0.02, "sig_worse": worse, "pooled_mean": mean,
                               "rejected_rule4": worse >= 2}
        d_hits = sum(1 for r in vt["v4k"]["m1_vs_backbone"] if r["point"] >= 0.05 and r["ci"][0] > 0)
        d_worse = sum(r["control_better"] for r in vt["v4k"]["mse_paired_vs_backbone"])
        h["H16-d"] = {"pass": d_hits >= 3 and d_worse == 0, "splits_m1_ge_0.05_and_ci_gt_0": d_hits,
                      "mse_sig_worse": d_worse, "m1_points": [r["point"] for r in vt["v4k"]["m1_vs_backbone"]]}
        gs = node["gate_share"]["CES_TI"]["pooled"]["m1_medae"]
        gs_seeds = [node["gate_share"]["CES_TI"]["per_seed"][str(s)]["m1_medae"]["skill_point"] for s in SEEDS]
        dq_gate = qw["v4k"]["win_rate_quiet_mid_variable"][0] - qw["v4kng"]["win_rate_quiet_mid_variable"][0]
        h["H16-e"] = {"pass": gs["skill_point"] >= 0.02 or dq_gate >= 0.05, "m1_v4k_minus_v4kng_pooled": gs,
                      "m1_per_seed": gs_seeds, "quiet_winrate_delta_v4k_minus_v4kng": dq_gate}
        f = {}
        for t in TARGETS:
            rho0, ss0, sn0 = INIT[t]
            tau0 = -GRID_DT_S / math.log(rho0) * 1e3
            rows = []
            for arm in ARMS:
                for p in node["targets"][t][arm]["physics"]:
                    rows.append(p)
            taus = [p["tau_ms"] for p in rows]
            # learned sigma in normalized units = physical / target std; the runner only stored the
            # physical values, so take the ratio to the init through the stored target std per run
            ratios = []
            for arm in ARMS:
                for s, p in zip(SEEDS, node["targets"][t][arm]["physics"]):
                    pp = json.loads((arm_dir(arm, pop, s) / "physics_params.json").read_text(encoding="utf-8"))
                    std = pp["target_std_used"][0 if t == "CES_TI" else 1]
                    ratios.append({"sigma_n": p["sigma_n"] / std / sn0, "sigma_s": p["sigma_s"] / std / ss0})
            ok = all(20 <= v <= 300 for v in taus) and all(0.3 <= r[k] <= 3 for r in ratios for k in r)
            f[t] = {"pass": ok, "tau_ms_range": [min(taus), max(taus)], "tau0_ms": tau0,
                    "sigma_ratio_range": {k: [min(r[k] for r in ratios), max(r[k] for r in ratios)] for k in ("sigma_n", "sigma_s")}}
        h["H16-f"] = f
        node["hypotheses"] = h
        out["pops"][pop] = node

    # --- decision rules (sec. 4)
    cut, incl = out["pops"]["cut"]["hypotheses"], out["pops"]["incl"]["hypotheses"]
    rules = {}
    adopt_cut = cut["H16-a"]["pass"] and cut["H16-c"]["v4k"]["pass"]
    adopt_incl = incl["H16-a"]["pass"] and incl["H16-c"]["v4k"]["pass"]
    rules["1_physics_rung_adopted"] = {"cut": adopt_cut, "incl": adopt_incl,
                                       "status": ("unconditional" if adopt_cut and adopt_incl else
                                                  "cut-conditional" if adopt_cut else "not adopted")}
    sign44 = all(all(r["point"] > 0 for r in out["pops"][p]["targets"]["CES_TI"]["v4k"]["mse_paired_vs_backbone"])
                 for p in POPS)
    rules["2_backbone_replaced"] = {"pass": adopt_cut and adopt_incl and sign44 and cut["H16-d"]["pass"],
                                    "ti_mse_sign_4of4_both_pops": sign44}
    rules["3_gate_negative_report_v4kng"] = {"cut": not cut["H16-e"]["pass"], "incl": not incl["H16-e"]["pass"]}
    rules["4_rejected_arms"] = {p: [arm for arm in ARMS if out["pops"][p]["hypotheses"]["H16-c"][arm]["rejected_rule4"]]
                                for p in POPS}
    out["rules"] = rules
    (DATA / ".b16_verdict.json").write_text(json.dumps(out, indent=1), encoding="utf-8")

    # --- print
    for pop in POPS:
        n = out["pops"][pop]
        print(f"\n===== {pop} =====")
        for t in TARGETS:
            for arm in ARMS:
                a = n["targets"][t][arm]
                m1 = " ".join(f"{r['point']:+.3f}{'*' if r['pass'] else ' '}" for r in a["m1_vs_backbone"])
                mse = " ".join(f"{r['point']:+.3f}{'+' if r['arm_better'] else '-' if r['control_better'] else ' '}"
                               for r in a["mse_paired_vs_backbone"])
                print(f"{t} {arm:5s} M1 vs backbone {m1} | pooled {a['m1_pooled']['skill_point']:+.3f} "
                      f"[{a['m1_pooled']['skill_ci95'][0]:+.3f},{a['m1_pooled']['skill_ci95'][1]:+.3f}] | "
                      f"MSE paired {mse} mean {a['mse_pooled_mean']:+.3f} | medAE/pers {a['median_ae_over_persistence']:.3f} "
                      f"(backbone {a['backbone_median_ae_over_persistence']:.3f}) | epochs {a['epochs']}")
            q = n["quiet_winrate"][t]
            print(f"{t} quiet/mid/variable win vs causal GP: backbone "
                  + "/".join(f"{v:.2f}" for v in q["backbone"]["win_rate_quiet_mid_variable"])
                  + "  v4k " + "/".join(f"{v:.2f}" for v in q["v4k"]["win_rate_quiet_mid_variable"])
                  + "  v4kng " + "/".join(f"{v:.2f}" for v in q["v4kng"]["win_rate_quiet_mid_variable"])
                  + f"  (n={q['backbone']['n_shots']})")
            g = n["gate_share"][t]["pooled"]["m1_medae"]
            print(f"{t} v4k - v4kng M1 pooled {g['skill_point']:+.3f} [{g['skill_ci95'][0]:+.3f},{g['skill_ci95'][1]:+.3f}]")
        print(json.dumps(n["hypotheses"], indent=1)[:3000])
    print("\n== rules ==")
    print(json.dumps(rules, indent=1))
    print(f"\nwrote {DATA / '.b16_verdict.json'}")


if __name__ == "__main__":
    main()
