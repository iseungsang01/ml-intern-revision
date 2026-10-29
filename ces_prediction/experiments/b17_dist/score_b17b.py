"""B.17b scoring (PREREGISTRATION_B17B.md): Student-t head `v2t` against the mixture `v2m2`, both
trained under the raised epoch cap (prefix .b17b), on the frozen TEST rows.

Reuses score_b17's closed forms (Gaussian mixture CRPS / NLL / CDF) and adds the Student-t ones
(CRPS: Jordan, Krueger & Lerch 2019, scoringRules `crps_t`). Per population, split and target:

  CRPS skill vs R0 (backbone point + oracle homoscedastic sigma) and vs R1 (arm mean + oracle sigma)
  NLL, PIT coverage 50 / 90 %, sd relations (energy, staleness), learned nu summary (v2t)
  v2t vs v2m2 paired CRPS (same rows), v2m2(cap 100) vs v2m2(cap 60, B.17) paired CRPS (cap readout)
  MSE guard from the runner summary (.b17b_runner_summary.json)

Then the B.17b hypotheses and rules verbatim. Output: data/.b17b_summary.json.
Usage (repo root):  py ces_prediction/experiments/b17_dist/score_b17b.py
"""

import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.special import beta as _beta
from scipy.stats import norm, spearmanr, t as student_t

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
DATA = REPO / "data"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "ces_prediction"))
from bootstrap_compare import BOOTSTRAP_SEED  # noqa: E402
from score_b17 import crps_mixture, nll_mixture, cdf_mixture, moments, skill, PRACTICAL_EPS  # noqa: E402

SEEDS = (42, 1, 7, 123)
POPS = ("cut", "incl")
ARMS = ("v2t", "v2m2")
TARGETS = ("CES_TI", "CES_VT")
PREFIX = ".b17b"


# ---- Student-t closed forms ----------------------------------------------------------

def crps_t(y, mu, sigma, nu):
    """CRPS of a location-scale Student-t (nu > 1), vectorised over rows."""
    y, mu, sigma, nu = (np.asarray(v, dtype=np.float64).reshape(-1) for v in (y, mu, sigma, nu))
    z = (y - mu) / sigma
    F = student_t.cdf(z, nu)
    f = student_t.pdf(z, nu)
    term = z * (2.0 * F - 1.0) + 2.0 * f * (nu + z ** 2) / (nu - 1.0) \
        - 2.0 * np.sqrt(nu) / (nu - 1.0) * _beta(0.5, nu - 0.5) / _beta(0.5, nu / 2.0) ** 2
    return sigma * term


def nll_t(y, mu, sigma, nu):
    y, mu, sigma, nu = (np.asarray(v, dtype=np.float64).reshape(-1) for v in (y, mu, sigma, nu))
    z = (y - mu) / sigma
    return -(student_t.logpdf(z, nu) - np.log(sigma))


def cdf_t(y, mu, sigma, nu):
    y, mu, sigma, nu = (np.asarray(v, dtype=np.float64).reshape(-1) for v in (y, mu, sigma, nu))
    return student_t.cdf((y - mu) / sigma, nu)


# ---- per run -----------------------------------------------------------------------

def load(prefix, arm, pop, s):
    return np.load(DATA / f"{prefix}_{arm}_{pop}_s{s}" / "dist_test.npz")


def arm_scores(z, t):
    """Per-row CRPS / NLL / PIT / mean / sd of an arm's dump (mixture or t)."""
    y, pi, mu, sg = z[f"{t}_y"], z[f"{t}_pi"], z[f"{t}_mu"], z[f"{t}_sigma"]
    if f"{t}_nu" in z.files:
        nu = z[f"{t}_nu"][:, 0]
        crps = crps_t(y, mu[:, 0], sg[:, 0], nu)
        nll = nll_t(y, mu[:, 0], sg[:, 0], nu)
        pit = cdf_t(y, mu[:, 0], sg[:, 0], nu)
        mean = mu[:, 0]
        sd = sg[:, 0] * np.sqrt(nu / (nu - 2.0))
        extra = {"nu_q10_50_90": [float(v) for v in np.percentile(nu, [10, 50, 90])]}
    else:
        crps = crps_mixture(y, pi, mu, sg)
        nll = nll_mixture(y, pi, mu, sg)
        pit = cdf_mixture(y, pi, mu, sg)
        mean, sd = moments(pi, mu, sg)
        extra = {"min_pi_median": float(np.median(pi.min(axis=1)))}
    return {"crps": crps, "nll": nll, "pit": pit, "mean": mean, "sd": sd, "extra": extra}


def score_target(z, t, rng):
    y = z[f"{t}_y"]
    shot, bb, en, dt, peak = z[f"{t}_shot"], z[f"{t}_backbone_pred"], z[f"{t}_energy"], z[f"{t}_dt_ms"], z[f"{t}_is_peak"].astype(bool)
    a = arm_scores(z, t)
    one = np.ones((len(y), 1))
    s0 = math.sqrt(float(np.mean(z[f"{t}_se_control"])))
    s1 = math.sqrt(float(np.mean(z[f"{t}_se_model"])))
    c_r0 = crps_mixture(y, one, bb[:, None], np.full((len(y), 1), s0))
    c_r1 = crps_mixture(y, one, a["mean"][:, None], np.full((len(y), 1), s1))
    pit = a["pit"]
    res = {
        "n": int(len(y)),
        "crps_mean": {"arm": float(a["crps"].mean()), "r0": float(c_r0.mean()), "r1": float(c_r1.mean())},
        "nll_mean": float(a["nll"].mean()),
        "crps_skill_vs_r0": skill(shot, a["crps"], c_r0, rng),
        "crps_skill_vs_r1": skill(shot, a["crps"], c_r1, rng),
        "coverage": {"50": float(np.mean((pit >= 0.25) & (pit <= 0.75))), "90": float(np.mean((pit >= 0.05) & (pit <= 0.95)))},
        "sd_relations": {"spearman_sd_energy": float(spearmanr(a["sd"], en).correlation),
                         "spearman_sd_staleness": float(spearmanr(a["sd"], dt).correlation),
                         "sd_peak_over_nonpeak": float(a["sd"][peak].mean() / a["sd"][~peak].mean()) if peak.any() and (~peak).any() else float("nan"),
                         "sd_median_phys": float(np.median(a["sd"]))},
        **a["extra"],
        "_rows": {"shot": shot, "row": z[f"{t}_row"], "crps": a["crps"], "crps_r0": c_r0, "crps_r1": c_r1},
    }
    return res


def pair(rows_a, rows_b, rng, label):
    if not (np.array_equal(rows_a["shot"], rows_b["shot"]) and np.array_equal(rows_a["row"], rows_b["row"])):
        raise SystemExit(f"FATAL: rows differ in {label}")
    return skill(rows_a["shot"], rows_a["crps"], rows_b["crps"], rng)


def main():
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    runner = json.loads((DATA / f"{PREFIX}_runner_summary.json").read_text(encoding="utf-8"))
    out = {"pops": {}}
    for pop in POPS:
        node = {"targets": {}}
        for t in TARGETS:
            tn = {arm: {"per_seed": {}} for arm in ARMS}
            pooled = {arm: [] for arm in ARMS}
            pair_rows = {"v2t_vs_v2m2": [], "v2m2_cap100_vs_cap60": []}
            for s in SEEDS:
                zs = {arm: load(PREFIX, arm, pop, s) for arm in ARMS}
                res = {arm: score_target(zs[arm], t, rng) for arm in ARMS}
                old = score_target(load(".b17", "v2m2", pop, s), t, rng)
                for arm in ARMS:
                    r = res[arm]
                    rows = r.pop("_rows")
                    pooled[arm].append(rows)
                    tn[arm]["per_seed"][str(s)] = r
                rows_t, rows_m, rows_old = pooled["v2t"][-1], pooled["v2m2"][-1], old.pop("_rows")
                tn["v2t"]["per_seed"][str(s)]["crps_skill_vs_v2m2"] = pair(rows_t, rows_m, rng, f"{pop} s{s} {t} v2t/v2m2")
                tn["v2m2"]["per_seed"][str(s)]["crps_skill_vs_cap60"] = pair(rows_m, rows_old, rng, f"{pop} s{s} {t} cap")
                pair_rows["v2t_vs_v2m2"].append((rows_t, rows_m))
                pair_rows["v2m2_cap100_vs_cap60"].append((rows_m, rows_old))
            for arm in ARMS:
                rows = pooled[arm]
                cat = lambda k: np.concatenate([r[k] for r in rows])
                tn[arm]["pooled"] = {"crps_skill_vs_r0": skill(cat("shot"), cat("crps"), cat("crps_r0"), rng),
                                     "crps_skill_vs_r1": skill(cat("shot"), cat("crps"), cat("crps_r1"), rng)}
            for key, prs in pair_rows.items():
                shot = np.concatenate([a["shot"] for a, _ in prs])
                tn["v2t" if key.startswith("v2t") else "v2m2"]["pooled"][key] = skill(
                    shot, np.concatenate([a["crps"] for a, _ in prs]), np.concatenate([b["crps"] for _, b in prs]), rng)
            for arm in ARMS:
                rr = runner["arms"][arm][pop]
                worse = sum(rr[str(s)][t]["control_better"] for s in SEEDS)
                pts = [rr[str(s)][t]["paired_vs_control"] for s in SEEDS]
                tn[arm]["mse_guard"] = {"sig_worse": worse, "points": pts, "pooled_mean": float(np.mean(pts)),
                                        "pass": worse <= 1 and float(np.mean(pts)) >= -PRACTICAL_EPS, "rejected": worse >= 2,
                                        "epochs": [rr[str(s)]["epochs"] for s in SEEDS]}
            node["targets"][t] = tn
        # ---- hypotheses (PREREGISTRATION_B17B.md sec. 4)
        ti, vt = node["targets"]["CES_TI"], node["targets"]["CES_VT"]
        h = {}
        for arm in ARMS:
            a = ti[arm]
            cov = [a["per_seed"][str(s)]["coverage"] for s in SEEDS]
            c_ok = sum(1 for c in cov if 0.85 <= c["90"] <= 0.95 and 0.45 <= c["50"] <= 0.55)
            r0_ok = sum(1 for s in SEEDS if a["per_seed"][str(s)]["crps_skill_vs_r0"]["pass"])
            guard_ok = all(node["targets"][tt][arm]["mse_guard"]["pass"] for tt in TARGETS)
            h[f"Hb-guards-{arm}"] = {"crps_r0_splits": r0_ok, "crps_r0_pooled_pass": a["pooled"]["crps_skill_vs_r0"]["pass"],
                                    "crps_r1_pooled_pass": a["pooled"]["crps_skill_vs_r1"]["pass"],
                                    "calibration_splits": c_ok, "mse_guard_both_targets": guard_ok,
                                    "pass": r0_ok >= 3 and a["pooled"]["crps_skill_vs_r0"]["pass"] and a["pooled"]["crps_skill_vs_r1"]["pass"] and c_ok >= 3 and guard_ok}
        tv = ti["v2t"]["pooled"]["v2t_vs_v2m2"]
        h["Hb-1_t_vs_mixture"] = {"pooled": tv, "tie": abs(tv["skill_point"]) < PRACTICAL_EPS or not tv["pass"],
                                  "t_better": tv["pass"] and tv["skill_point"] >= PRACTICAL_EPS,
                                  "mixture_better": tv["pass"] and tv["skill_point"] <= -PRACTICAL_EPS}
        nus = [ti["v2t"]["per_seed"][str(s)]["nu_q10_50_90"] for s in SEEDS]
        h["Hb-2_nu"] = {"nu_median_per_split": [n[1] for n in nus], "in_2_to_8": all(2.0 <= n[1] <= 8.0 for n in nus)}
        cap = ti["v2m2"]["pooled"]["v2m2_cap100_vs_cap60"]
        h["Hb-3_cap"] = {"pooled": cap, "cap_mattered": cap["pass"] and cap["skill_point"] >= PRACTICAL_EPS}
        node["hypotheses"] = h
        out["pops"][pop] = node
    rules = {}
    for pop in POPS:
        h = out["pops"][pop]["hypotheses"]
        passing = [arm for arm in ARMS if h[f"Hb-guards-{arm}"]["pass"]]
        if not passing:
            named = None
        elif len(passing) == 1:
            named = passing[0]
        else:
            named = "v2t" if h["Hb-1_t_vs_mixture"]["t_better"] else "v2m2" if h["Hb-1_t_vs_mixture"]["mixture_better"] else "v2t"  # tie -> fewer parameters
        rules[pop] = {"arms_passing_guards": passing, "named_arm": named, "claim_supported": named is not None}
    rules["status"] = ("unconditional" if rules["cut"]["claim_supported"] and rules["incl"]["claim_supported"]
                       else "cut-conditional" if rules["cut"]["claim_supported"]
                       else "inclusive-conditional" if rules["incl"]["claim_supported"] else "not supported")
    out["rules"] = rules
    (DATA / f"{PREFIX}_summary.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")

    def f(r):
        return f"{r['skill_point']:+.3f} [{r['skill_ci95'][0]:+.3f},{r['skill_ci95'][1]:+.3f}]{'*' if r['pass'] else ' '}"
    for pop in POPS:
        print(f"\n===== {pop} =====")
        for t in TARGETS:
            for arm in ARMS:
                a = out["pops"][pop]["targets"][t][arm]
                per = " ".join(f"{a['per_seed'][str(s)]['crps_skill_vs_r0']['skill_point']:+.3f}{'*' if a['per_seed'][str(s)]['crps_skill_vs_r0']['pass'] else ' '}" for s in SEEDS)
                cov = " ".join(f"{a['per_seed'][str(s)]['coverage']['50']:.2f}/{a['per_seed'][str(s)]['coverage']['90']:.2f}" for s in SEEDS)
                print(f"{t} {arm:4s} CRPS vs R0 {per} | pooled R0 {f(a['pooled']['crps_skill_vs_r0'])} R1 {f(a['pooled']['crps_skill_vs_r1'])} | cov50/90 {cov} | MSE guard {[round(x,3) for x in a['mse_guard']['points']]} worse {a['mse_guard']['sig_worse']} epochs {a['mse_guard']['epochs']}")
            print(f"{t}      v2t vs v2m2 pooled {f(out['pops'][pop]['targets'][t]['v2t']['pooled']['v2t_vs_v2m2'])} | v2m2 cap100 vs cap60 {f(out['pops'][pop]['targets'][t]['v2m2']['pooled']['v2m2_cap100_vs_cap60'])}")
            print(f"{t}      v2t nu q10/50/90 per split: " + "  ".join(str([round(v,1) for v in out['pops'][pop]['targets'][t]['v2t']['per_seed'][str(s)]['nu_q10_50_90']]) for s in SEEDS))
        print(json.dumps({k: v for k, v in out["pops"][pop]["hypotheses"].items()}, indent=1, default=float)[:2500])
    print("\n== rules ==")
    print(json.dumps(rules, indent=1))
    print(f"\nwrote {DATA / (PREFIX + '_summary.json')}")


if __name__ == "__main__":
    main()
