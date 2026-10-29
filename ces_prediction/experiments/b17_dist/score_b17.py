"""B.17 scoring (PREREGISTRATION_B17.md sec. 3-4) over frozen `dist_test.npz` artifacts.

No training, no inference. For every population, split and target:

  CRPS / NLL of the arm's predictive distribution, paired row-for-row against two
  references that need no training --
    R0  N(backbone point, sigma_0)   sigma_0 = the frozen backbone's own TEST RMSE (oracle scale)
    R1  N(arm mean,       sigma_1)   sigma_1 = the arm's own TEST RMSE (isolates the
                                     heteroscedastic part from any shift of the mean)
  skill = 1 - sum CRPS_arm / sum CRPS_ref, shot-cluster bootstrap (bootstrap_compare, B = 10^4);
  PIT coverage of the central 50 % / 90 % intervals;
  the physics readout: Spearman(sd, fast-channel increment energy) and Spearman(sd, staleness);
  v2m2 only: bimodal-row fraction (min pi >= 0.1 and |mu1 - mu2| >= 2 max sigma), its
  enrichment in event (peak) rows, and v2m2 - v2g paired CRPS.

Then H17-a..e and the decision rules, verbatim from the registration. H17-e (the MSE guard
on the posterior mean) reads the runner summary `.b17_runner_summary.json`.

Usage (repo root):  py ces_prediction/experiments/b17_dist/score_b17.py
Output: data/.b17_summary.json (+ printed table).
"""

import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.stats import norm, spearmanr

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
DATA = REPO / "data"
sys.path.insert(0, str(REPO / "ces_prediction"))
from bootstrap_compare import BOOTSTRAP_SEED, _bootstrap  # noqa: E402

SEEDS = (42, 1, 7, 123)
POPS = ("cut", "incl")
ARMS = ("v2g", "v2m2")
TARGETS = ("CES_TI", "CES_VT")
PRACTICAL_EPS = 0.02
BIMODAL_MIN_PI = 0.1
BIMODAL_SEP = 2.0


# ---- closed forms ------------------------------------------------------------------

def _A(m, s):
    """E|X - 0| for X ~ N(m, s^2): m (2 Phi(m/s) - 1) + 2 s phi(m/s)."""
    z = m / s
    return m * (2.0 * norm.cdf(z) - 1.0) + 2.0 * s * norm.pdf(z)


def crps_mixture(y, pi, mu, sigma):
    """CRPS of a Gaussian mixture (Grimit et al. 2006), vectorised over rows. (n,), (n,K)x3."""
    y = np.asarray(y, dtype=np.float64)[:, None]
    pi, mu, sigma = (np.asarray(v, dtype=np.float64) for v in (pi, mu, sigma))
    first = (pi * _A(y - mu, sigma)).sum(axis=1)
    s2 = np.sqrt(sigma[:, :, None] ** 2 + sigma[:, None, :] ** 2)
    second = (pi[:, :, None] * pi[:, None, :] * _A(mu[:, :, None] - mu[:, None, :], s2)).sum(axis=(1, 2))
    return first - 0.5 * second


def nll_mixture(y, pi, mu, sigma):
    y = np.asarray(y, dtype=np.float64)[:, None]
    z = (y - mu) / sigma
    logp = np.log(pi) - 0.5 * z ** 2 - np.log(sigma) - 0.5 * math.log(2 * math.pi)
    m = logp.max(axis=1, keepdims=True)
    return -(m[:, 0] + np.log(np.exp(logp - m).sum(axis=1)))


def cdf_mixture(y, pi, mu, sigma):
    return (pi * norm.cdf((np.asarray(y)[:, None] - mu) / sigma)).sum(axis=1)


def moments(pi, mu, sigma):
    m = (pi * mu).sum(axis=1)
    var = (pi * (sigma ** 2 + (mu - m[:, None]) ** 2)).sum(axis=1)
    return m, np.sqrt(var)


def skill(shot, cost_a, cost_b, rng):
    r = _bootstrap(shot, cost_a, cost_b, rng)
    return {k: r[k] for k in ("skill_point", "skill_ci95", "pass", "n_shots")}


# ---- per run ---------------------------------------------------------------------

def load(arm, pop, s):
    return np.load(DATA / f".b17_{arm}_{pop}_s{s}" / "dist_test.npz")


def score_target(z, t, rng):
    y, pi, mu, sg = z[f"{t}_y"], z[f"{t}_pi"], z[f"{t}_mu"], z[f"{t}_sigma"]
    shot, bb, en, dt, peak = z[f"{t}_shot"], z[f"{t}_backbone_pred"], z[f"{t}_energy"], z[f"{t}_dt_ms"], z[f"{t}_is_peak"].astype(bool)
    mean, sd = moments(pi, mu, sg)
    one = np.ones((len(y), 1))
    s0 = math.sqrt(float(np.mean(z[f"{t}_se_control"])))
    s1 = math.sqrt(float(np.mean(z[f"{t}_se_model"])))
    c_arm = crps_mixture(y, pi, mu, sg)
    c_r0 = crps_mixture(y, one, bb[:, None], np.full((len(y), 1), s0))
    c_r1 = crps_mixture(y, one, mean[:, None], np.full((len(y), 1), s1))
    n_arm = nll_mixture(y, pi, mu, sg)
    n_r0 = nll_mixture(y, one, bb[:, None], np.full((len(y), 1), s0))
    n_r1 = nll_mixture(y, one, mean[:, None], np.full((len(y), 1), s1))
    pit = cdf_mixture(y, pi, mu, sg)
    pit0 = norm.cdf((y - bb) / s0)
    res = {
        "n": int(len(y)), "sigma_ref_backbone": s0, "sigma_ref_arm": s1,
        "crps_mean": {"arm": float(c_arm.mean()), "r0": float(c_r0.mean()), "r1": float(c_r1.mean())},
        "nll_mean": {"arm": float(n_arm.mean()), "r0": float(n_r0.mean()), "r1": float(n_r1.mean())},
        "crps_skill_vs_r0": skill(shot, c_arm, c_r0, rng),
        "crps_skill_vs_r1": skill(shot, c_arm, c_r1, rng),
        "nll_diff_vs_r0": float((n_arm - n_r0).mean()),
        "coverage": {"50": float(np.mean((pit >= 0.25) & (pit <= 0.75))),
                     "90": float(np.mean((pit >= 0.05) & (pit <= 0.95))),
                     "r0_50": float(np.mean((pit0 >= 0.25) & (pit0 <= 0.75))),
                     "r0_90": float(np.mean((pit0 >= 0.05) & (pit0 <= 0.95)))},
        "sd_relations": {"spearman_sd_energy": float(spearmanr(sd, en).correlation),
                         "spearman_sd_staleness": float(spearmanr(sd, dt).correlation),
                         "sd_peak_over_nonpeak": float(sd[peak].mean() / sd[~peak].mean()) if peak.any() and (~peak).any() else float("nan"),
                         "sd_median_phys": float(np.median(sd))},
        "strata": {},
    }
    for lbl, m in (("peak", peak), ("nonpeak", ~peak)):
        if m.sum() >= 200:
            res["strata"][lbl] = {"n": int(m.sum()),
                                  "crps_skill_vs_r0": skill(shot[m], c_arm[m], c_r0[m], rng)["skill_point"],
                                  "mse_skill_vs_backbone": float(1 - z[f"{t}_se_model"][m].sum() / z[f"{t}_se_control"][m].sum())}
    if pi.shape[1] == 2:
        sep = np.abs(mu[:, 0] - mu[:, 1]) / sg.max(axis=1)
        bim = (pi.min(axis=1) >= BIMODAL_MIN_PI) & (sep >= BIMODAL_SEP)
        fp, fn = float(bim[peak].mean()) if peak.any() else float("nan"), float(bim[~peak].mean()) if (~peak).any() else float("nan")
        # the enrichment ratio is undefined without a population of bimodal rows (0/0 is not >= 2);
        # it needs at least 50 such rows and a non-empty nonpeak denominator
        enr = (fp / fn) if (bim.sum() >= 50 and fn > 0) else float("nan")
        res["bimodal"] = {"fraction": float(bim.mean()), "fraction_peak": fp, "fraction_nonpeak": fn,
                          "enrichment_peak": enr, "n_rows": int(bim.sum())}
        res["_bim_mask"] = bim
    res["_rows"] = {"shot": shot, "crps": c_arm, "crps_r0": c_r0, "crps_r1": c_r1}
    return res


def main():
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    runner = json.loads((DATA / ".b17_runner_summary.json").read_text(encoding="utf-8"))
    out = {"pops": {}}
    for pop in POPS:
        node = {"targets": {}}
        for t in TARGETS:
            tn = {}
            per = {arm: {} for arm in ARMS}
            pooled_rows = {arm: [] for arm in ARMS}
            gate_rows, gate_bim_rows = [], []
            for s in SEEDS:
                zs = {arm: load(arm, pop, s) for arm in ARMS}
                for arm in ARMS:
                    r = score_target(zs[arm], t, rng)
                    pooled_rows[arm].append(r.pop("_rows"))
                    bim = r.pop("_bim_mask", None)
                    per[arm][str(s)] = r
                    if arm == "v2m2":
                        g = zs["v2g"]
                        if not np.array_equal(g[f"{t}_shot"], zs["v2m2"][f"{t}_shot"]) or not np.array_equal(g[f"{t}_row"], zs["v2m2"][f"{t}_row"]):
                            raise SystemExit(f"FATAL: {pop} s{s} {t}: v2g and v2m2 rows differ")
                        cg = pooled_rows["v2g"][-1]["crps"]
                        cm = pooled_rows["v2m2"][-1]["crps"]
                        gate_rows.append((g[f"{t}_shot"], cm, cg))
                        gate_bim_rows.append((g[f"{t}_shot"][bim], cm[bim], cg[bim]))
                        per["v2m2"][str(s)]["crps_skill_vs_v2g"] = skill(g[f"{t}_shot"], cm, cg, rng)
            for arm in ARMS:
                rows = pooled_rows[arm]
                shot = np.concatenate([r["shot"] for r in rows])
                ca = np.concatenate([r["crps"] for r in rows])
                tn[arm] = {"per_seed": per[arm],
                           "pooled": {"crps_skill_vs_r0": skill(shot, ca, np.concatenate([r["crps_r0"] for r in rows]), rng),
                                      "crps_skill_vs_r1": skill(shot, ca, np.concatenate([r["crps_r1"] for r in rows]), rng)}}
            shot = np.concatenate([g[0] for g in gate_rows])
            tn["v2m2"]["pooled"]["crps_skill_vs_v2g"] = skill(shot, np.concatenate([g[1] for g in gate_rows]),
                                                              np.concatenate([g[2] for g in gate_rows]), rng)
            nb = sum(len(g[0]) for g in gate_bim_rows)
            if nb >= 200:
                tn["v2m2"]["pooled"]["crps_skill_vs_v2g_bimodal_rows"] = skill(
                    np.concatenate([g[0] for g in gate_bim_rows]), np.concatenate([g[1] for g in gate_bim_rows]),
                    np.concatenate([g[2] for g in gate_bim_rows]), rng)
            # MSE guard from the runner summary (posterior mean, paired vs backbone)
            for arm in ARMS:
                rr = runner["arms"][arm][pop]
                worse = sum(rr[str(s)][t]["control_better"] for s in SEEDS)
                pts = [rr[str(s)][t]["paired_vs_control"] for s in SEEDS]
                tn[arm]["mse_guard"] = {"sig_worse": worse, "points": pts, "pooled_mean": float(np.mean(pts)),
                                        "pass": worse <= 1 and float(np.mean(pts)) >= -PRACTICAL_EPS,
                                        "rejected": worse >= 2}
            node["targets"][t] = tn
        # ---- hypotheses (sec. 4)
        ti, vt = node["targets"]["CES_TI"], node["targets"]["CES_VT"]
        h = {}
        a_r0 = sum(1 for s in SEEDS if ti["v2g"]["per_seed"][str(s)]["crps_skill_vs_r0"]["pass"])
        h["H17-a"] = {"pass": a_r0 >= 3 and ti["v2g"]["pooled"]["crps_skill_vs_r0"]["pass"] and ti["v2g"]["pooled"]["crps_skill_vs_r1"]["pass"],
                      "splits_pass_vs_r0": a_r0, "pooled_vs_r0": ti["v2g"]["pooled"]["crps_skill_vs_r0"],
                      "pooled_vs_r1": ti["v2g"]["pooled"]["crps_skill_vs_r1"]}
        rho_e = [ti["v2g"]["per_seed"][str(s)]["sd_relations"]["spearman_sd_energy"] for s in SEEDS]
        ratio = [ti["v2g"]["per_seed"][str(s)]["sd_relations"]["sd_peak_over_nonpeak"] for s in SEEDS]
        rho_dt = [vt["v2g"]["per_seed"][str(s)]["sd_relations"]["spearman_sd_staleness"] for s in SEEDS]
        rho_e_vt = [vt["v2g"]["per_seed"][str(s)]["sd_relations"]["spearman_sd_energy"] for s in SEEDS]
        h["H17-b"] = {"ti_pass": all(r >= 0.2 for r in rho_e) and all(r >= 1.3 for r in ratio),
                      "vt_pass": all(r >= 0.2 for r in rho_dt) and all(abs(e) < abs(d) for e, d in zip(rho_e_vt, rho_dt)),
                      "ti_spearman_sd_energy": rho_e, "ti_sd_peak_over_nonpeak": ratio,
                      "vt_spearman_sd_staleness": rho_dt, "vt_spearman_sd_energy": rho_e_vt}
        cov = [ti["v2g"]["per_seed"][str(s)]["coverage"] for s in SEEDS]
        c_ok = sum(1 for c in cov if 0.85 <= c["90"] <= 0.95 and 0.45 <= c["50"] <= 0.55)
        h["H17-c"] = {"pass": c_ok >= 3, "splits_ok": c_ok, "coverage": cov}
        bim = [ti["v2m2"]["per_seed"][str(s)]["bimodal"] for s in SEEDS]
        enr = [b["enrichment_peak"] for b in bim]
        h["H17-d"] = {"pass": ti["v2m2"]["pooled"]["crps_skill_vs_v2g"]["pass"] and sum(1 for e in enr if np.isfinite(e) and e >= 2.0) >= 3,
                      "pooled_v2m2_vs_v2g": ti["v2m2"]["pooled"]["crps_skill_vs_v2g"],
                      "bimodal_fraction": [b["fraction"] for b in bim], "enrichment_peak": enr}
        h["H17-e"] = {arm: {t: node["targets"][t][arm]["mse_guard"] for t in TARGETS} for arm in ARMS}
        node["hypotheses"] = h
        out["pops"][pop] = node
    cut, incl = out["pops"]["cut"]["hypotheses"], out["pops"]["incl"]["hypotheses"]
    claim = lambda hh: hh["H17-a"]["pass"] and hh["H17-c"]["pass"] and hh["H17-e"]["v2g"]["CES_TI"]["pass"]
    out["rules"] = {
        "1_calibrated_nowcaster_claim": {"cut": claim(cut), "incl": claim(incl),
                                         "status": "unconditional" if claim(cut) and claim(incl) else "cut-conditional" if claim(cut) else "not supported"},
        "2_reported_arm": {p: ("v2m2" if out["pops"][p]["hypotheses"]["H17-d"]["pass"] else "v2g") for p in POPS},
        "3_sigma_is_event_physics": {p: out["pops"][p]["hypotheses"]["H17-b"] for p in POPS},
        "4_rejected_arms": {p: [arm for arm in ARMS if any(out["pops"][p]["targets"][t][arm]["mse_guard"]["rejected"] for t in TARGETS)] for p in POPS},
    }
    (DATA / ".b17_summary.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")

    def f(r):
        return f"{r['skill_point']:+.3f} [{r['skill_ci95'][0]:+.3f},{r['skill_ci95'][1]:+.3f}]{'*' if r['pass'] else ' '}"
    for pop in POPS:
        print(f"\n===== {pop} =====")
        for t in TARGETS:
            for arm in ARMS:
                a = out["pops"][pop]["targets"][t][arm]
                per = " ".join(f"{a['per_seed'][str(s)]['crps_skill_vs_r0']['skill_point']:+.3f}{'*' if a['per_seed'][str(s)]['crps_skill_vs_r0']['pass'] else ' '}" for s in SEEDS)
                cov = " ".join(f"{a['per_seed'][str(s)]['coverage']['90']:.2f}" for s in SEEDS)
                print(f"{t} {arm:4s} CRPS vs R0 {per} | pooled R0 {f(a['pooled']['crps_skill_vs_r0'])} R1 {f(a['pooled']['crps_skill_vs_r1'])}"
                      f" | cov90 {cov} | MSE guard {a['mse_guard']['points']} worse {a['mse_guard']['sig_worse']}")
                if arm == "v2m2":
                    print(f"{t}      v2m2 vs v2g pooled {f(a['pooled']['crps_skill_vs_v2g'])}  bimodal frac "
                          + " ".join(f"{a['per_seed'][str(s)]['bimodal']['fraction']:.3f}" for s in SEEDS)
                          + "  enrich " + " ".join(f"{a['per_seed'][str(s)]['bimodal']['enrichment_peak']:.1f}" for s in SEEDS))
            g = out["pops"][pop]["targets"][t]["v2g"]["per_seed"]
            print(f"{t} v2g  rho(sd,energy) " + " ".join(f"{g[str(s)]['sd_relations']['spearman_sd_energy']:+.2f}" for s in SEEDS)
                  + "  rho(sd,staleness) " + " ".join(f"{g[str(s)]['sd_relations']['spearman_sd_staleness']:+.2f}" for s in SEEDS)
                  + "  sd peak/nonpeak " + " ".join(f"{g[str(s)]['sd_relations']['sd_peak_over_nonpeak']:.2f}" for s in SEEDS))
        print(json.dumps({k: v for k, v in out["pops"][pop]["hypotheses"].items() if k != "H17-e"}, indent=1, default=float)[:2500])
    print("\n== rules ==")
    print(json.dumps(out["rules"], indent=1, default=float))
    print(f"\nwrote {DATA / '.b17_summary.json'}")


if __name__ == "__main__":
    main()
