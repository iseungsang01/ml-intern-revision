"""Collect every paper-facing number from the frozen evaluation artifacts.

The paper claims its numbers are read from frozen artifacts; this script makes that
literally true and mechanically checkable. It reads only committed run directories and
the batch verdict files under ``data/`` (never retrains, never re-scores) and writes one
JSON that the LaTeX sources and the figure scripts are transcribed from.

Confirmed protocol (THESIS_RESULTS.md sec. 8v, PREREGISTRATION_W2.md): W = 2, held-free
training AND evaluation, per-file sample cap 500, and TWO co-primary populations --
``cut`` (CES_TI fit-failure spikes > 3 keV treated as missing everywhere,
``CES_TI_SPIKE_CUT_EV=3000``) and ``incl`` (no cut). An unqualified claim must hold in
both. Every block below carries the population it was measured in.

Sources (all under data/):
  .b5_summary.json           B.5 full re-score, both populations (sec. 8ab)
  .b1_gate_summary.json      B.1 backbone gate, 16-run grid + budget equalization (sec. 8x)
  .b2c_v3_summary.json       B.2 attention-readout candidate, TEST confirmation (sec. 8y)
  .b3c_b3k8_summary.json,
  .b3_probe_summary_b3k8.json  B.3 interpretable rung + probes (sec. 8z)
  .b4_scale_summary.json     B.4 width-scaling ceiling (sec. 8aa)
  .wsweep_hf_summary.json    held-free window sweep that selected W = 2 (sec. 8f)
  .protocol_audit_stats.json data ledger, T_i tail, V_rot precision (sec. 8w)
  .b5_spike_structure.json   fit-failure spike structure audit (sec. 8ab memo)
  .latency_benchmark.json    inference latency (window family; seq_v2 if measured)
  per-run comparison_metrics*.json  physical-unit RMSE ladders
  .b22_verdict.json          physics-filter dual readout, tab:dual (sec. 8bp)
  .b23_verdict.json, .b24_verdict.json, .b26_precheck.json
                             V_rot photon floor: fitted jump process / noise law, sqrt 2 ratio (sec. 8bq-8bt)
  .b17b_summary.json         predictive-distribution heads, sec:dist (sec. 8bj)
  .b14c_verdict.json, .b14s_cut_rescore.json
                             B.14 sec. 10 own-objective selection / epoch cap, sec:bulk (sec. 8bu)
  .b27_summary_R10.json, .b27c_hier.json
                             B.27/B.27c shot-level repeated 5-fold replicate, tab:replicate (sec. 8bw, 8by)
  .b28_stage1.json, .b28_stage2.json, .b28_stage3.json, .b28_stage4.json
                             B.28 what-the-backbone-computes: surrogate ratios and the trained equation (sec. 8bz)

The paper sections written after the B.5 pipeline (sec:physfilter, sec:dist) transcribe numbers by
hand; `transcription_check` re-reads the values printed in main.tex from these files and fails on
any mismatch, so the hand transcription is mechanically checked.

Usage (repo root):  py ces_prediction/collect_paper_numbers.py
Writes: docs/paper/paper_numbers.json
"""

import json
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA = REPO_ROOT / "data"
OUT_PATH = REPO_ROOT / "docs" / "paper" / "paper_numbers.json"

SEEDS = ("42", "1", "7", "123")
TARGETS = ("CES_TI", "CES_VT")
POPS = ("cut", "incl")
POP_LABEL = {"cut": "spike-cut (CES_TI > 3 keV treated as missing)", "incl": "spike-inclusive (no cut)"}
RUN = {  # run directories per population / arm (identical to run_b5.py's table)
    "cut": {"seq": ".b1_seqv2_s{s}_i{s}", "win": ".b1_w2cut_s{s}", "b3": ".b3c_b3k8_s{s}", "anchor": ".b3_anchor_s{s}"},
    "incl": {"seq": ".b5i_seqv2_s{s}", "win": ".b5i_w2_s{s}", "b3": ".b5i_b3k8_s{s}", "anchor": ".b5i_anchor_s{s}"},
}
LADDER_BASELINES = ("persistence", "ar_local", "gp_causal", "linear", "pchip", "gp")


def _load(path):
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"missing artifact: {p} -- run the batch that produces it; nothing is fabricated here")
    return json.loads(p.read_text(encoding="utf-8"))


def _report(run_dir):
    """TEST report of a run dir: the split-tagged file when present, else the legacy unsuffixed
    one (which is TEST by the one-report-per-split rule)."""
    p = run_dir / "comparison_metrics_test.json"
    if not p.exists():
        p = run_dir / "comparison_metrics.json"
    return _load(p)


def _mean(xs):
    return float(np.mean([float(x) for x in xs]))


# ------------------------------------------------------------------ blocks
def protocol(b5, b1):
    env = b1["protocol"]["env"]
    return {
        "window": 2, "held": "excluded from training and evaluation (genuine measurements only)",
        "per_file_sample_cap": int(env["CES_MAX_SAMPLES_PER_FILE"]),
        "populations": {p: POP_LABEL[p] for p in POPS},
        "cut_ev": {p: float(b5["protocol"]["populations"][p]["CES_TI_SPIKE_CUT_EV"]) for p in POPS},
        "seeds": [int(s) for s in SEEDS],
        "significance": "shot-clustered paired bootstrap, 10,000 resamples, 95% CI excludes 0 (PR4)",
        "backbone": "seq_v2 (full-grid causal LSTM, V_rot branch blocked from the fast diagnostics)",
        "window_control": "iter009 GRU + observation-masked attention at W = 2 (held-free; cut / no-cut per population)",
    }


def data_ledger(audit, spikes):
    n = audit["n_rows"]
    L = audit["ledger_crosscheck"]
    out = {"n_files": audit["n_files"], "n_rows": n, "targets": {}}
    for t in TARGETS:
        nan, held = L[t]["nan"], L[t]["held"]
        obs = n - nan
        out["targets"][t] = {"missing_frac": nan / n, "held_frac_of_rows": held / n,
                             "held_frac_of_observed": held / obs if obs else None,
                             "independent_frac_of_rows": (obs - held) / n}
    ti = audit["ces_ti_observed"]
    out["ces_ti_tail"] = {"n_observed": ti["n"], "p99_eV": ti["p99_eV"], "p999_eV": ti["p999_eV"], "max_eV": ti["max_eV"],
                          "n_gt_2500": ti["n_gt_2500"], "n_gt_3000": ti["n_gt_3000"], "n_gt_4000": ti["n_gt_4000"],
                          "pct_gt_3000": ti["pct_gt_3000"]}
    out["ces_vt_precision"] = audit["ces_vt_precision"]
    out["dataset_spec"] = audit["dataset_spec"]
    out["time_deltas"] = audit["time_deltas"]
    out["spike_structure"] = {"CES_TI": spikes["CES_TI"], "CES_VT": spikes["CES_VT"], "definitions": spikes["definitions"]}
    return out


def headline(b5):
    H = b5["headline"]
    out = {}
    for pop in POPS:
        rows = {}
        for arm in ("seq", "win"):
            per = {}
            for s in SEEDS:
                per[s] = {t: {b: H[pop][arm][s][t][b] for b in H[pop][arm][s][t]} for t in TARGETS}
            rows[arm] = {"per_seed": per,
                         "pr4_pass": H[pop][f"{arm}_pr4_pass"],
                         "mean_skill_vs_pchip": {t: _mean([per[s][t]["pchip"]["skill"] for s in SEEDS]) for t in TARGETS},
                         "mean_skill_vs_persistence": {t: _mean([per[s][t]["persistence"]["skill"] for s in SEEDS]) for t in TARGETS}}
        rows["seq_vs_win_paired"] = H[pop]["seq_vs_win_paired"]
        rows["seq_vs_win_paired_mean_TI"] = _mean([H[pop]["seq_vs_win_paired"][s]["CES_TI"]["skill"] for s in SEEDS])
        rows["seq_vs_win_positive_TI"] = sum(int(H[pop]["seq_vs_win_paired"][s]["CES_TI"]["skill"] > 0) for s in SEEDS)
        rows["seq_vs_win_sig_TI"] = sum(int(H[pop]["seq_vs_win_paired"][s]["CES_TI"]["a_better"]) for s in SEEDS)
        # physical-unit RMSE ladder from the seq run's TEST report (all arms scored on one population)
        ladder = {}
        for s in SEEDS:
            rep = _report(DATA / RUN[pop]["seq"].format(s=s))
            wrep = _report(DATA / RUN[pop]["win"].format(s=s))
            ladder[s] = {}
            for t in TARGETS:
                pt = rep["per_target"][t]
                ladder[s][t] = {"n": pt["n"], "n_shots": None,
                                "rmse_model_seq_v2": pt["rmse_model"],
                                "rmse_model_window": wrep["per_target"][t]["rmse_model"],
                                "rmse_baselines": {b: pt["baselines"][b]["rmse"] for b in LADDER_BASELINES if b in pt["baselines"]},
                                "future_neighbor_fraction": pt["future_neighbor_fraction"]}
        rows["rmse_ladder"] = ladder
        out[pop] = rows
    return out


def gate_b1(b1):
    g = b1["seq_grid"]
    return {"n_runs": g["pooled"]["n_runs"], "pooled_mean_TI": g["pooled"]["mean"], "pooled_ci95": g["pooled"]["run_cluster_ci95"],
            "per_split_init_mean_TI": {s: g["per_split"][s]["init_mean"] for s in SEEDS},
            "per_split_paired_TI": {s: g["per_split"][s]["ti_paired_skills"] for s in SEEDS},
            "n_positive": int(sum(int(x > 0) for s in SEEDS for x in g["per_split"][s]["ti_paired_skills"])),
            "vt_significant_deficit_runs": g["vt_significant_deficit_runs"],
            "equalized_10epoch_paired_TI": b1["equalized"]["ti_paired_skills"],
            "gate": b1["gate"],
            "window_family_cut": {s: {t: b1["window_family"][s][t]["gates"] for t in TARGETS} for s in SEEDS}}


def b2_v3(b2c):
    rows = {str(r["seed"]): r for r in b2c["rows"]}
    return {"variant": b2c["variant"], "per_seed": rows, "verdict": b2c.get("verdict"),
            "n_positive_TI": sum(int(rows[s]["paired_TI"] > 0) for s in SEEDS),
            "n_sig_TI": sum(int(rows[s]["paired_TI_sig"]) for s in SEEDS),
            "n_gp_causal_pass_TI": sum(int(rows[s]["vs_gp_causal_TI_pass"]) for s in SEEDS)}


def ladder(b5, b3c, probe):
    L = b5["ladder"]
    out = {}
    for pop in POPS:
        per = L[pop]["per_seed"]
        rungs = {}
        for arm in ("persistence", "anchor", "b3", "win", "seq"):
            rungs[arm] = {t: {"per_seed": {s: per[s][arm][t] for s in SEEDS}, "mean": _mean([per[s][arm][t] for s in SEEDS])} for t in TARGETS}
        out[pop] = {"skill_vs_pchip": rungs,
                    "b3_vs_anchor": {s: per[s]["b3_vs_anchor"] for s in SEEDS},
                    "b3_vs_seq": {s: per[s]["b3_vs_seq"] for s in SEEDS},
                    "verdict": L[pop]["verdict"]}
    b3rows = {str(r["seed"]): r for r in b3c["rows"]}
    out["b3k8"] = {"params": 21498, "latent_dims": {"CES_TI": 8, "CES_VT": 4},
                   "per_seed": {s: {k: b3rows[s][k] for k in ("vs_w2cut_TI", "vs_gp_causal_TI", "vs_gp_causal_TI_pass", "vs_backbone_VT")} for s in SEEDS},
                   "verdict_cut": b3c["verdict"]}
    ps = {str(r["seed"]): r for r in probe["per_seed"]}
    out["probes"] = {"routing_structural_check_all_pass": probe["routing_structural_check_all_pass"],
                     "ti_full_latent_r2": {q: {s: ps[s]["probes_z_ti"][q]["full_latent_r2"] for s in SEEDS} for q in ps["42"]["probes_z_ti"]},
                     "vt_full_latent_r2": {q: {s: ps[s]["probes_z_vt"][q]["full_latent_r2"] for s in SEEDS} for q in ps["42"]["probes_z_vt"]},
                     "ti_correction_share_of_pred_var": {s: ps[s]["decomposition_test"]["ti"]["correction_share_of_pred_var"] for s in SEEDS},
                     "ti_correction_std": {s: ps[s]["decomposition_test"]["ti"]["correction_std"] for s in SEEDS},
                     "ti_anchor_std": {s: ps[s]["decomposition_test"]["ti"]["anchor_std"] for s in SEEDS}}
    return out


def scaling_b4(b4):
    out = {"controlled_variable": b4["controlled_variable"], "widths": {}}
    for w, blk in b4["curve"].items():
        pts = blk["points"]
        out["widths"][w] = {"params": pts[0]["params"],
                            "skill_TI_per_seed": {str(p["seed"]): p["skill_TI"] for p in pts},
                            "skill_TI_mean": _mean([p["skill_TI"] for p in pts]),
                            "skill_VT_mean": _mean([p["skill_VT"] for p in pts]),
                            "paired_TI_vs_160": {str(p["seed"]): p.get("paired_TI_vs_160") for p in pts},
                            "paired_TI_vs_160_mean": _mean([p["paired_TI_vs_160"] for p in pts]) if "paired_TI_vs_160" in pts[0] else None,
                            "sig_win_vs_160": sum(int(p.get("paired_TI_sig_win", False)) for p in pts),
                            "sig_loss_vs_160": sum(int(p.get("paired_TI_sig_loss", False)) for p in pts),
                            "pchip_pass_TI": sum(int(p["pchip_pass_TI"]) for p in pts),
                            "gp_causal_pass_TI": sum(int(p["gp_causal_pass_TI"]) for p in pts),
                            "epochs": {str(p["seed"]): [p["best_epoch"], p["epochs_run"]] for p in pts}}
    out["verdict"] = b4["verdict"]
    return out


def window_sweep(ws):
    pts = {}
    for p in ws["per_point"]:
        key = "hist0" if p.get("hist0") else str(p["window"])
        pts[key] = {"window": p["window"], "history_obs": p["history_len"],
                    "skill_TI_mean": p["skill_TI_mean"], "skill_TI_per_seed": p["skill_TI_per_seed"], "pchip_pass_TI": p["pchip_pass_TI"],
                    "skill_VT_mean": p["skill_VT_mean"], "skill_VT_per_seed": p["skill_VT_per_seed"], "pchip_pass_VT": p["pchip_pass_VT"]}
    return {"protocol": ws["protocol"], "points": pts,
            "note": "held-free, cap 500, no spike cut (the frozen W = 2 point of this sweep is the inclusive-population window control)"}


def latency(lat):
    runs = lat["runs"]
    return {"env": lat["_env"], "scope": lat["_scope"],
            "runs": [{k: r[k] for k in ("model", "window", "device", "batch", "params", "median_ms", "p95_ms", "p99_ms", "amortized_ms", "throughput_samples_per_s")
                      if k in r} for r in runs]}


def _rng(xs, nd=3):
    return [round(float(min(xs)), nd), round(float(max(xs)), nd)]


def physfilter(b22, b23, b24, b26):
    out = {"dual_readout": {}, "vrot_floor": {}}
    for pop in POPS:
        out["dual_readout"][pop] = {}
        for t in TARGETS:
            c = b22[pop][t]
            out["dual_readout"][pop][t] = {
                "median_over_persistence_range": _rng(c["median_over_persistence_split"]),
                "m1_vs_backbone_range": _rng([e["point"] for e in c["m1_vs_backbone"]["per_seed"]], 2),
                "m1_vs_backbone_sig": sum(e["ci"][0] > 0 for e in c["m1_vs_backbone"]["per_seed"]),
                "m1_vs_persistence_range": _rng([e["point"] for e in c["m1_vs_persistence"]], 2),
                "m1_vs_persistence_sig": sum(e["ci"][0] > 0 for e in c["m1_vs_persistence"]),
                "mse_vs_backbone_range": _rng(c["mse_vs_backbone"], 2)}
        out["dual_readout"][pop]["mean_estimator"] = b22[pop]["mean_estimator"]["arm"]
    out["dual_readout"]["rules"] = b22["rules"]
    for pop in POPS:
        out["vrot_floor"][pop] = {
            "b23_median_over_persistence_pooled": b23[pop]["median_over_persistence_pooled"],
            "b24_median_over_persistence_pooled": b24[pop]["median_over_persistence_pooled"],
            "b24_median_over_persistence_range": _rng(b24[pop]["median_over_persistence_split"]),
            "b24_beats_persistence": sum(e["ci"][0] > 0 for e in b24[pop]["m1_vs_persistence"]),
            "b24_fit": {s: {k: b24[pop]["fit"][s][k] for k in ("sigma_n_kms", "nu", "lam_per_step", "at_grid_edge")}
                        for s in SEEDS}}
    out["vrot_floor"]["rules_b24"] = b24["rules"]
    out["vrot_floor"]["photon_ratio_A1"] = {k: b26[k]["A1"]["R_median"] for k in b26 if k != "verdict"}
    out["vrot_floor"]["b26_verdict"] = b26["verdict"]
    return out


def dist_head(b17b):
    out = {"rules": b17b["rules"]}
    for pop in POPS:
        arm = b17b["rules"][pop]["named_arm"]
        out[pop] = {"named_arm": arm,
                    "t_vs_mixture_TI": b17b["pops"][pop]["hypotheses"]["Hb-1_t_vs_mixture"]["pooled"]["skill_point"],
                    "nu_median_TI": b17b["pops"][pop]["hypotheses"]["Hb-2_nu"]["nu_median_per_split"]}
        for t in TARGETS:
            a = b17b["pops"][pop]["targets"][t][arm]
            out[pop][t] = {"crps_vs_r0_pooled": a["pooled"]["crps_skill_vs_r0"]["skill_point"],
                           "crps_vs_r0_splits_sig": sum(a["per_seed"][s]["crps_skill_vs_r0"]["skill_ci95"][0] > 0 for s in SEEDS),
                           "coverage_50_range": _rng([a["per_seed"][s]["coverage"]["50"] for s in SEEDS], 2),
                           "coverage_90_range": _rng([a["per_seed"][s]["coverage"]["90"] for s in SEEDS], 2)}
    return out


def b14_select(v, rescore_cut):
    a, b = v["A_cut_select_loss"]["huber"]["CES_TI"], v["B_incl_cap100"]["huber"]["CES_TI"]
    gp = rescore_cut["results"]["cut"]["pooled"]["CES_TI"]["huber_s_vs_gp_causal"]["m1_medae"]
    return {"rules": v["rules"],
            "cut_huber_TI": {"m1_pooled": a["m1_pooled"]["skill_point"], "m1_ci": a["m1_pooled"]["skill_ci95"],
                             "m1_splits_pos": a["m1_splits_pos"], "mse_pooled": a["mse_pooled"]["skill_point"],
                             "mse_sig_deficits": a["mse_sig_deficits"],
                             "median_over_persistence": a["median_over_persistence_pooled"],
                             "m1_vs_gp_causal": gp["skill_point"], "m1_vs_gp_causal_ci": gp["skill_ci95"]},
            "incl_huber_TI": {"m1_pooled": b["m1_pooled"]["skill_point"], "mse_sig_deficits": b["mse_sig_deficits"],
                              "best_epochs": [b["per_split"][s]["epochs"][1] for s in SEEDS]}}


def replicate_b27(r10, hier):
    """B.27/B.27c (sec. 8bw/8by): fold-level Nadeau-Bengio interval over 50 folds and the registered
    two-level (repeat x shot) bootstrap over the eight post-registration repeats (r2-r9)."""
    out = {"design": {k: r10["design"][k] for k in ("K", "repeat_seeds", "test_over_train_files")},
           "primary_rule": "two-level bootstrap on r2-r9: beats iff 2.5th percentile > 0 (PREREGISTRATION_B27.md sec. 8)"}
    for pop in POPS:
        out[pop] = {}
        for t in TARGETS:
            out[pop][t] = {}
            for ref in ("pchip", "persistence", "gp_causal", "gp"):
                a, b = r10["cells"][f"{pop}/{t}/{ref}"], hier[f"{pop}/{t}/{ref}"]
                out[pop][t][ref] = {
                    "fold_mean": a["mean"], "fold_sd": a["sd"], "nb_ci95": a["ci95_corrected"],
                    "nb_verdict": a["verdict"], "fold_pass": a["per_fold_pass"], "n_folds": a["n"],
                    "two_level_ci95": b["primary_r2_r9"]["ci95"], "two_level_verdict": b["primary_r2_r9"]["verdict"],
                    "per_repeat_pooled": b["per_repeat_pooled"]}
    return out


# values printed in tab:replicate (3 decimals): (pop, target, ref) -> (nb lo, nb hi, pass, two-level lo, two-level hi)
TAB_REPLICATE = {
    ("cut", "CES_TI", "pchip"): (0.186, 0.253, 50, 0.188, 0.248),
    ("cut", "CES_TI", "persistence"): (0.368, 0.410, 50, 0.369, 0.408),
    ("cut", "CES_TI", "gp_causal"): (0.084, 0.138, 47, 0.085, 0.132),
    ("cut", "CES_TI", "gp"): (-0.044, 0.020, 1, -0.043, 0.014),
    ("cut", "CES_VT", "pchip"): (-0.021, 0.405, 15, 0.072, 0.701),
    ("cut", "CES_VT", "persistence"): (0.190, 0.601, 42, 0.277, 0.763),
    ("cut", "CES_VT", "gp_causal"): (-0.060, 0.512, 11, 0.045, 0.701),
    ("cut", "CES_VT", "gp"): (-0.234, 0.312, 0, -0.119, 0.635),
    ("incl", "CES_TI", "pchip"): (0.188, 0.307, 46, 0.192, 0.294),
    ("incl", "CES_TI", "persistence"): (0.361, 0.445, 50, 0.364, 0.438),
    ("incl", "CES_TI", "gp_causal"): (0.067, 0.170, 40, 0.076, 0.157),
    ("incl", "CES_TI", "gp"): (-0.026, 0.094, 12, -0.017, 0.075),
    ("incl", "CES_VT", "pchip"): (-0.053, 0.391, 11, 0.046, 0.683),
    ("incl", "CES_VT", "persistence"): (0.159, 0.587, 38, 0.139, 0.747),
    ("incl", "CES_VT", "gp_causal"): (-0.096, 0.502, 10, 0.023, 0.683),
    ("incl", "CES_VT", "gp"): (-0.268, 0.300, 0, -0.139, 0.609),
}


def equation_b28(s1, s2, s3, s4):
    """B.28 (sec. 8bz): stage-1 surrogate ratios on the frozen rows, the stage-2 trained equation,
    and the stage-3 measurements (sec. 8bz-3: Jacobian form, truncation decay, MC norm trajectory)."""
    out = {"stage1": {}, "stage2": {}, "stage3": {}, "stage4": {}}
    for rec in s1["splits"]:
        out["stage1"].setdefault(rec["pop"], {})[str(rec["seed"])] = {
            "alpha_star": rec["alpha_star"],
            "ratio_ewma_vs_persistence": rec["ratios"]["ewma_star"]["persistence"]["ratio_point"],
            "ratio_ewma_inc_vs_persistence": rec["ratios"]["ewma_inc"]["persistence"]["ratio_point"],
            "ewma_vs_pchip": rec["skills"]["ewma_star"]["pchip"]["skill_point"],
            "ewma_inc_vs_pchip": rec["skills"]["ewma_inc"]["pchip"]["skill_point"]}
    for pop in POPS:
        out["stage2"][pop] = {}
        for s in SEEDS:
            r = s2["arms"]["b28"][pop][s]
            out["stage2"][pop][s] = {"ratio_vs_pchip": r["CES_TI"]["ratio_vs_pchip"],
                                     "ratio_vs_persistence": r["CES_TI"]["ratio_vs_persistence"],
                                     "paired_vs_control_TI": r["CES_TI"]["paired_vs_control"],
                                     "paired_vs_control_VT": r["CES_VT"]["paired_vs_control"],
                                     "physics": {k: r["physics"][k] for k in ("K0", "K_inf", "s_K_ms", "a_sum_BES_eV", "a_sum_ECEI_eV", "a_sum_MC_eV")}}
        out["stage2"][pop]["verdict"] = s2["verdict"][pop]
    jac = {(r["pop"], r["seed"]): r for r in s3["jac"]["splits"]}
    res2 = {(r["pop"], r["seed"]): r for r in s3["resid2"]["splits"]}
    trunc = {(r["pop"], r["seed"]): r for r in s3["trunc"]["splits"] if "fit" in r}
    for pop in POPS:
        out["stage3"][pop] = {}
        for s in SEEDS:
            j, r2 = jac[(pop, int(s))], res2[(pop, int(s))]
            bins_bes = j["saturation"]["BES"]["bins"]
            cell = {"rho_amp": j["amplitude"]["fit_lag0_3"]["rho"],
                    "gap_closed_registered_all": j["residual"]["families"]["all"]["gap_closed"],
                    "gap_closed_fir_plus_state": r2["families"]["fir+ewma_state"]["gap_closed"],
                    "gap_closed_hp_mean10": r2["families"]["hp_mean10"]["gap_closed"],
                    "bes_gain_rel_max_le2sigma": max(b["grad_rel_to_quiet_bin"] for b in bins_bes[1:5]),
                    "bes_gain_rel_3_5sigma": bins_bes[6]["grad_rel_to_quiet_bin"],
                    "impulse_eV_per_sigma": {g: j["kernels"][g]["impulse_eV_per_sigma_sum"] for g in ("BES", "ECEI", "MC")}}
            if (pop, int(s)) in trunc:
                cell["rho_mse_k_le10"] = trunc[(pop, int(s))]["fit"]["mse_excess_k_le10"]["rho"]
            out["stage3"][pop][s] = cell
    out["stage3"]["mcnorm"] = {"mc_over_bes_growth_slope": s3["mcnorm"]["mc_over_bes_growth_slope"],
                               "mc_r2_linear": s3["mcnorm"]["random_walk_check"]["MC"]["r2_linear_in_steps"]}
    for pop in POPS:
        out["stage4"][pop] = {}
        for s in SEEDS:
            r = s4["pops"][pop].get(s)
            if r is None:
                continue
            out["stage4"][pop][s] = {"ratio_vs_pchip": r["CES_TI"]["ratio_vs_pchip"],
                                     "ratio_vs_persistence": r["CES_TI"]["ratio_vs_persistence"],
                                     "b28h_vs_b28_TI": r["b28h_vs_b28_TI"],
                                     "paired_vs_control_VT": r["CES_VT"]["paired_vs_control"],
                                     "physics": {k: r["physics"][k] for k in ("K0", "K_inf", "s_K_ms", "theta_sigma", "a_sum_BES_eV", "a_sum_ECEI_eV", "a_sum_MC_eV")}}
        out["stage4"][pop]["verdict"] = s4["verdict"].get(pop)
    return out


def transcription_check(n):
    """Values printed in main.tex (sec:physfilter tab:dual, sec:dist) against the artifacts."""
    d, v, h = n["physfilter"]["dual_readout"], n["physfilter"]["vrot_floor"], n["dist_head"]
    printed = [
        ("tab:dual cut T_i median range", d["cut"]["CES_TI"]["median_over_persistence_range"], [0.911, 0.920], 0.0015),
        ("tab:dual incl T_i median range", d["incl"]["CES_TI"]["median_over_persistence_range"], [0.906, 0.915], 0.0015),
        ("tab:dual cut V_rot median range", d["cut"]["CES_VT"]["median_over_persistence_range"], [0.939, 1.000], 0.0015),
        ("tab:dual incl V_rot median range", d["incl"]["CES_VT"]["median_over_persistence_range"], [0.938, 1.003], 0.0015),
        ("tab:dual cut V_rot vs persistence sig", [d["cut"]["CES_VT"]["m1_vs_persistence_sig"]], [1], 0),
        ("tab:dual cut T_i MSE vs backbone", d["cut"]["CES_TI"]["mse_vs_backbone_range"], [-0.41, -0.29], 0.015),
        ("physfilter B.24 V_rot median range (0.97-0.98 pooled)", [v["cut"]["b24_median_over_persistence_pooled"],
                                                                v["cut"]["b23_median_over_persistence_pooled"]], [0.97, 0.98], 0.0051),
        ("physfilter B.24 beats persistence (<= 2/4)", [v["cut"]["b24_beats_persistence"], v["incl"]["b24_beats_persistence"]], [2, 2], 0),
        ("physfilter B.24 nu", [v["cut"]["b24_fit"][s]["nu"] for s in SEEDS], [2.2] * 4, 1e-9),
        ("sec:dist cut T_i CRPS (+0.229)", [h["cut"]["CES_TI"]["crps_vs_r0_pooled"]], [0.229], 0.0005),
        ("sec:dist cut V_rot CRPS (+0.363)", [h["cut"]["CES_VT"]["crps_vs_r0_pooled"]], [0.363], 0.0005),
        ("sec:dist incl T_i CRPS (+0.444)", [h["incl"]["CES_TI"]["crps_vs_r0_pooled"]], [0.444], 0.0005),
        ("sec:dist incl V_rot CRPS (+0.351)", [h["incl"]["CES_VT"]["crps_vs_r0_pooled"]], [0.351], 0.0005),
        ("sec:dist cut t vs mixture (+0.009)", [h["cut"]["t_vs_mixture_TI"]], [0.009], 0.0005),
        ("sec:dist cut coverage 50", h["cut"]["CES_TI"]["coverage_50_range"], [0.49, 0.51], 0.0101),
        ("sec:dist cut coverage 90", h["cut"]["CES_TI"]["coverage_90_range"], [0.91, 0.92], 0.0051),
        ("sec:dist incl coverage 50", h["incl"]["CES_TI"]["coverage_50_range"], [0.53, 0.55], 0.0051),
        ("sec:dist incl coverage 90", h["incl"]["CES_TI"]["coverage_90_range"], [0.90, 0.91], 0.0051),
        ("sec:bulk B.14 sec.10 cut huber T_i M1 (+0.128 [+0.114, +0.141])",
         [n["b14_select"]["cut_huber_TI"]["m1_pooled"]] + n["b14_select"]["cut_huber_TI"]["m1_ci"], [0.128, 0.114, 0.141], 0.0005),
        ("sec:bulk B.14 sec.10 cut huber T_i MSE (+0.007, 0/4)", [n["b14_select"]["cut_huber_TI"]["mse_pooled"],
         n["b14_select"]["cut_huber_TI"]["mse_sig_deficits"], n["b14_select"]["cut_huber_TI"]["m1_splits_pos"]], [0.007, 0, 4], 0.0005),
        ("sec:bulk B.14 sec.10 cut median / persistence (0.894)", [n["b14_select"]["cut_huber_TI"]["median_over_persistence"]], [0.894], 0.0005),
        ("sec:bulk B.14 sec.10 cut median vs causal GP (+0.073, CI > 0)", [n["b14_select"]["cut_huber_TI"]["m1_vs_gp_causal"],
         float(n["b14_select"]["cut_huber_TI"]["m1_vs_gp_causal_ci"][0] > 0)], [0.073, 1.0], 0.0005),
        ("sec:bulk B.14 sec.10 inclusive best epochs 24-59", [min(n["b14_select"]["incl_huber_TI"]["best_epochs"]),
         max(n["b14_select"]["incl_huber_TI"]["best_epochs"])], [24, 59], 0),
        ("sec:bulk B.14 sec.10 inclusive M1 (+0.285) and deficit 1/4", [n["b14_select"]["incl_huber_TI"]["m1_pooled"],
         n["b14_select"]["incl_huber_TI"]["mse_sig_deficits"]], [0.285, 1], 0.0005),
        ("sec:dist CRPS significant splits", [h[p][t]["crps_vs_r0_splits_sig"] for p in POPS for t in TARGETS], [4] * 4, 0),
    ]
    rep = n["replicate_b27"]
    for (pop, t, ref), (lo, hi, npass, lo2, hi2) in TAB_REPLICATE.items():
        c = rep[pop][t][ref]
        printed.append((f"tab:replicate {pop} {t} vs {ref} fold-level", c["nb_ci95"] + [c["fold_pass"]], [lo, hi, npass], 0.0005))
        printed.append((f"tab:replicate {pop} {t} vs {ref} two-level", c["two_level_ci95"], [lo2, hi2], 0.0005))
    printed.append(("sec:headline V_rot vs PCHIP r2-r9 pooled range (cut +0.28..+0.59, incl +0.09..+0.56)",
                    [f(rep[p]["CES_VT"]["pchip"]["per_repeat_pooled"][str(r)] for r in range(2, 10))
                     for p in POPS for f in (min, max)], [0.28, 0.59, 0.09, 0.56], 0.005))
    printed.append(("sec:headline V_rot vs PCHIP fold sd 0.18 (cut)", [rep["cut"]["CES_VT"]["pchip"]["fold_sd"]], [0.18], 0.005))
    e1, e2 = n["equation_b28"]["stage1"]["cut"], n["equation_b28"]["stage2"]["cut"]
    printed.append(("sec:model B.28 stage-1 EWMA-only ratio range 0.73-0.82 (cut)",
                    [min(v["ratio_ewma_vs_persistence"] for v in e1.values()), max(v["ratio_ewma_vs_persistence"] for v in e1.values())], [0.73, 0.82], 0.005))
    printed.append(("sec:model B.28 stage-1 EWMA+inc ratio range 0.77-0.85 (cut), all >= 0.75",
                    [min(v["ratio_ewma_inc_vs_persistence"] for v in e1.values()), max(v["ratio_ewma_inc_vs_persistence"] for v in e1.values()),
                     float(all(v["ratio_ewma_inc_vs_persistence"] >= 0.75 for v in e1.values()))], [0.77, 0.85, 1.0], 0.005))
    printed.append(("sec:model B.28 stage-2 ratio vs PCHIP 0.50-0.66 (cut)",
                    [min(e2[s]["ratio_vs_pchip"] for s in SEEDS), max(e2[s]["ratio_vs_pchip"] for s in SEEDS)], [0.50, 0.66], 0.005))
    printed.append(("sec:model B.28 stage-2 ratio vs persistence 0.81-0.83 (cut)",
                    [min(e2[s]["ratio_vs_persistence"] for s in SEEDS), max(e2[s]["ratio_vs_persistence"] for s in SEEDS)], [0.81, 0.83], 0.005))
    ph = [e2[s]["physics"] for s in SEEDS]
    printed.append(("sec:model B.28 K0 0.30-0.31, K_inf 0.52-0.54, s_K 37-40 ms (cut)",
                    [min(p["K0"] for p in ph), max(p["K0"] for p in ph), min(p["K_inf"] for p in ph), max(p["K_inf"] for p in ph),
                     min(p["s_K_ms"] for p in ph), max(p["s_K_ms"] for p in ph)], [0.30, 0.31, 0.52, 0.54, 37, 40], 0.5))
    printed.append(("sec:model B.28 sum a_BES +43..+51, a_ECEI +39..+46 eV/sigma (cut)",
                    [min(p["a_sum_BES_eV"] for p in ph), max(p["a_sum_BES_eV"] for p in ph), min(p["a_sum_ECEI_eV"] for p in ph), max(p["a_sum_ECEI_eV"] for p in ph)],
                    [43, 51, 39, 46], 0.5))
    e3 = n["equation_b28"]["stage3"]["cut"]
    printed.append(("sec:model B.28 stage-3 FIR+state closes 32-45 % of the gap (cut)",
                    [min(e3[s]["gap_closed_fir_plus_state"] for s in SEEDS), max(e3[s]["gap_closed_fir_plus_state"] for s in SEEDS)], [0.32, 0.45], 0.005))
    printed.append(("sec:model B.28 stage-3 BES gain flat to 2 sigma (max rel 1.23) and cut beyond 3 sigma (0.38-0.68)",
                    [max(e3[s]["bes_gain_rel_max_le2sigma"] for s in SEEDS), min(e3[s]["bes_gain_rel_3_5sigma"] for s in SEEDS), max(e3[s]["bes_gain_rel_3_5sigma"] for s in SEEDS)],
                    [1.23, 0.38, 0.68], 0.005))
    e4 = n["equation_b28"]["stage4"]["cut"]
    if all(s in e4 for s in SEEDS):
        printed.append(("sec:model B.28 stage-4 b28h ratio vs PCHIP 0.72-0.78 (cut, 4/4 >= 0.70)",
                        [min(e4[s]["ratio_vs_pchip"] for s in SEEDS), max(e4[s]["ratio_vs_pchip"] for s in SEEDS),
                         float(all(e4[s]["ratio_vs_pchip"] >= 0.70 for s in SEEDS))], [0.72, 0.78, 1.0], 0.005))
        printed.append(("sec:model B.28 stage-4 b28h ratio vs persistence 0.87-0.90 (cut)",
                        [min(e4[s]["ratio_vs_persistence"] for s in SEEDS), max(e4[s]["ratio_vs_persistence"] for s in SEEDS)], [0.87, 0.90, ], 0.005))
        printed.append(("sec:model B.28 stage-4 b28h - b28 paired +0.034..+0.042, CIs > 0 (cut)",
                        [min(e4[s]["b28h_vs_b28_TI"]["skill_point"] for s in SEEDS), max(e4[s]["b28h_vs_b28_TI"]["skill_point"] for s in SEEDS),
                         float(all(e4[s]["b28h_vs_b28_TI"]["ci95"][0] > 0 for s in SEEDS))], [0.034, 0.042, 1.0], 0.0005))
        p4 = [e4[s]["physics"] for s in SEEDS]
        printed.append(("sec:model B.28 stage-4 theta 2.9-3.0 sigma (cut, printed to one decimal)",
                        [min(p["theta_sigma"] for p in p4), max(p["theta_sigma"] for p in p4)], [2.9, 3.0], 0.05))
        printed.append(("sec:model B.28 stage-4 K0 0.27-0.30 (cut)",
                        [min(p["K0"] for p in p4), max(p["K0"] for p in p4)], [0.27, 0.30], 0.005))
        printed.append(("sec:model B.28 stage-4 sum a_BES +48..+53, a_ECEI +44..+50 eV/sigma (cut)",
                        [min(p["a_sum_BES_eV"] for p in p4), max(p["a_sum_BES_eV"] for p in p4), min(p["a_sum_ECEI_eV"] for p in p4), max(p["a_sum_ECEI_eV"] for p in p4)],
                        [48, 53, 44, 50], 0.5))
    printed.append(("sec:model B.28 cross term < 0.001 (cut, 4/4)",
                    [float(all(abs(x) < 0.001 for x in n["equation_b28"]["stage2"]["cut"]["verdict"]["H28_4_cross_vs_b28_TI"]))], [1.0], 0))
    problems = []
    for name, got, want, tol in printed:
        if len(got) != len(want) or any(abs(float(a) - b) > tol for a, b in zip(got, want)):
            problems.append(f"{name}: artifact {got} vs paper {want}")
    return problems


def crosscheck(numbers, b5, b3c, b4):
    problems = []
    for pop in POPS:
        for s in SEEDS:
            a = b5["ladder"][pop]["per_seed"][s]["seq"]["CES_TI"]
            b = b5["headline"][pop]["seq"][s]["CES_TI"]["pchip"]["skill"]
            if abs(a - b) > 1e-6:
                problems.append(f"ladder/headline seq skill differ {pop} s{s}: {a:.6f} vs {b:.6f}")
    for r in b3c["rows"]:
        s = str(r["seed"])
        a = r["vs_backbone_TI"]; b = b5["ladder"]["cut"]["per_seed"][s]["b3_vs_seq"]["CES_TI"]["skill"]
        if abs(a - b) > 1e-6:
            problems.append(f"b3 vs backbone differs from B.5 ladder s{s}: {a:.6f} vs {b:.6f}")
    for p in b4["curve"]["160"]["points"]:
        s = str(p["seed"])
        b = b5["headline"]["cut"]["seq"][s]["CES_TI"]["pchip"]["skill"]
        if abs(p["skill_TI"] - b) > 1e-4:  # re-scored in a later session: CUDA float32 drift ~1e-5 (bounded-drift rule)
            problems.append(f"B.4 width-160 point differs from headline s{s}: {p['skill_TI']:.6f} vs {b:.6f}")
    for pop in POPS:
        for s in SEEDS:
            n_lad = numbers["headline"][pop]["rmse_ladder"][s]["CES_TI"]["n"]
            n_cov = b5["coverage_pr2"][pop]["seq"][s]["CES_TI"]["n_scored"]
            if n_lad != n_cov:
                problems.append(f"scored-row count differs report vs npz {pop} s{s}: {n_lad} vs {n_cov}")
    return problems


def main():
    b5 = _load(DATA / ".b5_summary.json")
    b1 = _load(DATA / ".b1_gate_summary.json")
    b2c = _load(DATA / ".b2c_v3_summary.json")
    b3c = _load(DATA / ".b3c_b3k8_summary.json")
    probe = _load(DATA / ".b3_probe_summary_b3k8.json")
    b4 = _load(DATA / ".b4_scale_summary.json")
    ws = _load(DATA / ".wsweep_hf_summary.json")
    audit = _load(DATA / ".protocol_audit_stats.json")
    spikes = _load(DATA / ".b5_spike_structure.json")
    lat = _load(DATA / ".latency_benchmark.json")
    b22, b23, b24 = (_load(DATA / f".b{k}_verdict.json") for k in (22, 23, 24))
    b26 = _load(DATA / ".b26_precheck.json")
    b17b = _load(DATA / ".b17b_summary.json")
    b14c, b14s_cut = _load(DATA / ".b14c_verdict.json"), _load(DATA / ".b14s_cut_rescore.json")
    r10, hier = _load(DATA / ".b27_summary_R10.json"), _load(DATA / ".b27c_hier.json")
    b28s1, b28s2, b28s3, b28s4 = _load(DATA / ".b28_stage1.json"), _load(DATA / ".b28_stage2.json"), _load(DATA / ".b28_stage3.json"), _load(DATA / ".b28_stage4.json")
    for name in ("headline", "coverage_pr2", "ladder", "conformal", "peak_and_largegap", "campaign", "cut_sensitivity", "ablation", "mnar"):
        if "error" in b5.get(name, {}):
            raise SystemExit(f".b5_summary.json block {name} failed: {b5[name]['error']}")

    numbers = {
        "_source": "frozen evaluation artifacts under data/ (no retraining, no re-scoring); see module docstring",
        "protocol": protocol(b5, b1),
        "data_ledger": data_ledger(audit, spikes),
        "headline": headline(b5),
        "gate_b1": gate_b1(b1),
        "b2_v3_confirmation": b2_v3(b2c),
        "coverage_pr2": b5["coverage_pr2"],
        "ladder": ladder(b5, b3c, probe),
        "scaling_b4": scaling_b4(b4),
        "window_sweep_held_free": window_sweep(ws),
        "conformal": b5["conformal"],
        "peak": b5["peak_and_largegap"]["peak"],
        "largegap_pooled": b5["peak_and_largegap"]["largegap"],
        "campaign": b5["campaign"],
        "cut_sensitivity": b5["cut_sensitivity"],
        "ablation_window_eval": b5["ablation"],
        "mnar": b5["mnar"],
        "latency": latency(lat),
        "physfilter": physfilter(b22, b23, b24, b26),
        "dist_head": dist_head(b17b),
        "b14_select": b14_select(b14c, b14s_cut),
        "replicate_b27": replicate_b27(r10, hier),
        "equation_b28": equation_b28(b28s1, b28s2, b28s3, b28s4),
    }
    problems = crosscheck(numbers, b5, b3c, b4) + transcription_check(numbers)
    if problems:
        raise SystemExit("artifacts disagree with each other:\n  " + "\n  ".join(problems))
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(numbers, indent=2, default=float), encoding="utf-8")
    print(f"wrote {OUT_PATH}")
    print("crosscheck: B.5 ladder/headline, B.3, B.4 and the TEST reports agree on every seed and population")
    print("transcription: sec:physfilter (tab:dual, V_rot floor), sec:dist and the sec:bulk B.14 sec. 10 values tab:replicate (B.27/B.27c) and the sec:model B.28 paragraph in main.tex match the artifacts")

    for pop in POPS:
        print(f"\n=== headline / {pop} (TEST, skill vs PCHIP [95% CI]) ===")
        for arm in ("seq", "win"):
            for t in TARGETS:
                cells = []
                for s in SEEDS:
                    e = numbers["headline"][pop][arm]["per_seed"][s][t]["pchip"]
                    cells.append(f"{e['skill']:+.3f}[{e['ci'][0]:+.2f},{e['ci'][1]:+.2f}]{'*' if e['pass'] else ' '}")
                pr = numbers["headline"][pop][arm]["pr4_pass"][t]
                print(f"  {arm:>4} {t:>6}  " + "  ".join(cells) + f"   PR4 pchip {pr['pchip']}/4 persist {pr['persistence']}/4 gp_causal {pr['gp_causal']}/4")
        print(f"  seq-win paired T_i: " + ", ".join(f"{numbers['headline'][pop]['seq_vs_win_paired'][s]['CES_TI']['skill']:+.3f}" for s in SEEDS)
              + f"  (positive {numbers['headline'][pop]['seq_vs_win_positive_TI']}/4, sig {numbers['headline'][pop]['seq_vs_win_sig_TI']}/4)")
    g = numbers["gate_b1"]
    print(f"\n=== B.1 gate: pooled {g['pooled_mean_TI']:+.3f} CI [{g['pooled_ci95'][0]:+.3f}, {g['pooled_ci95'][1]:+.3f}], positive {g['n_positive']}/16, backbone = {g['gate']['backbone']}")
    for pop in POPS:
        v = numbers["ladder"][pop]["verdict"]
        print(f"=== ladder / {pop}: b3 vs anchor {v['b3_vs_anchor_TI_sig_win']}, mean b3 vs seq {v['mean_b3_vs_seq_TI']:+.3f}, tolerance {'met' if v['within_backbone_tolerance'] else 'FAILED'}")
    for pop in POPS:
        c = numbers["campaign"][pop]["pr4_pass_counts"]
        print(f"=== campaign / {pop}: seq vs pchip {c['seq']['CES_TI']['pchip']}/4, win OFF {c['win']['CES_TI']['pchip']}/4, seq-win sig {c['seq_vs_win_TI_sig']}/4")


if __name__ == "__main__":
    main()
