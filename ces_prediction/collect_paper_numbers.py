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
    }
    problems = crosscheck(numbers, b5, b3c, b4)
    if problems:
        raise SystemExit("artifacts disagree with each other:\n  " + "\n  ".join(problems))
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(numbers, indent=2, default=float), encoding="utf-8")
    print(f"wrote {OUT_PATH}")
    print("crosscheck: B.5 ladder/headline, B.3, B.4 and the TEST reports agree on every seed and population")

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
