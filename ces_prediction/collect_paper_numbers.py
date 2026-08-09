"""Collect every paper-facing number from the frozen evaluation artifacts.

The paper claims its headline numbers are read from frozen artifacts; this script
makes that literally true and mechanically checkable. It reads only committed run
directories (never retrains, never re-scores) and writes one JSON that the LaTeX
sources are transcribed from, plus a human-readable table.

Two evaluation treatments are reported side by side, because they answer different
questions and the paper must not blur them:

  genuine  held/forward-filled V_rot values EXCLUDED from scoring (evaluate.py's
           default). These are not measurements, so this is the defensible headline.
  stuck0   held values KEPT (the historical convention). Deflates V_rot RMSE by
           35-55% because a held target has ~0 baseline error.

Usage (repo root):  py ces_prediction/collect_paper_numbers.py
Writes: docs/paper/paper_numbers.json
"""

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = REPO_ROOT / "docs" / "paper" / "paper_numbers.json"

SEEDS = (42, 1, 7, 123)
RUN_DIR = {
    42: REPO_ROOT / "data" / ".vt_repro_out",
    1: REPO_ROOT / "data" / ".vt_repro_ms_1",
    7: REPO_ROOT / "data" / ".vt_repro_ms_7",
    123: REPO_ROOT / "data" / ".vt_repro_ms_123",
}
ABLATION_DIR = {
    "no_fast": REPO_ROOT / "data" / ".vt_repro_ab_no_fast",
    "no_history": REPO_ROOT / "data" / ".vt_repro_ab_no_history",
}
BEFORE_DIR = REPO_ROOT / "data" / ".final_out"          # iter2 "before" baseline
WSWEEP = REPO_ROOT / "data" / ".wsweep_hf_summary.json"  # held-free window sweep
STUCKFREE = REPO_ROOT / "data" / ".sf_summary.json"      # held-free 4-seed retrain
GP_ARM = REPO_ROOT / "data" / ".gp_analysis.json"        # §8p post-hoc GP baseline arm
FITFAIL = REPO_ROOT / "data" / ".fitfail_analysis.json"  # §8q CES fit-failure sensitivity
PEAKHELD = REPO_ROOT / "data" / ".peakheld_analysis.json"  # §8r peak x held crosstab
PERSHOT_CAMPAIGN = REPO_ROOT / "data" / ".pershot_summary.json"          # §8s part 1
PERSHOT_RANDOM = REPO_ROOT / "data" / ".pershot_random_summary.json"     # §8s part 2
TREATMENTS = {"genuine": "__test_genuine", "stuck0": "__test_stuck0"}
TARGETS = ("CES_TI", "CES_VT")
BASELINES = ("persistence", "linear", "pchip", "ar_local")


def _load(path):
    return json.loads(Path(path).read_text(encoding="utf-8")) if Path(path).exists() else None


def headline():
    """Per-seed skill + shot-clustered CI, both treatments, vs pchip and linear."""
    out = {}
    for treatment, suffix in TREATMENTS.items():
        rows = {}
        for seed in SEEDS:
            d = RUN_DIR[seed]
            cm = _load(d / f"comparison_metrics{suffix}.json")
            bs = _load(d / f"bootstrap_summary{suffix}.json")
            if cm is None or bs is None:
                rows[seed] = {"missing": True}
                continue
            entry = {"eval_samples": cm["eval_samples"]}
            for t in TARGETS:
                pt = cm["per_target"][t]
                boot = bs["splits"]["test"][t]
                e = {
                    "n": pt["n"],
                    "rmse_model": pt["rmse_model"],
                    "rmse": {b: pt["baselines"][b]["rmse"] for b in BASELINES
                             if b in pt["baselines"]},
                    "skill_vs_pchip": pt["skill_vs_pchip"],
                    "future_neighbor_fraction": pt["future_neighbor_fraction"],
                }
                for arm in ("pchip", "linear"):
                    if arm in boot:
                        e[f"ci_vs_{arm}"] = {
                            "skill": boot[arm]["skill_point"],
                            "ci95": boot[arm]["skill_ci95"],
                            "pass": boot[arm]["pass"],
                            "n_shots": boot[arm]["n_shots"],
                        }
                entry[t] = e
            rows[seed] = entry
        # verdict tallies the paper quotes
        summary = {}
        for t in TARGETS:
            for arm in ("pchip", "linear"):
                sk, ps = [], 0
                for seed in SEEDS:
                    e = rows[seed].get(t, {}).get(f"ci_vs_{arm}")
                    if e:
                        sk.append(e["skill"])
                        ps += int(e["pass"])
                if sk:
                    summary[f"{t}_vs_{arm}"] = {
                        "skills": sk, "min": min(sk), "max": max(sk),
                        "mean": sum(sk) / len(sk), "pass": f"{ps}/{len(sk)}",
                    }
        out[treatment] = {"per_seed": rows, "summary": summary}
    return out


def gap_bins():
    """Δt-stratified bins for seed 42, both treatments (§4.3 of THESIS_RESULTS)."""
    out = {}
    for treatment, suffix in TREATMENTS.items():
        cm = _load(RUN_DIR[42] / f"comparison_metrics{suffix}.json")
        if cm:
            out[treatment] = {t: cm["per_target"][t]["bins"] for t in TARGETS}
    return out


def ablations():
    """Input-modality ablation, validation split, skill vs persistence.

    The `full` arm is the headline run's own validation evaluation, so all three
    arms share one split, one budget, and one persistence baseline (persistence is
    always computed from the REAL history, never the ablated inputs).
    """
    out = {}
    for name, d in {"full": RUN_DIR[42], **ABLATION_DIR}.items():
        em = _load(d / "eval_metrics.json")
        if not em:
            continue
        out[name] = {
            "split": em.get("split", "val"),
            "val_shots": em.get("val_shots"),
            "per_target": {
                t: {k: v for k, v in em["per_target"][t].items()
                    if k in ("n", "rmse_model", "rmse_persistence", "rmse_mean_baseline",
                             "skill_vs_persistence", "r2_vs_mean")}
                for t in TARGETS if t in em["per_target"]
            },
        }
    return out


def peak():
    """High-variability ('peak') skill vs the matching global skill on the same run.

    peak_analysis scores the same validation npz compare_baselines wrote, so the
    global reference must come from that run's own comparison_metrics.json --
    quoting a global number from a different treatment would make the
    global-to-peak jump meaningless.
    """
    pm = _load(RUN_DIR[42] / "peak_metrics.json")
    cm = _load(RUN_DIR[42] / "comparison_metrics.json")
    if pm is None:
        return None
    out = {"split": pm.get("split"), "params": pm.get("params"), "per_target": {}}
    for t in TARGETS:
        blk = pm.get("per_target", {}).get(t, {}).get("input_only", {}).get("ces_activity")
        if not blk:
            continue
        e = {k: blk.get(k) for k in
             ("n_peak_rows", "n_peak_shots", "peak_skill_vs_pchip",
              "peak_rmse_model", "peak_rmse_pchip", "peak_skill_ci95", "pass")}
        if blk.get("vs_linear"):
            e["vs_linear"] = blk["vs_linear"]
        if cm:
            e["global_skill_vs_pchip"] = cm["per_target"][t]["skill_vs_pchip"]
            e["global_n"] = cm["per_target"][t]["n"]
        out["per_target"][t] = e
    return out


def before_baseline():
    """The iter2 'before' model of the honest-progression figure."""
    cm = _load(BEFORE_DIR / "comparison_metrics.json")
    bs = _load(BEFORE_DIR / "bootstrap_summary.json")
    if cm is None:
        return None
    out = {"eval_samples": cm["eval_samples"]}
    for t in TARGETS:
        out[t] = {"n": cm["per_target"][t]["n"],
                  "rmse_model": cm["per_target"][t]["rmse_model"],
                  "skill_vs_pchip": cm["per_target"][t]["skill_vs_pchip"]}
        if bs:
            for split in bs.get("splits", {}):
                arm = bs["splits"][split].get(t, {}).get("pchip")
                if arm:
                    out[t][f"ci_{split}"] = {"skill": arm["skill_point"],
                                             "ci95": arm["skill_ci95"],
                                             "pass": arm["pass"]}
    return out


def window_sweep():
    s = _load(WSWEEP)
    if not s:
        return None
    return {"protocol": s["protocol"],
            "per_point": [{k: v for k, v in p.items() if k != "statuses"}
                          for p in s["per_point"]],
            "test_file_invariance": s["test_file_invariance"]}


def stuckfree():
    s = _load(STUCKFREE)
    if not s:
        return None
    return {k: v for k, v in s.items() if k != "seeds"} | {
        "per_seed": [{"seed": e["seed"],
                      "skill_vs_pchip_TI": e.get("skill_vs_pchip_TI"),
                      "pchip_pass_TI": e.get("pchip_pass_TI"),
                      "skill_vs_pchip_VT": e.get("skill_vs_pchip_VT"),
                      "pchip_pass_VT": e.get("pchip_pass_VT"),
                      "paired_TI": e.get("paired_TI"),
                      "paired_VT": e.get("paired_VT")}
                     for e in s.get("seeds", []) if e.get("status") == "ok"]}


def gp_arm():
    """§8p: the post-hoc GP arm -- the strongest offline smoother, which the model ties.

    Scored on the byte-identical population as the headline (the added `gp` method cannot
    shrink `valid`: its NaN condition equals `ar_local`'s), so `model_vs_pchip_crosscheck`
    must reproduce the headline skill. `crosscheck()` enforces exactly that.
    """
    s = _load(GP_ARM)
    if not s:
        return None
    arms = ("model_vs_gp", "gp_vs_pchip", "model_vs_pchip_crosscheck",
            "model_vs_gp_peak", "model_vs_gp_dt_le_15ms", "model_vs_gp_dt_gt_15ms")
    out = {"npz": s["npz"], "bootstrap_resamples": s["bootstrap_resamples"],
           "bootstrap_seed": s["seed"], "per_seed": {}, "summary": {}}
    for seed in SEEDS:
        blk = s["per_seed"].get(str(seed))
        if not blk:
            continue
        out["per_seed"][seed] = {
            t: {"n": blk[t]["n"], "rmse": blk[t]["rmse"],
                **{a: blk[t][a] for a in arms if a in blk[t]}}
            for t in TARGETS if t in blk
        }
    for t in TARGETS:
        # the subset arms matter too: the paper quotes "subsets do not break the tie"
        for arm in ("gp_vs_pchip", "model_vs_gp", "model_vs_gp_peak",
                    "model_vs_gp_dt_le_15ms", "model_vs_gp_dt_gt_15ms"):
            pts = [out["per_seed"][s_][t][arm]["skill_point"]
                   for s_ in out["per_seed"] if arm in out["per_seed"][s_].get(t, {})]
            ps = sum(out["per_seed"][s_][t][arm]["pass"]
                     for s_ in out["per_seed"] if arm in out["per_seed"][s_].get(t, {}))
            if pts:
                out["summary"][f"{t}_{arm}"] = {
                    "skills": pts, "min": min(pts), "max": max(pts),
                    "mean": sum(pts) / len(pts), "pass": f"{ps}/{len(pts)}",
                }
    return out


def fitfail_sensitivity():
    """§8q: dropping CES spectral-fit failures (Ti > 3 keV) from every arm alike.

    The artifacts DEFLATE the headline -- the paper quotes this to say the pre-registered
    population is conservative, so the dropped-row counts belong in the artifact too.
    """
    s = _load(FITFAIL)
    if not s:
        return None
    cuts = ("full", "le_3000ev", "le_2089ev")
    out = {"npz": s["npz"], "target": s["target"], "thresholds_ev": s["thresholds_ev"],
           "bootstrap_resamples": s["bootstrap_resamples"], "bootstrap_seed": s["seed"],
           "per_seed": {}, "summary": {}}
    for seed in SEEDS:
        blk = s["per_seed"].get(str(seed))
        if not blk:
            continue
        out["per_seed"][seed] = {k: blk[k] for k in ("n_total", "y_true_max_ev")} | {
            c: blk[c] for c in cuts if c in blk
        }
    for c in cuts:
        pts = [out["per_seed"][s_][c]["skill_point"] for s_ in out["per_seed"]
               if c in out["per_seed"][s_]]
        ps = sum(out["per_seed"][s_][c]["pass"] for s_ in out["per_seed"]
                 if c in out["per_seed"][s_])
        if pts:
            out["summary"][c] = {"skills": pts, "min": min(pts), "max": max(pts),
                                 "mean": sum(pts) / len(pts), "pass": f"{ps}/{len(pts)}"}
    return out


def peak_held():
    """§8r: the peak x held crosstab that decomposes the global `CES_VT` n.s.

    Scored on the held-KEPT population, so these numbers must never be quoted next to the
    `genuine` headline without saying so -- the treatment is carried in the JSON.
    """
    s = _load(PEAKHELD)
    if not s:
        return None
    cells = ("peak_genuine", "peak_held", "bulk_genuine", "bulk_held")
    subsets = ("all", "genuine_only", "held_only")
    out = {"treatment": s["treatment"], "held_definition": s["held_definition"],
           "bootstrap_resamples": s["bootstrap_resamples"], "bootstrap_seed": s["seed"],
           "per_seed": {}, "summary": {}}
    for seed in SEEDS:
        blk = s["per_seed"].get(str(seed))
        if not blk:
            continue
        out["per_seed"][seed] = {
            t: {k: v[t][k] for k in ("n", "held_fraction", "peak_fraction",
                                     "held_fraction_in_peak", "held_fraction_in_bulk",
                                     "crosstab", "cells", "subsets")}
            for t, v in ((t, blk["per_target"]) for t in TARGETS) if t in blk["per_target"]
        }
    for t in TARGETS:
        rows = [out["per_seed"][s_][t] for s_ in out["per_seed"] if t in out["per_seed"][s_]]
        if not rows:
            continue
        # the headline of §8r: peaks are HELD-RICHER than the bulk, on every seed
        out["summary"][f"{t}_held_richer_in_peak_seeds"] = sum(
            1 for r in rows
            if (r["held_fraction_in_peak"] or 0) > (r["held_fraction_in_bulk"] or 0))
        for group, keys in (("cells", cells), ("subsets", subsets)):
            for k in keys:
                pts = [r[group][k]["skill_point"] for r in rows if r[group][k]["n"]]
                ps = sum(r[group][k]["pass"] for r in rows if r[group][k]["n"])
                pers = [r[group][k]["vs_persistence"]["skill_point"] for r in rows
                        if r[group][k]["n"] and r[group][k]["vs_persistence"]["n_shots"]]
                pers_ps = sum(r[group][k]["vs_persistence"]["pass"] for r in rows
                              if r[group][k]["n"])
                if pts:
                    out["summary"][f"{t}_{k}"] = {
                        "skills_vs_pchip": pts, "min": min(pts), "max": max(pts),
                        "pass_vs_pchip": f"{ps}/{len(pts)}",
                        "pass_vs_persistence": f"{pers_ps}/{len(pers)}" if pers else None,
                    }
    return out


def per_shot_norm():
    """§8s: the campaign-transfer repair, on both the split it targets and the headline.

    Both arms are carried because the verdict is a trade: a large significant gain under
    campaign shift, against point estimates that are slightly negative (but never
    significantly so) on the headline split. Quoting either half alone would be the §8j
    error in reverse.
    """
    camp, rand = _load(PERSHOT_CAMPAIGN), _load(PERSHOT_RANDOM)
    if not camp and not rand:
        return None
    out = {}
    if camp:
        out["campaign_split"] = {
            "design": camp["design"], "verdict_rule": camp["verdict_rule"],
            "verdict": camp["verdict"],
            "per_seed": {r["seed"]: {t: {"base": r["base"][t]["skill_vs_pchip"],
                                         "per_shot": r["per_shot"][t]["skill_vs_pchip"],
                                         "base_pass": r["base"][t]["pass"],
                                         "per_shot_pass": r["per_shot"][t]["pass"],
                                         "paired": r["paired"][t]}
                                     for t in TARGETS}
                         for r in camp["runs"] if r["status"] == "ok"},
        }
    if rand:
        out["headline_split"] = {
            "design": rand["design"], "verdict_rule": rand["verdict_rule"],
            "verdict": rand["verdict"],
            "per_seed": {r["seed"]: {t: {"per_shot": r[t]["skill_vs_pchip"],
                                         "per_shot_pass": r[t]["pass"],
                                         "paired_vs_control": r[t]["paired_vs_control"]}
                                     for t in TARGETS if t in r}
                         for r in rand["runs"] if r.get("CES_TI")},
        }
    return out


def crosscheck(numbers, tol=1e-4):
    """Every post-hoc arm re-scores the headline population; it must reproduce the headline.

    §8p did this by hand ("recomputed model-vs-PCHIP skills match the headline to 4
    decimals"). Doing it here means a future re-run that silently changes the scored
    population fails loudly instead of quietly publishing two different headlines.
    """
    problems = []
    per_seed = numbers["headline"]["genuine"]["per_seed"]
    for name, key in (("gp_arm", "model_vs_pchip_crosscheck"), ("fitfail_sensitivity", "full")):
        blk = numbers.get(name)
        if not blk:
            continue
        for seed, row in blk["per_seed"].items():
            arms = {"CES_TI": row[key]} if name == "fitfail_sensitivity" else \
                   {t: row[t][key] for t in TARGETS if key in row.get(t, {})}
            for t, arm in arms.items():
                ref = per_seed.get(seed, {}).get(t, {}).get("ci_vs_pchip", {}).get("skill")
                if ref is None:
                    continue
                if abs(arm["skill_point"] - ref) > tol:
                    problems.append(
                        f"{name}/{seed}/{t}: {arm['skill_point']:.6f} != headline {ref:.6f}")
    return problems


def main():
    numbers = {
        "_source": "frozen evaluation artifacts under data/ (no retraining, no re-scoring)",
        "_treatments": {
            "genuine": "held/forward-filled V_rot excluded from scoring (headline)",
            "stuck0": "held values kept (historical convention; deflates V_rot RMSE)",
        },
        "headline": headline(),
        "gap_bins_seed42": gap_bins(),
        "ablation_val_vs_persistence": ablations(),
        "peak_seed42": peak(),
        "before_baseline_iter2": before_baseline(),
        "window_sweep_held_free": window_sweep(),
        "stuckfree_4seed": stuckfree(),
        "gp_arm": gp_arm(),
        "fitfail_sensitivity": fitfail_sensitivity(),
        "peak_held_crosstab": peak_held(),
        "per_shot_norm": per_shot_norm(),
    }
    problems = crosscheck(numbers)
    if problems:
        raise SystemExit("post-hoc arms disagree with the headline population:\n  "
                         + "\n  ".join(problems))
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(numbers, indent=2), encoding="utf-8")
    print(f"wrote {OUT_PATH}")
    print("crosscheck: post-hoc arms reproduce the headline population on every seed")

    for treatment in TREATMENTS:
        print(f"\n=== headline / {treatment} ===")
        print(f"{'seed':>5} {'target':>7} {'n':>7} {'rmse':>9} "
              f"{'skill_pchip':>12} {'ci95':>22} {'gate':>5}")
        for seed in SEEDS:
            row = numbers["headline"][treatment]["per_seed"][seed]
            for t in TARGETS:
                e = row.get(t)
                if not e:
                    continue
                c = e.get("ci_vs_pchip", {})
                ci = c.get("ci95", [float("nan")] * 2)
                print(f"{seed:>5} {t:>7} {e['n']:>7} {e['rmse_model']:>9.2f} "
                      f"{c.get('skill', float('nan')):>+12.4f} "
                      f"[{ci[0]:+.3f}, {ci[1]:+.3f}]".ljust(58)
                      + f"{'PASS' if c.get('pass') else 'n.s.':>5}")
        for k, v in numbers["headline"][treatment]["summary"].items():
            print(f"   {k:24s} {v['min']:+.4f}..{v['max']:+.4f}  PASS {v['pass']}")

    for name in ("gp_arm", "fitfail_sensitivity"):
        blk = numbers.get(name)
        if not blk:
            print(f"\n=== {name} === MISSING ARTIFACT")
            continue
        print(f"\n=== {name} (genuine, held-out TEST) ===")
        for k, v in blk["summary"].items():
            print(f"   {k:32s} {v['min']:+.4f}..{v['max']:+.4f}  "
                  f"mean {v['mean']:+.4f}  PASS {v['pass']}")

    blk = numbers.get("per_shot_norm")
    if blk:
        for split, label in (("campaign_split", "campaign (temporal) split"),
                             ("headline_split", "headline (random) split")):
            sub = blk.get(split)
            if not sub:
                continue
            print(f"\n=== per_shot_norm / {label} "
                  f"-> {sub['verdict'].get('overall')} ===")
            for t_ in TARGETS:
                v = sub["verdict"].get(t_, {})
                if "favor_seeds" in v:
                    print(f"   {t_}: paired mean {v['mean']:+.4f}  favour {v['favor_seeds']}  "
                          f"significant {v['significant_favor_seeds']}")
                else:
                    print(f"   {t_}: paired mean {v['mean']:+.4f}  "
                          f"significant losses {v['significant_losses']}  wins {v['significant_wins']}")

    blk = numbers.get("peak_held_crosstab")
    if blk:
        print("\n=== peak_held_crosstab (held-KEPT population, held-out TEST) ===")
        for k, v in blk["summary"].items():
            if isinstance(v, int):
                print(f"   {k:32s} {v}/4 seeds")
                continue
            print(f"   {k:32s} {v['min']:+.3f}..{v['max']:+.3f}  "
                  f"vs pchip {v['pass_vs_pchip']}  vs persistence {v['pass_vs_persistence']}")


if __name__ == "__main__":
    main()
