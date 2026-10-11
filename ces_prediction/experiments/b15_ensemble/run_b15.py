"""B.15 runner -- PREREGISTRATION_B15.md: the 4-init deep ensemble, re-inference only.

Stage 1 (cut): members = the B.1 grid `.b1_seqv2_s{s}_i{42,1,7,123}`; nothing trains.
Stage 2 (incl): members = `.b5i_seqv2_s{s}` (init = s) + the three other inits trained
by this runner with the B.5 inclusive env (`--train-incl`, GPU), then the same scoring.

For every split: eval_seq with CES_SEQ_ENSEMBLE_DIRS (normalized predictions averaged;
the members share the split's train files, hence its stats), the PR4 ladder bootstrap,
paired vs the canonical run (init = split seed, the paper's headline run) and vs each
member. The arm dir gets the canonical run's metrics.json (same stats) plus the member
list, so eval_seq / paired_model_compare run unchanged.

Usage (repo root):
  py ces_prediction/experiments/b15_ensemble/run_b15.py                 # stage 1
  py ces_prediction/experiments/b15_ensemble/run_b15.py --train-incl    # stage 2 (GPU)
Resumable. Output: data/.b15_ens4_{cut,incl}_s{s}/, data/.b15_summary.json.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
CES_DIR = REPO_ROOT / "ces_prediction"
SEQ_DIR = CES_DIR / "experiments" / "seq"
PAIRED = CES_DIR / "experiments" / "paired_model_compare.py"
DATA = REPO_ROOT / "data"
sys.path.insert(0, str(CES_DIR / "experiments" / "b1_gate"))
from run_b1_gate import GATE_ENV, AMBIENT_VARS, run_step, BOOTSTRAP_SEED, BOOTSTRAP_B  # noqa: E402

SEEDS = (42, 1, 7, 123)
INITS = (42, 1, 7, 123)
PRACTICAL_EPS = 0.02
POPS = {
    "cut": {"cut": "3000", "member": ".b1_seqv2_s{s}_i{i}", "canonical": ".b1_seqv2_s{s}_i{s}",
            "control_metrics": ".b1_w2cut_s{s}", "split": ".b1_w2cut_split_s{s}"},
    "incl": {"cut": "0", "member": ".b15_member_incl_s{s}_i{i}", "canonical": ".b5i_seqv2_s{s}",
             "control_metrics": ".wsweep_hf_w2_s{s}", "split": ".b1_manifest_s{s}"},
}


def member_dir(pop, s, i):
    if pop == "incl" and i == s:
        return DATA / POPS[pop]["canonical"].format(s=s)
    return DATA / POPS[pop]["member"].format(s=s, i=i)


def base_env(pop):
    env = os.environ.copy()
    for var in AMBIENT_VARS:
        env.pop(var, None)
    env.update(GATE_ENV)
    env.update({"CES_TI_SPIKE_CUT_EV": POPS[pop]["cut"], "CES_SEQ_MODEL": "v2", "CES_PER_SHOT_NORM": "1",
                "CES_SPLIT_TAG": "test"})
    return env


def train_incl_members(resume):
    """Stage 2: the three non-canonical inits of the inclusive backbone (B.5 env, cap 30)."""
    p = POPS["incl"]
    for s in SEEDS:
        for i in INITS:
            if i == s:
                continue
            out = member_dir("incl", s, i)
            if resume and (out / "weights" / "seq_lstm.pth").exists() and (out / "metrics.json").exists():
                continue
            out.mkdir(parents=True, exist_ok=True)
            env = base_env("incl")
            env.update({"CES_SEED": str(s), "CES_INIT_SEED": str(i), "CES_SPLIT_DIR": str(DATA / p["split"].format(s=s)),
                        "CES_OUTPUT_DIR": str(out), "CES_SEQ_DEVICE": "cuda",
                        "CES_CONTROL_METRICS": str(DATA / p["control_metrics"].format(s=s) / "metrics.json")})
            print(f"[b15] === train {out.name}", flush=True)
            if not run_step([SEQ_DIR / "train_seq.py"], env, out / "run.log", "train",
                            artifacts=(out / "weights" / "seq_lstm.pth", out / "metrics.json")):
                raise SystemExit(f"FATAL: {out.name} failed")


def score_split(pop, s):
    p = POPS[pop]
    members = [member_dir(pop, s, i) for i in INITS]
    for m in members:
        if not (m / "weights" / "seq_lstm.pth").exists():
            raise SystemExit(f"FATAL: member missing: {m}")
    canonical = DATA / p["canonical"].format(s=s)
    out = DATA / f".b15_ens4_{pop}_s{s}"
    out.mkdir(parents=True, exist_ok=True)
    log = out / "run.log"
    if not (out / "metrics.json").exists():
        m = json.loads((canonical / "metrics.json").read_text(encoding="utf-8"))
        # identical stats across members is a precondition of averaging in normalized space
        for d in members:
            other = json.loads((d / "metrics.json").read_text(encoding="utf-8"))
            if other["normalization"] != m["normalization"]:
                raise SystemExit(f"FATAL: {d.name} has different normalization stats than {canonical.name}")
        m["experiment"] = "b15_ens4"
        m["ensemble_members"] = [d.name for d in members]
        (out / "metrics.json").write_text(json.dumps(m, indent=2), encoding="utf-8")
    env = base_env(pop)
    env.update({"CES_SEED": str(s), "CES_INIT_SEED": str(s), "CES_SPLIT_DIR": str(DATA / p["split"].format(s=s)),
                "CES_OUTPUT_DIR": str(out), "CES_CONTROL_METRICS": str(DATA / p["control_metrics"].format(s=s) / "metrics.json"),
                "CES_SEQ_ENSEMBLE_DIRS": os.pathsep.join(str(d) for d in members)})
    print(f"[b15] === {out.name} <- {[d.name for d in members]}", flush=True)
    if not (out / "comparison_errors_test.npz").exists():
        if not run_step([SEQ_DIR / "eval_seq.py"], env, log, "eval(test, ens4)",
                        artifacts=(out / "comparison_errors_test.npz", out / "comparison_metrics.json")):
            raise SystemExit(f"FATAL: eval failed for {out.name}")
    if not (out / "bootstrap_summary.json").exists():
        if not run_step([CES_DIR / "bootstrap_compare.py"], env, log, "bootstrap",
                        artifacts=(out / "bootstrap_summary.json",)):
            raise SystemExit(f"FATAL: bootstrap failed for {out.name}")
    pairs = [("paired_vs_canonical.json", canonical)] + [(f"paired_vs_member_i{i}.json", d) for i, d in zip(INITS, members)]
    for name, other in pairs:
        if (out / name).exists():
            continue
        if not run_step([PAIRED, "--a", out / "comparison_errors_test.npz", "--b", other / "comparison_errors_test.npz",
                         "--out", out / name], env, log, name[:-5], artifacts=(out / name,)):
            raise SystemExit(f"FATAL: {name} failed for {out.name}")
    return out


def summarize(pops):
    summary = {"practical_eps": PRACTICAL_EPS, "pops": {}}
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    for pop in pops:
        rows, ti_vals, vt_deficit = {}, [], 0
        for s in SEEDS:
            out = DATA / f".b15_ens4_{pop}_s{s}"
            if not (out / "paired_vs_canonical.json").exists():
                continue
            can = json.loads((out / "paired_vs_canonical.json").read_text(encoding="utf-8"))["targets"]
            mem = {i: json.loads((out / f"paired_vs_member_i{i}.json").read_text(encoding="utf-8"))["targets"]
                   for i in INITS}
            bs = json.loads((out / "bootstrap_summary.json").read_text(encoding="utf-8"))["splits"]["test"]
            cm = json.loads((out / "comparison_metrics.json").read_text(encoding="utf-8"))["per_target"]
            rows[str(s)] = {}
            for t in ("CES_TI", "CES_VT"):
                rows[str(s)][t] = {
                    "paired_vs_canonical": can[t]["skill_point"], "ci95": can[t]["skill_ci95"],
                    "ens_better": can[t]["a_better"], "canonical_better": can[t]["b_better"],
                    "paired_vs_members": {str(i): mem[i][t]["skill_point"] for i in INITS},
                    "min_vs_member": min(mem[i][t]["skill_point"] for i in INITS),
                    "skill_vs_pchip": cm[t]["skill_vs_pchip"], "pchip_pass": bs[t]["pchip"]["pass"],
                    "skill_vs_gp_causal": bs[t].get("gp_causal", {}).get("skill_point"),
                    "gp_causal_pass": bs[t].get("gp_causal", {}).get("pass"),
                }
            ti_vals.append(can["CES_TI"]["skill_point"])
            vt_deficit += int(can["CES_VT"]["b_better"])
        if not rows:
            continue
        arr = np.asarray(ti_vals)
        idx = rng.integers(0, len(arr), size=(BOOTSTRAP_B, len(arr)))
        ci = [float(np.percentile(arr[idx].mean(1), 2.5)), float(np.percentile(arr[idx].mean(1), 97.5))]
        n_pass = sum(rows[s]["CES_TI"]["ens_better"] for s in rows)
        verdict = ("deployment_recommendation" if (n_pass >= 3 and arr.mean() > PRACTICAL_EPS and vt_deficit == 0)
                   else "tie" if abs(arr.mean()) <= PRACTICAL_EPS
                   else "deficit_check_implementation" if ci[1] < 0 else "inconclusive")
        summary["pops"][pop] = {"per_split": rows, "ti_mean": float(arr.mean()), "ti_run_ci95": ci,
                                "ti_splits_significant": n_pass, "vt_significant_deficits": vt_deficit,
                                "verdict": verdict}
        print(f"[b15] {pop}: paired T_i vs canonical " + " ".join(f"{v:+.3f}" for v in ti_vals)
              + f" mean {arr.mean():+.3f} CI {ci[0]:+.3f}..{ci[1]:+.3f}; sig {n_pass}/{len(arr)}; "
              f"V_rot deficits {vt_deficit} -> {verdict}")
    (DATA / ".b15_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pops", nargs="*", default=["cut"])
    ap.add_argument("--train-incl", action="store_true", help="stage 2: train the 3 missing inclusive inits (GPU)")
    args = ap.parse_args()
    if args.train_incl:
        train_incl_members(resume=True)
        if "incl" not in args.pops:
            args.pops.append("incl")
    for pop in args.pops:
        for s in SEEDS:
            score_split(pop, s)
    summarize(args.pops)


if __name__ == "__main__":
    main()
