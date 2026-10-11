"""B.14 runner -- PREREGISTRATION_B14.md: the training loss's influence function.

Arms `huber` / `tnll` (CES_SEQ_LOSS), seq_v2, split {42, 1, 7, 123} x init = split seed,
both populations. Everything but the loss is the B.1 stage-B env (cut) or the B.5
inclusive-backbone env (incl): same manifests, same control metrics, per-shot ON,
held-free, patience 6. Epoch cap 60 (sec. 2: non-binding for a slower-converging loss;
the control's cap was 30 and its s123 run hit it). The loss scale is the sec. 8aq
order-4 noise bound in physical units, pinned here, never defaulted.

Control = the frozen backbone runs (`.b1_seqv2_s{s}_i{s}`, `.b5i_seqv2_s{s}`): no
retraining. Each arm is paired row-for-row against its control (paired_model_compare)
and scored on the PR4 ladder (bootstrap_compare). The robust co-metrics (B.13) are
produced by experiments/b13_robust/rescore_b13.py over the arm dirs afterwards.

Usage (repo root):  py ces_prediction/experiments/b14_loss/run_b14.py [--smoke] [--arms huber tnll]
Resumable: finished steps are skipped on re-run.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
CES_DIR = REPO_ROOT / "ces_prediction"
SEQ_DIR = CES_DIR / "experiments" / "seq"
PAIRED = CES_DIR / "experiments" / "paired_model_compare.py"
DATA = REPO_ROOT / "data"
sys.path.insert(0, str(CES_DIR / "experiments" / "b1_gate"))
from run_b1_gate import GATE_ENV, AMBIENT_VARS, SMOKE_OVERRIDES, run_step  # noqa: E402

SEEDS = (42, 1, 7, 123)
ARMS = ("huber", "tnll")
EPOCH_CAP = "60"
# sec. 8aq order-4 difference bound (data/.noise_floor.json), per population.
POPS = {
    "cut": {"cut": "3000", "sigma_ti": "129.9", "sigma_vt": "16.4",
            "control": ".b1_seqv2_s{s}_i{s}", "control_metrics": ".b1_w2cut_s{s}",
            "split": ".b1_w2cut_split_s{s}"},
    "incl": {"cut": "0", "sigma_ti": "274.3", "sigma_vt": "16.4",
             "control": ".b5i_seqv2_s{s}", "control_metrics": ".wsweep_hf_w2_s{s}",
             "split": ".b1_manifest_s{s}"},
}


def arm_dir(arm, pop, s, smoke=False):
    return DATA / (f".b14_{arm}_{pop}_s{s}" + ("_smoke" if smoke else ""))


def base_env(pop, arm, smoke):
    env = os.environ.copy()
    for var in AMBIENT_VARS:
        env.pop(var, None)
    env.update(GATE_ENV)
    if smoke:
        env.update(SMOKE_OVERRIDES)
    p = POPS[pop]
    env.update({
        "CES_TI_SPIKE_CUT_EV": p["cut"],
        "CES_SEQ_LOSS": arm,
        "CES_SEQ_LOSS_SIGMA_TI_EV": p["sigma_ti"],
        "CES_SEQ_LOSS_SIGMA_VT_KMS": p["sigma_vt"],
        "CES_SEQ_LOSS_NU": "3",
        "CES_SEQ_EPOCHS": EPOCH_CAP,
        "CES_SEQ_MODEL": "v2",
        "CES_PER_SHOT_NORM": "1",
        "CES_SEQ_DEVICE": "cuda",
    })
    if smoke:
        env.update({"CES_SEQ_EPOCHS": "2", "CES_SEQ_MAX_FILES": "40"})
    return env


def one_run(arm, pop, s, smoke):
    p = POPS[pop]
    out = arm_dir(arm, pop, s, smoke)
    control = DATA / p["control"].format(s=s)
    split = DATA / p["split"].format(s=s)
    cm = DATA / p["control_metrics"].format(s=s) / "metrics.json"
    for needed in (control / "comparison_errors_test.npz", split / "split_manifest.json", cm):
        if not needed.exists():
            raise SystemExit(f"FATAL: missing {needed}")
    out.mkdir(parents=True, exist_ok=True)
    log = out / "run.log"
    env = base_env(pop, arm, smoke)
    env.update({"CES_SEED": str(s), "CES_INIT_SEED": str(s), "CES_SPLIT_DIR": str(split),
                "CES_OUTPUT_DIR": str(out), "CES_CONTROL_METRICS": str(cm), "CES_SPLIT_TAG": "test"})
    rec = {"arm": arm, "pop": pop, "seed": s, "out": str(out), "status": "ok"}
    t0 = time.time()
    print(f"[b14] === {out.name}", flush=True)
    if not (out / "weights" / "seq_lstm.pth").exists() or not (out / "metrics.json").exists():
        if not run_step([SEQ_DIR / "train_seq.py"], env, log, "train",
                        artifacts=(out / "weights" / "seq_lstm.pth", out / "metrics.json")):
            rec["status"] = "train_failed"
            return rec
    if not (out / "comparison_errors_test.npz").exists():
        if not run_step([SEQ_DIR / "eval_seq.py"], env, log, "eval(test)",
                        artifacts=(out / "comparison_errors_test.npz", out / "comparison_metrics.json")):
            rec["status"] = "compare_failed"
            return rec
    if smoke:
        rec["minutes"] = round((time.time() - t0) / 60, 1)
        return rec
    if not (out / "bootstrap_summary.json").exists():
        if not run_step([CES_DIR / "bootstrap_compare.py"], env, log, "bootstrap",
                        artifacts=(out / "bootstrap_summary.json",)):
            rec["status"] = "bootstrap_failed"
            return rec
    paired = out / "paired_vs_control.json"
    if not paired.exists():
        if not run_step([PAIRED, "--a", out / "comparison_errors_test.npz",
                         "--b", control / "comparison_errors_test.npz", "--out", paired],
                        env, log, "paired-vs-control", artifacts=(paired,)):
            rec["status"] = "paired_failed"
            return rec
    m = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    rec.update({"epochs_run": m["epochs_run"], "best_epoch": m["best_epoch"],
                "cap_bound": m["epochs_run"] >= int(EPOCH_CAP), "minutes": round((time.time() - t0) / 60, 1)})
    return rec


def summarize(arms):
    """Per-arm, per-population table of paired MSE skill vs the control (sec. 4 inputs)."""
    summary = {"epoch_cap": int(EPOCH_CAP), "arms": {}}
    for arm in arms:
        for pop in POPS:
            rows = {}
            for s in SEEDS:
                out = arm_dir(arm, pop, s)
                pj, bj, mj = (out / "paired_vs_control.json", out / "bootstrap_summary.json", out / "metrics.json")
                if not (pj.exists() and bj.exists() and mj.exists()):
                    continue
                tg = json.loads(pj.read_text(encoding="utf-8"))["targets"]
                bs = json.loads(bj.read_text(encoding="utf-8"))["splits"]["test"]
                m = json.loads(mj.read_text(encoding="utf-8"))
                rows[str(s)] = {
                    t: {"paired_vs_control": tg[t]["skill_point"], "ci95": tg[t]["skill_ci95"],
                        "arm_better": tg[t]["a_better"], "control_better": tg[t]["b_better"],
                        "skill_vs_pchip": bs[t]["pchip"]["skill_point"], "pchip_pass": bs[t]["pchip"]["pass"],
                        "skill_vs_gp_causal": bs[t].get("gp_causal", {}).get("skill_point")}
                    for t in ("CES_TI", "CES_VT")}
                rows[str(s)]["epochs"] = [m["epochs_run"], m["best_epoch"]]
            summary["arms"].setdefault(arm, {})[pop] = rows
    (DATA / ".b14_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    for arm, pops in summary["arms"].items():
        for pop, rows in pops.items():
            for t in ("CES_TI", "CES_VT"):
                vals = [rows[s][t]["paired_vs_control"] for s in rows]
                sig_up = sum(rows[s][t]["arm_better"] for s in rows)
                sig_dn = sum(rows[s][t]["control_better"] for s in rows)
                print(f"[b14] {arm:5s} {pop:4s} {t}: paired vs control "
                      + " ".join(f"{v:+.3f}" for v in vals)
                      + f"  (arm sig {sig_up}/{len(vals)}, control sig {sig_dn}/{len(vals)})")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--arms", nargs="*", default=list(ARMS))
    ap.add_argument("--pops", nargs="*", default=list(POPS))
    args = ap.parse_args()
    if not any(DATA.glob("s*.csv")):
        raise SystemExit("FATAL: no shot CSVs in data/ -- real data required.")
    recs = []
    for pop in args.pops:
        for s in ((42,) if args.smoke else SEEDS):
            for arm in args.arms:
                recs.append(one_run(arm, pop, s, args.smoke))
    bad = [r for r in recs if r["status"] != "ok"]
    for r in bad:
        print(f"[b14] FAILED: {r['out']} ({r['status']})", flush=True)
    if args.smoke or bad:
        sys.exit(1 if bad else 0)
    summarize(args.arms)


if __name__ == "__main__":
    main()
