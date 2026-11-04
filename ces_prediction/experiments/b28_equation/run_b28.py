"""B.28 runner -- PREREGISTRATION_B28.md: does the two-term equation read off the frozen seq_v2
explain the backbone?

Stage 1 (CPU, no training): `stage1_score.py` -- the EWMA / EWMA + increment surrogates scored
on the frozen TEST rows of the four cut and four inclusive backbone runs.
Stage 2 (GPU, 16 runs): the equation made trainable (`CES_SEQ_MODEL=b28`, 20 T_i parameters,
V_rot branch = seq_v2's) and the single cross-term variant (`b28x`, +1 parameter), split
{42, 1, 7, 123} x init = split seed, both populations. Everything but the T_i computation is
the B.1 stage-B env (cut) or the B.5 inclusive env (incl): same manifests, same control
metrics, per-shot ON, held-free, MSE loss, val-MSE selection, patience 6, epoch cap 60.
Control = the frozen backbone (`.b1_seqv2_s{s}_i{s}`, `.b5i_seqv2_s{s}`): no retraining. Each
arm is paired row-for-row against its control and against the other arm (paired_model_compare)
and scored on the PR4 ladder (bootstrap_compare). The learned equation parameters (K_0, K_inf,
s_K, a_c in eV per sigma, kappa) are written to physics_params.json beside each run.

Usage (repo root):
  py ces_prediction/experiments/b28_equation/run_b28.py --stage 1|2|all [--smoke] [--resume]
Resumable: finished steps are skipped on re-run (--resume is accepted for symmetry; every
step checks its own artifact). GPU batches: one at a time.
"""

import argparse
import ctypes
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
HERE = Path(__file__).resolve().parent
DATA = REPO_ROOT / "data"
sys.path.insert(0, str(CES_DIR / "experiments" / "b1_gate"))
from run_b1_gate import GATE_ENV, AMBIENT_VARS, SMOKE_OVERRIDES, run_step  # noqa: E402

SEEDS = (42, 1, 7, 123)
ARMS = ("b28", "b28x")
EPOCH_CAP = "60"
POPS = {
    "cut": {"cut": "3000", "control": ".b1_seqv2_s{s}_i{s}", "control_metrics": ".b1_w2cut_s{s}",
            "split": ".b1_w2cut_split_s{s}"},
    "incl": {"cut": "0", "control": ".b5i_seqv2_s{s}", "control_metrics": ".wsweep_hf_w2_s{s}",
             "split": ".b1_manifest_s{s}"},
}
STAGE2_OUT = DATA / ".b28_stage2.json"


def arm_dir(arm, pop, s, smoke=False):
    return DATA / (f".b28_{arm}_{pop}_s{s}" + ("_smoke" if smoke else ""))


def base_env(pop, arm, smoke):
    env = os.environ.copy()
    for var in AMBIENT_VARS:
        env.pop(var, None)
    env.update(GATE_ENV)
    if smoke:
        env.update(SMOKE_OVERRIDES)
    env.update({
        "CES_TI_SPIKE_CUT_EV": POPS[pop]["cut"],
        "CES_SEQ_LOSS": "mse",
        "CES_SEQ_SELECT": "mse",
        "CES_SEQ_EPOCHS": EPOCH_CAP,
        "CES_SEQ_MODEL": arm,
        "CES_PER_SHOT_NORM": "1",
        "CES_SEQ_DEVICE": "cuda",
    })
    if smoke:
        env.update({"CES_SEQ_EPOCHS": "2", "CES_SEQ_MAX_FILES": "40"})
    return env


def write_physics(out, arm):
    sys.path.insert(0, str(SEQ_DIR))
    import torch  # noqa: E402
    from seq_models import SEQ_MODELS  # noqa: E402
    m = SEQ_MODELS[arm]()
    m.load_state_dict(torch.load(out / "weights" / "seq_lstm.pth", map_location="cpu"))
    std = json.loads((out / "metrics.json").read_text(encoding="utf-8"))["normalization"]["stats"]["target"]["std"]
    phys = m.physics(target_std=std)
    phys["target_std_used"] = list(std)
    (out / "physics_params.json").write_text(json.dumps(phys, indent=2), encoding="utf-8")
    return phys


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
    print(f"[b28] === {out.name}", flush=True)
    if not (out / "weights" / "seq_lstm.pth").exists() or not (out / "metrics.json").exists():
        if not run_step([SEQ_DIR / "train_seq.py"], env, log, "train",
                        artifacts=(out / "weights" / "seq_lstm.pth", out / "metrics.json")):
            rec["status"] = "train_failed"
            return rec
    if not (out / "physics_params.json").exists():
        write_physics(out, arm)
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
    if arm == "b28x":
        other = arm_dir("b28", pop, s) / "comparison_errors_test.npz"
        paired_x = out / "paired_b28x_vs_b28.json"
        if other.exists() and not paired_x.exists():
            if not run_step([PAIRED, "--a", out / "comparison_errors_test.npz", "--b", other, "--out", paired_x],
                            env, log, "paired-b28x-vs-b28", artifacts=(paired_x,)):
                rec["status"] = "paired_x_failed"
                return rec
    m = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    rec.update({"epochs_run": m["epochs_run"], "best_epoch": m["best_epoch"],
                "cap_bound": m["epochs_run"] >= int(EPOCH_CAP), "minutes": round((time.time() - t0) / 60, 1)})
    return rec


def summarize():
    """PREREGISTRATION_B28.md sec. 4 rules on whatever runs are finished."""
    summary = {"epoch_cap": int(EPOCH_CAP), "arms": {}, "verdict": {}}
    for arm in ARMS:
        summary["arms"][arm] = {}
        for pop in POPS:
            rows = {}
            for s in SEEDS:
                out = arm_dir(arm, pop, s)
                control = DATA / POPS[pop]["control"].format(s=s)
                need = [out / n for n in ("paired_vs_control.json", "bootstrap_summary.json", "metrics.json",
                                           "physics_params.json")] + [control / "bootstrap_summary.json"]
                if not all(n.exists() for n in need):
                    continue
                tg = json.loads(need[0].read_text(encoding="utf-8"))["targets"]
                bs = json.loads(need[1].read_text(encoding="utf-8"))["splits"]["test"]
                m = json.loads(need[2].read_text(encoding="utf-8"))
                phys = json.loads(need[3].read_text(encoding="utf-8"))
                cb = json.loads(need[4].read_text(encoding="utf-8"))["splits"]["test"]
                row = {}
                for t in ("CES_TI", "CES_VT"):
                    arm_sk = {ref: bs[t][ref]["skill_point"] for ref in ("pchip", "persistence", "gp_causal", "gp") if ref in bs[t]}
                    ctl_sk = {ref: cb[t][ref]["skill_point"] for ref in ("pchip", "persistence", "gp_causal", "gp") if ref in cb[t]}
                    row[t] = {"paired_vs_control": tg[t]["skill_point"], "ci95": tg[t]["skill_ci95"],
                              "arm_better": tg[t]["a_better"], "control_better": tg[t]["b_better"],
                              "skill": arm_sk, "control_skill": ctl_sk,
                              "pass": {ref: bs[t][ref]["pass"] for ref in arm_sk},
                              "ratio_vs_pchip": (arm_sk["pchip"] / ctl_sk["pchip"]) if ctl_sk.get("pchip") else None,
                              "ratio_vs_persistence": (arm_sk["persistence"] / ctl_sk["persistence"]) if ctl_sk.get("persistence") else None}
                row["epochs"] = [m["epochs_run"], m["best_epoch"]]
                row["physics"] = phys
                px = out / "paired_b28x_vs_b28.json"
                if px.exists():
                    tx = json.loads(px.read_text(encoding="utf-8"))["targets"]["CES_TI"]
                    row["b28x_vs_b28_TI"] = {"skill_point": tx["skill_point"], "ci95": tx["skill_ci95"],
                                             "x_better": tx["a_better"], "b28_better": tx["b_better"]}
                rows[str(s)] = row
            summary["arms"][arm][pop] = rows
    # rules
    for pop in POPS:
        r28 = summary["arms"]["b28"].get(pop, {})
        rx = summary["arms"]["b28x"].get(pop, {})
        ratios = [r28[s]["CES_TI"]["ratio_vs_pchip"] for s in r28]
        v = {"n_runs": len(r28), "H28_3_ratio_vs_pchip": ratios,
             "H28_3_pass_count_ge_0.70": int(sum(x is not None and x >= 0.70 for x in ratios)),
             "H28_3_mean_ratio": float(np.mean([x for x in ratios if x is not None])) if ratios else None,
             "vt_significant_deficit_runs": int(sum(r28[s]["CES_VT"]["control_better"] for s in r28)),
             "ti_control_significantly_better_runs": int(sum(r28[s]["CES_TI"]["control_better"] for s in r28)),
             "a_sum_BES_eV": [r28[s]["physics"]["a_sum_BES_eV"] for s in r28],
             "a_sum_ECEI_eV": [r28[s]["physics"]["a_sum_ECEI_eV"] for s in r28],
             "a_sum_MC_eV": [r28[s]["physics"]["a_sum_MC_eV"] for s in r28],
             "K0": [r28[s]["physics"]["K0"] for s in r28], "K_inf": [r28[s]["physics"]["K_inf"] for s in r28],
             "s_K_ms": [r28[s]["physics"]["s_K_ms"] for s in r28]}
        if len(r28) == 4:
            mean_ratio = v["H28_3_mean_ratio"]
            v["H28_3_verdict"] = ("equation explains the model (>= 0.70 on 4/4)" if v["H28_3_pass_count_ge_0.70"] == 4
                                  else "event-row nonlinear term needed (mean 0.50-0.70)" if 0.50 <= mean_ratio < 0.70
                                  else "equation discarded (mean < 0.50)" if mean_ratio < 0.50
                                  else "mean >= 0.70 but not 4/4: not supported by the registered rule")
        xs = [rx[s].get("b28x_vs_b28_TI") for s in rx if rx[s].get("b28x_vs_b28_TI")]
        v["H28_4_cross_vs_b28_TI"] = [x["skill_point"] for x in xs]
        v["H28_4_cross_significant_better_runs"] = int(sum(x["x_better"] for x in xs))
        if len(xs) == 4:
            v["H28_4_verdict"] = ("cross term named (significant on >= 3/4)" if v["H28_4_cross_significant_better_runs"] >= 3
                                  else "cross term null")
        summary["verdict"][pop] = v
    STAGE2_OUT.write_text(json.dumps(summary, indent=1), encoding="utf-8")
    return summary


def keep_awake(on):
    """Windows: hold the machine out of standby for the batch (B.27 r0 f1 lost 22 min to it)."""
    if sys.platform != "win32":
        return
    ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
    ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | (ES_SYSTEM_REQUIRED if on else 0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=["1", "2", "all"])
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--resume", action="store_true", help="accepted; every step is artifact-checked anyway")
    ap.add_argument("--arms", nargs="*", default=list(ARMS))
    ap.add_argument("--pops", nargs="*", default=list(POPS))
    ap.add_argument("--seeds", nargs="*", type=int, default=list(SEEDS))
    args = ap.parse_args()
    t0 = time.time()
    if args.stage in ("1", "all"):
        cmd = [sys.executable, str(HERE / "stage1_score.py")] + (["--smoke"] if args.smoke else [])
        print(f"[b28] stage 1: {' '.join(cmd)}", flush=True)
        rc = subprocess.run(cmd, cwd=REPO_ROOT).returncode
        if rc != 0:
            raise SystemExit(f"FATAL: stage 1 failed rc={rc}")
    if args.stage in ("2", "all"):
        keep_awake(True)
        records = []
        try:
            for pop in args.pops:
                for s in args.seeds:
                    for arm in args.arms:       # b28 before b28x so the arm-vs-arm pairing can run
                        records.append(one_run(arm, pop, s, args.smoke))
                        print(f"[b28] {records[-1]}", flush=True)
                        if not args.smoke:
                            summarize()
        finally:
            keep_awake(False)
        if not args.smoke:
            summary = summarize()
            print(json.dumps(summary["verdict"], indent=1), flush=True)
        (DATA / (".b28_runs_smoke.json" if args.smoke else ".b28_runs.json")).write_text(
            json.dumps(records, indent=1), encoding="utf-8")
    print(f"[b28] done in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
