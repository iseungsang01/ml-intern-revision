"""B.16 runner -- PREREGISTRATION_B16.md: the physics-structured output (Kalman anchor + gated residual).

Arms `v4k` / `v4kng` (CES_SEQ_MODEL), split {42, 1, 7, 123} x init = split seed, both
populations. Everything but the output structure is the B.1 stage-B env (cut) or the B.5
inclusive-backbone env (incl): same manifests, same control metrics, per-shot ON,
held-free, MSE loss, val-MSE selection, patience 6, epoch cap 60 (reported if bound).

Control = the frozen backbone runs (`.b1_seqv2_s{s}_i{s}`, `.b5i_seqv2_s{s}`): no
retraining. Each arm is paired row-for-row against its control (paired_model_compare) and
scored on the PR4 ladder (bootstrap_compare); the robust co-metrics come from
experiments/b13_robust/rescore_b13.py afterwards (`--extra v4k=.b16_v4k_{pop}_s{s} ...`).
The learned physics parameters (tau, sigma_n, sigma_s, alpha) are written to
`physics_params.json` next to each run's metrics, and the per-discharge win rate by
target-spread tercile (the P08 readout) to `shot_winrate.json`.

Usage (repo root):  py ces_prediction/experiments/b16_physics/run_b16.py [--smoke] [--arms v4k v4kng] [--pops cut incl]
Resumable: finished steps are skipped on re-run.
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
from run_b1_gate import GATE_ENV, AMBIENT_VARS, SMOKE_OVERRIDES, run_step  # noqa: E402

SEEDS = (42, 1, 7, 123)
ARMS = ("v4k", "v4kng")
EPOCH_CAP = "60"
POPS = {
    "cut": {"cut": "3000", "control": ".b1_seqv2_s{s}_i{s}", "control_metrics": ".b1_w2cut_s{s}",
            "split": ".b1_w2cut_split_s{s}"},
    "incl": {"cut": "0", "control": ".b5i_seqv2_s{s}", "control_metrics": ".wsweep_hf_w2_s{s}",
             "split": ".b1_manifest_s{s}"},
}


SELECT = "mse"  # B.20: --select medae -> CES_SEQ_SELECT=medae, run dirs .b20_{arm}_*


def arm_dir(arm, pop, s, smoke=False):
    prefix = ".b20" if SELECT == "medae" else ".b16"
    return DATA / (f"{prefix}_{arm}_{pop}_s{s}" + ("_smoke" if smoke else ""))


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
        "CES_SEQ_LOSS": "mse",
        "CES_SEQ_SELECT": SELECT,
        "CES_SEQ_EPOCHS": EPOCH_CAP,
        "CES_SEQ_MODEL": arm,
        "CES_PER_SHOT_NORM": "1",
        "CES_SEQ_DEVICE": "cuda",
    })
    if smoke:
        env.update({"CES_SEQ_EPOCHS": "2", "CES_SEQ_MAX_FILES": "40"})
    return env


def write_physics(out, arm):
    """Reload the checkpoint and write the learned Kalman parameters in physical units."""
    sys.path.insert(0, str(SEQ_DIR))
    import torch  # noqa: E402
    from seq_models import SEQ_MODELS  # noqa: E402
    m = SEQ_MODELS[arm]()
    m.load_state_dict(torch.load(out / "weights" / "seq_lstm.pth", map_location="cpu"))
    # metrics.json: {"normalization": {"stats": {"target": {"mean": [..], "std": [..]}, ...}}}
    norm = json.loads((out / "metrics.json").read_text(encoding="utf-8")).get("normalization", {})
    std = norm.get("stats", {}).get("target", {}).get("std") if isinstance(norm, dict) else None
    if not std:
        raise SystemExit(f"FATAL: {out / 'metrics.json'} carries no target std -- cannot report physical units")
    phys = m.physics(target_std=std)
    phys["target_std_used"] = list(std)
    (out / "physics_params.json").write_text(json.dumps(phys, indent=2), encoding="utf-8")
    return phys


def shot_winrate(out):
    """Per-discharge win rate vs the causal GP by tercile of the target's spread (P08)."""
    z = np.load(out / "comparison_errors_test.npz")
    res = {}
    for t in ("CES_TI", "CES_VT"):
        shot, y = z[f"{t}_shot"], z[f"{t}_y_true"].astype(float)
        sm, sg = z[f"{t}_se_model"].astype(float), z[f"{t}_se_gp_causal"].astype(float)
        rows = []
        for s in np.unique(shot):
            k = shot == s
            if k.sum() < 20:
                continue
            rows.append((float(np.std(y[k])), float(sm[k].mean() < sg[k].mean()),
                         float(1 - sm[k].sum() / sg[k].sum())))
        if len(rows) < 6:
            res[t] = {"n_shots": len(rows)}
            continue
        rows.sort(key=lambda r: r[0])
        n = len(rows)
        terc = [rows[: n // 3], rows[n // 3: 2 * n // 3], rows[2 * n // 3:]]
        res[t] = {"n_shots": n,
                  "win_rate_quiet_mid_variable": [float(np.mean([r[1] for r in g])) for g in terc],
                  "median_skill_quiet_mid_variable": [float(np.median([r[2] for r in g])) for g in terc],
                  "win_rate_all": float(np.mean([r[1] for r in rows]))}
    (out / "shot_winrate.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    return res


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
    print(f"[b16] === {out.name}", flush=True)
    if not (out / "weights" / "seq_lstm.pth").exists() or not (out / "metrics.json").exists():
        if not run_step([SEQ_DIR / "train_seq.py"], env, log, "train",
                        artifacts=(out / "weights" / "seq_lstm.pth", out / "metrics.json")):
            rec["status"] = "train_failed"
            return rec
    if not (out / "physics_params.json").exists():
        rec["physics"] = write_physics(out, arm)
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
    if not (out / "shot_winrate.json").exists():
        shot_winrate(out)
    m = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    rec.update({"epochs_run": m["epochs_run"], "best_epoch": m["best_epoch"],
                "cap_bound": m["epochs_run"] >= int(EPOCH_CAP), "minutes": round((time.time() - t0) / 60, 1)})
    return rec


SUMMARY_NAME = ".b16_summary.json"


def summarize(arms):
    summary = {"epoch_cap": int(EPOCH_CAP), "select": SELECT, "arms": {}}
    for arm in arms:
        for pop in POPS:
            rows = {}
            for s in SEEDS:
                out = arm_dir(arm, pop, s)
                need = [out / n for n in ("paired_vs_control.json", "bootstrap_summary.json",
                                           "metrics.json", "physics_params.json", "shot_winrate.json")]
                if not all(n.exists() for n in need):
                    continue
                tg = json.loads(need[0].read_text(encoding="utf-8"))["targets"]
                bs = json.loads(need[1].read_text(encoding="utf-8"))["splits"]["test"]
                m = json.loads(need[2].read_text(encoding="utf-8"))
                rows[str(s)] = {
                    t: {"paired_vs_control": tg[t]["skill_point"], "ci95": tg[t]["skill_ci95"],
                        "arm_better": tg[t]["a_better"], "control_better": tg[t]["b_better"],
                        "skill_vs_pchip": bs[t]["pchip"]["skill_point"], "pchip_pass": bs[t]["pchip"]["pass"],
                        "skill_vs_gp_causal": bs[t].get("gp_causal", {}).get("skill_point")}
                    for t in ("CES_TI", "CES_VT")}
                rows[str(s)]["epochs"] = [m["epochs_run"], m["best_epoch"]]
                rows[str(s)]["physics"] = json.loads(need[3].read_text(encoding="utf-8"))
                rows[str(s)]["shot_winrate"] = json.loads(need[4].read_text(encoding="utf-8"))
            summary["arms"].setdefault(arm, {})[pop] = rows
    (DATA / SUMMARY_NAME).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    for arm, pops in summary["arms"].items():
        for pop, rows in pops.items():
            for t in ("CES_TI", "CES_VT"):
                vals = [rows[s][t]["paired_vs_control"] for s in rows]
                sig_up = sum(rows[s][t]["arm_better"] for s in rows)
                sig_dn = sum(rows[s][t]["control_better"] for s in rows)
                print(f"[b16] {arm:6s} {pop:4s} {t}: paired MSE vs control "
                      + " ".join(f"{v:+.3f}" for v in vals)
                      + f"  (arm sig {sig_up}/{len(vals)}, control sig {sig_dn}/{len(vals)})")
            key, unit = ("tau_ms", "tau [ms]") if "tau_ms" in rows[next(iter(rows))]["physics"]["CES_TI"] else ("jump_rate_hz", "jump rate [Hz]")
            vals = [rows[s]["physics"]["CES_TI"][key] for s in rows]
            print(f"[b16] {arm:6s} {pop:4s} learned T_i {unit}: " + " ".join(f"{v:.1f}" for v in vals))
    return summary


def main():
    global SUMMARY_NAME, SELECT
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--arms", nargs="*", default=list(ARMS))
    ap.add_argument("--summary-name", default=SUMMARY_NAME,
                    help="sec. 10 sensitivity rows write .b16s_summary.json, never the frozen .b16_summary.json")
    ap.add_argument("--pops", nargs="*", default=list(POPS))
    ap.add_argument("--select", choices=("mse", "medae"), default="mse",
                    help="B.20: checkpoint / early stop / LR plateau on val median AE (dirs .b20_*)")
    args = ap.parse_args()
    SUMMARY_NAME = args.summary_name
    SELECT = args.select
    if SELECT != "mse" and SUMMARY_NAME.startswith(".b16"):
        raise SystemExit("FATAL: --select medae needs its own --summary-name (e.g. .b20_summary.json)")
    if set(args.arms) - set(ARMS) and SUMMARY_NAME == ".b16_summary.json":
        raise SystemExit("FATAL: sensitivity arms must use --summary-name .b16s_summary.json (the frozen B.16 summary is not overwritten)")
    if not any(DATA.glob("s*.csv")):
        raise SystemExit("FATAL: no shot CSVs in data/ -- real data required.")
    recs = []
    for pop in args.pops:
        for s in ((42,) if args.smoke else SEEDS):
            for arm in args.arms:
                recs.append(one_run(arm, pop, s, args.smoke))
    bad = [r for r in recs if r["status"] != "ok"]
    for r in bad:
        print(f"[b16] FAILED: {r['out']} ({r['status']})", flush=True)
    if args.smoke or bad:
        for r in recs:
            print(json.dumps({k: v for k, v in r.items() if k != "physics"}))
        sys.exit(1 if bad else 0)
    summarize(args.arms)


if __name__ == "__main__":
    main()
