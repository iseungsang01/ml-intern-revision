"""B.16 sec. 10 row (b): the pure learned anchor, residual removed -- inference only.

For every frozen `v4kng` checkpoint (.b16_v4kng_{pop}_s{seed}) this scores the SAME weights with
CES_SEQ_ANCHOR_ONLY=1, i.e. y_hat = a(t), the trained scalar OU-Kalman filter per target with the
LSTM residual set to zero. No training. It is the P02 estimator of sec. 8bb (history-only OU
filter) with the parameters the batch learned, and it asks whether the 0.921x V_rot median bound
is reachable by the filter alone.

Steps per (pop, seed): copy metrics.json + weights into .b16_anchor_{pop}_s{seed}, eval_seq.py
on TEST, bootstrap ladder, paired vs the backbone and paired vs v4kng. Resumable.

Usage (repo root):  py ces_prediction/experiments/b16_physics/anchor_readout_b16.py
"""

import json
import os
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run_b16 import (DATA, CES_DIR, SEQ_DIR, PAIRED, POPS, SEEDS, base_env, run_step, shot_winrate)  # noqa: E402


def main():
    recs = []
    for pop in POPS:
        p = POPS[pop]
        for s in SEEDS:
            src = DATA / f".b16_v4kng_{pop}_s{s}"
            out = DATA / f".b16_anchor_{pop}_s{s}"
            control = DATA / p["control"].format(s=s)
            split = DATA / p["split"].format(s=s)
            cm = DATA / p["control_metrics"].format(s=s) / "metrics.json"
            for needed in (src / "weights" / "seq_lstm.pth", src / "metrics.json", control / "comparison_errors_test.npz", cm):
                if not needed.exists():
                    raise SystemExit(f"FATAL: missing {needed}")
            out.mkdir(parents=True, exist_ok=True)
            (out / "weights").mkdir(exist_ok=True)
            if not (out / "weights" / "seq_lstm.pth").exists():
                shutil.copy2(src / "weights" / "seq_lstm.pth", out / "weights" / "seq_lstm.pth")
                shutil.copy2(src / "metrics.json", out / "metrics.json")
            env = base_env(pop, "v4kng", False)
            env.update({"CES_SEED": str(s), "CES_INIT_SEED": str(s), "CES_SPLIT_DIR": str(split),
                        "CES_OUTPUT_DIR": str(out), "CES_CONTROL_METRICS": str(cm), "CES_SPLIT_TAG": "test",
                        "CES_SEQ_ANCHOR_ONLY": "1"})
            log = out / "run.log"
            print(f"[b16-anchor] === {out.name}", flush=True)
            rec = {"pop": pop, "seed": s, "status": "ok"}
            if not (out / "comparison_errors_test.npz").exists():
                if not run_step([SEQ_DIR / "eval_seq.py"], env, log, "eval(test, anchor only)",
                                artifacts=(out / "comparison_errors_test.npz", out / "comparison_metrics.json")):
                    rec["status"] = "compare_failed"; recs.append(rec); continue
            if not (out / "bootstrap_summary.json").exists():
                if not run_step([CES_DIR / "bootstrap_compare.py"], env, log, "bootstrap", artifacts=(out / "bootstrap_summary.json",)):
                    rec["status"] = "bootstrap_failed"; recs.append(rec); continue
            for tag, other in (("control", control), ("v4kng", src)):
                paired = out / f"paired_vs_{tag}.json"
                if not paired.exists():
                    if not run_step([PAIRED, "--a", out / "comparison_errors_test.npz", "--b", other / "comparison_errors_test.npz",
                                     "--out", paired], env, log, f"paired-vs-{tag}", artifacts=(paired,)):
                        rec["status"] = f"paired_{tag}_failed"; break
            if not (out / "shot_winrate.json").exists():
                shot_winrate(out)
            recs.append(rec)
    bad = [r for r in recs if r["status"] != "ok"]
    for r in bad:
        print(f"[b16-anchor] FAILED: {r}")
    if bad:
        sys.exit(1)
    summary = {}
    for pop in POPS:
        for s in SEEDS:
            out = DATA / f".b16_anchor_{pop}_s{s}"
            pc = json.loads((out / "paired_vs_control.json").read_text(encoding="utf-8"))["targets"]
            pv = json.loads((out / "paired_vs_v4kng.json").read_text(encoding="utf-8"))["targets"]
            bs = json.loads((out / "bootstrap_summary.json").read_text(encoding="utf-8"))["splits"]["test"]
            summary.setdefault(pop, {})[str(s)] = {
                t: {"vs_backbone": pc[t]["skill_point"], "vs_backbone_ci": pc[t]["skill_ci95"],
                    "vs_v4kng": pv[t]["skill_point"], "vs_v4kng_ci": pv[t]["skill_ci95"],
                    "vs_pchip": bs[t]["pchip"]["skill_point"], "vs_persistence": bs[t]["persistence"]["skill_point"],
                    "vs_gp_causal": bs[t].get("gp_causal", {}).get("skill_point")}
                for t in ("CES_TI", "CES_VT")}
            summary[pop][str(s)]["shot_winrate"] = json.loads((out / "shot_winrate.json").read_text(encoding="utf-8"))
    (DATA / ".b16_anchor_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    for pop in POPS:
        for t in ("CES_TI", "CES_VT"):
            print(f"[b16-anchor] {pop:4s} {t}: MSE skill vs backbone " + " ".join(f"{summary[pop][str(s)][t]['vs_backbone']:+.3f}" for s in SEEDS)
                  + " | vs persistence " + " ".join(f"{summary[pop][str(s)][t]['vs_persistence']:+.3f}" for s in SEEDS))
    print("[b16-anchor] M1 / median-vs-persistence: rescore_b13.py --extra anchor=.b16_anchor_{pop}_s{s} --out data/.b16_anchor_rescore.json")


if __name__ == "__main__":
    main()
