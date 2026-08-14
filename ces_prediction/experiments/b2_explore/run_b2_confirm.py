"""B.2 confirmatory TEST run for a promoted candidate (PREREGISTRATION_W2.md sec. 6).

Runs the candidate on all four splits (init = split seed) under the confirmed
protocol, scores TEST once, and applies the pre-committed decision rule:
backbone promotion (paired vs the B.1 seq_v2 backbone) and claim-2
reinstatement (vs gp_causal) are reported as two independent verdicts.

Usage (repo root):
  py ces_prediction/experiments/b2_explore/run_b2_confirm.py --variant v3 [--resume]
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
CES_DIR = REPO_ROOT / "ces_prediction"
SEQ_DIR = CES_DIR / "experiments" / "seq"
PAIRED = CES_DIR / "experiments" / "paired_model_compare.py"
DATA = REPO_ROOT / "data"
sys.path.insert(0, str(CES_DIR / "experiments" / "b1_gate"))
from run_b1_gate import GATE_ENV, AMBIENT_VARS, run_step  # noqa: E402

SEEDS = (42, 1, 7, 123)


def one_run(variant, seed, resume):
    out_dir = DATA / f".b2c_{variant}_s{seed}"
    control_out = DATA / f".b1_seqv2_s{seed}_i{seed}"
    split_dir = DATA / f".b1_w2cut_split_s{seed}"
    paired_json = out_dir / "paired_vs_seqv2.json"
    if resume and paired_json.exists():
        print(f"[b2c] === {out_dir.name}: complete, skipping (--resume)", flush=True)
        return {"seed": seed, "out_dir": str(out_dir), "status": "ok", "resumed": True}
    if not (control_out / "comparison_errors_test.npz").exists():
        raise SystemExit(f"FATAL: backbone control missing: {control_out}")
    out_dir.mkdir(parents=True, exist_ok=True)
    log = out_dir / "run.log"

    env = os.environ.copy()
    for var in AMBIENT_VARS:
        env.pop(var, None)
    env.update(GATE_ENV)
    env.update({
        "CES_SEED": str(seed),
        "CES_INIT_SEED": str(seed),
        "CES_SPLIT_DIR": str(split_dir),
        "CES_OUTPUT_DIR": str(out_dir),
        "CES_SPLIT_TAG": "test",
        "CES_CONTROL_METRICS": str((DATA / f".b1_w2cut_s{seed}") / "metrics.json"),
        "CES_SEQ_MODEL": variant,
        "CES_PER_SHOT_NORM": "1",
    })

    record = {"seed": seed, "out_dir": str(out_dir), "status": "ok"}
    start = time.time()
    print(f"[b2c] === {out_dir.name}", flush=True)
    try:
        if not run_step([SEQ_DIR / "train_seq.py"], env, log, "train",
                        artifacts=(out_dir / "weights" / "seq_lstm.pth", out_dir / "metrics.json")):
            record["status"] = "train_failed"
            return record
        if not run_step([SEQ_DIR / "eval_seq.py"], env, log, "eval(test)",
                        artifacts=(out_dir / "comparison_errors_test.npz",
                                   out_dir / "comparison_metrics.json")):
            record["status"] = "compare_failed"
            return record
        if not run_step([CES_DIR / "bootstrap_compare.py"], env, log, "bootstrap",
                        artifacts=(out_dir / "bootstrap_summary.json",)):
            record["status"] = "bootstrap_failed"
            return record
        if not run_step([PAIRED,
                         "--a", out_dir / "comparison_errors_test.npz",
                         "--b", control_out / "comparison_errors_test.npz",
                         "--out", paired_json],
                        env, log, "paired-vs-seqv2", artifacts=(paired_json,)):
            record["status"] = "paired_failed"
    finally:
        record["minutes"] = round((time.time() - start) / 60.0, 1)
    return record


def summarize(variant, records):
    rows, verdict = [], {}
    pos = sig = vt_deficit = gpc_pass = 0
    for rec in sorted(records, key=lambda r: SEEDS.index(r["seed"])):
        out_dir = Path(rec["out_dir"])
        pr = json.loads((out_dir / "paired_vs_seqv2.json").read_text(encoding="utf-8"))["targets"]
        bs = json.loads((out_dir / "bootstrap_summary.json").read_text(encoding="utf-8"))
        row = {"seed": rec["seed"], "minutes": rec.get("minutes")}
        ti, vt = pr["CES_TI"], pr["CES_VT"]
        row["paired_TI"] = ti["skill_point"]
        row["paired_TI_ci"] = ti["skill_ci95"]
        row["paired_TI_sig"] = ti["a_better"]
        row["paired_VT"] = vt["skill_point"]
        row["paired_VT_deficit"] = vt["b_better"]
        pos += int(ti["skill_point"] > 0)
        sig += int(ti["a_better"])
        vt_deficit += int(vt["b_better"])
        for t in ("CES_TI", "CES_VT"):
            g = bs["splits"]["test"][t].get("gp_causal")
            if g:
                row[f"vs_gp_causal_{t[-2:]}"] = g["skill_point"]
                row[f"vs_gp_causal_{t[-2:]}_pass"] = g["pass"]
        gpc_pass += int(bs["splits"]["test"]["CES_TI"].get("gp_causal", {}).get("pass", False))
        rows.append(row)
    verdict = {
        "backbone_promotion": {
            "ti_positive": f"{pos}/4", "ti_significant": f"{sig}/4",
            "vt_significant_deficit": f"{vt_deficit}/4",
            "promote": bool(pos == 4 and sig >= 3 and vt_deficit == 0),
        },
        "claim2_reinstatement": {
            "ti_vs_gp_causal_pass": f"{gpc_pass}/4",
            "reinstate": bool(gpc_pass == 4),
        },
    }
    out = {"variant": variant, "rows": rows, "verdict": verdict}
    out_path = DATA / f".b2c_{variant}_summary.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    records = []
    start = time.time()
    for seed in SEEDS:
        records.append(one_run(args.variant, seed, args.resume))
    print(f"[b2c] wall: {(time.time() - start) / 60.0:.0f} min", flush=True)
    bad = [r for r in records if r["status"] != "ok"]
    for r in bad:
        print(f"[b2c] FAILED: {r['out_dir']} ({r['status']})", flush=True)
    if bad:
        sys.exit(1)
    summarize(args.variant, records)
    sys.exit(0)


if __name__ == "__main__":
    main()
