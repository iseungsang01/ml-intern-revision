"""B.3 confirmatory TEST run (PREREGISTRATION_W2.md sec. 6, B.3 -- rule committed first).

Runs the selected minimal interpretable variant on all four splits (init = split
seed) under the confirmed protocol, scores TEST once, additively TEST-scores the
retrained anchor family, and applies the pre-committed decision rule:

  cond1  interpretable rung:   paired `T_i` (b3 - anchor) significant win 4/4
  cond1b routing not degraded: paired `V_rot` vs anchor significant deficit 0/4
  cond2  backbone tolerance:   mean over 4 splits of paired `T_i` (b3 - seq_v2)
                               >= -0.05
  reported, not gated: vs w2cut window family, vs gp_causal, skill_vs_pchip,
  probe R^2 / decomposition / structural routing check (probe_b3.py).

The runner refuses to start until the rule text exists in PREREGISTRATION_W2.md.

Usage (repo root):
  py ces_prediction/experiments/b3_interp/run_b3_confirm.py --variant b3k8 [--resume]
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
sys.path.insert(0, str(CES_DIR / "experiments" / "b3_interp"))
from run_b1_gate import GATE_ENV, AMBIENT_VARS, run_step  # noqa: E402
from run_b3_anchor import one_test_score  # noqa: E402

SEEDS = (42, 1, 7, 123)
BACKBONE_TOLERANCE = -0.05   # pre-registered: mean paired TI (b3 - seq_v2) must be >= this


def one_run(variant, seed, resume):
    out_dir = DATA / f".b3c_{variant}_s{seed}"
    split_dir = DATA / f".b1_w2cut_split_s{seed}"
    control_metrics = DATA / f".b1_w2cut_s{seed}" / "metrics.json"
    pair_targets = {
        "paired_vs_anchor.json": DATA / f".b3_anchor_s{seed}" / "comparison_errors_test.npz",
        "paired_vs_seqv2.json": DATA / f".b1_seqv2_s{seed}_i{seed}" / "comparison_errors_test.npz",
        "paired_vs_w2cut.json": DATA / f".b1_w2cut_s{seed}" / "comparison_errors_test.npz",
    }
    if resume and all((out_dir / name).exists() for name in pair_targets):
        print(f"[b3c] === {out_dir.name}: complete, skipping (--resume)", flush=True)
        return {"seed": seed, "out_dir": str(out_dir), "status": "ok", "resumed": True}
    for name, npz in pair_targets.items():
        if not npz.exists():
            raise SystemExit(f"FATAL: pairing control missing: {npz}")
    out_dir.mkdir(parents=True, exist_ok=True)
    log = out_dir / "run.log"

    env = os.environ.copy()
    for var in AMBIENT_VARS + ("CES_MODEL_FILE",):
        env.pop(var, None)
    env.update(GATE_ENV)
    env.update({
        "CES_SEED": str(seed),
        "CES_INIT_SEED": str(seed),
        "CES_SPLIT_DIR": str(split_dir),
        "CES_OUTPUT_DIR": str(out_dir),
        "CES_SPLIT_TAG": "test",
        "CES_CONTROL_METRICS": str(control_metrics),
        "CES_SEQ_MODEL": variant,
        "CES_PER_SHOT_NORM": "1",
    })

    record = {"seed": seed, "out_dir": str(out_dir), "status": "ok"}
    start = time.time()
    print(f"[b3c] === {out_dir.name}", flush=True)
    try:
        if not run_step([SEQ_DIR / "train_seq.py"], env, log, "train",
                        artifacts=(out_dir / "weights" / "seq_lstm.pth",
                                   out_dir / "metrics.json")):
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
        for name, npz in pair_targets.items():
            if not run_step([PAIRED,
                             "--a", out_dir / "comparison_errors_test.npz",
                             "--b", npz,
                             "--out", out_dir / name],
                            env, log, name.replace(".json", ""),
                            artifacts=(out_dir / name,)):
                record["status"] = "paired_failed"
                return record
    finally:
        record["minutes"] = round((time.time() - start) / 60.0, 1)
    return record


def summarize(variant, records):
    rows = []
    anchor_win = anchor_vt_deficit = 0
    backbone_points = []
    w2cut_pos = gpc_pass = 0
    for rec in sorted(records, key=lambda r: SEEDS.index(r["seed"])):
        out_dir = Path(rec["out_dir"])
        pa = json.loads((out_dir / "paired_vs_anchor.json").read_text(encoding="utf-8"))["targets"]
        pb = json.loads((out_dir / "paired_vs_seqv2.json").read_text(encoding="utf-8"))["targets"]
        pw = json.loads((out_dir / "paired_vs_w2cut.json").read_text(encoding="utf-8"))["targets"]
        cm = json.loads((out_dir / "comparison_metrics.json").read_text(encoding="utf-8"))
        bs = json.loads((out_dir / "bootstrap_summary.json").read_text(encoding="utf-8"))
        row = {
            "seed": rec["seed"], "minutes": rec.get("minutes"),
            "skill_vs_pchip_TI": cm["per_target"]["CES_TI"]["skill_vs_pchip"],
            "vs_anchor_TI": pa["CES_TI"]["skill_point"],
            "vs_anchor_TI_ci": pa["CES_TI"]["skill_ci95"],
            "vs_anchor_TI_sig_win": pa["CES_TI"]["a_better"],
            "vs_anchor_VT": pa["CES_VT"]["skill_point"],
            "vs_anchor_VT_sig_deficit": pa["CES_VT"]["b_better"],
            "vs_backbone_TI": pb["CES_TI"]["skill_point"],
            "vs_backbone_TI_ci": pb["CES_TI"]["skill_ci95"],
            "vs_backbone_VT": pb["CES_VT"]["skill_point"],
            "vs_w2cut_TI": pw["CES_TI"]["skill_point"],
        }
        g = bs["splits"]["test"]["CES_TI"].get("gp_causal")
        if g:
            row["vs_gp_causal_TI"] = g["skill_point"]
            row["vs_gp_causal_TI_pass"] = g["pass"]
            gpc_pass += int(g["pass"])
        anchor_win += int(pa["CES_TI"]["a_better"])
        anchor_vt_deficit += int(pa["CES_VT"]["b_better"])
        backbone_points.append(pb["CES_TI"]["skill_point"])
        w2cut_pos += int(pw["CES_TI"]["skill_point"] > 0)
        rows.append(row)

    mean_backbone = sum(backbone_points) / len(backbone_points)
    verdict = {
        "cond1_anchor_sig_win_TI": f"{anchor_win}/4",
        "cond1b_anchor_VT_sig_deficit": f"{anchor_vt_deficit}/4",
        "cond2_mean_vs_backbone_TI": round(mean_backbone, 4),
        "cond2_tolerance": BACKBONE_TOLERANCE,
        "interpretable_rung": bool(anchor_win == 4 and anchor_vt_deficit == 0),
        "within_backbone_tolerance": bool(mean_backbone >= BACKBONE_TOLERANCE),
        "reported_vs_w2cut_TI_positive": f"{w2cut_pos}/4",
        "reported_vs_gp_causal_TI_pass": f"{gpc_pass}/4",
    }
    out = {"variant": variant, "rows": rows, "verdict": verdict}
    out_path = DATA / f".b3c_{variant}_summary.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    prereg = (CES_DIR / "experiments" / "PREREGISTRATION_W2.md").read_text(encoding="utf-8")
    if "B.3 확증 판정 규칙" not in prereg:
        raise SystemExit("FATAL: B.3 decision rule not committed to PREREGISTRATION_W2.md "
                         "-- TEST stays frozen.")

    start = time.time()
    anchor_records = [one_test_score(seed) for seed in SEEDS]
    bad = [r for r in anchor_records if r["status"] != "ok"]
    if bad:
        for r in bad:
            print(f"[b3c] anchor TEST FAILED: {r['out_dir']} ({r['status']})", flush=True)
        sys.exit(1)

    records = [one_run(args.variant, seed, args.resume) for seed in SEEDS]
    print(f"[b3c] wall: {(time.time() - start) / 60.0:.0f} min", flush=True)
    bad = [r for r in records if r["status"] != "ok"]
    for r in bad:
        print(f"[b3c] FAILED: {r['out_dir']} ({r['status']})", flush=True)
    if bad:
        sys.exit(1)
    summarize(args.variant, records)
    sys.exit(0)


if __name__ == "__main__":
    main()
