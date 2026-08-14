"""B.2 exploration runner -- PREREGISTRATION_W2.md section 6 (VAL ONLY, test frozen).

Trains a seq-family candidate under the confirmed protocol and scores it on the
VAL split only, paired against the B.1 seq_v2 backbone (whose val scores are
produced additively into its frozen run dir -- the test npz is never touched).
The named objective is on the report: beat the causal GP.

Discipline: this runner never sets CES_SPLIT_TAG=test. Promotion of a promising
candidate to a 4-seed confirmatory TEST run happens only after its decision rule
is committed to PREREGISTRATION_W2.md.

Usage (repo root):
  py ces_prediction/experiments/b2_explore/run_b2_explore.py --variant v3 --smoke
  py ces_prediction/experiments/b2_explore/run_b2_explore.py --variant v3 [--seeds 42 7]
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
from run_b1_gate import GATE_ENV, AMBIENT_VARS, SMOKE_OVERRIDES, run_step  # noqa: E402

EXPLORE_SEEDS = (42, 7)   # exploration splits; confirmation uses all four


def base_env(smoke):
    env = os.environ.copy()
    for var in AMBIENT_VARS:
        env.pop(var, None)
    env.update(GATE_ENV)
    if smoke:
        env.update(SMOKE_OVERRIDES)
    env["CES_SPLIT_TAG"] = "val"   # the whole point of this runner
    return env


def seq_env(env, seed, init, out_dir, smoke):
    control_out = DATA / (f".b1_w2cut_s{seed}" + ("_smoke" if smoke else ""))
    split_dir = DATA / (f".b1_w2cut_split_s{seed}" + ("_smoke" if smoke else ""))
    env.update({
        "CES_SEED": str(seed),
        "CES_INIT_SEED": str(init),
        "CES_SPLIT_DIR": str(split_dir),
        "CES_OUTPUT_DIR": str(out_dir),
        "CES_CONTROL_METRICS": str(control_out / "metrics.json"),
        "CES_PER_SHOT_NORM": "1",
    })
    if smoke:
        env.update({"CES_SEQ_EPOCHS": "2", "CES_SEQ_MAX_FILES": "40", "CES_SEQ_DEVICE": "cpu"})
    return env


def ensure_baseline_val(seed, smoke, resume):
    """Additively score the B.1 seq_v2 backbone run on VAL (its dir keeps the
    frozen test artifacts untouched)."""
    out_dir = DATA / (f".b1_seqv2_s{seed}_i{seed}" + ("_smoke" if smoke else ""))
    if smoke:
        out_dir = DATA / ".b1_seqv2_s42_i42_smoke"
    val_npz = out_dir / "comparison_errors_val.npz"
    if resume and val_npz.exists():
        return val_npz
    if not (out_dir / "weights" / "seq_lstm.pth").exists():
        raise SystemExit(f"FATAL: B.1 backbone run missing: {out_dir}")
    env = seq_env(base_env(smoke), seed, seed, out_dir, smoke)
    env["CES_SEQ_MODEL"] = "v2"
    if not run_step([SEQ_DIR / "eval_seq.py"], env, out_dir / "run.log",
                    f"baseline-val s{seed}", artifacts=(val_npz,)):
        raise SystemExit(f"FATAL: baseline val scoring failed for seed {seed}")
    return val_npz


def one_candidate(variant, seed, smoke, resume):
    out_dir = DATA / (f".b2_{variant}_s{seed}" + ("_smoke" if smoke else ""))
    paired_json = out_dir / "paired_vs_seqv2_val.json"
    if resume and not smoke and paired_json.exists():
        print(f"[b2] === {out_dir.name}: complete, skipping (--resume)", flush=True)
        return {"seed": seed, "out_dir": str(out_dir), "status": "ok", "resumed": True}
    out_dir.mkdir(parents=True, exist_ok=True)
    log = out_dir / "run.log"
    baseline_npz = ensure_baseline_val(seed, smoke, resume=True)

    env = seq_env(base_env(smoke), seed, seed, out_dir, smoke)
    env["CES_SEQ_MODEL"] = variant

    record = {"seed": seed, "variant": variant, "out_dir": str(out_dir), "status": "ok"}
    start = time.time()
    print(f"[b2] === {out_dir.name}", flush=True)
    try:
        if not run_step([SEQ_DIR / "train_seq.py"], env, log, "train",
                        artifacts=(out_dir / "weights" / "seq_lstm.pth", out_dir / "metrics.json")):
            record["status"] = "train_failed"
            return record
        if not run_step([SEQ_DIR / "eval_seq.py"], env, log, "eval(val)",
                        artifacts=(out_dir / "comparison_errors_val.npz",
                                   out_dir / "comparison_metrics.json")):
            record["status"] = "compare_failed"
            return record
        if smoke:
            return record
        if not run_step([CES_DIR / "bootstrap_compare.py"], env, log, "bootstrap(val)",
                        artifacts=(out_dir / "bootstrap_summary.json",)):
            record["status"] = "bootstrap_failed"
            return record
        if not run_step([PAIRED,
                         "--a", out_dir / "comparison_errors_val.npz",
                         "--b", baseline_npz,
                         "--out", paired_json],
                        env, log, "paired-vs-seqv2(val)", artifacts=(paired_json,)):
            record["status"] = "paired_failed"
    finally:
        record["minutes"] = round((time.time() - start) / 60.0, 1)
    return record


def summarize(variant, records):
    rows = []
    for rec in records:
        out_dir = Path(rec["out_dir"])
        row = {"seed": rec["seed"], "status": rec["status"], "minutes": rec.get("minutes")}
        try:
            pr = json.loads((out_dir / "paired_vs_seqv2_val.json").read_text(encoding="utf-8"))["targets"]
            bs = json.loads((out_dir / "bootstrap_summary.json").read_text(encoding="utf-8"))
            for t in ("CES_TI", "CES_VT"):
                row[f"paired_{t[-2:]}"] = pr[t]["skill_point"]
                row[f"paired_{t[-2:]}_ci"] = pr[t]["skill_ci95"]
                row[f"paired_{t[-2:]}_verdict"] = ("A" if pr[t]["a_better"]
                                                  else ("B" if pr[t]["b_better"] else "ns"))
                g = bs["splits"]["val"][t].get("gp_causal")
                if g:
                    row[f"vs_gp_causal_{t[-2:]}"] = g["skill_point"]
                    row[f"vs_gp_causal_{t[-2:]}_pass"] = g["pass"]
        except (FileNotFoundError, KeyError) as exc:
            row["metrics_missing"] = repr(exc)
        rows.append(row)
    out = {"variant": variant, "split": "val (exploration -- test frozen)", "rows": rows}
    out_path = DATA / f".b2_{variant}_summary.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=list(EXPLORE_SEEDS))
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    seeds = (42,) if args.smoke else tuple(args.seeds)
    records = []
    for seed in seeds:
        records.append(one_candidate(args.variant, seed, args.smoke, args.resume))
    bad = [r for r in records if r["status"] != "ok"]
    for r in bad:
        print(f"[b2] FAILED: {r['out_dir']} ({r['status']})", flush=True)
    if not args.smoke and not bad:
        summarize(args.variant, records)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
