"""B.3 exploration runner -- VAL ONLY, test frozen (PREREGISTRATION_W2.md sec. 6, B.3).

Trains the minimal interpretable candidates (b3k4 / b3k8: the T_i latent width is
the ONE explored variable) under the confirmed protocol and scores them on VAL,
paired against (a) the B.1 seq_v2 backbone and (b) the retrained anchor family
(run_b3_anchor.py must have produced the val npz first). Selection rule, stated
up front: pick the K with the smaller paired-vs-backbone `T_i` deficit on the two
exploration splits; on a tie (|difference| < 0.005 mean) prefer the smaller K.

Discipline: this runner never sets CES_SPLIT_TAG=test. Promotion to the 4-seed
confirmatory TEST run happens only after the decision rule is committed to
PREREGISTRATION_W2.md.

Usage (repo root):
  py ces_prediction/experiments/b3_interp/run_b3_explore.py --smoke
  py ces_prediction/experiments/b3_interp/run_b3_explore.py [--variants b3k4 b3k8]
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
sys.path.insert(0, str(CES_DIR / "experiments" / "b2_explore"))
from run_b1_gate import GATE_ENV, AMBIENT_VARS, SMOKE_OVERRIDES, run_step  # noqa: E402
from run_b2_explore import ensure_baseline_val  # noqa: E402

EXPLORE_SEEDS = (42, 7)
VARIANTS = ("b3k4", "b3k8")


def base_env(smoke):
    env = os.environ.copy()
    for var in AMBIENT_VARS + ("CES_MODEL_FILE",):
        env.pop(var, None)
    env.update(GATE_ENV)
    if smoke:
        env.update(SMOKE_OVERRIDES)
    env["CES_SPLIT_TAG"] = "val"
    return env


def seq_env(env, seed, out_dir, smoke):
    control_out = DATA / (f".b1_w2cut_s{seed}" + ("_smoke" if smoke else ""))
    split_dir = DATA / (f".b1_w2cut_split_s{seed}" + ("_smoke" if smoke else ""))
    env.update({
        "CES_SEED": str(seed),
        "CES_INIT_SEED": str(seed),
        "CES_SPLIT_DIR": str(split_dir),
        "CES_OUTPUT_DIR": str(out_dir),
        "CES_CONTROL_METRICS": str(control_out / "metrics.json"),
        "CES_PER_SHOT_NORM": "1",   # seq-family definition (sec. 8t)
    })
    if smoke:
        env.update({"CES_SEQ_EPOCHS": "2", "CES_SEQ_MAX_FILES": "40", "CES_SEQ_DEVICE": "cpu"})
    return env


def one_candidate(variant, seed, smoke, resume):
    out_dir = DATA / (f".b3_{variant}_s{seed}" + ("_smoke" if smoke else ""))
    paired_backbone = out_dir / "paired_vs_seqv2_val.json"
    paired_anchor = out_dir / "paired_vs_anchor_val.json"
    if resume and not smoke and paired_backbone.exists() and paired_anchor.exists():
        print(f"[b3x] === {out_dir.name}: complete, skipping (--resume)", flush=True)
        return {"seed": seed, "variant": variant, "out_dir": str(out_dir),
                "status": "ok", "resumed": True}
    out_dir.mkdir(parents=True, exist_ok=True)
    log = out_dir / "run.log"
    backbone_npz = ensure_baseline_val(seed, smoke, resume=True)
    anchor_npz = (DATA / (f".b3_anchor_s{seed}" + ("_smoke" if smoke else ""))
                  / "comparison_errors_val.npz")
    if not anchor_npz.exists():
        raise SystemExit(f"FATAL: anchor val npz missing ({anchor_npz}) -- "
                         "run run_b3_anchor.py first.")

    env = seq_env(base_env(smoke), seed, out_dir, smoke)
    env["CES_SEQ_MODEL"] = variant

    record = {"seed": seed, "variant": variant, "out_dir": str(out_dir), "status": "ok"}
    start = time.time()
    print(f"[b3x] === {out_dir.name}", flush=True)
    try:
        if not run_step([SEQ_DIR / "train_seq.py"], env, log, "train",
                        artifacts=(out_dir / "weights" / "seq_lstm.pth",
                                   out_dir / "metrics.json")):
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
        for label, other_npz, out_json in (
            ("paired-vs-seqv2(val)", backbone_npz, paired_backbone),
            ("paired-vs-anchor(val)", anchor_npz, paired_anchor),
        ):
            if not run_step([PAIRED,
                             "--a", out_dir / "comparison_errors_val.npz",
                             "--b", other_npz,
                             "--out", out_json],
                            env, log, label, artifacts=(out_json,)):
                record["status"] = "paired_failed"
                return record
    finally:
        record["minutes"] = round((time.time() - start) / 60.0, 1)
    return record


def summarize(records):
    rows = []
    deficits = {}
    for rec in records:
        out_dir = Path(rec["out_dir"])
        row = {"variant": rec["variant"], "seed": rec["seed"],
               "status": rec["status"], "minutes": rec.get("minutes")}
        try:
            cm = json.loads((out_dir / "comparison_metrics.json").read_text(encoding="utf-8"))
            bs = json.loads((out_dir / "bootstrap_summary.json").read_text(encoding="utf-8"))
            pb = json.loads((out_dir / "paired_vs_seqv2_val.json").read_text(encoding="utf-8"))["targets"]
            pa = json.loads((out_dir / "paired_vs_anchor_val.json").read_text(encoding="utf-8"))["targets"]
            row["skill_vs_pchip_TI"] = cm["per_target"]["CES_TI"]["skill_vs_pchip"]
            for t in ("CES_TI", "CES_VT"):
                row[f"vs_backbone_{t[-2:]}"] = pb[t]["skill_point"]
                row[f"vs_backbone_{t[-2:]}_ci"] = pb[t]["skill_ci95"]
                row[f"vs_anchor_{t[-2:]}"] = pa[t]["skill_point"]
                row[f"vs_anchor_{t[-2:]}_sig_win"] = pa[t]["a_better"]
                g = bs["splits"]["val"][t].get("gp_causal")
                if g:
                    row[f"vs_gp_causal_{t[-2:]}"] = g["skill_point"]
                    row[f"vs_gp_causal_{t[-2:]}_pass"] = g["pass"]
            deficits.setdefault(rec["variant"], []).append(pb["CES_TI"]["skill_point"])
        except (FileNotFoundError, KeyError) as exc:
            row["metrics_missing"] = repr(exc)
        rows.append(row)

    choice = None
    means = {v: sum(d) / len(d) for v, d in deficits.items() if d}
    if len(means) == len(VARIANTS):
        ks = sorted(means, key=lambda v: int(v[3:]))          # small K first
        best = max(means, key=means.get)
        choice = ks[0] if abs(means[ks[0]] - means[ks[-1]]) < 0.005 else best
    out = {"split": "val (exploration -- test frozen)", "rows": rows,
           "mean_vs_backbone_TI": means,
           "selection_rule": "max mean paired-vs-backbone TI; tie (<0.005) -> smaller K",
           "selected_variant": choice}
    out_path = DATA / ".b3_explore_summary.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS))
    ap.add_argument("--seeds", nargs="+", type=int, default=list(EXPLORE_SEEDS))
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    seeds = (42,) if args.smoke else tuple(args.seeds)
    variants = (args.variants[0],) if args.smoke else tuple(args.variants)
    records = []
    for variant in variants:
        for seed in seeds:
            records.append(one_candidate(variant, seed, args.smoke, args.resume))
    bad = [r for r in records if r["status"] != "ok"]
    for r in bad:
        print(f"[b3x] FAILED: {r['out_dir']} ({r['status']})", flush=True)
    if not args.smoke and not bad:
        summarize(records)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
