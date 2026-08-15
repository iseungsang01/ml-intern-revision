"""B.4 size-scaling ceiling (PREREGISTRATION_W2.md sec. 6, B.4 -- rule fixed before the run).

ONE controlled variable: the seq_v2 T_i-encoder width `hidden_ti` in {24, 40, 80, 160, 260}
(34k / 49k / 114k / 358k / 879k params; 260 is the last width under the 1M cap). The V_rot
branch, heads, depth, data treatment and every training constant are fixed. The 160 point
IS the B.1 backbone run (`.b1_seqv2_s{seed}_i{seed}`), reused; the new widths train with a
non-binding 100-epoch cap so every point terminates by the same patience rule.

Per width x split: train -> TEST score once -> bootstrap -> paired vs the 160 point (and vs
the w2cut window family, reported). Verdict is descriptive (no backbone re-selection):
  ceiling  : 260 significantly better than 160 on < 3/4 splits
  knee     : the largest width that is significantly WORSE than 160 on >= 3/4 splits
  V_rot    : must not move with T_i width (branch fixed) -- internal consistency check

Usage (repo root):
  py ces_prediction/experiments/b4_scale/run_b4_scale.py --smoke
  py ces_prediction/experiments/b4_scale/run_b4_scale.py [--resume] [--widths 24 40 80 260]
"""

import argparse
import json
import os
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
WIDTHS = (24, 40, 80, 160, 260)
NEW_WIDTHS = (24, 40, 80, 260)          # 160 = the B.1 backbone runs
EPOCH_CAP = "100"                        # non-binding; termination by patience (B.3 lesson)


def variant_for(width):
    return "v2" if width == 160 else f"v2w{width}"


def run_dir(width, seed, smoke=False):
    if width == 160:
        return DATA / (f".b1_seqv2_s{seed}_i{seed}" + ("_smoke" if smoke else ""))
    return DATA / (f".b4_w{width}_s{seed}" + ("_smoke" if smoke else ""))


def one_run(width, seed, smoke, resume):
    out_dir = run_dir(width, seed, smoke)
    ref_dir = run_dir(160, seed, smoke)
    split_dir = DATA / (f".b1_w2cut_split_s{seed}" + ("_smoke" if smoke else ""))
    control_out = DATA / (f".b1_w2cut_s{seed}" + ("_smoke" if smoke else ""))
    paired_ref = out_dir / "paired_vs_w160.json"
    paired_win = out_dir / "paired_vs_w2cut.json"
    if resume and not smoke and paired_ref.exists() and paired_win.exists():
        print(f"[b4] === {out_dir.name}: complete, skipping (--resume)", flush=True)
        return {"width": width, "seed": seed, "out_dir": str(out_dir), "status": "ok", "resumed": True}
    if not smoke and not (ref_dir / "comparison_errors_test.npz").exists():
        raise SystemExit(f"FATAL: 160-point (B.1 backbone) run missing: {ref_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    log = out_dir / "run.log"

    env = os.environ.copy()
    for var in AMBIENT_VARS + ("CES_MODEL_FILE",):
        env.pop(var, None)
    env.update(GATE_ENV)
    if smoke:
        env.update(SMOKE_OVERRIDES)
    env.update({
        "CES_SEED": str(seed),
        "CES_INIT_SEED": str(seed),
        "CES_SPLIT_DIR": str(split_dir),
        "CES_OUTPUT_DIR": str(out_dir),
        "CES_SPLIT_TAG": "test",
        "CES_CONTROL_METRICS": str(control_out / "metrics.json"),
        "CES_SEQ_MODEL": variant_for(width),
        "CES_PER_SHOT_NORM": "1",
        "CES_SEQ_EPOCHS": EPOCH_CAP,
    })
    if smoke:
        env.update({"CES_SEQ_EPOCHS": "2", "CES_SEQ_MAX_FILES": "40", "CES_SEQ_DEVICE": "cpu"})

    record = {"width": width, "seed": seed, "out_dir": str(out_dir), "status": "ok"}
    start = time.time()
    print(f"[b4] === {out_dir.name} (hidden_ti={width})", flush=True)
    try:
        trained = (resume and not smoke and (out_dir / "weights" / "seq_lstm.pth").exists()
                   and (out_dir / "metrics.json").exists())
        if trained:
            print("[b4]   train: weights exist, skipping (--resume)", flush=True)
        elif not run_step([SEQ_DIR / "train_seq.py"], env, log, "train",
                          artifacts=(out_dir / "weights" / "seq_lstm.pth", out_dir / "metrics.json")):
            record["status"] = "train_failed"
            return record
        if not run_step([SEQ_DIR / "eval_seq.py"], env, log, "eval(test)",
                        artifacts=(out_dir / "comparison_errors_test.npz",
                                   out_dir / "comparison_metrics.json")):
            record["status"] = "compare_failed"
            return record
        if smoke:
            return record
        if not run_step([CES_DIR / "bootstrap_compare.py"], env, log, "bootstrap",
                        artifacts=(out_dir / "bootstrap_summary.json",)):
            record["status"] = "bootstrap_failed"
            return record
        for label, other, out_json in (
            ("paired-vs-w160", ref_dir / "comparison_errors_test.npz", paired_ref),
            ("paired-vs-w2cut", control_out / "comparison_errors_test.npz", paired_win),
        ):
            if not run_step([PAIRED, "--a", out_dir / "comparison_errors_test.npz",
                             "--b", other, "--out", out_json], env, log, label,
                            artifacts=(out_json,)):
                record["status"] = "paired_failed"
                return record
    finally:
        record["minutes"] = round((time.time() - start) / 60.0, 1)
    return record


def summarize():
    curve = {}
    for width in WIDTHS:
        pts = []
        for seed in SEEDS:
            d = run_dir(width, seed)
            cm = json.loads((d / "comparison_metrics.json").read_text(encoding="utf-8"))["per_target"]
            m = json.loads((d / "metrics.json").read_text(encoding="utf-8"))
            bs = json.loads((d / "bootstrap_summary.json").read_text(encoding="utf-8"))["splits"]["test"]
            pt = {"seed": seed, "params": m["n_params"], "best_epoch": m["best_epoch"],
                  "epochs_run": m["epochs_run"],
                  "skill_TI": cm["CES_TI"]["skill_vs_pchip"], "skill_VT": cm["CES_VT"]["skill_vs_pchip"],
                  "pchip_pass_TI": bs["CES_TI"]["pchip"]["pass"],
                  "gp_causal_pass_TI": bs["CES_TI"].get("gp_causal", {}).get("pass")}
            if width != 160:
                pr = json.loads((d / "paired_vs_w160.json").read_text(encoding="utf-8"))["targets"]
                pt["paired_TI_vs_160"] = pr["CES_TI"]["skill_point"]
                pt["paired_TI_vs_160_ci"] = pr["CES_TI"]["skill_ci95"]
                pt["paired_TI_sig_win"] = pr["CES_TI"]["a_better"]
                pt["paired_TI_sig_loss"] = pr["CES_TI"]["b_better"]
                pt["paired_VT_vs_160"] = pr["CES_VT"]["skill_point"]
                pt["paired_VT_sig"] = "A" if pr["CES_VT"]["a_better"] else ("B" if pr["CES_VT"]["b_better"] else "ns")
            pts.append(pt)
        curve[width] = {"points": pts,
                        "mean_skill_TI": float(np.mean([p["skill_TI"] for p in pts])),
                        "mean_skill_VT": float(np.mean([p["skill_VT"] for p in pts]))}
        if width != 160:
            curve[width]["sig_win_vs_160"] = sum(p["paired_TI_sig_win"] for p in pts)
            curve[width]["sig_loss_vs_160"] = sum(p["paired_TI_sig_loss"] for p in pts)
            curve[width]["mean_paired_TI_vs_160"] = float(np.mean([p["paired_TI_vs_160"] for p in pts]))

    ceiling = curve[260]["sig_win_vs_160"] < 3
    knee = None
    for width in sorted(w for w in WIDTHS if w < 160):
        if curve[width]["sig_loss_vs_160"] >= 3:
            knee = width           # keep the LARGEST such width
    verdict = {
        "ceiling_reached_at_or_below_160": bool(ceiling),
        "w260_sig_win_vs_160": f"{curve[260]['sig_win_vs_160']}/4",
        "knee_width_sig_loss_ge3of4": knee,
        "note": ("no width < 160 loses significantly on >= 3/4 splits: the curve is flat down to "
                 "the smallest width measured" if knee is None else
                 f"significant loss first appears at hidden_ti={knee}"),
    }
    out = {"controlled_variable": "seq_v2 hidden_ti (T_i encoder width)", "widths": WIDTHS,
           "curve": curve, "verdict": verdict}
    out_path = DATA / ".b4_scale_summary.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({"verdict": verdict,
                      "mean_skill_TI": {w: round(curve[w]["mean_skill_TI"], 4) for w in WIDTHS},
                      "mean_paired_TI_vs_160": {w: round(curve[w]["mean_paired_TI_vs_160"], 4)
                                                for w in WIDTHS if w != 160}}, indent=2))
    print(f"[b4] summary saved {out_path}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--widths", nargs="+", type=int, default=list(NEW_WIDTHS))
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    if not any(DATA.glob("s*.csv")):
        raise SystemExit("FATAL: no shot CSVs in data/ -- real data required, aborting.")
    prereg = (CES_DIR / "experiments" / "PREREGISTRATION_W2.md").read_text(encoding="utf-8")
    if "B.4 실행 규칙" not in prereg:
        raise SystemExit("FATAL: B.4 rule not committed to PREREGISTRATION_W2.md.")

    start = time.time()
    records = []
    if args.smoke:
        records.append(one_run(24, 42, True, False))
    else:
        for width in args.widths:
            for seed in SEEDS:
                records.append(one_run(width, seed, False, args.resume))
    print(f"[b4] wall: {(time.time() - start) / 60.0:.0f} min", flush=True)
    bad = [r for r in records if r["status"] != "ok"]
    for r in bad:
        print(f"[b4] FAILED: {r['out_dir']} ({r['status']})", flush=True)
    if args.smoke or bad:
        sys.exit(1 if bad else 0)
    if set(args.widths) >= set(NEW_WIDTHS):
        summarize()
    sys.exit(0)


if __name__ == "__main__":
    main()
