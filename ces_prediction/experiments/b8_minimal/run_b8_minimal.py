"""B.8 minimal ladder: how far down can the parameter count go before skill breaks?

B.4 swept the T_i encoder width (24-260) and found the curve flat -- but it bottomed out
near 34k parameters, because only `hidden_ti` moved. The V_rot branch (18,688) and both
heads were held fixed, so 24k was a floor set by the parts that were never varied, not by
the problem. This batch shrinks every part together.

Two ladders, same protocol as B.1/B.4 (W=2, held-free, cut population, per-shot norm,
frozen split manifests, TEST scored once, shot-clustered bootstrap):

  seq_v2 ladder   v2m12k (11,890) / v2m7k (6,866) / v2m4k (3,898) / v2m2k (2,362)
  b3 ladder       b3m7k (6,750) / b3m2k (2,404) / b3m1k (1,208)

Each run is paired against (a) the B.1 backbone of its own split (357,570 params) and
(b) that split's W=2 window control, on identical rows.

Reference points already measured: backbone 357,570 -> skill_TI 0.236; width 40 (49,170)
-> 0.236; width 24 (34,162) -> 0.230; b3k8 (21,498) -> 0.237; anchor+delta (1,258) -> -0.261.
The open question is what happens between 1,258 and 21,498.

Usage (repo root):
  py ces_prediction/experiments/b8_minimal/run_b8_minimal.py --smoke
  py ces_prediction/experiments/b8_minimal/run_b8_minimal.py [--resume] [--variants v2m7k]
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
VARIANTS = ("v2m12k", "v2m7k", "v2m4k", "v2m2k", "b3m7k", "b3m2k", "b3m1k")
EPOCH_CAP = "100"          # non-binding; patience terminates (B.3 lesson)


def run_dir(variant, seed, smoke=False):
    return DATA / (".b8_" + variant + "_s" + str(seed) + ("_smoke" if smoke else ""))


def backbone_dir(seed, smoke=False):
    return DATA / (".b1_seqv2_s" + str(seed) + "_i" + str(seed) + ("_smoke" if smoke else ""))


def one_run(variant, seed, smoke, resume):
    out_dir = run_dir(variant, seed, smoke)
    ref_dir = backbone_dir(seed, smoke)
    split_dir = DATA / (".b1_w2cut_split_s" + str(seed) + ("_smoke" if smoke else ""))
    control_out = DATA / (".b1_w2cut_s" + str(seed) + ("_smoke" if smoke else ""))
    paired_ref = out_dir / "paired_vs_backbone.json"
    paired_win = out_dir / "paired_vs_w2cut.json"
    if resume and not smoke and paired_ref.exists() and paired_win.exists():
        print("[b8] === " + out_dir.name + ": complete, skipping (--resume)", flush=True)
        return {"variant": variant, "seed": seed, "out_dir": str(out_dir),
                "status": "ok", "resumed": True}
    if not smoke and not (ref_dir / "comparison_errors_test.npz").exists():
        raise SystemExit("FATAL: B.1 backbone run missing: " + str(ref_dir))
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
        "CES_SEQ_MODEL": variant,
        "CES_PER_SHOT_NORM": "1",
        "CES_SEQ_EPOCHS": EPOCH_CAP,
    })
    if smoke:
        env.update({"CES_SEQ_EPOCHS": "2", "CES_SEQ_MAX_FILES": "40", "CES_SEQ_DEVICE": "cpu"})

    record = {"variant": variant, "seed": seed, "out_dir": str(out_dir), "status": "ok"}
    start = time.time()
    print("[b8] === " + out_dir.name, flush=True)
    try:
        trained = (resume and not smoke and (out_dir / "weights" / "seq_lstm.pth").exists()
                   and (out_dir / "metrics.json").exists())
        if trained:
            print("[b8]   train: weights exist, skipping (--resume)", flush=True)
        elif not run_step([SEQ_DIR / "train_seq.py"], env, log, "train",
                          artifacts=(out_dir / "weights" / "seq_lstm.pth",
                                     out_dir / "metrics.json")):
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
        for ref, tag, dst in ((ref_dir, "vs_backbone", paired_ref),
                              (control_out, "vs_w2cut", paired_win)):
            if not run_step([PAIRED, "--a", out_dir / "comparison_errors_test.npz",
                             "--b", ref / "comparison_errors_test.npz", "--out", dst],
                            env, log, "paired(" + tag + ")", artifacts=(dst,)):
                record["status"] = "paired_" + tag + "_failed"
                return record
    finally:
        record["seconds"] = round(time.time() - start, 1)

    try:
        m = json.loads((out_dir / "metrics.json").read_text())
        record["params"] = m.get("n_params")
        record["best_epoch"] = m.get("best_epoch")
    except Exception:
        pass
    try:
        b = json.loads((out_dir / "bootstrap_summary.json").read_text())
        for t in ("CES_TI", "CES_VT"):
            node_t = (b.get("targets", {}) or {}).get(t, {}) or {}
            for base in ("pchip", "gp_causal"):
                node = node_t.get(base, {}) or {}
                if node:
                    record["skill_" + t + "_" + base] = node.get("skill_point")
                    record["pass_" + t + "_" + base] = node.get("pass")
    except Exception:
        pass
    try:
        p = json.loads(paired_ref.read_text())
        for t in ("CES_TI", "CES_VT"):
            node = (p.get("targets", {}) or {}).get(t, {}) or {}
            record["paired_" + t + "_vs_backbone"] = node.get("skill_point")
            record["paired_" + t + "_vs_backbone_ci"] = node.get("skill_ci95")
    except Exception:
        pass
    return record


def fmt(value, spec="+.3f"):
    if value is None:
        return "n/a"
    return format(value, spec)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--variants", nargs="*", default=list(VARIANTS))
    args = ap.parse_args()
    variants = [v for v in args.variants if v in VARIANTS] or list(VARIANTS)
    seeds = SEEDS[:1] if args.smoke else SEEDS

    records = []
    for variant in variants:
        for seed in seeds:
            rec = one_run(variant, seed, args.smoke, args.resume)
            records.append(rec)
            print("[b8]   -> " + str(rec.get("status")) + " (" + str(rec.get("seconds"))
                  + "s, params=" + str(rec.get("params")) + ")", flush=True)
    if args.smoke:
        print("\n[b8] smoke done")
        return

    summary = {"variants": variants, "seeds": list(SEEDS), "runs": records, "ladder": {}}
    print("\n" + "=" * 100)
    header = ("variant".rjust(9) + "params".rjust(9) + "skill_TI".rjust(10)
              + "PCHIP".rjust(8) + "causalGP".rjust(10) + "vs backbone".rjust(13)
              + "sig loss".rjust(10))
    print(header)
    for variant in variants:
        pts = [r for r in records if r.get("variant") == variant and r.get("status") == "ok"]
        if not pts:
            continue
        sk = [r["skill_CES_TI_pchip"] for r in pts if r.get("skill_CES_TI_pchip") is not None]
        pv = [r["paired_CES_TI_vs_backbone"] for r in pts
              if r.get("paired_CES_TI_vs_backbone") is not None]
        pp = sum(1 for r in pts if r.get("pass_CES_TI_pchip"))
        gg = sum(1 for r in pts if r.get("pass_CES_TI_gp_causal"))
        loss = sum(1 for r in pts if (r.get("paired_CES_TI_vs_backbone_ci") or [0, 0])[1] < 0)
        node = {"params": pts[0].get("params"), "n": len(pts),
                "mean_skill_TI": float(np.mean(sk)) if sk else None,
                "pchip_pass": pp, "gp_causal_pass": gg,
                "mean_paired_vs_backbone": float(np.mean(pv)) if pv else None,
                "sig_loss_vs_backbone": loss}
        summary["ladder"][variant] = node
        print(variant.rjust(9) + str(node["params"]).rjust(9)
              + fmt(node["mean_skill_TI"]).rjust(10)
              + (str(pp) + "/" + str(len(pts))).rjust(8)
              + (str(gg) + "/" + str(len(pts))).rjust(10)
              + fmt(node["mean_paired_vs_backbone"]).rjust(13)
              + str(loss).rjust(10))
    out = DATA / ".b8_minimal_summary.json"
    out.write_text(json.dumps(summary, indent=1))
    print("\n[b8] wrote " + str(out))


if __name__ == "__main__":
    main()
