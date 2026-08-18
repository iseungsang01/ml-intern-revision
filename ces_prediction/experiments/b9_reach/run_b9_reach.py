"""B.9 axis A: the honest reach ladder — `seq_v2` TRAINED at each reach, not truncated.

§8ac drew its ladder by cutting one trained backbone's recurrent state. §8ae showed that
rung is not an information measurement: a model *trained* at reach 2 recovers +0.260 of the
0.310 that truncation cost on `T_i`, 4/4 significant, so **at least 84% of the deficit was
cold start**. But the trained-at-reach arm available there was the window `iter009` control —
a different architecture — so reach and structure were confounded and the pure-reach value
stayed unmeasured. This batch measures it.

One variable: the reach `r` the model is trained AND scored at (`CES_SEQ_TRAIN_CTX` /
`CES_SEQ_EVAL_CTX`). The architecture is `seq_v2` for every rung, byte-identical to the B.1
backbone; `r = full` is that frozen backbone, reused rather than retrained.

Rungs 2 / 7 / 15 / 31 / 63 are chosen so a dilated causal TCN's receptive field
(`2^(L+1) − 1` for kernel 3) lands on the same integers — axis B compares against these
points directly (PREREGISTRATION_B9.md §2.1, H4). 63 covers §8ac's `T_i` saturation at 50.

**Rows per batch is held constant, not blocks per batch.** Chunking a median-298-row block
at `r = 2` makes 149 sequences out of 1, so a fixed `CES_SEQ_BATCH=16` would shrink the
effective batch from ~4,800 rows to 32 and change the optimizer's gradient noise by two
orders of magnitude — a second variable, and a far larger one than the reach. Every rung
therefore trains on ~`ROWS_PER_BATCH` rows per step, with the block-batch derived from `r`.
The full-reach backbone's own value (16 blocks × 298 median rows) sets the constant.

Each rung is paired against (a) the B.1 backbone of its own split (full reach, same
architecture) = **the pure-reach contrast**, and (b) that split's `W = 2` window control
(reach 2, different architecture) — at `r = 2` that second pairing is the pure-architecture
contrast §8ae could not make.

Usage (repo root):
  py ces_prediction/experiments/b9_reach/run_b9_reach.py --smoke
  py ces_prediction/experiments/b9_reach/run_b9_reach.py [--resume] [--reaches 2 7]
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
REACHES = (2, 7, 15, 31, 63)
EPOCH_CAP = "100"          # non-binding; patience terminates (B.3 lesson)
ROWS_PER_BATCH = 4800      # = 16 blocks x 298 median rows, the backbone's own value (§8ac)
PRACTICAL_EPS = 0.02       # PREREGISTRATION_B9.md §3.1


def run_dir(reach, seed, smoke=False):
    return DATA / (f".b9_v2r{reach}_s{seed}" + ("_smoke" if smoke else ""))


def backbone_dir(seed, smoke=False):
    return DATA / (f".b1_seqv2_s{seed}_i{seed}" + ("_smoke" if smoke else ""))


def one_run(reach, seed, smoke, resume):
    out_dir = run_dir(reach, seed, smoke)
    ref_dir = backbone_dir(seed, smoke)
    split_dir = DATA / (f".b1_w2cut_split_s{seed}" + ("_smoke" if smoke else ""))
    control_out = DATA / (f".b1_w2cut_s{seed}" + ("_smoke" if smoke else ""))
    paired_ref = out_dir / "paired_vs_backbone.json"
    paired_win = out_dir / "paired_vs_w2cut.json"
    if resume and not smoke and paired_ref.exists() and paired_win.exists():
        print(f"[b9a] === {out_dir.name}: complete, skipping (--resume)", flush=True)
        return {"reach": reach, "seed": seed, "out_dir": str(out_dir),
                "status": "ok", "resumed": True}
    if not smoke and not (ref_dir / "comparison_errors_test.npz").exists():
        raise SystemExit(f"FATAL: B.1 backbone run missing: {ref_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    log = out_dir / "run.log"

    env = os.environ.copy()
    for var in AMBIENT_VARS + ("CES_MODEL_FILE", "CES_SEQ_TRAIN_CTX", "CES_SEQ_EVAL_CTX",
                               "CES_SEQ_CONTEXTS", "CES_SEQ_BATCH"):
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
        "CES_SEQ_MODEL": "v2",
        "CES_PER_SHOT_NORM": "1",
        "CES_SEQ_EPOCHS": EPOCH_CAP,
        "CES_SEQ_TRAIN_CTX": str(reach),
        "CES_SEQ_EVAL_CTX": str(reach),
        "CES_SEQ_BATCH": str(max(8, round(ROWS_PER_BATCH / reach))),
    })
    if smoke:
        env.update({"CES_SEQ_EPOCHS": "2", "CES_SEQ_MAX_FILES": "40", "CES_SEQ_DEVICE": "cpu"})

    record = {"reach": reach, "seed": seed, "out_dir": str(out_dir), "status": "ok",
              "seq_batch": int(env["CES_SEQ_BATCH"])}
    start = time.time()
    print(f"[b9a] === {out_dir.name} (batch={env['CES_SEQ_BATCH']} blocks)", flush=True)
    try:
        trained = (resume and not smoke and (out_dir / "weights" / "seq_lstm.pth").exists()
                   and (out_dir / "metrics.json").exists())
        if trained:
            print("[b9a]   train: weights exist, skipping (--resume)", flush=True)
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
                            env, log, f"paired({tag})", artifacts=(dst,)):
                record["status"] = f"paired_{tag}_failed"
                return record
    finally:
        record["seconds"] = round(time.time() - start, 1)

    try:
        m = json.loads((out_dir / "metrics.json").read_text())
        record["best_epoch"] = m.get("best_epoch")
        record["train_ctx"] = m.get("train_ctx")
    except Exception:
        pass
    try:
        b = json.loads((out_dir / "bootstrap_summary.json").read_text())
        for t in ("CES_TI", "CES_VT"):
            node_t = (b.get("targets", {}) or {}).get(t, {}) or {}
            for base in ("persistence", "pchip", "gp_causal"):
                node = node_t.get(base, {}) or {}
                if node:
                    record[f"skill_{t}_{base}"] = node.get("skill_point")
                    record[f"pass_{t}_{base}"] = node.get("pass")
    except Exception:
        pass
    for dst, tag in ((paired_ref, "backbone"), (paired_win, "w2cut")):
        try:
            p = json.loads(dst.read_text())
            for t in ("CES_TI", "CES_VT"):
                node = (p.get("targets", {}) or {}).get(t, {}) or {}
                record[f"paired_{t}_vs_{tag}"] = node.get("skill_point")
                record[f"paired_{t}_vs_{tag}_ci"] = node.get("skill_ci95")
        except Exception:
            pass
    return record


def truncation_ladder():
    """§8ac's truncated rungs, so the two ladders are never quoted apart (§3.4)."""
    path = DATA / ".reach_summary.json"
    if not path.exists():
        return {}
    per_seed = json.loads(path.read_text()).get("per_seed", {})
    out = {}
    for t in ("CES_TI", "CES_VT"):
        for r in REACHES:
            vals = [per_seed[str(s)][t]["ladder"][str(r)]["paired_vs_full"]
                    for s in SEEDS
                    if str(r) in per_seed.get(str(s), {}).get(t, {}).get("ladder", {})]
            if vals:
                out[f"{t}_r{r}"] = float(np.mean(vals))
    return out


def fmt(value, spec="+.3f"):
    return "n/a" if value is None else format(value, spec)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--reaches", nargs="*", type=int, default=list(REACHES))
    args = ap.parse_args()
    reaches = [r for r in args.reaches if r in REACHES] or list(REACHES)
    seeds = SEEDS[:1] if args.smoke else SEEDS

    records = []
    for reach in reaches:
        for seed in seeds:
            rec = one_run(reach, seed, args.smoke, args.resume)
            records.append(rec)
            print(f"[b9a]   -> {rec.get('status')} ({rec.get('seconds')}s)", flush=True)
    if args.smoke:
        print("\n[b9a] smoke done")
        return

    trunc = truncation_ladder()
    summary = {"question": "what is full-block context worth once the model is TRAINED at "
                           "each reach?",
               "protocol": {"env": GATE_ENV, "seeds": list(SEEDS), "reaches": list(REACHES),
                            "architecture": "seq_v2 (identical at every rung)",
                            "rows_per_batch": ROWS_PER_BATCH,
                            "practical_eps": PRACTICAL_EPS,
                            "prereg": "experiments/PREREGISTRATION_B9.md axis A"},
               "runs": records, "ladder": {}, "truncation_ladder_8ac": trunc}

    print("\n" + "=" * 108)
    print("paired vs the full-reach backbone (same architecture) — positive = the rung wins")
    print("reach".rjust(6) + "TI paired".rjust(11) + "sig".rjust(6) + "TI vs pers".rjust(12)
          + "VT paired".rjust(11) + "sig".rjust(6) + "VT vs pers".rjust(12)
          + "  §8ac trunc TI/VT")
    for reach in reaches:
        pts = [r for r in records if r.get("reach") == reach and r.get("status") == "ok"]
        if not pts:
            continue
        node = {"n": len(pts), "seq_batch": pts[0].get("seq_batch")}
        for t in ("CES_TI", "CES_VT"):
            pv = [r[f"paired_{t}_vs_backbone"] for r in pts
                  if r.get(f"paired_{t}_vs_backbone") is not None]
            sk = [r[f"skill_{t}_persistence"] for r in pts
                  if r.get(f"skill_{t}_persistence") is not None]
            node[f"mean_paired_{t}_vs_backbone"] = float(np.mean(pv)) if pv else None
            node[f"mean_skill_{t}_persistence"] = float(np.mean(sk)) if sk else None
            node[f"sig_deficit_{t}"] = sum(
                1 for r in pts if (r.get(f"paired_{t}_vs_backbone_ci") or [0, 0])[1] < 0)
            node[f"saturated_{t}"] = bool(
                pv and abs(float(np.mean(pv))) < PRACTICAL_EPS and node[f"sig_deficit_{t}"] <= 1)
        summary["ladder"][str(reach)] = node
        print(str(reach).rjust(6)
              + fmt(node["mean_paired_CES_TI_vs_backbone"]).rjust(11)
              + f"{node['sig_deficit_CES_TI']}/{len(pts)}".rjust(6)
              + fmt(node["mean_skill_CES_TI_persistence"]).rjust(12)
              + fmt(node["mean_paired_CES_VT_vs_backbone"]).rjust(11)
              + f"{node['sig_deficit_CES_VT']}/{len(pts)}".rjust(6)
              + fmt(node["mean_skill_CES_VT_persistence"]).rjust(12)
              + "  " + fmt(trunc.get(f"CES_TI_r{reach}")) + " / "
              + fmt(trunc.get(f"CES_VT_r{reach}")))

    for t in ("CES_TI", "CES_VT"):
        sat = next((r for r in reaches
                    if summary["ladder"].get(str(r), {}).get(f"saturated_{t}")), None)
        summary[f"saturation_{t}"] = sat
        print(f"[b9a] {t}: trained-at-reach saturation (|paired| < {PRACTICAL_EPS}, "
              f"sig deficits <= 1/4) = {sat if sat else 'not reached within ' + str(max(reaches))}")

    out = DATA / ".b9_reach_ladder.json"
    out.write_text(json.dumps(summary, indent=1))
    print(f"\n[b9a] wrote {out}")


if __name__ == "__main__":
    main()
