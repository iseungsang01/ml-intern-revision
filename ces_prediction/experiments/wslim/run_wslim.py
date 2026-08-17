"""W-SLIM batch: does sizing the window model to its window cost any skill? (§8ad)

승상님, 2026-08-17: "W = 2 is a different input -- it can be made smaller." §8ac §3 confirmed
the cost side (25.6k params / 21 leaf ops / 0.66 ms vs 201k / 57 / 3.02 ms) but cost is not
the claim; this batch measures the skill.

One controlled variable: the architecture (`CES_MODEL_FILE` -> `model_win_slim.py`).
Everything else is B.1 stage A verbatim -- same `GATE_ENV` (W = 2, held-free, 3 keV cut,
per-file cap 500, per-shot standardization OFF for the window family), same frozen split
manifests (`data/.b1_manifest_s*`, so the split is regenerated identically and the
test-isolation assert can be re-run), same seeds, same eval harness. The control is the
frozen `.b1_w2cut_s{seed}` run it pairs against row-for-row.

Reading (fixed before the run, no promotion rule attached -- this is a cost/skill
trade-off measurement, not a backbone gate):
  * `T_i` paired vs `w2cut` negative and significant on >= 3/4 splits => the structure
    that `iter009` spends on the window is doing real work, and the slim model is a
    cheaper-but-worse point on the trade-off.
  * within +/- 0.02 and 0/4 significant deficits => an 7.9x parameter reduction and a
    4.6x latency reduction are free, and `iter009`'s shape is an artifact of the search
    that produced it (as `anchor/` first suspected).

Usage (repo root):
  py ces_prediction/experiments/wslim/run_wslim.py --smoke
  py ces_prediction/experiments/wslim/run_wslim.py            # 4 seeds
  py ces_prediction/experiments/wslim/run_wslim.py --resume
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
CES_DIR = REPO_ROOT / "ces_prediction"
EXP_DIR = CES_DIR / "experiments"
DATA = REPO_ROOT / "data"
MODEL_FILE = Path(__file__).resolve().parent / "model_win_slim.py"
PAIRED = EXP_DIR / "paired_model_compare.py"

sys.path.insert(0, str(CES_DIR))
sys.path.insert(1, str(EXP_DIR))
from b1_gate.run_b1_gate import GATE_ENV, AMBIENT_VARS, SEEDS, run_step  # noqa: E402


def count_params(checkpoint):
    """train.py's metrics.json carries no parameter count, so read the checkpoint."""
    import torch
    sd = torch.load(checkpoint, map_location="cpu")
    return int(sum(v.numel() for v in sd.values() if hasattr(v, "numel")))


def names(seed, smoke):
    tag = f".wslim_s{seed}" + ("_smoke" if smoke else "")
    return DATA / tag, DATA / (f".wslim_split_s{seed}" + ("_smoke" if smoke else ""))


def one(seed, smoke, resume):
    out_dir, split_dir = names(seed, smoke)
    control = DATA / f".b1_w2cut_s{seed}"
    paired_json = out_dir / "paired_vs_w2cut.json"
    if resume and not smoke and paired_json.exists():
        print(f"[wslim] === s{seed}: complete, skipping (--resume)", flush=True)
        return {"seed": seed, "out_dir": str(out_dir), "status": "ok", "resumed": True}

    manifest = DATA / f".b1_manifest_s{seed}" / "split_manifest.json"
    for need in (manifest, control / "comparison_errors_test.npz", control / "metrics.json"):
        if not need.exists():
            raise SystemExit(f"FATAL: B.1 artifact missing, run b1_gate first: {need}")
    src_test = sorted(json.loads(manifest.read_text(encoding="utf-8"))["test_files"])

    out_dir.mkdir(parents=True, exist_ok=True)
    log = out_dir / "run.log"
    env = os.environ.copy()
    for var in AMBIENT_VARS:
        env.pop(var, None)
    env.pop("CES_MODEL_FILE", None)
    env.update(GATE_ENV)
    env.update({
        "CES_SEED": str(seed),
        "CES_INIT_SEED": str(seed),
        "CES_SPLIT_DIR": str(split_dir),
        "CES_OUTPUT_DIR": str(out_dir),
        "CES_SPLIT_TAG": "test",
        "CES_FILE_SPLIT_FROM": str(manifest),
        "CES_PER_SHOT_NORM": "0",          # window family: OFF, as in B.1 stage A
        "CES_MODEL_FILE": str(MODEL_FILE),  # the single controlled variable
    })
    if smoke:
        env.update({"CES_EPOCHS": "1", "CES_MAX_TRAIN_SAMPLES": "4000",
                    "CES_MAX_VAL_SAMPLES": "1200", "CES_MAX_TEST_SAMPLES": "1200",
                    "CES_MAX_SAMPLES_PER_FILE": "50"})

    print(f"[wslim] === s{seed} -> {out_dir.name}", flush=True)
    rec = {"seed": seed, "out_dir": str(out_dir), "status": "ok"}
    start = time.time()
    try:
        if not run_step([CES_DIR / "train.py"], env, log, f"train s{seed}",
                        artifacts=(out_dir / "metrics.json",
                                   out_dir / "weights" / "multimodal_ces.pth")):
            rec["status"] = "train_failed"
            return rec
        got = sorted(json.loads((split_dir / "split_manifest.json")
                                .read_text(encoding="utf-8"))["test_files"])
        if not smoke and got != src_test:
            rec["status"] = "test_isolation_failed"
            return rec
        rec["test_isolation"] = "ok" if got == src_test else "smoke-skip"
        rec["params"] = count_params(out_dir / "weights" / "multimodal_ces.pth")
        if not run_step([CES_DIR / "compare_baselines.py"], env, log, f"compare s{seed}",
                        artifacts=(out_dir / "comparison_errors_test.npz",
                                   out_dir / "comparison_metrics.json")):
            rec["status"] = "compare_failed"
            return rec
        if smoke:
            return rec
        if not run_step([CES_DIR / "bootstrap_compare.py"], env, log, f"bootstrap s{seed}",
                        artifacts=(out_dir / "bootstrap_summary.json",)):
            rec["status"] = "bootstrap_failed"
            return rec
        if not run_step([PAIRED, "--a", out_dir / "comparison_errors_test.npz",
                         "--b", control / "comparison_errors_test.npz",
                         "--out", paired_json], env, log, f"paired s{seed}",
                        artifacts=(paired_json,)):
            rec["status"] = "paired_failed"
    finally:
        rec["minutes"] = round((time.time() - start) / 60.0, 1)
    return rec


def summarize(records):
    out = {"question": "does sizing the window model to W = 2 cost skill?",
           "controlled_variable": "CES_MODEL_FILE (model_win_slim.py)",
           "protocol": {"env": GATE_ENV, "control": "data/.b1_w2cut_s{seed} (B.1 stage A)"},
           "per_seed": {}, "verdict": {}}
    for r in records:
        seed = r["seed"]
        out_dir = Path(r["out_dir"])
        cm = json.loads((out_dir / "comparison_metrics.json").read_text(encoding="utf-8"))
        bs = json.loads((out_dir / "bootstrap_summary.json").read_text(encoding="utf-8"))
        pv = json.loads((out_dir / "paired_vs_w2cut.json").read_text(encoding="utf-8"))["targets"]
        # Counted here too, so a --resume run reports it as well as a fresh one.
        entry = {"params": count_params(out_dir / "weights" / "multimodal_ces.pth"),
                 "minutes": r.get("minutes")}
        for t in ("CES_TI", "CES_VT"):
            gates = bs["splits"]["test"][t]
            entry[t] = {
                "skill_vs_pchip": cm["per_target"][t]["skill_vs_pchip"],
                "gates": {b: gates[b]["pass"] for b in ("pchip", "persistence", "gp", "gp_causal")
                          if b in gates},
                "paired_vs_w2cut": pv[t]["skill_point"],
                "ci95": pv[t]["skill_ci95"],
                "slim_better": pv[t]["a_better"], "control_better": pv[t]["b_better"],
            }
        out["per_seed"][seed] = entry

    for t in ("CES_TI", "CES_VT"):
        vals = [out["per_seed"][r["seed"]][t]["paired_vs_w2cut"] for r in records]
        worse = sum(1 for r in records if out["per_seed"][r["seed"]][t]["control_better"])
        better = sum(1 for r in records if out["per_seed"][r["seed"]][t]["slim_better"])
        out["verdict"][t] = {
            "paired_mean": float(np.mean(vals)), "paired_per_seed": vals,
            "significant_deficits": worse, "significant_wins": better,
            "reading": ("iter009's extra structure earns its keep" if worse >= 3 else
                        "the reduction is free at this measurement's resolution" if worse == 0
                        else "mixed -- report per split"),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--jobs", type=int, default=4)
    args = ap.parse_args()
    if not any(DATA.glob("s*.csv")):
        raise SystemExit("FATAL: no shot CSVs in data/ -- real data required, aborting.")

    seeds = (42,) if args.smoke else SEEDS
    start = time.time()
    jobs = max(1, min(args.jobs, len(seeds)))
    if jobs > 1:
        with ThreadPoolExecutor(max_workers=jobs) as ex:
            futures = [ex.submit(one, s, args.smoke, args.resume) for s in seeds]
            records = [f.result() for f in futures]
    else:
        records = [one(s, args.smoke, args.resume) for s in seeds]
    print(f"[wslim] wall time: {(time.time() - start) / 60.0:.1f} min", flush=True)

    bad = [r for r in records if r["status"] != "ok"]
    for r in bad:
        print(f"[wslim] FAILED: {r['out_dir']} ({r['status']})", flush=True)
    if bad or args.smoke:
        sys.exit(1 if bad else 0)

    summary = summarize(records)
    (DATA / ".wslim_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n{'seed':>6} {'params':>9} {'TI skill':>9} {'TI paired':>11} "
          f"{'VT skill':>9} {'VT paired':>11}")
    for r in records:
        e = summary["per_seed"][r["seed"]]
        print(f"{r['seed']:>6} {format(e['params'] or 0, ','):>9} "
              f"{e['CES_TI']['skill_vs_pchip']:>+9.4f} {e['CES_TI']['paired_vs_w2cut']:>+11.4f} "
              f"{e['CES_VT']['skill_vs_pchip']:>+9.4f} {e['CES_VT']['paired_vs_w2cut']:>+11.4f}")
    for t in ("CES_TI", "CES_VT"):
        v = summary["verdict"][t]
        print(f"{t}: paired mean {v['paired_mean']:+.4f}, "
              f"{v['significant_deficits']}/4 significant deficits, "
              f"{v['significant_wins']}/4 significant wins -> {v['reading']}")
    print(f"[wslim] saved {DATA / '.wslim_summary.json'}", flush=True)
    sys.exit(0)


if __name__ == "__main__":
    main()
