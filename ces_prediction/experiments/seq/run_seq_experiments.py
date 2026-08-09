"""Runner for the seq-LSTM controlled experiment (4 seeds, paired vs iter009 control).

Single controlled variable: the modelling FRAMING. Full-grid causal LSTM with
loss-side per-target masking, vs the control family's window-sample pipeline.
Training keeps held values (control parity); evaluation is the standard genuine-only
population; control arm = data/.vt_repro_* (never retrained). The split manifests are
read read-only from the control split dirs, so the shot split is identical by
construction. No model.py swapping -- the seq model is self-contained.

Usage (repo root):
  py ces_prediction/experiments/seq/run_seq_experiments.py --smoke    # CPU tiny sanity
  py ces_prediction/experiments/seq/run_seq_experiments.py            # full 4-seed
  py ces_prediction/experiments/seq/run_seq_experiments.py --resume

Summary: data/.seq_summary.json.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path


def step_ok(ok, step_start, *artifacts):
    """Windows CUDA teardown can fastfail (0xC0000409) AFTER all work is saved,
    racing even an os._exit(0) guard. Trust fresh artifacts over the exit code."""
    if ok:
        return True
    for a in artifacts:
        p = Path(a)
        if not p.exists() or p.stat().st_mtime < step_start:
            return False
    print("[runner]   note: nonzero rc but all artifacts fresh "
          "(CUDA teardown fastfail) -> continuing")
    return True

REPO_ROOT = Path(__file__).resolve().parents[3]
EXP_DIR = REPO_ROOT / "ces_prediction" / "experiments"
SEQ_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP_DIR))
import runner_common as rc  # noqa: E402  (shared run_step / env / split+control paths)

SEEDS = (42, 1, 7, 123)

# Arms. `seq`/`seq_sf` are §8d as published (shared encoder, control = the held-KEPT
# iter009 family). `seq_v2` is §8t: the same full-grid framing PLUS the two things §8d
# and §8s each identified separately -- the iter009 V_rot routing and per-shot input
# standardization -- paired against the HELD-FREE iter009 family, which is the control
# that shares its data treatment.
ARMS = {
    "seq":    {"drop_stuck": False, "model": "v1", "per_shot": False, "control": "vt_repro"},
    "seq_sf": {"drop_stuck": True,  "model": "v1", "per_shot": False, "control": "vt_repro"},
    "seq_v2": {"drop_stuck": True,  "model": "v2", "per_shot": True,  "control": "sf_iter009"},
    # §8t's own named follow-up: v2 with per-shot standardization OFF, so that the routing
    # and the standardization can be attributed separately (seq_sf -> seq_v2_nops isolates
    # the routing; seq_v2_nops -> seq_v2 isolates the standardization).
    "seq_v2_nops": {"drop_stuck": True, "model": "v2", "per_shot": False, "control": "sf_iter009"},
}
CONTROL_DIR = {
    "vt_repro":   lambda seed: rc.BASELINE_OUT[seed],
    "sf_iter009": lambda seed: REPO_ROOT / "data" / f".sf_iter009_s{seed}",
}


def one_run(seed, smoke, resume=False, arm="seq"):
    cfg = ARMS[arm]
    variant = arm
    control_out = CONTROL_DIR[cfg["control"]](seed)
    tag = f"{variant}_s{seed}" + ("_smoke" if smoke else "")
    out_dir = REPO_ROOT / "data" / (f".{variant}_smoke" if smoke
                                    else f".{variant}_lstm_s{seed}")
    paired_json = out_dir / f"paired_vs_{cfg['control']}.json"
    if resume and not smoke and paired_json.exists():
        print(f"[runner] === {tag}: complete, skipping (--resume)")
        return {"seed": seed, "smoke": smoke, "out_dir": str(out_dir), "status": "ok", "resumed": True}
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "run.log"

    split_src = rc.SPLIT_SRC[seed]
    if not (split_src / "split_manifest.json").exists():
        raise SystemExit(f"FATAL: control split manifest missing: {split_src}")

    env = os.environ.copy()
    env.update(rc.FULL_ENV)
    env.update({
        "CES_SEED": str(seed),
        "CES_INIT_SEED": str(seed),
        "CES_SPLIT_DIR": str(split_src),  # read-only: manifest supplies the file split
        "CES_OUTPUT_DIR": str(out_dir),
        "CES_SPLIT_TAG": "test",
        # Harness normalization round-trip must reuse the control's stats so
        # se_pchip stays bit-identical to the control npz (pairing guard).
        "CES_CONTROL_METRICS": str(control_out / "metrics.json"),
        # every knob pinned explicitly, never inherited (§8f)
        "CES_DROP_STUCK_TARGETS": "1" if cfg["drop_stuck"] else "0",
        "CES_SEQ_MODEL": cfg["model"],
        "CES_PER_SHOT_NORM": "1" if cfg["per_shot"] else "0",
    })
    env.pop("CES_ABLATE", None)
    if smoke:
        env.update({"CES_SEQ_EPOCHS": "2", "CES_SEQ_MAX_FILES": "40",
                    "CES_SEQ_DEVICE": "cpu"})  # never contend with a running GPU batch

    print(f"[runner] === {tag} -> {out_dir.name}")
    record = {"seed": seed, "smoke": smoke, "out_dir": str(out_dir), "status": "ok"}
    start = time.time()
    try:
        train_done = (resume and not smoke
                      and (out_dir / "weights" / "seq_lstm.pth").exists()
                      and (out_dir / "metrics.json").exists())
        if train_done:
            print("[runner]   train_seq: already complete, skipping (--resume)")
        else:
            t0 = time.time()
            ok = rc.run_step([SEQ_DIR / "train_seq.py"], env, log_path, "train_seq")
            if not step_ok(ok, t0, out_dir / "weights" / "seq_lstm.pth", out_dir / "metrics.json"):
                record["status"] = "train_failed"
                return record
        t0 = time.time()
        ok = rc.run_step([SEQ_DIR / "eval_seq.py"], env, log_path, "eval_seq(test)")
        if not step_ok(ok, t0, out_dir / "comparison_errors_test.npz",
                       out_dir / "comparison_metrics.json"):
            record["status"] = "compare_failed"
            return record
        if smoke:
            return record
        if not rc.run_step([rc.CES_DIR / "bootstrap_compare.py"], env, log_path, "bootstrap"):
            record["status"] = "bootstrap_failed"
            return record
        ok = rc.run_step(
            [EXP_DIR / "paired_model_compare.py",
             "--a", out_dir / "comparison_errors_test.npz",
             "--b", control_out / "comparison_errors_test.npz",
             "--out", paired_json],
            env, log_path, f"paired-vs-{cfg['control']}",
        )
        if not ok:
            record["status"] = "paired_failed"
    finally:
        record["minutes"] = round((time.time() - start) / 60.0, 1)
    return record


def summarize(records, out_path, arm="seq"):
    rows = []
    agg = {t: {"favor": 0, "against": 0, "skills": []} for t in ("TI", "VT")}
    pchip_pass_ti = 0
    for rec in sorted((r for r in records if not r["smoke"]), key=lambda r: r["seed"]):
        row = {"seed": rec["seed"], "status": rec["status"], "minutes": rec.get("minutes")}
        out_dir = Path(rec["out_dir"])
        try:
            cm = json.loads((out_dir / "comparison_metrics.json").read_text(encoding="utf-8"))
            bs = json.loads((out_dir / "bootstrap_summary.json").read_text(encoding="utf-8"))
            pr = json.loads((out_dir / f"paired_vs_{ARMS[arm]['control']}.json")
                            .read_text(encoding="utf-8"))
            for tgt, short in (("CES_TI", "TI"), ("CES_VT", "VT")):
                row[f"skill_vs_pchip_{short}"] = cm["per_target"][tgt]["skill_vs_pchip"]
                row[f"pchip_pass_{short}"] = bs["splits"]["test"][tgt]["pchip"]["pass"]
                p = pr["targets"][tgt]
                row[f"paired_{short}"] = {
                    "skill_point": p["skill_point"], "skill_ci95": p["skill_ci95"],
                    "a_better": p["a_better"], "b_better": p["b_better"],
                }
                agg[short]["skills"].append(p["skill_point"])
                agg[short]["favor"] += int(p["a_better"])
                agg[short]["against"] += int(p["b_better"])
            pchip_pass_ti += int(row["pchip_pass_TI"])
        except (FileNotFoundError, KeyError) as exc:
            row["missing"] = repr(exc)
        rows.append(row)

    n_ok = len(agg["TI"]["skills"])
    complete = n_ok == len(rows) == 4
    means = {t: (sum(a["skills"]) / len(a["skills"]) if a["skills"] else None)
             for t, a in agg.items()}
    if complete and means["TI"] > 0 and agg["TI"]["favor"] >= 3 and pchip_pass_ti == 4:
        verdict = "KEEP_CANDIDATE"
    elif complete and ((means["TI"] > 0 and agg["TI"]["favor"] >= 2)
                       or (means["VT"] > 0 and agg["VT"]["favor"] >= 2)):
        verdict = "SIGNAL"
    elif complete:
        verdict = "FAIL"
    else:
        verdict = "INCOMPLETE"

    summary = {
        "design": "full-grid causal LSTM + masked loss (held kept) vs iter009 window pipeline",
        "seeds": rows,
        "mean_paired_skill": {t: means[t] for t in ("TI", "VT")},
        "paired_favor_seeds": {t: agg[t]["favor"] for t in ("TI", "VT")},
        "paired_against_seeds": {t: agg[t]["against"] for t in ("TI", "VT")},
        "pchip_pass_TI_seeds": pchip_pass_ti,
        "verdict": verdict,
    }
    print(f"\n[summary] seq-LSTM: verdict={verdict} "
          f"mean_paired TI={means['TI'] if means['TI'] is None else round(means['TI'], 4)} "
          f"VT={means['VT'] if means['VT'] is None else round(means['VT'], 4)} "
          f"favor TI={agg['TI']['favor']}/4 VT={agg['VT']['favor']}/4 "
          f"pchip_pass_TI={pchip_pass_ti}/4")
    for row in rows:
        p, q = row.get("paired_TI"), row.get("paired_VT")
        detail = (f"TI={p['skill_point']:+.4f}[{p['skill_ci95'][0]:+.3f},{p['skill_ci95'][1]:+.3f}] "
                  f"VT={q['skill_point']:+.4f}[{q['skill_ci95'][0]:+.3f},{q['skill_ci95'][1]:+.3f}]"
                  if p and q else f"missing={row.get('missing', row['status'])}")
        print(f"  seed {row['seed']:>3}: {row['status']:<14} {detail}")
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[summary] saved {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS), choices=SEEDS)
    ap.add_argument("--smoke", action="store_true", help="tiny CPU sanity run (seed 42)")
    ap.add_argument("--resume", action="store_true",
                    help="skip seeds whose paired json already exists")
    ap.add_argument("--arm", choices=sorted(ARMS), default="seq",
                    help="seq = §8d shared encoder; seq_sf = + held-free; "
                         "seq_v2 = + V_rot routing + per-shot norm (§8t)")
    ap.add_argument("--drop-stuck", action="store_true",
                    help="legacy alias for --arm seq_sf")
    args = ap.parse_args()

    arm = "seq_sf" if (args.drop_stuck and args.arm == "seq") else args.arm

    if not any((REPO_ROOT / "data").glob("s*.csv")):
        raise SystemExit("FATAL: no shot CSVs in data/ -- real data required, aborting.")

    records = []
    if args.smoke:
        records.append(one_run(42, smoke=True, arm=arm))
        rec = records[0]
        print(f"[smoke] {rec['status']} ({rec.get('minutes')} min)")
        sys.exit(0 if rec["status"] == "ok" else 1)
    for seed in args.seeds:
        records.append(one_run(seed, smoke=False, resume=args.resume, arm=arm))
    summarize(records, REPO_ROOT / "data" / f".{arm}_summary.json", arm=arm)


if __name__ == "__main__":
    main()
