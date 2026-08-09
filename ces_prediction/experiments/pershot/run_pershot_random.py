"""§8s part 2: does per-shot input standardization cost anything on the HEADLINE split?

Part 1 (`experiments/campaign/`) showed the repair works where it was aimed: on the
strictly-temporal campaign split, `CES_PER_SHOT_NORM=1` beats the §8n control on `CES_TI`
4/4 init seeds, every paired CI excluding zero. That is only half a verdict. Per-shot
standardization throws away the absolute level of BES/ECEI/MC, and absolute level plausibly
carries `T_i` information -- the paper says so explicitly. So the arm can only be adopted if
it does NOT regress the random-split headline the whole paper rests on.

This runs the same controlled change on the random (headline) split. The control is
**`.sf_iter009_s*`** -- the §8c held-free 4-seed family -- not `.vt_repro_*`: §8c established
that held/forward-filled values must be dropped from TRAINING too, so
`CES_DROP_STUCK_TARGETS=1` is the treatment every new run uses, and it is what part 1
(campaign) used. Pairing against the older held-kept family would have made the data
treatment a second uncontrolled variable. Everything else is pinned to the control: same
file split (via CES_FILE_SPLIT_FROM), same caps, same architecture, same held-free scoring
population (n = 32,787 / 10,729 on seed 42). Single controlled variable: CES_PER_SHOT_NORM.

Verdict rule (pre-registered, fixed before looking): ADOPT iff the paired `CES_TI` skill vs
the control is >= 0 within noise on >= 3/4 seeds (no significant loss on any seed);
CAMPAIGN_ONLY if it significantly regresses the headline while still winning §8s part 1.

Usage (repo root):
  py ces_prediction/experiments/pershot/run_pershot_random.py --smoke
  py ces_prediction/experiments/pershot/run_pershot_random.py [--resume]

Summary: data/.pershot_random_summary.json
"""

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
EXP_DIR = REPO_ROOT / "ces_prediction" / "experiments"
SF_DIR = REPO_ROOT / "ces_prediction" / "experiments" / "stuckfree"
sys.path.insert(0, str(EXP_DIR))
sys.path.insert(0, str(SF_DIR))
import runner_common as rc  # noqa: E402  (shared run_step / env / split+control paths)
from run_stuckfree import check_test_isolation  # noqa: E402  (same split-leak guard)

SEEDS = (42, 1, 7, 123)
TARGETS = ("CES_TI", "CES_VT")
# §8c held-free family: the control this arm is paired against
CONTROL_OUT = {s: REPO_ROOT / "data" / f".sf_iter009_s{s}" for s in SEEDS}
OUT_PATH = REPO_ROOT / "data" / ".pershot_random_summary.json"


def one_run(seed, smoke, resume=False):
    tag = f"pershot_rand_s{seed}" + ("_smoke" if smoke else "")
    out_dir = REPO_ROOT / "data" / (".psr_smoke" if smoke else f".psr_s{seed}")
    split_dir = REPO_ROOT / "data" / (".psr_smoke_split" if smoke else f".psr_split_s{seed}")
    paired_json = out_dir / "paired_vs_control.json"
    if resume and not smoke and paired_json.exists():
        print(f"[psr] === {tag}: complete, skipping (--resume)")
        return {"seed": seed, "smoke": smoke, "out_dir": str(out_dir),
                "status": "ok", "resumed": True}
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "run.log"

    env = dict(os.environ)
    env.update(rc.SMOKE_ENV if smoke else rc.FULL_ENV)
    env.update({
        "CES_PER_SHOT_NORM": "1",        # the single controlled variable
        "CES_DROP_STUCK_TARGETS": "1",   # §8c treatment, same as the control and as part 1
        "CES_SEED": str(seed),
        "CES_INIT_SEED": str(seed),
        "CES_SPLIT_DIR": str(split_dir),
        "CES_OUTPUT_DIR": str(out_dir),
        "CES_SPLIT_TAG": "test",
    })
    if not smoke:
        # Pin the file-level split to the control manifest. per-shot norm changes input
        # VALUES only, but drop_stuck can shrink the valid-file list, and an unpinned
        # seeded shuffle would then repartition and leak control test shots into training
        # (§8c saw exactly that on seeds 7/123).
        env["CES_FILE_SPLIT_FROM"] = str(rc.SPLIT_SRC[seed] / "split_manifest.json")
    env.pop("CES_ABLATE", None)

    print(f"[psr] === {tag} -> {out_dir.name}")
    rec = {"seed": seed, "smoke": smoke, "out_dir": str(out_dir), "status": "ok"}
    start = time.time()
    try:
        train_done = (resume and not smoke
                      and (out_dir / "weights" / "multimodal_ces.pth").exists()
                      and (out_dir / "metrics.json").exists()
                      and (split_dir / "split_manifest.json").exists())
        if train_done:
            print("[psr]   train: already complete, skipping (--resume)")
        else:
            if split_dir.exists():
                shutil.rmtree(split_dir)
            split_dir.mkdir(parents=True)
            if not rc.run_step([rc.CES_DIR / "train.py"], env, log_path, "train"):
                rec["status"] = "train_failed"
                return rec
        if not smoke:
            if not check_test_isolation(split_dir, seed):
                rec["status"] = "split_leak"
                return rec
            # Score on the CONTROL manifest: our test set is a superset of the control's,
            # so the population matches the control npz row-for-row (§8c does the same).
            env = dict(env)
            env["CES_SPLIT_DIR"] = str(rc.SPLIT_SRC[seed])
        if not rc.run_step([rc.CES_DIR / "compare_baselines.py"], env, log_path, "compare(test)"):
            rec["status"] = "compare_failed"
            return rec
        if not rc.run_step([rc.CES_DIR / "bootstrap_compare.py"], env, log_path, "bootstrap"):
            rec["status"] = "bootstrap_failed"
            return rec
        if not smoke:
            if not rc.run_step(
                [EXP_DIR / "paired_model_compare.py",
                 "--a", out_dir / "comparison_errors_test.npz",
                 "--b", CONTROL_OUT[seed] / "comparison_errors_test.npz",
                 "--out", paired_json],
                env, log_path, "paired-vs-control",
            ):
                rec["status"] = "paired_failed"
    finally:
        rec["minutes"] = round((time.time() - start) / 60.0, 1)
    return rec


def summarize(records):
    rows = []
    for rec in sorted((r for r in records if not r["smoke"]), key=lambda r: r["seed"]):
        row = {"seed": rec["seed"], "status": rec["status"], "minutes": rec.get("minutes")}
        out_dir = Path(rec["out_dir"])
        try:
            cm = json.loads((out_dir / "comparison_metrics.json").read_text(encoding="utf-8"))
            bs = json.loads((out_dir / "bootstrap_summary.json").read_text(encoding="utf-8"))
            pj = json.loads((out_dir / "paired_vs_control.json").read_text(encoding="utf-8"))
            for t in TARGETS:
                pt, b = cm["per_target"][t], bs["splits"]["test"][t]["pchip"]
                row[t] = {
                    "n": pt["n"], "rmse_model": pt["rmse_model"],
                    "skill_vs_pchip": pt["skill_vs_pchip"],
                    "ci95": b["skill_ci95"], "pass": b["pass"],
                    "paired_vs_control": pj["targets"][t],
                }
        except (FileNotFoundError, KeyError) as exc:
            row["missing"] = repr(exc)
        rows.append(row)

    ok = [r for r in rows if r.get("CES_TI")]
    verdict = {}
    for t in TARGETS:
        pts = [r[t]["paired_vs_control"]["skill_point"] for r in ok]
        # "significant loss" = paired CI entirely below zero
        losses = sum(1 for r in ok
                     if r[t]["paired_vs_control"]["skill_ci95"][1] < 0)
        wins = sum(1 for r in ok if r[t]["paired_vs_control"]["pass"])
        verdict[t] = {"paired_skill_points": pts,
                      "mean": (sum(pts) / len(pts)) if pts else None,
                      "significant_losses": f"{losses}/{len(ok)}",
                      "significant_wins": f"{wins}/{len(ok)}"}
    n_loss = int(verdict["CES_TI"]["significant_losses"].split("/")[0]) if ok else 0
    verdict["overall"] = ("ADOPT" if (len(ok) == 4 and n_loss == 0)
                          else "CAMPAIGN_ONLY" if len(ok) == 4
                          else "INCOMPLETE")

    summary = {
        "design": ("random (headline) file split, control = .sf_iter009_s* (the §8c held-free "
                   "4-seed family); both arms train and score with CES_DROP_STUCK_TARGETS=1; "
                   "single controlled variable CES_PER_SHOT_NORM"),
        "verdict_rule": ("ADOPT iff no seed shows a significant paired CES_TI loss vs the "
                         "control; CAMPAIGN_ONLY otherwise"),
        "protocol": rc.FULL_ENV, "runs": rows, "verdict": verdict,
    }
    OUT_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n[psr] saved {OUT_PATH}\n")
    for t in TARGETS:
        print(f"=== {t} (headline split, vs PCHIP) ===")
        print(f"{'seed':>5} {'per_shot':>18} {'paired vs control':>26}")
        for r in rows:
            e = r.get(t)
            if not e:
                print(f"{r['seed']:>5} {r.get('status')} {r.get('missing','')}")
                continue
            p = e["paired_vs_control"]
            print(f"{r['seed']:>5} "
                  f"{e['skill_vs_pchip']:+.4f}{'*' if e['pass'] else ' ':>12} "
                  f"{p['skill_point']:+.4f} [{p['skill_ci95'][0]:+.3f},"
                  f"{p['skill_ci95'][1]:+.3f}]{'*' if p['pass'] else ' '}")
        v = verdict[t]
        print(f"   mean paired {v['mean']:+.4f}, significant losses {v['significant_losses']}, "
              f"wins {v['significant_wins']}\n")
    print(f"VERDICT: {verdict['overall']}")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    args = ap.parse_args()

    if not any((REPO_ROOT / "data").glob("s*.csv")):
        raise SystemExit("FATAL: no shot CSVs in data/ -- real data required, aborting.")

    records, batch_start = [], time.time()
    try:
        for seed in args.seeds:
            records.append(one_run(seed, smoke=args.smoke, resume=args.resume))
            print(f"[psr] s{seed}: {records[-1]['status']} ({records[-1].get('minutes')} min)")
            if args.smoke:
                break
    finally:
        print(f"[psr] total wall time: {(time.time() - batch_start) / 60.0:.1f} min")

    if args.smoke:
        sys.exit(1 if records[0]["status"] != "ok" else 0)
    summarize(records)
    sys.exit(1 if [r for r in records if r["status"] != "ok"] else 0)


if __name__ == "__main__":
    main()
