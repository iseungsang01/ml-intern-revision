"""Controlled experiment: does training WITHOUT held (stuck) CES values change the result?

Motivation (승상님, 2026-07-30): 54% of observed CES_VT values are forward-filled holds.
The final model's physics assumption -- "V_rot is strongly persistent on the 10 ms grid,
so past CES history is the dominant V_rot source" -- was learned FROM that artifact:
held values sit in the training targets AND in the ces_history input channels, teaching
the model that copying history is near-optimal. The earlier A/B (data/.ab_stuck_B,
docs/재현_절차_기록.md §6.8) that "settled" this was a single-seed val-metrics check --
below the project's own ≥4-seed-paired bar for any CES_VT claim.

Design (one controlled variable: CES_DROP_STUCK_TARGETS=1 for the whole pipeline):
  - Model: the thesis final architecture, pinned iter009 snapshot
    (ces_prediction/model_iter009.py) -- NOT current model.py.
  - Treatment arm: train + evaluate with held values NaN'd at dataset load, which removes
    them from (a) supervision targets, (b) ces_history values & observation flags,
    (c) normalization stats, and (d) interpolation-baseline anchors.
  - Control arm: the trusted data/.vt_repro_* family (same architecture, held kept at
    train time, evaluated genuine-only) -- already on disk, not retrained.
  - Splits are REGENERATED per seed (empty split dir): the pinned split CSVs were derived
    from the held-included dataset and cannot re-derive under drop_stuck=1. The file-level
    split depends only on the (unchanged) file list + CES_SEED, so the test shots must
    come out identical -- asserted against the control manifest after training.
  - Pairing safety: paired_model_compare.py aborts unless the per-target shot arrays match
    and se_pchip is bit-identical, so a population mismatch cannot slip through.

Usage (repo root):
  py ces_prediction/experiments/stuckfree/run_stuckfree.py --smoke      # 1-epoch sanity
  py ces_prediction/experiments/stuckfree/run_stuckfree.py              # full 4-seed batch
  py ces_prediction/experiments/stuckfree/run_stuckfree.py --resume     # skip finished seeds

Summary: data/.sf_summary.json. No git actions, no Slack dependency.
"""

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
EXP_DIR = REPO_ROOT / "ces_prediction" / "experiments"
sys.path.insert(0, str(EXP_DIR))
import runner_common as rc  # noqa: E402  (shared run_step / env / split+control paths)

SEEDS = (42, 1, 7, 123)


def check_test_isolation(split_dir, seed):
    """Pairing safety: every CONTROL test shot must be held out of OUR training.

    drop_stuck=1 can shrink the valid-file list (a shot whose only observations
    were held values drops out), which shifts the seeded partition -- our
    regenerated test set may gain extra shots. That is safe (superset): scoring
    then simply uses the CONTROL manifest. What is NOT safe is a control test
    shot landing in our train/val (leak) -- that kills the seed.
    """
    ours = json.loads((split_dir / "split_manifest.json").read_text(encoding="utf-8"))
    ref = json.loads((rc.SPLIT_SRC[seed] / "split_manifest.json").read_text(encoding="utf-8"))
    our_train = set(ours["train_files"]) | set(ours["val_files"])
    a, b = set(ours["test_files"]), set(ref["test_files"])
    leaked = b & our_train  # control test shots actually trained on -- fatal
    if leaked:
        print(f"[runner]   FATAL(seed {seed}): control test shots leaked into our "
              f"training: {sorted(leaked)[:5]} -- seed unusable.")
        return False
    vanished = b - a - our_train  # absent from the drop_stuck dataset entirely
    if vanished:
        print(f"[runner]   note: {sorted(vanished)[:3]} vanished under drop_stuck "
              f"(no genuine targets); eval excludes them symmetrically in both arms")
    extra = a - b
    if extra:
        print(f"[runner]   our test is a superset of control (+{len(extra)}: "
              f"{sorted(extra)[:3]}); evaluating on the control manifest ({len(b)} shots)")
    else:
        print(f"[runner]   test-file set matches control ({len(a)} shots present)")
    return True


def one_run(seed, smoke, resume=False):
    tag = f"stuckfree_s{seed}" + ("_smoke" if smoke else "")
    out_dir = REPO_ROOT / "data" / (".sf_smoke" if smoke else f".sf_iter009_s{seed}")
    split_dir = REPO_ROOT / "data" / (".sf_smoke_split" if smoke else f".sf_split_s{seed}")
    paired_json = out_dir / "paired_vs_heldkept_iter009.json"
    if resume and not smoke and paired_json.exists():
        print(f"[runner] === {tag}: complete, skipping (--resume)")
        return {"seed": seed, "smoke": smoke, "out_dir": str(out_dir), "status": "ok", "resumed": True}
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "run.log"

    env = dict(__import__("os").environ)
    env.update(rc.SMOKE_ENV if smoke else rc.FULL_ENV)
    env.update({
        "CES_DROP_STUCK_TARGETS": "1",  # the single controlled variable
        "CES_SEED": str(seed),
        "CES_INIT_SEED": str(seed),
        "CES_SPLIT_DIR": str(split_dir),
        "CES_OUTPUT_DIR": str(out_dir),
        "CES_SPLIT_TAG": "test",
    })
    if not smoke:
        # Pin the file-level split to the control manifest: drop_stuck can shrink
        # the valid-file list, and the seeded shuffle would then repartition and
        # leak control test shots into training (seen on seeds 7/123).
        env["CES_FILE_SPLIT_FROM"] = str(rc.SPLIT_SRC[seed] / "split_manifest.json")
    env.pop("CES_ABLATE", None)

    print(f"[runner] === {tag} -> {out_dir.name}")
    record = {"seed": seed, "smoke": smoke, "out_dir": str(out_dir), "status": "ok"}
    start = time.time()
    try:
        train_done = (resume and not smoke
                      and (out_dir / "weights" / "multimodal_ces.pth").exists()
                      and (out_dir / "metrics.json").exists()
                      and (split_dir / "split_manifest.json").exists())
        if train_done:
            print("[runner]   train: already complete, skipping (--resume)")
        else:
            # Always regenerate the split: pinned rows from the held-included
            # dataset cannot re-derive once held values are NaN'd.
            if split_dir.exists():
                shutil.rmtree(split_dir)
            split_dir.mkdir(parents=True)
            if not rc.run_step([rc.CES_DIR / "train.py"], env, log_path, "train"):
                record["status"] = "train_failed"
                return record
        if not smoke:
            if not check_test_isolation(split_dir, seed):
                record["status"] = "split_leak"
                return record
            # Score on the CONTROL manifest: its test set is a subset of ours,
            # so the population matches the control npz row-for-row.
            env = dict(env)
            env["CES_SPLIT_DIR"] = str(rc.SPLIT_SRC[seed])
        if not rc.run_step([rc.CES_DIR / "compare_baselines.py"], env, log_path, "compare(test)"):
            record["status"] = "compare_failed"
            return record
        if not rc.run_step([rc.CES_DIR / "bootstrap_compare.py"], env, log_path, "bootstrap"):
            record["status"] = "bootstrap_failed"
            return record
        if not smoke:
            ok = rc.run_step(
                [rc.PAIRED_MODEL_COMPARE,
                 "--a", out_dir / "comparison_errors_test.npz",
                 "--b", rc.BASELINE_OUT[seed] / "comparison_errors_test.npz",
                 "--out", paired_json],
                env, log_path, "paired-vs-heldkept",
            )
            if not ok:
                record["status"] = "paired_failed"
    finally:
        record["minutes"] = round((time.time() - start) / 60.0, 1)
    return record


def summarize(records, out_path):
    rows = []
    vt_favor = vt_against = ti_against = vt_pr4 = 0
    vt_skills = []
    for rec in sorted((r for r in records if not r["smoke"]), key=lambda r: r["seed"]):
        row = {"seed": rec["seed"], "status": rec["status"], "minutes": rec.get("minutes")}
        out_dir = Path(rec["out_dir"])
        try:
            cm = json.loads((out_dir / "comparison_metrics.json").read_text(encoding="utf-8"))
            bs = json.loads((out_dir / "bootstrap_summary.json").read_text(encoding="utf-8"))
            pr = json.loads((out_dir / "paired_vs_heldkept_iter009.json").read_text(encoding="utf-8"))
            for tgt in ("CES_TI", "CES_VT"):
                row[f"skill_vs_pchip_{tgt[-2:]}"] = cm["per_target"][tgt]["skill_vs_pchip"]
                row[f"pchip_pass_{tgt[-2:]}"] = bs["splits"]["test"][tgt]["pchip"]["pass"]
                p = pr["targets"][tgt]
                row[f"paired_{tgt[-2:]}"] = {
                    "skill_point": p["skill_point"], "skill_ci95": p["skill_ci95"],
                    "a_better": p["a_better"], "b_better": p["b_better"],
                }
            vt = pr["targets"]["CES_VT"]
            vt_skills.append(vt["skill_point"])
            vt_favor += int(vt["a_better"])
            vt_against += int(vt["b_better"])
            ti_against += int(pr["targets"]["CES_TI"]["b_better"])
            vt_pr4 += int(row["pchip_pass_VT"])
        except (FileNotFoundError, KeyError) as exc:
            row["missing"] = repr(exc)
        rows.append(row)

    n_ok = len(vt_skills)
    mean_vt = sum(vt_skills) / n_ok if n_ok else None
    complete = n_ok == len(rows) == 4
    if complete and mean_vt > 0 and vt_favor >= 3 and ti_against == 0:
        verdict = "KEEP_CANDIDATE"
    elif complete and mean_vt > 0 and vt_favor >= 2 and ti_against <= 1:
        verdict = "SIGNAL"
    elif complete:
        verdict = "NO_EFFECT_OR_WORSE"
    else:
        verdict = "INCOMPLETE"
    if complete and ti_against >= 2:
        verdict += "+TI_REGRESSION"

    summary = {
        "design": "iter009 trained with CES_DROP_STUCK_TARGETS=1 vs data/.vt_repro_* (held kept)",
        "seeds": rows,
        "mean_paired_skill_VT": mean_vt,
        "paired_VT_favor_seeds": vt_favor,
        "paired_VT_against_seeds": vt_against,
        "paired_TI_against_seeds": ti_against,
        "own_VT_pchip_pass_seeds": vt_pr4,
        "verdict": verdict,
    }
    print(f"\n[summary] stuck-free training: verdict={verdict} "
          f"mean_paired_VT={mean_vt if mean_vt is None else round(mean_vt, 4)} "
          f"VT favor={vt_favor}/4 against={vt_against}/4 "
          f"own VT PR4 pass={vt_pr4}/4 TI regression seeds={ti_against}/4")
    for row in rows:
        p = row.get("paired_VT")
        detail = (f"VT paired={p['skill_point']:+.4f} CI=[{p['skill_ci95'][0]:+.4f},{p['skill_ci95'][1]:+.4f}]"
                  if p else f"missing={row.get('missing', row['status'])}")
        print(f"  seed {row['seed']:>3}: {row['status']:<14} {detail}")
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[summary] saved {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS), choices=SEEDS)
    ap.add_argument("--smoke", action="store_true", help="1-epoch tiny-cap sanity run (seed 42)")
    ap.add_argument("--resume", action="store_true",
                    help="skip seeds whose paired_vs_heldkept_iter009.json already exists")
    args = ap.parse_args()

    if not any((REPO_ROOT / "data").glob("s*.csv")):
        raise SystemExit("FATAL: no shot CSVs in data/ -- real data required, aborting.")

    records = []
    if args.smoke:
        records.append(one_run(42, smoke=True))
    else:
        for seed in args.seeds:
            records.append(one_run(seed, smoke=False, resume=args.resume))

    if args.smoke:
        rec = records[0]
        print(f"[smoke] {rec['status']} ({rec.get('minutes')} min)")
        sys.exit(0 if rec["status"] == "ok" else 1)
    summarize(records, REPO_ROOT / "data" / ".sf_summary.json")


if __name__ == "__main__":
    main()
