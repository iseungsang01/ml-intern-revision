"""Runner for the interpretable anchor+correction redesign experiment.

The question is NOT "does a 1,258-parameter model win". It is: **how much of the
201,258-parameter model's margin over interpolation is reachable by a model whose
every term has a name?** The anchor model predicts

    y = [anchor: nearest observed CES, smoothed] + [slope x gap] + [per-modality rate x gap]

with every learned term zero-initialised, so training starts exactly at persistence
and the skill is literally "what learning added". `decompose()` returns the per-term
contribution, so a transient shot can be explained term by term. Together with the
interpolation baselines this gives a complexity ladder --
interpolation (0 params) -> anchor+delta (1,258) -> full model (201,258) --
which is the constructive answer to "the model is too complex to explain".

Protocol (mirrors experiments/stuckfree/run_stuckfree.py, NOT the older CT flow). Per seed:
  1. regenerate the split under the held-free treatment with the FILE-LEVEL split pinned
     to the control manifest, then verify no control test shot leaked into training,
  2. point CES_MODEL_FILE at experiments/anchor/model_anchor.py (no file is rewritten),
  3. train.py -> compare_baselines.py (test, scored on the CONTROL manifest) ->
     bootstrap_compare.py,
  4. paired_model_compare.py vs the **held-free** iter009 control (data/.sf_iter009_s*).

**Data treatment is pinned explicitly** (`CES_DROP_STUCK_TARGETS=1`, train AND eval) per
THESIS_RESULTS §8c. Letting it default is what produced a wrong window-sweep conclusion
once (§8f), and here it would be worse than wrong: the control must share the treatment
or the paired difference confounds architecture with data cleaning. The held-free control
`.sf_iter009_s*` has a row-for-row identical scored population to `.vt_repro_*` genuine
(verified: TI 32,787/35,856/36,083/34,399, VT 10,729/13,660/14,698/14,126).

Usage (repo root):
  py ces_prediction/experiments/anchor/run_anchor_experiments.py --smoke   # 1-epoch sanity
  py ces_prediction/experiments/anchor/run_anchor_experiments.py           # full 4-seed batch

Summary: data/.anchor_summary.json (+ printed table). No git actions, no Slack dependency.
"""

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
CES_DIR = REPO_ROOT / "ces_prediction"
ANCHOR_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CES_DIR / "experiments"))
sys.path.insert(0, str(CES_DIR / "experiments" / "stuckfree"))
from runner_common import PAIRED_MODEL_COMPARE, SEEDS, SPLIT_SRC, run_step  # noqa: E402
from run_stuckfree import check_test_isolation  # noqa: E402  (pairing safety check)
MODEL_ANCHOR = ANCHOR_DIR / "model_anchor.py"

VARIANT = "anchor"
# The paired control is the HELD-FREE iter009 family, not .vt_repro_*: this batch trains
# held-free, so the control must too or the paired difference mixes architecture with data
# treatment. Both families score the identical population row-for-row.
BASELINE_OUT = {seed: REPO_ROOT / "data" / f".sf_iter009_s{seed}" for seed in SEEDS}

# Pinned to the trusted baseline family exactly (docs/재현_절차_기록.md §6.5/§6.9): any cap or
# window drift would regenerate splits and silently break pairing.
FULL_ENV = {
    "CES_TEST_FRACTION": "0.15",
    "CES_WINDOW_SIZE": "4",
    "CES_EPOCHS": "10",
    "CES_BATCH_SIZE": "512",
    "CES_LR": "1e-3",
    "CES_VAL_FRACTION": "0.2",
    "CES_MAX_TRAIN_SAMPLES": "200000",
    "CES_MAX_VAL_SAMPLES": "40000",
    "CES_MAX_TEST_SAMPLES": "40000",
    "CES_TEMPORAL_SUBSETS": "1",
    # Explicit, never inherited (§8c convention; §8f shows what a silent default costs).
    "CES_DROP_STUCK_TARGETS": "1",
}
SMOKE_ENV = {
    **FULL_ENV,
    "CES_EPOCHS": "1",
    "CES_MAX_TRAIN_SAMPLES": "4000",
    "CES_MAX_VAL_SAMPLES": "1200",
    "CES_MAX_TEST_SAMPLES": "1200",
}


def check_variant():
    """Fail before a multi-hour batch if the variant file is not what we think it is."""
    marker = "ANCHOR-EXPERIMENT (anchor)"
    if marker not in MODEL_ANCHOR.read_text(encoding="utf-8"):
        raise SystemExit(f"FATAL: {MODEL_ANCHOR} lacks marker {marker!r}")



def one_run(seed, smoke, resume=False):
    tag = f"{VARIANT}_s{seed}" + ("_smoke" if smoke else "")
    out_dir = REPO_ROOT / "data" / (".anchor_smoke" if smoke else f".anchor_s{seed}")
    split_dir = REPO_ROOT / "data" / (".anchor_smoke_split" if smoke else f".anchor_split_s{seed}")
    paired_json = out_dir / "paired_vs_iter009.json"
    if resume and not smoke and paired_json.exists():
        print(f"[runner] === {tag}: complete, skipping (--resume)")
        return {"variant": VARIANT, "seed": seed, "smoke": smoke,
                "out_dir": str(out_dir), "status": "ok", "resumed": True}
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "run.log"

    env = os.environ.copy()
    env.update(SMOKE_ENV if smoke else FULL_ENV)
    env.update({
        "CES_SEED": str(seed),
        "CES_INIT_SEED": str(seed),
        "CES_SPLIT_DIR": str(split_dir),
        "CES_OUTPUT_DIR": str(out_dir),
        "CES_SPLIT_TAG": "test",
    })
    if not smoke:
        # Pin the FILE-level split to the control manifest. Under drop_stuck the
        # valid-file list can shrink, and the seeded shuffle would then repartition
        # and leak control test shots into our training (observed on seeds 7/123
        # during the stuckfree batch).
        env["CES_FILE_SPLIT_FROM"] = str(SPLIT_SRC[seed] / "split_manifest.json")
    env.pop("CES_ABLATE", None)
    # Select the architecture through the env, like every other CES_* knob -- nothing on
    # disk is rewritten (see ces_prediction/model.py).
    env["CES_MODEL_FILE"] = str(MODEL_ANCHOR)

    print(f"[runner] === {tag} -> {out_dir.name}")
    record = {"variant": VARIANT, "seed": seed, "smoke": smoke, "out_dir": str(out_dir), "status": "ok"}
    start = time.time()
    try:
        # Always regenerate the split: row indices pinned under the held-included
        # dataset cannot re-derive once held values are NaN'd.
        if split_dir.exists():
            shutil.rmtree(split_dir)
        split_dir.mkdir(parents=True)
        if not run_step([CES_DIR / "train.py"], env, log_path, "train"):
            record["status"] = "train_failed"
            return record
        if not smoke:
            if not check_test_isolation(split_dir, seed):
                record["status"] = "split_leak"
                return record
            # Score on the CONTROL manifest so the population matches the control
            # npz row-for-row (our test set may be a superset).
            env = dict(env)
            env["CES_SPLIT_DIR"] = str(SPLIT_SRC[seed])
        if not run_step([CES_DIR / "compare_baselines.py"], env, log_path, "compare(test)"):
            record["status"] = "compare_failed"
            return record
        if not run_step([CES_DIR / "bootstrap_compare.py"], env, log_path, "bootstrap"):
            record["status"] = "bootstrap_failed"
            return record
        if not smoke:
            base_npz = BASELINE_OUT[seed] / "comparison_errors_test.npz"
            if not base_npz.exists():
                record["status"] = f"missing held-free control {base_npz}"
                return record
            ok = run_step(
                [PAIRED_MODEL_COMPARE,
                 "--a", out_dir / "comparison_errors_test.npz",
                 "--b", base_npz,
                 "--out", paired_json],
                env, log_path, "paired-vs-heldfree-iter009",
            )
            if not ok:
                record["status"] = "paired_failed"
    finally:
        record["minutes"] = round((time.time() - start) / 60.0, 1)
    return record


def summarize(records, out_path):
    summary = {"records": records, "per_variant": {}}
    recs = [r for r in records if not r["smoke"]]

    rows, skills = [], []
    favor = against = pchip_pass = 0
    for rec in sorted(recs, key=lambda r: r["seed"]):
        row = {"seed": rec["seed"], "status": rec["status"], "minutes": rec.get("minutes")}
        out_dir = Path(rec["out_dir"])
        try:
            cm = json.loads((out_dir / "comparison_metrics.json").read_text(encoding="utf-8"))
            bs = json.loads((out_dir / "bootstrap_summary.json").read_text(encoding="utf-8"))
            pr = json.loads((out_dir / "paired_vs_iter009.json").read_text(encoding="utf-8"))
            row["skill_vs_pchip_TI"] = cm["per_target"]["CES_TI"]["skill_vs_pchip"]
            row["skill_vs_pchip_VT"] = cm["per_target"]["CES_VT"]["skill_vs_pchip"]
            row["pchip_pass_TI"] = bs["splits"]["test"]["CES_TI"]["pchip"]["pass"]
            ti = pr["targets"]["CES_TI"]
            row["paired_TI"] = {
                "skill_point": ti["skill_point"], "skill_ci95": ti["skill_ci95"],
                "a_better": ti["a_better"], "b_better": ti["b_better"],
            }
            row["paired_VT"] = {
                "skill_point": pr["targets"]["CES_VT"]["skill_point"],
                "skill_ci95": pr["targets"]["CES_VT"]["skill_ci95"],
            }
            skills.append(ti["skill_point"])
            favor += int(ti["a_better"])
            against += int(ti["b_better"])
            pchip_pass += int(row["pchip_pass_TI"])
        except (FileNotFoundError, KeyError) as exc:
            row["missing"] = repr(exc)
        rows.append(row)

    n = len(rows)
    mean_skill = sum(skills) / len(skills) if skills else None
    complete = len(skills) == n == 4
    if complete and mean_skill > 0 and favor >= 3 and pchip_pass == 4:
        verdict = "KEEP_CANDIDATE"
    elif complete and mean_skill > 0 and favor >= 2:
        verdict = "SIGNAL"
    elif complete:
        verdict = "FAIL"
    else:
        verdict = "INCOMPLETE"
    summary["per_variant"][VARIANT] = {
        "seeds": rows,
        "mean_paired_skill_TI": mean_skill,
        "ci_excl0_favor_seeds": favor,
        "ci_excl0_against_seeds": against,
        "pchip_pass_TI_seeds": pchip_pass,
        "verdict": verdict,
    }
    print(f"\n[summary] {VARIANT}: verdict={verdict} "
          f"mean_paired_skill_TI={mean_skill if mean_skill is None else round(mean_skill, 4)} "
          f"favor={favor}/4 against={against}/4 pchip_pass={pchip_pass}/4")
    for row in rows:
        p = row.get("paired_TI")
        detail = (f"paired={p['skill_point']:+.4f} CI=[{p['skill_ci95'][0]:+.4f},{p['skill_ci95'][1]:+.4f}]"
                  if p else f"missing={row.get('missing', row['status'])}")
        print(f"  seed {row['seed']:>3}: {row['status']:<14} {detail}")

    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n[summary] saved {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS), choices=SEEDS)
    ap.add_argument("--smoke", action="store_true", help="1-epoch tiny-cap sanity run (seed 42 only)")
    ap.add_argument("--resume", action="store_true",
                    help="skip seeds whose paired_vs_iter009.json already exists")
    args = ap.parse_args()

    if not any((REPO_ROOT / "data").glob("s*.csv")):
        raise SystemExit("FATAL: no shot CSVs in data/ — real data required, aborting (no synthetic fallback).")
    check_variant()

    records = []
    if args.smoke:
        records.append(one_run(42, smoke=True))
    else:
        for seed in args.seeds:
            records.append(one_run(seed, smoke=False, resume=args.resume))

    if args.smoke:
        for rec in records:
            print(f"[smoke] {VARIANT}: {rec['status']} ({rec.get('minutes')} min)")
        bad = [r for r in records if r["status"] != "ok"]
        sys.exit(1 if bad else 0)
    summarize(records, REPO_ROOT / "data" / ".anchor_summary.json")


if __name__ == "__main__":
    main()
