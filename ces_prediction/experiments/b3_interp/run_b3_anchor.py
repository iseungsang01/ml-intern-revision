"""B.3 anchor control family under the confirmed protocol (PREREGISTRATION_W2.md sec. 6, B.3).

The B.3 gate's first condition compares the minimal interpretable model against the
named-terms anchor+delta of section 8k -- but that family (.anchor_s*) is a W = 4
artifact and per section 8v cannot control a new confirmatory claim. This runner
retrains it under the confirmed protocol: W = 2, held-free, 3 keV cut, cap 500,
per-shot OFF (window-family rule), file split pinned to the frozen B.1 manifests
with a full three-list isolation check.

Stages:
  (default)      train the 4 seeds + score VAL only (exploration reference; TEST frozen)
  --stage test   additive TEST scoring -- run ONLY after the B.3 decision rule is
                 committed to PREREGISTRATION_W2.md (the runner refuses otherwise
                 unless --force-test).

Usage (repo root):
  py ces_prediction/experiments/b3_interp/run_b3_anchor.py --smoke
  py ces_prediction/experiments/b3_interp/run_b3_anchor.py [--resume]
  py ces_prediction/experiments/b3_interp/run_b3_anchor.py --stage test
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
DATA = REPO_ROOT / "data"
MODEL_ANCHOR = CES_DIR / "experiments" / "anchor" / "model_anchor.py"
sys.path.insert(0, str(CES_DIR / "experiments" / "b1_gate"))
from run_b1_gate import GATE_ENV, AMBIENT_VARS, SMOKE_OVERRIDES, run_step  # noqa: E402

SEEDS = (42, 1, 7, 123)
# CES_MODEL_FILE is treatment-relevant here but absent from the B.1 scrub list
# (no window run there varied the architecture); scrub it too.
EXTRA_AMBIENT = ("CES_MODEL_FILE",)


def dirs(seed, smoke):
    sfx = "_smoke" if smoke else ""
    return (DATA / f".b3_anchor_s{seed}{sfx}",
            DATA / f".b3_anchor_split_s{seed}{sfx}",
            DATA / (f".b1_w2cut_split_s{seed}" + sfx))


def base_env(smoke):
    env = os.environ.copy()
    for var in AMBIENT_VARS + EXTRA_AMBIENT:
        env.pop(var, None)
    env.update(GATE_ENV)
    if smoke:
        env.update(SMOKE_OVERRIDES)
    return env


def check_isolation(split_dir, seed):
    """All three file lists must reproduce the w2cut control family exactly.

    The reference is the FAMILY's split dir, not the reconstructed B.1 sweep
    manifest: the sweep ran without the spike cut, so a file can lose its last
    valid sample under the cut and legitimately drop from a written list. Both
    this run and the w2cut control pin CES_FILE_SPLIT_FROM to the same manifest
    and share the treatment, so their derived lists must be identical.
    """
    got = json.loads((split_dir / "split_manifest.json").read_text(encoding="utf-8"))
    want = json.loads((DATA / f".b1_w2cut_split_s{seed}" / "split_manifest.json")
                      .read_text(encoding="utf-8"))
    for k in ("train_files", "val_files", "test_files"):
        if sorted(got[k]) != sorted(want[k]):
            print(f"[b3a]   FATAL: {k} moved vs the w2cut family for seed {seed}", flush=True)
            return False
    return True


def one_run(seed, smoke, resume):
    out_dir, split_dir, score_split = dirs(seed, smoke)
    done = out_dir / "bootstrap_summary.json"
    if resume and not smoke and done.exists():
        print(f"[b3a] === {out_dir.name}: complete, skipping (--resume)", flush=True)
        return {"seed": seed, "out_dir": str(out_dir), "status": "ok", "resumed": True}
    out_dir.mkdir(parents=True, exist_ok=True)
    log = out_dir / "run.log"

    env = base_env(smoke)
    env.update({
        "CES_SEED": str(seed),
        "CES_INIT_SEED": str(seed),
        "CES_SPLIT_DIR": str(split_dir),
        "CES_OUTPUT_DIR": str(out_dir),
        "CES_SPLIT_TAG": "val",          # TEST stays frozen until the rule is committed
        "CES_FILE_SPLIT_FROM": str(DATA / f".b1_manifest_s{seed}" / "split_manifest.json"),
        "CES_PER_SHOT_NORM": "0",        # window family: OFF (PREREGISTRATION_W2.md sec. 1)
        "CES_MODEL_FILE": str(MODEL_ANCHOR),
    })

    print(f"[b3a] === anchor s{seed} -> {out_dir.name}", flush=True)
    record = {"seed": seed, "out_dir": str(out_dir), "status": "ok"}
    start = time.time()
    try:
        trained = (resume and not smoke
                   and (out_dir / "weights" / "multimodal_ces.pth").exists()
                   and (out_dir / "metrics.json").exists()
                   and (split_dir / "split_manifest.json").exists())
        if trained:
            print(f"[b3a]   train: weights exist, skipping (--resume)", flush=True)
        else:
            # Row-index pins cannot re-derive across treatment changes: always regenerate.
            if split_dir.exists():
                shutil.rmtree(split_dir)
            split_dir.mkdir(parents=True)
            if not run_step([CES_DIR / "train.py"], env, log, "train",
                            artifacts=(out_dir / "metrics.json",
                                       out_dir / "weights" / "multimodal_ces.pth")):
                record["status"] = "train_failed"
                return record
        if not smoke and not check_isolation(split_dir, seed):
            record["status"] = "split_isolation_failed"
            return record
        # Score on the frozen w2cut split dir -- the exact directory every seq eval
        # reads -- so the population matches the whole family row-for-row.
        if not smoke:
            env["CES_SPLIT_DIR"] = str(score_split)
        if not run_step([CES_DIR / "compare_baselines.py"], env, log, "compare(val)",
                        artifacts=(out_dir / "comparison_errors_val.npz",
                                   out_dir / "comparison_metrics.json")):
            record["status"] = "compare_failed"
            return record
        if not smoke and not run_step([CES_DIR / "bootstrap_compare.py"], env, log,
                                      "bootstrap", artifacts=(done,)):
            record["status"] = "bootstrap_failed"
            return record
    finally:
        record["minutes"] = round((time.time() - start) / 60.0, 1)
    return record


def one_test_score(seed):
    """Additive TEST scoring of an already-trained anchor arm."""
    out_dir, _, score_split = dirs(seed, smoke=False)
    npz = out_dir / "comparison_errors_test.npz"
    if npz.exists():
        print(f"[b3a] === anchor s{seed}: test npz exists, skipping", flush=True)
        return {"seed": seed, "out_dir": str(out_dir), "status": "ok", "resumed": True}
    if not (out_dir / "weights" / "multimodal_ces.pth").exists():
        raise SystemExit(f"FATAL: anchor arm not trained yet: {out_dir}")
    log = out_dir / "run.log"
    env = base_env(smoke=False)
    env.update({
        "CES_SEED": str(seed),
        "CES_INIT_SEED": str(seed),
        "CES_SPLIT_DIR": str(score_split),
        "CES_OUTPUT_DIR": str(out_dir),
        "CES_SPLIT_TAG": "test",
        "CES_PER_SHOT_NORM": "0",
        "CES_MODEL_FILE": str(MODEL_ANCHOR),
    })
    record = {"seed": seed, "out_dir": str(out_dir), "status": "ok"}
    if not run_step([CES_DIR / "compare_baselines.py"], env, log, "compare(test)",
                    artifacts=(npz, out_dir / "comparison_metrics.json")):
        record["status"] = "compare_failed"
        return record
    if not run_step([CES_DIR / "bootstrap_compare.py"], env, log, "bootstrap(+test)",
                    artifacts=(out_dir / "bootstrap_summary.json",)):
        record["status"] = "bootstrap_failed"
    return record


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("train", "test"), default="train")
    ap.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--force-test", action="store_true")
    args = ap.parse_args()

    if not any(DATA.glob("s*.csv")):
        raise SystemExit("FATAL: no shot CSVs in data/ -- real data required, aborting.")
    marker = "ANCHOR-EXPERIMENT (anchor)"
    if marker not in MODEL_ANCHOR.read_text(encoding="utf-8"):
        raise SystemExit(f"FATAL: {MODEL_ANCHOR} lacks marker {marker!r}")

    if args.stage == "test":
        prereg = (CES_DIR / "experiments" / "PREREGISTRATION_W2.md").read_text(encoding="utf-8")
        if "B.3 확증 판정 규칙" not in prereg and not args.force_test:
            raise SystemExit("FATAL: B.3 decision rule not committed to "
                             "PREREGISTRATION_W2.md -- TEST stays frozen.")
        records = [one_test_score(seed) for seed in args.seeds]
    else:
        seeds = (42,) if args.smoke else tuple(args.seeds)
        records = [one_run(seed, args.smoke, args.resume) for seed in seeds]

    bad = [r for r in records if r["status"] != "ok"]
    for r in bad:
        print(f"[b3a] FAILED: {r['out_dir']} ({r['status']})", flush=True)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
