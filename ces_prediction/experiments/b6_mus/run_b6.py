"""B.6 runner: the frozen fold protocol over the microsecond shot set.

Executes PREREGISTRATION_B6.md sec. 4.2 exactly, for each (arm x population):

    folds        leave-one-out over the pool (folds.py), init seed 42, epoch cap 100
                 with patience 6 (cap non-binding -- the B.3 lesson is checked)
    median       the median best_epoch over the folds
    refits       train on ALL pool shots for exactly that many epochs
                 (CES_SEQ_FIXED_EPOCHS=1), init seeds 42 / 1 / 7 / 123
    TEST         eval_seq once per refit; a run dir that already holds
                 comparison_errors_test.npz is REFUSED, never re-scored

Arms (sec. 4.1): a0 control, a1 = a0 + MC features into the V_rot branch
(CES_SEQ_EXTRA_VT), a2 = a1 minus the decimated MC channels (CES_SEQ_ZERO_MC).
Populations (W2 sec. 1.1): cut (CES_TI_SPIKE_CUT_EV=3000) and incl (0), co-primary.

Data staging: the 12 frozen CSVs are copied to data/.b6_data so the grid loader and
the eval harness see exactly the shot set folds.py froze -- nothing else. Every stage
is resumable (existing artifacts are kept); the eval control stats come from the a0
seed-42 refit of the same population so se_pchip pairs bit-identically across arms.

Real data only. `--plumbing-smoke` verifies the machinery end-to-end before the raw
delivery: real CSVs, 2 folds, 2 epochs, VAL scoring only (the TEST budget is untouched)
-- but the A1 features do not exist yet, so the smoke fabricates random feature tables
under data/.b6_plumb_features and stamps every output dir with PLUMBING_SMOKE.json.
Those numbers are machinery checks, never results, and the real path refuses to run
while CES_SEQ_EXTRA_VT points at a fabricated table.

Usage (repo root):
  py ces_prediction/experiments/b6_mus/run_b6.py --plumbing-smoke
  py ces_prediction/experiments/b6_mus/run_b6.py                    # real (needs features)
"""

import argparse
import json
import os
import shutil
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE.parents[1]))          # ces_prediction/
sys.path.insert(1, str(HERE.parents[0]))          # experiments/
sys.path.insert(2, str(HERE.parents[0] / "hires_shots"))
sys.path.insert(3, str(HERE.parents[0] / "b1_gate"))

from runner_common import run_step  # noqa: E402
from folds import TEST, POOL, FOLDS, COMPANIONS  # noqa: E402
from run_b1_gate import GATE_ENV  # noqa: E402

DATA = REPO_ROOT / "data"
SEQ = HERE.parents[0] / "seq"
STAGE_DATA = DATA / ".b6_data"
FEATURE_DIR = DATA / ".b6_features"
REFIT_SEEDS = (42, 1, 7, 123)
FOLD_EPOCH_CAP = "100"
ARMS = ("a0", "a1", "a2")
POPULATIONS = {"cut": "3000", "incl": "0"}


def stage_csvs():
    STAGE_DATA.mkdir(parents=True, exist_ok=True)
    for shot in (*TEST, *POOL, *COMPANIONS):
        dst = STAGE_DATA / f"s{shot}.csv"
        src = DATA / f"s{shot}.csv"
        if not src.exists():
            raise SystemExit(f"FATAL: {src} missing -- real data required.")
        if not dst.exists() or dst.stat().st_size != src.stat().st_size:
            shutil.copy2(src, dst)
    # Companions are staged so H3 can use them later, but they are in NO manifest.


def write_manifest(dst, train, val, test):
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "split_manifest.json").write_text(json.dumps({
        "train_files": [f"s{s}.csv" for s in train],
        "val_files": [f"s{s}.csv" for s in val],
        "test_files": [f"s{s}.csv" for s in test],
    }, indent=1), encoding="utf-8")


def base_env(arm, cut_ev, feature_dir):
    env = dict(os.environ)
    env.update(GATE_ENV)
    env.update({
        "CES_DATA_DIR": str(STAGE_DATA),
        "CES_SEQ_MODEL": "v2",
        "CES_PER_SHOT_NORM": "1",
        "CES_DROP_STUCK_TARGETS": "1",
        "CES_TI_SPIKE_CUT_EV": cut_ev,
        "CES_SEQ_BATCH": "16",
        "CES_LR": "1e-3",
    })
    env.pop("CES_SEQ_EXTRA_VT", None)
    env.pop("CES_SEQ_ZERO_MC", None)
    if arm in ("a1", "a2"):
        env["CES_SEQ_EXTRA_VT"] = str(feature_dir)
    if arm == "a2":
        env["CES_SEQ_ZERO_MC"] = "1"
    return env


def has_artifacts(out_dir):
    return (out_dir / "metrics.json").exists() and (out_dir / "weights" / "seq_lstm.pth").exists()


def train_run(tag, env, split_dir, out_dir, seed, init_seed, epochs, fixed, log):
    if has_artifacts(out_dir):
        print(f"[b6]   {tag}: artifacts exist, skip")
        return True
    out_dir.mkdir(parents=True, exist_ok=True)
    e = dict(env)
    e.update({"CES_SPLIT_DIR": str(split_dir), "CES_OUTPUT_DIR": str(out_dir),
              "CES_SEED": str(seed), "CES_INIT_SEED": str(init_seed),
              "CES_SEQ_EPOCHS": str(epochs)})
    if fixed:
        e["CES_SEQ_FIXED_EPOCHS"] = "1"
    ok = run_step([SEQ / "train_seq.py"], e, log, tag)
    if not ok and has_artifacts(out_dir):
        # The Windows CUDA teardown fastfail (0xC0000409) fires AFTER everything is
        # saved (the known seq-family trap); artifacts on disk are the ground truth.
        print(f"[b6]   {tag}: rc nonzero but artifacts saved -- CUDA teardown trap, accepted")
        return True
    return ok


def eval_run(tag, env, split_dir, out_dir, control_metrics, split_tag, log):
    marker = out_dir / f"comparison_errors_{split_tag}.npz"
    if split_tag == "test" and marker.exists():
        print(f"[b6]   {tag}: TEST already scored -- the once-only lock refuses a re-score")
        return True
    if marker.exists():
        print(f"[b6]   {tag}: {split_tag} scored, skip")
        return True
    e = dict(env)
    e.update({"CES_SPLIT_DIR": str(split_dir), "CES_OUTPUT_DIR": str(out_dir),
              "CES_CONTROL_METRICS": str(control_metrics), "CES_SPLIT_TAG": split_tag,
              "CES_WINDOW_SIZE": "2", "CES_SEED": "42"})
    ok = run_step([SEQ / "eval_seq.py"], e, log, tag)
    if not ok and marker.exists():
        print(f"[b6]   {tag}: rc nonzero but npz saved -- CUDA teardown trap, accepted")
        return True
    return ok


def check_cap(out_dir, cap):
    m = json.loads((out_dir / "metrics.json").read_text(encoding="utf-8"))
    if not m.get("fixed_epochs") and m["best_epoch"] >= int(cap):
        print(f"[b6]   WARNING: {out_dir.name} best_epoch {m['best_epoch']} == cap {cap} "
              "(B.3 lesson: the cap binds; raise it before trusting this fold)")
    return m["best_epoch"]


def fabricate_plumbing_features(pool_test_shots):
    """PLUMBING SMOKE ONLY: random tables so the A1 path can be exercised pre-arrival."""
    import numpy as np
    import pandas as pd
    d = DATA / ".b6_plumb_features"
    d.mkdir(parents=True, exist_ok=True)
    (d / "PLUMBING_SMOKE.json").write_text(json.dumps(
        {"warning": "fabricated tables for plumbing verification only -- never a result"}),
        encoding="utf-8")
    k = 15
    (d / "feature_meta.json").write_text(json.dumps(
        {"k": k, "z_exempt_channels": [12, 13, 14], "plumbing_smoke": True}),
        encoding="utf-8")
    rng = np.random.default_rng(0)
    for shot in pool_test_shots:
        t = pd.read_csv(STAGE_DATA / f"s{shot}.csv", usecols=["time"])["time"].to_numpy(float)
        feat = rng.standard_normal((len(t), k)).astype("float32")
        feat[:, 13] = rng.integers(-1, 2, len(t))
        feat[:, 14] = rng.integers(0, 2, len(t))
        np.savez(d / f"mus_features_s{shot}.npz", time=t, feat=feat)
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plumbing-smoke", action="store_true")
    ap.add_argument("--arms", nargs="*", default=list(ARMS))
    args = ap.parse_args()
    smoke = args.plumbing_smoke

    stage_csvs()
    if smoke:
        feature_dir = fabricate_plumbing_features((*TEST, *POOL, *COMPANIONS))
        pops = {"cut": "3000"}
        folds = FOLDS[:2]
        fold_epochs, split_tag = "2", "val"
        prefix = DATA / ".b6_plumb"
        if args.arms == list(ARMS):
            args.arms = ["a1"]      # a1 exercises every new path: features, dims, guard
    else:
        feature_dir = FEATURE_DIR
        if any(a in args.arms for a in ("a1", "a2")):
            if not (feature_dir / "feature_meta.json").exists():
                raise SystemExit("FATAL: no extracted features under "
                                 f"{feature_dir} -- run mus_features.py first (real data only).")
            meta = json.loads((feature_dir / "feature_meta.json").read_text(encoding="utf-8"))
            if meta.get("plumbing_smoke"):
                raise SystemExit("FATAL: the feature dir is a fabricated plumbing table; "
                                 "the real path refuses it.")
        pops = POPULATIONS
        folds = FOLDS
        fold_epochs, split_tag = FOLD_EPOCH_CAP, "test"
        prefix = DATA / ".b6"
    log = DATA / (".b6_plumb.log" if smoke else ".b6_batch.log")

    for pop, cut_ev in pops.items():
        control_metrics = prefix.parent / f"{prefix.name}_{pop}_a0_refit_s42" / "metrics.json"
        for arm in args.arms:
            if smoke and not control_metrics.exists():
                # plumbing only: self-stats are fine, pairing is not being judged
                control_metrics = (prefix.parent
                                   / f"{prefix.name}_{pop}_{arm}_refit_s42" / "metrics.json")
            env = base_env(arm, cut_ev, feature_dir)
            best = []
            for f in folds:
                tag = f"{pop}/{arm}/fold{f['fold']}"
                split_dir = prefix.parent / f"{prefix.name}_{pop}_{arm}_fold{f['fold']}_split"
                out_dir = prefix.parent / f"{prefix.name}_{pop}_{arm}_fold{f['fold']}"
                write_manifest(split_dir, f["train"], f["val"], TEST)
                if not train_run(tag, env, split_dir, out_dir, 42, 42,
                                 fold_epochs, False, log):
                    raise SystemExit(f"FATAL: {tag} failed -- see {log}")
                best.append(check_cap(out_dir, fold_epochs))
            median_epoch = int(statistics.median(best))
            print(f"[b6] {pop}/{arm}: fold best epochs {best} -> refit at {median_epoch}")

            refit_split = prefix.parent / f"{prefix.name}_{pop}_{arm}_refit_split"
            # Refit trains on the whole pool; the "val"列 is the pool itself, i.e. the
            # scheduler signal is in-train (no selection happens: fixed epochs).
            write_manifest(refit_split, POOL, POOL, TEST)
            for init in REFIT_SEEDS:
                tag = f"{pop}/{arm}/refit_s{init}"
                out_dir = prefix.parent / f"{prefix.name}_{pop}_{arm}_refit_s{init}"
                if not train_run(tag, env, refit_split, out_dir, 42, init,
                                 median_epoch, True, log):
                    raise SystemExit(f"FATAL: {tag} failed -- see {log}")
                if not eval_run(f"{tag}/eval", env, refit_split, out_dir,
                                control_metrics, split_tag, log):
                    raise SystemExit(f"FATAL: {tag} eval failed -- see {log}")
                if smoke:
                    (out_dir / "PLUMBING_SMOKE.json").write_text(
                        json.dumps({"warning": "machinery check only"}), encoding="utf-8")
            if smoke:
                break     # one arm chain per population is enough to prove the plumbing
    print("[b6] done")


if __name__ == "__main__":
    main()
