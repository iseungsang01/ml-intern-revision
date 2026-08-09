"""Shared plumbing for the controlled-experiment runners under `experiments/`.

Every batch pairs its arm against the same frozen baseline family, so the split
directories, the control output directories, and the training env must be *identical*
across batches — that is the whole point of the paired protocol. Keeping them in one
module is what makes "same split, same caps, same treatment" checkable instead of
hopeful.

Two standing rules live here by construction (`experiments/README.md`):

1. **Pin the data treatment explicitly.** `FULL_ENV` deliberately omits
   `CES_DROP_STUCK_TARGETS`; each runner sets it in its own env dict so the choice is
   visible at the call site. A silent fallback to `train.py`'s default produced a
   window-sweep conclusion that was pure artifact (`THESIS_RESULTS.md` §8f).
2. **Never mutate tracked source.** Architecture variants are selected with
   `CES_MODEL_FILE` in the subprocess env (see `ces_prediction/model.py`); nothing
   copies a variant over `model.py`.
"""

import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CES_DIR = REPO_ROOT / "ces_prediction"
EXP_DIR = Path(__file__).resolve().parent
PAIRED_MODEL_COMPARE = EXP_DIR / "paired_model_compare.py"

SEEDS = (42, 1, 7, 123)

# Frozen split directories of the baseline family (docs/재현_절차_기록.md §6.5/§6.9).
SPLIT_SRC = {
    42: REPO_ROOT / "data" / ".improve_split" / "w4",
    1: REPO_ROOT / "data" / ".ms_split_1",
    7: REPO_ROOT / "data" / ".ms_split_7",
    123: REPO_ROOT / "data" / ".ms_split_123",
}
# Frozen run directories of the held-kept control arm.
BASELINE_OUT = {
    42: REPO_ROOT / "data" / ".vt_repro_out",
    1: REPO_ROOT / "data" / ".vt_repro_ms_1",
    7: REPO_ROOT / "data" / ".vt_repro_ms_7",
    123: REPO_ROOT / "data" / ".vt_repro_ms_123",
}

# Matches the trusted baseline family exactly: any cap or window drift regenerates the
# splits and silently breaks pairing, so pin everything.
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
}
SMOKE_ENV = {
    **FULL_ENV,
    "CES_EPOCHS": "1",
    "CES_MAX_TRAIN_SAMPLES": "4000",
    "CES_MAX_VAL_SAMPLES": "1200",
    "CES_MAX_TEST_SAMPLES": "1200",
}


def run_step(cmd, env, log_path, label):
    start = time.time()
    with open(log_path, "a", encoding="utf-8") as log:
        log.write(f"\n===== {label}: {' '.join(map(str, cmd))} =====\n")
        log.flush()
        proc = subprocess.run(
            [sys.executable, *map(str, cmd)],
            cwd=REPO_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
        )
    mins = (time.time() - start) / 60.0
    print(f"[runner]   {label}: rc={proc.returncode} ({mins:.1f} min)")
    return proc.returncode == 0


def prepare_split_copy(seed, dst, smoke):
    """Copy a frozen baseline split so `train.py` can rewrite its manifest in place."""
    if smoke:
        # Smoke uses its own tiny-cap splits; never mix with the paired full splits.
        if dst.exists():
            shutil.rmtree(dst)
        dst.mkdir(parents=True)
        return
    src = SPLIT_SRC[seed]
    if not (src / "split_manifest.json").exists():
        raise SystemExit(f"FATAL: baseline split dir missing: {src}")
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
