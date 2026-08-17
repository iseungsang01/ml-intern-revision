# -*- coding: utf-8 -*-
"""Train/val rotation for the ten microsecond-fetch shots.

Structure (fixed 2026-08-17, after the power analysis in `power_analysis.py`):

    test  = 3 shots, frozen, never trained on and never used for model selection
    pool  = the remaining 7 shots, rotated leave-one-shot-out
    fold  = train 6 / val 1, seven folds, each pool shot is the val shot exactly once

Why test stays at 3 rather than dropping to 2: the gate resamples SHOTS, so k test shots
means k bootstrap clusters. At k = 2 the resample space is four draws, half of which repeat
the same shot, and the CI collapses -- measured pass rate goes UP from k=3 to k=2 (28.7 % ->
40.5 % for CES_TI) even though nothing improved. That extra pass rate is false positives,
not power. k = 3 is the smallest size that does not sit on that artifact.

Why val is 1 rather than 2: with a 7-shot pool, val = 2 costs a training shot (train would
drop to 5). A single val shot is a noisy early-stopping signal on its own, so the protocol
is: run all seven folds, take the MEDIAN stopping epoch across folds, and refit on all
seven pool shots at that epoch. The fold-to-fold spread of the val metric is itself the
model-selection stability estimate.

Bootstrap policy for the final test evaluation:
    primary   -- shot-clustered (identical to every other batch in THESIS_RESULTS.md §8)
    secondary -- shot x 500 ms block clusters, pre-registered, reported alongside
Both are always reported together, so a reader can see the block assumption did not
manufacture the conclusion.
"""
from __future__ import annotations

# Frozen seed-42 split membership; none of these roles were reassigned.
TEST = (31921, 31873, 31114)                                     # s42 test
POOL = (31359, 32027, 32097, 31745, 31604, 31074, 31937)         # s42 val (4) + train (3)

FOLDS = tuple({"fold": i + 1, "val": (v,), "train": tuple(s for s in POOL if s != v)}
              for i, v in enumerate(POOL))

BOOTSTRAP = {
    "primary": {"cluster": "shot", "B": 10000, "seed": 12345},
    "secondary": {"cluster": "shot_x_500ms_block", "B": 10000, "seed": 12345},
    "report": "both, always",
}

# Measured power at these sizes (power_analysis.py, seq_v2 vs W=2 control on the real
# 96-shot test set). Recorded here so nobody re-derives the sizing by intuition.
POWER_AT_K3_SHOT = {"CES_TI": 0.287, "CES_VT": 0.682}
POWER_AT_K3_BLOCK = {"CES_TI": 0.412, "CES_VT": 0.685}


def _check():
    assert len(TEST) == 3 and len(POOL) == 7, "10 shots: 3 test + 7 pool"
    assert not (set(TEST) & set(POOL)), "test must be disjoint from the rotation pool"
    seen = [f["val"][0] for f in FOLDS]
    assert sorted(seen) == sorted(POOL), "each pool shot is the val shot exactly once"
    for f in FOLDS:
        assert len(f["train"]) == 6, "train is 6 shots in every fold"
        assert not (set(f["train"]) & set(f["val"])), "no shot is both train and val"
        assert not (set(f["train"]) & set(TEST)), "test never leaks into train"
    allshots = sorted(TEST + POOL)
    gaps = [b - a for a, b in zip(allshots, allshots[1:])]
    assert min(gaps) >= 7, f"same-session pair present (min gap {min(gaps)})"


_check()


if __name__ == "__main__":
    print(f"test (frozen, {len(TEST)}): " + "  ".join(str(s) for s in TEST))
    print(f"pool ({len(POOL)}, rotated leave-one-out): " + "  ".join(str(s) for s in POOL))
    print()
    for f in FOLDS:
        print(f"  fold {f['fold']}  val={f['val'][0]}   "
              f"train={' '.join(str(s) for s in f['train'])}")
    print(f"\ntrain {len(FOLDS[0]['train'])} / val {len(FOLDS[0]['val'])} / test {len(TEST)}"
          f"   ({len(FOLDS)} folds)")
    print(f"bootstrap: primary={BOOTSTRAP['primary']['cluster']}, "
          f"secondary={BOOTSTRAP['secondary']['cluster']} (both reported)")
    allshots = sorted(TEST + POOL)
    print(f"smallest shot-number gap: "
          f"{min(b - a for a, b in zip(allshots, allshots[1:]))}")
