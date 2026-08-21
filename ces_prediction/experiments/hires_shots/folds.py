# -*- coding: utf-8 -*-
"""Roles for the twelve microsecond-fetch shots (frozen 2026-08-21).

Twelve shots are requested; ten carry a learning role. This file replaces the 2026-08-17
structure (3 test / 7 pool) after the full 641-shot literature scan completed and 승상님
made two decisions on 2026-08-21: the test set grows to FOUR, and literature shot #32092
enters the pool. The selection itself is literature-first (2026-08-20 decision): published
discharges take slots even when they fail the data gate, and `score_v2` fills the rest.

    test      = 4 shots, frozen, never trained on and never used for model selection
    pool      = 6 shots, rotated leave-one-shot-out
    fold      = train 5 / val 1, six folds, each pool shot is the val shot exactly once
    companion = 2 shots, in neither -- see below

Why test grew from 3 to 4: literature-first keeps #31873 (ELM-suppression paper, s42-test)
in the test set, and its rotation channel is held for the whole discharge (1 independent
V_rot row) -- so with 3 test shots CES_VT has only TWO effective bootstrap clusters. At
k = 2 the resample space collapses and the measured pass rate RISES from k = 3 to k = 2
(0.665 -> 0.770 for CES_VT): false positives, not power. Adding the best V_rot-carrying
s42-test shot (#31902, 412 rows) restores three effective clusters and lifts measured
power to 0.750 (CES_VT) / 0.368 (CES_TI). The price is one pool shot (7 -> 6): data shot
#31914 (542 V_rot rows) lost its slot. The B.6 preregistration's §1.2 eligibility gate
(>= 3 effective V_rot test clusters) PASSES under this structure: 31921 (296), 31114
(311), 31902 (412).

Why val is 1 rather than 2: with a 6-shot pool, val = 2 costs a training shot (train
would drop to 4). A single val shot is a noisy early-stopping signal on its own, so the
protocol is: run all six folds, take the MEDIAN stopping epoch across folds, and refit on
all six pool shots at that epoch. The fold-to-fold spread of the val metric is itself the
model-selection stability estimate.

Why the two companions are held out of BOTH: each is the same-session partner of a shot
that already has a role (31923 of test shot 31921, 31357 of pool shot 31359). Session
similarity measured that adjacent shots share plasma setup and diagnostic gain/offset, so
putting a companion in train while its partner is test would put near-identical
calibration on both sides of the split. They are acquisition targets for differential
physics, not extra training data, and they never enter the bootstrap either.

The one-shot-per-session rule was REMOVED by 승상님 on 2026-08-20 for role holders. The
frozen list contains one close pair inside the pool -- #32092 and #32097, gap 5 (the
3-to-6-gap band sits at a median calibration distance of 1.49 against 4.28 for random
pairs) -- which is acceptable because both are on the SAME side of every split: session
calibration can leak train->val inside the rotation, never across the test boundary. A
targeted assert below keeps any close pair from straddling test.

Bootstrap policy for the final test evaluation:
    primary   -- shot-clustered (identical to every other batch in THESIS_RESULTS.md §8)
    secondary -- shot x 500 ms block clusters, pre-registered, reported alongside
Both are always reported together, so a reader can see the block assumption did not
manufacture the conclusion.
"""
from __future__ import annotations

# Frozen s42 split membership is kept for every shot (test shots are s42-test members).
TEST = (31921, 31873, 31114, 31902)                  # V_rot rows: 296 / 1 / 311 / 412
POOL = (31097, 31359, 31747, 32027, 32092, 32097)    # 5 literature + 1 data (score_v2)

# Same-session partners of 31921 and 31359. Fetched, never trained on, never in the gate.
COMPANIONS = {31923: 31921, 31357: 31359}

FOLDS = tuple({"fold": i + 1, "val": (v,), "train": tuple(s for s in POOL if s != v)}
              for i, v in enumerate(POOL))

BOOTSTRAP = {
    "primary": {"cluster": "shot", "B": 10000, "seed": 12345},
    "secondary": {"cluster": "shot_x_500ms_block", "B": 10000, "seed": 12345},
    "report": "both, always",
}

# Measured power (power_analysis.py, seq_v2 vs W=2 control replayed on the real 96-shot
# test set). Recorded here so nobody re-derives the sizing by intuition. k = 4 is the
# frozen structure; k = 3 is kept to show what the extra shot bought.
POWER_AT_K4_SHOT = {"CES_TI": 0.368, "CES_VT": 0.750}
POWER_AT_K3_SHOT = {"CES_TI": 0.287, "CES_VT": 0.682}
POWER_AT_K3_BLOCK = {"CES_TI": 0.412, "CES_VT": 0.685}   # block sweep has no k=4 row


def _check():
    assert len(TEST) == 4 and len(POOL) == 6, "learning roles: 4 test + 6 pool"
    assert len(COMPANIONS) == 2, "12 shots requested: 10 with roles + 2 companions"
    assert not (set(TEST) & set(POOL)), "test must be disjoint from the rotation pool"
    assert not (set(COMPANIONS) & (set(TEST) | set(POOL))), \
        "a companion must not also hold a learning role"
    for comp, partner in COMPANIONS.items():
        assert partner in TEST or partner in POOL, "a companion pairs with a role-holding shot"
        assert abs(comp - partner) <= 2, "a companion is a same-session partner"
    seen = [f["val"][0] for f in FOLDS]
    assert sorted(seen) == sorted(POOL), "each pool shot is the val shot exactly once"
    for f in FOLDS:
        assert len(f["train"]) == 5, "train is 5 shots in every fold"
        assert not (set(f["train"]) & set(f["val"])), "no shot is both train and val"
        assert not (set(f["train"]) & set(TEST)), "test never leaks into train"
        assert not (set(f["train"]) & set(COMPANIONS)), "companions never enter train"
    # B.6 §1.2 eligibility gate: >= 3 test shots with >= 200 valid independent V_rot rows.
    VT_ROWS = {31921: 296, 31873: 1, 31114: 311, 31902: 412}
    assert sum(VT_ROWS[s] >= 200 for s in TEST) >= 3, \
        "CES_VT confirmatory verdicts need >= 3 effective test clusters"
    # Session leakage may never straddle the test boundary (the in-pool 32092/32097 pair
    # is deliberate; the blanket one-per-session rule was removed 2026-08-20).
    for t in TEST:
        for s in POOL:
            assert abs(t - s) > 6, f"same-session pair across the test boundary: {t}/{s}"


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
    print("companions (fetched, never trained on, never in the gate): "
          + "  ".join(f"{c} (with {p})" for c, p in COMPANIONS.items()))
    print(f"bootstrap: primary={BOOTSTRAP['primary']['cluster']}, "
          f"secondary={BOOTSTRAP['secondary']['cluster']} (both reported)")
    print(f"measured power at k=4 (shot clusters): {POWER_AT_K4_SHOT}")
    print(f"total shots to request: {len(TEST) + len(POOL) + len(COMPANIONS)}")
