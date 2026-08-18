# -*- coding: utf-8 -*-
"""Roles for the twelve microsecond-fetch shots.

Twelve shots are requested; only ten of them carry a learning role. The structure below
was fixed 2026-08-17 after the power analysis in `power_analysis.py` and is NOT changed by
the two shots added on 2026-08-18:

    test      = 3 shots, frozen, never trained on and never used for model selection
    pool      = 7 shots, rotated leave-one-shot-out
    fold      = train 6 / val 1, seven folds, each pool shot is the val shot exactly once
    companion = 2 shots, in neither -- see below

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

Why the two companions are held out of BOTH: each is the same-session partner of a shot
that already has a role (31923 of test shot 31921, 31357 of pool shot 31359). Screen 3
measured that adjacent shots share plasma setup and diagnostic gain/offset, so putting a
companion in train while its partner is test would put near-identical calibration on both
sides of the split -- the exact leakage screen 3 exists to prevent. They are acquisition
targets for differential physics, not extra training data, and they never enter the
bootstrap either: two of k test clusters drawn from one session is the k=2 artifact again.

Why they are worth fetching anyway: the redundancy that demoted them was measured on the
100 Hz grid. The published difference between 31921 and 31923 is a weakly coherent mode at
~50 kHz -- three orders of magnitude above that grid's Nyquist frequency. "Redundant in the
band we already have" is not "redundant in the band we are buying". Holding the pair also
turns screen 3's inference into a measurement: with both members in hand the size of the
session-calibration leakage can be quantified rather than assumed.

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

# Same-session partners of 31921 and 31359. Fetched, never trained on, never in the gate.
COMPANIONS = {31923: 31921, 31357: 31359}

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
    assert len(TEST) == 3 and len(POOL) == 7, "learning roles: 3 test + 7 pool"
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
        assert len(f["train"]) == 6, "train is 6 shots in every fold"
        assert not (set(f["train"]) & set(f["val"])), "no shot is both train and val"
        assert not (set(f["train"]) & set(TEST)), "test never leaks into train"
        assert not (set(f["train"]) & set(COMPANIONS)), "companions never enter train"
    # The one-shot-per-session rule applies to the role-holding ten only; the companions
    # are deliberate exceptions and are excluded from this check by construction.
    roles = sorted(TEST + POOL)
    gaps = [b - a for a, b in zip(roles, roles[1:])]
    assert min(gaps) >= 7, f"same-session pair among role holders (min gap {min(gaps)})"


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
    roles = sorted(TEST + POOL)
    print(f"smallest shot-number gap among role holders: "
          f"{min(b - a for a, b in zip(roles, roles[1:]))}")
    print(f"total shots to request: {len(TEST) + len(POOL) + len(COMPANIONS)}")
