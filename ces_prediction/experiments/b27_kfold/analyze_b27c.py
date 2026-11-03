"""B.27c (PREREGISTRATION_B27.md sec. 8): two-level (model x shot) bootstrap on the new repeats.

Primary set = repeats r2..r9 (never seen before this rule was written). Each repeat's five TEST
folds are pooled (every shot once). One bootstrap draw = pick one repeat uniformly (training
variation: a different trained model set on a different TRAIN partition) and resample that
repeat's shots with replacement (shot sampling); skill = 1 - sum SE_model / sum SE_ref.
beats iff the 2.5th percentile > 0, loses iff the 97.5th < 0, else undecided.
Secondary: the same on all ten repeats; per-repeat pooled skills.
Reads data/.b27_seqv2_*/comparison_errors_test.npz, writes data/.b27c_hier.json.
"""
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "data"
REFS = ("pchip", "persistence", "gp_causal", "gp")
B, SEED, K = 20_000, 2728, 5
PRIMARY, ALL = tuple(range(2, 10)), tuple(range(10))


def shot_sums(pop, r, target, ref):
    shot, m, b = [], [], []
    for k in range(K):
        z = np.load(DATA / f".b27_seqv2_{pop}_r{r}_f{k}" / "comparison_errors_test.npz")
        shot.append(z[f"{target}_shot"])
        m.append(z[f"{target}_se_model"].astype(np.float64))
        b.append(z[f"{target}_se_{ref}"])
    shot, m, b = np.concatenate(shot), np.concatenate(m), np.concatenate(b)
    u, inv = np.unique(shot, return_inverse=True)
    return np.bincount(inv, weights=m, minlength=len(u)), np.bincount(inv, weights=b, minlength=len(u))


def hier(sums, repeats, rng):
    draws = rng.choice(np.array(repeats), size=B)
    out = np.empty(B)
    for r in repeats:
        sel = np.flatnonzero(draws == r)
        m, b = sums[r]
        idx = rng.integers(0, len(m), size=(len(sel), len(m)))
        out[sel] = 1.0 - m[idx].sum(1) / b[idx].sum(1)
    lo, hi = np.percentile(out, [2.5, 97.5])
    verdict = "beats" if lo > 0 else "loses" if hi < 0 else "undecided"
    return {"ci95": [float(lo), float(hi)], "median": float(np.median(out)), "verdict": verdict}


def main():
    out = {}
    for pop in ("cut", "incl"):
        for target in ("CES_TI", "CES_VT"):
            for ref in REFS:
                sums = {r: shot_sums(pop, r, target, ref) for r in ALL}
                per_rep = {r: float(1.0 - sums[r][0].sum() / sums[r][1].sum()) for r in ALL}
                prim = hier(sums, PRIMARY, np.random.default_rng(SEED))
                sec = hier(sums, ALL, np.random.default_rng(SEED))
                key = f"{pop}/{target}/{ref}"
                out[key] = {"primary_r2_r9": prim, "secondary_all10": sec, "per_repeat_pooled": per_rep}
                pr = " ".join(f"{per_rep[r]:+.2f}" for r in ALL)
                print(f"{key:26s} PRIMARY [{prim['ci95'][0]:+.3f},{prim['ci95'][1]:+.3f}] -> {prim['verdict']:9s}"
                      f" | all10 [{sec['ci95'][0]:+.3f},{sec['ci95'][1]:+.3f}] {sec['verdict']:9s} | per-repeat {pr}")
    (DATA / ".b27c_hier.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("wrote data/.b27c_hier.json")


if __name__ == "__main__":
    main()
