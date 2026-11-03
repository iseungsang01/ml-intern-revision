"""B.27b (PREREGISTRATION_B27.md sec. 7): pooled out-of-fold readout, post hoc.

Per population x repeat, concatenate the five TEST folds (every shot once) and put a paired
shot-cluster bootstrap CI on the pooled ratio skill. A cell beats iff the CI lower bound > 0
in both repeats. Reads data/.b27_seqv2_*/comparison_errors_test.npz; writes
data/.b27b_pooled.json. Run from the repo root.
"""
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "data"
REFS = ("pchip", "persistence", "gp_causal", "gp")
B, SEED, K = 10_000, 2727, 5


def pooled(pop, r, target):
    shot, sm, sb = [], [], {ref: [] for ref in REFS}
    for k in range(K):
        z = np.load(DATA / f".b27_seqv2_{pop}_r{r}_f{k}" / "comparison_errors_test.npz")
        shot.append(z[f"{target}_shot"]); sm.append(z[f"{target}_se_model"].astype(np.float64))
        for ref in REFS:
            sb[ref].append(z[f"{target}_se_{ref}"])
    shot = np.concatenate(shot)
    return shot, np.concatenate(sm), {ref: np.concatenate(v) for ref, v in sb.items()}


def shot_sums(shot, v):
    u, inv = np.unique(shot, return_inverse=True)
    return u, np.bincount(inv, weights=v, minlength=len(u))


def boot(shot, se_m, se_b, rng):
    u, m = shot_sums(shot, se_m)
    _, b = shot_sums(shot, se_b)
    idx = rng.integers(0, len(u), size=(B, len(u)))
    skills = 1.0 - m[idx].sum(1) / b[idx].sum(1)
    return float(1.0 - m.sum() / b.sum()), [float(x) for x in np.percentile(skills, [2.5, 97.5])], len(u)


def main():
    out = {}
    for pop in ("cut", "incl"):
        for target in ("CES_TI", "CES_VT"):
            data = [pooled(pop, r, target) for r in (0, 1)]
            for ref in REFS:
                reps = []
                for r, (shot, se_m, sb) in enumerate(data):
                    rng = np.random.default_rng(SEED)
                    s, ci, n = boot(shot, se_m, sb[ref], rng)
                    reps.append({"skill": s, "ci95": ci, "n_shots": n})
                beats = all(x["ci95"][0] > 0 for x in reps)
                loses = all(x["ci95"][1] < 0 for x in reps)
                verdict = "beats" if beats else "loses" if loses else "undecided"
                key = f"{pop}/{target}/{ref}"
                out[key] = {"repeats": reps, "verdict": verdict}
                print(f"{key:26s} " + "  ".join(
                    f"r{i} {x['skill']:+.3f} [{x['ci95'][0]:+.3f},{x['ci95'][1]:+.3f}] n={x['n_shots']}"
                    for i, x in enumerate(reps)) + f"  -> {verdict}")
    (DATA / ".b27b_pooled.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("wrote data/.b27b_pooled.json")


if __name__ == "__main__":
    main()
