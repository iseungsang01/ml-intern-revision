"""How much of §8ac's reach deficit was missing information, and how much was cold start?

§8ac drew its reach ladder by TRUNCATING one trained model: the recurrent state is reset to
zero `ctx` steps before every scored row. That measures what *this* model loses when its
state is thrown away — which is not the same quantity as what a model *trained* at that
reach can do. §8ac said so in "What it does not show" ("part of the short-ctx deficit is
warm-up rather than missing information — the two cannot be separated by this design"), but
its Verdict 1 then read the ctx = 2 rung as an information statement.

The separation was already possible with artifacts on disk. `data/.b1_w2cut_s{seed}` is a
model **trained at reach 2** (the B.1 window control, §8x), scored by the same
`compare_baselines` path on the **same rows** as `data/.reach_s{seed}`. Three points per
target then bracket the question:

    full        backbone, whole-block inference          (.reach_s*  se_model)
    trained@2   window iter009 W = 2, trained at reach 2 (.b1_w2cut_s* se_model)
    trunc@2     backbone with the state cut to 2 steps   (.reach_s*  se_model_ctx2)

    truncation deficit = full - trunc@2          (what §8ac's ladder reported)
    recovered by training = trained@2 - trunc@2  (what training at that reach buys back)
    genuine reach value = full - trained@2       (what whole-block context is actually worth)

`CES_VT` is the internal control. §8ab's routing result says the `V_rot` branch never reads
the fast diagnostics and rides carried input channels that `build_blocks` computes over the
whole block regardless of truncation — so cutting the recurrent state should cost `V_rot`
almost nothing, and any warm-up interpretation of the `CES_TI` gap has to survive that check.

**Confound, stated up front and not resolved here.** `trained@2` is a different
ARCHITECTURE (window `iter009`), not `seq_v2` retrained with a 2-step state. It shares the
reach and the information, not the structure, so this script bounds the warm-up share rather
than measuring it exactly. The clean arm is `seq_v2` trained with the state reset every
`ctx` steps (pre-registered as B.9's reach ladder); until that runs, quote this as
"a model trained at this reach reaches X", never as "seq_v2 at reach 2 reaches X".

No retraining, no new scoring: every number below is a re-read of frozen npz files.

Usage (repo root):  py ces_prediction/experiments/reach/trained_vs_truncated.py
Writes: data/.reach_trained_vs_truncated.json
"""

import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
CES_DIR = REPO_ROOT / "ces_prediction"
DATA = REPO_ROOT / "data"
sys.path.insert(0, str(CES_DIR))

from bootstrap_compare import BOOTSTRAP_SEED, TARGET_NAMES, _bootstrap  # noqa: E402

SEEDS = (42, 1, 7, 123)
CTX = 2                    # the rung §8ac's Verdict 1 rests on
REF = "persistence"        # the arm the claim displaces on the 10 ms grid


def paired_arrays(seed):
    """The two frozen npz files, verified to describe the same rows before anything is read.

    `shot` + `dt_ms` pin the row set independently of normalization (the rule
    `paired_model_compare` enforces); `y_true` and `se_persistence` are checked bit-exact on
    top, because both arms were scored by the same `compare_baselines` path and any
    difference at all would mean the populations diverged.
    """
    trained = np.load(DATA / f".b1_w2cut_s{seed}" / "comparison_errors_test.npz")
    reach = np.load(DATA / f".reach_s{seed}" / "comparison_errors_test.npz")
    for t in TARGET_NAMES:
        for key in ("shot", "dt_ms", "y_true", "se_" + REF):
            a, b = trained[f"{t}_{key}"], reach[f"{t}_{key}"]
            if a.shape != b.shape or not np.array_equal(a, b, equal_nan=True):
                raise SystemExit(
                    f"FATAL: seed {seed} {t}_{key} differs between .b1_w2cut_s{seed} and "
                    f".reach_s{seed} — the arms are not paired; refusing to compare."
                )
    return trained, reach


def analyze(seed):
    trained, reach = paired_arrays(seed)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    per_target = {}
    for t in TARGET_NAMES:
        shot = reach[f"{t}_shot"]
        se_ref = reach[f"{t}_se_{REF}"]
        se = {"full": reach[f"{t}_se_model"],
              "trained": trained[f"{t}_se_model"],
              "trunc": reach[f"{t}_se_model_ctx{CTX}"]}
        mse_ref = float(np.mean(se_ref))
        skill = {k: 1.0 - float(np.mean(v)) / mse_ref for k, v in se.items()}

        deficit = skill["full"] - skill["trunc"]
        recovered = skill["trained"] - skill["trunc"]
        per_target[t] = {
            "n_rows": int(len(shot)),
            "n_shots": int(len(np.unique(shot))),
            f"skill_vs_{REF}": skill,
            "truncation_deficit": deficit,
            "recovered_by_training": recovered,
            "genuine_reach_value": skill["full"] - skill["trained"],
            "warmup_share": (recovered / deficit) if abs(deficit) > 1e-9 else float("nan"),
            # skill_A_vs_B > 0 means A has the lower error.
            "paired_trained_vs_trunc": _bootstrap(shot, se["trained"], se["trunc"], rng),
            "paired_full_vs_trained": _bootstrap(shot, se["full"], se["trained"], rng),
        }
    return per_target


def main():
    per_seed = {}
    for seed in SEEDS:
        per_seed[str(seed)] = analyze(seed)
        print(f"[tvt] seed {seed} ok", flush=True)

    means = {}
    for t in TARGET_NAMES:
        rows = [per_seed[str(s)][t] for s in SEEDS]
        means[t] = {
            f"skill_vs_{REF}": {k: float(np.mean([r[f"skill_vs_{REF}"][k] for r in rows]))
                                for k in ("full", "trained", "trunc")},
            "truncation_deficit": float(np.mean([r["truncation_deficit"] for r in rows])),
            "recovered_by_training": float(np.mean([r["recovered_by_training"] for r in rows])),
            "genuine_reach_value": float(np.mean([r["genuine_reach_value"] for r in rows])),
            "trained_beats_trunc_sig": sum(1 for r in rows
                                           if r["paired_trained_vs_trunc"]["skill_ci95"][0] > 0),
            "full_beats_trained_sig": sum(1 for r in rows
                                          if r["paired_full_vs_trained"]["skill_ci95"][0] > 0),
        }
        d, rec = means[t]["truncation_deficit"], means[t]["recovered_by_training"]
        means[t]["warmup_share"] = (rec / d) if abs(d) > 1e-9 else float("nan")

    summary = {
        "question": ("how much of the sec.8ac ctx=2 deficit is missing information vs "
                     "cold-start warm-up?"),
        "protocol": {
            "retrained": False,
            "ctx": CTX,
            "reference": REF,
            "seeds": list(SEEDS),
            "trained_at_reach_arm": "window iter009 W=2 (.b1_w2cut_s*) -- DIFFERENT "
                                    "architecture; bounds the warm-up share, does not "
                                    "measure it (see module docstring)",
            "statistic": ("shot-clustered paired bootstrap B=10000 seed 12345 "
                          "(bootstrap_compare._bootstrap)"),
        },
        "per_seed": per_seed,
        "mean_over_seeds": means,
    }
    out = DATA / ".reach_trained_vs_truncated.json"
    out.write_text(json.dumps(summary, indent=1), encoding="utf-8")

    print("\n" + "=" * 92)
    print(f"skill vs {REF}, mean over {len(SEEDS)} splits (identical rows, no retraining)")
    print("target".ljust(9) + "full".rjust(9) + "trained@2".rjust(11) + "trunc@2".rjust(9)
          + "deficit".rjust(10) + "recovered".rjust(11) + "warm-up%".rjust(10)
          + "reach worth".rjust(13))
    for t in TARGET_NAMES:
        m = means[t]
        s = m[f"skill_vs_{REF}"]
        print(t.ljust(9) + f"{s['full']:+.3f}".rjust(9) + f"{s['trained']:+.3f}".rjust(11)
              + f"{s['trunc']:+.3f}".rjust(9) + f"{m['truncation_deficit']:+.3f}".rjust(10)
              + f"{m['recovered_by_training']:+.3f}".rjust(11)
              + f"{100 * m['warmup_share']:.0f}%".rjust(10)
              + f"{m['genuine_reach_value']:+.3f}".rjust(13))
    print(f"\n[tvt] wrote {out}")


if __name__ == "__main__":
    main()
