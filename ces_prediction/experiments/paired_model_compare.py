"""Shot-clustered paired bootstrap between TWO model arms' comparison_errors npz files.

Arm A (variant) vs arm B (baseline model, e.g. the pinned iter009 run) must have been
scored by compare_baselines.py on the SAME split — same rows in the same order. This is
verified before any statistic is computed:
  - `{target}_shot` AND `{target}_dt_ms` arrays must be bit-identical (dt_ms derives
    from raw times only, so together they pin the exact row set independently of any
    normalization stats), and
  - `{target}_se_pchip` must match to within float32 round-trip tolerance. When the two
    arms trained with different normalization stats (e.g. the stuck-free arm), the
    physical target y = denorm(norm(raw)) differs at the last-ulp level, which the
    (pchip - y)^2 gradient amplifies to O(1e-4) relative — legitimate pairing, not a
    population mismatch. Bit-exact equality still passes trivially.

Reports skill_A_vs_B = 1 - sum(SE_A)/sum(SE_B) (> 0 means the variant beats the baseline
model) with the same B=10000 / seed 12345 shot-clustered bootstrap used by the PR4 gate
(reuses bootstrap_compare._bootstrap — never hand-roll the eval loop or the statistic).

Usage (repo root):
  py ces_prediction/experiments/paired_model_compare.py \
      --a data/.anchor_s42/comparison_errors_test.npz \
      --b data/.vt_repro_out/comparison_errors_test.npz \
      --out data/.anchor_s42/paired_vs_iter009.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # ces_prediction/
from bootstrap_compare import BOOTSTRAP_SEED, TARGET_NAMES, _bootstrap  # noqa: E402


def compare(path_a, path_b):
    a = np.load(path_a)
    b = np.load(path_b)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    result = {}
    for name in TARGET_NAMES:
        shot_a, shot_b = a[f"{name}_shot"], b[f"{name}_shot"]
        if shot_a.shape != shot_b.shape or not np.array_equal(shot_a, shot_b):
            raise SystemExit(
                f"FATAL: {name} shot arrays differ ({shot_a.shape} vs {shot_b.shape}) — "
                f"arms are not paired; refusing to compare."
            )
        dt_a, dt_b = a[f"{name}_dt_ms"], b[f"{name}_dt_ms"]
        if not np.array_equal(dt_a, dt_b, equal_nan=True):
            raise SystemExit(
                f"FATAL: {name} dt_ms arrays differ -- different row sets; refusing to compare."
            )
        pchip_a, pchip_b = a[f"{name}_se_pchip"], b[f"{name}_se_pchip"]
        if not np.array_equal(pchip_a, pchip_b):
            # dt_ms already pins the row set; se_pchip may differ by the float32
            # normalization round-trip of y when the arms trained with different
            # stats. Accept ulp-level discrepancy, refuse anything larger.
            denom = np.maximum(np.abs(pchip_b), 1.0)
            rel = float(np.max(np.abs(pchip_a - pchip_b) / denom))
            if not np.allclose(pchip_a, pchip_b, rtol=1e-3, atol=1e-2):
                raise SystemExit(
                    f"FATAL: {name} se_pchip differs beyond stats round-trip tolerance "
                    f"(max rel diff={rel:.3e}) -- different eval populations; refusing to compare."
                )
            print(f"[paired] {name}: se_pchip within round-trip tolerance "
                  f"(max rel diff={rel:.3e}); arms use different normalization stats.")
        res = _bootstrap(shot_a, a[f"{name}_se_model"], b[f"{name}_se_model"], rng)
        res["n_rows"] = int(len(shot_a))
        res["a_better"] = bool(res["skill_ci95"][0] > 0.0)
        res["b_better"] = bool(res["skill_ci95"][1] < 0.0)
        result[name] = res
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="variant comparison_errors npz")
    ap.add_argument("--b", required=True, help="baseline-model comparison_errors npz")
    ap.add_argument("--out", required=True, help="output json path")
    args = ap.parse_args()

    result = {
        "arm_a": str(args.a),
        "arm_b": str(args.b),
        "statistic": "skill_A_vs_B = 1 - sum(SE_A)/sum(SE_B); > 0 means A (variant) better",
        "bootstrap": {"B": 10000, "seed": BOOTSTRAP_SEED, "cluster": "shot"},
        "targets": compare(args.a, args.b),
    }
    for name, res in result["targets"].items():
        verdict = "A better" if res["a_better"] else ("B better" if res["b_better"] else "n.s.")
        print(
            f"{name}: skill_A_vs_B={res['skill_point']:+.4f} "
            f"CI95=[{res['skill_ci95'][0]:+.4f}, {res['skill_ci95'][1]:+.4f}] {verdict} "
            f"(n_rows={res['n_rows']}, n_shots={res['n_shots']})"
        )
    Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
