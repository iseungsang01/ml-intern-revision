"""Analyze the GP baseline arm added by rerun_compare_gp.py.

Questions answered per seed (genuine treatment, headline population):
1. Where does GP land in the RMSE ladder? (vs linear/pchip/persistence)
2. Does the model beat GP? (shot-clustered paired bootstrap, PR4 convention:
   B=10,000, seed 12345, PASS iff 95% skill CI lower bound > 0)
3. Is GP stronger or weaker than the pre-registered headline baseline PCHIP?
   (paired gp-vs-pchip bootstrap)

Cross-check: the model-vs-pchip skill recomputed here must match the recorded
headline (paper_numbers.json genuine per-seed values) -- printed for eyeballing.

Output: data/.gp_analysis.json
Usage (repo root):  py ces_prediction/experiments/gp/analyze_gp.py
"""

import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
CES_DIR = REPO_ROOT / "ces_prediction"
sys.path.insert(0, str(CES_DIR / "experiments"))
from runner_common import BASELINE_OUT  # noqa: E402

SEEDS = (42, 1, 7, 123)
TARGETS = ("CES_TI", "CES_VT")
NPZ_NAME = "comparison_errors_test__test_genuine_gp.npz"
B = 10000
BOOTSTRAP_SEED = 12345


def _per_shot_sums(shot, values):
    shots, inv = np.unique(shot, return_inverse=True)
    sums = np.zeros(len(shots), dtype=np.float64)
    np.add.at(sums, inv, values)
    counts = np.zeros(len(shots), dtype=np.float64)
    np.add.at(counts, inv, 1.0)
    return shots, sums, counts


def _bootstrap(shot, se_a, se_b, rng):
    """95% CI of skill = 1 - MSE_a/MSE_b under whole-shot resampling.
    PASS iff CI lower bound > 0 (arm a beats arm b)."""
    _, sa, cnt = _per_shot_sums(shot, se_a)
    _, sb, _ = _per_shot_sums(shot, se_b)
    n = len(sa)
    skill_point = (1.0 - sa.sum() / sb.sum()) if sb.sum() > 0 else float("nan")
    idx = rng.integers(0, n, size=(B, n))
    skill = 1.0 - sa[idx].sum(axis=1) / sb[idx].sum(axis=1)
    finite = np.isfinite(skill)
    lo, hi = (np.percentile(skill[finite], [2.5, 97.5]) if finite.any()
              else (float("nan"), float("nan")))
    return {
        "n_shots": int(n),
        "skill_point": float(skill_point),
        "skill_ci95": [float(lo), float(hi)],
        "pass": bool(lo > 0.0),
    }


def main():
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    out = {"npz": NPZ_NAME, "bootstrap_resamples": B, "seed": BOOTSTRAP_SEED,
           "per_seed": {}}
    for seed in SEEDS:
        path = BASELINE_OUT[seed] / NPZ_NAME
        if not path.exists():
            raise SystemExit(f"FATAL: missing {path} -- run rerun_compare_gp.py first")
        d = np.load(path)
        rec = {}
        for name in TARGETS:
            shot = d[f"{name}_shot"]
            se = {arm: d[f"{name}_se_{arm}"]
                  for arm in ("model", "gp", "pchip", "linear", "persistence")}
            rmse = {arm: float(np.sqrt(np.mean(v))) for arm, v in se.items()}
            rec[name] = {
                "n": int(len(shot)),
                "rmse": rmse,
                "model_vs_gp": _bootstrap(shot, se["model"], se["gp"], rng),
                "gp_vs_pchip": _bootstrap(shot, se["gp"], se["pchip"], rng),
                "model_vs_pchip_crosscheck": _bootstrap(shot, se["model"], se["pchip"], rng),
            }
            # Where could the tie break? GP is a smoother: the model's edge (if
            # any) should sit where smoothing fails -- the input-only peak
            # subset -- and GP's edge where its future anchor helps most (large
            # dt). Both subsets ship in the npz, so measure rather than assert.
            for label, mask in (
                ("peak", d[f"{name}_is_peak"].astype(bool)),
                ("dt_le_15ms", d[f"{name}_dt_ms"] <= 15.0),
                ("dt_gt_15ms", d[f"{name}_dt_ms"] > 15.0),
            ):
                if int(mask.sum()) >= 200:
                    r = _bootstrap(shot[mask], se["model"][mask], se["gp"][mask], rng)
                    r["n"] = int(mask.sum())
                    rec[name][f"model_vs_gp_{label}"] = r
        out["per_seed"][str(seed)] = rec

    print(f"\n=== GP baseline arm (genuine, B={B}) ===")
    for name in TARGETS:
        print(f"\n{name}")
        print(f"  {'seed':>5} {'RMSE gp':>9} {'pchip':>8} {'linear':>8} {'model':>8}"
              f"  {'model vs gp':>26}  {'gp vs pchip':>26}")
        for seed in SEEDS:
            r = out["per_seed"][str(seed)][name]
            mg, gp_ = r["model_vs_gp"], r["gp_vs_pchip"]
            def fmt(x):
                return (f"{x['skill_point']:+.3f} [{x['skill_ci95'][0]:+.3f},"
                        f"{x['skill_ci95'][1]:+.3f}]{' PASS' if x['pass'] else ' n.s.'}")
            print(f"  {seed:>5} {r['rmse']['gp']:>9.2f} {r['rmse']['pchip']:>8.2f} "
                  f"{r['rmse']['linear']:>8.2f} {r['rmse']['model']:>8.2f}  {fmt(mg):>26}  {fmt(gp_):>26}")
        xc = [out["per_seed"][str(s)][name]["model_vs_pchip_crosscheck"]["skill_point"]
              for s in SEEDS]
        print(f"  cross-check model_vs_pchip: {[round(v, 4) for v in xc]}")
        for label in ("peak", "dt_le_15ms", "dt_gt_15ms"):
            rows = []
            for s in SEEDS:
                r = out["per_seed"][str(s)][name].get(f"model_vs_gp_{label}")
                rows.append("---" if r is None else
                            f"{r['skill_point']:+.3f}{'*' if r['pass'] else ''}(n={r['n']})")
            print(f"  model vs gp [{label:>11}]: " + "  ".join(rows) + "   (* = CI95>0)")

    out_path = REPO_ROOT / "data" / ".gp_analysis.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nsaved -> {out_path}")


if __name__ == "__main__":
    main()
