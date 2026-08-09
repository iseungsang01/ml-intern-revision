"""Re-judge the large-gap regime against a CAUSAL baseline.

THESIS_RESULTS §4.3 reports `CES_TI` skill collapsing in the wide-Δt bins (−8.98 at
Δt > 105 ms), and the window sweep reproduces it at every W. Read against PCHIP that
looks like "the model fails exactly where gap-filling is hardest". But PCHIP reads a
*future* anchor, so as Δt grows its problem gets easier (interpolate between two real
observations) while ours gets harder (extrapolate from the past alone) -- and no
real-time system can run PCHIP at all, because the future anchor does not exist yet.

So the question that decides whether this is a model weakness is: at large gaps, does
the model still beat the best baseline a real-time system COULD run -- persistence?

Inputs: `comparison_errors_test__{treatment}_persist.npz` per seed, written by
rerun_compare_persistence.py (verified bit-identical to the frozen artifacts on every
pre-existing key). The four seeds are POOLED, because the wide bins hold only tens of
samples per seed. Clustering is on the physical shot, so a discharge appearing in more
than one seed's test split is a single cluster rather than several -- conservative,
and it keeps the resampling unit the same as everywhere else in the paper.

Usage (repo root):  py ces_prediction/experiments/largegap/analyze_largegap.py
Writes: data/.largegap_analysis.json
"""

import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "ces_prediction"))
from bootstrap_compare import B, BOOTSTRAP_SEED, _bootstrap  # noqa: E402
from analyze_gap import BIN_EDGES_MS  # noqa: E402

SEEDS = (42, 1, 7, 123)
RUN_DIR = {
    42: REPO_ROOT / "data" / ".vt_repro_out",
    1: REPO_ROOT / "data" / ".vt_repro_ms_1",
    7: REPO_ROOT / "data" / ".vt_repro_ms_7",
    123: REPO_ROOT / "data" / ".vt_repro_ms_123",
}
SUFFIX = {"genuine": "__test_genuine_persist", "stuck0": "__test_stuck0_persist"}
TARGETS = ("CES_TI", "CES_VT")
BASELINES = ("pchip", "persistence")
# Coarse cuts on top of the fine bins: "the adjacent-history regime" vs everything
# beyond it, and the regime the sweep flagged as negative.
COARSE = {"dt <= 15 ms": (0.0, 15.0), "dt > 15 ms": (15.0, np.inf),
          "dt > 45 ms": (45.0, np.inf)}


def load_pooled(treatment, target):
    """Concatenate the four seeds' per-sample arrays for one target."""
    cols = {k: [] for k in ("shot", "dt", "model", "pchip", "persistence", "seed")}
    for seed in SEEDS:
        path = RUN_DIR[seed] / f"comparison_errors_test{SUFFIX[treatment]}.npz"
        if not path.exists():
            raise SystemExit(f"FATAL: missing {path}. Run rerun_compare_persistence.py first.")
        d = np.load(path)
        cols["shot"].append(d[f"{target}_shot"])
        cols["dt"].append(d[f"{target}_dt_ms"])
        cols["model"].append(d[f"{target}_se_model"])
        cols["pchip"].append(d[f"{target}_se_pchip"])
        cols["persistence"].append(d[f"{target}_se_persistence"])
        cols["seed"].append(np.full(len(d[f"{target}_shot"]), seed))
    return {k: np.concatenate(v) for k, v in cols.items()}


def judge(pool, mask, rng):
    """Skill + shot-clustered CI against each baseline on the masked subset."""
    if not mask.any():
        return None
    out = {"n": int(mask.sum()), "n_shots": int(len(np.unique(pool["shot"][mask]))),
           "rmse_model": float(np.sqrt(pool["model"][mask].mean()))}
    for base in BASELINES:
        out[base] = _bootstrap(pool["shot"][mask], pool["model"][mask],
                               pool[base][mask], rng)
        out[f"rmse_{base}"] = float(np.sqrt(pool[base][mask].mean()))
    return out


def main():
    results = {"_design": {
        "bootstrap_resamples": B, "bootstrap_seed": BOOTSTRAP_SEED,
        "cluster": "physical shot, pooled over the 4 seeds",
        "why": ("PCHIP reads a future anchor and is unavailable in real time; "
                "persistence is the causal reference the model must actually beat"),
    }, "treatments": {}}

    for treatment in SUFFIX:
        per_target = {}
        for target in TARGETS:
            pool = load_pooled(treatment, target)
            rng = np.random.default_rng(BOOTSTRAP_SEED)
            entry = {"n_total": int(len(pool["dt"])),
                     "n_shots_total": int(len(np.unique(pool["shot"]))),
                     "fine_bins": [], "coarse": {}}
            lo = 0.0
            for hi in list(BIN_EDGES_MS) + [float("inf")]:
                mask = (pool["dt"] > lo) & (pool["dt"] <= hi)
                res = judge(pool, mask, rng)
                if res:
                    label = f"({lo:g},{hi:g}]" if np.isfinite(hi) else f"({lo:g},inf)"
                    entry["fine_bins"].append({"bin_ms": label, **res})
                lo = hi
            for label, (a, b) in COARSE.items():
                mask = (pool["dt"] > a) & (pool["dt"] <= b)
                res = judge(pool, mask, rng)
                if res:
                    entry["coarse"][label] = res
            per_target[target] = entry
        results["treatments"][treatment] = per_target

    out = REPO_ROOT / "data" / ".largegap_analysis.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"wrote {out}\n")

    for treatment, per_target in results["treatments"].items():
        for target, entry in per_target.items():
            print(f"=== {target} / {treatment} "
                  f"(pooled n={entry['n_total']:,}, {entry['n_shots_total']} shots)")
            # ASCII header: the Windows console codepage mangles a literal delta.
            print(f"{'dt bin':>12} {'n':>7} {'shots':>6} "
                  f"{'skill vs PCHIP':>26} {'skill vs PERSISTENCE':>28}")
            rows = [(b["bin_ms"], b) for b in entry["fine_bins"]]
            rows += [(f"[{k}]", v) for k, v in entry["coarse"].items()]
            for label, b in rows:
                cells = []
                for base in BASELINES:
                    r = b[base]
                    ci = r["skill_ci95"]
                    gate = "PASS" if r["pass"] else "n.s."
                    cells.append(f"{r['skill_point']:+7.3f} [{ci[0]:+6.2f},{ci[1]:+6.2f}] {gate}")
                print(f"{label:>12} {b['n']:>7,} {b['n_shots']:>6} {cells[0]:>26} {cells[1]:>28}")
            print()


if __name__ == "__main__":
    main()
