"""Shot-clustered CI for the campaign-shift result against the CAUSAL baseline.

`bootstrap_compare.py` only gates against the interpolants, so the campaign batch left
the claim that actually survives -- "still beats persistence across a campaign
boundary" -- as a bare point estimate. That claim is load-bearing for the paper's
headline, so it needs the same PR4 treatment as everything else: whole shots resampled
with replacement, 10,000 times.

Reads the campaign runs' own npz (which carry `{target}_se_persistence`) and reuses
`bootstrap_compare._bootstrap`, so the statistic is identical to the one used elsewhere.

Usage (repo root):  py ces_prediction/experiments/campaign/bootstrap_vs_persistence.py
Writes: data/.campaign_vs_persistence.json
"""

import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "ces_prediction"))
from bootstrap_compare import B, BOOTSTRAP_SEED, _bootstrap  # noqa: E402

SEEDS = (42, 1, 7, 123)
TARGETS = ("CES_TI", "CES_VT")
ARMS = ("persistence", "pchip", "linear")


def main():
    out = {"_design": {"bootstrap_resamples": B, "seed": BOOTSTRAP_SEED,
                       "split": "temporal campaign split (fixed); seeds vary init only"},
           "per_seed": {}}
    for seed in SEEDS:
        path = REPO_ROOT / "data" / f".campaign_s{seed}" / "comparison_errors_test.npz"
        if not path.exists():
            raise SystemExit(f"FATAL: missing {path}")
        z = np.load(path)
        rng = np.random.default_rng(BOOTSTRAP_SEED)
        entry = {}
        for target in TARGETS:
            shot = z[f"{target}_shot"]
            se_model = z[f"{target}_se_model"]
            entry[target] = {
                arm: _bootstrap(shot, se_model, z[f"{target}_se_{arm}"], rng)
                for arm in ARMS if f"{target}_se_{arm}" in z.files
            }
        out["per_seed"][seed] = entry

    path = REPO_ROOT / "data" / ".campaign_vs_persistence.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote {path}\n")

    for target in TARGETS:
        print(f"=== {target} under campaign shift, shot-clustered 95% CI")
        print(f"{'seed':>5} {'vs arm':>12} {'skill':>8} {'95% CI':>22} {'gate':>5} {'shots':>6}")
        tally = {a: 0 for a in ARMS}
        for seed in SEEDS:
            for arm in ARMS:
                r = out["per_seed"][seed][target].get(arm)
                if not r:
                    continue
                tally[arm] += int(r["pass"])
                ci = r["skill_ci95"]
                print(f"{seed:>5} {arm:>12} {r['skill_point']:>+8.3f} "
                      + f"[{ci[0]:+.3f}, {ci[1]:+.3f}]".rjust(22)
                      + f" {'PASS' if r['pass'] else 'n.s.':>5} {r['n_shots']:>6}")
        print("      " + "  ".join(f"{a}: {tally[a]}/4 PASS" for a in ARMS))
        print()


if __name__ == "__main__":
    main()
