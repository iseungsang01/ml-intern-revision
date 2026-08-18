"""The §8ac truncation ladder, re-scored on B.9's rungs so the two curves can be compared.

§8ae compared "trained at reach 2" against "truncated to ctx 2" and found ≥ 84% of the
`T_i` deficit was cold start. That was one point. B.9 axis A trains a rung at 2 / 7 / 15 /
31 / 63, and §8ac's ladder only measured 1 / 2 / 3 / 5 / 10 / 20 / 50 / 100 / 300 — so four
of the five rungs have no truncated partner and the comparison would stay anecdotal.

This re-scores the **same frozen B.1 backbones** at B.9's rungs. No retraining, no new
training population: it copies each checkpoint into its own directory and runs `eval_seq`
with `CES_SEQ_CONTEXTS` set to the rungs, exactly as `run_reach.py` does. §8ac's own
artifacts are untouched — `score_one(out_tag=...)` writes to `.b9trunc_s*` instead of
`.reach_s*`, so nothing published is overwritten to obtain this.

The identity check still applies and is the warrant for putting the two curves on one axis:
`ctx = full` in each new directory must reproduce the frozen `se_model` / `se_pchip` /
`shot` / `dt_ms` arrays bit-identically, which is what makes the truncated and trained
ladders describe the same rows.

Usage (repo root, after run_b9_reach.py):
  py ces_prediction/experiments/b9_reach/truncated_at_rungs.py [--resume] [--jobs 4]
Writes: data/.b9_truncated_at_rungs.json
"""

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
CES_DIR = REPO_ROOT / "ces_prediction"
DATA = REPO_ROOT / "data"
sys.path.insert(0, str(CES_DIR / "experiments" / "reach"))
sys.path.insert(1, str(CES_DIR / "experiments" / "b9_reach"))
from run_reach import score_one, analyze, SEEDS  # noqa: E402
from run_b9_reach import REACHES, PRACTICAL_EPS  # noqa: E402

OUT_TAG = "b9trunc"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--jobs", type=int, default=4)
    args = ap.parse_args()

    if not any(DATA.glob("s*.csv")):
        raise SystemExit("FATAL: no shot CSVs in data/ -- real data required, aborting.")

    start = time.time()
    jobs = max(1, min(args.jobs, len(SEEDS)))
    with ThreadPoolExecutor(max_workers=jobs) as ex:
        futures = [ex.submit(score_one, s, REACHES, False, args.resume, OUT_TAG)
                   for s in SEEDS]
        records = [f.result() for f in futures]
    bad = [r for r in records if r.get("status") != "ok"]
    if bad:
        raise SystemExit(f"FATAL: {len(bad)} seed(s) failed: {bad}")

    per_seed = {str(r["seed"]): analyze(r["seed"], REACHES, False, OUT_TAG) for r in records}
    summary = {"question": "the 8ac truncation ladder, on B.9's rungs",
               "protocol": {"retrained": False, "rungs": list(REACHES), "seeds": list(SEEDS),
                            "identity_check": {r["seed"]: r.get("identity_vs_frozen")
                                               for r in records},
                            "source": "frozen B.1 backbones (.b1_seqv2_s*_i*)",
                            "note": "compare against data/.b9_reach_ladder.json, which "
                                    "trains a model AT each rung instead of cutting one"},
               "per_seed": per_seed, "mean": {}}

    print("\n" + "=" * 72)
    print("truncated (state cut) vs full, mean over 4 splits -- paired skill")
    print("rung".rjust(6) + "CES_TI".rjust(10) + "sig".rjust(6)
          + "CES_VT".rjust(10) + "sig".rjust(6))
    for r in REACHES:
        node = {}
        for t in ("CES_TI", "CES_VT"):
            vals = [per_seed[str(s)][t]["ladder"][str(r)]["paired_vs_full"] for s in SEEDS]
            sig = sum(1 for s in SEEDS
                      if per_seed[str(s)][t]["ladder"][str(r)]["significant_deficit"])
            node[f"mean_{t}"] = float(np.mean(vals))
            node[f"sig_deficit_{t}"] = sig
        summary["mean"][str(r)] = node
        print(str(r).rjust(6) + f"{node['mean_CES_TI']:+.3f}".rjust(10)
              + f"{node['sig_deficit_CES_TI']}/4".rjust(6)
              + f"{node['mean_CES_VT']:+.3f}".rjust(10)
              + f"{node['sig_deficit_CES_VT']}/4".rjust(6))

    out = DATA / ".b9_truncated_at_rungs.json"
    out.write_text(json.dumps(summary, indent=1))
    print(f"\n[b9trunc] {(time.time() - start) / 60:.1f} min; wrote {out}")
    print(f"[b9trunc] practical floor for the trained ladder is {PRACTICAL_EPS}; this "
          f"ladder uses run_reach's own 0.002 (same-model truncation, see its note)")


if __name__ == "__main__":
    main()
