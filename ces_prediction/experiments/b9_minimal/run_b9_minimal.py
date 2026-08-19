"""B.9 axis D: does the family tie survive the 1k-10k band, where capacity is scarce?

Axis B showed three sequence-operator families are indistinguishable on skill at ~300k
parameters once the reach matches. `b8_minimal` sweeps 1k-10k, but with **recurrent arms
only** (`v2m*` / `b3m*`), so the band where an inductive bias would matter most had never
been asked the family question. This adds the convolutional side of it.

Reach is fixed at 15 for every arm — above the 7-step saturation axis A measured, and equal
to a 3-layer TCN's receptive field — so the operator is the only variable against the LSTM
rungs. Pre-registered as PREREGISTRATION_B9.md §2.3-D (H6) before any of these numbers
existed.

    tcn8k  8,094 params    vs  v2m7k  6,866
    tcn3k  3,238 params    vs  v2m2k  2,362
    tcn2k  1,808 params    vs  b3m1k  1,208

**This is no longer a deployment argument.** Axis C found that with a minimal-operator step
even the 357k backbone clears 1 ms, so small models are not required by the budget. What is
left is the measurement itself: how far down does "the family does not matter" hold.

Each arm is paired against (a) `v2r15` — the same reach, full width, from axis A — and
(b) the size-matched recurrent arm from `b8_minimal`, when that batch has been run.

Usage (repo root, after run_b9_reach.py; b8_minimal for the size-matched column):
  py ces_prediction/experiments/b9_minimal/run_b9_minimal.py --smoke
  py ces_prediction/experiments/b9_minimal/run_b9_minimal.py [--resume] [--arms tcn3k]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
CES_DIR = REPO_ROOT / "ces_prediction"
DATA = REPO_ROOT / "data"
sys.path.insert(0, str(CES_DIR / "experiments" / "b9_reach"))
sys.path.insert(1, str(CES_DIR / "experiments" / "b1_gate"))
from run_b9_reach import SEEDS, PRACTICAL_EPS, PAIRED, one_run, run_dir  # noqa: E402
from run_b1_gate import GATE_ENV, run_step  # noqa: E402

REACH = 15
# arm -> the size-matched recurrent arm in b8_minimal (its run dir prefix is .b8_<name>).
ARMS = {"tcn8k": "v2m7k", "tcn3k": "v2m2k", "tcn2k": "b3m1k"}


def pair_one(arm, seed, ref_dir, label, smoke=False, resume=False):
    out_dir = run_dir(arm, seed, smoke)
    dst = out_dir / f"paired_vs_{label}.json"
    if resume and dst.exists():
        return True
    if not (ref_dir / "comparison_errors_test.npz").exists():
        return None                       # control not run yet; column stays empty
    return run_step([PAIRED, "--a", out_dir / "comparison_errors_test.npz",
                     "--b", ref_dir / "comparison_errors_test.npz", "--out", dst],
                    dict(GATE_ENV), out_dir / "run.log", f"paired(vs_{label})",
                    artifacts=(dst,))


def read_paired(arm, seed, label):
    path = run_dir(arm, seed) / f"paired_vs_{label}.json"
    if not path.exists():
        return {}
    p = json.loads(path.read_text())
    out = {}
    for t in ("CES_TI", "CES_VT"):
        node = (p.get("targets", {}) or {}).get(t, {}) or {}
        out[t] = (node.get("skill_point"), node.get("skill_ci95"))
    return out


def fmt(v, spec="+.3f"):
    return "n/a" if v is None else format(v, spec)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--arms", nargs="*", default=list(ARMS))
    args = ap.parse_args()
    arms = [a for a in args.arms if a in ARMS] or list(ARMS)
    seeds = SEEDS[:1] if args.smoke else SEEDS

    records = []
    for arm in arms:
        for seed in seeds:
            rec = one_run(REACH, seed, args.smoke, args.resume, variant=arm, tag=arm)
            if rec.get("status") == "ok" and not args.smoke:
                pair_one(arm, seed, run_dir(f"v2r{REACH}", seed), f"v2r{REACH}",
                         resume=args.resume)
                pair_one(arm, seed, DATA / f".b8_{ARMS[arm]}_s{seed}", "sizematch",
                         resume=args.resume)
            records.append(rec)
            print(f"[b9d]   -> {rec.get('status')} ({rec.get('seconds')}s, "
                  f"params={rec.get('params')})", flush=True)
    if args.smoke:
        print("\n[b9d] smoke done")
        return

    summary = {"question": "does the family tie survive the 1k-10k band?",
               "protocol": {"env": GATE_ENV, "seeds": list(SEEDS), "reach": REACH,
                            "arms": ARMS, "practical_eps": PRACTICAL_EPS,
                            "prereg": "experiments/PREREGISTRATION_B9.md 2.3-D (H6)"},
               "runs": records, "arms": {}}

    print("\n" + "=" * 100)
    print("paired: positive = the TCN arm wins")
    print("arm".rjust(7) + "params".rjust(8) + "vs v2r15".rjust(11) + "sig".rjust(6)
          + "size-matched".rjust(15) + "vs".rjust(8) + "sig".rjust(6) + "verdict".rjust(11))
    for arm in arms:
        pts = [r for r in records if r.get("tag") == arm and r.get("status") == "ok"]
        if not pts:
            continue
        node = {"params": pts[0].get("params"), "n": len(pts), "sizematch_arm": ARMS[arm]}
        for label in (f"v2r{REACH}", "sizematch"):
            vals, wins, losses = [], 0, 0
            for r in pts:
                sk, ci = read_paired(arm, r["seed"], label).get("CES_TI", (None, None))
                if sk is None:
                    continue
                vals.append(sk)
                wins += ci[0] > 0
                losses += ci[1] < 0
            node[f"mean_TI_vs_{label}"] = float(np.mean(vals)) if vals else None
            node[f"wins_vs_{label}"], node[f"losses_vs_{label}"] = wins, losses
        mean = node["mean_TI_vs_sizematch"]
        sig = max(node["wins_vs_sizematch"], node["losses_vs_sizematch"])
        if mean is None:
            node["verdict"] = "no control"
        elif abs(mean) < PRACTICAL_EPS and sig <= 1:
            node["verdict"] = "tie"
        elif abs(mean) >= PRACTICAL_EPS and sig >= 3:
            node["verdict"] = "differs"
        else:
            node["verdict"] = "undecided"
        summary["arms"][arm] = node
        print(arm.rjust(7) + str(node["params"]).rjust(8)
              + fmt(node[f"mean_TI_vs_v2r{REACH}"]).rjust(11)
              + f"{node[f'wins_vs_v2r{REACH}']}/{node[f'losses_vs_v2r{REACH}']}".rjust(6)
              + ARMS[arm].rjust(15) + fmt(node["mean_TI_vs_sizematch"]).rjust(8)
              + f"{node['wins_vs_sizematch']}/{node['losses_vs_sizematch']}".rjust(6)
              + node["verdict"].rjust(11))
    print("  (sig = significant wins / significant losses out of 4 splits)")

    out = DATA / ".b9_minimal_family.json"
    out.write_text(json.dumps(summary, indent=1))
    print(f"\n[b9d] wrote {out}")


if __name__ == "__main__":
    main()
