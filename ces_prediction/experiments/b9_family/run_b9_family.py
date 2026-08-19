"""B.9 axis B: does the sequence-operator FAMILY matter once the reach is matched?

§8ac left one architecture question explicitly open — "it does not establish that recurrence
is the only way to reach 50 steps: a dilated causal TCN reaches 63 steps with 5 layers and
remains an untested candidate". §8ae then showed the reach ladder it was arguing from had
been measuring cold start, so the question is now sharper: **hold the reach fixed and vary
only the operator.**

Three families, all with the seq_v2 routing (`V_rot` never sees the fast diagnostics), all
trained and scored at the reach their receptive field declares:

    tcn15    dilated causal conv, 3 layers, RF 15      vs  v2r15
    tcn63    dilated causal conv, 5 layers, RF 63      vs  v2r63
    xfmr63   causal banded attention, 2 layers, RF 63  vs  v2r63

Each arm gets three pairings on identical rows: against the full-reach B.1 backbone,
against the `W = 2` window control, and — the one that answers H3/H4 — against the axis A
rung trained at its own reach. That third pairing is what makes "family" the single
variable; against the backbone alone, family and reach would be confounded again.

**Axis A must have run first.** The matching `v2r{RF}` runs are the controls, so this
refuses to start without them rather than silently falling back to a backbone comparison.

Usage (repo root):
  py ces_prediction/experiments/b9_family/run_b9_family.py --smoke
  py ces_prediction/experiments/b9_family/run_b9_family.py [--resume] [--arms tcn63]
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
from run_b9_reach import (SEEDS, PRACTICAL_EPS, PAIRED, one_run, run_dir,  # noqa: E402
                          backbone_dir)
from run_b1_gate import GATE_ENV, run_step  # noqa: E402

# arm -> the receptive field it declares, which is also the axis A rung it pairs against.
ARMS = {"tcn15": 15, "tcn63": 63, "xfmr63": 63, "xfmr15": 15,
        # per-family low rungs: is the 70 ms threshold family-invariant?
        "tcn3": 3, "tcn7": 7, "xfmr7": 7}


def control_dir(reach, seed, smoke=False):
    return run_dir(f"v2r{reach}", seed, smoke)


def pair_against_control(arm, reach, seed, smoke, resume):
    """The H3/H4 pairing: this family vs the LSTM trained at the SAME reach."""
    out_dir = run_dir(arm, seed, smoke)
    ctrl = control_dir(reach, seed, smoke)
    dst = out_dir / f"paired_vs_v2r{reach}.json"
    if resume and dst.exists():
        return True
    if not (ctrl / "comparison_errors_test.npz").exists():
        raise SystemExit(f"FATAL: axis A control missing: {ctrl} — run run_b9_reach.py first")
    env = dict(GATE_ENV)
    return run_step([PAIRED, "--a", out_dir / "comparison_errors_test.npz",
                     "--b", ctrl / "comparison_errors_test.npz", "--out", dst],
                    env, out_dir / "run.log", f"paired(vs_v2r{reach})", artifacts=(dst,))


def fmt(value, spec="+.3f"):
    return "n/a" if value is None else format(value, spec)


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
        reach = ARMS[arm]
        for seed in seeds:
            rec = one_run(reach, seed, args.smoke, args.resume, variant=arm, tag=arm)
            if rec.get("status") == "ok" and not args.smoke:
                if not pair_against_control(arm, reach, seed, args.smoke, args.resume):
                    rec["status"] = "paired_vs_control_failed"
                else:
                    try:
                        p = json.loads((run_dir(arm, seed) /
                                        f"paired_vs_v2r{reach}.json").read_text())
                        for t in ("CES_TI", "CES_VT"):
                            node = (p.get("targets", {}) or {}).get(t, {}) or {}
                            rec[f"paired_{t}_vs_control"] = node.get("skill_point")
                            rec[f"paired_{t}_vs_control_ci"] = node.get("skill_ci95")
                    except Exception:
                        pass
            records.append(rec)
            print(f"[b9b]   -> {rec.get('status')} ({rec.get('seconds')}s, "
                  f"params={rec.get('params')})", flush=True)
    if args.smoke:
        print("\n[b9b] smoke done")
        return

    summary = {"question": "does the sequence-operator family matter once reach is matched?",
               "protocol": {"env": GATE_ENV, "seeds": list(SEEDS), "arms": ARMS,
                            "practical_eps": PRACTICAL_EPS,
                            "control": "the axis A v2r{RF} rung, same reach, same rows",
                            "prereg": "experiments/PREREGISTRATION_B9.md axis B"},
               "runs": records, "arms": {}}

    print("\n" + "=" * 104)
    # ASCII only in prints: this console is cp949 (see run_b9_reach.py).
    print("paired vs the SAME-REACH LSTM rung (H3/H4): positive = this family wins")
    print("arm".rjust(8) + "RF".rjust(5) + "params".rjust(9) + "TI vs ctrl".rjust(12)
          + "sig".rjust(6) + "VT vs ctrl".rjust(12) + "sig".rjust(6)
          + "TI vs backbone".rjust(16) + "verdict".rjust(10))
    for arm in arms:
        pts = [r for r in records if r.get("tag") == arm and r.get("status") == "ok"]
        if not pts:
            continue
        node = {"reach": ARMS[arm], "n": len(pts), "params": pts[0].get("params")}
        for t in ("CES_TI", "CES_VT"):
            for ref in ("control", "backbone"):
                vals = [r[f"paired_{t}_vs_{ref}"] for r in pts
                        if r.get(f"paired_{t}_vs_{ref}") is not None]
                node[f"mean_{t}_vs_{ref}"] = float(np.mean(vals)) if vals else None
                node[f"wins_{t}_vs_{ref}"] = sum(
                    1 for r in pts if (r.get(f"paired_{t}_vs_{ref}_ci") or [0, 0])[0] > 0)
                node[f"losses_{t}_vs_{ref}"] = sum(
                    1 for r in pts if (r.get(f"paired_{t}_vs_{ref}_ci") or [0, 0])[1] < 0)
        # PREREGISTRATION_B9.md §3.2 applied to the H3/H4 contrast.
        mean_ti = node["mean_CES_TI_vs_control"]
        sig = max(node["wins_CES_TI_vs_control"], node["losses_CES_TI_vs_control"])
        if mean_ti is None:
            node["verdict_CES_TI"] = "n/a"
        elif abs(mean_ti) < PRACTICAL_EPS and sig <= 1:
            node["verdict_CES_TI"] = "tie"
        elif abs(mean_ti) >= PRACTICAL_EPS and sig >= 3:
            node["verdict_CES_TI"] = "differs"
        else:
            node["verdict_CES_TI"] = "undecided"
        summary["arms"][arm] = node
        print(arm.rjust(8) + str(ARMS[arm]).rjust(5) + str(node["params"]).rjust(9)
              + fmt(node["mean_CES_TI_vs_control"]).rjust(12)
              + f"{node['wins_CES_TI_vs_control']}/{node['losses_CES_TI_vs_control']}".rjust(6)
              + fmt(node["mean_CES_VT_vs_control"]).rjust(12)
              + f"{node['wins_CES_VT_vs_control']}/{node['losses_CES_VT_vs_control']}".rjust(6)
              + fmt(node["mean_CES_TI_vs_backbone"]).rjust(16)
              + node["verdict_CES_TI"].rjust(10))
    print("  (sig column = significant wins / significant losses out of 4 splits)")

    out = DATA / ".b9_family.json"
    out.write_text(json.dumps(summary, indent=1))
    print(f"\n[b9b] wrote {out}")


if __name__ == "__main__":
    main()
