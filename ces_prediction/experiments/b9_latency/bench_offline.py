"""Step time of the OFFLINE references (GP, PCHIP, linear interpolation), in the harness of the
latency ladder (`bench_budget.py`): real neighbour sets of the frozen TEST files, 200 warm-up +
2,000 timed calls, one thread, CPU, five sessions in separate processes, max-p99 rule.

Asked by the author (2026-11-16): does the offline GP take more than one 10 ms frame, and does that
say anything about which estimator is more accurate?  The second part needs no measurement: the
offline references are excluded from real-time use because they read the NEXT CES sample, whatever
they cost, and the accuracy interval (network vs offline GP -0.010 [-0.043, +0.014]) does not depend
on their speed.  This script answers the first part with a number.  `persistence` and `gp_causal`
are re-timed in the same sessions as anchors, because absolutes move between sessions.

Run from the repo root:
    py ces_prediction/experiments/b9_latency/bench_offline.py --all
Artifacts: data/.b9_offline_s{1..5}.json, data/.b9_offline.json
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import bench_budget as BB  # noqa: E402  (the harness: _time_loop, _neighbor_sets, SESSIONS)

ARMS = (("persistence", "predict_persistence", "causal"),
        ("gp_causal", "predict_gp_causal", "causal"),
        ("gp_offline", "predict_gp", "reads the next CES sample"),
        ("pchip_offline", "predict_pchip", "reads the next CES sample"),
        ("linear_offline", "predict_linear", "reads the next CES sample"))


def run_session(index):
    torch.set_num_threads(1)
    import numpy as np
    import baselines_interpolation as B
    sets = BB._neighbor_sets()[1]                 # column 1 = CES_TI, as in the ladder
    n_future = [int(np.sum(t > tt)) for t, _v, tt in sets]
    n_past = [int(np.sum(t < tt)) for t, _v, tt in sets]
    out = {"session": index, "warmup": BB.WARMUP, "iters": BB.ITERS, "sets": len(sets),
           "share_with_a_future_neighbour": float(np.mean([n > 0 for n in n_future])),
           "median_past_neighbours": float(np.median(n_past)),
           "median_future_neighbours": float(np.median(n_future)), "arms": {}}
    for name, fn_name, note in ARMS:
        fn = getattr(B, fn_name)
        counter = {"i": 0}

        def call():
            times, values, tt = sets[counter["i"] % len(sets)]
            counter["i"] += 1
            fn(times, values, tt)
        stats = BB._time_loop(call)
        stats["note"] = note
        out["arms"][name] = stats
        print(f"[b9-offline] s{index} {name:>14} median={stats['median_ms']:7.3f} p99={stats['p99_ms']:7.3f} ms",
              flush=True)
    (BB.DATA / f".b9_offline_s{index}.json").write_text(json.dumps(out, indent=1))


def aggregate():
    ses = [json.loads((BB.DATA / f".b9_offline_s{i}.json").read_text()) for i in range(1, BB.SESSIONS + 1)]
    res = {"sessions": len(ses), "warmup": BB.WARMUP, "iters": BB.ITERS,
           "share_with_a_future_neighbour": ses[0]["share_with_a_future_neighbour"],
           "median_past_neighbours": ses[0]["median_past_neighbours"],
           "median_future_neighbours": ses[0]["median_future_neighbours"], "arms": {}}
    print("| arm | uses | median (ms) | max p99 (ms) | max over sessions (ms) | inside 10 ms |")
    print("|---|---|---:|---:|---:|---|")
    for name, _fn, note in ARMS:
        med = sorted(s["arms"][name]["median_ms"] for s in ses)[len(ses) // 2]
        p99 = max(s["arms"][name]["p99_ms"] for s in ses)
        mx = max(s["arms"][name]["max_ms"] for s in ses)
        res["arms"][name] = {"median_ms": med, "max_p99_ms": p99, "max_ms": mx, "note": note,
                             "verdict_10ms": BB.verdict(p99, 10.0)}
        print(f"| `{name}` | {note} | {med:.3f} | {p99:.3f} | {mx:.3f} | {BB.verdict(p99, 10.0)} |")
    (BB.DATA / ".b9_offline.json").write_text(json.dumps(res, indent=1))
    print("wrote", BB.DATA / ".b9_offline.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", type=int)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--aggregate", action="store_true")
    a = ap.parse_args()
    if a.session:
        run_session(a.session)
    elif a.all:
        for i in range(1, BB.SESSIONS + 1):
            subprocess.run([sys.executable, str(Path(__file__)), "--session", str(i)], cwd=BB.REPO_ROOT, check=True)
        aggregate()
    elif a.aggregate:
        aggregate()
    else:
        ap.error("one of --session / --all / --aggregate")


if __name__ == "__main__":
    main()
