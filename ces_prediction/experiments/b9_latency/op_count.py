"""How many operators does one online step dispatch? — the machine-independent half of cost.

`bench_budget.py` measures milliseconds, and milliseconds on this machine are not reliable:
the five-session pass of 2026-08-19 17:06 recorded a **21.8x** session-to-session p99 spread,
and the pre-registered rule (PREREGISTRATION_B9.md §4) correctly refused to resolve a 1 ms
verdict from it. But §8ah's actual finding was never about this machine — it was that latency
tracks **dispatched operator count** at ~1.3-1.6 us per op, essentially independent of the
arithmetic inside each op. That quantity is deterministic. It does not move between sessions,
it does not depend on who else is using the CPU, and it is the thing that scales with reach.

So this counts it. For each arm the online step is executed once under `torch.profiler`, and
every dispatched `aten::` operator is tallied. The three families then answer the reach
question structurally rather than statistically:

    recurrent    O(1) in reach     -- the state carries the past; the step is the same step
    dilated conv O(log R)          -- reach 2^(L+1)-1 needs L layers, each a fixed op count
    attention    O(1) in reach     -- one query against a cached band, but with a big constant

Counts are exact and reproducible on any machine; the microseconds-per-op conversion is the
only machine-specific number, and it is quoted from the min-of-five-sessions ladder rather
than from a contaminated mean.

Usage (repo root):
  py ces_prediction/experiments/b9_latency/op_count.py
"""

import json
import sys
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_budget import DATA, make_arm  # noqa: E402

# (arm label, kind, spec) — the same construction bench_budget times, so the counts and the
# milliseconds describe the same callable.
ARMS = (
    ("seq_v2", "seq", "v2"),                     # stock nn.LSTM step
    ("seq_v2_lean", "seq_lean", "v2"),
    ("seq_v2_tight", "seq_tight", "v2"),
    ("v2m2k_lean", "seq_lean", "v2m2k"),
    ("v2m2k_tight", "seq_tight", "v2m2k"),
    # reach ladder within each family: this is where the scaling law shows up
    ("tcn3_lean", "seq_lean", "tcn3"),           # RF 3,  1 layer
    ("tcn5_lean", "seq_lean", "tcn5"),           # RF 5,  2 layers (dilations 1,1)
    ("tcn7_lean", "seq_lean", "tcn7"),           # RF 7,  2 layers
    ("tcn15_lean", "seq_lean", "tcn15"),         # RF 15, 3 layers
    ("tcn63_lean", "seq_lean", "tcn63"),         # RF 63, 5 layers
    ("xfmr5_lean", "seq_lean", "xfmr5"),         # band 5
    ("xfmr7_lean", "seq_lean", "xfmr7"),         # band 7
    ("xfmr15_lean", "seq_lean", "xfmr15"),       # band 15
    ("xfmr63_lean", "seq_lean", "xfmr63"),       # band 63
    ("xfmr15_tight", "seq_tight", "xfmr15"),
    ("xfmr63_tight", "seq_tight", "xfmr63"),
)


def count_ops(call, warmup=5):
    """Dispatched aten:: operators in ONE step (lazy init warmed out first)."""
    for _ in range(warmup):
        call()
    with profile(activities=[ProfilerActivity.CPU]) as prof:
        call()
    ops = {}
    for evt in prof.key_averages():
        if evt.key.startswith("aten::"):
            ops[evt.key] = ops.get(evt.key, 0) + evt.count
    return sum(ops.values()), ops


def main():
    torch.set_grad_enabled(False)
    torch.set_num_threads(1)
    rows = {}
    print("arm".rjust(14) + "mode".rjust(18) + "params".rjust(9) + "aten ops".rjust(10)
          + "  top operators")
    for label, kind, spec in ARMS:
        call, mode, n_params = make_arm(kind, spec)
        total, ops = count_ops(call)
        top = ", ".join(f"{k.replace('aten::', '')}x{v}" for k, v in
                        sorted(ops.items(), key=lambda kv: -kv[1])[:4])
        rows[label] = {"mode": mode, "params": n_params, "aten_ops": total, "ops": ops}
        print(label.rjust(14) + mode.rjust(18) + f"{n_params:,}".rjust(9)
              + str(total).rjust(10) + "  " + top)

    out = DATA / ".b9_op_counts.json"
    out.write_text(json.dumps(rows, indent=1))
    print(f"\n[b9c] wrote {out}")

    def ops_of(name):
        return rows[name]["aten_ops"] if name in rows else None

    print("\nreach scaling (aten ops per online step):")
    for fam, arms in (("dilated conv", ["tcn3_lean", "tcn7_lean", "tcn15_lean", "tcn63_lean"]),
                      ("attention", ["xfmr7_lean", "xfmr15_lean", "xfmr63_lean"])):
        vals = [(a, ops_of(a)) for a in arms if ops_of(a)]
        print(f"  {fam:13s} " + "  ".join(f"{a}={v}" for a, v in vals))
    base = ops_of("seq_v2_tight")
    if base:
        for a in ("xfmr15_tight", "xfmr63_tight", "seq_v2_lean", "seq_v2"):
            v = ops_of(a)
            if v:
                print(f"  {a:14s} = {v:4d} ops = {v / base:.1f}x the tight recurrent step")


if __name__ == "__main__":
    main()
