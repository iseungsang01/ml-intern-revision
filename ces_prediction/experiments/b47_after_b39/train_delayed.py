"""B.47 arm L (PREREGISTRATION_B47.md section 3): train_seq with a delayed CES history.

Launched by run_b47.py INSTEAD of seq/train_seq.py for the runs of arm L.  It imports train_seq as a
module, replaces the block builder that train_seq.main() calls by a wrapper, runs train_seq.main()
and then writes b47_delay.json next to metrics.json.  No shared file is edited.

The wrapper calls the original seq_data.build_blocks and overwrites the six CES-derived input
channels of every block (last value, log1p staleness, observed flag; T_i and V_rot) with
run_b42.ces_channels(time, standardized targets, observed mask, L, "true"): a value measured at row
u is available at row t only if u <= t - 1 - L, and the staleness is the true age.  With L = 0 the
rebuilt channels equal those of build_blocks (B.42 check 1; re-verified by check_b47.py).

Call sites: train_seq binds `build_blocks` into its own module namespace
(`from seq_data import ..., build_blocks, ...`) and calls it once, inside the `make` lambda of
main(), for the train and the validation blocks.  The lambda looks the name up in train_seq's
globals at call time, so rebinding `train_seq.build_blocks` covers both.  `seq_data.build_blocks` is
rebound as well (nothing else in this process calls it).  The wrapper counts its calls; the run fails
if main() returned without a single call.

Env: B47_CES_DELAY (required, integer >= 0; arm L uses 2 and the control 0) plus every CES_* variable
of train_seq, pinned by run_b47.py.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SEQ_DIR = HERE.parent / "seq"
CES_DIR = HERE.parents[1]
sys.path.insert(0, str(SEQ_DIR))
sys.path.insert(1, str(CES_DIR))


def read_delay():
    raw = os.environ.get("B47_CES_DELAY")
    if raw is None or raw.strip() == "":
        raise SystemExit("FATAL: B47_CES_DELAY is required (no default).")
    L = int(raw)
    if L < 0:
        raise SystemExit(f"FATAL: B47_CES_DELAY must be >= 0, got {L}")
    return L


def make_wrapper(original, L, ces_channels, gap):
    """Wrapper around seq_data.build_blocks that applies delay L to the six CES channels."""
    state = {"calls": 0, "blocks": 0}

    def delayed_build_blocks(arr, dims, stats, *args, **kwargs):
        blks = original(arr, dims, stats, *args, **kwargs)
        state["calls"] += 1
        time = arr[:, 0].astype(np.float64)
        tgt = arr[:, 1:3].astype(np.float64)
        tz = (tgt - stats["target"]["mean"].astype(np.float64)) / stats["target"]["std"].astype(np.float64)
        obs = ~np.isnan(tgt)
        # the block bounds of build_blocks (a step >= gap or <= 0 ends a block; blocks < 2 rows skipped)
        bounds = [0] + [i for i in range(1, len(time)) if not (0 < time[i] - time[i - 1] < gap)] + [len(time)]
        spans = [(a, b) for a, b in zip(bounds[:-1], bounds[1:]) if b - a >= 2]
        if len(spans) != len(blks):
            raise SystemExit("FATAL: b47 wrapper: block spans do not match build_blocks")
        base = dims["n_fast"] + 1
        out = []
        for blk, (a, b) in zip(blks, spans):
            if blk["x"].shape[0] != b - a or not np.array_equal(blk["time"], arr[a:b, 0]):
                raise SystemExit("FATAL: b47 wrapper: a block does not match its span")
            x = blk["x"].copy()
            x[:, base:base + 6] = ces_channels(time[a:b], tz[a:b], obs[a:b], L, "true")
            out.append({**blk, "x": x})
        state["blocks"] += len(out)
        return out

    return delayed_build_blocks, state


def install(L):
    """Import train_seq and rebind its block builder; returns (train_seq module, wrapper state)."""
    import seq_data
    import train_seq
    from dataset import STUCK_GAP_SECONDS
    sys.path.insert(0, str(HERE.parent / "b42_delay"))
    from run_b42 import ces_channels
    if train_seq.build_blocks is not seq_data.build_blocks:
        raise SystemExit("FATAL: train_seq.build_blocks is not seq_data.build_blocks; the patch would miss it")
    original = seq_data.build_blocks
    wrapper, state = make_wrapper(original, L, ces_channels, STUCK_GAP_SECONDS)
    train_seq.build_blocks = wrapper
    seq_data.build_blocks = wrapper
    return train_seq, state


def main():
    L = read_delay()
    train_seq, state = install(L)
    print(f"[b47] train_seq with the CES history delayed by L = {L} frames (true staleness)", flush=True)
    train_seq.main()
    if state["calls"] == 0:
        raise SystemExit("FATAL: the b47 wrapper was never called; the delay was not applied")
    out_dir = Path(os.environ["CES_OUTPUT_DIR"])
    rec = {"delay_frames": L, "staleness": "true", "rule": "value measured at row u available at row t iff u <= t - 1 - L",
           "wrapper_calls": state["calls"], "wrapped_blocks": state["blocks"],
           "registration": "ces_prediction/experiments/PREREGISTRATION_B47.md"}
    (out_dir / "b47_delay.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
    print(f"[b47] wrapper calls {state['calls']}, blocks {state['blocks']}; wrote b47_delay.json", flush=True)


if __name__ == "__main__":
    main()
    # as train_seq: skip the Windows CUDA teardown so the return code reflects the run
    os._exit(0)
