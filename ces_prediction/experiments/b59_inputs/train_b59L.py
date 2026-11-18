"""B.59 arm L: seq/train_seq.py with the fast channels one row earlier (fast_shift, B59_FAST_SHIFT).

Launched by run_b59.py instead of seq/train_seq.py for the runs of arm L.  Writes b59_shift.json next to
metrics.json.  No shared file is edited.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import fast_shift as FS


def main():
    S = FS.read_shift()
    train_seq, state = FS.install_train(S)
    print(f"[b59] train_seq with the fast channels shifted by {S} row(s) within each block", flush=True)
    train_seq.main()
    if state["calls"] == 0:
        raise SystemExit("FATAL: the b59 wrapper was never called; the shift was not applied")
    rec = {"fast_shift_rows": S, "rule": "x[i, :n_fast] = x[max(i - S, 0), :n_fast] within each block",
           "wrapper_calls": state["calls"], "wrapped_blocks": state["blocks"],
           "registration": "ces_prediction/experiments/PREREGISTRATION_B59.md"}
    (Path(os.environ["CES_OUTPUT_DIR"]) / "b59_shift.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
    print(f"[b59] wrapper calls {state['calls']}, blocks {state['blocks']}; wrote b59_shift.json", flush=True)


if __name__ == "__main__":
    main()
    os._exit(0)   # as train_seq: skip the Windows CUDA teardown so the return code reflects the run
