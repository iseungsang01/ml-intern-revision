"""B.59 arm L (PREREGISTRATION_B59.md section 3): the fast channels one row earlier, in training and in scoring.

`wrap(original, S)` returns a replacement for seq_data.build_blocks that calls the original and then, in every
block, replaces the fast channels x[:, :n_fast] (BES 9, ECEI 4, MC 2, already standardized) of row i by those of
row max(i - S, 0) of the same block: with S = 1 every row reads the fast channels of the row before it and the
first row of a block keeps its own.  Nothing else of the block (dt, the six CES-history channels, y, mask, time)
is touched.  With S = 0 the blocks are returned unchanged.

The standardization constants are those of the estimator (fit_stats on the unshifted grid of the training
discharges; build_blocks standardizes before the shift).  A row-wise shift and a fixed affine standardization
commute, so this equals standardizing the shifted raw channels with the same constants.

`install_train(S)` rebinds the name build_blocks in train_seq (its `make` lambda in main() looks it up in the
module globals at call time) and in seq_data; `install_eval(S)` rebinds it in eval_seq (build_pred_lookup).
check_b59.py verifies that these are the only references.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SEQ_DIR = HERE.parent / "seq"
CES_DIR = HERE.parents[1]
for _p in (str(CES_DIR), str(SEQ_DIR)):      # seq/ ends up first, as train_seq / eval_seq put it
    if _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)

ENV_VAR = "B59_FAST_SHIFT"


def read_shift():
    raw = os.environ.get(ENV_VAR)
    if raw is None or raw.strip() == "":
        raise SystemExit(f"FATAL: {ENV_VAR} is required (no default).")
    S = int(raw)
    if S not in (0, 1):
        raise SystemExit(f"FATAL: {ENV_VAR} must be 0 or 1, got {S}")
    return S


def shift_x(x, n_fast, S):
    """(L, F) block inputs -> copy with x[i, :n_fast] = x[max(i - S, 0), :n_fast]."""
    if S == 0:
        return x
    out = x.copy()
    src = np.maximum(np.arange(x.shape[0]) - S, 0)
    out[:, :n_fast] = x[src, :n_fast]
    return out


def wrap(original, S):
    state = {"calls": 0, "blocks": 0}

    def shifted_build_blocks(arr, dims, stats, *args, **kwargs):
        blks = original(arr, dims, stats, *args, **kwargs)
        state["calls"] += 1
        state["blocks"] += len(blks)
        if S == 0:
            return blks
        n_fast = int(dims["n_fast"])
        return [{**b, "x": shift_x(b["x"], n_fast, S)} for b in blks]

    return shifted_build_blocks, state


def install_train(S):
    import seq_data
    import train_seq
    if train_seq.build_blocks is not seq_data.build_blocks:
        raise SystemExit("FATAL: train_seq.build_blocks is not seq_data.build_blocks; the patch would miss it")
    w, state = wrap(seq_data.build_blocks, S)
    train_seq.build_blocks = w
    seq_data.build_blocks = w
    return train_seq, state


def install_eval(S):
    import seq_data
    import eval_seq
    if eval_seq.build_blocks is not seq_data.build_blocks:
        raise SystemExit("FATAL: eval_seq.build_blocks is not seq_data.build_blocks; the patch would miss it")
    w, state = wrap(seq_data.build_blocks, S)
    eval_seq.build_blocks = w
    seq_data.build_blocks = w
    return eval_seq, state
