"""B.59 arm L: seq/eval_seq.py with the fast channels one row earlier (the same shift as in training).

Refuses to score a run whose b59_shift.json records another shift.  Writes b59_shift_eval.json.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import fast_shift as FS


def main():
    S = FS.read_shift()
    out = Path(os.environ["CES_OUTPUT_DIR"])
    rec_p = out / "b59_shift.json"
    if rec_p.exists():
        trained = json.loads(rec_p.read_text(encoding="utf-8"))["fast_shift_rows"]
        if trained != S:
            raise SystemExit(f"FATAL: trained with a shift of {trained}, scored with {S}")
    elif os.environ.get("B59_ALLOW_NO_TRAIN_RECORD") != "1":     # check_b59 scores the frozen B.39 estimator
        raise SystemExit(f"FATAL: {rec_p} missing (the run was not trained by train_b59L.py)")
    eval_seq, state = FS.install_eval(S)
    print(f"[b59] eval_seq with the fast channels shifted by {S} row(s) within each block", flush=True)
    eval_seq.compare()
    if state["calls"] == 0:
        raise SystemExit("FATAL: the b59 wrapper was never called at scoring")
    (out / "b59_shift_eval.json").write_text(json.dumps({"fast_shift_rows": S, "wrapper_calls": state["calls"],
                                                         "wrapped_blocks": state["blocks"]}, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
    os._exit(0)
