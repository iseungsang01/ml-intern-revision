"""B.53 arm C: seq/eval_seq.py with `v2hx` registered (launched by run_b53.py instead of eval_seq.py).

eval_seq itself refuses a run whose CES_SEQ_EXTRA_TI width differs from the one in metrics.json; this
wrapper adds that the table is the one the run was trained with (b53_features.json).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import register_v2hx as RV


def main():
    RV.check_env()
    out = Path(os.environ["CES_OUTPUT_DIR"])
    rec = json.loads((out / "b53_features.json").read_text(encoding="utf-8"))
    if Path(rec["feature_dir"]).resolve() != Path(os.environ["CES_SEQ_EXTRA_TI"]).resolve():
        raise SystemExit(f"FATAL: trained with {rec['feature_dir']}, scored with {os.environ['CES_SEQ_EXTRA_TI']}")
    import eval_seq
    reg = RV.register()
    if eval_seq.SEQ_MODELS is not reg:
        raise SystemExit("FATAL: eval_seq does not use the shared registry dict; v2hx would be missed")
    eval_seq.compare()


if __name__ == "__main__":
    main()
    os._exit(0)
