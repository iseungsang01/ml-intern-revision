"""B.53 arm C: seq/train_seq.py with `v2hx` registered (launched by run_b53.py instead of train_seq.py).

Imports train_seq as a module, adds v2hx to the registry dict train_seq bound at import, checks that
the feature table belongs to this split, runs train_seq.main() and writes b53_features.json (the
table's meta) next to metrics.json.  No shared file is edited.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import register_v2hx as RV


def main():
    meta = RV.check_env()
    import train_seq
    reg = RV.register()
    if train_seq.SEQ_MODELS is not reg:
        raise SystemExit("FATAL: train_seq does not use the shared registry dict; v2hx would be missed")
    print(f"[b53] train_seq with v2hx, extra T_i channels {meta['components']} from {os.environ['CES_SEQ_EXTRA_TI']}",
          flush=True)
    train_seq.main()
    out = Path(os.environ["CES_OUTPUT_DIR"])
    m = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    if m["seq_model"] != RV.MODEL_NAME or m["extra_ti_channels"] != meta["k"]:
        raise SystemExit(f"FATAL: metrics.json records model {m['seq_model']} with {m['extra_ti_channels']} extra channels")
    (out / "b53_features.json").write_text(json.dumps({"feature_dir": os.environ["CES_SEQ_EXTRA_TI"], **meta}, indent=1),
                                           encoding="utf-8")


if __name__ == "__main__":
    main()
    os._exit(0)   # as train_seq: skip the Windows CUDA teardown so the return code reflects the run
