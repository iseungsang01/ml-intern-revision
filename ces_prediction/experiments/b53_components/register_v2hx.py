"""B.53: add `v2hx` to the seq registry of this process, and refuse a feature table of another split.

Used by train_b53.py and eval_b53.py (arm C).  train_seq and eval_seq bind the registry dict itself
(`from seq_models import SEQ_MODELS`), so an entry added to that dict before their main() runs is seen
by both.  No shared file is edited.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SEQ_DIR = HERE.parent / "seq"
CES_DIR = HERE.parents[1]
for p in (str(HERE), str(CES_DIR), str(SEQ_DIR)):
    if p in sys.path:
        sys.path.remove(p)
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(SEQ_DIR))   # seq first, as train_seq / eval_seq put it
sys.path.insert(1, str(CES_DIR))

MODEL_NAME = "v2hx"


def check_env():
    """The arm-C pairing: model v2hx <-> a feature table written for THIS split (CES_SPLIT_DIR)."""
    model = os.environ.get("CES_SEQ_MODEL", "")
    feat = os.environ.get("CES_SEQ_EXTRA_TI", "").strip()
    if model != MODEL_NAME or not feat:
        raise SystemExit(f"FATAL: B.53 arm C needs CES_SEQ_MODEL={MODEL_NAME} and CES_SEQ_EXTRA_TI set "
                         f"(got model {model!r}, table {feat!r})")
    if os.environ.get("CES_SEQ_EXTRA_VT", "").strip():
        raise SystemExit("FATAL: B.53 arm C is defined without CES_SEQ_EXTRA_VT")
    meta = json.loads((Path(feat) / "feature_meta.json").read_text(encoding="utf-8"))
    if Path(meta["split_dir"]).resolve() != Path(os.environ["CES_SPLIT_DIR"]).resolve():
        raise SystemExit(f"FATAL: the feature table {feat} was standardized on split {meta['split_dir']}, "
                         f"this run uses {os.environ['CES_SPLIT_DIR']}")
    if sorted(meta["z_exempt_channels"]) != list(range(int(meta["k"]))):
        raise SystemExit(f"FATAL: {feat}: every channel must be z-score exempt (B.53 section 3)")
    return meta


def register():
    import seq_models
    from model_seq_v2hx import SeqCESLSTMv2HistX
    if MODEL_NAME in seq_models.SEQ_MODELS and seq_models.SEQ_MODELS[MODEL_NAME] is not SeqCESLSTMv2HistX:
        raise SystemExit(f"FATAL: {MODEL_NAME} is already in the seq registry with another class")
    seq_models.SEQ_MODELS[MODEL_NAME] = SeqCESLSTMv2HistX
    return seq_models.SEQ_MODELS
