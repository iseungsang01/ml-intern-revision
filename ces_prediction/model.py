"""The architecture the rest of the pipeline trains and scores.

`from model import MultimodalCESPredictor` resolves to **`model_iter009.py`** — the
byte-identical archive snapshot that produced every number in the paper,
`THESIS_RESULTS.md`, and the decks. It is pinned by SHA-256 in
`tests/test_architecture.py`; do not edit it.

Controlled experiments that vary the architecture (e.g. `experiments/anchor/`) point
`CES_MODEL_FILE` at their own `.py` file instead. Like every other `CES_*` knob, a runner
sets it in the subprocess env, so `train.py` / `evaluate.py` / `compare_baselines.py` all
pick up the same variant without anything on disk being rewritten.

The variant file must define `MultimodalCESPredictor` with the fixed contract signature
`forward(self, bes, ecei, mc, time_features=None, ces_history=None) -> (batch, 2)`.
"""

import importlib.util
import os
from pathlib import Path

_MODEL_FILE = os.environ.get("CES_MODEL_FILE", "").strip()

if _MODEL_FILE:
    _path = Path(_MODEL_FILE).resolve()
    if not _path.is_file():
        raise SystemExit(f"CES_MODEL_FILE does not exist: {_path}")
    _spec = importlib.util.spec_from_file_location("_ces_architecture", _path)
    _module = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_module)
    if not hasattr(_module, "MultimodalCESPredictor"):
        raise SystemExit(f"CES_MODEL_FILE defines no MultimodalCESPredictor: {_path}")
    MultimodalCESPredictor = _module.MultimodalCESPredictor
    ARCHITECTURE_SOURCE = str(_path)
else:
    from model_iter009 import MultimodalCESPredictor  # noqa: F401

    ARCHITECTURE_SOURCE = str(Path(__file__).resolve().parent / "model_iter009.py")

__all__ = ["MultimodalCESPredictor", "ARCHITECTURE_SOURCE"]
