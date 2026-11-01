"""B.25: the per-row readout keeps exactly the anchor-defined observed rows and its anchor equals
the model's anchor_only path. Untrained models on a hand-built block: arithmetic only, no data."""
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "ces_prediction", "experiments", "seq"))
sys.path.insert(0, os.path.join(ROOT, "ces_prediction", "experiments", "b16_physics"))

from residual_rows_b25 import readout  # noqa: E402
from seq_data import N_FEATURES  # noqa: E402
from seq_models import SEQ_MODELS  # noqa: E402


def _block(L=30, seed=0):
    g = np.random.default_rng(seed)
    x = g.normal(size=(L, N_FEATURES)).astype(np.float32)
    mask = (g.random((L, 2)) < 0.4).astype(np.float32)
    y = np.where(mask > 0, g.normal(size=(L, 2)), 0).astype(np.float32)
    return {"x": x, "y": y, "mask": mask, "time": np.arange(L) * 0.01}


def test_rows_and_anchor_check():
    torch.manual_seed(0)
    blk = _block()
    mean, std = np.array([600.0, 70.0]), np.array([650.0, 75.0])
    for variant in ("v4kjt", "v4kjtf"):
        m = SEQ_MODELS[variant]().eval()
        rows = readout(m, [(0, blk)], torch.device("cpu"), mean, std)
        for t in range(2):
            obs = np.flatnonzero(blk["mask"][:, t] > 0)
            assert len(rows[t]["y"]) == max(len(obs) - 1, 0)
            np.testing.assert_allclose(rows[t]["y"], blk["y"][obs[1:], t] * std[t] + mean[t], rtol=1e-6)
            np.testing.assert_allclose(rows[t]["prev"], blk["y"][obs[:-1], t] * std[t] + mean[t], rtol=1e-6)
            if variant == "v4kjt":
                assert np.all(rows[t]["g"] == 1.0)
            else:
                assert np.all((rows[t]["g"] > 0) & (rows[t]["g"] < 1))
