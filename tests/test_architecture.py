import hashlib
import os
import sys
from pathlib import Path

import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../ces_prediction")))

from dataset import KSTAR_CES_Dataset
from model import MultimodalCESPredictor

# The two architectures behind published numbers, pinned by content hash. Neither is model.py
# (which the AutoML search later rewrote into a Transformer), so they can neither drift silently
# nor be "cleaned up" without this test failing. Provenance: PROJECT_KNOWLEDGE.md.
CES_DIR = Path(__file__).resolve().parents[1] / "ces_prediction"
PINNED_ARCHITECTURES = {
    # final model: every headline number
    "model_iter009.py": (
        "c6e3dbb0d63625d200eba7a95847c3ec1c1d755ca5dab74c59164db030d287da",
        "Mask the CES-history attention pool",
    ),
    # the "before" model of the honest-progression figure (§6)
    "model_iter002.py": (
        "20c41cce3c527d4310279cf6d75630583b6c58dff1e2670721f70654ae5c3226",
        "Target-aware fusion",
    ),
}


def test_published_architectures_are_present_and_unmodified():
    for name, (expected, marker) in PINNED_ARCHITECTURES.items():
        path = CES_DIR / name
        assert path.exists(), f"a published architecture is missing: {path}"
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == expected, (
            f"{name} no longer matches its archived snapshot ({digest} != {expected})"
        )
        text = path.read_text(encoding="utf-8")
        assert marker in text  # the marker the runners check
        assert "class MultimodalCESPredictor" in text


def test_dry_run():
    model = MultimodalCESPredictor(window_size=10)

    bes = torch.randn(2, 10, 9)
    ecei = torch.randn(2, 10, 4)
    mc = torch.randn(2, 10, 2)
    time_features = torch.rand(2, 10, 4)

    out = model(bes, ecei, mc, time_features)
    assert out.shape == (2, 2)

    loss = out.sum()
    loss.backward()


def test_real_csv_sample_forward():
    data_dir = Path(os.environ.get("CES_DATA_DIR", Path(__file__).resolve().parents[1] / "data"))
    dataset = KSTAR_CES_Dataset(data_dir=data_dir, window_size=10)
    sample = dataset[0]

    assert dataset.feature_dims == {"bes": 9, "ecei": 4, "mc": 2, "time": 4, "ces_history": 4}
    assert sample["time_features"].shape == (10, 4)
    assert sample["ces_history"].shape == (10, 4)
    assert sample["target_mask"].shape == (2,)
    assert torch.all((sample["target_mask"] == 0) | (sample["target_mask"] == 1))
    assert sample["target_mask"].sum().item() >= 1
    assert sample["time_features"][-1, 0].item() == 0.0
    assert torch.all(sample["time_features"][:, 1] >= 0.0)
    # target timestep fully masked in CES history (both values and both flags)
    assert torch.all(sample["ces_history"][-1, :] == 0.0)

    model = MultimodalCESPredictor.from_dataset(dataset, window_size=10)
    out = model(
        sample["bes"].unsqueeze(0),
        sample["ecei"].unsqueeze(0),
        sample["mc"].unsqueeze(0),
        sample["time_features"].unsqueeze(0),
        sample["ces_history"].unsqueeze(0),
    )

    assert out.shape == (1, 2)


def test_temporal_subset_sample_uses_previous_ces_only():
    data_dir = Path(os.environ.get("CES_DATA_DIR", Path(__file__).resolve().parents[1] / "data"))
    dataset = KSTAR_CES_Dataset(
        data_dir=data_dir,
        window_size=4,
        temporal_subset_augmentation=True,
    )
    sample = dataset[0]
    valid_steps = int(sample["input_mask"].sum().item())

    assert 2 <= valid_steps <= 4
    assert sample["bes"].shape == (4, 9)
    assert sample["target"].shape == (2,)
    assert sample["target_mask"].shape == (2,)
    assert sample["ces_history"].shape == (4, 4)
    # target timestep fully masked: both normalized values and both observed flags
    assert sample["ces_history"][valid_steps - 1, :2].abs().sum().item() == 0.0
    assert torch.all(sample["ces_history"][valid_steps - 1, 2:] == 0.0)
    # every observed history row has at least one observed CES target (TI or VT)
    assert torch.all(sample["ces_history"][: valid_steps - 1, 2:].sum(dim=1) >= 1.0)
    assert torch.all(sample["ces_history"][valid_steps:, :] == 0.0)
