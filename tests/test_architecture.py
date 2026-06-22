import os
import sys
from pathlib import Path

import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../ces_prediction")))

from dataset import KSTAR_CES_Dataset
from model import MultimodalCESPredictor


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
