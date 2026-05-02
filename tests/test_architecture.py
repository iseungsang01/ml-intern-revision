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
    data_dir = Path(__file__).resolve().parents[1] / "data"
    dataset = KSTAR_CES_Dataset(data_dir=data_dir, window_size=10)
    sample = dataset[0]

    assert dataset.feature_dims == {"bes": 9, "ecei": 4, "mc": 2, "time": 4}
    assert sample["time_features"].shape == (10, 4)
    assert sample["time_features"][-1, 0].item() == 0.0
    assert torch.all(sample["time_features"][:, 1] >= 0.0)

    model = MultimodalCESPredictor.from_dataset(dataset, window_size=10)
    out = model(
        sample["bes"].unsqueeze(0),
        sample["ecei"].unsqueeze(0),
        sample["mc"].unsqueeze(0),
        sample["time_features"].unsqueeze(0),
    )

    assert out.shape == (1, 2)
