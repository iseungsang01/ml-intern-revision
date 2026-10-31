"""B.20: the val median-AE selection key (CES_SEQ_SELECT=medae). No data needed."""
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "ces_prediction", "experiments", "seq"))

from train_seq import masked_pass, median_ae_key  # noqa: E402


def test_key_is_mean_of_per_target_medians():
    a = [[np.array([1.0, 3.0]), np.array([2.0])], [np.array([10.0, 0.0, 4.0])]]
    assert median_ae_key(a) == (2.0 + 4.0) / 2


class _Const(torch.nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c = torch.nn.Parameter(torch.tensor(c))

    def forward(self, x, lengths):
        return self.c.expand(x.shape[0], x.shape[1], 2)


def test_masked_pass_collects_observed_rows_only(monkeypatch):
    import train_seq
    y = torch.tensor([[[1.0, 5.0], [2.0, 7.0], [9.0, 9.0]]])
    m = torch.tensor([[[1.0, 1.0], [1.0, 0.0], [0.0, 1.0]]])
    monkeypatch.setattr(train_seq, "batched", lambda *a, **k: iter([(torch.zeros(1, 3, 4), y, m, [3])]))
    ae = [[], []]
    masked_pass(_Const([0.0, 0.0]), None, 1, "cpu", -10.0, abs_err=ae)
    assert np.allclose(np.concatenate(ae[0]), [1.0, 2.0]) and np.allclose(np.concatenate(ae[1]), [5.0, 9.0])
    assert median_ae_key(ae) == (1.5 + 7.0) / 2
