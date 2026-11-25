"""One fixed small tabular MLP; normalized scalar output only."""
import torch
from torch import nn


class ScalarMLP(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, 64), nn.GELU(), nn.Linear(64, 64), nn.GELU(), nn.Linear(64, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def portable_weights(model):
    return {str(i) + '_' + name: value.detach().cpu().numpy().copy()
            for i in (0, 2, 4) for name, value in model.net[i].named_parameters()}
