"""Two fixed scalar neural candidates; normalized scalar output only."""
import math
import torch
from torch import nn

ARCHITECTURES = ('resnet128', 'ple128')


class ScalarNet(nn.Module):
    def __init__(self, dim, architecture, knots=None):
        super().__init__()
        assert architecture in ARCHITECTURES
        self.architecture = architecture
        if architecture == 'resnet128':
            self.input = nn.Linear(dim, 128)
            self.blocks = nn.ModuleList([nn.Sequential(nn.Linear(128, 128), nn.GELU(),
                                                       nn.Linear(128, 128)) for _ in range(3)])
            self.output = nn.Linear(128, 1)
        else:
            assert knots is not None and tuple(knots.shape) == (dim, 17)
            self.register_buffer('knots', torch.as_tensor(knots, dtype=torch.float32))
            self.layers = nn.Sequential(nn.Linear(dim * 16, 128), nn.GELU(),
                                        nn.Linear(128, 128), nn.GELU(), nn.Linear(128, 1))

    def forward(self, x):
        if self.architecture == 'resnet128':
            h = torch.nn.functional.gelu(self.input(x))
            for block in self.blocks:
                h = (h + block(h)) / math.sqrt(2)
            return self.output(h).squeeze(-1)
        width = (self.knots[:, 1:] - self.knots[:, :-1]).clamp_min(1e-6)
        encoded = ((x.unsqueeze(-1) - self.knots[:, :-1]) / width).clamp(0, 1)
        return self.layers(encoded.flatten(1)).squeeze(-1)
