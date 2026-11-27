"""Same seeded B116 LSTM core, zero-initialized visible-CES residual output."""
import torch
import common119 as C
import models116 as B


class AnchorLSTM(B.LSTM):
    def __init__(self):
        super().__init__()
        torch.nn.init.zeros_(self.readout[3].weight)
        torch.nn.init.zeros_(self.readout[3].bias)

    @staticmethod
    def anchor(x):
        return torch.where(x[..., 18] > .5, x[..., 16], torch.zeros_like(x[..., 16]))

    def forward(self, x, lengths=None):
        return self.anchor(x) + super().forward(x, lengths)

    def forward_step(self, x, state=None):
        row = x[:, 0] if x.ndim == 3 else x
        delta, state = super().forward_step(x, state)
        return self.anchor(row) + delta, state


def make_model(arch, seed):
    assert arch == C.ARCH
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    return AnchorLSTM()
