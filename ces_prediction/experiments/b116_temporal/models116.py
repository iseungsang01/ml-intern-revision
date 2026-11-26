"""Causal F-only temporal architectures; importing this module performs no fit."""
import torch
from torch import nn
from torch.nn import functional as F

ARCHS = ('lstm', 'decay_gru', 'innovation_gru')
WIDTH, HIDDEN = 35, 64


def _head():
    return nn.Sequential(nn.LayerNorm(HIDDEN), nn.Linear(HIDDEN, HIDDEN),
                         nn.GELU(), nn.Linear(HIDDEN, 1))


class LSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.recurrent = nn.LSTM(WIDTH, HIDDEN, batch_first=True)
        self.readout = _head()

    def forward(self, x, lengths=None):
        h, _ = self.recurrent(x)
        return self.readout(h).squeeze(-1)

    def forward_step(self, x, state=None):
        if x.ndim == 2:
            x = x[:, None, :]
        assert x.shape[1:] == (1, WIDTH)
        h, state = self.recurrent(x, state)
        return self.readout(h[:, 0]).squeeze(-1), state


class _CellSequence(nn.Module):
    def forward(self, x, lengths=None):
        state = None
        outputs = []
        for t in range(x.shape[1]):
            pred, state = self.forward_step(x[:, t], state)
            outputs.append(pred)
        return torch.stack(outputs, dim=1)

    @staticmethod
    def row(x):
        if x.ndim == 3:
            assert x.shape[1] == 1
            x = x[:, 0]
        assert x.ndim == 2 and x.shape[-1] == WIDTH
        return x


class DecayGRU(_CellSequence):
    def __init__(self):
        super().__init__()
        self.recurrent = nn.GRUCell(WIDTH, HIDDEN)
        self.rate = nn.Linear(3, HIDDEN)
        self.readout = _head()

    def forward_step(self, x, state=None):
        x = self.row(x)
        h = x.new_zeros((len(x), HIDDEN)) if state is None else state
        condition = x[:, [15, 17, 18]]
        dt = torch.expm1(x[:, 15:16]).clamp_min(0)
        decay = torch.exp(-(F.softplus(self.rate(condition)) * dt).clamp_max(80))
        h = self.recurrent(x, h * decay)
        return self.readout(h).squeeze(-1), h


class InnovationGRU(_CellSequence):
    """Assimilate visible CES once when its causal source timestamp advances."""
    def __init__(self):
        super().__init__()
        self.recurrent = nn.GRUCell(WIDTH, HIDDEN)
        self.gain = nn.Linear(HIDDEN + 3, HIDDEN)
        self.innovation = nn.Linear(1, HIDDEN)
        self.readout = _head()

    def forward_step(self, x, state=None):
        x = self.row(x)
        has = x[:, 18:19] > .5
        age = torch.expm1(x[:, 17:18]).clamp_min(0)
        dt = torch.expm1(x[:, 15:16]).clamp_min(0)
        if state is None:
            h = x.new_zeros((len(x), HIDDEN))
            old_age = torch.zeros_like(age)
            old_has = torch.zeros_like(has)
        else:
            h, old_age, old_has = state
        transition = x.clone()
        transition[:, 16] = 0
        prior = self.recurrent(transition, h)
        event = has & (~old_has | ((dt + old_age - age) > 1e-5))
        residual = x[:, 16:17] - self.readout(prior)
        gain = torch.sigmoid(self.gain(torch.cat((prior, x[:, [15, 17, 18]]), dim=-1)))
        h = prior + event.to(x.dtype) * gain * torch.tanh(self.innovation(residual))
        return self.readout(h).squeeze(-1), (h, age, has)


def make_model(arch, seed):
    if arch not in ARCHS:
        raise ValueError(arch)
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    return {'lstm': LSTM, 'decay_gru': DecayGRU,
            'innovation_gru': InnovationGRU}[arch]()
