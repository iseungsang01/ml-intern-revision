"""Minimal-operator online steps — the same math, without the fat `nn.Module` call paths.

B.9 axis C's first pass measured every family through its stock module, and every arm that
beats persistence landed at or above a 1 ms deadline. Profiling the operators explains why,
and the explanation is not the model:

    torch.mm (32x32, 32k MAC)        1.5 us
    nn.LSTM, 1 layer, hidden 8       70.1 us   (~1k MAC)
    nn.LSTMCell, hidden 8            25.7 us
    nn.Conv1d, 128 channels, k=3    214.8 us
    nn.MultiheadAttention, d=128    108.8 us

`nn.LSTM`'s single-step path costs 47x a matrix multiply that does 32x more arithmetic. The
budget was being spent on module machinery — packing, shape checks, the cuDNN-shaped call
protocol — not on the network. So the arms are re-expressed here with explicit `addmm` /
`bmm` calls over the SAME parameters, and each one is asserted equal to its model's own
forward before it is timed.

**Every family gets the same treatment, on purpose.** Hand-optimising only the recurrent arm
would have manufactured §8ah's "recurrence is cheapest" conclusion; the dilated conv becomes
three `addmm`s per layer (a kernel-3 convolution over a ring buffer IS three matrix
multiplies) and the attention block becomes a packed QKV projection plus two `bmm`s over a
preallocated KV cache. Whatever ordering survives that is a property of the operator, not of
which one someone bothered to optimise.

Used by `bench_budget.py`; equivalence is enforced there and by
`tests/test_architecture.py::test_lean_steps_match_their_models`.
"""

import torch
import torch.nn as nn


class LeanSeqV2Step(nn.Module):
    """seq_v2 / its width variants: explicit LSTM cells, stock LayerNorm and heads.

    LayerNorm and Linear are left as ATen calls because they are already single fused
    kernels (5.3 us / 7.3 us); only `nn.LSTM` is worth replacing.
    """

    def __init__(self, model):
        super().__init__()
        self.m = model
        self.n_fast = model.n_fast
        self.w_ti = self._pack(model.lstm_ti)
        self.w_vt = self._pack(model.lstm_vt)

    @staticmethod
    def _pack(lstm):
        """Per layer: (W_ih^T, W_hh^T, b_ih + b_hh) — the two biases are only ever summed."""
        return [(getattr(lstm, f"weight_ih_l{i}").t().contiguous(),
                 getattr(lstm, f"weight_hh_l{i}").t().contiguous(),
                 getattr(lstm, f"bias_ih_l{i}") + getattr(lstm, f"bias_hh_l{i}"))
                for i in range(lstm.num_layers)]

    def init_state(self, lstm):
        return ([torch.zeros(1, lstm.hidden_size) for _ in range(lstm.num_layers)],
                [torch.zeros(1, lstm.hidden_size) for _ in range(lstm.num_layers)])

    def stream_init(self):
        h_ti, c_ti = self.init_state(self.m.lstm_ti)
        h_vt, c_vt = self.init_state(self.m.lstm_vt)
        return {"h_ti": h_ti, "c_ti": c_ti, "h_vt": h_vt, "c_vt": c_vt}

    @staticmethod
    def _run(layers, x, hs, cs):
        for k, (wi, wh, b) in enumerate(layers):
            g = torch.addmm(b, x, wi).addmm_(hs[k], wh)
            i, f, gg, o = g.chunk(4, 1)
            cs[k] = torch.sigmoid(f) * cs[k] + torch.sigmoid(i) * torch.tanh(gg)
            hs[k] = torch.sigmoid(o) * torch.tanh(cs[k])
            x = hs[k]
        return x

    def forward(self, x_t, state):
        """x_t (1, n_in) -> (1, 2). State is mutated in place, as an online loop would."""
        m = self.m
        h_ti = self._run(self.w_ti, x_t, state["h_ti"], state["c_ti"])
        h_vt = self._run(self.w_vt, x_t[:, self.n_fast:], state["h_vt"], state["c_vt"])
        return torch.cat([m.head_ti(m.norm_ti(h_ti)), m.head_vt(m.norm_vt(h_vt))], dim=-1)


class LeanTCNStep(nn.Module):
    """Dilated causal TCN: a kernel-3 convolution over a ring buffer is three `addmm`s.

    `nn.Conv1d` on a (1, C, 2d+1) buffer costs 34-215 us depending on width; the same
    arithmetic as three matrix multiplies over the three tapped positions costs ~5 us.
    """

    def __init__(self, model):
        super().__init__()
        self.m = model
        self.n_fast = model.n_fast
        self.ti = self._pack(model.tcn_ti)
        self.vt = self._pack(model.tcn_vt)

    @staticmethod
    def _pack(stack):
        """Per layer: (taps as [W0^T, W1^T, W2^T], bias, dilation)."""
        out = []
        for conv, pad in zip(stack.convs, stack.pads):
            w = conv.weight                                  # (out, in, k)
            taps = [w[:, :, j].t().contiguous() for j in range(w.shape[2])]
            out.append((taps, conv.bias, pad // (w.shape[2] - 1)))
        return out

    def _init(self, stack):
        hidden = stack.proj.out_features
        # Each layer keeps its own inputs for the positions its taps read.
        return [[torch.zeros(1, hidden) for _ in range(pad + 1)] for pad in stack.pads]

    def stream_init(self):
        return {"ti": self._init(self.m.tcn_ti), "vt": self._init(self.m.tcn_vt)}

    @staticmethod
    def _run(stack, layers, buffers, x):
        h = stack.proj(x)
        for k, ((taps, bias, dilation), norm) in enumerate(zip(layers, stack.norms)):
            buf = buffers[k]
            buf.append(h)
            del buf[0]
            # taps read t-2d, t-d, t — the same positions the dilated kernel covers.
            z = torch.addmm(bias, buf[-1 - 2 * dilation], taps[0])
            z = z.addmm_(buf[-1 - dilation], taps[1]).addmm_(buf[-1], taps[2])
            h = norm(h + nn.functional.gelu(z))
        return h

    def forward(self, x_t, state):
        m = self.m
        h_ti = self._run(m.tcn_ti, self.ti, state["ti"], x_t)
        h_vt = self._run(m.tcn_vt, self.vt, state["vt"], x_t[:, self.n_fast:])
        return torch.cat([m.head_ti(h_ti), m.head_vt(h_vt)], dim=-1)


class LeanXfmrStep(nn.Module):
    """Causal banded attention: packed QKV projection + two `bmm`s over a KV cache.

    `nn.MultiheadAttention` costs 108.8 us at d = 128 for a single query; the same
    computation written out is a handful of ops, because one query against a 32-step band
    is a tiny matrix product.
    """

    def __init__(self, model):
        super().__init__()
        self.m = model
        self.n_fast = model.n_fast

    def _init(self, enc):
        band = enc.band
        return [{"k": torch.zeros(1, band, enc.d_model),
                 "v": torch.zeros(1, band, enc.d_model),
                 "n": 0} for _ in enc.attn]

    def stream_init(self):
        return {"ti": self._init(self.m.enc_ti), "vt": self._init(self.m.enc_vt)}

    @staticmethod
    def _run(enc, caches, x):
        h = enc.proj(x).unsqueeze(1)                                  # (1, 1, d)
        for attn, ff, n1, n2, cache in zip(enc.attn, enc.ff, enc.n1, enc.n2, caches):
            z = n1(h)
            qkv = torch.addmm(attn.in_proj_bias, z[:, 0], attn.in_proj_weight.t())
            q, k, v = qkv.chunk(3, 1)
            # Roll the band by one and write the new key/value at the end.
            cache["k"] = torch.cat([cache["k"][:, 1:], k.unsqueeze(1)], dim=1)
            cache["v"] = torch.cat([cache["v"][:, 1:], v.unsqueeze(1)], dim=1)
            cache["n"] = min(cache["n"] + 1, enc.band)
            span, band = cache["n"], enc.band
            keys = cache["k"][:, band - span:]
            vals = cache["v"][:, band - span:]
            heads, dim = attn.num_heads, enc.d_model // attn.num_heads
            qh = q.view(heads, 1, dim)
            kh = keys.view(span, heads, dim).transpose(0, 1)
            vh = vals.view(span, heads, dim).transpose(0, 1)
            bias = -enc.slope * torch.arange(span - 1, -1, -1, dtype=q.dtype)
            scores = torch.bmm(qh, kh.transpose(1, 2)) * (dim ** -0.5) + bias
            ctx = torch.bmm(torch.softmax(scores, -1), vh).reshape(1, enc.d_model)
            h = h + torch.addmm(attn.out_proj.bias, ctx, attn.out_proj.weight.t()).unsqueeze(1)
            h = h + ff(n2(h))
        return h[:, 0]

    def forward(self, x_t, state):
        m = self.m
        h_ti = self._run(m.enc_ti, state["ti"], x_t)
        h_vt = self._run(m.enc_vt, state["vt"], x_t[:, self.n_fast:])
        return torch.cat([m.head_ti(h_ti), m.head_vt(h_vt)], dim=-1)


LEAN = {"lstm": LeanSeqV2Step, "tcn": LeanTCNStep, "xfmr": LeanXfmrStep}


def build(model):
    """Pick the lean step for a model, by the branch attributes it carries."""
    if hasattr(model, "lstm_ti"):
        return LeanSeqV2Step(model).eval()
    if hasattr(model, "tcn_ti"):
        return LeanTCNStep(model).eval()
    if hasattr(model, "enc_ti"):
        return LeanXfmrStep(model).eval()
    raise SystemExit(f"no lean step for {type(model).__name__}")
