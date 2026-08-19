"""The seq_v2 online step compressed to as few dispatched operators as it can be.

The lean step (`lean_steps.LeanSeqV2Step`) already removed `nn.LSTM`'s call protocol and
got 5.9x. Profiling what was left showed the remaining cost is still almost exactly linear
in **operator count** — 1.3-1.6 us per ATen op, and essentially independent of arithmetic
(tcn15 does half of seq_v2's MACs and takes longer, because it dispatches more ops). So the
way to go faster is not less arithmetic, it is fewer dispatches.

Four fusions, all exact:

1. **Both branches' first layer in one matmul.** `T_i` reads all 22 channels and `V_rot`
   reads the 7-channel tail, so a single block matrix over `[x, h_ti, h_vt]` produces both
   gate vectors at once — with zeros where a branch must not see a channel, which also keeps
   §8ab's routing structurally exact rather than merely intended.
2. **Input and recurrent weights in one matmul.** `W_ih @ x + W_hh @ h` becomes
   `[W_ih | W_hh] @ [x; h]`: one `addmm` instead of two.
3. **Gates reordered so each activation is one contiguous slice.** PyTorch lays gates out
   `i, f, g, o`, which needs sigmoid on three separated blocks and tanh on one. Permuting the
   weight rows at build time to `[i, f, o | g]` — across both branches — makes it exactly one
   `sigmoid` and one `tanh` per step, for the whole model.
4. **The cell update runs on both branches at once**, because their states are stored as one
   concatenated vector, so `c = f*c + i*g` and `h = o*tanh(c)` are 5 ops total rather than 5
   per branch.

Nothing is approximated: the permutation and the block layout are re-indexings of the same
weights, and `tests/test_architecture.py::test_tight_step_matches_its_model` requires the
output to match the model's own forward.

Restricted to the `seq_v2` family (`SeqCESLSTMv2` and its width variants), which is the one
the 1 ms budget actually turns on.
"""

import torch
import torch.nn as nn


class TightSeqV2Step(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.m = model
        self.n_fast = model.n_fast
        self.h1 = model.lstm_ti.hidden_size
        self.h2 = model.lstm_vt.hidden_size
        self.deep_ti = model.lstm_ti.num_layers - 1
        if model.lstm_vt.num_layers != 1:
            raise SystemExit("tight step assumes a single V_rot layer (seq_v2's shape)")

        n_in = model.lstm_ti.input_size
        h1, h2 = self.h1, self.h2
        # --- fusion 1 + 2 + 3: one weight for [x | h_ti | h_vt] -> [i,f,o | g] of both ---
        wi_ti = self._perm(model.lstm_ti.weight_ih_l0, h1)     # (4h1, n_in)
        wh_ti = self._perm(model.lstm_ti.weight_hh_l0, h1)     # (4h1, h1)
        wi_vt = self._perm(model.lstm_vt.weight_ih_l0, h2)     # (4h2, n_slow)
        wh_vt = self._perm(model.lstm_vt.weight_hh_l0, h2)     # (4h2, h2)
        b_ti = self._perm_b(model.lstm_ti.bias_ih_l0 + model.lstm_ti.bias_hh_l0, h1)
        b_vt = self._perm_b(model.lstm_vt.bias_ih_l0 + model.lstm_vt.bias_hh_l0, h2)

        # Gate-MAJOR across branches: [i_ti i_vt | f_ti f_vt | o_ti o_vt | g_ti g_vt].
        # Branch-major would make `sigmoid(gates[:, :3n])` slice the wrong rows -- which is
        # exactly the bug the equivalence assert caught on the first version.
        n = h1 + h2
        cols = n_in + h1 + h2
        w = torch.zeros(4 * n, cols)
        b = torch.zeros(4 * n)
        for gate in range(4):                       # i, f, o, g  (already permuted)
            r = gate * n
            w[r:r + h1, :n_in] = wi_ti[gate * h1:(gate + 1) * h1]
            w[r:r + h1, n_in:n_in + h1] = wh_ti[gate * h1:(gate + 1) * h1]
            b[r:r + h1] = b_ti[gate * h1:(gate + 1) * h1]
            w[r + h1:r + n, self.n_fast:n_in] = wi_vt[gate * h2:(gate + 1) * h2]
            w[r + h1:r + n, n_in + h1:] = wh_vt[gate * h2:(gate + 1) * h2]
            b[r + h1:r + n] = b_vt[gate * h2:(gate + 1) * h2]

        self.register_buffer("w0", w.t().contiguous())
        self.register_buffer("b0", b)

        # Deeper T_i layers stay sequential (they consume the layer below), but still get
        # fusions 2 and 3.
        self.deep = []
        for i in range(1, model.lstm_ti.num_layers):
            wi = self._perm(getattr(model.lstm_ti, f"weight_ih_l{i}"), h1)
            wh = self._perm(getattr(model.lstm_ti, f"weight_hh_l{i}"), h1)
            bb = self._perm_b(getattr(model.lstm_ti, f"bias_ih_l{i}")
                              + getattr(model.lstm_ti, f"bias_hh_l{i}"), h1)
            self.deep.append((torch.cat([wi, wh], 1).t().contiguous(), bb))

    @staticmethod
    def _perm(weight, h):
        """PyTorch order i,f,g,o -> i,f,o,g so sigmoid and tanh are each one slice."""
        i, f, g, o = weight.split(h, 0)
        return torch.cat([i, f, o, g], 0)

    @staticmethod
    def _perm_b(bias, h):
        i, f, g, o = bias.split(h, 0)
        return torch.cat([i, f, o, g])

    def stream_init(self):
        return {"h": torch.zeros(1, self.h1 + self.h2),
                "c": torch.zeros(1, self.h1 + self.h2),
                "deep_h": [torch.zeros(1, self.h1) for _ in range(self.deep_ti)],
                "deep_c": [torch.zeros(1, self.h1) for _ in range(self.deep_ti)]}

    def forward(self, x_t, state):
        """x_t (1, n_in) -> (1, 2). Same weights, same output as the model's forward."""
        m = self.m
        v = torch.cat([x_t, state["h"]], 1)
        gates = torch.addmm(self.b0, v, self.w0)
        n = self.h1 + self.h2
        s = torch.sigmoid(gates[:, :3 * n])
        g = torch.tanh(gates[:, 3 * n:])
        c = s[:, n:2 * n] * state["c"] + s[:, :n] * g
        h = s[:, 2 * n:] * torch.tanh(c)
        state["h"], state["c"] = h, c

        top = h[:, :self.h1]
        for k, (wk, bk) in enumerate(self.deep):
            gk = torch.addmm(bk, torch.cat([top, state["deep_h"][k]], 1), wk)
            sk = torch.sigmoid(gk[:, :3 * self.h1])
            gg = torch.tanh(gk[:, 3 * self.h1:])
            ck = sk[:, self.h1:2 * self.h1] * state["deep_c"][k] + sk[:, :self.h1] * gg
            top = sk[:, 2 * self.h1:] * torch.tanh(ck)
            state["deep_h"][k], state["deep_c"][k] = top, ck

        return torch.cat([m.head_ti(m.norm_ti(top)),
                          m.head_vt(m.norm_vt(h[:, self.h1:]))], dim=-1)


class TightXfmrStep(nn.Module):
    """The attention step with the same fusions applied, so the family comparison is fair.

    `LeanXfmrStep` already replaced `nn.MultiheadAttention`. What remained per layer was
    bookkeeping that costs dispatches without doing arithmetic, and at 1.3-1.6 us per
    operator that bookkeeping IS the cost:

    1. **The ALiBi bias was rebuilt every step** (`arange` + `mul`). It only depends on the
       span, so it is precomputed once per layer.
    2. **Keys and values were cached separately** — two slices and two concatenations per
       layer. They are produced by one projection, so they are cached as one `(1, band, 2d)`
       tensor: one slice, one concatenation.
    3. **The attention scale was a separate multiply.** Folded into the query rows of the
       packed QKV weight at build time, which is exact.
    4. **Bias-add and the score product were separate.** `baddbmm` does both.

    Same weights, same output — asserted against the model's own forward like every other
    step in this directory.
    """

    def __init__(self, model):
        super().__init__()
        self.m = model
        self.n_fast = model.n_fast
        self.ti = self._pack(model.enc_ti)
        self.vt = self._pack(model.enc_vt)

    @staticmethod
    def _pack(enc):
        packed = []
        for attn in enc.attn:
            d = enc.d_model
            heads = attn.num_heads
            dim = d // heads
            w = attn.in_proj_weight.clone()
            b = attn.in_proj_bias.clone()
            w[:d] *= dim ** -0.5          # fold the softmax scale into the query rows
            b[:d] *= dim ** -0.5
            bias = -enc.slope * torch.arange(enc.band - 1, -1, -1, dtype=w.dtype)
            packed.append({"w": w.t().contiguous(), "b": b, "heads": heads, "dim": dim,
                           "bias": bias.view(1, 1, enc.band)})
        return packed

    def _init(self, enc):
        return [torch.zeros(1, enc.band, 2 * enc.d_model) for _ in enc.attn]

    def stream_init(self):
        return {"ti": self._init(self.m.enc_ti), "vt": self._init(self.m.enc_vt),
                "n": [0, 0]}

    def _run(self, enc, packed, caches, x, filled):
        """`filled` = how many cache slots hold real steps. Slots never written must not be
        attended: the batch path masks them with -inf, so including their zeros would put
        phantom keys in the softmax. Missing this is what the equivalence assert caught."""
        h = enc.proj(x).unsqueeze(1)
        d = enc.d_model
        band = enc.band
        span = min(filled + 1, band)
        for i, (p, ff, n1, n2) in enumerate(zip(packed, enc.ff, enc.n1, enc.n2)):
            z = n1(h)
            qkv = torch.addmm(p["b"], z[:, 0], p["w"])
            kv = torch.cat([caches[i][:, 1:], qkv[:, d:].view(1, 1, 2 * d)], 1)
            caches[i] = kv
            heads, dim = p["heads"], p["dim"]
            live = kv[:, band - span:]
            kh = live[..., :d].view(span, heads, dim).transpose(0, 1)
            vh = live[..., d:].view(span, heads, dim).transpose(0, 1)
            qh = qkv[:, :d].view(heads, 1, dim)
            scores = torch.baddbmm(p["bias"][..., band - span:], qh, kh.transpose(1, 2))
            ctx = torch.bmm(torch.softmax(scores, -1), vh).reshape(1, d)
            attn = enc.attn[i].out_proj
            h = h + torch.addmm(attn.bias, ctx, attn.weight.t()).unsqueeze(1)
            h = h + ff(n2(h))
        return h[:, 0]

    def forward(self, x_t, state):
        m = self.m
        h_ti = self._run(m.enc_ti, self.ti, state["ti"], x_t, state["n"][0])
        h_vt = self._run(m.enc_vt, self.vt, state["vt"], x_t[:, self.n_fast:], state["n"][1])
        state["n"][0] = min(state["n"][0] + 1, m.enc_ti.band)
        state["n"][1] = min(state["n"][1] + 1, m.enc_vt.band)
        return torch.cat([m.head_ti(h_ti), m.head_vt(h_vt)], dim=-1)


def build(model):
    """Pick the tight step for a model; only the two families that have one."""
    if hasattr(model, "lstm_ti"):
        return TightSeqV2Step(model).eval()
    if hasattr(model, "enc_ti"):
        return TightXfmrStep(model).eval()
    raise SystemExit(f"no tight step for {type(model).__name__}")
