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


class TightTCNStep(nn.Module):
    """The TCN step with both branches packed into ONE state vector — same fusions as the
    LSTM tight step, applied to the convolutional family.

    `LeanTCNStep` already turned each `nn.Conv1d` into three `addmm`s over a ring buffer;
    what remains is that every layer still runs TWICE (once per branch) and every kernel tap
    is its own matmul. Three fusions, all exact:

    1. **Both branches' projection in one matmul.** `T_i` reads all `n_in` channels and
       `V_rot` reads the slow tail, so a `(n_in -> h1+h2)` block matrix with zeros where the
       `V_rot` branch must not see a channel produces both hidden vectors at once — §8ab's
       routing enforced structurally, exactly like fusion 1 of `TightSeqV2Step`.
    2. **The three taps in one matmul, for both branches.** The layer's three tapped
       positions are concatenated into one `(1, 3(h1+h2))` vector and hit with one
       block-diagonal `(3(h1+h2) -> h1+h2)` weight: one `addmm` where lean used six
       (3 taps x 2 branches).
    3. **GELU and the residual run on the packed vector**, so they are one op each per layer
       instead of one per branch. Only LayerNorm stays per-branch (the two widths normalize
       separately by definition), via one `split` and two `layer_norm`s.

    The heads collapse the same way: one `(h1+h2 -> 2*head)` matmul, one GELU, and one
    `(2*head -> 2)` matmul whose two rows ARE the `[CES_TI, CES_VT]` output — the final
    concatenation disappears into the weight layout.

    Nothing is approximated: every packed weight is a re-indexing of the model's own, and
    `bench_budget.make_arm`'s equivalence gate replays a 40-step block against the model's
    batch forward before this step is ever timed or counted.
    """

    def __init__(self, model):
        super().__init__()
        self.m = model
        ti, vt = model.tcn_ti, model.tcn_vt
        h1, h2 = ti.proj.out_features, vt.proj.out_features
        self.h1, self.h2 = h1, h2
        n = h1 + h2
        if len(ti.pads) != len(vt.pads) or ti.pads != vt.pads:
            raise SystemExit("tight TCN step assumes both branches share the dilation "
                             "schedule (SeqCESTCN builds them that way)")
        self.dilations = [pad // 2 for pad in ti.pads]      # kernel 3: pad = 2*dilation

        n_in = ti.proj.in_features
        w = torch.zeros(n, n_in)
        w[:h1] = ti.proj.weight
        w[h1:, model.n_fast:] = vt.proj.weight
        self.register_buffer("w0", w.t().contiguous())
        self.register_buffer("b0", torch.cat([ti.proj.bias, vt.proj.bias]))

        for k, (c1, c2) in enumerate(zip(ti.convs, vt.convs)):
            wt = torch.zeros(n, 3 * n)
            bt = torch.cat([c1.bias, c2.bias])
            for j in range(3):                              # tap j multiplies input t-(2-j)d
                wt[:h1, j * n:j * n + h1] = c1.weight[:, :, j]
                wt[h1:, j * n + h1:(j + 1) * n] = c2.weight[:, :, j]
            self.register_buffer(f"wc{k}", wt.t().contiguous())
            self.register_buffer(f"bc{k}", bt)

        # Heads: Linear -> GELU -> Linear per branch, packed so row 0 of the last matmul is
        # CES_TI and row 1 is CES_VT — the output needs no final cat.
        head = model.head_ti[0].out_features
        wh1 = torch.zeros(2 * head, n)
        wh1[:head, :h1] = model.head_ti[0].weight
        wh1[head:, h1:] = model.head_vt[0].weight
        wh2 = torch.zeros(2, 2 * head)
        wh2[0, :head] = model.head_ti[2].weight
        wh2[1, head:] = model.head_vt[2].weight
        self.register_buffer("wh1", wh1.t().contiguous())
        self.register_buffer("bh1", torch.cat([model.head_ti[0].bias, model.head_vt[0].bias]))
        self.register_buffer("wh2", wh2.t().contiguous())
        self.register_buffer("bh2", torch.cat([model.head_ti[2].bias, model.head_vt[2].bias]))

    def stream_init(self):
        n = self.h1 + self.h2
        # Python lists cost zero dispatches to rotate; zeros replay the batch left-pad.
        return [[torch.zeros(1, n) for _ in range(2 * d + 1)] for d in self.dilations]

    def forward(self, x_t, state):
        """x_t (1, n_in) -> (1, 2). Same weights, same output as the model's forward."""
        m = self.m
        h = torch.addmm(self.b0, x_t, self.w0)
        for k, d in enumerate(self.dilations):
            buf = state[k]
            buf.append(h)
            del buf[0]
            v = torch.cat([buf[-1 - 2 * d], buf[-1 - d], buf[-1]], 1)
            z = h + nn.functional.gelu(
                torch.addmm(getattr(self, f"bc{k}"), v, getattr(self, f"wc{k}")))
            a, b = z.split([self.h1, self.h2], 1)
            n1, n2 = m.tcn_ti.norms[k], m.tcn_vt.norms[k]
            h = torch.cat([nn.functional.layer_norm(a, (self.h1,), n1.weight, n1.bias, n1.eps),
                           nn.functional.layer_norm(b, (self.h2,), n2.weight, n2.bias, n2.eps)],
                          1)
        u = nn.functional.gelu(torch.addmm(self.bh1, h, self.wh1))
        return torch.addmm(self.bh2, u, self.wh2)


class _TCNStepPure(nn.Module):
    """The tight TCN step as a pure tensor function, so `torch.jit.trace` + `freeze` can
    collapse the remaining dispatches into fused kernels — §8aj's named next lever
    ("compile the step, don't shrink the model") applied to this family.

    State is the layer ring buffers as plain tensors `(1, n, 2d+1)`; each call returns the
    output and the shifted buffers. No python-side containers, so the traced graph is the
    whole step."""

    def __init__(self, tight):
        super().__init__()
        self.t = tight

    def forward(self, x_t, b0, b1, b2):
        t = self.t
        h1, h2 = t.h1, t.h2
        m = t.m
        h = torch.addmm(t.b0, x_t, t.w0)
        outs = []
        for k, (d, buf) in enumerate(zip(t.dilations, (b0, b1, b2))):
            nb = torch.cat([buf[:, :, 1:], h.unsqueeze(2)], 2)
            outs.append(nb)
            v = torch.cat([nb[:, :, 0], nb[:, :, d], nb[:, :, 2 * d]], 1)
            z = h + nn.functional.gelu(
                torch.addmm(getattr(t, f"bc{k}"), v, getattr(t, f"wc{k}")))
            a, b = z.split([h1, h2], 1)
            n1, n2 = m.tcn_ti.norms[k], m.tcn_vt.norms[k]
            h = torch.cat([nn.functional.layer_norm(a, (h1,), n1.weight, n1.bias, n1.eps),
                           nn.functional.layer_norm(b, (h2,), n2.weight, n2.bias, n2.eps)], 1)
        u = nn.functional.gelu(torch.addmm(t.bh1, h, t.wh1))
        return torch.addmm(t.bh2, u, t.wh2), outs[0], outs[1], outs[2]


def build(model):
    """Pick the tight step for a model; the three families that have one."""
    if hasattr(model, "lstm_ti"):
        return TightSeqV2Step(model).eval()
    if hasattr(model, "enc_ti"):
        return TightXfmrStep(model).eval()
    if hasattr(model, "tcn_ti"):
        return TightTCNStep(model).eval()
    raise SystemExit(f"no tight step for {type(model).__name__}")
