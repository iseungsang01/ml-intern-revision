"""Diagonal state-space model — the fourth family, and the only real competitor on price.

§8ag closed with "a state-space model (S4/Mamba) would test whether the tie extends to an
operator with O(1) state *and* long reach", and §8aj then made that question sharp rather
than fashionable. The batch's cost finding is that latency is dispatched operator count,
and on that axis the three families measured so far sit at 111 (recurrent) / 209–305
(dilated conv) / 473–565 (attention) per online step. Recurrence wins on price because its
step is O(1) in reach. A diagonal SSM is the one operator that is **O(1) in reach like the
LSTM and parallel in training like the convolution**, so it is the only arm that can beat
the backbone at its own argument.

**The same operator, twice.** Training runs the closed-form convolution: with a diagonal
`A`, `h_t = A h_{t-1} + x_t` unrolls to `y_t = Σ_l K[l] x_{t-l}` where `K[l] = Re(Σ_n C_n
A_n^l)`, so the whole sequence is one FFT convolution instead of L sequential steps.
Inference runs the recurrence itself, which is an element-wise multiply-add. Those are the
same function — `tests/test_architecture.py::test_seq_family_streaming_equals_batch`
requires them to agree to float precision, which is what licenses pricing the recurrence
and scoring the convolution.

**Stability is structural, not learned.** `A = exp(-exp(log_decay)) · exp(i·theta)` has
modulus in (0, 1) for every real `log_decay`, so the state cannot blow up at any point in
training and the kernel is always summable. `B` is fixed to 1 and `C` is complex-learned —
the S4D parameterisation, which is exactly as expressive and has fewer knobs to confound.

**Routing is seq_v2's, unchanged.** The `V_rot` branch is a separate stack over the
non-fast tail only, so this arm differs from `seq_v2` in the sequence operator and nothing
else (§8ab). Reach is not declared: like the LSTM the state carries the whole past, so a
rung of the ladder is made by truncating context at train and eval time
(`CES_SEQ_TRAIN_CTX` / `CES_SEQ_EVAL_CTX`), not by a receptive-field bound.
"""

import torch
import torch.nn as nn

from seq_data import N_FEATURES, N_FAST_CHANNELS


class _DiagSSM(nn.Module):
    """One diagonal SSM: `h_t = A h_{t-1} + x_t`, `y_t = Re(C h_t) + D x_t`, per channel."""

    def __init__(self, channels, state=16):
        super().__init__()
        self.channels, self.state = int(channels), int(state)
        # |A| = exp(-exp(log_decay)) in (0, 1) for any real log_decay -- stable by
        # construction, so nothing has to be clamped during training.
        self.log_decay = nn.Parameter(torch.rand(channels, state) * 2.0 - 3.0)
        self.theta = nn.Parameter(torch.rand(channels, state) * 3.14159)
        self.c_re = nn.Parameter(torch.randn(channels, state) / state ** 0.5)
        self.c_im = nn.Parameter(torch.randn(channels, state) / state ** 0.5)
        self.d = nn.Parameter(torch.ones(channels))

    def _log_a(self):
        return torch.complex(-torch.exp(self.log_decay), self.theta)   # (H, N)

    def kernel(self, length):
        """`K[l] = Re(Σ_n C_n A_n^l)` for l = 0 … length-1 -> (H, L).

        Computed as `exp(l · log A)` rather than by repeated multiplication: one pass, no
        error accumulation down the length, and it stays differentiable.
        """
        l = torch.arange(length, device=self.log_decay.device, dtype=self.log_decay.dtype)
        powers = torch.exp(self._log_a().unsqueeze(-1) * l)            # (H, N, L)
        c = torch.complex(self.c_re, self.c_im).unsqueeze(-1)          # (H, N, 1)
        return (c * powers).sum(dim=1).real                            # (H, L)

    DIRECT_MAX = 128        # above this the Toeplitz matrix costs more than an FFT

    def forward(self, x):
        """x (B, L, H) -> (B, L, H). Causal by construction: y_t reads x_{<=t} only.

        Two paths for one function, chosen by length.

        **Short (L <= DIRECT_MAX): a lower-triangular Toeplitz product.** `T[h,t,s] =
        K[h, t-s]` for `s <= t` and zero above the diagonal, so `y = T x` reads the future
        *never* -- causality is a property of the matrix, not an argument about round-off.
        Every rung of the reach ladder trains at `L = reach <= 63`, so this is the path the
        ladder actually uses, and it is ~3x faster there than the transform.

        **Long: an FFT convolution in float64.** A spectral convolution is causal in
        structure -- the kernel has no support at negative lag -- but it is a global
        operation, so in float32 a change at `t > s` moves the output at `s` by round-off
        (measured 2.6e-7 against outputs of order 0.1). `eval_seq` scores this model from
        windows that contain rows after the scored row, so that margin is worth buying:
        float64 drops the same leak to 5e-16, below float32's own resolution.
        """
        length = x.shape[1]
        k = self.kernel(length)                                        # (H, L)
        if length <= self.DIRECT_MAX:
            idx = (torch.arange(length, device=x.device).unsqueeze(1)
                   - torch.arange(length, device=x.device).unsqueeze(0))   # t - s
            toe = torch.where(idx >= 0, k[:, idx.clamp(min=0)],
                              torch.zeros((), device=x.device, dtype=k.dtype))
            y = torch.einsum("hts,bsh->bth", toe, x)
        else:
            n = 2 * length                                             # no circular wrap
            xd = x.transpose(1, 2).contiguous().double()               # (B, H, L)
            xf = torch.fft.rfft(xd, n=n)
            kf = torch.fft.rfft(k.double(), n=n)
            y = torch.fft.irfft(xf * kf, n=n)[..., :length].to(x.dtype).transpose(1, 2)
        return y + x * self.d

    def stream_init(self, device, dtype):
        return torch.zeros(1, self.channels, self.state, device=device,
                           dtype=torch.complex64 if dtype == torch.float32 else torch.complex128)

    def stream_step(self, state, x_t):
        """(1, 1, H) + state -> (1, 1, H). The recurrence the kernel came from."""
        a = torch.exp(self._log_a())                                   # (H, N)
        state.mul_(a).add_(x_t[:, 0].unsqueeze(-1))                    # h = A h + x
        c = torch.complex(self.c_re, self.c_im)
        y = (state * c).sum(dim=-1).real                               # (1, H)
        return (y + x_t[:, 0] * self.d).unsqueeze(1)


class _SSMStack(nn.Module):
    """`layers` residual blocks: diagonal SSM -> GELU -> position-wise mix -> LayerNorm.

    The position-wise linear is not decoration: a diagonal SSM never mixes channels, so
    without it the stack would be `channels` independent scalar filters. This is the same
    block shape the TCN arm uses, with the convolution swapped for the SSM -- which is what
    keeps "only the operator differs" true.
    """

    def __init__(self, n_in, hidden, layers, state=16, dropout=0.1):
        super().__init__()
        self.proj = nn.Linear(n_in, hidden)
        self.ssms = nn.ModuleList(_DiagSSM(hidden, state) for _ in range(layers))
        self.mixes = nn.ModuleList(nn.Linear(hidden, hidden) for _ in range(layers))
        self.norms = nn.ModuleList(nn.LayerNorm(hidden) for _ in range(layers))
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        h = self.proj(x)
        for ssm, mix, norm in zip(self.ssms, self.mixes, self.norms):
            z = self.drop(mix(nn.functional.gelu(ssm(h))))
            h = norm(h + z)
        return h

    def stream_init(self, device, dtype):
        return [s.stream_init(device, dtype) for s in self.ssms]

    def stream_step(self, state, x_t):
        h = self.proj(x_t)
        for i, (ssm, mix, norm) in enumerate(zip(self.ssms, self.mixes, self.norms)):
            z = mix(nn.functional.gelu(ssm.stream_step(state[i], h)))
            h = norm(h + z)
        return h


class SeqCESSSM(nn.Module):
    # `layers=2` is seq_v2's own T_i depth. Depth is not free in dispatched operators
    # (3 layers cost 401 per step against 2 layers' 293), so matching the backbone's depth
    # is what keeps the price comparison about the operator rather than about the stack.
    def __init__(self, n_in=N_FEATURES, n_fast=N_FAST_CHANNELS, layers=2,
                 hidden_ti=208, hidden_vt=64, state=24, head=64, dropout=0.1):
        super().__init__()
        self.n_fast = int(n_fast)
        self.n_slow = int(n_in) - self.n_fast
        if self.n_slow <= 0:
            raise ValueError(f"n_fast={n_fast} leaves no non-fast channels of {n_in}")

        self.ssm_ti = _SSMStack(n_in, hidden_ti, layers, state, dropout)
        self.head_ti = nn.Sequential(nn.Linear(hidden_ti, head), nn.GELU(), nn.Linear(head, 1))
        self.ssm_vt = _SSMStack(self.n_slow, hidden_vt, layers, state, dropout)
        self.head_vt = nn.Sequential(nn.Linear(hidden_vt, head), nn.GELU(), nn.Linear(head, 1))

        # No `receptive_field`: the state carries the whole past, exactly like seq_v2's
        # LSTM. A ladder rung is a truncated context, not a bounded operator.
        #
        # `spectral` tells the causality test that the batch path is an FFT convolution:
        # causal by construction, but global in floating point, so it is checked against a
        # numerical floor instead of bit equality (see `forward` above).
        self.spectral = True
        n_params = sum(p.numel() for p in self.parameters())
        if n_params >= 1_000_000:
            raise ValueError(f"parameter budget exceeded: {n_params:,} >= 1,000,000")
        self.n_params = n_params

    def forward(self, x, lengths=None):
        """x (B, L, n_in) -> (B, L, 2) normalized [CES_TI, CES_VT]. `lengths` unused."""
        h_ti = self.ssm_ti(x)
        h_vt = self.ssm_vt(x[..., self.n_fast:])
        return torch.cat([self.head_ti(h_ti), self.head_vt(h_vt)], dim=-1)

    def stream_init(self, device=None, dtype=torch.float32):
        device = device or next(self.parameters()).device
        return {"ti": self.ssm_ti.stream_init(device, dtype),
                "vt": self.ssm_vt.stream_init(device, dtype)}

    def stream_step(self, state, x_t):
        """One online step: (1, 1, n_in) -> (1, 1, 2). Equals `forward`'s row t."""
        h_ti = self.ssm_ti.stream_step(state["ti"], x_t)
        h_vt = self.ssm_vt.stream_step(state["vt"], x_t[..., self.n_fast:])
        return torch.cat([self.head_ti(h_ti), self.head_vt(h_vt)], dim=-1)
