"""seq b3 (B.3): the minimal interpretable latent-bottleneck model on the seq_v2 structure.

PREREGISTRATION_W2.md section 6 (B.3) asks for a minimal interpretable model of the ADOPTED
backbone (seq_v2, THESIS_RESULTS.md section 8x), with a 4-8 dim latent, per-term decomposition,
and the V_rot routing preserved as a structural property. Section 8k's lesson governs the form:
the anchor+delta experiment showed the recoverable structure is substantially NON-linear, so
"more named linear terms" cannot close the gap -- interpretability has to come from somewhere
else. Here it comes from three structural commitments instead of from linearity of the encoder:

  1. **Anchor + correction.** The prediction is `carried_value + correction`, where the anchor
     term is parameter-free carry-forward persistence (the carried feature the data layer
     already provides, strictly causal). The readout weights are ZERO-INITIALIZED, so training
     starts exactly at persistence and everything learned is, by construction, the correction.
  2. **Latent bottleneck.** ALL learned information available to a target's readout passes
     through a K-dim tanh-bounded latent z (K in [4, 8], chosen on val). The encoder may be
     nonlinear (a small causal GRU -- section 8k says it must be), but its entire output is
     K visible numbers per step, each individually probeable against physical quantities.
  3. **Linear readout.** prediction = anchor + sum_k w_k * z_k + b. The decomposition into
     named terms is therefore EXACT, not an attribution approximation: term k contributes
     w_k * z_k, and `decompose()` returns every term.

Routing is seq_v2's, at the encoder as always: the V_rot branch sees only the non-fast tail
(dt + per-target carried/staleness/has), so fast-channel perturbations leave V_rot output
bit-identical (verified structurally by probe_b3.py).

Parameter budget: ~21k (6% of seq_v2's 358k), against the project hard cap of 1M.
"""

import torch
import torch.nn as nn

from seq_data import N_FEATURES, N_FAST_CHANNELS


class SeqCESB3(nn.Module):
    """Anchor + latent-bottleneck correction, per target, with seq_v2 routing.

    Feature layout (seq_data): [fast(15) | dt | TI(carried, stale, has) | VT(carried,
    stale, has)] -- carried values are on the GLOBAL target z-scale, i.e. exactly the
    model's output space, which is what makes the additive anchor term well-posed.
    """

    def __init__(self, n_in=N_FEATURES, n_fast=N_FAST_CHANNELS,
                 hidden_ti=64, hidden_vt=32, latent_ti=6, latent_vt=4):
        super().__init__()
        self.n_fast = int(n_fast)
        self.n_slow = int(n_in) - self.n_fast
        if self.n_slow <= 0:
            raise ValueError(f"n_fast={n_fast} leaves no non-fast channels of {n_in}")
        self.latent_ti = int(latent_ti)
        self.latent_vt = int(latent_vt)
        # anchor feature columns: carried_TI / carried_VT
        self.anchor_col_ti = self.n_fast + 1
        self.anchor_col_vt = self.n_fast + 4

        self.gru_ti = nn.GRU(n_in, hidden_ti, batch_first=True)
        self.to_z_ti = nn.Linear(hidden_ti, self.latent_ti)
        self.w_ti = nn.Linear(self.latent_ti, 1)
        self.gru_vt = nn.GRU(self.n_slow, hidden_vt, batch_first=True)
        self.to_z_vt = nn.Linear(hidden_vt, self.latent_vt)
        self.w_vt = nn.Linear(self.latent_vt, 1)
        for lin in (self.w_ti, self.w_vt):  # start exactly at carry-forward persistence
            nn.init.zeros_(lin.weight)
            nn.init.zeros_(lin.bias)

        n_params = sum(p.numel() for p in self.parameters())
        if n_params >= 1_000_000:
            raise ValueError(f"parameter budget exceeded: {n_params:,} >= 1,000,000")
        self.n_params = n_params

    @staticmethod
    def _run(gru, x, lengths):
        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False)
            out, _ = gru(packed)
            out, _ = nn.utils.rnn.pad_packed_sequence(
                out, batch_first=True, total_length=x.shape[1])
        else:
            out, _ = gru(x)
        return out

    def latents(self, x, lengths=None):
        """(B, L, n_in) -> tanh-bounded latents z_ti (B, L, K_ti), z_vt (B, L, K_vt)."""
        z_ti = torch.tanh(self.to_z_ti(self._run(self.gru_ti, x, lengths)))
        z_vt = torch.tanh(self.to_z_vt(self._run(self.gru_vt, x[..., self.n_fast:], lengths)))
        return z_ti, z_vt

    def forward(self, x, lengths=None):
        z_ti, z_vt = self.latents(x, lengths)
        y_ti = x[..., self.anchor_col_ti] + self.w_ti(z_ti).squeeze(-1)
        y_vt = x[..., self.anchor_col_vt] + self.w_vt(z_vt).squeeze(-1)
        return torch.stack([y_ti, y_vt], dim=-1)

    def decompose(self, x, lengths=None):
        """EXACT named additive terms: prediction = anchor + sum_k contrib[..., k] + bias."""
        z_ti, z_vt = self.latents(x, lengths)
        out = {}
        for tag, z, w, col in (("ti", z_ti, self.w_ti, self.anchor_col_ti),
                               ("vt", z_vt, self.w_vt, self.anchor_col_vt)):
            contrib = z * w.weight[0]              # (B, L, K): per-latent-dim term
            out[f"anchor_{tag}"] = x[..., col]
            out[f"z_{tag}"] = z
            out[f"contrib_{tag}"] = contrib
            out[f"bias_{tag}"] = w.bias.detach()
            out[f"prediction_{tag}"] = x[..., col] + contrib.sum(-1) + w.bias
        return out
