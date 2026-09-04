"""seq b3vq (B.11): b3 with the readout selected by a discrete code instead of shared.

PREREGISTRATION_B11.md fixes the hypothesis: 8an found that the one covariate predicting a
win is how much the target moves within a discharge, and the model loses on quiet
discharges (T_i 42%, V_rot 34%) because the causal GP is already near-optimal there while
a continuous latent has no way to stand down. A codebook turns that into an explicit,
learnable switch.

The ONE controlled variable against `b3k8` is the readout:

    b3     y = carried + <w, z> + b                    (one linear map, always)
    b3vq   y = carried + <w_c(t), z> + b_c(t)          (a linear map per code)
           c(t) = argmin_c ||z_t - E_c||

Everything else -- encoder, anchor, routing, latent width, loss, optimizer, data treatment
-- is identical, and every `w_c`, `b_c` is zero-initialized so training still starts exactly
at carry-forward persistence.

Three commitments follow the pre-registration:

  * **The output is never quantized.** 8aq measured the model at 2.3-3.4x the target's own
    scatter in the bulk, so quantizing the prediction would simply discard that margin. The
    code selects *which* linear map is applied; the map is applied to the continuous `z`.
  * **The decomposition stays exact.** Within a step, prediction = anchor + sum_k
    contrib[k] + bias with contrib[k] = w_{c,k} * z_k, so `decompose()` is still an identity
    rather than an attribution.
  * **Routing is structural, as always.** The V_rot branch and its codebook see only the
    non-fast tail, so perturbing the fast channels leaves V_rot bit-identical.

The codebook is EMA-updated (van den Oord et al. 2017, appendix) rather than trained by
gradient, with dead-code restarts, because 452 training blocks is a small population for a
codebook and collapse is the expected failure mode -- S1 of the pre-registration exists to
catch exactly that. The commitment term is exposed as `self.aux_loss` for the runner to add;
`forward` keeps returning a plain tensor so the rest of the seq harness is unaffected.
"""

import torch
import torch.nn as nn

from seq_data import N_FEATURES, N_FAST_CHANNELS


class VQGate(nn.Module):
    """Nearest-code lookup over a K-dim latent, with an EMA codebook.

    Holds no gradient of its own: the codebook moves by EMA and the encoder is pulled
    toward it by the commitment term the caller adds to the loss.
    """

    def __init__(self, n_codes, dim, decay=0.99, eps=1e-5, dead_after=0.01):
        super().__init__()
        self.n_codes = int(n_codes)
        self.dim = int(dim)
        self.decay = float(decay)
        self.eps = float(eps)
        self.dead_after = float(dead_after)
        # Latents are tanh-bounded, so initialize the codebook inside the same cube.
        self.register_buffer("codebook", torch.empty(self.n_codes, self.dim).uniform_(-0.5, 0.5))
        self.register_buffer("cluster_size", torch.zeros(self.n_codes))
        self.register_buffer("ema_sum", self.codebook.clone())

    def forward(self, z, valid):
        """z (B, L, K) in [-1, 1]; valid (B, L) bool -> code indices (B, L) and commitment.

        Padded positions are excluded from both the assignment statistics and the EMA, so a
        short block cannot vote with its padding.
        """
        flat = z.reshape(-1, self.dim)
        vfloat = valid.reshape(-1)
        vflat = vfloat.bool()
        d = (flat.pow(2).sum(1, keepdim=True)
             - 2 * flat @ self.codebook.t()
             + self.codebook.pow(2).sum(1))
        idx = d.argmin(1)

        if self.training and vflat.any():
            with torch.no_grad():
                sel = flat[vflat]
                sidx = idx[vflat]
                onehot = torch.zeros(sel.shape[0], self.n_codes, device=z.device)
                onehot.scatter_(1, sidx.unsqueeze(1), 1.0)
                counts = onehot.sum(0)
                sums = onehot.t() @ sel
                self.cluster_size.mul_(self.decay).add_(counts, alpha=1 - self.decay)
                self.ema_sum.mul_(self.decay).add_(sums, alpha=1 - self.decay)
                n = self.cluster_size.sum()
                smoothed = ((self.cluster_size + self.eps)
                            / (n + self.n_codes * self.eps) * n)
                self.codebook.copy_(self.ema_sum / smoothed.unsqueeze(1))
                # Dead-code restart: a code nobody uses is re-seeded onto a live latent,
                # which is the standard remedy and keeps S1 from failing for want of one.
                dead = self.cluster_size < self.dead_after * max(float(n), 1.0) / self.n_codes
                if dead.any() and sel.shape[0] > 0:
                    pick = torch.randint(0, sel.shape[0], (int(dead.sum()),), device=z.device)
                    self.codebook[dead] = sel[pick]
                    self.cluster_size[dead] = 1.0
                    self.ema_sum[dead] = sel[pick]

        e = self.codebook[idx].reshape(z.shape)
        # Commitment only: the codebook is EMA-updated, so no codebook loss term.
        commit = ((z - e.detach()).pow(2).mean(-1) * valid).sum() / valid.sum().clamp(min=1)
        return idx.reshape(z.shape[:2]), commit

    def usage(self):
        p = self.cluster_size / self.cluster_size.sum().clamp(min=1e-9)
        live = int((p > 0.01).sum())
        perplexity = float(torch.exp(-(p * (p + 1e-12).log()).sum()))
        return {"live_codes": live, "perplexity": perplexity,
                "usage": [float(v) for v in p]}


class SeqCESB3VQ(nn.Module):
    """b3 with a per-code linear readout. See the module docstring for the contract."""

    def __init__(self, n_in=N_FEATURES, n_fast=N_FAST_CHANNELS,
                 hidden_ti=64, hidden_vt=32, latent_ti=8, latent_vt=4,
                 n_codes=8, beta=0.25):
        super().__init__()
        self.n_fast = int(n_fast)
        self.n_slow = int(n_in) - self.n_fast
        if self.n_slow <= 0:
            raise ValueError(f"n_fast={n_fast} leaves no non-fast channels of {n_in}")
        self.latent_ti = int(latent_ti)
        self.latent_vt = int(latent_vt)
        self.n_codes = int(n_codes)
        self.beta = float(beta)
        self.anchor_col_ti = self.n_fast + 1
        self.anchor_col_vt = self.n_fast + 4

        self.gru_ti = nn.GRU(n_in, hidden_ti, batch_first=True)
        self.to_z_ti = nn.Linear(hidden_ti, self.latent_ti)
        self.gru_vt = nn.GRU(self.n_slow, hidden_vt, batch_first=True)
        self.to_z_vt = nn.Linear(hidden_vt, self.latent_vt)

        self.vq_ti = VQGate(self.n_codes, self.latent_ti)
        self.vq_vt = VQGate(self.n_codes, self.latent_vt)

        # Per-code readouts, zero-initialized so training starts exactly at persistence.
        self.w_ti = nn.Parameter(torch.zeros(self.n_codes, self.latent_ti))
        self.b_ti = nn.Parameter(torch.zeros(self.n_codes))
        self.w_vt = nn.Parameter(torch.zeros(self.n_codes, self.latent_vt))
        self.b_vt = nn.Parameter(torch.zeros(self.n_codes))

        self.aux_loss = torch.zeros(())
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

    def _valid(self, x, lengths):
        if lengths is None:
            return torch.ones(x.shape[:2], device=x.device)
        ar = torch.arange(x.shape[1], device=x.device).unsqueeze(0)
        return (ar < lengths.to(x.device).unsqueeze(1)).float()

    def latents(self, x, lengths=None):
        z_ti = torch.tanh(self.to_z_ti(self._run(self.gru_ti, x, lengths)))
        z_vt = torch.tanh(self.to_z_vt(self._run(self.gru_vt, x[..., self.n_fast:], lengths)))
        return z_ti, z_vt

    def codes(self, x, lengths=None):
        """Code indices per step, for the pre-registration's S3 mutual-information test."""
        z_ti, z_vt = self.latents(x, lengths)
        valid = self._valid(x, lengths)
        was = self.training
        self.eval()
        i_ti, _ = self.vq_ti(z_ti, valid)
        i_vt, _ = self.vq_vt(z_vt, valid)
        self.train(was)
        return i_ti, i_vt

    def forward(self, x, lengths=None):
        z_ti, z_vt = self.latents(x, lengths)
        valid = self._valid(x, lengths)
        i_ti, c_ti = self.vq_ti(z_ti, valid)
        i_vt, c_vt = self.vq_vt(z_vt, valid)
        self.aux_loss = self.beta * (c_ti + c_vt)

        y_ti = x[..., self.anchor_col_ti] + (z_ti * self.w_ti[i_ti]).sum(-1) + self.b_ti[i_ti]
        y_vt = x[..., self.anchor_col_vt] + (z_vt * self.w_vt[i_vt]).sum(-1) + self.b_vt[i_vt]
        return torch.stack([y_ti, y_vt], dim=-1)

    def decompose(self, x, lengths=None):
        """EXACT named terms, per step: prediction = anchor + sum_k contrib[k] + bias."""
        z_ti, z_vt = self.latents(x, lengths)
        valid = self._valid(x, lengths)
        was = self.training
        self.eval()
        i_ti, _ = self.vq_ti(z_ti, valid)
        i_vt, _ = self.vq_vt(z_vt, valid)
        self.train(was)
        out = {}
        for tag, z, idx, w, b, col in (
                ("ti", z_ti, i_ti, self.w_ti, self.b_ti, self.anchor_col_ti),
                ("vt", z_vt, i_vt, self.w_vt, self.b_vt, self.anchor_col_vt)):
            contrib = z * w[idx]
            out[f"anchor_{tag}"] = x[..., col]
            out[f"z_{tag}"] = z
            out[f"code_{tag}"] = idx
            out[f"contrib_{tag}"] = contrib
            out[f"bias_{tag}"] = b[idx]
            out[f"prediction_{tag}"] = x[..., col] + contrib.sum(-1) + b[idx]
        return out
