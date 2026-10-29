"""seq v2 with a predictive-DISTRIBUTION head (B.17, PREREGISTRATION_B17.md).

Everything upstream of the heads is seq_v2 byte-for-byte: two causal LSTMs, the V_rot
branch blind to the 15 fast channels (the observation model of sec. 8t), the same widths.
The ONE controlled variable against the backbone is the output: instead of a point
`y_hat`, each target's head emits the parameters of a conditional distribution
p(y_t | h_t) and the model is trained by its negative log-likelihood.

  dist = "gauss":  (mu, sigma)                          heteroscedastic Gaussian
  dist = "mix2" :  (pi_1, mu_1, mu_2, sigma_1, sigma_2)  two-component Gaussian mixture
  dist = "t"    :  (mu, sigma, nu)                      Student-t, nu > 2 learnable (B.17b)

Why these two and nothing else (sec. 8bb): the event physics says the one-step
conditional of T_i at a pedestal event is TWO-valued (crash / no crash) while the quiet
bulk is one noisy OU value. A Gaussian head can only widen; a 2-mixture can split. The
pair isolates "does the model know its own error scale" (gauss) from "does it know the
error is two-valued" (mix2 - gauss).

`forward(x, lengths)` returns the posterior MEAN, shape (B, L, 2), so eval_seq.py, the
paired comparison, bootstrap ladder and rescore all score the arm with zero changes;
`forward(x, lengths, return_dist=True)` returns the raw parameter tensor (B, L, 2, P)
that train_seq's NLL path and eval_dist_b17.py consume. `mean`, `nll`, `moments` are the
only places the parametrisation is decoded.
"""

import math

import torch
import torch.nn as nn

from model_seq_v2 import SeqCESLSTMv2

LOG_2PI = math.log(2.0 * math.pi)
SIGMA_FLOOR = 1e-3
# softplus(b) = 1 -> the predictive sd starts at one normalized unit (the target's own
# train std), i.e. maximally uninformative, and shrinks only where the likelihood pays.
SIGMA_RAW_INIT = math.log(math.e - 1.0)
NU_FLOOR = 2.05                      # nu > 2 keeps the predictive variance finite
NU_RAW_INIT = math.log(math.exp(5.0 - NU_FLOOR) - 1.0)   # softplus(NU_RAW_INIT) + NU_FLOOR = 5


class SeqCESLSTMv2Dist(SeqCESLSTMv2):
    def __init__(self, dist="gauss", **kw):
        super().__init__(**kw)
        if dist not in ("gauss", "mix2", "t"):
            raise ValueError(f"dist must be gauss|mix2|t, got {dist!r}")
        self.dist = dist
        self.n_comp = 2 if dist == "mix2" else 1
        self.n_param = {"gauss": 2, "mix2": 5, "t": 3}[dist]
        # Replace the scalar output layer of each head by a P-way one; the hidden layer
        # (Linear -> GELU) is the backbone's.
        for name in ("head_ti", "head_vt"):
            head = getattr(self, name)
            last = nn.Linear(head[-1].in_features, self.n_param)
            with torch.no_grad():
                # sigma raw channels start at sd = 1; mixture logit starts at pi = 1/2
                if dist == "gauss":
                    last.bias[1] = SIGMA_RAW_INIT
                elif dist == "t":
                    last.bias[1] = SIGMA_RAW_INIT
                    last.bias[2] = NU_RAW_INIT   # nu starts at ~5 (finite variance, clearly heavy-tailed)
                else:
                    last.bias[0] = 0.0
                    last.bias[3] = SIGMA_RAW_INIT
                    last.bias[4] = SIGMA_RAW_INIT
            head[-1] = last
        n_params = sum(p.numel() for p in self.parameters())
        if n_params >= 1_000_000:
            raise ValueError(f"parameter budget exceeded: {n_params:,} >= 1,000,000")
        self.n_params = n_params

    # ---- parametrisation ---------------------------------------------------------
    def decode(self, params):
        """(..., P) -> (pi (..., K), mu (..., K), sigma (..., K)) in normalized units."""
        if self.dist in ("gauss", "t"):
            mu = params[..., 0:1]
            sigma = nn.functional.softplus(params[..., 1:2]) + SIGMA_FLOOR
            pi = torch.ones_like(mu)
        else:
            p1 = torch.sigmoid(params[..., 0:1])
            pi = torch.cat([p1, 1.0 - p1], dim=-1)
            mu = params[..., 1:3]
            sigma = nn.functional.softplus(params[..., 3:5]) + SIGMA_FLOOR
        return pi, mu, sigma

    def nu(self, params):
        """Student-t degrees of freedom (..., 1); only defined for dist == "t"."""
        return nn.functional.softplus(params[..., 2:3]) + NU_FLOOR

    def mean(self, params):
        pi, mu, _ = self.decode(params)
        return (pi * mu).sum(dim=-1)

    def moments(self, params):
        """Mean and sd of the predictive distribution, shape params[..., :-1]."""
        pi, mu, sigma = self.decode(params)
        m = (pi * mu).sum(dim=-1, keepdim=True)
        var = (pi * (sigma ** 2 + (mu - m) ** 2)).sum(dim=-1)
        if self.dist == "t":
            nu = self.nu(params).squeeze(-1)
            var = var * nu / (nu - 2.0)
        return m.squeeze(-1), torch.sqrt(var)

    def nll(self, params, y):
        """Per-element negative log-likelihood of y under the predictive distribution."""
        pi, mu, sigma = self.decode(params)
        z = (y.unsqueeze(-1) - mu) / sigma
        if self.dist == "t":
            nu = self.nu(params)
            logp = (torch.lgamma((nu + 1) / 2) - torch.lgamma(nu / 2) - 0.5 * torch.log(nu * math.pi)
                    - torch.log(sigma) - (nu + 1) / 2 * torch.log1p(z ** 2 / nu))
            return -logp.squeeze(-1)
        logp = torch.log(pi) - 0.5 * z ** 2 - torch.log(sigma) - 0.5 * LOG_2PI
        return -torch.logsumexp(logp, dim=-1)

    # ---- forward -------------------------------------------------------------------
    def forward(self, x, lengths=None, return_dist=False):
        """x (B, L, n_in) -> mean (B, L, 2), or the parameters (B, L, 2, P)."""
        vt_end = self.n_base + self.n_extra_vt
        x_ti = x[..., :self.n_base]
        if self.n_extra_ti:
            x_ti = torch.cat([x_ti, x[..., vt_end:vt_end + self.n_extra_ti]], dim=-1)
        h_ti = self._run(self.lstm_ti, self.norm_ti, x_ti, lengths)
        h_vt = self._run(self.lstm_vt, self.norm_vt, x[..., self.n_fast:vt_end], lengths)
        params = torch.stack([self.head_ti(h_ti), self.head_vt(h_vt)], dim=-2)  # (B, L, 2, P)
        if return_dist:
            return params
        return self.mean(params)
