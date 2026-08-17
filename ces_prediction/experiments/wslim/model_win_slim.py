"""W-SLIM: the window model with its structure sized to the window it actually gets.

Motivation (승상님, 2026-08-17). `iter009` was found by architecture search at a time when
the window was large, and its shape never followed the window down. At the confirmed
protocol's `W = 2` the published model spends its structure on things a length-2 sequence
cannot use:

  * two stacked `Conv1d(kernel_size=3, padding=1)` per sensor -- receptive field 5 over a
    sequence of 2, so a third of every kernel only ever reads zero padding;
  * `AdaptiveAvgPool1d(1)` after them, which averages the 2 timesteps into 1 and *discards
    the ordering* that the window existed to provide;
  * a bidirectional GRU plus a 4-head additive-attention pool over those same 2 steps.

None of it is wrong at `W = 8`; all of it is overhead at `W = 2`. Note what this does NOT
change: `W = 2` and `W = 4` instantiate the *same* 201,258 `iter009` weights, because the
window sets sequence length rather than kernel shapes -- shrinking the parameter count
requires re-deriving the structure from the input, which is what this file does.

Design: the window is small enough to flatten, so each stream is
`flatten -> Linear -> GELU` and the routing of `iter009` is kept exactly (`T_i` sees the
fast diagnostics plus history plus time; `V_rot` sees history plus time ONLY -- §8ab
re-verified that block bit-identically, and it is a physics claim, not a tuning detail).
The observation flags stay in the history channels where the flatten exposes them
directly; at `W = 2` the target timestep is fully masked by the data contract, so the
masked-attention pool it replaces has at most one observed step to weigh.

~25.6k parameters and 21 leaf ops vs 201k and 57 -- and §8aa (skill flat 34k…879k) plus
§8z (a 21k latent model matches the backbone on `T_i`) say this size is not the binding
constraint. Whether the skill survives is the experiment; this file only makes it askable.
"""

import torch
import torch.nn as nn


class MultimodalCESPredictor(nn.Module):
    def __init__(
        self,
        window_size=2,
        bes_channels=9,
        ecei_channels=4,
        mc_channels=2,
        time_channels=4,
        ces_history_channels=4,
        sensor_feature_dim=48,
        history_feature_dim=32,
        time_feature_dim=16,
        hidden=96,
    ):
        super().__init__()
        self.window_size = int(window_size)
        self.time_channels = time_channels
        self.ces_history_channels = ces_history_channels
        W = self.window_size

        def stream(in_channels, out_dim):
            return nn.Sequential(nn.Flatten(1), nn.Linear(in_channels * W, out_dim), nn.GELU())

        ctx = time_channels + ces_history_channels
        self.bes_extractor = stream(bes_channels + ctx, sensor_feature_dim)
        self.ecei_extractor = stream(ecei_channels + ctx, sensor_feature_dim)
        self.mc_extractor = stream(mc_channels + ctx, sensor_feature_dim)
        self.time_extractor = stream(time_channels, time_feature_dim)
        self.history_extractor = stream(ces_history_channels + time_channels,
                                        history_feature_dim)

        ti_in = sensor_feature_dim * 3 + time_feature_dim + history_feature_dim
        self.ti_head = nn.Sequential(
            nn.Linear(ti_in, hidden), nn.GELU(), nn.Linear(hidden, 1)
        )
        # V_rot: history + time only. Identical restriction to iter009 / seq_v2.
        vt_in = history_feature_dim + time_feature_dim
        self.vt_head = nn.Sequential(
            nn.Linear(vt_in, hidden // 2), nn.GELU(), nn.Linear(hidden // 2, 1)
        )

    @classmethod
    def from_dataset(cls, dataset, **kwargs):
        dims = dataset.feature_dims
        return cls(
            bes_channels=dims["bes"],
            ecei_channels=dims["ecei"],
            mc_channels=dims["mc"],
            time_channels=dims["time"],
            ces_history_channels=dims.get("ces_history", 4),
            **kwargs,
        )

    def _fill(self, tensor, reference, channels):
        if tensor is None:
            batch, steps = reference.shape[:2]
            return reference.new_zeros(batch, steps, channels)
        return tensor

    def forward(self, bes, ecei, mc, time_features=None, ces_history=None):
        time_features = self._fill(time_features, bes, self.time_channels)
        ces_history = self._fill(ces_history, bes, self.ces_history_channels)
        if ces_history.shape[-1] != self.ces_history_channels:
            raise ValueError(
                f"Expected {self.ces_history_channels} CES history channels, "
                f"got {ces_history.shape[-1]}"
            )

        ctx = torch.cat((time_features, ces_history), dim=-1)
        hist = self.history_extractor(torch.cat((ces_history, time_features), dim=-1))
        tim = self.time_extractor(time_features)

        ti = torch.cat(
            (
                self.bes_extractor(torch.cat((bes, ctx), dim=-1)),
                self.ecei_extractor(torch.cat((ecei, ctx), dim=-1)),
                self.mc_extractor(torch.cat((mc, ctx), dim=-1)),
                tim,
                hist,
            ),
            dim=-1,
        )
        vt = torch.cat((hist, tim), dim=-1)
        return torch.cat((self.ti_head(ti), self.vt_head(vt)), dim=-1)
