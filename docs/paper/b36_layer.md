# B.36 — a layer on top of the causal variant (registered: PREREGISTRATION_B36.md)

CPU, no network trained. Frozen B.30 row tables, T_i, cut population, natural cadence. A gradient-boosted model of the signed residual on the inputs known at the row, fitted on one parity fold of the test discharges and applied to the other. Skill = 1 - MSE ratio. Intervals: 95 %, 1,000 resamples of discharges.

- **L1** (layer A, 15 inputs, over the causal variant): 0 of 4 splits -> does not hold.
- **L2** (layer B, with the history-only twin added, over the causal variant): 0 of 4 splits -> does not hold.

| split | rows | L1: layer A over causal | L2: layer B over causal | gated layer over causal | linear stack over causal | layer A over the stack | layer B over the stack | median error, layer A over causal | correlation of predicted and actual residual in the top tenth |
|---|---|---|---|---|---|---|---|---|---|
| 42 | 32,127 | +0.010 [-0.011, +0.030] | +0.005 [-0.018, +0.027] | +0.006 [-0.012, +0.024] | +0.036 [+0.017, +0.056] | -0.027 [-0.053, -0.001] | -0.032 [-0.061, -0.003] | 0.980 [0.931, 1.033] | +0.171 [+0.115, +0.232] |
| 1 | 35,138 | +0.007 [-0.026, +0.038] | +0.015 [-0.016, +0.046] | -0.004 [-0.029, +0.018] | +0.028 [+0.015, +0.044] | -0.022 [-0.059, +0.012] | -0.014 [-0.049, +0.020] | 0.893 [0.833, 0.950] | +0.222 [+0.177, +0.275] |
| 7 | 35,357 | +0.017 [-0.011, +0.047] | +0.019 [-0.011, +0.050] | +0.017 [-0.005, +0.043] | +0.021 [+0.012, +0.031] | -0.003 [-0.031, +0.024] | -0.001 [-0.032, +0.027] | 1.003 [0.959, 1.071] | +0.237 [+0.168, +0.300] |
| 123 | 33,662 | +0.023 [-0.016, +0.061] | +0.020 [-0.020, +0.058] | +0.021 [-0.006, +0.048] | +0.032 [+0.021, +0.046] | -0.010 [-0.051, +0.027] | -0.013 [-0.053, +0.024] | 0.969 [0.928, 1.015] | +0.243 [+0.173, +0.317] |
