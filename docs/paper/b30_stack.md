# B.30 row tables: does a combination of the stored estimators have a lower error than the network? (post hoc, not registered)

T_i, a CES sample visible every 10 ms. Least-squares weights fitted on one half of the test discharges and applied to the other half, and the reverse. Skill = 1 - SSE(stack) / SSE(reference), 95 % interval over 2,000 resamples of discharges.

## cut

| stack | split | weights (mean of the two folds) | RMSE reference -> stack (eV) | skill of the stack over its reference |
|---|---|---|---|---|
| causal (vs causal variant) | 42 | causal variant +0.64, causal GP +0.39, hold -0.01, history-only network -0.03 | 164.0 -> 160.9 | +0.037 [+0.013, +0.064] |
| causal (vs causal variant) | 1 | causal variant +0.62, causal GP +0.30, hold -0.11, history-only network +0.20 | 199.6 -> 197.2 | +0.024 [+0.008, +0.039] |
| causal (vs causal variant) | 7 | causal variant +0.86, causal GP +0.31, hold -0.11, history-only network -0.06 | 180.9 -> 179.0 | +0.021 [+0.010, +0.031] |
| causal (vs causal variant) | 123 | causal variant +0.78, causal GP +0.34, hold -0.11, history-only network +0.01 | 207.6 -> 204.2 | +0.032 [+0.020, +0.045] |
| headline (vs network) | 42 | network +0.64, causal GP +0.37, hold -0.00 | 163.2 -> 160.0 | +0.038 [+0.020, +0.060] |
| headline (vs network) | 1 | network +0.82, causal GP +0.28, hold -0.10 | 195.3 -> 194.8 | +0.005 [-0.006, +0.015] |
| headline (vs network) | 7 | network +0.82, causal GP +0.27, hold -0.09 | 181.1 -> 179.6 | +0.016 [+0.007, +0.025] |
| headline (vs network) | 123 | network +0.72, causal GP +0.34, hold -0.06 | 206.7 -> 204.3 | +0.024 [+0.005, +0.045] |
| offline (vs network) | 42 | network +0.67, PCHIP (two-sided) +0.33 | 163.2 -> 157.2 | +0.072 [+0.037, +0.109] |
| offline (vs network) | 1 | network +0.80, PCHIP (two-sided) +0.21 | 195.3 -> 195.1 | +0.001 [-0.019, +0.024] |
| offline (vs network) | 7 | network +0.79, PCHIP (two-sided) +0.21 | 181.1 -> 178.9 | +0.024 [+0.006, +0.045] |
| offline (vs network) | 123 | network +0.80, PCHIP (two-sided) +0.20 | 206.7 -> 204.3 | +0.024 [+0.001, +0.051] |

