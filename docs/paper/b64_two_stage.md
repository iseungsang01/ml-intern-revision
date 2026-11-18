# B.64 — the two-stage estimator under the pooled protocol

Registered: ces_prediction/experiments/PREREGISTRATION_B64.md. T_i, cut population, re-partitions r2 to r4 of B.39 (15 outer folds); 120 GPU runs of the out-of-fold networks, the second stage (B.38's model) on a CPU. Rows: the rows scored by B.39. Intervals: 95 %, two-level bootstrap (a re-partition drawn, then its discharges), 20,000 draws for sums, 2,000 for medians.

- **T1** (two-stage network over the network: skill interval above zero): +0.093 [+0.078, +0.107] -> **holds**.
- **T2** (two-stage network over the causal GP: interval above zero and point above the network's): +0.176 [+0.155, +0.196] (network alone over the causal GP +0.092 [+0.068, +0.114]) -> **holds**.
- **T3** (two-stage network over the two-stage twin: interval above zero): +0.097 [+0.077, +0.116] -> **holds**.
- **T4** (median absolute error, two-stage network over hold: interval below 1): 0.909 [0.889, 0.931] -> **holds**.
- **T5** (share within 20 %, two-stage network minus causal GP: interval above zero): +0.021 [+0.018, +0.025] -> **holds**.
- **T6** (two-stage twin over the twin: interval above zero): +0.048 [+0.038, +0.059] -> **holds**.

## Skills (mean-squared error)

| estimator | over hold | over causal GP | over offline GP | over PCHIP | over KFX |
|---|---|---|---|---|---|
| two-stage network | +0.436 [+0.418, +0.452] | +0.176 [+0.155, +0.196] | +0.063 [+0.038, +0.086] | +0.279 [+0.251, +0.304] | +0.130 [+0.111, +0.149] |
| two-stage twin | +0.375 [+0.359, +0.391] | +0.088 [+0.075, +0.100] | -0.037 [-0.056, -0.020] | +0.202 [+0.173, +0.228] | +0.037 [+0.023, +0.051] |
| network | +0.378 [+0.358, +0.397] | +0.092 [+0.068, +0.114] | -0.033 [-0.062, -0.005] | +0.205 [+0.173, +0.234] | +0.042 [+0.020, +0.061] |

| pair | MSE skill | Huber skill (200 eV) | 1 %-trimmed skill | RMSE difference (eV) | share within 20 %, difference | within 10 %, difference |
|---|---|---|---|---|---|---|
| two-stage network over network | +0.093 [+0.078, +0.107] | +0.100 [+0.086, +0.112] | +0.119 [+0.102, +0.137] | -8.730 [-10.195, -7.237] | +0.023 [+0.017, +0.033] | +0.036 [+0.029, +0.047] |
| two-stage network over causal GP | +0.176 [+0.155, +0.196] | +0.172 [+0.158, +0.186] | +0.177 [+0.145, +0.204] | -17.807 [-20.191, -15.448] | +0.021 [+0.018, +0.025] | +0.030 [+0.023, +0.035] |
| network over causal GP | +0.092 [+0.068, +0.114] | +0.080 [+0.062, +0.098] | +0.066 [+0.024, +0.100] | -9.076 [-11.450, -6.620] | -0.002 [-0.013, +0.007] | -0.007 [-0.023, +0.004] |
| two-stage network over two-stage twin | +0.097 [+0.077, +0.116] | +0.104 [+0.089, +0.119] | +0.133 [+0.102, +0.158] | -9.143 [-11.081, -7.248] | +0.015 [+0.012, +0.019] | +0.018 [+0.011, +0.024] |
| network over twin | +0.053 [+0.028, +0.076] | +0.055 [+0.031, +0.078] | +0.073 [+0.033, +0.101] | -5.058 [-7.354, -2.624] | +0.006 [-0.010, +0.015] | +0.002 [-0.019, +0.016] |
| two-stage twin over twin | +0.048 [+0.038, +0.059] | +0.051 [+0.038, +0.062] | +0.057 [+0.044, +0.070] | -4.646 [-5.727, -3.592] | +0.013 [+0.008, +0.020] | +0.020 [+0.014, +0.027] |
| two-stage twin over causal GP | +0.088 [+0.075, +0.100] | +0.076 [+0.067, +0.084] | +0.051 [+0.034, +0.067] | -8.664 [-10.012, -7.309] | +0.006 [+0.004, +0.008] | +0.012 [+0.009, +0.015] |

| estimator | RMSE (eV) | mean absolute error (eV) | median absolute error (eV) | share within 20 % | within 10 % |
|---|---|---|---|---|---|
| two-stage network | 175.4 [165.9, 185.3] | 83.9 [79.0, 89.1] | 41.2 [38.8, 44.0] | 0.793 [0.774, 0.812] | 0.599 [0.577, 0.620] |
| two-stage twin | 184.6 [174.8, 194.6] | 89.8 [84.4, 95.5] | 43.0 [40.3, 46.0] | 0.778 [0.759, 0.797] | 0.581 [0.559, 0.602] |
| network | 184.2 [174.3, 194.4] | 90.8 [85.4, 96.5] | 45.8 [42.7, 49.4] | 0.770 [0.747, 0.792] | 0.563 [0.536, 0.587] |
| twin | 189.2 [179.3, 199.5] | 93.5 [87.8, 99.6] | 45.4 [42.6, 48.7] | 0.765 [0.743, 0.785] | 0.561 [0.538, 0.584] |
| hold | 233.6 [220.1, 247.3] | 107.9 [100.8, 115.4] | 45.4 [42.4, 48.7] | 0.749 [0.728, 0.769] | 0.544 [0.523, 0.565] |
| causal GP | 193.3 [183.0, 203.8] | 94.2 [88.6, 100.3] | 44.1 [41.3, 47.3] | 0.772 [0.752, 0.791] | 0.569 [0.548, 0.591] |
| offline GP | 181.2 [171.1, 191.7] | 84.8 [79.4, 90.6] | 37.5 [34.9, 40.4] | 0.805 [0.785, 0.824] | 0.617 [0.594, 0.639] |
| PCHIP | 206.6 [194.2, 219.3] | 95.0 [88.5, 101.9] | 38.1 [35.5, 41.2] | 0.786 [0.765, 0.806] | 0.591 [0.569, 0.613] |
| KFX | 188.1 [177.9, 198.7] | 88.7 [83.4, 94.5] | 41.8 [39.3, 44.7] | 0.783 [0.763, 0.802] | 0.583 [0.561, 0.605] |

Median absolute error ratios: two-stage network over hold 0.909 [0.889, 0.931]; two-stage network over network 0.900 [0.871, 0.923]; two-stage network over causal GP 0.934 [0.918, 0.951]; two-stage twin over hold 0.949 [0.932, 0.966]; network over hold 1.010 [0.972, 1.062].

## Thirds of the discharges by median T_i

| third | two-stage network over network | over causal GP | network over causal GP | two-stage network over two-stage twin | two-stage twin over twin |
|---|---|---|---|---|---|
| low | +0.050 [+0.028, +0.069] | +0.118 [+0.094, +0.142] | +0.072 [+0.040, +0.105] | +0.037 [+0.020, +0.055] | +0.027 [+0.014, +0.043] |
| middle | +0.085 [+0.063, +0.111] | +0.165 [+0.124, +0.202] | +0.087 [+0.039, +0.128] | +0.089 [+0.054, +0.121] | +0.040 [+0.023, +0.057] |
| high | +0.144 [+0.123, +0.165] | +0.242 [+0.200, +0.276] | +0.115 [+0.062, +0.156] | +0.163 [+0.119, +0.199] | +0.075 [+0.056, +0.093] |

## Reported beside the statements (addendum 1): the boosted model's own handling of missing inputs on all rows

Not a statement. The same second stages, applied on every row (also where the BES / ECEI window inputs are missing).

| quantity | value |
|---|---|
| T1 form: two-stage network over the network | +0.093 [+0.077, +0.108] |
| T2 form: over the causal GP | +0.176 [+0.156, +0.196] |
| T3 form: two-stage network over two-stage twin | +0.096 [+0.077, +0.116] |
| T4 form: median absolute error over hold | 0.904 [0.883, 0.926] |
| T5 form: share within 20 %, minus causal GP | +0.023 [+0.019, +0.026] |
| T6 form: two-stage twin over twin | +0.049 [+0.039, +0.060] |
| RMSE (eV): two-stage network / two-stage twin | 175.4 [165.8, 185.1] / 184.5 [174.7, 194.5] |

## Checks

- B.39 network over twin on r2 to r4 from the stored errors: +0.053 [+0.027, +0.076] (registered +0.053 [+0.027, +0.076], tolerance 0.002) -> passes.
- Inputs: the twin's second stage reads time step (rows of 10 ms), age of the last CES value (ms), hold, first stage minus hold, first stage minus causal GP, |first stage minus hold|, |first stage minus causal GP|; the network's also BES level, BES departure from the 10-row mean, BES 10-row sd, BES one-row increment, ECEI level, ECEI departure from the 10-row mean, ECEI 10-row sd, ECEI one-row increment.
- Inner runs: 120; reached the epoch cap: ['.b64_v2h_cut_r2_f0_i3', '.b64_v2h_cut_r2_f3_i1', '.b64_v2h_cut_r3_f4_i0', '.b64_v2_cut_r4_f3_i1'].
- Test rows where the network's second-stage inputs are missing and both arms keep the first stage (addendum 1): network 0.0235, twin 0.0235 of rows.
- Second stage run time per row: network 2.2 us, twin 1.5 us (batch prediction, CPU).
