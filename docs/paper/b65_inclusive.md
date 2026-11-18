# B.65 -- the pooled result on the population that keeps the values above 3 keV

Registration: `ces_prediction/experiments/PREREGISTRATION_B65.md`. Re-partitions r2 to r4 (15 folds), network `v2` and twin `v2h` trained and scored with CES_TI_SPIKE_CUT_EV=0; references by the B.39 code on the same rows. Interval: re-partition drawn, then discharges resampled, 20,000 draws.

## Statements

| statement | value [95 %] | holds | expectation |
|---|---|---|---|
| **I1**: network over causal GP, MSE skill | +0.109 [+0.070, +0.146] | holds | +0.00 to +0.05, uncertain |
| **I2**: network over twin, MSE skill | -0.002 [-0.017, +0.010] | does not hold | +0.00 to +0.04, uncertain |
| **I3**: network over causal GP, Huber skill | +0.037 [+0.003, +0.070] | holds | +0.04 to +0.08, met |
| **I4**: network over twin, Huber skill | -0.012 [-0.038, +0.006] | does not hold | +0.02 to +0.05, met |

## All inclusive rows

Rows per re-partition: r2 192,377, r3 192,420, r4 192,505; of them above 3 keV: r2 883, r3 890, r4 887; discharges: r2 634, r3 635, r4 633.

| A over R | MSE skill | Huber skill (200 eV) | 1 % trimmed skill |
|---|---|---|---|
| network over causal GP | +0.109 [+0.070, +0.146] | +0.037 [+0.003, +0.070] | +0.034 [-0.075, +0.124] |
| network over twin | -0.002 [-0.017, +0.010] | -0.012 [-0.038, +0.006] | -0.049 [-0.115, -0.005] |
| network over hold | +0.395 [+0.359, +0.426] | +0.173 [+0.142, +0.202] | -0.160 [-0.329, -0.025] |
| network over offline GP | +0.023 [-0.023, +0.063] | -0.060 [-0.104, -0.020] | +0.033 [-0.088, +0.133] |
| network over PCHIP | +0.238 [+0.185, +0.283] | +0.070 [+0.028, +0.106] | +0.348 [+0.241, +0.434] |
| twin over causal GP | +0.111 [+0.075, +0.146] | +0.049 [+0.023, +0.073] | +0.080 [-0.013, +0.156] |

Per re-partition (r2, r3, r4): mse|c|gp +0.112, +0.114, +0.102; mse|c|t -0.002, +0.004, -0.009; huber|c|gp +0.053, +0.034, +0.025; huber|c|t -0.005, -0.009, -0.024.

| estimator | RMSE (eV) [95 %] | mean abs. error (eV) | RMSE, trimmed rows (eV) | median abs. error (eV) |
|---|---|---|---|---|
| network | 476.7 [434.2, 520.5] | 136.7 | 212.4 | 59.5 |
| twin | 476.2 [433.4, 519.8] | 135.1 | 207.4 | 58.3 |
| causal GP | 505.1 [459.7, 551.1] | 133.9 | 216.1 | 45.2 |
| hold | 613.0 [555.6, 672.0] | 148.7 | 197.2 | 45.8 |
| offline GP | 482.4 [438.0, 527.5] | 122.7 | 216.0 | 38.8 |
| PCHIP | 546.1 [493.5, 599.6] | 134.0 | 263.0 | 38.5 |

Point values: mean over the three re-partitions; the median absolute error is the mean of the three per-re-partition medians (no interval).

## Rows with a value at or below 3 keV

Rows per re-partition: r2 191,494, r3 191,530, r4 191,618; of them above 3 keV: r2 0, r3 0, r4 0; discharges: r2 634, r3 635, r4 633.

| A over R | MSE skill | Huber skill (200 eV) | 1 % trimmed skill |
|---|---|---|---|
| network over causal GP | +0.476 [+0.405, +0.537] | +0.077 [+0.029, +0.125] | +0.193 [+0.084, +0.275] |
| network over twin | -0.015 [-0.115, +0.045] | -0.017 [-0.058, +0.011] | -0.057 [-0.165, +0.009] |
| network over hold | +0.768 [+0.735, +0.795] | +0.240 [+0.201, +0.278] | -0.039 [-0.167, +0.042] |
| network over offline GP | +0.367 [+0.291, +0.435] | -0.043 [-0.100, +0.013] | +0.177 [+0.070, +0.261] |
| network over PCHIP | +0.661 [+0.611, +0.701] | +0.127 [+0.076, +0.175] | +0.476 [+0.395, +0.539] |
| twin over causal GP | +0.484 [+0.417, +0.540] | +0.093 [+0.056, +0.129] | +0.237 [+0.165, +0.303] |

Per re-partition (r2, r3, r4): mse|c|gp +0.496, +0.478, +0.455; mse|c|t +0.013, +0.024, -0.082; huber|c|gp +0.101, +0.072, +0.060; huber|c|t -0.004, -0.011, -0.037.

| estimator | RMSE (eV) [95 %] | mean abs. error (eV) | RMSE, trimmed rows (eV) | median abs. error (eV) |
|---|---|---|---|---|
| network | 219.2 [205.6, 233.2] | 113.0 | 173.0 | 59.1 |
| twin | 217.7 [205.2, 230.7] | 111.3 | 168.3 | 57.9 |
| causal GP | 303.0 [279.0, 327.4] | 111.9 | 192.6 | 44.8 |
| hold | 455.0 [414.7, 496.2] | 126.3 | 169.7 | 45.4 |
| offline GP | 275.5 [254.0, 297.6] | 101.2 | 190.7 | 38.5 |
| PCHIP | 376.7 [342.5, 411.2] | 112.6 | 238.9 | 38.2 |

Point values: mean over the three re-partitions; the median absolute error is the mean of the three per-re-partition medians (no interval).
