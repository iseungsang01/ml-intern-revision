# B.34 — predicting the size of the error (registered: PREREGISTRATION_B34.md)

CPU, no network trained. Frozen B.30 row tables, T_i, cut population, natural cadence. Gradient-boosted models on 15 inputs known at the row, fitted on one parity fold of the test discharges and applied to the other. Intervals: 95 %, 1,000 resamples of discharges.

- **E1** (the tenth of rows with the largest predicted size holds more than 0.30 of the squared error): 4 of 4 splits -> holds.
- **E2** (the adaptive 90 % interval covers 0.87 to 0.93 and is shorter than a constant one): 2 of 4 splits -> does not hold.
- **E3** (a switch between hold and the causal variant lowers the median error against both without losing the mean): 0 of 4 splits -> does not hold.

| split | rows | E1: share of squared error in the top tenth | same, without the fast-channel inputs | same, ranked by abs(causal - hold) alone | E2: coverage | E2: median half-width over constant | median half-width (eV) | constant half-width (eV) |
|---|---|---|---|---|---|---|---|---|
| 42 | 32,127 | 0.495 [0.406, 0.575] | 0.480 [0.397, 0.554] | 0.402 [0.330, 0.469] | 0.873 [0.851, 0.891] | 0.611 [0.534, 0.706] | 111.8 | 192.5 |
| 1 | 35,138 | 0.433 [0.375, 0.489] | 0.402 [0.349, 0.451] | 0.337 [0.289, 0.380] | 0.875 [0.852, 0.895] | 0.644 [0.556, 0.726] | 145.6 | 244.0 |
| 7 | 35,357 | 0.503 [0.422, 0.574] | 0.461 [0.394, 0.521] | 0.378 [0.324, 0.428] | 0.856 [0.837, 0.873] | 0.577 [0.504, 0.666] | 112.2 | 178.3 |
| 123 | 33,662 | 0.497 [0.412, 0.578] | 0.466 [0.399, 0.528] | 0.388 [0.327, 0.444] | 0.865 [0.845, 0.884] | 0.464 [0.360, 0.630] | 105.3 | 235.6 |

| split | E3: median error, switch over causal | switch over hold | skill of the switch over causal | over hold | rows using the causal variant | median abs error: causal / hold / switch (eV) |
|---|---|---|---|---|---|---|
| 42 | 0.943 [0.907, 0.975] | 0.958 [0.909, 1.024] | -0.027 [-0.044, -0.010] | +0.339 [+0.293, +0.376] | 0.773 | 44.9 / 44.2 / 42.4 |
| 1 | 0.842 [0.798, 0.899] | 0.950 [0.901, 1.005] | -0.002 [-0.015, +0.011] | +0.384 [+0.355, +0.408] | 0.616 | 55.9 / 49.5 / 47.1 |
| 7 | 0.947 [0.928, 0.970] | 0.889 [0.850, 0.941] | -0.002 [-0.021, +0.020] | +0.400 [+0.375, +0.423] | 0.743 | 46.7 / 49.8 / 44.3 |
| 123 | 0.958 [0.936, 0.973] | 0.954 [0.898, 1.008] | -0.013 [-0.036, +0.012] | +0.399 [+0.365, +0.428] | 0.815 | 43.0 / 43.2 / 41.2 |

Inputs the error model uses most (permutation importance, drop of R2 on the applied fold):

- split 42: |causal - hold| 0.151; hold value 0.092; |causal - GP| 0.084; ECEI sd of 100 ms 0.027; BES level 0.017; ECEI level 0.015
- split 1: |causal - hold| 0.143; |causal - GP| 0.082; BES level 0.055; hold value 0.033; ECEI level 0.032; causal - hold 0.014
- split 7: |causal - hold| 0.136; BES level 0.093; hold value 0.080; |causal - GP| 0.071; ECEI level 0.060; age of the last CES sample 0.018
- split 123: |causal - hold| 0.151; BES level 0.078; ECEI level 0.075; hold value 0.065; |causal - GP| 0.058; BES increment 0.018
