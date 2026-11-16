# B.37 — four readouts on frozen estimators (registered: PREREGISTRATION_B37.md)

CPU, no network trained. T_i, cut population, natural cadence. Intervals: 95 %, resamples of discharges.

## V. Models fitted on validation discharges, applied once to the test rows

- **V1** (the tenth of rows with the largest predicted size holds more than 0.30 of the squared error): 4 of 4 -> holds.
- **V2** (the adaptive 90 % interval covers 0.87 to 0.93 and is shorter than the constant one): 3 of 4 -> holds.
- **V3** (the switch lowers the median error against both and keeps the mean): 4 of 4 -> holds.
- **V4** (a boosted layer on the signed residual lowers the error): 4 of 4 -> holds.

| split | validation discharges / rows | test rows | V1 share | V2 coverage | V2 median half-width over constant | half-widths (eV): adaptive median / constant | coverage of the constant interval | reported: coverage with the scale from out-of-fold predictions (width ratio) |
|---|---|---|---|---|---|---|---|---|
| 42 | 127 / 49,830 | 32,127 | 0.545 [0.459, 0.624] | 0.887 [0.872, 0.900] | 0.538 [0.472, 0.635] | 110.6 / 205.5 | 0.917 [0.894, 0.936] | 0.900 [0.887, 0.913] (0.566) |
| 1 | 128 / 46,919 | 35,138 | 0.472 [0.400, 0.537] | 0.867 [0.845, 0.887] | 0.752 [0.643, 0.856] | 139.0 / 184.8 | 0.865 [0.833, 0.896] | 0.881 [0.860, 0.899] (0.791) |
| 7 | 128 / 45,068 | 35,357 | 0.515 [0.432, 0.598] | 0.888 [0.875, 0.900] | 0.576 [0.493, 0.701] | 118.4 / 205.6 | 0.900 [0.870, 0.925] | 0.901 [0.889, 0.912] (0.605) |
| 123 | 128 / 46,217 | 33,662 | 0.529 [0.444, 0.601] | 0.885 [0.866, 0.899] | 0.499 [0.419, 0.650] | 105.8 / 212.3 | 0.893 [0.857, 0.921] | 0.898 [0.880, 0.911] (0.526) |

| split | V3 median error, switch over causal | switch over hold | skill of the switch over causal | rows using the causal variant | V4 layer over causal (skill) | median error, layer over causal |
|---|---|---|---|---|---|---|
| 42 | 0.923 [0.891, 0.951] | 0.937 [0.893, 0.988] | +0.000 [-0.017, +0.017] | 0.737 | +0.057 [+0.028, +0.088] | 0.895 [0.867, 0.922] |
| 1 | 0.827 [0.773, 0.874] | 0.934 [0.882, 0.978] | +0.011 [-0.003, +0.027] | 0.554 | +0.082 [+0.053, +0.110] | 0.831 [0.790, 0.873] |
| 7 | 0.938 [0.913, 0.965] | 0.881 [0.844, 0.922] | +0.009 [-0.009, +0.027] | 0.694 | +0.053 [+0.032, +0.074] | 0.936 [0.912, 0.961] |
| 123 | 0.940 [0.913, 0.957] | 0.936 [0.885, 0.983] | +0.008 [-0.006, +0.024] | 0.797 | +0.064 [+0.040, +0.091] | 0.924 [0.892, 0.957] |

Check before the readout (five test files per split, recomputed against the row table): 0 of 1,165 rows deviate, 0 of 1,545 rows deviate, 0 of 2,493 rows deviate, 0 of 957 rows deviate.

## F. Predicting an unusable CES value

- **F1** (the fast-channel window adds to the history of unusable rows, area under the ROC curve): 4 of 4 -> holds.

| split | test rows | base rate | AUC history / full | F1 difference | average precision history / full | onsets: rows | base rate | AUC history / full | difference | average precision history / full |
|---|---|---|---|---|---|---|---|---|---|---|
| 42 | 34,501 | 0.069 | 0.949 / 0.974 | +0.025 [+0.017, +0.034] | 0.765 / 0.811 | 32,138 | 0.022 | 0.858 / 0.940 | +0.082 [+0.059, +0.109] | 0.328 / 0.451 |
| 1 | 38,138 | 0.079 | 0.919 / 0.961 | +0.042 [+0.030, +0.054] | 0.702 / 0.761 | 35,143 | 0.032 | 0.818 / 0.929 | +0.111 [+0.088, +0.138] | 0.296 / 0.422 |
| 7 | 38,046 | 0.071 | 0.903 / 0.962 | +0.059 [+0.041, +0.077] | 0.656 / 0.739 | 35,367 | 0.030 | 0.784 / 0.933 | +0.149 [+0.116, +0.179] | 0.238 / 0.378 |
| 123 | 36,687 | 0.082 | 0.928 / 0.963 | +0.035 [+0.027, +0.046] | 0.678 / 0.728 | 33,669 | 0.036 | 0.861 / 0.947 | +0.086 [+0.065, +0.108] | 0.377 / 0.458 |

## Q. Skill of the causal variant with the fit noise taken out (reported)

sigma^2 = minus the lag-1 autocovariance of the 10 ms increments on the test discharges; subtracted from both mean squared errors.

| split | sigma, plain (eV) | sigma, clipped (eV) | RMSE hold / GP / twin / causal (eV) | over hold: raw, clipped-corrected | over the causal GP: raw, clipped-corrected | over the twin: raw, clipped-corrected | over the causal GP, plain-corrected |
|---|---|---|---|---|---|---|---|
| 42 | 120.9 [105.0, 136.9] | 85.6 [73.9, 97.3] | 204.5 / 170.1 / 169.2 / 164.5 | +0.353, +0.428 [+0.363, +0.490] | +0.065, +0.087 [+0.014, +0.149] | +0.055, +0.074 [+0.018, +0.128] | +0.131 [+0.021, +0.228] |
| 1 | 155.0 [131.8, 180.8] | 114.2 [95.8, 133.0] | 255.4 / 210.7 / 203.1 / 200.3 | +0.385, +0.481 [+0.431, +0.535] | +0.096, +0.136 [+0.098, +0.175] | +0.028, +0.041 [+0.006, +0.075] | +0.209 [+0.141, +0.284] |
| 7 | 135.9 [114.5, 157.1] | 98.4 [82.6, 116.1] | 233.7 / 194.2 / 192.4 / 181.2 | +0.399, +0.485 [+0.437, +0.540] | +0.130, +0.175 [+0.127, +0.233] | +0.113, +0.153 [+0.119, +0.199] | +0.254 [+0.176, +0.359] |
| 123 | 162.6 [134.9, 189.0] | 113.3 [93.8, 132.9] | 270.0 / 218.6 / 214.1 / 208.0 | +0.407, +0.494 [+0.437, +0.541] | +0.095, +0.130 [+0.078, +0.190] | +0.057, +0.079 [+0.033, +0.135] | +0.213 [+0.126, +0.320] |

## S. Means of the raw stream instead of snapshots

7 discharges, 3,537 scored rows inside the raw windows.

- **S1** (10 ms means over the network as delivered): -0.034 [-0.061, -0.026] -> does not hold.
- Reported: 1 ms means +0.008 [-0.004, +0.034]; the raw sample at the stamp (check) -0.005 [-0.012, +0.000].

| discharge | split | rows | skill, 10 ms means over delivered | BES sd, 10 ms means over delivered | median relative deviation of the raw sample from the delivered value (BES, ECEI) |
|---|---|---|---|---|---|
| 31114 | 42 | 505 | -0.067 | 1.030 | 1.15e-05, 0.00e+00 |
| 31873 | 42 | 730 | -0.043 | 1.383 | 0.00e+00, 2.63e-02 |
| 31902 | 42 | 580 | -0.043 | 1.320 | 1.42e-05, 0.00e+00 |
| 31921 | 42 | 444 | -0.023 | 0.934 | 0.00e+00, 0.00e+00 |
| 31923 | 1 | 388 | -0.022 | 0.929 | 3.64e-05, 0.00e+00 |
| 32027 | 123 | 395 | -0.043 | 1.130 | 1.12e-05, 0.00e+00 |
| 32092 | 1 | 495 | -0.099 | 1.034 | 1.85e-05, 0.00e+00 |
