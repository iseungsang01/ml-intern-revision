# B.33 — clusters of the fast-channel window (registered: PREREGISTRATION_B33.md)

CPU, nothing trained. Frozen B.30 row tables, T_i, cut population, natural cadence. k-means on nine causal features of the 100 ms fast-channel window, fitted on the non-test files of each split. Intervals: 95 %, 2,000 resamples of discharges.

- **K1** (the fast-channel increment differs between clusters; picked on one parity fold, tested on the other): holds on 1 of 4 splits -> does not hold.
- **K2** (a per-cluster affine correction of the causal variant's residual against a global one, cross-fitted): holds on 0 of 4 splits -> does not hold.

## Primary, k = 6

| split | scored rows | left out (no window) | K1 picked on E: high minus low on O | K1 picked on O: on E | K2 per-cluster over global | per-cluster over uncorrected | global over uncorrected |
|---|---|---|---|---|---|---|---|
| 42 | 32,127 | 0.024 | +0.081 [-0.113, +0.233] | +0.095 [+0.006, +0.217] | -0.011 [-0.021, -0.003] | -0.020 [-0.031, -0.010] | -0.009 [-0.015, -0.003] |
| 1 | 35,138 | 0.022 | +0.064 [+0.009, +0.142] | +0.089 [-0.029, +0.189] | +0.004 [-0.005, +0.013] | -0.007 [-0.020, +0.005] | -0.011 [-0.021, -0.003] |
| 7 | 35,357 | 0.023 | +0.126 [+0.059, +0.202] | +0.141 [+0.037, +0.260] | -0.001 [-0.007, +0.004] | -0.000 [-0.009, +0.007] | +0.000 [-0.004, +0.004] |
| 123 | 33,662 | 0.022 | -0.015 [-0.167, +0.107] | +0.032 [-0.025, +0.105] | +0.002 [-0.007, +0.011] | +0.010 [+0.001, +0.021] | +0.009 [+0.004, +0.014] |

## Sensitivity to k (decides nothing)

| k | K2 per-cluster over global, four splits | splits with K2 | splits with K1 |
|---|---|---|---|
| 3 | -0.003, +0.008, +0.001, +0.005 | 1 | 0 |
| 4 | -0.006, +0.008, -0.002, +0.006 | 1 | 1 |
| 6 | -0.011, +0.004, -0.001, +0.002 | 0 | 1 |
| 8 | -0.012, +0.005, +0.004, +0.002 | 0 | 0 |
| 12 | -0.006, +0.003, +0.003, -0.006 | 0 | 0 |

## K3: the clusters of split 42 (192 first rows after a blip among the scored rows)

| cluster | share of rows | BES level | BES departure | BES sd | ECEI level | ECEI departure | time step | hold RMSE (eV) | causal variant RMSE (eV) | increment | share of the first rows after a blip | rows within ten after a blip |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.553 | +0.70 | +0.07 | -0.10 | -0.44 | -0.00 | -0.06 | 135.0 | 105.4 | +0.085 | 0.000 | 0.049 |
| 1 | 0.207 | -1.23 | -0.06 | -0.37 | -0.37 | -0.02 | -0.06 | 274.2 | 227.6 | -0.016 | 0.000 | 0.036 |
| 3 | 0.139 | +0.03 | +0.37 | +0.29 | +1.32 | +0.77 | -0.06 | 256.8 | 208.9 | +0.123 | 0.000 | 0.099 |
| 2 | 0.087 | -0.04 | -0.36 | +0.20 | +0.43 | -1.56 | -0.06 | 239.5 | 176.0 | +0.088 | 0.000 | 0.066 |
| 4 | 0.008 | -0.62 | -2.37 | +3.09 | +0.37 | -0.39 | -0.06 | 417.9 | 363.1 | -0.003 | 0.000 | 0.152 |
| 5 | 0.007 | -0.81 | -3.00 | -0.04 | -0.03 | -0.47 | +9.14 | 352.5 | 229.1 | +0.387 | 1.000 | 0.919 |

## K3: the clusters of split 1 (192 first rows after a blip among the scored rows)

| cluster | share of rows | BES level | BES departure | BES sd | ECEI level | ECEI departure | time step | hold RMSE (eV) | causal variant RMSE (eV) | increment | share of the first rows after a blip | rows within ten after a blip |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 5 | 0.374 | +0.67 | +0.06 | -0.09 | -0.51 | -0.01 | -0.04 | 155.1 | 118.9 | +0.047 | 0.000 | 0.052 |
| 0 | 0.279 | -1.28 | -0.05 | -0.36 | -0.42 | -0.01 | -0.03 | 258.3 | 209.8 | -0.002 | 0.115 | 0.042 |
| 2 | 0.156 | -0.15 | -0.07 | +0.10 | +1.39 | -0.02 | -0.04 | 353.9 | 267.6 | -0.003 | 0.021 | 0.053 |
| 4 | 0.104 | +0.24 | +0.73 | +0.40 | +0.94 | +1.48 | -0.04 | 254.1 | 224.3 | +0.049 | 0.005 | 0.095 |
| 1 | 0.071 | -0.05 | -0.36 | +0.19 | +0.23 | -1.79 | -0.03 | 342.1 | 245.0 | +0.085 | 0.021 | 0.068 |
| 3 | 0.015 | -0.77 | -3.04 | +2.49 | +0.14 | -0.47 | +1.65 | 442.9 | 348.0 | +0.121 | 0.839 | 0.314 |

## K3: the clusters of split 7 (186 first rows after a blip among the scored rows)

| cluster | share of rows | BES level | BES departure | BES sd | ECEI level | ECEI departure | time step | hold RMSE (eV) | causal variant RMSE (eV) | increment | share of the first rows after a blip | rows within ten after a blip |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.436 | +0.67 | +0.05 | -0.10 | -0.52 | -0.02 | -0.04 | 129.4 | 96.6 | +0.113 | 0.000 | 0.041 |
| 0 | 0.228 | -1.28 | -0.04 | -0.39 | -0.44 | -0.01 | -0.03 | 228.8 | 185.6 | +0.049 | 0.102 | 0.040 |
| 3 | 0.147 | -0.15 | -0.08 | +0.08 | +1.38 | -0.00 | -0.04 | 317.2 | 247.4 | +0.088 | 0.011 | 0.051 |
| 5 | 0.103 | +0.30 | +0.80 | +0.51 | +0.92 | +1.43 | -0.04 | 276.8 | 233.9 | +0.184 | 0.005 | 0.105 |
| 2 | 0.066 | +0.01 | -0.34 | +0.22 | +0.23 | -1.81 | -0.03 | 386.8 | 264.3 | +0.146 | 0.016 | 0.078 |
| 4 | 0.020 | -0.76 | -2.98 | +2.31 | +0.17 | -0.47 | +1.42 | 404.5 | 300.2 | +0.168 | 0.866 | 0.285 |

## K3: the clusters of split 123 (190 first rows after a blip among the scored rows)

| cluster | share of rows | BES level | BES departure | BES sd | ECEI level | ECEI departure | time step | hold RMSE (eV) | causal variant RMSE (eV) | increment | share of the first rows after a blip | rows within ten after a blip |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.481 | +0.67 | +0.04 | -0.10 | -0.49 | -0.02 | -0.04 | 125.3 | 97.7 | +0.116 | 0.000 | 0.049 |
| 3 | 0.206 | -1.27 | -0.04 | -0.37 | -0.41 | -0.01 | -0.03 | 290.7 | 231.7 | +0.021 | 0.074 | 0.039 |
| 5 | 0.155 | -0.19 | -0.08 | +0.11 | +1.52 | +0.01 | -0.04 | 390.7 | 295.0 | +0.013 | 0.016 | 0.054 |
| 0 | 0.085 | +0.36 | +0.75 | +0.44 | +0.77 | +1.38 | -0.04 | 355.3 | 282.3 | +0.099 | 0.016 | 0.112 |
| 4 | 0.056 | +0.04 | -0.33 | +0.20 | +0.17 | -1.75 | -0.03 | 434.6 | 313.5 | +0.096 | 0.021 | 0.072 |
| 2 | 0.016 | -0.73 | -2.95 | +2.38 | +0.18 | -0.49 | +1.37 | 435.2 | 340.8 | +0.104 | 0.874 | 0.379 |

Cluster centres are in standardized feature units (zero = mean of the fit rows).
