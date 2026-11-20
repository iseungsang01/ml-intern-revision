# B.68 — an auxiliary training target denoised by the neighbouring values

Registered: ces_prediction/experiments/PREREGISTRATION_B68.md. GPU runs (one training per fold, initialisation seed of the fold); read-outs on a CPU. T_i, cut population, the fifteen folds of re-partitions r2, r3, r4 of the pooled cross-validation of B.39. X is seq_v2 trained with one more T_i head on the offline-GP estimate of T_i (the row's own value withheld; weight 0.5); it is scored with its main head only. Skill of A over B = 1 - sum of squared errors of A / that of B on the same rows; 95 % two-level interval (a re-partition drawn among those read, then its discharges resampled; 20,000 draws); point = mean over re-partitions.

## Statements

| statement | value (interval) | per re-partition | holds | expectation (section 6) | point inside the expected range |
|---|---|---|---|---|---|
| **X1** X over the network (v2), interval above zero | -0.019 [-0.062, +0.012] | r2 -0.001, r3 -0.022, r4 -0.033 | does not hold | -0.01 to +0.03, uncertain | False |
| **X2** X over the twin (v2h), interval above zero | +0.035 [+0.001, +0.062] | r2 +0.044, r3 +0.030, r4 +0.031 | holds | +0.04 to +0.07, met | False |
| **X3** median absolute error of X / that of the causal GP, interval below 1 | 1.016 [0.988, 1.045] | r2 1.025, r3 1.019, r4 1.004 | does not hold | uncertain; the network's median error is about that of the causal GP | - |

## Every network over the references

| network | over the causal GP | over hold | over the offline GP |
|---|---|---|---|
| network v2 (B.39) | +0.092 [+0.068, +0.114] | +0.378 [+0.358, +0.396] | -0.033 [-0.063, -0.006] |
| twin v2h (B.39) | +0.041 [+0.022, +0.059] | +0.344 [+0.325, +0.362] | -0.090 [-0.117, -0.066] |
| X: v2 trained with the auxiliary target | +0.075 [+0.034, +0.107] | +0.366 [+0.335, +0.392] | -0.052 [-0.100, -0.012] |

Network over the twin: +0.053 [+0.027, +0.075] (r2 +0.045, r3 +0.051, r4 +0.062).

## By rows

Rows 1 to 3 after an absent frame: B.54 row table (flag blip13) on the rows of the network. Rows per re-partition: r2: 191,934 rows, 3,094 after an absent frame, 635 discharges; r3: 192,092 rows, 3,130 after an absent frame, 635 discharges; r4: 192,132 rows, 3,111 after an absent frame, 635 discharges.

| comparison | rows 1 to 3 after an absent frame | other rows |
|---|---|---|
| X over network | -0.008 [-0.065, +0.030] | -0.019 [-0.063, +0.012] |
| X over twin | +0.146 [+0.094, +0.202] | +0.031 [-0.004, +0.059] |
| network over twin | +0.153 [+0.088, +0.220] | +0.049 [+0.025, +0.072] |

## Median absolute errors

Point: mean over re-partitions of the plain median (eV) or of the ratio of the plain medians; interval: the two-level draws, each median the weighted median of the rows with their discharge's resampling count as weight, the same draws for numerator and denominator.

| network or reference | median absolute error (eV) |
|---|---|
| network v2 (B.39) | 45.8 |
| twin v2h (B.39) | 45.4 |
| X: v2 trained with the auxiliary target | 44.8 |
| causal GP | 44.1 |
| hold | 45.4 |
| offline GP | 37.5 |

| ratio of median absolute errors | value (interval) | per re-partition |
|---|---|---|
| X / causal GP (X3) | 1.016 [0.988, 1.045] | r2 1.025, r3 1.019, r4 1.004 |
| network / causal GP | 1.038 [1.003, 1.089] | r2 1.067, r3 1.020, r4 1.027 |
| twin / causal GP | 1.029 [1.004, 1.058] | r2 1.013, r3 1.044, r4 1.032 |
| X / network | 0.979 [0.948, 1.010] | r2 0.960, r3 0.999, r4 0.977 |

## The auxiliary target in training (b68_aux.json of each run)

| re-partition | fold | weight | training rows with a target | measured T_i rows with a target | validation MSE of the main heads / of the auxiliary head at the best epoch (standardized) |
|---|---|---|---|---|---|
| r2 | f0 | 0.5 | 0.9938 | 0.9971 | 0.2549 / 0.1082 |
| r2 | f1 | 0.5 | 0.9938 | 0.9971 | 0.8052 / 0.1240 |
| r2 | f2 | 0.5 | 0.9942 | 0.9970 | 0.8762 / 0.1409 |
| r2 | f3 | 0.5 | 0.9948 | 0.9971 | 0.8593 / 0.1246 |
| r2 | f4 | 0.5 | 0.9938 | 0.9970 | 0.9529 / 0.1279 |
| r3 | f0 | 0.5 | 0.9945 | 0.9969 | 0.8287 / 0.1353 |
| r3 | f1 | 0.5 | 0.9945 | 0.9969 | 0.4266 / 0.1401 |
| r3 | f2 | 0.5 | 0.9946 | 0.9969 | 0.3124 / 0.1144 |
| r3 | f3 | 0.5 | 0.9947 | 0.9971 | 0.3352 / 0.1107 |
| r3 | f4 | 0.5 | 0.9947 | 0.9970 | 0.3382 / 0.1343 |
| r4 | f0 | 0.5 | 0.9933 | 0.9969 | 0.2063 / 0.0924 |
| r4 | f1 | 0.5 | 0.9933 | 0.9969 | 0.2624 / 0.0958 |
| r4 | f2 | 0.5 | 0.9935 | 0.9969 | 0.2812 / 0.0955 |
| r4 | f3 | 0.5 | 0.9932 | 0.9968 | 0.2695 / 0.1012 |
| r4 | f4 | 0.5 | 0.9942 | 0.9970 | 0.3168 / 0.1004 |

## Epochs run (best epoch)

| re-partition | runs |
|---|---|
| r2 | est f0 43 (37); twin f0 45 (39); X f0 24 (18); est f1 45 (39); twin f1 44 (38); X f1 43 (37); est f2 26 (20); twin f2 48 (42); X f2 25 (19); est f3 30 (24); twin f3 70 (64); X f3 39 (33); est f4 48 (42); twin f4 27 (21); X f4 51 (45) |
| r3 | est f0 35 (29); twin f0 35 (29); X f0 23 (17); est f1 26 (20); twin f1 25 (19); X f1 25 (19); est f2 40 (34); twin f2 56 (50); X f2 32 (26); est f3 47 (41); twin f3 30 (24); X f3 63 (57); est f4 37 (31); twin f4 80 (80); X f4 22 (16) |
| r4 | est f0 22 (16); twin f0 30 (24); X f0 24 (18); est f1 46 (40); twin f1 35 (29); X f1 56 (50); est f2 43 (37); twin f2 45 (39); X f2 49 (43); est f3 56 (50); twin f3 37 (31); X f3 38 (32); est f4 46 (40); twin f4 44 (38); X f4 42 (36) |

## Choices fixed in the implementation

- Networks of B.39 on the same folds: `.b39p_v2_cut_r{r}_f{k}` (the network) and `.b39p_v2h_cut_r{r}_f{k}` (the twin).
- The auxiliary target is the offline-GP estimator of the B.39 harness evaluated on the harness's file array (rows with at least one measured target); a grid row with no measured target is inserted into that array at its time. It is NaN (masked) where the estimator gives no value (no earlier measured T_i in the block).
- X3 interval: the weighted lower median under the discharge resampling counts; the point is the plain median.

