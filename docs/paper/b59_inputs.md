# B.59 — what the estimator must read, with pooled power

Registered: ces_prediction/experiments/PREREGISTRATION_B59.md. GPU runs (one training per fold, initialisation seed of the fold); read-outs on a CPU. T_i, cut population, the fifteen folds of re-partitions r2, r3, r4 of the pooled cross-validation of B.39. Skill of A over R = 1 - sum of squared errors of A / that of R on the same rows; 95 % two-level interval (a re-partition drawn among those read, then its discharges resampled; 20,000 draws); point = mean over re-partitions.

## Statements

| statement | skill (interval) | per re-partition | holds | expectation (section 6) | point inside the expected range |
|---|---|---|---|---|---|
| **L1** L over the twin, interval above zero | -0.006 [-0.025, +0.016] | r2 -0.012, r3 -0.012, r4 +0.007 | does not hold | -0.01 to +0.02, does not hold | True |
| **L2** estimator over L, interval above zero | +0.058 [+0.044, +0.072] | r2 +0.056, r3 +0.062, r4 +0.056 | holds | +0.03 to +0.06, holds | True |
| **N1** N over the twin, interval above zero | +0.060 [+0.023, +0.085] | r2 +0.066, r3 +0.044, r4 +0.072 | holds | +0.035 to +0.06, holds | False |
| **N2** estimator over N, interval inside [-0.01, +0.01] | -0.008 [-0.036, +0.024] | r2 -0.023, r3 +0.008, r4 -0.010 | does not hold | -0.008 to +0.012, interval about 0.012 wide on each side; may fail by width | False |
| **E1** E over the twin, interval above zero | -0.011 [-0.092, +0.022] | r2 -0.005, r3 -0.043, r4 +0.014 | does not hold | +0.01 to +0.03, holds or fails by a small margin | False |
| **B1** B over the twin, interval above zero | +0.044 [+0.020, +0.066] | r2 +0.047, r3 +0.034, r4 +0.053 | holds | +0.035 to +0.06, holds | True |

## Every network over the references

| network | over the causal GP | over hold | over the offline GP |
|---|---|---|---|
| estimator (B.39) | +0.092 [+0.067, +0.114] | +0.378 [+0.358, +0.397] | -0.033 [-0.063, -0.005] |
| twin (B.39) | +0.041 [+0.022, +0.059] | +0.344 [+0.324, +0.362] | -0.090 [-0.117, -0.065] |
| L: fast channels one row old | +0.036 [+0.013, +0.057] | +0.340 [+0.319, +0.359] | -0.096 [-0.127, -0.068] |
| N: no Mirnov | +0.099 [+0.062, +0.126] | +0.383 [+0.356, +0.405] | -0.024 [-0.070, +0.009] |
| E: ECEI + history | +0.030 [-0.045, +0.063] | +0.336 [+0.281, +0.363] | -0.102 [-0.195, -0.061] |
| B: BES + history | +0.084 [+0.058, +0.107] | +0.373 [+0.351, +0.392] | -0.042 [-0.076, -0.013] |

| comparison | skill (interval) | per re-partition |
|---|---|---|
| estimator over E | +0.063 [+0.036, +0.131] | r2 +0.050, r3 +0.090, r4 +0.049 |
| estimator over B | +0.009 [-0.019, +0.031] | r2 -0.002, r3 +0.018, r4 +0.010 |
| B over E | +0.055 [+0.025, +0.115] | r2 +0.052, r3 +0.073, r4 +0.039 |
| estimator over the twin | +0.053 [+0.027, +0.076] | r2 +0.045, r3 +0.051, r4 +0.062 |

## Over the twin, by rows

Rows 1 to 3 after an absent frame: run_b54 row location (data/.b54/rows_r{r}_f{k}.npz, flag blip13) on the rows of the estimator. Rows per re-partition: r2: 191,934 rows, 3,094 after an absent frame, 635 discharges; r3: 192,092 rows, 3,130 after an absent frame, 635 discharges; r4: 192,132 rows, 3,111 after an absent frame, 635 discharges.

| network | rows 1 to 3 after an absent frame | other rows |
|---|---|---|
| estimator (B.39) | +0.153 [+0.089, +0.220] | +0.049 [+0.025, +0.072] |
| L: fast channels one row old | -0.018 [-0.071, +0.022] | -0.005 [-0.025, +0.017] |
| N: no Mirnov | +0.162 [+0.103, +0.223] | +0.057 [+0.019, +0.082] |
| E: ECEI + history | -0.008 [-0.042, +0.022] | -0.011 [-0.095, +0.023] |
| B: BES + history | +0.147 [+0.091, +0.207] | +0.041 [+0.016, +0.062] |

## Median absolute errors (eV, mean over re-partitions)

| network or reference | median absolute error |
|---|---|
| estimator (B.39) | 45.8 |
| twin (B.39) | 45.4 |
| L: fast channels one row old | 47.9 |
| N: no Mirnov | 44.7 |
| E: ECEI + history | 45.4 |
| B: BES + history | 46.0 |
| causal GP | 44.1 |
| hold | 45.4 |
| offline GP | 37.5 |

## Epochs run (best epoch)

| re-partition | runs |
|---|---|
| r2 | est f0 43 (37); twin f0 45 (39); L f0 37 (31); N f0 54 (48); E f0 57 (51); B f0 69 (63); est f1 45 (39); twin f1 44 (38); L f1 37 (31); N f1 36 (30); E f1 31 (25); B f1 52 (46); est f2 26 (20); twin f2 48 (42); L f2 20 (14); N f2 28 (22); E f2 30 (24); B f2 21 (15); est f3 30 (24); twin f3 70 (64); L f3 32 (26); N f3 50 (44); E f3 16 (10); B f3 50 (44); est f4 48 (42); twin f4 27 (21); L f4 29 (23); N f4 41 (35); E f4 43 (37); B f4 41 (35) |
| r3 | est f0 35 (29); twin f0 35 (29); L f0 25 (19); N f0 42 (36); E f0 27 (21); B f0 48 (42); est f1 26 (20); twin f1 25 (19); L f1 27 (21); N f1 22 (16); E f1 34 (28); B f1 59 (53); est f2 40 (34); twin f2 56 (50); L f2 29 (23); N f2 39 (33); E f2 59 (53); B f2 32 (26); est f3 47 (41); twin f3 30 (24); L f3 49 (43); N f3 62 (56); E f3 49 (43); B f3 37 (31); est f4 37 (31); twin f4 80 (80); L f4 28 (22); N f4 57 (51); E f4 23 (17); B f4 43 (37) |
| r4 | est f0 22 (16); twin f0 30 (24); L f0 25 (19); N f0 38 (32); E f0 25 (19); B f0 27 (21); est f1 46 (40); twin f1 35 (29); L f1 42 (36); N f1 47 (41); E f1 31 (25); B f1 41 (35); est f2 43 (37); twin f2 45 (39); L f2 33 (27); N f2 53 (47); E f2 16 (10); B f2 80 (78); est f3 56 (50); twin f3 37 (31); L f3 36 (30); N f3 39 (33); E f3 35 (29); B f3 73 (67); est f4 46 (40); twin f4 44 (38); L f4 41 (35); N f4 53 (47); E f4 22 (16); B f4 51 (45) |
