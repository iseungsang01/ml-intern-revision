# B.61 parts A and B -- absolute errors, robust skills, sensitivity to the discharges (registered: ces_prediction/experiments/PREREGISTRATION_B61.md)

CPU, no network run. T_i, cut population. Rows: the pooled rows of B.39 (80 runs, re-partitions r2 to r9, row tables of B.54) for the estimator, the twin, the causal GP, hold and the offline GP; the filters KF0 and KFX of B.56 on r3 to r9 (r2, their development set, reported apart), with every other estimator recomputed on those rows. Intervals: 95 %, 2,000 draws of the two-level bootstrap (a re-partition drawn, then its discharges resampled); every quantity and every difference of a table on the same draws. Point = mean over re-partitions of the per-re-partition value. Huber loss with threshold 200 eV: e^2/2 for |e| <= 200 eV, 200 (|e| - 100) above. Trimmed: within each re-partition the rows whose squared error of hold is at or above its 99th percentile are removed (the same rows for every estimator). Part C of the registration is run separately.

## Check before the readout

- Pooled skills of B.39: P1 +0.0889 (recorded +0.0889), P2 +0.0481 (recorded +0.0481); equal to three decimals: yes.
- Percentiles of the absolute error of B.54 (five estimators, p50 to p99, 25 values): 25 of 25 equal to one decimal (largest difference 0.0e+00 eV).
- Filter rows (B.56 cache) identical to the B.54 rows with the same reported values in 40 of 40 folds (largest value difference 0.0e+00 eV).
- The check passes.

## Statements

| statement | point [95 % interval] | reference | verdict |
|---|---|---|---|
| **A1**: Huber skill (200 eV) of the estimator over the causal GP | +0.074 [+0.052, +0.094] | 0 | **holds** |
| **A2**: Huber skill of the estimator over the twin | +0.052 [+0.031, +0.075] | 0 | **holds** |
| **A3**: trimmed skill (1 % of rows by hold error removed) over the causal GP | +0.067 [+0.029, +0.101] | 0 | **holds** |
| **A4**: trimmed skill over the twin | +0.070 [+0.037, +0.100] | 0 | **holds** |
| **A5**: Huber skill of KFX over KF0 (r3 to r9) | +0.067 [+0.058, +0.076] | 0 | **holds** |
| **B1**: the 64 discharges with the largest reduction removed: over the causal GP | +0.030 [+0.002, +0.055] | 0 | **holds** |
| **B2**: the 64 removed: over the twin | +0.027 [-0.004, +0.051] | 0 | **does not hold** |

## Part A: absolute errors (pooled rows, r2 to r9)

| estimator | RMSE (eV) | RMSE minus causal GP (eV) | RMSE vs causal GP (%) | MAE (eV) | MAE minus causal GP (eV) | MAE vs causal GP (%) | RMSE, trimmed rows (eV) |
|---|---|---|---|---|---|---|---|
| estimator | 184.6 [174.4, 194.6] | -8.8 [-11.3, -6.3] | -4.6 [-5.7, -3.3] | 91.2 [85.5, 97.1] | -3.0 [-4.7, -1.5] | -3.2 [-4.8, -1.6] | 139.2 [132.4, 146.6] |
| twin | 189.2 [179.4, 199.6] | -4.2 [-6.1, -2.2] | -2.2 [-3.1, -1.2] | 94.0 [88.3, 100.2] | -0.2 [-1.9, +1.5] | -0.3 [-2.0, +1.7] | 144.4 [137.4, 152.1] |
| causal GP | 193.4 [182.9, 203.9] | | | 94.2 [88.3, 100.3] | | | 144.1 [136.8, 151.7] |
| hold | 233.4 [219.9, 247.2] | +40.0 [+36.1, +44.1] | +20.7 [+19.2, +22.2] | 107.8 [100.5, 115.2] | +13.6 [+11.8, +15.5] | +14.4 [+13.0, +15.8] | 158.8 [150.1, 167.7] |
| offline GP | 181.3 [170.7, 191.6] | -12.1 [-13.4, -10.7] | -6.3 [-7.0, -5.5] | 84.8 [79.1, 90.6] | -9.4 [-10.2, -8.7] | -10.0 [-10.8, -9.2] | 134.6 [127.3, 142.0] |

Robust skills (r2 to r9):

| A over R | MSE skill | Huber skill (200 eV) | trimmed skill |
|---|---|---|---|
| estimator over causal GP | +0.089 [+0.064, +0.112] | +0.074 [+0.052, +0.094] | +0.067 [+0.029, +0.101] |
| estimator over twin | +0.048 [+0.022, +0.073] | +0.052 [+0.031, +0.075] | +0.070 [+0.037, +0.100] |
| estimator over hold | +0.375 [+0.351, +0.396] | +0.265 [+0.244, +0.286] | +0.231 [+0.188, +0.271] |
| estimator over offline GP | -0.037 [-0.068, -0.008] | -0.057 [-0.083, -0.030] | -0.070 [-0.119, -0.027] |
| twin over causal GP | +0.043 [+0.023, +0.061] | +0.022 [+0.004, +0.042] | -0.003 [-0.030, +0.021] |

Rows trimmed per re-partition: r2 1,920, r3 1,921, r4 1,922, r5 1,921, r6 1,922, r7 1,921, r8 1,922, r9 1,922.

## Part A with the filters (r3 to r9, every estimator on the same rows)

| estimator | RMSE (eV) | RMSE minus causal GP (eV) | RMSE vs causal GP (%) | MAE (eV) | MAE minus causal GP (eV) | MAE vs causal GP (%) | RMSE, trimmed rows (eV) |
|---|---|---|---|---|---|---|---|
| estimator | 184.6 [174.5, 195.0] | -8.8 [-11.2, -6.1] | -4.5 [-5.8, -3.2] | 91.2 [85.8, 96.9] | -3.1 [-4.6, -1.5] | -3.2 [-4.8, -1.6] | 139.1 [132.2, 146.0] |
| twin | 189.3 [179.0, 199.4] | -4.1 [-6.2, -2.2] | -2.1 [-3.2, -1.1] | 94.2 [88.5, 100.0] | -0.1 [-1.1, +1.5] | -0.1 [-1.1, +1.6] | 144.4 [137.4, 151.4] |
| causal GP | 193.4 [183.1, 203.8] | | | 94.2 [88.4, 100.1] | | | 144.2 [137.2, 151.0] |
| hold | 233.4 [219.7, 246.7] | +40.0 [+35.9, +43.9] | +20.7 [+19.0, +22.1] | 107.8 [100.6, 115.1] | +13.6 [+11.7, +15.4] | +14.4 [+12.9, +15.8] | 158.8 [150.2, 167.0] |
| offline GP | 181.3 [171.1, 191.6] | -12.1 [-13.6, -10.8] | -6.3 [-7.1, -5.6] | 84.8 [79.2, 90.4] | -9.4 [-10.2, -8.7] | -10.0 [-10.8, -9.2] | 134.5 [127.4, 141.4] |
| KF0 | 192.9 [182.5, 203.1] | -0.5 [-1.9, +0.9] | -0.3 [-1.0, +0.5] | 92.9 [87.1, 98.7] | -1.3 [-2.2, -0.4] | -1.4 [-2.3, -0.4] | 145.1 [137.9, 152.0] |
| KFX | 188.1 [177.5, 198.2] | -5.3 [-6.9, -3.8] | -2.8 [-3.6, -2.0] | 88.7 [83.2, 94.2] | -5.5 [-6.4, -4.7] | -5.9 [-6.7, -5.1] | 138.8 [131.8, 145.6] |

| A over R | MSE skill | Huber skill (200 eV) | trimmed skill |
|---|---|---|---|
| KFX over KF0 | +0.049 [+0.041, +0.059] | +0.067 [+0.058, +0.076] | +0.084 [+0.074, +0.096] |
| KFX over causal GP | +0.055 [+0.039, +0.070] | +0.085 [+0.073, +0.096] | +0.073 [+0.054, +0.090] |
| KF0 over causal GP | +0.005 [-0.009, +0.020] | +0.019 [+0.006, +0.032] | -0.013 [-0.031, +0.005] |
| KFX over twin | +0.013 [-0.007, +0.033] | +0.066 [+0.051, +0.082] | +0.075 [+0.054, +0.099] |
| KFX over estimator | -0.038 [-0.060, -0.012] | +0.013 [-0.009, +0.033] | +0.004 [-0.029, +0.038] |
| KFX over hold | +0.351 [+0.334, +0.365] | +0.274 [+0.260, +0.287] | +0.236 [+0.205, +0.262] |
| KFX over offline GP | -0.076 [-0.098, -0.055] | -0.044 [-0.061, -0.028] | -0.065 [-0.092, -0.041] |
| estimator over causal GP | +0.089 [+0.062, +0.112] | +0.073 [+0.052, +0.095] | +0.069 [+0.033, +0.101] |
| estimator over twin | +0.049 [+0.024, +0.074] | +0.054 [+0.033, +0.076] | +0.071 [+0.041, +0.099] |

r2 (development set of the filters, reported apart; interval over its discharges only): RMSE KFX 188.3 [178.3, 199.1] eV, KF0 193.1 [183.2, 203.8], causal GP 193.3 [183.2, 203.9]; Huber skill KFX over KF0 +0.066 [+0.057, +0.076], KFX over the causal GP +0.083 [+0.072, +0.093]; trimmed skill KFX over KF0 +0.083 [+0.072, +0.095].

## Part B: discharges

| removed (largest reduction against the causal GP, per re-partition) | share of the reduction they hold | over causal GP | over twin | over hold |
|---|---|---|---|---|
| 10 | 0.214 | +0.076 [+0.050, +0.099] | +0.041 [+0.016, +0.064] | +0.368 [+0.345, +0.388] |
| 32 | 0.493 | +0.055 [+0.029, +0.078] | +0.034 [+0.006, +0.056] | +0.359 [+0.334, +0.380] |
| 64 | 0.769 | +0.030 [+0.002, +0.055] | +0.027 [-0.004, +0.051] | +0.346 [+0.322, +0.368] |

Part B skills: run_b54.hier, 20,000 draws, resampling the remaining discharges only (the registration's 'pooled interval on the remaining discharges'). The k = 10 point equals B.54's; B.54's interval resampled every discharge with the removed ones contributing nothing, so its interval differs slightly (B.54: over the causal GP [+0.050, +0.098], over the twin [+0.016, +0.064]). Remaining discharges per re-partition for k = 64: r2 571, r3 571, r4 571, r5 571, r6 570, r7 570, r8 570, r9 569.

Overlap of the test sets of the four file-level splits (42, 1, 7, 123; `split_manifest.json`): test discharges per split 42: 96, 1: 96, 7: 96, 123: 96; 301 distinct discharges are a test discharge of at least one split, of them 224 in exactly 1, 72 in exactly 2, 4 in exactly 3, 1 in exactly 4 of the four; 338 of the 639 are a test discharge of none. Counted from the stored scored rows of the four estimator runs instead: 301 discharges, 224 in exactly 1, 72 in exactly 2, 4 in exactly 3, 1 in exactly 4.

## Expectations of the registration (section 6) beside the findings

| expectation | observed | range met | verdict | verdict met |
|---|---|---|---|---|
| RMSE of the estimator 180 to 190 eV | +184.5959 | yes |  |  |
| RMSE of the causal GP 4 to 5 % higher than the estimator's | +4.7665 | yes |  |  |
| RMSE of the twin 2 to 3 % higher than the estimator's | +2.4955 | yes |  |  |
| MAE of the estimator within 1 % of the causal GP's | -3.2119 | no |  |  |
| A1 +0.01 to +0.04, holds | +0.0735 | no | holds | yes |
| A2 +0.005 to +0.03, holds or fails by a small margin | +0.0524 | no | holds |  |
| A3 +0.02 to +0.07, holds | +0.0670 | yes | holds | yes |
| A4 +0.01 to +0.05, holds | +0.0700 | no | holds | yes |
| A5 +0.02 to +0.05, holds | +0.0667 | no | holds | yes |
| B1 +0.03 to +0.06, holds | +0.0298 | no | holds | yes |
| B2 +0.01 to +0.04, holds | +0.0266 | yes | does not hold | no |

## Choices made where the registration left room

- Huber loss in its usual form, e^2/2 below the threshold and 200 (|e| - 100) above, so that it is continuous; the skill is a ratio, so the factor 1/2 does not change it.
- Trimming: the 1 % of rows with the largest squared error of hold is taken as the rows at or above the 99th percentile of that error within each re-partition (the rule of run_b41 for 'top 1 % of the hold error'); on the filter rows (r3 to r9) the same rule within each re-partition.
- The five estimators are read on r2 to r9 (the statements A1 to A4); the filters and A5 on r3 to r9, where every estimator is recomputed on the same rows; r2 for the filters reported apart.
- Differences to the causal GP: in eV, RMSE_A - RMSE_GP; in per cent, 100 (RMSE_A / RMSE_GP - 1); likewise for the MAE; both within a draw.
- Part B: per re-partition the discharges are ranked by their summed squared-error reduction against the causal GP (as B.54) and the top k removed for every reference; skills with run_b54.hier (20,000 draws) on the remaining discharges.
- Overlap: from the test_files of the four split manifests (`data/.b1_w2cut_split_s{seed}`); the count from the stored runs is given beside it.
- Generators: run_b54.rng_for (default_rng([20261154, crc32(name)])).

## Limits

The Huber threshold and the trimming share are choices fixed before the readout. Removing discharges by their reduction selects on the outcome and gives a lower bound.

Run time: 7 s.
