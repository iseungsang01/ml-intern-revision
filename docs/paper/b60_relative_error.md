# B.60 -- the error as a share of the reported value (registered: ces_prediction/experiments/PREREGISTRATION_B60.md)

CPU, no network run. T_i, cut population, the rows of the pooled cross-validation of B.39 (row tables of B.54: 80 runs, repeats r2 to r9, five folds each, folds of a repeat concatenated). Relative error of a row = absolute error / reported value of the row. Estimators: the estimator (`.b39p_v2_cut`), the history-only twin, the causal GP, hold and the offline GP (stored errors of eval_seq). Intervals: 95 %, 2,000 draws of the two-level bootstrap of B.54 (a repeat drawn, then its discharges resampled); differences between estimators within a draw. Point = mean over the eight repeats. Medians are weighted medians over the resampled rows (`run_b41.wquant`).

## Check before the readout

Percentiles of the absolute error (eV, mean over repeats) against section 2 of the registration, to one decimal:

| estimator | p50 / p75 / p90 here | registered | equal |
|---|---|---|---|
| estimator | 46.00 / 100.56 / 205.87 | 46.0 / 100.6 / 205.9 | yes |
| causal GP | 44.13 / 104.00 / 217.59 | 44.1 / 104.0 / 217.6 | yes |
| hold | 45.39 / 111.38 / 254.69 | 45.4 / 111.4 / 254.7 | yes |
| offline GP | 37.54 / 91.45 / 198.00 | 37.5 / 91.5 / 198.0 | yes |
| twin | 46.08 / 103.77 / 216.04 | 46.1 / 103.8 / 216.0 | yes |

The check passes.

## Rows with a small reported value

Of 1,536,645 scored rows (all repeats), 0 have a reported value at or below zero, 20,865 (0.0136) are below 50 eV and 3,761 below 10 eV; the smallest is 0.0022 eV. The definition is applied unchanged: these rows have large relative errors and count as outside 10 % and 20 % unless the estimate is as close in relative terms. Share of them within 20 %: estimator 0.001, twin 0.000, causal GP 0.047, hold 0.075, offline GP 0.071. The cell 'rows with reported value >= 50 eV' below is an after-the-fact sensitivity readout, not registered.

## Statements

| statement | point [95 % interval] | reference | verdict |
|---|---|---|---|
| **T20**: share within 20 %, estimator minus causal GP; interval above 0 | -0.002 [-0.012, +0.006] | +0.00 | **does not hold** |
| **T20h**: share within 20 %, estimator minus hold; interval above 0 | +0.021 [+0.011, +0.031] | +0.00 | **holds** |
| **T10**: share within 10 %, estimator minus causal GP; interval above -0.02 | -0.007 [-0.020, +0.003] | -0.02 | **does not hold** |

## Shares and medians, all rows and strata

### all rows

Rows per repeat (mean) 192,081 (share of the scored rows 1.0000); discharges with rows per repeat 633 to 635.

| estimator | within 10 % | within 20 % | median relative error |
|---|---|---|---|
| estimator | 0.562 [0.539, 0.585] | 0.770 [0.749, 0.791] | 0.084 [0.079, 0.089] |
| twin | 0.556 [0.529, 0.581] | 0.761 [0.738, 0.783] | 0.085 [0.079, 0.092] |
| causal GP | 0.570 [0.549, 0.591] | 0.772 [0.753, 0.792] | 0.081 [0.077, 0.086] |
| hold | 0.544 [0.522, 0.565] | 0.749 [0.728, 0.770] | 0.087 [0.082, 0.093] |
| offline GP | 0.617 [0.594, 0.639] | 0.805 [0.785, 0.824] | 0.070 [0.066, 0.075] |

| estimator minus | within 10 % | within 20 % | median relative error |
|---|---|---|---|
| causal GP | -0.007 [-0.020, +0.003] | -0.002 [-0.012, +0.006] | +0.002 [-0.000, +0.005] |
| hold | +0.018 [+0.005, +0.029] | +0.021 [+0.011, +0.031] | -0.004 [-0.007, -0.001] |
| twin | +0.006 [-0.017, +0.025] | +0.009 [-0.008, +0.020] | -0.001 [-0.006, +0.004] |
| offline GP | -0.055 [-0.068, -0.044] | -0.035 [-0.045, -0.026] | +0.014 [+0.011, +0.017] |

### low third by median T_i

Rows per repeat (mean) 39,357 (share of the scored rows 0.2049); discharges with rows per repeat 211 to 212.

| estimator | within 10 % | within 20 % | median relative error |
|---|---|---|---|
| estimator | 0.297 [0.256, 0.336] | 0.495 [0.446, 0.545] | 0.203 [0.173, 0.237] |
| twin | 0.299 [0.262, 0.339] | 0.497 [0.453, 0.545] | 0.202 [0.172, 0.234] |
| causal GP | 0.342 [0.304, 0.381] | 0.534 [0.485, 0.580] | 0.177 [0.151, 0.211] |
| hold | 0.330 [0.293, 0.370] | 0.525 [0.475, 0.576] | 0.182 [0.153, 0.219] |
| offline GP | 0.386 [0.344, 0.430] | 0.575 [0.526, 0.623] | 0.152 [0.128, 0.182] |

| estimator minus | within 10 % | within 20 % | median relative error |
|---|---|---|---|
| causal GP | -0.045 [-0.082, -0.022] | -0.038 [-0.070, -0.013] | +0.026 [+0.009, +0.045] |
| hold | -0.033 [-0.072, -0.011] | -0.029 [-0.063, -0.003] | +0.021 [+0.002, +0.042] |
| twin | -0.002 [-0.065, +0.032] | -0.002 [-0.058, +0.031] | +0.001 [-0.021, +0.036] |
| offline GP | -0.089 [-0.129, -0.062] | -0.079 [-0.113, -0.052] | +0.051 [+0.035, +0.072] |

### middle third by median T_i

Rows per repeat (mean) 76,780 (share of the scored rows 0.3997); discharges with rows per repeat 211 to 212.

| estimator | within 10 % | within 20 % | median relative error |
|---|---|---|---|
| estimator | 0.660 [0.628, 0.688] | 0.843 [0.816, 0.866] | 0.065 [0.060, 0.071] |
| twin | 0.660 [0.625, 0.690] | 0.836 [0.808, 0.860] | 0.063 [0.059, 0.070] |
| causal GP | 0.665 [0.633, 0.693] | 0.839 [0.812, 0.862] | 0.062 [0.058, 0.068] |
| hold | 0.652 [0.621, 0.681] | 0.839 [0.811, 0.863] | 0.066 [0.061, 0.071] |
| offline GP | 0.719 [0.687, 0.748] | 0.871 [0.845, 0.894] | 0.053 [0.049, 0.057] |

| estimator minus | within 10 % | within 20 % | median relative error |
|---|---|---|---|
| causal GP | -0.005 [-0.022, +0.009] | +0.004 [-0.003, +0.011] | +0.003 [-0.000, +0.006] |
| hold | +0.008 [-0.009, +0.021] | +0.004 [-0.004, +0.012] | -0.001 [-0.004, +0.003] |
| twin | -0.000 [-0.019, +0.026] | +0.007 [-0.004, +0.018] | +0.002 [-0.004, +0.005] |
| offline GP | -0.059 [-0.076, -0.046] | -0.028 [-0.036, -0.021] | +0.012 [+0.009, +0.016] |

### high third by median T_i

Rows per repeat (mean) 75,944 (share of the scored rows 0.3954); discharges with rows per repeat 211 to 211.

| estimator | within 10 % | within 20 % | median relative error |
|---|---|---|---|
| estimator | 0.601 [0.573, 0.629] | 0.838 [0.817, 0.858] | 0.077 [0.072, 0.083] |
| twin | 0.585 [0.557, 0.614] | 0.821 [0.799, 0.842] | 0.080 [0.074, 0.086] |
| causal GP | 0.591 [0.565, 0.618] | 0.828 [0.809, 0.847] | 0.079 [0.073, 0.084] |
| hold | 0.546 [0.518, 0.573] | 0.774 [0.749, 0.799] | 0.087 [0.081, 0.095] |
| offline GP | 0.634 [0.607, 0.661] | 0.858 [0.839, 0.876] | 0.069 [0.064, 0.075] |

| estimator minus | within 10 % | within 20 % | median relative error |
|---|---|---|---|
| causal GP | +0.010 [-0.002, +0.021] | +0.010 [+0.003, +0.017] | -0.002 [-0.004, +0.001] |
| hold | +0.056 [+0.041, +0.069] | +0.064 [+0.053, +0.076] | -0.010 [-0.014, -0.007] |
| twin | +0.016 [+0.001, +0.029] | +0.017 [+0.010, +0.024] | -0.003 [-0.006, +0.000] |
| offline GP | -0.032 [-0.045, -0.020] | -0.019 [-0.027, -0.012] | +0.008 [+0.005, +0.011] |

### rows 1 to 3 after an absent frame

Rows per repeat (mean) 3,123 (share of the scored rows 0.0163); discharges with rows per repeat 470 to 475.

| estimator | within 10 % | within 20 % | median relative error |
|---|---|---|---|
| estimator | 0.394 [0.370, 0.418] | 0.626 [0.598, 0.657] | 0.136 [0.125, 0.148] |
| twin | 0.350 [0.322, 0.380] | 0.562 [0.529, 0.594] | 0.164 [0.147, 0.183] |
| causal GP | 0.348 [0.324, 0.370] | 0.559 [0.534, 0.584] | 0.167 [0.154, 0.180] |
| hold | 0.355 [0.331, 0.379] | 0.555 [0.530, 0.580] | 0.165 [0.153, 0.179] |
| offline GP | 0.456 [0.432, 0.480] | 0.664 [0.640, 0.689] | 0.116 [0.108, 0.127] |

| estimator minus | within 10 % | within 20 % | median relative error |
|---|---|---|---|
| causal GP | +0.046 [+0.021, +0.072] | +0.067 [+0.042, +0.092] | -0.031 [-0.043, -0.019] |
| hold | +0.039 [+0.013, +0.063] | +0.072 [+0.048, +0.095] | -0.029 [-0.041, -0.018] |
| twin | +0.044 [+0.014, +0.072] | +0.064 [+0.036, +0.091] | -0.028 [-0.044, -0.013] |
| offline GP | -0.062 [-0.086, -0.037] | -0.038 [-0.060, -0.015] | +0.020 [+0.010, +0.030] |

### rows not flagged by the value rules

Rows per repeat (mean) 170,440 (share of the scored rows 0.8873); discharges with rows per repeat 615 to 617.

| estimator | within 10 % | within 20 % | median relative error |
|---|---|---|---|
| estimator | 0.617 [0.596, 0.639] | 0.837 [0.820, 0.853] | 0.073 [0.069, 0.077] |
| twin | 0.612 [0.585, 0.633] | 0.828 [0.810, 0.843] | 0.074 [0.070, 0.079] |
| causal GP | 0.620 [0.601, 0.639] | 0.832 [0.817, 0.846] | 0.072 [0.068, 0.076] |
| hold | 0.594 [0.574, 0.614] | 0.810 [0.793, 0.825] | 0.076 [0.072, 0.081] |
| offline GP | 0.667 [0.647, 0.686] | 0.861 [0.846, 0.874] | 0.062 [0.059, 0.065] |

| estimator minus | within 10 % | within 20 % | median relative error |
|---|---|---|---|
| causal GP | -0.003 [-0.016, +0.008] | +0.005 [-0.004, +0.013] | +0.002 [-0.001, +0.004] |
| hold | +0.023 [+0.009, +0.035] | +0.027 [+0.017, +0.037] | -0.003 [-0.006, -0.000] |
| twin | +0.006 [-0.015, +0.026] | +0.009 [-0.006, +0.018] | -0.000 [-0.005, +0.003] |
| offline GP | -0.049 [-0.062, -0.038] | -0.024 [-0.033, -0.016] | +0.011 [+0.009, +0.014] |

### after the fact: rows with reported value >= 50 eV

Rows per repeat (mean) 189,472 (share of the scored rows 0.9864); discharges with rows per repeat 633 to 635.

| estimator | within 10 % | within 20 % | median relative error |
|---|---|---|---|
| estimator | 0.570 [0.544, 0.594] | 0.781 [0.758, 0.802] | 0.082 [0.077, 0.088] |
| twin | 0.564 [0.537, 0.589] | 0.771 [0.748, 0.792] | 0.083 [0.077, 0.090] |
| causal GP | 0.577 [0.554, 0.598] | 0.782 [0.762, 0.800] | 0.080 [0.075, 0.085] |
| hold | 0.551 [0.529, 0.572] | 0.758 [0.737, 0.778] | 0.086 [0.081, 0.091] |
| offline GP | 0.625 [0.602, 0.646] | 0.815 [0.795, 0.833] | 0.068 [0.065, 0.073] |

| estimator minus | within 10 % | within 20 % | median relative error |
|---|---|---|---|
| causal GP | -0.007 [-0.020, +0.003] | -0.002 [-0.011, +0.007] | +0.002 [-0.000, +0.005] |
| hold | +0.019 [+0.005, +0.030] | +0.022 [+0.012, +0.032] | -0.004 [-0.006, -0.000] |
| twin | +0.006 [-0.017, +0.025] | +0.009 [-0.008, +0.020] | -0.001 [-0.006, +0.004] |
| offline GP | -0.055 [-0.069, -0.044] | -0.035 [-0.044, -0.026] | +0.014 [+0.011, +0.017] |

## Expectations of the registration (section 6) beside the findings

| expectation | observed | range met | verdict | verdict met |
|---|---|---|---|---|
| within 20 %: estimator 0.74 to 0.80 | +0.7700 | yes |  |  |
| within 20 %: causal GP lower by 0.003 to 0.02 (T20 holds or fails by a small margin) | -0.0022 | no | does not hold |  |
| within 20 %: hold lower by 0.015 to 0.04; T20h holds | +0.0210 | yes | holds | yes |
| within 10 %: all five between 0.48 and 0.58 | estimator 0.5624, twin 0.5564, causal GP 0.5695, hold 0.5440, offline GP 0.6170 | no |  |  |
| within 10 %: estimator below the causal GP by 0.00 to 0.02 (difference -0.02 to 0); T10 holds | -0.0071 | yes | does not hold | no |
| median relative error (estimator) 0.085 to 0.10 | +0.0836 | no |  |  |
| median relative error 0.085 to 0.10 (all five, read alongside) | estimator 0.0836, twin 0.0846, causal GP 0.0814, hold 0.0874, offline GP 0.0698 | no |  |  |

## Choices made where the registration left room

- Rows, thirds (run_b44 code within a repeat), rows 1 to 3 after an absent frame (run_b41.part_b row walk) and rows not flagged (run_b41.plain_mask) are those of B.54 (data/.b54 caches).
- Shares are ratios of per-discharge sums (rows within the threshold over rows) under discharge-multiplicity weights; in a stratum, every discharge of the repeat is resampled and contributes only its rows in the stratum (as B.54's cells).
- 'Median relative error 0.085 to 0.10' in section 6 is read for the estimator; the five medians are listed beside it.
- 'Causal GP lower by 0.003 to 0.02' is read as estimator minus causal GP in [+0.003, +0.02]; 'the estimator below the causal GP by 0.00 to 0.02' as estimator minus causal GP in [-0.02, 0].
- Generators: numpy default_rng([20261154, crc32('b60|' + cell name)]) (run_b54.rng_for).

## Limits

The reported value carries its own scatter, so a share within 10 % of it is bounded by that scatter and is not a share within 10 % of the ion temperature. Low values of T_i make the relative error large for a fixed error in eV. The comparison with published real-time analyses is not like for like: their reference is an offline analysis of the same spectra, ours is the reported value of the frame itself.

Run time: 119 s.

## Addendum 1: the filters of B.56

Registered after the statements of B.60 and the skills of B.56 were read (Addendum 1 of the registration), before any relative error of a filter was computed. KF0 and KFX: the five-parameter robust filters of B.56 on log T_i (KF0 without fast channels, KFX with them; signed one-step outputs from `data/.b56/r{r}_f{k}.npz`). Rows: the scored rows of re-partitions r3 to r9 (the confirmation set); r2, the development set of the filters, is reported apart and not pooled. Every estimator, the recurrent estimator, the twin, the causal GP, hold and the offline GP included, is read on these same rows. Definitions, interval (2,000 draws, a re-partition then its discharges, differences within a draw) and strata are those of B.60 above.

### Check before the readout

- Rows: the B.56 cache and the B.54 row tables hold the same stored rows (discharge and row index) with the same reported values in 40 of 40 folds of r2 to r9 (largest value difference 0.0e+00 eV; rows per fold identical: yes).
- Median absolute error, mean over r3 to r9 (eV): KFX 41.83 against recorded 41.8 (equal); KF0 43.31 against recorded 43.3 (equal).
- The check passes.

### Statements (r3 to r9)

| statement | point [95 % interval] | reference | verdict |
|---|---|---|---|
| **F20**: share within 20 %, KFX minus causal GP; interval above 0 | +0.011 [+0.008, +0.014] | +0.00 | **holds** |
| **F10**: share within 10 %, KFX minus causal GP; interval above 0 | +0.014 [+0.010, +0.018] | +0.00 | **holds** |
| **F20x**: share within 20 %, KFX minus KF0; interval above 0 | +0.014 [+0.012, +0.017] | +0.00 | **holds** |
| **F20c**: coldest third, share within 20 %, KFX minus causal GP; interval above -0.02 | +0.012 [+0.002, +0.023] | -0.02 | **holds** |

### Shares and medians by stratum (r3 to r9)

#### all rows

Rows per repeat (mean) 192,102 (share 1.0000); discharges with rows per repeat 633 to 635.

| estimator | within 10 % | within 20 % | median relative error |
|---|---|---|---|
| KFX | 0.583 [0.561, 0.606] | 0.783 [0.763, 0.803] | 0.079 [0.074, 0.084] |
| KF0 | 0.566 [0.543, 0.588] | 0.769 [0.749, 0.789] | 0.082 [0.077, 0.088] |
| estimator | 0.564 [0.540, 0.587] | 0.771 [0.749, 0.792] | 0.083 [0.078, 0.089] |
| twin | 0.555 [0.528, 0.578] | 0.759 [0.736, 0.781] | 0.085 [0.079, 0.092] |
| causal GP | 0.570 [0.547, 0.591] | 0.772 [0.752, 0.791] | 0.081 [0.077, 0.087] |
| hold | 0.544 [0.522, 0.566] | 0.749 [0.727, 0.770] | 0.087 [0.082, 0.093] |
| offline GP | 0.617 [0.594, 0.639] | 0.805 [0.785, 0.824] | 0.070 [0.066, 0.075] |

| difference | within 10 % | within 20 % | median relative error |
|---|---|---|---|
| KFX minus causal GP | +0.014 [+0.010, +0.018] | +0.011 [+0.008, +0.014] | -0.003 [-0.004, -0.002] |
| KFX minus KF0 | +0.017 [+0.014, +0.021] | +0.014 [+0.012, +0.017] | -0.004 [-0.005, -0.003] |
| KFX minus hold | +0.039 [+0.034, +0.045] | +0.034 [+0.029, +0.039] | -0.009 [-0.011, -0.007] |
| KFX minus twin | +0.029 [+0.021, +0.046] | +0.024 [+0.017, +0.036] | -0.006 [-0.011, -0.004] |
| KFX minus estimator | +0.020 [+0.012, +0.029] | +0.012 [+0.005, +0.021] | -0.005 [-0.007, -0.003] |
| KFX minus offline GP | -0.034 [-0.039, -0.028] | -0.022 [-0.025, -0.019] | +0.009 [+0.007, +0.010] |
| KF0 minus causal GP | -0.004 [-0.007, -0.000] | -0.004 [-0.007, -0.000] | +0.001 [-0.000, +0.002] |
| estimator minus causal GP | -0.006 [-0.016, +0.003] | -0.001 [-0.010, +0.006] | +0.002 [-0.000, +0.004] |

#### low third by median T_i

Rows per repeat (mean) 39,387 (share 0.2050); discharges with rows per repeat 211 to 212.

| estimator | within 10 % | within 20 % | median relative error |
|---|---|---|---|
| KFX | 0.345 [0.306, 0.385] | 0.546 [0.495, 0.594] | 0.172 [0.147, 0.203] |
| KF0 | 0.346 [0.306, 0.387] | 0.542 [0.493, 0.592] | 0.174 [0.147, 0.206] |
| estimator | 0.301 [0.267, 0.337] | 0.499 [0.452, 0.546] | 0.201 [0.172, 0.235] |
| twin | 0.295 [0.259, 0.329] | 0.494 [0.450, 0.535] | 0.204 [0.177, 0.236] |
| causal GP | 0.342 [0.302, 0.383] | 0.534 [0.485, 0.582] | 0.177 [0.149, 0.211] |
| hold | 0.331 [0.292, 0.372] | 0.525 [0.473, 0.576] | 0.182 [0.153, 0.221] |
| offline GP | 0.386 [0.342, 0.432] | 0.575 [0.524, 0.625] | 0.152 [0.127, 0.183] |

| difference | within 10 % | within 20 % | median relative error |
|---|---|---|---|
| KFX minus causal GP | +0.003 [-0.006, +0.013] | +0.012 [+0.002, +0.023] | -0.005 [-0.013, +0.001] |
| KFX minus KF0 | -0.001 [-0.005, +0.003] | +0.003 [+0.000, +0.007] | -0.002 [-0.004, +0.000] |
| KFX minus hold | +0.015 [+0.007, +0.023] | +0.020 [+0.011, +0.029] | -0.010 [-0.019, -0.004] |
| KFX minus twin | +0.050 [+0.026, +0.089] | +0.052 [+0.029, +0.085] | -0.032 [-0.053, -0.018] |
| KFX minus estimator | +0.044 [+0.026, +0.065] | +0.046 [+0.025, +0.074] | -0.029 [-0.047, -0.017] |
| KFX minus offline GP | -0.041 [-0.054, -0.028] | -0.030 [-0.040, -0.018] | +0.020 [+0.013, +0.027] |
| KF0 minus causal GP | +0.004 [-0.005, +0.014] | +0.008 [-0.002, +0.020] | -0.003 [-0.011, +0.003] |
| estimator minus causal GP | -0.041 [-0.063, -0.022] | -0.035 [-0.062, -0.014] | +0.024 [+0.010, +0.042] |

#### middle third by median T_i

Rows per repeat (mean) 76,676 (share 0.3991); discharges with rows per repeat 211 to 212.

| estimator | within 10 % | within 20 % | median relative error |
|---|---|---|---|
| KFX | 0.677 [0.645, 0.707] | 0.851 [0.824, 0.876] | 0.061 [0.057, 0.066] |
| KF0 | 0.668 [0.635, 0.699] | 0.841 [0.812, 0.866] | 0.061 [0.056, 0.067] |
| estimator | 0.660 [0.626, 0.692] | 0.844 [0.816, 0.869] | 0.065 [0.060, 0.071] |
| twin | 0.659 [0.624, 0.691] | 0.836 [0.808, 0.862] | 0.064 [0.058, 0.070] |
| causal GP | 0.665 [0.632, 0.695] | 0.839 [0.811, 0.865] | 0.062 [0.058, 0.068] |
| hold | 0.652 [0.620, 0.681] | 0.839 [0.809, 0.865] | 0.066 [0.061, 0.071] |
| offline GP | 0.719 [0.686, 0.750] | 0.871 [0.845, 0.895] | 0.053 [0.049, 0.057] |

| difference | within 10 % | within 20 % | median relative error |
|---|---|---|---|
| KFX minus causal GP | +0.013 [+0.007, +0.018] | +0.012 [+0.008, +0.017] | -0.001 [-0.002, -0.000] |
| KFX minus KF0 | +0.009 [+0.004, +0.015] | +0.011 [+0.008, +0.013] | -0.000 [-0.002, +0.001] |
| KFX minus hold | +0.026 [+0.019, +0.032] | +0.012 [+0.008, +0.017] | -0.004 [-0.006, -0.003] |
| KFX minus twin | +0.018 [+0.009, +0.039] | +0.016 [+0.011, +0.026] | -0.002 [-0.007, -0.001] |
| KFX minus estimator | +0.017 [+0.006, +0.031] | +0.007 [+0.001, +0.014] | -0.004 [-0.007, -0.001] |
| KFX minus offline GP | -0.042 [-0.049, -0.034] | -0.020 [-0.025, -0.015] | +0.009 [+0.007, +0.010] |
| KF0 minus causal GP | +0.003 [-0.002, +0.009] | +0.002 [-0.003, +0.007] | -0.001 [-0.002, +0.000] |
| estimator minus causal GP | -0.004 [-0.019, +0.008] | +0.005 [-0.002, +0.011] | +0.003 [-0.000, +0.006] |

#### high third by median T_i

Rows per repeat (mean) 76,038 (share 0.3958); discharges with rows per repeat 211 to 211.

| estimator | within 10 % | within 20 % | median relative error |
|---|---|---|---|
| KFX | 0.612 [0.585, 0.638] | 0.838 [0.816, 0.857] | 0.074 [0.069, 0.080] |
| KF0 | 0.577 [0.549, 0.604] | 0.814 [0.791, 0.835] | 0.081 [0.075, 0.088] |
| estimator | 0.602 [0.574, 0.629] | 0.838 [0.817, 0.857] | 0.077 [0.072, 0.083] |
| twin | 0.584 [0.556, 0.612] | 0.820 [0.798, 0.840] | 0.080 [0.074, 0.086] |
| causal GP | 0.592 [0.565, 0.617] | 0.829 [0.808, 0.847] | 0.079 [0.073, 0.084] |
| hold | 0.546 [0.519, 0.572] | 0.774 [0.749, 0.798] | 0.087 [0.081, 0.094] |
| offline GP | 0.634 [0.607, 0.660] | 0.858 [0.837, 0.876] | 0.069 [0.064, 0.075] |

| difference | within 10 % | within 20 % | median relative error |
|---|---|---|---|
| KFX minus causal GP | +0.021 [+0.014, +0.027] | +0.009 [+0.006, +0.013] | -0.004 [-0.005, -0.003] |
| KFX minus KF0 | +0.035 [+0.029, +0.041] | +0.024 [+0.020, +0.028] | -0.007 [-0.008, -0.005] |
| KFX minus hold | +0.066 [+0.056, +0.076] | +0.064 [+0.055, +0.073] | -0.013 [-0.016, -0.010] |
| KFX minus twin | +0.028 [+0.018, +0.039] | +0.018 [+0.012, +0.025] | -0.006 [-0.008, -0.004] |
| KFX minus estimator | +0.010 [+0.001, +0.021] | -0.000 [-0.007, +0.006] | -0.002 [-0.005, -0.000] |
| KFX minus offline GP | -0.022 [-0.030, -0.014] | -0.020 [-0.024, -0.015] | +0.005 [+0.003, +0.007] |
| KF0 minus causal GP | -0.014 [-0.019, -0.010] | -0.015 [-0.019, -0.011] | +0.002 [+0.001, +0.004] |
| estimator minus causal GP | +0.011 [-0.002, +0.022] | +0.010 [+0.003, +0.017] | -0.002 [-0.004, +0.001] |

#### rows 1 to 3 after an absent frame

Rows per repeat (mean) 3,127 (share 0.0163); discharges with rows per repeat 470 to 475.

| estimator | within 10 % | within 20 % | median relative error |
|---|---|---|---|
| KFX | 0.372 [0.350, 0.394] | 0.602 [0.577, 0.627] | 0.148 [0.139, 0.159] |
| KF0 | 0.366 [0.342, 0.388] | 0.586 [0.560, 0.610] | 0.157 [0.146, 0.168] |
| estimator | 0.394 [0.370, 0.418] | 0.626 [0.598, 0.655] | 0.136 [0.126, 0.148] |
| twin | 0.348 [0.321, 0.373] | 0.560 [0.529, 0.589] | 0.165 [0.151, 0.183] |
| causal GP | 0.347 [0.323, 0.371] | 0.559 [0.534, 0.583] | 0.167 [0.154, 0.179] |
| hold | 0.354 [0.330, 0.378] | 0.554 [0.528, 0.579] | 0.165 [0.154, 0.181] |
| offline GP | 0.456 [0.432, 0.481] | 0.663 [0.638, 0.687] | 0.116 [0.107, 0.128] |

| difference | within 10 % | within 20 % | median relative error |
|---|---|---|---|
| KFX minus causal GP | +0.025 [+0.002, +0.046] | +0.044 [+0.027, +0.060] | -0.019 [-0.027, -0.010] |
| KFX minus KF0 | +0.006 [-0.012, +0.023] | +0.017 [+0.003, +0.029] | -0.009 [-0.014, -0.002] |
| KFX minus hold | +0.018 [-0.005, +0.039] | +0.048 [+0.030, +0.065] | -0.017 [-0.027, -0.008] |
| KFX minus twin | +0.024 [+0.004, +0.049] | +0.042 [+0.022, +0.069] | -0.017 [-0.032, -0.006] |
| KFX minus estimator | -0.022 [-0.046, +0.001] | -0.024 [-0.047, -0.003] | +0.012 [+0.004, +0.021] |
| KFX minus offline GP | -0.084 [-0.108, -0.062] | -0.061 [-0.079, -0.044] | +0.032 [+0.024, +0.040] |
| KF0 minus causal GP | +0.018 [+0.000, +0.037] | +0.027 [+0.012, +0.042] | -0.010 [-0.018, -0.003] |
| estimator minus causal GP | +0.047 [+0.022, +0.073] | +0.068 [+0.042, +0.093] | -0.031 [-0.043, -0.018] |

#### rows not flagged by the value rules

Rows per repeat (mean) 170,449 (share 0.8873); discharges with rows per repeat 615 to 616.

| estimator | within 10 % | within 20 % | median relative error |
|---|---|---|---|
| KFX | 0.637 [0.617, 0.655] | 0.845 [0.829, 0.859] | 0.069 [0.066, 0.073] |
| KF0 | 0.618 [0.598, 0.637] | 0.830 [0.814, 0.844] | 0.072 [0.068, 0.076] |
| estimator | 0.619 [0.596, 0.639] | 0.838 [0.820, 0.854] | 0.073 [0.069, 0.077] |
| twin | 0.610 [0.586, 0.631] | 0.827 [0.810, 0.842] | 0.074 [0.070, 0.079] |
| causal GP | 0.621 [0.600, 0.639] | 0.832 [0.816, 0.846] | 0.072 [0.068, 0.076] |
| hold | 0.594 [0.574, 0.613] | 0.810 [0.793, 0.825] | 0.076 [0.072, 0.081] |
| offline GP | 0.667 [0.646, 0.686] | 0.861 [0.846, 0.875] | 0.062 [0.059, 0.066] |

| difference | within 10 % | within 20 % | median relative error |
|---|---|---|---|
| KFX minus causal GP | +0.017 [+0.013, +0.021] | +0.013 [+0.010, +0.016] | -0.003 [-0.004, -0.002] |
| KFX minus KF0 | +0.019 [+0.015, +0.023] | +0.015 [+0.012, +0.017] | -0.003 [-0.004, -0.002] |
| KFX minus hold | +0.043 [+0.038, +0.049] | +0.035 [+0.030, +0.041] | -0.007 [-0.009, -0.006] |
| KFX minus twin | +0.027 [+0.019, +0.044] | +0.018 [+0.013, +0.028] | -0.005 [-0.008, -0.003] |
| KFX minus estimator | +0.018 [+0.010, +0.028] | +0.007 [+0.000, +0.014] | -0.004 [-0.006, -0.002] |
| KFX minus offline GP | -0.030 [-0.035, -0.024] | -0.016 [-0.019, -0.013] | +0.007 [+0.006, +0.008] |
| KF0 minus causal GP | -0.002 [-0.006, +0.001] | -0.002 [-0.005, +0.002] | +0.000 [-0.001, +0.001] |
| estimator minus causal GP | -0.002 [-0.012, +0.008] | +0.006 [-0.001, +0.013] | +0.001 [-0.001, +0.003] |

#### rows with reported value below 50 eV (after the fact)

Rows per repeat (mean) 2,609 (share 0.0136); discharges with rows per repeat 342 to 350.

| estimator | within 10 % | within 20 % | median relative error |
|---|---|---|---|
| KFX | 0.029 [0.022, 0.038] | 0.064 [0.052, 0.077] | 3.690 [2.973, 4.422] |
| KF0 | 0.029 [0.021, 0.037] | 0.063 [0.052, 0.077] | 3.682 [2.926, 4.503] |
| estimator | 0.001 [0.000, 0.004] | 0.001 [0.000, 0.006] | 6.714 [5.920, 7.598] |
| twin | 0.000 [0.000, 0.001] | 0.000 [0.000, 0.002] | 6.684 [5.885, 7.498] |
| causal GP | 0.025 [0.018, 0.034] | 0.047 [0.035, 0.059] | 5.496 [4.532, 6.437] |
| hold | 0.035 [0.028, 0.044] | 0.074 [0.061, 0.089] | 2.719 [2.056, 3.704] |
| offline GP | 0.035 [0.026, 0.046] | 0.071 [0.056, 0.089] | 4.958 [3.811, 5.869] |

| difference | within 10 % | within 20 % | median relative error |
|---|---|---|---|
| KFX minus causal GP | +0.004 [-0.005, +0.013] | +0.017 [+0.006, +0.029] | -1.806 [-2.206, -1.393] |
| KFX minus KF0 | +0.001 [-0.006, +0.007] | +0.000 [-0.005, +0.006] | +0.008 [-0.131, +0.100] |
| KFX minus hold | -0.006 [-0.016, +0.004] | -0.010 [-0.023, +0.003] | +0.971 [+0.591, +1.208] |
| KFX minus twin | +0.029 [+0.022, +0.038] | +0.064 [+0.052, +0.077] | -2.994 [-3.559, -2.515] |
| KFX minus estimator | +0.029 [+0.021, +0.038] | +0.063 [+0.051, +0.076] | -3.024 [-3.626, -2.592] |
| KFX minus offline GP | -0.006 [-0.017, +0.004] | -0.007 [-0.023, +0.008] | -1.268 [-1.730, -0.750] |
| KF0 minus causal GP | +0.003 [-0.006, +0.012] | +0.017 [+0.006, +0.028] | -1.814 [-2.211, -1.380] |
| estimator minus causal GP | -0.025 [-0.033, -0.017] | -0.046 [-0.058, -0.034] | +1.217 [+0.712, +1.877] |

#### rows with reported value >= 50 eV (after the fact)

Rows per repeat (mean) 189,492 (share 0.9864); discharges with rows per repeat 633 to 635.

| estimator | within 10 % | within 20 % | median relative error |
|---|---|---|---|
| KFX | 0.591 [0.570, 0.613] | 0.793 [0.775, 0.812] | 0.077 [0.073, 0.082] |
| KF0 | 0.573 [0.553, 0.596] | 0.778 [0.760, 0.798] | 0.081 [0.076, 0.086] |
| estimator | 0.572 [0.549, 0.595] | 0.782 [0.761, 0.802] | 0.082 [0.077, 0.087] |
| twin | 0.563 [0.538, 0.586] | 0.770 [0.749, 0.790] | 0.083 [0.078, 0.090] |
| causal GP | 0.577 [0.556, 0.599] | 0.782 [0.764, 0.801] | 0.080 [0.075, 0.085] |
| hold | 0.551 [0.531, 0.573] | 0.758 [0.739, 0.779] | 0.086 [0.080, 0.091] |
| offline GP | 0.625 [0.604, 0.647] | 0.815 [0.797, 0.834] | 0.068 [0.064, 0.073] |

| difference | within 10 % | within 20 % | median relative error |
|---|---|---|---|
| KFX minus causal GP | +0.014 [+0.010, +0.018] | +0.011 [+0.008, +0.014] | -0.003 [-0.004, -0.002] |
| KFX minus KF0 | +0.018 [+0.014, +0.021] | +0.015 [+0.012, +0.017] | -0.004 [-0.005, -0.003] |
| KFX minus hold | +0.040 [+0.035, +0.045] | +0.035 [+0.030, +0.040] | -0.009 [-0.010, -0.007] |
| KFX minus twin | +0.028 [+0.021, +0.046] | +0.023 [+0.017, +0.036] | -0.006 [-0.011, -0.004] |
| KFX minus estimator | +0.020 [+0.012, +0.029] | +0.012 [+0.004, +0.020] | -0.005 [-0.007, -0.003] |
| KFX minus offline GP | -0.034 [-0.039, -0.029] | -0.022 [-0.025, -0.019] | +0.009 [+0.007, +0.010] |
| KF0 minus causal GP | -0.004 [-0.007, -0.000] | -0.004 [-0.007, -0.000] | +0.001 [-0.000, +0.002] |
| estimator minus causal GP | -0.006 [-0.016, +0.003] | -0.001 [-0.009, +0.007] | +0.002 [-0.000, +0.004] |

### Rows with a reported value below 50 eV (r3 to r9)

18,266 of 1,344,711 rows (0.0136); 0 at or below zero. Definition applied unchanged.

| estimator | within 10 % | within 20 % | median relative error |
|---|---|---|---|
| KFX | 0.0292 | 0.0639 | 3.686 |
| KF0 | 0.0285 | 0.0635 | 3.678 |
| estimator | 0.0005 | 0.0010 | 6.724 |
| twin | 0.0002 | 0.0003 | 6.685 |
| causal GP | 0.0251 | 0.0468 | 5.490 |
| hold | 0.0353 | 0.0743 | 2.694 |
| offline GP | 0.0351 | 0.0710 | 4.962 |

### r2, the development set (reported apart)

| cell | KFX within 20 % | causal GP within 20 % | KFX minus causal GP, 20 % | KFX minus causal GP, 10 % | KFX minus KF0, 20 % |
|---|---|---|---|---|---|
| all rows | 0.782 [0.761, 0.801] | 0.772 [0.751, 0.790] | +0.010 [+0.007, +0.013] | +0.014 [+0.010, +0.018] | +0.014 [+0.012, +0.016] |
| low third by median T_i | 0.540 [0.491, 0.587] | 0.530 [0.482, 0.578] | +0.010 [+0.000, +0.021] | +0.003 [-0.005, +0.012] | +0.003 [+0.000, +0.006] |

(r2 alone: the interval resamples its discharges only.)

### Expectations of Addendum 1 beside the findings

| expectation | observed | range met | verdict | verdict met |
|---|---|---|---|---|
| KFX within 20 % 0.775 to 0.80 | +0.7832 | yes |  |  |
| KFX above the causal GP within 20 % by +0.003 to +0.02 (F20 holds or fails by a small margin) | +0.0110 | yes | holds |  |
| KFX within 10 % 0.575 to 0.61 | +0.5835 | yes |  |  |
| KFX above the causal GP within 10 % by +0.005 to +0.03; F10 holds | +0.0139 | yes | holds | yes |
| F20x +0.003 to +0.015, holds | +0.0145 | yes | holds | yes |
| F20c -0.01 to +0.01, holds | +0.0115 | no | holds | yes |

Choices: the strata are those of B.60 (B.54 definitions: thirds within a re-partition by median T_i, rows 1 to 3 after an absent frame by the row walk of run_b41.part_b, rows not flagged by run_b41.plain_mask), not the ten-row after-blip class of B.56. The cell of rows below 50 eV and the cell at or above 50 eV are reported (the first is listed in the addendum; the second is after the fact). Generators: run_b54.rng_for('b60a1|' + cell name).

Run time of the addendum stage: 279 s.
