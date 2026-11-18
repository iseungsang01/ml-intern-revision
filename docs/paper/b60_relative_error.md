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
