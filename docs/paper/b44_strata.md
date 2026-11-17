# B.44 -- in which discharges the margins hold: strata of the test discharges (registered: ces_prediction/experiments/PREREGISTRATION_B44.md)

CPU, nothing trained. T_i, cut population, natural cadence (k = 1), frozen B.30 row tables. Skill of the causal variant over the history-only twin, the causal GP and hold inside thirds of the test discharges of each split. Intervals: 95 %, 2,000 resamples of the discharges of the cell. A statement holds if it holds on at least three of the four splits.

## Registered statements

- **W1** (every third of the median T_i: causal variant over the twin, interval above zero): 1 of 4 splits -> **does not hold**. split 42: low +0.010 [-0.007, +0.031]; middle +0.057 [-0.005, +0.120]; high +0.090 [+0.007, +0.159] | split 1: low +0.007 [-0.031, +0.034]; middle +0.012 [-0.030, +0.057]; high +0.064 [+0.025, +0.107] | split 7: low +0.091 [+0.069, +0.112]; middle +0.113 [+0.063, +0.165]; high +0.150 [+0.084, +0.197] | split 123: low +0.025 [+0.012, +0.040]; middle +0.103 [-0.014, +0.237]; high +0.076 [+0.011, +0.130].
- **W2** (every third of the median T_i: causal variant over the causal GP, interval above zero): 2 of 4 splits -> **does not hold**. split 42: low +0.077 [+0.030, +0.115]; middle +0.066 [+0.011, +0.119]; high +0.054 [-0.074, +0.157] | split 1: low +0.106 [+0.050, +0.146]; middle +0.080 [+0.028, +0.132]; high +0.105 [+0.070, +0.150] | split 7: low +0.098 [+0.053, +0.135]; middle +0.143 [+0.084, +0.206]; high +0.167 [+0.069, +0.236] | split 123: low +0.094 [+0.062, +0.132]; middle +0.115 [+0.018, +0.185]; high +0.088 [-0.014, +0.162].

## Section 5 rule: strata whose highest-minus-lowest difference excludes zero on at least three splits

| stratum | vs twin | vs causal GP | vs hold |
|---|---|---|---|
| median measured T_i (eV) | 1 of 4 (no) | 0 of 4 (no) | 2 of 4 (no) |
| robust white component / median T_i | 0 of 4 (no) | 0 of 4 (no) | 4 of 4 (condition) |
| share of rows with a BES channel > +3 SD above its 100 ms mean | 0 of 4 (no) | 1 of 4 (no) | 4 of 4 (condition) |
| absent frames per second of measured time | 1 of 4 (no) | 0 of 4 (no) | 1 of 4 (no) |
| shot number | 1 of 4 (no) | 0 of 4 (no) | 2 of 4 (no) |

## Check before the readout

| split | rows (B.44 population) | discharges | skill vs twin (all thirds) | vs causal GP | vs hold |
|---|---|---|---|---|---|
| 42 | 32,921 | 96 | +0.055 [+0.017, +0.092] | +0.065 [+0.013, +0.109] | +0.353 [+0.306, +0.392] |
| 1 | 35,936 | 96 | +0.028 [+0.005, +0.049] | +0.096 [+0.068, +0.122] | +0.385 [+0.353, +0.412] |
| 7 | 36,196 | 96 | +0.113 [+0.088, +0.138] | +0.130 [+0.094, +0.165] | +0.399 [+0.366, +0.426] |
| 123 | 34,428 | 96 | +0.057 [+0.022, +0.095] | +0.095 [+0.057, +0.134] | +0.407 [+0.365, +0.437] |

Same estimators on the B.30 population (has_future, PCHIP and every network finite) against docs/paper/b30_thinned.json:

| split | rows (mine / B.30) | causal vs GP (mine / B.30) | causal vs hold (mine / B.30) | causal vs twin (mine / from B.30 RMSE) |
|---|---|---|---|---|
| 42 | 32,818 / 32,818 | +0.0677 / +0.0677 | +0.3518 / +0.3518 | +0.0555 / +0.0555 |
| 1 | 35,831 / 35,831 | +0.0957 / +0.0957 | +0.3853 / +0.3853 | +0.0274 / +0.0274 |
| 7 | 36,094 / 36,094 | +0.1259 / +0.1259 | +0.3974 / +0.3974 | +0.1146 / +0.1146 |
| 123 | 34,327 / 34,327 | +0.0959 / +0.0959 | +0.4079 / +0.4079 | +0.0565 / +0.0565 |

BES read from the CSV against the grid: files with a different row count 0, largest time difference 9.46e-07 s. Sum of the rows of the three thirds equals the population on every split and stratum: True.

## Operational choices made before the run (where the registration leaves room)

- Population: rows of the frozen table with k = 1 and hold, causal GP, twin (hist80), causal variant and the measurement finite (as B.41); the discharges of a split are the test discharges that have at least one such row.
- All five quantities come from the cut-population grid (seq_data.load_grid_files with held values and T_i above 3 keV set to NaN) with the first and last row of the file dropped; the BES departures are read by posthoc_b32.channel_departures from the same CSV (first and last row dropped there too).
- Median T_i: median over the measured (finite) T_i rows of the discharge.
- White component: B.40 triples (three consecutive rows with T_i measured and both steps 10 ms within 0.1 ms); sigma = 1.4826 MAD(second difference) / sqrt(6), divided by the median T_i. A discharge with fewer than 30 triples has no value and is left out of this stratum only (fixed before the run; listed below).
- BES share: rows with a finite departure (ten earlier rows in the same segment) and not within ten rows after an absent frame (the denominator of B.32 B'); a row counts when at least one of the nine channels departs by more than +3 (per-discharge standard deviations) from the mean of the ten rows before it.
- Absent frames: a step of more than 15 and less than 45 ms between grid rows holds round(step / 10 ms) - 1 absent frames; measured time = sum of the grid steps shorter than 0.5 s.
- Thirds: rank-based within each split over the discharges with a value, sorted by (value, shot number) so ties go by shot number; sizes differ by at most one (np.array_split, larger thirds first). Edges = the largest value of the low and of the middle third.
- Skill = 1 - SSE(causal variant) / SSE(reference) over the rows of the cell. Intervals: 2,000 multinomial resamples of the discharges of that third; the highest-minus-lowest interval is the 2.5 and 97.5 % quantiles of the difference of independent resamples of the two thirds. One generator (seed 20261118) in a fixed order.
- W1 (W2) holds on a split when the lower ends of all three intervals of the median-T_i thirds against the twin (causal GP) are above zero; it holds when that is so on at least three of the four splits.
- Section 5 rule: a stratum is reported as a condition on which a margin depends when its highest-minus-lowest interval excludes zero on at least three splits, read for each of the three references.

## Tables by stratum

### median measured T_i (eV)

| split | third | discharges | rows | range | vs twin | vs causal GP | vs hold |
|---|---|---|---|---|---|---|---|
| 42 | low | 32 | 6,115 | 54 to 380 | +0.010 [-0.007, +0.031] | +0.077 [+0.030, +0.115] | +0.420 [+0.389, +0.447] |
| 42 | middle | 32 | 12,457 | 391 to 698 | +0.057 [-0.005, +0.120] | +0.066 [+0.011, +0.119] | +0.281 [+0.214, +0.334] |
| 42 | high | 32 | 14,349 | 709 to 1532 | +0.090 [+0.007, +0.159] | +0.054 [-0.074, +0.157] | +0.332 [+0.217, +0.407] |
| 42 | high minus low | | | edges 380, 698 | +0.080 [-0.007, +0.153] | -0.022 [-0.153, +0.090] | -0.088 [-0.207, -0.008] |
| 1 | low | 32 | 8,055 | 62 to 336 | +0.007 [-0.031, +0.034] | +0.106 [+0.050, +0.146] | +0.397 [+0.327, +0.436] |
| 1 | middle | 32 | 14,029 | 341 to 685 | +0.012 [-0.030, +0.057] | +0.080 [+0.028, +0.132] | +0.331 [+0.274, +0.374] |
| 1 | high | 32 | 13,852 | 687 to 1383 | +0.064 [+0.025, +0.107] | +0.105 [+0.070, +0.150] | +0.430 [+0.388, +0.459] |
| 1 | high minus low | | | edges 336, 685 | +0.057 [+0.008, +0.113] | -0.000 [-0.056, +0.070] | +0.033 [-0.023, +0.111] |
| 7 | low | 32 | 8,491 | 85 to 360 | +0.091 [+0.069, +0.112] | +0.098 [+0.053, +0.135] | +0.411 [+0.368, +0.444] |
| 7 | middle | 32 | 12,856 | 371 to 649 | +0.113 [+0.063, +0.165] | +0.143 [+0.084, +0.206] | +0.365 [+0.285, +0.426] |
| 7 | high | 32 | 14,849 | 682 to 1315 | +0.150 [+0.084, +0.197] | +0.167 [+0.069, +0.236] | +0.414 [+0.336, +0.463] |
| 7 | high minus low | | | edges 360, 649 | +0.060 [-0.006, +0.112] | +0.069 [-0.036, +0.150] | +0.002 [-0.081, +0.069] |
| 123 | low | 32 | 7,334 | 69 to 335 | +0.025 [+0.012, +0.040] | +0.094 [+0.062, +0.132] | +0.451 [+0.421, +0.472] |
| 123 | middle | 32 | 14,796 | 336 to 673 | +0.103 [-0.014, +0.237] | +0.115 [+0.018, +0.185] | +0.335 [+0.228, +0.391] |
| 123 | high | 32 | 12,298 | 676 to 1523 | +0.076 [+0.011, +0.130] | +0.088 [-0.014, +0.162] | +0.367 [+0.261, +0.437] |
| 123 | high minus low | | | edges 335, 673 | +0.051 [-0.016, +0.110] | -0.007 [-0.112, +0.077] | -0.084 [-0.194, -0.010] |

### robust white component / median T_i

| split | third | discharges | rows | range | vs twin | vs causal GP | vs hold |
|---|---|---|---|---|---|---|---|
| 42 | low | 28 | 13,378 | 0.028 to 0.061 | +0.064 [-0.029, +0.146] | +0.079 [-0.117, +0.226] | +0.230 [+0.005, +0.363] |
| 42 | middle | 28 | 10,274 | 0.061 to 0.097 | +0.059 [-0.076, +0.158] | +0.025 [-0.140, +0.148] | +0.295 [+0.190, +0.373] |
| 42 | high | 28 | 8,993 | 0.097 to 1.375 | +0.055 [+0.017, +0.096] | +0.072 [+0.038, +0.104] | +0.395 [+0.357, +0.430] |
| 42 | high minus low | | | edges 0.061, 0.097 | -0.009 [-0.102, +0.093] | -0.007 [-0.164, +0.191] | +0.165 [+0.024, +0.399] |
| 1 | low | 30 | 13,072 | 0.025 to 0.073 | -0.029 [-0.093, +0.033] | +0.049 [-0.076, +0.163] | +0.231 [+0.112, +0.312] |
| 1 | middle | 29 | 12,121 | 0.075 to 0.124 | +0.065 [+0.010, +0.119] | +0.125 [+0.081, +0.173] | +0.351 [+0.292, +0.390] |
| 1 | high | 29 | 10,576 | 0.125 to 1.462 | +0.022 [+0.001, +0.040] | +0.087 [+0.060, +0.112] | +0.431 [+0.397, +0.458] |
| 1 | high minus low | | | edges 0.073, 0.124 | +0.050 [-0.015, +0.117] | +0.038 [-0.080, +0.163] | +0.200 [+0.113, +0.319] |
| 7 | low | 31 | 14,383 | 0.023 to 0.068 | +0.109 [+0.020, +0.197] | +0.127 [-0.003, +0.247] | +0.207 [+0.107, +0.296] |
| 7 | middle | 31 | 11,850 | 0.068 to 0.114 | +0.112 [+0.070, +0.147] | +0.111 [+0.050, +0.170] | +0.381 [+0.341, +0.414] |
| 7 | high | 31 | 9,944 | 0.118 to 2.236 | +0.115 [+0.083, +0.148] | +0.138 [+0.095, +0.184] | +0.436 [+0.399, +0.467] |
| 7 | high minus low | | | edges 0.068, 0.114 | +0.005 [-0.089, +0.097] | +0.012 [-0.118, +0.150] | +0.229 [+0.135, +0.339] |
| 123 | low | 30 | 14,878 | 0.026 to 0.055 | +0.089 [-0.099, +0.261] | -0.036 [-0.225, +0.154] | +0.080 [-0.135, +0.265] |
| 123 | middle | 30 | 10,161 | 0.056 to 0.111 | +0.041 [+0.010, +0.079] | +0.122 [+0.068, +0.171] | +0.371 [+0.305, +0.419] |
| 123 | high | 30 | 9,251 | 0.120 to 3.047 | +0.056 [+0.025, +0.092] | +0.106 [+0.068, +0.147] | +0.452 [+0.422, +0.474] |
| 123 | high minus low | | | edges 0.055, 0.111 | -0.033 [-0.207, +0.159] | +0.142 [-0.048, +0.335] | +0.372 [+0.184, +0.584] |

Discharges without a value (left out of this stratum): split 42: s30801.csv, s31160.csv, s31334.csv, s31341.csv, s31365.csv, s32073.csv, s32141.csv, s32143.csv, s32161.csv, s32441.csv, s32443.csv, s32449.csv; split 1: s31341.csv, s31430.csv, s31482.csv, s31484.csv, s32152.csv, s32437.csv, s32441.csv, s32612.csv; split 7: s31767.csv, s32443.csv, s32445.csv; split 123: s30801.csv, s31161.csv, s31340.csv, s31496.csv, s32442.csv, s32449.csv.

### share of rows with a BES channel > +3 SD above its 100 ms mean

| split | third | discharges | rows | range | vs twin | vs causal GP | vs hold |
|---|---|---|---|---|---|---|---|
| 42 | low | 31 | 11,449 | 0.000 to 0.000 | +0.108 [-0.000, +0.213] | +0.022 [-0.160, +0.160] | +0.242 [+0.034, +0.365] |
| 42 | middle | 30 | 13,830 | 0.000 to 0.004 | +0.051 [-0.026, +0.107] | +0.068 [-0.033, +0.156] | +0.350 [+0.274, +0.406] |
| 42 | high | 30 | 7,602 | 0.005 to 0.058 | +0.037 [-0.005, +0.087] | +0.079 [+0.041, +0.115] | +0.388 [+0.342, +0.428] |
| 42 | high minus low | | | edges 0.000, 0.004 | -0.071 [-0.188, +0.046] | +0.057 [-0.090, +0.241] | +0.146 [+0.009, +0.353] |
| 1 | low | 31 | 12,200 | 0.000 to 0.001 | +0.012 [-0.056, +0.081] | +0.035 [-0.082, +0.129] | +0.301 [+0.150, +0.380] |
| 1 | middle | 31 | 15,688 | 0.001 to 0.007 | +0.047 [+0.005, +0.092] | +0.077 [+0.043, +0.108] | +0.399 [+0.355, +0.433] |
| 1 | high | 31 | 8,038 | 0.008 to 0.105 | +0.019 [-0.008, +0.048] | +0.128 [+0.094, +0.162] | +0.401 [+0.355, +0.437] |
| 1 | high minus low | | | edges 0.001, 0.007 | +0.007 [-0.066, +0.080] | +0.093 [-0.007, +0.215] | +0.100 [+0.008, +0.255] |
| 7 | low | 31 | 13,687 | 0.000 to 0.000 | +0.134 [+0.071, +0.205] | +0.190 [+0.116, +0.273] | +0.248 [+0.176, +0.312] |
| 7 | middle | 31 | 13,712 | 0.000 to 0.005 | +0.106 [+0.059, +0.152] | +0.121 [+0.045, +0.186] | +0.412 [+0.357, +0.456] |
| 7 | high | 31 | 8,778 | 0.005 to 0.034 | +0.112 [+0.085, +0.137] | +0.114 [+0.080, +0.150] | +0.428 [+0.397, +0.456] |
| 7 | high minus low | | | edges 0.000, 0.005 | -0.022 [-0.101, +0.048] | -0.076 [-0.165, +0.010] | +0.180 [+0.112, +0.257] |
| 123 | low | 31 | 12,546 | 0.000 to 0.002 | +0.001 [-0.071, +0.091] | -0.015 [-0.145, +0.090] | +0.243 [+0.048, +0.350] |
| 123 | middle | 31 | 12,969 | 0.002 to 0.007 | +0.105 [+0.036, +0.175] | +0.136 [+0.072, +0.190] | +0.383 [+0.332, +0.425] |
| 123 | high | 31 | 8,895 | 0.007 to 0.051 | +0.050 [+0.022, +0.084] | +0.114 [+0.081, +0.155] | +0.473 [+0.447, +0.491] |
| 123 | high minus low | | | edges 0.002, 0.007 | +0.049 [-0.047, +0.129] | +0.130 [+0.021, +0.260] | +0.231 [+0.117, +0.424] |

Ties across an edge (discharges with the edge value on both sides, assigned by shot number): split 42 low/middle edge 0.000: 39 discharges; split 7 low/middle edge 0.000: 39 discharges.

Discharges without a value (left out of this stratum): split 42: s32073.csv, s32141.csv, s32143.csv, s32443.csv, s32449.csv; split 1: s31482.csv, s31484.csv, s32437.csv; split 7: s31767.csv, s32443.csv, s32445.csv; split 123: s31496.csv, s32442.csv, s32449.csv.

### absent frames per second of measured time

| split | third | discharges | rows | range | vs twin | vs causal GP | vs hold |
|---|---|---|---|---|---|---|---|
| 42 | low | 32 | 3,787 | 0.00 to 0.34 | -0.028 [-0.167, +0.053] | +0.055 [-0.096, +0.138] | +0.395 [+0.269, +0.460] |
| 42 | middle | 32 | 14,497 | 0.35 to 0.55 | +0.058 [+0.017, +0.114] | +0.074 [+0.027, +0.115] | +0.364 [+0.317, +0.401] |
| 42 | high | 32 | 14,637 | 0.56 to 2.99 | +0.090 [+0.030, +0.143] | +0.060 [-0.043, +0.159] | +0.316 [+0.216, +0.393] |
| 42 | high minus low | | | edges 0.34, 0.55 | +0.118 [+0.006, +0.261] | +0.005 [-0.145, +0.177] | -0.079 [-0.201, +0.054] |
| 1 | low | 32 | 6,474 | 0.00 to 0.40 | +0.018 [-0.008, +0.043] | +0.090 [+0.026, +0.140] | +0.428 [+0.357, +0.472] |
| 1 | middle | 32 | 15,891 | 0.45 to 0.62 | +0.047 [+0.003, +0.089] | +0.109 [+0.074, +0.151] | +0.351 [+0.283, +0.398] |
| 1 | high | 32 | 13,571 | 0.62 to 2.53 | +0.011 [-0.029, +0.052] | +0.083 [+0.033, +0.133] | +0.387 [+0.347, +0.420] |
| 1 | high minus low | | | edges 0.40, 0.62 | -0.007 [-0.054, +0.041] | -0.007 [-0.082, +0.069] | -0.041 [-0.103, +0.035] |
| 7 | low | 32 | 7,705 | 0.00 to 0.43 | +0.105 [+0.065, +0.138] | +0.096 [+0.046, +0.148] | +0.414 [+0.353, +0.457] |
| 7 | middle | 32 | 15,440 | 0.45 to 0.62 | +0.100 [+0.072, +0.130] | +0.139 [+0.088, +0.189] | +0.379 [+0.336, +0.408] |
| 7 | high | 32 | 13,051 | 0.62 to 4.46 | +0.136 [+0.073, +0.186] | +0.151 [+0.068, +0.214] | +0.406 [+0.327, +0.462] |
| 7 | high minus low | | | edges 0.43, 0.62 | +0.031 [-0.042, +0.093] | +0.056 [-0.044, +0.135] | -0.008 [-0.098, +0.072] |
| 123 | low | 32 | 6,662 | 0.00 to 0.40 | +0.048 [+0.023, +0.078] | +0.095 [+0.063, +0.140] | +0.475 [+0.446, +0.495] |
| 123 | middle | 32 | 15,233 | 0.43 to 0.59 | +0.038 [-0.024, +0.126] | +0.111 [+0.052, +0.165] | +0.378 [+0.295, +0.427] |
| 123 | high | 32 | 12,533 | 0.60 to 4.46 | +0.083 [+0.018, +0.143] | +0.078 [-0.026, +0.162] | +0.357 [+0.237, +0.425] |
| 123 | high minus low | | | edges 0.40, 0.59 | +0.035 [-0.038, +0.100] | -0.016 [-0.130, +0.073] | -0.118 [-0.241, -0.042] |

Ties across an edge (discharges with the edge value on both sides, assigned by shot number): split 1 middle/high edge 0.62: 4 discharges; split 7 middle/high edge 0.62: 3 discharges.

### shot number

| split | third | discharges | rows | range | vs twin | vs causal GP | vs hold |
|---|---|---|---|---|---|---|---|
| 42 | low | 32 | 10,806 | 30801 to 31549 | +0.094 [+0.044, +0.149] | +0.066 [-0.015, +0.145] | +0.385 [+0.308, +0.438] |
| 42 | middle | 32 | 15,230 | 31553 to 31988 | +0.097 [+0.005, +0.164] | +0.053 [-0.079, +0.135] | +0.297 [+0.148, +0.404] |
| 42 | high | 32 | 6,885 | 32011 to 32751 | +0.009 [-0.047, +0.056] | +0.071 [-0.003, +0.137] | +0.363 [+0.310, +0.403] |
| 42 | high minus low | | | edges 31549, 31988 | -0.085 [-0.161, -0.017] | +0.005 [-0.101, +0.111] | -0.022 [-0.097, +0.061] |
| 1 | low | 32 | 10,561 | 30811 to 31482 | +0.009 [-0.034, +0.050] | +0.057 [+0.002, +0.123] | +0.366 [+0.301, +0.411] |
| 1 | middle | 32 | 15,636 | 31484 to 32030 | +0.041 [-0.005, +0.091] | +0.093 [+0.037, +0.137] | +0.383 [+0.300, +0.433] |
| 1 | high | 32 | 9,739 | 32035 to 32751 | +0.025 [-0.009, +0.056] | +0.120 [+0.083, +0.153] | +0.398 [+0.354, +0.433] |
| 1 | high minus low | | | edges 31482, 32030 | +0.016 [-0.036, +0.068] | +0.063 [-0.015, +0.127] | +0.032 [-0.030, +0.107] |
| 7 | low | 32 | 13,267 | 31024 to 31450 | +0.105 [+0.056, +0.170] | +0.169 [+0.104, +0.244] | +0.308 [+0.220, +0.367] |
| 7 | middle | 32 | 14,256 | 31499 to 31978 | +0.134 [+0.088, +0.178] | +0.146 [+0.081, +0.207] | +0.420 [+0.374, +0.458] |
| 7 | high | 32 | 8,673 | 31980 to 32751 | +0.099 [+0.062, +0.129] | +0.095 [+0.044, +0.142] | +0.417 [+0.367, +0.456] |
| 7 | high minus low | | | edges 31450, 31978 | -0.006 [-0.080, +0.052] | -0.073 [-0.160, +0.009] | +0.109 [+0.029, +0.208] |
| 123 | low | 32 | 13,007 | 30801 to 31512 | -0.004 [-0.078, +0.049] | +0.025 [-0.096, +0.136] | +0.310 [+0.145, +0.392] |
| 123 | middle | 32 | 12,793 | 31513 to 32018 | +0.093 [+0.027, +0.153] | +0.120 [+0.034, +0.186] | +0.397 [+0.293, +0.461] |
| 123 | high | 32 | 8,628 | 32023 to 32751 | +0.054 [+0.019, +0.105] | +0.103 [+0.076, +0.138] | +0.442 [+0.410, +0.462] |
| 123 | high minus low | | | edges 31512, 32018 | +0.058 [-0.011, +0.152] | +0.078 [-0.038, +0.202] | +0.132 [+0.044, +0.296] |

## After the fact (not registered; not used for any verdict)

Rank correlation (Spearman) of each quantity with the median T_i over the test discharges of the split, to read how far the strata are separate conditions:

| stratum | split 42 | split 1 | split 7 | split 123 |
|---|---|---|---|---|
| robust white component / median T_i | -0.30 | -0.16 | -0.32 | -0.33 |
| share of rows with a BES channel > +3 SD above its 100 ms mean | -0.19 | -0.13 | -0.32 | -0.32 |
| absent frames per second of measured time | +0.42 | +0.32 | +0.25 | +0.35 |
| shot number | -0.22 | -0.03 | -0.15 | -0.08 |

The BES share is zero on many discharges, so its low third (and part of the middle third) is a set of zero-share discharges divided by shot number. Read once more after the fact: discharges with no upward departure against those with at least one (own generator, the registered numbers above are unchanged):

| split | group | discharges | rows | vs twin | vs causal GP | vs hold |
|---|---|---|---|---|---|---|
| 42 | zero | 39 | 13,805 | +0.073 [-0.060, +0.182] | +0.030 [-0.172, +0.189] | +0.247 [+0.073, +0.370] |
| 42 | positive | 52 | 19,076 | +0.049 [+0.021, +0.084] | +0.075 [+0.043, +0.108] | +0.380 [+0.344, +0.413] |
| 42 | positive minus zero | | | -0.024 [-0.139, +0.116] | +0.046 [-0.114, +0.251] | +0.132 [+0.003, +0.316] |
| 1 | zero | 30 | 10,722 | +0.020 [-0.051, +0.089] | +0.070 [-0.008, +0.150] | +0.333 [+0.225, +0.394] |
| 1 | positive | 63 | 25,204 | +0.029 [+0.005, +0.052] | +0.101 [+0.072, +0.129] | +0.394 [+0.359, +0.423] |
| 1 | positive minus zero | | | +0.009 [-0.065, +0.084] | +0.031 [-0.051, +0.116] | +0.061 [-0.008, +0.176] |
| 7 | zero | 39 | 15,967 | +0.117 [+0.057, +0.186] | +0.163 [+0.079, +0.239] | +0.254 [+0.183, +0.319] |
| 7 | positive | 54 | 20,210 | +0.112 [+0.084, +0.140] | +0.122 [+0.081, +0.163] | +0.426 [+0.395, +0.453] |
| 7 | positive minus zero | | | -0.005 [-0.076, +0.062] | -0.041 [-0.126, +0.052] | +0.172 [+0.100, +0.248] |
| 123 | zero | 28 | 9,752 | +0.005 [-0.086, +0.109] | -0.033 [-0.183, +0.088] | +0.224 [-0.016, +0.351] |
| 123 | positive | 65 | 24,658 | +0.068 [+0.035, +0.108] | +0.121 [+0.091, +0.153] | +0.438 [+0.408, +0.458] |
| 123 | positive minus zero | | | +0.063 [-0.041, +0.163] | +0.154 [+0.026, +0.306] | +0.214 [+0.083, +0.447] |

## Expectations of the registration (section 6)

| expectation | observed | met |
|---|---|---|
| W1 does not hold | W1 does not hold (1 of 4) | yes |
| lowest third of median T_i: increment over the twin not separated from zero on two or more splits | 2 of 4 splits | yes |
| lowest third: increment +0.00 to +0.06 (each split) | +0.0100, +0.0072, +0.0908, +0.0250 | no |
| highest third: increment +0.05 to +0.15 (each split) | +0.0896, +0.0641, +0.1505, +0.0763 | no |
| W2 holds on two or three splits | 2 of 4 splits | yes |
| increment over the twin grows with the BES upward-departure share (high minus low > 0 on at least three splits) | points -0.071, +0.007, -0.022, +0.049; interval above 0 on 0 of 4 | no |
| increment over the twin does not depend on the shot number (high minus low interval contains 0 on at least three splits) | 3 of 4 contain 0; points -0.085, +0.016, -0.006, +0.058 | yes |

4 of 7 expectations met.

## Deviations

None from the registered definitions, thresholds or populations. The registration leaves the treatment of ties, of discharges without a value and the exact counting of absent frames open; the choices above were fixed before the run.

## Limits

About 32 discharges per third of a split, so the intervals are wide. The strata are computed from the same files as the scores. They are not confinement modes.

Run time: 17 s.
