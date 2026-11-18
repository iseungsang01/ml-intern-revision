# B.52 -- readouts asked for by the eleventh independent grading (registered: ces_prediction/experiments/PREREGISTRATION_B52.md)

CPU; no network trained or run (part L fits ridge-penalised least squares). T_i, cut population, the four file-level splits (42, 1, 7, 123), test discharges. Rows: the k = 1 rows of the B.48 row table in which the estimator of B.39, the twin, hold and the causal GP are finite. E / E_h: the average of the three initialisations of the estimator / of the twin (B.49 cache). Skill of A over B = 1 - SSE_A / SSE_B on the same rows. Intervals: 95 %, 2,000 resamples of the test discharges. A statement holds on a split when its interval is above zero (D1, D2: above 0.5), and holds when that is so on at least three of four splits.

## Checks before the readout

| split | scored rows; grid T_i vs y: max abs diff (eV), rows > 1e-3; files outside test | estimator over twin / GP / hold (mine = B.48) | E (Addendum 1): B.49 rows joined / all; scored rows with E; y max diff (eV); E over GP with the stored GP errors = B.49 | part L: design test rows; joined; y max diff; v1 vs hold max diff (eV), rows > 1e-3 |
|---|---|---|---|---|
| 42 | 32,921; 0.0e+00, 0; 0 | +0.0649 = +0.0649 / +0.0747 = +0.0747 / +0.3596 = +0.3596 | 32,589 / 32,589; 32,589; 0.0e+00; +0.0780 = +0.0780 | 32,120; 32,120; 0.0e+00; 0.0e+00, 0 |
| 1 | 35,936; 0.0e+00, 0; 0 | +0.0277 = +0.0277 / +0.0961 = +0.0961 / +0.3851 = +0.3851 | 35,582 / 35,582; 35,582; 0.0e+00; +0.1053 = +0.1053 | 35,134; 35,134; 0.0e+00; 0.0e+00, 0 |
| 7 | 36,196; 0.0e+00, 0; 0 | +0.1160 = +0.1160 / +0.1324 = +0.1324 / +0.4006 = +0.4006 | 35,759 / 35,759; 35,759; 0.0e+00; +0.1533 = +0.1533 | 35,357; 35,357; 0.0e+00; 0.0e+00, 0 |
| 123 | 34,428; 0.0e+00, 0; 0 | +0.0568 = +0.0568 / +0.0953 = +0.0953 / +0.4067 = +0.4067 | 34,038 / 34,038; 34,038; 0.0e+00; +0.1149 = +0.1149 | 33,657; 33,657; 0.0e+00; 0.0e+00, 0 |

## The two computations of the causal GP (reported, Addendum 1)

B.49 used the causal-GP errors stored by `eval_seq`; every readout here uses the causal GP of the B.48 row table. Rows with E joined; errors against y of the table.

| split | E over causal GP: stored GP (check) | table GP (used in the readouts) | B.49 E1 | rows whose absolute GP error differs by > 1e-3 eV (> 0.5 eV) | largest difference (eV) | SSE GP stored / table | discharges | B.48 check (b): share of stored rows with the GP within 0.5 eV (rows not agreeing) |
|---|---|---|---|---|---|---|---|---|
| 42 | +0.07798 | +0.07709 | +0.07798 | 16 (15) | 653.2 | 1.00096 | s31409.csv, s32377.csv | 0.99926 (24) |
| 1 | +0.10528 | +0.10528 | +0.10528 | 0 (0) | 0.0 | 1.00000 | - | 0.99978 (8) |
| 7 | +0.15326 | +0.15280 | +0.15326 | 15 (15) | 357.3 | 1.00054 | s31978.csv | 0.99941 (21) |
| 123 | +0.11488 | +0.11488 | +0.11488 | 0 (0) | 0.0 | 1.00000 | - | 0.99976 (8) |

The rows that differ:

| split | file | grid row | row in its span | visible values before it in the span | age (ms) | GP of the table (eV) | gp_unscaled of the table | abs error stored / table (eV) |
|---|---|---|---|---|---|---|---|---|
| 42 | s31409.csv | 54 | 53 | 3 | 10 | 18.6 | nan | 4.0 / 5.9 |
| 42 | s31409.csv | 55 | 54 | 4 | 10 | 12.2 | nan | 6.5 / 8.7 |
| 42 | s31409.csv | 56 | 55 | 5 | 10 | 26.0 | nan | 17.8 / 9.2 |
| 42 | s31409.csv | 57 | 56 | 6 | 10 | 42.8 | nan | 6.5 / 7.5 |
| 42 | s31409.csv | 58 | 57 | 7 | 10 | 33.7 | nan | 10.0 / 7.6 |
| 42 | s31409.csv | 59 | 58 | 8 | 10 | 41.8 | nan | 7.0 / 4.6 |
| 42 | s31409.csv | 60 | 59 | 9 | 10 | 47.6 | nan | 3.8 / 0.6 |
| 42 | s31409.csv | 61 | 60 | 10 | 10 | 46.0 | nan | 10.4 / 13.6 |
| 42 | s31409.csv | 62 | 61 | 11 | 10 | 28.3 | nan | 9.0 / 7.8 |
| 42 | s31409.csv | 63 | 62 | 12 | 10 | 38.8 | nan | 19.3 / 20.9 |
| 42 | s31409.csv | 64 | 63 | 13 | 10 | 20.2 | nan | 1.4 / 1.8 |
| 42 | s31409.csv | 65 | 64 | 14 | 10 | 19.1 | nan | 26.4 / 29.3 |
| 42 | s31409.csv | 66 | 65 | 15 | 10 | 48.2 | nan | 0.9 / 5.3 |
| 42 | s31409.csv | 67 | 66 | 16 | 10 | 40.4 | nan | 6.6 / 1.9 |
| 42 | s31409.csv | 68 | 67 | 17 | 10 | 37.8 | nan | 6.6 / 7.3 |
| 42 | s32377.csv | 283 | 282 | 168 | 70 | 324.3 | nan | 974.4 / 321.1 |
| 7 | s31978.csv | 272 | 271 | 210 | 30 | 480.5 | nan | 194.7 / 330.4 |
| 7 | s31978.csv | 274 | 273 | 211 | 20 | 474.7 | nan | 750.4 / 522.3 |
| 7 | s31978.csv | 275 | 274 | 212 | 10 | 556.2 | nan | 512.8 / 523.3 |
| 7 | s31978.csv | 276 | 275 | 213 | 10 | 604.5 | nan | 853.0 / 495.7 |
| 7 | s31978.csv | 277 | 276 | 214 | 10 | 482.7 | nan | 145.1 / 132.7 |
| 7 | s31978.csv | 278 | 277 | 215 | 10 | 574.9 | nan | 500.9 / 513.7 |
| 7 | s31978.csv | 279 | 278 | 216 | 10 | 445.7 | nan | 522.7 / 501.4 |
| 7 | s31978.csv | 280 | 279 | 217 | 10 | 606.6 | nan | 208.4 / 196.9 |
| 7 | s31978.csv | 281 | 280 | 218 | 10 | 594.8 | nan | 48.8 / 49.7 |
| 7 | s31978.csv | 282 | 281 | 219 | 10 | 569.8 | nan | 285.5 / 276.5 |
| 7 | s31978.csv | 283 | 282 | 220 | 10 | 534.8 | nan | 49.2 / 40.2 |
| 7 | s31978.csv | 284 | 283 | 221 | 10 | 502.5 | nan | 253.6 / 336.3 |
| 7 | s31978.csv | 285 | 284 | 222 | 10 | 565.9 | nan | 36.8 / 109.6 |
| 7 | s31978.csv | 286 | 285 | 223 | 10 | 560.8 | nan | 215.9 / 157.5 |
| 7 | s31978.csv | 287 | 286 | 224 | 10 | 549.8 | nan | 20.2 / 3.7 |

Why they differ (checked within the ten minutes allowed, not a readout): the GP of the table is reproduced exactly by `baselines_interpolation.predict_gp_causal` on the neighbours of the B.48 machinery (the last 16 visible values in the span), with the times in float32 or in float64 from the CSV alike, so time precision is not the cause, and `gp_unscaled` is not involved (it is NaN at k = 1). The differing rows come in runs of consecutive rows of one discharge (rows 54 to 68 of s31409 in split 42, rows 272 to 287 of s31978 in split 7) plus one isolated row (s32377, split 42). Each run is 15 to 16 rows long, as long as one visible value stays among the 16 nearest past values that enter the fit; this points to one neighbour value that differs between the neighbour set of eval_seq (`build_neighbor_set` on the dataset array) and that of the row table. Which value differs was not identified. It is of the same size as the disagreement recorded by check (b) of B.48 (column above).

All checks pass on every split. Leakage: the history inputs index the T_i column only at visible rows strictly before the row (asserted in the code), and the first history input equals `hold` of the table on every joined row (column above). Part P cross-check against b48_reread.json (b41 G percentiles of the estimator, twin, causal GP, hold): largest |difference| 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00 eV.

## Registered statements

Per-split values in the order 42, 1, 7, 123.

- **D1** (share of the test discharges with >= 50 scored rows in which the estimator has a lower MSE than the causal GP; interval above 0.5): 0.647 [0.541, 0.753] (n = 85); 0.727 [0.636, 0.818] (n = 88); 0.731 [0.634, 0.817] (n = 93); 0.725 [0.637, 0.813] (n = 91); 4 of 4 -> **holds**.
- **D2** (the same against the twin): 0.706 [0.612, 0.800]; 0.614 [0.511, 0.716]; 0.849 [0.763, 0.914]; 0.637 [0.538, 0.736]; 4 of 4 -> **holds**.
- **L1** (estimator over the linear filter LF): +0.109 [+0.074, +0.143]; +0.075 [+0.052, +0.098]; +0.119 [+0.092, +0.142]; +0.106 [+0.070, +0.135]; 4 of 4 -> **holds**.
- **L2** (LF over LH): +0.055 [+0.027, +0.094]; +0.047 [+0.032, +0.061]; +0.046 [+0.027, +0.066]; +0.032 [+0.015, +0.049]; 4 of 4 -> **holds**.
- **G1** (rows whose last visible value is 20 ms old or older, estimator over the causal GP): +0.056 [-0.074, +0.177]; +0.089 [+0.026, +0.153]; +0.109 [+0.040, +0.185]; +0.105 [+0.049, +0.178]; 3 of 4 -> **holds**.
- **V1** (T = 2,000 eV, estimator over the causal GP): +0.109 [+0.058, +0.156]; +0.091 [+0.058, +0.125]; +0.137 [+0.096, +0.177]; +0.118 [+0.077, +0.157]; 4 of 4 -> **holds**.
- **V2** (T = 2,000 eV, estimator over the twin): +0.089 [+0.047, +0.127]; +0.043 [+0.017, +0.071]; +0.140 [+0.107, +0.170]; +0.082 [+0.046, +0.120]; 4 of 4 -> **holds**.

Outcome under section 5 of the registration: D1 holds: the share is stated in the main result. L1 holds: the recurrent estimator is separated from the linear filter with the fast channels. L2 holds: quoted as support for the fast-channel increment from a model with no early stopping and no initialisation.

## Part D -- per discharge

Eligible discharges: at least 50 scored rows (for E: at least 50 scored rows with E). Shares with the 95 % interval over resamples of the eligible discharges; 'reduction' = SSE of the reference minus SSE of the estimator per discharge, over all scored discharges.

| split | estimator | reference | discharges (eligible / all) | share lower MSE | share lower median abs error | per-discharge skill: quartiles | top ten share of the reduction | discharges with a positive reduction |
|---|---|---|---|---|---|---|---|---|
| 42 | estimator | causal GP | 85 / 96 | 0.647 [0.541, 0.753] | 0.612 [0.506, 0.706] | -0.035, +0.079, +0.196 | 0.748 | 59 |
| 42 | estimator | twin | 85 / 96 | 0.706 [0.612, 0.800] | 0.765 [0.671, 0.847] | -0.017, +0.062, +0.200 | 0.808 | 68 |
| 42 | estimator | hold | 85 / 96 | 0.882 [0.812, 0.941] | 0.635 [0.529, 0.741] | +0.195, +0.345, +0.433 | 0.395 | 79 |
| 42 | E | causal GP | 84 / 96 | 0.714 [0.607, 0.810] | 0.631 [0.524, 0.738] | -0.022, +0.086, +0.191 | 0.777 | 64 |
| 42 | E | twin | 84 / 96 | 0.750 [0.655, 0.845] | 0.726 [0.631, 0.821] | +0.003, +0.073, +0.223 | 0.829 | 70 |
| 42 | E | E_h | 84 / 96 | 0.762 [0.667, 0.857] | 0.583 [0.476, 0.690] | +0.008, +0.065, +0.207 | 0.854 | 73 |
| 42 | E | hold | 84 / 96 | 0.881 [0.810, 0.940] | 0.667 [0.560, 0.762] | +0.174, +0.348, +0.439 | 0.416 | 78 |
| 1 | estimator | causal GP | 88 / 96 | 0.727 [0.636, 0.818] | 0.398 [0.295, 0.500] | -0.012, +0.073, +0.175 | 0.470 | 69 |
| 1 | estimator | twin | 88 / 96 | 0.614 [0.511, 0.716] | 0.432 [0.330, 0.534] | -0.064, +0.023, +0.077 | 1.053 | 56 |
| 1 | estimator | hold | 88 / 96 | 0.852 [0.773, 0.920] | 0.477 [0.375, 0.580] | +0.211, +0.342, +0.426 | 0.420 | 80 |
| 1 | E | causal GP | 88 / 96 | 0.761 [0.670, 0.852] | 0.466 [0.364, 0.568] | +0.015, +0.089, +0.184 | 0.465 | 71 |
| 1 | E | twin | 88 / 96 | 0.614 [0.511, 0.716] | 0.466 [0.364, 0.580] | -0.038, +0.022, +0.082 | 0.887 | 58 |
| 1 | E | E_h | 88 / 96 | 0.648 [0.545, 0.750] | 0.432 [0.330, 0.534] | -0.020, +0.024, +0.095 | 0.815 | 59 |
| 1 | E | hold | 88 / 96 | 0.886 [0.818, 0.943] | 0.523 [0.420, 0.625] | +0.210, +0.338, +0.435 | 0.404 | 84 |
| 7 | estimator | causal GP | 93 / 96 | 0.731 [0.634, 0.817] | 0.581 [0.484, 0.677] | -0.003, +0.097, +0.243 | 0.525 | 69 |
| 7 | estimator | twin | 93 / 96 | 0.849 [0.763, 0.914] | 0.677 [0.581, 0.763] | +0.032, +0.105, +0.187 | 0.516 | 82 |
| 7 | estimator | hold | 93 / 96 | 0.882 [0.817, 0.946] | 0.602 [0.505, 0.699] | +0.203, +0.343, +0.442 | 0.482 | 83 |
| 7 | E | causal GP | 93 / 96 | 0.774 [0.688, 0.860] | 0.667 [0.570, 0.763] | +0.022, +0.116, +0.256 | 0.507 | 73 |
| 7 | E | twin | 93 / 96 | 0.849 [0.774, 0.925] | 0.699 [0.613, 0.785] | +0.050, +0.139, +0.234 | 0.482 | 82 |
| 7 | E | E_h | 93 / 96 | 0.796 [0.710, 0.871] | 0.699 [0.613, 0.796] | +0.007, +0.078, +0.173 | 0.589 | 77 |
| 7 | E | hold | 93 / 96 | 0.903 [0.839, 0.957] | 0.688 [0.602, 0.785] | +0.231, +0.347, +0.440 | 0.493 | 85 |
| 123 | estimator | causal GP | 91 / 96 | 0.725 [0.637, 0.813] | 0.549 [0.440, 0.659] | -0.011, +0.073, +0.195 | 0.620 | 68 |
| 123 | estimator | twin | 91 / 96 | 0.637 [0.538, 0.736] | 0.615 [0.516, 0.714] | -0.025, +0.034, +0.114 | 0.884 | 62 |
| 123 | estimator | hold | 91 / 96 | 0.879 [0.813, 0.945] | 0.549 [0.440, 0.648] | +0.193, +0.333, +0.432 | 0.520 | 83 |
| 123 | E | causal GP | 91 / 96 | 0.780 [0.703, 0.868] | 0.593 [0.495, 0.692] | +0.017, +0.133, +0.235 | 0.606 | 73 |
| 123 | E | twin | 91 / 96 | 0.780 [0.692, 0.857] | 0.703 [0.604, 0.791] | +0.004, +0.054, +0.192 | 0.834 | 74 |
| 123 | E | E_h | 91 / 96 | 0.725 [0.626, 0.813] | 0.681 [0.582, 0.769] | -0.004, +0.048, +0.180 | 0.809 | 69 |
| 123 | E | hold | 91 / 96 | 0.890 [0.824, 0.945] | 0.659 [0.560, 0.748] | +0.227, +0.353, +0.455 | 0.514 | 83 |

## Part L -- linear filters fitted on the training discharges

| split | train / val rows (discharges) | test rows scored (share of scored; discharges) | lambda LH / LF | estimator over LF (L1) | LF over LH (L2) | LH over causal GP | LF over causal GP | LH over hold | LF over hold |
|---|---|---|---|---|---|---|---|---|---|
| 42 | 137,853 (394) / 49,826 (124) | 32,120 (0.976; 91) | 0 / 0 | +0.109 [+0.074, +0.143] | +0.055 [+0.027, +0.094] | -0.098 [-0.162, -0.045] | -0.038 [-0.083, +0.003] | +0.243 [+0.188, +0.286] | +0.284 [+0.243, +0.319] |
| 1 | 137,758 (392) / 46,907 (124) | 35,134 (0.978; 93) | 0 / 0.001 | +0.075 [+0.052, +0.098] | +0.047 [+0.032, +0.061] | -0.032 [-0.061, -0.003] | +0.017 [-0.011, +0.042] | +0.302 [+0.273, +0.325] | +0.335 [+0.306, +0.358] |
| 7 | 139,380 (392) / 45,062 (124) | 35,357 (0.977; 93) | 0.01 / 0.001 | +0.119 [+0.092, +0.142] | +0.046 [+0.027, +0.066] | -0.034 [-0.065, -0.004] | +0.013 [-0.025, +0.055] | +0.290 [+0.259, +0.312] | +0.322 [+0.292, +0.347] |
| 123 | 139,932 (390) / 46,210 (126) | 33,657 (0.978; 93) | 0.01 / 0 | +0.106 [+0.070, +0.135] | +0.032 [+0.015, +0.049] | -0.051 [-0.083, -0.018] | -0.017 [-0.057, +0.025] | +0.314 [+0.284, +0.340] | +0.336 [+0.306, +0.361] |

| split | twin over LH | twin over LF | E over LF (rows) | estimator over LH | estimator over causal GP, same rows | RMSE (eV): estimator / twin / GP / hold / LH / LF |
|---|---|---|---|---|---|---|
| 42 | +0.096 [+0.052, +0.144] | +0.044 [+0.001, +0.080] | +0.112 [+0.072, +0.149] (31,801) | +0.158 [+0.112, +0.209] | +0.075 [+0.031, +0.117] | 163.5 / 169.4 / 170.0 / 204.6 / 178.1 / 173.2 |
| 1 | +0.094 [+0.067, +0.117] | +0.049 [+0.020, +0.074] | +0.084 [+0.062, +0.105] (34,783) | +0.118 [+0.094, +0.143] | +0.091 [+0.062, +0.117] | 200.2 / 203.0 / 209.9 / 255.3 / 213.2 / 208.2 |
| 7 | +0.049 [+0.034, +0.063] | +0.003 [-0.024, +0.028] | +0.139 [+0.114, +0.163] (34,934) | +0.160 [+0.130, +0.188] | +0.131 [+0.095, +0.168] | 181.4 / 192.9 / 194.6 / 234.7 / 197.8 / 193.3 |
| 123 | +0.084 [+0.042, +0.115] | +0.054 [+0.005, +0.093] | +0.125 [+0.075, +0.160] (33,281) | +0.135 [+0.097, +0.164] | +0.091 [+0.049, +0.129] | 208.1 / 214.1 / 218.3 / 270.2 / 223.7 / 220.1 |

Validation MSE (standardized) by lambda, LH then LF: split 42: 0.0: 0.24466, 0.001: 0.24466, 0.01: 0.24470, 0.1: 0.24598, 1.0: 0.29593, 10.0: 0.63918 | 0.0: 0.23199, 0.001: 0.23234, 0.01: 0.23353, 0.1: 0.23579, 1.0: 0.28864, 10.0: 0.61833; split 1: 0.0: 0.23742, 0.001: 0.23743, 0.01: 0.23758, 0.1: 0.24014, 1.0: 0.30712, 10.0: 0.73418 | 0.0: 0.22575, 0.001: 0.22567, 0.01: 0.22615, 0.1: 0.22902, 1.0: 0.29583, 10.0: 0.70895; split 7: 0.0: 0.28399, 0.001: 0.28398, 0.01: 0.28393, 0.1: 0.28462, 1.0: 0.33457, 10.0: 0.70267 | 0.0: 0.27515, 0.001: 0.27508, 0.01: 0.27562, 0.1: 0.27651, 1.0: 0.32593, 10.0: 0.67808; split 123: 0.0: 0.27674, 0.001: 0.27674, 0.01: 0.27673, 0.1: 0.27763, 1.0: 0.32752, 10.0: 0.70385 | 0.0: 0.26284, 0.001: 0.26316, 0.01: 0.26412, 0.1: 0.26597, 1.0: 0.32059, 10.0: 0.68351.

## Part G -- by the age of the last visible value

| split | bin | rows (share) | discharges | over causal GP | over twin | over hold | twin over causal GP |
|---|---|---|---|---|---|---|---|
| 42 | 10 ms | 31,986 (0.9716) | 96 | +0.080 [+0.026, +0.124] | +0.071 [+0.033, +0.106] | +0.365 [+0.312, +0.410] | +0.010 [-0.032, +0.048] |
| 42 | 20 to 50 ms | 790 (0.0240) | 80 | +0.072 [-0.099, +0.220] | +0.060 [-0.014, +0.160] | +0.322 [+0.196, +0.439] | +0.013 [-0.154, +0.129] |
| 42 | above 50 ms | 145 (0.0044) | 42 | +0.004 [-0.215, +0.188] | -0.012 [-0.095, +0.062] | +0.383 [+0.213, +0.528] | +0.015 [-0.170, +0.189] |
| 42 | 20 ms or more | 935 (0.0284) | 86 | +0.056 [-0.074, +0.177] | +0.043 [-0.015, +0.121] | +0.339 [+0.250, +0.430] | +0.013 [-0.118, +0.116] |
| 1 | 10 ms | 34,608 (0.9630) | 96 | +0.098 [+0.065, +0.127] | +0.016 [-0.008, +0.038] | +0.384 [+0.345, +0.416] | +0.083 [+0.051, +0.113] |
| 1 | 20 to 50 ms | 1,201 (0.0334) | 87 | +0.107 [+0.044, +0.182] | +0.092 [+0.048, +0.146] | +0.403 [+0.309, +0.496] | +0.017 [-0.045, +0.079] |
| 1 | above 50 ms | 127 (0.0035) | 42 | +0.007 [-0.318, +0.202] | -0.032 [-0.202, +0.102] | +0.333 [+0.192, +0.446] | +0.038 [-0.200, +0.223] |
| 1 | 20 ms or more | 1,328 (0.0370) | 90 | +0.089 [+0.026, +0.153] | +0.070 [+0.024, +0.125] | +0.390 [+0.314, +0.469] | +0.021 [-0.040, +0.080] |
| 7 | 10 ms | 34,944 (0.9654) | 96 | +0.139 [+0.101, +0.176] | +0.119 [+0.088, +0.151] | +0.411 [+0.373, +0.443] | +0.022 [-0.004, +0.046] |
| 7 | 20 to 50 ms | 1,137 (0.0314) | 87 | +0.100 [+0.029, +0.185] | +0.109 [+0.055, +0.161] | +0.387 [+0.305, +0.455] | -0.010 [-0.081, +0.080] |
| 7 | above 50 ms | 115 (0.0032) | 44 | +0.144 [-0.057, +0.350] | +0.085 [-0.056, +0.220] | +0.241 [+0.067, +0.395] | +0.065 [-0.131, +0.253] |
| 7 | 20 ms or more | 1,252 (0.0346) | 92 | +0.109 [+0.040, +0.185] | +0.104 [+0.047, +0.156] | +0.363 [+0.288, +0.422] | +0.005 [-0.069, +0.089] |
| 123 | 10 ms | 33,008 (0.9588) | 96 | +0.093 [+0.046, +0.131] | +0.060 [+0.025, +0.096] | +0.406 [+0.357, +0.444] | +0.035 [-0.006, +0.069] |
| 123 | 20 to 50 ms | 1,303 (0.0378) | 86 | +0.145 [+0.076, +0.238] | +0.087 [+0.044, +0.150] | +0.430 [+0.344, +0.513] | +0.063 [+0.003, +0.140] |
| 123 | above 50 ms | 117 (0.0034) | 43 | -0.093 [-0.352, +0.161] | -0.150 [-0.329, -0.020] | +0.315 [+0.178, +0.450] | +0.049 [-0.148, +0.232] |
| 123 | 20 ms or more | 1,420 (0.0412) | 89 | +0.105 [+0.049, +0.178] | +0.046 [-0.002, +0.100] | +0.410 [+0.342, +0.473] | +0.061 [+0.003, +0.128] |

## Part V -- value threshold (reported value and last visible value at or below T)

| split | T (eV) | rows (share) | discharges | over causal GP | over twin | over hold |
|---|---|---|---|---|---|---|
| 42 | all rows | 32,921 (1.0000) | 96 | +0.075 [+0.028, +0.116] | +0.065 [+0.031, +0.100] | +0.360 [+0.316, +0.396] |
| 42 | 2000 | 32,688 (0.9929) | 96 | +0.109 [+0.058, +0.156] | +0.089 [+0.047, +0.127] | +0.352 [+0.311, +0.386] |
| 42 | 2500 | 32,830 (0.9972) | 96 | +0.096 [+0.057, +0.136] | +0.081 [+0.044, +0.116] | +0.368 [+0.336, +0.395] |
| 1 | all rows | 35,936 (1.0000) | 96 | +0.096 [+0.068, +0.123] | +0.028 [+0.006, +0.050] | +0.385 [+0.352, +0.411] |
| 1 | 2000 | 35,619 (0.9912) | 96 | +0.091 [+0.058, +0.125] | +0.043 [+0.017, +0.071] | +0.360 [+0.326, +0.390] |
| 1 | 2500 | 35,811 (0.9965) | 96 | +0.085 [+0.056, +0.113] | +0.036 [+0.012, +0.061] | +0.378 [+0.347, +0.406] |
| 7 | all rows | 36,196 (1.0000) | 96 | +0.132 [+0.096, +0.170] | +0.116 [+0.091, +0.142] | +0.401 [+0.366, +0.429] |
| 7 | 2000 | 35,922 (0.9924) | 96 | +0.137 [+0.096, +0.177] | +0.140 [+0.107, +0.170] | +0.366 [+0.325, +0.404] |
| 7 | 2500 | 36,104 (0.9975) | 96 | +0.140 [+0.100, +0.182] | +0.134 [+0.104, +0.164] | +0.384 [+0.345, +0.418] |
| 123 | all rows | 34,428 (1.0000) | 96 | +0.095 [+0.059, +0.131] | +0.057 [+0.025, +0.093] | +0.407 [+0.366, +0.436] |
| 123 | 2000 | 33,990 (0.9873) | 96 | +0.118 [+0.077, +0.157] | +0.082 [+0.046, +0.120] | +0.374 [+0.330, +0.409] |
| 123 | 2500 | 34,250 (0.9948) | 96 | +0.114 [+0.076, +0.150] | +0.076 [+0.044, +0.110] | +0.399 [+0.360, +0.428] |

## Part P -- percentiles of the absolute error (eV)

On every scored row (estimator, twin, causal GP, hold) and, separately, on the scored rows that carry E.

| split | rows | estimate | 50th | 75th | 90th | 95th | 99th |
|---|---|---|---|---|---|---|---|
| 42 | all scored | estimator | 41.9 [36.7, 48.6] | 90.6 [76.7, 108.3] | 181.2 [154.3, 213.4] | 268.8 [236.8, 302.9] | 595.8 [491.7, 736.8] |
| 42 | all scored | twin | 46.0 [40.4, 53.4] | 97.5 [83.4, 115.7] | 192.3 [164.4, 226.2] | 283.6 [246.7, 319.4] | 628.0 [517.2, 751.9] |
| 42 | all scored | causal GP | 42.7 [37.2, 50.1] | 97.2 [81.7, 114.6] | 193.5 [164.2, 224.5] | 282.0 [247.4, 319.2] | 633.4 [528.2, 780.5] |
| 42 | all scored | hold | 44.2 [37.9, 52.1] | 105.0 [86.3, 129.5] | 228.7 [186.0, 274.2] | 338.7 [287.8, 386.3] | 798.9 [647.3, 1040.0] |
| 42 | with E | E | 41.8 [36.7, 48.2] | 89.2 [75.8, 106.1] | 176.5 [151.1, 206.2] | 262.2 [229.3, 294.5] | 576.0 [473.2, 685.7] |
| 42 | with E | E_h | 43.8 [38.2, 51.3] | 93.6 [80.4, 110.7] | 188.7 [160.2, 220.7] | 278.8 [243.5, 310.1] | 606.8 [498.6, 736.3] |
| 42 | with E | estimator | 41.4 [36.4, 47.9] | 88.7 [75.6, 105.6] | 176.9 [150.5, 206.6] | 260.9 [229.0, 291.7] | 570.1 [470.8, 681.3] |
| 42 | with E | twin | 45.5 [40.0, 52.8] | 95.9 [82.2, 113.6] | 188.7 [160.7, 221.4] | 278.0 [241.3, 309.4] | 588.5 [495.4, 711.0] |
| 42 | with E | causal GP | 42.2 [36.7, 49.5] | 95.6 [80.5, 112.8] | 188.4 [160.0, 219.9] | 273.5 [240.7, 310.3] | 607.4 [509.6, 747.0] |
| 42 | with E | hold | 43.8 [37.7, 51.7] | 103.6 [85.5, 128.3] | 225.4 [182.9, 269.9] | 332.2 [283.1, 378.5] | 759.2 [620.6, 974.4] |
| 1 | all scored | estimator | 56.2 [46.6, 67.3] | 117.7 [99.2, 138.5] | 226.7 [190.7, 265.3] | 340.3 [288.9, 391.6] | 790.8 [674.3, 942.1] |
| 1 | all scored | twin | 50.5 [40.8, 62.5] | 119.6 [98.0, 142.7] | 239.2 [201.9, 279.1] | 350.2 [303.5, 407.7] | 812.3 [664.4, 959.7] |
| 1 | all scored | causal GP | 48.7 [38.5, 61.4] | 121.6 [98.4, 145.9] | 246.3 [205.2, 288.6] | 367.2 [313.4, 427.0] | 840.0 [715.4, 1010.1] |
| 1 | all scored | hold | 49.4 [39.5, 62.8] | 135.4 [103.4, 172.3] | 296.3 [247.1, 352.6] | 441.1 [370.0, 516.3] | 1165.0 [929.0, 1374.5] |
| 1 | with E | E | 52.5 [43.6, 63.5] | 113.8 [94.6, 134.9] | 222.4 [186.7, 261.3] | 333.7 [282.3, 383.7] | 756.0 [641.7, 906.6] |
| 1 | with E | E_h | 49.6 [40.1, 61.2] | 117.0 [95.2, 141.8] | 236.5 [197.8, 276.7] | 347.8 [297.0, 402.5] | 800.0 [656.9, 937.5] |
| 1 | with E | estimator | 55.7 [46.3, 66.6] | 116.4 [97.9, 136.8] | 223.2 [187.1, 261.3] | 335.2 [284.6, 385.4] | 763.9 [651.7, 915.6] |
| 1 | with E | twin | 49.9 [40.4, 61.8] | 118.2 [96.6, 140.9] | 235.9 [198.9, 274.4] | 343.4 [297.3, 396.8] | 788.6 [649.3, 938.0] |
| 1 | with E | causal GP | 48.1 [38.0, 60.6] | 119.9 [96.7, 143.7] | 241.1 [201.5, 283.2] | 361.4 [308.8, 416.4] | 813.6 [687.0, 985.2] |
| 1 | with E | hold | 48.9 [39.2, 62.3] | 133.4 [101.8, 170.5] | 292.5 [243.7, 348.0] | 435.5 [365.1, 509.3] | 1126.6 [876.0, 1330.0] |
| 7 | all scored | estimator | 46.2 [40.0, 54.2] | 101.0 [84.6, 121.6] | 204.7 [170.4, 243.8] | 310.2 [259.9, 359.8] | 681.5 [572.6, 803.5] |
| 7 | all scored | twin | 48.8 [42.2, 57.9] | 109.4 [92.3, 132.0] | 224.1 [188.1, 267.9] | 340.5 [286.0, 401.9] | 772.7 [643.9, 886.8] |
| 7 | all scored | causal GP | 46.8 [39.8, 55.8] | 107.4 [90.7, 129.4] | 219.8 [185.0, 263.2] | 333.7 [278.2, 391.4] | 776.9 [639.1, 928.8] |
| 7 | all scored | hold | 49.5 [41.6, 59.7] | 121.0 [98.0, 151.5] | 260.1 [212.0, 314.6] | 388.7 [320.4, 465.4] | 972.4 [788.3, 1224.7] |
| 7 | with E | E | 44.1 [38.1, 51.8] | 97.7 [81.6, 118.2] | 198.3 [164.6, 235.7] | 295.6 [249.2, 344.3] | 634.7 [543.0, 749.2] |
| 7 | with E | E_h | 47.0 [40.5, 55.7] | 104.6 [88.4, 126.0] | 212.2 [175.4, 253.9] | 317.1 [267.4, 374.6] | 700.7 [580.3, 797.7] |
| 7 | with E | estimator | 45.6 [39.5, 53.2] | 99.0 [83.5, 119.2] | 198.9 [166.0, 237.8] | 300.3 [250.5, 350.0] | 631.5 [543.2, 744.6] |
| 7 | with E | twin | 48.1 [41.8, 57.0] | 107.1 [90.6, 129.1] | 218.9 [183.0, 261.4] | 330.6 [277.8, 390.8] | 731.4 [609.2, 827.0] |
| 7 | with E | causal GP | 46.1 [39.4, 54.7] | 105.6 [89.0, 126.4] | 214.0 [180.1, 254.9] | 322.4 [269.8, 377.3] | 721.7 [595.9, 856.5] |
| 7 | with E | hold | 48.8 [41.1, 58.7] | 118.6 [96.2, 148.8] | 255.5 [207.4, 306.3] | 376.8 [309.2, 453.8] | 908.7 [733.3, 1147.1] |
| 123 | all scored | estimator | 43.1 [36.8, 52.7] | 101.9 [81.8, 127.3] | 223.7 [182.7, 268.6] | 343.4 [283.2, 409.9] | 861.0 [724.3, 1029.4] |
| 123 | all scored | twin | 43.6 [36.5, 54.1] | 107.7 [85.8, 135.9] | 235.6 [194.4, 286.9] | 366.6 [303.2, 443.0] | 874.4 [741.8, 1023.4] |
| 123 | all scored | causal GP | 42.0 [35.0, 52.5] | 108.6 [85.6, 136.2] | 241.8 [197.5, 294.7] | 378.0 [314.3, 459.2] | 924.1 [776.9, 1088.7] |
| 123 | all scored | hold | 43.2 [35.8, 54.7] | 115.3 [88.8, 152.3] | 276.4 [219.9, 346.5] | 455.2 [353.5, 562.1] | 1281.1 [1037.5, 1620.2] |
| 123 | with E | E | 40.3 [33.7, 50.2] | 97.7 [78.2, 122.5] | 213.7 [175.2, 259.1] | 333.1 [273.4, 390.2] | 822.4 [696.9, 979.7] |
| 123 | with E | E_h | 42.6 [35.8, 53.2] | 105.4 [84.0, 132.8] | 231.1 [190.9, 281.8] | 359.2 [296.3, 433.8] | 842.4 [720.8, 994.8] |
| 123 | with E | estimator | 42.6 [36.5, 51.9] | 99.8 [80.2, 124.3] | 218.0 [178.3, 264.4] | 336.2 [277.9, 401.9] | 829.2 [681.7, 990.3] |
| 123 | with E | twin | 43.0 [36.2, 53.4] | 105.7 [84.4, 133.4] | 229.8 [189.6, 282.6] | 357.9 [296.1, 433.9] | 844.9 [720.1, 1002.0] |
| 123 | with E | causal GP | 41.4 [34.6, 51.9] | 106.6 [83.7, 133.4] | 235.1 [194.0, 284.9] | 370.6 [304.6, 447.6] | 889.8 [749.0, 1055.0] |
| 123 | with E | hold | 42.7 [35.5, 53.9] | 113.1 [87.5, 149.0] | 271.5 [214.8, 339.1] | 443.6 [345.9, 547.5] | 1249.2 [992.5, 1569.3] |

## Expectations of the registration (section 6)

| expectation | observed | met |
|---|---|---|
| D1: share 0.50 to 0.65 | 0.647, 0.727, 0.731, 0.725 | no |
| D1: interval above 0.5 on one to three splits | 4 of 4 | no |
| D2: share 0.50 to 0.65 | 0.706, 0.614, 0.849, 0.637 | no |
| D2: interval above 0.5 on one to three splits | 4 of 4 | no |
| by the median absolute error (vs causal GP): 0.35 to 0.55 | 0.612, 0.398, 0.581, 0.549 | no |
| by the median absolute error (vs twin): 0.35 to 0.55 | 0.765, 0.432, 0.677, 0.615 | no |
| ten discharges with the largest reduction hold 0.5 to 1.2 of the reduction vs the causal GP | 0.748, 0.470, 0.525, 0.620 | no |
| L2 holds | 4 of 4 | yes |
| LF over LH +0.01 to +0.05 | +0.055, +0.047, +0.046, +0.032 | no |
| L1 holds | 4 of 4 | yes |
| estimator over LF +0.03 to +0.10 | +0.109, +0.075, +0.119, +0.106 | no |
| LH over the causal GP -0.05 to +0.03 | -0.098, -0.032, -0.034, -0.051 | no |
| rows of 20 ms or more: 0.02 to 0.04 of the scored rows | 0.028, 0.037, 0.035, 0.041 | no |
| skill there over the causal GP +0.03 to +0.25 | +0.056, +0.089, +0.109, +0.105 | yes |
| G1: interval above zero on two to four splits | 3 of 4 | yes |
| V1 holds | 4 of 4 | yes |
| V2 holds | 4 of 4 | yes |
| V1 skill within 0.03 of the value on all rows | +0.034, -0.005, +0.005, +0.023 | no |
| V2 skill within 0.03 of the value on all rows | +0.024, +0.015, +0.024, +0.025 | yes |

7 of 19 expectations met.

## Choices made where the registration left room

- Scored rows: the k = 1 rows of the B.48 row table in which the estimator (`causal`), the twin (`hist80`), hold, the causal GP and y are finite (the population of B.48 / B.50). `file` = index into the sorted file names of the cut grid (`run_b37.Ctx`), `row` = grid row.
- E and E_h: means of the three CPU estimates of the causal variant (resp. the twin) cached by B.49 (`data/.b49/s{seed}.npz`, initialisations as in b49_average.json); B.49 `shot` indexes the files with at least one finite target and `row` the compacted rows of that file (`run_b49.compact_index`); mapped to (file, grid row) and joined to the scored rows. Every quantity involving E uses the scored rows that carry E; the errors of E are taken against y of the table.
- Bootstrap: 2,000 multinomial resamples of the discharges of each row set from `numpy.random.default_rng(20261118 + split)` (a fresh generator per row set: all scored rows; the part L rows; the rows with E; the eligible discharges of part D); interval = 2.5 and 97.5 % quantiles (`run_b45.ci`). Skills: 1 - SSE_A / SSE_B on the same rows, per-discharge sums resampled. Percentile intervals: weighted quantiles under the resamples (`run_b49.wquant`).
- Part D: eligible = test discharges with at least 50 scored rows (for E: at least 50 scored rows with E). 'Lower' is strict. Share interval: the share over resamples of the eligible discharges. Quartiles of the per-discharge skill 1 - MSE_A / MSE_B over the eligible discharges. Reduction against a reference = SSE_ref - SSE_est per discharge over all scored discharges (not only the eligible ones); the share of the ten largest reductions is their sum over the total (it exceeds 1 when the others sum to a negative number). For E, D1 and D2 are given against the causal GP and the single twin; E against E_h is added.
- Part L rows: the T_i of the cut grid (held values removed, values above 3 keV removed) is finite at the row; the three most recent finite T_i strictly before the row lie in the segment of the row (segments of `run_b37.Ctx`: a new segment after a time step of at least 0.5 s); the ten rows before lie in the segment (`dep10` of run_b51: position 10 or later in the segment, all ten finite). Ages in ms from the row time; time step = step to the row before over 10 ms (`run_b37.Ctx.tstep`). Fast channels = grid columns 3 to 17 (9 BES, 4 ECEI, 2 MC; raw values), departure = value minus the mean of the ten rows before.
- Part L fitting: every input and the target standardized with the mean and standard deviation over the part L rows of the training discharges of the split (manifest `train_files`). Ridge: minimise mean squared error + lambda * |w|^2 on the standardized scale (the intercept unpenalised; lambda = 0 by `numpy.linalg.lstsq`); lambda chosen from {0, 0.001, 0.01, 0.1, 1, 10} by the mean squared error on the part L rows of the validation discharges (ties to the smaller lambda); the fit on the training rows with that lambda is applied once to the test rows (no refit on training plus validation).
- Part L scoring: the scored rows that are also part L rows (joined by (file, grid row)); checked that y of the design equals y of the table and that the first history input (the last visible value) equals `hold` of the table on every joined row (1e-3 eV). Every skill of part L is on those rows; E over LF on those rows that also carry E.
- Part G: age of the last visible value rounded to a multiple of 10 ms; bins 10 ms, 20 to 50 ms, above 50 ms; G1 reads the union of the last two bins ('20 ms or more').
- Part V: a scored row enters at threshold T when y <= T and hold <= T (hold is the last visible value).
- Part P: percentiles (numpy linear interpolation) of |estimate - y| on the scored rows for the estimator, the twin, the causal GP and hold, cross-checked against `b48_reread.json` (b41 G); and for all of them with E and E_h on the scored rows that carry E.
- Checks (stop rules): grid T_i = y of the table on every scored row; the unstratified skills of the estimator over the twin, the causal GP and hold equal B.48 (b44 overall) to three decimals (|difference| < 5e-4, as run_b50); every B.49 row joins a scored row and the grid T_i there equals y of the table; (Addendum 1) E over the causal GP, with the causal-GP errors stored by eval_seq (those B.49 used) and the errors of E against y of the table, equals the E1 point of b49_average.json to three decimals (|difference| < 5e-4); the part L join check above. All checks run on all four splits before any readout.

## Deviations

Addendum 1 of the registration (2026-11-18, committed in f656d60 before any readout was computed): the first run stopped at the E check on split 42 (E over the causal GP +0.0771 with the causal GP of the B.48 row table against +0.0780 of B.49; splits 1, 7 and 123 passed). The join itself was exact (every B.49 row joined, y equal); B.49 had used the causal-GP errors stored by eval_seq, which differ from the GP of the row table on a few rows. The check is restated: E over the causal GP is compared with B.49 using the stored causal-GP errors (the run stops if a split differs at the third decimal); every readout uses the causal GP of the row table, as registered; the rows on which the two computations differ and the skill of E with both are reported. No other change from the registered definitions, rows, intervals, seeds or rules.

## Limits

Stated in advance (section 7): a discharge is one unit whatever its length; the linear filter is one choice of inputs; the value threshold is applied to the scored rows and the last visible value only (the estimator was trained with values up to 3 keV in its inputs); one initialisation for parts G and V; the test rows are those of every earlier readout.

Run time: 216 s.
