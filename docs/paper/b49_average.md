# B.49 — the average of three initialisations as the estimator

Registered: PREREGISTRATION_B49.md, after S1 and S2 of B.39 had failed and before any of these readouts was computed. CPU only; nothing is trained. T_i, cut population, the four file-level splits, the test rows of `eval_seq`. E is the mean of the T_i estimates of the three initialisations of the causal variant of B.39 at each scored row; E_h is the same mean for the three initialisations of the history-only twin. Hold, the causal GP, PCHIP and the offline GP are the stored errors of the same rows. Skill of A over R = 1 - sum of squared errors of A over that of R on the same rows. Intervals: 95 %, 2,000 resamples of the test discharges; no initialisation is drawn, since E is one estimator. A statement holds on a split when the whole interval is above zero, and holds when it does on at least three of the four splits. These statements concern an estimator other than the single training of S1 and S2 and were registered after those had failed.

## Checks before the readout

Each member was run forward on the CPU over the test discharges; its absolute error at every scored row is compared with the square root of its stored squared error (tolerance 0.5 eV, required share 99 %). All six members of a split were scored on the same rows with the same `y_true`, and the stored reference errors agree.

| split | member | run | rows | share within 0.5 eV | largest deviation (eV) |
|---|---|---|---|---|---|
| 42 | v2 i42 | `.b39_v2_cut_s42_i42` | 32,589 | 0.9990 | 1.344 |
| 42 | v2 i1 | `.b39_v2_cut_s42_i1` | 32,589 | 0.9985 | 3.315 |
| 42 | v2 i7 | `.b39_v2_cut_s42_i7` | 32,589 | 0.9991 | 1.404 |
| 42 | v2h i42 | `.b29a1_v2h_cut_s42` | 32,589 | 0.9996 | 4.273 |
| 42 | v2h i1 | `.b39_v2h_cut_s42_i1` | 32,589 | 1.0000 | 0.266 |
| 42 | v2h i7 | `.b39_v2h_cut_s42_i7` | 32,589 | 1.0000 | 0.635 |
| 1 | v2 i1 | `.b39_v2_cut_s1_i1` | 35,582 | 1.0000 | 0.552 |
| 1 | v2 i7 | `.b39_v2_cut_s1_i7` | 35,582 | 1.0000 | 0.737 |
| 1 | v2 i123 | `.b39_v2_cut_s1_i123` | 35,582 | 1.0000 | 0.402 |
| 1 | v2h i1 | `.b29a1_v2h_cut_s1` | 35,582 | 0.9999 | 1.255 |
| 1 | v2h i7 | `.b39_v2h_cut_s1_i7` | 35,582 | 0.9999 | 0.682 |
| 1 | v2h i123 | `.b39_v2h_cut_s1_i123` | 35,582 | 0.9999 | 1.000 |
| 7 | v2 i7 | `.b39_v2_cut_s7_i7` | 35,759 | 1.0000 | 0.498 |
| 7 | v2 i123 | `.b39_v2_cut_s7_i123` | 35,759 | 0.9999 | 0.786 |
| 7 | v2 i42 | `.b39_v2_cut_s7_i42` | 35,759 | 0.9981 | 4.350 |
| 7 | v2h i7 | `.b29a1_v2h_cut_s7` | 35,759 | 1.0000 | 0.274 |
| 7 | v2h i123 | `.b39_v2h_cut_s7_i123` | 35,759 | 0.9999 | 0.609 |
| 7 | v2h i42 | `.b39_v2h_cut_s7_i42` | 35,759 | 0.9998 | 0.866 |
| 123 | v2 i123 | `.b39_v2_cut_s123_i123` | 34,038 | 0.9999 | 0.558 |
| 123 | v2 i42 | `.b39_v2_cut_s123_i42` | 34,038 | 0.9999 | 1.262 |
| 123 | v2 i1 | `.b39_v2_cut_s123_i1` | 34,038 | 0.9991 | 2.106 |
| 123 | v2h i123 | `.b29a1_v2h_cut_s123` | 34,038 | 0.9984 | 5.985 |
| 123 | v2h i42 | `.b39_v2h_cut_s123_i42` | 34,038 | 0.9995 | 0.996 |
| 123 | v2h i1 | `.b39_v2h_cut_s123_i1` | 34,038 | 0.9996 | 1.446 |

Largest difference between the grid T_i at the mapped row and the stored `y_true` (eV): split 42 0.00024, split 1 0.00024, split 7 0.00024, split 123 0.00024.

## Registered statements

- **E1** (E over the causal GP): 4 of 4 -> **holds**.
- **E2** (E over the twin average E_h): 4 of 4 -> **holds**.

| split | rows | discharges | initialisations | E1: E over causal GP | interval above zero | E2: E over E_h | interval above zero |
|---|---|---|---|---|---|---|---|
| 42 | 32,589 | 96 | 42, 1, 7 | +0.078 [+0.019, +0.125] | yes | +0.067 [+0.015, +0.108] | yes |
| 1 | 35,582 | 96 | 1, 7, 123 | +0.105 [+0.076, +0.132] | yes | +0.039 [+0.015, +0.062] | yes |
| 7 | 35,759 | 96 | 7, 123, 42 | +0.153 [+0.117, +0.191] | yes | +0.088 [+0.057, +0.124] | yes |
| 123 | 34,038 | 96 | 123, 42, 1 | +0.115 [+0.065, +0.161] | yes | +0.066 [+0.029, +0.105] | yes |

## E beside the single trainings of B.39

Single-training values from `docs/paper/b39_primary.json` (stored GPU errors, point values; S1 and S2 with the registered interval that also draws the initialisation).

| split | each member over causal GP (B.39) | S1 (B.39) | E over causal GP | S2 (B.39) | E over E_h |
|---|---|---|---|---|---|
| 42 | i42: +0.081; i1: +0.061; i7: +0.058 | +0.067 [-0.014, +0.120] | +0.078 [+0.019, +0.125] | +0.071 [-0.012, +0.146] | +0.067 [+0.015, +0.108] |
| 1 | i1: +0.094; i7: +0.101; i123: +0.104 | +0.100 [+0.068, +0.129] | +0.105 [+0.076, +0.132] | +0.041 [+0.006, +0.082] | +0.039 [+0.015, +0.062] |
| 7 | i7: +0.144; i123: +0.138; i42: +0.142 | +0.141 [+0.101, +0.182] | +0.153 [+0.117, +0.191] | +0.089 [+0.037, +0.145] | +0.088 [+0.057, +0.124] |
| 123 | i123: +0.101; i42: +0.130; i1: +0.064 | +0.098 [-0.017, +0.162] | +0.115 [+0.065, +0.161] | +0.062 [-0.046, +0.147] | +0.066 [+0.029, +0.105] |

## Reported

E over the references:

| split | over hold | over PCHIP | over the offline GP | E_h over causal GP |
|---|---|---|---|---|
| 42 | +0.360 [+0.304, +0.403] | +0.174 [+0.087, +0.244] | -0.052 [-0.136, +0.016] | +0.012 [-0.023, +0.045] |
| 1 | +0.387 [+0.356, +0.413] | +0.224 [+0.167, +0.270] | +0.008 [-0.033, +0.040] | +0.069 [+0.040, +0.098] |
| 7 | +0.414 [+0.378, +0.444] | +0.270 [+0.208, +0.320] | +0.036 [-0.014, +0.080] | +0.071 [+0.045, +0.095] |
| 123 | +0.421 [+0.374, +0.457] | +0.273 [+0.198, +0.327] | +0.005 [-0.064, +0.060] | +0.053 [+0.020, +0.081] |

Gain from averaging (members on the CPU, same arithmetic as E):

| split | E over each member | mean of those three | skill of E over causal GP minus mean skill of the members over causal GP | members over causal GP (CPU): causal variant / twin |
|---|---|---|---|---|
| 42 | i42: -0.003 [-0.030, +0.011]; i1: +0.018 [-0.003, +0.055]; i7: +0.021 [+0.001, +0.056] | +0.012 [+0.004, +0.027] | +0.011 [+0.004, +0.028] | i42: +0.081 / +0.011; i1: +0.061 / -0.042; i7: +0.058 / +0.017 |
| 1 | i1: +0.012 [+0.006, +0.018]; i7: +0.004 [-0.010, +0.023]; i123: +0.002 [-0.006, +0.008] | +0.006 [+0.004, +0.009] | +0.006 [+0.004, +0.008] | i1: +0.094 / +0.074; i7: +0.101 / +0.067; i123: +0.104 / +0.042 |
| 7 | i7: +0.011 [+0.003, +0.021]; i123: +0.018 [+0.010, +0.026]; i42: +0.014 [-0.010, +0.044] | +0.014 [+0.008, +0.022] | +0.012 [+0.007, +0.019] | i7: +0.144 / +0.024; i123: +0.138 / +0.071; i42: +0.141 / +0.076 |
| 123 | i123: +0.015 [-0.002, +0.031]; i42: -0.018 [-0.038, -0.002]; i1: +0.055 [+0.008, +0.123] | +0.017 [+0.009, +0.029] | +0.016 [+0.008, +0.032] | i123: +0.101 / +0.044; i42: +0.130 / +0.018; i1: +0.064 / +0.056 |

Averages of two members (the dependence on which trainings enter):

| split | pair | average of two causal-variant members over causal GP | over the average of the two twins of the same initialisations |
|---|---|---|---|
| 42 | i42 + i1 | +0.082 [+0.029, +0.126] | +0.081 [+0.037, +0.120] |
| 42 | i42 + i7 | +0.079 [+0.025, +0.123] | +0.063 [+0.020, +0.100] |
| 42 | i1 + i7 | +0.064 [-0.019, +0.125] | +0.060 [-0.022, +0.114] |
| 1 | i1 + i7 | +0.103 [+0.070, +0.130] | +0.031 [+0.006, +0.056] |
| 1 | i1 + i123 | +0.101 [+0.074, +0.127] | +0.037 [+0.017, +0.058] |
| 1 | i7 + i123 | +0.107 [+0.075, +0.134] | +0.049 [+0.023, +0.076] |
| 7 | i7 + i123 | +0.146 [+0.111, +0.182] | +0.089 [+0.061, +0.120] |
| 7 | i7 + i42 | +0.153 [+0.116, +0.193] | +0.093 [+0.061, +0.131] |
| 7 | i123 + i42 | +0.152 [+0.115, +0.192] | +0.083 [+0.047, +0.123] |
| 123 | i123 + i42 | +0.125 [+0.084, +0.168] | +0.084 [+0.050, +0.125] |
| 123 | i123 + i1 | +0.097 [+0.036, +0.146] | +0.046 [-0.009, +0.090] |
| 123 | i42 + i1 | +0.110 [+0.052, +0.163] | +0.063 [+0.024, +0.105] |

| split | range over the three pairs: over causal GP | over the twin average |
|---|---|---|
| 42 | +0.064 to +0.082 | +0.060 to +0.081 |
| 1 | +0.101 to +0.107 | +0.031 to +0.049 |
| 7 | +0.146 to +0.153 | +0.083 to +0.093 |
| 123 | +0.097 to +0.125 | +0.046 to +0.084 |

Absolute error (eV; point with 95 % interval over the resamples):

| split | median: E | E_h | hold | causal GP | ratio E / causal GP | 90th percentile: E | E_h | hold | causal GP | ratio E / causal GP |
|---|---|---|---|---|---|---|---|---|---|---|
| 42 | 41.8 [36.7, 48.2] | 43.8 [38.2, 51.3] | 43.8 [37.7, 51.7] | 42.2 [36.7, 49.5] | 0.991 [0.950, 1.031] | 176.5 [151.1, 206.2] | 188.7 [160.2, 220.7] | 225.4 [182.9, 269.9] | 188.4 [160.0, 219.9] | 0.937 [0.908, 0.975] |
| 1 | 52.5 [43.6, 63.5] | 49.6 [40.1, 61.2] | 48.9 [39.2, 62.3] | 48.1 [38.0, 60.6] | 1.093 [1.015, 1.182] | 222.4 [186.7, 261.3] | 236.5 [197.8, 276.7] | 292.5 [243.7, 348.0] | 241.1 [201.5, 283.2] | 0.922 [0.896, 0.948] |
| 7 | 44.1 [38.1, 51.8] | 47.0 [40.5, 55.7] | 48.8 [41.1, 58.7] | 46.1 [39.4, 54.7] | 0.957 [0.919, 0.996] | 198.3 [164.6, 235.7] | 212.2 [175.4, 253.9] | 255.5 [207.4, 306.3] | 214.0 [180.1, 254.9] | 0.927 [0.893, 0.952] |
| 123 | 40.3 [33.7, 50.2] | 42.6 [35.8, 53.2] | 42.7 [35.5, 53.9] | 41.4 [34.6, 51.9] | 0.973 [0.936, 1.015] | 213.7 [175.2, 259.1] | 231.1 [190.9, 281.8] | 271.5 [214.8, 339.1] | 235.1 [194.0, 284.9] | 0.909 [0.876, 0.943] |

Root-mean-square error (eV):

| split | E | E_h | causal-variant members | twin members | hold | causal GP | PCHIP | offline GP |
|---|---|---|---|---|---|---|---|---|
| 42 | 157.8 | 163.3 | 157.5, 159.2, 159.5 | 163.4, 167.7, 162.9 | 197.2 | 164.3 | 173.6 | 153.8 |
| 1 | 196.1 | 200.0 | 197.3, 196.6, 196.3 | 199.6, 200.3, 203.0 | 250.5 | 207.4 | 222.7 | 196.9 |
| 7 | 172.1 | 180.3 | 173.1, 173.7, 173.3 | 184.8, 180.3, 179.8 | 224.8 | 187.1 | 201.5 | 175.3 |
| 123 | 200.6 | 207.5 | 202.1, 198.8, 206.3 | 208.5, 211.3, 207.1 | 263.7 | 213.2 | 235.2 | 201.1 |

## Expectations of section 6

| expectation | values | met |
|---|---|---|
| E over the causal GP: +0.08 to +0.16 (each split) | 42: +0.078, 1: +0.105, 7: +0.153, 123: +0.115 (3 of 4 in range) | missed |
| E over the causal GP above the mean of its members by +0.005 to +0.03 on each split | 42: +0.011, 1: +0.006, 7: +0.012, 123: +0.016 (4 of 4 in range) | met |
| E1 holds on three or four splits | 4 of 4 | met |
| E over E_h: +0.04 to +0.10 (each split) | 42: +0.067, 1: +0.039, 7: +0.088, 123: +0.066 (3 of 4 in range) | missed |
| E2 holds on three or four splits | 4 of 4 | met |
| E_h over the causal GP: +0.02 to +0.08 (each split) | 42: +0.012, 1: +0.069, 7: +0.071, 123: +0.053 (3 of 4 in range) | missed |
| ratio of the median absolute error of E to that of the causal GP: 0.97 to 1.10 (each split) | 42: 0.991, 1: 1.093, 7: 0.957, 123: 0.973 (3 of 4 in range) | missed |

## Choices made where the registration left room

- Scored rows: `CES_TI_row` is the index into the per-file array of dataset.py (grid rows with at least one finite target, `compact_index` of B.45); the estimate of a member at a scored row is its output at that grid row. The grid T_i at the mapped row equals the stored `y_true` within the tolerance printed under the checks.
- Errors of E, E_h, the members and the two-member averages are computed against the stored `y_true`; the references (hold, causal GP, PCHIP, offline GP) are the stored squared errors of the same rows.
- Each network runs with the standardization recorded in its own metrics.json (causal variant: statistics of the training discharges; twin: `per_shot_input_norm` as recorded, which it does not read since it sees no fast channel).
- Members are compared with E using their CPU estimates (the same arithmetic as E); the stored GPU errors of the members give the B.39 single-training values, listed beside them.
- "E over each of its members and the mean of those three": the mean of the three skills of E over a member, with the interval of that mean over the same resamples. The expectation "above the mean of its members by +0.005 to +0.03" is read as skill(E over the causal GP) minus the mean of the three skills of the members over the causal GP, with its own interval.
- Resamples: discharges drawn with replacement, 2,000 times, from `default_rng(20261118 + split seed)`; the same draws for every quantity of a split. Percentiles under resampling are weighted quantiles (inverted CDF, weight = the number of times the discharge is drawn); the point values are the plain median and 90th percentile.
- An expectation stated as a range is counted as met when the point value lies in the range on all four splits; the counts of splits in range are given.

## Deviations

- None from the registered design.

Limits stated in advance (section 7): three members; the test rows were read by every earlier readout; the members were trained independently, not as an ensemble.

Run time: 41 s.
