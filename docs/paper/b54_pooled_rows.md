# B.54 -- the row-level readouts on the rows of the pooled cross-validation (registered: ces_prediction/experiments/PREREGISTRATION_B54.md)

CPU, nothing trained (except the propensity model of B.37 F, refitted per fold), no network run. T_i, cut population, the 80 pooled runs of B.39 arm P (repeats r2 to r9, five folds each): the estimator `.b39p_v2_cut_r{r}_f{k}` and the history-only twin `.b39p_v2h_cut_r{r}_f{k}`; the five folds of a repeat concatenated (639 discharges partitioned per repeat; the resampled units are the discharges with scored rows, 633 to 635 per repeat, Addendum 1). Skill of the estimator over R = 1 - SSE_est / SSE_R on the same rows. Intervals: 95 %, two-level bootstrap of the pooled readout (a repeat drawn, then its discharges resampled); 20,000 draws for skills, 2,000 for percentile ratios and shares. Point = mean over the eight repeats. A statement holds if its interval excludes the reference on the stated side.

## Checks before the readout

- **(a)** P1 and P2 of B.39 from the stored errors: P1 +0.089 [+0.064, +0.112] vs recorded +0.089 [+0.064, +0.112]; P2 +0.048 [+0.022, +0.073] vs recorded +0.048 [+0.022, +0.073]. Points equal to three decimals: yes; intervals equal to three decimals: yes -> **passes**.
- **(b)** (as fixed by Addendum 1) by the fold manifests, test entries / distinct test files / universe per repeat: r2 639/639/639; r3 639/639/639; r4 639/639/639; r5 639/639/639; r6 639/639/639; r7 639/639/639; r8 639/639/639; r9 639/639/639, i.e. each of the 639 discharges in exactly one test fold. Discharges with scored rows (the resampled units), each in exactly one fold: r2 635 (yes); r3 635 (yes); r4 635 (yes); r5 635 (yes); r6 634 (yes); r7 634 (yes); r8 634 (yes); r9 633 (yes). The 8 discharges of the 639 without scored rows in at least one repeat: s31002 (no rows in r2, r3, r4, r5, r6, r7, r8, r9); s31331 (no rows in r2, r3, r4, r5, r6, r7, r8, r9); s31482 (no rows in r8); s31483 (no rows in r7, r9); s31485 (no rows in r9); s31767 (no rows in r6); s32501 (no rows in r2, r3, r4, r5, r6, r7, r8, r9); s32570 (no rows in r2, r3, r4, r5, r6, r7, r8, r9). Files with a target outside the 639: s30967, s31702 -> **passes**.
- **(c)** estimator and twin scored on identical rows with equal stored reference errors in 40 of 40 folds (rows per fold 37,483 to 38,920) -> **passes**.
- **(d)** value in the data file at the located row against `CES_TI_y_true`: largest absolute difference 2.44e-04 eV over 1,536,645 rows; rows above 0.01 eV 0; rows on a first or last file row 0; discharges outside their fold's test files 0 -> **passes**. Reported: the hold located in the file reproduces sqrt(se_persistence) within 0.5 eV on a share 1.00000 to 1.00000 of the rows of a fold (largest deviation 4.88e-04 eV).
- **(e)** code applied to split 42, against the recorded numbers:

| quantity | this code | recorded | where | equal to three decimals |
|---|---|---|---|---|
| flagged_share | 0.1025 | 0.1025 | b48_reread.json b41.G.42.flagged | yes |
| increment_rows_not_flagged | 0.1177 | 0.1177 | b48_reread.json b41.G.42.plain.skill_causal_vs_twin | yes |
| quiet_share | 0.8890 | 0.8890 | b45_ablation_rows.json splits.42.groups.6.row_share (stored rows of .b39_v2_cut_s42_i42) | yes |
| increment_top_fifth_risk | 0.0637 | 0.0637 | b50_missing_like.json splits.42.fifths[4].skill.hist80 | yes |

-> **passes**. Further comparisons on split 42 (reported):

| quantity | this code | recorded | equal to three decimals |
|---|---|---|---|
| propensity_auc | 0.9739 | 0.9739 | yes |
| skill_top_fifth_risk_over_gp | 0.0671 | 0.0671 | yes |
| blip_rows_1_3_increment | 0.2605 | 0.2605 | yes |
| third_low_over_hist80 | 0.0113 | 0.0113 | yes |
| third_low_over_gp | 0.0779 | 0.0779 | yes |
| third_middle_over_hist80 | 0.0719 | 0.0719 | yes |
| third_middle_over_gp | 0.0804 | 0.0804 | yes |
| third_high_over_hist80 | 0.1028 | 0.1028 | yes |
| third_high_over_gp | 0.0680 | 0.0680 | yes |
| age_20ms_or_more_over_gp | 0.0556 | 0.0556 | yes |
| share_lower_mse_over_gp | 0.6471 | 0.6471 | yes |
| top10_share_over_gp | 0.7484 | 0.7484 | yes |

Propensity join on the B.48 rows of split 42: 32,127 of 32,921 rows with a window, 0 of them without p, 0 without a window but with p.

All five checks pass.

## Statements

### A

| statement | point [95 % interval] | reference | verdict | per repeat (r2 ... r9) |
|---|---|---|---|---|
| **A1**: 90th percentile of the absolute error, estimator over causal GP; interval below 1 | 0.946 [0.928, 0.965] (excl. 1) | 1 | **holds** | 0.938, 0.949, 0.937, 0.948, 0.950, 0.949, 0.942, 0.958 |

### B

| statement | point [95 % interval] | reference | verdict | per repeat (r2 ... r9) |
|---|---|---|---|---|
| **B1**: share of discharges (>= 50 rows) where the estimator has the lower MSE than the causal GP; above 0.5 | 0.684 [0.640, 0.730] (excl. 0.5) | 0.5 | **holds** | 0.670, 0.701, 0.689, 0.673, 0.678, 0.669, 0.706, 0.688 |
| **B2**: the same against the twin; above 0.5 | 0.661 [0.591, 0.723] (excl. 0.5) | 0.5 | **holds** | 0.609, 0.674, 0.691, 0.642, 0.666, 0.650, 0.651, 0.708 |

Discharges with at least 50 scored rows (the units of B1 and B2) per repeat: r2 585, r3 585, r4 585, r5 584, r6 580, r7 583, r8 581, r9 583; resampled discharges (with scored rows) per repeat for every other statement: r2 635, r3 635, r4 635, r5 635, r6 634, r7 634, r8 634, r9 633.

### C

| statement | point [95 % interval] | reference | verdict | per repeat (r2 ... r9) |
|---|---|---|---|---|
| **C1**: rows not flagged by the value rules: estimator over the twin; above 0 | +0.092 [+0.062, +0.125] (excl. 0) | 0 | **holds** | +0.089, +0.101, +0.105, +0.079, +0.110, +0.080, +0.085, +0.088 |
| **C2**: rows not flagged: estimator over the causal GP; above 0 | +0.096 [+0.066, +0.123] (excl. 0) | 0 | **holds** | +0.099, +0.102, +0.101, +0.093, +0.097, +0.082, +0.101, +0.089 |
| **C3**: rows not flagged hold more than half of the reduction against the causal GP; above 0.5 | 0.463 [0.355, 0.610] (incl. 0.5) | 0.5 | **does not hold** | 0.476, 0.488, 0.456, 0.488, 0.442, 0.463, 0.452, 0.437 |
| **C4**: the same against the twin; above 0.5 | 0.874 [0.681, 1.385] (excl. 0.5) | 0.5 | **holds** | 0.874, 0.898, 0.762, 1.056, 0.817, 0.966, 0.831, 0.787 |

### D

| statement | point [95 % interval] | reference | verdict | per repeat (r2 ... r9) |
|---|---|---|---|---|
| D1, low third: estimator over the causal GP, interval above 0 in each third by median T_i | +0.066 [+0.032, +0.100] (excl. 0) | 0 | interval above 0 | |
| D1, middle third: estimator over the causal GP, interval above 0 in each third by median T_i | +0.091 [+0.044, +0.131] (excl. 0) | 0 | interval above 0 | |
| D1, high third: estimator over the causal GP, interval above 0 in each third by median T_i | +0.110 [+0.061, +0.153] (excl. 0) | 0 | interval above 0 | |
| **D1** (all three thirds) | | | **holds** | |
| D2, low third: the same over the twin | +0.007 [-0.023, +0.031] (incl. 0) | 0 | interval not above 0 | |
| D2, middle third: the same over the twin | +0.042 [+0.009, +0.074] (excl. 0) | 0 | interval above 0 | |
| D2, high third: the same over the twin | +0.092 [+0.040, +0.137] (excl. 0) | 0 | interval above 0 | |
| **D2** (all three thirds) | | | **does not hold** | |

### E

| statement | point [95 % interval] | reference | verdict | per repeat (r2 ... r9) |
|---|---|---|---|---|
| **E1**: last value 20 ms old or older: estimator over the causal GP; above 0 | +0.069 [+0.010, +0.127] (excl. 0) | 0 | **holds** | +0.055, +0.071, +0.087, +0.053, +0.101, +0.037, +0.072, +0.078 |
| **E2**: last value older than 50 ms: over the causal GP; above 0 | +0.054 [-0.095, +0.185] (incl. 0) | 0 | **does not hold** | +0.029, +0.038, +0.097, +0.063, +0.081, -0.006, +0.049, +0.082 |

### F

| statement | point [95 % interval] | reference | verdict | per repeat (r2 ... r9) |
|---|---|---|---|---|
| **F1**: top fifth by risk of losing the value: over the causal GP; above 0 | +0.087 [+0.053, +0.117] (excl. 0) | 0 | **holds** | +0.087, +0.083, +0.092, +0.077, +0.099, +0.071, +0.096, +0.091 |
| **F2**: top fifth by risk: over the twin; above 0 | +0.037 [+0.007, +0.065] (excl. 0) | 0 | **holds** | +0.035, +0.033, +0.052, +0.020, +0.050, +0.027, +0.037, +0.039 |
| **F3**: rows weighted by the odds of that risk: over the causal GP; above 0 | +0.074 [+0.033, +0.115] (excl. 0) | 0 | **holds** | +0.068, +0.072, +0.075, +0.065, +0.093, +0.058, +0.076, +0.088 |
| **F4**: odds-weighted: over the twin; above 0 | +0.020 [-0.014, +0.049] (incl. 0) | 0 | **does not hold** | +0.016, +0.017, +0.042, -0.004, +0.031, +0.007, +0.022, +0.027 |

### G

| statement | point [95 % interval] | reference | verdict | per repeat (r2 ... r9) |
|---|---|---|---|---|
| **G1**: rows 1 to 3 after an absent frame: over the twin; above 0 | +0.151 [+0.093, +0.215] (excl. 0) | 0 | **holds** | +0.137, +0.143, +0.179, +0.151, +0.167, +0.136, +0.139, +0.154 |

### H

| statement | point [95 % interval] | reference | verdict | per repeat (r2 ... r9) |
|---|---|---|---|---|
| **H1**: quiet rows: over the twin; above 0 | +0.054 [+0.026, +0.082] (excl. 0) | 0 | **holds** | +0.048, +0.060, +0.073, +0.038, +0.066, +0.039, +0.050, +0.056 |
| **H2**: quiet rows hold more than half of the reduction against the twin; above 0.5 | 0.874 [0.734, 1.098] (excl. 0.5) | 0.5 | **holds** | 0.828, 0.918, 0.913, 0.894, 0.849, 0.841, 0.865, 0.881 |

Statements that hold: A1, B1, B2, C1, C2, C4, D1, E1, F1, F2, F3, G1, H1, H2. Statements that do not hold: C3, D2, E2, F4.

## Reported quantities

### Overall (all pooled rows)

| over | skill of the estimator |
|---|---|
| twin | +0.048 [+0.022, +0.073] |
| causal GP | +0.089 [+0.064, +0.112] |
| hold | +0.375 [+0.352, +0.394] |
| offline GP | -0.037 [-0.068, -0.008] |

### Distribution of the absolute error

| estimator over | ratio of the 90th percentiles | ratio of the medians |
|---|---|---|
| causal GP | 0.946 [0.928, 0.965] (excl. 1) | 1.042 [1.007, 1.082] (excl. 1) |
| hold | 0.808 [0.788, 0.833] (excl. 1) | 1.013 [0.974, 1.056] (incl. 1) |
| twin | 0.953 [0.935, 0.971] (excl. 1) | 0.999 [0.944, 1.060] (incl. 1) |

Percentiles of the absolute error (eV, mean over repeats):

| estimator | p50 | p75 | p90 | p95 | p99 |
|---|---|---|---|---|---|
| estimator | 46.0 | 100.6 | 205.9 | 309.8 | 715.4 |
| twin | 46.1 | 103.8 | 216.0 | 326.8 | 741.2 |
| causal GP | 44.1 | 104.0 | 217.6 | 333.5 | 777.1 |
| hold | 45.4 | 111.4 | 254.7 | 388.6 | 1031.4 |
| offline GP | 37.5 | 91.5 | 198.0 | 304.6 | 720.2 |

### Per discharge

- Share of discharges with the lower MSE than hold: 0.845 [0.810, 0.877].
- Share of discharges by the median absolute error (lower than the causal GP): 0.480 [0.421, 0.535]; lower than the twin: 0.537 [0.412, 0.621].
- Eligible discharges (>= 50 scored rows) per repeat: 585, 585, 585, 584, 580, 583, 581, 583.
- Share of the total squared-error reduction against the causal GP held by the ten discharges with the largest reduction: 0.214 (per repeat 0.251, 0.199, 0.213, 0.221, 0.207, 0.228, 0.184, 0.207); by the twentieth (32 discharges): 0.493 (per repeat 0.536, 0.479, 0.494, 0.513, 0.466, 0.528, 0.438, 0.488). Against the twin, the ten with the largest reduction against the twin: 0.354.
- With those ten removed (lower bound; the removal selects on the outcome): over the causal GP +0.076 [+0.050, +0.098] (excl. 0); over the twin +0.041 [+0.016, +0.064] (excl. 0); over hold +0.368 [+0.346, +0.388].

### Skills by class of rows (every skill also against hold)

| rows | rows per repeat (mean) | share of rows | over twin | over causal GP | over hold | over offline GP |
|---|---|---|---|---|---|---|
| not flagged by the value rules | 170,440 | 0.8873 | +0.092 [+0.062, +0.125] | +0.096 [+0.066, +0.123] | +0.288 [+0.253, +0.321] | -0.034 [-0.075, +0.003] |
| flagged | 21,641 | 0.1127 | +0.012 [-0.016, +0.036] | +0.084 [+0.049, +0.115] | +0.426 [+0.399, +0.450] | -0.038 [-0.079, -0.004] |
| low third by median T_i | 39,357 | 0.2049 | +0.007 [-0.023, +0.031] | +0.066 [+0.032, +0.100] | +0.398 [+0.372, +0.422] |
| middle third by median T_i | 76,780 | 0.3997 | +0.042 [+0.009, +0.074] | +0.091 [+0.044, +0.131] | +0.322 [+0.275, +0.363] |
| high third by median T_i | 75,944 | 0.3954 | +0.092 [+0.040, +0.137] | +0.110 [+0.061, +0.153] | +0.386 [+0.345, +0.421] |
| age of the last value 10 ms | 187,071 | 0.9739 | +0.049 [+0.025, +0.074] | +0.093 [+0.068, +0.117] | +0.372 [+0.348, +0.393] |
| age of the last value 20 to 50 ms | 4,525 | 0.0236 | +0.048 [+0.001, +0.091] | +0.073 [+0.014, +0.133] | +0.396 [+0.338, +0.450] |
| age of the last value above 50 ms | 485 | 0.0025 | +0.015 [-0.084, +0.115] | +0.054 [-0.095, +0.185] | +0.355 [+0.244, +0.453] |
| age of the last value 20 ms or more | 5,010 | 0.0261 | +0.042 [-0.003, +0.086] | +0.069 [+0.010, +0.127] | +0.389 [+0.336, +0.439] |
| top fifth by risk | 37,515 | 0.1953 | +0.037 [+0.007, +0.065] | +0.087 [+0.053, +0.117] | +0.387 [+0.356, +0.412] |
| rows with a ten-row window, weighted by the odds | 187,574 | 0.9765 | +0.020 [-0.014, +0.049] | +0.074 [+0.033, +0.115] | +0.357 [+0.323, +0.389] |
| rows with a ten-row window, unweighted | 187,574 | 0.9765 | +0.049 [+0.023, +0.073] | +0.085 [+0.059, +0.108] | +0.376 [+0.353, +0.396] |
| rows 1 to 3 after an absent frame | 3,123 | 0.0163 | +0.151 [+0.093, +0.215] | +0.258 [+0.183, +0.330] | +0.404 [+0.338, +0.472] |
| class 1 after an absent frame | 11,001 | 0.0573 | +0.072 [+0.015, +0.128] | +0.141 [+0.071, +0.199] | +0.359 [+0.305, +0.406] |  |
| class 2 low beam emission | 2,122 | 0.0110 | -0.057 [-0.201, +0.038] | -0.146 [-0.369, +0.012] | +0.105 [-0.113, +0.250] |  |
| class 3 upward departure | 818 | 0.0043 | +0.070 [-0.024, +0.150] | +0.066 [-0.047, +0.162] | +0.260 [+0.153, +0.360] |  |
| class 4 downward departure | 149 | 0.0008 | +0.196 [+0.035, +0.421] | +0.300 [+0.041, +0.618] | +0.319 [-0.098, +0.663] |  |
| class 5 after a departure | 2,281 | 0.0119 | -0.008 [-0.074, +0.051] | +0.142 [+0.050, +0.236] | +0.487 [+0.397, +0.567] |  |
| class 6 quiet | 171,202 | 0.8913 | +0.054 [+0.026, +0.082] | +0.088 [+0.061, +0.112] | +0.387 [+0.364, +0.408] | -0.021 [-0.053, +0.007] |
| class 7 departure not defined | 4,508 | 0.0235 | +0.026 [-0.045, +0.095] | +0.233 [+0.111, +0.346] | +0.297 [+0.195, +0.383] |  |
| all but the ten discharges with the largest reduction | 188,604 | 0.9819 | +0.041 [+0.016, +0.064] | +0.076 [+0.050, +0.098] | +0.368 [+0.346, +0.388] |

Shares of the reduction: rows not flagged hold 0.463 [0.355, 0.610] of the reduction against the causal GP and 0.874 [0.681, 1.385] against the twin; quiet rows hold 0.874 [0.734, 1.098] against the twin and 0.766 [0.660, 0.896] against the causal GP. A share above 1 means the other rows have a net increase.

Thirds: edges of median T_i (eV) r2 336, 667; r3 336, 667; r4 336, 667; r5 336, 667; r6 337, 667; r7 337, 667; r8 337, 667; r9 337, 667; discharges without a median: 0, 0, 0, 0, 0, 0, 0, 0.

Odds weighting: effective number of rows 11,566 (share of the rows with a window 0.062, mean over repeats). Per repeat: r2 11,519 (0.061), clip 1.420, unusable rows in the top fifth 0.970, rows without a window 4,520; r3 11,574 (0.062), clip 1.422, unusable rows in the top fifth 0.970, rows without a window 4,517; r4 11,473 (0.061), clip 1.443, unusable rows in the top fifth 0.970, rows without a window 4,522; r5 11,456 (0.061), clip 1.431, unusable rows in the top fifth 0.970, rows without a window 4,488; r6 11,710 (0.062), clip 1.375, unusable rows in the top fifth 0.969, rows without a window 4,490; r7 11,548 (0.062), clip 1.417, unusable rows in the top fifth 0.970, rows without a window 4,501; r8 11,590 (0.062), clip 1.400, unusable rows in the top fifth 0.969, rows without a window 4,534; r9 11,660 (0.062), clip 1.381, unusable rows in the top fifth 0.970, rows without a window 4,482.

Rows whose age could not be binned (rounded age below 10 ms) per repeat: 0, 0, 0, 0, 0, 0, 0, 0.

## Expectations of the registration (section 6) beside the findings

| expectation | observed (point; verdict) | range met | verdict met |
|---|---|---|---|
| A1: ratio 0.92 to 0.95, holds | +0.946; holds | yes | yes |
| A1-median: median ratio to the causal GP 0.99 to 1.05 | +1.042 | yes |  |
| B1: 0.62 to 0.70, holds | +0.684; holds | yes | yes |
| B2: 0.58 to 0.68, holds | +0.661; holds | yes | yes |
| top10: share of the ten discharges with the largest reduction 0.25 to 0.45 | +0.214 | no |  |
| without10: skill over the causal GP without them +0.04 to +0.07, interval above zero | +0.076 [+0.050, +0.098] | no | yes |
| C1: +0.03 to +0.07, holds | +0.092; holds | no | yes |
| C2: +0.03 to +0.07, holds | +0.096; holds | no | yes |
| C3: share 0.35 to 0.55, does not hold | +0.463; does not hold | yes | yes |
| C4: 0.60 to 0.80, holds | +0.874; holds | no | yes |
| D1: +0.05 to +0.12 in each third, holds | low +0.066, middle +0.091, high +0.110; statement holds | yes | yes |
| D2: +0.02 to +0.08, holds in the hottest and one other third, the statement does not hold | low +0.007, middle +0.042, high +0.092; holds in: middle, high; statement does not hold | no | yes |
| E1: +0.06 to +0.11, holds | +0.069; holds | yes | yes |
| E2: -0.03 to +0.08, does not hold | +0.054; does not hold | yes | yes |
| F1: +0.07 to +0.12, holds | +0.087; holds | yes | yes |
| F2: +0.04 to +0.08, holds | +0.037; holds | no | yes |
| F3: +0.03 to +0.10, holds | +0.074; holds | yes | yes |
| F4: +0.01 to +0.06, does not hold | +0.020; does not hold | yes | yes |
| G1: +0.15 to +0.25, holds | +0.151; holds | yes | yes |
| H1: +0.02 to +0.06, holds | +0.054; holds | yes | yes |
| H2: 0.55 to 0.80, holds | +0.874; holds | no | yes |

## Addendum 1 and deviations

The first run stopped at check (b), read as 639 discharges with scored rows in every repeat (635, 634, 633 were found); checks (a), (c), (d), (e) had passed and no statement had been read. Addendum 1 of the registration (committed before any statement was read) fixes check (b) on the fold manifests plus no discharge with scored rows in two folds, and records the implementation choices below as the registered ones. No other deviation.

## Choices made where the registration left room

- Rows: every T_i row of `comparison_errors_test.npz` of the 80 pooled runs; the five folds of a repeat are concatenated. The squared errors of the estimator, the twin, hold (`se_persistence`), the causal GP (`se_gp_causal`, the eval_seq computation) and the offline GP (`se_gp`) are the stored ones.
- Location: `CES_TI_shot` indexes `dataset.valid_files` of eval_seq, i.e. the sorted grid files with at least one finite target (the convention of `run_b45.main`; it is a global file index, not an index within the fold); `CES_TI_row` is mapped to the grid row with `run_b45.compact_index`. The last visible value and its age: `run_b45.hold_compacted` (T_i of the previous kept row in the same block) and the time between the two grid rows.
- Discharge unit of every interval: the data file (grid file index).
- Value-rule flags: `run_b41.plain_mask` (the four rules of sensitivity_b30 as re-read by B.48) called on the located rows with y = the value in the file and hold = the value located above. The registration names `b42_delay/run_b42.py`; the flags read by B.48 (statements G1, G2 there) are those of `run_b41.plain_mask`, and B.42 contributes the definition of the last available value (`hold` = last earlier measured value of the block), which is what is located here.
- Classes: `run_b45.Classes.of_file` unchanged (class 6 = quiet; group 7 = departure not defined, in no statement); shares of the reduction are over all rows, group 7 included (as run_b45).
- Rows 1 to 3 after an absent frame: the event rule and row walk of `run_b41.part_b`, copied unchanged; a row enters once.
- Thirds: `run_b44.discharge_quantities` (median measured T_i of the file) and `run_b44.split_thirds`, within each repeat over its discharges.
- Risk: `run_b50.propensity` called unchanged per fold (its cache pointed at data/.b54/prop) through a `run_b37.Ctx` whose manifest returns the fold manifest. It fits on every file that is not a test file of the fold (train and validation files), as run_b50 does on the four splits; the registration's 'training discharges' is read as that. Rows without a ten-row window (rows 2 to 10 of a segment) have no p and are outside F (B.50 Addendum 1). Fifth edges, the 99th-percentile clip of the odds and the effective number of rows are computed within a repeat over its rows with a p (as within a split in B.50); the p of each row comes from the model of its own fold.
- Per-discharge shares: `run_b52.part_d` (at least 50 scored rows; mean squared error per discharge; ties count as not lower). Concentration: per repeat over all its discharges (as part_d), mean over repeats; the twentieth = round(discharges / 20) = 32 discharges. 'Those ten removed': the ten discharges with the largest reduction against the causal GP of each repeat are removed for both skills (over the causal GP and over the twin); the interval resamples the remaining discharges.
- Age bins: `run_b52` part G (age rounded to 10 ms: 20 ms or more = rounded age >= 2; older than 50 ms = > 5).
- Intervals: the two-level bootstrap of `analyze_b39.arm_p` (`hier`: a repeat drawn uniformly, then its discharges resampled with replacement), generalised to any ratio of per-discharge sums; 20,000 draws for skills, 2,000 for percentile ratios and shares (the per-discharge shares of B1, B2 included). Percentile ratios use `run_b41.wquant` with the discharge multiplicities as row weights. Point = mean over the eight repeats of the per-repeat value. One generator per quantity: numpy default_rng([20261154, crc32(name)]).
- A statement holds when its interval excludes the reference on the stated side; D1 and D2 when that is so in each of the three thirds.
- Check (a): P1 and P2 recomputed from the cached stored errors with analyze_b39's procedure and its generator (default_rng(2728), P1 then P2); the point must equal to three decimals (the interval is reported beside it).
- Check (b) (Addendum 1): by the fold manifests each of the 639 discharges is in exactly one test fold of every repeat, and no discharge has scored rows in two folds of a repeat (from the stored discharge indices); the resampled units are the discharges with scored rows (633 to 635 per repeat). Check (c): stored discharge and row indices identical between the estimator's and the twin's run of a fold, and their stored reference errors equal. Check (d): the grid T_i at the located row equals `CES_TI_y_true` within 0.01 eV on every row, no row is the first or last row of its file, every discharge is a test file of its fold; the hold located in the file is compared with sqrt(`se_persistence`) and reported.
- Check (e): flags on the k = 1 rows of the B.48 table of split 42 (population of run_b41) against b48_reread.json (flagged share; increment over the twin in the rows not flagged); classes through the pooled location code on the stored rows of `.b39_v2_cut_s42_i42`, which are the rows B.45 read (share of the quiet rows against b45_ablation_rows.json; no quiet share is recorded on the B.48 table rows); risk: propensity refitted here for split 42 and the top fifth of the B.48 rows against b50_missing_like.json (increment over the twin). Third, blip-row, age and per-discharge code are compared on split 42 as well and reported.

## Limits

The pooled folds re-partition the discharges on which the recurrent design was adopted; no untouched hold-out exists. One training per fold: the two-level interval covers the difference between re-partitions (in which the initialisation changes too), not a spread between initialisations on a fixed fold. The classes are defined from BES and the value rules are thresholds. Removing the discharges with the largest reduction selects on the outcome.

Run time: stages {'files': 11.8, 'rows': 14.2, 'prop': 78.7, 'checks': 6.6, 'readout': 42.0}; readout 54 s.
