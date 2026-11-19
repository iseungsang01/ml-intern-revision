# B.72 — the D1 result under other validation conventions

Registered: ces_prediction/experiments/PREREGISTRATION_B72.md (with ces_prediction/experiments/PREREGISTRATION_B72_ADDENDUM1.md). CPU only: the stored D1 estimates and squared errors of B.70 (T_i, pooled r2-r4, 222,672 rows per re-partition, 635 discharges) are re-read; nothing is trained and no network is run. Skill of A over B = 1 - (sum of squared errors of A) / (that of B) = the MSE reduction. S(m, d) = the sum of squared errors of estimator m over the D1 rows of discharge d, mean over r2-r4. 20,000 draws for every bootstrap; 95 % intervals as [2.5, 97.5] percentiles unless stated. Filters KF0 and KFX: included (B.70 phase 2 stored their estimates on the D1 rows; every fold maps exactly).

## Statements

| statement | value | verdict | registered expectation |
|---|---|---|---|
| **V1** each of D1-1 to D1-4 keeps its verdict under (b), (c), (d) and (e) | D1-1: (b) holds, (c) holds, (d) holds, (e) holds; D1-2: (b) holds, (c) holds, (d) holds, (e) holds; D1-3: (b) holds, (c) holds, (d) holds, (e) holds; D1-4: (b) holds, (c) holds, (d) holds, (e) holds | **holds** | holds for D1-1, D1-2, D1-4; for D1-3 holds under (b) and (c), uncertain under (d) and (e) |
| **V2** the 95 % model confidence set of set C is the two-stage estimator alone | set C, loss S(m, d): {two-stage estimator}; largest MCS p-value of the others 0.00e+00; loss S(m, d) / n(d): {two-stage estimator} | **holds** | holds |
| **V3** after Holm's correction every one of the seven remains below 0.05 | largest Holm-adjusted p 0.0246 (D1-6 (D1-2 form)) | **holds** | holds |
| **V4** the median per-discharge MSE reduction of the two-stage estimator lies above zero (interval), over the causal GP and over the two-stage twin | median rho(d) over the causal GP +0.150 [+0.130, +0.170] (holds); over the two-stage twin +0.072 [+0.058, +0.088] (holds) | **holds** | holds over the causal GP; weak expectation over the twin |
| **V5** no interval of (b), (c) or (d) is more than 1.25 times as wide as (a) for D1-1 to D1-4 | largest width ratio 1.63 (D1-4 d); D1-1 b 0.94, D1-1 c 0.94, D1-1 d 1.20, D1-2 b 0.92, D1-2 c 0.93, D1-2 d 1.07, D1-3 b 0.71, D1-3 c 0.71, D1-3 d 1.46, D1-4 b 0.89, D1-4 c 0.90, D1-4 d 1.63 | **does not hold** | holds for (b) and (c); uncertain for (d) |

## R1. Field metrics (all D1 rows; point = mean over r2-r4)

R² = 1 - SSE / SST, SST about the mean of the measured values (sample variance of T_i on the D1 rows); interval from the discharge bootstrap of R3 (b). Relative error = |estimate - measured| / measured. Murphy's terms (moments divided by N): R² = r² - conditional bias - mean bias; conditional bias = (r - s_f / s_o)²; mean bias = ((mean f - mean o) / s_o)²; slope = regression slope of measured on estimated (r s_o / s_f). Correlation and Murphy's terms need the estimate with its sign; for hold and the GP / PCHIP references only squared errors are stored (-).

| estimator | R² [95 %] | correlation r | mean relative error | median relative error | r² | conditional bias | mean bias | slope |
|---|---|---|---|---|---|---|---|---|
| two-stage estimator | 0.7899 [0.7679, 0.8099] | 0.8888 | 1.0328 | 0.0749 | 0.7900 | 0.00000 | 0.00000 | 0.999 |
| network | 0.7688 [0.7457, 0.7899] | 0.8768 | 1.0653 | 0.0833 | 0.7688 | 0.00001 | 0.00001 | 0.996 |
| history-only twin | 0.7560 [0.7320, 0.7777] | 0.8695 | 1.0629 | 0.0833 | 0.7561 | 0.00003 | 0.00009 | 0.996 |
| two-stage twin | 0.7674 [0.7441, 0.7886] | 0.8760 | 1.1115 | 0.0784 | 0.7674 | 0.00000 | 0.00000 | 0.999 |
| causal GP | 0.7448 [0.7192, 0.7681] | - | 1.0769 | 0.0812 | - | - | - | - |
| hold | 0.6284 [0.5874, 0.6655] | - | 1.0680 | 0.0873 | - | - | - | - |
| offline GP | 0.7757 [0.7516, 0.7976] | - | 0.9914 | 0.0697 | - | - | - | - |
| PCHIP | 0.7088 [0.6747, 0.7398] | - | 0.9872 | 0.0745 | - | - | - | - |
| KF0 | 0.7460 [0.7207, 0.7689] | 0.8664 | 0.9189 | 0.0821 | 0.7507 | 0.00290 | 0.00178 | 0.942 |
| KFX | 0.7585 [0.7335, 0.7813] | 0.8732 | 0.9243 | 0.0784 | 0.7624 | 0.00251 | 0.00134 | 0.946 |

## R2. One score per discharge

Discharges with at least 20 D1 rows: 602 of 635 (33 left out). rho(d) = 1 - S(A, d) / S(B, d), the MSE reduction of discharge d. Intervals: the eligible discharges resampled, 20,000 draws. Equal-weight skill = 1 - mean_d MSE_A(d) / mean_d MSE_B(d), MSE(d) = S(m, d) / n(d). Pooled skill = 1 - sum_d S(A, d) / sum_d S(B, d) over all 635 discharges (point, for comparison).

| A over B | median rho(d) [95 %] | quartiles of rho(d) | share of discharges with rho(d) > 0 [95 %] | equal-weight MSE skill [95 %] | equal-weight MSE reduction (%) | pooled skill (point) |
|---|---|---|---|---|---|---|
| two-stage estimator over causal GP | +0.150 [+0.130, +0.170] | +0.047 / +0.150 / +0.292 | 0.814 [0.782, 0.846] | +0.147 [+0.125, +0.170] | +14.7 [+12.5, +17.0] | +0.177 |
| two-stage estimator over two-stage twin | +0.072 [+0.058, +0.088] | -0.016 / +0.072 / +0.187 | 0.719 [0.683, 0.754] | +0.060 [+0.043, +0.078] | +6.0 [+4.3, +7.8] | +0.097 |
| two-stage estimator over network | +0.082 [+0.071, +0.090] | +0.026 / +0.082 / +0.163 | 0.852 [0.824, 0.880] | +0.074 [+0.061, +0.089] | +7.4 [+6.1, +8.9] | +0.092 |
| two-stage estimator over offline GP | -0.016 [-0.039, +0.005] | -0.240 / -0.016 / +0.120 | 0.467 [0.427, 0.507] | +0.039 [+0.016, +0.061] | +3.9 [+1.6, +6.1] | +0.064 |
| two-stage estimator over hold | +0.371 [+0.359, +0.388] | +0.230 / +0.371 / +0.470 | 0.890 [0.864, 0.915] | +0.427 [+0.407, +0.446] | +42.7 [+40.7, +44.6] | +0.435 |
| two-stage estimator over KFX | +0.101 [+0.086, +0.115] | +0.001 / +0.101 / +0.201 | 0.749 [0.714, 0.782] | +0.131 [+0.111, +0.150] | +13.1 [+11.1, +15.0] | +0.130 |
| network over history-only twin | +0.048 [+0.033, +0.059] | -0.033 / +0.048 / +0.147 | 0.658 [0.620, 0.696] | +0.028 [+0.009, +0.047] | +2.8 [+0.9, +4.7] | +0.052 |
| network over causal GP | +0.085 [+0.070, +0.103] | -0.050 / +0.085 / +0.210 | 0.683 [0.645, 0.719] | +0.079 [+0.054, +0.102] | +7.9 [+5.4, +10.2] | +0.094 |

## R3. The pooled MSE skill under five inference methods

(a) the two-level bootstrap of B.70, draw for draw; (b) discharges resampled on S(m, d); (c) D(d) = S(B, d) - S(A, d), t-test over the 635 discharges, skill interval by linearising the ratio; (d) the skill of each of the 15 folds, variance of the mean (1/15 + 1/4) s², Student t with 14 degrees of freedom; (e) Wilcoxon signed-rank on D(d). Verdict at 5 %: the interval lies above zero; for (e) p < 0.05 with the positive differences carrying the larger rank sum. Width relative to (a) = width of the interval / width of (a)'s.

| A over B (statement) | method | point | 95 % interval | p | width / (a) | verdict |
|---|---|---|---|---|---|---|
| two-stage estimator over causal GP (D1-1) | (a) two-level bootstrap (B.70) | +0.177 | [+0.156, +0.196] | - | 1.00 | holds |
| two-stage estimator over causal GP (D1-1) | (b) discharge bootstrap | +0.177 | [+0.158, +0.195] | - | 0.94 | holds |
| two-stage estimator over causal GP (D1-1) | (c) cluster-robust paired t | +0.177 | [+0.158, +0.196] | 2.39e-36 | 0.94 | holds |
| two-stage estimator over causal GP (D1-1) | (d) corrected resampled t (15 folds) | +0.177 | [+0.153, +0.201] | 2.47e-10 | 1.20 | holds |
| two-stage estimator over causal GP (D1-1) | (e) Wilcoxon signed-rank | T+ 179,789, T- 22,141 (share D(d) > 0: 0.786) | - | 3.86e-65 | - | holds |
| two-stage estimator over two-stage twin (D1-2) | (a) two-level bootstrap (B.70) | +0.097 | [+0.078, +0.116] | - | 1.00 | holds |
| two-stage estimator over two-stage twin (D1-2) | (b) discharge bootstrap | +0.097 | [+0.080, +0.114] | - | 0.92 | holds |
| two-stage estimator over two-stage twin (D1-2) | (c) cluster-robust paired t | +0.097 | [+0.079, +0.114] | 1.10e-19 | 0.93 | holds |
| two-stage estimator over two-stage twin (D1-2) | (d) corrected resampled t (15 folds) | +0.098 | [+0.077, +0.118] | 6.11e-08 | 1.07 | holds |
| two-stage estimator over two-stage twin (D1-2) | (e) Wilcoxon signed-rank | T+ 159,978, T- 41,952 (share D(d) > 0: 0.694) | - | 2.73e-37 | - | holds |
| network over history-only twin (D1-3) | (a) two-level bootstrap (B.70) | +0.052 | [+0.029, +0.074] | - | 1.00 | holds |
| network over history-only twin (D1-3) | (b) discharge bootstrap | +0.052 | [+0.037, +0.068] | - | 0.71 | holds |
| network over history-only twin (D1-3) | (c) cluster-robust paired t | +0.052 | [+0.036, +0.068] | 3.24e-09 | 0.71 | holds |
| network over history-only twin (D1-3) | (d) corrected resampled t (15 folds) | +0.053 | [+0.020, +0.086] | 0.0041 | 1.46 | holds |
| network over history-only twin (D1-3) | (e) Wilcoxon signed-rank | T+ 142,049, T- 59,881 (share D(d) > 0: 0.635) | - | 6.47e-19 | - | holds |
| network over causal GP (D1-4) | (a) two-level bootstrap (B.70) | +0.094 | [+0.071, +0.114] | - | 1.00 | holds |
| network over causal GP (D1-4) | (b) discharge bootstrap | +0.094 | [+0.074, +0.113] | - | 0.89 | holds |
| network over causal GP (D1-4) | (c) cluster-robust paired t | +0.094 | [+0.074, +0.113] | 8.17e-16 | 0.90 | holds |
| network over causal GP (D1-4) | (d) corrected resampled t (15 folds) | +0.094 | [+0.059, +0.129] | 5.44e-05 | 1.63 | holds |
| network over causal GP (D1-4) | (e) Wilcoxon signed-rank | T+ 151,468, T- 50,462 (share D(d) > 0: 0.658) | - | 9.22e-28 | - | holds |
| two-stage estimator over network | (a) two-level bootstrap (B.70) | +0.092 | [+0.078, +0.105] | - | 1.00 | holds |
| two-stage estimator over network | (b) discharge bootstrap | +0.092 | [+0.080, +0.103] | - | 0.83 | holds |
| two-stage estimator over network | (c) cluster-robust paired t | +0.092 | [+0.080, +0.103] | 2.92e-33 | 0.83 | holds |
| two-stage estimator over network | (d) corrected resampled t (15 folds) | +0.092 | [+0.071, +0.112] | 1.24e-07 | 1.44 | holds |
| two-stage estimator over network | (e) Wilcoxon signed-rank | T+ 170,964, T- 20,926 (share D(d) > 0: 0.819) | - | 9.81e-64 | - | holds |
| two-stage estimator over offline GP | (a) two-level bootstrap (B.70) | +0.064 | [+0.039, +0.086] | - | 1.00 | holds |
| two-stage estimator over offline GP | (b) discharge bootstrap | +0.064 | [+0.041, +0.085] | - | 0.94 | holds |
| two-stage estimator over offline GP | (c) cluster-robust paired t | +0.064 | [+0.041, +0.086] | 7.08e-07 | 0.94 | holds |
| two-stage estimator over offline GP | (d) corrected resampled t (15 folds) | +0.064 | [+0.033, +0.095] | 5.27e-04 | 1.29 | holds |
| two-stage estimator over offline GP | (e) Wilcoxon signed-rank | T+ 115,631, T- 86,299 (share D(d) > 0: 0.450) | - | 0.0015 | - | holds |

Fold skills of (d) (r2 f0-f4, r3 f0-f4, r4 f0-f4) and their sd:

- two-stage estimator over causal GP: +0.151, +0.206, +0.176, +0.164, +0.181, +0.192, +0.155, +0.209, +0.147, +0.203, +0.184, +0.184, +0.156, +0.176, +0.175; sd 0.020; without the correction the interval would be [+0.166, +0.188].
- two-stage estimator over two-stage twin: +0.087, +0.124, +0.098, +0.072, +0.097, +0.108, +0.071, +0.108, +0.082, +0.124, +0.106, +0.104, +0.085, +0.112, +0.086; sd 0.017; without the correction the interval would be [+0.088, +0.107].
- network over history-only twin: +0.026, +0.068, +0.040, +0.014, +0.080, +0.034, +0.014, +0.050, +0.061, +0.094, +0.035, +0.091, +0.029, +0.075, +0.077; sd 0.027; without the correction the interval would be [+0.037, +0.068].
- network over causal GP: +0.047, +0.130, +0.087, +0.078, +0.115, +0.091, +0.056, +0.131, +0.069, +0.130, +0.065, +0.121, +0.068, +0.103, +0.119; sd 0.029; without the correction the interval would be [+0.078, +0.110].
- two-stage estimator over network: +0.108, +0.087, +0.097, +0.094, +0.075, +0.111, +0.105, +0.089, +0.084, +0.084, +0.127, +0.071, +0.094, +0.082, +0.064; sd 0.017; without the correction the interval would be [+0.082, +0.101].
- two-stage estimator over offline GP: +0.034, +0.095, +0.049, +0.059, +0.071, +0.070, +0.033, +0.104, +0.039, +0.095, +0.091, +0.089, +0.036, +0.050, +0.044; sd 0.025; without the correction the interval would be [+0.050, +0.078].

## R4. Model confidence set (95 %)

Implementation: arch 9.0.0, MCS(size=0.05, reps=20000, block_size=1, method='R', bootstrap='stationary'); the stationary bootstrap with block length 1 resamples discharges independently. One row per discharge (635). Loss 1 = S(m, d); loss 2 = S(m, d) / n(d) (the per-discharge MSE). An estimator is in the set when its MCS p-value is at least 0.05; the MCS p-value is the p-value at which it leaves (running maximum over the eliminations; 1 for the last one left).

| estimator | set C, loss 1 | set C, loss 2 | set A, loss 1 | set A, loss 2 |
|---|---|---|---|---|
| two-stage estimator | **in the set** (p 1.0000) | **in the set** (p 1.0000) | **in the set** (p 1.0000) | **in the set** (p 1.0000) |
| network | 0.00e+00 (leaves 3 of 8) | 0.00e+00 (leaves 4 of 8) | 0.00e+00 (leaves 4 of 10) | 0.00e+00 (leaves 4 of 10) |
| history-only twin | 0.00e+00 (leaves 5 of 8) | 0.00e+00 (leaves 6 of 8) | 0.00e+00 (leaves 6 of 10) | 0.00e+00 (leaves 6 of 10) |
| two-stage twin | 0.00e+00 (leaves 7 of 8) | 0.00e+00 (leaves 7 of 8) | 0.00e+00 (leaves 8 of 10) | 0.00e+00 (leaves 8 of 10) |
| causal GP | 0.00e+00 (leaves 2 of 8) | 0.00e+00 (leaves 3 of 8) | 0.00e+00 (leaves 2 of 10) | 0.00e+00 (leaves 2 of 10) |
| hold | 0.00e+00 (leaves 1 of 8) | 0.00e+00 (leaves 1 of 8) | 0.00e+00 (leaves 1 of 10) | 0.00e+00 (leaves 1 of 10) |
| KF0 | 0.00e+00 (leaves 4 of 8) | 0.00e+00 (leaves 2 of 8) | 0.00e+00 (leaves 3 of 10) | 0.00e+00 (leaves 3 of 10) |
| KFX | 0.00e+00 (leaves 6 of 8) | 0.00e+00 (leaves 5 of 8) | 0.00e+00 (leaves 7 of 10) | 0.00e+00 (leaves 5 of 10) |
| offline GP | - | - | 0.00e+00 (leaves 9 of 10) | 0.0032 (leaves 9 of 10) |
| PCHIP | - | - | 0.00e+00 (leaves 5 of 10) | 0.00e+00 (leaves 7 of 10) |

- set C, loss 1: members {two-stage estimator}; elimination order hold -> causal GP -> network -> KF0 -> history-only twin -> KFX -> two-stage twin -> two-stage estimator.

- set C, loss 2: members {two-stage estimator}; elimination order hold -> KF0 -> causal GP -> network -> KFX -> history-only twin -> two-stage twin -> two-stage estimator.

- set A, loss 1: members {two-stage estimator}; elimination order hold -> causal GP -> KF0 -> network -> PCHIP -> history-only twin -> KFX -> two-stage twin -> offline GP -> two-stage estimator.

- set A, loss 2: members {two-stage estimator}; elimination order hold -> causal GP -> KF0 -> network -> KFX -> history-only twin -> PCHIP -> two-stage twin -> offline GP -> two-stage estimator.

## R5. The seven D1 statements: bootstrap p-values and Holm's correction

Two-sided p = 2 x the smaller share of the two-level draws (B.70's draws, reproduced) on either side of the null value, not below 1 / 20,001. Holm over the seven.

| statement | form | value [95 %] | null | raw p | Holm-adjusted p | adjusted p < 0.05 |
|---|---|---|---|---|---|---|
| D1-1 | two-stage estimator over causal GP, MSE skill | +0.177 [+0.156, +0.196] | 0 | 5.00e-05 | 3.50e-04 | yes |
| D1-2 | two-stage estimator over two-stage twin, MSE skill | +0.097 [+0.078, +0.116] | 0 | 5.00e-05 | 3.50e-04 | yes |
| D1-3 | network over history-only twin, MSE skill | +0.052 [+0.029, +0.074] | 0 | 5.00e-05 | 3.50e-04 | yes |
| D1-4 | network over causal GP, MSE skill | +0.094 [+0.071, +0.114] | 0 | 5.00e-05 | 3.50e-04 | yes |
| D1-5 | median |e|, two-stage estimator / hold | 0.908 [0.887, 0.931] | 1 | 5.00e-05 | 3.50e-04 | yes |
| D1-6 (D1-1 form) | two-stage estimator over causal GP, MSE skill, rows the mixture does not flag | +0.207 [+0.153, +0.250] | 0 | 5.00e-05 | 3.50e-04 | yes |
| D1-6 (D1-2 form) | two-stage estimator over two-stage twin, MSE skill, rows the mixture does not flag | +0.065 [+0.010, +0.107] | 0 | 0.0246 | 0.0246 | yes |

## Checks

- (a) reproduces `docs/paper/b70_d1.json` (points and bounds, tolerance 0.001): passes; largest difference 0.0e+00.
- D1-5 and the two D1-6 forms reproduce `b70_d1.json` (tolerance 0.001): passes; D1-5 difference 0.0e+00, D1-6 differences D1-1 form 0.0e+00, D1-2 form 0.0e+00; flagged rows per re-partition r2 35,246 (B.70 35,246), r3 35,050 (B.70 35,050), r4 35,724 (B.70 35,724).
- Rows and measured values of the stage-2 files and the D1 re-scorings: same rows in 15 of 15 folds; largest |y difference| 0.0e+00 eV.
- Same D1 rows and discharges in r2, r3, r4: True / True; largest |y difference| across re-partitions 4.9e-04 eV; reference squared errors across re-partitions, largest difference: hold 2.8e+00, causal GP 2.5e+00, offline GP 2.6e+00, PCHIP 2.7e+00.
- Signed estimates against the squared errors of the statements, largest | |estimate - y| - sqrt(se) | (eV): r2 two-stage estimator 0.0e+00, network 2.3e-13, history-only twin 2.3e-13, two-stage twin 0.0e+00; r3 two-stage estimator 0.0e+00, network 2.3e-13, history-only twin 2.3e-13, two-stage twin 0.0e+00; r4 two-stage estimator 0.0e+00, network 2.3e-13, history-only twin 2.3e-13, two-stage twin 0.0e+00.
- Filters: 15 of 15 folds map exactly onto the D1 rows (same shot and row arrays, y within 1e-6 eV, every row located, every estimate finite) -> included.
- Murphy identity R² = r² - conditional bias - mean bias: largest deviation 2.2e-16.
- Smallest measured T_i on the D1 rows: 0.0 eV (relative errors defined).
- Run time 968 s (CPU, 2 threads).
