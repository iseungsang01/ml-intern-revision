# B.50 -- the skill in scored rows that resemble the rows without a usable T_i (registered: ces_prediction/experiments/PREREGISTRATION_B50.md)

CPU, no network trained or run. T_i, cut population, the four file-level splits (42, 1, 7, 123), test discharges. Rows: the k = 1 rows of the B.48 row table (`data/.b48/rows_cut_s{seed}.npz`) in which the estimator of B.39 (`causal`), the history-only twin (`hist80`), hold and the causal GP are finite. Propensity p: the full model of B.37 F (probability that T_i is not usable in the row, from the history flags and the fast-channel window), fitted on the files that are not test files of the split. Skill of the estimator over R = 1 - SSE_est / SSE_R on the same rows (weighted: 1 - sum w*se_est / sum w*se_R). Intervals: 95 %, 2,000 resamples of the test discharges; strata and weights fixed on the full test set. A statement holds on a split when the lower end of its interval is above zero, and holds when that is so on at least three of four splits.

## Checks before the readout

| split | (a) AUC of p, test files (mine / B.37 F) | (a) | join: scored rows, grid T_i vs y max abs diff (eV), rows > 1e-3 | (b) scored rows with a ten-row window: with a p / all (without p) | (b) scored rows without the window (outside the population; share; discharges) | (b) | unstratified skill on every scored row vs B.48 (twin / GP / hold, mine = B.48) |
|---|---|---|---|---|---|---|---|
| 42 | 0.9739 / 0.9739 (+0.0000) | passes | 32,921, 0.0e+00, 0 | 32,127 / 32,127 (0) | 794 (0.0241; 94) | passes | +0.0649 = +0.0649 / +0.0747 = +0.0747 / +0.3596 = +0.3596 |
| 1 | 0.9615 / 0.9615 (+0.0000) | passes | 35,936, 0.0e+00, 0 | 35,138 / 35,138 (0) | 798 (0.0222; 94) | passes | +0.0277 = +0.0277 / +0.0961 = +0.0961 / +0.3851 = +0.3851 |
| 7 | 0.9621 / 0.9621 (+0.0000) | passes | 36,196, 0.0e+00, 0 | 35,357 / 35,357 (0) | 839 (0.0232; 96) | passes | +0.1160 = +0.1160 / +0.1324 = +0.1324 / +0.4006 = +0.4006 |
| 123 | 0.9629 / 0.9629 (+0.0000) | passes | 34,428, 0.0e+00, 0 | 33,662 / 33,662 (0) | 766 (0.0222; 94) | passes | +0.0568 = +0.0568 / +0.0953 = +0.0953 / +0.4067 = +0.4067 |

All checks pass on every split. Scored rows marked unusable in the propensity table (must be 0): 0, 0, 0, 0.

Unstratified skill (point values) on the population of this readout (scored rows from the eleventh row of a segment on) against the B.48 value on every scored row (twin / causal GP / hold; population minus B.48): split 42: +0.0683 vs +0.0649 (+0.0034) / +0.0759 vs +0.0747 (+0.0012) / +0.3624 vs +0.3596 (+0.0028); split 1: +0.0272 vs +0.0277 (-0.0005) / +0.0907 vs +0.0961 (-0.0053) / +0.3850 vs +0.3851 (-0.0001); split 7: +0.1162 vs +0.1160 (+0.0002) / +0.1311 vs +0.1324 (-0.0013) / +0.4030 vs +0.4006 (+0.0024); split 123: +0.0556 vs +0.0568 (-0.0011) / +0.0911 vs +0.0953 (-0.0042) / +0.4068 vs +0.4067 (+0.0001).

## Registered statements

Per-split values in the order 42, 1, 7, 123.

- **P1** (top fifth of p, skill over the causal GP, interval above zero): +0.067 [+0.003, +0.131]; +0.105 [+0.067, +0.143]; +0.143 [+0.094, +0.195]; +0.091 [+0.038, +0.146]; 4 of 4 -> **holds**.
- **P2** (top fifth of p, skill over the twin, interval above zero): +0.064 [+0.019, +0.119]; +0.039 [+0.007, +0.074]; +0.112 [+0.084, +0.137]; +0.054 [+0.010, +0.104]; 4 of 4 -> **holds**.
- **P3** (rows weighted by the odds of being unusable, skill over the causal GP, interval above zero): -0.009 [-0.151, +0.120]; +0.102 [+0.042, +0.164]; +0.158 [+0.102, +0.218]; +0.038 [-0.036, +0.114]; 2 of 4 -> **does not hold**.
- **P4** (the same over the twin): +0.005 [-0.040, +0.062]; +0.029 [-0.016, +0.075]; +0.095 [+0.057, +0.123]; +0.018 [-0.025, +0.062]; 1 of 4 -> **does not hold**.

Outcome under section 5 of the registration: P1 or P3 does not hold: stated as a limit of the claim.

## Reported

### Skill by fifth of p

| split | fifth | p range | rows | discharges | share of the unusable test rows | over causal GP | over twin | over hold |
|---|---|---|---|---|---|---|---|---|
| 42 | 1 | 0.0006 to 0.0020 | 6,426 | 51 | 0.002 | +0.053 [-0.121, +0.226] | +0.103 [-0.009, +0.240] | +0.381 [+0.244, +0.567] |
| 42 | 2 | 0.0020 to 0.0029 | 6,425 | 69 | 0.007 | +0.098 [+0.052, +0.142] | +0.108 [+0.051, +0.162] | +0.390 [+0.312, +0.453] |
| 42 | 3 | 0.0029 to 0.0045 | 6,426 | 77 | 0.005 | +0.106 [+0.038, +0.160] | +0.057 [-0.012, +0.115] | +0.405 [+0.328, +0.466] |
| 42 | 4 | 0.0045 to 0.0098 | 6,424 | 83 | 0.013 | +0.078 [+0.029, +0.137] | +0.061 [+0.003, +0.125] | +0.342 [+0.244, +0.429] |
| 42 | 5 | 0.0098 to 0.9734 | 6,426 | 91 | 0.973 | +0.067 [+0.003, +0.131] | +0.064 [+0.019, +0.119] | +0.353 [+0.289, +0.405] |
| 42 | p > median p of the unusable rows (0.7109) | | 186 (share 0.0058) | 42 | | -0.021 [-0.334, +0.227] | +0.022 [-0.045, +0.125] | +0.318 [+0.153, +0.467] |
| 1 | 1 | 0.0007 to 0.0018 | 7,028 | 45 | 0.003 | +0.040 [-0.068, +0.139] | +0.037 [-0.025, +0.100] | +0.245 [+0.113, +0.338] |
| 1 | 2 | 0.0018 to 0.0033 | 7,027 | 69 | 0.004 | +0.102 [+0.048, +0.167] | +0.027 [-0.010, +0.063] | +0.392 [+0.307, +0.458] |
| 1 | 3 | 0.0033 to 0.0060 | 7,028 | 83 | 0.009 | +0.073 [+0.025, +0.118] | +0.002 [-0.051, +0.043] | +0.334 [+0.261, +0.401] |
| 1 | 4 | 0.0060 to 0.0170 | 7,027 | 91 | 0.026 | +0.053 [-0.007, +0.110] | +0.006 [-0.035, +0.036] | +0.382 [+0.310, +0.439] |
| 1 | 5 | 0.0170 to 0.9641 | 7,028 | 92 | 0.958 | +0.105 [+0.067, +0.143] | +0.039 [+0.007, +0.074] | +0.399 [+0.359, +0.430] |
| 1 | p > median p of the unusable rows (0.6241) | | 339 (share 0.0096) | 50 | | +0.134 [+0.018, +0.230] | +0.018 [-0.079, +0.109] | +0.330 [+0.246, +0.399] |
| 7 | 1 | 0.0010 to 0.0020 | 7,072 | 50 | 0.001 | +0.145 [+0.086, +0.198] | +0.145 [+0.090, +0.198] | +0.458 [+0.345, +0.569] |
| 7 | 2 | 0.0020 to 0.0029 | 7,071 | 61 | 0.003 | +0.105 [+0.059, +0.158] | +0.129 [+0.085, +0.185] | +0.416 [+0.331, +0.487] |
| 7 | 3 | 0.0029 to 0.0056 | 7,071 | 84 | 0.006 | +0.127 [+0.065, +0.187] | +0.082 [+0.036, +0.130] | +0.337 [+0.243, +0.406] |
| 7 | 4 | 0.0056 to 0.0177 | 7,071 | 90 | 0.032 | +0.092 [+0.022, +0.149] | +0.138 [+0.075, +0.198] | +0.404 [+0.325, +0.462] |
| 7 | 5 | 0.0177 to 0.9589 | 7,072 | 90 | 0.958 | +0.143 [+0.094, +0.195] | +0.112 [+0.084, +0.137] | +0.407 [+0.357, +0.449] |
| 7 | p > median p of the unusable rows (0.6150) | | 383 (share 0.0108) | 62 | | +0.159 [+0.071, +0.249] | +0.075 [+0.012, +0.127] | +0.337 [+0.242, +0.408] |
| 123 | 1 | 0.0001 to 0.0017 | 6,733 | 42 | 0.001 | +0.043 [-0.056, +0.125] | +0.043 [-0.033, +0.110] | +0.241 [+0.104, +0.341] |
| 123 | 2 | 0.0017 to 0.0026 | 6,732 | 62 | 0.004 | +0.082 [+0.022, +0.153] | +0.000 [-0.026, +0.025] | +0.310 [+0.246, +0.366] |
| 123 | 3 | 0.0026 to 0.0050 | 6,732 | 76 | 0.007 | +0.029 [-0.041, +0.103] | -0.008 [-0.061, +0.032] | +0.363 [+0.266, +0.435] |
| 123 | 4 | 0.0050 to 0.0163 | 6,732 | 86 | 0.017 | +0.133 [+0.069, +0.197] | +0.121 [+0.070, +0.180] | +0.401 [+0.333, +0.461] |
| 123 | 5 | 0.0163 to 0.9603 | 6,733 | 91 | 0.972 | +0.091 [+0.038, +0.146] | +0.054 [+0.010, +0.104] | +0.423 [+0.365, +0.462] |
| 123 | p > median p of the unusable rows (0.5806) | | 541 (share 0.0161) | 52 | | +0.020 [-0.093, +0.143] | +0.003 [-0.065, +0.061] | +0.312 [+0.199, +0.391] |

### Rows of each fifth

| split | fifth | median age of the last visible value (ms) | median T_i (eV) | RMSE estimator | RMSE twin | RMSE causal GP | RMSE hold |
|---|---|---|---|---|---|---|---|
| 42 | 1 | 10.0 | 666.3 | 70.3 | 74.2 | 72.2 | 89.3 |
| 42 | 2 | 10.0 | 738.1 | 112.4 | 118.9 | 118.3 | 143.8 |
| 42 | 3 | 10.0 | 691.0 | 124.7 | 128.4 | 131.9 | 161.6 |
| 42 | 4 | 10.0 | 666.8 | 144.1 | 148.7 | 150.1 | 177.6 |
| 42 | 5 | 10.0 | 463.3 | 282.5 | 292.0 | 292.5 | 351.2 |
| 42 | above median p of unusable | 60.0 | 130.6 | 717.2 | 725.3 | 709.8 | 868.7 |
| 1 | 1 | 10.0 | 530.5 | 53.7 | 54.8 | 54.8 | 61.8 |
| 1 | 2 | 10.0 | 572.5 | 115.7 | 117.2 | 122.0 | 148.3 |
| 1 | 3 | 10.0 | 698.2 | 173.4 | 173.6 | 180.0 | 212.4 |
| 1 | 4 | 10.0 | 628.0 | 185.3 | 185.8 | 190.4 | 235.8 |
| 1 | 5 | 10.0 | 402.5 | 346.1 | 353.1 | 365.9 | 446.6 |
| 1 | above median p of unusable | 30.0 | 227.3 | 645.2 | 651.0 | 693.3 | 788.4 |
| 7 | 1 | 10.0 | 642.0 | 67.2 | 72.7 | 72.7 | 91.3 |
| 7 | 2 | 10.0 | 697.4 | 90.6 | 97.0 | 95.7 | 118.5 |
| 7 | 3 | 10.0 | 542.2 | 121.7 | 127.0 | 130.3 | 149.4 |
| 7 | 4 | 10.0 | 531.5 | 174.6 | 188.0 | 183.2 | 226.1 |
| 7 | 5 | 10.0 | 355.6 | 326.3 | 346.3 | 352.5 | 423.9 |
| 7 | above median p of unusable | 30.0 | 211.3 | 656.2 | 682.2 | 715.6 | 805.6 |
| 123 | 1 | 10.0 | 571.7 | 54.2 | 55.4 | 55.4 | 62.2 |
| 123 | 2 | 10.0 | 588.9 | 115.2 | 115.2 | 120.2 | 138.6 |
| 123 | 3 | 10.0 | 703.0 | 143.3 | 142.7 | 145.4 | 179.5 |
| 123 | 4 | 10.0 | 745.2 | 178.0 | 189.8 | 191.1 | 229.9 |
| 123 | 5 | 10.0 | 359.1 | 385.0 | 395.8 | 403.9 | 507.0 |
| 123 | above median p of unusable | 20.0 | 244.5 | 676.0 | 677.0 | 682.9 | 815.1 |

### Odds-weighted population

| split | scored rows | unusable test rows | median p scored / unusable | clip of w | rows clipped | effective rows (share) | over causal GP | over twin | over hold | unweighted, same rows: over causal GP / twin |
|---|---|---|---|---|---|---|---|---|---|---|
| 42 | 32,127 | 2,372 | 0.0035 / 0.7109 | 1.577 | 322 | 1472 (0.046) | -0.009 [-0.151, +0.120] | +0.005 [-0.040, +0.062] | +0.311 [+0.194, +0.392] | +0.076 [+0.032, +0.118] / +0.068 [+0.035, +0.105] |
| 1 | 35,138 | 2,999 | 0.0043 / 0.6241 | 1.613 | 352 | 2247 (0.064) | +0.102 [+0.042, +0.164] | +0.029 [-0.016, +0.075] | +0.381 [+0.338, +0.416] | +0.091 [+0.062, +0.117] / +0.027 [+0.003, +0.049] |
| 7 | 35,357 | 2,689 | 0.0037 / 0.6150 | 1.687 | 354 | 2283 (0.065) | +0.158 [+0.102, +0.218] | +0.095 [+0.057, +0.123] | +0.382 [+0.333, +0.419] | +0.131 [+0.095, +0.168] / +0.116 [+0.089, +0.141] |
| 123 | 33,662 | 3,024 | 0.0035 / 0.5806 | 1.936 | 337 | 2090 (0.062) | +0.038 [-0.036, +0.114] | +0.018 [-0.025, +0.062] | +0.344 [+0.275, +0.385] | +0.091 [+0.049, +0.129] / +0.056 [+0.020, +0.093] |

## Expectations of the registration (section 6)

| expectation | observed | met |
|---|---|---|
| P1 holds | 4 of 4 | yes |
| top fifth, skill over the causal GP +0.03 to +0.15 | +0.067, +0.105, +0.143, +0.091 | yes |
| top fifth, increment over the twin -0.02 to +0.08 | +0.064, +0.039, +0.112, +0.054 | no |
| P2: interval above zero on one to three splits | 4 of 4 | no |
| odds-weighted skill over the causal GP +0.03 to +0.15 | -0.009, +0.102, +0.158, +0.038 | no |
| P3 holds | 2 of 4 | no |
| odds-weighted increment over the twin 0.00 to +0.08 | +0.005, +0.029, +0.095, +0.018 | no |
| top fifth holds 0.70 to 0.95 of the unusable test rows | +0.973, +0.958, +0.958, +0.972 | no |
| effective number of rows 3 to 15 % of the scored rows | +0.046, +0.064, +0.065, +0.062 | yes |

3 of 9 expectations met.

## Choices made where the registration left room

- Propensity: the lines of `run_b37.part_f` that build the inputs and fit the full model are copied unchanged (same feature construction through `run_b37.Ctx.feats`, same target `isnan(T_i)` on the cut, held-free grid, `HistGradientBoostingClassifier(random_state=20261116)`, fitted on the rows of the non-test files); the only addition is that the grid row of every test row is kept. The history-only model of B.37 F is not fitted (it is not part of B.50).
- Join: a scored row of the B.48 table (k = 1; causal, twin, hold, causal GP and y finite) is matched to the propensity table by (file index into the sorted grid, grid row). Before the readout, the grid T_i at that (file, row) is compared with `y` of the table on every scored row (tolerance 1e-3 eV); the run stops if one row differs.
- Fifths: edges are the 20, 40, 60 and 80 % quantiles (linear interpolation) of p over the scored rows with a p; fifth j holds the rows with e_(j-1) < p <= e_j. The same edges place the unusable test rows into fifths.
- Unusable test rows: rows of the test files with a ten-row window whose T_i is not usable (the target of the propensity model, i.e. NaN after the 3 keV cut and the removal of held values).
- Weights: w = p / (1 - p) clipped at the 99th percentile of w over the scored rows of the split.
- Bootstrap: 2,000 multinomial resamples of the test discharges from `numpy.random.default_rng(20261118 + split)` (as `run_b45`), one set per split shared by every cell; per-discharge sums of (weighted) squared errors; interval = 2.5 and 97.5 % quantiles (`run_b45.ci`). A statement holds on a split when the lower end is above zero.
- Cross-check before the readout: the unstratified skills (estimator over the twin, the causal GP and hold) on every scored k = 1 row equal the B.48 values (`docs/paper/b48_reread.json`, B.44 overall) to three decimals (stop rule); the same skills on the population (rows with a ten-row window) are reported beside them.
- Position in segment: segments as in `run_b37.Ctx` (a new segment after a time step of at least 0.5 s); a row has a ten-row window when it is at position 10 or later (0-based) in its segment, as `run_b33.window_feats` requires.

## Deviations

Addendum 1 of the registration (2026-11-18, committed in f276233 before any readout was computed): the first run stopped at check (b) on all four splits (0.9759, 0.9778, 0.9768, 0.9778 of the scored rows received a p, against the threshold of 0.98); every row without a p was row 2 to 10 of its segment, where the inputs of B.37 F have no ten-row window. Check (b) is restated: every scored row with ten rows before it in its segment receives a p (the run stops otherwise); the scored rows without such a window are outside the population and are counted. No other change from the registered definitions, rows, intervals, seeds or rules. The cross-check against B.48 is made on every scored row (the same row set as B.48, a stop rule); the unstratified skill on the population of this readout is reported beside it.

## Limits

Resemblance is measured in the inputs of the propensity model only; the unusable rows may differ from the scored rows in ways no input records (a larger fit error, for one). The weights concentrate on few rows. One initialisation (the split seed). The test rows are those of every earlier readout. (Addendum 1) The first rows after a break of at least 0.5 s (rows 2 to 10 of a segment, which have no ten-row window) are not covered.

Run time: 18 s.
