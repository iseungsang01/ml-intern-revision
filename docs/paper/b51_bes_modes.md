# B.51 -- the beam-emission array split into its common mode and its differences between channels (registered: ces_prediction/experiments/PREREGISTRATION_B51.md)

CPU; no network trained (part B fits gradient-boosted regressions). T_i, cut population. c = mean of the logarithms of the nine BES channels (common mode); g1, g2 = radial differences (column 06 - 05, 07 - 06), g = g1 + g2; h1, h2 = poloidal differences (row 02 - 01, 03 - 02), h = h1 + h2. Intervals: 95 %, 2,000 resamples of discharges. A statement of part B holds if its interval is above zero on at least three of the four splits.

## Registered statements

- **A1** (change of T_i against change of g, minus the shifted null): **+0.284 [+0.216, +0.336]** (309 discharges) -> **holds**.
- **A2** (the same for the common mode c): **+0.428 [+0.357, +0.471]** (309 discharges) -> **holds**.
- **A3** (the same for g with the change of c removed from both (partial rank correlation)): **+0.002 [-0.041, +0.038]** (309 discharges) -> **does not hold**.
- **B1** (model D (differences) over model 0, interval above zero): split 42 +0.044 [+0.010, +0.077]; split 1 +0.050 [+0.030, +0.071]; split 7 +0.044 [+0.028, +0.059]; split 123 +0.055 [+0.032, +0.077]; 4 of 4 -> **holds**.
- **B2** (model C (common mode) over model 0, interval above zero): split 42 +0.071 [+0.026, +0.113]; split 1 +0.056 [+0.033, +0.080]; split 7 +0.073 [+0.048, +0.097]; split 123 +0.060 [+0.034, +0.086]; 4 of 4 -> **holds**.

What goes into the paper (section 6): B1 holds: the caveat on the beam is restated with the differences between channels (A1 or A3 holds and is quoted with it). Control ratio 0.121 is not above 0.5.

## Part A

### Control at the isolated absent frames (reported)

Events of B.35: 1,366 in 498 discharges (b35_blip_response.json: 1,366, 498; equal: True); 1,366 with every component finite.

| events | n | median abs change c | g | h | ratio g / c | ratio h / c | median of the per-event ratio g / c, h / c | median signed change c | g | h |
|---|---|---|---|---|---|---|---|---|---|---|
| all | 1,366 | 0.424 [0.416, 0.435] | 0.051 [0.048, 0.055] | 0.028 [0.027, 0.031] | **0.121 [0.113, 0.129]** | 0.067 [0.062, 0.074] | 0.123, 0.072 | -0.423 [-0.435, -0.416] | -0.025 [-0.031, -0.021] | -0.021 [-0.023, -0.019] |
| one absent frame | 1,235 | 0.419 [0.412, 0.428] | 0.051 [0.048, 0.055] | 0.029 [0.027, 0.032] | **0.123 [0.115, 0.130]** | 0.069 [0.064, 0.076] | 0.123, 0.073 | -0.418 [-0.428, -0.412] | -0.026 [-0.032, -0.021] | -0.022 [-0.025, -0.020] |

### Association of 200 ms block changes with the change of T_i

309 discharges with at least ten pairs of blocks after the exclusion, 5,771 pairs (before the exclusion, all discharges: 10,423). Median over discharges of the Spearman correlation; null = one series shifted circularly by half its length.

| component | discharges | median rho | median null | difference | share rho > 0 |
|---|---|---|---|---|---|
| c (common mode) | 309 | +0.414 [+0.356, +0.436] | -0.014 [-0.042, +0.021] | **+0.428 [+0.357, +0.471]** | 0.867 |
| g (radial, 07 - 05) | 309 | +0.264 [+0.214, +0.292] | -0.021 [-0.057, +0.016] | **+0.284 [+0.216, +0.336]** | 0.764 |
| g1 (06 - 05) | 309 | +0.257 [+0.218, +0.313] | -0.013 [-0.053, +0.009] | **+0.270 [+0.222, +0.334]** | 0.783 |
| g2 (07 - 06) | 309 | +0.168 [+0.129, +0.227] | +0.018 [-0.031, +0.056] | **+0.150 [+0.094, +0.227]** | 0.725 |
| h1 + h2 (poloidal, 03 - 01) | 309 | -0.036 [-0.066, -0.011] | -0.014 [-0.050, +0.018] | **-0.022 [-0.069, +0.019]** | 0.405 |
| h1 (02 - 01) | 309 | -0.063 [-0.088, -0.028] | -0.011 [-0.049, +0.015] | **-0.052 [-0.089, -0.002]** | 0.388 |
| h2 (03 - 02) | 309 | -0.006 [-0.035, +0.031] | -0.009 [-0.050, +0.037] | **+0.003 [-0.055, +0.060]** | 0.492 |
| g after c removed (partial) | 309 | -0.009 [-0.037, +0.011] | -0.011 [-0.038, +0.022] | **+0.002 [-0.041, +0.038]** | 0.479 |
| c after g removed (partial) | 309 | +0.297 [+0.254, +0.348] | +0.004 [-0.029, +0.035] | **+0.292 [+0.244, +0.358]** | 0.828 |

By third of the discharge median T_i (difference to the null):

| component | low third | middle | high |
|---|---|---|---|
| c (common mode) | +0.238 [+0.109, +0.329] (103) | +0.505 [+0.443, +0.587] (103) | +0.417 [+0.330, +0.476] (103) |
| g (radial, 07 - 05) | +0.288 [+0.162, +0.366] (103) | +0.361 [+0.258, +0.462] (103) | +0.196 [+0.105, +0.303] (103) |
| g after c removed (partial) | +0.102 [+0.002, +0.234] (103) | -0.004 [-0.111, +0.057] (103) | -0.035 [-0.138, +0.017] (103) |
| h1 + h2 (poloidal, 03 - 01) | -0.030 [-0.118, +0.050] (103) | -0.022 [-0.097, +0.029] (103) | -0.029 [-0.122, +0.077] (103) |

Check: the B.43 read-out (ECEI level, no exclusion, at least ten pairs) recomputed with the same functions: 400 discharges, median rho +0.3215, null -0.0061 (B.43: 400, +0.3215, -0.0061); equal: True.

Rows with a non-positive BES channel: 117 of 245,925 grid rows (31 files); dropped.

## Part B

### Checks

| split | k = 1 rows | y vs grid T_i mismatches | rows recomputed | twin within 0.5 eV | hold | causal GP | max dev twin / hold / GP (eV) | scoring code: causal over twin, over GP (mine / B.48) | twin over GP (mine / B.42 L = 0 in B.48) |
|---|---|---|---|---|---|---|---|---|---|
| 42 | 32,921 | 0 | 32,921 | 1.00000 | 1.00000 | 1.00000 | 0.00e+00 / 0.00e+00 / 1.21e-04 | +0.0649 / +0.0649, +0.0747 / +0.0747 | +0.0105 / +0.0105 |
| 1 | 35,936 | 0 | 35,936 | 1.00000 | 1.00000 | 1.00000 | 0.00e+00 / 0.00e+00 / 1.22e-04 | +0.0277 / +0.0277, +0.0961 / +0.0961 | +0.0703 / +0.0703 |
| 7 | 36,196 | 0 | 36,196 | 1.00000 | 1.00000 | 1.00000 | 0.00e+00 / 0.00e+00 / 1.17e-04 | +0.1160 / +0.1160, +0.1324 / +0.1324 | +0.0186 / +0.0186 |
| 123 | 34,428 | 0 | 34,428 | 1.00000 | 1.00000 | 1.00000 | 0.00e+00 / 0.00e+00 / 1.21e-04 | +0.0568 / +0.0568, +0.0953 / +0.0953 | +0.0408 / +0.0408 |

### Rows

| split | training discharges | validation discharges / rows (all, used) | test rows (k = 1, used, dropped) | test discharges | quiet rows |
|---|---|---|---|---|---|
| 42 | 415 | 127 / 50,888, 49,706 | 32,921, 31,936, 985 | 89 | 29,124 |
| 1 | 414 | 128 / 47,979, 46,798 | 35,936, 35,037, 899 | 92 | 31,760 |
| 7 | 414 | 128 / 46,099, 44,860 | 36,196, 35,247, 949 | 93 | 32,152 |
| 123 | 414 | 128 / 47,289, 46,004 | 34,428, 33,571, 857 | 93 | 30,651 |

### Skills (all rows)

| split | D over 0 (B1) | C over 0 (B2) | CD over 0 | CD over C | CD over D | reduction of D / reduction of CD |
|---|---|---|---|---|---|---|
| 42 | +0.044 [+0.010, +0.077] | +0.071 [+0.026, +0.113] | +0.074 [+0.037, +0.111] | +0.004 [-0.016, +0.023] | +0.031 [+0.014, +0.049] | 0.600 [0.234, 0.798] |
| 1 | +0.050 [+0.030, +0.071] | +0.056 [+0.033, +0.080] | +0.071 [+0.046, +0.095] | +0.015 [+0.005, +0.026] | +0.022 [+0.008, +0.036] | 0.706 [0.540, 0.871] |
| 7 | +0.044 [+0.028, +0.059] | +0.073 [+0.048, +0.097] | +0.075 [+0.049, +0.101] | +0.003 [-0.013, +0.016] | +0.033 [+0.012, +0.055] | 0.582 [0.424, 0.794] |
| 123 | +0.055 [+0.032, +0.077] | +0.060 [+0.034, +0.086] | +0.071 [+0.044, +0.099] | +0.011 [-0.004, +0.024] | +0.017 [-0.004, +0.035] | 0.775 [0.577, 1.080] |

| split | 0 over twin | C over twin | D over twin | CD over twin | 0 over causal GP | C over GP | D over GP | CD over GP | twin over GP |
|---|---|---|---|---|---|---|---|---|---|
| 42 | +0.038 [+0.017, +0.059] | +0.106 [+0.054, +0.158] | +0.080 [+0.039, +0.121] | +0.109 [+0.064, +0.155] | +0.045 [+0.018, +0.071] | +0.112 [+0.067, +0.155] | +0.087 [+0.049, +0.124] | +0.115 [+0.072, +0.156] | +0.007 [-0.030, +0.046] |
| 1 | +0.016 [+0.000, +0.034] | +0.072 [+0.044, +0.101] | +0.065 [+0.043, +0.091] | +0.086 [+0.059, +0.116] | +0.079 [+0.054, +0.105] | +0.131 [+0.101, +0.164] | +0.125 [+0.095, +0.156] | +0.144 [+0.111, +0.180] | +0.064 [+0.034, +0.092] |
| 7 | +0.061 [+0.039, +0.083] | +0.129 [+0.099, +0.160] | +0.102 [+0.078, +0.126] | +0.131 [+0.101, +0.163] | +0.076 [+0.054, +0.100] | +0.144 [+0.109, +0.180] | +0.117 [+0.090, +0.143] | +0.146 [+0.110, +0.183] | +0.017 [-0.012, +0.044] |
| 123 | +0.020 [+0.003, +0.036] | +0.079 [+0.043, +0.114] | +0.073 [+0.043, +0.104] | +0.089 [+0.054, +0.123] | +0.056 [+0.021, +0.085] | +0.113 [+0.066, +0.153] | +0.107 [+0.066, +0.143] | +0.122 [+0.073, +0.165] | +0.037 [+0.002, +0.066] |

### Quiet rows (class 6 of B.45)

| split | CD over 0 | CD over C | CD over D | D over 0 | C over 0 | CD over twin | 0 over twin |
|---|---|---|---|---|---|---|---|
| 42 | +0.075 [+0.041, +0.113] | +0.007 [-0.008, +0.023] | +0.029 [+0.011, +0.048] | +0.048 [+0.021, +0.076] | +0.069 [+0.032, +0.109] | +0.101 [+0.061, +0.146] | +0.028 [+0.008, +0.048] |
| 1 | +0.070 [+0.043, +0.099] | +0.012 [+0.005, +0.021] | +0.021 [+0.006, +0.036] | +0.050 [+0.029, +0.073] | +0.058 [+0.032, +0.084] | +0.076 [+0.048, +0.107] | +0.007 [-0.008, +0.023] |
| 7 | +0.086 [+0.058, +0.115] | +0.009 [-0.003, +0.021] | +0.038 [+0.014, +0.064] | +0.050 [+0.036, +0.066] | +0.078 [+0.051, +0.103] | +0.152 [+0.118, +0.186] | +0.072 [+0.052, +0.096] |
| 123 | +0.078 [+0.052, +0.110] | +0.011 [-0.001, +0.022] | +0.025 [+0.011, +0.040] | +0.055 [+0.032, +0.081] | +0.068 [+0.042, +0.097] | +0.092 [+0.057, +0.127] | +0.014 [-0.001, +0.030] |

### For comparison: the BES network of B.39 over the twin on the rows both score

| split | rows both / harness rows | y agrees, stored twin error agrees (share within 0.5 eV) | BES network over twin | B.39 D2 (all harness rows) | 0 over twin | C over twin | D over twin | CD over twin |
|---|---|---|---|---|---|---|---|---|
| 42 | 31,617 / 32,589 | 1.0000, 0.9996 | +0.080 [+0.047, +0.117] | +0.074 [+0.039, +0.112] | +0.035 [+0.013, +0.057] | +0.110 [+0.056, +0.162] | +0.083 [+0.046, +0.122] | +0.111 [+0.071, +0.154] |
| 1 | 34,687 / 35,582 | 1.0000, 0.9999 | +0.025 [+0.005, +0.049] | +0.026 [+0.007, +0.047] | +0.010 [-0.005, +0.027] | +0.067 [+0.039, +0.098] | +0.060 [+0.039, +0.085] | +0.082 [+0.055, +0.111] |
| 7 | 34,825 / 35,759 | 1.0000, 1.0000 | +0.104 [+0.068, +0.133] | +0.105 [+0.072, +0.134] | +0.061 [+0.042, +0.080] | +0.132 [+0.102, +0.164] | +0.104 [+0.081, +0.125] | +0.135 [+0.105, +0.168] |
| 123 | 33,198 / 34,038 | 1.0000, 0.9984 | +0.063 [+0.035, +0.096] | +0.063 [+0.036, +0.095] | +0.019 [+0.002, +0.036] | +0.083 [+0.047, +0.118] | +0.075 [+0.044, +0.106] | +0.091 [+0.055, +0.127] |

RMSE (eV): split 42: 0 166.0, C 160.1, D 162.3, CD 159.8, twin 169.3, gp 169.9; split 1: 0 201.3, C 195.5, D 196.2, CD 194.1, twin 203.0, gp 209.8; split 7: 0 187.1, C 180.2, D 183.0, CD 180.0, twin 193.1, gp 194.7; split 123: 0 212.2, C 205.7, D 206.3, CD 204.6, twin 214.3, gp 218.4.

## Expectations of the registration (section 7)

| expectation | observed | met |
|---|---|---|
| control ratio 0.05 to 0.3 | 0.121 | yes |
| A2 holds, difference to the null +0.10 to +0.35 | +0.428, holds | no |
| A1 uncertain: difference +0.00 to +0.15 | +0.284, holds | no |
| A3 uncertain (no range) | +0.002, does not hold | n/a |
| model C over model 0: +0.01 to +0.06, B2 holds on three or four splits | points +0.071, +0.056, +0.073, +0.060; B2 on 4 of 4 | no |
| model D over model 0: 0.00 to +0.04, B1 holds on one to three splits | points +0.044, +0.050, +0.044, +0.055; B1 on 4 of 4 | no |
| model CD over model 0: +0.02 to +0.08 | +0.074, +0.071, +0.075, +0.071 | yes |
| reduction of D over that of CD: 0.2 to 0.7 | 0.600, 0.706, 0.582, 0.775 | no |

2 of 7 expectations with a range met.

## Choices made where the registration left room

- Components: natural logarithm of the nine raw BES values of a grid row; c = mean over the nine; the column of a channel is the last two digits of its name (05, 06, 07), the poloidal row the two digits before (01, 02, 03); g1 = mean(col 06) - mean(col 05), g2 = mean(col 07) - mean(col 06), g = g1 + g2, h1 = mean(row 02) - mean(row 01), h2 = mean(row 03) - mean(row 02), h = h1 + h2. The non-positive count is over the grid rows without the first and last row of each file.
- Control: the events of run_b35.events (step of more than 15 and less than 45 ms after five 10 ms steps, T_i observed in at least three of the six rows before, values above 3 keV removed, held values kept), rebuilt on the grid of seq_data.load_grid_files without held removal and checked equal in number to b35_blip_response.json. Change = value in the first row after the step minus the mean of the six rows before (rows j - 5 to j). The ratio is the median over events of |change of g| over the median of |change of c|; the median over events of the per-event ratio is reported beside it. Interval: 2,000 resamples of the discharges (numpy default_rng(20261120)).
- Part A uses run_b43.s_blocks unchanged (20-row blocks inside segments split at steps of 0.5 s or more, trailing partial block dropped, a block used when at least 10 of its rows have a T_i value, block means of T_i over the measured rows and of the component over all rows, pairs of consecutive used blocks) on the cut grid with held values removed, and run_b43.summarize_s for the median, the null and the interval. Exclusion: the component is set to NaN on every row that is one of the ten rows after an absent frame (run_b43.after_rows: rows 1 to 10 after a 20 to 40 ms step inside the segment) or has a non-positive BES channel, and pairs whose component change is not finite are dropped. A block that contains the rows after a gap is thereby left out; when the gap falls exactly between two blocks, the block after it is left out (the block before has none of its rows after the gap). The pairs and the discharges are the same for every component. Discharges with at least ten pairs after the exclusion.
- A3: within a discharge, the ranks of the change of T_i and of the change of g are each regressed linearly on the ranks of the change of c; the partial correlation is the Pearson correlation of the two residuals; the null shifts the residual of g circularly by floor(n/2). The same for c after g (reported).
- Intervals of part A: every component is summarised with a fresh numpy default_rng(20261119), so the discharge resamples are the same for every component that keeps the same discharges.
- Part B: twin, hold and causal GP on the validation discharges with run_b37.Ctx.rows (unchanged) and the network pointed at the twin `data/.b29a1_v2h_cut_s{seed}` (model v2h, per-discharge standardization, as run_b30.NETS). The same machinery was run on every test discharge and matched by (file, row) to every k = 1 row of the B.48 table. Row table rows: file = index into the sorted file names of the cut grid, row = grid row (checked: the T_i of the grid at that row equals y on every row).
- Part B population: k = 1 rows of the B.48 table with y, hold, the causal GP, the twin and the age finite, and every base and component feature finite (the same rows for the four models); validation rows by the same rule. Time step = the step to the row before over 10 ms (run_b37.Ctx.tstep). Component features: s minus its mean over the grid rows of the training discharges of the split; s minus the mean of s over the ten rows before in the same segment (0.5 s rule of run_b37.Ctx); the latter at the row before (NaN when that row is in another segment).
- Regression: run_b37.gbr() (HistGradientBoostingRegressor, random_state 20261116, defaults), fitted on the validation rows with target y minus the twin, applied once to the test rows. Skill of model M over N = 1 - SSE(y - twin - correction_M) / SSE(y - twin - correction_N). Intervals: 2,000 resamples of the test discharges, numpy default_rng(20261118 + split seed), the same resamples for every quantity of a split.
- Quiet rows: class 6 of run_b45.Classes (verbatim). 'The same three skills restricted to quiet rows' is read as CD over 0, over C and over D; D over 0, C over 0 and every model over the twin are also given.
- BES network of B.39: the stored squared errors of `data/.b39_v2b_cut_s{seed}_i{seed}` and of the twin run, mapped from the compacted row index of the harness to the grid row (run_b45.compact_index) and intersected with the rows of part B.

## Deviations

None from the registered definitions, rows, intervals or rules.

## Limits

The common mode is not a pure beam factor (attenuation depends on the density along the beam). The fast channels are single raw samples. Part B shows what information the components carry beyond the twin on about 127 validation discharges per split, not what the network of B.39 uses. One initialisation of the twin. The test rows are those of every earlier readout.

Run time: part A 13.4 s; part B readout 219.3 s (cache build per split: {'42+1': 216.3, '7+123': 219.2}).
