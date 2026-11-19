# B.67 — weighing the sources before the recurrent layer: source-gated fusion

Registered: ces_prediction/experiments/PREREGISTRATION_B67.md. GPU runs (one training per fold, initialisation seed of the fold); read-outs on a CPU. T_i, cut population, the fifteen folds of re-partitions r2, r3, r4 of the pooled cross-validation of B.39. Skill of X over Y = 1 - sum of squared errors of X / that of Y on the same rows; 95 % two-level interval (a re-partition drawn among those read, then its discharges resampled; 20,000 draws); point = mean over re-partitions.

## Statements

| statement | value (interval) | per re-partition | holds | expectation (section 6) | point inside the expected range |
|---|---|---|---|---|---|
| **G1** S over the network (v2), interval above zero | +0.009 [-0.052, +0.040] | r2 +0.024, r3 +0.019, r4 -0.015 | does not hold | -0.01 to +0.02, not met | True |
| **G2** S over the twin, interval above zero | +0.062 [+0.009, +0.087] | r2 +0.068, r3 +0.069, r4 +0.048 | holds | +0.03 to +0.06, met | False |
| **G3** A over the network, interval above zero | -0.006 [-0.059, +0.026] | r2 +0.011, r3 +0.002, r4 -0.033 | does not hold | -0.03 to +0.01, not met | True |
| **G4** arm S: BES gate in rows 1 to 3 after an absent frame minus that in rows right after a measured frame, interval above zero | -0.004 [-0.008, -0.001] | r2 -0.005, r3 -0.004, r4 -0.004 | does not hold | met | - |

## Every network over the references

| network | over the causal GP | over hold | over the offline GP |
|---|---|---|---|
| network v2 (B.39) | +0.092 [+0.068, +0.114] | +0.378 [+0.357, +0.396] | -0.033 [-0.063, -0.005] |
| twin v2h (B.39) | +0.041 [+0.022, +0.059] | +0.344 [+0.324, +0.362] | -0.090 [-0.116, -0.065] |
| S: gated concatenation (v2s) | +0.100 [+0.043, +0.129] | +0.384 [+0.343, +0.408] | -0.023 [-0.090, +0.012] |
| A: attention pooling (v2a) | +0.086 [+0.035, +0.114] | +0.374 [+0.335, +0.398] | -0.039 [-0.099, -0.003] |

| comparison | skill (interval) | per re-partition |
|---|---|---|
| S over A | +0.015 [-0.002, +0.032] | r2 +0.013, r3 +0.017, r4 +0.017 |
| network over the twin | +0.053 [+0.028, +0.076] | r2 +0.045, r3 +0.051, r4 +0.062 |
| A over the twin | +0.047 [+0.002, +0.070] | r2 +0.056, r3 +0.053, r4 +0.032 |

## By rows

Rows 1 to 3 after an absent frame: B.54 row table (flag blip13) on the rows of the network. Rows per re-partition: r2: 191,934 rows, 3,094 after an absent frame, 635 discharges; r3: 192,092 rows, 3,130 after an absent frame, 635 discharges; r4: 192,132 rows, 3,111 after an absent frame, 635 discharges.

| comparison | rows 1 to 3 after an absent frame | other rows |
|---|---|---|
| S over twin | +0.161 [+0.089, +0.225] | +0.058 [+0.004, +0.083] |
| A over twin | +0.133 [+0.073, +0.186] | +0.044 [-0.001, +0.067] |
| network over twin | +0.153 [+0.089, +0.220] | +0.049 [+0.024, +0.072] |
| S over network | +0.009 [-0.090, +0.064] | +0.009 [-0.051, +0.041] |
| A over network | -0.024 [-0.120, +0.031] | -0.006 [-0.058, +0.026] |

## Median absolute errors (eV, mean over re-partitions)

| network or reference | median absolute error |
|---|---|
| network v2 (B.39) | 45.8 |
| twin v2h (B.39) | 45.4 |
| S: gated concatenation (v2s) | 44.4 |
| A: attention pooling (v2a) | 45.2 |
| causal GP | 44.1 |
| hold | 45.4 |
| offline GP | 37.5 |

## Gate readout, arm S

Row-weighted mean gate of each source on the test rows (re-partitions [2, 3, 4]); interval: a re-partition drawn, its discharges resampled, the same draws for every class. The gates of a row sum to one; uniform gates are 0.25 each. A gate is which source the trained network weighs, not a physical cause; gates near uniform do not show that the sources are used equally (registration section 7).

| rows | rows (per re-partition) | BES | ECEI | Mirnov | CES-derived |
|---|---|---|---|---|---|
| all scored rows | 191,934, 192,092, 192,132 | 0.171 [0.144, 0.199] | 0.097 [0.080, 0.122] | 0.098 [0.081, 0.118] | 0.634 [0.596, 0.662] |
| rows 1 to 3 after an absent frame | 3,094, 3,130, 3,111 | 0.166 [0.140, 0.194] | 0.102 [0.082, 0.129] | 0.116 [0.090, 0.144] | 0.616 [0.584, 0.642] |
| rows right after a measured frame | 184,973, 185,007, 185,112 | 0.170 [0.144, 0.198] | 0.096 [0.079, 0.122] | 0.097 [0.080, 0.117] | 0.636 [0.598, 0.664] |
| other rows (neither of the two above) | 3,867, 3,955, 3,909 | 0.196 [0.166, 0.232] | 0.124 [0.104, 0.154] | 0.118 [0.099, 0.147] | 0.561 [0.510, 0.598] |
| burst phase (0 to 30 ms after an onset), classes B and C | 2,664, 2,656, 2,669 | 0.189 [0.157, 0.220] | 0.110 [0.092, 0.138] | 0.117 [0.093, 0.151] | 0.585 [0.544, 0.613] |
| inter-burst phase (>= 50 ms from every onset), classes B and C | 102,750, 102,301, 102,731 | 0.171 [0.144, 0.200] | 0.096 [0.077, 0.122] | 0.096 [0.078, 0.117] | 0.637 [0.597, 0.669] |
| class A (no burst) | 74,006, 74,644, 74,206 | 0.169 [0.141, 0.199] | 0.097 [0.080, 0.125] | 0.099 [0.081, 0.121] | 0.635 [0.592, 0.665] |
| class B (burst rate up to the median) | 84,700, 84,159, 84,879 | 0.166 [0.137, 0.197] | 0.090 [0.070, 0.116] | 0.090 [0.074, 0.109] | 0.653 [0.609, 0.690] |
| class C (burst rate above the median) | 33,076, 33,143, 32,909 | 0.187 [0.159, 0.216] | 0.113 [0.092, 0.142] | 0.116 [0.089, 0.152] | 0.584 [0.546, 0.613] |
| coldest third of the discharges | 39,141, 39,359, 39,130 | 0.169 [0.142, 0.204] | 0.092 [0.076, 0.116] | 0.095 [0.070, 0.130] | 0.644 [0.606, 0.677] |
| middle third | 77,505, 76,536, 76,531 | 0.169 [0.140, 0.202] | 0.093 [0.072, 0.120] | 0.095 [0.073, 0.121] | 0.644 [0.602, 0.675] |
| hottest third | 75,288, 76,197, 76,471 | 0.174 [0.145, 0.199] | 0.103 [0.084, 0.133] | 0.103 [0.090, 0.117] | 0.620 [0.573, 0.660] |

| difference of the mean gate | BES | ECEI | Mirnov | CES-derived |
|---|---|---|---|---|
| rows 1 to 3 after an absent frame minus rows right after a measured frame (G4: BES) | -0.004 [-0.008, -0.001] | +0.006 [+0.002, +0.009] | +0.019 [+0.009, +0.028] | -0.020 [-0.030, -0.008] |
| burst phase minus inter-burst phase (classes B and C) | +0.017 [+0.009, +0.026] | +0.014 [+0.007, +0.024] | +0.020 [+0.010, +0.037] | -0.052 [-0.071, -0.035] |
| class C minus class A | +0.018 [+0.005, +0.031] | +0.016 [+0.004, +0.028] | +0.017 [+0.003, +0.038] | -0.051 [-0.078, -0.021] |
| hottest third minus coldest third | +0.005 [-0.011, +0.023] | +0.011 [-0.004, +0.029] | +0.008 [-0.020, +0.025] | -0.024 [-0.058, +0.020] |

Gate quantiles over rows (5 %, 50 %, 95 %) per re-partition: r2: BES 0.107/0.189/0.308, ECEI 0.048/0.113/0.212, Mirnov 0.028/0.071/0.172, CES-derived 0.378/0.614/0.775; r3: BES 0.071/0.141/0.239, ECEI 0.028/0.069/0.185, Mirnov 0.034/0.095/0.271, CES-derived 0.448/0.667/0.824; r4: BES 0.089/0.162/0.281, ECEI 0.025/0.081/0.180, Mirnov 0.020/0.081/0.220, CES-derived 0.425/0.658/0.818.

Rows right after a measured frame, B.54 rule against the network's own input (T_i staleness 10 ms, outside rows 1 to 3 after an absent frame): r2: agree on 191,934 of 191,934 rows (B.54 rule 184,973, input rule 184,973); r3: agree on 192,092 of 192,092 rows (B.54 rule 185,007, input rule 185,007); r4: agree on 192,132 of 192,132 rows (B.54 rule 185,112, input rule 185,112).

Alignment checks of the gate files (squared errors reproduced in the scoring process): r2: f0 pass, f1 pass, f2 pass, f3 pass, f4 pass; r3: f0 pass, f1 pass, f2 pass, f3 pass, f4 pass; r4: f0 pass, f1 pass, f2 pass, f3 pass, f4 pass.

## Epochs run (best epoch)

| re-partition | runs |
|---|---|
| r2 | est f0 43 (37); twin f0 45 (39); S f0 31 (25); A f0 46 (40); est f1 45 (39); twin f1 44 (38); S f1 31 (25); A f1 37 (31); est f2 26 (20); twin f2 48 (42); S f2 42 (36); A f2 43 (37); est f3 30 (24); twin f3 70 (64); S f3 35 (29); A f3 76 (70); est f4 48 (42); twin f4 27 (21); S f4 43 (37); A f4 47 (41) |
| r3 | est f0 35 (29); twin f0 35 (29); S f0 34 (28); A f0 40 (34); est f1 26 (20); twin f1 25 (19); S f1 24 (18); A f1 25 (19); est f2 40 (34); twin f2 56 (50); S f2 32 (26); A f2 39 (33); est f3 47 (41); twin f3 30 (24); S f3 35 (29); A f3 44 (38); est f4 37 (31); twin f4 80 (80); S f4 35 (29); A f4 39 (33) |
| r4 | est f0 22 (16); twin f0 30 (24); S f0 33 (27); A f0 49 (43); est f1 46 (40); twin f1 35 (29); S f1 33 (27); A f1 37 (31); est f2 43 (37); twin f2 45 (39); S f2 45 (39); A f2 58 (52); est f3 56 (50); twin f3 37 (31); S f3 38 (32); A f3 52 (46); est f4 46 (40); twin f4 44 (38); S f4 38 (32); A f4 80 (78) |

## Choices fixed in the implementation

- Networks of B.39 on the same folds: `.b39p_v2_cut_r{r}_f{k}` (the network) and `.b39p_v2h_cut_r{r}_f{k}` (the twin).
- Rows right after a measured frame: outside rows 1 to 3 after an absent frame, the previous kept row 10 ms earlier (|age - 10 ms| < 0.5 ms) with a finite T_i (B.54 `age_ms`, `hold_val`).
- Burst and inter-burst phases: `run_b66.phase_masks` on the B.54 rows, restricted to the discharges of B.66 classes B and C (B.66 R3); classes A, B, C read as whole classes as well.
- Temperature thirds: within each re-partition, by the median measured T_i of a discharge (`run_b44.split_thirds`).
- Gate intervals: one generator for all classes (the same draws), so the differences are formed draw by draw.

