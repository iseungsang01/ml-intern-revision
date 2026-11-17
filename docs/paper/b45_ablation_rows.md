# B.45 -- in which rows each fast-channel group carries the increment (registered: ces_prediction/experiments/PREREGISTRATION_B45.md)

CPU, nothing trained, no network run. T_i, cut population, the four file-level splits, test rows of the B.39 harness. Networks (epoch cap 80, initialisation = split seed): full causal variant, ECEI-only, BES-only and no-Mirnov variants (B.39 arm A) and the history-only twin (B.29). Skill of A over R = 1 - SSE_A / SSE_R on the same rows. Classes (exclusive, first that applies): 1 after an absent frame (one of the ten rows after a 20 to 40 ms step inside a segment); 2 low beam emission (within 30 ms after a row whose BES level is below half the discharge median); 3 upward departure (a BES channel more than 3 SD above its mean over the ten rows before); 4 downward departure (more than 3 SD below); 5 after a departure (a row with a channel beyond +-3 SD in the three rows before); 6 quiet. Group 7: departure not defined (fewer than ten rows of the segment before), in no statement. Share of the reduction in a class = sum over its rows of (twin error squared minus network error squared) over the same sum over all rows. Intervals: 95 %, 2,000 resamples of the test discharges. A statement holds on a split when its interval is above zero, and holds when that is so on at least three of four splits.

## Checks before the readout

(a) The five runs of each split are scored on the same rows with the same reference errors (causal GP, hold, measured value): split 42 yes, split 1 yes, split 7 yes, split 123 yes.

(b) Skill over the twin on all rows against docs/paper/b39_primary.md (to three decimals):

| split | full (mine / B.39) | BES-only | ECEI-only | no-Mirnov |
|---|---|---|---|---|
| 42 | +0.0709 / +0.0709 (equal) | +0.0740 / +0.0740 (equal) | +0.0095 / +0.0095 (equal) | +0.0485 / +0.0485 (equal) |
| 1 | +0.0223 / +0.0223 (equal) | +0.0258 / +0.0258 (equal) | +0.0008 / +0.0008 (equal) | +0.0433 / +0.0433 (equal) |
| 7 | +0.1232 / +0.1232 (equal) | +0.1047 / +0.1047 (equal) | +0.0837 / +0.0837 (equal) | +0.1350 / +0.1350 (equal) |
| 123 | +0.0603 / +0.0603 (equal) | +0.0628 / +0.0628 (equal) | +0.0375 / +0.0375 (equal) | +0.0742 / +0.0742 (equal) |

(c) Meaning of CES_TI_row: hold recomputed from the 10 ms file of five test discharges per split. Convention found: **the row index of the per-file array of dataset.py (rows with at least one of CES_TI / CES_VT finite after the 3 keV cut and the removal of held values); hold = T_i of the previous kept row in the same block (window W = 2)**.

| split | discharges | rows | matched, that convention | matched, row = grid row (other reading) | after the fact: all test discharges, that convention |
|---|---|---|---|---|---|
| 42 | s31079, s31559, s31815, s31901, s32751 | 1,792 | 1.0000 | 0.0000 | 1.0000 |
| 1 | s31320, s31398, s31511, s31990, s32492 | 1,152 | 1.0000 | 0.0000 | 1.0000 |
| 7 | s31104, s31623, s31806, s31967, s32598 | 1,943 | 1.0000 | 0.0000 | 1.0000 |
| 123 | s31590, s31974, s32027, s32173, s32751 | 1,487 | 1.0000 | 0.0000 | 1.0000 |

BES read from the CSV by channel_departures against the grid: 301 files, row-count mismatches 0, largest time difference 4.65e-07 s.

## Registered statements

- **R1** (classes 3 to 6 (no beam interruption behind them): BES-only over the twin, interval above zero): 3 of 4 splits -> **holds**. split 42 +0.085 [+0.050, +0.124]; split 1 +0.019 [-0.000, +0.042]; split 7 +0.105 [+0.070, +0.134]; split 123 +0.063 [+0.034, +0.097].
- **R2** (class 6, quiet: BES-only over the twin, interval above zero): 3 of 4 splits -> **holds**. split 42 +0.089 [+0.052, +0.130]; split 1 +0.016 [-0.004, +0.041]; split 7 +0.112 [+0.077, +0.143]; split 123 +0.065 [+0.035, +0.102].
- **R3** (class 1, after an absent frame: ECEI-only over the twin, interval above zero): 1 of 4 splits -> **does not hold**. split 42 -0.002 [-0.179, +0.115]; split 1 -0.005 [-0.045, +0.028]; split 7 +0.103 [+0.067, +0.141]; split 123 +0.053 [-0.002, +0.103].

## Reported, per class and split

### Rows and skills over the twin

| split | group | rows | share of rows | full over twin | BES-only over twin | ECEI-only over twin | no-Mirnov over twin |
|---|---|---|---|---|---|---|---|
| 42 | 1 after an absent frame | 1,956 | 0.060 [0.054, 0.066] | +0.183 [+0.016, +0.306] | +0.215 [+0.110, +0.309] | -0.002 [-0.179, +0.115] | +0.196 [+0.070, +0.299] |
| 42 | 2 low beam emission | 447 | 0.014 [0.007, 0.021] | -0.051 [-0.142, -0.000] | -0.101 [-0.206, -0.059] | -0.037 [-0.067, +0.111] | -0.225 [-1.070, -0.002] |
| 42 | 3 upward departure | 97 | 0.003 [0.002, 0.004] | +0.040 [-0.236, +0.184] | +0.076 [-0.150, +0.166] | -0.020 [-0.112, +0.105] | +0.057 [-0.161, +0.178] |
| 42 | 4 downward departure | 28 | 0.001 [0.000, 0.001] | +0.176 [-0.113, +0.645] | -0.026 [-0.257, +0.358] | +0.090 [-0.006, +0.288] | +0.110 [-0.216, +0.622] |
| 42 | 5 after a departure | 308 | 0.009 [0.007, 0.012] | +0.124 [-0.017, +0.223] | +0.018 [-0.090, +0.085] | +0.000 [-0.073, +0.105] | +0.096 [-0.052, +0.209] |
| 42 | 6 quiet | 28,970 | 0.889 [0.878, 0.898] | +0.077 [+0.045, +0.111] | +0.089 [+0.052, +0.130] | +0.017 [-0.007, +0.038] | +0.066 [+0.027, +0.102] |
| 42 | 7 departure not defined | 783 | 0.024 [0.021, 0.028] | -0.108 [-0.588, +0.177] | -0.123 [-0.757, +0.245] | -0.005 [-0.223, +0.157] | -0.109 [-0.610, +0.190] |
| 42 | classes 3 to 6 (outside 1 and 2) | 29,403 | 0.902 [0.891, 0.912] | +0.079 [+0.044, +0.114] | +0.085 [+0.050, +0.124] | +0.015 [-0.006, +0.035] | +0.068 [+0.029, +0.103] |
| 42 | classes 1 and 2 | 2,403 | 0.074 [0.065, 0.084] | +0.055 [-0.043, +0.181] | +0.041 [-0.026, +0.156] | -0.021 [-0.100, +0.085] | -0.036 [-0.339, +0.158] |
| 42 | all scored rows | 32,589 | 1.000 [1.000, 1.000] | +0.071 [+0.035, +0.105] | +0.074 [+0.037, +0.111] | +0.010 [-0.015, +0.033] | +0.048 [-0.029, +0.099] |
| 1 | 1 after an absent frame | 2,043 | 0.057 [0.052, 0.063] | +0.041 [-0.037, +0.132] | +0.069 [-0.010, +0.168] | -0.005 [-0.045, +0.028] | +0.059 [-0.035, +0.164] |
| 1 | 2 low beam emission | 502 | 0.014 [0.007, 0.024] | +0.048 [-0.069, +0.145] | +0.050 [-0.050, +0.128] | +0.055 [-0.028, +0.143] | -0.110 [-0.456, +0.115] |
| 1 | 3 upward departure | 168 | 0.005 [0.004, 0.006] | +0.083 [-0.061, +0.160] | +0.058 [-0.076, +0.147] | +0.064 [+0.014, +0.110] | +0.136 [+0.041, +0.208] |
| 1 | 4 downward departure | 31 | 0.001 [0.000, 0.001] | +0.269 [+0.113, +0.371] | +0.270 [+0.043, +0.394] | -0.057 [-0.148, +0.007] | +0.321 [+0.113, +0.420] |
| 1 | 5 after a departure | 484 | 0.014 [0.010, 0.017] | +0.039 [-0.071, +0.124] | +0.023 [-0.075, +0.084] | +0.061 [-0.006, +0.125] | +0.066 [-0.062, +0.170] |
| 1 | 6 quiet | 31,558 | 0.887 [0.875, 0.896] | +0.013 [-0.012, +0.036] | +0.016 [-0.004, +0.041] | -0.011 [-0.028, +0.005] | +0.047 [+0.021, +0.072] |
| 1 | 7 departure not defined | 796 | 0.022 [0.019, 0.026] | +0.051 [-0.049, +0.156] | +0.071 [-0.000, +0.153] | +0.031 [-0.035, +0.098] | +0.095 [-0.004, +0.205] |
| 1 | classes 3 to 6 (outside 1 and 2) | 32,241 | 0.906 [0.895, 0.915] | +0.018 [-0.005, +0.041] | +0.019 [-0.000, +0.042] | -0.004 [-0.019, +0.010] | +0.053 [+0.028, +0.077] |
| 1 | classes 1 and 2 | 2,545 | 0.072 [0.063, 0.082] | +0.044 [-0.029, +0.112] | +0.060 [-0.000, +0.121] | +0.025 [-0.025, +0.077] | -0.024 [-0.204, +0.104] |
| 1 | all scored rows | 35,582 | 1.000 [1.000, 1.000] | +0.022 [+0.001, +0.044] | +0.026 [+0.007, +0.049] | +0.001 [-0.014, +0.016] | +0.043 [+0.007, +0.074] |
| 7 | 1 after an absent frame | 1,956 | 0.055 [0.049, 0.062] | +0.187 [+0.127, +0.252] | +0.163 [+0.088, +0.239] | +0.103 [+0.067, +0.141] | +0.168 [+0.095, +0.241] |
| 7 | 2 low beam emission | 554 | 0.015 [0.008, 0.026] | -0.041 [-0.190, +0.062] | -0.042 [-0.248, +0.085] | +0.006 [-0.065, +0.067] | -0.030 [-0.234, +0.098] |
| 7 | 3 upward departure | 129 | 0.004 [0.003, 0.005] | +0.110 [-0.224, +0.339] | +0.016 [-0.325, +0.280] | +0.170 [+0.053, +0.265] | +0.076 [-0.204, +0.276] |
| 7 | 4 downward departure | 27 | 0.001 [0.000, 0.001] | +0.396 [+0.187, +0.615] | +0.522 [+0.303, +0.770] | +0.248 [-0.373, +0.693] | +0.531 [+0.224, +0.800] |
| 7 | 5 after a departure | 361 | 0.010 [0.007, 0.013] | -0.006 [-0.182, +0.141] | -0.032 [-0.209, +0.102] | +0.041 [-0.029, +0.118] | -0.011 [-0.212, +0.145] |
| 7 | 6 quiet | 31,907 | 0.892 [0.880, 0.902] | +0.131 [+0.106, +0.158] | +0.112 [+0.077, +0.143] | +0.088 [+0.071, +0.105] | +0.149 [+0.115, +0.186] |
| 7 | 7 departure not defined | 825 | 0.023 [0.020, 0.027] | +0.060 [-0.109, +0.207] | +0.132 [+0.032, +0.215] | +0.001 [-0.135, +0.110] | +0.063 [-0.120, +0.217] |
| 7 | classes 3 to 6 (outside 1 and 2) | 32,424 | 0.907 [0.894, 0.916] | +0.126 [+0.100, +0.153] | +0.105 [+0.070, +0.134] | +0.087 [+0.071, +0.105] | +0.141 [+0.107, +0.176] |
| 7 | classes 1 and 2 | 2,510 | 0.070 [0.060, 0.082] | +0.116 [+0.052, +0.178] | +0.099 [+0.030, +0.161] | +0.073 [+0.034, +0.113] | +0.107 [+0.035, +0.169] |
| 7 | all scored rows | 35,759 | 1.000 [1.000, 1.000] | +0.123 [+0.097, +0.150] | +0.105 [+0.070, +0.133] | +0.084 [+0.067, +0.102] | +0.135 [+0.101, +0.170] |
| 123 | 1 after an absent frame | 1,977 | 0.058 [0.052, 0.066] | +0.075 [-0.038, +0.175] | +0.104 [+0.031, +0.179] | +0.053 [-0.002, +0.103] | +0.041 [-0.100, +0.157] |
| 123 | 2 low beam emission | 346 | 0.010 [0.006, 0.016] | -0.002 [-0.042, +0.048] | +0.010 [-0.045, +0.051] | +0.003 [-0.063, +0.091] | -0.015 [-0.061, +0.033] |
| 123 | 3 upward departure | 146 | 0.004 [0.003, 0.005] | +0.093 [-0.006, +0.163] | +0.103 [-0.039, +0.174] | +0.016 [-0.027, +0.067] | +0.080 [-0.046, +0.145] |
| 123 | 4 downward departure | 15 | 0.000 [0.000, 0.001] | +0.551 [-0.679, +0.795] | +0.550 [-0.234, +0.832] | +0.184 [-0.564, +0.350] | +0.510 [-0.513, +0.758] |
| 123 | 5 after a departure | 401 | 0.012 [0.009, 0.015] | -0.029 [-0.157, +0.092] | -0.046 [-0.158, +0.055] | +0.011 [-0.080, +0.097] | -0.028 [-0.139, +0.062] |
| 123 | 6 quiet | 30,401 | 0.893 [0.881, 0.902] | +0.064 [+0.027, +0.106] | +0.065 [+0.035, +0.102] | +0.039 [-0.009, +0.080] | +0.087 [+0.050, +0.128] |
| 123 | 7 departure not defined | 752 | 0.022 [0.019, 0.026] | +0.128 [+0.062, +0.208] | +0.048 [-0.018, +0.128] | +0.098 [+0.050, +0.155] | +0.119 [+0.053, +0.201] |
| 123 | classes 3 to 6 (outside 1 and 2) | 30,963 | 0.910 [0.899, 0.918] | +0.062 [+0.027, +0.100] | +0.063 [+0.034, +0.097] | +0.037 [-0.008, +0.076] | +0.083 [+0.048, +0.121] |
| 123 | classes 1 and 2 | 2,323 | 0.068 [0.060, 0.079] | +0.043 [-0.018, +0.119] | +0.065 [+0.015, +0.119] | +0.032 [-0.007, +0.075] | +0.017 [-0.067, +0.105] |
| 123 | all scored rows | 34,038 | 1.000 [1.000, 1.000] | +0.060 [+0.027, +0.098] | +0.063 [+0.035, +0.094] | +0.038 [-0.001, +0.073] | +0.074 [+0.038, +0.112] |

### Networks against each other, the causal GP, and hold

| split | group | full over BES-only | full over ECEI-only | BES-only over causal GP | RMSE of hold (eV) |
|---|---|---|---|---|---|
| 42 | 1 after an absent frame | -0.041 [-0.223, +0.074] | +0.184 [+0.094, +0.264] | +0.297 [+0.183, +0.396] | 199.6 [169.7, 230.9] |
| 42 | 2 low beam emission | +0.046 [+0.002, +0.091] | -0.013 [-0.241, +0.032] | -0.481 [-1.113, -0.095] | 415.6 [218.4, 568.5] |
| 42 | 3 upward departure | -0.039 [-0.144, +0.064] | +0.060 [-0.354, +0.168] | -0.033 [-0.397, +0.090] | 467.4 [284.2, 635.5] |
| 42 | 4 downward departure | +0.196 [+0.095, +0.511] | +0.094 [-0.116, +0.526] | -0.141 [-0.456, +0.512] | 246.8 [150.8, 329.8] |
| 42 | 5 after a departure | +0.108 [-0.016, +0.204] | +0.124 [-0.033, +0.227] | +0.056 [-0.053, +0.202] | 485.2 [354.4, 601.3] |
| 42 | 6 quiet | -0.014 [-0.040, +0.008] | +0.061 [+0.023, +0.102] | +0.116 [+0.078, +0.157] | 185.5 [162.4, 210.4] |
| 42 | 7 departure not defined | +0.014 [-0.176, +0.114] | -0.102 [-0.740, +0.224] | +0.003 [-0.880, +0.430] | 162.0 [107.7, 220.2] |
| 42 | classes 3 to 6 (outside 1 and 2) | -0.007 [-0.032, +0.016] | +0.064 [+0.026, +0.103] | +0.108 [+0.074, +0.146] | 192.7 [167.8, 219.6] |
| 42 | classes 1 and 2 | +0.014 [-0.049, +0.072] | +0.074 [+0.017, +0.140] | -0.052 [-0.268, +0.198] | 254.1 [195.7, 314.5] |
| 42 | all scored rows | -0.003 [-0.025, +0.017] | +0.062 [+0.024, +0.101] | +0.084 [+0.040, +0.126] | 197.2 [173.9, 222.5] |
| 1 | 1 after an absent frame | -0.030 [-0.078, +0.001] | +0.045 [-0.016, +0.124] | +0.211 [+0.110, +0.318] | 267.0 [209.5, 321.3] |
| 1 | 2 low beam emission | -0.002 [-0.144, +0.100] | -0.008 [-0.058, +0.043] | +0.001 [-0.112, +0.108] | 479.5 [324.3, 660.2] |
| 1 | 3 upward departure | +0.026 [-0.040, +0.080] | +0.020 [-0.141, +0.107] | +0.126 [-0.098, +0.322] | 566.9 [426.3, 712.1] |
| 1 | 4 downward departure | -0.002 [-0.085, +0.127] | +0.308 [+0.200, +0.381] | -0.070 [-1.712, +0.460] | 366.9 [134.4, 518.2] |
| 1 | 5 after a departure | +0.017 [-0.031, +0.058] | -0.023 [-0.103, +0.024] | +0.089 [-0.019, +0.193] | 476.6 [385.0, 561.5] |
| 1 | 6 quiet | -0.004 [-0.024, +0.015] | +0.024 [+0.003, +0.046] | +0.086 [+0.058, +0.116] | 235.8 [203.9, 268.2] |
| 1 | 7 departure not defined | -0.021 [-0.093, +0.041] | +0.021 [-0.055, +0.099] | +0.285 [+0.181, +0.396] | 247.1 [184.1, 310.6] |
| 1 | classes 3 to 6 (outside 1 and 2) | -0.001 [-0.021, +0.017] | +0.022 [+0.001, +0.043] | +0.088 [+0.059, +0.118] | 244.2 [211.3, 276.7] |
| 1 | classes 1 and 2 | -0.016 [-0.089, +0.038] | +0.020 [-0.022, +0.069] | +0.120 [+0.039, +0.207] | 320.3 [264.2, 375.7] |
| 1 | all scored rows | -0.004 [-0.025, +0.015] | +0.022 [+0.002, +0.042] | +0.098 [+0.070, +0.125] | 250.5 [219.0, 281.7] |
| 7 | 1 after an absent frame | +0.029 [-0.039, +0.095] | +0.094 [+0.028, +0.162] | +0.270 [+0.172, +0.371] | 299.7 [237.8, 361.1] |
| 7 | 2 low beam emission | +0.002 [-0.056, +0.071] | -0.047 [-0.186, +0.025] | -0.086 [-0.354, +0.106] | 316.7 [210.1, 444.1] |
| 7 | 3 upward departure | +0.096 [-0.151, +0.305] | -0.071 [-0.442, +0.243] | -0.112 [-0.452, +0.160] | 419.7 [303.7, 533.9] |
| 7 | 4 downward departure | -0.264 [-1.496, +0.251] | +0.198 [-0.784, +0.472] | +0.825 [+0.418, +0.960] | 617.8 [143.3, 1054.6] |
| 7 | 5 after a departure | +0.025 [-0.043, +0.097] | -0.049 [-0.199, +0.057] | +0.074 [-0.073, +0.200] | 424.1 [326.8, 527.8] |
| 7 | 6 quiet | +0.022 [-0.010, +0.057] | +0.048 [+0.027, +0.070] | +0.116 [+0.072, +0.151] | 213.6 [182.1, 248.1] |
| 7 | 7 departure not defined | -0.084 [-0.207, +0.037] | +0.059 [-0.086, +0.170] | +0.199 [-0.030, +0.364] | 163.0 [124.0, 202.4] |
| 7 | classes 3 to 6 (outside 1 and 2) | +0.023 [-0.007, +0.057] | +0.042 [+0.021, +0.065] | +0.116 [+0.077, +0.148] | 218.9 [186.6, 253.5] |
| 7 | classes 1 and 2 | +0.019 [-0.030, +0.071] | +0.047 [-0.008, +0.096] | +0.172 [+0.079, +0.263] | 303.6 [247.4, 359.3] |
| 7 | all scored rows | +0.021 [-0.009, +0.054] | +0.043 [+0.022, +0.066] | +0.126 [+0.090, +0.159] | 224.8 [192.5, 259.5] |
| 123 | 1 after an absent frame | -0.033 [-0.134, +0.029] | +0.023 [-0.098, +0.125] | +0.166 [+0.019, +0.287] | 295.8 [239.7, 355.7] |
| 123 | 2 low beam emission | -0.012 [-0.042, +0.032] | -0.005 [-0.068, +0.045] | -0.078 [-0.413, +0.127] | 559.8 [397.4, 701.4] |
| 123 | 3 upward departure | -0.011 [-0.055, +0.055] | +0.078 [-0.042, +0.154] | +0.101 [-0.055, +0.231] | 620.4 [463.0, 764.8] |
| 123 | 4 downward departure | +0.004 [-0.869, +0.286] | +0.450 [-0.180, +0.743] | +0.568 [+0.177, +0.882] | 167.1 [89.4, 203.7] |
| 123 | 5 after a departure | +0.016 [-0.024, +0.076] | -0.041 [-0.129, +0.060] | +0.071 [-0.076, +0.267] | 524.8 [399.8, 642.0] |
| 123 | 6 quiet | -0.002 [-0.031, +0.023] | +0.026 [-0.011, +0.076] | +0.107 [+0.078, +0.137] | 248.1 [210.9, 288.3] |
| 123 | 7 departure not defined | +0.085 [+0.032, +0.152] | +0.034 [-0.014, +0.094] | +0.226 [+0.104, +0.333] | 250.2 [169.1, 329.8] |
| 123 | classes 3 to 6 (outside 1 and 2) | -0.001 [-0.029, +0.022] | +0.026 [-0.009, +0.073] | +0.105 [+0.078, +0.134] | 256.6 [218.3, 298.2] |
| 123 | classes 1 and 2 | -0.024 [-0.065, +0.020] | +0.011 [-0.069, +0.084] | +0.073 [-0.071, +0.195] | 348.1 [286.5, 408.7] |
| 123 | all scored rows | -0.003 [-0.029, +0.020] | +0.024 [-0.008, +0.066] | +0.104 [+0.072, +0.134] | 263.7 [226.7, 305.7] |

### Share of the total squared-error reduction against the twin

A share above 1 or below 0 means the other rows have a net increase. Total reduction (sum over all rows of twin error squared minus network error squared, eV^2): split 42 BES-only 6.438e+07, full 6.173e+07; split 1 BES-only 3.663e+07, full 3.158e+07; split 7 BES-only 1.279e+08, full 1.505e+08; split 123 BES-only 9.288e+07, full 8.924e+07.

| split | group | share of rows | BES-only | full |
|---|---|---|---|---|
| 42 | 1 after an absent frame | 0.060 | +0.194 [+0.103, +0.347] | +0.172 [+0.017, +0.341] |
| 42 | 2 low beam emission | 0.014 | -0.111 [-0.325, -0.021] | -0.058 [-0.210, -0.000] |
| 42 | 3 upward departure | 0.003 | +0.025 [-0.022, +0.089] | +0.014 [-0.054, +0.069] |
| 42 | 4 downward departure | 0.001 | -0.002 [-0.031, +0.015] | +0.011 [-0.013, +0.034] |
| 42 | 5 after a departure | 0.009 | +0.010 [-0.040, +0.063] | +0.072 [-0.007, +0.171] |
| 42 | 6 quiet | 0.889 | +0.920 [+0.754, +1.323] | +0.822 [+0.643, +1.198] |
| 42 | 7 departure not defined | 0.024 | -0.036 [-0.348, +0.062] | -0.033 [-0.270, +0.049] |
| 42 | classes 3 to 6 (outside 1 and 2) | 0.902 | +0.953 [+0.790, +1.351] | +0.919 [+0.735, +1.309] |
| 42 | classes 1 and 2 | 0.074 | +0.083 [-0.096, +0.202] | +0.114 [-0.139, +0.290] |
| 1 | 1 after an absent frame | 0.057 | +0.186 [-0.058, +0.557] | +0.128 [-0.336, +0.681] |
| 1 | 2 low beam emission | 0.014 | +0.130 [-0.136, +0.502] | +0.144 [-0.446, +0.899] |
| 1 | 3 upward departure | 0.005 | +0.078 [-0.112, +0.340] | +0.129 [-0.161, +0.853] |
| 1 | 4 downward departure | 0.001 | +0.024 [+0.001, +0.109] | +0.028 [+0.000, +0.199] |
| 1 | 5 after a departure | 0.014 | +0.045 [-0.167, +0.306] | +0.091 [-0.305, +0.581] |
| 1 | 6 quiet | 0.887 | +0.472 [-0.329, +0.843] | +0.427 [-1.194, +1.191] |
| 1 | 7 departure not defined | 0.022 | +0.065 [-0.002, +0.234] | +0.054 [-0.134, +0.388] |
| 1 | classes 3 to 6 (outside 1 and 2) | 0.906 | +0.619 [+0.033, +0.927] | +0.674 [-0.352, +1.335] |
| 1 | classes 1 and 2 | 0.072 | +0.316 [-0.015, +0.828] | +0.272 [-0.387, +1.091] |
| 7 | 1 after an absent frame | 0.055 | +0.153 [+0.095, +0.221] | +0.150 [+0.104, +0.201] |
| 7 | 2 low beam emission | 0.015 | -0.018 [-0.099, +0.039] | -0.015 [-0.066, +0.021] |
| 7 | 3 upward departure | 0.004 | +0.002 [-0.042, +0.038] | +0.011 [-0.022, +0.036] |
| 7 | 4 downward departure | 0.001 | +0.012 [+0.001, +0.033] | +0.008 [+0.001, +0.019] |
| 7 | 5 after a departure | 0.010 | -0.012 [-0.083, +0.035] | -0.002 [-0.056, +0.041] |
| 7 | 6 quiet | 0.892 | +0.843 [+0.750, +0.957] | +0.840 [+0.774, +0.928] |
| 7 | 7 departure not defined | 0.023 | +0.020 [+0.004, +0.041] | +0.008 [-0.014, +0.028] |
| 7 | classes 3 to 6 (outside 1 and 2) | 0.907 | +0.845 [+0.765, +0.928] | +0.857 [+0.801, +0.920] |
| 7 | classes 1 and 2 | 0.070 | +0.135 [+0.047, +0.212] | +0.135 [+0.068, +0.195] |
| 123 | 1 after an absent frame | 0.058 | +0.135 [+0.047, +0.224] | +0.101 [-0.077, +0.233] |
| 123 | 2 low beam emission | 0.010 | +0.009 [-0.032, +0.050] | -0.002 [-0.060, +0.046] |
| 123 | 3 upward departure | 0.004 | +0.047 [-0.008, +0.134] | +0.044 [-0.001, +0.136] |
| 123 | 4 downward departure | 0.000 | +0.005 [-0.000, +0.018] | +0.005 [-0.001, +0.021] |
| 123 | 5 after a departure | 0.012 | -0.023 [-0.109, +0.020] | -0.015 [-0.122, +0.036] |
| 123 | 6 quiet | 0.893 | +0.811 [+0.682, +0.932] | +0.823 [+0.647, +0.991] |
| 123 | 7 departure not defined | 0.022 | +0.016 [-0.005, +0.050] | +0.044 [+0.015, +0.118] |
| 123 | classes 3 to 6 (outside 1 and 2) | 0.910 | +0.840 [+0.733, +0.947] | +0.857 [+0.720, +1.001] |
| 123 | classes 1 and 2 | 0.068 | +0.144 [+0.040, +0.248] | +0.099 [-0.073, +0.225] |

### RMSE of every estimator per group (eV, point values)

| split | group | hold | causal GP | twin | full | BES-only | ECEI-only | no-Mirnov |
|---|---|---|---|---|---|---|---|---|
| 42 | 1 after an absent frame | 199.6 | 182.0 | 172.3 | 155.8 | 152.7 | 172.5 | 154.5 |
| 42 | 2 low beam emission | 415.6 | 343.5 | 398.3 | 408.3 | 417.9 | 405.7 | 441.0 |
| 42 | 3 upward departure | 467.4 | 441.8 | 467.1 | 457.6 | 449.0 | 471.8 | 453.5 |
| 42 | 4 downward departure | 246.8 | 351.6 | 370.9 | 336.8 | 375.6 | 353.8 | 349.9 |
| 42 | 5 after a departure | 485.2 | 348.3 | 341.5 | 319.6 | 338.3 | 341.5 | 324.6 |
| 42 | 6 quiet | 185.5 | 153.4 | 151.1 | 145.2 | 144.2 | 149.8 | 146.0 |
| 42 | 7 departure not defined | 162.0 | 164.8 | 155.3 | 163.4 | 164.5 | 155.6 | 163.5 |
| 42 | classes 3 to 6 (outside 1 and 2) | 192.7 | 158.8 | 156.7 | 150.5 | 149.9 | 155.5 | 151.3 |
| 42 | classes 1 and 2 | 254.1 | 221.2 | 231.7 | 225.3 | 226.9 | 234.2 | 235.8 |
| 42 | all scored rows | 197.2 | 164.3 | 163.4 | 157.5 | 157.3 | 162.6 | 159.4 |
| 1 | 1 after an absent frame | 267.0 | 238.7 | 219.7 | 215.2 | 212.0 | 220.3 | 213.2 |
| 1 | 2 low beam emission | 479.5 | 423.1 | 433.9 | 423.3 | 422.8 | 421.7 | 457.1 |
| 1 | 3 upward departure | 566.9 | 561.7 | 541.2 | 518.4 | 525.3 | 523.6 | 503.1 |
| 1 | 4 downward departure | 366.9 | 268.3 | 324.8 | 277.8 | 277.6 | 334.0 | 267.6 |
| 1 | 5 after a departure | 476.6 | 402.7 | 388.8 | 381.1 | 384.4 | 376.8 | 375.8 |
| 1 | 6 quiet | 235.8 | 190.6 | 183.7 | 182.5 | 182.2 | 184.7 | 179.3 |
| 1 | 7 departure not defined | 247.1 | 234.1 | 205.4 | 200.0 | 198.0 | 202.1 | 195.4 |
| 1 | classes 3 to 6 (outside 1 and 2) | 244.2 | 199.2 | 192.2 | 190.4 | 190.3 | 192.5 | 187.0 |
| 1 | classes 1 and 2 | 320.3 | 284.7 | 275.5 | 269.3 | 267.1 | 272.1 | 278.7 |
| 1 | all scored rows | 250.5 | 207.4 | 199.6 | 197.3 | 197.0 | 199.5 | 195.2 |
| 7 | 1 after an absent frame | 299.7 | 265.8 | 248.1 | 223.7 | 227.1 | 235.1 | 226.3 |
| 7 | 2 low beam emission | 316.7 | 307.4 | 313.8 | 320.1 | 320.3 | 312.8 | 318.4 |
| 7 | 3 upward departure | 419.7 | 325.4 | 346.1 | 326.4 | 343.2 | 315.4 | 332.7 |
| 7 | 4 downward departure | 617.8 | 540.8 | 327.5 | 254.5 | 226.3 | 284.1 | 224.2 |
| 7 | 5 after a departure | 424.1 | 382.6 | 362.5 | 363.6 | 368.2 | 354.9 | 364.5 |
| 7 | 6 quiet | 213.6 | 174.0 | 173.6 | 161.8 | 163.6 | 165.8 | 160.2 |
| 7 | 7 departure not defined | 163.0 | 158.9 | 152.6 | 148.0 | 142.1 | 152.5 | 147.7 |
| 7 | classes 3 to 6 (outside 1 and 2) | 218.9 | 179.1 | 178.0 | 166.5 | 168.4 | 170.1 | 165.0 |
| 7 | classes 1 and 2 | 303.6 | 275.5 | 264.0 | 248.2 | 250.7 | 254.3 | 249.6 |
| 7 | all scored rows | 224.8 | 187.1 | 184.8 | 173.1 | 174.9 | 176.9 | 171.9 |
| 123 | 1 after an absent frame | 295.8 | 255.6 | 246.6 | 237.2 | 233.4 | 239.9 | 241.5 |
| 123 | 2 low beam emission | 559.8 | 481.5 | 502.4 | 502.8 | 499.9 | 501.7 | 506.2 |
| 123 | 3 upward departure | 620.4 | 540.9 | 541.6 | 515.9 | 513.0 | 537.4 | 519.4 |
| 123 | 4 downward departure | 167.1 | 236.6 | 231.6 | 155.1 | 155.4 | 209.1 | 162.2 |
| 123 | 5 after a departure | 524.8 | 363.0 | 342.1 | 347.0 | 349.9 | 340.1 | 346.8 |
| 123 | 6 quiet | 248.1 | 199.0 | 194.6 | 188.3 | 188.1 | 190.8 | 185.9 |
| 123 | 7 departure not defined | 250.2 | 224.1 | 202.0 | 188.6 | 197.1 | 191.8 | 189.6 |
| 123 | classes 3 to 6 (outside 1 and 2) | 256.6 | 204.9 | 200.2 | 194.0 | 193.8 | 196.5 | 191.8 |
| 123 | classes 1 and 2 | 348.1 | 300.2 | 298.9 | 292.5 | 289.1 | 294.1 | 296.3 |
| 123 | all scored rows | 263.7 | 213.2 | 208.5 | 202.1 | 201.8 | 204.5 | 200.6 |

## Comparison with the expectations (section 7)

| expectation | observed (splits 42, 1, 7, 123) | met |
|---|---|---|
| share of rows, class 1: 0.03 to 0.08 | 0.060, 0.057, 0.055, 0.058 | yes |
| share of rows, class 2: below 0.04 | 0.014, 0.014, 0.015, 0.010 | yes |
| share of rows, class 3: 0.01 to 0.04 | 0.003, 0.005, 0.004, 0.004 | no |
| share of rows, class 4: 0.01 to 0.04 | 0.001, 0.001, 0.001, 0.000 | no |
| share of rows, class 5: 0.03 to 0.10 | 0.009, 0.014, 0.010, 0.012 | no |
| share of rows, quiet: 0.75 to 0.90 | 0.889, 0.887, 0.892, 0.893 | yes |
| R1: +0.02 to +0.09, holds on three or four splits | points +0.085, +0.019, +0.105, +0.063; holds on 3 of 4 | no |
| R2: 0.00 to +0.05, holds on two or three splits | points +0.089, +0.016, +0.112, +0.065; holds on 3 of 4 | no |
| R3: +0.03 to +0.20, holds on two to four splits | points -0.002, -0.005, +0.103, +0.053; holds on 1 of 4 | no |
| share of the squared-error reduction of the BES-only network in classes 1 and 2 together: 0.25 to 0.60 | 0.083, 0.316, 0.135, 0.144 | no |

3 of 10 expectations met.

## Choices made where the registration left room

- Rows: every row of comparison_errors_test.npz (T_i), i.e. the scored test rows of the B.39 harness; no further population filter.
- Meaning of CES_TI_row (check c): the index into the per-file array of dataset.py, which keeps only the rows with at least one of CES_TI / CES_VT finite after the 3 keV cut and the removal of held values (the CSV after dropna on the inputs, then spike cut, then held removal, then rows without any target dropped). It is mapped to the grid row of seq_data.load_grid_files(DATA, True, 3000.0) through the finite-target rows of that grid; the grid row minus one is the index of posthoc_b32.channel_departures. Hold of the harness (window W = 2) is the T_i of the previous kept row in the same block (step < 0.5 s).
- Check (c): five test discharges per split drawn with numpy default_rng(20261118) (one generator for the four splits, in the order 42, 1, 7, 123); a row matches when the absolute error of hold and the measured value both agree with the stored ones within 0.5 eV (the tolerance of B.30, amendment 1). The other reading (row = grid row, hold = last finite T_i before it in the segment) is reported next to it.
- Absent frame: a step of more than 15 and less than 45 ms (a 20 to 40 ms step on the 10 ms grid) inside a segment (steps of 0.5 s or more end a segment); class 1 = the `after` counter of channel_departures is 1 to 10.
- Class 2: Ctx.after_low_bes of b41_events/run_b41.py, verbatim (BES level = mean of the nine raw channels of the grid row; rows with a stamp from t_j to t_j + 30 ms of a row j whose level is below half the median level of the whole grid file).
- Classes 3 and 4: departure of channel_departures (per-discharge z-score of each BES channel, minus the mean of the ten rows before in the same segment); class 3 when any channel is above +3, class 4 when any is below -3 (and none above +3).
- Class 5: a row with any channel beyond +3 or -3 (whatever its own class) among the three grid rows before, in the same segment.
- Group 7: every scored row whose departure is not defined (fewer than ten rows of the segment before) is put in group 7 before classes 1 to 6 are assigned, also when it is one of the ten rows after an absent frame or a low-beam row (the literal reading of 'enter no statement'). The number of such rows that the other reading would have put in classes 1 and 2 is reported.
- Statement sets: R1 = classes 3 to 6 (group 7 excluded); R2 = class 6; R3 = class 1.
- Intervals: 2,000 multinomial resamples of the test discharges of the split (numpy default_rng(20261118 + split seed)); the same resamples for every quantity of the split. A resample in which a denominator is zero is dropped from the interval of that quantity.
- Share of the rows and of the reduction: over all scored rows of the split, group 7 included.
- Expectations (section 7): a range is met when the point value of every one of the four splits lies inside it; for R1 to R3 the count of splits must also be in the expected set.

Group 7 rows that the other reading would have put in classes 1 to 6: split 42: class 1 9, class 2 193, class 3 0, class 4 0, class 5 0, class 6 581; split 1: class 1 0, class 2 154, class 3 0, class 4 0, class 5 0, class 6 642; split 7: class 1 9, class 2 185, class 3 0, class 4 0, class 5 0, class 6 631; split 123: class 1 1, class 2 165, class 3 0, class 4 0, class 5 0, class 6 586.

After the fact (not registered): R3 with those rows kept in class 1: split 42 -0.001 [-0.178, +0.116]; split 7 +0.102 [+0.067, +0.141]; split 123 +0.053 [-0.002, +0.103].

## Deviations

None from the registered definitions, thresholds, populations or statements.

## Limits

One initialisation per network. The classes are defined from BES, which the ECEI-only network does not read. Classes 1 to 4 hold few rows, so their intervals are wide. The test rows of these splits were read before by D1, D2 and earlier registered readouts.

Run time: 8 s.
