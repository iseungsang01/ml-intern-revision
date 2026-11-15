# MAST gate, addendum 2 - breakdown by campaign and machine configuration

Registration: `ces_prediction/experiments/PREREGISTRATION_MAST_GATE_ADDENDUM2.md`. Post hoc: the pooled TEST results were seen before this breakdown was registered. Hyperparameters are those of addendum 1; nothing is retuned. TEST shots, population G (relative T_i error < 0.1). Intervals: 2,000 bootstrap resamples of the shots inside the stratum; a stratum with fewer than 8 TEST shots carrying scored rows has point values only ("too few"). No correction for multiplicity; only S1-S4 are statements.

Integrity: the pooled addendum-1 numbers (rows, skills, mean reduced chi-square of every arm, regime number, error ratio) are reproduced exactly for both targets; `mast_gate.json` and `mast_gate_addendum1.json` unchanged: True.

## 1. Statements

| statement | target | M6 | M7 | M8 | M9 | campaigns holding | verdict |
|---|---|---|---|---|---|---|---|
| S1: regime number above 0 in every campaign | T_i | +0.109 [+0.056, +0.337], above 0 | +0.226 [+0.101, +0.339], above 0 | +0.345 [+0.197, +0.595], above 0 | +0.359 [+0.205, +0.497], above 0 | 4 / 4 | holds |
| S1: regime number above 0 in every campaign | V_i | +0.383 [+0.210, +0.460], above 0 | +0.419 [+0.330, +0.590], above 0 | +0.569 [+0.382, +0.605], above 0 | +0.524 [+0.418, +0.606], above 0 | 4 / 4 | holds |
| S2: mean reduced chi-square of the causal GP above 1.5 in every campaign | T_i | 1.727 [1.123, 2.737], includes 1.5 | 3.978 [2.397, 5.899], above 1.5 | 2.999 [1.968, 4.619], above 1.5 | 1.070 [0.695, 1.517], includes 1.5 | 2 / 4 | does not hold |
| S2: mean reduced chi-square of the causal GP above 1.5 in every campaign | V_i | 13.073 [9.355, 19.362], above 1.5 | 24.388 [16.713, 32.968], above 1.5 | 19.550 [12.935, 27.795], above 1.5 | 7.252 [5.041, 9.577], above 1.5 | 4 / 4 | holds |
| S3: running-mean skill vs persistence not above 0 in every campaign | T_i | -0.003 [-0.011, +0.002], includes 0 | -0.009 [-0.017, -0.000], below 0 | -0.018 [-0.029, -0.008], below 0 | -0.012 [-0.023, -0.002], below 0 | 4 / 4 | holds |
| S3: running-mean skill vs persistence not above 0 in every campaign | V_i | -0.019 [-0.028, -0.010], below 0 | -0.018 [-0.035, -0.010], below 0 | -0.030 [-0.039, -0.023], below 0 | -0.025 [-0.036, -0.015], below 0 | 4 / 4 | holds |

S4: gate shots without a beam block, by catalogue heating (all 264 shots with data; "no block" = usable frame spacing but no block of at least 20 frames with 10 good chords).

Population G:

| heating | with data | unusable spacing | no block | block, no scored chord | block and chords | share no block |
|---|---|---|---|---|---|---|
| 2 beams | 82 | 0 | 4 | 7 | 71 | 0.049 |
| Ohmic | 2 | 0 | 1 | 0 | 1 | 0.500 |
| SS Beam | 49 | 0 | 5 | 9 | 35 | 0.102 |
| no record | 131 | 3 | 31 | 35 | 62 | 0.237 |

Without the SS beam (SW Beam, Ohmic; 2 shots): share without a block 0.500 (registered: at least 0.8). With the SS beam (131 shots): 0.069 (registered: at most 0.2). S4 does not hold on population G.

Population W:

| heating | with data | unusable spacing | no block | block, no scored chord | block and chords | share no block |
|---|---|---|---|---|---|---|
| 2 beams | 82 | 0 | 4 | 0 | 78 | 0.049 |
| Ohmic | 2 | 0 | 0 | 0 | 2 | 0.000 |
| SS Beam | 49 | 0 | 5 | 2 | 42 | 0.102 |
| no record | 131 | 3 | 24 | 8 | 96 | 0.183 |

Without the SS beam (SW Beam, Ohmic; 2 shots): share without a block 0.000 (registered: at least 0.8). With the SS beam (131 shots): 0.069 (registered: at most 0.2). S4 does not hold on population W.

Not part of the registered S4 reading: of the 300 gate shots, those that carry no `act` arrays at all, by heating tag - 2 beams: 0 of 82; Ohmic: 0 of 2; SS Beam: 0 of 49; SW Beam: 17 of 17; no record: 19 of 150. Every SW-beam-only gate shot is among the shots without arrays, so the registered comparison has only the Ohmic shots on its no-SS side.

## 2. Readouts per stratum

### Campaign (primary)

**T_i**

| stratum | TEST shots | with rows | rows | regime number (ref 0) | causal GP mean chi-square (ref 1.5) | slope filter mean chi-square | causal GP skill | slope filter skill | running mean skill | error ratio (ref 1) |
|---|---|---|---|---|---|---|---|---|---|---|
| M6 | 33 | 16 | 6734 | +0.109 [+0.056, +0.337] | 1.73 [1.12, 2.74] | 1.70 [1.12, 2.64] | +0.098 [+0.016, +0.162] | +0.083 [-0.000, +0.161] | -0.003 [-0.011, +0.002] | 0.701 [0.551, 1.013] |
| M7 | 32 | 19 | 14208 | +0.226 [+0.101, +0.339] | 3.98 [2.40, 5.90] | 4.34 [2.74, 6.22] | +0.149 [+0.011, +0.353] | -0.054 [-0.301, +0.110] | -0.009 [-0.017, -0.000] | 0.792 [0.627, 0.964] |
| M8 | 35 | 29 | 22155 | +0.345 [+0.197, +0.595] | 3.00 [1.97, 4.62] | 3.13 [2.12, 4.57] | +0.168 [+0.058, +0.290] | +0.177 [+0.088, +0.270] | -0.018 [-0.029, -0.008] | 0.530 [0.441, 0.676] |
| M9 | 31 | 24 | 15747 | +0.359 [+0.205, +0.497] | 1.07 [0.70, 1.52] | 1.09 [0.71, 1.55] | +0.074 [+0.022, +0.125] | +0.056 [-0.006, +0.123] | -0.012 [-0.023, -0.002] | 0.358 [0.269, 0.518] |

**V_i**

| stratum | TEST shots | with rows | rows | regime number (ref 0) | causal GP mean chi-square (ref 1.5) | slope filter mean chi-square | causal GP skill | slope filter skill | running mean skill | error ratio (ref 1) |
|---|---|---|---|---|---|---|---|---|---|---|
| M6 | 33 | 16 | 6734 | +0.383 [+0.210, +0.460] | 13.07 [9.36, 19.36] | 13.64 [9.29, 21.29] | +0.051 [-0.045, +0.143] | +0.090 [-0.012, +0.204] | -0.019 [-0.028, -0.010] | 2.538 [1.925, 3.905] |
| M7 | 32 | 19 | 14208 | +0.419 [+0.330, +0.590] | 24.39 [16.71, 32.97] | 29.11 [20.69, 38.65] | +0.088 [+0.015, +0.227] | +0.066 [-0.028, +0.237] | -0.018 [-0.035, -0.010] | 3.150 [2.609, 3.756] |
| M8 | 35 | 29 | 22155 | +0.569 [+0.382, +0.605] | 19.55 [12.93, 27.79] | 24.58 [17.34, 33.09] | +0.159 [+0.063, +0.259] | +0.131 [+0.040, +0.240] | -0.030 [-0.039, -0.023] | 2.721 [2.256, 3.412] |
| M9 | 31 | 24 | 15747 | +0.524 [+0.418, +0.606] | 7.25 [5.04, 9.58] | 7.74 [5.36, 10.20] | +0.098 [-0.025, +0.224] | +0.174 [+0.051, +0.312] | -0.025 [-0.036, -0.015] | 1.366 [0.926, 2.109] |

### Heating (catalogue)

**T_i**

| stratum | TEST shots | with rows | rows | regime number (ref 0) | causal GP mean chi-square (ref 1.5) | slope filter mean chi-square | causal GP skill | slope filter skill | running mean skill | error ratio (ref 1) |
|---|---|---|---|---|---|---|---|---|---|---|
| 2 beams | 40 | 34 | 26424 | +0.302 [+0.181, +0.423] | 2.65 [1.77, 4.05] | 2.70 [1.82, 3.90] | +0.129 [+0.036, +0.236] | +0.132 [+0.051, +0.217] | -0.015 [-0.025, -0.006] | 0.511 [0.406, 0.661] |
| Ohmic | 1 | 1 | 374 | +0.687 (too few) | 2.30 (too few) | 2.35 (too few) | +0.008 (too few) | -0.103 (too few) | -0.027 (too few) | 0.406 (too few) |
| SS Beam | 25 | 18 | 11104 | +0.474 [+0.246, +0.573] | 1.11 [0.67, 1.88] | 1.29 [0.72, 2.27] | +0.204 [+0.104, +0.293] | +0.203 [+0.127, +0.271] | -0.026 [-0.032, -0.019] | 0.342 [0.262, 0.474] |
| no record | 65 | 35 | 20942 | +0.159 [+0.093, +0.315] | 3.25 [2.05, 4.78] | 3.49 [2.26, 5.01] | +0.129 [+0.034, +0.256] | -0.001 [-0.153, +0.110] | -0.007 [-0.013, -0.001] | 0.759 [0.630, 0.911] |

**V_i**

| stratum | TEST shots | with rows | rows | regime number (ref 0) | causal GP mean chi-square (ref 1.5) | slope filter mean chi-square | causal GP skill | slope filter skill | running mean skill | error ratio (ref 1) |
|---|---|---|---|---|---|---|---|---|---|---|
| 2 beams | 40 | 34 | 26424 | +0.467 [+0.347, +0.597] | 16.04 [10.84, 22.92] | 19.09 [13.28, 26.36] | +0.096 [+0.002, +0.189] | +0.111 [+0.021, +0.215] | -0.026 [-0.034, -0.018] | 2.391 [1.912, 3.098] |
| Ohmic | 1 | 1 | 374 | +0.596 (too few) | 7.68 (too few) | 8.83 (too few) | +0.204 (too few) | +0.387 (too few) | -0.036 (too few) | 2.128 (too few) |
| SS Beam | 25 | 18 | 11104 | +0.585 [+0.501, +0.644] | 10.87 [4.83, 20.86] | 14.29 [6.18, 25.99] | +0.287 [+0.221, +0.367] | +0.234 [+0.158, +0.317] | -0.039 [-0.045, -0.032] | 1.354 [0.825, 2.299] |
| no record | 65 | 35 | 20942 | +0.398 [+0.330, +0.492] | 20.75 [14.47, 27.99] | 24.13 [16.88, 32.40] | +0.076 [+0.020, +0.163] | +0.074 [-0.007, +0.184] | -0.019 [-0.029, -0.012] | 2.946 [2.402, 3.555] |

### RMP coil (catalogue)

**T_i**

| stratum | TEST shots | with rows | rows | regime number (ref 0) | causal GP mean chi-square (ref 1.5) | slope filter mean chi-square | causal GP skill | slope filter skill | running mean skill | error ratio (ref 1) |
|---|---|---|---|---|---|---|---|---|---|---|
| False | 34 | 29 | 18619 | +0.497 [+0.197, +0.687] | 2.63 [1.46, 4.49] | 2.79 [1.58, 4.37] | +0.172 [+0.052, +0.305] | +0.165 [+0.066, +0.273] | -0.019 [-0.032, -0.008] | 0.468 [0.362, 0.606] |
| True | 13 | 10 | 8279 | +0.302 [+0.220, +0.392] | 1.87 [0.80, 3.30] | 1.97 [0.82, 3.69] | +0.140 [+0.032, +0.216] | +0.200 [+0.070, +0.268] | -0.024 [-0.031, -0.013] | 0.394 [0.260, 0.640] |
| no record | 84 | 49 | 31946 | +0.196 [+0.101, +0.337] | 2.73 [1.91, 3.78] | 2.86 [1.98, 3.92] | +0.116 [+0.041, +0.213] | +0.014 [-0.109, +0.100] | -0.007 [-0.012, -0.002] | 0.643 [0.522, 0.787] |

**V_i**

| stratum | TEST shots | with rows | rows | regime number (ref 0) | causal GP mean chi-square (ref 1.5) | slope filter mean chi-square | causal GP skill | slope filter skill | running mean skill | error ratio (ref 1) |
|---|---|---|---|---|---|---|---|---|---|---|
| False | 34 | 29 | 18619 | +0.569 [+0.325, +0.606] | 19.11 [11.85, 28.55] | 23.14 [14.54, 33.06] | +0.160 [+0.035, +0.284] | +0.142 [+0.024, +0.284] | -0.032 [-0.042, -0.023] | 2.641 [2.029, 3.465] |
| True | 13 | 10 | 8279 | +0.497 [+0.382, +0.599] | 8.16 [4.44, 12.41] | 11.56 [6.39, 17.51] | +0.244 [+0.177, +0.282] | +0.265 [+0.198, +0.299] | -0.040 [-0.044, -0.032] | 1.279 [0.725, 2.297] |
| no record | 84 | 49 | 31946 | +0.456 [+0.371, +0.545] | 17.48 [12.77, 22.93] | 20.20 [14.39, 26.76] | +0.075 [+0.027, +0.139] | +0.078 [+0.013, +0.160] | -0.019 [-0.026, -0.013] | 2.532 [2.062, 3.057] |

### CXRS chord columns (stored arrays)

**T_i**

| stratum | TEST shots | with rows | rows | regime number (ref 0) | causal GP mean chi-square (ref 1.5) | slope filter mean chi-square | causal GP skill | slope filter skill | running mean skill | error ratio (ref 1) |
|---|---|---|---|---|---|---|---|---|---|---|
| 160 | 80 | 56 | 34777 | +0.278 [+0.139, +0.384] | 2.82 [1.90, 4.03] | 2.86 [1.99, 3.97] | +0.115 [+0.045, +0.199] | +0.118 [+0.057, +0.189] | -0.013 [-0.021, -0.007] | 0.498 [0.423, 0.596] |
| 96 | 51 | 32 | 24067 | +0.307 [+0.196, +0.386] | 2.22 [1.42, 3.27] | 2.51 [1.59, 3.64] | +0.174 [+0.064, +0.328] | -0.004 [-0.215, +0.126] | -0.010 [-0.017, -0.002] | 0.611 [0.455, 0.812] |

**V_i**

| stratum | TEST shots | with rows | rows | regime number (ref 0) | causal GP mean chi-square (ref 1.5) | slope filter mean chi-square | causal GP skill | slope filter skill | running mean skill | error ratio (ref 1) |
|---|---|---|---|---|---|---|---|---|---|---|
| 160 | 80 | 56 | 34777 | +0.467 [+0.416, +0.582] | 16.95 [12.12, 23.27] | 20.46 [15.22, 26.81] | +0.103 [+0.035, +0.187] | +0.087 [+0.011, +0.186] | -0.023 [-0.031, -0.017] | 2.422 [2.021, 2.944] |
| 96 | 51 | 32 | 24067 | +0.450 [+0.356, +0.554] | 16.30 [10.95, 22.16] | 19.12 [12.30, 26.58] | +0.121 [+0.058, +0.208] | +0.144 [+0.076, +0.240] | -0.024 [-0.035, -0.014] | 2.158 [1.494, 2.996] |

### Frame spacing (stored times)

**T_i**

| stratum | TEST shots | with rows | rows | regime number (ref 0) | causal GP mean chi-square (ref 1.5) | slope filter mean chi-square | causal GP skill | slope filter skill | running mean skill | error ratio (ref 1) |
|---|---|---|---|---|---|---|---|---|---|---|
| 10.0 ms | 2 | 0 | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| 5 ms | 129 | 88 | 58844 | +0.297 [+0.197, +0.358] | 2.57 [1.92, 3.39] | 2.71 [2.05, 3.49] | +0.136 [+0.073, +0.212] | +0.076 [-0.020, +0.148] | -0.012 [-0.017, -0.007] | 0.540 [0.463, 0.635] |

**V_i**

| stratum | TEST shots | with rows | rows | regime number (ref 0) | causal GP mean chi-square (ref 1.5) | slope filter mean chi-square | causal GP skill | slope filter skill | running mean skill | error ratio (ref 1) |
|---|---|---|---|---|---|---|---|---|---|---|
| 10.0 ms | 2 | 0 | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| 5 ms | 129 | 88 | 58844 | +0.463 [+0.416, +0.554] | 16.69 [12.90, 20.95] | 19.91 [15.37, 24.60] | +0.110 [+0.058, +0.167] | +0.108 [+0.049, +0.175] | -0.024 [-0.030, -0.018] | 2.314 [1.924, 2.774] |

### Plasma shape (auxiliary)

**T_i**

| stratum | TEST shots | with rows | rows | regime number (ref 0) | causal GP mean chi-square (ref 1.5) | slope filter mean chi-square | causal GP skill | slope filter skill | running mean skill | error ratio (ref 1) |
|---|---|---|---|---|---|---|---|---|---|---|
| Connected Double Null | 30 | 23 | 14302 | +0.573 [+0.392, +0.709] | 1.50 [1.07, 2.03] | 1.83 [1.15, 2.75] | +0.266 [+0.161, +0.389] | +0.256 [+0.169, +0.345] | -0.028 [-0.040, -0.015] | 0.361 [0.312, 0.411] |
| Lower Single Null | 17 | 16 | 12596 | +0.212 [+0.121, +0.298] | 3.41 [1.62, 6.13] | 3.34 [1.69, 5.76] | -0.061 [-0.165, +0.018] | -0.026 [-0.106, +0.072] | -0.001 [-0.010, +0.006] | 0.572 [0.360, 0.938] |
| no record | 84 | 49 | 31946 | +0.196 [+0.097, +0.334] | 2.73 [1.91, 3.69] | 2.86 [2.00, 3.87] | +0.116 [+0.043, +0.215] | +0.014 [-0.115, +0.099] | -0.007 [-0.012, -0.002] | 0.643 [0.524, 0.800] |

**V_i**

| stratum | TEST shots | with rows | rows | regime number (ref 0) | causal GP mean chi-square (ref 1.5) | slope filter mean chi-square | causal GP skill | slope filter skill | running mean skill | error ratio (ref 1) |
|---|---|---|---|---|---|---|---|---|---|---|
| Connected Double Null | 30 | 23 | 14302 | +0.602 [+0.529, +0.678] | 10.77 [8.75, 13.10] | 16.28 [11.13, 23.28] | +0.294 [+0.226, +0.373] | +0.240 [+0.090, +0.381] | -0.042 [-0.048, -0.036] | 2.038 [1.715, 2.374] |
| Lower Single Null | 17 | 16 | 12596 | +0.371 [+0.217, +0.458] | 21.39 [10.37, 35.77] | 23.31 [12.37, 37.54] | -0.060 [-0.224, +0.130] | +0.014 [-0.081, +0.136] | -0.016 [-0.031, -0.003] | 2.109 [1.133, 4.198] |
| no record | 84 | 49 | 31946 | +0.456 [+0.369, +0.540] | 17.48 [12.92, 22.79] | 20.20 [14.86, 26.32] | +0.075 [+0.025, +0.137] | +0.078 [+0.014, +0.160] | -0.019 [-0.026, -0.013] | 2.532 [2.083, 3.089] |

### Current range (auxiliary)

**T_i**

| stratum | TEST shots | with rows | rows | regime number (ref 0) | causal GP mean chi-square (ref 1.5) | slope filter mean chi-square | causal GP skill | slope filter skill | running mean skill | error ratio (ref 1) |
|---|---|---|---|---|---|---|---|---|---|---|
| 1000 kA | 11 | 9 | 4747 | +0.709 [+0.444, +0.745] | 1.27 [0.71, 2.15] | 1.13 [0.63, 1.88] | +0.240 [+0.044, +0.428] | +0.253 [+0.125, +0.372] | -0.018 [-0.031, -0.002] | 0.298 [0.237, 0.384] |
| 400 kA | 9 | 7 | 6931 | +0.205 (too few) | 0.48 (too few) | 0.49 (too few) | +0.116 (too few) | +0.054 (too few) | -0.005 (too few) | 0.244 (too few) |
| 700 kA | 46 | 37 | 26224 | +0.306 [+0.197, +0.452] | 2.82 [1.97, 4.25] | 2.96 [2.11, 4.21] | +0.107 [+0.028, +0.185] | +0.107 [+0.041, +0.183] | -0.017 [-0.026, -0.008] | 0.585 [0.480, 0.731] |
| no record | 65 | 35 | 20942 | +0.159 [+0.096, +0.307] | 3.25 [2.09, 4.73] | 3.49 [2.32, 5.03] | +0.129 [+0.033, +0.246] | -0.001 [-0.148, +0.110] | -0.007 [-0.013, -0.001] | 0.759 [0.629, 0.912] |

**V_i**

| stratum | TEST shots | with rows | rows | regime number (ref 0) | causal GP mean chi-square (ref 1.5) | slope filter mean chi-square | causal GP skill | slope filter skill | running mean skill | error ratio (ref 1) |
|---|---|---|---|---|---|---|---|---|---|---|
| 1000 kA | 11 | 9 | 4747 | +0.606 [+0.552, +0.723] | 8.36 [6.24, 10.87] | 7.67 [5.77, 9.41] | +0.254 [+0.217, +0.293] | +0.410 [+0.311, +0.505] | -0.045 [-0.050, -0.037] | 1.650 [1.219, 2.232] |
| 400 kA | 9 | 7 | 6931 | +0.606 (too few) | 3.40 (too few) | 4.03 (too few) | +0.158 (too few) | +0.246 (too few) | -0.031 (too few) | 0.814 (too few) |
| 700 kA | 46 | 37 | 26224 | +0.456 [+0.303, +0.569] | 18.46 [12.92, 25.84] | 22.96 [16.83, 30.62] | +0.126 [+0.029, +0.214] | +0.095 [+0.019, +0.179] | -0.026 [-0.034, -0.019] | 2.810 [2.323, 3.481] |
| no record | 65 | 35 | 20942 | +0.398 [+0.330, +0.494] | 20.75 [14.87, 27.70] | 24.13 [17.22, 32.09] | +0.076 [+0.013, +0.162] | +0.074 [-0.010, +0.181] | -0.019 [-0.029, -0.011] | 2.946 [2.425, 3.530] |

## 3. Counts over the 264 gate shots with data (TUNE and TEST)

### Campaign (primary)

| stratum | with data | usable spacing | TEST | G: no block | G: block, no scored chord | G: block and chords | W: no block | W: block, no scored chord | W: block and chords |
|---|---|---|---|---|---|---|---|---|---|
| M6 | 67 | 67 | 33 | 19 | 19 | 29 | 13 | 4 | 50 |
| M7 | 64 | 61 | 32 | 12 | 16 | 33 | 11 | 4 | 46 |
| M8 | 70 | 70 | 35 | 6 | 6 | 58 | 5 | 0 | 65 |
| M9 | 63 | 63 | 31 | 4 | 10 | 49 | 4 | 2 | 57 |

### Heating (catalogue)

| stratum | with data | usable spacing | TEST | G: no block | G: block, no scored chord | G: block and chords | W: no block | W: block, no scored chord | W: block and chords |
|---|---|---|---|---|---|---|---|---|---|
| 2 beams | 82 | 82 | 40 | 4 | 7 | 71 | 4 | 0 | 78 |
| Ohmic | 2 | 2 | 1 | 1 | 0 | 1 | 0 | 0 | 2 |
| SS Beam | 49 | 49 | 25 | 5 | 9 | 35 | 5 | 2 | 42 |
| no record | 131 | 128 | 65 | 31 | 35 | 62 | 24 | 8 | 96 |

### RMP coil (catalogue)

| stratum | with data | usable spacing | TEST | G: no block | G: block, no scored chord | G: block and chords | W: no block | W: block, no scored chord | W: block and chords |
|---|---|---|---|---|---|---|---|---|---|
| False | 65 | 65 | 34 | 4 | 8 | 53 | 3 | 1 | 61 |
| True | 31 | 31 | 13 | 0 | 4 | 27 | 0 | 0 | 31 |
| no record | 168 | 165 | 84 | 37 | 39 | 89 | 30 | 9 | 126 |

### CXRS chord columns (stored arrays)

| stratum | with data | usable spacing | TEST | G: no block | G: block, no scored chord | G: block and chords | W: no block | W: block, no scored chord | W: block and chords |
|---|---|---|---|---|---|---|---|---|---|
| 160 | 165 | 163 | 80 | 20 | 30 | 113 | 19 | 3 | 141 |
| 96 | 99 | 98 | 51 | 21 | 21 | 56 | 14 | 7 | 77 |

### Frame spacing (stored times)

| stratum | with data | usable spacing | TEST | G: no block | G: block, no scored chord | G: block and chords | W: no block | W: block, no scored chord | W: block and chords |
|---|---|---|---|---|---|---|---|---|---|
| 10.0 ms | 3 | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 0 |
| 5 ms | 261 | 261 | 129 | 41 | 51 | 169 | 33 | 10 | 218 |

### Plasma shape (auxiliary)

| stratum | with data | usable spacing | TEST | G: no block | G: block, no scored chord | G: block and chords | W: no block | W: block, no scored chord | W: block and chords |
|---|---|---|---|---|---|---|---|---|---|
| Connected Double Null | 58 | 58 | 30 | 3 | 11 | 44 | 3 | 1 | 54 |
| Lower Single Null | 38 | 38 | 17 | 1 | 1 | 36 | 0 | 0 | 38 |
| no record | 168 | 165 | 84 | 37 | 39 | 89 | 30 | 9 | 126 |

### Current range (auxiliary)

| stratum | with data | usable spacing | TEST | G: no block | G: block, no scored chord | G: block and chords | W: no block | W: block, no scored chord | W: block and chords |
|---|---|---|---|---|---|---|---|---|---|
| 1000 kA | 17 | 17 | 11 | 1 | 2 | 14 | 1 | 0 | 16 |
| 400 kA | 26 | 26 | 9 | 3 | 2 | 21 | 3 | 2 | 21 |
| 700 kA | 90 | 90 | 46 | 6 | 12 | 72 | 5 | 0 | 85 |
| no record | 131 | 128 | 65 | 31 | 35 | 62 | 24 | 8 | 96 |

## 4. Notes

- Strata are confounded with each other and with campaign; half of the heating, RMP and shape tags are missing. No causal reading is attached to a configuration stratum.
- The stratum bootstrap resamples the TEST shots of that stratum (shots without scored rows included, as in the gate); the seed sequence is (20261116, dimension index, stratum index).
- Counts use the T_i masks; the V_i counts equal them in both populations (gate report).
- `chord_columns` is the column count of `act_ss_temperature`; `frame_spacing` is the stored median frame interval, "5 ms" meaning inside 5.0 +- 0.1 ms (the usable rule).

## 5. Stratum (f): normalized radius

rho = (R_chord - R_axis) / (R_out - R_axis) from the catalogue EFIT scalars, one value per discharge (time variation inside the beam block ignored), low-field side only. Geometry source over the TEST shots: {'efit': 122, 'd_alpha_outer': 0, 'missing': 9}; median R_axis 0.912 m, median R_out 1.396 m. The KSTAR-matched bin is 0.7-0.9 (nominal KSTAR geometry, (2.198 - 1.8) / 0.5 = 0.80; not an EFIT mapping of the KSTAR discharges). Intervals: 2,000 bootstrap resamples of TEST shots.

**T_i, population G** - rows in a bin 43941 of 58844; scored chords outside the bins: {'rho_ge_1': 12, 'high_field_side': 476, 'no_geometry_or_radius': 110}

| rho | shots with rows | rows | regime number (ref 0) | shots with a regime number | causal GP mean chi-square (ref 1.5) | causal GP median chi-square (ref 1) | causal GP skill | slope filter skill | running mean skill | error ratio (ref 1) |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.0-0.3 | 75 | 19052 | +0.271 [+0.145, +0.379] | 75 | 2.19 [1.67, 2.84] | 0.401 [0.327, 0.491] | +0.127 [+0.068, +0.194] | +0.113 [+0.057, +0.169] | -0.013 [-0.018, -0.007] | 0.493 [0.421, 0.581] |
| 0.3-0.5 | 66 | 12380 | +0.346 [+0.198, +0.420] | 66 | 3.41 [2.45, 4.55] | 0.439 [0.345, 0.555] | +0.101 [+0.037, +0.165] | +0.094 [+0.042, +0.148] | -0.019 [-0.027, -0.012] | 0.569 [0.473, 0.695] |
| 0.5-0.7 | 54 | 8567 | +0.346 [+0.167, +0.424] | 54 | 3.22 [2.01, 5.09] | 0.465 [0.325, 0.626] | +0.061 [-0.064, +0.168] | +0.070 [-0.001, +0.142] | -0.014 [-0.022, -0.006] | 0.651 [0.491, 0.847] |
| 0.7-0.9 | 26 | 3256 | +0.103 [-0.079, +0.257] | 26 | 4.44 [1.54, 9.88] | 0.480 [0.287, 0.824] | -0.095 [-0.363, +0.088] | +0.031 [-0.056, +0.098] | -0.005 [-0.012, +0.004] | 0.833 [0.536, 1.239] |
| 0.9-1.0 | 6 | 686 | +0.435 (too few) | 6 | 1.43 (too few) | 0.482 (too few) | +0.153 (too few) | +0.214 (too few) | -0.017 (too few) | 0.703 (too few) |

**T_i, population W** - rows in a bin 97265 of 138795; scored chords outside the bins: {'rho_ge_1': 109, 'high_field_side': 1139, 'no_geometry_or_radius': 244}

| rho | shots with rows | rows | regime number (ref 0) | shots with a regime number | causal GP mean chi-square (ref 1.5) | causal GP median chi-square (ref 1) | causal GP skill | slope filter skill | running mean skill | error ratio (ref 1) |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.0-0.3 | 105 | 37335 | +0.165 [+0.121, +0.283] | 105 | 2.16 [1.63, 2.90] | 0.348 [0.305, 0.404] | +0.048 [-0.070, +0.155] | -0.035 [-0.136, +0.060] | -0.002 [-0.006, +0.002] | 0.423 [0.380, 0.478] |
| 0.3-0.5 | 99 | 23548 | +0.287 [+0.223, +0.335] | 99 | 2.77 [2.11, 3.49] | 0.391 [0.337, 0.453] | +0.111 [+0.037, +0.153] | +0.042 [-0.028, +0.091] | -0.007 [-0.016, -0.001] | 0.497 [0.446, 0.560] |
| 0.5-0.7 | 93 | 20197 | +0.341 [+0.230, +0.403] | 93 | 2.88 [2.13, 3.79] | 0.365 [0.306, 0.437] | +0.022 [-0.025, +0.081] | -0.027 [-0.187, +0.043] | -0.008 [-0.019, -0.003] | 0.514 [0.445, 0.599] |
| 0.7-0.9 | 72 | 13286 | +0.136 [+0.036, +0.230] | 72 | 2.68 [1.73, 4.18] | 0.327 [0.268, 0.394] | +0.060 [-0.238, +0.443] | -0.035 [-0.217, +0.125] | -0.001 [-0.014, +0.015] | 0.576 [0.490, 0.689] |
| 0.9-1.0 | 29 | 2899 | -0.112 [-0.296, -0.000] | 29 | 4.64 [1.74, 10.04] | 0.302 [0.203, 0.472] | +0.006 [-0.424, +0.247] | -0.290 [-0.553, +0.077] | +0.006 [-0.007, +0.013] | 0.552 [0.376, 0.805] |

**V_i, population G** - rows in a bin 43941 of 58844; scored chords outside the bins: {'rho_ge_1': 12, 'high_field_side': 476, 'no_geometry_or_radius': 110}

| rho | shots with rows | rows | regime number (ref 0) | shots with a regime number | causal GP mean chi-square (ref 1.5) | causal GP median chi-square (ref 1) | causal GP skill | slope filter skill | running mean skill | error ratio (ref 1) |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.0-0.3 | 75 | 19052 | +0.478 [+0.406, +0.546] | 75 | 18.20 [13.32, 24.23] | 2.650 [2.001, 3.412] | +0.114 [+0.041, +0.188] | +0.103 [+0.026, +0.192] | -0.025 [-0.032, -0.017] | 2.195 [1.749, 2.763] |
| 0.3-0.5 | 66 | 12380 | +0.499 [+0.416, +0.598] | 66 | 15.50 [12.16, 19.14] | 2.474 [1.873, 3.224] | +0.161 [+0.096, +0.226] | +0.132 [+0.054, +0.209] | -0.030 [-0.035, -0.025] | 2.590 [2.029, 3.295] |
| 0.5-0.7 | 54 | 8567 | +0.401 [+0.282, +0.495] | 54 | 18.28 [12.86, 25.44] | 2.525 [1.915, 3.361] | +0.061 [-0.048, +0.157] | +0.087 [+0.012, +0.158] | -0.021 [-0.028, -0.014] | 3.038 [2.274, 4.014] |
| 0.7-0.9 | 26 | 3256 | +0.147 [+0.038, +0.337] | 26 | 40.19 [11.13, 78.94] | 2.636 [1.605, 4.742] | -0.036 [-0.313, +0.020] | -0.009 [-0.087, +0.051] | -0.002 [-0.019, +0.001] | 3.490 [2.088, 6.269] |
| 0.9-1.0 | 6 | 686 | +0.212 (too few) | 6 | 13.99 (too few) | 3.623 (too few) | -0.083 (too few) | +0.044 (too few) | -0.015 (too few) | 3.768 (too few) |

**V_i, population W** - rows in a bin 97265 of 138795; scored chords outside the bins: {'rho_ge_1': 109, 'high_field_side': 1139, 'no_geometry_or_radius': 244}

| rho | shots with rows | rows | regime number (ref 0) | shots with a regime number | causal GP mean chi-square (ref 1.5) | causal GP median chi-square (ref 1) | causal GP skill | slope filter skill | running mean skill | error ratio (ref 1) |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.0-0.3 | 105 | 37335 | +0.386 [+0.331, +0.455] | 105 | 17.55 [12.96, 23.20] | 1.930 [1.553, 2.406] | +0.032 [-0.021, +0.098] | +0.049 [+0.016, +0.093] | -0.015 [-0.021, -0.011] | 1.584 [1.347, 1.880] |
| 0.3-0.5 | 99 | 23548 | +0.465 [+0.379, +0.501] | 99 | 13.33 [10.95, 16.22] | 1.939 [1.610, 2.305] | +0.131 [+0.093, +0.170] | +0.114 [+0.067, +0.161] | -0.027 [-0.031, -0.023] | 1.956 [1.648, 2.292] |
| 0.5-0.7 | 93 | 20197 | +0.393 [+0.322, +0.438] | 93 | 14.99 [11.95, 18.50] | 1.892 [1.584, 2.233] | +0.054 [+0.014, +0.123] | +0.056 [+0.022, +0.116] | -0.015 [-0.025, -0.008] | 2.226 [1.888, 2.625] |
| 0.7-0.9 | 72 | 13286 | +0.149 [+0.064, +0.219] | 72 | 22.27 [13.73, 33.65] | 1.589 [1.283, 2.009] | -0.088 [-0.177, +0.121] | -0.096 [-0.160, +0.010] | -0.001 [-0.007, +0.005] | 2.245 [1.844, 2.813] |
| 0.9-1.0 | 29 | 2899 | -0.177 [-0.279, +0.042] | 29 | 28.14 [13.96, 45.64] | 1.619 [1.035, 2.692] | -0.379 [-0.594, -0.124] | -0.117 [-0.202, -0.017] | +0.002 [-0.008, +0.011] | 2.449 [1.667, 3.748] |

### Does the measurement weaken outward (samples before the fit rule)

Usable TEST shots with a population-W beam block, geometry and a radius per chord column: 112 (skipped: {'unusable': 2, 'no_W_block': 12, 'no_geometry': 5, 'radius_columns_mismatch': 0}); emissivity columns match the chords in 109 of them. Every low-field chord column with at least one finite T_i value in the block, every block frame.

| rho | shots | samples | fail share, cut 0.1 | fail share, cut 0.5 | non-finite share | median relative T_i error | median V_i error (km/s) | median emissivity (shots) |
|---|---|---|---|---|---|---|---|---|
| 0.0-0.3 | 112 | 91023 | 0.231 [0.189, 0.276] | 0.045 [0.030, 0.061] | 0.001 [0.000, 0.003] | 0.050 [0.045, 0.056] | 1.59 [1.46, 1.76] | 92.493 [60.795, 13437905180347584.000] (109) |
| 0.3-0.5 | 112 | 62028 | 0.226 [0.185, 0.273] | 0.077 [0.052, 0.105] | 0.001 [0.000, 0.003] | 0.048 [0.043, 0.053] | 1.39 [1.27, 1.55] | 106.386 [64.026, 12343410772305454.000] (109) |
| 0.5-0.7 | 112 | 63211 | 0.262 [0.222, 0.308] | 0.105 [0.078, 0.135] | 0.001 [0.000, 0.003] | 0.047 [0.043, 0.053] | 1.14 [1.05, 1.23] | 131.309 [71.741, 8555510093578221.000] (109) |
| 0.7-0.9 | 112 | 64524 | 0.430 [0.384, 0.478] | 0.184 [0.151, 0.221] | 0.001 [0.000, 0.003] | 0.072 [0.063, 0.084] | 1.10 [1.01, 1.21] | 181.588 [90.450, 2862930529039123.000] (109) |
| 0.9-1.0 | 105 | 29995 | 0.653 [0.593, 0.715] | 0.348 [0.299, 0.399] | 0.002 [0.001, 0.003] | 0.171 [0.128, 0.223] | 1.63 [1.39, 1.97] | 137.061 [58.718, 325159700607795.062] (103) |

The emissivity intervals are not usable: a few shots store emissivity on a scale about 1e14 times larger than the rest, so the upper bootstrap bounds jump to that scale. The medians are reported as stored; no statement rests on them.

### Statements S5-S8

- S5, T_i: regime number in rho 0.7-0.9 - population G +0.103 [-0.079, +0.257], includes 0; population W +0.136 [+0.036, +0.230], above 0. Read on population G: does not hold.
- S5, V_i: regime number in rho 0.7-0.9 - population G +0.147 [+0.038, +0.337], above 0; population W +0.149 [+0.064, +0.219], above 0. Read on population G: holds.
- S6, T_i, population G: causal GP mean reduced chi-square over bins ['0.0-0.3', '0.3-0.5', '0.5-0.7', '0.7-0.9'] = [2.19, 3.41, 3.22, 4.44]; non-decreasing: no (registered population).
- S6, T_i, population W: causal GP mean reduced chi-square over bins ['0.0-0.3', '0.3-0.5', '0.5-0.7', '0.7-0.9', '0.9-1.0'] = [2.16, 2.77, 2.88, 2.68, 4.64]; non-decreasing: no (co-reported).
- S6, V_i, population G: causal GP mean reduced chi-square over bins ['0.0-0.3', '0.3-0.5', '0.5-0.7', '0.7-0.9'] = [18.2, 15.5, 18.28, 40.19]; non-decreasing: no (registered population).
- S6, V_i, population W: causal GP mean reduced chi-square over bins ['0.0-0.3', '0.3-0.5', '0.5-0.7', '0.7-0.9', '0.9-1.0'] = [17.55, 13.33, 14.99, 22.27, 28.14]; non-decreasing: no (co-reported).
- S7, fail share at the cut 0.1: bins ['0.0-0.3', '0.3-0.5', '0.5-0.7', '0.7-0.9', '0.9-1.0'] = [0.231, 0.226, 0.262, 0.43, 0.653]; non-decreasing: no; largest in 0.9-1.0: yes; does not hold.
- S7, median relative T_i error: bins ['0.0-0.3', '0.3-0.5', '0.5-0.7', '0.7-0.9', '0.9-1.0'] = [0.05, 0.048, 0.047, 0.072, 0.171]; non-decreasing: no; largest in 0.9-1.0: yes; does not hold.
- S7 overall: does not hold.
- S8, T_i regime number, population W: bins ['0.0-0.3', '0.3-0.5', '0.5-0.7', '0.7-0.9', '0.9-1.0'] = [0.165, 0.287, 0.341, 0.136, -0.112]; non-increasing: no; difference (0.7-0.9 minus 0.0-0.3) -0.029 [-0.185, +0.070], includes 0; does not hold (registered population).
- S8, T_i regime number, population G: bins ['0.0-0.3', '0.3-0.5', '0.5-0.7', '0.7-0.9'] = [0.271, 0.346, 0.346, 0.103]; non-increasing: no; difference (0.7-0.9 minus 0.0-0.3) -0.167 [-0.377, +0.018], includes 0; does not hold (co-reported).
