# B.39 — the causal variant as the primary estimator, and which diagnostic carries the increment

Registered: PREREGISTRATION_B39.md. GPU runs; read-outs on a CPU. T_i, cut population, epoch cap 80 for every network, fast channels standardized with the statistics of the training discharges. Skill = 1 - sum of squared errors of A over that of R on the same test rows.

Splits read: 4 of 4.

- **S1** (causal variant over the causal GP; interval over initialisations and discharges): 2 of 4.
- **S2** (causal variant over the history-only twin): 2 of 4.
- **D1** (ECEI-only variant over the twin): 1 of 4.
- **D2** (BES-only variant over the twin): 4 of 4.

## Arm A: four file-level splits

| split | rows | initialisations | S1: causal variant over causal GP | S2: over the twin | twin over causal GP | spread of the skill over GP between initialisations | over hold | over PCHIP | over the offline GP |
|---|---|---|---|---|---|---|---|---|---|
| 42 | 32,589 | 42, 1, 7 | +0.067 [-0.014, +0.120] | +0.071 [-0.012, +0.146] | -0.005 [-0.074, +0.046] | 0.023 | +0.352 [+0.284, +0.401] | +0.164 [+0.057, +0.238] | -0.065 [-0.170, +0.009] |
| 1 | 35,582 | 1, 7, 123 | +0.100 [+0.068, +0.129] | +0.041 [+0.006, +0.082] | +0.061 [+0.019, +0.098] | 0.009 | +0.383 [+0.347, +0.410] | +0.220 [+0.156, +0.269] | +0.002 [-0.040, +0.038] |
| 7 | 35,759 | 7, 123, 42 | +0.141 [+0.101, +0.182] | +0.089 [+0.037, +0.145] | +0.057 [+0.002, +0.097] | 0.006 | +0.405 [+0.372, +0.438] | +0.260 [+0.196, +0.314] | +0.023 [-0.034, +0.072] |
| 123 | 34,038 | 123, 42, 1 | +0.098 [-0.017, +0.162] | +0.062 [-0.046, +0.147] | +0.039 [-0.049, +0.082] | 0.067 | +0.411 [+0.320, +0.458] | +0.259 [+0.148, +0.330] | -0.013 [-0.157, +0.063] |

Ablations of the fast-channel groups (initialisation = split seed for every network; 2,000 resamples of discharges):

| split | D1: ECEI-only over twin | D2: BES-only over twin | no-Mirnov over twin | full over twin | full over ECEI-only | full over BES-only | full over no-Mirnov |
|---|---|---|---|---|---|---|---|
| 42 | +0.010 [-0.016, +0.034] | +0.074 [+0.039, +0.112] | +0.048 [-0.021, +0.100] | +0.071 [+0.036, +0.105] | +0.062 [+0.028, +0.098] | -0.003 [-0.026, +0.017] | +0.024 [-0.007, +0.076] |
| 1 | +0.001 [-0.014, +0.015] | +0.026 [+0.007, +0.047] | +0.043 [+0.009, +0.073] | +0.022 [+0.001, +0.043] | +0.022 [+0.003, +0.043] | -0.004 [-0.023, +0.015] | -0.022 [-0.045, +0.008] |
| 7 | +0.084 [+0.066, +0.101] | +0.105 [+0.072, +0.134] | +0.135 [+0.101, +0.170] | +0.123 [+0.095, +0.151] | +0.043 [+0.022, +0.066] | +0.021 [-0.009, +0.056] | -0.014 [-0.033, +0.007] |
| 123 | +0.038 [-0.000, +0.072] | +0.063 [+0.036, +0.095] | +0.074 [+0.037, +0.112] | +0.060 [+0.028, +0.099] | +0.024 [-0.008, +0.070] | -0.003 [-0.031, +0.022] | -0.015 [-0.033, +0.003] |

| split | over the causal GP: ECEI-only | BES-only | no-Mirnov | skill of each initialisation over the causal GP: causal variant / twin | median absolute error (eV): hold / causal variant / twin / ECEI-only / BES-only / no-Mirnov | epochs run (best) |
|---|---|---|---|---|---|---|
| 42 | +0.020 [-0.033, +0.063] | +0.084 [+0.041, +0.127] | +0.059 [-0.018, +0.115] | i42: +0.081 / +0.011; i1: +0.061 / -0.042; i7: +0.058 / +0.017 | 43.8 / 42.1 / 44.6 / 43.0 / 42.2 / 42.0 | v2 i42 53 (47), v2 i1 53 (47), v2 i7 40 (34), v2h i42 49 (43), v2h i1 12 (6), v2h i7 47 (41), v2e 48 (42), v2b 80 (76), v2n 43 (37) |
| 1 | +0.074 [+0.045, +0.101] | +0.098 [+0.070, +0.127] | +0.114 [+0.069, +0.146] | i1: +0.094 / +0.074; i7: +0.101 / +0.067; i123: +0.104 / +0.042 | 48.9 / 53.0 / 50.6 / 51.2 / 50.0 / 50.3 | v2 i1 22 (16), v2 i7 35 (29), v2 i123 22 (16), v2h i1 46 (40), v2h i7 49 (43), v2h i123 25 (19), v2e 27 (21), v2b 37 (31), v2n 37 (31) |
| 7 | +0.105 [+0.075, +0.132] | +0.126 [+0.092, +0.160] | +0.155 [+0.115, +0.200] | i7: +0.144 / +0.024; i123: +0.138 / +0.071; i42: +0.142 / +0.076 | 48.8 / 45.2 / 47.5 / 45.7 / 45.8 / 45.8 | v2 i7 44 (38), v2 i123 30 (24), v2 i42 50 (44), v2h i7 11 (5), v2h i123 63 (57), v2h i42 61 (55), v2e 50 (44), v2b 69 (63), v2n 46 (40) |
| 123 | +0.080 [+0.032, +0.120] | +0.104 [+0.070, +0.134] | +0.115 [+0.075, +0.154] | i123: +0.101 / +0.044; i42: +0.130 / +0.018; i1: +0.064 / +0.056 | 42.7 / 41.6 / 42.7 / 41.6 / 41.8 / 41.0 | v2 i123 26 (20), v2 i42 37 (31), v2 i1 54 (48), v2h i123 58 (52), v2h i42 74 (68), v2h i1 40 (34), v2e 50 (44), v2b 31 (25), v2n 31 (25) |

## Arm P: pooled cross-validation (repeats r2 to r9, five folds each)

Not read yet: 0 of 80 runs are complete.
