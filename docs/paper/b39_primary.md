# B.39 — the causal variant as the primary estimator, and which diagnostic carries the increment

Registered: PREREGISTRATION_B39.md. GPU runs; read-outs on a CPU. T_i, cut population, epoch cap 80 for every network, fast channels standardized with the statistics of the training discharges. Skill = 1 - sum of squared errors of A over that of R on the same test rows.

Splits read: 4 of 4 (some with the initialisation = split seed only; the further initialisations are not yet complete).

- **S1** (causal variant over the causal GP; interval over initialisations and discharges): 4 of 4.
- **S2** (causal variant over the history-only twin): 4 of 4.
- **D1** (ECEI-only variant over the twin): 1 of 4.
- **D2** (BES-only variant over the twin): 4 of 4.

## Arm A: four file-level splits

| split | rows | initialisations | S1: causal variant over causal GP | S2: over the twin | twin over causal GP | spread of the skill over GP between initialisations | over hold | over PCHIP | over the offline GP |
|---|---|---|---|---|---|---|---|---|---|
| 42 | 32,589 | 42 | +0.081 [+0.036, +0.122] | +0.071 [+0.037, +0.107] | +0.011 [-0.029, +0.046] | 0.000 | +0.362 [+0.313, +0.404] | +0.177 [+0.094, +0.242] | -0.049 [-0.124, +0.010] |
| 1 | 35,582 | 1 | +0.094 [+0.067, +0.120] | +0.022 [+0.001, +0.044] | +0.074 [+0.047, +0.102] | 0.000 | +0.379 [+0.348, +0.405] | +0.215 [+0.152, +0.261] | -0.005 [-0.047, +0.030] |
| 7 | 35,759 | 7 | +0.144 [+0.106, +0.179] | +0.123 [+0.096, +0.152] | +0.024 [-0.005, +0.049] | 0.000 | +0.407 [+0.375, +0.436] | +0.262 [+0.202, +0.313] | +0.026 [-0.028, +0.067] |
| 123 | 34,038 | 123 | +0.101 [+0.063, +0.140] | +0.060 [+0.025, +0.099] | +0.044 [+0.010, +0.076] | 0.000 | +0.413 [+0.367, +0.446] | +0.262 [+0.188, +0.316] | -0.010 [-0.068, +0.036] |

Ablations of the fast-channel groups (initialisation = split seed for every network; 2,000 resamples of discharges):

| split | D1: ECEI-only over twin | D2: BES-only over twin | no-Mirnov over twin | full over twin | full over ECEI-only | full over BES-only | full over no-Mirnov |
|---|---|---|---|---|---|---|---|
| 42 | +0.010 [-0.015, +0.034] | +0.074 [+0.039, +0.111] | +0.048 [-0.023, +0.098] | +0.071 [+0.037, +0.107] | +0.062 [+0.025, +0.102] | -0.003 [-0.025, +0.017] | +0.024 [-0.007, +0.079] |
| 1 | +0.001 [-0.015, +0.015] | +0.026 [+0.007, +0.049] | +0.043 [+0.009, +0.073] | +0.022 [+0.001, +0.045] | +0.022 [+0.001, +0.043] | -0.004 [-0.024, +0.015] | -0.022 [-0.045, +0.008] |
| 7 | +0.084 [+0.066, +0.101] | +0.105 [+0.071, +0.134] | +0.135 [+0.103, +0.170] | +0.123 [+0.095, +0.151] | +0.043 [+0.022, +0.065] | +0.021 [-0.009, +0.055] | -0.014 [-0.033, +0.007] |
| 123 | +0.038 [-0.001, +0.073] | +0.063 [+0.036, +0.096] | +0.074 [+0.039, +0.114] | +0.060 [+0.025, +0.099] | +0.024 [-0.010, +0.069] | -0.003 [-0.029, +0.021] | -0.015 [-0.033, +0.003] |

| split | over the causal GP: ECEI-only | BES-only | no-Mirnov | skill of each initialisation over the causal GP: causal variant / twin | median absolute error (eV): hold / causal variant / twin / ECEI-only / BES-only / no-Mirnov | epochs run (best) |
|---|---|---|---|---|---|---|
| 42 | +0.020 [-0.031, +0.062] | +0.084 [+0.040, +0.125] | +0.059 [-0.018, +0.115] | i42: +0.081 / +0.011 | 43.8 / 41.4 / 45.5 / 43.0 / 42.2 / 42.0 | v2 i42 53 (47), v2h i42 49 (43), v2e 48 (42), v2b 80 (76), v2n 43 (37) |
| 1 | +0.074 [+0.045, +0.102] | +0.098 [+0.071, +0.125] | +0.114 [+0.069, +0.146] | i1: +0.094 / +0.074 | 48.9 / 55.7 / 49.9 / 51.2 / 50.0 / 50.3 | v2 i1 22 (16), v2h i1 46 (40), v2e 27 (21), v2b 37 (31), v2n 37 (31) |
| 7 | +0.105 [+0.076, +0.134] | +0.126 [+0.092, +0.160] | +0.155 [+0.115, +0.199] | i7: +0.144 / +0.024 | 48.8 / 45.6 / 48.1 / 45.7 / 45.8 / 45.8 | v2 i7 44 (38), v2h i7 11 (5), v2e 50 (44), v2b 69 (63), v2n 46 (40) |
| 123 | +0.080 [+0.029, +0.119] | +0.104 [+0.072, +0.135] | +0.115 [+0.071, +0.153] | i123: +0.101 / +0.044 | 42.7 / 42.6 / 43.0 / 41.6 / 41.8 / 41.0 | v2 i123 26 (20), v2h i123 58 (52), v2e 50 (44), v2b 31 (25), v2n 31 (25) |

## Arm P: pooled cross-validation (repeats r2 to r9, five folds each)

Not read yet: 0 of 80 runs are complete.
