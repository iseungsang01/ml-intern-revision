# B.53 — networks that read one component of the BES array, and training under a fixed budget

Registered: ces_prediction/experiments/PREREGISTRATION_B53.md. GPU runs (one initialisation = split seed); read-outs on a CPU. T_i, cut population, test rows of every earlier read-out. Skill of A over R = 1 - sum of squared errors of A / that of R on the same rows; 95 % interval from 2,000 resamples of the test discharges (one initialisation, so the spread between initialisations is not in it).

Splits read: 4 of 4.

- **C1** (D (differences) over the twin): interval above zero on 2 of 4 -> does not hold.
- **C2** (C (common mode) over the twin): interval above zero on 3 of 4 -> holds.
- **F1** (fixed-budget estimator over the fixed-budget twin): interval above zero on 4 of 4 -> holds.
- **F2** (fixed-budget estimator over the causal GP): interval above zero on 4 of 4 -> holds.

Paper (section 5): C1 does not hold, C2 holds: a trained network needs the common mode; F1 and F2 hold: the increment does not rest on early stopping.

## Arm C: components

| split | rows / discharges | C1: D over twin | C2: C over twin | CD over twin | BES network (B.39) over twin | CD over C | CD over D |
|---|---|---|---|---|---|---|---|
| 42 | 32,589 / 96 | +0.012 [-0.082, +0.074] | +0.064 [-0.004, +0.117] | +0.068 [-0.003, +0.119] | +0.074 [+0.040, +0.110] | +0.004 [-0.013, +0.022] | +0.057 [+0.021, +0.103] |
| 1 | 35,582 / 96 | +0.010 [-0.007, +0.027] | +0.038 [+0.021, +0.056] | +0.037 [+0.014, +0.062] | +0.026 [+0.006, +0.049] | -0.000 [-0.017, +0.015] | +0.028 [+0.011, +0.044] |
| 7 | 35,759 / 96 | +0.087 [+0.070, +0.104] | +0.100 [+0.078, +0.121] | +0.092 [+0.064, +0.119] | +0.105 [+0.069, +0.133] | -0.009 [-0.036, +0.016] | +0.006 [-0.015, +0.026] |
| 123 | 34,038 / 96 | +0.037 [+0.015, +0.065] | +0.049 [+0.021, +0.075] | +0.004 [-0.027, +0.039] | +0.063 [+0.035, +0.096] | -0.047 [-0.065, -0.030] | -0.034 [-0.062, -0.015] |

| split | C over BES network | D over BES network | CD over BES network | C over causal GP | D over causal GP | CD over causal GP |
|---|---|---|---|---|---|---|
| 42 | -0.011 [-0.056, +0.022] | -0.067 [-0.151, -0.008] | -0.006 [-0.059, +0.027] | +0.074 [+0.001, +0.127] | +0.023 [-0.078, +0.093] | +0.078 [+0.009, +0.130] |
| 1 | +0.012 [+0.000, +0.023] | -0.017 [-0.038, -0.002] | +0.012 [-0.009, +0.030] | +0.109 [+0.080, +0.137] | +0.083 [+0.051, +0.111] | +0.108 [+0.079, +0.138] |
| 7 | -0.005 [-0.029, +0.025] | -0.020 [-0.048, +0.015] | -0.014 [-0.048, +0.028] | +0.121 [+0.093, +0.149] | +0.109 [+0.080, +0.136] | +0.114 [+0.076, +0.151] |
| 123 | -0.015 [-0.034, -0.001] | -0.028 [-0.051, -0.007] | -0.063 [-0.090, -0.042] | +0.090 [+0.055, +0.122] | +0.079 [+0.045, +0.109] | +0.047 [+0.007, +0.082] |

## Arm F: fixed budget of 40 epochs

| split | F1: estimator over twin (fixed) | F2: estimator over causal GP (fixed) | increment of B.39 (same init) | fixed minus B.39 increment | twin fixed over twin B.39 | estimator fixed over estimator B.39 |
|---|---|---|---|---|---|---|
| 42 | +0.065 [+0.030, +0.097] | +0.075 [+0.025, +0.116] | +0.071 [+0.037, +0.106] | -0.006 [-0.013, -0.001] | -0.000 [-0.001, +0.000] | -0.007 [-0.014, -0.001] |
| 1 | +0.042 [+0.003, +0.075] | +0.113 [+0.073, +0.145] | +0.022 [+0.001, +0.045] | +0.020 [-0.007, +0.043] | +0.000 [+0.000, +0.000] | +0.021 [-0.007, +0.044] |
| 7 | +0.093 [+0.057, +0.134] | +0.144 [+0.106, +0.180] | +0.123 [+0.095, +0.150] | -0.030 [-0.047, -0.012] | +0.033 [+0.014, +0.051] | -0.000 [-0.001, +0.001] |
| 123 | +0.079 [+0.035, +0.126] | +0.116 [+0.061, +0.164] | +0.060 [+0.027, +0.100] | +0.019 [+0.003, +0.034] | -0.004 [-0.013, +0.001] | +0.016 [-0.005, +0.034] |

## Epochs, validation losses, median absolute errors

| split | network | epochs run (best) | best validation masked MSE | last-epoch validation masked MSE | median absolute error (eV) |
|---|---|---|---|---|---|
| 42 | C | 57 (51) | 0.2868 | 0.2870 | 42.3 |
| 42 | D | 73 (67) | 0.2894 | 0.2894 | 44.0 |
| 42 | CD | 57 (51) | 0.2841 | 0.2841 | 42.1 |
| 42 | estimator, 40 fixed | 40 (40) | 0.2843 | 0.2843 | 41.7 |
| 42 | twin, 40 fixed | 40 (40) | 0.2996 | 0.2996 | 45.7 |
| 42 | twin (B.39) | 49 (43) | 0.2995 | 0.2995 | 45.5 |
| 42 | estimator (B.39) | 53 (47) | 0.2840 | 0.2842 | 41.4 |
| 42 | BES network (B.39) | 80 (76) | 0.2842 | 0.2843 | 42.2 |
| 42 | hold / causal GP | - | - | - | 43.8 / 42.2 |
| 1 | C | 45 (39) | 0.2316 | 0.2378 | 48.4 |
| 1 | D | 22 (16) | 0.2368 | 0.2381 | 53.3 |
| 1 | CD | 36 (30) | 0.2304 | 0.2305 | 52.0 |
| 1 | estimator, 40 fixed | 40 (36) | 0.2308 | 0.2310 | 50.3 |
| 1 | twin, 40 fixed | 40 (40) | 0.2402 | 0.2402 | 49.9 |
| 1 | twin (B.39) | 46 (40) | 0.2402 | 0.2415 | 49.9 |
| 1 | estimator (B.39) | 22 (16) | 0.2327 | 0.2330 | 55.7 |
| 1 | BES network (B.39) | 37 (31) | 0.2328 | 0.2332 | 50.0 |
| 1 | hold / causal GP | - | - | - | 48.9 / 48.1 |
| 7 | C | 76 (70) | 0.9260 | 0.9261 | 45.9 |
| 7 | D | 33 (27) | 0.9336 | 0.9344 | 47.9 |
| 7 | CD | 31 (25) | 0.9318 | 0.9437 | 48.9 |
| 7 | estimator, 40 fixed | 40 (38) | 0.9292 | 0.9295 | 45.4 |
| 7 | twin, 40 fixed | 40 (40) | 0.9371 | 0.9371 | 46.5 |
| 7 | twin (B.39) | 11 (5) | 0.9456 | 0.9458 | 48.1 |
| 7 | estimator (B.39) | 44 (38) | 0.9292 | 0.9293 | 45.6 |
| 7 | BES network (B.39) | 69 (63) | 0.9211 | 0.9212 | 45.8 |
| 7 | hold / causal GP | - | - | - | 48.8 / 46.1 |
| 123 | C | 62 (56) | 0.2620 | 0.2621 | 41.7 |
| 123 | D | 25 (19) | 0.2670 | 0.2708 | 43.0 |
| 123 | CD | 16 (10) | 0.2709 | 0.2724 | 43.6 |
| 123 | estimator, 40 fixed | 40 (40) | 0.2555 | 0.2555 | 40.7 |
| 123 | twin, 40 fixed | 40 (39) | 0.2690 | 0.2693 | 42.2 |
| 123 | twin (B.39) | 58 (52) | 0.2688 | 0.2689 | 43.0 |
| 123 | estimator (B.39) | 26 (20) | 0.2594 | 0.2626 | 42.6 |
| 123 | BES network (B.39) | 31 (25) | 0.2627 | 0.2646 | 41.8 |
| 123 | hold / causal GP | - | - | - | 42.7 / 41.4 |

Note: under the fixed budget the scored weights are those of epoch 40 (CES_SEQ_FIXED_EPOCHS=1), whose validation loss is the last-epoch column; 'best' is the best epoch seen, kept only by the early-stopped runs.
