# B.57: a noise floor, the gap to the offline estimate, and the persistence of the fast-channel correction

Registration: `ces_prediction/experiments/PREREGISTRATION_B57.md` with Addendum 1. Runner: `ces_prediction/experiments/b57_yardsticks/run_b57.py`. CPU only; the 40 stored twins of the pooled cross-validation were run forward for their signed estimates; nothing was trained. T_i, cut population, the rows of the pooled cross-validation of B.39, re-partitions r2 to r9.

Sources: squared errors of the network, twin, hold, causal GP, offline GP and PCHIP are the stored ones of the pooled runs; the filters' (KF0, KFX) are from B.56 (`data/.b56/`). Signed estimates for the persistence: the network from B.55 (`data/.b55/run_r{r}_f{k}.npz`, `est_base`), the twin from this batch (`data/.b57/twin_r{r}_f{k}.npz`), the filters from B.56. Point estimates are the mean over the eight re-partitions; intervals: a re-partition drawn, then its discharges resampled, 2,000 draws. beta_h = cov(d_t, y_{t+h} - B_t) / var(d_t) (centred, pooled over the rows of a re-partition), and rho_h = beta_h / beta_0 with both computed on the rows whose frames t to t+h are consecutive, scored and 10 ms apart.

## Checks

- (i) twins run on the CPU reproduce their stored squared errors: pass; the smallest share of rows within 0.5 eV over the 40 twins is 0.9989 (required 0.99); median deviation 0.0204 eV.
- (ii) section 2 reproduced on point estimates (three significant figures): pass; floor_eV2 look 18414.7 batch 18415; mse_over_floor_net look 1.3312 batch 1.3312; mse_over_floor_gp look 1.4717 batch 1.4717; mse_over_floor_offline_gp look 1.2792 batch 1.2792; removable_net_over_gp look 0.2977 batch 0.29766; removable_net_over_twin look 0.2017 batch 0.20167.
- (iii) with the value of frame t+1 replaced by that of frame t, rho_1 = 1.000000000000, 1.000000000000, 1.000000000000, 1.000000000000, 1.000000000000, 1.000000000000: pass.

## Statements

| | Statement | Value [95 %] | Verdict | Expected | Point inside the expected range |
|---|---|---|---|---|---|
| N1 | every interval of MSE over the floor above 1 | network 1.331 [1.246, 1.427]; twin 1.415 [1.325, 1.519]; causal GP 1.472 [1.389, 1.568]; hold 2.123 [2.031, 2.227]; offline GP 1.279 [1.222, 1.342]; PCHIP 1.630 [1.587, 1.675]; KF0 1.469 [1.387, 1.562]; KFX 1.362 [1.291, 1.444] | holds | holds; network 1.25 to 1.45 | network yes |
| N2 | interval above 0.5 | 0.734 [0.529, 0.936] | holds | 0.60 to 0.85, interval reaching down to about 0.45; may fail | yes |
| N3 | interval above 0.5 | 0.810 [0.753, 0.866] | holds | rho_1 0.70 to 0.90, holds | yes |
| N4 | interval above 0.5 | 0.741 [0.670, 0.812] | holds | rho_1 0.70 to 0.95, holds | yes |
| N5 | interval inside [-0.15, +0.15] | +0.016 [-0.052, +0.088] | holds | -0.10 to +0.10, holds | yes |

N1 MSE of every estimator over the floor; N2 gap closure of the network, (MSE_GP - MSE_net) / (MSE_GP - MSE_offline GP); N3 rho_1 of the network minus its twin; N4 rho_1 of KFX minus KF0; N5 rho_3 - rho_1 of the network minus its twin.

## Reported

Floor (lower bound of the mean white variance, on 1,085,452 triple rows over the eight re-partitions): 18,415 [15,969, 21,226] eV^2, root mean square 135.696 [126.368, 145.693] eV (expected 126 to 145 eV).

| | All rows | Rows not after an absent frame |
|---|---|---|
| MSE over the floor, network | 1.331 [1.246, 1.427] | 1.313 [1.227, 1.416] |
| MSE over the floor, twin | 1.415 [1.325, 1.519] | 1.400 [1.310, 1.505] |
| MSE over the floor, causal GP | 1.472 [1.389, 1.568] | 1.455 [1.371, 1.556] |
| MSE over the floor, hold | 2.123 [2.031, 2.227] | 2.100 [2.011, 2.208] |
| MSE over the floor, offline GP | 1.279 [1.222, 1.342] | 1.267 [1.211, 1.332] |
| MSE over the floor, PCHIP | 1.630 [1.587, 1.675] | 1.623 [1.582, 1.674] |
| MSE over the floor, KF0 | 1.469 [1.387, 1.562] | 1.452 [1.370, 1.550] |
| MSE over the floor, KFX | 1.362 [1.291, 1.444] | 1.348 [1.275, 1.436] |
| removable-error skill, network over causal GP | +0.298 [+0.191, +0.408] | +0.312 [+0.205, +0.429] |
| removable-error skill, network over twin | +0.202 [+0.105, +0.303] | +0.218 [+0.114, +0.328] |
| removable-error skill, network over hold | +0.705 [+0.642, +0.767] | +0.716 [+0.650, +0.781] |
| removable-error skill, KFX over causal GP | +0.233 [+0.175, +0.288] | +0.234 [+0.173, +0.297] |
| removable-error skill, KFX over twin | +0.128 [+0.047, +0.195] | +0.128 [+0.051, +0.198] |
| removable-error skill, KFX over hold | +0.678 [+0.629, +0.725] | +0.684 [+0.632, +0.733] |
| gap closure, network | 0.734 [0.529, 0.936] | 0.727 [0.513, 0.934] |
| gap closure, twin | 0.354 [0.200, 0.509] | 0.345 [0.171, 0.512] |
| gap closure, KFX | 0.447 [0.328, 0.567] | 0.416 [0.281, 0.555] |
| gap closure, KF0 | 0.041 [-0.076, 0.151] | 0.023 [-0.106, 0.153] |
| beta0, network - twin | 0.693 [0.589, 0.790] | 0.692 [0.576, 0.789] |
| rho1, network - twin | 0.810 [0.753, 0.866] | 0.803 [0.741, 0.865] |
| rho3, network - twin | 0.827 [0.734, 0.912] | 0.818 [0.720, 0.914] |
| rho3_minus_rho1, network - twin | +0.016 [-0.052, +0.088] | +0.016 [-0.060, +0.093] |
| beta0, KFX - KF0 | 1.081 [0.985, 1.172] | 1.057 [0.955, 1.155] |
| rho1, KFX - KF0 | 0.741 [0.670, 0.812] | 0.715 [0.639, 0.794] |
| rho3, KFX - KF0 | 0.719 [0.631, 0.814] | 0.699 [0.598, 0.804] |
| rho3_minus_rho1, KFX - KF0 | -0.022 [-0.089, +0.040] | -0.016 [-0.099, +0.056] |
| beta0, twin - KF0 (reference) | 0.655 [0.573, 0.737] | 0.650 [0.568, 0.728] |
| rho1, twin - KF0 (reference) | 1.122 [1.038, 1.220] | 1.121 [1.033, 1.227] |
| rho3, twin - KF0 (reference) | 1.242 [1.094, 1.397] | 1.274 [1.116, 1.450] |
| rho3_minus_rho1, twin - KF0 (reference) | +0.120 [+0.005, +0.236] | +0.153 [+0.029, +0.286] |

Expected: floor 126 to 145 eV; MSE over the floor 1.25 to 1.45 for the network; removable-error skill over the causal GP +0.19 to +0.41; beta_0 0.6 to 0.9.

Rows: all r2 191,934, r3 192,092, r4 192,132, r5 192,032, r6 192,145, r7 192,056, r8 192,127, r9 192,127; chains h = 1 160,686, 160,622, 160,469, 160,455, 160,183, 160,048, 160,067, 160,300; h = 3 117,988, 117,034, 115,968, 116,123, 114,542, 114,317, 114,323, 115,390.

Rows not after an absent frame: B.56's definition (the ten grid rows starting at the first row after a step of 15 ms to 0.5 s); for the persistence the row t must be outside them.
