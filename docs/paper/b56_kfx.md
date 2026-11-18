# B.56: a robust state-space filter with the beam-emission array as input

Registration: `ces_prediction/experiments/PREREGISTRATION_B56.md` with Addendum 1. Runner: `ces_prediction/experiments/b56_kfx/run_b56.py`. CPU only; no network trained or run. T_i, cut population.

Frozen specification (byte-identical copies of the development scripts):

- `kfx_model.py` SHA-256 `210dd1435762ef98f8c89b22813a18474772b2e433b275e7bb22081de44acca8` (only its definition part is executed)
- `kfx_components.py` SHA-256 `716b55c5c7632a4d458791028e9575e69b500bfd08de2d1d99070f610a6f89fb` (the gain fit with a fixed clip, reimplemented in `fit_unit`)

Declared forking paths (repeated from the registration and its addendum): (1) the agent saw the test error of a filter with an unclipped drive once on r2 (worse than KF0, -0.07 [-0.29, +0.03]) and then added the clip; (2) a joint fit of all five parameters was tried on r2 and abandoned because it degenerated in three of five folds. r2 is the development set and is shown beside the confirmation set (r3 to r9), not pooled into it. The look chose the clip per fold from {0.02, 0.05, 0.1, 0.2, none} (every r2 fold chose 0.2); the batch fixes it at 0.2.

Fit in two stages: KF0 (q, r, k; Nelder-Mead from log 0.002, log 0.01, log 2, at most 250 iterations), then the gains of Kc, Kg (at most 200 iterations, as in `kfx_components.py`) and KFX (at most 300) from zero with q, r, k held. Training discharges: every discharge not in the test fold (on the splits: not in the split's test set).

## Checks

- (a) r2 point estimates of section 2: pass
  - kf0_over_gp: look +0.002, batch +0.0024
  - kfx_over_kf0: look +0.049, batch +0.0492
  - net_over_kfx: look +0.040, batch +0.0396
  - twin_over_kf0: look +0.044, batch +0.0439
  - kg_over_kf0: look +0.032, batch +0.0316
  - kc_over_kf0: look +0.042, batch +0.0418
  - kfx_over_kf0_quiet: look +0.045, batch +0.0452
  - median kfx: look 41.8 eV, batch 41.85 eV
  - median kf0: look 43.3 eV, batch 43.29 eV
  - median G: look 44.1 eV, batch 44.10 eV
  - median H: look 45.3 eV, batch 45.34 eV
  - median A: look 47.1 eV, batch 47.07 eV
- (b) identical rows: pass; 1694675 stored rows over 45 units, all located, max |grid value - stored value| 0.0002 eV; the located grid rows agree with B.54's independent location of the same rows on every pooled fold (B.54's grid row minus this batch's is 1 on every row: B.54 counts the first file row, this batch does not).
- Deviation in the location of the scored rows (declared): the frozen `map_rows` of `kfx_model.py` matches the stored values with a tolerance of 0.01 eV and takes the first grid value within it; where two values of a discharge differ by less than 0.01 eV it put a row one or more grid rows early (45 rows of 1,694,675, all on the pooled folds, found by the cross-check against B.54). The batch locates rows with `map_rows_exact` (tolerance 0.0003 eV, closest value): the filter and its fit are unchanged (the fit does not use the location); only which grid row's output is scored changes. Check (a) gives the same numbers with either location.
- (c) causality: pass; 30 perturbations of the CES values from row t on, max change of the outputs up to row t 0.0 eV; later rows moved in 30.

## Statements (re-partitions r3 to r9, two-level bootstrap, 20,000 draws; medians 2,000)

Point estimates are the mean over the seven re-partitions of the skill of each (as in B.39 arm P); the median ratio is the mean over re-partitions of the ratio of the row medians, its interval from weighted medians under the same two-level resampling. Rows not flagged by the value rules: the `plain` column of B.54's row table of the same fold. Thirds: the discharges of a re-partition split by their median T_i (`data/.b54/files.npz`, ties by shot, as `run_b44.split_thirds`). Rows after an absent frame: the ten grid rows starting at the first row after a step of 15 ms to 0.5 s.

| | Statement | Value [95 %] | Verdict | Expected | Inside the expected range |
|---|---|---|---|---|---|
| KX1 | interval above 0 | +0.049 [+0.040, +0.059] | holds | +0.03 to +0.06, holds | yes; verdict as expected |
| KX2 | interval above 0 | +0.032 [+0.025, +0.039] | holds | +0.02 to +0.04, holds | yes; verdict as expected |
| KX3 | interval above 0 | +0.036 [+0.012, +0.058] | holds | +0.02 to +0.06, holds | yes; verdict as expected |
| KX4 | 90 % interval inside [-0.02, +0.02] | +0.005 [-0.007, +0.017] | holds | -0.01 to +0.01, holds | yes; verdict as expected |
| KX5 | interval below 1 | 0.921 [0.906, 0.940] | holds | ratio 0.90 to 0.95, holds | yes; verdict as expected |
| KX6 | interval above 0 | +0.045 [+0.036, +0.055] | holds | +0.03 to +0.055, holds | yes; verdict as expected |

KX1 KFX over KF0; KX2 Kg (radial difference only) over KF0; KX3 the network over KFX on the same rows; KX4 KF0 over the causal GP; KX5 median absolute error of KFX over that of hold; KX6 KFX over KF0 in rows not within ten rows after an absent frame.

## Reported (r3 to r9)

| Skill | All rows | Rows not after an absent frame | Rows not flagged by the value rules |
|---|---|---|---|
| twin over KF0 | +0.037 [+0.021, +0.053] | +0.037 [+0.021, +0.053] | +0.052 [+0.026, +0.074] |
| KFX over twin | +0.013 [-0.007, +0.032] | +0.009 [-0.011, +0.029] | +0.054 [+0.028, +0.082] |
| Kc over KF0 | +0.042 [+0.033, +0.051] | +0.039 [+0.030, +0.049] | +0.095 [+0.084, +0.106] |
| KFX over Kc | +0.008 [+0.004, +0.011] | +0.006 [+0.003, +0.010] | +0.009 [+0.004, +0.015] |
| KFX over Kg | +0.018 [+0.012, +0.024] | +0.016 [+0.010, +0.022] | +0.044 [+0.037, +0.050] |
| KFX over causal GP | +0.055 [+0.039, +0.070] | +0.048 [+0.032, +0.064] | +0.057 [+0.038, +0.075] |
| KFX over offline GP | -0.076 [-0.097, -0.056] | -0.076 [-0.098, -0.055] | -0.079 [-0.105, -0.054] |
| network over causal GP | +0.089 [+0.063, +0.112] | +0.084 [+0.059, +0.107] | +0.095 [+0.065, +0.123] |
| network over twin | +0.049 [+0.023, +0.073] | +0.046 [+0.020, +0.071] | +0.093 [+0.063, +0.126] |
| KFX over KF0 | +0.049 [+0.040, +0.059] | +0.045 [+0.036, +0.055] | +0.104 [+0.092, +0.116] |
| Kg over KF0 | +0.032 [+0.025, +0.039] | +0.030 [+0.023, +0.037] | +0.063 [+0.051, +0.074] |
| network over KFX | +0.036 [+0.012, +0.058] | +0.037 [+0.013, +0.059] | +0.041 [+0.014, +0.066] |
| KF0 over causal GP | +0.005 [-0.009, +0.020] | +0.003 [-0.012, +0.018] | -0.052 [-0.070, -0.035] |

Clip sensitivity (refits, reported only): KFX over KF0 at clip 0.1 +0.048 [+0.040, +0.056], at 0.2 +0.049 [+0.040, +0.059], at 0.4 +0.048 [+0.039, +0.057].

Median absolute error (eV, mean over re-partitions): A 45.8, Tw 46.3, G 44.1, H 45.4, O 37.5, kf0 43.3, kc 41.1, kg 43.5, kfx 41.8 (A network, Tw twin, G causal GP, H hold, O offline GP).

### Thirds of the discharges by median T_i (within a re-partition)

| Third | KFX over KF0 | Kg over KF0 | network over KFX | twin over KF0 | KFX over causal GP | median KFX / KF0 / hold (eV) |
|---|---|---|---|---|---|---|
| coldest | +0.001 [-0.007, +0.009] | +0.000 [-0.004, +0.005] | +0.046 [+0.015, +0.072] | +0.041 [+0.026, +0.055] | +0.022 [-0.005, +0.047] | 38.0 / 38.2 / 40.7 |
| middle | +0.042 [+0.031, +0.054] | +0.016 [+0.008, +0.024] | +0.005 [-0.036, +0.039] | +0.006 [-0.022, +0.032] | +0.085 [+0.060, +0.110] | 30.7 / 30.9 / 32.8 |
| hottest | +0.101 [+0.085, +0.117] | +0.073 [+0.059, +0.088] | +0.049 [+0.001, +0.086] | +0.055 [+0.027, +0.081] | +0.064 [+0.037, +0.090] | 62.1 / 67.7 / 71.6 |

## Development set r2 (not pooled into the confirmation; intervals: its discharges resampled, 20,000 draws)

twin over KF0 +0.044 [+0.032, +0.055], KFX over twin +0.006 [-0.009, +0.021], Kc over KF0 +0.042 [+0.033, +0.051], KFX over Kc +0.008 [+0.004, +0.011], KFX over Kg +0.018 [+0.012, +0.024], KFX over causal GP +0.051 [+0.037, +0.066], KFX over offline GP -0.077 [-0.097, -0.058], network over causal GP +0.089 [+0.063, +0.113], network over twin +0.045 [+0.021, +0.066], KFX over KF0 +0.049 [+0.040, +0.058], Kg over KF0 +0.032 [+0.025, +0.039], network over KFX +0.040 [+0.015, +0.060], KF0 over causal GP +0.002 [-0.012, +0.016].

Median ratio KFX / hold 0.923 [0.907, 0.940]; medians: A 47.1, Tw 44.7, G 44.1, H 45.3, O 37.5, kf0 43.3, kc 41.1, kg 43.5, kfx 41.8 eV.

## Four file-level splits (secondary; statements read as at least three of four)

KX3 and the twin use the three initialisations of B.39 with the registered interval of the splits (an initialisation drawn, then the test discharges resampled); the filters carry no initialisation (discharges resampled, 20,000 draws).

| Split | KX1 | KX2 | KX3 | KX4 (90 %) | KX5 | KX6 | twin over KF0 | KFX over causal GP |
|---|---|---|---|---|---|---|---|---|
| 42 | +0.063 [+0.032, +0.099] | +0.042 [+0.014, +0.069] | +0.021 [-0.054, +0.065] | -0.018 [-0.059, +0.019] | 0.914 [0.877, 0.952] | +0.053 [+0.022, +0.089] | +0.013 [-0.047, +0.058] | +0.047 [-0.012, +0.094] |
| 1 | +0.046 [+0.027, +0.066] | +0.031 [+0.017, +0.048] | +0.035 [+0.001, +0.064] | +0.023 [-0.005, +0.049] | 0.936 [0.887, 0.982] | +0.042 [+0.025, +0.062] | +0.039 [+0.006, +0.070] | +0.068 [+0.036, +0.098] |
| 7 | +0.062 [+0.042, +0.084] | +0.033 [+0.019, +0.049] | +0.056 [+0.021, +0.088] | +0.031 [+0.010, +0.051] | 0.909 [0.868, 0.954] | +0.056 [+0.037, +0.078] | +0.027 [-0.022, +0.062] | +0.091 [+0.060, +0.120] |
| 123 | +0.046 [+0.024, +0.070] | +0.029 [+0.015, +0.044] | +0.038 [-0.082, +0.095] | +0.018 [-0.005, +0.041] | 0.923 [0.883, 0.958] | +0.040 [+0.018, +0.064] | +0.022 [-0.067, +0.055] | +0.063 [+0.029, +0.098] |

Four-split verdicts: KX1 4/4 (holds); KX2 4/4 (holds); KX3 2/4 (does not hold); KX4 0/4 (does not hold); KX5 4/4 (holds); KX6 4/4 (holds). Expected: KX1 four of four; KX3 two to four of four.

Network over KFX with the initialisation equal to the split seed only: 42 +0.036 [+0.007, +0.068], 1 +0.029 [-0.001, +0.055], 7 +0.059 [+0.029, +0.085], 123 +0.041 [+0.008, +0.067].

## Split by time (B.29 campaign split; reported)

95 test discharges, 20062 rows. KFX over KF0 +0.027 [+0.016, +0.038]; Kg over KF0 +0.018 [+0.009, +0.028]; KF0 over causal GP +0.060 [+0.022, +0.091]; KFX over causal GP +0.085 [+0.044, +0.118]; the causal variant of B.29 (epoch cap 30, four initialisations, one drawn per resample) over KFX +0.024 [-0.004, +0.052]. No twin exists on this split. Medians: kfx 51.5, kf0 52.5, G 62.2, H 56.3, A_mean_over_inits 61.2 eV.

## Fitted parameters

| Unit | log q | log r | log k | KFX b_c | KFX b_g | Kc b_c | Kg b_g | KFX b at clip 0.1 | at 0.4 |
|---|---|---|---|---|---|---|---|---|---|
| r2_f0 | -4.267 | -2.967 | 1.060 | 0.476 | 0.466 | 0.647 | 0.760 | 0.532, 0.583 | 0.427, 0.409 |
| r2_f1 | -4.236 | -2.974 | 1.050 | 0.475 | 0.477 | 0.649 | 0.771 | 0.530, 0.604 | 0.418, 0.432 |
| r2_f2 | -4.346 | -3.092 | 1.059 | 0.491 | 0.434 | 0.648 | 0.733 | 0.547, 0.563 | 0.428, 0.381 |
| r2_f3 | -4.196 | -2.884 | 1.015 | 0.509 | 0.441 | 0.669 | 0.749 | 0.567, 0.560 | 0.442, 0.391 |
| r2_f4 | -4.312 | -3.082 | 1.038 | 0.556 | 0.449 | 0.738 | 0.772 | 0.670, 0.549 | 0.475, 0.399 |
| r3_f0 | -4.291 | -3.050 | 1.045 | 0.548 | 0.448 | 0.734 | 0.773 | 0.650, 0.554 | 0.470, 0.415 |
| r3_f1 | -4.369 | -3.048 | 1.089 | 0.506 | 0.452 | 0.666 | 0.762 | 0.558, 0.573 | 0.452, 0.393 |
| r3_f2 | -4.258 | -2.996 | 1.060 | 0.475 | 0.453 | 0.638 | 0.738 | 0.528, 0.585 | 0.419, 0.392 |
| r3_f3 | -4.198 | -2.916 | 0.953 | 0.504 | 0.457 | 0.667 | 0.763 | 0.571, 0.574 | 0.443, 0.397 |
| r3_f4 | -4.223 | -2.973 | 1.069 | 0.473 | 0.457 | 0.646 | 0.743 | 0.532, 0.574 | 0.406, 0.408 |
| r4_f0 | -4.241 | -2.993 | 1.057 | 0.495 | 0.436 | 0.653 | 0.736 | 0.551, 0.561 | 0.432, 0.384 |
| r4_f1 | -4.209 | -2.950 | 1.066 | 0.483 | 0.449 | 0.645 | 0.744 | 0.538, 0.583 | 0.425, 0.389 |
| r4_f2 | -4.326 | -3.033 | 1.067 | 0.497 | 0.460 | 0.662 | 0.772 | 0.558, 0.567 | 0.435, 0.423 |
| r4_f3 | -4.212 | -2.905 | 1.014 | 0.528 | 0.478 | 0.725 | 0.786 | 0.632, 0.590 | 0.452, 0.421 |
| r4_f4 | -4.297 | -3.025 | 0.998 | 0.492 | 0.450 | 0.654 | 0.742 | 0.549, 0.565 | 0.438, 0.396 |
| r5_f0 | -4.239 | -3.003 | 1.058 | 0.496 | 0.476 | 0.667 | 0.786 | 0.551, 0.609 | 0.443, 0.426 |
| r5_f1 | -4.294 | -3.043 | 1.041 | 0.503 | 0.453 | 0.668 | 0.762 | 0.568, 0.564 | 0.441, 0.400 |
| r5_f2 | -4.203 | -2.942 | 1.055 | 0.479 | 0.457 | 0.649 | 0.748 | 0.534, 0.585 | 0.411, 0.406 |
| r5_f3 | -4.375 | -3.113 | 1.061 | 0.528 | 0.431 | 0.710 | 0.739 | 0.634, 0.537 | 0.455, 0.382 |
| r5_f4 | -4.288 | -2.962 | 1.011 | 0.492 | 0.447 | 0.649 | 0.743 | 0.547, 0.563 | 0.434, 0.392 |
| r6_f0 | -4.327 | -2.998 | 1.013 | 0.512 | 0.423 | 0.667 | 0.736 | 0.574, 0.550 | 0.447, 0.379 |
| r6_f1 | -4.197 | -2.861 | 1.022 | 0.435 | 0.502 | 0.622 | 0.764 | 0.487, 0.626 | 0.384, 0.433 |
| r6_f2 | -4.253 | -2.984 | 1.077 | 0.487 | 0.473 | 0.651 | 0.768 | 0.544, 0.594 | 0.436, 0.409 |
| r6_f3 | -4.302 | -3.094 | 1.030 | 0.577 | 0.417 | 0.757 | 0.766 | 0.694, 0.516 | 0.482, 0.389 |
| r6_f4 | -4.234 | -3.016 | 1.059 | 0.498 | 0.447 | 0.657 | 0.746 | 0.548, 0.570 | 0.443, 0.392 |
| r7_f0 | -4.277 | -2.974 | 1.062 | 0.505 | 0.442 | 0.661 | 0.750 | 0.556, 0.568 | 0.443, 0.390 |
| r7_f1 | -4.397 | -3.095 | 1.076 | 0.502 | 0.443 | 0.657 | 0.748 | 0.566, 0.563 | 0.436, 0.387 |
| r7_f2 | -4.239 | -2.970 | 1.059 | 0.484 | 0.466 | 0.647 | 0.755 | 0.547, 0.590 | 0.426, 0.407 |
| r7_f3 | -4.341 | -3.066 | 1.059 | 0.517 | 0.465 | 0.723 | 0.767 | 0.621, 0.566 | 0.442, 0.416 |
| r7_f4 | -4.215 | -2.997 | 1.049 | 0.484 | 0.453 | 0.654 | 0.761 | 0.538, 0.573 | 0.434, 0.410 |
| r8_f0 | -4.263 | -2.997 | 1.050 | 0.439 | 0.504 | 0.625 | 0.770 | 0.498, 0.626 | 0.384, 0.450 |
| r8_f1 | -4.282 | -3.088 | 1.035 | 0.566 | 0.423 | 0.744 | 0.758 | 0.665, 0.524 | 0.503, 0.376 |
| r8_f2 | -4.253 | -2.922 | 1.029 | 0.513 | 0.462 | 0.674 | 0.772 | 0.570, 0.596 | 0.439, 0.408 |
| r8_f3 | -4.270 | -2.987 | 1.054 | 0.485 | 0.454 | 0.651 | 0.754 | 0.544, 0.569 | 0.431, 0.399 |
| r8_f4 | -4.256 | -2.965 | 1.045 | 0.511 | 0.416 | 0.663 | 0.728 | 0.565, 0.543 | 0.443, 0.371 |
| r9_f0 | -4.292 | -3.049 | 1.045 | 0.512 | 0.439 | 0.690 | 0.734 | 0.606, 0.557 | 0.436, 0.385 |
| r9_f1 | -4.246 | -2.972 | 1.062 | 0.517 | 0.463 | 0.687 | 0.787 | 0.574, 0.593 | 0.453, 0.417 |
| r9_f2 | -4.213 | -2.884 | 1.033 | 0.487 | 0.478 | 0.658 | 0.773 | 0.548, 0.592 | 0.434, 0.415 |
| r9_f3 | -4.307 | -3.062 | 1.039 | 0.486 | 0.463 | 0.654 | 0.758 | 0.545, 0.581 | 0.423, 0.413 |
| r9_f4 | -4.311 | -3.062 | 1.031 | 0.497 | 0.422 | 0.650 | 0.726 | 0.561, 0.535 | 0.436, 0.374 |
| s42 | -4.325 | -3.018 | 1.069 | 0.501 | 0.450 | 0.665 | 0.763 | 0.562, 0.567 | 0.435, 0.410 |
| s1 | -4.223 | -2.990 | 1.051 | 0.484 | 0.462 | 0.647 | 0.751 | 0.540, 0.582 | 0.431, 0.399 |
| s7 | -4.310 | -3.077 | 1.036 | 0.463 | 0.480 | 0.639 | 0.760 | 0.531, 0.593 | 0.405, 0.429 |
| s123 | -4.186 | -2.923 | 1.046 | 0.482 | 0.462 | 0.647 | 0.751 | 0.542, 0.579 | 0.422, 0.408 |
| camp | -4.235 | -3.003 | 1.056 | 0.491 | 0.439 | 0.655 | 0.739 | 0.549, 0.562 | 0.433, 0.388 |

Gains of KFX on the confirmation folds: b_c 0.435 to 0.577 (expected 0.4 to 0.6), b_g 0.416 to 0.504 (expected 0.4 to 0.5).
