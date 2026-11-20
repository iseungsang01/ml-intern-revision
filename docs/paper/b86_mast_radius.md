# B.86 — MAST from the core to the edge: the same estimator at every radius

Registered: `ces_prediction/experiments/PREREGISTRATION_B86.md`; implementation choices in `ces_prediction/experiments/PREREGISTRATION_B86_ADDENDUM1.md`, written before the row build, the checks and any readout. CPU only; public FAIR-MAST files on disk; T_i of the charge-exchange chords in five bands by the distance d of the chord inside the edge of the Thomson density profile. Exploratory on used data (registration section 1). No statement here is transferred to the KSTAR core.

Conventions. Skill of A over R = 1 - SSE(A) / SSE(R) on identical rows. Each value: point estimate, then the 95 % interval with sessions (calendar days) resampled (S), 20,000 draws; the interval with shots resampled is in the statement table and in the JSON. Verdicts on the session interval. M1 is re-partition 1 (one split, five folds of whole shots); M3 pools the scored fifths 2 to 5. Each band is fitted and scored on its rows complete under D5 and D100. RMSE, MAE and median absolute error in eV. Target L5 (median of the good values of the next five frames) unless F is named. Checks: `docs/paper/b86_checks.md` (all four passed before this readout).

## Counts per band (L5 common set)

| band | d (cm) | candidate rows | rows | shots | sessions | median rho | median T_i (eV) | median d (cm) | rows with rho < 0 | statements |
|---|---|---|---|---|---|---|---|---|---|---|
| E | 1.5 to 5 | 372,094 | 145,048 | 2,517 | 252 | 0.899 | 239 | 3.3 | 0.000 | yes |
| B2 | 5 to 10 | 548,129 | 239,297 | 2,633 | 254 | 0.822 | 319 | 7.5 | 0.000 | yes |
| B3 | 10 to 20 | 1,003,401 | 485,836 | 2,803 | 258 | 0.677 | 559 | 14.9 | 0.000 | yes |
| B4 | 20 to 35 | 1,312,041 | 678,409 | 2,697 | 255 | 0.423 | 976 | 27.5 | 0.010 | yes |
| B5 | 35 to 60 | 1,832,147 | 1,074,935 | 2,789 | 259 | 0.034 | 989 | 46.5 | 0.440 | yes |

Innermost band with statements: B5.

## Statements (target L5; judged on the lower end of the session interval)

| id | statement | value | session interval | shot interval | verdict | registered expectation | expectation | note |
|---|---|---|---|---|---|---|---|---|
| R1.E | M1, D100: TX over T0 lies above zero in band E | +0.296 | [+0.246, +0.354] | [+0.261, +0.333] | **holds** | holds in every band | met | TX over T0, M1, D100, target L5, 145,048 rows |
| R2.E | M1, D5: TX over T0 lies above zero in band E | +0.104 | [+0.074, +0.135] | [+0.078, +0.130] | **holds** | holds in E (B.76); uncertain inside | met | TX over T0, M1, D5, target L5, 145,048 rows |
| R3.E | M3, D100: TX over T0 lies above zero in band E | +0.209 | [+0.095, +0.305] | [+0.135, +0.278] | **holds** | holds in E (B.76) | met | TX over T0, M3, D100, target L5, 126,806 rows |
| R4.E | M1, D5: T0 over G lies above zero in band E (the learned history-only estimate against the causal GP) | +0.180 | [+0.119, +0.244] | [+0.125, +0.233] | **holds** | holds in every band | met | T0 over G, M1, D5, target L5, 145,048 rows |
| R1.B2 | M1, D100: TX over T0 lies above zero in band B2 | +0.518 | [+0.441, +0.600] | [+0.470, +0.569] | **holds** | holds in every band | met | TX over T0, M1, D100, target L5, 239,297 rows |
| R2.B2 | M1, D5: TX over T0 lies above zero in band B2 | +0.150 | [+0.086, +0.205] | [+0.113, +0.186] | **holds** | uncertain inside (holds in E) | n/a (uncertain) | TX over T0, M1, D5, target L5, 239,297 rows |
| R3.B2 | M3, D100: TX over T0 lies above zero in band B2 | +0.507 | [+0.408, +0.594] | [+0.449, +0.561] | **holds** | uncertain, leaning to holds, inside | n/a (uncertain) | TX over T0, M3, D100, target L5, 219,162 rows |
| R4.B2 | M1, D5: T0 over G lies above zero in band B2 (the learned history-only estimate against the causal GP) | +0.225 | [+0.147, +0.308] | [+0.152, +0.294] | **holds** | holds in every band | met | T0 over G, M1, D5, target L5, 239,297 rows |
| R1.B3 | M1, D100: TX over T0 lies above zero in band B3 | +0.704 | [+0.671, +0.735] | [+0.681, +0.724] | **holds** | holds in every band | met | TX over T0, M1, D100, target L5, 485,836 rows |
| R2.B3 | M1, D5: TX over T0 lies above zero in band B3 | +0.196 | [+0.159, +0.234] | [+0.172, +0.220] | **holds** | uncertain inside (holds in E) | n/a (uncertain) | TX over T0, M1, D5, target L5, 485,836 rows |
| R3.B3 | M3, D100: TX over T0 lies above zero in band B3 | +0.658 | [+0.585, +0.716] | [+0.625, +0.689] | **holds** | uncertain, leaning to holds, inside | n/a (uncertain) | TX over T0, M3, D100, target L5, 460,942 rows |
| R4.B3 | M1, D5: T0 over G lies above zero in band B3 (the learned history-only estimate against the causal GP) | +0.296 | [+0.257, +0.334] | [+0.272, +0.320] | **holds** | holds in every band | met | T0 over G, M1, D5, target L5, 485,836 rows |
| R1.B4 | M1, D100: TX over T0 lies above zero in band B4 | +0.644 | [+0.618, +0.669] | [+0.628, +0.660] | **holds** | holds in every band | met | TX over T0, M1, D100, target L5, 678,409 rows |
| R2.B4 | M1, D5: TX over T0 lies above zero in band B4 | +0.198 | [+0.182, +0.213] | [+0.184, +0.211] | **holds** | uncertain inside (holds in E) | n/a (uncertain) | TX over T0, M1, D5, target L5, 678,409 rows |
| R3.B4 | M3, D100: TX over T0 lies above zero in band B4 | +0.608 | [+0.579, +0.634] | [+0.591, +0.624] | **holds** | uncertain, leaning to holds, inside | n/a (uncertain) | TX over T0, M3, D100, target L5, 667,078 rows |
| R4.B4 | M1, D5: T0 over G lies above zero in band B4 (the learned history-only estimate against the causal GP) | +0.312 | [+0.288, +0.336] | [+0.294, +0.329] | **holds** | holds in every band | met | T0 over G, M1, D5, target L5, 678,409 rows |
| R1.B5 | M1, D100: TX over T0 lies above zero in band B5 | +0.600 | [+0.573, +0.625] | [+0.584, +0.616] | **holds** | holds in every band | met | TX over T0, M1, D100, target L5, 1,074,935 rows |
| R2.B5 | M1, D5: TX over T0 lies above zero in band B5 | +0.187 | [+0.169, +0.205] | [+0.172, +0.202] | **holds** | uncertain inside (holds in E) | n/a (uncertain) | TX over T0, M1, D5, target L5, 1,074,935 rows |
| R3.B5 | M3, D100: TX over T0 lies above zero in band B5 | +0.552 | [+0.524, +0.579] | [+0.535, +0.569] | **holds** | uncertain, leaning to holds, inside | n/a (uncertain) | TX over T0, M3, D100, target L5, 1,053,840 rows |
| R4.B5 | M1, D5: T0 over G lies above zero in band B5 (the learned history-only estimate against the causal GP) | +0.264 | [+0.241, +0.286] | [+0.248, +0.280] | **holds** | holds in every band | met | T0 over G, M1, D5, target L5, 1,074,935 rows |
| R5 | M1: the cost of staleness of T0 at D100 (MSE at D100 over MSE at D5, same rows) is larger in band E than in the innermost band with statements | -2.272 | [-2.622, -1.927] | [-2.507, -2.040] | **does not hold** | holds (the edge changes faster) | not met | E minus B5; E 1.678, B5 3.951 |
| R6 | M1, D100: the Thomson increment (TX over T0) in band E minus that in the innermost band with statements lies above zero | -0.304 | [-0.357, -0.245] | [-0.341, -0.265] | **does not hold** | uncertain | n/a (uncertain) | E minus B5 |
| R7 | M1, D5: the share of estimates of TX within 10 % of the level is larger in the innermost band with statements than in band E | +0.133 | [+0.116, +0.149] | [+0.123, +0.142] | **holds** | holds (the series is smoother inside) | met | B5 minus E; B5 0.673, E 0.541 |

## MAST from the core to the edge (summary per band)

Error with the previous frame visible (M1, D5): RMSE of TX and its share within 10 % of the level; cost of staleness of T0 at 100 ms (M1); the Thomson increment (TX over T0) under the two rules and the two designs.

| band | RMSE TX, D5 | TX within 10 %, D5 | cost of staleness T0, D100 | TX over T0, M1 D5 | TX over T0, M1 D100 | TX over T0, M3 D5 | TX over T0, M3 D100 |
|---|---|---|---|---|---|---|---|
| E | 148.3 | 0.541 S [0.518, 0.564] | 1.678 S [1.552, 1.832] | +0.104 S [+0.075, +0.135] | +0.296 S [+0.245, +0.354] | +0.002 S [-0.054, +0.052] | +0.209 S [+0.096, +0.306] |
| B2 | 127.4 | 0.589 S [0.569, 0.609] | 2.338 S [2.048, 2.737] | +0.150 S [+0.086, +0.205] | +0.518 S [+0.440, +0.600] | +0.139 S [+0.095, +0.191] | +0.507 S [+0.409, +0.596] |
| B3 | 129.1 | 0.612 S [0.592, 0.633] | 3.945 S [3.545, 4.438] | +0.196 S [+0.160, +0.235] | +0.704 S [+0.671, +0.734] | +0.172 S [+0.130, +0.220] | +0.658 S [+0.586, +0.716] |
| B4 | 162.6 | 0.680 S [0.659, 0.700] | 4.265 S [3.890, 4.680] | +0.198 S [+0.182, +0.213] | +0.644 S [+0.618, +0.669] | +0.179 S [+0.151, +0.205] | +0.608 S [+0.579, +0.634] |
| B5 | 162.7 | 0.673 S [0.652, 0.694] | 3.951 S [3.625, 4.293] | +0.187 S [+0.168, +0.205] | +0.600 S [+0.574, +0.625] | +0.162 S [+0.140, +0.183] | +0.552 S [+0.524, +0.579] |

## Per band, design and rule

H hold of the newest visible value; G causal GP of the last up to 16 visible values; T0 G + boosted correction on the history scalars and rho; TX with the nine Thomson scalars added; O the frame's own reading (against L5).

### Band E, M1 random (re-partition 1, five folds of whole shots) (145,048 rows / 2,517 shots / 252 sessions)

| rule | estimator | RMSE | MAE | median abs. error | median rel. error | within 10 % | within 20 % |
|---|---|---|---|---|---|---|---|
| D5 | H | 200.5 | 63.8 | 26.1 | 0.105 | 0.481 S [0.456, 0.506] | 0.746 S [0.727, 0.765] |
| D5 | G | 172.9 | 64.0 | 28.5 | 0.113 | 0.455 S [0.429, 0.480] | 0.723 S [0.703, 0.743] |
| D5 | T0 | 156.7 | 55.2 | 24.3 | 0.097 | 0.511 S [0.487, 0.536] | 0.758 S [0.739, 0.776] |
| D5 | TX | 148.3 | 49.8 | 22.5 | 0.090 | 0.541 S [0.518, 0.564] | 0.787 S [0.770, 0.803] |
| D5 | O | 181.9 | 53.3 | 21.9 | 0.088 | 0.550 S [0.527, 0.573] | 0.803 S [0.788, 0.819] |
| D100 | H | 308.6 | 152.5 | 83.8 | 0.353 | 0.150 S [0.133, 0.168] | 0.291 S [0.264, 0.319] |
| D100 | G | 296.3 | 161.6 | 92.1 | 0.394 | 0.129 S [0.114, 0.144] | 0.255 S [0.230, 0.280] |
| D100 | T0 | 203.0 | 100.6 | 61.2 | 0.234 | 0.242 S [0.221, 0.264] | 0.440 S [0.412, 0.467] |
| D100 | TX | 170.2 | 65.4 | 34.8 | 0.136 | 0.389 S [0.371, 0.406] | 0.656 S [0.635, 0.676] |
| D100 | O | 181.9 | 53.3 | 21.9 | 0.088 | 0.550 S [0.527, 0.573] | 0.803 S [0.788, 0.819] |

| skill | D5 | D100 |
|---|---|---|
| TX over T0 | +0.104 S [+0.075, +0.135] | +0.296 S [+0.245, +0.354] |
| T0 over G | +0.180 S [+0.119, +0.244] | +0.531 S [+0.436, +0.622] |
| T0 over H | +0.390 S [+0.325, +0.454] | +0.567 S [+0.486, +0.634] |
| TX over G | +0.265 S [+0.212, +0.322] | +0.670 S [+0.584, +0.748] |
| TX over H | +0.453 S [+0.399, +0.509] | +0.696 S [+0.631, +0.751] |
| G over H | +0.256 S [+0.212, +0.299] | +0.078 S [-0.059, +0.217] |
| TX over O | +0.336 S [+0.248, +0.416] | +0.124 S [-0.000, +0.242] |
| T0 over O | +0.259 S [+0.159, +0.349] | -0.245 S [-0.428, -0.083] |

| quantity | value |
|---|---|
| cost of staleness H (SSE D100 / SSE D5) | 2.367 S [1.926, 2.946] |
| cost of staleness G (SSE D100 / SSE D5) | 2.935 S [2.340, 3.854] |
| cost of staleness T0 (SSE D100 / SSE D5) | 1.678 S [1.552, 1.832] |
| cost of staleness TX (SSE D100 / SSE D5) | 1.318 S [1.232, 1.413] |
| recovered share at D100 | +0.733 S [+0.630, +0.840] |
| TX over T0, D100 minus D5 | +0.192 S [+0.152, +0.236] |

### Band E, M3 rolling origin (fifths 2-5 pooled) (126,806 rows / 2,017 shots / 193 sessions)

| rule | estimator | RMSE | MAE | median abs. error | median rel. error | within 10 % | within 20 % |
|---|---|---|---|---|---|---|---|
| D5 | H | 181.0 | 60.4 | 25.4 | 0.103 | 0.488 S [0.460, 0.515] | 0.752 S [0.730, 0.773] |
| D5 | G | 154.0 | 60.9 | 27.7 | 0.111 | 0.462 S [0.434, 0.490] | 0.729 S [0.706, 0.751] |
| D5 | T0 | 141.7 | 57.3 | 26.3 | 0.106 | 0.477 S [0.449, 0.505] | 0.733 S [0.709, 0.756] |
| D5 | TX | 141.6 | 54.9 | 25.0 | 0.101 | 0.498 S [0.471, 0.526] | 0.754 S [0.732, 0.776] |
| D5 | O | 164.1 | 50.1 | 21.2 | 0.086 | 0.558 S [0.533, 0.583] | 0.809 S [0.792, 0.826] |
| D100 | H | 281.4 | 149.0 | 83.2 | 0.354 | 0.151 S [0.132, 0.170] | 0.292 S [0.261, 0.322] |
| D100 | G | 258.2 | 154.3 | 91.2 | 0.394 | 0.128 S [0.111, 0.144] | 0.254 S [0.226, 0.281] |
| D100 | T0 | 204.0 | 113.4 | 70.2 | 0.272 | 0.210 S [0.190, 0.230] | 0.393 S [0.362, 0.423] |
| D100 | TX | 181.4 | 79.7 | 41.4 | 0.165 | 0.324 S [0.303, 0.345] | 0.579 S [0.553, 0.603] |
| D100 | O | 164.1 | 50.1 | 21.2 | 0.086 | 0.558 S [0.533, 0.583] | 0.809 S [0.792, 0.826] |

| skill | D5 | D100 |
|---|---|---|
| TX over T0 | +0.002 S [-0.054, +0.052] | +0.209 S [+0.096, +0.306] |
| T0 over G | +0.153 S [+0.077, +0.226] | +0.376 S [+0.306, +0.442] |
| T0 over H | +0.387 S [+0.312, +0.455] | +0.475 S [+0.386, +0.551] |
| TX over G | +0.155 S [+0.050, +0.245] | +0.506 S [+0.407, +0.593] |
| TX over H | +0.388 S [+0.298, +0.464] | +0.585 S [+0.493, +0.656] |
| G over H | +0.276 S [+0.222, +0.326] | +0.159 S [+0.014, +0.291] |
| TX over O | +0.256 S [+0.111, +0.371] | -0.221 S [-0.521, +0.023] |
| T0 over O | +0.255 S [+0.125, +0.361] | -0.545 S [-0.887, -0.267] |

| quantity | value |
|---|---|
| cost of staleness H (SSE D100 / SSE D5) | 2.418 S [1.975, 2.916] |
| cost of staleness G (SSE D100 / SSE D5) | 2.810 S [2.378, 3.364] |
| cost of staleness T0 (SSE D100 / SSE D5) | 2.072 S [1.828, 2.394] |
| cost of staleness TX (SSE D100 / SSE D5) | 1.641 S [1.446, 1.896] |
| recovered share at D100 | +0.405 S [+0.192, +0.571] |
| TX over T0, D100 minus D5 | +0.208 S [+0.120, +0.288] |

### Band B2, M1 random (re-partition 1, five folds of whole shots) (239,297 rows / 2,633 shots / 254 sessions)

| rule | estimator | RMSE | MAE | median abs. error | median rel. error | within 10 % | within 20 % |
|---|---|---|---|---|---|---|---|
| D5 | H | 171.7 | 66.9 | 31.6 | 0.098 | 0.506 S [0.480, 0.531] | 0.764 S [0.745, 0.783] |
| D5 | G | 156.9 | 68.7 | 34.3 | 0.105 | 0.482 S [0.457, 0.507] | 0.745 S [0.724, 0.764] |
| D5 | T0 | 138.2 | 57.2 | 28.7 | 0.088 | 0.549 S [0.527, 0.571] | 0.792 S [0.775, 0.808] |
| D5 | TX | 127.4 | 50.3 | 26.1 | 0.079 | 0.589 S [0.569, 0.609] | 0.826 S [0.811, 0.840] |
| D5 | O | 145.8 | 53.3 | 25.3 | 0.079 | 0.589 S [0.565, 0.612] | 0.829 S [0.814, 0.843] |
| D100 | H | 333.0 | 206.5 | 134.8 | 0.432 | 0.122 S [0.107, 0.138] | 0.235 S [0.210, 0.260] |
| D100 | G | 336.0 | 221.2 | 151.0 | 0.492 | 0.098 S [0.085, 0.111] | 0.192 S [0.170, 0.214] |
| D100 | T0 | 211.3 | 125.9 | 83.6 | 0.239 | 0.232 S [0.215, 0.249] | 0.433 S [0.407, 0.459] |
| D100 | TX | 146.6 | 66.4 | 39.9 | 0.121 | 0.428 S [0.413, 0.443] | 0.710 S [0.692, 0.726] |
| D100 | O | 145.8 | 53.3 | 25.3 | 0.079 | 0.589 S [0.565, 0.612] | 0.829 S [0.814, 0.843] |

| skill | D5 | D100 |
|---|---|---|
| TX over T0 | +0.150 S [+0.086, +0.205] | +0.518 S [+0.440, +0.600] |
| T0 over G | +0.225 S [+0.148, +0.309] | +0.605 S [+0.561, +0.644] |
| T0 over H | +0.352 S [+0.284, +0.422] | +0.597 S [+0.550, +0.638] |
| TX over G | +0.341 S [+0.251, +0.438] | +0.810 S [+0.761, +0.853] |
| TX over H | +0.449 S [+0.375, +0.528] | +0.806 S [+0.768, +0.842] |
| G over H | +0.164 S [+0.120, +0.206] | -0.018 S [-0.117, +0.097] |
| TX over O | +0.236 S [+0.079, +0.362] | -0.012 S [-0.216, +0.165] |
| T0 over O | +0.102 S [-0.021, +0.212] | -1.101 S [-1.508, -0.780] |

| quantity | value |
|---|---|
| cost of staleness H (SSE D100 / SSE D5) | 3.763 S [3.149, 4.479] |
| cost of staleness G (SSE D100 / SSE D5) | 4.585 S [3.809, 5.569] |
| cost of staleness T0 (SSE D100 / SSE D5) | 2.338 S [2.048, 2.737] |
| cost of staleness TX (SSE D100 / SSE D5) | 1.324 S [1.251, 1.421] |
| recovered share at D100 | +0.906 S [+0.832, +0.973] |
| TX over T0, D100 minus D5 | +0.369 S [+0.309, +0.424] |

### Band B2, M3 rolling origin (fifths 2-5 pooled) (219,162 rows / 2,117 shots / 190 sessions)

| rule | estimator | RMSE | MAE | median abs. error | median rel. error | within 10 % | within 20 % |
|---|---|---|---|---|---|---|---|
| D5 | H | 166.1 | 65.8 | 31.0 | 0.097 | 0.511 S [0.484, 0.539] | 0.766 S [0.746, 0.787] |
| D5 | G | 152.7 | 67.6 | 33.5 | 0.103 | 0.489 S [0.462, 0.516] | 0.749 S [0.727, 0.770] |
| D5 | T0 | 139.1 | 60.4 | 30.0 | 0.092 | 0.529 S [0.503, 0.556] | 0.775 S [0.754, 0.794] |
| D5 | TX | 129.0 | 54.5 | 27.8 | 0.086 | 0.557 S [0.533, 0.582] | 0.805 S [0.786, 0.823] |
| D5 | O | 139.1 | 52.1 | 24.9 | 0.077 | 0.593 S [0.568, 0.618] | 0.831 S [0.816, 0.847] |
| D100 | H | 330.3 | 206.3 | 133.4 | 0.430 | 0.125 S [0.108, 0.142] | 0.238 S [0.211, 0.266] |
| D100 | G | 330.1 | 220.0 | 149.9 | 0.491 | 0.098 S [0.085, 0.113] | 0.192 S [0.169, 0.217] |
| D100 | T0 | 231.0 | 141.3 | 93.4 | 0.270 | 0.201 S [0.187, 0.215] | 0.389 S [0.364, 0.413] |
| D100 | TX | 162.2 | 79.3 | 45.6 | 0.141 | 0.372 S [0.353, 0.389] | 0.650 S [0.626, 0.672] |
| D100 | O | 139.1 | 52.1 | 24.9 | 0.077 | 0.593 S [0.568, 0.618] | 0.831 S [0.816, 0.847] |

| skill | D5 | D100 |
|---|---|---|
| TX over T0 | +0.139 S [+0.095, +0.191] | +0.507 S [+0.409, +0.596] |
| T0 over G | +0.171 S [+0.093, +0.260] | +0.510 S [+0.464, +0.558] |
| T0 over H | +0.299 S [+0.232, +0.373] | +0.511 S [+0.448, +0.567] |
| TX over G | +0.286 S [+0.195, +0.389] | +0.759 S [+0.697, +0.812] |
| TX over H | +0.397 S [+0.321, +0.482] | +0.759 S [+0.721, +0.798] |
| G over H | +0.155 S [+0.103, +0.202] | +0.001 S [-0.105, +0.123] |
| TX over O | +0.140 S [-0.029, +0.283] | -0.359 S [-0.706, -0.077] |
| T0 over O | +0.001 S [-0.188, +0.145] | -1.757 S [-2.496, -1.197] |

| quantity | value |
|---|---|
| cost of staleness H (SSE D100 / SSE D5) | 3.953 S [3.257, 4.794] |
| cost of staleness G (SSE D100 / SSE D5) | 4.672 S [3.823, 5.804] |
| cost of staleness T0 (SSE D100 / SSE D5) | 2.759 S [2.293, 3.404] |
| cost of staleness TX (SSE D100 / SSE D5) | 1.580 S [1.427, 1.762] |
| recovered share at D100 | +0.795 S [+0.704, +0.875] |
| TX over T0, D100 minus D5 | +0.368 S [+0.283, +0.435] |

### Band B3, M1 random (re-partition 1, five folds of whole shots) (485,836 rows / 2,803 shots / 258 sessions)

| rule | estimator | RMSE | MAE | median abs. error | median rel. error | within 10 % | within 20 % |
|---|---|---|---|---|---|---|---|
| D5 | H | 177.5 | 96.3 | 51.9 | 0.093 | 0.525 S [0.498, 0.552] | 0.764 S [0.744, 0.784] |
| D5 | G | 171.6 | 97.7 | 54.2 | 0.097 | 0.512 S [0.487, 0.538] | 0.756 S [0.736, 0.776] |
| D5 | T0 | 144.0 | 80.6 | 46.1 | 0.081 | 0.579 S [0.557, 0.601] | 0.813 S [0.796, 0.830] |
| D5 | TX | 129.1 | 71.6 | 41.9 | 0.075 | 0.612 S [0.592, 0.633] | 0.841 S [0.826, 0.856] |
| D5 | O | 145.9 | 76.1 | 40.3 | 0.073 | 0.612 S [0.588, 0.636] | 0.832 S [0.816, 0.847] |
| D100 | H | 505.0 | 365.6 | 267.8 | 0.509 | 0.123 S [0.106, 0.140] | 0.226 S [0.201, 0.251] |
| D100 | G | 538.2 | 401.6 | 308.8 | 0.598 | 0.095 S [0.081, 0.110] | 0.177 S [0.155, 0.200] |
| D100 | T0 | 285.9 | 199.8 | 141.1 | 0.233 | 0.233 S [0.217, 0.248] | 0.442 S [0.418, 0.466] |
| D100 | TX | 155.7 | 95.5 | 62.5 | 0.113 | 0.454 S [0.438, 0.469] | 0.740 S [0.721, 0.757] |
| D100 | O | 145.9 | 76.1 | 40.3 | 0.073 | 0.612 S [0.588, 0.636] | 0.832 S [0.816, 0.847] |

| skill | D5 | D100 |
|---|---|---|
| TX over T0 | +0.196 S [+0.160, +0.235] | +0.704 S [+0.671, +0.734] |
| T0 over G | +0.296 S [+0.257, +0.334] | +0.718 S [+0.683, +0.747] |
| T0 over H | +0.342 S [+0.303, +0.380] | +0.679 S [+0.648, +0.708] |
| TX over G | +0.434 S [+0.382, +0.484] | +0.916 S [+0.900, +0.930] |
| TX over H | +0.471 S [+0.425, +0.511] | +0.905 S [+0.890, +0.918] |
| G over H | +0.065 S [+0.029, +0.100] | -0.136 S [-0.176, -0.085] |
| TX over O | +0.217 S [+0.132, +0.288] | -0.138 S [-0.237, -0.051] |
| T0 over O | +0.026 S [-0.056, +0.109] | -2.840 S [-3.414, -2.352] |

| quantity | value |
|---|---|
| cost of staleness H (SSE D100 / SSE D5) | 8.095 S [7.067, 9.305] |
| cost of staleness G (SSE D100 / SSE D5) | 9.835 S [8.463, 11.364] |
| cost of staleness T0 (SSE D100 / SSE D5) | 3.945 S [3.545, 4.438] |
| cost of staleness TX (SSE D100 / SSE D5) | 1.454 S [1.374, 1.550] |
| recovered share at D100 | +0.943 S [+0.923, +0.961] |
| TX over T0, D100 minus D5 | +0.508 S [+0.481, +0.532] |

### Band B3, M3 rolling origin (fifths 2-5 pooled) (460,942 rows / 2,241 shots / 202 sessions)

| rule | estimator | RMSE | MAE | median abs. error | median rel. error | within 10 % | within 20 % |
|---|---|---|---|---|---|---|---|
| D5 | H | 177.3 | 96.1 | 51.6 | 0.092 | 0.527 S [0.500, 0.555] | 0.766 S [0.745, 0.786] |
| D5 | G | 171.3 | 97.5 | 53.9 | 0.096 | 0.515 S [0.488, 0.541] | 0.758 S [0.737, 0.778] |
| D5 | T0 | 152.2 | 86.0 | 48.8 | 0.087 | 0.555 S [0.531, 0.578] | 0.804 S [0.785, 0.822] |
| D5 | TX | 138.5 | 78.5 | 46.2 | 0.084 | 0.572 S [0.550, 0.595] | 0.831 S [0.813, 0.848] |
| D5 | O | 145.4 | 75.9 | 40.1 | 0.072 | 0.615 S [0.590, 0.639] | 0.833 S [0.817, 0.849] |
| D100 | H | 508.6 | 367.1 | 266.8 | 0.507 | 0.125 S [0.108, 0.143] | 0.228 S [0.203, 0.255] |
| D100 | G | 542.6 | 404.0 | 309.1 | 0.598 | 0.097 S [0.082, 0.112] | 0.179 S [0.156, 0.202] |
| D100 | T0 | 319.0 | 223.0 | 156.7 | 0.268 | 0.189 S [0.177, 0.201] | 0.380 S [0.358, 0.401] |
| D100 | TX | 186.6 | 113.9 | 72.3 | 0.134 | 0.386 S [0.365, 0.406] | 0.678 S [0.652, 0.702] |
| D100 | O | 145.4 | 75.9 | 40.1 | 0.072 | 0.615 S [0.590, 0.639] | 0.833 S [0.817, 0.849] |

| skill | D5 | D100 |
|---|---|---|
| TX over T0 | +0.172 S [+0.130, +0.220] | +0.658 S [+0.586, +0.716] |
| T0 over G | +0.211 S [+0.145, +0.273] | +0.654 S [+0.614, +0.689] |
| T0 over H | +0.263 S [+0.201, +0.324] | +0.607 S [+0.568, +0.643] |
| TX over G | +0.346 S [+0.267, +0.421] | +0.882 S [+0.845, +0.907] |
| TX over H | +0.390 S [+0.320, +0.452] | +0.865 S [+0.831, +0.892] |
| G over H | +0.066 S [+0.029, +0.103] | -0.138 S [-0.179, -0.086] |
| TX over O | +0.093 S [-0.032, +0.200] | -0.645 S [-0.963, -0.370] |
| T0 over O | -0.095 S [-0.219, +0.024] | -3.811 S [-4.677, -3.074] |

| quantity | value |
|---|---|
| cost of staleness H (SSE D100 / SSE D5) | 8.233 S [7.129, 9.527] |
| cost of staleness G (SSE D100 / SSE D5) | 10.035 S [8.575, 11.687] |
| cost of staleness T0 (SSE D100 / SSE D5) | 4.395 S [3.829, 5.095] |
| cost of staleness TX (SSE D100 / SSE D5) | 1.814 S [1.676, 1.935] |
| recovered share at D100 | +0.852 S [+0.787, +0.901] |
| TX over T0, D100 minus D5 | +0.486 S [+0.442, +0.526] |

### Band B4, M1 random (re-partition 1, five folds of whole shots) (678,409 rows / 2,697 shots / 255 sessions)

| rule | estimator | RMSE | MAE | median abs. error | median rel. error | within 10 % | within 20 % |
|---|---|---|---|---|---|---|---|
| D5 | H | 217.7 | 131.9 | 80.0 | 0.082 | 0.570 S [0.543, 0.597] | 0.807 S [0.789, 0.824] |
| D5 | G | 218.9 | 136.8 | 85.8 | 0.088 | 0.548 S [0.522, 0.574] | 0.789 S [0.770, 0.807] |
| D5 | T0 | 181.6 | 108.0 | 63.3 | 0.066 | 0.655 S [0.633, 0.676] | 0.856 S [0.841, 0.870] |
| D5 | TX | 162.6 | 97.5 | 58.5 | 0.061 | 0.680 S [0.659, 0.700] | 0.877 S [0.864, 0.889] |
| D5 | O | 183.1 | 105.5 | 62.1 | 0.065 | 0.658 S [0.636, 0.681] | 0.863 S [0.850, 0.877] |
| D100 | H | 725.4 | 561.6 | 448.8 | 0.479 | 0.129 S [0.112, 0.148] | 0.244 S [0.218, 0.272] |
| D100 | G | 793.8 | 631.4 | 533.8 | 0.591 | 0.094 S [0.080, 0.108] | 0.185 S [0.162, 0.210] |
| D100 | T0 | 375.0 | 272.8 | 193.5 | 0.191 | 0.286 S [0.267, 0.306] | 0.518 S [0.491, 0.544] |
| D100 | TX | 223.7 | 149.3 | 99.8 | 0.108 | 0.471 S [0.453, 0.488] | 0.753 S [0.735, 0.770] |
| D100 | O | 183.1 | 105.5 | 62.1 | 0.065 | 0.658 S [0.636, 0.681] | 0.863 S [0.850, 0.877] |

| skill | D5 | D100 |
|---|---|---|
| TX over T0 | +0.198 S [+0.182, +0.213] | +0.644 S [+0.618, +0.669] |
| T0 over G | +0.312 S [+0.288, +0.336] | +0.777 S [+0.757, +0.794] |
| T0 over H | +0.304 S [+0.278, +0.330] | +0.733 S [+0.710, +0.753] |
| TX over G | +0.448 S [+0.426, +0.469] | +0.921 S [+0.912, +0.927] |
| TX over H | +0.442 S [+0.421, +0.462] | +0.905 S [+0.896, +0.913] |
| G over H | -0.011 S [-0.029, +0.009] | -0.197 S [-0.218, -0.176] |
| TX over O | +0.212 S [+0.182, +0.237] | -0.492 S [-0.604, -0.394] |
| T0 over O | +0.017 S [-0.020, +0.052] | -3.193 S [-3.646, -2.793] |

| quantity | value |
|---|---|
| cost of staleness H (SSE D100 / SSE D5) | 11.104 S [10.036, 12.228] |
| cost of staleness G (SSE D100 / SSE D5) | 13.151 S [11.855, 14.476] |
| cost of staleness T0 (SSE D100 / SSE D5) | 4.265 S [3.890, 4.680] |
| cost of staleness TX (SSE D100 / SSE D5) | 1.893 S [1.792, 2.003] |
| recovered share at D100 | +0.841 S [+0.819, +0.863] |
| TX over T0, D100 minus D5 | +0.446 S [+0.420, +0.473] |

### Band B4, M3 rolling origin (fifths 2-5 pooled) (667,078 rows / 2,165 shots / 190 sessions)

| rule | estimator | RMSE | MAE | median abs. error | median rel. error | within 10 % | within 20 % |
|---|---|---|---|---|---|---|---|
| D5 | H | 217.3 | 131.8 | 79.9 | 0.082 | 0.571 S [0.544, 0.598] | 0.808 S [0.789, 0.825] |
| D5 | G | 218.6 | 136.7 | 85.7 | 0.087 | 0.549 S [0.523, 0.576] | 0.789 S [0.770, 0.808] |
| D5 | T0 | 191.7 | 117.7 | 70.5 | 0.074 | 0.619 S [0.597, 0.640] | 0.848 S [0.832, 0.862] |
| D5 | TX | 173.7 | 108.3 | 66.9 | 0.073 | 0.634 S [0.613, 0.653] | 0.869 S [0.856, 0.882] |
| D5 | O | 182.7 | 105.3 | 62.0 | 0.064 | 0.659 S [0.636, 0.683] | 0.864 S [0.850, 0.878] |
| D100 | H | 727.1 | 562.6 | 448.8 | 0.478 | 0.130 S [0.113, 0.149] | 0.245 S [0.219, 0.273] |
| D100 | G | 795.6 | 632.5 | 534.0 | 0.590 | 0.094 S [0.081, 0.109] | 0.186 S [0.163, 0.211] |
| D100 | T0 | 405.2 | 300.4 | 219.8 | 0.230 | 0.215 S [0.201, 0.228] | 0.436 S [0.413, 0.458] |
| D100 | TX | 253.7 | 173.2 | 116.4 | 0.130 | 0.394 S [0.377, 0.410] | 0.700 S [0.680, 0.718] |
| D100 | O | 182.7 | 105.3 | 62.0 | 0.064 | 0.659 S [0.636, 0.683] | 0.864 S [0.850, 0.878] |

| skill | D5 | D100 |
|---|---|---|
| TX over T0 | +0.179 S [+0.151, +0.205] | +0.608 S [+0.579, +0.634] |
| T0 over G | +0.231 S [+0.193, +0.265] | +0.741 S [+0.719, +0.760] |
| T0 over H | +0.222 S [+0.183, +0.258] | +0.689 S [+0.663, +0.713] |
| TX over G | +0.369 S [+0.339, +0.397] | +0.898 S [+0.888, +0.907] |
| TX over H | +0.361 S [+0.332, +0.388] | +0.878 S [+0.867, +0.888] |
| G over H | -0.012 S [-0.030, +0.008] | -0.197 S [-0.218, -0.177] |
| TX over O | +0.097 S [+0.051, +0.137] | -0.928 S [-1.102, -0.774] |
| T0 over O | -0.100 S [-0.161, -0.048] | -3.917 S [-4.542, -3.374] |

| quantity | value |
|---|---|
| cost of staleness H (SSE D100 / SSE D5) | 11.193 S [10.100, 12.322] |
| cost of staleness G (SSE D100 / SSE D5) | 13.245 S [11.928, 14.592] |
| cost of staleness T0 (SSE D100 / SSE D5) | 4.468 S [4.095, 4.868] |
| cost of staleness TX (SSE D100 / SSE D5) | 2.135 S [2.020, 2.255] |
| recovered share at D100 | +0.783 S [+0.757, +0.808] |
| TX over T0, D100 minus D5 | +0.429 S [+0.399, +0.458] |

### Band B5, M1 random (re-partition 1, five folds of whole shots) (1,074,935 rows / 2,789 shots / 259 sessions)

| rule | estimator | RMSE | MAE | median abs. error | median rel. error | within 10 % | within 20 % |
|---|---|---|---|---|---|---|---|
| D5 | H | 207.4 | 127.0 | 78.0 | 0.081 | 0.577 S [0.551, 0.602] | 0.819 S [0.803, 0.834] |
| D5 | G | 210.3 | 134.1 | 85.6 | 0.088 | 0.547 S [0.520, 0.573] | 0.794 S [0.776, 0.811] |
| D5 | T0 | 180.5 | 108.7 | 63.7 | 0.066 | 0.648 S [0.626, 0.670] | 0.854 S [0.839, 0.868] |
| D5 | TX | 162.7 | 98.6 | 59.0 | 0.063 | 0.673 S [0.652, 0.694] | 0.876 S [0.864, 0.888] |
| D5 | O | 179.7 | 104.3 | 61.9 | 0.065 | 0.659 S [0.636, 0.681] | 0.870 S [0.857, 0.881] |
| D100 | H | 621.1 | 475.6 | 369.6 | 0.395 | 0.151 S [0.133, 0.170] | 0.286 S [0.259, 0.313] |
| D100 | G | 690.6 | 541.8 | 434.7 | 0.483 | 0.116 S [0.101, 0.132] | 0.225 S [0.201, 0.251] |
| D100 | T0 | 358.7 | 258.1 | 177.6 | 0.180 | 0.305 S [0.285, 0.324] | 0.541 S [0.514, 0.566] |
| D100 | TX | 226.7 | 152.6 | 101.7 | 0.109 | 0.465 S [0.447, 0.483] | 0.746 S [0.728, 0.763] |
| D100 | O | 179.7 | 104.3 | 61.9 | 0.065 | 0.659 S [0.636, 0.681] | 0.870 S [0.857, 0.881] |

| skill | D5 | D100 |
|---|---|---|
| TX over T0 | +0.187 S [+0.168, +0.205] | +0.600 S [+0.574, +0.625] |
| T0 over G | +0.264 S [+0.241, +0.285] | +0.730 S [+0.708, +0.750] |
| T0 over H | +0.242 S [+0.220, +0.263] | +0.666 S [+0.639, +0.691] |
| TX over G | +0.402 S [+0.379, +0.422] | +0.892 S [+0.884, +0.899] |
| TX over H | +0.384 S [+0.362, +0.404] | +0.867 S [+0.857, +0.875] |
| G over H | -0.029 S [-0.043, -0.014] | -0.236 S [-0.253, -0.219] |
| TX over O | +0.180 S [+0.147, +0.211] | -0.592 S [-0.701, -0.491] |
| T0 over O | -0.008 S [-0.045, +0.025] | -2.984 S [-3.349, -2.645] |

| quantity | value |
|---|---|
| cost of staleness H (SSE D100 / SSE D5) | 8.971 S [8.379, 9.589] |
| cost of staleness G (SSE D100 / SSE D5) | 10.780 S [10.100, 11.465] |
| cost of staleness T0 (SSE D100 / SSE D5) | 3.951 S [3.625, 4.293] |
| cost of staleness TX (SSE D100 / SSE D5) | 1.942 S [1.850, 2.037] |
| recovered share at D100 | +0.804 S [+0.780, +0.827] |
| TX over T0, D100 minus D5 | +0.413 S [+0.382, +0.443] |

### Band B5, M3 rolling origin (fifths 2-5 pooled) (1,053,840 rows / 2,250 shots / 195 sessions)

| rule | estimator | RMSE | MAE | median abs. error | median rel. error | within 10 % | within 20 % |
|---|---|---|---|---|---|---|---|
| D5 | H | 206.4 | 126.6 | 77.7 | 0.081 | 0.578 S [0.551, 0.604] | 0.820 S [0.804, 0.836] |
| D5 | G | 209.6 | 133.7 | 85.3 | 0.088 | 0.548 S [0.521, 0.575] | 0.795 S [0.777, 0.812] |
| D5 | T0 | 185.0 | 113.1 | 66.4 | 0.070 | 0.634 S [0.610, 0.656] | 0.851 S [0.836, 0.865] |
| D5 | TX | 169.3 | 103.9 | 62.9 | 0.068 | 0.650 S [0.629, 0.671] | 0.873 S [0.860, 0.885] |
| D5 | O | 179.0 | 103.9 | 61.6 | 0.064 | 0.660 S [0.637, 0.683] | 0.871 S [0.858, 0.883] |
| D100 | H | 620.6 | 475.1 | 368.7 | 0.394 | 0.151 S [0.133, 0.171] | 0.287 S [0.260, 0.315] |
| D100 | G | 690.5 | 541.4 | 433.8 | 0.482 | 0.117 S [0.102, 0.133] | 0.226 S [0.202, 0.252] |
| D100 | T0 | 380.5 | 278.6 | 198.2 | 0.206 | 0.260 S [0.244, 0.275] | 0.488 S [0.463, 0.511] |
| D100 | TX | 254.7 | 174.0 | 117.0 | 0.129 | 0.399 S [0.381, 0.415] | 0.694 S [0.673, 0.714] |
| D100 | O | 179.0 | 103.9 | 61.6 | 0.064 | 0.660 S [0.637, 0.683] | 0.871 S [0.858, 0.883] |

| skill | D5 | D100 |
|---|---|---|
| TX over T0 | +0.162 S [+0.140, +0.183] | +0.552 S [+0.524, +0.579] |
| T0 over G | +0.221 S [+0.190, +0.249] | +0.696 S [+0.668, +0.722] |
| T0 over H | +0.196 S [+0.168, +0.223] | +0.624 S [+0.587, +0.657] |
| TX over G | +0.347 S [+0.318, +0.374] | +0.864 S [+0.850, +0.876] |
| TX over H | +0.327 S [+0.299, +0.352] | +0.832 S [+0.814, +0.847] |
| G over H | -0.031 S [-0.046, -0.016] | -0.238 S [-0.255, -0.220] |
| TX over O | +0.105 S [+0.067, +0.141] | -1.024 S [-1.202, -0.858] |
| T0 over O | -0.068 S [-0.109, -0.029] | -3.518 S [-3.991, -3.094] |

| quantity | value |
|---|---|
| cost of staleness H (SSE D100 / SSE D5) | 9.044 S [8.435, 9.680] |
| cost of staleness G (SSE D100 / SSE D5) | 10.859 S [10.170, 11.567] |
| cost of staleness T0 (SSE D100 / SSE D5) | 4.230 S [3.880, 4.606] |
| cost of staleness TX (SSE D100 / SSE D5) | 2.262 S [2.111, 2.423] |
| recovered share at D100 | +0.723 S [+0.694, +0.752] |
| TX over T0, D100 minus D5 | +0.390 S [+0.362, +0.419] |

## The value of the frame (target F, D5, M1)

PCHIP and the offline GP wait for the next good value of the chord (frame t left out); they are scored only on rows where a good value exists before and after frame t in the block. T0 and TX are fitted to F.

### Band E (161,229 rows / 2,641 shots / 255 sessions; left out without a reference 4,102; offline GP fell back to hold on 0)

| estimator | RMSE | MAE | median abs. error | median rel. error | within 10 % | within 20 % |
|---|---|---|---|---|---|---|
| H | 220.5 | 48.5 | 16.0 | 0.063 | 0.664 S [0.647, 0.682] | 0.857 S [0.845, 0.868] |
| G | 196.9 | 50.6 | 18.7 | 0.074 | 0.613 S [0.594, 0.631] | 0.826 S [0.812, 0.840] |
| T0 | 188.6 | 47.6 | 16.7 | 0.065 | 0.650 S [0.631, 0.668] | 0.838 S [0.824, 0.851] |
| TX | 185.2 | 46.3 | 16.2 | 0.063 | 0.657 S [0.639, 0.676] | 0.842 S [0.829, 0.855] |
| PCHIP | 182.1 | 39.2 | 12.0 | 0.047 | 0.738 S [0.721, 0.755] | 0.894 S [0.883, 0.903] |
| offline GP | 181.1 | 41.6 | 12.9 | 0.051 | 0.715 S [0.697, 0.733] | 0.875 S [0.863, 0.886] |

| skill | value |
|---|---|
| TX over PCHIP | -0.034 S [-0.120, +0.047] |
| TX over offline GP | -0.045 S [-0.126, +0.029] |
| T0 over PCHIP | -0.072 S [-0.167, +0.015] |
| T0 over offline GP | -0.084 S [-0.175, -0.001] |
| G over PCHIP | -0.168 S [-0.257, -0.084] |
| G over offline GP | -0.181 S [-0.263, -0.105] |
| H over PCHIP | -0.466 S [-0.586, -0.359] |
| offline GP over PCHIP | +0.011 S [-0.019, +0.038] |

### Band B2 (260,567 rows / 2,731 shots / 257 sessions; left out without a reference 5,540; offline GP fell back to hold on 0)

| estimator | RMSE | MAE | median abs. error | median rel. error | within 10 % | within 20 % |
|---|---|---|---|---|---|---|
| H | 160.7 | 38.3 | 16.4 | 0.050 | 0.755 S [0.740, 0.771] | 0.911 S [0.902, 0.919] |
| G | 148.9 | 42.4 | 19.1 | 0.057 | 0.705 S [0.687, 0.722] | 0.880 S [0.868, 0.891] |
| T0 | 143.6 | 37.1 | 15.6 | 0.047 | 0.760 S [0.744, 0.775] | 0.906 S [0.896, 0.915] |
| TX | 139.2 | 35.2 | 15.0 | 0.045 | 0.770 S [0.754, 0.785] | 0.912 S [0.902, 0.921] |
| PCHIP | 127.6 | 27.9 | 11.0 | 0.033 | 0.834 S [0.821, 0.847] | 0.942 S [0.935, 0.948] |
| offline GP | 128.6 | 30.1 | 11.8 | 0.035 | 0.817 S [0.803, 0.831] | 0.929 S [0.921, 0.937] |

| skill | value |
|---|---|
| TX over PCHIP | -0.190 S [-0.355, -0.068] |
| TX over offline GP | -0.172 S [-0.314, -0.060] |
| T0 over PCHIP | -0.265 S [-0.442, -0.136] |
| T0 over offline GP | -0.246 S [-0.397, -0.134] |
| G over PCHIP | -0.361 S [-0.529, -0.233] |
| G over offline GP | -0.340 S [-0.475, -0.235] |
| H over PCHIP | -0.584 S [-0.781, -0.451] |
| offline GP over PCHIP | -0.016 S [-0.052, +0.018] |

### Band B3 (520,191 rows / 2,863 shots / 259 sessions; left out without a reference 9,006; offline GP fell back to hold on 0)

| estimator | RMSE | MAE | median abs. error | median rel. error | within 10 % | within 20 % |
|---|---|---|---|---|---|---|
| H | 120.6 | 44.3 | 23.8 | 0.043 | 0.811 S [0.796, 0.825] | 0.940 S [0.933, 0.947] |
| G | 116.2 | 48.6 | 26.3 | 0.046 | 0.774 S [0.757, 0.790] | 0.917 S [0.907, 0.926] |
| T0 | 108.3 | 41.0 | 21.8 | 0.039 | 0.825 S [0.811, 0.839] | 0.942 S [0.934, 0.949] |
| TX | 99.9 | 37.4 | 20.6 | 0.037 | 0.843 S [0.828, 0.856] | 0.951 S [0.943, 0.957] |
| PCHIP | 92.0 | 29.2 | 14.5 | 0.026 | 0.897 S [0.885, 0.907] | 0.967 S [0.961, 0.972] |
| offline GP | 93.6 | 31.0 | 15.2 | 0.027 | 0.886 S [0.873, 0.897] | 0.961 S [0.954, 0.966] |

| skill | value |
|---|---|
| TX over PCHIP | -0.179 S [-0.309, -0.079] |
| TX over offline GP | -0.137 S [-0.234, -0.057] |
| T0 over PCHIP | -0.387 S [-0.600, -0.234] |
| T0 over offline GP | -0.337 S [-0.498, -0.219] |
| G over PCHIP | -0.597 S [-0.892, -0.386] |
| G over offline GP | -0.540 S [-0.762, -0.375] |
| H over PCHIP | -0.721 S [-0.966, -0.547] |
| offline GP over PCHIP | -0.037 S [-0.084, -0.003] |

### Band B4 (719,777 rows / 2,751 shots / 256 sessions; left out without a reference 11,111; offline GP fell back to hold on 0)

| estimator | RMSE | MAE | median abs. error | median rel. error | within 10 % | within 20 % |
|---|---|---|---|---|---|---|
| H | 129.3 | 60.7 | 35.9 | 0.037 | 0.843 S [0.830, 0.855] | 0.954 S [0.948, 0.959] |
| G | 132.8 | 68.2 | 40.4 | 0.042 | 0.799 S [0.785, 0.814] | 0.934 S [0.926, 0.941] |
| T0 | 120.2 | 56.5 | 32.4 | 0.034 | 0.859 S [0.848, 0.870] | 0.959 S [0.954, 0.963] |
| TX | 108.0 | 51.5 | 30.3 | 0.032 | 0.875 S [0.865, 0.885] | 0.967 S [0.962, 0.971] |
| PCHIP | 101.8 | 42.2 | 22.0 | 0.023 | 0.911 S [0.902, 0.919] | 0.974 S [0.971, 0.978] |
| offline GP | 104.7 | 44.1 | 22.9 | 0.024 | 0.904 S [0.896, 0.913] | 0.971 S [0.967, 0.975] |

| skill | value |
|---|---|
| TX over PCHIP | -0.126 S [-0.190, -0.073] |
| TX over offline GP | -0.065 S [-0.115, -0.017] |
| T0 over PCHIP | -0.394 S [-0.469, -0.331] |
| T0 over offline GP | -0.318 S [-0.378, -0.262] |
| G over PCHIP | -0.702 S [-0.831, -0.594] |
| G over offline GP | -0.609 S [-0.711, -0.521] |
| H over PCHIP | -0.614 S [-0.704, -0.542] |
| offline GP over PCHIP | -0.058 S [-0.089, -0.033] |

### Band B5 (1,137,872 rows / 2,849 shots / 259 sessions; left out without a reference 15,690; offline GP fell back to hold on 0)

| estimator | RMSE | MAE | median abs. error | median rel. error | within 10 % | within 20 % |
|---|---|---|---|---|---|---|
| H | 139.4 | 66.6 | 39.0 | 0.040 | 0.823 S [0.808, 0.836] | 0.946 S [0.940, 0.952] |
| G | 142.7 | 74.8 | 44.4 | 0.045 | 0.776 S [0.758, 0.792] | 0.923 S [0.914, 0.931] |
| T0 | 130.3 | 63.3 | 36.2 | 0.037 | 0.835 S [0.821, 0.848] | 0.950 S [0.944, 0.955] |
| TX | 119.9 | 58.1 | 33.6 | 0.035 | 0.855 S [0.842, 0.867] | 0.958 S [0.953, 0.963] |
| PCHIP | 114.1 | 50.4 | 26.4 | 0.027 | 0.883 S [0.871, 0.893] | 0.965 S [0.961, 0.969] |
| offline GP | 116.8 | 51.7 | 26.8 | 0.028 | 0.879 S [0.867, 0.890] | 0.962 S [0.957, 0.966] |

| skill | value |
|---|---|
| TX over PCHIP | -0.104 S [-0.150, -0.064] |
| TX over offline GP | -0.054 S [-0.095, -0.019] |
| T0 over PCHIP | -0.304 S [-0.356, -0.257] |
| T0 over offline GP | -0.246 S [-0.293, -0.201] |
| G over PCHIP | -0.565 S [-0.639, -0.497] |
| G over offline GP | -0.495 S [-0.561, -0.433] |
| H over PCHIP | -0.493 S [-0.539, -0.450] |
| offline GP over PCHIP | -0.047 S [-0.066, -0.030] |

Readout time 105.0 s.
