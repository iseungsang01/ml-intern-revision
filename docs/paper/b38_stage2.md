# B.38 — the second stage fitted on cross-fitted predictions of the training discharges

Registered: PREREGISTRATION_B38.md with amendment 1. 16 GPU runs (the causal variant trained on three of four folds of the training discharges of each split, epoch cap 30); read-outs on a CPU. T_i, cut population, natural cadence. The correction is applied once to the test rows of the frozen row table. Intervals: 95 %, 1,000 resamples of test discharges.

- **G1** (the correction fitted on training and validation discharges over the one fitted on validation discharges alone): 3 of 4 -> holds.
- **G2** (over the linear stack fitted on the same rows): 4 of 4 -> holds.
- **G3** (the adaptive 90 % interval covers 0.87 to 0.93 and is shorter than a constant one): 4 of 4 -> holds.

| split | rows: out-of-fold / validation / test | G1: both over validation only | G2: both over the stack | G3: coverage | G3: median half-width over constant | reported: coverage with an out-of-fold scale (width ratio) |
|---|---|---|---|---|---|---|
| 42 | 137,876 / 49,830 / 32,127 | +0.022 [-0.003, +0.046] | +0.036 [+0.007, +0.063] | 0.905 [0.894, 0.915] | 0.528 [0.453, 0.638] | 0.909 [0.898, 0.919] (0.538) |
| 1 | 137,776 / 46,919 / 35,138 | +0.033 [+0.017, +0.054] | +0.085 [+0.058, +0.113] | 0.891 [0.873, 0.907] | 0.693 [0.593, 0.833] | 0.895 [0.877, 0.911] (0.705) |
| 7 | 139,408 / 45,068 / 35,357 | +0.035 [+0.017, +0.055] | +0.074 [+0.044, +0.102] | 0.901 [0.890, 0.911] | 0.567 [0.486, 0.684] | 0.905 [0.894, 0.916] (0.577) |
| 123 | 139,954 / 46,217 / 33,662 | +0.022 [+0.008, +0.039] | +0.056 [+0.025, +0.090] | 0.894 [0.880, 0.906] | 0.499 [0.413, 0.653] | 0.898 [0.885, 0.910] (0.508) |

## Reported

| split | correction (both) over the causal variant | over the causal GP | over hold | fitted on validation only, over causal | fitted on out-of-fold rows only, over causal | linear stack (both) over causal | median error, correction over causal | over hold |
|---|---|---|---|---|---|---|---|---|
| 42 | +0.078 [+0.043, +0.113] | +0.140 [+0.103, +0.182] | +0.407 [+0.374, +0.436] | +0.057 [+0.028, +0.089] | +0.073 [+0.045, +0.105] | +0.043 [+0.021, +0.069] | 0.897 [0.868, 0.925] | 0.911 [0.865, 0.962] |
| 1 | +0.113 [+0.083, +0.147] | +0.194 [+0.162, +0.225] | +0.455 [+0.422, +0.483] | +0.082 [+0.054, +0.113] | +0.111 [+0.080, +0.144] | +0.031 [+0.015, +0.046] | 0.819 [0.776, 0.866] | 0.924 [0.864, 0.985] |
| 7 | +0.087 [+0.060, +0.112] | +0.204 [+0.160, +0.246] | +0.453 [+0.418, +0.482] | +0.053 [+0.029, +0.073] | +0.089 [+0.062, +0.117] | +0.014 [-0.002, +0.032] | 0.917 [0.895, 0.941] | 0.861 [0.820, 0.907] |
| 123 | +0.085 [+0.054, +0.119] | +0.168 [+0.123, +0.217] | +0.457 [+0.422, +0.488] | +0.064 [+0.039, +0.093] | +0.080 [+0.048, +0.115] | +0.030 [+0.016, +0.047] | 0.894 [0.864, 0.924] | 0.890 [0.849, 0.938] |

| split | RMSE (eV): hold / causal GP / causal variant / corrected | median absolute error (eV): hold / causal variant / corrected | half-widths (eV): adaptive median / constant |
|---|---|---|---|
| 42 | 204.8 / 170.1 / 164.3 / 157.7 | 44.2 / 44.9 / 40.3 | 115.4 / 218.7 |
| 1 | 255.3 / 210.0 / 200.2 / 188.6 | 49.5 / 55.9 / 45.8 | 144.2 / 208.0 |
| 7 | 234.7 / 194.6 / 181.6 / 173.6 | 49.8 / 46.7 / 42.8 | 119.7 / 211.3 |
| 123 | 270.3 / 218.3 / 208.2 / 199.1 | 43.2 / 43.0 / 38.4 | 104.6 / 209.7 |

## The out-of-fold networks

Skill over hold and over the causal GP of each out-of-fold network on its held-out fold of training discharges, beside that of the full network on the test discharges.

| split | full network on test: over hold / over GP | fold 0 | fold 1 | fold 2 | fold 3 | epochs run (best) per fold |
|---|---|---|---|---|---|---|
| 42 | +0.357 / +0.068 | +0.368 / +0.082 | +0.277 / -0.052 | +0.385 / +0.089 | +0.365 / +0.022 | 30 (28), 30 (28), 30 (27), 30 (30) |
| 1 | +0.385 / +0.091 | +0.352 / +0.051 | +0.366 / +0.052 | +0.349 / +0.049 | +0.368 / +0.054 | 22 (16), 30 (24), 22 (16), 26 (20) |
| 7 | +0.401 / +0.128 | +0.376 / +0.106 | +0.387 / +0.063 | +0.348 / +0.018 | +0.355 / +0.058 | 30 (30), 28 (22), 14 (8), 30 (30) |
| 123 | +0.407 / +0.091 | +0.354 / +0.081 | +0.360 / +0.036 | +0.364 / +0.069 | +0.364 / +0.044 | 30 (30), 30 (26), 30 (30), 27 (21) |

Check before each readout (five test files recomputed against the row table): 0 of 1,165 rows deviate, 0 of 1,545 rows deviate, 0 of 2,493 rows deviate, 0 of 957 rows deviate.
