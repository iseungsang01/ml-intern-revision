# B.39 arm A, each initialisation by itself (after the fact)

T_i, cut population, the four file-level splits, test rows of `eval_seq`. Causal variant and history-only twin of B.39 (epoch cap 80), three initialisations per split. Skill = 1 - sum of squared errors of A over that of R on the same rows. Intervals: 95 %, 2,000 resamples of the test discharges (one initialisation at a time; the registered readout of S1 and S2 also draws the initialisation). Nothing here changes a registered verdict.

| split | initialisation | causal variant over causal GP | over the twin of the same initialisation | that twin over causal GP |
|---|---|---|---|---|
| 42 | 42 | +0.081 [+0.034, +0.121] | +0.071 [+0.038, +0.106] | +0.011 [-0.026, +0.047] |
| 42 | 1 | +0.061 [-0.029, +0.120] | +0.099 [+0.015, +0.159] | -0.042 [-0.089, -0.003] |
| 42 | 7 | +0.058 [-0.030, +0.118] | +0.042 [-0.036, +0.097] | +0.017 [-0.025, +0.054] |
| 1 | 1 | +0.094 [+0.067, +0.121] | +0.022 [+0.001, +0.046] | +0.074 [+0.046, +0.100] |
| 1 | 7 | +0.101 [+0.064, +0.134] | +0.037 [+0.000, +0.069] | +0.067 [+0.038, +0.095] |
| 1 | 123 | +0.104 [+0.076, +0.131] | +0.064 [+0.042, +0.087] | +0.042 [+0.008, +0.075] |
| 7 | 7 | +0.144 [+0.107, +0.183] | +0.123 [+0.097, +0.152] | +0.024 [-0.006, +0.051] |
| 7 | 123 | +0.138 [+0.101, +0.173] | +0.072 [+0.039, +0.109] | +0.071 [+0.043, +0.097] |
| 7 | 42 | +0.142 [+0.096, +0.189] | +0.071 [+0.023, +0.121] | +0.076 [+0.050, +0.100] |
| 123 | 123 | +0.101 [+0.062, +0.140] | +0.060 [+0.025, +0.101] | +0.044 [+0.006, +0.075] |
| 123 | 42 | +0.130 [+0.084, +0.178] | +0.114 [+0.052, +0.184] | +0.018 [-0.077, +0.081] |
| 123 | 1 | +0.064 [-0.058, +0.146] | +0.008 [-0.115, +0.086] | +0.056 [+0.029, +0.082] |

Of the 12 trainings the causal variant has a lower squared error than the causal GP in 12 and than the twin of the same initialisation in 12; the discharge-level interval is above zero in 9 and in 10.

## Mean squared error over the three initialisations, discharges resampled

| split | causal variant over causal GP | over the twin (mean over its three initialisations) |
|---|---|---|
| 42 | +0.067 [-0.002, +0.120] | +0.071 [+0.008, +0.118] |
| 1 | +0.100 [+0.068, +0.126] | +0.041 [+0.017, +0.065] |
| 7 | +0.141 [+0.103, +0.178] | +0.089 [+0.057, +0.124] |
| 123 | +0.098 [+0.036, +0.150] | +0.062 [+0.027, +0.099] |

## The initialisation with the largest error on each split

| split | initialisation | share of its positive excess squared error over the causal GP held by three discharges | discharges with a positive excess |
|---|---|---|---|
| 42 | 7 | 0.79 | 33 of 96 |
| 1 | 1 | 0.47 | 26 of 96 |
| 7 | 123 | 0.57 | 26 of 96 |
| 123 | 1 | 0.82 | 28 of 96 |
