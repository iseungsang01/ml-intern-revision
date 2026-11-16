# B.37 addendum 1 — the second stage against a linear stack, in the inclusive population, and its step time

Registered: PREREGISTRATION_B37_ADDENDUM1.md. CPU, no network trained. Models fitted on the validation discharges of each split and applied once to its test rows. T_i, natural cadence. Intervals: 95 %, 1,000 resamples of test discharges.

- **A1** (the layer over a linear stack of the causal variant, the causal GP and hold; cut): 3 of 4 -> holds.
- **A2** (the layer over the causal variant with values above 3 keV kept): 1 of 4 -> does not hold.

## Cut population

| split | test rows | layer over causal | stack over causal | layer over the stack | layer fitted on the stack, over the stack | same, over causal | median error, layer over causal |
|---|---|---|---|---|---|---|---|
| 42 | 32,127 | +0.057 [+0.027, +0.090] | +0.038 [+0.024, +0.053] | +0.020 [-0.010, +0.049] | +0.032 [+0.008, +0.055] | +0.069 [+0.042, +0.096] | 0.895 [0.866, 0.921] |
| 1 | 35,138 | +0.082 [+0.053, +0.109] | +0.031 [+0.016, +0.047] | +0.053 [+0.027, +0.079] | +0.052 [+0.027, +0.080] | +0.082 [+0.052, +0.113] | 0.831 [0.790, 0.875] |
| 7 | 35,357 | +0.053 [+0.030, +0.073] | +0.010 [-0.008, +0.029] | +0.044 [+0.018, +0.066] | +0.056 [+0.024, +0.084] | +0.065 [+0.037, +0.088] | 0.936 [0.914, 0.961] |
| 123 | 33,662 | +0.064 [+0.039, +0.094] | +0.034 [+0.022, +0.048] | +0.031 [+0.007, +0.062] | +0.029 [+0.003, +0.061] | +0.062 [+0.035, +0.093] | 0.924 [0.892, 0.955] |

## Inclusive population

| split | test rows | layer over causal | stack over causal | layer over the stack | layer fitted on the stack, over the stack | same, over causal | median error, layer over causal |
|---|---|---|---|---|---|---|---|
| 42 | 32,254 | +0.012 [-0.020, +0.048] | +0.018 [+0.009, +0.031] | -0.006 [-0.042, +0.025] | +0.007 [-0.026, +0.036] | +0.025 [-0.005, +0.056] | 0.904 [0.869, 0.941] |
| 1 | 35,291 | +0.022 [-0.001, +0.054] | +0.021 [+0.004, +0.041] | +0.001 [-0.028, +0.034] | +0.007 [-0.013, +0.031] | +0.028 [+0.006, +0.059] | 0.644 [0.562, 0.725] |
| 7 | 35,500 | +0.041 [+0.026, +0.056] | +0.028 [+0.013, +0.045] | +0.013 [-0.002, +0.026] | +0.026 [+0.015, +0.040] | +0.054 [+0.034, +0.078] | 0.789 [0.755, 0.829] |
| 123 | 33,876 | -0.008 [-0.032, +0.016] | +0.015 [+0.004, +0.033] | -0.024 [-0.051, +0.000] | -0.001 [-0.017, +0.013] | +0.014 [-0.002, +0.033] | 0.718 [0.660, 0.793] |

## A4. Step time (one row at a time, one thread, five repetitions of 2,000 calls)

- layer: median 1.257 ms, largest 99th percentile 2.185 ms.
- error model: median 1.474 ms, largest 99th percentile 2.652 ms.
