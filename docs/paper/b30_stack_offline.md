# B.30 rows: the causal estimators and their combinations against the offline GP (post hoc, not registered)

T_i, cut population, a CES sample visible every 10 ms, rows with a visible sample on both sides. Skill = 1 - SSE(estimator) / SSE(offline GP); the offline GP reads up to 16 samples on each side of the row. 95 % interval over 2,000 resamples of discharges. Stack weights are fitted on one half of the test discharges and applied to the other half, and the reverse.

| estimator | split 42 | split 1 | split 7 | split 123 | intervals excluding 0 (above / below) |
|---|---|---|---|---|---|
| causal variant | -0.073 [-0.154, -0.009] | -0.014 [-0.054, +0.024] | +0.005 [-0.048, +0.047] | -0.014 [-0.073, +0.033] | 0 / 1 |
| network | -0.062 [-0.132, -0.005] | +0.030 [-0.015, +0.067] | +0.003 [-0.051, +0.041] | -0.006 [-0.076, +0.046] | 0 / 1 |
| causal GP | -0.151 [-0.206, -0.111] | -0.121 [-0.159, -0.091] | -0.139 [-0.184, -0.104] | -0.122 [-0.161, -0.090] | 0 / 4 |
| causal stack (causal variant + causal GP + hold + history-only network) | -0.034 [-0.088, +0.007] | +0.011 [-0.024, +0.043] | +0.025 [-0.021, +0.064] | +0.018 [-0.029, +0.057] | 0 / 0 |
| headline stack (network + causal GP + hold) | -0.021 [-0.073, +0.020] | +0.034 [-0.005, +0.067] | +0.018 [-0.030, +0.055] | +0.018 [-0.027, +0.053] | 0 / 0 |

RMSE of the offline GP on these rows (eV): 158.3, 198.2, 181.3, 206.1.
