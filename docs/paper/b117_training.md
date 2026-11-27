# B117: one LSTM training diagnostic

registered convergence criterion met

Unit: V1_r2_k0; architecture: lstm; seed: 1701.

All four original validation designs are accounted for below; none is primary.

V1_r2_k0: RUN: TRAIN/VAL optimization pilot only.
V2_s1_k0: NOT RUN.
V3_q3: NOT RUN.
V4a: NOT RUN.

| Quantity | Original B116 fit | Focused fit |
|---|---:|---:|
| Completed epochs | 17 | 60 |
| Best epoch | 15 | 59 |
| Completed updates | 1410 | 4975 |
| Total updates including partial epoch | 1439 | 4975 |
| Partial updates outside complete history | 29 | 0 |
| Elapsed seconds | 900.1772 | 3044.9323 |
| Best equal-rule VAL MSE (eV²) | 44305.892 | 41055 |
| Best VAL RMSE (eV) | 210.48965 | 202.62033 |
| Online TRAIN MSE at best VAL epoch (eV²) | 52492.957 | 47326.726 |
| Descriptive VAL minus online TRAIN gap (eV²) | -8187.0649 | -6271.7261 |
| Final complete-epoch VAL MSE (eV²) | 45497.67 | 41061.206 |
| Stop reason | deadline_in_incomplete_epoch | registered_statistical_stop |

Best equal-rule VAL MSE change: -3250.8919 eV²; relative reduction: +7.337381%.

Selected best epoch relative to original fit: later.

Selected weight arrays exactly identical to the original checkpoint: False.

| VAL rule | Original best-checkpoint MSE | Focused best-checkpoint MSE | Change (eV²) | Relative reduction |
|---|---:|---:|---:|---:|
| D10 | 31098.755 | 28285.546 | -2813.2089 | +9.046050% |
| D50 | 36041.349 | 32614.707 | -3426.6418 | +9.507529% |
| D100 | 39827.056 | 36319.583 | -3507.4731 | +8.806760% |
| D300 | 47188.522 | 43858.582 | -3329.9398 | +7.056673% |
| D650 | 53575.416 | 48890.79 | -4684.6268 | +8.743986% |
| D1000 | 55029.478 | 54315.768 | -713.70982 | +1.296959% |
| S | 47380.669 | 43100.026 | -4280.6432 | +9.034577% |

One fixed unit/architecture/seed pilot, freshly initialized; no TEST evaluation or GP comparison.
The checkpoint and convergence decision use VAL repeatedly; these are optimization diagnostics.
Online TRAIN loss uses changing sampled-rule supports and changing parameters within each epoch.
TRAIN MSE is normalized SSE/count multiplied by the frozen TRAIN Ti standard deviation squared.
VAL is an equal-seven-rule mean on fixed common VAL supports; the TRAIN/VAL gap is descriptive.
The registered stop is an operational VAL plateau criterion, not mathematical convergence or guaranteed optimality.
No confidence interval or independent-sample generalization conclusion is produced.
A safety stop is not successful convergence, even when the best VAL error decreases.

The latest validated recovery checkpoint retains optimizer, scheduler, RNG and next-epoch state; it is distinct from the selected best weights.
Recovery hashes and the last validated epoch are recorded in the JSON. No automatic restart is authorized.
The figure shows all logged complete epochs. The vertical marker indicates the original fit epoch boundary.
Full curves, all epoch learning rates and cumulative updates are retained in the JSON.
