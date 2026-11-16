# B.37, addendum 1 — the second stage against a linear stack, in the inclusive population, and its step time

Registered 2026-11-16, after part V of B.37 was read (a boosted layer on the signed residual of the causal variant, fitted on validation discharges, has skill +0.053 to +0.082 over the causal variant on four of four cut splits) and before any readout below was computed. CPU only; no network is trained. The GPU was in use by another project when this was written, so the GPU batch that part V calls for waits.

Everything is as in part V (validation discharges of the split, the same 15 inputs, `HistGradientBoostingRegressor` with default settings, applied once to the test rows of the frozen row table; 95 % intervals over 1,000 resamples of test discharges; a statement holds on at least three of four splits), with the same check of five test files before each readout.

- **A1 (more than averaging).** A linear stack of the causal variant, the causal GP and hold (least squares with an intercept, fitted on the same validation rows) is applied to the test rows. The skill of the layer over that stack has an interval above zero. Cut population.
- **A2 (inclusive population).** With values above 3 keV kept (`rows_incl_s*`, causal variant `data/.b29_norm0_incl_s{seed}`, split `data/.b1_manifest_s{seed}`), the skill of the layer over the causal variant has an interval above zero.
- **A3 (reported).** The stack over the causal variant in both populations; the layer over the stack in the inclusive population; the layer fitted on the stack's three estimators plus the 15 inputs; the median absolute error of the layer over that of the causal variant in the inclusive population.
- **A4 (step time, reported).** One row at a time, one thread, 2,000 calls after 200 warm-up calls, five repetitions: median and largest 99th percentile of the prediction time of the layer and of the error model.

Decision: A1 holds → the second stage is described as a correction and not as an average of estimators. A1 fails → the gain of part V is stated as not separated from that of a three-weight average. A2 fails → the second stage is stated for the cut population only.

Expectations: A1 +0.010 to +0.050, holds; stack over causal +0.015 to +0.040; A2 +0.000 to +0.040, holding on two to four splits; step time 0.2 to 2 ms at the median.
