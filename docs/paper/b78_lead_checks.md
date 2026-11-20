# B.78 — the session lead's checks

Written 2026-11-20 after the batch was read, with code written independently of the implementing agent's
(`ces_prediction/experiments/b78_sparse/lead_checks/verify_b78.py`). Skill of A over B = 1 - SSE(A) / SSE(B) on identical rows; target L5.

`py ces_prediction/experiments/b78_sparse/lead_checks/verify_b78.py .`

```
(A) own pooling of the stored estimates, target L5
  TX over T0 under K10: V1 +0.191, V2 +0.163, V3 +0.075, V4a +0.023, V4b +0.105
  V1 by rule (K1, K2, K5, K10, K20, K50): +0.093, +0.096, +0.136, +0.191, +0.208, +0.279
  K-5: 1 - MSE(TX, K10) / MSE(T0, K5), V1: +0.015 | K-6: MSE(TX, K10) / MSE(TX, K1), V1: 1.430 | K-7: TE over T0, V1, K10: +0.107
  scored rows of one re-partition: 60523; common set L5: 60523
(B) own frame index, kept grid and K10 inputs
  frame index equal to the cached one: True
  scored rows: mine 88749, cached 88749, equal: True
  K10 on the scored rows: rows with a visible value mine 88432, cached 88432; largest |mine - cached|: H 0.00e+00 eV, age 0.00e+00 ms, G 0.00e+00 eV, history scalars 0.00e+00; anchor rows equal: True
  my K10 inputs finite on the cached common set: True (60523 rows)
  V4a, K10 (3587 test rows): T0 largest |mine - stored| 0.00e+00 eV; RMSE hold 235.2, G 176.5, T0 159.4, TX mine 156.8 (stored 157.5), own reading 243.9; TX over T0 mine +0.033, stored +0.023
```

- A: the values of K-1 to K-7 equal those of `b78_sparse.md`.
- B: the frame index, the scored rows and every history input under K10 (newest kept value, its age, the history scalars, the causal GP, the anchor row) recomputed with own code equal the cached ones bit for bit; the lead's fit of T0 under V4 (fit to 31991) equals the stored estimates. TX with the lead's own fast and anchor scalars gives +0.033 where the stored fit gives +0.023 (3,587 test rows; registered interval [-0.077, +0.120]).
