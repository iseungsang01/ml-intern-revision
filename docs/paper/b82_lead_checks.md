# B.82 — the session lead's checks

Written 2026-11-20 after the batch was read, with code written independently of the implementing agent's
(`ces_prediction/experiments/b82_localize/lead_checks/verify_b82.py`): the time since the last notch recomputed from the grid stamps and the
increment by its classes from the stored estimates of B.75; the robust measures of V1, D650 with own pooling; the specificity
controls and the seed spread pooled again from the stored fits. Skill of A over B = 1 - SSE(A) / SSE(B) on identical rows; target L5.

`py ces_prediction/experiments/b82_localize/lead_checks/verify_b82.py .`

```
(A) tau: largest |mine - cached| over the D1 rows 0 ms; rows with no notch before: mine 69290, cached 69290
   V1 D650 TX over T0: up to 100 ms +0.452 (12462 rows); 100 to 300 ms +0.398 (23038 rows); above 300 ms +0.264 (108715 rows); no notch before +0.417 (16128 rows)
   V1 S    TX over T0: up to 100 ms +0.224 (13718 rows); 100 to 300 ms +0.342 (24352 rows); above 300 ms +0.334 (109767 rows); no notch before -
   V3 D650 TX over T0: up to 100 ms +0.293 (9467 rows); 100 to 300 ms +0.257 (17254 rows); above 300 ms +0.077 (73030 rows); no notch before +0.372 (12448 rows)
   V3 S    TX over T0: up to 100 ms +0.118 (10406 rows); 100 to 300 ms +0.265 (18277 rows); above 300 ms +0.234 (74001 rows); no notch before -
   A5 V1, S: up to 100 ms minus above 300 ms: -0.110 | D650: +0.188
(B) V1, D650, own pooling (measure per re-partition, mean of three)
   T0 MAE 142.3 eV, median 84.9 eV, 90th percentile 336.1 eV, within 10 % 0.395, within 20 % 0.633
   TX MAE 117.5 eV, median 72.0 eV, 90th percentile 274.0 eV, within 10 % 0.445, within 20 % 0.686
   O  MAE 91.9 eV, median 43.4 eV, 90th percentile 217.8 eV, within 10 % 0.600, within 20 % 0.792
   ratios TX over T0: MAE 0.826, median 0.849, 90th percentile 0.815; share within 10 %, TX minus T0: +0.049
   MAE ratio TX over T0, V3 D650: 0.951
   MAE ratio TX over T0, V1 S: 0.828
   MAE ratio TX over T0, V4a D650: 0.867
(C) pooled again from the stored fits of this batch
   V1  D650 over T0: TM +0.084, TXM +0.351, TXs -0.083, TBi +0.238, TBm +0.246, TBo +0.220, TX +0.336 | TX over TM +0.275 | TXM over TX +0.022
   V1  S    over T0: TM +0.074, TXM +0.337, TXs +0.019, TBi +0.235, TBm +0.241, TBo +0.234, TX +0.327 | TX over TM +0.273 | TXM over TX +0.015
   V3  D650 over T0: TM +0.074, TXM +0.210, TXs -0.136, TBi +0.112, TBm +0.142, TBo +0.130, TX +0.190 | TX over TM +0.126 | TXM over TX +0.025
   V3  S    over T0: TM -0.016, TXM +0.227, TXs -0.078, TBi +0.188, TBm +0.177, TBo +0.158, TX +0.231 | TX over TM +0.243 | TXM over TX -0.006
   V4a D650 over T0: TM +0.068, TXM +0.238, TXs -0.092, TBi -0.033, TBm +0.040, TBo -0.007, TX +0.237 | TX over TM +0.181 | TXM over TX +0.002
   V4a S    over T0: TM +0.113, TXM +0.094, TXs -0.681, TBi -0.004, TBm -0.094, TBo -0.159, TX +0.028 | TX over TM -0.096 | TXM over TX +0.068
(D) increment of the chronological fit over ten seeds of the regression
   V4a D650 registered +0.237; ten seeds: smallest +0.216, median +0.225, largest +0.239, sd 0.007
   V4a S    registered +0.028; ten seeds: smallest -0.074, median -0.026, largest +0.041, sd 0.035
   V4b D650 registered +0.216; ten seeds: smallest +0.201, median +0.219, largest +0.234, sd 0.011
   V4b S    registered -0.254; ten seeds: smallest -0.250, median -0.183, largest -0.081, sd 0.055
```

- A: the recomputed tau equals the batch's on every D1 row; the increments by class of tau under V1 and V3 and the differences of A5 equal those of `b82_localize.md`.
- B: MAE, median and 90th percentile of the absolute error, and the ratios of B1, B2, B3, B4, B7 and B8 equal the batch's. The shares within 10 % and 20 % printed here are over the rows with L5 above 100 eV; over all rows, as the batch counts them, they are 0.390 and 0.624 (T0), 0.439 and 0.677 (TX), 0.593 and 0.785 (O), equal to the batch's.
- C, D: the pooled values of C1, C2, C3, C4, C5, C6, C7, of the three BES columns and of the seed spread equal the batch's.
