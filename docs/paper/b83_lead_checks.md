# B.83 — the session lead's checks

Written 2026-11-21 after the batch was read, with code written independently of the implementing agent's
(`ces_prediction/experiments/b83_interp/lead_checks/verify_b83.py`). Skill of A over B = 1 - SSE(A) / SSE(B) on identical rows; target: the value of the frame.

`py ces_prediction/experiments/b83_interp/lead_checks/verify_b83.py .`

```
(A) own PCHIP on 19355 rows of 60 discharges: largest |mine - cached| 539 eV; exactly equal on 19353 rows
(B) statement values from the stored estimates and the cached references (target: the value of the frame)
   V1  rows of one part 222005 | MSE skill: TX over PCHIP +0.278, TX over offline GP +0.059, T0 over PCHIP +0.209, T0 over offline GP -0.031
       ratios TX over offline GP: median 1.059, MAE 0.977, 90th percentile 0.947; TX over PCHIP: median 1.043
   V2  rows of one part 222005 | MSE skill: TX over PCHIP +0.261, TX over offline GP +0.037, T0 over PCHIP +0.204, T0 over offline GP -0.038
   V3  rows of one part 161199 | MSE skill: TX over PCHIP +0.224, TX over offline GP -0.007, T0 over PCHIP +0.186, T0 over offline GP -0.056
   V4a rows of one part  19945 | MSE skill: TX over PCHIP +0.153, TX over offline GP -0.035, T0 over PCHIP +0.136, T0 over offline GP -0.055
```

- A: PCHIP recomputed with scipy on the grid of B.73, through every measured value of the row's segment with the row left out, equals the cached reference exactly on 19,353 of 19,355 rows of 60 discharges. The two other rows lie beside a stretch of 0.63 s without a kept row: B.70's block ends there (the reference, as registered, is formed inside that block), the grid segment of B.73 does not. A first version of this check took only the D1 rows as neighbours and differed on 532 rows; the reference reads every measured value of the block, also those of rows that are not scored.
- B: the MSE skills of I1 to I10 and the ratios of I11 to I14 equal those of `b83_interp.md`.
