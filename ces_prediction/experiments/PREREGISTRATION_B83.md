# B.83 — the nowcast against interpolation that waits for the next value, with the paper's estimator, under every design

Registered 2026-11-20 (night), before any run of this batch. CPU only; no network, no GPU, no new fit of the paper's estimators (stored estimates of B.73) and the two interpolation references computed from the files. The session lead designed it. An AI agent implements and runs it under the author's supervision. No existing file is edited.

## 1. Why

The author's framing of the work (2026-11-16, restored 2026-11-20 night, `PAPER_DESIGN.md` section 1): solve nowcasting at the pedestal top, that is, find the best model against interpolation and the like, at 10 ms in real time. The comparison with interpolation exists on the one definition of the data (D1) only for the recurrent network and its two-stage form, under the random design (B.70, `docs/paper/b70_d1.md`): RMSE on the value of the frame 232.6 eV (hold), 206.0 (PCHIP), 192.8 (causal GP), 180.8 (offline GP), 174.9 (two-stage estimator); two-stage over PCHIP +0.279 [+0.252, +0.303], over the offline GP +0.064 [+0.039, +0.086]; median absolute error 38.1 (PCHIP), 37.5 (offline GP), 41.2 (two-stage). The paper's estimator is now the network-free TX (B.73), which has the same MSE as the two-stage estimator on the value of the frame under the random design. This batch makes the comparison directly for T0 and TX, under all four designs, with cluster intervals and with the robust measures.

**Status.** Exploratory on used data. Known when this was written: the B.70 numbers above (random design, network estimators); every readout of B.73 to B.82. Not known: any comparison of T0 or TX with PCHIP or the offline GP; any such comparison under the cluster-blocked, rolling-origin or chronological design; the robust measures of T0 and TX on the value of the frame.

## 2. Data, rows, target, estimators

- Rows: the D1 rows of `data/.b73/rows.npz` on which the value of the frame is the target (target F of B.73) and the stored estimates of B.73 exist: `T0[F]`, `TX[F]` (fitted to F with the previous frame visible), hold H and the causal GP G. Units and designs of B.73 (V1, V2, V3, V4a, V4b).
- **Interpolation references** (they read values after row t and are not available in real time): PCHIP through the measured values of the segment with row t left out, evaluated at the stamp of row t; and the offline GP of `baselines_interpolation.predict_gp` (neighbours on both sides, row t left out), exactly as B.70 computed them (`ces_prediction/experiments/b70_d1/`; the agent reuses that code path or its cached per-row references and states which in the addendum). A row where a reference cannot be formed (no measured value on one side within the segment) is left out of every comparison with that reference; counts are reported. The value of row t itself is never read by any estimator or reference.
- Held values and values above 3 keV are missing everywhere, as in D1 (also for the references).
- Target: F, the reported value of row t. The level L5 is not used for the references, because they read rows inside its window; for L5 the table of B.82 (robust measures with the previous frame visible) is cited.

## 3. Intervals

95 %, 20,000 draws, shot clusters resampled (re-partition or assignment drawn first under V1 and V2), seed stream `[20261183, crc32(<statement id or cell name>)]`; the discharge form beside it; verdicts on the cluster interval. Skill of A over R = 1 - MSE(A)/MSE(R) on identical rows; a ratio of error measures is A over R.

## 4. Statements

| id | statement | rule of judgement | expectation |
|---|---|---|---|
| I1 | V1: TX over PCHIP lies above zero | lower end above zero | holds |
| I2 | V1: TX over the offline GP lies above zero | lower end above zero | holds |
| I3 | V2: TX over PCHIP lies above zero | lower end above zero | holds |
| I4 | V2: TX over the offline GP lies above zero | lower end above zero | uncertain, leaning to holds |
| I5 | V3: TX over PCHIP lies above zero | lower end above zero | holds |
| I6 | V3: TX over the offline GP lies above zero | lower end above zero | uncertain |
| I7 | V4a: TX over PCHIP lies above zero | lower end above zero | holds |
| I8 | V4a: TX over the offline GP lies above zero | lower end above zero | uncertain |
| I9 | V1: T0 (history only) over PCHIP lies above zero | lower end above zero | holds |
| I10 | V1: T0 over the offline GP has an interval that contains zero or lies above it (the causal history-only estimate is not worse than the interpolation that waits) | lower end of the interval above zero, or the interval contains zero | holds |
| I11 | V1: the ratio of the median absolute errors, TX over the offline GP, lies below one | upper end below one | does not hold (in B.70 the offline GP has the lower median) |
| I12 | V1: the ratio of the median absolute errors, TX over PCHIP, lies below one | upper end below one | does not hold |
| I13 | V1: the ratio of the mean absolute errors, TX over the offline GP, lies below one | upper end below one | uncertain |
| I14 | V1: the ratio of the 90th percentiles of the absolute error, TX over the offline GP, lies below one | upper end below one | holds |

Reported without a statement: for every design the table of H, PCHIP, G, the offline GP, T0 and TX with RMSE, MAE, median, 75th and 90th percentile of the absolute error, the median relative error (rows with the value above 100 eV), and the shares of rows within 10 % and 20 % of the value; the pairs T0 and TX over each of H, G, PCHIP and the offline GP with intervals; V4b; the counts of rows left out for a missing reference.

## 5. Checks before the readout (each run once; a failed check stops the batch and is reported)

- **References.** On the rows of B.70's readout (re-partitions r2 to r4) the PCHIP and offline-GP values of this batch equal B.70's bit for bit, and the pooled RMSE of hold, causal GP, PCHIP and the offline GP reproduce the values of `b70_d1.md` to the printed digit.
- **No look at row t.** For 200 random rows the two references are recomputed from a copy of the file in which the value of row t is (a) removed and (b) raised by 1000 eV: unchanged bit for bit.
- **Rows.** Every estimator of a comparison is scored on the same rows; counts per design printed.
- **Stored estimates.** The pooled RMSE of `T0[F]`, `TX[F]`, H and G under V1 reproduce those of `b73_targets.md` to the printed digit on B.73's rows.

Choices that this text leaves open are written into `PREREGISTRATION_B83_ADDENDUM1.md` before the checks and before any readout. This file is not edited after registration.

## 6. What goes into the paper (decided now)

- The table of the estimators (the main table of the paper: hold, PCHIP, causal GP, offline GP, the history-only estimator, the paper's estimator, the recurrent network and its two-stage form) is filled from this batch for the value of the frame, with the two references marked as reading the following value.
- If I1 and I2 hold, the abstract and the conclusion say that the causal nowcast has a lower MSE than interpolation that waits for the next value, with the least favourable design in which each holds named. For a design in which a statement does not hold, the text says that it is not established there.
- If I2 does not hold, the sentence is written for PCHIP only (if I1 holds) and the text says that against the offline GP the nowcast is not established to be lower.
- I11 to I14 are reported whatever they are. If I11 and I12 do not hold, as expected, the text says in the same paragraph that the interpolation that waits has the lower median error and that the nowcast's advantage is in the mean squared error and the upper tail.
- No estimator of the paper changes because of this batch.

## 7. Output

Code in `ces_prediction/experiments/b83_interp/`; caches in `data/.b83/` (git-ignored); `docs/paper/b83_interp.md` and `.json` (statements with value, cluster interval, discharge interval, verdict, expectation; the tables; counts), `docs/paper/b83_checks.md`.
