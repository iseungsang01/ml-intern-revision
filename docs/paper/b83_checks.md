# B.83 checks

Registration `ces_prediction/experiments/PREREGISTRATION_B83.md`, section 5; implementation `ces_prediction/experiments/PREREGISTRATION_B83_ADDENDUM1.md`, section 7. Each check was run once, before any readout of this batch.

## References (PCHIP and the offline GP against B.70)

- Rows compared over the 15 comparison files (r2-r4 x 5 folds): 668,016.
- Values not bit-identical (through the stored squared errors): PCHIP 0, offline GP 0.

| estimator | r2 | r3 | r4 | pooled (mean of r2-r4) | printed | b70_d1.md | reproduced |
|---|---|---|---|---|---|---|---|
| hold | 232.637 | 232.637 | 232.637 | 232.637 | 232.6 | 232.6 | True |
| causal GP | 192.786 | 192.786 | 192.786 | 192.786 | 192.8 | 192.8 | True |
| PCHIP | 205.958 | 205.958 | 205.958 | 205.958 | 206.0 | 206.0 | True |
| offline GP | 180.754 | 180.754 | 180.754 | 180.754 | 180.8 | 180.8 | True |

Hold and causal GP: B.70's own squared errors; PCHIP and offline GP: this batch's values against B.70's `y_true`.

References passes: **True**

## No look at row t

200 D1 rows (stream [20261183, crc32('no-look')]); the value of row t (a) set to missing, (b) raised by 1000 eV, in a copy of the dataset array; PCHIP and the offline GP recomputed. Rows whose value changed: removed 0, raised 0.

No look passes: **True**

## Rows

D1 rows 222,672; reference formed (a measured value before and after row t in B.70's block) on 222,005; not formed 667 (no value before 0, no value after 667). Offline GP fits that failed on every grid point and fell back to hold: 0. PCHIP on exactly two neighbour times (linear): 2.

| design | scored rows | reference rows | left out (no reference) | discharges | clusters | every estimator finite on the same rows |
|---|---|---|---|---|---|---|
| V1 random (r2-r4) | 222,672 | 222,005 | 667 | 627 | 85 | True |
| V2 cluster-blocked (assignments 1-3) | 222,672 | 222,005 | 667 | 627 | 85 | True |
| V3 rolling origin (fifths 2-5) | 161,744 | 161,199 | 545 | 500 | 64 | True |
| V4a chronological, fit to 31991, score from 32312 | 20,062 | 19,945 | 117 | 95 | 19 | True |
| V4b chronological, fit to 32310, score from 32312 | 20,062 | 19,945 | 117 | 95 | 19 | True |

Rows passes: **True**

## Stored estimates of B.73 (V1, target F, all D1 rows)

| estimator | r2 | r3 | r4 | pooled | printed | b73_targets.md | reproduced |
|---|---|---|---|---|---|---|---|
| T0[F] | 182.986 | 182.971 | 182.860 | 182.939 | 182.9 | 182.9 | True |
| TX[F] | 174.674 | 174.801 | 174.931 | 174.802 | 174.8 | 174.8 | True |
| H | 232.637 | 232.637 | 232.637 | 232.637 | 232.6 | 232.6 | True |
| G | 192.715 | 192.715 | 192.715 | 192.715 | 192.7 | 192.7 | True |

Stored estimates passes: **True**

All four checks pass: **True**
