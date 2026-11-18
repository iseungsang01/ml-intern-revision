# B.60 — the error as a share of the reported value: how many frames are within 10 % and within 20 %

Registered 2026-11-18, before any readout. CPU only, stored errors of the pooled cross-validation of B.39; no network is run. Designed by the session lead, run by a delegated agent with the code of B.54.

## 1. Why

The accuracy of ion-temperature measurements is specified and reported as a share of the value: requirement documents ask for 10 % (Young 2008; Mazon et al. 2025 give 10 to 30 % for the ITER charge-exchange system), and the real-time charge-exchange analysis of DIII-D is characterised by the share of measurements within 20 % of the reference analysis, 88 % on core views and 79 % on edge views (Chrystal et al. 2026). The manuscript gives errors in eV and as skills, so a reader cannot set them beside those numbers. The same readout also shows, in units a diagnostician uses, where the estimator differs from the references: its median absolute error is not lower and its 90th percentile is.

## 2. Seen before

Pooled (B.54): percentiles of the absolute error in eV, estimator 46.0 (median), 100.6 (75th), 205.9 (90th); causal GP 44.1, 104.0, 217.6; hold 45.4, 111.4, 254.7; offline GP 37.5, 91.5, 198.0; twin 46.1, 103.8, 216.0. The median reported T_i of the chord is about 509 eV (records). Not seen: any error as a share of the reported value.

## 3. Design

T_i, cut population, the rows of the pooled cross-validation (the row tables of B.54). For every scored row, the relative error of an estimator is its absolute error over the reported value of the row. Quantities, for the estimator, the twin, the causal GP, hold and the offline GP: the share of rows whose relative error is at most 0.10, and at most 0.20; the median relative error. Interval: a re-partition drawn, then its discharges resampled (2,000 draws); differences between two estimators are taken within a draw.

Also by third of the discharges by median T_i, in the rows 1 to 3 after an absent frame, and in the rows that the value rules do not flag (definitions of B.54).

Check before any statement is read: the percentiles of the absolute error of section 2 are reproduced to one decimal.

## 4. Statements

- **T20.** The share of rows within 20 % is higher for the estimator than for the causal GP: the interval of the difference is above zero.
- **T20h.** The same against hold.
- **T10.** The share of rows within 10 % is not lower for the estimator than for the causal GP by more than 0.02: the interval of the difference lies above -0.02.

Reported: every share with its interval; the differences against the twin and the offline GP; the median relative errors; the strata.

## 5. What goes into the paper

- The shares within 10 % and within 20 % of the estimator, the causal GP and hold are given in the main result in one sentence, beside the requirement of 10 % and the published 79 % within 20 % of a real-time edge analysis, with the statement that the reference of that number is an offline analysis of the same spectra and ours is the reported value of the frame itself, so the two are not the same comparison.
- T20 holds: the sentence says that more frames are within 20 % with the estimator than with the causal GP. T20 fails: the shares are given and no ordering is stated at 20 %.
- T10 fails: the limit on the typical row says that fewer frames are within 10 % with the estimator than with the causal GP, with the numbers.

## 6. Expectations

Within 20 %: estimator 0.74 to 0.80, causal GP lower by 0.003 to 0.02 (T20 holds or fails by a small margin), hold lower by 0.015 to 0.04 (T20h holds). Within 10 %: all between 0.48 and 0.58, the estimator below the causal GP by 0.00 to 0.02 (T10 holds). Median relative error 0.085 to 0.10.

## 7. Limits known in advance

The reported value carries its own scatter, so a share within 10 % of it is bounded above by that scatter and is not a share within 10 % of the ion temperature. Low values of T_i make the relative error large for a fixed error in eV; the thirds show this.

## Addendum 1 (2026-11-18): the filters of B.56 — registered after the statements of B.60 and the skills of B.56 were read, before any relative error of a filter was computed

B.60 found that, as a share of the reported value, the recurrent estimator is not above the causal GP (within 20 %: 0.770 against 0.772) and is below it in the coldest third of the discharges. B.56 found that a five-parameter filter whose state is the logarithm of the value has a median absolute error below that of hold (ratio 0.921 [0.906, 0.940]) and a fast-channel increment of +0.049. An estimator of the logarithm weighs a small value like a large one, so the same readout is made for the filters KF0 and KFX of B.56 (signed outputs in its cache), on the rows of re-partitions r3 to r9 (r2 was the development set of the filter and is reported apart), with the definitions, interval and strata of B.60.

- **F20.** The share of rows within 20 % is higher for KFX than for the causal GP: interval of the difference above zero.
- **F10.** The share of rows within 10 % is higher for KFX than for the causal GP: interval above zero.
- **F20x.** The share within 20 % is higher for KFX than for KF0: interval above zero.
- **F20c.** In the coldest third of the discharges the share within 20 % of KFX is not below that of the causal GP by more than 0.02: interval above -0.02.

Reported: the shares and median relative errors of KF0 and KFX beside those of B.60 on the same rows; KFX against hold, the twin and the recurrent estimator; the strata of B.60; the rows with a reported value below 50 eV.

What follows: if F20 and F10 hold, the paper says that an estimator of the logarithm with the same inputs raises the share of frames within 10 % and within 20 % of the reported value above that of the CES history alone, which the recurrent estimator does not, and the filter is named as the estimator to use where relative accuracy is the requirement. If they fail, the shares are given and the limit on the typical row stays as B.60 left it.

Expectations: KFX within 20 % 0.775 to 0.80, above the causal GP by +0.003 to +0.02 (F20 holds or fails by a small margin); within 10 % 0.575 to 0.61, above the causal GP by +0.005 to +0.03 (F10 holds); F20x +0.003 to +0.015, holds; F20c -0.01 to +0.01, holds.

