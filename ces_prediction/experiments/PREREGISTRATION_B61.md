# B.61 — absolute errors, a score that 1 % of the rows does not decide, sensitivity to discharges, and the dose of a beam interruption

Registered 2026-11-18, before any readout of this batch. CPU only, stored errors and data files; no network is run. Designed by the session lead after the twelfth independent grading, run by delegated agents with the code of B.54, B.56 and B.58.

## 1. Why

The twelfth independent grading (PPCF 20 %, range 10 to 32) named four things that the data in hand can answer. (1) The manuscript gives skills and no absolute size: a reader cannot see that +0.089 in skill is a few per cent of the root-mean-square error. (2) One per cent of the rows holds about half of the squared error, so the primary score is decided by few rows; a co-primary score that is not is asked for. (3) The four test sets overlap and no sensitivity to the discharges is given beyond the ten with the largest reduction. (4) The fall of the reported T_i after a beam interruption has no dose–response: interruptions that remove one, two or three frames should give a graded response if the fall is a response to the interruption.

## 2. Seen before

Pooled root-mean-square error of the estimator 184.6 [174.5, 194.7] eV (B.55). Percentiles of the absolute error (B.54): estimator 46.0, 100.6, 205.9, 309.8, 715.4 eV at the 50th, 75th, 90th, 95th, 99th; causal GP 44.1, 104.0, 217.6, 333.5, 777.1. Without the ten discharges with the largest reduction the skill over the causal GP is +0.076 [+0.050, +0.098] (B.54). After events of one absent frame the first-row ratio of T_i is 0.814; for 67 events of two absent frames 0.926 and for 65 of three 0.830, with irregular shapes (B.58, reported apart). Not seen: absolute errors of the other estimators with intervals; any trimmed or bounded score; removal of more than ten discharges; the overlap of the test sets; the dose readout with intervals.

## 3. Design

T_i, cut population.

**Part A — absolute and robust scores (pooled rows of B.39; filters of B.56 on r3 to r9).** For the estimator, the twin, the causal GP, hold, the offline GP, KF0 and KFX: the root-mean-square error and the mean absolute error in eV; the difference of each to the causal GP in eV and in per cent. Two robust skills of A over R, on the same rows: (i) the Huber skill, 1 minus the ratio of summed Huber losses with the threshold at 200 eV (quadratic below, linear above); (ii) the trimmed skill, the mean-squared-error skill on the rows that remain when the 1 % of rows with the largest squared error of hold are removed (the same rows for every estimator). Interval: a re-partition drawn, then its discharges resampled (2,000 draws).

**Part B — discharges.** Skill of the estimator over the causal GP and over the twin with the k discharges with the largest squared-error reduction against the causal GP removed, k = 10, 32 and 64 (about a tenth of the discharges), per re-partition, mean over re-partitions, with the pooled interval on the remaining discharges; the number of discharges that are test discharges in two, three or four of the four file-level splits.

**Part C — dose of a beam interruption (all 641 files; definitions of B.58).** Events with one, two and three absent frames, six contiguous rows before. First-row ratio to the median of the six rows before, and the mean ratio of rows 1 to 3, for T_i, the ECEI level and the BES level, with 2,000 resamples of discharges; the same for the recurrent estimator and its twin on the events in test discharges of the pooled folds is not part of this batch.

Checks before any statement: part A reproduces the pooled skills of B.39 and the percentiles of B.54; part C reproduces the first-row ratios of B.58 for events of one absent frame.

## 4. Statements

- **A1.** Huber skill of the estimator over the causal GP: interval above zero. **A2.** Over the twin: interval above zero.
- **A3.** Trimmed skill of the estimator over the causal GP: interval above zero. **A4.** Over the twin: interval above zero.
- **A5.** Huber skill of KFX over KF0 (r3 to r9): interval above zero.
- **B1.** With the 64 discharges with the largest reduction removed, the estimator has skill over the causal GP: interval above zero. **B2.** Over the twin: interval above zero.
- **C1.** The first-row deficit of T_i (one minus the ratio) is larger after two absent frames than after one: the interval of the difference is above zero. **C2.** The same for the ECEI level.

Reported: every absolute error with its interval; the per cent reductions; the robust skills against hold and the offline GP; parts B and C in full; the three-frame events.

## 5. What goes into the paper

- The absolute errors go beside the skills in the main table and one sentence of the abstract gives the size in eV or per cent.
- A1 to A4 hold: the paper states that the margins do not rest on the 1 % of rows with the largest errors, with the robust skills as a co-primary readout. Any fails: stated in the limits ("the margin over ... is carried by the rows with the largest errors"), and the mean-squared-error skill is described as a score of the tail.
- B1, B2 hold: the per-discharge sentence gives the skill without a tenth of the discharges. Either fails: the paper says how many discharges carry the margin.
- C1 and C2 hold: the discussion of the beam interruption gives the dose–response. C1 fails: the paper says that the fall does not grow with the length of the interruption in the events available, and that reading of the fall as a response to the interruption is not supported by a dose.

## 6. Expectations

Root-mean-square error: estimator 180 to 190 eV, causal GP 4 to 5 % higher, twin 2 to 3 % higher. Mean absolute error: estimator within 1 % of the causal GP. A1 +0.01 to +0.04, holds; A2 +0.005 to +0.03, holds or fails by a small margin; A3 +0.02 to +0.07, holds; A4 +0.01 to +0.05, holds; A5 +0.02 to +0.05, holds. B1 +0.03 to +0.06, holds; B2 +0.01 to +0.04, holds. C1: uncertain (the two-frame events seen had a smaller fall, 0.926 against 0.814), expected to fail; C2 uncertain.

## 7. Limits known in advance

The Huber threshold and the trimming share are choices; both are fixed here before the readout. Removing discharges by their reduction selects on the outcome and gives a lower bound. Events of two and three absent frames are few (tens) and may differ from single-frame events in more than their length.
