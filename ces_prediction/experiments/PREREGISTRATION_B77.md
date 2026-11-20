# B.77 — the frame's own reading, corrected and flagged with the fast samples of the same frame

Registered 2026-11-20, before any run of this batch. CPU only, boosted regressions and classifiers on cached rows; no network. The session lead designed it; the author asked that every candidate proposal be tried ("candidate A"). An AI agent implements and runs it under the author's supervision. No existing file is edited.

## 1. Why

The motivation of the work is that the CES value at the pedestal top is inaccurate (author, 2026-11-20). The critiques of 2026-11-20 and B.73/B.74 found that the frame-specific departure of the reported value co-varies with the beam-emission light of the same exposure, and that this co-variation is not information about the level. Candidate A turns that finding into a method: **if the fast samples announce the departure of a frame's reading, they can be used to correct that reading and to flag it.**

Every estimator so far estimated the value of frame t without reading it. This batch is about another product: the value of frame t is read, and the question is how close to the level a corrected value comes, and what the fast samples add to the correction. The novelty critique (`docs/paper/review_loop/critique_20261120_novelty.md`) judges the phenomenon itself (edge bursts disturb CXRS fits) known; what it did not find is a fast diagnostic of the same beam used as a per-frame monitor of the CES value.

**Status.** Exploratory on used data. The rows, the level target, the designs and the fast inputs are those of B.73 and B.75, whose results are known. The estimators that read the frame's own value are new and fixed here before any of their numbers has been read.

## 2. Data, target, designs

- Rows: the D1 rows complete under rule D10 of B.75 (`data/.b75/sets.npz`, `complete_D10`; history scalars, causal GP and fourteen fast scalars finite) with the target defined. Inputs from `data/.b75/inputs/D10.npz` and B.75's code for the fast and anchor scalars.
- Target: L5, the following level (median of the measured values among rows t+1 to t+5, at least three). It contains no value of row t.
- Designs: the units of B.73 (V1 random, V2 cluster-blocked, V3 rolling origin, V4 chronological with fits to shot 31991 and to shot 32310).

## 3. Estimators (all scored against L5)

| Name | Reads the value of frame t | Definition |
|---|---|---|
| O | yes | the reported value y of frame t, as it is |
| MED3 | yes | the median of y and the two newest earlier measured values |
| T0, TX | no | B.75's estimators under D10, refitted on these rows (history; history plus fourteen fast scalars) |
| C0 | yes | G plus a boosted regression of (L5 − G) on the ten history scalars and four scalars of the frame's value: y − G, y − H, y − (mean of the last ten values), \|y − G\| |
| CX | yes | C0's inputs plus the fourteen fast scalars (eight of row t, six anchor scalars) |
| CB, CE | yes | C0's inputs plus the BES half (seven scalars) or the ECEI half |

Boosted regression: `HistGradientBoostingRegressor(random_state=20261116)`, default settings.

## 4. Flags

Label of a row: the reading departs from the following level, y / L5 above 1.35 or below 0.74 (the thresholds of B.74).

| Classifier | Inputs |
|---|---|
| PH | the ten history scalars (no value of frame t: available before the frame is analysed) |
| PHX | PH's inputs plus the fourteen fast scalars |
| FH | the ten history scalars and the four scalars of the frame's value |
| FHX | FH's inputs plus the fourteen fast scalars |

`HistGradientBoostingClassifier(random_state=20261116)`, default settings. Area under the ROC curve per design.

**Screening.** In each test unit the rows are ranked by the classifier's probability; the q = 5, 10, 20 % highest are dropped; the MSE of O against L5 on the kept rows is reported for FHX, FH, PHX, PH and for no screening.

## 5. Readouts

Per design: RMSE of every estimator; skill of each over O; C0 over T0; CX, CB, CE over C0; TX over T0 on the same rows; the areas under the curve and their differences; the screening table; under V1 the increments by chronological block and by the four classes of the BES sample relative to its recent median (r_B below 0.9, 0.9–0.97, 0.97–1.03, above 1.03; B.73 Addendum 1), a stratum on an input.

**Controls (V1):** the fourteen fast scalars permuted over rows (within fit rows and within test rows) and CX refitted; the fast scalars of row t − 1 in place of those of row t.

**Intervals.** 95 %, 20,000 draws, shot clusters resampled (the discharge form beside it), as B.73. A statement is judged on the cluster interval.

## 6. Statements

| Id | Statement | Expectation |
|---|---|---|
| A1 | V1: the MSE of C0 lies below that of O (skill above zero) | holds (+0.5 to +0.7) |
| A2 | V1: the MSE of C0 lies below that of T0 (reading the frame helps) | holds (+0.05 to +0.30) |
| A3 | V1: CX over C0 lies above zero | holds (+0.03 to +0.15) |
| A4 | V2: CX over C0 lies above zero | holds |
| A5 | V3: CX over C0 lies above zero | uncertain |
| A6 | V4 (fit to 31991): CX over C0 lies above zero | uncertain |
| A7 | V1: CX over C0 exceeds TX over T0 (interval of the difference above zero) | uncertain, leaning to holds |
| A8 | V1 and V4: the area under the curve of FHX minus that of FH lies above zero under each | holds |
| A9 | V1, q = 10 %: the MSE of O on the rows kept by FHX lies below that on the rows kept by FH (interval of the ratio below 1) | uncertain |
| A10 | V1: CE over C0 lies above zero | uncertain |
| A11 | V1, placebo: the interval of CX over C0 contains zero or lies below it | holds |

Each statement is judged separately; the tables are reported whole.

## 7. What goes into the paper (decided now)

- If A3 and A4 hold: the paper proposes the correction of the frame's reading with the fast samples of the same frame as a product, with the sizes found; A5 and A6 decide whether the sentence names a forward-in-time design or says it is not established there.
- If A3 fails: the paper says that once the frame's value and the history are read, the fast samples add nothing to the correction, and candidate A is not proposed.
- If A8 holds: the flag is described as a use, with the screening table.
- A1 and A2 are stated as sizes; they are expected of any smoother and are not presented as a finding.
- If A11 fails, the batch is not used until the cause is found.

## 8. Limits known in advance

- Exploratory on used data.
- The correction is about scatter around the level built from reported values; a systematic error of the analysis is in the target too.
- The history is one frame old (rule D10): the product is a correction for a reader that delivers every frame, or for use after the discharge. After the discharge a two-sided smoother is available; it is not scored here because the following frames are the target.
- Whether the departure that the fast samples announce is the fit, a displacement or the temperature is not decided here.

## 9. Checks and files

- T0 and TX refitted on these rows reproduce B.75's D10 values on the rows both have (reported; the row sets differ, so equality is not required), and C0 with the four scalars of the frame's value removed equals T0 bit for bit.
- No input of C0 or CX contains a value of a row later than t (tested by changing those values for 200 rows and recomputing).
- Code: `ces_prediction/experiments/b77_correct/`. Caches: `data/.b77/`. Output: `docs/paper/b77_correct.{md,json}`, checks `docs/paper/b77_checks.md`.
- Seed 20261177, one stream per quantity. Implementation choices left open go into `PREREGISTRATION_B77_ADDENDUM1.md`, written before any readout. This file is not edited after the first run.
