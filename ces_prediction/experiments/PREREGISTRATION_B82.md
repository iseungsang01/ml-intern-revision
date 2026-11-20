# B.82 — where the fast-channel increment arises, how it reads in robust error measures, whether it is specific to the co-located channels, and how much it moves with the fit

Registered 2026-11-20, before any run of this batch. CPU only (stored estimates of B.75, B.79 and B.81, and boosted regressions of the kind of B.75; no network, no GPU). The session lead designed it. An AI agent implements and runs it under the author's supervision. No existing file is edited.

## 1. Why

An independent referee report on the rewritten manuscript (round 16, 2026-11-20; the grader had the two PDFs only) ranks four objections that existing data can answer:

1. **The gain is not localized.** After a beam notch the reported T_i is at 0.76 to 0.84 of its level and BES at 0.64 to 0.67 (`docs/paper/b75_last_block.md`); under the schedule rule every scored row follows a notch that the history has not seen. An estimator with BES input could gain by tracking that dip, which may be a beam-dependent response of the measurement and not a change of the plasma. The beam scalars (B.79) remove part of this, not all. Wanted: the increment by time since the last notch and by the size of the change of the fast level since the anchor row.
2. **Squared error on a heavy-tailed quantity.** A few rows dominate every MSE. Wanted: mean and median absolute error, relative error, error quantiles, the shares of rows within 10 % and 20 % of the level, and the distribution of the gain over discharges.
3. **The specificity of "same radius" is untested.** The only competing inputs so far are the beam waveforms. Wanted with the existing files: a diagnostic that is not at the radius (the two Mirnov signals) given the same treatment as the fast inputs; the BES radial columns separately; fast inputs taken from another discharge at the same time in the discharge.
4. **Intervals are conditional on one fit.** A refit with inputs equal up to rounding moves a single-unit increment of the last block by 0.02 (`b75_lead_checks.md`). Wanted: the spread of the increment over seeds of the regression.

**Status.** Exploratory on used data. Known when this was written: every readout of B.73 to B.81; the unregistered check of the session lead that under D650 (one re-partition of V1) the increment on rows more than 100 ms after a notch is +0.320 and that the time since the last notch alone adds +0.023 to the history (`b75_lead_checks.md`); that with the previous frame visible (D10) the Mirnov inputs were not needed by the recurrent network (earlier batches); that earlier gains of the network were in the upper tail of the error and not in its median. Not known: any stratum by time since a notch with an interval, any robust error measure of T0 and TX under a stale history, any fit with Mirnov or swapped inputs under D650 or S, any seed spread.

## 2. Data, rows, target, units, intervals

Exactly those of B.75: D1 rows (`data/.b73/rows.npz`), inputs `data/.b75/inputs/{D10,D650,S}.npz`, the common set (D10, D650) and the S set (S), target L5, units V1 (r2 to r4 x 5 folds), V2 (three assignments x 5 folds), V3 (fifths two to five), V4a (fit to 31991), V4b (fit to 32310). Stored estimates of T0, TX, TE, TB (B.75, B.81), TN, TXN (B.79). Intervals: 95 %, 20,000 draws, shot clusters resampled (re-partition or assignment drawn first under V1 and V2), seed stream `[20261182, crc32(<statement id or cell name>)]`; the discharge form beside it; verdicts on the cluster interval. Skill of A over R = 1 - MSE(A)/MSE(R) on the same rows; a ratio of two error measures is A over R.

## 3. Part A — strata on the time structure and on inputs (stored estimates, no fit)

For a scored row t, tau = stamp of row t minus the stamp of the first row after the most recent absent step (a time step of 15 to 45 ms inside a segment, as in B.75) that ends at or before t, in ms; rows with no absent step earlier in their segment form the class "no notch before" (they exist under D650 only).

- **Classes of tau:** up to 100 ms; above 100 and up to 300 ms; above 300 ms; no notch before.
- **Classes of the change since the anchor row:** c = |a2(BES)| + |a2(ECEI)|, the sum of the absolute changes of the two standardized array means since the anchor row (the anchor scalars a2 of B.75, computed per unit as there). Thirds of c over the scored rows of the rule and unit (the cut points come from the unit's fit rows). Rows where c is not finite are reported as a fourth class.

| id | statement | rule of judgement | expectation |
|---|---|---|---|
| A1 | V1, D650, rows with tau above 300 ms: TX over T0 lies above zero | lower end above zero | holds |
| A2 | V1, S, rows with tau above 300 ms: TX over T0 lies above zero | lower end above zero | holds |
| A3 | V3, D650, rows with tau above 300 ms: TX over T0 lies above zero | lower end above zero | uncertain, leaning to holds |
| A4 | V1, D650, rows with no notch before: TX over T0 lies above zero | lower end above zero | holds |
| A5 | V1, S: TX over T0 on rows with tau up to 100 ms minus that on rows with tau above 300 ms lies above zero (the gain is larger just after a notch) | lower end of the difference above zero | uncertain, leaning to holds |
| A6 | V1, D650, lowest third of c: TX over T0 lies above zero | lower end above zero | uncertain |
| A7 | V1, D650: TX over T0 in the highest third of c minus that in the lowest third lies above zero | lower end of the difference above zero | holds |
| A8 | V1, S, rows with tau above 300 ms: TXN over TN lies above zero (beam scalars given) | lower end above zero | holds |

Reported without a statement: every class, both rules, every design, for TX over T0, TE over T0, TB over T0 and TXN over TN, with RMSE and counts; the same for the last block of shots under V1.

## 4. Part B — robust error measures (stored estimates, no fit)

For H, G, T0, TX and O, under D10, D650 (common set) and S (S set), every design: mean absolute error (MAE), median absolute error, the 75th and 90th percentile of the absolute error, the median of |estimate - L5| / L5 over rows with L5 above 100 eV, the shares of rows with |estimate - L5| / L5 at most 0.10 and at most 0.20, and per discharge (those with at least 50 scored rows) the skill of TX over T0. Ratios are TX over T0; a difference of shares is TX minus T0.

| id | statement | rule of judgement | expectation |
|---|---|---|---|
| B1 | V1, D650: MAE(TX) / MAE(T0) lies below one | upper end below one | holds |
| B2 | V1, D650: the ratio of the median absolute errors lies below one | upper end below one | uncertain |
| B3 | V3, D650: MAE ratio below one | upper end below one | uncertain, leaning to holds |
| B4 | V1, S: MAE ratio below one | upper end below one | holds |
| B5 | V1, D650: share of rows within 10 % of the level, TX minus T0, lies above zero | lower end above zero | uncertain, leaning to holds |
| B6 | V1, D650: the share of discharges whose own skill of TX over T0 is positive lies above one half | lower end above 0.5 | holds |
| B7 | V4a, D650: MAE ratio below one | upper end below one | uncertain |
| B8 | V1, D650: the ratio of the 90th percentiles of the absolute error lies below one | upper end below one | holds |

Reported without a statement: the whole table of measures for every estimator, rule and design; the quartiles over discharges of the per-discharge skill; MAST is not in this batch.

## 5. Part C — specificity (new fits; rules D650 and S; units of V1, V3 and V4a; target L5)

Same regression and target transform as B.75 (`HistGradientBoostingRegressor(random_state=20261116)`, default settings, fitted to L5 minus G).

- **TM (Mirnov in place of the fast diagnostics).** T0's ten history scalars plus seven scalars built from the mean of the two Mirnov columns of the shot files (`MC*`), each column standardized with the fit discharges' mean and standard deviation exactly as the BES and ECEI channels are: the four scalars of the present (level, departure from the mean of the ten rows before, their standard deviation, one-row increment) and the three anchor scalars (a1, a2, a3). The Mirnov columns are raw snapshots at the stamp like the fast channels (`docs/paper/file_rules.md`).
- **TXM.** TX's inputs plus those seven Mirnov scalars.
- **TXs (fast inputs of another discharge at the same time).** TX's inputs with the fourteen fast and anchor columns of each row replaced by those of the row with the same frame index (10 ms frames counted from the start of the segment, as in B.78) in a donor discharge: for every discharge one donor is drawn at random among the other discharges of the same side (fit or test) of the unit whose segment has that many frames; rows without a donor row keep missing values. Stream `[20261182, crc32("swap|<unit>|<rule>")]`. The ten history columns stay in place. (The permuted control of B.81 destroys the time in the discharge as well; this one keeps it.)
- **TBi, TBm, TBo (the three radial columns of BES separately).** T0's inputs plus the BES half of the fast and anchor scalars (seven scalars) computed from the three channels of one radial column only (the channel names give the column: the last two digits). The agent states in the addendum which channels form which column and which is innermost by the name.

| id | statement | rule of judgement | expectation |
|---|---|---|---|
| C1 | V1, D650: TM over T0 has an interval that contains zero or lies below it (a diagnostic that is not at the radius adds nothing) | lower end at most zero | uncertain |
| C2 | V1, D650: TX over TM lies above zero | lower end above zero | holds |
| C3 | V3, D650: TX over TM lies above zero | lower end above zero | uncertain, leaning to holds |
| C4 | V1, D650: TXs over T0 has an interval that contains zero or lies below it | lower end at most zero | holds |
| C5 | V1, S: TXs over T0 has an interval that contains zero or lies below it | lower end at most zero | holds |
| C6 | V1, D650: TXM over TX has an interval that contains zero or lies below it (Mirnov adds nothing on top of the fast diagnostics) | lower end at most zero | uncertain, leaning to holds |
| C7 | V1, S: TX over TM lies above zero | lower end above zero | holds |

Reported without a statement: TM, TXM, TXs, TBi, TBm, TBo over T0 for V1, V3, V4a under both rules, with intervals and RMSE.

## 6. Part D — spread over the seed of the regression (new fits)

T0 and TX refitted with `random_state` 1 to 10 (everything else as B.75) for V4a and V4b under D650 and S, and for the five units of re-partition r2 of V1 under D650 and S. Reported: for each cell the registered increment (seed 20261116), and over the ten seeds the smallest, the median and the largest increment and their standard deviation; for V1 r2 the pooled increment per seed. No statement is judged; the numbers go beside the intervals.

## 7. Part E — description of the data set (file facts, no estimator)

For the manuscript's table of the data set, over the D1 discharges: number of discharges and grid rows; per discharge the number of grid rows (10th, 50th, 90th percentile) and of absent steps; over the D1 rows the 10th, 50th and 90th percentile of the reported T_i and of L5; the share of grid rows without a usable T_i; from the beam files the 10th, 50th and 90th percentile of the total beam power over grid rows with any source on (the 13 files with invalid NBI-2 columns left out of this line) and the share of rows with one, two, three, four and five sources on. The plasma current, field and density are not in the files and are not reported.

## 8. Checks before the readout (each run once; a failed check stops the batch and is reported)

- **Reuse.** T0 and TX of unit `V1_r2_k0`, rule D650, refitted with this batch's code and seed 20261116 equal the stored estimates of B.75 bit for bit.
- **tau.** For 500 random scored rows under S, tau recomputed from the row's shot file alone (time column only) equals the batch's tau exactly; under S no scored row is in the class "no notch before".
- **Classes.** In every stratum table the class counts sum to the rows of the set, and pooling the classes reproduces the stored increment of B.75 to the last printed digit.
- **Swap.** In TXs the ten history columns are bit-identical to TX's; no row carries the fast scalars of its own discharge; the donor of a test discharge is a test discharge.
- **Mirnov.** The Mirnov scalars of 200 random rows recomputed from the shot file alone equal the batch's bit for bit.

Choices that this text leaves open are written into `PREREGISTRATION_B82_ADDENDUM1.md` before the checks and before any readout. This file is not edited after registration.

## 9. What goes into the paper (decided now)

- **Where the gain arises (A).** If A1 and A2 hold, the manuscript states that the increment is not confined to the rows after a notch and gives the table by tau. If either does not hold, the manuscript states that the increment is established only within 300 ms of a notch under that rule, that it may reflect the response of the measurement to the beam and not a change of the plasma, and the abstract carries that restriction. A5 and A7 are reported either way; if A6 does not hold the text says that the increment needs a change of the fast level since the anchor row.
- **Robust measures (B).** A table of the robust measures enters the main text beside the RMSE table whatever the outcome. If B2 does not hold while B1 and B8 hold, the manuscript says that the reduction is in the larger errors and that the median error is not established to be lower. The sentence comparing the error with the 10 % of the requirement documents is written from the shares within 10 % and 20 %, whatever they are.
- **Specificity (C).** If C1 holds and C2 holds, the manuscript may say that a magnetic diagnostic that is not at the radius, treated the same way, gives no increment, with the caveat that its samples are snapshots of a fast signal. If C1 does not hold, the manuscript says that the Mirnov signals carry an increment as well, gives its size beside that of the fast diagnostics, and no sentence claims that the increment is specific to the radius; the word "co-located" then names where the channels are and nothing more. If C4 or C5 does not hold (lower end above zero), the size of the reduction with swapped inputs is printed beside every increment of that rule and the increments are read net of it.
- **Seeds (D).** The spread is printed beside the intervals of the chronological design; if for a cell the sign of the increment changes among the ten seeds, the manuscript says so for that cell.
- **Data set (E).** The table enters the data section.
- No estimator of the paper changes because of this batch.

## 10. Output

Code in `ces_prediction/experiments/b82_localize/`; caches in `data/.b82/` (git-ignored); `docs/paper/b82_localize.md` and `.json` (statements with value, cluster interval, discharge interval, verdict, expectation; every reported table; counts), `docs/paper/b82_checks.md`.
