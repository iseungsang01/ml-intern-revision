# B.50 — the skill in scored rows that resemble the rows without a usable value

Registered 2026-11-18, before any of the readouts below was computed. CPU only; no network is trained. Designed by the session lead and executed by a delegated agent from this document.

## 1. Why

The paper is motivated by the frames in which CES gives no usable T_i (7.7 % of the grid rows are missing, and 0.53 % exceed 3 keV). Every skill of the paper is scored on rows where a value was measured and withheld from the estimator. The rows without a usable value are not a random subset: the missing rate is 18.0 % in the lowest fifth of the beam-emission level against 3.4 % in the highest. If the margins of the estimator sit in rows that do not resemble the unusable ones, the estimator is weakest where it is needed. This readout asks whether the two margins (over the causal GP, over the history-only twin) hold in the scored rows that resemble the unusable rows.

## 2. Seen before

B.37 F: whether T_i is unusable in a row is predicted from its history flags and the fast-channel window with an area under the ROC curve of 0.961 to 0.974 on the four splits. B.45: in the scored rows within 30 ms after a row of low beam emission (0.010 to 0.015 of the rows) no network has a lower error than the twin. B.33: in the cluster of rows with low beam emission the increment is −0.016 to +0.049. B.48: in the rows that value rules flag the increment is +0.022, +0.015, +0.081, +0.038. The log holds an earlier readout reweighted towards missing rows for an earlier model against PCHIP and hold; it is not used. Not seen: any skill of the estimator of B.39 as a function of how much a scored row resembles an unusable one.

## 3. Design

T_i, cut population, the four file-level splits, test discharges.

Estimators: the row table of B.48 at k = 1 (`data/.b48/rows_cut_s{seed}.npz`): `causal` (the causal variant of B.39, initialisation = split seed), `hist80` (the history-only twin), `hold`, `gp` (the causal GP), `y` (the reported value). Scored rows: the k = 1 rows in which all of these are finite.

Propensity p: the full model of B.37 F, unchanged (`b37_plan/run_b37.py::part_f`: `HistGradientBoostingClassifier`, the same inputs and seed; target "T_i is not usable in the row", fitted on the files that are not test files of the split), evaluated at every grid row of the test files that has a ten-row window, and joined to the scored rows by file and row.

Checks before the readout (the run stops if one fails): (a) the area under the ROC curve of p on the test files equals that of B.37 F for the split (`docs/paper/b37_plan.json`) within 0.005; (b) at least 98 % of the scored rows receive a p; the rows that do not are dropped and counted.

Strata: the fifths of p among the scored rows of the split. A further stratum: the scored rows whose p is above the median p of the unusable rows of the test files ("rows that look like a typical unusable row").

Weighted population: each scored row has weight w = p / (1 − p), clipped at the 99th percentile of w in the split; the weighted skill is 1 − Σ w·se_A / Σ w·se_R. The effective number of rows (Σw)² / Σw² is reported.

Intervals: 95 %, 2,000 resamples of the test discharges; the strata and the weights are those of the full test set and are not recomputed in a resample.

## 4. Statements

Four splits; a statement holds if it holds on at least three.

- **P1.** In the top fifth of p the estimator has skill over the causal GP with the interval above zero.
- **P2.** In the top fifth of p the estimator has skill over the twin with the interval above zero.
- **P3.** With the rows weighted by the odds of being unusable, the skill over the causal GP has an interval above zero.
- **P4.** The same over the twin.
- Reported: both skills and the skill over hold in every fifth and in the stratum above the median p of the unusable rows, with the row share of that stratum; the share of the unusable test rows whose p falls into each fifth of the scored rows; the effective number of rows; per fifth the median age of the last visible value, the median T_i, and the root-mean-square error of the four estimators.

## 5. What goes into the paper

- P1 to P4 hold: one sentence in the section on where the gain is, and the motivation is stated without the reservation.
- P1 and P3 hold, P2 or P4 does not: the paper states that in rows resembling the unusable ones the estimator keeps its margin over the causal GP while the fast-channel increment is not established there. The measurement that would decide it is named: values for the unusable frames themselves, from a CES re-analysis with relaxed acceptance, which the files do not hold; within the files, a replay that hides measured values in runs placed by p.
- P1 or P3 does not hold: stated as a limit of the claim in the abstract-level summary of the limits.

## 6. Expectations

P1 holds, with +0.03 to +0.15 in the top fifth. P2 is uncertain: the increment in the top fifth is −0.02 to +0.08 and its interval is above zero on one to three splits. The odds-weighted skill over the causal GP is +0.03 to +0.15 (P3 holds) and over the twin 0.00 to +0.08 (P4 uncertain). The top fifth of the scored rows holds 0.70 to 0.95 of the unusable test rows. The effective number of rows is 3 to 15 % of the scored rows.

## 7. Limits known in advance

Resemblance is measured in the inputs of the propensity model only. The unusable rows may differ from the scored rows in ways that no input records (a larger fit error, for one), and nothing in the files can test that. The weights concentrate on few rows. One initialisation. The test rows are those of every earlier readout.
