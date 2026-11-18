# B.52 — readouts asked for by the eleventh independent grading: discharges, a linear filter, gaps, value thresholds

Registered 2026-11-18, before any of the readouts below was computed. CPU only; no network is trained (part L fits linear filters by least squares). Designed by the session lead and executed by a delegated agent from this document.

## 1. Why

The eleventh independent grading of the manuscript named readouts that the stored row-level errors can give and that the text lacks: (i) the manuscript reports skills pooled over rows and never says in what share of the discharges the estimator has the lower error, although its median error is not lowered; (ii) no baseline whose form is fixed on the training discharges reads the fast channels, so it is not shown what the recurrent network adds to a linear filter; (iii) the gap-length readout exists only for an earlier network; (iv) the sensitivity of the skills to the value threshold exists only for an earlier network; (v) the 90th percentile of the absolute error is given as a ratio and not in eV.

## 2. Seen before

B.39 and B.48: skills pooled over rows; the median absolute error of the estimator is not below that of the causal GP; on the two splits whose interval includes zero three discharges carry 0.79 and 0.82 of the excess squared error of the weakest initialisation. Seen while this document was written (`docs/paper/b48_reread.json`): the 90th percentile of the absolute error is 181.2, 226.7, 204.7 and 223.7 eV for the estimator and 193.5, 246.3, 219.8 and 241.8 eV for the causal GP. The supplement holds, for an earlier network, a readout by gap length (beyond 50 ms the skill over the causal GP has an interval that includes zero) and a 20-parameter filter whose form was read from the network on test rows. Not seen: any quantity per discharge, any linear filter fitted on training discharges, any readout by gap length or by value threshold for the estimator of B.39.

## 3. Design

T_i, cut population, the four file-level splits, test discharges. Row table of B.48 at k = 1 (`data/.b48/rows_cut_s{seed}.npz`): `causal` (the estimator of B.39, initialisation = split seed), `hist80` (the twin), `hold`, `gp` (the causal GP), `y`, `age_ms`. The average of three initialisations E and of the three twins E_h: the cached estimates of B.49 (`data/.b49/s{seed}.npz`), joined by discharge and row; the join is checked (the reported value agrees on every joined row; the skill of E over the causal GP on the joined rows equals that of B.49 to three decimals). Intervals: 95 %, 2,000 resamples of the test discharges.

**Part D — per discharge.** Test discharges with at least 50 scored rows. For each discharge the mean squared error and the median absolute error of every estimator.

**Part L — a linear filter fitted on the training discharges.** Rows: the target is measured, three earlier visible values exist in the segment, and the ten rows before exist in the segment. History inputs (7): the last three visible values, their ages, the time step. Fast inputs (30): the fifteen fast channels at the row and, for each, its departure from the mean of the ten rows before. All inputs and the target are standardized with the training discharges of the split. Model LH reads the history inputs, model LF the history and the fast inputs; both are linear with an intercept, fitted by least squares with a ridge penalty chosen from {0, 0.001, 0.01, 0.1, 1, 10} by the mean squared error on the validation discharges. Scored on the rows of the B.48 table that satisfy the same condition (the join is checked by the reported value).

**Part G — gap length.** The scored rows by the age of the last visible value: 10 ms; 20 to 50 ms; above 50 ms.

**Part V — value threshold.** The scored rows whose reported value and whose last visible value are both at or below T, for T = 2,000 and 2,500 eV. The estimator is the one trained with the 3 keV rule; nothing is retrained.

**Part P — absolute errors.** The 50th, 75th, 90th, 95th and 99th percentile of the absolute error in eV for the estimator, E, the twin, the causal GP and hold.

## 4. Statements

Four splits; a statement holds if it holds on at least three.

- **D1.** The share of the test discharges in which the estimator has a lower mean squared error than the causal GP has an interval above 0.5.
- **D2.** The same against the twin.
- **L1.** The estimator has skill over the linear filter LF with the interval above zero.
- **L2.** LF has skill over LH with the interval above zero.
- **G1.** In the rows whose last visible value is 20 ms old or older the estimator has skill over the causal GP with the interval above zero.
- **V1.** With T = 2,000 eV the estimator has skill over the causal GP with the interval above zero. **V2.** The same over the twin.
- Reported: D1 and D2 for E; the shares by the median absolute error; the quartiles over discharges of the skill per discharge; the share of the squared-error reduction against the causal GP that the ten discharges with the largest reduction hold; LH and LF over the causal GP and over hold, the twin over LH, E over LF, the chosen penalties; every skill of part G per bin with the row shares, and the skill over the twin; part V at both thresholds with the row shares; part P.

## 5. What goes into the paper

- Part D is stated in the main result whatever it shows. If D1 does not hold, the sentence reads that the mean squared error is lower in that share of the discharges and that the gain of the mean comes from a minority of discharges, and the limit list says so.
- Part L: the two filters enter the main table as rows. If L1 does not hold, the paper states that a linear filter with the fast channels is not separated from the recurrent estimator, and the estimator of the paper is described as one way of reading the fast channels, not as the needed one. If L2 holds it is quoted as support for the fast-channel increment from a model with no early stopping and no initialisation.
- Parts G, V and P go to the supplement, with one sentence each in the main text where a statement fails.

## 6. Expectations

D1: share 0.50 to 0.65, interval above 0.5 on one to three splits; D2 the same. By the median absolute error: 0.35 to 0.55. The ten discharges with the largest reduction hold 0.5 to 1.2 of the total reduction against the causal GP. L2 holds, +0.01 to +0.05. L1 holds, +0.03 to +0.10. LH over the causal GP: −0.05 to +0.03. G1: the rows of 20 ms or more are 0.02 to 0.04 of the scored rows; skill there +0.03 to +0.25, interval above zero on two to four splits. V1 and V2 hold, each skill within 0.03 of its value on all rows.

## 7. Limits known in advance

A discharge is one unit whatever its length. The linear filter is one choice of inputs; another choice could do better. The value threshold is applied to the scored rows and to the last visible value only; the estimator was trained with values up to 3 keV in its inputs, and a retraining with another rule needs the GPU. One initialisation for parts G and V. The test rows are those of every earlier readout.
