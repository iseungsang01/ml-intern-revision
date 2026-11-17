# B.41 — where the gain sits: rows that pass the value rules against rows that do not, the distribution of the absolute error, and averages conditioned on beam blips and edge bursts

Registered 2026-11-18, before any of the readouts below was computed. CPU only; nothing is trained. Designed by the session lead and executed by a delegated agent from this document.

## 1. Why

Two items of the tenth independent grading that need no training. (i) The mean-squared-error gain is carried by few rows, and a referee cannot tell whether those rows are changes of the plasma or excursions of the CES fit: divide the gain between rows that pass plain value rules and rows that do not, and show the distribution of the absolute error, not one skill. (ii) A reader of plasma physics has no figure that shows in which events the estimator gains: averages conditioned on the time of a beam blip and of an edge burst, for the measured T_i, the ECEI level, the BES level and the estimates.

## 2. Seen before

The 1 % of rows with the largest squared error hold 48 % of the total (cut). `docs/paper/b30_sensitivity.md`: with four exclusions (scored value or last visible sample below 150 eV or above 2.5 keV; a jump of more than 1 keV between them; within 30 ms after a row whose BES level is below half the discharge median) 0.885 to 0.898 of the rows are kept and the fast-channel increment is +0.084 to +0.145 for the whole-discharge variant and +0.044 to +0.152 for the causal variant. The blip response (measured T_i 0.819, ECEI 0.926, return over about 90 ms; B.35: 41.0 and 40.2 ms). B.32: in the first row after a blip the causal variant predicts most of the measured fall and the twin none. B.31: skill of the network over its twin +0.197 in burst-active rows and +0.135 in quiet rows (six test discharges), T_i missing more often after a burst. Not seen: the share of the squared-error reduction in each class of rows; quantiles of the absolute error; any estimate aligned to an event over several rows; T_i, ECEI and BES aligned to burst onsets.

## 3. Data

Frozen B.30 row tables `data/.b30/rows_cut_s{42,1,7,123}.npz`, T_i, cut population, natural cadence (k = 1), rows with `hold`, `gp`, `hist80` (history-only twin), `causal` (causal variant) and the measurement finite. Intervals: 95 %, 2,000 resamples of test discharges; a statement holds if it holds on at least three of the four splits. Bursts: the B.31 detector unchanged (`b31_bursts/run_b31.py::bursts_and_blips`), the twelve discharges with a raw stream.

## 4. Statements

**Rows by the value rules.** A row is *plain* if it passes all four exclusions of the sensitivity analysis and *flagged* otherwise.

- **G1.** The plain rows hold more than half of the total squared-error reduction of the causal variant against the twin (sum over rows of twin error squared minus causal error squared): the interval of their share lies above 0.5.
- **G2.** The same against the causal GP.
- Reported: the share of rows in each class; the skill of the causal variant over the twin, the GP and hold inside each class; the same shares for the 1 % of rows with the largest hold error against the rest.

**Distribution of the absolute error.**

- **Q1.** The 90th percentile of the absolute error of the causal variant over that of the causal GP, and over that of hold, has an interval below 1 in both cases.
- Reported: the 50th, 75th, 90th, 95th and 99th percentiles of the absolute error of hold, the causal GP, the twin and the causal variant, and the ratio of the causal variant to each; the mean signed error (estimate minus measurement) of each by third of the last visible sample; the slope of a least-squares line of the measurement on each estimate.

**Events.**

- **U1 (bursts, measurement only, twelve discharges).** Events: burst onsets with no other onset in the 50 ms before and not within 100 ms after an absent frame, with at least two measured T_i rows whose stamps lie 50 to 10 ms before the onset and at least one 0 to 30 ms after it. Per event, the ratio of the mean T_i in the rows 0 to 30 ms after to the mean in the rows 50 to 10 ms before. The median ratio has an interval below 1 (2,000 resamples of discharges). Not testable with fewer than 30 events.
- Reported: the same for the ECEI level and the BES level; the traces by 10 ms bin from −50 to +100 ms; on the raw-stream discharges that are test discharges, the mean estimate of the causal variant and of the twin in the same bins and their squared errors in the rows 0 to 30 ms after an onset.
- Reported (blips, test discharges of the four splits): for the rows 1 to 15 after an absent frame, the mean of the measured T_i, of the causal variant, of the twin, of hold and of the GP, each over the level before (median of the measured values in the six rows before), with intervals; the skill of the causal variant over the twin in rows 1 to 3 and in rows 4 to 10.

A figure for the manuscript is produced from these traces (measurement, ECEI, BES and the two networks against the time since a blip and since a burst onset), with its data file.

## 5. What goes into the paper

- G1 holds: the paper leads with the plain rows and says that most of the fast-channel gain is on rows that pass the value rules. G1 fails: the paper says that the gain sits mostly on rows the value rules flag, and the claim is bounded to them.
- Q1 holds: the gain is described at the 90th percentile as well as in the mean. Fails: stated.
- U1 holds: the fall of T_i after a burst is given as a number with its interval. Fails or not testable: stated.

## 6. Expectations

Plain rows 0.88 to 0.90 of the rows; their share of the reduction against the twin 0.30 to 0.65, G1 holding on at most two splits (expected to fail); against the GP 0.25 to 0.60. Q1: ratios 0.90 to 0.99, holding against the GP on three or four splits and against hold on four. U1: 40 to 200 events, ratio 0.75 to 0.95, holds.

## 7. Limits known in advance

The value rules are plain thresholds, not a fit-quality flag. The twelve raw-stream discharges were chosen for other reasons; six or seven are test discharges. Bursts are detected on beam emission. The estimators are frozen and the test rows of these splits have been read by earlier registered readouts.
