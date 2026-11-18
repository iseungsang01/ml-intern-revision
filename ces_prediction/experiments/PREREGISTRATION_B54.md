# B.54 — the row-level readouts on the rows of the pooled cross-validation

Registered 2026-11-18, before any readout of this batch. CPU only, no training and no network is run: the stored row-level squared errors of the 80 pooled runs of B.39 (arm P) are read with features of the rows taken from the data files. Designed by the session lead, run by a delegated agent.

## 1. Why

The manuscript's primary readout is pooled (639 discharges, five folds, eight re-partitions) and holds (B.39: over the causal GP +0.089 [+0.064, +0.112], over the twin +0.048 [+0.022, +0.073]). Every readout of where the gain is (which rows, which discharges, which level, which part of the error distribution, rows that resemble the rows without a usable value) was made on four splits of 96 test discharges, and several of their statements failed there or held on two or three splits. A reader cannot tell whether those are properties of the estimator or of 96 discharges. The pooled rows score every discharge once per re-partition and answer that with one interval per statement.

## 2. Seen before

On the four splits (B.48, B.50, B.52; one initialisation): 90th percentile of the absolute error 0.920 to 0.937 of that of the causal GP (4/4), median ratio 0.982 to 1.154; the estimator has the lower squared error than the causal GP in 0.647 to 0.731 of the test discharges and the ten discharges with the largest reduction hold 0.470 to 0.748 of it; in rows not flagged by value rules the increment over the twin is +0.044 to +0.159 (4/4), and those rows hold 0.295 to 0.639 of the reduction against the causal GP (the statement "more than half" failed, 0/4) and 0.593 to 0.818 against the twin (1/4); thirds of the discharges by median T_i: neither margin present in every third (1/4, 2/4); last value 20 ms old or older +0.056 to +0.109 over the causal GP (3/4), older than 50 ms every interval includes zero; rows in the fifth most at risk of losing the value: both margins 4/4; weighted by the odds of that risk: 2/4 over the causal GP, 1/4 over the twin; rows 1 to 3 after an absent frame +0.195 to +0.260 over the twin (3/4). Pooled: only the skills of B.39 arm P and the median absolute errors (hold 45.4 eV, estimator 46.0 eV, twin 46.1 eV). Not seen: any of the readouts above on the pooled rows.

## 3. Design

Rows: the T_i rows of `comparison_errors_test.npz` of `data/.b39p_v2_cut_r{r}_f{k}` (the estimator) and `data/.b39p_v2h_cut_r{r}_f{k}` (the twin), r = 2..9, k = 0..4; the five folds of a repeat are concatenated. The squared errors of the estimator, the twin, hold, the causal GP and the offline GP are those stored by `eval_seq`. A row is located in its data file through the stored discharge index and row index, and its features (time, last visible value, age of that value, the BES channels) are read from the file.

Definitions are those of the registered split-level readouts, taken from their code and not re-derived: value-rule flags of B.42/B.48 (`b42_delay/run_b42.py`, as re-read by `b48_reread/run_b48.py`), thirds by the median T_i of a discharge (`b44_strata/run_b44.py`), rows after an absent frame (`b41_events/run_b41.py`), quiet rows and the other BES classes (`b45_ablation_rows/run_b45.py`), the risk of losing the value and the odds weights (`b50_missing_like/run_b50.py`: the full model of B.37 part F, here fitted on the training discharges of each fold and applied to its test fold), the per-discharge shares with at least 50 scored rows and the age of the last value (`b52_grading_readouts/run_b52.py`).

Intervals: the two-level bootstrap of the pooled readout (a repeat is drawn, then its discharges are resampled with replacement); 20,000 draws for skills, 2,000 for percentile ratios and shares. Thirds are formed within a repeat from all its discharges.

Checks before any statement is read (the readout stops if one fails): (a) P1 and P2 of B.39 are reproduced from the stored errors to three decimals; (b) in every repeat each of the 639 discharges appears in exactly one fold; (c) the estimator and the twin of a fold are scored on identical rows; (d) the value in the data file at the located row equals the stored reported value on every row (to 0.01 eV); (e) the flag, class, third, blip-row and risk code, applied to the rows of split 42 of B.48, reproduces the numbers recorded there (flagged share, increment in the rows not flagged, share of the quiet rows, increment in the top fifth by risk) to three decimals.

## 4. Statements

One interval per statement; a statement holds if its interval excludes the reference on the stated side.

- **A1.** The 90th percentile of the absolute error of the estimator over that of the causal GP: interval below 1.
- **B1.** Share of the discharges (at least 50 scored rows) in which the estimator has the lower mean squared error than the causal GP: interval above one half. **B2.** The same against the twin.
- **C1.** In the rows not flagged by the value rules, the estimator over the twin: interval above zero. **C2.** In those rows, over the causal GP: above zero. **C3.** Those rows hold more than half of the squared-error reduction against the causal GP: interval above one half. **C4.** The same against the twin.
- **D1.** The estimator over the causal GP has an interval above zero in each third of the discharges by median T_i. **D2.** The same over the twin.
- **E1.** In rows whose last value is 20 ms old or older, the estimator over the causal GP: above zero. **E2.** In rows whose last value is older than 50 ms: above zero.
- **F1.** In the fifth of the rows most at risk of losing the value, the estimator over the causal GP: above zero. **F2.** Over the twin. **F3.** With the rows weighted by the odds of that risk, over the causal GP: above zero. **F4.** Over the twin.
- **G1.** In rows 1 to 3 after an absent frame, the estimator over the twin: above zero.
- **H1.** In the quiet rows (no absent frame, low beam emission or large departure of a BES channel shortly before), the estimator over the twin: above zero. **H2.** The quiet rows hold more than half of the squared-error reduction against the twin: interval above one half.

Reported: the ratio of the medians of the absolute error (estimator to causal GP, to hold, to the twin); the share of discharges by the median absolute error; the share of the total squared-error reduction against the causal GP held by the ten discharges with the largest reduction and by the twentieth of the discharges with the largest reduction (per repeat, mean over repeats), and the skill over the causal GP and over the twin with those ten removed (a lower bound, since the removal selects on the outcome); every skill of this section also against hold; the shares of rows in every class; the skill against the offline GP in the rows not flagged and in the quiet rows; the effective number of rows of the odds weighting.

## 5. What goes into the paper

Each pooled statement is written beside its split-level counterpart in the supplement, and the main text is changed where the verdict differs.

- A1 holds and the median ratio includes or exceeds 1: the sentence on the distribution stays, with the pooled numbers. A1 fails: the statement that the gain is in the upper part of the error distribution is withdrawn from the abstract.
- B1, B2 hold: the per-discharge sentence carries the pooled shares; the concentration is stated with the pooled share of the ten discharges and the skill without them. If the skill over the causal GP without those ten has an interval that includes zero, the main text says that the size of the margin over the causal GP rests on about ten of 639 discharges.
- C3 or C4 fails pooled as on the splits: the sentence that the margin over the causal GP leans on rows that a value rule would question stays. C3 holds pooled: that sentence is removed and the limit on the tail is shortened.
- D1 and D2 hold: the limit "in the hottest third and not in the others" is removed. Either fails: the limit stays, with the pooled thirds.
- F3 and F4 hold: the limit on rows without a value reads that both margins remain under the weighting. Either fails: the limit stays, with the pooled numbers.
- E2 fails: the main text keeps "20 or 30 ms old" as the range for which the margins are shown, and says that beyond 50 ms nothing is established.
- H1, H2: the sentence on quiet rows in the section on the beam is given with the pooled numbers for the full estimator beside those of the BES network on the splits; H1 failing is stated as a limit of that section.
- Failed statements are reported as failed.

## 6. Expectations

A1: ratio 0.92 to 0.95, holds; median ratio to the causal GP 0.99 to 1.05. B1: 0.62 to 0.70, holds; B2: 0.58 to 0.68, holds. Share of the ten discharges with the largest reduction: 0.25 to 0.45; skill over the causal GP without them +0.04 to +0.07, interval above zero. C1: +0.03 to +0.07, holds; C2: +0.03 to +0.07, holds; C3: share 0.35 to 0.55, fails; C4: 0.60 to 0.80, holds. D1: +0.05 to +0.12 in each third, holds; D2: +0.02 to +0.08, holds in the hottest and one other third, the statement fails. E1: +0.06 to +0.11, holds; E2: -0.03 to +0.08, fails. F1: +0.07 to +0.12, holds; F2: +0.04 to +0.08, holds; F3: +0.03 to +0.10, holds; F4: +0.01 to +0.06, fails. G1: +0.15 to +0.25, holds. H1: +0.02 to +0.06, holds; H2: 0.55 to 0.80, holds.

## 7. Limits known in advance

The pooled folds re-partition the discharges on which the recurrent design was adopted; no untouched hold-out exists. One training per fold: the two-level interval covers the difference between re-partitions, in which the initialisation changes too, but not a spread between initialisations on a fixed fold. The classes of rows are defined from BES, and the value rules are thresholds. The replays with a late or sparse CES value are not part of this batch (they need the 80 networks to be run again). Removing the discharges with the largest reduction selects on the outcome.
