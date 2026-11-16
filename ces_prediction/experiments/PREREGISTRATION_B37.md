# B.37 — four readouts proposed on 2026-11-16 and put into the day's plan by the author

Registered 2026-11-16, before any of the readouts below was computed. CPU only; no network is trained (the frozen causal variant is run forward).

Seen before: B.32 to B.36 (in particular B.34: cross-fitted on parity folds of the test discharges, the adaptive 90 % interval is 0.464 to 0.644 of the constant half-width and covers 0.856 to 0.875; B.36: a boosted layer on the signed residual has skill +0.007 to +0.023 over the causal variant with every interval including zero, a linear stack +0.021 to +0.036). Not seen: any prediction of the network on validation discharges, any model of a missing CES value, any noise-corrected skill, any run of the network on averaged raw input.

## V. Error model, interval, switch and layer fitted on validation discharges

B.34 and B.36 fitted their models on half of the test discharges. Here the causal variant (`data/.b29_norm0_cut_s{seed}`) is run forward on the validation discharges of its split (`data/.b1_w2cut_split_s{seed}/split_manifest.json`, about 127 discharges, not used for the gradient), with hold and the causal GP computed as in `run_b30.rows` at the natural cadence. The models of B.34 and layer A of B.36 (same inputs, same settings) are fitted on the validation rows and applied once to all test rows of the frozen row table. A check precedes the readout: on five test files the recomputed predictions of the network, hold and the GP agree with the row table within 0.5 eV on at least 99 % of the rows; otherwise the part stops.

- **V1.** The tenth of test rows with the largest predicted size holds a share of the squared error whose interval lies above 0.30.
- **V2.** The adaptive 90 % interval covers 0.87 to 0.93 of the test rows and the ratio of its median half-width to the constant half-width has an interval below 1.
- **V3.** The median absolute error of the switch over that of the causal variant and over that of hold has an interval below 1 in both cases, and its mean-squared-error skill over the causal variant has an interval above −0.02.
- **V4.** The skill of layer A over the causal variant has an interval above zero.

Each holds if it holds on at least three of four splits. Intervals: 95 %, 1,000 resamples of test discharges. Decision: V1 and V2 hold → the manuscript gains an interval beside the value (fitted on validation discharges), and a variance output is registered as a GPU batch. V4 holds → a stacked stage is registered as a GPU batch. Expectations: V1 0.40 to 0.55 (four of four); V2 coverage 0.86 to 0.92, width ratio 0.45 to 0.75, holding on two to four splits; V3 does not hold; V4 +0.005 to +0.035, holding on one to three splits.

## F. Can a missing CES value be predicted?

Rows: every grid row with a ten-row fast-channel window (grid `load_grid_files(data, True, 3000.0)`). Target: T_i is not usable in the row (missing, or above 3 keV). Fit on the files that are not test files of the split, apply to its test files. Models: `HistGradientBoostingClassifier` (default, `random_state=20261116`); history inputs (7): unusable flags of the five preceding rows, the time step, the number of rows since the last usable value (capped at 50); full inputs: those plus the nine window features of B.33.

- **F1.** The area under the ROC curve of the full model minus that of the history model has an interval above zero (300 resamples of test discharges), on at least three of four splits.
- Reported: both areas, the average precision of both, the base rate, and the same for the rows whose preceding row was usable (onsets of a gap).

Expectations: base rate 0.07 to 0.09; history AUC 0.85 to 0.95; difference +0.005 to +0.05, holds; on onsets the full model 0.70 to 0.85 and the difference +0.05 to +0.20.

## Q. Skill with the fit noise taken out (reported, decides nothing)

If a measured value is a slowly varying signal plus white fit noise of variance σ², the lag-1 autocovariance of the increments is −σ² plus that of the signal's increments, so σ² is at least minus that autocovariance when the signal's increments are not negatively correlated. On the test discharges of each split (cut population, triples of consecutive observed rows): σ² = −cov(Δy_t, Δy_{t+1}), once plain and once with increments clipped at their 1st and 99th percentiles. Reported: σ, the mean squared error of hold, the causal GP, the history-only twin and the causal variant on the scored rows, and the skill of the causal variant over each with σ² subtracted from both mean squared errors, with intervals. Stated with it: hold carries the noise of its own sample, so subtracting one σ² from hold is a lower bound of what is removable there. Expectation: σ 100 to 160 eV plain, 30 to 60 eV clipped; with the clipped σ the skill over the causal GP rises by a factor 1.1 to 1.6.

## S. A frame mean instead of a snapshot (twelve discharges with a raw stream)

A row of the 10 ms files holds one raw sample per fast channel. For the discharges with a raw stream that are test discharges of a split, the causal variant of that split is run three times on the same file: as delivered; with the BES and ECEI values of the rows inside the raw window replaced by the raw sample at the stamp (a check: it reproduces the delivered values); and with them replaced by the mean of the raw stream over the 10 ms before the stamp (and, reported, over 1 ms). Scored on the rows inside the raw window with a measured T_i (at most 3 keV), natural cadence. Each discharge enters once, in the first split of the order 42, 1, 7, 123 in which it is a test discharge.

- **S1.** The skill of the network with 10 ms means over the network as delivered has an interval above zero (2,000 resamples of the discharges).
- Reported: the same for 1 ms means; the standard deviation of the fast-channel input (in the network's standardized units) as delivered and averaged; per-discharge skills.

The network was trained on snapshots, so averaged input is outside its training distribution; S1 failing does not show that a frame mean carries less. Expectation: seven discharges; skill −0.08 to +0.02; S1 does not hold.
