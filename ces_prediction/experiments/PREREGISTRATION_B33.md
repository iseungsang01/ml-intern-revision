# B.33 — do clusters of the fast-channel window carry structure that the trained estimator leaves unused?

Registered 2026-11-16, before any cluster was fitted. CPU only; no network is trained.

## 1. Why

The author proposed to redesign the estimator from its encoder: cluster the fast-channel input first and build an encoder and decoder on the clusters (or adapt an existing encoder). B.32 showed that rows of different kinds have different relations between the beam emission and T_i (the fast-channel increment is +0.494 to +0.677 in the first row after a blip with a departure beyond three standard deviations, against +0.064 to +0.113 over all rows). Before a network is designed and trained on a GPU, this check asks on the frozen predictions whether a clustering of the fast-channel window (a) separates rows in which the fast channels help by different amounts and (b) carries a correction that the present network does not already make. Only (b) is grounds for a cluster-conditioned design.

## 2. Seen before

The B.32 readouts (`docs/paper/b32_two_sources.md`, `b32_posthoc.md`, `b32_posthoc_model.md`) and `fast_departure.md`. The B.30 sensitivity table by rows after a blip. Not seen: any clustering of the fast channels, any error of any estimator by cluster, any correction of a residual.

## 3. Data, features, clusters

Rows and predictions: the frozen B.30 row tables `data/.b30/rows_cut_s{42,1,7,123}.npz`, T_i, cut population, natural cadence (k = 1), estimators `causal` (causal variant), `hist80` (history-only twin) and `hold`. Grid: `seq_data.load_grid_files(data, True, 3000.0)`.

For each split, the fit files are every file that is not a test file of that split; the test files are those of the row table.

Features of a row, causal (nothing after the row, no statistic of the scored discharge): each of the 15 fast channels is standardized with the mean and standard deviation of the fit files. Then, for BES (mean of 9 channels) and for ECEI (mean of 4): the level in the row, the departure (level minus its mean over the preceding ten rows), the standard deviation over the preceding ten rows, and the one-row increment; plus the time step to the previous row in units of 10 ms. Nine features. A row needs ten preceding rows inside its segment (segments break at steps of 0.5 s or more); rows without them are left out and their share is reported. Each feature is standardized on the fit rows and clipped at ten standard deviations.

Clusters: k-means (scikit-learn, `n_init=10`, `random_state=20261116`) on the feature rows of the fit files (a random 200,000 rows if there are more), **k = 6** primary; k = 3, 4, 8, 12 are reported as sensitivity and decide nothing. Test rows are assigned to the nearest centre.

## 4. Statements

Folds: the test discharges of a split are divided by shot-number parity into fold E (even) and fold O (odd). Intervals: 95 %, 2,000 resamples of discharges. A statement holds if it holds on at least three of the four splits.

- **K1 (the increment differs between clusters).** Among clusters holding at least 2 % of the scored test rows, the cluster with the highest and the one with the lowest fast-channel increment (skill of `causal` over `hist80`) are picked on fold E; on fold O the difference of their increments (high minus low) has an interval above zero. The same with the folds exchanged; a split holds if both directions hold.
- **K2 (the clusters carry a correction the estimator does not make).** The residual of `causal` (measurement minus prediction, eV) is regressed, by least squares, on an intercept, the BES departure and the ECEI departure, (i) with one set of three coefficients for all rows and (ii) with one set per cluster. Both are fitted on one fold and applied to the other, in both directions, and the corrected predictions of the two folds are joined. K2 holds on a split if the skill of the per-cluster correction over the global correction has an interval above zero.
- **K2b (reported, decides nothing).** Skill of the per-cluster correction over the uncorrected `causal`, and of the global correction over the uncorrected `causal`.
- **K3 (reported).** What the clusters are: per cluster the share of scored test rows, the centre in feature units, the hold RMSE, the increment, the share of rows that are the first row after an absent frame and the share within ten rows after one.

## 5. Decision

- K2 holds: a cluster-conditioned encoder or decoder is registered as a GPU batch against the present estimator (same splits, same epoch cap).
- K2 fails and K1 holds: the clusters separate rows where the fast channels help differently, and the present network already uses that; a clustering of these nine 10 ms features is then not grounds for a redesign. The measurement that would overturn this is named in the record: clusters of features the 10 ms files do not carry (the raw stream, twelve discharges) or a learned representation (an encoder pretrained on MAST), which is the author's second route.
- Both fail: stated as such, with the same named next measurement.

## 6. Expectations

K1 holds on four of four splits; the high-minus-low difference is +0.10 to +0.50. K2: per-cluster over global +0.000 to +0.010, holding on at most two of four splits (expected not to hold). Global correction over uncorrected: −0.002 to +0.005. Share of rows left out for lack of a window: 0.02 to 0.08. The first rows after an absent frame concentrate in one cluster (at least 0.6 of them in a single cluster).

## 7. Limits known in advance

The correction is fitted on test discharges of the same split (cross-fitted by parity), about 48 discharges per fold. Nine hand-made features and k-means are one clustering among many; a failure of K2 says nothing about learned representations. The row tables are of estimators trained at the natural cadence. The cut population only.
