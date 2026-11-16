# B.34 — can the estimator know in advance which rows it will get wrong?

Registered 2026-11-16, before any error model was fitted. CPU only; no network is trained.

## 1. Why

The skill of the recurrent estimator is carried by few rows (the 1 % of rows with the largest squared error hold 48 % of its total in the cut population) and on a typical row it is not better than hold (median absolute error 47.2 against 46.1 eV). Every estimator so far gives one value and no statement of how far to trust it. If the size of the error can be predicted from what is known at the row, three things follow that have not been tried: an interval beside the value, a switch between hold and the network, and a named place where a redesign has room. This check asks whether it can, on frozen predictions.

## 2. Seen before

The tail and median numbers above; B.33 (clusters of the fast-channel window carry no correction of the mean); the B.32 readouts. Not seen: any model of the size of an error, any interval, any switch.

## 3. Data and the error model

Frozen B.30 row tables `data/.b30/rows_cut_s{42,1,7,123}.npz`, T_i, cut, natural cadence, rows with `hold`, `gp`, `causal` and the measurement finite and a ten-row fast-channel window (as in B.33). The test discharges of a split are divided by shot-number parity; every model is fitted on one fold and applied to the other, in both directions, and the two halves are joined.

Inputs known at the row (15): the nine window features of B.33; the age of the last CES sample; the hold value; `causal − hold`; `causal − gp`; and the absolute values of those two differences. Model: `sklearn.ensemble.HistGradientBoostingRegressor` (default settings, `random_state=20261116`).

- Error model: target log(|causal − y| + 1 eV); its prediction, exponentiated, is the predicted size s.
- Switch model: target |hold − y| − |causal − y|; the switch uses `causal` where the prediction is above zero and `hold` elsewhere.
- Interval: half-width q·s, with q the 0.90 quantile of |causal − y| / s on the fitting fold; the constant interval has half-width equal to the 0.90 quantile of |causal − y| on the fitting fold.

Intervals of the statistics: 95 %, 1,000 resamples of discharges. A statement holds if it holds on at least three of the four splits.

## 4. Statements

- **E1 (the large errors are predictable).** The tenth of rows with the largest predicted size holds a share of the summed squared error of `causal` whose interval lies above 0.30 (0.10 if unpredictable).
- **E2 (an interval that adapts is shorter).** The adaptive interval covers 0.87 to 0.93 of the rows, and the ratio of its median half-width to the constant half-width has an interval below 1.
- **E3 (a switch lowers the typical error without losing the mean).** The median absolute error of the switch, over that of `causal` and over that of `hold`, has an interval below 1 in both cases, and the mean-squared-error skill of the switch over `causal` has an interval above −0.02.
- Reported: the share of rows on which the switch uses `causal`; the mean-squared-error skill of the switch over hold; which inputs the error model uses most (permutation importance on the applied fold); E1 with the fast-channel features removed.

## 5. Decision

- E1 and E2 hold: a variance output is registered as a GPU batch (the same network with a second head trained on the Gaussian negative log-likelihood, against the present one), and the manuscript gains an interval beside the value, marked as fitted after the fact on test discharges.
- E3 holds: the switch is registered for training on the training discharges.
- E1 fails: the tail is not predictable from these inputs; the record names what was not tried (the raw stream, the hidden state of the network).

## 6. Expectations

E1: share 0.35 to 0.60, four of four. E2: coverage 0.88 to 0.92, width ratio 0.45 to 0.80, four of four. E3: does not hold (median ratio against hold 0.97 to 1.03); the switch uses `causal` on 0.3 to 0.7 of the rows; skill of the switch over `causal` −0.03 to +0.01. Most used input: |causal − hold|.

## 7. Limits known in advance

The models are fitted on about 48 test discharges of the same split, not on training discharges, and the network's own predictions on training discharges are not stored. Cut population, natural cadence. A share of squared error is a tail statistic.
