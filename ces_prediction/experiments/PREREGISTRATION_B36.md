# B.36 — a layer on top: can the signed error of the causal variant be corrected from what is known at the row?

Registered 2026-11-16, before any correction model was fitted. CPU only; no network is trained.

## 1. Why

B.34 showed that the size of the error of the causal variant is predictable at the row (the tenth of rows named by a model holds 0.433 to 0.503 of the squared error). The author's point: if the rows that will be wrong are known, a further layer can be stacked on the estimator. A layer lowers the error only if the sign of the error is predictable as well as its size. B.33 found that an affine correction, global or per cluster, adds nothing (−0.011 to +0.009). A nonlinear layer has not been measured.

## 2. Seen before

B.33, B.34, and the stacking readout (`docs/paper/b30_stack.md`): a linear mix of the causal variant with the causal GP, hold and the history-only twin has skill +0.021 to +0.037 over the causal variant. Not seen: any nonlinear model of the signed residual.

## 3. Design

Rows, inputs (15), folds and model family as in B.34 (`HistGradientBoostingRegressor`, default settings, `random_state=20261116`; fitted on one parity fold of the test discharges, applied to the other, both directions joined). Intervals: 95 %, 1,000 resamples of discharges. A statement holds if it holds on at least three of four splits.

- Layer A: target y − causal; corrected estimate = causal + prediction.
- Layer B: as A, with the history-only twin's value minus hold added as a 16th input (what the linear stack used).
- Gated layer: layer A applied only in the tenth of rows with the largest predicted size (the B.34 error model of the same fold), causal elsewhere.

## 4. Statements

- **L1.** The skill of layer A over the causal variant has an interval above zero.
- **L2.** The skill of layer B over the causal variant has an interval above zero.
- **L3 (reported).** The gated layer over the causal variant; layers A and B over the linear stack of the same four estimators (weights fitted on the same folds); the median absolute error of layer A over that of the causal variant; in the top tenth of predicted size, the correlation of the predicted and the actual signed residual.

## 5. Decision

- L1 or L2 holds: stacked depth has room on these inputs, and a GPU batch is registered in which the same inputs feed a further stage trained on training discharges (a deeper network, or a second stage on the residual of the first).
- Both fail: the size of the error is predictable and its sign is not, on these inputs; the use of B.34 is then an interval or a warning, not a correction, and the record names the inputs that were not tried (the hidden state of the network, the raw stream).

## 6. Expectations

L1: +0.000 to +0.030, holding on one to three splits. L2: +0.010 to +0.040, holding on two to four. Gated layer within 0.005 of layer A. Layers over the linear stack: −0.02 to +0.02. Correlation in the top tenth: 0.05 to 0.30.

## 7. Limits known in advance

Fitted on about 48 test discharges per fold; default settings, one model family; cut population, natural cadence. A failure says nothing about a layer trained end to end inside the network on 500 training discharges.
