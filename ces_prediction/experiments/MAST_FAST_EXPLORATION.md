# MAST fast-signal exploration — protocol note (2026-11-16, written before any exploration output)

Direction (set 2026-11-15, restated 2026-11-16): look on FAIR-MAST for the place where signals other
than the charge-exchange past help to estimate a charge-exchange frame, state it as a result on
MAST, and carry it to KSTAR as a hypothesis to be tested when further KSTAR data exist. The model
form is free in this stage; a small form that can run inside one 10 ms interval is preferred, and
the result has to say why it differs from holding the last value (persistence).

## Split

- Discovery: the odd-numbered shots of `mast/coupling_shots.txt`. Everything in this stage
  (features, model form, hyperparameters, cells, thresholds) is chosen here. No registration per
  trial; numbers from this stage are exploratory and are never quoted as confirmed.
- Confirmation: the even-numbered shots of the same list. Not opened by `mast/explore_fast.py`.
  Before they are scored, one frozen model, its statements and its numerical expectations are
  registered in a separate file. Known beforehand about these shots: the lagged correlations of
  `docs/paper/mast_coupling.md` and the regime numbers of `docs/paper/mast_edge.md` (aggregates over
  both halves or over the confirmation half); no hidden-frame prediction error with a fast input has
  been computed on them.

## Task

Same decimation as the MAST gate (`PREREGISTRATION_MAST_GATE.md` section 2): block frames at odd
positions are hidden, the estimate for a hidden frame uses charge-exchange values of earlier visible
frames only, and beam power / D-alpha up to the end of the hidden frame window. Scored rows,
populations and the slope-state filter parameters are those of the gate and its addendum 1.

## First pass (this commit)

Row table per target with the charge-exchange past, the neighbouring chords at the last visible
frame, frame-window means and maxima of mid-plane D-alpha, and beam-power increments at lags 0 to
60 ms. Gradient-boosted trees with shot-level 5-fold prediction, four input groups (own past; plus
neighbours; plus fast signals; all), read by normalized radius and by D-alpha step class, with the
share of the persistence squared error that the reported errors account for.

## What counts as a place worth registering

A cell (radius range, event class, target) where the fast-signal increment over the same model
without fast signals has a shot-bootstrap interval above 0 on the discovery half and where the cell
can be defined on KSTAR from signals a KSTAR discharge has.
