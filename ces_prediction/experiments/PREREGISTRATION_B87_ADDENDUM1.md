# B.87 addendum 1 — all-missing columns

2026-11-22, before any successful fit or numerical readout. The first attempted fit failed in installed scikit-learn binning on all-NaN columns. No predictions were saved and no error was computed.

Construct the five slots as registered. Before each fit, remove columns that are entirely NaN on that unit's FIT rows only. Apply that exact selector to its test inputs. This also removes unrevealed slots at J0. Never select by test values, target association, or score. Record retained columns in fit caches and verify test mutation cannot change the selector. All-missing columns contain no information; estimator family, hyperparameters, units, row masks, horizons and registered statements remain as registered. This is a compatibility change, not a tuning choice. Registration is unchanged.
