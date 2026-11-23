# B.101 addendum1: concrete implementation review gates

Before execution, Fable code review completed end_turn. Clarify that comparison
with carried-forward data establishes preservation, not independent numerical
verification; the unchanged original post audit supplies main-point and paired
bootstrap verification. Supplementary interval algorithms remain source-audited.
Inspect the auditor's complete source: it reads neural TX/TH metrics and never
constructs baseline paths/RNG from supplementary keys, so TREE_ is compatible.

Additional gates: bind all available/missing supplementary baseline files;
require the exact prediction filename set and hashes unchanged; compare report
JSON with readout JSON; allow markdown differences only in estimator table
rows and order neural rows first. Recompute two non-overwritten marginal
metrics as implementation controls (one TH collision cell, one TX full-support
cell) with exact equality to cached output. Independently check reconstructed
RMSE/MAE/median points against direct errors in every repaired cell. The controls
do not select or change any result. Preserve selection/checkpoint hashes too.
Record carried-forward versus independently verified scopes accurately.
