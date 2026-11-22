# B.87 — matched regression at fixed reporting delays

Registered 2026-11-22 before implementation or readout. CPU only. Exploratory on previously used D1 data; no fresh confirmation set. Enters question 1 of PAPER_DESIGN, without changing the author's framing or estimator.

## Motivation and estimators

B.83 compares a trained causal regression with untrained interpolation. Separate information available after waiting from the regression family. Reuse B.73 rows, grid, fit/test discharges, history matrix, fast-window normalization and default seeded HistGradientBoostingRegressor, fitted to F minus causal GP. All four designs V1, V2, V3, V4a and V4b are reported, all existing units and repeats.

For each row, construct five future CES slots at exactly +10,+20,+30,+40,+50 ms (tolerance 3 ms), same file and segment, using D1 measured values. The target row is never a slot. Delays J=0,1,2,5 reveal slots 1 through J; later slots are NaN. Future fast channels are never used. At each delay fit history-only HJ and history-plus-present-fast XJ. Always append all five slot columns (including NaNs); J0 is a refit baseline, not asserted identical to stored TX. Missing future measurements stay NaN: no conditioning on their presence. Target F is available only for scoring/training, never as an input. A fixed delay is an idealized CES-reporting oracle, not an actual online deployment. If the CES exposure extends beyond its timestamp, physical availability adds that exposure and processing delay; 0/10/20/50 ms here denotes the timestamp information horizon.

All fits use B.73 complete_mask for causal inputs, D1 targets, and original GP fallback for incomplete causal rows. No tuning, stopping or choice based on scores. Pin CES data directory, held removal, 3000 eV cut, per-shot normalization off, expanding normalization off, zero-MC off; hide CUDA. Existing caches are mandatory; fail if absent. Cache only under data/.b87.

## Checks before scores

Check disjoint fit/test files; future slots same file/segment and strictly after t within the specified horizon; target mutation by +1000 eV leaves that row's inputs unchanged; mutation beyond horizon leaves inputs unchanged; present-fast windows unchanged across delays; finite predictions and identical test row indices across all arms. Write checks before any error calculation. Failed checks abort.

## Readout

For every design and arm, report row/shot/cluster counts, RMSE and MAE with 95% shot-cluster and discharge bootstrap intervals. For V1/V2 choose repeat/assignment first, then resample units; 20000 draws, deterministic seed 20261187 plus crc32(cell). Point estimates average repeat statistics. Report paired MSE skills X0 over H0, X0 over H1/H2/H5, X1/X2/X5 over X0, and XJ over HJ for each J, on identical rows. No claim that adding future measurements must improve held-out error.

Statements, each evaluated separately in every design: A, X0 over H1 has cluster lower bound >0 (expect uncertain); B, X0 over H5 lower bound >0 (expect false); C, X5 over X0 lower bound >0 (expect true); D, X5 over H5 lower bound >0 (expect uncertain). Report every statement and interval including failures; these exploratory intervals are not multiplicity-adjusted confirmation.

If A holds: say current fast diagnostics reduce error relative to a matched history-only estimator allowed one later timestamp, only in the named designs. If B fails: state that superiority to a longer-waiting learned comparator is not established; do not infer equivalence. If C holds: quantify the cost of zero waiting with matched estimators. If D fails: say the incremental value after waiting is not established. Neither outcome proves physical temperature recovery, local transport, or control benefit. No manuscript numbers are copied manually.

## Artifacts and independent verification

New code: ces_prediction/experiments/b87_latency/. Output docs/paper/b87_latency.{json,md}, b87_checks.md, b87_lead_checks.md; fit caches data/.b87. Independently recompute selected predictions' pooled errors from caches with a separate verifier and directly reconstruct sampled future slots from the grid. Update wiring, batch status and records after completion. Registration committed and pushed before running.
