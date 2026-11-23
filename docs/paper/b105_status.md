# B.105: TCN merit screening

State: FOUR GPU PRODUCTION FITS RUNNING,2026-11-24 03:19KST observation.
All four were dispatched03:18:54KST. GPU74%,2133MiB at03:19:21; no failure.
Registration and all execution sources were committed/pushed as1aee77b before
preparation/dispatch. Source-bound real inputs/model/causality checks passed;
all64 fair CPU reference fits and one CUDA smoke exited normally. Longest16
real-TRAIN-batch forward/backward probes admitted four reserved memory slots.
Supervisor and queue process identities are live, heartbeat fresh, independent
monitor notification accepted. Scientific sources are frozen during this run.
This is progress evidence, not predictive merit or TEST performance.

After four clean exits: separate fit/VAL-arithmetic audit -> common-row VAL
predictions -> all-unit allocation gate -> conditional TEST -> paired report
and independent raw-prediction/interval/gate audit -> bounded Fable post review.
Suite started03:16KST,150min cap; individual production soft3600s/hard4020s.
Original B94/B104 clocks and artifacts remain unchanged.

Author authorized GPU training and merit evaluation on2026-11-24.
One anchor-enabled35-channel TCN, four previously fixed units, seed1701.
Fair internal-TRAIN-only specialized/mixture GP residual trees precede GPU.
New code and registration must be committed before execution; real input/model
checks and a normally exited CUDA smoke must pass before four production fits.

Gate: equal14cell VAL mean normalizedMSE ratio to the fair specialized tree
<=1.05 in every unit. This is resource allocation, not inferential equivalence.
No candidate TEST if gate fails, no automatic wider search. Existing TCN22/B84
mixture comparisons are evaluated separately on registered stored predictions.
Legacy LSTM+correction remains explicitly unavailable under verified manifests.

Entry point: `py ces_prediction/experiments/b105_merit/pipeline105.py prepare`,
then `register` after committed-source checks. `status` reads effective status.
Ignored evidence: `data/.b105/queue_status.json`, `watch_status.json`,
`confirm/*/process.log`, `trees/*/process.log`, `gate.json`, `audit.json`.
Never print, transmit or commit local `watch_config.json` (UI authentication).
Final reports `b105_merit.json/md`, bounded Fable advice `b105_claude.md`.
