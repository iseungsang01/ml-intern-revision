# B.106: train LSTM + correction, compare MSE first

State: launched2026-11-24 10:17KST; full GPU group dispatched10:18:37KST.
Sources/registration4f7bd65 and pre-fit runtime correction/addendumf3a233e
were committed and pushed before the effective cache preparation and launch.
The initial partial preparation is preserved separately and is not reused.
Existing pytest89 passed. Real-data original-branch/causal/OOF checks passed;
CUDA smoke completed one epoch and exited0. Longest-real-batch backward memory
checks passed for all four units and admitted four reserved workers
(1613MiB per slot;7726MiB free at admission).
Four outer LSTM children are running; sixteen inner OOF paths await slots.
No production fit or final comparison has yet completed at this status update.
Author authorized training; Claude/API review is explicitly unavailable and
no new API calls are made. Fixed four B105 units and current report F.
Twenty GPU trajectories = four full LSTMs + sixteen file-OOF LSTMs.
Two post-correction policies (D10-frozen and mixed) use honest OOF residuals.
All eight availability rules are evaluated on common rows against GP+fast
boosting,history-only boosting,GP,hold and saved TCN outputs.

The registration and addendum are committed; execution sources stay frozen.
Entry `py ces_prediction/experiments/b106_lstm/pipeline106.py prepare`, then
`register`. Status action is read-only. Caches under ignored `data/.b106/`.
No source mutation after launch. Whole8h cap is an upper bound,not an ETA;
each trajectory can stop early. No new architecture search is implied.

The independent watcher accepted local notifications and the durable supervisor
has a live process-bound heartbeat. It reports phase/failure and five-minute
progress. Its queue proceeds through all fits, OOF inference, CPU corrections,
frozen scenario replay, MSE-first readout and independent audit without a new
prompt. No performance-based architecture search follows automatically.
