# B107: continue the LSTM comparison to its validation patience rule

2026-11-24. Author explicitly requests continuation after the working agent
disclosed that B106's arbitrary one-hour fit cap stopped two outer models
before the six-epoch validation-patience rule. Registered before any new fit,
handoff, calibration, or readout. No Claude/API calls. No new model search.

Keep B106 sources, registrations, artifacts and original clocks unchanged.
New code lives in b107_patience and ignored evidence in data/.b107. Same four
outer units, twenty full/OOF trajectories, seed1701, original LSTM Ti branch,
real caches/manifests/statistics, F-only D10 objective, batch16, optimizer,
negative-Ti penalty, scheduler and original absolute-epoch shuffle order.
Do not inspect TEST metrics to allocate continuation. Reason is the training
stop condition and TRAIN/VAL state, not observed comparative performance.

## Registered stopping and continuation policy

Apply one rule to ALL twenty trajectories: finish after six consecutive fully
validated epochs without a lower D10/F validation MSE. B106 stop='converged'
means this patience condition only, not proof of mathematical convergence.
Keep such paths unchanged. Continue every time/epoch-limited original path
from latest.pt's last fully validated epoch, restoring model, AdamW moments,
learning rate, ReduceLROnPlateau state, accumulated best checkpoint, historical
validation history, updates and trailing no-improvement count. Discard any
incomplete epoch exactly as B106's validated checkpoint contract specifies.
Unstarted paths train fresh under the same patience rule. No best-weight-only
restart, scheduler reset, or repeated scientific attempts after a failure.

Old checkpoints did not save random-generator states. The resumed dropout
stream therefore uses a declared deterministic seed1701+CRC32(case), set after
restoring the optimizer/model; this is optimizer-state continuation, NOT a
bitwise reproduction of an uninterrupted run. New checkpoints save Python,
NumPy, CPU and CUDA RNG states. Fresh paths retain original seed1701.
Absolute epoch ordering remains random.Random(epoch*10000+1701).

Emergency guards only: total400 epochs per path,24h per new child,72h suite.
These are not planned training durations. A guard hit is INCOMPLETE and blocks
the final adequate-training comparison, not a negative merit result. No limit
extension or architecture/hyperparameter selection is automatic.

## Controlled orchestration handoff and resource admission

Before handoff commit/push all execution sources and this registration. The
new controller may briefly suspend ONLY the process-identity-verified B106
queue, open query/synchronize handles for every unfinished original training
child, then terminate ONLY that queue (not the GPU children or their tree).
This prevents old pending dispatch and old downstream calibration/readout.
The original supervisor observes the deliberate queue exit; save a distinct
superseded/handoff record, never describe it as a scientific fit failure.
Never stop another service's or user's GPU process. Existing children finish
unchanged; new controller records actual handle-observed exit codes and runs
the original frozen fit verifier. Preserve original configs and artifacts.

Global admission is four GPU workers INCLUDING adopted children, using B106's
passed full-batch reserve1613MiB per new slot and actual free memory. Completed
limited outer paths have continuation priority as slots become free; all
others follow fixed case order. No score-based allocation. Open/checkpoint/
process identity failures abort and notify rather than guessing or retrying.
After killing the old queue, enumerate its direct children again to detect
any process not accounted for. Adoption handles preserve actual exit status.

## Final comparison

All twenty selected paths must pass source/config/manifest/statistics/checkpoint,
normal-exit and validation arithmetic checks AND the patience stopping rule.
Keep original reference hashes pinned through B106. Then reuse frozen B106
held/test inference, fifteen-feature OOF corrections (D10-only and mixed),
eight availability rules, identical common F rows, row-weighted physical MSE,
RMSE,20,000 paired cluster/discharge bootstrap intervals, all four designs,
S1/S2/S3 and separate raw-source/statistical audit. No change of estimand or
selection based on TEST. Original B106 readout is superseded by this amended
training protocol; do not silently relabel old numbers or checkpoints.

Inherited scopes and limitations remain: one unit/init per design, reused-data
exploration, different LSTM and TCN objectives, historical file-OOF rather
than prospective inner temporal calibration, fixed-fit intervals without
refit/model-selection uncertainty. A patience stop is not optimization proof.
Report original and continued epoch counts, parent hashes and RNG limitation.

Outputs b107_lstm.json/md and b107_status.md. Expected incremental merit is
uncertain. If S1/S2 unsupported: the adequately patience-stopped comparison did
not establish an increment in every registered unit/scenario. If supported:
report the point,95% interval and zero exclusion within the fixed exploratory
scope; no architecture originality or new-physics claim. A negative conclusion
could change with independently registered unused-data/budget/init evidence.

New source wiring: common107 owns provenance/selected-path resolution;
train107 restores or starts trajectories; pipeline107 controls handoff and
four-worker admission; analysis107 adapts frozen B106 producers/auditor to
resolved paths; watch107 sends local phase/failure/five-minute notifications.
No credentials in plaintext configs or source: reuse existing local DPAPI
protected notifier config. Watcher must accept notification before handoff.
