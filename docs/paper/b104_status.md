# B.104: matched-trajectory budget follow-up

## Running: 2026-11-24 00:36 KST

Registration and all execution sources were committed/pushed as8dd55a6 before
preparation or GPU work.89 tests and record checks passed. Prepared artifact,
raw grid/rows, stats/manifests and prior four-worker qualification were bound.
Four real-data GPU smoke children exited0; observer off/on validation losses
in both epochs and all best/final parameters were bitwise equal for both models.
Snapshot VAL reload and independent validation arithmetic passed.

Production began00:29:59 KST: HISTORY1701/1702/1703 and FAST1701 are active;
FAST1702/1703 are pending work-conserving slots. At00:36 all four had logged
four complete epochs, no failure. GPU samples72-74%,3138MiB. These are progress
observations, not TEST performance. SHORT states remain in CPU memory until
fit completion to avoid disk I/O shifting the deadline.

Tasks `B93_GPU_B104_budget` and `B93_Research_B104` are running; independent
monitor CLI acceptance observed, including training/e4 progress notice. Suite
clock began00:27:15 KST, nominal deadline03:27:15 KST, old B94 clock unchanged.
Prediction24 files, registered readout, independent audit, explicit report
publication and two bounded Claude post reviews are queued after all fits.
No source modification is permitted while this batch is running.

## Preparation (2026-11-24 KST)

The following paragraphs record the pre-launch state and are superseded above.

Implementation now includes common104/train104 frozen-loop adapters,
pipeline104, independent audit104, separate watch104, and Opus-generated
stats104. The large initial Fable code attempt exhausted its output budget
entirely in thinking; no partial file was accepted. All attempts/usage remain
local. A subsequent focused Opus statistics implementation and full Opus
scaffold review completed. Definite retry/timeout/liveness/publication and
smoke-control issues were fixed before first registration commit or GPU.
Exact best/final parameter and per-epoch VAL comparisons are recorded by the
real-prefix controls. Local notification success means CLI acceptance, not
proof the author read a message. UI config is ignored and never archived/API.

Commands after committed-source checks: `py ces_prediction/experiments/b104_budget/pipeline104.py prepare`,
then `register`. This starts the independent notifier first, verifies CLI
acceptance, runs four smoke controls and only then dispatches six fits. Status:
`py ces_prediction/experiments/b104_budget/pipeline104.py status`; local
`queue_status.json`, `watch_status.json`, `dispatch_events.json`, case
`process.log`. No other untracked research or manuscript files are touched.

Author requested continuation after the audited B.94 and corrected Claude plan.
No B.104 GPU fit has started yet. Source/implementation and real-data smoke
gates must pass first. Existing results and original source closures remain
read-only; untracked author presentation/ICML files are outside this scope.

Bounded official Claude reviews: Fable reviewed redacted existing TRAIN/VAL
curves; Opus adversarially selected six fresh trajectories rather than reusing
old early-stopped weights. FAST1701/1703 had converged, while all HISTORY fits
hit the old budget. Longer training therefore need not raise fast-channel skill.

Fixed registered study: V3_q3 only, three existing seeds, same FAST/HISTORY TCN
and original numerical train loop,40epochs/earlystop unchanged. Instrument each
fresh path to retain its1200s SHORT checkpoint, compare with its best up to3600s
LONG checkpoint. Four global workers, longer HISTORY jobs first. New3h clock
starts only on registration/launch; the old completed suite clock is unchanged.

Four real-prefix smoke controls compare observer off/on before production.
Endpoints D10/D50 by F/L5, common B.75 rows, fixed-init conditional intervals
and separate init-mixture intervals, per-init values, paired skill-change CI.
No numerical TEST readout before all six fit/exit audits. All directions remain
TEST-informed exploratory and cannot establish all-design or independent
physical generalization. Full contract `PREREGISTRATION_B104.md`.

Code generation/review attempts and usage: ignored `data/.b104/claude/`.
No secrets, raw observations or author identity are supplied to Claude.
Background monitor must deliver an actual status notification before training;
poll15s, immediate phase/failure and5min progress. No uncertain paid replay.
