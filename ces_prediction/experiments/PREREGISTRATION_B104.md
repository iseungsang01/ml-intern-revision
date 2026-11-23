# B.104: matched-trajectory training-budget sensitivity

2026-11-23, before any B.104 smoke or fit. The author requests continuation
after the completed B.94/Claude review and authorizes GPU use and bounded safe
recovery/status reporting. This is TEST-informed exploratory work, not a new
independent confirmation. Purpose/framing of the manuscript remains fixed.

## Fixed scope and expectation

One fixed previously proposed temporal unit V3_q3, seeds1701/1702/1703,
FAST(TX)/HISTORY(TH) TCN: six fresh fits, not checkpoint continuation. Original
TRAIN/VAL curves show two FAST fits converged before the budget but all HISTORY
fits budget-limited. More training may therefore reduce rather than increase
fast-channel skill; TEST improvement is not assumed. No model/loss/unit/seed
search. Other validation designs remain in the existing B.94 report; this batch
cannot establish a result across all four designs or new physical truth.

Same frozen train92/model/data/loss/optimizer/batch16/dropout.1/LR scheduler and
earlystop6. Retain40epochs; increase only training+validation soft wall1200 to
3600seconds. Six fresh fits, four global GPU slots; order HISTORY1701/1702/1703,
FAST1701/1702/1703. Work-conserving dispatch, VRAM guard; record concurrency and
actual epochs/updates. No unrelated process is stopped. Suite has a new3h clock
starting at launch; the old completed B.94 clock is never reset or extended.
Epoch/time caps are not convergence guarantees. Time estimates are provisional.

## Matched SHORT/LONG construction and controls

SHORT is the best validation checkpoint admitted by the1200s guard along the
same new LONG trajectory. The imported loop checks deadlines before every
training and validation batch, and once after a validated epoch. A read-only
iterator records batch-entry times; SHORT eligibility requires every training
and validation batch of the epoch entered strictly before t0+1200. Last-batch
completion after1200 is allowed, matching original soft admission semantics;
starts within1ms are flagged. This is an instrumented budget under this actual
schedule, not an exact replay of old wall times.

Snapshot improved eligible state to CPU memory, save after training. Observer
must preserve python/numpy/torch CPU/CUDA RNG hashes and record overhead;
maximum6seconds cumulative observer overhead per fit before TEST is accepted.
Identical SHORT/LONG cases contribute exact zero and are never dropped.

Run four real-data prefix smoke fits (FAST/HISTORY, observer off/on, seed1701,
two epochs,120s each). Same frozen numerical loop; compare every epoch's VAL
MSE and all best/final-validated checkpoint parameters with rtol1e-6,atol1e-7,
report exact equality separately.
Reload snapshots and reproduce smoke VAL MSE with rtol1e-6,atol1e-7. Require
source/software/prepared/manifest/statistics/env binding, all seven validation
rules and independent validation-arithmetic checks. A failed smoke stops;
do not tune tolerances after readout. Only real cached TRAIN/VAL arrays used.

## Registered readout

TEST construction starts only after all six fits/exit0 and independent fit
checks pass. Predict SHORT/LONG for D10/D50 only, F and L5 heads on the existing
B.75 common_F/common_L5 masks. No new mask/support selection. Same rows across
models/budgets/inits; no cross-design cluster intersection. D rules are CES
history availability delays, not forecast horizons. F/L5 are reported targets.

For each of four rule/head cells report all three per-init values and mean:
SHORT and LONG fast skill=1-MSE_FAST/MSE_HISTORY; per-model budget skill=
1-MSE_LONG/MSE_SHORT; primary change=LONG fast skill minus SHORT fast skill.
No inference from changed significance labels. Ratios use identical rows.

Use20000 registered draws with NumPy seed components20261104 and CRC32 of
resampling form. One cluster/discharge multinomial weight vector per draw on
the fixed prediction-support universe, shared by all four errors, all three
inits and endpoints. Primary percentile95% intervals hold the three inits fixed
and average their statistics within every draw. Supplementary init-mixture
intervals select one paired init per draw, matching the old B.94 convention;
these are not evidence for global initialization stability. Report undefined
draw counts, never silently redraw/drop inits. Original data/error arithmetic
and bootstrap must be independently reproduced before interpretation.

## Judgement and wording

Report every cell, point,95% interval and whether it excludes zero. Positive
change interval excluding zero: "At this unit under the registered schedule,
the longer-budget fast increment increased by the reported amount [interval]."
Negative change interval excluding zero: "The longer-budget fast increment
decreased by the reported amount [interval]." Otherwise: "Budget sensitivity
was not established at this resolution; the interval is reported." None shows
equivalence, architectural inferiority or independent physical truth. A
measurement overturning a budget explanation is the same paired contrast with
adequate TRAIN/VAL-chosen budget and independent unused-campaign confirmation.

## Integrity, failures and follow-up

Freeze/commit registration and all execution sources before GPU. Old artifacts
read-only. Explicit CES env. Distinct FAST/HISTORY and SHORT/LONG namespaces.
Parent independently records actual normal exit and full config/artifact hashes.
One fresh same-seed retry only for an external/native interrupted incomplete
attempt without Python diagnostic/NaN/smoke failure, after archival with hashes,
no live old child, and remaining suite time. This is not exact RNG continuation.
No general numerical retry or uncertain paid API replay. Separate status
observer15s, immediate phase/error and5min local notifications; verify local
CLI acceptance (not proof that the author read the notification).
New scientific directions require new registration.

Claude preference: preserve all attempts/usage, at most8 paid calls for this
batch's design/implementation/review/post planning. Fable design and Opus
adversarial design already completed on redacted TRAIN/VAL summaries only.
Implementation review precedes GPU; post review sees anonymized aggregates only
after independent audit. Keep TEST-informed proposals exploratory and bounded.

Pre-launch implementation clarifications: abort immediately on observer
overhead>6s. Parent hard-timeout is not retryable. A complete.json without
parent normal-exit proof requires inspection, not a fresh retry. Require4920s
remaining before a new confirm fit (4020s hard allowance plus900s readout
reserve). Fatal parent exceptions terminate only its birth-verified owned
children and record exit evidence. Rule supports must match within each
model/budget/init pairing; the resampling universe is their union across the
two rules. Markdown includes all intervals/per-init values/undefined draws.
Local ignored UI credentials are excluded from all archives, API evidence and
published reports; post-review removes unit/seed/config identifiers. Missing
diagnostic logs refuse automatic restart. All clarifications precede the first
registration commit and every scientific run.
