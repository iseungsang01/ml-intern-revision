# Fable training efficiency and candidate reduction review

Author-authorized official API. Redacted protocols only, no raw observations, identity, credentials or scientific test scores. Independent methodological advice, not evidence of performance or novelty. Exact prompts/responses and usage retained locally. Initial thinking-disabled requests were rejected by this model. Two adaptive/high-effort requests exhausted their output budgets without final answers; their exact records remain training.json/selection.json and are not treated as completed reviews. The bounded final-answer requests use adaptive/low effort on the SAME Fable model, one search maximum.

## training

Returned model `claude-fable-5-2`, stop `end_turn`, usage `{"input_tokens": 16723, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0, "cache_creation": {"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 0}, "output_tokens": 4999, "output_tokens_details": {"thinking_tokens": 2276}, "service_tier": "standard", "inference_geo": "global", "speed": "standard", "server_tool_use": {"web_search_requests": 1, "web_fetch_requests": 0}}`.

### Exact prompt

Act as a skeptical senior collaborator on an actual small diagnostic nowcasting study.
We need substantially faster progress and a defensible SMALL method shortlist, not an architecture zoo.
Task: causal scalar temperature nowcast at one pedestal-top measurement chord using fast diagnostics
and only previously available temperature measurements. Direct two-head normalized F=current report
and L5=median of at least3 subsequent5 reports, a future-derived TRAINING LABEL ONLY, not truth.
Real previously used data: about246k grid rows,639 files, mostly10ms, segments reset at gaps>=.5s.
22 input channels:13 fast plus2 zero placeholders; log1p(dt), carried past temperature, log1p(age),
has-history and unused rotation slots. Train-file-only normalization. Seven availability rules mixed
by segment/epoch; an every10th-frame rule is excluded from training/validation. No current/future CES
as input, no test-time normalization. Four designs random-file,clusterblocked,rollingorigin,chronological,
none primary. Existing boosted-tree/history baselines exist. No new architecture/physics discovery or
journal acceptance claim is needed. Standard regression is eligible if useful against fair baselines.
Training:2layer160 LSTM,LayerNorm,two64GELU heads,AdamW.001,batch16 full segments,cap80,plateaupatience2,
earlystop6,masked equal-head MSE+.1negativepenalty. Train about170krows/1328segments, val34k/235segments
in firstunit. Val replays ALL7 rules EVERYepoch, equalrule/head MSE chooses checkpoint. Split units15:
V1 fivefold,V2 fivefold,V3 four origins,V4a finaltime;3seeds;X/H historytwin =>90fits/model family.
Hardware:8GB laptopGPU,Windows,Python3.14,torch2.11CUDA12.8. Baseline uses packed variable-length LSTM.
cuDNN path computed then interpreter aborted on teardown; nativeATenCUDA clean but lowGPUuse~10%,oneCPUcore.
Eachfit about40min; training83batches/epoch versus7xvalidation105batches/epoch. No profiler breakdown yet.
Frozen ongoing files cannot be changed; existing completed fits remain historical. New implementation
and revised allocation require committed preregistration, independent real-input/loss/gradient checks,
normal interpreter exit0 (no os._exit bypass) and source/config bound artifacts. Own queued supervisors
can be superseded; never stop unrelated GPU holders. No synthetic observations or external private data.
Currently queued:anchor/current shared13->32->16encoder, lastvisibleCES MEASUREMENT-time anchor, matched
no-anchor control, originalLSTM fused39;90fits. Then7candidates:lagdecayGRU,causalTCN128/6blocks/RF127,
small diagonal discrete/continuous learned observers(NOT actual RKN/CRU reproductions),LSTM Huber,
fixedStudentt4,LSTM Gaussianmean/logvar;210one-seedscreenfits15unitsXH,validation-onlytop2+120fits.
This cost is excessive. No test score has been used to choose candidates. Need practical preregistered
revision with honest reused-data exploration, no inflated claims from reduced validation/seed budgets.
Use web_search at most3times for PRIMARY papers/official technical docs, cite exact URLs. Give concrete
best recommendations and failure tests; do not promise unmeasured speedups or recite30methods.
Answer under2200words so the response finishes. Tables okay. No code implementation necessary.

Analyze ONLY current model TRAINING methodology/runtime, independently of shrinking candidate count.
Rank concrete bottleneck fixes by expected benefit/engineeringrisk and classify arithmetic-equivalent
implementation change vs statistical-method change. Examine packed vs dense CAUSAL right-padded LSTM
with supervision masking (no backward crosspadding), cuDNN/trainingflag/dropout/cleanexit/version issues,
precache validation tensors and batching rules, equal-rule/head aggregation, avoidingscalar sync,
lengthbucketing, batchsize, mixedprecision/compile, boundedcontextTBPTT and validationfrequency.
Choose the smallest highconfidence implementation/profiling plan that can run after the current fit,
not after90fits. Specify realtrace batch/gradient/optimizerstep/stream equivalence tests, tolerances,
maxbenchmarkbudget, decision gates, rollback and preserving completed results. Is fused dense likely
worth testing before handrolledcell/observer models? Identify misleading acceleration checks, including
evalmode cuDNN backward and dropoutRNG. Propose a fixed training budget for a NEW batch if exact
baseline budget is too expensive; don't silently call changedstatisticssamecomparison. End with exact
recommended next3actions and the minimum uncertainty that must be resolved by measurement.

IMPORTANT: Deliver your final actionable answer now, at most900words. Use at most ONE web search, then finish. Prioritize the concrete plan over extended deliberation.

### Response

**Bottom line:** profile first, then make validation cheap and exact, and test dense fused ATen before any hand-rolled cell. Nothing below is a measured speedup.

## Reading of your numbers
- **Batch counts:** 1328/16 ≈ 83 training batches; 235/16 ≈ 15 × 7 rules = 105 forward-only validation batches. Validation is plausibly a third to half of an epoch, not obviously dominant.
- **Likely regime:** ~10% GPU with one CPU core saturated suggests launch-bound work (per-timestep kernels of the non-cuDNN path, pack/unpack, Python collation, `.item()` syncs). Segments average ~128 steps with a tiny model. If so, AMP will not help.

## Ranked fixes

| # | Fix | Class | Benefit / risk |
|---|---|---|---|
| 1 | Profiler breakdown of 2 real epochs (data build, pack, LSTM forward, backward, optimizer, validation, sync) | none | Mandatory / nil |
| 2 | Precache all 7 validation rule tensors on GPU once; forward in a few large length-sorted batches | Equivalent only if rule replay is deterministic and aggregation uses per-rule/head sums and counts | High / low |
| 3 | Accumulate losses on-device, one sync per epoch | Equivalent | Medium / nil |
| 4 | Dense right-padded LSTM with supervision mask, native ATen | Equivalent up to float rounding | Unknown, test / low |
| 5 | Dense cuDNN | Equivalent up to float rounding, but exit-0 is unresolved | Possibly large / high |
| 6 | Training length bucketing, larger batch, TBPTT, validation every k epochs | Statistical change | Defer |
| 7 | AMP, `torch.compile` | Arithmetic change; toolchain risk on Windows/3.14 | Skip |

**Dense causal equivalence.** For a unidirectional LSTM with right padding, outputs at valid positions cannot depend on the padding. Three caveats:
- Never use `h_n`; gather at length−1 if a final state is needed.
- Keep the masked-loss denominators identical.
- LayerNorm must be per-timestep over features only.

**Aggregation trap.** If the current checkpoint metric is a mean of batch means, rebatching validation changes the selector. Reproduce the existing formula from stored per-segment sums and counts, and verify it against a completed fit's logged values.

**cuDNN.** Treat it as optional.
- Test dense and packed separately; the teardown abort may be specific to one path.
- Gate: 20 of 20 consecutive normal exits after explicit `del`, `synchronize`, no workers, `cudnn.benchmark=False`.
- If it fails, drop cuDNN. A Python 3.12/3.13 environment is a separate preregistered version change.
-

PyTorch's LSTM documentation ties cuDNN's persistent algorithm to non-packed input, but also to float16 data and a V100 GPU

, so dense input alone will not get you that path (https://docs.pytorch.org/docs/main/generated/torch.nn.modules.rnn.LSTM.html).

## Misleading checks to ban
- **Eval-mode gradient checks:** cuDNN RNN backward requires training mode. An eval-mode check either errors or silently exercises the non-cuDNN path.
- **Dropout:** cuDNN and ATen use different RNG streams. Run equivalence with dropout 0 only. If the baseline has dropout above 0, stream equivalence across backends is unattainable, so say so.
- **Unsynchronized timing:** timing without `cuda.synchronize`, with the first-epoch warm-up included, or on synthetic lengths.
- **GPU utilization:** only wall-clock per real epoch counts as speed.
- **Bitwise loss trajectories:** AdamW's normalized update amplifies rounding where gradients are near zero.

## Equivalence tests
Use real trace batches, the longest and shortest segments, and all 7 rules.

| Test | Tolerance |
|---|---|
| float64 CPU, packed vs dense: outputs and both head losses | ≤1e-10 abs |
| float32 GPU: outputs at valid positions | ≤1e-5 abs (normalized units) |
| Per-parameter gradient, relative L2 | ≤1e-4 (float32), ≤1e-8 (float64) |
| Padding perturbation: fill pads with random values | Loss and gradients unchanged to the tolerances above |
| One AdamW step from identical state: parameter relative difference | ≤1e-5 |
| 50-step stream, same order and seed: loss relative difference | ≤1e-3; report, do not require tighter |
| Validation selector vs frozen code on a saved checkpoint, per rule and head | ≤1e-6 relative |
| One full fit on one unit | Within the existing 3-seed spread for that unit; same best epoch ±2 |

## Budget, gates, rollback
- **Benchmark budget:** at most 3 GPU-hours, one unit (V1 fold 0), starting after the current fit finishes. Supersede only your own queued supervisor.
- **Adoption gate:** all tests pass, exit 0, and at least 2× measured wall-clock per epoch. Below 1.3×, abandon the change and keep the frozen path.
- **Rollback:** the new code lives in a new source/config hash and artifact directory. Completed fits stay untouched and are labeled "historical implementation".
- **Pooling:** results from old and new implementations are pooled only if the full-fit gate passes. Otherwise the baseline is rerun under the new implementation.

## Fused dense before hand-rolled cells?
Yes. Hand-rolled cells and observers are Python per-timestep loops in the same launch-bound regime. The dense infrastructure (cached tensors, masks, no syncs) is shared by them, and the profile will show whether per-step loops are affordable at all. If dense ATen shows no gain, expect the observers to be at least as slow as today, and cut or shrink them on cost grounds.

## Fixed budget for a new batch
If the equivalent fixes still leave fits above about 15 minutes, define "Budget B". First inspect the best-epoch distribution in the historical logs and disclose that as reused-data exploration. Then fix:
- an epoch cap near the 95th percentile of best epochs (for example 40);
- the same plateau and early-stop rules;
- all-rule validation every epoch, with the selector unchanged.

Rerun the baseline LSTM under Budget B as the only valid comparator. Never tabulate Budget B candidates against cap-80 historical fits as the same comparison. One-seed screens are reported as screening, not ranking evidence.

## Next 3 actions
1. Commit a preregistration addendum covering the profiling protocol, the tests and tolerances above, the gates, the 3-hour budget and the Budget B rule.
2. After the current fit, run the 2-epoch profile on the frozen code. Then implement validation precache, on-device accumulation and the dense ATen path, and run the equivalence suite.
3. Run one full fit plus the 20-exit cuDNN trial, then decide whether to adopt, fall back to ATen only, or roll back. Re-plan the queue only after that.

## Minimum uncertainty to measure
- Time split: training backward vs validation vs CPU-side prep.
- Whether dense ATen is faster at all on this GPU.
- Whether cuDNN exits cleanly on either path.
- Whether the current validation aggregation is batch-mean-based.
- Whether the baseline uses nonzero dropout.

## selection

Returned model `claude-fable-5-2`, stop `end_turn`, usage `{"input_tokens": 18330, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0, "cache_creation": {"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 0}, "output_tokens": 5333, "output_tokens_details": {"thinking_tokens": 2694}, "service_tier": "standard", "inference_geo": "global", "speed": "standard", "server_tool_use": {"web_search_requests": 1, "web_fetch_requests": 0}}`.

### Exact prompt

Act as a skeptical senior collaborator on an actual small diagnostic nowcasting study.
We need substantially faster progress and a defensible SMALL method shortlist, not an architecture zoo.
Task: causal scalar temperature nowcast at one pedestal-top measurement chord using fast diagnostics
and only previously available temperature measurements. Direct two-head normalized F=current report
and L5=median of at least3 subsequent5 reports, a future-derived TRAINING LABEL ONLY, not truth.
Real previously used data: about246k grid rows,639 files, mostly10ms, segments reset at gaps>=.5s.
22 input channels:13 fast plus2 zero placeholders; log1p(dt), carried past temperature, log1p(age),
has-history and unused rotation slots. Train-file-only normalization. Seven availability rules mixed
by segment/epoch; an every10th-frame rule is excluded from training/validation. No current/future CES
as input, no test-time normalization. Four designs random-file,clusterblocked,rollingorigin,chronological,
none primary. Existing boosted-tree/history baselines exist. No new architecture/physics discovery or
journal acceptance claim is needed. Standard regression is eligible if useful against fair baselines.
Training:2layer160 LSTM,LayerNorm,two64GELU heads,AdamW.001,batch16 full segments,cap80,plateaupatience2,
earlystop6,masked equal-head MSE+.1negativepenalty. Train about170krows/1328segments, val34k/235segments
in firstunit. Val replays ALL7 rules EVERYepoch, equalrule/head MSE chooses checkpoint. Split units15:
V1 fivefold,V2 fivefold,V3 four origins,V4a finaltime;3seeds;X/H historytwin =>90fits/model family.
Hardware:8GB laptopGPU,Windows,Python3.14,torch2.11CUDA12.8. Baseline uses packed variable-length LSTM.
cuDNN path computed then interpreter aborted on teardown; nativeATenCUDA clean but lowGPUuse~10%,oneCPUcore.
Eachfit about40min; training83batches/epoch versus7xvalidation105batches/epoch. No profiler breakdown yet.
Frozen ongoing files cannot be changed; existing completed fits remain historical. New implementation
and revised allocation require committed preregistration, independent real-input/loss/gradient checks,
normal interpreter exit0 (no os._exit bypass) and source/config bound artifacts. Own queued supervisors
can be superseded; never stop unrelated GPU holders. No synthetic observations or external private data.
Currently queued:anchor/current shared13->32->16encoder, lastvisibleCES MEASUREMENT-time anchor, matched
no-anchor control, originalLSTM fused39;90fits. Then7candidates:lagdecayGRU,causalTCN128/6blocks/RF127,
small diagonal discrete/continuous learned observers(NOT actual RKN/CRU reproductions),LSTM Huber,
fixedStudentt4,LSTM Gaussianmean/logvar;210one-seedscreenfits15unitsXH,validation-onlytop2+120fits.
This cost is excessive. No test score has been used to choose candidates. Need practical preregistered
revision with honest reused-data exploration, no inflated claims from reduced validation/seed budgets.
Use web_search at most3times for PRIMARY papers/official technical docs, cite exact URLs. Give concrete
best recommendations and failure tests; do not promise unmeasured speedups or recite30methods.
Answer under2200words so the response finishes. Tables okay. No code implementation necessary.

Analyze ONLY CANDIDATE REDUCTION methodology. Choose at most THREE model configurations INCLUDING
the baseline/control, not three broad families with dozens of hidden subvariants. Account for redundant
handrolled observers, lack of fit-error labels and weak calibration evidence; distinguish precision gain
from addingphysicalvalue. Protect explicitanchor hypothesis and a strongsimpletemporalbaseline ifjustified.
Should robust losses or uncertainty be deferred rather than searched? Compare fixedsmallpilot vs
successivehalving/Hyperband/ASHA, learningcurveearlyrankmisordering, samevalidationfit reuse, temporal
designs, splitunits and pairedX/H controls. Recommend an EXACT small budget: whichoneunit/design/seed,
epochsorwallbudget, valonlyranking/stopcriterion, numberpromoted, expanded4designcoverage, totalfits,
testlock and honest intervals/modelselectionlimitations. We can report all4designs without three seeds
for every weakcandidate; don't assert unchanged evidence strength. Do not select using alreadyseen
testmetrics or dropdifficulttemporaldesigns. Include whatmeasurementwouldre-open a deferredcandidate.
End with ONE preferred portfolio, exacteligibility/rankingrules and a succinct protocol we can register.

IMPORTANT: Deliver your final actionable answer now, at most900words. Use at most ONE web search, then finish. Prioritize the concrete plan over extended deliberation.

### Response

**Recommendation:** cut to three fixed configurations, run a 6-fit pilot on one unit, then cover all 15 units with one seed plus a small seed-variance subset. That is at most 138 fits, against 420 planned (90 + 210 + 120).

## Portfolio

I assume X/H means the fast-diagnostics model and its history-only twin; correct me if not.

| ID | Configuration | Role |
|---|---|---|
| C0 | Matched no-anchor control (frozen LSTM recipe) | Control |
| C1 | Explicit last-visible-CES anchor, same encoder | Protected hypothesis |
| C2 | Causal TCN 128/6 blocks/RF127, same loss and optimizer | Single temporal challenger |

Existing boosted-tree/history baselines are reported alongside at no fit cost.

I chose the TCN over the lag-decay GRU because the GRU's decay duplicates the dt/age/has-history inputs the LSTM already sees. The TCN is a different inductive bias and avoids the packed-LSTM path; any speed benefit is unmeasured.

## Deferred, with re-open triggers

| Deferred | Why | Re-open if |
|---|---|---|
| Both hand-rolled observers | Redundant with each other, not reproductions, and nothing to validate their gains or variances against without fit-error labels | C0–C2 residuals grow with age faster than carried-history persistence; then the continuous one only |
| Huber, Student-t4 | They change the loss while selection stays MSE, and L5 is already a median | Top 1% of validation residuals carry over 30% of F-head MSE and concentrate in identifiable files; then Huber, one δ |
| Gaussian mean/logvar | It would learn label scatter, not measurement uncertainty, and calibration is unverifiable | Fit-error labels become available, or an interval requirement is preregistered with a coverage test |
| Lag-decay GRU | Redundant with existing inputs | C2 degrades specifically where age exceeds its 1.27 s receptive field |

So robust losses and uncertainty should be deferred, not searched. Each needs a measured trigger and a single preregistered variant.

## Fixed pilot, not halving

ASHA-style methods are the wrong tool here. Their savings come from

aggressive early-stopping

, and a three-arm comparison is far from the

"large-scale regime"

 ASHA was designed for (https://arxiv.org/abs/1810.05934). The same objection covers successive halving and Hyperband.

With three arms, a wrong early cut costs more than the compute it saves. TCN and LSTM learning curves can cross, and plateau scheduling shifts ranks late. Every fit therefore runs the frozen stopping rule (cap 80, early stop 6), with no truncated rungs.

## Exact budget

**Pilot (6 fits):** C0, C1 and C2, each X and H, on V3 rolling-origin origin 2, seed 0.
- V3 is temporal, so it is not the optimistic random-file case.
- It is not the smallest-training origin.
- It leaves V4a untouched for confirmation.

The pilot is an eligibility gate, not a ranking:
- Interpreter exit 0, with no bypass.
- Real-input, loss and gradient checks pass.
- No rule/head validation loss is non-finite.
- Wall time is recorded, with a profiler breakdown.

**C2 promotion rule (validation only):** C2-X's equal-rule/head validation MSE must be no worse than C0-X's by more than 2× the seed SD. Estimate that SD from the three historical baseline seeds on the same unit. If those fits used a different validation protocol, substitute a fixed 5% margin and register that now.

C0 and C1 are promoted unconditionally, since the anchor hypothesis is not subject to a one-seed veto. So two or three configurations go forward.

**Expansion, per promoted configuration:**
- Seed 0 on all 15 units, X and H: 30 fits. The pilot fits count toward these.
- Seeds 1 and 2 on four units, X and H: 16 fits. The units are V1 fold 1, V2 fold 1, V3 last origin and V4a.

| Outcome | Total fits | At the measured ~40 min/fit |
|---|---|---|
| C2 promoted | 138 | ~92 h |
| C2 rejected | 94 (92 + 2 pilot) | ~63 h |
| Current plan | 420 | ~280 h |

All four designs are kept, and none is primary.

## Test lock

- Run test inference once, after all fits finish and artifact hashes are committed.
- Report every promoted configuration on every design. There is no test-based selection or dropping.
- Disclose that historical fits' test metrics were already seen and played no part in this shortlist.

## Honest reporting

- **Validation reuse:** the same validation data picks the checkpoint and gates C2. Validation numbers are therefore optimistic. Label them selection metrics, never performance estimates.
- **Intervals:** use a paired file-level bootstrap of differences (C1−C0, C2−C0, X−H) within each unit. Across units, give per-design medians, ranges and sign counts. Folds share training data, so do not pool as n=15.
- **Seed variance:** it is measured on four units only. Single-seed intervals exclude it; say so and show the four-unit seed SD beside them.
- **V4a:** it is a single unit, so it is descriptive only.
- **Evidence strength:** this is weaker than the 3-seed plan and must be stated as such. Shortlisting on reused data is exploration, and deferred candidates are "not evaluated," not "worse."

## Precision versus physical value

- C1−C0 measures better use of past measurements. That is a precision gain.
- X−H within a configuration measures what the fast diagnostics add. Only this supports a physical-value statement.
- Report X−H stratified by measurement age and on the held-out every-10th-frame rule.
- L5-head gains are agreement with a future-derived label, not truth.

## Failure tests

- **Anchor:** it fails if the C1−C0 interval includes zero or flips sign in the rolling-origin or chronological designs, even if the random-file design is favorable.
- **Diagnostics:** if the X−H interval includes zero at short age, report the model as a history interpolator.
- **Baselines:** if any configuration fails to beat the boosted-tree/history baseline in a temporal design, report it.

## Protocol to register

1. Configurations C0, C1 and C2 are fixed as above. No hyperparameter changes after commit.
2. Pilot: V3 origin 2, seed 0, X and H, frozen stopping rule. Eligibility is exit 0, checks and finite losses.
3. C2 is promoted if and only if it meets the validation margin against C0-X. C0 and C1 are always promoted.
4. Expansion: 30 + 16 fits per promoted configuration, so at most 138 in total.
5. Ranking uses equal-rule/head validation MSE only. Test is locked until the hashes are committed, then run once.
6. Report all four designs, paired file-bootstrap differences, the four-unit seed SD and the stated limitations.
7. Deferred candidates re-open only through the listed triggers, each by a new preregistration.
8. Queued supervisors for the 210 + 120 plan are superseded. Frozen and completed fits remain historical.
