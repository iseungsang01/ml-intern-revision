# B.85 Claude methodological design review

The author explicitly authorized this external review. Only the redacted methodological prompt below was sent to the official Anthropic API; no repository file, identity, raw data, credential, or unpublished numerical result was transmitted. This is not a manuscript acceptance grade.

Model returned: `claude-opus-4-6`. API version: `2023-06-01`. Usage: `{"input_tokens": 1238, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0, "cache_creation": {"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 0}, "output_tokens": 3030, "service_tier": "standard", "inference_geo": "global", "speed": "standard"}`. Stop reason: `end_turn`.

## Exact transmitted prompt

Review a proposed preregistered experiment as a skeptical ML/time-series methods collaborator. Give blocking flaws, exact leakage tests, and feasible fixes BEFORE implementation/training. Do not estimate journal acceptance, claim novelty, or invent results. This summary has no author identity, shot IDs, raw data, or experimental performance results.

Task: one causal estimator of an edge diagnostic's reported temperature from its past measurements and current fast diagnostic channels. Existing condition-specific regression is a comparator. We want one model without condition-specific refitting or post-fit correction.

Data: real previously used campaign data, exploratory rather than untouched confirmation. Full timestamp grid includes unlabeled context. Supervision uses a measured current-frame scalar F and L5, median of the next five nominal 10-ms frames with at least three measured values in the same segment. L5 is a future-derived offline label only, not independent physical truth. Labels appear only in loss/scoring.

Model: two-layer 160-unit unidirectional LSTM, layernorm and 64-unit GELU MLP, two scalar heads F/L5 sharing a state. Input slots: fast channels; log1p elapsed time; last-visible temperature normalized by training-file-only mean/std; log1p age; has-visible flag. Unused channels zeroed. Identical-parameter history-only twin zeros all fast channels; paired initializations/augmentation. No scenario ID, self-prediction feedback, target-row measurement, or future normalization. State resets at original segment boundaries (time gap at least 0.5 seconds). Train/test inference use whole segments.

Availability: one constant rule per segment; a new rule is sampled uniformly for each segment each training epoch, deterministic and paired for twins. Rules: last measurement at least 10,50,100,300,650,1000 ms old; or only values preceding the most recent ALREADY OBSERVED 15-45-ms gap in that segment (before any such gap: no history). Every visible source index must be strictly before the current row, with 3-ms timestamp tolerance. Availability limit is nondecreasing through a fixed-rule segment. Held-out cadence rule (measurements at every tenth nominal frame, past only) is excluded from both training and validation. At test each entire segment is replayed separately under each rule with state reset and unchanged weights. No switching rules mid-state.

Splits: four designs reported side by side, none primary: random whole-file folds, clustered-file folds, rolling-origin, final chronological block. Three initialization seeds, full and twin models. Within each outer fit set reserve 15% files for validation: seeded random for non-temporal designs, latest files for temporal designs; training normalization excludes validation/test. Exact manifests saved and disjoint. Comparators used all original fit files, so network has fewer training files; disclose rather than claiming training budgets identical.

Optimization fixed before readout: AdamW 0.001, weight decay 0.0001, batch16 full segments, clip1, dropout0.1, cap80, plateau scheduler factor0.5 patience2; stop after six non-improving epochs. Average masked MSE over the two heads, plus 0.1 negative-temperature ReLU penalty in normalized units. Select best validation mean MSE with equal weights for seven training rules and two heads. Validation deterministically replays every rule. Held-out cadence never selects settings. One-epoch real-data CUDA smoke is separate, plumbing only.

Checks before test errors: match cached anchors/ages/flags of independently implemented condition-specific pipeline; frozen-stat mutation of all values not yet visible at a sampled scored time by removal and large additive offset must leave the entire input prefix unchanged; future fast-input mutation leaves prefix unchanged; current target excluded; no cross-segment carry; deterministic augmentation; held-out cadence excluded; normalization provenance; disjoint manifests; equal labels; finite forward/backward on real CUDA batch; prediction invariant under mutation of the test label arrays alone. Any failed check aborts.

Evaluation: exact same rows as stored comparator predictions, intersection masks explicit and baseline inventory fixed before scoring. RMSE, MAE, median absolute error, share within 10% on targets above fixed floor, out-of-range predictions, age bins and cold-start rows. Conditional 95% percentile bootstrap: initialization sampled first (paired twins), then shot clusters; discharge resampling beside it, 20000 draws, deterministic per-cell seeds. Fixed fitted weights, no refitting; no multiplicity adjustment and small cluster counts disclosed. Each initialization and its mean reported, no post-hoc ensemble. The model replaces the existing estimator only if fresh-history and delayed/schedule errors meet a prespecified 5% MSE noninferiority margin and held-out cadence improves over its condition-specific history-only comparator in every design; fast-input increment must also persist in time-ordered designs. Otherwise retain comparator and report failures. CPU single-step inference latency is measured separately from preprocessing and actual diagnostic acquisition latency.

Return: (1) blocking design/implementation ambiguities, (2) most important additional tests, (3) minimal pre-readout clarifications, (4) which interpretations remain unsupported even if all thresholds pass. Prioritize a short, actionable response. Distinguish actual leakage from limitations and avoid proposing a large architecture search.

## Response



# Skeptical Methods Review

## 1. Blocking Design / Implementation Ambiguities

### 1a. L5 Label Is Not Independent Ground Truth — This Contaminates Every Conclusion

L5 is defined as the median of the next five nominal frames with ≥3 measured values. This is a **derived future summary of the same sensor** you are trying to predict. It is not an independent physical measurement. The L5 head therefore trains the shared LSTM state to approximate a low-pass-filtered future of the diagnostic itself. Any claim that the model "predicts temperature" is actually "predicts a smoothed future self-reading." Since both heads share hidden state, the L5 objective biases representations used by the F head as well.

**Blocking because:** If you later interpret F-head accuracy as "the model knows the physical temperature," that is unsupported. The dual-head design means you cannot isolate what the F head learned from physics-carrying fast channels versus what it inherited from the L5 objective's implicit smoothing prior.

**Fix:** Clearly pre-commit that L5 is a **denoising surrogate**, not validation against independent truth. Report F-head and L5-head errors separately. Consider an ablation (single F-head only) to quantify how much shared-state L5 training changes F-head performance. This need not be a full architecture search — just one frozen comparison.

---

### 1b. "Same Segment" for L5 Is Ambiguous Under Availability Rules

L5 uses "the next five nominal 10-ms frames with at least three measured values **in the same segment**." But segments end at gaps ≥0.5 s. Near segment boundaries, L5 may be computed from fewer than five frames or be undefined. The proposal does not state:

- What happens when fewer than five qualifying future frames remain in-segment.
- Whether L5 is masked (excluded from loss) or computed from fewer points.
- Whether segment boundaries used for L5 computation are identical to the boundaries used for state resets.

**Blocking because:** If L5 is silently computed across a segment boundary that triggers a state reset, there is a train/inference inconsistency. If L5 is computed from 3–4 frames instead of 5 near boundaries, label noise is non-stationary and concentrated at segment edges — exactly where cold-start evaluation matters most.

**Fix:** Pre-specify: (a) L5 is masked when <5 qualifying future frames remain in-segment, or (b) the minimum count is stated and the reduced-count rows are flagged in evaluation. Confirm segment boundary definitions are identical for state resets and L5 windowing.

---

### 1c. Training-File-Only Normalization Statistics Are Not Fully Specified

"Last-visible temperature normalized by training-file-only mean/std" — but which training files? In rolling-origin and chronological-block designs, the training set changes across folds. The proposal says normalization excludes validation/test, but does not state:

- Whether normalization stats are recomputed per fold.
- Whether they are computed over all measured temperatures or only over rows where measurements are visible under the applied availability rule.
- Whether they are computed once (rule-agnostic) or per-rule.

**Blocking because:** If stats are computed over all measurements but inference only sees rule-filtered measurements, there is a distribution shift in the input. If stats are fixed across folds, rolling-origin designs leak future distributional information.

**Fix:** Pre-specify: stats recomputed per outer fold, over all measured temperatures in training files regardless of availability rule. Log and compare the stats across folds to flag drift.

---

### 1d. Availability Rule "Only Values Preceding the Most Recent Already-Observed 15–45 ms Gap" Is Under-Specified

This rule requires tracking gaps **as they are observed** during sequential replay. But:

- "15–45 ms gap" — is this in elapsed wall-clock time or nominal frame count? Nominal 10-ms frames with jitter could produce wall-clock gaps in this range routinely.
- "Before any such gap: no history" — does this mean the model receives zero temperature history (has-visible=0, age undefined) for all rows until the first qualifying gap, then receives everything before that gap? Or does it receive measurements that arrived before the gap, excluding measurements after the gap?
- What happens if multiple gaps in the 15–45 ms range occur? "Most recent" suggests the window slides forward — but then availability can become **non-monotonic** (a new gap could exclude previously visible points), violating the stated "nondecreasing" constraint.

**Blocking because:** Non-monotonic availability under this rule contradicts an explicit stated invariant. If the implementation silently enforces monotonicity by taking the union, the rule semantics change. This affects both training augmentation and test-time replay.

**Fix:** Provide a worked example with timestamps. Specify whether the visible set is cumulative (union of all pre-gap windows) or only the window before the most recent qualifying gap. If cumulative, say so; if not, remove the "nondecreasing" claim for this specific rule or exclude it.

---

### 1e. The 3-ms Timestamp Tolerance Creates Ambiguity About Strict Causality

"Every visible source index must be strictly before the current row, with 3-ms timestamp tolerance." Does this mean a measurement timestamped up to 3 ms **after** the current row's timestamp can still be treated as "before"? In a 10-ms frame system, 3 ms tolerance means roughly one-third of a frame of potential future information.

**Blocking because:** If the tolerance allows future-timestamped data into the input, the causal claim is invalid. The frozen-stat mutation test checks that "values not yet visible" don't affect the prefix — but if the tolerance is applied before determining visibility, the mutation test itself uses the wrong visibility boundary.

**Fix:** Pre-commit to one of: (a) tolerance means the source timestamp must be ≤ current_timestamp − 3 ms (conservative), or (b) tolerance means |source_timestamp − current_timestamp| ≤ 3 ms is allowed (permissive, but then do not call it "strictly before"). State which interpretation the mutation test uses.

---

## 2. Most Important Additional Leakage / Integrity Tests

### 2a. Cross-Rule State Leakage Within an Epoch

Each training epoch samples one rule per segment. Segments are replayed as full sequences. Confirm: **LSTM hidden state is zeroed between segments within a batch**, not just at segment boundaries defined by time gaps. If segments from the same file are packed into one batch and states carry across segments, the model sees information from a different-rule replay of a neighboring segment.

**Test:** Run two single-segment batches vs. one two-segment batch from the same file. Outputs for the second segment must be bitwise identical. Fail → state leakage.

---

### 2b. Fast-Channel Alignment Audit

Fast diagnostic channels are "current frame." But if fast channels have different acquisition latencies than the edge diagnostic, "current" may actually be future-aligned relative to the temperature measurement's physical timestamp. The proposal has no latency-alignment specification.

**Test:** For each fast channel, compute cross-correlation with the target at lags −3 to +3 frames. If peak correlation is at a negative lag (fast channel leads), the channel may carry future information. Report this table before training.

---

### 2c. Availability-Rule Determinism Under Floating-Point Timestamp Comparison

The gap rules (15–45 ms, 0.5 s) and age computations depend on timestamp arithmetic. Floating-point comparison of timestamps can produce platform-dependent visibility decisions.

**Test:** Replay 100 segments on CPU and GPU, compare visibility masks bit-for-bit. Any discrepancy → round timestamps to integer microseconds before all comparisons.

---

### 2d. Normalization Statistic Leakage via Validation Files in Non-Temporal Designs

For random and clustered folds, 15% of training-set files are reserved for validation. Confirm: normalization stats exclude these validation files, not just test files. The text says "training normalization excludes validation/test" but the implementation must enforce this per fold.

**Test:** Recompute stats with and without validation files for each fold. If they differ by more than a pre-specified tolerance (suggest 0.5%), flag and investigate.

---

### 2e. Label Equality Check Is Necessary But Not Sufficient

"Equal labels" between network and comparator is checked. But the comparator was fit on the full original fit set, while the network excludes validation files. If the comparator's predictions were generated using all fit data, the comparator may have seen test-segment data during its own fitting (depending on the original comparator's protocol). This is not the network's problem, but it makes the comparison asymmetric in a way that could favor either side depending on the comparator's overfitting behavior.

**Test:** Obtain and disclose the comparator's original train/test partition. If it overlaps with any of the network's test sets, flag the affected comparisons as confounded.

---

### 2f. Segment-Boundary Cold-Start Label Density

Under the gap-based availability rule, early-segment rows have no history. These are also the rows most likely to have L5 labels computed from fewer future frames (per issue 1b). Report the fraction of cold-start evaluation rows where L5 is computed from fewer than 5 frames, stratified by availability rule.

---

## 3. Minimal Pre-Readout Clarifications Required

| # | Item | Why It Cannot Wait |
|---|------|--------------------|
| 1 | Exact definition of L5 masking near segment ends | Changes loss surface and evaluation row count |
| 2 | Normalization: per-fold recomputation, measured over which rows | Changes input distribution |
| 3 | 3-ms tolerance direction (conservative vs permissive) | Changes causal claim validity |
| 4 | Gap-rule monotonicity resolution | Changes availability mask for ~1/7 of training augmentations |
| 5 | Whether the comparator's original fit set overlaps any network test set | Changes interpretability of every comparison |
| 6 | State zeroing granularity (per-segment vs per-batch) | Changes whether training is causal |
| 7 | Fast-channel temporal alignment verification method | Changes whether "current frame" is actually current |

All seven must be resolved and the resolution frozen in the preregistration before any test-set readout.

---

## 4. Interpretations That Remain Unsupported Even If All Thresholds Pass

**4a. "The model knows the physical temperature."** Both labels (F and L5) come from the same edge diagnostic sensor. Passing noninferiority against a condition-specific regression of the same sensor validates *self-consistency*, not physical accuracy. If the sensor has systematic biases (e.g., during transients, detachment, or coating degradation), the model will reproduce them.

**4b. "The model generalizes to new operating regimes."** All four split designs use the same campaign. Even the chronological block is within-campaign. Cross-campaign generalization is untested and unsupported.

**4c. "The model eliminates the need for condition-specific treatment."** The noninferiority test is against a condition-specific comparator that had more training data. If the model passes by a small margin, it may be that it implicitly partitions conditions via its hidden state — functionally reimplementing condition-specific behavior. The twin ablation (zeroed fast channels) partially addresses this, but passing the noninferiority threshold does not prove the model works for the *same reasons* as a physics-informed approach. Any mechanistic claim requires further analysis (e.g., gradient attribution to fast channels vs. history features).

**4d. "Held-out cadence improvement implies robustness to arbitrary unseen schedules."** One held-out cadence is one point in a large space of possible schedules. Improvement on this single cadence does not support claims about arbitrary novel measurement patterns.

**4e. "Fast-channel causal contribution."** The twin difference (full minus history-only) is an *association* between fast-channel availability and prediction quality, not a causal estimate in the interventionist sense. If fast channels correlate with operating regime, the full model may simply be better at regime identification, not at using fast-channel physics. This distinction matters if fast channels become unavailable in future campaigns.

**4f. "Bootstrap CIs have nominal coverage."** With three initialization seeds and a small number of shot clusters, the two-stage bootstrap (sample initializations, then clusters) has very limited discrete support at the outer level. Effective coverage will be below nominal. Disclose this; do not interpret CIs as having 95% frequentist coverage.

**4g. "Single-step CPU latency implies real-time feasibility."** Latency measured separately from preprocessing and diagnostic acquisition is a component benchmark, not a system-level feasibility demonstration.
