# Correctness review: CES interpolation-comparison analysis modules

Scope (READ-ONLY review; a GPU AutoML loop is running):
- ces_prediction/baselines_interpolation.py
- ces_prediction/compare_baselines.py
- ces_prediction/bootstrap_compare.py
- ces_prediction/evaluate.py (build_clean_val_subset refactor only)

Cross-referenced against dataset.py, analyze_gap.py, the pre-registration plan
(.omc/plans/ces-interpolation-comparison-consensus.md), and the existing
tests/test_baselines_interpolation.py.

Verdict: No correctness, leakage, or statistical-validity bugs found. The pre-registered
fairness rules (PR1/PR2/PR4) and the no-drift/no-leakage invariants are implemented faithfully.
Findings below are LOW-severity observations plus one MEDIUM-severity robustness gap in the
bootstrap denominator. Nothing blocks the headline.

## Severity legend
- HIGH   correctness bug / leakage / invalid statistic that would corrupt the headline.
- MEDIUM real defect, edge-case or fragility that can bite under plausible data.
- LOW    minor / cosmetic / defensible-as-is, recorded for completeness.

## 1. baselines_interpolation.py

Leakage - PASS
- build_neighbor_set (baselines_interpolation.py:71-72) removes the target row:
  idx = idx[idx != row_index]. Target own value never enters any predictor. Confirmed by
  the sentinel test (tests/test_baselines_interpolation.py:36-50).
- NaN targets dropped per-target (:74 keep = ~np.isnan(vals)).

Gap refusal mirrors dataset.py block detection - PASS
- contiguous_block_bounds (:53,56) walks while (times[k]-times[k-1]) < gap, gap=0.5 (:39),
  exactly dataset.py:271 (deltas < 0.5). Confirmed by test_gap_refusal_excludes_across_half_second_gap.

PR2 persistence fallback - PASS
- predict_linear (:90-91), predict_pchip (:99-100), predict_gp (:132-133) fall back to
  predict_persistence when no in-block future neighbor exists. pchip also: <2 unique pts ->
  persistence (:104-105); 2 unique or no scipy -> linear (:106-107).

Predictor math - PASS
- persistence (:78-84) most-recent past via argmax(past_times); correct on unsorted arrays.
- linear (:92-93) sort + np.interp; correct.
- pchip (:101-108) de-dups ties via np.unique before PchipInterpolator; no overshoot
  (test_pchip_no_overshoot_on_monotone_step).
- ar_local (:111-123) AR order 1 (two-point local slope), re-filters to past (:114) so causal
  even though it receives past+future neighbors; zero-dt guarded (:120).
- gp (:126-142) Matern nu=1.5 + WhiteKernel, normalize_y; NaN only if sklearn missing.

LOW-1 (baselines_interpolation.py:81-82,115-116): persistence/ar_local return NaN with no past
obs; unreachable on kept rows (kept => window persistence => a past obs in the full-shot block,
a superset of the window). Defensible; the valid &= ~isnan guard is belt-and-suspenders.

LOW-2 (baselines_interpolation.py:46-58): O(block) outward rescan per call, repeated per
(sample,target) in compare. CPU-negligible vs forward pass. No correctness impact.

## 2. compare_baselines.py

Identical-sample reuse of build_clean_val_subset - PASS
- compare (compare_baselines.py:64-66) calls the same helper evaluate.py uses
  (evaluate.py:152-154) with the same seed and max_val_samples default (40000); the seeded
  subsample uses seed+202 inside the helper (evaluate.py:122), identical in both. For
  CES_SPLIT_TAG=val the scored (file,row_index) set is byte-identical to evaluate.py.
  For test it uses manifest[test_files] (pre-registered PR3 population).

Keep mask = target observed AND window persistence available - PASS
- compare_baselines.py:112 ((batch[target_mask]>0.5) & has_obs.cpu()) == evaluate.py:214
  mask[:,t] & persist_obs[:,t], has_obs from the same _persistence_from_history
  (compare_baselines.py:108, evaluate.py:182). PR2 population (not thinned by future-neighbor).

Persistence baseline matches evaluate.py rmse_persistence - PASS
- persistence arm uses window-based last-observed (compare_baselines.py:135 = to_phys of
  _persistence_from_history), the same quantity evaluate.py denormalizes for rmse_persistence
  (evaluate.py:194,200,221). Valid regression cross-check. Other arms read full-shot block
  neighbors (disclosed neighbor-access asymmetry, not a bug).

Physical-unit denorm - PASS
- to_phys = t*std+mean (compare_baselines.py:86-87) == evaluate.py:198-200, applied to model
  output/target/persistence. Non-persistence baselines computed in raw CSV units from
  file_arrays (already physical) and correctly NOT denormalized.

Per-shot squared errors saved with shot id - PASS
- shot_chunks stores path_to_idx[f] (:141), stable int shot id; bootstrap payload (:218-222)
  writes {name}_shot,_dt_ms,_se_model,_se_pchip,_se_linear all masked by valid -> row-aligned.

Gap bins - PASS
- edges = list(BIN_EDGES_MS)+[inf] (:199) reuses analyze_gap.BIN_EDGES_MS (:38); half-open
  binning + final open bin (:201-204) matches analyze_gap.py:156-158; empty bins suppressed (:206).
  dt = target_time - max(past times) per target (:126-128), ms (:147).

LOW-3 (compare_baselines.py:169-171): valid = keep[:,t] & all-arms-not-NaN. On this data
valid == keep[:,t] (no method NaNs on a kept row), so model population == evaluate.py. If a
future method could NaN on a kept row, this &= would silently shrink/de-pair vs evaluate.py.
Recommend asserting valid.sum()==keep[:,t].sum() (or logging the drop) to make drift loud.

LOW-4 (compare_baselines.py:130-132,190): future_neighbor_fraction counts n_future/n_total over
ALL scored samples (not keep/valid, not per-bin), while RMSEs are over valid. PR2 asks for the
fraction among kept points per dt bin; this aggregate is slightly mis-scoped. Diagnostic only,
not the headline; tighten if quoted in the thesis.

LOW-5 (compare_baselines.py:182,212): skill is NaN if mse_head==0 / mb==0. Correct degenerate
handling; a zero-MSE PCHIP would itself be a red flag.

## 3. bootstrap_compare.py

Resamples whole SHOTS (not rows) - PASS
- _per_shot_sums (:26-32) aggregates SE to per-shot sums/counts via np.add.at; _bootstrap draws
  idx = rng.integers(0, n, size=(B,n)) with n = number of distinct shots (:42), summing per-shot
  totals across resampled shots (:43-45). Resampling unit is the shot. Correct (PR4).

Paired SE_model - SE_base - PASS
- se_model and se_base aggregated over the SAME shot vector (same np.unique partition/order); the
  SAME bootstrap idx selects shots for both tot_sm and tot_sb (:43-44). diff=(tot_sm-tot_sb)/tot_cnt
  (:47) is the paired mean. Correct.

Skill CI = percentile; gate = CI lower bound > 0 - PASS
- skill = 1 - tot_sm/tot_sb (:46); CI np.percentile(skill,[2.5,97.5]) (:49);
  pass: bool(skill_lo>0.0) (:57) == PR4 gate. paired-diff CI reported alongside (:50), direction
  consistent (positive skill lo <-> negative diff).

Deterministic seed - PASS
- rng = np.random.default_rng(BOOTSTRAP_SEED=12345) (:23,63); single rng reused across
  splits/targets -> reproducible report. _bootstrap deterministic for a given rng state (verified
  by tests/test_bootstrap_compare.py).

MEDIUM-1 (bootstrap_compare.py:46-47): unguarded division 1 - tot_sm/tot_sb and /tot_cnt.
tot_sb==0 (a resample of all-zero-baseline shots) yields inf/nan skill propagating into
np.percentile. tot_cnt==0 is impossible. For real continuous CES errors tot_sb==0 cannot occur,
so no real-data impact, BUT it would corrupt a degenerate/synthetic case and it is an unguarded
division in a statistic-of-record. Recommend guarding (skip/NaN resamples with tot_sb==0, or
assert sb.sum()>0). The accompanying tests avoid all-zero baselines for this reason (se_base>0).

LOW-6 (bootstrap_compare.py:63): one shared rng stream across splits/targets/baselines (intended,
one canonical seeded stream). A test calling _bootstrap twice must re-seed to compare.

LOW-7 (bootstrap_compare.py:90): small-gap stratum gated on > 50 vs plan min-n; cosmetic.

## 4. evaluate.py build_clean_val_subset refactor

Returns (dataset, val_indices, val_file_ids) - PASS (evaluate.py:123).

val_shots uses val_file_ids; byte-identical to pre-refactor - PASS
- evaluate() reports val_shots: len(val_file_ids) (evaluate.py:203). val_file_ids = dataset file
  indices whose basename is in val_files (:110). The helper selection (:109-122) is the same
  sequence the original inline code performed and that analyze_gap.py:88-95 still performs verbatim:
  basename match -> val_file_ids -> val_indices by sample_file_indices membership ->
  select_seeded_random_indices(..., seed+202). Same inputs/order -> eval_metrics.json byte-identical
  before/after for a fixed seed/model (AC12).
- compare_baselines.py imports/reuses the same helper (compare_baselines.py:35,64): single source
  of truth (Principle 4 / AC3). NOTE: analyze_gap.py:88-95 keeps a functional COPY of the selection
  block (out of scope here) - keep the three in sync on future edits.

## Summary table

| ID    | Sev    | File:line                                   | Issue |
|-------|--------|---------------------------------------------|-------|
| LOW-1 | LOW    | baselines_interpolation.py:81-82,115-116    | persistence/ar_local NaN w/o past obs; unreachable on kept rows |
| LOW-2 | LOW    | baselines_interpolation.py:46-58            | O(block) rescan per call; CPU-negligible |
| LOW-3 | LOW    | compare_baselines.py:169-171                | valid could de-pair vs keep if a method NaNs on kept row; add assert/log |
| LOW-4 | LOW    | compare_baselines.py:130-132,190            | future_neighbor_fraction over all samples, not valid/per-bin |
| LOW-5 | LOW    | compare_baselines.py:182,212                | NaN skill on zero-MSE baseline (correct degenerate handling) |
| MED-1 | MEDIUM | bootstrap_compare.py:46-47                  | unguarded /tot_sb (and /tot_cnt); NaN/inf on all-zero-baseline resample (no real-data impact) |
| LOW-6 | LOW    | bootstrap_compare.py:63                     | shared rng stream across splits/targets (intended) |
| LOW-7 | LOW    | bootstrap_compare.py:90                     | > 50 vs min-n; cosmetic |

No HIGH findings. Leakage, fairness (identical samples / keep mask / persistence cross-check),
PR2 fallback, PR4 shot-clustered paired bootstrap, and the byte-identical refactor are all
implemented correctly.
