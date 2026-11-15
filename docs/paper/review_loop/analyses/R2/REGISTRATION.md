# Review-loop round 2 analyses R2-a ... R2-e: registration (2026-11-15)

Written and committed before any number below is computed. CPU only, real data and frozen
artifacts only, no neural network is trained. Source of the items: `round_2.md` section 6
(a-d) and `venue_plan/PLAN_review.md` section 5 step 2 (e, the memoryless GBM).

Shared conventions
- Populations: cut (`CES_TI_SPIKE_CUT_EV=3000`) and inclusive (`0`), co-primary. Held-free
  (`CES_DROP_STUCK_TARGETS=1`), W = 2 harness rows, `CES_PER_SHOT_NORM=1`, all pinned.
- Canonical cells: cut `data/.b1_seqv2_s{s}_i{s}`, inclusive `data/.b5i_seqv2_s{s}`,
  s in {42, 1, 7, 123}; manifests `.b1_w2cut_split_s{s}` / `.b1_manifest_s{s}`.
  Row identity (shot, row) of every canonical cell is taken from the stage-4 surrogate file
  `data/.b28_b28h_{pop}_s{s}/comparison_errors_test.npz` (it stores `*_row`), after checking
  that shot ids, dt_ms and the stored PCHIP squared errors are bit-identical to the backbone file.
- Skill of arm A over arm R = 1 - sum SE_A / sum SE_R on identical rows. Per split:
  shot-cluster bootstrap, B = 10,000, seed 12345, percentile 95 %. Pooled over the B.27c
  repartitions: the registered two-level reader `analyze_b27c.hier` (repeats r2-r9, its own
  B = 20,000, seed 2728), as in A1.
- Fit-failure rule F(1000) is the A1 rule, unchanged: a scored V_rot row is dropped from every
  arm when the target, the persistence anchor or any of the up-to-two observed neighbours on
  each side has |V_rot| > 1000 km/s. It is a scoring rule decided post hoc in round 1.
- Results are reported as point estimate + 95 % interval + whether the interval excludes 0.

## R2-a. V_rot running-mean arm under F(1000)

Arm. EWMA of the past observed held-free V_rot values of the same contiguous segment
(gap < 0.5 s), one update per observation, m <- m + alpha (v - m), first observation taken
whole; prediction at a row = the state before that row. alpha grid {0.05, 0.10, ..., 1.00}
(alpha = 1 is persistence). alpha* minimizes the SSE on the VAL files of the same split or fold
(rows: observed target with at least one past observation in the segment, F(1000) applied);
TEST is never used for alpha.
Two input conventions, both scored on the same F(1000) rows:
- gated (PRIMARY): observations with |V_rot| > 1000 km/s never enter the running mean (a
  causal range gate that uses the value itself only). This is the stronger reference, so it
  is the one the decision is read on.
- ungated (sensitivity): every observation enters.
Readouts. (1) Two-level bootstrap over B.27c repeats r2-r9 (the folds store shot/row arrays;
alpha* per fold from that fold's VAL files), both populations: backbone vs EWMA*. (2) Per
split on the 4 canonical splits x 2 populations. Also reported: EWMA* vs persistence, PCHIP,
causal GP; alpha*; rows dropped.
Decision rule. Read on (1), gated arm. If the backbone-vs-running-mean interval includes 0
in either population, Claim 2 narrows to "a running mean of the CES past" (the backbone's
V_rot branch is not shown to add to a one-parameter smoother). If it excludes 0 above 0 in
both, Claim 2 keeps the backbone as ahead of a running mean, with the size stated. If it
excludes 0 below 0 in either, the running mean is the better V_rot estimator and Claim 2 is
restated around it. The per-split count (2) is reported next to it, not used for the decision.

## R2-b. Surrogate with every fast-channel coefficient a_c = 0

Arms on the frozen T_i rows of the 8 canonical cells:
- S0-zero: the trained stage-4 surrogate (`.b28_b28h_{pop}_s{s}`) with a_c := 0, nothing
  refit (K0, K_inf, s_K, b0 as trained jointly with the fast term).
- S0-refit (PRIMARY): the same 4-parameter filter y = m(t) + b0, K(s) = K_inf + (K0 - K_inf)
  exp(-s / s_K), refit on the split's TRAIN files only: masked MSE on observed T_i rows that
  have a past observation, full batch, L-BFGS from the module defaults (K0 0.35, K_inf 0.60,
  s_K 30 ms, b0 0), CPU, no validation-based selection. `SeqCESB28` is used as the forward
  code, so the filter is the surrogate's own.
- Full surrogate (stored errors), backbone (stored errors), causal GP, PCHIP, persistence,
  offline GP (stored errors).
Check before scoring: the unmodified surrogate checkpoint run through this script must
reproduce the stored surrogate squared errors (max abs difference of the prediction < 0.05 eV).
Readouts per split: S0 vs causal GP; full surrogate vs S0; backbone vs S0; S0 vs PCHIP and
persistence; retention of S0 = skill_S0 / skill_backbone against PCHIP and causal GP.
Decision rule. Fast-channel share inside the equation family = skill(full surrogate vs
S0-refit). If its interval excludes 0 above 0 on at least 3/4 cut splits, the surrogate's
fast term is reported as a measured increment of that size over the best 4-parameter CES-only
filter; otherwise the surrogate is described as a CES-only filter whose fast term is not
resolved. Separately, backbone vs S0-refit is reported as the upper estimate of what fast
channels plus the LSTM add over a 4-parameter CES-only filter; if it includes 0 on 2 or more
cut splits, Claim 1's attribution to the fast channels is not supported by this measurement.
This does not replace the history-only seq_v2 (GPU): it attributes within the surrogate.

## R2-c. Table 1 V_rot RMSE column under F(1000)

Seed 42, cut (the table's cell) and inclusive. Arms: causal GP, running mean (R2-a, gated
and ungated), persistence, two-point extrapolation (`ar_local`, recomputed from the row
index), seq_v2, surrogate file's V_rot branch, windowed control (`.b1_w2cut_s42` /
`.wsweep_hf_w2_s42`, used only if its rows are bit-identical), offline GP, linear, PCHIP.
RMSE with and without the rule, rows dropped. No decision rule: a re-derivation. Stated
expectation: the ordering of the causal arms is unchanged and every RMSE falls.

## R2-d. Causal-GP arm for the per_shot = 0 artifacts

Artifacts `data/.seq_v2_nops_lstm_s{42,1,7,123}`: EARLIER protocol (runner_common.FULL_ENV:
W = 4 harness rows, test cap 40,000, held-free scoring, no 3 keV cut, no per-file cap of
500; control stats `.sf_iter009_s{s}`, splits `runner_common.SPLIT_SRC`). Their npz files hold
only model, PCHIP and linear errors. The row population is re-derived with the same harness
call as `eval_seq.py` (no model), accepted only if the count, shot ids, dt_ms and the PCHIP
squared errors reproduce the stored arrays; then `gp_causal` and persistence are computed per
row with `baselines_interpolation`. The same arms are attached to the per_shot = 1 sibling
`data/.seq_v2_lstm_s{s}` when its rows are bit-identical.
Readouts per split: per_shot = 0 model vs causal GP, persistence, PCHIP (T_i and V_rot);
per_shot = 1 sibling vs the same; per_shot = 1 vs per_shot = 0 paired.
Decision rule. If the per_shot = 0 T_i interval vs causal GP excludes 0 above 0 on at least
3/4 splits, the statement "the T_i margin over the causal GP does not rest on the offline
per-discharge standardization" holds under the earlier protocol; otherwise it is not
established. In both cases the confirmed-protocol rerun with `CES_PER_SHOT_NORM=0` (GPU)
stays the gate; this item only says what the existing artifacts show.

## R2-e. Memoryless gradient boosting

Library: sklearn `HistGradientBoostingRegressor` (LightGBM is not installed), squared loss.
"The 22 inputs" are the 22 per-step channels of `seq_data.build_blocks`: 15 fast channels,
log1p(dt to the previous grid row), and per target (carried value, log1p staleness, has flag).
Two of the 22 are CES values, so the two variants are:
- GBM-A (no CES value): 20 features = 15 fast + dt + staleness and has flag of both targets
  (staleness is the time since the last CES). Target = the CES value (normalized).
- GBM-B (last observed CES value added): all 22 features. Target = value minus the carried
  value of the same target (increment form; prediction = carried + f(x)), the stronger
  parameterization for trees.
One model per target, variant, split and population. Train rows: every observed held-free
target row with a past observation in the split's TRAIN files (full grid, no window, no
per-file cap). No row sees any other row: no lags, no recurrent state.
Hyperparameters, chosen on VAL files only (same row definition), by VAL MSE: input scaling of
the fast channels {per-discharge z-score as the backbone, train-file global z-score};
max_leaf_nodes {15, 63}; min_samples_leaf {20, 200}; learning rate 0.1; number of trees
1..600 read from staged predictions. random_state = split seed. No TEST quantity enters.
Scoring: the backbone's frozen TEST rows (T_i and V_rot), stored reference errors: skill vs
persistence, PCHIP, causal GP, and paired backbone vs GBM. V_rot is read both as registered
(no rule) and under F(1000).
Decision rule. T_i, GBM-B: if backbone vs GBM-B excludes 0 above 0 on at least 3/4 cut splits,
the backbone's sequence memory adds to a memoryless regressor that sees the same row inputs,
with the size stated; if not, the T_i result is reported as reachable by a memoryless
regressor on fast channels plus the last CES value. GBM-A is the diagnostics-only reference
(the mainstream memoryless design); its skill vs persistence is reported, with the stated
expectation that it is below 0 for both targets. V_rot: reported, same reading, no claim
depends on it beyond R2-a.

Outputs: scripts and json in this folder; one record per item appended to THESIS_RESULTS.md
section 8. The paper is not edited.
