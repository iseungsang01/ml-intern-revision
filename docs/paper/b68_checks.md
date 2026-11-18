# B.68 checks before any GPU run

Registered: ces_prediction/experiments/PREREGISTRATION_B68.md (section 3). Written by ces_prediction/experiments/b68_auxtarget/check_b68.py on 2026-11-19 04:04; CPU only (one optimisation step on a tiny batch in (d) and (e)). Real data: data/s*.csv.

Auxiliary-target cache: data\.b68_aux\aux_ti_offline_gp.npz (build_aux_b68.py, built 2026-11-19 03:40:36: 641 discharges, 247,207 grid rows; estimator 78 s wall / 306 s CPU on 4 workers). Rows of grid blocks with a target: 244,419 of 245,928 (0.9939); with a measured T_i: 0.9969; the GP with values on both sides 243,112, the estimator's fallback to the last earlier value 1,307, no value 1,509.

## (a) Dry run

`run_b68.py --dry-run`: return code 0; 15 runs listed (15 runs: X 15); directories .b68_X_cut_r2_f0 .. .b68_X_cut_r4_f4 in the registered order: True.

- **(a) dry run**: PASS -- 15 runs (r2 f0 .. r4 f4, one initialisation per fold), nothing started

## (b) The auxiliary target reproduces the offline-GP reference of B.39 (test rows of r2 f0)

| quantity | value |
|---|---|
| discharges whose harness file array (time, T_i, V_rot) equals the grid rows with >= 1 measured target | 641 of 641 |
| stored test rows (B.39 network r2 f0) / with a measured T_i in the cache equal to the harness's value | 37,483 / 37,483 |
| rows where the cached target equals the estimator re-evaluated on the harness array (bit for bit) | 37,483 |
| rows where (cached target - y_true)^2 equals the stored se_gp (bit for bit); max abs difference of the absolute errors (eV) | 37,483; 0 |
| skill of the cached target over the stored offline GP on these rows (0 = the same errors) | +0.00e+00 |
| row kinds of the stored rows | {0: 37483} (0 = T_i measured, own value withheld) |

- **(b) target = stored offline-GP reference**: PASS -- bit for bit on all 37,483 test rows of r2 f0: the cache was evaluated on exactly the harness's file arrays (all discharges), and its value and squared error are those of the stored offline-GP reference

## (c) Withheld-row rule

| quantity | value |
|---|---|
| sampled rows with a measured T_i and measured values on both sides (seed 68), from 224,398 | 300 |
| target re-evaluated on the unchanged discharge equals the cache | 300 of 300 |
| target unchanged when the row's own T_i is replaced: x1.25 / +1000 eV / NaN | 300 / 300 / 300 of 300 |
| control: target changed when the nearest earlier measured T_i is multiplied by 1.25 | 300 of 300 |
| rows without a measured target, target re-evaluated equals the cache (NaN = NaN) | 100 of 100 |

- **(c) withheld-row rule**: PASS -- the target at a measured row does not depend on that row's value (bit for bit), and it does depend on the neighbouring measured values

## (d) The loss on a real CPU batch

Batch: the first 4 training discharges of r2 f0 (4 blocks, 1,366 rows, 1,214 measured T_i, 1,359 rows with an auxiliary target); statistics fit on the 383 training discharges as train_seq does; initialisation seed 27020.

| quantity | value |
|---|---|
| batched_aux gives train_seq.batched's x, y, mask, lengths (in order and shuffled) | True |
| aux_mask = rows with a cached target; aux = cached target standardized with the run's T_i statistics (max abs diff to the float64 reference) | True (1.0e-07) |
| same seed: the seq_v2 part of SeqCESLSTMv2Aux = plain seq_v2 (keys and values); state_dict keys = seq_v2's; auxiliary keys left out | True; True; ['head_aux.0.bias', 'head_aux.0.weight', 'head_aux.2.bias', 'head_aux.2.weight'] |
| its state dict loads into plain seq_v2 with strict=True; a dict carrying the auxiliary head is refused | True; True |
| main heads = plain seq_v2 output, eval mode / train mode (same dropout seed) | True / True |
| weight 0: loss = seq_v2's masked MSE + 0.1 penalty (bit for bit) | True (0.571522) |
| weight 0.5: loss vs main + 0.5 x auxiliary MSE computed in float64 from the cache, relative difference; auxiliary MSE relative difference | 5.3e-08 (0.902108 vs 0.902108); 5.4e-08 |
| one optimisation step (1 batch): train_seq.masked_pass on seq_v2 vs the wrapper with weight 0: returned values equal; max abs difference of the seq_v2 weights after the step | True; 0.0e+00 |
| same step with weight 0.5: max abs change against weight 0 in the T_i branch / in the V_rot branch | 2.0e-03 / 1.1e-04 |
| validation pass: train_seq.masked_pass (seq_v2) vs the wrapper with weight 0.5 (same seq_v2 weights): returned (main MSE, penalty, main MSE) equal | True (0.571522) |

- **(d) loss**: PASS -- weight 0 reproduces seq_v2's loss and optimisation step; weight 0.5 adds exactly 0.5 x the masked MSE of the auxiliary head on the cached target; early stopping sees the main heads only

## (e) A CPU run of train_b68.py: the checkpoint is plain seq_v2 and eval_seq scores it

| quantity | value |
|---|---|
| train_b68.py (fold r2 f0 env, CES_SEQ_DEVICE=cpu, CES_SEQ_MAX_FILES=2, CES_SEQ_EPOCHS=1; 0.2 min): seq_model / epochs / optimisation steps (training batches) | v2 / 1 / 1 |
| hooks active in the training process: load_grid_files calls / build_blocks calls / auxiliary weight / training rows with a target | 1 / 4 / 0.5 / 466 of 471 |
| the saved checkpoint: keys 24, none of the auxiliary head; loads into SEQ_MODELS['v2'] with strict=True (here and inside train_b68.py) | True; True |
| seq/eval_seq.py unchanged, CES_SEQ_MODEL=v2, whole test split of r2 f0 (3.0 min): rows / same rows as the B.39 network / reference errors and measured values identical / RMSE of the one-step network (eV) | 37,483 / True / True / 401.4 |

- **(e) checkpoint**: PASS -- one CPU optimisation step through train_b68.py; the checkpoint is plain seq_v2 (strict loading), and the unchanged eval_seq.py scores it as v2 on the B.39 network's rows of r2 f0

## (f) The 15 environments

For 15 folds (whole env, CES_* and the rest; B68_* keys aside, since B.59 does not strip them): variables that differ from B.59 arm N of the fold: CES_OUTPUT_DIR, CES_SEQ_MODEL (CES_SEQ_MODEL v2 for v2n); from run_b39.pooled_env('v2') of the fold: CES_OUTPUT_DIR; B68_* variables in the run env with B68_LEFTOVER=1 set in the shell: none; aux_b68.check_env accepts every env.

- **(f) environments**: PASS -- each env is the B.39 network's env of its fold with only the output directory changed, i.e. B.59 arm N's apart from CES_SEQ_MODEL and CES_OUTPUT_DIR

## (h) analyze_b68.py

| run | return code | output written | statements read | last line |
|---|---|---|---|---|
| strict | 1 | False | - | [b68] not every run is present; nothing written (use --partial to read what is complete) |
| partial | 0 | True | {'X1': False, 'X2': False, 'X3': False} | written data\.b68_check_analyze\partial\b68_auxtarget.md | partial: True | selftest: False |
| selftest | 0 | True | {'X1': True, 'X2': True, 'X3': True} | written data\.b68_check_analyze\selftest\b68_auxtarget.md | partial: False | selftest: True |

Selftest (the B.39 network read as X): X1 +0.000 [+0.000, +0.000] (the network over itself); X2 +0.053 [+0.028, +0.076] (= the B.39 network over the twin; registration: +0.053 [+0.027, +0.076]); X3 1.038 [1.003, 1.089] (= the network / causal GP 1.038).

Weighted median (analyze_b68.wmedian, bucketed) against the direct cumulative sums on the causal-GP errors of r2 (191,934 rows, 300 resampling draws, the first with unit weights): equal True; unit weights give the lower plain median: True.

- **(h) analyze**: PASS -- without --partial it stops and writes nothing; with --partial on zero B.68 runs it exits 0 with every statement 'not read'; the selftest reads X1-X3 and the row and median readouts

## (g) No tracked file is changed

`git status --porcelain`: 7 lines; tracked files changed: none; untracked B.68 paths: ['?? ces_prediction/experiments/b68_auxtarget/', '?? docs/paper/b68_checks.md'].

- **(g) no tracked file changed**: PASS -- only untracked paths; the auxiliary head and loss are installed at run time and no shared file is edited

## Result

All checks pass.
