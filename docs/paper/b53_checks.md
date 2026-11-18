# B.53 checks before any GPU run

Registered: ces_prediction/experiments/PREREGISTRATION_B53.md (section 3). Written by ces_prediction/experiments/b53_components/check_b53.py on 2026-11-18 12:16; CPU only, no network trained. Real data: data/s*.csv (641 files).

## (1) The extra channels reach the T_i branch and only it

| table | K | T_i encoder layer-0 W_ih | V_rot encoder layer-0 W_ih | parameters (v2h 347,970) | T_i output change when the extras are perturbed | V_rot output change | output change when the fast channels are perturbed |
|---|---|---|---|---|---|---|---|
| C_s42 | 1 | (640, 8) (v2h (640, 7)) | (256, 7) (v2h (256, 7)) | 348,610 | 0.0389 | 0 | 0 |
| D_s42 | 4 | (640, 11) (v2h (640, 7)) | (256, 7) (v2h (256, 7)) | 350,530 | 0.156 | 0 | 0 |
| CD_s42 | 5 | (640, 12) (v2h (640, 7)) | (256, 7) (v2h (256, 7)) | 351,170 | 0.145 | 0 | 0 |

- **(1) routing**: PASS -- T_i layer-0 input 7 -> 7 + K for K = 1, 4, 5; V_rot branch shapes equal to v2h; perturbing the extras moves T_i and leaves V_rot bit-identical; perturbing BES/ECEI/MC changes nothing; K = 0: weights under one seed equal to v2h True, forward equal True; table columns in load_grid_files vs table max |diff| 0, in the block x[:, 22:] 0 (unscaled, exempt); v2hx in the registry dict of train_seq and eval_seq True
Code path: seq_data.load_grid_files appends the table after the base columns (seq_data.py 164-167, _load_extra_vt 76-108, every channel exempt so lines 102-107 scale nothing); build_blocks writes it to x[:, 22:] (seq_data.py 246-248, 286-287); v2hx reads x[..., 15:22] and x[..., 22:22+K] into lstm_ti and x[..., 15:22] into lstm_vt (model_seq_v2hx.py forward). v2h itself (model_seq_v2h.py 28-40) passes n_extra_ti=0 and reads x[..., 15:22] only, so the table would not reach it.

## (2) The feature tables match the grid rows of every file

| table | files | rows | time equal to the grid (rows) | seq_data matcher hits | zero rows (non-positive BES) | size on disk (MB) |
|---|---|---|---|---|---|---|
| C_s42 | 641 | 247,207 | 247,207 | 247,207 | 759 | 3.1 |
| D_s42 | 641 | 247,207 | 247,207 | 247,207 | 759 | 6.0 |
| CD_s42 | 641 | 247,207 | 247,207 | 247,207 | 759 | 6.9 |
| C_s1 | 641 | 247,207 | 247,207 | 247,207 | 759 | 3.1 |
| D_s1 | 641 | 247,207 | 247,207 | 247,207 | 759 | 6.0 |
| CD_s1 | 641 | 247,207 | 247,207 | 247,207 | 759 | 6.9 |
| C_s7 | 641 | 247,207 | 247,207 | 247,207 | 759 | 3.1 |
| D_s7 | 641 | 247,207 | 247,207 | 247,207 | 759 | 6.0 |
| CD_s7 | 641 | 247,207 | 247,207 | 247,207 | 759 | 6.9 |
| C_s123 | 641 | 247,207 | 247,207 | 247,207 | 759 | 3.1 |
| D_s123 | 641 | 247,207 | 247,207 | 247,207 | 759 | 6.0 |
| CD_s123 | 641 | 247,207 | 247,207 | 247,207 | 759 | 6.9 |

- **(2) grid rows**: PASS -- every table has 641 files whose times equal the grid times of seq_data.load_grid_files row for row (100 %; seq_data aborts below 99 %); zero rows are exactly the rows with a non-positive BES channel

## (3) The constants of a split come from its training discharges only

| split | train / val / test files | overlap train with val or test | rows | max |mean - recomputed| | max |std - recomputed| | max |table - (x - mean)/std| | mean of c with test added minus train-only | with val added |
|---|---|---|---|---|---|---|---|---|
| 42 | 415 / 127 / 96 | 0 | 155,466 | 0 | 0 | 0 | +0.0142 | +0.0305 |
| 1 | 414 / 128 / 96 | 0 | 154,840 | 0 | 0 | 0 | -0.0207 | -0.0086 |
| 7 | 414 / 128 / 96 | 0 | 155,765 | 0 | 0 | 0 | +0.0000 | -0.0023 |
| 123 | 414 / 128 / 96 | 0 | 156,049 | 0 | 0 | 0 | +0.0025 | -0.0038 |

- **(3) constants**: PASS -- recomputed from the manifest train_files alone they equal the stored constants and rebuild every table exactly; adding the test (or validation) discharges moves them, so no other discharge entered

## (4) With no extra table the runner reproduces the env of the B.39 twin

| split | B.39 twin run | variables in the env | env equal (whole dict) | agrees with the twin's metrics.json | variables that change when table D is set (same output dir) |
|---|---|---|---|---|---|
| 42 | .b29a1_v2h_cut_s42 | 118 | True | True (epochs run 49, best 43) | CES_SEQ_EXTRA_TI, CES_SEQ_MODEL |
| 1 | .b29a1_v2h_cut_s1 | 118 | True | True (epochs run 46, best 40) | CES_SEQ_EXTRA_TI, CES_SEQ_MODEL |
| 7 | .b29a1_v2h_cut_s7 | 118 | True | True (epochs run 11, best 5) | CES_SEQ_EXTRA_TI, CES_SEQ_MODEL |
| 123 | .b29a1_v2h_cut_s123 | 118 | True | True (epochs run 58, best 52) | CES_SEQ_EXTRA_TI, CES_SEQ_MODEL |

- **(4) twin env**: PASS -- c_env(None, ...) == run_b39.split_env('v2h', s, s, twin dir) for all four splits; with a table only CES_SEQ_EXTRA_TI and CES_SEQ_MODEL (v2h -> v2hx) change

## (5) Arm F differs from B.39 only in the epoch budget and the output directory

| run | B.39 reference | non-CES variables equal | CES variables that differ |
|---|---|---|---|
| .b53F_v2_cut_s42_i42 | .b39_v2_cut_s42_i42 | True | CES_OUTPUT_DIR: .b39_v2_cut_s42_i42 -> .b53F_v2_cut_s42_i42; CES_SEQ_EPOCHS: 80 -> 40; CES_SEQ_FIXED_EPOCHS: 0 -> 1 |
| .b53F_v2h_cut_s42_i42 | .b29a1_v2h_cut_s42 | True | CES_OUTPUT_DIR: .b29a1_v2h_cut_s42 -> .b53F_v2h_cut_s42_i42; CES_SEQ_EPOCHS: 80 -> 40; CES_SEQ_FIXED_EPOCHS: 0 -> 1 |
| .b53F_v2_cut_s1_i1 | .b39_v2_cut_s1_i1 | True | CES_OUTPUT_DIR: .b39_v2_cut_s1_i1 -> .b53F_v2_cut_s1_i1; CES_SEQ_EPOCHS: 80 -> 40; CES_SEQ_FIXED_EPOCHS: 0 -> 1 |
| .b53F_v2h_cut_s1_i1 | .b29a1_v2h_cut_s1 | True | CES_OUTPUT_DIR: .b29a1_v2h_cut_s1 -> .b53F_v2h_cut_s1_i1; CES_SEQ_EPOCHS: 80 -> 40; CES_SEQ_FIXED_EPOCHS: 0 -> 1 |
| .b53F_v2_cut_s7_i7 | .b39_v2_cut_s7_i7 | True | CES_OUTPUT_DIR: .b39_v2_cut_s7_i7 -> .b53F_v2_cut_s7_i7; CES_SEQ_EPOCHS: 80 -> 40; CES_SEQ_FIXED_EPOCHS: 0 -> 1 |
| .b53F_v2h_cut_s7_i7 | .b29a1_v2h_cut_s7 | True | CES_OUTPUT_DIR: .b29a1_v2h_cut_s7 -> .b53F_v2h_cut_s7_i7; CES_SEQ_EPOCHS: 80 -> 40; CES_SEQ_FIXED_EPOCHS: 0 -> 1 |
| .b53F_v2_cut_s123_i123 | .b39_v2_cut_s123_i123 | True | CES_OUTPUT_DIR: .b39_v2_cut_s123_i123 -> .b53F_v2_cut_s123_i123; CES_SEQ_EPOCHS: 80 -> 40; CES_SEQ_FIXED_EPOCHS: 0 -> 1 |
| .b53F_v2h_cut_s123_i123 | .b29a1_v2h_cut_s123 | True | CES_OUTPUT_DIR: .b29a1_v2h_cut_s123 -> .b53F_v2h_cut_s123_i123; CES_SEQ_EPOCHS: 80 -> 40; CES_SEQ_FIXED_EPOCHS: 0 -> 1 |

- **(5) fixed-budget env**: PASS -- for all eight runs exactly CES_SEQ_EPOCHS (80 -> 40), CES_SEQ_FIXED_EPOCHS (0 -> 1) and CES_OUTPUT_DIR differ; train_seq.py 198-203 / 310-315: with CES_SEQ_FIXED_EPOCHS=1 no early stop and the final weights are saved

## Result

All five checks pass.
