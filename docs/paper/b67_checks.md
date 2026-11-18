# B.67 checks before any GPU run

Registered: ces_prediction/experiments/PREREGISTRATION_B67.md (section 3). Written by ces_prediction/experiments/b67_gated/check_b67.py on 2026-11-19 02:52; CPU only, no network trained (randomly initialised v2s / v2a weights where a checkpoint is needed). Real data: data/s*.csv.

## (a) Dry run of each arm

| arms | return code | runs listed | runs per arm |
|---|---|---|---|
| S | 0 | 15 | 15 runs: S 15 |
| A | 0 | 15 | 15 runs: A 15 |
| S A | 0 | 30 | 30 runs: S 15, A 15 |

- **(a) dry run**: PASS -- each arm alone and both together list 15 runs per arm (30 in all) and start nothing

## (b) Forward shapes, parameter counts, source groups

| quantity | v2s | v2a | v2 (reference) |
|---|---|---|---|
| output shape for x (3, 64, 22) | (3, 64, 2) | (3, 64, 2) | (3, 64, 2) |
| parameters (model.n_params / counted) | 430,758 / 430,758 | 369,318 / 369,318 | 357,570 / 357,570 |
| gates shape; max abs(sum of gates - 1) | (3, 64, 4); 1.2e-07 | (3, 64, 4); 1.2e-07 | - |
| T_i LSTM input weight (4 x 160, input width) | (640, 128) | (640, 32) | (640, 22) |
| forward == forward_with_gates[0]; state dict reloads (strict) and gives the same output | True, True | True, True | - |
| gate output layer zeroed: max abs(gate - 0.25); max abs(LSTM input - plain concatenation (S) / mean (A) of the embeddings) | 0.0e+00; 0.0e+00 | 0.0e+00; 0.0e+00 | - |

Source groups: BES x[..., 0:9], ECEI x[..., 9:13], Mirnov x[..., 13:15], CES-derived x[..., 15:22]. CSV columns of the first file: 9 BES_*, 4 ECEI_*, 2 MC* in that order; grid dims {'bes': 9, 'ecei': 4, 'mc': 2, 'n_fast': 15, 'extra_vt': 0, 'extra_ti': 0}; model_seq_v2sub.GROUPS {'bes': (0, 1, 2, 3, 4, 5, 6, 7, 8), 'ecei': (9, 10, 11, 12), 'mc': (13, 14)}. On the first block of s30801.csv (33 rows) the 22 channels of seq_data.build_blocks are: the BES, ECEI and MC columns standardized with the network's training statistics (x[:, 0:15]), log1p of the step (x[:, 15]), and the T_i staleness log1p(t - last measured T_i) (x[:, 17]); the has-flags (x[:, 18], x[:, 21]) are 0 or 1: True.

- **(b) shapes and parameters**: PASS -- v2s 430,758 and v2a 369,318 parameters (< 1,000,000; seq_v2 357,570); outputs (B, L, 2), gates (B, L, 4) summing to one; the groups are BES 0-8, ECEI 9-12, Mirnov 13-14, CES-derived 15-21 of the real layout; uniform gates give the plain concatenation (S) / the mean (A)

## (c) Causality

| model | mode | lengths | outputs and gates at rows <= t unchanged (max abs diff) | later outputs changed | gates of other rows unchanged when only row t is perturbed | V_rot output unchanged when every fast channel is perturbed |
|---|---|---|---|---|---|---|
| v2s | eval | whole block | True (0.0e+00) | True | True | True |
| v2s | eval | packed [64, 55, 35] | True (0.0e+00) | True | True | True |
| v2s | train | whole block | True (0.0e+00) | True | True | True |
| v2s | train | packed [64, 55, 35] | True (0.0e+00) | True | True | True |
| v2a | eval | whole block | True (0.0e+00) | True | True | True |
| v2a | eval | packed [64, 55, 35] | True (0.0e+00) | True | True | True |
| v2a | train | whole block | True (0.0e+00) | True | True | True |
| v2a | train | packed [64, 55, 35] | True (0.0e+00) | True | True | True |

- **(c) causality**: PASS -- with the inputs after row t = 29 replaced, every output and gate at rows 0..29 is identical (bit for bit), in eval mode and in train mode (dropout masks drawn from the same seed), on whole blocks and packed sequences; a row's gates change only when that row changes; the V_rot output does not move when the 15 fast channels are replaced

## (d) The wrapper registers the variants for training and for scoring

| quantity | value |
|---|---|
| references to train_seq SEQ_MODELS | [('<module>', 'IMPORT_FROM'), ('<module>', 'STORE_NAME'), ('main', 'LOAD_GLOBAL')] |
| references to eval_seq SEQ_MODELS | [('<module>', 'IMPORT_FROM'), ('<module>', 'STORE_NAME'), ('build_pred_lookup', 'LOAD_GLOBAL')] |
| references to eval_seq load_grid_files | [('<module>', 'IMPORT_FROM'), ('<module>', 'STORE_NAME'), ('build_pred_lookup', 'LOAD_GLOBAL')] |
| references to eval_seq build_clean_val_subset | [('<module>', 'IMPORT_FROM'), ('<module>', 'STORE_NAME'), ('compare', 'LOAD_GLOBAL')] |
| train_b67.main() under the env of S r2 f0 (CPU, 2 training files), stopped at the first training pass | {'reached': True, 'variant': 'v2s', 'class': 'SeqCESLSTMv2Gate', 'mode': 'S', 'n_params': 430758, 'train_blocks': 2, 'registry_is_train_seq_registry': True, 'written': []} |
| train_b67.main() under the env of A r2 f0 (CPU, 2 training files), stopped at the first training pass | {'reached': True, 'variant': 'v2a', 'class': 'SeqCESLSTMv2Gate', 'mode': 'A', 'n_params': 369318, 'train_blocks': 2, 'registry_is_train_seq_registry': True, 'written': []} |
| eval_b67.main() with v2a (random weights) under the env of A r2 f0 (CPU), through eval_seq.compare up to the prediction lookup of two test files | {'reached': True, 'variant': 'v2a', 'files': ['s30805.csv', 's30811.csv'], 'entries': 484, 'finite': True, 'registry_is_eval_seq_registry': True, 'grid_calls': 1, 'dataset_calls': 1} |

- **(d) registry seen by training and scoring**: PASS -- train_seq (main) and eval_seq (build_pred_lookup) resolve CES_SEQ_MODEL through the module global SEQ_MODELS, the dict register() inserts into; under the run's env train_b67.main() builds a SeqCESLSTMv2Gate of mode S / A (stopped before any optimisation step, nothing written) and eval_b67.main() predicts with v2a through eval_seq's own lookup; eval_seq reaches load_grid_files and build_clean_val_subset only through the module globals that the gate readout's captures wrap (each called once); the end-to-end scoring with v2s is in (f)

## (e) The 30 planned environments equal those of B.59 arm N apart from CES_SEQ_MODEL and the directories

| arm | runs | variables that differ from B.59 arm N of the same fold (whole env) | differ from the B.39 network of the fold | B.59 arm N metrics.json agree (init and split seeds, standardization, held, fixed epochs, cap) |
|---|---|---|---|---|
| S | 15 | CES_OUTPUT_DIR, CES_SEQ_MODEL | CES_OUTPUT_DIR, CES_SEQ_MODEL | True |
| A | 15 | CES_OUTPUT_DIR, CES_SEQ_MODEL | CES_OUTPUT_DIR, CES_SEQ_MODEL | True |

- **(e) environments**: PASS -- for every fold of r2 to r4, the whole env (CES_* and the rest) of S and of A equals that of B.59 arm N except CES_SEQ_MODEL (v2s / v2a for v2n) and CES_OUTPUT_DIR; against the pooled network of B.39 likewise; the stored B.59 arm N runs record the same seeds and data treatment

## (f) The gate readout is aligned to the scored rows

| quantity | value |
|---|---|
| B.39 network r2 f0 rescored on the CPU through the gate readout's table: rows / found / squared errors identical to the stored (GPU) ones | 37,483 / 37,483 / 22 |
| same: |difference of the absolute errors| median / 99th pct / max (eV); skill over the stored errors | 0.0295 / 0.175 / 1.7; +3.65e-05 |
| negative control, every row looked up one row later: found / median / 99th pct (eV); skill over the stored errors; check passes | 37,370 / 19.7 / 310; +3.03e-01; False |
| eval_b67.py, v2s with random weights, fold r2 f0 on the CPU (3.0 min): rows / same rows as the B.39 network / reference errors and measured values identical | 37,483 / True / True |
| its gate file: on the scored rows / rows found / squared errors reproduced bit for bit in the scoring process / gates (N, 4), max abs(sum - 1) / check passes | True / 37,483 / 37,483 of 37,483 / (37483, 4), 1.5e-07 / True |
| its gate file: share of rows whose T_i staleness in the network's input is 10 ms (rows right after a measured frame, before excluding rows 1 to 3 after an absent frame) | 0.977 |

- **(f) gate alignment**: PASS -- the gate readout looks rows up under eval_seq's own key: on the stored B.39 network of r2 f0 it reproduces the stored GPU errors within the B.59 tolerance (CPU against cuDNN arithmetic), and shifted by one row it fails; eval_b67.py scores v2s on the network's rows with identical reference errors, and its gate file is on those rows with every squared error reproduced bit for bit

## (g) analyze_b67.py

| run | return code | output written | statements read | last line |
|---|---|---|---|---|
| strict | 1 | False | - | [b67] not every run is present; nothing written (use --partial to read what is complete) |
| partial | 0 | True | {'G1': False, 'G2': False, 'G3': False, 'G4': False} | written C:\Users\lss\Documents\GitHub\ml-intern-revision\data\.b67_check_analyze\partial\b67_gated.md | partial: True | selftest: False |
| selftest | 0 | True | {'G1': True, 'G2': True, 'G3': True, 'G4': True} | written C:\Users\lss\Documents\GitHub\ml-intern-revision\data\.b67_check_analyze\selftest\b67_gated.md | partial: False | selftest: True |

Selftest: S over the twin (= the B.39 network over the twin on r2 to r4) +0.053 [+0.027, +0.076]; G1 (the network over itself) +0.000; G4 on synthetic gates -0.0006.

- **(g) analyze**: PASS -- without --partial it stops and writes nothing; with --partial on zero B.67 runs it exits 0 with every B.67 statement 'not read'; the selftest runs every readout (statements, rows, gate classes and differences)

## (h) No tracked file is changed

`git status --porcelain`: 6 lines; tracked files changed: none; untracked B.67 paths: ['?? ces_prediction/experiments/b67_gated/'].

- **(h) no tracked file changed**: PASS -- only untracked paths; the variants are registered at run time and no shared file is edited

## Result

All checks pass.
