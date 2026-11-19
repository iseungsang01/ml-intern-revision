# B.70 checks (phase 1)

Registered: ces_prediction/experiments/PREREGISTRATION_B70.md (Addendum 1). Written by `ces_prediction/experiments/b70_d1/checks_b70.py report`; the numbers are in `data/.b70/checks.json`.

## (a) The re-scoring path reproduces the stored B.39 network on the old subset

`.b39p_v2_cut_r2_f0` re-scored by its own scorer (seq/eval_seq.py) under `run_b39.pooled_env` with CES_MAX_VAL_SAMPLES=40000 (the stored value), CES_SEQ_DEVICE=cpu and a new output directory, against the stored `comparison_errors_test.npz` (scored on the GPU).

| key | shape | bit-identical | max abs difference | notes |
|---|---|---|---|---|
| CES_TI_dt_ms | [37483] | True | 0 |  |
| CES_TI_is_peak | [37483] | True | 0 |  |
| CES_TI_row | [37483] | True | 0 |  |
| CES_TI_se_gp | [37483] | True | 0 |  |
| CES_TI_se_gp_causal | [37483] | True | 0 |  |
| CES_TI_se_linear | [37483] | True | 0 |  |
| CES_TI_se_model | [37483] | False | 2.68e+03 | absolute error differs by median 2.9e-02, 99th percentile 1.8e-01, max 1.70 (eV or km/s); rows above 0.5: 18; relative MSE difference 3.7e-05 |
| CES_TI_se_pchip | [37483] | True | 0 |  |
| CES_TI_se_persistence | [37483] | True | 0 |  |
| CES_TI_shot | [37483] | True | 0 |  |
| CES_TI_y_true | [37483] | True | 0 |  |
| CES_VT_dt_ms | [16262] | True | 0 |  |
| CES_VT_is_peak | [16262] | True | 0 |  |
| CES_VT_row | [16262] | True | 0 |  |
| CES_VT_se_gp | [16262] | True | 0 |  |
| CES_VT_se_gp_causal | [16262] | True | 0 |  |
| CES_VT_se_linear | [16262] | True | 0 |  |
| CES_VT_se_model | [16262] | False | 53 | absolute error differs by median 2.3e-03, 99th percentile 1.4e-02, max 0.04 (eV or km/s); rows above 0.5: 0; relative MSE difference 3.4e-07 |
| CES_VT_se_pchip | [16262] | True | 0 |  |
| CES_VT_se_persistence | [16262] | True | 0 |  |
| CES_VT_shot | [16262] | True | 0 |  |
| CES_VT_y_true | [16262] | True | 0 |  |

Everything except the network's squared errors is bit-identical: **True**. The network's squared errors are bit-identical: False (the stored run scored the LSTM on CUDA, this one on the CPU; the difference is float32 rounding of the forward pass).

## (b) Every eligible row is kept with CES_MAX_VAL_SAMPLES=0

`dataset.select_seeded_random_indices` returns the list unchanged when the cap is <= 0 (dataset.py:547-549). Eligible samples (eval_samples of the D1 run) against an independent recount from the grid (cut + held removed; a sample = a kept row whose previous kept row is < 0.5 s before it); scored T_i rows against the recount (T_i finite on the row and on the previous kept row); the old 40,000-sample rows are a subset of the D1 rows. Ledger range (1.4, r2/r5/r9 checked): 40,666-52,744.

| fold | eligible samples (D1 / recount) | in ledger range | T_i rows (D1 / recount) | discharges (D1 / recount) | max rows of one discharge (D1 / recount) | old rows, subset of D1 | passes |
|---|---|---|---|---|---|---|---|
| r2f0 | 40,666 / 40,666 | True | 38,107 / 38,107 | 126 / 126 | 1,477 / 1,477 | 37,483, True | True |
| r2f1 | 47,369 / 47,369 | True | 45,341 / 45,341 | 128 / 128 | 1,441 / 1,441 | 38,297, True | True |
| r2f2 | 47,452 / 47,452 | True | 45,872 / 45,872 | 127 / 127 | 832 / 832 | 38,629, True | True |
| r2f3 | 43,593 / 43,593 | True | 42,190 / 42,190 | 127 / 127 | 1,431 / 1,431 | 38,732, True | True |
| r2f4 | 52,744 / 52,744 | True | 51,162 / 51,162 | 127 / 127 | 1,441 / 1,441 | 38,793, True | True |
| r3f0 | 42,326 / 42,326 | True | 40,503 / 40,503 | 127 / 127 | 1,431 / 1,431 | 38,283, True | True |
| r3f1 | 46,269 / 46,269 | True | 44,703 / 44,703 | 126 / 126 | 1,432 / 1,432 | 38,646, True | True |
| r3f2 | 53,106 / 53,106 | False | 51,328 / 51,328 | 128 / 128 | 1,441 / 1,441 | 38,682, True | True |
| r3f3 | 45,359 / 45,359 | True | 42,992 / 42,992 | 128 / 128 | 1,477 / 1,477 | 37,934, True | True |
| r3f4 | 44,764 / 44,764 | True | 43,146 / 43,146 | 126 / 126 | 831 / 831 | 38,547, True | True |
| r4f0 | 45,070 / 45,070 | True | 43,415 / 43,415 | 128 / 128 | 1,426 / 1,426 | 38,519, True | True |
| r4f1 | 44,519 / 44,519 | True | 43,255 / 43,255 | 127 / 127 | 828 / 828 | 38,877, True | True |
| r4f2 | 49,247 / 49,247 | True | 46,831 / 46,831 | 127 / 127 | 1,477 / 1,477 | 38,043, True | True |
| r4f3 | 50,486 / 50,486 | True | 48,665 / 48,665 | 128 / 128 | 1,441 / 1,441 | 38,565, True | True |
| r4f4 | 42,502 / 42,502 | True | 40,506 / 40,506 | 125 / 125 | 831 / 831 | 38,128, True | True |

The ledger's range was measured on r2, r5 and r9 only; a fold of r3 or r4 outside it (r3 f2) is not a failure. The check passes when the D1 counts equal the recount and the old rows are a subset of the D1 rows.

## (c) Every arm of a fold is scored on identical rows

Against the D1 re-scoring of the B.39 network of the fold: same (shot, row) list, bit-identical T_i values and bit-identical reference errors (hold, causal GP, offline GP, PCHIP, linear).

| fold | arms compared | all same rows, values and references |
|---|---|---|
| r2f0 | 8 | True |
| r2f1 | 8 | True |
| r2f2 | 8 | True |
| r2f3 | 8 | True |
| r2f4 | 8 | True |
| r3f0 | 8 | True |
| r3f1 | 8 | True |
| r3f2 | 8 | True |
| r3f3 | 8 | True |
| r3f4 | 8 | True |
| r4f0 | 8 | True |
| r4f1 | 8 | True |
| r4f2 | 8 | True |
| r4f3 | 8 | True |
| r4f4 | 8 | True |

## (d) The refitted second stage reproduces the stored B.64 predictions on the old rows

The B.64 second-stage models were not saved (stage2_b64.fit_fold stores predictions only). They were refitted with stage2_b64's code, model and seed on the unchanged R-fit64 rows and applied to the old B.39 rows exactly as in fit_fold. Tolerance 1e-6 eV.

| fold | network: max deviation (two-stage / own missing-input handling / first stage), eV | fit rows = B.64 | twin: max deviation, eV | fit rows = B.64 | passes |
|---|---|---|---|---|---|
| r2f0 | 0.0e+00 / 0.0e+00 / 0.0e+00 | True | 0.0e+00 / 0.0e+00 / 0.0e+00 | True | True |
| r2f1 | 0.0e+00 / 0.0e+00 / 0.0e+00 | True | 0.0e+00 / 0.0e+00 / 0.0e+00 | True | True |
| r2f2 | 0.0e+00 / 0.0e+00 / 0.0e+00 | True | 0.0e+00 / 0.0e+00 / 0.0e+00 | True | True |
| r2f3 | 0.0e+00 / 0.0e+00 / 0.0e+00 | True | 0.0e+00 / 0.0e+00 / 0.0e+00 | True | True |
| r2f4 | 0.0e+00 / 0.0e+00 / 0.0e+00 | True | 0.0e+00 / 0.0e+00 / 0.0e+00 | True | True |
| r3f0 | 0.0e+00 / 0.0e+00 / 0.0e+00 | True | 0.0e+00 / 0.0e+00 / 0.0e+00 | True | True |
| r3f1 | 0.0e+00 / 0.0e+00 / 0.0e+00 | True | 0.0e+00 / 0.0e+00 / 0.0e+00 | True | True |
| r3f2 | 0.0e+00 / 0.0e+00 / 0.0e+00 | True | 0.0e+00 / 0.0e+00 / 0.0e+00 | True | True |
| r3f3 | 0.0e+00 / 0.0e+00 / 0.0e+00 | True | 0.0e+00 / 0.0e+00 / 0.0e+00 | True | True |
| r3f4 | 0.0e+00 / 0.0e+00 / 0.0e+00 | True | 0.0e+00 / 0.0e+00 / 0.0e+00 | True | True |
| r4f0 | 0.0e+00 / 0.0e+00 / 0.0e+00 | True | 0.0e+00 / 0.0e+00 / 0.0e+00 | True | True |
| r4f1 | 0.0e+00 / 0.0e+00 / 0.0e+00 | True | 0.0e+00 / 0.0e+00 / 0.0e+00 | True | True |
| r4f2 | 0.0e+00 / 0.0e+00 / 0.0e+00 | True | 0.0e+00 / 0.0e+00 / 0.0e+00 | True | True |
| r4f3 | 0.0e+00 / 0.0e+00 / 0.0e+00 | True | 0.0e+00 / 0.0e+00 / 0.0e+00 | True | True |
| r4f4 | 0.0e+00 / 0.0e+00 / 0.0e+00 | True | 0.0e+00 / 0.0e+00 / 0.0e+00 | True | True |

## The shared-reference scorer (Addendum 1, A6) against the arms' own scoring scripts

Fold r2 f0 was scored twice: each arm by its own scoring script (seq/eval_seq.py, eval_b59L.py, eval_b67.py; data/.b70_rescore/) and by `score_b70.py` (rows and references once, the network lookup per arm; data/.b70_rescore_check/). Every key of every arm's npz (both targets) and the metrics json:

| arm | keys | keys not bit-identical | metrics json equal |
|---|---|---|---|
| b39 v2 | 22 | none | True |
| b39 v2h | 22 | none | True |
| b59 L | 22 | none | True |
| b59 N | 22 | none | True |
| b59 E | 22 | none | True |
| b59 B | 22 | none | True |
| b67 S | 22 | none | True |
| b67 A | 22 | none | True |
| b68 X | 22 | none | True |

Passes: **True**. The other 14 folds were scored by `score_b70.py`; r2 f0 keeps the outputs of the arms' own scripts.

## Environment of the re-scored runs

All 135 runs are re-scored under their runner's own env function with exactly CES_MAX_VAL_SAMPLES, CES_SEQ_DEVICE and CES_OUTPUT_DIR changed (runs_b70.execute refuses any other difference); the stored metrics.json agrees with the env on the model, the initialisation seed and the input standardisation: True.

## Summary

| check | passes |
|---|---|
| shared_scorer_equals_own_scripts_r2f0 | True |
| a_references | True |
| b | True |
| c | True |
| d | True |
| env | True |

## Phase 2

Registered: PREREGISTRATION_B70.md, Addendum 2. Each batch's own checks, run on the D1 rows; where a check compared recorded old values it is pointed at a D1 reference (C2). Files: `data/.b70p2/`.

| batch | check | result | passes |
|---|---|---|---|
| B.54 | (a) P1/P2 from the row tables vs analyze_b39.arm_p on the D1 runs | P1 +0.0938 vs +0.0938; P2 +0.0524 vs +0.0524 | True |
| B.54 | (b) every discharge in one test fold per re-partition; (c) network and twin on the same rows, references agree; (d) the value of every row in the file, no first/last grid row | True, True, True | True |
| B.54 | (e) row-independent codes on split 42 (unchanged) | | True |
| B.55 | (a) every network reproduces its D1 errors (B.49 rule) | largest dev 0.0002 eV | True |
| B.55 | (b), (c) raw streams (row-independent) | | True, True |
| B.55 | (d) part G unshifted reproduces P1, P2 of the D1 runs | max diff 2.1e-09 | True |
| B.55 | C4: rows 1-3 after an absent frame by the B.54 walk, equal to run_b54's blip13 flags in every file | 10,878 rows (B.55's own rule 11,079) | True |
| B.56 | (b) D1 rows located, file values within 0.01 eV, filter outputs finite, grid row = B.54's + 1 | 15 of 15 folds | True |
| B.56 | (c) the output for row t does not move when CES values from t on change | 0.0 eV | True |
| B.56 | fits on the training discharges (D1 item 2.9), held repeats removed; parameters written for B.71 | 15 of 15 parameter files; fit files [383, 384]; held values removed from the filter grid 1; unmatched rows 0 | True |
| B.57 | (i) twins reproduce their D1 errors | min share within 0.5 eV 1.0000 | True |
| B.57 | (ii) section-2 values (recorded old numbers; C2: reported, not required) | | False |
| B.57 | (iii) rho_1 = 1 when the next value is replaced by the current one | | True |
| B.60 | percentiles of |e| from the D1 row tables vs computed from the D1 npz directly (C2) | 5 of 5 equal to 1 decimal | True |
| B.60 | addendum: filter rows identical to the D1 row tables; KFX/KF0 medians vs the D1 filter caches | rows True | True |
| B.61 | P1/P2 vs the D1 B.39 values; percentiles vs the D1 B.54 readout; filter rows | P1 True, P2 True; filter rows 15 of 15 | True |
| B.66 | (a) detector = B.54 classes; (b) quiet share split 42; (c) D1 row tables and filter rows; (d) 639 discharges | True, True, True, True | True |
| B.65 (cut) | the row path reproduces the D1 errors of the B.39 network bit for bit, every fold | 15 of 15 folds | True |
| replays | L = 0: network, hold, causal GP within 0.5 eV of the D1 npz (>= 99 % of the fixed rows); rebuilt inputs equal build_blocks on every file (checked in the run) | {'c': 1.0, 'hold': 1.0, 'gp': 0.9995935730349256} | True |

Summary: b54 passes, b55 passes, b56 passes, b57 passes, b60 passes, b61 passes, b66 passes, b65 passes, replay passes.
