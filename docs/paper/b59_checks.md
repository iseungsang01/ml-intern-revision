# B.59 checks before any GPU run

Registered: ces_prediction/experiments/PREREGISTRATION_B59.md (section 3). Written by ces_prediction/experiments/b59_inputs/check_b59.py on 2026-11-18 15:16; CPU only, no network trained. Real data: data/s*.csv (641 files).

## (a) Dry run of every arm

| arms | return code | runs listed | runs per arm |
|---|---|---|---|
| L | 0 | 15 | 15 runs: L 15 |
| N | 0 | 15 | 15 runs: N 15 |
| E | 0 | 15 | 15 runs: E 15 |
| B | 0 | 15 | 15 runs: B 15 |
| L N E B | 0 | 60 | 60 runs: L 15, N 15, E 15, B 15 |

- **(a) dry run**: PASS -- every arm alone and all four together list 15 runs per arm (60 in all) and start nothing

## (b) Arm L changes the fast channels and nothing else, in training and in scoring alike

Data treatment of the pooled estimator (fold r2 f0: split .b27_split_r2_f0, CES_PER_SHOT_NORM=0, CES_DROP_STUCK_TARGETS=1, CES_TI_SPIKE_CUT_EV=3000); statistics of its 383 training discharges. Every file of the grid: 641 files (383 train, 128 validation, 128 test), 693 blocks, 245,928 rows.

| quantity | value |
|---|---|
| blocks whose dt, CES-history channels, y, mask and time are identical under shift 1 | 693 of 693 |
| blocks whose fast channels at rows 1.. equal those of the row before (shift 1) | 693 of 693 |
| rows whose fast channels change under shift 1 | 245,235 of 245,928 |
| rows after the first of a block whose fast channels happen to equal the previous row's (no change) | 0 |
| references to build_blocks in train_seq.py | [('<lambda>', 'LOAD_GLOBAL'), ('<module>', 'IMPORT_FROM'), ('<module>', 'STORE_NAME')] |
| references to build_blocks in eval_seq.py | [('<module>', 'IMPORT_FROM'), ('<module>', 'STORE_NAME'), ('build_pred_lookup', 'LOAD_GLOBAL')] |
| shifted blocks returned through train_seq.build_blocks (s30801.csv) and eval_seq.build_blocks (s30805.csv) after install | True |

- **(b) only the fast channels**: PASS -- under shift 1 only x[:, :15] changes, and it equals the row before in every block of every file (train, validation and test files alike); train_seq (the `make` lambda of main) and eval_seq (build_pred_lookup) reach build_blocks only through the module global that fast_shift rebinds, and called through those globals the blocks come back shifted

## (c) With a shift of zero rows the inputs are those of the estimator of B.39

| quantity | value |
|---|---|
| blocks identical (x, y, mask, time; dtype) under shift 0 | 693 of 693 |
| arm L env without B59_FAST_SHIFT (output dir set to the estimator's) equal to run_b39.pooled_env('v2', r, k) | True (15 folds) |
| the stored estimator .b39p_v2_cut_r2_f0: metrics.json agrees with that env | True |
| standardization statistics recomputed here equal those stored by the estimator (atol 1e-6) | True |

- **(c) shift 0 = estimator**: PASS -- with a shift of zero rows every block equals seq_data.build_blocks array by array under the estimator's env and statistics (the scoring with shift 0 in (f) also reproduces the estimator's stored errors)

## (d) The first row of every block keeps its own values

- **(d) first row**: PASS -- 693 of 693 blocks keep the fast channels of their first row under shift 1

## (e) Arms N, E, B: the env of B.39 arm A except for the split, the initialisation and the output directory

| arm | model | runs | variables that differ from B.39 arm A (any split seed) | non-CES variables equal | differ from the pooled estimator of the fold | B.39 arm A metrics.json agrees (model, standardization, held, extras, early stop) |
|---|---|---|---|---|---|---|
| N | v2n | 15 | CES_CONTROL_METRICS, CES_INIT_SEED, CES_OUTPUT_DIR, CES_SEED, CES_SPLIT_DIR | True | CES_SEQ_MODEL | True |
| E | v2e | 15 | CES_CONTROL_METRICS, CES_INIT_SEED, CES_OUTPUT_DIR, CES_SEED, CES_SPLIT_DIR | True | CES_SEQ_MODEL | True |
| B | v2b | 15 | CES_CONTROL_METRICS, CES_INIT_SEED, CES_OUTPUT_DIR, CES_SEED, CES_SPLIT_DIR | True | CES_SEQ_MODEL | True |

- **(e) env of N, E, B**: PASS -- against B.39 arm A only CES_SPLIT_DIR, CES_CONTROL_METRICS (the harness statistics of the split), CES_SEED (the seed that names the split), CES_INIT_SEED and CES_OUTPUT_DIR differ; against the pooled estimator of the same fold only CES_SEQ_MODEL

## (f) Every run is scored on the rows of the estimator of its fold

| quantity | value |
|---|---|
| every run of the four arms: CES variables that differ from the estimator of its fold within CES_SEQ_MODEL, CES_OUTPUT_DIR, B59_FAST_SHIFT | True (60 runs) |
| frozen estimator r2 f0 scored on the CPU through eval_b59L.py, shift 0: rows / same rows as stored / references identical | 37,483 / True / True |
| shift 0: |difference| of the absolute error to the stored (GPU) scoring, median / 99th pct / max (eV); skill over the stored estimator; skill over the causal GP; wrapper calls; minutes | 0.0295 / 0.175 / 1.7; +3.66e-05; +0.0487; 128; 5.9 |
| frozen estimator r2 f0 scored on the CPU through eval_b59L.py, shift 1: rows / same rows as stored / references identical | 37,483 / True / True |
| shift 1: |difference| of the absolute error to the stored (GPU) scoring, median / 99th pct / max (eV); skill over the stored estimator; skill over the causal GP; wrapper calls; minutes | 13.5 / 196 / 781; -7.83e-02; -0.0257; 128; 5.8 |
| B.39 arm A: v2n, v2e, v2b scored on the rows of v2 (4 splits x 3) | 12 of 12 |

- **(f) scored rows**: PASS -- the rows are fixed by the split, the harness statistics, CES_SEED, the window, the caps and the data treatment, which every run shares with the estimator of its fold; scoring through the arm-L path gives the estimator's rows with either shift (shift 0 also reproduces its errors: skill over the stored GPU scoring within 1e-3, 99 % of rows within 0.5 eV; the residue is CPU against cuDNN arithmetic), and the network does not enter the rows (B.39 arm A)

Inferred, not run: that N, E, B runs of the pooled folds will be scored on the estimator's rows rests on the shared env (verified above) and on eval_seq choosing rows without the network (shown on the four splits of B.39 arm A); it is checked again at readout (analyze_b59.py stops if a run's rows differ).

## Result

All checks pass.
