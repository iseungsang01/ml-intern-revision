# B.70 — one data definition (D1) for every number of the paper

Registered 2026-11-19, before any run of this batch. CPU only: stored weights are run forward and stored errors re-read; nothing is trained. The session lead designed the batch. An AI agent implements and runs it under the author's supervision. No existing file is edited except where section 6 says so.

## 1. Why

The author asked that every result of the paper use one data definition. The ledger `docs/paper/dataset_ledger.md` (commit b2b0d85) found 17 distinct treatments among the results the paper cites. The ones that touch the main text:

- re-partitions r2–r9 (network rows), r2–r4 (two-stage estimator, B.59), r3–r9 (filters) and the four splits (B.53, delay replay);
- a random 40,000 scored rows per pooled fold (about 86 % of eligible rows), against every eligible row on the four splits;
- several row definitions (scorer rows, B.30/B.48 replay rows, delay rows);
- epoch caps 80, 40 and 30;
- held values kept by the filters.

The manuscript also states a per-discharge cap of 500 scored rows that the scoring code does not have (`evaluate.py:130`, `eval_seq.py:163`; `CES_MAX_SAMPLES_PER_FILE` is read only by `train.py`).

## 2. The definition D1

1. **Discharges.** The 639 of the pooled universe (`run_b27.stage_folds`). The number with at least one scored row is reported per re-partition.
2. **Grid.** The 10 ms rows of the files without the first and last row. Blocks are cut at gaps of 0.5 s or more, and no estimator crosses a cut.
3. **Values withheld everywhere.**
   - T_i above 3 keV (`CES_TI_SPIKE_CUT_EV=3000`).
   - Held repeats (`_stuck_repeat_mask`).
   - Both apply to training targets, history inputs, normalisation statistics, reference anchors and scored rows. This holds for the networks, the second stage, the filters and every reference.
4. **Splits.** Pooled five-fold cross-validation, re-partitions r2, r3, r4 (15 outer folds, those of B.64).
   - One readout per estimator: the five test folds of a re-partition joined, the point is the mean over the three re-partitions.
   - 95 % two-level interval: a re-partition drawn, then its test discharges resampled; 20,000 draws.
5. **Scored rows (T_i).** The rows of `evaluate.build_clean_val_subset` steps 1–2, with **no random cap** (`CES_MAX_VAL_SAMPLES=0`, which keeps every row, `dataset.py:547-553`):
   - T_i is measured and withheld from the estimator;
   - the previous kept row carries T_i;
   - every reference, including PCHIP and the offline GP, is defined.

   Every estimator of a table is scored on the same rows.
6. **Rows after an absent frame.** The B.54 definition (`run_b54.py:118-137`): rows 1–3 after a 15–45 ms step, with five regular rows and at least three finite T_i before.
7. **Training.** The stored networks: epoch cap 80, early stopping (patience 6, best-validation checkpoint), one initialisation per fold (`CES_INIT_SEED = 27000 + 10 r + k`), `CES_PER_SHOT_NORM=0`, fast-channel and target statistics from the training files.
8. **Second stage.** As registered in B.64: fitted on the out-of-fold first-stage estimates of the non-test discharges (R-fit64 rows), applied to the D1 rows. The fit standardises over all non-test files including validation; this is stated in the text.
9. **Filters (KF0, KFX).** Refitted on the training discharges of each r2–r4 fold, held repeats removed as in item 3. The filter's own `iloc[1:-1]` loader stays.

## 3. What is re-read

**Phase 1 (core numbers; the statements of section 4 rest on it).**

- (a) Re-score every stored network of the r2–r4 folds on the D1 rows with `eval_seq.py`, `CES_MAX_VAL_SAMPLES=0`, `CES_SEQ_DEVICE=cpu`, everything else from the run's own env:
  - B.39 arm P network and twin (`.b39p_v2_cut_r{r}_f{k}`, `.b39p_v2h_cut_r{r}_f{k}`; 30);
  - B.59 arms (60);
  - B.67 S and A (30);
  - B.68 (15).

  The references (hold, causal GP, PCHIP, offline GP) come out of the same scorer on the same rows. Each re-scored run is written to a new directory; the stored runs are not touched.
- (b) Apply the B.64 second stage to the D1 rows, for the network and for the twin. If the fitted second-stage models were not saved, refit them with `stage2_b64.py`'s code and seeds on the unchanged R-fit64 rows. Then check that the refit reproduces the stored B.64 predictions on the old 40,000-row subset (tolerance 1e-6 eV).
- (c) The main table on D1, for the six models (two-stage estimator, network, twin, causal GP, hold, offline GP) and PCHIP:
  - MSE, RMSE and median absolute error;
  - paired MSE skill (reported also as MSE reduction in %);
  - paired RMSE difference.

**Phase 2 (the other readouts the paper cites, on D1 rows and the r2–r4 folds).**

- B.54 row readouts, with the D1 definition of rows after an absent frame.
- B.55 sample timing: forward runs with the fast channels one row old, on the r2–r4 networks.
- B.57 yardsticks, B.60 relative errors, B.61 absolute and robust scores, B.66 regimes.
- B.56 filters refitted per item 2.9.
- B.65 on the cut population: score its inclusive networks with the cut harness.
- Delay replay with L = 1, 2 frames and the 50 ms thinning replay, as forward runs of the r2–r4 networks with the history rebuilt as in B.42 and B.48, on one fixed row set for all L. The fixed row set is the D1 rows whose history holds a value at least L + 1 frames old for L = 2.

Results that D1 cannot reach without training are not re-read. They stay in the supplement only, each with its treatment named in one table:

- B.53 (four splits, fixed 40 epochs);
- the four-split readouts;
- the time split;
- the networks with cap 30.

## 4. Statements (D1, T_i, cut population)

- **D1-1.** Two-stage estimator over the causal GP: the interval is above zero.
- **D1-2.** Two-stage estimator over the two-stage twin: the interval is above zero.
- **D1-3.** Network over the twin: the interval is above zero.
- **D1-4.** Network over the causal GP: the interval is above zero.
- **D1-5.** Median absolute error, two-stage estimator over hold: the interval of the ratio is below 1.
- **D1-6 (sensitivity of the outlier rule).** D1-1 and D1-2 also hold when the scored rows that a good-and-bad Gaussian mixture flags are removed. The mixture follows Box and Tiao (1968) as used for KSTAR profile fitting by Kim et al. (2026, Nucl. Fusion 66 096038):
  - Residual r = log T_i − the centred running median of log T_i over ±5 rows of the same block.
  - Two zero-mean Gaussians: a narrow good one and a wide bad one, plus the bad fraction. They are fitted by EM on the residuals of the training discharges of each fold.
  - A row is flagged when its posterior bad probability exceeds 0.5.
  - Also reported: the flagged count; how many flagged rows are above 3 keV, below 75 eV, or single isolated frames; and the share of the 1,197 values above 3 keV that the mixture flags when run before the cut.
  - The flag uses both neighbours. It is a scoring-only check, not an input rule.

Also reported, without a statement:

- every main-text number on D1 next to its old value, with the difference;
- the number of scored rows and discharges per re-partition;
- the largest number of scored rows of one discharge.

## 5. What goes into the paper

- Every number in the main text and the supplement comes from the D1 outputs (`docs/paper/b70_d1.{md,json}` and the phase-2 files), through the table generator.
- The sentence on a per-discharge cap of 500 is replaced by the D1 row rule of item 2.5.
- A statement whose verdict changes under D1 is rewritten with the D1 verdict, and the change is named in `PAPER_DESIGN.md` 11.
- If D1-6 fails, the outlier rule is not changed silently. The paper states that the result depends on which values are treated as fit failures, and the next GPU batch retrains under the mixture rule.
- `PAPER_DESIGN.md` 3 and 4 are rewritten to D1 when this batch is registered, since that is the definition, not a result.

## 6. Expectations

- D1-1 to D1-5 hold.
- Points within 0.01 of the r2–r4 values of B.64, since the old subset is about 86 % of the D1 rows.
- D1-6 holds.
- The mixture flags most values above 3 keV and about one to two thousand rows below them.

## 7. Limits known in advance

- Three re-partitions instead of the eight of B.39, so the intervals of the network rows widen.
- One initialisation per fold.
- The 3 keV threshold is set from the data, not from a published rule:
  - the 99th percentile of measured T_i is 2,089 eV;
  - the values above 3 keV are 0.53 % of measured T_i;
  - 77 % of them are single frames, with median 14 times their neighbours;
  - they are 3.18 times as frequent in the weakest BES quintile (B.43 F1).

  The literature supports the reading that KSTAR CES outliers are failed spectral fits (Kim et al. 2024, Nucl. Fusion 64 106052). D1-6 is the check against a published statistical rule.

## Addendum 1 (2026-11-19, before the phase-1 batch and before any readout)

Written after the step-1 timing run (`.b39p_v2_cut_r2_f0` re-scored on all eligible rows, 447 s) and the check re-scorings of fold r2 f0 (step 2), and before the other folds were re-scored. No statement, table value or mixture fit had been computed. Facts and choices only.

**A1. Re-scoring path (section 3 (a)).**
- Each stored run is re-scored by its own batch's scoring script: `seq/eval_seq.py` (B.39, B.59 N/E/B, B.68), `b59_inputs/eval_b59L.py` (B.59 L), `b67_gated/eval_b67.py` (B.67).
- Env: the runner's own env function (`run_b39.pooled_env`, `run_b59.arm_env`, `run_b67.arm_env`, `run_b68.arm_env`) with exactly `CES_MAX_VAL_SAMPLES=0`, `CES_SEQ_DEVICE=cpu` and `CES_OUTPUT_DIR` changed (`b70_d1/runs_b70.py` refuses any other difference). Not `CES_*`: `CUDA_VISIBLE_DEVICES=""`, `PYTHONUNBUFFERED=1`.
- The stored run's `metrics.json`, weights and sidecar json (`b59_shift.json`, `b68_aux.json`) are copied into `data/.b70_rescore/<run name without the dot>/`, which the script reads and writes. The stored directories are only read.
- The stored runs scored the LSTM on CUDA, the re-scoring runs it on the CPU. Rows, values and reference errors are bit-identical; the network's squared errors differ by float32 rounding. Check (a) reports the size.
- The scorer also scores V_rot as before; phase 1 reads T_i only. B.67 arm S also writes its gate readout into the new directory; phase 1 does not read it.

**A2. Second stage (section 3 (b), item 2.8).**
- B.64 did not save the fitted models (`stage2_b64.fit_fold` stores predictions only). They are refitted with `stage2_b64`'s code: `run_b37.gbr` (fixed `random_state`), the fifteen inputs, the unchanged R-fit64 rows, `OMP_NUM_THREADS=4` as in B.64.
- Check (d): the refit is applied to the old B.39 rows exactly as in `fit_fold` and must reproduce the stored two-stage predictions within 1e-6 eV.
- On the D1 rows the first-stage estimate is rebuilt by `fit_fold`'s rule, y + sign(CPU prediction - y) x |error|, with the error of the D1 re-scoring.
- B.64 addendum 1 is kept: where any of the network's fifteen inputs is missing, both arms keep the first stage. The boosted model's own handling of missing inputs is computed beside it and reported only.
- KFX is a phase-2 estimator; the B.64 pairs with KFX are not in the phase-1 tables.

**A3. Mixture of D1-6.**
- Residual r = ln T_i - median of ln T_i over the finite values among the 11 grid rows centred on the row (±5 rows of 10 ms, the row included).
- Window edges: the window is truncated at the block edges, with no padding and no reflection. Blocks are cut at a time step >= 0.5 s (or <= 0), as in `seq_data`.
- The residual is defined when the window holds at least three finite values. Otherwise the row enters no fit and is never flagged; the number of such rows is reported. The first and last grid row of each file are excluded.
- EM fit: two zero-mean Gaussians and the bad fraction w.
  - Start: s_g = 1.4826 x median |r|, s_b = 10 s_g, w = 0.01.
  - Stop: relative change of the log-likelihood <= 1e-12, or 5,000 iterations.
  - "Bad" is the wider component.
- The fit uses the residuals of the fold's training discharges (`train_files` of the manifest; not validation, not test). A scored row of the fold's test discharges is flagged when its posterior bad probability exceeds 0.5.
- D1-6 holds when the D1-1 and the D1-2 intervals, on the D1 rows without the flagged rows, are both above zero.
- "Run before the cut": the same procedure on the inclusive grid (`seq_data.load_grid_files` with cut 0; held repeats removed as everywhere). The fit is per fold on its training discharges; each value is judged by the fold in which its discharge is a test discharge. Counts are reported per re-partition. The number of measured values above 3 keV on that grid is checked against 1,197.
- A "single isolated frame" is a flagged value whose previous and next measured values in its block are not flagged; a block edge counts as not flagged. "Below 75 eV" means T_i < 75 eV.

**A4. Intervals.**
- Sums (MSE skills, RMSE, RMSE differences, Huber and trimmed skills, shares): the two-level bootstrap of `analyze_b39.arm_p` (a re-partition drawn among r2-r4, then its test discharges resampled with replacement), in the weighted form of `stage2_b64.cell`, with 20,000 draws.
- Medians and median ratios: weighted quantiles under the same draws (`stage2_b64.medians`), with 20,000 draws (B.64 used 2,000).
- Seed 20261170, one stream per quantity (crc32 of its name, as `run_b54.rng_for`).
- MSE intervals are the squares of the RMSE bounds on the same draws.
- "MSE reduction in %" = 100 x skill.

**A5. Old values.**
- Two-stage rows and every pair: `docs/paper/b64_two_stage.json` (r2-r4, old subset).
- Network rows of the main table, as cited: `b39_primary.json` arm P and `b61_absolute_robust.json` (r2-r9). The r2-r4 values of B.64 are given beside them where they exist.
- B.59: `b59_inputs.json`. B.67 and B.68: `b67_gated.json` and `b68_auxtarget.json`.

**A6. Shared references (timing, step 1).**
- Timing: one network with its references on all 40,666 eligible samples of r2 f0 took 447 s on the CPU. The 15 folds hold 695,472 eligible samples, so the 135 runs take about 19 process-hours.
- Six processes at once ran at about 1.5 times the speed of one: each scorer holds about 1.8 GB, beside an 11.6 GB process of another session, and the machine was on battery. That puts the 135 runs at about 11 hours, over the 10-hour limit of the task.
- The rows, values and reference errors of a fold do not depend on the network: same split, harness statistics, seed, window and data treatment. Check (c) verifies this.
- `b70_d1/score_b70.py` therefore runs eval_seq's own code for the rows and references once per fold, and `eval_seq.build_pred_lookup` once per arm. It writes per arm the files that `eval_seq.compare()` writes. The loop and the post-loop block are copied from `compare()`; the functions are imported.
- Arm handling: B.59 arm L installs `fast_shift.install_eval(1)` around the lookup, as `eval_b59L.py` does. B.67 registers its variants, as `eval_b67.py` does; the gate readout is not written.
- It is used for the folds other than r2 f0 only if, on r2 f0, every key of every arm's npz and the metrics json are bit-identical to the output of that arm's own scoring script (`runs_b70.execute`, A1). Otherwise the batch uses the arms' own scripts.

## Addendum 2 (2026-11-19, phase 2, before any phase-2 readout)

Written after phase 1 was read and while the B.56 filters were being refitted. No phase-2 statement, table or readout had been computed. Facts and choices only.

**C1. How each readout is re-read.**
- Each batch's own readout code runs unchanged. Its module constants are pointed at D1 caches: the stored-run directory function `run_b39.pooled_run_dir` (to the D1 re-scorings, A1), cache directories, output paths, and every set of re-partitions.
- Every pooled set becomes r2-r4: B.39's r2-r9, the filters' confirmation set r3-r9, B.66's `KFX_REPEATS`. Where a script reports r2 alone (the filters' development set), r2 alone is still reported.
- Outputs go to `data/.b70p2/paper/` with the structure of the original `docs/paper/b*.json`. `docs/paper/b70_d1_phase2.{md,json}` sets them key by key beside the old values.
- Intervals, draws and seeds are each script's own; only the first level is drawn among r2-r4.
- Not re-read: parts on the four file-level splits, the time split or the raw-stream overlap of the four test sets (B.56 units s42/s1/s7/s123 and camp, B.61 part B overlap, B.54 check (e) and B.66 check (b) on split 42, which stay as checks only).
- The four-split statements of B.56 (KX1-KX6) are read in their pooled form, in which B.56 already reports them: kfx over kf0, kg over kf0, network over kfx, kf0 over the causal GP (90 %), median ratio kfx / hold, kfx over kf0 on rows not after an absent frame.

**C2. Checks that compared with recorded old values.**
- B.54 (a), B.55 (d), B.57 (ii), B.60's section-2 percentiles, B.60 addendum medians, B.61 pooled skills and percentiles, and B.56 (a) compared with recorded old numbers. They are pointed at D1 references:
  - P1 and P2 from `analyze_b39.arm_p`'s code on the D1 runs of r2-r4 (`data/.b70p2/paper/b39_primary.json`);
  - the D1 B.54 readout (for B.61);
  - percentiles and medians computed directly from the D1 npz files and the D1 filter caches.
- Checks that do not depend on the scored rows run unchanged against the old files: B.54 (e), B.55 (b) and (c), B.66 (a) and (b).

**C3. Row tables and propensity.**
- B.54 row tables: `run_b54.locate` on the D1 rows. Per-file features (classes, blip rows, median T_i; `data/.b54/files.npz`) and the B.50 propensity fits of each fold (`p_s*.npz`) do not depend on the scored rows. They are reused and joined to the D1 rows with `run_b54.join_p`.

**C4. Rows after an absent frame (D1 item 2.6).**
- Every stratum a script defines by its own rule uses the B.54 definition, `run_b54.blip_rows_13`:
  - B.55's `after_absent`, which counted rows 1-3 after a 15-40 ms step with no condition before. B.55 also reads rows 1, 2 and 3 separately: the row number k is the smallest k with which `blip_rows_13`'s walk marks the row;
  - the rows not after an absent frame of B.56 (`quiet_mask`, ten rows after a 15 ms - 0.5 s step) and of B.57 (`after_absent10`).
- The original ten-row variants of B.56 and B.57 are reported beside.
- B.45 classes (B.54 H, B.66) are classes, not this stratum, and are unchanged.

**C5. Filters (D1 item 2.9).**
- KF0, Kc, Kg and KFX at clip 0.2, and KFX at clips 0.1 and 0.4, are refitted with `run_b56.fit_unit`'s code: starts, Nelder-Mead options and order unchanged. Two things change:
  - fit set: the manifest's `train_files` of the fold, not every non-test discharge;
  - held repeats removed from the filter's T_i (Y and LY, all files) where `seq_data.load_grid_files` (cut, held removal) has no value. Filter row j is grid row j + 1, checked file by file. One value is removed.
- The filter's own `iloc[1:-1]` loader and raw-log BES drive stay.
- Rows: the D1 rows of the fold, located by `run_b56.map_rows_exact`.
- Parameters per fold go to `data/.b70/kf_params_r{r}_f{k}.json`, keys "KF0", "KFX" (theta in `kfx_model.run`'s order), "use_x".

**C6. B.55.**
- Parts R (the twelve raw discharges, row-independent raw caches of B.55 reused), G and D run on the D1 rows of the 15 r2-r4 networks.

**C7. B.65 on the cut population.**
- The 30 inclusive networks (`.b65_{v2,v2h}_incl_r{r}_f{k}`) are run by `eval_seq.build_pred_lookup` under `run_b39.pooled_env` of the same model and fold. That env is cut, with the cut harness statistics, CPU and A1's three changes. Inputs and outputs use each network's own (inclusive) training statistics.
- Their T_i errors on the D1 rows, with the D1 references, are read by `analyze_b65.cell` (20,000 draws, B.65's seed) as I1-I4.

**C8. Delay replay (`run_b42`'s rule and functions).**
- Delay rule: a T_i sample at grid row u is available at row t if u <= t - 1 - L. The six CES channels are rebuilt (`run_b42.ces_channels`, true staleness; the no-delay staleness reported).
- Networks: the B.39 network and twin of each r2-r4 fold, on its test discharges.
- References: hold = last available value; causal GP = `predict_gp_causal` on the 16 last available samples.
- L = 0, 1, 2 on one fixed row set: the D1 rows (grid rows by `compact_index`) whose block holds a T_i at u <= t - 3.
- Statements:
  - T1: the network over the causal GP at L = 1 and at L = 2;
  - T2: the network over the twin at L = 2.
  - Each is read as the pooled two-level interval (20,000 draws) above zero. The rule of three of four splits has no D1 form.
- Checks at L = 0: the rebuilt inputs equal `build_blocks` on every test file; network, hold and GP lie within 0.5 eV of the D1 npz on at least 99 % of the fixed rows.

**C9. 50 ms thinning replay (`run_b30.rows` at k = 5, copied).**
- In each block every fifth row is visible (phases o = 0..4). History and networks see visible values only.
- References: GP length scales x 5 (unscaled reported); PCHIP on up to four visible values on each side; hold = last visible value.
- Rows: the D1 rows. In each phase a row is scored when it has a visible earlier and a visible later T_i in its block and every estimator is finite (the B.30 population). A row counts once per phase.
- Statements H1 (network over the causal GP) and H2 (network over the twin) are read as pooled two-level intervals (20,000 draws) above zero.
- References and phases depend only on the discharge. They are computed once per discharge and paired with the network of the fold in which it is a test discharge, in each re-partition.

**C10. Old values.**
- Each original `docs/paper/b*.json` is the old value. B.48 is the old value for the replays: the B.39 four-split network; the per-split values and their mean are given.
