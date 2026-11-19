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
