# B.71, addendum 1 — implementation choices the registration leaves open

Written 2026-11-19, before any readout of B.71: no error against a measured T_i has been computed under any scenario, and no statement, bin value or interval exists. Written after the runner was implemented (`ces_prediction/experiments/b71_scenarios/`) and after checks 1–5 and a timing were run on fold r2 f0, restricted to its first five test files (`docs/paper/b71_checks.md`). Facts and choices only; the registration is unchanged.

## C1. Time comparisons

- Every comparison of a time with a cut uses integer units of 0.1 ms (the grid time, stored as float32, rounded to 1e-4 s).
- c_j = (time of the first row of the block) + j × 1 s, j = 1, 2, … while c_j is not after the last row of the block.
- A sample is "before c" when its rounded time is below c, and "at or after c" otherwise.

## C2. Cuts

- "At least three T_i samples available before c": measured T_i values of the D1 grid (above 3 keV and held repeats removed) in the same block with time below c.
- "The block continues at least 0.5 s after c": the last row of the block lies at c + 0.5 s or later.
- t0 is the first row of the block at or after c. Every enumerated cut is stored with the reason it was dropped.

## C3. Horizons

- F and R: the rows t0 … last row with time ≤ c + 3 s, within the block.
- G: the rows t0 … last row with time ≤ c + g + 1 s, within the block. Rows in [c, c + g) are replayed but not scored (s ≤ 0).
- Pairs with e above 3,000 ms fall outside every bin. They are counted and not read.

## C4. Onset of R and the V_rot channels

- At t0 the inputs of R are those of U and F: the sample of row t0 − 1 was measured before c and is available.
- From t0 + 1 on, the six CES channels hold the estimate of row t − 1, for both targets:
  - the value is the network's normalised output (its own target standardisation; the twin uses the twin's);
  - the staleness is log1p(time[t] − time[t − 1]);
  - the flag is 1.
- The V_rot channels are fed back the same way also when no V_rot sample was available before c (the registration sets flag 1 for both targets).

## C5. Replay from the U state

- F, G and R start from the LSTM state of the U pass at row t0 − 1. The U pass is the same block forward, split at the cut rows.
- Rows before t0 are U's outputs.
- F and G are one batched forward over the replay rows of every cut of a file (one per scenario and gap). R is stepped row by row, batched over the cuts of a file.
- U itself is one forward per block, as in `eval_seq.build_pred_lookup` (check 2).
- G is computed for the network only (the registration reads no other estimator under G).

## C6. Added checks

The run stops if one fails, as for the registered checks.

- **4b.** The batched F/G continuation from the U state, fed the U inputs, equals the batched U forward within 1e-4 eV.
- **5.** The causal GP under F is fitted once per cut on the 16 last available samples, in `predict_gp_causal`'s order. The hyperparameter choice does not depend on the target time. The posterior mean is then evaluated at every row of the horizon. It must equal `baselines_interpolation.predict_gp_causal`, called row by row, within 1e-6 eV.
- **D1 mapping.** Each D1 row of the B.70 re-scoring must map onto a grid row whose T_i equals the stored `CES_TI_y_true` within 0.01 eV.

## C7. D1 rows and check 2

- The scored rows of a fold are the T_i rows of B.70's re-scoring of the network, `data/.b70_rescore/b39p_v2_cut_r{r}_f{k}/comparison_errors_test.npz`.
- Its indices are mapped onto the grid:
  - `CES_TI_shot` follows the dataset's file order: sorted CSVs with at least one target after the D1 treatment;
  - `CES_TI_row` counts the rows with at least one target.
- Check 2 compares absolute errors, because B.70 stores squared errors, not predictions. The deviation is | |p_U − y| − sqrt(se) |, with y = `CES_TI_y_true`.

## C8. Scope of the checks in the full run

- Check 1: every test file of every fold, both networks.
- Check 2: every D1 T_i row of every fold.
- Checks 3, 4, 4b and 5: the first five test files (sorted names) of every fold.

## C9. Units, weights, points

- A discharge of a re-partition is a D1 discharge with at least one scored T_i row in its five folds.
- A unit (scored row, cut) has weight 1 within its discharge.
- Draws: `stage2_b64.weights` (the two-level draw of B.70 A4), 20,000 draws. Seed [20261171, crc32(name)], one stream per cell:
  - F cells are named "F|(lo, hi]";
  - G cells are named "G<g in ms>|(lo, hi]".
- Every estimator, pair and ratio of a cell uses the cell's draws.
- Point: the mean over r2–r4 of the per-re-partition value. MSE interval: the square of the RMSE bounds.

## C10. Medians, signed errors, shares

- Median absolute error: point only, the mean over r2–r4 of the per-re-partition median. No interval.
- Mean signed error and the share outside [0, 3,000] eV: sum-based, with intervals.
- A non-finite estimate counts as outside [0, 3,000] eV.

## C11. Divergence rule

- In a bin, an R estimator is divergent when either:
  - the lower bound of the interval of MSE(R) / MSE(hold F) exceeds 2; or
  - the point share of its estimates outside [0, 3,000] eV, over the bin's units, exceeds 0.05.
- A bin without units is not evaluable and does not count as divergent. The number of evaluable bins is reported with S3.

## C12. Return rule and S5

- "The network has returned by bin b" means the return rule holds in b and in every later bin up to (300, 1000] ms.
- The first bin in which the rule holds is reported beside it.
- S5 holds when, for each of g = 0.1, 0.3 and 1.0 s, the network has returned by (30, 100] ms or an earlier bin.

## C13. Training range of the staleness input

- For each fold, the T_i staleness (true staleness, ms) is taken at every grid row of the fold's training discharges (`train_files` of the manifest) whose T_i flag is 1. The 50th, 99th, 99.9th percentile and the maximum are reported.
- The same quantities on the rows with T_i measured are reported beside them.
- A bin of e is marked "outside the training range" when its lower edge exceeds the 99.9th percentile of any fold. The table gives, per bin, the number of the 15 folds whose 99.9th percentile the lower edge exceeds, and every fold's 99.9th percentile.
- Changed 2026-11-19 22:56, at the session lead's request and before any readout or row of the full run. The first version required all 15 folds; the rule now requires any fold, which is more cautious.

## C14. Filters

- The filter rows are computed by a separate step (`run_b71.py kf`). It reads the files that B.70 phase 2 writes (`b70_d1/p2_b56_fit.py`): `data/.b70/kf_params_r{r}_f{k}.json`, with keys `KF0`, `KFX`, `clip` and `use_x`.
  - When this was written, the files of r2 f0, f2 and f4 existed.
  - The step runs only when all 15 exist. Otherwise the readout states that the filter rows are left out.
- The step refuses a file whose `use_x`, `fold` or `held_repeats_removed` do not match phase 2's D1 filters.
- The recursion is `b56_kfx/kfx_model.run`, copied, because that module fits and scores at import. It is batched over the cuts of a block. A sample that is not available makes no update.
- Filter rows follow kfx_model's own loader, as phase 2 does:
  - the CSV without its first and last row, in float64; filter row j = grid row j + 1, checked on the time column;
  - time and T_i from the CSV in float64, with T_i missing wherever the D1 grid has none (above 3 keV, held repeats), which is phase 2's filter grid;
  - BES drive c = mean log of the BES channels, g = mean log(col 07) − mean log(col 05);
  - the state restarts at each grid block start.
- Filter check (added 2026-11-20, before the readout). At the first replay row t0 of every cut, F holds the same samples as U. So KF0 F and KFX F must equal phase 2's stored U-scenario estimates on that row (`data/.b70p2/b56/r{r}_f{k}.npz`) within 1e-6 eV. Also:
  - KF0 F must be constant after every cut;
  - every F unit must have a filter output.
- First run of `kf` (2026-11-19/20): time and T_i were taken from the float32 grid. The check failed by 4.4e-3 eV on 4,842 rows. With time and T_i from the CSV in float64, as written above, the deviation is 0.0 eV. The filter rows were recomputed before the readout ran (`kf_checks.json` 00:10, readout 00:20).

## C15. CPU

- CUDA is hidden. Two threads (`torch.set_num_threads(2)`, OMP/MKL/OpenBLAS 2), because the CPU is shared with B.70 phase 1.

## C16. Arithmetic precision of the replays

- Fact (r2 f0, first five test files, 1,084 replayed rows per network): in float32, the arithmetic the networks were trained and scored in, check 4 fails.
  - R stepped row by row with every sample available differs from the batched forward by up to 3.7e-4 eV for the network and 2.8e-4 eV for the twin.
  - The batched continuation from the U state (check 4b) differs by up to 2.8e-4 and 2.3e-4 eV.
  - In normalised units this is at most 9.5e-7, about eight float32 units in the last place, from the different order of the sums in the LSTM.
  - 124 (network) and 125 (twin) of 1,084 rows lie above 1e-4 eV.
- The same comparisons in float64 agree within 1e-12 eV (network 6.9e-13, twin 8.6e-13). The stepping and the continuation are therefore exact, and the float32 difference is rounding.
- Choice: every forward pass of B.71 (U, F, G, R, and the U pass that gives the states) runs in float64.
  - The stored float32 weights are cast to float64.
  - The inputs are build_blocks' float32 values, cast to float64.
  - The estimate fed back in R is the float64 output.
  - Physical units: the float32 target statistics, in float64 arithmetic.
- The registered tolerance of check 4 (1e-4 eV) is kept.
- Approved by the session lead on 2026-11-19, before the full run.
- Check 2 in float64: largest deviation from B.70's float32 re-scoring 3.2e-4 eV on 839 rows (0.5 eV allowed); in float32 it is 1.5e-4 eV.

## Facts from the checks on r2 f0 (first five test files), before any readout

- D1 mapping: 38,107 D1 T_i rows of r2 f0 in 126 discharges map onto the grid. The largest |grid T_i − `CES_TI_y_true`| is 2.4e-4 eV.
- Check 1: passes on the five files, both networks.
- Check 2 (float64): 839 of 839 rows within 0.5 eV.
- Check 3: 10 of 10 blocks equal.
- Checks 4 and 4b (float64): pass (above).
- Check 5: 811 rows, 6 cuts, largest deviation 0.0 eV.
- Timing (float64, two threads, CPU shared with B.70):
  - the five files take 2.8 s (6 kept cuts, 1,084 F rows); loading the grid takes 38 s;
  - the 15 folds hold 5,298 kept cuts and 1,182,369 F rows;
  - projection for the full batch: 0.7 h (scaled by cuts) to 0.9 h (scaled by F rows), below the 8-hour stop of the registration.
