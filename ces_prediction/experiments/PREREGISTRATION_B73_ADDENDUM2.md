# B.73 — Addendum 2 (implementing agent): implementation choices

Written 2026-11-20 before any readout of B.73. No estimate has been scored against a target, apart from the registered reproduction checks (a) and (b) of section 10.

Sections 1-10 were first written before the first fit, by mistake over the session lead's committed Addendum 1 (commit 73b5e3b). The session lead restored Addendum 1 and moved that text here. Sections 1-10 are repeated here unchanged. Section 11 records an implementation change made after check (c). Section 12 holds the choices for the lead's items A1-A4, written after Addendum 1 was read and before any readout.

Facts and choices only. No registered definition is changed. Code: `ces_prediction/experiments/b73_targets/`; caches `data/.b73/`.

## 1. Rows and units

- **Grid.** `seq_data.load_grid_files(data, True, 3000)`: 641 files, 247,207 rows. Values above 3 keV and held repeats are missing. A block is cut at a time step of 0.5 s or more.
- **D1 rows.** The test rows of `data/.b70/stage2_r2_f{0..4}.npz`: 222,672 rows of 635 discharges. r3 and r4 contain the same rows (checked; values agree to 0.0005 eV). Each estimator and reference is read on this one row table.
- **Units of every design.** The 635 D1 discharges. The six other files (four discharges of the universe without a D1 row, two files outside it) enter no fit and no score, the filters included.
- **Shot clusters.** The shot numbers of the 641 delivered files are sorted, and a new cluster starts where a shot number exceeds the previous one by more than 3. That gives 89 clusters, 86 with D1 discharges, of 1 to 31 D1 discharges each (median 4; largest five 24, 26, 28, 29, 31).

## 2. Targets

- **Rows t+j.** Row t+j is the grid row of the same file and block whose time is 10 j ms after t, within 3 ms. Every grid time step lies within 3 ms of a multiple of 10 ms, and 98.9 % equal 10 ms. A position counts as not measured when there is no such row (a time step of 20-40 ms, or the end of the block), or when its value is missing in the cut, held-free grid.
- **Counts of D1 rows with the target defined.** F 222,672; N1 214,768; L3 204,733; L5 215,369; L10 215,667.
- **Fit and score rows.** An estimator fitted to target X is fitted on the fit rows where X is defined, and scored where X is defined. A comparison across targets (Q1, Q9) is read on the rows where both targets are defined, which are the L5 rows.

## 3. Inputs

- **Hold, causal GP G and age.** These come from `data/.b64/base.npz`. Recomputing hold and age from the grid agrees exactly.
- **Time step.** The time to the previous grid row of the file, in units of 10 ms.
- **The five history scalars.** Take the last ten measured values before t in the file's sequence of measured rows and keep those in the block of t. The scalars are their mean, their standard deviation (ddof 0), the last minus the mean, the last minus the second last, and the number kept. They are undefined when fewer than three are kept, and then the input set is incomplete.
- **M5.** The median of the last up to five measured values of the same block, among the last ten. At least one exists on every D1 row.
- **The eight fast scalars.** The nine BES and four ECEI channels are standardised with the mean and standard deviation (ddof 0) over all grid rows of the unit's fit discharges. This is run_b37's `fast_stats` with the fit discharges in place of "every file not in the test fold". The scalars are the array means and `run_b33.window_feats` of each. TXlag uses the same eight on grid row t-1 of the same block. TB uses the four BES scalars; TE the four ECEI scalars.
- **Complete input set.** The ten history scalars, the eight fast scalars of row t, the eight of row t-1 and G are all finite. One mask holds for every tree estimator. The share of D1 rows without it is reported. Such a row keeps G in every tree estimator and is not a fit row.
- **The boosted regression.** It is fitted to (target − G) with `HistGradientBoostingRegressor(random_state=20261116)`, otherwise default settings.

## 4. Designs

- **V1.** The fit discharges are the D1 discharges outside the test fold of the re-partition (its other four folds), for trees and filters alike. B.70 refitted the filters on the manifest training files only; check (e) reproduces that one fit for r2 f0.
- **V2.** The D1 clusters are put in a random order (`numpy.random.default_rng(seed).permutation`, seeds 1, 2, 3). Each cluster goes to the fold with the fewest D1 discharges so far, the lowest fold index on ties. Fold sizes: 126/129/130/124/126, 126/126/119/125/139, 129/132/125/124/125 discharges.
- **V3.** The D1 discharges are sorted by shot and cut into fifths of 127: 30801-31334, 31335-31589, 31590-31924, 31925-32185, 32188-32751. Fifth q (q = 2..5) is scored by a fit on fifths 1..q-1. The value is pooled over the four scored fifths, with each fifth also given.
- **V4a.** Fit on shots up to 31991 (412 D1 discharges); score from 32312 (95). The validation block 32002-32310 (128) is neither fitted nor scored.
- **V4b.** Fit on shots up to 32310 (540); score the same 95.
- **Gaps.** No D1 shot lies between 31991 and 32002 or equals 32311.
- **Statements on V4.** Q5, Q8, Q9 (V4) and Q10 are judged on V4a, the split the registration names first. V4b is reported beside it.

## 5. Filters

- **Code.** `kfx_model.run` is executed by `run_b56.ns()`. Held repeats are removed as in `p2_b56_fit.ns_held`, against the cached grid, with the same checks.
- **Fits.** K0 has three parameters, fitted first with the start and Nelder-Mead options of `p2_b56_fit`. KX keeps K0's three noise parameters at K0's fit for the same target and unit. Its two gains b_c and b_g are then fitted: B.56 / B.70 fit order, clip 0.2, same starts and options.
- **Loss for F.** The frozen loss: the mean squared eV error over the filter rows of the fit discharges that carry a measured value and an output.
- **Loss for L5.** The same rows, restricted to those with L5 defined, against L5.
- **Scoring.** On the D1 rows of the test discharges (filter row = grid row − 1).

## 6. Network-based estimates (V1 only)

- **Estimates.** c, t, c2 and t2 of the stage2 files.
- **Gain λ.** One per re-partition, fold, target and pair: the least-squares slope through the origin of (target − history-only estimate) on the correction, over the D1 rows of the other four folds of that re-partition with the target defined. The pairs are (t, c − t) and (t2, c2 − t2). The gain-fitted estimate is the history-only estimate + λ · correction.

## 7. Readouts

- **Skill.** Skill = 1 − SSE(A) / SSE(B) on identical rows; MSE reduction in % = 100 × skill. G in every readout is the per-row causal GP of base.npz. The GP of the comparison files enters only the reproduction check of B.70's values.
- **Points.** V1: the mean of the three re-partitions' values. V2: the mean of the three assignments. V3: pooled over fifths 2-5. V4a, V4b: the test block.
- **Intervals.** 20,000 draws, 2.5 and 97.5 % quantiles. In each draw the units are resampled with replacement in one level, and the per-unit sums of squared errors are re-added. Under V1 each draw first picks a re-partition (uniform over r2-r4) and resamples its units; under V2 the same with an assignment; under V3, V4a and V4b the units of the scored rows are resampled.
- **Units of the two interval forms.** Discharge form: the D1 discharges with at least one row in the cell. Cluster form: the shot clusters, each with all rows of its discharges in the cell.
- **Shared draws.** All comparisons of one cell (design × row set × form) use the same draws, from the stream `[20261173, crc32("design|row set|form")]`. Differences and ratios are therefore formed draw by draw.
- **Strata under V1.**
  - By chronological block: shots up to 31991, 32002-32310, from 32312.
  - By fifth of the BES level of row t: the raw mean of the nine BES channels at row t, with fifths cut at its quintiles over all D1 rows.
  - A stratum resamples the units that have rows in it.
- **Per-estimator values.** RMSE and median absolute error are points. Skills over G and over H get both interval forms.

## 8. Statements: how each is judged (cluster form)

| Statement | Quantity | Holds if |
|---|---|---|
| Q1 | ratio skill_L5(c2 over t2) / skill_F(c2 over t2) on the L5 rows, V1 as built | upper end below 0.5 |
| Q2-Q6, Q8 | as registered | lower end above zero |
| Q7 | skill of TX[L5] over the gain-fitted two-stage estimator against L5, V1 | upper end at least zero |
| Q9 | skill_F(TX[F] over T0[F]) − skill_L5(TX[L5] over T0[L5]) on the L5 rows, separately for V1 and V4a | lower end above zero |
| Q10 | AUC(history + fast) − AUC(history), V4a | lower end above zero |

For Q7, "not below zero" is read as "the interval does not lie entirely below zero". Whether its interval lies above zero is reported beside.

## 9. Usability classifier

- **Rows.** Grid rows of D1 discharges, other than a file's first or last row, with at least one measured value among the ten grid rows before (same block): 240,792 rows, 6.6 % positive.
- **Label.** The row has no value in the cut, held-free grid. The positives are counted by kind: missing in the file, above 3 keV, held repeat.
- **History inputs.** Time step; age of the last measured value of the block (ms); number of measured values among the ten rows before; whether row t-1 (same block) has one.
- **Fast inputs.** The eight fast scalars of row t, with the unit's standardisation.
- **Exclusions.** Rows without the fast scalars (fewer than ten earlier rows in the block) are left out of both classifiers. Their count is reported.
- **Model.** `HistGradientBoostingClassifier(random_state=20261116)`, otherwise default settings; the seed fixes the early-stopping split.
- **AUC.** Under V1, on the cross-fitted scores of all rows of each re-partition, then the mean over r2-r4. Under V4a and V4b, on the test rows.
- **Interval.** The same draws as above. The AUC of a draw is computed exactly from the unit-by-unit counts of concordant pairs (ties count one half).

## 10. Checks (section 10 of the registration)

| Check | What is compared | Pass rule |
|---|---|---|
| (a) | D1-1 to D1-4 of B.70 per re-partition, from the stored estimates and the comparison files' GP | within 0.001 |
| (b) | T0 and TX under V1, target F, against the reviewer's values (logs of `gbt_lean.py` r2 and r4, `gbt_variants2.py` r3): TX over T0, T0 over the stored-file GP, TX over the stored c2 and over c, T0 over t and over t2 | each within 0.005 per re-partition |
| (c) | leakage, on unit V1 r2 fold 0 (procedure below) | every estimate up to t* unchanged |
| (d) | V2 folds keep every cluster whole; the fits of V3 and V4 contain no discharge later than the first test discharge | both true |
| (e) | the filter code: the F fit on r2 f0's manifest training files against `data/.b70/kf_params_r2_f0.json` | reproduces it |
| (f) | the base GP against the comparison files' GP | share of rows apart by more than 0.5 eV, reported |

Procedure of (c):

1. One D1 row t* per test discharge, drawn from the stream `crc32("leakage")`.
2. Every T_i value of rows t* and later in that discharge is multiplied by a random factor in [0.5, 1.5]. Half of its missing values from t* on are given a value.
3. The fast channels of rows after t* are multiplied by random factors in [0.5, 1.5].
4. Recomputed: the history inputs, the fast scalars, the causal GP at t* (by `baselines_interpolation.predict_gp_causal`), every tree estimator (the unit's fits redone) and the two filters (with the unit's F parameters).

Checks (a) and (b) are run first. If either fails, the batch stops and is reported.

## 11. Implementation change after two runs of check (c), before any readout

The first two runs of check (c) failed on bit-identity of the fast scalars, not on causality. In both runs the history inputs, the causal GP and the filters were unchanged up to the cut.

| Run | Implementation | Why the scalars differed |
|---|---|---|
| 1 | `window_feats` once over all files laid end to end | the running sums crossed file boundaries |
| 2 | `window_feats` per file; channel standard deviations from sums of squares minus the squared mean | numpy reductions round differently depending on memory alignment, so the standard deviations differed in the last digits between two processes. The scalars of fit rows moved by up to 1.7e-7, and one tree estimate up to the cut moved by 16.7 eV where a value crossed a bin edge |

The implementation now:

- applies `window_feats` per file, as `run_b37.Ctx.feats` does;
- computes each channel's mean and standard deviation by two passes of exactly rounded sums (`math.fsum`);
- forms the array means as explicit sequential sums.

Every tree estimator and classifier was refitted after each change, and checks (b) and (c) were run again. The definitions are unchanged. Both earlier runs are in `docs/paper/b73_checks.md`.

## 12. The session lead's Addendum 1 (A1-A4, Q11-Q13): implementation

### Common

- **Rows.** Grid rows other than a file's first or last row.
- **BES array mean of a row.** The arithmetic mean of the nine raw BES channels; the ECEI array mean is that of the four raw ECEI channels.
- **Regular row.** Row t is regular when rows t-10 .. t are consecutive rows of the same block with every step within 3 ms of 10 ms. This is "ten regular 10 ms rows before, no absent frame in the last 100 ms".

### A1

- **r_B.** Defined on a regular row, as the BES array mean of row t over the median of the BES array mean of rows t-10 .. t-1, when that median exceeds 0.02. The 0.02 guard is the reviewer's (C18).
- **Classes.** Below 0.9; 0.9 to 0.97; 0.97 to 1.03; above 1.03, each lower edge included. The fifth class is "r_B undefined".
- **Increments per class.** Each estimator keeps its fit on all fit rows of its target and is only scored by class: TX over T0, KX over K0, c2 over t2 and c over t as built and with the gain fitted. Each is read against F (F rows of the class) and against L5 (L5 rows of the class), point with both intervals.
- **Share of the F-reduction.** Computed per pair: (SSE(B) − SSE(A)) on the class's F rows over the same on all F rows, with both intervals. The draws are those of the F cell of the design.
- **Share of rows.** The class's share of the D1 rows.
- **V4.** The same for TX over T0 under V4a and V4b.

### A2

- **Absent frame.** A step of more than 15 ms and less than 45 ms between consecutive grid rows of one block (B.54). It lies between those two rows.
- **Classes of a scored row.** Absent frames both earlier and later in its file; only later (the row is before the first); only earlier (the row is after the last); a file without any absent frame.
- **Readouts.** Those of A1, under V1.

### A3

- **Rows.** Regular rows with r_B defined.
- **T_i ratio.** The value of row t (cut, held-free grid) over the median of the measured values of rows t-10 .. t-1. Requires at least six of them (C18).
- **Fresh V_rot.** The delivered rotation value (`stage_vt.py`, no held removal) of a row that differs from the delivered value of the previous row and lies within ±1000 km/s (C18).
- **V_rot ratio.** The fresh value of row t over the median of the fresh values of rows t-10 .. t-1. Requires at least six and a median above 20 km/s (C18).
- **ECEI ratio.** The ECEI array mean of row t over the median of rows t-10 .. t-1. Requires a median above 0.02.
- **Table by r_B class.** For each of the nine r_B classes: the number of rows with a T_i ratio and its median; the number with a V_rot ratio and its median; the median ECEI ratio.
- **Slopes.** Least squares with intercept of the log ratio on log r_B, on the rows with both a T_i and a V_rot ratio, both positive. As in C18, r_B is restricted to 0.6 to 1.6 and the log ratios are clipped at ±1.5. The interval is 20,000 draws with discharges resampled, from the stream `crc32("A3|slopes")`. Q13 is judged on these. The slopes without the restriction and the clip, on all rows with both positive ratios, are reported beside.
- **Repeats.** The share of delivered rotation values that equal the previous row's is reported.

### A4

- **Rows exposed without beam emission.** A row whose ten preceding grid rows lie in its block, where every BES channel's median over those rows exceeds zero, and whose every BES channel is below 0.4 of that median.
- **Reported.** Their number, how many carry a T_i value in the cut, held-free grid, and how many are D1 rows.
- **Sensitivity refit.** Under V1, T0 and TX for F and L5 are refitted with these rows removed from the fit rows, and scored on the D1 rows without them. The history inputs are not rebuilt. Reported: TX over T0 against L5 (the quantity of Q2) and against F, beside the values on the same rows without the removal from the fit.

### Q11-Q13 (cluster form, except Q13)

| Statement | Holds if |
|---|---|
| Q11 | in class r_B above 1.03 under V1: the lower end of TX over T0 against F is above zero and the lower end against L5 is not above zero |
| Q12 | in class r_B below 0.9 under V1: the lower end against L5 is above zero |
| Q13 | the lower end of the T_i slope is above 0.5, and both ends of the V_rot slope lie within ±0.1 (discharge intervals) |
