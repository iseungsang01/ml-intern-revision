# B.83 — Addendum 1 (implementing agent): implementation choices

Written 2026-11-21, after reading the registration and the code of B.70, B.73 and B.82, and before the references were computed, before any check and before any readout. No error, skill, MSE or interval of this batch had been computed. Facts and choices only; no registered definition is changed. Code: `ces_prediction/experiments/b83_interp/`; caches `data/.b83/`.

## 1. Where the references come from

- **What B.70 stored.** B.70 scored the references once per fold with `eval_seq`'s own harness (`b70_d1/score_b70.py`, `harness`). Its comparison files (`data/.b70_rescore/b39p_v2_cut_r{r}_f{k}/comparison_errors_test.npz`) keep the **squared errors** of PCHIP (`CES_TI_se_pchip`) and of the offline GP (`CES_TI_se_gp`) per scored row, beside `CES_TI_y_true`. The values themselves were not stored.
- **Coverage.** The D1 rows of B.73 (`data/.b73/rows.npz`, 222,672 rows of 635 discharges) are the test rows of the five folds of r2, and the same rows in r3 and r4 (B.73 Addendum 2, section 1). So B.70 has a squared error of both references on **every** D1 row, not only on part of them. Nothing is missing.
- **Why recompute.** The leave-out rule of section 2 of the registration and the robust measures need the values. They are recomputed per D1 row with B.70's code path: `baselines_interpolation.build_neighbor_set`, `predict_pchip` and `predict_gp`, called exactly as in `score_b70.harness` (`E.B.PREDICTORS[m](times, values, target_time)`).
- **The arrays they read.** B.70's harness read `KSTAR_CES_Dataset.file_arrays`. These are the file's rows with complete inputs, after the 3 keV cut and the removal of held repeats, **keeping only rows where CES_TI or CES_VT has a value** (`dataset.py:138`). Blocks are found on those rows (a step of 0.5 s or more). This batch rebuilds those arrays:
  - `seq_data.load_grid_files(data, True, 3000)` (the same column order, the same input dropna, cut and held rule as `dataset.py`);
  - then the rows with a value of CES_TI or CES_VT are kept, as float32, in the dataset's column order `time, CES_TI, CES_VT, ...`.
  - The row of a D1 row in that array is its position among the kept rows. It must equal the dataset row `row` of `data/.b70/stage2_r2_f{k}.npz` (which pairs `grow` with `row`); the stage stops if one differs.
- **The segment** of the registration is therefore B.70's block on the dataset array. Because rows without any CES value are not in that array, a stretch of 0.5 s or more without CES cuts a block that the grid of B.73 does not cut. Section 2 of the registration says "exactly as B.70 computed them", so B.70's block is used.
- **Bit-for-bit check.** The values are compared with B.70 through the stored squared errors: `(v - float64(CES_TI_y_true))**2` must equal `CES_TI_se_pchip` and `CES_TI_se_gp` exactly (`numpy.array_equal`), on every row of the 15 comparison files. A sign flip of the error (v' = 2y - v) would pass this test; it is not otherwise excluded.
- **The values of hold and the causal GP in the reference check** are B.70's own (`CES_TI_se_persistence`, `CES_TI_se_gp_causal` of the same files), because the check reproduces B.70's printed RMSE. Everywhere else hold H and the causal GP G are the stored estimates of B.73 (`rows.npz` `hold`, `gp`, from `data/.b64/base.npz`), as the registration says. B.73 check (f) found the two causal GPs apart by more than 0.5 eV on 0.04 % of the rows.

## 2. Rows left out for a missing reference

- **The rule.** `predict_pchip` and `predict_gp` fall back to hold when the block has no measured value after row t, and B.70 scored those fallback values.
  - Here, a row whose neighbour set (`build_neighbor_set`: B.70's block, row t excluded) has **no measured value after t, or none before t**, is a row where the reference "cannot be formed".
  - The condition is the same for PCHIP and for the offline GP, so both references are formed on the same rows.
- **What else is counted.**
  - Rows where the offline GP's fit fails on every grid point (`_gp_fit_predict` returns NaN) and `predict_gp` falls back to hold are counted and reported. They are not left out; the registration names no such case, and the value is B.70's.
  - The PCHIP rows with exactly two distinct neighbour times, where `predict_pchip` interpolates linearly, are counted and kept.
- **Reference rows.** For each design: the scored rows of the design (below) with both references formed.
- **Which rows each part uses.**
  - The table of the six estimators (H, PCHIP, G, offline GP, T0, TX) and every pair of section 4 use the reference rows. That includes T0 and TX over H and G, so every estimator of a table and of a statement is scored on the same rows (check "Rows").
  - Beside it, T0 and TX over H and over G are also given on all scored rows of the design (MSE skill only, both interval forms), since the leave-out rule names only comparisons with a reference.
  - The counts of rows left out are given per design.

## 3. Designs, units and estimates

- **Units.** `b73_targets/designs.units()`: V1 (r2-r4 x 5 folds), V2 (assignments 1-3 x 5 folds), V3 (fifths 2-5, scored by fits on the earlier fifths; units `V3_q1..q4`), V4a, V4b.
- **Scored rows of a design.** The D1 rows of the test discharges of its units, as B.73's `assemble_design`. Under V1 and V2, every D1 row is scored once in each re-partition or assignment.
- **T0 and TX.** `data/.b73/trees/{unit}__{T0,TX}__F.npz` (`idx`, `est`). Rows without a complete input set carry G there, as in B.73. Nothing is refitted.
- **Target.** `tg_F` of `rows.npz` (the value of row t as stored by B.73, which is B.70's r2 `y`).
- **Points.**
  - V1: the mean over the three re-partitions of the statistic computed in each.
  - V2: the same over the three assignments.
  - V3: pooled over fifths 2-5.
  - V4a, V4b: the test block.
  - The references, H and G are the same in every re-partition or assignment.

## 4. Measures (on the reference rows of a design)

- **RMSE** = sqrt(SSE / n).
- **MAE**, and the **median, 75th and 90th percentiles of |error|**.
- **Median relative error.** The median of |error| / value over the rows whose value exceeds 100 eV.
- **Shares within 10 % and 20 %.** The share of all reference rows with |error| / value at most 0.10 and at most 0.20. This is B.82's convention: the share uses every row, and only the median relative error is restricted to values above 100 eV.
- **Skill** of A over R = 1 - SSE(A) / SSE(R) on identical rows.
- **Ratios** of A over R: MAE, median, p75, p90 and median relative error, each A / R. Also the differences of the two shares, A minus R.

## 5. Intervals

- **Seed and draws.** 95 %, 20,000 draws. Seed stream `numpy.random.default_rng([20261183, crc32(name)])`.
  - The two forms draw from one stream in the order cluster, then discharge, with `b82_localize/boot82.draws`.
  - Under V1 (V2) a re-partition (an assignment) is drawn first for each draw. Then the shot clusters (or the discharges) with at least one row in the cell are resampled as multinomial counts.
  - Under V3, V4a and V4b one level only.
- **Sums.** Sums over rows are counts @ per-unit sums.
- **Quantiles.** Medians and percentiles use `boot82.WQ`: the exact weighted quantile, equal to the quantile of the resampled rows. B.82 verified this equality directly.
- **Clusters.** Those of B.73 (`rows.npz` `clus`).
- **Stream names.**
  - Statement Ik: `"Ik"`.
  - The table and pairs of design d: `"table|d"`.
  - The skills over H and G on all scored rows: `"allrows|d"`.
  - Within one stream, every measure and pair is formed on the same draws, so ratios and differences are draw by draw.
- **Points.** The statistic on the unresampled rows, as the mean over re-partitions or assignments (section 3).

## 6. Statements: how each is judged (cluster interval)

| id | quantity | holds if |
|---|---|---|
| I1-I8 | MSE skill of TX over PCHIP / the offline GP, in the design named | lower end > 0 |
| I9 | skill of T0 over PCHIP, V1 | lower end > 0 |
| I10 | skill of T0 over the offline GP, V1 | lower end > 0, or the interval contains zero; that is, upper end >= 0 |
| I11 | ratio of median absolute errors TX / offline GP, V1 | upper end < 1 |
| I12 | the same, TX / PCHIP, V1 | upper end < 1 |
| I13 | ratio of MAE, TX / offline GP, V1 | upper end < 1 |
| I14 | ratio of the 90th percentiles of |error|, TX / offline GP, V1 | upper end < 1 |

The discharge interval is reported beside each statement from the same stream.

## 7. Checks (section 5 of the registration), each run once, before the readout

| check | what is done | passes if |
|---|---|---|
| References | (i) The bit-for-bit comparison of section 1 on all rows of the 15 comparison files (r2-r4 x 5 folds). (ii) Per re-partition, the RMSE over all its rows of hold and causal GP (B.70's squared errors) and of PCHIP and the offline GP (this batch's values against B.70's `y_true`). The mean over r2-r4 is formatted to one decimal. | (i) every value equal; (ii) 232.6, 192.8, 206.0 and 180.8 eV, the values of `b70_d1.md` |
| No look at row t | 200 D1 rows drawn from the stream `crc32("no-look")`. For each, a copy of its dataset array is made in which the CES_TI value of row t is (a) set to missing (the row stays in the array, so the block is unchanged) and (b) raised by 1000 eV. PCHIP and the offline GP are recomputed from that copy. | both unchanged bit for bit in both cases |
| Rows | For every design and every sub (re-partition or assignment), the reference rows are formed, and each of the six estimators must have a finite value on every one of them. The masks must be the same in every sub. Counts printed per design: scored rows, reference rows, rows left out, discharges, clusters. | all finite; masks equal |
| Stored estimates | Under V1, on all D1 rows: the pooled RMSE of `T0[F]`, `TX[F]`, H and G (mean over r2-r4 of the per-re-partition RMSE), formatted to one decimal | 182.9, 174.8, 232.6 and 192.7 eV, the values of `b73_targets.md` |

- The results go to `data/.b83/checks/*.json` and to `docs/paper/b83_checks.md`.
- The readout refuses to run unless all four pass.
- The two RMSE checks compute an error on published rows in order to reproduce published values. The registration allows them; no other error of this batch is computed before the readout.

## 8. Compute

- CPU only, at most 4 threads in total.
- The references are computed by 4 worker processes with one BLAS thread each. The readout runs in one process with 2 BLAS threads (the setting of `b73_targets/common.py`).
- Output that is printed carries names, counts and check verdicts only, until the checks pass.
