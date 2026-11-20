# B.75 — Addendum 1 (implementing agent): implementation choices

Written 2026-11-20 before any run of this batch's code and before any readout. No statement, MSE, skill or other score under any availability rule had been computed when this text was written. What had been looked at: the registration, the B.73 code and caches, and these facts of the inputs (no target, no estimate): the grid time steps, the signs of the raw BES and ECEI array means, the row counts of `data/.b73/rows.npz`. They are given in section 1.

Facts and choices only. No registered definition is changed. Code: `ces_prediction/experiments/b75_availability/`; caches `data/.b75/`; outputs `docs/paper/b75_availability.{md,json}`, `docs/paper/b75_checks.md`. A change made after this text was first written is added in a dated section at the end, with what had been seen at that time.

## 1. Rows, grid and facts looked at before any run

- **D1 rows.** `data/.b73/rows.npz`: 222,672 rows of 635 discharges (the registration's section 2 says 639 discharges; the cached D1 rows, the same rows as B.73, have 635). Every row carries a T_i value in the cut, held-free grid.
- **Grid.** `data/.b73/grid.npz` (B.73's cut, held-free grid of `seq_data.load_grid_files`): 641 files, 247,207 rows. It contains each file's first row (t = 0) and last row (t = 30 s), which carry no T_i value. The step from the first row to the next is at least 0.402 s (in 8 files below 0.5 s, so that row joins the first block); the step to the last row is at least 11.6 s. Neither row is part of an absent step. Apart from these two rows per file the grid has the 245,925 rows of the reviewer's grid (C0).
- **Time steps inside a block.** The smallest is 9.998 ms; no step is below 7 ms. 1,414 steps lie between 15 and 45 ms; 88 lie between 45 ms and 0.5 s.
- **Raw array means.** The mean of the nine raw BES channels is at most zero on 640 grid rows (36 D1 rows); the mean of the four raw ECEI channels on 1,853 grid rows (1,284 D1 rows).

## 2. Availability rules

- **Visible values, fixed delay d.** The measured values (finite in the cut, held-free grid) of the rows of the block of t whose stamp is at most s(t) − d + 0.003 s. Because no step inside a block is below 7 ms, D10 makes every earlier row of the block visible, as the registration says.
- **Absent step.** A step between consecutive grid rows of the same block that is above 0.015 s and below 0.045 s. It lies between rows j and j + 1 and ends at row j + 1.
- **Rule S.** For row t, take the most recent absent step (j, j + 1) of the block of t with j + 1 ≤ t (row index). The limit is the stamp of row j. Visible: the measured values of the rows of the block whose stamp is at most that limit + 0.003 s, that is rows up to and including j. A row with no such step has no visible value under S and is not scored under S. Row j need not carry a value; the newest visible value is then older than the limit.
- **Two ages under S.** The age of the limit, s(t) − stamp of row j (the reviewer's quantity in C12), and the age of the newest visible value, s(t) − its stamp. Check C12 uses the first, the readout of section 7 the second (both are reported).

## 3. Estimator inputs under a rule

All are computed per file from the grid arrays of that file and the grid row of t. Nothing reads whether row t carries a value, so the same code runs on a copy of the file with values removed (visibility check).

- **Measured positions.** The rows of the file with a finite value, in time order. p = the number of them with stamp at most the visibility limit + 0.003 s. b = the position of the first of them in the block of t. The visible values of the block are positions b .. p − 1.
- **H** = the value at position p − 1 (when p > b). **Age** = (s(t) − its stamp) × 1000, in ms. The **anchor row** is that row.
- **The five history scalars**, as B.73 (`common.file_history`) with "before t" replaced by "visible": the values at positions max(p − 10, b) .. p − 1 (the last ten measured values of the file up to the limit, those of the block of t kept), in time order. Mean, standard deviation (ddof 0), last minus mean, last minus second last, number kept. Undefined when fewer than three are kept. **The last increment** is the difference of the last two kept values whatever the time between them (as B.73).
- **M5** = the median of the last up to five of those values.
- **G** = `baselines_interpolation.predict_gp_causal(stamps, values, s(t))` on positions max(p − 16, b) .. p − 1, in time order (as `stage2_b64.base_rows_of`). With one visible value G is that value; with none G is undefined.
- **Time step of row t.** B.73's `tstep` of `rows.npz` (time to the previous grid row of the file, in units of 10 ms); the same under every rule.
- **The ten history scalars.** In B.73's order: time step, age, H, G − H, |G − H|, mean, standard deviation, H − mean, last increment, number kept.
- **The eight fast scalars of row t.** `stage_trees.Feats(...).window(fit discharges)` of B.73, unchanged, per unit.
- **The anchor scalars.**
  - Array mean of a row: BES = the mean of the nine channels, ECEI = of the four channels, formed by explicit sequential sums over the channels. The standardised array mean uses the channels standardised exactly as in `stage_trees.Feats.window` (`Feats.stats` of the unit's fit discharges, then sequential sums); the raw array mean uses the float64 values of the stored float32 channels.
  - m(i) = the mean of the array means of the grid rows i − k, k = 0 .. 9, that lie in the block of i (block numbers are unique across files, so the window never crosses a file), added in the order k = 0, 1, ..., 9 and divided by their number. Defined when at least five rows enter, otherwise undefined.
  - a1 = m_std(anchor); a2 = m_std(t) − m_std(anchor); a3 = ln(m_raw(t) / m_raw(anchor)) when both raw means are above zero, otherwise undefined. Order of the six: a1, a2, a3 of BES, then of ECEI.
- **Inputs of the learners.** T0: the ten history scalars. TX8: those and the eight fast scalars (BES level, departure, sd, increment; then ECEI). TX: TX8's and the six anchor scalars. TB: the ten, the four BES fast scalars, a1-a3 of BES. TE: the ten, the four ECEI fast scalars, a1-a3 of ECEI. X0: the time step and the eight fast scalars.

## 4. Rows of each fit and score

- **Complete under a rule.** The ten history scalars, G, the eight fast scalars of row t and the six anchor scalars are finite. Finiteness does not depend on the unit (the standardisation is affine with positive scales; a3 uses raw means), so every set below is the same for every unit and design.
- **Common set (per target X in F, L5).** D1 rows complete under each of D10, D50, D100, D300, D650, D1000, with X defined.
- **S set (per target).** D1 rows complete under S and complete under D10, with X defined. The D10 condition makes "fresh history, same rows" possible; the number of rows complete under S but not under D10 is reported.
- **Fits.** The fit rows of a unit are the rows of its fit discharges in the set, the test rows those of its test discharges in the set. The boosted regression (`HistGradientBoostingRegressor(random_state=20261116)`, default settings) is fitted to (X − G under the rule) and the estimate is G under the rule plus its prediction. Every test row has an estimate; there is no fallback.
- **X0.** On the common set only (registration section 6): the median of X over the fit rows plus a regression of (X − that median) on its inputs. Under S it is not fitted; the S rows of the tables show "-" for X0.
- **Applied without refit.** Under a fixed-delay rule r: the D10 fit on the common set (T0, TX), evaluated on the test rows of the common set with the inputs of rule r: G under r plus the D10 model's prediction from the rule-r inputs (true age, rule-r anchor). Under S: the fresh-history fit on the S set, evaluated with the S inputs of the S set's test rows.
- **Reproduction check (section 9).** T0 and TX8 under D10 with the fit rows of B.73: its complete set (ten history scalars, G, eight fast scalars of row t and of row t − 1), not B.75's. Test rows without that set keep G, as in B.73, so that every stored B.73 estimate of the unit is compared.

## 5. Designs

- **Units.** `b73_targets/designs.units`, unchanged (V1 r2-r4 x 5 folds, V2 seeds 1-3 x 5 folds, V3 fifths 2-5, V4a, V4b).
- **V3c.** For the rolling-origin units of fifths 2, 3 and 4 (V3_q1, V3_q2, V3_q3) and seeds 1, 2, 3: `numpy.random.default_rng(seed).choice(D, n, replace=False)`, sorted, where D is the D1 discharges outside the test fifth in increasing file index and n the number of fit discharges of the unit. The standardisation of the fast channels uses those discharges. T0 and TX, rules D10 and D650 (common set), target L5. Fifth 5 is left out (registration). R13 uses fifths 2 and 3.
- **"V4" in a statement without a qualifier** (R10, R11) is V4a (fit to 31991), as R6 and R8 name it and as B.73 judged; V4b is given in the note.

## 6. Readouts and intervals

- **Skill** of A over B = 1 − SSE(A) / SSE(B) on identical rows; MSE reduction in % = 100 × skill. Increment = TX over T0 (and TX8, TB, TE over T0) under the same rule on the same rows.
- **Points.** V1: the mean of the three re-partitions' values; V2: of the three assignments; V3: pooled over the scored fifths; V4a, V4b: the test block. RMSE points are the mean over re-partitions or assignments of the per-split RMSE. Ratios and shares: the mean over splits of the per-split value.
- **Cost of staleness.** SSE under the rule / SSE under D10 on the same rows (fixed-delay rules: common set; S: S fit over the fresh-history fit, S set). **Recovered share.** (SSE(T0) − SSE(TX)) / (SSE(T0) − SSE(T0 under D10)) under the same rule and rows (S: the fresh-history T0); not formed under D10.
- **O** = the reported value of row t, scored against L5 only.
- **Intervals.** As B.73 (Addendum 2 section 7) with seed 20261175: 20,000 draws, 2.5 and 97.5 % quantiles; units resampled with replacement in one level and the per-unit sums of squared errors re-added. One stream per cell, `[20261175, crc32("design|row set|form")]`; every quantity of a cell is formed draw by draw on the same draws. V1: a re-partition drawn first; V2: an assignment drawn first; V3: the units of the scored rows (fifths 2-5 pooled, or one fifth); V4a, V4b: the units of the test block. V3c cells: a seed drawn first, then the units of the scored fifth(s); the V3 estimates are the same under each seed. Units: discharges (discharge form) or shot clusters (cluster form) with at least one row in the cell.
- **Row sets of the cells.** "common|X" (every fixed-delay rule and X0 on it), "S|X" (S and the fresh-history fits), the V1 strata "common|X|block b" and "S|X|block b" (block of the test discharge: up to 31991, 32002-32310, from 32312), "V3|q|X" per fifth and "V3|q1+q2|X" for R13.
- **Age under S.** On the S set: quantiles and the shares of the bands [0, 0.1), [0.1, 0.3), [0.3, 1), [1, ∞) s of the age of the newest visible value, and the same for the age of the limit.
- **Counts.** Rows of the common set and of the S set per target, in total and scored per design; rows not scored under S because no absent step precedes them in the block (share of D1 rows); rows complete under each rule; rows dropped by a3 alone.
- **Computation time.** In the readout process, with `threadpoolctl.threadpool_limits(1)`: 2,000 rows drawn from the common set (stream `crc32("timing")`), rule D650. Per row, `time.perf_counter` around `predict_gp_causal` on the row's visible values plus one `predict` call of the TX model of unit V1_r2_k0, D650, L5 (refitted in that process, identical to the cached fit) on that single row; 20 warm-up rows are not counted. Median and 99th percentile in ms, with the CPU and the fact that the machine was shared.

## 7. Statements: how each is judged (cluster form)

| Statement | Quantity | Holds if |
|---|---|---|
| R1 | SSE(T0 under D650) / SSE(T0 under D10), V1, L5, common set | lower end above 1.5 |
| R2, R4, R5, R6, R7, R8 | TX over T0 as registered (V3 pooled over fifths 2-5) | lower end above zero |
| R3 | (TX over T0 under D650) − (TX over T0 under D10), V1, L5, common set | lower end above zero |
| R9 | TE over T0, V1, L5, D650 | lower end above zero |
| R10 | T0 over G, L5, D650, under V1 and under V4a | both lower ends above zero |
| R11 | 1 − SSE(TX) / SSE(O), L5, D10, common set, under V1 and under V4a | both lower ends above zero |
| R12 | 1 − SSE(TX) / SSE(O), V1, L5, rule S, S set | lower end above zero |
| R13 | (TX over T0 of V3) − (TX over T0 of V3c), L5, D10, common set, fifths 2 and 3 pooled | upper end below zero |

## 8. Checks (registration section 9)

| Check | What is compared | Pass rule |
|---|---|---|
| 1 | G under D10 on every D1 row against `gp` of `data/.b64/base.npz` | largest deviation at most 1e-6 eV |
| 2 | T0 and TX8 under D10 (section 4, last item) against `data/.b73/trees/{unit}__{T0,TX}__{F,L5}.npz`, units V1_r2_k0 and V4a | largest deviation at most 1e-6 eV over all test rows |
| 3 | 200 test rows of unit V1_r2_k0 in the common set under D650 and 200 in the S set under S (streams `crc32("visibility|D650")`, `crc32("visibility|S")`). For each row a copy of its file in which every T_i value with stamp above the row's limit + 0.003 s is (a) removed, (b) raised by 1000 eV; the ten history scalars, G, the eight fast scalars, the six anchor scalars and the anchor row recomputed | every input identical bit for bit |
| 4 | Ages of the limit under S on the reviewer's population of C12: grid rows other than a file's first and last, with a finite T_i in the files as delivered (no cut, no held removal; read from the CSVs), with an absent step earlier in the block | median within 0.02 s of 0.65 s; band shares within 0.01 of 0.071, 0.166, 0.498, 0.266 |
| 5 | The test discharges of every B.73 unit against the test rows stored in `data/.b73/trees/{unit}__T0__F.npz` | identical |

Besides check 4, the same ages on the D1 rows (cut, held-free) and the share of valued rows without an absent step earlier in the block are reported. Beside check 1, H, age and the five history scalars under D10 are compared with `rows.npz` (largest deviations reported). No check's tolerance is changed after a run.

## 9. Change before any run of the checks (2026-11-20)

Written while the inputs of the rules were being computed, before any check had been run and before any fit. Check 3 does not recompute or compare the eight fast scalars and the running means m(i): they are functions of the fast channels and of the fit discharges only and take no T_i argument, so the copy of the file cannot change them. The anchor scalars are recomputed from the recomputed anchor row (they enter through the anchor row), and M5 is compared in addition to the inputs named in section 8.

## 10. Notes after checks 1, 2, 4 and 5, before any fit of the batch and before any readout (2026-11-20)

- Checks 1, 2, 4 and 5 were run once each and pass (G, H, age, M5 and the five history scalars under D10 bit-identical to the caches; T0 and TX8 reproduce B.73 with deviation 0.0 eV). Nothing was changed because of them.
- Check 4: the band shares differ from C12 by up to 0.0023 (within the tolerance). The cause was located after the run and is reported in `b75_checks.md`: the grid stamps are float32, and about 1,000 to 1,300 ages each lie on the edges 0.1, 0.3 and 1.0 s; with the float64 stamps of the CSV text C12's six printed bins are reproduced exactly. The check computation itself was not changed; the float64 comparison was added beside it as a diagnostic.
- Stream names of the intervals as implemented: `[20261175, crc32("design|target|set|stratum|form")]`, e.g. "V1|L5|common|all|cluster", "V1|L5|S|block up to 31991|discharge"; V3c cells "V3c|fifth 2|L5|all|form" and "V3c|fifths 2+3|L5|all|form". This is the one-stream-per-cell rule of section 6 with the cell's full name.
- Readouts added beside the registered ones (no statement uses them): skill of T0 and TX over X0 on the common set; skill of H, G, T0 over O beside TX over O; the cost of staleness of the estimators applied without refit.

## 11. After the readout (2026-11-20): what happened, nothing changed

- Check 3 was run after the row sets were built and passes (400 rows, each identical bit for bit under both modifications). All fits (1,566, plus the 8 of check 2) were then run once; no fit was repeated or changed.
- Before its first run, the readout's progress log was changed to print only row counts, so that no estimate was seen during the run.
- The first run of `readout.py` computed every readout and wrote `b75_availability.json`, then failed when rendering the markdown: `import render` resolved to `b73_targets/render.py`, which comes first on `sys.path` (same module name). The markdown was rendered from that JSON with this directory's `render.py`, and `readout.py` was changed to load its own `render.py` by path. The readout was then run once more end to end; its JSON is identical to the first apart from the computation-time readout and the run time (median 1.92 ms / 99th percentile 4.68 ms in the first run, 1.95 / 4.59 ms in the second; the files carry the second). No definition, row, fit or statement rule was changed after any number had been read.
