# B.81 — Addendum 1 (implementing agent): implementation choices

Written 2026-11-20 before any run of this batch's code, before any check and before any readout. No fit of this batch had been run; no MSE, skill, interval or statement of any estimator of this batch had been computed when this text was written. What had been looked at: the registration; B.75's registration, Addendum 1, code and the layout of `docs/paper/b75_availability.json` (its numbers are known, as the registration says); B.78's Addendum 1 and code; `b75_availability/lead_checks/stuck_ecei.py`; the keys and shapes of `data/.b75/inputs/*.npz`, `data/.b75/sets.npz` and the names of the files in `data/.b75/fits/`.

Facts and choices only. No registered definition is changed. Code: `ces_prediction/experiments/b81_controls/`; caches `data/.b81/`; outputs `docs/paper/b81_controls.{md,json}`, `docs/paper/b81_checks.md`. A change made after this text was first written is added in a dated section at the end, with what had been seen at that time.

## 1. What is reused

- **Inputs, rows, units.** `common75.load_inputs` (`data/.b75/inputs/{D650,S}.npz`), `common75.history_matrix` (the ten history scalars), `common75.anchor_scalars`, `common75.zmeans`, `common75.raw_means`, `common75.run_mean10`, `designs.units` of B.73 (via `common75.DS`), `stage_trees.Feats` of B.73 (the eight fast scalars per unit). Row sets: `common_L5` of `data/.b75/sets.npz` under D650, `S_L5` under S.
- **Learner matrices.** `b78_sparse/fits78.inputs_of`, B.78's implementation of T0, TX, TXp (before the permutation), TB, TE and TXo, imported unchanged. Column order of TX: the ten history scalars, the eight fast scalars of row t (BES level, departure, sd, increment; then ECEI), the six anchor scalars (a1, a2, a3 of BES, then of ECEI). TXo: the ten history scalars, the eight fast scalars of the anchor row (`Feats.window` of the unit taken at the anchor's grid index), a1 of BES and a1 of ECEI.
- **Regression.** As `b75_availability/stage_fits.py` (`gbr`, `fit_store`): `HistGradientBoostingRegressor(random_state=20261116)`, default settings, fitted to (L5 − G under the rule) on the fit rows; the estimate of a test row is G under the rule plus the prediction. Fit rows = rows of the unit's fit discharges in the rule's set; test rows = rows of its test discharges in the set. Two OpenMP threads per process (as B.75, set by `b73_targets/common.py`), at most two processes at a time.

## 2. Fits of this batch (target L5)

| Learner | Units | Rules |
|---|---|---|
| TXp, TXo | V1 (15), V2 (15), V3 (4), V4a, V4b | D650, S |
| TE, TB | V2 (15), V3 (4) | D650, S |

220 fits, stored as `data/.b81/fits/{unit}__{learner}__{rule}__L5.npz` (idx = rows of `rows.npz`, est, fit_rows, n_iter; TXo also the number of fit and test rows whose anchor-row fast scalars are not all finite).

- **TXp.** One generator per unit and rule, `numpy.random.default_rng([20261181, crc32("placebo|<unit>|<rule>")])`, e.g. "placebo|V1_r2_k0|D650". The fit rows' permutation is drawn first (`permutation(number of fit rows)`), then the test rows' one, from the same generator; columns 10 to 23 (the fourteen fast and anchor columns) are reordered jointly by them, as in `fits78.run_unit`.
- **TXo.** Rows with a non-finite anchor-row fast scalar are kept and left to the regression's handling of missing values; their counts are reported.
- **Estimates read, not refitted.** T0 and TX for every unit and both rules, and TE and TB for V1, V4a, V4b, from `data/.b75/fits/`. TE and TB for V2 and V3 from `data/.b81/fits/`.

## 3. Checks (registration section 6), run once each after the fits and before any readout

| Check | What is compared | Pass rule |
|---|---|---|
| reuse | T0, TX, TE and TB of unit V1_r2_k0, rule D650, target L5 refitted with this batch's code (the matrices of `inputs_of`, the regression above) against `data/.b75/fits/V1_r2_k0__{T0,TX,TE,TB}__D650__L5.npz` | idx identical; est identical bit for bit (byte comparison of the float64 arrays) |
| permutation | for every unit of section 2 and both rules: the TXp matrices (fit and test) rebuilt by the fit code against the TX matrices of the same rows | columns 0-9 bit-identical; each of columns 10-23 has the same multiset of values (sorted byte comparison, NaN included) within the fit rows and within the test rows; in addition, the 14-column rows of TXp are a joint reordering of those of TX (the rows sorted lexicographically are identical) |
| anchor row | unit V1_r2_k0, 200 test rows under D650 and 200 under S (streams `[20261181, crc32("anchor|D650")]`, `crc32("anchor|S")`), drawn from the rule's test rows. Columns 10-17 of TXo against the eight fast scalars recomputed at the anchor's grid index from the anchor's file alone: the channels standardised with `Feats.stats` of the unit's fit discharges, the BES and ECEI array means by sequential sums, `common.window_feats` of B.73 on the file's rows | bit for bit for every row (NaN on both sides counts as equal) |
| rows | for every unit of section 2 and both rules: idx of T0, TX, TXp, TXo, TE, TB (stored files of B.75 or B.81 as in section 2) and the rows of the unit's test discharges in the rule's set | all identical; counts printed per unit and rule |

Only idx arrays, matrices and estimates are compared; no target, error or skill is formed in any check. No tolerance is changed after a run.

## 4. Readout

- **Skill** of A over R = 1 − SSE(A)/SSE(R) on identical rows.
- **Per design (V1, V2, V3, V4a, V4b) and rule (D650 on `common_L5`, S on `S_L5`), stratum "all":** TX/T0, TXp/T0, TXo/T0, TX/TXo, TE/T0, TB/T0, and the RMSE (eV) of T0, TX, TXp, TXo, TE, TB. TX/T0, TE/T0 and TB/T0 under V1 and V4 are recomputed from B.75's stored estimates with this batch's draws; their B.75 intervals stay in `b75_availability.json`.
- **Points.** V1: mean over the three re-partitions of the per-re-partition skill; V2: over the three assignments; V3: fifths 2-5 pooled; V4a, V4b: the test block. RMSE: mean over the re-partitions or assignments of the per-split RMSE.
- **Intervals.** 95 %, 20,000 draws, quantiles 2.5 and 97.5 %, with the mechanics of `b75_availability/readout.py` (`draws`): under V1 (V2) a re-partition (an assignment) is drawn for each draw first, then the units (discharges, or shot clusters, with at least one row in the cell) are resampled with replacement and the per-unit sums of squared errors re-added; under V3, V4a, V4b the units of the scored rows. Numerator and denominator of a skill are formed on the same draws.
- **Streams.** A quantity that is the subject of a statement uses `[20261181, crc32(<statement id>)]` (e.g. "C1"), as registered. Every other quantity uses `[20261181, crc32("<design>|<rule>|<stratum>|<comparison>")]`, e.g. "V2|S|all|TXo/T0". In each stream the cluster-form draws are taken first, then the discharge-form draws from the same generator. The main block of the JSON carries, for a quantity that is a statement, the statement's values (the same draws), so that one quantity has one interval.
- **ECEI-constant stratum (V1 only, described after the fact).** The definition of `lead_checks/stuck_ecei.py`, reimplemented without change of arithmetic: on the grid of `data/.b73/grid.npz`, a row is "ECEI constant" when it and each of the nine rows before it have all four raw ECEI channels equal to those of the previous row of the same file (run of equal steps of at least ten). A scored row is left out when its grid row (gi) or its anchor row under the rule (`anc_g` of the rule's inputs) is ECEI constant. On the remaining rows ("ECEI moving"): TXo/T0, TXp/T0, TX/TXo and TX/T0 beside them, rules D650 and S, with the rows left out counted.
- **Statements** (registration section 5), judged on the cluster interval: C1, C2 hold when the lower end is at most zero ("the interval contains zero or lies below it"); C3-C9 and E1-E8 hold when the lower end is above zero. C7 is V4a (fit to 31991); V4b is reported beside it in the note. V3 = fifths 2-5 pooled.
- **Expectation met.** "holds" expected: met when the verdict holds; "uncertain": not applicable.
- **Progress log.** Row counts and names only; no estimate or error is printed during the readout.

## 5. Output layout

`docs/paper/b81_controls.json`: `statements` (id -> text, value {point, discharge, cluster}, holds, expected, note), `expectations`, `main` (design -> rule -> {counts, rmse_eV, comparison -> {point, discharge, cluster}}), `ecei_moving` (rule -> same, V1), `counts` (rows, discharges, clusters per design and rule; fit rows per unit; TXo rows with non-finite anchor-row scalars), `checks`, run times. `docs/paper/b81_controls.md` rendered from it; `docs/paper/b81_checks.md` from the check files in `data/.b81/checks/`.

## 6. Notes after the fits and the checks, before any readout (2026-11-20)

- All 220 fits of section 2 were run once (two processes of two threads, about 3.6 min); no fit was repeated or changed. The fit logs carry unit names, fit counts and times only.
- The four checks were run once each and pass (`docs/paper/b81_checks.md`): reuse (T0, TX, TE, TB of V1_r2_k0, D650, L5 bit-identical to B.75, same rows, same fit rows and iterations); permutation (72 unit-rule cells, fit and test rows); anchor row (200 of 200 rows under D650 and under S); rows (72 cells, six learners each). Nothing was changed because of them.
- The anchor-row check also counts the drawn rows whose anchor is itself a D1 row and whose TXo fast scalars equal TX's fast scalars of that D1 row (199 under D650, 197 under S). Counted afterwards from the inputs alone: exactly 199 and 197 of the drawn anchors are D1 rows, so every such row agrees; the others are measured grid rows that are not D1 rows.
- What had been seen at that time besides the checks: row counts per unit and rule. No MSE, skill, interval or statement of this batch had been computed.

## 7. After the readout (2026-11-20): what happened, nothing changed

- `readout81.py` was run once end to end (44 s) and wrote `docs/paper/b81_controls.json` and `.md`; it refuses to run unless the four check files say "passes". No definition, row, fit, stream or statement rule was changed after any number had been read. Wall time of the batch from the first fit to the readout: about 8.5 min.
