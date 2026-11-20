# B.84 — Addendum 1 (implementing agent): implementation choices

Written 2026-11-21 before any run of this batch's code, before the checks and before any readout. No estimate, error, MSE, skill or interval of any estimator of this batch had been computed or looked at when this text was written. What had been looked at: the registration; B.75's registration, Addendum 1 and code; B.78's Addendum 1 and code; the code of B.81 and B.82 and the key layout of `docs/paper/b82_localize.json`; the row counts of `data/.b75/sets.json` and `data/.b78/sets.json`; the presence of the stored fit files of B.75 (T0, TX, T0a, TXa, target L5) and B.78 (T0, TX, K10, L5) for the ten units below.

Facts and choices only. No registered definition is changed. Code: `ces_prediction/experiments/b84_onefit/`; caches `data/.b84/`; outputs `docs/paper/b84_onefit.{md,json}`, `docs/paper/b84_checks.md`. A change made after this text was first written is added in a dated section at the end, with what had been seen at that time.

## 1. Target, units, rules, rows

- **Target.** L5 only (the registration's statements and B.75's references are in L5).
- **Units.** `V1_r2_k0` to `V1_r2_k4` (re-partition r2 of V1), `V3_q1` to `V3_q4` (fifths 2 to 5), `V4a` (fit to 31991, scored from 32312), as defined by `b73_targets/designs.units`.
- **Rule sets.** Fixed delays D10, D50, D100, D300, D650, D1000: B.75's common set `common_L5` (`data/.b75/sets.npz`). S: B.75's `S_L5`. K10: B.78's `common_L5` (`data/.b78/sets.npz`). The fit rows of a unit under a rule are the rows of its fit discharges in the rule's set, the test rows those of its test discharges; the row order is that of `data/.b73/rows.npz`.
- **Rules reported.** The registration names D10, D100, D650, D1000, S and K10. The tables also carry D50 and D300 (members of the mixture), without a statement.

## 2. Inputs

All inputs of a rule come from B.75's stored inputs `data/.b75/inputs/{rule}.npz` (K10: B.78's `data/.b78/inputs/K10.npz`) and from the unit's fast arrays, formed by the functions of B.75 (`common75.history_matrix`, `anchor_scalars`, `zmeans`, `run_mean10`, `stage_trees.Feats.window`), imported, not copied.

- **T0all, T0** inputs: the ten history scalars of B.75 under the rule (`history_matrix`).
- **TXall, TX** inputs: those, the eight fast scalars of row t, the six anchor scalars (B.75's TX columns, in that order; 24).
- **TXrel** (part C): TX's 24 columns (counted from 1) without column 11 (BES level x_t, first fast scalar), column 15 (ECEI level x_t, fifth fast scalar), column 19 (a1 of BES) and column 22 (a1 of ECEI); the remaining 20 in their order: the ten history scalars, the BES departure, sd, increment, the ECEI departure, sd, increment, a2 and a3 of BES, a2 and a3 of ECEI.
- **T0raw** (23 columns): the time step of row t (B.73's `tstep`), the age of the newest visible value in ms (B.75's `age`), G − H; then the ten differences v_k − H, k = 0 .. 9, newest first (k = 0 is the newest visible value, so its difference is zero; kept as registered); then the ten ages (s(t) − s(row of v_k)) × 1000 in ms, newest first. The visible values are the last up to ten measured values of the block of t that are visible under the rule (the same values from which B.75 forms its five history scalars: positions max(p − 10, b) .. p − 1 of B.75 Addendum 1 section 3). Where fewer than ten are visible, the missing k are NaN (the regression's native handling of missing values). Stamps are those of the grid (`data/.b73/grid.npz`), the age formed as B.75 forms its age, `(tm[t] − tm[j]) * 1000.0`.
- **Visible values of the raw history**, recomputed per file from the grid: fixed delays and S by `common75.limit_stamps` and the 3 ms tolerance exactly as `common75.file_inputs` (measured rows of the block whose stamp is at most the limit + 0.003 s); K10 by B.78's rule (measured rows of the block before t whose frame index of `data/.b78/frames.npz` is a multiple of 10). On every row of every set the newest of them must be B.75's (B.78's) stored anchor row and its value B.75's (B.78's) H, otherwise the stage stops.
- **TXraw** (45 columns): T0raw's 23; the unit's standardised BES array mean (`common75.zmeans` with the unit's fit discharges) at grid rows t, t − 1, ..., t − 9 (newest first, ten values); the same for ECEI (ten values); the ten-row running mean m_std (`run_mean10`) of BES and of ECEI at the anchor row (B.75's a1 of BES and of ECEI, two values). Every row of every set has rows t − 9 .. t in its block (B.75's completeness requires row t − 10 in the block), so no fast value is missing.
- **Standardisation.** With the statistics of the unit's fit discharges (`Feats.stats`), as in B.75; the same arrays under every rule.

## 3. Fits

- Regression: `HistGradientBoostingRegressor(random_state=20261116)`, default settings, fitted to L5 − G_r; the estimate under rule r is G_r plus the prediction from the inputs under r.
- **Mixture stack** of a unit (T0all, TXall, T0raw, TXraw): the blocks of the rules in the order D10, D50, D100, D300, D650, D1000, S, each block the unit's fit rows in the rule's set in row order, with the inputs and the target L5 − G of that rule. One fit per learner and unit (four per unit, forty in all). Under default settings the regression holds out a random tenth of the stack for early stopping (more than 10,000 rows); copies of one row under different rules can fall on both sides of that split. This is a consequence of the registered default settings and is not changed.
- **Predictions** of each mixture learner: the unit's test rows under each of the seven rules of the mixture and under K10, with the inputs of that rule.
- **TXrel**: per unit and rule (D650 on the common set, S on the S set), fitted on the unit's fit rows of the rule as TX in B.75. Twenty fits.
- **References, not refitted**: T0, TX, T0a, TXa of B.75 (`data/.b75/fits/{unit}__{L}__{rule}__L5.npz`; T0a and TXa under a fixed delay are the D10 fit applied, under S the fresh-history fit on the S set applied, as B.75 defined them; they do not exist under D10), T0 and TX of B.78 under K10 (`data/.b78/fits/{unit}__{L}__K10__L5.npz`).
- **Threads.** One fit process with OpenMP/MKL/OpenBLAS at two threads (set by `b73_targets/common.py`); no other process of this batch runs at the same time. Memory is limited on the shared machine; the stack is built once per learner and freed after the fit.
- Fit outputs: `data/.b84/fits/{unit}__{learner}__{rule}.npz` (idx = rows of `rows.npz`, est), plus `data/.b84/fits/{unit}__{learner}__fit.json` (stacked rows per rule, n_iter). The fit logs carry unit names, counts and times only.

## 4. Checks (registration section 6), run once each after the fits and before any readout

| Check | What is compared | Pass rule |
|---|---|---|
| reuse | T0 and TX of `V1_r2_k0`, D650, refitted with this batch's input and fit code (the D650 block of the mixture code, fitted alone), against B.75's stored estimates and test rows | test rows identical, largest absolute difference of the estimates 0.0 |
| mixture | for `V1_r2_k0`, learners T0all, TXall, T0raw, TXraw: the stack rebuilt; its D650 block against the per-rule D650 inputs and targets of B.75 (T0all, TXall: B.75's T0 and TX matrices and L5 − G on the fit rows; T0raw, TXraw: the same rows and targets, inputs recomputed by the raw-window code for D650 alone); test discharges in the stack; stacked rows per rule printed | bit for bit; zero test-discharge rows |
| raw windows | 200 test rows of `V1_r2_k0` in the common set under D650 (stream `[20261184, crc32("raw windows")]`). (a) The twenty fast values against the array means recomputed from the grid rows of the row's file with the unit's `Feats.stats`, one Python float sum over the channels in channel order per value; (b) the 23 history columns recomputed (raw windows by this batch's code, age and G by `common75.file_inputs`) from a copy of the file's T_i in which every value not visible under D650 (stamp above the row's D650 limit + 0.003 s, or in another block of the file) is raised by 1000 eV | (a) bit for bit; (b) bit for bit |
| rows | for every design, rule and estimator of a comparison: the test rows of the mixture learners and of TXrel against those of B.75's stored TX (B.78's under K10); per design the scored rows are the same for every estimator of a rule; counts per design and rule printed | identical |

The fits are run before the checks because the reuse and rows checks compare fit outputs; no fit output except the bit-for-bit comparisons of the checks is read before the checks pass. If a check fails the batch stops and the session lead is told.

## 5. Readout

- **Points.** Pooled over the units of the design: V1 = the five folds of r2 (one split), V3 = fifths 2 to 5, V4a = its test block. RMSE = sqrt(SSE / rows). MAE = mean absolute error. Share within 10 % = share of rows with |estimate − L5| / L5 ≤ 0.10 (the definition of B.82).
- **Skill** of A over R = 1 − SSE(A) / SSE(R) on identical rows.
- **Intervals.** 95 %, 20,000 draws, quantiles 2.5 and 97.5 %. Units (shot clusters for the cluster interval, discharges for the discharge interval) with at least one row in the cell are resampled with replacement as multinomial counts and the per-unit sums re-added (`b82_localize/boot82.draws`, which draws the cluster form first and then the discharge form from the same stream). One stream per quantity: `[20261184, crc32(name)]`, name = the statement id for a statement ("U1" ... "U11"), otherwise the cell name "design|rule|A over R" (skills) and "design|rule|robust" (MAE and share within 10 %, all estimators of the cell on the same draws).
- **Verdicts** on the cluster interval: "lies above zero" = lower end above zero; "not worse by more than 5 %" = lower end above −0.05.
- **Statements, quantities.** U1, U2, U3: TXall over TX (V1; D650, D10, S). U4: TXall over TXa (V1, D650). U5: TXall over T0all (V1, D650). U6: TXall over B.78's T0 of K10 (V1, K10 rows). U7: TXall over TX (V3, D650). U8: TXraw over TXall (V1, D650). U9: TXrel over TX (V3, D650). U10: TXrel over B.75's T0 (V4a, S). U11: TXrel over TX (V1, D650).
- **Reported tables** (no statement): per design and rule, RMSE of T0, TX, T0a, TXa (where they exist), T0all, TXall, T0raw, TXraw, TXrel (D650, S); skills with both intervals of T0all over T0, TXall over TX, T0raw over T0, TXraw over TX, TXall over T0all, T0raw over T0all, TXraw over T0all, TXraw over TXall, TX over T0, and where they exist TXall over TXa, T0all over T0a, TXrel over TX, TXrel over T0; MAE and share within 10 % with intervals of TX, TXall, TXraw under D10, D650, S (and of the others under the same cells, as they cost nothing); counts of rows, discharges and clusters per design and rule; stacked fit rows per rule and unit.
- The JSON follows the layout of `docs/paper/b82_localize.json`: `statements` keyed by id (text, comparison, location, judged, value with point, cluster and discharge interval, holds, expected), `expectations`, a block `cells` keyed design → rule → {counts, rmse_eV, estimators → {MAE, share within 10 %}, skills → "A over R" → value}, `fit_rows`, `checks`.

## 6. Notes after the fits and the checks, before any readout (2026-11-21)

- Two wordings of section 4 were made precise before any check had been run: the TXrel columns are counted from 1, and the raw-window check raises every value not visible under D650, values of other blocks of the file included (stricter than the first wording).
- The raw history windows (`raw84.py`) were computed for the eight rules; on every row of every rule's set the newest window value, its row and its age equal B.75's (B.78's) stored H, anchor row and age. The first run stopped at K10 on a file without a kept value (an indexing error in the code, no number read); after the fix it was run again.
- All 60 fits (40 mixture, 20 TXrel) were run once in one process at two threads (543 s); no fit was repeated or changed. Every mixture fit ran the full 100 iterations of the default settings (no early stop); the log and the fit files of the mixture carry counts, iterations and times only.
- The four checks were run once each and pass (`docs/paper/b84_checks.md`): reuse bit for bit (largest difference 0.0 eV for T0 and TX); the D650 block of each of the four stacks bit for bit, no test-discharge row in any stack; raw windows identical on 200 of 200 rows (fast and history); every estimator of every comparison on identical rows in all 24 cells. Nothing was changed because of them. No error, MSE, skill or interval of this batch had been computed when this note was written.

## 7. After the readout (2026-11-21): what happened, nothing changed

- `readout84.py` was run once end to end after the four checks had passed (218 s); it wrote `docs/paper/b84_onefit.json` and rendered `b84_onefit.md` from it. Its progress log prints row counts only. No definition, row, fit or statement rule was changed after any number had been read.
