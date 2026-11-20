# B.77 — Addendum 1 (implementing agent): implementation choices

Written 2026-11-20 before any fit of this batch and before any readout. When this text was written, no estimate, classifier probability, departure label, MSE, skill, AUC or screening value of this batch had been computed or seen. What had been looked at: the registration (`PREREGISTRATION_B77.md`); the registrations, addenda and code of B.73 and B.75 (`b73_targets/`, `b75_availability/`) and their caches; B.73's `docs/paper/b73_checks.md` (layout only); in `docs/paper/b74_exposure.md` only the lines that contain the two thresholds 1.35 and 0.74 (they show B.74's shares of rows beyond those thresholds by stratum). `docs/paper/b75_availability.md` was not opened. The facts of the inputs in section 1 were computed for this text; they are row counts and properties of the inputs and of the target, with no estimate and no label.

Facts and choices only. No registered definition is changed. Code: `ces_prediction/experiments/b77_correct/`; caches `data/.b77/`; outputs `docs/paper/b77_correct.{md,json}`, `docs/paper/b77_checks.md`. A change made after this text was first written is added in a dated section at the end, with what had been seen at that time.

## 1. Facts looked at before any fit

- **D1 rows.** `data/.b73/rows.npz`: 222,672 rows of 635 discharges, each with a T_i value `y`.
- **Row set.** `complete_D10` of `data/.b75/sets.npz` (216,337 rows) and L5 defined (215,369 rows): **209,432 rows of 606 discharges in 85 shot clusters.** 29 D1 discharges have no row in it (each has at most 40 D1 rows). B.75's common set for L5 (160,343 rows, on which B.75's D10 fits were made) is contained in it.
- **The reported value y.** `y` of `rows.npz` (B.73's target F, B.75's O). It differs from the value of the same row in the cut grid (`grid.npz`, `ti`) by at most 2.4e-4 eV, on 30,329 rows (rounding of the stored files).
- **L5.** `tg_L5` of `rows.npz` equals a recomputation from the grid with `common.later_values` and `common.targets_from_later` bit for bit, undefined rows included. It is built from rows t+1 to t+5 only.
- **History under D10.** Every row of the set has at least three earlier measured values in its block (the five history scalars exist only then). The anchor row (the newest earlier measured row of the block) is row t − 1 on 98.7 % of the rows; H equals the grid value of the anchor row bit for bit.
- **Row t − 1.** 565 rows of the set have no row t − 11 in their block; for them the eight fast scalars of row t − 1 do not exist.
- **Strata on inputs (counts of rows of the set).** r_B classes (`data/.b73/strata_inputs.npz`, `rb_class`): below 0.9 9,387; 0.9–0.97 37,890; 0.97–1.03 94,373; above 1.03 54,319; r_B undefined 13,463. Chronological block of the discharge: up to 31991 160,401; 32002–32310 30,754; from 32312 18,277.

## 2. Rows and units

- **Row set.** The 209,432 rows of section 1, the same for every estimator, classifier, design and unit. Fit rows of a unit: the rows of the set in its fit discharges. Test rows: the rows of the set in its test discharges. Every test row has an estimate of every estimator; there is no fallback.
- **Units.** `b73_targets/designs.units`, unchanged: V1 r2–r4 × five folds, V2 seeds 1–3 × five folds, V3 fifths 2–5, V4a (fit to 31991), V4b (fit to 32310), 36 units. No V3c.
- **"V4" in a statement without a qualifier** (A8) is V4a, as A6 names it and as B.73 and B.75 judged; V4b is given in the note.

## 3. Inputs (every one taken from B.75's code, nothing redefined)

- **The ten history scalars** under D10: `common75.history_matrix(rows.tstep, D10 inputs)`, B.73's order: time step, age, H, G − H, |G − H|, mean, standard deviation, H − mean, last increment, number kept. G, H and the five scalars from `data/.b75/inputs/D10.npz`.
- **The four scalars of the frame's value**, in the registration's order: y − G, y − H, y − m10, |y − G|, with y the reported value of row t (section 1), G and H under D10, and m10 = the mean of the last up to ten earlier measured values of the block, i.e. the first of the five history scalars (`th[:, 0]` of the D10 inputs; at least three values).
- **The eight fast scalars of row t and the six anchor scalars** per unit: `stage_fits.Ctx.unit_arrays(unit)` and `Ctx.matrices("D10", ...)` of B.75, unchanged (`Lazy.H10`, `Lazy.f8`, `Lazy.A`). The fourteen fast scalars in this order: BES level, departure, sd, increment; ECEI the same; a1, a2, a3 of BES; a1, a2, a3 of ECEI.
- **BES half (seven scalars):** the four BES fast scalars and a1–a3 of BES. **ECEI half:** the four ECEI fast scalars and a1–a3 of ECEI (B.75's TB and TE halves).
- **Inputs of the learners** (columns in this order):

| Learner | Columns |
|---|---|
| T0, PH | ten history scalars (B.75's `Lazy["T0"]`) |
| TX, PHX | ten history scalars, fourteen fast scalars (B.75's `Lazy["TX"]`) |
| C0, FH | ten history scalars, four frame scalars |
| CX, FHX | ten history scalars, four frame scalars, fourteen fast scalars |
| CB | ten history scalars, four frame scalars, BES half |
| CE | ten history scalars, four frame scalars, ECEI half |

## 4. Estimators

- **O** = y. **MED3** = the median of y, the value of the anchor row (= H) and the value of the measured row before the anchor row in the same block. On the row set every row has at least three earlier measured values in its block (section 1), so MED3 always takes three values; no other rule is needed. A row of the set where this is not so would stop the run.
- **T0, TX, C0, CX, CB, CE.** `HistGradientBoostingRegressor(random_state=20261116)` (B.75's `stage_fits.gbr`), default settings, fitted on the unit's fit rows to (L5 − G); the estimate is G plus its prediction on the test rows.
- **Classifiers PH, PHX, FH, FHX.** `HistGradientBoostingClassifier(random_state=20261116)` (B.73's `stage_trees.gbc`), default settings, fitted on the unit's fit rows to the label; the score of a test row is the predicted probability of the label.
- **Label.** 1 when y / L5 > 1.35 or y / L5 < 0.74, else 0 (the run checks that L5 is above zero on every row of the set and stops otherwise).

## 5. Areas under the ROC curve

- **V1 and V2.** Per re-partition (V1) or assignment (V2): the AUC over all rows of the set, each scored by the fold in which it is a test row; the point is the mean over the three. **V3:** the pooled scores of fifths 2–5. **V4a, V4b:** the test rows.
- Ties count one half. Differences reported: PHX − PH, FHX − FH, FH − PH, FHX − PHX.
- **Interval.** 20,000 draws for every AUC cell. A draw picks a re-partition (V1) or assignment (V2) first, then resamples the units (discharges or shot clusters) with replacement. The AUC of a draw is computed exactly from the per-unit counts: A[a, b] = the concordant pairs (positive of unit a, negative of unit b, ties one half), P and N the positives and negatives per unit; AUC = Σ w_a w_b A[a, b] / ((Σ w_a P_a)(Σ w_b N_b)), with w the multiplicities of the draw (`b73_targets/readout.auc_matrix`, imported). This is the AUC of the resampled rows, with the scores as fitted.

## 6. Screening

- **Test unit.** The test rows of one unit of the design (one fold of a re-partition or assignment, one fifth under V3, the test block under V4a and V4b).
- **Dropped rows.** In each test unit with n rows, the k = floor(q n + 0.5) rows with the highest score, q = 0.05, 0.10, 0.20. Rank: decreasing score, ties in the order of the rows of `rows.npz` (`numpy.argsort(-p, kind="stable")`).
- **MSE.** Of O against L5 over the kept rows of every test unit of the sub (sum of squared errors over the kept rows divided by their number); the point under V1 and V2 is the mean over the three subs. "No screening" = the MSE of O over all rows.
- **Ratios.** The MSE on the rows kept by one classifier over the MSE on the rows kept by another (A9: FHX over FH) and over no screening, formed draw by draw.
- **Interval.** The kept flag of every row is fixed by the ranking within its test unit, made once as the registration defines the screening; the threshold is not re-derived inside a draw. A draw re-adds the per-unit sums of the squared errors over kept rows and the per-unit numbers of kept rows (an exact decomposition of the MSE given the flags); it uses the draws of the design's main cell (section 9).

## 7. Controls (V1 units only)

- **Placebo (CXp).** In each V1 unit, one permutation of the unit's fit rows and one of its test rows (stream `[20261177, crc32("placebo|<unit>")]`, fit rows first); each moves the fourteen fast scalars of a row together, as one vector, to another row; the ten history scalars and the four frame scalars stay. CX refitted on these inputs; CXp over C0 on the same rows.
- **Row t − 1 (CXlag).** The fourteen fast scalars with the fast samples of row t − 1 in place of those of row t: the eight fast scalars of row t − 1 of the same block (`stage_trees.Feats.at`, as B.73's TXlag) and the anchor scalars `common75.anchor_scalars(gi − 1, anchor of row t, ...)`, i.e. a1 = m_std(anchor) unchanged, a2 = m_std(t − 1) − m_std(anchor), a3 = ln(m_raw(t − 1) / m_raw(anchor)); the anchor row is that of row t under D10. Rows: the rows of the set where all fourteen are finite (about 565 rows fewer, section 1; the count is reported). CXlag is fitted on the fit rows among them and scored on the test rows among them. Reported on those rows: CXlag over C0, CX over C0 (C0 and CX from the main fits), CXlag over CX.

## 8. Strata under V1 (each estimator keeps its fit; scored by stratum)

- **Chronological block** of the test discharge (`blk` of `rows.npz`): up to 31991; 32002–32310; from 32312.
- **r_B class** (B.73 Addendum 1, `rb_class` of `data/.b73/strata_inputs.npz`): below 0.9; 0.9–0.97; 0.97–1.03; above 1.03; the undefined class reported beside.
- **Per stratum:** row counts, RMSE of every estimator, C0 over O, C0 over T0, TX over T0, CX, CB, CE over C0, and (CX over C0) − (TX over T0).

## 9. Readouts and intervals

- **Skill** of A over B = 1 − SSE(A) / SSE(B) on identical rows; MSE reduction in % = 100 × skill.
- **Per design** (V1, V2, V3 pooled over fifths 2–5, V4a, V4b), on all test rows of the set: RMSE of O, MED3, T0, TX, C0, CX, CB, CE (G and H beside, no statement uses them); skill over O of each of the others; C0 over T0; CX, CB, CE over C0; TX over T0; (CX over C0) − (TX over T0); the AUC table; the screening table. Under V1 also the strata (section 8) and the two controls (section 7).
- **Points.** V1, V2: the mean of the three re-partitions' or assignments' values (RMSE: mean of the per-split RMSE). V3 pooled, V4a, V4b: the scored rows.
- **Intervals.** 95 %, 2.5 and 97.5 % quantiles of 20,000 draws. Units resampled with replacement in one level and the per-unit sums re-added; V1 a re-partition drawn first, V2 an assignment drawn first, V3 and V4 the units of the scored rows. Two forms: discharges (the discharge form) and shot clusters (the cluster form; a statement is judged on it). Units: those with at least one row in the cell.
- **Streams.** Seed 20261177. One stream per cell, `numpy.random.default_rng([20261177, crc32(name)])`, shared by every quantity of the cell, so that differences and ratios are formed draw by draw: name = "design|stratum|form" for the squared-error cells (e.g. "V1|all|cluster", "V1|block up to 31991|discharge", "V1|rB below 0.9|cluster", "V1|lag rows|cluster"), "design|flags|form" for the AUC cells.
- **Order of work.** All fits and the checks of section 11 first; the readout computes every number into the JSON in one pass, then renders the markdown. The progress log prints row counts and times only.

## 10. Statements: how each is judged (cluster form)

| Statement | Quantity | Holds if |
|---|---|---|
| A1 | C0 over O, V1 | lower end above zero |
| A2 | C0 over T0, V1 | lower end above zero |
| A3, A4, A5, A6 | CX over C0 under V1, V2, V3 (fifths 2–5 pooled), V4a | lower end above zero |
| A7 | (CX over C0) − (TX over T0), V1, same rows and draws | lower end above zero |
| A8 | AUC(FHX) − AUC(FH) under V1 and under V4a | both lower ends above zero |
| A9 | MSE(O, rows kept by FHX) / MSE(O, rows kept by FH), V1, q = 10 % | upper end below 1 |
| A10 | CE over C0, V1 | lower end above zero |
| A11 | CXp over C0, V1 | lower end at most zero ("contains zero or lies below it") |

Expectations are copied from the registration; "expectation met" is read as in B.75 (verdict against "holds"; "uncertain" not applicable; the stated range checked on the point).

## 11. Checks (registration section 9), run before the readout

| Check | What is compared | Pass rule |
|---|---|---|
| 1a | T0 and TX of this batch against B.75's stored D10 L5 estimates (`data/.b75/fits/{unit}__{T0,TX}__D10__L5.npz`) on the rows both have, every unit | reported (largest and median absolute deviation); equality not required |
| 1b | (added) T0 and TX refitted with this batch's code on B.75's common set for L5, units V1_r2_k0 and V4a | bit-identical to B.75's stored estimates |
| 1c | C0 with the four frame scalars removed (the same code with that block empty) against T0, units V1_r2_k0, V2_s1_k0, V3_q1, V4a, V4b | bit-identical |
| 2 | Leakage. 200 test rows of unit V1_r2_k0 (stream `crc32("leakage")`). For each, a copy of its file in which (a) every T_i value of a row after t is removed; (b) raised by 1000 eV; (c) multiplied by a factor drawn from [0.5, 1.5], half of the missing values after t given a value, and the fast channels of every row after t multiplied by factors drawn from [0.5, 1.5]. Recomputed from the copy: the D10 history inputs (`common75.file_inputs`), the four frame scalars, MED3, the eight fast scalars and the anchor scalars (in (c), with `Feats.window`, `zmeans`, `run_mean10` and `raw_means` on the modified fast channels and the unit's standardisation), and the estimates of C0 and CX (models refitted in the check process; their test estimates must first equal the stored ones bit for bit) | every input and estimate identical bit for bit |
| 2+ | (added, positive control) the same rows with y of row t raised by 1000 eV, and with the fast channels of row t multiplied by 1.5 | the frame scalars, resp. the fast scalars, change on every row |
| 3 | (added) L5 recomputed from the grid equals `tg_L5`; H equals the grid value of the anchor row; at least three earlier values on every row; L5 above zero | all true |

If a check fails it is not loosened: the cause is found; a bug of the implementation is fixed and the checks rerun; a real difference is reported to the session lead and the readout waits.

## 12. After the checks and the readout (2026-11-20): what happened, nothing changed

- All 390 fits (36 units x 10 learners, plus CXp and CXlag on the 15 V1 units) were run once, in two processes; no fit was repeated or changed.
- Checks 3 and 1b were run before the fits, checks 1a, 1c and 2 after them, each once, and all pass (1a has no pass rule). After the first run of check 1a (largest deviations 981 eV for T0 and 997 eV for TX, median 2.8 and 4.7 eV), the rows of those deviations were looked at (high, scattered histories near 2 keV) and two counts (rows above 100 eV and above 300 eV) were added to the report of check 1a; the check was rerun with the same comparison. No score against L5 was computed in the checks.
- Positive control of check 2: raising y of row t by 1000 eV changed the four frame scalars on all 200 rows; on one row both the C0 and the CX estimate stayed the same (the raised scalars fall into the same bins of the trees). The pass rule (inputs change on every row) is met.
- The readout was run once, end to end (90 s). The rows without the fourteen fast scalars of row t - 1 are 566 (section 1 said about 565; one more row lacks a3 at row t - 1). The AUC points per sub were compared with `sklearn.metrics.roc_auc_score` and agree to six decimals. The stratum of the undefined r_B class is labelled "rB r_B undefined" in the output (its stream name is "V1|rB r_B undefined|form").
