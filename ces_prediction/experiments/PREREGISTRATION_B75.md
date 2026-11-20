# B.75 — the estimate when the CES history is as old as it really is

Registered 2026-11-20, before any run of this batch. CPU only: a causal GP, boosted regressions and re-reads of cached rows; no network is trained or run. The session lead designed it. An AI agent implements and runs it under the author's supervision. No existing file is edited.

## 1. Why

Every causal estimator so far reads the CES value of the previous frame. The fourth root of the CXRS critique (`docs/paper/review_loop/critique_20261120_cxrs.md`, Objection 4): the analysis that makes these values cannot deliver a value until the next beam-off frame, so the newest value it could deliver is 0.65 s old at the median (unregistered check C12). B.71 replayed a stop of the updates with the recurrent network as trained on fresh history; from 50 ms on it was not below the causal GP. That replay did not fit an estimator to the stale history, because a network needs a GPU run per condition. By the rule registered in B.73 the paper's estimator needs no network, so the fit can now be repeated for every availability of the history.

The author's question of 2026-11-20: what is gained by not calling the task nowcasting. The author's statement of the same day: the chord at R = 2.198 m is at the pedestal, and the work exists because the CES value is inaccurate there. This batch measures the two things those statements turn on:

1. With the CES history d old and the fast samples of the present, how far is the estimate from the level, and what do the fast channels add? This is the nowcasting case as it would occur.
2. Against the level of the following frames, is the causal estimate closer than the frame's own reading? This is the accuracy statement of the motivation.

**Status of this registration.** No untouched data exist. The rows, targets, designs and learners are those of B.73, whose results are known. The availability rules and the anchor inputs of section 4 are new and are fixed here before any number under them has been read.

## 2. Data

- Rows: the D1 rows of B.70 as cached by B.73 (`data/.b73/rows.npz`, `data/.b73/grid.npz`): 639 discharges, T_i, values above 3 keV and held repeats withheld.
- Targets of a scored row t (as B.73): **F**, the reported value of row t; **L5**, the median of the measured values among rows t+1 to t+5 of the same block, at least three measured.
- Blocks: cut at a time step of 0.5 s or more, as before.

## 3. Availability rules

For a row t with stamp s(t), the rule says which measured values of the same block an estimator may read.

| Rule | Visible values |
|---|---|
| D10 | stamp at most s(t) − 10 ms (every earlier row; the condition of every earlier batch) |
| D50, D100, D300, D650, D1000 | stamp at most s(t) − d, d = 50, 100, 300, 650, 1000 ms |
| S (schedule) | stamp at most the stamp of the last row before the most recent absent step that ends at or before row t |

- Tolerance of every stamp comparison: 3 ms.
- **Absent step:** a time step between consecutive grid rows of the same block above 15 ms and below 45 ms (one to three frames taken without the beam; `docs/paper/file_rules.md`). It "ends" at the first row after it.
- Rule S stands for the beam-modulation analysis: the frames between two beam-off frames can be analysed once the second one has been taken. Zero analysis time is assumed, which favours the estimators. A row with no absent step earlier in its block has no visible value under S and is not scored under S; its share is reported.
- The fast samples (BES, ECEI) of rows up to and including t are visible under every rule.

## 4. Estimators

All are fitted per rule and per target; "T0[D650, L5]" is T0 with the D650 history fitted to L5.

| Name | Definition under a rule |
|---|---|
| H | the newest visible value |
| M5 | the median of the last five visible values |
| G | `baselines_interpolation.predict_gp_causal` on the last up to 16 visible values, evaluated at s(t) (the causal GP of the earlier batches; no fit to a target) |
| T0 | G plus a boosted regression of (target − G) on the ten history scalars of B.73 computed from the visible values: time step of row t, age of the newest visible value, H, G − H, \|G − H\|, and of the last ten visible values the mean, the standard deviation, H minus the mean, the last increment, the number present (undefined with fewer than three) |
| TX8 | T0's inputs plus the eight fast scalars of row t (B.73's TX) |
| TX | TX8's inputs plus six anchor scalars (below) |
| TB, TE | T0's inputs plus the BES (ECEI) half of TX's fast scalars: four of row t and three anchor scalars |
| X0 | no CES history: the median target of the fit rows plus a boosted regression of (target − that median) on the time step and the eight fast scalars of row t |
| O | the frame's own reading, the reported value of row t (scored against L5 only; not an estimator that exists before the frame) |

**Anchor scalars.** The anchor row is the row of the newest visible value. For the BES array mean and for the ECEI array mean, with m(i) the mean of the samples of the up to ten grid rows of the same block ending at row i (at least five rows):

- a1: m(anchor), standardised as the eight fast scalars are (statistics of the fit discharges);
- a2: m(t) − m(anchor), same standardisation;
- a3: ln(m_raw(t) / m_raw(anchor)), from the unstandardised samples, where both are positive.

a3 does not depend on a constant factor of the channel. The reason for the anchor scalars: the eight fast scalars of B.73 span 100 ms, and under a stale history what matters is the change of the fast level since the newest visible value was taken.

- Boosted regression: `HistGradientBoostingRegressor(random_state=20261116)`, default settings, no tuning (as B.73).
- **Applied without refit (the condition of B.71):** T0 and TX fitted under D10 are also evaluated on the inputs of every other rule, the age being the true one.

## 5. Rows of each fit and score

- **Complete under a rule:** the ten history scalars, G and the fourteen fast scalars of TX are finite, and the target exists.
- **Common set (per target):** the D1 rows complete under each of the six fixed-delay rules. Every fit and every score under a fixed-delay rule uses the common set only, so that the six rules differ in the availability of the history and in nothing else.
- **S set:** the D1 rows complete under S. Fits and scores under S use it. Beside them, T0 and TX with the D10 history are fitted and scored on the S set ("fresh history, same rows").
- For the check of section 9, T0 and TX8 under D10 are also fitted on all rows complete under D10, as B.73 did.
- Counts of every set per design are reported.

## 6. Validation designs

The units of B.73 (`b73_targets/designs.py`): V1 random (15 units), V2 whole shot clusters held out (15), V3 rolling origin (4), V4 chronological (fit to shot 31991 and fit to shot 32310, scored from shot 32312).

| Learners | Units | Rules | Targets |
|---|---|---|---|
| T0, TX | all | D10, D50, D100, D300, D650, D1000, S, fresh history on the S set | F, L5 |
| TX8, TB, TE | V1, V4 | D10, D650, S | F, L5 |
| X0 | all | (common set) | F, L5 |

**V3c, the control for the size of the fit.** For each rolling-origin unit, a fit on the same number of discharges drawn at random (three draws, seeds 1 to 3) from all discharges outside the test fifth. T0 and TX, rules D10 and D650, target L5. It separates "fitted on earlier discharges only" from "fitted on fewer discharges". For the last fifth the control equals the unit and is left out.

## 7. Readouts

Per design, target and rule, on identical rows:

- MSE and RMSE of H, M5, G, T0, TX8, TX, X0, and of O against L5;
- skill of T0 over G and over H; the fast-channel increment (TX over T0, TX8 over T0, TB over T0, TE over T0) as MSE reduction in %;
- the cost of staleness: MSE under the rule divided by MSE under D10, for T0 and for TX (fixed-delay rules: common set; S: against the fresh-history fit on the S set);
- the recovered share: (MSE(T0) − MSE(TX)) / (MSE(T0) − MSE(T0 under D10)) under the same rule and rows;
- the estimators applied without refit beside the refitted ones;
- under V1, every increment by chronological block of the test discharge (up to shot 31991; 32002–32310; from 32312);
- V3 beside V3c, per fifth;
- under S, the distribution of the age of the newest visible value on the S set.

**Computation time.** Wall time per row of G plus the tree prediction of TX, one thread, on 2,000 random rows: median and 99th percentile.

**Intervals.** 95 %, 20,000 draws, shot clusters resampled; the discharge form beside it. Under V1 a re-partition is drawn first, under V2 an assignment; under V3 and V4 the test clusters are resampled. A statement is judged on the cluster interval.

No stratum is defined by the scored value or by later values.

## 8. Statements

"Increment" = TX over T0 under the same rule on the same rows.

| Id | Statement | Expectation |
|---|---|---|
| R1 | V1, L5, common set: MSE(T0 under D650) / MSE(T0 under D10) lies above 1.5 | holds (2 to 6) |
| R2 | V1, L5, D650: the increment lies above zero | holds (+0.05 to +0.25) |
| R3 | V1, L5: the increment under D650 exceeds the increment under D10 (interval of the difference above zero) | uncertain, leaning to holds |
| R4 | V2, L5, D650: the increment lies above zero | holds |
| R5 | V3, L5, D650: the increment lies above zero | uncertain |
| R6 | V4 (fit to 31991), L5, D650: the increment lies above zero | uncertain |
| R7 | V1, L5, rule S: the increment lies above zero | holds |
| R8 | V4 (fit to 31991), L5, rule S: the increment lies above zero | uncertain |
| R9 | V1, L5, D650: TE over T0 lies above zero | uncertain (+0.02 to +0.10) |
| R10 | V1 and V4, L5, D650: T0 over G lies above zero under each | holds |
| R11 | V1 and V4, L5, D10, common set: the MSE of TX lies below that of the frame's own reading O under each (interval of 1 − MSE(TX)/MSE(O) above zero) | holds (+0.3 to +0.6) |
| R12 | V1, L5, rule S: the MSE of TX lies below that of O | does not hold |
| R13 | L5, D10, fifths 2 and 3 pooled: the increment of V3 minus the increment of V3c lies below zero (the order in time costs more than the size of the fit) | uncertain |

Each statement is judged separately; the tables of section 7 are reported whole, whatever the verdicts.

## 9. Checks (before any readout)

- G under D10 equals the cached causal GP of `data/.b64/base.npz` (largest deviation at most 1e-6 eV).
- T0 and TX8 under D10 fitted on all rows complete under D10 reproduce the stored estimates of B.73 for units V1_r2_k0 and V4a, targets F and L5 (largest deviation at most 1e-6 eV; otherwise the difference is reported and explained before any readout).
- Visibility: for 200 random test rows under D650 and 200 under S, the inputs are recomputed from a copy of the file in which every T_i value later than the row's visibility limit is (a) removed and (b) raised by 1000 eV; all inputs are unchanged bit for bit.
- The ages under S reproduce the reviewer's check C12 on the valued grid rows with an absent step earlier in the block (median within 0.02 s of 0.65 s; shares of the bands 0–0.1, 0.1–0.3, 0.3–1, above 1 s within 0.01 of 0.071, 0.166, 0.498, 0.266). A difference is reported with its cause.
- The units are those of B.73 (same discharges per unit).

## 10. What goes into the paper (decided now)

- The table of RMSE by availability rule is printed whatever the verdicts, with the sentence that zero analysis time is assumed under S. It replaces every sentence about real-time use.
- **The word "nowcasting".** The session lead's rule, for the author to decide on: the word stays in the title and abstract if R4 and R5 hold (the increment under a 0.65 s old history holds with whole shot clusters held out and forward in time). If R2 and R4 hold and R5 does not, the word is used in the text only, with the design named, and the title says causal estimation. If R2 fails, the word is not used. "Pedestal top" stays in the title in every case: it is the location of the chord and the motivation (author, 2026-11-20).
- If R11 holds: the first result of the abstract is the error against the following level, with the frame's own reading, hold and the causal GP as references.
- If R12 does not hold, as expected: the text states that with the present analysis schedule a stale history does not replace the reading of the frame, with the two RMSE values.
- R13: if it holds, the text says the increment does not transfer forward in time at equal fit size; if the interval contains zero, that the forward-in-time designs do not separate the order in time from the number of discharges.

## 11. Limits known in advance

- Exploratory on used data (section 1). The anchor scalars are new inputs chosen before any readout; TX8 beside TX shows what they do.
- Rule S assumes the beam-modulation analysis with zero computation time. A reader of the frame's own spectrum would give another availability; no such values exist for these discharges.
- The history values are the offline product in every rule.
- The following level is built from reported values; a bias of the analysis that lasts 50 ms or more is in the target too. R11 is about scatter around the level, not about a systematic error of the diagnostic.
- The fixed-delay rules drop the first second of every block (the common set).
- One estimator per rule is fitted. An estimator fitted once for all ages is not part of this batch; the rows "applied without refit" bound it from one side.
- V4 is one block of about 95 discharges; V3c is informative for fifths 2 and 3 only.

## 12. Files

- Code: `ces_prediction/experiments/b75_availability/` (imports `b73_targets/common.py` and `designs.py`; edits nothing).
- Caches: `data/.b75/`. Output: `docs/paper/b75_availability.{md,json}`, checks `docs/paper/b75_checks.md`.
- Seed 20261175, one stream per quantity (crc32 of its name).
- Implementation choices that this text leaves open go into `PREREGISTRATION_B75_ADDENDUM1.md`, written before any readout. This file is not edited after the first run.
