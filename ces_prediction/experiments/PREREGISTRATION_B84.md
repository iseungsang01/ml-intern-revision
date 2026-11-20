# B.84 — is the second stage too bound to rules? One fit for every scenario, raw windows in place of hand-made scalars, and the scalars without absolute levels

Registered 2026-11-21, before any run of this batch. CPU only (boosted regressions of the kind of B.75; no network, no GPU). The session lead designed it. An AI agent implements and runs it under the author's supervision. No existing file is edited.

## 1. Why

The author's remarks of 2026-11-21: the second-stage estimator was attached by rules, and this may be bad for connecting the estimator to the scenarios of use; fixing so much by rules may have distorted it.

What the records say about it. The paper's estimator is a causal GP with a boosted regression on 24 hand-made scalars (ten of the visible history, eight of the present fast samples, six that tie the present to the anchor row), fitted separately for every availability rule. Three findings point at the rules:

1. **Per-rule fits.** An estimator fitted with the previous frame visible and then given a history 0.65 s old has an RMSE of 270.3 eV (T0) and 259.7 eV (TX) where the estimators fitted under that rule have 226.2 and 184.3 eV, and its fast-channel increment is +0.077 [+0.043, +0.110] where the refitted one is +0.336 [+0.285, +0.380] (B.75). The section on use therefore does not apply the estimator found for the nowcast; it fits a new one for each scenario.
2. **Absolute levels.** The fast scalars taken at the anchor row add +0.087 under the random design and do harm in time order (-0.411 [-0.940, -0.118] under rolling origin, B.81): a tree on absolute standardized levels memorizes what a level meant in the fit discharges. Under the schedule rule the increment is not established in the last block of shots (B.75), and its sign changes with the seed (B.82).
3. **Hand-made scalars.** The windows (ten rows), the departure, the standard deviation, the increment and the log ratio were chosen by hand in an earlier batch on the same discharges.

This batch asks, with existing data and without a network: (A) does one fit across all ages of the history serve every scenario as well as the per-rule fits; (B) do raw windows do as well as the hand-made scalars; (C) do the scalars without absolute levels do better in time order. A model with state trained across scenarios (the recurrent network with randomized delay and thinning of the CES history) is the other way to answer the author's remark; it needs a GPU, which is paused, and is not part of this batch.

**Status.** Exploratory on used data. Known when this was written: every readout of B.73 to B.82 (B.83 is running). Not known: any fit across rules, any fit on raw windows under a stale history, any fit without the absolute levels.

## 2. Data, rows, target, units

Those of B.75: D1 rows, inputs `data/.b75/inputs/{rule}.npz`, the common set (fixed delays) and the S set, target L5, the regression of B.75 (`HistGradientBoostingRegressor(random_state=20261116)`, default settings, fitted to L5 minus the causal-GP estimate G of the rule). Units: re-partition r2 of V1 (five folds), V3 (fifths two to five) and V4a (fit to 31991). The stored per-rule estimates of B.75 (T0, TX; `T0a`, `TXa` = fitted with the previous frame visible and applied) and of B.78 under K10 are the references and are not refitted, except in the reuse check.

**The mixture.** M = {D10, D50, D100, D300, D650, D1000, S}. For a unit, the fit set of a mixture learner stacks, for every rule r of M, the unit's fit rows in the set of r (common set for the fixed delays, S set for S) with the inputs as they are under r and the target L5 minus G_r; a row enters up to seven times with different visible histories. One regression is fitted on the stack. Under a rule r the estimate of a test row is G_r plus the regression's value for the inputs under r. Rule K10 (a value every tenth frame, B.78) is not in the mixture and is a scenario the mixture learners have not been fitted on.

## 3. Estimators fitted here

- **T0all, TXall (part A).** The inputs of T0 (ten history scalars, among them the age of the newest visible value) and of TX (24 scalars), fitted once on the mixture.
- **T0raw, TXraw (part B),** fitted once on the mixture. History: the time step, the age of the newest visible value, G minus H, and for the last ten visible values (newest first) the value minus H and the age of the value in ms; missing where fewer than ten are visible. Fast (TXraw only): the standardized BES and ECEI array means of rows t-9 to t (twenty values) and the ten-row running mean of each at the anchor row (two values); no departure, standard deviation, increment or log ratio. Standardization with the fit discharges of the unit, as in B.75.
- **TXrel (part C),** fitted per rule (D650 and S), as TX in B.75 but without the four scalars that carry an absolute fast level: the level x_t of the BES and of the ECEI array mean, and the anchor scalar a1 of each. Twenty scalars.

## 4. Intervals

As B.75: 95 %, 20,000 draws, shot clusters resampled, seed stream `[20261184, crc32(<statement id or cell name>)]`; the discharge form beside it; verdicts on the cluster interval. Skill of A over R = 1 - MSE(A)/MSE(R) on identical rows. "Not worse by more than 5 %" means: the lower end of the interval of the skill of A over R lies above -0.05.

## 5. Statements

| id | statement | rule of judgement | expectation |
|---|---|---|---|
| U1 | V1, D650: TXall is not worse than the per-rule TX by more than 5 % | lower end above -0.05 | uncertain, leaning to holds |
| U2 | V1, D10: TXall is not worse than the per-rule TX by more than 5 % | lower end above -0.05 | uncertain, leaning to holds |
| U3 | V1, S: TXall is not worse than the per-rule TX by more than 5 % | lower end above -0.05 | uncertain |
| U4 | V1, D650: TXall over TXa (fitted with the previous frame visible, applied) lies above zero | lower end above zero | holds |
| U5 | V1, D650: TXall over T0all lies above zero (the fast-channel increment inside the one fit) | lower end above zero | holds |
| U6 | V1, K10 (not in the mixture): TXall over the per-rule T0 of K10 lies above zero | lower end above zero | uncertain, leaning to holds |
| U7 | V3, D650: TXall is not worse than the per-rule TX by more than 5 % | lower end above -0.05 | uncertain |
| U8 | V1, D650: TXraw is not worse than TXall by more than 5 % (the hand-made scalars are not needed) | lower end above -0.05 | uncertain |
| U9 | V3, D650: TXrel over the per-rule TX lies above zero (without absolute levels the estimate is better in time order) | lower end above zero | uncertain |
| U10 | V4a, S: TXrel over the per-rule T0 lies above zero (the last block under the schedule) | lower end above zero | uncertain, leaning to does not hold |
| U11 | V1, D650: TXrel is not worse than the per-rule TX by more than 5 % | lower end above -0.05 | uncertain |

Reported without a statement: for V1, V3 and V4a and the rules D10, D100, D650, D1000, S and K10: RMSE of T0, TX (per rule), T0a, TXa, T0all, TXall, T0raw, TXraw, and under D650 and S TXrel; the skills of each mixture learner over the per-rule one and over T0all; TXall over T0all for every rule (the increment along the ladder inside one fit); the mean absolute error and the share within 10 % of the level for TX, TXall and TXraw under D10, D650 and S.

## 6. Checks before the readout (each run once; a failed check stops the batch and is reported)

- **Reuse.** T0 and TX of unit `V1_r2_k0`, rule D650, refitted with this batch's code equal the stored estimates of B.75 bit for bit.
- **Mixture.** For that unit: the rows of the stack that belong to rule D650 carry exactly the inputs and targets of the per-rule fit of D650 (bit for bit); the stack holds no test discharge; the number of stacked rows per rule is printed.
- **Raw windows.** For 200 random test rows under D650 the twenty fast values of TXraw equal the standardized array means recomputed from the row's file with the unit's fit statistics, and the history part reads no value younger than the rule allows (recomputed from a copy of the file in which every T_i value not visible under D650 is raised by 1000 eV: unchanged bit for bit).
- **Rows.** Every estimator of a comparison is scored on the same rows; counts per design and rule printed.

Choices that this text leaves open are written into `PREREGISTRATION_B84_ADDENDUM1.md` before the checks and before any readout. This file is not edited after registration.

## 7. What goes into the paper (decided now)

- **One estimator for every scenario (A).** If U1, U2 and U3 hold, the estimator of the paper is the one fitted once across the ages of the history (TXall): the section on which estimator is best and the section on use then speak of the same fitted estimator, and the per-rule fits become a reference in the supplement. If U1 or U3 does not hold, the manuscript says that the second stage has to be refitted for the condition of use, gives the loss of the single fit, and names a model with state trained across conditions as the open step. U6 says whether the single fit carries to a scenario it was not fitted on, and is reported either way.
- **Hand-made scalars (B).** If U8 holds, the manuscript says that raw windows give the same error and that the scalars are a convenience. If TXraw is worse by more than 5 %, it says that the summaries matter for a tree on this amount of data.
- **Absolute levels (C).** U9 to U11 are reported as the diagnosis of the time-order failure. If U9 holds and U11 holds, the lead proposes to the author an estimator without absolute levels; nothing changes without the author's decision.
- No estimator of the paper changes because of this batch without the author's decision.

## 8. Output

Code in `ces_prediction/experiments/b84_onefit/`; caches in `data/.b84/` (git-ignored); `docs/paper/b84_onefit.md` and `.json` (statements with value, cluster interval, discharge interval, verdict, expectation; the reported tables; counts), `docs/paper/b84_checks.md`.
