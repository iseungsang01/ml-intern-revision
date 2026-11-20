# B.81 — the two controls and the two halves of the fast inputs under the stale-history rules, every design

Registered 2026-11-20, before any run of this batch. CPU only (boosted regressions of the kind of B.75; no network, no GPU). The session lead designed it. An AI agent implements and runs it under the author's supervision. No existing file is edited.

## 1. Why

`PAPER_DESIGN.md` section 6 requires two controls beside every table of fast inputs: the fast inputs permuted over rows and the estimator refitted (the reduction should be zero), and the fast inputs taken at the anchor row instead of row t (are the samples of the present needed). For the head rules of the paper, the history 0.65 s old (D650) and the beam-modulation schedule (S), these controls exist only as an unregistered check of the session lead on one unit of the chronological design, without an interval (`docs/paper/b75_lead_checks.md`): permuted +0.013, at the anchor row -0.128, present +0.205. Registered controls with intervals exist for other rules (B.78 at K10, B.79 at D650 for the estimator with beam scalars, B.76 on MAST).

The table "which diagnostic" has the ECEI half (TE) and the BES half (TB) under the random and the chronological design only; B.75 did not fit them under the cluster-blocked and the rolling-origin design.

This batch fills both gaps so that the manuscript's tables carry the registered controls for the rules they report, and the two halves under all four designs.

**Status.** Exploratory on used data. Known when this was written: every readout of B.75, B.78, B.79 and B.80, the lead's one-unit control values above, and the unregistered diagnostic on rows where the ECEI is constant (`docs/paper/b75_stuck_ecei.md`). Not known: any control under V1, V2 or V3 at D650 or S, any interval of a control at these rules, and TE or TB under V2 or V3.

## 2. Data, rows, target, units

Exactly those of B.75: the D1 rows of `data/.b73/rows.npz`, the inputs of `data/.b75/inputs/{D650,S}.npz`, the common set (D650) and the S set (S) of `data/.b75/sets.npz`, target L5, and the fit and test units of B.73 (V1: r2 to r4 x 5 folds; V2: three assignments x 5 folds; V3: fifths two to five; V4a: fit to 31991, scored from 32312; V4b: fit to 32310, scored from 32312). The stored estimates of T0 and TX of B.75 (`data/.b75/fits/`) are the references and are not refitted.

## 3. Estimators fitted here

Same regression as B.75 (`HistGradientBoostingRegressor(random_state=20261116)`, default settings, target L5 minus the causal GP estimate as in B.75; the agent follows `b75_availability/stage_fits.py` and states in the addendum which function it reuses).

- **TXp (permuted).** TX's inputs with the fourteen fast and anchor columns permuted jointly over rows, within the fit rows and within the test rows separately; one permutation of each per unit and rule, stream `[20261181, crc32("placebo|<unit>|<rule>")]`; the ten history columns stay in place. (Definition of B.78 Addendum 1.)
- **TXo (fast inputs at the anchor row).** The ten history scalars, the eight fast scalars of the anchor row, and a1 of BES and of ECEI; a2 and a3 are zero by construction and left out. Rows whose anchor-row fast scalars are not finite are kept (the regression handles missing inputs), so the rows are those of T0 and TX. (Definition of B.78 Addendum 1.)
- **TE, TB** as defined in B.75 Addendum 1, for the units of V2 and V3 (B.75 holds them for V1 and V4).

Rules D650 and S; all units of V1, V2, V3, V4a, V4b for TXp and TXo.

## 4. Intervals

As B.75: 95 %, 20,000 draws, shot clusters resampled (re-partition or assignment drawn first under V1 and V2), seed stream `[20261181, crc32(<statement id>)]`. The discharge form is reported beside it. A statement is judged on the cluster interval.

## 5. Statements

Skill of A over R = 1 - MSE(A)/MSE(R) on the same rows.

| id | statement | rule of judgement | expectation |
|---|---|---|---|
| C1 | V1, D650: TXp over T0 | the interval contains zero or lies below it | holds |
| C2 | V1, S: TXp over T0 | the interval contains zero or lies below it | holds |
| C3 | V1, D650: TX over TXo lies above zero (the present samples add to those of the anchor row) | lower end above zero | holds |
| C4 | V1, S: TX over TXo lies above zero | lower end above zero | holds |
| C5 | V2, D650: TX over TXo lies above zero | lower end above zero | holds |
| C6 | V3, D650: TX over TXo lies above zero | lower end above zero | uncertain, leaning to holds |
| C7 | V4a, D650: TX over TXo lies above zero | lower end above zero | uncertain (the lead's one-unit point is +0.205 for present, -0.128 for the anchor row, no interval) |
| C8 | V2, S: TX over TXo lies above zero | lower end above zero | holds |
| C9 | V3, S: TX over TXo lies above zero | lower end above zero | uncertain, leaning to holds |
| E1 | V2, D650: TE over T0 lies above zero | lower end above zero | holds |
| E2 | V3, D650: TE over T0 lies above zero | lower end above zero | uncertain, leaning to holds |
| E3 | V2, D650: TB over T0 lies above zero | lower end above zero | holds |
| E4 | V3, D650: TB over T0 lies above zero | lower end above zero | uncertain (under V4a it is -0.014 [-0.257, +0.203]) |
| E5 | V2, S: TE over T0 lies above zero | lower end above zero | holds |
| E6 | V3, S: TE over T0 lies above zero | lower end above zero | uncertain, leaning to holds |
| E7 | V2, S: TB over T0 lies above zero | lower end above zero | holds |
| E8 | V3, S: TB over T0 lies above zero | lower end above zero | uncertain |

**Reported without a statement:** TXo over T0, TXp over T0 and TX over TXo for every design and both rules (with cluster intervals); V4a and V4b under S; the same three with the rows left out where the ECEI is constant at row t or at the anchor row (the definition of `lead_checks/stuck_ecei.py`: all four ECEI channels unchanged for at least ten rows), under V1 only, as a description beside the main values (after the fact, since that stratum was defined after B.75 had been read).

## 6. Checks before the readout (each run once; a failed check stops the batch and is reported)

- **Reuse.** For unit `V1_r2_k0`, rule D650, target L5: T0 and TX refitted with this batch's code equal the stored estimates of B.75 bit for bit. For the same unit TE and TB equal the stored ones bit for bit.
- **Permutation.** In TXp the ten history columns are bit-identical to those of TX, and each of the fourteen permuted columns has the same multiset of values as in TX, within fit rows and within test rows.
- **Anchor row.** For 200 random test rows of that unit the eight fast scalars of TXo equal the fast scalars of TX computed at the anchor row's grid index, bit for bit.
- **Rows.** Every estimator of a comparison is scored on the same rows (count per unit and rule printed).

Choices that this text leaves open are written into `PREREGISTRATION_B81_ADDENDUM1.md` before the checks and before any readout. This file is not edited after registration.

## 7. What goes into the paper (decided now)

- **Controls table.** The registered values of this batch replace the one-unit check of the session lead in the table of controls. If C1 and C2 hold, the table shows that permuted fast inputs give no reduction under the two head rules. If either does not hold (lower end above zero), the manuscript states the size of the reduction with permuted inputs beside every fast-channel increment of that rule and reads the increments net of it.
- **Present samples.** If C3 holds, the manuscript may say for the random design that the fast samples of the present add to those taken when the newest visible CES value was taken; for each other design the sentence is written only if its statement (C5, C6, C7; C4, C8, C9 for the schedule) holds, and otherwise the table shows the interval and the text says it is not established there. If C3 does not hold, the manuscript does not attribute the increment to the present samples and says that fast samples as old as the CES value give the same reduction.
- **TXo over T0 is reported as it is.** A positive value means that part of the increment is carried by where the fast level stood at the anchor row (a state of the discharge), which the manuscript then says.
- **Which diagnostic.** The dashes of the table are filled. A sentence on the ECEI half alone names the least favourable design in which its statement holds; a design in which it does not hold is named as such. The same for the BES half.
- No estimator of the paper changes because of this batch.

## 8. Output

Code in `ces_prediction/experiments/b81_controls/`; caches in `data/.b81/` (git-ignored); `docs/paper/b81_controls.md` and `.json` (every statement with value, cluster interval, discharge interval, verdict, expectation; the reported cells; the counts), `docs/paper/b81_checks.md`.
