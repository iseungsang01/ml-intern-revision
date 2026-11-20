# B.86 — MAST from the core to the edge: the same estimator at every radius

Registered 2026-11-21, before any run of this batch. CPU only (public FAIR-MAST data already on disk, the row builder and the estimators of B.76; boosted regressions; no network, no GPU). The session lead designed it. An AI agent implements and runs it under the author's supervision. No existing file is edited.

## 1. Why

The author's question of 2026-11-21: is nowcasting at the pedestal worth anything for control, given that the sensors of the KSTAR files sit at the edge and the core is not known, or is this a structure with which everything up to the inside can be nowcast? The KSTAR files hold one chord and the channels around it, so the question cannot be answered there. The method itself needs, for any chord, its charge-exchange history and a fast diagnostic at its radius. MAST has charge-exchange chords and Thomson scattering over the whole radius in the public data. B.76 used only the pedestal-top band (chords 1.5 to 5 cm inside the edge of the Thomson density profile). This batch runs the same estimator in bands from that edge to the core and reports, by radius, the error, the cost of a stale history and what the Thomson values at the chord's radius add. It belongs to the section on use of the paper (`PAPER_DESIGN.md` section 9, question 4) and carries no statement about the KSTAR core.

**Status.** Exploratory on used data (the MAST shots were read in earlier batches). Known when this was written: B.76 for the pedestal-top band (Thomson increment at 100 ms +0.305 [+0.247, +0.371] under M1; with the previous frame visible +0.107 [+0.064, +0.152]); an earlier unregistered look by radius (the share of failed fits rises towards the edge, 0.270 to 0.695, and the lag-1 autocorrelation of the increments falls from +0.299 to -0.049: the series is smoother inside); that on MAST the interpolation that waits has a much lower error than a causal estimate at the pedestal top (earlier batch). Not known: any error, cost of staleness or Thomson increment of T0 and TX outside the pedestal-top band.

## 2. Data, rows, bands

- Shots, frames (5 ms), good-value rule, Thomson windows and the definition of the distance to the density edge exactly as B.76 (`ces_prediction/experiments/b76_mast_availability/`, `PREREGISTRATION_B76.md` and its addendum). The row builder of B.76 is reused with the band as a parameter.
- **Bands** by the distance d of the chord inside the edge of the Thomson density profile (B.76's `dband`), in cm: E = [1.5, 5) (the pedestal-top band of B.76), B2 = [5, 10), B3 = [10, 20), B4 = [20, 35), B5 = [35, 60). The agent reports per band the rows, shots, sessions, the median normalized radius rho (as B.76 defines it) and the median reported T_i. A band with fewer than 5,000 rows in the common set of its rules is reported and left out of the statements.
- A row needs, as in B.76, a good T_i value, a Thomson value at the chord's radius at frame t and at the anchor frame, and the target.

## 3. Targets, rules, estimators, designs

- Targets: the level L5 (median of the good values of the next five frames, at least three) for all statements; the value of the frame F for the reported comparison with the references that wait.
- Rules: D5 (previous frame visible, the nowcast condition) and D100 (history 100 ms old), each band on the rows complete under both rules.
- Estimators, as B.76: hold H, causal GP G, T0 (GP and boosted regression on the history scalars and rho), TX (T0's inputs and the nine Thomson scalars), one fit per band, rule and target. O, the reading of the frame, for L5.
- References that wait, for F under D5 only: PCHIP and the offline GP of `baselines_interpolation.py` through the good values of the chord with frame t left out (as B.83 on KSTAR).
- Designs: M1 random (re-partition 1 of B.76, five folds of whole shots) and M3 rolling origin (fifths two to five), with sessions resampled.

## 4. Intervals

As B.76: 95 %, 20,000 draws, sessions resampled, seed stream `[20261186, crc32(<statement id or cell name>)]`; the shot form beside it; verdicts on the session interval. Skill of A over R = 1 - MSE(A)/MSE(R) on identical rows.

## 5. Statements (target L5)

For a band b of {E, B2, B3, B4, B5}:

| id | statement | rule of judgement | expectation |
|---|---|---|---|
| R1.b | M1, D100: TX over T0 lies above zero in band b | lower end above zero | holds in every band |
| R2.b | M1, D5: TX over T0 lies above zero in band b | lower end above zero | holds in E (B.76); uncertain inside |
| R3.b | M3, D100: TX over T0 lies above zero in band b | lower end above zero | holds in E (B.76); uncertain, leaning to holds, inside |
| R4.b | M1, D5: T0 over G lies above zero in band b (the learned history-only estimate against the causal GP) | lower end above zero | holds in every band |

And across bands:

| id | statement | rule of judgement | expectation |
|---|---|---|---|
| R5 | M1: the cost of staleness of T0 at D100 (MSE at D100 over MSE at D5, same rows) is larger in band E than in the innermost band with statements | lower end of the difference above zero | holds (the edge changes faster) |
| R6 | M1, D100: the Thomson increment (TX over T0) in band E minus that in the innermost band with statements lies above zero | lower end of the difference above zero | uncertain |
| R7 | M1, D5: the share of estimates of TX within 10 % of the level is larger in the innermost band with statements than in band E | lower end of the difference above zero | holds (the series is smoother inside) |

Differences across bands are formed on common draws of sessions.

Reported without a statement: per band, rule and design the RMSE, the mean and median absolute error, the median relative error and the shares within 10 % and 20 % of the level for H, G, T0, TX and O; per band under D5 and M1 the value of the frame: RMSE of H, G, T0, TX, PCHIP and the offline GP and the skills of TX over the two references that wait; the recovered share at D100.

## 6. Checks before the readout (each run once; a failed check stops the batch and is reported)

- **Band E.** The rows of band E built by this batch equal B.76's rows (`data/.b76/rows.npz`) bit for bit in every field used, and T0 and TX of one M1 unit under D100 equal B.76's stored estimates bit for bit.
- **Visibility.** For 200 random test rows of band B4 under D100 the history inputs and the Thomson scalars are recomputed from a copy of the shot in which every T_i value younger than the rule allows is raised by 1000 eV: unchanged bit for bit.
- **References.** For 200 random rows the two references that wait do not change when the value of frame t of the chord is raised by 1000 eV.
- **Rows.** Every estimator of a comparison is scored on the same rows; counts per band, rule and design printed.

Choices that this text leaves open are written into `PREREGISTRATION_B86_ADDENDUM1.md` before the checks and before any readout. This file is not edited after registration.

## 7. What goes into the paper (decided now)

- A table "MAST from the core to the edge" enters the section on use: per band the error with the previous frame visible, the share within 10 % of the level, the cost of staleness at 100 ms and the Thomson increment under the two rules and the two designs.
- If R1 holds in every band, the manuscript says that on MAST the estimator applies from the core to the edge and that a fast diagnostic at the chord's radius adds to a stale charge-exchange history at every radius. For a band in which R1 or R3 does not hold, the text names the band.
- R5, R6 and R7 give the sentence on where a nowcast is needed most and where it is most accurate; they are reported whatever they are. If R7 holds, the discussion says that inside the pedestal the requirement of 10 % is met more often, with the numbers of the table, as a statement about MAST.
- No sentence transfers these statements to the KSTAR core, and no sentence speaks of control beyond what the table shows.

## 8. Output

Code in `ces_prediction/experiments/b86_mast_radius/`; caches in `data/.b86/` (git-ignored); `docs/paper/b86_mast_radius.md` and `.json` (statements with value, session interval, shot interval, verdict, expectation; the tables; counts per band), `docs/paper/b86_checks.md`.
