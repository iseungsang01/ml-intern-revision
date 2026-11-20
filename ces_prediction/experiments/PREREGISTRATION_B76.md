# B.76 — MAST: the estimate under a stale charge-exchange history, with Thomson scattering of the present at the same radius

Registered 2026-11-20, before any run of this batch. CPU only: a causal GP, boosted regressions and two closed forms on the public FAIR-MAST files already on disk; no network is trained or run. The session lead designed it; the author chose the direction ("B": the fast diagnostics carry the level while the charge-exchange value is late) and asked for this batch. An AI agent implements and runs it under the author's supervision. No existing file is edited.

## 1. Why

B.75 (`docs/paper/b75_availability.md`): on the KSTAR pedestal-top chord, with the CES history 0.65 s old, the fast channels reduce the MSE of the level estimate by 19 to 34 % under four validation designs; ECEI alone, an electron-temperature diagnostic at the same radius, gives most of it. The proposal the paper is to be built around: **while the charge-exchange value is late, a diagnostic of the present at the same radius carries the level of the ion temperature.** One device and one pair of diagnostics do not carry such a sentence. MAST has charge-exchange T_i at 5 ms frames and Thomson scattering T_e and n_e at the chord radius, measured by another instrument with another light source.

What is known on MAST (`docs/paper/mast_ped_confirmation.md`): with the previous frame visible (5 ms old), Thomson added to the charge-exchange past gives +0.047 [+0.033, +0.068] at the pedestal top. Nothing has been computed there with a history older than one frame.

A physical reason to expect an effect: at pedestal-top density and temperature the electron-ion energy exchange time is of the order of 10 ms, far below the delays studied here, so a change of T_e at a radius is followed by T_i there. This batch does not test that reason; it tests the estimate.

**Status.** Exploratory on used data: both halves of the MAST shot list (odd-numbered discovery, even-numbered confirmation) were read in the earlier MAST batches with a one-frame history. The availability rules, the anchor inputs, the level target and the validation designs below are new on MAST and are fixed here before any number under them has been read. Nothing from MAST is written as support for a KSTAR number; MAST carries the statement that the direction is not specific to one device and one pair of diagnostics.

## 2. Data

- Shots: every shot of `ces_prediction/experiments/mast/coupling_shots.txt` (both halves) that has a charge-exchange file, a Thomson file (`data/mast/extra/`) and the geometry used before.
- Frames, chords and "good" values: as `mast/explore_fast.py` and `mast/explore_ped.py` build them (population W, a chord good in at least half of the block frames; relative reported error below 0.5), with the one difference that **no frame is hidden**: every good frame of a chord is a candidate scored row and a candidate history value.
- **Pedestal top (as MAST_PED):** the chord lies 1.5 to 5 cm inside the edge of the Thomson density profile of the scored frame.
- Target quantity: T_i. (The velocity is not part of this batch.)

## 3. Targets

For a scored row (chord c, frame t):

| Target | Definition |
|---|---|
| F | the reported value of frame t |
| L5 (following level) | the median of the good values of the same chord among frames t+1 to t+5 (5 to 25 ms later), at least three good |

No estimator reads a charge-exchange value of frame t or later.

## 4. Availability rules

| Rule | Visible charge-exchange values of chord c |
|---|---|
| D5 | frames up to t − 1 (5 ms old; the condition of the earlier MAST batches) |
| D15, D25, D50, D100, D150 | frames whose time is at most t − d, d = 15, 25, 50, 100, 150 ms |

Thomson scattering, D-alpha and the other signals are visible up to the window of frame t under every rule. **The headline rule is D100**, fixed here; the whole ladder is reported.

## 5. Estimators

Fitted per rule and per target.

| Name | Definition under a rule |
|---|---|
| H | the newest visible good value of the chord |
| M5 | the median of the last five visible good values |
| G | `baselines_interpolation.predict_gp_causal` on the last up to 16 visible good values of the chord, evaluated at the time of frame t |
| T0 | G plus a boosted regression of (target − G) on the history scalars: age of the newest visible value, H, G − H, \|G − H\|, and of the last ten visible good values the mean, the standard deviation, H minus the mean, the last increment, the number present (undefined with fewer than three); and the normalised radius of the chord |
| TX | T0's inputs plus the Thomson scalars: T_e and n_e at the chord radius in the window of frame t; the same in the window of the anchor frame (the frame of the newest visible value); their differences (present minus anchor); ln of their ratios (present over anchor); the distance of the chord to the density edge at frame t |
| TD | T0's inputs plus D-alpha: the frame mean and the frame maximum at frame t and at the anchor frame, and their differences, scaled per shot as in `explore_fast.py` |
| TA | T0's inputs plus the Thomson and the D-alpha scalars |
| RT | closed form without a fit: H × T_e(t) / T_e(anchor) (the ion temperature is taken to follow the electron temperature at the radius since the newest visible value) |
| RTa | two coefficients fitted on the fit shots by least squares: ln(estimate) = ln H + α ln(T_e(t) / T_e(anchor)) + β |
| O | the frame's own reported value (scored against L5 only; not an estimator that exists before the frame) |

- Boosted regression: `HistGradientBoostingRegressor(random_state=20261116)`, default settings, as B.73 and B.75. The settings tuned on the MAST discovery half in the earlier batch are not used, so that the learner is the one of the KSTAR batches.
- Thomson values at a chord and frame: `explore_ped.to_chords` and `window_mean`, unchanged.

## 6. Rows of each fit and score

- **Complete under a rule:** the history scalars, G, and the Thomson scalars at frame t and at the anchor frame are finite; the row is in the pedestal-top band; the target exists.
- **Common set (per target):** rows complete under each of the six rules. Every fit and score uses the common set, so that the rules differ in the availability of the history only. Row and shot counts per rule and of the common set are reported.
- TD and TA use the rows of the common set that also have D-alpha; T0 and TX are refitted on those rows for the comparison.

## 7. Validation designs

A session is the calendar day of the shot from the FAIR-MAST metadata; if the metadata carry no time, a run of shot numbers whose consecutive listed shots differ by at most 3 (as on KSTAR). Which one applies is written in the addendum before any readout.

| Design | Fit and test |
|---|---|
| M1 random | shot-level five folds, three re-partitions (seeds 1, 2, 3) |
| M2 session-blocked | whole sessions assigned to five folds of near-equal shot counts, three assignments (seeds 1, 2, 3) |
| M3 rolling origin | the shots in order cut into fifths; fifths 2 to 5 each scored by a fit on the earlier fifths |
| M4 chronological | fit on the first four fifths, score on the last |

T0, TX, RT, RTa: every design, every rule, both targets. TD, TA: M1 and M4, rules D5 and D100.

## 8. Readouts

Per design, target and rule on identical rows: MSE and RMSE of H, M5, G, T0, TX, TD, TA, RT, RTa and of O against L5; skill of T0 over G and over H; the increments TX, TD, TA over T0; RT and RTa over H and over T0; the cost of staleness MSE(rule) / MSE(D5) for T0 and TX; the recovered share (MSE(T0) − MSE(TX)) / (MSE(T0) − MSE(T0 under D5)); the fitted α and β per rule; under M1 the increment by fifth of the shot order of the test shot.

**Controls under M1, D100, L5** (registered because the same controls were informative after B.75):
- placebo: the Thomson scalars permuted over rows (within the fit rows and within the test rows), TX refitted;
- Thomson as old as the history: only the Thomson scalars of the anchor frame are given;
- T0 with the D5 history beside the D100 estimators on the same rows.

**Intervals.** 95 %, 20,000 draws, sessions resampled; the shot form beside it. M1: a re-partition drawn first; M2: an assignment drawn first; M3, M4: the scored sessions. A statement is judged on the session interval.

No stratum is defined by the scored value or by later values.

## 9. Statements

"Increment" = TX over T0 under the same rule on the same rows, target L5.

| Id | Statement | Expectation |
|---|---|---|
| S1 | M1: MSE(T0 under D100) / MSE(T0 under D5) lies above 1.5 | holds (2 to 5) |
| S2 | M1, D100: the increment lies above zero | holds (+0.05 to +0.30) |
| S3 | M1: the increment under D100 exceeds the increment under D5 (interval of the difference above zero) | holds |
| S4 | M2, D100: the increment lies above zero | holds |
| S5 | M3, D100: the increment lies above zero | uncertain, leaning to holds |
| S6 | M4, D100: the increment lies above zero | uncertain |
| S7 | M1, D100: RTa has a lower MSE than H (interval of the skill above zero) | uncertain |
| S8 | M1, D100: TD over T0 lies above zero | uncertain |
| S9 | M1, D100, placebo: the interval of TX over T0 contains zero or lies below it | holds |
| S10 | M1, D5: the MSE of TX lies below that of the frame's own reading O against L5 | uncertain |

Each statement is judged separately; the tables of section 8 are reported whole, whatever the verdicts.

## 10. What goes into the paper (decided now)

- If S2 and S4 hold: the paper states that on a second device, with another electron diagnostic at the same radius, the present electron channel lowers the error of the level estimate under a stale charge-exchange history, with the sizes found and the design named. If S5 or S6 also holds, the sentence names the forward-in-time design.
- If S2 fails: the MAST paragraph says the direction is not confirmed on MAST, and the proposal is stated for KSTAR only.
- S7: if it holds, the closed form is given in the text as the simplest version of the proposal; if not, the text says a plain ratio of two Thomson values is too noisy to carry the level and the regression is needed.
- S9 failing (a placebo increment above zero) stops the batch's use until the cause is found.
- The ladder (RMSE by rule) is printed whatever the verdicts.

## 11. Limits known in advance

- Exploratory on used data (section 1).
- MAST has no beam-modulation latency; the delays are imposed by us. The headline delay of 100 ms is a choice, fixed beforehand; MAST discharges are short, so the common set drops the first 150 ms and more of every chord's record.
- Thomson scattering on MAST has one to two samples per frame and reported errors of its own; RT divides two of them.
- The pedestal-top band is defined from the Thomson density profile, the same instrument that supplies the inputs of TX.
- The following level is built from reported values; an error of the analysis that lasts 25 ms or more is in the target too.
- One estimator per rule; no estimator fitted once for all ages.

## 12. Checks (before any readout)

- Under D5, H equals the last visible value of the earlier MAST row tables for rows that both builds contain (largest deviation reported; the builds differ in that no frame is hidden here, so only rows whose previous frame is good are compared).
- Visibility: for 200 random test rows under D100, the inputs are recomputed from a copy of the shot in which every T_i value later than the row's limit is (a) removed and (b) raised by 1000 eV; all history inputs and the anchor frame are unchanged bit for bit.
- The folds of M2 keep every session whole; the fits of M3 and M4 contain no shot later than the first test shot.
- The Thomson value used at frame t comes from laser times inside the window of frame t only (the window of `window_mean`), reported with the share of rows that have one and two samples.

## 13. Files

- Code: `ces_prediction/experiments/b76_mast_availability/` (imports the loaders of `experiments/mast/` and `baselines_interpolation.py`; edits nothing).
- Caches: `data/.b76/`. Output: `docs/paper/b76_mast_availability.{md,json}`, checks `docs/paper/b76_checks.md`.
- Seed 20261176, one stream per quantity (crc32 of its name).
- Implementation choices that this text leaves open go into `PREREGISTRATION_B76_ADDENDUM1.md`, written before any readout. This file is not edited after the first run.
