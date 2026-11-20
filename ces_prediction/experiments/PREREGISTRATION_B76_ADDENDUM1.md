# B.76 — Addendum 1 (implementing agent): implementation choices

Written 2026-11-20 before any run of this batch's code and before any readout. No row of this batch had been built and no statement, MSE, skill or other score under any availability rule had been computed when this text was written. What had been looked at: the registration; the code of `experiments/mast/` (`run_gate.py`, `run_coupling.py`, `explore_fast.py`, `explore_ped.py`, `confirm_ped.py`), of `b75_availability/` and `b73_targets/`; `PREREGISTRATION_MAST_PED.md`, `docs/paper/mast_ped_confirmation.md`, `PREREGISTRATION_B75.md` with its addendum and `docs/paper/b75_availability.md`; the date fields of `data/mast/meta/l1_shots.parquet` for the listed shots (section 1); the array shapes of one shot file (17555). No target, estimate or Thomson value of this batch had been looked at.

Facts and choices only. No registered definition is changed. Code: `ces_prediction/experiments/b76_mast_availability/`; caches `data/.b76/`; outputs `docs/paper/b76_mast_availability.{md,json}`, `docs/paper/b76_checks.md`. A change made after this text was first written is added in a dated section at the end, with what had been seen at that time.

## 1. Shots and sessions

- **Shots.** Every shot of `mast/coupling_shots.txt` (3,686 listed, both halves) that passes, in this order: a charge-exchange file in `data/mast/act/`; the geometry of `explore_ped.build` (magnetic axis `radii_magnetic__radius_efit` and an outer radius `radii_router_radius_efit`, else `radii_router_radius_d_alpha`, both finite and outer > axis); the shot filters of `explore_fast.shot_rows` (median stored frame step within 0.1 ms of 5 ms, one major radius per chord); `explore_fast.prepare(d, "T_i", 0.5, 0.5)` with status "ok", no stored step inside the block off 5 ms by more than 0.1 ms, all stored block times finite; an extra file in `data/mast/extra/` from which `explore_ped.load_extra` returns Thomson profiles ("ts"). The filter of `shot_rows` that needs at least one hidden-frame row is not applied (no frame is hidden here). The count of shots lost at each step is reported.
- **Session** = the calendar day of the shot, from the FAIR-MAST metadata (it carries dates and times, so the shot-number rule is not used): `shot_experiment_date` of `l1_shots.parquet`. For the one listed shot without it (18524) the date of `timestamp` is used. (Nine listed shots have `timestamp` at 23:00 of the previous day and no experiment time; their experiment date is used, as for every other shot.)

## 2. Frames, chords, values

- **Frame grid.** As `explore_fast`: block frames j = 0 .. L − 1 of `prepare` (first to last frame with at least ten good chords, at least 20 frames), times t_j = t_a + 5 j ms with t_a the stored time of the first block frame (the nominal grid; the shot filter keeps stored steps within 0.1 ms of it). Frames outside the block are not used, neither as rows nor as history nor in L5.
- **Good** = `run_gate.good_masks(d, 0.5)["T_i"]`: T_i and its reported error finite, both above zero, T_i below 5,000 eV, reported error / T_i below 0.5. **Chords**: good in at least half of the block frames (`prepare` with frac 0.5).
- **Rows.** Every (chord, block frame) with a good value. Target F = that value. Target L5 = the median of the good values of the same chord at block frames j + 1 .. j + 5, defined when at least three are good (frames beyond the block end count as not good).
- **Normalised radius** of a chord: (R − R_axis) / (R_outer − R_axis) with the geometry above (as `explore_fast.shot_rows`).

## 3. Availability rules

- A good value of chord c at frame j' is visible for row (c, j) under rule Dd when t_j' ≤ t_j − d + 0.5 ms (tolerance 0.5 ms; on the 5 ms nominal grid every tolerance below 5 ms gives the same sets). D5 thus shows frames up to j − 1, D15 up to j − 3, D25 j − 5, D50 j − 10, D100 j − 20, D150 j − 30.
- The **anchor frame** is the frame of the newest visible good value. The input code reads the value array of the chord only at frames up to the visibility limit (asserted per row).

## 4. Inputs

- **History scalars** (T0, in this order): age of the newest visible value (ms); H; G − H; |G − H|; of the last up to ten visible good values of the chord (in time order) the mean, the standard deviation (ddof 0), H − mean, the last increment (difference of the last two of them, whatever the time between) and their number, all five undefined with fewer than three; the normalised radius.
- **M5** = median of the last up to five visible good values. **G** = `baselines_interpolation.predict_gp_causal(times in s, values, t_j in s)` on the last up to 16 visible good values in time order (with one visible value G is that value).
- **Thomson** (`explore_ped.load_extra`, the core system `ayc`, else `atm`; `to_chords` and `window_mean` unchanged): T_e and n_e (in 1e19 m^-3) at the chord radius, mean over the laser times in [t − 2.5, t + 2.5) ms of the frame. Present = window of frame j; anchor = window of the anchor frame.
- **Thomson scalars** (TX adds them, in this order): T_e(t), n_e(t), T_e(anchor), n_e(anchor), T_e(t) − T_e(anchor), n_e(t) − n_e(anchor), ln(T_e(t) / T_e(anchor)), ln(n_e(t) / n_e(anchor)), and the distance of the chord to the density edge at frame t (below).
- **Density edge, two uses.**
  - *Pedestal-top band (row selection, as MAST_PED):* `explore_ped.add_extra` unchanged in substance: the edge of each profile is `explore_ped.edge_radius` with the magnetic axis; the edge of a frame is the median of the profile edges with laser time in [t − 7.5, t + 7.5] ms; distance = frame edge − R_chord; band 1.5 cm ≤ distance < 5 cm. This reads profiles up to 7.5 ms after the frame time; it fixes the population, the same under every rule and estimator, and is not an input.
  - *Input of TX:* the mean of the profile edges over the laser times in the window of frame t ([t − 2.5, t + 2.5) ms, `window_mean`) minus R_chord, so that no input reads Thomson beyond the window of frame t (registration section 4).
- **D-alpha** (TD, TA; `run_coupling.load_drivers`): frame mean `run_coupling.frame_means` and frame maximum `explore_fast.frame_max` over [t − 2.5, t + 2.5) ms for the block frames, divided by the shot's scale, the median of |frame mean| over the block frames (as `explore_fast`; a shot-level constant that uses the whole block, as in the earlier batches). Six scalars in this order: mean(t), max(t), mean(anchor), max(anchor), mean(t) − mean(anchor), max(t) − max(anchor).
- **Learner inputs.** T0: the ten history scalars. TX: those and the nine Thomson scalars. TD: the ten and the six D-alpha scalars. TA: the ten, the nine, the six.
- **RT** = H · T_e(t) / T_e(anchor). **RTa**: α, β by ordinary least squares (numpy `lstsq`) of ln(target) − ln(H) on [ln(T_e(t) / T_e(anchor)), 1] over the fit rows; estimate H · exp(α ln(T_e(t) / T_e(anchor)) + β).

## 5. Row sets

- **Candidate rows**: rows in the band at frame t with T_e(t), n_e(t) and the input edge distance finite. Only candidate rows get inputs under the rules (rows outside cannot be complete); counts at each condition are reported (all rows; band; band with Thomson at t; candidates).
- **Complete under a rule**: the ten history scalars and G finite, the nine Thomson scalars finite (both T_e and n_e at the anchor finite and the ratios positive; rows with finite but non-positive ratios are counted).
- **Common set (per target)**: candidate rows complete under each of the six rules, with the target defined. Since F exists on every row, the F set contains the L5 set.
- **DA set (per target)**: common-set rows whose six D-alpha scalars are finite under D5 and under D100 (the same rows for both rules). TD, TA and the refitted T0 and TX ("T0d", "TXd") use it.

## 6. Designs

- **Shots of the designs**: the shots with at least one row in the F common set, in increasing shot number. One set of units for both targets.
- **M1**: fold = `numpy.random.default_rng(seed).permutation(n_shots) % 5` over the shots in increasing order; seeds 1, 2, 3; units M1_s{seed}_k{fold}.
- **M2**: sessions in increasing date, taken in the order `default_rng(seed).permutation(n_sessions)`; each goes to the fold with the fewest shots so far (ties: lowest fold index); seeds 1, 2, 3; units M2_s{seed}_k{fold}.
- **M3**: shots in increasing number cut into fifths by `numpy.array_split`; units M3_q{q}, q = 1 .. 4 (fifths 2 .. 5): fit on fifths < q, score fifth q.
- **M4**: fit on the first four fifths, score the last. Its fit and test shots are those of M3_q4; it is fitted again as its own unit (identical numbers expected).
- **Fits.** `HistGradientBoostingRegressor(random_state=20261116)`, default settings, fitted to (target − G under the rule) on the fit shots' rows of the set; estimate = G + prediction on the test shots' rows. T0, TX, RTa: every unit, rule and target on the common set. TD, TA, T0d, TXd: units of M1 and M4, rules D5 and D100, both targets, DA set.
- **Controls** (M1 units, D100, L5, common set): *placebo* "TXp": TX with the nine Thomson scalars permuted over rows as one block (one row permutation applied to all nine columns), separately within the fit rows and within the test rows (streams `[20261176, crc32("placebo|{unit}|fit")]` and `"|test"`); *Thomson as old as the history* "TXo": the ten history scalars and T_e(anchor), n_e(anchor) only; *T0 with the D5 history*: T0 under D5 on the same rows (the common set makes the rows identical).

## 7. Readouts and intervals

- **Skill** of A over B = 1 − SSE(A) / SSE(B) on identical rows. Increment = TX over T0, same rule, same rows. Cost of staleness = SSE(rule) / SSE(D5) (T0, TX). Recovered share = (SSE(T0) − SSE(TX)) / (SSE(T0) − SSE(T0 under D5)) under the same rule and rows; not formed under D5.
- **Points.** M1: mean over the three re-partitions of the per-split value; M2: over the three assignments; M3: pooled over the scored fifths; M4: the test block. RMSE: mean over splits of the per-split RMSE.
- **Intervals.** As B.75: 20,000 draws, 2.5 and 97.5 % quantiles; units (sessions, and shots for the form beside it) with at least one row in the cell resampled with replacement in one level (multinomial weights) and their per-unit SSE sums re-added; every quantity of a cell is formed draw by draw on the same draws. M1: a re-partition drawn first, M2: an assignment drawn first (uniform over the three); M3: the units of the scored rows; M4: the units of the test block. One stream per cell, `numpy.random.default_rng([20261176, zlib.crc32(name.encode())])` with name "design|target|set|stratum|form", e.g. "M1|L5|common|all|session". The multinomial weights are drawn in chunks of 1,000 draws per sub (deterministic).
- **Strata.** M1: all rows, and by fifth of the shot order of the test shot (the M3 fifths). Added beside the registered readouts (no statement uses it): M3 per scored fifth.
- **α, β** per design, rule and target: mean, smallest and largest over the units.
- **Counts.** Shots lost per filter; rows at each candidate condition; rows and shots complete under each rule; rows with a non-positive Thomson ratio; common and DA sets per target; scored rows / shots / sessions per design and target; sessions that straddle a fifth boundary.

## 8. Statements: how each is judged (session form)

| Statement | Quantity | Holds if |
|---|---|---|
| S1 | SSE(T0 under D100) / SSE(T0 under D5), M1, L5, common set | lower end above 1.5 |
| S2, S4, S5, S6 | TX over T0, L5, D100, common set, under M1, M2, M3 (fifths 2-5 pooled), M4 | lower end above zero |
| S3 | (TX over T0 under D100) − (TX over T0 under D5), M1, L5, common set | lower end above zero |
| S7 | RTa over H, M1, L5, D100, common set | lower end above zero |
| S8 | TD over T0d, M1, L5, D100, DA set | lower end above zero |
| S9 | TXp over T0, M1, L5, D100, common set | lower end at or below zero |
| S10 | 1 − SSE(TX) / SSE(O), M1, L5, D5, common set | lower end above zero |

S1, S7 and S8 name no target; they use L5, the target of the statements' definition of "increment" (registration section 9) and of S10.

## 9. Checks (registration section 12)

| Check | What is compared | Pass rule |
|---|---|---|
| 1 | H under D5 against `y1` of `data/mast/.ped_rows_T_i.npz` and `.ped_conf_rows_T_i.npz` on the rows both builds contain (matched by shot, block frame and chord radius in float32) whose frame j − 1 is good | float32(H) equals `y1` on every matched row; largest deviation in eV reported. Beside it (diagnostics, no pass rule): T_e(t), n_e(t) against `te0`, `ne0`; the band distance against `d_edge`; the normalised radius; the frame time |
| 2 | 200 rows of the L5 common set, test rows of unit M1_s1_k0, stream `crc32("visibility|D100")`. For each, a copy of the shot in which every T_i value at a frame with nominal time above the row's limit + 0.5 ms (all chords) is (a) removed (NaN), (b) raised by 1000 eV; the good mask recomputed; the block start recomputed and required equal; the anchor frame, the number of visible values, H, age, M5, G, the five scalars, T_e and n_e at the anchor and the D-alpha anchor scalars recomputed | every one identical bit for bit |
| 3 | Units: M1 and M2 put every shot in exactly one test fold per re-partition or assignment; M2 keeps every session in one fold; M3 and M4 fit only shots with a lower number than the first test shot | all true |
| 4 | 2,000 rows of the F common set (stream `crc32("thomson|window")`): T_e(t) and n_e(t) recomputed from the extra file with only the laser times in [t − 2.5, t + 2.5) ms; shares of rows with one, two and three or more laser samples (finite at the chord) in the window, on the F and L5 common sets | recomputed values identical bit for bit |

The chord selection (good in half of the block frames) uses the whole block; check 2 keeps the row's chord fixed and tests the inputs, not the selection. No check's tolerance is changed after a run.

## 10. Notes after the row build and the checks, before any readout (2026-11-20)

Written after the row build, the row sets and the four checks, while the fits were running; no fit output, MSE, skill or statement had been read. Nothing above was changed.

- **Changes to the check code before its first run.** Check 2 forms the recomputed block start directly as the first frame with at least ten good chords instead of calling `run_coupling.block_of`, which returns nothing when the block of the modified copy is shorter than 20 frames (a shot-level filter of the original data, not a property of the row's history). Check 1 leaves out rows whose key (shot, block frame, chord radius in float32) occurs twice in either build (two chords at one radius); this build has no such row (0), the earlier tables' count is not recorded.
- **Counts.** Of 3,686 listed shots, 3,232 pass every filter (277 without a block, 14 without a chord good in half of the block frames, 89 with a stored step off 5 ms inside the block, 74 without Thomson profiles; every listed shot has an extra file). Good chord frames 10,021,523; in the pedestal-top band 470,048; with Thomson at frame t 372,117; candidates 372,094 rows of 3,132 shots. Complete under all six rules 96,449 rows of 1,967 shots (common F); common L5 81,360 rows of 1,785 shots; DA L5 76,049 of 1,630. No row has a non-positive Thomson ratio. Design shots 1,967 in 231 sessions.
- **Checks.** Each run once; all four pass (`docs/paper/b76_checks.md`). Beside check 1, T_e(t) differs from `te0` of the earlier tables on 96 of 161,461 compared rows and the band distance on 86; a diagnostic (added after check 1 ran, in `stage_checks.py diag`) shows that on every one of them a laser time lies on the other side of a window edge when the window is formed from the earlier build's float32 frame time instead of the float64 time used here. The float64 time is kept.
- **Workers.** The fits run in three processes of two threads each.

## 11. After the readout (2026-11-20): what happened, nothing changed

- All 1,546 fit outputs (1,126 boosted fits and 420 two-coefficient fits) were computed once; no fit was repeated or changed. Before its first run, the readout was given the diagnostic of section 10 to carry in its JSON (`checks.hold_diag`). `readout.py` was then run once end to end (75 s); its JSON and markdown are the files in `docs/paper/`. No definition, row, fit or statement rule was changed after any number had been read.
- One descriptive count was computed after the readout and is not in the JSON (no statement uses it): on the L5 common set the ratio T_e(t) / T_e(anchor) has median 1.003 under D5 and 1.048 under D100, 99.9th percentile 8.0 and 19.2, largest 74.2 and 57.6; it lies outside [0.5, 2] on 3.7 % of the rows under D5 and 24.0 % under D100. T_e(t) at the chord radius: 1st percentile 50.8 eV, median 201 eV, 99th percentile 956 eV.
