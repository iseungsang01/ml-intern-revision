# B.86 — Addendum 1 (implementing agent): implementation choices

Written 2026-11-21 before the row build of this batch, before any check and before any readout. What had been looked at when this was written: the registration; `PREREGISTRATION_B76.md` and its addendum; the code of `b76_mast_availability/`, `b83_interp/` (`stage_refs83.py`, `common83.py`), `b82_localize/boot82.py` and `baselines_interpolation.py`; the layouts of `docs/paper/b76_mast_availability.{json,md}` and `docs/paper/b82_localize.json`; `data/.b76/sets.json` (counts of B.76); a timing run of the row builder below on four shots (17719 to 17722; row counts only). No row table, error, skill, MSE or interval of this batch existed.

Facts and choices only. No registered definition is changed. Code: `ces_prediction/experiments/b86_mast_radius/`; caches `data/.b86/`; outputs `docs/paper/b86_mast_radius.{md,json}`, `docs/paper/b86_checks.md`. A change made after this text was first written is added in a dated section at the end, with what had been seen at that time.

## 1. Shots, frames, values, sessions

Exactly B.76 (its Addendum 1 sections 1 and 2): the shot filters of `common76.prepare_shot`, the block frames, the good-value rule (population W, relative error below 0.5), chords good in at least half of the block frames, the normalised radius rho = (R - R_axis) / (R_outer - R_axis), sessions = calendar day of `shot_experiment_date` (else of `timestamp`).

## 2. Rows and bands

- **Builder.** `common76.build_shot` has the band hard-coded (`BAND`) and builds six rules and the D-alpha inputs. `common86.build_shot86` re-implements only that function: the band condition is the union [1.5, 60) cm of the five bands, the rules are D5 and D100, D-alpha is not loaded (no estimator of this batch uses it), and the chord series of the block (values and good mask) is stored for the references. Every other expression is that of `build_shot`; `prepare_shot`, `chord_history`, `thomson_grids`, `hist_matrix`, `ts_matrix` are imported from `common76` unchanged.
- **Band** of a row = the band of its distance d (B.76's `dband`: median of the profile edges within ±7.5 ms minus R_chord) in cm: E [1.5, 5), B2 [5, 10), B3 [10, 20), B4 [20, 35), B5 [35, 60). A candidate row (B.76 Addendum 1 section 5: good value, in a band, T_e(t), n_e(t) and the input edge distance finite) belongs to exactly one band. Chords with d of 60 cm or more are not built. No condition on rho is added (a chord in B5 can lie inboard of the magnetic axis; the share of rows with rho below zero is reported per band).
- **Stored tables.** Band E keeps every candidate row (the check of section 6 compares them with B.76's `rows.npz`); B2 to B5 keep only the candidate rows complete under D5 and under D100, the only rows any fit or score uses. Candidate counts are reported for every band.
- **Complete under a rule** as B.76 (the ten history scalars and G finite, the nine Thomson scalars finite). **Common set per band and target X**: complete under D5 and under D100, target X defined. The F set contains the L5 set.
- **Statements rule (registration section 2).** A band whose L5 common set has fewer than 5,000 rows is reported and left out of the statements. "The innermost band with statements" (R5 to R7) is the innermost of B5, B4, B3, B2 that has statements.
- **Counts per band** (on the L5 common set): rows, shots, sessions, median rho, median reported T_i (the value of frame t), median d; beside them the candidate counts.

## 3. Targets, rules, inputs

L5 and F, D5 and D100, the history scalars and the nine Thomson scalars, H, G and O exactly as B.76 Addendum 1 sections 2 to 4. O = the value of frame t, scored against L5 only.

## 4. Designs (per band)

- **Design shots of a band**: the shots with at least one row in the band's F common set, in increasing shot number. One set of units for both targets.
- **M1** = re-partition 1 of B.76's procedure applied to the band's design shots: fold = `numpy.random.default_rng(1).permutation(n_shots) % 5`; units M1_s1_k0 .. k4. The procedure, not B.76's fold list, is reused, because the shots differ between bands (and band E's set differs from B.76's, which required all six rules).
- **M3**: the band's design shots cut into fifths by `numpy.array_split`; units M3_q1 .. q4: fit on the earlier fifths, score fifth q. Scored rows pooled.

## 5. Fits

- `HistGradientBoostingRegressor(random_state=20261116)`, default settings, fitted to (target - G under the rule) on the fit shots' rows of the band's common set of the target; estimate = G + prediction on the test shots' rows. T0 on the ten history scalars, TX on those and the nine Thomson scalars.
- Plan per band: T0 and TX, rules D5 and D100, target L5, every unit of M1 and M3 (36 fits per band); T0 and TX, rule D5, target F, the units of M1 (10 fits per band; the value-of-the-frame table, registration section 5, is under D5 and M1 only). No F fit under D100 or M3 (no readout asks for one).
- Two processes of two threads (4 threads in total); the row build and the references use four processes of one thread.

## 6. References that wait (F, D5, M1)

- For every row of the F common set of a band: the neighbour set is the good values of the row's chord at the block frames other than frame t (frame t left out; frames outside the block are not used, as for every other input), times in s from the block start (as `chord_history` passes them to the causal GP); `baselines_interpolation.predict_pchip` and `predict_gp` evaluated at the time of frame t.
- As B.83 (its Addendum 1 section 2): a row without a good value before frame t or without one after it in the block is a row where the references cannot be formed; it is left out of every comparison with a reference (counts reported). Rows where the offline GP's fit fails on every grid point and `predict_gp` falls back to hold are counted and kept.
- **Value-of-the-frame rows** of a band: the F common set, scored under M1 (every design shot is tested once), with both references formed. H, G, T0, TX (fitted to F under D5), PCHIP and the offline GP are scored on these rows.

## 7. Measures and intervals

- Error e = estimate - target. RMSE = sqrt(SSE / n); MAE = mean |e|; median absolute error; median relative error = median of |e| / target; share within 10 % (20 %) of the level = share of rows with |e| <= 0.10 (0.20) x target (target L5; for F the same with F). Skill of A over R = 1 - SSE(A) / SSE(R) on identical rows. Cost of staleness of T0 (TX) = SSE(T0 under D100) / SSE(T0 under D5) on the same rows (the common set makes them identical). Recovered share at D100 = (SSE(T0, D100) - SSE(TX, D100)) / (SSE(T0, D100) - SSE(T0, D5)).
- **Single sub.** M1 is one re-partition and M3 one pooled set of fifths, so every point is the statistic of the totals and there is no re-partition draw.
- **Draws.** 95 %, 20,000 draws, 2.5 and 97.5 % quantiles. The units (sessions; shots for the form beside it) with at least one row in the cell are resampled with replacement in one level as multinomial counts (chunks of 1,000 draws), and per-unit sums of every quantity are re-added; every quantity of a cell is formed draw by draw on the same draws.
- **Streams.** `numpy.random.default_rng([20261186, zlib.crc32(name.encode())])`. For a statement the name is its id (R1.E, ..., R4.B5, R5, R6, R7); within the stream the session draws are taken first, then the shot draws. For a reported table cell the name is "band|design|target|rule|cell" (for example "B3|M1|L5|D100|skills"), session draws first, then shot draws.
- **Across bands (R5 to R7).** The units are the union of the units (sessions, or shots) with a row in either band's cell; each band's per-unit sums are placed on the union index (zero where a unit has no row in that band), so both bands are formed on the same draws.
- **Intervals given** for every statement and its components, for every skill and difference in the tables, for the cost of staleness, the recovered share and the shares within 10 % and 20 % (ratios of per-unit sums). Medians (absolute and relative) are given as points only: no statement or difference uses them.

## 8. Statements: how each is judged (session form)

| Statement | Quantity (target L5) | Holds if |
|---|---|---|
| R1.b | skill of TX over T0, M1, D100, band b | lower end above zero |
| R2.b | skill of TX over T0, M1, D5, band b | lower end above zero |
| R3.b | skill of TX over T0, M3 (fifths 2 to 5 pooled), D100, band b | lower end above zero |
| R4.b | skill of T0 over G, M1, D5, band b | lower end above zero |
| R5 | cost of staleness of T0 (M1) in E minus that in the innermost band with statements | lower end above zero |
| R6 | skill of TX over T0 (M1, D100) in E minus that in the innermost band with statements | lower end above zero |
| R7 | share of TX (M1, D5) within 10 % of the level in the innermost band with statements minus that in E | lower end above zero |

If band E has fewer than 5,000 rows, R5 to R7 cannot be formed and are reported as not formed (registration section 2 leaves the band out of the statements).

## 9. Checks (registration section 6)

Each run once, before any readout. No error, skill or interval is computed by a check.

| Check | What is compared | Pass rule |
|---|---|---|
| Band E | (a) the band-E rows of this build against `data/.b76/rows.npz`: same number of rows in the same order and, bit for bit, every field this batch uses: shot, c, craw, R, rho, j, t, y, sig, prevgood, te_t, ne_t, dedge_t, dband, nts, nlas, L5, nL5, and under D5 and D100: H, age, M5, G, th, anc, nvis, te_a, ne_a, nts_a. (b) T0 and TX of unit M1_s1_k0 under D100, target L5, refitted by this batch's fit code on this batch's band-E rows with B.76's L5 common set (`data/.b76/sets.npz`) and B.76's unit, against `data/.b76/fits/M1_s1_k0__{T0,TX}__D100__L5.npz` (row indices and estimates) | all identical bit for bit |
| Visibility | 200 rows of band B4's L5 common set among the test rows of unit M1_s1_k0 of B4 (stream `crc32("visibility|B4|D100")`). For each, a copy of the shot in which every T_i value at a frame with nominal time above the row's limit + 0.5 ms (all chords) is raised by 1000 eV; good mask recomputed; block start recomputed and required equal; the anchor frame, number of visible values, H, age, M5, G, the five scalars, and the nine Thomson scalars (T_e and n_e at t and at the anchor, their differences and log ratios, the input edge distance) recomputed from the copy | identical bit for bit |
| References | 200 rows drawn from the value-of-the-frame rows of all bands pooled (stream `crc32("references")`). For each, a copy of the shot in which the T_i value of the row's chord at frame t is raised by 1000 eV; good mask recomputed; PCHIP and offline GP recomputed from the copy (frame t left out) | identical bit for bit to the stored references |
| Rows | For every band, design, rule and target: every estimator of a comparison (H, G, T0, TX, O; for the value of the frame also PCHIP and the offline GP) has a finite value on every scored row; the units of each design put every design shot in exactly one test set (M1) or test only shots after every fit shot (M3); counts per band, rule and design printed | all true |

A failed check stops the batch and is reported to the session lead.

## 10. Output

`docs/paper/b86_mast_radius.json`: `statements` (id, text, band, value, session interval, shot interval, verdict, expectation), `bands` (band -> design -> rule -> estimator -> measures, with the skills "A over R" and their intervals, the cost of staleness and the recovered share), `value_of_frame` (per band), `counts` (per band), the checks. `docs/paper/b86_mast_radius.md` renders it; `docs/paper/b86_checks.md` the checks.

## 11. Notes after the row build, the fits and the checks, before any readout (2026-11-21)

Written after the four checks had passed and before the readout was run; no error, skill, MSE, share or interval of this batch had been computed or looked at. Nothing above was changed.

- **Counts.** Candidate rows (band E equals B.76's 372,094): E 372,094; B2 548,129; B3 1,003,401; B4 1,312,041; B5 1,832,147. Rows complete under D5 and D100 (= F common set): E 165,331; B2 266,107; B3 529,197; B4 730,888; B5 1,153,562. L5 common set: E 145,048 rows / 2,517 shots / 252 sessions; B2 239,297 / 2,633 / 254; B3 485,836 / 2,803 / 258; B4 678,409 / 2,697 / 255; B5 1,074,935 / 2,789 / 259. Every band has at least 5,000 rows, so every band has statements and the innermost band with statements is B5.
- **References.** Rows of the F common set without a good value on one side of frame t (left out of the comparisons with the references): E 4,102, B2 5,540, B3 9,006, B4 11,111, B5 15,690. The offline GP fell back to hold on no row.
- **Fits.** 230 boosted fits (46 per band), each once, two processes of two threads; references in four processes of one thread.
- **Reference check, first start.** The first start of the reference check was stopped by the agent after about 20 minutes, before it had compared any row: the code built its pool of candidate rows by reading an array from the compressed npz file once per row (3.2 million reads). The pool is now built from arrays loaded once; the draw (stream `crc32("references")` over the same pool in the same order), the modification and the comparison are unchanged. The check was then run once to the end and passed.
- **Stream names of the table cells.** A table cell holds both rules (the cost of staleness and the recovered share need both), so its name is "band|design|L5|D5+D100|table" (for example "B3|M1|L5|D5+D100|table"); the value-of-the-frame cell is "band|M1|F|D5|value_of_frame". The statements use their ids as stated in section 7.
