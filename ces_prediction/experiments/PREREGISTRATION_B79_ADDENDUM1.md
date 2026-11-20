# B.79 — Addendum 1 (implementing agent): implementation choices

Written 2026-11-20, before any fit of this batch and before any readout. When this text was written, no statement, MSE, RMSE, skill or other score had been computed for any estimator with a beam input, nor for T0 or TX in this batch.

What had been looked at:
- the registration;
- the B.73 and B.75 code and caches;
- `docs/paper/nbi_files.md` and `ces_prediction/analyze_nbi_files.py`;
- these facts of the inputs (no target, no estimate): the header and first rows of one NBI file, the distribution of each source power on the grid rows, empty values, the NB1B voltage while the source is on and off, the join of the beam rows to the grid rows, the number of rows whose beam scalars are defined, and the row sets;
- the three checks of registration section 9 (join, reproduction of B.75's estimates, causality of the beam scalars), which were run before this text was finished. All three pass (section 8).

The facts are given in section 1. Only facts and choices are recorded here; no registered definition is changed. Code: `ces_prediction/experiments/b79_beam_inputs/`; caches `data/.b79/`; outputs `docs/paper/b79_beam_inputs.{md,json}`, `docs/paper/b79_checks.md`. A change made after this text was first written is added in a dated section at the end, together with what had been seen at that time.

## 1. Facts looked at before any fit

- **Join.** Each of the 641 shot files of `data/.b73/grid.npz` has its NBI file, with the same number of rows. Over all 247,207 rows (245,925 without each file's first and last row) the largest difference between the beam stamp and the grid stamp is 9.5e-7 s, and the beam stamp rounded to float32 equals the grid stamp on every row. Mismatches at 1e-4 s: 0.
- **NBI-1 columns.** NB1B power: never empty, largest 1.65 MW, smallest -0.001 MW. NB1C power: empty on 295 grid rows (one shot), largest 1.31 MW. The NB1B voltage is finite on every grid row, both while NB1B is on and while it is off.
- **NBI-2 columns in thirteen files.** Every source power above 3 MW lies in the NBI-2 columns of thirteen files: shots 30801, 30805, 30809, 30810, 30811, 30812, 30842, 30843, 30845, 30895, 30906, 30907, 30909, all D1 discharges.
  - Largest values: NB2A 143.8, NB2B 75.3, NB2C 50.5 "MW".
  - These values sit on 3,153 grid rows and 2,652 D1 rows. On those rows the voltage times the current of the same source, divided by 1000, has a median of 0.4 to 3.3 per file and source. The power column of these files is therefore on another scale, not the source.
  - While a source is off, the same files read NBI-2 powers as low as -3.5 MW (18 file-source pairs below -0.1 MW).
  - Outside these files no power exceeds 3 MW and no NBI-2 power lies below -0.1 MW.
  - The thirteen files hold 2,913 D1 rows, of them 1,783 in the common set for L5 and 1,640 in the S set for L5.
- **Row sets.** The nineteen beam scalars are defined on every row of B.75's common set and S set (both targets), so the row sets of this batch equal B.75's (section 5).

## 2. Beam columns

- **Columns.** Read from `data/nbi/s{shot}_NBI.csv`: `time`, `NB1B_P`, `NB1C_P`, `NB2A_P`, `NB2B_P`, `NB2C_P`, `NB1B_V`. NB1A is not used: it is never on, and the registration's P1 is NB1B + NB1C.
- **Join.** The rows of an NBI file are taken in file order onto the grid rows of its shot file, after checking that the row count is the same and that every stamp agrees within 1e-4 s (check 1).
- **Empty values.** An empty power is zero.
- **Values as delivered** (section 7).

## 3. The nineteen beam scalars

- **Powers.** P1 = P(NB1B) + P(NB1C). P2 = P(NB2A) + P(NB2B) + P(NB2C), added in that order. Pt = P1 + P2.
- **Anchor row.** The global grid row `anc_g` of the rule's inputs (`data/.b75/inputs/{rule}.npz`): the row of the newest visible CES value under D10, D650 or S. It always lies before row t.
- **Order of the scalars.**
  1. P1(t), P1(anchor), P1(t) - P1(anchor);
  2. the same three for P2, then for Pt;
  3. mean10 Pt(t), mean50 Pt(t), mean10 Pt(anchor), mean50 Pt(anchor);
  4. the on flags of NB1B, NB1C, NB2A, NB2B, NB2C at row t;
  5. the NB1B voltage at row t.
- **Running means.** meanK(i) is the mean of Pt over rows i - k, k = 0 .. K - 1, that lie in the block of i and are grid rows.
  - Blocks are those of `grid.npz`: a cut at a step of 0.5 s or more.
  - A file's first row (t = 0) and last row are not grid rows and never enter. In eight files the first row would otherwise join the first block.
  - The rows are added in the order k = 0, 1, ... and the sum is divided by their number.
  - A mean is undefined with fewer than five rows. At a block start this means the first four rows of a block have no mean. From the fifth row on, the mean covers all rows of the block so far, up to ten or fifty. A mean never reaches past its row: it ends at row t, or at the anchor row.
- **On flags.** 1 when the power is above 0.1 MW, otherwise 0; an empty power reads as off.
- **NB1B voltage.** The `NB1B_V` value where P(NB1B) is above 0.1 MW, otherwise 0.
- **Visibility.** Every beam value of row t comes from rows up to and including t: row t itself, the anchor row, and the running means ending at those rows (check 3).

## 4. Learners

- **Inputs.** The ten history scalars, the eight fast scalars of row t and the six anchor scalars are built by B.75's code, unchanged (`b75_availability/stage_fits.Ctx.unit_arrays` and `.matrices`). The beam scalars come from `data/.b79/beam_{rule}.npz`.
- **Column orders.**

| Learner | Columns |
|---|---|
| T0 | the ten history scalars (B.75's order) |
| TN | T0's columns, then the nineteen beam scalars |
| TX | T0's columns, the eight fast scalars, the six anchor scalars (B.75's TX) |
| TXN | TX's columns, then the nineteen beam scalars |
| TEN | TN's columns, then the ECEI half: the four ECEI fast scalars of row t, then a1, a2, a3 of ECEI (B.75's TE half) |
| TBN | TN's columns, then the BES half: the four BES fast scalars of row t, then a1, a2, a3 of BES (B.75's TB half) |

- **Fits.** `HistGradientBoostingRegressor(random_state=20261116)`, default settings, fitted to (target - G under the rule). The estimate is G plus the prediction. Every input is finite on every fit and test row; this is asserted before each fit.
- **Which fits.**
  - T0, TN, TX, TXN: all 36 units of B.73 (V1 15, V2 15, V3 4, V4a, V4b); rules D10, D650, S; targets F and L5.
  - TEN, TBN: the V1 units, V4a and V4b; rules D650 and S; targets F and L5.
- **Controls** (V1 units, D650, L5):
  - **TNp** is TN with the nineteen beam scalars permuted jointly: one permutation of the rows is applied to all nineteen columns, separately within the fit rows and within the test rows. Stream `[20261179, crc32("placebo TNp|<unit>")]`; the fit permutation is drawn first.
  - **TXNp** is TXN with the fourteen fast scalars (eight of row t and six anchor scalars) permuted jointly in the same way, with the history and beam scalars unchanged. Stream `crc32("placebo TXNp|<unit>")`.

## 5. Rows of each fit and score

- **Common set** (per target, rules D10 and D650): B.75's common set restricted to the rows whose nineteen beam scalars are defined under D10 and under D650. D10 and D650 are thus fitted and scored on identical rows, as in B.75.
- **S set** (per target, rule S): B.75's S set restricted to the rows whose beam scalars are defined under S.
- **Counts.** No row is dropped. Common set: F 165,418, L5 160,343. S set: F 152,203, L5 147,837.
- **Consequence.** These are B.75's sets, so T0 and TX of this batch are B.75's fits refitted on the same rows. Check 2 shows them bit-identical for V1_r2_k0 and V4a under D650. After the fits every T0 and TX file is compared with B.75's (section 8).

## 6. Readouts and intervals

- **Skill.** The skill of A over B is 1 - SSE(A) / SSE(B) on identical rows.
- **Points.** As B.75:
  - V1 and V2: the mean of the three splits' values;
  - V3: pooled over fifths 2 to 5;
  - V4a and V4b: the test block.
  - RMSE: the mean over splits of the per-split RMSE.
  - Ratios: the mean over splits of the per-split value.
- **The ratio of the two MSE reductions (registered form).** (skill of TXN over TN) / (skill of TX over T0). Both skills are formed from the same draw's SSEs, and the ratio is taken draw by draw. Beside it, and not used by any statement, the ratio of the absolute reductions: (SSE(TN) - SSE(TXN)) / (SSE(T0) - SSE(TX)). Quantiles are over the draws with a finite value; the number of draws left out is reported.
- **Intervals.** B.75's readout code (`b75_availability/readout.py`, class `Group`), imported, with the seed of this batch.
  - The module's `rng_for` is replaced by this batch's: `[20261179, crc32(name)]`.
  - One stream per cell: `"design|target|set|stratum|form"`, for example `"V1|L5|common|all|cluster"`.
  - 20,000 draws; 2.5 and 97.5 % quantiles.
  - V1: a re-partition is drawn first. V2: an assignment is drawn first. V3: the units of fifths 2 to 5 pooled. V4a and V4b: the units of the test block.
  - Units are discharges (discharge form) or shot clusters (cluster form) with at least one row in the cell.
  - Every quantity of a cell (skills, differences, ratios) is formed draw by draw on the same draws.
- **Cells.** "common" holds D10, D650 and the two controls; "S" holds S.
- **V1 by chronological block** of the test discharge (up to 31991; 32002 to 32310; from 32312): every increment (TN over T0, TXN over TN, TX over T0, TXN over TX, and TEN and TBN over TN where fitted) under D10, D650 and S.
- **By time since the last notch** (V1; D650 on the common set, S on the S set; TXN over TN and TX over T0, with the other increments beside them).
  - An absent step is B.75's: a step between consecutive grid rows of the same block above 15 ms and below 45 ms.
  - j + 1 is the first row after the most recent absent step that ends at or before t in the block of t (B.75's limit row j of rule S, `jS_g` in `data/.b75/inputs/S.npz`). The time since the last notch is s(t) - s(j + 1).
  - Strata:
    1. "within 100 ms after an absent step": the time since is below 0.095 s. These are the first row after the step and the following nine 10 ms rows; the 5 ms margin keeps the float32 stamp of the row at 100 ms out of the stratum.
    2. "later rows": the time since is 0.095 s or more.
    3. "no absent step earlier in the block": the registration's "rows of blocks without one", read on the input. Whether an absent step comes later in the block is not known at row t, so it does not define a stratum. The number of rows of this stratum whose block has an absent step later is reported.
  - Under S the third stratum is empty (rule S needs an absent step earlier) and is not formed. A stratum with no row is not formed.
- **Statements** (target L5; judged on the cluster interval; "V4" is V4a, with V4b in the note):

| Statement | Quantity | Holds if |
|---|---|---|
| N1 | TN over T0, V1, D650 | lower end above zero |
| N2, N3, N4, N5 | TXN over TN, D650, under V1, V2, V3 (fifths 2-5 pooled), V4a | lower end above zero |
| N6, N7 | TXN over TN, rule S, under V1, V4a | lower end above zero |
| N8 | TN over T0, V4a, S | lower end above zero |
| N9 | TXN over TX, V1, D650 | lower end above zero |
| N10 | TXN over TN, V1, D10 | lower end above zero |
| N11 | TNp over T0, V1, D650 | lower end at or below zero |

- **Readouts beside the registered ones** (no statement uses them):
  - TXN over T0;
  - TXNp over TN, which should be near zero when the fast scalars carry nothing beyond history and beam;
  - TN over TNp;
  - RMSE of TNp and TXNp;
  - the absolute form of the ratio.
- **O** is the reported value of row t, scored against L5 only.

## 7. The thirteen files with NBI-2 power columns on another scale

- **Choice.** The registration defines the beam scalars from the power columns, with an empty power as zero, and registers no other treatment. The columns are therefore used as delivered.
  - The on flags of those files are right: real powers are above 0.1 MW, and the negative offsets read as off.
  - P2, Pt and the running means of Pt are not on the scale of the other files there.
- **Sensitivity beside the registered readouts.** Not used for any verdict. The quantities of N1 to N11 are recomputed on the scored rows outside the thirteen files, with the fits unchanged: the thirteen files remain in the fit rows of every unit that fits on them.
  - Under V3 and V4 these files are never scored (first fifth; fit block), so the sensitivity equals the main readout there.
  - Under V1 and V2 it is a stratum of the scored rows, with its own intervals (stratum name "outside the 13 files").
- **Session lead.** The lead was told of these files before any fit, and of this default. A different decision received before any readout is added below in a dated section.

## 8. Checks (registration section 9)

| Check | What is compared | Pass rule |
|---|---|---|
| 1 | Every grid row of the 641 shot files against its beam row: same row count per file, stamp within 1e-4 s | mismatches 0 |
| 2 | T0 and TX under D650, fitted on B.75's sets with this batch's code, against `data/.b75/fits/{unit}__{T0,TX}__D650__{F,L5}.npz`, units V1_r2_k0 and V4a | largest deviation at most 1e-6 eV |
| 3 | 200 rows per rule (D10 and D650 from the common set for F, S from the S set for F; stream `crc32("causality|<rule>")`). For each row, its file's beam columns with every row after t (a) raised (power + 1 MW after reading empty as zero, NB1B voltage + 10 kV) or (b) emptied; the nineteen scalars of row t recomputed with the row's anchor | identical bit for bit in (a) and (b); positive control (the raise applied from row t on) changes every row |
| after the fits | every T0 and TX file of this batch against B.75's file of the same unit, rule and target | same test rows; largest deviation reported |

Results before this text was finished:
- Check 1: 0 mismatches.
- Check 2: every deviation 0.0 eV, bit-identical, same fit rows and iterations.
- Check 3: 200 of 200 rows identical under each rule and modification; positive control 0 of 200 identical under each rule.

A first run of check 3 printed its result but stopped on a JSON type error after writing its file. The cast was fixed and the check rerun with the same result. No tolerance was changed.

## 9. Addition while the fits ran, before any readout (2026-11-20)

The readout code also forms TXN over TXNp (V1, D650, L5): the control's counterpart to TXN over TN. No statement uses it. When this was added, the fits had started and nothing had been scored. The session lead had not yet answered on the thirteen files, so section 7 stands.

## 10. After the readout (2026-11-20): what happened, nothing changed

- **Fits.** All 1,030 fits were run once, in four processes of two threads each, in about 15 minutes. No fit was repeated or changed.
- **Comparison with B.75.** Every T0 and TX file was then compared with B.75's: 432 files, all bit-identical.
- **Readout.** `readout79.py` was run once end to end and wrote the JSON and the markdown. Its progress log carried row counts only.
- **Answer on the thirteen files.** No answer from the session lead on the thirteen files had arrived before the readout. Section 7 was applied as written.
- **Changes.** No definition, row, fit or statement rule was changed after any number had been read.
- **Cross-check after the readout.** The points of TX over T0 and the RMSE of G, T0 and TX equal B.75's published points exactly (largest difference 0). The intervals differ slightly because the seed differs (20261179 against 20261175).

## 11. Validity rule of the beam inputs: the session lead's decision (2026-11-20), and what had been seen

**Order of events.**
1. The session lead was told of the thirteen files before any fit, with the default of section 7.
2. No answer came before the fits started.
3. The fits and the readout of section 10 were run with the columns as delivered. I read that readout: every table, and the verdicts. N1 to N6 and N9 to N11 held; N7 and N8 did not; the scoring-only sensitivity without the thirteen files changed no verdict.
4. After that, the session lead's decision arrived: option (b), stated as a validity rule of the beam inputs. The lead's message refers to no result. It was written in the belief that the fits had not been run, and I told the lead at once that they had.
5. The rule below is the lead's, not chosen by me.
6. This section was written after that readout and before any fit under the rule.

**The rule (the lead's wording).**
- A file-source pair is invalid if its power column is above 5 MW or below -0.1 MW in any grid row of the file. The sources are the five used: NB1B, NB1C, NB2A, NB2B, NB2C. Grid rows exclude a file's first and last row. An empty power violates nothing.
- A file with at least one invalid pair has no valid beam scalars: all nineteen are set non-finite on every row of that file.
- By the registration's row rule (complete for TXN: history, fast and beam scalars finite), those rows are in no fit and no score of this batch, for every learner, including T0 and TX. All learners therefore stay on identical rows.

**Selection.** Checked before any fit under the rule, on the power columns only, with no estimate. The rule selects exactly the thirteen files named in section 1 and no other; 29 file-source pairs are invalid. `docs/paper/b79_checks.md` section 4 gives, for each pair, the range of the power column, the rows above 5 MW and below -0.1 MW, and the median of V x I / 1000 on the rows with power above 0.1 MW. The beam scalars of the other files are bit-identical to those computed before the rule.

**Rows removed.**

| Set | Rows removed (rows left) | Discharges removed |
|---|---|---|
| Common set, F | 1,948 (163,470) | 11 |
| Common set, L5 | 1,783 (158,560) | 11 |
| S set, F | 1,783 (150,420) | 10 |
| S set, L5 | 1,640 (146,197) | 10 |

- No other row is removed.
- The thirteen files hold 2,913 D1 rows in 13 discharges. Two of these discharges have no row in the common set.

**What changes.**
- **Check 2** stays on B.75's row sets, as registered. The fits of the batch use the reduced sets.
- **The post-fit comparison** of T0 and TX with B.75 reports:
  - that every B.75 test row missing here is a row of the thirteen files;
  - the deviation of the estimates on the shared test rows. These are now refits on fewer fit rows and are not expected to be identical.
- **Check 3** is rerun with its 200 rows per rule drawn from the reduced sets.
- **Counts per unit.** Fit and test counts (discharges and rows with a row in the set) are reported per unit, in the output for V3 and V4 and in the JSON for every unit. Under V3 the first fifth, the fit of V3_q1, loses these early discharges.
- **One treatment.** No sensitivity with the columns as delivered, and no reconstruction of the power from V and I. The scoring-only sensitivity of section 7 is withdrawn: the readout no longer forms the stratum "outside the 13 files".
- **The as-delivered run** (fits, beam scalars, sets, three check files, logs and the three output files of section 10) is moved to `data/.b79/superseded_as_delivered/` (git-ignored). It is kept for the record and not reported.
- `docs/paper/b79_beam_inputs.{md,json}` and `b79_checks.md` are overwritten by the run under the rule. Their text states that the as-delivered readout had been seen before this run.
- Sections 2 to 6 and 8 are otherwise unchanged. No definition, learner, design, interval rule or statement rule is changed.

## 12. The run under the validity rule (2026-11-20): what happened

- **Checks before the fits.** Check 2, on B.75's sets, was rerun: every deviation 0.0 eV. Check 3 was rerun on the reduced sets: 200 of 200 rows identical under each rule and each modification, positive control 0 of 200.
- **Fits.** All 1,030 fits were run once on the reduced sets, in four processes of two threads each, in about 14 minutes.
- **Post-fit comparison with B.75.** The comparison of the T0 and TX files was run after the fits. It computes estimate differences only, no score against a target.
  - No test row of this batch lies outside B.75's test rows.
  - Every removed test row is a row of the thirteen files.
  - The refits differ from B.75's estimates on the shared test rows: largest single deviation 1002.6 eV; root mean square deviation per file median 35.6 eV, largest 91.1 eV.
  - After its first run, the root mean square deviation was added to the comparison and the comparison was rerun; the other figures were unchanged.
- **Code edits without effect on any number.** An unused grid load in `stage_beam79.py run_sets` and a needless f-string in `checks79.py` were removed.
- **Readout.** `readout79.py` was run once end to end. It overwrote `docs/paper/b79_beam_inputs.{md,json}`, and `checks79.py write` overwrote `docs/paper/b79_checks.md`.
- **Changes.** No definition, row rule, fit or statement rule was changed after any number of this run had been read.

## 13. Main readout and refit (the session lead's final instruction, 2026-11-20)

**What happened.**
- The session lead's instruction was to leave the thirteen files out of every fit and score through the validity rule of section 11. By the lead's account it was sent at about 16:50, before the fits. It was not applied: it reached the implementing agent only after the fits had run and after the readout with the columns as delivered had been run (17:04) and read.
- **The main readout** is that readout: the beam columns as delivered, as the registration reads (sections 7 and 10). Its statement table N1 to N11 is the judged one.
- **The refit** without the thirteen files (sections 11 and 12) was run after the main readout had been seen. Every number of the main readout was known at that time:
  - N1 to N11 with values, intervals and verdicts: N1 to N6 and N9 to N11 held, N7 and N8 did not;
  - the RMSE and skill tables by design and rule;
  - the ratio of the two MSE reductions, the halves, the strata by time since the last notch, V1 by chronological block, the two controls and the counts;
  - the scoring-only sensitivity of section 7, which changed no verdict.

**How the refit is reported.** It is a sensitivity, not judged as the registered statements, in its own section of `docs/paper/b79_beam_inputs.md` and under the key `refit_without_13_files` of the JSON. Section 11's "one treatment" and "the as-delivered run is superseded and not reported", and section 12's overwriting of the outputs, no longer apply. They are reversed by the lead's final instruction.

**What was done.**
- **Main outputs restored.** `docs/paper/b79_beam_inputs.{md,json}` and `docs/paper/b79_checks.md` were restored exactly as written at 17:04, and the refit was appended:
  - the markdown begins with the 17:04 text byte for byte, followed by the refit section;
  - the JSON equals the 17:04 JSON plus the key `refit_without_13_files`;
  - the checks begin with the 17:04 text, followed by the refit's checks.
- **Code.** The scripts select the run with `B79_VARIANT` (`main`, the default, or `refit`). The main path of `readout79.py`, `render79.py` and `checks79.py` was restored to its 17:04 form. A `merge` step writes the final JSON and markdown.
- **Verification of the restored code.**
  - Rendering the 17:04 JSON with the restored code gives the 17:04 markdown byte for byte, and the restored checks text equals the 17:04 checks file.
  - Both readouts were run once more into a scratch location, and each reproduced its stored JSON in every key except the run time.
  - The validity rule selects exactly the thirteen files (section 11).
- **Layout of the runs.**
  - Main run: `data/.b79/fits/`, `beam_{rule}.npz`, `sets.npz`, `checks/`, `readout_main.json`.
  - Refit: `data/.b79/refit_valid/fits/`, `beam_{rule}.npz`, `sets.npz`, `checks/`, `readout_refit.json`.
  - Copies of the three output files as written at 17:04: `data/.b79/main_outputs_as_written_1704/`.
  - The folder `data/.b79/superseded_as_delivered/` named in section 11 no longer exists: its caches are back at `data/.b79/`.
- **Removed rows.** The rows and discharges removed by the refit from each set and each unit are in the refit section and in the JSON (`removed_per_set`, `removed_per_unit`).
