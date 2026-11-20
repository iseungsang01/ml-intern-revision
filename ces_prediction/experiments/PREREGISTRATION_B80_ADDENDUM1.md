# B.80 — Addendum 1 (implementing agent): implementation choices

Written 2026-11-20, before the check of registration section 8 was run and before any readout. When this text was written, no stored estimate of B.75 or B.79 had been read by any code of this batch: no RMSE, skill, difference, gated estimate, Spearman correlation or statement had been computed, overall or by class.

What had been looked at:
- the registration;
- B.75: its registration and Addendum 1, `readout.py`, `render.py`, `common75.py`, and the published `docs/paper/b75_availability.md` (its tables existed before this batch was registered; the check of section 8 compares against them);
- B.79: its registration and Addendum 1, `readout79.py`, `common79.py`;
- B.73: `common.py`, `designs.py`;
- the session lead's diagnostic `docs/paper/b75_last_block.md` and `lead_checks/last_block.py`;
- the keys, shapes and dtypes of the cached arrays (`grid.npz`, `rows.npz`, `sets.npz`) and of three fit files (`idx`, `est`, `fit_rows`, `n_iter`), not their estimate values;
- input facts of the ECEI columns and of the measures of section 2, computed by `b80_ecei_gate/measures80.py` (no CES value, no estimate). They are given in section 1.

Facts and choices only; no registered definition, cut or statement is changed. Code: `ces_prediction/experiments/b80_ecei_gate/`; caches `data/.b80/`; outputs `docs/paper/b80_ecei_gate.{md,json}`. A change made after this text was first written is added in a dated section at the end, with what had been seen at that time.

## 1. Facts looked at before any readout (inputs only)

- **ECEI columns.** `fast[:, 9:13]` of `data/.b73/grid.npz`. No non-finite value on any of the 247,207 rows.
- **Grid rows per discharge** (a file's first and last row excluded): 46 of the 641 files have fewer than 50 (41 of the 635 D1 discharges). 131 files with C_E have at most 200 grid rows.
- **Constant channels.** Twelve files have at least one ECEI channel with zero variance over their grid rows: 30967, 31334, 31482, 31483, 31484, 31485, 31702, 32149, 32150, 32151, 32152, 32153. Eight of them have fewer than 50 grid rows. The other four (32149, 32150, 32151, 32153; 288 grid rows each, all D1, block 32002-32310) have channel 4 constant. Over the first 200 grid rows, 31898 and 31900 also have a constant channel.
- **Distribution of C_E over the D1 discharges**, cuts as registered (computed by `measures80.py summary`; the full table is in the output):

| block of shots | D1 discharges | with C_E | low / middle / high | share low | C_E median (25-75 %) |
|---|---|---|---|---|---|
| all | 635 | 590 | 170 / 135 / 285 | 0.288 | 0.895 (0.626-0.961) |
| up to 31991 | 412 | 393 | 87 / 95 / 211 | 0.221 | 0.911 (0.760-0.962) |
| 32002-32310 | 128 | 115 | 40 / 32 / 43 | 0.348 | 0.864 (0.583-0.953) |
| from 32312 | 95 | 82 | 43 / 8 / 31 | 0.524 | 0.684 (0.385-0.960) |

- **C_E early.** Defined for 588 D1 discharges; the same class as C_E for 0.852 of them (0.810 among those with more than 200 grid rows).
- **N_E.** Defined for 259 D1 discharges (215, 42 and 2 per block); median 0.920.

The cuts 0.7 and 0.9 are those of the registration and are not changed.

## 2. The measures (`measures80.py`, cached in `data/.b80/measures.npz`)

- **Discharge and its grid rows.** A discharge is a shot file of the grid. Its grid rows are the rows of the file other than the first (t = 0) and the last (t = 30 s), as `b79_beam_inputs/common79.grid_row_mask`. All blocks of the file are pooled, in time order. Every grid row enters, whether it carries a CES value or not; no CES value is read.
- **C_E.** For discharges with at least 50 grid rows: the six Pearson correlations of the four raw ECEI columns (float64 of the stored float32) over the grid rows (`numpy.corrcoef`), and their mean.
  - **A channel with zero variance** over the rows has undefined correlations, so the mean of the six does not exist. Such a discharge has no C_E (status "constant channel"; four D1 discharges, section 1). It belongs to no class.
  - A non-finite ECEI value would stop the stage (none exists).
- **C_E early.** The same over the first 200 grid rows of the discharge in time order, whatever block they belong to; where the first block is shorter they cross a block boundary. A discharge with 50 to 200 grid rows has C_E early = C_E (the count is reported). C_E early can be undefined where C_E is defined (a channel constant over the first 200 rows). **Agreement** = the share of the discharges with both in which both fall into the same class; also given per block and for discharges with more than 200 grid rows only.
- **N_E.**
  - Absent step: B.75's, a step between consecutive grid rows j and j + 1 of the same block above 15 ms and below 45 ms (`AB_LO`, `AB_HI` of `common75.py`).
  - "Five rows of the block before it": rows j − 4 to j are grid rows of the block of j.
  - ECEI array mean of a row: the raw mean of the four channels (`common75.raw_means`).
  - Ratio of a step: the array mean of row j + 1 over the median of the array means of rows j − 4 to j. A step whose median is not above zero gives no ratio (six steps over the D1 discharges).
  - N_E = the median of the ratios of the discharge, defined for discharges with at least 50 grid rows and at least three ratios.
  - Reported only.
- **Classes.** low: C_E < 0.7; middle: 0.7 ≤ C_E < 0.9; high: C_E ≥ 0.9 (float64 comparison). Without C_E: no class. The class of a row is the class of its discharge.
- **Block of a discharge.** B.73's blocks (`common.BLOCKS`, `rows.npz` key `blk`): up to 31991, 32002-32310, from 32312. Every D1 discharge is in one of the three.
- **Population of the distribution tables.** The D1 discharges (635), per block. The same over all 641 grid files is given beside it.

## 3. Rows, estimates, strata and intervals

- **Units and scored rows.** The units of B.73 (`designs.units`), as B.75 (`readout.subs_of`): V1 r2-r4 × 5 folds, V2 seeds 1-3 × 5 folds, V3 fifths 2-5 (one sub), V4a, V4b. The scored rows of a design are the test rows of its units within B.75's set: `common_L5` for D650, `S_L5` for S (`data/.b75/sets.npz`).
- **Estimates read** (never written):
  - `data/.b75/fits/{unit}__{T0|TX|TB|TE}__{D650|S}__L5.npz`, TB and TE for V1, V4a and V4b only;
  - `data/.b79/fits/{unit}__{TN|TXN}__{D650|S}__L5.npz`, the main run of B.79 (beam columns as delivered), not `data/.b79/refit_valid/`.
  - A scored row without an estimate stops the readout (`Group.add`).
- **The gated estimator TG.** Per row and per split: the stored TX estimate of the same unit and rule where the discharge's C_E is 0.7 or above (middle or high class), the stored T0 estimate otherwise (low class, and every discharge without C_E). No fit.
- **Strata of a cell** (design × rule; target L5):
  - "all";
  - "class low", "class middle", "class high", "no class";
  - "low+high": the rows of the low and the high class together, for the differences.
  - Under V1, in addition, the same strata within each block ("block up to 31991", "block 32002-32310", "block from 32312", each with "|class ..." and "|low+high") and within "blocks up to 32310" (the two earlier blocks pooled; G5).
  - A stratum without rows is not formed. A difference is formed only where both classes have rows.
- **Points.** As B.75: V1 the mean of the three re-partitions' values, V2 of the three assignments, V3 pooled over fifths 2-5, V4a and V4b the test block. RMSE: the mean over splits of the per-split RMSE. Skill of A over B = 1 − SSE(A) / SSE(B) on the rows of the stratum.
- **Intervals.** B.75's class `Group` (`b75_availability/readout.py`), imported by path, with its `rng_for` replaced by this batch's, `[20261180, crc32(name)]`.
  - 20,000 draws; 2.5 and 97.5 % quantiles.
  - V1: a re-partition is drawn first; V2: an assignment; V3 and V4: the units of the scored rows.
  - Units: discharges (discharge form) or shot clusters (cluster form) with at least one row in the stratum.
  - One stream per cell, stratum and form: `"design|L5|set|stratum|form"`, for example `"V1|L5|S|class high|cluster"`, with set "common" (D650) or "S".
  - Every quantity of a stratum and form is formed draw by draw on the same draws.
- **Skill of one class.** The class's stratum, with its own draws over the units that have rows in the class (as B.75's strata by block).
- **Difference of two class skills** (G1, G4, G5, and the same difference in the tables).
  - Formed in the stratum "low+high". For each learner A, two masked squared errors: A@high (the squared error on rows of the high class, zero elsewhere) and A@low; they are summed per unit.
  - The draws resample the units that have rows in the low or the high class. Per draw: (1 − SSE_high(TX) / SSE_high(T0)) − (1 − SSE_low(TX) / SSE_low(T0)), from the same draw's sums.
  - A cluster with discharges of both classes enters both terms with the same weight.
  - **A draw with no unit of one class** gives 0/0 for that class: the value is not finite. The quantiles are taken over the finite draws, and the number of draws left out is reported per form.
  - Point: the mean over splits of the per-split difference (the difference of the two class points).
  - The same difference is formed for TE over T0, TB over T0 and TXN over TN; no statement uses those.

## 4. The Spearman correlation (V1, rule S)

- **Population.** D1 discharges with C_E and at least 50 rows in the S set for L5. Under V1 each is scored once in each re-partition.
- **Per-discharge skill under re-partition r.** 1 − SSE_d(TX) / SSE_d(T0) over the S-set rows of discharge d, with the stored estimates under S of the unit of r that scores d.
- **Point.** The mean over the three re-partitions of the Spearman correlation (`scipy.stats.spearmanr`, average ranks for ties) between C_E and the per-discharge skill of that re-partition.
- **Interval.** Per draw, a re-partition is drawn first; then units are resampled with replacement: discharges (discharge form) or shot clusters (cluster form; a drawn cluster brings all its discharges in the population). The Spearman correlation is computed on the resampled discharges, a discharge drawn k times entering k times. 20,000 draws; stream `"V1|L5|S|spearman|<stratum>|<form>"`.
- **Beside it** (added here, not registered, no statement): the same within each block.

## 5. Statements: how each is judged (cluster form, target L5)

| Statement | Quantity | Holds if |
|---|---|---|
| G1 | V1, S, stratum "low+high": TX over T0 in the high class minus in the low class | lower end above zero |
| G2 | V1, S, "class low": TE over T0 | lower end at or below zero (the interval contains zero or lies below it) |
| G3 | V1, S, "class high": TE over T0 | lower end above zero |
| G4 | V1, D650, "low+high": TX over T0, high minus low | lower end above zero |
| G5 | V1, S, "blocks up to 32310\|low+high": TX over T0, high minus low | lower end above zero |
| G6 | V3, S, "class high": TX over T0 | lower end above zero |
| G7 | V1, S, "block from 32312": TG over T0 | lower end above zero |
| G8 | V4a (fit to 31991), S, "all": TG over T0 | lower end above zero |
| G9 | V1, S, "all": TG over TX | upper end at or above zero (the interval is not below zero) |

- "V4" in G8 is V4a, as the statement says; V4b is given in the note.
- A statement whose stratum has no rows is reported as not formed, neither held nor not held. For a difference, a non-finite draw is left out of the quantiles as in section 3.
- **Expectation met.** B.75's `expectation_met` (imported). The expectations are copied from the registration unchanged.

## 6. The check of registration section 8 (before any readout by class)

- **Quantity.** TX over T0, target L5, stratum "all", under D650 (common set) and S (S set), for V1, V2, V3, V4a and V4b. It is computed from the stored estimates read by this batch, with B.75's point rule.
- **Pass rule.** The point rounded to three decimals equals the value printed in `docs/paper/b75_availability.md` (table "The fast-channel increment TX over T0 by rule and design, target L5"). The difference from the full-precision point of `docs/paper/b75_availability.json` is reported beside it.
- **Order.** It runs in `check80.py` before the readout. If it fails, no readout by class is run until the cause is found and reported. The readout recomputes the same points in its pass and records both.
- **C_E reads only the ECEI columns**: stated (registration section 8). `measures80.py` loads the grid's time, block, row number and fast columns and indexes only columns 9 to 12 of the fast array.

## 7. Output and run

- Every readout goes into `docs/paper/b80_ecei_gate.json` in one pass of `readout80.py`; the markdown `docs/paper/b80_ecei_gate.md` is rendered from that JSON by `render80.py`.
- The progress log of the readout prints row counts only.
- One process with two threads; no model is fitted.
- **Readouts beside the registered ones** (no statement uses them): the RMSE of TN, TXN, TB, TE and TG; the "no class" stratum; the differences high minus low for TE over T0, TB over T0 and TXN over TN; the Spearman correlation within each block; the distribution over all 641 grid files.

## 8. After the readout (2026-11-20): what happened, nothing changed

- `check80.py` was run once, after this text had been written and before the readout: all ten cells pass (largest difference from B.75's JSON 2.2e-16).
- `readout80.py` was run once end to end (96 s) and wrote the JSON and the markdown; its progress log carried row counts only. The check recomputed in the readout pass passes.
- No draw of any difference of class skills was left out (every count 0).
- The "no class" stratum holds the four discharges with a constant channel only; the discharges with fewer than 50 grid rows have no row in the S set or the common set.
- After the readout, the points of G1 (the two class skills and their difference), of TG over TX and the RMSE of TG under V1 and S were recomputed by separate code without the interval machinery; they equal the JSON values.
- No definition, cut, row, stratum or statement rule was changed after any number had been read.
