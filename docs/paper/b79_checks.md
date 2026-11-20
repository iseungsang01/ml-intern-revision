# B.79 checks

Registration: `ces_prediction/experiments/PREREGISTRATION_B79.md` section 9; implementation: `ces_prediction/experiments/PREREGISTRATION_B79_ADDENDUM1.md`. Run before any readout. No tolerance was changed after a run.

## 1. The join of the beam rows to the grid rows

Shot files 641; without an NBI file: 0; with another number of rows: 0. Rows 247,207 (grid rows without each file's first and last row: 245,925). Rows whose beam stamp differs from the grid stamp by more than 1e-04 s or that have no beam row: **0** (grid rows: 0). Largest stamp difference 9.5e-07 s; rows whose beam stamp rounded to float32 equals the grid stamp: 247,207.

Check 1 passes: **True**

## 2. T0 and TX on B.75's row sets against B.75's stored estimates (D650)

| unit / learner / rule / target | same test rows | largest deviation (eV) | bit-identical | fit rows (B.75) | iterations (B.75) |
|---|---|---|---|---|---|
| V1_r2_k0 / T0 / D650 / F | True | 0.0e+00 | True | 137,852 (137,852) | 100 (100) |
| V1_r2_k0 / T0 / D650 / L5 | True | 0.0e+00 | True | 133,674 (133,674) | 100 (100) |
| V1_r2_k0 / TX / D650 / F | True | 0.0e+00 | True | 137,852 (137,852) | 100 (100) |
| V1_r2_k0 / TX / D650 / L5 | True | 0.0e+00 | True | 133,674 (133,674) | 100 (100) |
| V4a / T0 / D650 / F | True | 0.0e+00 | True | 130,192 (130,192) | 100 (100) |
| V4a / T0 / D650 / L5 | True | 0.0e+00 | True | 127,270 (127,270) | 100 (100) |
| V4a / TX / D650 / F | True | 0.0e+00 | True | 130,192 (130,192) | 100 (100) |
| V4a / TX / D650 / L5 | True | 0.0e+00 | True | 127,270 (127,270) | 100 (100) |

The row sets of this batch equal B.75's: common_F True, common_L5 True, S_F True, S_L5 True (no row of B.75's sets lacks a beam scalar), so there is no difference on the rows of this batch to report.

Check 2 passes: **True**

After the fits: every T0 and TX file of this batch (36 units x D10, D650, S x F, L5) against B.75's file of the same unit, rule and target: 432 files, bit-identical 432, largest deviation 0.0e+00 eV, files with other rows 0.

## 3. No beam scalar of row t uses a row after t

For each row, a copy of its file's beam columns in which every row after t is changed (raised: every power + 1 MW with an empty power read as zero first, NB1B voltage + 10 kV; emptied: every power and the voltage made empty); the nineteen scalars of row t recomputed with the row's anchor and compared bit for bit with the main pass. Positive control: the same raise applied from row t on must change the scalars of every row.

| rule | rows (candidates) | raised: rows identical | emptied: rows identical | positive control: rows identical (must be 0) |
|---|---|---|---|---|
| D10 | 200 (165,418) | 200 | 200 | 0 |
| D650 | 200 (165,418) | 200 | 200 | 0 |
| S | 200 (152,203) | 200 | 200 | 0 |

Check 3 passes: **True**

## The beam columns (input facts, no target and no estimate)

Grid rows 245,925. An empty power is read as zero.

| source | empty rows | largest (MW) | smallest (MW) | 99th percentile (MW) | rows above 3 MW (files) | rows below -0.1 MW (files) |
|---|---|---|---|---|---|---|
| NB1B | 0 | 1.65 | -0.00 | 1.583 | 0 (0) | 0 (0) |
| NB1C | 295 | 1.31 | -0.01 | 1.236 | 0 (0) | 0 (0) |
| NB2A | 15,323 | 143.83 | -3.48 | 64.414 | 3,078 (12) | 445 (9) |
| NB2B | 37,226 | 75.29 | -2.02 | 1.097 | 2,045 (9) | 763 (9) |
| NB2C | 136,872 | 50.46 | -0.02 | 0.850 | 993 (3) | 0 (0) |

Every power above 3 MW lies in thirteen files (shots 30801, 30805, 30809, 30810, 30811, 30812, 30842, 30843, 30845, 30895, 30906, 30907, 30909), all NBI-2 columns; 2,652 D1 rows carry one. In the same files the NBI-2 powers read as low as -3.5 MW while the source is off. The NB1B voltage is empty on 0 rows with NB1B on and on 0 rows with it off.

## Row sets

D1 rows 222,672. Rows whose nineteen beam scalars are defined: D10 220,407, D650 185,522, S 153,332.

| set | B.75 | B.79 | dropped by the beam scalars | rows of the thirteen files |
|---|---|---|---|---|
| common, F | 165,418 | 165,418 | D10 0, D650 0 | 1,948 |
| S, F | 152,203 | 152,203 | 0 | 1,783 |
| common, L5 | 160,343 | 160,343 | D10 0, D650 0 | 1,783 |
| S, L5 | 147,837 | 147,837 | 0 | 1,640 |

Undefined beam scalars over all D1 rows (they lie outside B.75's sets, which already require the anchor and its running means): D10: mean10 Pt(t) 1,658, mean50 Pt(t) 1,658, mean10 Pt(anchor) 2,265, mean50 Pt(anchor) 2,265; D650: P1(anchor) 34,975, P1(t)-P1(anchor) 34,975, P2(anchor) 34,975, P2(t)-P2(anchor) 34,975, Pt(anchor) 34,975, Pt(t)-Pt(anchor) 34,975, mean10 Pt(t) 1,658, mean50 Pt(t) 1,658, mean10 Pt(anchor) 37,150, mean50 Pt(anchor) 37,150; S: P1(anchor) 69,290, P1(t)-P1(anchor) 69,290, P2(anchor) 69,290, P2(t)-P2(anchor) 69,290, Pt(anchor) 69,290, Pt(t)-Pt(anchor) 69,290, mean10 Pt(t) 1,658, mean50 Pt(t) 1,658, mean10 Pt(anchor) 69,340, mean50 Pt(anchor) 69,340.

## Refit without the thirteen files (sensitivity; Addendum 1 sections 11 to 13)

Run after the main readout had been read, under the session lead's validity rule. Caches: `data/.b79/refit_valid/`. Not judged as the registered statements.

### R1. The validity rule selects exactly the thirteen files

Rule: a file-source pair (sources NB1B, NB1C, NB2A, NB2B, NB2C) is invalid if its power column is above 5.0 MW or below -0.1 MW in any grid row of the file. A file with at least one invalid pair has no valid beam scalars: all nineteen are non-finite on every row of the file, and by the registration's row rule (complete for TXN) its rows are in no fit and no score, for every learner.

Files selected: 13 (30801, 30805, 30809, 30810, 30811, 30812, 30842, 30843, 30845, 30895, 30906, 30907, 30909); exactly the thirteen files named beforehand: **True**. Invalid file-source pairs: 29.

| shot | source | power column, smallest to largest (MW) | rows above 5 MW | rows below -0.1 MW | rows with power above 0.1 MW | median V x I / 1000 on those rows |
|---|---|---|---|---|---|---|
| 30801 | NB2A | 135.74 to 137.16 | 33 | 0 | 33 | 1.12 |
| 30801 | NB2B | -1.00 to 0.04 | 0 | 28 | 0 | - |
| 30805 | NB2A | -2.76 to -1.18 | 0 | 36 | 0 | - |
| 30805 | NB2B | -0.42 to 74.16 | 16 | 8 | 28 | 0.20 |
| 30809 | NB2A | 1.42 to 143.83 | 297 | 0 | 298 | 1.15 |
| 30809 | NB2B | -2.02 to 75.16 | 48 | 124 | 121 | 0.00 |
| 30810 | NB2A | -2.76 to 142.97 | 386 | 47 | 390 | 1.15 |
| 30810 | NB2B | -1.21 to 75.29 | 143 | 147 | 241 | 2.41 |
| 30811 | NB2A | -2.89 to 141.96 | 392 | 54 | 393 | 1.14 |
| 30811 | NB2B | -1.06 to 74.90 | 335 | 49 | 382 | 2.41 |
| 30812 | NB2A | -2.61 to 141.62 | 389 | 56 | 392 | 1.14 |
| 30812 | NB2B | -1.37 to 74.87 | 337 | 32 | 397 | 2.40 |
| 30842 | NB2A | -2.68 to 67.42 | 444 | 2 | 445 | 1.96 |
| 30842 | NB2B | 64.34 to 66.50 | 447 | 0 | 447 | 2.01 |
| 30842 | NB2C | 1.50 to 50.46 | 399 | 0 | 447 | 1.28 |
| 30843 | NB2A | -2.55 to 63.69 | 39 | 149 | 40 | 1.94 |
| 30843 | NB2B | -1.22 to 0.83 | 0 | 146 | 17 | 0.00 |
| 30845 | NB2A | -2.55 to 63.86 | 118 | 70 | 119 | 1.95 |
| 30845 | NB2B | -1.19 to 0.94 | 0 | 134 | 16 | 0.00 |
| 30895 | NB2A | 81.25 to 82.59 | 297 | 0 | 297 | 2.79 |
| 30895 | NB2B | -0.18 to 0.03 | 0 | 95 | 0 | - |
| 30906 | NB2A | -3.48 to 83.43 | 91 | 29 | 91 | 2.82 |
| 30906 | NB2B | 62.91 to 64.45 | 120 | 0 | 120 | 1.96 |
| 30907 | NB2A | 82.00 to 83.55 | 297 | 0 | 297 | 2.82 |
| 30907 | NB2B | 62.62 to 64.24 | 297 | 0 | 297 | 1.96 |
| 30907 | NB2C | 39.28 to 41.39 | 297 | 0 | 297 | 0.99 |
| 30909 | NB2A | -1.79 to 92.99 | 293 | 2 | 295 | 3.29 |
| 30909 | NB2B | 64.46 to 65.84 | 297 | 0 | 297 | 2.00 |
| 30909 | NB2C | 39.45 to 41.35 | 297 | 0 | 297 | 1.00 |

Rows over the file's grid rows (first and last row excluded). Beam scalars of the refit against the main run: D10: rows of the other files bit-identical True (219,759 rows), rows of the thirteen files all non-finite True (2,913 rows); D650: rows of the other files bit-identical True (219,759 rows), rows of the thirteen files all non-finite True (2,913 rows); S: rows of the other files bit-identical True (219,759 rows), rows of the thirteen files all non-finite True (2,913 rows).

### R2. Row sets of the refit

D1 rows of the thirteen files: 2,913 in 13 discharges.

| set | main (= B.75) rows (discharges) | refit rows (discharges) | rows removed: of the thirteen files / otherwise | discharges removed |
|---|---|---|---|---|
| common, F | 165,418 (564) | 163,470 (553) | 1,948 / 0 | 11 |
| S, F | 152,203 (506) | 150,420 (496) | 1,783 / 0 | 10 |
| common, L5 | 160,343 (562) | 158,560 (551) | 1,783 / 0 | 11 |
| S, L5 | 147,837 (503) | 146,197 (493) | 1,640 / 0 | 10 |

### R3. Check 2 rerun (on B.75's sets, as registered) and check 3 rerun (on the refit's sets)

Check 2: largest deviation 0.0e+00 eV over 8 fits; passes **True**.

| rule | rows (candidates) | raised: rows identical | emptied: rows identical | positive control: rows identical (must be 0) |
|---|---|---|---|---|
| D10 | 200 (163,470) | 200 | 200 | 0 |
| D650 | 200 (163,470) | 200 | 200 | 0 |
| S | 200 (150,420) | 200 | 200 | 0 |

Check 3 passes: **True**

### R4. The refit's T0 and TX against B.75's

Every T0 and TX file of the refit (36 units x D10, D650, S x F, L5; 432 files) against B.75's file of the same unit, rule and target. Every test row of the refit is a test row of B.75's; every B.75 test row not in the refit is a row of the thirteen files (problems found: 0). Because the fit rows lose the rows of the thirteen files, the estimates differ on the shared test rows: largest deviation 1002.6 eV, median over files of the per-file largest deviation 483.1 eV; root mean square deviation per file: median 35.6 eV, largest 91.1 eV. Units V1_r2_k0 and V4a under D650:

| unit / learner / rule / target | test rows (B.75) | fit rows (B.75) | largest deviation on shared test rows (eV) | root mean square deviation (eV) |
|---|---|---|---|---|
| V1_r2_k0 / T0 / D650 / F | 27,094 (27,566) | 136,376 (137,852) | 475.5 | 29.7 |
| V1_r2_k0 / T0 / D650 / L5 | 26,266 (26,669) | 132,294 (133,674) | 610.8 | 27.2 |
| V1_r2_k0 / TX / D650 / F | 27,094 (27,566) | 136,376 (137,852) | 511.7 | 41.7 |
| V1_r2_k0 / TX / D650 / L5 | 26,266 (26,669) | 132,294 (133,674) | 534.4 | 37.4 |
| V4a / T0 / D650 / F | 10,662 (10,662) | 128,244 (130,192) | 364.6 | 32.2 |
| V4a / T0 / D650 / L5 | 10,146 (10,146) | 125,487 (127,270) | 488.3 | 28.7 |
| V4a / TX / D650 / F | 10,662 (10,662) | 128,244 (130,192) | 304.2 | 48.5 |
| V4a / TX / D650 / L5 | 10,146 (10,146) | 125,487 (127,270) | 308.1 | 43.4 |
