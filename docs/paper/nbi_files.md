# The NBI files

Delivered 2026-11-20 (`data/nbi/s{shot}_NBI.csv`, not in the repository). Written by `ces_prediction/analyze_nbi_files.py`; no estimator and no scoring. A source counts as on in a row when its power is above 0.1 MW. The first (t = 0) and last (t = 30 s) rows of a file are left out of every count.

## 1. Files and rows

- NBI files: 884 (shots 30801 to 32765). Shot files in use: 641; with an NBI file: 641. NBI files of shots without a shot file: 243.
- Same number of rows as the shot file: 641 of 641 shots; same time stamps (within 0.1 ms): 641. An NBI file is the shot file's row list with the beam columns; it has no row for an absent frame. Grid rows: 245,925.
- Columns: `time`; for each of NB1A, NB1B, NB1C, NB2A, NB2B, NB2C: power `_P` (MW), voltage `_V` (kV), current `_I` (A), energies of the full, half and third component `_E1`, `_E2`, `_E3` (keV, empty while the source is off); `PNB_tot`; `Eb_mean`.
- `PNB_tot` equals the sum of the positive source powers (largest difference over all rows 0.0000 MW).

## 2. Sources

| source | rows on (share) | shots with it on | voltage while on, median over shots (10th to 90th percentile), kV | rows with an empty power (shots) |
|---|---|---|---|---|
| NB1A | 0 (0.000) | 0 | - | 0 (0) |
| NB1B | 242,613 (0.987) | 617 | 91.0 (80.8 to 91.1) | 0 (0) |
| NB1C | 177,334 (0.721) | 484 | 59.4 (54.4 to 69.1) | 295 (1) |
| NB2A | 216,808 (0.882) | 562 | 74.2 (69.2 to 87.9) | 15,323 (57) |
| NB2B | 179,211 (0.729) | 492 | 70.7 (70.3 to 80.4) | 37,226 (107) |
| NB2C | 75,168 (0.306) | 220 | 58.3 (53.4 to 67.9) | 136,872 (357) |

An empty power is read as off. Source combinations over a shot (sources on in at least one row):

| block of shots | shots | most frequent combinations (shots) |
|---|---|---|
| up to 31991 | 416 | 1B+1C+2A+2B (152); 1B+1C+2A+2B+2C (109); 1B+2A+2B (54); 1B+1C+2A (24); 1B+1C+2A+2C (22); 1B (18) |
| 32002-32310 | 128 | 1B+1C+2A+2B+2C (55); 1B+1C+2A+2B (28); 1B+2A (15); 1B+2A+2B (8); 1B (7); 1B+1C+2A (6) |
| from 32312 | 97 | 1B+1C+2A+2B (40); 1B+1C+2A+2B+2C (14); 1B+2A+2B (11); none (11); 1B+1C (8); 1B (4) |

Shots in which no source is above 0.1 MW in any row: 19; they have 149 rows in all, 145 of them with a T_i value (largest power in them 0.061 MW). Rows with every source below 0.1 MW, all shots: 149, of them with a T_i value: 145.

## 3. An absent frame is a notch of the NBI-1 sources

Absent steps (a time step of 20 to 40 ms inside a discharge): 1,414. Power of a source in the last row before and in the first row after the step, relative to its median in rows 5 to 10 after it (steps followed by ten 10 ms rows, source above 0.3 MW there):

| source | steps | last row before, median (10th percentile) | first row after, median (10th percentile) | share of first rows below 0.9 |
|---|---|---|---|---|
| NB1B | 1,350 | 0.999 (0.995) | 0.614 (0.593) | 0.976 |
| NB1C | 950 | 0.994 (0.987) | 0.609 (0.583) | 0.996 |
| NB2A | 1,250 | 1.000 (0.994) | 1.000 (0.995) | 0.026 |
| NB2B | 1,014 | 1.000 (0.995) | 1.000 (0.995) | 0.018 |
| NB2C | 412 | 1.000 (0.983) | 0.999 (0.984) | 0.039 |

The two NBI-1 sources that run (B and C) are low together in the first row after an absent frame; the NBI-2 sources are not. The absent frames are the notches of NBI-1, and the power written in the first row after one is a sample taken while the beam returns.

## 4. Is a CES value written, by the state of the sources

| state | rows | share with a T_i value | share with a V_rot value |
|---|---|---|---|
| NBI-1 off, NBI-2 off | 149 | 0.973 | 0.839 |
| NBI-1 off, NBI-2 on | 50 | 0.880 | 0.960 |
| 1C only | 1,177 | 0.965 | 0.613 |
| 1C and NBI-2 | 1,936 | 0.782 | 0.848 |
| 1B only | 16,387 | 0.917 | 0.665 |
| 1B and NBI-2 | 52,005 | 0.922 | 0.766 |
| 1B and 1C | 7,563 | 0.825 | 0.554 |
| 1B, 1C and NBI-2 | 166,658 | 0.930 | 0.784 |

Values are written with either NBI-1 source on. The rows with NBI-1 off are few.

## 5. Power columns that cannot be used

A power column of a file is taken as invalid when it is above 5 MW or below -0.1 MW in any row. Invalid file-source pairs: 29, in 13 shot files: 30801, 30805, 30809, 30810, 30811, 30812, 30842, 30843, 30845, 30895, 30906, 30907, 30909. All of them NBI-2 sources: yes.

| shot | source | smallest power (MW) | largest power (MW) | median of V x I / 1000 on the on rows (MW) | rows |
|---|---|---|---|---|---|
| 30801 | NB2A | 135.74 | 137.16 | 1.12 | 33 |
| 30801 | NB2B | -1.00 | 0.04 | nan | 33 |
| 30805 | NB2A | -2.76 | -1.18 | nan | 36 |
| 30805 | NB2B | -0.42 | 74.16 | 0.20 | 36 |
| 30809 | NB2A | 1.42 | 143.83 | 1.15 | 298 |
| 30809 | NB2B | -2.02 | 75.16 | 0.00 | 298 |
| 30810 | NB2A | -2.76 | 142.97 | 1.15 | 438 |
| 30810 | NB2B | -1.21 | 75.29 | 2.41 | 438 |
| 30811 | NB2A | -2.89 | 141.96 | 1.14 | 448 |
| 30811 | NB2B | -1.06 | 74.90 | 2.41 | 448 |
| 30812 | NB2A | -2.61 | 141.62 | 1.14 | 448 |
| 30812 | NB2B | -1.37 | 74.87 | 2.40 | 448 |
| 30842 | NB2A | -2.68 | 67.42 | 1.96 | 447 |
| 30842 | NB2B | 64.34 | 66.50 | 2.01 | 447 |
| 30842 | NB2C | 1.50 | 50.46 | 1.28 | 447 |
| 30843 | NB2A | -2.55 | 63.69 | 1.94 | 189 |
| 30843 | NB2B | -1.22 | 0.83 | 0.00 | 189 |
| 30845 | NB2A | -2.55 | 63.86 | 1.95 | 189 |
| 30845 | NB2B | -1.19 | 0.94 | 0.00 | 189 |
| 30895 | NB2A | 81.25 | 82.59 | 2.79 | 297 |
| 30895 | NB2B | -0.18 | 0.03 | nan | 297 |
| 30906 | NB2A | -3.48 | 83.43 | 2.82 | 120 |
| 30906 | NB2B | 62.91 | 64.45 | 1.96 | 120 |
| 30907 | NB2A | 82.00 | 83.55 | 2.82 | 297 |
| 30907 | NB2B | 62.62 | 64.24 | 1.96 | 297 |
| 30907 | NB2C | 39.28 | 41.39 | 0.99 | 297 |
| 30909 | NB2A | -1.79 | 92.99 | 3.29 | 297 |
| 30909 | NB2B | 64.46 | 65.84 | 2.00 | 297 |
| 30909 | NB2C | 39.45 | 41.35 | 1.00 | 297 |

In the valid files the power is a steady fraction of V x I (median over shots of the per-shot median of P / (V x I / 1000), source on in at least 20 rows): NB1B 0.381; NB1C 0.588; NB2A 0.298; NB2B 0.308; NB2C 0.353. In the files above the voltage and the current of the source are of the usual size and the power column is not: the column is wrong, not the source. The on and off states of sections 2 to 4 are not affected by a wrong scale (a real power is above 0.1 MW); a negative offset reads as off. Batch B.79 reads its statements with the columns as delivered (its registration) and, beside that, refitted with these files left out of every fit and score.

