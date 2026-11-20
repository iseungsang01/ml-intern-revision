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

