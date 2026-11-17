# B.43 — checks of the reported T_i value that the files allow (registered: ces_prediction/experiments/PREREGISTRATION_B43.md)

CPU; no model trained. Part D runs the frozen network of the trace figure forward on the CPU for one discharge. Intervals: 95 %, 2,000 resamples of discharges unless stated. Grid = rows of the 641 files without the first (t = 0) and last (t = 30 s) row. T_i with held values removed; values above 3 keV removed except in part F.

## Registered statements

- **S1** (median over discharges of the Spearman correlation of 200 ms block-mean changes of T_i and of the ECEI level, minus the median of the circularly shifted null, interval above zero): +0.268 [+0.185, +0.372] minus +0.003 [-0.178, +0.103] = **+0.265 [+0.126, +0.495]** (15 discharges) -> **holds**.
- **F1** (share of rows above 3 keV in the lowest fifth of the BES level over that in the highest fifth, interval above 1): **3.18 [2.49, 4.04]** -> **holds**.
- **H1** (discharge 31923: an absent frame within 30 ms of 3.5 s and of 5.0 s, and mean T_i over 4.10-4.90 s over that over 3.60-3.95 s with an interval above 1): absent frames within 30 ms of 3.5 s: none (the grid of the file begins at 3.512 s); within 30 ms of 5.0 s: 5.002 s; ratio **1.037 [1.011, 1.067]** (2,000 resamples of 100 ms blocks of rows) -> **does not hold**.

## S. Co-variation with the ECEI level and the BES level

Blocks of consecutive grid rows inside each contiguous segment (breaks of at least 0.5 s); a block is used when at least half its rows have a measured T_i; block means of T_i over its measured rows, of the ECEI level (mean of the four channels) and of the BES level (mean of the nine channels) over all its rows; one value per discharge with enough pairs of consecutive used blocks. Null: the level changes shifted circularly by half the number of pairs.

| blocks | level | discharges | median rho | median null | difference | share of discharges with rho > 0 |
|---|---|---|---|---|---|---|
| 20 rows (200ms), min 10 measured, min 40 pairs | ECEI | 15 | +0.268 [+0.185, +0.372] | +0.003 [-0.178, +0.103] | +0.265 [+0.126, +0.495] | 0.933 |
| 20 rows (200ms), min 10 measured, min 40 pairs | BES | 15 | +0.508 [+0.352, +0.724] | +0.048 [-0.016, +0.138] | +0.460 [+0.285, +0.730] | 0.867 |
| 10 rows (100ms), min 5 measured, min 80 pairs | ECEI | 16 | +0.351 [+0.303, +0.443] | -0.030 [-0.137, +0.067] | +0.381 [+0.277, +0.496] | 0.938 |
| 10 rows (100ms), min 5 measured, min 80 pairs | BES | 16 | +0.581 [+0.173, +0.763] | -0.005 [-0.152, +0.042] | +0.586 [+0.188, +0.798] | 0.938 |
| 50 rows (500ms), min 25 measured, min 16 pairs | ECEI | 8 | +0.296 [-0.083, +0.538] | +0.037 [-0.144, +0.274] | +0.259 [-0.264, +0.666] | 0.750 |
| 50 rows (500ms), min 25 measured, min 16 pairs | BES | 8 | +0.487 [+0.224, +0.751] | -0.026 [-0.100, +0.062] | +0.513 [+0.217, +0.784] | 1.000 |

By third of the discharge median T_i (200 ms blocks):

| level | third | discharges | median T_i range (eV) | median rho | median null | difference |
|---|---|---|---|---|---|---|
| ECEI | 1 | 5 | 186-440 | +0.251 [-0.274, +0.604] | -0.227 [-0.319, +0.103] | +0.478 [-0.047, +0.923] |
| ECEI | 2 | 5 | 525-661 | +0.268 [+0.001, +0.389] | +0.003 [-0.137, +0.215] | +0.265 [-0.057, +0.511] |
| ECEI | 3 | 5 | 667-933 | +0.339 [+0.153, +0.650] | +0.106 [-0.178, +0.145] | +0.233 [+0.008, +0.544] |
| BES | 1 | 5 | 186-440 | +0.228 [-0.158, +0.580] | +0.048 [-0.016, +0.289] | +0.180 [-0.296, +0.367] |
| BES | 2 | 5 | 525-661 | +0.724 [+0.457, +0.780] | +0.024 [-0.137, +0.241] | +0.700 [+0.334, +0.917] |
| BES | 3 | 5 | 667-933 | +0.508 [+0.407, +0.859] | +0.063 [-0.268, +0.231] | +0.445 [+0.177, +1.127] |

Quantiles 0.1/0.25/0.5/0.75/0.9 of the per-discharge rho (200 ms, ECEI): +0.062 / +0.182 / +0.268 / +0.380 / +0.599.

## F. Where the values above 3 keV sit

Grid rows with a T_i value (held removed, no 3 keV cut): 226,990 in 638 discharges, 1,197 above 3 keV. BES level of a row = mean of the nine channels over the discharge median (as in B.40); fifths are pooled quantiles over these rows (edges 0.933, 0.985, 1.021, 1.074).

| fifth of the BES level | rows with T_i | above 3 keV | share above 3 keV | missing T_i (all grid rows, same definition) |
|---|---|---|---|---|
| 1 | 45,398 | 613 | 1.35 % [1.10, 1.63] | 20.28 % [18.25, 22.35] |
| 2 | 45,398 | 175 | 0.39 % [0.28, 0.50] | 5.46 % [4.46, 6.54] |
| 3 | 45,398 | 124 | 0.27 % [0.20, 0.35] | 3.57 % [2.92, 4.31] |
| 4 | 45,398 | 92 | 0.20 % [0.15, 0.26] | 3.68 % [3.00, 4.44] |
| 5 | 45,398 | 193 | 0.43 % [0.34, 0.52] | 5.50 % [4.56, 6.49] |

Ratio lowest over highest fifth: 3.18 [2.49, 4.04] (resamples with no value above 3 keV in the highest fifth: 0).

Values above 3 keV within ten rows after an absent frame: 50 of 1197, share 4.18 % [2.85, 5.58]; the same share for all rows with a T_i value 5.73 % [5.49, 6.00]; ratio 0.73 [0.50, 0.98].

## H. The published time course of discharge 31923

Grid of the file: 3.512 to 7.992 s. Absent frames between 3.0 and 5.5 s (stamps, s): 5.002. Steps other than 10 ms in that range (time of the row before, step in s): 4.992, 0.020.

Mean T_i 3.60-3.95 s: 395.7 [386.0, 404.5] eV (35 of 35 rows measured, 4 blocks); 4.10-4.90 s: 410.5 [405.0, 415.3] eV (80 of 80 rows, 8 blocks). Intervals of the means: the same block resampling.

| 100 ms from (s) | rows | measured | mean T_i (eV) | sd (eV) |
|---|---|---|---|---|
| 3.5 | 9 | 9 | 266 | 103 |
| 3.6 | 10 | 10 | 398 | 49 |
| 3.7 | 10 | 10 | 408 | 34 |
| 3.8 | 10 | 10 | 388 | 31 |
| 3.9 | 10 | 10 | 393 | 29 |
| 4.0 | 10 | 10 | 419 | 31 |
| 4.1 | 10 | 10 | 419 | 24 |
| 4.2 | 10 | 10 | 417 | 45 |
| 4.3 | 10 | 10 | 412 | 19 |
| 4.4 | 10 | 10 | 416 | 23 |
| 4.5 | 10 | 10 | 403 | 38 |
| 4.6 | 10 | 10 | 396 | 29 |
| 4.7 | 10 | 10 | 416 | 37 |
| 4.8 | 10 | 10 | 405 | 47 |
| 4.9 | 10 | 10 | 401 | 24 |

Discharge 31921, rows within 50 ms of the published profile times:

| time (s) | rows | measured | mean T_i (eV) | sd (eV) | values above 3 keV removed |
|---|---|---|---|---|---|
| 5.40 | 10 | 10 | 396 | 46 | 0 |
| 8.05 | 9 | 6 | 735 | 401 | 0 |

## D. The fall of the estimate in panel (c) of the trace figure (descriptive)

Discharge s31343.csv, window 5.722-6.382 s (fig_trace.json: 5.722-6.382), measured range 765.7-1029.5 eV. Estimator: recurrent estimator of the figure (data/.b1_seqv2_s1_i1, whole-discharge standardization), CPU. Check against the stored errors of the frozen evaluation (393 scored rows): largest deviation of the absolute error 1.34e-01 eV (network), passed: True. Lowest estimate in the window 654.1 eV.

| time (s) | estimate (eV) | measured at the row (eV) | previous measured (eV, s) | next measured (eV, s) | ms since the last absent frame | mean BES departure (sd, B.32) | rows since blip (B.32) | ECEI / mean of 100 ms before |
|---|---|---|---|---|---|---|---|---|
| 5.762 | 654.1 | 789.0 | 1023.0, 5.742 | 858.8, 5.772 | 10 | -5.05 | 1 | 0.941 |

Context: absent frames from 0.2 s before the window to its end: 5.752; quantiles 0.1/0.5/0.9 of the mean BES departure over the window rows -0.40 / -0.01 / +0.51 sd.

Caption sentence: 1 row(s) at 5.762 s with estimates 654 eV (measured range 766 to 1029 eV); time since the last absent frame 10 ms; mean BES departure -5.05 sd; ECEI level over the 100 ms before 0.941.

## After the fact (not registered; not used for any verdict)

1. Part S, 200 ms blocks, ECEI: pairs of used blocks per discharge, quantiles 0.1/0.5/0.9 2 / 13 / 32; discharges with at least 10, 20, 40 pairs: 400, 236, 15. The registered minimum of 40 pairs leaves few discharges; the same read-out with lower minima:

| min pairs | discharges | median rho | median null | difference | share rho > 0 |
|---|---|---|---|---|---|
| 10 | 400 | +0.322 [+0.280, +0.353] | -0.006 [-0.030, +0.016] | +0.328 [+0.275, +0.370] | 0.800 |
| 20 | 236 | +0.306 [+0.253, +0.344] | -0.006 [-0.035, +0.018] | +0.312 [+0.258, +0.364] | 0.814 |

## Checks before the read-out

- 641 files, 9 BES and 4 ECEI channels; every file opens at t = 0 and closes at t = 30 s, and these two rows are dropped in every tally.
- Missing T_i by fifth of the BES level, file_rules.md definition (per-discharge z-score, 245,923 rows): lowest 17.96 %, highest 3.37 % (to reproduce 18.0 % and 3.4 % within 0.5 points): reproduced. All five: 18.0, 8.2, 5.3, 3.6, 3.4 %.
- Part F covers 641 files (245,925 grid rows; 638 with at least one T_i value); raw and held-free grids misaligned in 0 files. The B.32 reader (kstar_file, None for files under 12 rows) was compared on 613 files: times, rows-after-blip and missing T_i differ in 0.
- Part D: CPU recomputation against the stored squared errors of the frozen evaluation for s31343.csv (393 rows): network max deviation 1.34e-01 eV, persistence and causal GP share within 1e-3 1.000 / 1.000; window and measured range equal to fig_trace.json: True.

## Expectations of the registration (section 5)

| expectation | observed | met |
|---|---|---|
| S1: median correlation +0.15 to +0.40 | +0.268 | yes |
| S1: null within 0.03 of zero | +0.003 | yes |
| S1 holds | holds | yes |
| F1: ratio 2 to 8 | 3.18 | yes |
| F1 holds | holds | yes |
| H1: both blips found | near 3.5 s: none (file begins at 3.512 s); near 5.0 s: 5.002 | no |
| H1: ratio 1.05 to 1.40 | 1.037 | no |
| H1 holds | does not hold | no |
| D: the rows follow an absent frame by 10 to 30 ms | 10 ms | yes |

6 of 9 expectations met.

## Deviations and choices left open by the registration

- Choice fixed before the run: part S uses only whole blocks; the trailing partial block of a segment is dropped. Block means of T_i are over the measured rows of the block, block means of the ECEI and BES levels over all its rows. The pairs of a discharge are taken over all its segments in time order, and the null shifts that vector circularly by floor(n/2).
- Choice fixed before the run: the 100 ms and 500 ms versions (reported, no statement) use blocks of 10 and 50 rows, at least half the rows measured, and at least 80 and 16 pairs (the same 8 s of data per discharge as 40 pairs of 200 ms).
- Choice fixed before the run: thirds of the discharge median T_i are taken over the discharges that enter the S read-out (median over the measured T_i rows of the cut grid).
- Choice fixed before the run: in part F 'missing' for the check is an empty CES_TI cell in the file (held values not counted as missing), over all grid rows. The 18.0 % and 3.4 % of file_rules.md were computed with the BES level standardized per discharge (z-score), not divided by the discharge median; the check is made with that definition, and the missing shares under the B.40 definition are reported beside the shares above 3 keV.
- Choice fixed before the run: 'within ten rows after an absent frame' is the B.32 rule (rows 1 to 10 after a step of more than 15 and less than 45 ms, inside the segment); the same share over all rows with a T_i value is given as the reference.
- Choice fixed before the run: part H windows include their end stamps (3.60 <= t <= 3.95 s, 4.10 <= t <= 4.90 s); 100 ms blocks start at the window start (the early window has a last block of 50 ms); blocks are resampled with replacement within each window independently, and a resampled mean is the sum of T_i over the measured rows of the drawn blocks over their count. Values above 3 keV are removed (cut grid).
- Correction after the first run (no definition, threshold or population changed): the first run took the rows of part F from the B.32 reader, which returns nothing for the 28 files with fewer than 12 grid rows, so 613 files (245,727 rows) entered. The registration says all grid rows, so the run was repeated with the same B.32 rule applied to all 641 files (checked identical to the reader on the 613). F1 changed from 3.18 [2.51, 4.03] to 3.18 [2.49, 4.04]; the missing-share check from 17.95 / 3.37 % to 17.96 / 3.37 %. Part D's expectation check now rounds the time since the absent frame to 0.1 ms (the first run read 9.9999 ms as below 10).
- Observation on H1 (verdict kept as registered): the 10 ms file of 31923 begins at 3.512 s, so no absent frame can lie before or at 3.5 s in it; the published blip at 3.5 s coincides with the start of the delivered grid (12 ms after 3.5 s). The rule was applied as written and H1 does not hold on that clause; the clause near 5.0 s and the ratio interval are met.
- Deviation: the registration names 'the estimate of the causal variant' in panel (c). The figure (fig_trace.py and the caption) draws the recurrent estimator with whole-discharge standardization (data/.b1_seqv2_s1_i1), not the causal variant. Part D describes the estimate that is drawn, recomputed on the CPU with the figure's own functions and checked against the stored errors; the causal variant was not run.

Run time: 34 s.
