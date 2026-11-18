# B.66 -- the increment by regime (edge-burst rate)

Registration: `ces_prediction/experiments/PREREGISTRATION_B66.md`. CPU only, stored errors. Script: `ces_prediction/experiments/b66_regimes/run_b66.py`.

## Checks (before any error was read)

| check | result | passes |
|---|---|---|
| (a) detector run again = B.54 cached classes | 641/641 files identical | True |
| (b) quiet-row share, stored rows of split 42 | 0.8890 (recorded 0.8890) | True |
| (c) B.54 checks pass; fold classes = B.54; B.56 rows aligned (r3-r9) | 40 folds | True |
| (d) universe | 639 discharges, 28 without detectable rows | True |

## Class counts (fixed before any error was read)

Median of the non-zero burst rates over the 639 discharges: 0.719 onsets per second of detectable rows.

| class | discharges (share) | scored rows r2 | median T_i of discharges (eV) | median scored T_i r2 (eV) | rate range (/s) | rate median (/s) |
|---|---|---|---|---|---|---|
| A no burst | 251 (0.393) | 74,006 | 629 | 641 | 0.000 to 0.000 | 0.000 |
| B rate up to the median of the non-zero rates | 201 (0.315) | 84,700 | 607 | 603 | 0.073 to 0.719 | 0.360 |
| C rate above that median | 159 (0.249) | 33,076 | 335 | 399 | 0.738 to 10.526 | 1.439 |
| rate not defined (addendum 1; no class, no statement) | 28 | 152 | | | 4 to 16 grid rows | |

Discharges by class and temperature third (thirds of the 639 by median T_i):

| class | low | middle | high |
|---|---|---|---|
| A | 60 | 80 | 108 |
| B | 44 | 82 | 75 |
| C | 81 | 50 | 28 |

Scored rows by phase, re-partition r2 (classes B and C): burst 2,664, inter-burst 102,750, neither 12,362 of 117,776; class C alone: burst 1,748, inter-burst 26,729.

## Statements (r2 to r9 pooled, 20,000 draws)

| statement | value [95 %] | holds | expectation | per re-partition |
|---|---|---|---|---|
| **R1**: class A (no bursts): increment over the twin above zero | +0.056 [-0.008, +0.108] | **does not hold** | +0.00 to +0.04, uncertain | +0.039, +0.050, +0.077, +0.037, +0.085, +0.043, +0.051, +0.063 |
| **R2**: class C (frequent bursts): increment over the twin above zero | +0.025 [-0.009, +0.052] | **does not hold** | +0.05 to +0.09, met | +0.022, +0.026, +0.042, -0.000, +0.036, +0.021, +0.030, +0.025 |
| **R3**: inter-burst phase of classes B and C: increment over the twin above zero | +0.050 [+0.027, +0.075] | **holds** | met | +0.050, +0.058, +0.062, +0.036, +0.059, +0.040, +0.048, +0.049 |
| **R4**: increment in class A minus class C: interval excludes zero | +0.030 [-0.028, +0.082] | **does not hold** | uncertain | +0.017, +0.024, +0.035, +0.037, +0.049, +0.021, +0.022, +0.038 |

## By class and phase

| rows | discharges (mean) | rows (r2) | median scored T_i (eV) | over twin | over causal GP | over hold | KFX over KF0 (r3-r9) |
|---|---|---|---|---|---|---|---|
| all scored rows | 634 | 191,934 | 593 | +0.048 [+0.022, +0.073] | +0.089 [+0.063, +0.112] | +0.375 [+0.352, +0.395] | +0.049 [+0.041, +0.059] |
| class A no burst | 248 | 74,006 | 641 | +0.056 [-0.008, +0.108] | +0.060 [-0.006, +0.113] | +0.288 [+0.225, +0.334] | +0.078 [+0.058, +0.100] |
| class B rate up to the median of the non-zero rates | 201 | 84,700 | 606 | +0.072 [+0.038, +0.105] | +0.109 [+0.070, +0.146] | +0.382 [+0.347, +0.415] | +0.069 [+0.054, +0.084] |
| class C rate above that median | 159 | 33,076 | 401 | +0.025 [-0.009, +0.052] | +0.091 [+0.059, +0.119] | +0.411 [+0.385, +0.434] | +0.018 [+0.007, +0.029] |
| classes B and C | 360 | 117,776 | 559 | +0.046 [+0.023, +0.068] | +0.098 [+0.074, +0.122] | +0.399 [+0.377, +0.419] | +0.040 [+0.030, +0.049] |
| burst phase, B and C | 348 | 2,664 | 456 | +0.029 [-0.027, +0.077] | +0.093 [+0.026, +0.154] | +0.408 [+0.343, +0.466] | +0.014 [-0.013, +0.040] |
| inter-burst phase, B and C | 359 | 102,750 | 569 | +0.050 [+0.027, +0.075] | +0.095 [+0.070, +0.120] | +0.410 [+0.386, +0.432] | +0.045 [+0.035, +0.055] |
| burst phase, B | 195 | 916 | 639 | +0.059 [-0.044, +0.146] | +0.137 [+0.001, +0.250] | +0.444 [+0.322, +0.531] | +0.060 [+0.000, +0.115] |
| inter-burst phase, B | 201 | 76,021 | 609 | +0.079 [+0.045, +0.113] | +0.118 [+0.081, +0.154] | +0.402 [+0.368, +0.433] | +0.070 [+0.055, +0.087] |
| burst phase, C | 153 | 1,748 | 382 | +0.019 [-0.055, +0.076] | +0.078 [-0.006, +0.152] | +0.396 [+0.317, +0.468] | -0.001 [-0.032, +0.027] |
| inter-burst phase, C | 158 | 26,729 | 408 | +0.026 [-0.010, +0.055] | +0.077 [+0.043, +0.106] | +0.416 [+0.386, +0.443] | +0.023 [+0.012, +0.035] |

Class A minus class C: over the twin +0.030 [-0.028, +0.082], over the causal GP -0.031 [-0.096, +0.025].

## Class by temperature third (increment over the twin; cells with at least 30 discharges)

| class | low | middle | high |
|---|---|---|---|
| A | -0.004 [-0.062, +0.053] (n 60) | +0.113 [+0.006, +0.220] (n 80) | +0.052 [-0.046, +0.120] (n 108) |
| B | +0.025 [-0.007, +0.058] (n 44) | +0.034 [-0.036, +0.095] (n 81) | +0.137 [+0.090, +0.184] (n 75) |
| C | +0.000 [-0.046, +0.030] (n 81) | +0.021 [-0.012, +0.049] (n 50) | n 28 (< 30) |
| all | +0.007 [-0.024, +0.031] (n 211) | +0.042 [+0.008, +0.075] (n 211) | +0.092 [+0.040, +0.137] (n 211) |

Over the causal GP in the same cells:

| class | low | middle | high |
|---|---|---|---|
| A | +0.050 [-0.054, +0.152] | +0.114 [+0.013, +0.202] | +0.046 [-0.052, +0.118] |
| B | +0.066 [+0.020, +0.113] | +0.060 [-0.039, +0.132] | +0.179 [+0.125, +0.234] |
| C | +0.069 [+0.020, +0.111] | +0.106 [+0.055, +0.153] | - |

## Expectations (registration section 6)

| key | expectation | observed | range met | verdict met |
|---|---|---|---|---|
| class A | about a third of the discharges and colder | 0.393 of 639; median T_i A 629, B 607, C 335 eV | False | - |
| R1 | +0.00 to +0.04, uncertain | +0.056 [-0.008, +0.108]; does not hold | False | None |
| R2 | +0.05 to +0.09, met | +0.025 [-0.009, +0.052]; does not hold | False | False |
| R3 | met | +0.050 [+0.027, +0.075]; holds | None | True |
| R4 | uncertain | +0.030 [-0.028, +0.082]; does not hold | None | None |

## Choices fixed in the implementation

- Detector: `run_b45.Classes.of_file` (the 10 ms BES departure rule that B.54 uses for its quiet rows; `posthoc_b32.channel_departures`: a BES channel more than 3 standard deviations from its mean over the ten rows before), run again on all 641 files and compared with B.54's cached classes. A burst row is an upward departure (class 3); rows after an absent frame (class 1) and low beam emission (class 2) take priority over class 3 in that code, which is the exclusion of beam blips of B.54. Downward departures (class 4) are not bursts (B.32: upward departures go with the raw-stream bursts, downward ones with dips of the beam).
- Burst onset: a class-3 grid row whose previous grid row in the same segment (steps < 0.5 s) is not an upward departure (an upward row in class 1 or 2 counts as upward, so an excursion that starts inside a blip window has no onset). The registration's 'the detector of B.31' is read as this 10 ms rule, since the raw-stream detector of B.31 exists for twelve discharges only and the registration says the detector is the one used by B.54 and B.58 for quiet rows.
- Burst rate: onsets per second of rows on which a burst can be detected (classes 3 to 6, 10 ms per row): rows without a defined departure (group 7) and blip rows (classes 1, 2) are not counted in the seconds.
- Universe: the 639 discharges of the fold manifests of the pooled re-partitions. Classes A (rate 0), B (0 < rate <= median of the non-zero rates of the 639), C (above), fixed before any error is read.
- Phases (classes B and C only, rows of classes 3 to 6): burst phase = 0 <= t - onset <= 30 ms for some onset (the onset row included); inter-burst = at least 50 ms from every onset, before or after. Rows 30 to 50 ms after, or less than 50 ms before, an onset are in neither phase.
- Class readouts: every scored row of the discharges of the class (no row excluded). Increment = 1 - SSE network / SSE twin; skill over the causal GP and over hold likewise; KFX over KF0 on r3 to r9 from data/.b56.
- Intervals: the two-level bootstrap of analyze_b39.arm_p (a re-partition drawn uniformly, then its discharges resampled with replacement; rows outside the cell contribute zero), 20,000 draws, one generator per quantity default_rng([20261166, crc32(name)]); point = mean over re-partitions. R4: the difference of the two skills under the same draws.
- Two-way table: thirds by the median measured T_i of a discharge within each re-partition (run_b44.split_thirds, as B.54); a cell's increment is computed when the cell has at least 30 discharges in every re-partition.
- Addendum 1: the 28 discharges whose files are too short for a defined departure (no row with ten rows of segment before) form the group 'rate not defined', in no class and no statement; thresholds from the other 611.
- Checks: (a) detector run again equals the B.54 cache in all files; (b) quiet-row share on the stored rows of split 42 equals 0.889 to three decimals; (c) B.54's checks pass, the classes of every pooled fold equal B.54's, B.56 rows of r3 to r9 identical to B.54's; (d) 639 discharges, each with detectable rows.

