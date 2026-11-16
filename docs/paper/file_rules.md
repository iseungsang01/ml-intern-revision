# How the 10 ms files relate to the raw signals

Descriptive, read from the files (`ces_prediction/analyze_file_rules.py`); no model is involved. Intervals are 95 % intervals over 2,000 resamples of discharges.

## 1. Grid

- 641 files, 247,207 rows as delivered. Every file opens with a row at t = 0 (641 files) and closes with a row at t = 30 s (641 files) that carry no CES value; the provider's note says to ignore them. Without them the grid has **245,925 rows** in 698 segments (breaks of at least 0.5 s).
- Time stamp modulo 10 ms: 2 ms on 245,925 rows. Every time stamp lies 2 ms after a multiple of 10 ms.
- Steps between rows: 10 ms: 243,733; 20 ms: 1,266; 30 ms: 73; 40 ms: 75; 50 ms: 1; 60 ms: 3; 76 steps between 60 and 500 ms, 57 of 500 ms or more. 99.37 % of the steps are 10 ms. Steps of 20, 30 and 40 ms are one, two and three absent frames: 1,637 absent frames in 1,414 steps.
- Missing on this grid: T_i 7.70 %, V_rot 23.51 % (the earlier ledger divided by the rows as delivered and counted the two edge rows as missing: 8.18 % and 23.91 %).

## 2. T_i missing rate by quintile of the BES level

BES level = mean of the nine channels, standardised per discharge; 245,923 rows.

| quintile | missing | 95 % interval |
|---|---|---|
| 1 | 18.0 % | [15.9, 20.0] |
| 2 | 8.2 % | [6.9, 9.7] |
| 3 | 5.3 % | [4.4, 6.3] |
| 4 | 3.6 % | [2.9, 4.3] |
| 5 | 3.4 % | [2.7, 4.1] |

## 3. Fast-channel values of a row against the raw sample at its time stamp

Share of rows whose values, all channels of the diagnostic at once, equal a raw sample within one raw sample of the time stamp, at the five decimals the files print.

| discharge | BES (rate, rows, share) | ECEI | Mirnov |
|---|---|---|---|
| 31097 | 500 kHz, 794, 1.000 | 500 kHz, 794, 1.000 | 2000 kHz, 794, 1.000 |
| 31114 | 500 kHz, 504, 1.000 | 500 kHz, 504, 1.000 | 2000 kHz, 504, 1.000 |
| 31357 | 500 kHz, 394, 1.000 | 500 kHz, 394, 1.000 | 2000 kHz, 394, 1.000 |
| 31359 | 500 kHz, 295, 1.000 | 500 kHz, 295, 1.000 | 2000 kHz, 295, 1.000 |
| 31747 | 1000 kHz, 543, 1.000 | 500 kHz, 543, 1.000 | 2000 kHz, 543, 0.980 |
| 31873 | 500 kHz, 760, 1.000 | 500 kHz, 742, 1.000 | 2000 kHz, 198, 1.000 |
| 31902 | 500 kHz, 583, 1.000 | 500 kHz, 583, 1.000 | 2000 kHz, 583, 1.000 |
| 31921 | 500 kHz, 506, 1.000 | 500 kHz, 506, 1.000 | 2000 kHz, 506, 1.000 |
| 31923 | 500 kHz, 445, 1.000 | 500 kHz, 445, 1.000 | 2000 kHz, 445, 1.000 |
| 32027 | 500 kHz, 394, 1.000 | 500 kHz, 394, 1.000 | 2000 kHz, 394, 1.000 |
| 32092 | 500 kHz, 643, 1.000 | 500 kHz, 643, 1.000 | 2000 kHz, 643, 0.992 |
| 32097 | 500 kHz, 641, 1.000 | 500 kHz, 641, 1.000 | 2000 kHz, 641, 0.989 |

All discharges: BES 6,502 of 6,502 rows (100.0 %), ECEI 6,484 of 6,484 rows (100.0 %), MC 5,917 of 5,940 rows (99.6 %). The fast channels of a row are single raw samples taken at the time stamp, not averages over the CES exposure.

## 4. Beam blips in the raw BES against the absent frames

A blip is a stretch of at least 3 ms in which the mean BES level (0.1 ms bins) is below half of its 200 ms running median.

- Absent frames inside the raw windows: 45 groups (45 with a blip, 0 without).
- Single absent frames (41): the blip starts median -1.7 ms (range -1.9 to +2.4) from the absent time stamp and ends median +8.8 ms (range +8.1 to +9.3) from it; duration median 10.5 ms (range 6.4 to 11.0).
- Groups of two or three absent frames (4): start median +0.0 ms (range -4.8 to +2.8) from the first absent stamp, end median +8.0 ms (range +6.7 to +12.4) from the last; duration median 26.7 ms (range 26.4 to 27.2).
- The frame that is absent is the one exposed without beam, so the time stamp of a row lies within about 2 ms of the start of its 10 ms exposure.

Beam interruptions that removed no frame (rows whose exposure, stamp - 2 ms to stamp + 8 ms, lies inside the interruption):

| discharge | interruption (s) | duration | T_i in rows exposed without beam (eV) | median T_i in the 70 ms before |
|---|---|---|---|---|
| 31873 | 13.0002 to 13.0095 | 9.3 ms | (no row) | 591 |
| 31873 | 13.0115 to 13.0205 | 9.0 ms | (no row) | 566 |
| 32027 | 5.1235 to 5.1492 | 25.7 ms | 979 | 1,063 |
| 32027 | 5.3583 to 5.3815 | 23.2 ms | 1,711 / 1,133 | 1,195 |

## 5. Rows after an absent frame, all files

1,394 absent-frame events with six contiguous rows before them, in 508 discharges (by number of absent frames: 1: 1,262, 2: 67, 3: 65). Each value is the ratio to the median of the six rows before the event; V_rot uses values that differ from the previous row only.

| row | T_i | ECEI level | fresh V_rot | BES level | T_i observed |
|---|---|---|---|---|---|
| -6 | 0.997 [0.992, 1.000] | 1.000 [0.998, 1.000] | 0.997 [0.991, 1.000] | 0.999 [0.997, 1.000] | 0.956 [0.944, 0.967] |
| -5 | 0.999 [0.995, 1.000] | 0.999 [0.996, 1.000] | 1.002 [0.998, 1.006] | 0.998 [0.997, 1.000] | 0.958 [0.946, 0.968] |
| -4 | 1.000 [0.997, 1.003] | 1.000 [1.000, 1.002] | 1.000 [0.997, 1.004] | 1.000 [0.999, 1.002] | 0.959 [0.948, 0.970] |
| -3 | 1.001 [0.999, 1.004] | 1.000 [1.000, 1.002] | 1.000 [0.995, 1.003] | 1.001 [0.999, 1.002] | 0.963 [0.952, 0.974] |
| -2 | 1.002 [1.000, 1.007] | 1.000 [0.998, 1.000] | 1.000 [0.996, 1.004] | 1.001 [0.999, 1.002] | 0.964 [0.954, 0.974] |
| -1 | 1.001 [1.000, 1.006] | 1.000 [1.000, 1.004] | 1.002 [0.997, 1.006] | 1.001 [0.999, 1.003] | 0.963 [0.953, 0.973] |
| +1 | 0.819 [0.796, 0.838] | 0.926 [0.910, 0.939] | 0.975 [0.964, 0.990] | 0.656 [0.649, 0.660] | 0.874 [0.851, 0.893] |
| +2 | 0.841 [0.826, 0.857] | 0.923 [0.911, 0.936] | 0.977 [0.965, 0.990] | 0.876 [0.870, 0.882] | 0.897 [0.878, 0.914] |
| +3 | 0.858 [0.841, 0.878] | 0.937 [0.924, 0.948] | 0.981 [0.968, 0.995] | 0.934 [0.928, 0.940] | 0.915 [0.898, 0.930] |
| +4 | 0.877 [0.866, 0.892] | 0.947 [0.936, 0.958] | 0.987 [0.976, 0.998] | 0.954 [0.949, 0.960] | 0.923 [0.906, 0.937] |
| +5 | 0.919 [0.904, 0.932] | 0.966 [0.958, 0.976] | 0.984 [0.974, 0.995] | 0.981 [0.976, 0.986] | 0.934 [0.919, 0.947] |
| +6 | 0.941 [0.929, 0.954] | 0.980 [0.970, 0.988] | 0.990 [0.979, 1.000] | 0.996 [0.992, 1.000] | 0.945 [0.931, 0.957] |
| +7 | 0.965 [0.954, 0.975] | 0.992 [0.983, 1.000] | 0.993 [0.980, 1.006] | 1.013 [1.008, 1.017] | 0.956 [0.945, 0.966] |
| +8 | 0.983 [0.971, 0.994] | 0.998 [0.991, 1.002] | 0.997 [0.984, 1.013] | 1.017 [1.013, 1.021] | 0.956 [0.943, 0.967] |
| +9 | 0.993 [0.980, 1.003] | 1.003 [1.000, 1.012] | 0.997 [0.985, 1.010] | 1.019 [1.015, 1.025] | 0.961 [0.949, 0.971] |
| +10 | 0.999 [0.991, 1.009] | 1.003 [0.997, 1.012] | 1.000 [0.990, 1.014] | 1.027 [1.022, 1.032] | 0.960 [0.948, 0.970] |
| +11 | 1.009 [0.998, 1.022] | 1.003 [0.999, 1.013] | 0.996 [0.983, 1.013] | 1.028 [1.021, 1.033] | 0.961 [0.949, 0.971] |
| +12 | 1.014 [1.004, 1.024] | 1.008 [1.001, 1.016] | 1.007 [0.996, 1.016] | 1.031 [1.026, 1.035] | 0.952 [0.939, 0.964] |
| +13 | 1.016 [1.005, 1.027] | 1.012 [1.004, 1.022] | 1.004 [0.995, 1.022] | 1.029 [1.024, 1.034] | 0.954 [0.940, 0.965] |
| +14 | 1.014 [1.002, 1.025] | 1.011 [1.002, 1.018] | 1.007 [0.993, 1.019] | 1.030 [1.024, 1.033] | 0.958 [0.947, 0.968] |
| +15 | 1.008 [0.996, 1.020] | 1.000 [0.996, 1.008] | 1.006 [0.991, 1.019] | 1.027 [1.021, 1.032] | 0.956 [0.944, 0.968] |

- Mean deficit over rows +1 to +6: T_i 0.124 [0.112, 0.137], ECEI 0.053 [0.045, 0.062]; ratio 2.33 [1.99, 2.75].
- Across events, rank correlation of the T_i ratio and the ECEI ratio (means of rows +1 to +4): +0.251 [+0.199, +0.304] (1,352 events).
- T_i ratio (rows +1 to +4) by ECEI ratio: 0.00-0.85: 0.799 (n 420); 0.85-0.95: 0.816 (n 303); 0.95-1.05: 0.903 (n 405); 1.05-99.00: 0.949 (n 216).
- By T_i before the event, rows +1/+2/+3: at least 700 eV 0.836 / 0.876 / 0.900; below 400 eV 0.855 / 0.825 / 0.833.

## 6. Raw BES level around a blip (1 ms bins, ratio to the 50 ms before)

41 blips with a single absent frame.

| ms from the first ms without beam | median level | quartiles |
|---|---|---|
| -10 | 1.010 | 0.983 to 1.029 |
| +0 | 0.377 | 0.346 to 0.453 |
| +2 | 0.137 | 0.111 to 0.225 |
| +5 | 0.127 | 0.103 to 0.192 |
| +9 | 0.118 | 0.097 to 0.193 |
| +10 | 0.407 | 0.393 to 0.446 |
| +11 | 0.604 | 0.577 to 0.658 |
| +12 | 0.693 | 0.673 to 0.757 |
| +13 | 0.740 | 0.717 to 0.806 |
| +15 | 0.732 | 0.708 to 0.770 |
| +17 | 0.762 | 0.725 to 0.801 |
| +20 | 0.870 | 0.837 to 0.916 |
| +25 | 0.881 | 0.841 to 0.965 |
| +30 | 0.957 | 0.900 to 1.009 |
| +40 | 0.978 | 0.934 to 1.016 |
| +50 | 0.984 | 0.939 to 1.049 |
| +60 | 0.984 | 0.960 to 1.045 |
| +80 | 1.010 | 0.982 to 1.058 |
| +100 | 1.021 | 0.993 to 1.046 |
| +150 | 1.041 | 1.005 to 1.087 |
