# B.61 part C: dose of a beam interruption

Registration: `ces_prediction/experiments/PREREGISTRATION_B61.md` (part C only; parts A and B are run separately). Runner: `ces_prediction/experiments/b61_blip_dose/run_b61c.py`, event and ratio code imported unchanged from `ces_prediction/experiments/b58_blip_calibration/uc_blip_shape.py`. CPU only, no model, all 641 files, cut population. Ratios are event medians of the value over the median of the six rows before. Intervals: 95 %, 2,000 resamples of the discharges that have any event; every class and every difference is computed within the same draw.

## Check before any statement

First-row ratios of events with one absent frame, against B.58: T_i 0.814 (B.58 0.814), ECEI level 0.929 (B.58 0.929), BES level 0.658 (B.58 0.658). Reproduced exactly: True.

## Statements

| statement | difference of the first-row deficit, two minus one absent frame | verdict | expected |
|---|---|---|---|
| C1 T_i deficit larger after two absent frames than after one | -0.112 [-0.238, +0.054] | **does not hold** | uncertain, expected to fail |
| C2 the same for the ECEI level | +0.047 [-0.003, +0.098] | **does not hold** | uncertain |

## Per class

| absent frames | events (discharges) | quantity | first-row ratio | mean ratio rows 1-3 |
|---|---|---|---|---|
| 1 | 1262 (489) | T_i | 0.814 [0.790, 0.835] | 0.838 [0.821, 0.853] |
| 1 | 1262 (489) | ECEI level | 0.929 [0.918, 0.941] | 0.930 [0.919, 0.941] |
| 1 | 1262 (489) | BES level | 0.658 [0.653, 0.663] | 0.820 [0.816, 0.825] |
| 2 | 67 (29) | T_i | 0.926 [0.768, 1.049] | 0.891 [0.772, 1.029] |
| 2 | 67 (29) | ECEI level | 0.882 [0.831, 0.932] | 0.909 [0.871, 0.953] |
| 2 | 67 (29) | BES level | 0.461 [0.438, 0.482] | 0.815 [0.801, 0.823] |
| 3 | 65 (33) | T_i | 0.830 [0.779, 0.963] | 0.836 [0.789, 0.913] |
| 3 | 65 (33) | ECEI level | 0.926 [0.804, 0.991] | 0.926 [0.831, 0.978] |
| 3 | 65 (33) | BES level | 0.601 [0.565, 0.707] | 0.816 [0.789, 0.862] |

## Differences of the deficit (one minus the ratio), within a draw (reported)

| difference | quantity | first row | mean of rows 1-3 |
|---|---|---|---|
| 2 minus 1 | T_i | -0.112 [-0.238, +0.054] | -0.054 [-0.199, +0.066] |
| 2 minus 1 | ECEI level | +0.047 [-0.003, +0.098] | +0.021 [-0.024, +0.060] |
| 2 minus 1 | BES level | +0.197 [+0.177, +0.220] | +0.006 [-0.004, +0.020] |
| 3 minus 1 | T_i | -0.016 [-0.157, +0.040] | +0.001 [-0.077, +0.050] |
| 3 minus 1 | ECEI level | +0.003 [-0.062, +0.125] | +0.004 [-0.048, +0.095] |
| 3 minus 1 | BES level | +0.057 [-0.049, +0.094] | +0.005 [-0.041, +0.031] |

## Event-median profiles, rows 1 to 10 (ratio to the median of the six rows before)

| row | T_i, 1 frame | T_i, 2 frames | T_i, 3 frames | ECEI level, 1 frame | ECEI level, 2 frames | ECEI level, 3 frames | BES level, 1 frame | BES level, 2 frames | BES level, 3 frames |
|---|---|---|---|---|---|---|---|---|---|
| +1 | 0.814 | 0.926 | 0.830 | 0.929 | 0.882 | 0.926 | 0.658 | 0.461 | 0.601 |
| +2 | 0.841 | 0.856 | 0.817 | 0.923 | 0.910 | 0.915 | 0.872 | 0.966 | 0.917 |
| +3 | 0.858 | 0.892 | 0.862 | 0.938 | 0.934 | 0.936 | 0.931 | 1.016 | 0.930 |
| +4 | 0.877 | 0.739 | 0.898 | 0.944 | 0.979 | 0.971 | 0.951 | 1.010 | 0.956 |
| +5 | 0.915 | 1.071 | 0.924 | 0.961 | 1.024 | 0.996 | 0.977 | 1.016 | 1.002 |
| +6 | 0.936 | 1.524 | 0.963 | 0.976 | 1.057 | 1.000 | 0.995 | 1.015 | 1.009 |
| +7 | 0.962 | 2.046 | 0.992 | 0.986 | 1.071 | 1.029 | 1.013 | 1.005 | 1.015 |
| +8 | 0.977 | 1.906 | 1.008 | 0.991 | 1.079 | 1.056 | 1.016 | 1.007 | 1.053 |
| +9 | 0.986 | 1.768 | 1.028 | 1.000 | 1.084 | 1.058 | 1.020 | 1.007 | 1.040 |
| +10 | 0.996 | 1.185 | 1.002 | 1.000 | 1.066 | 1.062 | 1.029 | 1.008 | 1.036 |

## Arithmetic, not a statement: how long the beam is off in each class

From the raw BES of the twelve discharges with raw streams (B.58 add-on: 0.1 ms level below half of its 200 ms running median for at least 3 ms, matched to the absent frames).

| absent frames | blips in the raw windows | beam off, median (ms) | range (ms) |
|---|---|---|---|
| 1 | 41 | 10.5 | 6.4 to 11.0 |
| 2 | 1 | 27.2 | 27.2 to 27.2 |
| 3 | 3 | 26.7 | 26.4 to 26.7 |

Interpretations: the first-row ratio and the mean of rows 1-3 are read from the event-median profile (the B.58 statistic `stat`), so 'mean ratio of rows 1 to 3' is the mean of three event medians; the bootstrap draws discharges once from all discharges with an event of any class and computes every class from that draw (a draw with no event of a class is skipped for that class).

Run time 24 s.
