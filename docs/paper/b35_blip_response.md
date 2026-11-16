# B.35 — the beam blip as a repeated perturbation (registered: PREREGISTRATION_B35.md)

CPU, the 10 ms files, no model. 1,366 blips of 498 discharges with T_i observed in at least three of the six rows before: 1,235 with one absent frame, 131 with two or three. Ratio = value over the median of the six rows before; integral recovery time = 10 ms x sum of the deficit over rows 1 to 15 / deficit of row 1 (median traces). Intervals: 95 %, 2,000 resamples of discharges.

- **N1** (dose: the T_i ratio in the first row is lower after two or three absent frames than after one): 0.877 [0.789, 0.976] against 0.814 [0.790, 0.833], difference +0.063 [-0.025, +0.169] -> does not hold.
- **N2** (time scale: the integral recovery time of T_i is longer in the highest third of the level before than in the lowest; thirds at 514 and 822 eV, 412 and 412 events): 59.2 [50.2, 75.5] ms against 28.2 [13.7, 40.6] ms, difference +31.0 [+16.1, +51.8] ms -> holds.

## N3 (reported)

| quantity | T_i | ECEI level |
|---|---|---|
| ratio in the first row, one absent frame | 0.814 [0.790, 0.833] | 0.929 [0.918, 0.943] |
| ratio in the first row, two or three | 0.877 [0.789, 0.976] | 0.895 [0.860, 0.948] |
| difference (dose) | +0.063 [-0.025, +0.169] | -0.034 [-0.070, +0.018] |
| dose difference within the lowest third | +0.135 [-0.025, +0.291] | -0.053 [-0.087, +0.005] |
| dose difference within the highest third | +0.108 [-0.090, +0.184] | +0.010 [-0.068, +0.090] |
| ratio in the first row, lowest third | 0.790 [0.756, 0.840] | 0.949 [0.928, 0.961] |
| ratio in the first row, highest third | 0.845 [0.814, 0.887] | 0.945 [0.924, 0.969] |
| integral recovery time, all events (ms) | 41.0 [35.7, 46.8] | 40.2 [31.0, 49.2] |
| integral recovery time, lowest third (ms) | 28.2 [13.7, 40.6] | 22.8 [-1.1, 42.3] |
| integral recovery time, highest third (ms) | 59.2 [50.2, 75.5] | 76.0 [56.6, 131.8] |
| highest minus lowest (ms) | +31.0 [+16.1, +51.8] | +53.2 [+26.0, +115.9] |

## Median traces (ratio by row after the blip)

| group | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| T_i, one absent frame | 0.814 | 0.841 | 0.858 | 0.877 | 0.915 | 0.936 | 0.962 | 0.977 | 0.986 | 0.996 | 1.006 | 1.014 | 1.016 | 1.013 | 1.011 |
| T_i, two or three | 0.877 | 0.819 | 0.865 | 0.892 | 0.992 | 1.038 | 1.107 | 1.225 | 1.130 | 1.084 | 1.108 | 0.989 | 1.002 | 1.007 | 0.976 |
| T_i, lowest third of the level | 0.790 | 0.797 | 0.801 | 0.849 | 0.885 | 0.934 | 0.974 | 1.004 | 1.022 | 1.051 | 1.047 | 1.054 | 1.073 | 1.066 | 1.062 |
| T_i, highest third | 0.845 | 0.882 | 0.899 | 0.900 | 0.919 | 0.924 | 0.937 | 0.949 | 0.955 | 0.973 | 0.968 | 0.985 | 0.983 | 0.977 | 0.986 |
| ECEI, one absent frame | 0.929 | 0.924 | 0.938 | 0.942 | 0.961 | 0.975 | 0.985 | 0.991 | 1.000 | 1.000 | 1.001 | 1.008 | 1.013 | 1.011 | 1.000 |
| ECEI, two or three | 0.895 | 0.910 | 0.935 | 0.973 | 1.000 | 1.017 | 1.053 | 1.058 | 1.068 | 1.065 | 1.023 | 1.015 | 1.012 | 1.000 | 1.020 |
