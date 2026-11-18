# B.62 What happens at the return of the beam

Registration: `ces_prediction/experiments/PREREGISTRATION_B62.md` with addendum 1. Runner: `ces_prediction/experiments/b62_beam_return/run_b62.py`. CPU only, no model, real data. Intervals: 95 %, 2,000 resamples of discharges; window values are event medians.

## Checks (before any statement)

- Part E: 41 interruptions of one absent frame (expected 41), median off-time 10.50 ms (expected 10.5): passed.
- Part E control (92 pseudo-times for ECEI, 120 for BES; no raw interruption and no absent frame within 200 ms, a 10.5 ms pseudo-interruption): largest deviation of a window median from 1: ECEI 0.0052, BES 0.0059 (limit 0.02): passed.
- Part X: first-row ratios of B.58 reproduced exactly (T_i, ECEI, BES 0.814, 0.929, 0.658); the rotation readout uses the same event set: True.

## Addendum 1: interruptions kept for the ECEI readout

ECEI readout: **31 interruptions in 10 discharges** (BES readout: 41 in 12). Excluded (mean of the 20 ms before not positive or below ten times the s.d. of its 1 ms bins):

| shot | beam off at (s) | ECEI mean before | s.d. of 1 ms bins |
|---|---|---|---|
| 31114 | 5.0303 | +0.52256 | 0.06400 |
| 31114 | 6.3003 | +0.49598 | 0.09148 |
| 31114 | 7.8003 | +0.54146 | 0.06617 |
| 31359 | 5.9905 | +0.21748 | 0.03050 |
| 31747 | 4.0003 | -0.00006 | 0.00025 |
| 31747 | 5.0003 | +0.00001 | 0.00030 |
| 31747 | 6.0003 | +0.00021 | 0.00029 |
| 31747 | 7.0002 | -0.00018 | 0.00029 |
| 31902 | 5.5003 | +0.85480 | 0.20770 |
| 31902 | 7.0003 | +0.75961 | 0.10329 |

With this many events the intervals are wide: E1 spans 0.046 in the ratio, E2 0.039 and E3 0.053 in the difference; a step of a few hundredths, the size expected, is near the resolution of this readout.

## Statements

| statement | finding | verdict | expected |
|---|---|---|---|
| E1 ECEI inside the interruption (last 3 ms before return) below 1 | 0.972 [0.953, 0.999] | **holds** | 0.97 to 1.01, fails |
| E2 ECEI step at the return: (0-3 ms after) minus (last 3 ms before) below 0 | -0.015 [-0.027, +0.012] | **does not hold** | -0.03 to -0.08, holds |
| E3 ECEI deepens: (10-13 ms after) minus (0-3 ms after) below 0 | -0.018 [-0.059, -0.006] | **holds** | -0.02 to +0.02, fails |
| X1 first-frame excess d above 0 | d = +0.030 [+0.008, +0.058] | **holds** | 0.01 to 0.04, holds or fails narrowly |
| V1 fresh rotation, first row, ratio excludes 1 | 0.973 [0.960, 0.989] | **holds** | 0.96 to 0.99, holds |

Differences are differences of event medians within a draw (the verdict); the median of the paired per-event differences is reported beside them.

### First run (implementation error, kept for the record)

The first run of this runner, read before the correction, averaged whole bins of a global 1 ms grid of the raw stream (bins starting at the raw start time) that fell inside each window, so a 3 ms window held two or three bins and was displaced by up to 1 ms from the event times; its 1 ms profiles kept only the events whose times fell on that grid. The run above anchors every window and every 1 ms bin at the beam-off and beam-return times of each event with sample resolution, which is what the registration describes. First-run values (ECEI, 31 interruptions): E1 0.969 [0.945, 1.001], E2 -0.010 [-0.027, +0.019], E3 -0.021 [-0.065, -0.007]; verdicts E1 does not hold, E2 does not hold, E3 holds, X1 holds, V1 holds.

## Part E, the four windows (event medians, ratio to the 20 ms before the beam goes off)

| window | ECEI (kept interruptions) | BES (all) | ECEI control | BES control |
|---|---|---|---|---|
| inside_last_3ms | 0.972 [0.953, 0.999] | 0.201 [0.186, 0.225] | 1.003 | 1.004 |
| return_0_3ms | 0.956 [0.929, 0.997] | 0.673 [0.657, 0.688] | 1.002 | 1.006 |
| return_10_13ms | 0.938 [0.879, 0.986] | 0.901 [0.860, 0.914] | 1.005 | 1.000 |
| return_20_23ms | 0.944 [0.888, 1.003] | 0.943 [0.909, 0.967] | 1.001 | 0.998 |
| E2_diff_of_medians | -0.015 [-0.027, +0.012] | +0.472 [+0.450, +0.488] | | |
| E2_median_paired | -0.014 [-0.021, -0.006] | +0.470 [+0.453, +0.497] | | |
| E3_diff_of_medians | -0.018 [-0.059, -0.006] | +0.228 [+0.187, +0.240] | | |
| E3_median_paired | -0.017 [-0.034, -0.006] | +0.222 [+0.199, +0.248] | | |

### 1 ms profiles aligned to the return of the beam (event medians; figure `docs/paper/b62_beam_return.png`)

| ms from return | ECEI | BES |
|---|---|---|
| -45 | 0.991 | 0.998 |
| -35 | 0.994 | 0.996 |
| -25 | 0.999 | 1.008 |
| -20 | 1.002 | 1.000 |
| -15 | 1.000 | 1.003 |
| -12 | 1.001 | 1.001 |
| -10 | 0.997 | 0.137 |
| -8 | 0.990 | 0.130 |
| -6 | 0.984 | 0.127 |
| -4 | 0.980 | 0.121 |
| -3 | 0.975 | 0.120 |
| -2 | 0.967 | 0.121 |
| -1 | 0.964 | 0.332 |
| 0 | 0.963 | 0.575 |
| 1 | 0.952 | 0.694 |
| 2 | 0.953 | 0.748 |
| 3 | 0.947 | 0.746 |
| 4 | 0.948 | 0.734 |
| 5 | 0.943 | 0.730 |
| 7 | 0.931 | 0.789 |
| 10 | 0.943 | 0.895 |
| 13 | 0.937 | 0.891 |
| 15 | 0.936 | 0.891 |
| 20 | 0.948 | 0.947 |
| 23 | 0.938 | 0.937 |
| 30 | 0.945 | 0.964 |
| 40 | 0.950 | 0.963 |
| 50 | 0.953 | 0.985 |
| 59 | 0.962 | 1.008 |

### 1 ms profiles aligned to the beam going off

| ms from off | ECEI | BES |
|---|---|---|
| -30 | 0.995 | 1.002 |
| -20 | 1.001 | 0.995 |
| -10 | 1.002 | 1.003 |
| -5 | 1.001 | 1.003 |
| -2 | 1.002 | 1.002 |
| -1 | 0.999 | 0.978 |
| 0 | 1.000 | 0.160 |
| 1 | 0.994 | 0.140 |
| 2 | 0.993 | 0.132 |
| 3 | 0.992 | 0.127 |
| 4 | 0.989 | 0.128 |
| 5 | 0.982 | 0.129 |
| 6 | 0.983 | 0.124 |
| 7 | 0.974 | 0.120 |
| 8 | 0.967 | 0.120 |
| 9 | 0.963 | 0.163 |
| 10 | 0.957 | 0.480 |
| 11 | 0.954 | 0.644 |

### Interruptions of two and three absent frames (raw; reported, not statements)

| shot | absent frames | off (ms) | ECEI inside / 0-3 / 10-13 / 20-23 | BES inside / 0-3 / 10-13 / 20-23 |
|---|---|---|---|---|
| 31902 | 3 | 26.7 | - / - / - / - | 0.350 / 0.604 / 0.923 / 0.970 |
| 31902 | 3 | 26.7 | - / - / - / - | 0.356 / 0.599 / 0.883 / 0.910 |
| 31902 | 3 | 26.4 | - / - / - / - | 0.325 / 0.535 / 0.910 / 0.898 |
| 32097 | 2 | 27.2 | 0.611 / 0.612 / 0.662 / 0.723 | 0.391 / 0.571 / 0.870 / 0.938 |

## Part X, first-frame excess (event-median T_i deficit, rows 1-8 = c x ECEI deficit + d x [row 1])

| absent frames | events (discharges) | c | d | d / first-row deficit | first-row deficit |
|---|---|---|---|---|---|
| 1 | 1262 (489) | 2.192 [1.864, 2.594] | +0.030 [+0.008, +0.058] | +0.160 [+0.046, +0.292] | 0.186 |
| 2 | 67 (29) | 7.237 [1.918, 11.033] | -0.778 [-1.323, -0.125] | -10.505 [-41.196, +132.781] | 0.074 |
| 3 | 65 (33) | 1.732 [0.462, 3.587] | +0.042 [-0.098, +0.168] | +0.246 [-2.159, +0.971] | 0.170 |

## Part V, fresh rotation values (ratio to the median of the fresh values of the six rows before)

| row | 1 absent frame | fresh values | 2 absent frames | 3 absent frames |
|---|---|---|---|---|
| +1 | 0.973 [0.960, 0.989] | 419 | 0.994 | 0.994 |
| +2 | 0.972 [0.958, 0.982] | 408 | 1.000 | 1.011 |
| +3 | 0.975 [0.964, 0.988] | 416 | 1.009 | 1.016 |
| +4 | 0.983 [0.970, 0.996] | 417 | 0.997 | 1.024 |
| +5 | 0.984 [0.973, 0.995] | 421 | 0.963 | 1.045 |
| +6 | 0.983 [0.977, 0.997] | 421 | 1.023 | 1.040 |
| +7 | 0.991 [0.975, 1.004] | 423 | 1.006 | 1.053 |
| +8 | 0.997 [0.984, 1.016] | 424 | 0.993 | 0.986 |
| +9 | 0.998 [0.985, 1.010] | 418 | 1.005 | 0.982 |
| +10 | 1.002 [0.988, 1.015] | 416 | 0.994 | 1.009 |

Events (discharges): 1 absent frame(s) 1262 (489), 2 absent frame(s) 67 (29), 3 absent frame(s) 65 (33). Shape index (rows 4-6 over row 1): 1: 0.608 [0.257, 1.404], 2: 0.855 [-7.174, 14.443], 3: -5.727 [-26.918, 18.900].

## Interpretations made in running the registration

1. Window values are event medians (as in B.58); E2 and E3 are differences of the event medians within a draw; the median of paired per-event differences is reported beside them.
2. Window edges and 1 ms bins are anchored at the B.58 beam-off and beam-return times of each interruption (0.1 ms resolution) and averaged over the raw samples with sample resolution; the reference is the 20 ms before the beam goes off; the s.d. of addendum 1 is that of the twenty 1 ms bins of that window.
3. The figure aligns to the return of the beam (45 ms before to 60 ms after it) and shades the median off interval; a profile aligned to the beam going off is tabulated.
4. Control pseudo-times: up to ten per discharge drawn uniformly inside the raw window with no raw interruption and no step above 15 ms of the 10 ms file within 200 ms; a pseudo-interruption of 10.5 ms; the ECEI rule of addendum 1 applies to them too.
5. The rotation readout repeats the event rule of `uc_blip_shape.py` in this runner (that script does not compute the rotation) and is checked to give the identical event set and identical T_i ratios; fresh = differs from the previous row and is finite; the level is the median of the fresh values among the six rows before, used if its magnitude exceeds 5 km/s (the rule of `analyze_file_rules.py`).
6. V1 'differs from its level before' is read as the interval of the first-row ratio excluding 1.

Run time 35 s.
