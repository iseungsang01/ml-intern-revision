# Departure of the fast channels from their 100 ms mean, and the T_i change that goes with it

Descriptive, all 641 files (`ces_prediction/analyze_fast_departure.py`); no model. A channel is standardized per discharge; its departure is its value minus its mean over the preceding ten rows of the same segment. Intervals: 95 %, 2,000 resamples of discharges.

## 1. How far the channels depart, by row class

| rows | BES: median departure (sd) | BES: share beyond 2 sd | BES: share beyond 3 sd | BES: share below -3 sd | ECEI: median departure (sd) | ECEI: share beyond 3 sd |
|---|---|---|---|---|---|---|
| first row after a blip (12,474 BES channel values) | -2.32 | 0.581 [0.540, 0.624] | 0.357 [0.319, 0.396] | 0.355 [0.318, 0.394] | -0.32 | 0.007 [0.004, 0.011] |
| rows 2 to 4 after a blip (37,476 BES channel values) | -0.20 | 0.047 [0.037, 0.057] | 0.010 [0.007, 0.013] | 0.007 [0.004, 0.010] | -0.20 | 0.004 [0.002, 0.006] |
| rows 5 to 10 after a blip (75,303 BES channel values) | +0.45 | 0.033 [0.028, 0.038] | 0.004 [0.003, 0.006] | 0.001 [0.000, 0.002] | +0.07 | 0.003 [0.002, 0.004] |
| all other rows (2,028,195 BES channel values) | -0.00 | 0.014 [0.013, 0.015] | 0.004 [0.003, 0.004] | 0.001 [0.001, 0.002] | -0.00 | 0.003 [0.003, 0.004] |

## 2. Response curve without a model (rows not within ten rows after a blip)

Change of the measured T_i from its last observed value, by the mean departure of the nine BES channels of the row. Values above 3 keV are left out.

| mean BES departure (sd) | rows | mean departure in the bin (sd) | mean T_i change (eV) |
|---|---|---|---|
| -9.0 to -3.0 | 187 | -4.59 | -54.0 [-110.1, +5.0] |
| -3.0 to -2.0 | 487 | -2.32 | -109.4 [-146.5, -74.0] |
| -2.0 to -1.0 | 5,966 | -1.31 | -45.9 [-58.8, -32.1] |
| -1.0 to -0.5 | 17,908 | -0.69 | -33.9 [-41.5, -27.2] |
| -0.5 to -0.2 | 33,440 | -0.33 | -20.7 [-24.5, -17.1] |
| -0.2 to +0.2 | 88,381 | -0.00 | -0.2 [-1.3, +0.9] |
| +0.2 to +0.5 | 32,653 | +0.33 | +20.8 [+17.3, +24.6] |
| +0.5 to +1.0 | 18,763 | +0.70 | +36.7 [+31.3, +42.6] |
| +1.0 to +2.0 | 6,942 | +1.32 | +42.4 [+32.9, +51.2] |
| +2.0 to +3.0 | 979 | +2.37 | +32.9 [+9.0, +56.0] |
| +3.0 to +9.0 | 299 | +4.18 | +65.3 [-15.5, +151.7] |
