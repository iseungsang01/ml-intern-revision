# B.32 — two sources of a large fast-channel excursion (registered: PREREGISTRATION_B32.md)

CPU, no model. Intervals: 95 %, 2,000 resamples of discharges (shots on MAST).

## A. KSTAR, all files: the first row after a beam blip, by the BES departure of that row

1,203 blips with T_i observed in the first row after and in at least three of the six rows before (449 discharges). d = mean over the nine BES channels of the departure from the 100 ms mean, in per-discharge standard deviations. Proportional reading = 50 eV per standard deviation.

| group | rows | mean d (sd) | median T_i ratio to the level before | mean T_i change (eV) | proportional reading (eV) | measured minus proportional (eV) |
|---|---|---|---|---|---|---|
| D1 (d <= -3) | 461 | -4.39 | 0.784 [0.755, 0.811] | -164.5 [-188.3, -141.3] | -219.3 | +54.8 [+28.7, +81.2] |
| D2 (-3 < d <= -2) | 257 | -2.48 | 0.786 [0.730, 0.820] | -142.7 [-182.5, -97.8] | -123.9 | -18.8 [-59.0, +25.8] |
| D3 (d > -2) | 485 | -1.16 | 0.886 [0.844, 0.917] | -61.6 [-102.3, -22.0] | -58.1 | -3.5 [-43.3, +34.9] |

- **A1** (the ratio of D1 is not below that of D3): D1 minus D3 = -0.103 [-0.147, -0.053] -> does not hold.
- **A2** (in D1 the measured change is above the proportional reading by at least 30 eV): +54.8 [+28.7, +81.2] eV -> holds.

## B. KSTAR, twelve discharges with a raw stream: departures beyond three standard deviations and bursts

Rows inside the raw window and not within ten rows after a blip: 5,939. Rows with a mean BES departure above +3: 10, of which 8 have a detected burst within 1 ms of the time stamp (share 0.800); base share over all rows 0.015.

- **B1**: not testable (fewer than 20 rows above +3).
- **B2** (reported): rows with a mean departure below -3: 11; with the raw BES level at the stamp below 0.8 of its 200 ms running median: 8 (share 0.727).

| discharge | rows | bursts | rows above +3 | of which with a burst | rows below -3 | of which with a dip |
|---|---|---|---|---|---|---|
| 31097 | 754 | 8 | 0 | 0 | 0 | 0 |
| 31114 | 464 | 16 | 0 | 0 | 0 | 0 |
| 31357 | 354 | 0 | 0 | 0 | 0 | 0 |
| 31359 | 265 | 51 | 1 | 1 | 0 | 0 |
| 31747 | 493 | 66 | 0 | 0 | 0 | 0 |
| 31873 | 690 | 52 | 0 | 0 | 6 | 3 |
| 31902 | 510 | 7 | 1 | 1 | 0 | 0 |
| 31921 | 466 | 63 | 3 | 2 | 0 | 0 |
| 31923 | 415 | 54 | 2 | 2 | 1 | 1 |
| 32027 | 354 | 13 | 0 | 0 | 4 | 4 |
| 32092 | 593 | 30 | 0 | 0 | 0 | 0 |
| 32097 | 581 | 53 | 3 | 2 | 0 | 0 |

## C. MAST: T_i after a step of the SW beam while the viewed SS beam stays on

Shots: {'used': 2309, 'no_block_or_irregular': 1088, 'no_beam_trace': 289}. Ratio of T_i to its level in the four frames before the step (5 ms frames; median over events of the median over chords).

- **step down, all**: 741 events in 122 shots. R_1 = 1.019 [0.988, 1.046], R_late (50 to 80 ms) = 0.850 [0.797, 0.900]; C1 direction holds; C2 delay statistic +0.112 [+0.015, +0.191] -> holds; half response at frame 8.
- **step down, core**: 728 events in 120 shots. R_1 = 1.019 [0.988, 1.050], R_late (50 to 80 ms) = 0.851 [0.793, 0.898]; C1 direction holds; C2 delay statistic +0.110 [+0.014, +0.191] -> holds; half response at frame 8.
- **step down, edge**: 151 events in 25 shots. R_1 = 1.032 [0.963, 1.063], R_late (50 to 80 ms) = 0.976 [0.872, 1.215]; C1 direction does not hold; C2 delay statistic -0.040 [-0.084, +0.125] -> does not hold; half response at frame 1.
- **step up, all**: 6816 events in 949 shots. R_1 = 1.332 [1.315, 1.349], R_late (50 to 80 ms) = 2.395 [2.315, 2.485]; C1 direction holds; C2 delay statistic +0.731 [+0.678, +0.797] -> holds; half response at frame 5.
- **step up, core**: 6686 events in 933 shots. R_1 = 1.328 [1.311, 1.345], R_late (50 to 80 ms) = 2.349 [2.267, 2.440]; C1 direction holds; C2 delay statistic +0.693 [+0.636, +0.758] -> holds; half response at frame 5.
- **step up, edge**: 752 events in 133 shots. R_1 = 1.194 [1.168, 1.228], R_late (50 to 80 ms) = 2.179 [2.001, 2.371]; C1 direction holds; C2 delay statistic +0.791 [+0.645, +0.933] -> holds; half response at frame 6.

| frame after the step | -4 | -3 | -2 | -1 | +0 | +1 | +2 | +3 | +4 | +5 | +6 | +7 | +8 | +9 | +10 | +11 | +12 | +13 | +14 | +15 | +16 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| step down, all radii | 0.982 | 0.994 | 1.007 | 1.016 | 1.019 | 1.019 | 1.005 | 0.995 | 0.988 | 0.975 | 0.958 | 0.935 | 0.904 | 0.887 | 0.877 | 0.872 | 0.861 | 0.846 | 0.839 | 0.834 | 0.819 |
| step up, all radii | 0.864 | 0.952 | 1.043 | 1.137 | 1.232 | 1.332 | 1.436 | 1.536 | 1.634 | 1.727 | 1.821 | 1.910 | 1.997 | 2.079 | 2.165 | 2.250 | 2.340 | 2.418 | 2.476 | 2.536 | 2.583 |
