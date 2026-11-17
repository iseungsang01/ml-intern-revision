# B.46 -- the reported T_i and the last beam-off frame (registered: ces_prediction/experiments/PREREGISTRATION_B46.md)

CPU, nothing trained. All 641 discharges, 10 ms files, grid rows only (first and last row of a file dropped). T_i with held values and values above 3 keV removed (`load_grid_files(DATA, True, 3000.0)`). An absent frame (beam-off frame) is a time step between 15 and 45 ms inside a segment (a segment ends at a step of 0.5 s or more); `tb` is the stamp before the step and `ta` the stamp after it. ECEI level: mean of the four ECEI channels. Part J: step = ln(mean after / mean before) with the window before (tb - 200 ms, tb] and the window after [ta + 150 ms, ta + 350 ms), at least 10 measured T_i rows in each; events have no other absent frame within 400 ms before tb or after ta; controls are stamps tc every 500 ms along a segment with no absent frame within 600 ms, windows (tc - 200 ms, tc] and [tc + 180 ms, tc + 380 ms). Spread = 1.4826 x median absolute deviation from the median. Part A: triples of consecutive measured rows (both steps 10 ms), relative component = 1.4826 MAD(second difference) / sqrt(6) / median T_i of the middle rows, by the age of the last absent frame of the segment. Intervals: 95 %, 2,000 resamples of discharges.

## Checks before the readout

- (a) Grid rows over all files: 245,925 (file_rules.md / PROJECT_KNOWLEDGE.md section 2: 245,925); T_i missing in the raw CSV column before the removal of held values and of values above 3 keV: 18,934 rows = 7.70 % (reference 7.70 %) -> reproduced. After the removal T_i is NaN on 8.19 % of the grid rows (1,197 rows above 3 keV, 1 held row). Segments: 698 (file_rules.md: 698).
- (b) Pooled robust component over all triples, 1.4826 MAD(second difference) / sqrt(6): 42.75 eV over 210,908 triples (b40_noise.md, robust sqrt(N_hi) of the cut population: 42.8 eV over 210,908 triples) -> reproduced within 0.5 eV.
- (c) Absent frames: 1,637 absent frames in 1,414 steps between 15 and 45 ms (file_rules.md: 1,637 in 1,414); 509 of 641 discharges have at least one.

## Registered statements

- **J1** (spread of the T_i step at absent frames over that at control stamps, interval above 1): 0.927 [0.814, 1.075]; 1188 events in 460 discharges, 1415 controls in 485 discharges -> **does not hold**.
- **J2** (J1 ratio over the same ratio for the ECEI level, interval above 1): 0.776 [0.639, 0.950] (T_i ratio on these rows 0.920 [0.812, 1.063], ECEI ratio 1.186 [0.998, 1.394]; 1180 events and 1405 controls with a valid ECEI step, 8 events and 10 controls dropped by the 5 % rule) -> **does not hold**.
- **A1** (median over discharges of the relative component at 1.0-2.0 s over that at 0.1-0.5 s, interval excludes 1): 1.013 [0.982, 1.051]; 292 discharges -> **does not hold**.

## Part J: the step across an absent frame

### Counts

| stage | events | controls |
|---|---|---|
| absent steps (events) / control stamps | 1,414 | 4,440 |
| isolated (400 ms; controls 600 ms) | 1,290 | 1,706 |
| both windows with >= 10 measured T_i rows (main) | 1,188 | 1,415 |
| same, moved window | 1,171 | 1,328 |

### Spread and median of the step (main windows)

| readout | controls | quantity | events | controls | events / controls (or difference) |
|---|---|---|---|---|---|
| registered | all discharges | T_i spread of step | 0.1309 [0.1155, 0.1482] | 0.1413 [0.1239, 0.1541] | 0.927 [0.814, 1.075] |
| registered | all discharges | T_i median step | -0.0007 [-0.0087, +0.0075] | +0.0011 [-0.0055, +0.0066] | -0.0017 [-0.0123, +0.0096] |
| registered | all discharges | ECEI spread of step | 0.0790 [0.0663, 0.0885] | 0.0666 [0.0575, 0.0764] | 1.186 [0.998, 1.394] |
| registered | all discharges | ECEI median step | -0.0014 [-0.0047, +0.0006] | +0.0000 [-0.0031, +0.0000] | -0.0014 [-0.0045, +0.0023] |
| registered | all discharges | J2 ratio | | | 0.776 [0.639, 0.950] |
| reported | discharges with an event | T_i spread of step | 0.1309 [0.1153, 0.1478] | 0.1307 [0.1136, 0.1471] | 1.002 [0.860, 1.154] |
| reported | discharges with an event | T_i median step | -0.0007 [-0.0088, +0.0069] | +0.0004 [-0.0068, +0.0071] | -0.0011 [-0.0122, +0.0094] |
| reported | discharges with an event | ECEI spread of step | 0.0790 [0.0665, 0.0892] | 0.0673 [0.0575, 0.0782] | 1.175 [0.988, 1.390] |
| reported | discharges with an event | ECEI median step | -0.0014 [-0.0049, +0.0006] | +0.0000 [-0.0033, +0.0000] | -0.0014 [-0.0045, +0.0029] |
| reported | discharges with an event | J2 ratio | | | 0.845 [0.687, 1.033] |
| moved window | all discharges | T_i spread of step | 0.1388 [0.1218, 0.1546] | 0.1450 [0.1287, 0.1618] | 0.957 [0.840, 1.075] |
| moved window | all discharges | T_i median step | +0.0006 [-0.0087, +0.0072] | +0.0059 [-0.0004, +0.0118] | -0.0053 [-0.0168, +0.0043] |
| moved window | all discharges | ECEI spread of step | 0.0850 [0.0731, 0.0931] | 0.0719 [0.0614, 0.0826] | 1.182 [1.011, 1.361] |
| moved window | all discharges | ECEI median step | -0.0003 [-0.0040, +0.0026] | +0.0000 [-0.0028, +0.0000] | -0.0003 [-0.0039, +0.0031] |
| moved window | all discharges | J2 ratio | | | 0.801 [0.672, 0.965] |
| moved, reported | discharges with an event | T_i spread of step | 0.1388 [0.1233, 0.1538] | 0.1396 [0.1210, 0.1552] | 0.994 [0.878, 1.126] |
| moved, reported | discharges with an event | T_i median step | +0.0006 [-0.0087, +0.0071] | +0.0048 [-0.0013, +0.0106] | -0.0042 [-0.0155, +0.0056] |
| moved, reported | discharges with an event | ECEI spread of step | 0.0850 [0.0732, 0.0929] | 0.0730 [0.0622, 0.0838] | 1.163 [0.988, 1.348] |
| moved, reported | discharges with an event | ECEI median step | -0.0003 [-0.0039, +0.0022] | +0.0000 [-0.0027, +0.0000] | -0.0003 [-0.0037, +0.0032] |
| moved, reported | discharges with an event | J2 ratio | | | 0.850 [0.707, 1.033] |

| readout | events (discharges) | controls (discharges) | resampled discharges | ECEI dropped (events / controls) | T_i excess spread (% of level) | bound (%) | ECEI excess spread (%) |
|---|---|---|---|---|---|---|---|
| main|all | 1188 (460) | 1415 (485) | 547 | 8 / 10 | 0.00 [0.00, 5.17] | 5.17 | 4.25 [0.00, 5.90] |
| main|event_discharges | 1188 (460) | 1273 (398) | 460 | 8 / 9 | 0.79 [0.00, 7.19] | 7.19 | 4.15 [0.00, 5.90] |
| moved|all | 1171 (454) | 1328 (452) | 533 | 9 / 7 | 0.00 [0.00, 5.38] | 5.38 | 4.52 [1.14, 6.06] |
| moved|event_discharges | 1171 (454) | 1201 (373) | 454 | 9 / 6 | 0.00 [0.00, 6.75] | 6.75 | 4.34 [0.00, 5.96] |

The step is a natural logarithm; a spread of 0.0100 is 1 % of the level. Excess spread = sqrt(max(0, spread_events^2 - spread_controls^2)); its interval is the 2.5 and 97.5 % quantiles of the resampled values (zero where the difference is negative), and the bound is the upper end.

### By third of the T_i level before the event (registered readout; edges 507 and 800 eV, tertiles of the events)

| third | events (discharges) | controls | T_i spread events | T_i spread controls | T_i ratio | T_i excess (%) | median T_i step events / controls | ECEI ratio | J2 ratio |
|---|---|---|---|---|---|---|---|---|---|
| 1 (29-507 eV) | 396 (207) | 631 | 0.1961 [0.1591, 0.2469] | 0.2475 [0.2143, 0.2999] | 0.792 [0.621, 0.982] | 0.00 [0.00, 0.00] | +0.0255 / +0.0375 | 1.001 [0.773, 1.495] | 0.780 [0.489, 1.075] |
| 2 (508-800 eV) | 396 (192) | 428 | 0.1089 [0.0935, 0.1310] | 0.0982 [0.0819, 0.1202] | 1.109 [0.903, 1.377] | 4.71 [0.00, 8.44] | -0.0001 / -0.0094 | 1.207 [0.894, 1.556] | 0.902 [0.668, 1.254] |
| 3 (800-2015 eV) | 396 (183) | 356 | 0.1124 [0.0949, 0.1350] | 0.0984 [0.0778, 0.1153] | 1.142 [0.927, 1.510] | 5.43 [0.00, 9.66] | -0.0192 / -0.0155 | 1.048 [0.828, 1.322] | 1.121 [0.864, 1.534] |

## Part A: the independent component by the age of the last absent frame

| age bin (s) | triples (discharges) | rows | relative component | component (eV) | median T_i (eV) | share missing | share above 3 keV | share held |
|---|---|---|---|---|---|---|---|---|
| 0.1-0.3 | 24,241 (498) | 26,865 | 0.0712 [0.0664, 0.0768] | 46.1 | 648 | 5.43 [4.51, 6.44] % | 0.49 [0.35, 0.65] % | 0.00 [0.00, 0.00] % |
| 0.3-0.6 | 33,418 (483) | 37,897 | 0.0717 [0.0666, 0.0782] | 45.5 | 635 | 7.20 [6.11, 8.45] % | 0.40 [0.30, 0.50] % | 0.00 [0.00, 0.01] % |
| 0.6-1.0 | 39,404 (459) | 45,113 | 0.0708 [0.0658, 0.0768] | 44.8 | 633 | 7.22 [6.07, 8.60] % | 0.43 [0.30, 0.61] % | 0.00 [0.00, 0.00] % |
| 1.0-1.5 | 32,259 (340) | 37,096 | 0.0666 [0.0609, 0.0733] | 39.4 | 592 | 6.23 [5.00, 7.53] % | 0.48 [0.35, 0.65] % | 0.00 [0.00, 0.00] % |
| 1.5-3 | 3,813 (27) | 3,943 | 0.0547 [0.0422, 0.0743] | 33.5 | 612 | 0.79 [0.11, 2.06] % | 0.08 [0.00, 0.25] % | 0.00 [0.00, 0.00] % |
| above 3 | 736 (7) | 743 | 0.0459 [0.0364, 0.0628] | 27.7 | 605 | 0.00 [0.00, 0.00] % | 0.00 [0.00, 0.00] % | 0.00 [0.00, 0.00] % |
| no absent frame before in the segment | 65,193 (611) | 79,588 | 0.0823 [0.0764, 0.0885] | 41.4 | 503 | 10.27 [8.88, 11.75] % | 0.61 [0.48, 0.75] % | 0.00 [0.00, 0.00] % |
| 0-0.1 (after the fact) | 11,844 (490) | 14,680 | 0.0709 [0.0661, 0.0762] | 44.2 | 624 | 6.62 [5.66, 7.73] % | 0.34 [0.23, 0.46] % | 0.00 [0.00, 0.00] % |

A1: 292 discharges with at least 30 triples in each window; median ratio 1.013 [0.982, 1.051]; quantiles 0.1 / 0.25 / 0.5 / 0.75 / 0.9 of the ratio 0.691 / 0.864 / 1.013 / 1.232 / 1.627; share of discharges with a ratio above 1: 0.517.

### Absent frames per discharge

| absent frames (or absent steps) per discharge | discharges by frames | discharges by steps |
|---|---|---|
| 0 | 132 | 132 |
| 1 | 134 | 137 |
| 2 | 105 | 111 |
| 3-5 | 209 | 234 |
| 6-10 | 52 | 24 |
| 11-20 | 6 | 3 |
| 21-50 | 3 | 0 |
| 51 or more | 0 | 0 |

Absent frames per discharge: mean 2.55, quantiles 0.1 / 0.25 / 0.5 / 0.75 / 0.9 0 / 1 / 2 / 3 / 5, largest 40 (s31905.csv 40, s31904.csv 35, s31907.csv 33, s31987.csv 20, s31902.csv 20).

Spacing between consecutive absent frames (tb of the next absent frame minus ta of the previous one, inside a segment; 902 spacings): quantiles 0.1 / 0.25 / 0.5 / 0.75 / 0.9 0.480 / 0.980 / 1.470 / 1.480 / 1.480 s.

| spacing (s) | count |
|---|---|
| 0-0.1 | 23 |
| 0.1-0.2 | 22 |
| 0.2-0.4 | 28 |
| 0.4-1.0 | 273 |
| 1.0-2.0 | 555 |
| 2.0-5.0 | 1 |
| 5.0 or more | 0 |

## After the fact (not registered; not used for any verdict)

1. The ECEI level is exactly constant over both windows (step exactly 0) at 78 controls (32 discharges) and 16 events (12 discharges); 9,388 of 245,284 consecutive grid-row pairs have the same ECEI level, and 3 files have a constant ECEI level (s30967.csv, s31334.csv, s31702.csv). Zero steps shrink the ECEI spread at the controls more than at the events. With them dropped for ECEI: ECEI spread events 0.0812 [0.0695, 0.0908], controls 0.0743 [0.0644, 0.0845], ratio 1.093 [0.928, 1.250]; T_i ratio on the same rows 0.940 [0.826, 1.087]; J2 ratio 0.859 [0.734, 1.055] (1164 events, 1327 controls).

## Expectations of the registration (section 7)

| expectation | observed | met |
|---|---|---|
| events 300 to 3,000 in 100 to 400 discharges | 1188 events in 460 discharges | no |
| J1 ratio 0.95 to 1.25 | 0.927 [0.814, 1.075] | no |
| J1 not established or weakly so (does not hold, or the lower end below 1.05) | J1 does not hold, lower end 0.814 | yes |
| J2 does not hold | J2 does not hold | yes |
| excess spread below 5 % of the level | 0.00 % [0.00, 5.17] | yes |
| A1 ratio 0.9 to 1.1 | 1.013 [0.982, 1.051] | yes |
| A1 interval includes 1 | [0.982, 1.051] | yes |

5 of 7 expectations met.

## Choices made where the registration left room

- Grid rows: the rows of `load_grid_files` (the fast channels are complete on every row) without the first and the last row of each file. T_i: the cut population (held values and values above 3 keV set to NaN), as in B.40 and B.44. The raw T_i column (`load_grid_files(DATA, False, 0.0)`) is read for the shares of missing values and of values above 3 keV.
- Absent frame: a step strictly between 15 and 45 ms inside a segment (a segment ends at a step of 0.5 s or more). One step is one event, whatever the number of frames it removes (20, 30 or 40 ms); `tb` and `ta` are the stamps on either side of the step. The count of absent frames is round(step / 10 ms) - 1 per step (as in B.44).
- Isolation: an event is dropped when another absent step (its open interval (tb', ta')) overlaps [tb - 400 ms, ta + 400 ms]. Steps of 45 ms to 0.5 s (76 in the grid) are not absent frames and do not count.
- Window membership uses a tolerance of 0.1 ms on the float32 time stamps; rows must lie in the segment of the event. A window may contain grid steps longer than 10 ms that are not absent frames; nothing further is required of it.
- T_i window mean: the mean of the measured (finite) T_i rows of the window. ECEI window mean: the mean over all grid rows of the window of the ECEI level (mean of the four ECEI channels, raw CSV values).
- ECEI 5 % rule: an event or control is dropped for ECEI when the ECEI mean of either window is below 5 % of the median over the grid rows of its discharge of the absolute ECEI level (this also removes non-positive means, for which the logarithm is undefined). The same rule is applied to the controls.
- Control stamps: in each segment the targets are segment start + 0.6 s + k x 0.5 s up to the last stamp of the segment; tc is the first grid stamp at or after the target. A control is kept when no absent step overlaps [tc - 600 ms, tc + 600 ms] and both windows hold 10 measured T_i rows.
- Controls: the registration says "in the same discharges"; following the lead's instruction the registered readout uses controls from all discharges (the discharge set of the data, which events and controls share), and controls restricted to discharges with at least one event are reported beside it.
- Resampling: 2,000 multinomial resamples of the discharges that carry at least one event or control of the cell's variant (events and controls of a discharge drawn together); the spread, the medians, the ratio and the excess are recomputed in each resample with the drawn multiplicities (weighted median = median of the repeated sample). One seed (20261118) with fixed offsets per variant. J2 uses the events and controls with a valid ECEI step for both the T_i ratio and the ECEI ratio.
- Excess spread: sqrt(max(0, spread_events^2 - spread_controls^2)) x 100 (the step is a natural logarithm, so a spread of 0.01 is 1 % of the level); the bound is the 97.5 % quantile of its resampled values.
- Moved window: the window after an event becomes [ta + 250 ms, ta + 450 ms); controls are moved by the same 100 ms ([tc + 280 ms, tc + 480 ms)); the window before and the isolation rules are unchanged; the 10-row requirement is applied to the moved windows, so the moved sets are not identical to the main sets.
- Thirds of the level: edges are the tertiles of the mean T_i of the window before over the events of the main readout; controls are assigned by the same edges; the resamples of the full readout are reused.
- Part A: triples as in B.40 (three consecutive measured rows, both steps 10 ms within 0.1 ms). The age of a triple (or a row) is its stamp minus ta of the last absent step of the same segment with ta at or before it. Age bins are half-open [lo, hi). The A1 windows are [0.1, 0.5) s and [1.0, 2.0) s. The pooled relative component of a bin is 1.4826 MAD(second difference) / sqrt(6) over the median T_i of the middle rows, both over the pooled triples of the bin; its interval resamples all 641 discharges.
- Shares by age bin are over grid rows of the bin: missing = NaN in the raw CSV; above 3 keV = raw value above 3,000 eV (denominator: all grid rows of the bin); held (removed as held, at most 3 keV) is reported beside them. An age bin 0-0.1 s (rows inside the recovery) is reported after the fact; it is not part of the registration.
- A1 interval: 2,000 resamples of the discharges that qualify (at least 30 triples in each window), median of the per-discharge ratios recomputed in each.

## Notes on the readout

- The absent frames are close to periodic: the median spacing between consecutive absent frames of a segment is about 1.47 s (table above). An exclusion of 600 ms on either side of a control stamp therefore removes most stamps of discharges with regular blips, and the controls of the registered readout come in part from discharges without events. The reported variant with controls restricted to discharges with an event is given beside it.
- For the same reason the age bins of 1.5-3 s and above 3 s hold few discharges (see the triple counts); they are discharges or segments without the regular blip pattern, and their components are not a within-discharge comparison. A1 compares ages inside a discharge.
- The group without an absent frame before in the segment is mostly the early part of each segment (before the first blip); its median T_i is lower than that of the other bins.

## Deviations

None from the registered definitions, thresholds or populations.

Run time: 64 s.
