# B.32, read after the fact (not registered)

Written after the registered readout (`b32_two_sources.md`) had been read. CPU, no model. Intervals: 95 %, 2,000 resamples of discharges (shots on MAST).

## A'. KSTAR, all files: measured T_i change against a proportional reading

Rows that are not within ten rows after a blip. Change of the measured T_i from its last observed value, by the mean departure d of the nine BES channels of the row; proportional reading = 50 eV per standard deviation times the mean d of the bin. Values above 3 keV left out.

| mean BES departure (sd) | rows | mean d (sd) | mean T_i change (eV) | proportional reading (eV) | measured minus proportional (eV) |
|---|---|---|---|---|---|
| -9.0 to -3.0 | 187 | -4.59 | -54.0 [-113.5, +2.7] | -229.5 | +175.5 [+113.5, +234.5] |
| -3.0 to -2.0 | 487 | -2.32 | -109.4 [-145.8, -73.4] | -115.9 | +6.4 [-30.5, +42.1] |
| -2.0 to -1.0 | 5,966 | -1.31 | -45.9 [-59.0, -32.3] | -65.3 | +19.5 [+6.6, +33.0] |
| -1.0 to -0.5 | 17,908 | -0.69 | -33.9 [-41.3, -27.0] | -34.7 | +0.8 [-6.5, +7.7] |
| -0.5 to -0.2 | 33,440 | -0.33 | -20.7 [-24.6, -16.9] | -16.3 | -4.4 [-8.3, -0.5] |
| -0.2 to +0.2 | 88,381 | -0.00 | -0.2 [-1.3, +0.9] | -0.1 | -0.2 [-1.3, +0.9] |
| +0.2 to +0.5 | 32,653 | +0.33 | +20.8 [+17.2, +24.7] | +16.4 | +4.4 [+0.8, +8.2] |
| +0.5 to +1.0 | 18,763 | +0.70 | +36.7 [+31.2, +42.4] | +35.0 | +1.7 [-3.8, +7.3] |
| +1.0 to +2.0 | 6,942 | +1.32 | +42.4 [+32.8, +51.8] | +65.9 | -23.6 [-33.2, -14.1] |
| +2.0 to +3.0 | 979 | +2.37 | +32.9 [+7.7, +56.4] | +118.7 | -85.8 [-111.3, -62.1] |
| +3.0 to +9.0 | 299 | +4.18 | +65.3 [-17.5, +154.1] | +209.0 | -143.7 [-226.2, -57.5] |

First row after a blip (1,203 rows, the registered table): median T_i ratio of D1 minus D2 = -0.002 [-0.052, +0.057]; D2 minus D3 = -0.101 [-0.157, -0.047].

| group | rows | median T_i change from the median of the six rows before (eV) | rows with the last row before observed | mean change from that last row (eV) | mean of that last row minus the median of six (eV) |
|---|---|---|---|---|---|
| D1 (d <= -3) | 461 | -124.2 [-144.0, -102.8] | 455 | -174.2 | +7.5 |
| D2 (-3 < d <= -2) | 257 | -111.3 [-145.2, -83.4] | 253 | -157.2 | +11.0 |
| D3 (d > -2) | 485 | -69.9 [-95.2, -50.5] | 477 | -99.1 | +38.5 |

## B'. KSTAR, twelve discharges with a raw stream: departures beyond three standard deviations per channel value

Rows inside the raw window and not within ten rows after a blip; nine BES channel values per row. Burst: the B.31 detector, unchanged. Ratio = share among the channel values (or rows) named, over the share among all rows.

Rows: 5,939. Channel values above +3: 110 in 8 discharges (rows with at least one: 28); below -3: 110 in 7 discharges.

- above +3, a detected burst within 1 ms of the time stamp: 110 values; share 0.564 against 0.015 over all rows; ratio 37.2 [25.0, 70.5].
- above +3, a burst onset from 3 ms before to 1 ms after the stamp: 110 values; share 0.700 against 0.027 over all rows; ratio 26.1 [17.0, 49.1].
- rows with at least one channel above +3, a burst within 1 ms: share 0.393; ratio 25.9 [16.9, 49.7].
- below -3, raw BES level at the stamp below 0.8 of its 200 ms running median: 110 values; share 0.727 against 0.016 over all rows; ratio 46.4 [25.0, 121.9].
- below -3, a detected burst within 1 ms: 110 values; share 0.000 against 0.015 over all rows; ratio 0.0 [0.0, 0.0].

| discharge | rows | rows with a burst within 1 ms | values above +3 | with a burst within 1 ms | values below -3 | with a raw dip |
|---|---|---|---|---|---|---|
| 31097 | 754 | 2 | 0 | 0 | 1 | 1 |
| 31114 | 464 | 5 | 1 | 0 | 0 | 0 |
| 31357 | 354 | 0 | 0 | 0 | 0 | 0 |
| 31359 | 265 | 11 | 15 | 9 | 0 | 0 |
| 31747 | 493 | 7 | 0 | 0 | 2 | 0 |
| 31873 | 690 | 14 | 1 | 1 | 53 | 27 |
| 31902 | 510 | 1 | 9 | 9 | 0 | 0 |
| 31921 | 466 | 15 | 34 | 19 | 8 | 6 |
| 31923 | 415 | 11 | 30 | 15 | 5 | 5 |
| 32027 | 354 | 3 | 1 | 0 | 38 | 38 |
| 32092 | 593 | 8 | 0 | 0 | 3 | 3 |
| 32097 | 581 | 13 | 19 | 9 | 0 | 0 |

## C'. MAST: one event per switch of the SW beam, and windows without a switch as a control

Event at frame 0: the first 5 ms frame in which the SW power is on the other side of half its level; the eight frames from -9 to -2 are in the state before (frame -1 may hold the switch), frames +1 to +16 in the state after, and the SS power is steady from -9 to +16. Ratio of T_i to its level in frames -5 to -2 (median over chords, then over events). Control: windows in which the SW beam stays as it was before the switch, weighted to the events' distribution of the frame index since the start of the block (bins of 5 frames, at least 20 control windows). Net = events minus control. Shots with a block and both beam traces: 2,309.

- **SW switched off, all**: 78 events in 78 shots (frame index median 38, quartiles 28 to 50). Ratio at frame 0 0.974 [0.944, 1.002], at +1 0.960 [0.921, 0.999], at 50 to 80 ms 0.803 [0.708, 0.866]; delay statistic +0.118 [+0.043, +0.216]; half response at frame 4. Against 5,906 control windows of 1014 shots (77 events matched): net at frame 0 -0.023 [-0.051, +0.005], at +1 -0.034 [-0.067, +0.002], at 50 to 80 ms -0.075 [-0.165, -0.018]; frames -9 to -6 (no response expected) +0.034 [-0.021, +0.061]; net delay statistic +0.008 [-0.065, +0.108]; half of the net response at frame 2.
- **SW switched off, core**: 77 events in 77 shots (frame index median 38, quartiles 28 to 51). Ratio at frame 0 0.972 [0.949, 1.008], at +1 0.950 [0.928, 1.009], at 50 to 80 ms 0.806 [0.705, 0.871]; delay statistic +0.094 [+0.032, +0.217]; half response at frame 5. Against 5,776 control windows of 994 shots (76 events matched): net at frame 0 -0.022 [-0.047, +0.007], at +1 -0.037 [-0.062, +0.015], at 50 to 80 ms -0.065 [-0.158, -0.001]; frames -9 to -6 (no response expected) +0.033 [-0.016, +0.058]; net delay statistic -0.009 [-0.072, +0.101]; half of the net response at frame 1.
- **SW switched off, edge**: 19 events in 19 shots (frame index median 39, quartiles 24 to 44). Ratio at frame 0 1.002 [0.917, 1.063], at +1 0.946 [0.871, 1.111], at 50 to 80 ms 0.944 [0.815, 1.202]; delay statistic -0.051 [-0.209, +0.117]; half response at frame 1. Against 1,380 control windows of 229 shots (19 events matched): net at frame 0 -0.015 [-0.105, +0.044], at +1 -0.072 [-0.154, +0.091], at 50 to 80 ms -0.035 [-0.182, +0.184]; frames -9 to -6 (no response expected) -0.020 [-0.070, +0.053]; net delay statistic -0.109 [-0.253, +0.093]; half of the net response at frame 1.
- **SW switched on, all**: 542 events in 541 shots (frame index median 12, quartiles 10 to 20). Ratio at frame 0 1.285 [1.266, 1.299], at +1 1.389 [1.351, 1.414], at 50 to 80 ms 2.091 [1.994, 2.174]; delay statistic +0.313 [+0.245, +0.378]; half response at frame 3. Against 4,504 control windows of 735 shots (493 events matched): net at frame 0 +0.040 [+0.024, +0.059], at +1 +0.068 [+0.044, +0.104], at 50 to 80 ms +0.278 [+0.181, +0.356]; frames -9 to -6 (no response expected) -0.024 [-0.041, -0.003]; net delay statistic +0.142 [+0.051, +0.204]; half of the net response at frame 3.
- **SW switched on, core**: 531 events in 530 shots (frame index median 12, quartiles 10 to 20). Ratio at frame 0 1.271 [1.251, 1.290], at +1 1.362 [1.340, 1.390], at 50 to 80 ms 2.032 [1.943, 2.109]; delay statistic +0.308 [+0.235, +0.368]; half response at frame 3. Against 4,460 control windows of 723 shots (483 events matched): net at frame 0 +0.028 [+0.011, +0.047], at +1 +0.065 [+0.034, +0.084], at 50 to 80 ms +0.259 [+0.177, +0.342]; frames -9 to -6 (no response expected) -0.018 [-0.034, +0.003]; net delay statistic +0.130 [+0.063, +0.207]; half of the net response at frame 4.
- **SW switched on, edge**: 93 events in 92 shots (frame index median 10, quartiles 10 to 15). Ratio at frame 0 1.267 [1.209, 1.301], at +1 1.412 [1.329, 1.449], at 50 to 80 ms 2.370 [2.217, 2.503]; delay statistic +0.547 [+0.421, +0.684]; half response at frame 4. Against 743 control windows of 148 shots (84 events matched): net at frame 0 +0.093 [+0.029, +0.129], at +1 +0.170 [+0.093, +0.213], at 50 to 80 ms +0.560 [+0.366, +0.723]; frames -9 to -6 (no response expected) +0.003 [-0.032, +0.028]; net delay statistic +0.220 [+0.054, +0.407]; half of the net response at frame 5.

| frame | -9 | -8 | -7 | -6 | -5 | -4 | -3 | -2 | -1 | +0 | +1 | +2 | +3 | +4 | +5 | +6 | +7 | +8 | +9 | +10 | +11 | +12 | +13 | +14 | +15 | +16 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SW off, core: events | 1.023 | 1.040 | 1.020 | 1.013 | 0.999 | 1.009 | 0.998 | 0.990 | 0.989 | 0.972 | 0.950 | 0.948 | 0.933 | 0.904 | 0.855 | 0.841 | 0.825 | 0.836 | 0.799 | 0.797 | 0.800 | 0.821 | 0.827 | 0.811 | 0.802 | 0.783 |
| SW off, core: control | 0.989 | 0.992 | 0.993 | 0.996 | 0.998 | 0.999 | 1.002 | 1.000 | 1.000 | 0.993 | 0.987 | 0.981 | 0.973 | 0.962 | 0.957 | 0.948 | 0.942 | 0.932 | 0.917 | 0.903 | 0.889 | 0.873 | 0.860 | 0.859 | 0.841 | 0.828 |
| SW off, core: net | +0.040 | +0.045 | +0.027 | +0.018 | +0.000 | +0.010 | -0.004 | -0.010 | -0.009 | -0.022 | -0.037 | -0.037 | -0.041 | -0.061 | -0.106 | -0.110 | -0.119 | -0.102 | -0.118 | -0.107 | -0.094 | -0.069 | -0.034 | -0.048 | -0.047 | -0.052 |
| SW off, edge: events | 0.929 | 0.956 | 0.935 | 0.963 | 0.982 | 0.996 | 1.012 | 1.019 | 1.029 | 1.002 | 0.946 | 1.002 | 0.925 | 0.979 | 0.992 | 1.033 | 1.040 | 0.984 | 0.923 | 0.988 | 0.939 | 0.955 | 0.923 | 0.948 | 0.951 | 0.902 |
| SW off, edge: control | 0.950 | 0.964 | 0.972 | 0.976 | 0.989 | 0.995 | 1.006 | 1.010 | 1.016 | 1.017 | 1.018 | 1.017 | 1.018 | 1.017 | 1.022 | 1.014 | 1.011 | 1.003 | 1.007 | 0.999 | 0.989 | 0.985 | 0.988 | 0.969 | 0.967 | 0.955 |
| SW off, edge: net | -0.021 | -0.008 | -0.037 | -0.013 | -0.007 | +0.001 | +0.006 | +0.009 | +0.013 | -0.015 | -0.072 | -0.015 | -0.094 | -0.037 | -0.030 | +0.019 | +0.029 | -0.019 | -0.084 | -0.012 | -0.050 | -0.030 | -0.065 | -0.021 | -0.015 | -0.053 |
| SW on, core: events | 0.590 | 0.655 | 0.714 | 0.795 | 0.875 | 0.958 | 1.042 | 1.118 | 1.185 | 1.271 | 1.362 | 1.454 | 1.537 | 1.606 | 1.638 | 1.717 | 1.777 | 1.866 | 1.917 | 1.977 | 2.020 | 2.043 | 2.046 | 2.058 | 2.046 | 2.036 |
| SW on, core: control | 0.615 | 0.686 | 0.758 | 0.827 | 0.896 | 0.966 | 1.035 | 1.101 | 1.166 | 1.223 | 1.277 | 1.331 | 1.377 | 1.426 | 1.478 | 1.526 | 1.558 | 1.591 | 1.637 | 1.671 | 1.692 | 1.708 | 1.723 | 1.761 | 1.773 | 1.791 |
| SW on, core: net | -0.007 | -0.018 | -0.021 | -0.024 | -0.015 | -0.006 | +0.004 | +0.008 | +0.004 | +0.028 | +0.065 | +0.085 | +0.114 | +0.141 | +0.123 | +0.149 | +0.170 | +0.223 | +0.239 | +0.255 | +0.282 | +0.257 | +0.275 | +0.261 | +0.259 | +0.224 |
| SW on, edge: events | 0.754 | 0.785 | 0.815 | 0.861 | 0.907 | 0.960 | 1.025 | 1.103 | 1.167 | 1.267 | 1.412 | 1.503 | 1.594 | 1.704 | 1.794 | 1.873 | 1.990 | 2.065 | 2.174 | 2.157 | 2.249 | 2.375 | 2.426 | 2.390 | 2.433 | 2.564 |
| SW on, edge: control | 0.734 | 0.775 | 0.826 | 0.874 | 0.928 | 0.977 | 1.022 | 1.069 | 1.122 | 1.174 | 1.240 | 1.286 | 1.366 | 1.431 | 1.474 | 1.529 | 1.564 | 1.594 | 1.683 | 1.751 | 1.818 | 1.848 | 1.870 | 1.901 | 1.923 | 1.922 |
| SW on, edge: net | +0.020 | +0.011 | -0.007 | -0.014 | -0.021 | -0.014 | +0.006 | +0.037 | +0.047 | +0.093 | +0.170 | +0.212 | +0.227 | +0.275 | +0.333 | +0.344 | +0.428 | +0.475 | +0.493 | +0.423 | +0.456 | +0.542 | +0.621 | +0.571 | +0.623 | +0.686 |
