# B.40 — a bracket for the white-noise variance of the CES T_i value (registered: ces_prediction/experiments/PREREGISTRATION_B40.md)

CPU, no model. y_t = s_t + e_t; over triples of consecutive measured rows (both steps one frame: 10 ms on KSTAR, 5 ms on MAST), N_lo = -cov(d_t, d_t+1), N_hi = var(second difference) / 6, in three versions: robust (primary; S = 1.4826 MAD, N_hi = S^2(sd)/6, N_lo = [S^2(sd) - S^2(span)]/4), moments, and moments with the differences clipped at their 1st and 99th percentiles. Intervals: 95 %, 2,000 resamples of discharges (shots on MAST); point values are exact, resampled medians use per-discharge histograms on 2,000 pooled-quantile bins.

## Registered statements

- **M1** (MAST, population G: the median squared reported error lies between N_lo and N_hi in at least four of five radius bins, and pooled N_lo / reference has an interval below 1 and N_hi / reference above 1): bins inside 1 of 5 (0.7-0.9); pooled N_lo / ref -2.751 [-2.910, -2.611], N_hi / ref 0.753 [0.724, 0.783] -> **does not hold**.
- **K1** (KSTAR, cut population, robust: N_hi / N_lo at most 1.30 with the upper end at most 1.45): 1.195 [1.147, 1.247] -> **holds**.
- **K2** (N_lo in the lowest fifth of the BES level over N_lo in the highest fifth, interval above 1): 0.216 [0.158, 0.278] -> **does not hold**.

## KSTAR: the bracket over all discharges

Cut population: 210,908 triples of 614 discharges; inclusive population: 212,349 triples of 614 discharges.

| set | version | N_lo (eV^2) | N_hi (eV^2) | sqrt(N_lo) (eV) | sqrt(N_hi) (eV) | N_hi / N_lo |
|---|---|---|---|---|---|---|
| cut | robust | 1529.5 [1290.3, 1826.9] | 1828.0 [1564.4, 2131.2] | 39.1 [35.9, 42.7] | 42.8 [39.6, 46.2] | 1.195 [1.147, 1.247] |
| cut | moments | 18762.3 [16279.9, 21418.7] | 19572.4 [17248.7, 22096.5] | 137.0 [127.6, 146.4] | 139.9 [131.3, 148.6] | 1.043 [1.018, 1.072] |
| cut | clipped | 9308.5 [8082.7, 10582.9] | 10635.0 [9465.0, 11833.7] | 96.5 [89.9, 102.9] | 103.1 [97.3, 108.8] | 1.143 [1.106, 1.187] |
| inclusive | robust | 1564.9 [1329.6, 1860.0] | 1867.2 [1615.5, 2165.9] | 39.6 [36.5, 43.1] | 43.2 [40.2, 46.5] | 1.193 [1.146, 1.244] |
| inclusive | moments | 107265.6 [83205.7, 132095.4] | 108890.2 [86972.6, 131444.0] | 327.5 [288.5, 363.4] | 330.0 [294.9, 362.6] | 1.015 [0.975, 1.066] |
| inclusive | clipped | 10823.6 [9549.0, 12324.1] | 13246.8 [11952.7, 14817.3] | 104.0 [97.7, 111.0] | 115.1 [109.3, 121.7] | 1.224 [1.181, 1.273] |

## KSTAR: by fifth of the BES level (centre row, mean of the nine channels over the discharge median; cut)

Fifth edges: 0.939, 0.987, 1.022, 1.072.

| set | version | N_lo (eV^2) | N_hi (eV^2) | sqrt(N_lo) (eV) | sqrt(N_hi) (eV) | N_hi / N_lo |
|---|---|---|---|---|---|---|
| fifth 1 (42,182) | robust | 600.2 [443.5, 782.5] | 1183.2 [1006.7, 1403.9] | 24.5 [21.1, 28.0] | 34.4 [31.7, 37.5] | 1.971 [1.721, 2.367] |
| fifth 1 (42,182) | moments | 19068.3 [14919.3, 23990.2] | 20416.2 [16960.2, 24404.1] | 138.1 [122.1, 154.9] | 142.9 [130.2, 156.2] | 1.071 [1.002, 1.155] |
| fifth 1 (42,182) | clipped | 8061.5 [6649.7, 9601.5] | 10100.3 [8806.8, 11476.6] | 89.8 [81.5, 98.0] | 100.5 [93.8, 107.1] | 1.253 [1.172, 1.353] |
| fifth 2 (42,181) | robust | 1530.2 [1267.1, 1830.2] | 1711.5 [1445.7, 2017.5] | 39.1 [35.6, 42.8] | 41.4 [38.0, 44.9] | 1.118 [1.074, 1.167] |
| fifth 2 (42,181) | moments | 17349.9 [14246.7, 20825.4] | 18036.6 [15316.9, 21031.2] | 131.7 [119.4, 144.3] | 134.3 [123.8, 145.0] | 1.040 [0.996, 1.093] |
| fifth 2 (42,181) | clipped | 8132.6 [6790.8, 9492.4] | 9207.7 [7967.9, 10502.2] | 90.2 [82.4, 97.4] | 96.0 [89.3, 102.5] | 1.132 [1.085, 1.191] |
| fifth 3 (42,182) | robust | 1502.9 [1237.1, 1829.2] | 1626.5 [1364.2, 1946.5] | 38.8 [35.2, 42.8] | 40.3 [36.9, 44.1] | 1.082 [1.043, 1.125] |
| fifth 3 (42,182) | moments | 14438.7 [11828.2, 17221.9] | 15098.0 [12856.4, 17558.4] | 120.2 [108.8, 131.2] | 122.9 [113.4, 132.5] | 1.046 [0.991, 1.120] |
| fifth 3 (42,182) | clipped | 7413.4 [6169.5, 8615.2] | 8206.8 [7103.8, 9293.8] | 86.1 [78.5, 92.8] | 90.6 [84.3, 96.4] | 1.107 [1.053, 1.172] |
| fifth 4 (42,181) | robust | 1915.1 [1525.0, 2371.5] | 1997.6 [1651.6, 2410.2] | 43.8 [39.1, 48.7] | 44.7 [40.6, 49.1] | 1.043 [1.004, 1.091] |
| fifth 4 (42,181) | moments | 16129.7 [13696.7, 18864.7] | 16562.1 [14344.8, 19003.6] | 127.0 [117.0, 137.3] | 128.7 [119.8, 137.9] | 1.027 [0.990, 1.073] |
| fifth 4 (42,181) | clipped | 9151.6 [7810.6, 10474.4] | 9764.7 [8561.1, 10952.6] | 95.7 [88.4, 102.3] | 98.8 [92.5, 104.7] | 1.067 [1.031, 1.109] |
| fifth 5 (42,182) | robust | 2773.0 [2205.8, 3479.5] | 3311.5 [2775.5, 3965.2] | 52.7 [47.0, 59.0] | 57.5 [52.7, 63.0] | 1.194 [1.110, 1.296] |
| fifth 5 (42,182) | moments | 26091.1 [22396.7, 29829.1] | 27238.4 [23774.6, 30858.3] | 161.5 [149.7, 172.7] | 165.0 [154.2, 175.7] | 1.044 [1.005, 1.091] |
| fifth 5 (42,182) | clipped | 14180.6 [12419.6, 15986.6] | 16344.3 [14653.8, 18112.2] | 119.1 [111.4, 126.4] | 127.8 [121.1, 134.6] | 1.153 [1.102, 1.212] |

| version | N_lo lowest fifth / highest fifth |
|---|---|
| robust | 0.216 [0.158, 0.278] |
| moments | 0.731 [0.590, 0.903] |
| clipped | 0.568 [0.491, 0.656] |

Inclusive population, same ratio: robust 0.218 [0.160, 0.287], moments 2.514 [1.750, 3.544], clipped 0.784 [0.669, 0.921].

## KSTAR: by third of the T_i level (mean of the three rows; cut)

Third edges: 447 and 764 eV.

| set | version | N_lo (eV^2) | N_hi (eV^2) | sqrt(N_lo) (eV) | sqrt(N_hi) (eV) | N_hi / N_lo |
|---|---|---|---|---|---|---|
| third 1 (70,303) | robust | 508.5 [390.4, 658.2] | 752.1 [619.7, 917.7] | 22.5 [19.8, 25.7] | 27.4 [24.9, 30.3] | 1.479 [1.360, 1.654] |
| third 1 (70,303) | moments | 4993.7 [3820.3, 6299.2] | 5320.4 [4254.3, 6512.2] | 70.7 [61.8, 79.4] | 72.9 [65.2, 80.7] | 1.065 [1.015, 1.132] |
| third 1 (70,303) | clipped | 3768.9 [2821.3, 4825.3] | 4186.2 [3359.1, 5111.0] | 61.4 [53.1, 69.5] | 64.7 [58.0, 71.5] | 1.111 [1.042, 1.207] |
| third 2 (70,302) | robust | 1115.3 [951.5, 1327.0] | 1274.2 [1111.9, 1476.5] | 33.4 [30.8, 36.4] | 35.7 [33.3, 38.4] | 1.143 [1.092, 1.206] |
| third 2 (70,302) | moments | 9621.8 [7884.8, 11669.4] | 10671.4 [8927.7, 12659.4] | 98.1 [88.8, 108.0] | 103.3 [94.5, 112.5] | 1.109 [1.069, 1.159] |
| third 2 (70,302) | clipped | 5093.0 [4248.4, 6094.1] | 6416.5 [5484.1, 7507.5] | 71.4 [65.2, 78.1] | 80.1 [74.1, 86.6] | 1.260 [1.197, 1.336] |
| third 3 (70,303) | robust | 8232.5 [6641.2, 9924.5] | 8347.4 [6881.9, 9835.4] | 90.7 [81.5, 99.6] | 91.4 [83.0, 99.2] | 1.014 [0.972, 1.062] |
| third 3 (70,303) | moments | 41669.0 [35134.6, 49026.2] | 42723.5 [36521.5, 49689.7] | 204.1 [187.4, 221.4] | 206.7 [191.1, 222.9] | 1.025 [0.998, 1.056] |
| third 3 (70,303) | clipped | 22932.7 [19569.5, 26449.7] | 26534.7 [23186.1, 30079.7] | 151.4 [139.9, 162.6] | 162.9 [152.3, 173.4] | 1.157 [1.114, 1.213] |

| third | median level (eV) | robust sqrt(N_lo) / level | robust sqrt(N_hi) / level | moments sqrt(N_lo) / level | moments sqrt(N_hi) / level |
|---|---|---|---|---|---|
| 1 | 288.4 [272.4, 305.0] | 0.078 [0.068, 0.090] | 0.095 [0.086, 0.106] | 0.245 [0.211, 0.278] | 0.253 [0.223, 0.283] |
| 2 | 606.6 [597.1, 616.3] | 0.055 [0.051, 0.060] | 0.059 [0.055, 0.063] | 0.162 [0.146, 0.178] | 0.170 [0.156, 0.186] |
| 3 | 989.1 [967.1, 1012.2] | 0.092 [0.083, 0.100] | 0.092 [0.085, 0.099] | 0.206 [0.190, 0.224] | 0.209 [0.194, 0.225] |

Per discharge (robust, cut; 518 discharges with at least 100 triples, N_lo <= 0 in 71): quantiles 0.1 / 0.25 / 0.5 / 0.75 / 0.9 of sqrt(N_lo) 0.0 / 19.4 / 34.0 / 75.6 / 164.9 eV; of sqrt(N_hi) 18.7 / 28.1 / 39.6 / 79.5 / 159.3 eV; of N_hi / N_lo (N_lo > 0) 0.84 / 0.95 / 1.07 / 1.34 / 1.98.

Per discharge (robust, inclusive; 519 discharges with at least 100 triples, N_lo <= 0 in 69): quantiles 0.1 / 0.25 / 0.5 / 0.75 / 0.9 of sqrt(N_lo) 0.0 / 20.3 / 34.5 / 77.0 / 167.6 eV; of sqrt(N_hi) 19.1 / 28.4 / 39.9 / 84.0 / 167.1 eV; of N_hi / N_lo (N_lo > 0) 0.84 / 0.94 / 1.08 / 1.35 / 1.98.

## KSTAR: estimators in units of the noise floor (test rows of the frozen row tables, natural cadence)

Mean squared error over N of the same test discharges (robust); the largest skill over hold an estimator of the signal could reach is 1 - N / MSE_hold.

| split | test discharges / rows | robust sqrt(N_lo), sqrt(N_hi) (eV) | RMSE hold / GP / twin / causal (eV) | hold / N_lo | GP / N_lo | twin / N_lo | causal / N_lo | causal / N_hi | largest skill over hold (N_lo; N_hi) |
|---|---|---|---|---|---|---|---|---|---|
| 42 | 96 / 32,921 | 39.7, 42.4 | 204.5 / 170.1 / 169.2 / 164.5 | 26.54 [17.51, 39.43] | 18.37 [11.99, 27.64] | 18.18 [11.91, 27.18] | 17.18 [11.27, 25.79] | 15.06 [10.37, 21.15] | 0.962 [0.943, 0.975]; 0.957 [0.938, 0.969] |
| 1 | 96 / 35,936 | 45.3, 47.6 | 255.4 / 210.7 / 203.1 / 200.3 | 31.84 [19.62, 50.60] | 21.66 [13.07, 34.28] | 20.14 [12.30, 31.96] | 19.58 [11.80, 31.37] | 17.67 [11.36, 26.61] | 0.969 [0.949, 0.980]; 0.965 [0.946, 0.977] |
| 7 | 96 / 36,196 | 43.8, 46.7 | 233.7 / 194.2 / 192.4 / 181.2 | 28.45 [18.71, 42.33] | 19.66 [12.79, 29.37] | 19.29 [12.68, 28.59] | 17.10 [11.20, 25.45] | 15.07 [10.37, 21.20] | 0.965 [0.947, 0.976]; 0.960 [0.942, 0.972] |
| 123 | 96 / 34,428 | 37.2, 40.3 | 270.0 / 218.6 / 214.1 / 208.0 | 52.60 [33.11, 78.14] | 34.49 [21.88, 51.12] | 33.08 [21.10, 48.75] | 31.21 [19.71, 46.18] | 26.67 [17.79, 38.27] | 0.981 [0.970, 0.987]; 0.978 [0.966, 0.985] |

Same, moments version (causal / N_lo; causal / N_hi): split 42: 1.85 [1.57, 2.24], 1.81 [1.58, 2.12]; split 1: 1.67 [1.43, 1.99], 1.66 [1.47, 1.90]; split 7: 1.78 [1.49, 2.17], 1.72 [1.48, 2.02]; split 123: 1.64 [1.40, 1.97], 1.66 [1.46, 1.92].

## MAST: the bracket beside the reported fit errors

Shots as in the MAST edge study (run_coupling.SHOTS_TXT, 3686 listed; status {"ok": 3686}); blocks and scored chords from run_gate.prepare; bins of normalized radius as in run_coupling. Reference: median over the triple centre rows of the squared reported error.

### Population G (relative T_i error below 0.1): 2,746,400 triples of 2593 shots

| bin | triples / shots | reference median sigma (eV) | robust N_lo / ref | robust N_hi / ref | ref inside (robust) | moments N_lo / ref | moments N_hi / ref | robust N_lo / mean ref | robust N_hi / mean ref |
|---|---|---|---|---|---|---|---|---|---|
| 0-0.3 | 1,305,008 / 2349 | 28.6 | -2.733 [-2.867, -2.607] | 0.631 [0.607, 0.656] | no | -1.519 [-1.739, -1.297] | 3.671 [3.407, 3.950] | -0.957 [-0.998, -0.919] | 0.221 [0.213, 0.229] |
| 0.3-0.5 | 788,708 / 2103 | 24.4 | -3.040 [-3.209, -2.893] | 0.766 [0.736, 0.798] | no | -2.158 [-2.528, -1.753] | 3.938 [3.576, 4.348] | -1.220 [-1.285, -1.163] | 0.307 [0.296, 0.320] |
| 0.5-0.7 | 489,547 / 1509 | 18.4 | -2.064 [-2.231, -1.913] | 0.900 [0.857, 0.950] | no | -0.493 [-1.554, 0.775] | 5.630 [4.468, 7.078] | -0.919 [-0.996, -0.851] | 0.401 [0.379, 0.424] |
| 0.7-0.9 | 138,888 / 570 | 15.5 | -0.904 [-1.102, -0.730] | 1.025 [0.939, 1.121] | yes | 1.948 [-0.425, 5.317] | 6.666 [3.767, 10.897] | -0.364 [-0.453, -0.289] | 0.412 [0.369, 0.462] |
| 0.9-1 | 24,249 / 135 | 16.8 | -0.566 [-0.757, -0.397] | 0.692 [0.564, 0.828] | no | 1.274 [-1.060, 6.414] | 4.100 [1.360, 10.541] | -0.282 [-0.371, -0.206] | 0.344 [0.301, 0.397] |
| all | 2,746,400 / 2593 | 23.9 | -2.751 [-2.910, -2.611] | 0.753 [0.724, 0.783] | no | -1.690 [-2.000, -1.369] | 4.435 [4.071, 4.807] | -0.920 [-0.964, -0.881] | 0.252 [0.243, 0.261] |

| bin | version | N_lo (eV^2) | N_hi (eV^2) | sqrt(N_lo) (eV) | sqrt(N_hi) (eV) | N_hi / N_lo |
|---|---|---|---|---|---|---|
| 0-0.3 | robust | -2232.9 [-2334.1, -2139.1] | 515.5 [500.0, 532.1] | nan [nan, nan] | 22.7 [22.4, 23.1] | -0.231 [-0.241, -0.221] |
| 0-0.3 | moments | -1240.9 [-1436.8, -1045.3] | 2999.6 [2807.0, 3189.6] | nan [nan, nan] | 54.8 [53.0, 56.5] | -2.417 [-2.981, -2.000] |
| 0-0.3 | clipped | -1546.0 [-1629.8, -1465.4] | 1520.4 [1481.7, 1558.8] | nan [nan, nan] | 39.0 [38.5, 39.5] | -0.983 [-1.042, -0.930] |
| 0.3-0.5 | robust | -1804.2 [-1896.8, -1721.9] | 454.7 [440.1, 470.0] | nan [nan, nan] | 21.3 [21.0, 21.7] | -0.252 [-0.262, -0.241] |
| 0.3-0.5 | moments | -1281.0 [-1496.1, -1039.7] | 2337.3 [2137.2, 2559.5] | nan [nan, nan] | 48.3 [46.2, 50.6] | -1.825 [-2.422, -1.463] |
| 0.3-0.5 | clipped | -1510.7 [-1594.2, -1426.4] | 1227.7 [1194.8, 1262.1] | nan [nan, nan] | 35.0 [34.6, 35.5] | -0.813 [-0.865, -0.765] |
| 0.5-0.7 | robust | -702.2 [-749.6, -657.1] | 306.2 [293.4, 318.8] | nan [nan, nan] | 17.5 [17.1, 17.9] | -0.436 [-0.462, -0.411] |
| 0.5-0.7 | moments | -167.7 [-533.5, 264.3] | 1915.0 [1522.8, 2410.0] | nan [1.7, 20.2] | 43.8 [39.0, 49.1] | -11.423 [-113.069, 91.772] |
| 0.5-0.7 | clipped | -745.7 [-810.7, -681.2] | 823.4 [790.0, 858.6] | nan [nan, nan] | 28.7 [28.1, 29.3] | -1.104 [-1.230, -1.005] |
| 0.7-0.9 | robust | -216.2 [-262.1, -176.9] | 245.2 [229.6, 262.7] | nan [nan, nan] | 15.7 [15.2, 16.2] | -1.134 [-1.371, -0.947] |
| 0.7-0.9 | moments | 465.8 [-103.8, 1255.9] | 1594.4 [922.2, 2606.2] | 21.6 [5.9, 35.9] | 39.9 [30.4, 51.1] | 3.423 [-28.480, 28.470] |
| 0.7-0.9 | clipped | -236.1 [-306.5, -170.7] | 558.6 [520.8, 598.6] | nan [nan, nan] | 23.6 [22.8, 24.5] | -2.366 [-3.291, -1.824] |
| 0.9-1 | robust | -160.5 [-210.3, -119.5] | 196.0 [175.5, 221.7] | nan [nan, nan] | 14.0 [13.2, 14.9] | -1.221 [-1.670, -0.938] |
| 0.9-1 | moments | 360.9 [-297.9, 1813.5] | 1161.7 [402.8, 2949.6] | 19.0 [15.3, 44.9] | 34.1 [20.1, 54.3] | 3.219 [-4.415, 4.515] |
| 0.9-1 | clipped | -173.7 [-229.7, -122.6] | 328.9 [291.7, 369.6] | nan [nan, nan] | 18.1 [17.1, 19.2] | -1.893 [-2.795, -1.378] |
| all | robust | -1568.6 [-1653.8, -1491.1] | 429.1 [417.0, 442.1] | nan [nan, nan] | 20.7 [20.4, 21.0] | -0.274 [-0.285, -0.262] |
| all | moments | -963.9 [-1150.3, -779.0] | 2528.9 [2343.5, 2724.9] | nan [nan, nan] | 50.3 [48.4, 52.2] | -2.624 [-3.466, -2.083] |
| all | clipped | -1313.3 [-1385.4, -1242.3] | 1234.5 [1205.3, 1265.8] | nan [nan, nan] | 35.1 [34.7, 35.6] | -0.940 [-0.998, -0.890] |

### Population W (relative T_i error below 0.5): 5,597,427 triples of 3229 shots

| bin | triples / shots | reference median sigma (eV) | robust N_lo / ref | robust N_hi / ref | ref inside (robust) | moments N_lo / ref | moments N_hi / ref | robust N_lo / mean ref | robust N_hi / mean ref |
|---|---|---|---|---|---|---|---|---|---|
| 0-0.3 | 2,208,383 / 3190 | 34.2 | -1.753 [-1.836, -1.669] | 0.538 [0.520, 0.555] | no | 0.753 [0.480, 1.039] | 5.934 [5.565, 6.331] | -0.282 [-0.300, -0.264] | 0.086 [0.082, 0.092] |
| 0.3-0.5 | 1,409,935 / 3059 | 27.8 | -2.068 [-2.171, -1.968] | 0.646 [0.628, 0.666] | no | -0.538 [-0.847, -0.198] | 5.096 [4.760, 5.478] | -0.465 [-0.497, -0.435] | 0.145 [0.138, 0.154] |
| 0.5-0.7 | 1,159,067 / 2722 | 20.3 | -1.287 [-1.356, -1.216] | 0.696 [0.674, 0.718] | no | 0.604 [-0.124, 1.424] | 7.182 [6.375, 8.076] | -0.329 [-0.360, -0.299] | 0.178 [0.163, 0.194] |
| 0.7-0.9 | 685,356 / 1992 | 17.4 | -0.226 [-0.265, -0.189] | 0.652 [0.623, 0.679] | no | 5.639 [4.031, 7.423] | 13.080 [11.076, 15.283] | -0.046 [-0.057, -0.037] | 0.132 [0.116, 0.151] |
| 0.9-1 | 134,686 / 764 | 16.9 | -0.069 [-0.131, -0.011] | 0.695 [0.641, 0.746] | no | 10.148 [5.842, 15.214] | 18.101 [13.461, 23.621] | -0.013 [-0.027, -0.002] | 0.127 [0.095, 0.175] |
| all | 5,597,427 / 3229 | 25.2 | -1.676 [-1.763, -1.590] | 0.652 [0.633, 0.671] | no | 0.899 [0.566, 1.238] | 7.796 [7.365, 8.256] | -0.248 [-0.264, -0.232] | 0.096 [0.091, 0.102] |

| bin | version | N_lo (eV^2) | N_hi (eV^2) | sqrt(N_lo) (eV) | sqrt(N_hi) (eV) | N_hi / N_lo |
|---|---|---|---|---|---|---|
| 0-0.3 | robust | -2047.3 [-2123.8, -1970.7] | 627.9 [607.7, 647.8] | nan [nan, nan] | 25.1 [24.7, 25.5] | -0.307 [-0.319, -0.295] |
| 0-0.3 | moments | 879.3 [554.6, 1215.6] | 6931.0 [6522.6, 7378.0] | 29.7 [23.5, 34.9] | 83.3 [80.8, 85.9] | 7.882 [5.963, 11.955] |
| 0-0.3 | clipped | -1052.2 [-1138.0, -963.9] | 2691.4 [2611.0, 2773.9] | nan [nan, nan] | 51.9 [51.1, 52.7] | -2.558 [-2.825, -2.333] |
| 0.3-0.5 | robust | -1595.7 [-1653.4, -1537.9] | 498.6 [484.8, 513.1] | nan [nan, nan] | 22.3 [22.0, 22.7] | -0.312 [-0.324, -0.302] |
| 0.3-0.5 | moments | -414.7 [-648.6, -152.1] | 3931.5 [3678.6, 4204.2] | nan [5.5, 8.2] | 62.7 [60.7, 64.8] | -9.480 [-26.717, -5.738] |
| 0.3-0.5 | clipped | -1340.6 [-1409.4, -1275.7] | 1676.0 [1634.3, 1720.0] | nan [nan, nan] | 40.9 [40.4, 41.5] | -1.250 [-1.325, -1.177] |
| 0.5-0.7 | robust | -527.8 [-550.1, -504.2] | 285.3 [276.7, 293.8] | nan [nan, nan] | 16.9 [16.6, 17.1] | -0.541 [-0.564, -0.519] |
| 0.5-0.7 | moments | 247.7 [-51.0, 583.8] | 2945.7 [2609.4, 3299.5] | 15.7 [5.4, 24.2] | 54.3 [51.1, 57.4] | 11.893 [-67.411, 90.574] |
| 0.5-0.7 | clipped | -726.9 [-771.8, -684.4] | 909.6 [881.3, 936.8] | nan [nan, nan] | 30.2 [29.7, 30.6] | -1.251 [-1.341, -1.168] |
| 0.7-0.9 | robust | -68.8 [-79.9, -58.1] | 198.3 [191.4, 205.1] | nan [nan, nan] | 14.1 [13.8, 14.3] | -2.884 [-3.410, -2.498] |
| 0.7-0.9 | moments | 1715.4 [1232.4, 2247.8] | 3979.4 [3365.0, 4653.9] | 41.4 [35.1, 47.4] | 63.1 [58.0, 68.2] | 2.320 [1.970, 2.846] |
| 0.7-0.9 | clipped | -147.1 [-175.6, -119.5] | 548.8 [529.4, 569.1] | nan [nan, nan] | 23.4 [23.0, 23.9] | -3.732 [-4.617, -3.121] |
| 0.9-1 | robust | -19.6 [-37.2, -3.2] | 198.6 [188.3, 209.6] | nan [0.5, 2.9] | 14.1 [13.7, 14.5] | -10.136 [-43.216, -5.007] |
| 0.9-1 | moments | 2900.2 [1672.6, 4340.8] | 5173.2 [3878.1, 6709.1] | 53.9 [40.9, 65.9] | 71.9 [62.3, 81.9] | 1.784 [1.478, 2.407] |
| 0.9-1 | clipped | 1.0 [-38.1, 40.1] | 539.5 [500.6, 581.6] | 1.0 [0.8, 6.6] | 23.2 [22.4, 24.1] | 524.882 [-350.028, 429.622] |
| all | robust | -1063.6 [-1107.5, -1019.5] | 413.4 [403.3, 423.8] | nan [nan, nan] | 20.3 [20.1, 20.6] | -0.389 [-0.405, -0.374] |
| all | moments | 570.2 [358.7, 787.4] | 4947.1 [4687.2, 5228.4] | 23.9 [18.9, 28.1] | 70.3 [68.5, 72.3] | 8.677 [6.529, 13.161] |
| all | clipped | -944.9 [-998.6, -889.5] | 1672.5 [1633.6, 1715.7] | nan [nan, nan] | 40.9 [40.4, 41.4] | -1.770 [-1.887, -1.663] |

M1 in the other versions (population G): moments 3 bins inside, pooled True; clipped 5 bins inside, pooled True. Population W (reported), robust: 0 bins inside, pooled False.

## After the fact (not registered; not used for any verdict)

1. MAST population G with triples two frames apart (10 ms, the KSTAR frame and the cadence of the regime number); 2,635,820 triples of 2593 shots. Robust: bins inside 5 of 5, pooled interval rule True.

| bin | reference median sigma (eV) | robust N_lo / ref | robust N_hi / ref | robust sqrt(N_hi) (eV) | moments N_lo / ref | moments N_hi / ref |
|---|---|---|---|---|---|---|
| 0-0.3 | 29.1 | -9.039 [-9.499, -8.575] | 1.331 [1.275, 1.391] | 33.6 [33.1, 34.2] | -6.288 [-6.703, -5.867] | 6.265 [5.887, 6.682] |
| 0.3-0.5 | 24.8 | -9.915 [-10.559, -9.301] | 1.701 [1.622, 1.786] | 32.3 [31.6, 33.0] | -8.726 [-9.326, -8.107] | 7.014 [6.528, 7.523] |
| 0.5-0.7 | 18.6 | -6.917 [-7.479, -6.367] | 1.956 [1.835, 2.079] | 26.0 [25.3, 26.6] | -6.136 [-7.710, -4.247] | 9.355 [7.803, 11.171] |
| 0.7-0.9 | 15.5 | -3.043 [-3.565, -2.558] | 1.975 [1.774, 2.188] | 21.7 [20.8, 22.6] | 0.093 [-4.253, 7.265] | 10.576 [6.452, 17.091] |
| 0.9-1 | 16.8 | -1.381 [-1.877, -0.953] | 1.311 [1.031, 1.617] | 19.2 [17.9, 20.4] | 2.263 [-2.770, 12.605] | 8.081 [2.584, 19.124] |
| all | 24.3 | -8.742 [-9.286, -8.192] | 1.618 [1.546, 1.696] | 30.9 [30.3, 31.4] | -7.571 [-8.138, -6.999] | 7.620 [7.102, 8.197] |

2. KSTAR K2 with the T_i level divided out: median T_i level by BES fifth (eV) 446.9 [413.1, 480.1], 594.3 [562.3, 621.8], 608.2 [576.7, 637.2], 641.5 [611.1, 670.4], 752.9 [708.7, 798.6]; (robust N_lo / level^2) lowest fifth over highest fifth 0.614 [0.460, 0.796].

## Checks before the read-out

- Identity on the KSTAR cut triples: -cov(d_t, d_t+1) = 18762.3294 eV^2 and [var(sd) - var(span)] / 4 = 18762.3294 eV^2 (the moment N_lo equals the variance form).
- Histogram approximation of the robust scale with unit weights against the exact value (KSTAR cut): S^2(sd) 10966.05 vs 10967.73; S^2(span) 4850.19 vs 4849.77 eV^2.
- Same check on MAST population G pooled: S^2(sd) 2574.6 vs 2574.6; S^2(span) 8849.2 vs 8849.1 eV^2.

## Expectations of the registration (section 7)

| expectation | observed | met |
|---|---|---|
| KSTAR robust sqrt(N_lo) 40 to 60 eV | 39.1 eV | no |
| N_hi / N_lo 1.05 to 1.25 (K1 holds) | 1.195, K1 holds | yes |
| K2 ratio 1.3 to 3 (holds) | 0.216, K2 does not hold | no |
| sigma / level 0.06 to 0.15 (robust sqrt(N_lo), each third) | 0.078, 0.055, 0.092 | no |
| sigma / level largest in the lowest third | 0.078, 0.055, 0.092 | no |
| causal variant at 2.5 to 5 floors of the robust bound (MSE / N_lo, each split) | 17.18, 19.58, 17.10, 31.21 | no |
| MAST: M1 holds in at least four bins | 1 bins, M1 does not hold | no |
| MAST: N_lo negative in the innermost bin (0-0.3) | -2232.9 eV^2 | yes |

2 of 8 expectations met.

## Deviations and interpretations

- No deviation from the registered definitions, thresholds or populations.
- Interpretation (not stated in the registration): a triple is three consecutive grid rows with all three T_i values measured and both time steps equal to one frame (10 ms on KSTAR within 0.1 ms, as in B.37 part Q; 5 ms on MAST within 0.1 ms of the stored times). Rows separated by an absent frame (beam blip) do not form a triple.
- Interpretation: the clipped version clips the first differences at the 1st and 99th percentiles of both increments pooled (for N_lo, as in B.37 Q) and the second difference at its own percentiles (for N_hi), within each set of triples.
- Interpretation: the BES level of a row is the mean of the nine BES channels divided by the median of that mean over the rows of its discharge (first and last row dropped); fifths and T_i thirds are pooled quantiles over all triples of the population.
- Interpretation: the MAST reference uses the centre row of each triple (the median over all three rows is stored beside it in the json); MAST discharges are resampled among the shots that contribute triples to the population.
- Interpretation: the shot list is run_coupling.SHOTS_TXT (3,686 shots), which is the primary set of the MAST edge study.
- Interpretation: in the row-table section the bracket is computed on the triples of the test discharges of the split, from the cut-population grid, and resampled together with the test rows (same discharge weights).
- Interpretation: the expectation 'sigma / level 0.06 to 0.15' is checked with the robust sqrt(N_lo); the expectation '2.5 to 5 floors of the robust bound' with MSE of the causal variant over the robust N_lo.

Run time: 118 s.
