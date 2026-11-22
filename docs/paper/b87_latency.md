# B.87 matched regression at fixed reporting delays

Exploratory used-data timestamp-horizon comparison; not a physical-temperature or deployment validation.

Point estimates and 95% shot-cluster intervals below; JSON also contains discharge intervals. Positive skill means reduced MSE on identical rows. Delays refer to timestamps; exposure and processing add physical latency.

Intervals condition on the fitted models and do not include refitting or feature-choice uncertainty. This is an exploratory analysis of previously used data, with no multiplicity adjustment or new confirmation set.

## V1

Rows 222672; shots 635; clusters 86.

| Arm | RMSE (eV) | MAE (eV) |
|---|---|---|
| H0 | 182.94 [157.17, 210.27] | 87.99 [73.90, 104.11] |
| X0 | 174.80 [150.02, 201.07] | 82.53 [69.39, 97.23] |
| H1 | 178.04 [152.75, 205.19] | 84.61 [70.70, 100.70] |
| X1 | 171.78 [147.20, 197.89] | 80.69 [67.65, 95.52] |
| H2 | 175.95 [151.02, 202.66] | 83.20 [69.65, 98.84] |
| X2 | 170.68 [146.53, 197.14] | 79.96 [66.99, 94.68] |
| H5 | 175.87 [150.64, 202.47] | 83.19 [69.88, 98.73] |
| X5 | 170.63 [145.89, 197.05] | 79.92 [67.02, 94.45] |

| Pair | MSE skill | Excludes zero |
|---|---|---|
| X0 over H0 | 0.0870 [0.0589, 0.1218] | True |
| X0 over H1 | 0.0360 [0.0089, 0.0696] | True |
| X0 over H2 | 0.0130 [-0.0114, 0.0420] | False |
| X0 over H5 | 0.0122 [-0.0111, 0.0394] | False |
| X1 over X0 | 0.0342 [0.0260, 0.0448] | True |
| X2 over X0 | 0.0466 [0.0370, 0.0581] | True |
| X5 over X0 | 0.0471 [0.0373, 0.0593] | True |
| X1 over H1 | 0.0690 [0.0440, 0.1021] | True |
| X2 over H2 | 0.0591 [0.0375, 0.0872] | True |
| X5 over H5 | 0.0587 [0.0388, 0.0849] | True |

| Statement | Pair | MSE skill (95% cluster interval) | Excludes zero | Verdict (lower bound > 0) | Expectation |
|---|---|---|---|---|---|
| A | X0 over H1 | 0.0360 [0.0089, 0.0696] | True | holds | uncertain |
| B | X0 over H5 | 0.0122 [-0.0111, 0.0394] | False | does not hold | false |
| C | X5 over X0 | 0.0471 [0.0373, 0.0593] | True | holds | true |
| D | X5 over H5 | 0.0587 [0.0388, 0.0849] | True | holds | uncertain |

## V2

Rows 222672; shots 635; clusters 86.

| Arm | RMSE (eV) | MAE (eV) |
|---|---|---|
| H0 | 183.56 [157.95, 211.36] | 88.34 [74.21, 104.64] |
| X0 | 176.84 [151.64, 203.48] | 84.00 [70.27, 99.45] |
| H1 | 178.78 [153.29, 205.59] | 84.98 [70.98, 100.70] |
| X1 | 173.45 [148.61, 199.95] | 81.84 [68.48, 96.67] |
| H2 | 176.80 [151.45, 203.51] | 83.70 [70.08, 99.29] |
| X2 | 172.30 [147.65, 198.57] | 81.01 [67.91, 96.00] |
| H5 | 176.42 [151.09, 203.68] | 83.60 [69.89, 99.20] |
| X5 | 172.23 [147.24, 198.89] | 81.02 [67.87, 96.02] |

| Pair | MSE skill | Excludes zero |
|---|---|---|
| X0 over H0 | 0.0719 [0.0484, 0.1026] | True |
| X0 over H1 | 0.0216 [-0.0037, 0.0523] | False |
| X0 over H2 | -0.0004 [-0.0219, 0.0241] | False |
| X0 over H5 | -0.0047 [-0.0267, 0.0195] | False |
| X1 over X0 | 0.0379 [0.0278, 0.0489] | True |
| X2 over X0 | 0.0507 [0.0399, 0.0634] | True |
| X5 over X0 | 0.0514 [0.0410, 0.0640] | True |
| X1 over H1 | 0.0587 [0.0367, 0.0873] | True |
| X2 over H2 | 0.0503 [0.0319, 0.0737] | True |
| X5 over H5 | 0.0469 [0.0286, 0.0711] | True |

| Statement | Pair | MSE skill (95% cluster interval) | Excludes zero | Verdict (lower bound > 0) | Expectation |
|---|---|---|---|---|---|
| A | X0 over H1 | 0.0216 [-0.0037, 0.0523] | False | does not hold | uncertain |
| B | X0 over H5 | -0.0047 [-0.0267, 0.0195] | False | does not hold | false |
| C | X5 over X0 | 0.0514 [0.0410, 0.0640] | True | holds | true |
| D | X5 over H5 | 0.0469 [0.0286, 0.0711] | True | holds | uncertain |

## V3

Rows 161744; shots 508; clusters 65.

| Arm | RMSE (eV) | MAE (eV) |
|---|---|---|
| H0 | 206.06 [182.45, 231.10] | 101.46 [88.77, 114.93] |
| X0 | 201.35 [178.92, 226.15] | 98.12 [85.90, 110.80] |
| H1 | 201.40 [177.61, 226.32] | 97.65 [84.97, 111.01] |
| X1 | 197.17 [173.99, 221.97] | 95.17 [83.17, 107.67] |
| H2 | 199.15 [175.68, 223.68] | 96.28 [83.76, 109.58] |
| X2 | 196.02 [173.22, 220.63] | 94.34 [82.29, 107.11] |
| H5 | 198.94 [176.03, 223.42] | 96.47 [84.09, 109.60] |
| X5 | 195.75 [172.72, 220.64] | 94.15 [82.19, 106.79] |

| Pair | MSE skill | Excludes zero |
|---|---|---|
| X0 over H0 | 0.0452 [0.0235, 0.0718] | True |
| X0 over H1 | 0.0005 [-0.0272, 0.0283] | False |
| X0 over H2 | -0.0222 [-0.0483, 0.0009] | False |
| X0 over H5 | -0.0243 [-0.0493, -0.0021] | True |
| X1 over X0 | 0.0411 [0.0296, 0.0556] | True |
| X2 over X0 | 0.0522 [0.0401, 0.0677] | True |
| X5 over X0 | 0.0548 [0.0416, 0.0706] | True |
| X1 over H1 | 0.0416 [0.0209, 0.0661] | True |
| X2 over H2 | 0.0311 [0.0154, 0.0510] | True |
| X5 over H5 | 0.0319 [0.0154, 0.0531] | True |

| Statement | Pair | MSE skill (95% cluster interval) | Excludes zero | Verdict (lower bound > 0) | Expectation |
|---|---|---|---|---|---|
| A | X0 over H1 | 0.0005 [-0.0272, 0.0283] | False | does not hold | uncertain |
| B | X0 over H5 | -0.0243 [-0.0493, -0.0021] | True | does not hold | false |
| C | X5 over X0 | 0.0548 [0.0416, 0.0706] | True | holds | true |
| D | X5 over H5 | 0.0319 [0.0154, 0.0531] | True | holds | uncertain |

## V4a

Rows 20062; shots 95; clusters 19.

| Arm | RMSE (eV) | MAE (eV) |
|---|---|---|
| H0 | 248.49 [201.21, 311.29] | 114.50 [100.79, 130.09] |
| X0 | 246.08 [198.27, 308.24] | 110.67 [98.05, 126.04] |
| H1 | 243.21 [196.87, 304.05] | 110.07 [95.83, 125.57] |
| X1 | 242.61 [195.23, 303.56] | 107.97 [94.84, 123.35] |
| H2 | 242.48 [195.94, 303.54] | 108.74 [95.01, 123.91] |
| X2 | 240.84 [192.98, 302.78] | 106.97 [94.03, 122.62] |
| H5 | 242.32 [195.71, 303.14] | 108.75 [95.15, 123.85] |
| X5 | 241.17 [193.54, 303.17] | 107.00 [94.24, 122.26] |

| Pair | MSE skill | Excludes zero |
|---|---|---|
| X0 over H0 | 0.0193 [0.0017, 0.0361] | True |
| X0 over H1 | -0.0238 [-0.0549, 0.0008] | False |
| X0 over H2 | -0.0300 [-0.0660, -0.0012] | True |
| X0 over H5 | -0.0313 [-0.0658, -0.0029] | True |
| X1 over X0 | 0.0280 [0.0120, 0.0461] | True |
| X2 over X0 | 0.0421 [0.0212, 0.0666] | True |
| X5 over X0 | 0.0395 [0.0227, 0.0584] | True |
| X1 over H1 | 0.0049 [-0.0120, 0.0231] | False |
| X2 over H2 | 0.0134 [-0.0005, 0.0297] | False |
| X5 over H5 | 0.0095 [-0.0062, 0.0240] | False |

| Statement | Pair | MSE skill (95% cluster interval) | Excludes zero | Verdict (lower bound > 0) | Expectation |
|---|---|---|---|---|---|
| A | X0 over H1 | -0.0238 [-0.0549, 0.0008] | False | does not hold | uncertain |
| B | X0 over H5 | -0.0313 [-0.0658, -0.0029] | True | does not hold | false |
| C | X5 over X0 | 0.0395 [0.0227, 0.0584] | True | holds | true |
| D | X5 over H5 | 0.0095 [-0.0062, 0.0240] | False | does not hold | uncertain |

## V4b

Rows 20062; shots 95; clusters 19.

| Arm | RMSE (eV) | MAE (eV) |
|---|---|---|
| H0 | 246.26 [199.19, 306.58] | 114.43 [101.14, 130.45] |
| X0 | 242.36 [196.43, 303.58] | 109.92 [97.50, 125.10] |
| H1 | 241.16 [196.12, 302.18] | 109.50 [95.45, 124.65] |
| X1 | 238.34 [192.99, 298.59] | 107.24 [93.92, 122.55] |
| H2 | 240.45 [194.96, 300.48] | 108.47 [94.85, 123.68] |
| X2 | 238.19 [192.76, 299.02] | 106.66 [93.97, 121.70] |
| H5 | 239.73 [194.62, 300.09] | 108.48 [94.91, 123.50] |
| X5 | 237.59 [192.47, 298.49] | 106.65 [93.93, 121.34] |

| Pair | MSE skill | Excludes zero |
|---|---|---|
| X0 over H0 | 0.0314 [0.0099, 0.0456] | True |
| X0 over H1 | -0.0099 [-0.0348, 0.0061] | False |
| X0 over H2 | -0.0160 [-0.0440, 0.0027] | False |
| X0 over H5 | -0.0220 [-0.0482, -0.0036] | True |
| X1 over X0 | 0.0329 [0.0156, 0.0530] | True |
| X2 over X0 | 0.0341 [0.0190, 0.0525] | True |
| X5 over X0 | 0.0390 [0.0226, 0.0589] | True |
| X1 over H1 | 0.0233 [0.0033, 0.0395] | True |
| X2 over H2 | 0.0187 [0.0014, 0.0325] | True |
| X5 over H5 | 0.0178 [0.0015, 0.0323] | True |

| Statement | Pair | MSE skill (95% cluster interval) | Excludes zero | Verdict (lower bound > 0) | Expectation |
|---|---|---|---|---|---|
| A | X0 over H1 | -0.0099 [-0.0348, 0.0061] | False | does not hold | uncertain |
| B | X0 over H5 | -0.0220 [-0.0482, -0.0036] | True | does not hold | false |
| C | X5 over X0 | 0.0390 [0.0226, 0.0589] | True | holds | true |
| D | X5 over H5 | 0.0178 [0.0015, 0.0323] | True | holds | uncertain |

