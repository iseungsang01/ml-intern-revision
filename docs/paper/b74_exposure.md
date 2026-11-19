# B.74 what is inside the exposure of a frame whose reported T_i departs (registration PREREGISTRATION_B74.md, addendum 1)

Raw BES stream (with raw ECEI and Mirnov) of the twelve discharges with a raw delivery; nothing is trained, no estimator is involved. Eleven discharges enter the statements; 31921 (B.31's discovery discharge) is reported apart. Every interval is a 95 % interval over 10,000 resamples of the eleven discharges (seed 20261174); with eleven units the intervals are coarse. Classes of a row from the raw stream: B = a burst onset inside the window; P = none inside, one in the 10 ms before; Q = none inside and none in the 30 ms before. W10 = [t - 2 ms, t + 8 ms), W5 = [t, t + 5 ms). Ratio = value of the row / median of the ten preceding regular rows.

## Statements

| Id | Statement (as registered) | Statistic | Point [95 % interval] | Expectation | Verdict |
|---|---|---|---|---|---|
| J1 | W10: share of rows without a usable value higher in class B than in class Q (interval of the difference above zero) | W10, share without a usable T_i, class B minus class Q | +0.101 [+0.017, +0.192] | holds | holds |
| J2 | W10, rows with a value: median T_i ratio of class B above that of class Q (interval of the difference above zero) | W10, median T_i ratio, class B minus class Q | +0.099 [+0.021, +0.187] | uncertain; holds if bursts are what raises the width for one frame | holds |
| J3 | W10: in class B the median following-level ratio does not lie above the class-Q value (interval of the difference contains zero or lies below) | W10, median following-level ratio, class B minus class Q | -0.014 [-0.096, +0.041] | holds if J2 holds | holds |
| J4 | W10: median ratio of fresh V_rot in class B within 0.03 of that of class Q (interval inside +-0.03) | W10, median fresh V_rot ratio, class B minus class Q | +0.006 [-0.025, +0.025] | holds | holds |
| J5 | Part D, 5 ms windows: largest rank correlation at a start between -2 and +5 ms, exceeding the values at -10 and +10 ms (intervals of both differences above zero) | part D, 5 ms windows, classes of W10: start of the largest rank correlation; differences to the starts -10 and +10 ms | start -4 ms, rho 0.451; minus rho(-10 ms): +0.090 [+0.026, +0.159]; minus rho(+10 ms): +0.163 [+0.073, +0.277] | holds | does not hold |
| J6 | Part E: class B over-represented among the upward one-frame departures (interval of the share ratio above 1) | W10, share of class B among upward one-frame departures / share of class B among all rows with a T_i ratio | 9.48 [4.65, 18.68] | uncertain | holds |

Reading of the registration: J2 with J3 and J4 is the reading "a burst of light in the exposure raises the reported width for that frame only, and not the shift". If J2 does not hold, the one-frame rises are not explained by bursts as B.31 defines them.

## A. Share of rows without a usable T_i, by class

| window | class B | class P | class Q | B minus Q | rows B / P / Q / O |
|---|---|---|---|---|---|
| W10 | 0.144 [0.051, 0.236] | 0.173 [0.042, 0.301] | 0.044 [0.008, 0.102] | +0.101 [+0.017, +0.192] | 291 / 208 / 4706 / 304 |
| W5 | 0.140 [0.033, 0.236] | 0.159 [0.049, 0.263] | 0.045 [0.009, 0.103] | +0.095 [-0.001, +0.182] | 164 / 232 / 4774 / 339 |

Per discharge (W10): share without a usable T_i

| discharge | B | P | Q |
|---|---|---|---|
| 31097 | 0.000 | 0.000 | 0.000 |
| 31114 | 0.000 | 0.000 | 0.000 |
| 31357 | - | - | 0.000 |
| 31359 | 0.381 | 0.520 | 0.054 |
| 31747 | 0.182 | 0.171 | 0.119 |
| 31873 | 0.047 | 0.000 | 0.014 |
| 31902 | 0.000 | 0.000 | 0.004 |
| 31923 | 0.186 | 0.353 | 0.038 |
| 32027 | 0.000 | 0.000 | 0.000 |
| 32092 | 0.250 | 0.286 | 0.239 |
| 32097 | 0.000 | 0.000 | 0.024 |
| 31921 (discovery) | 0.250 | 0.571 | 0.049 |

## B. Rows with a value: median ratios by class

### W10

| quantity | class B | class P | class Q | B minus Q | rows B / P / Q |
|---|---|---|---|---|---|
| T_i ratio | 1.105 [1.030, 1.188] | 0.930 [0.802, 1.028] | 1.006 [1.001, 1.014] | +0.099 [+0.021, +0.187] | 212 / 149 / 4210 |
| following level / level before | 0.990 [0.909, 1.042] | 0.971 [0.898, 1.018] | 1.004 [0.998, 1.007] | -0.014 [-0.096, +0.041] | 200 / 143 / 4177 |
| fresh V_rot ratio | 1.008 [0.963, 1.026] | 1.011 [0.915, 1.058] | 1.001 [0.989, 1.014] | +0.006 [-0.025, +0.025] | 62 / 38 / 879 |
| ECEI ratio (sample at the stamp) | 0.996 [0.976, 1.012] | 0.989 [0.883, 1.009] | 1.003 [0.998, 1.010] | -0.007 [-0.030, +0.010] | 157 / 108 / 3687 |

| share of rows | class B | class P | class Q |
|---|---|---|---|
| T_i ratio above 1.35 | 0.259 [0.126, 0.372] | 0.101 [0.032, 0.178] | 0.042 [0.021, 0.071] |
| T_i ratio below 0.74 | 0.179 [0.080, 0.289] | 0.242 [0.108, 0.389] | 0.045 [0.025, 0.071] |

### W5

| quantity | class B | class P | class Q | B minus Q | rows B / P / Q |
|---|---|---|---|---|---|
| T_i ratio | 1.097 [1.035, 1.267] | 0.942 [0.802, 1.080] | 1.006 [1.001, 1.014] | +0.091 [+0.022, +0.264] | 119 / 168 / 4259 |
| following level / level before | 0.954 [0.879, 1.029] | 0.980 [0.900, 1.028] | 1.004 [0.998, 1.006] | -0.049 [-0.126, +0.029] | 111 / 159 / 4224 |
| fresh V_rot ratio | 1.008 [0.950, 1.092] | 1.005 [0.915, 1.017] | 1.002 [0.990, 1.019] | +0.006 [-0.044, +0.093] | 38 / 47 / 891 |
| ECEI ratio (sample at the stamp) | 0.995 [0.951, 1.010] | 0.982 [0.891, 1.006] | 1.003 [0.997, 1.010] | -0.008 [-0.057, +0.008] | 92 / 125 / 3725 |

| share of rows | class B | class P | class Q |
|---|---|---|---|
| T_i ratio above 1.35 | 0.261 [0.113, 0.383] | 0.137 [0.052, 0.213] | 0.044 [0.022, 0.074] |
| T_i ratio below 0.74 | 0.176 [0.065, 0.299] | 0.250 [0.113, 0.400] | 0.046 [0.026, 0.072] |

Per discharge (W10), medians of class B / class Q

| discharge | T_i ratio | following-level ratio | fresh V_rot ratio | ECEI ratio | above 1.35 B / Q |
|---|---|---|---|---|---|
| 31097 | 1.063 / 1.004 | 0.806 / 1.005 | - / - | 0.912 / 0.999 | 0.000 / 0.014 |
| 31114 | 1.016 / 1.002 | 0.920 / 1.000 | 1.089 / 0.994 | 0.990 / 1.006 | 0.067 / 0.021 |
| 31357 | - / 1.002 | - / 0.986 | - / 1.059 | - / 1.006 | - / 0.026 |
| 31359 | 0.975 / 0.995 | 1.159 / 1.001 | 0.984 / 0.996 | 1.140 / 1.017 | 0.333 / 0.057 |
| 31747 | 1.043 / 1.061 | 0.908 / 1.047 | 0.855 / 0.894 | - / - | 0.045 / 0.086 |
| 31873 | 1.329 / 0.997 | 1.115 / 1.000 | - / - | 1.008 / 1.001 | 0.459 / 0.020 |
| 31902 | 1.321 / 1.001 | 1.003 / 0.997 | 1.015 / 1.003 | 1.147 / 0.997 | 0.500 / 0.009 |
| 31923 | 1.107 / 1.021 | 0.772 / 1.001 | - / - | 0.997 / 1.008 | 0.381 / 0.106 |
| 32027 | 1.044 / 1.014 | 0.935 / 1.008 | - / 1.018 | - / - | 0.091 / 0.003 |
| 32092 | 1.076 / 1.012 | 1.004 / 1.013 | - / 0.997 | 0.970 / 0.987 | 0.133 / 0.048 |
| 32097 | 1.193 / 0.988 | 0.973 / 0.986 | 1.009 / 1.066 | 0.968 / 1.051 | 0.381 / 0.131 |
| 31921 (discovery) | 1.698 / 0.994 | 1.349 / 0.995 | 0.984 / 1.008 | 1.004 / 1.003 | 0.607 / 0.028 |

## C. Dose: medians by bins of the excess light (all classes, rows with a value)

### W10 (quartile edges of the positive values: 0.0052, 0.0081, 0.0135; rows 4790, of which zero 119)

| bin | rows | median T_i ratio | median following-level ratio |
|---|---|---|---|
| zero | 119 | 0.949 [0.567, 0.973] | 0.994 [0.899, 1.005] |
| Q1 | 1168 | 0.980 [0.969, 0.993] | 1.004 [0.996, 1.009] |
| Q2 | 1168 | 1.006 [0.993, 1.021] | 1.005 [0.996, 1.011] |
| Q3 | 1167 | 1.008 [0.997, 1.013] | 1.004 [0.989, 1.009] |
| Q4 | 1168 | 1.044 [1.029, 1.083] | 0.998 [0.992, 1.007] |

### W5 (quartile edges of the positive values: 0.0033, 0.0078, 0.0145; rows 4790, of which zero 298)

| bin | rows | median T_i ratio | median following-level ratio |
|---|---|---|---|
| zero | 298 | 0.956 [0.888, 0.984] | 1.005 [0.983, 1.024] |
| Q1 | 1123 | 0.990 [0.968, 0.999] | 1.006 [0.999, 1.010] |
| Q2 | 1123 | 1.000 [0.991, 1.012] | 1.005 [0.996, 1.011] |
| Q3 | 1123 | 1.010 [1.002, 1.023] | 1.002 [0.988, 1.009] |
| Q4 | 1123 | 1.040 [1.027, 1.065] | 0.998 [0.989, 1.006] |

## D. Where the association sits: rank correlation of the T_i ratio and the window-mean BES ratio (class Q and B rows of W10; 4422 rows)

| start (ms from the stamp) | 1 ms window | 2 ms window | 5 ms window |
|---|---|---|---|
| -10 | +0.320 [+0.249, +0.386] | +0.325 [+0.255, +0.390] | +0.361 [+0.312, +0.411] |
| -9 | +0.325 [+0.256, +0.389] | +0.335 [+0.272, +0.396] | +0.376 [+0.338, +0.421] |
| -8 | +0.338 [+0.280, +0.395] | +0.347 [+0.298, +0.400] | +0.393 [+0.357, +0.434] |
| -7 | +0.350 [+0.309, +0.398] | +0.366 [+0.330, +0.410] | +0.411 [+0.370, +0.457] |
| -6 | +0.374 [+0.338, +0.417] | +0.389 [+0.351, +0.434] | +0.430 [+0.381, +0.481] |
| -5 | +0.392 [+0.348, +0.444] | +0.403 [+0.354, +0.459] | +0.444 [+0.389, +0.499] |
| -4 | +0.408 [+0.352, +0.464] | +0.416 [+0.359, +0.474] | +0.451 [+0.394, +0.504] |
| -3 | +0.410 [+0.352, +0.470] | +0.424 [+0.363, +0.481] | +0.446 [+0.393, +0.495] |
| -2 | +0.421 [+0.364, +0.477] | +0.427 [+0.369, +0.481] | +0.443 [+0.396, +0.485] |
| -1 | +0.415 [+0.358, +0.467] | +0.424 [+0.371, +0.474] | +0.431 [+0.385, +0.472] |
| +0 | +0.419 [+0.370, +0.469] | +0.417 [+0.368, +0.465] | +0.414 [+0.364, +0.460] |
| +1 | +0.400 [+0.349, +0.450] | +0.400 [+0.349, +0.449] | +0.391 [+0.332, +0.448] |
| +2 | +0.387 [+0.332, +0.439] | +0.382 [+0.324, +0.437] | +0.369 [+0.293, +0.438] |
| +3 | +0.363 [+0.301, +0.422] | +0.360 [+0.293, +0.424] | +0.350 [+0.261, +0.428] |
| +4 | +0.344 [+0.268, +0.415] | +0.338 [+0.258, +0.413] | +0.328 [+0.230, +0.411] |
| +5 | +0.320 [+0.233, +0.402] | +0.316 [+0.221, +0.402] | +0.314 [+0.205, +0.401] |
| +6 | +0.302 [+0.202, +0.392] | +0.301 [+0.197, +0.391] | +0.305 [+0.191, +0.392] |
| +7 | +0.288 [+0.178, +0.380] | +0.282 [+0.168, +0.372] | +0.298 [+0.181, +0.386] |
| +8 | +0.277 [+0.162, +0.365] | +0.281 [+0.165, +0.367] | +0.298 [+0.184, +0.386] |
| +9 | +0.278 [+0.163, +0.364] | +0.287 [+0.172, +0.370] | +0.295 [+0.185, +0.383] |
| +10 | +0.285 [+0.174, +0.363] | +0.285 [+0.177, +0.368] | +0.287 [+0.185, +0.372] |
| +11 | +0.277 [+0.171, +0.362] | +0.279 [+0.175, +0.365] | +0.275 [+0.182, +0.354] |
| +12 | +0.269 [+0.174, +0.349] | +0.264 [+0.172, +0.340] | +0.263 [+0.183, +0.335] |
| +13 | +0.248 [+0.165, +0.315] | +0.250 [+0.168, +0.317] | +0.254 [+0.187, +0.318] |
| +14 | +0.245 [+0.167, +0.309] | +0.244 [+0.173, +0.305] | +0.247 [+0.196, +0.303] |

- 5 ms windows: largest at start -4 ms (rho +0.451); minus rho(-10 ms) +0.090 [+0.026, +0.159]; minus rho(+10 ms) +0.163 [+0.073, +0.277]; share of draws whose own maximum lies in [-2, +5] ms: 0.276; quantiles 2.5/50/97.5 % of that start: -4, -4, -1 ms.
- Largest point estimate for 1 ms windows at -2 ms, for 2 ms windows at -2 ms.
- Tied values of the T_i ratio among these rows: 0.

Per discharge (5 ms windows, W10 classes): rank correlation at starts -10 / s* / +10 ms

| discharge | -10 ms | s* | +10 ms |
|---|---|---|---|
| 31097 | +0.499 | +0.585 | +0.384 |
| 31114 | +0.307 | +0.621 | +0.458 |
| 31357 | +0.433 | +0.467 | +0.459 |
| 31359 | +0.316 | +0.351 | +0.394 |
| 31747 | +0.401 | +0.410 | +0.290 |
| 31873 | +0.330 | +0.379 | +0.307 |
| 31902 | +0.210 | +0.496 | -0.048 |
| 31923 | +0.450 | +0.423 | +0.238 |
| 32027 | +0.474 | +0.603 | +0.213 |
| 32092 | +0.317 | +0.235 | +0.206 |
| 32097 | +0.365 | +0.470 | +0.449 |
| 31921 (discovery) | +0.276 | +0.359 | +0.282 |

## D. Where the association sits: rank correlation of the T_i ratio and the window-mean BES ratio (class Q and B rows of W5; 4378 rows) (beside)

| start (ms from the stamp) | 1 ms window | 2 ms window | 5 ms window |
|---|---|---|---|
| -10 | +0.324 [+0.251, +0.391] | +0.328 [+0.257, +0.395] | +0.366 [+0.317, +0.418] |
| -9 | +0.329 [+0.260, +0.395] | +0.341 [+0.277, +0.403] | +0.384 [+0.347, +0.428] |
| -8 | +0.344 [+0.285, +0.403] | +0.353 [+0.303, +0.407] | +0.401 [+0.367, +0.442] |
| -7 | +0.356 [+0.314, +0.405] | +0.373 [+0.338, +0.417] | +0.420 [+0.379, +0.466] |
| -6 | +0.381 [+0.348, +0.422] | +0.396 [+0.357, +0.440] | +0.438 [+0.389, +0.492] |
| -5 | +0.399 [+0.354, +0.452] | +0.412 [+0.362, +0.468] | +0.450 [+0.394, +0.509] |
| -4 | +0.418 [+0.363, +0.474] | +0.425 [+0.367, +0.485] | +0.457 [+0.399, +0.513] |
| -3 | +0.420 [+0.361, +0.480] | +0.431 [+0.369, +0.491] | +0.451 [+0.395, +0.504] |
| -2 | +0.428 [+0.369, +0.488] | +0.434 [+0.375, +0.493] | +0.447 [+0.398, +0.492] |
| -1 | +0.422 [+0.364, +0.480] | +0.430 [+0.377, +0.483] | +0.435 [+0.388, +0.477] |
| +0 | +0.424 [+0.376, +0.474] | +0.420 [+0.370, +0.470] | +0.417 [+0.369, +0.463] |
| +1 | +0.405 [+0.352, +0.456] | +0.407 [+0.356, +0.455] | +0.394 [+0.336, +0.449] |
| +2 | +0.395 [+0.341, +0.446] | +0.387 [+0.330, +0.442] | +0.371 [+0.295, +0.440] |
| +3 | +0.366 [+0.304, +0.426] | +0.362 [+0.295, +0.426] | +0.351 [+0.264, +0.428] |
| +4 | +0.345 [+0.270, +0.416] | +0.338 [+0.260, +0.412] | +0.329 [+0.233, +0.411] |
| +5 | +0.320 [+0.235, +0.402] | +0.316 [+0.224, +0.403] | +0.314 [+0.208, +0.402] |
| +6 | +0.304 [+0.204, +0.394] | +0.303 [+0.200, +0.394] | +0.306 [+0.193, +0.393] |
| +7 | +0.291 [+0.182, +0.382] | +0.284 [+0.172, +0.375] | +0.299 [+0.183, +0.385] |
| +8 | +0.277 [+0.163, +0.366] | +0.282 [+0.168, +0.368] | +0.299 [+0.186, +0.386] |
| +9 | +0.278 [+0.165, +0.364] | +0.288 [+0.174, +0.370] | +0.295 [+0.187, +0.381] |
| +10 | +0.287 [+0.178, +0.366] | +0.286 [+0.179, +0.368] | +0.288 [+0.188, +0.371] |
| +11 | +0.278 [+0.173, +0.362] | +0.281 [+0.178, +0.366] | +0.277 [+0.186, +0.355] |
| +12 | +0.273 [+0.178, +0.353] | +0.265 [+0.176, +0.339] | +0.267 [+0.188, +0.338] |
| +13 | +0.250 [+0.169, +0.317] | +0.252 [+0.173, +0.317] | +0.258 [+0.193, +0.322] |
| +14 | +0.247 [+0.172, +0.309] | +0.249 [+0.179, +0.310] | +0.250 [+0.201, +0.306] |

- 5 ms windows: largest at start -4 ms (rho +0.457); minus rho(-10 ms) +0.091 [+0.026, +0.163]; minus rho(+10 ms) +0.169 [+0.078, +0.283]; share of draws whose own maximum lies in [-2, +5] ms: 0.220; quantiles 2.5/50/97.5 % of that start: -4, -4, -1 ms.
- Largest point estimate for 1 ms windows at -2 ms, for 2 ms windows at -2 ms.
- Tied values of the T_i ratio among these rows: 0.

## E. One-frame departures (ratio beyond the threshold, next measured value back within 15 % of the level before)

| window | direction | departures (in class B) | share of class B among them | share of class B among all rows with a value | share ratio |
|---|---|---|---|---|---|
| W10 | up | 31 (13) | 0.419 [0.224, 0.654] | 0.044 [0.025, 0.067] | 9.48 [4.65, 18.68] |
| W10 | down | 46 (4) | 0.087 [0.026, 0.152] | 0.044 [0.025, 0.067] | 1.96 [0.63, 4.25] |
| W5 | up | 31 (7) | 0.226 [0.111, 0.412] | 0.025 [0.014, 0.038] | 9.09 [4.13, 20.60] |
| W5 | down | 46 (1) | 0.022 [0.000, 0.080] | 0.025 [0.014, 0.038] | 0.88 [0.00, 4.40] |

This conditions on the reported value on purpose: it describes the target; no estimator is judged.

Per discharge (W10): share of class B among upward departures / among all rows with a value

| discharge | upward | all rows |
|---|---|---|
| 31097 | - | 0.008 |
| 31114 | 0.333 | 0.034 |
| 31357 | - | 0.000 |
| 31359 | 1.000 | 0.091 |
| 31747 | 0.000 | 0.130 |
| 31873 | 0.714 | 0.056 |
| 31902 | 0.200 | 0.012 |
| 31923 | 0.667 | 0.065 |
| 32027 | - | 0.033 |
| 32092 | 0.000 | 0.036 |
| 32097 | 0.222 | 0.077 |
| 31921 (discovery) | 0.667 | 0.074 |

## F. Reported beside

| discharge | rows (file / set / with T_i ratio) | bursts (raw stream) | W10 classes B / P / Q / O | W5 classes B / P / Q / O | rows left out (beam interruption) | kept rows with an interruption bin in W10 / in [-10, +19] ms |
|---|---|---|---|---|---|---|
| 31097 | 796 / 761 / 740 | 8 | 6 / 6 / 739 / 10 | 5 / 6 / 740 / 10 | 9 | 0 / 6 |
| 31114 | 506 / 464 / 443 | 16 | 16 / 15 / 407 / 26 | 9 / 15 / 414 / 26 | 16 | 1 / 7 |
| 31357 | 396 / 361 / 340 | 0 | 0 / 0 / 361 / 0 | 0 / 0 / 361 / 0 | 9 | 0 / 6 |
| 31359 | 297 / 265 / 165 | 51 | 42 / 25 / 166 / 32 | 25 / 30 / 172 / 38 | 6 | 0 / 3 |
| 31747 | 545 / 431 / 338 | 66 | 55 / 41 / 285 / 50 | 32 / 44 / 292 / 63 | 88 | 5 / 19 |
| 31873 | 762 / 715 / 658 | 52 | 43 / 33 / 592 / 47 | 24 / 35 / 602 / 54 | 22 | 2 / 14 |
| 31902 | 875 / 538 / 490 | 7 | 6 / 6 / 514 / 12 | 3 / 6 / 517 / 12 | 21 | 0 / 11 |
| 31921 (discovery) | 508 / 474 / 378 | 63 | 48 / 28 / 371 / 27 | 26 / 36 / 378 / 34 | 9 | 0 / 5 |
| 31923 | 447 / 415 / 323 | 54 | 43 / 34 / 291 / 47 | 23 / 37 / 303 / 52 | 6 | 0 / 3 |
| 32027 | 396 / 351 / 330 | 13 | 12 / 8 / 317 / 14 | 5 / 8 / 322 / 16 | 19 | 2 / 10 |
| 32092 | 645 / 607 / 414 | 30 | 24 / 14 / 543 / 26 | 11 / 18 / 551 / 27 | 12 | 0 / 8 |
| 32097 | 643 / 601 / 549 | 53 | 44 / 26 / 491 / 40 | 27 / 33 / 500 / 41 | 16 | 1 / 10 |

| window | Mirnov RMS ratio, class B | class Q | ECEI change across the burst onset, class B | ECEI change across the stamp, class Q |
|---|---|---|---|---|
| W10 | 0.960 [0.898, 0.995] (n 248) | 0.960 [0.902, 0.983] (n 4029) | +0.002 [-0.000, +0.016] (n 217) | +0.002 [+0.000, +0.006] (n 4096) |
| W5 | 1.004 [0.853, 1.240] (n 144) | 0.942 [0.827, 0.977] (n 4083) | +0.002 [-0.000, +0.016] (n 123) | +0.002 [+0.000, +0.006] (n 4148) |

Per discharge (W10): Mirnov RMS ratio B / Q; ECEI change across the onset (B) / across the stamp (Q)

| discharge | Mirnov B / Q | ECEI B / Q |
|---|---|---|
| 31097 | 1.412 / 0.986 | -0.131 / +0.002 |
| 31114 | 0.974 / 0.898 | -0.007 / +0.022 |
| 31357 | - / 1.011 | - / +0.000 |
| 31359 | 0.990 / 0.994 | +0.030 / +0.000 |
| 31747 | 0.961 / 0.929 | - / - |
| 31873 | 0.968 / 0.999 | -0.000 / +0.000 |
| 31902 | 0.815 / 0.663 | +0.007 / +0.016 |
| 31923 | 1.027 / 0.975 | +0.001 / +0.001 |
| 32027 | 0.776 / 0.890 | - / - |
| 32092 | 0.994 / 0.975 | +0.003 / +0.000 |
| 32097 | 0.421 / 0.443 | +0.003 / +0.003 |
| 31921 (discovery) | 1.019 / 1.002 | -0.005 / +0.001 |

## Checks (registration section 7)

| discharge | bursts (this run) | bursts (b31_bursts.md) | onsets from the recomputed residual equal | BES rows in raw span | share equal to a raw sample at the stamp |
|---|---|---|---|---|---|
| 31097 | 8 | 8 | yes | 794 | 1.0000 |
| 31114 | 16 | 16 | yes | 504 | 1.0000 |
| 31357 | 0 | 0 | yes | 394 | 1.0000 |
| 31359 | 51 | 51 | yes | 295 | 1.0000 |
| 31747 | 66 | 66 | yes | 543 | 1.0000 |
| 31873 | 52 | 52 | yes | 760 | 1.0000 |
| 31902 | 7 | 7 | yes | 583 | 1.0000 |
| 31921 | 63 | 63 | yes | 506 | 1.0000 |
| 31923 | 54 | 54 | yes | 445 | 1.0000 |
| 32027 | 13 | 13 | yes | 394 | 1.0000 |
| 32092 | 30 | 30 | yes | 643 | 1.0000 |
| 32097 | 53 | 53 | yes | 641 | 1.0000 |

## Limits

- Eleven discharges in the statements, one campaign, selected earlier for their published context; not a sample of the 639.
- A burst is defined on the beam-emission light; it is not confirmed as an ELM by D-alpha, which is not in the files.
- The exposure window is not known to better than a few ms; two windows are carried and part D reports where the association peaks.
- No spectrum, fit amplitude or fit error is available: an association with bursts says in which frames the width departs, not why.
- Exploratory: the reviewers have looked at these streams for other quantities.
