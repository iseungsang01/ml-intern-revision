# B118: fixed focused LSTM evaluation

Exploratory reused-TEST evaluation of one frozen B117 checkpoint; no new fit or model choice.
The original short fit comparator is the same single B116 LSTM seed, not an ensemble.
Frozen best epochs: focused 59, short fit 15.
All four original validation designs are listed; none is primary. Only the fixed first design was evaluated.

V1_r2_k0: EVALUATED: one fixed single fit.
V2_s1_k0: NOT RUN.
V3_q3: NOT RUN.
V4a: NOT RUN.

Macro MSE equally weights all eight scenario MSEs before computing skill = 1 - candidate/comparator.
Macro draws use the same global group multiplicities across overlapping scenario supports.
Pointwise 95% intervals condition on the fixed fits and selected checkpoints; no refitting, selection or repeated-exploration uncertainty is included.
No requirement that all individual scenarios or strata have positive skill. No latency, physical-truth or causal failure-condition claim.

| Scope | Model | MSE [95% cluster] | RMSE [95% cluster] | Bias |
|---|---|---|---|---:|
| overall_8rules | LSTM_focused | 50666.545 [36597.641, 68421.647] | 225.0923 [191.3051, 261.57532] | +8.287382 |
| overall_8rules | LSTM_short | 51007.189 [37006.866, 68611.497] | 225.84771 [192.37169, 261.93796] | +0.15842026 |
| overall_8rules | GP_fast_boost | 45313.666 [33240.001, 59587.405] | 212.87007 [182.3184, 244.10531] | +0.58454422 |
| overall_D10 | LSTM_focused | 36798.651 [26011.33, 50124.391] | 191.82974 [161.28028, 223.88477] | +4.1211517 |
| overall_D10 | LSTM_short | 38487.389 [27456.423, 52224.781] | 196.18203 [165.6998, 228.52742] | +0.95150107 |
| overall_D10 | GP_fast_boost | 31256.152 [22009.985, 42355.034] | 176.79409 [148.35763, 205.80339] | +1.3424822 |
| V1_r2_k0/D10 | LSTM_focused | 36798.651 [26108.239, 50084.72] | 191.82974 [161.58044, 223.79616] | +4.1211517 |
| V1_r2_k0/D10 | LSTM_short | 38487.389 [27729.532, 52298] | 196.18203 [166.52187, 228.68756] | +0.95150107 |
| V1_r2_k0/D10 | GP_fast_boost | 31256.152 [22212.658, 42416.805] | 176.79409 [149.03912, 205.95341] | +1.3424822 |
| V1_r2_k0/D50 | LSTM_focused | 41643.624 [29474.563, 56280.81] | 204.06769 [171.68157, 237.23577] | +3.3552365 |
| V1_r2_k0/D50 | LSTM_short | 43808.156 [31266.119, 59311.238] | 209.30398 [176.82228, 243.53899] | -1.6783631 |
| V1_r2_k0/D50 | GP_fast_boost | 38824.468 [27629.432, 52013.524] | 197.03925 [166.22103, 228.06474] | +1.0221426 |
| V1_r2_k0/D100 | LSTM_focused | 45790.677 [32530.767, 62704.407] | 213.98756 [180.36287, 250.40848] | +2.3063988 |
| V1_r2_k0/D100 | LSTM_short | 47698.049 [33995.399, 65230.856] | 218.39883 [184.37841, 255.40332] | -3.5788937 |
| V1_r2_k0/D100 | GP_fast_boost | 42264.672 [30453.937, 56848.564] | 205.58373 [174.51056, 238.42937] | +0.021991364 |
| V1_r2_k0/D300 | LSTM_focused | 53712.359 [38352.845, 73459.299] | 231.75927 [195.83882, 271.03376] | +2.2932064 |
| V1_r2_k0/D300 | LSTM_short | 54296.002 [39053.445, 74408.16] | 233.01503 [197.61945, 272.77859] | -4.7120672 |
| V1_r2_k0/D300 | GP_fast_boost | 48495.019 [35701.425, 63801.113] | 220.21585 [188.94821, 252.58882] | +2.2385361 |
| V1_r2_k0/D650 | LSTM_focused | 59911.81 [43172.854, 81423.351] | 244.76889 [207.78078, 285.34777] | +6.1828426 |
| V1_r2_k0/D650 | LSTM_short | 60080.177 [43498.099, 81688.223] | 245.11258 [208.56198, 285.81152] | -6.4861645 |
| V1_r2_k0/D650 | GP_fast_boost | 55809.391 [41219.308, 73141.991] | 236.24011 [203.02539, 270.44776] | -1.1431928 |
| V1_r2_k0/D1000 | LSTM_focused | 68852.067 [48682.451, 95469.331] | 262.39677 [220.641, 308.98112] | +13.10201 |
| V1_r2_k0/D1000 | LSTM_short | 64427.322 [46611.409, 88101.942] | 253.82538 [215.89676, 296.81971] | -3.3867508 |
| V1_r2_k0/D1000 | GP_fast_boost | 57293.341 [43048.194, 74145.796] | 239.36027 [207.48059, 272.29726] | +0.11410299 |
| V1_r2_k0/S | LSTM_focused | 57827.343 [40676.68, 78805.618] | 240.47316 [201.6846, 280.72338] | +23.385579 |
| V1_r2_k0/S | LSTM_short | 55787.154 [39734.132, 74378.207] | 236.19304 [199.33422, 272.72368] | +16.084473 |
| V1_r2_k0/S | GP_fast_boost | 51604.82 [36928.86, 69220.283] | 227.16694 [192.16883, 263.09748] | -1.5704624 |
| V1_r2_k0/K10 | LSTM_focused | 40795.831 [28865.786, 56408.294] | 201.97978 [169.89934, 237.5043] | +11.552631 |
| V1_r2_k0/K10 | LSTM_short | 43473.266 [30765.482, 60286.745] | 208.50244 [175.40092, 245.53359] | +4.0736278 |
| V1_r2_k0/K10 | GP_fast_boost | 36961.469 [25991.166, 50630.012] | 192.25366 [161.21776, 225.01114] | +2.6507536 |

| Scope | Paired MSE skill | Point [95% cluster] | Excludes zero |
|---|---|---|---|
| overall_8rules | LSTM_focused_over_GP_fast_boost | -0.118129 [-0.208432, -0.047248] | True |
| overall_8rules | LSTM_focused_over_LSTM_short | +0.006678 [-0.079949, +0.086047] | False |
| overall_D10 | LSTM_focused_over_GP_fast_boost | -0.177325 [-0.250246, -0.115735] | True |
| overall_D10 | LSTM_focused_over_LSTM_short | +0.043878 [-0.006283, +0.103322] | False |
| V1_r2_k0/D10 | LSTM_focused_over_GP_fast_boost | -0.177325 [-0.250649, -0.115447] | True |
| V1_r2_k0/D10 | LSTM_focused_over_LSTM_short | +0.043878 [-0.005482, +0.102681] | False |
| V1_r2_k0/D50 | LSTM_focused_over_GP_fast_boost | -0.072613 [-0.119444, -0.032795] | True |
| V1_r2_k0/D50 | LSTM_focused_over_LSTM_short | +0.049409 [-0.001349, +0.105422] | False |
| V1_r2_k0/D100 | LSTM_focused_over_GP_fast_boost | -0.083427 [-0.138909, -0.036075] | True |
| V1_r2_k0/D100 | LSTM_focused_over_LSTM_short | +0.039988 [-0.020522, +0.103463] | False |
| V1_r2_k0/D300 | LSTM_focused_over_GP_fast_boost | -0.107585 [-0.223227, -0.016019] | True |
| V1_r2_k0/D300 | LSTM_focused_over_LSTM_short | +0.010749 [-0.090167, +0.103994] | False |
| V1_r2_k0/D650 | LSTM_focused_over_GP_fast_boost | -0.073508 [-0.211322, +0.036335] | False |
| V1_r2_k0/D650 | LSTM_focused_over_LSTM_short | +0.002802 [-0.121577, +0.108384] | False |
| V1_r2_k0/D1000 | LSTM_focused_over_GP_fast_boost | -0.201746 [-0.410341, -0.037689] | True |
| V1_r2_k0/D1000 | LSTM_focused_over_LSTM_short | -0.068678 [-0.236247, +0.066826] | False |
| V1_r2_k0/S | LSTM_focused_over_GP_fast_boost | -0.120580 [-0.213111, -0.037231] | True |
| V1_r2_k0/S | LSTM_focused_over_LSTM_short | -0.036571 [-0.128796, +0.042015] | False |
| V1_r2_k0/K10 | LSTM_focused_over_GP_fast_boost | -0.103739 [-0.196651, -0.023944] | True |
| V1_r2_k0/K10 | LSTM_focused_over_LSTM_short | +0.061588 [+0.007828, +0.119457] | True |

Separate registered statements: {'R1': False, 'R2': False, 'R3': False}.

R1: V1_r2_k0, overall_8rules, focused versus GP_fast_boost: skill -0.118129 [-0.208432, -0.047248] (95% cluster); specified increment not established.
R2: V1_r2_k0, overall_D10, focused versus GP_fast_boost: skill -0.177325 [-0.250246, -0.115735] (95% cluster); specified increment not established.
R3: V1_r2_k0, overall_8rules, focused versus LSTM_short: skill +0.006678 [-0.079949, +0.086047] (95% cluster); specified increment not established.

Input-conditional strata retain frozen B116 TRAIN cuts, including available-CES-only age bins, history presence and elapsed-step bins.
These are descriptive associations with observed inputs. No residual-based bins or shot selection were made.

| Scenario | Stratum | Rows | Clusters | Focused MSE | Short MSE | GP+fast MSE | Focused/GP skill [95% cluster] |
|---|---|---:|---:|---:|---:|---:|---|
| V1_r2_k0/D10 | age/low | 27558 | 43 | 36746.427 | 38441.813 | 31248.176 | -0.17595 [-0.24737, -0.11499] |
| V1_r2_k0/D10 | age/middle | 8 | 7 | 216694.68 | 195486.62 | 58730.84 | -2.68962 descriptive only |
| V1_r2_k0/D10 | age/high | 0 | 0 | | | | unavailable |
| V1_r2_k0/D10 | fastchange/low | 12983 | 43 | 22715.15 | 24123.769 | 16523.335 | -0.37473 [-0.59351, -0.20691] |
| V1_r2_k0/D10 | fastchange/middle | 9758 | 41 | 34654.282 | 36495.2 | 29964.847 | -0.15650 [-0.24809, -0.09055] |
| V1_r2_k0/D10 | fastchange/high | 4825 | 40 | 79030.948 | 81165.664 | 73510.393 | -0.07510 [-0.16587, +0.00601] |
| V1_r2_k0/D10 | fastlevel/low | 8176 | 30 | 16819.124 | 18389.399 | 13264.771 | -0.26795 [-0.52102, -0.14624] |
| V1_r2_k0/D10 | fastlevel/middle | 9640 | 37 | 38724.436 | 43411.402 | 31764.429 | -0.21911 [-0.40158, -0.11109] |
| V1_r2_k0/D10 | fastlevel/high | 9750 | 40 | 51648.705 | 50472.383 | 45840.535 | -0.12670 [-0.21345, -0.04193] |
| V1_r2_k0/D10 | history/absent | 0 | 0 | | | | unavailable |
| V1_r2_k0/D10 | history/present | 27566 | 43 | 36798.651 | 38487.389 | 31256.152 | -0.17733 [-0.25141, -0.11584] |
| V1_r2_k0/D10 | dt/normal | 27396 | 43 | 36273.136 | 37980.686 | 30732.464 | -0.18029 [-0.25431, -0.11763] |
| V1_r2_k0/D10 | dt/large | 170 | 33 | 121486.9 | 120144.08 | 115650 | -0.05047 [-0.16747, +0.03514] |
| V1_r2_k0/D50 | age/low | 27461 | 43 | 40920.356 | 43202.384 | 38417.089 | -0.06516 [-0.10707, -0.02964] |
| V1_r2_k0/D50 | age/middle | 101 | 22 | 238272.17 | 209144.1 | 149319.01 | -0.59573 [-3.63807, +0.17370] |
| V1_r2_k0/D50 | age/high | 4 | 1 | 42190.382 | 27851.266 | 45595.271 | +0.07468 descriptive only |
| V1_r2_k0/D50 | fastchange/low | 11227 | 43 | 20677.652 | 21376.979 | 18245.604 | -0.13330 [-0.24272, -0.03991] |
| V1_r2_k0/D50 | fastchange/middle | 9152 | 42 | 38623.224 | 40972.048 | 37156.181 | -0.03948 [-0.09136, -0.00010] |
| V1_r2_k0/D50 | fastchange/high | 7187 | 42 | 78241.324 | 82460.008 | 73095.661 | -0.07040 [-0.15846, -0.01139] |
| V1_r2_k0/D50 | fastlevel/low | 8176 | 30 | 21170.795 | 23174.459 | 20165.607 | -0.04985 [-0.19561, +0.03226] |
| V1_r2_k0/D50 | fastlevel/middle | 9640 | 37 | 43374.016 | 49223.956 | 40268.95 | -0.07711 [-0.15536, -0.02035] |
| V1_r2_k0/D50 | fastlevel/high | 9750 | 40 | 57100.534 | 55756.134 | 53042.933 | -0.07650 [-0.13383, -0.02873] |
| V1_r2_k0/D50 | history/absent | 0 | 0 | | | | unavailable |
| V1_r2_k0/D50 | history/present | 27566 | 43 | 41643.624 | 43808.156 | 38824.468 | -0.07261 [-0.12133, -0.03277] |
| V1_r2_k0/D50 | dt/normal | 27396 | 43 | 41107.94 | 43250.527 | 38296.251 | -0.07342 [-0.12139, -0.03383] |
| V1_r2_k0/D50 | dt/large | 170 | 33 | 127970.73 | 133671.57 | 123948.23 | -0.03245 [-0.19327, +0.18694] |
| V1_r2_k0/D100 | age/low | 17295 | 43 | 38479.355 | 39808.695 | 36020.473 | -0.06826 [-0.14472, -0.01184] |
| V1_r2_k0/D100 | age/middle | 10263 | 43 | 58109.357 | 60991.991 | 52769.242 | -0.10120 [-0.18525, -0.03603] |
| V1_r2_k0/D100 | age/high | 8 | 2 | 48627.538 | 49031.264 | 65390.202 | +0.25635 descriptive only |
| V1_r2_k0/D100 | fastchange/low | 10031 | 43 | 19476.845 | 19941.965 | 17088.728 | -0.13975 [-0.26141, -0.05162] |
| V1_r2_k0/D100 | fastchange/middle | 9149 | 42 | 38161.661 | 39722.437 | 37094.018 | -0.02878 [-0.09902, +0.03001] |
| V1_r2_k0/D100 | fastchange/high | 8386 | 42 | 85589.38 | 89600.048 | 78020.24 | -0.09702 [-0.17629, -0.03597] |
| V1_r2_k0/D100 | fastlevel/low | 8176 | 30 | 24536.336 | 26217.052 | 24266.362 | -0.01113 [-0.18215, +0.09587] |
| V1_r2_k0/D100 | fastlevel/middle | 9640 | 37 | 48437.491 | 54338.524 | 43468.874 | -0.11430 [-0.19164, -0.04122] |
| V1_r2_k0/D100 | fastlevel/high | 9750 | 40 | 60996.852 | 59145.686 | 56166.792 | -0.08599 [-0.15017, -0.03529] |
| V1_r2_k0/D100 | history/absent | 0 | 0 | | | | unavailable |
| V1_r2_k0/D100 | history/present | 27566 | 43 | 45790.677 | 47698.049 | 42264.672 | -0.08343 [-0.13958, -0.03634] |
| V1_r2_k0/D100 | dt/normal | 27396 | 43 | 45281.85 | 47119.47 | 41783.453 | -0.08373 [-0.14019, -0.03651] |
| V1_r2_k0/D100 | dt/large | 170 | 33 | 127789.64 | 140937.78 | 119814.49 | -0.06656 [-0.21907, +0.05524] |
| V1_r2_k0/D300 | age/low | 0 | 0 | | | | unavailable |
| V1_r2_k0/D300 | age/middle | 27520 | 43 | 53721.444 | 54302.772 | 48438.463 | -0.10907 [-0.22366, -0.01647] |
| V1_r2_k0/D300 | age/high | 46 | 5 | 48277.684 | 50245.691 | 82330.544 | +0.41361 descriptive only |
| V1_r2_k0/D300 | fastchange/low | 8366 | 43 | 23237.54 | 23618.853 | 22177.136 | -0.04782 [-0.14871, +0.03295] |
| V1_r2_k0/D300 | fastchange/middle | 9174 | 43 | 38170.988 | 40595.646 | 37802.064 | -0.00976 [-0.06071, +0.03510] |
| V1_r2_k0/D300 | fastchange/high | 10026 | 42 | 93362.158 | 92430.063 | 80239.741 | -0.16354 [-0.34915, -0.02478] |
| V1_r2_k0/D300 | fastlevel/low | 8176 | 30 | 29901.632 | 30967.738 | 31829.318 | +0.06056 [-0.17076, +0.20461] |
| V1_r2_k0/D300 | fastlevel/middle | 9640 | 37 | 57060.545 | 63930.387 | 50709.494 | -0.12524 [-0.23410, -0.02673] |
| V1_r2_k0/D300 | fastlevel/high | 9750 | 40 | 70368.769 | 64332.557 | 60280.786 | -0.16735 [-0.36726, -0.02414] |
| V1_r2_k0/D300 | history/absent | 0 | 0 | | | | unavailable |
| V1_r2_k0/D300 | history/present | 27566 | 43 | 53712.359 | 54296.002 | 48495.019 | -0.10759 [-0.22280, -0.01550] |
| V1_r2_k0/D300 | dt/normal | 27396 | 43 | 53254.364 | 53775.021 | 48085.674 | -0.10749 [-0.22303, -0.01361] |
| V1_r2_k0/D300 | dt/large | 170 | 33 | 127519.63 | 138253.56 | 114462.12 | -0.11408 [-0.38429, +0.07720] |
| V1_r2_k0/D650 | age/low | 0 | 0 | | | | unavailable |
| V1_r2_k0/D650 | age/middle | 6402 | 36 | 52051.489 | 46027.323 | 41138.009 | -0.26529 [-0.70037, +0.06768] |
| V1_r2_k0/D650 | age/high | 21164 | 43 | 62289.517 | 64331.092 | 60247.407 | -0.03390 [-0.13047, +0.05419] |
| V1_r2_k0/D650 | fastchange/low | 7224 | 40 | 24847.679 | 24941.37 | 24019.095 | -0.03450 [-0.16777, +0.08717] |
| V1_r2_k0/D650 | fastchange/middle | 8539 | 43 | 39686.804 | 41077.139 | 40403.557 | +0.01774 [-0.05928, +0.08533] |
| V1_r2_k0/D650 | fastchange/high | 11803 | 42 | 96004.72 | 95334.746 | 86412.078 | -0.11101 [-0.30957, +0.04630] |
| V1_r2_k0/D650 | fastlevel/low | 8176 | 30 | 36296.585 | 38417.996 | 41673.534 | +0.12903 [-0.03421, +0.24134] |
| V1_r2_k0/D650 | fastlevel/middle | 9640 | 37 | 59802.961 | 70611.696 | 57542.275 | -0.03929 [-0.12277, +0.03906] |
| V1_r2_k0/D650 | fastlevel/high | 9750 | 40 | 79822.312 | 67832.601 | 65949.88 | -0.21035 [-0.49659, -0.00704] |
| V1_r2_k0/D650 | history/absent | 0 | 0 | | | | unavailable |
| V1_r2_k0/D650 | history/present | 27566 | 43 | 59911.81 | 60080.177 | 55809.391 | -0.07351 [-0.21739, +0.03793] |
| V1_r2_k0/D650 | dt/normal | 27396 | 43 | 59405.769 | 59570.903 | 55364.921 | -0.07299 [-0.21495, +0.03871] |
| V1_r2_k0/D650 | dt/large | 170 | 33 | 141461.86 | 142151.18 | 127436.96 | -0.11005 [-0.45477, +0.05077] |
| V1_r2_k0/D1000 | age/low | 0 | 0 | | | | unavailable |
| V1_r2_k0/D1000 | age/middle | 0 | 0 | | | | unavailable |
| V1_r2_k0/D1000 | age/high | 27566 | 43 | 68852.067 | 64427.322 | 57293.341 | -0.20175 [-0.40885, -0.03766] |
| V1_r2_k0/D1000 | fastchange/low | 6477 | 37 | 27757.353 | 27650.248 | 27379.112 | -0.01381 [-0.11640, +0.06509] |
| V1_r2_k0/D1000 | fastchange/middle | 8446 | 40 | 41702.874 | 42511.036 | 39667.209 | -0.05132 [-0.16964, +0.04703] |
| V1_r2_k0/D1000 | fastchange/high | 12643 | 43 | 108041.54 | 97909.095 | 84393.299 | -0.28021 [-0.58445, -0.05075] |
| V1_r2_k0/D1000 | fastlevel/low | 8176 | 30 | 43159.844 | 44851.955 | 43093.607 | -0.00154 [-0.26107, +0.15222] |
| V1_r2_k0/D1000 | fastlevel/middle | 9640 | 37 | 65163.485 | 73499.143 | 60131.838 | -0.08368 [-0.20419, +0.01224] |
| V1_r2_k0/D1000 | fastlevel/high | 9750 | 40 | 94043.61 | 71873.05 | 66394.255 | -0.41644 [-0.88806, -0.02476] |
| V1_r2_k0/D1000 | history/absent | 0 | 0 | | | | unavailable |
| V1_r2_k0/D1000 | history/present | 27566 | 43 | 68852.067 | 64427.322 | 57293.341 | -0.20175 [-0.41383, -0.03998] |
| V1_r2_k0/D1000 | dt/normal | 27396 | 43 | 68399.529 | 63971.752 | 56859.455 | -0.20296 [-0.41465, -0.03599] |
| V1_r2_k0/D1000 | dt/large | 170 | 33 | 141779.97 | 137843.74 | 127215.4 | -0.11449 [-0.39198, +0.00643] |
| V1_r2_k0/S | age/low | 1878 | 41 | 71151.774 | 63876.348 | 55399.189 | -0.28435 [-0.51470, -0.07695] |
| V1_r2_k0/S | age/middle | 11035 | 41 | 54365.533 | 55294.226 | 49636.674 | -0.09527 [-0.17518, -0.01281] |
| V1_r2_k0/S | age/high | 12318 | 38 | 58897.142 | 54995.463 | 52789.481 | -0.11570 [-0.27840, +0.02622] |
| V1_r2_k0/S | fastchange/low | 6430 | 39 | 19721.151 | 19004.011 | 20113.146 | +0.01949 [-0.15481, +0.16411] |
| V1_r2_k0/S | fastchange/middle | 8191 | 39 | 41784.908 | 43048.096 | 39830.213 | -0.04908 [-0.18002, +0.04760] |
| V1_r2_k0/S | fastchange/high | 10610 | 40 | 93305.797 | 87913.566 | 79779.868 | -0.16954 [-0.29439, -0.05660] |
| V1_r2_k0/S | fastlevel/low | 7765 | 29 | 36051.242 | 39290.922 | 34951.857 | -0.03145 [-0.20271, +0.09280] |
| V1_r2_k0/S | fastlevel/middle | 9007 | 36 | 57580.215 | 58258.697 | 52491.705 | -0.09694 [-0.21179, +0.01264] |
| V1_r2_k0/S | fastlevel/high | 8459 | 40 | 78080.009 | 68298.332 | 65947.186 | -0.18398 [-0.38384, -0.01668] |
| V1_r2_k0/S | history/absent | 0 | 0 | | | | unavailable |
| V1_r2_k0/S | history/present | 25231 | 41 | 57827.343 | 55787.154 | 51604.82 | -0.12058 [-0.21330, -0.03843] |
| V1_r2_k0/S | dt/normal | 25020 | 41 | 57335.579 | 55249.898 | 51270.12 | -0.11830 [-0.21149, -0.03410] |
| V1_r2_k0/S | dt/large | 211 | 37 | 116139.75 | 119494.02 | 91292.853 | -0.27217 [-0.76626, -0.08494] |
| V1_r2_k0/K10 | age/low | 9960 | 41 | 37196.243 | 39464.307 | 33479.735 | -0.11101 [-0.20364, -0.03169] |
| V1_r2_k0/K10 | age/middle | 455 | 33 | 119731.58 | 131441.86 | 112312.06 | -0.06606 [-0.30173, +0.15422] |
| V1_r2_k0/K10 | age/high | 12 | 4 | 35472.905 | 35434.015 | 69757.272 | +0.49148 descriptive only |
| V1_r2_k0/K10 | fastchange/low | 4047 | 41 | 24190.635 | 25383.549 | 19602.116 | -0.23408 [-0.43711, -0.04534] |
| V1_r2_k0/K10 | fastchange/middle | 3590 | 39 | 31741.041 | 33879.79 | 30020.089 | -0.05733 [-0.19247, +0.02438] |
| V1_r2_k0/K10 | fastchange/high | 2790 | 38 | 76533.437 | 82057.376 | 71073.602 | -0.07682 [-0.21199, +0.02477] |
| V1_r2_k0/K10 | fastlevel/low | 3156 | 28 | 26879.072 | 27891.887 | 22608.43 | -0.18890 [-0.46443, -0.02221] |
| V1_r2_k0/K10 | fastlevel/middle | 3609 | 32 | 43535.388 | 49965.529 | 38972.837 | -0.11707 [-0.23891, -0.02552] |
| V1_r2_k0/K10 | fastlevel/high | 3662 | 35 | 50089.721 | 50503.375 | 47349.006 | -0.05788 [-0.17780, +0.06769] |
| V1_r2_k0/K10 | history/absent | 0 | 0 | | | | unavailable |
| V1_r2_k0/K10 | history/present | 10427 | 41 | 40795.831 | 43473.266 | 36961.469 | -0.10374 [-0.19700, -0.02516] |
| V1_r2_k0/K10 | dt/normal | 10375 | 41 | 40457.351 | 43085.856 | 36645.635 | -0.10402 [-0.19308, -0.02739] |
| V1_r2_k0/K10 | dt/large | 52 | 14 | 108328.92 | 120769.17 | 99976.374 | -0.08355 [-1.18581, +0.27961] |

Trace: D10 anonymous discharge 335, all 147 supported observations.
discharge support count closest to 150 D10 rows; lowest anonymous code breaks ties. segments ordered by their earliest physical time, then increasing actual time within segment; lines never connect segments.
The figure plots the measured CES observations and all three predictions on exactly these rows.
JSON retains discharge intervals, stratum RMSE/bias/counts and all accepted/rejected bootstrap counts.
An adequately optimized matched evaluation on unused shots could overturn an unfavorable result.
