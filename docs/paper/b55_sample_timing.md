# B.55 — the instant of the fast-channel sample (registered: PREREGISTRATION_B55.md)

CPU, nothing trained. T_i, cut population. The 40 estimator networks of the pooled cross-validation of B.39 (`data/.b39p_v2_cut_r{r}_f{k}`, causal variant, fast channels standardized with the statistics of the training discharges), each run forward over the discharges of its own test fold with changed fast-channel inputs. The twin reads no fast channel and the causal GP none either: their errors are the stored errors of the same rows. Change skill = 1 - SSE(moved samples) / SSE(file values) on the same rows. Equivalence margin 0.01: 'equivalent' means the 95 % interval of the change skill lies inside [-0.01, +0.01]. Intervals: two-level (a re-partition is drawn, then its discharges are resampled), 2,000 draws in part R, 20,000 in part G.

## Checks before any statement

- **(a)** (rule of addendum 1, the B.49 rule) file values, every network against its stored squared errors: the absolute error computed here lies within 0.5 eV of the stored one on at least 99.869 % of the rows of every network (required 99 %). Distribution of the difference over 1,536,645 rows of 40 networks: median 0.026 eV, 99th percentile 0.151 eV, maximum 2.566 eV; 345 rows above 0.5 eV (362,914 above 0.05 eV, the registered threshold that failed before the addendum). The stored errors come from the GPU, these from the CPU; the relative size (about 1e-4) points to the arithmetic of the two devices (inferred, not verified). Every change of an estimate and every change skill below compares two estimates computed on the CPU, so the device difference does not enter them. Grid T_i against stored y_true, largest difference 0.00024 eV -> **passes**.
- **(b)** nearest raw sample (within one raw sample) against the file value, rule of the file rules run again; rows and shares identical to `file_rules.json` on all 36 (discharge, diagnostic) pairs: **passes**.

| discharge | BES rows, share (recorded) / at nearest sample | ECEI | Mirnov |
|---|---|---|---|
| 31097 | 794, 1.000 (1.000) / 0.518 | 794, 1.000 (1.000) / 1.000 | 794, 1.000 (1.000) / 0.820 |
| 31114 | 504, 1.000 (1.000) / 0.502 | 504, 1.000 (1.000) / 1.000 | 504, 1.000 (1.000) / 0.897 |
| 31357 | 394, 1.000 (1.000) / 0.444 | 394, 1.000 (1.000) / 1.000 | 394, 1.000 (1.000) / 1.000 |
| 31359 | 295, 1.000 (1.000) / 0.542 | 295, 1.000 (1.000) / 1.000 | 295, 1.000 (1.000) / 1.000 |
| 31747 | 543, 1.000 (1.000) / 0.484 | 543, 1.000 (1.000) / 1.000 | 543, 0.980 (0.980) / 0.937 |
| 31873 | 760, 1.000 (1.000) / 0.520 | 742, 1.000 (1.000) / 0.000 | 198, 1.000 (1.000) / 1.000 |
| 31902 | 583, 1.000 (1.000) / 0.443 | 583, 1.000 (1.000) / 1.000 | 583, 1.000 (1.000) / 1.000 |
| 31921 | 506, 1.000 (1.000) / 0.518 | 506, 1.000 (1.000) / 1.000 | 506, 1.000 (1.000) / 0.945 |
| 31923 | 445, 1.000 (1.000) / 0.506 | 445, 1.000 (1.000) / 1.000 | 445, 1.000 (1.000) / 1.000 |
| 32027 | 394, 1.000 (1.000) / 0.401 | 394, 1.000 (1.000) / 1.000 | 394, 1.000 (1.000) / 1.000 |
| 32092 | 643, 1.000 (1.000) / 0.507 | 643, 1.000 (1.000) / 1.000 | 643, 0.992 (0.992) / 0.882 |
| 32097 | 641, 1.000 (1.000) / 0.509 | 641, 1.000 (1.000) / 1.000 | 641, 0.989 (0.989) / 0.878 |

- **(c)** at d = 10 ms, rows whose previous row is 10 ms earlier: the raw sample 10 ms before the stamp (within one raw sample) equals the file value of the previous row: BES 1.0000 of 6,445 rows (strict, held sample minus 10 ms: 0.2681); ECEI 1.0000 of 6,427 rows (strict, held sample minus 10 ms: 1.0000); MC 0.9961 of 5,888 rows (strict, held sample minus 10 ms: 0.8983) -> **passes** (every discharge and diagnostic at least its recorded share minus 0.0005).

| discharge | BES share (c) / recorded (b) | ECEI | Mirnov |
|---|---|---|---|
| 31097 | 1.000 / 1.000 (790) | 1.000 / 1.000 (790) | 1.000 / 1.000 (790) |
| 31114 | 1.000 / 1.000 (500) | 1.000 / 1.000 (500) | 1.000 / 1.000 (500) |
| 31357 | 1.000 / 1.000 (390) | 1.000 / 1.000 (390) | 1.000 / 1.000 (390) |
| 31359 | 1.000 / 1.000 (292) | 1.000 / 1.000 (292) | 1.000 / 1.000 (292) |
| 31747 | 1.000 / 1.000 (538) | 1.000 / 1.000 (538) | 0.980 / 0.980 (538) |
| 31873 | 1.000 / 1.000 (753) | 1.000 / 1.000 (735) | 1.000 / 1.000 (196) |
| 31902 | 1.000 / 1.000 (575) | 1.000 / 1.000 (575) | 1.000 / 1.000 (575) |
| 31921 | 1.000 / 1.000 (502) | 1.000 / 1.000 (502) | 1.000 / 1.000 (502) |
| 31923 | 1.000 / 1.000 (442) | 1.000 / 1.000 (442) | 1.000 / 1.000 (442) |
| 32027 | 1.000 / 1.000 (390) | 1.000 / 1.000 (390) | 1.000 / 1.000 (390) |
| 32092 | 1.000 / 1.000 (638) | 1.000 / 1.000 (638) | 0.992 / 0.992 (638) |
| 32097 | 1.000 / 1.000 (635) | 1.000 / 1.000 (635) | 0.989 / 0.989 (635) |

- **(d)** part G with no shift (estimates of this run, file values) in the pooled readout of B.39: P1 +0.0889 [+0.0637, +0.1125] against +0.0889 [+0.0637, +0.1125]; P2 +0.0481 [+0.0225, +0.0729] against +0.0481 [+0.0225, +0.0729]; largest difference 1.07e-05 -> **passes**.

All four checks pass.

## Registered statements

- **R1** (one raw sample earlier, equivalent): change skill -0.0049 [-0.0297, +0.0096] -> **does not hold**.
- **R2** (0.5 ms earlier, equivalent): change skill -0.0234 [-0.0892, +0.0133] -> **does not hold**.
- **R3** (2 ms earlier, equivalent): change skill -0.0071 [-0.0585, +0.0380] -> **does not hold**.
- **G1** (fast channels one row old, skill over the twin with the interval above zero): -0.0266 [-0.0495, -0.0080] -> **does not hold**.
- **G2** (fast channels one row old, equivalent): change skill -0.0785 [-0.0964, -0.0613] -> **does not hold**.

## Part R — raw streams, twelve discharges

Scored rows per re-partition: 4,706, 4,729, 4,616, 4,643, 4,721, 4,623, 4,751, 4,792 (discharges per re-partition: 12, 12, 12, 12, 12, 12, 12, 12); each discharge is estimated by eight networks, one per re-partition. Fold cover of the twelve discharges (test folds per re-partition): one in every re-partition for all twelve.

With the file values: RMS error 192.5 [135.2, 245.1] eV, skill over the causal GP +0.022 [-0.144, +0.157], over the twin +0.025 [-0.064, +0.109].

| variant | RMS change of the estimate (eV) | RMS error (eV) | change skill | inside [-0.01, +0.01] | skill over the causal GP | skill over the twin |
|---|---|---|---|---|---|---|
| file values (sample at the stamp) | 0 | 192.5 [135.2, 245.1] | 0 | - | +0.022 [-0.144, +0.157] | +0.025 [-0.064, +0.109] |
| one raw sample earlier | 12.6 [9.3, 19.5] | 193.0 [135.0, 246.3] | -0.0049 [-0.0297, +0.0096] | no | +0.017 [-0.144, +0.159] | +0.021 [-0.062, +0.101] |
| 0.1 ms earlier | 38.7 [25.8, 58.2] | 193.4 [135.7, 247.2] | -0.0093 [-0.0431, +0.0151] | no | +0.012 [-0.158, +0.160] | +0.016 [-0.075, +0.106] |
| 0.5 ms earlier | 46.1 [30.6, 65.7] | 194.7 [137.0, 248.2] | -0.0234 [-0.0892, +0.0133] | no | -0.001 [-0.173, +0.130] | +0.003 [-0.092, +0.081] |
| 1 ms earlier | 46.8 [33.3, 65.2] | 193.1 [137.2, 246.2] | -0.0064 [-0.0603, +0.0262] | no | +0.015 [-0.176, +0.147] | +0.019 [-0.094, +0.098] |
| 2 ms earlier | 53.1 [38.9, 69.2] | 193.2 [137.6, 245.4] | -0.0071 [-0.0585, +0.0380] | no | +0.015 [-0.145, +0.149] | +0.019 [-0.077, +0.106] |
| 5 ms earlier | 57.0 [42.2, 76.0] | 196.4 [140.5, 250.0] | -0.0407 [-0.1083, +0.0230] | no | -0.019 [-0.195, +0.116] | -0.014 [-0.109, +0.086] |
| 10 ms earlier | 53.5 [40.6, 67.1] | 200.0 [144.5, 252.2] | -0.0792 [-0.1637, -0.0369] | no | -0.056 [-0.250, +0.074] | -0.051 [-0.160, +0.029] |
| mean of the samples in the 0.1 ms up to the stamp | 37.1 [20.6, 52.7] | 191.3 [134.6, 243.9] | +0.0126 [-0.0297, +0.0503] | no | +0.034 [-0.142, +0.168] | +0.038 [-0.038, +0.109] |
| mean of the samples in the 1 ms up to the stamp | 40.0 [23.5, 55.3] | 190.3 [134.0, 242.7] | +0.0227 [-0.0206, +0.0652] | no | +0.044 [-0.126, +0.170] | +0.048 [-0.030, +0.120] |

Change of the fast-channel inputs on the replaced rows, RMS in training standard deviations (mean of the 40 networks' training statistics):

| variant | BES | ECEI | Mirnov |
|---|---|---|---|
| one raw sample earlier | 0.065 | 0.048 | 0.411 |
| 0.1 ms earlier | 0.103 | 0.089 | 2.549 |
| 0.5 ms earlier | 0.209 | 0.095 | 2.320 |
| 1 ms earlier | 0.207 | 0.138 | 2.531 |
| 2 ms earlier | 0.211 | 0.161 | 2.286 |
| 5 ms earlier | 0.236 | 0.137 | 2.400 |
| 10 ms earlier | 0.210 | 0.145 | 2.324 |
| mean of the samples in the 0.1 ms up to the stamp | 0.068 | 0.057 | 1.734 |
| mean of the samples in the 1 ms up to the stamp | 0.120 | 0.067 | 1.655 |

Sample-to-sample scatter of the raw streams at the held sample (RMS of the one-sample difference / sqrt 2), in training standard deviations, median (range) over the channels of the twelve discharges: BES 0.046 (0.036 to 0.054); ECEI 0.023 (0.008 to 0.078); MC 0.241 (0.086 to 0.671).

### Per discharge (pooled over its eight networks)

| discharge | rows | RMS error, file values (eV) | 1samp: RMS change (eV) / change skill | 0.1ms: RMS change (eV) / change skill | 0.5ms: RMS change (eV) / change skill | 1ms: RMS change (eV) / change skill | 2ms: RMS change (eV) / change skill | 5ms: RMS change (eV) / change skill | 10ms: RMS change (eV) / change skill | mean0.1ms: RMS change (eV) / change skill | mean1ms: RMS change (eV) / change skill |
|---|---|---|---|---|---|---|---|---|---|---|---|
| s31097 | 666 | 53.9 | 6.4 / +0.0130 | 12.7 / +0.0309 | 14.0 / +0.0384 | 14.5 / +0.0076 | 19.0 / -0.1010 | 25.0 / -0.2663 | 32.6 / -0.3734 | 8.6 / +0.0573 | 10.1 / +0.0506 |
| s31114 | 435 | 153.3 | 13.9 / -0.0010 | 34.8 / -0.0112 | 41.6 / +0.0009 | 50.7 / -0.0259 | 67.7 / -0.0505 | 71.7 / -0.1502 | 63.0 / -0.1935 | 21.9 / -0.0082 | 31.6 / +0.0272 |
| s31357 | 338 | 44.7 | 4.8 / -0.0077 | 10.8 / -0.0081 | 11.8 / -0.0180 | 12.0 / +0.0239 | 12.7 / -0.0048 | 12.9 / +0.0348 | 9.6 / +0.0062 | 9.2 / +0.0535 | 9.4 / +0.0565 |
| s31359 | 178 | 328.7 | 14.0 / +0.0085 | 35.0 / -0.0018 | 27.3 / -0.0525 | 45.9 / -0.0786 | 62.2 / +0.0960 | 77.2 / +0.1307 | 41.1 / -0.0907 | 19.5 / +0.0009 | 20.7 / -0.0069 |
| s31747 | 405 | 172.6 | 15.3 / +0.0010 | 78.6 / -0.0385 | 79.6 / -0.0995 | 83.6 / -0.0565 | 84.0 / -0.0653 | 86.7 / -0.1125 | 68.2 / -0.0668 | 78.7 / +0.0502 | 80.9 / +0.0548 |
| s31873 | 177 | 50.9 | 6.9 / +0.0027 | 9.8 / +0.0012 | 11.8 / -0.0477 | 12.5 / -0.0686 | 18.6 / -0.1315 | 20.2 / -0.0785 | 19.2 / -0.1194 | 7.2 / +0.0041 | 8.5 / -0.0147 |
| s31902 | 480 | 125.6 | 12.9 / +0.0133 | 33.1 / -0.0046 | 41.1 / +0.0221 | 49.7 / -0.0146 | 54.7 / -0.0260 | 71.8 / -0.1730 | 78.9 / -0.4526 | 22.7 / +0.0310 | 27.0 / +0.0612 |
| s31921 | 381 | 280.0 | 19.2 / -0.0352 | 28.6 / -0.0139 | 53.7 / -0.0300 | 43.5 / +0.0015 | 52.3 / +0.0065 | 41.8 / -0.0389 | 40.1 / -0.0184 | 24.6 / -0.0292 | 28.4 / -0.0067 |
| s31923 | 330 | 340.1 | 20.8 / -0.0019 | 37.3 / +0.0068 | 58.8 / -0.0202 | 46.9 / +0.0068 | 42.0 / -0.0086 | 50.9 / -0.0382 | 52.9 / -0.0445 | 41.6 / +0.0292 | 42.7 / +0.0205 |
| s32027 | 344 | 132.9 | 10.1 / +0.0039 | 25.2 / -0.0181 | 30.1 / -0.0747 | 27.9 / -0.0733 | 42.5 / -0.1376 | 57.6 / -0.2069 | 63.1 / -0.1188 | 15.3 / -0.0295 | 21.2 / -0.0346 |
| s32092 | 422 | 111.1 | 10.1 / +0.0036 | 51.8 / -0.0020 | 46.3 / +0.0111 | 50.1 / -0.0338 | 54.7 / -0.1044 | 57.3 / -0.0550 | 49.0 / -0.1586 | 42.4 / +0.0467 | 40.1 / +0.0588 |
| s32097 | 540 | 281.0 | 10.5 / -0.0013 | 44.2 / -0.0176 | 63.4 / +0.0022 | 58.4 / +0.0415 | 67.6 / +0.0051 | 60.7 / -0.0171 | 58.6 / -0.0279 | 57.1 / +0.0237 | 63.6 / +0.0451 |

Raw windows (intersection of the three diagnostics) and replaced rows: 31097 3.012-10.992 s, 793 rows; 31114 4.002-9.082 s, 503 rows; 31357 3.002-6.982 s, 393 rows; 31359 4.002-6.982 s, 294 rows; 31747 3.012-8.492 s, 542 rows; 31873 5.412-7.412 s, 197 rows; 31902 1.752-7.722 s, 582 rows; 31921 3.512-8.612 s, 505 rows; 31923 3.512-7.992 s, 444 rows; 32027 2.212-6.192 s, 393 rows; 32092 3.012-9.492 s, 642 rows; 32097 3.012-9.492 s, 640 rows.
Held BES sample, all twelve discharges: offset -1 on 2,263 rows, offset 0 on 3,210 rows, offset 1 on 1,017 rows, offset 99 on 0 rows of 6,490 (99 = no match).
Offset of the sample the file holds (rows of the full file inside the window; 99 = no match, nearest kept): 31097 BES {"-1": 255, "0": 411, "1": 127, "99": 0}, ECEI {"-1": 0, "0": 793, "1": 0, "99": 0}, MC {"-1": 71, "0": 650, "1": 72, "99": 0}; 31114 BES {"-1": 188, "0": 253, "1": 62, "99": 0}, ECEI {"-1": 0, "0": 503, "1": 0, "99": 0}, MC {"-1": 26, "0": 451, "1": 26, "99": 0}; 31357 BES {"-1": 164, "0": 174, "1": 55, "99": 0}, ECEI {"-1": 0, "0": 393, "1": 0, "99": 0}, MC {"-1": 0, "0": 393, "1": 0, "99": 0}; 31359 BES {"-1": 97, "0": 160, "1": 37, "99": 0}, ECEI {"-1": 0, "0": 294, "1": 0, "99": 0}, MC {"-1": 0, "0": 294, "1": 0, "99": 0}; 31747 BES {"-1": 0, "0": 263, "1": 279, "99": 0}, ECEI {"-1": 0, "0": 542, "1": 0, "99": 0}, MC {"-1": 11, "0": 508, "1": 12, "99": 11}; 31873 BES {"-1": 364, "0": 395, "1": 0, "99": 0}, ECEI {"-1": 741, "0": 0, "1": 0, "99": 0}, MC {"-1": 0, "0": 197, "1": 0, "99": 0}; 31902 BES {"-1": 275, "0": 258, "1": 49, "99": 0}, ECEI {"-1": 0, "0": 582, "1": 0, "99": 0}, MC {"-1": 0, "0": 582, "1": 0, "99": 0}; 31921 BES {"-1": 244, "0": 261, "1": 0, "99": 0}, ECEI {"-1": 0, "0": 505, "1": 0, "99": 0}, MC {"-1": 14, "0": 477, "1": 14, "99": 0}; 31923 BES {"-1": 107, "0": 225, "1": 112, "99": 0}, ECEI {"-1": 0, "0": 444, "1": 0, "99": 0}, MC {"-1": 0, "0": 444, "1": 0, "99": 0}; 32027 BES {"-1": 191, "0": 158, "1": 44, "99": 0}, ECEI {"-1": 0, "0": 393, "1": 0, "99": 0}, MC {"-1": 0, "0": 393, "1": 0, "99": 0}; 32092 BES {"-1": 189, "0": 326, "1": 127, "99": 0}, ECEI {"-1": 0, "0": 642, "1": 0, "99": 0}, MC {"-1": 35, "0": 566, "1": 36, "99": 5}; 32097 BES {"-1": 189, "0": 326, "1": 125, "99": 0}, ECEI {"-1": 0, "0": 640, "1": 0, "99": 0}, MC {"-1": 35, "0": 562, "1": 36, "99": 7}.

## Part G — fast channels one row old, all discharges

Rows per re-partition: 191,934, 192,092, 192,132, 192,032, 192,145, 192,056, 192,127, 192,127; discharges 635.

| rows | rows per re-partition | RMS change (eV) | RMS error, file values (eV) | RMS error, one row old (eV) | change skill | inside [-0.01, +0.01] | skill over causal GP: file / one row old | skill over the twin: file / one row old |
|---|---|---|---|---|---|---|---|---|
| all scored rows | 191,934 | 48.4 [42.7, 53.3] | 184.6 [174.5, 194.7] | 191.7 [181.6, 201.8] | -0.0785 [-0.0964, -0.0613] | no | +0.089 [+0.063, +0.112] / +0.017 [-0.004, +0.038] | +0.048 [+0.022, +0.073] / -0.027 [-0.050, -0.008] |
| rows 1 to 3 after an absent frame | 3,155 | 118.6 [109.0, 129.8] | 249.1 [221.3, 277.3] | 278.2 [252.7, 304.7] | -0.2474 [-0.3350, -0.1789] | no | +0.265 [+0.191, +0.335] / +0.083 [-0.005, +0.164] | +0.150 [+0.094, +0.212] / -0.060 [-0.120, -0.011] |
| row 1 after an absent frame | 1,034 | 163.4 [150.4, 177.2] | 301.4 [258.2, 347.3] | 349.3 [311.3, 390.1] | -0.3428 [-0.5112, -0.2273] | no | +0.274 [+0.182, +0.375] / +0.025 [-0.044, +0.096] | +0.230 [+0.152, +0.321] / -0.034 [-0.092, +0.010] |
| row 2 after an absent frame | 1,045 | 97.8 [86.9, 115.2] | 233.9 [189.1, 283.6] | 258.6 [215.8, 305.6] | -0.2234 [-0.3742, -0.1327] | no | +0.299 [+0.100, +0.479] / +0.144 [-0.103, +0.348] | +0.059 [-0.028, +0.151] / -0.151 [-0.297, -0.047] |
| row 3 after an absent frame | 1,076 | 78.2 [68.0, 96.4] | 202.8 [167.3, 238.6] | 210.2 [175.3, 245.8] | -0.0751 [-0.1600, +0.0026] | no | +0.193 [+0.047, +0.334] / +0.132 [-0.016, +0.277] | +0.058 [-0.066, +0.164] / -0.013 [-0.134, +0.094] |
| all other rows | 188,779 | 46.4 [40.5, 51.1] | 183.3 [173.2, 193.5] | 189.9 [179.8, 200.1] | -0.0733 [-0.0907, -0.0566] | no | +0.082 [+0.057, +0.105] / +0.015 [-0.008, +0.036] | +0.045 [+0.018, +0.069] / -0.025 [-0.049, -0.007] |

Change of the skill over the twin (one row old minus file values), all rows: -0.0747 [-0.0901, -0.0594].

## Expectations of the registration (section 6)

| expectation | found | inside |
|---|---|---|
| RMS change of the estimate, one raw sample earlier: 2 to 8 eV | 12.6 | outside |
| RMS change of the estimate, 0.5 ms earlier: 3 to 10 eV | 46.1 | outside |
| RMS change of the estimate, 2 ms earlier: 4 to 12 eV | 53.1 | outside |
| RMS change of the estimate, 10 ms earlier: 10 to 30 eV | 53.5 | outside |
| RMS error near 200 eV (part R, file values; read as 150 to 250 eV) | 192.5 | inside |
| RMS error near 200 eV (part G, file values; read as 150 to 250 eV) | 184.6 | inside |
| R1 change skill -0.003 to +0.003 | -0.0049 | outside |
| R1 holds | does not hold | outside |
| R2 change skill -0.004 to +0.004 | -0.0234 | outside |
| R2 holds | does not hold | outside |
| R3 change skill -0.006 to +0.004 | -0.0071 | outside |
| R3 holds | does not hold | outside |
| G2 change skill -0.030 to -0.005 | -0.0785 | outside |
| G2 does not hold | does not hold | inside |
| G1 skill over the twin +0.020 to +0.045 | -0.0266 | outside |
| G1 holds | does not hold | outside |
| rows 1 to 3 after an absent frame, fast channels one row old: change skill -0.15 to -0.03 | -0.2474 | outside |

## Choices made where the registration left room

- Treatment of every network: read from `run_b39.pooled_env` (identical for the 40 runs: drop of held values on, T_i spike cut 3000 eV, fast channels standardized with the statistics of the training discharges, no expanding standardizer, no zeroed Mirnov, v2); each metrics.json is checked against it (model, standardization, seed and initialisation seed of the runner).
- Scored rows: `CES_TI_row` indexes the grid rows with at least one finite target (`compact_index` of B.45/B.49); `CES_TI_shot` indexes the discharges with any finite target in the cut population. The grid T_i at the mapped row is compared with the stored `y_true` (largest difference listed under check (a)).
- The sample the file holds: for each row and diagnostic, the raw sample at offset 0, -1 or +1 from the stamp (nearest sample = rint((stamp - t0) fs)) whose values equal the file values of all channels of the diagnostic at the five printed decimals, tried in the order 0, -1, +1; a row without such a sample (Mirnov only) keeps the nearest sample. 'd before' = that index minus round(d fs) samples; one raw sample = 1 sample of each diagnostic (2 us for BES and ECEI, 1 us for the BES of 31747, 0.5 us for Mirnov).
- The two means: the mean of the round(w fs) raw samples ending at the held sample (w = 0.1 ms or 1 ms), i.e. the samples in the w up to and including the stamp.
- Raw window of a discharge: the intersection of the windows of BES, ECEI and Mirnov (BES, ECEI and Mirnov are moved together, so a row is replaced only where all three streams exist). A grid row is replaced in every variant only if all samples of every variant exist (10 ms plus one sample before the stamp), so the replaced rows are the same in every variant; rows outside keep the file values. Every replaced row has its stamp minus 10 ms inside the window; the scored rows of part R are the replaced rows that the pooled run scored.
- Part G: the 15 standardized fast channels of row i of a block are those of row i - 1 of the block; the first row of each block (blocks of `build_blocks`, split at steps of at least 0.5 s) keeps its own. The time step and the CES channels are unchanged.
- Rows 1 to 3 after an absent frame: the first grid row after a step of 20 to 40 ms and the next two rows if they follow in 10 ms steps (the rule of `analyze_file_rules`).
- Quantities: the change skill is 1 - SSE(moved) / SSE(file values) with both estimates computed on the CPU here; the skills over the causal GP and the twin use their stored squared errors of the same rows. Point values are the mean over the eight re-partitions of the value of each re-partition (the point rule of `analyze_b39.hier`); the root-mean-square quantities likewise. Intervals: a re-partition is drawn, then its discharges (part R: the twelve discharges; part G: all scored discharges of the re-partition) are resampled with replacement; the same draws for every quantity and variant of a part (seeds 20261118 for part R, 20261119 for part G; the subsets of part G reuse the generator seed of part G).
- Check (a), as fixed by addendum 1: for every network, |estimate - y_true| computed here lies within 0.5 eV of the square root of the stored squared error on at least 99 % of its rows (the registered rule, largest difference below 0.05 eV, failed before any statement was read; see Deviations).
- Check (b): the rule of `analyze_file_rules.raw_side` run again; it passes when the number of rows and the shares (within one sample, and at offsets -1, 0, +1) equal those stored in `docs/paper/file_rules.json` for every discharge and diagnostic. The share at the nearest sample alone is also given.
- Check (c): on the rows of the file rules whose previous row is 10 ms earlier, the raw samples 10 ms (round(0.010 fs) samples) before the nearest sample, within one raw sample, are compared with the file values of the previous row (all channels, five decimals); it passes when, for every discharge and diagnostic, this share is at least the share recorded in the file rules minus 0.0005. The strict share (held index of the row minus 10 ms, no tolerance of one sample) is reported beside it.
- Check (d): the per-discharge sums of the squared errors computed here (file values) replace the stored ones in `analyze_b39.arm_p.hier`, replayed with its generator (seed 2728, 20,000 draws, P1 then P2); every number (point and both ends of the interval) must be within 0.0005 of `docs/paper/b39_primary.json`.
- Expectation 'RMS error near 200 eV' is read as 150 to 250 eV.

## Deviations

Check (a) as registered (largest difference of the estimate below 0.05 eV) failed (largest difference 2.57 eV, 23.6 % of the rows above 0.05 eV) before any statement was read; addendum 1 of the registration replaced it with the B.49 rule, which is the rule applied above. The implementation choices listed above are recorded in addendum 1 as implemented. No other deviation from the registered definitions, margin, intervals or rules.

## Limits stated in advance (section 7)

The twelve discharges with raw streams were chosen for other purposes and the raw windows cover part of each. The networks were trained with samples at the stamp: a difference shows how these networks respond to another instant, not what a network trained with it would reach. The start of the exposure is inferred from beam blips to within about 2 ms.

Run time (s): raw 4, nets 74, checks 2.

## Part D — by diagnostic (addendum 2; registered after parts R and G were read)

Same 40 networks, rows, quantities and intervals as parts R and G (same draws). The estimate with the file values was recomputed and equals that of parts R and G to 0.0e+00 eV.

### Statements

- **D1** (0.5 ms: RMS change with only Mirnov moved over that with BES and ECEI moved, interval above 1): ratio 1.29 [0.64, 1.98] -> **does not hold**.
- **D2** (BES and ECEI 2 ms earlier, Mirnov at the file values, equivalent): change skill -0.0103 [-0.0472, +0.0161] -> **does not hold**.
- **D3** (only BES one row old, change skill with the interval below zero): -0.0503 [-0.0603, -0.0404] -> **holds**.
- **D4** (only ECEI one row old, equivalent): -0.0198 [-0.0279, -0.0136] -> **does not hold**.
- **D5** (only Mirnov one row old, equivalent): -0.0010 [-0.0037, +0.0007] -> **holds**.
- **D6** (Mirnov at its training mean, skill over the twin with the interval above zero): +0.0529 [+0.0283, +0.0780] -> **holds**.
- **D7** (Mirnov at its training mean, equivalent): +0.0051 [-0.0021, +0.0166] -> **does not hold**.

### Twelve discharges (raw streams)

With the file values: RMS error 192.5 [135.2, 245.1] eV, skill over the causal GP +0.022 [-0.144, +0.157], over the twin +0.025 [-0.064, +0.109]. Rows per re-partition: 4,706, 4,729, 4,616, 4,643, 4,721, 4,623, 4,751, 4,792.

| variant | RMS change of the estimate (eV) | RMS error (eV) | change skill | inside [-0.01, +0.01] | skill over the causal GP | skill over the twin |
|---|---|---|---|---|---|---|
| only Mirnov moved, one raw sample earlier | 7.6 [4.4, 13.0] | 192.8 [135.3, 245.7] | -0.0033 [-0.0199, +0.0056] | no | +0.018 [-0.144, +0.156] | +0.022 [-0.063, +0.102] |
| only Mirnov moved, 0.5 ms earlier | 36.1 [17.3, 56.2] | 194.2 [136.6, 247.5] | -0.0181 [-0.0700, +0.0276] | no | +0.005 [-0.165, +0.140] | +0.008 [-0.090, +0.087] |
| only Mirnov moved, 2 ms earlier | 36.3 [18.8, 55.2] | 192.2 [135.5, 245.7] | +0.0033 [-0.0432, +0.0512] | no | +0.025 [-0.132, +0.167] | +0.029 [-0.059, +0.121] |
| BES and ECEI moved, one raw sample earlier; Mirnov at the file values | 10.0 [7.5, 13.6] | 192.7 [135.0, 245.7] | -0.0019 [-0.0136, +0.0064] | no | +0.020 [-0.147, +0.159] | +0.024 [-0.063, +0.106] |
| BES and ECEI moved, 0.5 ms earlier; Mirnov at the file values | 28.0 [20.6, 35.3] | 193.0 [135.5, 246.2] | -0.0046 [-0.0244, +0.0131] | no | +0.017 [-0.156, +0.149] | +0.021 [-0.075, +0.102] |
| BES and ECEI moved, 2 ms earlier; Mirnov at the file values | 38.6 [28.3, 49.8] | 193.5 [137.7, 246.3] | -0.0103 [-0.0472, +0.0161] | no | +0.011 [-0.160, +0.137] | +0.015 [-0.084, +0.100] |

Per discharge (pooled over its eight networks), RMS change (eV) / change skill:

| discharge | rows | Dmc_1samp | Dmc_0.5ms | Dmc_2ms | Dbe_1samp | Dbe_0.5ms | Dbe_2ms |
|---|---|---|---|---|---|---|---|
| s31097 | 666 | 2.4 / +0.0051 | 4.1 / +0.0130 | 4.3 / +0.0034 | 6.0 / +0.0087 | 12.8 / +0.0297 | 18.6 / -0.1054 |
| s31114 | 435 | 8.6 / +0.0024 | 16.3 / +0.0048 | 15.9 / +0.0039 | 11.4 / -0.0039 | 37.1 / +0.0005 | 65.3 / -0.0501 |
| s31357 | 338 | 1.7 / +0.0023 | 10.5 / -0.0290 | 11.0 / +0.0196 | 4.4 / -0.0095 | 5.6 / +0.0087 | 6.5 / -0.0247 |
| s31359 | 178 | 9.5 / +0.0018 | 23.5 / -0.0349 | 59.6 / +0.0960 | 8.9 / +0.0069 | 14.6 / -0.0189 | 12.7 / -0.0001 |
| s31747 | 405 | 12.4 / -0.0032 | 74.5 / -0.1062 | 76.7 / -0.0541 | 8.8 / +0.0041 | 27.3 / +0.0099 | 32.7 / -0.0123 |
| s31873 | 177 | 2.1 / +0.0015 | 3.3 / -0.0092 | 2.5 / +0.0080 | 6.7 / +0.0003 | 11.2 / -0.0369 | 18.5 / -0.1408 |
| s31902 | 480 | 2.8 / -0.0002 | 4.8 / +0.0004 | 7.5 / -0.0000 | 12.5 / +0.0138 | 41.1 / +0.0210 | 54.5 / -0.0242 |
| s31921 | 381 | 12.1 / -0.0171 | 37.9 / -0.0119 | 30.7 / -0.0089 | 16.4 / -0.0202 | 38.2 / -0.0174 | 40.4 / +0.0184 |
| s31923 | 330 | 13.7 / -0.0039 | 49.3 / -0.0013 | 32.0 / +0.0105 | 14.2 / +0.0022 | 23.3 / -0.0157 | 28.7 / -0.0206 |
| s32027 | 344 | 4.8 / +0.0020 | 10.1 / -0.0031 | 9.7 / +0.0041 | 8.7 / +0.0023 | 28.1 / -0.0703 | 40.8 / -0.1377 |
| s32092 | 422 | 6.6 / -0.0036 | 36.9 / -0.0019 | 37.6 / -0.0142 | 8.0 / +0.0067 | 30.3 / +0.0015 | 40.4 / -0.0888 |
| s32097 | 540 | 5.7 / +0.0015 | 57.5 / -0.0162 | 56.6 / -0.0172 | 8.7 / -0.0028 | 27.5 / +0.0178 | 38.3 / +0.0200 |

### All discharges

| variant | rows | RMS change (eV) | RMS error, file values (eV) | RMS error, variant (eV) | change skill | inside [-0.01, +0.01] | skill over causal GP: file / variant | skill over the twin: file / variant |
|---|---|---|---|---|---|---|---|---|
| only BES one row old | all scored rows | 37.6 [34.2, 40.6] | 184.6 [174.5, 194.7] | 189.2 [179.1, 199.3] | -0.0503 [-0.0603, -0.0404] | no | +0.089 [+0.063, +0.112] / +0.043 [+0.020, +0.064] | +0.048 [+0.022, +0.073] / +0.000 [-0.024, +0.022] |
| only ECEI one row old | all scored rows | 22.4 [17.4, 27.1] | 184.6 [174.5, 194.7] | 186.4 [176.3, 196.5] | -0.0198 [-0.0279, -0.0136] | no | +0.089 [+0.063, +0.112] / +0.071 [+0.048, +0.092] | +0.048 [+0.022, +0.073] / +0.029 [+0.006, +0.050] |
| only Mirnov one row old | all scored rows | 12.7 [10.6, 14.9] | 184.6 [174.5, 194.7] | 184.7 [174.6, 194.8] | -0.0010 [-0.0037, +0.0007] | yes | +0.089 [+0.063, +0.112] / +0.088 [+0.063, +0.111] | +0.048 [+0.022, +0.073] / +0.047 [+0.021, +0.072] |
| Mirnov at its training mean (0 after standardization) at every row | all scored rows | 18.5 [14.2, 27.0] | 184.6 [174.5, 194.7] | 184.1 [174.1, 194.2] | +0.0051 [-0.0021, +0.0166] | no | +0.089 [+0.063, +0.112] / +0.094 [+0.069, +0.116] | +0.048 [+0.022, +0.073] / +0.053 [+0.028, +0.078] |
| only BES one row old | rows 1 to 3 after an absent frame | 108.3 [100.1, 118.2] | 249.1 [221.3, 277.3] | 273.5 [247.3, 300.4] | -0.2051 [-0.2749, -0.1496] | no | +0.265 [+0.191, +0.335] / +0.114 [+0.030, +0.193] | +0.150 [+0.094, +0.212] / -0.024 [-0.078, +0.026] |
| only ECEI one row old | rows 1 to 3 after an absent frame | 29.5 [20.3, 41.1] | 249.1 [221.3, 277.3] | 251.9 [224.6, 279.7] | -0.0225 [-0.0462, -0.0087] | no | +0.265 [+0.191, +0.335] / +0.249 [+0.176, +0.316] | +0.150 [+0.094, +0.212] / +0.131 [+0.075, +0.190] |
| only Mirnov one row old | rows 1 to 3 after an absent frame | 17.4 [11.4, 24.6] | 249.1 [221.3, 277.3] | 249.2 [221.3, 277.3] | -0.0004 [-0.0149, +0.0145] | no | +0.265 [+0.191, +0.335] / +0.265 [+0.191, +0.336] | +0.150 [+0.094, +0.212] / +0.150 [+0.092, +0.213] |
| Mirnov at its training mean (0 after standardization) at every row | rows 1 to 3 after an absent frame | 22.9 [13.7, 45.7] | 249.1 [221.3, 277.3] | 249.1 [221.2, 277.2] | +0.0004 [-0.0206, +0.0294] | no | +0.265 [+0.191, +0.335] / +0.265 [+0.194, +0.334] | +0.150 [+0.094, +0.212] / +0.150 [+0.094, +0.212] |

### Expectations of addendum 2

| expectation | found | inside |
|---|---|---|
| D1 ratio 2 to 6 | 1.29 | outside |
| D1 holds | does not hold | outside |
| D2 change skill -0.010 to +0.005 | -0.0103 | outside |
| D2 interval about 0.02 wide on each side (read as 0.01 to 0.03 below and above the point) | 0.0369 below, 0.0264 above | outside |
| D2 does not hold | does not hold | inside |
| D3 change skill -0.09 to -0.04 | -0.0503 | inside |
| D3 holds | holds | inside |
| D4 change skill -0.010 to +0.002 | -0.0198 | outside |
| D4 holds | does not hold | outside |
| D5 change skill -0.015 to +0.005 | -0.0010 | inside |
| D5 does not hold | holds | outside |
| D6 skill over the twin +0.030 to +0.055 | +0.0529 | inside |
| D6 holds | holds | inside |
| D7 change skill -0.020 to +0.010 | +0.0051 | inside |
| D7 does not hold | does not hold | inside |

### Choices made for part D

- Same rows, generators and draws as parts R and G: the twelve-discharge variants use the draws of part R (seed 20261118, 2,000), the all-discharge variants those of part G (seed 20261119, 20,000); the subset of rows 1 to 3 after an absent frame reuses the seed of part G. The file-value estimate is recomputed and must equal the one of parts R and G (largest difference given above).
- Moved samples on the twelve discharges: the raw values of the variant of part R for the moved channels, the file values for the others, on the same replaced rows.
- One row old for one diagnostic: as in part G, only on the standardized channels of that diagnostic.
- Mirnov at its training mean: the two standardized Mirnov channels set to 0 at every row of every block.
- D1: ratio of the root-mean-square change with only Mirnov moved to that with BES and ECEI moved, both at 0.5 ms; point = mean over the re-partitions of the ratio of each, interval from the draws of part R; holds if the lower end is above 1. D3 holds if the upper end of the change skill is below zero.
- Expectation 'interval about 0.02 wide on each side' of D2 is read as each side between 0.01 and 0.03.

Limit added by addendum 2: a network that was trained with the Mirnov channels and is given their mean is not a network trained without them.

Run time of part D (s): netsD 166.
