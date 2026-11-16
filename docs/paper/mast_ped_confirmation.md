# MAST pedestal top, confirmation half (registration PREREGISTRATION_MAST_PED.md)

Models trained once on the discovery half (odd-numbered shots) and applied to the even-numbered shots. Skill a vs b = 1 - SSE_a / SSE_b, 95 % interval from 2,000 shot-level bootstrap resamples. hold = last visible frame; interp = mean of the two neighbouring visible frames (not causal); gb_ces = trees on the own charge-exchange past; gb_ts = plus Thomson T_e / n_e at the chord radius; gb_da = plus D-alpha; gb_all = plus Thomson, D-alpha, line density, saddle-coil signal and H-mode rating.

## Registered statements (T_i)

- **P1**: holds -- +0.148 [+0.122, +0.177]
- **P2**: holds -- +0.047 [+0.033, +0.068]
- **P3**: holds -- +0.023 [+0.012, +0.038]
- **P4**: holds -- +0.098 [+0.071, +0.122]
- **P5**: holds -- 0.797 [0.782, 0.809]
- **P6**: holds -- -0.161 [-0.299, -0.021]

## T_i: 2326823 rows, 1651 confirmation shots

| cell | rows | shots | hold RMSE | gb_all vs hold | gb_ces vs hold | gb_ts vs gb_ces | gb_da vs gb_ces | gb_all vs gb_all_no_ts | gb_all vs gb_ces | gb_all vs interp | interp vs hold |
|---|---|---|---|---|---|---|---|---|---|---|---|
| pedestal top | 74060 | 1434 | 107.2 | +0.148 [+0.122, +0.177] | +0.101 [+0.080, +0.122] | +0.047 [+0.033, +0.068] | +0.023 [+0.012, +0.038] | +0.020 [+0.012, +0.029] | +0.052 [+0.034, +0.076] | -0.161 [-0.299, -0.021] | +0.266 [+0.160, +0.352] |
| core (> 20 cm inside) | 750310 | 1489 | 114.9 | +0.286 [+0.266, +0.305] | +0.141 [+0.131, +0.151] | +0.145 [+0.125, +0.166] | +0.083 [+0.066, +0.099] | +0.066 [+0.055, +0.077] | +0.169 [+0.150, +0.188] | -0.102 [-0.170, -0.043] | +0.352 [+0.314, +0.390] |
| pedestal top, H-mode rating >= 0.5 | 22063 | 738 | 126.1 | +0.133 [+0.093, +0.176] | +0.104 [+0.068, +0.134] | +0.030 [+0.015, +0.057] | +0.009 [-0.003, +0.022] | +0.011 [+0.000, +0.025] | +0.033 [+0.015, +0.058] | -0.299 [-0.504, -0.120] | +0.333 [+0.208, +0.439] |
| all rows | 2326823 | 1651 | 124.0 | +0.209 [+0.197, +0.220] | +0.126 [+0.120, +0.132] | +0.052 [+0.044, +0.060] | +0.047 [+0.039, +0.055] | +0.021 [+0.017, +0.025] | +0.095 [+0.086, +0.105] | -0.202 [-0.235, -0.168] | +0.342 [+0.323, +0.360] |
| Thomson available | 1258542 | 1495 | 109.9 | +0.259 [+0.243, +0.276] | +0.136 [+0.128, +0.145] | +0.122 [+0.106, +0.138] | +0.069 [+0.056, +0.082] | +0.052 [+0.045, +0.060] | +0.142 [+0.127, +0.158] | -0.119 [-0.170, -0.073] | +0.338 [+0.308, +0.368] |

Pedestal top: median absolute error of hold 14.25; ratio of the all-input model to it 0.797 [0.782, 0.809]. Thomson increment, core minus pedestal top: +0.098 [+0.071, +0.122].

## V_i: 2326823 rows, 1651 confirmation shots

| cell | rows | shots | hold RMSE | gb_all vs hold | gb_ces vs hold | gb_ts vs gb_ces | gb_da vs gb_ces | gb_all vs gb_all_no_ts | gb_all vs gb_ces | gb_all vs interp | interp vs hold |
|---|---|---|---|---|---|---|---|---|---|---|---|
| pedestal top | 74060 | 1434 | 5.4 | +0.206 [+0.170, +0.246] | +0.100 [+0.084, +0.118] | +0.107 [+0.080, +0.138] | +0.065 [+0.048, +0.084] | +0.042 [+0.032, +0.054] | +0.118 [+0.087, +0.153] | +0.100 [-0.290, +0.367] | +0.118 [-0.240, +0.378] |
| core (> 20 cm inside) | 750310 | 1489 | 8.8 | +0.376 [+0.352, +0.398] | +0.232 [+0.214, +0.250] | +0.148 [+0.133, +0.163] | +0.085 [+0.071, +0.100] | +0.071 [+0.062, +0.080] | +0.187 [+0.169, +0.204] | -0.389 [-0.503, -0.284] | +0.550 [+0.507, +0.587] |
| pedestal top, H-mode rating >= 0.5 | 22063 | 738 | 4.6 | +0.270 [+0.226, +0.316] | +0.099 [+0.068, +0.134] | +0.150 [+0.108, +0.194] | +0.124 [+0.093, +0.155] | +0.052 [+0.035, +0.071] | +0.190 [+0.136, +0.236] | -0.213 [-0.409, -0.032] | +0.398 [+0.294, +0.482] |
| all rows | 2326823 | 1651 | 10.3 | +0.246 [+0.230, +0.263] | +0.164 [+0.154, +0.175] | +0.043 [+0.037, +0.049] | +0.047 [+0.040, +0.053] | +0.017 [+0.014, +0.020] | +0.098 [+0.088, +0.109] | -0.346 [-0.415, -0.278] | +0.440 [+0.409, +0.470] |
| Thomson available | 1258542 | 1495 | 8.4 | +0.326 [+0.303, +0.349] | +0.197 [+0.182, +0.212] | +0.129 [+0.115, +0.144] | +0.075 [+0.063, +0.088] | +0.058 [+0.051, +0.066] | +0.161 [+0.144, +0.179] | -0.335 [-0.441, -0.229] | +0.495 [+0.448, +0.536] |

Pedestal top: median absolute error of hold 1.70; ratio of the all-input model to it 0.852 [0.843, 0.863]. Thomson increment, core minus pedestal top: +0.041 [+0.007, +0.071].
