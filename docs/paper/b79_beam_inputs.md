# B.79 — the beam waveforms as inputs: what the fast diagnostics add beyond what the actuators already say

Registered: `ces_prediction/experiments/PREREGISTRATION_B79.md`; implementation choices in `ces_prediction/experiments/PREREGISTRATION_B79_ADDENDUM1.md`, written before any fit and any readout. CPU only; no network trained or run. T_i, cut population, the D1 rows of B.70 as cached by B.73 (222,672 rows, 635 discharges). Exploratory on used data (registration section 1): rows, targets, designs, learners, rules and fast inputs are B.75's; the nineteen beam scalars are new.

Learners (all fitted to target - G under the rule): T0 the ten history scalars; TN T0 and the nineteen beam scalars; TX T0 and the fourteen fast scalars of B.75; TXN TX and the nineteen beam scalars; TEN, TBN TN and the ECEI or BES half of the fast scalars; TNp TN with the beam scalars permuted over rows; TXNp TXN with the fast scalars permuted. G causal GP of the visible history; O the frame's own reading (against L5 only).

Conventions. Skill of A over B = 1 - SSE(A) / SSE(B) on identical rows. Each comparison: point, then 95 % intervals with discharges resampled (d) and with shot clusters resampled (c), 20,000 draws (V1: a re-partition drawn first; V2: an assignment drawn first; V3, V4: the scored units). Verdicts on the cluster form. Points under V1 and V2 = mean over the three re-partitions or assignments. D10 and D650 are fitted and scored on the common set, S on the S set; the sets equal B.75's. RMSE in eV. Checks: `docs/paper/b79_checks.md`.

The beam columns are used as delivered. In thirteen early files (shots 30801-30909) the NBI-2 power columns are on another scale (up to 144 MW; Addendum 1 section 7); a scoring-only sensitivity without those files is at the end.

Order of work. Before this readout, these results existed: the checks of registration section 9, the comparison of this batch's T0 and TX files with B.75's, and the counts of the row sets. No MSE, skill or statement with a beam input had been computed.

## Statements (target L5, cluster interval)

| Id | statement | value: point; d interval; c interval | verdict | registered expectation | expectation met | note |
|---|---|---|---|---|---|---|
| N1 | V1, D650: TN over T0 lies above zero (the beam waveforms add to the history) | +0.148; d [+0.094, +0.193]; c [+0.086, +0.199] | **holds** | holds (+0.03 to +0.15) | yes; point in the stated range: yes |  |
| N2 | V1, D650: TXN over TN lies above zero (the fast diagnostics add beyond the beam waveforms) | +0.258; d [+0.222, +0.293]; c [+0.205, +0.304] | **holds** | holds (+0.20 to +0.35) | yes; point in the stated range: yes | TX over T0: +0.336; c [+0.285, +0.381] |
| N3 | V2, D650: TXN over TN lies above zero | +0.298; d [+0.238, +0.361]; c [+0.206, +0.378] | **holds** | holds | yes |  |
| N4 | V3, D650: TXN over TN lies above zero | +0.263; d [+0.198, +0.324]; c [+0.162, +0.363] | **holds** | holds | yes |  |
| N5 | V4 (fit to 31991), D650: TXN over TN lies above zero | +0.245; d [+0.146, +0.341]; c [+0.061, +0.357] | **holds** | uncertain, leaning to holds | n/a (uncertain) | V4b (fit to 32310): +0.275; c [+0.151, +0.407] |
| N6 | V1, S: TXN over TN lies above zero | +0.253; d [+0.202, +0.305]; c [+0.184, +0.319] | **holds** | holds | yes |  |
| N7 | V4 (fit to 31991), S: TXN over TN lies above zero | +0.102; d [-0.059, +0.230]; c [-0.156, +0.242] | **does not hold** | uncertain | n/a (uncertain) | V4b: -0.062; c [-0.488, +0.150] |
| N8 | V4 (fit to 31991), S: TN over T0 lies above zero | -0.100; d [-0.274, +0.042]; c [-0.282, +0.119] | **does not hold** | uncertain | n/a (uncertain) | V4b: -0.051; c [-0.193, +0.130] |
| N9 | V1, D650: TXN over TX lies above zero (the beam waveforms add on top of the fast diagnostics) | +0.047; d [+0.023, +0.069]; c [+0.023, +0.070] | **holds** | uncertain (0.00 to +0.05) | n/a (uncertain); point in the stated range: yes |  |
| N10 | V1, D10: TXN over TN lies above zero | +0.097; d [+0.074, +0.128]; c [+0.063, +0.144] | **holds** | holds (+0.05 to +0.10) | yes; point in the stated range: yes |  |
| N11 | V1, D650, beam scalars permuted: the interval of TN over T0 contains zero or lies below it | -0.003; d [-0.007, +0.001]; c [-0.008, +0.001] | **holds** | holds | yes | fast scalars permuted, TXNp over TN: +0.001; c [-0.006, +0.009] |

'V4' without a qualifier is V4a (fit to 31991); V4b is in the note.

## RMSE by design and rule, target L5

Points; under V1 and V2 the mean over the three splits. O = the frame's own reading.

| design | rule | rows / discharges / clusters | G | T0 | TN | TX | TXN | TEN | TBN | O |
|---|---|---|---|---|---|---|---|---|---|---|
| V1 random (r2-r4) | D10 | 160,343 / 562 / 79 | 144.1 | 131.5 | 130.6 | 124.5 | 124.1 | - | - | 183.3 |
| V1 random (r2-r4) | D650 | 160,343 / 562 / 79 | 277.6 | 226.2 | 208.8 | 184.3 | 179.9 | 190.1 | 188.9 | 183.3 |
| V1 random (r2-r4) | S (schedule) | 147,837 / 503 / 71 | 226.3 | 217.6 | 202.6 | 178.5 | 175.1 | 183.5 | 187.3 | 178.2 |
| V2 cluster-blocked (3 assignments) | D10 | 160,343 / 562 / 79 | 144.1 | 133.3 | 135.0 | 129.4 | 130.0 | - | - | 183.3 |
| V2 cluster-blocked (3 assignments) | D650 | 160,343 / 562 / 79 | 277.6 | 236.7 | 241.8 | 200.6 | 202.6 | - | - | 183.3 |
| V2 cluster-blocked (3 assignments) | S (schedule) | 147,837 / 503 / 71 | 226.3 | 223.2 | 215.1 | 187.7 | 187.7 | - | - | 178.2 |
| V3 rolling origin (fifths 2-5 pooled) | D10 | 112,199 / 441 / 60 | 160.8 | 150.6 | 153.4 | 148.5 | 149.4 | - | - | 205.7 |
| V3 rolling origin (fifths 2-5 pooled) | D650 | 112,199 / 441 / 60 | 307.5 | 268.5 | 284.9 | 241.7 | 244.6 | - | - | 205.7 |
| V3 rolling origin (fifths 2-5 pooled) | S (schedule) | 102,684 / 395 / 56 | 249.7 | 249.1 | 237.2 | 218.5 | 215.5 | - | - | 200.8 |
| V4 chronological: fit to 31991, score from 32312 | D10 | 10,146 / 75 / 18 | 170.8 | 145.0 | 142.3 | 138.4 | 139.9 | - | - | 245.3 |
| V4 chronological: fit to 31991, score from 32312 | D650 | 10,146 / 75 / 18 | 203.6 | 202.3 | 208.9 | 176.7 | 181.4 | 195.7 | 194.7 | 245.3 |
| V4 chronological: fit to 31991, score from 32312 | S (schedule) | 7,180 / 50 / 14 | 187.4 | 184.8 | 193.9 | 182.3 | 183.7 | 181.4 | 187.8 | 247.4 |
| V4 chronological: fit to 32310, score from 32312 | D10 | 10,146 / 75 / 18 | 170.8 | 138.1 | 134.7 | 132.5 | 131.8 | - | - | 245.3 |
| V4 chronological: fit to 32310, score from 32312 | D650 | 10,146 / 75 / 18 | 203.6 | 189.8 | 198.1 | 168.1 | 168.7 | 184.9 | 179.1 | 245.3 |
| V4 chronological: fit to 32310, score from 32312 | S (schedule) | 7,180 / 50 / 14 | 187.4 | 170.3 | 174.7 | 190.8 | 180.0 | 191.9 | 170.2 | 247.4 |

## RMSE by design and rule, target F

Points; under V1 and V2 the mean over the three splits.

| design | rule | rows / discharges / clusters | G | T0 | TN | TX | TXN | TEN | TBN |
|---|---|---|---|---|---|---|---|---|---|
| V1 random (r2-r4) | D10 | 165,418 / 564 / 79 | 189.8 | 179.7 | 179.6 | 170.3 | 170.5 | - | - |
| V1 random (r2-r4) | D650 | 165,418 / 564 / 79 | 317.0 | 275.4 | 260.5 | 232.6 | 229.2 | 240.2 | 239.7 |
| V1 random (r2-r4) | S (schedule) | 152,203 / 506 / 71 | 272.9 | 265.4 | 252.4 | 224.7 | 222.0 | 232.8 | 233.0 |
| V2 cluster-blocked (3 assignments) | D10 | 165,418 / 564 / 79 | 189.8 | 180.3 | 180.7 | 173.0 | 173.0 | - | - |
| V2 cluster-blocked (3 assignments) | D650 | 165,418 / 564 / 79 | 317.0 | 283.3 | 286.5 | 247.4 | 247.3 | - | - |
| V2 cluster-blocked (3 assignments) | S (schedule) | 152,203 / 506 / 71 | 272.9 | 271.3 | 265.6 | 233.4 | 234.8 | - | - |
| V3 rolling origin (fifths 2-5 pooled) | D10 | 116,471 / 442 / 60 | 213.9 | 205.5 | 206.6 | 199.3 | 199.3 | - | - |
| V3 rolling origin (fifths 2-5 pooled) | D650 | 116,471 / 442 / 60 | 352.4 | 320.3 | 332.0 | 294.2 | 293.4 | - | - |
| V3 rolling origin (fifths 2-5 pooled) | S (schedule) | 106,417 / 398 / 56 | 303.7 | 305.4 | 296.1 | 274.4 | 271.5 | - | - |
| V4 chronological: fit to 31991, score from 32312 | D10 | 10,662 / 75 / 18 | 259.3 | 248.1 | 249.5 | 244.9 | 244.7 | - | - |
| V4 chronological: fit to 31991, score from 32312 | D650 | 10,662 / 75 / 18 | 301.5 | 297.5 | 299.8 | 284.2 | 284.2 | 296.8 | 296.7 |
| V4 chronological: fit to 31991, score from 32312 | S (schedule) | 7,551 / 50 / 14 | 299.8 | 298.2 | 296.1 | 293.6 | 296.6 | 292.2 | 298.1 |
| V4 chronological: fit to 32310, score from 32312 | D10 | 10,662 / 75 / 18 | 259.3 | 244.6 | 244.2 | 238.0 | 239.5 | - | - |
| V4 chronological: fit to 32310, score from 32312 | D650 | 10,662 / 75 / 18 | 301.5 | 292.3 | 299.4 | 277.3 | 279.2 | 291.9 | 284.6 |
| V4 chronological: fit to 32310, score from 32312 | S (schedule) | 7,551 / 50 / 14 | 299.8 | 292.3 | 295.6 | 286.3 | 298.1 | 304.3 | 291.9 |

## Increments by design and rule, target L5

Cells: skill, d interval, c interval.

| design | rule | TN over T0 (beam over history) | TXN over TN (fast over history and beam) | TX over T0 (fast over history, B.75) | TXN over TX (beam over history and fast) | TXN over T0 (both) |
|---|---|---|---|---|---|---|
| V1 random (r2-r4) | D10 | +0.014<br>d [-0.012, +0.038]<br>c [-0.012, +0.040] | +0.097<br>d [+0.074, +0.128]<br>c [+0.063, +0.144] | +0.103<br>d [+0.079, +0.127]<br>c [+0.069, +0.140] | +0.007<br>d [-0.003, +0.018]<br>c [-0.004, +0.020] | +0.110<br>d [+0.084, +0.136]<br>c [+0.074, +0.152] |
| V1 random (r2-r4) | D650 | +0.148<br>d [+0.094, +0.193]<br>c [+0.086, +0.199] | +0.258<br>d [+0.222, +0.293]<br>c [+0.205, +0.304] | +0.336<br>d [+0.299, +0.372]<br>c [+0.285, +0.381] | +0.047<br>d [+0.023, +0.069]<br>c [+0.023, +0.070] | +0.368<br>d [+0.331, +0.402]<br>c [+0.315, +0.412] |
| V1 random (r2-r4) | S (schedule) | +0.133<br>d [+0.085, +0.179]<br>c [+0.078, +0.187] | +0.253<br>d [+0.202, +0.305]<br>c [+0.184, +0.319] | +0.327<br>d [+0.277, +0.375]<br>c [+0.263, +0.388] | +0.038<br>d [+0.005, +0.069]<br>c [+0.003, +0.069] | +0.352<br>d [+0.302, +0.401]<br>c [+0.288, +0.416] |
| V2 cluster-blocked (3 assignments) | D10 | -0.026<br>d [-0.070, -0.000]<br>c [-0.088, +0.009] | +0.074<br>d [+0.047, +0.102]<br>c [+0.039, +0.109] | +0.058<br>d [+0.033, +0.079]<br>c [+0.020, +0.088] | -0.009<br>d [-0.023, +0.010]<br>c [-0.028, +0.017] | +0.049<br>d [+0.027, +0.069]<br>c [+0.016, +0.077] |
| V2 cluster-blocked (3 assignments) | D650 | -0.044<br>d [-0.138, +0.049]<br>c [-0.207, +0.095] | +0.298<br>d [+0.238, +0.361]<br>c [+0.206, +0.378] | +0.282<br>d [+0.244, +0.318]<br>c [+0.215, +0.343] | -0.020<br>d [-0.074, +0.019]<br>c [-0.080, +0.021] | +0.268<br>d [+0.211, +0.312]<br>c [+0.184, +0.343] |
| V2 cluster-blocked (3 assignments) | S (schedule) | +0.072<br>d [+0.012, +0.133]<br>c [-0.019, +0.157] | +0.238<br>d [+0.172, +0.301]<br>c [+0.157, +0.318] | +0.293<br>d [+0.243, +0.341]<br>c [+0.215, +0.365] | +0.000<br>d [-0.045, +0.036]<br>c [-0.051, +0.038] | +0.293<br>d [+0.238, +0.344]<br>c [+0.217, +0.367] |
| V3 rolling origin (fifths 2-5 pooled) | D10 | -0.037<br>d [-0.060, -0.017]<br>c [-0.069, -0.009] | +0.050<br>d [+0.029, +0.071]<br>c [+0.014, +0.082] | +0.028<br>d [+0.004, +0.049]<br>c [-0.012, +0.067] | -0.013<br>d [-0.026, -0.001]<br>c [-0.033, +0.003] | +0.015<br>d [-0.008, +0.036]<br>c [-0.025, +0.051] |
| V3 rolling origin (fifths 2-5 pooled) | D650 | -0.126<br>d [-0.244, -0.026]<br>c [-0.370, +0.047] | +0.263<br>d [+0.198, +0.324]<br>c [+0.162, +0.363] | +0.190<br>d [+0.133, +0.242]<br>c [+0.032, +0.290] | -0.025<br>d [-0.062, +0.010]<br>c [-0.098, +0.030] | +0.170<br>d [+0.115, +0.220]<br>c [+0.013, +0.275] |
| V3 rolling origin (fifths 2-5 pooled) | S (schedule) | +0.093<br>d [+0.032, +0.150]<br>c [+0.021, +0.166] | +0.175<br>d [+0.121, +0.225]<br>c [+0.100, +0.252] | +0.231<br>d [+0.167, +0.291]<br>c [+0.144, +0.325] | +0.027<br>d [-0.023, +0.074]<br>c [-0.023, +0.069] | +0.252<br>d [+0.191, +0.306]<br>c [+0.182, +0.328] |
| V4 chronological: fit to 31991, score from 32312 | D10 | +0.037<br>d [-0.012, +0.082]<br>c [-0.016, +0.082] | +0.034<br>d [-0.066, +0.100]<br>c [-0.035, +0.096] | +0.089<br>d [+0.015, +0.139]<br>c [+0.031, +0.135] | -0.022<br>d [-0.077, +0.025]<br>c [-0.079, +0.029] | +0.069<br>d [-0.055, +0.149]<br>c [-0.040, +0.148] |
| V4 chronological: fit to 31991, score from 32312 | D650 | -0.066<br>d [-0.383, +0.145]<br>c [-0.497, +0.326] | +0.245<br>d [+0.146, +0.341]<br>c [+0.061, +0.357] | +0.237<br>d [+0.071, +0.358]<br>c [+0.040, +0.415] | -0.054<br>d [-0.148, +0.024]<br>c [-0.132, +0.034] | +0.196<br>d [+0.007, +0.327]<br>c [-0.035, +0.395] |
| V4 chronological: fit to 31991, score from 32312 | S (schedule) | -0.100<br>d [-0.274, +0.042]<br>c [-0.282, +0.119] | +0.102<br>d [-0.059, +0.230]<br>c [-0.156, +0.242] | +0.028<br>d [-0.134, +0.158]<br>c [-0.167, +0.170] | -0.016<br>d [-0.121, +0.071]<br>c [-0.154, +0.106] | +0.012<br>d [-0.218, +0.172]<br>c [-0.272, +0.184] |
| V4 chronological: fit to 32310, score from 32312 | D10 | +0.048<br>d [+0.017, +0.076]<br>c [+0.001, +0.094] | +0.043<br>d [-0.010, +0.090]<br>c [+0.002, +0.095] | +0.079<br>d [+0.036, +0.113]<br>c [+0.058, +0.112] | +0.011<br>d [-0.013, +0.031]<br>c [-0.004, +0.033] | +0.088<br>d [+0.037, +0.128]<br>c [+0.060, +0.136] |
| V4 chronological: fit to 32310, score from 32312 | D650 | -0.089<br>d [-0.349, +0.089]<br>c [-0.454, +0.220] | +0.275<br>d [+0.177, +0.384]<br>c [+0.151, +0.407] | +0.216<br>d [+0.102, +0.307]<br>c [+0.077, +0.359] | -0.008<br>d [-0.087, +0.058]<br>c [-0.106, +0.063] | +0.210<br>d [+0.097, +0.308]<br>c [+0.058, +0.374] |
| V4 chronological: fit to 32310, score from 32312 | S (schedule) | -0.051<br>d [-0.155, +0.048]<br>c [-0.193, +0.130] | -0.062<br>d [-0.258, +0.079]<br>c [-0.488, +0.150] | -0.254<br>d [-0.519, -0.054]<br>c [-0.537, +0.043] | +0.109<br>d [+0.031, +0.187]<br>c [-0.019, +0.192] | -0.117<br>d [-0.275, +0.008]<br>c [-0.331, +0.038] |

## Increments by design and rule, target F

Cells: skill, d interval, c interval.

| design | rule | TN over T0 (beam over history) | TXN over TN (fast over history and beam) | TX over T0 (fast over history, B.75) | TXN over TX (beam over history and fast) | TXN over T0 (both) |
|---|---|---|---|---|---|---|
| V1 random (r2-r4) | D10 | +0.001<br>d [-0.004, +0.006]<br>c [-0.004, +0.006] | +0.099<br>d [+0.082, +0.117]<br>c [+0.067, +0.139] | +0.102<br>d [+0.085, +0.120]<br>c [+0.068, +0.145] | -0.003<br>d [-0.008, +0.003]<br>c [-0.008, +0.003] | +0.100<br>d [+0.082, +0.118]<br>c [+0.066, +0.141] |
| V1 random (r2-r4) | D650 | +0.105<br>d [+0.071, +0.135]<br>c [+0.066, +0.141] | +0.226<br>d [+0.196, +0.257]<br>c [+0.180, +0.268] | +0.287<br>d [+0.257, +0.315]<br>c [+0.240, +0.328] | +0.029<br>d [+0.010, +0.049]<br>c [+0.009, +0.050] | +0.307<br>d [+0.275, +0.337]<br>c [+0.259, +0.352] |
| V1 random (r2-r4) | S (schedule) | +0.095<br>d [+0.064, +0.126]<br>c [+0.064, +0.128] | +0.226<br>d [+0.192, +0.260]<br>c [+0.174, +0.277] | +0.283<br>d [+0.243, +0.321]<br>c [+0.227, +0.338] | +0.024<br>d [+0.006, +0.042]<br>c [+0.009, +0.043] | +0.300<br>d [+0.262, +0.336]<br>c [+0.246, +0.354] |
| V2 cluster-blocked (3 assignments) | D10 | -0.004<br>d [-0.009, +0.001]<br>c [-0.011, +0.003] | +0.083<br>d [+0.069, +0.098]<br>c [+0.058, +0.114] | +0.080<br>d [+0.064, +0.095]<br>c [+0.052, +0.113] | -0.000<br>d [-0.007, +0.006]<br>c [-0.007, +0.007] | +0.079<br>d [+0.065, +0.095]<br>c [+0.052, +0.113] |
| V2 cluster-blocked (3 assignments) | D650 | -0.023<br>d [-0.086, +0.041]<br>c [-0.138, +0.070] | +0.255<br>d [+0.211, +0.298]<br>c [+0.190, +0.316] | +0.237<br>d [+0.201, +0.271]<br>c [+0.176, +0.286] | +0.001<br>d [-0.019, +0.020]<br>c [-0.024, +0.025] | +0.238<br>d [+0.199, +0.272]<br>c [+0.174, +0.291] |
| V2 cluster-blocked (3 assignments) | S (schedule) | +0.042<br>d [-0.005, +0.084]<br>c [-0.022, +0.099] | +0.218<br>d [+0.173, +0.268]<br>c [+0.154, +0.287] | +0.260<br>d [+0.217, +0.301]<br>c [+0.196, +0.320] | -0.012<br>d [-0.040, +0.015]<br>c [-0.041, +0.019] | +0.251<br>d [+0.206, +0.296]<br>c [+0.183, +0.315] |
| V3 rolling origin (fifths 2-5 pooled) | D10 | -0.010<br>d [-0.020, -0.002]<br>c [-0.031, +0.004] | +0.069<br>d [+0.053, +0.085]<br>c [+0.041, +0.103] | +0.059<br>d [+0.045, +0.075]<br>c [+0.036, +0.091] | +0.000<br>d [-0.006, +0.006]<br>c [-0.009, +0.009] | +0.060<br>d [+0.044, +0.075]<br>c [+0.035, +0.094] |
| V3 rolling origin (fifths 2-5 pooled) | D650 | -0.074<br>d [-0.142, -0.015]<br>c [-0.205, +0.031] | +0.219<br>d [+0.169, +0.266]<br>c [+0.154, +0.277] | +0.156<br>d [+0.111, +0.198]<br>c [+0.030, +0.237] | +0.005<br>d [-0.019, +0.029]<br>c [-0.037, +0.035] | +0.161<br>d [+0.120, +0.199]<br>c [+0.045, +0.239] |
| V3 rolling origin (fifths 2-5 pooled) | S (schedule) | +0.060<br>d [+0.023, +0.097]<br>c [+0.017, +0.108] | +0.159<br>d [+0.118, +0.201]<br>c [+0.103, +0.213] | +0.193<br>d [+0.149, +0.235]<br>c [+0.137, +0.251] | +0.021<br>d [-0.006, +0.048]<br>c [-0.011, +0.055] | +0.210<br>d [+0.166, +0.252]<br>c [+0.156, +0.265] |
| V4 chronological: fit to 31991, score from 32312 | D10 | -0.011<br>d [-0.023, +0.000]<br>c [-0.020, +0.004] | +0.038<br>d [+0.018, +0.063]<br>c [+0.014, +0.060] | +0.026<br>d [+0.001, +0.056]<br>c [+0.004, +0.055] | +0.001<br>d [-0.010, +0.013]<br>c [-0.008, +0.010] | +0.027<br>d [+0.004, +0.056]<br>c [+0.006, +0.055] |
| V4 chronological: fit to 31991, score from 32312 | D650 | -0.015<br>d [-0.103, +0.051]<br>c [-0.151, +0.114] | +0.101<br>d [+0.032, +0.165]<br>c [+0.038, +0.180] | +0.088<br>d [+0.018, +0.143]<br>c [+0.006, +0.165] | +0.000<br>d [-0.041, +0.041]<br>c [-0.071, +0.080] | +0.088<br>d [+0.002, +0.153]<br>c [-0.038, +0.201] |
| V4 chronological: fit to 31991, score from 32312 | S (schedule) | +0.014<br>d [-0.084, +0.090]<br>c [-0.096, +0.103] | -0.004<br>d [-0.085, +0.059]<br>c [-0.135, +0.073] | +0.031<br>d [-0.039, +0.086]<br>c [-0.045, +0.100] | -0.021<br>d [-0.133, +0.057]<br>c [-0.179, +0.051] | +0.010<br>d [-0.149, +0.119]<br>c [-0.208, +0.122] |
| V4 chronological: fit to 32310, score from 32312 | D10 | +0.003<br>d [-0.014, +0.019]<br>c [-0.010, +0.016] | +0.038<br>d [+0.014, +0.062]<br>c [+0.026, +0.057] | +0.053<br>d [+0.029, +0.077]<br>c [+0.031, +0.069] | -0.013<br>d [-0.024, +0.001]<br>c [-0.022, +0.006] | +0.041<br>d [+0.018, +0.065]<br>c [+0.024, +0.063] |
| V4 chronological: fit to 32310, score from 32312 | D650 | -0.049<br>d [-0.148, +0.025]<br>c [-0.206, +0.083] | +0.131<br>d [+0.072, +0.190]<br>c [+0.070, +0.206] | +0.100<br>d [+0.057, +0.141]<br>c [+0.063, +0.161] | -0.014<br>d [-0.049, +0.016]<br>c [-0.059, +0.034] | +0.088<br>d [+0.031, +0.138]<br>c [+0.015, +0.180] |
| V4 chronological: fit to 32310, score from 32312 | S (schedule) | -0.022<br>d [-0.064, +0.025]<br>c [-0.071, +0.048] | -0.017<br>d [-0.114, +0.063]<br>c [-0.148, +0.116] | +0.041<br>d [-0.003, +0.088]<br>c [+0.002, +0.099] | -0.084<br>d [-0.148, -0.031]<br>c [-0.157, +0.004] | -0.040<br>d [-0.122, +0.031]<br>c [-0.140, +0.065] |

## The ratio of the two MSE reductions, target L5

Registered form: (skill of TXN over TN) / (skill of TX over T0), draw by draw on the same draws. Beside it the absolute form (SSE(TN) - SSE(TXN)) / (SSE(T0) - SSE(TX)). Points: mean over splits of the per-split ratio.

| design | rule | (TXN over TN) / (TX over T0) | absolute form | TX over T0 (the denominator's skill) |
|---|---|---|---|---|
| V1 random (r2-r4) | D10 | 0.944<br>d [0.763, 1.195]<br>c [0.764, 1.196] | 0.931<br>d [0.737, 1.206]<br>c [0.739, 1.207] | +0.103<br>d [+0.079, +0.127]<br>c [+0.069, +0.140] |
| V1 random (r2-r4) | D650 | 0.766<br>d [0.679, 0.867]<br>c [0.657, 0.878] | 0.653<br>d [0.551, 0.783]<br>c [0.531, 0.798] | +0.336<br>d [+0.299, +0.372]<br>c [+0.285, +0.381] |
| V1 random (r2-r4) | S (schedule) | 0.772<br>d [0.648, 0.902]<br>c [0.613, 0.902] | 0.669<br>d [0.538, 0.820]<br>c [0.505, 0.826] | +0.327<br>d [+0.277, +0.375]<br>c [+0.263, +0.388] |
| V2 cluster-blocked (3 assignments) | D10 | 1.286<br>d [0.928, 1.984]<br>c [0.795, 2.752] | 1.321<br>d [0.935, 2.100]<br>c [0.791, 2.929] | +0.058<br>d [+0.033, +0.079]<br>c [+0.020, +0.088] |
| V2 cluster-blocked (3 assignments) | D650 | 1.057<br>d [0.853, 1.306]<br>c [0.745, 1.485] | 1.106<br>d [0.813, 1.472]<br>c [0.676, 1.780] | +0.282<br>d [+0.244, +0.318]<br>c [+0.215, +0.343] |
| V2 cluster-blocked (3 assignments) | S (schedule) | 0.812<br>d [0.609, 0.999]<br>c [0.560, 1.089] | 0.755<br>d [0.535, 0.982]<br>c [0.483, 1.108] | +0.293<br>d [+0.243, +0.341]<br>c [+0.215, +0.365] |
| V3 rolling origin (fifths 2-5 pooled) | D10 | 1.814<br>d [1.074, 6.128]<br>c [-6.429, 10.336] | 1.882<br>d [1.096, 6.419]<br>c [-6.731, 10.797] | +0.028<br>d [+0.004, +0.049]<br>c [-0.012, +0.067] |
| V3 rolling origin (fifths 2-5 pooled) | D650 | 1.383<br>d [1.020, 1.957]<br>c [0.710, 5.012] | 1.556<br>d [1.053, 2.393]<br>c [0.674, 6.593] | +0.190<br>d [+0.133, +0.242]<br>c [+0.032, +0.290] |
| V3 rolling origin (fifths 2-5 pooled) | S (schedule) | 0.756<br>d [0.570, 0.981]<br>c [0.512, 1.036] | 0.686<br>d [0.495, 0.934]<br>c [0.434, 1.002] | +0.231<br>d [+0.167, +0.291]<br>c [+0.144, +0.325] |
| V4 chronological: fit to 31991, score from 32312 | D10 | 0.380<br>d [-2.173, 0.882]<br>c [-1.001, 0.888] | 0.367<br>d [-2.146, 0.868]<br>c [-1.005, 0.863] | +0.089<br>d [+0.015, +0.139]<br>c [+0.031, +0.135] |
| V4 chronological: fit to 31991, score from 32312 | D650 | 1.037<br>d [0.536, 3.042]<br>c [0.124, 5.347] | 1.105<br>d [0.459, 4.073]<br>c [0.082, 7.665] | +0.237<br>d [+0.071, +0.358]<br>c [+0.040, +0.415] |
| V4 chronological: fit to 31991, score from 32312 | S (schedule) | 3.658<br>d [-15.949, 16.871]<br>c [-12.378, 15.993] | 4.023<br>d [-18.122, 18.808]<br>c [-13.992, 18.067] | +0.028<br>d [-0.134, +0.158]<br>c [-0.167, +0.170] |
| V4 chronological: fit to 32310, score from 32312 | D10 | 0.542<br>d [-0.212, 0.957]<br>c [0.023, 1.133] | 0.516<br>d [-0.200, 0.932]<br>c [0.021, 1.126] | +0.079<br>d [+0.036, +0.113]<br>c [+0.058, +0.112] |
| V4 chronological: fit to 32310, score from 32312 | D650 | 1.271<br>d [0.773, 2.651]<br>c [0.512, 3.484] | 1.384<br>d [0.707, 3.419]<br>c [0.399, 4.648] | +0.216<br>d [+0.102, +0.307]<br>c [+0.077, +0.359] |
| V4 chronological: fit to 32310, score from 32312 | S (schedule) | 0.245<br>d [-0.990, 0.695]<br>c [-2.998, 2.935] | 0.258<br>d [-1.078, 0.667]<br>c [-3.468, 3.201] | -0.254<br>d [-0.519, -0.054]<br>c [-0.537, +0.043] |

## The ratio of the two MSE reductions, target F

Registered form: (skill of TXN over TN) / (skill of TX over T0), draw by draw on the same draws. Beside it the absolute form (SSE(TN) - SSE(TXN)) / (SSE(T0) - SSE(TX)). Points: mean over splits of the per-split ratio.

| design | rule | (TXN over TN) / (TX over T0) | absolute form | TX over T0 (the denominator's skill) |
|---|---|---|---|---|
| V1 random (r2-r4) | D10 | 0.967<br>d [0.913, 1.026]<br>c [0.913, 1.032] | 0.966<br>d [0.910, 1.028]<br>c [0.910, 1.034] | +0.102<br>d [+0.085, +0.120]<br>c [+0.068, +0.145] |
| V1 random (r2-r4) | D650 | 0.788<br>d [0.705, 0.875]<br>c [0.684, 0.885] | 0.706<br>d [0.613, 0.807]<br>c [0.593, 0.820] | +0.287<br>d [+0.257, +0.315]<br>c [+0.240, +0.328] |
| V1 random (r2-r4) | S (schedule) | 0.800<br>d [0.713, 0.890]<br>c [0.699, 0.891] | 0.724<br>d [0.627, 0.830]<br>c [0.615, 0.830] | +0.283<br>d [+0.243, +0.321]<br>c [+0.227, +0.338] |
| V2 cluster-blocked (3 assignments) | D10 | 1.047<br>d [0.958, 1.157]<br>c [0.960, 1.219] | 1.051<br>d [0.959, 1.166]<br>c [0.959, 1.230] | +0.080<br>d [+0.064, +0.095]<br>c [+0.052, +0.113] |
| V2 cluster-blocked (3 assignments) | D650 | 1.075<br>d [0.903, 1.268]<br>c [0.821, 1.462] | 1.101<br>d [0.868, 1.372]<br>c [0.765, 1.659] | +0.237<br>d [+0.201, +0.271]<br>c [+0.176, +0.286] |
| V2 cluster-blocked (3 assignments) | S (schedule) | 0.841<br>d [0.710, 1.014]<br>c [0.673, 1.053] | 0.807<br>d [0.655, 1.014]<br>c [0.613, 1.073] | +0.260<br>d [+0.217, +0.301]<br>c [+0.196, +0.320] |
| V3 rolling origin (fifths 2-5 pooled) | D10 | 1.166<br>d [1.020, 1.361]<br>c [0.904, 1.578] | 1.178<br>d [1.024, 1.384]<br>c [0.904, 1.623] | +0.059<br>d [+0.045, +0.075]<br>c [+0.036, +0.091] |
| V3 rolling origin (fifths 2-5 pooled) | D650 | 1.400<br>d [1.073, 1.928]<br>c [0.817, 4.925] | 1.504<br>d [1.094, 2.175]<br>c [0.793, 5.832] | +0.156<br>d [+0.111, +0.198]<br>c [+0.030, +0.237] |
| V3 rolling origin (fifths 2-5 pooled) | S (schedule) | 0.826<br>d [0.666, 1.000]<br>c [0.611, 1.060] | 0.776<br>d [0.607, 0.969]<br>c [0.551, 1.031] | +0.193<br>d [+0.149, +0.235]<br>c [+0.137, +0.251] |
| V4 chronological: fit to 31991, score from 32312 | D10 | 1.469<br>d [0.854, 5.952]<br>c [0.869, 3.295] | 1.486<br>d [0.857, 6.044]<br>c [0.866, 3.337] | +0.026<br>d [+0.001, +0.056]<br>c [+0.004, +0.055] |
| V4 chronological: fit to 31991, score from 32312 | D650 | 1.159<br>d [0.508, 3.207]<br>c [0.328, 5.543] | 1.176<br>d [0.493, 3.458]<br>c [0.296, 6.093] | +0.088<br>d [+0.018, +0.143]<br>c [+0.006, +0.165] |
| V4 chronological: fit to 31991, score from 32312 | S (schedule) | -0.129<br>d [-9.080, 9.770]<br>c [-11.937, 13.105] | -0.128<br>d [-9.160, 10.023]<br>c [-11.967, 13.304] | +0.031<br>d [-0.039, +0.086]<br>c [-0.045, +0.100] |
| V4 chronological: fit to 32310, score from 32312 | D10 | 0.720<br>d [0.337, 1.155]<br>c [0.516, 1.189] | 0.718<br>d [0.333, 1.170]<br>c [0.509, 1.193] | +0.053<br>d [+0.029, +0.077]<br>c [+0.031, +0.069] |
| V4 chronological: fit to 32310, score from 32312 | D650 | 1.304<br>d [0.811, 2.151]<br>c [0.594, 2.649] | 1.367<br>d [0.792, 2.438]<br>c [0.547, 3.164] | +0.100<br>d [+0.057, +0.141]<br>c [+0.063, +0.161] |
| V4 chronological: fit to 32310, score from 32312 | S (schedule) | -0.412<br>d [-8.998, 4.358]<br>c [-14.748, 3.679] | -0.421<br>d [-9.149, 4.543]<br>c [-14.971, 3.946] | +0.041<br>d [-0.003, +0.088]<br>c [+0.002, +0.099] |

## The halves: TEN and TBN over TN (V1, V4)

TEN = TN with the ECEI half of the fast scalars (four of row t, three anchor scalars); TBN with the BES half.

| design | target | rule | TEN over TN (ECEI) | TBN over TN (BES) | TXN over TN (both) | RMSE TN / TEN / TBN / TXN |
|---|---|---|---|---|---|---|
| V1 random (r2-r4) | L5 | D650 | +0.170<br>d [+0.127, +0.214]<br>c [+0.110, +0.228] | +0.181<br>d [+0.147, +0.217]<br>c [+0.138, +0.227] | +0.258<br>d [+0.222, +0.293]<br>c [+0.205, +0.304] | 208.8 / 190.1 / 188.9 / 179.9 |
| V1 random (r2-r4) | L5 | S (schedule) | +0.179<br>d [+0.133, +0.223]<br>c [+0.112, +0.238] | +0.145<br>d [+0.101, +0.189]<br>c [+0.090, +0.197] | +0.253<br>d [+0.202, +0.305]<br>c [+0.184, +0.319] | 202.6 / 183.5 / 187.3 / 175.1 |
| V4 chronological: fit to 31991, score from 32312 | L5 | D650 | +0.122<br>d [+0.020, +0.212]<br>c [+0.007, +0.260] | +0.131<br>d [+0.011, +0.249]<br>c [-0.018, +0.198] | +0.245<br>d [+0.146, +0.341]<br>c [+0.061, +0.357] | 208.9 / 195.7 / 194.7 / 181.4 |
| V4 chronological: fit to 31991, score from 32312 | L5 | S (schedule) | +0.124<br>d [-0.019, +0.247]<br>c [-0.152, +0.248] | +0.061<br>d [-0.269, +0.270]<br>c [-0.207, +0.320] | +0.102<br>d [-0.059, +0.230]<br>c [-0.156, +0.242] | 193.9 / 181.4 / 187.8 / 183.7 |
| V4 chronological: fit to 32310, score from 32312 | L5 | D650 | +0.128<br>d [-0.024, +0.274]<br>c [-0.102, +0.405] | +0.183<br>d [+0.065, +0.299]<br>c [+0.044, +0.332] | +0.275<br>d [+0.177, +0.384]<br>c [+0.151, +0.407] | 198.1 / 184.9 / 179.1 / 168.7 |
| V4 chronological: fit to 32310, score from 32312 | L5 | S (schedule) | -0.208<br>d [-0.499, -0.005]<br>c [-0.710, +0.145] | +0.051<br>d [-0.099, +0.177]<br>c [-0.045, +0.142] | -0.062<br>d [-0.258, +0.079]<br>c [-0.488, +0.150] | 174.7 / 191.9 / 170.2 / 180.0 |
| V1 random (r2-r4) | F | D650 | +0.150<br>d [+0.119, +0.181]<br>c [+0.100, +0.198] | +0.153<br>d [+0.130, +0.177]<br>c [+0.123, +0.186] | +0.226<br>d [+0.196, +0.257]<br>c [+0.180, +0.268] | 260.5 / 240.2 / 239.7 / 229.2 |
| V1 random (r2-r4) | F | S (schedule) | +0.150<br>d [+0.112, +0.185]<br>c [+0.090, +0.205] | +0.148<br>d [+0.116, +0.178]<br>c [+0.110, +0.183] | +0.226<br>d [+0.192, +0.260]<br>c [+0.174, +0.277] | 252.4 / 232.8 / 233.0 / 222.0 |
| V4 chronological: fit to 31991, score from 32312 | F | D650 | +0.020<br>d [-0.055, +0.089]<br>c [-0.121, +0.153] | +0.020<br>d [-0.056, +0.083]<br>c [-0.067, +0.080] | +0.101<br>d [+0.032, +0.165]<br>c [+0.038, +0.180] | 299.8 / 296.8 / 296.7 / 284.2 |
| V4 chronological: fit to 31991, score from 32312 | F | S (schedule) | +0.026<br>d [-0.023, +0.076]<br>c [-0.029, +0.080] | -0.014<br>d [-0.082, +0.046]<br>c [-0.081, +0.064] | -0.004<br>d [-0.085, +0.059]<br>c [-0.135, +0.073] | 296.1 / 292.2 / 298.1 / 296.6 |
| V4 chronological: fit to 32310, score from 32312 | F | D650 | +0.049<br>d [-0.015, +0.116]<br>c [-0.040, +0.169] | +0.097<br>d [+0.043, +0.151]<br>c [+0.042, +0.152] | +0.131<br>d [+0.072, +0.190]<br>c [+0.070, +0.206] | 299.4 / 291.9 / 284.6 / 279.2 |
| V4 chronological: fit to 32310, score from 32312 | F | S (schedule) | -0.060<br>d [-0.173, +0.039]<br>c [-0.214, +0.084] | +0.025<br>d [-0.052, +0.081]<br>c [-0.029, +0.090] | -0.017<br>d [-0.114, +0.063]<br>c [-0.148, +0.116] | 295.6 / 304.3 / 291.9 / 298.1 |

## By time since the last notch (V1; a stratum on an input)

Time since the last notch = s(t) - s(j+1), j+1 the first row after the most recent absent step (15-45 ms) that ends at or before t in the block of t. Within 100 ms: below 0.095 s (the first row after the step and the next nine). The third stratum holds the rows with no absent step earlier in the block; under S it is empty.

### Target L5

| set, rule | stratum | rows / discharges / clusters | RMSE T0 / TN / TX / TXN | TXN over TN | TX over T0 | TN over T0 | TXN over TX |
|---|---|---|---|---|---|---|---|
| common, D650 | within 100 ms after an absent step | 11,278 / 464 / 65 | 296.4 / 254.7 / 220.5 / 212.4 | +0.305<br>d [+0.228, +0.373]<br>c [+0.230, +0.375] | +0.447<br>d [+0.358, +0.525]<br>c [+0.324, +0.537] | +0.262<br>d [+0.153, +0.361]<br>c [+0.119, +0.385] | +0.072<br>d [+0.006, +0.141]<br>c [+0.016, +0.121] |
| common, D650 | later rows after an absent step | 132,937 / 499 / 71 | 212.4 / 201.1 / 177.5 / 173.7 | +0.254<br>d [+0.213, +0.296]<br>c [+0.196, +0.309] | +0.302<br>d [+0.262, +0.339]<br>c [+0.253, +0.345] | +0.103<br>d [+0.041, +0.154]<br>c [+0.034, +0.162] | +0.042<br>d [+0.019, +0.064]<br>c [+0.017, +0.066] |
| common, D650 | no absent step earlier in the block | 16,128 / 388 / 65 | 274.8 / 233.6 / 209.7 / 203.4 | +0.241<br>d [+0.155, +0.324]<br>c [+0.163, +0.315] | +0.417<br>d [+0.336, +0.490]<br>c [+0.312, +0.504] | +0.277<br>d [+0.196, +0.351]<br>c [+0.159, +0.373] | +0.059<br>d [-0.008, +0.112]<br>c [-0.006, +0.119] |
| S, S (schedule) | within 100 ms after an absent step | 12,449 / 489 / 70 | 194.1 / 187.4 / 171.7 / 169.0 | +0.186<br>d [+0.106, +0.259]<br>c [+0.097, +0.264] | +0.218<br>d [+0.139, +0.289]<br>c [+0.124, +0.295] | +0.068<br>d [-0.007, +0.138]<br>c [-0.021, +0.137] | +0.031<br>d [-0.012, +0.072]<br>c [-0.029, +0.077] |
| S, S (schedule) | later rows after an absent step | 135,388 / 501 / 71 | 219.6 / 203.9 / 179.1 / 175.6 | +0.258<br>d [+0.205, +0.312]<br>c [+0.184, +0.326] | +0.335<br>d [+0.283, +0.383]<br>c [+0.268, +0.398] | +0.138<br>d [+0.088, +0.185]<br>c [+0.081, +0.195] | +0.038<br>d [+0.004, +0.071]<br>c [+0.001, +0.071] |
| S, S (schedule) | no absent step earlier in the block | 0 | - | - | - | - | - |

Rows per stratum (all V1 scored rows of the set): V1|L5|common: within 100 ms after an absent step 11,278, later rows after an absent step 132,937, no absent step earlier in the block 16,128, no absent step earlier, but one later in the block 13,140; V1|L5|S: within 100 ms after an absent step 12,449, later rows after an absent step 135,388, no absent step earlier in the block 0, no absent step earlier, but one later in the block 0.

### Target F

| set, rule | stratum | rows / discharges / clusters | RMSE T0 / TN / TX / TXN | TXN over TN | TX over T0 | TN over T0 | TXN over TX |
|---|---|---|---|---|---|---|---|
| common, D650 | within 100 ms after an absent step | 11,490 / 475 / 65 | 348.6 / 316.8 / 273.3 / 269.2 | +0.278<br>d [+0.219, +0.333]<br>c [+0.224, +0.317] | +0.385<br>d [+0.321, +0.443]<br>c [+0.307, +0.450] | +0.174<br>d [+0.099, +0.242]<br>c [+0.064, +0.271] | +0.030<br>d [-0.010, +0.064]<br>c [-0.015, +0.065] |
| common, D650 | later rows after an absent step | 136,968 / 503 / 71 | 261.1 / 250.6 / 222.9 / 219.8 | +0.231<br>d [+0.197, +0.265]<br>c [+0.178, +0.279] | +0.271<br>d [+0.239, +0.302]<br>c [+0.221, +0.316] | +0.078<br>d [+0.040, +0.115]<br>c [+0.037, +0.119] | +0.027<br>d [+0.009, +0.049]<br>c [+0.008, +0.051] |
| common, D650 | no absent step earlier in the block | 16,960 / 394 / 66 | 326.8 / 294.5 / 275.4 / 270.7 | +0.156<br>d [+0.100, +0.208]<br>c [+0.099, +0.214] | +0.290<br>d [+0.226, +0.348]<br>c [+0.206, +0.368] | +0.187<br>d [+0.135, +0.237]<br>c [+0.102, +0.271] | +0.034<br>d [-0.010, +0.073]<br>c [-0.016, +0.081] |
| S, S (schedule) | within 100 ms after an absent step | 12,675 / 500 / 70 | 261.8 / 258.4 / 236.3 / 234.0 | +0.180<br>d [+0.121, +0.238]<br>c [+0.116, +0.239] | +0.185<br>d [+0.126, +0.241]<br>c [+0.122, +0.244] | +0.026<br>d [-0.008, +0.058]<br>c [-0.010, +0.063] | +0.020<br>d [-0.003, +0.047]<br>c [-0.006, +0.056] |
| S, S (schedule) | later rows after an absent step | 139,528 / 505 / 71 | 265.7 / 251.8 / 223.6 / 220.9 | +0.231<br>d [+0.195, +0.266]<br>c [+0.176, +0.284] | +0.292<br>d [+0.251, +0.330]<br>c [+0.234, +0.349] | +0.101<br>d [+0.068, +0.133]<br>c [+0.067, +0.137] | +0.024<br>d [+0.006, +0.043]<br>c [+0.010, +0.044] |
| S, S (schedule) | no absent step earlier in the block | 0 | - | - | - | - | - |

Rows per stratum (all V1 scored rows of the set): V1|F|common: within 100 ms after an absent step 11,490, later rows after an absent step 136,968, no absent step earlier in the block 16,960, no absent step earlier, but one later in the block 13,661; V1|F|S: within 100 ms after an absent step 12,675, later rows after an absent step 139,528, no absent step earlier in the block 0, no absent step earlier, but one later in the block 0.

## V1 by chronological block of the test discharge, target L5

| block | rule | rows / discharges / clusters | TN over T0 | TXN over TN | TX over T0 | TXN over TX | TEN over TN | TBN over TN |
|---|---|---|---|---|---|---|---|---|
| up to 31991 | D10 | 127,270 / 375 / 48 | +0.018<br>d [-0.018, +0.045]<br>c [-0.017, +0.050] | +0.110<br>d [+0.081, +0.150]<br>c [+0.066, +0.172] | +0.117<br>d [+0.089, +0.143]<br>c [+0.073, +0.163] | +0.011<br>d [-0.003, +0.024]<br>c [-0.004, +0.027] | - | - |
| up to 31991 | D650 | 127,270 / 375 / 48 | +0.156<br>d [+0.103, +0.203]<br>c [+0.093, +0.211] | +0.246<br>d [+0.208, +0.281]<br>c [+0.191, +0.294] | +0.332<br>d [+0.291, +0.370]<br>c [+0.272, +0.383] | +0.047<br>d [+0.017, +0.073]<br>c [+0.017, +0.075] | +0.149<br>d [+0.104, +0.192]<br>c [+0.084, +0.210] | +0.174<br>d [+0.139, +0.209]<br>c [+0.131, +0.224] |
| 32002-32310 | D10 | 22,927 / 112 / 14 | -0.003<br>d [-0.047, +0.027]<br>c [-0.040, +0.028] | +0.068<br>d [+0.025, +0.103]<br>c [+0.027, +0.100] | +0.063<br>d [+0.017, +0.103]<br>c [+0.035, +0.091] | +0.003<br>d [-0.021, +0.023]<br>c [-0.020, +0.022] | - | - |
| 32002-32310 | D650 | 22,927 / 112 / 14 | +0.171<br>d [+0.058, +0.261]<br>c [+0.077, +0.304] | +0.301<br>d [+0.213, +0.380]<br>c [+0.121, +0.400] | +0.386<br>d [+0.293, +0.464]<br>c [+0.290, +0.466] | +0.055<br>d [+0.011, +0.102]<br>c [+0.005, +0.095] | +0.226<br>d [+0.114, +0.320]<br>c [+0.049, +0.340] | +0.209<br>d [+0.125, +0.292]<br>c [+0.085, +0.303] |
| from 32312 | D10 | 10,146 / 75 / 18 | +0.029<br>d [-0.012, +0.064]<br>c [-0.027, +0.078] | +0.050<br>d [-0.018, +0.100]<br>c [-0.010, +0.097] | +0.086<br>d [+0.028, +0.138]<br>c [+0.044, +0.141] | -0.009<br>d [-0.043, +0.015]<br>c [-0.042, +0.024] | - | - |
| from 32312 | D650 | 10,146 / 75 / 18 | -0.105<br>d [-0.517, +0.137]<br>c [-0.632, +0.252] | +0.252<br>d [+0.010, +0.463]<br>c [+0.009, +0.505] | +0.157<br>d [-0.032, +0.282]<br>c [-0.016, +0.332] | +0.021<br>d [-0.057, +0.101]<br>c [-0.051, +0.091] | +0.244<br>d [+0.076, +0.419]<br>c [+0.015, +0.460] | +0.163<br>d [-0.130, +0.401]<br>c [-0.123, +0.439] |
| up to 31991 | S (schedule) | 119,655 / 349 / 44 | +0.138<br>d [+0.086, +0.187]<br>c [+0.075, +0.199] | +0.269<br>d [+0.215, +0.324]<br>c [+0.192, +0.344] | +0.339<br>d [+0.284, +0.393]<br>c [+0.263, +0.413] | +0.046<br>d [+0.013, +0.083]<br>c [+0.011, +0.081] | +0.188<br>d [+0.137, +0.238]<br>c [+0.110, +0.260] | +0.160<br>d [+0.112, +0.208]<br>c [+0.098, +0.219] |
| 32002-32310 | S (schedule) | 21,002 / 104 / 14 | +0.137<br>d [-0.012, +0.247]<br>c [+0.006, +0.256] | +0.213<br>d [+0.096, +0.311]<br>c [+0.050, +0.329] | +0.319<br>d [+0.203, +0.417]<br>c [+0.218, +0.375] | +0.002<br>d [-0.084, +0.085]<br>c [-0.104, +0.093] | +0.159<br>d [+0.041, +0.254]<br>c [+0.017, +0.259] | +0.101<br>d [+0.006, +0.187]<br>c [-0.034, +0.200] |
| from 32312 | S (schedule) | 7,180 / 50 / 14 | -0.002<br>d [-0.244, +0.172]<br>c [-0.274, +0.214] | +0.086<br>d [-0.082, +0.228]<br>c [-0.136, +0.231] | +0.066<br>d [-0.129, +0.221]<br>c [-0.168, +0.245] | +0.023<br>d [-0.077, +0.122]<br>c [-0.095, +0.135] | +0.061<br>d [-0.078, +0.254]<br>c [-0.121, +0.240] | +0.021<br>d [-0.195, +0.182]<br>c [-0.162, +0.191] |

## V1 by chronological block of the test discharge, target F

| block | rule | rows / discharges / clusters | TN over T0 | TXN over TN | TX over T0 | TXN over TX | TEN over TN | TBN over TN |
|---|---|---|---|---|---|---|---|---|
| up to 31991 | D10 | 130,192 / 376 / 48 | +0.002<br>d [-0.005, +0.009]<br>c [-0.006, +0.010] | +0.136<br>d [+0.112, +0.160]<br>c [+0.091, +0.188] | +0.140<br>d [+0.115, +0.164]<br>c [+0.093, +0.196] | -0.003<br>d [-0.010, +0.004]<br>c [-0.011, +0.005] | - | - |
| up to 31991 | D650 | 130,192 / 376 / 48 | +0.124<br>d [+0.084, +0.160]<br>c [+0.078, +0.169] | +0.253<br>d [+0.221, +0.284]<br>c [+0.198, +0.302] | +0.325<br>d [+0.290, +0.357]<br>c [+0.264, +0.374] | +0.032<br>d [+0.005, +0.058]<br>c [+0.005, +0.058] | +0.166<br>d [+0.128, +0.201]<br>c [+0.101, +0.225] | +0.171<br>d [+0.143, +0.198]<br>c [+0.132, +0.216] |
| 32002-32310 | D10 | 24,564 / 113 / 14 | -0.001<br>d [-0.013, +0.010]<br>c [-0.010, +0.012] | +0.044<br>d [+0.021, +0.068]<br>c [+0.019, +0.072] | +0.046<br>d [+0.024, +0.068]<br>c [+0.019, +0.073] | -0.003<br>d [-0.013, +0.006]<br>c [-0.013, +0.005] | - | - |
| 32002-32310 | D650 | 24,564 / 113 / 14 | +0.087<br>d [+0.013, +0.147]<br>c [+0.007, +0.153] | +0.189<br>d [+0.114, +0.271]<br>c [+0.084, +0.280] | +0.239<br>d [+0.169, +0.303]<br>c [+0.172, +0.288] | +0.028<br>d [+0.002, +0.056]<br>c [-0.003, +0.063] | +0.129<br>d [+0.057, +0.209]<br>c [+0.028, +0.235] | +0.138<br>d [+0.082, +0.195]<br>c [+0.066, +0.198] |
| from 32312 | D10 | 10,662 / 75 / 18 | +0.003<br>d [-0.010, +0.017]<br>c [-0.009, +0.019] | +0.062<br>d [+0.034, +0.089]<br>c [+0.038, +0.083] | +0.066<br>d [+0.040, +0.090]<br>c [+0.039, +0.088] | -0.001<br>d [-0.013, +0.012]<br>c [-0.011, +0.011] | - | - |
| from 32312 | D650 | 10,662 / 75 / 18 | -0.015<br>d [-0.089, +0.043]<br>c [-0.118, +0.081] | +0.115<br>d [+0.053, +0.178]<br>c [+0.067, +0.190] | +0.099<br>d [+0.039, +0.149]<br>c [+0.052, +0.168] | +0.003<br>d [-0.021, +0.025]<br>c [-0.017, +0.026] | +0.087<br>d [+0.033, +0.146]<br>c [+0.025, +0.167] | +0.063<br>d [-0.027, +0.126]<br>c [+0.007, +0.131] |
| up to 31991 | S (schedule) | 122,230 / 350 / 44 | +0.116<br>d [+0.079, +0.152]<br>c [+0.078, +0.157] | +0.267<br>d [+0.223, +0.308]<br>c [+0.200, +0.330] | +0.333<br>d [+0.284, +0.377]<br>c [+0.266, +0.400] | +0.027<br>d [+0.004, +0.053]<br>c [+0.007, +0.057] | +0.180<br>d [+0.132, +0.226]<br>c [+0.098, +0.251] | +0.171<br>d [+0.131, +0.209]<br>c [+0.126, +0.218] |
| 32002-32310 | S (schedule) | 22,422 / 106 / 14 | +0.055<br>d [-0.015, +0.123]<br>c [+0.005, +0.115] | +0.155<br>d [+0.094, +0.210]<br>c [+0.081, +0.213] | +0.186<br>d [+0.112, +0.258]<br>c [+0.135, +0.225] | +0.019<br>d [-0.010, +0.051]<br>c [+0.003, +0.034] | +0.096<br>d [+0.043, +0.148]<br>c [+0.016, +0.156] | +0.116<br>d [+0.056, +0.173]<br>c [+0.040, +0.183] |
| from 32312 | S (schedule) | 7,551 / 50 / 14 | +0.010<br>d [-0.045, +0.076]<br>c [-0.048, +0.081] | +0.061<br>d [-0.023, +0.131]<br>c [-0.021, +0.147] | +0.060<br>d [-0.001, +0.125]<br>c [-0.003, +0.132] | +0.010<br>d [-0.029, +0.050]<br>c [-0.041, +0.052] | +0.029<br>d [-0.030, +0.094]<br>c [-0.039, +0.103] | +0.012<br>d [-0.087, +0.084]<br>c [-0.076, +0.098] |

## Controls (V1, D650, L5)

TNp: TN refitted with the nineteen beam scalars permuted jointly over rows (within the fit rows and within the test rows). TXNp: TXN refitted with the fourteen fast scalars permuted jointly in the same way.

| quantity | value |
|---|---|
| TNp over T0 (N11) | -0.003<br>d [-0.007, +0.001]<br>c [-0.008, +0.001] |
| TN over TNp | +0.151<br>d [+0.096, +0.195]<br>c [+0.088, +0.201] |
| TXNp over TN | +0.001<br>d [-0.005, +0.008]<br>c [-0.006, +0.009] |
| TXN over TXNp | +0.257<br>d [+0.221, +0.292]<br>c [+0.204, +0.303] |
| RMSE T0 / TNp / TN / TXNp / TXN | 226.2 / 226.6 / 208.8 / 208.6 / 179.9 |

## Sensitivity: the statements' quantities without the thirteen files (not used for a verdict)

Scored rows outside the thirteen files whose NBI-2 power columns are on another scale; fits unchanged. Under V3 and V4 those files are never scored, and the value is the main one.

| Id | value: point; d interval; c interval | rows / discharges / clusters | main value (all scored rows) |
|---|---|---|---|
| N1 | +0.141; d [+0.086, +0.186]; c [+0.078, +0.192] | 158,560 / 551 / 76 | +0.148; c [+0.086, +0.199] |
| N2 | +0.257; d [+0.221, +0.292]; c [+0.203, +0.305] | 158,560 / 551 / 76 | +0.258; c [+0.205, +0.304] |
| N3 | +0.292; d [+0.235, +0.355]; c [+0.204, +0.372] | 158,560 / 551 / 76 | +0.298; c [+0.206, +0.378] |
| N4 | +0.263; d [+0.198, +0.324]; c [+0.162, +0.363] (same rows as main) | 112,199 / 441 / 60 | +0.263; c [+0.162, +0.363] |
| N5 | +0.245; d [+0.146, +0.341]; c [+0.061, +0.357] (same rows as main) | 10,146 / 75 / 18 | +0.245; c [+0.061, +0.357] |
| N6 | +0.253; d [+0.204, +0.303]; c [+0.184, +0.316] | 146,197 / 493 / 68 | +0.253; c [+0.184, +0.319] |
| N7 | +0.102; d [-0.059, +0.230]; c [-0.156, +0.242] (same rows as main) | 7,180 / 50 / 14 | +0.102; c [-0.156, +0.242] |
| N8 | -0.100; d [-0.274, +0.042]; c [-0.282, +0.119] (same rows as main) | 7,180 / 50 / 14 | -0.100; c [-0.282, +0.119] |
| N9 | +0.045; d [+0.021, +0.067]; c [+0.020, +0.067] | 158,560 / 551 / 76 | +0.047; c [+0.023, +0.070] |
| N10 | +0.097; d [+0.073, +0.129]; c [+0.062, +0.146] | 158,560 / 551 / 76 | +0.097; c [+0.063, +0.144] |
| N11 | -0.003; d [-0.007, +0.001]; c [-0.008, +0.001] | 158,560 / 551 / 76 | -0.003; c [-0.008, +0.001] |

## Counts

- Common set: F 165,418, L5 160,343 rows; S set: F 152,203, L5 147,837 rows (equal to B.75's sets; no row lacks a beam scalar). D1 rows of the thirteen files: 2,913 (common L5 1,783, S L5 1,640).

| design | common F | common L5 | S set F | S set L5 | fit rows per unit, TXN L5: D10 / D650 / S |
|---|---|---|---|---|---|
| V1 random (r2-r4) | 165,418 / 564 / 79 | 160,343 / 562 / 79 | 152,203 / 506 / 71 | 147,837 / 503 / 71 | 122,684-133,674 / 122,684-133,674 / 112,909-123,415 |
| V2 cluster-blocked (3 assignments) | 165,418 / 564 / 79 | 160,343 / 562 / 79 | 152,203 / 506 / 71 | 147,837 / 503 / 71 | 113,440-138,804 / 113,440-138,804 / 105,194-128,163 |
| V3 rolling origin (fifths 2-5 pooled) | 116,471 / 442 / 60 | 112,199 / 441 / 60 | 106,417 / 398 / 56 | 102,684 / 395 / 56 | 48,144-143,366 / 48,144-143,366 / 45,153-135,332 |
| V4 chronological: fit to 31991, score from 32312 | 10,662 / 75 / 18 | 10,146 / 75 / 18 | 7,551 / 50 / 14 | 7,180 / 50 / 14 | 127,270-127,270 / 127,270-127,270 / 119,655-119,655 |
| V4 chronological: fit to 32310, score from 32312 | 10,662 / 75 / 18 | 10,146 / 75 / 18 | 7,551 / 50 / 14 | 7,180 / 50 / 14 | 150,197-150,197 / 150,197-150,197 / 140,657-140,657 |

Cells: rows / discharges / clusters scored.

Readout run time 43 s.

## Limits (registration section 8, unchanged)

- Exploratory on used data.
- The beam columns are samples on the 10 ms rows; the notch itself has no row. The power in the first row after a notch is a sample of the returning beam and carries the same event as the BES sample of that row.
- Only the powers and one voltage are used; currents and energy fractions are not. No beam-deposition or heating calculation is made.
- The beam waveforms are what was delivered, not what was programmed; in real time the programmed waveform is known earlier.
- One estimator per rule; the level target is built from reported values.

## Sensitivity: refit without the thirteen files (the session lead's validity rule; not judged as the registered statements)

Order of events (Addendum 1 sections 11 to 13). The session lead's instruction to leave the thirteen files out of every fit and score was sent before the fits and did not reach the implementing agent before the main readout above had been run and read. The main readout uses the beam columns as delivered, as the registration reads, and its statement table is the judged one. This refit was run afterwards, with every number of the main readout known.

Rule: a file-source pair (sources NB1B, NB1C, NB2A, NB2B, NB2C) is invalid if its power column is above 5.0 MW or below -0.1 MW in any grid row of the file. Files selected: 30801, 30805, 30809, 30810, 30811, 30812, 30842, 30843, 30845, 30895, 30906, 30907, 30909 (exactly the thirteen named beforehand: True). Every learner of the batch is fitted again with the rows of those files removed from fit and score; rules, targets, designs, learners, intervals (seed 20261179, cluster and discharge forms) as in the main readout. Checks: `docs/paper/b79_checks.md`, section 'Refit without the thirteen files'. Per-unit estimates: `data/.b79/refit_valid/fits/` (main run: `data/.b79/fits/`).

### Statements N1 to N11: main beside the refit (target L5)

| Id | statement | main: point; c interval; verdict | refit: point; d interval; c interval | refit: the statement's cluster-interval rule met |
|---|---|---|---|---|
| N1 | V1, D650: TN over T0 lies above zero (the beam waveforms add to the history) | +0.148; c [+0.086, +0.199]; **holds** | +0.149; d [+0.091, +0.196]; c [+0.084, +0.200] | yes |
| N2 | V1, D650: TXN over TN lies above zero (the fast diagnostics add beyond the beam waveforms) | +0.258; c [+0.205, +0.304]; **holds** | +0.248; d [+0.212, +0.282]; c [+0.195, +0.295] | yes |
| N3 | V2, D650: TXN over TN lies above zero | +0.298; c [+0.206, +0.378]; **holds** | +0.299; d [+0.246, +0.359]; c [+0.218, +0.377] | yes |
| N4 | V3, D650: TXN over TN lies above zero | +0.263; c [+0.162, +0.363]; **holds** | +0.246; d [+0.191, +0.298]; c [+0.165, +0.333] | yes |
| N5 | V4 (fit to 31991), D650: TXN over TN lies above zero | +0.245; c [+0.061, +0.357]; **holds** | +0.249; d [+0.127, +0.355]; c [+0.120, +0.369] | yes |
| N6 | V1, S: TXN over TN lies above zero | +0.253; c [+0.184, +0.319]; **holds** | +0.250; d [+0.199, +0.303]; c [+0.178, +0.316] | yes |
| N7 | V4 (fit to 31991), S: TXN over TN lies above zero | +0.102; c [-0.156, +0.242]; **does not hold** | -0.070; d [-0.294, +0.093]; c [-0.437, +0.076] | no |
| N8 | V4 (fit to 31991), S: TN over T0 lies above zero | -0.100; c [-0.282, +0.119]; **does not hold** | +0.091; d [-0.089, +0.245]; c [-0.084, +0.236] | no |
| N9 | V1, D650: TXN over TX lies above zero (the beam waveforms add on top of the fast diagnostics) | +0.047; c [+0.023, +0.070]; **holds** | +0.040; d [+0.006, +0.067]; c [+0.007, +0.065] | yes |
| N10 | V1, D10: TXN over TN lies above zero | +0.097; c [+0.063, +0.144]; **holds** | +0.093; d [+0.071, +0.117]; c [+0.061, +0.131] | yes |
| N11 | V1, D650, beam scalars permuted: the interval of TN over T0 contains zero or lies below it | -0.003; c [-0.008, +0.001]; **holds** | -0.003; d [-0.006, +0.001]; c [-0.007, +0.002] | yes |

### Skills of the refit by design and rule, target L5

Cells: refit skill, d interval, c interval; then the main point.

| design | rule | rows / discharges / clusters | TN over T0 | TXN over TN | TX over T0 | TXN over TX |
|---|---|---|---|---|---|---|
| V1 random (r2-r4) | D10 | 158,560 / 551 / 76 | +0.018<br>d [-0.003, +0.036]<br>c [-0.003, +0.038]<br>main +0.014 | +0.093<br>d [+0.071, +0.117]<br>c [+0.061, +0.131]<br>main +0.097 | +0.101<br>d [+0.075, +0.125]<br>c [+0.067, +0.139]<br>main +0.103 | +0.009<br>d [-0.001, +0.018]<br>c [-0.002, +0.019]<br>main +0.007 |
| V1 random (r2-r4) | D650 | 158,560 / 551 / 76 | +0.149<br>d [+0.091, +0.196]<br>c [+0.084, +0.200]<br>main +0.148 | +0.248<br>d [+0.212, +0.282]<br>c [+0.195, +0.295]<br>main +0.258 | +0.334<br>d [+0.296, +0.371]<br>c [+0.281, +0.379]<br>main +0.336 | +0.040<br>d [+0.006, +0.067]<br>c [+0.007, +0.065]<br>main +0.047 |
| V1 random (r2-r4) | S (schedule) | 146,197 / 493 / 68 | +0.114<br>d [+0.049, +0.182]<br>c [+0.046, +0.185]<br>main +0.133 | +0.250<br>d [+0.199, +0.303]<br>c [+0.178, +0.316]<br>main +0.253 | +0.309<br>d [+0.249, +0.358]<br>c [+0.237, +0.366]<br>main +0.327 | +0.039<br>d [+0.010, +0.067]<br>c [+0.010, +0.066]<br>main +0.038 |
| V2 cluster-blocked (3 assignments) | D10 | 158,560 / 551 / 76 | -0.021<br>d [-0.056, +0.001]<br>c [-0.073, +0.007]<br>main -0.026 | +0.075<br>d [+0.054, +0.095]<br>c [+0.046, +0.103]<br>main +0.074 | +0.061<br>d [+0.034, +0.086]<br>c [+0.021, +0.093]<br>main +0.058 | -0.006<br>d [-0.022, +0.014]<br>c [-0.023, +0.022]<br>main -0.009 |
| V2 cluster-blocked (3 assignments) | D650 | 158,560 / 551 / 76 | -0.039<br>d [-0.135, +0.049]<br>c [-0.201, +0.105]<br>main -0.044 | +0.299<br>d [+0.246, +0.359]<br>c [+0.218, +0.377]<br>main +0.298 | +0.281<br>d [+0.237, +0.321]<br>c [+0.208, +0.342]<br>main +0.282 | -0.013<br>d [-0.054, +0.030]<br>c [-0.068, +0.041]<br>main -0.020 |
| V2 cluster-blocked (3 assignments) | S (schedule) | 146,197 / 493 / 68 | +0.073<br>d [+0.017, +0.126]<br>c [-0.012, +0.157]<br>main +0.072 | +0.239<br>d [+0.190, +0.288]<br>c [+0.165, +0.305]<br>main +0.238 | +0.293<br>d [+0.237, +0.346]<br>c [+0.200, +0.369]<br>main +0.293 | +0.001<br>d [-0.033, +0.033]<br>c [-0.037, +0.036]<br>main +0.000 |
| V3 rolling origin (fifths 2-5 pooled) | D10 | 112,199 / 441 / 60 | -0.024<br>d [-0.044, -0.007]<br>c [-0.052, +0.002]<br>main -0.037 | +0.032<br>d [+0.009, +0.052]<br>c [+0.003, +0.063]<br>main +0.050 | +0.021<br>d [-0.002, +0.042]<br>c [-0.012, +0.053]<br>main +0.028 | -0.012<br>d [-0.027, +0.001]<br>c [-0.036, +0.009]<br>main -0.013 |
| V3 rolling origin (fifths 2-5 pooled) | D650 | 112,199 / 441 / 60 | -0.131<br>d [-0.230, -0.046]<br>c [-0.361, +0.031]<br>main -0.126 | +0.246<br>d [+0.191, +0.298]<br>c [+0.165, +0.333]<br>main +0.263 | +0.156<br>d [+0.101, +0.208]<br>c [-0.002, +0.266]<br>main +0.190 | -0.011<br>d [-0.052, +0.028]<br>c [-0.082, +0.063]<br>main -0.025 |
| V3 rolling origin (fifths 2-5 pooled) | S (schedule) | 102,684 / 395 / 56 | +0.086<br>d [+0.036, +0.132]<br>c [+0.030, +0.138]<br>main +0.093 | +0.186<br>d [+0.129, +0.239]<br>c [+0.095, +0.272]<br>main +0.175 | +0.239<br>d [+0.174, +0.299]<br>c [+0.145, +0.333]<br>main +0.231 | +0.022<br>d [-0.032, +0.075]<br>c [-0.051, +0.096]<br>main +0.027 |
| V4 chronological: fit to 31991, score from 32312 | D10 | 10,146 / 75 / 18 | -0.016<br>d [-0.129, +0.068]<br>c [-0.161, +0.056]<br>main +0.037 | +0.077<br>d [+0.008, +0.143]<br>c [+0.014, +0.160]<br>main +0.034 | +0.104<br>d [+0.037, +0.161]<br>c [+0.040, +0.154]<br>main +0.089 | -0.046<br>d [-0.107, +0.001]<br>c [-0.129, +0.001]<br>main -0.022 |
| V4 chronological: fit to 31991, score from 32312 | D650 | 10,146 / 75 / 18 | -0.045<br>d [-0.280, +0.128]<br>c [-0.387, +0.267]<br>main -0.066 | +0.249<br>d [+0.127, +0.355]<br>c [+0.120, +0.369]<br>main +0.245 | +0.203<br>d [+0.070, +0.311]<br>c [+0.049, +0.373]<br>main +0.237 | +0.016<br>d [-0.064, +0.083]<br>c [-0.067, +0.102]<br>main -0.054 |
| V4 chronological: fit to 31991, score from 32312 | S (schedule) | 7,180 / 50 / 14 | +0.091<br>d [-0.089, +0.245]<br>c [-0.084, +0.236]<br>main -0.100 | -0.070<br>d [-0.294, +0.093]<br>c [-0.437, +0.076]<br>main +0.102 | -0.025<br>d [-0.225, +0.144]<br>c [-0.336, +0.141]<br>main +0.028 | +0.052<br>d [-0.069, +0.170]<br>c [-0.060, +0.154]<br>main -0.016 |
| V4 chronological: fit to 32310, score from 32312 | D10 | 10,146 / 75 / 18 | +0.050<br>d [+0.016, +0.081]<br>c [-0.006, +0.100]<br>main +0.048 | +0.026<br>d [-0.018, +0.069]<br>c [-0.022, +0.079]<br>main +0.043 | +0.070<br>d [+0.027, +0.110]<br>c [+0.040, +0.125]<br>main +0.079 | +0.005<br>d [-0.023, +0.034]<br>c [-0.031, +0.032]<br>main +0.011 |
| V4 chronological: fit to 32310, score from 32312 | D650 | 10,146 / 75 / 18 | -0.112<br>d [-0.397, +0.092]<br>c [-0.507, +0.196]<br>main -0.089 | +0.322<br>d [+0.180, +0.450]<br>c [+0.160, +0.493]<br>main +0.275 | +0.224<br>d [+0.138, +0.303]<br>c [+0.127, +0.350]<br>main +0.216 | +0.029<br>d [-0.032, +0.084]<br>c [-0.035, +0.111]<br>main -0.008 |
| V4 chronological: fit to 32310, score from 32312 | S (schedule) | 7,180 / 50 / 14 | +0.076<br>d [-0.062, +0.198]<br>c [-0.061, +0.210]<br>main -0.051 | -0.219<br>d [-0.521, -0.010]<br>c [-0.607, +0.113]<br>main -0.062 | -0.161<br>d [-0.405, +0.017]<br>c [-0.504, +0.142]<br>main -0.254 | +0.030<br>d [-0.105, +0.132]<br>c [-0.104, +0.185]<br>main +0.109 |

### Skills of the refit by design and rule, target F

Cells: refit skill, d interval, c interval; then the main point.

| design | rule | rows / discharges / clusters | TN over T0 | TXN over TN | TX over T0 | TXN over TX |
|---|---|---|---|---|---|---|
| V1 random (r2-r4) | D10 | 163,470 / 553 / 76 | -0.000<br>d [-0.006, +0.005]<br>c [-0.006, +0.006]<br>main +0.001 | +0.100<br>d [+0.082, +0.118]<br>c [+0.068, +0.140]<br>main +0.099 | +0.102<br>d [+0.083, +0.121]<br>c [+0.068, +0.145]<br>main +0.102 | -0.002<br>d [-0.010, +0.008]<br>c [-0.010, +0.008]<br>main -0.003 |
| V1 random (r2-r4) | D650 | 163,470 / 553 / 76 | +0.101<br>d [+0.070, +0.130]<br>c [+0.066, +0.136]<br>main +0.105 | +0.228<br>d [+0.201, +0.255]<br>c [+0.184, +0.268]<br>main +0.226 | +0.285<br>d [+0.254, +0.314]<br>c [+0.237, +0.326]<br>main +0.287 | +0.030<br>d [+0.013, +0.048]<br>c [+0.012, +0.049]<br>main +0.029 |
| V1 random (r2-r4) | S (schedule) | 150,420 / 496 / 68 | +0.083<br>d [+0.048, +0.119]<br>c [+0.047, +0.119]<br>main +0.095 | +0.240<br>d [+0.206, +0.273]<br>c [+0.188, +0.292]<br>main +0.226 | +0.280<br>d [+0.240, +0.318]<br>c [+0.224, +0.335]<br>main +0.283 | +0.032<br>d [+0.015, +0.049]<br>c [+0.015, +0.049]<br>main +0.024 |
| V2 cluster-blocked (3 assignments) | D10 | 163,470 / 553 / 76 | -0.007<br>d [-0.013, -0.001]<br>c [-0.013, +0.000]<br>main -0.004 | +0.083<br>d [+0.068, +0.099]<br>c [+0.056, +0.116]<br>main +0.083 | +0.078<br>d [+0.063, +0.093]<br>c [+0.050, +0.113]<br>main +0.080 | -0.001<br>d [-0.008, +0.006]<br>c [-0.008, +0.006]<br>main -0.000 |
| V2 cluster-blocked (3 assignments) | D650 | 163,470 / 553 / 76 | -0.020<br>d [-0.080, +0.031]<br>c [-0.122, +0.060]<br>main -0.023 | +0.254<br>d [+0.215, +0.291]<br>c [+0.198, +0.307]<br>main +0.255 | +0.237<br>d [+0.203, +0.272]<br>c [+0.177, +0.284]<br>main +0.237 | +0.002<br>d [-0.018, +0.023]<br>c [-0.023, +0.029]<br>main +0.001 |
| V2 cluster-blocked (3 assignments) | S (schedule) | 150,420 / 496 / 68 | +0.048<br>d [+0.012, +0.083]<br>c [-0.003, +0.102]<br>main +0.042 | +0.219<br>d [+0.178, +0.261]<br>c [+0.158, +0.282]<br>main +0.218 | +0.258<br>d [+0.214, +0.300]<br>c [+0.193, +0.316]<br>main +0.260 | -0.002<br>d [-0.025, +0.022]<br>c [-0.027, +0.028]<br>main -0.012 |
| V3 rolling origin (fifths 2-5 pooled) | D10 | 116,471 / 442 / 60 | -0.004<br>d [-0.010, +0.001]<br>c [-0.012, +0.003]<br>main -0.010 | +0.063<br>d [+0.048, +0.078]<br>c [+0.036, +0.097]<br>main +0.069 | +0.056<br>d [+0.040, +0.071]<br>c [+0.030, +0.090]<br>main +0.059 | +0.003<br>d [-0.002, +0.009]<br>c [-0.007, +0.011]<br>main +0.000 |
| V3 rolling origin (fifths 2-5 pooled) | D650 | 116,471 / 442 / 60 | -0.068<br>d [-0.125, -0.017]<br>c [-0.185, +0.032]<br>main -0.074 | +0.212<br>d [+0.171, +0.252]<br>c [+0.165, +0.258]<br>main +0.219 | +0.150<br>d [+0.108, +0.190]<br>c [+0.035, +0.227]<br>main +0.156 | +0.010<br>d [-0.014, +0.034]<br>c [-0.035, +0.053]<br>main +0.005 |
| V3 rolling origin (fifths 2-5 pooled) | S (schedule) | 106,417 / 398 / 56 | +0.063<br>d [+0.026, +0.100]<br>c [+0.026, +0.101]<br>main +0.060 | +0.165<br>d [+0.127, +0.205]<br>c [+0.108, +0.227]<br>main +0.159 | +0.184<br>d [+0.136, +0.232]<br>c [+0.120, +0.248]<br>main +0.193 | +0.042<br>d [+0.013, +0.070]<br>c [+0.009, +0.073]<br>main +0.021 |
| V4 chronological: fit to 31991, score from 32312 | D10 | 10,662 / 75 / 18 | -0.010<br>d [-0.023, +0.004]<br>c [-0.022, +0.001]<br>main -0.011 | +0.014<br>d [-0.009, +0.043]<br>c [-0.012, +0.043]<br>main +0.038 | +0.021<br>d [-0.006, +0.050]<br>c [+0.005, +0.048]<br>main +0.026 | -0.016<br>d [-0.033, -0.001]<br>c [-0.036, +0.000]<br>main +0.001 |
| V4 chronological: fit to 31991, score from 32312 | D650 | 10,662 / 75 / 18 | -0.038<br>d [-0.127, +0.039]<br>c [-0.197, +0.114]<br>main -0.015 | +0.138<br>d [+0.089, +0.187]<br>c [+0.077, +0.191]<br>main +0.101 | +0.103<br>d [+0.031, +0.162]<br>c [+0.029, +0.181]<br>main +0.088 | +0.003<br>d [-0.033, +0.040]<br>c [-0.032, +0.055]<br>main +0.000 |
| V4 chronological: fit to 31991, score from 32312 | S (schedule) | 7,551 / 50 / 14 | -0.034<br>d [-0.095, +0.015]<br>c [-0.094, +0.028]<br>main +0.014 | +0.087<br>d [+0.041, +0.138]<br>c [+0.038, +0.158]<br>main -0.004 | +0.033<br>d [-0.045, +0.100]<br>c [-0.062, +0.121]<br>main +0.031 | +0.024<br>d [-0.009, +0.063]<br>c [-0.024, +0.102]<br>main -0.021 |
| V4 chronological: fit to 32310, score from 32312 | D10 | 10,662 / 75 / 18 | -0.004<br>d [-0.015, +0.008]<br>c [-0.017, +0.008]<br>main +0.003 | +0.054<br>d [+0.029, +0.079]<br>c [+0.035, +0.068]<br>main +0.038 | +0.050<br>d [+0.025, +0.074]<br>c [+0.029, +0.068]<br>main +0.053 | +0.001<br>d [-0.008, +0.010]<br>c [-0.010, +0.010]<br>main -0.013 |
| V4 chronological: fit to 32310, score from 32312 | D650 | 10,662 / 75 / 18 | -0.063<br>d [-0.163, +0.017]<br>c [-0.231, +0.085]<br>main -0.049 | +0.152<br>d [+0.089, +0.219]<br>c [+0.072, +0.233]<br>main +0.131 | +0.089<br>d [+0.047, +0.136]<br>c [+0.051, +0.163]<br>main +0.100 | +0.010<br>d [-0.024, +0.038]<br>c [-0.028, +0.045]<br>main -0.014 |
| V4 chronological: fit to 32310, score from 32312 | S (schedule) | 7,551 / 50 / 14 | -0.017<br>d [-0.059, +0.023]<br>c [-0.059, +0.040]<br>main -0.022 | -0.004<br>d [-0.069, +0.063]<br>c [-0.130, +0.093]<br>main -0.017 | -0.017<br>d [-0.073, +0.040]<br>c [-0.058, +0.042]<br>main +0.041 | -0.004<br>d [-0.033, +0.026]<br>c [-0.056, +0.030]<br>main -0.084 |

### Rows and discharges removed

| set | main rows (discharges) | refit rows (discharges) | rows removed | discharges removed |
|---|---|---|---|---|
| common, F | 165,418 (564) | 163,470 (553) | 1,948 | 11 |
| S, F | 152,203 (506) | 150,420 (496) | 1,783 | 10 |
| common, L5 | 160,343 (562) | 158,560 (551) | 1,783 | 11 |
| S, L5 | 147,837 (503) | 146,197 (493) | 1,640 | 10 |

Per unit (target L5): fit and test discharges with a row in the set, and rows, removed from the main run's count (main count in brackets).

| unit | common set: fit discharges / rows | common set: test discharges / rows | S set: fit discharges / rows | S set: test discharges / rows |
|---|---|---|---|---|
| V1_r2_k0 | 7 / 1,380 (455 / 133,674) | 4 / 403 (107 / 26,669) | 7 / 1,263 (411 / 123,415) | 3 / 377 (92 / 24,422) |
| V1_r2_k1 | 9 / 1,471 (445 / 127,536) | 2 / 312 (117 / 32,807) | 8 / 1,343 (397 / 118,198) | 2 / 297 (106 / 29,639) |
| V1_r2_k2 | 9 / 1,394 (454 / 126,937) | 2 / 389 (108 / 33,406) | 8 / 1,273 (403 / 116,688) | 2 / 367 (100 / 31,149) |
| V1_r2_k3 | 11 / 1,783 (450 / 130,415) | 0 / 0 (112 / 29,928) | 10 / 1,640 (405 / 120,138) | 0 / 0 (98 / 27,699) |
| V1_r2_k4 | 8 / 1,104 (444 / 122,810) | 3 / 679 (118 / 37,533) | 7 / 1,041 (396 / 112,909) | 3 / 599 (107 / 34,928) |
| V1_r3_k0 | 8 / 1,492 (452 / 131,715) | 3 / 291 (110 / 28,628) | 7 / 1,354 (408 / 121,032) | 3 / 286 (95 / 26,805) |
| V1_r3_k1 | 10 / 1,771 (451 / 127,931) | 1 / 12 (111 / 32,412) | 10 / 1,640 (402 / 117,974) | 0 / 0 (101 / 29,863) |
| V1_r3_k2 | 9 / 1,490 (449 / 122,684) | 2 / 293 (113 / 37,659) | 8 / 1,380 (400 / 113,073) | 2 / 260 (103 / 34,764) |
| V1_r3_k3 | 8 / 1,068 (448 / 129,711) | 3 / 715 (114 / 30,632) | 7 / 959 (403 / 120,280) | 3 / 681 (100 / 27,557) |
| V1_r3_k4 | 9 / 1,311 (448 / 129,331) | 2 / 472 (114 / 31,012) | 8 / 1,227 (399 / 118,989) | 2 / 413 (104 / 28,848) |
| V1_r4_k0 | 8 / 1,201 (449 / 129,342) | 3 / 582 (113 / 31,001) | 7 / 1,120 (399 / 119,540) | 3 / 520 (104 / 28,297) |
| V1_r4_k1 | 8 / 1,543 (452 / 129,265) | 3 / 240 (110 / 31,078) | 8 / 1,412 (405 / 119,347) | 2 / 228 (98 / 28,490) |
| V1_r4_k2 | 9 / 1,394 (449 / 125,692) | 2 / 389 (113 / 34,651) | 8 / 1,273 (401 / 115,936) | 2 / 367 (102 / 31,901) |
| V1_r4_k3 | 9 / 1,479 (446 / 124,325) | 2 / 304 (116 / 36,018) | 8 / 1,352 (403 / 114,249) | 2 / 288 (100 / 33,588) |
| V1_r4_k4 | 10 / 1,515 (452 / 132,748) | 1 / 268 (110 / 27,595) | 9 / 1,403 (404 / 122,276) | 1 / 237 (99 / 25,561) |
| V2_s1_k0 | 11 / 1,783 (447 / 121,578) | 0 / 0 (115 / 38,765) | 10 / 1,640 (401 / 113,102) | 0 / 0 (102 / 34,735) |
| V2_s1_k1 | 3 / 449 (456 / 133,356) | 8 / 1,334 (106 / 26,987) | 3 / 395 (410 / 121,816) | 7 / 1,245 (93 / 26,021) |
| V2_s1_k2 | 11 / 1,783 (439 / 116,598) | 0 / 0 (123 / 43,745) | 10 / 1,640 (387 / 106,479) | 0 / 0 (116 / 41,358) |
| V2_s1_k3 | 11 / 1,783 (464 / 138,804) | 0 / 0 (98 / 21,539) | 10 / 1,640 (422 / 128,163) | 0 / 0 (81 / 19,674) |
| V2_s1_k4 | 8 / 1,334 (442 / 131,036) | 3 / 449 (120 / 29,307) | 7 / 1,245 (392 / 121,788) | 3 / 395 (111 / 26,049) |
| V2_s2_k0 | 10 / 1,593 (455 / 131,849) | 1 / 190 (107 / 28,494) | 9 / 1,445 (413 / 121,473) | 1 / 195 (90 / 26,364) |
| V2_s2_k1 | 8 / 1,334 (448 / 127,007) | 3 / 449 (114 / 33,336) | 7 / 1,245 (401 / 117,217) | 3 / 395 (102 / 30,620) |
| V2_s2_k2 | 11 / 1,783 (459 / 128,588) | 0 / 0 (103 / 31,755) | 10 / 1,640 (413 / 117,592) | 0 / 0 (90 / 30,245) |
| V2_s2_k3 | 8 / 1,404 (454 / 128,355) | 3 / 379 (108 / 31,988) | 8 / 1,263 (406 / 119,602) | 2 / 377 (97 / 28,235) |
| V2_s2_k4 | 7 / 1,018 (432 / 125,573) | 4 / 765 (130 / 34,770) | 6 / 967 (379 / 115,464) | 4 / 673 (124 / 32,373) |
| V2_s3_k0 | 7 / 1,018 (449 / 135,902) | 4 / 765 (113 / 24,441) | 6 / 967 (404 / 125,516) | 4 / 673 (99 / 22,321) |
| V2_s3_k1 | 7 / 1,144 (453 / 137,708) | 4 / 639 (109 / 22,635) | 6 / 1,050 (417 / 127,137) | 4 / 590 (86 / 20,700) |
| V2_s3_k2 | 11 / 1,783 (450 / 133,716) | 0 / 0 (112 / 26,627) | 10 / 1,640 (397 / 123,448) | 0 / 0 (106 / 24,389) |
| V2_s3_k3 | 11 / 1,783 (446 / 120,606) | 0 / 0 (116 / 39,737) | 10 / 1,640 (395 / 110,053) | 0 / 0 (108 / 37,784) |
| V2_s3_k4 | 8 / 1,404 (450 / 113,440) | 3 / 379 (112 / 46,903) | 8 / 1,263 (399 / 105,194) | 2 / 377 (104 / 42,643) |
| V3_q1 | 11 / 1,783 (121 / 48,144) | 0 / 0 (108 / 27,883) | 10 / 1,640 (108 / 45,153) | 0 / 0 (105 / 26,484) |
| V3_q2 | 11 / 1,783 (229 / 76,027) | 0 / 0 (118 / 43,406) | 10 / 1,640 (213 / 71,637) | 0 / 0 (113 / 40,944) |
| V3_q3 | 11 / 1,783 (347 / 119,433) | 0 / 0 (108 / 23,933) | 10 / 1,640 (326 / 112,581) | 0 / 0 (97 / 22,751) |
| V3_q4 | 11 / 1,783 (455 / 143,366) | 0 / 0 (107 / 16,977) | 10 / 1,640 (423 / 135,332) | 0 / 0 (80 / 12,505) |
| V4a | 11 / 1,783 (375 / 127,270) | 0 / 0 (75 / 10,146) | 10 / 1,640 (349 / 119,655) | 0 / 0 (50 / 7,180) |
| V4b | 11 / 1,783 (487 / 150,197) | 0 / 0 (75 / 10,146) | 10 / 1,640 (453 / 140,657) | 0 / 0 (50 / 7,180) |

The refit's full readout (RMSE, every skill, the ratio, the halves, the strata, the controls) is under the key `refit_without_13_files.readout` of the JSON.
