# B.77 — the frame's own reading, corrected and flagged with the fast samples of the same frame

Registered: `ces_prediction/experiments/PREREGISTRATION_B77.md`; implementation choices in `ces_prediction/experiments/PREREGISTRATION_B77_ADDENDUM1.md`, written before any fit and any readout. CPU only; no network trained or run. T_i, cut population; rows: the D1 rows of B.73 complete under rule D10 of B.75 with the following level L5 defined (209,432 of 222,672 D1 rows; 606 discharges; 85 shot clusters). Exploratory on used data (registration section 1): rows, target, designs and fast inputs are those of B.73 and B.75; the estimators that read the frame's own value are new.

Estimators (all scored against L5, the median of the measured values among rows t+1 to t+5): O the reported value of frame t; MED3 the median of it and the two newest earlier measured values; G the causal GP and H the hold under D10 (beside, no statement uses them); T0, TX B.75's estimators under D10 refitted on these rows (history; history plus fourteen fast scalars); C0 = G + boosted regression on the ten history scalars and four scalars of the frame's value; CX C0's inputs plus the fourteen fast scalars; CB, CE C0's inputs plus the BES half or the ECEI half.

Conventions. Skill of A over B = 1 - SSE(A) / SSE(B) on identical rows; MSE reduction in % = 100 x skill. Each comparison: point, then 95 % intervals with discharges resampled (d) and with shot clusters resampled (c), 20,000 draws (V1: a re-partition drawn first; V2: an assignment drawn first; V3, V4: the scored units). Verdicts on the cluster form. Points under V1 and V2 = mean over the three re-partitions or assignments. RMSE in eV, MSE in eV^2. Checks: `docs/paper/b77_checks.md`.

Order of work. Before this readout, these results existed: the checks of registration section 9 (and the added ones of Addendum 1 section 11) and the row counts of Addendum 1 section 1. No MSE, skill, AUC, screening value or statement of this batch had been computed.

## Statements (cluster interval)

| Id | statement | value: point; d interval; c interval | verdict | registered expectation | expectation met | note |
|---|---|---|---|---|---|---|
| A1 | V1: the MSE of C0 lies below that of O (skill above zero) | +0.572; d [+0.545, +0.598]; c [+0.518, +0.621] | **holds** | holds (+0.5 to +0.7) | yes; point in the stated range: yes |  |
| A2 | V1: the MSE of C0 lies below that of T0 (reading the frame helps) | +0.147; d [+0.137, +0.157]; c [+0.132, +0.164] | **holds** | holds (+0.05 to +0.30) | yes; point in the stated range: yes |  |
| A3 | V1: CX over C0 lies above zero | +0.048; d [+0.029, +0.063]; c [+0.023, +0.072] | **holds** | holds (+0.03 to +0.15) | yes; point in the stated range: yes |  |
| A4 | V2: CX over C0 lies above zero | +0.013; d [-0.002, +0.027]; c [-0.012, +0.037] | **does not hold** | holds | no |  |
| A5 | V3: CX over C0 lies above zero | -0.009; d [-0.027, +0.009]; c [-0.049, +0.023] | **does not hold** | uncertain | n/a (uncertain) |  |
| A6 | V4 (fit to 31991): CX over C0 lies above zero | +0.010; d [-0.014, +0.033]; c [-0.009, +0.032] | **does not hold** | uncertain | n/a (uncertain) | V4b (fit to 32310): +0.043; c [+0.033, +0.062] |
| A7 | V1: CX over C0 exceeds TX over T0 (interval of the difference above zero) | -0.038; d [-0.052, -0.018]; c [-0.054, -0.018] | **does not hold** | uncertain, leaning to holds | n/a (uncertain) | CX/C0 +0.048; c [+0.023, +0.072]; TX/T0 +0.085; c [+0.046, +0.120] |
| A8 | V1 and V4: the area under the curve of FHX minus that of FH lies above zero under each | V1: +0.005; d [+0.004, +0.006]; c [+0.003, +0.008]<br>V4a: -0.000; d [-0.005, +0.004]; c [-0.007, +0.007] | **does not hold** | holds | no | V4b: +0.001; c [-0.004, +0.008] |
| A9 | V1, q = 10 %: the MSE of O on the rows kept by FHX lies below that on the rows kept by FH (interval of the ratio below 1) | 0.989; d [0.978, 0.999]; c [0.975, 1.002] | **does not hold** | uncertain | n/a (uncertain) | MSE of O (eV^2): kept by FHX 14513, by FH 14680, no screening 33642 |
| A10 | V1: CE over C0 lies above zero | +0.014; d [+0.002, +0.025]; c [+0.003, +0.025] | **holds** | uncertain | n/a (uncertain) |  |
| A11 | V1, placebo: the interval of CX over C0 contains zero or lies below it | -0.004; d [-0.008, +0.000]; c [-0.008, +0.000] | **holds** | holds | yes |  |

A1, A2, A10 and A3-A6 are MSE skills; A7 is a difference of skills; A8 differences of areas under the ROC curve; A9 a ratio of MSEs (judged on its upper end below 1); A11 holds when the lower end is at most zero. 'V4' without a qualifier is V4a; V4b is in the note.

## RMSE against L5 by design (eV)

| design | rows | O | MED3 | G | H | T0 | TX | C0 | CX | CB | CE |
|---|---|---|---|---|---|---|---|---|---|---|---|
| V1 random (r2-r4) | 209,432 | 183.4 | 145.1 | 143.1 | 193.3 | 129.9 | 124.2 | 120.0 | 117.1 | 117.5 | 119.1 |
| V2 cluster-blocked (3 assignments) | 209,432 | 183.4 | 145.1 | 143.1 | 193.3 | 131.4 | 127.6 | 121.1 | 120.2 | 119.6 | 121.6 |
| V3 rolling origin (fifths 2-5 pooled) | 150,614 | 203.3 | 158.6 | 157.2 | 214.3 | 146.1 | 143.9 | 134.5 | 135.1 | 133.2 | 136.3 |
| V4 chronological: fit to 31991, score from 32312 | 18,277 | 234.7 | 155.7 | 158.8 | 243.4 | 133.9 | 128.2 | 120.7 | 120.1 | 121.7 | 119.3 |
| V4 chronological: fit to 32310, score from 32312 | 18,277 | 234.7 | 155.7 | 158.8 | 243.4 | 126.6 | 123.6 | 117.1 | 114.5 | 115.7 | 116.3 |

## Skills by design

| design | C0/O | C0/T0 | TX/T0 | CX/C0 | CB/C0 | CE/C0 | CX/C0 minus TX/T0 |
|---|---|---|---|---|---|---|---|
| V1 random (r2-r4) | +0.572<br>d [+0.545, +0.598]<br>c [+0.518, +0.621] | +0.147<br>d [+0.137, +0.157]<br>c [+0.132, +0.164] | +0.085<br>d [+0.052, +0.107]<br>c [+0.046, +0.120] | +0.048<br>d [+0.029, +0.063]<br>c [+0.023, +0.072] | +0.041<br>d [+0.027, +0.054]<br>c [+0.021, +0.063] | +0.014<br>d [+0.002, +0.025]<br>c [+0.003, +0.025] | -0.038<br>d [-0.052, -0.018]<br>c [-0.054, -0.018] |
| V2 cluster-blocked (3 assignments) | +0.564<br>d [+0.537, +0.591]<br>c [+0.507, +0.616] | +0.151<br>d [+0.140, +0.161]<br>c [+0.134, +0.167] | +0.056<br>d [+0.037, +0.074]<br>c [+0.026, +0.085] | +0.013<br>d [-0.002, +0.027]<br>c [-0.012, +0.037] | +0.024<br>d [+0.011, +0.036]<br>c [+0.004, +0.045] | -0.009<br>d [-0.019, +0.001]<br>c [-0.024, +0.006] | -0.043<br>d [-0.052, -0.033]<br>c [-0.056, -0.029] |
| V3 rolling origin (fifths 2-5 pooled) | +0.562<br>d [+0.532, +0.591]<br>c [+0.497, +0.621] | +0.153<br>d [+0.140, +0.166]<br>c [+0.132, +0.175] | +0.030<br>d [+0.010, +0.049]<br>c [-0.007, +0.063] | -0.009<br>d [-0.027, +0.009]<br>c [-0.049, +0.023] | +0.019<br>d [+0.004, +0.034]<br>c [-0.009, +0.045] | -0.026<br>d [-0.039, -0.015]<br>c [-0.047, -0.006] | -0.038<br>d [-0.050, -0.026]<br>c [-0.054, -0.021] |
| V4 chronological: fit to 31991, score from 32312 | +0.735<br>d [+0.690, +0.779]<br>c [+0.692, +0.791] | +0.187<br>d [+0.151, +0.224]<br>c [+0.137, +0.233] | +0.083<br>d [+0.032, +0.124]<br>c [+0.032, +0.126] | +0.010<br>d [-0.014, +0.033]<br>c [-0.009, +0.032] | -0.017<br>d [-0.050, +0.017]<br>c [-0.043, +0.013] | +0.024<br>d [+0.005, +0.042]<br>c [+0.002, +0.046] | -0.073<br>d [-0.121, -0.024]<br>c [-0.118, -0.021] |
| V4 chronological: fit to 32310, score from 32312 | +0.751<br>d [+0.708, +0.793]<br>c [+0.710, +0.802] | +0.145<br>d [+0.118, +0.175]<br>c [+0.112, +0.181] | +0.047<br>d [+0.020, +0.075]<br>c [+0.019, +0.078] | +0.043<br>d [+0.028, +0.059]<br>c [+0.033, +0.062] | +0.024<br>d [+0.005, +0.043]<br>c [+0.008, +0.042] | +0.013<br>d [-0.009, +0.035]<br>c [-0.004, +0.038] | -0.005<br>d [-0.028, +0.020]<br>c [-0.029, +0.023] |

## Skill over the frame's own reading O

| design | MED3/O | G/O | H/O | T0/O | TX/O | C0/O | CX/O | CB/O | CE/O |
|---|---|---|---|---|---|---|---|---|---|
| V1 random (r2-r4) | +0.375<br>d [+0.343, +0.405]<br>c [+0.309, +0.432] | +0.392<br>d [+0.357, +0.424]<br>c [+0.317, +0.453] | -0.111<br>d [-0.130, -0.092]<br>c [-0.143, -0.082] | +0.498<br>d [+0.464, +0.530]<br>c [+0.427, +0.560] | +0.541<br>d [+0.508, +0.570]<br>c [+0.484, +0.593] | +0.572<br>d [+0.545, +0.598]<br>c [+0.518, +0.621] | +0.593<br>d [+0.567, +0.616]<br>c [+0.545, +0.636] | +0.590<br>d [+0.565, +0.613]<br>c [+0.542, +0.634] | +0.578<br>d [+0.552, +0.603]<br>c [+0.525, +0.626] |
| V2 cluster-blocked (3 assignments) | +0.375<br>d [+0.343, +0.405]<br>c [+0.308, +0.432] | +0.392<br>d [+0.358, +0.423]<br>c [+0.315, +0.454] | -0.111<br>d [-0.130, -0.092]<br>c [-0.144, -0.082] | +0.487<br>d [+0.452, +0.520]<br>c [+0.411, +0.553] | +0.516<br>d [+0.483, +0.547]<br>c [+0.449, +0.575] | +0.564<br>d [+0.537, +0.591]<br>c [+0.507, +0.616] | +0.570<br>d [+0.543, +0.596]<br>c [+0.513, +0.621] | +0.575<br>d [+0.548, +0.600]<br>c [+0.520, +0.624] | +0.561<br>d [+0.532, +0.587]<br>c [+0.501, +0.614] |
| V3 rolling origin (fifths 2-5 pooled) | +0.391<br>d [+0.357, +0.424]<br>c [+0.321, +0.453] | +0.402<br>d [+0.365, +0.436]<br>c [+0.323, +0.466] | -0.112<br>d [-0.132, -0.092]<br>c [-0.147, -0.081] | +0.483<br>d [+0.444, +0.520]<br>c [+0.397, +0.558] | +0.499<br>d [+0.463, +0.532]<br>c [+0.418, +0.570] | +0.562<br>d [+0.532, +0.591]<br>c [+0.497, +0.621] | +0.558<br>d [+0.530, +0.586]<br>c [+0.496, +0.618] | +0.570<br>d [+0.543, +0.598]<br>c [+0.511, +0.626] | +0.551<br>d [+0.519, +0.581]<br>c [+0.477, +0.616] |
| V4 chronological: fit to 31991, score from 32312 | +0.560<br>d [+0.493, +0.630]<br>c [+0.481, +0.659] | +0.542<br>d [+0.486, +0.596]<br>c [+0.472, +0.611] | -0.076<br>d [-0.109, -0.044]<br>c [-0.123, -0.037] | +0.675<br>d [+0.624, +0.724]<br>c [+0.623, +0.738] | +0.702<br>d [+0.650, +0.751]<br>c [+0.653, +0.764] | +0.735<br>d [+0.690, +0.779]<br>c [+0.692, +0.791] | +0.738<br>d [+0.694, +0.781]<br>c [+0.696, +0.793] | +0.731<br>d [+0.688, +0.773]<br>c [+0.690, +0.786] | +0.742<br>d [+0.698, +0.784]<br>c [+0.699, +0.796] |
| V4 chronological: fit to 32310, score from 32312 | +0.560<br>d [+0.493, +0.630]<br>c [+0.480, +0.658] | +0.542<br>d [+0.486, +0.597]<br>c [+0.472, +0.611] | -0.076<br>d [-0.108, -0.045]<br>c [-0.123, -0.038] | +0.709<br>d [+0.660, +0.756]<br>c [+0.663, +0.766] | +0.723<br>d [+0.675, +0.768]<br>c [+0.677, +0.781] | +0.751<br>d [+0.708, +0.793]<br>c [+0.710, +0.802] | +0.762<br>d [+0.719, +0.803]<br>c [+0.721, +0.812] | +0.757<br>d [+0.713, +0.799]<br>c [+0.715, +0.808] | +0.755<br>d [+0.711, +0.797]<br>c [+0.713, +0.806] |

## V1 by stratum on an input (each estimator keeps its fit)

Chronological block of the test discharge; class of r_B, the BES sample relative to its recent median (B.73 Addendum 1). The class 'r_B undefined' is reported beside.

| stratum | rows | discharges | clusters | RMSE O / C0 / CX (eV) | CX/C0 | CB/C0 | CE/C0 | TX/T0 | C0/T0 | C0/O | CX/C0 minus TX/T0 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| block up to 31991 | 160,401 | 402 | 53 | 159.8 / 114.7 / 111.2 | +0.060<br>d [+0.040, +0.077]<br>c [+0.030, +0.090] | +0.054<br>d [+0.039, +0.069]<br>c [+0.030, +0.081] | +0.015<br>d [-0.001, +0.028]<br>c [-0.001, +0.028] | +0.105<br>d [+0.081, +0.128]<br>c [+0.070, +0.144] | +0.155<br>d [+0.143, +0.167]<br>c [+0.134, +0.176] | +0.485<br>d [+0.453, +0.513]<br>c [+0.415, +0.544] | -0.045<br>d [-0.059, -0.032]<br>c [-0.062, -0.029] |
| block 32002-32310 | 30,754 | 121 | 14 | 251.4 / 146.7 / 145.7 | +0.013<br>d [-0.032, +0.043]<br>c [-0.026, +0.047] | +0.007<br>d [-0.037, +0.033]<br>c [-0.024, +0.042] | +0.009<br>d [-0.018, +0.033]<br>c [-0.019, +0.029] | +0.028<br>d [-0.091, +0.076]<br>c [-0.065, +0.068] | +0.124<br>d [+0.100, +0.146]<br>c [+0.103, +0.149] | +0.660<br>d [+0.609, +0.705]<br>c [+0.588, +0.709] | -0.015<br>d [-0.049, +0.062]<br>c [-0.051, +0.044] |
| block from 32312 | 18,277 | 83 | 19 | 234.7 / 115.3 / 113.2 | +0.035<br>d [+0.010, +0.061]<br>c [+0.013, +0.062] | +0.023<br>d [-0.005, +0.047]<br>c [+0.001, +0.051] | +0.025<br>d [+0.007, +0.048]<br>c [+0.008, +0.049] | +0.062<br>d [+0.034, +0.090]<br>c [+0.036, +0.090] | +0.141<br>d [+0.114, +0.168]<br>c [+0.104, +0.181] | +0.759<br>d [+0.714, +0.801]<br>c [+0.715, +0.809] | -0.027<br>d [-0.050, -0.003]<br>c [-0.049, -0.000] |
| rB below 0.9 | 9,387 | 526 | 77 | 310.5 / 208.4 / 197.7 | +0.100<br>d [+0.059, +0.142]<br>c [+0.047, +0.164] | +0.094<br>d [+0.053, +0.134]<br>c [+0.044, +0.155] | +0.029<br>d [+0.010, +0.050]<br>c [+0.007, +0.049] | +0.190<br>d [+0.137, +0.240]<br>c [+0.123, +0.269] | +0.211<br>d [+0.183, +0.239]<br>c [+0.184, +0.240] | +0.549<br>d [+0.478, +0.615]<br>c [+0.425, +0.656] | -0.090<br>d [-0.118, -0.060]<br>c [-0.118, -0.061] |
| rB 0.9-0.97 | 37,890 | 597 | 85 | 167.5 / 116.6 / 113.7 | +0.049<br>d [+0.030, +0.068]<br>c [+0.026, +0.072] | +0.038<br>d [+0.020, +0.057]<br>c [+0.017, +0.060] | +0.022<br>d [+0.005, +0.036]<br>c [+0.004, +0.036] | +0.076<br>d [+0.050, +0.098]<br>c [+0.049, +0.103] | +0.133<br>d [+0.111, +0.155]<br>c [+0.108, +0.159] | +0.516<br>d [+0.474, +0.555]<br>c [+0.450, +0.573] | -0.027<br>d [-0.046, -0.006]<br>c [-0.047, -0.006] |
| rB 0.97-1.03 | 94,373 | 601 | 85 | 142.6 / 91.9 / 90.8 | +0.024<br>d [+0.005, +0.040]<br>c [-0.004, +0.047] | +0.018<br>d [+0.002, +0.031]<br>c [-0.005, +0.039] | +0.008<br>d [-0.007, +0.019]<br>c [-0.008, +0.019] | +0.035<br>d [+0.001, +0.054]<br>c [-0.003, +0.062] | +0.095<br>d [+0.081, +0.110]<br>c [+0.080, +0.113] | +0.584<br>d [+0.551, +0.615]<br>c [+0.532, +0.632] | -0.011<br>d [-0.029, +0.011]<br>c [-0.033, +0.012] |
| rB above 1.03 | 54,319 | 601 | 84 | 215.3 / 136.1 / 133.7 | +0.035<br>d [+0.010, +0.055]<br>c [+0.006, +0.060] | +0.029<br>d [+0.011, +0.045]<br>c [+0.008, +0.050] | +0.011<br>d [-0.008, +0.028]<br>c [-0.008, +0.028] | +0.076<br>d [+0.044, +0.103]<br>c [+0.037, +0.109] | +0.154<br>d [+0.136, +0.171]<br>c [+0.127, +0.180] | +0.600<br>d [+0.564, +0.635]<br>c [+0.536, +0.659] | -0.041<br>d [-0.060, -0.022]<br>c [-0.060, -0.019] |
| rB r_B undefined | 13,463 | 495 | 72 | 218.2 / 146.3 / 140.2 | +0.081<br>d [+0.014, +0.126]<br>c [+0.004, +0.143] | +0.079<br>d [+0.026, +0.121]<br>c [+0.018, +0.140] | +0.012<br>d [-0.019, +0.043]<br>c [-0.019, +0.040] | +0.105<br>d [-0.040, +0.175]<br>c [-0.039, +0.193] | +0.183<br>d [+0.153, +0.214]<br>c [+0.129, +0.242] | +0.550<br>d [+0.485, +0.607]<br>c [+0.476, +0.624] | -0.024<br>d [-0.070, +0.062]<br>c [-0.078, +0.059] |

## Controls (V1)

| control | rows | RMSE (eV) | quantity | value |
|---|---|---|---|---|
| placebo CXp: the fourteen fast scalars permuted over rows (within fit rows and within test rows), CX refitted | 209,432 | CXp 120.2, C0 120.0, CX 117.1 | CXp over C0 | -0.004<br>d [-0.008, +0.000]<br>c [-0.008, +0.000] |
| | | | CX over CXp | +0.052<br>d [+0.031, +0.067]<br>c [+0.025, +0.077] |
| row t - 1: CXlag, the fast scalars of row t - 1 in place of those of row t | 208,866 (566 rows of the set without them) | CXlag 117.8, C0 120.0, CX 117.1 | CXlag over C0 | +0.035<br>d [+0.020, +0.048]<br>c [+0.014, +0.055] |
| | | | CX over C0 (same rows) | +0.048<br>d [+0.029, +0.063]<br>c [+0.022, +0.072] |
| | | | CXlag over CX | -0.013<br>d [-0.021, -0.005]<br>c [-0.023, -0.005] |
| | | | (CX over C0) - (CXlag over C0) | +0.012<br>d [+0.004, +0.020]<br>c [+0.005, +0.021] |

## Flags: area under the ROC curve

Label: y / L5 above 1.35 or below 0.74. Positives on the row set: 30,552 of 209,432 (14.6 %; above 1.35: 15,350; below 0.74: 15,202). PH: history; PHX: history + fourteen fast scalars; FH: history + four scalars of the frame's value; FHX: FH + fourteen fast scalars. V1, V2: AUC over all rows of a re-partition or assignment (cross-fitted), mean over the three.

| design | rows (positives) | PH | PHX | FH | FHX | PHX - PH | FHX - FH | FH - PH | FHX - PHX |
|---|---|---|---|---|---|---|---|---|---|
| V1 random (r2-r4) | 209,432 (30,552) | 0.8707<br>d [0.8605, 0.8798]<br>c [0.8436, 0.8884] | 0.8864<br>d [0.8773, 0.8944]<br>c [0.8626, 0.9021] | 0.9171<br>d [0.9107, 0.9228]<br>c [0.9035, 0.9260] | 0.9223<br>d [0.9163, 0.9276]<br>c [0.9101, 0.9302] | +0.016<br>d [+0.013, +0.018]<br>c [+0.012, +0.021] | +0.005<br>d [+0.004, +0.006]<br>c [+0.003, +0.008] | +0.046<br>d [+0.041, +0.052]<br>c [+0.035, +0.063] | +0.036<br>d [+0.031, +0.041]<br>c [+0.026, +0.050] |
| V2 cluster-blocked (3 assignments) | 209,432 (30,552) | 0.8660<br>d [0.8551, 0.8757]<br>c [0.8387, 0.8845] | 0.8769<br>d [0.8671, 0.8856]<br>c [0.8526, 0.8933] | 0.9141<br>d [0.9071, 0.9202]<br>c [0.8995, 0.9239] | 0.9163<br>d [0.9097, 0.9221]<br>c [0.9031, 0.9252] | +0.011<br>d [+0.008, +0.014]<br>c [+0.007, +0.015] | +0.002<br>d [+0.001, +0.004]<br>c [+0.000, +0.005] | +0.048<br>d [+0.043, +0.054]<br>c [+0.037, +0.065] | +0.039<br>d [+0.034, +0.045]<br>c [+0.029, +0.055] |
| V3 rolling origin (fifths 2-5 pooled) | 150,614 (24,386) | 0.8460<br>d [0.8332, 0.8576]<br>c [0.8162, 0.8689] | 0.8521<br>d [0.8398, 0.8633]<br>c [0.8249, 0.8736] | 0.9044<br>d [0.8965, 0.9116]<br>c [0.8870, 0.9175] | 0.9056<br>d [0.8978, 0.9128]<br>c [0.8895, 0.9184] | +0.006<br>d [+0.004, +0.009]<br>c [+0.002, +0.011] | +0.001<br>d [-0.000, +0.003]<br>c [-0.001, +0.004] | +0.058<br>d [+0.052, +0.066]<br>c [+0.046, +0.075] | +0.053<br>d [+0.047, +0.061]<br>c [+0.042, +0.069] |
| V4 chronological: fit to 31991, score from 32312 | 18,277 (3,248) | 0.7685<br>d [0.7390, 0.7953]<br>c [0.7303, 0.7929] | 0.7693<br>d [0.7385, 0.7966]<br>c [0.7374, 0.7920] | 0.8763<br>d [0.8556, 0.8953]<br>c [0.8360, 0.9072] | 0.8759<br>d [0.8553, 0.8948]<br>c [0.8375, 0.9065] | +0.001<br>d [-0.008, +0.010]<br>c [-0.009, +0.015] | -0.000<br>d [-0.005, +0.004]<br>c [-0.007, +0.007] | +0.108<br>d [+0.087, +0.132]<br>c [+0.080, +0.135] | +0.107<br>d [+0.085, +0.131]<br>c [+0.077, +0.134] |
| V4 chronological: fit to 32310, score from 32312 | 18,277 (3,248) | 0.7754<br>d [0.7458, 0.8017]<br>c [0.7381, 0.7977] | 0.7840<br>d [0.7537, 0.8101]<br>c [0.7547, 0.8028] | 0.8865<br>d [0.8674, 0.9038]<br>c [0.8505, 0.9133] | 0.8877<br>d [0.8688, 0.9047]<br>c [0.8539, 0.9136] | +0.009<br>d [-0.002, +0.019]<br>c [-0.004, +0.025] | +0.001<br>d [-0.003, +0.005]<br>c [-0.004, +0.008] | +0.111<br>d [+0.091, +0.135]<br>c [+0.087, +0.136] | +0.104<br>d [+0.082, +0.129]<br>c [+0.075, +0.134] |

## Screening: MSE of O against L5 on the kept rows (eV^2)

In each test unit the rows ranked by the classifier's probability; the q highest dropped (Addendum 1 section 6). Each cell: MSE, then MSE over no screening.

### V1 random (r2-r4)

No screening: 33642<br>d [30133, 37419]<br>c [24977, 44294]

| classifier | q = 5 % | q = 10 % | q = 20 % |
|---|---|---|---|
| FHX | 17623<br>d [15845, 19528]<br>c [13290, 23026]<br>ratio 0.524<br>d [0.490, 0.558]<br>c [0.455, 0.596] | 14513<br>d [13053, 16053]<br>c [10991, 18962]<br>ratio 0.431<br>d [0.400, 0.464]<br>c [0.369, 0.502] | 10758<br>d [9642, 11943]<br>c [8212, 14112]<br>ratio 0.320<br>d [0.291, 0.350]<br>c [0.265, 0.384] |
| FH | 17757<br>d [15965, 19683]<br>c [13402, 23179]<br>ratio 0.528<br>d [0.495, 0.562]<br>c [0.459, 0.601] | 14680<br>d [13203, 16248]<br>c [11123, 19164]<br>ratio 0.436<br>d [0.405, 0.469]<br>c [0.374, 0.506] | 11026<br>d [9881, 12248]<br>c [8363, 14518]<br>ratio 0.328<br>d [0.299, 0.358]<br>c [0.271, 0.394] |
| PHX | 27782<br>d [24900, 30917]<br>c [20743, 36685]<br>ratio 0.826<br>d [0.797, 0.853]<br>c [0.778, 0.874] | 23753<br>d [21235, 26477]<br>c [17609, 31700]<br>ratio 0.706<br>d [0.667, 0.744]<br>c [0.639, 0.774] | 17207<br>d [15356, 19175]<br>c [12801, 22984]<br>ratio 0.511<br>d [0.469, 0.556]<br>c [0.449, 0.584] |
| PH | 29527<br>d [26444, 32892]<br>c [21887, 39139]<br>ratio 0.878<br>d [0.855, 0.899]<br>c [0.840, 0.915] | 24816<br>d [22176, 27714]<br>c [18308, 33291]<br>ratio 0.738<br>d [0.700, 0.774]<br>c [0.673, 0.806] | 18068<br>d [16147, 20122]<br>c [13316, 24354]<br>ratio 0.537<br>d [0.495, 0.580]<br>c [0.469, 0.611] |

| ratio of MSEs on kept rows | q = 5 % | q = 10 % | q = 20 % |
|---|---|---|---|
| FHX over FH | 0.992<br>d [0.980, 1.005]<br>c [0.977, 1.007] | 0.989<br>d [0.978, 0.999]<br>c [0.975, 1.002] | 0.976<br>d [0.951, 0.996]<br>c [0.954, 0.995] |
| PHX over PH | 0.941<br>d [0.919, 0.962]<br>c [0.911, 0.969] | 0.957<br>d [0.931, 0.984]<br>c [0.921, 0.996] | 0.952<br>d [0.919, 0.986]<br>c [0.900, 1.001] |
| FH over PH | 0.601<br>d [0.567, 0.637]<br>c [0.528, 0.673] | 0.592<br>d [0.559, 0.625]<br>c [0.524, 0.660] | 0.610<br>d [0.573, 0.648]<br>c [0.537, 0.682] |
| FHX over PHX | 0.634<br>d [0.599, 0.670]<br>c [0.559, 0.703] | 0.611<br>d [0.576, 0.647]<br>c [0.535, 0.684] | 0.625<br>d [0.580, 0.670]<br>c [0.541, 0.706] |

### V2 cluster-blocked (3 assignments)

No screening: 33642<br>d [30151, 37410]<br>c [24980, 44366]

| classifier | q = 5 % | q = 10 % | q = 20 % |
|---|---|---|---|
| FHX | 17692<br>d [15900, 19637]<br>c [13394, 23159]<br>ratio 0.526<br>d [0.492, 0.562]<br>c [0.458, 0.597] | 14581<br>d [13116, 16176]<br>c [11083, 19062]<br>ratio 0.433<br>d [0.402, 0.467]<br>c [0.372, 0.502] | 11071<br>d [9915, 12354]<br>c [8388, 14601]<br>ratio 0.329<br>d [0.300, 0.361]<br>c [0.274, 0.395] |
| FH | 17803<br>d [16021, 19720]<br>c [13493, 23284]<br>ratio 0.529<br>d [0.497, 0.564]<br>c [0.462, 0.599] | 14643<br>d [13184, 16236]<br>c [11137, 19158]<br>ratio 0.435<br>d [0.405, 0.468]<br>c [0.374, 0.504] | 11146<br>d [9953, 12464]<br>c [8427, 14709]<br>ratio 0.331<br>d [0.301, 0.364]<br>c [0.276, 0.398] |
| PHX | 28227<br>d [25275, 31495]<br>c [21002, 37506]<br>ratio 0.839<br>d [0.804, 0.868]<br>c [0.790, 0.882] | 24553<br>d [21915, 27471]<br>c [18119, 33018]<br>ratio 0.730<br>d [0.692, 0.766]<br>c [0.668, 0.790] | 18396<br>d [16326, 20714]<br>c [13519, 24938]<br>ratio 0.547<br>d [0.504, 0.593]<br>c [0.482, 0.621] |
| PH | 29290<br>d [26213, 32706]<br>c [21667, 39111]<br>ratio 0.871<br>d [0.842, 0.897]<br>c [0.828, 0.909] | 24972<br>d [22320, 27951]<br>c [18422, 33597]<br>ratio 0.742<br>d [0.705, 0.779]<br>c [0.681, 0.805] | 18836<br>d [16709, 21230]<br>c [13868, 25487]<br>ratio 0.560<br>d [0.514, 0.609]<br>c [0.493, 0.636] |

| ratio of MSEs on kept rows | q = 5 % | q = 10 % | q = 20 % |
|---|---|---|---|
| FHX over FH | 0.994<br>d [0.979, 1.011]<br>c [0.976, 1.014] | 0.996<br>d [0.983, 1.008]<br>c [0.982, 1.011] | 0.993<br>d [0.969, 1.017]<br>c [0.968, 1.018] |
| PHX over PH | 0.964<br>d [0.943, 0.985]<br>c [0.939, 0.985] | 0.983<br>d [0.960, 1.007]<br>c [0.957, 1.010] | 0.977<br>d [0.943, 1.012]<br>c [0.934, 1.020] |
| FH over PH | 0.608<br>d [0.574, 0.643]<br>c [0.537, 0.679] | 0.586<br>d [0.552, 0.623]<br>c [0.516, 0.659] | 0.592<br>d [0.556, 0.628]<br>c [0.520, 0.662] |
| FHX over PHX | 0.627<br>d [0.589, 0.666]<br>c [0.553, 0.699] | 0.594<br>d [0.557, 0.632]<br>c [0.519, 0.671] | 0.602<br>d [0.564, 0.641]<br>c [0.526, 0.680] |

### V3 rolling origin (fifths 2-5 pooled)

No screening: 41319<br>d [36886, 46114]<br>c [32477, 51893]

| classifier | q = 5 % | q = 10 % | q = 20 % |
|---|---|---|---|
| FHX | 23420<br>d [20899, 26185]<br>c [17934, 30053]<br>ratio 0.567<br>d [0.530, 0.605]<br>c [0.487, 0.646] | 19660<br>d [17577, 21924]<br>c [15148, 25061]<br>ratio 0.476<br>d [0.442, 0.511]<br>c [0.404, 0.548] | 15554<br>d [13860, 17389]<br>c [11919, 19955]<br>ratio 0.376<br>d [0.345, 0.410]<br>c [0.313, 0.444] |
| FH | 23763<br>d [21243, 26539]<br>c [18303, 30422]<br>ratio 0.575<br>d [0.539, 0.612]<br>c [0.499, 0.650] | 19295<br>d [17249, 21519]<br>c [14810, 24687]<br>ratio 0.467<br>d [0.432, 0.503]<br>c [0.397, 0.538] | 15008<br>d [13373, 16805]<br>c [11432, 19345]<br>ratio 0.363<br>d [0.332, 0.397]<br>c [0.301, 0.428] |
| PHX | 36295<br>d [32407, 40484]<br>c [28550, 45817]<br>ratio 0.878<br>d [0.853, 0.902]<br>c [0.843, 0.913] | 32344<br>d [28867, 36100]<br>c [25045, 41533]<br>ratio 0.783<br>d [0.746, 0.818]<br>c [0.723, 0.844] | 26745<br>d [23646, 30142]<br>c [20244, 35094]<br>ratio 0.647<br>d [0.600, 0.693]<br>c [0.566, 0.730] |
| PH | 37032<br>d [33029, 41370]<br>c [28880, 47118]<br>ratio 0.896<br>d [0.874, 0.917]<br>c [0.862, 0.929] | 32382<br>d [28834, 36246]<br>c [25061, 41571]<br>ratio 0.784<br>d [0.750, 0.817]<br>c [0.724, 0.846] | 25859<br>d [22902, 29117]<br>c [19595, 33988]<br>ratio 0.626<br>d [0.577, 0.675]<br>c [0.538, 0.717] |

| ratio of MSEs on kept rows | q = 5 % | q = 10 % | q = 20 % |
|---|---|---|---|
| FHX over FH | 0.986<br>d [0.970, 1.001]<br>c [0.964, 1.004] | 1.019<br>d [1.001, 1.038]<br>c [0.989, 1.051] | 1.036<br>d [1.019, 1.054]<br>c [1.009, 1.067] |
| PHX over PH | 0.980<br>d [0.960, 1.001]<br>c [0.957, 1.007] | 0.999<br>d [0.971, 1.029]<br>c [0.968, 1.031] | 1.034<br>d [1.004, 1.067]<br>c [0.999, 1.075] |
| FH over PH | 0.642<br>d [0.605, 0.678]<br>c [0.563, 0.713] | 0.596<br>d [0.560, 0.632]<br>c [0.523, 0.663] | 0.580<br>d [0.545, 0.617]<br>c [0.508, 0.652] |
| FHX over PHX | 0.645<br>d [0.608, 0.682]<br>c [0.565, 0.718] | 0.608<br>d [0.572, 0.644]<br>c [0.534, 0.678] | 0.582<br>d [0.544, 0.621]<br>c [0.507, 0.661] |

### V4 chronological: fit to 31991, score from 32312

No screening: 55085<br>d [45385, 65949]<br>c [40486, 79043]

| classifier | q = 5 % | q = 10 % | q = 20 % |
|---|---|---|---|
| FHX | 20814<br>d [17097, 24700]<br>c [14084, 25263]<br>ratio 0.378<br>d [0.317, 0.447]<br>c [0.247, 0.550] | 16655<br>d [13414, 20016]<br>c [10411, 20611]<br>ratio 0.302<br>d [0.242, 0.370]<br>c [0.173, 0.459] | 13079<br>d [10498, 15742]<br>c [7490, 16788]<br>ratio 0.237<br>d [0.186, 0.296]<br>c [0.120, 0.381] |
| FH | 20095<br>d [16292, 24099]<br>c [13437, 24562]<br>ratio 0.365<br>d [0.299, 0.436]<br>c [0.229, 0.531] | 16567<br>d [13382, 19806]<br>c [10355, 20709]<br>ratio 0.301<br>d [0.240, 0.370]<br>c [0.171, 0.466] | 13988<br>d [11113, 16906]<br>c [7873, 17954]<br>ratio 0.254<br>d [0.198, 0.319]<br>c [0.130, 0.406] |
| PHX | 51150<br>d [41897, 61595]<br>c [37944, 73360]<br>ratio 0.929<br>d [0.890, 0.964]<br>c [0.884, 0.968] | 46891<br>d [38624, 56168]<br>c [35687, 67209]<br>ratio 0.851<br>d [0.804, 0.899]<br>c [0.789, 0.918] | 38302<br>d [31998, 45470]<br>c [30508, 51561]<br>ratio 0.695<br>d [0.622, 0.772]<br>c [0.597, 0.798] |
| PH | 51270<br>d [42055, 61578]<br>c [37896, 74271]<br>ratio 0.931<br>d [0.895, 0.962]<br>c [0.886, 0.974] | 46034<br>d [38381, 54708]<br>c [35583, 64345]<br>ratio 0.836<br>d [0.780, 0.894]<br>c [0.766, 0.912] | 37684<br>d [31742, 44361]<br>c [30992, 49173]<br>ratio 0.684<br>d [0.617, 0.760]<br>c [0.578, 0.800] |

| ratio of MSEs on kept rows | q = 5 % | q = 10 % | q = 20 % |
|---|---|---|---|
| FHX over FH | 1.036<br>d [0.979, 1.110]<br>c [0.961, 1.136] | 1.005<br>d [0.971, 1.058]<br>c [0.971, 1.067] | 0.935<br>d [0.860, 1.027]<br>c [0.898, 0.987] |
| PHX over PH | 0.998<br>d [0.966, 1.030]<br>c [0.977, 1.018] | 1.019<br>d [0.969, 1.071]<br>c [0.977, 1.085] | 1.016<br>d [0.941, 1.097]<br>c [0.949, 1.101] |
| FH over PH | 0.392<br>d [0.320, 0.470]<br>c [0.243, 0.566] | 0.360<br>d [0.284, 0.443]<br>c [0.202, 0.537] | 0.371<br>d [0.287, 0.464]<br>c [0.195, 0.538] |
| FHX over PHX | 0.407<br>d [0.340, 0.482]<br>c [0.265, 0.587] | 0.355<br>d [0.280, 0.439]<br>c [0.196, 0.526] | 0.341<br>d [0.266, 0.427]<br>c [0.177, 0.512] |

### V4 chronological: fit to 32310, score from 32312

No screening: 55085<br>d [45186, 66054]<br>c [40379, 78959]

| classifier | q = 5 % | q = 10 % | q = 20 % |
|---|---|---|---|
| FHX | 18370<br>d [15065, 21878]<br>c [12184, 22621]<br>ratio 0.333<br>d [0.271, 0.405]<br>c [0.203, 0.507] | 14918<br>d [12119, 17728]<br>c [9266, 18891]<br>ratio 0.271<br>d [0.213, 0.339]<br>c [0.150, 0.431] | 11681<br>d [9484, 13884]<br>c [6653, 15472]<br>ratio 0.212<br>d [0.165, 0.268]<br>c [0.106, 0.355] |
| FH | 18804<br>d [15456, 22329]<br>c [12790, 23147]<br>ratio 0.341<br>d [0.277, 0.416]<br>c [0.215, 0.518] | 16042<br>d [13018, 19207]<br>c [10105, 20084]<br>ratio 0.291<br>d [0.232, 0.359]<br>c [0.169, 0.453] | 12823<br>d [10281, 15375]<br>c [7094, 16955]<br>ratio 0.233<br>d [0.180, 0.295]<br>c [0.116, 0.387] |
| PHX | 47934<br>d [39563, 57355]<br>c [36666, 67885]<br>ratio 0.870<br>d [0.834, 0.906]<br>c [0.816, 0.929] | 43250<br>d [36161, 51141]<br>c [33615, 59158]<br>ratio 0.785<br>d [0.738, 0.836]<br>c [0.716, 0.858] | 33930<br>d [28423, 40216]<br>c [27915, 44578]<br>ratio 0.616<br>d [0.547, 0.693]<br>c [0.519, 0.726] |
| PH | 51113<br>d [41935, 61294]<br>c [37810, 73359]<br>ratio 0.928<br>d [0.893, 0.960]<br>c [0.887, 0.969] | 45449<br>d [37613, 54185]<br>c [34713, 63461]<br>ratio 0.825<br>d [0.763, 0.884]<br>c [0.740, 0.904] | 35988<br>d [30326, 42243]<br>c [29858, 45803]<br>ratio 0.653<br>d [0.580, 0.732]<br>c [0.530, 0.779] |

| ratio of MSEs on kept rows | q = 5 % | q = 10 % | q = 20 % |
|---|---|---|---|
| FHX over FH | 0.977<br>d [0.938, 1.023]<br>c [0.901, 1.028] | 0.930<br>d [0.882, 0.973]<br>c [0.846, 0.972] | 0.911<br>d [0.862, 0.958]<br>c [0.863, 0.969] |
| PHX over PH | 0.938<br>d [0.898, 0.982]<br>c [0.883, 1.001] | 0.952<br>d [0.900, 1.003]<br>c [0.901, 1.002] | 0.943<br>d [0.863, 1.026]<br>c [0.864, 1.072] |
| FH over PH | 0.368<br>d [0.298, 0.449]<br>c [0.231, 0.554] | 0.353<br>d [0.277, 0.439]<br>c [0.206, 0.529] | 0.356<br>d [0.275, 0.450]<br>c [0.188, 0.525] |
| FHX over PHX | 0.383<br>d [0.313, 0.463]<br>c [0.237, 0.556] | 0.345<br>d [0.271, 0.428]<br>c [0.197, 0.524] | 0.344<br>d [0.272, 0.424]<br>c [0.188, 0.507] |

## Row counts

| design | scored rows | discharges | clusters | fit rows per unit (min-max) | units |
|---|---|---|---|---|---|
| V1 random (r2-r4) | 209,432 | 606 | 85 | 161,014-173,692 | 15 |
| V2 cluster-blocked (3 assignments) | 209,432 | 606 | 85 | 151,083-179,173 | 15 |
| V3 rolling origin (fifths 2-5 pooled) | 150,614 | 479 | 64 | 58,818-181,942 | 4 |
| V4 chronological: fit to 31991, score from 32312 | 18,277 | 83 | 19 | 160,401-160,401 | 1 |
| V4 chronological: fit to 32310, score from 32312 | 18,277 | 83 | 19 | 191,155-191,155 | 1 |

## Checks (details in `docs/paper/b77_checks.md`)

- 1a: T0 and TX against B.75's D10 L5 estimates on the rows both have: median absolute deviation 2.80 / 4.70 eV, largest 980.9 / 996.7 eV (row sets differ; reported).
- 1b: this code on B.75's rows reproduces B.75 bit for bit: True.
- 1c: C0 without the four frame scalars equals T0 bit for bit: True.
- 2: no input or estimate of C0 or CX changes when values of rows after t change: True; positive control: True.
- 3: inputs and target: True.

Readout run time 90 s.
