# B.81 — the two controls and the two halves of the fast inputs under the stale-history rules, every design

Registered: `ces_prediction/experiments/PREREGISTRATION_B81.md`; implementation choices in `ces_prediction/experiments/PREREGISTRATION_B81_ADDENDUM1.md`, written before any run. CPU only; boosted regressions of the kind of B.75, no network. Target L5 (the median of the measured values among the next five rows). Rows, inputs, units, regression and target transform of B.75; rule D650 on B.75's common set, rule S on B.75's S set. Exploratory on used data (registration section 1).

Estimators. T0: the ten history scalars. TX: T0's inputs, the eight fast scalars of row t and the six anchor scalars. TXp: TX's inputs with the fourteen fast and anchor columns permuted jointly over rows (within the fit rows and within the test rows) and the estimator refitted. TXo: the ten history scalars, the eight fast scalars of the anchor row (the row of the newest visible CES value) and a1 of BES and of ECEI. TE (TB): T0's inputs and the ECEI (BES) half of TX's fast inputs. T0, TX and, under V1 and V4, TE and TB are the stored estimates of B.75; the rest is fitted here.

Conventions. Skill of A over R = 1 - MSE(A) / MSE(R) on identical rows. Each value: point; 95 % interval with shot clusters resampled (c); with discharges resampled (d); 20,000 draws (V1: a re-partition drawn first; V2: an assignment drawn first; V3, V4: the scored units). Verdicts on the cluster interval. Points under V1 and V2: mean over the three re-partitions or assignments. RMSE in eV. Checks: `docs/paper/b81_checks.md` (all passed before this readout: reuse True, permutation True, anchor True, rows True).

## Statements (target L5, cluster interval)

| Id | statement | point | cluster interval | discharge interval | verdict | registered expectation | as expected | note |
|---|---|---|---|---|---|---|---|---|
| C1 | V1, D650: TXp over T0 | -0.004 | [-0.008, +0.001] | [-0.008, +0.001] | holds | holds | yes |  |
| C2 | V1, S: TXp over T0 | -0.000 | [-0.010, +0.009] | [-0.009, +0.010] | holds | holds | yes |  |
| C3 | V1, D650: TX over TXo lies above zero (the present samples add to those of the anchor row) | +0.273 | [+0.229, +0.317] | [+0.234, +0.309] | holds | holds | yes |  |
| C4 | V1, S: TX over TXo lies above zero | +0.276 | [+0.215, +0.333] | [+0.227, +0.323] | holds | holds | yes |  |
| C5 | V2, D650: TX over TXo lies above zero | +0.298 | [+0.232, +0.363] | [+0.255, +0.336] | holds | holds | yes |  |
| C6 | V3, D650: TX over TXo lies above zero | +0.426 | [+0.307, +0.562] | [+0.354, +0.495] | holds | uncertain, leaning to holds | n/a (uncertain) |  |
| C7 | V4a, D650: TX over TXo lies above zero | +0.357 | [+0.235, +0.523] | [+0.249, +0.453] | holds | uncertain (the lead's one-unit point is +0.205 for present, -0.128 for the anchor row, no interval) | n/a (uncertain) | V4b (fit to 32310): +0.192; c [+0.084, +0.286] |
| C8 | V2, S: TX over TXo lies above zero | +0.296 | [+0.205, +0.391] | [+0.233, +0.362] | holds | holds | yes |  |
| C9 | V3, S: TX over TXo lies above zero | +0.274 | [+0.192, +0.373] | [+0.212, +0.334] | holds | uncertain, leaning to holds | n/a (uncertain) |  |
| E1 | V2, D650: TE over T0 lies above zero | +0.163 | [+0.090, +0.231] | [+0.112, +0.210] | holds | holds | yes |  |
| E2 | V3, D650: TE over T0 lies above zero | +0.058 | [-0.067, +0.166] | [-0.002, +0.115] | does not hold | uncertain, leaning to holds | n/a (uncertain) |  |
| E3 | V2, D650: TB over T0 lies above zero | +0.225 | [+0.163, +0.285] | [+0.184, +0.266] | holds | holds | yes |  |
| E4 | V3, D650: TB over T0 lies above zero | +0.122 | [+0.028, +0.234] | [+0.070, +0.175] | holds | uncertain (under V4a it is -0.014 [-0.257, +0.203]) | n/a (uncertain) |  |
| E5 | V2, S: TE over T0 lies above zero | +0.181 | [+0.098, +0.257] | [+0.124, +0.234] | holds | holds | yes |  |
| E6 | V3, S: TE over T0 lies above zero | +0.160 | [+0.066, +0.257] | [+0.091, +0.222] | holds | uncertain, leaning to holds | n/a (uncertain) |  |
| E7 | V2, S: TB over T0 lies above zero | +0.221 | [+0.157, +0.297] | [+0.170, +0.272] | holds | holds | yes |  |
| E8 | V3, S: TB over T0 lies above zero | +0.171 | [+0.119, +0.229] | [+0.119, +0.221] | holds | uncertain | n/a (uncertain) |  |

C1 and C2 hold when the lower end of the cluster interval is at most zero; the others when it is above zero.

## The two controls by design and rule

TXp over T0 (fast inputs permuted over rows), TXo over T0 (fast inputs of the anchor row), TX over TXo (the present samples beside those of the anchor row), with TX over T0 (the fast-channel increment of B.75, recomputed here on this batch's draws) beside them.

### D650 (history 0.65 s old)

| design | TX/T0 | TXp/T0 | TXo/T0 | TX/TXo |
|---|---|---|---|---|
| V1 random (r2-r4) | +0.336<br>c [+0.285, +0.381]<br>d [+0.299, +0.372] | -0.004<br>c [-0.008, +0.001]<br>d [-0.008, +0.001] | +0.087<br>c [+0.033, +0.134]<br>d [+0.047, +0.127] | +0.273<br>c [+0.229, +0.317]<br>d [+0.234, +0.309] |
| V2 cluster-blocked (3 assignments) | +0.282<br>c [+0.214, +0.343]<br>d [+0.244, +0.318] | -0.005<br>c [-0.010, -0.000]<br>d [-0.009, -0.000] | -0.023<br>c [-0.103, +0.047]<br>d [-0.070, +0.022] | +0.298<br>c [+0.232, +0.363]<br>d [+0.255, +0.336] |
| V3 rolling origin (fifths 2-5 pooled) | +0.190<br>c [+0.031, +0.291]<br>d [+0.133, +0.242] | -0.001<br>c [-0.004, +0.004]<br>d [-0.005, +0.004] | -0.411<br>c [-0.940, -0.118]<br>d [-0.636, -0.234] | +0.426<br>c [+0.307, +0.562]<br>d [+0.354, +0.495] |
| V4a chronological: fit to 31991, score from 32312 | +0.237<br>c [+0.041, +0.416]<br>d [+0.069, +0.356] | +0.012<br>c [-0.010, +0.028]<br>d [-0.005, +0.029] | -0.187<br>c [-0.470, +0.048]<br>d [-0.503, +0.032] | +0.357<br>c [+0.235, +0.523]<br>d [+0.249, +0.453] |
| V4b chronological: fit to 32310, score from 32312 | +0.216<br>c [+0.075, +0.360]<br>d [+0.101, +0.307] | -0.012<br>c [-0.026, +0.000]<br>d [-0.022, -0.003] | +0.030<br>c [-0.059, +0.178]<br>d [-0.110, +0.137] | +0.192<br>c [+0.084, +0.286]<br>d [+0.120, +0.264] |

### S (beam-modulation schedule)

| design | TX/T0 | TXp/T0 | TXo/T0 | TX/TXo |
|---|---|---|---|---|
| V1 random (r2-r4) | +0.327<br>c [+0.262, +0.388]<br>d [+0.277, +0.375] | -0.000<br>c [-0.010, +0.009]<br>d [-0.009, +0.010] | +0.070<br>c [+0.006, +0.136]<br>d [+0.013, +0.126] | +0.276<br>c [+0.215, +0.333]<br>d [+0.227, +0.323] |
| V2 cluster-blocked (3 assignments) | +0.293<br>c [+0.217, +0.364]<br>d [+0.242, +0.341] | +0.000<br>c [-0.011, +0.011]<br>d [-0.010, +0.010] | -0.005<br>c [-0.150, +0.109]<br>d [-0.101, +0.071] | +0.296<br>c [+0.205, +0.391]<br>d [+0.233, +0.362] |
| V3 rolling origin (fifths 2-5 pooled) | +0.231<br>c [+0.145, +0.325]<br>d [+0.166, +0.289] | -0.000<br>c [-0.010, +0.009]<br>d [-0.009, +0.008] | -0.059<br>c [-0.161, +0.047]<br>d [-0.136, +0.009] | +0.274<br>c [+0.192, +0.373]<br>d [+0.212, +0.334] |
| V4a chronological: fit to 31991, score from 32312 | +0.028<br>c [-0.164, +0.173]<br>d [-0.133, +0.159] | +0.022<br>c [-0.038, +0.068]<br>d [-0.034, +0.069] | +0.038<br>c [-0.255, +0.293]<br>d [-0.199, +0.215] | -0.011<br>c [-0.408, +0.171]<br>d [-0.201, +0.142] |
| V4b chronological: fit to 32310, score from 32312 | -0.254<br>c [-0.535, +0.044]<br>d [-0.525, -0.051] | +0.029<br>c [+0.002, +0.064]<br>d [-0.002, +0.068] | -0.120<br>c [-0.253, +0.046]<br>d [-0.295, +0.016] | -0.120<br>c [-0.443, +0.132]<br>d [-0.391, +0.116] |

## Which diagnostic: the ECEI half (TE) and the BES half (TB) over T0

Under V1 and V4 from B.75's stored estimates (intervals recomputed on this batch's draws), under V2 and V3 fitted here.

### D650 (history 0.65 s old)

| design | TE/T0 | TB/T0 | TX/T0 |
|---|---|---|---|
| V1 random (r2-r4) | +0.229<br>c [+0.162, +0.289]<br>d [+0.188, +0.268] | +0.246<br>c [+0.199, +0.292]<br>d [+0.209, +0.282] | +0.336<br>c [+0.285, +0.381]<br>d [+0.299, +0.372] |
| V2 cluster-blocked (3 assignments) | +0.163<br>c [+0.090, +0.231]<br>d [+0.112, +0.210] | +0.225<br>c [+0.163, +0.285]<br>d [+0.184, +0.266] | +0.282<br>c [+0.214, +0.343]<br>d [+0.244, +0.318] |
| V3 rolling origin (fifths 2-5 pooled) | +0.058<br>c [-0.067, +0.166]<br>d [-0.002, +0.115] | +0.122<br>c [+0.028, +0.234]<br>d [+0.070, +0.175] | +0.190<br>c [+0.031, +0.291]<br>d [+0.133, +0.242] |
| V4a chronological: fit to 31991, score from 32312 | +0.177<br>c [+0.085, +0.306]<br>d [+0.082, +0.269] | -0.014<br>c [-0.267, +0.201]<br>d [-0.295, +0.172] | +0.237<br>c [+0.041, +0.416]<br>d [+0.069, +0.356] |
| V4b chronological: fit to 32310, score from 32312 | +0.145<br>c [+0.047, +0.294]<br>d [+0.059, +0.230] | +0.094<br>c [-0.063, +0.247]<br>d [-0.053, +0.203] | +0.216<br>c [+0.075, +0.360]<br>d [+0.101, +0.307] |

### S (beam-modulation schedule)

| design | TE/T0 | TB/T0 | TX/T0 |
|---|---|---|---|
| V1 random (r2-r4) | +0.221<br>c [+0.156, +0.286]<br>d [+0.169, +0.271] | +0.242<br>c [+0.187, +0.294]<br>d [+0.195, +0.285] | +0.327<br>c [+0.262, +0.388]<br>d [+0.277, +0.375] |
| V2 cluster-blocked (3 assignments) | +0.181<br>c [+0.098, +0.257]<br>d [+0.124, +0.234] | +0.221<br>c [+0.157, +0.297]<br>d [+0.170, +0.272] | +0.293<br>c [+0.217, +0.364]<br>d [+0.242, +0.341] |
| V3 rolling origin (fifths 2-5 pooled) | +0.160<br>c [+0.066, +0.257]<br>d [+0.091, +0.222] | +0.171<br>c [+0.119, +0.229]<br>d [+0.119, +0.221] | +0.231<br>c [+0.145, +0.325]<br>d [+0.166, +0.289] |
| V4a chronological: fit to 31991, score from 32312 | -0.045<br>c [-0.242, +0.089]<br>d [-0.218, +0.088] | -0.047<br>c [-0.317, +0.255]<br>d [-0.303, +0.151] | +0.028<br>c [-0.164, +0.173]<br>d [-0.133, +0.159] |
| V4b chronological: fit to 32310, score from 32312 | -0.296<br>c [-0.593, +0.037]<br>d [-0.635, -0.055] | -0.096<br>c [-0.268, +0.185]<br>d [-0.323, +0.100] | -0.254<br>c [-0.535, +0.044]<br>d [-0.525, -0.051] |

## RMSE (eV) and counts

| design | rule | rows | discharges | clusters | T0 | TX | TXp | TXo | TE | TB |
|---|---|---|---|---|---|---|---|---|---|---|
| V1 | D650 | 160,343 | 562 | 79 | 226.2 | 184.3 | 226.7 | 216.1 | 198.6 | 196.4 |
| V1 | S | 147,837 | 503 | 71 | 217.6 | 178.5 | 217.6 | 209.8 | 192.0 | 189.4 |
| V2 | D650 | 160,343 | 562 | 79 | 236.7 | 200.6 | 237.3 | 239.4 | 216.5 | 208.3 |
| V2 | S | 147,837 | 503 | 71 | 223.2 | 187.7 | 223.2 | 223.7 | 202.0 | 197.0 |
| V3 | D650 | 112,199 | 441 | 60 | 268.5 | 241.7 | 268.6 | 318.9 | 260.7 | 251.6 |
| V3 | S | 102,684 | 395 | 56 | 249.1 | 218.5 | 249.2 | 256.4 | 228.4 | 226.8 |
| V4a | D650 | 10,146 | 75 | 18 | 202.3 | 176.7 | 201.0 | 220.4 | 183.5 | 203.8 |
| V4a | S | 7,180 | 50 | 14 | 184.8 | 182.3 | 182.8 | 181.3 | 188.9 | 189.1 |
| V4b | D650 | 10,146 | 75 | 18 | 189.8 | 168.1 | 190.9 | 187.0 | 175.5 | 180.7 |
| V4b | S | 7,180 | 50 | 14 | 170.3 | 190.8 | 167.8 | 180.3 | 193.9 | 178.4 |

Fit rows per unit and TXo rows whose anchor-row fast scalars are not all finite (kept, left to the regression):

| design | rule | fit rows per unit (min-max) | TXo: such test rows (total) | TXo: such fit rows per unit (max) |
|---|---|---|---|---|
| V1 | D650 | 122,684-133,674 | 42 | 14 |
| V1 | S | 112,909-123,415 | 225 | 75 |
| V2 | D650 | 113,440-138,804 | 42 | 14 |
| V2 | S | 105,194-128,163 | 225 | 75 |
| V3 | D650 | 48,144-143,366 | 4 | 14 |
| V3 | S | 45,153-135,332 | 75 | 32 |
| V4a | D650 | 127,270-127,270 | 0 | 14 |
| V4a | S | 119,655-119,655 | 43 | 0 |
| V4b | D650 | 150,197-150,197 | 0 | 14 |
| V4b | S | 140,657-140,657 | 43 | 32 |

## V1 with the rows left out where the ECEI is constant (a description, defined after the fact)

A row is left out when the four ECEI channels have not changed for at least ten rows at row t or at the anchor row (definition of `b75_availability/lead_checks/stuck_ecei.py`). The stratum was defined after B.75 had been read; no statement uses it.

| rule | rows kept | rows left out | discharges | clusters | TX/T0 | TXp/T0 | TXo/T0 | TX/TXo |
|---|---|---|---|---|---|---|---|---|
| D650 | 153,655 | 6,688 | 559 | 78 | +0.348<br>c [+0.298, +0.392]<br>d [+0.310, +0.383] | -0.004<br>c [-0.008, +0.001]<br>d [-0.008, +0.001] | +0.088<br>c [+0.035, +0.136]<br>d [+0.048, +0.128] | +0.285<br>c [+0.244, +0.325]<br>d [+0.248, +0.321] |
| S | 140,634 | 7,203 | 495 | 69 | +0.334<br>c [+0.272, +0.391]<br>d [+0.283, +0.382] | -0.000<br>c [-0.010, +0.009]<br>d [-0.010, +0.009] | +0.068<br>c [-0.002, +0.137]<br>d [+0.008, +0.126] | +0.286<br>c [+0.224, +0.341]<br>d [+0.237, +0.331] |

Beside it, V1 on all rows (from the tables above):

| rule | TX/T0 | TXp/T0 | TXo/T0 | TX/TXo |
|---|---|---|---|---|
| D650 | +0.336<br>c [+0.285, +0.381]<br>d [+0.299, +0.372] | -0.004<br>c [-0.008, +0.001]<br>d [-0.008, +0.001] | +0.087<br>c [+0.033, +0.134]<br>d [+0.047, +0.127] | +0.273<br>c [+0.229, +0.317]<br>d [+0.234, +0.309] |
| S | +0.327<br>c [+0.262, +0.388]<br>d [+0.277, +0.375] | -0.000<br>c [-0.010, +0.009]<br>d [-0.009, +0.010] | +0.070<br>c [+0.006, +0.136]<br>d [+0.013, +0.126] | +0.276<br>c [+0.215, +0.333]<br>d [+0.227, +0.323] |

Readout run time 44.2 s.
