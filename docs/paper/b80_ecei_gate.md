# B.80 — the coherence of the ECEI channels as an input-only condition for the fast-channel increment

Registered: `ces_prediction/experiments/PREREGISTRATION_B80.md`; implementation choices in `ces_prediction/experiments/PREREGISTRATION_B80_ADDENDUM1.md`, written before any readout. CPU only; **no model is fitted**: the stored estimates of B.75 (T0, TX, TB, TE; `data/.b75/fits/`) and of the main run of B.79 (TN, TXN, beam columns as delivered; `data/.b79/fits/`), target L5, rule S on B.75's S set and rule D650 on B.75's common set, and input-only measures of the four raw ECEI channels. Exploratory on used data (registration section 1).

**C_E** = the mean of the six pairwise Pearson correlations of the four raw ECEI channels over the grid rows of the discharge (first and last row of the file excluded; discharges with at least 50 grid rows; undefined where a channel is constant). Classes: low C_E < 0.7, middle 0.7 to below 0.9, high 0.9 and above; the class of a row is the class of its discharge; a discharge without C_E belongs to no class. **TG** = TX where C_E is 0.7 or above, T0 otherwise (also without C_E), assembled row by row from the stored estimates; no fit.

Conventions. Skill of A over B = 1 - SSE(A) / SSE(B) on identical rows. Each value: point, then 95 % intervals with discharges resampled (d) and with shot clusters resampled (c), 20,000 draws, seed 20261180 (V1: a re-partition drawn first; V2: an assignment drawn first; V3, V4: the scored units). Verdicts on the cluster form. Points under V1 and V2 = mean over the three splits. A skill within a class resamples the units with rows in that class; a difference high minus low is formed draw by draw on draws over the units with rows in the low or the high class (a draw with no unit of one class is left out; the count is given where it is not zero). RMSE in eV.

Order of work. Before this readout, these results existed: the input-only measures (their distribution, section 'The measures'), and the check of registration section 8. No skill, RMSE or statement by class and no gated estimate had been computed.

## Statements (cluster interval)

| Id | statement | value: point; d interval; c interval | verdict | registered expectation | expectation met | note |
|---|---|---|---|---|---|---|
| G1 | V1, S: TX over T0 in the high class minus that in the low class lies above zero | +0.151; d [+0.018, +0.276]; c [-0.017, +0.277] | **does not hold** | holds | no | high +0.370; c [+0.293, +0.433]; low +0.218; c [+0.113, +0.371]; middle +0.292; c [+0.170, +0.397] |
| G2 | V1, S: TE over T0 in the low class has an interval that contains zero or lies below it (the ECEI half adds nothing where its channels are not coherent) | +0.165; d [+0.035, +0.280]; c [+0.043, +0.302] | **does not hold** | holds | no | TB over T0 in the low class: +0.144; c [+0.046, +0.262] |
| G3 | V1, S: TE over T0 in the high class lies above zero | +0.240; d [+0.176, +0.303]; c [+0.152, +0.317] | **holds** | holds | yes | TB over T0 in the high class: +0.269; c [+0.204, +0.336] |
| G4 | V1, D650: TX over T0 in the high class minus that in the low class lies above zero | +0.253; d [+0.168, +0.339]; c [+0.154, +0.354] | **holds** | holds | yes | high +0.396; c [+0.349, +0.439]; low +0.143; c [+0.039, +0.232]; middle +0.319; c [+0.245, +0.393] |
| G5 | V1, S, blocks up to 32310 only: TX over T0 in the high class minus that in the low class lies above zero (the ordering holds inside the earlier blocks, not only between blocks) | +0.141; d [+0.000, +0.276]; c [-0.046, +0.270] | **does not hold** | uncertain, leaning to holds | n/a (uncertain) | high +0.373; c [+0.299, +0.436]; low +0.232; c [+0.116, +0.407]; middle +0.304; c [+0.175, +0.410] |
| G6 | V3, S: TX over T0 in the high class lies above zero | +0.287; d [+0.215, +0.355]; c [+0.194, +0.396] | **holds** | holds | yes | low class: -0.032; c [-0.190, +0.199] |
| G7 | V1, S, block from 32312: TG over T0 lies above zero | +0.010; d [-0.148, +0.132]; c [-0.198, +0.112] | **does not hold** | uncertain | n/a (uncertain) | TX over T0 in the block: +0.066; c [-0.170, +0.248]; TG over TX: -0.061; c [-0.279, +0.046] |
| G8 | V4 (fit to 31991), S: TG over T0 lies above zero | -0.025; d [-0.132, +0.046]; c [-0.163, +0.052] | **does not hold** | uncertain | n/a (uncertain) | V4b (fit to 32310): +0.016; c [-0.112, +0.096]; V4a TX over T0: +0.028; c [-0.169, +0.175] |
| G9 | V1, S, all blocks: TG is not worse than TX (the interval of TG over TX is not below zero) | -0.067; d [-0.121, -0.029]; c [-0.128, -0.027] | **does not hold** | holds | no | TG over T0: +0.282; c [+0.210, +0.350] |

G1, G4 and G5 are differences of two class skills (TX over T0, high class minus low class). G2 holds when the lower end of the cluster interval is at or below zero; G9 holds when its upper end is at or above zero. 'V4' in G8 is V4a (fit to 31991); V4b is in the note.

## Check of registration section 8

TX over T0 over all rows (all classes), target L5, recomputed from the stored estimates read by this batch, against the values printed in `docs/paper/b75_availability.md`. Pass rule: equal to the third decimal.

| design | rule | rows | recomputed | published | difference to the JSON of B.75 | pass |
|---|---|---|---|---|---|---|
| V1 random (r2-r4) | D650 | 160,343 | +0.336289 | +0.336 | +0.0e+00 | yes |
| V1 random (r2-r4) | S | 147,837 | +0.327200 | +0.327 | +0.0e+00 | yes |
| V2 cluster-blocked (3 assignments) | D650 | 160,343 | +0.281972 | +0.282 | -1.7e-16 | yes |
| V2 cluster-blocked (3 assignments) | S | 147,837 | +0.292835 | +0.293 | +0.0e+00 | yes |
| V3 rolling origin (fifths 2-5 pooled) | D650 | 112,199 | +0.189962 | +0.190 | -1.1e-16 | yes |
| V3 rolling origin (fifths 2-5 pooled) | S | 102,684 | +0.231036 | +0.231 | -1.1e-16 | yes |
| V4 chronological: fit to 31991, score from 32312 | D650 | 10,146 | +0.236781 | +0.237 | +0.0e+00 | yes |
| V4 chronological: fit to 31991, score from 32312 | S | 7,180 | +0.027778 | +0.028 | +0.0e+00 | yes |
| V4 chronological: fit to 32310, score from 32312 | D650 | 10,146 | +0.216219 | +0.216 | -1.1e-16 | yes |
| V4 chronological: fit to 32310, score from 32312 | S | 7,180 | -0.254064 | -0.254 | +2.2e-16 | yes |

All cells pass: yes (stage `check80.py`); recomputed in the readout pass: yes.

## The measures, per block of shots

Inputs only; population: the D1 discharges. C_E early: the same measure over the first 200 grid rows of the discharge (equal to C_E for discharges with at most 200 grid rows). N_E: the median over the absent steps of the discharge with five rows of the block before them of the ECEI array mean in the first row after the step over its median over the five rows before (at least three steps). No statement uses C_E early or N_E.

| block | D1 discharges | with C_E | without C_E: fewer than 50 grid rows / constant channel | low / middle / high | share low | C_E 10 / 25 / 50 / 75 / 90 % |
|---|---|---|---|---|---|---|
| all blocks | 635 | 590 | 41 / 4 | 170 / 135 / 285 | 0.288 | 0.276 / 0.626 / 0.895 / 0.961 / 0.980 |
| block up to 31991 | 412 | 393 | 19 / 0 | 87 / 95 / 211 | 0.221 | 0.279 / 0.760 / 0.911 / 0.962 / 0.982 |
| block 32002-32310 | 128 | 115 | 9 / 4 | 40 / 32 / 43 | 0.348 | 0.323 / 0.583 / 0.864 / 0.953 / 0.978 |
| block from 32312 | 95 | 82 | 13 / 0 | 43 / 8 / 31 | 0.524 | 0.141 / 0.385 / 0.684 / 0.960 / 0.979 |

| block | C_E early defined | same class as C_E (share) | same class, discharges with more than 200 grid rows | discharges with at most 200 grid rows | C_E early median | N_E defined | N_E 10 / 50 / 90 % | absent steps used | steps with a reference not above zero |
|---|---|---|---|---|---|---|---|---|---|
| all blocks | 588 | 0.852 (588) | 0.810 | 130 | 0.885 | 259 | 0.668 / 0.920 / 1.051 | 1401 | 6 |
| block up to 31991 | 391 | 0.821 (391) | 0.782 | 70 | 0.905 | 215 | 0.660 / 0.901 / 1.039 | 1097 | 2 |
| block 32002-32310 | 115 | 0.887 (115) | 0.863 | 20 | 0.853 | 42 | 0.777 / 0.966 / 1.097 | 232 | 3 |
| block from 32312 | 82 | 0.951 (82) | 0.905 | 40 | 0.654 | 2 | 0.996 / 1.013 / 1.031 | 72 | 1 |

All 641 grid files: 591 with C_E (low / middle / high 171 / 135 / 285; without C_E: 46 with fewer than 50 grid rows, 4 with a constant channel); median C_E 0.895.

## By class of C_E, per design and rule

Cells: rows / discharges / clusters scored; RMSE (points); skills with intervals. TB and TE exist for V1 and V4 only. The last row is the difference of the class skills, high class minus low class.

### V1 random (r2-r4), rule S (S set)

| stratum | rows / discharges / clusters | RMSE T0 | RMSE TX | TX over T0 | TXN over TN | TB over T0 | TE over T0 |
|---|---|---|---|---|---|---|---|
| all | 147,837 / 503 / 71 | 217.6 | 178.5 | +0.327<br>d [+0.277, +0.375]<br>c [+0.264, +0.388] | +0.253<br>d [+0.202, +0.305]<br>c [+0.183, +0.319] | +0.242<br>d [+0.194, +0.285]<br>c [+0.188, +0.293] | +0.221<br>d [+0.168, +0.271]<br>c [+0.156, +0.285] |
| class low | 45,766 / 135 / 46 | 177.1 | 156.6 | +0.218<br>d [+0.106, +0.338]<br>c [+0.113, +0.371] | +0.129<br>d [+0.029, +0.222]<br>c [+0.029, +0.247] | +0.144<br>d [+0.048, +0.236]<br>c [+0.046, +0.262] | +0.165<br>d [+0.035, +0.280]<br>c [+0.043, +0.302] |
| class middle | 35,335 / 123 / 39 | 168.8 | 141.9 | +0.292<br>d [+0.167, +0.398]<br>c [+0.170, +0.397] | +0.212<br>d [+0.104, +0.321]<br>c [+0.094, +0.332] | +0.259<br>d [+0.151, +0.349]<br>c [+0.152, +0.351] | +0.209<br>d [+0.031, +0.334]<br>c [+0.026, +0.332] |
| class high | 66,363 / 241 / 51 | 261.5 | 207.6 | +0.370<br>d [+0.310, +0.425]<br>c [+0.293, +0.433] | +0.301<br>d [+0.240, +0.362]<br>c [+0.223, +0.368] | +0.269<br>d [+0.210, +0.324]<br>c [+0.204, +0.336] | +0.240<br>d [+0.176, +0.303]<br>c [+0.152, +0.317] |
| no class | 373 / 4 / 1 | 227.7 | 197.9 | +0.242<br>d [-0.567, +0.537]<br>c [+0.158, +0.354] | -0.086<br>d [-0.490, +0.329]<br>c [-0.326, +0.144] | +0.149<br>d [-0.867, +0.559]<br>c [+0.070, +0.191] | +0.337<br>d [-0.221, +0.573]<br>c [+0.229, +0.420] |
| high minus low (rows of low and high: 112,129 / 376 / 68) | | | | +0.151<br>d [+0.018, +0.276]<br>c [-0.017, +0.277] | +0.172<br>d [+0.073, +0.276]<br>c [+0.040, +0.282] | +0.125<br>d [+0.013, +0.233]<br>c [-0.012, +0.243] | +0.075<br>d [-0.058, +0.209]<br>c [-0.086, +0.213] |

### V2 cluster-blocked (3 assignments), rule S (S set)

| stratum | rows / discharges / clusters | RMSE T0 | RMSE TX | TX over T0 | TXN over TN | TB over T0 | TE over T0 |
|---|---|---|---|---|---|---|---|
| all | 147,837 / 503 / 71 | 223.2 | 187.7 | +0.293<br>d [+0.241, +0.341]<br>c [+0.216, +0.365] | +0.238<br>d [+0.172, +0.301]<br>c [+0.159, +0.317] | - | - |
| class low | 45,766 / 135 / 46 | 183.6 | 163.4 | +0.207<br>d [+0.083, +0.352]<br>c [+0.060, +0.383] | +0.078<br>d [+0.007, +0.151]<br>c [-0.016, +0.169] | - | - |
| class middle | 35,335 / 123 / 39 | 173.7 | 147.3 | +0.279<br>d [+0.119, +0.400]<br>c [+0.097, +0.410] | +0.279<br>d [+0.149, +0.424]<br>c [+0.140, +0.438] | - | - |
| class high | 66,363 / 241 / 51 | 267.3 | 219.8 | +0.323<br>d [+0.255, +0.386]<br>c [+0.232, +0.406] | +0.275<br>d [+0.193, +0.346]<br>c [+0.177, +0.356] | - | - |
| no class | 373 / 4 / 1 | 211.9 | 198.0 | +0.121<br>d [-0.430, +0.419]<br>c [+0.054, +0.238] | +0.032<br>d [-0.371, +0.249]<br>c [-0.195, +0.199] | - | - |
| high minus low (rows of low and high: 112,129 / 376 / 68) | | | | +0.116<br>d [-0.059, +0.266]<br>c [-0.085, +0.290] | +0.198<br>d [+0.100, +0.296]<br>c [+0.072, +0.320] | - | - |

### V3 rolling origin (fifths 2-5 pooled), rule S (S set)

| stratum | rows / discharges / clusters | RMSE T0 | RMSE TX | TX over T0 | TXN over TN | TB over T0 | TE over T0 |
|---|---|---|---|---|---|---|---|
| all | 102,684 / 395 / 56 | 249.1 | 218.5 | +0.231<br>d [+0.167, +0.290]<br>c [+0.146, +0.326] | +0.175<br>d [+0.121, +0.226]<br>c [+0.101, +0.254] | - | - |
| class low | 30,469 / 111 / 40 | 182.7 | 185.6 | -0.032<br>d [-0.243, +0.139]<br>c [-0.190, +0.199] | -0.075<br>d [-0.202, +0.039]<br>c [-0.215, +0.093] | - | - |
| class middle | 27,197 / 97 / 32 | 175.0 | 152.6 | +0.240<br>d [+0.119, +0.347]<br>c [+0.102, +0.362] | +0.258<br>d [+0.126, +0.378]<br>c [+0.100, +0.413] | - | - |
| class high | 44,645 / 183 / 38 | 317.4 | 267.9 | +0.287<br>d [+0.215, +0.355]<br>c [+0.194, +0.396] | +0.210<br>d [+0.146, +0.269]<br>c [+0.125, +0.293] | - | - |
| no class | 373 / 4 / 1 | 260.0 | 187.5 | +0.480<br>d [+0.360, +0.572]<br>c [+0.480, +0.480] | +0.110<br>d [-0.317, +0.296]<br>c [+0.110, +0.110] | - | - |
| high minus low (rows of low and high: 75,114 / 294 / 53) | | | | +0.319<br>d [+0.131, +0.543]<br>c [+0.091, +0.492] | +0.285<br>d [+0.153, +0.425]<br>c [+0.101, +0.451] | - | - |

### V4 chronological: fit to 31991, score from 32312, rule S (S set)

| stratum | rows / discharges / clusters | RMSE T0 | RMSE TX | TX over T0 | TXN over TN | TB over T0 | TE over T0 |
|---|---|---|---|---|---|---|---|
| all | 7,180 / 50 / 14 | 184.8 | 182.3 | +0.028<br>d [-0.132, +0.159]<br>c [-0.169, +0.175] | +0.102<br>d [-0.057, +0.232]<br>c [-0.155, +0.245] | -0.047<br>d [-0.302, +0.148]<br>c [-0.317, +0.253] | -0.045<br>d [-0.221, +0.088]<br>c [-0.250, +0.091] |
| class low | 4,686 / 25 / 10 | 190.1 | 182.7 | +0.077<br>d [-0.092, +0.226]<br>c [-0.112, +0.234] | +0.098<br>d [-0.080, +0.247]<br>c [-0.209, +0.261] | +0.039<br>d [-0.226, +0.258]<br>c [-0.254, +0.287] | -0.037<br>d [-0.224, +0.100]<br>c [-0.227, +0.149] |
| class middle | 473 / 4 / 3 | 160.8 | 201.0 | -0.561<br>d [-2.317, +0.226]<br>c [-3.114, +0.233] | -0.445<br>d [-0.785, +0.187]<br>c [-0.875, +0.187] | -0.441<br>d [-0.931, +0.003]<br>c [-1.153, +0.003] | -0.443<br>d [-1.781, +0.364]<br>c [-2.330, +0.370] |
| class high | 2,021 / 21 / 8 | 177.5 | 176.5 | +0.011<br>d [-0.206, +0.161]<br>c [-0.185, +0.302] | +0.238<br>d [-0.010, +0.369]<br>c [+0.076, +0.409] | -0.199<br>d [-0.998, +0.188]<br>c [-0.715, +0.314] | +0.011<br>d [-0.425, +0.256]<br>c [-0.461, +0.311] |
| high minus low (rows of low and high: 6,707 / 46 / 13) | | | | -0.066<br>d [-0.328, +0.164]<br>c [-0.298, +0.273] | +0.140<br>d [-0.146, +0.366]<br>c [-0.060, +0.453] | -0.238<br>d [-1.082, +0.247]<br>c [-0.865, +0.315] | +0.047<br>d [-0.422, +0.361]<br>c [-0.491, +0.418] |

### V4 chronological: fit to 32310, score from 32312, rule S (S set)

| stratum | rows / discharges / clusters | RMSE T0 | RMSE TX | TX over T0 | TXN over TN | TB over T0 | TE over T0 |
|---|---|---|---|---|---|---|---|
| all | 7,180 / 50 / 14 | 170.3 | 190.8 | -0.254<br>d [-0.517, -0.051]<br>c [-0.539, +0.041] | -0.062<br>d [-0.260, +0.080]<br>c [-0.476, +0.148] | -0.096<br>d [-0.322, +0.101]<br>c [-0.267, +0.179] | -0.296<br>d [-0.640, -0.053]<br>c [-0.595, +0.036] |
| class low | 4,686 / 25 / 10 | 169.6 | 201.9 | -0.417<br>d [-0.847, -0.161]<br>c [-0.791, +0.059] | -0.211<br>d [-0.596, -0.022]<br>c [-0.889, +0.142] | -0.165<br>d [-0.471, +0.113]<br>c [-0.421, +0.206] | -0.495<br>d [-1.074, -0.161]<br>c [-0.868, +0.072] |
| class middle | 473 / 4 / 3 | 164.0 | 182.0 | -0.231<br>d [-1.118, +0.165]<br>c [-1.511, +0.165] | -0.284<br>d [-0.473, +0.143]<br>c [-0.508, +0.143] | -0.019<br>d [-0.190, +0.146]<br>c [-0.213, +0.146] | +0.045<br>d [-0.467, +0.351]<br>c [-0.616, +0.401] |
| class high | 2,021 / 21 / 8 | 173.5 | 164.3 | +0.103<br>d [-0.228, +0.322]<br>c [-0.228, +0.462] | +0.258<br>d [+0.117, +0.382]<br>c [+0.096, +0.385] | +0.039<br>d [-0.329, +0.268]<br>c [-0.379, +0.446] | +0.074<br>d [-0.250, +0.287]<br>c [-0.179, +0.386] |
| high minus low (rows of low and high: 6,707 / 46 / 13) | | | | +0.520<br>d [+0.097, +0.981]<br>c [-0.153, +1.104] | +0.470<br>d [+0.221, +0.859]<br>c [+0.028, +1.123] | +0.204<br>d [-0.262, +0.581]<br>c [-0.456, +0.763] | +0.568<br>d [+0.098, +1.174]<br>c [-0.155, +1.157] |

### V1 random (r2-r4), rule D650 (common set)

| stratum | rows / discharges / clusters | RMSE T0 | RMSE TX | TX over T0 | TXN over TN | TB over T0 | TE over T0 |
|---|---|---|---|---|---|---|---|
| all | 160,343 / 562 / 79 | 226.2 | 184.3 | +0.336<br>d [+0.299, +0.371]<br>c [+0.285, +0.380] | +0.258<br>d [+0.222, +0.292]<br>c [+0.204, +0.305] | +0.246<br>d [+0.209, +0.282]<br>c [+0.201, +0.292] | +0.229<br>d [+0.187, +0.267]<br>c [+0.162, +0.291] |
| class low | 50,581 / 162 / 53 | 173.6 | 160.6 | +0.143<br>d [+0.067, +0.212]<br>c [+0.039, +0.232] | +0.119<br>d [+0.040, +0.215]<br>c [+0.013, +0.236] | +0.108<br>d [+0.037, +0.171]<br>c [+0.018, +0.208] | +0.066<br>d [-0.015, +0.140]<br>c [-0.024, +0.146] |
| class middle | 38,895 / 132 / 43 | 179.4 | 148.1 | +0.319<br>d [+0.247, +0.392]<br>c [+0.245, +0.393] | +0.247<br>d [+0.162, +0.343]<br>c [+0.162, +0.344] | +0.254<br>d [+0.192, +0.324]<br>c [+0.191, +0.318] | +0.202<br>d [+0.126, +0.277]<br>c [+0.120, +0.289] |
| class high | 70,394 / 264 / 55 | 277.3 | 215.5 | +0.396<br>d [+0.348, +0.438]<br>c [+0.349, +0.439] | +0.303<br>d [+0.262, +0.341]<br>c [+0.250, +0.350] | +0.284<br>d [+0.235, +0.331]<br>c [+0.229, +0.339] | +0.283<br>d [+0.230, +0.330]<br>c [+0.195, +0.353] |
| no class | 473 / 4 / 1 | 192.7 | 199.4 | -0.073<br>d [-0.433, +0.153]<br>c [-0.243, +0.032] | -0.018<br>d [-0.327, +0.168]<br>c [-0.047, +0.019] | -0.034<br>d [-0.128, +0.054]<br>c [-0.079, +0.003] | -0.221<br>d [-0.847, +0.059]<br>c [-0.503, -0.078] |
| high minus low (rows of low and high: 120,975 / 426 / 75) | | | | +0.253<br>d [+0.168, +0.339]<br>c [+0.154, +0.354] | +0.184<br>d [+0.084, +0.274]<br>c [+0.064, +0.293] | +0.176<br>d [+0.095, +0.260]<br>c [+0.059, +0.282] | +0.217<br>d [+0.126, +0.309]<br>c [+0.095, +0.331] |

### V2 cluster-blocked (3 assignments), rule D650 (common set)

| stratum | rows / discharges / clusters | RMSE T0 | RMSE TX | TX over T0 | TXN over TN | TB over T0 | TE over T0 |
|---|---|---|---|---|---|---|---|
| all | 160,343 / 562 / 79 | 236.7 | 200.6 | +0.282<br>d [+0.244, +0.318]<br>c [+0.214, +0.342] | +0.298<br>d [+0.238, +0.362]<br>c [+0.207, +0.377] | - | - |
| class low | 50,581 / 162 / 53 | 181.3 | 173.7 | +0.083<br>d [-0.020, +0.172]<br>c [-0.125, +0.231] | +0.260<br>d [+0.120, +0.395]<br>c [+0.012, +0.414] | - | - |
| class middle | 38,895 / 132 / 43 | 185.7 | 153.0 | +0.321<br>d [+0.252, +0.392]<br>c [+0.248, +0.391] | +0.290<br>d [+0.137, +0.407]<br>c [+0.111, +0.420] | - | - |
| class high | 70,394 / 264 / 55 | 291.0 | 238.2 | +0.330<br>d [+0.284, +0.372]<br>c [+0.268, +0.396] | +0.312<br>d [+0.251, +0.379]<br>c [+0.213, +0.411] | - | - |
| no class | 473 / 4 / 1 | 197.1 | 204.7 | -0.084<br>d [-0.576, +0.186]<br>c [-0.209, +0.081] | +0.388<br>d [+0.149, +0.503]<br>c [+0.306, +0.435] | - | - |
| high minus low (rows of low and high: 120,975 / 426 / 75) | | | | +0.247<br>d [+0.146, +0.362]<br>c [+0.089, +0.451] | +0.051<br>d [-0.074, +0.197]<br>c [-0.123, +0.309] | - | - |

### V3 rolling origin (fifths 2-5 pooled), rule D650 (common set)

| stratum | rows / discharges / clusters | RMSE T0 | RMSE TX | TX over T0 | TXN over TN | TB over T0 | TE over T0 |
|---|---|---|---|---|---|---|---|
| all | 112,199 / 441 / 60 | 268.5 | 241.7 | +0.190<br>d [+0.132, +0.242]<br>c [+0.032, +0.291] | +0.263<br>d [+0.198, +0.324]<br>c [+0.164, +0.361] | - | - |
| class low | 33,842 / 131 / 45 | 179.5 | 204.2 | -0.293<br>d [-0.508, -0.100]<br>c [-0.753, +0.093] | +0.188<br>d [-0.095, +0.396]<br>c [-0.176, +0.470] | - | - |
| class middle | 30,044 / 103 / 33 | 187.4 | 171.6 | +0.162<br>d [+0.027, +0.290]<br>c [-0.003, +0.302] | +0.369<br>d [+0.213, +0.491]<br>c [+0.158, +0.511] | - | - |
| class high | 47,840 / 203 / 40 | 351.8 | 297.5 | +0.285<br>d [+0.235, +0.334]<br>c [+0.200, +0.361] | +0.240<br>d [+0.179, +0.300]<br>c [+0.156, +0.315] | - | - |
| no class | 473 / 4 / 1 | 206.2 | 219.7 | -0.135<br>d [-0.995, +0.123]<br>c [-0.135, -0.135] | +0.804<br>d [+0.727, +0.857]<br>c [+0.804, +0.804] | - | - |
| high minus low (rows of low and high: 81,682 / 334 / 57) | | | | +0.578<br>d [+0.374, +0.795]<br>c [+0.203, +0.994] | +0.052<br>d [-0.169, +0.337]<br>c [-0.247, +0.431] | - | - |

### V4 chronological: fit to 31991, score from 32312, rule D650 (common set)

| stratum | rows / discharges / clusters | RMSE T0 | RMSE TX | TX over T0 | TXN over TN | TB over T0 | TE over T0 |
|---|---|---|---|---|---|---|---|
| all | 10,146 / 75 / 18 | 202.3 | 176.7 | +0.237<br>d [+0.073, +0.358]<br>c [+0.043, +0.415] | +0.245<br>d [+0.148, +0.342]<br>c [+0.061, +0.357] | -0.014<br>d [-0.295, +0.170]<br>c [-0.263, +0.205] | +0.177<br>d [+0.083, +0.271]<br>c [+0.085, +0.303] |
| class low | 6,100 / 41 / 14 | 194.8 | 159.3 | +0.332<br>d [+0.209, +0.463]<br>c [+0.207, +0.524] | +0.208<br>d [+0.050, +0.374]<br>c [+0.048, +0.384] | +0.137<br>d [-0.079, +0.302]<br>c [-0.037, +0.369] | +0.182<br>d [+0.048, +0.319]<br>c [+0.049, +0.354] |
| class middle | 823 / 8 / 4 | 264.2 | 215.0 | +0.338<br>d [+0.032, +0.547]<br>c [+0.009, +0.545] | +0.258<br>d [-0.275, +0.453]<br>c [-0.251, +0.461] | -0.046<br>d [-0.308, +0.244]<br>c [-0.256, +0.204] | +0.302<br>d [-0.051, +0.532]<br>c [-0.091, +0.552] |
| class high | 3,223 / 26 / 8 | 197.9 | 196.2 | +0.017<br>d [-0.580, +0.309]<br>c [-0.248, +0.394] | +0.284<br>d [+0.167, +0.403]<br>c [+0.058, +0.422] | -0.278<br>d [-1.270, +0.178]<br>c [-0.687, +0.274] | +0.113<br>d [-0.034, +0.200]<br>c [-0.031, +0.214] |
| high minus low (rows of low and high: 9,323 / 67 / 17) | | | | -0.315<br>d [-0.928, +0.008]<br>c [-0.722, +0.097] | +0.076<br>d [-0.125, +0.282]<br>c [-0.133, +0.212] | -0.415<br>d [-1.418, +0.098]<br>c [-1.010, +0.199] | -0.069<br>d [-0.269, +0.091]<br>c [-0.342, +0.101] |

### V4 chronological: fit to 32310, score from 32312, rule D650 (common set)

| stratum | rows / discharges / clusters | RMSE T0 | RMSE TX | TX over T0 | TXN over TN | TB over T0 | TE over T0 |
|---|---|---|---|---|---|---|---|
| all | 10,146 / 75 / 18 | 189.8 | 168.1 | +0.216<br>d [+0.102, +0.306]<br>c [+0.075, +0.360] | +0.275<br>d [+0.177, +0.383]<br>c [+0.151, +0.407] | +0.094<br>d [-0.050, +0.203]<br>c [-0.062, +0.248] | +0.145<br>d [+0.060, +0.233]<br>c [+0.049, +0.295] |
| class low | 6,100 / 41 / 14 | 179.9 | 154.9 | +0.259<br>d [+0.165, +0.366]<br>c [+0.161, +0.408] | +0.266<br>d [+0.105, +0.460]<br>c [+0.078, +0.478] | +0.184<br>d [+0.038, +0.317]<br>c [+0.076, +0.373] | +0.119<br>d [+0.005, +0.238]<br>c [-0.002, +0.317] |
| class middle | 823 / 8 / 4 | 244.4 | 205.4 | +0.294<br>d [-0.003, +0.505]<br>c [-0.050, +0.531] | +0.224<br>d [+0.063, +0.318]<br>c [+0.062, +0.359] | +0.096<br>d [-0.012, +0.199]<br>c [-0.034, +0.185] | +0.260<br>d [-0.056, +0.524]<br>c [-0.103, +0.540] |
| class high | 3,223 / 26 / 8 | 192.2 | 180.9 | +0.114<br>d [-0.257, +0.299]<br>c [-0.052, +0.364] | +0.299<br>d [+0.180, +0.401]<br>c [+0.212, +0.396] | -0.058<br>d [-0.532, +0.193]<br>c [-0.311, +0.293] | +0.140<br>d [-0.009, +0.258]<br>c [-0.050, +0.288] |
| high minus low (rows of low and high: 9,323 / 67 / 17) | | | | -0.145<br>d [-0.529, +0.068]<br>c [-0.394, +0.123] | +0.033<br>d [-0.195, +0.235]<br>c [-0.159, +0.228] | -0.243<br>d [-0.726, +0.053]<br>c [-0.628, +0.172] | +0.021<br>d [-0.170, +0.184]<br>c [-0.283, +0.183] |

## V1 by chronological block of the test discharge and class of C_E

### V1, rule S (S set), block up to 31991

| stratum | rows / discharges / clusters | RMSE T0 | RMSE TX | TX over T0 | TXN over TN | TB over T0 | TE over T0 |
|---|---|---|---|---|---|---|---|
| block up to 31991 | 119,655 / 349 / 44 | 215.5 | 175.2 | +0.339<br>d [+0.283, +0.393]<br>c [+0.262, +0.412] | +0.269<br>d [+0.214, +0.323]<br>c [+0.191, +0.344] | +0.260<br>d [+0.210, +0.307]<br>c [+0.195, +0.323] | +0.229<br>d [+0.168, +0.285]<br>c [+0.148, +0.306] |
| class low | 34,412 / 74 / 26 | 171.0 | 146.6 | +0.265<br>d [+0.107, +0.419]<br>c [+0.121, +0.521] | +0.175<br>d [+0.063, +0.282]<br>c [+0.069, +0.368] | +0.183<br>d [+0.054, +0.297]<br>c [+0.055, +0.368] | +0.214<br>d [+0.042, +0.363]<br>c [+0.051, +0.445] |
| class middle | 29,188 / 88 / 28 | 165.6 | 138.1 | +0.304<br>d [+0.163, +0.421]<br>c [+0.169, +0.427] | +0.246<br>d [+0.125, +0.360]<br>c [+0.117, +0.375] | +0.260<br>d [+0.136, +0.361]<br>c [+0.140, +0.368] | +0.214<br>d [+0.002, +0.364]<br>c [-0.003, +0.367] |
| class high | 56,055 / 187 / 34 | 258.6 | 205.8 | +0.367<br>d [+0.300, +0.428]<br>c [+0.278, +0.438] | +0.299<br>d [+0.233, +0.365]<br>c [+0.210, +0.372] | +0.280<br>d [+0.219, +0.338]<br>c [+0.206, +0.355] | +0.236<br>d [+0.166, +0.301]<br>c [+0.130, +0.320] |
| high minus low (rows of low and high: 90,467 / 261 / 42) | | | | +0.101<br>d [-0.068, +0.272]<br>c [-0.165, +0.260] | +0.124<br>d [+0.003, +0.246]<br>c [-0.078, +0.248] | +0.097<br>d [-0.037, +0.240]<br>c [-0.101, +0.241] | +0.021<br>d [-0.146, +0.206]<br>c [-0.225, +0.198] |

### V1, rule S (S set), block 32002-32310

| stratum | rows / discharges / clusters | RMSE T0 | RMSE TX | TX over T0 | TXN over TN | TB over T0 | TE over T0 |
|---|---|---|---|---|---|---|---|
| block 32002-32310 | 21,002 / 104 / 14 | 241.7 | 199.5 | +0.319<br>d [+0.204, +0.416]<br>c [+0.220, +0.373] | +0.213<br>d [+0.097, +0.310]<br>c [+0.049, +0.329] | +0.206<br>d [+0.076, +0.313]<br>c [+0.121, +0.272] | +0.211<br>d [+0.083, +0.316]<br>c [+0.116, +0.288] |
| class low | 6,668 / 36 / 10 | 208.7 | 196.1 | +0.116<br>d [-0.017, +0.250]<br>c [+0.003, +0.265] | +0.015<br>d [-0.121, +0.107]<br>c [-0.123, +0.094] | +0.085<br>d [-0.032, +0.194]<br>c [-0.032, +0.190] | +0.036<br>d [-0.089, +0.157]<br>c [-0.115, +0.157] |
| class middle | 5,674 / 31 / 9 | 185.5 | 154.5 | +0.303<br>d [+0.089, +0.494]<br>c [+0.129, +0.483] | +0.079<br>d [-0.024, +0.184]<br>c [-0.026, +0.166] | +0.299<br>d [+0.129, +0.462]<br>c [+0.109, +0.461] | +0.202<br>d [-0.023, +0.431]<br>c [+0.006, +0.437] |
| class high | 8,287 / 33 / 9 | 295.0 | 227.5 | +0.405<br>d [+0.242, +0.527]<br>c [+0.305, +0.542] | +0.318<br>d [+0.174, +0.434]<br>c [+0.130, +0.411] | +0.231<br>d [+0.013, +0.387]<br>c [+0.076, +0.449] | +0.280<br>d [+0.080, +0.422]<br>c [+0.164, +0.444] |
| no class | 373 / 4 / 1 | 227.7 | 197.9 | +0.242<br>d [-0.567, +0.537]<br>c [+0.158, +0.354] | -0.086<br>d [-0.490, +0.329]<br>c [-0.326, +0.144] | +0.149<br>d [-0.867, +0.559]<br>c [+0.070, +0.191] | +0.337<br>d [-0.221, +0.573]<br>c [+0.229, +0.420] |
| high minus low (rows of low and high: 14,955 / 69 / 13) | | | | +0.289<br>d [+0.068, +0.481]<br>c [+0.105, +0.482] | +0.303<br>d [+0.142, +0.451]<br>c [+0.108, +0.445] | +0.145<br>d [-0.102, +0.345]<br>c [-0.085, +0.431] | +0.244<br>d [+0.006, +0.443]<br>c [+0.081, +0.459] |

### V1, rule S (S set), block from 32312

| stratum | rows / discharges / clusters | RMSE T0 | RMSE TX | TX over T0 | TXN over TN | TB over T0 | TE over T0 |
|---|---|---|---|---|---|---|---|
| block from 32312 | 7,180 / 50 / 14 | 173.1 | 167.3 | +0.066<br>d [-0.129, +0.220]<br>c [-0.170, +0.248] | +0.086<br>d [-0.080, +0.229]<br>c [-0.143, +0.230] | -0.016<br>d [-0.208, +0.153]<br>c [-0.177, +0.197] | +0.073<br>d [-0.103, +0.227]<br>c [-0.104, +0.237] |
| class low | 4,686 / 25 / 10 | 171.2 | 163.6 | +0.084<br>d [-0.059, +0.276]<br>c [-0.070, +0.343] | +0.046<br>d [-0.152, +0.220]<br>c [-0.201, +0.264] | -0.025<br>d [-0.270, +0.216]<br>c [-0.227, +0.285] | +0.072<br>d [-0.170, +0.267]<br>c [-0.159, +0.331] |
| class middle | 473 / 4 / 3 | 154.2 | 201.9 | -0.752<br>d [-3.070, +0.168]<br>c [-2.754, +0.172] | -0.304<br>d [-1.487, +0.355]<br>c [-1.563, +0.357] | -0.575<br>d [-1.548, +0.139]<br>c [-1.667, +0.048] | -0.098<br>d [-0.498, +0.197]<br>c [-0.476, +0.229] |
| class high | 2,021 / 21 / 8 | 181.2 | 166.5 | +0.154<br>d [-0.194, +0.367]<br>c [-0.197, +0.448] | +0.222<br>d [-0.022, +0.385]<br>c [+0.038, +0.364] | +0.086<br>d [-0.266, +0.329]<br>c [-0.189, +0.398] | +0.091<br>d [-0.358, +0.337]<br>c [-0.528, +0.347] |
| high minus low (rows of low and high: 6,707 / 46 / 13) | | | | +0.069<br>d [-0.359, +0.356]<br>c [-0.381, +0.476] | +0.176<br>d [-0.121, +0.441]<br>c [-0.107, +0.460] | +0.111<br>d [-0.356, +0.479]<br>c [-0.353, +0.576] | +0.019<br>d [-0.524, +0.388]<br>c [-0.673, +0.415] |

### V1, rule S (S set), blocks up to 32310

| stratum | rows / discharges / clusters | RMSE T0 | RMSE TX | TX over T0 | TXN over TN | TB over T0 | TE over T0 |
|---|---|---|---|---|---|---|---|
| blocks up to 32310 | 140,657 / 453 / 58 | 219.6 | 179.0 | +0.335<br>d [+0.283, +0.383]<br>c [+0.271, +0.397] | +0.259<br>d [+0.208, +0.310]<br>c [+0.187, +0.326] | +0.250<br>d [+0.200, +0.294]<br>c [+0.194, +0.303] | +0.226<br>d [+0.171, +0.276]<br>c [+0.158, +0.292] |
| class low | 41,080 / 110 / 36 | 177.7 | 155.7 | +0.232<br>d [+0.108, +0.361]<br>c [+0.116, +0.407] | +0.138<br>d [+0.041, +0.234]<br>c [+0.041, +0.268] | +0.161<br>d [+0.060, +0.258]<br>c [+0.057, +0.293] | +0.174<br>d [+0.036, +0.301]<br>c [+0.039, +0.334] |
| class middle | 34,862 / 119 / 37 | 169.0 | 140.9 | +0.304<br>d [+0.173, +0.408]<br>c [+0.175, +0.410] | +0.220<br>d [+0.114, +0.323]<br>c [+0.105, +0.333] | +0.268<br>d [+0.161, +0.357]<br>c [+0.161, +0.360] | +0.212<br>d [+0.032, +0.339]<br>c [+0.025, +0.336] |
| class high | 64,342 / 220 / 43 | 263.6 | 208.8 | +0.373<br>d [+0.312, +0.427]<br>c [+0.299, +0.436] | +0.302<br>d [+0.241, +0.364]<br>c [+0.225, +0.369] | +0.272<br>d [+0.213, +0.327]<br>c [+0.206, +0.339] | +0.243<br>d [+0.179, +0.303]<br>c [+0.156, +0.319] |
| no class | 373 / 4 / 1 | 227.7 | 197.9 | +0.242<br>d [-0.567, +0.537]<br>c [+0.158, +0.354] | -0.086<br>d [-0.490, +0.329]<br>c [-0.326, +0.144] | +0.149<br>d [-0.867, +0.559]<br>c [+0.070, +0.191] | +0.337<br>d [-0.221, +0.573]<br>c [+0.229, +0.420] |
| high minus low (rows of low and high: 105,422 / 330 / 55) | | | | +0.141<br>d [+0.000, +0.276]<br>c [-0.046, +0.270] | +0.164<br>d [+0.061, +0.270]<br>c [+0.021, +0.272] | +0.111<br>d [-0.006, +0.226]<br>c [-0.033, +0.234] | +0.068<br>d [-0.076, +0.217]<br>c [-0.111, +0.216] |

### V1, rule D650 (common set), block up to 31991

| stratum | rows / discharges / clusters | RMSE T0 | RMSE TX | TX over T0 | TXN over TN | TB over T0 | TE over T0 |
|---|---|---|---|---|---|---|---|
| block up to 31991 | 127,270 / 375 / 48 | 218.8 | 178.8 | +0.332<br>d [+0.291, +0.370]<br>c [+0.271, +0.383] | +0.246<br>d [+0.207, +0.280]<br>c [+0.191, +0.294] | +0.242<br>d [+0.203, +0.278]<br>c [+0.194, +0.291] | +0.211<br>d [+0.162, +0.255]<br>c [+0.126, +0.289] |
| class low | 36,670 / 84 / 29 | 165.9 | 153.6 | +0.143<br>d [+0.030, +0.238]<br>c [-0.009, +0.281] | +0.106<br>d [+0.011, +0.188]<br>c [-0.035, +0.221] | +0.098<br>d [-0.002, +0.184]<br>c [-0.027, +0.269] | +0.073<br>d [-0.038, +0.171]<br>c [-0.063, +0.181] |
| class middle | 32,232 / 92 / 31 | 171.2 | 144.5 | +0.287<br>d [+0.217, +0.356]<br>c [+0.214, +0.350] | +0.231<br>d [+0.144, +0.326]<br>c [+0.143, +0.323] | +0.225<br>d [+0.166, +0.285]<br>c [+0.157, +0.274] | +0.162<br>d [+0.105, +0.215]<br>c [+0.097, +0.218] |
| class high | 58,368 / 199 / 38 | 266.3 | 208.3 | +0.388<br>d [+0.337, +0.434]<br>c [+0.338, +0.433] | +0.285<br>d [+0.239, +0.325]<br>c [+0.234, +0.327] | +0.281<br>d [+0.231, +0.327]<br>c [+0.234, +0.341] | +0.256<br>d [+0.190, +0.310]<br>c [+0.138, +0.342] |
| high minus low (rows of low and high: 95,038 / 283 / 45) | | | | +0.245<br>d [+0.129, +0.371]<br>c [+0.105, +0.399] | +0.179<br>d [+0.086, +0.278]<br>c [+0.061, +0.318] | +0.182<br>d [+0.083, +0.292]<br>c [+0.009, +0.320] | +0.183<br>d [+0.064, +0.306]<br>c [+0.023, +0.344] |

### V1, rule D650 (common set), block 32002-32310

| stratum | rows / discharges / clusters | RMSE T0 | RMSE TX | TX over T0 | TXN over TN | TB over T0 | TE over T0 |
|---|---|---|---|---|---|---|---|
| block 32002-32310 | 22,927 / 112 / 14 | 277.5 | 217.3 | +0.386<br>d [+0.291, +0.463]<br>c [+0.291, +0.465] | +0.301<br>d [+0.213, +0.381]<br>c [+0.122, +0.401] | +0.298<br>d [+0.200, +0.387]<br>c [+0.192, +0.396] | +0.303<br>d [+0.202, +0.384]<br>c [+0.183, +0.395] |
| class low | 7,811 / 37 / 10 | 206.2 | 195.2 | +0.103<br>d [-0.053, +0.226]<br>c [-0.104, +0.250] | +0.046<br>d [-0.090, +0.142]<br>c [-0.154, +0.145] | +0.104<br>d [+0.021, +0.195]<br>c [-0.010, +0.194] | -0.003<br>d [-0.173, +0.141]<br>c [-0.160, +0.141] |
| class middle | 5,840 / 32 / 9 | 210.7 | 157.8 | +0.439<br>d [+0.217, +0.608]<br>c [+0.214, +0.683] | +0.286<br>d [+0.081, +0.613]<br>c [+0.112, +0.651] | +0.387<br>d [+0.194, +0.557]<br>c [+0.202, +0.618] | +0.335<br>d [-0.000, +0.557]<br>c [+0.037, +0.650] |
| class high | 8,803 / 39 / 9 | 362.4 | 265.5 | +0.463<br>d [+0.363, +0.539]<br>c [+0.412, +0.534] | +0.387<br>d [+0.299, +0.465]<br>c [+0.192, +0.459] | +0.339<br>d [+0.208, +0.446]<br>c [+0.175, +0.422] | +0.391<br>d [+0.287, +0.472]<br>c [+0.283, +0.492] |
| no class | 473 / 4 / 1 | 192.7 | 199.4 | -0.073<br>d [-0.433, +0.145]<br>c [-0.243, +0.032] | -0.018<br>d [-0.327, +0.188]<br>c [-0.047, +0.019] | -0.034<br>d [-0.128, +0.054]<br>c [-0.079, +0.003] | -0.221<br>d [-0.847, +0.055]<br>c [-0.503, -0.078] |
| high minus low (rows of low and high: 16,614 / 76 / 13) | | | | +0.360<br>d [+0.207, +0.518]<br>c [+0.190, +0.558] | +0.342<br>d [+0.210, +0.485]<br>c [+0.169, +0.545] | +0.235<br>d [+0.075, +0.372]<br>c [+0.004, +0.421] | +0.395<br>d [+0.213, +0.592]<br>c [+0.216, +0.573] |

### V1, rule D650 (common set), block from 32312

| stratum | rows / discharges / clusters | RMSE T0 | RMSE TX | TX over T0 | TXN over TN | TB over T0 | TE over T0 |
|---|---|---|---|---|---|---|---|
| block from 32312 | 10,146 / 75 / 18 | 184.8 | 169.7 | +0.157<br>d [-0.035, +0.281]<br>c [-0.021, +0.329] | +0.252<br>d [+0.010, +0.466]<br>c [+0.003, +0.507] | +0.052<br>d [-0.148, +0.187]<br>c [-0.147, +0.233] | +0.172<br>d [+0.094, +0.254]<br>c [+0.086, +0.313] |
| class low | 6,100 / 41 / 14 | 172.9 | 152.9 | +0.218<br>d [+0.110, +0.343]<br>c [+0.100, +0.408] | +0.290<br>d [+0.012, +0.599]<br>c [+0.006, +0.620] | +0.166<br>d [+0.046, +0.297]<br>c [+0.049, +0.356] | +0.153<br>d [+0.064, +0.268]<br>c [+0.047, +0.363] |
| class middle | 823 / 8 / 4 | 241.6 | 203.6 | +0.290<br>d [+0.018, +0.486]<br>c [-0.009, +0.507] | +0.269<br>d [-0.038, +0.596]<br>c [-0.025, +0.621] | +0.124<br>d [-0.009, +0.248]<br>c [-0.030, +0.233] | +0.264<br>d [-0.039, +0.520]<br>c [-0.073, +0.550] |
| class high | 3,223 / 26 / 8 | 189.6 | 189.1 | +0.004<br>d [-0.636, +0.279]<br>c [-0.260, +0.309] | +0.182<br>d [-0.294, +0.432]<br>c [-0.077, +0.475] | -0.157<br>d [-0.856, +0.171]<br>c [-0.476, +0.264] | +0.162<br>d [+0.022, +0.274]<br>c [-0.039, +0.294] |
| high minus low (rows of low and high: 9,323 / 67 / 17) | | | | -0.214<br>d [-0.883, +0.099]<br>c [-0.558, +0.123] | -0.108<br>d [-0.620, +0.308]<br>c [-0.413, +0.305] | -0.323<br>d [-1.060, +0.044]<br>c [-0.765, +0.157] | +0.009<br>d [-0.171, +0.155]<br>c [-0.293, +0.177] |

### V1, rule D650 (common set), blocks up to 32310

| stratum | rows / discharges / clusters | RMSE T0 | RMSE TX | TX over T0 | TXN over TN | TB over T0 | TE over T0 |
|---|---|---|---|---|---|---|---|
| blocks up to 32310 | 150,197 / 487 / 62 | 228.7 | 185.2 | +0.344<br>d [+0.305, +0.381]<br>c [+0.293, +0.390] | +0.258<br>d [+0.222, +0.291]<br>c [+0.204, +0.305] | +0.254<br>d [+0.217, +0.291]<br>c [+0.208, +0.302] | +0.232<br>d [+0.188, +0.272]<br>c [+0.162, +0.297] |
| class low | 44,481 / 121 / 39 | 173.7 | 161.7 | +0.133<br>d [+0.047, +0.210]<br>c [+0.014, +0.232] | +0.090<br>d [+0.017, +0.156]<br>c [-0.015, +0.171] | +0.100<br>d [+0.020, +0.168]<br>c [+0.001, +0.211] | +0.054<br>d [-0.038, +0.136]<br>c [-0.053, +0.141] |
| class middle | 38,072 / 124 / 40 | 177.8 | 146.7 | +0.320<br>d [+0.245, +0.394]<br>c [+0.246, +0.396] | +0.244<br>d [+0.153, +0.349]<br>c [+0.155, +0.352] | +0.260<br>d [+0.195, +0.330]<br>c [+0.194, +0.326] | +0.199<br>d [+0.122, +0.276]<br>c [+0.118, +0.285] |
| class high | 67,171 / 238 / 47 | 280.8 | 216.7 | +0.405<br>d [+0.357, +0.447]<br>c [+0.360, +0.446] | +0.307<br>d [+0.267, +0.346]<br>c [+0.255, +0.354] | +0.293<br>d [+0.244, +0.341]<br>c [+0.242, +0.349] | +0.286<br>d [+0.232, +0.333]<br>c [+0.195, +0.355] |
| no class | 473 / 4 / 1 | 192.7 | 199.4 | -0.073<br>d [-0.433, +0.145]<br>c [-0.243, +0.032] | -0.018<br>d [-0.327, +0.168]<br>c [-0.047, +0.019] | -0.034<br>d [-0.128, +0.054]<br>c [-0.079, +0.003] | -0.221<br>d [-0.847, +0.055]<br>c [-0.503, -0.078] |
| high minus low (rows of low and high: 111,652 / 359 / 58) | | | | +0.271<br>d [+0.178, +0.368]<br>c [+0.165, +0.392] | +0.217<br>d [+0.140, +0.302]<br>c [+0.124, +0.329] | +0.194<br>d [+0.110, +0.283]<br>c [+0.069, +0.309] | +0.231<br>d [+0.133, +0.335]<br>c [+0.103, +0.364] |

## The gated estimator TG (TX where C_E is 0.7 or above, T0 otherwise)

| design | rule | stratum | rows / discharges / clusters | rows gated to TX (middle and high class) | RMSE T0 / TX / TG | TG over T0 | TG over TX |
|---|---|---|---|---|---|---|---|
| V1 random (r2-r4) | S | all | 147,837 / 503 / 71 | 101,698 | 217.6 / 178.5 / 184.4 | +0.282<br>d [+0.231, +0.330]<br>c [+0.210, +0.350] | -0.067<br>d [-0.121, -0.029]<br>c [-0.128, -0.027] |
| V1 random (r2-r4) | S | block up to 31991 | 119,655 / 349 / 44 | 85,243 | 215.5 / 175.2 / 181.4 | +0.291<br>d [+0.235, +0.345]<br>c [+0.205, +0.373] | -0.073<br>d [-0.141, -0.024]<br>c [-0.153, -0.023] |
| V1 random (r2-r4) | S | block 32002-32310 | 21,002 / 104 / 14 | 13,961 | 241.7 / 199.5 / 204.0 | +0.287<br>d [+0.171, +0.389]<br>c [+0.169, +0.339] | -0.047<br>d [-0.104, +0.003]<br>c [-0.122, -0.001] |
| V1 random (r2-r4) | S | block from 32312 | 7,180 / 50 / 14 | 2,494 | 173.1 / 167.3 / 172.2 | +0.010<br>d [-0.148, +0.132]<br>c [-0.198, +0.112] | -0.061<br>d [-0.230, +0.034]<br>c [-0.279, +0.046] |
| V1 random (r2-r4) | S | blocks up to 32310 | 140,657 / 453 / 58 | 99,204 | 219.6 / 179.0 / 185.0 | +0.290<br>d [+0.240, +0.338]<br>c [+0.218, +0.359] | -0.068<br>d [-0.124, -0.028]<br>c [-0.129, -0.026] |
| V2 cluster-blocked (3 assignments) | S | all | 147,837 / 503 / 71 | 101,698 | 223.2 / 187.7 / 193.5 | +0.249<br>d [+0.197, +0.300]<br>c [+0.173, +0.324] | -0.062<br>d [-0.127, -0.022]<br>c [-0.135, -0.015] |
| V3 rolling origin (fifths 2-5 pooled) | S | all | 102,684 / 395 / 56 | 71,842 | 249.1 / 218.5 / 218.0 | +0.234<br>d [+0.178, +0.287]<br>c [+0.167, +0.309] | +0.004<br>d [-0.035, +0.045]<br>c [-0.045, +0.046] |
| V4 chronological: fit to 31991, score from 32312 | S | all | 7,180 / 50 / 14 | 2,494 | 184.8 / 182.3 / 187.2 | -0.025<br>d [-0.132, +0.046]<br>c [-0.163, +0.052] | -0.054<br>d [-0.189, +0.053]<br>c [-0.203, +0.060] |
| V4 chronological: fit to 32310, score from 32312 | S | all | 7,180 / 50 / 14 | 2,494 | 170.3 / 190.8 / 169.0 | +0.016<br>d [-0.083, +0.110]<br>c [-0.112, +0.096] | +0.215<br>d [+0.094, +0.337]<br>c [-0.035, +0.365] |
| V1 random (r2-r4) | D650 | all | 160,343 / 562 / 79 | 109,289 | 226.2 / 184.3 / 187.9 | +0.310<br>d [+0.271, +0.346]<br>c [+0.254, +0.359] | -0.040<br>d [-0.065, -0.017]<br>c [-0.079, -0.009] |
| V1 random (r2-r4) | D650 | block up to 31991 | 127,270 / 375 / 48 | 90,600 | 218.8 / 178.8 / 182.0 | +0.308<br>d [+0.265, +0.348]<br>c [+0.243, +0.367] | -0.035<br>d [-0.067, -0.007]<br>c [-0.085, +0.002] |
| V1 random (r2-r4) | D650 | block 32002-32310 | 22,927 / 112 / 14 | 14,643 | 277.5 / 217.3 / 220.6 | +0.368<br>d [+0.275, +0.444]<br>c [+0.262, +0.447] | -0.031<br>d [-0.077, +0.019]<br>c [-0.113, +0.027] |
| V1 random (r2-r4) | D650 | block from 32312 | 10,146 / 75 / 18 | 4,046 | 184.8 / 169.7 / 180.9 | +0.042<br>d [-0.151, +0.162]<br>c [-0.086, +0.154] | -0.136<br>d [-0.238, -0.058]<br>c [-0.345, -0.034] |
| V1 random (r2-r4) | D650 | blocks up to 32310 | 150,197 / 487 / 62 | 105,243 | 228.7 / 185.2 / 188.4 | +0.322<br>d [+0.282, +0.359]<br>c [+0.265, +0.372] | -0.034<br>d [-0.060, -0.011]<br>c [-0.074, -0.002] |
| V2 cluster-blocked (3 assignments) | D650 | all | 160,343 / 562 / 79 | 109,289 | 236.7 / 200.6 / 202.7 | +0.267<br>d [+0.231, +0.301]<br>c [+0.213, +0.320] | -0.021<br>d [-0.047, +0.005]<br>c [-0.070, +0.029] |
| V3 rolling origin (fifths 2-5 pooled) | D650 | all | 112,199 / 441 / 60 | 77,884 | 268.5 / 241.7 / 235.6 | +0.230<br>d [+0.186, +0.271]<br>c [+0.151, +0.294] | +0.049<br>d [+0.017, +0.085]<br>c [-0.016, +0.136] |
| V4 chronological: fit to 31991, score from 32312 | D650 | all | 10,146 / 75 / 18 | 4,046 | 202.3 / 176.7 / 197.0 | +0.052<br>d [-0.101, +0.173]<br>c [-0.071, +0.162] | -0.242<br>d [-0.400, -0.118]<br>c [-0.555, -0.082] |
| V4 chronological: fit to 32310, score from 32312 | D650 | all | 10,146 / 75 / 18 | 4,046 | 189.8 / 168.1 / 182.4 | +0.076<br>d [-0.037, +0.172]<br>c [-0.003, +0.166] | -0.178<br>d [-0.278, -0.095]<br>c [-0.380, -0.060] |

## Spearman correlation between C_E and the per-discharge skill of TX over T0 (V1, rule S)

Per-discharge skill = 1 - SSE(TX) / SSE(T0) over the discharge's S-set rows, per re-partition; discharges with C_E and at least 50 S-set rows. Point = mean over the three re-partitions of the Spearman correlation; intervals: a re-partition drawn first, then discharges (d) or shot clusters (c) resampled. Registered: all blocks; the rows by block are added (Addendum 1 section 4).

Discharges with C_E but fewer than 50 S-set rows: 55; with at least 50 S-set rows but no C_E: 3.

| discharges of | discharges / clusters | Spearman (point) | per re-partition (r2, r3, r4) | d interval | c interval | median per-discharge skill |
|---|---|---|---|---|---|---|
| all blocks | 444 / 65 | +0.183 | +0.219, +0.171, +0.158 | [+0.078, +0.288] | [+0.071, +0.283] | +0.228 |
| block up to 31991 | 327 / 44 | +0.189 | +0.231, +0.173, +0.162 | [+0.067, +0.314] | [+0.067, +0.295] | +0.235 |
| block 32002-32310 | 80 / 12 | +0.249 | +0.326, +0.200, +0.223 | [+0.000, +0.475] | [+0.048, +0.463] | +0.261 |
| block from 32312 | 37 / 10 | -0.184 | -0.235, -0.065, -0.250 | [-0.546, +0.214] | [-0.625, +0.406] | -0.034 |

Readout run time 96 s.

## Limits (registration section 7, unchanged)

- Exploratory on used data; the measure was chosen after the block comparison had been seen.
- C_E is computed over the whole discharge. No field, ECEI setting or channel radius is in the files, so what lowers the coherence is not decided here.
- The cuts 0.7 and 0.9 are fixed by hand before the distribution was seen; the tables by class show what other cuts would give only coarsely.
- TB and TE exist for V1 and V4 only; no new fit is made, so there is no estimator refitted within a class.
