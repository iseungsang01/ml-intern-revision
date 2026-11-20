# B.78 — a CES value every k-th frame: the level in between, with and without the fast diagnostics

Registered: `ces_prediction/experiments/PREREGISTRATION_B78.md`; implementation choices in `ces_prediction/experiments/PREREGISTRATION_B78_ADDENDUM1.md`, written before any readout. CPU only; no network trained or run. T_i, cut population, the D1 rows of B.70 as cached by B.73 (222,672 rows, 635 discharges). Exploratory on used data (registration section 1): rows, targets, designs, learners and the fast and anchor inputs are those of B.73 and B.75; the rate rules are new.

**The rate is lowered after the fact.** Under Kk only the measured values whose frame index is a multiple of k are visible; the withheld frames were taken with the beam on, and the targets are built from all measured values.

Scored rows: the D1 rows whose frame index is odd and not a multiple of 5 (88,749 rows); they lie on no kept grid for k = 2, 5, 10, 20, 50, so the same rows are scored under every rule.

Conventions. Skill of A over B = 1 - SSE(A) / SSE(B) on identical rows. Increment = TX over T0 under the same rule on the same rows. Each comparison: point, then 95 % intervals with discharges resampled (d) and with shot clusters resampled (c), 20,000 draws (V1: a re-partition drawn first; V2: an assignment drawn first; V3, V4: the scored units). Verdicts on the cluster form. Points under V1 and V2 = mean over the three re-partitions or assignments. Every fit and score uses the common set (scored rows complete under all six rules, target defined). RMSE in eV. Checks: `docs/paper/b78_checks.md`.

Order of work. Before this readout, these results existed: the checks of registration section 10 and the counts of the row sets. No MSE, skill or statement under any rule had been computed.

## Statements (target L5, cluster interval)

| Id | statement | value: point; d interval; c interval | verdict | registered expectation | expectation met | note |
|---|---|---|---|---|---|---|
| K-1 | V1, K10: the increment lies above zero | +0.191; d [+0.152, +0.227]; c [+0.133, +0.247] | **holds** | holds (+0.10 to +0.30) | yes; point in the stated range: yes |  |
| K-2 | V2, K10: the increment lies above zero | +0.163; d [+0.130, +0.196]; c [+0.095, +0.231] | **holds** | holds | yes |  |
| K-3 | V3, K10: the increment lies above zero | +0.075; d [+0.031, +0.113]; c [+0.012, +0.123] | **holds** | uncertain, leaning to holds | n/a (uncertain) |  |
| K-4 | V4 (fit to 31991), K10: the increment lies above zero | +0.023; d [-0.118, +0.139]; c [-0.077, +0.120] | **does not hold** | uncertain | n/a (uncertain) | V4b (fit to 32310): +0.105; c [+0.018, +0.201] |
| K-5 | V1: the MSE of TX under K10 lies below that of T0 under K5 (interval of 1 - MSE(TX K10)/MSE(T0 K5) above zero) | +0.015; d [-0.038, +0.064]; c [-0.046, +0.078] | **does not hold** | uncertain | n/a (uncertain) | RMSE TX K10 144.6 eV, T0 K5 145.6 eV |
| K-6 | V1: MSE(TX under K10) / MSE(TX under K1) lies below 2 | 1.430; d [1.361, 1.507]; c [1.332, 1.530] | **holds** | holds (1.3 to 2.0) | yes; point in the stated range: yes |  |
| K-7 | V1, K10: TE over T0 lies above zero | +0.107; d [+0.067, +0.143]; c [+0.052, +0.161] | **holds** | holds | yes |  |
| K-8 | V1, K10, placebo: the interval of TX over T0 contains zero or lies below it | -0.003; d [-0.014, +0.009]; c [-0.015, +0.009] | **holds** | holds | yes |  |

K-5 is the skill of TX under K10 over T0 under K5; K-6 is a ratio of MSEs (judged on its upper end below 2); K-8 holds when the lower end of the placebo's interval is at most zero. 'V4' without a qualifier is V4a; V4b is in the note.

## RMSE by rate rule

H newest visible value; M5 median of the last five visible values; G causal GP of the last up to 16 visible values; T0 G + boosted correction on the ten history scalars; TX with the eight fast scalars of row t and the six anchor scalars added; TB, TE the BES and the ECEI half (V1, V4; K1, K10); O the frame's own reading (against L5 only). Points; under V1 and V2 the mean over the three splits.

### V1 random (r2-r4), target L5 (60,523 rows / 507 discharges / 72 clusters)

| rule | H | M5 | G | T0 | TX | TB | TE | O |
|---|---|---|---|---|---|---|---|---|
| K1 (every frame) | 183.2 | 147.8 | 136.4 | 127.0 | 120.9 | 122.6 | 124.4 | 177.1 |
| K2 (20 ms) | 182.5 | 154.7 | 145.2 | 130.2 | 123.8 | - | - | 177.1 |
| K5 (50 ms) | 198.6 | 174.0 | 160.2 | 145.6 | 135.4 | - | - | 177.1 |
| K10 (100 ms) | 217.6 | 200.2 | 185.4 | 160.7 | 144.6 | 148.8 | 151.8 | 177.1 |
| K20 (200 ms) | 234.4 | 228.2 | 213.7 | 179.7 | 159.9 | - | - | 177.1 |
| K50 (500 ms) | 272.4 | 284.2 | 266.3 | 212.2 | 180.1 | - | - | 177.1 |

### V2 cluster-blocked (3 assignments), target L5 (60,523 rows / 507 discharges / 72 clusters)

| rule | H | M5 | G | T0 | TX | TB | TE | O |
|---|---|---|---|---|---|---|---|---|
| K1 (every frame) | 183.2 | 147.8 | 136.4 | 128.6 | 125.5 | - | - | 177.1 |
| K2 (20 ms) | 182.5 | 154.7 | 145.2 | 132.9 | 129.5 | - | - | 177.1 |
| K5 (50 ms) | 198.6 | 174.0 | 160.2 | 151.2 | 142.5 | - | - | 177.1 |
| K10 (100 ms) | 217.6 | 200.2 | 185.4 | 169.9 | 155.5 | - | - | 177.1 |
| K20 (200 ms) | 234.4 | 228.2 | 213.7 | 188.7 | 171.4 | - | - | 177.1 |
| K50 (500 ms) | 272.4 | 284.2 | 266.3 | 226.4 | 196.6 | - | - | 177.1 |

### V3 rolling origin (fifths 2-5 pooled), target L5 (41,985 rows / 397 discharges / 57 clusters)

| rule | H | M5 | G | T0 | TX | TB | TE | O |
|---|---|---|---|---|---|---|---|---|
| K1 (every frame) | 204.3 | 163.0 | 151.4 | 144.1 | 143.8 | - | - | 198.7 |
| K2 (20 ms) | 203.3 | 171.4 | 161.1 | 150.8 | 148.3 | - | - | 198.7 |
| K5 (50 ms) | 222.5 | 194.1 | 179.0 | 172.4 | 167.3 | - | - | 198.7 |
| K10 (100 ms) | 243.3 | 224.8 | 207.6 | 199.6 | 192.0 | - | - | 198.7 |
| K20 (200 ms) | 264.1 | 255.1 | 237.7 | 216.8 | 207.3 | - | - | 198.7 |
| K50 (500 ms) | 304.0 | 308.9 | 290.4 | 264.8 | 231.5 | - | - | 198.7 |

### V4 chronological: fit to 31991, score from 32312, target L5 (3,587 rows / 59 discharges / 16 clusters)

| rule | H | M5 | G | T0 | TX | TB | TE | O |
|---|---|---|---|---|---|---|---|---|
| K1 (every frame) | 237.3 | 147.2 | 161.8 | 141.2 | 138.7 | 138.9 | 139.9 | 243.9 |
| K2 (20 ms) | 235.6 | 144.4 | 167.8 | 146.3 | 141.3 | - | - | 243.9 |
| K5 (50 ms) | 251.3 | 144.6 | 173.1 | 150.0 | 143.2 | - | - | 243.9 |
| K10 (100 ms) | 235.2 | 148.3 | 176.5 | 159.4 | 157.5 | 169.8 | 153.3 | 243.9 |
| K20 (200 ms) | 221.4 | 148.9 | 172.1 | 167.1 | 166.4 | - | - | 243.9 |
| K50 (500 ms) | 302.5 | 173.1 | 203.4 | 209.9 | 176.3 | - | - | 243.9 |

### V4 chronological: fit to 32310, score from 32312, target L5 (3,587 rows / 59 discharges / 16 clusters)

| rule | H | M5 | G | T0 | TX | TB | TE | O |
|---|---|---|---|---|---|---|---|---|
| K1 (every frame) | 237.3 | 147.2 | 161.8 | 137.5 | 132.7 | 135.9 | 133.1 | 243.9 |
| K2 (20 ms) | 235.6 | 144.4 | 167.8 | 135.0 | 132.1 | - | - | 243.9 |
| K5 (50 ms) | 251.3 | 144.6 | 173.1 | 146.4 | 139.5 | - | - | 243.9 |
| K10 (100 ms) | 235.2 | 148.3 | 176.5 | 153.9 | 145.6 | 150.3 | 148.1 | 243.9 |
| K20 (200 ms) | 221.4 | 148.9 | 172.1 | 162.1 | 158.4 | - | - | 243.9 |
| K50 (500 ms) | 302.5 | 173.1 | 203.4 | 199.4 | 170.0 | - | - | 243.9 |

### V1 random (r2-r4), target F (62,162 rows / 512 discharges / 72 clusters)

| rule | H | M5 | G | T0 | TX | TB | TE |
|---|---|---|---|---|---|---|---|
| K1 (every frame) | 215.0 | 189.8 | 179.6 | 171.1 | 163.3 | 165.4 | 166.9 |
| K2 (20 ms) | 214.9 | 200.5 | 185.4 | 175.7 | 167.1 | - | - |
| K5 (50 ms) | 230.3 | 220.1 | 201.4 | 191.8 | 179.0 | - | - |
| K10 (100 ms) | 252.8 | 245.1 | 224.3 | 207.8 | 188.1 | 193.1 | 196.3 |
| K20 (200 ms) | 271.5 | 271.7 | 252.8 | 226.2 | 202.7 | - | - |
| K50 (500 ms) | 306.6 | 320.7 | 303.9 | 255.3 | 221.9 | - | - |

### V2 cluster-blocked (3 assignments), target F (62,162 rows / 512 discharges / 72 clusters)

| rule | H | M5 | G | T0 | TX | TB | TE |
|---|---|---|---|---|---|---|---|
| K1 (every frame) | 215.0 | 189.8 | 179.6 | 171.8 | 165.8 | - | - |
| K2 (20 ms) | 214.9 | 200.5 | 185.4 | 176.8 | 169.8 | - | - |
| K5 (50 ms) | 230.3 | 220.1 | 201.4 | 194.3 | 184.2 | - | - |
| K10 (100 ms) | 252.8 | 245.1 | 224.3 | 213.8 | 196.5 | - | - |
| K20 (200 ms) | 271.5 | 271.7 | 252.8 | 232.9 | 214.1 | - | - |
| K50 (500 ms) | 306.6 | 320.7 | 303.9 | 264.6 | 234.9 | - | - |

### V3 rolling origin (fifths 2-5 pooled), target F (43,352 rows / 401 discharges / 57 clusters)

| rule | H | M5 | G | T0 | TX | TB | TE |
|---|---|---|---|---|---|---|---|
| K1 (every frame) | 242.8 | 213.5 | 202.8 | 196.3 | 191.0 | - | - |
| K2 (20 ms) | 242.7 | 224.9 | 208.5 | 202.8 | 195.8 | - | - |
| K5 (50 ms) | 260.7 | 247.2 | 227.1 | 224.2 | 213.1 | - | - |
| K10 (100 ms) | 284.4 | 276.0 | 252.2 | 247.4 | 236.3 | - | - |
| K20 (200 ms) | 306.9 | 305.1 | 282.3 | 267.1 | 248.9 | - | - |
| K50 (500 ms) | 342.9 | 351.9 | 334.3 | 308.6 | 276.8 | - | - |

### V4 chronological: fit to 31991, score from 32312, target F (3,738 rows / 60 discharges / 16 clusters)

| rule | H | M5 | G | T0 | TX | TB | TE |
|---|---|---|---|---|---|---|---|
| K1 (every frame) | 305.0 | 254.6 | 261.0 | 250.9 | 248.4 | 248.2 | 249.2 |
| K2 (20 ms) | 307.1 | 259.5 | 263.5 | 251.0 | 248.9 | - | - |
| K5 (50 ms) | 319.1 | 266.8 | 269.9 | 262.9 | 255.4 | - | - |
| K10 (100 ms) | 313.5 | 272.8 | 272.2 | 268.7 | 259.2 | 266.8 | 261.3 |
| K20 (200 ms) | 310.8 | 272.0 | 273.4 | 273.0 | 270.6 | - | - |
| K50 (500 ms) | 376.9 | 290.4 | 303.3 | 311.0 | 285.5 | - | - |

### V4 chronological: fit to 32310, score from 32312, target F (3,738 rows / 60 discharges / 16 clusters)

| rule | H | M5 | G | T0 | TX | TB | TE |
|---|---|---|---|---|---|---|---|
| K1 (every frame) | 305.0 | 254.6 | 261.0 | 249.2 | 241.9 | 243.9 | 247.2 |
| K2 (20 ms) | 307.1 | 259.5 | 263.5 | 247.4 | 239.0 | - | - |
| K5 (50 ms) | 319.1 | 266.8 | 269.9 | 259.4 | 249.4 | - | - |
| K10 (100 ms) | 313.5 | 272.8 | 272.2 | 265.9 | 260.5 | 262.1 | 262.7 |
| K20 (200 ms) | 310.8 | 272.0 | 273.4 | 270.3 | 266.4 | - | - |
| K50 (500 ms) | 376.9 | 290.4 | 303.3 | 297.3 | 284.6 | - | - |

## The fast-channel increment TX over T0 by rule and design, target L5

Cells: skill, d interval, c interval.

| design | K1 (every frame) | K2 (20 ms) | K5 (50 ms) | K10 (100 ms) | K20 (200 ms) | K50 (500 ms) |
|---|---|---|---|---|---|---|
| V1 random (r2-r4) | +0.093<br>d [+0.070, +0.116]<br>c [+0.062, +0.131] | +0.096<br>d [+0.073, +0.119]<br>c [+0.068, +0.129] | +0.136<br>d [+0.097, +0.175]<br>c [+0.092, +0.186] | +0.191<br>d [+0.152, +0.227]<br>c [+0.133, +0.247] | +0.208<br>d [+0.159, +0.257]<br>c [+0.156, +0.259] | +0.279<br>d [+0.226, +0.327]<br>c [+0.214, +0.342] |
| V2 cluster-blocked (3 assignments) | +0.049<br>d [+0.013, +0.076]<br>c [+0.005, +0.087] | +0.051<br>d [+0.028, +0.074]<br>c [+0.017, +0.084] | +0.112<br>d [+0.080, +0.143]<br>c [+0.074, +0.151] | +0.163<br>d [+0.130, +0.196]<br>c [+0.095, +0.231] | +0.175<br>d [+0.117, +0.226]<br>c [+0.114, +0.233] | +0.245<br>d [+0.169, +0.307]<br>c [+0.149, +0.322] |
| V3 rolling origin (fifths 2-5 pooled) | +0.003<br>d [-0.026, +0.030]<br>c [-0.042, +0.046] | +0.033<br>d [+0.006, +0.057]<br>c [-0.021, +0.075] | +0.058<br>d [+0.030, +0.087]<br>c [+0.012, +0.116] | +0.075<br>d [+0.031, +0.113]<br>c [+0.012, +0.123] | +0.085<br>d [+0.047, +0.123]<br>c [+0.027, +0.154] | +0.236<br>d [+0.172, +0.304]<br>c [+0.146, +0.351] |
| V4 chronological: fit to 31991, score from 32312 | +0.035<br>d [-0.030, +0.096]<br>c [-0.009, +0.120] | +0.066<br>d [-0.034, +0.157]<br>c [-0.048, +0.244] | +0.088<br>d [-0.019, +0.162]<br>c [+0.027, +0.168] | +0.023<br>d [-0.118, +0.139]<br>c [-0.077, +0.120] | +0.008<br>d [-0.117, +0.113]<br>c [-0.083, +0.175] | +0.294<br>d [+0.022, +0.466]<br>c [+0.076, +0.442] |
| V4 chronological: fit to 32310, score from 32312 | +0.069<br>d [+0.007, +0.127]<br>c [+0.021, +0.149] | +0.042<br>d [-0.051, +0.138]<br>c [-0.039, +0.184] | +0.092<br>d [+0.015, +0.158]<br>c [+0.024, +0.190] | +0.105<br>d [+0.027, +0.187]<br>c [+0.018, +0.201] | +0.045<br>d [-0.061, +0.143]<br>c [-0.037, +0.196] | +0.273<br>d [+0.110, +0.394]<br>c [+0.128, +0.415] |

### TB and TE over T0 (V1, V4), target L5

| design | rule | TX over T0 | TB over T0 (BES half) | TE over T0 (ECEI half) |
|---|---|---|---|---|
| V1 random (r2-r4) | K1 (every frame) | +0.093<br>d [+0.070, +0.116]<br>c [+0.062, +0.131] | +0.068<br>d [+0.047, +0.087]<br>c [+0.038, +0.105] | +0.041<br>d [+0.025, +0.056]<br>c [+0.025, +0.055] |
| V1 random (r2-r4) | K10 (100 ms) | +0.191<br>d [+0.152, +0.227]<br>c [+0.133, +0.247] | +0.142<br>d [+0.114, +0.171]<br>c [+0.109, +0.180] | +0.107<br>d [+0.067, +0.143]<br>c [+0.052, +0.161] |
| V4 chronological: fit to 31991, score from 32312 | K1 (every frame) | +0.035<br>d [-0.030, +0.096]<br>c [-0.009, +0.120] | +0.032<br>d [-0.061, +0.088]<br>c [-0.000, +0.087] | +0.019<br>d [-0.020, +0.063]<br>c [-0.021, +0.071] |
| V4 chronological: fit to 31991, score from 32312 | K10 (100 ms) | +0.023<br>d [-0.118, +0.139]<br>c [-0.077, +0.120] | -0.134<br>d [-0.433, +0.048]<br>c [-0.298, +0.032] | +0.075<br>d [+0.026, +0.139]<br>c [+0.037, +0.128] |
| V4 chronological: fit to 32310, score from 32312 | K1 (every frame) | +0.069<br>d [+0.007, +0.127]<br>c [+0.021, +0.149] | +0.023<br>d [-0.025, +0.059]<br>c [-0.004, +0.058] | +0.063<br>d [+0.007, +0.119]<br>c [+0.014, +0.138] |
| V4 chronological: fit to 32310, score from 32312 | K10 (100 ms) | +0.105<br>d [+0.027, +0.187]<br>c [+0.018, +0.201] | +0.046<br>d [-0.068, +0.140]<br>c [-0.048, +0.144] | +0.075<br>d [+0.035, +0.119]<br>c [+0.033, +0.124] |

## The fast-channel increment TX over T0 by rule and design, target F

Cells: skill, d interval, c interval.

| design | K1 (every frame) | K2 (20 ms) | K5 (50 ms) | K10 (100 ms) | K20 (200 ms) | K50 (500 ms) |
|---|---|---|---|---|---|---|
| V1 random (r2-r4) | +0.088<br>d [+0.066, +0.110]<br>c [+0.052, +0.136] | +0.095<br>d [+0.071, +0.120]<br>c [+0.058, +0.140] | +0.129<br>d [+0.103, +0.155]<br>c [+0.088, +0.173] | +0.180<br>d [+0.148, +0.213]<br>c [+0.134, +0.229] | +0.197<br>d [+0.158, +0.236]<br>c [+0.146, +0.248] | +0.245<br>d [+0.201, +0.286]<br>c [+0.187, +0.299] |
| V2 cluster-blocked (3 assignments) | +0.068<br>d [+0.049, +0.088]<br>c [+0.039, +0.101] | +0.078<br>d [+0.057, +0.099]<br>c [+0.049, +0.108] | +0.101<br>d [+0.070, +0.137]<br>c [+0.058, +0.149] | +0.155<br>d [+0.128, +0.184]<br>c [+0.113, +0.196] | +0.154<br>d [+0.112, +0.198]<br>c [+0.100, +0.209] | +0.211<br>d [+0.165, +0.256]<br>c [+0.148, +0.270] |
| V3 rolling origin (fifths 2-5 pooled) | +0.053<br>d [+0.035, +0.072]<br>c [+0.024, +0.089] | +0.067<br>d [+0.047, +0.087]<br>c [+0.035, +0.106] | +0.097<br>d [+0.072, +0.121]<br>c [+0.055, +0.140] | +0.088<br>d [+0.045, +0.126]<br>c [+0.019, +0.147] | +0.132<br>d [+0.099, +0.161]<br>c [+0.069, +0.182] | +0.196<br>d [+0.145, +0.246]<br>c [+0.118, +0.277] |
| V4 chronological: fit to 31991, score from 32312 | +0.020<br>d [-0.024, +0.071]<br>c [-0.029, +0.047] | +0.017<br>d [-0.029, +0.066]<br>c [-0.030, +0.050] | +0.056<br>d [+0.019, +0.095]<br>c [+0.037, +0.098] | +0.070<br>d [+0.030, +0.109]<br>c [+0.034, +0.118] | +0.018<br>d [-0.043, +0.075]<br>c [-0.028, +0.103] | +0.157<br>d [+0.018, +0.280]<br>c [+0.024, +0.244] |
| V4 chronological: fit to 32310, score from 32312 | +0.058<br>d [+0.018, +0.100]<br>c [+0.007, +0.089] | +0.067<br>d [+0.027, +0.108]<br>c [+0.033, +0.096] | +0.076<br>d [+0.035, +0.119]<br>c [+0.055, +0.131] | +0.040<br>d [+0.002, +0.080]<br>c [+0.006, +0.097] | +0.028<br>d [-0.039, +0.093]<br>c [-0.026, +0.136] | +0.084<br>d [-0.018, +0.170]<br>c [-0.032, +0.186] |

### TB and TE over T0 (V1, V4), target F

| design | rule | TX over T0 | TB over T0 (BES half) | TE over T0 (ECEI half) |
|---|---|---|---|---|
| V1 random (r2-r4) | K1 (every frame) | +0.088<br>d [+0.066, +0.110]<br>c [+0.052, +0.136] | +0.065<br>d [+0.047, +0.083]<br>c [+0.038, +0.094] | +0.048<br>d [+0.029, +0.069]<br>c [+0.017, +0.091] |
| V1 random (r2-r4) | K10 (100 ms) | +0.180<br>d [+0.148, +0.213]<br>c [+0.134, +0.229] | +0.136<br>d [+0.111, +0.163]<br>c [+0.104, +0.172] | +0.108<br>d [+0.077, +0.140]<br>c [+0.061, +0.155] |
| V4 chronological: fit to 31991, score from 32312 | K1 (every frame) | +0.020<br>d [-0.024, +0.071]<br>c [-0.029, +0.047] | +0.021<br>d [-0.017, +0.061]<br>c [-0.030, +0.052] | +0.014<br>d [-0.015, +0.048]<br>c [-0.011, +0.025] |
| V4 chronological: fit to 31991, score from 32312 | K10 (100 ms) | +0.070<br>d [+0.030, +0.109]<br>c [+0.034, +0.118] | +0.014<br>d [-0.061, +0.072]<br>c [-0.047, +0.107] | +0.055<br>d [+0.020, +0.092]<br>c [+0.037, +0.083] |
| V4 chronological: fit to 32310, score from 32312 | K1 (every frame) | +0.058<br>d [+0.018, +0.100]<br>c [+0.007, +0.089] | +0.042<br>d [-0.008, +0.090]<br>c [-0.011, +0.079] | +0.016<br>d [-0.012, +0.048]<br>c [-0.012, +0.030] |
| V4 chronological: fit to 32310, score from 32312 | K10 (100 ms) | +0.040<br>d [+0.002, +0.080]<br>c [+0.006, +0.097] | +0.029<br>d [-0.016, +0.070]<br>c [-0.009, +0.088] | +0.024<br>d [-0.008, +0.061]<br>c [-0.013, +0.073] |

## T0 over G and over H, target L5

| design | K1 (every frame) | K2 (20 ms) | K5 (50 ms) | K10 (100 ms) | K20 (200 ms) | K50 (500 ms) |
|---|---|---|---|---|---|---|
| V1 random (r2-r4): T0 over G | +0.134<br>d [+0.110, +0.159]<br>c [+0.099, +0.173] | +0.196<br>d [+0.168, +0.223]<br>c [+0.162, +0.229] | +0.173<br>d [+0.138, +0.208]<br>c [+0.128, +0.218] | +0.248<br>d [+0.199, +0.295]<br>c [+0.150, +0.329] | +0.292<br>d [+0.232, +0.348]<br>c [+0.161, +0.396] | +0.365<br>d [+0.311, +0.414]<br>c [+0.259, +0.436] |
| V1 random (r2-r4): T0 over H | +0.520<br>d [+0.488, +0.551]<br>c [+0.471, +0.565] | +0.491<br>d [+0.460, +0.521]<br>c [+0.444, +0.534] | +0.462<br>d [+0.419, +0.502]<br>c [+0.391, +0.524] | +0.455<br>d [+0.407, +0.499]<br>c [+0.392, +0.511] | +0.412<br>d [+0.355, +0.465]<br>c [+0.327, +0.480] | +0.394<br>d [+0.335, +0.448]<br>c [+0.336, +0.448] |
| V2 cluster-blocked (3 assignments): T0 over G | +0.111<br>d [+0.086, +0.136]<br>c [+0.071, +0.154] | +0.162<br>d [+0.131, +0.195]<br>c [+0.115, +0.209] | +0.109<br>d [+0.068, +0.148]<br>c [+0.041, +0.176] | +0.159<br>d [+0.109, +0.212]<br>c [+0.089, +0.224] | +0.220<br>d [+0.156, +0.281]<br>c [+0.112, +0.307] | +0.277<br>d [+0.202, +0.337]<br>c [+0.168, +0.355] |
| V2 cluster-blocked (3 assignments): T0 over H | +0.507<br>d [+0.474, +0.539]<br>c [+0.454, +0.556] | +0.470<br>d [+0.436, +0.502]<br>c [+0.415, +0.522] | +0.420<br>d [+0.371, +0.466]<br>c [+0.305, +0.505] | +0.390<br>d [+0.327, +0.446]<br>c [+0.267, +0.483] | +0.351<br>d [+0.285, +0.414]<br>c [+0.273, +0.432] | +0.309<br>d [+0.233, +0.375]<br>c [+0.239, +0.383] |
| V3 rolling origin (fifths 2-5 pooled): T0 over G | +0.095<br>d [+0.070, +0.120]<br>c [+0.052, +0.143] | +0.124<br>d [+0.089, +0.159]<br>c [+0.047, +0.190] | +0.072<br>d [+0.037, +0.109]<br>c [+0.009, +0.139] | +0.076<br>d [+0.015, +0.137]<br>c [-0.014, +0.184] | +0.169<br>d [+0.121, +0.214]<br>c [+0.094, +0.225] | +0.168<br>d [+0.069, +0.248]<br>c [+0.033, +0.263] |
| V3 rolling origin (fifths 2-5 pooled): T0 over H | +0.503<br>d [+0.464, +0.541]<br>c [+0.439, +0.561] | +0.450<br>d [+0.409, +0.489]<br>c [+0.368, +0.521] | +0.399<br>d [+0.346, +0.449]<br>c [+0.265, +0.494] | +0.327<br>d [+0.241, +0.404]<br>c [+0.113, +0.474] | +0.327<br>d [+0.272, +0.381]<br>c [+0.243, +0.423] | +0.241<br>d [+0.150, +0.320]<br>c [+0.148, +0.342] |
| V4 chronological: fit to 31991, score from 32312: T0 over G | +0.238<br>d [+0.189, +0.296]<br>c [+0.192, +0.310] | +0.240<br>d [+0.171, +0.309]<br>c [+0.155, +0.284] | +0.249<br>d [+0.181, +0.315]<br>c [+0.161, +0.301] | +0.185<br>d [+0.033, +0.307]<br>c [-0.054, +0.289] | +0.058<br>d [-0.107, +0.202]<br>c [-0.214, +0.161] | -0.065<br>d [-0.464, +0.242]<br>c [-0.466, +0.218] |
| V4 chronological: fit to 31991, score from 32312: T0 over H | +0.646<br>d [+0.548, +0.740]<br>c [+0.530, +0.768] | +0.615<br>d [+0.508, +0.713]<br>c [+0.498, +0.719] | +0.644<br>d [+0.565, +0.716]<br>c [+0.568, +0.691] | +0.541<br>d [+0.392, +0.646]<br>c [+0.343, +0.624] | +0.431<br>d [+0.221, +0.597]<br>c [+0.164, +0.530] | +0.518<br>d [+0.148, +0.659]<br>c [+0.175, +0.686] |
| V4 chronological: fit to 32310, score from 32312: T0 over G | +0.278<br>d [+0.204, +0.351]<br>c [+0.198, +0.343] | +0.353<br>d [+0.269, +0.422]<br>c [+0.257, +0.405] | +0.284<br>d [+0.205, +0.360]<br>c [+0.182, +0.342] | +0.240<br>d [+0.088, +0.367]<br>c [+0.025, +0.353] | +0.113<br>d [-0.040, +0.274]<br>c [-0.094, +0.200] | +0.039<br>d [-0.280, +0.268]<br>c [-0.285, +0.246] |
| V4 chronological: fit to 32310, score from 32312: T0 over H | +0.664<br>d [+0.567, +0.756]<br>c [+0.563, +0.769] | +0.672<br>d [+0.582, +0.751]<br>c [+0.581, +0.764] | +0.661<br>d [+0.581, +0.730]<br>c [+0.572, +0.709] | +0.572<br>d [+0.422, +0.677]<br>c [+0.380, +0.655] | +0.464<br>d [+0.254, +0.633]<br>c [+0.232, +0.552] | +0.565<br>d [+0.141, +0.713]<br>c [+0.172, +0.706] |

## T0 over G and over H, target F

| design | K1 (every frame) | K2 (20 ms) | K5 (50 ms) | K10 (100 ms) | K20 (200 ms) | K50 (500 ms) |
|---|---|---|---|---|---|---|
| V1 random (r2-r4): T0 over G | +0.093<br>d [+0.074, +0.112]<br>c [+0.073, +0.116] | +0.102<br>d [+0.082, +0.122]<br>c [+0.081, +0.124] | +0.093<br>d [+0.071, +0.114]<br>c [+0.065, +0.123] | +0.142<br>d [+0.107, +0.180]<br>c [+0.072, +0.221] | +0.199<br>d [+0.154, +0.244]<br>c [+0.099, +0.298] | +0.295<br>d [+0.248, +0.338]<br>c [+0.201, +0.366] |
| V1 random (r2-r4): T0 over H | +0.367<br>d [+0.341, +0.393]<br>c [+0.334, +0.398] | +0.332<br>d [+0.306, +0.358]<br>c [+0.299, +0.365] | +0.306<br>d [+0.269, +0.345]<br>c [+0.264, +0.347] | +0.324<br>d [+0.280, +0.368]<br>c [+0.278, +0.364] | +0.306<br>d [+0.261, +0.350]<br>c [+0.241, +0.362] | +0.307<br>d [+0.258, +0.354]<br>c [+0.258, +0.353] |
| V2 cluster-blocked (3 assignments): T0 over G | +0.085<br>d [+0.067, +0.104]<br>c [+0.065, +0.109] | +0.090<br>d [+0.071, +0.111]<br>c [+0.069, +0.116] | +0.069<br>d [+0.042, +0.093]<br>c [+0.037, +0.099] | +0.091<br>d [+0.056, +0.130]<br>c [+0.041, +0.138] | +0.151<br>d [+0.107, +0.196]<br>c [+0.077, +0.219] | +0.242<br>d [+0.189, +0.290]<br>c [+0.149, +0.314] |
| V2 cluster-blocked (3 assignments): T0 over H | +0.362<br>d [+0.335, +0.386]<br>c [+0.330, +0.392] | +0.324<br>d [+0.298, +0.349]<br>c [+0.291, +0.358] | +0.288<br>d [+0.247, +0.329]<br>c [+0.232, +0.336] | +0.284<br>d [+0.232, +0.337]<br>c [+0.202, +0.347] | +0.264<br>d [+0.213, +0.312]<br>c [+0.209, +0.318] | +0.255<br>d [+0.199, +0.307]<br>c [+0.204, +0.307] |
| V3 rolling origin (fifths 2-5 pooled): T0 over G | +0.063<br>d [+0.046, +0.081]<br>c [+0.048, +0.080] | +0.054<br>d [+0.033, +0.075]<br>c [+0.023, +0.082] | +0.025<br>d [+0.005, +0.046]<br>c [-0.004, +0.058] | +0.037<br>d [+0.005, +0.070]<br>c [+0.000, +0.083] | +0.104<br>d [+0.072, +0.137]<br>c [+0.050, +0.150] | +0.148<br>d [+0.076, +0.208]<br>c [+0.042, +0.225] |
| V3 rolling origin (fifths 2-5 pooled): T0 over H | +0.347<br>d [+0.318, +0.375]<br>c [+0.307, +0.382] | +0.302<br>d [+0.273, +0.330]<br>c [+0.259, +0.342] | +0.260<br>d [+0.219, +0.302]<br>c [+0.191, +0.318] | +0.243<br>d [+0.179, +0.301]<br>c [+0.107, +0.332] | +0.242<br>d [+0.199, +0.287]<br>c [+0.177, +0.311] | +0.190<br>d [+0.123, +0.253]<br>c [+0.125, +0.258] |
| V4 chronological: fit to 31991, score from 32312: T0 over G | +0.075<br>d [+0.019, +0.124]<br>c [+0.044, +0.137] | +0.092<br>d [+0.041, +0.133]<br>c [+0.054, +0.141] | +0.051<br>d [+0.019, +0.084]<br>c [+0.022, +0.070] | +0.025<br>d [-0.014, +0.063]<br>c [-0.034, +0.055] | +0.003<br>d [-0.038, +0.044]<br>c [-0.055, +0.032] | -0.051<br>d [-0.252, +0.107]<br>c [-0.206, +0.084] |
| V4 chronological: fit to 31991, score from 32312: T0 over H | +0.323<br>d [+0.229, +0.427]<br>c [+0.196, +0.505] | +0.332<br>d [+0.251, +0.417]<br>c [+0.233, +0.465] | +0.321<br>d [+0.252, +0.385]<br>c [+0.230, +0.383] | +0.265<br>d [+0.163, +0.362]<br>c [+0.147, +0.354] | +0.228<br>d [+0.100, +0.366]<br>c [+0.096, +0.301] | +0.319<br>d [+0.047, +0.484]<br>c [+0.061, +0.504] |
| V4 chronological: fit to 32310, score from 32312: T0 over G | +0.088<br>d [+0.043, +0.134]<br>c [+0.060, +0.143] | +0.118<br>d [+0.055, +0.169]<br>c [+0.059, +0.201] | +0.077<br>d [+0.035, +0.116]<br>c [+0.034, +0.115] | +0.045<br>d [-0.004, +0.099]<br>c [-0.031, +0.086] | +0.022<br>d [-0.012, +0.057]<br>c [-0.042, +0.052] | +0.039<br>d [-0.079, +0.143]<br>c [-0.081, +0.124] |
| V4 chronological: fit to 32310, score from 32312: T0 over H | +0.332<br>d [+0.241, +0.436]<br>c [+0.207, +0.508] | +0.351<br>d [+0.266, +0.440]<br>c [+0.245, +0.496] | +0.339<br>d [+0.263, +0.407]<br>c [+0.246, +0.410] | +0.281<br>d [+0.175, +0.377]<br>c [+0.156, +0.357] | +0.244<br>d [+0.125, +0.372]<br>c [+0.124, +0.307] | +0.378<br>d [+0.067, +0.554]<br>c [+0.106, +0.537] |

## Cost of the lower rate and recovered share, target L5

Cost = MSE under the rule / MSE under K1 (same rows). Recovered share = (MSE(T0) - MSE(TX)) / (MSE(T0) - MSE(T0 under K1)) under the same rule.

| design | rule | cost, T0 | cost, TX | recovered share | TX over T0 of the next denser rule |
|---|---|---|---|---|---|
| V1 random (r2-r4) | K2 (20 ms) | 1.052<br>d [1.028, 1.079]<br>c [1.022, 1.081] | 1.049<br>d [1.025, 1.077]<br>c [1.022, 1.075] | 1.944<br>d [1.163, 3.773]<br>c [1.062, 5.344] | over T0 under K1: +0.049<br>d [+0.012, +0.083]<br>c [+0.005, +0.103] |
| V1 random (r2-r4) | K5 (50 ms) | 1.316<br>d [1.259, 1.380]<br>c [1.220, 1.435] | 1.255<br>d [1.196, 1.322]<br>c [1.172, 1.351] | 0.564<br>d [0.401, 0.734]<br>c [0.378, 0.848] | over T0 under K2: -0.081<br>d [-0.140, -0.029]<br>c [-0.159, -0.005] |
| V1 random (r2-r4) | K10 (100 ms) | 1.602<br>d [1.509, 1.709]<br>c [1.451, 1.788] | 1.430<br>d [1.361, 1.507]<br>c [1.332, 1.530] | 0.508<br>d [0.414, 0.593]<br>c [0.376, 0.647] | over T0 under K5: +0.015<br>d [-0.038, +0.064]<br>c [-0.046, +0.078] |
| V1 random (r2-r4) | K20 (200 ms) | 2.004<br>d [1.857, 2.171]<br>c [1.754, 2.283] | 1.751<br>d [1.635, 1.882]<br>c [1.591, 1.922] | 0.415<br>d [0.325, 0.503]<br>c [0.327, 0.500] | over T0 under K10: +0.009<br>d [-0.060, +0.075]<br>c [-0.073, +0.076] |
| V1 random (r2-r4) | K50 (500 ms) | 2.793<br>d [2.541, 3.068]<br>c [2.376, 3.285] | 2.220<br>d [2.047, 2.419]<br>c [1.994, 2.465] | 0.435<br>d [0.360, 0.500]<br>c [0.351, 0.510] | over T0 under K20: -0.004<br>d [-0.092, +0.079]<br>c [-0.099, +0.078] |
| V2 cluster-blocked (3 assignments) | K2 (20 ms) | 1.067<br>d [1.038, 1.099]<br>c [1.026, 1.106] | 1.065<br>d [1.028, 1.100]<br>c [1.021, 1.118] | 0.815<br>d [0.428, 1.465]<br>c [0.238, 2.659] | over T0 under K1: -0.013<br>d [-0.047, +0.020]<br>c [-0.065, +0.048] |
| V2 cluster-blocked (3 assignments) | K5 (50 ms) | 1.382<br>d [1.309, 1.466]<br>c [1.235, 1.595] | 1.291<br>d [1.213, 1.375]<br>c [1.164, 1.464] | 0.404<br>d [0.289, 0.520]<br>c [0.263, 0.632] | over T0 under K2: -0.150<br>d [-0.219, -0.089]<br>c [-0.289, -0.042] |
| V2 cluster-blocked (3 assignments) | K10 (100 ms) | 1.745<br>d [1.618, 1.893]<br>c [1.509, 2.112] | 1.536<br>d [1.448, 1.632]<br>c [1.377, 1.724] | 0.381<br>d [0.314, 0.446]<br>c [0.259, 0.486] | over T0 under K5: -0.057<br>d [-0.109, -0.008]<br>c [-0.116, -0.002] |
| V2 cluster-blocked (3 assignments) | K20 (200 ms) | 2.152<br>d [1.968, 2.375]<br>c [1.813, 2.587] | 1.866<br>d [1.721, 2.033]<br>c [1.630, 2.135] | 0.327<br>d [0.226, 0.412]<br>c [0.228, 0.411] | over T0 under K10: -0.017<br>d [-0.092, +0.051]<br>c [-0.132, +0.078] |
| V2 cluster-blocked (3 assignments) | K50 (500 ms) | 3.097<br>d [2.789, 3.483]<br>c [2.564, 3.736] | 2.454<br>d [2.243, 2.691]<br>c [2.131, 2.809] | 0.362<br>d [0.256, 0.444]<br>c [0.231, 0.459] | over T0 under K20: -0.085<br>d [-0.188, +0.007]<br>c [-0.218, +0.022] |
| V3 rolling origin (fifths 2-5 pooled) | K2 (20 ms) | 1.095<br>d [1.056, 1.138]<br>c [1.031, 1.156] | 1.062<br>d [1.027, 1.103]<br>c [1.009, 1.113] | 0.377<br>d [0.073, 0.785]<br>c [-0.320, 1.417] | over T0 under K1: -0.059<br>d [-0.108, -0.014]<br>c [-0.126, +0.019] |
| V3 rolling origin (fifths 2-5 pooled) | K5 (50 ms) | 1.432<br>d [1.345, 1.529]<br>c [1.258, 1.655] | 1.353<br>d [1.267, 1.446]<br>c [1.181, 1.543] | 0.193<br>d [0.094, 0.304]<br>c [0.043, 0.485] | over T0 under K2: -0.231<br>d [-0.305, -0.164]<br>c [-0.373, -0.096] |
| V3 rolling origin (fifths 2-5 pooled) | K10 (100 ms) | 1.918<br>d [1.723, 2.154]<br>c [1.521, 2.531] | 1.781<br>d [1.617, 1.973]<br>c [1.441, 2.249] | 0.156<br>d [0.067, 0.234]<br>c [0.029, 0.279] | over T0 under K5: -0.240<br>d [-0.334, -0.153]<br>c [-0.398, -0.077] |
| V3 rolling origin (fifths 2-5 pooled) | K20 (200 ms) | 2.263<br>d [2.042, 2.518]<br>c [1.792, 2.911] | 2.077<br>d [1.871, 2.315]<br>c [1.652, 2.655] | 0.153<br>d [0.084, 0.224]<br>c [0.049, 0.306] | over T0 under K10: -0.079<br>d [-0.143, -0.022]<br>c [-0.176, -0.003] |
| V3 rolling origin (fifths 2-5 pooled) | K50 (500 ms) | 3.379<br>d [2.964, 3.868]<br>c [2.642, 4.282] | 2.590<br>d [2.299, 2.926]<br>c [2.097, 3.257] | 0.335<br>d [0.248, 0.423]<br>c [0.208, 0.490] | over T0 under K20: -0.141<br>d [-0.236, -0.054]<br>c [-0.251, -0.058] |
| V4 chronological: fit to 31991, score from 32312 | K2 (20 ms) | 1.073<br>d [0.960, 1.217]<br>c [0.968, 1.300] | 1.039<br>d [0.942, 1.158]<br>c [0.931, 1.139] | 0.969<br>d [-4.091, 6.548]<br>c [-3.840, 5.982] | over T0 under K1: -0.002<br>d [-0.125, +0.093]<br>c [-0.070, +0.114] |
| V4 chronological: fit to 31991, score from 32312 | K5 (50 ms) | 1.128<br>d [0.949, 1.353]<br>c [0.981, 1.383] | 1.066<br>d [0.910, 1.285]<br>c [0.908, 1.342] | 0.776<br>d [-3.906, 5.542]<br>c [-3.008, 5.769] | over T0 under K2: +0.042<br>d [-0.170, +0.190]<br>c [-0.191, +0.280] |
| V4 chronological: fit to 31991, score from 32312 | K10 (100 ms) | 1.275<br>d [1.073, 1.584]<br>c [1.097, 1.599] | 1.290<br>d [1.036, 1.714]<br>c [1.041, 1.805] | 0.108<br>d [-0.640, 0.904]<br>c [-0.314, 0.837] | over T0 under K5: -0.103<br>d [-0.440, +0.124]<br>c [-0.418, +0.111] |
| V4 chronological: fit to 31991, score from 32312 | K20 (200 ms) | 1.400<br>d [1.119, 1.840]<br>c [1.134, 2.012] | 1.439<br>d [1.144, 1.894]<br>c [1.150, 1.957] | 0.028<br>d [-0.561, 0.406]<br>c [-0.446, 0.499] | over T0 under K10: -0.089<br>d [-0.290, +0.068]<br>c [-0.256, +0.088] |
| V4 chronological: fit to 31991, score from 32312 | K50 (500 ms) | 2.210<br>d [1.729, 2.783]<br>c [1.933, 2.540] | 1.616<br>d [1.279, 2.182]<br>c [1.180, 2.416] | 0.538<br>d [0.044, 0.801]<br>c [0.132, 0.860] | over T0 under K20: -0.114<br>d [-0.382, +0.098]<br>c [-0.378, +0.149] |
| V4 chronological: fit to 32310, score from 32312 | K2 (20 ms) | 0.964<br>d [0.886, 1.079]<br>c [0.886, 1.090] | 0.992<br>d [0.905, 1.097]<br>c [0.912, 1.053] | -1.133<br>d [-13.854, 13.621]<br>c [-14.905, 17.138] | over T0 under K1: +0.077<br>d [-0.039, +0.174]<br>c [+0.012, +0.190] |
| V4 chronological: fit to 32310, score from 32312 | K5 (50 ms) | 1.134<br>d [0.989, 1.329]<br>c [0.994, 1.379] | 1.106<br>d [0.991, 1.275]<br>c [0.967, 1.326] | 0.780<br>d [-1.551, 4.433]<br>c [-2.119, 5.054] | over T0 under K2: -0.068<br>d [-0.216, +0.058]<br>c [-0.245, +0.142] |
| V4 chronological: fit to 32310, score from 32312 | K10 (100 ms) | 1.253<br>d [1.066, 1.551]<br>c [1.045, 1.593] | 1.205<br>d [1.016, 1.509]<br>c [0.980, 1.612] | 0.520<br>d [0.128, 1.542]<br>c [0.057, 2.209] | over T0 under K5: +0.011<br>d [-0.185, +0.165]<br>c [-0.200, +0.195] |
| V4 chronological: fit to 32310, score from 32312 | K20 (200 ms) | 1.390<br>d [1.144, 1.790]<br>c [1.147, 1.877] | 1.426<br>d [1.155, 1.864]<br>c [1.171, 1.865] | 0.160<br>d [-0.245, 0.574]<br>c [-0.164, 0.753] | over T0 under K10: -0.059<br>d [-0.248, +0.098]<br>c [-0.285, +0.155] |
| V4 chronological: fit to 32310, score from 32312 | K50 (500 ms) | 2.103<br>d [1.745, 2.563]<br>c [1.857, 2.572] | 1.641<br>d [1.328, 2.171]<br>c [1.245, 2.406] | 0.521<br>d [0.213, 0.757]<br>c [0.215, 0.850] | over T0 under K20: -0.099<br>d [-0.333, +0.074]<br>c [-0.410, +0.134] |

## Cost of the lower rate and recovered share, target F

Cost = MSE under the rule / MSE under K1 (same rows). Recovered share = (MSE(T0) - MSE(TX)) / (MSE(T0) - MSE(T0 under K1)) under the same rule.

| design | rule | cost, T0 | cost, TX | recovered share | TX over T0 of the next denser rule |
|---|---|---|---|---|---|
| V1 random (r2-r4) | K2 (20 ms) | 1.055<br>d [1.034, 1.076]<br>c [1.028, 1.081] | 1.047<br>d [1.024, 1.071]<br>c [1.020, 1.074] | 1.832<br>d [1.272, 2.829]<br>c [1.161, 2.973] | over T0 under K1: +0.046<br>d [+0.018, +0.072]<br>c [+0.010, +0.083] |
| V1 random (r2-r4) | K5 (50 ms) | 1.258<br>d [1.217, 1.301]<br>c [1.195, 1.337] | 1.201<br>d [1.164, 1.242]<br>c [1.155, 1.255] | 0.631<br>d [0.505, 0.761]<br>c [0.460, 0.851] | over T0 under K2: -0.038<br>d [-0.080, +0.002]<br>c [-0.105, +0.031] |
| V1 random (r2-r4) | K10 (100 ms) | 1.476<br>d [1.407, 1.554]<br>c [1.362, 1.627] | 1.327<br>d [1.275, 1.385]<br>c [1.256, 1.416] | 0.559<br>d [0.473, 0.637]<br>c [0.453, 0.679] | over T0 under K5: +0.038<br>d [+0.005, +0.071]<br>c [-0.004, +0.083] |
| V1 random (r2-r4) | K20 (200 ms) | 1.749<br>d [1.644, 1.870]<br>c [1.564, 1.998] | 1.540<br>d [1.462, 1.632]<br>c [1.420, 1.691] | 0.460<br>d [0.380, 0.532]<br>c [0.373, 0.541] | over T0 under K10: +0.048<br>d [+0.003, +0.091]<br>c [-0.004, +0.095] |
| V1 random (r2-r4) | K50 (500 ms) | 2.227<br>d [2.047, 2.436]<br>c [1.917, 2.655] | 1.845<br>d [1.721, 1.987]<br>c [1.669, 2.082] | 0.444<br>d [0.376, 0.506]<br>c [0.368, 0.505] | over T0 under K20: +0.038<br>d [-0.020, +0.094]<br>c [-0.029, +0.101] |
| V2 cluster-blocked (3 assignments) | K2 (20 ms) | 1.059<br>d [1.037, 1.081]<br>c [1.033, 1.083] | 1.048<br>d [1.025, 1.072]<br>c [1.021, 1.076] | 1.403<br>d [0.943, 2.163]<br>c [0.886, 2.288] | over T0 under K1: +0.023<br>d [-0.004, +0.050]<br>c [-0.008, +0.054] |
| V2 cluster-blocked (3 assignments) | K5 (50 ms) | 1.279<br>d [1.232, 1.334]<br>c [1.200, 1.399] | 1.234<br>d [1.192, 1.280]<br>c [1.166, 1.327] | 0.462<br>d [0.326, 0.598]<br>c [0.283, 0.678] | over T0 under K2: -0.086<br>d [-0.135, -0.043]<br>c [-0.179, -0.011] |
| V2 cluster-blocked (3 assignments) | K10 (100 ms) | 1.549<br>d [1.453, 1.656]<br>c [1.376, 1.821] | 1.404<br>d [1.333, 1.484]<br>c [1.278, 1.603] | 0.438<br>d [0.375, 0.500]<br>c [0.358, 0.542] | over T0 under K5: -0.023<br>d [-0.063, +0.015]<br>c [-0.084, +0.026] |
| V2 cluster-blocked (3 assignments) | K20 (200 ms) | 1.837<br>d [1.703, 2.000]<br>c [1.583, 2.235] | 1.667<br>d [1.560, 1.793]<br>c [1.484, 1.943] | 0.339<br>d [0.254, 0.416]<br>c [0.247, 0.423] | over T0 under K10: -0.003<br>d [-0.054, +0.045]<br>c [-0.063, +0.045] |
| V2 cluster-blocked (3 assignments) | K50 (500 ms) | 2.372<br>d [2.157, 2.630]<br>c [2.002, 2.895] | 2.007<br>d [1.849, 2.188]<br>c [1.767, 2.355] | 0.366<br>d [0.293, 0.430]<br>c [0.276, 0.442] | over T0 under K20: -0.018<br>d [-0.084, +0.042]<br>c [-0.105, +0.059] |
| V3 rolling origin (fifths 2-5 pooled) | K2 (20 ms) | 1.068<br>d [1.043, 1.093]<br>c [1.037, 1.099] | 1.052<br>d [1.025, 1.079]<br>c [1.015, 1.089] | 1.064<br>d [0.694, 1.672]<br>c [0.558, 1.925] | over T0 under K1: +0.004<br>d [-0.025, +0.033]<br>c [-0.033, +0.043] |
| V3 rolling origin (fifths 2-5 pooled) | K5 (50 ms) | 1.305<br>d [1.252, 1.364]<br>c [1.206, 1.442] | 1.245<br>d [1.201, 1.295]<br>c [1.172, 1.352] | 0.413<br>d [0.315, 0.512]<br>c [0.278, 0.582] | over T0 under K2: -0.104<br>d [-0.152, -0.060]<br>c [-0.192, -0.027] |
| V3 rolling origin (fifths 2-5 pooled) | K10 (100 ms) | 1.589<br>d [1.475, 1.727]<br>c [1.355, 1.968] | 1.531<br>d [1.429, 1.650]<br>c [1.333, 1.835] | 0.237<br>d [0.128, 0.330]<br>c [0.063, 0.379] | over T0 under K5: -0.111<br>d [-0.175, -0.055]<br>c [-0.223, -0.009] |
| V3 rolling origin (fifths 2-5 pooled) | K20 (200 ms) | 1.852<br>d [1.698, 2.034]<br>c [1.536, 2.355] | 1.699<br>d [1.577, 1.846]<br>c [1.460, 2.081] | 0.286<br>d [0.225, 0.340]<br>c [0.184, 0.367] | over T0 under K10: -0.012<br>d [-0.058, +0.028]<br>c [-0.067, +0.031] |
| V3 rolling origin (fifths 2-5 pooled) | K50 (500 ms) | 2.473<br>d [2.207, 2.791]<br>c [1.990, 3.164] | 2.100<br>d [1.911, 2.327]<br>c [1.783, 2.629] | 0.329<br>d [0.253, 0.402]<br>c [0.220, 0.449] | over T0 under K20: -0.073<br>d [-0.143, -0.012]<br>c [-0.154, -0.013] |
| V4 chronological: fit to 31991, score from 32312 | K2 (20 ms) | 1.001<br>d [0.932, 1.067]<br>c [0.937, 1.111] | 1.004<br>d [0.936, 1.073]<br>c [0.932, 1.118] | 27.741<br>d [-10.007, 9.613]<br>c [-8.172, 7.153] | over T0 under K1: +0.016<br>d [-0.081, +0.118]<br>c [-0.124, +0.105] |
| V4 chronological: fit to 31991, score from 32312 | K5 (50 ms) | 1.098<br>d [0.991, 1.217]<br>c [1.027, 1.264] | 1.057<br>d [0.986, 1.149]<br>c [0.986, 1.173] | 0.635<br>d [-1.494, 3.650]<br>c [0.289, 2.063] | over T0 under K2: -0.035<br>d [-0.120, +0.035]<br>c [-0.122, +0.060] |
| V4 chronological: fit to 31991, score from 32312 | K10 (100 ms) | 1.147<br>d [1.038, 1.273]<br>c [1.067, 1.301] | 1.089<br>d [1.001, 1.208]<br>c [0.958, 1.262] | 0.546<br>d [0.199, 1.739]<br>c [0.168, 1.519] | over T0 under K5: +0.028<br>d [-0.043, +0.098]<br>c [-0.056, +0.153] |
| V4 chronological: fit to 31991, score from 32312 | K20 (200 ms) | 1.184<br>d [1.072, 1.337]<br>c [1.089, 1.422] | 1.187<br>d [1.097, 1.325]<br>c [1.070, 1.355] | 0.116<br>d [-0.425, 0.407]<br>c [-0.205, 0.510] | over T0 under K10: -0.014<br>d [-0.093, +0.054]<br>c [-0.099, +0.039] |
| V4 chronological: fit to 31991, score from 32312 | K50 (500 ms) | 1.536<br>d [1.290, 1.891]<br>c [1.378, 1.788] | 1.321<br>d [1.171, 1.543]<br>c [1.157, 1.612] | 0.451<br>d [0.064, 0.664]<br>c [0.067, 0.678] | over T0 under K20: -0.093<br>d [-0.217, +0.007]<br>c [-0.195, -0.004] |
| V4 chronological: fit to 32310, score from 32312 | K2 (20 ms) | 0.985<br>d [0.936, 1.042]<br>c [0.935, 1.065] | 0.977<br>d [0.940, 1.021]<br>c [0.931, 1.043] | -4.511<br>d [-35.632, 34.519]<br>c [-24.673, 18.645] | over T0 under K1: +0.080<br>d [+0.015, +0.138]<br>c [-0.005, +0.142] |
| V4 chronological: fit to 32310, score from 32312 | K5 (50 ms) | 1.083<br>d [1.004, 1.180]<br>c [1.015, 1.224] | 1.063<br>d [0.989, 1.151]<br>c [1.005, 1.150] | 0.986<br>d [0.246, 4.868]<br>c [0.487, 4.177] | over T0 under K2: -0.016<br>d [-0.080, +0.040]<br>c [-0.058, +0.046] |
| V4 chronological: fit to 32310, score from 32312 | K10 (100 ms) | 1.138<br>d [1.043, 1.257]<br>c [1.049, 1.304] | 1.160<br>d [1.079, 1.269]<br>c [1.056, 1.288] | 0.333<br>d [0.018, 0.819]<br>c [0.072, 1.012] | over T0 under K5: -0.008<br>d [-0.065, +0.048]<br>c [-0.060, +0.093] |
| V4 chronological: fit to 32310, score from 32312 | K20 (200 ms) | 1.176<br>d [1.076, 1.326]<br>c [1.080, 1.411] | 1.214<br>d [1.113, 1.365]<br>c [1.098, 1.350] | 0.189<br>d [-0.321, 0.586]<br>c [-0.239, 0.703] | over T0 under K10: -0.004<br>d [-0.086, +0.064]<br>c [-0.082, +0.077] |
| V4 chronological: fit to 32310, score from 32312 | K50 (500 ms) | 1.423<br>d [1.271, 1.644]<br>c [1.283, 1.724] | 1.384<br>d [1.220, 1.646]<br>c [1.196, 1.710] | 0.283<br>d [-0.061, 0.574]<br>c [-0.094, 0.624] | over T0 under K20: -0.108<br>d [-0.246, -0.003]<br>c [-0.249, +0.029] |

## Equivalent rate, target L5

k' = the rate at which T0 would have the MSE of TX under k (linear interpolation of T0's MSE in ln k between the rules); factor = k / k'. 'below 1': TX under k has an MSE at most that of T0 under K1. Intervals draw by draw; shares of censored draws in the cluster form.

| design | k | k' point | factor point | k' d | k' c | factor c | cluster draws below 1 / above 50 | T0 curve monotone (point; share of cluster draws) |
|---|---|---|---|---|---|---|---|---|
| V1 random (r2-r4) | 1 | below 1 | above 1 | [below 1, below 1] | [below 1, below 1] | [above 1, above 1] | 1.000 / 0.000 | True; 0.999 |
| V1 random (r2-r4) | 2 | below 1 | above 2 | [below 1, below 1] | [below 1, below 1] | [above 2, above 2] | 0.987 / 0.000 | True; 0.999 |
| V1 random (r2-r4) | 5 | 2.69 | 1.86 | [2.26, 3.18] | [2.05, 3.25] | [1.54, 2.44] | 0.004 / 0.000 | True; 0.999 |
| V1 random (r2-r4) | 10 | 4.67 | 2.14 | [3.81, 5.57] | [3.59, 5.78] | [1.73, 2.78] | 0.000 / 0.000 | True; 0.999 |
| V1 random (r2-r4) | 20 | 9.64 | 2.07 | [7.61, 11.56] | [7.53, 11.86] | [1.69, 2.66] | 0.000 / 0.000 | True; 0.999 |
| V1 random (r2-r4) | 50 | 20.20 | 2.48 | [15.56, 24.14] | [15.41, 24.71] | [2.02, 3.24] | 0.000 / 0.000 | True; 0.999 |
| V2 cluster-blocked (3 assignments) | 1 | below 1 | above 1 | [below 1, below 1] | [below 1, below 1] | [above 1, above 1] | 0.986 / 0.000 | True; 0.999 |
| V2 cluster-blocked (3 assignments) | 2 | 1.14 | 1.76 | [below 1, 1.49] | [below 1, 1.69] | [1.18, above 2] | 0.330 / 0.000 | True; 0.999 |
| V2 cluster-blocked (3 assignments) | 5 | 3.19 | 1.57 | [2.77, 3.64] | [2.45, 3.75] | [1.33, 2.04] | 0.000 / 0.000 | True; 0.999 |
| V2 cluster-blocked (3 assignments) | 10 | 5.81 | 1.72 | [5.13, 6.46] | [5.03, 7.00] | [1.43, 1.99] | 0.000 / 0.000 | True; 0.999 |
| V2 cluster-blocked (3 assignments) | 20 | 10.52 | 1.90 | [8.52, 12.80] | [7.96, 13.69] | [1.46, 2.51] | 0.000 / 0.000 | True; 0.999 |
| V2 cluster-blocked (3 assignments) | 50 | 23.88 | 2.09 | [19.57, 29.06] | [18.51, 30.85] | [1.62, 2.70] | 0.000 / 0.000 | True; 0.999 |
| V3 rolling origin (fifths 2-5 pooled) | 1 | below 1 | above 1 | [below 1, 1.20] | [below 1, 1.37] | [0.73, above 1] | 0.569 / 0.000 | True; 0.998 |
| V3 rolling origin (fifths 2-5 pooled) | 2 | 1.54 | 1.30 | [1.16, 1.90] | [below 1, 2.15] | [0.93, above 2] | 0.065 / 0.000 | True; 0.998 |
| V3 rolling origin (fifths 2-5 pooled) | 5 | 3.98 | 1.25 | [3.53, 4.46] | [2.97, 4.75] | [1.05, 1.68] | 0.000 / 0.000 | True; 0.998 |
| V3 rolling origin (fifths 2-5 pooled) | 10 | 8.16 | 1.23 | [7.35, 9.12] | [6.66, 9.58] | [1.04, 1.50] | 0.000 / 0.000 | True; 0.998 |
| V3 rolling origin (fifths 2-5 pooled) | 20 | 13.58 | 1.47 | [11.14, 15.98] | [10.16, 17.60] | [1.14, 1.97] | 0.000 / 0.000 | True; 0.998 |
| V3 rolling origin (fifths 2-5 pooled) | 50 | 25.98 | 1.92 | [22.48, 29.90] | [22.07, 31.78] | [1.57, 2.27] | 0.000 / 0.000 | True; 0.998 |
| V4 chronological: fit to 31991, score from 32312 | 1 | below 1 | above 1 | [below 1, 2.31] | [below 1, 2.28] | [0.44, above 1] | 0.932 / 0.000 | True; 0.498 |
| V4 chronological: fit to 31991, score from 32312 | 2 | 1.02 | 1.96 | [below 1, 2.44] | [below 1, 2.70] | [0.74, above 2] | 0.474 / 0.000 | True; 0.498 |
| V4 chronological: fit to 31991, score from 32312 | 5 | 1.31 | 3.81 | [below 1, 5.16] | [below 1, 4.05] | [1.23, above 5] | 0.359 / 0.000 | True; 0.498 |
| V4 chronological: fit to 31991, score from 32312 | 10 | 8.69 | 1.15 | [1.12, 21.78] | [1.12, 13.70] | [0.73, 8.96] | 0.016 / 0.000 | True; 0.498 |
| V4 chronological: fit to 31991, score from 32312 | 20 | 18.81 | 1.06 | [3.80, 26.05] | [1.82, 22.54] | [0.89, 10.98] | 0.004 / 0.000 | True; 0.498 |
| V4 chronological: fit to 31991, score from 32312 | 50 | 23.95 | 2.09 | [11.47, 45.01] | [4.98, 37.97] | [1.32, 10.03] | 0.002 / 0.001 | True; 0.498 |
| V4 chronological: fit to 32310, score from 32312 | 1 | below 1 | above 1 | [below 1, below 1] | [below 1, below 1] | [above 1, above 1] | 0.999 / 0.000 | False; 0.170 |
| V4 chronological: fit to 32310, score from 32312 | 2 | below 1 | above 2 | [below 1, 2.25] | [below 1, below 1] | [above 2, above 2] | 0.995 / 0.000 | False; 0.170 |
| V4 chronological: fit to 32310, score from 32312 | 5 | 2.85 | 1.76 | [below 1, 4.41] | [below 1, 4.33] | [1.16, above 5] | 0.349 / 0.000 | False; 0.170 |
| V4 chronological: fit to 32310, score from 32312 | 10 | 4.68 | 2.14 | [below 1, 8.57] | [below 1, 8.71] | [1.15, above 10] | 0.135 / 0.000 | False; 0.170 |
| V4 chronological: fit to 32310, score from 32312 | 20 | 14.56 | 1.37 | [4.32, 22.29] | [3.85, 21.27] | [0.94, 5.20] | 0.012 / 0.000 | False; 0.170 |
| V4 chronological: fit to 32310, score from 32312 | 50 | 23.89 | 2.09 | [13.64, 33.75] | [9.60, 35.13] | [1.42, 5.21] | 0.002 / 0.001 | False; 0.170 |

## Equivalent rate, target F

k' = the rate at which T0 would have the MSE of TX under k (linear interpolation of T0's MSE in ln k between the rules); factor = k / k'. 'below 1': TX under k has an MSE at most that of T0 under K1. Intervals draw by draw; shares of censored draws in the cluster form.

| design | k | k' point | factor point | k' d | k' c | factor c | cluster draws below 1 / above 50 | T0 curve monotone (point; share of cluster draws) |
|---|---|---|---|---|---|---|---|---|
| V1 random (r2-r4) | 1 | below 1 | above 1 | [below 1, below 1] | [below 1, below 1] | [above 1, above 1] | 1.000 / 0.000 | True; 1.000 |
| V1 random (r2-r4) | 2 | below 1 | above 2 | [below 1, below 1] | [below 1, below 1] | [above 2, above 2] | 0.996 / 0.000 | True; 1.000 |
| V1 random (r2-r4) | 5 | 2.40 | 2.08 | [1.96, 2.81] | [1.44, 2.99] | [1.67, 3.48] | 0.001 / 0.000 | True; 1.000 |
| V1 random (r2-r4) | 10 | 4.03 | 2.48 | [3.36, 4.85] | [2.99, 5.09] | [1.97, 3.34] | 0.000 / 0.000 | True; 1.000 |
| V1 random (r2-r4) | 20 | 7.96 | 2.51 | [6.62, 9.86] | [6.47, 10.12] | [1.98, 3.09] | 0.000 / 0.000 | True; 1.000 |
| V1 random (r2-r4) | 50 | 16.88 | 2.96 | [13.47, 21.20] | [13.08, 21.84] | [2.29, 3.82] | 0.000 / 0.000 | True; 1.000 |
| V2 cluster-blocked (3 assignments) | 1 | below 1 | above 1 | [below 1, below 1] | [below 1, below 1] | [above 1, above 1] | 1.000 / 0.000 | True; 1.000 |
| V2 cluster-blocked (3 assignments) | 2 | below 1 | above 2 | [below 1, 1.04] | [below 1, 1.08] | [1.85, above 2] | 0.928 / 0.000 | True; 1.000 |
| V2 cluster-blocked (3 assignments) | 5 | 2.92 | 1.71 | [2.47, 3.43] | [2.13, 3.63] | [1.38, 2.35] | 0.000 / 0.000 | True; 1.000 |
| V2 cluster-blocked (3 assignments) | 10 | 5.39 | 1.85 | [4.63, 6.01] | [4.22, 6.17] | [1.62, 2.37] | 0.000 / 0.000 | True; 1.000 |
| V2 cluster-blocked (3 assignments) | 20 | 10.10 | 1.98 | [8.43, 12.09] | [8.40, 12.53] | [1.60, 2.38] | 0.000 / 0.000 | True; 1.000 |
| V2 cluster-blocked (3 assignments) | 50 | 21.14 | 2.36 | [16.78, 25.36] | [15.76, 26.83] | [1.86, 3.17] | 0.000 / 0.000 | True; 1.000 |
| V3 rolling origin (fifths 2-5 pooled) | 1 | below 1 | above 1 | [below 1, below 1] | [below 1, below 1] | [above 1, above 1] | 1.000 / 0.000 | True; 1.000 |
| V3 rolling origin (fifths 2-5 pooled) | 2 | below 1 | above 2 | [below 1, 1.24] | [below 1, 1.36] | [1.47, above 2] | 0.583 / 0.000 | True; 1.000 |
| V3 rolling origin (fifths 2-5 pooled) | 5 | 3.08 | 1.63 | [2.67, 3.46] | [2.33, 3.65] | [1.37, 2.15] | 0.000 / 0.000 | True; 1.000 |
| V3 rolling origin (fifths 2-5 pooled) | 10 | 7.11 | 1.41 | [6.14, 8.29] | [5.23, 9.11] | [1.10, 1.91] | 0.000 / 0.000 | True; 1.000 |
| V3 rolling origin (fifths 2-5 pooled) | 20 | 10.53 | 1.90 | [9.02, 12.37] | [8.92, 13.50] | [1.48, 2.24] | 0.000 / 0.000 | True; 1.000 |
| V3 rolling origin (fifths 2-5 pooled) | 50 | 24.45 | 2.04 | [20.76, 28.47] | [20.73, 30.86] | [1.62, 2.41] | 0.000 / 0.000 | True; 1.000 |
| V4 chronological: fit to 31991, score from 32312 | 1 | below 1 | above 1 | [below 1, 2.10] | [below 1, 1.58] | [0.63, above 1] | 0.811 / 0.000 | True; 0.321 |
| V4 chronological: fit to 31991, score from 32312 | 2 | below 1 | above 2 | [below 1, 2.92] | [below 1, 6.36] | [0.31, above 2] | 0.564 / 0.000 | True; 0.321 |
| V4 chronological: fit to 31991, score from 32312 | 5 | 2.78 | 1.80 | [below 1, 4.07] | [below 1, 3.66] | [1.37, above 5] | 0.178 / 0.000 | True; 0.321 |
| V4 chronological: fit to 31991, score from 32312 | 10 | 3.73 | 2.68 | [below 1, 6.74] | [below 1, 7.28] | [1.37, above 10] | 0.157 / 0.000 | True; 0.321 |
| V4 chronological: fit to 31991, score from 32312 | 20 | 13.43 | 1.49 | [4.14, 22.81] | [3.28, 21.43] | [0.93, 6.09] | 0.001 / 0.000 | True; 0.321 |
| V4 chronological: fit to 31991, score from 32312 | 50 | 26.65 | 1.88 | [18.50, 44.05] | [20.07, 43.82] | [1.14, 2.49] | 0.000 / 0.013 | True; 0.321 |
| V4 chronological: fit to 32310, score from 32312 | 1 | below 1 | above 1 | [below 1, below 1] | [below 1, below 1] | [above 1, above 1] | 0.985 / 0.000 | False; 0.242 |
| V4 chronological: fit to 32310, score from 32312 | 2 | below 1 | above 2 | [below 1, below 1] | [below 1, 1.07] | [1.87, above 2] | 0.962 / 0.000 | False; 0.242 |
| V4 chronological: fit to 32310, score from 32312 | 5 | 2.32 | 2.16 | [below 1, 3.53] | [below 1, 2.92] | [1.71, above 5] | 0.488 / 0.000 | False; 0.242 |
| V4 chronological: fit to 32310, score from 32312 | 10 | 5.61 | 1.78 | [2.96, 9.39] | [below 1, 8.88] | [1.13, above 10] | 0.026 / 0.000 | False; 0.242 |
| V4 chronological: fit to 32310, score from 32312 | 20 | 10.87 | 1.84 | [3.94, 23.24] | [3.13, 22.01] | [0.91, 6.39] | 0.004 / 0.000 | False; 0.242 |
| V4 chronological: fit to 32310, score from 32312 | 50 | 32.06 | 1.56 | [20.33, above 50] | [14.63, above 50] | [below 1.00, 3.42] | 0.000 / 0.055 | False; 0.242 |

## Against the frame's own reading O (target L5)

Skill over O = 1 - MSE(estimator) / MSE(O); above zero means closer to the following level than the frame's own reported value.

| design | estimator | K1 (every frame) | K2 (20 ms) | K5 (50 ms) | K10 (100 ms) | K20 (200 ms) | K50 (500 ms) |
|---|---|---|---|---|---|---|---|
| V1 random (r2-r4) | T0 over O | +0.486<br>d [+0.445, +0.525]<br>c [+0.416, +0.545] | +0.459<br>d [+0.418, +0.498]<br>c [+0.392, +0.517] | +0.324<br>d [+0.263, +0.380]<br>c [+0.206, +0.417] | +0.177<br>d [+0.096, +0.249]<br>c [+0.018, +0.300] | -0.030<br>d [-0.138, +0.066]<br>c [-0.247, +0.142] | -0.435<br>d [-0.617, -0.277]<br>c [-0.779, -0.165] |
| V1 random (r2-r4) | TX over O | +0.534<br>d [+0.499, +0.567]<br>c [+0.479, +0.582] | +0.511<br>d [+0.475, +0.545]<br>c [+0.455, +0.561] | +0.415<br>d [+0.364, +0.462]<br>c [+0.324, +0.492] | +0.334<br>d [+0.275, +0.387]<br>c [+0.231, +0.421] | +0.185<br>d [+0.106, +0.254]<br>c [+0.040, +0.302] | -0.034<br>d [-0.149, +0.068]<br>c [-0.226, +0.120] |
| V2 cluster-blocked (3 assignments) | T0 over O | +0.472<br>d [+0.430, +0.512]<br>c [+0.398, +0.536] | +0.437<br>d [+0.392, +0.479]<br>c [+0.362, +0.504] | +0.271<br>d [+0.196, +0.337]<br>c [+0.097, +0.396] | +0.079<br>d [-0.030, +0.174]<br>c [-0.191, +0.262] | -0.136<br>d [-0.281, -0.012]<br>c [-0.452, +0.095] | -0.634<br>d [-0.874, -0.437]<br>c [-1.081, -0.299] |
| V2 cluster-blocked (3 assignments) | TX over O | +0.498<br>d [+0.457, +0.537]<br>c [+0.432, +0.557] | +0.466<br>d [+0.425, +0.504]<br>c [+0.394, +0.531] | +0.352<br>d [+0.290, +0.408]<br>c [+0.210, +0.459] | +0.229<br>d [+0.150, +0.300]<br>c [+0.061, +0.359] | +0.064<br>d [-0.039, +0.156]<br>c [-0.145, +0.227] | -0.232<br>d [-0.387, -0.095]<br>c [-0.508, -0.008] |
| V3 rolling origin (fifths 2-5 pooled) | T0 over O | +0.474<br>d [+0.425, +0.520]<br>c [+0.389, +0.547] | +0.424<br>d [+0.372, +0.474]<br>c [+0.323, +0.510] | +0.247<br>d [+0.163, +0.322]<br>c [+0.043, +0.388] | -0.009<br>d [-0.169, +0.126]<br>c [-0.464, +0.270] | -0.190<br>d [-0.360, -0.043]<br>c [-0.671, +0.111] | -0.777<br>d [-1.083, -0.519]<br>c [-1.403, -0.322] |
| V3 rolling origin (fifths 2-5 pooled) | TX over O | +0.476<br>d [+0.431, +0.518]<br>c [+0.393, +0.545] | +0.443<br>d [+0.397, +0.487]<br>c [+0.364, +0.512] | +0.291<br>d [+0.213, +0.361]<br>c [+0.099, +0.431] | +0.066<br>d [-0.071, +0.184]<br>c [-0.313, +0.305] | -0.089<br>d [-0.253, +0.048]<br>c [-0.547, +0.201] | -0.358<br>d [-0.578, -0.174]<br>c [-0.891, -0.027] |
| V4 chronological: fit to 31991, score from 32312 | T0 over O | +0.665<br>d [+0.555, +0.758]<br>c [+0.591, +0.747] | +0.640<br>d [+0.524, +0.740]<br>c [+0.516, +0.732] | +0.622<br>d [+0.496, +0.725]<br>c [+0.520, +0.709] | +0.573<br>d [+0.444, +0.671]<br>c [+0.475, +0.658] | +0.531<br>d [+0.368, +0.650]<br>c [+0.351, +0.631] | +0.260<br>d [-0.110, +0.524]<br>c [+0.141, +0.414] |
| V4 chronological: fit to 31991, score from 32312 | TX over O | +0.677<br>d [+0.570, +0.767]<br>c [+0.609, +0.760] | +0.664<br>d [+0.562, +0.753]<br>c [+0.581, +0.761] | +0.655<br>d [+0.543, +0.745]<br>c [+0.579, +0.732] | +0.583<br>d [+0.446, +0.688]<br>c [+0.469, +0.677] | +0.535<br>d [+0.368, +0.657]<br>c [+0.396, +0.649] | +0.478<br>d [+0.289, +0.617]<br>c [+0.344, +0.596] |
| V4 chronological: fit to 32310, score from 32312 | T0 over O | +0.682<br>d [+0.579, +0.771]<br>c [+0.606, +0.756] | +0.694<br>d [+0.601, +0.773]<br>c [+0.593, +0.770] | +0.640<br>d [+0.523, +0.735]<br>c [+0.547, +0.712] | +0.602<br>d [+0.477, +0.698]<br>c [+0.505, +0.681] | +0.559<br>d [+0.406, +0.673]<br>c [+0.401, +0.652] | +0.332<br>d [+0.067, +0.529]<br>c [+0.154, +0.447] |
| V4 chronological: fit to 32310, score from 32312 | TX over O | +0.704<br>d [+0.605, +0.788]<br>c [+0.640, +0.778] | +0.707<br>d [+0.613, +0.788]<br>c [+0.638, +0.786] | +0.673<br>d [+0.569, +0.760]<br>c [+0.603, +0.748] | +0.644<br>d [+0.519, +0.741]<br>c [+0.548, +0.725] | +0.578<br>d [+0.421, +0.695]<br>c [+0.444, +0.686] | +0.515<br>d [+0.337, +0.642]<br>c [+0.346, +0.629] |

## Controls (V1, K10, L5)

TXp: TX refitted with the fourteen fast and anchor scalars permuted over rows (within the fit rows and within the test rows). TXo: TX refitted with the eight fast scalars of the anchor row (the row of the newest visible value) in place of row t, and a1 of BES and ECEI.

| quantity | value |
|---|---|
| RMSE T0 / TX / TXp / TXo | 160.7 / 144.6 / 160.9 / 154.3 |
| TXp over T0 | -0.003<br>d [-0.014, +0.009]<br>c [-0.015, +0.009] |
| TXo over T0 | +0.078<br>d [+0.046, +0.109]<br>c [+0.038, +0.116] |
| TX over TXo | +0.122<br>d [+0.088, +0.156]<br>c [+0.076, +0.170] |

Rows of TXo whose anchor-row fast scalars are not finite: at most 0 fit rows and 0 test rows per unit (left to the regression's handling of missing values).

## By position between two kept values, target L5

uf = frame index of row t minus that of the row of the newest visible value. Thirds of the kept spacing k: 3 uf < k; k <= 3 uf < 2k; 2k <= 3 uf < 3k; 'beyond one spacing': uf >= k (the newest kept frame before t carries no visible value). Empty strata are empty by the scored-row rule (Addendum 1 section 6).

| design | rule | position | rows / discharges / clusters | RMSE T0 | RMSE TX | TX over T0 |
|---|---|---|---|---|---|---|
| V1 random (r2-r4) | K2 (20 ms) | third 1 | 0 | - | - | - |
| V1 random (r2-r4) | K2 (20 ms) | third 2 | 59,648 / 507 / 72 | 127.1 | 120.8 | +0.096<br>d [+0.073, +0.119]<br>c [+0.068, +0.131] |
| V1 random (r2-r4) | K2 (20 ms) | third 3 | 0 | - | - | - |
| V1 random (r2-r4) | K2 (20 ms) | beyond one spacing | 875 / 222 / 46 | 268.8 | 255.8 | +0.094<br>d [-0.007, +0.185]<br>c [-0.013, +0.180] |
| V1 random (r2-r4) | K5 (50 ms) | third 1 | 15,126 / 504 / 72 | 131.7 | 125.3 | +0.095<br>d [+0.042, +0.142]<br>c [+0.051, +0.134] |
| V1 random (r2-r4) | K5 (50 ms) | third 2 | 29,827 / 504 / 72 | 140.6 | 130.3 | +0.142<br>d [+0.101, +0.181]<br>c [+0.098, +0.186] |
| V1 random (r2-r4) | K5 (50 ms) | third 3 | 14,154 / 498 / 72 | 146.3 | 134.0 | +0.161<br>d [+0.108, +0.217]<br>c [+0.088, +0.239] |
| V1 random (r2-r4) | K5 (50 ms) | beyond one spacing | 1,416 / 252 / 51 | 301.4 | 281.3 | +0.128<br>d [+0.015, +0.230]<br>c [+0.006, +0.234] |
| V1 random (r2-r4) | K10 (100 ms) | third 1 | 30,102 / 504 / 72 | 139.8 | 131.0 | +0.122<br>d [+0.090, +0.154]<br>c [+0.089, +0.159] |
| V1 random (r2-r4) | K10 (100 ms) | third 2 | 0 | - | - | - |
| V1 random (r2-r4) | K10 (100 ms) | third 3 | 28,356 / 501 / 72 | 162.5 | 145.0 | +0.204<br>d [+0.161, +0.244]<br>c [+0.146, +0.263] |
| V1 random (r2-r4) | K10 (100 ms) | beyond one spacing | 2,065 / 252 / 47 | 330.8 | 271.4 | +0.326<br>d [+0.199, +0.428]<br>c [+0.121, +0.437] |
| V1 random (r2-r4) | K20 (200 ms) | third 1 | 15,210 / 498 / 72 | 146.8 | 138.9 | +0.105<br>d [+0.061, +0.147]<br>c [+0.057, +0.149] |
| V1 random (r2-r4) | K20 (200 ms) | third 2 | 29,082 / 502 / 72 | 174.3 | 155.1 | +0.209<br>d [+0.165, +0.250]<br>c [+0.161, +0.255] |
| V1 random (r2-r4) | K20 (200 ms) | third 3 | 14,015 / 491 / 72 | 183.1 | 158.6 | +0.249<br>d [+0.190, +0.307]<br>c [+0.164, +0.330] |
| V1 random (r2-r4) | K20 (200 ms) | beyond one spacing | 2,216 / 200 / 42 | 351.2 | 302.3 | +0.259<br>d [+0.054, +0.431]<br>c [+0.078, +0.368] |
| V1 random (r2-r4) | K50 (500 ms) | third 1 | 17,945 / 503 / 72 | 196.9 | 172.0 | +0.237<br>d [+0.170, +0.296]<br>c [+0.178, +0.302] |
| V1 random (r2-r4) | K50 (500 ms) | third 2 | 23,092 / 491 / 72 | 216.4 | 175.4 | +0.344<br>d [+0.276, +0.407]<br>c [+0.234, +0.445] |
| V1 random (r2-r4) | K50 (500 ms) | third 3 | 15,667 / 472 / 72 | 209.1 | 178.1 | +0.274<br>d [+0.211, +0.331]<br>c [+0.201, +0.340] |
| V1 random (r2-r4) | K50 (500 ms) | beyond one spacing | 3,819 / 148 / 38 | 261.6 | 242.8 | +0.138<br>d [-0.025, +0.252]<br>c [-0.031, +0.257] |
| V4 chronological: fit to 31991, score from 32312 | K2 (20 ms) | third 1 | 0 | - | - | - |
| V4 chronological: fit to 31991, score from 32312 | K2 (20 ms) | third 2 | 3,544 / 59 / 16 | 145.7 | 140.7 | +0.068<br>d [-0.031, +0.159]<br>c [-0.046, +0.242] |
| V4 chronological: fit to 31991, score from 32312 | K2 (20 ms) | third 3 | 0 | - | - | - |
| V4 chronological: fit to 31991, score from 32312 | K2 (20 ms) | beyond one spacing | 43 / 12 / 6 | 184.7 | 187.5 | -0.030<br>d [-0.676, +0.212]<br>c [-0.074, +0.269] |
| V4 chronological: fit to 31991, score from 32312 | K5 (50 ms) | third 1 | 927 / 59 / 16 | 154.7 | 150.7 | +0.051<br>d [-0.043, +0.157]<br>c [-0.032, +0.188] |
| V4 chronological: fit to 31991, score from 32312 | K5 (50 ms) | third 2 | 1,769 / 59 / 16 | 147.0 | 142.7 | +0.058<br>d [-0.055, +0.151]<br>c [-0.005, +0.161] |
| V4 chronological: fit to 31991, score from 32312 | K5 (50 ms) | third 3 | 816 / 58 / 16 | 139.6 | 123.9 | +0.212<br>d [+0.039, +0.324]<br>c [+0.089, +0.290] |
| V4 chronological: fit to 31991, score from 32312 | K5 (50 ms) | beyond one spacing | 75 / 20 / 10 | 240.6 | 228.9 | +0.094<br>d [-0.386, +0.312]<br>c [-0.005, +0.410] |
| V4 chronological: fit to 31991, score from 32312 | K10 (100 ms) | third 1 | 1,843 / 59 / 16 | 160.0 | 151.8 | +0.101<br>d [-0.003, +0.198]<br>c [+0.033, +0.183] |
| V4 chronological: fit to 31991, score from 32312 | K10 (100 ms) | third 2 | 0 | - | - | - |
| V4 chronological: fit to 31991, score from 32312 | K10 (100 ms) | third 3 | 1,644 / 58 / 16 | 154.9 | 156.9 | -0.026<br>d [-0.207, +0.139]<br>c [-0.214, +0.156] |
| V4 chronological: fit to 31991, score from 32312 | K10 (100 ms) | beyond one spacing | 100 / 20 / 10 | 212.3 | 247.6 | -0.360<br>d [-1.260, +0.352]<br>c [-0.650, +0.390] |
| V4 chronological: fit to 31991, score from 32312 | K20 (200 ms) | third 1 | 966 / 59 / 16 | 144.2 | 153.9 | -0.138<br>d [-0.340, +0.071]<br>c [-0.315, +0.118] |
| V4 chronological: fit to 31991, score from 32312 | K20 (200 ms) | third 2 | 1,685 / 58 / 16 | 173.0 | 168.1 | +0.056<br>d [-0.060, +0.168]<br>c [-0.046, +0.205] |
| V4 chronological: fit to 31991, score from 32312 | K20 (200 ms) | third 3 | 823 / 58 / 16 | 169.8 | 166.4 | +0.039<br>d [-0.111, +0.176]<br>c [-0.046, +0.218] |
| V4 chronological: fit to 31991, score from 32312 | K20 (200 ms) | beyond one spacing | 113 / 15 / 8 | 227.4 | 230.9 | -0.031<br>d [-0.619, +0.355]<br>c [-0.236, +0.350] |
| V4 chronological: fit to 31991, score from 32312 | K50 (500 ms) | third 1 | 1,069 / 57 / 16 | 189.8 | 157.7 | +0.310<br>d [-0.004, +0.539]<br>c [+0.037, +0.563] |
| V4 chronological: fit to 31991, score from 32312 | K50 (500 ms) | third 2 | 1,415 / 59 / 16 | 210.2 | 172.3 | +0.328<br>d [+0.092, +0.476]<br>c [+0.147, +0.527] |
| V4 chronological: fit to 31991, score from 32312 | K50 (500 ms) | third 3 | 857 / 58 / 16 | 214.6 | 180.7 | +0.291<br>d [+0.033, +0.460]<br>c [-0.025, +0.488] |
| V4 chronological: fit to 31991, score from 32312 | K50 (500 ms) | beyond one spacing | 246 / 12 / 8 | 266.9 | 246.2 | +0.149<br>d [-0.663, +0.625]<br>c [-0.426, +0.397] |
| V4 chronological: fit to 32310, score from 32312 | K2 (20 ms) | third 1 | 0 | - | - | - |
| V4 chronological: fit to 32310, score from 32312 | K2 (20 ms) | third 2 | 3,544 / 59 / 16 | 134.2 | 131.4 | +0.042<br>d [-0.051, +0.138]<br>c [-0.045, +0.185] |
| V4 chronological: fit to 32310, score from 32312 | K2 (20 ms) | third 3 | 0 | - | - | - |
| V4 chronological: fit to 32310, score from 32312 | K2 (20 ms) | beyond one spacing | 43 / 12 / 6 | 190.9 | 184.3 | +0.068<br>d [-0.252, +0.360]<br>c [-0.691, +0.208] |
| V4 chronological: fit to 32310, score from 32312 | K5 (50 ms) | third 1 | 927 / 59 / 16 | 148.5 | 141.9 | +0.087<br>d [+0.011, +0.176]<br>c [+0.019, +0.209] |
| V4 chronological: fit to 32310, score from 32312 | K5 (50 ms) | third 2 | 1,769 / 59 / 16 | 144.7 | 142.3 | +0.033<br>d [-0.054, +0.104]<br>c [-0.024, +0.143] |
| V4 chronological: fit to 32310, score from 32312 | K5 (50 ms) | third 3 | 816 / 58 / 16 | 134.9 | 120.6 | +0.201<br>d [+0.045, +0.326]<br>c [+0.081, +0.372] |
| V4 chronological: fit to 32310, score from 32312 | K5 (50 ms) | beyond one spacing | 75 / 20 / 10 | 246.7 | 215.3 | +0.238<br>d [+0.015, +0.435]<br>c [-0.073, +0.405] |
| V4 chronological: fit to 32310, score from 32312 | K10 (100 ms) | third 1 | 1,843 / 59 / 16 | 153.1 | 144.3 | +0.111<br>d [+0.042, +0.193]<br>c [+0.073, +0.176] |
| V4 chronological: fit to 32310, score from 32312 | K10 (100 ms) | third 2 | 0 | - | - | - |
| V4 chronological: fit to 32310, score from 32312 | K10 (100 ms) | third 3 | 1,644 / 58 / 16 | 151.9 | 143.7 | +0.105<br>d [-0.016, +0.231]<br>c [-0.041, +0.285] |
| V4 chronological: fit to 32310, score from 32312 | K10 (100 ms) | beyond one spacing | 100 / 20 / 10 | 197.3 | 193.3 | +0.040<br>d [-0.395, +0.286]<br>c [-0.099, +0.391] |
| V4 chronological: fit to 32310, score from 32312 | K20 (200 ms) | third 1 | 966 / 59 / 16 | 138.5 | 141.7 | -0.047<br>d [-0.232, +0.095]<br>c [-0.252, +0.138] |
| V4 chronological: fit to 32310, score from 32312 | K20 (200 ms) | third 2 | 1,685 / 58 / 16 | 167.7 | 159.2 | +0.100<br>d [-0.011, +0.225]<br>c [+0.003, +0.270] |
| V4 chronological: fit to 32310, score from 32312 | K20 (200 ms) | third 3 | 823 / 58 / 16 | 162.9 | 156.8 | +0.073<br>d [-0.026, +0.161]<br>c [-0.012, +0.182] |
| V4 chronological: fit to 32310, score from 32312 | K20 (200 ms) | beyond one spacing | 113 / 15 / 8 | 239.3 | 260.9 | -0.189<br>d [-0.872, +0.199]<br>c [-0.370, +0.255] |
| V4 chronological: fit to 32310, score from 32312 | K50 (500 ms) | third 1 | 1,069 / 57 / 16 | 182.8 | 152.6 | +0.303<br>d [+0.147, +0.463]<br>c [+0.140, +0.527] |
| V4 chronological: fit to 32310, score from 32312 | K50 (500 ms) | third 2 | 1,415 / 59 / 16 | 204.9 | 164.3 | +0.357<br>d [+0.167, +0.517]<br>c [+0.164, +0.573] |
| V4 chronological: fit to 32310, score from 32312 | K50 (500 ms) | third 3 | 857 / 58 / 16 | 200.1 | 176.7 | +0.220<br>d [-0.008, +0.419]<br>c [-0.044, +0.456] |
| V4 chronological: fit to 32310, score from 32312 | K50 (500 ms) | beyond one spacing | 246 / 12 / 8 | 231.4 | 236.8 | -0.047<br>d [-0.289, +0.258]<br>c [-0.334, +0.354] |

## By position between two kept values, target F

uf = frame index of row t minus that of the row of the newest visible value. Thirds of the kept spacing k: 3 uf < k; k <= 3 uf < 2k; 2k <= 3 uf < 3k; 'beyond one spacing': uf >= k (the newest kept frame before t carries no visible value). Empty strata are empty by the scored-row rule (Addendum 1 section 6).

| design | rule | position | rows / discharges / clusters | RMSE T0 | RMSE TX | TX over T0 |
|---|---|---|---|---|---|---|
| V1 random (r2-r4) | K2 (20 ms) | third 1 | 0 | - | - | - |
| V1 random (r2-r4) | K2 (20 ms) | third 2 | 60,983 / 512 / 72 | 164.6 | 155.8 | +0.104<br>d [+0.079, +0.129]<br>c [+0.063, +0.149] |
| V1 random (r2-r4) | K2 (20 ms) | third 3 | 0 | - | - | - |
| V1 random (r2-r4) | K2 (20 ms) | beyond one spacing | 1,179 / 242 / 48 | 475.5 | 465.5 | +0.042<br>d [-0.017, +0.096]<br>c [-0.016, +0.092] |
| V1 random (r2-r4) | K5 (50 ms) | third 1 | 15,338 / 510 / 72 | 174.2 | 166.0 | +0.092<br>d [+0.053, +0.129]<br>c [+0.043, +0.140] |
| V1 random (r2-r4) | K5 (50 ms) | third 2 | 30,621 / 510 / 72 | 170.4 | 158.2 | +0.138<br>d [+0.106, +0.170]<br>c [+0.090, +0.190] |
| V1 random (r2-r4) | K5 (50 ms) | third 3 | 14,436 / 502 / 72 | 193.8 | 175.2 | +0.183<br>d [+0.134, +0.228]<br>c [+0.122, +0.242] |
| V1 random (r2-r4) | K5 (50 ms) | beyond one spacing | 1,767 / 275 / 54 | 470.3 | 451.1 | +0.080<br>d [+0.027, +0.135]<br>c [+0.023, +0.144] |
| V1 random (r2-r4) | K10 (100 ms) | third 1 | 30,506 / 511 / 72 | 178.6 | 165.4 | +0.142<br>d [+0.112, +0.172]<br>c [+0.093, +0.189] |
| V1 random (r2-r4) | K10 (100 ms) | third 2 | 0 | - | - | - |
| V1 random (r2-r4) | K10 (100 ms) | third 3 | 29,220 / 508 / 72 | 204.5 | 183.1 | +0.199<br>d [+0.164, +0.231]<br>c [+0.147, +0.252] |
| V1 random (r2-r4) | K10 (100 ms) | beyond one spacing | 2,436 / 269 / 51 | 448.1 | 398.2 | +0.210<br>d [+0.123, +0.300]<br>c [+0.124, +0.309] |
| V1 random (r2-r4) | K20 (200 ms) | third 1 | 15,430 / 509 / 72 | 179.7 | 168.8 | +0.117<br>d [+0.071, +0.162]<br>c [+0.066, +0.164] |
| V1 random (r2-r4) | K20 (200 ms) | third 2 | 29,728 / 508 / 72 | 216.5 | 190.2 | +0.227<br>d [+0.188, +0.267]<br>c [+0.171, +0.288] |
| V1 random (r2-r4) | K20 (200 ms) | third 3 | 14,421 / 502 / 72 | 230.9 | 204.7 | +0.214<br>d [+0.171, +0.257]<br>c [+0.145, +0.278] |
| V1 random (r2-r4) | K20 (200 ms) | beyond one spacing | 2,583 / 215 / 45 | 449.5 | 409.8 | +0.169<br>d [+0.020, +0.295]<br>c [+0.044, +0.278] |
| V1 random (r2-r4) | K50 (500 ms) | third 1 | 18,210 / 511 / 72 | 228.4 | 205.3 | +0.192<br>d [+0.145, +0.235]<br>c [+0.135, +0.248] |
| V1 random (r2-r4) | K50 (500 ms) | third 2 | 23,517 / 503 / 72 | 260.9 | 219.8 | +0.290<br>d [+0.234, +0.343]<br>c [+0.217, +0.363] |
| V1 random (r2-r4) | K50 (500 ms) | third 3 | 16,285 / 490 / 72 | 251.7 | 215.2 | +0.269<br>d [+0.209, +0.326]<br>c [+0.185, +0.339] |
| V1 random (r2-r4) | K50 (500 ms) | beyond one spacing | 4,150 / 161 / 39 | 336.0 | 310.9 | +0.144<br>d [+0.044, +0.228]<br>c [+0.030, +0.246] |
| V4 chronological: fit to 31991, score from 32312 | K2 (20 ms) | third 1 | 0 | - | - | - |
| V4 chronological: fit to 31991, score from 32312 | K2 (20 ms) | third 2 | 3,675 / 60 / 16 | 245.2 | 242.5 | +0.022<br>d [-0.023, +0.069]<br>c [-0.031, +0.057] |
| V4 chronological: fit to 31991, score from 32312 | K2 (20 ms) | third 3 | 0 | - | - | - |
| V4 chronological: fit to 31991, score from 32312 | K2 (20 ms) | beyond one spacing | 63 / 12 / 6 | 481.7 | 495.6 | -0.059<br>d [-0.205, +0.217]<br>c [-0.121, +0.182] |
| V4 chronological: fit to 31991, score from 32312 | K5 (50 ms) | third 1 | 943 / 60 / 16 | 250.7 | 250.5 | +0.002<br>d [-0.084, +0.077]<br>c [-0.035, +0.072] |
| V4 chronological: fit to 31991, score from 32312 | K5 (50 ms) | third 2 | 1,869 / 60 / 16 | 252.1 | 242.1 | +0.078<br>d [+0.038, +0.128]<br>c [+0.046, +0.115] |
| V4 chronological: fit to 31991, score from 32312 | K5 (50 ms) | third 3 | 830 / 58 / 16 | 268.2 | 254.5 | +0.100<br>d [+0.020, +0.201]<br>c [+0.039, +0.222] |
| V4 chronological: fit to 31991, score from 32312 | K5 (50 ms) | beyond one spacing | 96 / 21 / 11 | 463.1 | 471.1 | -0.035<br>d [-0.230, +0.112]<br>c [-0.256, +0.152] |
| V4 chronological: fit to 31991, score from 32312 | K10 (100 ms) | third 1 | 1,871 / 60 / 16 | 253.7 | 246.3 | +0.058<br>d [+0.019, +0.101]<br>c [+0.007, +0.089] |
| V4 chronological: fit to 31991, score from 32312 | K10 (100 ms) | third 2 | 0 | - | - | - |
| V4 chronological: fit to 31991, score from 32312 | K10 (100 ms) | third 3 | 1,735 / 59 / 16 | 278.5 | 269.4 | +0.064<br>d [+0.005, +0.130]<br>c [+0.024, +0.152] |
| V4 chronological: fit to 31991, score from 32312 | K10 (100 ms) | beyond one spacing | 132 / 22 / 11 | 336.7 | 297.5 | +0.219<br>d [-0.006, +0.354]<br>c [+0.112, +0.316] |
| V4 chronological: fit to 31991, score from 32312 | K20 (200 ms) | third 1 | 978 / 60 / 16 | 239.2 | 244.7 | -0.047<br>d [-0.114, +0.004]<br>c [-0.144, -0.007] |
| V4 chronological: fit to 31991, score from 32312 | K20 (200 ms) | third 2 | 1,765 / 59 / 16 | 250.7 | 244.0 | +0.053<br>d [-0.029, +0.139]<br>c [+0.006, +0.161] |
| V4 chronological: fit to 31991, score from 32312 | K20 (200 ms) | third 3 | 846 / 59 / 16 | 325.0 | 331.4 | -0.040<br>d [-0.122, +0.038]<br>c [-0.118, +0.084] |
| V4 chronological: fit to 31991, score from 32312 | K20 (200 ms) | beyond one spacing | 149 / 18 / 10 | 388.2 | 338.8 | +0.238<br>d [+0.061, +0.358]<br>c [+0.087, +0.404] |
| V4 chronological: fit to 31991, score from 32312 | K50 (500 ms) | third 1 | 1,091 / 60 / 16 | 291.5 | 272.2 | +0.128<br>d [-0.049, +0.338]<br>c [-0.046, +0.283] |
| V4 chronological: fit to 31991, score from 32312 | K50 (500 ms) | third 2 | 1,431 / 59 / 16 | 299.3 | 264.2 | +0.221<br>d [+0.065, +0.349]<br>c [+0.087, +0.322] |
| V4 chronological: fit to 31991, score from 32312 | K50 (500 ms) | third 3 | 944 / 59 / 16 | 314.1 | 295.9 | +0.113<br>d [-0.028, +0.232]<br>c [-0.060, +0.234] |
| V4 chronological: fit to 31991, score from 32312 | K50 (500 ms) | beyond one spacing | 272 / 12 / 8 | 417.5 | 389.4 | +0.130<br>d [-0.121, +0.323]<br>c [-0.009, +0.283] |
| V4 chronological: fit to 32310, score from 32312 | K2 (20 ms) | third 1 | 0 | - | - | - |
| V4 chronological: fit to 32310, score from 32312 | K2 (20 ms) | third 2 | 3,675 / 60 / 16 | 241.9 | 233.1 | +0.072<br>d [+0.036, +0.110]<br>c [+0.034, +0.102] |
| V4 chronological: fit to 32310, score from 32312 | K2 (20 ms) | third 3 | 0 | - | - | - |
| V4 chronological: fit to 32310, score from 32312 | K2 (20 ms) | beyond one spacing | 63 / 12 / 6 | 467.2 | 469.5 | -0.010<br>d [-0.258, +0.258]<br>c [-0.122, +0.195] |
| V4 chronological: fit to 32310, score from 32312 | K5 (50 ms) | third 1 | 943 / 60 / 16 | 252.2 | 242.3 | +0.077<br>d [+0.034, +0.133]<br>c [+0.046, +0.134] |
| V4 chronological: fit to 32310, score from 32312 | K5 (50 ms) | third 2 | 1,869 / 60 / 16 | 247.3 | 236.8 | +0.083<br>d [+0.044, +0.139]<br>c [+0.048, +0.161] |
| V4 chronological: fit to 32310, score from 32312 | K5 (50 ms) | third 3 | 830 / 58 / 16 | 260.8 | 253.0 | +0.059<br>d [-0.014, +0.138]<br>c [+0.001, +0.167] |
| V4 chronological: fit to 32310, score from 32312 | K5 (50 ms) | beyond one spacing | 96 / 21 / 11 | 465.6 | 447.0 | +0.079<br>d [-0.181, +0.176]<br>c [-0.282, +0.146] |
| V4 chronological: fit to 32310, score from 32312 | K10 (100 ms) | third 1 | 1,871 / 60 / 16 | 251.4 | 245.9 | +0.043<br>d [-0.003, +0.088]<br>c [+0.010, +0.099] |
| V4 chronological: fit to 32310, score from 32312 | K10 (100 ms) | third 2 | 0 | - | - | - |
| V4 chronological: fit to 32310, score from 32312 | K10 (100 ms) | third 3 | 1,735 / 59 / 16 | 275.9 | 268.8 | +0.051<br>d [+0.000, +0.098]<br>c [+0.012, +0.111] |
| V4 chronological: fit to 32310, score from 32312 | K10 (100 ms) | beyond one spacing | 132 / 22 / 11 | 326.3 | 338.5 | -0.076<br>d [-0.365, +0.098]<br>c [-0.337, +0.202] |
| V4 chronological: fit to 32310, score from 32312 | K20 (200 ms) | third 1 | 978 / 60 / 16 | 238.7 | 242.2 | -0.029<br>d [-0.109, +0.023]<br>c [-0.093, +0.023] |
| V4 chronological: fit to 32310, score from 32312 | K20 (200 ms) | third 2 | 1,765 / 59 / 16 | 250.7 | 239.5 | +0.088<br>d [+0.008, +0.177]<br>c [+0.025, +0.225] |
| V4 chronological: fit to 32310, score from 32312 | K20 (200 ms) | third 3 | 846 / 59 / 16 | 323.7 | 318.3 | +0.033<br>d [-0.045, +0.101]<br>c [-0.034, +0.146] |
| V4 chronological: fit to 32310, score from 32312 | K20 (200 ms) | beyond one spacing | 149 / 18 / 10 | 345.8 | 376.0 | -0.183<br>d [-0.656, +0.083]<br>c [-0.402, +0.170] |
| V4 chronological: fit to 32310, score from 32312 | K50 (500 ms) | third 1 | 1,091 / 60 / 16 | 274.6 | 270.1 | +0.033<br>d [-0.051, +0.122]<br>c [-0.054, +0.150] |
| V4 chronological: fit to 32310, score from 32312 | K50 (500 ms) | third 2 | 1,431 / 59 / 16 | 286.2 | 262.2 | +0.160<br>d [+0.030, +0.283]<br>c [+0.035, +0.296] |
| V4 chronological: fit to 32310, score from 32312 | K50 (500 ms) | third 3 | 944 / 59 / 16 | 305.4 | 296.1 | +0.060<br>d [-0.099, +0.197]<br>c [-0.136, +0.206] |
| V4 chronological: fit to 32310, score from 32312 | K50 (500 ms) | beyond one spacing | 272 / 12 / 8 | 397.5 | 392.6 | +0.025<br>d [-0.284, +0.175]<br>c [-0.244, +0.193] |

## Age of the newest visible value on the common set

| target set | rule | rows | age 10 % (ms) | median | 90 % | third 1 | third 2 | third 3 | beyond one spacing |
|---|---|---|---|---|---|---|---|---|---|
| F | K1 | 62,162 | 10 | 10 | 10 | - | - | - | - |
| F | K2 | 62,162 | 10 | 10 | 10 | 0.000 | 0.981 | 0.000 | 0.019 |
| F | K5 | 62,162 | 10 | 30 | 40 | 0.247 | 0.493 | 0.232 | 0.028 |
| F | K10 | 62,162 | 10 | 70 | 90 | 0.491 | 0.000 | 0.470 | 0.039 |
| F | K20 | 62,162 | 10 | 110 | 190 | 0.248 | 0.478 | 0.232 | 0.042 |
| F | K50 | 62,162 | 70 | 270 | 470 | 0.293 | 0.378 | 0.262 | 0.067 |
| L5 | K1 | 60,523 | 10 | 10 | 10 | - | - | - | - |
| L5 | K2 | 60,523 | 10 | 10 | 10 | 0.000 | 0.986 | 0.000 | 0.014 |
| L5 | K5 | 60,523 | 10 | 30 | 40 | 0.250 | 0.493 | 0.234 | 0.023 |
| L5 | K10 | 60,523 | 10 | 70 | 90 | 0.497 | 0.000 | 0.469 | 0.034 |
| L5 | K20 | 60,523 | 10 | 110 | 190 | 0.251 | 0.481 | 0.232 | 0.037 |
| L5 | K50 | 60,523 | 70 | 270 | 470 | 0.296 | 0.382 | 0.259 | 0.063 |

## Counts

- D1 rows 222,672; scored rows 88,749; scored rows with the eight fast scalars 86,449; complete under all six rules 62,162; common set F 62,162, L5 60,523 (scored rows with the target: F 88,749, L5 85,738).
- Per rule (scored rows): K1: visible 88,749, fewer than three 640, complete 86,000, dropped by a3 alone 438; K2: visible 88,742, fewer than three 1,205, complete 85,963, dropped by a3 alone 442; K5: visible 88,656, fewer than three 2,548, complete 85,618, dropped by a3 alone 460; K10: visible 88,432, fewer than three 4,954, complete 82,985, dropped by a3 alone 465; K20: visible 88,004, fewer than three 9,853, complete 77,710, dropped by a3 alone 431; K50: visible 86,728, fewer than three 24,182, complete 62,213, dropped by a3 alone 308.

| design | common F | common L5 | fit rows per unit F | fit rows per unit L5 |
|---|---|---|---|---|
| V1 random (r2-r4) | 62,162 / 512 / 72 | 60,523 / 507 / 72 | 47,287-51,735 | 45,984-50,432 |
| V2 cluster-blocked (3 assignments) | 62,162 / 512 / 72 | 60,523 / 507 / 72 | 43,828-53,781 | 42,459-52,379 |
| V3 rolling origin (fifths 2-5 pooled) | 43,352 / 401 / 57 | 41,985 / 397 / 57 | 18,810-55,637 | 18,538-54,317 |
| V4 chronological: fit to 31991, score from 32312 | 3,738 / 60 / 16 | 3,587 / 59 / 16 | 49,534-49,534 | 48,456-48,456 |
| V4 chronological: fit to 32310, score from 32312 | 3,738 / 60 / 16 | 3,587 / 59 / 16 | 58,424-58,424 | 56,936-56,936 |

Cells: rows / discharges / clusters scored.

Readout run time 58 s.

## Limits (registration section 9, unchanged)

- Exploratory on used data.
- The rate is lowered by us after the fact. The withheld frames were taken with the beam on; an operation that takes fewer CES frames would change the beam schedule and with it the plasma between the frames.
- The kept values are the offline product.
- One estimator per rule; the phase of the kept grid is one (frame index a multiple of k).
- The following level is built from reported values.
