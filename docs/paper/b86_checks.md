# B.86 checks

Registration: `ces_prediction/experiments/PREREGISTRATION_B86.md` section 6; implementation: `ces_prediction/experiments/PREREGISTRATION_B86_ADDENDUM1.md` section 9. Each check was run once, before any readout; no error, skill or interval was computed by a check. No tolerance was changed after a run.

## 1. Band E against B.76

Rows: this build 372,094, B.76 `data/.b76/rows.npz` 372,094 (same order required).

| field | identical bit for bit | dtype |
|---|---|---|
| shot | True | int32 |
| c | True | int16 |
| craw | True | int16 |
| R | True | float64 |
| rho | True | float64 |
| j | True | int32 |
| t | True | float64 |
| y | True | float64 |
| sig | True | float64 |
| prevgood | True | bool |
| te_t | True | float64 |
| ne_t | True | float64 |
| dedge_t | True | float64 |
| dband | True | float64 |
| nts | True | int16 |
| nlas | True | int16 |
| L5 | True | float64 |
| nL5 | True | int8 |
| D5_H | True | float64 |
| D5_age | True | float64 |
| D5_M5 | True | float64 |
| D5_G | True | float64 |
| D5_th | True | float64 |
| D5_anc | True | int32 |
| D5_nvis | True | int32 |
| D5_te_a | True | float64 |
| D5_ne_a | True | float64 |
| D5_nts_a | True | int16 |
| D100_H | True | float64 |
| D100_age | True | float64 |
| D100_M5 | True | float64 |
| D100_G | True | float64 |
| D100_th | True | float64 |
| D100_anc | True | int32 |
| D100_nvis | True | int32 |
| D100_te_a | True | float64 |
| D100_ne_a | True | float64 |
| D100_nts_a | True | int16 |

T0 and TX of unit M1_s1_k0 under D100, target L5, refitted by this batch's fit code on this batch's band-E rows with B.76's L5 common set and unit, against B.76's stored estimates:

| learner | fit rows (B.76) | test rows | row indices identical | estimates identical | iterations (B.76) |
|---|---|---|---|---|---|
| T0 | 64,292 (64,292) | 17,068 | True | True | 100 (100) |
| TX | 64,292 (64,292) | 17,068 | True | True | 100 (100) |

Check passes: **True**

## 2. Visibility (band B4, D100)

Unit M1_s1_k0 of band B4, 200 rows drawn from 133,981 test rows of the L5 common set. For each, a copy of its shot in which every T_i value at a frame with nominal time above the row's limit + 0.5 ms (all chords) is raised by 1000 eV; good mask and block start recomputed; anchor frame, number of visible values, H, age, M5, G, the five scalars of the last ten values and the nine Thomson scalars recomputed and compared bit for bit. Rows identical: 200 of 200; T_i values raised: 967,724.

Check passes: **True**

## 3. References that wait

200 rows drawn from the 2,799,636 value-of-the-frame rows of all bands (by band: E 9, B2 22, B3 34, B4 55, B5 80). For each, the T_i value of the row's chord at frame t raised by 1000 eV in a copy of the shot, good mask recomputed, PCHIP and offline GP recomputed with frame t left out: identical bit for bit on 200 of 200.

Check passes: **True**

## 4. Rows

For every band, design, rule and target: scored rows and shots; every estimator (H, G, T0, TX, O; for F also PCHIP and the offline GP on the value-of-the-frame rows) finite on every scored row, every scored row estimated exactly once by T0 and TX; M1 tests every design shot once with fit and test disjoint; M3 fits only shots before its test shots.

| band | design shots | M1 units | M3 units | cell | rows | shots | value-of-frame rows (left out) | all finite |
|---|---|---|---|---|---|---|---|---|
| E | 2,674 | True | True | M1|F|D5 | 165,331 | 2,674 | 161,229 (4,102) | True |
| E | 2,674 | True | True | M1|L5|D5 | 145,048 | 2,517 | - | True |
| E | 2,674 | True | True | M1|L5|D100 | 145,048 | 2,517 | - | True |
| E | 2,674 | True | True | M3|L5|D5 | 126,806 | 2,017 | - | True |
| E | 2,674 | True | True | M3|L5|D100 | 126,806 | 2,017 | - | True |
| B2 | 2,754 | True | True | M1|F|D5 | 266,107 | 2,754 | 260,567 (5,540) | True |
| B2 | 2,754 | True | True | M1|L5|D5 | 239,297 | 2,633 | - | True |
| B2 | 2,754 | True | True | M1|L5|D100 | 239,297 | 2,633 | - | True |
| B2 | 2,754 | True | True | M3|L5|D5 | 219,162 | 2,117 | - | True |
| B2 | 2,754 | True | True | M3|L5|D100 | 219,162 | 2,117 | - | True |
| B3 | 2,874 | True | True | M1|F|D5 | 529,197 | 2,874 | 520,191 (9,006) | True |
| B3 | 2,874 | True | True | M1|L5|D5 | 485,836 | 2,803 | - | True |
| B3 | 2,874 | True | True | M1|L5|D100 | 485,836 | 2,803 | - | True |
| B3 | 2,874 | True | True | M3|L5|D5 | 460,942 | 2,241 | - | True |
| B3 | 2,874 | True | True | M3|L5|D100 | 460,942 | 2,241 | - | True |
| B4 | 2,765 | True | True | M1|F|D5 | 730,888 | 2,765 | 719,777 (11,111) | True |
| B4 | 2,765 | True | True | M1|L5|D5 | 678,409 | 2,697 | - | True |
| B4 | 2,765 | True | True | M1|L5|D100 | 678,409 | 2,697 | - | True |
| B4 | 2,765 | True | True | M3|L5|D5 | 667,078 | 2,165 | - | True |
| B4 | 2,765 | True | True | M3|L5|D100 | 667,078 | 2,165 | - | True |
| B5 | 2,865 | True | True | M1|F|D5 | 1,153,562 | 2,865 | 1,137,872 (15,690) | True |
| B5 | 2,865 | True | True | M1|L5|D5 | 1,074,935 | 2,789 | - | True |
| B5 | 2,865 | True | True | M1|L5|D100 | 1,074,935 | 2,789 | - | True |
| B5 | 2,865 | True | True | M3|L5|D5 | 1,053,840 | 2,250 | - | True |
| B5 | 2,865 | True | True | M3|L5|D100 | 1,053,840 | 2,250 | - | True |

Check passes: **True**
