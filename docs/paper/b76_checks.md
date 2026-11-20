# B.76 checks

Registration: `ces_prediction/experiments/PREREGISTRATION_B76.md` section 12; implementation: `ces_prediction/experiments/PREREGISTRATION_B76_ADDENDUM1.md` section 9. Run before any readout. No tolerance was changed after a run.

## 1. H under D5 against the earlier MAST row tables

Earlier tables `data/mast/.ped_rows_T_i.npz` (odd-numbered shots) and `.ped_conf_rows_T_i.npz` (even-numbered), matched by shot, block frame and chord radius (float32). Only rows whose frame j - 1 is good are compared (the earlier build hid every odd frame; here no frame is hidden).

| table | rows in the table | rows in both builds | of them with frame j - 1 good | anchor = frame j - 1 on all of them |
|---|---|---|---|---|
| .ped_rows_T_i.npz | 2,301,107 | 80,192 | 80,192 | True |
| .ped_conf_rows_T_i.npz | 2,326,823 | 81,269 | 81,269 | True |

Compared rows 161,461; float32(H) equal to `y1` on 161,461. Largest deviation: 0 eV in float32; 0 eV between this build's float64 value and the stored float32 value (rounding of the stored table).

Diagnostics (no pass rule):

| quantity | both finite | finite pattern same (rows) | float32 equal | largest deviation |
|---|---|---|---|---|
| T_e(t) vs te0 | 161,461 | 161,461 | 161,365 | 70.2 |
| n_e(t) vs ne0 | 161,461 | 161,461 | 161,365 | 0.678 |
| band distance vs d_edge | 161,461 | 161,461 | 161,375 | 0.00933 |
| normalised radius vs rho | 161,461 | 161,461 | 161,461 | 0 |
| frame time vs t0 | 161,461 | 161,461 | 161,461 | 0 |

Cause of the T_e differences (diagnostic added after check 1 ran, before any fit): of the 96 rows where T_e(t) differs from `te0`, on 96 a laser time lies on the other side of a window edge when the window is formed from the earlier build's float32 frame time `t0` instead of the float64 frame time used here. The band distance differs on 86 rows; on 86 of them a laser time changes side of the ±7.5 ms edge window in the same way.

Check 1 passes: **True**

## 2. Visibility

Unit M1_s1_k0, rule D100, 200 rows drawn from 17,068 test rows of the L5 common set. For each row a copy of its shot in which every T_i value at a frame with nominal time above the row's limit + 0.5 ms (all chords) is removed or raised by 1000 eV; good mask and block start recomputed; anchor frame, number of visible values, H, age, M5, G, the five scalars of the last ten values, T_e and n_e at the anchor and the D-alpha anchor scalars recomputed and compared bit for bit with the main pass.

| modification | rows identical | T_i values changed |
|---|---|---|
| removed | 200 of 200 | 771,842 |
| raised | 200 of 200 | 771,842 |

Check 2 passes: **True**

## 3. Design units

| design | every shot tested once | fit and test disjoint | sessions whole | test shots per fold |
|---|---|---|---|---|
| M1 seed 1 | True | True | True | [394, 394, 393, 393, 393] |
| M1 seed 2 | True | True | True | [394, 394, 393, 393, 393] |
| M1 seed 3 | True | True | True | [394, 394, 393, 393, 393] |
| M2 seed 1 | True | True | True | [392, 388, 396, 400, 391] |
| M2 seed 2 | True | True | True | [400, 392, 398, 389, 388] |
| M2 seed 3 | True | True | True | [399, 386, 395, 395, 392] |

| unit | fit shots | test shots | last fit shot | first test shot | fit before test |
|---|---|---|---|---|---|
| M3_q1 | 394 | 394 | 20930 | 20931 | True |
| M3_q2 | 788 | 393 | 23811 | 23812 | True |
| M3_q3 | 1181 | 393 | 27249 | 27250 | True |
| M3_q4 | 1574 | 393 | 29089 | 29090 | True |
| M4 | 1574 | 393 | 29089 | 29090 | True |

M4 has the fit and test shots of M3_q4: True.

Check 3 passes: **True**

## 4. Thomson value at frame t from the window of frame t only

2,000 rows drawn from the F common set (96,449); T_e(t) and n_e(t) recomputed from the extra file with only the laser times in [t - 2.5, t + 2.5) ms: identical bit for bit on 2,000 (T_e) and 2,000 (n_e) rows; largest T_e deviation 0 eV.

Laser samples finite at the chord in the window of frame t, share of rows:

| set | rows | one | two | three or more | laser times in the window: 0 / 1 / 2 / 3 / 4 |
|---|---|---|---|---|---|
| common F | 96,449 | 0.814 | 0.186 | 0.000 | 0.000 / 0.810 / 0.190 / 0.000 / 0.000 |
| common L5 | 81,360 | 0.812 | 0.188 | 0.000 | 0.000 / 0.811 / 0.189 / 0.000 / 0.000 |

Check 4 passes: **True**
