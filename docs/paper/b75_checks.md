# B.75 checks

Registration: `ces_prediction/experiments/PREREGISTRATION_B75.md` section 9; implementation: `ces_prediction/experiments/PREREGISTRATION_B75_ADDENDUM1.md` section 8. Run before any readout. No tolerance was changed after a run.

## 1. G under D10 against the cached causal GP of `data/.b64/base.npz`

D1 rows 222,672. Largest deviation 0.0e+00 eV; bit-identical on 222,672 rows (the 0 rows without a visible value are undefined in both: True). Beside it: H 0.0e+00 eV, age 0.0e+00 ms, M5 0.0e+00 eV; the five history scalars against `rows.npz`: bit-identical True, same undefined rows True.

Check 1 passes: **True**

## 2. T0 and TX8 under D10 against the stored estimates of B.73

Fit rows: B.73's complete set (Addendum 1 section 4). Test rows without it keep G, as in B.73.

| unit / learner / target | same test rows | largest deviation (eV) | fit rows (B.73) | iterations (B.73) | same complete mask |
|---|---|---|---|---|---|
| V1_r2_k0 / T0 / F | True | 0.0e+00 | 179,874 (179,874) | 49 (49) | True |
| V1_r2_k0 / T0 / L5 | True | 0.0e+00 | 174,151 (174,151) | 100 (100) | True |
| V1_r2_k0 / TX8 / F | True | 0.0e+00 | 179,874 (179,874) | 100 (100) | True |
| V1_r2_k0 / TX8 / L5 | True | 0.0e+00 | 174,151 (174,151) | 100 (100) | True |
| V4a / T0 / F | True | 0.0e+00 | 164,356 (164,356) | 84 (84) | True |
| V4a / T0 / L5 | True | 0.0e+00 | 160,486 (160,486) | 87 (87) | True |
| V4a / TX8 / F | True | 0.0e+00 | 164,356 (164,356) | 100 (100) | True |
| V4a / TX8 / L5 | True | 0.0e+00 | 160,486 (160,486) | 100 (100) | True |

Check 2 passes: **True**

## 3. Visibility

Unit V1_r2_k0. For each row, a copy of its file in which every T_i value with stamp above the row's limit + 3 ms is removed, or raised by 1000 eV; the ten history scalars, G, M5, the six anchor scalars and the anchor row recomputed and compared bit for bit with the main pass. Note: the eight fast scalars and the running means m(i) take no T_i argument (they are functions of the fast channels and the fit discharges only) and are not compared; the anchor scalars are recomputed from the recomputed anchor row; M5 is compared too.

| rule | rows (candidates) | removed: rows identical (values changed) | raised: rows identical (values changed) |
|---|---|---|---|
| D650 | 200 (27,566) | 200 (60,050) | 200 (60,050) |
| S | 200 (25,231) | 200 (57,362) | 200 (57,362) |

Check 3 passes: **True**

## 4. Ages under S against the reviewer's check C12

Population: grid rows other than a file's first and last, finite T_i as delivered (no cut, no held removal): 226,991 rows, 155,576 with an absent step earlier in the block; share without 0.315 (reviewer: 0.315). Age = stamp of the row minus the stamp of the last row before the most recent absent step that ends at or before it.

| quantity | ours | reviewer (C12) |
|---|---|---|
| median (s) | 0.650 | 0.65 |
| share 0-0.1 s | 0.071 | 0.071 |
| share 0.1-0.3 s | 0.164 | 0.166 |
| share 0.3-1 s | 0.500 | 0.498 |
| share above 1 s | 0.265 | 0.266 |

Check 4 passes: **True**

The band shares differ from C12 by up to 0.0023. Cause: the grid stamps are float32, and 1,330 ages lie within 1e-5 s of 0.1 s, 1,245 ages lie within 1e-5 s of 0.3 s, 1,043 ages lie within 1e-5 s of 1.0 s; such an age falls on either side of the edge depending on the rounding of the stamps. With the float64 stamps of the CSV text (155,576 rows): median 0.650 s, bands 0-0.1 s 0.0712, 0.1-0.3 s 0.1657, 0.3-1 s 0.4976, above 1 s 0.2654; C12's six printed bins (0-0.05, 0.05-0.1, 0.1-0.3, 0.3-1, 1-3, above 3 s) come out as 0.030, 0.041, 0.166, 0.498, 0.261, 0.005, whose sums are C12's four bands.

On the D1 rows (cut, held-free): 153,382 of 222,672 have a visible value under S (share without 0.311). Age of the limit: median 0.650 s, bands 0-0.1 s 0.071, 0.1-0.3 s 0.164, 0.3-1 s 0.500, above 1 s 0.265. Age of the newest visible value: median 0.650 s, bands 0-0.1 s 0.070, 0.1-0.3 s 0.164, 0.3-1 s 0.500, above 1 s 0.266.

## 5. Units

Every B.73 unit has the test rows stored by B.73: **True** (36 units). V3c units: V3c_q1_s1: fit 127, test 127, fit discharges shared with V3 27; V3c_q1_s2: fit 127, test 127, fit discharges shared with V3 28; V3c_q1_s3: fit 127, test 127, fit discharges shared with V3 31; V3c_q2_s1: fit 254, test 127, fit discharges shared with V3 126; V3c_q2_s2: fit 254, test 127, fit discharges shared with V3 138; V3c_q2_s3: fit 254, test 127, fit discharges shared with V3 126; V3c_q3_s1: fit 381, test 127, fit discharges shared with V3 290; V3c_q3_s2: fit 381, test 127, fit discharges shared with V3 287; V3c_q3_s3: fit 381, test 127, fit discharges shared with V3 284.

## Row sets

D1 rows 222,672; rows with the eight fast scalars 217,459.

| rule | rows with a visible value | history and G complete | anchor means defined | complete | dropped by a3 alone |
|---|---|---|---|---|---|
| D10 | 222,672 | 221,357 | 220,413 | 216,337 | 1,091 |
| D50 | 220,289 | 218,957 | 218,023 | 216,143 | 1,211 |
| D100 | 217,372 | 216,044 | 215,115 | 213,678 | 1,304 |
| D300 | 206,463 | 205,203 | 204,444 | 202,990 | 1,272 |
| D650 | 187,697 | 186,255 | 185,523 | 184,129 | 1,154 |
| D1000 | 168,889 | 167,511 | 166,926 | 165,651 | 984 |
| S | 153,382 | 153,347 | 153,332 | 152,256 | 1,071 |

Common set: F 165,418, L5 160,343 (rows with the target: F 222,672, L5 215,369). S set: F 152,203, L5 147,837. Rows complete under S but not under D10: 53. Rows without a visible value under S: 69,290.
