# B.77 checks

Registration: `ces_prediction/experiments/PREREGISTRATION_B77.md` section 9; implementation: `ces_prediction/experiments/PREREGISTRATION_B77_ADDENDUM1.md` section 11. Run before any readout; no score against a target is computed here. No tolerance was changed after a run.

## 3. Inputs and target (added)

Row set 209,432 rows, 606 discharges, 85 shot clusters. L5 recomputed from the grid bit-identical to `rows.npz`: True. H equals the grid value of the anchor row: True. Fewest earlier values kept on a row: 3. L5 above zero on every row: True. MED3 defined on 209,432 rows. The reported y differs from the grid value by at most 2.4e-04 eV. Anchor row = row t - 1 on 98.7 % of the rows.

Check 3 passes: **True**

## 1b. This batch's code on B.75's common set: T0 and TX against B.75's stored D10 L5 estimates (added)

| unit / learner | same test rows | bit-identical | largest deviation (eV) | fit rows (B.75) | iterations (B.75) |
|---|---|---|---|---|---|
| V1_r2_k0 / T0 | True | True | 0.0e+00 | 133,674 (133,674) | 69 (69) |
| V1_r2_k0 / TX | True | True | 0.0e+00 | 133,674 (133,674) | 100 (100) |
| V4a / T0 | True | True | 0.0e+00 | 127,270 (127,270) | 89 (89) |
| V4a / TX | True | True | 0.0e+00 | 127,270 (127,270) | 100 (100) |

Check 1b passes: **True**

## 1a. T0 and TX of this batch (fitted on the B.77 row set) against B.75's D10 L5 estimates on the rows both have

reported; the row sets differ (B.75: common set; B.77: complete under D10), so equality is not required.

| learner | rows both have (all units) | share identical | median abs. deviation (eV) | 90th percentile (eV) | rows above 100 eV | rows above 300 eV | largest (eV) | every B.75 test row in B.77 |
|---|---|---|---|---|---|---|---|---|
| T0 | 1,094,549 | 0.000 | 2.80 | 17.83 | 8,251 | 344 | 980.9 | True |
| TX | 1,094,549 | 0.000 | 4.70 | 28.70 | 15,438 | 643 | 996.7 | True |

The fit rows differ (B.77 adds about 49,000 rows that are complete under D10 but not under every longer delay). The largest deviations lie on rows with a high and scattered history (values about 2 keV, standard deviation of the last ten values 330-690 eV), where the trees of the two fits differ. Check 1b shows that this code reproduces B.75 bit for bit on B.75's rows. Per unit: `data/.b77/checks/compare.json`.

## 1c. C0 with the four frame scalars removed against T0

| unit | columns | same test rows | bit-identical | largest deviation (eV) | iterations (T0) |
|---|---|---|---|---|---|
| V1_r2_k0 | 10 | True | True | 0.0e+00 | 100 (100) |
| V2_s1_k0 | 10 | True | True | 0.0e+00 | 100 (100) |
| V3_q1 | 10 | True | True | 0.0e+00 | 53 (53) |
| V4a | 10 | True | True | 0.0e+00 | 76 (76) |
| V4b | 10 | True | True | 0.0e+00 | 100 (100) |

Check 1c passes: **True**

## 2. No input of C0 or CX contains a value of a row later than t

Unit V1_r2_k0, 200 test rows drawn from 35,740. The C0 and CX models refitted in the check process equal the stored ones bit for bit: {'C0': True, 'CX': True}. The file-local recomputation of the fourteen fast scalars (B.73/B.75 code on a one-file copy, the unit's standardisation passed in) reproduces the main pass on 200 of 200 rows. Compared bit for bit per row: the ten history scalars, G, the four frame scalars, MED3, the fourteen fast scalars, the anchor row, and the C0 and CX estimates.

| modification | rows | inputs identical | estimates identical | frame scalars changed | fast scalars changed | T_i values modified | fast rows modified |
|---|---|---|---|---|---|---|---|
| removed | 200 | 200 | 200 | 0 | 0 | 52,384 | 0 |
| raised | 200 | 200 | 200 | 0 | 0 | 52,384 | 0 |
| changed with fast | 200 | 200 | 200 | 0 | 0 | 53,813 | 55,314 |
| control: y of row t raised | 200 | 0 | 1 | 200 | 0 | 0 | 0 |
| control: fast of row t x 1.5 | 200 | 0 | 0 | 0 | 200 | 0 | 0 |

The first three rows modify rows after t only (removed; raised by 1000 eV; T_i multiplied by factors in [0.5, 1.5], half of the missing values given a value, and the fast channels multiplied by factors in [0.5, 1.5]). The last two rows are the positive control (row t itself modified): the comparison detects a change there.

Check 2 passes: **True**; positive control: **True**
