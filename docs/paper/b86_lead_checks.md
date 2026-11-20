# B.86: recomputation by the session lead

Code: `ces_prediction/experiments/b86_mast_radius/lead_checks/verify_b86.py` (no import from the batch).

## A. Targets and history values from the value arrays

266 shots drawn at random (seed 8601), 400,657 stored rows. L5 = median of the good values of the chord at frames j+1 to j+5 (at least three); the newest visible value under D5 is the newest good value at frames up to j-1, under D100 at frames up to j-20; age = 5 ms times the frame distance.

| quantity | rows that differ from the stored value |
|---|---|
| L5 | 0 |
| D5_H | 0 |
| D5_age | 0 |
| D100_H | 0 |
| D100_age | 0 |

## B. Statement values from the stored estimates

| band | quantity | own code | readout of the batch | equal to 1e-9 |
|---|---|---|---|---|
| E | R1 | +0.296363 | +0.296363 | True |
| E | R2 | +0.103975 | +0.103975 | True |
| E | R3 | +0.209410 | +0.209410 | True |
| E | R4 | +0.179515 | +0.179515 | True |
| E | RMSE H, D5 (eV) | 200.5462 | 200.5462 | True |
| E | RMSE G, D5 (eV) | 172.9443 | 172.9443 | True |
| E | RMSE T0, D5 (eV) | 156.6540 | 156.6540 | True |
| E | RMSE TX, D5 (eV) | 148.2865 | 148.2865 | True |
| E | share of TX within 10 % of the level, D5 | 0.540876 | 0.540876 | True |
| B2 | R1 | +0.518424 | +0.518424 | True |
| B2 | R2 | +0.149599 | +0.149599 | True |
| B2 | R3 | +0.507052 | +0.507052 | True |
| B2 | R4 | +0.224619 | +0.224619 | True |
| B2 | RMSE H, D5 (eV) | 171.6740 | 171.6740 | True |
| B2 | RMSE G, D5 (eV) | 156.9400 | 156.9400 | True |
| B2 | RMSE T0, D5 (eV) | 138.1946 | 138.1946 | True |
| B2 | RMSE TX, D5 (eV) | 127.4392 | 127.4392 | True |
| B2 | share of TX within 10 % of the level, D5 | 0.589222 | 0.589222 | True |
| B3 | R1 | +0.703606 | +0.703606 | True |
| B3 | R2 | +0.195603 | +0.195603 | True |
| B3 | R3 | +0.657990 | +0.657990 | True |
| B3 | R4 | +0.296359 | +0.296359 | True |
| B3 | RMSE H, D5 (eV) | 177.4831 | 177.4831 | True |
| B3 | RMSE G, D5 (eV) | 171.6298 | 171.6298 | True |
| B3 | RMSE T0, D5 (eV) | 143.9687 | 143.9687 | True |
| B3 | RMSE TX, D5 (eV) | 129.1229 | 129.1229 | True |
| B3 | share of TX within 10 % of the level, D5 | 0.612443 | 0.612443 | True |
| B4 | R1 | +0.644059 | +0.644059 | True |
| B4 | R2 | +0.197857 | +0.197857 | True |
| B4 | R3 | +0.607864 | +0.607864 | True |
| B4 | R4 | +0.311822 | +0.311822 | True |
| B4 | RMSE H, D5 (eV) | 217.6895 | 217.6895 | True |
| B4 | RMSE G, D5 (eV) | 218.8847 | 218.8847 | True |
| B4 | RMSE T0, D5 (eV) | 181.5790 | 181.5790 | True |
| B4 | RMSE TX, D5 (eV) | 162.6266 | 162.6266 | True |
| B4 | share of TX within 10 % of the level, D5 | 0.679910 | 0.679910 | True |
| B5 | R1 | +0.600474 | +0.600474 | True |
| B5 | R2 | +0.187343 | +0.187343 | True |
| B5 | R3 | +0.552047 | +0.552047 | True |
| B5 | R4 | +0.263662 | +0.263662 | True |
| B5 | RMSE H, D5 (eV) | 207.3564 | 207.3564 | True |
| B5 | RMSE G, D5 (eV) | 210.3238 | 210.3238 | True |
| B5 | RMSE T0, D5 (eV) | 180.4791 | 180.4791 | True |
| B5 | RMSE TX, D5 (eV) | 162.6974 | 162.6974 | True |
| B5 | share of TX within 10 % of the level, D5 | 0.673429 | 0.673429 | True |
| E, B5 | R5 | -2.272065 | -2.272065 | True |
| E, B5 | R6 | -0.304112 | -0.304112 | True |
| E, B5 | R7 | +0.132553 | +0.132553 | True |

Largest absolute difference: 4.83e-12.

## C. Session intervals with an own bootstrap (4,000 draws, seed 8602)

| statement | own interval | interval of the batch (20,000 draws) | largest difference of an end |
|---|---|---|---|
| R1.E | [+0.246, +0.353] | [+0.246, +0.354] | 0.002 |
| R3.E | [+0.089, +0.305] | [+0.095, +0.305] | 0.005 |
| R5 | [-2.620, -1.934] | [-2.622, -1.927] | 0.006 |
| R7 | [+0.117, +0.149] | [+0.116, +0.149] | 0.000 |

