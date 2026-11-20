# B.84 checks

Registration `ces_prediction/experiments/PREREGISTRATION_B84.md`, Addendum 1 `ces_prediction/experiments/PREREGISTRATION_B84_ADDENDUM1.md` section 4. Each check was run once, after the fits and before any readout; no error, MSE or skill of this batch had been computed when they ran.

| Check | Result | Pass |
|---|---|---|
| reuse | `V1_r2_k0`, D650, L5, refitted with this batch's code: T0 test rows identical True, largest difference 0.0 eV; TX test rows identical True, largest difference 0.0 eV; fit rows 133,674 (B.75 133,674) | yes |
| mixture | T0all: D650 block rows / inputs / targets equal True / True / True, test-discharge rows 0, 925,459 stacked rows; TXall: D650 block rows / inputs / targets equal True / True / True, test-discharge rows 0, 925,459 stacked rows; T0raw: D650 block rows / inputs / targets equal True / True / True, test-discharge rows 0, 925,459 stacked rows; TXraw: D650 block rows / inputs / targets equal True / True / True, test-discharge rows 0, 925,459 stacked rows | yes |
| raw windows | 200 test rows of `V1_r2_k0` under D650: twenty fast values identical on 200 (largest difference 0.0); 23 history columns identical on 200 after raising every value not visible under D650 by 1000 eV | yes |
| rows | every estimator of every comparison on the test rows of B.75's (B.78's) stored TX, 24 cells | yes |

## Stacked fit rows per rule (unit `V1_r2_k0`)

| Learner | D10 | D50 | D100 | D300 | D650 | D1000 | S | total |
|---|---|---|---|---|---|---|---|---|
| T0all | 133,674 | 133,674 | 133,674 | 133,674 | 133,674 | 133,674 | 123,415 | 925,459 |
| TXall | 133,674 | 133,674 | 133,674 | 133,674 | 133,674 | 133,674 | 123,415 | 925,459 |
| T0raw | 133,674 | 133,674 | 133,674 | 133,674 | 133,674 | 133,674 | 123,415 | 925,459 |
| TXraw | 133,674 | 133,674 | 133,674 | 133,674 | 133,674 | 133,674 | 123,415 | 925,459 |

## Rows per design and rule

| Design | Rule | Rows | Discharges | Clusters | Same rows |
|---|---|---|---|---|---|
| V1 | D10 | 160,343 | 562 | 79 | yes |
| V1 | D50 | 160,343 | 562 | 79 | yes |
| V1 | D100 | 160,343 | 562 | 79 | yes |
| V1 | D300 | 160,343 | 562 | 79 | yes |
| V1 | D650 | 160,343 | 562 | 79 | yes |
| V1 | D1000 | 160,343 | 562 | 79 | yes |
| V1 | S | 147,837 | 503 | 71 | yes |
| V1 | K10 | 60,523 | 507 | 72 | yes |
| V3 | D10 | 112,199 | 441 | 60 | yes |
| V3 | D50 | 112,199 | 441 | 60 | yes |
| V3 | D100 | 112,199 | 441 | 60 | yes |
| V3 | D300 | 112,199 | 441 | 60 | yes |
| V3 | D650 | 112,199 | 441 | 60 | yes |
| V3 | D1000 | 112,199 | 441 | 60 | yes |
| V3 | S | 102,684 | 395 | 56 | yes |
| V3 | K10 | 41,985 | 397 | 57 | yes |
| V4a | D10 | 10,146 | 75 | 18 | yes |
| V4a | D50 | 10,146 | 75 | 18 | yes |
| V4a | D100 | 10,146 | 75 | 18 | yes |
| V4a | D300 | 10,146 | 75 | 18 | yes |
| V4a | D650 | 10,146 | 75 | 18 | yes |
| V4a | D1000 | 10,146 | 75 | 18 | yes |
| V4a | S | 7,180 | 50 | 14 | yes |
| V4a | K10 | 3,587 | 59 | 16 | yes |
