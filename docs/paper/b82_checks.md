# B.82 checks

Registration: `ces_prediction/experiments/PREREGISTRATION_B82.md` section 8; implementation: `ces_prediction/experiments/PREREGISTRATION_B82_ADDENDUM1.md` section 9. Each check was run once, after the fits and before any readout. No tolerance was changed after a run. Checks reuse, tau, swap and Mirnov compare idx arrays, inputs and estimates only; check classes forms, as registered, the pooled increments of the stored estimates of B.75, B.79 and B.81 (known numbers) and the class counts; no value of a class was formed.

## 1. Reuse

Unit V1_r2_k0, rule D650, target L5, random_state 20261116: refitted with this batch's code and compared with `data/.b75/fits/`.

| learner | test rows | same rows | estimates bit-identical | fit rows (this batch / B.75) | iterations (this batch / B.75) |
|---|---|---|---|---|---|
| T0 | 26,669 | True | True | 133,674 / 133,674 | 100 / 100 |
| TX | 26,669 | True | True | 133,674 / 133,674 | 100 / 100 |

Check passes: **True**

## 2. tau

500 rows drawn from the S set (L5); tau recomputed from the row's shot file alone (time column, stamps as float32 as the grid stores them): identical for 500 of 500. S-set rows in the class 'no notch before': 0 (common set: 16,128). Rows within 1 ms of a class edge (103, 303 ms): {'103.0': 0, '303.0': 0}.

| set | up to 100 ms | above 100 and up to 300 ms | above 300 ms | no notch before |
|---|---|---|---|---|
| common_L5 | 12,462 | 23,038 | 108,715 | 16,128 |
| S_L5 | 13,718 | 24,352 | 109,767 | 0 |

Check passes: **True**

## 3. Classes

For every design, rule (D650, S), block (V1: all rows and the block from 32312) and comparison that a stored readout holds: the pooled increment of the stored estimates on this batch's rows, printed to three decimals, against the printed value of B.75 (TX/T0; TE/T0 and TB/T0 of V1 and V4), B.81 (TE/T0 and TB/T0 of V2 and V3) or B.79 (TXN/TN). For each stratification (tau, c): the class counts sum to the rows of the set (under V1 in each re-partition), and the class sums of squared errors add to the pooled ones (relative tolerance 1e-9).

| cell | source | stored | pooled here | tau: counts / sums | c: counts / sums |
|---|---|---|---|---|---|
| V1|D650|all|TX/T0 | B.75 | +0.336 | +0.336 | True / True | True / True |
| V1|D650|all|TE/T0 | B.75 | +0.229 | +0.229 | True / True | True / True |
| V1|D650|all|TB/T0 | B.75 | +0.246 | +0.246 | True / True | True / True |
| V1|D650|all|TXN/TN | B.79 | +0.258 | +0.258 | True / True | True / True |
| V1|D650|block from 32312|TX/T0 | B.75 | +0.157 | +0.157 | True / True | True / True |
| V1|D650|block from 32312|TE/T0 | B.75 | +0.172 | +0.172 | True / True | True / True |
| V1|D650|block from 32312|TB/T0 | B.75 | +0.052 | +0.052 | True / True | True / True |
| V1|D650|block from 32312|TXN/TN | B.79 | +0.252 | +0.252 | True / True | True / True |
| V1|S|all|TX/T0 | B.75 | +0.327 | +0.327 | True / True | True / True |
| V1|S|all|TE/T0 | B.75 | +0.221 | +0.221 | True / True | True / True |
| V1|S|all|TB/T0 | B.75 | +0.242 | +0.242 | True / True | True / True |
| V1|S|all|TXN/TN | B.79 | +0.253 | +0.253 | True / True | True / True |
| V1|S|block from 32312|TX/T0 | B.75 | +0.066 | +0.066 | True / True | True / True |
| V1|S|block from 32312|TE/T0 | B.75 | +0.073 | +0.073 | True / True | True / True |
| V1|S|block from 32312|TB/T0 | B.75 | -0.016 | -0.016 | True / True | True / True |
| V1|S|block from 32312|TXN/TN | B.79 | +0.086 | +0.086 | True / True | True / True |
| V2|D650|all|TX/T0 | B.75 | +0.282 | +0.282 | True / True | True / True |
| V2|D650|all|TE/T0 | B.81 | +0.163 | +0.163 | True / True | True / True |
| V2|D650|all|TB/T0 | B.81 | +0.225 | +0.225 | True / True | True / True |
| V2|D650|all|TXN/TN | B.79 | +0.298 | +0.298 | True / True | True / True |
| V2|S|all|TX/T0 | B.75 | +0.293 | +0.293 | True / True | True / True |
| V2|S|all|TE/T0 | B.81 | +0.181 | +0.181 | True / True | True / True |
| V2|S|all|TB/T0 | B.81 | +0.221 | +0.221 | True / True | True / True |
| V2|S|all|TXN/TN | B.79 | +0.238 | +0.238 | True / True | True / True |
| V3|D650|all|TX/T0 | B.75 | +0.190 | +0.190 | True / True | True / True |
| V3|D650|all|TE/T0 | B.81 | +0.058 | +0.058 | True / True | True / True |
| V3|D650|all|TB/T0 | B.81 | +0.122 | +0.122 | True / True | True / True |
| V3|D650|all|TXN/TN | B.79 | +0.263 | +0.263 | True / True | True / True |
| V3|S|all|TX/T0 | B.75 | +0.231 | +0.231 | True / True | True / True |
| V3|S|all|TE/T0 | B.81 | +0.160 | +0.160 | True / True | True / True |
| V3|S|all|TB/T0 | B.81 | +0.171 | +0.171 | True / True | True / True |
| V3|S|all|TXN/TN | B.79 | +0.175 | +0.175 | True / True | True / True |
| V4a|D650|all|TX/T0 | B.75 | +0.237 | +0.237 | True / True | True / True |
| V4a|D650|all|TE/T0 | B.75 | +0.177 | +0.177 | True / True | True / True |
| V4a|D650|all|TB/T0 | B.75 | -0.014 | -0.014 | True / True | True / True |
| V4a|D650|all|TXN/TN | B.79 | +0.245 | +0.245 | True / True | True / True |
| V4a|S|all|TX/T0 | B.75 | +0.028 | +0.028 | True / True | True / True |
| V4a|S|all|TE/T0 | B.75 | -0.045 | -0.045 | True / True | True / True |
| V4a|S|all|TB/T0 | B.75 | -0.047 | -0.047 | True / True | True / True |
| V4a|S|all|TXN/TN | B.79 | +0.102 | +0.102 | True / True | True / True |
| V4b|D650|all|TX/T0 | B.75 | +0.216 | +0.216 | True / True | True / True |
| V4b|D650|all|TE/T0 | B.75 | +0.145 | +0.145 | True / True | True / True |
| V4b|D650|all|TB/T0 | B.75 | +0.094 | +0.094 | True / True | True / True |
| V4b|D650|all|TXN/TN | B.79 | +0.275 | +0.275 | True / True | True / True |
| V4b|S|all|TX/T0 | B.75 | -0.254 | -0.254 | True / True | True / True |
| V4b|S|all|TE/T0 | B.75 | -0.296 | -0.296 | True / True | True / True |
| V4b|S|all|TB/T0 | B.75 | -0.096 | -0.096 | True / True | True / True |
| V4b|S|all|TXN/TN | B.79 | -0.062 | -0.062 | True / True | True / True |

Check passes: **True**

## 4. Swap

Anchor rows of every grid row (needed for the donor rows) against B.75's anchor rows on the D1 rows: {'D650': True, 'S': True}. For every unit of part C and both rules, fit and test rows: the ten history columns of TXs bit-identical to TX's; no row carries the inputs of its own discharge (donor file and donor row's file differ from the row's); every donor lies on the row's side (fit or test) of the unit; the donor row has the row's segment ordinal and frame index; the donors stored with the fit equal the rebuilt ones.

| unit, rule | rows (fit / test) | history identical | own discharge never | same side | same frame | rows without a donor row (fit / test) | donor files (fit / test) |
|---|---|---|---|---|---|---|---|
| V1_r2_k0|D650 | 133,674 / 26,669 | True / True | True / True | True / True | True / True | 5,391 / 1,286 | 242 / 54 |
| V1_r2_k0|S | 123,415 / 24,422 | True / True | True / True | True / True | True / True | 5,730 / 1,259 | 211 / 49 |
| V1_r2_k1|D650 | 127,536 / 32,807 | True / True | True / True | True / True | True / True | 6,524 / 1,097 | 222 / 66 |
| V1_r2_k1|S | 118,198 / 29,639 | True / True | True / True | True / True | True / True | 5,022 / 1,088 | 199 / 51 |
| V1_r2_k2|D650 | 126,937 / 33,406 | True / True | True / True | True / True | True / True | 5,160 / 2,710 | 234 / 57 |
| V1_r2_k2|S | 116,688 / 31,149 | True / True | True / True | True / True | True / True | 4,637 / 2,101 | 207 / 52 |
| V1_r2_k3|D650 | 130,415 / 29,928 | True / True | True / True | True / True | True / True | 6,646 / 1,884 | 229 / 54 |
| V1_r2_k3|S | 120,138 / 27,699 | True / True | True / True | True / True | True / True | 5,434 / 1,453 | 209 / 49 |
| V1_r2_k4|D650 | 122,810 / 37,533 | True / True | True / True | True / True | True / True | 6,424 / 1,921 | 227 / 63 |
| V1_r2_k4|S | 112,909 / 34,928 | True / True | True / True | True / True | True / True | 4,779 / 1,647 | 196 / 49 |
| V1_r3_k0|D650 | 131,715 / 28,628 | True / True | True / True | True / True | True / True | 6,409 / 2,326 | 233 / 55 |
| V1_r3_k0|S | 121,032 / 26,805 | True / True | True / True | True / True | True / True | 5,676 / 1,734 | 200 / 49 |
| V1_r3_k1|D650 | 127,931 / 32,412 | True / True | True / True | True / True | True / True | 6,304 / 1,965 | 227 / 58 |
| V1_r3_k1|S | 117,974 / 29,863 | True / True | True / True | True / True | True / True | 5,568 / 1,440 | 215 / 56 |
| V1_r3_k2|D650 | 122,684 / 37,659 | True / True | True / True | True / True | True / True | 4,896 / 2,600 | 224 / 64 |
| V1_r3_k2|S | 113,073 / 34,764 | True / True | True / True | True / True | True / True | 3,926 / 2,129 | 208 / 54 |
| V1_r3_k3|D650 | 129,711 / 30,632 | True / True | True / True | True / True | True / True | 5,615 / 1,939 | 224 / 58 |
| V1_r3_k3|S | 120,280 / 27,557 | True / True | True / True | True / True | True / True | 4,287 / 1,793 | 212 / 51 |
| V1_r3_k4|D650 | 129,331 / 31,012 | True / True | True / True | True / True | True / True | 6,272 / 1,441 | 235 / 62 |
| V1_r3_k4|S | 118,989 / 28,848 | True / True | True / True | True / True | True / True | 5,414 / 1,038 | 202 / 55 |
| V1_r4_k0|D650 | 129,342 / 31,001 | True / True | True / True | True / True | True / True | 6,202 / 1,197 | 217 / 53 |
| V1_r4_k0|S | 119,540 / 28,297 | True / True | True / True | True / True | True / True | 5,625 / 1,478 | 206 / 56 |
| V1_r4_k1|D650 | 129,265 / 31,078 | True / True | True / True | True / True | True / True | 6,501 / 883 | 230 / 57 |
| V1_r4_k1|S | 119,347 / 28,490 | True / True | True / True | True / True | True / True | 5,596 / 788 | 210 / 49 |
| V1_r4_k2|D650 | 125,692 / 34,651 | True / True | True / True | True / True | True / True | 4,792 / 2,561 | 228 / 55 |
| V1_r4_k2|S | 115,936 / 31,901 | True / True | True / True | True / True | True / True | 4,071 / 1,913 | 211 / 52 |
| V1_r4_k3|D650 | 124,325 / 36,018 | True / True | True / True | True / True | True / True | 5,470 / 2,180 | 231 / 53 |
| V1_r4_k3|S | 114,249 / 33,588 | True / True | True / True | True / True | True / True | 4,077 / 2,062 | 205 / 49 |
| V1_r4_k4|D650 | 132,748 / 27,595 | True / True | True / True | True / True | True / True | 6,657 / 1,001 | 227 / 56 |
| V1_r4_k4|S | 122,276 / 25,561 | True / True | True / True | True / True | True / True | 5,738 / 794 | 212 / 51 |
| V3_q1|D650 | 48,144 / 27,883 | True / True | True / True | True / True | True / True | 573 / 1,000 | 61 / 54 |
| V3_q1|S | 45,153 / 26,484 | True / True | True / True | True / True | True / True | 546 / 947 | 58 / 56 |
| V3_q2|D650 | 76,027 / 43,406 | True / True | True / True | True / True | True / True | 1,618 / 1,323 | 117 / 63 |
| V3_q2|S | 71,637 / 40,944 | True / True | True / True | True / True | True / True | 1,506 / 1,671 | 110 / 60 |
| V3_q3|D650 | 119,433 / 23,933 | True / True | True / True | True / True | True / True | 2,778 / 2,514 | 179 / 60 |
| V3_q3|S | 112,581 / 22,751 | True / True | True / True | True / True | True / True | 2,641 / 2,041 | 167 / 51 |
| V3_q4|D650 | 143,366 / 16,977 | True / True | True / True | True / True | True / True | 5,244 / 2,900 | 236 / 57 |
| V3_q4|S | 135,332 / 12,505 | True / True | True / True | True / True | True / True | 4,840 / 1,649 | 229 / 44 |
| V4a|D650 | 127,270 / 10,146 | True / True | True / True | True / True | True / True | 4,405 / 2,637 | 179 / 42 |
| V4a|S | 119,655 / 7,180 | True / True | True / True | True / True | True / True | 4,126 / 1,712 | 179 / 32 |

Check passes: **True**

## 5. Mirnov

Unit V1_r2_k0: 200 test rows per rule; the seven Mirnov scalars recomputed from the row's shot file alone (time and the two MC columns as float32, blocks, standardisation with the unit's statistics, window scalars, running means, anchor scalars with B.75's anchor row) against the batch's.

| rule | rows | identical | rows with an undefined scalar |
|---|---|---|---|
| D650 | 200 | 200 | 146 |
| S | 200 | 200 | 150 |

Check passes: **True**

## Beside the checks: the Mirnov columns joined to the grid

641 files, 247,207 rows; files with another row count [], with other stamps [], with other BES or MC headers []; non-finite MC values 0. Passes: True.

## Beside the checks: the weighted quantile on an input

Weighted quantiles (0.5, 0.75, 0.9) of c over the test rows of V1_r2_k0 under D650 against the quantiles of the explicitly resampled rows, 20 draws per form: largest absolute difference 0 (identical: True).
