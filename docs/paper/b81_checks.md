# B.81 checks

Registration: `ces_prediction/experiments/PREREGISTRATION_B81.md` section 6; implementation: `ces_prediction/experiments/PREREGISTRATION_B81_ADDENDUM1.md` section 3. Each check was run once, after the fits and before any readout. Only idx arrays, matrices and estimates are compared; no target, error or skill was formed. No tolerance was changed after a run.

## 1. Reuse

Unit V1_r2_k0, rule D650, target L5: refitted with this batch's code and compared with `data/.b75/fits/`.

| learner | test rows | same rows | estimates bit-identical | fit rows (this batch / B.75) | iterations (this batch / B.75) |
|---|---|---|---|---|---|
| T0 | 26,669 | True | True | 133,674 / 133,674 | 100 / 100 |
| TX | 26,669 | True | True | 133,674 / 133,674 | 100 / 100 |
| TE | 26,669 | True | True | 133,674 / 133,674 | 100 / 100 |
| TB | 26,669 | True | True | 133,674 / 133,674 | 100 / 100 |

Check passes: **True**

## 2. Permutation

Every unit (36) and both rules: the TXp matrices rebuilt by the fit code against TX's on the same rows. History columns 0-9 bit-identical; each of columns 10-23 the same multiset of values (sorted, byte comparison); the fourteen columns of a row moved together (rows sorted lexicographically identical); within the fit rows and within the test rows separately.

| unit, rule | fit rows | test rows | history identical (fit / test) | same multisets (fit / test) | joint reordering (fit / test) | rows moved (fit / test) |
|---|---|---|---|---|---|---|
| V1_r2_k0|D650 | 133,674 | 26,669 | True / True | True / True | True / True | 133,673 / 26,668 |
| V1_r2_k0|S | 123,415 | 24,422 | True / True | True / True | True / True | 123,415 / 24,422 |
| V1_r2_k1|D650 | 127,536 | 32,807 | True / True | True / True | True / True | 127,535 / 32,806 |
| V1_r2_k1|S | 118,198 | 29,639 | True / True | True / True | True / True | 118,198 / 29,639 |
| V1_r2_k2|D650 | 126,937 | 33,406 | True / True | True / True | True / True | 126,937 / 33,405 |
| V1_r2_k2|S | 116,688 | 31,149 | True / True | True / True | True / True | 116,688 / 31,148 |
| V1_r2_k3|D650 | 130,415 | 29,928 | True / True | True / True | True / True | 130,414 / 29,927 |
| V1_r2_k3|S | 120,138 | 27,699 | True / True | True / True | True / True | 120,137 / 27,696 |
| V1_r2_k4|D650 | 122,810 | 37,533 | True / True | True / True | True / True | 122,808 / 37,533 |
| V1_r2_k4|S | 112,909 | 34,928 | True / True | True / True | True / True | 112,909 / 34,925 |
| V1_r3_k0|D650 | 131,715 | 28,628 | True / True | True / True | True / True | 131,714 / 28,628 |
| V1_r3_k0|S | 121,032 | 26,805 | True / True | True / True | True / True | 121,032 / 26,805 |
| V1_r3_k1|D650 | 127,931 | 32,412 | True / True | True / True | True / True | 127,930 / 32,408 |
| V1_r3_k1|S | 117,974 | 29,863 | True / True | True / True | True / True | 117,974 / 29,862 |
| V1_r3_k2|D650 | 122,684 | 37,659 | True / True | True / True | True / True | 122,683 / 37,659 |
| V1_r3_k2|S | 113,073 | 34,764 | True / True | True / True | True / True | 113,072 / 34,762 |
| V1_r3_k3|D650 | 129,711 | 30,632 | True / True | True / True | True / True | 129,710 / 30,631 |
| V1_r3_k3|S | 120,280 | 27,557 | True / True | True / True | True / True | 120,277 / 27,557 |
| V1_r3_k4|D650 | 129,331 | 31,012 | True / True | True / True | True / True | 129,329 / 31,011 |
| V1_r3_k4|S | 118,989 | 28,848 | True / True | True / True | True / True | 118,987 / 28,847 |
| V1_r4_k0|D650 | 129,342 | 31,001 | True / True | True / True | True / True | 129,342 / 31,001 |
| V1_r4_k0|S | 119,540 | 28,297 | True / True | True / True | True / True | 119,539 / 28,297 |
| V1_r4_k1|D650 | 129,265 | 31,078 | True / True | True / True | True / True | 129,265 / 31,077 |
| V1_r4_k1|S | 119,347 | 28,490 | True / True | True / True | True / True | 119,347 / 28,488 |
| V1_r4_k2|D650 | 125,692 | 34,651 | True / True | True / True | True / True | 125,690 / 34,650 |
| V1_r4_k2|S | 115,936 | 31,901 | True / True | True / True | True / True | 115,934 / 31,901 |
| V1_r4_k3|D650 | 124,325 | 36,018 | True / True | True / True | True / True | 124,323 / 36,017 |
| V1_r4_k3|S | 114,249 | 33,588 | True / True | True / True | True / True | 114,247 / 33,587 |
| V1_r4_k4|D650 | 132,748 | 27,595 | True / True | True / True | True / True | 132,748 / 27,593 |
| V1_r4_k4|S | 122,276 | 25,561 | True / True | True / True | True / True | 122,276 / 25,559 |
| V2_s1_k0|D650 | 121,578 | 38,765 | True / True | True / True | True / True | 121,578 / 38,765 |
| V2_s1_k0|S | 113,102 | 34,735 | True / True | True / True | True / True | 113,100 / 34,734 |
| V2_s1_k1|D650 | 133,356 | 26,987 | True / True | True / True | True / True | 133,354 / 26,986 |
| V2_s1_k1|S | 121,816 | 26,021 | True / True | True / True | True / True | 121,815 / 26,020 |
| V2_s1_k2|D650 | 116,598 | 43,745 | True / True | True / True | True / True | 116,596 / 43,745 |
| V2_s1_k2|S | 106,479 | 41,358 | True / True | True / True | True / True | 106,478 / 41,357 |
| V2_s1_k3|D650 | 138,804 | 21,539 | True / True | True / True | True / True | 138,801 / 21,539 |
| V2_s1_k3|S | 128,163 | 19,674 | True / True | True / True | True / True | 128,161 / 19,672 |
| V2_s1_k4|D650 | 131,036 | 29,307 | True / True | True / True | True / True | 131,035 / 29,304 |
| V2_s1_k4|S | 121,788 | 26,049 | True / True | True / True | True / True | 121,788 / 26,047 |
| V2_s2_k0|D650 | 131,849 | 28,494 | True / True | True / True | True / True | 131,847 / 28,493 |
| V2_s2_k0|S | 121,473 | 26,364 | True / True | True / True | True / True | 121,470 / 26,364 |
| V2_s2_k1|D650 | 127,007 | 33,336 | True / True | True / True | True / True | 127,006 / 33,336 |
| V2_s2_k1|S | 117,217 | 30,620 | True / True | True / True | True / True | 117,216 / 30,619 |
| V2_s2_k2|D650 | 128,588 | 31,755 | True / True | True / True | True / True | 128,586 / 31,752 |
| V2_s2_k2|S | 117,592 | 30,245 | True / True | True / True | True / True | 117,590 / 30,244 |
| V2_s2_k3|D650 | 128,355 | 31,988 | True / True | True / True | True / True | 128,354 / 31,988 |
| V2_s2_k3|S | 119,602 | 28,235 | True / True | True / True | True / True | 119,601 / 28,235 |
| V2_s2_k4|D650 | 125,573 | 34,770 | True / True | True / True | True / True | 125,573 / 34,768 |
| V2_s2_k4|S | 115,464 | 32,373 | True / True | True / True | True / True | 115,463 / 32,372 |
| V2_s3_k0|D650 | 135,902 | 24,441 | True / True | True / True | True / True | 135,901 / 24,440 |
| V2_s3_k0|S | 125,516 | 22,321 | True / True | True / True | True / True | 125,515 / 22,318 |
| V2_s3_k1|D650 | 137,708 | 22,635 | True / True | True / True | True / True | 137,707 / 22,634 |
| V2_s3_k1|S | 127,137 | 20,700 | True / True | True / True | True / True | 127,136 / 20,698 |
| V2_s3_k2|D650 | 133,716 | 26,627 | True / True | True / True | True / True | 133,716 / 26,627 |
| V2_s3_k2|S | 123,448 | 24,389 | True / True | True / True | True / True | 123,447 / 24,389 |
| V2_s3_k3|D650 | 120,606 | 39,737 | True / True | True / True | True / True | 120,606 / 39,735 |
| V2_s3_k3|S | 110,053 | 37,784 | True / True | True / True | True / True | 110,052 / 37,783 |
| V2_s3_k4|D650 | 113,440 | 46,903 | True / True | True / True | True / True | 113,439 / 46,903 |
| V2_s3_k4|S | 105,194 | 42,643 | True / True | True / True | True / True | 105,191 / 42,641 |
| V3_q1|D650 | 48,144 | 27,883 | True / True | True / True | True / True | 48,143 / 27,880 |
| V3_q1|S | 45,153 | 26,484 | True / True | True / True | True / True | 45,153 / 26,483 |
| V3_q2|D650 | 76,027 | 43,406 | True / True | True / True | True / True | 76,027 / 43,406 |
| V3_q2|S | 71,637 | 40,944 | True / True | True / True | True / True | 71,637 / 40,942 |
| V3_q3|D650 | 119,433 | 23,933 | True / True | True / True | True / True | 119,433 / 23,932 |
| V3_q3|S | 112,581 | 22,751 | True / True | True / True | True / True | 112,581 / 22,751 |
| V3_q4|D650 | 143,366 | 16,977 | True / True | True / True | True / True | 143,366 / 16,976 |
| V3_q4|S | 135,332 | 12,505 | True / True | True / True | True / True | 135,330 / 12,503 |
| V4a|D650 | 127,270 | 10,146 | True / True | True / True | True / True | 127,269 / 10,146 |
| V4a|S | 119,655 | 7,180 | True / True | True / True | True / True | 119,655 / 7,180 |
| V4b|D650 | 150,197 | 10,146 | True / True | True / True | True / True | 150,195 / 10,146 |
| V4b|S | 140,657 | 7,180 | True / True | True / True | True / True | 140,657 / 7,179 |

Check passes: **True**

## 3. Anchor row

Unit V1_r2_k0: 200 test rows per rule. Columns 10-17 of TXo against the eight fast scalars recomputed at the anchor's grid index from the anchor's file alone (channels standardised with the unit's statistics, sequential array means, B.73 `window_feats`).

| rule | test rows | rows drawn | identical | rows with undefined anchor-row fast scalars | anchor a D1 row and equal to TX's fast scalars of that row |
|---|---|---|---|---|---|
| D650 | 26,669 | 200 | 200 | 0 | 199 |
| S | 24,422 | 200 | 200 | 0 | 197 |

Check passes: **True**

## 4. Rows

For every unit and both rules, the test rows of T0, TX, TXp, TXo, TE and TB (stored files of B.75 or B.81) against the rows of the unit's test discharges in the rule's set (D650: common set, L5; S: S set, L5).

| unit, rule | rows | T0 | TX | TXp | TXo | TE | TB |
|---|---|---|---|---|---|---|---|
| V1_r2_k0|D650 | 26,669 | True | True | True | True | True | True |
| V1_r2_k0|S | 24,422 | True | True | True | True | True | True |
| V1_r2_k1|D650 | 32,807 | True | True | True | True | True | True |
| V1_r2_k1|S | 29,639 | True | True | True | True | True | True |
| V1_r2_k2|D650 | 33,406 | True | True | True | True | True | True |
| V1_r2_k2|S | 31,149 | True | True | True | True | True | True |
| V1_r2_k3|D650 | 29,928 | True | True | True | True | True | True |
| V1_r2_k3|S | 27,699 | True | True | True | True | True | True |
| V1_r2_k4|D650 | 37,533 | True | True | True | True | True | True |
| V1_r2_k4|S | 34,928 | True | True | True | True | True | True |
| V1_r3_k0|D650 | 28,628 | True | True | True | True | True | True |
| V1_r3_k0|S | 26,805 | True | True | True | True | True | True |
| V1_r3_k1|D650 | 32,412 | True | True | True | True | True | True |
| V1_r3_k1|S | 29,863 | True | True | True | True | True | True |
| V1_r3_k2|D650 | 37,659 | True | True | True | True | True | True |
| V1_r3_k2|S | 34,764 | True | True | True | True | True | True |
| V1_r3_k3|D650 | 30,632 | True | True | True | True | True | True |
| V1_r3_k3|S | 27,557 | True | True | True | True | True | True |
| V1_r3_k4|D650 | 31,012 | True | True | True | True | True | True |
| V1_r3_k4|S | 28,848 | True | True | True | True | True | True |
| V1_r4_k0|D650 | 31,001 | True | True | True | True | True | True |
| V1_r4_k0|S | 28,297 | True | True | True | True | True | True |
| V1_r4_k1|D650 | 31,078 | True | True | True | True | True | True |
| V1_r4_k1|S | 28,490 | True | True | True | True | True | True |
| V1_r4_k2|D650 | 34,651 | True | True | True | True | True | True |
| V1_r4_k2|S | 31,901 | True | True | True | True | True | True |
| V1_r4_k3|D650 | 36,018 | True | True | True | True | True | True |
| V1_r4_k3|S | 33,588 | True | True | True | True | True | True |
| V1_r4_k4|D650 | 27,595 | True | True | True | True | True | True |
| V1_r4_k4|S | 25,561 | True | True | True | True | True | True |
| V2_s1_k0|D650 | 38,765 | True | True | True | True | True | True |
| V2_s1_k0|S | 34,735 | True | True | True | True | True | True |
| V2_s1_k1|D650 | 26,987 | True | True | True | True | True | True |
| V2_s1_k1|S | 26,021 | True | True | True | True | True | True |
| V2_s1_k2|D650 | 43,745 | True | True | True | True | True | True |
| V2_s1_k2|S | 41,358 | True | True | True | True | True | True |
| V2_s1_k3|D650 | 21,539 | True | True | True | True | True | True |
| V2_s1_k3|S | 19,674 | True | True | True | True | True | True |
| V2_s1_k4|D650 | 29,307 | True | True | True | True | True | True |
| V2_s1_k4|S | 26,049 | True | True | True | True | True | True |
| V2_s2_k0|D650 | 28,494 | True | True | True | True | True | True |
| V2_s2_k0|S | 26,364 | True | True | True | True | True | True |
| V2_s2_k1|D650 | 33,336 | True | True | True | True | True | True |
| V2_s2_k1|S | 30,620 | True | True | True | True | True | True |
| V2_s2_k2|D650 | 31,755 | True | True | True | True | True | True |
| V2_s2_k2|S | 30,245 | True | True | True | True | True | True |
| V2_s2_k3|D650 | 31,988 | True | True | True | True | True | True |
| V2_s2_k3|S | 28,235 | True | True | True | True | True | True |
| V2_s2_k4|D650 | 34,770 | True | True | True | True | True | True |
| V2_s2_k4|S | 32,373 | True | True | True | True | True | True |
| V2_s3_k0|D650 | 24,441 | True | True | True | True | True | True |
| V2_s3_k0|S | 22,321 | True | True | True | True | True | True |
| V2_s3_k1|D650 | 22,635 | True | True | True | True | True | True |
| V2_s3_k1|S | 20,700 | True | True | True | True | True | True |
| V2_s3_k2|D650 | 26,627 | True | True | True | True | True | True |
| V2_s3_k2|S | 24,389 | True | True | True | True | True | True |
| V2_s3_k3|D650 | 39,737 | True | True | True | True | True | True |
| V2_s3_k3|S | 37,784 | True | True | True | True | True | True |
| V2_s3_k4|D650 | 46,903 | True | True | True | True | True | True |
| V2_s3_k4|S | 42,643 | True | True | True | True | True | True |
| V3_q1|D650 | 27,883 | True | True | True | True | True | True |
| V3_q1|S | 26,484 | True | True | True | True | True | True |
| V3_q2|D650 | 43,406 | True | True | True | True | True | True |
| V3_q2|S | 40,944 | True | True | True | True | True | True |
| V3_q3|D650 | 23,933 | True | True | True | True | True | True |
| V3_q3|S | 22,751 | True | True | True | True | True | True |
| V3_q4|D650 | 16,977 | True | True | True | True | True | True |
| V3_q4|S | 12,505 | True | True | True | True | True | True |
| V4a|D650 | 10,146 | True | True | True | True | True | True |
| V4a|S | 7,180 | True | True | True | True | True | True |
| V4b|D650 | 10,146 | True | True | True | True | True | True |
| V4b|S | 7,180 | True | True | True | True | True | True |

Check passes: **True**
