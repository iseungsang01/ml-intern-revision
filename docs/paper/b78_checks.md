# B.78 checks

Registration: `ces_prediction/experiments/PREREGISTRATION_B78.md` section 10; implementation: `ces_prediction/experiments/PREREGISTRATION_B78_ADDENDUM1.md` section 8. Run before any readout. No tolerance was changed after a run.

## 1. Frame index and scored rows

Grid rows 247,207; rows without a frame index 1,282 (each file's first and last row, which are not grid rows); rows off the 10 ms lattice by more than 3 ms: 0. Largest distance of a stamp from the first grid row of its block plus 10 ms x its index: 0.0010 ms. Measured rows 225,793, of them without an index: 0. D1 rows 222,672; scored rows (index odd and not a multiple of 5) 88,749; by index modulo 10: 1: 22,488, 3: 22,455, 7: 22,500, 9: 21,306. Scored rows whose index is a multiple of k: K2 0, K5 0, K10 0, K20 0, K50 0.

Check passes: **True**

## 2. Inputs under K1 against B.75's D10 inputs (`data/.b75/inputs/D10.npz`) on the scored rows

Scored rows 88,749.

| input | bit-identical | largest deviation | same undefined rows |
|---|---|---|---|
| H | True | 0.0e+00 | True (0 rows undefined) |
| age | True | 0.0e+00 | True (0 rows undefined) |
| M5 | True | 0.0e+00 | True (0 rows undefined) |
| G | True | 0.0e+00 | True (0 rows undefined) |
| th | True | 0.0e+00 | True (640 rows undefined) |
| anc_g | True | - | - |
| nvis | True | - | - |

The six anchor scalars are functions of the anchor row (`anc_g`) and of the unit only, so they are identical too. (`th` = the five history scalars; `anc_g` = the anchor row; `nvis` = the number of visible values of the block.)

Check passes: **True**

## 3. Visibility under K10

Unit V1_r2_k0; 200 rows drawn from its 10,091 test rows in the common set (L5). For each row, a copy of its file in which every T_i value off the kept grid (index not a multiple of 10) or not before row t is removed, or raised by 1000 eV; the ten history scalars, G, M5, the number of visible values, the anchor row and the six anchor scalars recomputed and compared bit for bit with the main pass. Note: the eight fast scalars and the running means m(i) take no T_i argument and are not recomputed; the anchor scalars are recomputed from the recomputed anchor row; M5 and the number of visible values are compared too.

| copy | rows identical | values changed (total) |
|---|---|---|
| removed | 200 / 200 | 121,803 |
| raised by 1000 eV | 200 / 200 | 121,803 |

Check passes: **True**

## 4. Units

Every B.73 unit has the test rows stored by B.73: **True** (36 units).

## Row sets

D1 rows 222,672; scored rows 88,749; scored rows with the eight fast scalars 86,449.

| rule | scored rows with a visible value | of them fewer than three | history and G complete | anchor means defined | complete | dropped by a3 alone |
|---|---|---|---|---|---|---|
| K1 | 88,749 | 640 | 88,109 | 87,643 | 86,000 | 438 |
| K2 | 88,742 | 1,205 | 87,537 | 87,639 | 85,963 | 442 |
| K5 | 88,656 | 2,548 | 86,108 | 87,588 | 85,618 | 460 |
| K10 | 88,432 | 4,954 | 83,478 | 86,376 | 82,985 | 465 |
| K20 | 88,004 | 9,853 | 78,151 | 83,810 | 77,710 | 431 |
| K50 | 86,728 | 24,182 | 62,546 | 76,695 | 62,213 | 308 |

Complete under all six rules: 62,162. Common set: F 62,162, L5 60,523 (scored rows with the target: F 88,749, L5 85,738).
