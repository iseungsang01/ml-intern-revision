# B.73 checks

Registration: `ces_prediction/experiments/PREREGISTRATION_B73.md` section 10, with Addendum 2 section 10 (implementation choices; Addendum 1 is the session lead's). Run before any readout; the batch continues only if (a) and (b) pass.

## (a) Stored estimates against B.70 (V1, target F; tolerance 0.001)

| statement | r2 | r3 | r4 | mean | B.70 mean | largest deviation | passes |
|---|---|---|---|---|---|---|---|
| D1-1 | +0.1761 | +0.1796 | +0.1747 | +0.1768 | +0.1768 | 1.8e-09 | True |
| D1-2 | +0.0955 | +0.0972 | +0.0981 | +0.0969 | +0.0969 | 4.3e-10 | True |
| D1-3 | +0.0463 | +0.0501 | +0.0608 | +0.0524 | +0.0524 | 2.2e-16 | True |
| D1-4 | +0.0927 | +0.0937 | +0.0950 | +0.0938 | +0.0938 | 2.2e-16 | True |

(a) passes: **True**

## (b) T0 and TX against the reviewer's values (V1, target F; tolerance 0.005)

Reviewer's values: `data/.critique_20261120_stats/logs/` (gbt_lean r2, gbt_variants2 r3, gbt_lean r4), the C7 rows of `docs/paper/review_loop/critique_20261120_stats.md`. G(stored) = the causal GP of the comparison files, as the reviewer used.

| comparison | r2 ours / reviewer | r3 ours / reviewer | r4 ours / reviewer | largest deviation |
|---|---|---|---|---|
| TX/T0 | +0.0888 / +0.0877 | +0.0873 / +0.0852 | +0.0848 / +0.0859 | 0.0021 |
| T0/G(stored) | +0.0991 / +0.0971 | +0.0992 / +0.0992 | +0.1003 / +0.0999 | 0.0020 |
| TX/c2 | +0.0036 / +0.0002 | -0.0021 / -0.0044 | +0.0024 / +0.0030 | 0.0034 |
| TX/c | +0.0952 / +0.0921 | +0.0928 / +0.0908 | +0.0903 / +0.0909 | 0.0031 |
| T0/t | +0.0530 / +0.0509 | +0.0558 / +0.0558 | +0.0664 / +0.0659 | 0.0021 |
| T0/t2 | +0.0109 / +0.0087 | +0.0087 / +0.0087 | +0.0168 / +0.0163 | 0.0022 |

Median absolute error (eV), TX / T0: r2 39.8 / 41.7; r3 39.7 / 41.7; r4 39.7 / 41.7 (reviewer: 39.8 / 41.8, 39.8 / 41.8, 39.7 / 41.7).

(b) passes: **True**

## (c) No input of row t contains a value of row t or later

Unit V1_r2_k0: 126 test discharges, one cut row each, 18,401 D1 rows up to the cuts compared. T_i values from the cut on multiplied by factors in [0.5, 1.5], half of the missing ones from the cut on given a value; fast channels after the cut multiplied by factors in [0.5, 1.5] (filter: its BES drive perturbed after the cut).

| quantity | result |
|---|---|
| history inputs (hold, age, five scalars, M5) at rows up to the cut | unchanged: True (rows after the cut changed: 19,706) |
| recomputed hold and age equal the row table | True |
| causal GP at the cut rows | largest change 0.00e+00 eV; recomputed vs table 0.00e+00 eV |
| eight fast scalars of row t and of row t-1 | unchanged: True |
| inputs of the fit rows | identical: True |
| tree estimates up to the cut (fits redone), largest change (eV) | T0[F] 0.0e+00, T0[L5] 0.0e+00, TX[F] 0.0e+00, TX[L5] 0.0e+00, TB[F] 0.0e+00, TB[L5] 0.0e+00, TE[F] 0.0e+00, TE[L5] 0.0e+00, TXlag[F] 0.0e+00, TXlag[L5] 0.0e+00 |
| the same against the stage's cached estimates (eV) | T0[F] 0.0e+00, T0[L5] 0.0e+00, TX[F] 0.0e+00, TX[L5] 0.0e+00, TB[F] 0.0e+00, TB[L5] 0.0e+00, TE[F] 0.0e+00, TE[L5] 0.0e+00, TXlag[F] 0.0e+00, TXlag[L5] 0.0e+00 |
| filters K0, KX up to the cut (V1_r2_k0__F.json) | largest change 0.0e+00, 0.0e+00 eV |

(c) passes: **True**

**Two earlier runs of (c), superseded before any readout.** Both failed on bit-identity of the fast scalars, not on causality. In both, the history inputs, the causal GP and the filters were unchanged up to the cut.

| run | implementation of the fast scalars | what (c) found | (b) TX over T0, r2 / r3 / r4 |
|---|---|---|---|
| 1 | `window_feats` once over all files laid end to end (running sums crossing file boundaries); channel statistics from per-file sums and sums of squares | fast scalars of rows up to the cut and of the fit rows not bit-identical after perturbing later rows of other files (rounding, at most about 1e-8); tree estimates up to the cut unchanged (largest change 0.0e+00 eV) | +0.0877, +0.0874, +0.0854 |
| 2 | `window_feats` per file; statistics as in run 1 | the standard deviations differed in the last digits between the two processes (sum of squares minus squared mean, numpy reductions whose rounding depends on memory alignment), so the fast scalars of fit rows differed by up to 1.7e-7; one tree estimate up to the cut moved by 16.7 eV when a value crossed a bin edge | +0.0897, +0.0873, +0.0858 |
| 3 (above) | `window_feats` per file; mean and sd by two passes of exactly rounded sums (`math.fsum`); array means by explicit sequential sums | see the table above | see (b) |

Every tree estimator and classifier was refitted after each change. The readout uses run 3 only. Run 1 to run 2 also shows how sensitive the boosted trees are to rounding-level changes of their inputs: TX over T0 moved by up to 0.002 on a re-partition.

## (d) Folds

- V2: every cluster whole in each fold: True
- V3, V4: every fit discharge earlier than the first test discharge: True (V3_q1: last fit shot 31334, first test shot 31335; V3_q2: last fit shot 31589, first test shot 31590; V3_q3: last fit shot 31924, first test shot 31925; V3_q4: last fit shot 32185, first test shot 32188; V4a: last fit shot 31991, first test shot 32312; V4b: last fit shot 32310, first test shot 32312)
- fit and test discharges disjoint in every unit: True; V1 and V2: every D1 discharge tested exactly once per re-partition / assignment: {'V1_2': True, 'V1_3': True, 'V1_4': True, 'V2_1': True, 'V2_2': True, 'V2_3': True}

(d) passes: **True**

## (e) The filter code against B.70

F fit on the manifest training files of r2 f0 against `data/.b70/kf_params_r2_f0.json`: largest deviation of the three K0 parameters 0.0e+00, of the two gains 0.0e+00.

(e) passes: **True**

## (f) Per-row references

Causal GP of `data/.b64/base.npz` against the comparison files: share of D1 rows apart by more than 0.5 eV 0.00040 (largest 1249.64 eV); hold 0.00000 (largest 0.00 eV).

## Counts

D1 rows 222,672 of 635 discharges; rows with the target defined: F 222,672, N1 214,768, L3 204,733, L5 215,369, L10 215,667; classifier rows 240,792 (0.0665 positive).
