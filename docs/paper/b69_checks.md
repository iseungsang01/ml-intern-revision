# B.69 checks before any GPU run

Registration: ces_prediction/experiments/PREREGISTRATION_B69.md. CPU only (CUDA_VISIBLE_DEVICES empty), no network trained. Run 2026-11-19 05:14 by check_b69.py; 0.3 min. Result: ALL PASS.

## (a) Dry run

`run_b69.py --dry-run`: rc 0; 40 run lines; summary line: `40 runs: outer 8, inner 32; v2 20, v2h 20`; env line: `40 of 40 runs differ only in CES_CONTROL_METRICS, CES_INIT_SEED, CES_OUTPUT_DIR, CES_SEED, CES_SPLIT_DIR`.

| # | kind | run | # | kind | run |
|---|---|---|---|---|---|
| 1 | outer | .b69_v2_camp_cut_s42 | 2 | outer | .b69_v2h_camp_cut_s42 |
| 3 | outer | .b69_v2_camp_cut_s1 | 4 | outer | .b69_v2h_camp_cut_s1 |
| 5 | outer | .b69_v2_camp_cut_s7 | 6 | outer | .b69_v2h_camp_cut_s7 |
| 7 | outer | .b69_v2_camp_cut_s123 | 8 | outer | .b69_v2h_camp_cut_s123 |
| 9 | inner | .b69_v2_camp_cut_s42_i0 | 10 | inner | .b69_v2h_camp_cut_s42_i0 |
| 11 | inner | .b69_v2_camp_cut_s42_i1 | 12 | inner | .b69_v2h_camp_cut_s42_i1 |
| 13 | inner | .b69_v2_camp_cut_s42_i2 | 14 | inner | .b69_v2h_camp_cut_s42_i2 |
| 15 | inner | .b69_v2_camp_cut_s42_i3 | 16 | inner | .b69_v2h_camp_cut_s42_i3 |
| 17 | inner | .b69_v2_camp_cut_s1_i0 | 18 | inner | .b69_v2h_camp_cut_s1_i0 |
| 19 | inner | .b69_v2_camp_cut_s1_i1 | 20 | inner | .b69_v2h_camp_cut_s1_i1 |
| 21 | inner | .b69_v2_camp_cut_s1_i2 | 22 | inner | .b69_v2h_camp_cut_s1_i2 |
| 23 | inner | .b69_v2_camp_cut_s1_i3 | 24 | inner | .b69_v2h_camp_cut_s1_i3 |
| 25 | inner | .b69_v2_camp_cut_s7_i0 | 26 | inner | .b69_v2h_camp_cut_s7_i0 |
| 27 | inner | .b69_v2_camp_cut_s7_i1 | 28 | inner | .b69_v2h_camp_cut_s7_i1 |
| 29 | inner | .b69_v2_camp_cut_s7_i2 | 30 | inner | .b69_v2h_camp_cut_s7_i2 |
| 31 | inner | .b69_v2_camp_cut_s7_i3 | 32 | inner | .b69_v2h_camp_cut_s7_i3 |
| 33 | inner | .b69_v2_camp_cut_s123_i0 | 34 | inner | .b69_v2h_camp_cut_s123_i0 |
| 35 | inner | .b69_v2_camp_cut_s123_i1 | 36 | inner | .b69_v2h_camp_cut_s123_i1 |
| 37 | inner | .b69_v2_camp_cut_s123_i2 | 38 | inner | .b69_v2h_camp_cut_s123_i2 |
| 39 | inner | .b69_v2_camp_cut_s123_i3 | 40 | inner | .b69_v2h_camp_cut_s123_i3 |

- **(a) dry run**: PASS -- 40 runs in the registered order: outer runs first (seed 42, 1, 7, 123; v2 then v2h), then the inner runs (seed, inner fold 0-3, v2 then v2h): True

## (b) The test block

| quantity | value |
|---|---|
| test files per seed | s42 97, s1 97, s7 97, s123 97 |
| test block identical for the four seeds | True |
| test block shot numbers | 32312 to 32751 |
| training / validation shot numbers | 30801 to 31991 / 32002 to 32310 |
| training / validation sizes | 413 / 128 |
| training and validation sets identical for the four seeds | True |
| sets checked against the test block / largest intersection | 56 / 0 |
| test discharges with scored T_i rows (B.29 s42 scoring) / rows | 95 / 20,062 |

Note: the B.5 temporal split's training (shots up to 31991) and validation (32002 to 32310) discharges are the same for the four seeds; the seeds differ in the initialisation and, in B.69, in the inner-fold partition (rng seed (s, 0)), not in the training / validation partition.

- **(b) test block**: PASS -- 97 discharges, identical for the four seeds, disjoint from all 56 training / validation / inner sets

## (c) Environments

Reference: `run_b39.pooled_env(model, r2, f0)` (the env of `.b39p_v2_cut_r2_f0` / `.b39p_v2h_cut_r2_f0`).

| run | CES variables that differ from the reference |
|---|---|
| .b69_v2_camp_cut_s42 | CES_CONTROL_METRICS, CES_INIT_SEED, CES_OUTPUT_DIR, CES_SEED, CES_SPLIT_DIR |
| .b69_v2h_camp_cut_s42 | CES_CONTROL_METRICS, CES_INIT_SEED, CES_OUTPUT_DIR, CES_SEED, CES_SPLIT_DIR |
| .b69_v2_camp_cut_s1 | CES_CONTROL_METRICS, CES_INIT_SEED, CES_OUTPUT_DIR, CES_SEED, CES_SPLIT_DIR |
| .b69_v2h_camp_cut_s1 | CES_CONTROL_METRICS, CES_INIT_SEED, CES_OUTPUT_DIR, CES_SEED, CES_SPLIT_DIR |
| .b69_v2_camp_cut_s7 | CES_CONTROL_METRICS, CES_INIT_SEED, CES_OUTPUT_DIR, CES_SEED, CES_SPLIT_DIR |
| .b69_v2h_camp_cut_s7 | CES_CONTROL_METRICS, CES_INIT_SEED, CES_OUTPUT_DIR, CES_SEED, CES_SPLIT_DIR |
| .b69_v2_camp_cut_s123 | CES_CONTROL_METRICS, CES_INIT_SEED, CES_OUTPUT_DIR, CES_SEED, CES_SPLIT_DIR |
| .b69_v2h_camp_cut_s123 | CES_CONTROL_METRICS, CES_INIT_SEED, CES_OUTPUT_DIR, CES_SEED, CES_SPLIT_DIR |
| .b69_v2_camp_cut_s42_i0 | CES_CONTROL_METRICS, CES_INIT_SEED, CES_OUTPUT_DIR, CES_SEED, CES_SPLIT_DIR |
| .b69_v2h_camp_cut_s42_i0 | CES_CONTROL_METRICS, CES_INIT_SEED, CES_OUTPUT_DIR, CES_SEED, CES_SPLIT_DIR |
| .b69_v2_camp_cut_s42_i1 | CES_CONTROL_METRICS, CES_INIT_SEED, CES_OUTPUT_DIR, CES_SEED, CES_SPLIT_DIR |
| .b69_v2h_camp_cut_s42_i1 | CES_CONTROL_METRICS, CES_INIT_SEED, CES_OUTPUT_DIR, CES_SEED, CES_SPLIT_DIR |
| .b69_v2_camp_cut_s42_i2 | CES_CONTROL_METRICS, CES_INIT_SEED, CES_OUTPUT_DIR, CES_SEED, CES_SPLIT_DIR |
| .b69_v2h_camp_cut_s42_i2 | CES_CONTROL_METRICS, CES_INIT_SEED, CES_OUTPUT_DIR, CES_SEED, CES_SPLIT_DIR |
| .b69_v2_camp_cut_s42_i3 | CES_CONTROL_METRICS, CES_INIT_SEED, CES_OUTPUT_DIR, CES_SEED, CES_SPLIT_DIR |
| .b69_v2h_camp_cut_s42_i3 | CES_CONTROL_METRICS, CES_INIT_SEED, CES_OUTPUT_DIR, CES_SEED, CES_SPLIT_DIR |
| .b69_v2_camp_cut_s1_i0 | CES_CONTROL_METRICS, CES_INIT_SEED, CES_OUTPUT_DIR, CES_SEED, CES_SPLIT_DIR |
| .b69_v2h_camp_cut_s1_i0 | CES_CONTROL_METRICS, CES_INIT_SEED, CES_OUTPUT_DIR, CES_SEED, CES_SPLIT_DIR |
| .b69_v2_camp_cut_s1_i1 | CES_CONTROL_METRICS, CES_INIT_SEED, CES_OUTPUT_DIR, CES_SEED, CES_SPLIT_DIR |
| .b69_v2h_camp_cut_s1_i1 | CES_CONTROL_METRICS, CES_INIT_SEED, CES_OUTPUT_DIR, CES_SEED, CES_SPLIT_DIR |
| .b69_v2_camp_cut_s1_i2 | CES_CONTROL_METRICS, CES_INIT_SEED, CES_OUTPUT_DIR, CES_SEED, CES_SPLIT_DIR |
| .b69_v2h_camp_cut_s1_i2 | CES_CONTROL_METRICS, CES_INIT_SEED, CES_OUTPUT_DIR, CES_SEED, CES_SPLIT_DIR |
| .b69_v2_camp_cut_s1_i3 | CES_CONTROL_METRICS, CES_INIT_SEED, CES_OUTPUT_DIR, CES_SEED, CES_SPLIT_DIR |
| .b69_v2h_camp_cut_s1_i3 | CES_CONTROL_METRICS, CES_INIT_SEED, CES_OUTPUT_DIR, CES_SEED, CES_SPLIT_DIR |
| .b69_v2_camp_cut_s7_i0 | CES_CONTROL_METRICS, CES_INIT_SEED, CES_OUTPUT_DIR, CES_SEED, CES_SPLIT_DIR |
| .b69_v2h_camp_cut_s7_i0 | CES_CONTROL_METRICS, CES_INIT_SEED, CES_OUTPUT_DIR, CES_SEED, CES_SPLIT_DIR |
| .b69_v2_camp_cut_s7_i1 | CES_CONTROL_METRICS, CES_INIT_SEED, CES_OUTPUT_DIR, CES_SEED, CES_SPLIT_DIR |
| .b69_v2h_camp_cut_s7_i1 | CES_CONTROL_METRICS, CES_INIT_SEED, CES_OUTPUT_DIR, CES_SEED, CES_SPLIT_DIR |
| .b69_v2_camp_cut_s7_i2 | CES_CONTROL_METRICS, CES_INIT_SEED, CES_OUTPUT_DIR, CES_SEED, CES_SPLIT_DIR |
| .b69_v2h_camp_cut_s7_i2 | CES_CONTROL_METRICS, CES_INIT_SEED, CES_OUTPUT_DIR, CES_SEED, CES_SPLIT_DIR |
| .b69_v2_camp_cut_s7_i3 | CES_CONTROL_METRICS, CES_INIT_SEED, CES_OUTPUT_DIR, CES_SEED, CES_SPLIT_DIR |
| .b69_v2h_camp_cut_s7_i3 | CES_CONTROL_METRICS, CES_INIT_SEED, CES_OUTPUT_DIR, CES_SEED, CES_SPLIT_DIR |
| .b69_v2_camp_cut_s123_i0 | CES_CONTROL_METRICS, CES_INIT_SEED, CES_OUTPUT_DIR, CES_SEED, CES_SPLIT_DIR |
| .b69_v2h_camp_cut_s123_i0 | CES_CONTROL_METRICS, CES_INIT_SEED, CES_OUTPUT_DIR, CES_SEED, CES_SPLIT_DIR |
| .b69_v2_camp_cut_s123_i1 | CES_CONTROL_METRICS, CES_INIT_SEED, CES_OUTPUT_DIR, CES_SEED, CES_SPLIT_DIR |
| .b69_v2h_camp_cut_s123_i1 | CES_CONTROL_METRICS, CES_INIT_SEED, CES_OUTPUT_DIR, CES_SEED, CES_SPLIT_DIR |
| .b69_v2_camp_cut_s123_i2 | CES_CONTROL_METRICS, CES_INIT_SEED, CES_OUTPUT_DIR, CES_SEED, CES_SPLIT_DIR |
| .b69_v2h_camp_cut_s123_i2 | CES_CONTROL_METRICS, CES_INIT_SEED, CES_OUTPUT_DIR, CES_SEED, CES_SPLIT_DIR |
| .b69_v2_camp_cut_s123_i3 | CES_CONTROL_METRICS, CES_INIT_SEED, CES_OUTPUT_DIR, CES_SEED, CES_SPLIT_DIR |
| .b69_v2h_camp_cut_s123_i3 | CES_CONTROL_METRICS, CES_INIT_SEED, CES_OUTPUT_DIR, CES_SEED, CES_SPLIT_DIR |

| quantity | value |
|---|---|
| runs whose diff is exactly CES_CONTROL_METRICS, CES_INIT_SEED, CES_OUTPUT_DIR, CES_SEED, CES_SPLIT_DIR | 40 of 40 |
| outer envs equal to `run_b29.run_env(arm, 'camp', 'cut', s, out, epochs='80', model)` | 8 of 8 |
| v2 outer env vs B.29's registered temporal run of the same seed (cap 30): differing variables | CES_OUTPUT_DIR, CES_SEQ_EPOCHS; CES_OUTPUT_DIR, CES_SEQ_EPOCHS; CES_OUTPUT_DIR, CES_SEQ_EPOCHS; CES_OUTPUT_DIR, CES_SEQ_EPOCHS |
| B.29 temporal runs' metrics.json: seed = init = s, train-file standardization, held-free, v2 | 4 of 4 |
| a CES_* variable set in the shell reaches a run env | False |

CES_CONTROL_METRICS is the split's harness statistics (read by eval_seq only, for the reference round trip); it is counted with the split directory, as B.29 paired it with `.b5_camp_cut_split_s{s}`.

- **(c) environments**: PASS -- every env differs from the pooled estimator's only in the split directory (and its harness statistics), the seed and the output directory

## (d) References on the test block

| quantity | value |
|---|---|
| B.29 temporal runs `.b29_norm0_camp_cut_s{s}`: rows and references (shot, row, y_true, se_persistence, se_gp_causal, se_gp, se_pchip) identical to s42, bit for bit | s42 True, s1 True, s7 True, s123 True |
| CPU eval_seq under the B.69 outer env of `.b69_v2_camp_cut_s42` (device cpu) on `.b29_norm0_camp_cut_s42`'s checkpoint (1.2 min): rows 20,062; equal bit for bit to the B.29 stored ones: shot True, row True, y_true True, se_persistence True, se_gp_causal True, se_gp True, se_pchip True; the network's own error vs B.29's GPU scoring: max |difference| 0.201 eV, rows above 0.5 eV 0 |
| CPU eval_seq under the B.69 outer env of `.b69_v2h_camp_cut_s42` (device cpu) on `.b29a1_v2h_cut_s42`'s checkpoint (1.5 min): rows 20,062; equal bit for bit to the B.29 stored ones: shot True, row True, y_true True, se_persistence True, se_gp_causal True, se_gp True, se_pchip True |

The CPU scoring env differs from the B.69 outer env only in: CES_OUTPUT_DIR, CES_SEQ_DEVICE (plus CUDA_VISIBLE_DEVICES and thread counts). The seeds 1, 7, 123 differ from 42 in CES_SEED / CES_INIT_SEED only, and B.29's references, scored under CES_SEED = s, are identical across the four seeds (first row).

- **(d) references**: PASS -- hold, causal GP, offline GP and PCHIP references on the test block reproduce B.29's stored temporal references bit for bit under the B.69 env (both models)

## (e) The B.64 second stage through stage2_b69's binding

| array | equal bit for bit | max abs difference |
|---|---|---|
| est_c | True | 0 |
| est_c2 | True | 0 |
| est_c2nan | True | 0 |
| est_t | True | 0 |
| est_t2 | True | 0 |
| est_t2nan | True | 0 |
| fallback_c2 | True | 0 |
| fallback_t2 | True | 0 |
| rows | True | 0 |
| y | True | 0 |

| quantity | value |
|---|---|
| B.69 base rows (`data/.b69/base.npz`) equal to B.64's (`data/.b64/base.npz`), every array | True |
| `stage2_b64.fit_fold` run through `stage2_b69.bound` on r2 f0 (B.39 outer networks, B.64 inner networks, B.54 row table) | r2 f0: cached (0.0 min) |
| fold information equal to the stored one (apart from run times) | True |
| stage2_b64's module names restored after the call | True |

- **(e) B.64 second stage imported**: PASS -- fit_fold of stage2_b64, unchanged, reproduces the stored test-row estimates of B.64 fold r2 f0 (both arms, two-stage, fallback and missing-input variants) bit for bit

## (h) Plumbing of the B.69 path on stand-ins (NOT A RESULT)

Stand-ins for seed 42: the outer AND inner network = B.29's temporal network `.b29_norm0_camp_cut_s42` (cap 30); the twin = B.29 addendum 1's `.b29a1_v2h_cut_s42` (trained on a file-level split that shares discharges with the test block); both scored on the CPU in (d). The out-of-fold rows are in-sample, so nothing here is a result.

| quantity | value |
|---|---|
| rows / discharges with rows / test-block discharges | 20,062 / 95 / 97 |
| stage2_b64.fit_fold on the temporal block | r42 fcamp: cached (0.1 min incl. readout) |
| v2: recomputed first stage / hold / causal GP vs the eval_seq errors, rows above 0.5 eV (of 20,062); max deviation (eV) | 0 / 0 / 15; 0.000174 / 0.000244 / 653 |
| v2h: recomputed first stage / hold / causal GP vs the eval_seq errors, rows above 0.5 eV (of 20,062); max deviation (eV) | 0 / 0 / 15; 0.000168 / 0.000244 / 653 |
| test rows that keep the first stage (B.64 addendum 1): network / twin | 0.0469 / 0.0469 |
| statements computed (values meaningless): C1 / C2 / C3 / C4 | +0.148 [+0.116, +0.177] / +0.044 [+0.020, +0.067] / +0.026 [+0.005, +0.047] / 1.000 [0.931, 1.082] |
| readout written | `data/.b69_plumb/b69_time_confirm_PLUMBING.md` |

- **(h) plumbing**: PASS -- rows, the imported second stage and the readout run end to end on the temporal block

## (g) analyze_b69.py with no run

| call | rc | last line | files written |
|---|---|---|---|
| `analyze_b69.py --partial` | 0 | [b69] --partial: no seed has both outer runs scored yet; nothing to read, nothing written | none |
| `analyze_b69.py` | 1 | [b69] not complete (44 missing, first: ['.b69_v2_camp_cut_s42', '.b69_v2_camp_cut_s42_i0', '.b69_v2_camp_cut_s42_i1', '.b69_v2_camp_cut_s42_i2']); nothing written (use --partial to read what is complete) | none |

Seeds with both outer runs scored when this check ran: 0.

- **(g) analyze with no run**: PASS -- --partial exits cleanly and writes nothing; without --partial it stops and writes nothing

## GPU time (reported, from earlier logs)

Median minutes per run: outer (train + eval, B.39 pooled cap 80) v2 4.3, v2h 4.8; inner (train only, B.64) v2 2.2, v2h 1.9. Estimate for the 40 runs: 1.7 h (8 outer 36 min, 32 inner 66 min); the second stage and the readout run on the CPU afterwards (minutes).

## (f) Tracked files

`git status --porcelain`: tracked files changed under `ces_prediction/` or in B.69's write set (`docs/paper/b69_*`, `data/`): 0. New files of B.69: `ces_prediction/experiments/b69_time_confirm/analyze_b69.py`, `ces_prediction/experiments/b69_time_confirm/check_b69.py`, `ces_prediction/experiments/b69_time_confirm/run_b69.py`, `ces_prediction/experiments/b69_time_confirm/stage2_b69.py`, `docs/paper/b69_checks.md`.

Other tracked files changed in the working tree while this check ran: 13, in `docs/paper/ppcf/`. They are outside every path the B.69 scripts write (data/.b69*, docs/paper/b69_*); they come from concurrent work on the paper, not from this batch.

- **(f) no tracked file changed by B.69**: PASS -- nothing tracked under ces_prediction/ or in B.69's write set changed; B.69 adds only new files

