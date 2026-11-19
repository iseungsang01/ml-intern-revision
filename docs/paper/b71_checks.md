# B.71 — checks and timing before the run (registered: PREREGISTRATION_B71.md; choices: PREREGISTRATION_B71_ADDENDUM1.md)

Fold r2 f0, restricted to its first five test files (sorted: s30805, s30811, s30843, s30906, s30909). CPU, two threads, while B.70 phase 1 shared the machine. No error against a measured T_i was computed under any scenario; no readout exists. Sources: `data/.b71/checks_r2f0.json`, `data/.b71/timing.json`. Commands: `run_b71.py check --fold r2f0 --files 5 [--precision float32]`, `run_b71.py timing --fold r2f0 --files 5`.

## Registered checks (section 6) and the added ones (addendum 1, C6)

| check | what | float64 (the arithmetic of the run, C16) | float32 (as trained) |
|---|---|---|---|
| D1 mapping | B.70 D1 T_i rows of r2 f0 onto grid rows; \|grid T_i − y_true\| ≤ 0.01 eV | 38,107 rows, 126 discharges, largest 2.4e-4 eV: passes | same |
| 1 | rebuilt inputs with every sample available = build_blocks (allclose 1e-6), network and twin | 5 of 5 files: passes | passes |
| 2 | U of the network vs the B.70 D1 re-scoring (`data/.b70_rescore/b39p_v2_cut_r2_f0/`), absolute errors, ≤ 0.5 eV on ≥ 99.9 % | 839 of 839 rows, largest 3.2e-4 eV: passes | 839 of 839, largest 1.5e-4 eV: passes |
| 3 | a cut after the block end: F, R, G equal U exactly | 10 of 10 blocks: passes | passes |
| 4 | R stepped row by row, every sample available, vs the batched forward, ≤ 1e-4 eV | network 6.9e-13 eV, twin 8.6e-13 eV (1,084 rows each): passes | network 3.7e-4 eV, twin 2.8e-4 eV (124 and 125 rows above 1e-4): **fails** |
| 4b | batched F/G continuation from the U state, every sample available, vs the batched forward, ≤ 1e-4 eV | network 5.1e-13 eV, twin 4.3e-13 eV: passes | network 2.8e-4 eV, twin 2.3e-4 eV: **fails** |
| 5 | causal GP fitted once per cut vs `predict_gp_causal` row by row, ≤ 1e-6 eV | 811 rows, 6 cuts, largest 0.0 eV: passes | same |

The float32 deviations of checks 4 and 4b are at most 9.5e-7 in normalised units, about eight float32 units in the last place. In float64 the same comparisons agree within 1e-12 eV, so the stepping and the state continuation are exact and the float32 difference is rounding in the LSTM sums. The run therefore uses float64 arithmetic for every forward pass, and the registered 1e-4 eV tolerance is kept (addendum 1, C16).

Units built on the five files: 6 kept cuts, 810 F/R units, 1,100 G units. Structure test of the readout (every estimator replaced by 1,000 eV; nothing read out): every skill 0 and every ratio 1. Filter step: it reads B.70 phase 2's `data/.b70/kf_params_r{r}_f{k}.json`. Only r2 f0, f2 and f4 exist now, so `run_b71.py kf` stops with "filter rows left out". A code-path run with the r2 f0 parameters on the five check files (scratch only; no error computed) gave a filter output on all 810 F units. KF0 stayed constant after every cut (predict step only) and KFX moved with the BES drive.

## Timing and projection

| quantity | value |
|---|---|
| loading the D1 grid (641 files) | 38 s |
| fold setup (two networks, D1 rows) / staleness training range | 0.5 s / 0.1 s |
| F + R + G (network and twin), hold, causal GP on the five files | 2.8 s (6 kept cuts, 1,084 F rows) |
| checks 3-5 on five files per fold | 4.4 s |
| all 15 folds: test discharges / kept cuts / F rows | 1,917 / 5,298 / 1,182,369 |
| projection of the full batch | 0.7 h (by cuts) to 0.9 h (by F rows) |

The five check files are short discharges (1.2 kept cuts per file against 2.8 on average), so the projection is scaled by cuts and by replayed rows rather than by files. Both are far below the 8-hour stop of the registration (section 9).

## Filter rows (run 2026-11-20, all 15 folds, before the readout)

Parameters: B.70 phase 2's `data/.b70/kf_params_r{r}_f{k}.json`, all 15 present (KF0, KFX, `use_x`, clip 0.2, fold and held-repeat flags checked on load). Source: `data/.b71/kf_checks.json`.

| check | result |
|---|---|
| every F unit has a KF0 and a KFX output | 1,099,107 of 1,099,107 units (0 without an output) |
| at the first replay row t0 of every cut, F holds the samples of U, so KF0 F and KFX F equal B.70 phase 2's stored U-scenario filter estimates on that row (`data/.b70p2/b56/r{r}_f{k}.npz`), ≤ 1e-6 eV | 4,842 rows; KF0 0.0 eV, KFX 0.0 eV: passes |
| KF0 F constant after every cut (predict step only, no drive) | largest range within a cut 0.0 eV: passes |

The first `kf` run took time and T_i from the float32 grid and failed this check by up to 4.4e-3 eV. kfx_model's own loader, which phase 2 uses, reads both from the CSV in float64. After taking them from there (T_i still missing wherever the D1 grid has none), the deviation is 0.0 eV. The filter rows used by the readout are those of the second run (addendum 1, C14).

## Full run (2026-11-19/20)

- Rows: 15 folds, 2,533 s (float64, two threads, beside B.70 phase 2), with 5,298 kept cuts, 1,099,107 F/R units and 1,280,739 G units.
- Check 1 passes on every test file of every fold.
- Check 2: every D1 T_i row of every fold lies within 0.5 eV of B.70; the largest deviation is 6.6e-4 eV.
- Checks 3–5 pass on the first five test files of every fold: check 4/4b largest deviation 8.6e-13 eV, check 5 0 eV.
- Filter rows: 365 s. Readout: one run.
