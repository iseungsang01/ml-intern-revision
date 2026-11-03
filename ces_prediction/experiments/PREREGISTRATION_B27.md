# PREREGISTRATION B.27 — shot-level repeated 5-fold replicate of the headline

Registered 2026-11-03, with code (`b27_kfold/run_b27.py`), before any B.27 number. Named in
THESIS_RESULTS.md §8bv.

## 1. Why

The headline (§8ab) is read by counting splits whose shot-cluster bootstrap CI excludes 0
("4/4"). §8bv showed the count is not a test (4/4 by chance = 1/16) and that the per-split CI
covers only which shots landed in TEST. The across-split t-interval it added is too narrow
because the four random splits share most TRAIN shots. B.27 fixes the design: every shot is
tested exactly once per repeat, and the interval is corrected for TRAIN overlap.

## 2. Design (one variable: the split design)

- **Folds.** Shot universe = the 639 files of the inclusive W = 2 manifest
  (`.b1_manifest_s42`; the cut population has 638 of them). Two repeats, partition seeds
  2701 and 2702; each repeat permutes the universe and cuts it into K = 5 folds
  (128/128/128/128/127 shots). For fold k: TEST = fold k; of the rest, the first 128 shots of
  the permutation are VAL (20 % of the universe, as in the headline) and the remaining 383 are
  TRAIN (headline: 415). Each shot is TEST once per repeat (asserted).
- **Model and env.** `seq_v2`, `CES_PER_SHOT_NORM=1`, `GATE_ENV` unchanged (W = 2, held-free,
  per-file cap 500, TEST row cap 40,000 per fold). Both populations: cut (`CES_TI_SPIKE_CUT_EV=3000`)
  and inclusive (0). Init seed 27000 + 10·r + k, distinct per run, so init variance is inside the
  interval. 2 populations × 2 repeats × 5 folds = **20 GPU runs**.
- **Harness stats.** `eval_seq.py` needs train-file-only window normalization stats. B.27 writes
  them with `dataset.fit_normalization_stats` on each fold's TRAIN files; the builder was checked
  to reproduce the frozen controls' stats bit-identically (`.b1_w2cut_s42`, `.wsweep_hf_w2_s42`)
  and the runner re-checks this before building any fold (stage `verify`).
- **Scoring.** Per fold, MSE skill = 1 − mean SE_model / mean SE_ref on the fold's TEST rows,
  ref ∈ {PCHIP, persistence, causal GP, GP}, per target, per population.

## 3. Readout and decision rule (primary)

For each population × target × reference, the 10 fold skills get the Nadeau–Bengio (2003)
corrected repeated-CV t-interval:

  mean ± t_{0.975, 9} · sqrt((1/10 + n_test/n_train) · s²),  n_test/n_train = mean TEST/TRAIN shot ratio (≈ 0.33).

- **beats** if the corrected 95 % interval lies above 0; **loses** if below; **undecided** if it
  spans 0. No other rule decides a claim.
- Secondary, reported, never deciding: the uncorrected t-interval, and the per-fold count of
  shot-cluster bootstrap passes (x/10).
- The frozen four-split headline and its registered 4/4 verdicts stay as recorded; B.27 is the
  replicate the paper's significance statements will cite.

## 4. Predictions (written before any number)

| cell | prediction | confidence |
|---|---|---|
| `T_i` vs PCHIP, persistence, causal GP (both populations) | beats | 0.9 each; causal GP 0.8 |
| `T_i` vs GP (offline) | undecided | 0.8 |
| `V_rot` vs persistence | beats | 0.75 |
| `V_rot` vs PCHIP | beats | 0.55 (§8bv lower bound +0.07 is the overlap-exposed one) |
| `V_rot` vs causal GP | undecided | 0.6 |
| `V_rot` vs GP | undecided | 0.85 |

Fold-level `T_i` vs PCHIP skill: +0.15 … +0.32 (slightly below the headline mean is possible:
TRAIN is 8 % smaller). Fold sd of `V_rot` skills larger than `T_i`'s by ≥ 2×.

## 5. What would change the paper

- `T_i` cell undecided → the headline significance statement is withdrawn to "replicated in
  4/4 random splits, not in the corrected interval".
- `V_rot` vs PCHIP beats → the paper may state the `V_rot` mean (tail) win with the corrected
  interval; the median tie at the photon floor (§8bs) is unaffected either way.
- Any verdict here supersedes the §8bv uncorrected interval for the same cell.

## 6. Not in scope

No architecture, loss, population or env change; no window-family control (the harness
stats replace it); the 4 frozen splits are not rerun.
