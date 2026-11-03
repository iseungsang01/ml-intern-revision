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

## 7. Addendum B.27b — pooled out-of-fold readout (registered 2026-11-03, after §8bw, before computing it)

**Status: post hoc.** Written after the §3 verdicts were seen (§8bw: `V_rot` vs PCHIP undecided).
It cannot overturn a §3 verdict. The paper must label it "post hoc, pooled out-of-fold". It may
add one sentence beside the §3 verdict and never replaces it.

**Why.** The §3 statistic averages ten fold-level ratio skills. Each `V_rot` fold has ≈ 80 shots,
and a few heavy-error shots set a fold's ratio (fold range −0.11…+0.49). Pooling every
out-of-fold row of a repeat scores all 639 shots in one ratio, so no fold is held hostage by
its few tail shots.

**Statistic.** For each population × repeat r ∈ {0, 1}: concatenate the five TEST folds'
rows from `comparison_errors_test.npz`. Every shot appears exactly once. Skill =
1 − Σ SE_model / Σ SE_ref over the pooled rows, the same ratio-of-means as §2. The CI is the
paired shot-cluster bootstrap (B = 10 000, seed 2727, same resampling as `bootstrap_compare.py`),
95 % percentile.

**Decision rule (fixed now).** A cell **beats** iff the pooled 95 % CI lower bound is > 0 in
**both** repeats. It **loses** iff the upper bound is < 0 in both. Otherwise it is undecided.
Requiring both repeats is the guard for training-set variation: the two repeats are disjoint
model sets trained on different TRAIN partitions. All 16 cells are reported (2 populations ×
2 targets × 4 references). None is reported alone.

**Known limitation (stated before the number).** The bootstrap conditions on the trained models.
It does not carry the TRAIN-overlap correlation that the Nadeau–Bengio factor corrects
(Bates, Hastie & Tibshirani 2023: naive CV intervals can undercover). This is why it stays a
secondary, post hoc readout.

**Prediction.** `T_i` cells reproduce the §3 verdicts (0.9). `V_rot` vs persistence beats (0.85).
`V_rot` vs PCHIP beats in both populations (0.5). `V_rot` vs causal GP beats (0.4). `V_rot`
vs GP undecided (0.8).

## 8. B.27c — eight new repeats, two-level bootstrap as the primary readout for the mean-error claims (registered 2026-11-03, before any r2–r9 run)

**Why.** §8bx (post hoc) found `V_rot` beating PCHIP and causal GP when all shots are scored at
once. It conditioned on two trained model sets, and its rule was written after the §8bw numbers
were seen. B.27c tests that claim on **new data**: eight new partitions, eight new model sets.
The readout carries training variation by construction.

**Design.** Same as §2 except repeat seeds 2703–2710 (r2–r9). That is 2 populations × 8 repeats ×
5 folds = **40 new GPU runs**. Init seed 27000 + 10r + k. r0–r1 are reused unchanged. Runner:
`run_b27.py` with `REPEAT_SEEDS = 2701..2710`, which resumes past the 20 finished runs. The
machine is kept out of standby during the batch.

**Primary readout (decides).** `analyze_b27c.py`. Each repeat's five TEST folds are pooled.
One bootstrap draw (B = 20 000, seed 2728) does two things. (i) It picks one repeat uniformly
from **r2–r9 only**: that is model and training-partition variation. (ii) It resamples that
repeat's shots with replacement: that is shot sampling. Skill = 1 − ΣSE_model/ΣSE_ref.
**beats** iff the 2.5th percentile > 0, **loses** iff the 97.5th < 0, else undecided. All 16
cells are reported. r0–r1 are excluded from the primary readout because their numbers shaped
this rule.

**What the readout is.** The 95 % range of the skill that one model trained by this procedure
on ≈ 383 shots attains on a fresh sample of the same number of shots. It includes the full
single-model training variance (not divided by the number of repeats), so it is stricter than
an interval on the repeat average.

**Secondary (reported, never decides).** The same two-level bootstrap on all ten repeats.
Per-repeat pooled skills. The Nadeau–Bengio interval over 50 folds (`data/.b27_summary_R10.json`).

**Limitation (stated now).** All repeats draw TRAIN from the same 639 shots, so the model-level
spread can still understate variation across fully independent training sets. Only new shots
(another campaign) remove that.

**Paper rule.** If `V_rot` vs PCHIP **beats** here, the paper states the `V_rot` mean-error win
over interpolation as a registered result. It cites B.27c, and §8bw stays as the
fold-interval reading beside it. If it is undecided, the paper keeps §8bw as is, and §8bx stays
a labeled post hoc sentence. The median tie at the photon floor (§8bs) is stated in either case.

**Predictions.** `T_i` vs PCHIP, persistence, causal GP: beats (0.9). `T_i` vs GP: undecided (0.85).
`V_rot` vs persistence: beats (0.75). `V_rot` vs PCHIP: beats (0.45). `V_rot` vs causal GP:
beats (0.35). `V_rot` vs GP: undecided (0.85).
