# PREREGISTRATION B.29 — what the fast channels add on the main model, and whether the estimator stays ahead of the causal GP under a causal standardizer

Registered 2026-11-15, before any B.29 run and before any B.29 scoring. This document is committed
first; the code (`b29_causal/run_b29.py`, `b29_causal/analyze_b29.py`, `seq/model_seq_v2h.py`,
`seq/model_seq_mlp.py`, the `CES_SEQ_NORM_EXPANDING` knob in `seq/seq_data.py`, the tests) follows
in a separate commit. Named by the review loop (`docs/paper/review_loop/round_1.md` §7 items 1, 2,
7; `round_2.md` "우선순위 유지(GPU)") and by the adopted venue plan
(`docs/paper/review_loop/venue_plan/PLAN_review.md` §5 steps 2–3).

## 1. Why

Two statements of the 8-page paper rest on a comparison that was never run on the main model.

1. **Fast-channel increment.** Claim 1 is written as "the full-input causal estimator exceeds the
   CES-only causal GP by +0.113 [+0.085, +0.132] (cut); the fast-channel share of that margin is
   not isolated". The only ablation on record zeroes BES / ECEI / MC at evaluation time in the
   W = 2 windowed control (§8ab), which is an out-of-distribution input for that model and is not
   the main model. A `seq_v2` whose `T_i` branch is *trained* without the fast channels does not
   exist. The paper's title is about what the co-located fast diagnostics add; that quantity has to
   be measured.
2. **"Causal".** `seq_v2` is trained and scored with `CES_PER_SHOT_NORM=1`: the 15 fast channels
   are z-scored with the mean and standard deviation of the *whole discharge*, including rows after
   the scored row. The recurrent operator is causal; the standardizer is not. The paper now
   discloses this as offline per-discharge standardization and dropped the word "causal" from the
   headline. Whether the result needs the offline standardizer is a measurement, not a wording
   choice.

A third comparison the reviewers asked for costs one model file: a memoryless regression on the
same 22 inputs (the Diag2Diag-type baseline).

## 2. Design

Everything not named here is the confirmed protocol (`PREREGISTRATION_W2.md` §1–§3) and the
B.1 / B.5 backbone env: `run_b1_gate.GATE_ENV` (W = 2 harness, held-free, per-file cap 500,
`CES_MAX_TEST_SAMPLES=48000`), MSE loss, val-MSE early stopping (patience 6), epoch cap 30 (the
backbone's), batch 16, lr 1e-3, split seeds {42, 1, 7, 123}, init seed = split seed, two
populations (`CES_TI_SPIKE_CUT_EV=3000` cut / `0` inclusive). Split manifests, harness statistics
(`CES_CONTROL_METRICS`) and TEST rows are those of the frozen backbone runs, so every arm is scored
on the backbone's exact rows; `paired_model_compare.py` verifies this row for row before any
statistic.

**Controls (frozen, not retrained).**

| population | split family | backbone run | split dir | harness stats |
|---|---|---|---|---|
| cut | four file-level splits | `data/.b1_seqv2_s{s}_i{s}` | `data/.b1_w2cut_split_s{s}` | `data/.b1_w2cut_s{s}/metrics.json` |
| inclusive | four file-level splits | `data/.b5i_seqv2_s{s}` | `data/.b1_manifest_s{s}` | `data/.wsweep_hf_w2_s{s}/metrics.json` |
| cut / inclusive | temporal (campaign) split, B.5, four initialisations | `data/.b5_camp_{pop}_seq_s{i}` | `data/.b5_camp_{pop}_split_s{i}` | `data/.b5_camp_{pop}_win_s{i}/metrics.json` |

**Arms (one controlled variable each against the backbone).**

- **A — history-only `seq_v2`** (`CES_SEQ_MODEL=v2h`, `seq/model_seq_v2h.py`). The `T_i` encoder
  reads the 7 slow channels only (dt, and per target: carried last observation, staleness,
  has-observation flag); BES / ECEI / MC are *absent from its input*, not zeroed. Widths, layers,
  heads, dropout and the `V_rot` branch are `seq_v2`'s. Parameter count 347,970 (backbone 357,570;
  the difference is the 15 removed input columns of the first `T_i` LSTM layer). Four file-level
  splits × two populations = 8 runs. `CES_PER_SHOT_NORM=1` is pinned for env identity; no model
  input depends on it.
- **B — train-file standardization** (`CES_SEQ_MODEL=v2`, `CES_PER_SHOT_NORM=0`). The fast channels
  are z-scored with the train-file statistics of the split, the same statistics every window-family
  run uses. No row of the scored discharge enters the standardizer. Four file-level splits × two
  populations (8 runs) + the temporal split, four initialisations × two populations (8 runs).
- **C — expanding-window per-discharge standardization** (`CES_SEQ_MODEL=v2`,
  `CES_PER_SHOT_NORM=1`, `CES_SEQ_NORM_EXPANDING=1`). The registered rule, one rule, no tuning:
  for a discharge with rows i = 0, 1, … in file order and fast channel c, let
  u_i = (x_i − μ_c)/σ_c with (μ_c, σ_c) the train-file statistics of the split, k = i + 1 and
  N₀ = 50. Then

      m_i = (Σ_{j≤i} u_j) / (N₀ + k)
      v_i = (N₀ + Σ_{j≤i} u_j²) / (N₀ + k) − m_i²
      z_i = (u_i − m_i) / sqrt(v_i)          (sqrt(v_i) < 1e-6 → 1)

  i.e. the mean and variance of the discharge's rows up to and including row i, pooled with N₀
  pseudo-rows drawn from the train-file distribution. At i = 0 the value is within 2 % of the
  train-file z-score; as k grows it converges to the per-discharge z-score. The statistics run
  across the whole file (they are not reset at a ≥ 0.5 s gap), as the offline per-discharge
  statistics do. N₀ = 50 rows = 500 ms, the measured saturation of the `T_i` context (§8al); it is
  fixed here and no other value is run. A unit test proves that no row after i changes z_i (bit
  identical) and that the offline per-discharge standardizer fails the same test. Same 16 runs as B.
- **D — memoryless MLP** (`CES_SEQ_MODEL=mlp22`, `seq/model_seq_mlp.py`), cheap optional arm, run
  last. The same 22 inputs at the scored row, no recurrent state: `T_i` = MLP(22 → 160 → 160 →
  LayerNorm → 64 → 1), `V_rot` = MLP(7 slow → 64 → LayerNorm → 64 → 1) with `seq_v2`'s routing.
  The carried last observation and its staleness are among the 22 inputs, so "memoryless" means
  "no state beyond what the feature row carries". `CES_PER_SHOT_NORM=1`. 8 runs.

Total 48 runs, sequential on one GPU, order A → B → C → D (the two title/wording rules first).

**Per run:** `train_seq.py` → `eval_seq.py` (TEST) → `bootstrap_compare.py` (arm vs PCHIP /
linear / persistence / offline GP / causal GP, shot-clustered, B = 10,000, seed 12345) →
`paired_model_compare.py` twice: `paired_vs_backbone.json` (a = arm, b = backbone) and
`paired_backbone_vs_arm.json` (a = backbone, b = arm). Output `data/.b29_{arm}_{pop}_s{s}` and
`data/.b29_{arm}_camp_{pop}_s{i}`. Analysis: `b29_causal/analyze_b29.py` →
`docs/paper/b29_causal.json` + `docs/paper/b29_causal.md`.

**Statistic.** skill_X_vs_Y = 1 − ΣSE_X / ΣSE_Y on the paired TEST rows; 95 % interval = 2.5 / 97.5
percentiles of the shot-clustered bootstrap, 10,000 resamples. "Excludes 0" below always names the
side. The fast-channel increment is Δ_fast = skill of the backbone vs arm A (`T_i`); positive
means the backbone's error is lower.

## 3. Decision rules (fixed now)

- **H29-1 (primary, title).** Δ_fast (`T_i`, backbone vs history-only `seq_v2`) has a 95 % interval
  with lower bound > 0 on **≥ 3 of the 4 cut splits** → the paper keeps its title (what the
  co-located fast diagnostics add to the `T_i` past) and quotes Δ_fast per split as the
  fast-channel increment on the main model, replacing the windowed-control zeroing table.
  Otherwise → the paper is retitled to noise-limited estimation of the CES past, Claim 1 is
  restated around the history-only estimator, and the fast channels are reported as an increment
  whose interval includes 0. Inclusive population: the same count is reported; if the cut rule
  passes and the inclusive count is < 3/4, the increment is stated as conditional on the cut.
- **H29-1b (reported with H29-1).** History-only `seq_v2` `T_i` skill vs the causal GP has lower
  bound > 0 on ≥ 3/4 cut splits → "a learned CES-only filter exceeds the CES-only causal GP";
  otherwise the history-only estimator is stated as not separated from the causal GP. This does
  not move the title; it fixes the sentence the retitled paper would lead with.
- **H29-2 (primary, wording).** Arm B (`CES_PER_SHOT_NORM=0`) `T_i` skill vs the causal GP has
  lower bound > 0 on **≥ 3 of the 4 cut splits** → the word "causal" is restored for the
  estimator, the estimator named is the train-file-standardized one, and the numbers quoted with
  that word are arm B's, not the per-discharge backbone's. Otherwise "causal" is not restored on
  the strength of arm B.
- **H29-2c (temporal split).** Arm B `T_i` skill vs the causal GP has lower bound > 0 on ≥ 3 of
  the 4 initialisations of the cut temporal split → the restored wording holds across the
  campaign split. Otherwise it is limited to the file-level splits, unless arm C passes H29-3c.
- **H29-3 / H29-3c (expanding standardizer).** The same two rules applied to arm C.
- **Which causal estimator the paper names (fixed now).** B if B passes H29-2 and H29-2c. Else C
  if C passes H29-3 and H29-3c. Else whichever of B, C passes its file-level rule (B first), with
  the temporal-split failure stated. If neither passes its file-level rule, "causal" stays out of
  the title and abstract and the offline per-discharge standardization stays a stated limit.
- **Cost of causal standardization (reported with H29-2 / H29-3).** Paired `T_i` skill of the arm
  vs the per-discharge backbone, per split, with its interval. If the backbone's error is lower
  with the interval excluding 0 on ≥ 3/4 cut splits, the paper states that increment as what the
  offline standardizer buys; it is not folded into a causal claim.
- **H29-4 (arm D, secondary).** Paired `T_i` skill of the backbone vs the memoryless MLP has lower
  bound > 0 on ≥ 3/4 cut splits → "the recurrent state adds to a memoryless regression on the same
  22 inputs", with the per-split values; otherwise the memoryless row is reported as not separated
  from the backbone and no statement about recurrence is made. The MLP's skill vs persistence,
  PCHIP and the causal GP is tabulated either way.
- **Guards (reported, never deciding).** `V_rot`: paired skill of each arm vs the backbone and the
  count of splits where the backbone is better with the interval excluding 0. In arms A, B, C the
  `V_rot` branch receives exactly the backbone's inputs (it reads no fast channel), so a deficit
  there is initialisation / checkpoint-selection variance, not a design result. Epochs run, best
  epoch, and whether the 30-epoch cap bound. Arm A / B / C vs PCHIP, persistence, offline GP.
- Inclusive-population counts are reported beside every cut count with the same rule. No rule
  other than the ones above decides a claim; nothing here changes a registered headline number of
  the per-discharge backbone.

## 4. Predictions (written before any B.29 run)

| cell | prediction | confidence |
|---|---|---|
| Δ_fast, `T_i`, cut, per split | +0.05 … +0.14 (centre +0.09) | 0.6 |
| H29-1 (lower bound > 0 on ≥ 3/4 cut) | supported | 0.75 |
| Δ_fast, `T_i`, inclusive | +0.02 … +0.10; lower bound > 0 on ≥ 3/4 | 0.5 |
| history-only vs PCHIP, `T_i`, cut | +0.05 … +0.15, lower bound > 0 on 4/4 | 0.65 |
| H29-1b (history-only vs causal GP, ≥ 3/4 cut) | not supported; point values −0.03 … +0.05 | 0.65 |
| arm B vs causal GP, `T_i`, cut, file-level | +0.06 … +0.14; H29-2 supported | 0.8 |
| arm B vs backbone, `T_i`, cut, file-level | −0.03 … +0.01; backbone better with interval excluding 0 on 0–1 of 4 | 0.65 |
| arm B, temporal split, vs backbone, `T_i`, cut | −0.12 … −0.02 (fast channels drift 1.22σ across the campaign, §8n) | 0.6 |
| H29-2c (arm B vs causal GP on ≥ 3/4 inits) | supported | 0.45 |
| arm C vs causal GP, `T_i`, cut, file-level; H29-3 | +0.07 … +0.14; supported | 0.8 |
| arm C vs backbone, `T_i`, cut, file-level | −0.03 … +0.01 | 0.65 |
| H29-3c (arm C, temporal split) | supported; vs backbone −0.04 … +0.01 | 0.7 |
| estimator named by the rule | C (B fails the temporal rule) | 0.4; B 0.4; neither 0.2 |
| H29-4 (backbone vs MLP, `T_i`, ≥ 3/4 cut) | supported; paired +0.10 … +0.25 | 0.8 |
| MLP vs PCHIP, `T_i`, cut | −0.15 … +0.05 | 0.6 |
| `V_rot` guard, any arm of A / B / C | backbone better with interval excluding 0 on 0–1 of 4 | 0.75 |
| run time | 3–20 min per run; batch 4–8 h | 0.7 |

Basis: §8ab (zeroing in the windowed control −0.25 … −0.43, cut; a history-only window model keeps
+0.15 … +0.23 over PCHIP in the inclusive population), review-loop A2 (a VAL-tuned running mean sits
at −0.060 … +0.003 vs the causal GP and the backbone is +0.11 … +0.14 above it), B.28 (the
fast-channel term carries a small share of the persistence-relative skill), §8t Finding 5
(per-discharge standardization moved the W = 4 `seq_v2` `T_i` by about +0.01), §8s / §8n (the
window model needs per-discharge standardization on the temporal split).

## 5. What would change the paper

- H29-1 supported → title kept; one table row "history-only `seq_v2`" and the sentence "the fast
  channels lower the `T_i` error of the main model by Δ_fast" with per-split intervals.
- H29-1 not supported → retitle (noise-limited estimation of the CES past); the fast channels
  become a reported increment whose interval includes 0; NN-CES / Diag2Diag distinction rewritten.
- H29-2 or H29-3 supported → "causal" restored for the named estimator with that arm's numbers;
  the per-discharge backbone's numbers stay as the offline-standardized row.
- Neither → "causal" stays out; the limit sentence stays; a learned or online standardizer is
  named as the measurement that would overturn it.
- The paper is not edited by this batch; results go to `THESIS_RESULTS.md` §8 first.

## 6. Disclosures

- Arm A removes 9,600 parameters with the 15 input columns; capacity is not re-matched (B.4 found
  `T_i` skill flat from 34k to 879k parameters, §8aa).
- Arms B and C use train-file statistics; for a *training* discharge those include its own rows.
  Scored (TEST) discharges never contribute to them.
- The `T_i` and `V_rot` branches share the early-stopping criterion (summed val MSE), so an arm
  that changes only the `T_i` input can still select a different `V_rot` checkpoint.
- Initialisation is seed-matched, not stream-matched: a different first-layer shape consumes the
  generator differently. The paired interval covers row noise, not init noise (B.15: 4-init spread
  of the backbone's `T_i` skill ≈ 0.02).
- The four file-level TEST sets overlap in discharges; the rules count splits and no pooled
  interval over the four is claimed.
- N₀ is not tuned and no sensitivity row is registered; a different N₀ would be a new registration.
- A plumbing smoke of every arm (split 42, 40 train files, 2 epochs, ≤ 1,200 TEST samples, into
  `data/.smoke_b29_*`) is run after this document is committed and before the batch. Its numbers
  are not results and are not quoted; it is recorded in §8 of this file when done.

## 7. Not in scope

No change to the backbone, the loss, the populations, the splits or the harness. No retraining of
any control. No surrogate (B.28) refit, no artificial-masking curve, no `V_rot`-branch routing
ablation, no GBM, no AR(p)/Kalman rows (CPU items of the review loop, separate). No CPU substitute
for the GPU batch.
