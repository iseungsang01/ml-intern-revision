# B.68 — an auxiliary training target denoised by the neighbouring values

Registered 2026-11-19, before any run of this batch. GPU. The session lead designed it from the literature map (`docs/paper/literature_map.md`, section 4.3, item 1). An AI agent implements and runs it under the author's supervision. No existing file is edited; the change is applied by wrapper scripts at run time. It runs after B.67.

## 1. Why

Three quarters of the network's mean squared error is scatter of the reported value from frame to frame (PROJECT_KNOWLEDGE 4.38). The offline GP, which uses the values on both sides of a row, has lower error than the network (4.32). Auxiliary outputs that are available only in offline training can improve a causal predictor (Caruana 1997, pp. 15–16). A target smoothed by the neighbours could pull the network toward the part of the value it can predict, without any change at inference.

## 2. Seen before

These are pooled B.39 results on r2 to r4 or r2 to r9:

| Comparison | Skill |
|---|---|
| network over the twin, r2 to r4 | +0.053 [+0.027, +0.076] |
| network over the causal GP, r2 to r9 | +0.089 [+0.064, +0.112] |

The offline GP is not separated from the network at the pooled level and is lower at the point. Earlier attempts did not change the target the network is trained on:

- a two-component mixture head and a Gaussian head (B.17);
- a Huber loss, which lowered the median error (section 3 of PROJECT_KNOWLEDGE).

## 3. Design

**Folds and environment.** The 15 folds of re-partitions r2, r3 and r4 of the pooled cross-validation of B.39. The environment of every run is `run_b39.pooled_env("v2", r, k, out)`:

- cut population;
- cap 80 with early stopping;
- `CES_PER_SHOT_NORM=0`;
- the seeds and statistics of the fold.

There is one initialisation per fold, and run directories are `data/.b68_X_cut_r{r}_f{k}`. 15 runs.

**Auxiliary target.** For every row of every training and validation discharge, it is the offline-GP estimate of T_i at that row: the estimator of the offline-GP reference of B.39, which uses the measured values on both sides. Where T_i is measured, the row's own value is withheld. It is computed from the discharge's own values only; no test discharge enters training. It is standardized with the target statistics of the network, and it is missing where the offline GP gives no value.

**Change to the network.** The T_i branch of `seq_v2` gets one more output head, of the same form as its T_i head, on the same hidden state. The loss becomes the masked MSE of `seq_v2` plus 0.5 × the masked MSE of the auxiliary head on the auxiliary target. The weight 0.5 is fixed here and not tuned.

**Unchanged.** Early stopping uses the validation loss of the main T_i and V_rot heads only, as in `seq_v2`. Inference and scoring use the main head only. The checkpoint, the architecture used at scoring and `eval_seq.py` are those of `seq_v2`.

**Readouts.** As B.59/B.67 pooled: `analyze_b39.arm_p`, 20,000 draws, against the stored B.39 errors of the network (`v2`) and the twin (`v2h`) on the same folds. Also reported:

- skill over the causal GP and the offline GP;
- median absolute error and its ratio to that of the causal GP;
- rows 1 to 3 after an absent frame.

**Checks before any run.**

- The auxiliary target reproduces the stored offline-GP errors of B.39 on the test rows of r2 f0.
- The withheld-row rule holds: the auxiliary target at a measured row does not depend on that row's value.
- With the auxiliary weight set to 0, the wrapper's loss equals that of `seq_v2` on a CPU batch.
- The 15 environments equal those of B.59 arm N apart from the model choice and the directories.
- No tracked file changes.

## 4. Statements (r2 to r4 pooled, cut population)

- **X1.** The arm over the network: the interval is above zero.
- **X2.** The arm over the twin: the interval is above zero.
- **X3.** The median absolute error of the arm divided by that of the causal GP: the interval is below 1.

## 5. What goes into the paper

- **X1 met.** The paper states that training against a target smoothed by the neighbouring values raises the skill of the causal estimator by the stated amount, and names this as the next confirmatory model. The estimator of the paper is not changed by this batch.
- **X1 not met.** The paper states that the frame-to-frame scatter of the reported value is not what limits the fit.
- **X3.** It is reported beside the median errors of the main table.

## 6. Expectations

- X1: −0.01 to +0.03, uncertain.
- X2: +0.04 to +0.07, met.
- X3: uncertain; the network's median error is about that of the causal GP.

## 7. Limits known in advance

- There is one initialisation per fold and three re-partitions, and the weight 0.5 is not tuned.
- The auxiliary target uses future values. It is used only in training on training discharges, and inference stays causal.
