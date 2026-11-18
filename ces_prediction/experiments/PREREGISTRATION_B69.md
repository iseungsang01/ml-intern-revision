# B.69 — the two-stage estimator on the later discharges: a confirmation out of time

Registered 2026-11-19, after B.64 was read and before any run of this batch. GPU, with the second stage on a CPU. The session lead designed it. An AI agent implements and runs it under the author's supervision. No existing file is edited. It runs after B.68.

## 1. Why

B.64 made the two-stage estimator the paper's estimator. Its statements rest on three re-partitions of the pooled cross-validation, in which training and test discharges come from the same weeks of the campaign. The fourteenth grading named a departure from the registered rule as a main weakness. That departure was the move to the causal variant after B.39. The remedy is a fresh test of the estimator as now defined, registered before it runs, on a split it has not seen: the later discharges of the campaign.

## 2. Seen before

- **The temporal split of B.5.** The test block is the last 97 discharges by shot number. The earlier discharges are divided into training and validation by four seeds (`.b5_camp_cut_split_s{42,1,7,123}`).
- **B.29 on that split.** The causal variant over the causal GP: +0.103 to +0.112, interval above zero on all four seeds. Those runs had a 30-epoch cap and no twin, and no second stage was involved.
- **B.64, pooled r2 to r4.** Six statements, all met:

  | Comparison | Value |
  |---|---|
  | two-stage over the network | +0.093 [+0.078, +0.107] |
  | two-stage over the causal GP | +0.176 [+0.155, +0.196] |
  | two-stage over the two-stage twin | +0.097 [+0.077, +0.116] |
  | median absolute error over that of hold | 0.909 [0.889, 0.931] |

## 3. Design

**Split and environment.** The temporal split of B.5, with its four seeds, in the cut population. The environment of every network run is that of the pooled estimator of B.39 (`run_b39`):

- cap 80 with early stopping on the validation discharges;
- `CES_PER_SHOT_NORM=0`;
- fast channels standardized with the statistics of the training discharges;
- every `CES_*` variable pinned.

The split directory is the seed's temporal split. The initialisation equals the seed.

**Outer networks.** The network (`v2`) and the twin (`v2h`), trained on the training discharges of each seed: 4 × 2 = 8 runs.

**Out-of-fold networks.** As in B.64:

- The training discharges of each seed are divided into four inner folds.
- The network and the twin are each trained on three inner folds, with the seed's validation discharges for early stopping, and predict the fourth: 4 × 4 × 2 = 32 runs.
- The second stage is B.38's model with the same inputs, the same hyper-parameters and the same input split for the twin as in B.64. Per seed it is fitted on the out-of-fold predictions of the training discharges together with the validation discharges, and applied once to the test block.
- On test rows where the network's second-stage inputs are missing, both arms keep the first stage (B.64, addendum 1).

In all, 40 GPU runs.

**Readouts.** On the rows of the test block scored as in B.39:

- skill over the network, the causal GP, the two-stage twin, hold, PCHIP and the offline GP;
- RMSE in eV;
- median absolute error;
- share within 20 %.

Intervals are 95 %, 20,000 draws for sums and 2,000 for medians. Each draw takes one of the four seeds uniformly, then resamples the test discharges; this is the rule of B.39 for initialisations.

**Checks before any run.**

- The test block is the same for the four seeds and is disjoint from every training and validation set.
- The 40 environments differ from `run_b39.pooled_env` only in the split directory, the seed and the output directories.
- The second-stage code is that of B.64, imported, not copied.
- The causal-GP and hold references on the test block reproduce those stored for B.29's temporal runs.
- No tracked file is changed.

## 4. Statements (temporal test block, cut population)

- **C1.** Two-stage estimator over the causal GP: the interval is above zero.
- **C2.** Two-stage estimator over the network: the interval is above zero.
- **C3.** Two-stage estimator over the two-stage twin: the interval is above zero.
- **C4.** Median absolute error of the two-stage estimator divided by that of hold: the interval is below 1.

## 5. What goes into the paper

The four statements go into the main results as a confirmation out of time, registered before it ran, beside the pooled result.

- **All met.** The paper states that the estimator chosen under the pooled protocol holds on the later discharges of the campaign.
- **Any not met.** The paper states which comparison did not hold out of time, and the abstract names the pooled cross-validation as the condition of its numbers.

The estimator is not changed by this batch whatever its outcome.

## 6. Expectations

- C1: met, +0.12 to +0.20.
- C2: met, +0.05 to +0.10.
- C3: met, +0.04 to +0.10, with a wider interval than in B.64 (one block of 97 discharges).
- C4: met, 0.88 to 0.95.

## 7. Limits known in advance

- There is one test block of 97 discharges from the end of one campaign. The four seeds differ in the training and validation partition and the initialisation, not in the test discharges.
- The second stage's hyper-parameters come from B.38, chosen on discharges that include the test block. Its inputs and its form were fixed before this batch.
