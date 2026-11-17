# B.38 — the second stage fitted on the training discharges (GPU batch; registered, not yet run)

Registered 2026-11-16. Called for by the decision rules of B.37 (V1, V2 and V4 hold). Not run: the GPU was in use by another project when this was written, and training batches are not run on a CPU.

## 1. Why

A boosted correction of the causal variant's residual, fitted on the 127 or 128 validation discharges of a split and applied once to its test rows, has skill +0.053 to +0.082 over the causal variant (cut, four of four splits) and +0.020 to +0.053 over a linear stack; fitted on about 48 discharges it was not separated from zero. The number of discharges the second stage sees was the binding quantity. The 414 or 415 training discharges cannot be used directly, because the network's residuals on its own training discharges are not those it makes on unseen ones. Cross-fitting removes that.

## 2. Design

Cut population, natural cadence, the four file-level splits (seeds 42, 1, 7, 123; `data/.b1_w2cut_split_s{seed}`).

1. The training discharges of a split are divided into four folds by a seeded permutation (seed = split seed). For each fold the causal variant (`seq_v2`, fast channels standardized with training-discharge statistics, the environment of the B.29 causal-variant runs with every `CES_*` variable pinned in the runner, epoch cap 80, the split's validation discharges for early stopping) is trained on the other three folds and run forward on the held-out fold. 16 runs.
2. Hold and the causal GP are computed on those rows as in `run_b30.rows`. The second-stage models of B.37 (same 15 inputs, same settings) are fitted on the out-of-fold rows of the training discharges together with the validation rows, and applied once to the test rows of the frozen row table, whose network values are those of the existing causal variant (`data/.b29_norm0_cut_s{seed}`).
3. Before the readout, the check of B.37 (five test files recomputed against the row table) is repeated.

## 3. Statements

Intervals: 95 %, 1,000 resamples of test discharges; a statement holds on at least three of four splits.

- **G1.** The skill of the correction fitted on training and validation discharges over the correction fitted on validation discharges alone (B.37 V) has an interval above zero.
- **G2.** Its skill over the linear stack fitted on the same rows has an interval above zero.
- **G3.** The adaptive 90 % interval covers 0.87 to 0.93 of the test rows with a median half-width whose ratio to the constant half-width has an interval below 1.
- Reported: the skill over the causal variant, over the causal GP and over hold; the median absolute error over that of the causal variant and of hold; the same with the out-of-fold rows only; the skill of each out-of-fold network on its held-out fold against the full network's test skill.

## 4. Decision

- G1 holds: the second stage is trained inside the network next (a residual stage and a variance output trained jointly on the training discharges), registered separately with its own model file.
- G1 fails: 127 discharges already saturate this second stage; the manuscript keeps the validation-fitted numbers and the limit is stated as measured.

## 5. Expectations

G1: +0.005 to +0.040, holding on two to four splits. Skill over the causal variant +0.06 to +0.11. G2 holds. G3: coverage 0.88 to 0.91, holds. Each out-of-fold network has a skill over hold 0.00 to 0.03 below that of the full network.

## 6. Limits known in advance

A network trained on three quarters of the training discharges is slightly weaker than the full one, so its out-of-fold residuals are slightly larger than those of the network the correction is applied to. Boosted trees with default settings. Cut population.

## Amendment 1 (2026-11-17, before any run)

Section 2 says that the out-of-fold networks are trained under an epoch cap of 80. The existing causal variant (`data/.b29_norm0_cut_s{seed}`), whose test predictions the correction is applied to, was trained under a cap of 30 (B.29; it reached the cap on splits 42 and 7). The out-of-fold networks are therefore trained under the same cap of 30 and the same environment (`run_b29.run_env("norm0", "split", "cut", seed, ...)`, with only the split directory and the output directory changed), so that they differ from that network in the training discharges only. Folds: the sorted training files of a split are permuted with `numpy.random.default_rng(seed)` and fold k takes every fourth element of the permutation starting at k. Fold split directories: `data/.b38_split_s{seed}_f{k}`; runs: `data/.b38_norm0_cut_s{seed}_f{k}`. Nothing else changes.
