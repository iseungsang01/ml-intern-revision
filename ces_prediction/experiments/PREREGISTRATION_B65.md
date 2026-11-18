# B.65 — the pooled result on the population that keeps the values above 3 keV

Registered 2026-11-18, before any run of this batch. GPU (network and twin), CPU (references). Designed by the session lead after the thirteenth independent grading; implemented and run by an AI agent under the author's supervision.

## 1. Why

The confirmed protocol has two co-primary populations: values above 3 keV treated as missing (cut) and kept (inclusive). Every pooled result of the paper is on the cut population; the paper's estimator was never scored on the inclusive one under the pooled protocol (only an earlier whole-discharge variant on single splits: +0.016 to +0.024, 2 of 4). The grading reads the result as conditional on a cut applied to the target.

## 2. Seen before

Pooled cut, r2 to r9: network over causal GP +0.089 [+0.064, +0.112], over twin +0.048 [+0.022, +0.073]; on r2 to r4 alone over twin +0.053 [+0.027, +0.076]. Values above 3 keV are a small share of the measured rows and are associated with fit failures (B.43). On the inclusive population mean-squared-error skills are dominated by those rows; earlier single-split readouts there were small and unstable. B.37: a second stage on the inclusive population did not raise the mean-squared-error skill (1 of 4) but lowered the median error on 4 of 4.

## 3. Design

Re-partitions r2, r3, r4 of B.39 (15 folds), same discharges and folds; the network (`v2`) and its twin (`v2h`) with the B.39 environment and every `CES_*` variable pinned, the only change `CES_TI_SPIKE_CUT_EV=0` (inputs and targets keep the values above 3 keV; training, early stopping and scoring on them): 30 GPU runs, one initialisation per fold as in B.39. References on the same rows: hold, causal GP (the code of B.39 with the same population change), offline GP, PCHIP. Scoring rows: every measured row of the test discharges in the inclusive population. Readouts as B.39 pooled (`analyze_b39.arm_p`, 20,000 draws) plus the Huber skill (200 eV) and the 1 %-trimmed skill of B.61, root-mean-square error and median absolute error, and every quantity also on the subset of rows with values at or below 3 keV.

Checks before any statement: the run environments differ from B.39 only in the population and the directories; the causal GP of the cut population reproduces the stored B.39 errors on the cut rows of r2 to r4; no file is in two of training, validation and test.

## 4. Statements (r2 to r4 pooled, inclusive population)

- **I1.** Network over causal GP, mean-squared-error skill: interval above zero.
- **I2.** Network over twin: interval above zero.
- **I3.** Network over causal GP, Huber skill: interval above zero.
- **I4.** Network over twin, Huber skill: interval above zero.

## 5. What goes into the paper

The inclusive result goes into the main table as a second block with I1 to I4. All met: the paper states that the result does not depend on the cut. I1 or I2 not met while I3 and I4 are met: the paper states that on the inclusive population the mean-squared-error margin is decided by the values above 3 keV (fit failures) and the Huber margins remain. I3 or I4 not met: the paper states that the result is established on the cut population only, and the cut is named in the abstract as a condition.

## 6. Expectations

I1 +0.00 to +0.05, uncertain (values above 3 keV dominate the squared error). I2 +0.00 to +0.04, uncertain. I3 +0.04 to +0.08, met. I4 +0.02 to +0.05, met.

## 7. Limits known in advance

Three re-partitions, one initialisation. The values above 3 keV are mostly fit failures; scoring on them measures the estimator against analysis errors as well.
