# B.64 — the two-stage estimator under the pooled protocol: does a second stage fitted on training discharges make the gain larger and reach the typical frame, and does the fast-channel increment survive it?

Registered 2026-11-18, before any run of this batch. GPU for the out-of-fold networks, CPU for the second stage. Designed by the session lead after the thirteenth independent grading; implemented and run by a delegated agent (an AI agent working under the author's supervision).

## 1. Why

The thirteenth grading names the small size of the effect as the third-heaviest weakness: on the pooled rows the recurrent estimator lowers the root-mean-square error by 8.8 eV (4.6 %) against the causal GP, does not lower the median absolute error below hold and does not raise the share of frames within 20 % of the value; the referee adds that the paper's own evidence favours another estimator. On four single file-level splits a second stage fitted on the residual of the causal variant, using out-of-fold predictions on the training discharges and the validation discharges (B.38), gave +0.078 to +0.113 over the causal variant and a median absolute error 0.861 to 0.924 of hold, all four excluding the reference. That design was never run under the pooled protocol, and never on the twin, so the increment of the fast channels after a second stage is unknown.

## 2. Seen before

B.38 (four splits, cut population): second stage over the causal variant +0.078 [+0.043, +0.113], +0.113, +0.087, +0.085; median absolute error 40.3, 45.8, 42.8, 38.4 eV against hold 44.2, 49.5, 49.8, 43.2. B.37: the same model fitted on validation discharges only gave +0.053 to +0.082; fitted on about 48 test discharges it gave nothing. Pooled B.39 on r2 to r4 alone: network over twin +0.053 [+0.027, +0.076]. Pooled (r2 to r9): network over causal GP +0.089 [+0.064, +0.112], over twin +0.048 [+0.022, +0.073]; median absolute error network 45.8, hold 45.4, causal GP 44.1 eV; within 20 % network 0.770, GP 0.772. Not seen: any second stage under the pooled protocol; any second stage on the twin.

## 3. Design

T_i, cut population (`CES_TI_SPIKE_CUT_EV=3000`), re-partitions r2, r3, r4 of B.39 (15 outer folds), the B.39 environment of the network (`v2`) and its twin (`v2h`) with every `CES_*` variable pinned as in `run_b39.split_env`.

**Out-of-fold networks.** For each outer fold, its training discharges are divided into four inner folds (seeded by the re-partition and the fold); the network and the twin are each trained on three inner folds with the outer fold's validation discharges for early stopping, and predict the fourth: 15 x 4 x 2 = 120 runs. The outer-fold networks of B.39 (stored) give the predictions on the validation and test discharges.

**Second stage.** The model of B.38 (same inputs, same hyper-parameters, same target: the residual of the first-stage estimate), fitted per outer fold on the out-of-fold predictions of its training discharges together with its validation discharges, applied once to its test discharges. For the network it takes the inputs of B.38. For the twin it takes the same inputs except every input computed from BES, ECEI or Mirnov; so the two-stage twin uses the CES history only, and the two-stage increment is the information of the fast channels after both stages. If B.38's inputs cannot be split that way, the agent stops before any fit and the session lead writes an addendum.

**Readouts.** Pooled over r2 to r4 as in B.39 (`analyze_b39.arm_p`, two-level bootstrap, 20,000 draws), on the rows scored by B.39. Mean-squared-error skill; root-mean-square error in eV; Huber skill (200 eV) as in B.61; median absolute error; share of rows within 20 % and 10 % of the value as in B.60.

Checks before any statement: the stored B.39 errors of the network and the twin are reproduced on r2 to r4 (skill +0.053 [+0.027, +0.076] within 0.002); every inner-fold run has its artifacts and early-stopped before the epoch cap or is reported; no file is in two of training, inner held-out, validation and test.

## 4. Statements (r2 to r4 pooled)

- **T1.** Two-stage network over the network: skill interval above zero.
- **T2.** Two-stage network over the causal GP: skill interval above zero, and its point value above that of the network over the causal GP on the same rows.
- **T3.** Two-stage network over the two-stage twin (the fast-channel increment after a second stage): interval above zero.
- **T4.** Median absolute error of the two-stage network below that of hold: the interval of the ratio below 1.
- **T5.** Share within 20 % of the value: two-stage network above the causal GP, interval of the difference above zero.
- **T6.** Two-stage twin over the twin: interval above zero (the second stage gains without fast channels as well).

Reported: every skill against hold, causal GP, offline GP, PCHIP, KFX (on the rows where KFX exists); root-mean-square errors in eV with intervals and the eV differences; the Huber and 1 %-trimmed skills; the thirds by temperature level; the second stage's run time per row.

## 5. What goes into the paper

- T1, T2 and T4 met: the paper's estimator becomes the two-stage estimator (network plus second stage), with the network alone as its first stage; the abstract gives its skill over the causal GP and its eV difference, and states that it also lowers the median error below hold. T5 met: also that it raises the share within 20 %. If T1 or T2 fails, the network stays the estimator and the second stage is reported as not established under the pooled protocol.
- T3 is the paper's increment after this change: met, it is stated as the fast-channel increment of the paper's estimator beside the first-stage increment; not met, the paper says that after a second stage the fast-channel increment is not separated from zero and the increment stays a statement about the first stage only, with this result beside it.
- T6 is stated whatever its outcome (it says whether the second stage's gain needs the fast channels).
- r2 to r4 only: the paper says that these statements rest on three re-partitions, with the primary statements of B.39 on eight unchanged.

## 6. Expectations

T1 +0.05 to +0.10, met. T2 met, about +0.14 to +0.18. T3 +0.03 to +0.07, met but narrower than T1. T4 ratio 0.88 to 0.95, met. T5 +0.005 to +0.02, uncertain. T6 +0.02 to +0.06, met. The session lead's expectations have been wrong several times this week; these are held with that in mind.

## 7. Limits known in advance

Three re-partitions, one initialisation per network. The second stage's hyper-parameters come from B.38, chosen on single splits whose discharges overlap these. The second stage reads the same rows' inputs as the network; it adds no new information, only a different use of it.
