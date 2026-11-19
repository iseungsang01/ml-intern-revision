# B.72 — the D1 result under the validation conventions of other fields

Registered 2026-11-20, before any run of this batch. CPU only: the stored D1 estimates and errors of B.70 are re-read; nothing is trained and no network is run. The session lead designed the batch. An AI agent implements and runs it under the author's supervision. No existing file is edited.

## 1. Why

The author asked why the paper validates the way it does, and whether a method used elsewhere would be better. The paper's method (D1, `PREREGISTRATION_B70.md` section 2):

- the discharge is the split unit; five-fold cross-validation, three random partitions (r2–r4);
- the score is the pooled MSE and the MSE skill against a reference, on identical rows;
- the 95 % interval is a two-level bootstrap (a partition drawn, then its test discharges resampled);
- a difference is stated when the paired interval excludes zero.

Other conventions answer questions this method leaves open:

| Convention | Where it is used | Question it answers |
|---|---|---|
| R², relative error, correlation on held-out discharges | ML papers on fusion diagnostics | Can a reader compare our numbers with theirs? |
| Skill decomposition (Murphy 1988) | forecast verification | Is the skill tracking, or a bias? |
| One score per discharge | applied ML, biostatistics | Is the gain in the typical discharge, or carried by a few? |
| Paired test on loss differences (Diebold–Mariano, cluster-robust) | forecast comparison | Does an analytic test agree with the bootstrap? |
| Corrected resampled t-test over folds (Nadeau–Bengio 2003) | comparing learned models | Is the interval too narrow because the folds share training data? |
| Model confidence set (Hansen, Lunde, Nason 2011) | forecast comparison | Among all estimators at once, which cannot be told from the best? |
| Holm correction | any set of tests | Do the statements survive a correction for their number? |

A survey of how the comparable papers validate is written beside this batch (`docs/paper/validation_methods.md`). It does not change this registration.

## 2. Data

- Rows: the D1 rows of B.70 (222,672 per re-partition, 635 discharges).
- Estimates with sign: `data/.b70/stage2_r{r}_f{k}.npz` — measured value `y`, network `est_c`, two-stage estimator `est_c2`, twin `est_t`, two-stage twin `est_t2`.
- Squared errors of the references: `data/.b70_rescore/b39p_v2_cut_r{r}_f{k}/comparison_errors_test.npz` — hold, causal GP, offline GP, PCHIP.
- Filters KF0 and KFX: included if B.70 phase 2 stored their errors on the D1 rows; otherwise left out, and the readout says so.
- T_i only.

**Per-discharge quantities.** For estimator m and discharge d: S(m, d) = the sum of squared errors over the discharge's D1 rows, averaged over the three re-partitions; n(d) = its number of D1 rows. Every discharge is a test discharge once per re-partition, on the same rows.

## 3. Readouts

**R1. Field metrics** (eight estimators, reported):
- R² = 1 − SSE / SST, with SST about the mean of the scored values of the re-partition; point = mean over r2–r4; interval by the discharge bootstrap of R3 (b);
- Pearson correlation of estimate and measured value (the four estimators with stored estimates);
- mean and median of |error| / measured value;
- Murphy's decomposition for the four estimators with stored estimates: r², the conditional-bias term (r − s_f / s_o)², the mean-bias term ((mean f − mean o) / s_o)², and the slope of measured on estimated.

**R2. One score per discharge** (discharges with at least 20 D1 rows; the number left out is reported):
- ρ(d) = 1 − S(A, d) / S(B, d);
- the median of ρ(d) and the share of discharges with ρ(d) > 0, each with a 95 % interval (discharges resampled, 20,000 draws);
- the mean over discharges of the per-discharge MSE (equal weight per discharge) and its skill.

**R3. Other inference for the pooled skill.** Pairs: the forms of D1-1 to D1-4, the two-stage estimator over the network, and the two-stage estimator over the offline GP.
- (a) the two-level bootstrap of B.70, reproduced;
- (b) single-level bootstrap: discharges resampled on S(m, d), 20,000 draws;
- (c) cluster-robust paired test: D(d) = S(B, d) − S(A, d); t = mean D / (sd D / √n), two-sided p from Student's t with n − 1 degrees of freedom; skill interval by linearising the ratio;
- (d) corrected resampled t-test: the skill of each of the 15 folds; variance of the mean = (1/15 + 1/4) × the sample variance of the fold skills; 14 degrees of freedom;
- (e) Wilcoxon signed-rank test on D(d) (`scipy.stats.wilcoxon`, two-sided).

Reported per pair: point, interval or p-value, width of the interval relative to (a), verdict at 5 %.

**R4. Model confidence set**, level 95 %, on the matrix S(m, d) with one row per discharge.
- `arch.bootstrap.MCS`, method `R`, 20,000 replications, block length 1 (discharges resampled independently), seed 20261172. If the package cannot be installed, the procedure of Hansen, Lunde and Nason (2011, section 3) is implemented with the same settings and the readout says so.
- Set C (causal): two-stage estimator, network, twin, two-stage twin, causal GP, hold, and the filters if available.
- Set A (all): set C with the offline GP and PCHIP.
- Reported: the members, and the p-value at which each estimator leaves.
- Beside it, the same with the per-discharge MSE S(m, d) / n(d) as the loss.

**R5. Number of statements.** Two-sided bootstrap p-values of D1-1 to D1-5 and the two forms of D1-6 (p = 2 × the smaller tail share of the draws about the null value, not below 1 / 20,001), with Holm's correction over the seven.

## 4. Statements

| Id | Statement | Expectation |
|---|---|---|
| V1 | Each of D1-1 to D1-4 keeps its verdict under (b), (c), (d) and (e) | holds for D1-1, D1-2, D1-4; for D1-3 holds under (b) and (c), uncertain under (d) and (e) |
| V2 | The 95 % model confidence set of set C is the two-stage estimator alone | holds |
| V3 | After Holm's correction every one of the seven remains below 0.05 | holds |
| V4 | The median per-discharge MSE reduction of the two-stage estimator lies above zero (interval), over the causal GP and over the two-stage twin | holds over the causal GP; weak expectation over the twin |
| V5 | No interval of (b), (c) or (d) is more than 1.25 times as wide as (a) for D1-1 to D1-4 | holds for (b) and (c); uncertain for (d) |

Each statement is judged separately.

## 5. What goes into the paper

- R² and the relative error are added to the main table so that the numbers can be set beside those of other papers. The sentence that R² is high for hold as well goes with it, with the value.
- If V5 fails for a pair, the paper's interval for that pair becomes the widest of (a)–(d), and `PAPER_DESIGN.md` 7 is changed.
- If V1 fails for a statement, the statement is reported with each method's verdict in one table, and the text says which methods disagree.
- If V2 or V3 fails, the text names the estimators or statements that are not separated.
- If all hold, the primary readout stays, and one supplement table gives the result under the other conventions.

## 6. Limits

- Every readout here re-reads the same 15 networks on the same random discharge-level partitions. None of them tests whether a random partition lets discharges of one session sit on both sides of the split. That needs training on chronological or session-blocked partitions: B.69 (the later discharges; GPU, stopped at 25 of 40 runs by the author's instruction), and a five-fold partition by shot order as a candidate batch.
- One initialisation per fold: no readout here separates the variance of initialisation from that of the training set.
- The corrected t-test (d) uses 15 fold values and a correction factor derived for random subsampling, not for repeated k-fold.

## 7. Checks and files

- The points of (a) reproduce `docs/paper/b70_d1.json` within 0.001.
- The rows and measured values of the two sources agree for every fold.
- Code: `ces_prediction/experiments/b72_methods/`. Output: `docs/paper/b72_methods.{md,json}`.
- Seed 20261172, one stream per quantity (crc32 of its name).
