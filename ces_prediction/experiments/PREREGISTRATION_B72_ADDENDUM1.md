# B.72, addendum 1 — implementation choices the registration leaves open

Written 2026-11-20, before any readout of B.72: no skill, interval, p-value, R², per-discharge score or model confidence set of this batch has been computed. Facts and choices only; the registration is unchanged. Facts checked before writing: the D1 row sets (discharge, grid row) of r2, r3 and r4 are identical (222,672 rows, 635 discharges); 33 of the 635 discharges have fewer than 20 D1 rows (smallest: 1 row); the B.70 phase 2 filter files `data/.b70p2/b56/r{r}_f{k}.npz` carry `shot`, `row` and `y` equal to those of `data/.b70/stage2_r{r}_f{k}.npz` on fold r2 f0, every row located (`gi` ≥ 0) and every estimate finite (the code checks every fold).

## E1. Arrays

- Squared errors: the arrays of `analyze_b70.load_all` (imported, not copied): network and twin = `CES_TI_se_model` of the D1 re-scorings; hold, causal GP, offline GP, PCHIP = `CES_TI_se_persistence`, `CES_TI_se_gp_causal`, `CES_TI_se_gp`, `CES_TI_se_pchip`; two-stage estimator and two-stage twin = (`est_c2` − y)², (`est_t2` − y)² of the stage-2 files (B.64 addendum-1 rule already applied: a row whose second-stage inputs are missing keeps the first stage).
- Estimates with sign: `est_c`, `est_c2`, `est_t`, `est_t2` of the stage-2 files. Check: (estimate − y)² equals the squared error used for the statements (largest absolute difference reported).
- Filters KF0 and KFX: `est_kf0` and `est_kfx` (clip 0.2) of `data/.b70p2/b56/r{r}_f{k}.npz`. Included only if, on every fold, `shot` and `row` equal those of the stage-2 file, `y` agrees within 1e-6 eV, every row is located and every estimate is finite; otherwise both filters are left out everywhere and the readout says so. Where they are included: R1 (all metrics; Murphy's terms too, since their signed estimates are stored), R2 (one pair, two-stage estimator over KFX), R4 (sets C and A). They enter no statement of R3 or R5 (not registered pairs).

## E2. Per-discharge quantities

- S(m, d) = sum of squared errors over the D1 rows of discharge d in re-partition r, then the mean over r2–r4 (each discharge is a test discharge once per re-partition, on the same rows). n(d) = its number of D1 rows (the same in every re-partition; checked).
- Discharge = the grid file (`fidx`). The four discharges that are never scored (no D1 row) are absent from every readout.

## E3. R1

- R² of re-partition r: 1 − SSE_r / SST, SST = Σ (y − ȳ)² over the 222,672 rows (the same rows in every re-partition, so SST does not depend on r). Point = mean over r2–r4 of the three R².
- Interval of R²: the discharge draws of R3 (b) (the same weights), R² of a draw = 1 − Σ w_d S(m, d) / [Σ w_d Syy_d − (Σ w_d Sy_d)² / Σ w_d n(d)], i.e. SST about the mean of the drawn rows.
- Pearson correlation of estimate and measured value, mean and median of |error| / measured value: computed over the rows of each re-partition, then the mean over r2–r4. Correlation only where signed estimates are stored (two-stage estimator, network, twin, two-stage twin, KF0, KFX); for hold, causal GP, offline GP and PCHIP only squared errors are stored, so their correlation and Murphy terms are not reported.
- Murphy's decomposition per re-partition with moments divided by N: r² ; conditional-bias term (r − s_f / s_o)² ; mean-bias term ((f̄ − ō) / s_o)² ; slope of measured on estimated = r s_o / s_f. Then the mean over r2–r4. Check: R² = r² − conditional bias − mean bias in each re-partition (|difference| below 1e-9).

## E4. R2

- Discharges with n(d) ≥ 20 (602 of 635); the 33 left out are reported. A discharge with S(B, d) = 0 would be left out of that pair and counted (none expected).
- Pairs (A over B): two-stage estimator over causal GP, over two-stage twin, over network, over offline GP, over hold, over KFX; network over twin, over causal GP.
- ρ(d) = 1 − S(A, d) / S(B, d). Point of the median: `numpy.median` over the eligible discharges. Share: fraction with ρ(d) > 0.
- Draws: 20,000; the eligible discharges resampled with replacement (multinomial counts); in a draw the median is the weighted quantile at 0.5 (smallest ρ whose cumulative count reaches half the total, the rule of `run_b41.wquant`), the share is the count-weighted share. 95 % interval = 2.5 and 97.5 percentiles.
- Equal-weight MSE: M(m) = mean over eligible discharges of S(m, d) / n(d); skill = 1 − M(A) / M(B); interval on the same draws.

## E5. R3 (pairs: D1-1 two-stage estimator over causal GP, D1-2 over two-stage twin, D1-3 network over twin, D1-4 network over causal GP, two-stage estimator over network, two-stage estimator over offline GP)

- (a) `stage2_b64.cell` on all D1 rows with `analyze_b70.rng_for("all")` (B.70's seed 20261170 and stream): the draws of B.70 are reproduced draw for draw, not drawn anew with the B.72 seed. Point = mean over r2–r4 of the per-re-partition skill.
- (b) All 635 discharges resampled with replacement, 20,000 draws (stream "R3b"; R1's R² interval uses the same draws), skill of a draw = 1 − Σ w_d S(A, d) / Σ w_d S(B, d). Point = 1 − Σ_d S(A, d) / Σ_d S(B, d) (ratio of the three-re-partition mean sums; it can differ from (a)'s mean of ratios in the fourth decimal).
- (c) D(d) = S(B, d) − S(A, d) over the 635 discharges; t = mean D / (sd D / √n), sd with n − 1; two-sided p from Student's t, n − 1 degrees of freedom. Skill ŝ = Σ D / Σ S(B, ·); linearised standard error SE = sd(D(d) − ŝ S(B, d)) / (√n × mean S(B, ·)) (first-order expansion of the ratio of means); interval ŝ ± t_{0.975, n−1} × SE.
- (d) The skill 1 − SSE_A / SSE_B on the test rows of each of the 15 folds (r2–r4 × 5); the three re-partitions enter as 15 values with no re-partition level. Mean m̄; variance of the mean = (1/15 + 1/4) × s² (s² with 14 in the denominator; 1/4 = test / training discharges of a five-fold partition, as registered); interval m̄ ± t_{0.975, 14} × √variance; p from t = m̄ / √variance, 14 degrees of freedom.
- (e) `scipy.stats.wilcoxon(D, alternative="two-sided")` (zero differences dropped, scipy's default method) on the 635 D(d). Direction = sign of T+ − T− (rank sums of positive and negative D).
- Verdict at 5 %: (a)–(d) "holds" when the lower bound of the interval is above zero; (e) "holds" when p < 0.05 and T+ > T−. Width relative to (a) = (upper − lower) / (upper_a − lower_a). A statement "keeps its verdict" under a method when that method's verdict is "holds".

## E6. R4

- Loss matrix: one row per discharge (635), one column per estimator; loss 1 = S(m, d); loss 2 = S(m, d) / n(d).
- Set C: two-stage estimator, network, twin, two-stage twin, causal GP, hold, KF0, KFX. Set A: set C with offline GP and PCHIP. Four runs: two sets × two losses.
- `py -m pip install arch` is tried once (after a `--dry-run` showed it adds arch 9.0.0, statsmodels 0.15.0, patsy, formulaic, interface_meta, wrapt and upgrades nothing installed). If `arch` imports: `arch.bootstrap.MCS(losses, size=0.05, reps=20000, block_size=1, method="R", bootstrap="stationary", seed=Generator)` — with block size 1 the stationary bootstrap starts a new block at every position, so discharges are resampled independently. Each of the four runs gets its own stream `default_rng([20261172, crc32(name)])`. Members = `included`; the p-value at which an estimator leaves = its MCS p-value (`pvalues`).
- If `arch` does not install or import: Hansen, Lunde and Nason (2011) section 3 by hand with the same settings: the range statistic T_R = max_{i,j} |d̄_ij| / √var̂(d̄_ij), the variance from the same bootstrap draws (centred), elimination of argmax_i max_j d̄_ij / √var̂(d̄_ij), MCS p-value = running maximum of the elimination p-values. The readout names which implementation was used.

## E7. R5

- Draws: those of B.70, reproduced. D1-1 to D1-4: the draws of (a). D1-5: `stage2_b64.weights` with `analyze_b70.rng_for("medians")` and 20,000 draws (as B.70 called `stage2_b64.medians`), weighted medians by `run_b41.wquant` for the two-stage estimator and hold only. D1-6: the good-and-bad mixture flags recomputed with `analyze_b70.mixture` (deterministic EM on the same grids; `stage2_b64.ctx()` loads the cut grid), then `stage2_b64.cell` on the unflagged rows with `analyze_b70.rng_for("D1-6")`.
- Null value: 0 for the skills, 1 for the median ratio. p = 2 × min(#draws ≤ null, #draws ≥ null) / 20,000, raised to 1 / 20,001 if smaller, capped at 1.
- Holm over the seven: sort ascending, adjusted p_(i) = max over j ≤ i of min(1, (8 − j) p_(j)).
- Check: the D1-5 interval and the two D1-6 intervals reproduce `docs/paper/b70_d1.json` (points within 0.001).

## E8. Statements

- V1: each of D1-1 to D1-4 holds under each of (b), (c), (d), (e) (E5 verdict rule); judged per statement and reported per method; V1 holds when all sixteen hold.
- V2: the 95 % set of set C under loss 1 (the registered loss) is the two-stage estimator alone; loss 2 reported beside.
- V3: every Holm-adjusted p below 0.05.
- V4: the lower bound of the 95 % interval of the median ρ(d) above zero, separately for the two-stage estimator over the causal GP and over the two-stage twin; V4 holds when both do.
- V5: width ratio ≤ 1.25 for (b), (c), (d) on D1-1 to D1-4 (twelve ratios).

## E9. Run

- CPU, numpy / BLAS / torch threads 2 (`analyze_b70` sets the BLAS variables to 4 on import; they are set back to 2 before torch is imported). Nothing is trained; no network is run; the only fits are the deterministic EM of the mixture, recomputed to recover B.70's D1-6 rows.
- Code: `ces_prediction/experiments/b72_methods/run_b72.py`. Output: `docs/paper/b72_methods.{md,json}`.
