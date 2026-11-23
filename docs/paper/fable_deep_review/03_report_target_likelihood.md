# Dossier: what F/L5 report data can estimate, and what faithful/β‑NLL training adds

## 0. Access and evidence tags

I read three sources in full: Stirn et al., AISTATS 2023 (21 pp); Seitzer et al., ICLR 2022 (24 pp); Bishop, Roach & von Hellermann, PPCF 35 (1993) 765–773.

The only code supplied is the listing in Seitzer App. D.5 (p.24, `beta_nll_loss`, lines 12–17). The repositories mentioned in Seitzer (p.10) and Stirn (p.5 fn.1, p.7 fn.2) were not provided, so I give no path or line citations to them. For Stirn the reference implementation is Eq. (5), p.4.

Tags: **[S]** read directly in the source; **[A]** author claim or result on their data; **[H]** application hypothesis for this study; **[U]** unknown. Nothing below is a new result. Statistical points without a citation are general reasoning, not drawn from these papers.

## 1. What the sources establish

**Bishop 1993 (JET, not KSTAR)**
- [S] A CXRS temperature comes from the width of a line obtained by nonlinear least squares on overlapping lines plus background. The data are photon counts per wavelength bin with photon-statistics errors; χ² uses a per-bin uncertainty ε_k (p.767, Eqs. 1–3). Instrument functions and calibration enter the fit (p.767).
- [S] The fit depends on the initial guess. It can fail to converge, converge to a local minimum, or swap lines (pp.765–768).
- [A] A substantial fraction of logged data commonly went unanalysed because of this (p.766).
- [S] Successive similar spectra may reuse the previous fit as the initial guess (p.772).
- [A] An MLP trained on conventional-fit outputs (≈3000 spectra, p.770) reproduced fitted ion temperatures to about 30% (p.771). It took 15 ms versus 500 ms on the JET mainframe (p.772).
- [A] The authors expect the network to perform well only on data similar to training (p.773). They propose forward reconstruction plus χ² as an output check (p.773).
- Scope: this is prior PPCF art for neural CXRS spectral analysis (spectrum to fit parameters). It says nothing about KSTAR timing, missingness or cross-diagnostic nowcasting.

**Seitzer 2022**
- [S] Assumes Y|X is Gaussian with conditionally independent targets (p.2).
- [S] NLL gradients (Eqs. 3–4, p.3): ∂/∂μ̂ = (μ̂−Y)/σ̂² and ∂/∂σ̂² = (σ̂²−(Y−μ̂)²)/(2σ̂⁴).
- [S] The mean gradient equals an MSE gradient under a reweighted distribution P̃ ∝ P/σ̂² (Eq. 6, p.5).
- [S] β‑NLL (Eq. 7) multiplies the per-sample NLL by stop‑gradient(σ̂^{2β}). The gradients become (μ̂−Y)/σ̂^{2−2β} and (σ̂²−(Y−μ̂)²)/(2σ̂^{4−2β}) (Eqs. 8–9, p.5). The D.5 listing matches: line 12 is the NLL, line 15 multiplies by `variance.detach() ** beta`.
- [S] MSE, NLL and β‑NLL share the same optimum for the mean but allocate capacity differently. Down-weighting is appropriate if the high variance is true noise and harmful if it marks hard-but-important points. The authors call this task-dependent with no one-loss-fits-all answer (pp.5–6).
- [S] The β‑NLL loss value is not an evaluation metric; monitor NLL and RMSE instead (p.5).
- [S] Variance is floored at 1e‑8 and clamped at 1000 (p.22). Without a floor the effective weighting may never stabilise (B.1, p.13).
- [A] β=0.5 is a good starting trade-off (p.9). Student‑t variants generally give better likelihood, β‑NLL better RMSE (Sec. 5.3).

**Stirn 2023**
- [S] Def. 1–2 (pp.2–3): a heteroscedastic network is "faithful" if its expected squared error is no worse than that of its mean-only subnetwork trained separately with the same algorithm.
- [S] Two modifications (p.3): (1) scale the mean gradient by Σ, which cancels the 1/σ² factor; (2) stop the covariance gradient from reaching shared trunk parameters.
- [S] Equivalent single loss, Eq. (5), p.4: ½|y−μ(x)|² − log N(y; ⌊μ(x)⌋, Σ(⌊f_trunk(x)⌋)).
- [S] Theorem 1 (p.4): the trunk and mean-head gradients are exactly the SSE gradients. With identical randomness the updates coincide at every step.
- [S] The variance gradient is the unmodified NLL one, "β=0 for variance"; Proposal 1 equals β=1 for the mean (App. A, p.10).
- [S] With β=1, β‑NLL variance gradients still reach shared parameters (p.4).
- [S] Stated limitation: the faithful mean can never beat its mean-only baseline (p.8).
- [S] Early stopping in their experiments monitored RMSE (C.2, p.15).
- [S] Their noise-variance recovery, Eq. (6), p.7, is Σ_noisy − Σ_clean. It needs two models, one trained on raw replicates and one on replicate means, i.e. "datasets where we have clean and noisy measurements" (p.7; CRISPR replicates, Sec. 3.4).
- [S] Calibration (ECE) is computed through the Normal CDF (p.5).
- [S] Table 1 (p.6) lists LL = −17.4 for the faithful model on "boston". A faithful mean does not guarantee good likelihood.

## 2. What is estimable from F/L5

**Notation.** T_t is the latent carbon temperature on the chord (not bulk-ion truth). F_t is the report. R_t=1 if a report exists. I_t is the strictly past/current input set. S_t=1 if at least 3 of the next 5 reports exist. L5_t is the median of those reports, defined only when S_t=1.

**Estimable (report level, on rows where the label exists):**
- E[F_t | I_t, R_t=1], and conditional quantiles of F.
- The same functionals of L5 given S_t=1.
- Marginal and stratum-wise coverage of predictive intervals for the report.
- Paired loss differences between methods on shared rows.

**Not estimable from these data:**
- T_t itself, bulk Ti, or any bias b_t (calibration, line-swap, instrument function).
- Per-frame fit uncertainty. No ε_k, χ² or fit covariance is available to the study.
- A split of residual variance into plasma variability, photon/fit noise and model misfit.
- Anything on rows with R_t=0 unless missingness assumptions are added.

**Derivation.** Write F = T + b + e. For any predictor m(I):

E[(F−m)² | I, R=1] = Var(T+b+e | I, R=1) + (E[T+b+e | I, R=1] − m)².

Only the sum Var(T|·) + Var(b+e|·) + 2Cov is identified. A learned σ̂²(I) targets this sum plus misfit. That matches Stirn's own statement that standard heteroscedastic variance mixes aleatoric and epistemic parts (p.2).

**Non-identifiability example.** World A: T constant given I, with white report noise of variance v. World B: no report noise, T with an I‑unpredictable frame-to-frame fluctuation of variance v. Both give the same law of F given I. Then:
- L5 behaves identically in both worlds, because a median of later frames attenuates white variation regardless of its origin.
- The F‑vs‑L5 contrast cannot separate them.
- Serially correlated fit error (plausible if fits are warm-started, Bishop p.772 [H]) further mimics slow physics.

**L5 is a different estimand, not a denoised F.** It is a forward-looking temporal median over roughly t+10 to t+50 ms, conditional on S=1. A method that predicts L5 better may be extrapolating trends, shrinking toward a local level, or benefiting from the S=1 case mix.

**Finite exposure.** Each report is already an exposure integral. The mapping from report timestamp to exposure window and availability time is [U]. So "current F" versus "current BES/ECEI snapshot" simultaneity is an assumption on the nominal grid, not an established latency.

**MNAR.** If reports are absent because of weak signal or fit failure, then R depends on unobserved spectrum quality and possibly on T and e. This is [H] for KSTAR, by analogy with Bishop pp.766–768. In that case E[F | I, R=1] ≠ E[F* | I]. No loss function repairs this. Heteroscedastic heads model dispersion among observed reports only.

## 3. Gradient mechanisms as published

**Plain NLL.** Each sample's mean gradient is scaled by 1/σ̂². Once some region is fit and its variance shrinks, its gradients dominate ("rich get richer", Seitzer Sec. 3.2). Points explained by large σ̂ lose mean-learning signal.

**β‑NLL.** One joint loss; all parameters, including shared ones, receive both gradients.
- Mean weight: σ̂^{−(2−2β)}. This is NLL at β=0, 1/σ̂ at β=0.5, and uniform (MSE) at β=1.
- Variance gradient: (σ̂²−r²)/(2σ̂^{4−2β}). Its zero is still σ̂² = E[r²|x] for every β, since the weight is positive and detached. Its conditioning changes with β (Stirn Fig. 6, p.10).
- For β<1 the finite-capacity mean is a precision-weighted fit. The population optimum is the same, but the solution differs (Seitzer p.6).
- For β=1 with a shared trunk, variance gradients still reshape the trunk, so the mean trajectory is not the MSE trajectory (Stirn p.4).

**Faithful.** One joint loss, optimised concurrently; nothing is frozen.
- The trunk and mean head follow the exact MSE trajectory.
- The variance head simultaneously receives the unmodified NLL variance gradient (σ²−r²)/(2σ⁴)·∂σ²/∂θ_Σ. It is evaluated on the current, still-moving mean residual and on detached trunk features.
- So faithful = "β=1 for the mean, β=0 for the variance, plus trunk shielding".
- The paper's sequential alternative (fit mean, freeze, then fit a separate variance network) is described as also faithful but deficient (p.3). It is a different algorithm. If a post-hoc head on a frozen trunk is ever used, it must be labelled as that variant, not as Eq. (5).

**Implications for this study.**
- [S→H] By Theorem 1 a faithful model's F/L5 point metrics equal those of its MSE twin. It cannot yield a mean gain, a fast-diagnostic increment, or a change in any hold/GP/interpolation comparison. Its only possible contribution is a report-level dispersion estimate.
- [H, my derivation] The identity needs per-parameter optimisers and no coupling between heads. Global-norm gradient clipping, a shared scheduler keyed to total loss, or early stopping on NLL would break it. Stirn stops on RMSE (p.15).
- [H] The detached trunk carries only mean-relevant features. If report dispersion depends on inputs the mean ignores (e.g. signal level), a separate variance network is the admissible case; Stirn p.3 allows f_trunk = identity.
- [H] At an RMSE-chosen stopping epoch the variance head may be immature.
- Both papers assume independent Gaussian targets. Serially correlated 10 ms reports violate this. Likelihoods and ECE are then descriptive scores, not calibrated probabilities.

**Relation to the robust-loss screens already run.** Which losses were screened is not stated to me [U].
- Huber-type losses weight by residual magnitude, not by input. They change the population target under asymmetric noise.
- NLL and β<1 weight by a learned, input-dependent σ̂(x), detached from the sample's own residual. They keep the conditional mean as the population optimum.
- Student‑t variants (Stirn B.3; Seitzer baselines) depend on both.
- Faithful has no robustness in the mean.
- Admissible reading [H]: a β‑NLL mean gain where residual-based robust losses gave none is compatible with input-predictable heteroscedasticity rather than sporadic outliers. It is equally compatible with capacity moving away from hard transients. Seitzer found the NLL-ignored points to be the most important ones in one task (p.6).
- Neither reading is identifiable from aggregate RMSE. β is a modelling choice (Seitzer p.6), not something the data select without external noise information.

**Extra noise labels.**
- Not required: to train either loss, or to assess report-level coverage on held-out reports.
- Required to call σ̂² measurement or fit noise: Stirn's Eq. (6) needs clean-target and noisy-target versions of the same conditional quantity (replicates at fixed x) and assumes the epistemic part cancels between two trained models. L5 is not such a clean target. It averages different times, on a different eligibility population. Σ̂_F − Σ̂_L5 must not be presented as noise variance.
- Required for fit uncertainty: per-frame χ², ε_k or covariance, or refits from perturbed initial guesses.
- Required for latent T: an independent paired measurement (e.g. UFCES; not established).

## 4. One low-cost report-level hypothesis (to register before any fit or readout)

**H‑R1 (TEST-informed exploratory; fixed FOUR‑3seed subset, reported separately from FULL15 one-seed).**
- Keep the already selected mean architecture and its MSE twin unchanged.
- Train the Eq. (5) faithful head jointly, as published, with the twin's seed and stopping rule.

**Gate.** Point predictions must match the twin to numerical tolerance. A mismatch is an implementation fault and voids the readout. No statistical test is needed, because Theorem 1 is an identity.

**Claim.** On common-support rows, pre-specified interval scores for F (fixed quantile pairs, e.g. 10/90 and 25/75) are lower for the faithful σ̂(I) than for:
- B0: a constant TRAIN-residual variance;
- B1: TRAIN-residual variance tabulated by anchor age × flag only.

The primary contrast is faithful versus B1. Staleness alone is the obvious smoothing-type explanation of "calibration", so B1 is the baseline that matters.

**Readouts.**
- Paired shot-level mean differences, with all per-shot values shown.
- Stratum-wise empirical coverage by age, flag, and fast-diagnostic availability.
- The same quantities for L5 on joint F/L5-eligible rows, shown side by side with no differencing interpretation.
- Distribution-free coverage rather than the Normal-CDF ECE.

**Decision rule.** Gain is declared only if the paired difference exceeds a pre-registered margin. Otherwise the result is "inconclusive", not "equivalent".

**Interpretation.** At most: fast or raw inputs predict report dispersion beyond staleness in this campaign. It says nothing about noise, fit error or T.

**Cost.** One training run per seed. The inference increment is one head. Fit within the 10 ms budget is unmeasured, and Bishop's 15 ms is not evidence for it.

**Why β‑NLL is not the vehicle.** It changes the mean. That would re-open every existing mean comparison and confound the hold/GP/interpolation framing. Both losses are published; using them is not algorithmic novelty.

## 5. One informative measurement audit: CES frame provenance and fit quality

Registered and done with the diagnostic owners. I ran no fits; this is design only. Draw a pre-specified stratified sample of shots and build one per-nominal-slot table:

1. **Timing.** Exposure start/stop, integration time, what the report timestamp denotes, and when the fitted value existed. This decides whether "strictly past" anchors and "current" snapshots are causal as assumed, and what source-age means physically.
2. **Missingness cause** for every empty slot. Categories: beam off or modulated (cross-checked against NBI source power/voltage, which is available), saturation or low signal, fit non-convergence, QC rejection, never analysed. Also record the eligibility rule that generates S.
3. **Fit quality** where a report exists. χ², parameter covariance, convergence flag, initial-guess source (warm start from previous frame or not), and a Bishop-style forward-reconstruction residual (p.773). If feasible, refits from perturbed initial guesses give a direct empirical fitting spread.

**Why it is informative in either direction.**
- If missingness is dominated by beam timing, it is largely a function of observables, and selection-population results become more defensible.
- If it is dominated by fit failure or QC tied to spectrum quality, MNAR is documented and bounds or sensitivity analyses become mandatory.
- If σ_fit exists, its association with the H‑R1 σ̂ can be reported. An association supports the interpretation but does not prove it.
- Warm-start usage tells whether serially correlated fit error is a live explanation for F-only gains.

**What it still cannot give.** Carbon-versus-bulk equivalence, absolute calibration, local heating, gradients.

## 6. Estimands, pairing, margins, small clusters

**Selection-population estimand.** Each method is scored on rows where it and the label are defined. This is deployment-like for that method, but case mix differs between methods. Two-sided PCHIP+GP needs future anchors, conditional-availability models need fast channels, and L5 needs S=1. Cross-method differences then confound skill with eligibility.

**Common-support estimand.** Score on the intersection of rows where all compared methods and the label exist. This gives a valid paired contrast, but for a selected and possibly easier population.

**Recommendation.** Report both, together with eligibility rates and covariate summaries of excluded rows (age, flag, NBI state). Never compare an F metric and an L5 metric computed on different row sets. Under MNAR neither estimand equals performance over all time points, and that should be stated.

**Pairing.** Compute per-row loss differences on identical rows, aggregate to the shot, and run inference on shot-level means. Separate marginal CIs, or their overlap, do not test a difference.

**What the papers do and do not license.** Stirn's paired t‑test on squared errors (p.5) and both papers' KS and G tests (Stirn p.5; Seitzer p.21) treat held-out points or splits as independent. They neither address clustered time series nor validate any bootstrap.

**Small-cluster assumptions to state explicitly:**
- Shots are treated as independent draws. This is questionable within one reused campaign: session and conditioning drift, shared calibration.
- The number of clusters is small and their lengths unequal, so asymptotic cluster-bootstrap or sandwich behaviour is not guaranteed.
- A few shots can dominate.
- Sign-flip or permutation tests need symmetry or exchangeability of shot differences.
- Frozen-fit intervals omit fit, selection and multiplicity uncertainty.
- Seed spread on FOUR does not transfer to FULL15.
- Showing every per-shot difference is the most honest display.

**Equivalence and noninferiority.**
- These require a margin Δ fixed in advance in report units and justified externally. Examples: a stated operational tolerance, or a fraction of the registered hold-versus-GP gap.
- Δ cannot be derived from "noise level", which is unidentified.
- Bishop's ~30% figure is surrogate-versus-fit agreement on JET and is not a margin.
- Noninferiority: the upper bound of the paired difference is below Δ. Equivalence: both bounds lie within ±Δ. Otherwise the result is inconclusive.
- Stirn's "never empirically unfaithful" (p.6) is non-rejection in a one-sided test. Their noninferiority rests on the Theorem 1 identity, not on that test.

## 7. Apparent gains explainable by smoothing or selection

| Apparent result | Smoothing/selection explanation | Evidence that discriminates (does not identify) |
|---|---|---|
| Gain on L5, not on F | Target is a forward median; any level-shrinking or trend-following predictor benefits; S=1 rows are steadier | Same rows for both targets; a causal past-median baseline with matched window; registered re-scoring against other offline window widths (dose-response); results on S=0 rows using F |
| Gain on F, not on L5 | Serially correlated fit error (warm starts) or exposure overlap tracked by inputs; or the median removes a fast component | Audit items 1 and 3; the existing same-learner future-information latency curve as a reference shape |
| Fast-diagnostic increment only where channels are available | Availability selects shots or phases | Common-support versus selection-population tables; the existing NBI, commonality, permutation and Mirnov controls |
| RMSE gain from NLL/β<1 | Capacity shifted to low-σ̂ rows, an implicit row selection | Per-stratum errors and log-scale residual histograms (as in Seitzer Fig. 6, B.1); distribution of effective weights σ̂^{−(2−2β)} across transient versus steady strata (as in Seitzer Fig. 5/S4) |
| "Calibrated" σ̂ | σ̂ merely encodes anchor age | The B1 baseline in H‑R1 |
| Better common-support score | Intersection removes hard rows | Eligibility rates; covariate comparison of excluded rows |

**Retained non-identifiability.** Even with every row of this table satisfied, the report data admit multiple triples (latent dynamics, report error process, selection mechanism) with the same observable law (Section 2). An F-only or L5-only gain, or a difference between them, identifies neither physics nor noise. Only the audit's external measurements, or paired independent diagnostics that are not available, can narrow the set. Until then every claim should read "predicts the CES report (or its forward median) on the stated row population".
