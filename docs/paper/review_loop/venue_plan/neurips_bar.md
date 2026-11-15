# What a NeurIPS main-conference paper would take (2026-11-15)

Scope: main track, Evaluations & Datasets track (the former Datasets & Benchmarks track), or the
Position Paper track. Workshops are out of scope. Inputs read: `round_1.md`, `round_2.md`,
`docs/paper/short/main_short.tex`, `THESIS_RESULTS.md` section 0, `PROJECT_KNOWLEDGE.md` headings.
No experiment was run. Web sources were searched on 2026-11-15; items marked *(from memory)* were not
re-fetched today and should be checked before they are cited in a paper.

## 0. Short answer

The current manuscript cannot be submitted to any NeurIPS proceedings track as it stands: one private
dataset, one device, no new method, and a headline that an offline GP matches (T_i vs offline GP
-0.010 [-0.043, +0.014], interval includes 0). The asset that travels is not the model. It is the
audit: a registered positive result that turned out to be carried by 0.05 % of rows in the baseline,
an aggregate skill whose median row is worse than persistence, and a one-parameter smoother that
takes about half the margin over interpolation. The only design with a realistic path is an
evaluation paper that shows these three effects on public data with published imputers, submitted
to the Evaluations & Datasets track. Estimated acceptance if the effects replicate: about 12-18 %.
If they do not replicate outside KSTAR, every design here is below 5 %.

## 1. Track structure and dates (verified)

| Item | Fact | Source |
|---|---|---|
| NeurIPS 2026 main track | abstract 4 May 2026, paper 6 May 2026 (AoE), decisions 24 Sep 2026; 9 pages, double-blind | [CFP 2026](https://neurips.cc/Conferences/2026/CallForPapers) |
| Evaluations & Datasets (E&D) track | renamed from Datasets & Benchmarks in 2026. In scope by name: audits and stress-tests of prior evaluations, analyses of how evaluation assumptions change conclusions, negative results, new protocols. No new model or state of the art required. Double-blind by default. Datasets must be hosted and reachable by reviewers at submission, with Croissant metadata; code is mandatory only for reusable artifacts. Same dates as main track. | [E&D call 2026](https://neurips.cc/Conferences/2026/CallForEvaluationsDatasets) |
| Position Paper track | argues what "should be done"; title and bold sentence must state the position; 9 pages. **The paper must be substantially human-written, AI limited to copy-editing, AI use disclosed** (stricter than the main track). Same dates. | [Position call 2026](https://neurips.cc/Conferences/2026/CallForPositionPapers) |
| Base rates | 2025 main track 5,290 / 21,575 = 24.5 %. D&B 2025: 1,995 submissions, rate not disclosed (2024: 25.3 %). Position track rate not found. | [PC chairs' reflections 2025](https://blog.neurips.cc/2025/09/30/reflections-on-the-2025-review-process-from-the-program-committee-chairs/), [D&B chairs 2025](https://blog.neurips.cc/2025/09/30/reflecting-on-the-2025-review-process-from-the-datasets-and-benchmarks-chairs/) |
| NeurIPS 2027 | not officially announced. Third-party trackers disagree (paper deadline 16 or 21 May 2027). Plan for early-to-mid May 2027, about 24 weeks from now. | unofficial; confirm on neurips.cc |

## 2. What accepted work in adjacent areas looks like

**Evaluation-critique papers (the closest genre to our assets).** The pattern is a simple, general
claim demonstrated at scale on public data.

- Zeng, Chen, Zhang, Xu. "Are Transformers Effective for Time Series Forecasting?" AAAI 2023.
  One-layer linear models against transformer forecasters on nine public datasets.
  https://arxiv.org/abs/2205.13504 *(from memory)*
- Brigato, Morand, Stromsdorfer, et al. "Position: There are no Champions in Long-Term Time Series
  Forecasting." arXiv 2502.14045 (2025). More than 3,500 networks on 14 datasets; small changes in
  setup reorder the leaderboard. Venue not confirmed. https://arxiv.org/abs/2502.14045
- Moretti, Marisca, Alippi, Cini. "Position: Current Benchmarking Hinders Real Progress in Deep
  Learning for Time Series Forecasting." ICML 2026 position track.
  https://icml.cc/virtual/2026/poster/67106 , https://arxiv.org/abs/2512.22702
- Klotergens, Yalavarthi, Stubbemann, Schmidt-Thieme. "Physiome-ODE: A Benchmark for Irregularly
  Sampled Multivariate Time-Series Forecasting Based on Biological ODEs." ICLR 2025. A
  constant-value forecaster is ahead of ODE-based models on three of the four standard irregular
  datasets; the paper then supplies 50 generated datasets. https://arxiv.org/abs/2502.07489
- Liu, Paparrizos. "The Elephant in the Room: Towards A Reliable Time-Series Anomaly Detection
  Benchmark." NeurIPS 2024 D&B. *(from memory)*
  Sarfraz et al. "Position: Quo Vadis, Unsupervised Time Series Anomaly Detection?" ICML 2024.
  https://arxiv.org/abs/2405.02678 *(from memory)*
- Agarwal, Schwarzer, Castro, Courville, Bellemare. "Deep Reinforcement Learning at the Edge of the
  Statistical Precipice." NeurIPS 2021 (outstanding paper). Interval estimates and robust aggregates
  in place of point scores. https://arxiv.org/abs/2108.13264 *(from memory)*
- McGreivy, Hakim. "Weak baselines and reporting biases lead to overoptimism in machine learning for
  fluid-related partial differential equations." Nature Machine Intelligence 6 (2024).
  https://arxiv.org/abs/2407.07218 *(from memory)*
- Kapoor, Narayanan. "Leakage and the reproducibility crisis in machine-learning-based science."
  Patterns 2023. https://arxiv.org/abs/2207.07048 *(from memory)* (covers our finding (d).)
- Hewamalage, Ackermann, Bergmeir. "Forecast evaluation for data scientists: common pitfalls and
  best practices." Data Min. Knowl. Disc. 2023. https://arxiv.org/abs/2203.10716

**Imputation and irregular series.**

- Du et al. "TSI-Bench: Benchmarking Time Series Imputation." arXiv 2406.12747. 28 imputers on 8
  datasets, with Mean / Median / LOCF / Linear as naive arms. https://arxiv.org/abs/2406.12747
- Dolatpour Fathkouhi, Namazi, Shakeri. "The Stationarity Bias: Stratified Stress-Testing for
  Time-Series Imputation in Regulated Dynamical Systems." arXiv 2602.15637 (Feb 2026, preprint).
  Linear interpolation leads on stable glucose intervals, deep models on transients; aggregate
  MSE/MAE hides the split. **This is the nearest existing statement of our finding (b).**
  https://arxiv.org/abs/2602.15637
- Li et al. "HyperIMTS: Hypergraph Neural Network for Irregular Multivariate Time Series
  Forecasting." ICML 2025. https://proceedings.mlr.press/v267/li25bl.html (typical methods paper:
  new architecture, standard public irregular datasets, cost table.)
- Che et al. "Recurrent Neural Networks for Multivariate Time Series with Missing Values" (GRU-D),
  Sci. Rep. 2018. *(from memory)* Our mask handling is this lineage, so it is not a contribution.

**Label noise and achievable-performance ceilings.**

- Ishida, Yamane, Charoenphakdee, Niu, Sugiyama. "Is the Performance of My Deep Network Too Good to
  Be True? A Direct Approach to Estimating the Bayes Error in Binary Classification." ICLR 2023.
  https://arxiv.org/abs/2202.00395 *(from memory)*
- Northcutt, Athalye, Mueller. "Pervasive Label Errors in Test Sets Destabilize Machine Learning
  Benchmarks." NeurIPS 2021 D&B. https://arxiv.org/abs/2103.14749 *(from memory)*
- Crusius et al. Faraday Discussions 256 (2025): realistic performance bounds from experimental
  error on nine drug, materials and molecular regression datasets; four are already at the bound.
  https://pubs.rsc.org/doi/d4fd00091a (title not re-read today).
- ConFrag (noisy-label regression, NeurIPS 2024) and "Bias-Corrected Ceilings of Emotion
  Predictability from Human Label Variation" (arXiv 2608.15619, preprint) surfaced in search; not
  read in full.
- Broderick, Giordano, Meager. "An Automatic Finite-Sample Robustness Metric: When Can Dropping a
  Little Data Make a Big Difference?" arXiv 2011.14999. *(from memory)* **This is the nearest
  existing statement of our finding (c)**: a conclusion that reverses when a fraction of a percent
  of the sample is dropped.

**Fusion at ML venues.** Rare in main tracks, and where present it comes with open data or a
control result.

- Rousseau, Jackson, Ordonez-Hurtado, et al. (16 authors; IBM Research Europe, UKAEA, STFC).
  "TokaMark: A Comprehensive Benchmark for MAST Tokamak Plasma Models." arXiv 2602.10132; the
  dataset card cites KDD 2026 Datasets and Benchmark Track. 11,573 MAST shots, 39 signals, 14
  tasks, NRMSE, "no resampling or imputation is performed", masked metrics. Task 3-3 reconstructs
  Thomson scattering from magnetics and interferometry. **No charge-exchange signal is among the 39.**
  https://arxiv.org/abs/2602.10132 , https://huggingface.co/datasets/sveln/tokamark-dataset
  (CC BY-SA 4.0, 388 GB zipped, about 2 TB unzipped)
- "TokaMind: A Multi-Modal Transformer Foundation Model for Tokamak Plasma Dynamics." arXiv
  2602.15084 (preprint). "Benchmarking Sensor Robustness in Plasma Diagnostic Models: A Systematic
  Evaluation on TokaMark." arXiv 2607.11915 (preprint, title only seen).
- "The Fusion Equilibrium Challenge: Inferring Magnetic Geometry Without Magnetic Diagnostics."
  NeurIPS 2026 competition; DIII-D and MAST shots on Hugging Face.
  https://neurips.cc/virtual/2026/competition/137686 , https://arxiv.org/abs/2609.01750
- ConStellaration (Proxima Fusion), NeurIPS 2025, arXiv 2506.19583: about 160k stellarator
  boundaries with optimization benchmarks. https://arxiv.org/abs/2506.19583
- Sonker, Capone, Rothstein, Kaga, Kolemen, Schneider. "Multi-Timescale Dynamics Model Bayesian
  Optimization for Plasma Stabilization in Tokamaks." ICML 2025 poster (validated on DIII-D).
- Jalalvand et al. "Multimodal super-resolution: discovering hidden physics and its application to
  fusion plasmas" (Diag2Diag). Nature Communications 16 (2025), doi 10.1038/s41467-025-63492-1.
  https://arxiv.org/abs/2405.05908 . Went to a science journal, not an ML conference.
- DisruptionBench, J. Fusion Energy 2025, https://link.springer.com/10.1007/s10894-025-00495-2 ;
  "Time Series Viewmakers for Robust Disruption Prediction", NeurIPS 2024 ML4PS **workshop**,
  arXiv 2410.11065.

**Open fusion data.**

| Source | Status | Relevance |
|---|---|---|
| FAIR-MAST (UKAEA) | data, metadata and site under CC BY-SA 4.0; Zarr/Parquet in a public S3 bucket (`s3://mast`, endpoint `s3.echo.stfc.ac.uk`) | [docs](https://ukaea.github.io/fair-mast/). Thomson scattering and Mirnov are present. **Whether MAST charge-exchange T_i/rotation is exposed was not confirmed today** (not in TokaMark's 39 signals). |
| LHD (NIFS) | open since FY2022, about 2 PB, also on the AWS Open Data registry; own terms of use, licence text not read | [repository](https://www-lhd.nifs.ac.jp/pub/Repository_en.html). NIFS documents CXS T_i/rotation; presence of analyzed CXS in the open repository not confirmed today. Heliotron, not a tokamak. |
| DIII-D | only the curated competition subset is public | equilibrium targets, not kinetic channels |
| KSTAR | no public data-release policy found; our manuscript states the shot files are not redistributable | a release needs KFE approval; do not plan around it |

**The bar, read off these papers.** (1) Evidence on several public datasets, usually five or more.
(2) Either a general method that leads on standard benchmarks, or a general finding shown at a scale
that removes the "one dataset" objection (thousands of runs, tens of datasets). (3) Code and data
released at submission. (4) A claim that fits in one sentence and concerns ML practice broadly.
Against that bar we have one private dataset, flat results on every model axis (size, operator
family, continuous-time encoders, ensembles), and seven findings that are each known in general
form. What is uncommon is having all of them measured, with pre-registration, on one real system.

## 3. Candidate designs

Odds are estimates for a single student with an 8 GB laptop GPU and no senior ML co-author; they
assume a clean submission in May 2027. A senior ML co-author would raise each by perhaps a third.

### Design A - evaluation audit on public data (E&D track). Recommended.

- **Claim.** In gap-filling of sparse, noisy sensor targets, the reported MSE skill of neural
  imputers over interpolation is (i) concentrated in a fraction of a percent of rows, some of which
  are artifact rows in the baseline's anchors and not model successes; (ii) of opposite sign to the
  median-row skill; (iii) recovered in large part by a causal noise-aware smoother with one to four
  parameters; (iv) bounded by a label-noise floor that can be estimated from the data. The paper
  delivers a four-readout audit (drop-k% fragility of the paired skill interval, mean vs median vs
  trimmed vs Huber skill, a baseline ladder from persistence through running mean and causal GP to
  offline GP, and floor-normalized skill) and applies it to published models.
- **New work required.** (1) Four to six public datasets with real (not synthetically masked)
  missingness and real measurement noise: PhysioNet 2012 or 2019 (labs against vitals), an
  air-quality set, a glucose-monitoring set, FAIR-MAST Thomson scattering against fast magnetics,
  optionally LHD. (2) Four to six published imputers run from their released code (SAITS, BRITS,
  CSDI, GRU-D, one transformer; PyPOTS / TSI-Bench provides them; all train on 8 GB). (3) The four
  readouts with a cluster bootstrap, about 3 seeds each. (4) A short theory section: the influence of
  a row on a ratio-of-sums skill, and the ceiling `1 - sigma_noise^2 / MSE_baseline`. (5) KSTAR as a
  fully worked case study with released error arrays (per-row errors are not raw shot data; confirm
  with the advisor that they may be released).
- **Prior work a reviewer will cite against it.** Stationarity Bias (2602.15637) for (ii);
  Broderick-Giordano-Meager for (i); Zeng et al., Physiome-ODE and TSI-Bench for (iii); Crusius et
  al. and the noise-ceiling literature for (iv); Agarwal et al. for robust aggregates; McGreivy and
  Hakim for weak baselines in scientific ML. The honest novelty is the conjunction and the
  anchor-artifact mechanism (the baseline, not the model, is what the outlier rows break), shown on
  natural missingness. That is an extension, not a new idea, and must be written that way.
- **Odds.** 12-18 % if at least two of the three effects replicate on at least three public
  datasets; under 3 % if they hold on KSTAR only.
- **Effort.** 14-18 weeks. Mostly CPU and data wrangling; GPU load is small models for hours, not
  days.

### Design B - noise-floor estimation for sparse time-series labels (main track or E&D).

- **Claim.** The achievable MSE for a sparse noisy target can be estimated without replicate
  measurements from successive differences and from pairs of targets that share a measurement
  process (our sqrt(2) scatter ratio), and published leaderboards for several imputation datasets
  are within a stated distance of that floor.
- **New work required.** An estimator with a consistency proof under stated assumptions (signal
  smooth relative to the sampling step, independent noise); validation on datasets where replicate
  noise is known; application to five or more datasets.
- **Against it.** Crusius et al.; Ishida et al.; fMRI noise-ceiling methods; successive-difference
  variance estimation goes back to von Neumann (1941). Our own record weakens it: the sqrt(T_i)
  scaling test failed on 0/4 splits x 2 populations, so the floor is "consistent with", not shown.
- **Odds.** 5-8 %. **Effort.** 12-16 weeks, theory-heavy. Best used as section (iv) of Design A.

### Design C - open benchmark for sparse-channel gap filling (E&D).

- **Claim.** A public multi-device task: fill a sparse kinetic channel (Thomson, charge exchange) at
  missing exposures from fast diagnostics plus the channel's own past, scored against a causal and
  an offline comparator ladder with floor-normalized metrics.
- **New work required.** Build from FAIR-MAST (and LHD) at terabyte scale; Croissant metadata;
  hosting; baselines; domain validation of MAST diagnostics we have never handled.
- **Against it.** TokaMark already covers MAST with 14 tasks, the data owners as authors, and a
  Thomson reconstruction task; TokaMind and the Fusion Equilibrium Challenge occupy the same
  ground. A reviewer will ask why this is not a fifteenth TokaMark task.
- **Odds.** 5-8 %. **Effort.** 20+ weeks; storage and bandwidth exceed a laptop.
- A KSTAR release would change this design, but nothing found suggests it can be obtained by May.

### Design D - position paper (Position track).

- **Claim.** "Gap-filling results for scientific sensors should be reported against a causal
  noise-aware smoother and a label-noise floor, with median and fragility readouts; skill over
  interpolation alone should not be accepted."
- **Against it.** Moretti et al. (ICML 2026) and Brigato et al. make the neighbouring argument for
  forecasting. Evidence from one private dataset is thin for a normative claim. The track's
  human-writing rule constrains how the text may be produced.
- **Odds.** 3-5 %. **Effort.** 6-8 weeks. Reasonable only as a by-product of Design A.

### Design E - methods paper (main track). Not recommended.

- **Claim.** A structured estimator for sparse targets from multi-rate sensors (gap-dependent gain
  plus gated fast-channel innovation), with optimality under an OU-plus-noise model.
- **Against it.** Our own record: the 20-parameter form was chosen post hoc on test rows, keeps
  0.72-0.78 of the PCHIP-relative skill but 0.31-0.50 of the causal-GP-relative skill; the derived
  structure "bought a third of what it promised" and the gate was negative; the full model does not
  separate from an offline GP. Prior work: GRU-D, BRITS, CSDI, SAITS, KalmanNet, deep Kalman
  filters, the soft-sensor literature, the TCV extended Kalman filter for Thomson intervals.
- **Odds.** 1-3 %. **Effort.** 20+ weeks with no evidence the method leads anywhere.

An AI-for-science main-track paper built on the physics question itself (what fast diagnostics add
to T_i, the bound for V_rot) is below 2 %: one device, one chord, private data, fast-channel share
not isolated. That paper belongs in a fusion journal, as the review loop already concluded.

## 4. Recommendation

Pursue **Design A**, with Design B's floor estimate as one section and Design D held back as a
possible companion. Keep the fusion-journal paper on its own track; Design A reuses its frozen
error arrays and does not need new KSTAR training.

**What must be true (kill criteria, each testable in 1-2 weeks on the laptop).**

1. **Fragility replicates.** On at least 2 of 3 quick public datasets (PhysioNet 2012, one
   air-quality set, one FAIR-MAST Thomson subset) and at least 2 published imputers (SAITS, BRITS),
   dropping the 0.1 % highest-influence rows changes the paired MSE skill over linear interpolation
   by at least half, or moves its 95 % interval to include 0. If skill is stable under a 1 % drop in
   every cell, stop: the effect is a KSTAR fit-failure story.
2. **Mean-median disagreement replicates.** At least 2 dataset-model cells with MSE skill interval
   above 0 and median-absolute-error skill interval below 0.
3. **Smoother share replicates.** A validation-tuned running mean or causal GP recovers at least
   40 % of the neural-over-interpolation skill in at least 2 datasets.
4. **Natural missingness is available.** At least 3 public datasets where targets are genuinely
   sparse and noisy. Benchmarks that mask clean ground truth (ETT-style) cannot show anchor
   artifacts and do not count.
5. **Not already published.** One day of search finds no paper that reports drop-k fragility of
   imputation skill together with a smoother ladder. Read 2602.15637 and TSI-Bench in full first.
6. **Release is permitted.** The advisor confirms that per-row KSTAR error arrays and code can be
   public at submission.

Pass rule: criteria 4-6 and at least two of 1-3. Anything less, do not spend the spring on NeurIPS.

**Fallbacks for the same manuscript** (not verified today): TMLR (rolling; judged on whether claims
are supported, not on novelty), the KDD Datasets and Benchmarks track (where TokaMark is cited),
DMLR. ICLR 2027's deadline has presumably passed.

## 5. Not verified

- NeurIPS 2027 dates and whether the track structure carries over.
- Charge-exchange availability in FAIR-MAST and in the open LHD repository; LHD licence text.
- Venues of Brigato et al., ConStellaration's track, and all items marked *(from memory)*.
- Position-track acceptance rate.
- KSTAR data-release policy (nothing found is not the same as no policy; ask KFE or the advisor).
- All acceptance odds are judgment, not measurement.
