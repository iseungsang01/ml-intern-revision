# Validation methods: field practice and methodology literature

Date: 2026-11-20. Read-only survey for the question "why did we validate this way, how do others validate, and is there a better method to borrow". It describes; it does not decide. Our method is summarised in section 4 from `PAPER_DESIGN.md` 3, 4, 7 and `ces_prediction/experiments/PREREGISTRATION_B70.md`.

## 0. How each source was read

- **Corpus PDFs** (`data/literature/`, git-ignored). Text was extracted with `pdftotext` (reading order). Each paper was searched for split, sample size, metric, baseline, interval, test and leakage terms, and the paragraphs around every hit were read. Pages are PDF page indices ("p. N"), as in `literature_map.md`; publisher cover pages count. This is targeted full-text reading, not a cover-to-cover read.
  - Read this way: Lee2026, Tian2026, Chai2019, Lin2024, Pavone2019, Jalalvand2025, Imrek2026, Wang2025TimeSeries, Jia2026, Yang2026, Shousha2023 (= Shousha et al. 2024, NF 64 026006), Wei2023, Joung2023, Abbate2021, Abbate2025, Boyer2021, Jung2026, Gill2024, Joung2026, KatesHarbeck2019, Keith2024, Kim2024KineticProfile, Kim2026AutomatedOutlier, Morosohk2023, Zheng2024, Matos2016, Gupta2026, Ravuri2021, Rasp2018; statistics: Gneiting2007, Goddard2013, Bates2024, Nadeau2003, Gallas2009, Owen2007, Ren2010, Saravanan2020, Weissgerber2015.
  - `murphy1988.pdf` is a scan without a text layer; its row below repeats `literature_map.md` 2.7 and was not re-read.
  - Not in the folder: the second Shousha 2024 (RTCAKENN) paper. Only the NF 64 026006 file exists.
- **Outside the corpus** (web; generic client; bibliographic data checked on Crossref/OpenAlex).
  - Full text, searched: Wan et al. 2021 (arXiv 2007.10552, pp. 7–8); Bergmeir, Hyndman, Koo 2018 (working-paper PDF); Benavoli et al. 2017 (JMLR PDF); Bouckaert & Frank 2004 (PDF).
  - Abstract only: Diebold & Mariano 1995; Diebold 2015; Hamill 1999; Wilks 2016; White 2000; Hansen 2005; Hansen, Lunde, Nason 2011; Giacomini & White 2006; Roberts et al. 2017; Cerqueira et al. 2020; Dietterich 1998; Bates et al. 2024 (also in corpus); Cameron, Gelbach, Miller 2008; Field & Welsh 2007.
  - Not verified (publisher abstract withheld; content from secondary descriptions found by search): Harvey, Leybourne, Newbold 1997; Tashman 2000; Bergmeir & Benítez 2012; Abbate et al. 2024 (Phys. Plasmas 31 042506). Not opened: Demšar 2006; Jolliffe & Stephenson 2012.

## 1. How comparable ML-fusion papers validate

"Unit" = what is assigned to train/test. "Hold-type ref." = a reference that uses the target's own past (persistence, constant profile). Dashes mean the paper does not state it in the pages read.

| # | key (device; venue) | unit | split | sizes | metric | references compared | interval / test | leakage note | pages |
|---|---|---|---|---|---|---|---|---|---|
| 1 | Lee2026 NN-CES (KSTAR; FED 222 115518) | discharge | single random 80/10/10, "shot-by-shot" | 321 discharges, 150,890 frames | R² per channel vs R | conventional beam-modulation fit (the label) | none on the metric; SD of IQR-filtered residuals used as error bars | test discharges unseen in training | p3–p4 |
| 2 | Tian2026 (HL-3; NF 67 016004) | discharge | 5-fold CV repeated 3 times, 15 estimates averaged; HL-2A → HL-3 transfer | 190 discharges | R² (primary), MAE, MRE | fit labels; per-channel models; single vs 10-model ensemble; scratch vs transfer | paired t-test over the 15 folds (ensemble vs single); SD over three seeds | discharge split because spectra of one shot are correlated; z-scores from training only | p4–p7, p12 |
| 3 | Chai2019 (EAST CXRS; PST 21 105103) | discharge | train / one test shot | 20 shots (1,878 frames) / 1 shot (95 frames) | relative error, MAPE | conventional fit | none | — | p5 |
| 4 | Lin2024 (EAST XCS; NF 64 106061) | not stated | single 70/20/10 | 530 shots, 2,649 slices | R², error histograms | Voigt fit; one discharge vs CXRS | spread over input ranges as uncertainty | — | p3–p4, p6 |
| 5 | Pavone2019 (W7-X; PPCF 61 075012) | synthetic sample | synthetic train/test; measured programmes by figure | 500,000 / 10,000 synthetic | visual agreement; k-NN distance to training set | full Bayesian (MCMC) inference | MCMC posterior SD | coverage of measured data checked by k-NN | p8, p10–p13 |
| 6 | Jalalvand2025 Diag2Diag (DIII-D; Nat. Commun. 16 8506) | "data-points" | single 80/20 train/validation | 4,000 random discharges (2017–22), ~3.4 M points | L1 loss; physics checks (ELM, RMP) where no ground truth exists | none quantitative (argues interpolation misses ELMs) | BNN epistemic vs Thomson error | — | p5, p7, p9 |
| 7 | Imrek2026 (COMPASS; PPCF 68 065049) | discharge | single 80/20, plus 5-fold inside the training part | 2,764 discharges, 51,476 slices (shots 17000–21812) | mean and median relative error; error / Thomson uncertainty | Thomson uncertainty | error bars = spread over the k folds | discharge split "to reduce correlations" | p5–p6, p10–p12 |
| 8 | Wang2025 (EAST T_e; NF 65 076008) | discharge | chronological 80/20 | 1,873 discharges; test 374 | MSE, R², similarity, coverage | MLP, LSTM, GRU, TST, InceptionTime | none | — | p5, p8–p10 |
| 9 | Jia2026 (EAST Z_eff; NF 67 016012) | discharge | chronological 80/20 | 1,566 discharges (2023–24) | MRE, MAE, RMSE, R², similarity, coverage | ablated architectures and inputs | none | chronological to mimic use on later discharges | p6, p8, p13 |
| 10 | Yang2026 FusionMAE (HL-3; Commun. Phys.) | discharge | single random | 2,445 discharges, four campaigns; 1,950 / 495 | PCC, RMSE; per-shot PCC vs date | downstream tasks on raw vs embedded inputs | band from random input masking | early-campaign shots score lower | p4, p8 |
| 11 | Shousha2023 RTCAKENN (DIII-D; NF 64 026006) | not stated | held-out slices | 696 discharges, 18,959 slices; test 1,334 slices | Pearson R | CAKE kinetic EFIT (label) | none (10-net mean) | — | p5 |
| 12 | Wei2023 (DIII-D q; NF 63 086020) | discharge | single random | 13,491 shots: 6,780 / 3,339 / 3,372; 164,782 test slices | MSE; q95, q_min | EFIT01 (no MSE) against EFIT02 | none | — | p4, p6 |
| 13 | Joung2023 GS-DeepNet (KSTAR; Sci. Rep. 13) | space-time point | random 80/5/15 | 50 discharges (2019), ~2×10⁷ points | R²; RMSE of tan γ vs MSE | EFIT; MSE data | MC dropout | — | p8–p9 |
| 14 | Abbate2021 (DIII-D; NF 61 046027) | discharge (last digit of shot number) | single fixed | test 389 shots / 15,132; val 415; train 3,117 | R² of the predicted 200 ms change | none explicit; the model predicts the change from the present profile | none | many shots "reloaded" and very similar; the net may memorise discharge patterns | p7, p9, p11 |
| 15 | Abbate2025 (DIII-D, AUG; NF 65 056014) | discharge, by regime | train at low I_p, test near and far (I_p > 1.3 MA); AUG data added | far test 1,720 samples from 309 discharges | ITER-1999-type profile error | **constant initial profile**, physics simulators, ML | 95 % two-sided t-test intervals of the difference | — | p5, p13–p17 |
| 16 | Boyer2021 (NSTX-U, NSTX; NF 61 046024) | discharge | random; also online: refit every 10th shot on earlier shots only | 281 shots / 30,663; NSTX 1,837 shots, ~995,000 slices | per-shot RMSE, distributions | all-data model vs online model | ensemble or dropout SD (found not to track error) | shot-level assignment against over-estimated generalisation | p4–p5, p10–p12 |
| 17 | Jung2026 (KSTAR; JKPS 88 1079) | discharge within a campaign | one model per year; unseen shots of the same year | 2020–22 (their Table 1) | range-normalised MAE | MCMC profile fit | none | per-year models avoid domain shift | p6–p8 |
| 18 | Gill2024 (DIII-D BES; MLST 5 035012) | discharge | stratified 72.5/15/12.5 | 330 discharges (2012–23); test 44 | F1, precision, recall, ROC AUC | — | none | — | p6–p7, p11–p12 |
| 19 | Joung2026 (DIII-D → KSTAR; Commun. Phys.) | device | train DIII-D only, zero-shot on KSTAR | ~1,400 DIII-D, ~200 KSTAR discharges | ELM probability, saliency statistics | — | SD across ELMs | — | p7–p10 |
| 20 | KatesHarbeck2019 (DIII-D, JET; Nature 568) | discharge | DIII-D uniform over shots; JET carbon wall → ITER-like wall; cross-machine | their Table 1 | AUC at 30 ms | classical ML (boosted trees best) | none in Table 1 | — | p2–p3, p7 |
| 21 | Keith2024 (C-Mod, DIII-D; JFE 43 21) | discharge | stratified 60/20/20 | 5,682 and 7,417 shots | AUROC, AUWTC | RF, Kaplan–Meier, Cox, deep survival | **test shots resampled, 50 draws, median and IQR; same draws for every model** | — | p7–p8 |
| 22 | Zheng2024 (HL-3; NF 64 126041) | discharge | chronological by shot number 70/20/10 | 1,159 shots, 1.5 M slices | r, R², normalised residual | EFIT (label) | none | normalisation from training only | p4, p9 |
| 23 | Wan2021 (EAST; NF 61 066015; outside corpus) | discharge | chronological by shot order 6:2:2 | 3,476 shots; test 695 | similarity | — | none | **adjacent shots have similar parameters, so the split follows shot order** | arXiv pp. 7–8 |
| 24 | Gupta2026 / TokaMark (MAST; preprint) | discharge | 80/10/10 shot-level | — | NRMSE (1 = predicting the mean) | mean; fill methods | none | — | p2, p5 |
| 25 | Matos2016 (JET; FED 114) | not stated | per-campaign 80/10/10 | four campaign sets | SSIM, PSNR, NRMSE | — | none | — | p6 |

Not tabulated (no held-out aggregate): Kim2024KineticProfile (120 T_i profiles, random 7:3, classifier F1; p5), Kim2026AutomatedOutlier (demonstrations on KSTAR shot 32364 against a Gaussian likelihood; p9–p13), Morosohk2023 (offline observer tests on selected DIII-D shots; p10).

Neighbouring field in the corpus: Ravuri2021 (radar nowcasting, Nature 597) trains on 2016–18 and tests on 2019, adds a weekly split, and tests differences with paired permutation tests using alternating weeks as the independent unit (n = 26) (p3, p7, p9). Rasp2018 (MWR 146) tests on a later year "to mimic operational conditions" and uses Diebold–Mariano tests with Benjamini–Hochberg control across stations (p3, p11).

## 2. Tally (25 rows of section 1; a paper can count under more than one design)

| item | count | rows |
|---|---|---|
| unit = discharge (or device) | 19 | 1–3, 7–10, 12, 14–24 |
| unit = rows, points, slices, synthetic, or not stated | 6 | 4, 5, 6, 11, 13, 25 |
| single hold-out split (random or fixed) | 18 | 1, 3–7, 10–14, 16–18, 20, 21, 24, 25 |
| k-fold CV that produces the test score | 1 | 2 (repeated 5-fold × 3, 15 estimates, as ours) |
| chronological (shot order, later campaign, online) | 6 | 8, 9, 16, 20 (JET), 22, 23 |
| regime extrapolation | 1 | 15 |
| cross-device | 4 | 2, 15, 19, 20 |
| any interval or test on the metric | 4 | 2 (paired t over folds), 7 (fold spread), 15 (t-test CI), 21 (shot bootstrap) |
| interval from resampling discharges | 1 | 21 |
| hold-type reference | 1 | 15 (constant initial profile); row 14 predicts the change only |
| gap-filling / reconstruction paper that compares with interpolation or hold | 0 of 5 | 6, 7, 8, 9, 10 |
| pre-registration mentioned | 0 | — |
| warns about near-duplicate or adjacent discharges | 2 | 14, 23 (rows 2 and 7 discuss correlation within a shot only) |

Metrics: R² or r (rows 1, 2, 4, 8, 9, 10, 11, 13, 14, 22); MAE, MRE or relative error (2, 3, 7, 9, 17); MSE, RMSE or NRMSE (8, 9, 10, 12, 16, 24, 25); AUC/F1 (18, 20, 21). No row uses an MSE skill against persistence or a causal reference; row 24 normalises by the mean predictor and row 15 reports differences against an ML baseline.

## 3. Methodology references

Fit: does it apply to rows nested in discharges nested in sessions, several estimators scored on identical rows, and 15 trained networks (3 partitions × 5 folds, one initialisation each)? Cost: "CPU" = re-read of stored predictions or fold scores; "GPU" = new training.

| reference | what it is | assumes | question it answers | fit to our structure | cost |
|---|---|---|---|---|---|
| Diebold & Mariano 1995 (doi:10.1080/07350015.1995.10524599); Harvey, Leybourne, Newbold 1997 (doi:10.1016/S0169-2070(96)00719-4, not verified) | test of equal expected loss of two forecasts from the loss-differential series; HLN adds a small-sample correction and t critical values | covariance-stationary loss differential; any loss | are two sets of forecasts equally accurate? | per-discharge mean loss differences give a clustered DM test; with 639 discharges the HLN correction is immaterial. Diebold 2015 (doi:10.1080/07350015.2014.983236): DM compares forecasts, not models | CPU |
| Giacomini & White 2006 (doi:10.1111/j.1468-0262.2006.00718.x) | conditional and unconditional tests of forecasting *methods*, estimation included | rolling estimation windows | is the method (training included) better, and when? | closer to our question than DM; needs refits at many origins | GPU |
| White 2000 (doi:10.1111/1468-0262.00152); Hansen 2005 SPA (doi:10.1198/073500105000000063) | bootstrap test that the best of many models beats a benchmark, allowing for the search | stationary bootstrap of loss differentials | is the selected estimator's edge more than search luck? | applies to estimators chosen after many batches on the same folds; needs stored predictions of the candidates | CPU (stored arms only) |
| Hansen, Lunde, Nason 2011 MCS (doi:10.3982/ECTA5771) | set of models that contains the best with given confidence | bootstrap of loss differentials | which estimators cannot be told apart? | estimators on identical rows; discharge-cluster bootstrap in place of the block bootstrap | CPU |
| Hamill 1999 (doi:10.1175/1520-0434(1999)014<0155:HTFENP>2.0.CO;2) | paired t, Wilcoxon and resampling tests; data pooled per case day to absorb spatial correlation | case days independent | is forecast A better than B? | our discharge plays the case day; between-day (session) correlation is not covered by it | CPU |
| Wilks 2016 (doi:10.1175/BAMS-D-15-00267.1) | false discovery rate control for many simultaneous tests | — | which of many "significant" results survive multiplicity? | applies to the set of registered statements | CPU |
| Jolliffe & Stephenson 2012 (doi:10.1002/9781119960003; not opened) | verification textbook | — | — | skill against persistence or climatology is standard there | — |
| Tashman 2000 (doi:10.1016/S0169-2070(00)00065-0; not verified) | rolling-origin out-of-sample evaluation | time order matters | how does accuracy hold forward in time? | shot order within the campaign | GPU |
| Bergmeir & Benítez 2012 (doi:10.1016/j.ins.2011.12.028; not verified); Bergmeir, Hyndman, Koo 2018 (doi:10.1016/j.csda.2017.11.003) | blocked CV for stationary series; K-fold CV is valid for purely autoregressive models when residuals are serially uncorrelated, else drop dependent neighbours or evaluate out of sample | stationarity; uncorrelated residuals | when is CV valid for dependent data? | our dependence within a series is absorbed by the discharge unit; the open part is dependence between discharges | CPU (residual correlation in shot order) |
| Roberts et al. 2017 (doi:10.1111/ecog.02881) | block CV along the dependence structure (time, space, random effects) | blocks match the structure | does random CV under-estimate error? Abstract: ignoring structure causes "serious underestimation"; blocking may induce extrapolation | blocks = sessions or shot-number ranges | GPU |
| Cerqueira et al. 2020 (doi:10.1007/s10994-020-05910-7) | empirical comparison of performance estimators | — | CV for stationary series; order-preserving out-of-sample methods most accurate for non-stationary real series | campaign drift question | GPU |
| Dietterich 1998 (doi:10.1162/089976698300017197) | five tests for comparing learners; 5×2cv | — | the paired t over random splits has high type I error; the 10-fold CV t is somewhat elevated | warns against a plain t over our 15 fold scores | CPU (5×2cv itself: GPU) |
| Nadeau & Bengio 2003 (doi:10.1023/A:1024068626366); Bouckaert & Frank 2004 (doi:10.1007/978-3-540-24775-3_3) | corrected variance (1/J + n_test/n_train) for overlapping training sets; corrected repeated k-fold test with better replicability | random or k-fold resampling | interval for the expected difference of the learning procedure | 15 (or 40) fold scores; already used in `PROJECT_KNOWLEDGE.md` 4.32 | CPU |
| Demšar 2006 (JMLR 7:1–30; not opened) | Wilcoxon and Friedman tests across many data sets | many independent data sets | ranking learners across data sets | no: one data set (two with MAST) | — |
| Benavoli et al. 2017 (JMLR 18(77):1–36) | Bayesian correlated t-test with a region of practical equivalence | correlation set by the Nadeau–Bengio ratio | probability that A is practically better, equal or worse on one data set | 15 fold scores | CPU |
| Bates, Hastie, Tibshirani 2024 (doi:10.1080/01621459.2023.2197686) | CV estimates error averaged over training sets; naive CV intervals under-cover; nested CV fixes it | — | coverage of a CV interval | our interval is conditional on the trained networks | GPU (heavy) |
| Cameron, Gelbach, Miller 2008 (doi:10.1162/rest.90.3.414) | wild cluster bootstrap-t | few clusters (5–30) over-reject | inference with few clusters | not needed for 639 discharges; relevant if sessions or folds are the unit | CPU |
| Field & Welsh 2007 (doi:10.1111/j.1467-9868.2007.00593.x) | theory of bootstraps for clustered data | — | the cluster bootstrap is consistent under both random-effect and transformation models | supports resampling whole discharges | CPU (current) |
| Ren2010; Saravanan2020 (corpus) | hierarchical bootstrap: resample the top level with replacement | levels known | interval for nested data; keeps type I rate, conservative | top level may be the session, not the discharge | CPU if a session label exists |
| Owen2007; Gallas2009 (corpus) | bootstraps for crossed factors; resampling both factors over-states variance | crossed design | variance with two random factors | partition × discharge is partly crossed | — |
| Goddard2013 (corpus) | MSSS with a two-level bootstrap: start dates resampled in neighbouring pairs, ensemble members redrawn | — | precedent for our two-level interval, with a block step for time correlation | the block step has no analogue in ours | CPU |
| Gneiting2007; murphy1988 (corpus) | compare on identical cases; skill scores are generally improper (Gneiting p5); MSE skill and its decomposition (Murphy) | — | — | met (identical rows) | — |
| Weissgerber2015 (corpus) | show distributions, not means | — | — | per-discharge distributions | CPU |

Fusion-specific methodology: Wan et al. 2021 (doi:10.1088/1741-4326/abf419) split by shot order because adjacent shots are similar; Abbate2021 p11 warns about reloaded, similar shots; Abbate et al. 2024 (Phys. Plasmas 31 042506; not verified) argues for large databases and use-case baselines and reports no significant advantage of transport codes over a two-parameter baseline for temperature profiles (from the abstract as quoted by search). No paper found blocks folds by run day or session.

## 4. Our method against field practice and the methodology literature

Our method (`PAPER_DESIGN.md` 3, 4, 7; `PREREGISTRATION_B70.md` 2–4): 639 discharges of one campaign; pooled 5-fold CV over discharges on re-partitions r2–r4 (each fold about 383 train, 128 validation, 127–128 test discharges; `dataset_ledger.md`); one initialisation per fold, epoch cap 80; the five test folds of a partition joined into one readout, the point the mean over three partitions; skill = 1 − MSE(A)/MSE(R) on identical rows; 95 % interval by drawing a partition and resampling its test discharges (20,000 draws); a difference claimed when the interval excludes zero; median absolute error beside every main result; statements registered before the run.

**(a) Against field practice**

- Stricter:
  - Split unit: discharge, as in 19 of 25 rows; 6 rows split rows, points or do not state the unit.
  - Repeated k-fold over discharges producing the test score: only Tian2026 does the same (5 × 3, 15 estimates). Tian2026 averages fold scores and tests with a paired t over the 15 folds; we pool the test rows and resample discharges.
  - References that read the target's own past (hold, causal GP): only Abbate2025 has one (constant profile). None of the five gap-filling papers (Diag2Diag, COMPASS, EAST T_e and Z_eff, FusionMAE) compares with interpolation or hold.
  - Interval: 4 of 25 give any interval or test; only Keith2024 resamples test discharges (50 draws, median and IQR, same draws for all models). Ours is paired, discharge-level, 20,000 draws.
  - Pre-registration and median error beside the mean: not seen in the 25 rows.
- Weaker:
  - Time order. Six rows use a chronological design (Wang2025, Jia2026, Zheng2024, Wan2021, Kates-Harbeck's JET test, Boyer's online refits), and both weather papers test on a later period. Our main readout uses random folds within one campaign; a single time split (shot-sorted, 416 train / 128 validation / 97 test, `dataset_ledger.md` row TS) is in the supplement only (`PAPER_DESIGN.md` 4.4), and B.69 on it is registered and partly run.
  - Neighbouring discharges. Wan2021 and Abbate2021 warn that adjacent or reloaded shots are similar. In our set, 430 of the 640 gaps between consecutive shot numbers equal 1, and 619 of 641 discharges have another discharge within ±3 shot numbers (computed from the file names). Under the random pooled folds each neighbour of a test discharge lies outside the test fold with probability about 0.8 (training about 0.6, validation about 0.2; validation discharges also enter the second-stage fit, `dataset_ledger.md` R-fit64). The files carry no date or session label.

**(b) Against the methodology literature**

- Random discharge folds vs session-blocked or chronological folds. Roberts 2017 (abstract): random CV under structured data leads to "serious underestimation" of error; blocks should follow the structure. Cerqueira 2020: order-preserving out-of-sample estimates are the most accurate for non-stationary real series. Our discharge unit handles dependence within a discharge (Bergmeir–Hyndman–Koo's condition concerns that level); dependence between neighbouring discharges is not blocked.
- Pooled-row MSE vs per-discharge scores. Our skill weights discharges by their scored rows. The bootstrap resamples whole discharges, which is the cluster bootstrap that Field & Welsh 2007 find consistent and Ren2010 prescribes at the top level. If sessions are the top level, resampling discharges treats correlated neighbours as independent (Ren2010, Saravanan2020). Hamill 1999 and Ravuri2021 pool to the independent unit (case day, week) before testing; Weissgerber2015 asks for the distribution. A per-discharge distribution is not part of the main readout.
- Bootstrap over test discharges vs variance from training. Bates 2024: CV estimates error averaged over training sets, and naive intervals under-cover. Our interval adds the partition draw (three partitions) but not initialisation (one per fold). Project record: with three initialisations per split and the initialisation drawn in the interval, the network-over-causal-GP and network-over-twin statements held in 2 of 4 splits, against 4 of 4 with one initialisation (`PROJECT_KNOWLEDGE.md` 4.26). On the 40-fold pooled arm, the Nadeau–Bengio fold interval for the network over the causal GP was +0.089 [+0.057, +0.121] against the two-level bootstrap +0.089 [+0.064, +0.112] (4.32): same point, about one third wider.
- Number of comparisons. B.70 registers six statements (D1-1 … D1-6), each read as its own 95 % interval; no multiplicity rule is stated in `PAPER_DESIGN.md` 7 or `PREREGISTRATION_B70.md`. Wilks 2016 (FDR) and White 2000 / Hansen 2005 (a best model chosen after a search) address this. The main estimator was designated after B.64 on r2–r4, and the main readout is on the same r2–r4 (`PAPER_DESIGN.md` 4.2, 5). The network arm also exists on r2–r9 (4.32); the second stage exists only on r2–r4.
- Paired scoring on identical rows meets Gneiting 2007 and Hamill 1999. A clustered DM test on per-discharge loss differences would answer the same question as our paired bootstrap.

## 5. Candidate changes (cost and what each would answer)

| # | change | cost | answers |
|---|---|---|---|
| C1 | Per-discharge readout: MSE difference and skill per test discharge, share of discharges where the estimator wins, discharge-weighted mean, median over discharges | CPU re-read | whether the pooled gain is carried by a few long or extreme discharges |
| C2 | Block bootstrap: resample runs of neighbouring shot numbers (e.g. gaps ≤ 3) instead of single discharges | CPU re-read | whether correlation between neighbouring discharges widens the interval |
| C3 | Neighbour stratification: score test discharges by how many of their ±1…±3 neighbours sit outside the test fold (training or validation) | CPU re-read | a weak leakage indicator: does proximity to training discharges raise skill? |
| C4 | Blocked folds: five folds of contiguous shot-number ranges, three offsets; retrain networks, twins and second stage | GPU (15 + 15 networks, second stage) | the score when neighbouring shots are never on both sides |
| C5 | Rolling origin in shot order (several origins; train on earlier, test on the next block), as Tashman 2000 and Boyer2021's online refits | GPU | forward-in-time use and drift within the campaign; the one-origin time split is a single case |
| C6 | Initialisation in the interval: three initialisations per fold on r2–r4, interval that draws one | GPU (30 more networks per arm) | whether the main statements survive initialisation spread on the pooled folds |
| C7 | Fold-level checks on the 15 fold scores: Nadeau–Bengio corrected t, Bouckaert–Frank corrected repeated k-fold test, Bayesian correlated t-test with a stated practical-equivalence region | CPU | agreement with methods that account for overlapping training sets; probability of practical equivalence |
| C8 | Multiplicity: Holm or Benjamini–Hochberg over the registered statements; a Model Confidence Set over the estimators on identical rows with a discharge-cluster bootstrap | CPU | which statements survive family-wise or FDR control; which estimators cannot be separated |
| C9 | Selection effect: Reality Check or SPA over candidates whose stored predictions exist; or confirm the designated estimator on partitions not used to select it (r5–r9 hold networks; the second stage there needs its inner-fold estimates, cost not checked here) | CPU for SPA; GPU or CPU for r5–r9 | whether the chosen estimator's edge survives the search |
| C10 | Nested CV interval (Bates 2024) | GPU, many trainings per partition | coverage of the CV interval itself |
| C11 | Clustered Diebold–Mariano test on per-discharge loss differences beside the bootstrap | CPU | the same question in the form forecast readers expect |
