# 통계 방법론 문헌 점검 2차 — 1차에서 선례를 못 찾은 다섯 항목 (2026-11-10)

1차 보고서: `docs/lit_2026-11-09/lit_statistics.md`. 이 문서는 1차의 "종합 표"에서 **미발견**으로 남은 항목만 다시 팠고, 1차에 이미 실린 문헌(Saravanan 2020, Ren 2010, Bouthillier 2021, Nadeau–Bengio, Bates 2024, Murphy 1988, Gneiting 계열, Hofman 2023, Klein & Roodman 2005, Lerch 2017 등)은 되풀이하지 않는다.

## 조사 경로와 표기

- 경로: OpenAlex API(`search`, `filter=fulltext.search:` — 1차에서 예산 소진으로 못 돌린 전문 검색을 이번에 실행), Crossref API(`query.bibliographic`로 DOI·서지 대조), arXiv API(id 조회)와 arXiv PDF 본문, 공개 PDF 본문(Springer OA, EMA, ICH)을 PyMuPDF로 추출해 문장 대조, WebSearch.
- **[본문]** PDF 본문에서 인용 문장을 직접 확인. **[초록]** OpenAlex·arXiv 초록만 확인. **[서지]** DOI·제목·저자·권호만 확인(내용 미확인). **[전문 색인]** OpenAlex 전문 검색에 걸렸으나 본문을 열지 못함(IOP 등 차단) — 해당 문자열이 본문에 있다는 것만 안다.
- "우리 방법과의 거리"는 세 단계로 적는다: **같은 구조** / **같은 발상·다른 수준** / **관련만 있음**.
- 음성 결과마다 같은 경로의 **양성 대조**를 붙였다.
- 2026년 arXiv·Research Square·Zenodo 문서는 **심사 전 원고**다. 표에 "preprint"로 표시했고, 논문에서 인용할 때도 그렇게 다뤄야 한다.

---

## 항목 1. 2수준 부트스트랩 — 1수준 = 학습된 모델(학습 반복·seed·분할), 2수준 = 시험 단위

우리 통계량: 풀링된 out-of-fold 예측 위에서 (학습 반복 → 방전) 순으로 재표집해 skill = 1 − ΣSE_model/ΣSE_ref 의 구간 하나를 얻는다. 구간은 학습 분산과 시험 표본 분산을 함께 담는다.

### 새로 찾은 선례

| 문헌 | DOI / id | 분야 | 우리 방법과의 거리 | 근거 인용 |
|---|---|---|---|---|
| Sellam, Yadlowsky, Wei, Saphra, D'Amour, Linzen 외, "The MultiBERTs: BERT Reproductions for Robustness Analysis", ICLR 2022 | arXiv:2106.16163 (10.48550/arXiv.2106.16163) | NLP 평가 방법론 | **같은 구조.** seed와 시험 사례를 각각 복원 재표집해 한 구간을 얻고, 두 모델을 같은 재표본으로 짝지어 차이의 분포를 본다. 차이점 셋: 시험 단위가 군집(방전)이 아니라 개별 사례, 1수준이 독립 seed(우리는 교차검증 반복), 통계량이 평균(우리는 두 합의 비). **[본문]** | "We consider randomness arising from: 1. The choice of pre-training seed S ∼ M 2. The choice of fine-tuning seed T ∼ N 3. The choice of test sample (X, Y) ∼ D. The Multi-Bootstrap procedure allows us to account for all of the above." / "we generate a bootstrap sample from the full sample with replacement separately over both the randomness from the pre-training seed s and from the test set (X, Y)." / (짝지은 설계) "one must use the same bootstrap samples of the seeds … and test examples … for both models." / 내포 구조는 "motivated by Field & Welsh (2007)'s recommendations for bootstrapping clustered data." |
| Knopov & Shvai, "Equivalence testing and run-count design for stochastic machine-learning experiments" (preprint, 2026-10) | arXiv:2610.23700 | ML 평가 방법론 | **같은 구조**(seed × item 교차 2수준, 짝지음, 백분위 구간). 차이: 시험 단위가 item, 용도가 보조적(동등성 판정은 seed 수준 TOST). **[본문]** | "We analyse the two-level, crossed resampling scheme that mirrors the design of the experiment. At the outer level, runs (seeds) are resampled with replacement within each method; at the inner level, one resample of ID item indices … are drawn per bootstrap iteration and shared by all selected runs of both methods, replaying the paired evaluation on a common test sample. The percentile interval … is reported (Davison & Hinkley, 1997)." |
| Goddard, Kumar, Solomon, Smith, Boer 외, "A verification framework for interannual-to-decadal predictions experiments", *Climate Dynamics* 40, 245–272 (2013) | 10.1007/s00382-012-1481-2 | 기후 예측 검증 | **같은 구조, 물리과학.** 시작 시점(시험 단위)을 복원 재표집한 뒤 각 시점 안에서 앙상블 멤버(모델 실현)를 복원 재표집하고, 예측과 기준 예측을 같은 재표본에 태워 **Murphy의 MSE 스킬 점수(MSSS)** 분포를 만든다. 우리와 수준 순서가 반대(시험 단위 → 모델 실현)이고 멤버가 시점 안에 내포. **[본문]** | "a set of n start times, J(j,k), are randomly selected from the set of years with replacement; followed by the random selection with replacement of N ensemble members taken from each start time" / "The same bootstrap-generated time series can be used to generate a distribution of values for the MSSS … The fraction of observed negative values again serves as the p value for the test" |
| Gallas, Bandos, Samuelson, Wagner, "A Framework for Random-Effects ROC Analysis: Biases with the Bootstrap and Other Variance Estimators", *Commun. Stat. Theory Methods* 38, 2586–2603 (2009) | 10.1080/03610920802610084 | 의료영상 판독 연구(MRMC) | **같은 구조(완전 교차 2요인: 판독자 × 사례).** 판독자를 "학습된 모델"로 읽으면 우리 설계와 동형. 중요한 경고가 붙는다 — 교차 부트스트랩의 분산 추정은 치우친다. **[초록]** | "we analyze the three-way bootstrap estimate of the variance of the reader-averaged nonparametric area under the ROC curve … The experiment we consider is fully crossed in that each reader reads each case … We find that the bootstrap estimate of the variance of the reader-averaged AUC is biased" |
| Beiden, Wagner, Campbell, "Components-of-variance models and multiple-bootstrap experiments: an alternative method for random-effects ROC analysis", *Acad. Radiol.* 7, 341–349 (2000) | 10.1016/S1076-6332(00)80008-2 | MRMC | 같은 구조(판독자·사례를 따로 또 함께 재표집해 분산 성분 분리)로 알려져 있으나 **[서지]** 만 확인 — 초록을 열지 못했다. 내용 인용은 본문 확인 뒤에. | 제목만: "multiple-bootstrap experiments … random-effects ROC analysis" |
| Beiden, Maloof, Wagner, "A general model for finite-sample effects in training and testing of competing classifiers", *IEEE TPAMI* 25, 1561–1569 (2003) | 10.1109/TPAMI.2003.1251149 | 패턴인식 평가 | **같은 발상·다른 수준.** MRMC 분산 성분 모형을 "학습 표본 × 시험 표본"으로 옮겨, 경쟁 알고리즘 비교에서는 학습 쪽 분산이 지배적일 수 있음을 보임. 재표집 대상이 학습 집합(seed가 아님). **[초록]** | "when competing algorithms are compared, the finite training sample more commonly dominates this uncertainty. This general problem in SPR is analyzed using a formal structure recently developed for multivariate random-effects receiver operating characteristic (ROC) analysis … the importance of the contribution of finite trainers to the uncertainties argues for some form of bootstrap analysis to sample that uncertainty." |
| Chen, Gallas, Yousef, "Classifier variability: Accounting for training and testing", *Pattern Recognition* 45, 2661–2671 (2012); 학회판 Chen & Gallas, *Proc. SPIE* 7624, 762404 (2010) | 10.1016/j.patcog.2011.12.024; 10.1117/12.844605 | 의료 AI 평가 | **같은 발상.** 학습을 무작위 효과로 두는 세 단계 평가 틀; 학습·시험 두 분산원을 MRMC의 판독자·사례에 대응시킴. 분산은 부트스트랩이 아니라 해석식(U-통계량)으로 구함. **[서지 + 2차 출처]** (초록이 OpenAlex에 없어 검색 스니펫으로만 확인) | (스니펫) "training treated as a random effect, the estimated performance generalizable to both the population of training sets and the population of testing sets" |
| Yousef, Wagner, Loew, "Assessing Classifiers from Two Independent Data Sets Using ROC Analysis: A Nonparametric Approach", *IEEE TPAMI* 28, 1809–1817 (2006) | 10.1109/TPAMI.2006.218 | 패턴인식 평가 | 같은 발상(학습 집합 조건부 AUC의 평균·분산을 분산 성분으로 분해). **[초록]** | "This expression exhibits several components of variance that facilitate an understanding for the sources of uncertainty of that estimate." |
| McCullagh, "Resampling and Exchangeable Arrays", *Bernoulli* 6, 285–301 (2000) | 10.2307/3318577 | 수리통계 | 이론 배경. 교차 배열에서는 정확한 부트스트랩이 없다. **[서지]** (Owen 2007·2012 초록의 인용으로 내용 확인) | Owen 2007 초록: "McCullagh [Bernoulli 6 (2000) 285–301] has shown that no bootstrap method is exact." |
| Owen, "The pigeonhole bootstrap", *Ann. Appl. Stat.* 1, 386–411 (2007); Owen & Eckles, "Bootstrapping data arrays of arbitrary order", *Ann. Appl. Stat.* 6, 895–927 (2012) | 10.1214/07-AOAS122; 10.1214/12-AOAS547 | 수리통계 | **같은 구조의 이론.** 행과 열을 따로 재표집하는 교차 2요인 부트스트랩의 성질: 평균적으로 일치하나 분산을 약간 과대추정(보수적). **[초록]** | "an alternative bootstrap separately resampling rows and columns of the data matrix satisfies a mean consistency property even in heteroscedastic crossed unbalanced random effects models" / "the proposed bootstrap is mildly conservative, meaning biased toward overestimating the variance" |
| Cameron, Gelbach, Miller, "Robust Inference With Multiway Clustering", *JBES* 29, 238–249 (2011); Menzel, "Bootstrap With Cluster-Dependence in Two or More Dimensions", *Econometrica* 89, 2143–2188 (2021); Davezies, D'Haultfœuille, Guyonvarch, "Empirical process results for exchangeable arrays", *Ann. Stat.* 49, 845–862 (2021) | 10.1198/jbes.2010.07136; 10.3982/ECTA15383; 10.1214/20-AOS1981 | 계량경제 | 같은 구조의 이론(비내포 2방향 군집). 모델 비교 용도는 아님. **[초록]** (Davezies는 [서지]) | Menzel: "We propose a bootstrap procedure for data that may exhibit cluster-dependence in two or more dimensions … there exists no procedure for estimating the limiting distribution of the sample mean under two-way clustering that achieves uniform consistency." |
| Tong & Brennan, "Bootstrap Estimates of Standard Errors in Generalizability Theory", *Educ. Psychol. Meas.* 67, 804–817 (2007) | 10.1177/0013164407301533 | 심리측정(일반화가능도 이론) | 같은 구조(사람 × 문항 × 채점자 다면 설계의 요인별 부트스트랩), 목적은 분산 성분의 표준오차. 어느 요인을 재표집할지에 따라 치우침이 달라진다는 같은 경고. **[초록]** | "Researchers have speculated about the potential applicability of the bootstrap … but they have identified problems (especially bias) in using the bootstrap … a proposed set of rules for picking a bootstrap procedure" |
| Tsamardinos, Greasidou, Borboudakis, "Bootstrapping the out-of-sample predictions for efficient and accurate cross-validation", *Machine Learning* 107, 1895–1922 (2018) | 10.1007/s10994-018-5714-4 | ML 방법론 | **같은 발상·다른 수준.** "풀링된 out-of-fold 예측을 재학습 없이 부트스트랩한다"는 우리 2수준의 재료와 같음. 1수준(학습 반복) 재표집은 없고 목적은 선택 편향 보정. **[초록]** | "BBC-CV's main idea is to bootstrap the whole process of selecting the best-performing configuration on the out-of-sample predictions of each configuration, without additional training of models." |
| Sälevä, Ataman, Lignos, "Beyond statistical significance: Quantifying uncertainty and statistical variability in multilingual and multitask NLP evaluation" (preprint, 2025) | arXiv:2509.22612 | NLP 평가 | 같은 발상(모델 쪽·데이터 쪽 변동을 함께 재표집). **[초록]** | "experimental variation in performance scores arises from both model and data-related sources, and that accounting for both of them is necessary to avoid substantially underestimating the overall variability over hypothetical replications." |
| Miller, "Adding Error Bars to Evals" (preprint, 2024) | arXiv:2411.00640 | LLM 평가 | 관련만 있음. 시험 문항이 묶여 있으면 군집 표준오차, 두 모델은 문항 수준 짝지은 차이로 추론. 학습 반복 수준은 없음. **[본문]** | "When questions are drawn in related groups, computing clustered standard errors" / "conducting statistical inference on the question-level paired differences" |

1차에 이미 실린 Bouthillier(2021)는 본문을 다시 확인했다: 데이터 표집 분산을 "bootstrap replicates of the data and measuring the out-of-bootstrap error"로 재고 seed 분산과 나란히 보고하지만, 두 분산원을 **한 재표집 절차로 합쳐 한 구간**을 내지는 않는다. Agarwal 외(2021, arXiv:2108.13264)는 과제를 층으로 고정하고 **실행(run)만** 재표집한다 — 1수준만 있는 경우다(초록 확인; 본문 문장은 추출 실패).

### 여전히 못 찾은 것과 양성 대조

- **세 요소를 동시에 갖춘 사례**: (i) 1수준이 교차검증 반복·분할, (ii) 2수준이 군집(환자·방전) 단위, (iii) 통계량이 기준선 대비 스킬 비율. 전문 검색 `"hierarchical bootstrap" "cross-validation" "folds" "subjects" "repeats" prediction`(13건), `"two-stage bootstrap" "trained models" "test" "patients"`(2건), `"pooled out-of-fold predictions" "cluster bootstrap"`(67건, 열어 본 것은 전부 1수준 군집 부트스트랩)에서 해당 없음. 가장 가까운 것은 MultiBERTs(seed × 사례)와 Goddard(시점 × 멤버, 스킬 점수).
- MRMC를 딥러닝 seed에 그대로 적용한 사례: `"MRMC" "deep learning" "random seeds" "training variability"` 0건.
- 양성 대조: 같은 경로에서 `"multi-bootstrap" seeds "test examples"` → MultiBERTs 1건 정확히 적중.

### 판정 변화의 근거와 주의

1차 판정 "결합의 명시적 선례 미발견"은 **틀렸다**. 같은 구조가 ML 평가(Sellam 2022, ICLR)와 기후 예측 검증(Goddard 2013)에 있고, 통계 이론은 MRMC(1992–2009)와 교차 배열 부트스트랩(McCullagh 2000, Owen 2007)에 정리돼 있다. 우리 것은 그 구조를 "교차검증 반복 × 방전 군집 × 스킬 비율"에 적용한 변형이다.

논문이 물려받아야 할 경고 두 가지:
1. 교차 2요인 부트스트랩은 정확하지 않고 분산을 **과대추정하는 쪽으로** 치우친다(McCullagh 2000; Owen 2007; Owen & Eckles 2012; Gallas 2009). 구간이 보수적이라는 뜻이므로 "0 제외" 판정에는 유리하지 않은 방향이다. 한 문장으로 적을 가치가 있다.
2. MultiBERTs의 정리(Theorem 1)는 seed가 독립 추출이라는 가정 위에 있다. 교차검증 반복은 학습 집합이 겹치므로 독립이 아니다 — Nadeau–Bengio 보정 표준오차를 나란히 보고하는 현재 구성이 이 틈을 메운다는 점을 명시하면 된다.

### 논문에 쓸 인용 문장 제안

- (한) "학습 반복과 방전을 함께 재표집하는 두 단계 부트스트랩은 seed와 시험 사례를 함께 재표집하는 Multi-Bootstrap(Sellam 외 2022) 및 시작 시점과 앙상블 멤버를 함께 재표집해 MSE 스킬 점수의 유의성을 판정하는 기후 예측 검증 절차(Goddard 외 2013)와 같은 구조이며, 여기서는 시험 단위를 방전 군집으로, 통계량을 기준선 대비 스킬 비율로 둔다."
- (En) "The two-level bootstrap, which resamples training repeats and discharges jointly, has the same structure as the Multi-Bootstrap over seeds and test examples (Sellam et al., 2022) and as the start-date/ensemble-member bootstrap used for mean-squared skill scores in decadal prediction (Goddard et al., 2013); here the test unit is the discharge cluster and the statistic is the skill ratio against a reference predictor."
- (한) "교차 배열의 부트스트랩은 정확하지 않으며 분산을 과대추정하는 방향으로 치우치므로(McCullagh 2000; Owen 2007), 보고한 구간은 보수적이다."
- (En) "Bootstraps of crossed arrays are not exact and err toward overestimating the variance (McCullagh, 2000; Owen, 2007), so the reported intervals are conservative."

---

## 항목 2. 실험 단위(방전) 군집 부트스트랩으로 구한 짝지은 스킬 구간 — 물리과학 ML, 특히 핵융합

### 새로 찾은 선례

| 문헌 | DOI / id | 분야 | 우리 방법과의 거리 | 근거 인용 |
|---|---|---|---|---|
| Keith, Nagpal, Rea, Tinguely, "Risk-Aware Framework Development for Disruption Prediction: Alcator C-Mod and DIII-D Survival Analysis", *J. Fusion Energy* 43 (2024) | 10.1007/s10894-024-00413-y | **토카막 붕괴 예측, 심사 논문** | **같은 발상(방전 단위 재표집, 모델 간 같은 재표본).** 차이: 50회 반복, 중앙값과 사분위 범위만 보고 — 짝지은 차이의 95 % 구간이나 검정은 아님. **[본문]** | "Shots in the test set were randomly sampled with replacement for 50 iterations to obtain the median and interquartile range for the distribution of calculated metrics. This sampling was seeded so each model's bootstrapped metrics were calculated on the same random selections of shots." |
| Koshizuka & Yaguchi, "Horizon-Aware Early Event Prediction for Tokamak Disruption Alarms" (Research Square preprint, 2026) | 10.21203/rs.3.rs-11399031/v1 | 토카막 붕괴 예측(DisruptionBench: DIII-D, C-Mod, EAST) | **같은 구조(우리의 1수준 없는 판).** 방전 단위 짝지은 부트스트랩 2,000회, 두 방법 차이의 95 % 백분위 구간, 0 제외 여부 표기. seed 분산이 구간에 안 들어간다는 점을 스스로 밝힘 — 우리 2수준 부트스트랩이 메우는 바로 그 틈. **[본문]** | "a paired shot-level bootstrap with 2,000 replicates. Each replicate resamples the test shots with replacement, separately within disruptive and non-disruptive shots … the same resampled shots are used for all methods and seeds … we report 95% percentile intervals for the difference between two methods. These intervals are conditional on the fixed split and training seeds and do not describe variation across data splits or devices" / "The shot-level bootstrap intervals are conditional on the trained seeds and do not include this variation." |
| Higginbottom, Ercan, Antony 외, "ModeCast: One-Shot Prediction of Linear MHD Eigenmodes and Growth Rates in Tokamaks" (preprint, 2026-10) | arXiv:2610.23844 | 토카막 MHD 서로게이트(시뮬레이션 자료) | 같은 구조(짝지은 사례 수준 부트스트랩 + 평형 단위 군집 부트스트랩), 단위가 방전이 아니라 시뮬레이션 평형. 학습 변동은 seed 퍼짐으로 따로 보고. **[본문]** | "All comparisons are paired on shared cases, with 95% bootstrap confidence intervals (CIs)" / "paired case-level bootstrap 95% CIs on medians; training variability is reported as seed spread over initializations" / "cluster bootstrap over equilibria" |
| Kumar, Clauser, Golfinopoulos, Rea 외, "Physics Attention Transformer Surrogate for Rapid Vertical Instability Growth Rate Prediction: Alcator C-Mod to SPARC" (preprint, 2026-08) | arXiv:2608.24785 | 토카막 서로게이트 | 같은 발상이나 **군집하지 않음** — 그리고 그 한계를 직접 적음. 방전 군집의 필요를 핵융합 저자가 스스로 지적한 문장으로 인용 가치가 있다. **[본문]** | "we computed 95% bootstrap confidence intervals using B = 1,000 equilibrium level resampling iterations … Because this bootstrap is not clustered by discharge or synthetic trajectory, the CIs should be read as conditional on the sampled equilibria and not as a full correction for intra shot autocorrelation." |
| Nakkina, Waller, Michoski 외, "The Fusion Equilibrium Challenge: Inferring Magnetic Geometry Without Magnetic Diagnostics" (preprint, 2026-09) | arXiv:2609.01750 | 토카막 벤치마크 제안 | 관련만 있음. 방전 단위 분할 + 순위표에 부트스트랩 불확도 요구. 재표집 단위는 명시 안 됨. **[본문]** | "requiring participants to report bootstrap confidence intervals alongside point predictions" / "all train/test splits are strictly shot-level" |
| Goddard 외 2013 (항목 1 표) | 10.1007/s00382-012-1481-2 | 기후 | 같은 구조: 실험 단위(시작 시점) 재표집으로 기준 예측 대비 MSSS의 유의성. **[본문]** | 항목 1 인용 참조 |
| Hamill, "Hypothesis Tests for Evaluating Numerical Precipitation Forecasts", *Weather and Forecasting* 14, 155–167 (1999) | 10.1175/1520-0434(1999)014<0155:HTFENP>2.0.CO;2 | 수치예보 검증 | **같은 발상.** 공간 상관을 피하려고 사례일(case day)을 단위로 묶고 두 예보를 짝지어 재표집 검정. "방전 = 사례일"로 읽으면 같은 논리. **[초록]** | "For each test, data from each competing forecast are collected into one sample for each case day to avoid problems with spatial correlation … the paired t test, the nonparametric Wilcoxon signed-rank test, and two resampling tests." |
| Geer, "Significance of changes in medium-range forecast scores", *Tellus A* 68, 30229 (2016) | 10.3402/tellusa.v68.30229 | 수치예보 검증 | 같은 발상(대조군 대비 짝지은 점수 차이의 유의성; 자기상관 때문에 단순 t 검정이 틀린다는 실증). **[초록]** | "With 1885 paired differences from about 2.5 yr of testing, an alternative significance test can be constructed that makes no statistical assumptions about the data … the naive application of Student's t-t[est] …" |
| DelSole & Tippett, "Comparing Forecast Skill", *Mon. Weather Rev.* 142, 4658–4678 (2014) | 10.1175/MWR-D-14-00045.1 | 예보 검증 | 관련(같은 사건에 대한 두 예보의 스킬 차이는 독립 표본 검정으로 다루면 안 됨). **[서지]** | — |

보조 확인: DisruptionBench(Spangher 외 2025, 10.1007/s10894-025-00495-2) 본문에는 "bootstrap"·"confidence interval" 문자열이 없다(Springer OA PDF 전문 추출로 확인) — 1차의 "CI 방식 미확인"을 "CI 없음"으로 고친다. TokaMark(arXiv:2602.10132) 본문에도 없다.

### 여전히 못 찾은 것과 양성 대조

- **핵융합에서 Nadeau–Bengio 보정 또는 corrected resampled t-test**: 전문 검색 `"tokamak" "corrected resampled t-test"` 0건; `"tokamak" "Nadeau"` 11건 전부 무관(동명 저자·물리 문헌).
- **방전 단위 반복 k-fold로 구간을 낸 핵융합 논문**: `"tokamak" "repeated k-fold"` 3건 — W7-X 서로게이트(Nucl. Fusion 2021, 10.1088/1741-4326/ac1a0d)와 W7-X 중성가스압(Phys. Plasmas 2026, 10.1063/5.0305811)이 걸리나 **[전문 색인]** 뿐이고 초록에는 구간 추정 언급이 없다. `"tokamak" "nested cross-validation" "discharges"` 1건 — Olofsson, Akçay, Sammuli, *J. Plasma Phys.* 88 (2022), 10.1017/S0022377822001003 **[전문 색인]**; 초록은 "Model selection … and calibration assessment procedures are detailed"라고만 한다(본문 PDF 추출 실패). 모델 선택용 중첩 교차검증으로 보이며 성능 구간용인지는 미확인.
- **핵융합 심사 논문에서 방전 군집 짝지은 차이의 95 % 구간**: 미발견. 심사 논문은 Keith 2024(사분위 범위)까지이고, 95 % 짝지은 구간은 2026년 preprint(Koshizuka & Yaguchi)뿐.
- `"tokamak" "shot-level bootstrap"`, `"tokamak" "bootstrap over shots"` 0건 (저자들이 다른 표현을 쓴다는 뜻 — 위 두 편은 `"sampled with replacement" "shots"`, `"paired bootstrap"`으로 찾았다).
- 양성 대조: `"tokamak" "disruption" "random forest" "k-fold"` → Montes 2019, Rea 2019, Zhu 2021 등 Nuclear Fusion 논문 정확히 적중(IOP 본문이 전문 색인에 들어 있음을 확인). `"tokamak" "test set" "sampled with replacement" "shots" "disruption"` → Keith 2024 적중.

### 논문에 쓸 인용 문장 제안

- (한) "방전을 재표집 단위로 삼는 부트스트랩은 붕괴 예측 문헌에 이미 쓰였고(Keith 외 2024), 군집하지 않은 부트스트랩 구간이 방전 내 자기상관을 바로잡지 못한다는 점도 지적돼 있다(Kumar 외 2026, preprint). 여기서는 이를 기준선 대비 짝지은 스킬 비율의 95 % 구간으로 쓰고, 학습 반복의 분산을 같은 구간에 넣는다."
- (En) "Bootstrapping with the discharge as the resampling unit has been used in disruption prediction (Keith et al., 2024), and it has been noted that an unclustered bootstrap does not correct for within-shot autocorrelation (Kumar et al., 2026, preprint). We use it for 95 % intervals on the paired skill ratio against a reference and carry training-repeat variance in the same interval."

---

## 항목 3. persistence·보간 대비 MSE 스킬 점수와 CRPS·PIT 보정 — 핵융합, 그리고 가장 가까운 이웃(우주기상·태양 예측)

### 새로 찾은 선례

| 문헌 | DOI / id | 분야 | 우리 방법과의 거리 | 근거 인용 |
|---|---|---|---|---|
| Gaytan-Villarreal, Xie, Pham, Sonker, Amendola, Cremonesi 외, "Real-Time Plasma State Prediction via FPGA-Accelerated Quantized Recurrent Probabilistic Neural Networks" (preprint, 2026-09) | arXiv:2609.23141 | **토카막(DIII-D) 플라즈마 상태 예측** | **같은 발상.** persistence가 짧은 예측 구간에서 매우 강하다는 것을 수치로 보이고 스킬을 persistence 대비 개선으로 정의. 지표는 설명분산(EV)이고 Murphy 꼴의 비율 점수는 아님. **[본문]** | "a persistence model that simply repeats s_s at every timestep scores 0.96 at t = 1, 0.55 at t = 25, and crosses zero only at t ≈ 99. Model skill is therefore measured as the improvement over this baseline, rather than by the EV value." |
| Rousseau, Jackson, Ordonez-Hurtado, Amorisco 외, "TokaMark: A Comprehensive Benchmark for MAST Tokamak Plasma Models" (preprint, 2026-02) | arXiv:2602.10132 | **토카막(MAST) 벤치마크** | 같은 발상(persistence와 평균을 표준 기준선으로 표에 넣음). 스킬 점수로 환산하지 않고 NRMSE를 나란히 둠. 불확도 구간 없음. **[본문]** | "two simple statistical models (Persistence and Mean)" / "the Persistence Model, which performs AR Forecasting for selected tasks by propagating initial conditions; and the Mean Model" |
| Abbate, Fable, Tardini, Fischer, Kolemen, *Nucl. Fusion* 65, 056014 (2025) — **이미 refs.bib에 있음**(`Abbate2025Combining`) | 10.1088/1741-4326/adc283 | 토카막 프로파일 예측 | 같은 발상(변화 없음 가정이 ML·물리 모형을 이기는 경우를 보고). 재확인만. **[초록]** | "both machine learning models extrapolated far-distribution and state-of-the-art 'physics-based' profile predictors fare worse than merely assuming plasma profiles do not change from their initial values." |
| "A data-driven model for stable long-horizon autoregressive prediction of plasma current and control-oriented boundary evolution in EAST", *Nucl. Fusion* (2026) | 10.1088/1741-4326/aeaa6c | 토카막(EAST) | **[전문 색인]** `"tokamak" "persistence baseline"`에 적중. 초록에는 언급 없음, IOP 본문 미확인. | — |
| Owens, Challen, Methven, Henley, "A 27 day persistence model of near-Earth solar wind conditions: A long lead-time forecast and a benchmark for dynamical models", *Space Weather* 11, 225–236 (2013) | 10.1002/swe.20040 | 우주기상(가장 가까운 플라즈마 이웃) | **같은 구조.** persistence를 기준 예측으로 세우고 스킬 점수로 수치 모형을 비교. **[초록]** | "persistence can provide a useful benchmark for more sophisticated forecast schemes … skill scores suggest persistence is out-performed by numerical models for almost all solar wind parameters." |
| Camporeale, "The Challenge of Machine Learning in Space Weather: Nowcasting and Forecasting", *Space Weather* 17, 1166–1207 (2019) | 10.1029/2018SW002061 (arXiv:1903.05192) | 우주기상 ML 총설 | **같은 구조**(신경망의 persistence 대비 스킬 점수 보고가 관행이며, 비교를 빠뜨린 논문을 결함으로 지적). 제목에 "nowcasting". **[본문, arXiv 판]** | "The skill score with respect to the persistence model was 22%." / "Unfortunately, a comparison against a persistence model (where the output one-hour ahead is simply taken as the value at current time) was not quantitatively performed." / "a persistence model yields an excellent performance in short-term predictions has been known for long." |
| Liemohn, McCollough, Jordanova, Ngwira 외, "Model Evaluation Guidelines for Geomagnetic Index Predictions", *Space Weather* 16, 2079–2102 (2018) | 10.1029/2018SW002067 | 우주기상 | 관련(지자기 지수 예측 모형의 표준 지표 묶음). **[초록]** | "These existing methods are synthesized into a baseline collection of metrics for benchmarking a new or updated geomagnetic index prediction model." |
| Marquez & Coimbra, "Proposed Metric for Evaluation of Solar Forecasting Models", *J. Sol. Energy Eng.* 135, 011016 (2013; 온라인 2012) | 10.1115/1.4007496 | 태양 일사량 예측 | 같은 구조(예측 오차를 변동성으로 나눈 비 — persistence 대비 스킬의 원형). **[초록]** | "the quality metric proposed here, which is defined as the ratio of solar uncertainty to solar variability … compare their performances with a persistence model." |
| Yang, Alessandrini, Antonanzas 외, "Verification of deterministic solar forecasts", *Solar Energy* 210, 20–37 (2020) | 10.1016/j.solener.2020.04.019 | 태양 예측 검증 합의문 | 같은 구조(RMSE 스킬 점수를 보편 보고 항목으로 권고; 기준은 기후값–persistence의 최적 볼록 결합). **[서지 + 2차 출처]** (초록이 OpenAlex에 없어 기관 저장소 스니펫으로 확인) | (스니펫) RMSE 스킬 점수를 "climatology–persistence" 결합 기준에 대해 보고하도록 권고 |
| Lauret, David, Pinson, "Verification of solar irradiance probabilistic forecasts", *Solar Energy* 194, 254–271 (2019) | 10.1016/j.solener.2019.10.041 | 태양 확률 예측 | 관련(CRPS·PIT/신뢰도 도표의 표준 묶음). **[서지]** | — |
| "Deterministic and probabilistic neural surrogates of global hybrid-Vlasov simulations", *Mach. Learn.: Sci. Technol.* (2026) | 10.1088/2632-2153/aead48 | 우주 플라즈마 서로게이트 | 관련(플라즈마 물리에서 CRPS를 학습 목적함수·보정 지표로 사용; 전문 색인에서 "persistence baseline"도 적중). **[초록]** | "a continuous ranked probability score objective is added to improve the calibration of the ensemble forecasts." |

### 여전히 못 찾은 것과 양성 대조

- **핵융합에서 Murphy 꼴 스킬 점수(1 − MSE/MSE_ref)를 persistence에 대해 보고**: `"tokamak" "skill score" "neural network"` 0건, `"tokamak" "skill score" "persistence"` 0건. persistence **기준선**은 2025–26년에 나타났으나(위 표) **스킬 점수** 형식은 미발견.
- **핵융합에서 보간(interpolation)을 기준 예측으로 둔 스킬**: 전용 검색 두 개 모두 해당 없음(0건, 4건 무관). 측정 사이를 채우는 과제 자체가 드물기 때문으로 보인다.
- **핵융합에서 CRPS**: 여전히 Sorokin 외 arXiv:2610.24525 한 편(1차와 동일). `"tokamak" "continuous ranked probability score"` 6건 중 토카막 예측 논문은 없음(나머지는 범용 ML 논문이 토카막을 예로 든 경우).
- **핵융합에서 PIT**: `"tokamak" "probability integral transform"` 2건 — *Reliab. Eng. Syst. Saf.* (2026), 10.1016/j.ress.2026.113535 (보조가열 서로게이트 UQ) **[전문 색인]**, 본문 미확인. 예측기 보정 진단으로 쓴 것인지 알 수 없음.
- 양성 대조: `"solar wind" "skill score" "persistence" "neural network"` → 94건(Space Weather, JSWSC, ApJ 다수) — 같은 경로가 우주기상에서는 이 관행을 대량으로 찾는다. 즉 핵융합의 0건은 경로 탓이 아니다.

### 논문에 쓸 인용 문장 제안

- (한) "변화 없음(persistence)을 기준 예측으로 두고 그 대비 스킬로 모형을 말하는 것은 우주기상과 태양 예측에서 표준이며(Owens 외 2013; Camporeale 2019; Yang 외 2020), 토카막 예측에서도 persistence가 강한 기준선이라는 보고가 최근 나왔다(Abbate 외 2025; Rousseau 외 2026, preprint). 이 논문은 같은 기준을 Murphy(1988)의 MSE 스킬 점수 형식으로 쓰고 보간 기준을 더한다."
- (En) "Reporting skill against a no-change (persistence) reference is standard in space-weather and solar forecasting (Owens et al., 2013; Camporeale, 2019; Yang et al., 2020), and persistence has recently been reported as a strong baseline for tokamak prediction (Abbate et al., 2025; Rousseau et al., 2026, preprint). We use the same reference in the mean-squared-error skill-score form of Murphy (1988) and add an interpolation reference."

---

## 항목 4. 사전등록(또는 실행 전 고정한 분석 계획·예측 장부) — 물리과학 ML과 플라즈마 물리; 핵융합의 블라인드 분석

### 새로 찾은 선례

| 문헌 | DOI / id | 분야 | 우리 방법과의 거리 | 근거 인용 |
|---|---|---|---|---|
| Vakhshouri, "A trust layer for fast pedestal-stability surrogates: calibrated error bars, out-of-distribution alarms and few-shot recalibration" (preprint, 2026-11-04) | arXiv:2611.06597 | **토카막 페데스탈 안정성 서로게이트** | **같은 구조.** 사전등록한 가설로 검정하고, 성립하지 않은 가설도 보고. 1차 보고서는 이 논문을 conformal coverage 사례로만 실었고 사전등록은 놓쳤다. 단독 저자 preprint, 6일 전 게시. **[본문]** | "we test it with pre-registered hypotheses" / "Not every pre-registered hypothesis held, and we report those results and the limits of all of them." / "A pre-registered evaluation of the trust layer on the public KARHU dataset (38,879 equilibria), under a controlled plasma-shape shift, with all outcomes reported." |
| Higginbottom 외, ModeCast (preprint, 2026-10) | arXiv:2610.23844 | 토카막 MHD 서로게이트 | 같은 발상(채점 전에 보고할 수치를 고정). **[본문]** | "These are the eight values pre-registered before scoring." |
| "Graph-SERO: A Temporal Graph Dynamics Replay Protocol for Event-Relative Tokamak Disruption Early-Warning Validation" (Zenodo 기술보고서, 2026) | 10.5281/zenodo.20743199 | 토카막 붕괴 예측 | 같은 발상(결과 없이 프로토콜만 먼저 공개). 심사 없음, 결과 없음 — **인용 대상이 아니라 "관행이 생기고 있다"는 표지**로만. **[초록]** | "This release is a preregistration-style protocol and paper draft. It does not report completed EAST, JET, MAST-U, or DIII-D validation results" |
| Hatfield, Gaffney, Anderson 외, "The data-driven future of high-energy-density physics", *Nature* 593, 351–361 (2021) | 10.1038/s41586-021-03382-w (arXiv:2111.11310) | 고에너지밀도 물리(관성 핵융합 포함) | 같은 발상의 **권고**(실행 사례 아님). 플라즈마 물리 쪽에서 블라인드 분석 도입을 명시적으로 요구한 심사 문헌. **[본문, arXiv 판]** | "using blinding methods are likely to become more necessary, where researchers deliberately hide some aspect of the labelling of the data from themselves to prevent subconscious bias or p-hacking. Using a blinding protocol does also present dangers and challenges; the complexity and bespoke properties common in many laser-plasma experiments make blinding difficult to implement." |
| Spears, Brandon, Casey, Field 외, "Predicting fusion ignition at the National Ignition Facility with physics-informed deep learning", *Science* 389, 727–731 (2025) | 10.1126/science.adm8201 | 관성 핵융합 ML | **같은 발상(예측을 실험 전에 고정하고 사후에 채점).** 예측 장부의 한 줄짜리 원형. 분석 계획 전체의 사전등록은 아님. **[초록]** | "Prior to the experiment, a generative machine learning model … was used to predict, with a probability greater than 70%, that ignition was the most likely outcome for this shot." |
| Rodriguez-Fernandez, White, Howard, Grierson 외, "Predict-first experiments and modeling of perturbative cold pulses in the DIII-D tokamak", *Phys. Plasmas* 26, 062503 (2019) | 10.1063/1.5096800 | **토카막 수송 실험** | 같은 발상·다른 수준. 모형 예측을 실험 전에 내고 실험으로 판정하는 "predict-first"는 자기 핵융합의 기존 관행. 통계 분석 계획의 동결이 아니라 물리 예측의 선행. **[초록]** | "The experimental results are consistent with predict-first simulations of heat transport enabled by the Trapped Gyro-Landau-Fluid transport model." |
| Kapoor, Cantrell, Peng, Pham 외, "REFORMS: Consensus-based Recommendations for Machine-learning-based Science", *Science Advances* 10, eadk3452 (2024); Kapoor & Narayanan, "Leakage and the reproducibility crisis in machine-learning-based science", *Patterns* 4, 100804 (2023) | 10.1126/sciadv.adk3452; 10.1016/j.patter.2023.100804 | ML 기반 과학 전반 | 관련(ML을 쓰는 과학 논문의 보고 기준·누설 분류). 사전등록을 직접 요구하는지는 본문 미확인. **[서지]** | — |

`"predict-first" "tokamak"` 전문 검색은 126건이다 — 표현 자체가 자기 핵융합에서 널리 쓰인다.

### 여전히 못 찾은 것과 양성 대조

- **심사를 통과한 핵융합·플라즈마 ML 논문에서의 사전등록**: 미발견. `"tokamak" "preregistered"` 12건, `"tokamak" "pre-registered"` 67건을 훑었으나 arXiv preprint 2편(위 표) 외에는 Zenodo·Harvard Dataverse의 비심사 자기 출판물과 무관 문헌이다.
- **핵융합 실험 분석에서의 블라인드 분석 실행 사례**: `"tokamak" "blind analysis"`, `"tokamak" "blinded analysis"` 각 3건 — Hatfield 2021(권고) 외 해당 없음. `"tokamak" "ITPA" "blind" "prediction" "prior to the experiment"` 0건.
- **신뢰도(확률)를 붙인 예측 장부를 held/missed로 채점**: 미발견(Spears 2025가 단일 예측으로 가장 가까움).
- 양성 대조: 같은 전문 검색 경로에서 `"preregistration" "machine learning" … "analysis plan"` → 다수 적중(2026년 ML 응용 preprint, 의학 AI 지침 등), `"predict-first" "tokamak"` → Nuclear Fusion·Phys. Plasmas 논문 적중.

### 논문에 쓸 인용 문장 제안

- (한) "분석 계획과 예측을 실행 전에 고정하는 절차는 등록보고서(Nosek & Lakens 2014)와 블라인드 분석(Klein & Roodman 2005)에서 왔고, 고에너지밀도 플라즈마 물리에서도 그 도입이 권고되었다(Hatfield 외 2021). 자기 핵융합에는 모형 예측을 실험보다 먼저 내는 predict-first 관행이 있으며(Rodriguez-Fernandez 외 2019), 관성 핵융합에서는 기계학습 예측을 실험 전에 고정한 사례가 보고되었다(Spears 외 2025). 통계 분석 계획 전체를 사전등록한 핵융합 진단 ML의 심사 논문은 찾지 못했다."
- (En) "Fixing the analysis plan and the predictions before the runs follows registered reports (Nosek & Lakens, 2014) and blind analysis (Klein & Roodman, 2005), whose adoption has been recommended for high-energy-density plasma physics (Hatfield et al., 2021). Magnetic fusion has a predict-first practice in which model predictions precede the experiment (Rodriguez-Fernandez et al., 2019), and a machine-learning prediction fixed before the shot has been reported in inertial fusion (Spears et al., 2025). We found no peer-reviewed fusion-diagnostic ML paper that pre-registers the full statistical analysis plan."

---

## 항목 5. 자료 품질 제외 유무에 따른 두 분석 모집단을 공동 1차로 보고

### 새로 찾은 선례

| 문헌 | DOI / 출처 | 분야 | 우리 방법과의 거리 | 근거 인용 |
|---|---|---|---|---|
| CPMP(현 EMA), "Points to consider on switching between superiority and non-inferiority", CPMP/EWP/482/99 (2000) | DOI 없음. EMA PDF: `ema.europa.eu/en/documents/scientific-guideline/points-consider-switching-between-superiority-non-inferiority_en.pdf` | 임상시험 규제 지침 | **같은 구조.** 제외 없는 집합(full analysis set, ITT)과 프로토콜 위반을 뺀 집합(per-protocol)을 **동등한 비중**으로 두고 두 결론이 일치해야 견고하다고 규정. 우리 "컷/포함 공동 1차"의 직접 선례. 이 지침은 EMA의 새 비열등성 지침으로 대체되는 중(EMA concept paper, 검색 스니펫)이라는 점을 함께 적어야 한다. **[본문]** | "In a non-inferiority trial, the full analysis set and the PP analysis set have equal importance and their use should lead to similar conclusions for a robust interpretation." |
| Piaggio, Elbourne, Pocock, Evans, Altman, "Reporting of Noninferiority and Equivalence Randomized Trials: Extension of the CONSORT 2010 Statement", *JAMA* 308, 2594–2604 (2012) | 10.1001/jama.2012.87802 | 임상시험 보고 기준 | 같은 구조의 **보고 규칙**(비열등성 시험은 ITT와 per-protocol 두 분석을 모두 보고)으로 알려져 있으나 초록에는 그 문장이 없다 — **[초록]** 수준에서는 "보고 기준 문서"라는 것만 확인. 해당 항목 문구는 본문 확인 뒤 인용. | (초록) "an updated extension of the CONSORT checklist for reporting noninferiority and equivalence trials" |
| Matilde Sanchez & Chen, "Choosing the analysis population in non-inferiority studies: per protocol or intent-to-treat", *Stat. Med.* 25, 1169–1181 (2006) | 10.1002/sim.2244 | 임상통계 | 같은 구조 + **경고.** 두 모집단 모두에서 보이라는 요건이 타당성을 보장하지는 않으며, 어느 쪽이 보수적인지는 제외·결측의 유형에 달려 있다 — 우리의 fit-failure 컷이 MNAR일 수 있다는 논점과 맞닿음. **[초록]** | "The requirement that non-inferiority be shown for both PP and ITT populations does not necessarily guarantee the validity of a non-inferiority conclusion … conservatism or anticonservatism of the PP or ITT analysis depends on many factors, including the type of protocol deviation and missingness" |
| ICH E9(R1), "Addendum on Estimands and Sensitivity Analysis in Clinical Trials" (Step 4, 2019-11) | DOI 없음. `database.ich.org/sites/default/files/E9-R1_Step4_Guideline_2019_1203.pdf` | 임상시험 국제 지침 | **같은 발상·다른 언어.** 모집단 정의가 다르면 추정 대상(estimand)이 다르다. 한 연구가 여러 estimand를 가질 수 있고 그때 다중성을 다뤄야 한다. 또한 "민감도 분석"(같은 estimand의 가정 점검)과 "보조 분석"을 구분 — 컷/포함은 민감도 분석이 아니라 **서로 다른 두 estimand**라는 우리 입장의 근거. **[본문]** | "a trial is likely to have multiple objectives translated into multiple estimands, each associated with statistical testing and estimation. The multiplicity issues arising should be addressed." / "Distinct from sensitivity analysis, where investigations are conducted with the intent of exploring robustness of departures from assumptions, other analyses … can be termed 'supplementary analysis'" |
| Thabane, Mbuagbaw, Zhang, Samaan 외, "A tutorial on sensitivity analyses in clinical trials: the what, why, when and how", *BMC Med. Res. Methodol.* 13, 92 (2013) | 10.1186/1471-2288-13-92 | 임상통계 | 관련(이상값·프로토콜 위반 처리에 따른 결론의 견고성을 계획해 보고). 공동 1차가 아니라 1차 + 민감도 구도. **[초록]** | "assess the impact, effect or influence of key assumptions or variations—such as different methods of analysis, definitions of outcomes, protocol deviations, missing data, and outliers—on the overall conclusions of a study." |
| Northcutt, Athalye, Mueller, "Pervasive Label Errors in Test Sets Destabilize Machine Learning Benchmarks", NeurIPS 2021 Datasets and Benchmarks | arXiv:2103.14749 | ML 벤치마크 | **같은 발상(ML 쪽).** 원래 시험 집합과 오류를 바로잡은 시험 집합에서 모델 순위가 바뀐다 — 품질 제외 유무가 결론을 뒤집을 수 있으므로 둘 다 봐야 한다는 실증. **[초록]** | "judging models over correctly labeled test sets may be more useful … on ImageNet with corrected labels: ResNet-18 outperforms ResNet-50 if the prevalence of originally mislabeled test examples increases by just 6%." |
| Ferro, "Measuring forecast performance in the presence of observation error", *Q. J. R. Meteorol. Soc.* 143, 2665–2676 (2017) | 10.1002/qj.3115 | 예보 검증 | 관련(검증 관측의 품질이 점수와 순위를 바꾼다). 해법은 두 모집단이 아니라 오차 보정 채점규칙. **[초록]** | "proper scoring rules favour good forecasts of observations rather than of truth and yield scores that vary with the quality of the observations." |

Lerch 외(2017)는 1차에 이미 있으므로 생략한다. 역할 분담은 이렇다: Lerch는 "극단 사건만 골라 평가하면 안 된다"(포함 모집단을 버리지 말아야 할 이유), 임상시험 지침은 "두 모집단을 동등 비중으로 두고 일치를 요구한다"(공동 1차의 형식), ICH E9(R1)은 "두 모집단은 두 estimand다"(해석).

### 여전히 못 찾은 것과 양성 대조

- **예측·예보 검증 문헌에서 품질 제외 유무 두 모집단을 "공동 1차"로 명명한 사례**: 미발견. 전용 전문 검색은 돌리지 않았고(조사 한계 참조) WebSearch 수준에서만 확인.
- **핵융합 ML에서**: 미발견(전용 검색 없음 — 핵융합 ML에 "1차 분석 모집단" 개념 자체가 없다).
- 양성 대조: EMA 지침 PDF와 ICH E9(R1) PDF를 직접 받아 문장을 대조했다(위 인용은 추출 본문 그대로).

### 논문에 쓸 인용 문장 제안

- (한) "적합 실패 행을 뺀 모집단과 포함한 모집단을 공동 1차로 둔 것은 비열등성 임상시험에서 전체 분석 집합과 프로토콜 순응 집합을 동등한 비중으로 두고 두 결론의 일치를 요구하는 규정(CPMP/EWP/482/99)과 같은 형식이다. 두 모집단은 서로 다른 추정 대상을 정의하므로(ICH E9(R1)) 한쪽을 다른 쪽의 민감도 분석으로 읽지 않는다."
- (En) "Treating the populations with and without fit-failure rows as co-primary follows the form used in non-inferiority trials, where the full analysis set and the per-protocol set carry equal importance and are required to agree (CPMP/EWP/482/99). The two populations define different estimands (ICH E9(R1)), so neither is read as a sensitivity analysis of the other."

---

## 종합 표

| 항목 | 1차 판정 | 2차 판정 | 바뀐 이유 |
|---|---|---|---|
| 1. 학습 반복 × 시험 단위 2수준 부트스트랩 | 차용 + **결합은 선례 없음**(신규 주장은 말 것) | **같은 구조의 선례 있음** — ML 평가(Sellam 외 2022 Multi-Bootstrap; Knopov & Shvai 2026 preprint), 기후 예측(Goddard 외 2013, 스킬 점수까지 같음), 이론(MRMC: Gallas 2009; 교차 배열: McCullagh 2000, Owen 2007). 남은 차이는 "교차검증 반복 × 군집 × 비율"이라는 조합뿐 | 1차는 신경과학 계층 부트스트랩만 봤고 NLP 평가·기후 검증·MRMC·교차 배열 문헌을 안 봤다. 전문 검색으로 Multi-Bootstrap을, 본문 추출로 Goddard 부록을 확인 |
| 2. 방전 군집 짝지은 부트스트랩 구간(핵융합) | 핵융합 **미발견** | **핵융합 선례 있음(약한 형태)** — 심사 논문 1편(Keith 외 2024: 방전 재표집, 사분위 범위), preprint에서 95 % 짝지은 구간(Koshizuka & Yaguchi 2026), 군집 부트스트랩(ModeCast 2026), 군집 누락의 한계 명시(Kumar 외 2026). Nadeau–Bengio 보정·반복 k-fold 구간은 여전히 미발견 | 1차는 arXiv 초록 검색만 했다. 이번에는 OpenAlex 전문 검색(본문 문자열)과 PDF 대조. 저자들이 "shot-level bootstrap"이 아니라 "sampled with replacement"·"paired bootstrap"이라 써서 초록 검색에 안 걸렸다 |
| 3. persistence·보간 MSE 스킬 + CRPS·PIT(핵융합) | CRPS preprint 1건, persistence 스킬 **미발견** | persistence **기준선**은 핵융합에 있음(Abbate 2025; TokaMark 2026; Gaytan-Villarreal 2026 — "skill = persistence 대비 개선"). Murphy 꼴 **스킬 점수**·보간 기준·PIT는 여전히 미발견. 우주기상·태양 예측에서는 표준(Owens 2013; Camporeale 2019; Yang 2020) | 전문 검색으로 본문 속 persistence 기준선을 찾음. 이웃 분야를 새로 조사 |
| 4. 사전등록·블라인드(핵융합·물리과학 ML) | 핵융합 **미발견** | 핵융합 ML **preprint 2편에서 실행**(Vakhshouri 2026; ModeCast 2026), 플라즈마 물리의 블라인드 **권고**(Hatfield 외 2021, Nature), 실험 전 ML 예측 고정(Spears 외 2025, Science), 자기 핵융합의 predict-first 관행(Rodriguez-Fernandez 외 2019). **심사 통과한 핵융합 ML 사전등록은 여전히 미발견** | 전문 검색 `"tokamak" "pre-registered"`가 1차의 초록 검색에서 안 보이던 본문 용례를 드러냄. Vakhshouri는 1차에 있었으나 사전등록 사실을 놓침 |
| 5. 두 모집단 공동 1차 | 신규에 가까움(근거는 Lerch 2017) | **형식의 선례 있음** — 비열등성 임상시험의 FAS·PP 동등 비중(CPMP/EWP/482/99, 문장 확인), 두 모집단 = 두 estimand(ICH E9(R1)), 한계 경고(Matilde Sanchez & Chen 2006), ML 쪽 실증(Northcutt 2021). 예측 검증·핵융합에서 "공동 1차" 명명은 미발견 | 1차는 임상시험 문헌을 안 봤다. 지침 원문 PDF로 문장 확인 |

한 줄 요약: 다섯 항목 모두 "선례 없음"에서 "형식의 선례는 다른 분야에(항목 1·5) 또는 핵융합 안에도 약한 형태로(항목 2·3·4) 있음"으로 바뀐다. 논문에서 "선례를 찾지 못했다"고 쓸 수 있는 범위는 다음으로 줄어든다 — (a) 교차검증 반복 × 방전 군집 × 스킬 비율의 조합, (b) 핵융합에서 Nadeau–Bengio 보정 구간, (c) 핵융합에서 persistence·보간 대비 MSE 스킬 **점수**와 PIT, (d) **심사를 통과한** 핵융합 ML 논문의 사전등록, (e) 핵융합에서 두 모집단 공동 1차.

## 조사 한계

- **심사 전 원고 의존.** 항목 2·3·4의 핵융합 선례 대부분이 2026년 preprint다(arXiv:2610.23844, 2608.24785, 2609.23141, 2611.06597, 2602.10132; Research Square 10.21203/rs.3.rs-11399031/v1). 게시 후 수정·철회될 수 있고, 논문 본문에서는 "preprint"로 표시해야 한다. 심사 논문 선례는 Keith 2024(항목 2), Abbate 2025(항목 3), Hatfield 2021·Spears 2025·Rodriguez-Fernandez 2019(항목 4)다.
- **본문을 못 연 것.** IOP(Nuclear Fusion) 본문은 차단. `10.1088/1741-4326/aeaa6c`(persistence baseline), `10.1088/1741-4326/ac1a0d`·`10.1063/5.0305811`(repeated k-fold), `10.1017/S0022377822001003`(nested cross-validation), `10.1016/j.ress.2026.113535`(PIT)는 전문 색인 적중만 있고 용도를 확인하지 못했다. 이 가운데 하나가 방전 단위 반복 교차검증 구간일 가능성은 남는다.
- **초록·서지만 확인한 것.** Beiden 2000, Chen 2012(2차 출처), Yang 2020(2차 출처), Piaggio 2012(초록에 해당 문장 없음), DelSole & Tippett 2014, Lauret 2019, Davezies 2021, McCullagh 2000, Kapoor 2023·2024. 표에 표시해 두었다. 논문에 인용하려면 본문 확인이 먼저다.
- **확인하지 않고 뺀 것.** GraphCast(Lam 외 2023)의 유의성 검정 절차는 arXiv 판에서 해당 문장을 찾지 못해 표에 넣지 않았다. Agarwal 2021의 "stratified bootstrap" 문장도 PDF 추출에 실패해 초록 수준으로만 적었다.
- **검색하지 않은 범위.** 항목 5의 예측 검증·핵융합 쪽은 전용 전문 검색을 돌리지 않았다. 항목 1의 일반화가능도 이론은 Tong & Brennan 한 편만 확인했고(Brennan 2001 교과서 6장 "Variability of Statistics in Generalizability Theory", 10.1007/978-1-4757-3456-0_6 은 서지만), 지진학·천문학·가속기 물리의 군집 부트스트랩 용례(항목 2)는 조사하지 않았다. Semantic Scholar는 쓰지 않았다.
- **전문 검색의 성질.** OpenAlex `fulltext.search`는 색인된 본문에만 닿고, 따옴표 구절이 정확히 일치해야 한다. 저자가 다른 표현을 쓰면 놓친다(항목 2에서 실제로 "shot-level bootstrap" 0건, "sampled with replacement" 적중). "미발견"은 "이 표현들로는 안 걸렸다"는 뜻이다.
- **Zenodo 자기 출판물.** `"tokamak" "pre-registered"` 67건의 다수는 심사 없는 Zenodo·Dataverse 문서였다. 선례로 세지 않았다.
