# 문헌 점검 2차: 해석 절차·사전 예측·강건 필터·나우캐스팅 용어·보정된 head

작성일 2026-11-10. 1차 보고서(`docs/lit_2026-11-09/lit_interpretation.md`, `lit_problem_models.md`)가 "미발견"으로 남긴 다섯 항목을 다시 뒤진 결과다. 1차에 이미 실린 문헌은 반복하지 않고, 판정을 바꾸는 데 필요한 경우에만 다시 언급한다.

## 0. 조사 경로와 표기

- 경로: OpenAlex API(`fulltext.search` 필터 포함, 이번에는 429 없이 동작. 키 없는 공용 예산이라 약 60회로 제한), Crossref API(DOI 해석·서지 검색), arXiv export API(ID 조회·초록 검색), 웹 검색, arXiv PDF 전문 내려받아 문자열 검색(PyMuPDF).
- 검증 표기
  - **✔ DOI**: Crossref `works/{doi}`에서 제목·저자·연도가 일치함을 이번 조사에서 직접 확인.
  - **✔ arXiv**: arXiv API에서 ID로 초록을 받아 확인.
  - **[전문]**: PDF 본문을 열어 인용 문장을 확인. **[초록]**: 초록(OpenAlex·arXiv)만 확인. **[미확인]**: 서지만 확인했고 내용은 2차 자료(슬라이드·검색 요약)나 기억에 의존. 인용 전 본문 확인 필요.
- 거리 등급: **A** = 같은 절차 / **B** = 같은 생각이나 사전에 고정한 문턱이 없음(또는 대상이 다름) / **C** = 관련 문헌(계보·대조용).
- "미발견"은 위 경로에서 나오지 않았다는 뜻이다. 각 항목에 그 검색 경로가 실제로 동작함을 보이는 양성 대조를 붙였다.

---

## 항목 1. 학습된 망을 읽어 작은 닫힌 식으로 적고, 같은 프로토콜로 처음부터 다시 학습해, 사전에 고정한 문턱으로 기술 보존을 검정하기

### 1-1. 새로 찾은 선례

| 문헌 | DOI / ID | 분야 | 거리 | 근거 인용 |
|---|---|---|---|---|
| Maheswaranathan N., Sussillo D. (2020) "How recurrent networks implement contextual processing in sentiment analysis", ICML 2020 | ✔ arXiv 2004.08013 [전문] | NLP, RNN 역공학 | **A에 가장 가까움** (문턱 사전 고정 없음, 분류 과제) | 초록: "Using the insights learned from the RNNs we improve baseline Bag-of-Words models with simple extensions that incorporate contextual modification, recovering greater than 90% of the RNN's performance increase over the baseline." 본문 기여 목록: "a mathematical model that describes this behavior, and a demonstration that simple and interpretable baseline models, augmented to incorporate this new understanding, recover almost all of the accuracy of the nonlinear RNNs." 9절 제목 "Augmented baseline models with insights from RNNs recover performance". 표 1에서 가장 큰 기준 모형 95.63 % 대 최고 RNN 95.84 %, "90% of the difference". |
| Schaeffer R., Khona M., Meshulam L., IBL, Fiete I. (2020) "Reverse-engineering recurrent neural network solutions to a hierarchical inference task for mice", NeurIPS 2020 | ✔ DOI 10.1101/2020.06.09.142745 (bioRxiv) [초록] | 계산신경과학 | B (저차원 모형으로 증류, 이상 베이즈 기준과 비교. 보존 비율 문턱 없음) | "RNNs learn behavior that is quantitatively similar to ideal Bayesian baselines … we perform model compression through a novel form of knowledge distillation on hidden representations – Representations and Dynamics Distillation (RADD) – to reduce the RNN dynamics to a low-dimensional, highly interpretable model." |
| Mikulik V., Delétang G., McGrath T., Genewein T., Martic M. 외 (2020) "Meta-trained agents implement Bayes-optimal agents", NeurIPS 2020 | ✔ arXiv 2010.11223 [초록] | 메타학습 | B (학습된 순환 망과 닫힌 형태의 베이즈 최적 필터가 서로를 모사하는지 검정. 재학습·문턱 없음) | "we show that meta-learned and Bayes-optimal agents not only behave alike, but they even share a similar computational structure, in the sense that one agent system can approximately simulate the other." |
| Valente A., Pillow J.W., Ostojic S. (2022) "Extracting computational mechanisms from neural data using low-rank RNNs", NeurIPS 35, 24072–24086 | ✔ DOI 10.52202/068431-1748 [초록] | 계산신경과학 | B (블랙박스 RNN의 궤적에 저계수 RNN을 다시 맞춰 기전 모형을 얻음) | "we reverse-engineer 'black-box' vanilla RNNs trained to perform cognitive tasks" · "LINT yields a mechanistic model of latent dynamics … and verifiable predictions for inactivations". |
| Dubreuil A., Valente A., Beiran M., Mastrogiuseppe F., Ostojic S. (2022) "The role of population structure in computations through neural dynamics", *Nat. Neurosci.* 25, 783–794 | ✔ DOI 10.1038/s41593-022-01088-4 [미확인: 초록 미제공] | 계산신경과학 | C (학습된 RNN을 저차원 동역학계로 환원) | 초록을 받지 못했다. 인용 전 본문 확인 필요. |
| Weinhardt D., Plomecka M.B., Tezcan I.M., Eckstein M., Musslick S. (2025) "Automated Discovery of Sparse and Interpretable Cognitive Equations" (SPICE), PsyArXiv | ✔ DOI 10.31234/osf.io/v86q5_v1 [초록] | 인지과학 | B (RNN → 희소 식 추출. 식이 기존 모형보다 낫다고 보고. 망 대비 보존 문턱 없음) | "SPICE fits recurrent neural networks to capture latent cognitive dynamics and then applies sparse equation discovery to extract concise mathematical expressions describing those dynamics." |
| Miller K.J., Eckstein M.K., Botvinick M., Kurth-Nelson Z. (2023) "Cognitive Model Discovery via Disentangled RNNs", NeurIPS 36, 61377–61394 | ✔ DOI 10.52202/075280-2682 [초록] | 인지과학 | C (식이 아니라 해석 가능한 희소 RNN 자체가 산출물) | "When fitting synthetic behavioral data from known cognitive models, our method recovers the underlying form of those models." |
| Ji-An L., Benna M.K., Mattar M.G. (2025) "Discovering cognitive strategies with tiny recurrent neural networks", *Nature* 644, 993–1001 | ✔ DOI 10.1038/s41586-025-09142-4 [초록] | 인지과학 | C (1–4 유닛 RNN이 큰 망과 같은 예측력을 가짐을 보이고 동역학계로 읽음) | "neural networks with just 1-4 units often outperform classical cognitive models and match larger neural networks … we then interpret the trained networks using dynamical systems concepts". |
| Michaud E.J., Liao I., Lad V., Liu Z., Mudide A. 외 (2024) "Opening the AI black box: program synthesis via mechanistic interpretability" (MIPS) | ✔ arXiv 2402.05110 [초록] | 기계적 해석 | B (학습된 RNN → 유한상태기계 → 기호 회귀 → 파이썬 코드. 성공 기준은 과제 정답 여부) | "auto-distilling the learned algorithm into Python code … MIPS uses an integer autoencoder to convert the RNN into a finite state machine, then applies Boolean or integer symbolic regression to capture the learned algorithm." |
| Nanda N., Chan L., Lieberum T., Smith J., Steinhardt J. (2023) "Progress measures for grokking via mechanistic interpretability", ICLR 2023 | ✔ arXiv 2301.05217 [초록] | 기계적 해석 | B (망의 알고리즘을 닫힌 식으로 완전 역공학, 절제로 확인. 재학습 없음) | "We fully reverse engineer the algorithm learned by these networks, which uses discrete Fourier transforms and trigonometric identities … We confirm the algorithm by analyzing the activations and weights and by performing ablations in Fourier space." |
| Weiss G., Goldberg Y., Yahav E. (2018) "Extracting Automata from Recurrent Neural Networks Using Queries and Counterexamples", ICML 2018 | ✔ arXiv 1711.09576 [초록] | 형식언어·RNN | C (RNN → 유한 오토마톤 추출, 일치도 측정) | "extract a deterministic finite automaton describing the state dynamics of a given trained RNN". |
| Wang K., Variengien A., Conmy A., Shlegeris B., Steinhardt J. (2022) "Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 small" | ✔ arXiv 2211.00593 [초록] | 기계적 해석 | C (설명의 정량 기준 세 가지를 명시) | "We evaluate the reliability of our explanation using three quantitative criteria — faithfulness, completeness and minimality. Though these criteria support our explanation, they also point to remaining gaps in our understanding." |
| Chan L. 외 (2022) "Causal scrubbing: a method for rigorously testing interpretability hypotheses" (Redwood Research, Alignment Forum 게시물, DOI 없음) | DOI 없음 [미확인: 2차 요약] | 기계적 해석 | B (가설이 설명하는 손실의 비율 "loss recovered"를 정의. 문턱 관행 없음) | 검색 요약 기준: 무작위화 손실을 0, 원 모형 손실을 100 %로 둔 회복 비율. 저자들이 "비율이 100 %를 넘거나 음수가 될 수 있다"고 명시. 학술지 판이 없으므로 논문 인용에는 부적합, 계보 설명용. |
| Eshbaugh J. (2025) "'Faithful to What?' On the Limits of Fidelity-Based Explanations" | ✔ arXiv 2506.12176 [초록] | XAI | **B, 우리 지표 선택의 직접 근거** | "surrogates can achieve high fidelity to a neural network while failing to recover the predictive gains that distinguish the network from simpler models. In several cases, high-fidelity surrogates underperform even linear baselines trained directly on the data." |
| Tan S., Caruana R., Hooker G., Lou Y. (2018) "Distill-and-Compare: Auditing Black-Box Models Using Transparent Model Distillation", AIES 2018, 303–310 | ✔ DOI 10.1145/3278721.3278725, ✔ arXiv 1710.06169 [초록] | 해석 가능 ML | B (증류한 학생 모형과, 정답으로 직접 학습한 같은 형태의 모형을 나란히 비교) | "We compare the student model trained with distillation to a second un-distilled transparent model trained on ground-truth outcomes, and use differences between the two models to gain insight into the black-box model." |
| Andrews R., Diederich J., Tickle A.B. (1995) "Survey and critique of techniques for extracting rules from trained artificial neural networks", *Knowledge-Based Systems* 8, 373–389 | ✔ DOI 10.1016/0950-7051(96)81920-4 [미확인: 초록 미열람] | 규칙 추출 | C (추출 모형 평가 기준으로 fidelity를 정식화한 출처로 통용) | 본문 미열람. "fidelity" 정의 출처로 인용하려면 본문 확인 필요. Craven & Shavlik (1994) "Using Sampling and Queries to Extract Rules from Trained Neural Networks", ICML 1994, ✔ DOI 10.1016/b978-1-55860-335-6.50013-1 도 같은 계보. TREPAN(Craven & Shavlik, NIPS 8, 1996)은 DOI가 없어 검증하지 못했다. |
| Beucler T., Grundner A., Shamekh S., Ukkonen P., Chantry M. 외 (2024) "Distilling Machine Learning's Added Value: Pareto Fronts in Atmospheric Applications" | ✔ arXiv 2408.02161 [초록] | 대기과학 | B (심층 모형의 "부가가치"를 소수 모수 식으로 응축, 오차–복잡도 평면에서 비교. 문턱 없음) | "This added value is condensed into a ten-parameter equation that rivals deep learning models." |
| Grundner A., Beucler T., Gentine P., Eyring V. (2024) "Data-Driven Equation Discovery of a Cloud Cover Parameterization", *JAMES* 16, e2023MS003763 | ✔ DOI 10.1029/2023MS003763 [초록] | 기후 모수화 | B (식은 망이 아니라 데이터에서 직접 찾음. 망은 성능 기준) | "Our best equation balances performance and complexity, achieving a performance comparable to that of neural networks (R²=0.94) while remaining simple (with only 13 trainable parameters)." |
| Zanna L., Bolton T. (2020) "Data-Driven Equation Discovery of Ocean Mesoscale Closures", *GRL* 47, e2020GL088376 | ✔ DOI 10.1029/2020GL088376 [미확인: 초록 미열람] | 해양 모수화 | C (식 발견과 CNN을 나란히 비교한 대표 사례로 통용) | 이번 조사에서 초록을 열지 않았다. |
| Ross A., Li Z., Perezhogin P., Fernandez-Granda C., Zanna L. (2023) "Benchmarking of Machine Learning Ocean Subgrid Parameterizations in an Idealized Model", *JAMES* 15, e2022MS003258 | ✔ DOI 10.1029/2022MS003258 [미확인] | 해양 모수화 | C | 초록 미열람. |
| Pahlavan H.A., Hassanzadeh P., Alexander M.J. (2024) "Explainable Offline-Online Training of Neural Networks for Parameterizations: A 1D Gravity Wave-QBO Testbed in the Small-Data Regime", *GRL* 51, e2023GL106324 | ✔ DOI 10.1029/2023gl106324 [초록] | 기후 모수화 | **C, 우리 '고역 통과 핵' 읽기와 같은 종류의 결론** | "Fourier analysis of these three NNs' kernels … reveals that these NNs primarily learn low-pass, high-pass, and a combination of band-pass filters". |
| Feng D., Liu J., Lawson K., Shen C. (2022) "Differentiable, Learnable, Regionalized Process-Based Models With Multiphysical Outputs can Approach State-Of-The-Art Hydrologic Prediction Accuracy", *Water Resour. Res.* 58, e2022WR032404 | ✔ DOI 10.1029/2022WR032404 [초록] | 수문학 | B (물리 구조 모형이 LSTM 성능에 얼마나 다가가는지를 주 결과로 보고. LSTM을 읽어서 만든 식은 아님) | "δ models can obtain a median Nash-Sutcliffe efficiency of 0.732 for 671 basins across the USA … compared to 0.748 from a state-of-the-art LSTM model with the same setup." |
| Lees T., Reece S., Kratzert F., Klotz D., Gauch M. 외 (2022) "Hydrological concept formation inside long short-term memory (LSTM) networks", *HESS* 26, 3079–3101 | ✔ DOI 10.5194/hess-26-3079-2022 [초록] | 수문학 | C (셀 상태에 선형 탐침. 식 추출·재학습 없음) | "We assess the hypothesis that the LSTM replicates real-world processes … Good correlations (R2 > 0.8) between the probe outputs and the target variables". 선행: Kratzert 외 (2019) "NeuralHydrology – Interpreting LSTMs in Hydrology", ✔ DOI 10.1007/978-3-030-28954-6_19. |
| De la Fuente L., Ehsani M.R., Gupta H.V., Condon L.E. (2024) "Toward interpretable LSTM-based modeling of hydrological systems", *HESS* 28, 945–971 | ✔ DOI 10.5194/hess-28-945-2024 [초록] | 수문학 | C (LSTM을 저수지 모형 꼴로 고쳐 적은 구조) | "The HydroLSTM-based models require fewer cell states to obtain similar performance to their LSTM-based counterparts." |
| Che Z., Purushotham S., Cho K., Sontag D., Liu Y. (2018) "Recurrent Neural Networks for Multivariate Time Series with Missing Values" (GRU-D), *Sci. Rep.* 8, 6085 | ✔ DOI 10.1038/s41598-018-24271-9 [초록] | 결측 시계열 | C (반대 방향: 간격 의존 지수 감쇠를 구조에 넣음. 우리는 표준 LSTM이 간격 의존 이득을 스스로 배웠음을 읽어냄) | "It takes two representations of missing patterns, i.e., masking and time interval, and effectively incorporates them into a deep model architecture". |
| Smyl S. (2020) "A hybrid method of exponential smoothing and recurrent neural networks for time series forecasting", *Int. J. Forecast.* 36, 75–85 | ✔ DOI 10.1016/j.ijforecast.2019.03.017 [미확인: 초록 미열람] | 예측 | C (반대 방향: 지수 평활을 RNN과 결합) | — |
| **Rothmann M., Li N., Chen G., Chi G.Y.H., Temple R., Tsou H.-H. (2003) "Design and analysis of non-inferiority mortality trials in oncology", *Stat. Med.* 22, 239–264** | ✔ DOI 10.1002/sim.1400 [초록] | 임상시험 통계 | **A (사전 고정 보존 분율 검정이라는 통계 형식의 선례. 대상은 약효)** | "testing for retention of a prespecified fraction of the control effect" · "an added requirement that a loss of some specified fraction of the assured effect must be ruled out". |
| **Holmgren E.B. (1999) "Establishing equivalence by showing that a specified percentage of the effect of the active control over placebo is maintained", *J. Biopharm. Stat.* 9, 651–659** | ✔ DOI 10.1081/BIP-100101201 [초록] | 임상시험 통계 | **A (같은 통계 형식)** | "We propose a procedure for establishing equivalence that determines whether a specified percentage of the treatment effect of a known active agent over placebo is maintained." 같은 계보: Hasselblad V., Kong D.F. (2001) *Drug Inf. J.* 35, 435–449, ✔ DOI 10.1177/009286150103500212 [미확인: 초록 미열람]. |

**핵심 정리.**

1. 절차 자체(망을 읽는다 → 읽은 내용을 작은 해석 가능한 모형으로 적는다 → 그 모형을 데이터에 직접 학습한다 → 기준선 대비 망의 이득 가운데 몇 %를 회복하는지 보고한다)는 Maheswaranathan & Sussillo 2020에 그대로 있다. 지표도 같다(기준선 대비 이득의 회복 비율, 90 % 이상). 차이는 세 가지다: 분류 과제이고, 회복 비율의 합격선을 미리 정하지 않았으며, 추출한 모형이 물리 모수를 가진 식이 아니다.
2. 우리 지표의 구조, 곧 (식의 skill) / (망의 skill)을 공통 기준선(PCHIP 또는 persistence) 위에서 재는 것은 임상 비열등성 시험의 "효과 보존(preservation of effect)" 검정과 같은 형식이다. 대조약이 위약 대비 갖는 효과의 정해진 분율을 새 약이 유지하는지를, 분율을 사전에 고정하고 검정한다(Holmgren 1999, Rothmann 2003). "문턱을 미리 고정한 보존 비율 검정"이라는 통계 형식은 그 분야의 표준이다.
3. Eshbaugh 2025는 대리 모형이 망의 출력을 잘 흉내 내도(fidelity가 높아도) 망이 단순 모형 대비 얻은 예측 이득은 회복하지 못할 수 있음을 보였다. 우리가 망 출력에 대한 일치도가 아니라 실제 타깃에 대한 skill 보존을 판정 지표로 삼은 것의 근거로 쓸 수 있다.

### 1-2. 여전히 못 찾은 것과 양성 대조

| 못 찾은 것 | 검색 | 양성 대조 |
|---|---|---|
| 해석 가능한 대리 모형의 보존 비율(또는 fidelity) 합격선을 **실행 전에 등록**한 ML 논문 | 웹 검색 "preregistered / pre-registered + interpretability + fidelity + threshold + surrogate"; OpenAlex 전문 `"preregistered" "knowledge distillation" … fidelity threshold`(6건, 전부 무관한 리뷰), `"pre-registered" "symbolic regression" neural network`(19건, 무관) | 같은 웹 검색이 Eshbaugh 2025(fidelity 대 예측 이득)를 돌려줌. 사전등록+예측 모형 문헌의 존재는 Hofman 외 (2023) "Pre-registration for Predictive Modeling" ✔ arXiv 2311.18807 을 ID로 직접 확인(단 이것은 ID 조회이지 검색 적중이 아님) |
| 학습된 RNN의 은닉 동역학을 읽어 **칼만 이득**(간격 의존 포함)을 닫힌 식으로 적은 논문 | 웹 검색(RNN + Kalman + reverse engineering), arXiv `all:"Kalman filter" AND ti:"recurrent" AND (learn/implement/emerge)`(3건, 전부 칼만을 구조에 넣는 쪽), 웹 검색(변화점 과제 RNN + 적응 이득) | 같은 웹 검색이 KalmanNet 계열(1차 기지)을 돌려줌. 변화점 쪽은 Schaeffer 2020을 돌려줌 |
| 학습된 LSTM이 **지수 가중 평균(EWMA)**을 구현함을 보인 논문 | arXiv `all:"exponential smoothing" AND all:LSTM AND (ti:equivalen OR interpret OR learn)`(4건, 전부 하이브리드 예측) | Crossref에서 Smyl 2020(ES-RNN)이 해석됨. 가장 가까운 이론은 1차의 Tallec & Ollivier 2018(망각 게이트 = 시간상수) |
| **핵융합** 예측기에서 닫힌 식을 추출해 재학습한 논문 | 웹 검색 "symbolic regression distill neural network surrogate tokamak / fusion plasma closed-form"(스텔러레이터 중성자 대리모형 → 멱법칙 식 1건: IEEE TPS 2026 게재 예정이라는 2차 요약, DOI 미확인. 실험 시계열 예측기 아님) | 같은 검색이 1차의 플라즈마 closure 리뷰(arXiv 2511.22486)를 다시 돌려줌 |
| 수문학에서 학습된 LSTM을 식으로 증류해 보존 비율을 잰 논문 | 웹 검색 "LSTM hydrology distill / extract conceptual model symbolic regression streamflow" | Lees 2022, De la Fuente 2024가 같은 검색에서 나옴 |

### 1-3. 논문에 쓸 인용 문장 제안

한국어:
> 학습된 순환 망을 읽어 얻은 기전을 작은 해석 가능한 모형으로 적고, 그 모형이 기준선 대비 망의 성능 이득 가운데 얼마를 회복하는지를 재는 절차는 Maheswaranathan과 Sussillo(2020)가 감성 분류 RNN에서 사용했다(회복 비율 90 % 이상). 본 연구는 이 절차를 물리 회귀 시계열에 적용하되, 추출 대상을 물리 모수 20개의 닫힌 식으로 두고, 회복 비율의 합격선(0.70)을 실행 전에 등록했다. 사전에 정한 분율의 효과 보존을 검정하는 형식은 임상 비열등성 시험의 효과 보존 검정(Holmgren 1999; Rothmann 외 2003)과 같다. 판정 지표를 망 출력에 대한 일치도가 아니라 측정값에 대한 skill로 둔 것은, 일치도가 높은 대리 모형도 망의 예측 이득을 회복하지 못할 수 있다는 보고(Eshbaugh 2025)를 따른 것이다. 핵융합 실험 시계열의 학습된 예측기에 이 절차를 적용한 선행 연구는 조사 범위에서 찾지 못했다.

English:
> The procedure of reading a trained recurrent network, writing the mechanism as a small interpretable model, and measuring how much of the network's gain over a baseline that model recovers was used by Maheswaranathan and Sussillo (2020) on sentiment-classification RNNs (more than 90 % recovered). We apply it to a physical regression time series, take a 20-parameter closed-form equation as the extracted model, and register the acceptance level of the recovered fraction (0.70) before the run. Testing retention of a prespecified fraction of an effect has the same form as the preservation-of-effect test of non-inferiority trials (Holmgren 1999; Rothmann et al. 2003). We score skill against the measurements, not agreement with the network output, because a surrogate with high fidelity can fail to recover the network's predictive gain (Eshbaugh 2025). We found no application of this procedure to a trained predictor of fusion experimental time series within the scope of our search.

---

## 항목 2. 물리에서 정량 예측을 먼저 적고 동결한 학습 모형에서 채점하기

### 2-1. 새로 찾은 선례

| 문헌 | DOI / ID | 분야 | 거리 | 근거 인용 |
|---|---|---|---|---|
| Hakim G.J., Masanam S. (2024) "Dynamical Tests of a Deep Learning Weather Prediction Model", *Artif. Intell. Earth Syst.* 3 | ✔ DOI 10.1175/AIES-D-23-0090.1 [초록] | 기상 ML | **B, 가장 가까움** (교과서 역학이 주는 기대를 동결 모형에 시험. 예측 목록의 사전등록·적중 채점표는 없음) | "we subject one such model, Pangu-Weather, to a set of four classical dynamical experiments that do not resemble the model training data … Perturbing the 500-hPa height field alone yields adjustment from a state of rest to one of wind–pressure balance over ∼6 h … We conclude that the model encodes realistic physics in all experiments". |
| Selz T., Craig G.C. (2023) "Can Artificial Intelligence-Based Weather Prediction Models Simulate the Butterfly Effect?", *GRL* 50, e2023GL105747 | ✔ DOI 10.1029/2023GL105747 [초록] | 기상 ML | **B** (이론이 기대하는 정량 거동을 동결 모형에서 검사, **실패**를 보고) | "from theoretical considerations it is expected that such small-amplitude initial condition perturbations would grow very fast initially … We find however, that the AI-based model completely fails to reproduce the rapid initial growth rates". |
| Geiger A., Lu H., Icard T., Potts C. (2021) "Causal Abstractions of Neural Networks", NeurIPS 2021 | ✔ arXiv 2106.02997 [초록] | ML 해석 | B (사람이 먼저 적은 인과 모형을 동결 망에 개입 실험으로 검정) | "neural representations are aligned with variables in interpretable causal models, and then interchange interventions are used to experimentally verify that the neural representations have the causal properties of their aligned variables." |
| Melloni L., Mudrik L., Pitts M. 외 (2023) "An adversarial collaboration protocol for testing contrasting predictions of global neuronal workspace and integrated information theory", *PLOS ONE* 18, e0268577 | ✔ DOI 10.1371/journal.pone.0268577 [미확인: 제목·서지만] | 신경과학 | B (이론에서 도출한 예측을 **사전등록**하고 통과/실패를 채점하는 형식. 대상은 학습 모형이 아니라 뇌 자료) | 제목 자체가 형식을 말한다: "protocol for testing contrasting predictions". 결과 논문: Cogitate Consortium (2025) *Nature* 642, 133–142, ✔ DOI 10.1038/s41586-025-08888-1. 초록은 열지 않았다. |
| Hofman J.M., Chatzimparmpas A., Sharma A., Watts D.J., Hullman J. (2023) "Pre-registration for Predictive Modeling" | ✔ arXiv 2311.18807 [초록] | ML 방법론 | C (예측 모형 연구의 사전등록 양식. 물리 예측 채점은 아님) | "we propose adapting pre-registration practices from explanatory modeling to predictive modeling … introduce a lightweight pre-registration template". |
| Karpatne A., Atluri G., Faghmous J.H. 외 (2017) "Theory-Guided Data Science: A New Paradigm for Scientific Discovery from Data", *IEEE TKDE* 29, 2318–2331 | ✔ DOI 10.1109/TKDE.2017.2720168 [미확인: 초록 미열람] | 과학 ML | C ("물리적 일관성"을 모형 평가 기준으로 세운 문헌으로 통용) | — |
| Seif A., Hafezi M., Jarzynski C. (2021) "Machine learning the thermodynamic arrow of time", *Nat. Phys.* 17, 105–113 | ✔ DOI 10.1038/s41567-020-1018-2 [미확인: 초록 미제공, 내용은 기억 기반] | 통계물리 ML | B로 추정 (이론이 주는 최적 판별 함수를 학습된 망이 재현하는지 확인) | 내용을 이번 조사에서 확인하지 못했다. 인용하려면 본문 확인 필수. |

### 2-2. 여전히 못 찾은 것과 양성 대조

- **못 찾은 것**: 물리 유도에서 나온 **수치 예측 목록**을 먼저 문서로 고정하고, 동결한 학습 모형에서 각 예측의 적중·실패를 **채점표**로 보고한 논문(ML 일반·기후 에뮬레이터·핵융합 모두).
- 검색: 웹 검색 "machine learning emulator a priori physical predictions tested trained model pre-registered hypotheses physics-informed evaluation climate emulator physical consistency test"(해당 없음. 물리 제약을 손실·구조에 넣는 논문만 나옴); OpenAlex 전문 `"pre-registered" "symbolic regression" neural network`(무관 19건).
- 양성 대조: 같은 주제의 동결 모형 물리 시험 두 편(Hakim & Masanam, Selz & Craig)은 Crossref·OpenAlex에서 해석되고 초록을 받았다. 사전등록 예측 채점 형식(Melloni 2023, Cogitate 2025)도 DOI가 해석된다. 즉 두 구성요소 각각은 검색 경로에 잡히고, 둘을 겸한 문헌만 나오지 않았다.
- 조사 폭의 한계: 이 항목은 고정된 용어가 없어 검색어 의존도가 높다. "prospective test", "blind prediction", "registered report" 같은 다른 표현으로 쓴 문헌은 놓쳤을 수 있다.

### 2-3. 논문에 쓸 인용 문장 제안

한국어:
> 학습을 마친 모형에 이론이 기대하는 거동을 시험하는 접근은 기상 예측 모형에서 보고되어 있다. Hakim과 Masanam(2024)은 Pangu-Weather에 고전 역학 실험 네 가지를 가해 지균 조절 같은 교과서 응답이 나오는지 확인했고, Selz와 Craig(2023)은 이론이 기대하는 초기 오차 성장률을 같은 종류의 모형이 재현하지 못함을 보였다. 이론에서 도출한 예측을 사전에 등록하고 통과 여부를 채점하는 형식은 신경과학의 적대적 협업(Melloni 외 2023)에 있다. 본 연구의 유도 문서는 두 형식을 합친 것으로, 수치 예측을 모형 동결 전에 적고 동결한 모형에서 각 예측의 적중 여부를 기록한다.

English:
> Testing a trained model against the behaviour that theory expects has been reported for weather prediction models: Hakim and Masanam (2024) subjected Pangu-Weather to four classical dynamical experiments and checked for textbook responses such as geostrophic adjustment, and Selz and Craig (2023) showed that a model of the same kind does not reproduce the initial error growth rate that theory expects. Registering theory-derived predictions in advance and scoring them as passed or failed is the format of adversarial collaborations in neuroscience (Melloni et al. 2023). Our derivation document combines the two: the numerical predictions are written before the model is frozen, and each is scored on the frozen model.

---

## 항목 3. Student-t(두꺼운 꼬리) 칼만 필터·점프(hold-or-jump, IMM, 변화점) 상태 모형의 플라즈마·분광 진단 시계열 적용

### 3-1. 새로 찾은 선례

| 문헌 | DOI / ID | 분야·장치 | 거리 | 근거 인용 |
|---|---|---|---|---|
| **Rath K., Rügamer D., Bischl B., von Toussaint U., Rea C., Maris A., Granetz R., Albert C.G. (2022) "Data augmentation for disruption prediction via robust surrogate models", *J. Plasma Phys.* 88** | ✔ DOI 10.1017/S0022377822000769 [초록] | DIII-D 진단 시계열 | **A에 가까움** (두꺼운 꼬리 잡음의 상태공간 모형을 베이즈 필터링으로 토카막 진단 시계열에 적용. 목적은 붕괴 예측용 자료 증강이며 CES·점프 과정은 아님) | "We apply Student t process regression in a state space formulation via Bayesian filtering to tackle challenges imposed by outliers and noise in the training data set … We demonstrate the efficacy of our approach on plasma diagnostics data of three different disruption classes from the DIII-D tokamak." (본문에서 "Kalman"이라는 단어를 쓰는지는 확인하지 못했다.) |
| Rath K. 외 (2023) "Dependent state space Student-t processes for imputation and data augmentation in plasma diagnostics", *Contrib. Plasma Phys.* 63, e202200175 | ✔ DOI 10.1002/ctpp.202200175 [초록] (1차 `lit_problem_models.md`에 수록, `lit_interpretation.md` Q4 판정에는 반영 안 됨) | 자속 루프에서 착안한 시험 사례 | A에 가까움 (같은 방법의 다변량판, 결측점 대입) | "The underlying Student-t process allows for a noise distribution with heavy tails and thus produces robust results in the case of outliers. We consider the state space form of the Student-t process". |
| **van den Brand H., de Baar M.R., van Berkel M., Blanken T.C., Felici F., Westerhof E., Willensdorfer M., ASDEX Upgrade Team (2016) "A model-based, multichannel, real-time capable sawtooth crash detector", *Plasma Phys. Control. Fusion* 58, 075002** | ✔ DOI 10.1088/0741-3335/58/7/075002 [초록] | ASDEX Upgrade ECE | **A에 가까움** (점프 모형 쪽: IMM 추정기를 토카막 진단 시계열에 적용. 사건은 톱니 붕괴, 측정잡음은 가우스) | "estimation of sawtooth crash times is demonstrated using the model-based interacting multiple model (IMM) estimator, based on simplified models for the sawtooth crash … The IMM estimator is tuned and applied to multiple ECE channels at once … For data provided at a sampling rate of 1 kHz, the run time of the IMM estimator is below 1 ms". |
| Blanken T.C., Felici F., de Baar M.R., Heemels W.P.M.H. (2015) "Modeling, observer design and robust control of the particle density profile in tokamak plasmas", CDC 2015, 7628–7635 | ✔ DOI 10.1109/cdc.2015.7403425 [미확인: 검색 요약] | TCV·AUG 밀도 | B (EKF 안에서 간섭계 fringe jump를 추정. 비가우스 사건을 상태에 넣은 사례) | 검색 요약 기준 "an EKF for density profiles that estimates fringe jumps". 학술지 판: Blanken 외 (2018) *Fusion Eng. Des.* 126, 87–103, ✔ DOI 10.1016/j.fusengdes.2017.11.006 (초록 미제공). |
| Pastore F., Sauter O., Felici F. 외 (2026) "Applications of a novel model-based real-time observer for electron density profile control experiments in TCV", *Nucl. Fusion* 66, 126051 | ✔ DOI 10.1088/1741-4326/aeab66 [초록] | TCV 밀도 | B | "yielding reproducible scenarios with a device-agnostic density metric and robust handling of interferometer fringe jumps." |
| Morishita Y., Murakami S., Yokoyama M., Ueno G. (2020) "Data assimilation system based on integrated transport simulation of Large Helical Device plasma", *Nucl. Fusion* 60, 056001 | ✔ DOI 10.1088/1741-4326/ab7596 [초록] | LHD | **B (측정된 T_i 시계열에 칼만형 필터를 적용한 사례.** 수송 코드 기반 앙상블 칼만 필터, 가우스) | "We use the ensemble Kalman filter (EnKF) … for the estimation of state variables composed of the electron and ion temperature, density … The time series data of experimentally measured temperature and density profiles are assimilated into TASK3D." T_i 측정이 하전교환분광인지는 초록에 없다. 후속: ASTI, ✔ DOI 10.1016/j.cpc.2022.108287; 제어 적용 ✔ DOI 10.1038/s41598-023-49432-3; 토카막(JT-60U) 확장 Ichikawa 외 (2025) ✔ DOI 10.1585/pfr.20.1403036. |
| Goumiri I.R., Rowley C.W., Sabbagh S.A. 외 (2016) "Modeling and control of plasma rotation for NSTX using neoclassical toroidal viscosity and neutral beam injection", *Nucl. Fusion* 56, 036023 | ✔ DOI 10.1088/0029-5515/56/3/036023 [초록 + 미확인: 관측기 부분은 학회 슬라이드 요약] | NSTX 회전 | B (회전 프로파일을 점 측정에서 추정하는 칼만 관측기. 모형 기반, 가우스, TRANSP 모의) | 초록: "the controller is able to attain desired plasma rotation profiles given practical constraints on the actuators and the available measurements of rotation." 관측기가 칼만 필터라는 점은 저자의 APS 2016 슬라이드 요약에 근거하며 논문 본문은 열지 못했다. 같은 계열: Wehner W.P., Barton J.E., Schuster E. (2015) "Toroidal rotation profile control for the DIII-D tokamak", ACC 2015, ✔ DOI 10.1109/acc.2015.7171899 (초록에는 상태 되먹임만 있고 관측기 언급 없음). |
| Dose V., von der Linden W. (1999) "Outlier Tolerant Parameter Estimation", in *Maximum Entropy and Bayesian Methods* (Garching 1998), 47–56 | ✔ DOI 10.1007/978-94-011-4710-1_4 [미확인: 본문 미열람] | IPP Garching, 베이즈 추론 | C (핵융합 연구소발 이상치 허용 우도의 원전으로 통용. 정적 추정) | 종설: von Toussaint U. (2011) "Bayesian inference in physics", *Rev. Mod. Phys.* 83, 943–999, ✔ DOI 10.1103/RevModPhys.83.943. |
| Leleux D.P., Claps R., Chen W., Tittel F.K., Harman T.L. (2002) "Applications of Kalman filtering to real-time trace gas concentration measurements", *Appl. Phys. B* 74, 85–93 | ✔ DOI 10.1007/s003400100751 [미확인: 초록 미열람] | 레이저 흡수 분광 | C (분광 측정 시계열의 실시간 칼만 필터, 가우스) | 제목이 내용을 말한다. 분광 일반에서 칼만 필터가 실시간 농도 시계열에 쓰인다는 계보용. |

**1차 판정의 정정.** 1차 `lit_interpretation.md` Q4는 "Student-t 칼만을 토카막 진단에 쓴 논문: 미발견", "CES 시계열에 칼만/강건 필터를 적용한 논문: 미발견", 점프 모형은 IAEA 기고 한 건뿐이라고 적었다. 2차에서 다음이 확인됐다.
- 두꺼운 꼬리 상태공간 필터링은 토카막 진단 시계열에 이미 쓰였다(Rath 2022, 2023).
- 점프(모드 전환) 추정기 IMM은 토카막 ECE에 실시간 요건으로 쓰였다(van den Brand 2016).
- 측정된 T_i 시계열에 칼만형 필터를 적용한 사례가 있다(Morishita 2020, 수송 모형 기반 앙상블 칼만 필터). 회전 프로파일의 칼만 관측기도 제어 문헌에 있다(Goumiri 2016, 본문 미확인).

### 3-2. 여전히 못 찾은 것과 양성 대조

| 못 찾은 것 | 검색 | 양성 대조 |
|---|---|---|
| **하전교환분광** T_i·V_rot 시계열에 **Student-t 측정잡음**을 둔 재귀 필터 | OpenAlex 전문 `"Student-t" "charge exchange" spectroscopy time series filter`(11건, 무관 + FDPVA 회의 요약 1건), `"charge exchange" "Kalman filter" tokamak "ion temperature"`(14건: 제어 논문·학위논문·자료동화) | 두 번째 질의가 Ichikawa 2025(ASTI), Goumiri 2016, Felici 학위논문을 돌려줌. `"Student-t" "Kalman" tokamak`은 Rath 2022를 돌려줌 |
| **hold-or-jump(유지 또는 점프) 잠재과정 + Student-t 측정잡음**을 결합한 필터를 플라즈마 진단에 적용 | OpenAlex 전문 `"jump" "Markov" "Kalman" tokamak sawtooth OR ELM detection`(5건 무관), `"robust Kalman" tokamak outlier diagnostic`(1건 무관), `"heavy-tailed" "state space" tokamak diagnostic`(8건: Rath 2022 외 무관) | `"interacting multiple model" tokamak`이 van den Brand 2016과 학위논문(✔ DOI 10.6100/ir843207)을 돌려줌 |
| 토카막 신호의 베이즈 온라인 변화점 검출 | arXiv `abs:"change point" AND abs:tokamak`(Leddy 2022 한 건, 1차 기지), Crossref 서지 검색(무관) | 같은 arXiv 경로가 Leddy 2022를 돌려줌 |
| Chilenski 2015·Ho 2019의 Student-t 우도 사용 여부 | 이번에 본문을 열지 못했다. 1차 판정(가우스 GP, Student-t는 Leddy 2022·Fischer 2024)을 그대로 둔다 | — |

### 3-3. 논문에 쓸 인용 문장 제안

한국어:
> 토카막 진단 시계열에서 두꺼운 꼬리 잡음과 불연속 사건을 다루는 상태공간 방법은 각각 선례가 있다. Rath 외(2022, 2023)는 Student-t 과정의 상태공간 형식을 베이즈 필터링으로 풀어 DIII-D 진단 신호의 이상치에 강건한 대리 모형을 만들었고, van den Brand 외(2016)는 상호작용 다중 모형(IMM) 추정기를 ASDEX Upgrade의 ECE 채널에 적용해 톱니 붕괴를 1 ms 안에 검출했다. 측정된 이온온도 시계열에 칼만형 필터를 적용한 사례로는 LHD의 수송 모형 기반 자료동화(Morishita 외 2020)가 있다. 본 연구의 물리 필터는 이 요소들을 하전교환분광 한 채널의 시간 방향 추정에 결합한 것으로, 잠재 상태는 유지 또는 점프 과정, 측정잡음은 Student-t 분포로 둔다. 이 결합을 하전교환분광 시계열에 적용한 선행 연구는 조사 범위에서 찾지 못했다.

English:
> State-space methods for heavy-tailed noise and for discontinuous events each have precedents in tokamak diagnostic time series. Rath et al. (2022, 2023) solved the state-space form of a Student-t process by Bayesian filtering to build surrogates of DIII-D diagnostic signals that are robust to outliers, and van den Brand et al. (2016) applied an interacting-multiple-model estimator to ASDEX Upgrade ECE channels and detected sawtooth crashes within 1 ms. A Kalman-type filter on measured ion-temperature time series exists in the transport-model-based data assimilation of LHD (Morishita et al. 2020). Our physical filter combines these elements for the time-direction estimate of a single charge-exchange channel: the latent state follows a hold-or-jump process and the measurement noise is Student-t. We found no application of this combination to charge-exchange time series within the scope of our search.

---

## 항목 4. "나우캐스팅"·"gap filling"·"virtual diagnostic"·"soft sensor" 용어와, 느린 진단의 시간 공백을 빠른 진단으로 채우는 틀

### 4-1. 용어 "nowcasting": 전문 검색 완료

1차는 OpenAlex 전문 검색을 요청 제한으로 끝내지 못했다. 이번에는 완료했다.

- OpenAlex `fulltext.search:nowcasting tokamak` → 32건. 핵융합 논문은 세 편뿐이고 나머지는 지진·우주날씨·일반 ML이다.
  - Gopakumar V. 외 (2024) "Plasma surrogate modelling using Fourier neural operators", *Nucl. Fusion*, DOI 10.1088/1741-4326/ad313a (arXiv 2311.05967) [전문]: "nowcast" 1회, **참고문헌 제목**에서만 등장(Shi 외 2015 "Convolutional LSTM network: A machine learning approach for precipitation nowcasting").
  - Ferreira D.R., Carvalho P.J., Fernandes H. (2019) "Deep Learning for Plasma Tomography and Disruption Prediction From Bolometer Data", *IEEE Trans. Plasma Sci.*, DOI 10.1109/tps.2019.2947304 (arXiv 1910.13257) [전문]: "nowcast" 1회, 같은 **참고문헌 제목**에서만 등장.
  - Pastore F. 외 (2026) *Nucl. Fusion* 66, 126051, DOI 10.1088/1741-4326/aeab66: 본문을 열지 못했다(IOP). 초록에는 없다. 위 두 편과 같이 참고문헌에서의 적중일 가능성이 있으나 **확인하지 못했다**.
- `fulltext.search:"nowcast" tokamak plasma diagnostic` → 11건, 같은 세 편 외 핵융합 논문 없음.
- 웹 검색 `"nowcasting" OR "nowcast" tokamak plasma diagnostics machine learning real-time` → 용어 사용 논문 없음(Diag2Diag, RTCAKENN 등 1차 기지 문헌만).
- 양성 대조: 전문 검색이 **참고문헌 목록 안의 단어까지** IOP *Nuclear Fusion* 논문에서 잡아낸다는 것이 위 Gopakumar 2024로 확인된다. 따라서 본문에서 이 용어를 쓴 *Nuclear Fusion*·IEEE TPS 논문이 OpenAlex 전문 색인 범위에 있었다면 잡혔을 것이다. 단 OpenAlex 전문 색인은 모든 논문을 덮지 않는다(특히 AIP, 학회 초록).
- 판정: 핵융합 문헌에서 진단 추정의 뜻으로 "nowcasting"을 쓴 사례는 **전문 수준에서도 미발견**(미확인 1편 제외).

### 4-2. 새로 찾은 선례: 틀(느린 진단의 시간 해상도를 빠른 진단으로 올리기)

| 문헌 | DOI / ID | 장치 | 거리 | 근거 인용 |
|---|---|---|---|---|
| **Imríšek M., Seidl J., Odložilík M., Sos M., Bílková P., Böhm P., Aksoy H.S., Weiand R.S. 외 (2026) "Temporal super-resolution of kinetic plasma profiles at COMPASS using neural networks", *Plasma Phys. Control. Fusion* 68, 065049** | ✔ DOI 10.1088/1361-6587/ae72c6 [초록] | COMPASS | **B, Diag2Diag와 같은 급의 선행** (빠른 진단 → 느린 Thomson의 시간 해상도 향상. 전자 채널, 동시각 사상으로 보이며 인과 필터·나이브 기준선 대조는 초록에 없음) | "reconstructing electron temperature and density profiles on the COMPASS tokamak using neural networks trained on soft x-ray and absolute extreme ultraviolet radiation diagnostics … enables kinetic profile inference with microsecond temporal resolution, far exceeding that of the standard Thomson scattering diagnostic … Typical reconstruction errors remain within about 10% in the plasma core". |
| **Pastore F., Felici F., Bosman T.O.S.J., Galperti C., Sauter O., Vincent B., Vu N.M.T. (2023) "Model-based electron density estimation using multiple diagnostics on TCV", *Fusion Eng. Des.* 192, 113615** | ✔ DOI 10.1016/j.fusengdes.2023.113615 [초록] | TCV | **B (모형 기반으로 같은 문제를 품: 저주파 진단의 공백을 고주파 진단과 수송 모형으로 인과적으로 채움.** 전자 밀도, 학습 아님) | "reconstruction of density profiles in the TCV tokamak using low-frequency Thomson Scattering measurements and high-frequency interferometer measurements simultaneously, both of which are available in real-time." 후속 Pastore 2026은 이를 "multi-rate electron density observer"로 부른다. |
| Lin Z., Zhang H., Wang F., Bae C., Fu J. 외 (2024) "Prediction of plasma rotation velocity and ion temperature profiles in EAST Tokamak using artificial neural network models", *Nucl. Fusion* 64, 106061 | ✔ DOI 10.1088/1741-4326/ad73e8 [초록] | EAST | C (빔이 필요 없는 다른 T_i·회전 원천인 X선 결정 분광기의 스펙트럼을 신경망으로 읽음. 측정 있는 시각의 추론) | "Artificial neural network models have been developed to predict rotation velocity and ion temperature profiles on the EAST tokamak based on spectral measurements from the x-ray crystal spectrometer … surpassing traditional methods by over tenfold." |
| Morishita 외 2020 (항목 3 표) | ✔ DOI 10.1088/1741-4326/ab7596 | LHD | B (수송 모형이 T_i 측정 사이를 채우는 자료동화) | 위 표 참조. |

### 4-3. 여전히 못 찾은 것과 양성 대조

| 못 찾은 것 | 검색 | 양성 대조 |
|---|---|---|
| 핵융합 문헌의 "nowcasting" 용어(진단 추정의 뜻) | 4-1 | Gopakumar 2024 참고문헌 적중 |
| "soft sensor" 용어의 토카막 진단 적용 | OpenAlex 전문 `"soft sensor" tokamak`(16건: Zenodo 자가출판 1종, 가속기 진공 제어, KSTAR 외부 자기 진단 평형 추론 Nam 외 2026 DOI 10.1088/1741-4326/aeb396 — 마지막 것은 본문에서 용어를 어떻게 쓰는지 미확인) | 같은 질의가 핵융합 논문(Nam 2026)을 색인에서 돌려줌 |
| "gap filling"을 진단 시간 공백에 쓴 핵융합 논문 | OpenAlex 전문 `"gap filling" tokamak diagnostic`(10건, 무관) | — (이 표현의 양성 대조는 확보하지 못했다. 판정 강도 낮음) |
| 이온온도의 "virtual diagnostic" | OpenAlex 전문 `"virtual diagnostic" tokamak "ion temperature"`(11건: 합성 진단·시뮬레이션 문맥. KSTAR 진단 개관 Ko J., Nam Y.U. 2026, DOI 10.1088/2058-6272/aeb854 포함, 본문 미확인) | 같은 질의가 1차 기지의 합성 진단 튜토리얼(DOI 10.1088/1361-6587/aeb280)을 돌려줌 |
| 중성자율에서 실시간 T_i를 추정해 CES 공백을 메우는 연구, 이온온도 "sensor fusion" | 웹 검색(real-time ion temperature + neutron rate / XICS + neural network), OpenAlex 전문 `"sensor fusion" "ion temperature" tokamak real-time neutron`(2건: KSTAR 진단 개관, DEMO 진단 개념 Biel 외 2022 DOI 10.1016/j.fusengdes.2022.113122) | 웹 검색이 Lin 2024(EAST XCS)와 1차 기지의 Pavone(W7-X XICS)을 돌려줌 |
| CES T_i·V_rot 자체의 시간 초해상(빠른 진단으로 10 ms 사이를 채움) | OpenAlex 전문 `"temporal super-resolution" tokamak diagnostic`(3건: Imríšek 2026 외 무관), `"upsampling" "Thomson scattering" neural network tokamak "time resolution"`(1건: Diag2Diag) | 두 질의가 각각 Imríšek 2026과 Diag2Diag를 돌려줌 |

### 4-4. 논문에 쓸 인용 문장 제안

한국어:
> 느린 진단의 시간 해상도를 빠른 진단으로 끌어올리는 연구는 전자 채널에 선례가 있다. Diag2Diag(Jalalvand 외 2025)와 COMPASS의 시간 초해상(Imríšek 외 2026)은 빠른 진단에서 Thomson 산란 프로파일로 가는 동시각 사상을 학습했고, TCV의 다중 속도 관측기(Pastore 외 2023)는 저주파 Thomson 산란과 고주파 간섭계를 수송 모형 기반 확장 칼만 필터로 결합했다. 본 연구는 같은 틀을 이온 채널(하전교환분광의 T_i와 V_rot)로 옮기고, 추정을 과거 정보만 쓰는 인과 형식으로 두며, 비인과 보간과 persistence에 대한 skill로 평가한다. 기상학의 용어를 빌려 이 과제를 나우캐스팅이라 부른다. 핵융합 문헌에서 이 용어를 진단 추정의 뜻으로 쓴 사례는 전문 검색에서 찾지 못했다.

English:
> Raising the time resolution of a slow diagnostic with fast ones has precedents in the electron channel. Diag2Diag (Jalalvand et al. 2025) and the temporal super-resolution at COMPASS (Imríšek et al. 2026) learn a same-time mapping from fast diagnostics to Thomson scattering profiles, and the multi-rate observer on TCV (Pastore et al. 2023) combines low-frequency Thomson scattering with high-frequency interferometry in a transport-model-based extended Kalman filter. We carry the same framing to the ion channel (T_i and V_rot from charge-exchange spectroscopy), make the estimate causal, and score it as skill against acausal interpolation and persistence. We borrow the meteorological term nowcasting for this task; a full-text search found no use of the term for diagnostic estimation in the fusion literature.

---

## 항목 5. 두꺼운 꼬리(Student-t)·혼합 예측 head를 CRPS와 PIT로 채점하는 물리과학 회귀: 기원 문헌

### 5-1. 새로 확인한 기원·선례

| 문헌 | DOI / ID | 분야 | 우리 head와의 관계 | 근거 인용 |
|---|---|---|---|---|
| Gneiting T., Raftery A.E., Westveld A.H., Goldman T. (2005) "Calibrated Probabilistic Forecasting Using Ensemble Model Output Statistics and Minimum CRPS Estimation", *Mon. Weather Rev.* 133, 1098–1118 | ✔ DOI 10.1175/MWR2904.1 [초록] | 기상 후처리 | 모수 예측 분포를 회귀로 내고 CRPS로 맞추고 보정을 확인하는 틀의 출발점(가우스) | "The EMOS technique yields probabilistic forecasts that take the form of Gaussian predictive probability density functions … the method of minimum continuous ranked probability score (CRPS) estimation is introduced … The EMOS predictive PDFs were sharp, and much better calibrated than the raw ensemble". |
| Gneiting T., Balabdaoui F., Raftery A.E. (2007) "Probabilistic Forecasts, Calibration and Sharpness", *J. R. Stat. Soc. B* 69, 243–268 | ✔ DOI 10.1111/j.1467-9868.2007.00587.x [미확인: 초록 미열람] | 통계 | PIT 히스토그램으로 보정을 진단하고 "보정 조건 아래 예리함 최대화"를 원칙으로 세운 문헌으로 통용 | PIT의 원전은 Dawid A.P. (1984) "Statistical Theory: The Prequential Approach", *J. R. Stat. Soc. A* 147, 278, ✔ DOI 10.2307/2981683 (미열람). |
| **Rasp S., Lerch S. (2018) "Neural Networks for Postprocessing Ensemble Weather Forecasts", *Mon. Weather Rev.* 146, 3885–3900** | ✔ DOI 10.1175/MWR-D-18-0187.1 [초록] | 기상 후처리 | **신경망이 예측 분포의 모수를 직접 내는 "분포 회귀 망"의 대표 출처** | "We propose a flexible alternative based on neural networks that can incorporate nonlinear relationships between arbitrary predictor variables and forecast distribution parameters that are automatically learned in a data-driven way". (CRPS를 손실로 쓴다는 점은 본문 사항이며 초록에는 없다.) |
| **Gebetsberger M., Messner J.W., Mayr G.J., Zeileis A. (2018) "Estimation Methods for Nonhomogeneous Regression Models: Minimum Continuous Ranked Probability Score versus Maximum Likelihood", *Mon. Weather Rev.* 146, 4323–4338** | ✔ DOI 10.1175/MWR-D-17-0364.1 [초록] | 기상 후처리 | **Student-t 예측 분포 + NLL 대 CRPS 추정 + 보정·예리함 평가. 우리 B.17·B.17b 구도와 가장 가깝다** | "The heavy-tailed logistic and Student's t distributions can improve forecast performance in terms of sharpness and calibration, and lead to only minor differences between the estimators employed … for a correctly specified model the maximum likelihood estimator is slightly more efficient than the CRPS estimator." |
| Baran S., Lerch S. (2016) "Mixture EMOS model for calibrating ensemble forecasts of wind speed", *Environmetrics* 27, 116–130 | ✔ DOI 10.1002/env.2380 [미확인: 초록 미열람] | 기상 후처리 | 혼합 예측 분포 | 제목 기준. |
| **D'Isanto A., Polsterer K.L. (2018) "Photometric redshift estimation via deep learning", *Astron. Astrophys.* 609, A111** | ✔ DOI 10.1051/0004-6361/201731326 [초록] | 천문학 | **혼합 밀도 망 head를 CRPS와 PIT 둘 다로 채점한 물리과학 회귀 사례** | "A modified version of a deep convolutional network was combined with a mixture density network. The estimates are expressed as Gaussian mixture models … In addition to the traditional scores, the continuous ranked probability score (CRPS) and the probability integral transform (PIT) were applied as performance criteria." |
| Jordan A., Krüger F., Lerch S. (2019) "Evaluating Probabilistic Forecasts with scoringRules", *J. Stat. Softw.* 90(12) | ✔ DOI 10.18637/jss.v090.i12 [미확인: 초록 미열람] | 통계 소프트웨어 | Student-t와 정규 혼합의 CRPS 닫힌 식 출처로 통용 | 정규 혼합 CRPS의 원전: Grimit E.P., Gneiting T., Berrocal V.J., Johnson N.A. (2006) *Q. J. R. Meteorol. Soc.* 132, 2925–2942, ✔ DOI 10.1256/qj.05.235 (미열람). |
| Schulz B., Lerch S. (2022) "Machine Learning Methods for Postprocessing Ensemble Forecasts of Wind Gusts: A Systematic Comparison", *Mon. Weather Rev.* 150, 235–257 | ✔ DOI 10.1175/MWR-D-21-0150.1 [미확인] | 기상 후처리 | 분포 회귀 망·분위수 망·히스토그램 망을 CRPS·PIT로 비교한 체계적 연구로 통용 | 종설: Vannitsem S. 외 (2021) *Bull. Am. Meteorol. Soc.* 102, E681–E699, ✔ DOI 10.1175/BAMS-D-19-0308.1 (미열람). |
| Salinas D., Flunkert V., Gasthaus J., Januschowski T. (2020) "DeepAR: Probabilistic forecasting with autoregressive recurrent networks", *Int. J. Forecast.* 36, 1181–1191 | ✔ DOI 10.1016/j.ijforecast.2019.07.001 [미확인: 초록 미열람] | 예측 | 순환 망 + 모수 분포 head를 NLL로 학습하는 구조의 대표 출처 | — |
| Sorokin D., Granovskiy A., Stokolesov M., Adishchev E., Nurgaliev M. 외 (2026) "Reconstruction of Unobserved Core Electron Density in Tokamak Plasmas with a Neural Operator Transport Surrogate and Uncertainty Quantification" | ✔ arXiv 2610.24525 (arXiv 초록 검색 `abs:CRPS AND (abs:tokamak OR abs:"fusion plasma")`에 적중. 초록 본문은 이번에 받지 않음) | 토카막 | **핵융합에서 CRPS를 쓴 사례** | 1차 `lit_problem_models.md`가 "불확도(CRPS·커버리지)"로 요약. |

**1차 보고서 사이의 불일치 정정.** `lit_interpretation.md` Q5는 "웹 검색 'CRPS tokamak'은 0건"이라 적었지만, 같은 날 `lit_problem_models.md`에는 Sorokin 외 2026(arXiv 2610.24525)이 CRPS·커버리지를 쓴다고 실려 있다. arXiv 초록 검색으로 이 프리프린트가 "CRPS"와 "tokamak"을 초록에 함께 가진 유일한 arXiv 문헌임을 확인했다. 따라서 "핵융합에서 CRPS는 0건"은 틀렸고, "프리프린트 1편(전자 밀도, 신경 연산자)"이 맞다.

### 5-2. 여전히 못 찾은 것과 양성 대조

- **못 찾은 것**: 토카막 회귀에서 Student-t 또는 혼합 head를 NLL로 학습하고 CRPS와 PIT로 채점한 논문.
- 검색: arXiv `abs:CRPS AND (abs:tokamak OR abs:"fusion plasma")` → 1건(Sorokin 2026, Student-t·혼합 head 여부 미확인); 웹 검색 `"CRPS" OR "continuous ranked probability score" tokamak OR "fusion plasma" neural network probabilistic prediction calibration PIT` → 핵융합 논문 없음.
- 양성 대조: arXiv 질의가 Sorokin 2026을 돌려줌.
- 1차의 "Seo 2025 본문 확인 권장"(IEEE Access MDN)은 이번에도 본문을 열지 못했다. 그대로 미해결.

### 5-3. 논문에 쓸 인용 문장 제안

한국어:
> 예측 분포의 모수를 회귀로 내고 CRPS로 채점하며 PIT로 보정을 확인하는 방식은 기상 앙상블 후처리에서 확립되었다(Gneiting 외 2005; Gneiting 외 2007). 신경망이 분포 모수를 직접 내는 형태는 Rasp와 Lerch(2018)가 도입했고, 가우스 가정이 맞지 않을 때 Student-t 같은 두꺼운 꼬리 분포가 보정과 예리함을 개선하며 최대우도 추정과 최소 CRPS 추정의 차이를 줄인다는 점은 Gebetsberger 외(2018)가 보였다. 혼합 밀도 head를 CRPS와 PIT로 함께 평가한 물리과학 사례로는 측광 적색편이 추정(D'Isanto와 Polsterer 2018)이 있다. 본 연구의 예측 head는 이 계보를 따른다.

English:
> Regressing the parameters of a predictive distribution, scoring it with the CRPS and checking calibration with the PIT was established in the post-processing of weather ensembles (Gneiting et al. 2005; Gneiting et al. 2007). Rasp and Lerch (2018) introduced the form in which a neural network outputs the distribution parameters, and Gebetsberger et al. (2018) showed that heavy-tailed distributions such as Student's t improve calibration and sharpness where the Gaussian assumption fails and reduce the difference between maximum-likelihood and minimum-CRPS estimation. A mixture-density head evaluated with both CRPS and PIT in a physical-science regression is the photometric-redshift work of D'Isanto and Polsterer (2018). Our predictive head follows this lineage.

---

## 종합 표

| 항목 | 1차 판정 | 2차 판정 | 바뀐 이유 |
|---|---|---|---|
| 1. 망 → 닫힌 식 → 재학습 → 사전 문턱 보존 검정 | 신규(사전등록 + 비율 규칙). 가장 가까운 선행은 Cranmer 2020 | **절차·지표·사전 문턱 각각 타 분야 선례 있음. 세 가지의 결합과 핵융합 적용은 미발견** | Maheswaranathan & Sussillo 2020이 같은 절차와 같은 지표(기준선 대비 RNN 이득의 90 % 이상 회복)를 썼다. 사전 고정 분율 보존 검정은 임상 비열등성 통계(Holmgren 1999, Rothmann 2003)의 표준 형식이다. Eshbaugh 2025가 일치도 대신 예측 이득 회복을 재야 하는 이유를 준다 |
| 2. 사전 정량 예측을 동결 모형에서 채점 | 신규 | **신규 유지. 단 인접 형식 두 갈래를 인용해야 함** | 동결 모형의 이론 기대 시험(Hakim & Masanam 2024, Selz & Craig 2023)과 사전등록 예측 채점(Melloni 2023, Cogitate 2025)이 각각 있다. 둘을 겸한 문헌은 못 찾았다 |
| 3. Student-t 칼만·점프 모형의 플라즈마 진단 적용 | 타 분야 차용·핵융합 미발견. CES 칼만 미발견 | **구성요소별로 핵융합 선례 있음. CES 대상과 두 요소의 결합은 미발견** | Rath 2022·2023(Student-t 상태공간 필터링, DIII-D 진단), van den Brand 2016(IMM, AUG ECE), Morishita 2020(측정 T_i 시계열의 앙상블 칼만 필터, LHD), Blanken 2015·Pastore 2026(fringe jump 처리) |
| 4. "나우캐스팅" 용어, 시간 공백 채움 틀 | 용어 미발견(전문 검색 미완) | **용어는 전문 검색에서도 미발견(미확인 1편). 틀의 선례 2편 추가** | 전문 검색 완료: 핵융합 적중 3편 중 2편은 참고문헌 제목. Imríšek 2026(COMPASS 시간 초해상), Pastore 2023(다중 속도 관측기)이 같은 틀의 전자 채널 선례 |
| 5. Student-t·혼합 head + CRPS·PIT | 타 분야 차용·핵융합 미발견("CRPS tokamak 0건") | **기원 문헌 확정. 핵융합 CRPS는 0건이 아니라 프리프린트 1편. Student-t·혼합 head는 미발견 유지** | Gebetsberger 2018(Student-t, NLL 대 CRPS), Rasp & Lerch 2018, D'Isanto & Polsterer 2018(MDN + CRPS + PIT). 1차 두 보고서의 불일치(Sorokin 2026) 정정 |

논문 서술에 주는 귀결 세 가지.
1. 항목 1의 "선례를 찾지 못했다"는 문장은 범위를 좁혀야 한다. 절차의 선례(Maheswaranathan & Sussillo 2020)와 통계 형식의 선례(비열등성 효과 보존 검정)를 먼저 적고, 찾지 못한 것은 "물리 회귀 시계열의 닫힌 물리 식 + 사전 등록 합격선, 그리고 핵융합 적용"으로 한정한다.
2. 항목 3에서 "토카막 진단 시계열에 Student-t 필터·점프 모형이 쓰인 적 없다"는 취지의 문장이 논문에 있다면 고쳐야 한다. Rath 2022와 van den Brand 2016이 반례다.
3. 항목 5에서 "핵융합에서 CRPS를 쓴 적 없다"는 문장은 쓰면 안 된다.

---

## 조사 한계

- **본문을 연 문헌은 세 편뿐이다**(Maheswaranathan & Sussillo 2020, Gopakumar 2024, Ferreira 2019, 모두 arXiv PDF). 나머지 인용 문장은 초록에서 가져왔고, 표에 [미확인]으로 적은 항목은 서지만 확인했다. 특히 다음은 인용 전에 본문 확인이 필요하다: Goumiri 2016의 관측기가 칼만 필터라는 점(학회 슬라이드 요약 기준), Rath 2022가 "Kalman"이라는 말을 쓰는지(LMU 저장소 PDF 내려받기 시간 초과), Seif 2021의 내용(기억 기반), Andrews 1995의 fidelity 정의, Rasp & Lerch 2018이 CRPS를 손실로 쓴다는 점, Pastore 2026 본문의 "nowcasting" 등장 위치, Morishita 2020의 T_i 측정이 하전교환분광인지.
- DOI가 없어 검증하지 못한 것: Chan 외 2022 causal scrubbing(Alignment Forum 게시물), TREPAN(Craven & Shavlik, NIPS 8), 스텔러레이터 중성자 대리모형의 기호 회귀 논문(검색 요약에만 등장).
- OpenAlex는 키 없는 공용 예산(하루 약 100회, 같은 IP 공유)이라 전문 검색을 약 30개 질의로 제한했다. 전문 색인은 출판사별로 덮는 범위가 다르다. AIP(*Phys. Plasmas*, *Rev. Sci. Instrum.*)와 학회 초록(APS DPP, EPS, IAEA FEC)은 덮이지 않는 것으로 보고 판정해야 한다.
- IOP·AIP·IEEE 본문은 열지 않았다. arXiv 판이 없는 논문은 초록 수준이다.
- 항목 2는 고정된 용어가 없어 검색어 의존도가 가장 높다. "미발견"의 강도는 다섯 항목 가운데 가장 낮다.
- 항목 4의 "gap filling"은 양성 대조를 확보하지 못했다.
- 비영어 문헌, 학위논문 본문, 사내 보고서는 범위 밖이다.
- arXiv API는 조사 후반에 일부 제목 질의에서 빈 결과를 돌려주었다(요청 간격 문제로 추정). 그 질의들(Valente 2022, Miller 2023, Schaeffer 2020의 arXiv ID 탐색)은 Crossref DOI로 대신 검증했다.
- 이 보고서의 인용 문장 제안은 초안이다. 논문에 넣기 전에 [초록]·[미확인] 표기 문헌은 본문에서 문장을 다시 확인해야 한다.
