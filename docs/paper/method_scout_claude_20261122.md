# Claude API 기반 pedestal nowcasting 방법 탐색

문헌·방법 탐색이며 실험 결과가 아니다. B.85 학습 파일이나 GPU 작업은 변경하지 않았다. 공식 Anthropic API에 식별 정보·원시 자료·미공개 성능 수치 없이 방법론 요약만 보냈다. Claude의 SEARCH-VERIFIED 표시는 API 검색 결과에 연결되었다는 뜻이며, 부모 에이전트의 별도 원문 검증과 구분한다.

모델 제안과 적용 가능성·연산 부담은 추론이며 실제 pedestal 성능이나 10 ms 충족을 입증하지 않는다. 현재 센서값과 이후 센서 수준은 독립적인 참 온도가 아니다.

## 원시 Claude 응답을 읽을 때의 교정 사항

- 아래 원문에 나오는 "easily within 10 ms", "3–10× slower", "microseconds", "calibrated uncertainty", "no leakage risk"는 이 자료에서 측정·입증한 결론이 아니다. 실제 구현과 CPU latency·보정 검사가 필요하다.
- KalmanNet은 원래 부분적으로 알려진 동역학/관측 모델을 활용한다. 동역학 모델 없이 그대로 적용 가능하다는 원문의 표현은 과도하다. 새로 학습하는 모델을 붙이면 별도 가정과 비교가 필요하다.
- CRU의 부모 별도 확인 원문은 [PMLR](https://proceedings.mlr.press/v162/schirmer22a.html), 공식 코드는 [Continuous-Recurrent-Units](https://github.com/boschresearch/Continuous-Recurrent-Units)다. 원시 응답의 boschresearch/CRU 주소는 확인된 공식 주소로 채택하지 않는다.
- 현재 기준은 Gaussian NLL이 아니라 MSE다. beta-NLL은 variance head 없이 단순 손실 치환으로 적용할 수 없다. Faithful heteroscedastic regression 원 논문의 stop-gradient/최적화 설계와 별도 mean-freeze 변형을 혼동하지 않는다.
- Conformal 구간은 조건 없는 물리적 보장이 아니다. CES/L5 residual은 해당 측정·분석 결과가 실제 도착한 뒤에만 calibration에 사용할 수 있다. 시계열 의존성과 가용 조건 이동 아래 포함률은 별도로 검사한다.
- Mixture의 outlier 성분이나 abstention이 실제 fit 실패를 식별했다고 간주하지 않는다. 선택한 행의 오차 감소는 같은 coverage의 기준과 비교하며, IPW로 MNAR 문제가 식별된다고 주장하지 않는다.
- Attention weight는 물리적 설명이나 인과 기여의 증거가 아니다. 구조가 causal이어도 전처리·정규화·사전학습·teacher 구성에 미래 정보나 test 파일이 들어가면 누출된다.
- 빠른 진단은 전체 자료에서 10 ms stamp 표본이다. 이것으로 관측하지 않은 고주파 스펙트럼·ELM 사건·수송을 복원했다고 주장하지 않는다. CES 지연만으로 ODE/CDE가 더 적합하다고 결론내리지 않는다.
- 10 ms 표본만으로 kHz 물리 모드를 식별할 수 없다. Koopman/SSM 고유값이 곧 물리 모드라는 해석, Granger 관계가 곧 물리적 인과라는 해석, 근거 없는 시간 가역성 제약은 채택하지 않는다. Collisionality·ELM phase 등 없는 검증 변수를 쓰는 제안은 추가 자료가 필요한 방법으로 분류한다.
- 온라인 보정은 CES가 도착한 시각의 feature가 아니라 그 CES가 실제 측정된 시각의 cached feature/예측과 짝지어야 한다. RTCAKENN의 sub-ms 속도·parameter 수에 관한 원시 응답은 이 보고서의 확인된 사실로 채택하지 않는다.
- 제안된 임의 성능 문턱, 수치적 연산비, 자료 규모 표현은 채택한 실험 설계나 확인된 자료 사실이 아니다. 사전학습도 fit 파일만 사용하며 held-out cadence와 test 파일은 선택에 사용하지 않는다.
- 짧은 attention/TCN 창의 길이는 최대 CES 지연과 fast-input anchor 정보를 보유할 수 있는지 따로 확인한다. 제공되지 않은 평형·PDE 계수·fit 오차가 필요한 방식은 현재 적용 가능 후보와 분리한다.

## API 실행 기록

| 관점 | 모델 | 검색 사용 | 입력 토큰 | 출력 토큰 | 종료 |
|---|---|---|---:|---:|---|
| dynamics | claude-opus-4-6 | True | 47690 | 3763 | end_turn |
| representation | claude-opus-4-6 | True | 50655 | 4241 | end_turn |
| observation | claude-opus-4-6 | True | 39609 | 4186 | end_turn |
| transfer_physics | claude-opus-4-6 | True | 41913 | 4043 | end_turn |
| adversarial synthesis | claude-opus-4-6 | prior four searches | 15222 | 4000 | max_tokens |

검색 도구 요청 14회가 기록되었고, 그중 제한 초과 오류 2회가 기록되었다. 오류도 아래 관점별 목록에 표시한다. 반환된 URL은 검색 inventory이며 모두 독립 확인한 1차 출처라는 뜻이 아니다.

정확한 프롬프트·원시 응답·검색 질의·URL·사용량은 ignored `data/.method_scout_20261122/*.json`에 보존한다. 재실행은 캐시를 재사용하며 자동 유료 재시도는 하지 않는다. 키는 읽기 전용 credential helper가 지정 경로에서 실행 시 읽으며 출력·커밋하지 않는다.

## 종합 판단과 완결성

네 관점의 검색 응답은 모두 완결되었고 각 여덟 항목, 합계 서른두 제안 항목을 냈다. S4/Mamba·ODE 계열 등이 관점 사이에 중복되므로 이를 서른두 개의 독립 방법이나 독립 증거로 세지 않는다. GRU-D/T-LSTM, S4D/S5/Mamba 등도 구현 예산을 정할 때 상위 기전 단위로 다시 묶는다.

마지막 Claude 종합 응답은 출력 한도에서 중단되었다. family map은 나왔지만 상위 다섯 비교와 비판은 완결되지 않았다. 따라서 그 미완성 순위를 채택하지 않는다. 원문과 종료 상태를 캐시에 보존했고 유료 재생성을 하지 않았다.

아래는 완결된 네 검색과 명시한 교정을 반영한 진행자의 요약이다. 아직 새 실험을 실행하거나 우위를 입증한 것은 아니다.

| 해결할 문제 | 검토할 기전 | 현재 자료에서의 최소 비교 | 제한 |
|---|---|---|---|
| CES 가용 조건 전이 | learned decay RNN(GRU-D/T-LSTM), 연속 시간 관측 갱신(CRU/RKN류) | 동일 입력·분할·head에서 기존 direct recurrent model과 비교; 학습하지 않은 cadence 포함 | 규칙을 많이 외운 것과 관측 갱신을 일반화한 것을 구분; 선형 dynamics 가정과 calibration을 검증 |
| 빠른 진단의 시간 표현 | causal TCN, 작은 S4D, 진단별 encoder/fusion | 같은 수용 정보·parameter/연산 예산에서 과거 CES만 쓰는 쌍둥이와 함께 비교 | 10 ms stamp 자료에 없는 고주파 정보는 생성하지 않음; 모든 전처리와 RF를 인과적으로 고정 |
| 표적 잡음과 신뢰도 | Student-t/Huber, faithful heteroscedastic head, 검증된 selective prediction | 평균 오차·꼬리 오차·구간 포함률·같은 coverage의 기준을 분리 | 낮은 오차나 outlier mixture는 실제 fit 오류 식별 또는 참 온도 복원의 증거가 아님 |
| 방전/시기 이동 | timestamp-aligned online last-layer update, shift-aware calibration | 실제 CES 도착 시점만 갱신하고 chronological 설계에서 frozen 기준과 비교 | 늦게 도착한 CES는 원 측정 시각의 cached feature와 짝지어야 함; MNAR/shift 보장 없음 |
| 물리적 쓸모 | 관측 가능한 하강/회복의 감시, 예측 기여·위상 지연의 통제 비교 | 입력으로 고정한 사건·조건에서 감시 지연과 거짓 경보를 같이 검증 | Granger·latent pole·attention만으로 인과 수송이나 물리 모드 발견을 선언하지 않음 |

부모가 추가로 제안한 적용 아이디어는 anchor/current Siamese encoder 또는 ΔTi/log-ratio 모델이다. 마지막 CES의 원 측정 시각에서의 fast feature와 현재 fast feature, age를 함께 써서 변화량을 직접 학습한다. 이는 이 Claude 네 응답에서 확인된 독립 문헌 결과가 아니라 프로젝트별 설계 제안이며, 기존 tree의 anchor-feature 계보를 인정하고 별도 등록·비교해야 한다.

## 탐색 관점: dynamics

후보 family: GRU-D, causal Neural CDE, ODE-RNN / latent ODE, continuous recurrent unit (CRU), mTAN, KalmanNet / learned filtering gain, T-LSTM, decay plus linear/GP baseline.

정확한 Claude 원문은 해당 관점 JSON의 `text`와 `content`에 보존했다. 검색 원문의 긴 문구를 보고서에 재게시하지 않으며, 아래 URL 목록은 검색이 반환한 자료의 inventory다. 이 중 2차 자료·잘못 연결된 자료도 있으므로 원 논문 검증과 구분한다.

### 실제 API 검색 질의

```json
[
  {
    "query": "GRU-D decay time-aware recurrent missing data Che 2018"
  },
  {
    "query": "neural controlled differential equation Kidger 2020 paper"
  },
  {
    "query": "mTAN multi-time attention network irregular time series Shukla 2021"
  },
  {
    "query": "Latent ODE Rubanova 2019 ODE-RNN CRU Schirmer state space irregular"
  }
]
```

### API가 반환한 검색·인용 URL

- [GitHub - PeterChe1990/GRU-D: GRU-D, a GRU-based model with trainable decays for multivariate time series classification with missing values/irregular samplings · GitHub](https://github.com/PeterChe1990/GRU-D)
- [(PDF) Recurrent Neural Networks for Multivariate Time Series with Missing Values](https://www.researchgate.net/publication/303822097_Recurrent_Neural_Networks_for_Multivariate_Time_Series_with_Missing_Values)
- [(PDF) Recurrent Neural Networks for Multivariate Time Series with Missing Values. (2018)](https://scispace.com/papers/recurrent-neural-networks-for-multivariate-time-series-with-15kdw8uqfw)
- [A Survey on Principles, Models and Methods for Learning from Irregularly Sampled Time Series](https://arxiv.org/pdf/2012.00168)
- [Set Functions for Time Series](https://arxiv.org/pdf/1909.12064)
- [Multi-view Integration Learning for Irregularly-sampled Clinical Time Series](https://arxiv.org/pdf/2101.09986)
- [Probabilistic Imputation for Time-series Classification with Missing Data](https://arxiv.org/pdf/2308.06738)
- [Recurrent Neural Networks for Multivariate Time Series with Missing Values](https://www.nature.com/articles/s41598-018-24271-9)
- [Recurrent Neural Networks for Multivariate Time Series ...](https://dspace.mit.edu/bitstream/handle/1721.1/134960/s41598-018-24271-9.pdf?sequence=2&isAllowed=y)
- [Neural Controlled Differential Equations for Irregular Time Series](https://proceedings.neurips.cc/paper/2020/hash/4a5876b450b45371f6cfe5047ac8cd45-Abstract.html)
- [Neural controlled differential equations for irregular time series](https://www.maths.ox.ac.uk/node/38469)
- [NeurIPS Poster Neural Controlled Differential Equations for Irregular Time Series](https://neurips.cc/virtual/2020/poster/17252)
- [arXiv:2005.08926v2 cs.LG 5 Nov 2020 Neural Controlled Differential](https://arxiv.org/pdf/2005.08926)
- [Neural controlled differential equations for irregular time series - ORA - Oxford University Research Archive](https://ora.ox.ac.uk/objects/uuid:aa145098-c206-4729-b0e7-1123b6f2b416)
- [GitHub - patrick-kidger/NeuralCDE: Code for "Neural Controlled Differential Equations for Irregular Time Series" (Neurips 2020 Spotlight) · GitHub](https://github.com/patrick-kidger/NeuralCDE)
- [Comprehensive Review of Neural Differential Equations for Time Series Analysis](https://arxiv.org/pdf/2502.09885)
- [Neural controlled differential equations for irregular time series](https://dl.acm.org/doi/10.5555/3495724.3496286)
- [Incremental Neural Controlled Differential Equations for Modeling of Path-dependent Material Behavior](https://arxiv.org/pdf/2311.17336)
- [On the Generalization and Approximation Capacities of Neural Controlled Differential Equations](https://arxiv.org/pdf/2305.16791)
- [mTAN: Multi-Time Attention Networks For Irregularly Sampled Time Series](https://iclr.cc/media/iclr-2021/Slides/2703.pdf)
- [Benchmarking with MIMIC-IV, an irregular, spare clinical time series dataset](https://arxiv.org/pdf/2401.15290)
- [Published as a conference paper at ICLR 2021 MULTI-TIME ATTENTION NETWORKS FOR](https://arxiv.org/pdf/2101.10318)
- [Mtan](https://www.scribd.com/document/940029750/mtan)
- [mTAN/README.md at main · reml-lab/mTAN · GitHub](https://github.com/reml-lab/mTAN/blob/main/README.md)
- [Beyond MSE: Rethinking the Evaluation Metric and Benchmarking for Irregular Time Series Forecasting](https://arxiv.org/pdf/2608.17293)
- [Fast and Flexible Characterisation of Astronomical Light Curves Using Multi-Time Attention](https://arxiv.org/pdf/2605.24095)
- [TACTiS: Transformer-Attentional Copulas for Time Series](https://arxiv.org/pdf/2202.03528)
- [Online Irregular Multivariate Time Series Forecasting via Uncertainty-Driven Dual-Expert Calibration](https://arxiv.org/pdf/2605.28603)
- [MILM: Large Language Models for Multimodal Irregular Time Series with Informative Sampling](https://arxiv.org/pdf/2605.13711)
검색 도구 오류: max_uses_exceeded

## 탐색 관점: representation

후보 family: causal TCN, S4D / diagonal SSM, Mamba / selective SSM, modality-specific encoders and fusion, masked time-series pretraining, teacher-student distillation, temporal contrastive learning (TS2Vec), compact causal attention.

정확한 Claude 원문은 해당 관점 JSON의 `text`와 `content`에 보존했다. 검색 원문의 긴 문구를 보고서에 재게시하지 않으며, 아래 URL 목록은 검색이 반환한 자료의 inventory다. 이 중 2차 자료·잘못 연결된 자료도 있으므로 원 논문 검증과 구분한다.

### 실제 API 검색 질의

```json
[
  {
    "query": "S4 structured state space sequence model Gu Rao 2022 paper code"
  },
  {
    "query": "causal temporal convolutional network TCN Bai Kolter Koltun 2018 paper"
  },
  {
    "query": "masked autoencoder time series pretraining survey 2024"
  },
  {
    "query": "Mamba selective state space model Gu Dao 2024 paper code"
  }
]
```

### API가 반환한 검색·인용 URL

- [Modeling Multivariate Biosignals With Graph Neural Networks and Structured State Space Models](https://arxiv.org/pdf/2211.11176)
- [Hieros: Hierarchical Imagination on Structured State Space Sequence World Models](https://arxiv.org/pdf/2310.05167)
- [In-Context Language Learning: Architectures and Algorithms](https://arxiv.org/pdf/2401.12973)
- [GitHub - raminmh/l-state-spaces: Sequence Modeling with Structured State Spaces · GitHub](https://github.com/raminmh/l-state-spaces)
- [GitHub - XuezheMax/s4: Sequence Modeling with Structured State Spaces · GitHub](https://github.com/XuezheMax/s4)
- [Rivaling Transformers: Multi-Scale Structured State-Space Mixtures for Agentic 6G O-RAN](https://arxiv.org/pdf/2510.05255)
- [Mastering Memory Tasks with World Models](https://arxiv.org/pdf/2403.04253)
- [GitHub - state-spaces/s4: Structured state space sequence models · GitHub](https://github.com/state-spaces/s4)
- [Efficiently Modeling Long Sequences with Structured State Spaces Albert Gu,](https://arxiv.org/pdf/2111.00396)
- [A deep convolutional neural network that is invariant to time rescaling](https://arxiv.org/pdf/2107.04616)
- [TCN - Nixtla](https://nixtlaverse.nixtla.io/neuralforecast/models.tcn.html)
- [Temporal Convolutional Network · GitHub](https://gist.github.com/JLDC/d57c1d5b9071270fa0baca0c06d4b518)
- [Early Recognition of Sepsis with Gaussian Process Temporal Convolutional Networks and Dynamic Time Warping](https://arxiv.org/pdf/1902.01659)
- [TCNCA: Temporal Convolution Network with Chunked Attention for Scalable Sequence Processing](https://arxiv.org/pdf/2312.05605)
- [Enhancing Black-Litterman Portfolio via Hybrid Forecasting Model Combining Multivariate Decomposition and Noise Reduction](https://arxiv.org/pdf/2505.01781)
- [Quant GANs: Deep Generation of Financial Time Series](https://arxiv.org/pdf/1907.06673)
- [PDF An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling](https://www.semanticscholar.org/paper/An-Empirical-Evaluation-of-Generic-Convolutional-Bai-Kolter/921196c32213a229245a9705ee4768bc941e7a26)
- [Solar Cycle Prediction Using TCN Deep Learning Model with One-Step Pattern](https://arxiv.org/pdf/2503.00806)
- [Research on an intelligent fault diagnosis method for nuclear power plants based on ETCN-SSA combined algorithm](https://arxiv.org/pdf/2411.06765)
- [GitHub - LMissher/TFMAE: ICDE'2024 Temporal-Frequency Masked Autoencoders for Time Series Anomaly Detection · GitHub](https://github.com/LMissher/TFMAE)
- [Spatial-Temporal-Decoupled Masked Pre-training for Spatiotemporal Forecasting](https://www.ijcai.org/proceedings/2024/0442.pdf)
- [2408.17253 VisionTS: Visual Masked Autoencoders Are Free-Lunch Zero-Shot Time Series Forecasters](https://arxiv.org/abs/2408.17253)
- [Self-Distilled Representation Learning for Time Series](https://arxiv.org/pdf/2311.11335)
- [HiMTM: Hierarchical Multi-Scale Masked Time Series Modeling with Self-Distillation for Long-Term Forecasting](https://arxiv.org/pdf/2401.05012)
- [A Survey of Post-Training in Time Series Foundation Models](https://arxiv.org/pdf/2607.20002)
- [ICLR Frequency-Aware Masked Autoencoders for Multimodal Pretraining on Biosignals](https://iclr.cc/virtual/2024/23539)
- [Be Wary of Your Time Series Preprocessing](https://arxiv.org/pdf/2602.17568)
- [Ti-MAE: Self-Supervised Masked Time Series Autoencoders](https://arxiv.org/pdf/2301.08871)
검색 도구 오류: max_uses_exceeded

## 탐색 관점: observation

후보 family: faithful heteroscedastic regression, beta-NLL, Student-t likelihood, EnbPI / time-series conformal, missing-data-augmentation conformal, inlier/outlier mixture likelihood, selective abstention, propensity/IPW weighting.

정확한 Claude 원문은 해당 관점 JSON의 `text`와 `content`에 보존했다. 검색 원문의 긴 문구를 보고서에 재게시하지 않으며, 아래 URL 목록은 검색이 반환한 자료의 inventory다. 이 중 2차 자료·잘못 연결된 자료도 있으므로 원 논문 검증과 구분한다.

### 실제 API 검색 질의

```json
[
  {
    "query": "heteroscedastic regression neural network calibration survey"
  },
  {
    "query": "conformal prediction missing data causal time series"
  },
  {
    "query": "robust loss Student-t likelihood neural network regression outliers beta-NLL"
  }
]
```

### API가 반환한 검색·인용 URL

- [Calibrated Reliable Regression using Maximum Mean Discrepancy Peng Cui1 2, |](https://arxiv.org/pdf/2006.10255)
- [Fully Heteroscedastic Count Regression with Deep Double Poisson Networks](https://arxiv.org/html/2406.09262v3)
- [Faithful Heteroscedastic Regression with Neural Networks Andrew Stirn](https://proceedings.mlr.press/v206/stirn23a/stirn23a.pdf)
- [Heteroscedastic Calibration of Uncertainty Estimators in Deep Learning](https://arxiv.org/html/1910.14179)
- [PDF Faithful Heteroscedastic Regression with Neural Networks](https://www.semanticscholar.org/paper/Faithful-Heteroscedastic-Regression-with-Neural-Stirn-Wessels/934ef7a5f1e876ceb4bcbf5c59cbacd7e12e4733)
- [Flexible Heteroscedastic Count Regression with Deep Double Poisson Networks](https://arxiv.org/html/2406.09262v1)
- [Variational Variance: Simple, Reliable, Calibrated Heteroscedastic Noise](https://arxiv.org/pdf/2006.04910)
- [Faithful Heteroscedastic Regression with Neural Networks](https://arxiv.org/pdf/2212.09184)
- [Accurate Uncertainties for Deep Learning Using Calibrated Regression](https://proceedings.mlr.press/v80/kuleshov18a/kuleshov18a.pdf)
- [Calibrated Reliable Regression using Maximum Mean Discrepancy Peng Cui1 2,](https://proceedings.neurips.cc/paper/2020/file/c74c4bf0dad9cbae3d80faa054b7d8ca-Paper.pdf)
- [Conformal prediction for time series Chen Xu∗1 and Yao Xie†1 1H. Milton Stewart](https://arxiv.org/pdf/2010.09107)
- [(PDF) Conformal Inference For Missing Data under Multiple Robust Learning](https://www.researchgate.net/publication/396747247_Conformal_Inference_For_Missing_Data_under_Multiple_Robust_Learning)
- [Conformal Prediction: A Data Perspective](https://arxiv.org/pdf/2410.06494)
- [2010.09107 Conformal prediction for time series](https://ar5iv.labs.arxiv.org/html/2010.09107)
- [Conformal Inference for Missing Data under Multiple Robust Learning](https://arxiv.org/html/2510.18149v1)
- [ESTIMATION AND PREDICTION PROBLEMS IN MISSING DATA Yachong Yang A DISSERTATION](https://repository.upenn.edu/server/api/core/bitstreams/d0f15895-e64c-42f5-81bf-55ec108f8c66/content)
- [Conformal Prediction with Missing Values](https://proceedings.mlr.press/v202/zaffran23a/zaffran23a.pdf)
- [Fair Conformal Prediction for Incomplete Covariate Data](https://arxiv.org/pdf/2504.12582)
- [Fair Conformal Prediction for Incomplete Covariate Data](https://arxiv.org/html/2504.12582)
- [Published as a conference paper at ICLR 2022 ON THE PITFALLS OF HETEROSCEDASTIC](https://arxiv.org/pdf/2203.09168)
- [Variational Inference based on Robust Divergences](https://arxiv.org/pdf/1710.06595)
- [GLM: Robust Regression using Custom Likelihood for Outlier Classification — PyMC example gallery](https://www.pymc.io/projects/examples/en/latest/generalized_linear_models/GLM-robust-with-outlier-detection.html)
- [Probabilistic Neural Networks (PNNs) with t-Distributed Outputs: Adaptive Prediction Intervals Beyond Gaussian Assumptions](https://arxiv.org/pdf/2503.12354)
- [Maximum Likelihood Uncertainty Estimation: Robustness to Outliers](https://arxiv.org/pdf/2202.03870)
- [Robust Bayesian Optimization with Student-t Likelihood](https://arxiv.org/pdf/1707.05729)
- [Robust Regression with Student’s 𝑇: The Role of Degrees of Freedom](https://arxiv.org/html/2603.00269)
- [Practical Bayesian optimization in the presence of outliers](https://proceedings.mlr.press/v84/martinez-cantin18a/martinez-cantin18a.pdf)
- [(PDF) On the Pitfalls of Heteroscedastic Uncertainty Estimation with Probabilistic Neural Networks](https://www.researchgate.net/publication/359309577_On_the_Pitfalls_of_Heteroscedastic_Uncertainty_Estimation_with_Probabilistic_Neural_Networks)

## 탐색 관점: transfer_physics

후보 family: Koopman latent-linear dynamics, S4/Mamba (overlap with representation), Koopman/Granger exploratory analysis, online Bayesian last-layer adaptation, latent ODE/SDE (overlap with dynamics), IRM/DANN domain regularization, online change-point-aware filtering, temporal self-supervised consistency.

정확한 Claude 원문은 해당 관점 JSON의 `text`와 `content`에 보존했다. 검색 원문의 긴 문구를 보고서에 재게시하지 않으며, 아래 URL 목록은 검색이 반환한 자료의 inventory다. 이 중 2차 자료·잘못 연결된 자료도 있으므로 원 논문 검증과 구분한다.

### 실제 API 검색 질의

```json
[
  {
    "query": "Koopman operator deep learning causal time series survey"
  },
  {
    "query": "RTCAKENN Diag2Diag fusion diagnostic neural network"
  },
  {
    "query": "physics-informed neural network domain generalization online adaptation time series survey 2024"
  }
]
```

### API가 반환한 검색·인용 URL

- [Deep Koopman operators for causal discovery (Journal Article)](https://www.osti.gov/pages/biblio/3021577-deep-koopman-operators-causal-discovery)
- [LA-UR-25-29646 Accepted Manuscript Deep Koopman operators for causal discovery](https://www.osti.gov/pages/servlets/purl/3021577)
- [Deep Koopman-layered Model with Universal Property Based on Toeplitz Matrices](https://arxiv.org/pdf/2410.02199)
- [Generative Modeling of Regular and Irregular Time Series Data via Koopman VAEs](https://arxiv.org/pdf/2310.02619)
- [Koopa: Learning Non-stationary Time Series Dynamics with Koopman Predictors](https://ise.thss.tsinghua.edu.cn/~mlong/doc/Koopa-nips23.pdf)
- [ViT-Koop: Vision-Transformer–Koopman Operators for Efﬁcient Time-Series](https://openaccess.thecvf.com/content/ICCV2025W/SEA/papers/Shinohara_ViT-Koop_Vision-Transformer-Koopman_Operators_for_Efficient_Time-Series_Forecasting_of_Earth-Observation_Data_ICCVW_2025_paper.pdf)
- [DeepKoopFormer: a Koopman enhanced transformer based architecture for time series forecasting](https://www.nature.com/articles/s44387-026-00085-3)
- [(PDF) Deep Koopman operator framework for causal discovery in nonlinear dynamical systems](https://www.researchgate.net/publication/391954336_Deep_Koopman_operator_framework_for_causal_discovery_in_nonlinear_dynamical_systems)
- [Koopman operator theory: fundamentals, control, and applications](https://arxiv.org/pdf/2607.01819)
- [Deep Koopman operator framework for causal discovery in ...](https://openreview.net/pdf/e22f5f9763bdf702b4d82f07dd9dc47e6fee757d.pdf)
- [(PDF) Summary report from the mini-conference on Digital Twins for Fusion Research](https://www.researchgate.net/publication/391807477_Summary_report_from_the_mini-conference_on_Digital_Twins_for_Fusion_Research)
- [Is Your AI Fast Enough to Run a Fusion Reactor?](https://arxiv.org/html/2610.00845v1)
- [Diag2Diag: Multimodal super-resolution](https://arxiv.org/pdf/2405.05908v2)
- [International Atomic Energy Agency Nuclear Fusion](https://plasmacont.deptcpanel.princeton.edu/assets/data/publications/pdfs/[303]%20Ricardo%20Shousha%20et%20al.%20%E2%80%9CMachine%20learning-based%20real-time%20kinetic%20profile%20reconstruction%20in%20DIII-D%E2%80%9D.pdf)
- [IFAC PapersOnLine 59-11 (2025) 198–203 ScienceDirect](https://www.osti.gov/pages/servlets/purl/2999845)
- [Diag2Diag Explained: 5000x Faster Fusion Diagnostics](https://www.bohrium.com/en/blog/research-notes/aaai-2026-multimodal-super-resolution-discovering-hidden-physics-fusion-plasmas/)
- [A Knowledge-Guide Data-Driven Model with Selective Wavelet Kernel Fusion Neural Network for Gearbox Intelligent Fault Diagnosis](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12736886/)
- [An Ensemble Deep Convolutional Neural Network Model with Improved D-S Evidence Fusion for Bearing Fault Diagnosis](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5579931/)
- [A Deep Learning Model for Fault Diagnosis with a Deep Neural Network and Feature Fusion on Multi-Channel Sensory Signals](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7436083/)
- [ADAPTING PHYSICS-INFORMED NEURAL NETWORKS TO](https://arxiv.org/pdf/2406.05108v1)
- [Adapting Physics-Informed Neural Networks to Improve ODE Optimization in Mosquito Population Dynamics](https://arxiv.org/pdf/2406.05108)
- [PINNMorph: Evolving Online Adaptation Policies for Physics-Informed Neural Networks](https://arxiv.org/pdf/2609.32685)
- [Adaptive Physics-informed Neural Networks: A Survey](https://arxiv.org/pdf/2503.18181)
- [Exploring Physics-Informed Neural Networks: From Fundamentals to Applications in Complex Systems](https://arxiv.org/pdf/2410.00422)
- [Ensemble Learning with Physics-Informed Neural Networks for Harsh Time Series Analysis](https://link.springer.com/chapter/10.1007/978-3-031-61418-7_5)
- [Adaptive Physics-informed Neural Networks: A Survey](https://www.researchgate.net/publication/390141971_Adaptive_Physics-informed_Neural_Networks_A_Survey)
- [Adaptive Physics-informed Neural Networks](https://arxiv.org/html/2503.18181)
- [Initialization-enhanced Physics-Informed Neural Network with Domain Decomposition (IDPINN)](https://arxiv.org/pdf/2406.03172)
- [Understanding and Mitigating Extrapolation Failures in Physics-Informed Neural Networks](https://arxiv.org/pdf/2306.09478)
