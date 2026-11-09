# 문헌 점검: 해석(interpretation) 방법과 물리 필터 — KSTAR CES 나우캐스팅 논문

작성일 2026-11-09. 조사 경로: OpenAlex API(앞부분; 이후 IP 단위 429 차단), Crossref API(DOI·서지 검증), arXiv export API(arXiv ID 검증), 웹 검색(출판사 초록·학회 초록). 아래 참고문헌은 모두 DOI 또는 arXiv ID가 API에서 **직접 확인된 것**만 "✔"로 표기했다. 확인하지 못한 항목은 "(DOI 미검증)" 또는 "(저자 미확인)"으로 적었다. "미발견"은 "검색에서 못 찾음"이지 "존재하지 않음"이 아니며, 어느 쪽인지 각 절에서 구분했다.

판정 범례: **표준(기존)** = 해당 분야에서 이미 널리 쓰이는 방법 / **타 분야 차용·핵융합 미발견** = ML·신호처리에서는 확립됐으나 핵융합 ML 문헌에서 적용 사례를 못 찾음 / **신규** = 두 분야 모두에서 같은 형태를 찾지 못함.

---

## 요약표

| 우리 방법 | 가장 가까운 선행 | 판정 |
|---|---|---|
| (a) 학습된 LSTM을 가중치가 아니라 Jacobian·임펄스/계단 응답·망각 게이트 통계로 읽기 | Sussillo & Barak 2013, Maheswaranathan 2019, Smith–Linderman–Sussillo 2021 (JSLDS), Tallec & Ollivier 2018 (게이트=시간상수) | ML에서는 **표준(기존)**; 핵융합 RNN에 적용한 사례 **미발견** → "타 분야 차용·핵융합 미발견" |
| (b) 망의 계산을 20모수 닫힌 식(EWMA + 간격 의존 Kalman형 이득 + 고역통과 선형 응답 + 3σ 기각)으로 적기 | Cranmer 2020(심볼릭 증류), Brunton 2016(SINDy), Allen & Bishop 1992(핵융합, 가중치 대신 응답으로 읽음), Murari 2020(핵융합, 데이터→식 직접) | 학습된 **핵융합 예측기**에서 닫힌 식을 추출해 재학습·검정한 사례 **미발견** → **신규**(방법 요소는 차용) |
| (c) 식을 동일 프로토콜로 처음부터 학습해 "≥70 % 기술 보존" 사전등록 규칙으로 검정 | 증류 문헌은 충실도(fidelity)를 재지만 사전등록 문턱은 없음 | **신규**(사전등록 + 비율 규칙) |
| (d) hold-or-jump 잠재과정 + Student-t 측정잡음 "물리 필터" | Roth 2013, Agamennoni 2012, Huang 2017(Student-t KF), Blom & Bar-Shalom 1988(IMM), Adams & MacKay 2007(BOCPD), Fisch 2021 | 신호처리 **표준(기존)**; 토카막 **진단 시계열**(특히 CES)에 적용 **미발견** → "타 분야 차용·핵융합 미발견" |
| (e) Student-t / 혼합 분포 head를 NLL로 학습, CRPS·coverage로 채점 | Chung–Char 2021(핵융합 과제 포함 보정), Char 2024(DIII-D coverage), Kim 2024(KSTAR BNN), Poels 2025, Gopakumar 2026(conformal) | 보정·coverage는 핵융합 ML에 **기존**; Student-t/혼합 head + CRPS 조합은 **미발견** → "타 분야 차용·핵융합 미발견" |
| (f) 물리에서 수치 예측을 먼저 적고 동결 모델에서 검사하는 "유도 문서" | Bonalumi 2024(XAI↔물리 정합성), Rossi 2024(물리 손실), Shen 2023(물리 유도 특징) | 사전 정량 예측 → 동결 모델 검정 형식은 **미발견** → **신규** |

---

## Q1. 학습된 RNN/LSTM을 Jacobian·임펄스 응답·게이트 통계로 읽기 — 그리고 핵융합 ML의 해석

### 1-1. 일반 ML(신경과학·NLP 계열)에서의 선례 — 표준(기존)

우리가 쓴 세 가지 도구(Jacobian ∂y_t/∂x_{t−k,c}, 임펄스/계단 응답, 게이트 통계로 본 유효 기억)는 모두 "역공학(reverse engineering) RNN" 문헌의 표준 도구다.

- **Sussillo D., Barak O.** "Opening the Black Box: Low-Dimensional Dynamics in High-Dimensional Recurrent Neural Networks," *Neural Computation* 25(3) 626–649 (2013). DOI 10.1162/neco_a_00409 ✔. 고정점 탐색과 그 주변 **선형화(Jacobian)** 로 RNN 계산을 읽는 출발점.
- **Maheswaranathan N., Williams A.H., Golub M.D., Ganguli S., Sussillo D.** "Reverse engineering recurrent networks for sentiment classification reveals line attractor dynamics," NeurIPS 2019, arXiv:1906.10720 ✔. 학습된 LSTM/GRU가 사실상 **선형 적분기(line attractor)** 라는 것을 Jacobian 고유구조로 보임 — 우리의 "EWMA" 읽기와 같은 종류의 결론.
- **Maheswaranathan N. et al.** "Universality and individuality in neural dynamics across large populations of recurrent networks," NeurIPS 2019, arXiv:1907.08549 ✔.
- **Smith J.T.H., Linderman S.W., Sussillo D.** "Reverse engineering recurrent neural networks with Jacobian switching linear dynamical systems," NeurIPS 2021, arXiv:2111.01256 ✔. **Jacobian 자체를 해석 가능한 대리 모델(구간별 선형계)로 만드는** 방법 — 우리의 "Jacobian을 읽어 선형 커널을 적는다"는 절차에 가장 가깝다.
- **Karpathy A., Johnson J., Fei-Fei L.** "Visualizing and Understanding Recurrent Networks," arXiv:1506.02078 (2015) ✔. 게이트 포화 통계(saturation plots)로 LSTM 셀의 역할을 읽음 — 우리의 망각 게이트 통계와 같은 계열.
- **Tallec C., Ollivier Y.** "Can recurrent neural networks warp time?" ICLR 2018, arXiv:1804.11188 ✔. **망각 게이트 = 시간상수(유효 기억 길이)** 라는 해석의 이론적 근거. 우리의 "유효 기억 1.3 스텝"은 이 해석을 그대로 적용한 것.
- **Krishnamurthy K., Can T., Schwab D.J.** "Theory of Gating in Recurrent Neural Networks," *Phys. Rev. X* 12, 011011 (2022). DOI 10.1103/PhysRevX.12.011011 ✔.
- **Strobelt H. et al.** "LSTMVis," *IEEE TVCG* (2017). DOI 10.1109/TVCG.2017.2744158 ✔. 은닉상태 시각화.
- 속성(attribution) 쪽: **Sundararajan M., Taly A., Yan Q.** "Axiomatic Attribution for Deep Networks," ICML 2017, arXiv:1703.01365 ✔ (integrated gradients). **Ismail A.A. et al.** "Benchmarking Deep Learning Interpretability in Time Series Predictions," NeurIPS 2020, arXiv:2010.13924 ✔ — 시계열 saliency가 **시간축에서는 신뢰도가 낮다**는 벤치마크 결과; 우리가 단일 saliency 대신 전 행 Jacobian + 응답 함수를 쓴 것의 정당화에 인용 가능. **Adebayo J. et al.** "Sanity Checks for Saliency Maps," NeurIPS 2018, arXiv:1810.03292 ✔.

판정: **표준(기존)**. 단, 위 문헌은 거의 전부 분류·인지 과제용 저차원 RNN이고, 물리 회귀 시계열에 "전 시험행 Jacobian → 지연 커널 → 닫힌 식"까지 밀고 간 형태는 Q2 판정에 포함.

### 1-2. 핵융합 ML의 해석 — 특징 중요도·SHAP·occlusion은 있으나 RNN 동역학 읽기는 미발견

- **Rea C., Montes K.J., Erickson K.G., Granetz R.S., Tinguely R.A.** "A real-time machine learning-based disruption predictor in DIII-D," *Nucl. Fusion* 59, 096016 (2019). DOI 10.1088/1741-4326/ab28bf ✔. 랜덤포레스트(DPRF)의 **feature contribution**을 실시간으로 제공.
- **Montes K.J. et al.** "Machine learning for disruption warnings on Alcator C-Mod, DIII-D, and EAST," *Nucl. Fusion* 59, 096015 (2019). DOI 10.1088/1741-4326/ab1df4 ✔.
- **Rea C., Montes K.J., Pau A., Granetz R.S., Sauter O.** "Progress Toward Interpretable Machine Learning-Based Disruption Predictors Across Tokamaks," *Fusion Sci. Technol.* 76(8) 912–924 (2020). DOI 10.1080/15361055.2020.1798589 ✔.
- **Zhu J.X., Rea C., Montes K., Granetz R.S., Sweeney R., Tinguely R.A.** "Hybrid deep-learning architecture for general disruption prediction across multiple tokamaks," *Nucl. Fusion* 61, 026007 (2021). DOI 10.1088/1741-4326/abc664 ✔.
- **Churchill R.M., Tobias B., Zhu Y., DIII-D team** "Deep convolutional neural networks for multi-scale time-series classification and application to tokamak disruption prediction using raw, high temporal resolution diagnostic data," *Phys. Plasmas* 27, 062510 (2020). DOI 10.1063/1.5144458 ✔. 원시 ECEi 입력 — 우리 BES/ECEI 고속 채널 사용의 선례.
- **Aymerich E. et al.** "Disruption prediction at JET through deep convolutional neural networks using spatiotemporal information from plasma profiles," *Nucl. Fusion* 62, 066005 (2022). DOI 10.1088/1741-4326/ac525e ✔.
- **Bonalumi L., Aymerich E., Alessi E., et al.** "eXplainable artificial intelligence applied to algorithms for disruption prediction in tokamak devices," *Front. Phys.* 12, 1359656 (2024). DOI 10.3389/fphy.2024.1359656 ✔. **occlusion과 saliency map** 두 방법(Grad-CAM 아님). 핵융합 CNN 해석의 대표 사례.
- **Shen C., Zheng W., Ding Y., et al.** "IDP-PGFE: an interpretable disruption predictor based on physics-guided feature extraction," *Nucl. Fusion* 63, 046024 (2023). DOI 10.1088/1741-4326/acbe0f ✔.
- **Liu D., Zhu X.L., Jiang Y., et al.** "An interpretable disruption predictor on EAST using improved XGBoost and SHAP," *Nucl. Fusion* (2025). DOI 10.1088/1741-4326/adeea8 ✔.
- **Qiu R., Yang Z., Xia F., et al.** "A trustworthy disruption prediction framework on HL-3: … real-time SHAP for interpretability," *Plasma Sci. Technol.* (2026). (DOI 미검증 — OpenAlex 결과가 잘림.)
- **Kim J., Lee J., Seo J., et al.** "Enhancing disruption prediction through Bayesian neural network in KSTAR," *Plasma Phys. Control. Fusion* 66, 075001 (2024). DOI 10.1088/1361-6587/ad48b7 ✔.
- **Jalalvand A., Kaptanoglu A.A., Garcia A.V., et al.** "Alfvén eigenmode classification based on ECE diagnostics at DIII-D using deep recurrent neural networks," *Nucl. Fusion* 62, 026007 (2022). DOI 10.1088/1741-4326/ac3be7 ✔. (reservoir RNN; 동역학 해석은 하지 않음.)
- **Joung S., Smith D.R., McKee G.R., et al.** "Tokamak edge localized mode onset prediction with deep neural network and pedestal turbulence," *Nucl. Fusion* (2024). DOI 10.1088/1741-4326/ad43fb ✔. BES 입력 선례.
- **Pyragius T., Colgan C., Lowe H., et al.** "Application of interpretable machine learning for cross-diagnostic inference on the ST40 spherical tokamak," arXiv:2407.18741 (2024) ✔. 블랙박스를 "grey-box"로 바꾸는 모델 불가지론적 방법(SXR → T_e, n_e).
- 1992년 초기 선례(Q3에서 상술): **Allen L., Bishop C.M.** "Neural network approach to energy confinement scaling in Tokamaks," *Plasma Phys. Control. Fusion* 34, 1291–1302 (1992). 가중치 대신 **출력 대 입력 응답**으로 망을 읽었다.

**학습된 핵융합 예측기에서 닫힌 식을 추출한 사례: 미발견.** 검색어 "closed-form equation extracted trained neural network tokamak", "symbolic regression neural network surrogate tokamak distill" 모두 해당 없음. 가장 가까운 것은 (i) Murari 그룹의 심볼릭 회귀(데이터 → 식, 망을 거치지 않음, Q2), (ii) Pyragius 2024의 grey-box(식이 아니라 모델 불가지론적 설명), (iii) Allen & Bishop 1992(응답 읽기, 식 추출은 명시적으로 포기). 핵융합 ML 전반에서 해석은 **특징 중요도(트리 모델)·SHAP·occlusion/saliency(CNN)** 에 머물러 있고, **RNN의 Jacobian·임펄스 응답·게이트 통계**를 쓴 논문은 찾지 못했다.

**판정: (a)는 "타 분야 차용·핵융합 미발견"; "닫힌 식 추출"은 Q2로 이월(신규).**

---

## Q2. 신경망의 해석 가능 모델로의 증류 / 식 발견 — 토카막 시계열 적용

### 2-1. 일반 방법 — 표준(기존)

- **Brunton S.L., Proctor J.L., Kutz J.N.** "Discovering governing equations from data by sparse identification of nonlinear dynamical systems," *PNAS* 113(15) 3932–3937 (2016). DOI 10.1073/pnas.1517384113 ✔.
- **Champion K., Lusch B., Kutz J.N., Brunton S.L.** "Data-driven discovery of coordinates and governing equations," *PNAS* 116(45) 22445 (2019). DOI 10.1073/pnas.1906995116 ✔. (오토인코더 + SINDy: 신경망 안에서 식을 찾는 하이브리드.)
- **Lusch B., Kutz J.N., Brunton S.L.** "Deep learning for universal linear embeddings of nonlinear dynamics," *Nat. Commun.* 9, 4950 (2018). DOI 10.1038/s41467-018-07210-0 ✔. (Koopman 선형화 — "비선형 망의 계산이 사실은 선형"이라는 읽기의 이론 축.)
- **Udrescu S.-M., Tegmark M.** "AI Feynman: A physics-inspired method for symbolic regression," *Sci. Adv.* 6, eaay2631 (2020). DOI 10.1126/sciadv.aay2631 ✔.
- **Cranmer M., Sanchez-Gonzalez A., Battaglia P., et al.** "Discovering Symbolic Models from Deep Learning with Inductive Biases," NeurIPS 2020, arXiv:2006.11287 ✔. **학습된 망(GNN)의 내부 메시지에 심볼릭 회귀를 맞춰 닫힌 식을 얻고, 그 식이 망만큼 일반화함을 보임** — "망에서 식을 읽어 재학습·검정"이라는 우리 절차의 가장 가까운 일반 ML 선례.
- **Revach G., Shlezinger N., Ni X., Escoriza A.L., van Sloun R.J.G., Eldar Y.C.** "KalmanNet: Neural Network Aided Kalman Filtering for Partially Known Dynamics," *IEEE Trans. Signal Process.* 70, 1532–1547 (2022). DOI 10.1109/TSP.2022.3158588 ✔. (반대 방향: 칼만 이득만 RNN으로 대체. 우리는 거꾸로 RNN에서 이득 K(s)를 읽어냄.)
- RNN↔칼만 해석: **Krishnan R.G., Shalit U., Sontag D.** "Deep Kalman Filters," arXiv:1511.05121 (2015) ✔; **Coskun H. et al.** "Long Short-Term Memory Kalman Filters," ICCV 2017, DOI 10.1109/ICCV.2017.589 ✔; **Goel G., Bartlett P.** "Can a Transformer Represent a Kalman Filter?" arXiv:2312.06937 (2023) ✔.
- 화이트박스 대리모델: **Frosst N., Hinton G.** "Distilling a Neural Network Into a Soft Decision Tree," arXiv:1711.09784 (2017) ✔; **Wu M., Hughes M.C., Parbhoo S., et al.** "Beyond Sparsity: Tree Regularization of Deep Models for Interpretability," AAAI 2018, DOI 10.1609/aaai.v32i1.11501 ✔.

### 2-2. 토카막 적용 — 식 발견은 데이터에서 직접, 망 증류는 미발견

- **Murari A., Lupelli I., Gelfusa M., Gaudio P.** "Non-power law scaling for access to the H-mode in tokamaks via symbolic regression," *Nucl. Fusion* 53, 043001 (2013). DOI 10.1088/0029-5515/53/4/043001 ✔.
- **Murari A., Peluso E., Gelfusa M., et al.** "A new approach to the formulation and validation of scaling expressions for plasma confinement in tokamaks," *Nucl. Fusion* 55, 073009 (2015). DOI 10.1088/0029-5515/55/7/073009 ✔.
- **Murari A., Peluso E., Lungaroni M., Rossi R., Gelfusa M., JET Contributors** "Investigating the Physics of Tokamak Global Stability with Interpretable Machine Learning Tools," *Appl. Sci.* 10(19), 6683 (2020). DOI 10.3390/app10196683 ✔. 분류기(SVM/트리) + 유전 프로그래밍 심볼릭 회귀로 **운전 경계를 식으로** 표현. 데이터에서 직접 식을 찾으며, 학습된 심층망을 증류하지는 않음.
- **Lore J.D., De Pascuale S., Laiu P., et al.** "Time-dependent SOLPS-ITER simulations of the tokamak plasma boundary for model predictive control using SINDy," *Nucl. Fusion* 63, 046015 (2023). DOI 10.1088/1741-4326/acbe0e ✔. **시뮬레이션** 데이터에 SINDy.
- **Kaptanoglu A.A., Morgan K.D., Hansen C.J., Brunton S.L.** "Physics-constrained, low-dimensional models for magnetohydrodynamics: First-principles and data-driven approaches," *Phys. Rev. E* 104, 015206 (2021). DOI 10.1103/PhysRevE.104.015206 ✔.
- **Morosohk S.M., Pajares A., Schuster E.** "Estimation of the Electron Temperature Profile in Tokamaks Using Analytical and Neural Network Models," ACC 2022, DOI 10.23919/ACC53348.2022.9867844 ✔; **Morosohk & Schuster** *Contrib. Plasma Phys.* 63 (2023) DOI 10.1002/ctpp.202200153 ✔. EKF 관측기 안에 신경망 서브모델 — KalmanNet 방향의 토카막 사례는 **미발견**(학습된 이득 없음).

**결론: "학습된 심층 예측기 → 닫힌 식 → 같은 프로토콜로 재학습 → 사전등록 비율 규칙으로 기술 보존 검정"을 토카막 실험 시계열에서 수행한 논문은 찾지 못했다.** 2025년 플라즈마 closure 리뷰(arXiv 2511.22486, 저자 미확인)도 "일반 심볼릭 회귀는 아직 플라즈마 closure에 적용되지 않았다"고 적고 있어 공백이 문헌에서도 인지돼 있다.

**판정: (b)·(c) = 신규** (구성요소인 SINDy·심볼릭 증류·칼만 이득 해석은 타 분야 표준이므로, 논문에서는 "Cranmer 2020·Smith 2021의 절차를 물리 회귀 시계열에 적용하고, 사전등록 보존 규칙을 추가했다"고 계보를 적는 것이 안전).

---

## Q3. 가중치 크기 ≠ 중요도

### 3-1. 일반 ML — 표준(기존)

- **LeCun Y., Denker J.S., Solla S.A.** "Optimal Brain Damage," NIPS 2, 598–605 (1990). (학회 논문집, DOI 없음.) 손실의 2차 미분(saliency)이 크기보다 나은 중요도.
- **Hassibi B., Stork D.G.** "Second order derivatives for network pruning: Optimal Brain Surgeon," NIPS 5 (1993). (DOI 없음.)
- **Molchanov P., Tyree S., Karras T., Aila T., Kautz J.** "Pruning Convolutional Neural Networks for Resource Efficient Inference," ICLR 2017, arXiv:1611.06440 ✔. **Taylor 전개(활성×기울기) 기준이 가중치 노름 기준을 이긴다**고 직접 비교.
- **Li H., Kadav A., Durdanovic I., Samet H., Graf H.P.** "Pruning Filters for Efficient ConvNets," ICLR 2017, arXiv:1608.08710 ✔ (노름 기준의 대표; 대조용).
- **Adebayo J. et al.** 2018 (위) — 가중치 무작위화 sanity check.
- 생태학·임상 쪽의 "connection weights" 전통과 그 한계: **Olden J.D., Jackson D.A.** "Illuminating the 'black box': a randomization approach for understanding variable contributions in artificial neural networks," *Ecol. Model.* 154, 135–150 (2002) (DOI 10.1016/S0304-3800(02)00064-9, OpenAlex 표시 기준 — 미검증); **Olden J.D., Joy M.K., Death R.G.** "An accurate comparison of methods for quantifying variable importance in artificial neural networks using simulated data," *Ecol. Model.* 178, 389–397 (2004) (DOI 미검증). **Garson G.D.** "Interpreting neural-network connection weights," *AI Expert* 6(4), 46–51 (1991) (DOI 없음) — 가중치 기반 중요도의 원조.

### 3-2. 핵융합 — 1층 가중치를 중요도로 보고한 논문: 미발견; 1992년에 이미 "가중치는 통찰이 없다"고 적은 선례 있음

- **Allen L., Bishop C.M.** "Neural network approach to energy confinement scaling in Tokamaks," *Plasma Phys. Control. Fusion* 34(7), 1291–1302 (1992). (OpenAlex 표시 DOI 10.1016/0741-3335(92)90121-L — 형식이 이례적이라 미검증; 서지는 Microsoft Research 사본으로 확인.) 본문 인용(PDF 원문 추출): *"Although an analytic form can be written down using equations (2) and (4), together with the values of the weights and thresholds from the trained network, this would provide little useful insight. Instead, we compare the LR and neural network results by plotting ln(τ_E/τ_E0) versus the regressor variables …"* — 즉 **가중치로 쓴 식을 버리고 응답(출력 대 입력)으로 읽는다**는 우리 입장과 동일. 또 선형 회귀가 틀린 지수(a4)를 줄 수 있음을 지적. 논문에서 "핵융합 ML 초기부터 가중치 해석을 피했다"는 대조로 인용 가능.
- 2007 HL-2A 붕괴 예측(*Chinese Physics B* 16(12) 3738, DOI 10.1088/1009-1963/16/12/030, 저자 미확인)은 입력 중요도를 **saliency analysis**(기울기)로 매김 — 가중치 아님.
- 가중치(1층 노름)를 중요도로 보고한 핵융합 논문: 웹 검색 "first layer / input layer weights feature importance tokamak"에서 **해당 없음**. 따라서 "대조할 핵융합 논문"은 제시할 수 없고, 대조는 일반 ML(Garson 1991·Li 2017 노름 기준)과 하는 것이 정확하다. (존재하지 않는다는 증명은 아님 — 학회 포스터나 비영어 문헌에 있을 수 있음.)

**판정: 표준(기존)** — 우리가 "가중치 노름은 정보가 없다"고 측정으로 보인 것은 Molchanov 2017·OBD 결과의 재확인이며, 핵융합 쪽 대조는 Allen & Bishop 1992의 진술로 대신한다.

---

## Q4. 강건(Student-t) 칼만 필터·점프 상태 모델 — 플라즈마 진단 적용

### 4-1. 신호처리 — 표준(기존)

- **Agamennoni G., Nieto J.I., Nebot E.M.** "Approximate Inference in State-Space Models With Heavy-Tailed Noise," *IEEE Trans. Signal Process.* 60(10), 5024–5037 (2012). DOI 10.1109/TSP.2012.2208106 ✔.
- **Roth M., Özkan E., Gustafsson F.** "A Student's t filter for heavy tailed process and measurement noise," ICASSP 2013, 5770–5774. DOI 10.1109/ICASSP.2013.6638770 ✔.
- **Piché R., Särkkä S., Hartikainen J.** "Recursive outlier-robust filtering and smoothing for nonlinear systems using the multivariate Student-t distribution," MLSP 2012. DOI 10.1109/MLSP.2012.6349794 ✔.
- **Aravkin A.Y., Burke J.V., Pillonetto G.** "Robust and Trend-Following Student's t Kalman Smoothers," *SIAM J. Control Optim.* 52(5) (2014). DOI 10.1137/130918861 ✔.
- **Huang Y., Zhang Y., Li N., Wu Z., Chambers J.A.** "A Novel Robust Student's t-Based Kalman Filter," *IEEE Trans. Aerosp. Electron. Syst.* 53(3), 1545–1554 (2017). DOI 10.1109/TAES.2017.2651684 ✔. **Huang Y., Zhang Y., Zhao Y., Chambers J.A.** "A Novel Robust Gaussian–Student's t Mixture Distribution Based Kalman Filter," *IEEE Trans. Signal Process.* 67(13) (2019). DOI 10.1109/TSP.2019.2916755 ✔ — 우리 "혼합 head"와 같은 Gaussian+t 혼합.
- 점프/스위칭: **Blom H.A.P., Bar-Shalom Y.** "The interacting multiple model algorithm for systems with Markovian switching coefficients," *IEEE Trans. Autom. Control* 33(8), 780–783 (1988). DOI 10.1109/9.1299 ✔. **Adams R.P., MacKay D.J.C.** "Bayesian Online Changepoint Detection," arXiv:0710.3742 (2007) ✔. **Fisch A.T.M., Eckley I.A., Fearnhead P.** "Innovative and Additive Outlier Robust Kalman Filtering With a Robust Particle Filter," *IEEE Trans. Signal Process.* 70 (2022). DOI 10.1109/TSP.2021.3125136 ✔ — "혁신 이상치(진짜 점프) vs 가산 이상치(측정 오류)"의 구분이 우리 hold-or-jump + t 잡음 구도와 정확히 대응.

### 4-2. 토카막 — 칼만 관측기는 제어 분야에서 확립; Student-t·점프 모델은 진단 시계열에 미적용

제어·상태 재구성(모두 가우스 칼만/EKF):
- **Felici F., Sauter O., Coda S., et al.** "Real-time physics-model-based simulation of the current density profile in tokamak plasmas," *Nucl. Fusion* 51, 083052 (2011). DOI 10.1088/0029-5515/51/8/083052 ✔ (RAPTOR).
- **Felici F., de Baar M., Steinbuch M.** "A dynamic state observer for real-time reconstruction of the tokamak plasma profile state and disturbances," ACC 2014, 4816–4823. DOI 10.1109/ACC.2014.6858781 ✔.
- **Blanken T.C., Felici F., Galperti C., et al.** "Real-time plasma state monitoring and supervisory control on TCV," *Nucl. Fusion* 59, 026017 (2019). DOI 10.1088/1741-4326/aaf451 ✔.
- **Messmer M., Felici F., Sauter O., et al.** "Optimal MSE polarisation angle and q-profile estimation using Kalman filters and the plasma simulator RAPTOR," *Plasma Phys. Control. Fusion* (2018). DOI 10.1088/1361-6587/aaf7ed ✔.
- **Bosman T.O.S.J., Kudláček O., Fable E., et al.** "Kalman filter density reconstruction in ICRH discharges on ASDEX Upgrade," *Fusion Eng. Des.* 170, 112510 (2021). DOI 10.1016/j.fusengdes.2021.112510 ✔.
- **Van Mulders S., Felici F., McIntosh S., et al.** "Model-based estimation of tokamak plasma profiles and physics parameters: algorithm overview and application to ITER," *Nucl. Fusion* 65, 066006 (2025). DOI 10.1088/1741-4326/add16e ✔.
- **di Grazia L.E., Mattei M., Pironti A., et al.** "An Extended Kalman Filter approach for tokamak plasma equilibrium reconstruction," *Fusion Eng. Des.* (2025). DOI 10.1016/j.fusengdes.2025.115363 ✔.
- **In Y., Kim J.S., Edgell D.H., et al.** "Model-based dynamic resistive wall mode identification and feedback control in the DIII-D tokamak," *Phys. Plasmas* 13, 062512 (2006). DOI 10.1063/1.2214637 ✔. 칼만 필터로 **ELM을 RWM과 분리** — ELM/sawtooth가 "델타형 사건"이라 가우스 가정이 깨진다는 지적이 후속 APS 포스터(2016)에 있음(학회 초록, 미검증).
- ELM·confinement 전이 HMM: IAEA FEC 2020/21 contribution #17698 "hidden Markov model of wavelet coefficients"(저자 미확인, 페이지 403) — 플라즈마에서 **상태 전환 모델**을 쓴 드문 사례.

강건(Student-t) 우도 — 칼만이 아니라 **정적 프로파일 피팅**에서만:
- **Fujii K., Suzuki C., Hasuo M.** "Robust Regression for Automatic Fusion Plasma Analysis Based on Generative Modeling," *IEEE Trans. Plasma Sci.* 47(7), 3305–3314 (2019). DOI 10.1109/TPS.2019.2921073 ✔ (LHD Thomson, Student-t).
- **Leddy J., Madireddy S., Howell E.C., Kruger S.E.** "Single Gaussian process method for arbitrary tokamak regimes with a statistical analysis," *Plasma Phys. Control. Fusion* (2022). DOI 10.1088/1361-6587/ac89ab ✔ (Student-t 우도 GP).
- **Fischer R., Bock A., Denk S.S., et al.** "Integrated Data Analysis and Validation," arXiv:2411.09270 (2024) ✔ (AUG에서 T_e, n_e 프로파일에 Student-t 우도 상용).
- **Kim M., Ko W.H., Kwak S., et al.** "Kinetic profile inference with outlier detection using support vector machine regression and Gaussian process regression," *Nucl. Fusion* 64, 106052 (2024). DOI 10.1088/1741-4326/ad7304 ✔ — **KSTAR** 운동학 프로파일(CES 포함 가능성)에서 이상치 처리; 시간 방향 필터는 아님.

**CES(전하교환분광) 시계열에 칼만/강건 필터를 적용한 논문: 미발견.** "jump-diffusion"/"hold-or-jump" 잠재과정을 토카막 신호에 쓴 논문: 미발견. Student-t 칼만을 토카막 진단에 쓴 논문: 미발견.

**판정: (d) = 타 분야 차용·핵융합 미발견.** 논문에서는 "Fisch 2021의 혁신/가산 이상치 구분과 Huang 2019의 Gaussian–t 혼합을, Fujii 2019·Fischer 2024가 정적 피팅에서 쓰던 Student-t 우도와 결합해 시간 방향 CES 필터로 만들었다"로 계보를 적을 수 있다.

---

## Q5. 핵융합 ML의 보정된 확률 회귀

- **Chilenski M.A., Greenwald M., Marzouk Y., et al.** "Improved profile fitting and quantification of uncertainty in experimental measurements of impurity transport coefficients using Gaussian process regression," *Nucl. Fusion* 55, 023012 (2015). DOI 10.1088/0029-5515/55/2/023012 ✔; **Chilenski et al.** *Nucl. Fusion* (2017) DOI 10.1088/1741-4326/aa8387 ✔.
- **Pavone A., Svensson J., Langenberg A., et al.** "Bayesian uncertainty calculation in neural network inference of ion and electron temperature profiles at W7-X," *Rev. Sci. Instrum.* 89 (2018). DOI 10.1063/1.5039286 ✔; "Neural network approximation of Bayesian models for the inference of ion and electron temperature profiles at W7-X," *Plasma Phys. Control. Fusion* 61, 075012 (2019). DOI 10.1088/1361-6587/ab1d26 ✔; **Pavone, Svensson, Kwak, et al.** *PPCF* 62, 045019 (2020). DOI 10.1088/1361-6587/ab7732 ✔. — **이온 온도 프로파일의 불확도**를 다룬 가장 가까운 선례(W7-X, 정적 추론).
- **Chung Y., Neiswanger W., Char I., Schneider J.** "Beyond Pinball Loss: Quantile Methods for Calibrated Uncertainty Quantification," NeurIPS 2021, arXiv:2011.09588 ✔. 평가 과제에 **핵융합 고차원 UQ 과제** 포함; 보정–예리함(sharpness) 트레이드오프.
- **Char I., Chung Y., Abbate J., Kolemen E., Schneider J.** "Full Shot Predictions for the DIII-D Tokamak via Deep Recurrent Networks," arXiv:2404.12416 (2024) ✔. 앙상블 불확도, **90 % 구간 coverage와 miscalibration area**로 보정 평가; 예측 지평이 길어질수록 과소신뢰(under-confident) 보고. CRPS는 쓰지 않음.
- **Kim J., Lee J., Seo J., et al.** *PPCF* 66, 075001 (2024) ✔ (KSTAR, BNN 붕괴 예측 — 불확도 출력).
- **Joung S.** "Bayesian neural network for plasma equilibria in KSTAR," arXiv:2301.11555 (2023) ✔.
- **Poels Y., Venturini C., Pau A., et al.** "Robust confinement state classification with uncertainty quantification through ensembled data-driven methods," *Nucl. Fusion* (2025). DOI 10.1088/1741-4326/adf349 ✔.
- **Gopakumar V., Gray A., Oskarsson J., et al.** "Uncertainty quantification of surrogate models using conformal prediction," *Mach. Learn.: Sci. Technol.* 7, 015025 (2026). DOI 10.1088/2632-2153/ae2e7b ✔.
- **Mathews A., Hughes J.W.** "Quantifying Experimental Edge Plasma Evolution Via Multidimensional Adaptive Gaussian Process Regression," *IEEE Trans. Plasma Sci.* (2021). DOI 10.1109/TPS.2021.3123046 ✔.
- **Leddy 2022** (Student-t GP, 위).
- **Seo J.** "Quantum Mixture-Density Network for Multimodal Probabilistic Prediction," *IEEE Access* 13, 199317–199325 (2025). DOI 10.1109/ACCESS.2025.3635294 ✔ — 핵융합 연구자의 MDN 논문이나 **토카막 데이터 적용 여부는 초록에서 확인 못 함**(인용 시 주의).

**Student-t 또는 혼합 분포 head를 NLL로 학습하고 CRPS로 채점한 토카막 회귀 논문: 미발견.** 웹 검색 "CRPS tokamak"은 0건(coverage·ECE·miscalibration area만 사용). "mixture density network tokamak"은 EPS 2025 포스터(MAST 이온화 분포, 저자 미확인) 1건뿐.

**판정: (e) = 타 분야 차용·핵융합 미발견.** 보정 평가(coverage) 자체는 Char 2024·Chung 2021로 핵융합 ML에 기존. Student-t/혼합 NLL head와 CRPS 채점은 기상 예보·통계 ML의 표준이며 핵융합 적용은 우리가 처음일 가능성이 높다(단 Seo 2025 본문 확인 권장).

---

## Q6. 물리에서 정량 예측을 먼저 적고 동결 ML 모델에서 검사하기

검색어: "physics consistency test machine learning tokamak", "does the network learn physics", "physically consistent verification learned model", "hypothesis test machine learning model behaviour physics prediction tokamak". **동일 형식(사전 유도 → 수치 예측 → 동결 모델에서 적중/실패 기록) 미발견.**

가장 가까운 것:
- **Bonalumi et al. 2024** (위, ✔): "XAI가 내놓는 설명과 물리 모델 사이에 정합성이 있어야 한다"는 원칙을 세우고 occlusion/saliency로 사후 확인 — **사후·정성**.
- **Rossi R., Gelfusa M., Craciunescu T., Murari A.** "A hybrid physics/data-driven logic to detect, classify, and predict anomalies and disruptions in tokamak plasmas," *Nucl. Fusion* 64, 046017 (2024). DOI 10.1088/1741-4326/ad2723 ✔ — 물리를 **손실에 넣는** 방향(검정이 아니라 제약).
- **Shen et al. 2023 IDP-PGFE** (✔) — 물리 유도 특징으로 해석 가능성 확보.
- **Murari et al. 2020** (✔) — 데이터에서 찾은 식을 Hugill·β 한계 같은 **기존 물리 경계와 비교**.
- **Allen & Bishop 1992** — 망의 응답을 이론 스케일링(ρ*/a 멱법칙 제약)과 비교하고, 선형 회귀의 지수 왜곡을 지적.
- 일반 ML 쪽에서는 **Adebayo 2018(sanity check)** 과 **Ismail 2020(벤치마크)** 가 "해석 방법 자체를 검정한다"는 정신에서 가장 가깝다.
- (참고) 우리 유도 문서의 재료인 OU 앵커·광자 예산 MNAR·운동량 균형·ELM 회복 시간척도에 대해 "Ornstein-Uhlenbeck 토카막 회전/요동" 검색은 **해당 없음**(SOL 요동은 Garcia 계열의 지수 펄스 중첩 모델이 표준 — 다른 종류의 확률과정).

**판정: (f) = 신규.** 논문에서는 "Bonalumi 2024가 요구한 XAI–물리 정합성을, 사후 설명이 아니라 사전 정량 예측과 동결 모델 검정으로 수행했다"로 위치시키면 된다.

---

## 논문 서술에 쓸 한 줄 계보(제안)

1. 해석: Sussillo & Barak 2013 → Maheswaranathan 2019 → Smith 2021(JSLDS)의 Jacobian 역공학을 물리 회귀 LSTM에 적용; 게이트=시간상수는 Tallec & Ollivier 2018; 가중치 노름의 무정보성은 Molchanov 2017·OBD와 Allen & Bishop 1992의 진술과 일치.
2. 식 추출·검정: Cranmer 2020의 "망 → 식 → 재학습" 절차 + SINDy 정신(Brunton 2016); 토카막에서 식 발견은 Murari 2013/2015/2020(데이터 직접)·Lore 2023(시뮬레이션)뿐 → 학습된 실험 예측기 증류 + 사전등록 보존 비율은 처음.
3. 물리 필터: Fisch 2021(혁신/가산 이상치) + Huang 2019(Gaussian–t 혼합) + Fujii 2019·Fischer 2024(핵융합 정적 피팅의 Student-t 우도)를 시간 방향 CES 필터로 결합; 토카막 제어의 칼만 관측기(Felici 2011·Blanken 2019·Van Mulders 2025)는 모두 가우스.
4. 확률 head: 보정 평가는 Chung 2021·Char 2024 계보, CRPS·Student-t/혼합 NLL은 핵융합에 처음(Seo 2025 본문 확인 후 단정).
5. 유도 문서: Bonalumi 2024의 정합성 요구를 사전등록·정량 검정 형식으로 — 선례 없음.

## 미해결·주의
- OpenAlex는 조사 중반부터 IP 차단(429)되어 Q5·Q6 일부는 웹 검색·Crossref만으로 확인했다. 미발견 판정 6건(핵융합 예측기 닫힌 식 추출, 1층 가중치 중요도 핵융합 논문, CES 칼만, Student-t 칼만·점프 모델 토카막 적용, Student-t/혼합 head+CRPS 핵융합, 사전 물리 예측 검정)은 **각각 2개 이상 독립 검색어로 0건**이었으나 학회 포스터·비영어 문헌은 포괄하지 못했다.
- DOI 미검증 표시 항목: Olden 2002/2004, Allen & Bishop 1992(OpenAlex DOI 형식 이례), Qiu 2026, IAEA #17698, EPS 2025 MDN 포스터, 2007 HL-2A saliency 논문 저자.
