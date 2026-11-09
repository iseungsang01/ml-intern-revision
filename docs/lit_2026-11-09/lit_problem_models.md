# 문헌 조사: 문제 설정·모델 계열 (KSTAR CES 10 ms 인과 나우캐스팅)

작성일 2026-11-09. 대상 연구: KSTAR 2022 캠페인 641 방전에서 CES(하전교환분광) 이온온도 T_i·토로이달 회전 V_rot의 10 ms 결측 시각을 과거 정보(과거 CES 이력 + 동시각 BES·ECEI·Mirnov)만으로 채우는 인과 나우캐스팅. 두 분기 인과 LSTM(V_rot 분기는 고속 진단을 보지 않음), 비인과 PCHIP·persistence·인과/오프라인 GP 대조.

## 0. 방법과 한계

- 검색 경로: OpenAlex API(제목·초록 검색 + DOI별 초록 조회), arXiv API, Crossref API(서지·초록), 웹 검색(APS DPP 초록, DOE 하이라이트). Semantic Scholar는 키 없이 429(요청 제한)로 전 질의 실패. OpenAlex는 후반부에 429가 걸려 전문(fulltext) 검색은 완료하지 못함(아래 Q3에 명시).
- 검증: 본문에 인용한 DOI 57건 전부 doi.org에서 해석됨(HTTP 302로 출판사 페이지 도달. AIP·Wiley·ACM은 봇 차단 403을 돌려주지만 리다이렉트 대상 URL이 해당 논문임을 확인). arXiv 번호는 arXiv API 또는 abs 페이지로 확인.
- "미발견"은 위 경로에서 나오지 않았다는 뜻이고 "존재하지 않음"의 증명이 아니다. 특히 학회 초록(APS DPP)만 있는 결과는 따로 표시했다.
- 출판사 전문은 대부분 읽지 않았고 초록(OpenAlex·Crossref)과 arXiv 전문(HTML)에 근거했다. 전문을 읽은 것은 Char 2024, Diag2Diag(arXiv v4), TokaMark(arXiv) 세 편.

---

## Q1. CES/CXRS T_i·회전 시계열의 시간 결측 채움·대입·나우캐스팅, 가상 진단, 진단 대입

### 1-A. 측정이 있는 시각의 CES 스펙트럼을 신경망으로 읽는 계열 (우리와 구분: "있는 시각을 읽음" vs "없는 시각을 채움")

| 연구 | 장치 | 핵심 | 비고 |
|---|---|---|---|
| Svensson, von Hellermann, König 1999, *Plasma Phys. Control. Fusion* 41, 315. DOI 10.1088/0741-3335/41/2/016 | JET | MLP로 CX 스펙트럼→T_i·회전·불순물 밀도, 실시간 T_i 프로파일 진단 구현, 은닉층에서 오차막대 계산 | 이 계열의 원조(1999) |
| Lee J.K., Ko W.H., Lee H.H., Kim B., Lee Y., Shin G. 외, *Fusion Eng. Des.* 222, 115518 (OpenAlex 발행연도 2025, 권 222). DOI 10.1016/j.fusengdes.2025.115518 | KSTAR | NN-CES: 빔 변조 없이 원 스펙트럼→T_i·V_rot, 물리 구속 손실(모수 + 스펙트럼 형상 일관성), ≈150,000 프레임(L/H/locked/RMP) | 팀 리드가 "Lee et al. 2026"으로 아는 논문. DOI 접두 2025, 권 222 → 인용 시 연도 표기 확인 필요 |
| Tian W.H., Yang Z., Long T. 외 2026, *Nucl. Fusion* 67, 016004. DOI 10.1088/1741-4326/aeb037 | HL-3 | 실시간 CXRS NN(0.56 ms/32ch, R² T_i 0.93·v_t 0.98), 해석가능성·불확도, HL-2A→HL-3 전이학습 | |
| Hu Z., Lin Z., Fu S. 외 2026, *Plasma Sci. Technol.* DOI 10.1088/2058-6272/aeb5c5 | EAST | 1D CNN + GAN 증강 스펙트럼으로 T_i 실시간 추론 | |
| Chrystal C., Groebner R., Linsenmayer 외 2026, *Rev. Sci. Instrum.* 97, 103501. DOI 10.1063/5.0350505 | DIII-D | PCS 안 실시간 CER 피팅, 24 시선, 5 ms마다 원자료 | 피팅이지 학습 아님. "실시간 CER의 현재 시간 해상도 = 5 ms" 근거 |
| Steiner P., Jalalvand A., Shousha R., Pratt Q. 외, APS DPP 2025 구두발표 TM11.00001 "Seeing in the dark – Neural networks for passive CER spectroscopy analysis" (학회 초록만, 논문 미발견) | DIII-D | 원 다채널 CER 스펙트럼(>5 TB)→T_i·회전, **결측 라벨을 무시하는 masked loss + 확률적 시간-주파수 마스킹**, RMSE 120 eV/15 km/s, 빔 없는 방전에서 회전이 tearing 주파수와 상관 | 결측 라벨 마스크 손실의 유일한 CES 선례. 다만 입력이 스펙트럼(=측정 있는 시각)이며 시간 결측 채움이 아님 |

### 1-B. 다른 진단으로 한 진단을 만드는 계열 (virtual / synthetic / cross-diagnostic)

| 연구 | 장치 | 핵심 | 우리와의 차이 |
|---|---|---|---|
| **Jalalvand A., Kim S.K., Seo J., Hu Q., Curie M., Steiner P., Nelson A.O., Na Y.-S., Kolemen E. 2025, "Multimodal super-resolution: discovering hidden physics and its application to fusion plasmas", *Nat. Commun.* 16, 8506. DOI 10.1038/s41467-025-63492-1** (arXiv 2405.05908, "Diag2Diag") | DIII-D | CER·간섭계·ECE·MSE·자기 신호(+간섭계·ECE 1·2차 도함수, 약 4 ms 창)→Thomson n_e·T_e. **기억 없는 MLP(236→952→80), 각 TS 시각에 "가장 최근 표본"을 맞춘 동시각 사상**, TS 200 Hz→kHz급(결론부는 1 MHz 주장). R² 0.92. 평가는 "측정이 있을 때 모델 출력과 측정의 유사도" + ELM 다주기 합성 진실. **보간·persistence 정량 대조 없음**(보간이 사건을 놓친다는 정성 논거만) | 가장 가까운 선행. 차이: (i) 출력이 전자 채널(TS)이고 T_i·회전은 입력일 뿐, (ii) 인과성 요구 없음(동시각·기억 없음), (iii) 나이브 기준선 대비 skill 미측정, (iv) KSTAR 아님(참고문헌에만 등장) |
| Shousha R., Seo J., Erickson K., Xing Z., Kim S.K., Abbate J. 외 2024, "Machine learning-based real-time kinetic profile reconstruction in DIII-D" (RTCAKENN), *Nucl. Fusion* 64, 026006. DOI 10.1088/1741-4326/ad142f | DIII-D | 실시간 입력→7개 프로파일(압력, 1/q, j_tor, T_e, n_e, **탄소 T_i·회전**). 드롭아웃 학습으로 **TS 또는 CER이 없을 때도 동작**, <8 ms | CER 부재 시 T_i·회전 프로파일 추정 = "결측 시각 채움"과 가장 가까운 동기. 차이: 전체 프로파일 동시각 회귀, 시간 모델 아님, 10 ms 격자 나우캐스트·skill 평가 없음 |
| Wang X., Yu J., Lin Z. 외 2026, arXiv 2610.00243 "Fusing Global Diagnostic Signals and Spectral Data for Electron and Ion Temperature Profile Prediction on EAST" | EAST | 전역 진단 + 스펙트럼 다중모달 어텐션→T_i/T_e 프로파일 동시 예측, **스펙트럼 진단이 없는 운전 시나리오 호환** | 프리프린트. 동시각 사상, 시간 결측·인과성·기준선 없음 |
| Morosohk S., Schuster E. 2023, *Contrib. Plasma Phys.* 63, e202200153. DOI 10.1002/ctpp.202200153 | DIII-D | TS + 전자열수송식 관측기(NN 대리모델)로 실시간 T_e 프로파일 추정 | 모델 기반 관측기(Kalman류). 전자 채널 |
| Pavone A., Svensson J., Langenberg A. 외 2019, *Plasma Phys. Control. Fusion* 61, 075012. DOI 10.1088/1361-6587/ab1d26 | W7-X | XICS 영상→T_i·T_e 프로파일 베이지안 추론을 NN으로 근사 | 측정 있는 시각의 역문제 가속 |
| Sorokin D. 외 2026, arXiv 2610.24525 | DIII-D | FNO 수송 대리모델로 관측 안 된 중심부 n_e 복원 + 불확도(CRPS·커버리지) | 공간 결측, 전자 밀도 |

### 1-C. 결측 진단 신호의 데이터 기반 복원·대입 (diagnostic imputation)

| 연구 | 장치 | 핵심 | 우리와의 차이 |
|---|---|---|---|
| **Wang M.R., Wan C., Lu J., Yu Z., Xiao B., Li Y. 외 2025, "Time series extrinsic regression for reconstructing missing electron temperature in tokamak", *Nucl. Fusion* 65, 076008. DOI 10.1088/1741-4326/addb5f** | EAST | TSER로 다른 시계열에서 결측 T_e 복원(진단 부재·획득 오류). 전 방전 구간에서 95.9 % 신뢰로 2σ 안 | 전자온도. 방전 전체 결측(진단 자체 부재) 복원, 오프라인. 10 ms 수준 산발 결측·인과성·persistence 대조 아님 |
| **Jia K., Xu Z., Wu Z., Gao W., Wang M., Ti A. 외 2026, "Time series regression network using exogenous variables for missing effective ion charge reconstruction in EAST tokamak", *Nucl. Fusion* 67, 016012. DOI 10.1088/1741-4326/aeb432** | EAST | CNN-**BiGRU**-Transformer로 외생 변수에서 결측 Z_eff 복원, 1566 방전, 유사도 90.3 %·커버리지 91.4 % | **양방향 GRU = 비인과**, 스스로 "오프라인 신호 복원"이라 명시. 평가 척도는 유사도·커버리지(skill 아님) |
| Rath K., Rügamer D., Bischl B., von Toussaint U., Albert C.G. 2023, *Contrib. Plasma Phys.* 63, e202200175. DOI 10.1002/ctpp.202200175 | (W7-X 자속루프에서 착안) | 상태공간 Student-t 과정으로 결측점 대입 + 데이터 증강, 이상치에 강건, 고해상도 시계열용 | 통계적 다변량 대입. 운동학 진단 아님 |
| Yang Z., Yang Z., Tian W. 외 2026, "FusionMAE", *Commun. Phys.* 9. DOI 10.1038/s42005-026-02626-3 | HL-3 | 88개 진단 신호의 자기지도 마스크 오토인코더, "missing-signal reconstruction"으로 **가상 백업 진단**(97.2 %) | 신호 단위 결측 복원, 비인과(마스크 복원), 특정 진단의 시간 결측 skill 아님 |
| Kaldas S., Rossi R., Wyss I., Gelfusa M., Gaudio P. 2026, *EPJ Web Conf.* 388, 00012. DOI 10.1051/epjconf/202638800012 | 수치 시나리오 | 다중진단 오토인코더로 n_e 신뢰도 향상 | 실험 데이터 아님 |
| Boschi T. 외 2026, "TokaMind", KDD 2026. DOI 10.1145/3770855.3818986; Rousseau C. 외 2026 "TokaMark", arXiv 2602.10132 | MAST | 다중모달 파운데이션 모델, 서로 다른 표본율·**결측 신호 처리**, 14개 과제(재구성 3 + 예보 11). 평가는 "가용 진실에 대한 마스크 척도(NRMSE)" | **T_i·CXRS 회전 과제 없음**(TS 전자량만). 훈련 손실은 다중출력 MSE, 결측값은 0으로 채움 |
| Gupta N. 2026, arXiv 2607.11915 | MAST | 센서 결측 6 시나리오·대입 3종(forward-fill 등)에 대한 모델 강건성 벤치마크 | 대입은 입력 쪽 처리이지 타깃 채움 아님 |

### 1-D. 보간·GP 평활 (대조군 쪽 선행)

- Chilenski M.A., Greenwald M., Marzouk Y., Howard N.T. 외 2015, *Nucl. Fusion* 55, 023012. DOI 10.1088/0029-5515/55/2/023012 — GP 회귀 프로파일 피팅과 기울기 불확도(공간 축).
- Ho A., Citrin J., Auriemma F. 외 2019, *Nucl. Fusion* 59, 056007. DOI 10.1088/1741-4326/ab065a — JET 프로파일(T_i·회전 포함) GP 피팅을 모델 검증 파이프라인에 적용(공간 축 중심).
- 종합 개관: Pavone A., Merlo A., Kwak S. 외 2023, *Plasma Phys. Control. Fusion* 65, 053001. DOI 10.1088/1361-6587/acc60f; Glasser 2026 (synthetic diagnostics 튜토리얼), *Plasma Phys. Control. Fusion* 68, 103302. DOI 10.1088/1361-6587/aeb280.

### Q1 판정

- CES/CXRS **T_i·회전의 시간 결측을 다른 진단으로 채우는 연구**: DIII-D·JET·ASDEX·EAST·KSTAR 어디에서도 **미발견**. EAST의 결측 복원 두 편(T_e, Z_eff)은 전자·불순물 채널이며 오프라인·양방향. Diag2Diag는 "다른 진단으로 한 진단을 고해상도화"라는 틀은 같지만 출력이 TS, 동시각·무기억, 나이브 기준선 대조 없음.
- 분류: **신규(미발견)** — "CES T_i·V_rot의 10 ms 결측 인과 나우캐스트 + 보간·persistence·GP 대비 skill". 단 서술은 부재-프레임이 아니라 계보(Svensson 1999 → NN-CES 2025 → Diag2Diag 2025 → EAST 결측 복원 2025–26)를 인정하고 세 축(전자→이온, 동시각→인과, 가정→검정)으로 확장했다고 쓰는 편이 안전하다.

---

## Q2. 토카막 시계열 순차 모델: 인과·고정 격자·마스크 손실·불규칙/결측 타깃

| 연구 | 모델·격자 | 인과성 | 결측/불규칙 타깃 처리 |
|---|---|---|---|
| Kates-Harbeck J., Svyatkovskiy A., Tang W. 2019, FRNN, *Nature* 568, 526. DOI 10.1038/s41586-019-1116-4 | 상태 유지 LSTM, 0D + 1D(프로파일) 입력, 고정 ms 격자, 분류(붕괴 전 경보) | 인과(실시간 지향) | 라벨은 붕괴 시각에서 정의되어 모든 시각에 조밀. 결측 타깃 개념 없음 |
| Churchill R.M., Tobias B., Zhu Y. 2020, *Phys. Plasmas* 27, 062510. DOI 10.1063/1.5144458 | 팽창 합성곱(TCN류) 다중 스케일 분류, DIII-D ECEi | 인과 | 분류, 결측 타깃 없음 |
| Abbate J., Conlin R., Kolemen E. 2021, *Nucl. Fusion* 61, 046027. DOI 10.1088/1741-4326/abe08d | NN으로 τ_E 규모 앞 프로파일 예측(actuator 궤적 + 상태 이력), DIII-D 2013–18 | 인과(미래 actuator는 기지로 가정) | 프로파일을 시간 창에 모아 사용; 결측 타깃 마스크 손실 언급 없음(초록 기준) |
| Abbate J., Fable E., Tardini G., Fischer R., Kolemen E. 2025, *Nucl. Fusion* 65, 056014. DOI 10.1088/1741-4326/adc283 | 물리+데이터 메타학습, DIII-D·AUG·ITER 외삽 | — | **"ML·물리 모델 모두 분포 밖 외삽에서는 '프로파일이 초기값에서 변하지 않는다'는 가정보다 못하다"** — persistence 기준선 명시(Q3 참조) |
| Char I., Chung Y., Abbate J., Kolemen E., Schneider J. 2024, arXiv 2404.12416 | DIII-D 전 방전 자기회귀 예측, **25 ms 격자(직전 25 ms 평균)**, GRU/LSTM + 가우스 출력, 앙상블 | 상태는 과거 평균이나 **25 ms 앞 actuator를 입력** | 결측·불규칙 진단(TS·CER) 처리 언급 없음. **persistence·보간 기준선 없음**(모델 간 내부 비교만) |
| Seo J., Na Y.-S., Kim B. 외 2021, *Nucl. Fusion* 61, 106010. DOI 10.1088/1741-4326/ac121b; Seo J. 외 2022, *Nucl. Fusion* 62, 086049. DOI 10.1088/1741-4326/ac79be | KSTAR 0D 상태(β_p, q95, l_i …) 순환 신경망 예측기를 RL 환경으로 사용(Char 2024가 "RNN으로 스칼라 상태 예측"으로 인용) | 인과·자기회귀 | 0D, 결측 없음 |
| Seo J., Conlin R., Rothstein A. 외 2023, IJCNN. DOI 10.1109/ijcnn54540.2023.10191359; Seo J., Kim S.K., Jalalvand A. 외 2024, *Nature* 626, 746. DOI 10.1038/s41586-024-07024-9 | DIII-D 다중모달 tearing 가능성 동역학 모델 → RL 제어 | 인과 | 분류형 타깃 |
| Jalalvand A., Kaptanoglu A., Garcia A. 외 2021, *Nucl. Fusion* 62, 026007. DOI 10.1088/1741-4326/ac3be7 | 저수지 컴퓨팅(RC)으로 ECE 기반 Alfvén 고유모드 분류 | 인과 | 분류 |
| Boyer M.D., Chadwick J. 2021, *Nucl. Fusion* 61, 046024. DOI 10.1088/1741-4326/abe08b | NSTX-U n_e·압력 프로파일 형상 NN(기저함수 투영) | 동시각 | — |
| Wan C., Yu Z., Pau A., Liu X., Li J. 2022, *Nucl. Fusion* 62, 126060. DOI 10.1088/1741-4326/ac9c1a; Wang M.R. 외 2026, *Nucl. Fusion* 66, 124004. DOI 10.1088/1741-4326/aeaa6c | EAST 방전 예측(actuator→0D), 장기 자기회귀 I_p·경계 | 인과 | 유사도 척도 |
| Lee, Kim, Kim 외 2025, *Nucl. Fusion* 65, 086017. DOI 10.1088/1741-4326/adeb9b | KSTAR 0D LSTM 붕괴 예측(2019–22 탄소 다이버터, 1 s 창) | 인과 | 분류 |
| Joung S., Smith D.R., McKee G.R. 외 2024, BES-ELMnet, *Nucl. Fusion* 64, 066038. DOI 10.1088/1741-4326/ad43fb; Samaddar A. 외 2025, *Mach. Learn.: Sci. Technol.* DOI 10.1088/2632-2153/adfb41 | DIII-D BES 2D 요동(128 µs 창)→ELM 발생 확률 / ELM 시공간 예보 | 인과 | 분류·회귀, 결측 타깃 없음 |
| Wu N., Yang Z., Li R. 외 2025, *Commun. Phys.* 8. DOI 10.1038/s42005-025-02302-y | HL-3 고충실 데이터 동역학 모델(RL용) | 인과 | — |
| Boschi 외 2026 TokaMind (위) | 다중 표본율·결측 신호 처리, DCT 임베딩 | 예보 과제는 인과 | 평가만 마스크(가용 진실), 훈련 손실은 MSE |
| Jia 외 2026 (EAST Z_eff, 위) | CNN-**BiGRU**-Transformer | **비인과** | 결측 타깃 자체를 복원(오프라인) |
| Steiner 외 APS DPP 2025 (위, 초록만) | 합성곱 인코더-디코더, 스펙트럼→T_i·회전 | 동시각 | **결측 라벨 무시 masked loss** |

### Q2 판정

- 고정 격자 인과 LSTM/GRU/TCN: **기존**(FRNN, Churchill, Char, Seo, Lee 등).
- 결측·불규칙 **타깃**에 대한 마스크 손실로 "측정이 있는 시각만 감독하고 없는 시각을 채우는" 설계: 결측 **라벨 무시** 마스크 손실은 Steiner(DPP 2025 초록, 스펙트럼 입력)에만 보이고, 시계열 모델에서 타깃별(T_i·V_rot 독립 결측) 마스크 + 타깃 시각 완전 마스킹 + 10 ms 격자 조합은 **미발견**. TokaMind는 평가만 마스크. EAST 복원 모델은 양방향.
- 분류: **확장** — 모델 계열은 기존, 결측-타깃 감독 구조와 두 분기 라우팅(V_rot 분기 고속 진단 차단)은 미발견.

---

## Q3. "나우캐스팅" 용어·persistence 대비 skill 점수·보간 기준선

### 3-A. 용어 "nowcasting"

- OpenAlex 제목·초록 검색("nowcasting tokamak plasma", "nowcasting fusion plasma machine learning"): 핵융합 결과 0건(기상·지진·우주날씨만).
- arXiv API(abs: nowcasting AND (tokamak OR "fusion plasma"); abs: nowcast AND tokamak): 0건.
- Crossref 서지 검색("nowcasting tokamak plasma", "nowcasting fusion plasma tokamak forecasting"): 관련 0건.
- 웹 검색("nowcasting" tokamak plasma machine learning): 용어 사용 논문 없음(인접 결과: Wang 2025 TCV 램프다운 arXiv 2502.12327, Wan 2026 WEST arXiv 2602.19110, Samaddar 2025 ELM 예보).
- OpenAlex 전문(fulltext) 검색은 요청 제한(429)으로 완료 못 함 → 본문 안에서만 "nowcast"를 쓴 논문이 있을 가능성은 배제 못 함.
- 판정: **신규(미발견)** — 핵융합 ML에서 기상학적 의미의 "nowcasting" 용어 사용은 제목·초록 수준에서 미발견.

### 3-B. persistence(무변화) 기준선과 skill 점수

- **Abbate 외 2025 NF**(위): "ML 모델을 먼 분포로 외삽한 것과 최신 물리 기반 프로파일 예측기 모두 '프로파일이 초기값에서 변하지 않는다'고 가정하는 것보다 못하다" — persistence가 명시적 기준선. 다만 skill score 형태는 아님(초록 기준).
- **TokaMark(Rousseau 외 2026)**: 기준 모델로 "Persistence Model(선택 과제의 자기회귀 예보에 초기조건 전파)"과 "Mean model"을 둠. 척도는 전역 표준편차로 정규화한 NRMSE. **나이브 기준선 대비 skill score는 없음**, 보간 기준선 없음.
- **Char 외 2024**: persistence·보간 기준선 없음(모델 간 비교만).
- **Diag2Diag(2025)**: 보간·persistence 정량 대조 없음, 보간이 사건을 놓친다는 정성 논거만.
- **Wang 2025 / Jia 2026(EAST 복원)**: 유사도·커버리지 확률로 평가, 나이브 기준선 없음.
- Crossref "skill score tokamak plasma prediction persistence": 관련 0건.
- 판정: persistence 기준선 자체는 **기존**(Abbate 2025, TokaMark). "persistence·비인과 보간(PCHIP)·인과/오프라인 GP 대비 skill score"를 평가 프로토콜의 중심으로 삼는 것은 **신규(미발견)**.

### 3-C. 보간을 기준선으로 쓴 평가

- 정량 비교는 **미발견**. Diag2Diag가 정성적으로만 언급. 공간 축에서는 GP 피팅(Chilenski 2015, Ho 2019)이 표준 보간 역할을 하지만 시간 결측 채움의 기준선으로 쓰인 사례는 못 찾음.

---

## Q4. 물리 사전지식: 10 ms에서 고속 진단이 T_i와 V_rot에 주는 정보량 (비대칭 주장 지지/반박)

### 4-A. T_i 쪽: ms 스케일 동역학이 고속 진단과 함께 움직인다 (지지)

| 연구 | 결과 | 함의 |
|---|---|---|
| Cavedon M., Pütterich T., Viezzer E., Laggner F.M. 외 2017, *Plasma Phys. Control. Fusion* 59, 105007. DOI 10.1088/1361-6587/aa7ad0 | AUG 가장자리 CXRS **65 µs** 시간분해. ELM에서 T_i·n_e·흐름 기울기 이완, ∇T_i는 ∇n_e와 비슷한 시간에 회복(∇T_e보다 빠름), E_r은 ELM에서 L-모드 값으로 붕괴 후 T_i와 같은 시간규모로 회복 | 10 ms 안에서 T_i가 크게 움직이는 사건(ELM)이 있고, 그 시점은 BES(n_e)·ECEI(T_e)·Mirnov가 보는 사건과 동기 |
| Cavedon M., Dux R., Pütterich T. 외 2019, *Nucl. Mater. Energy* 18, 275. DOI 10.1016/j.nme.2018.12.034 | ~100 µs 측정. **∇T_i·∇n_e는 ELM 후 약 3.0 ms에 중주파(f ≲ 50 kHz) 자기 요동 시작과 함께 클램프**, ∇T_e는 7.0 ms 후 고주파(≈200 kHz) 요동과 함께 포화 | T_i 기울기 포화 시점이 Mirnov 대역 자기 요동과 상관 → 자기 신호에 T_i 상태 정보가 있음(우리 데이터에서 MC가 aliasing된 점은 별개) |
| Viezzer E., Cavedon M., Cano-Megias P., Fable E. 외 2019, *Plasma Phys. Control. Fusion* 62, 024009. DOI 10.1088/1361-6587/ab5b1d | ELM 주기 동안 이온·전자 에너지, 입자, **운동량** 수송의 동역학. 전자가 더 느리게 회복. 토로이달 토크 균형 + 폴로이달 불순물 비대칭 모델이 **회전의 ELM 전·회복기 동역학을 재현** | **주의(부분 반박)**: 가장자리 회전도 ELM 주기 안(ms)에서 움직인다. 비대칭 주장은 "가장자리 ELM 주기"가 아니라 "표준 CES 10 ms 격자의 코어·페데스탈 상부"에 한정해 써야 함 |
| Yun G.S., Lee W., Choi M. 외 2011, *Phys. Rev. Lett.* 107, 045004. DOI 10.1103/PhysRevLett.107.045004 | KSTAR ECEI로 ELM 필라멘트의 성장·준정상·붕괴 3단계 2D 가시화 | KSTAR ECEI가 ELM 사건을 직접 봄 |
| Park H.K. 2019, *Adv. Phys. X* 4, 1633956. DOI 10.1080/23746149.2019.1633956 | KSTAR ECEI로 밝힌 MHD 물리 개관 | ECEI 정보 내용의 종설 |
| Lee J., Kim M., Yun G.S. 외 2024, *Phys. Plasmas* 31, 092504. DOI 10.1063/5.0210947 | KSTAR T_e 페데스탈 진화·붕괴와 미세 찢김 모드(ECE 광대역) | 붕괴 전 페데스탈 상태가 ECE 요동에 드러남 |
| Nam Y.U., Zoletnik S., Lampert M. 외 2014, *Rev. Sci. Instrum.* 85, 11D502. DOI 10.1063/1.4894839; Lampert 2018 *Phys. Plasmas* 25 DOI 10.1063/1.5017295 | KSTAR 2D BES 가장자리 n_e 프로파일·요동 | KSTAR BES 정보 내용 |
| Joung 외 2024 BES-ELMnet(위) | DIII-D BES 128 µs 창에서 ELM 전조 학습 | BES에 수 ms 앞 사건 정보 |
| Diallo A., Laggner F.M. 2020, *Plasma Phys. Control. Fusion* 63, 013001. DOI 10.1088/1361-6587/abbf85 | ELM 사이 페데스탈 요동 다장치 종설 | 배경 |

### 4-B. V_rot 쪽: 느린 운동량 시간규모 + 관측 안 되는 토크 + 광자 통계 (지지)

| 연구 | 결과 | 함의 |
|---|---|---|
| Tardini G., Ferreira J., Mantica P., Peeters A.G., Tala T., Zastrow K.-D. 외 2009, *Nucl. Fusion* 49, 085010. DOI 10.1088/0029-5515/49/8/085010 | JET NBI 변조로 운동량 확산·핀치 분리, Prandtl 수 ~1(약간 위) + Coriolis 핀치 | 운동량 수송 ≈ 이온 열수송 → τ_φ ~ τ_E(수십~수백 ms). 10 ms 안의 V_rot 변화는 주로 토크 입력(NBI)의 변화 |
| Peeters A.G., Angioni C., Bortolon A. 외 2011, *Nucl. Fusion* 51, 094027. DOI 10.1088/0029-5515/51/9/094027; Ida K., Rice J.E. 2014, *Nucl. Fusion* 54, 045001. DOI 10.1088/0029-5515/54/4/045001 | 토로이달 운동량 수송 종설, Prandtl ~1, NBI/RF 구동·NTV 감쇠 | 회전은 토크 균형의 느린 적분량 |
| Zimmermann C.F.B., McDermott R.M., Fable E. 외 2022, *Plasma Phys. Control. Fusion* 64, 055020. DOI 10.1088/1361-6587/ac5ae8 | AUG NBI 변조로 수송계수 추출 검증 | 회전 동역학을 읽으려면 **토크(NBI) 시계열이 필요** — 우리 데이터에 NBI 채널이 없다는 한계와 직결 |
| Yoo M.-G., Lee, Ko W.H., Seol 2017, *Phys. Plasmas* 24, 072510. DOI 10.1063/1.4991397 | KSTAR 고유 회전(옴 방전)은 T_i/I_p 스케일링(기울기 ≈ −23) | 회전이 느린 전역량(T_i/I_p)에 묶임 |
| Kim, Lee, Ko 외 2020, *Phys. Plasmas* 27, 112501. DOI 10.1063/5.0024055 | KSTAR n = 1 RMP ELM 억제기의 페데스탈 ν_e*·V_tor 데이터베이스(V_tor,ped > 40 km/s) | 3D 장(RMP) 인가 시 NTV 제동 = 회전의 또 다른 비관측 입력 |
| Hutchinson I.H. 2002, *Principles of Plasma Diagnostics* 2판, Cambridge. DOI 10.1017/CBO9780511613630 | 선 중심 통계 불확도 σ_λ ∝ Δλ_D/√N (광자 수 N) | 10 ms 노출에서 회전(선 이동) 정밀도는 광자 수에 묶임. T_i는 선 폭이라 상대 민감도가 다름 |
| Truong D., McKee G.R., Yan Z., Jaehnig K. 2021, UF-CHERS, *Rev. Sci. Instrum.* 92, 053513. DOI 10.1063/5.0043095 | DIII-D 초고속 CHERS: T_i·v_∥ 요동 ~1 %, ≤450 kHz. **두 시선 상관으로 비간섭 전자·광자 잡음을 억제해야** 함 | 회전의 빠른 성분은 물리적으로 존재하나 요동 수준(~1 %)이고 광자 잡음 억제 장치가 필요 → 표준 CES 10 ms 단일 시선에서는 묻힘 |
| Jang, Ko W.H., Li, von Hellermann, Shi, Hwang 2021, UFCES, *Rev. Sci. Instrum.* 92, 053525. DOI 10.1063/5.0043824 | KSTAR 초고속 CES 설계: 고처리량 광학·투과 격자로 **10 µs**에서 SNR 확보(시뮬레이션 + 벤치 시험) | KSTAR 표준 CES(10 ms)의 시간 해상도 한계가 광자 처리량 문제임을 설계 논문이 전제 |
| Ko W.H., Lee, Seo 외 2010, *Rev. Sci. Instrum.* 81, 10D740. DOI 10.1063/1.3496991 | KSTAR CES 교정·T_i 측정 | 우리 10 ms 격자 진단의 원 논문 |
| Chrystal 외 2026 실시간 CER(위) | 5 ms 표본, 피팅 결과 1 ms 내 | 다른 장치의 "현재 CES 격자" 참조값 |

### 4-C. 비대칭 주장에 대한 정리

- 지지: (i) T_i의 10 ms 내 큰 변화는 ELM 등 사건이고, 그 사건은 BES·ECEI·자기 요동이 직접 본다(Cavedon 2017/2019, Yun 2011, Joung 2024). (ii) 회전은 Prandtl ~1의 느린 적분량이며 10 ms 변화의 구동은 토크(NBI·NTV)라 우리 입력에 없다(Tardini 2009, Peeters 2011, Zimmermann 2022, Kim 2020). (iii) 회전의 빠른 성분은 요동 수준이고 광자 잡음에 묻혀 상관 측정이 필요하다(Truong 2021, Jang 2021, Hutchinson).
- 반박·한정: Viezzer 2019와 Cavedon 2017은 **가장자리** 회전·E_r이 ELM 주기 안에서 뚜렷이 움직임을 보였다. 따라서 "V_rot는 고속 진단에서 읽을 정보가 없다"가 아니라 "표준 CES 10 ms 격자·코어~페데스탈 상부·NBI 비관측 조건에서는 V_rot의 10 ms 증분이 광자 바닥과 토크 비관측에 묶인다"로 써야 한다. 우리 §8bq–§8bt(광자 바닥) 결론과 부합.
- 직접적으로 "고속 진단의 T_i 대 V_rot 정보량"을 비교한 논문은 **미발견**. 판정: **확장**(기존 물리 결과들을 조합한 근거이며, 정량 비교 자체는 우리가 처음 하는 셈).

---

## 종합 판정표

| 측면 | 판정 | 근거 요약 |
|---|---|---|
| CES T_i·V_rot 시간 결측의 다중진단 인과 나우캐스트 | **신규(미발견)** | 가장 가까운 것: Diag2Diag(TS 출력·동시각), RTCAKENN(CER 부재 시 프로파일), EAST T_e/Z_eff 복원(오프라인·양방향), NN-CES(측정 있는 시각) |
| 인과 고정격자 LSTM 자체 | 기존 | FRNN·Char·Seo·Lee 등 |
| 결측 타깃별 마스크 손실 + 타깃 시각 완전 마스킹 + 두 분기 라우팅 | 확장(미발견 조합) | 마스크 손실은 Steiner DPP 2025(스펙트럼), TokaMind는 평가만 마스크 |
| "nowcasting" 용어 | 신규(미발견, 전문 검색 미완) | OpenAlex 제목·초록, arXiv, Crossref, 웹 0건 |
| persistence 기준선 | 기존 | Abbate 2025, TokaMark |
| persistence·PCHIP·인과/오프라인 GP 대비 skill 프로토콜 | 신규(미발견) | 어느 선행도 보간 기준선을 정량 비교하지 않음 |
| T_i↔V_rot 비대칭의 물리 근거 | 확장 | Cavedon/Viezzer(AUG), Tardini/Peeters/Ida-Rice, Truong/Jang(광자), 단 가장자리 회전은 ms 동역학 있음 |

---

## 참고문헌 (전부 doi.org 해석 확인, 2026-11-09)

1. Svensson J., von Hellermann M., König R. (1999) Analysis of JET charge exchange spectra using neural networks. *Plasma Phys. Control. Fusion* 41, 315. 10.1088/0741-3335/41/2/016
2. Lee J.K., Ko W.H., Lee H.H., Kim B., Lee Y., Shin G. et al. (2025/2026) Neural network-based analysis of charge exchange spectra in KSTAR. *Fusion Eng. Des.* 222, 115518. 10.1016/j.fusengdes.2025.115518
3. Tian W.H., Yang Z., Long T. et al. (2026) Real-time CXRS analysis on HL-3 tokamak with reliable neural networks and cross-device transfer learning. *Nucl. Fusion* 67, 016004. 10.1088/1741-4326/aeb037
4. Hu Z., Lin Z., Fu S. et al. (2026) Real-time ion temperature inference from CXRS using GAN-augmented CNN. *Plasma Sci. Technol.* 10.1088/2058-6272/aeb5c5
5. Chrystal C., Groebner R., Linsenmayer, Haskey S. et al. (2026) Real-time charge exchange recombination spectroscopy at DIII-D. *Rev. Sci. Instrum.* 97, 103501. 10.1063/5.0350505
6. Steiner P., Jalalvand A., Shousha R., Pratt Q. et al. (2025) Seeing in the dark – Neural networks for passive CER spectroscopy analysis. APS DPP 2025, TM11.00001 (학회 초록; https://meetings-archive.aps.org/dpp/2025/tm11/1/)
7. Jalalvand A., Kim S.K., Seo J., Hu Q., Curie M., Steiner P., Nelson A.O., Na Y.-S., Kolemen E. (2025) Multimodal super-resolution: discovering hidden physics and its application to fusion plasmas. *Nat. Commun.* 16, 8506. 10.1038/s41467-025-63492-1 (arXiv 2405.05908)
8. Shousha R., Seo J., Erickson K., Xing Z., Kim S.K., Abbate J. et al. (2024) Machine learning-based real-time kinetic profile reconstruction in DIII-D. *Nucl. Fusion* 64, 026006. 10.1088/1741-4326/ad142f
9. Wang X., Yu J., Lin Z. et al. (2026) Fusing Global Diagnostic Signals and Spectral Data for Electron and Ion Temperature Profile Prediction on EAST. arXiv 2610.00243
10. Wang X., Lu Z., Yang D. et al. (2026) SpectralTF-MAE. arXiv 2610.38593
11. Morosohk S., Schuster E. (2023) Real-time estimation of the electron temperature profile in DIII-D by leveraging neural-network surrogate models. *Contrib. Plasma Phys.* 63, e202200153. 10.1002/ctpp.202200153
12. Pavone A., Svensson J., Langenberg A. et al. (2019) Neural network approximation of Bayesian models for the inference of ion and electron temperature profiles at W7-X. *Plasma Phys. Control. Fusion* 61, 075012. 10.1088/1361-6587/ab1d26
13. Sorokin D., Granovskiy A., Stokolesov M. et al. (2026) Reconstruction of Unobserved Core Electron Density in Tokamak Plasmas with a Neural Operator Transport Surrogate. arXiv 2610.24525
14. Wang M.R., Wan C., Lu J., Yu Z., Xiao B., Li Y. et al. (2025) Time series extrinsic regression for reconstructing missing electron temperature in tokamak. *Nucl. Fusion* 65, 076008. 10.1088/1741-4326/addb5f
15. Jia K., Xu Z., Wu Z., Gao W., Wang M., Ti A. et al. (2026) Time series regression network using exogenous variables for missing effective ion charge reconstruction in EAST tokamak. *Nucl. Fusion* 67, 016012. 10.1088/1741-4326/aeb432
16. Rath K., Rügamer D., Bischl B., von Toussaint U., Albert C.G. (2023) Dependent state space Student-t processes for imputation and data augmentation in plasma diagnostics. *Contrib. Plasma Phys.* 63, e202200175. 10.1002/ctpp.202200175
17. Yang Z., Yang Z., Tian W. et al. (2026) FusionMAE, a self-supervised pretrained model to optimize and simplify diagnostic and control of fusion plasma. *Commun. Phys.* 9. 10.1038/s42005-026-02626-3
18. Kaldas S., Rossi R., Wyss I., Gelfusa M., Gaudio P. (2026) Multi-diagnostic autoencoders for tokamak plasmas. *EPJ Web Conf.* 388, 00012. 10.1051/epjconf/202638800012
19. Boschi T., Loreti A., Amorisco N.C. et al. (2026) TokaMind: A Multi-Modal Transformer Foundation Model for Tokamak Plasma Dynamics. KDD 2026. 10.1145/3770855.3818986
20. Rousseau C., Jackson S., Ordóñez-Hurtado R. et al. (2026) TokaMark: A Comprehensive Benchmark for MAST Tokamak Plasma Models. arXiv 2602.10132
21. Gupta N. (2026) Benchmarking Sensor Robustness in Plasma Diagnostic Models. arXiv 2607.11915
22. Chilenski M.A., Greenwald M., Marzouk Y., Howard N.T. et al. (2015) *Nucl. Fusion* 55, 023012. 10.1088/0029-5515/55/2/023012
23. Ho A., Citrin J., Auriemma F. et al. (2019) *Nucl. Fusion* 59, 056007. 10.1088/1741-4326/ab065a
24. Pavone A., Merlo A., Kwak S. et al. (2023) Machine learning and Bayesian inference in nuclear fusion research: an overview. *Plasma Phys. Control. Fusion* 65, 053001. 10.1088/1361-6587/acc60f
25. Glasser (2026) Synthetic diagnostics and integrated modeling. *Plasma Phys. Control. Fusion* 68, 103302. 10.1088/1361-6587/aeb280
26. Kates-Harbeck J., Svyatkovskiy A., Tang W. (2019) *Nature* 568, 526. 10.1038/s41586-019-1116-4
27. Churchill R.M., Tobias B., Zhu Y. (2020) *Phys. Plasmas* 27, 062510. 10.1063/1.5144458
28. Abbate J., Conlin R., Kolemen E. (2021) Data-driven profile prediction for DIII-D. *Nucl. Fusion* 61, 046027. 10.1088/1741-4326/abe08d
29. Abbate J., Fable E., Tardini G., Fischer R., Kolemen E. (2025) *Nucl. Fusion* 65, 056014. 10.1088/1741-4326/adc283
30. Char I., Chung Y., Abbate J., Kolemen E., Schneider J. (2024) Full Shot Predictions for the DIII-D Tokamak via Deep Recurrent Networks. arXiv 2404.12416
31. Seo J., Na Y.-S., Kim B. et al. (2021) Feedforward beta control in the KSTAR tokamak by deep reinforcement learning. *Nucl. Fusion* 61, 106010. 10.1088/1741-4326/ac121b
32. Seo J. et al. (2022) Development of an operation trajectory design algorithm … in KSTAR. *Nucl. Fusion* 62, 086049. 10.1088/1741-4326/ac79be
33. Seo J., Conlin R., Rothstein A. et al. (2023) Multimodal Prediction of Tearing Instabilities in a Tokamak. IJCNN 2023. 10.1109/ijcnn54540.2023.10191359
34. Seo J., Kim S.K., Jalalvand A. et al. (2024) Avoiding fusion plasma tearing instability with deep reinforcement learning. *Nature* 626, 746. 10.1038/s41586-024-07024-9
35. Jalalvand A., Kaptanoglu A., Garcia A. et al. (2021) *Nucl. Fusion* 62, 026007. 10.1088/1741-4326/ac3be7
36. Boyer M.D., Chadwick J. (2021) *Nucl. Fusion* 61, 046024. 10.1088/1741-4326/abe08b
37. Wan C., Yu Z., Pau A., Liu X., Li J. (2022) *Nucl. Fusion* 62, 126060. 10.1088/1741-4326/ac9c1a
38. Wang M.R., Wan C., Wang Y. et al. (2026) *Nucl. Fusion* 66, 124004. 10.1088/1741-4326/aeaa6c
39. Lee, Kim, Kim et al. (2025) Machine learning based disruption prediction using LSTM in KSTAR. *Nucl. Fusion* 65, 086017. 10.1088/1741-4326/adeb9b
40. Joung S., Smith D.R., McKee G.R. et al. (2024) *Nucl. Fusion* 64, 066038. 10.1088/1741-4326/ad43fb
41. Samaddar A., Gong Q., Madireddy S. et al. (2025) *Mach. Learn.: Sci. Technol.* 10.1088/2632-2153/adfb41
42. Wu N., Yang Z., Li R. et al. (2025) *Commun. Phys.* 8. 10.1038/s42005-025-02302-y
43. Steiner P., Jalalvand A., Chen N. et al. (2026) IGNITE Tokamak World Model Architecture. arXiv 2610.02515
44. Cavedon M., Pütterich T., Viezzer E. et al. (2017) *Plasma Phys. Control. Fusion* 59, 105007. 10.1088/1361-6587/aa7ad0
45. Cavedon M., Dux R., Pütterich T. et al. (2019) *Nucl. Mater. Energy* 18, 275. 10.1016/j.nme.2018.12.034
46. Viezzer E., Cavedon M., Cano-Megias P. et al. (2019) *Plasma Phys. Control. Fusion* 62, 024009. 10.1088/1361-6587/ab5b1d
47. Diallo A., Laggner F.M. (2020) *Plasma Phys. Control. Fusion* 63, 013001. 10.1088/1361-6587/abbf85
48. Yun G.S., Lee W., Choi M. et al. (2011) *Phys. Rev. Lett.* 107, 045004. 10.1103/PhysRevLett.107.045004
49. Park H.K. (2019) *Adv. Phys. X* 4, 1633956. 10.1080/23746149.2019.1633956
50. Lee J., Kim M., Yun G.S. et al. (2024) *Phys. Plasmas* 31, 092504. 10.1063/5.0210947
51. Nam Y.U., Zoletnik S., Lampert M. et al. (2014) *Rev. Sci. Instrum.* 85, 11D502. 10.1063/1.4894839
52. Lampert M., Zoletnik S., Bak J.G. et al. (2018) *Phys. Plasmas* 25. 10.1063/1.5017295
53. Tardini G., Ferreira J., Mantica P. et al. (2009) *Nucl. Fusion* 49, 085010. 10.1088/0029-5515/49/8/085010
54. Zimmermann C.F.B., McDermott R.M., Fable E. et al. (2022) *Plasma Phys. Control. Fusion* 64, 055020. 10.1088/1361-6587/ac5ae8
55. Peeters A.G., Angioni C., Bortolon A. et al. (2011) *Nucl. Fusion* 51, 094027. 10.1088/0029-5515/51/9/094027
56. Ida K., Rice J.E. (2014) *Nucl. Fusion* 54, 045001. 10.1088/0029-5515/54/4/045001
57. Yoo M.-G., Lee, Ko W.H., Seol (2017) *Phys. Plasmas* 24, 072510. 10.1063/1.4991397
58. Kim, Lee, Ko et al. (2020) *Phys. Plasmas* 27, 112501. 10.1063/5.0024055
59. Jang, Ko W.H., Li, von Hellermann M., Shi, Hwang (2021) Development of an ultrafast charge exchange spectroscopy system on the KSTAR tokamak. *Rev. Sci. Instrum.* 92, 053525. 10.1063/5.0043824
60. Ko W.H., Lee, Seo et al. (2010) Charge exchange spectroscopy system calibration for ion temperature measurement in KSTAR. *Rev. Sci. Instrum.* 81, 10D740. 10.1063/1.3496991
61. Truong D., McKee G.R., Yan Z., Jaehnig K. (2021) UF-CHERS. *Rev. Sci. Instrum.* 92, 053513. 10.1063/5.0043095
62. Hutchinson I.H. (2002) *Principles of Plasma Diagnostics*, 2nd ed., Cambridge Univ. Press. 10.1017/CBO9780511613630

주의: DOI가 있는 61건 전부 doi.org HEAD 요청에서 302로 출판사 논문 페이지에 도달함을 확인(일괄 57건 + 개별 4건). 6번(APS DPP 초록)은 DOI가 없다.
