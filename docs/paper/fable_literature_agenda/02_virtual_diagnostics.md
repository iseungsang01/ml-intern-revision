> NN-CES 원문 상태 정정(2026-11-24): 전문은 기존 로컬 문헌 폴더에 이미 있었습니다. 아래는 당시 API에 제공한 자료 범위의 기록이며, 현재 원문 미확보의 근거로 쓰지 않습니다. [현재 원천 기록](../../reference/nnces_source.md)을 우선합니다. 원래 API 요청·응답은 ignored cache에 보존했습니다.

# 가상 Ti·프로파일 진단 및 초해상 관련 1차 문헌 조사 (검색 5회, 비망라)

## 0. 검색 경계와 접근 수준

- 검색은 5회로 끝냈고 망라적이지 않습니다. 아래 근거는 모두 초록 또는 검색 발췌 수준이며, 전문을 읽은 논문은 없습니다. 학습/평가 분할, 시간순·독립장치 검증, 수치 성능은 어느 행에서도 확인하지 못했습니다.
- RKN, CRU, GRU-D, KalmanNet은 이번 FOCUS 밖이라 검색하지 않았고, 따라서 미확인 상태로 남습니다.
- "Imrek"은 검색에서 나오지 않았습니다. 가장 가까운 기록은 Imrisek 외 2026 PPCF(COMPASS)이며, 지칭하신 논문과 같은지는 확인이 필요합니다.
- 발췌에서 음성 결과(초해상이 보간보다 못한 경우 등)를 명시한 문헌은 찾지 못했습니다. 이는 없다는 증거가 아니라 접근 한계입니다.

## 1. 근거 매트릭스 (8건)

| # | 서지 / 접근 수준 | 관측된 입력 → 목표 | 과거/미래·교사 정보 | 설계·배치 근거 | 본 과제와의 중첩 / 차이 (차이는 필자 추론) |
|---|---|---|---|---|---|
| 1 | Lee, Ko, Lee, Kim, Lee, Shin, Kim, Lee, "Neural network-based analysis of charge exchange spectra in KSTAR", Fusion Eng. Des. 222:115518 (2026). 제3자 참고문헌 목록으로 서지만 확인. DOI 문자열은 직접 보지 못했고 논문번호만 일치. 경로: https://arxiv.org/html/2610.38593 | 제목상 CES 스펙트럼 → 분석량. 입력·출력·속도 미확인 | 미확인 | 미확인 | 같은 장치·같은 진단. 다만 본 과제에는 원시 스펙트럼이 없으므로 "현재 스펙트럼을 빨리 피팅"과 "과거 보고값+고속 스냅샷으로 현재 보고값 추정"은 다른 문제 |
| 2 | Jang, Ko, Li, von Hellermann, Shi, Hwang, "Development of an ultrafast charge exchange spectroscopy system on the KSTAR tokamak", RSI 92, 053525 (2021), doi:10.1063/5.0043824. 초록 수준. https://pubmed.ncbi.nlm.nih.gov/34243333/ | Ti와 토로이달 회전 요동 측정이 설계 목적이며, 관측선은 C VI 529.05 nm | 해당 없음(하드웨어) | 10 µs 분해능과 충분한 SNR은 코드 시뮬레이션에서 나온 수치. 첫 실측은 2021 캠페인으로 예고된 상태 | 고속 Ti의 하드웨어 경로. 실측 쌍 자료가 아니므로 고속 Ti 진리값 근거로 쓸 수 없음 |
| 3 | Shousha, Seo, Erickson, Xing, Kim, Abbate, Kolemen, "Machine learning-based real-time kinetic profile reconstruction in DIII-D", Nucl. Fusion 64, 026006, doi:10.1088/1741-4326/ad142f. 발췌 수준. https://iopscience.iop.org/article/10.1088/1741-4326/ad142f | 입력에 RT-EFIT 압력·전류(q) 프로파일과 TS·CER 기반 ne, Te, Ti, 회전 프로파일이 포함. 출력은 운동학적 평형을 만족하는 7개 프로파일(Ti 포함) | 현재 시점 Ti·회전은 실시간 CER 알고리즘 CERREAL에서 들어옴. 목표가 오프라인 CAKE이므로 학습 전용의 풍부한 교사에 해당(필자 해석) | DIII-D PCS 탑재와 CAKE 수준 정확도, 기존 실시간 대안 대비 우위가 초록의 주장. 특정 방전에서 10개 망 평균까지 포함한 실행시간은 약 13 ms. CER 부재 조건도 논문에서 다뤄지는 것으로 보임. 분할 방식 미확인 | 실시간 NN Ti 프로파일, 진단 결손 강건성, 약 10 ms 예산이 이미 시연됨. 차이: 현재 CER 사용, 프로파일·전역 입력, 목표가 CES 보고값이 아닌 CAKE |
| 4 | Shousha, Steiner, Seo, Conlin, Abbate, Xing, Kim, Erickson, Farre-Kaga, Rothstein, Kolemen, Majeski, "Bridging the gap: Deploying AI-based Models in Real-Time Fusion Plasma Control Systems", IFAC-PapersOnLine 59-11 (2025) 198–203. 발췌 수준. https://www.osti.gov/pages/servlets/purl/2999845 | #3과 동일 계열 | 보관된 실시간 자료가 학습에 포함 | 주기당 계산시간 약 8 ms로 기술. KSTAR로의 확장이 Figure 5에 제시. KSTAR PCS에는 배치 전 타이밍 점검용 하드웨어 테스트 모드가 존재 | KSTAR 배치 선례가 있음. KSTAR에서의 Ti 정확도와 입력 구성은 미확인 |
| 5 | Jalalvand, Kim, Seo, Hu, Curie, Steiner, Nelson, Na, Kolemen, "Multimodal super-resolution: discovering hidden physics and its application to fusion plasmas", Nat. Commun. 2025 (DOI 미확인). arXiv:2405.05908(v2 제목 "Diag2Diag…"). 발췌 수준. https://arxiv.org/pdf/2405.05908 | 입력은 간섭계, ECE, 자기탐침, CER(200 Hz), MSE → TS(ne, Te) | TS 자체는 입력에서 제외. TS 표본 간격이 불균일해 무기억 망을 택하고, 시간 정보는 ECE·간섭계의 1·2차 미분으로 보충. 미분 창이 미래를 쓰는지는 미확인 | 시간분해능 0.2 kHz → 1 MHz 향상이 저자 주장. 코드: https://github.com/PlasmaControl/diag2diag. 실시간 배치 근거는 보지 못함 | "고속 진단 → 저속 진단 재구성"이 Nature 계열에 이미 있음. 차이: 목표가 전자 채널, 보정된 다채널 입력, 목표의 과거 앵커 미사용, 사후 물리 분석 목적 |
| 6 | Imrisek, Seidl, Odložilík, Šos, Bílková, Böhm 외, "Temporal super-resolution of kinetic plasma profiles at COMPASS using neural networks", PPCF 68(6), 2026, doi:10.1088/1361-6587/ae72c6. 서론 발췌만. https://iopscience.iop.org/article/10.1088/1361-6587/ae72c6/meta | 미확인. 제목상 운동학적 프로파일의 시간 초해상 | 미확인 | 게재지는 PPCF의 ECPD 2025 특집호. 장치는 COMPASS | 목표 학술지에 같은 표제어가 이미 실림. Ti 포함 여부와 인과성(미래 창) 여부는 반드시 전문 확인 필요 |
| 7 | Morosohk, Schuster, "Real-time estimation of the electron temperature profile in DIII-D by leveraging neural-network surrogate models", Contrib. Plasma Phys. 2023, doi:10.1002/ctpp.202200153. 초록 수준. https://www.osti.gov/pages/biblio/1970193 | TS 자료와 전자 열수송 방정식 모델을 결합한 관측기로 Te 프로파일 추정 | 관측기 구조상 과거 중심으로 보이나 미확인 | 검증은 오프라인 시험 단계 | 희소 진단+모델 관측기의 오래된 계보. 본 과제의 grey-box·observer 후보와 개념이 겹침. 차이: 수송모델·소스항이 필요하고 전자 채널 |
| 8 | "Initial measurement of ultrafast charge exchange recombination spectroscopy on the EAST tokamak", Plasma Sci. Technol. 2024. 인용 발췌만(저자·DOI 미확인). https://www.researchgate.net/publication/377164565 | 고속 CXRS 실측(예비) | 해당 없음 | 선행 시연 장치로 TFTR, DIII-D, ASDEX와 함께 KSTAR가 열거됨 | 아래 §3의 충돌 참조 |

경로로만 확인한 인접 문헌도 있습니다. J-TEXT에서는 완전연결망이 플라즈마 변수와 SXR 세기로 Te를 예측했다는 서술이 #6 서론에 있고, 원 논문은 찾지 않았습니다.

## 2. 넓은 신규성 주장을 막는 기존 시연 (1차 문헌 주장)

- **실시간 가상 Ti 프로파일**: #3이 PCS 탑재, 약 10 ms급 실행, 진단 결손 대응을 시연했고, #4가 KSTAR 확장을 기술합니다. "KSTAR 최초의 실시간 ML Ti 추정" 같은 주장은 쓸 수 없습니다.
- **고속 진단으로 저속 진단 초해상**: #5가 ECE 등을 입력으로 이미 수행했습니다. CER는 그 연구에서 입력이었고 목표가 아니었습니다.
- **시간 초해상이라는 틀**: #6이 PPCF에 이미 있습니다.
- **KSTAR CES에 신경망 적용**: #1이 존재합니다.

남는 좁은 공간은 필자 추론이며 1차 문헌의 주장이 아닙니다. 단일 페데스탈-톱 현, 엄격히 과거인 CES 앵커와 나이, 비보정 저율 스냅샷, hold/GP 대비 증분, 네 검증 설계의 병렬 보고가 그것입니다. 이 조합을 한 논문에서 다룬 기록은 이번 검색에서 보지 못했지만, #6 전문과 #4의 KSTAR 부분을 읽기 전에는 확정할 수 없습니다.

## 3. 충돌과 불일치

- **RTCAKENN 실행시간**: #3은 약 13 ms, #4는 약 8 ms(DIII-D에서 10 ms 미만이라는 표현도 #4에 있음)로 적습니다. 앙상블 수, 측정 범위, 버전 차이일 수 있으나 미확인입니다. 본 과제의 10 ms 예산을 선행 연구와 비교할 때 계산시간과 종단 지연을 구분해야 합니다.
- **KSTAR UFCES 상태**: #2는 설계와 시뮬레이션이고, #8 발췌는 KSTAR를 "시연" 목록에 넣습니다. 그 인용이 #2를 가리킨다면 과대 인용일 수 있습니다. 본 원고는 "설계 성능"으로만 인용하는 편이 안전합니다.
- **Diag2Diag 제목·저자 변동**: arXiv 판마다 제목과 저자 순서가 달라, 인용 시 Nat. Commun. 최종본을 확인해야 합니다.

## 4. 검증할 가치가 있는 미해결 질문 3개

1. **과거 CES 앵커를 조건으로 둔 뒤에도 BES/ECEI 스냅샷이 증분 정보를 주는가?** #5는 목표 진단의 과거값을 입력에서 뺐기 때문에 이 조건부 질문에 답하지 않습니다. 반증 조건: 네 설계 모두에서 고속 추가 모델과 이력 전용 쌍둥이의 차이가 사전등록 허용폭 안에 들거나, 순열·공통성 대조와 구분되지 않는 경우.
2. **F와 L5 중 어느 목표 정의에서 이득이 나타나는가?** 가상진단 문헌(#3, #5)은 교사나 목표를 진리로 취급합니다. 반증 조건: 이득이 L5에서만 나타나고 F에서는 사라지면, 미래 유래 평활 목표에 맞춘 것이지 현재 보고값의 nowcast라고 할 수 없습니다.
3. **무작위 분할의 이득이 시간순·군집 분할에서 유지되는가?** 선행 연구의 분할 방식은 이번 접근으로 확인하지 못했습니다. 반증 조건: rolling-origin과 최종 시간순 블록에서 부호가 뒤집히거나 hold/GP 이하로 떨어지는 경우.

## 5. 유혹적이지만 중복이거나 근거 없는 제안 3개

1. **"KSTAR 최초 실시간 ML Ti 진단"이나 "다중모달 초해상 최초" 표제**: #3, #4, #5, #6과 충돌합니다.
2. **UFCES나 NN-CES를 고속 Ti 진리값 또는 교사로 가정하는 실험**: #2는 시뮬레이션 설계이고, #1의 내용은 미확인이며, 쌍 자료와 원시 스펙트럼이 계약에 없습니다.
3. **Diag2Diag식 MHz 초해상이나 ELM 물리 해석으로 확장**: 10 ms 스냅샷, 비보정 전압, 식별성 가정 부재라는 조건에서는 지지되지 않습니다. 이미 목록에 있는 GRU-D, CRU, KalmanNet 등의 재나열도 새 계획이 아닙니다.

## 6. 다음에 전문 확인이 필요한 항목

- #6의 입력·목표·인과성
- #3의 분할 방식과 CER 부재 실험
- #4의 KSTAR 입력 구성
- #1의 입력과 지연
- 미검색 계보: TCV RAPTOR 실시간 운동학적 평형, JET·AUG의 가상 Ti와 중성자 기반 Ti 추정, CERREAL 원 논문

방법 순위는 요청대로 매기지 않았습니다.
