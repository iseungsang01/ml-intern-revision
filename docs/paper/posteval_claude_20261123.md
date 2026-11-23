# 평가 후 Claude 연구 정리 초안

원래 평가의 정상 종료와 독립 최종 감사가 통과한 뒤 실행했다. 아래 수치 표는 원본 JSON에서 직접 생성했으며, 후속 연구는 Claude가 제안한 TEST-informed 탐색 초안이다. 새 학습·원고 수정·물리 발견·수락률 추정은 실행하지 않았다.

Fable 근거 검토 → Opus 반대 검토 → Fable 연구 설계 → Opus 최종 수정. 원자료·개인 정보·키를 전송하지 않고 익명 집계와 명시된 한계만 제공했다. 실제 API 사용·근거 hash는 같은 이름의 JSON에 있다.

## 다음 연구 제안

기존 계획을 두 리뷰와 대조해 고쳤습니다. 후보 3건은 모두 TEST를 본 뒤의 탐색적 제안입니다. /b94/scopes는 크기 제한 때문에 읽지 못해 부호 정의를 아직 확인하지 못했습니다. FULL15 1-seed와 FOUR-unit 3-seed 결과는 합치지 않습니다. B98은 internal-VAL로만 다룹니다.

**바로 다음 단계:** 학습 없이 /b94/scopes의 더 좁은 하위 포인터를 읽어 paired 지표의 비교 대상과 부호 정의를 확정합니다.

### 1. 기존 행의 공통 support 재집계(학습 없음)

- 가설: 설계 간 paired 패턴 차이와 D10의 F/F_full 차이 중 일부는 support 차이에서 생긴 것일 수 있습니다.
- 기존 작업과 중복: B94의 기존 paired 행과 저장된 예측만 다시 쓰며 새 모델은 만들지 않습니다.
- 최소 실험: V1·V2·V3·V4a의 공통 cluster와 F/F_full 공통 행에서 paired 차이와 cluster 구간을 다시 집계합니다.
- 개발 범위: 공통 support 규칙은 TRAIN/VAL만으로 정해 사전등록합니다.
- 독립 확인: 사용하지 않은 캠페인에 같은 규칙을 적용해 확인합니다.
- 반증 측정: 공통 support에서도 부호와 0 배제 여부가 같으면 이 가설은 기각합니다.
- 중단 기준: 사전등록한 셀만 한 번 재집계하고 끝냅니다.
- 양의 결과 문장: 공통 support로 재집계하자 0 배제 패턴이 바뀌었으며, 이는 탐색적 관찰입니다.
- 음의 결과 문장: 공통 support로 재집계해도 설계 간 패턴은 그대로였습니다.
- 시간 근거: 측정된 소요 시간이 없습니다.
- 근거 JSON pointer: `/b94/paired_by_design/FULL15_one_seed/V1`, `/b94/paired_by_design/FOUR_units_three_seeds/V1`, `/reviews/adversarial_review`

### 2. FOUR-unit 단기 horizon의 학습 예산 민감도

- 가설: D10/D50의 paired 차이는 대부분의 적합이 예산 한도에서 끝난 영향을 받았을 수 있습니다.
- 기존 작업과 중복: 같은 모델, unit, seed를 쓰고 예산만 바꿉니다.
- 최소 실험: FOUR unit의 V1·V3에서 TH/TX를 사전등록한 확대 예산 한 수준으로 다시 적합합니다.
- 개발 범위: 예산과 수렴 기준은 TRAIN/VAL 학습 곡선으로만 정합니다.
- 독립 확인: 사용하지 않은 캠페인에서 같은 비교를 반복합니다.
- 반증 측정: 예산을 늘려도 부호와 0 배제 여부가 같으면 이 가설은 기각합니다.
- 중단 기준: 예산 1수준, 설계 2개, seed 3개만 돌리고 추가 탐색은 하지 않습니다.
- 양의 결과 문장: 예산을 늘리자 4개 unit에서 D10/D50 구간의 0 배제 여부가 바뀌었으며, 이는 탐색적 결과입니다.
- 음의 결과 문장: 예산을 늘려도 D10/D50 패턴은 유지되었습니다.
- 시간 근거: 확대 예산에서의 소요 시간은 측정된 적이 없습니다.
- 근거 JSON pointer: `/b94/budget_summary`, `/b94/paired_by_design/FOUR_units_three_seeds/V3`

### 3. B98 M-H 차이를 cluster 비중첩 행으로 재점검

- 가설: B98의 단기 M-H 차이는 cluster 중첩에 의존할 수 있습니다.
- 기존 작업과 중복: 이미 알려진 grey-box 비교를 다시 쓰는 것이며, M-H는 ECEI/BES와 NBI를 함께 더한 효과입니다.
- 최소 실험: 중첩 cluster를 뺀 internal-VAL 행에서 기존 M·H 예측을 다시 집계합니다.
- 개발 범위: TRAIN/VAL 안에서만 수행하고 B94 TEST와는 비교하지 않습니다.
- 독립 확인: cluster가 겹치지 않는 미사용 캠페인에서 확인합니다.
- 반증 측정: 비중첩 행에서도 구간이 같은 부호로 0을 배제하면 이 가설은 기각합니다.
- 중단 기준: 비중첩 cluster가 사전등록한 최소 개수에 못 미치면 추정하지 않고 멈춥니다.
- 양의 결과 문장: 비중첩 internal-VAL에서는 M-H 구간이 0을 포함했으며, 이는 탐색적 관찰입니다.
- 음의 결과 문장: 비중첩 internal-VAL에서도 M-H 구간이 0을 배제했지만 해석은 internal-VAL 범위에 한정됩니다.
- 시간 근거: 측정된 소요 시간이 없습니다.
- 근거 JSON pointer: `/b98/units`, `/context/limits`

## 지지되지 않는 주장

- 부호 정의를 확인하지 않은 상태에서 어느 모델이 유리한지 해석하는 것
- FULL15와 FOUR-unit 결과를 합쳐 초기화 안정성을 주장하는 것
- F/L5를 물리적 참값으로 보거나 수송 시간·채널별 기작을 식별했다고 보는 것
- B98 MSE를 B94 TEST와 직접 비교하는 것
- 10ms 아래 계산 지연을 측정부터 전송까지 포함한 실시간 적합성으로 해석하는 것
- 0을 포함하는 구간을 차이가 없다는 증거로 쓰는 것

## 원본에서 생성한 paired skill

모든 설계/두 evidence scope를 함께 읽는다. point [95% cluster interval]; zero 제외 여부이며, 등록 statement의 비열등 threshold 판정과는 별개다.

| Scope | Design | Cell | Rows | Clusters | Point [interval] | Excludes zero |
|---|---|---|---:|---:|---|---|
| FOUR_units_three_seeds | V1 | D10/F | 27566 | 43 | -0.041142 [-0.252989, 0.095155] | False |
| FOUR_units_three_seeds | V1 | D10/F_full | 38107 | 46 | -0.089399 [-0.345956, 0.045062] | False |
| FOUR_units_three_seeds | V1 | D10/L5 | 26669 | 43 | -0.109059 [-0.489664, 0.117913] | False |
| FOUR_units_three_seeds | V1 | D100/F | 27566 | 43 | 0.134821 [-0.023867, 0.253057] | False |
| FOUR_units_three_seeds | V1 | D100/L5 | 26669 | 43 | 0.143750 [-0.114403, 0.325272] | False |
| FOUR_units_three_seeds | V1 | D1000/F | 27566 | 43 | 0.261200 [0.147060, 0.360111] | True |
| FOUR_units_three_seeds | V1 | D1000/L5 | 26669 | 43 | 0.310957 [0.151770, 0.431830] | True |
| FOUR_units_three_seeds | V1 | D300/F | 27566 | 43 | 0.243303 [0.101242, 0.368116] | True |
| FOUR_units_three_seeds | V1 | D300/L5 | 26669 | 43 | 0.299072 [0.089073, 0.464128] | True |
| FOUR_units_three_seeds | V1 | D50/F | 27566 | 43 | 0.076387 [-0.094770, 0.189627] | False |
| FOUR_units_three_seeds | V1 | D50/L5 | 26669 | 43 | 0.055367 [-0.222222, 0.236297] | False |
| FOUR_units_three_seeds | V1 | D650/F | 27566 | 43 | 0.253786 [0.127777, 0.361548] | True |
| FOUR_units_three_seeds | V1 | D650/L5 | 26669 | 43 | 0.309278 [0.130213, 0.445709] | True |
| FOUR_units_three_seeds | V1 | K10/F | 10427 | 41 | 0.135690 [-0.044440, 0.269233] | False |
| FOUR_units_three_seeds | V1 | K10/L5 | 10091 | 41 | 0.160112 [-0.107128, 0.353182] | False |
| FOUR_units_three_seeds | V1 | S/F | 25231 | 41 | 0.188926 [0.093686, 0.258790] | True |
| FOUR_units_three_seeds | V1 | S/L5 | 24422 | 41 | 0.177627 [0.035916, 0.262482] | True |
| FOUR_units_three_seeds | V2 | D10/F | 39785 | 16 | -0.060899 [-0.247729, 0.062290] | False |
| FOUR_units_three_seeds | V2 | D10/F_full | 51505 | 18 | -0.059915 [-0.210409, 0.040670] | False |
| FOUR_units_three_seeds | V2 | D10/L5 | 38765 | 16 | -0.211393 [-0.611990, -0.001227] | True |
| FOUR_units_three_seeds | V2 | D100/F | 39785 | 16 | 0.072022 [-0.103998, 0.215043] | False |
| FOUR_units_three_seeds | V2 | D100/L5 | 38765 | 16 | -0.003396 [-0.306884, 0.166210] | False |
| FOUR_units_three_seeds | V2 | D1000/F | 39785 | 16 | 0.112443 [-0.067425, 0.313203] | False |
| FOUR_units_three_seeds | V2 | D1000/L5 | 38765 | 16 | 0.099303 [-0.114673, 0.325963] | False |
| FOUR_units_three_seeds | V2 | D300/F | 39785 | 16 | 0.117760 [-0.057386, 0.259732] | False |
| FOUR_units_three_seeds | V2 | D300/L5 | 38765 | 16 | 0.091896 [-0.184029, 0.254056] | False |
| FOUR_units_three_seeds | V2 | D50/F | 39785 | 16 | 0.037256 [-0.146166, 0.173709] | False |
| FOUR_units_three_seeds | V2 | D50/L5 | 38765 | 16 | -0.067641 [-0.413416, 0.111630] | False |
| FOUR_units_three_seeds | V2 | D650/F | 39785 | 16 | 0.136625 [-0.028984, 0.302579] | False |
| FOUR_units_three_seeds | V2 | D650/L5 | 38765 | 16 | 0.124593 [-0.082221, 0.305513] | False |
| FOUR_units_three_seeds | V2 | K10/F | 15162 | 14 | 0.085813 [-0.145667, 0.224093] | False |
| FOUR_units_three_seeds | V2 | K10/L5 | 14857 | 14 | 0.017297 [-0.319846, 0.169979] | False |
| FOUR_units_three_seeds | V2 | S/F | 35638 | 15 | 0.187639 [0.032732, 0.322186] | True |
| FOUR_units_three_seeds | V2 | S/L5 | 34735 | 15 | 0.193970 [0.006857, 0.339184] | True |
| FOUR_units_three_seeds | V3 | D10/F | 25123 | 14 | -0.292145 [-0.532838, -0.141157] | True |
| FOUR_units_three_seeds | V3 | D10/F_full | 35289 | 14 | -0.364282 [-0.573614, -0.223980] | True |
| FOUR_units_three_seeds | V3 | D10/L5 | 23933 | 14 | -0.681579 [-1.404068, -0.362520] | True |
| FOUR_units_three_seeds | V3 | D100/F | 25123 | 14 | -0.070580 [-0.252769, 0.051415] | False |
| FOUR_units_three_seeds | V3 | D100/L5 | 23933 | 14 | -0.225627 [-0.605608, -0.039902] | True |
| FOUR_units_three_seeds | V3 | D1000/F | 25123 | 14 | 0.121863 [-0.016119, 0.246594] | False |
| FOUR_units_three_seeds | V3 | D1000/L5 | 23933 | 14 | 0.136860 [-0.057478, 0.275714] | False |
| FOUR_units_three_seeds | V3 | D300/F | 25123 | 14 | 0.045994 [-0.090535, 0.146403] | False |
| FOUR_units_three_seeds | V3 | D300/L5 | 23933 | 14 | 0.008840 [-0.222858, 0.205900] | False |
| FOUR_units_three_seeds | V3 | D50/F | 25123 | 14 | -0.140680 [-0.350556, 0.002507] | False |
| FOUR_units_three_seeds | V3 | D50/L5 | 23933 | 14 | -0.375656 [-0.881883, -0.153564] | True |
| FOUR_units_three_seeds | V3 | D650/F | 25123 | 14 | 0.111674 [-0.007449, 0.214040] | False |
| FOUR_units_three_seeds | V3 | D650/L5 | 23933 | 14 | 0.110198 [-0.048247, 0.235567] | False |
| FOUR_units_three_seeds | V3 | K10/F | 8939 | 13 | -0.032919 [-0.272466, 0.110600] | False |
| FOUR_units_three_seeds | V3 | K10/L5 | 8638 | 13 | -0.161291 [-0.610413, 0.023421] | False |
| FOUR_units_three_seeds | V3 | S/F | 23776 | 14 | 0.145988 [0.000754, 0.239076] | True |
| FOUR_units_three_seeds | V3 | S/L5 | 22751 | 14 | 0.159981 [-0.022039, 0.262553] | False |
| FOUR_units_three_seeds | V4a | D10/F | 10662 | 18 | -0.141317 [-0.313782, -0.041891] | True |
| FOUR_units_three_seeds | V4a | D10/F_full | 20062 | 19 | -0.240895 [-0.447256, -0.113684] | True |
| FOUR_units_three_seeds | V4a | D10/L5 | 10146 | 18 | -0.385427 [-0.959759, -0.068603] | True |
| FOUR_units_three_seeds | V4a | D100/F | 10662 | 18 | -0.053309 [-0.214027, 0.048842] | False |
| FOUR_units_three_seeds | V4a | D100/L5 | 10146 | 18 | -0.105521 [-0.488352, 0.161755] | False |
| FOUR_units_three_seeds | V4a | D1000/F | 10662 | 18 | -0.006639 [-0.224492, 0.166361] | False |
| FOUR_units_three_seeds | V4a | D1000/L5 | 10146 | 18 | 0.069713 [-0.333810, 0.431878] | False |
| FOUR_units_three_seeds | V4a | D300/F | 10662 | 18 | -0.037981 [-0.206500, 0.078734] | False |
| FOUR_units_three_seeds | V4a | D300/L5 | 10146 | 18 | -0.042329 [-0.390854, 0.223814] | False |
| FOUR_units_three_seeds | V4a | D50/F | 10662 | 18 | -0.076285 [-0.234227, 0.020659] | False |
| FOUR_units_three_seeds | V4a | D50/L5 | 10146 | 18 | -0.194628 [-0.641984, 0.090344] | False |
| FOUR_units_three_seeds | V4a | D650/F | 10662 | 18 | -0.027079 [-0.204546, 0.114807] | False |
| FOUR_units_three_seeds | V4a | D650/L5 | 10146 | 18 | 0.013951 [-0.336449, 0.340435] | False |
| FOUR_units_three_seeds | V4a | K10/F | 3738 | 16 | -0.068640 [-0.287498, 0.054366] | False |
| FOUR_units_three_seeds | V4a | K10/L5 | 3587 | 16 | -0.083454 [-0.534376, 0.203009] | False |
| FOUR_units_three_seeds | V4a | S/F | 7551 | 14 | -0.038013 [-0.203849, 0.075568] | False |
| FOUR_units_three_seeds | V4a | S/L5 | 7180 | 14 | 0.008143 [-0.373788, 0.222638] | False |
| FULL15_one_seed | V1 | D10/F | 165418 | 79 | -0.048294 [-0.098713, 0.000328] | False |
| FULL15_one_seed | V1 | D10/F_full | 222672 | 86 | -0.071329 [-0.117606, -0.026520] | True |
| FULL15_one_seed | V1 | D10/L5 | 160343 | 79 | -0.092744 [-0.189134, -0.007796] | True |
| FULL15_one_seed | V1 | D100/F | 165418 | 79 | 0.128804 [0.075146, 0.184547] | True |
| FULL15_one_seed | V1 | D100/L5 | 160343 | 79 | 0.157265 [0.073414, 0.237693] | True |
| FULL15_one_seed | V1 | D1000/F | 165418 | 79 | 0.255534 [0.197523, 0.309178] | True |
| FULL15_one_seed | V1 | D1000/L5 | 160343 | 79 | 0.306647 [0.243622, 0.361192] | True |
| FULL15_one_seed | V1 | D300/F | 165418 | 79 | 0.230917 [0.157468, 0.302052] | True |
| FULL15_one_seed | V1 | D300/L5 | 160343 | 79 | 0.291507 [0.193602, 0.371216] | True |
| FULL15_one_seed | V1 | D50/F | 165418 | 79 | 0.069713 [0.022500, 0.119815] | True |
| FULL15_one_seed | V1 | D50/L5 | 160343 | 79 | 0.064797 [-0.008691, 0.134796] | False |
| FULL15_one_seed | V1 | D650/F | 165418 | 79 | 0.253134 [0.193502, 0.308849] | True |
| FULL15_one_seed | V1 | D650/L5 | 160343 | 79 | 0.306226 [0.236598, 0.363803] | True |
| FULL15_one_seed | V1 | K10/F | 62162 | 72 | 0.142501 [0.065832, 0.222823] | True |
| FULL15_one_seed | V1 | K10/L5 | 60523 | 72 | 0.175740 [0.061640, 0.291967] | True |
| FULL15_one_seed | V1 | S/F | 152203 | 71 | 0.203250 [0.148399, 0.252917] | True |
| FULL15_one_seed | V1 | S/L5 | 147837 | 71 | 0.228956 [0.163657, 0.284747] | True |
| FULL15_one_seed | V2 | D10/F | 165418 | 79 | -0.369224 [-0.603493, -0.180967] | True |
| FULL15_one_seed | V2 | D10/F_full | 222672 | 86 | -0.362319 [-0.549007, -0.207777] | True |
| FULL15_one_seed | V2 | D10/L5 | 160343 | 79 | -0.764881 [-1.332448, -0.352365] | True |
| FULL15_one_seed | V2 | D100/F | 165418 | 79 | -0.105254 [-0.295185, 0.049958] | False |
| FULL15_one_seed | V2 | D100/L5 | 160343 | 79 | -0.286930 [-0.715508, 0.010217] | False |
| FULL15_one_seed | V2 | D1000/F | 165418 | 79 | 0.072438 [-0.113125, 0.216952] | False |
| FULL15_one_seed | V2 | D1000/L5 | 160343 | 79 | 0.030186 [-0.253689, 0.237184] | False |
| FULL15_one_seed | V2 | D300/F | 165418 | 79 | 0.041651 [-0.153865, 0.197129] | False |
| FULL15_one_seed | V2 | D300/L5 | 160343 | 79 | -0.020834 [-0.382544, 0.218393] | False |
| FULL15_one_seed | V2 | D50/F | 165418 | 79 | -0.188711 [-0.382106, -0.024586] | True |
| FULL15_one_seed | V2 | D50/L5 | 160343 | 79 | -0.456291 [-0.909845, -0.129058] | True |
| FULL15_one_seed | V2 | D650/F | 165418 | 79 | 0.067239 [-0.120193, 0.211073] | False |
| FULL15_one_seed | V2 | D650/L5 | 160343 | 79 | 0.021712 [-0.288089, 0.238252] | False |
| FULL15_one_seed | V2 | K10/F | 62162 | 72 | -0.096604 [-0.295368, 0.061007] | False |
| FULL15_one_seed | V2 | K10/L5 | 60523 | 72 | -0.253178 [-0.659390, 0.030911] | False |
| FULL15_one_seed | V2 | S/F | 152203 | 71 | -0.009021 [-0.190377, 0.138244] | False |
| FULL15_one_seed | V2 | S/L5 | 147837 | 71 | -0.107814 [-0.401178, 0.117835] | False |
| FULL15_one_seed | V3 | D10/F | 116471 | 60 | -0.518453 [-0.717474, -0.347711] | True |
| FULL15_one_seed | V3 | D10/F_full | 161744 | 65 | -0.635944 [-0.915694, -0.413918] | True |
| FULL15_one_seed | V3 | D10/L5 | 112199 | 60 | -0.806058 [-1.196699, -0.549590] | True |
| FULL15_one_seed | V3 | D100/F | 116471 | 60 | -0.220177 [-0.339800, -0.120108] | True |
| FULL15_one_seed | V3 | D100/L5 | 112199 | 60 | -0.340646 [-0.588123, -0.169494] | True |
| FULL15_one_seed | V3 | D1000/F | 116471 | 60 | 0.069262 [-0.056618, 0.174975] | False |
| FULL15_one_seed | V3 | D1000/L5 | 112199 | 60 | 0.110498 [-0.073649, 0.241736] | False |
| FULL15_one_seed | V3 | D300/F | 116471 | 60 | -0.034063 [-0.155903, 0.063223] | False |
| FULL15_one_seed | V3 | D300/L5 | 112199 | 60 | -0.050723 [-0.268837, 0.094863] | False |
| FULL15_one_seed | V3 | D50/F | 116471 | 60 | -0.322847 [-0.465441, -0.195590] | True |
| FULL15_one_seed | V3 | D50/L5 | 112199 | 60 | -0.514352 [-0.785996, -0.319950] | True |
| FULL15_one_seed | V3 | D650/F | 116471 | 60 | 0.049048 [-0.081832, 0.151742] | False |
| FULL15_one_seed | V3 | D650/L5 | 112199 | 60 | 0.071812 [-0.111830, 0.203419] | False |
| FULL15_one_seed | V3 | K10/F | 43352 | 57 | -0.220348 [-0.363519, -0.102499] | True |
| FULL15_one_seed | V3 | K10/L5 | 41985 | 57 | -0.324986 [-0.599578, -0.145914] | True |
| FULL15_one_seed | V3 | S/F | 106417 | 56 | -0.003179 [-0.129368, 0.093521] | False |
| FULL15_one_seed | V3 | S/L5 | 102684 | 56 | -0.012164 [-0.159382, 0.105752] | False |
| FULL15_one_seed | V4a | D10/F | 10662 | 18 | -0.193010 [-0.357599, -0.079352] | True |
| FULL15_one_seed | V4a | D10/F_full | 20062 | 19 | -0.322402 [-0.491092, -0.183972] | True |
| FULL15_one_seed | V4a | D10/L5 | 10146 | 18 | -0.319330 [-0.764544, -0.038685] | True |
| FULL15_one_seed | V4a | D100/F | 10662 | 18 | -0.098158 [-0.253680, 0.013727] | False |
| FULL15_one_seed | V4a | D100/L5 | 10146 | 18 | -0.081574 [-0.464599, 0.183915] | False |
| FULL15_one_seed | V4a | D1000/F | 10662 | 18 | -0.030676 [-0.255424, 0.161256] | False |
| FULL15_one_seed | V4a | D1000/L5 | 10146 | 18 | 0.086782 [-0.373304, 0.471108] | False |
| FULL15_one_seed | V4a | D300/F | 10662 | 18 | -0.081390 [-0.244222, 0.041378] | False |
| FULL15_one_seed | V4a | D300/L5 | 10146 | 18 | -0.033645 [-0.409858, 0.242337] | False |
| FULL15_one_seed | V4a | D50/F | 10662 | 18 | -0.116519 [-0.278817, -0.013460] | True |
| FULL15_one_seed | V4a | D50/L5 | 10146 | 18 | -0.156447 [-0.548364, 0.111509] | False |
| FULL15_one_seed | V4a | D650/F | 10662 | 18 | -0.044606 [-0.236694, 0.115190] | False |
| FULL15_one_seed | V4a | D650/L5 | 10146 | 18 | 0.053393 [-0.357631, 0.382733] | False |
| FULL15_one_seed | V4a | K10/F | 3738 | 16 | -0.129298 [-0.343603, 0.008713] | False |
| FULL15_one_seed | V4a | K10/L5 | 3587 | 16 | -0.083582 [-0.522013, 0.228302] | False |
| FULL15_one_seed | V4a | S/F | 7551 | 14 | -0.063318 [-0.182294, 0.036153] | False |
| FULL15_one_seed | V4a | S/L5 | 7180 | 14 | 0.053502 [-0.218956, 0.257520] | False |
