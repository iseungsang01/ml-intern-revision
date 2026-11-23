# 검산 후 연구 계획 보완 검토

원래 네 단계 Claude 초안은 보존했다. 코드에서 확인한 skill 부호·모델·가용 규칙을 제공해 Opus가 후속 계획을 보완했다. 아래 제안은 TEST-informed 탐색이며 새 학습은 실행하지 않았다.

**비교 정의:** skill은 초기화별 `1 − MSE(TCN 진단 포함) / MSE(TCN 이력만)`의 평균이다. 양수는 진단 포함 모델의 MSE 감소, 음수는 증가를 뜻한다. 과거 LSTM 2단계 estimator와의 비교가 아니다. D 규칙은 미래 예측 시간이 아니라 CES 이력의 가용 지연이다.

## 작업자 확인

Opus 응답 후 원래 final gate·감사·게시 hash를 다시 확인했다. 초안이 제시한 즉시 조치인 gate 재확인은 이미 완료했다. 후속 후보는 아직 실행하지 않은 탐색적 제안이며 별도 사전등록과 독립 확인이 필요하다.
Claude 요약의 포괄적인 지연 조건 표현이나 고정 cluster 개수는 셀별 원본값을 우선한다. 아래 분류는 원본 구간에서 직접 생성했다. D50의 F와 L5처럼 표적별로 판정이 다를 수 있다. 구간에 0이 포함되는 것은 차이가 없다는 증거가 아니다.

| Scope | Design | 양수·0 제외 셀 | 음수·0 제외 셀 |
|---|---|---|---|
| FOUR_units_three_seeds | V1 | D1000/F, D1000/L5, D300/F, D300/L5, D650/F, D650/L5, S/F, S/L5 | 없음 |
| FOUR_units_three_seeds | V2 | S/F, S/L5 | D10/L5 |
| FOUR_units_three_seeds | V3 | S/F | D10/F, D10/F_full, D10/L5, D100/L5, D50/L5 |
| FOUR_units_three_seeds | V4a | 없음 | D10/F, D10/F_full, D10/L5 |
| FULL15_one_seed | V1 | D100/F, D100/L5, D1000/F, D1000/L5, D300/F, D300/L5, D50/F, D650/F, D650/L5, K10/F, K10/L5, S/F, S/L5 | D10/F_full, D10/L5 |
| FULL15_one_seed | V2 | 없음 | D10/F, D10/F_full, D10/L5, D50/F, D50/L5 |
| FULL15_one_seed | V3 | 없음 | D10/F, D10/F_full, D10/L5, D100/F, D100/L5, D50/F, D50/L5, K10/F, K10/L5 |
| FULL15_one_seed | V4a | 없음 | D10/F, D10/F_full, D10/L5, D50/F |

## 보완된 연구 계획 (Claude 제안)

원본 최종 계획은 그대로 두고 별도의 수정본만 제출합니다. 검증된 정의에 따르면 양수 skill은 같은 행에서 TX(빠른 진단 포함)의 MSE가 TH(이력만 사용)보다 작다는 뜻입니다. D10~D1000은 예측 horizon이 아니라 CES 이력의 가용 지연이고, 원래의 부호 확인 후보는 삭제합니다. FULL15 1-seed: V1은 D50·D100·D300 이상, S, K10에서 0을 배제하는 양수이고 D10/L5와 F_full은 0을 배제하는 음수입니다. V2·V3은 짧은 지연에서 0을 배제하는 음수이며 긴 지연에서는 대체로 0을 포함합니다. V4a(18 cluster)는 D10에서 음수이고 나머지는 대부분 0을 포함합니다. FOUR-unit 3-seed: V1은 D300·D650·D1000·S에서 0을 배제하는 양수이고, V2는 S만 0을 배제합니다. V3은 D10과 D50/L5·D100/L5에서 0을 배제하는 음수이고, V4a는 D10에서만 0을 배제하는 음수입니다. 두 scope는 합치지 않으며, 0을 포함하는 구간을 차이가 없다는 증거로 쓰지 않습니다. confirm 적합은 TH 23건 중 20건, TX 23건 중 18건이 예산 한도에서 끝났습니다. 따라서 음수 skill은 고정 예산에서의 결과일 뿐 구조가 열등하다는 증거가 아닙니다. B98은 internal-VAL로만 다루며, V3에는 7개, V4a에는 5개 cluster뿐이라 독립 검정이 아닙니다. 원본 gate(post_audit·hash 동일·exit0)는 제출 전에 확인했고, 제출 후 재확인은 다음 조치로 남깁니다.

**바로 다음 단계:** 학습 없이 원본 final gate와 hash를 다시 확인하고, 원본 계획과 이 수정본을 분리해 보존합니다.

### 1. 고정 예산 한도의 영향 점검(사전등록 시에만)

- 가설: V2·V3의 짧은 지연 음수 skill은 TH/TX가 대부분 예산 한도에서 멈춘 것과 관련될 수 있습니다.
- 기존 작업과 중복: 기존 TH/TX 구조, unit, seed를 그대로 쓰고 예산 한 수준만 바꿉니다.
- 최소 실험: FOUR-unit V3에서 사전등록한 확대 예산 한 수준으로 TH/TX를 seed 3개씩 다시 적합합니다.
- 개발 범위: 예산과 수렴 기준은 TRAIN/VAL 학습 곡선만으로 정합니다.
- 독립 확인: TEST에 쓰지 않은 캠페인에서 같은 사전등록 규칙으로 확인합니다.
- 반증 측정: 수렴 비율이 높아져도 D10/D50의 점추정 부호와 구간이 유지되면 기각합니다.
- 중단 기준: 설계 1개, 예산 1수준, seed 3개로 끝내고 추가 탐색은 하지 않습니다.
- 양의 결과 문장: 확대 예산에서 V3의 짧은 지연 음수 skill이 줄었으며, 이는 탐색적 관찰입니다.
- 음의 결과 문장: 확대 예산에서도 V3의 짧은 지연 음수 skill이 유지되었습니다.
- 시간 근거: confirm 적합의 중앙값이 약 1200초 상한에 걸렸고, 확대 예산의 소요 시간은 측정된 적이 없습니다.

### 2. 설계별 D10 F와 F_full의 support 차이 기술(학습 없음)

- 가설: D10/F와 F_full은 행 집합이 달라, FULL15 V1에서 보이는 0 배제 여부 차이의 일부는 support 차이에서 왔을 수 있습니다.
- 기존 작업과 중복: 새 적합 없이 저장된 예측과 기존 행만 씁니다.
- 최소 실험: 설계마다 별도로, F_full에서 등록된 common mask 밖 행의 분포를 기술 통계로 집계합니다.
- 개발 범위: 집계 규칙은 TRAIN/VAL에서 정해 사전등록하고, 설계 간 cluster 교집합은 만들지 않습니다.
- 독립 확인: 미사용 캠페인에서 같은 기술 통계를 반복합니다.
- 반증 측정: mask 밖 행이 무시할 만큼 적거나 비슷하면 support 설명을 기각합니다.
- 중단 기준: 설계별 기술 통계 한 번으로 끝내고 새로운 합산 readout은 만들지 않습니다.
- 양의 결과 문장: F_full의 추가 행은 특정 조건에 몰려 있었으며, 이는 탐색적 기술입니다.
- 음의 결과 문장: F_full의 추가 행에서 뚜렷한 편중은 보이지 않았습니다.
- 시간 근거: 측정된 소요 시간이 없습니다.

### 3. B98 cluster 수의 실현 가능성만 사전 점검

- 가설: B98 V3·V4a는 비중첩 cluster가 너무 적어 M-H 구간 해석이 불안정할 수 있습니다.
- 기존 작업과 중복: 기존 B98 internal-VAL 예측만 사용합니다.
- 최소 실험: 재집계 전에 설계별 비중첩 cluster 수가 사전등록한 최소 기준을 넘는지만 확인합니다.
- 개발 범위: internal-VAL 범위에서만 다루고 B94 TEST와는 비교하지 않습니다.
- 독립 확인: cluster가 충분한 미사용 캠페인이 있을 때만 확인합니다.
- 반증 측정: 기준을 넘는 설계가 없으면 이 재집계는 수행하지 않습니다.
- 중단 기준: 최소 기준에 못 미치면 추정 없이 멈춥니다.
- 양의 결과 문장: 일부 설계에서 기준 이상의 비중첩 cluster가 확인되었고, 해석은 internal-VAL에 한정됩니다.
- 음의 결과 문장: 비중첩 cluster가 부족해 B98 재집계는 수행하지 않았습니다.
- 시간 근거: 측정된 소요 시간이 없습니다.

## 지지되지 않는 주장

- D10~D1000을 예측 horizon으로 해석하는 것
- TX와 TH의 차이를 manuscript GP/tree 또는 기존 LSTM과의 비교로 해석하는 것
- 고정 예산에서 나온 음수 skill을 TCN 구조의 열등성으로 보는 것
- 설계 간 cluster 교집합이나 유의성 표시가 바뀐 것을 근거로 효과가 다르다고 주장하는 것
- FULL15와 FOUR-unit 결과를 합치거나 B98을 독립 검정으로 보는 것
- 보고된 CES target을 물리적 참값으로 보거나, 측정된 결과를 물리 기작으로 해석하는 것

## 검산된 조건별 paired skill

두 범위와 네 검증 설계를 구분한다. 아래는 현재값 전체 support와 지연 조건의 예시이며, 전체 등록 조건은 b94_methods.md/json에 있다. point [95% cluster interval], % 단위. F/L5는 보고값 표적이며 구간은 재학습/모델선택 불확실성을 포함하지 않는다.

| Scope | Design | Cell | Skill % [95% interval] | 0 제외 |
|---|---|---|---|---|
| FOUR_units_three_seeds | V1 | D10/F_full | -8.9 [-34.6, 4.5] | False |
| FOUR_units_three_seeds | V1 | D650/L5 | 30.9 [13.0, 44.6] | True |
| FOUR_units_three_seeds | V2 | D10/F_full | -6.0 [-21.0, 4.1] | False |
| FOUR_units_three_seeds | V2 | D650/L5 | 12.5 [-8.2, 30.6] | False |
| FOUR_units_three_seeds | V3 | D10/F_full | -36.4 [-57.4, -22.4] | True |
| FOUR_units_three_seeds | V3 | D650/L5 | 11.0 [-4.8, 23.6] | False |
| FOUR_units_three_seeds | V4a | D10/F_full | -24.1 [-44.7, -11.4] | True |
| FOUR_units_three_seeds | V4a | D650/L5 | 1.4 [-33.6, 34.0] | False |
| FULL15_one_seed | V1 | D10/F_full | -7.1 [-11.8, -2.7] | True |
| FULL15_one_seed | V1 | D650/L5 | 30.6 [23.7, 36.4] | True |
| FULL15_one_seed | V2 | D10/F_full | -36.2 [-54.9, -20.8] | True |
| FULL15_one_seed | V2 | D650/L5 | 2.2 [-28.8, 23.8] | False |
| FULL15_one_seed | V3 | D10/F_full | -63.6 [-91.6, -41.4] | True |
| FULL15_one_seed | V3 | D650/L5 | 7.2 [-11.2, 20.3] | False |
| FULL15_one_seed | V4a | D10/F_full | -32.2 [-49.1, -18.4] | True |
| FULL15_one_seed | V4a | D650/L5 | 5.3 [-35.8, 38.3] | False |
