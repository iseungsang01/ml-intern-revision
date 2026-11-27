# B117 complete

B117 COMPLETE / independently audited. Single fixed LSTM TRAIN/VAL diagnostic; CONVERGENCE CRITERION MET. No TEST evaluation, new GP comparison or further fit queued.

동일한 단일 LSTM을 처음부터 학습하여 기존 시간 제한 학습과 전체 학습 곡선, 검증 오차, 학습률 및 갱신 횟수를 비교했다. 등록한 수렴 기준 충족 여부와 안전 한도 종료를 구분했다. 학습 손실은 고정된 학습 표준편차로 물리 단위로 변환했다. 학습 구간은 규칙 혼합과 갱신 중 모수에 의존하고 검증은 고정 지지의 규칙별 평균이므로 두 손실의 차이는 기술적 진단이다. 검증 자료를 반복 사용한 단일 학습이며 독립 표본 일반화, 신뢰구간 또는 TEST 성능을 주장하지 않는다. 수치와 수렴 판정은 독립 감사된 `docs/paper/b117_training.md`와 JSON 및 학습 곡선에서 확인한다.
