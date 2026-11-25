# B114 밤샘 실행 목록

로컬 GPU 한 개에서 아래 후보를 순차 학습한다. 이전 B113은 재학습하지 않는다. 사전등록: `ces_prediction/experiments/PREREGISTRATION_B114.md`.

| 후보 | 고정 설정 | 본 학습 수 |
|---|---|---:|
| CatBoost 얕은 트리 | 깊이4 / 1000회 | 24 |
| CatBoost 깊은 트리 | 깊이8 / 1000회 | 24 |
| CatBoost 학습 연장 | 깊이6 / 4000회, 마지막 모델 | 24 |
| CatBoost 규제 강화 | 깊이6 / 1000회 / L2=10 | 24 |
| ResNet128 | 잔차 MLP 블록3개 | 24 |
| PLE128 | TRAIN 분위수 기반 수치 입력 인코딩+MLP | 24 |

각 후보는 직접형·GP 잔차형 × 초기화3개 × 검증 설계4개다. 총144개 본 학습과 실자료 스모크3개다. 신경망은 적합마다 최대80epoch·patience8·180초 예산이며 완료epoch/VAL 뒤 시간제한을 확인한다. CatBoost는 지정 반복 횟수만큼 학습한다. 전부 실제 자료이며 입력·분할·표적은 기존과 같다.

선택은 **각 외부 검증 설계 안의 VAL만** 사용한다. 설계 간 VAL을 합치면 다른 설계의 TEST와 겹칠 수 있어 금지한다. 각 계열에서 해당 설계의 일곱 VAL조건 평균 상대 MSE가 최소인 설정·형태를 고정한다. 전체 학습이 정상 종료된 뒤 선택한 모델만 동일한32개 TEST조건에서 평가한다. 선택·학습 조건부 구간이며 독립 확증이 아니다.

순서: 해시·GPU 확인 → 스모크·독립 예측 재현 → CatBoost96학습 → 신경망48학습 → VAL선택 고정 → 두 계열 TEST예측 → 독립 트리/신경망 재현·구간 검산 → 결과·기록 게시와 커밋·푸시. 실패 시 부분 결과를 보존하고 종속 단계를 중단하며 자동 재시도·추가 후보 생성은 없다.

GPU 작업 경계는2026-11-26 08:00 KST, CPU검산 경계는09:00 KST다. 더 일찍 끝나면 추가 학습을 채우지 않는다. 다른 GPU 프로세스를 종료하지 않는다. 현재 상태: `docs/paper/b114_status.md`, 실시간 상세: ignored `data/.b114/workflow/status.json`, `data/.b114/{cat,neural}/progress.json`.

비교 기준은 분리한다: matched HGB, 이전 B113 CatBoost, 현행 GP+fast. 각 점추정과95%구간·영제외 여부를 함께 보고한다. 모든 검증 설계를 보존하며 결과가 불리해도 보고한다. 이번 배치는 실시간10ms 지연을 측정하지 않는다.

Claude 공식API는 직전 시도에서 HTTP401을 반환했다. 실패 기록은 보존하며 인증을 바꾸거나 같은 요청을 반복하지 않는다. B114 Claude 검토가 완료됐다고 주장하지 않고 독립 구현·검토·수치 검산으로 진행한다.

구현 근거: [CatBoost 공식 학습 설정](https://catboost.ai/docs/en/references/training-parameters/common), [PyTorch AdamW](https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW). 설치된 실제 버전과 전체 설정은 각 적합 영수증에 보존한다.
