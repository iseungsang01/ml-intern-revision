# 후속 실험 자동 대기열

저자 승인: "다 세팅해줘 근데 소요 시간이 그렇게 길어?" (2026-11-22). Claude 방법 조사에서 추린 다섯 우선 방향을 아래 등록 실험으로 연결한다. 문헌 조사의 겹치는 제안 전체를 독립 모델로 구현한 것은 아니다. B.69·MAST의 옛 GPU 학습은 이 대기열에 포함하지 않는다.

| 순서 | 작업 | 배정 | 시작 조건 |
|---|---|---|---|
| 현재 | B.85 통합 학습 | 90 fits | 이미 실행 중 |
| 다음 | B.89 CUDA 속도·종료 점검 | 환경 네 종류 × 자식 세 번 | B.85 판독까지 정상 완료 + 90 clean fits |
| 다음 | B.88 anchor/current encoder | A/C × 15 units × 초기화 셋 = 90 fits | B.89 점검의 정상 완료, B.85 의존 검증 |
| 다음 | B.90 후보 비교 | 일곱 후보 × X/H × 15 units × 초기화 하나 = 210 fits | B.88 독립 산출물 검사까지 정상 완료 |
| 다음 | B.90 추가 반복 | validation 상위 둘 × X/H × 15 units × 추가 초기화 둘 = 120 fits | 모든 screen fit 정상 완료, 선택 기록 고정 |
| 마지막 | 예측·구간·독립 산출물 검사 | screen 1-seed와 확대 3-seed 결과를 별도 보고 | 모든 필수 fit 정상 완료 |

후보 일곱: G 감쇠 GRU, T causal TCN, R 대각 이산 observer, C 대각 연속시간 observer, U LSTM Huber, V LSTM Student-t, N LSTM 평균/분산. 원 논문의 GRU-D/RKN/CRU를 재현했다고 부르지 않는다. 명시적 시간 입력과 대각 상태를 쓰는 축소 비교이며 구조·파라미터 수를 함께 보고한다. 정식 원형 구현, 별도 장치 확장, 물리 발견은 이 큐의 완료 조건이 아니다.

검증 오차 선택은 모든 15 unit에서 candidate X/B.85 X의 best validation MSE 비를 평균한다. 테스트 예측 전에 상위 둘과 입력 파일 해시를 고정한다. 나머지 후보도 네 검증 설계의 결과를 모두 보고하지만 초기화 하나에 조건부인 탐색 결과다. 테스트 선택·모델 선택·재학습 불확실성을 포함하지 않는 구간임을 밝힌다. 원고와 estimator는 자동 교체하지 않는다.

## 시간과 현재 속도의 의미

처음 두 B.85 fit 실측은 약38분과41분이다. 같은 비용이면 90회가 약60시간이며, 이는 배치 전체의 정밀 ETA나 신뢰구간이 아니다. B.85/B.88 두 배치만 같은 비용으로 단순 환산하면 약5일이다. 후속 후보는 파라미터 수와 실행 방식이 달라 아직 전체 완료 날짜를 확정할 근거가 없다. 모든 추가420 fits가 첫 B.85 두 fit와 같다고 가정하는 비용 환산은 약278시간이며, 현재 B.85 잔여 약58시간을 더하면 약14일 규모다. 이 수치는 예측·판독 시간을 제외한 조건부 계산으로, 실제 후보 소요 시간의 예측값이 아니다. 첫 후보별 완료 기록이 쌓이면 갱신한다.

모든 후보를 세 초기화로 돌리면 B.90만630 fits다. 현재 등록은 모든 후보를 실행하면서 validation으로 반복 예산을 배정해330 fits로 제한한다. epoch cap·조기 종료·데이터·검증 규칙은 같다. TCN의 시간축 병렬성과 작은 observer가 실제로 시간을 줄이는지는 실측 전에는 단정하지 않는다. 현재 LSTM의 낮은 GPU 사용률은 native ATen의 작은 순환 작업과 host 동기화, 일곱 조건의 반복 검증 비용과 맞는다. 근거는 `runtime_plan_20261122.md`다.

B.89는 정상 종료 가능한 더 빠른 cuDNN 경로를 실제 입력으로 점검한다. 실측 성공 전 배수를 약속하지 않고, 권고가 나와도 고정된 B.88/B.90 backend를 조용히 바꾸지 않는다. 적용하려면 해당 구조의 검증과 실행 전 추가 등록이 필요하다.

## 상태 확인과 운영

통합 상태: `py ces_prediction/experiments/queued_suite_20261122.py --status`

계획만 출력: `py ces_prediction/experiments/queued_suite_20261122.py --plan`

실제 상태 원천은 `data/.queued_suite_20261122/status.json`, 단계 로그는 같은 폴더의 `B89.log`, `B88.log`, `B90.log`다. 대기 중에는 각 후속 child가 아직 없을 수 있다. supervisor가 순서와 의존 조건을 보유한다. 현재 학습은 `docs/paper/b85_status.md`와 `data/.b85/logs/`에서 확인한다. B.88/B.90이 시작되면 각 `.b88`/`.b90/queue_status.json`이 세부 단계를 기록한다.

상태는 waiting_dependency / running / failed / blocked_dependency / complete를 구분한다. 실패하면 다음 단계로 넘어가지 않고 자동 재시도하지 않는다. 재개 전 원인과 현재 source/config를 검사한다. 같은 해시·정상 종료가 증명된 fit만 재사용한다. 중복 supervisor는 안전한 Windows PID 조회와 lock으로 막는다. `os.kill(pid,0)`나 다른 프로세스 종료는 사용하지 않는다.

CPU readiness와 진행자의 독립 입력·모델 식·loss·bootstrap 검사가 통과해야 supervisor를 시작한다. GPU smoke와 학습 검사는 의존 배치가 끝난 후 새 자식에서 수행한다. 대기열 설정 완료와 실험 결과 완료는 별개다. 실제 실행 PID·시각은 launch 확인 뒤 아래에 기록한다.

## 실행 확인

2026-11-22 **23:52:49 KST**, 숨김 통합 supervisor **PID955752** 시작. **23:52:55** 상태는 `waiting_dependency`, 의존 대상 `B85`, 예정 단계 `B89/B88/B90`임을 실제 JSON에서 확인했다. stderr는 비어 있었고 기존 B.85 supervisor PID947056도 계속 실행 중이었다. 이때 B.85는2/90 완료, 실패0, `V1_r2_k0__1702__X` 학습 중이었다. B.88/B.90 실제 GPU 자식은 아직 시작하지 않았다.

최종 확인: 실제 입력·분할·가용 규칙, 모델의 streaming/reset/padding, 독립 GRU/observer 식과 NumPy 손실, 실제 B.87 예측의 명시적 재표집 bootstrap, 조기 판독 차단, B.89 정상 종료 CPU preflight 통과. 저장소 테스트89개 통과, 기록 검사 출처 없는 수치0. 코드·등록·배선은 main에 커밋·푸시했고 자료·키·다른 사용자의 발표 파일은 포함하지 않았다.
