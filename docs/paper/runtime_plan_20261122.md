# GPU 실행 시간과 다음 대기열 계획

2026-11-22 23:36 KST의 읽기 전용 점검이다. B.85 프로세스를 멈추거나 소스를 고치지 않았고, 실행 중인 GPU에 추가 benchmark를 올리지 않았다. 성능 개선 배수는 측정하지 않았으므로 제시하지 않는다.

후속 결정: 저자의 전체 후보 대기열 설정 요청에 따라 아래 제안 중 runtime probe는 B.89로 등록했고 B.85 완료 뒤/B.88 전에 배치했다. 후보 비교는 B.90의 validation-only 추가 반복 방식으로 등록했다. 아래 마지막 절의 '다음에 등록'과 'B.88 뒤'는 당시 제안이며, 현재 실행 순서는 `queued_experiments_20261122.md`와 실제 등록을 따른다. 속도 점검 결과는 권고이며 기존 등록 backend를 자동 교체하지 않는다.

## 실제 진행 속도

| 정식 fit | epoch 수 | config 기록 → 완료 기록 | 경과 |
|---|---:|---|---:|
| V1_r2_k0, init1701, X | 32 | 21:57:54 → 22:35:59 | 38분 05초 |
| V1_r2_k0, init1701, H | 40 | 22:36:16 → 23:17:38 | 41분 22초 |

두 fit 모두 clean-exit 검증을 통과해 완료로 집계되었다. 이 시간에는 자료 준비·매 epoch 검증·checkpoint 기록도 들어간다. 나눗셈으로 구한 평균은 각각 약 71초/epoch와 62초/epoch이며, 순수 GPU 학습 시간은 아니다.

관측 시점에는 세 번째 fit `V1_r2_k0__1702__X`의 epoch14까지 기록되었다. batch 상태는 완료 2/90, 실패 0이었다. GPU는 약 356 MiB, 사용률 9%, 전력 14.86 W였다. 해당 학습 자식 프로세스의 누적 user+kernel CPU 시간은 경과 시간과 거의 같아, 논리 CPU 16개 중 대략 한 개를 계속 사용하는 양상이다. 이 수치들은 순간 관측 및 OS 누적값이며 정밀 profiler 결과가 아니다.

두 완료 fit의 평균 비용만 같은 것으로 놓으면 90회 학습에 약 60시간, 현재 시점의 남은 학습에 약 58시간이라는 계획값이 나온다. 이는 95% 구간이나 완료 시각 예측이 아니다. 이후 fold의 자료 크기, 초기화별 조기 종료, 다른 작업의 점유가 다르고 예측·bootstrap 판독 비용도 별도다. 따라서 현 관측으로는 **B.85 학습만 대략 며칠 규모**로 봐야 한다. 모든 fit이 cap80까지 가는 경우에는 더 오래 걸린다. B.88 또한 90회이므로, B.85와 같은 fit 비용이라고 가정하면 별도로 비슷한 규모가 더해지지만 B.88의 실제 실행 속도는 아직 측정하지 않았다.

근거 파일: `data/.b85/runs/*/{config,complete,metrics,process_exit}.json`, `data/.b85/logs/`, `docs/paper/b85_status.md`. test 오차는 읽지 않았다.

## 왜 작은 모델인데 느린가

1. **현재는 cuDNN의 fused LSTM 경로를 쓰지 않는다.** 이 설치에서 cuDNN LSTM은 계산 후 native 종료 오류를 냈고, 등록 부록에 따라 같은 LSTM을 native ATen CUDA로 실행한다. 작은 행렬을 여러 시점에 반복 적용하는 순환 계산이므로, GPU 메모리가 남는다고 연산을 한꺼번에 모두 처리할 수 있는 구조가 아니다. 낮은 GPU 사용률과 높은 단일 CPU 점유는 작은 CUDA 작업을 순차적으로 내보내는 host/launch 비용이 크다는 해석과 맞는다. 정확한 점유 비율은 profile 전에는 단정하지 않는다.
2. **검증도 매 epoch 일곱 조건 전부를 재생한다.** 첫 unit의 train은 170,239행/1,328 segment, validation은 33,629행/235 segment다. validation은 일곱 규칙을 모두 돌리므로 235,403행 상당의 forward를 수행한다. batch16 기준 약 83개 train batch와 105개 validation batch다. 검증 자료가 단순히 train의 일부라는 인상보다 실제 비용이 크다. train에는 역전파도 있으므로 이 횟수만으로 검증이 전체의 몇 %인지는 알 수 없다.
3. **호스트와 GPU 사이의 동기화가 잦다.** `compute_loss`와 `validate`의 mask 합에 대한 Python 분기, scalar `.item()`, batch별 `.to(device)`, packed sequence 처리가 있다. 여기에 매 epoch CPU padding/tensor 구성과 checkpoint 기록도 있다. 이들은 소스에서 확인되는 비용 후보이며, 각각의 절대 시간은 아직 나누어 측정하지 않았다.
4. **90은 모델 구조 수가 아니라 반복 실행 수다.** 네 검증 설계의 15 unit × 초기화 셋 × full/twin 둘이다. 단일 fit의 반복학습과 일곱 가용 조건 검증에 이 반복 수가 곱해진다.

작은 VRAM 점유를 보고 현재 batch를 임의로 키우거나, 완료한 run만 남겨서 seed/fold를 줄이는 것은 해결책으로 적용하지 않는다. optimizer update와 검증 설계가 달라진다.

## 지금 안전하게 할 일

- B.85는 현재 고정 코드와 등록된 조기 종료 규칙 그대로 이어 간다. B.88은 등록대로 B.85의 90개 clean fit와 dependency 완료를 기다린다. 두 배치를 중단하거나 순서를 바꾸지 않는다.
- 다음 후보들은 등록·CPU 입력 감사·설계 검토·실행 계획을 준비해 dependency queue에 놓을 수 있다. GPU fitting은 기존 배치와 겹치지 않는다. 준비 완료와 fitting 완료를 구분해 상태를 표시한다.
- 진행 보고는 전체 queue의 시작 시각 하나보다 **현재 fit, 완료 fit 수, 현재 epoch, 최근 완료 fit의 시간**을 보여 준다. 완료 fit가 더 쌓이면 최근 실행들의 실측 분포와 자료 크기를 함께 제시한다. 아직 두 fit뿐인 숫자로 정밀한 ETA를 약속하지 않는다.
- source/config/clean-exit가 일치하는 완료 fit는 그대로 재사용한다. 가용 조건별로 모델을 새로 맞추거나, 이미 끝난 seed를 이유 없이 재실행하지 않는다.

## 속도 개선 후보: 적용 범위와 필요한 검증

| 후보 | 기대하는 작동 원리 | 현재 B.85/B.88에 바로 적용 가능한가 | 다음 단계 |
|---|---|---|---|
| 자료·정규화 segment cache의 unit별 재사용 | 같은 unit의 X/H·초기화마다 반복하는 CPU 준비를 공유 | 실행 소스 해시가 고정돼 있으므로 수정하지 않음 | 새 실행 경로에서 원본 입력·순서·mask·통계의 일치를 먼저 검증. 전체 속도의 지배 요인이라고 가정하지 않음 |
| validation batch·입력 tensor cache | 반복 CPU padding/copy 감소 | 현재 코드·검증 합산 순서 변경은 보류 | 새 등록에서 같은 row별 출력, 검증 MSE, epoch 선택의 재현 여부 확인 |
| scalar 동기화 축소·GPU상 loss 집계 | host 대기와 작은 kernel/동기화 반복 감소 | 고정된 trainer에 패치하지 않음 | 실제 batch의 loss/gradient/optimizer update를 기준 경로와 대조. 합산 순서가 달라지면 수치 차이도 기록 |
| `torch.compile`/CUDA graph 또는 다른 fused recurrent 경로 | host launch 비용 축소 | 별도 경로이며 성공·속도 모두 미측정 | 동적 길이·packed sequence·dropout RNG·streaming 동등성부터 검사. 무조건 지원되거나 빨라진다고 가정하지 않음 |
| 별도 환경에서 cuDNN 종료 문제 해결 | 안정적인 fused GPU LSTM 복구 | 실행 중인 환경·driver를 교체하지 않음 | 독립 환경에서 실제 trace의 forward/backward·정상 종료·저장 복원·step 일치 검사 후 같은 계산 예산을 측정 |
| batch 확대·길이별 재배열·짧은 TBPTT | GPU에 더 큰 일감을 주거나 연산량 축소 | **학습 실험 변경** | 새 등록 필요. update 수, segment 문맥, shuffle/mini-batch 구성 변화를 통제하고 별도 비교 |
| epoch cap/seed/fold/검증 규칙 수 축소 | 총 실행 수·평가량 자체 축소 | **검증·선택 기준 변경** | 새 등록 필요. 현재 결과에 섞거나 사후로 유리한 부분만 채택하지 않음 |
| 다른 architecture 후보의 작은 사전 예산 | 비싼 전체 비교 전 실행 가능성 확인 | **새 후보 배치** | 제한된 pilot의 자원·validation 판정·진행/중단 기준을 먼저 등록. pilot test 성능으로 임의 선별하지 않음 |

소스만 빨라 보이게 만드는 변경도 reduction/RNG/epoch 선택에 영향을 줄 수 있다. 현재 배치는 source hash까지 고정되어 있으므로, 과학적 의미가 같을 것으로 기대되는 최적화 역시 기존 run 중간에 섞지 않는다. 별도 등록 또는 미리 명시한 실행 구현 검증을 거쳐 새 source hash로 실행한다.

## 다음 throughput scout의 최소 계획안

이 문단은 실행 전 제안이며 등록을 대신하지 않는다. **현재 B.85와 B.88의 fitting dependency가 끝난 뒤** 다음을 별도 등록한다.

1. 현 native ATen CUDA를 기준 경로로 두고, 원본 real segment·동일 초기 가중치·동일 batch/길이/precision을 사용한다. raw 자료나 식별 정보는 외부 서비스로 보내지 않는다.
2. `diagnose_runtime85.py`의 실제 trace를 이용해 후보 cuDNN 환경의 정상 종료부터 검증한다. 정상 종료·유한한 gradient·save/load·streaming 비교를 모두 통과해야 timing으로 간다. native 오류를 exit bypass로 감추지 않는다.
3. 별도 optimizer timing에서는 같은 실제 batch 순서·update 수·seed·validation 일곱 규칙을 고정한다. 준비/host-to-device/forward/backward/optimizer/validation/checkpoint를 분리해 측정하고 총시간도 함께 보고한다. 단순 GPU kernel 시간만 비교하지 않는다.
4. 변경한 경로의 출력·loss·gradient·update·dropout/RNG 재현성을 먼저 비교한다. 일치 허용치와 반복 횟수를 등록한 후 판독한다. 실제로 측정한 개선만 제시하고, 개선이 없으면 현재 안전한 경로를 유지한다.
5. 미래 후보의 queue 예산은 이 실측 후 정한다. 동일한 GPU 시간 또는 동일 update 예산 중 무엇을 비교할지 미리 고정하고, 시간 예산 때문에 줄인 검증을 완료된 네 설계·세 초기화 결과처럼 부르지 않는다.

권고: **현재 배치를 유지하면서 다음 후보들의 CPU 준비와 등록을 끝내고, GPU 실행은 의존 순서대로 예약한다. 속도 자체는 별도 scout로 검증한다.** 가장 먼저 시도할 개선 가설은 host launch/동기화 비용 감소 또는 정상 종료 가능한 fused backend 복구다. 지금 관측은 VRAM 부족이나 모델 parameter 수가 주된 병목이라는 설명을 지지하지 않는다.
