# 학습 효율과 후보 축소: Fable 검토의 채택·수정

2026-11-23. 저자는 두 가지를 따로 분석하고 실제 진행하라고 요청했다: 현재 모델의 학습을 빠르게 하는 방법, 그리고 근거가 강한 후보로 줄이는 방법. 직전의 전체 후보 대기열을 유지하면서 시간을 추정하는 것으로는 요청을 충족하지 못한다. B.85 부록4로 기존 실행 계획을 대체하고, 새 B.91/B.92를 실행 전에 등록·푸시했다. TEST 성능으로 계획을 고르지 않았다.

## 실제 Claude API 사용

공식 API 모델 목록에서 `claude-fable-5-2`를 확인하고 그 모델로 두 관점의 검토를 받았다. Opus로 대체하지 않았다. 정확한 비식별 요청·응답·검색·usage는 ignored `data/.fable_efficiency_20261123/`, 최종 답변 전문은 `fable_efficiency_20261123.md`, 재현 도구는 `ces_prediction/research/fable_efficiency_20261123.py`다. 원시 배열·식별 정보·키·TEST 점수는 보내지 않았다.

초기 `thinking:disabled` 설정은 Fable에서 거부됐다. adaptive/high 설정의 두 응답은 생각 토큰에 한도를 써서 최종 답변 없이 끝났으며 완료된 검토로 취급하지 않았다. 같은 모델의 adaptive/low, 검색 최대 한 번인 별도 두 요청은 모두 `end_turn`으로 최종 답변을 반환했다. 성공 응답 네 번의 보고된 합계는 입력72,011/출력26,955토큰이다. 이 중 미완성 두 응답의 입력36,958/출력16,623도 포함해 낭비를 숨기지 않는다. 저자의 실제 모델 성능을 검토한 독립 채점이나 수락 가능성 평가는 아니다.

## 1. 현재 모델의 학습 방법

Fable의 핵심 권고는 profile → 동일한 검증 목적함수의 cache/집계 → dense/packed 실행 경로 실측 순서다. 현재 낮은 GPU 사용률만으로 메모리를 더 쓰면 빨라진다고 결론 내리지 않는다. 작은 native ATen LSTM의 시점별 launch/동기화와 매 epoch 일곱 조건 재검증이 병목 후보다. 두 완료 fit의 약40분만으로 이후 모든 모델 비용을 같다고 가정한 이전 추정은 계획용 조건부 환산이었다. 이후 한 fit는 더 길어져 실제 편차도 확인됐다.

| 이번에 실행 | 원래 학습 문제와의 관계 | 채택 조건 |
|---|---|---|
| native packed/dense CUDA, cuDNN packed/dense, CPU native와 MKLDNN 경로 비교 | 실제 입력·동일 초기 가중치·FP32로 실행 경로 비교 | 유효 위치 출력·loss·gradient·AdamW update·stream/reset 일치, 독립 자식 세 번 정상 종료 |
| 오른쪽 padding의 dense 인과 LSTM | 단방향·행별 LayerNorm·유효 target mask일 때 과거의 유효 출력이 padding에 의존하지 않음 | 마지막 padded hidden state 사용 금지, padding/future 변이 검사; dropout RNG 변화는 새 학습 경로로 명시 |
| 검증 입력과 batch 재사용, device에서 합계/개수 집계 | 원래 목적함수의 구현 개선 | 규칙마다 head별 SSE/count → 비어 있지 않은 head 평균 → 규칙 평균을 유지하고 독립 NumPy 계산과 대조 |
| 짧은 새 예산·선택된 후보만 반복 | 통계적 설계 변경 | 새 B.92로 등록, 기존80epoch 결과와 같은 예산의 비교라고 부르지 않음 |

이번에 AMP·compile·학습 batch 확대·TBPTT·검증 간격 변경은 넣지 않는다. 여러 축을 동시에 바꾸면 속도와 선택 목적함수의 영향을 분리하기 어렵다. dense만으로 cuDNN 종료 오류가 해결된다고 주장하지 않는다. 과거 dense probe도 종료 실패를 보였으므로 모든 후보에 정상 종료를 요구한다. cuDNN backward는 RNN training flag가 필요해, 동등성 시험은 RNN training/dropout0으로 CPU와 CUDA에 똑같이 적용한다. 실제 새 fit는 dropout.1을 복원한다.

원천 확인: [PyTorch LSTM](https://docs.pytorch.org/docs/stable/generated/torch.nn.LSTM.html)의 특정 persistent algorithm 조건은 V100/float16까지 포함하므로 이 GPU의 FP32에 일반화하지 않았다. [PyTorch issue149194](https://github.com/pytorch/pytorch/issues/149194)는 packed 경로 성능 문제의 실제 보고이며 이 기계의 속도 증거를 대신하지 않는다. 실제 B.91 측정과 clean-exit artifact를 최종 판단 근거로 삼는다. 입력/배치 속도를 전체 epoch 가속률로 부르지 않는다.

## 2. 후보군을 줄이는 방법

Fable도 C/A/TCN 중심의 작은 portfolio를 권했다. 다만 제안한 최대138fit도 비용이 크므로, 부모의 별도 설계 검토를 거쳐 B.92에서 최대58fit로 더 줄였다. 후보를 줄이는 근거는 문제와의 연결·중복·해석 가능성이다. TEST 결과나 성공할 것 같은 숫자를 골라 제외하지 않았다.

| 남기는 구성 | 검증하려는 것 |
|---|---|
| C: B.88에서 anchor embedding만 끈 encoder | A와 동일 구조·파라미터 수인 필수 대조. 원래 B.85 LSTM과 혼동하지 않음 |
| A: CES가 실제 측정된 시각의 fast 입력을 명시한 encoder | 명시적 시각 정렬이 기억만 사용하는 C에 더하는 예측 정보 |
| T: 고정 causal TCN | 검증된 단순 convolutional temporal 대안 하나. 선택될 때만 history-only twin을 추가 |

GRU 변형·두 observer·Student-t/Huber·Gaussian variance는 이번 큐에서 보류한다. 원래 RKN/CRU를 재현하지 않은 작은 observer를 많이 돌려도 원 방법의 유용성을 판단하기 어렵다. 별도 오차 정보나 구간 요구 없이 plain variance head를 추가하는 것도 평균 정확도라는 현재 질문의 우선순위가 낮다. [TCN](https://arxiv.org/abs/1803.01271)은 이 대안을 비교할 근거이고 그 benchmark 결과가 이 자료의 성능을 보장하지는 않는다. [Hyperband](https://arxiv.org/abs/1603.06560)와 [ASHA](https://arxiv.org/abs/1810.05934)의 큰 탐색 공간용 자원 배분을 그대로 도입할 필요는 없다. 작은 고정 pilot와 명시적 validation gate를 쓴다. 짧은 pilot가 느린 수렴을 놓칠 수 있다는 한계는 남는다.

배분은 C/A/T × 네 설계의 고정 unit × seed하나 =12개의 pilot다. 같은 예산의 C 대비 validation 비율만으로 A/T 중 하나를 고른다. 조건 미달이면 확장 없이 멈춘다. 선택 pair(A/C 또는 T-X/H)만 전체15unit의 seed하나30fit와 고정 네 unit의 추가 두 seed16fit를 실행한다. 모두 새 trajectory로 시작한다. **12+30+16=58**, 기존 미실행420fit 대비 약86% 감소다. 이 수는 smoke를 제외한 최대 배정이며 성공/수렴/정밀도 예측이 아니다.

Pilot cap12epoch/10분, 확인 cap40epoch/20분, 기존 early-stop6과 일곱 규칙 검증은 유지한다. 시간 제한이면 마지막 완전히 검증된 checkpoint만 쓰고 `budget_limited`를 표시한다. 17시간20분은 fit 예산의 합계로 준비·시작·smoke·판독 시간을 뺀 값이다. B.92 supervisor는 전체24시간 상한도 가진다. 실제 가속 경로와 첫 pilot의 실측 후 예상 시간을 갱신한다.

## 채택하지 않은 Fable 해석

- 파일 bootstrap만 쓰라는 조언 대신 기존 shot-cluster bootstrap과 discharge 버전을 유지한다.
- X/H 차이는 예측적 입력 증분이다. 물리 인과나 독립적인 참 온도를 증명하지 않는다.
- B.85 TEST 결과를 이미 읽었다는 Fable의 전제는 사실이 아니다. 해당 readout으로 선택하지 않았다.
- 전체15unit·한 seed와 네 unit·세 seed를 섞어 전체15unit의 초기화 안정성이라고 부르지 않는다. 초기화 관련 진술은 그 부분집합에만 적용한다.
- 구간이0을 포함하는 것을 동등성이나 방법의 본질적 실패로 바꾸지 않는다. 모든 결과는 예산·설계·scope를 붙이고 새 자료/충분한 예산의 반증 측정을 적는다.
- 자동 원고 교체·SOTA·물리 발견·수락 가능성 추정은 하지 않는다. 우선 완결 가능한 직접 비교를 만든다.

정확한 실행 규칙은 `PREREGISTRATION_B91.md`와 부록, `PREREGISTRATION_B92.md`가 우선한다. 초기106fit 대안 메모 `candidate_reduction_notes_20261123.md`는 현재 실행안이 아니다. 소스·입력·loss·판독의 독립 검증 후 실제 queue를 시작하고, 상태/실측은 b91_/b92_ 문서와 ignored cache의 artifact로 기록한다.

## 실제 실행 경로의 검증 결과와 수정

B.91은 일곱 경로의 세 번씩, 총21개 자식이 모두 정상 종료했다. 원래 CUDA packed native의 배치 중앙값은0.625471초, CUDA dense cuDNN은0.056705초로 dropout0 시험에서는11.03배였다. 부모가 모든 결과·exit 해시, 중앙값과 일곱 규칙 목적함수를 별도 코드로 재계산했다. 근거는 `b91_runtime_lead_checks.{md,json}`와 ignored 실제 probe 기록이다.

그러나 **실제 dropout.1 C smoke는 계산을 마친 후0xc0000409로 종료**했다. 따라서11배 경로를 실제 학습에 채택했다고 해석하면 안 된다. 해당 complete 파일은 있어도 exit 증거가 nonzero이므로 완료로 재사용하지 않았고, pilot은 시작하지 않았다. 실패한 CUDA 산출물과 `.b92/queue_status.json`을 보존했다.

B.92 부록2를 새로 커밋한 뒤, cuDNN을 쓰지 않는 실측 경로 가운데 가장 빠른 **CPU dense MKLDNN8**을 명시적으로 선택했다. 배치 중앙값0.116811초, 원래 경로 대비 약5.35배이며 이것도 전체 fit 배수는 아니다. 원래 immutable B.91 결정은 고치지 않고 `.b92/production_override_r2.json`에 원 결정·실패 exit·선택 CPU 결과 세 개의 해시와 규칙을 기록했다. 소스와 캐시는 새 `.b92/cpu_r2/`로 분리하고 입력/모델/loss/선택 목적함수/재표집 검증을 다시 실행한다. 새 CPU dropout.1 smoke가 정상 종료해야 실제 pilot을 시작한다. 기존24h clock은 이어받아 이미 쓴 시간을 지우지 않는다. 중간 fit의 device를 바꾸거나 실패를 성공으로 포장하지 않는다.

**실행 확인:** CPU_r2의 실제 dropout.1 C/A/T smoke는 모두 정상 종료0을 확인했다(각 약20초 전체 자식 시간). 2026-11-23 02:06:13 KST에 supervisor PID954168이 pilot 단계로 넘어갔다. 실제 학습은 새 경로에서 진행하며, 소스/입력/목적함수/gradient/정상 종료 감사와 저장소89테스트를 통과했다. Pilot 성능은 아직 해석하지 않았고 TEST는 선택에 쓰지 않는다. 현재 상태는 `b92_status.md`와 live JSON을 따른다.
