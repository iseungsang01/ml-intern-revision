# 후보 축소와 제한 예산의 실행안

**초기 대안 메모이며 현재 실행 규칙이 아니다.** 이후 Fable 검토와 저자의 강한 비용 축소 요구를 반영해 B.92는 C/A/T pilot12회 + 선택 pair 확인46회, 최대58회를 등록했다. 현재 규칙은 `PREREGISTRATION_B92.md`와 `efficiency_decision_20261123.md`를 따른다. 아래106회 안은 검토 경로로 보존한다.

2026-11-23. 부모 요청에 따른 읽기 전용 설계 메모다. B.88/B.90 등록·모델 코드와 method_selection_20261122.md를 읽었고 원천 문헌의 초록·개요를 재확인했다. TEST 예측이나 TEST 오차는 읽지 않았다. 기존 등록을 수정하거나 새 학습을 실행하지 않는다. 아래 숫자는 제안하는 계산 배분이며 측정한 성능·소요 시간 예측이 아니다.

## 남길 세 가지 선택과 필요한 대조

| 과학적 선택 | 역할 | 새 학습에서 필요한 arm |
|---|---|---|
| 직접 LSTM | 현재의 강한 단순 신경망 기준 | LSTM-X; 기존 B.85 X/H 결과는 최종 기준으로 유지 |
| 측정 시각 anchor encoder | 늦게 도착한 CES의 원 측정 시각을 명시적으로 정렬하는 가설 | B.88 A와 C를 함께 유지. A만 남기면 encoder 변경과 anchor 정보 효과가 섞임 |
| 작은 causal TCN | 학습 병렬성과 고정 과거 창을 가진 실제 대안 | screen은 T-X만, 선택될 때만 T-H를 추가 |

따라서 선택지는 세 가지지만, anchor 대조를 지키려면 pilot의 실행 arm은 LSTM-X/A/C/T-X 네 개다. B.88 C는 별도 유망 방법으로 순위를 매길 후보가 아니라 anchor 효과를 분리하기 위한 대조다. 기존 hold/인과 GP/학습된 T0·TX도 최종 기준으로 남기되 다시 학습하지 않는다.

G/R/C/N과 Student-t를 이번 큐에서 뺀다. GRU-D식 감쇠는 시간·나이 입력을 쓰는 LSTM과 질문이 가깝고 현재의 핵심 정렬 가설을 추가로 분리하지 않는다. B.90 R/C는 등록에도 명시했듯 축소된 diagonal observer이며 원래 RKN/CRU의 재현이 아니다. 여기서 낮은 성능이 나와도 원 방법에 대한 결론을 낼 수 없어 계산 대비 해석 가치가 작다. N의 Gaussian NLL은 calibrated uncertainty를 제공하지 않으며, 평균 정확도가 목표인 현재 비교에 불필요한 최적화 축을 더한다. 이러한 제외는 열등하다는 실험적 판정이 아니다.

Huber는 별도 architecture 후보가 아니다. TRAIN/VAL에서만 확인한 잔차 분포와 RMSE/MAE의 상충이 robust loss의 필요성을 지지할 때, 선택한 backbone의 단일 loss 대조로 새로 등록한다. TEST에서 큰 오차를 발견한 뒤 같은 TEST에 맞춰 loss를 고르지 않는다. 당장은 Student-t/Huber/variance head 세 축을 동시에 돌리지 않는다.

## 고정 배분 제안

- Pilot unit은 `V1_r2_k0`, `V2_s1_k0`, `V3_q3`, `V4a`로 결과를 보기 전에 고정한다. 네 설계를 하나씩 포함하며, 어떤 설계도 primary로 부르지 않는다. 모든 원래 TRAIN/VAL 파일을 유지하고 임의의 작은 shot 부분집합으로 평가 문제를 바꾸지 않는다.
- 초기화는 1701 하나. LSTM-X/A/C/T-X 네 arm, 네 unit이므로 **16개의 짧은 pilot fit**이다. 성능을 비교하기 전에 같은 실행 backend, batch 방식, 데이터 순서, validation 및 checkpoint 규칙을 맞춘다. 이미 80 epoch까지 본 LSTM의 최적 validation과 12 epoch 후보를 직접 비교하지 않는다. 동일 예산의 LSTM prefix checkpoint가 보존되어 있고 실행 조건까지 같으면 그 네 fit만 재사용할 수 있다.
- Pilot 한 fit의 상한은 **12 epoch 또는 학습·validation 구간의 15분**, 먼저 도달하는 쪽이다. 데이터 캐시 구성 시간은 따로 기록한다. 시간 제한은 원자적 checkpoint가 가능한 안전 경계에서 적용하고 초과분을 기록한다. 엄밀한 hard deadline이 필요하면 최대 2분 저장 유예 뒤 실패 종료로 처리한다. validation이 한 번도 완료되지 않은 경우 점수를 만들지 않고 자원 한도 미완료로 보고한다. 이런 미완료는 모델의 정확도 열등함을 뜻하지 않는다.
- 선택한 **한 family만** 원래 15 unit, 세 초기화, 필요한 두 arm으로 확인한다. Anchor를 선택하면 A/C, TCN을 선택하면 T-X/T-H다. 각각 **90개의 confirmation fit**, fit당 **40 epoch 또는 30분 + 최대 2분 저장 유예**를 고정 상한으로 제안한다. Pilot checkpoint를 조용히 재사용하거나 training cap을 바꾸지 않도록 confirmation은 별도 새 trajectory로 시작한다. 기존 B.85 전체 예산 기준과는 예산이 다르다는 점을 함께 보고한다.
- 보수적으로 새 fit 상한은 **16 + 90 = 106회**다. 기존 B.88 90 + B.90 330 = 420회 대비 약 75% 감소다. 동일 예산 LSTM pilot을 정당하게 재사용하면 102회다. 이 수는 plumbing smoke를 제외한다. 새 smoke는 필요한 서로 다른 실행 arm당 한 번을 넘기지 않고, 기존 정확히 source-valid한 smoke의 재사용 가능성을 먼저 확인한다.
- 자원 배분의 보수적 상한은 pilot 4시간, confirmation 45시간에 저장 유예를 더한 값이다. 실제 소요 시간의 예측이 아니다. 운영상 전체 상한이 더 짧아야 한다면 시작 전에 그 상한과 미완료 결과의 처리까지 다시 고정한다. 고정 90회를 완수하지 못했는데 완료된 일부만 최종 확인 결과로 발표하지 않는다.

처음부터 모든 losing 후보의 history-only twin과 세 seed를 학습하는 것이 낭비의 주원인이다. 반면 최종 선택 모델의 짝 대조와 세 초기화는 증분과 초기화 변동을 구분하는 데 필요하므로 끝까지 남긴다.

## validation만 쓰는 선택 규칙

Pilot의 checkpoint 선택은 원래처럼 일곱 TRAIN/VAL 가용 규칙과 두 head의 pure normalized MSE다. K10은 여전히 TRAIN/VAL에서 완전히 제외한다. 각 방법의 점수는 네 unit에서 `best_val_MSE(method)/best_val_MSE(fresh matched-budget LSTM-X)`를 계산한 뒤 동일 가중 산술 평균한다. 모든 분모와 점수의 출처·예산·정규화·config hash를 기록한다.

새 family의 승격 조건을 다음처럼 시작 전에 고정할 수 있다.

1. 평균 비가 0.98 미만이고 V3_q3/V4a 각각의 비가 1.05 이하여야 한다. 이 문턱은 pilot 계산을 배분하기 위한 공학적 선택이지 2% 차이의 통계적 유의성이나 비열등성 증명이 아니다.
2. Anchor A에는 추가로 네 unit 중 세 unit 이상에서 A의 validation MSE가 C보다 작아야 한다. 그렇지 않으면 anchor 가설 때문에 A를 승격하지 않는다. A와 C 사이의 최종 통계 판단은 confirmation까지 유보한다.
3. 조건을 만족하는 A/T 중 평균 비가 가장 작은 **하나만** 승격한다. 비 차이가 0.005 이내이면 TRAIN/VAL의 실제 trace로 측정한 CPU step p99가 작은 쪽, 그래도 같으면 고정 코드 순서로 정한다. TEST latency나 TEST 오차는 선택에 쓰지 않는다.
4. 아무 후보도 조건을 채우지 못하면 기존 LSTM을 유지하고 새 confirmation 90회를 실행하지 않는다. 이 경우 pilot 배정에서 승격하지 않았다고만 적으며, 더 긴 학습에서 개선될 가능성을 부정하지 않는다. 부모가 무조건 한 후보 확인을 원한다면 조건 대신 평균 점수 순위로 한 개를 고르는 규칙으로 **실행 전에** 바꿔야 한다.

순위와 모든 입력 validation 값 및 hash를 immutable promotion 기록으로 잠근 뒤에만 선택 family의 TEST 추론을 허용한다. Losing 후보의 TEST 점수를 추가로 보고 싶은 경우에도 선택이 잠긴 뒤 탐색 결과로 분리한다. TEST를 보고 차순위 후보로 교체하지 않는다.

## 해석과 효율성의 한계

- 네 unit·한 초기화·짧은 예산의 screening은 느리게 수렴하는 방법을 놓칠 수 있다. 선택에 충분한 제한된 자원 절차일 뿐, 제외 방법 전체의 열등함이나 특정 구조의 원천적 한계를 검증하지 않는다.
- 최종 확인은 원래 15 unit과 세 초기화를 모두 사용해 네 설계를 보고한다. 이미 사용한 캠페인이며 validation 선택과 모델 탐색을 거쳤으므로 bootstrap 구간은 선택·재학습 불확실도를 포괄하지 않는다. 새로운 독립 확증이라고 쓰지 않는다.
- TCN의 현재 receptive field는 127개 행이다. 시간 격자가 끊기므로 항상 정확히 1.27초의 물리적 문맥은 아니다. 마지막 CES/나이 입력은 유지되지만 anchor 이전의 fast history 접근은 LSTM과 다를 수 있다. 이를 숨기거나 성능이 낮을 때만 창 길이를 바꾸지 않는다.
- 학습 효율은 같은 실제 입력에서 padding 낭비 축소, 정규화·가용 rule cache 재사용, 검증된 backend 병렬화처럼 학습 문제를 바꾸지 않는 방법부터 개선한다. Batch/precision/backend가 바뀌면 frozen source와 단위 결과의 동등성을 먼저 확인하고, 실행 중인 trainer를 수정하지 않는다.
- 기존 B.88/B.90 등록은 그대로 보존한다. 아직 실행하지 않은 대규모 큐를 새 제한 배분으로 대체한다면 새 등록과 명시적 supersession 기록이 필요하다. 이미 끝난 fit과 비용은 삭제하지 않는다.

## 재확인한 원천

[TCN](https://arxiv.org/abs/1803.01271)은 단순 convolutional sequence 모델을 recurrent 모델과 실제로 비교할 근거다. 해당 benchmark 성능을 KSTAR 성능이나 CPU latency로 옮기지는 않는다. [RKN](https://proceedings.mlr.press/v97/becker19a.html)과 [CRU](https://proceedings.mlr.press/v162/schirmer22a.html)는 구체적인 불확실도·동역학 구조를 갖춘 원 방법이며 현재의 단순 diagonal 근사와 구분한다. [Faithful heteroscedastic regression](https://proceedings.mlr.press/v206/stirn23a.html)은 일반 Gaussian NLL에서 평균과 분산 추정이 함께 나빠질 수 있음을 지적하므로, plain NLL head를 손쉬운 정확도 향상 수단으로 우선하지 않는다.
