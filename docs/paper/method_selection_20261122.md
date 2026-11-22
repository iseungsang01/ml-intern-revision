# Pedestal nowcasting 방법 선택 메모

2026-11-22. 저자의 요청에 따라 Claude Opus API의 실제 web search로 네 관점의 후보 조사를 수행했고, 진행자가 주요 원천 문헌과 적용 조건을 별도로 확인했다. 원문·검색 이력·사용량은 [Claude 조사 기록](method_scout_claude_20261122.md)에 있다. 아래 우선순위는 진행자의 판단이며 실험 결과나 수락 가능성 평가가 아니다. B.85 학습은 그대로 유지한다. 이 메모로 새 학습을 실행하지 않는다.

## 무엇을 고르는가

목표는 pedestal-top CES nowcast의 정확도·가용 조건 대응·실시간성을 높이는 것이다. 표준 모델도 유효한 후보이며, 새로운 구조나 새로운 물리 발견을 필수 조건으로 두지 않는다. 다만 현재 자료에서의 성능은 실제 비교 전에는 모른다. F는 CES 보고값이고 L5는 그 이후 값으로 만든 학습 정답이다. 어느 쪽도 독립적인 참 온도가 아니다.

네 Claude 검색 응답은 각각 완료되었고 총 32개 제안 항목을 포함한다. 이 숫자는 서로 다른 방법 32개를 뜻하지 않는다. ODE·SSM·attention·표현 학습의 중복을 구분해야 한다. 마지막 교차 비판 응답은 출력 한도에서 잘렸고 순위도 완성되지 않았다. 따라서 그 부분의 순위를 채택하지 않고 아래처럼 다시 판단했다. 전수 문헌 조사나 문헌상 부재의 증명은 아니다.

## 먼저 비교할 다섯 방향

| 우선 | 방식 | 이 문제에 맞는 이유 | 최소 비교와 반증 조건 |
|---|---|---|---|
| 1 | **Anchor/current encoder로 온도 변화를 직접 학습** | CES가 도착한 시각과 실제로 측정한 시각이 다르다. 현재 빠른 신호와 마지막 CES의 **측정 시각**에서 빠른 신호를 함께 주면, RNN이 오래전 신호와 측정값을 암묵적으로 맞춰야 하는 부담을 줄인다 | 동일한 단일 신경망에서 anchor 입력 유무만 비교. 새 조건과 시간순 설계에서 증분이 없으면 추가하지 않는다. 현재 과제에 맞춘 진행자의 제안이며, Claude가 문헌상 새로움을 확인한 결과가 아니다 |
| 2 | **GRU-D 계열의 시간·결측 인식 모델** | 현재의 마지막 값·나이·flag 입력에 학습 가능한 상태 감쇠를 더하는 작은 비교다. 큰 모델 없이 오래된 정보의 신뢰도를 학습할 수 있다 | B.85와 같은 입력 범위·가용 조건·학습 예산에서 비교. 새 CES가 들어올 때와 오래 끊길 때를 모두 보고, 장기 공백에서 평균으로 치우치면 실패로 적는다. [GRU-D 원논문](https://doi.org/10.1038/s41598-018-24271-9) |
| 3 | **작은 causal TCN** | 과거 창의 다중 시간 척도를 직접 읽는다. 지금 LSTM을 유지해야 할 이유가 성능인지 구조 관성인지 확인할 수 있다 | 모델 크기와 과거 정보 범위를 명시하고 같은 행에서 비교. 최대 학습 지연과 anchor까지 포함할 창이 필요하다. 학습·CPU 추론 시간은 실제로 재며 빨라진다고 가정하지 않는다. [TCN 원문](https://arxiv.org/abs/1803.01271) |
| 4 | **평균 추정과 관측 불확실도를 분리한 학습** | 큰 CES 벗어남을 모두 실제 온도 변화처럼 맞추는 것이 유리한지 시험한다. 원시 스펙트럼 없이도 보고값 예측의 오차 규모는 모델링할 수 있다 | 먼저 같은 backbone의 MSE와 한 가지 robust loss를 비교하고, 별도 비교에서 평균+분산 head를 추가한다. MAE·RMSE·구간 포함률·폭을 함께 보고, 큰 분산으로 어려운 행을 무시하는지 확인한다. [Faithful heteroscedastic regression](https://proceedings.mlr.press/v206/stirn23a.html) |
| 5 | **통합 학습하는 상태공간 관측기: RKN/CRU 계열** | 상태의 시간 진행과 들어온 관측의 갱신을 명시적으로 나눈다. 단순 사후 rule 보정과 달리 모델 전체를 함께 학습할 수 있다 | CES 없는 구간, 복귀, 지연 도착을 재생하고 발산·편향·회복 시간을 비교. 잠재 상태를 물리적 온도라고 단정하지 않는다. [RKN](https://proceedings.mlr.press/v97/becker19a.html), [CRU](https://proceedings.mlr.press/v162/schirmer22a.html) |

첫째 방향은 기존 B.75의 수작업 anchor 스칼라와 계보가 있다. 그 아이디어를 숨기거나 새로운 원리로 부르지 않는다. 차이는 **하나의 학습 모델이 anchor와 현재의 표현 및 변화량을 함께 학습하는가**다. 덧셈형 변화 추정이나 양수성을 가진 출력은 별도 고정 선택으로 등록할 수 있다. 로그 비율은 값의 양수성·결측 처리·공통 채점 행을 확인하기 전에는 기본값으로 삼지 않는다. 측정 시각의 anchor는 추론 시점까지 실제로 확보한 과거 신호에서만 가져온다.

GRU-D로 옮길 때는 '과거 값이 하나라도 있음'과 '이번에 새 관측이 도착함'을 혼동하지 않는다. 새 관측 mask는 실제로 보이게 된 anchor index의 변화에서 만들고, 값의 나이는 원 측정 시각으로 계산한다. 비교 모델에도 동일한 정보 접근을 보장한다. 원 논문의 의료 분류 설정을 그대로 복사하는 실험이 아니다.

## 넓게 찾은 후보와 적용 판정

아래는 중복 이름을 합쳐 비교할 수 있게 만든 목록이다. 모든 행은 **미검증 적용 후보**다. '가능'은 원자료 요구가 대체로 충족된다는 뜻이며 성능이나 배포 가능성이 확인되었다는 뜻이 아니다. 원천 링크가 없는 일반 전략은 확정된 단일 논문 구현으로 취급하지 않는다. 상세 후보별 링크·검색 여부는 Claude 기록에서 확인한다.

| 방식/전략 | 지금 자료로의 적용 | 우선도·핵심 조건 |
|---|---|---|
| 시간 감쇠 GRU-D / time-aware LSTM | 가능 | 높음. Ti 결측·나이와 dense fast 입력의 역할을 구분 |
| 과거 창만 읽는 causal TCN | 가능 | 높음. receptive field와 온라인 상태/창 비용 고정 |
| Anchor/current encoder + 학습한 변화량 | 가능, 진행자의 과제별 제안 | 높음. 측정 시각 정렬과 anchor 제공 효과를 독립 비교 |
| 진단별 encoder + 학습한 fusion | 가능 | 중간. BES/ECEI 단독·합침 및 동일 예산 단순 concat 대조 |
| Student-t likelihood / Huber / MAE | 가능, 서로 다른 목적함수 | 높음. 하나씩 고정 비교; heavy tail이 실제 빠른 변화일 가능성도 평가 |
| 평균+분산 heteroscedastic head, faithful/β-NLL 최적화 | 가능 | 높음~중간. 현재 MSE 모델에 β-NLL만 바꾸는 것이 아니며 분산 head가 필요 |
| Recurrent Kalman Network | 가능, 잠재 상태와 관측 모형을 새로 설계 | 중간. 상태·잡음 식별성과 장기 공백 안정성 |
| Continuous Recurrent Unit | 가능, 추가 구현 필요 | 중간. 확률적 출력이 자동으로 보정된 불확실도는 아님 |
| KalmanNet | 부분 가능 | 중간~낮음. 원 방법의 부분적으로 알려진 상태/관측 모형 요구를 채워야 함 |
| 선형/ARX/작은 상태공간 모형·인과 GP | 가능 | 필수 단순 기준선. 계수나 네트워크 크기보다 실제 오차로 선택 |
| S4D/S5 등 구조적 sequence SSM | 가능, 구현/Windows 실행 확인 필요 | 중간. 긴 기억이 실제로 필요한지, 작은 TCN보다 나은지부터 확인 |
| Mamba 등 selective SSM | 조건부 가능 | 낮음~중간. custom kernel·Windows·CPU 비용을 확인, 이름만으로 우선하지 않음 |
| 작은 causal attention / 시간 attention mTAN | 조건부 가능 | 중간. key/value와 query에 미래 관측이 들어가지 않도록 마스크·정규화 확인 |
| Causal Neural CDE | 조건부 가능 | 낮음~중간. 온라인 보간을 써야 하며 일반 cubic interpolation은 그대로 쓰지 않음 |
| ODE-RNN / Latent ODE / GRU-ODE-Bayes | 조건부 가능 | 낮음~중간. 빠른 입력은 이미 거의 규칙적인 격자다. solver 비용을 감수할 이득이 있는지 확인 |
| Masked fast-diagnostic pretraining | 가능 | 중간. train 파일만 사용하고 추론 encoder는 과거 창만 읽음 |
| TS2Vec / TS-TCC / CPC식 contrastive representation | 조건부 가능 | 중간. 원 구현이 자동으로 causal하다고 보지 않음; augmentation이 진단 의미를 지우지 않도록 제한 |
| Teacher–student distillation | 조건부 가능 | 중간. teacher의 풍부한 정보는 train 단계만 허용. student 추론 정보는 동일하게 유지 |
| 서로 다른 가용성 view의 일관성/비대칭 distillation | 가능, 구체적 목적함수는 새 설계 | 중간. 서로 정보량이 다른 view를 억지로 같게 만들어 강한 view까지 나빠지는지 검사 |
| Inlier/outlier 관측 mixture | 조건부 가능 | 중간. 잔차가 크다는 이유만으로 fit 실패라고 부르지 않음 |
| EnbPI / sequential conformal calibration | 조건부 가능 | 중간. 시계열 가정, 라벨의 실제 도착 지연, 조건별 포함률과 폭을 평가 |
| Missingness-pattern conformal | 조건부 가능 | 중간~낮음. 입력 결측 이론을 CES 표적의 MNAR 결측 보장으로 옮기지 않음 |
| 선택적 출력/abstention | 가능 | 중간. 출력하는 행의 비율이 같은 기준선과 오차·놓침을 비교 |
| 관측 propensity/IPW | 조건부 가능 | 낮음. MAR·positivity 가정 없이는 관측되지 않은 참값의 정확도를 식별하지 못함 |
| Koopman/latent linear dynamics | 조건부 가능 | 낮음~중간. 측정 신호의 유효 동역학과 물리적 수송 모드를 구분 |
| Bayesian last layer / online supervised adaptation | 조건부 가능 | 중간. 늦은 CES는 현재 입력이 아니라 **측정 시각의 저장된 입력**과 연결해 갱신 |
| IRM / DANN / group robustness | 조건부 가능 | 낮음~중간. 관측된 domain 정의가 필요; 실제 온도 수준 차이까지 지우는지 확인 |
| Bayesian online change-point / regime-aware filtering | 조건부 가능 | 중간. 과거 입력만으로 전환을 감지하고 reset 비용·거짓 전환을 평가 |
| 약한 물리 제약을 가진 system identification | 부분 가능 | 중간. 양수성·안정성 등은 가능하나 열수송 PDE를 잔차로 걸 데이터는 부족 |
| Granger/causal-discovery 분석 | 예측적 의존성 탐색만 가능 | 낮음. 공통 빔·계측 과정·누락 변수를 배제하지 못하므로 물리 인과 발견으로 해석하지 않음 |

S4/S5/Mamba는 같은 넓은 SSM 계열이고, T-LSTM/GRU-D나 여러 contrastive 방법도 관련이 있다. 위 표의 행 수로 독립적인 새 원리의 수를 주장하지 않는다. 서로 다른 이름을 모두 GPU 배치로 돌리는 계획도 아니다.

## 원천 문헌을 별도로 확인한 핵심 항목

원논문/저자 원문/공식 코드의 초록·개요 수준 확인이다. 모든 전문과 코드를 재현한 것은 아니며 다른 분야의 성능을 KSTAR 성능으로 옮기지 않는다.

| 원천 | 별도로 확인한 내용과 적용 한계 |
|---|---|
| [GRU-D](https://doi.org/10.1038/s41598-018-24271-9), [저자 preprint](https://arxiv.org/abs/1606.01865) | 결측과 시간 간격을 이용한 학습 가능한 감쇠. 여기서의 우위는 미검증 |
| [Causal TCN](https://arxiv.org/abs/1803.01271) | causal convolution 기반 sequence 비교의 근거. 다른 분야 benchmark 결과로 추론 지연을 보장하지 않음 |
| [GRU-ODE-Bayes](https://proceedings.neurips.cc/paper_files/paper/2019/hash/455cb2657aaa59e32fad80cb0b65b9dc-Abstract.html) | 연속 시간 latent 진행과 드문 관측 갱신. 현재 fast grid가 dense라는 점은 별도 고려 |
| [Neural CDE](https://proceedings.neurips.cc/paper/2020/hash/4a5876b450b45371f6cfe5047ac8cd45-Abstract.html), [온라인 확장](https://arxiv.org/abs/2106.11028) | 원 기본 보간의 비인과성 문제와 온라인 보간의 필요성. 원 구현을 그대로 nowcast에 사용하지 않음 |
| [RKN](https://proceedings.mlr.press/v97/becker19a.html) | 잠재 공간의 상태 추정과 불확실도 표현. 잠재 상태가 참 온도로 식별된다는 결과가 아님 |
| [CRU 논문](https://proceedings.mlr.press/v162/schirmer22a.html), [공식 코드](https://github.com/boschresearch/Continuous-Recurrent-Units) | 불규칙 관측을 위한 확률적 recurrent 모형. Claude의 미확인 `/boschresearch/CRU` 주소는 이 주소로 바로잡음 |
| [KalmanNet](https://arxiv.org/abs/2107.10043) | 부분적으로 알려진 동역학을 활용하는 neural-aided filtering. dynamics-free 일반 추정기로 소개하지 않음 |
| [mTAN](https://arxiv.org/abs/2101.10318), [공식 코드](https://github.com/reml-lab/mTAN) | 시간 attention으로 불규칙 샘플을 표현. 기본 보간·전체 시계열 처리와 인과 추론의 차이를 확인해야 함 |
| [S4D](https://arxiv.org/abs/2206.11893), [공식 설명](https://github.com/state-spaces/s4/blob/main/models/s4/README.md) | diagonal sequence SSM의 구조·초기화. 별도의 물리적 상태모형이라는 의미는 아님 |
| [TS2Vec](https://cdn.aaai.org/ojs/20881/20881-13-24894-1-2-20220628.pdf), [TS-TCC](https://www.ijcai.org/proceedings/2021/324) | 표현 학습의 근거. 사용자의 데이터에서 causal encoder와 평가를 새로 설계해야 함 |
| [Faithful heteroscedastic regression](https://proceedings.mlr.press/v206/stirn23a.html), [β-NLL](https://arxiv.org/abs/2203.09168) | Gaussian NLL의 평균·분산 최적화 문제가 존재. canonical faithful 알고리즘을 단순한 frozen-mean 2단계 학습으로 설명하지 않음 |
| [EnbPI 논문](https://proceedings.mlr.press/v139/xu21h.html), [공식 코드](https://github.com/hamrel-cxu/EnbPI) | 시계열 가정하의 근사적 포함률. 임의의 캠페인 변화·MNAR에서도 정확한 유한 표본 보장이 있다고 쓰지 않음 |
| [Conformal Prediction with Missing Values](https://proceedings.mlr.press/v202/zaffran23a.html) | 결측을 고려한 구간 구성. 현재의 표적 관측 과정 전체가 그 이론을 만족한다고 가정하지 않음 |
| [Noise2Noise](https://proceedings.mlr.press/v80/lehtinen18a/lehtinen18a.pdf) | noisy pairs로 학습하는 결과에는 잡음/조건부 평균 등의 전제가 있다. 시간에 따라 변하는 Ti의 이웃 프레임을 같은 참값의 독립 반복 측정으로 취급하지 않음 |

## 그대로 채택하지 않을 제안

- 주어진 스펙트럼·평형·공간 profile 없이 PINN을 붙였다는 이유로 열수송 물리를 식별했다고 하지 않는다. 단순한 양수성/안정성 제약과 검증된 물리 방정식을 구분한다.
- 현재 전체 캠페인의 10 ms snapshot 입력만으로 kHz MHD 모드나 ELM 내부 구조를 새로 복원했다고 하지 않는다. 별도의 고속 원시 자료와 독립 사건 표지가 필요하다.
- BiRNN, 양방향 imputer, noncausal attention/보간은 실시간 모델의 입력 경로에 넣지 않는다. offline teacher로 쓴다면 train 단계에 한정하고 별도 비교한다.
- '몇 배 빠르다', '10 ms를 만족한다', '자동으로 calibrated', 'attention weight가 물리적 설명이다'는 미측정 주장을 채택하지 않는다.
- 온라인 보정기는 CES 또는 L5 라벨이 **실제로 도착한 뒤**에만 갱신한다. 측정 시각/도착 시각을 바꾸어 쓰거나 test 파일 전체 통계로 정규화하지 않는다.
- generative model로 가짜 plasma/스펙트럼 자료를 만들지 않는다. 실제 자료의 마스킹·표현 학습과 합성 관측 생성을 구분한다.

## B.85 뒤의 실행 선택

지금 B.85는 '가용 조건을 섞은 단일 recurrent 모델로 충분한가'를 묻는다. 그 결과를 보고 특정 조건의 실패를 설명하는 **작은 다음 배치 하나**를 정한다. 성능을 읽고 바꾼 가설임을 밝히고, 새 가설·공통 행·학습 예산·초기화·네 설계·판정 규칙·양쪽 결과의 문장을 등록한 뒤 실행한다.

| 관찰될 수 있는 문제 | 다음에 고정해 비교할 것 |
|---|---|
| 오래된 CES에서만 나쁨 | anchor/current 정렬 encoder 또는 GRU-D 중 하나 |
| 기억 길이에 따라 나쁘거나 실행 비용이 큼 | 같은 정보 범위의 작은 causal TCN |
| F와 L5가 다른 방향으로 움직임 | 단일 head 대 공유 head, 또는 한 가지 robust loss; 동시에 여러 요소를 바꾸지 않음 |
| 평균 오차는 괜찮지만 큰 실패를 미리 알 수 없음 | 평균 추정 유지 + 불확실도 head/구간 보정, matched-coverage 대조 |
| 갱신 중단·복귀에서 편향이나 발산 | 통합 학습 상태공간 관측기, 측정 시각을 보존한 갱신 |
| 시간순에서만 성능 저하 | 상대적 anchor 표현이나 사전에 정의한 domain robustness; 새로운 물리 인과 주장으로 대체하지 않음 |

어떤 후보도 현재 KSTAR에서 SOTA로 확인된 것이 아니다. 후보를 넓게 찾는 단계와, pedestal 진단으로 실제 작동함을 보이는 단계를 구분한다.
