# Lean 적용 검토 — Claude API 조사와 독립 검산

작성일: 2026-11-23. 이 문서는 방법 조사이며 새 학습 배치나 TEST 분석이 아니다. 논문의 pedestal-top CES nowcasting 목적과 다섯 질문을 유지한다. **현재 가치가 가장 큰 용도는 가용정보·시간 인과성 계약을 정확하게 만드는 것**이다. 새로운 수학 정리, 물리 발견, Lean 증명 완료는 이번 조사에서 확립하지 않았다.

## 조사와 판단

공식 Claude API로 Fable 5.2 세 건, Opus 5.5 세 건을 수행했다. 최초 세 관점은 형식 검증, 정보의 나이와 추정 이론, 물리·제어 가치였고, 이후 반대 검토, TorchLean 연결 검토, 수학 오류 재검산을 수행했다. 실제 모델·종료 상태·사용량·응답 해시는 [호출 감사 기록](claude_lean_reviews_20261123.md)에 있다. 비식별 방법 설명만 전송했으며 원자료·저자 신원·shot ID·미공개 성능은 전송하지 않았다. 응답의 동의 여부를 사실의 증거로 삼지 않았다.

Lean은 명시한 가정에서 명제가 따라오는지를 검사한다. 가정이 실제 장치에 성립하는지는 별도 측정 문제다. 최근 시스템 식별 논문은 기존 보조정리의 ‘모든 실현’과 ‘거의 확실히’의 차이를 형식화 과정에서 드러냈다. 우리에게도 유용한 태도지만, 그 논문의 정리가 우리 모델의 성능을 보증하지 않는다. [Lean 소개](https://lean-lang.org/theorem_proving_in_lean4/Introduction/), [시스템 식별 사례](https://arxiv.org/abs/2611.18220).

| 후보 | 이 연구에서 할 수 있는 일 | 한계 및 판단 |
|---|---|---|
| 가용정보와 비간섭 | 보이지 않는 CES·미래 fast·평가 label을 바꿔도 고정된 가시 입력 prefix의 예측이 같다는 명제 | 검증 가치 높음. 알려진 소프트웨어 성질이며 물리적 독창성을 대체하지 않음 |
| TCN 인과성·stream 동치 | 고정 가중치, inference dropout off, 올바른 reset과 left padding 아래 prefix 의존성 증명 | 구조 정리와 실제 PyTorch 실행의 연결을 따로 증명해야 함 |
| age–gain 이론 | CES 과거 정보가 낡을 때 fast 정보의 추가 예측 가치를 설명하는 조건부 모형 | Gaussian/Kalman/AoI의 기존 원리. 실제 BES/ECEI에서 가정을 검증하기 전에는 설명 모형 |
| 가용 정책별 위험 | 같은 관측 label 행에서 정보 제한 정책의 쌍대 위험 차이 해석 | 물리 인과효과·진짜 결측 label의 위험과 구별. 현 등록 실험과 상당 부분 중복 |
| 진단의 물리적 식별성 | 보고값 예측과 독립 물리 진실의 차이를 명시 | 새 물리 주장은 독립 진단·보정·공간 대응 측정 필요 |
| 전체 TorchLean 이식 | 정확한 연산 그래프와 부동소수점 경계를 검증하는 장기 경로 | 이번에는 현재 TCN 전체의 이식·컴파일·인증서를 확인하지 못함. 즉시 채택 근거 부족 |

표준 회귀기도 과제와 검증에 맞게 유용하면 연구 결과가 될 수 있다. 새 구조나 Lean 사용 자체를 기여 조건으로 두지 않는다. 다만 진단 결측 대응만으로 차별성을 주장할 수는 없다. RTCAKENN이 이미 관련 문제를 다루므로, 지금 필요한 증거는 현재 과제의 동일한 평가 지지집합에서 얻는 예측 개선과 적용 조건이다. [RTCAKENN](https://doi.org/10.1088/1741-4326/ad142f).

## 먼저 증명할 만한 정확한 계약

고정된 모델 가중치·훈련 통계·설정에서 두 기록이 시각 t까지의 **허용된 관측과 prefix로 결정되는 구간 정보**에 대해 같다면, t까지 생성되는 모델 입력과 유효 예측도 같아야 한다. CES는 동일 파일·구간의 엄격히 이전 유한 관측이어야 한다. fast 입력도 선택한 시간 계약을 따라야 한다. 라벨은 입력 생성이나 예측값 계산에 쓰지 않는다. 훈련 시점의 통계·가중치가 미래 자료에 의존하는 문제는 이 inference 명제와 별도다.

현재 구현의 age cutoff는 `lim=t-D+0.003` 및 엄격히 이전 index 조건이므로 선택된 source의 age는 **D−3 ms 이상**이다. D+3 ms 이하라는 다른 리뷰의 해석은 반대 방향이었다. 파일·구간 경계, 유한값 선택, 미래를 보고 정한 segment나 통계가 없는지도 계약에 포함해야 한다. `build_segments`는 라벨 결측 때문에 입력 행을 제거하는 방식이 아니며, 이번 읽기에서 리뷰가 가정한 그러한 버그는 발견하지 않았다. 예측값의 비간섭과 라벨 유효성에 따른 저장·채점 행의 선택은 구별한다.

현재 TCN의 수용영역은 six one-convolution blocks, kernel 3, dilation 1…32에서 **127행**이다. 간격이 불규칙하므로 고정된 초 단위 길이와 같지 않다. 정확히 10 ms 격자일 때만 첫 행과 마지막 행 사이 126간격이 1.26초다. row별 LayerNorm, segment reset, inference dropout off가 전제다. padding 위치의 출력까지 0이라는 주장은 하지 않는다. 실제 backend의 batch/stream 반올림 차이는 별도 검증 대상이다.

장치 실시간성에는 적어도 다음 두 조건이 필요하다.

```
selected_input_arrival <= compute_start
compute_start + preprocessing_time + inference_time <= decision_deadline
```

CES exposure 시작 timestamp만으로 arrival을 알 수 없다. 현재 acquisition·분석·전송 지연이 측정되지 않았으므로 index 인과성 증명을 실제 가용 시각 증명이라고 부르지 않는다. 일률적인 ‘k×cadence−10 ms 이내면 충분’이라는 리뷰의 식도 채택하지 않는다.

## 수학적으로 남길 내용과 고친 오류

### 더 많은 정보와 실제 학습 성능

Y가 제곱 적분 가능하고 정보 σ-대수가 H⊆G이면 Bayes squared risk 차이는

`R*(H)−R*(G) = E[(E[Y|G]−E[Y|H])²] >= 0`.

이는 조건부 기대값의 직교사영 성질이다. mathlib에는 관련 L2 조건부 기대값 기반이 있다. 새로운 정리로 제시하지 않는다. 서로 다른 age 요약이나 다른 채점 행은 이 포함 관계를 자동으로 만족하지 않는다. 유한 학습·추정 오차·분포 변화 아래 실제 적합 모델의 위험 순서는 보장하지 않는다. [mathlib CondexpL2](https://leanprover-community.github.io/mathlib4_docs/Mathlib/MeasureTheory/Function/ConditionalExpectation/CondexpL2.html).

history-only twin을 fast-enabled 모델의 fast stem 열을 0으로 두어 나타내는 구조적 포함은 유효한 이상적 입력마다 **점별로** 성립한다. ‘거의 확실히만 성립’한다는 초기 리뷰는 잘못이었다. 이 함수 클래스 포함도 훈련된 모델의 성능 보장은 아니다. 실제 IEEE 연산에서는 NaN/Inf·반올림과 실행 경계를 따로 다뤄야 한다.

### OU에서 hold와 최적 추정자를 혼동하지 않기

평균 0 정상 OU 상태의 분산을 S, 상관을 ρ=exp(−θa), 이전 보고를 C=X_s+e_s, 목표 보고를 Y=X_t+e_t라 하자. 보고 오차는 상태와 독립, 서로 독립, 평균 0이며 분산이 r_old, r_new라는 가정이다.

```
MSE(hold C -> Y) = 2S(1-rho) + r_old + r_new
MSE(rho X_s -> Y) = S(1-rho^2) + r_new
```

둘째 식은 잡음 없는 이전 상태를 이용한 Gaussian 최적 조건부 평균의 식이며 hold의 식이 아니다. 초기 물리 가치 리뷰가 이를 혼동했고 최종 Opus 재검산이 오류를 확인했다. OU 공분산만 가정하면 둘째의 최적성은 선형 추정자 범위이며 Gaussian/OU 가정이 필요하다. 오차끼리 공분산 c가 있으면 hold 식에 −2c가 추가된다. 상태와 오차의 교차 상관도 있다면 위 식을 그대로 쓰지 않는다. OU 원격 추정과 정보 나이의 연결 자체는 기존 문헌이다. [Ornee–Sun](https://arxiv.org/abs/1902.03552).

### anchor 차분 모형에서만 얻는 조건부 단조성

별도 설명 모형으로 C=X_s+ε, Z_j=hX_j+b+η_j를 둔다. 결합 Gaussian 상태·독립 오차, Var(ε)=r≥0, Var(η_j)=q>0, h≠0, 시간 불변 offset b를 가정한다. 서로 다른 두 관측의 D=Z_t−Z_s만 추가하면 b가 소거된다. k=S/(S+r), u=1−ρ, A=h²S, B=h²S²라 할 때 상태 X_t의 Bayes MSE 감소는

```
G(u) = B u² (1+k-ku)² / [A u(2-ku)+2q]
```

이다. 비Gaussian이면 같은 공분산 계산은 최적 선형 추정의 감소를 주며 일반 Bayes 위험 감소라고 읽을 수 없다. 양의 분모와 0≤u,k≤1에서 도함수는 다음 인수분해로 비음수임을 확인할 수 있다.

```
G'(u) = 2Bu(1+k-ku) *
        [Au(1+k-3ku+k²u²)+2q(1+k-2ku)] / [Au(2-ku)+2q]²
1+k-3ku+k²u² = (1-k)² + k(1-u)[3-k(u+1)]
1+k-2ku = (1-k)+2k(1-u)
```

**정해진 모형 안에서의 단조성은 성립한다.** 반대 리뷰의 offset drift 반례는 원래 모형의 가정을 바꾸는 반례다. 반대로 실제 장치까지 단조성 결론을 확대할 수는 없다. 독립 offset drift 분산 τ²a가 분모에 추가되면 a→∞에서 이득이 0으로 가므로 일반적인 비감소성은 깨진다. 작은 age에서 O(a²)는 q>0에 의존하며 q=0의 비퇴화 경우 O(a)가 된다. a→0은 서로 다른 noisy samples가 가까워지는 극한이며, 동일한 하나의 관측을 두 번 빼는 a=0과는 구별한다.

실제 anchor 모델은 공유 비선형 encoder 출력의 **연결(concatenation)** 이다. 차분이나 offset 불변성이 강제되지 않는다. 현재 fast를 직접 쓰는 TCN에도 위 정리를 대입할 수 없다. 직접 proxy의 경우 P(a)=S(1−kρ²)에서 추가 이득 h²P²/(h²P+q)는 이전 CES 잡음 r>0이면 age→0에서도 양수일 수 있다. 동일한 현재 목표 보고를 입력으로 그대로 공개한 경우와는 다른 모형이다. BES/ECEI가 보정된 Ti 온도계라고 가정하지 않는다.

위 식의 공분산 전개·도함수·인수분해·극한은 `check_lean_candidate_algebra_20261123.py`가 SymPy의 정확한 기호식으로 확인한다. 확률모형의 타당성, 부호 논증 전체, Python 모델, Lean 증명을 확인하는 프로그램은 아니다. 합성 관측이나 실험 수치를 생성하지 않는다.

### 지정한 정보 정책과 실제 age를 구별하기

기록 W, 변하지 않는 라벨 유효성 V=v(W), 지정한 masking 정책 J, 실제 age A=g(W,J)를 구별한다. J가 W와 독립이고 양의 할당 확률·유한 위험·고정된 예측 정책을 만족하면

`E[L_J(W)|J=j,V=1] = E[L_j(W)|V=1]`.

학습 기록에 조건부로 말하려면 그 조건 아래의 독립성이 필요하다. **J를 무작위화해도 실제 A는 W와 독립이 되지 않는다.** 관측 간격과 결측 상태가 A를 결정하기 때문이다. 일정한 every-kth cadence도 물리 상태와 위상 정렬 가능성을 없애지 않는다.

모든 정책을 동일한 기록·동일한 유효 라벨 행에서 replay할 수 있다면 무작위 할당 없이도 쌍대 위험 차이를 계산할 수 있다. 이는 관측된 보고값에 대한 정보 정책의 예측 평가다. 플라즈마에 대한 인과효과나 V=0의 온도 진실을 식별하지 않는다. 재사용된 campaign에서 선택한 모델이라는 한계와 fixed-fit bootstrap이 재학습·선택 불확실성을 포함하지 않는 한계도 유지한다.

## TorchLean과 물리 주장에 대한 경계

TorchLean은 실제로 존재하며 실수 의미, IEEE binary32 의미, 그래프 lowering, 인증서 검증, native 실행과의 경계를 다룬다. 공식 문서는 convolution·normalization을 포함한 IBP 규칙도 설명한다. 따라서 ‘Lean은 부동소수점을 못 다룬다’, ‘conv/norm 지원이 없다’라고 결론 내리지 않는다. 다만 지원되는 부분 연산의 정리와 우리 TCN의 preprocessing·GELU·dilation·LayerNorm·stream buffer·native CUDA까지의 완전한 연결은 다르다. 현재 모델의 그 연결을 이번에 검증하지 못했다. [TorchLean 검증 문서](https://lean-dojo.github.io/TorchLean/blueprint/Verification-and-Certificates/Neural-Network-Verification/), [부동소수점·native 경계](https://lean-dojo.github.io/TorchLean/blueprint/Floating-Point-and-Native-Boundaries/).

보고값 R=T+e만 관측하면, 허용하는 오차모형에 추가 제약이 없는 한 T를 T+c(X), e를 e−c(X)로 바꿔 같은 R을 만들 수 있다. 보고값 예측 성공만으로 실제 bulk Ti나 pedestal 수송 메커니즘을 확정할 수 없는 식별성 문제다. 이를 뒤집을 측정은 위치 대응·보정·오차 상관을 확인한 독립 진단이다. 다른 CES chord를 무조건 독립 진실로 취급하지 않는다.

제어 활용을 주장하려면 plant/controller와 추정 오차의 관계가 필요하다. Lyapunov·small-gain 형식화가 있어도 이 빠진 요소를 만들어 주지는 않는다. 관측 label에 대한 marginal conformal coverage 역시 실제 상태의 결정론적 오차 상한이 아니다. [Lean 제어 기초 연구](https://arxiv.org/abs/2607.19727).

## 다음 실행 순서와 중단 기준

1. 현재 등록된 B.94/B.96 학습의 수치 코드·후보·TEST 선택을 유지한다. 기존 B.75와 현재 seven-condition/K10 평가가 이미 age·정보 제한 질문을 상당 부분 다룬다. 동일한 실험을 다른 이름으로 추가하지 않는다. 결과를 point estimate, 95% interval, zero 제외 여부로 네 설계 모두 보고한다.
2. 현재 자료에서 더 강한 후보는 **같은 관측 보고값 행에서 CES age/가용 조건에 따른 fast 정보의 추가 예측 가치와 그 한계**이다. 물리 법칙이나 순수 age 인과효과로 부르지 않는다. 새 OU 적합·추세 검정·mask 정책은 별도 사전등록 전에는 TEST에서 실행하지 않는다. 실제 비단조·이득 부재·설계 간 불안정성이 나타나면 그 조건부 한계를 보고한다.
3. Lean을 실제 도입한다면 먼저 가시 source 선택과 비간섭 명제 하나로 범위를 제한한다. pinned Lean/mathlib, `lake build`, `#print axioms`, 임의 사용자 공리와 `sorry` 없는 증거, 실제 훈련/검증 trace와 Python의 대응 검사를 산출물로 요구한다. 그 명제가 선택 알고리즘을 가정으로 다시 말한 것뿐이면 Python 연결을 확보하기 전 ‘pipeline 검증’이라고 부르지 않는다.
4. 전체 TCN 이식·확률론 정리 전체·제어 정리 확장은 현 논문을 위해 즉시 진행하지 않는다. 좁은 명제에서도 구현 연결을 못 만들거나 기존 유한 검사 이상의 결론을 주지 못하면 중단하고 기존 실행 검사의 범위를 명시한다.

조사 당시 상태는 Lean 설치·증명 이전이었다. **후속 저자 지시로 B.97의 여섯 선택기 정리 컴파일과 등록 실자료80질의 연결 검사를 완료했다.** 범위와 경로 key 정규화의 실패·수정은 [후속 결과](b97_lean.md)를 따른다. 새 학습 실험이나 물리 발견은 없다. GPU 운영 장애는 별도 [상태 기록](b94_status.md)을 따른다.
