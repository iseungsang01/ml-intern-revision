# 더 단순한 물리 수식으로 접근하기

2026-11-23 방법 조사. **현재 자료로 바로 비교할 만한 후보는 에너지 완화에서 동기를 얻은 ‘강제항이 있는 1차 응답식’이다.** 실제 물리 메커니즘을 구별하는 후보는 pedestal의 이동과 계측 선폭을 분석하는 것이지만, 기하·보정·원시 스펙트럼 등이 추가로 필요하다. 어떤 후보도 이번에 실자료에 적합하거나 성능을 측정하지 않았다. 논문의 pedestal-top nowcasting 목적과 두 표적·네 검증 설계를 유지한다.

## Claude를 어떻게 사용했는가

공식 API에서 계측 forward model(Fable5.2), 에너지 수지(Fable5.2), pedestal 기하(Opus5.5)를 독립 검토하고, 새 Opus 반대 검토와 구체 수식 재검산을 추가했다. 다섯 호출 모두 첫 시도 `end_turn`으로 완료됐다. 입력107747/output29546 tokens, 서버 검색6회이며 output에는 보고된 thinking tokens가 포함된다. 모델·사용량·응답 해시와 기호 검산은 [감사 JSON](physics_formulations_20261123.json)에 있다. 키·저자 신원·실측 배열·shot ID·미공개 성능은 보내지 않았다. 원문과 내부 reasoning은 ignored local cache에만 둔다.

Claude 검색은 주로 snippet 수준이었다. 아래 채택 판단은 그 답변을 그대로 옮긴 것이 아니라 원천 문헌·현재 자료 계약·별도 기호 검산을 함께 확인한 결과다. 일부 제안식의 오류와 과장은 뒤에서 명시적으로 정정한다. 새 경험적 배치나 GPU 작업은 실행하지 않았다.

## 후보를 세 개로 줄인 판단

| 후보 | 핵심 수식 | 현재 가능성 | 얻을 수 있는 것 |
|---|---|---|---|
| 1. 강제 1차 완화 | τ dU/dt + U = c+β_E E+β_B B+β_P P | 현재 scalar 입력으로 시험 설계 가능 | 소수 계수의 안정된 응답 구조가 보고값 예측에 충분한지 |
| 2. pedestal 이동 대 열역학 변화 | δT_E = ΔT_L − ξ·∇T₀ | 단순 통계 모형은 가능하나 물리적 분리는 추가 진단 필요 | 고정 chord 변화가 위치 이동과 일관되는지 |
| 3. CES 선폭의 혼합 분산 | σ²_v = ⟨T_C[J]/m_C⟩_w + Var_w(v_LOS)+σ²_inst | 항등식은 명확하나 기존 온도·회전 보고값만으로 항 분리 불가 | 보고 온도의 변화에 유동·노출·공간 혼합이 얼마나 기여하는지 |

## 1. 권장하는 최소 모형: 완화식 하나와 fast 입력

입자 source가 없는 species 내부에너지식을 T[J]로 쓰면 다음 구조다.

```
(3/2)n(∂t+v·∇)T + nT ∇·v = −∇·q − Π:∇v + Q
```

한 지점의 T만으로 우변을 분해할 수는 없다. 밀도와 운전점 변화가 작고, 수송·교환·이류·압축을 국소적으로 선형화할 수 있으며 안정된 하나의 느린 mode가 지배한다는 **추가 가정** 아래 저차 완화식이 동기를 갖는다. 탄소 species의 에너지식을 쓰는 데 T_C=T_D를 요구하지 않는다. 주이온 온도로 해석할 때만 별도 검증이 필요하다.

제안하는 보고값 영역의 근사는

\[
\tau\dot U+U=m(t),\qquad
m(t)=c+\beta_E E(t)+\beta_B B(t)+\beta_P P(t).
\]

- U,c는 eV, τ는 초다. E,B는 훈련 통계로 표준화한 기존 ECEI/BES 그룹 평균이며 무차원이다. β_E,β_B는 eV, P가 주입 MW이면 β_P는 eV/MW다.
- 학습 계수는 **τ,c,β_E,β_B,β_P의 다섯 개**다. E를 전자온도, B를 밀도, P를 국소 흡수열로 취급하지 않는다. 부호는 강제하지 않는다.
- τ>0은 같은 강제항을 받는 초기 오차의 감소를 보장한다. 이는 에너지 보존 증명이나 예측값 양수 보장과 다르다. 독립 기호식 검산으로 이 구조를 확인했다.
- τ는 유효 기억시간 후보다. 밀도·기울기·계측 응답·국소 열원 없이 충돌시간·열확산계수로 해석할 수 없다. τ가 입력 변화나 관측 간격에 비해 지나치게 크거나 작으면 계수와 교환돼 식별이 약해진다.

이런 물리 동기의 저차·선형 응답 접근은 기존 tokamak 제어 연구에도 있다. 따라서 새 식 자체를 기여로 주장하지 않는다. 우리 과제의 동일 조건에서 그 제한된 구조가 얼마나 유용한지가 검증 질문이다. [Moreau 등의 모델 기반 연구](https://fusion.gat.com/pubs-ext/IAEA12/A27429.pdf).

### anchor에서 시작하는 닫힌 식

허용된 과거 CES 보고값 y_s를 초기값으로 놓으면

\[
\hat U(t)=e^{-(t-s)/\tau}y_s+
\int_s^t e^{-(t-u)/\tau}m(u)\,du/\tau.
\]

이는 **잡음·노출 평균을 가진 y_s를 순간 상태처럼 초기화하는 결정론적 예측 근사**다. 실제 Ti의 최적 조건부 평균이라는 주장은 하지 않는다. τ→∞이고 나머지 계수가 고정이면 hold로 수렴한다. fast/P 계수를 0으로 두어도 c로 완화하는 이력 모형이 남으므로 hold와 별도 대조다.

구간마다 왼쪽 끝 입력을 유지한다고 명시하면 불규칙 간격에서도

```
a_j = exp(−(t_j−t_{j−1})/τ)
U_j = a_j U_{j−1} + (1−a_j)m_{j−1}
```

이다. 같은 동역학의 보조 필터 z를 한 번 계산해 두면

`Uhat(t)=z(t)+exp(−(t−s)/τ)[y_s−z(s)]`

로 anchor별 재적분을 피할 수 있다. z의 임의 초기 오차는 상쇄된다. 실제 구현은 같은 구간 내 z(s)를 저장하고 기존 gap reset·엄격히 과거인 source 규칙을 따라야 한다. 연산 수는 작지만 실제10ms 실행시간과 자료 도착시간은 별도 측정해야 한다.

**현재 fast 표본의 역할을 숨기지 않는다.** 왼쪽 유지식에는 t의 표본이 직전 구간 강제항으로 들어가지 않는다. 현재 표본을 그 구간 전체에 대입하는 것은 다른 보간/구적 가정이며, timestamp 기준 정보 누출 여부와 연속시간 모형의 정확성을 구별한다. 현재 fast 정보를 쓰는 중첩 대조가 필요하면 `γ_E[E(t)−E(s)]+γ_B[B(t)−B(s)]` 두 항을 별도로 추가한 일곱 계수 모형을 명시할 수 있다. 이는 현상학적 direct-feedthrough 항이며 열수지에서 자동 유도되는 항이 아니다. 이번에는 어느 모형도 실행하지 않았다.

### 최소 비교와 중단 기준

추후 사전등록 후보는 기본 완화식, fast/P를 뺀 대조, 같은 그룹 입력의 정적 선형 대조로 제한한다. 현재 fast 직접항을 시험한다면 두 항을 미리 고정한다. 기존 GP·hold·같은 공통 행의 회귀 결과와 함께 비교한다. 입력·이력 길이·훈련 파일 차이는 명시한다. τ 하나의 bounded 연속 적합과 조건부 선형 계수 추정으로 충분한지 먼저 설계하고 큰 architecture 탐색은 추가하지 않는다.

유용성 판단은 네 검증 설계에서 같은 행의 점추정·shot-cluster95% 구간·zero 제외 여부로 한다. 이점이 없거나, τ가 경계로 몰리거나, 설계별 계수가 불안정하면 저차 가정의 한계로 보고한다. 단순한 예측 이득 부재가 물리법칙의 반증은 아니다. 더 나은 독립 자료·보정·식별 가능한 입력 변화가 이 부정적 판단을 바꿀 측정이다. 복잡한 회귀기와의 차이도 순수 ‘비선형성의 크기’라고 해석하지 않는다.

## 2. 물리적 의미가 더 직접적인 후보: 위치 이동

평형 profile T₀의 작은 변위를 ξ라 하면 같은 위치의 Eulerian 변화는 `δT_E=ΔT_L−ξ·∇T₀`다. ΔT_L에는 비가역 가열뿐 아니라 가역 압축도 들어간다. pedestal처럼 기울기가 큰 곳에서는 위치 변화만으로 고정 chord 값이 변할 수 있다는 설명을 준다. 다만 현재 자료에는 기울기·정확한 채널 위치·평형이 없어 ξ와 열역학 변화를 분리할 수 없다.

ECEI에서 pedestal 변위와 복사전달을 함께 해석하는 실제 선행 연구가 있다. 그것은 보정된 synthetic diagnostic과 기하를 사용한 해석이며 우리 전압의 PCA mode를 곧바로 미터 단위 변위로 바꿔주지 않는다. [Yu 등의 DIII-D 연구](https://collaborate.princeton.edu/en/publications/ecei-characterization-of-pedestal-fluctuations-in-quiescent-h-mod/).

더 간단한 열역학식은 이상기체의 가역 단열 parcel에서 `T n^(1−γ)=constant`, 작은 변화에서 `ΔT_L/T=(γ−1)Δn_L/n`이다. 하지만 열유속·가열·교환이 무시 가능해야 하고 Lagrangian 변화에 관한 식이다. 고정 chord의 Eulerian 전압 변화를 그대로 넣을 수 없으며, BES는 보정된 n이 아니다. 현재 자료에서 γ를 적합해 단열 지수를 측정했다고 하거나 γ−1을 강제한 loss를 넣는 것은 근거가 부족하다. 이 관계를 실제로 시험하려면 보정된 밀도·온도와 profile 이동을 구별할 자료가 우선이다.

`ΔV_k=g_k ξ+noise`의 rank-1 제약은 값싼 통계 대조가 될 수 있다. 그러나 g_k도 학습 계수이며 PCA를 쓰면 전체 계수가 사라지는 것이 아니다. beam·gain·공통 pickup·ECE 광학효과도 같은 구조를 만든다. 분리된 위치의 coherent mode는 공간 위상과 진단 전달함수 때문에 항상0/π 위상차를 갖지 않는다. 원시 high-rate 부분집합으로 coherent-mode 가설을 제한적으로 검사할 수는 있지만, 물리적 변위 판정에는 채널 좌표·profile 기울기·광학 깊이/평형 대응을 추가해야 한다. 현재 최우선 학습 후보로 확대하지 않는다.

## 3. 새 물리 측정에 가까운 후보: 선폭 예산

노출 시간과 시선 방향에 대한 양의 intensity 가중치 w 아래, 단순 Doppler 모형의 전체 분산 법칙은

\[
\sigma_{v,obs}^2=\langle T_C[\mathrm J]/m_C\rangle_w+
\operatorname{Var}_w(v_{LOS})+\sigma_{v,inst}^2.
\]

독립 instrument convolution을 가정하고 fine structure·Zeeman·단면적 의존성 등은 별도로 다뤄야 한다. instrument 보정 후 온도로 환산한 2차 moment에는 `⟨T_C[eV]⟩_w+(m_C/e)Var_w(v_LOS)`가 들어간다. 혼합이 비Gaussian이면 단일 Gaussian-fit 선폭은 이 2차 moment와 일반적으로 같지 않다. 기존 CES 회전 보고값 하나는 intra-frame 속도분산을 알려주지 않으므로 온도에서 임의로 빼면 안 된다.

이 방향은 ‘예측 정확도’보다 ‘보고값을 무엇이 바꾸는가’를 직접 묻는 물리 연구로 발전할 여지가 있다. 필요한 추가 자료는 원시 CES 스펙트럼·기기함수·시선 geometry·노출/밝기 가중치·fit uncertainty다. 현재 보유한 raw fast stream이 이 CES 자료까지 포함한다고 가정하지 않는다. 과거 회전 증분은 나중에 등록된 noise-covariate 대조로 쓸 수 있지만 유동 broadening의 측정이라고 부르지 않는다. [CXRS 계측 선행 논문 서지·초록](https://research.tue.nl/files/151708008/1.5132936.pdf)은 확인했으나 이 식을 해당 논문의 검증 결과로 귀속하지 않는다.

## 계측식에서 바로 물리량을 역산하지 않은 이유

균일 방사층의 단순 ECE 모형은 `I=I_bg exp(−τopt)+Bν(Te)[1−exp(−τopt)]`, 전압은 추가 gain·offset을 포함한다. optically thick·thermal·localized·calibrated라는 조건을 확인해야 Te와 연결된다. ECE 전압을 무조건 Te로 두면 밀도·광학 깊이·비국소 복사를 온도 변화로 오인할 수 있다. [복사전달을 사용한 CHD 연구](https://www.jstage.jst.go.jp/article/pfr/21/0/21_1402023/_html/-char/en).

BES는 단순히 밀도 자체가 아니라 beam density·여기율·감쇠·광학계의 함수다. beam excitation/absorption과 시선 적분을 포함하는 forward model이 이미 연구돼 있다. [Field 등의 BES 연구](https://scientific-publications.ukaea.uk/papers/beam-emission-spectroscopy-for-density-turbulence-measurements-on-the-mast-spherical-tokamak/), [Marchuk 등의 atomic-data 연구](https://www.nist.gov/publications/atomic-data-beam-stimulated-plasma-spectroscopy-fusion-plasmas).

단일 energy 성분·고정 단면/geometry라면 beam density에 `P/Vacc^(3/2)` scaling을 유도할 수 있다. 하지만 실제 다중 source/energy 성분과 offset을 모르면 총 주입 power로 나누는 보정이 되지 않는다. anchor 차분은 일정 additive offset을 제거하고, 양의 offset-corrected signal의 log-ratio는 일정 gain을 제거한다. 시간 변화하는 beam attenuation은 남는다. 이런 log 보정은 기존 log-ratio·NBI 특징의 계수 제약과 중복되므로 source mapping 감사 전에 새 실험으로 늘리지 않는다. beam-off 표본과 유효 과거 CES가 공존하는지 여부도 이번에 세지 않았다.

## Claude 답변에서 수정한 핵심 오류

- noisy OU에서 `mu+rho(y_s−mu)`를 최적 평균으로 쓴 식은 anchor 잡음 shrinkage가 빠졌다. point-observation Gaussian 조건에서는 `mu+rho*S/(S+r)*(y_s−mu)`다. τ의 semigroup 검정도 관측 잡음과 노출 평균을 분리해야 한다.
- full-history OU Kalman과 GP의 대응을 ‘항상 anchor 하나만 읽는 GP’라고 줄이지 않는다. 실제 기존 GP는 Matérn-3/2이며, OU 지수 커널과 동일하지 않다.
- L5는 미래 유효 보고값의 중앙값이다. 감쇠계수 rho를 평균해 L5의 조건부 평균을 얻는다는 반대 검토의 제안은 채택하지 않는다. F의 finite exposure도 실제 미래 강제항 없이 정확히 적분할 수 없다. 두 표적의 보고 체계를 유지하고 첫 물리 가설은 F에서 명확히 정의한다.
- τ를 현재 자료로 항상 식별할 수 있다는 주장을 채택하지 않는다. source 입력·보고 잡음·aliasing·노출을 분리할 조건이 필요하다. 시간 간격10ms는100Hz 표본/명목 Nyquist50Hz이며 ‘50Hz 격자’라는 답변 표현도 바로잡는다. anti-alias가 없는 snapshot으로 원래 난류 스펙트럼을 복원할 수 없다.
- T_C=T_D를 확인하려면 독립 주이온 온도가 필요하며 평형 파일만으로는 충분하지 않다. ECEI/BES의 통계 mode나 예측 성공도 이 등식을 보장하지 않는다.

## 검산과 현재 도달한 결론

`check_physics_formulations_20261123.py`가 ODE forcing 적분·보조필터 초기값 상쇄·동일 forcing의 오차 감소·OU 노출 평균의 공분산/분산·혼합 선폭 분산 등10개 기호 항등식을 확인했다. 관측을 생성하거나 읽지 않는 SymPy 검산이며 Lean 증명이나 물리 가정의 검증은 아니다.

**현재 자료로 가장 가볍게 시작할 방법은 다섯 계수 완화 모형이며, 물리 발견에 더 가까운 방향은 독립 기하/보정을 확보한 변위 분리 또는 원시 CES 선폭 예산이다.** 이 셋을 같은 수준의 ‘이미 가능한 물리 분석’으로 취급하지 않는다. 새 모델 학습·원시자료 event 분석·TEST 판독은 후속 사전등록 전에는 실행하지 않았다.
