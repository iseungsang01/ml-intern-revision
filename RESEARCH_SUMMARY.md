# KSTAR 다중 진단 데이터 기반 Multimodal 신경망을 활용한 CES 결측 구간 예측(Gap-filling) 연구

## 1. 연구 개요
- **연구 주제**: Multimodal 신경망을 활용한 KSTAR의 Pedestal Top에서의 다중 진단(BES, ECEI, Mirnov coil) 데이터 기반 CES(Charge Exchange Spectroscopy) **결측 구간 예측(Gap-filling / Nowcasting)**
- **달성 목표**: 항상 조밀하게 측정되는 다중 진단 데이터(BES, ECEI, MC)와 과거 CES 이력을 융합하여, **CES가 결측된 10ms 격자 시점**에서 CES의 주요 파라미터인 $T_i$(이온 온도) 및 $V_{rot}$(플라즈마 회전 속도)를 예측하는 딥러닝 모델 개발. 단순히 직전 CES를 재사용하는 persistence나 평균(mean) baseline보다 정확한 예측을 목표로 함.

## 2. 연구 배경 및 필요성 (문제 제기)
- **CES 결측 문제**: CES는 충분한 신호 대 잡음비(SNR) 확보를 위해 광자(photon)를 오래 수집해야 하므로, 노출 시간/신호 품질 문제로 특정 시점의 측정값이 자주 누락된다. 실제 데이터에서 동일한 10ms 격자 위에서 $T_i$(CES_TI)는 약 8%, $V_{rot}$(CES_VT)는 약 24%의 시점이 결측되며, **두 타겟은 서로 독립적으로 결측**된다.
- **빠른 진단은 항상 존재**: BES, ECEI, Mirnov coil(MC) 등 빠른 진단은 동일 격자에서 100% 조밀하게(결측 없이) 측정된다. 즉 "항상 있는 빠른 진단"으로 "자주 비는 CES"를 채워 넣는 것이 본 연구의 핵심이다.
- **기존 대안(UFCES)의 한계**: 제한된 파장 채널만 사용하며, 측정값을 이온 온도/속도로 역산(Inverse mapping)하는 과정에 **매우 강한 물리적 가정**이 필요하다.
- **Multimodal AI 도입의 강점**:
  - **결측 구간 보완**: 빠른 진단과 과거 CES 이력으로부터 CES가 비어 있는 시점의 $T_i$, $V_{rot}$를 데이터 기반으로 추정한다.
  - **물리적 가정 최소화**: 핵융합에서 가장 기본적인 물리적 가정인 '축 대칭(Axis Symmetry)' 수준만으로 구현 가능하여, 강한 역산 가정에 의존하지 않는다.

## 3. 기대 효과 및 목적
- **연속적인 CES 가용성(동기)**: CES가 누락된 시점에도 $T_i$, $V_{rot}$ 추정값을 제공해 pedestal top 물리량을 끊김 없이 활용한다.
- **데이터 기반 가상 센서**: 강한 역산(Inverse mapping) 가정 없이 빠른 진단으로부터 CES를 추정한다.
- **검증 방법**: 진짜 결측 구간은 참값이 없어 직접 검증할 수 없다. 대신 **실제로 관측된 CES 값을 가린 뒤 모델이 복원하게 하고, 그 정확도를 직전 CES 재사용(persistence)·평균(mean)과 target별로 비교**한다(per-target RMSE/MAE, persistence 대비 skill, mean 대비 R²).
- **결론**: 이 masking 검증에서 baseline을 이기면 결측 구간에서도 잘 예측할 것으로 **추정**한다. 단, 이는 확신이 아니라 masking 검증으로 뒷받침되는 추정이며, 결측이 무작위라는 보장이 없어 결측 지점 정확도를 단정하지 않는다.

## 4. 데이터 구성 및 파이프라인
### 데이터 수집 기준
- **플라즈마 상태 타겟**: H-mode ELM suppression (RMP 인가) 상태를 우선 타겟.
  - ELM suppression이 유지되는 동안 $D_\alpha$ 신호가 크게 튀는 구간을 중심으로 약 100ms 길이로 잘라 학습 데이터 구성.
- **KSTAR 샷(Shot) 번호 기준**: 장비 하드웨어 업데이트 이력(2017년 MicroTCA 업데이트, 2020년 UFCES 도입, 2023년 텅스텐 디버터 교체 등)을 고려하여 **#24000 ~ #33000** 번호대 샷을 1~3순위로 선정.

### 데이터 처리 및 모델링 파이프라인 (수정 반영)
1. **결측치 처리(No Fake Data)**: 선형 보간 등으로 '가짜 데이터(Fake Data)'를 만들지 않습니다. 진단 입력(BES, ECEI, MC)이 온전히 관측되고 **타겟($T_i, V_{rot}$) 중 적어도 하나가 관측된** 시점의 샘플을 사용하며, 각 샘플의 `target_mask`로 **관측된 타겟만 손실에 반영**합니다(결측 타겟은 참값 없이 마스킹되어 제외). 이렇게 하면 한쪽 타겟만 관측된 행도 버리지 않고 활용합니다.
2. **불규칙 시계열 대응(Irregular Time-steps)**: 결측치 제거로 인해 시간 간격이 불규칙해지므로, 연속성에 의존하는 LSTM 대신 **1D CNN** 기반의 지역 패턴 추출기를 모든 다중 진단 장치(BES, ECEI, Mirnov Coil) 특성 추출에 일괄 적용합니다.
3. **Late Fusion Multimodal Architecture**: 각 장치의 데이터를 섣불리 평균 내어 공간적 특성을 잃어버리는 것을 방지하고, 각 진단 장치별 전용 CNN 모듈을 거친 후 마지막 층에서 융합(Concatenation)하여 최종 값을 예측합니다.

## 5. 자동화된 Multi-Agent R&D 파이프라인 (제안 및 적용)
단일 모델 구현에 그치지 않고, 최고 성능의 가상 센서를 도출하기 위해 **LLM 기반의 자율 ML 연구 시스템(Autonomous AI Scientist)**을 도입합니다.
- **Evaluation Agent**: 정의된 파이프라인(`train.py`)을 실행하여 모델을 학습하고 검증 손실(Validation Loss) 및 지표를 수집합니다.
- **Briefing Agent**: 이전까지 시도했던 접근법과 모델 성능(History)을 기록하고 요약합니다. 모델 성능이 정체(Plateau)에 빠졌는지, 계속 학습 중인지 판단하여 다음 모델 방향성을 설정합니다.
- **Researcher Agent**: Briefing Agent의 리포트를 받아, 성능 향상 중일 때는 모델의 하이퍼파라미터를 미세 조정하고, **Plateau(정체)에 빠졌을 때는 Transformer, Mamba, Graph Neural Network 등 기존과 완전히 다른 새로운 모델 아키텍처**를 탐색하고 실제 코드로 구현(`model.py` 업데이트)합니다.
