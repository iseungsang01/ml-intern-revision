**원문 상태 정정(2026-11-24):** NN-CES 전문은 기존 로컬 보관본에 이미 있습니다. 현재 상태·해시·본문 확인은 [원천 기록](../../reference/nnces_source.md)을 우선하며, 특정 API 요청에서 전문을 받지 못한 사실과 프로젝트의 원문 보유 상태를 구분합니다.

# 원천 문헌의 별도 확인

아래는 작업자가 Fable 응답과 독립적으로 확인한 항목이다. Fable의 문헌표 전체를 전문 검토했다는 뜻은 아니다. 논문에서 검증한 성능과 현재 CES 자료에 대한 적용 가설을 구분한다.

| 원천 | 확인한 내용 | 현재 연구에서 주의할 점 |
|---|---|---|
| [Jalalvand et al., Nature Communications 2025](https://doi.org/10.1038/s41467-025-63492-1), [원문 보관본](https://pmc.ncbi.nlm.nih.gov/articles/PMC12475251/) | Diag2Diag와 이 논문은 동일한 연구 계보다. 다른 진단의 상관관계로 Thomson scattering의 시공간 정보를 추정하는 framework를 제시한다. | Diag2Diag와 Jalalvand를 독립적인 두 연구의 증거로 중복 집계하지 않는다. 새 sampling grid 생성과 독립적 고속 정답 검증을 구분해야 한다. |
| [Shousha et al., Nuclear Fusion 2024](https://doi.org/10.1088/1741-4326/ad142f), [저자 소속 기관 논문 기록](https://collaborate.princeton.edu/en/publications/machine-learning-based-real-time-kinetic-profile-reconstruction-i/) | 초록에서 CAKE 수준의 kinetic profiles, 탄소 불순물 온도·회전 포함, 진단 dropout 대응, DIII-D PCS 구현을 확인했다. | 단순한 가상 진단이나 CES 부재 대응 자체를 새로운 원리로 제시할 수 없다. 현재 연구와 다른 입력·정답·profile reconstruction을 동일 비교로 취급하지 않는다. |
| [Revach et al., IEEE TSP 2022](https://doi.org/10.1109/TSP.2022.3158588), [저자 원문](https://www.weizmann.ac.il/math/yonina/sites/math.yonina/files/KalmanNet_Neural_Network_Aided_Kalman_Filtering_for_Partially_Known_Dynamics_0.pdf) | Section II의 문제 정의는 state transition과 observation function이 알려졌거나 근사되고 noise statistics가 알려지지 않은 경우를 설명한다. | 현재 CES 자료에 물리적 상태·관측식이 저절로 주어지는 것은 아니다. KalmanNet이라는 이름만 바꾸어 physics baseline 또는 physical identifiability를 확보했다고 쓰지 않는다. |
| [Schirmer et al., ICML 2022](https://proceedings.mlr.press/v162/schirmer22a.html), [논문 PDF](https://proceedings.mlr.press/v162/schirmer22a/schirmer22a.pdf), [공식 코드](https://github.com/boschresearch/Continuous-Recurrent-Units) | 불규칙 시계열에 대한 latent linear SDE와 continuous-discrete Kalman update의 닫힌 형태를 사용한다. | filtering·interpolation·smoothing 설정의 정보 집합을 확인해야 한다. 논문의 interpolation 결과를 현재 시점 causal nowcasting 성능으로 옮겨 쓰지 않는다. |
| [Joung et al., RSI 2018](https://doi.org/10.1063/1.5038938), [저자 보관본](https://pure.mpg.de/rest/items/item_3009538_4/component/file_3022758/content) | Maxwell 방정식의 제약과 GP를 결합해 빠진 magnetic signals를 추정한다. | 명시적인 물리 제약을 쓸 수 있는 계측과 현재 미보정 BES/ECEI에서 단일 CES 온도를 추정하는 문제는 정보 조건이 다르다. |

## 이름과 서지정보

기존 BibTeX key `Imrek2026TemporalSuperresolution`의 실제 저자 표기는 Martin Imrisek 등이며 제목은 *Temporal super-resolution of kinetic plasma profiles at COMPASS using neural networks*, DOI [10.1088/1361-6587/ae72c6](https://doi.org/10.1088/1361-6587/ae72c6)이다. key와 저자 이름을 혼동하지 않는다. 출판사 전문 접근이 확인되기 전에는 초록 이상의 세부 검증 설계를 확인했다고 쓰지 않는다.

## 실험 제안의 판정 원칙

문헌이 공개돼 있거나 다른 데이터에서 성능을 보였다는 사실은 현재 KSTAR 자료의 효과를 입증하지 않는다. 일반적인 다중 진단 추정·감쇠 recurrent model·Kalman update·representation learning을 새 방법으로 포장하지 않는다. 먼저 어떤 과학적 설명이 현재 결과와 양립하는지 정하고, 그 설명을 구분하는 가장 작은 실험을 고른다. 기존 TEST를 본 뒤의 제안은 탐색적이며 실제 실행 전에 별도 사전등록한다.

## 초안 해석에서 추가로 주의할 점

- `05_shift_and_representation.md`의 주변분포 변화·hold 잔차 변화·채널별 계단 변화는 진단용 서술 통계다. 이런 관측 패턴만으로 계측 gain 변화와 실제 플라즈마 상태 변화가 식별되는 것은 아니다. 실제 물리 변화도 채널 주변분포를 바꿀 수 있고, CES 자체의 계측 변화도 hold 잔차를 바꿀 수 있다. gain/offset 보정 metadata·독립 기준 진단·보정된 전후 대응이 없이 물리/계측 원인을 분리했다고 쓰지 않는다.
- `03_multirate_observers.md`의 fast 증분이 앵커 나이에 단조 증가하거나 신선한 앵커에서 zero로 수렴한다는 것은 검증할 가설이지 일반 정리가 아니다. 일반 유한표본·유한예산 회귀기의 학습 오차는 정보집합 포함 관계만으로 단조 감소하지 않는다. CES 보고 잡음·표적 정의·신호 대 잡음·학습 조건에 따라 형태가 달라질 수 있다.
- 같은 시점의 센서 보고값과 합치하는 것은 물리 참값 검증과 다르다. L5, smoothing, 분산 head, conformal interval 어느 것도 독립 참값을 새로 제공하지 않는다. source timestamp 인과성과 실제 도착시각 인과성도 구분한다.
- Diag2Diag는 원문에서 burst-mode Thomson 자료와의 비교, 기존 TS와의 정량 비교, 알려진 플라즈마 현상과의 일관성 확인을 구분해 설명한다. 따라서 “고속 정답 검증이 전혀 없다”는 단정도 부정확하다. 원문 Methods 및 Validation methodology and limitations 절을 확인했으며, 그 검증 수준을 현재 CES에 그대로 물려받는 것은 아니다.
- 같은 Methods는 가장 최근의 측정 표본으로 TS에 맞춘 정렬, 전체 data-points의 학습/검증 분할, 고속 입력의 이동평균 및 미분 전처리를 설명한다. 이 설명만으로 방전 단위 또는 시간순 외부 검증, 전처리의 실시간 가용 시각이 보장된다고 단정하지 않는다. 반대로 원문 확인 없이 미래 누출이 있었다고 판정하지도 않는다. 비교할 질문은 두 연구의 정보 집합·분할·고속 검증의 차이다.

## 원문 감사 응답에 대한 작업자 정정

`09_primary_text_audit` API 응답은 출력 한도에서 잘렸으므로 완결된 검토로 취급하지 않는다. 그 응답의 다음 적용 해석은 채택하지 않는다.

- Diag2Diag는 Jalalvand 논문이다. “Jalalvand 원문을 제공하지 않았다”는 응답 문장은 명칭을 혼동한 것이다. Methods 및 선택된 논의 원문을 실제로 제공했다.
- “L5에서만 증분이면 느린 추세”, “F에서만 증분이면 공통 잡음”은 가능한 설명의 예일 뿐 식별된 원인이 아니다. 노출 평균, 표적 선택, 이력 잡음, 빠른 물리 변화 등이 같은 양상을 만들 수 있다.
- Mamba/S4·자기지도 방법이 full-rate 원자료를 필수로 요구하는 것은 아니다. 현재 10 ms 시계열에도 적용할 수 있다. 다만 기존 후보 대비 새로운 반증 가능한 가설과 자원상 이득이 확인되지 않아 우선순위를 낮출 수 있다.
- 이분산 또는 conformal/selective 예측은 관측된 CES 보고값에 대한 calibration·coverage를 평가할 수 있다. frame fit 불확도가 없다는 이유로 그 평가 자체가 불가능한 것은 아니다. 다만 물리적 참 온도의 불확도 보정을 검증하는 것은 별개다.
- snapshot과 causal window 특징의 성능 차이만으로 aliasing을 확인했다고 쓰지 않는다. 높은 sampling rate 원자료·노출 정의·분석 대역을 이용한 별도 신호 분석이 필요하다.
- 신선한 앵커가 있어도 noisy report의 조건부 nowcast에서 fast 정보의 이득이 반드시 zero가 되는 것은 아니다. 안정성·단조성·불변성은 데이터 및 추정기 가정과 함께 검증할 대상이다.

## 실험 계획 판정의 교정과 동시 작업

최종 전 메모 E1의 별도 신뢰구간 겹침은 효과 차이가 없다는 판정이 아니다. 같은 행의 paired 차이를 직접 추정해야 한다. E2의 차이 구간이 zero를 포함하는 것도 동등성의 증거가 아니며, 동등성 또는 비열등성을 주장하려면 사전등록한 margin과 구간 전체의 위치를 평가해야 한다. 정확도·정밀도가 부족하면 불확실로 보고한다. 이 교정은 `11_corrected_decision_memo.md`에 Fable이 반영하도록 별도로 요청했다.

조사 도중 별도 작업에서 `ces_prediction/experiments/PREREGISTRATION_B104.md`가 생성되어 있음을 읽기 전용으로 확인했다. 이는 같은 trajectory의 SHORT/LONG TCN 학습 예산 효과를 직접 paired 비교하는 등록이다. 등록을 읽은 것은 실행·완료·결과 검증이 아니다. 예산 민감도를 새 실험으로 중복 제안하지 않으며, 이 조사 작업은 그 소스·프로세스·등록을 수정하거나 실행하지 않는다. Fable 최종 교정 요청이 나간 뒤 확인한 사항이므로 작업자가 별도 기록한다.

`seq_data.py`에는 과거 전용 expanding 표준화가 있고, 현재 줄기의 B.84에는 절대 수준 제거·pooled availability·raw-window 대조가 있다. 그 비교와 구조적으로 다른 실험인지 확인하기 전에는 새로운 계열로 분류하지 않는다. 이전에 제거된 연구의 수치는 다시 가져오거나 인용하지 않는다.

추가 원문 위치 확인: KalmanNet 저자 보관 PDF의 p.13에는 필터별 평균 inference timing 비교가 있다. Fable 원문 감사에는 p.1–10만 제공했으므로 그 감사 범위에 없었던 것이다. 따라서 “Bishop만 inference timing을 보고했다”는 문장은 문헌 전체에 대한 결론으로 쓰지 않는다. 이것과 실제 센서 취득부터 출력까지의 end-to-end timing은 다른 양이다.
