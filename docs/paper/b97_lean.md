# B.97 — 가시 source 선택기의 Lean 증명과 실제 trace 연결

2026-11-23 완료. 사전등록 `2a97d02`, 구현 `8c42b9c`, 경로 키 정규화 부록 `a159806`이 각각 해당 실행 전에 커밋·푸시됐다. 결과 원천은 [b97_lean.json](b97_lean.json), 점검·Claude 검토·실패 보존 내역은 [b97_lean_checks.md](b97_lean_checks.md)다.

**여섯 이산 정리를 Lean 4.36.0으로 컴파일했고, 실제 TRAIN/VAL의 등록된 선택 질의 80개에서 Lean 실행·기존 Python 선택기·별도 직접 탐색이 정확히 일치했다.** 새 모델 학습, TEST 점수 계산, 물리 정리 증명은 수행하지 않았다.

| 진술 | 확인 결과 | 적용 범위 |
|---|---|---|
| S1: 범위·적격성·최신성·부재·prefix·readout 비간섭 | 여섯 정리 컴파일 및 공리 감사 통과 | 자연수 index와 Boolean 적격 조건의 일반 명제 |
| S2: 실제 trace에서 compiled Lean/Python 연결 | 80/80 일치, 독립 입력 재구성·Python 재호출·native 재실행 통과 | 고정 unit의 훈련 파일 하나·검증 파일 하나, 실제 구간의 등록 위치, 가용 규칙 여덟 |
| 기대와 판정 | 조건부 일치 예상과 부합 | 결정론적 검사여서 군집 신뢰구간·zero 제외 여부는 적용 대상이 아님 |

## 증명과 경계

[Visibility.lean](../../formal/visibility/Visibility.lean)은 `latest eligible n`을 실행 가능한 함수로 정의한다. `latest_sound`는 반환 index의 과거 범위와 적격성, `latest_none_iff`는 부재 조건, `latest_greatest`는 적격 후보 중 최신성을 증명한다. `latest_prefix`, `eligibility_noninterference`, `readout_noninterference`는 허용된 prefix·관측 유효성·관측값이 같을 때 선택과 readout이 같음을 증명한다. 현재 timestamp metadata까지 바꾸어도 선택이 같다는 정리는 아니다.

공리 출력에서 첫 정리는 `propext`, 나머지는 `propext, Quot.sound`에 의존한다. 사용자 공리나 `sorryAx`, native-decision 공리는 없다. Std만 사용하며 mathlib 의존성은 없다. 공식 배포본 SHA-256은 `5af5f955df6cfe37082d71c56c7d47fe620faa9f59e14b917b43976c08dc676b`다. 툴체인·실행 파일·입력·질의·소스 해시는 JSON에 있다.

## 실제 trace와 독립 검사

기존 준비 artifact의 manifest에서 등록된 TRAIN/VAL 파일만 고른 뒤 원래 `data85.anchors`를 적용했다. 제한된 입력을 제공하도록 loader와 K10 frame 제공자만 local process에서 교체했으며 선택 알고리즘의 소스는 변경하지 않았다. TEST segment/label을 만들지 않았다. 별도 코드가 manifest·grid·frame 정렬·규칙별 후보를 재구성하고 원래 Python 및 Lean 실행을 재확인했다.

독립 검사의 첫 입력 해시 비교는 Windows/POSIX 경로 구분자 때문에 실패했다. 파일 내용 해시는 모두 같았다. 원래 결과·검증 소스·실패 기록을 보존하고 부록1을 등록한 뒤 새 adapter가 key의 경로 표기만 정규화했다. 이후 전체 독립 검사가 통과했다. producer를 재실행하거나 표본·임계값·선택기를 바꾸지 않았다.

train/val×여덟 규칙의 16개 cell에서 선택값 존재 44개, 부재 36개였다. 훈련 trace의 D300/D650/D1000/S는 각각 다섯 질의 모두 부재였다. 따라서 모든 규칙에서 양의 선택 사례를 실자료로 확인했다고 말할 수 없다. coverage를 보고 표본을 확대하지 않았으며 일반 선택기 정리와 유한 검사의 역할을 구별한다. 다른 실제 trace에서 mismatch가 발견되면 Python 연결 주장을 다시 제한해야 한다.

## 해석

가시 source 선택의 이산 명세와 고정 훈련·검증 예에서의 유한 대응을 확인한 부록용 검증이다. **전체 preprocessing·신경망·CUDA의 형식 검증이 아니다.** eligibility를 timestamp/finite flag로 바꾸는 단계, native compiler/runtime, 실측 acquisition/arrival 지연은 별도 경계다.

새 예측 성능·pedestal 물리 메커니즘·새 수학 정리·학술지 수락 가능성의 증거로 해석하지 않는다. fast 진단의 추가 가치에 대한 과학적 실험은 별도 등록 배치의 판독을 따른다. GPU 장애 복구나 중단 학습 재개는 이 작업에 포함하지 않았다.
