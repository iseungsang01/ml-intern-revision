# B.98 검사와 해석의 범위

등록 `a2aa9df`, 구현 `7ff5d48`은 첫 실제 적합 전에 커밋·푸시했다. 네 고정 대표 unit에서 TRAIN-only 적합과 내부 VAL 판독을 완료했다. 원래 GPU 학습 코드나 기존 등록은 바꾸지 않았다. 실행 환경의 데이터 처리 pins는 명시했고, 재전처리 대신 이전 D1 artifact hash·train-only 통계·기존 F mask·원래 grid 보고값과의 일치를 검증했다.

## Claude source 검토

공식 Fable5.2에 비식별 등록/코드만 보냈다. 실제 배열·개인 정보·성능은 전송하지 않았다. 첫 호출은 max_tokens로 끝났고, 명시된 한 번의 후속 호출은 end_turn으로 완료했다. 두 응답과 사용량은 local `data/.b98/review/*_attempt*.json`에 보존했다. 입력 합계25074/output15321 tokens이며 output에는 thinking이 포함된다. 최종 답변을 컴파일/실행 증거로 사용하지 않았다.

채택한 보강: 원래 NBI CSV와 파일별 cache power를 직접 비교해 positional join을 검증; 일곱 준비 입력의 gi/fast/label/mask 일치; 현재·미래 CES 변경과 left-end ODE에서 현재·미래 forcing 변경의 prefix 불변; 별도 코드의 원래 grid/공통 행/hold/채널 mapping/군집 재구성; 등록된 seed 고정 및 등록되지 않은 평균 skill 제거. 모델·표본·규칙·τ 예산은 확대하지 않았다.

Fable의 invalid-file 지적은 schema 확인이 없는 조건부 우려였다. 실제 grid의 shot은 파일별 배열이고 B.79 invalid_files는 shot 번호이므로 원래 비교 키가 잘못됐다고 확정하지 않았다. 대신 배열 크기·ID 집합·manifest 안에서 제외된 파일 집합의 정확한 일치를 검사했다. label-only 검사와 원시 CES 현재/미래 값 검사를 구별했고 둘 다 실행했다. segment gap과 τ 하한이 보장하는 exponent 범위를 확인했다. 관측 시각 인과성이 실측 frame 도착 시각을 뜻한다는 주장은 하지 않았다.

## 실제 검사

- 기존 prepared artifact SHA·원자료 SHA, train/val/test file 분리, 정확한 train-only normalization 출처.
- 원래 joined beam cache와 실제 파일별 CSV의 power 및 time 대응. 기존 invalid-file 집합의 전부/정확한 제외.
- segment 안의 strict-past source, cutoff 및 S 규칙, source/age/history의 원래 준비 입력 대응.
- blocked float64 필터 대 별도 scalar 직접 재귀, full/prefix, current/future CES, current/future forcing, label isolation. static L의 직접 산술·미래 입력 검사도 포함.
- 별도 `input98.py`가 model98/data98 helper 없이 raw grid와 준비 artifact에서 공통 행·source·hold·target·채널 mapping·원래 shot 군집을 재구성. 고정 실제 VAL 구간의 scalar recurrence도 재실행.
- 별도 판독의 source/예측 hash, native numerical replay, raw reconstruction, 직접 MSE/MAE 대 군집 합계 비교, 같은 draw의 paired cluster bootstrap.

정확한 오차·질의 수·계수·solver·군집 수·TRAIN/VAL 군집 중복·구간은 [b98_physics.json](b98_physics.json)에 있다. 빈 bootstrap draw는 양의 denominator 검사로 배제했다. 재학습·선택 불확실성은 포함하지 않으며, 적은 군집의 구간과 다중 비교는 기술적으로만 읽는다. 네 대표 unit의 내부 VAL은 전체 네 outer 평가를 대신하지 않는다.

검사에 통과한 것은 수치 구현과 유한 raw correspondence다. 열수지 가정·계측 보정·transport parameter·현재 model의 최적성·outer TEST 성능을 증명하지 않았다. 과학적 판독은 [결과](b98_physics.md)를 따른다.
