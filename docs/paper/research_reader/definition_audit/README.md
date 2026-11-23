# 연구 목적과 운용 정의의 타당성 감사

저자가 L5의 반복과 연구 고유 정의가 지나치게 많다고 지적하고, 최신 TCN B105 결과도 검토에 포함하라고 요청했다. 기존 규칙을 잘 구현했는지에 그치지 않고 **그 규칙과 해석을 사용해도 되는지**를 공식 Claude API에 물었다. 읽기 쉬운 한국어 요약은 [index.html](index.html)이다.

Opus가 현재 원고·PAPER_DESIGN·표적/제외 코드·관련 등록과 기존 B73 표적 비교·B105 집계 결과를 비판적으로 검토했다. 이어 Fable에 Opus를 그대로 수용하지 말고 반박하도록 요청했다. 두 응답 모두 정상 종료했다. 기존 USD5 상한의 잔액을 사용했고 원문/사용량은 `../figure_verification/api_usage.json`과 ignored `data/.thesis_figure_verification/`에 결속한다. 이번 두 요청의 추가 추산은 USD0.98384, 앞선 세 요청을 포함한 누계는 USD3.03128이다. 실제 청구서 확인액은 아니다.

## 결과의 지위

공통 권고는 현재 보고값 F를 본문 중심으로 하고 L5를 전방 창 보고 수준의 보조 결과로 정리하는 것이다. L5 정의 자체는 가능하지만 그것을 현재 온도의 독립적인 참값이나 잡음 제거로 해석할 근거가 부족하다는 지적이다. 제외 기준과 가용성 규칙도 운용 선택과 물리적 사실을 구분해야 한다. **이것은 설계 권고이며 현재 논문·코드·등록을 변경한 결과가 아니다.** 계산한 부정적 결과를 지우거나 TCN을 구제하기 위해 표적을 바꾸자는 작업이 아니다.

Fable은 Opus의 과장도 지적했다. B73 표적 변형 비교가 이미 존재한다. B105의 VAL 대조와 TEST incumbent는 같은 적합이 아니다. 집계 gate 통과는 등록된 계산 배분 결정이므로 삭제할 일이 아니며, 현재값 조건의 악화와 나란히 밝혀야 한다. stale-F의 일부 결과도 이미 B105에 있으므로 곧바로 중복 실험을 요청하면 안 된다. 어디까지 기존 자료로 답할 수 있는지 먼저 확인해야 한다.

작업자는 조언을 검증된 사실과 분리했다. B105 원본에서 네 D10/F TEST 점추정과 구간의 음수 부호를 확인했고, 등록문의 gate가 조건별 상한이 아니라 여러 rule/head의 평균 손실 비율임을 확인했다. 어떤 rule/head가 얼마를 상쇄했는지에 대한 인과적 해석은 단정하지 않았다. 민감도 변경이 단순 재채점으로 가능한지, 전처리·이력·정규화와 적합까지 다시 해야 하는지도 별도 판단이 필요하다.

## 파일과 재생성

- `04_opus_definition_validity.json`, `.txt`: 첫 비판. 빈 근거를 관측된 부재처럼 단정한 부분도 있어 Fable 교차 검토와 함께 읽는다.
- `05_fable_definition_challenge.json`, `.txt`: 반박과 최소 설계 권고.
- `evidence_manifest.json`: 제공한 코드·등록·집계의 범위와 해시. 개인/개별 관측 자료는 보내지 않았다.
- `verification.json`: 요청·응답·사용량·출처 검증, 기존 paper/protocol 불변 확인과 비용.
- `index.html`: 위 두 검토와 원본 B105의 집계만 읽어 생성한 한국어 보고서. 메인 논문 HTML과 별개다.

`ces_prediction/research/thesis_reader/audit_definitions.py`는 bounded 요청을 보존하고, `write_definition_feedback.py`는 읽기용 보고서만 생성한다. 후자는 무료 재생성이다. 완료한 유료 요청을 다른 원고로 묵시적으로 재실행하지 않는다. 새 실험·적합·GPU 작업은 실행하지 않았다. 향후 분석 제안은 TEST-informed 탐색이며 별도 사전등록 대상이다. 이전 등록 파일은 수정하지 않는다.

## 후속 용어 확인과 부록 편집

저자의 후속 지시로 본문의 표적 정의를 짧게 줄이고 세부를 접을 수 있는 부록 A로 이동했다. 기존 결과·그림과 등록 정의는 유지한다. 최초 검토 시점의 `verification.json`과 달리 이 편집은 실제 HTML에 반영됐다. 최신 상태는 `editorial_closeout.json`이다.

`06_fable_concise_L5_appendix.json` → `integrate_longform.py` → 본문·부록, `build_reader.py` → HTML로 연결된다. 호스트는 같은 파일·연속 구간/사용 가능한 보고값 조건을 명료화하고, 학습과 평가의 표적이라는 표현을 바로잡았다. `check_l5_terminology.py`는 공개 문헌의 추출 텍스트 검색·표적 설명·웹 원문 위치를 `terminology_evidence.json`에 기록하고 공식 Fable 검토 `07_fable_L5_terminology.json/txt`를 받는다. 254개 캐시 레코드(고유 PDF 해시 253개)의 기호 검색이며 전체 재독은 아니다. 정확한 L5 선례·표준 이름은 확인하지 못했다. Ross 등의 미래 구간 중앙값 기반 이진 레이블은 유사 사례일 뿐 동일한 회귀 표적은 아니다.

`write_definition_feedback.py`가 이 용어 검토도 읽기용 보고서에 넣는다. `close_definition_edit.py`는 추가 API 호출 없이 완료 응답·공통 비용 상한·원래 과학 소스 불변·HTML 검증을 기록한다. 비용은 figure_verification/api_usage.json의 같은 5달러 상한에 합산하며 청구서 확인액은 아니다.
