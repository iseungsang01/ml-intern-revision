# USD30 상한의 원문과 공식 코드 심화 검토

이전 조사에서 초록·검색 수준으로 남았던 가장 가까운 선행연구를 전문과 공식 소스로 더 확인하고, 현재 연구의 실험 목표·모델 계열·사용 시나리오를 다시 검토한다. 추가 비용 상한은 사용자가 승인한 USD30이며 이전 조사 비용은 별도다. 돈을 소진하는 것이 목표가 아니다.

## 읽는 순서

- `DECISION.md`: 작업자가 소스·수학 확인과 마지막 추가 문헌을 반영한 최종 요약.
- `13_delayed_measurement_source_addendum.md`: 마지막으로 보완한 지연측정 1차 문헌과 P2 순위 재검토. `12_corrected_korean_plan.md`의 해당 부분보다 우선한다.
- `12_corrected_korean_plan.md`: 최종 연구 방향과 우선 실험 제안. `11_final_korean_decision.md`의 권고에 작업자 수학·소스 확인과 후속 API 정정을 반영한다.
- `09_substantive_research_programs.md`, `10_adversarial_program_review.md`: 실질적인 연구 프로그램 비교와 별도 반대 검토.
- `05_completed_fusion_evidence.md`부터 `08_completed_representation_evidence.md`: 완결된 좁은 쟁점의 후속 검토. 원래 해당 분야 요청이 완결된 경우에는 원래 분야 보고가 대신한다.
- `EXISTING_WORK.md`: 등록·구현·실제 pilot·외부 TEST 평가를 구분한 기존 실험 지도. 이전에 모든 후보를 실험한 것처럼 읽힌 부분을 바로잡는다.
- `PRIMARY_CHECKS.md`, `source_manifest.json`: 별도로 확인한 원문·코드·접근 한계와 원본 hash.
- `METHOD_CHECKS.md`: 비선형 차분, 과거 측정 갱신, 공정한 입력 비교와 불확실도 평가에 대한 작업자 별도 확인.
- `cost_ledger.json`: 공식 공개 단가 기준 추정 누계, 진행 중/불확실 예약액, 추가 사용 가능액.

파일 존재와 `status.json`을 함께 확인한다. 파일이 아직 없으면 진행 중인 단계이며 완료됐다고 해석하지 않는다. `_INCOMPLETE` 응답은 출력 한도에 도달한 기록이며 완결된 검토가 아니다. 이를 삭제하거나 성공으로 바꾸지 않는다. 좁힌 후속 요청은 별도 과제와 비용으로 기록한다.

## 근거와 실제 실행 범위

RTCAKENN과 COMPASS 출판사 PDF 전체, Diag2Diag 원문 body와 공식 Zenodo 코드, RKN/CRU/KalmanNet 전체 PDF와 commit이 고정된 공식 소스를 확보했다. Faithful regression, β-NLL, TS2Vec, generalized distillation 원문도 검토에 사용한다. 외부 코드와 원문은 읽기 자료이며 실행하지 않는다. KSTAR NN-CES 전문은 접근에 실패했으므로 전문 검토로 기록하지 않는다.

이 조사에서 새로운 과학 적합·GPU 실행·TEST 재채점을 수행하지 않는다. 최신 집계는 기존 독립 final-audit gate를 통과한 익명 packet만 제공한다. 원 관측값·개인정보·credential·로컬 절대경로는 연구 요청에 넣지 않는다. 실제 평가가 없던 모델을 이미 반증된 후보로 취급하지 않는다. 제안은 TEST-informed 탐색이므로 별도 사전등록과 필요한 독립 확인 후 실행한다.

## 비용 관리와 완료 기준

공식 Fable base input/output 단가와 무료 token-count endpoint를 확인했다. 호출 전 입력 추정에 여유를 더하고 최대 출력 비용까지 예약한다. 이전 불확실 요청이 있으면 예약액을 남기고 무작정 재전송하지 않는다. 이 작업의 paid 생성 요청에는 추가 토큰을 자동으로 늘리는 server-side search/fetch를 넣지 않았다. 원문·공식 소스는 host가 별도로 확보한다. response usage와 공개 단가의 곱은 계정 청구서 확정액이 아니며, 환율·세금·계정 할인은 추정하지 않는다.

완료 기준은 지출액이 아니라, 가장 가까운 연구의 정확한 검증 수준과 기존 실험 중복을 확인하고, 구체적인 estimator/scenario 가설·최소 반증 실험·대조·자원·중단 조건을 갖춘 계획을 얻는 것이다. 원래 요청이 출력 한도에 닿으면 보존하고 더 좁은 후속 과제로 마무리한다. 전체 과제의 비용 예약 합계가 USD30을 넘으면 새 호출을 시작하지 않는다.
