# 장문 한국어 논문 본문 재작성

기존 HTML에는 결과·고찰·결론을 쓰는 작업 메모가 논문 본문 대신 들어 있었고 이론 설명도 평가 규칙에 치우쳐 있었다. 저자 요청으로 모든 본문 절을 장문으로 다시 썼다. 최종 입력은 `../../introduction_review/manuscript_sections.json`, 읽기 화면은 `../index.html`이다. 영문 PDF와 그 수치·표·그림은 변경하지 않았다.

## 작성과 검토 범위

- Fable에 기존 목차, 익명화된 영문 방법·결과·고찰·결론과 관련 문헌별 감사 28건을 제공했다. 개인 정보·연구 원 관측값·자격증명은 제공하지 않았다.
- 두 Fable 응답은 출력 한도에서 중단되었다. 완전히 닫힌 서론, 이론의 앞 세 소절, 방법만 복구해 사용했다. 잘린 문단은 사용하지 않았고 원시 응답·과금은 보존했다. 완료되지 않은 호출을 완결 검토로 표시하지 않는다.
- `host_theory_completion.json`의 나머지 이론과 `host_results_discussion_conclusion.json`의 독립 장문 본문을 합쳤다. 후자의 결과 주장은 기존 영문 절에 연결했다. 사용하지 않은 Fable 결과도 `02_fable_recovered_complete_sections.json`에 보존했다.
- **실제로 합친 전체 초안** `draft_longform.json`을 새 Opus 문맥에서 점검했다. `03_opus_source_and_prose_check.json`은 완결된 검토지만 원고 합격 채점이나 원문 전체 재독이 아니다.
- 작업자와 별도 에이전트가 전체 초안을 대조했다. `../LONGFORM_QA.md`는 오류와 원문 확인 범위를 기록한다. Ko2015·NN-CES·Shousha·Vanovac의 관련 원문을 직접 확인했고 Shousha 결측 Ti 그림도 확인했다. 작업자는 Snyder2011 원본 PDF 2쪽의 두 EPED 제약을 직접 확인했다. 기존 전수 문헌 감사를 전부 새로 수행한 것은 아니다.
- `review_decisions.json`은 Opus 제안 중 채택·정밀화·제외 결정을 보존한다. EPED의 올바른 설명을 감사 요약에 없다는 이유로 삭제하지 않았고, 이미 확인한 서지정보를 미확인으로 되돌리지 않았다. 새로운 성능 수치, 노출 시간과 격자 간격의 혼동, 'PCHIP 차이는 MSE에서만' 같은 잘못된 제안도 그대로 채택하지 않았다.
- `host_corrections.json`은 F/L5 정보시점, 제곱오차 단위, 차이·비의 기준값, D650/S, RTCAKENN 입력 유지와 전체 추정의 차이, Doppler 단위·관측 기하, 진단 설정의 미확인 원인 등을 교정한다. 본문 작성 지시·실행 상태 문장도 실제 연구 서술로 바꿨다.

## 재생성과 검증

```powershell
py ces_prediction/research/thesis_reader/integrate_longform.py
py ces_prediction/research/thesis_reader/build_reader.py
py ces_prediction/research/thesis_reader/verify_reader.py
```

`integration_status.json`에 절별 실제 본문 문자 수, 인용 키, 자동 생성 참고문헌 수와 적용한 교정을 남긴다. 원래 Dataset·수집기준·시나리오·시간 구간·실험 연도/장치 상태·NN 선택/개조·단계별 연구의 목차 계층을 보존했다. 결과 본문 뒤의 기존 PDF 페이지와 방법 표 링크를 유지했다. 새 실험·적합·GPU 실행은 없다.

`rewrite_longform.py`의 완료한 유료 요청을 다시 보낼 필요는 없다. 최초 USD30 허용액의 미사용 범위 안에서 상한 USD6로 시작했고, 저자가 요청한 추가 가독성 편집까지 포함해 USD8.5·다섯 시도로 제한했다. 실제 반환 usage 추산은 `api_usage.json`에 보존했다. 계정 청구서 확인액과는 다르다. 정확한 요청/응답은 ignored `data/.thesis_longform/`에 있다. 최종 비용 합산과 코드·원고 해시는 `final_verification.json`을 따른다.

## 수치 밀도와 표·그림 편집

저자가 충분한 본문 분량·표와 그림의 위치·문단당 수치 최대 3~4개를 추가로 요청하여 `review_readability.py`로 Fable 편집을 수행했다. 전체 원고를 제공한 네 번째 응답은 출력 제한에서 중단되었고, 완결된 분량 평가와 교정 아홉 건만 `04_fable_readability_recovered.json`에 복구했다. 이를 완결된 전체 검토로 세지 않는다. 그 의견과 호스트 교정을 적용한 뒤의 문단·결론·전시 목록을 제공한 **범위를 좁힌 다섯 번째 편집 검토는 정상 종료**했고 `05_fable_editorial_closeout.json`에 보존했다. 이 요청은 HTML 이미지를 본 검토나 새로운 과학 검증이 아니다.

기존 PDF의 표 열한 개와 그림 일곱 개를 직접 렌더하여 캡션·전 영역·잘림을 확인한 뒤 `exhibits.json`으로 배치한다. 특히 Results 페이지 묶음 밖으로 밀린 TCN 표도 포함했다. 표 11의 TCN 시간과 본선 GP·boosting 시간은 분리하고, 후자는 보충 PDF의 표 S13으로 연결한다. 사진 속 숫자를 다시 타이핑하지 않는다. 본문에 남은 논문 라벨은 클릭 가능한 표·그림 번호로 풀어 표시한다. 전체 영문 페이지 묶음은 중복을 줄이도록 접힌 상태로 두었다.

숫자 밀도 점검은 단위나 한글 조사가 붙은 수치도 세며, 인용 키·모델 식별자·수식·표·참고문헌은 제외한다. 아라비아 숫자를 한글로 바꾸어 회피하지 않고 논리적 문단 분리와 계산 예시의 표 이전으로 정리했다. `numeric_density_after.json`과 UI 검증이 본문 문단의 네 개 초과 여부를 검사한다. 최종 명세와 실제 배치·링크·모바일·인용 검증 결과는 `build_status.json`, `qa_status.json`, `final_verification.json`에 있다.
