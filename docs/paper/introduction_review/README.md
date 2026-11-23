# 전수 문헌 검토와 절별 원고

**후속 장문 재작성:** 저자가 짧은 개요·작성 지시문을 지적하여 한국어 HTML 전체를 실제 논문 본문으로 다시 썼다. 최신 입력에는 `active_longform`이 있으며 `thesis_reader/integrate_manuscript.py`는 이를 보존한다. 최종 장문과 교정·원문 대조 범위·별도 사용량은 `../research_reader/longform/README.md`, `integration_status.json`, `LONGFORM_QA.md`를 따른다. 아래 초기 통합 기록과 비용은 이 후속 편집 이전 시점의 기록이다.

**최종 완료:** [HTML 원고](../research_reader/index.html)와 [PPCF PDF](../ppcf_revision/main_ppcf.pdf)를 작성했다. HTML은 서론 / 이론적 배경 / 자료·연구 방법 / 결과 / 고찰 / 결론 / 참고문헌으로 나뉘며 인용 근거를 열 수 있다. HWP의 Dataset·수집기준·연도/장치 상태 분류·장치 설계·NN 선택/개조 이유 계층을 보존했다.

전체 254개 PDF 보관 파일 중 중복 2개와 공개 여부 미확인 가이드 1개를 제외한 **고유 공개 출처 파일 251개를 개별 검토**했다. 전체 PDF를 제공한 검토 41개, 전체 추출문을 제공한 검토 210개다. 독립 연구 251편을 의미하지 않으며 모든 그림·수식의 육안 검증을 뜻하지 않는다. PDF 제공 감사에도 개별 판독 한계가 있다. 초기 목록의 `inventory_only`는 스냅샷 상태다. 최종 범위·원문/감사 해시는 `all_corpus_status.json`을 따른다.

| 최종 파일 | 읽는 곳과 역할 |
|---|---|
| `paper_*.json`, `background_*.json` | 개별 Opus 출처 감사, HTML 근거 창과 최종 집계의 입력 |
| `bibliography_checks.json` | 251개 감사의 독립 DOI/제목 확인. 과학적 주장 검증과 별개 |
| `02_interim_36_source_package.json` | 출력 제한 응답에서 완전히 닫힌 필드만 복구한 중간본 |
| `final_manuscript_sections_review.json` | Fable 초안·개별 원문 감사 후 Opus 통합문. JSON 형식 교정은 별도 필드에 기록 |
| `final_introduction_en.tex`, `final_theory_en.tex` | Opus 반환문 보존본 |
| `integrated_introduction_en.tex`, `integrated_theory_en.tex`, `host_integration.json` | 정보시점·L5·온도식·검증 범위를 교정한 실제 삽입문과 기록 |
| `manuscript_sections.json` | 원래 목차 계층을 보존한 한국어 원고, HTML 생성 입력 |
| `manuscript_writing_provenance.json`, `verification.json` | 글쓰기 원문 페이지·장부와 기존 결과 보존·해시·빌드 확인 |
| `all_corpus_status.json`, `cost_summary.json` | 최종 범위와 비용의 기준 기록 |

요약 선별·핵심 PDF 검토 후 저자가 전편으로 범위를 확장했다. `introduction_all_sources.py`가 나머지 전체 원문을 검토했고 `introduction_manuscript_finalize.py`가 Opus 통합문을 받았다. 마지막 Zimmermann 원문은 글쓰기 호출에서 전체 PDF로 검토했으며 별도 장부를 표시했다. 완료한 유료 워크플로를 재실행할 필요는 없다.

무료 재생성은 `integrate_source_sections.py`, `thesis_reader/integrate_manuscript.py` 실행 → PPCF 재빌드 → `thesis_reader/build_reader.py` → `thesis_reader/verify_reader.py` → `introduction_closeout.py` 순서다. 모두 `ces_prediction/research/` 아래에 있다. 독립 서지 확인은 `introduction_bibliography_check.py`다.

현재 목표는 현재 CES **보고값**의 nowcasting이며 NN-CES는 연결 가능한 상류 스펙트럼 판독기다. 실제 결합·미래 물리 온도 예보·sub-ms 복원 실증은 주장하지 않는다. NN 비교 후 후속 구조를 설계한다는 계획과 완료 결과를 구분했다. 새 적합·GPU 실행·TEST 재평가는 없다. 기존 결과 본문은 변경하지 않았다.

추가 문헌/HTML 비용은 반환 usage와 공개 단가 기준 **USD49.869956**, 추가 상한 USD50 안에서 종료했다. 기존 원고 편집 장부 누계 USD2.283684와 앞선 심층 자문 USD19.134768은 합쳐 최초 USD30 범위 안이다. 합산 누계 USD71.288408이며 계정 청구서 확인액은 아니다. 추가 유료 호출 예정은 없다. 정확한 요청/응답은 ignored 로컬 캐시에 보존했다.

`grade3.md`, `novelty2.md`는 앞선 동결 PDF의 독립 모의 평가다. 이번 서론·구조 수정본을 다시 채점한 결과가 아니다.

## 아래는 초기 범위와 작업 경로 기록

아래의 약 서른~마흔 편 선별 범위와 미완료 상태는 작업 초기 기록이다. 최종 완료 상태와 실제 존재하는 결과 파일은 위 표와 `all_corpus_status.json`이 우선한다.

저자 요청: 전체 보관 문헌의 목록·초록을 먼저 보고 서론과 맞는 문헌을 고른 뒤, 약 서른에서 마흔 편을 전문 검토한다. Fable이 구조와 초안을 만들고 Opus가 논문별 사실·인용·표적·검증·타이밍을 확인한다. 실제 원문 검토와 목록화는 다른 상태다.

## 배선과 읽는 순서

- `corpus_inventory.json`: 전체 PDF 경로·해시·페이지 수·추출 상태·중복 후보. 목록화만 완료한 상태다.
- `catalogue.json`: 기존 로컬 INDEX의 서지정보와 PDF 목록을 연결한다. 기관 사용자 가이드는 공개 논문 여부 확인 전 외부 전송하지 않는다.
- `budget_estimate.json`: 전편 텍스트 검토를 가정한 무료 토큰 계산 결과다. 실제 지급액이나 최종 선택 범위가 아니다.
- `01_fable_screen_and_draft.json`, `01_initial_introduction.tex`: 전체 첫 페이지·초록 수준에서의 선별과 초기 초안. 전문 확인 전 주장으로 구분한다.
- `paper_*.json`: 선택 문헌마다 새 Opus 문맥에 원문 PDF 전체를 제공해 작성한 출처 감사. 실제 생성·완료된 문헌만 검토 완료로 센다.
- `02_source_corrected_introduction.tex`, `03_opus_checked_introduction.tex`: 전문 감사 뒤의 Fable 수정본과 Opus 검토본이다. 작업자 확인·LaTeX 빌드 전에는 제출용 완성본이 아니다.
- `api_usage.json`, `progress.json`, `status.json`: 추가 비용 상한, 실패·미확정 예약액, 논문별 완료와 남은 일을 기록한다.

실행 코드는 `ces_prediction/research/introduction_corpus_inventory.py`, `introduction_corpus_plan.py`, `introduction_corpus_review.py`다. 원문·전체 추출문·API 요청과 응답은 ignored `data/.introduction_corpus/`와 `data/literature/`에 보존한다. 개인 정보나 연구 원 관측값은 전송하지 않는다. 공개 논문의 저자·제목·출판 정보는 서지정보다.

최종 연구 틀은 pedestal-top 현재 보고값의 nowcasting이다. NN-CES는 원시 스펙트럼을 해석하는 상류 단계로 연결할 수 있지만, 실제 결합과 그 성능을 검증한 것은 아니다. 현재 F와 미래 유래 L5는 모두 표적이며 둘 다 현재 입력으로 사용하지 않는다. 문헌 검토는 새 연구 적합·TEST 재평가·GPU 실행이 아니고, 독립 원고 채점과도 구분한다.

## 예산과 범위 변경

처음 전편 검토 비용을 추정했으나 저자가 요약 선별 후 관련 문헌 약 서른에서 마흔 편으로 범위를 구체화했다. 추가 상한은 공개 단가 기준 USD50이며 이를 소진하는 것이 목표가 아니다. 요청 전 무료 token count, 최대 출력 예약과 반환 usage를 기록한다. 과거 다른 연구 자문 비용과 이번 장부는 별도다. 실제 계정 청구서는 확인하지 않았다.
