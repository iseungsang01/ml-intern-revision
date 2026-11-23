# 절별 연구 원고와 문헌 근거 읽기

**최신 그림·근거 검증:** [추가 API 검증 기록](figure_verification/README.md)에 새 Opus/Fable/Opus 세 검토와 원문 대조 범위를 남겼다. 현재 그림 일곱 개는 원본 plot PDF의 벡터 이미지이며 한국어 상세 캡션과 확대 링크를 갖는다. 표는 기존 crop을 유지한다. 본문 두 가지 사실·조건 한정과 그림 6의 가로 구간 정의를 교정했다. 현재 HTML 해시는 `figure_verification/verification.json`을 따른다.

**장문 재작성 적용:** 현재 HTML은 개요·작업 메모를 대체한 한국어 논문 본문이다. 서론·이론·방법·결과·고찰·결론은 약 5만 6천 자이며, 본문 인용 문헌의 실제 서지 목록을 마지막 절에 자동 생성한다. 원래 목차 계층을 유지하고 긴 절 안의 소목차 이동을 추가했다. 결과 해설 뒤에는 기존 PDF 결과 페이지를 표시하며, 방법의 실제 표·식도 원본 PDF 링크로 확인한다. [장문 작성·점검 기록](longform/README.md)과 `LONGFORM_QA.md`를 참조한다. 이 확장은 한국어 HTML에 적용했으며 영문 PDF를 장문 번역본으로 바꾼 것은 아니다.

`index.html`은 로컬에서 바로 열 수 있는 독립 HTML이다. CSS·상호작용 코드·원고와 문헌 검토 데이터가 파일 안에 들어 있으며 자동 네트워크 요청을 하지 않는다. DOI 링크를 누를 때만 출판사 원문을 연다. 로컬 PDF와 JSON 링크는 저장소의 상대 경로를 사용하므로 이 HTML만 다른 폴더로 옮기면 해당 링크는 유지되지 않는다.

## 입력과 출력

| 파일 | 역할 |
|---|---|
| `original_outline.json` | 로컬 HWP에서 추출하고 확인한 원래 목차. 개인 이름이 들어가는 원본 파일명과 절대경로는 공개 기록에서 제거 |
| `01_fable_outline.json` | 개인 정보·본문·원자료 없이 목차와 비식별 연구 계약만 제공한 Fable 구조 초안 |
| `02_opus_structure_check.json` | 구조와 완료/계획 구분에 대한 새 Opus 검토. 원고 전체의 채점 아님 |
| `outline_mapping.json` | 기존 HWP 목차와 최신 절별 구성의 연결. 상위 장을 나눠도 Dataset/수집기준/장치설계의 하위 자리를 보존 |
| `outline_mapping.md` | 같은 목차 연결을 사람이 읽는 표로 표시 |
| `api_usage.json` | 두 호출의 사용량·응답 해시·추정액. 추가 호출 없음 |
| `build_status.json` | 사용한 원고 입력과 문헌 읽기 범위별 실제 건수 |
| `index.html` | 원고 읽기, 절별 이동, 문헌 검색·필터, 인용 근거 열기 |

HWP 원본과 추출한 사적 본문은 수정하지 않고 ignored `data/.thesis_reader/`에만 보존했다. 원본에 있던 UFCES ground truth, 높은 시간 해상도 복원 등의 표현을 완료된 결과로 옮기지 않는다. 현재 보고값 nowcasting과 앞으로의 NN 비교·설계 계획을 구분한다. 원문을 이용한 결과 수치는 새로 타이핑하지 않으며 검증된 기존 PDF로 연결한다.

## 원고 콘텐츠 schema

기본 입력은 `docs/paper/introduction_review/manuscript_sections.json`이다.

```json
{
  "title": "KSTAR CES 연구 원고",
  "subtitle": "절별 작업본",
  "status_note": "문헌 검증과 편집 상태",
  "sections": [
    {
      "id": "introduction",
      "title": "Ⅰ. 서론",
      "status": "초안",
      "content_markdown": "본문과 [@Lee2026NeuralNetworkbased] 인용",
      "subsections": [
        {"id": "goals", "title": "a. 구체적인 연구 계획과 목표", "content_markdown": "본문"}
      ]
    }
  ]
}
```

상위 id는 `introduction`, `background`, `materials-methods`, `results`, `discussion`, `conclusion`, `references`를 권장한다. `subsections`는 재귀적으로 사용할 수 있다. Markdown 대신 `content_latex`도 받을 수 있고, `pdf_path`·`pdf_page`·`pdf_label`로 원문 PDF를 연결할 수 있다. `[@CorpusKey]`와 `\citep{BibKey}`는 문헌 근거 버튼으로 변환한다. 수식은 빌드 때 MathML로 변환하며 지원하지 못한 표현은 원식 텍스트로 남긴다.

문헌 데이터는 `../introduction_review/catalogue.json`, `paper_*.json`, `background_*.json`에서 읽는다. 전체 PDF 제공 검토와 페이지 그림 없는 전체 추출문 검토를 서로 다른 상태로 표시한다. 중복 보관본과 외부 전송 보류 가이드는 별도 상태다. 새 데이터가 들어오면 HTML을 다시 빌드해야 하며, 파일 자체가 실시간으로 다른 파일을 읽거나 API를 부르지는 않는다.

본문 인용과 문헌 목록은 같은 고정 `[번호]`를 쓴다. 번호는 생성 시점의 전체 목록 제목순이며 검색·필터로 바뀌지 않는다. 코퍼스 키는 버튼의 내부 연결과 접근성 설명 및 근거 창에서만 사용한다. 저자 배열은 자연스러운 구분자로 연결하며 `visible_trailing_response_note`는 근거 창의 ‘서지 확인 메모’로 펼쳐 볼 수 있다. 보관 파일 수와 고유 공개 문헌 수는 독립 연구의 수와 같지 않다.

결과 절에 `"pdf_path": "../ppcf_revision/main_ppcf.pdf", "pdf_preview_section": "Results"`를 설정하면 빌드할 때 해당 PDF의 북마크를 다시 읽고 Results부터 다음 동급 절 직전까지 130 dpi PNG로 렌더하여 HTML에 내장한다. 페이지 단위라 시작 페이지에는 앞 절의 끝부분이 포함될 수 있다. 미리보기는 접을 수 있고 인쇄할 때 펼쳐진다. 수치·표·그림은 다시 입력하지 않는다. 이 기능은 이 저장소의 기존 `ppcf_revision/main_ppcf.pdf`만 허용하며 외부 논문 PDF는 내장하지 않는다. 페이지 목록은 `build_status.json`의 `pdf_previews`에 남는다.

## 빌드와 확인

저장소 루트에서 실행한다.

```powershell
py ces_prediction/research/thesis_reader/extract_outline.py
py ces_prediction/research/thesis_reader/build_reader.py
py ces_prediction/research/thesis_reader/verify_reader.py
```

구조 검토를 재실행하는 `outline_review.py`는 이미 보존한 응답을 재사용한다. 다른 작업·원문으로 바꾸면 자동 유료 재요청하지 않고 멈춘다. 새로운 호출에는 별도 명시 범위와 상한이 필요하다. 의존성은 `olefile`, `PyMuPDF`, `Markdown`, 선택적 `latex2mathml`이며, 현재 환경에는 설치돼 있다. UI 검증은 Playwright를 사용한다.

이 가지의 최종 유료 호출은 Fable 구조 초안과 Opus 구조 검토 두 건이며, 공개 단가 기준 추정 합계는 USD 0.781884다. 합의한 별도 상한 USD 1.10 안에서 종료했으며 추가 유료 요청을 수행하지 않는다. 남은 예산은 주 문헌 검토 가지가 관리한다.

이 디렉터리는 기존 PPCF LaTeX/PDF나 동결 학습 코드를 수정하지 않는다. 구조와 읽기 화면을 제공하며, 최종 과학 문장과 인용 채택은 주 작업의 원문 검증을 따른다.
