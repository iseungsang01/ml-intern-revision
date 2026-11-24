# GP 이론적 배경 보강

한국어 논문 HTML의 **Ⅱ. 이론적 배경 → 7. Gaussian process 회귀의 원리와 CES 이력 추정**에 반영했다. `section.json`이 8개 소절의 편집 원본이다. 기존 시간 모형 개관은 8절로 유지했다. 약 7,800자의 설명, 표시 수식 9개와 CES 전용/멀티모달 입력 비교 표를 포함한다. 새 실험 결과나 곡선을 만들지 않았다.

## 원문과 확인 범위

- Rasmussen와 Williams의 *Gaussian Processes for Machine Learning*: 저자 공개본 2·4·5장의 관련 회귀식, Matérn 커널, 주변우도 설명을 확인했다. 전체 책 재독은 아니다.
- Williams와 Rasmussen의 *Gaussian Processes for Regression*: 원 논문 2절의 정의·조건부 예측·계산량을 확인했다. 학회는 1995년, proceedings 서지는 1996년이다.
- Roberts 등의 *Gaussian processes for time-series modelling*: 저자 공개본 3·4절의 시간 커널·조건부 추론 설명을 확인했다.
- Chilenski 2015 및 Michoski 2024: 기존 로컬 원문 검토 기록을 사용했다. 공간 프로파일 적합을 인과적 CES 시간 추정의 직접 검증으로 쓰지 않는다.
- Friedman 2001: 기존 BibTeX의 원 논문 서지를 연결했다. 이번 원문 재접근은 출판사 보안 화면·저자 PDF 연결 실패로 완료하지 못했다. 잔차와 함수 공간 경사의 관계는 공식 scikit-learn 문서의 Mathematical formulation 절과 독립 대조했다. 이 범위를 `sources.json`에 표시했다.

`sources.json`의 추가 서지 네 건은 `build_reader.load_sources()`가 읽는다. 과거 전체 문헌 API 검토의 제공 범위를 소급 변경하지 않는다. 기존 검토 254개 레코드와 이번 추가 4개를 합친 현재 탐색 레코드는 258개다. 독립 연구 수나 전체 정독 수가 아니다.

## 편집·검증 배선

`section.json` → `integrate_longform.py` → `introduction_review/manuscript_sections.json` → `build_reader.py` → `research_reader/index.html`. 수식은 MathML로 표시하고 긴 식은 모바일에서 해당 식 안에서만 가로 이동한다. 인용 버튼은 서지·확인 위치를 연다. 기존 표·그림을 유지한다.

`verify_reader.py`는 표시 수식·입력 비교 표·GP 원문 인용 링크·모바일·오프라인 동작을 확인한다. `close_gp_background.py`는 인용 연결·수치 밀도·과학 소스 불변과 원문/코드 대조 내용을 `verification.json`으로 기록한다. GP 사후 평균·분산, 관측 잡음과 잠재 분산 구분, Matérn 매끄러움, 주변우도와 국소 표준화·후보 탐색·평균만 반환하는 실제 코드를 독립 확인했다. 전체 테스트 89개가 통과했다.

## Claude 요청의 실제 상태

`review_gp_background.py`로 초안·공개 문헌 위치·개별 관측 없는 코드의 공식 Fable 검토를 준비했으나 첫 token-count preflight가 실패했다. 이후 무료 최소 진단 두 건으로 HTTP 400/invalid_request_error와 잔액 부족 범주를 확인했다. **유료 messages 요청과 완료한 Claude 검토는 없다.** 과거 검토를 이번 새 본문 검토로 소급하지 않는다. `api_preflight_status.json`, `review_attempt.json`이 실제 상태다. 키·서버 오류 본문은 저장하지 않는다.

추가 API 검토는 기존 5달러 상한의 미사용 범위에서만 허용하도록 준비되어 있으나 실행되지 않았다. `--sources-only`는 API 없이 서지 파일만 재생성한다. 자동 유료 재시도는 없다.
