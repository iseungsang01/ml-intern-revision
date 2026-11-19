# 발표자료 — KSTAR CES nowcasting

지금 쓰는 것만 이 폴더 맨 위에 둔다. 이전 판은 `archive/`에 있다(2026-11-19 정리).

## 지금 쓰는 것

| 파일 | 형식 | 용도 |
|---|---|---|
| **`KSTAR_CES_포스터_초안.pptx`** | PowerPoint, 1장, 세로 45 × 60 cm (200 % 인쇄 시 90 × 120 cm) | **포스터 한국어판** (2026-11-19). 내용은 두 단계 estimator(B.64)와 3 keV 컷 조건(B.65). 결과 수치는 표에만 둔다. 빠른 채널을 뺀 신경망은 "비교군"이다. 미리보기 `KSTAR_CES_포스터_초안.png` |
| **`KSTAR_CES_poster_en.pptx`** | PowerPoint, 1장, 같은 크기 | **포스터 영어판**: 한국어판과 같은 배치이고, history-only control이라는 용어를 쓴다. 미리보기 `KSTAR_CES_poster_en.png` |
| **`KSTAR_CES_poster_PPCF.pptx`**, **`KSTAR_CES_poster_PPCF_ko.pptx`** | PowerPoint, 1장, A0 세로 84.1 × 118.9 cm (1:1 인쇄) | **PPCF 원고의 영어 학회 포스터** (2026-11-19). 제목은 원고 제목이고, 세 단에 여덟 절을 두었다. 문장은 원고를 줄여 옮겼고, 절마다 원고 줄 번호를 `poster_ppcf_sources.md`에 적었다. 그림은 원고 그림을 A0 크기로 다시 그렸다(`make_figures_poster_ppcf.py` → `figures/poster_ppcf/`). 빌더 `build_poster_ppcf.py`. 숫자 대조 108개, 불일치 0. 미리보기 `KSTAR_CES_poster_PPCF.png`. 한국어 번역판(`--lang ko`)은 같은 배치·표·수치이고, history-only control을 비교군으로 옮겼다 |
| **`KSTAR_CES_발표_PPCF.pptx`** | PowerPoint, 10장 (16:9) | **발표 덱** (2026-11-19): 표지, 배경과 문제(결측 실태 표와 원본 파일 #30909 발췌), PPCF 포스터의 나머지 일곱 절, 회전 V_rot 보충 한 장. 배경은 `졸논 정리.pptx`의 레이아웃(세 색 띠, 남색 제목 띠)을 쓴다. 제목은 절 이름, 본문은 개조식 사실·표·그림, 결과 수치는 표에만 둔다(파란 값 = 구간이 0을 제외). 해설은 발표자 노트에 있다. 빌더 `build_deck_ppcf.py [template] [out]`. 같은 내용의 claude.ai 덱도 있다 |
| `figs_20261119/` | PNG·PDF | 2026-11-19 발표 덱(claude.ai Slides 아티팩트)의 그림. 빌더 `build_figs_20261119.py`와 `build_figs_progress_20261119.py` |

**포스터 빌드**
- 그림: `make_figures_poster.py`
- 포스터: `build_poster.py --lang ko|en`. `--anon --out`을 주면 이름을 가린 사본을 만든다.
- 숫자 대조: `check_poster_numbers.py [pptx]`가 포스터의 숫자를 기록·원고와 대조하고 금지어를 찾는다.

**도구**
- `preview_pptx.py`: pptx 미리보기.

**미추적 파일:** `KSTAR_CES_의사결정.pptx`, `build_pptx_decision.py`, `check_deck_rules.py`, `audit_deck_geometry.py`, `PPT_4STEP.md`는 다른 작업이 쓰는 미추적 파일이다. 손대지 않는다.

## archive/

2026-08부터 11-12까지 만든 덱과 그 빌더·그림이다. 수치와 결론은 당시 기록(W = 4 윈도 시절, THESIS_RESULTS §0 등)에서 왔다. **지금의 근거로 쓰지 않는다.** 근거는 `PROJECT_KNOWLEDGE.md`이다.

**들어 있는 것**
- 1시간·20분 발표 덱
- 종합방어, 연구흐름, 졸논정리, 물리근거 덱
- 1-pager
- `기말발표/` v1–v4 (2026-11-12)
- `physics_report.html`, `survey_artifact.html`

당시 README는 `archive/README.md`에 그대로 있다.

옮긴 빌더의 경로는 `archive/` 안에서 맞게 남겨 두었지만, 유지하지 않는다.
