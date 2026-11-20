# 발표자료 — KSTAR CES nowcasting

2026-11-19 판의 발표 덱과 포스터다. 내용은 그날의 줄기(순환 신경망과 2단계 estimator, B.64·B.65)이고, 지금 원고(신경망 없는 estimator, 자료 정의 D1)에 맞춘 갱신은 저자가 "수정"을 말할 때 한다. 그보다 앞선 덱들의 보관함(`archive/`)은 2026-11-21에 지웠다(git 이력).

## 지금 쓰는 것

| 파일 | 형식 | 용도 |
|---|---|---|
| **`KSTAR_CES_포스터_초안.pptx`** | PowerPoint, 1장, 세로 45 × 60 cm (200 % 인쇄 시 90 × 120 cm) | **포스터 한국어판** (2026-11-19). 내용은 두 단계 estimator(B.64)와 3 keV 컷 조건(B.65). 결과 수치는 표에만 둔다. 빠른 채널을 뺀 신경망은 "비교군"이다. 미리보기 `KSTAR_CES_포스터_초안.png` |
| **`KSTAR_CES_poster_en.pptx`** | PowerPoint, 1장, 같은 크기 | **포스터 영어판**: 한국어판과 같은 배치이고, history-only control이라는 용어를 쓴다. 미리보기 `KSTAR_CES_poster_en.png` |
| **`KSTAR_CES_poster_PPCF.pptx`**, **`KSTAR_CES_poster_PPCF_ko.pptx`** | PowerPoint, 1장, A0 세로 84.1 × 118.9 cm (1:1 인쇄) | **PPCF 원고의 영어 학회 포스터** (2026-11-19). 제목은 원고 제목이고, 세 단에 여덟 절을 두었다. 문장은 원고를 줄여 옮겼고, 절마다 원고 줄 번호를 `poster_ppcf_sources.md`에 적었다. 그림은 원고 그림을 A0 크기로 다시 그렸다(`make_figures_poster_ppcf.py` → `figures/poster_ppcf/`). 빌더 `build_poster_ppcf.py`. 숫자 대조 108개, 불일치 0. 미리보기 `KSTAR_CES_poster_PPCF.png`. 한국어 번역판(`--lang ko`)은 같은 배치·표·수치이고, history-only control을 비교군으로 옮겼다 |
| `중간 보고 2_final.pptx` (저장소 밖, 바탕화면) | PowerPoint, 12장 (16:9) | **발표 덱** (2026-11-19): 표지, 배경과 문제, 2절 한 장(KFE 2024 자료 15쪽 배치도에 이 연구의 진단 표시, 받은 ABCD 위치 그림과 Mirnov 배치도에 파일 채널 표시, 자료 표), 3–8절, 보충 세 장(회전 V_rot, 빔 blip 한 건: #31923 5.00 s의 원시 BES·CES Ti·원시 ECEI와 Heo 2025 그림 2, 방전 분포 639개). 배경은 `졸논 정리.pptx`의 레이아웃. 남의 그림이 들어 있어 출력은 `data/deck_out/`(git 제외)과 바탕화면에만 둔다. 그림 준비 `make_figures_deck_ppcf.py`(입력은 `data/deck_inputs/`), 빌더 `build_deck_ppcf.py`. 직접 그린 방전 분포도는 `figures/deck_ppcf/` |
| `figs_20261119/` | PNG·PDF | 2026-11-19 발표 덱(claude.ai Slides 아티팩트)의 그림. 빌더 `build_figs_20261119.py`와 `build_figs_progress_20261119.py` |

**포스터 빌드**
- 그림: `make_figures_poster.py`
- 포스터: `build_poster.py --lang ko|en`. `--anon --out`을 주면 이름을 가린 사본을 만든다.
- 숫자 대조: `check_poster_numbers.py [pptx]`가 포스터의 숫자를 기록·원고와 대조하고 금지어를 찾는다.

**도구**
- `preview_pptx.py`: pptx 미리보기.

**미추적 파일:** `KSTAR_CES_의사결정.pptx`, `build_pptx_decision.py`, `check_deck_rules.py`, `audit_deck_geometry.py`, `PPT_4STEP.md`는 다른 작업이 쓰는 미추적 파일이다. 손대지 않는다.
