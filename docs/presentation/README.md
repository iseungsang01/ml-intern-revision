# 발표자료 — KSTAR CES Nowcasting

이 폴더는 프로젝트를 이해하기 위한 산출물을 담고 있습니다.

## 산출물

| 파일 | 형식 | 용도 |
|---|---|---|
| **`KSTAR_CES_발표자료.pptx`** | PowerPoint, 54 슬라이드 (16:9) | **약 1시간 학위논문 발표용** 덱 (2026-08-27 초록체 재작성 · §8ac–§8ap 반영) |
| **`KSTAR_CES_발표자료_20분.pptx`** | PowerPoint, 24 슬라이드 (16:9) | **20분 내부 발표용** 덱 (원자핵공학과 대학원 세미나; 2026-08-27 초록체 재작성 · B.9 압축 슬라이드 신설) |
| **`KSTAR_CES_종합방어.pptx`** | PowerPoint, 22 슬라이드 (16:9) | **연구 종합 정리·방어** 덱 (2026-08-16 §8ab 빌드) — 판정표·재실험 이유·B.1/B.2 모델 선택 서사·novelty 검증·예상질문 방어표 3장·재현성·결정 3건 |
| **`KSTAR_CES_연구흐름.pptx`** | PowerPoint, 22 슬라이드 (16:9) | **논문 집필용 참조** 덱 — 슬라이드 한 장 = `main_ko.tex`의 한 절(§3.1–§11), 노트에 `\label` (2026-08-16 빌드) |
| **`KSTAR_CES_1pager.pdf`** | A4 PDF, 1 페이지 | **한 장 요약** (배포용; 2026-08-27 초록체 재작성 · B.9 반영) |
| `KSTAR_CES_1pager.png` | PNG | 1-pager 미리보기 이미지 |

> **✅ 2026-08-27 문체·범위 재작성 (승상님 지시).** 발표자료의 문체를 전부 **논문 초록 문체**(서술형 종결, 객관·비인칭,
> 배경 → 방법 → 결과 → 결론, 불릿도 완결된 서술문)로 통일하였다. 슬라이드 제목은 주장을 담은 서술문이 되었고,
> `header()`가 실제 폰트 메트릭으로 제목 크기를 한 줄에 맞춰 자동 축소한다(시그니처 불변). 20분 덱의 발표자 노트도
> 같은 문체로 다시 썼다. 범위는 2026-08-16 §8ab 이후의 기록까지 확장되었다: 1시간 덱에 **7장 "문맥·구조·비용"**(6장)이
> 신설되어 도달 범위(reach) 사다리와 warm-up 분해(§8ac–§8af), 밀집 사다리·통합 재채점(§8al–§8am, 그림
> `docs/paper/figures/fig_context_family_ladder.png`), 계열 비교(§8ag·§8ak·§8ai), 연산자 수 비용 모델(§8ah·§8aj),
> 승패 방전 분석(§8al §4·§8an), μs shot 집합 동결(§8ao)과 양자 가지 종결(§8ap)을 담고, 결론은 §9의 프레이밍(약 50 ms
> 포화·전형성, 계열은 비용으로, V_rot의 메커니즘)으로 5항으로 개정되었다. 앞에는 **초록**과 **용어 정리** 슬라이드가
> 들어갔다. 배치 슬라이드의 윈도 대조군 지연(구 18.9 ms p99)은 §8ac의 오염된 측정이었으므로 같은 세션 값(4.46 ms)과
> "순서·연산자 수만 주장" 서술로 교체하였다. 20분 덱은 `t_context`(결과 ⑨) 한 장으로 B.9를 압축하고 Q&A 대비에
> 계열·문맥 질문을 추가했다. 빌드 로그 FIT WARNING 0 (두 덱 모두). **종합방어·연구흐름 덱은 아직 2026-08-16 판**이며
> 같은 문체·범위로 재작성 예정이다(아래 각 절 참조).

> **✅ 2026-08-16 전면 재빌드 (§8ab 기준).** 논문이 확정 프로토콜(W = 2 · held-free · 파일당 500 · 두 모집단 공동 1차 ·
> 백본 `seq_v2` · 인과 GP 기준선)로 개정된 뒤, 덱 4종 + 1-pager + 그림 스크립트를 전부 새 `paper_numbers.json`
> (schema v2, `collect_paper_numbers.py`)과 `docs/paper/outline_ko_v2.tex`/`main_ko.tex` 기준으로 다시 썼다. 바뀐 것:
> (1) 주 모델은 `seq_v2`(전체격자 인과 시퀀스, `fig_architecture_seq.png`), 옛 주 모델(iter009 윈도)은 W = 2 대조군
> (`fig_architecture.png`); (2) 모든 결과 슬라이드가 두 모집단(컷/포함)을 함께 보이고 무조건부·조건부를 구분; (3) B.1 관문
> (`fig_gate_b1.png`)·캠페인 분할(`fig_campaign.png`)·사다리+폭 스윕(`fig_ladder_scaling.png`) 슬라이드 신설, progression
> (iter2→iter9)·held 포함/제외 이중 보고·anchor+Δ 31.5%·seq +0.045·6.4 ms 등 W = 4 시대 서사·수치 전부 제거;
> (4) 트랜지언트 시연은 seq_v2 held-out TEST shot #31815(`fig_transient_seq_31815.png`); (5) 종합방어 덱의 예상질문 방어표를
> §8ab 기준으로 새로 작성.
> **폐기된 그림(쓰지 말 것)**: `fig_progression.png`(삭제), `fig_seq_paired.png`, `fig_stuckfree_paired.png`,
> `fig_transient_31815/30842/…png`(W = 4 윈도 모델), `fig_window_sweep_heldkept.png`(참고용).

> 1시간 덱은 8개 파트 구성: 초록·용어·목차 → ① 배경·문제 ② 접근법(어려운 평가 bar) ③ 데이터·파이프라인
> ④ 모델 아키텍처 ⑤ 평가 방법론(통계) ⑥ 결과 ⑦ 문맥·구조·비용(B.9) ⑧ 결론·한계·향후 연구.

### 논문 집필용 덱 (`KSTAR_CES_연구흐름.pptx`)

앞의 두 덱이 **결과를 설득하는** 자료라면, 이 덱은 **논문을 쓰는 동안 옆에 두는 참조판**입니다.
슬라이드 한 장이 `docs/paper/main_ko.tex`의 한 절에 대응하고, 각 장의 노트에 그 절의 `\label`과
인용 시 주의가 적혀 있습니다.

구성: ① 표지 → ② 논문 골격 지도(절 ↔ 확정한 것 ↔ 슬라이드) → ③ 두 주장의 분리(오프라인 vs 인과)
→ ④ 데이터·문제 설정 §3 → ⑤ 유지값 감사 §3.4 → ⑥ 모델 §4 → ⑦ 평가 방법론 §5 →
⑧–⑮ 결과 8장(인과 압도 / 헤드라인 / 간극 층화 / MNAR / 캠페인+수리 / 비대칭 / window+사다리 /
peak) → ⑯ 배치 가능성 §8 → ⑰ 남은 개선 여지 §9 → ⑱ 한계·결론 §10–11.

- **2026-08-16 판.** `main_ko.tex`는 2026-08-16 이후 개정되지 않았으므로(B.9 §8ac–§8an은 논문 본문에 아직 없다)
  이 덱은 논문과 일치한다. 논문에 B.9 절을 추가할 때 이 덱도 함께 재작성한다(초록체).
- 모든 수치는 `docs/paper/main_ko.tex`(= 동결 산출물 `paper_numbers.json`)에서 그대로
  옮겼으므로 여기서 인용하면 논문 본문과 어긋날 수 없다.
- 논문 수치가 갱신되면 `collect_paper_numbers.py` → `paper_numbers.json` → 논문 순으로 고친 뒤
  이 덱의 해당 슬라이드를 맞춘다.
- 이 덱의 카드는 `fcard()`가 실제 폰트 메트릭으로 재서 자동으로 크기를 맞추고, 넘치면 빌드
  로그에 `FIT WARNING`으로 알린다(경고 0이 정상).

### 20분 덱 (`KSTAR_CES_발표자료_20분.pptx`)

1시간 덱과 **같은 수치·같은 그림·같은 디자인 시스템**을 쓰되, 20분 안에 실제로 말할 수 있는
분량으로 재구성한 별도 덱입니다. 슬라이드 순서는 ① 배경·문제 → ② 접근법 → ③ 데이터·모델 →
④ 평가 방법론 → ⑤ 결과(9장, ⑨ = B.9 압축) → ⑥ 결론 · ⑦ 한계.

- **결론 우선(message-first) 구성** — 2번 슬라이드에서 세 가지 핵심 메시지를 먼저 제시한다.
- **모든 슬라이드에 발표자 노트** — `⏱ mm:ss–mm:ss` 러닝 클록(총 19:20), 초록체 발표 원고, 질문 대비.
  마지막 슬라이드 노트에는 Q&A 대비 8문항(계열·문맥 질문 포함)이 정리되어 있다.
- 1시간 덱 대비 병합: 진단+결측+연구질문 → 1장, 데이터+계약+split+held → 1장, bootstrap+두 모집단+모델 선택 → 1장,
  MNAR+캠페인 → 1장, B.9 6장 → 1장.
- 1시간 덱 대비 제외: 초록·용어 정리, 아키텍처 상세, gap별 층화 분석, 윈도 스윕, 배치 슬라이드, 확장 프로그램(μs·양자),
  Mirnov 재가공·Tₑ~NBI 추가 검증, 섹션 divider.
- 시간이 밀릴 때 우선 줄일 슬라이드: 2번(30초로 축약) → 20번(급변 case study) → 10번(윈도 대조군).

## 핵심 메시지 (자료가 담은 결론, 2026-08-27)

- **Tᵢ는 미래를 읽는 PCHIP를 두 모집단 모두에서 4/4+4/4로 능가하였다** (컷 +0.17~+0.26 · 포함 +0.23~+0.32, shot 군집
  95% CI). 인과 GP는 8/8 셀에서 이겼고, 최강 오프라인 평활기(GP)와는 동률이다.
- **배치 주장은 두 스트레스를 생존하였다** — 결측 재가중 인과 대비 8/8(vs PCHIP 2/4·4/4), 캠페인 시간 분할 4/4+4/4
  (윈도 대조군은 2/4·0/4로 붕괴, 원인은 드리프트 BES 1.22σ vs 타깃 0.115σ).
- **약 50 ms의 연속 인과 문맥에서 skill이 포화하며, 문맥이 사는 것은 전형성이다** — 20 ms에서도 인과 GP를 이기지만
  승리 방전 비율은 0.52이고 70 ms에서 0.66으로 평평해진다(301 방전 통합, 문맥 10배당 +0.050).
- **세 계열(순환·확장 합성곱·attention)은 같은 문맥에서 0.023 이내로 동률이므로 아키텍처는 비용으로 선택한다** —
  지연은 t ≈ N_ops × 2–3 µs이며 순환 O(1)·합성곱 O(log R)·attention 상수 4.3배. 10 ms 예산은 구속 조건이 아니고
  1 ms 판정은 보류(5세션 p99 산포 21.84배).
- **V_rot는 전역 동률이며 우위는 회전이 변하는 방전에 집중된다** — 방전 단위 승률 0.48, 상위 5개 방전 제거 시 0 이하;
  승패를 예측하는 유일한 공변량은 방전 내 타깃 산포(조용 34% → 변동 55%). 구동 변수(NBI 토크)의 부재가 원인이며
  검정력 문제가 아니다.
- **상한은 추정기가 아니라 정보에 있다** — 21,498 파라미터 b3k8 = 백본(컷), 폭 34k→879k 평평, 1,808 파라미터 tcn2k도
  인과 GP 4/4. 남은 레버는 CES 피팅 품질 메타데이터 · 원본 kHz Mirnov(B.6 shot 집합 동결) · NBI 토크 채널이다.

모든 수치의 출처: 6장까지는 `docs/paper/paper_numbers.json`(동결 산출물에서 `collect_paper_numbers.py`가 자동 수집),
7장은 `THESIS_RESULTS.md` §8ac–§8ap의 표, 프레이밍은 §9.

## 재생성 방법

수치/그림을 바꾸려면 아래 순서로 실행합니다 (저장소 루트에서):

```bash
py ces_prediction/collect_paper_numbers.py            # 0) 얼린 산출물 -> docs/paper/paper_numbers.json (교차검증)
py docs/presentation/make_figures.py                   # 1) figures/*.png 8종 (paper_numbers.json 판독)
py docs/presentation/make_figure_architecture_seq.py   # 1b) seq_v2 도식 · make_figure_architecture.py = W=2 대조군 도식
py docs/presentation/make_figure_transient_seq.py      # 1c) seq_v2 트랜지언트 시연 (held-out TEST shot, B.1 s42 체크포인트)
py docs/presentation/build_pptx.py                     # 2) 1시간 덱 54장 (헬퍼는 다른 덱이 import; 7장은 docs/paper/figures/fig_context_family_ladder.png를 읽음)
py docs/presentation/build_pptx_20min.py               # 3) 20분 덱 24장 (1시간 덱 슬라이드 재사용)
py docs/presentation/build_pptx_flow.py                # 4) 연구흐름(논문 참조) 덱
py docs/presentation/build_pptx_defense.py             # 5) 종합방어 덱
py docs/presentation/build_1pager.py                   # 6) 1-pager .pdf/.png
```

레이아웃(겹침·넘침)을 PowerPoint 없이 확인하려면:

```bash
python docs/presentation/preview_pptx.py docs/presentation/KSTAR_CES_발표자료_20분.pptx
# -> docs/presentation/.preview/slide_NN.png + 넘침/이탈 경고 출력
```

의존성: `python-pptx`, `matplotlib`, `pillow` (모두 현재 환경에 설치됨). 한글 폰트는
`Malgun Gothic`을 사용합니다(Windows 기본). 다른 OS에서는 스크립트 상단의 폰트 후보 목록을 수정하세요.

## 생성 스크립트

- `make_figures.py` — 8개 그림(모두 `paper_numbers.json` 판독, `docs/paper/make_figures_en.py`의 국문 쌍둥이):
  forest(두 모집단), RMSE ladder(인과 GP 포함), ladder+scaling(b3k8·B.4), ablation(두 모집단), peak, campaign,
  missing(데이터 장부), gate_b1(B.1 16 run).
- `make_figure_architecture_seq.py` — 주 모델 seq_v2 도식(파라미터 수는 모듈에서 실측); `make_figure_architecture.py` — W=2
  윈도 대조군 도식; `make_figure_transient_seq.py` — seq_v2 held-out TEST shot 시연(#31815, #30842; 학습 shot은 거부);
  `make_figure_transient.py`(구 윈도 모델 시연)·`make_figure_seq.py`·`make_figure_stuckfree.py`는 W = 4 시대 — 덱에서 미사용;
  `make_figure_mirnov.py` — 자기상관 사실(유효).
- `make_figure_window_sweep.py` — window sweep 곡선 (`fig_window_sweep.png`). history 길이 vs
  held-out test `skill_vs_pchip`, CES_TI/VT 패널 분리, seed 4점 + 평균. 24-run 배치
  (`data/.wsweep_summary.json`)를 읽으며, 요약이 없으면 `data/.wsweep_*` run 산출물을 직접 훑습니다.
  결과 해석은 THESIS_RESULTS.md §8f.

`figures/` 중 스크립트가 만들지 않는 두 장(중간 보고 덱에서 가져옴, 재생성 불가):
`fig_raw_csv_missing.png` (원본 shot CSV 스크린샷 — CES_TI 빈칸·CES_VT held 반복·빠른 진단은
전부 채워진 상태가 한눈에 보임), `fig_ar_formula.png` (local AR/선형 외삽 수식).
- `build_pptx.py` — 1시간 발표 덱 (네이티브 도형 아키텍처 다이어그램 + 그림 임베드).
- `build_pptx_20min.py` — 20분 발표 덱(23장). `build_pptx.py`를 import해 팔레트·레이아웃 헬퍼와
  그대로 쓸 수 있는 슬라이드를 재사용하고, 나머지는 병합/압축 버전으로 대체합니다.
  **수치를 고칠 때는 `build_pptx.py`를 고치면 두 덱에 함께 반영됩니다** (재사용 슬라이드에 한해).
  2026-07-30 중간 보고에서 손으로 고친 내용(Overview 부제 → "결론", 발표 순서 스트립 삭제,
  결측 슬라이드에 원본 CSV 스크린샷 추가, 평가 방법론 PR1–PR4 슬라이드 추가, held 제거 학습
  슬라이드 제외, seq 결과를 표로 교체 등)이 스크립트에 반영돼 있습니다. 1시간 덱에서 재사용하는
  슬라이드의 문구 수정은 `drop_shape` / `drop_para` / `set_para` 헬퍼로 빌드 후 적용하므로
  `build_pptx.py`가 만드는 1시간 덱은 영향을 받지 않습니다.
- `build_pptx_flow.py` — 논문 집필용 덱(18장). `build_pptx.py`의 팔레트·레이아웃 헬퍼와
  `figures/`를 재사용하지만 슬라이드는 전부 자체 정의이고, 카드는 `preview_pptx.py`와 같은 폰트
  메트릭으로 크기를 맞추는 `fcard()`로 그립니다. 논문 본문이 바뀌면 대응 슬라이드를 함께 고치세요.
- `preview_pptx.py` — PPTX → PNG 근사 렌더러(레이아웃 QC 전용, PowerPoint/LibreOffice 불필요).
  실제 폰트·좌표로 그려서 텍스트 넘침과 도형 이탈을 잡아냅니다. 정밀 렌더러가 아니므로
  최종 확인은 PowerPoint에서 하세요.
- `build_1pager.py` — A4 한 장 요약 (PDF + PNG).
