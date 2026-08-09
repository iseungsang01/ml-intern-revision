# 발표자료 — KSTAR CES Nowcasting

이 폴더는 프로젝트를 이해하기 위한 산출물을 담고 있습니다.

## 산출물

| 파일 | 형식 | 용도 |
|---|---|---|
| **`KSTAR_CES_종합방어.pptx`** | PowerPoint, 20 슬라이드 (16:9) | **연구 종합 정리·방어** 덱 (2026-08-05 전면 감사; `docs/연구방어_종합문서.md`의 발표판) — novelty 검증·예상질문 방어표 포함 |
| **`KSTAR_CES_발표자료.pptx`** | PowerPoint, 40 슬라이드 (16:9) | **약 1시간 학위논문 발표용** 덱 |
| **`KSTAR_CES_발표자료_20분.pptx`** | PowerPoint, 23 슬라이드 (16:9) | **20분 내부 발표용** 덱 (원자핵공학과 대학원 세미나) |
| **`KSTAR_CES_연구흐름.pptx`** | PowerPoint, 15 슬라이드 (16:9) | **연구 진행 흐름** 덱 — 지도교수 보고 / 랩 미팅 / 본인 정리용 |
| **`KSTAR_CES_1pager.pdf`** | A4 PDF, 1 페이지 | **한 장 요약** (배포용) |
| `KSTAR_CES_1pager.png` | PNG | 1-pager 미리보기 이미지 |

> 1시간 덱은 7개 파트 구성: ① 배경·문제 ② 접근법(어려운 평가 bar) ③ 데이터·파이프라인
> ④ 모델 아키텍처 ⑤ 평가 방법론(통계) ⑥ 결과 ⑦ 결론·한계·향후 연구.

> **✅ 2026-08-09 재정합.** 저장소 정비(배선 통일 + 논문에 안 쓰이는 과정 제거)에 맞춰
> 전 덱을 다시 빌드했다. 바뀐 것: (1) 20분 덱의 "진행 중인 후속 연구 — 연속시간 모델 4종"
> 슬라이드 삭제(실험은 끝났고 §8e에서 기각됐으며 코드도 제거 — 24→23장), (2) seq 슬라이드를
> §8t `seq_v2` 결과로 갱신(V_rot 유의 열세 4/4 → 0/4), (3) "현재 model.py는 후속 재작성본"
> 경고 삭제 — `model.py`가 `model_iter009.py`를 재수출하므로 더는 함정이 아니다,
> (4) `litreview/NOVELTY.md` → `docs/paper/NOVELTY.md` 경로 갱신, (5) 흐름 덱에서도 연속시간
> 인코더를 전부 제거 — 현황 카드 항목, 2026-07-30 표의 한 행("세 실험"→"두 실험"), 기각 경로
> 표의 한 행(6종→5종), 교훈 문구. 판정 기록은 `THESIS_RESULTS.md` §8e에만 남긴다.
>
> **✅ 2026-08-05 전면 재빌드 완료.** 4개 덱 전부 논문과 같은 genuine-only headline
> (+0.18~+0.28)로 통일했고, 스트레스 테스트 2종(§8i 재가중·§8n 캠페인 분할)·§8g 간극
> 통합 분석·복잡도 사다리·conformal·latency를 반영했다(1시간 덱에 결과 ⑨⑩ 신설, 흐름
> 덱의 현황/다음 작업 갱신, 1-pager 수치 교체). `make_figures.py`는 이제
> `docs/paper/paper_numbers.json`을 읽으므로 그림 수치가 논문과 어긋날 수 없다.
> §8c에서 뒤집힌 "held는 학습을 오염시키지 않는다" 서술도 전 덱에서 교정됨.

### 연구 흐름 덱 (`KSTAR_CES_연구흐름.pptx`)

앞의 두 덱이 **결과를 설득하는** 자료라면, 이 덱은 **연구가 어떤 경로로 여기까지 왔는지**를
보여줍니다. 음성 결과와 재현성 함정이 각주가 아니라 본문입니다.

구성: ① 현황 한 장 → ② 타임라인(6개 분기점) → ③ 연구 질문의 세 번의 전환 →
④ 확정된 것 3장(헤드라인 / Tᵢ↔V_rot 비대칭 / peak) → ⑤ 평가가 단단해진 과정 →
⑥ 데이터를 정직하게 만든 세 번의 수정 → ⑦ 2026-07-30 3연발 실험 → ⑧ window sweep →
⑨ 기각된 경로 5종 → ⑩ 재현성 함정 4종 → ⑪ 다음 작업(우선순위·판정 기준) → ⑫ 한 장 요약.

- "지금 어디에 있나"(2번)와 "다음 작업"(14번) 두 장만으로 현재 상태가 전달되도록 설계.
- 모든 수치는 `THESIS_RESULTS.md`(2026-07-14 재생성 체크포인트 패밀리) 또는
  `PROJECT_KNOWLEDGE.md`에서 가져왔고, 각 슬라이드 노트에 출처 섹션이 적혀 있습니다.
- 결과가 갱신되면 이 덱의 2 / 10 / 11 / 14번을 함께 고쳐야 합니다.

### 20분 덱 (`KSTAR_CES_발표자료_20분.pptx`)

1시간 덱과 **같은 수치·같은 그림·같은 디자인 시스템**을 쓰되, 20분 안에 실제로 말할 수 있는
분량으로 재구성한 별도 덱입니다. 슬라이드 순서는 ① 배경·문제 → ② 접근법 → ③ 데이터·모델 →
④ 평가 방법론 → ⑤ 결과(6장) → ⑥ 결론·한계.

- **결론 우선(message-first) 구성** — 2번 슬라이드에서 세 가지 핵심 메시지를 먼저 제시.
- **모든 슬라이드에 발표자 노트** — `⏱ mm:ss–mm:ss` 러닝 클록, 말할 대사, 예상 질문 대비.
  마지막 슬라이드 노트에는 Q&A 대비 6문항이 정리되어 있습니다 (PowerPoint 발표자 보기에서 확인).
- 1시간 덱 대비 병합: 진단+결측 → 1장, 데이터+계약+split → 1장, split/사전등록+부트스트랩 → 1장.
- 학습 절차는 2장으로 상세화: 샘플 구성(블록→윈도→증강) 1장 + 손실·최적화·keep/discard 선택 1장.
- 1시간 덱 대비 제외: 아키텍처 상세, gap별 층화 분석, Mirnov 재가공 음성 결과(한계 슬라이드에 한 줄로 축약),
  섹션 divider 6장.
- 시간이 밀릴 때 우선 줄일 슬라이드: 2번(30초로 축약) → 16번(급변 case study).

## 핵심 메시지 (자료가 담은 결론)

- **causal baseline(persistence·AR) 압도** — 강건하고 방어 가능한 결과 (온라인/실시간에서 명확한 승자).
- **CES_TI는 미래까지 보는 오프라인 보간도 통계적으로 유의하게 능가** — genuine +0.18~+0.28,
  4 seed 모두 shot-clustered 95% CI가 0을 제외 (PASS), held 포함 평가에서도 4/4.
- **배치 주장은 인과 우위** — 결측 재가중·캠페인 분할 두 스트레스 테스트를 생존하는 유일한
  주장 (+0.29 / +0.22). 오프라인 보간 대비 우위는 관측 모집단 한정.
- **CES_VT는 보간과 동률(n.s.)** — `Tᵢ ↔ V_rot` 비대칭. 빠른 진단은 10 ms 격자에서 Tᵢ 정보는
  운반하나 V_rot 정보는 거의 없음(미관측 NBI 토크 + Mirnov aliasing). 물리로 예측되고 ablation으로 확인.
- **모델의 가치는 고변동(peak) 구간에 집중** — Tᵢ global +0.272 → peak **+0.702**,
  V_rot global +0.131 → peak **+0.438** (둘 다 PASS, validation split). 출처 `THESIS_RESULTS.md` §5.1.
- **아키텍처 선택 게이트를 val loss → clean skill로 교체**해 n.s.(+0.088) → 유의(+0.20~+0.30)로 개선.

모든 수치의 출처: `docs/paper/paper_numbers.json`(동결 산출물 `data/.vt_repro_*` 계열에서
`collect_paper_numbers.py`가 자동 수집)과 `THESIS_RESULTS.md` §8g–§8n / `PROJECT_KNOWLEDGE.md`.

## 재생성 방법

수치/그림을 바꾸려면 아래 순서로 실행합니다 (저장소 루트에서):

```bash
python docs/presentation/make_figures.py       # 1) figures/*.png 생성 (matplotlib)
python docs/presentation/build_pptx.py          # 2) 1시간 발표 .pptx 빌드 (python-pptx)
python docs/presentation/build_pptx_20min.py    # 3) 20분 발표 .pptx 빌드
python docs/presentation/build_pptx_flow.py     # 4) 연구 흐름 .pptx 빌드
python docs/presentation/build_1pager.py        # 5) 1-pager .pdf/.png 빌드 (matplotlib)
```

레이아웃(겹침·넘침)을 PowerPoint 없이 확인하려면:

```bash
python docs/presentation/preview_pptx.py docs/presentation/KSTAR_CES_발표자료_20분.pptx
# -> docs/presentation/.preview/slide_NN.png + 넘침/이탈 경고 출력
```

의존성: `python-pptx`, `matplotlib`, `pillow` (모두 현재 환경에 설치됨). 한글 폰트는
`Malgun Gothic`을 사용합니다(Windows 기본). 다른 OS에서는 스크립트 상단의 폰트 후보 목록을 수정하세요.

## 생성 스크립트

- `make_figures.py` — 6개 그림: forest(headline), RMSE ladder, n.s.→유의 진전, 입력 ablation,
  peak, 결측/held 통계.
- `make_figure_architecture.py` / `make_figure_mirnov.py` / `make_figure_transient.py` — 개별 그림.
- `make_figure_window_sweep.py` — window sweep 곡선 (`fig_window_sweep.png`). history 길이 vs
  held-out test `skill_vs_pchip`, CES_TI/VT 패널 분리, seed 4점 + 평균. 24-run 배치
  (`data/.wsweep_summary.json`)를 읽으며, 요약이 없으면 `data/.wsweep_*` run 산출물을 직접 훑습니다.
  결과 해석은 THESIS_RESULTS.md §8f.

`figures/` 중 스크립트가 만들지 않는 두 장(중간 보고 덱에서 가져옴, 재생성 불가):
`fig_raw_csv_missing.png` (원본 shot CSV 스크린샷 — CES_TI 빈칸·CES_VT held 반복·빠른 진단은
전부 채워진 상태가 한눈에 보임), `fig_ar_formula.png` (local AR/선형 외삽 수식).
- `build_pptx.py` — 1시간 발표 덱 (네이티브 도형 아키텍처 다이어그램 + 그림 임베드).
- `build_pptx_20min.py` — 20분 발표 덱(24장). `build_pptx.py`를 import해 팔레트·레이아웃 헬퍼와
  그대로 쓸 수 있는 슬라이드를 재사용하고, 나머지는 병합/압축 버전으로 대체합니다.
  **수치를 고칠 때는 `build_pptx.py`를 고치면 두 덱에 함께 반영됩니다** (재사용 슬라이드에 한해).
  2026-07-30 중간 보고에서 손으로 고친 내용(Overview 부제 → "결론", 발표 순서 스트립 삭제,
  결측 슬라이드에 원본 CSV 스크린샷 추가, 평가 방법론 PR1–PR4 슬라이드 추가, held 제거 학습
  슬라이드 제외, seq 결과를 표로 교체 등)이 스크립트에 반영돼 있습니다. 1시간 덱에서 재사용하는
  슬라이드의 문구 수정은 `drop_shape` / `drop_para` / `set_para` 헬퍼로 빌드 후 적용하므로
  `build_pptx.py`가 만드는 1시간 덱은 영향을 받지 않습니다.
- `build_pptx_flow.py` — 연구 흐름 덱(15장). `build_pptx.py`의 팔레트·레이아웃 헬퍼와 `figures/`를
  재사용하지만 슬라이드는 전부 자체 정의입니다(다른 두 덱과 내용이 겹치지 않음). 새 실험 라운드가
  끝나면 타임라인(3번)·실험 요약(10·11번)·다음 작업(14번)을 갱신하세요.
- `preview_pptx.py` — PPTX → PNG 근사 렌더러(레이아웃 QC 전용, PowerPoint/LibreOffice 불필요).
  실제 폰트·좌표로 그려서 텍스트 넘침과 도형 이탈을 잡아냅니다. 정밀 렌더러가 아니므로
  최종 확인은 PowerPoint에서 하세요.
- `build_1pager.py` — A4 한 장 요약 (PDF + PNG).
