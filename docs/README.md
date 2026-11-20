# docs/ 색인 — 쓰는 것과 참고용

2026-11-16 기준. "쓰는 것"은 지금 주장과 다음 실험의 근거가 되는 문서이고, "참고용"은 지우지 않고 남긴 것으로 갱신하지 않으며 인용하지 않는다. 지금 쓰는 기록의 중심은 저장소 맨 위의 `PROJECT_KNOWLEDGE.md`와 `PAPER_DESIGN.md`이다.

## 쓰는 것

| 문서 | 내용 |
|---|---|
| `paper/ppcf/` | 원고(`main_ppcf.tex`, `supp_ppcf.tex`와 PDF), 표와 수치 매크로(`make_tables_v2.py` → `gen_v2.tex`), 그림(`fig_v2.py` → `figv2_*`), 검사(`check_numbers_v2.py`). `earlier_line/`은 2026-11-19까지의 원고를 얼려 둔 것 |
| `paper/b70_d1.md` … `paper/b86_mast_radius.md` (각각 `.json`, `_checks.md`, `_lead_checks.md`) | 2026-11-19 이후의 등록 배치 결과. 목록은 저장소 루트의 `HANDOVER.md` 6절 |
| `paper/review_loop/` | 독립 채점 기록과 고정 지시문(`GRADER_PROMPT.md`), 비판 기록 |
| `paper/file_rules.md`, `.json` | 10 ms 파일과 원시 자료의 관계: 격자, 표본, 빔 blip, blip 뒤 응답 |
| `paper/b27c_hier.json` | pooled 구간(주 수치) |
| `paper/b29_causal.md`, `b29_addendum1.md` | 빠른 채널의 증분, 인과 변형 |
| `paper/b30_thinned.md`, `b30_posthoc.md`, `b30_sensitivity.md`, `b30_stack.md`, `b30_stack_offline.md` | 드문 CES, 민감도, 결합 시험 |
| `paper/b31_bursts.md` | 가장자리 버스트 |
| `paper/mast_ped_confirmation.md`, `mast_edge.md`, `mast_gate*.md`, `mast_coupling.md`, `mast_fast_exploration.md` | MAST |
| `paper/refs.bib` | 참고문헌 |
| `presentation/KSTAR_CES_포스터_초안.pptx`, `build_poster.py`, `make_figures_poster.py`, `check_poster_numbers.py` | 포스터 초안 |

## 참고용 (갱신하지 않음)

| 문서 | 왜 참고용인가 |
|---|---|
| `reference/PROJECT_KNOWLEDGE_until_2026-11-16.md` | 정리 전의 PROJECT_KNOWLEDGE 전문. W = 4 시절과 닫았다고 적은 축들이 섞여 있다 |
| `reference/THESIS_RESULTS_section0_2026-11-11.md` | 2026-11-11의 요약 절. 이후 결과와 결측 수치 정정이 없다 |
| `../RESEARCH_CONTEXT.md`, `연구방어_종합문서.md`, `연구설명_목차별정리.md`, `재현_절차_기록.md` | W = 4 프로토콜 시절의 설명·방어·재현 문서 |
| `paper/short/` (8쪽판, 영어·한국어), `paper/paper_numbers.json`, `paper/make_figures_*.py`, `paper/figs/`, `paper/figures/` | 2026-11-11 판. 결측 수치가 옛 값(8.2 %, 23.9 % 등)이다. 원고 수치 대조의 출처로만 읽힌다 |
| `paper/overview_ko.*`, `paper/outline_ko_v2.*`, `paper/NOVELTY.md`, `paper/final_pass_2026-11-09.md`, `paper/trim_2026-11-10.md`, `paper/wording_pass_2026-11-09.md` | 전체판 논문 시절의 개요와 손질 기록 |
| `paper/review_loop/` | 채점·리뷰 기록. 독립 채점자에게 주지 않는다 |
| `paper/b28_*.json`, `../docs/b28_weight_physics_report.md` | 20-parameter 식의 세부. 원고에는 한 문단만 쓴다 |
| `theory_foundations.md`, `theory_plasma_physics.md`, `theory_results_physics.md` | 유도 과정. `analyze_foundations.py`가 대조한다. 원고의 수치 출처는 아니다 |
| `interpolation_baselines_references.md`, `lit_2026-11-09/`, `lit_2026-11-10/` | 문헌 조사 기록 |
| `ionq_qpu_실험기록.md` | 닫힌 곁가지(양자 회로) |
| `presentation/`의 다른 덱과 한 장 요약 | 2026-11-11 이전 수치 |
| `졸논 정리.pptx` | 초기 제안 덱 |
