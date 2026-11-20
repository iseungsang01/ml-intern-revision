# docs/ 색인

2026-11-21 정리 뒤의 상태다. 여기 있는 것은 모두 지금 줄기가 쓰거나, 지금 줄기의 코드가 읽는 것이다. 배선은 저장소 루트의 `README.md`에 있다.

| 위치 | 내용 |
|---|---|
| `paper/ppcf/` | 원고(`main_ppcf.tex`, `supp_ppcf.tex`와 PDF), 표와 수치 매크로(`make_tables_v2.py` → `gen_v2.tex`), 그림(`fig_v2.py` → `figv2_*`), 검사(`check_numbers_v2.py`). `earlier_line/`은 2026-11-19까지의 원고를 얼려 둔 것으로, 보충 자료의 자료 절(`make_supp_fragment.py`)이 거기서 온다. `fig_paper.py`는 보충 자료의 그림 둘(`fig_geometry`, `fig_rows`)과 포스터 그림을 만든다 |
| `paper/b70_*` … `paper/b86_*` (`.md`, `.json`, `_checks.md`, `_lead_checks.md`) | 지금 줄기의 등록 배치 결과. 목록은 `HANDOVER.md` 6절 |
| `paper/b27*` … `paper/b68*`, `paper/mast_*` | 이어받은 배치의 결과 파일. 지금 줄기의 코드나 발표 그림이 읽는다. 수치는 D1의 판독(`PROJECT_KNOWLEDGE.md` 4.48절, 4.50절)이 대체하므로 인용하지 않는다 |
| `paper/file_rules.md`, `nbi_files.md`, `dataset_ledger.md` | 자료 사실: 10 ms 파일과 원시 자료의 관계, NBI 파일, D1의 행과 방전 수 |
| `paper/prior_work_missing_ti.md`, `validation_methods.md` | 선행 연구 검토, 다른 논문의 검증 방식 |
| `paper/review_loop/` | 2026-11-20의 비판 넷과 그 점검 코드, 독립 채점 기록(16–18차), 고정 채점 지시문 `GRADER_PROMPT.md` |
| `presentation/` | 2026-11-19 판의 발표 덱·포스터와 그 빌더(`README.md` 참조) |
| `reference/ces_accuracy_literature_2026-11-18.md` | CES 정확도에 관한 문헌 노트(`PAPER_DESIGN.md`가 가리킨다) |
| `졸논 정리.pptx` | 발표 덱의 배경 틀(`presentation/build_deck_ppcf.py`가 읽는다) |
