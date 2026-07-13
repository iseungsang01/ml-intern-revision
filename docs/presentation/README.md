# 발표자료 — KSTAR CES Nowcasting

이 폴더는 프로젝트를 이해하기 위한 두 가지 산출물을 담고 있습니다.

## 산출물

| 파일 | 형식 | 용도 |
|---|---|---|
| **`KSTAR_CES_발표자료.pptx`** | PowerPoint, 33 슬라이드 (16:9) | **약 1시간 학위논문 발표용** 덱 |
| **`KSTAR_CES_1pager.pdf`** | A4 PDF, 1 페이지 | **한 장 요약** (배포용) |
| `KSTAR_CES_1pager.png` | PNG | 1-pager 미리보기 이미지 |

> 발표 덱은 8개 파트 구성: ① 배경·문제 ② 접근법(어려운 평가 bar) ③ 데이터·파이프라인
> ④ 모델 아키텍처 ⑤ 평가 방법론(통계) ⑥ 결과 ⑦ AutoML 자율 연구 루프 ⑧ 결론·한계·향후 연구.

## 핵심 메시지 (자료가 담은 결론)

- **causal baseline(persistence·AR) 압도** — 강건하고 방어 가능한 결과 (온라인/실시간에서 명확한 승자).
- **CES_TI는 미래까지 보는 오프라인 보간도 통계적으로 유의하게 능가** — 최종 모델·4 seed 모두
  shot-clustered 95% CI가 0을 제외 (PASS), genuine-only 평가에서도 강건.
- **CES_VT는 보간과 동률(n.s.)** — `Tᵢ ↔ V_rot` 비대칭. 빠른 진단은 10 ms 격자에서 Tᵢ 정보는
  운반하나 V_rot 정보는 거의 없음(미관측 NBI 토크 + Mirnov aliasing). 물리로 예측되고 ablation으로 확인.
- **모델의 가치는 고변동(peak) 구간에 집중** — Tᵢ peak skill +0.86, V_rot peak +0.69 (둘 다 PASS).
- **Claude 기반 keep/discard autoresearch**로 n.s.(+0.088) → 유의(+0.20~+0.30)로 개선.

모든 수치의 출처: `data/.improve_final_out/`, `data/.ms_out_{1,7,123}/`, `data/.final_out/`의
frozen artifacts와 `THESIS_RESULTS.md` / `PROJECT_KNOWLEDGE.md`.

## 재생성 방법

수치/그림을 바꾸려면 아래 순서로 실행합니다 (저장소 루트에서):

```bash
python docs/presentation/make_figures.py    # 1) figures/*.png 생성 (matplotlib)
python docs/presentation/build_pptx.py       # 2) 발표 .pptx 빌드 (python-pptx)
python docs/presentation/build_1pager.py     # 3) 1-pager .pdf/.png 빌드 (matplotlib)
```

의존성: `python-pptx`, `matplotlib`, `pillow` (모두 현재 환경에 설치됨). 한글 폰트는
`Malgun Gothic`을 사용합니다(Windows 기본). 다른 OS에서는 스크립트 상단의 폰트 후보 목록을 수정하세요.

## 생성 스크립트

- `make_figures.py` — 6개 그림: forest(headline), RMSE ladder, n.s.→유의 진전, 입력 ablation,
  peak, 결측/held 통계.
- `build_pptx.py` — 발표 덱 (네이티브 도형 아키텍처 다이어그램 + 그림 임베드).
- `build_1pager.py` — A4 한 장 요약 (PDF + PNG).
