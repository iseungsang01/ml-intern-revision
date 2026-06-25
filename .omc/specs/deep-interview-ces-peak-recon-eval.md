# Deep Interview Spec: 고변동(피크) 구간 복원 평가·사례·이유 분석

## Metadata
- Interview ID: ces-peak-recon-2026-06-23
- Rounds: 6
- Final Ambiguity Score: 19%
- Type: brownfield
- Generated: 2026-06-23
- Threshold: 0.2
- Threshold Source: default
- Initial Context Summarized: no
- Status: PASSED

## Clarity Breakdown
| Dimension | Score | Weight | Weighted |
|-----------|-------|--------|----------|
| Goal Clarity | 0.84 | 0.35 | 0.293 |
| Constraint Clarity | 0.75 | 0.25 | 0.188 |
| Success Criteria | 0.80 | 0.25 | 0.199 |
| Context Clarity | 0.85 | 0.15 | 0.128 |
| **Total Clarity** | | | **0.808** |
| **Ambiguity** | | | **0.192 (19%)** |

## Topology
| Component | Status | Description | Coverage / Deferral Note |
|-----------|--------|-------------|--------------------------|
| 통계 유의성 baseline 유지 | active | 선형/스플라인(PCHIP) 보간 대비 통계적 유의성을 *기본 전제*로 계속 유지 | 이미 `compare_baselines.py` + `bootstrap_compare.py`로 구현됨. 피크 작업이 이를 깨지 않고 위에 얹힘 (AC1) |
| 피크 중심 복원 평가 | active | 비운 실측 CES의 고변동(피크) 구간 복원 오차를 보간 대비 측정 | 신규 평가 경로 (AC2, AC3, AC4) |
| 피크 사례 수집·시각화 | active | best/median/worst skill 스펙트럼 사례를 시계열 그림으로 수집 | 신규 plotting (현재 코드 0개) (AC5, AC6) |
| 왜 잘 잡는지 이유 분석 | active | 단일 ablation(fast diagnostics 제거)으로 멀티모달 인과 입증 | `evaluate.py` no_fast hook 활용 (AC7) |
| 피크 가중 학습/목적함수 변경 | **deferred** | 피크에서 loss가 더 줄도록 학습 목적함수 자체를 바꾸는 것 | **사용자 명시 non-goal (2026-06-23).** "피크 가중 없이 일반 MSE로 학습한 모델이 피크까지 잘 잡음"을 입증하는 게 목적. 만약 못 잡으면 그때 가중학습이 옵션이 될 수 있으나, 그 *판단·제안은 `automl_agent_loop.py`(Claude-researcher 루프)가 자동으로* 하도록 둠 — 수동 설계하지 않음 |

## Goal
기존에는 "선형/스플라인 보간 대비 통계적 유의성만 보이면 만족"이었으나, 이제는 그 유의성을 **기본 전제로 유지**하면서 한 단계 더 나아간다:

**피크 가중치를 일절 주지 않고 평범한 균일 masked-MSE로 학습한 바로 그 모델이, 비워둔 실측 CES 값 중에서도 특히 *변동성이 큰 지점(국소 극값/급변/고분산, ELM·sawtooth 같은 transient가 대표 예시)*에서 보간보다 통계적으로 유의하게 잘 복원함을 (1) 정량 지표로 증명하고, (2) 공정한 best/median/worst 사례 스펙트럼으로 보이며, (3) 그 이유(보간엔 없는 멀티모달 BES/ECEI/MC 신호를 모델이 봄)를 단일 ablation으로 입증한다.**

## Constraints
- **data/model 계약을 깨지 않는다** (CLAUDE.md 중앙 불변식): `model.forward` 시그니처, 정규화 `[CES_TI, CES_VT]` 출력, train-file-only NaN-aware 정규화, ces_history 마스킹, per-target masked MSE 등 그대로.
- **학습/목적함수/모델 구조를 바꾸지 않는다.** 이번 작업은 *기존 학습된 모델 위에서의 평가·시각화·분석*만. (피크 가중 학습은 deferred non-goal.)
- **피크는 CES 신호 자체에서 검출**한다 — 국소 극값(local max/min), 큰 |ΔCES|, 높은 국소 분산. **MC/ECEI 기반 ELM 검출기는 구현하지 않는다** (Round 5에서 ELM-특이성은 철회; ELM은 서술용 예시일 뿐).
- **MNAR 인지**: ELM 순간 CES는 저 S/N으로 결측될 수 있음 (`PROJECT_KNOWLEDGE.md:72`). 따라서 평가는 **관측된(hold-out 가능한) CES 값 중 고변동 지점**에 한정한다. 결측된 transient 순간 자체는 ground-truth가 없어 직접 평가 불가.
- **clean(비증강) 검증 split만 사용** — `compare_baselines.py`/`evaluate.py`의 `build_clean_val_subset` 단일 소스, file-level split, shot 누수 없음.
- **물리 단위(denormalized)·per-target(CES_TI/CES_VT 분리)** 로 리포트 — 기존 평가 관례 유지. (기존 결과상 CES_TI는 보간을 이기나 CES_VT는 아닐 수 있음 → 피크 분석도 per-target 비대칭이 드러날 수 있음.)
- **no-leakage**: 피크 평가도 target 자기 값 미사용(기존 `build_neighbor_set` 규칙), acausal 보간은 future 없을 때 persistence fallback 유지.
- **공정성**: 사례는 cherry-pick 금지 — skill 기준 best/median/worst 스펙트럼으로 성공·실패를 함께 제시.
- **git 자동 커밋 금지** (CLAUDE.md): 명시 요청 없이는 commit/push 하지 않음.

## Non-Goals
- 피크 가중 손실/목적함수 설계 (deferred — automl 루프가 필요 시 자동 제안).
- model.py / train.py 의 학습 로직 변경.
- MC/ECEI/Dα 기반 ELM 타이밍 검출기 구현.
- 새 데이터 수집·합성 (실데이터만; [[no-fake-data]]).
- 통계 유의성 프레임 자체의 교체 (기존 부트스트랩 CI 게이트는 유지·재사용).

## Acceptance Criteria
- [ ] **AC1 (baseline 유지):** 기존 `compare_baselines.py` + `bootstrap_compare.py`의 전역 skill_vs_pchip 및 shot-clustered 95% CI 게이트(CI 하한 > 0)가 그대로 동작하며, 피크 작업 추가 후에도 회귀 없음 (`tests/test_baselines_interpolation.py`, `tests/test_bootstrap_compare.py` 통과).
- [ ] **AC2 (피크 검출):** 관측된 CES 시계열에서 고변동 지점을 검출하는 함수가 추가됨 — 국소 극값/큰 |ΔCES|/높은 국소 분산 기준, 합리적 기본 파라미터(예: contiguous block 내 1·2차 차분 + rolling-std top-N%), 결정적(seed 고정), no-leakage 준수.
- [ ] **AC3 (피크 구간 지표):** 피크로 판정된 timestep 부분집합에 대해 per-target `peak_skill_vs_pchip`(및 vs linear), `peak_rmse_model`/`peak_rmse_pchip`를 물리 단위로 산출하고 JSON으로 출력.
- [ ] **AC4 (피크 유의성):** 피크 부분집합에 대한 shot-clustered 부트스트랩 95% CI를 산출하여, "피크 구간에서도 보간 대비 유의하게 우수(CI 하한 > 0)"한지 per-target으로 판정. (CES_TI/CES_VT 비대칭 결과 그대로 보고.)
- [ ] **AC5 (사례 수집):** 피크 사례를 skill(model vs 보간 개선폭) 기준 best/median/worst 각 k개(기본 k=3~5) 선별하는 결정적 로직.
- [ ] **AC6 (시각화):** 선별된 각 사례에 대해 CES 시계열 그림 생성 — 실측값·모델 예측·보간(PCHIP/linear)을 한 그림에 겹쳐, 피크 지점 강조. matplotlib `savefig`로 파일 저장(현재 plotting 코드 0개 → 신규).
- [ ] **AC7 (이유 분석 ablation):** fast diagnostics(BES/ECEI/MC) zero-out(`no_fast`) 조건에서 피크 구간 skill이 유의하게 하락함을 보여, "모델이 보간엔 없는 멀티모달 신호를 봐서 고변동 지점을 잡는다"는 인과를 정량 입증. full vs no_fast 피크 skill 비교를 리포트.
- [ ] **AC8 (루프 연동):** 피크 구간 지표가 평가 산출물(JSON)에 노출되어 `automl_agent_loop.py`/`program.md`가 읽을 수 있어, 피크 성능이 약하면 루프가 (가중학습 등) 개선을 자동 제안할 근거가 됨. (스코어링 기본 동작은 바꾸지 않되 지표를 가용하게.)
- [ ] **AC9 (테스트·문서):** 신규 코드에 단위 테스트 추가, `python -m pytest -q` 통과; 행동이 바뀌면 smoke 학습 1회 실행 기록; `HANDOFF.md` 갱신, 주요 발견은 `PROJECT_KNOWLEDGE.md` 반영.

## Assumptions Exposed & Resolved
| Assumption | Challenge | Resolution |
|------------|-----------|------------|
| "보간 대비 유의성"을 새 목표로 교체 | 사용자: 유의성은 기본으로 유지 | 통계 유의성은 *foundation*으로 유지, 피크 분석은 그 위에 추가 (AC1) |
| 피크 = ELM, MC/ECEI로 검출 | 사용자: ELM은 *예시*일 뿐, 목적은 고변동 지점 일반 | 피크 = CES 신호의 고변동 지점; ELM 검출기 불필요 → 구현 단순화 (AC2) |
| ELM 순간 CES를 직접 평가 | MNAR: ELM 시 CES 결측 (PROJECT_KNOWLEDGE.md:72) | 관측된 고변동 지점에 한정 평가 |
| 피크 가중 학습을 지금 설계 | 사용자: non-goal; 필요 시 *루프가 자동 제안* | deferred; 피크 지표를 루프에 노출해 자동 제안 근거만 제공 (AC8) |
| best-case만 모아 보여줌 (Contrarian) | cherry-picking 논문 심사 비판 | best/median/worst 스펙트럼으로 공정성 확보 (AC5) |
| 이유 분석을 깊게(어텐션+타이밍+ablation) (Simplifier) | 가장 단순하지만 가치 있는 버전? | 단일 ablation(no_fast)으로 인과 입증 (AC7) |

## Technical Context (brownfield, from explore agent)
- **평가:** `evaluate.py` — clean 비증강 val, 물리 단위, per-target, `skill_vs_persistence`(L202-235). `no_fast`/`no_history` ablation hook 존재(L80-92,135-137).
- **baseline:** `compare_baselines.py` — headline PCHIP, `skill_vs_pchip`(L181-182), gap-stratified bins(L195-215), per-sample 제곱오차 npz 아카이브(L217-222). `baselines_interpolation.py` — persistence/linear/pchip/ar_local/gp, no-leakage `build_neighbor_set`(L61-75), block 검출 `GAP_SECONDS=0.5`.
- **유의성:** `bootstrap_compare.py` — shot-clustered 부트스트랩(L26-32), 95% CI, 게이트 CI 하한 > 0(L63), seed 12345.
- **손실:** `train.py` per-target masked MSE + 0.1·relu(zero_floor−TI) (L484-488). 피크 가중 없음(균일).
- **데이터:** `dataset.py` — target timestep 완전 마스킹(L381-382), per-target 독립 결측(≈8%/24%), block은 time delta<0.5. 컬럼 `BES_*/ECEI_*/MC*/time/CES_TI/CES_VT` (ELM/Dα 라벨 없음).
- **루프:** `automl_agent_loop.py` — `comparison_skill`=mean per-target `skill_vs_pchip`(L117-133)로 keep/discard.
- **없는 것:** 피크/스파이크 처리, plotting/figure 코드 전무.

## Ontology (Key Entities)
| Entity | Type | Fields | Relationships |
|--------|------|--------|---------------|
| CES_TI | core target | 정규화/물리 이온온도, observed flag | Peak가 발생하는 신호; 모델이 보간을 이김(기존) |
| CES_VT | core target | 정규화/물리 토로이달 회전, observed flag | Peak 발생; 보간 대비 우위 약할 수 있음(비대칭) |
| Peak / 고변동 지점 | core domain | 국소 극값/큰 |ΔCES|/높은 국소 분산, timestep | observed CES에서 검출; 평가·사례·분석의 대상 |
| Interpolation baseline | external/comparison | PCHIP/linear/spline, persistence | model이 피크 구간에서 이겨야 할 대상 |
| Peak-subset skill | metric | peak_skill_vs_pchip, peak_rmse, CI | Peak 구간에 한정한 model vs baseline |
| Shot (CSV) | data unit | file, rows, time | 부트스트랩의 클러스터 단위 |
| Fast diagnostics | model input | BES/ECEI/MC 채널 | ablation 대상; "이유"의 핵심(보간엔 없음) |
| Example case | deliverable | best/median/worst, skill, shot, t | 시계열 그림으로 시각화 |
| Figure (time-series) | deliverable | 실측·모델·보간 overlay, 피크 강조 | 사례별 savefig |
| Bootstrap CI | statistical gate | shot-clustered 95% CI, seed 12345 | 유의성 판정(CI 하한>0) |
| AutoML loop | consumer | comparison_skill, program.md | 피크 지표를 읽어 개선 자동 제안 |
| ELM | illustrative example | transient crash | 고변동 지점의 물리 예시(검출 기준 아님) |

## Ontology Convergence
| Round | Entity Count | New | Changed | Stable | Stability Ratio |
|-------|-------------|-----|---------|--------|----------------|
| 1 | 9 | 9 | - | - | N/A |
| 2 | 11 | 2 | 1 (Peak→ELM transient) | 8 | 82% |
| 3 | 12 | 1 (ELM detector) | 1 | 11 | 90% |
| 4 | 13 | 1 (사례 스펙트럼) | 0 | 12 | 92% |
| 5 | 12 | 0 | 1 (Peak 재정의/ELM 강등) | 11 | 85% |
| 6 | 12 | 0 | 0 | 12 | 95% |

## Interview Transcript
<details>
<summary>Full Q&A (6 rounds + Round 0 topology)</summary>

### Round 0 — Topology
**확정:** 활성 4 (통계유의성 유지 / 피크 복원 평가 / 사례 시각화 / 이유 분석), 보류 1 (피크 가중 학습 = non-goal).

### Round 1
**Q:** CES 시계열에서 "튀는 피크 값"을 코드가 어떻게 검출/정의?
**A:** 국소 극값/급변 (local max/min 또는 |ΔCES| 급변).
**Ambiguity:** 55%

### Round 2
**Q:** 피크 구간에서 "잘 잡았다"를 어떤 지표로 측정하고 합격 기준은?
**A:** 피크 구간 skill_vs_보간 (+ 부트스트랩 CI 하한 > 0).
**Ambiguity:** 50%

### Round 3
**Q:** ELM/피크 지점을 어떻게 식별? (ELM 라벨 없음, ELM 시 CES 결측 가능)
**A:** MC/ECEI로 ELM 시점 추론. *(→ Round 5에서 철회됨)*
**Ambiguity:** 40%

### Round 4 (Contrarian)
**Q:** 피크 사례를 어떻게 선별해야 공정하면서 설득력 있을까? (cherry-picking 회피)
**A:** 개선폭(skill) 기준 상·중·하 스펙트럼.
**Ambiguity:** 32%

### Round 5
**Q:** 일반 학습 모델이 ELM 피크를 잡는 "이유"를 무엇으로 입증?
**A:** "ELM을 콕 집는 게 목적이 아니라 변동성 큰 지점에서 모델이 잘 맞는다는 것. ELM은 예시일 뿐." → 피크 정의를 CES 신호 고변동 지점으로 정정, MC/ECEI 검출기 철회.
**Ambiguity:** 29%

### Round 6 (Simplifier)
**Q:** "이유 분석/설명"을 어느 깊이까지?
**A:** ablation 1개 (fast diagnostics 제거 → 고변동 구간 skill 하락).
**Ambiguity:** 19% ✅

</details>
