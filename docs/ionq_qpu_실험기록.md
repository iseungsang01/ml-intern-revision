# IonQ QPU 실험 기록 (2026-07-26)

`docs/ces_vt_proposals.md`(2026-08-09 제거, git 이력 참조) 부록("양자컴퓨팅으로 이 문제를 풀 수 있는가")은 문헌 근거만으로
"아니다"라고 결론냈다. 이 문서는 **실제로 IonQ 계정을 붙여서 돌려본 기록**이다. 다음에
다시 시도할지 판단할 근거를 남기는 것이 목적이다.

결론 요약:

1. **하드웨어를 못 썼다.** 크레딧은 멀쩡히 있는데 Forte 전 기종이 `unavailable`이라
   QPU 타깃 job이 전부 ideal 시뮬레이터로 강등됐다.
2. **동일 조건 비교에서 VQC가 고전 MLP에 완패했다.** 이건 하드웨어와 무관하게 나온 결과다.
3. 다시 시도할 가치가 있는지는 §5의 판단 기준으로 결정한다.

---

## 1. 계정 상태 (확인된 사실)

`GET https://api.ionq.co/v0.4/projects` 응답:

```json
{"name": "Personal Workspace (lss010330@snu.ac.kr)",
 "quotaUsage": 0, "quotaLimit": 1493.92, "unit": "USD",
 "allowedTargets": ["qpu.forte-1"]}
```

- 크레딧은 **시간이 아니라 $1,493.92 USD**이고 **사용액 0**이다.
  (처음에 "10분/30분" 시간 단위로 알고 계산했던 shot 예산은 전제가 틀렸다.)
- `qpu.forte-1`은 **허용 목록에 정상 등록**되어 있다. 즉 권한 문제가 아니다.

`GET /v0.4/backends`:

| 백엔드 | 상태 |
|---|---|
| `qpu.harmony`, `qpu.aria-1`, `qpu.aria-2` | **retired** (영구 퇴역) |
| `qpu.forte-1`, `qpu.forte-enterprise-1`, `qpu.forte-enterprise-3` | **unavailable** |
| `simulator` | available |

## 2. QPU 강등 증거

`target: "qpu.forte-1"`로 최소 회로(1큐빗 H, 1 shot)를 제출한 job
`019f9a4e-bed2-765c-bc39-d34700c5f8f9` 응답:

```json
{"backend": "simulator",          // forte-1로 제출했으나 강등됨
 "noise": {"model": "ideal"},     // 노이즈 없음 = 실제 하드웨어 아님
 "execution_duration_ms": 0,
 "status": "completed"}
```

`qpu.forte-enterprise-1`, `qpu.forte-enterprise-3`도 동일하게 `backend=simulator`,
`noise.model=ideal`로 처리됐다.

**중요한 함정:** job은 `completed`로 정상 종료되고 결과도 나온다. `backend` 필드를 확인하지
않으면 QPU 결과라고 착각하기 쉽다. 다음에 돌릴 때 **반드시 `backend`와 `noise.model`을
검증**해야 한다. `noise.model == "ideal"`이면 하드웨어가 아니다.

## 3. VQC vs 고전 MLP (하드웨어 없이도 유효한 결과)

`ces_prediction/quantum_vqc.py`로 실행. 공정성을 위해 통제한 것:

- **동일 입력**: 8큐빗에 92차원을 못 올리므로 train-only PCA로 92 → 8차원 축소
  (분산 88.86% 보존). 고전 대조군도 **byte-identical한 축소 입력**을 받는다.
- **동일 파라미터 예산**: VQC 98개 vs MLP 101개(hidden=10). 815,788개짜리 운영 모델과
  비교하는 것은 무의미하므로 하지 않았다.
- **동일 평가**: `evaluate.py`와 같은 seed·cap·split manifest → 같은 val 샘플,
  같은 persistence 기준, 물리 단위 `skill_vs_persistence`.
- **동일 lr 스윕**: 어느 쪽에도 튜닝 이점을 주지 않기 위해 5개 lr을 양쪽 모두 시도.

타깃 `CES_TI`, train 4,000 / val 8,000, 12 epoch 기준:

| 모델 | lr | skill_vs_persist | RMSE |
|---|---|---|---|
| MLP | 0.01 | **+0.3634** | 419.13 |
| MLP | 0.02 | +0.3593 | 420.48 |
| MLP | 0.05 | +0.3495 | 423.70 |
| MLP | 0.10 | +0.3345 | 428.54 |
| MLP | 0.20 | +0.2943 | 441.30 |
| VQC | 0.20 | **+0.1848** | 474.31 |
| VQC | 0.02 | +0.1788 | 476.03 |
| VQC | 0.01 | +0.1559 | 482.65 |
| VQC | 0.05 | +0.1369 | 488.04 |
| VQC | 0.10 | +0.1310 | 489.69 |

**가장 나쁜 MLP(+0.2943)가 가장 좋은 VQC(+0.1848)보다 낫다.** lr 선택의 문제가 아니라
구조적 격차다. VQC는 학습 손실도 0.83~0.85에서 정체했고 MLP는 0.66까지 내려갔다.

학습 시간(같은 조건, 로컬 상태벡터 시뮬레이터): **VQC 23~35초 vs MLP 0.1초 = 200~350배**.
이건 실제 하드웨어가 아니라 시뮬레이터에서 잰 값이고, 하드웨어에서는 더 나빠진다.

> 주 30 epoch × 20,000 샘플 본 실행은 도중에 중단했다. 위 스윕에서 이미 순위가
> 뒤집힐 여지가 없어 보였고, 사용자 판단으로 학습을 멈췄다. 다시 돌린다면
> `QVQC_MAX_TRAIN=20000 QVQC_EPOCHS=30`.

## 4. shot noise — QPU가 이 문제에 안 맞는 핵심 이유

현행 운영 모델의 `eval_metrics.json` 기준 신호 크기:

| 타깃 | 신호 (persistence 대비 RMSE 개선) | 타깃 std | 필요한 shots/샘플 |
|---|---|---|---|
| CES_TI | 505.4 − 382.8 = **122.6** eV | 496.2 | ~410 |
| CES_VT | 30.24 − 24.98 = **5.25** km/s | 54.86 | ~2,730 |

(shot noise를 신호의 1/5로 낮추는 기준. `<Z>` 추정 오차는 1/√shots.)

`CES_VT`는 100 shots에서 shot noise가 **5.49 km/s로 신호 5.25 km/s보다 크다.** 즉
개선하고 싶었던 타깃이 하필 QPU와 가장 안 맞는다.

IonQ 클라우드 시뮬레이터(3샘플 × 200 shots) 실측: `<Z0>` 편차 std **0.0892**,
이론 shot noise 0.0707과 같은 자릿수로 일치 — 계산이 맞다는 확인.

**학습은 여전히 불가능하다.** parameter-shift는 파라미터당 회로 2회이므로 배치 32 ·
98파라미터 = 6,272 circuits/step. Forte 게이트 시간(1q 130 µs, 2q 600 µs)에서 이 회로는
128×1q + 32×2q = **shot당 약 40 ms**이므로, 100 shots 기준 **gradient 1 step ≈ QPU 7시간**이다.
크레딧이 달러였으므로 "크레딧 초과"는 아니지만, 시간·비용 모두 비현실적인 건 그대로다.
→ **하드웨어는 추론에만 쓴다**는 설계는 유지.

## 5. 다음에 판단할 기준

재시도 전에 확인할 것:

1. **`GET /v0.4/backends`에서 `qpu.forte-1`이 `available`인가.** 아니면 아무것도 하지 않는다.
2. 제출 후 **`backend` 필드가 `qpu.forte-1`이고 `noise.model != "ideal"`인가.**
   아니면 그 결과는 하드웨어 결과가 아니다.

재시도할 가치가 있는가 — 솔직한 평가:

- §3 결과가 뒤집힐 가능성은 낮다. 하드웨어는 시뮬레이터보다 **나쁘기만** 하다
  (노이즈 추가). 즉 QPU에서 VQC가 MLP를 이길 경로가 없다.
- 그래도 돌릴 이유가 있다면 **"IonQ 크레딧 리포트"나 논문 부록의 정량화된 negative
  result**로서다. 그 목적이면 `CES_TI` 기준 500 shots × 수십~수백 샘플이면 충분하고,
  크레딧 $1,493.92는 넉넉하다.
- 반대로 성능 개선이 목적이라면 하지 않는 게 맞다. 그 제안서 §부록 4번
  ("병목이 모델 용량이 아니다")이 이번 실험으로 다시 확인됐다.

## 6. 추가된 코드

| 파일 | 역할 |
|---|---|
| `ces_prediction/quantum_vqc.py` | VQC vs matched-parameter MLP 통제 비교. 로컬 시뮬레이터 학습. `quantum_vqc_weights.pt` 저장 |
| `ces_prediction/ionq_infer.py` | 학습된 회로를 IonQ에서 추론. **기본이 시뮬레이터**, 하드웨어는 `--hardware --yes` 필요. 사전 예산(shots/시간/비용) 출력 |

둘 다 기존 파이프라인(`dataset.py` / `model.py` / `train.py` / `evaluate.py`)을 건드리지 않는다.
데이터 계약도 변경하지 않았다. `evaluate.py`의 `_persistence_from_history`, `_load_stats`와
동일 seed·cap을 재사용해 같은 val 샘플에서 비교된다.

실행:

```bash
py ces_prediction/quantum_vqc.py              # 비교 실험 (로컬, 무료)
py ces_prediction/ionq_infer.py               # IonQ 클라우드 시뮬레이터 추론 (무료)
py ces_prediction/ionq_infer.py --hardware    # 예산만 출력하고 중단
py ces_prediction/ionq_infer.py --hardware --yes   # 실제 QPU (available일 때만 의미 있음)
```

환경변수: `QVQC_N_QUBITS`(8), `QVQC_N_LAYERS`(4), `QVQC_MAX_TRAIN`, `QVQC_EPOCHS`,
`QVQC_TARGET`(CES_TI), `QVQC_LR_VQC`(0.2), `QVQC_LR_MLP`(0.01),
`IONQ_N_SAMPLES`(24), `IONQ_SHOTS`(500), `IONQ_QPU_BACKEND`(forte-1).

`IONQ_API_KEY`는 `.env`에 있고 `.gitignore:12`가 잡고 있어 커밋되지 않는다.

## 7. 부수적으로 발견한 것

실험 중 `torch.cuda.is_available()`이 **False**로 나왔다. 메모리에 기록된 GPU 사용 가능
상태(RTX 5060, torch 2.11.0+cu128)와 어긋난다. 이번 실험은 8큐빗 CPU 시뮬레이션이라
영향이 없었지만, **본 학습 파이프라인에는 영향이 크므로 별도 확인이 필요하다.**

---

# 재방문 (2026-08-23): 하드웨어 최초 도달, 크레딧 출처 규명, 실측 과금표

2026-07-26 기록은 §5에 "재시도 전 확인할 것" 두 가지를 남기고 닫혔다. 오늘 그 두 관문을
**처음으로 둘 다 통과**했다. 이 절은 그 기록이다.

## 8. §5 관문 통과

**기준 1 (백엔드 가용성).** `GET /v0.4/backends` 응답이 7월과 달라졌다.

| 백엔드 | 2026-07-26 | 2026-08-23 |
|---|---|---|
| `qpu.forte-1` | unavailable | **available** (36q, `degraded: false`) |
| `qpu.forte-enterprise-1` | unavailable | **available** (36q) |
| `qpu.aria-1/2`, `qpu.harmony` | retired | retired |

`allowedTargets`도 `["qpu.forte-1"]`에서 `forte-1` + `forte-enterprise-1/2/3` 4종으로 늘었다.

**기준 2 (실제 하드웨어 도달).** 1큐빗·1게이트·1샷 최소 회로를 `qpu.forte-1`로 제출
(job `01a02d61-cbfa-7148-9420-ca2627171cfa`):

```json
{"backend": "qpu.forte-1",          // 강등되지 않음 (7월엔 전부 simulator)
 "dry_run": false,
 "cost_model": "2QGE_operations",
 "execution_duration_ms": 669,      // 7월 시뮬레이터 job은 0이었다
 "submitted_at": "...06:49:39Z", "started_at": "...06:54:41Z",   // 큐 5분
 "stats": {"qubits": 1, "gate_counts": {"1q": 1, "2q": 0},
           "billed_quantum_compute_time_us": 0}}
```

과금 $25.79가 실제로 발생했다(`quotaUsage` 0 → 25.7899). **시뮬레이터 job은 무료이므로
과금 발생 자체가 하드웨어 실행의 독립 증거다.**

> **7월 §2의 판별법을 정정한다.** 그때 "`noise.model == "ideal"`이면 하드웨어가 아니다"라고
> 적었는데, 실하드웨어 job에는 `noise` 필드가 **아예 없다**(노이즈 모델은 시뮬레이터 개념).
> 필드 부재를 강등으로 오판하지 말 것. 올바른 판별은 세 가지다 — `backend`가 요청한 QPU와
> 같은가, `execution_duration_ms > 0`인가, 과금이 발생했는가.

또 하나 정정: `average_queue_time`은 forte-1이 20,471,079로 표시되지만 **실측 큐는 5분**이었다.
이 필드로 처리량을 계획하지 말 것.

## 9. 크레딧 출처 (7월에 규명하지 못했던 것)

7월 §1은 "크레딧은 시간이 아니라 $1,493.92 USD"라고 정정했다. **과금 단위로는 맞지만
출처를 놓쳤다.** 메일 기록으로 확인한 실제 구조는 이렇다.

- **2026-07-01, `kqhub@koreaqc.org`** — "IonQ 양자컴퓨팅 클라우드 서비스 체험 프로그램" 선정.
  - **배정시간: 1인당 20분**
  - **사용기간: 계정 발급일 ~ 9월 30일**
  - 문의: `helpdesk@koreaqc.org`
- **2026-07-02, `support@ionq.com`** — 초대장. 초대 링크 JWT를 풀면
  `organization: kr.re.kisti`, `orgName: KISTI`, `role: user`, 초대자 `Seung Hee LEE`.

즉 **원 grant는 20분짜리 QPU 시간이고, IonQ가 그것을 USD로 환산해 프로젝트 예산에 넣은 것이
$1,493.92다.** 두 표현이 모두 맞다. 교차검증도 맞는다 — 우리 8큐빗·4레이어 회로 기준
$1,493.92는 대략 14~20분의 Forte 벽시계 시간에 해당한다.

**이것이 API 401들도 설명한다.** `/users/me`, `/organizations`, `/characterizations`,
`/jobs/estimate`가 전부 401인 것은 키가 잘못된 게 아니라 **KISTI 조직 하위 `user` 롤**이라
조직 조회·셀프 견적 권한이 없어서다. 앞으로도 그대로일 것이다.

**따라서 실질 제약은 잔고가 아니라 마감이다: 2026-09-30.** 쓰지 않으면 소멸한다.

무관한 과거 건: SKKU 양자정보연구지원센터가 2025 퀀텀챌린지로 지급한 **$300**(2025-10-21)이
있었고 그 지원은 **2025-12-31 종료**됐다(2025-12-02 "out of budget" 메일이 그 흔적).
지금 잔고와는 별개다.

## 10. 실측 과금표 — 2단 정액제

**`dry_run: true`로 제출하면 과금 없이(`quotaUsage` 변화 0.0000) 정확한 견적이 나온다.**
`POST /jobs` with `dry_run`, 그 다음 `GET /jobs/{id}/cost`. 401인 `/jobs/estimate` 대신
이 경로를 쓸 것. 이걸로 뽑은 실제 가격표(8큐빗·4레이어, 128×1q + 32×2q):

| shots | 비용 | debiasing |
|---|---|---|
| 1 · 100 · 200 · 300 · **400** | **$25.79** | off |
| 500 · 1000 · 2000 | $168.20 | **강제 on** (32 variants) |
| 5000 | $284.32 | 32 variants |
| 10000 | $568.64 | 32 variants |

핵심은 세 가지다.

1. **회로 크기와 무관하다.** 2q 게이트를 8개에서 32개까지 바꿔도 500샷이면 전부 $168.20.
   `cost_model`이 `2QGE_operations`이지만 우리 회로는 연산량이 최소요금 아래로 깔린다.
   즉 **실질적으로 job당 정액제**이고, 두 값은 공개된 "error mitigation off/on" 최소요금과
   정확히 일치한다.
2. **경계는 400 ↔ 500이다.** 400샷까지 $25.79, 500샷부터 $168.20. 7월 §4에서 계산한
   `CES_TI` 분해 필요 샷 수(~410)가 공교롭게 이 경계에 걸친다.
3. **debias는 끌 수 없다.** `error_mitigation: {"debias": false}`를 보내도 500샷 이상에서는
   여전히 32 variants가 적용되고 가격도 그대로다.

부수적으로 **멀티회로 job은 지원되지 않는다** (`input.circuit is required` / `input must be
of type object`). 샘플당 job 1개가 강제되므로 정액요금을 분산할 방법이 없다.

**예산 효율이 7배 갈린다.** 남은 $1,468.13 기준 400샷 job은 56개, 2000샷 job은 8개다.
총 샷 수도 400샷 쪽이 많다(22,400 vs 16,000).

## 11. "더 빠른가"에 대한 정면 답 — 아니오, 5~8 자릿수

7월 기록은 학습 시간만 비교했다. 추론 지연을 실측·계산하면 이렇다. 우리 회로는 shot당
128×1q + 32×2q, Forte 게이트 시간(1q 130 µs, 2q 600 µs)으로 **35.84 ms/shot**.

| | 예측 1건 |
|---|---|
| 고전 matched MLP (2026-08-23 실측, 1스레드) | **35.3 µs** |
| Forte 500 shots 순수 실행 | **17.9 s** = 약 **51만 배** 느림 |
| + 큐 (실측 5분) | 약 **850만 배** |

본 과제는 10 ms 주기 실시간 nowcasting이다. QPU 추론 1건이 **제어 주기 전체의 1,792배**를
쓴다. 게이트가 1000배 빨라져도 18 ms로 여전히 예산 초과다. **실시간 경로에서는 구조적으로
닫혀 있다.** 살아있는 갈래는 오프라인뿐이다.

## 12. 신호 예산 — 이 실험의 핵심 숫자

`ionq_hw_ladder.py`가 실제 val 작동점에서 뽑은 노이즈 없는 ⟨Z₀⟩ 범위는 **[-0.0368, +0.0508]**
(span 0.0876). 체크포인트의 `out_scale = 0.9871`, `CES_TI` 정규화 std = 632.86 eV로 환산하면:

| 항목 | 값 |
|---|---|
| 학습된 회로의 **출력 창 전체** | **54.7 eV** |
| 이겨야 할 persistence baseline RMSE | 449.6 eV |
| 고전 matched MLP 도달점 | 378.0 eV |

**회로가 예측을 움직일 수 있는 폭 전체가 baseline 오차의 12%밖에 안 된다.** 여기에 측정
노이즈가 얹힌다:

| shots | σ(⟨Z₀⟩) | 물리 오차 | 출력 창 대비 |
|---|---|---|---|
| 100 | 0.1000 | 62.5 eV | **114%** (신호가 통째로 잠김) |
| 400 | 0.0500 | 31.2 eV | 57% |
| 2000 | 0.0224 | 14.0 eV | 26% |
| 10000 | 0.0100 | 6.2 eV | 11% |

즉 VQC의 패배는 두 요인의 **곱**이다: 표현력이 부족해 출력 창이 애초에 좁고(54.7 eV),
살 수 있는 정밀도가 그 좁은 창의 절반 이상을 먹는다(400샷에서 57%). 7월 §3의
skill 열세(-0.100 / +0.185 vs MLP +0.293 / +0.363)가 이 두 숫자로 설명된다.

## 13. 추가된 코드

| 파일 | 역할 |
|---|---|
| `ces_prediction/experiments/quantum/ionq_hw_ladder.py` | 샷 사다리(100/200/400) 하드웨어 측정. dry-run 선(先)견적 → `--budget` 하드캡 → 매 job `backend` 검증 → job마다 결과 파일 갱신(중단해도 지불한 결과를 잃지 않음). 기본 타깃은 무료 시뮬레이터, 하드웨어는 `--hardware --yes` 필요 |

**검증 완료 사항** (무료 시뮬레이터, 3샘플):

- PCA 기저 drift **0.000e+00** — 체크포인트와 정확히 일치, 각도가 학습 때와 동일
- **비트 순서는 little-endian** — `z0_little_endian`이 정확한 노이즈 없는 값과 3/3 일치,
  big-endian은 3/3 불일치. IonQ 확률 dict의 정수 키에서 큐빗 0은 **최하위 비트**다.
- PennyLane `Rot(φ,θ,ω)` = `RZ(φ) RY(θ) RZ(ω)`, 단일 레이어 entangler는 range-1 CNOT 링.
  이 번역이 IonQ `qis` 게이트셋에서 정확히 재현됨을 확인.

## 14. 다음에 판단할 기준 (§5 갱신)

7월 §5의 두 관문은 통과했으므로 폐기하고, 남은 판단 기준을 다시 적는다.

1. **마감이 2026-09-30이다.** 잔고가 아니라 이 날짜가 제약이다.
2. **성능 개선 목적이라면 여전히 하지 말 것.** §12의 신호 예산이 7월 §3의 결론을 강화했지
   약화시키지 않았다. 하드웨어는 시뮬레이터에 노이즈만 더한다.
3. **돌릴 가치가 있는 목적은 하나뿐이다** — 실하드웨어에서 실측한 정량적 음성 결과. 그
   목적이면 400샷 티어($25.79/job)로 샷 사다리를 태우는 것이 유일하게 합리적인 설계다.
4. **음성 결과를 뒤집을 측정을 지목하자면**: 샷 사다리에서 편차가 1/√shots로 계속 내려가면
   "돈만 쓰면 되긴 된다"(오프라인 한정)이고, 게이트 오차 바닥에 멈추면 "샷을 무한히 늘려도
   못 넘는다"는 영구 판정이다. 이 두 갈래를 가르는 것이 사다리의 존재 이유다.
