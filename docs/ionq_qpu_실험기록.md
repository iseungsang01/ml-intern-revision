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

> **이 절은 2026-08-23 당일 한 번 고쳐 썼다.** 처음에는 3샘플에서 뽑은 ⟨Z₀⟩ 범위
> [-0.0368, +0.0508]을 근거로 "출력 창 54.7 eV"라고 적었는데, **3샘플은 범위를 심하게
> 과소추정한다.** 아래는 val 1,500점으로 다시 잰 값이다. 결론의 방향이 바뀌므로 옛 숫자를
> 인용하지 말 것.

노이즈 없는 ⟨Z₀⟩를 실제 val 작동점 1,500개에서 계산하면:

| 통계 | ⟨Z₀⟩ | 물리 환산 (`out_scale` 0.9871 × std 632.86 eV) |
|---|---|---|
| 전체 span | 0.7834 | **489.4 eV** |
| p1–p99 span | 0.5784 | **361.3 eV** |
| 1 σ | 0.1160 | 72.5 eV |

비교 대상:

| 항목 | 값 |
|---|---|
| 이겨야 할 persistence baseline RMSE | 449.6 eV |
| 고전 matched MLP 도달점 | 378.0 eV |
| VQC 도달점 (시뮬레이터, 정확 확률) | 471.5 eV |

샷 노이즈를 같은 단위로 얹으면:

| shots | σ(⟨Z₀⟩) | 물리 오차 | p1–p99 창 대비 |
|---|---|---|---|
| 100 | 0.1000 | 62.5 eV | 17% |
| 200 | 0.0707 | 44.2 eV | 12% |
| 400 | 0.0500 | 31.2 eV | **9%** |
| 2000 | 0.0224 | 14.0 eV | 4% |

**정정된 결론: 샷 노이즈는 400샷에서 이미 구속 조건이 아니다(창의 9%).** 처음 쓴 "57%가
잠긴다"는 과소추정된 창에서 나온 수치였다. 회로는 출력을 ±72.5 eV(1σ)만큼 실제로 움직인다.

따라서 **VQC의 패배는 측정 한계가 아니라 거의 순수하게 표현력·정확도 문제다.** 회로는
움직이기는 하는데 **틀린 방향으로** 움직여서, RMSE 471.5 eV로 persistence(449.6 eV)보다도
나쁘다(skill -0.100). 7월 §3의 열세는 "신호가 노이즈에 잠겨서"가 아니라 "함수를 못 배워서"다.

이것은 §11의 지연 판정과 독립적이며, 둘 다 같은 방향을 가리킨다.

### 12.1 하드웨어는 실제로 나쁘다 — 다만 ⟨Z₀⟩ 한 비트로는 안 보인다

⟨Z₀⟩만 보면 하드웨어 편차가 이론 샷 노이즈보다 작게 나와 혼란스럽다(첫 4점 기준
χ²/dof ≈ 0.06). 256개 상태 **분포 전체**를 total variation distance로 재면 이야기가 다르다:

| val | shots | TVD 관측 | TVD 이상적 샘플러 | 비율 |
|---|---|---|---|---|
| 356 | 100 | 0.6887 | 0.4579 | 1.50 |
| 3095 | 100 | 0.7019 | 0.4189 | 1.68 |
| 356 | 400 | 0.4371 | 0.2495 | 1.75 |
| 3095 | 400 | 0.4549 | 0.2299 | **1.98** |

하드웨어 분포는 완전 샘플러보다 **1.5~2.0배 멀고, 샷이 늘수록 비율이 커진다** — 샘플링
노이즈가 줄면서 고정된 하드웨어 오차가 드러나는 계통 오차의 서명이다.

⟨Z₀⟩에서 이게 잘 안 보이는 이유는 **탈분극 계열 오차가 분포를 균등분포로 끌어당기는데,
균등분포의 ⟨Z₀⟩가 정확히 0**이기 때문이다. 우리 작동점은 1σ가 0.116으로 0 근처에 몰려
있어서, 오차가 밀어낼 여지가 애초에 작다. **하드웨어가 좋아서가 아니라 회로가 망가뜨릴
신호를 적게 만들어서 견디는 것**이고, 이는 §12 본문의 결론과 같은 이야기다.

> **측정 설계 교훈:** 단일 관측량(⟨Z₀⟩)의 편차만으로 QPU 품질을 판단하지 말 것.
> 그 관측량의 이상적 값이 노이즈 고정점 근처면 오차에 둔감해 보인다. 분포 수준 지표를
> 함께 볼 것.

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

## 15. 최종 판정 (2026-08-23) — 샷 제한, 바닥 미검출

사다리 36측정(12샘플 × 100/200/400샷, `qpu.forte-1`, **$928.44**) 완료. 전체 기록과 논거는
`THESIS_RESULTS.md` §8ap에 있고, 여기에는 결론만 남긴다.

| shots | n | rms 편차 | 이상적 1/√N | 비율 | 실효 샷 |
|---|---|---|---|---|---|
| 100 | 12 | 0.14899 | 0.10000 | 1.490 | 명목의 45% |
| 200 | 12 | 0.08746 | 0.07071 | 1.237 | 명목의 65% |
| 400 | 12 | 0.07011 | 0.05000 | 1.402 | 명목의 51% |

두 가지가 동시에 참이다.

1. **하드웨어는 이상적 샘플러보다 확실히 나쁘다.** χ² = 68.59 / 36 dof = **1.905
   (p = 8.5e-4)**. 같은 정밀도를 얻으려면 **샷이 2.39배** 필요하다 — 하드웨어 오차가
   샷의 절반가량을 먹는다. §12.1의 TVD 관측(1.5~2.0배)과 일치한다.
2. **그러나 없앨 수 없는 바닥은 검출되지 않았다.** `a = 1.546`, `b = 0`,
   부트스트랩 95% CI **[0, 0.0934] = [0, 58.3] eV**로 0을 포함한다. 비율이 샷에 따라
   커지지 않고 평평하다(1.49 → 1.24 → 1.40) — 고정 오프셋이 아니라 **부풀려진 샷 노이즈**의
   서명이다. **판정: 100~400샷 구간에서 샷 제한.**

> §12.1에서 첫 4점을 보고 "게이트 오차 바닥의 서명"이라고 읽었던 것은 **레벨당 n = 12에서
> 살아남지 못했다.** 분석기는 이제 3레벨·레벨당 5샘플 미만이면 판정을 거부한다.

**논문에 쓸 문장은 하나다: 측정 정밀도는 애초에 구속 조건이 아니었다.** 회로의 출력 창은
361.3 eV(§12)인데 400샷 하드웨어 rms는 43.8 eV로 그 **12%**이고, 바닥 상한도 16% 미만이다.
VQC는 **양자 하드웨어가 개입하기 전에** 이미 진다 — 노이즈 없는 시뮬레이터에서 RMSE
471.5 eV로 persistence(449.6 eV)보다 나쁘다. 표현력 문제이지 노이즈 문제가 아니다.

**뒤집을 측정 (§14.4 갱신):** 100~400샷은 1/√N이 커서 바닥이 그 아래 숨는다. 가르는 측정은
**2000샷 debias-on**($168.20/job, 잔여 크레딧 ~$540로 3건 가능)이며, 이상적 샘플링이면
0.0224이므로 ~0.02 이상의 바닥은 숨지 못한다. 다만 이는 *왜* QPU가 제한되는지를 날카롭게 할
뿐, **이 과제의 판정은 바꾸지 않는다.** 성능 축에서 되살리려면 시뮬레이션에서 먼저 파라미터
동수 고전 모델을 이기는 회로족이 필요하고, §8에 그런 증거는 없다.

## 16. λ 측정 (2026-08-24) — §12.1의 추론이 확인됐고, §15의 판정이 수정된다

18점 × 400샷, `qpu.forte-1`, **$464.22** (`ionq_depolarizing.py`). 전체 논거는
`THESIS_RESULTS.md` §8ap Addendum 2에 있다.

| 항목 | 값 |
|---|---|
| 기울기 (1 − λ) | **0.3386 ± 0.0396** |
| **λ** | **0.6614 ± 0.0396**, **16.7 σ** |
| 절편 | +0.0360 (탈분극 예측은 0) |
| 잔차 σ | 0.0672 (샷 노이즈 0.0500) |

**Forte는 회로가 의도한 신호 진폭의 34%만 돌려준다.** §12.1에서 "탈분극이 분포를 균등분포로
끌어당기는데 균등분포의 ⟨Z₀⟩가 0이라 0 근처 작동점은 멀쩡해 보인다"고 **추론**했던 것이
측정으로 확인됐고, 크기가 크다.

**그리고 §15의 "바닥 미검출"을 수정해야 한다.** 수축은 샷과 무관하므로 사다리 모형에서
`a`가 아니라 `b`에 들어가야 한다. 사다리의 작동점들에서 이 수축이 만드는 샷-무관 편차는
rms **0.0689**이고, 이는 사다리 자신의 CI [0, 0.0934] 안 **74% 지점**이다. `b`를 거기
고정하고 다시 맞추면 `a`가 1.546에서 **1.202**로 내려가 순수 샘플링 이론값 1.0에 가까워진다.

> **"바닥 미검출"은 검정력 부족이었지 부재의 증거가 아니었다.** 바닥은 실재하고 `b ≈ 0.069`이며,
> 정체는 무작위 게이트 노이즈가 아니라 **곱셈적 진폭 손실**이다.

**논문에 미치는 영향.** 하드웨어의 지배적 효과가 **곱셈적**이라는 것은 덧셈적 그림보다 나쁘다.
회로의 출력 창 361.3 eV가 노이즈가 얹히기 **전에** 이미 **122.3 eV**로 줄고, 따라서 400샷의
덧셈 노이즈 43.8 eV는 이상적 창의 12%가 아니라 **살아남은 창의 36%**다. 수축된 창은 이겨야 할
persistence RMSE의 27%에 불과하다.

다만 **최상위 판정은 그대로다.** 그것은 애초에 하드웨어 노이즈에 기대지 않았다 — VQC는
노이즈 **없는** 시뮬레이터에서 이미 진다(471.5 vs 449.6 eV). "측정 정밀도는 구속 조건이
아니었다"는 문장은 이제 이렇게 읽어야 한다: *정밀도가 아니라 진폭이 구속 조건이었고, 회로가
하드웨어 이전에 지는 한 둘 다 중요하지 않다.*

**방법 교훈.** 이 절에서 좁은 구간의 2-파라미터 적합이 자신 있는 오답을 낸 것이 두 번째다
(4점·2레벨에서 한 번, 18점·3레벨이지만 ⟨Z₀⟩ 폭 0.39에서 한 번). 두 번 다 같은 설정에서
표본을 늘리는 게 아니라 **독립변수의 폭을 넓혀서** 풀렸다. 샷 의존 항과 샷 무관 항을 갈라야
할 때는 n보다 **span을 먼저** 사라.

## 17. 남은 QML 계열 전수 검사 (2026-08-24, 전부 무료)

§8ap이 "성능 축에서 되살리려면 시뮬레이션에서 먼저 이겨야 한다"는 조건을 걸어놓고 변분 회로
하나만 시험했으므로, 나머지 계열을 전부 돌렸다. 러너 `experiments/quantum/qfamilies_probe.py`,
`qfeature_probe.py`. **비용 $0.** 모든 팔은 §8z 분해(`anchor + 보정`)에서 같은 val 행으로
채점하고, **폭과 학습 예산이 같은 고전 대조군**과 짝지었다. 양자 팔은 *자기 대조군*을 이겨야
의미가 있다.

| 계열 | 양자 | 고전 대조군 | 판정 |
|---|---|---|---|
| 고정 특징맵 (K=8) | +0.0029 | 무작위 맵 +0.0095 | 닫힘 |
| 저수지 (시간 기억) | +0.0002 | ESN +0.0099 | 닫힘 |
| 커널 릿지 | +0.3279 | RBF **+0.3307** | 닫힘 (이점 없음) |
| 학습된 인코더 | −0.0047 | +0.0261 | **미결** |

**학습된 인코더 팔은 판정하지 않는다.** 대조군이 +0.0261인데 같은 입력에서 학습된 MLP는
+0.365를 낸다 — 자기 천장의 1/14이다. 설정(12 epoch·lr 0.02·잔차·zero-init·K=8 병목)이
수렴하지 않았고, 두 약한 모델끼리의 비교라 검정력이 없다.

### 17.1 왜 좁은 팔에서만 지는가 — 짝함수 진단

기저를 짝/홀 성분으로 분해해 측정했다.

```
입력에 대해 EVEN(부호 무시)인 에너지     양자 1층 84.2% · 2층 52.2% · 4층 51.2%   고전 tanh 16.9%
순수 ODD 타깃(sum z) 선형 프로브 R^2     양자 2층 0.2076                          고전 tanh 0.8780
```

`RY(θ)|0⟩`을 Z로 읽으면 `⟨Z⟩ = cos θ`이고 cos은 짝함수라, **부호 정보가 1차에서 사라진다.**
우리 타깃은 `y − anchor`, 즉 부호가 본질인 보정이다. 층을 늘려도 51%에서 멈추므로 깊이로는
못 고친다. 적도 시작(H 먼저)도 시험했으나 개선되지 않았다(1층 71.6% / 2층 61.3%) — 첫 얽힘
층이 지나면 초기 상태의 이점이 씻긴다. **고칠 방법을 안다고 주장하지 않는다.**

이 설명은 스스로 검증된다. **커널 팔만 비긴 이유**가 여기 있다 — 커널 릿지는 훈련점 900개를
기저로 쓰므로 개별 기저 함수의 모양이 중요하지 않다. 반대로 K=8짜리 좁은 팔에서는 기저 모양이
전부다. 즉 진 이유는 "양자라서"가 아니라 **"좁은 기저를 쓸 때 그 기저가 타깃과 안 맞아서"**다.
기저를 넓히면 격차가 사라지지만, 그때는 하드웨어에서 O(N²) 회로가 필요해 회로당 $25.79로
논외가 된다.

### 17.2 순서를 거꾸로 밟았다

$1,418은 **하드웨어 특성**을 샀지 성능 답을 사지 않았다. 성능 답은 처음부터 시뮬레이터에
무료로 있었고, 이 절의 결과 전부가 $0이다. **무료 팔을 먼저 돌려라** — 같은 규칙이 이 세션에서
두 번 값을 했다(16큐빗 8시간 학습을 고전 대조군으로 8초 만에 기각, 계열 둘을 돈 없이 폐쇄).

> **본실험과의 분리.** 이 절의 어떤 수치도 본 파이프라인 주장에 인용되지 않는다. 다만 대조군
> 쪽에서 관찰된 것 하나는 기록해 둔다 — `compare_baselines.py`의 팔(persistence · linear ·
> pchip · ar_local · gp_causal)은 **전부 과거 CES만 본다**. 진단 신호를 입력으로 받는
> 비신경망 팔이 없다. 여기서 잰 RBF 릿지(+0.3307, 900행)는 **다른 프로토콜**(W=4 평탄화 ·
> val · PCA-8)이라 백본 수치와 비교할 수 없고, "이 팔이 약하지 않다"는 관찰까지만 유효하다.
> 본실험에 반영할지는 별도 판단이며 이 문서는 아무것도 바꾸지 않는다.

---

# 부록 A — THESIS §8ap에서 옮겨온 상세 기록 (2026-08-24)

본실험 기록(`THESIS_RESULTS.md`)에는 요약과 포인터만 남기고, 아래 상세를 이 문서로 옮겼다.
양자 갈래는 본 파이프라인과 코드·데이터·프로토콜 어느 쪽으로도 연결되지 않는다.

## 8ap. The quantum arm reaches real hardware and comes back shot-limited (2026-08-23) — a QPU cannot fail this task interestingly, because the circuit fails it first

**Status: closed, negative, and now hardware-verified.** The July attempt
(`docs/ionq_qpu_실험기록.md`) could not reach a QPU at all — every Forte target was
`unavailable` and jobs were silently downgraded to the ideal simulator. Both gates it left
behind passed on 2026-08-23, so the measurement it asked for finally ran.

### Design

The July experiment already settled accuracy under a matched comparison: a variational quantum
circuit (98 params, 8 qubits, 4 data-re-uploading layers on a train-only PCA 92→8 projection)
lost to a classical MLP with 101 params on byte-identical reduced inputs, across an identical
5-point lr sweep on both sides. Hardware can only add noise to that, so re-running for
*performance* was never the point.

What hardware can answer that a simulator cannot: **is the error that a QPU adds a shrinking
sampling error, or a fixed floor?** A sampled circuit estimates `<Z>` with error ~1/√N, so the
task's effect size sets a shot floor. Gate and readout error add a term more shots cannot
remove. Which dominates decides whether a QPU could *ever* resolve this task.

So: the trained circuit, evaluated on 12 real val operating points at three shot counts, with
the deviation from the exact noiseless `<Z_0>` fitted to

    rms(N)² = a²/N + b²

`a` is the sampling coefficient (prediction ≈ 1); `b` is the irreducible floor. Runner
`experiments/quantum/ionq_hw_ladder.py`, analysis `analyze_hw_ladder.py`, 36 measurements on
`qpu.forte-1`, **$928.44** of the KISTI trial credit.

The ladder stays at 100/200/400 shots because IonQ's billing is a two-tier flat fee that
ignores circuit size — **$25.79 up to 400 shots, $168.20 from 500** (debiasing forced on, 32
variants, not disableable), measured by free `dry_run` submissions. The low tier buys 7× more
jobs per dollar, and one constant error-mitigation setting across the ladder.

### Result

| shots | n | rms deviation | ideal 1/√N | ratio | effective shots | rms in eV |
|---|---|---|---|---|---|---|
| 100 | 12 | 0.14899 | 0.10000 | 1.490 | 45% of nominal | 93.1 |
| 200 | 12 | 0.08746 | 0.07071 | 1.237 | 65% of nominal | 54.6 |
| 400 | 12 | 0.07011 | 0.05000 | 1.402 | 51% of nominal | 43.8 |

Two things are true at once.

**The hardware is measurably worse than an ideal sampler.** χ² = 68.59 on 36 dof
(**χ²/dof = 1.905, p = 8.5e-4**) against the ideal-sampling null. Forte delivers roughly the
precision of a perfect sampler given **half** the shots; matching an ideal sampler costs
**2.39× the shots**. An independent distribution-level check agrees — total variation distance
over all 256 basis states sits 1.5–2.0× further from ideal than a perfect sampler.

**But no irreducible floor is resolved.** The fit returns `a = 1.546`, `b = 0.00000`, with a
bootstrap 95% CI for the floor of **[0, 0.0934] = [0, 58.3] eV**, which reaches zero. The
ratio column is flat across the ladder (1.49, 1.24, 1.40) rather than growing, which is the
signature of *inflated sampling noise*, not a fixed offset. **Verdict: shot-limited over
100–400 shots** — **corrected by Addendum 2 below**, which resolves the floor this fit could
not and shows the excess is a multiplicative contraction rather than inflated sampling.

> An earlier read of the first four measurements suggested the opposite — a gate-error floor —
> because the ratio appeared to grow with shots. It did not survive n = 12 per level. The
> analysis script now refuses a verdict below 3 shot levels and 5 samples per level, because a
> two-parameter quadrature fit to two levels is exactly determined and will report a confident
> `a` and `b` from pure noise.

### What this means for the thesis

**Measurement precision was never the binding constraint.** The circuit's own output window is
**361.3 eV** (p1–p99 of the noiseless `<Z_0>` over 1,500 val points, through `out_scale` and
the `CES_TI` normalisation). Hardware rms at 400 shots is 43.8 eV — **12% of that window** —
and the floor's upper bound is under 16% of it.

The failure is upstream of the hardware entirely:

| | RMSE (eV) |
|---|---|
| persistence baseline | 449.6 |
| classical matched MLP | 378.0 |
| **VQC, noiseless simulator, exact probabilities** | **471.5** |

The VQC is worse than doing nothing **before any quantum hardware is involved**. It moves its
output — ±72.5 eV at 1σ — but in the wrong direction. This is an expressivity result, not a
noise result, and hardware access did not change it.

With the latency arithmetic below, the arm is closed on two independent axes.

### Latency — closed structurally, not marginally

One prediction: classical matched MLP **35.3 µs** (measured, single thread) against Forte
**7.4 s at 100 shots / 22.9 s at 400** (measured `execution_duration_ms`). The task is 10 ms
nowcasting, so one QPU inference costs ~1,800× the entire control cycle. A 1000× faster gate
set still overruns it. **The real-time path is structurally closed**; only an offline use
survives, and offline is precisely where the classical baselines are strongest (§8-GP).

### Addendum (2026-08-24) — scaling the circuit is closed too, and it cost 8 seconds to find out

The obvious objection to all of the above is that the circuit is a toy: 8 qubits, 4 layers, and
a PCA that throws away 12% of the input variance. So scale it up. A 16-qubit / 12-layer circuit
(578 params, PCA variance 0.956) was set up to train — ~20 min/epoch on CPU with
`lightning.qubit` + adjoint, about 8 hours for 25 epochs.

It was not run, because the premise can be tested on the classical control in seconds. If more
input dimensions help at a fixed parameter budget, the classical model will show it first.

At a matched ~578-parameter budget, on the same val samples:

| PCA dim | variance kept | MLP hidden | RMSE (eV) | skill |
|---|---|---|---|---|
| **8** | 0.873 | 58 | **419.4** | **+0.3625** |
| 12 | 0.922 | 41 | 432.9 | +0.3208 |
| 16 | 0.956 | 32 | 437.8 | +0.3054 |
| 20 | 0.981 | 26 | 453.9 | +0.2534 |
| 24 | 0.991 | 22 | 455.1 | +0.2496 |
| 32 | 0.997 | 17 | 452.4 | +0.2585 |
| 92 (no PCA) | 1.000 | 6 | 431.2 | +0.3264 |

**More input dimensions make it monotonically worse.** At a fixed budget the hidden width has to
shrink to pay for the wider input (58 → 32 → 22 → 6), and training loss falls the whole way
(0.774 → 0.636) while validation skill falls with it. Ordinary overfitting. **The PCA-8
bottleneck does not bind** — 8 dimensions is the best choice at this budget, so a 16-qubit
circuit would have been trained on a worse representation, not a richer one.

The parameter axis is equally flat: the July 101-param MLP scored +0.3634 and this 578-param
MLP scores +0.3625. **5.7× the parameters buys nothing.**

So both axes along which a variational circuit could be "scaled up" — wider encoding (qubits)
and more parameters (layers) — are already saturated or actively harmful for the *classical*
model on this task. There is no headroom for a bigger circuit to exploit. This closes the
scaling objection without spending either credit or a CPU night, and it is the cheapest result
in this section by a wide margin.

### Addendum 2 (2026-08-24) — the floor was real; the ladder just could not resolve it

The ladder above reported `b = 0` with a bootstrap CI of [0, 0.0934] and absorbed the excess
into `a = 1.546`, reading it as inflated sampling noise. A second experiment with 1.7× the
`<Z_0>` span settles it, and the reading changes.

**Design.** A depolarizing channel is a pure contraction, `<Z_0>_measured = (1 - λ)·<Z_0>_exact`,
so the slope of measured against exact returns λ directly. Real val inputs span `<Z_0>` in only
[-0.31, +0.48]; optimizing the encoded angles inside the same ±π/2 box the PCA squash produces
reaches [-0.55, +0.77]. Same trained circuit, same depth, so the error budget matches the
ladder. 18 points × 400 shots on `qpu.forte-1`, **$464.22**
(`experiments/quantum/ionq_depolarizing.py`).

**Result.**

| quantity | value |
|---|---|
| slope (1 − λ) | **0.3386 ± 0.0396** |
| **λ** | **0.6614 ± 0.0396**, confirmed at **16.7 σ** |
| intercept | +0.0360 (depolarizing predicts 0) |
| residual σ | 0.0672 against shot noise 0.0500 |

**Forte returns 34% of the circuit's intended signal amplitude.** The §12.1 explanation in the
experiment log — that depolarizing-type error pulls the state toward maximally mixed, whose
`<Z_0>` is exactly 0, so operating points near 0 look unharmed — is confirmed, and it is large.

**This closes the ladder's open question.** A contraction is shot-independent, so it belongs in
`b`, not `a`. At the ladder's own operating points the fitted contraction produces a
shot-independent deviation of rms **0.0689** — which sits at the 74% point of the ladder's own
CI [0, 0.0934]. Refitting the ladder with `b` fixed there drops `a` from 1.546 to **1.202**,
close to the 1.0 that pure sampling predicts. Predicted rms then tracks the observed values
(ratios 1.23 / 0.89 / 0.82 across the three levels) where shot noise alone was short at every
level (1.49 / 1.24 / 1.40).

So the earlier verdict needs correcting: **"no floor resolved" was a power limitation, not
evidence of absence.** The floor is real, it is `b ≈ 0.069`, and its mechanism is not random
gate noise but a multiplicative amplitude loss.

**What changes for the thesis.** The hardware's dominant effect is *multiplicative*, and that
is worse than the additive picture given above. The circuit's 361.3 eV output window contracts
to **122.3 eV** on hardware before any noise is added, so the 43.8 eV of additive noise at 400
shots is **36% of the window that actually survives**, not the 12% computed against the ideal
window. The contracted window is only 27% of the persistence RMSE it would have to beat.

The top-line verdict is unchanged, because it never rested on hardware noise: the VQC loses on
the *noiseless* simulator (471.5 eV against persistence at 449.6). Expressivity is still the
binding failure. But the sentence "measurement precision was never the binding constraint"
should now read: *precision was not the constraint; amplitude was, and neither matters while the
circuit loses before hardware is involved.*

**Method note.** This is the second time in this section that a two-parameter fit over a narrow
range produced a confident wrong reading — first `a`/`b` from two shot levels on four points,
then `a`/`b` from three levels over a `<Z_0>` span of 0.39. Both were fixed by widening the
independent variable, not by adding samples at the same settings. When a fit has to separate a
shot-dependent term from a shot-independent one, buy span before buying n.

### The measurement that would reopen it

Per the standing rule that a negative result must name its own reversal: the 100–400 shot range
cannot separate `a` from `b` well, because 1/√N is still large there and a floor hides under it.
The discriminating measurement is **2000-shot debiased jobs** ($168.20 each; ~$540 of credit
remains, so three are affordable), where ideal sampling would give 0.0224 and any floor above
~0.02 would dominate instead of hiding.

That test would sharpen *why* the QPU is limited. It would not change the verdict on this task,
because the VQC already loses on the noiseless simulator by 21.9 eV against persistence. To
reopen the arm on *performance* one would need a circuit family that beats a
parameter-matched classical model in simulation first — and §8 has no evidence such a family
exists here.

### Addendum 3 (2026-08-24) — the pre-verification §8ap demanded, run for free before any spending

§8ap set the condition for reopening the quantum arm on performance: *a circuit family must beat
a parameter-matched classical model in simulation first.* Only one family had ever been tested —
the variational circuit (VQC), where the circuit **is** the model and its parameters are trained.
A second family exists and is a better structural fit, so it was checked before any credit moved.
Runner `experiments/quantum/qfeature_probe.py`; cost **$0**.

**Why a feature map is the better candidate.** §8z established that the backbone's `T_i` skill
decomposes exactly as `anchor + Σ w_k z_k + b` with `z` bounded in [-1, 1] and K = 8. Quantum
expectation values are natively bounded in [-1, 1], so they slot into that shape — and because
the readout is linear it is fitted in closed form by ridge, which removes the parameter-shift
wall that made VQC training impossible on hardware. The circuit is fixed, so nothing is trained
on the device at all.

**The control that decides it** is a classical *random* feature map of the same width on the
same inputs through the same readout. Without it the experiment only measures "does having K
features help". Same rows, same persistence anchor, same ridge sweep, same metric.

| arm | K = 8, 2 layers | K = 12, 3 layers |
|---|---|---|
| persistence anchor alone | +0.0000 | +0.0000 |
| **quantum feature map** | **+0.0029** | −0.0006 |
| classical random map | +0.0095 | −0.0002 |
| *trained* MLP, same inputs, W = 2 | **+0.126** | — |

**Two findings, and the second is the important one.**

The quantum map never leads: it loses to a classical random map of identical width at K = 8, and
both collapse to the anchor by K = 12. §8ap's condition is not met by this family either.

But the dominant effect is not quantum-vs-classical — it is **frozen-vs-trained**. The best fixed
map recovers +0.0095 where a trained encoder on the same inputs reaches +0.126: **freezing the
map costs ~92% of the achievable skill.** That is fatal to the whole appeal of this route, because
"no gradients needed" was precisely its selling point over the VQC. The structural analogy to
§8z was only half right — the shape (bounded latents into a linear readout) matches, but §8z's
encoder is *learned*, and that turns out to be where the skill lives.

**What this closes and what it leaves.** Two of the three QML families are now closed on this
task: variational circuits (§8ap, hardware-verified) and fixed feature maps (here, free). The
third — quantum reservoir computing over the sequence — differs only in that the fixed map
carries memory across timesteps. It is *predicted* dead by the same measurement, since the
useful state §8z identifies (carried `T_i` plus a `T_e`-like proxy) is already supplied by the
anchor, but it has not been measured and should be labelled as untested rather than closed.

**Method note.** This is what §8ap's reopening condition is for, and it cost nothing. The
$1,418 already spent bought hardware characterisation, not a performance answer; the performance
answer was always available in simulation for free. Run the free arm first.

### Provenance

Checkpoint is the July `quantum_vqc_weights.pt` (`window_size = 4`, the pre-reset protocol).
That is deliberate and does not contaminate the confirmed W = 2 results: the deviation physics
measured here is a property of the circuit and the device, not of the data protocol, and no
number in this section is quoted alongside the backbone's skill figures. The PCA basis is
re-fit train-only and cross-checked against the checkpoint (drift **0.000e+00**).

Bit ordering was validated before spending: IonQ's integer probability keys are **little-endian**
(qubit 0 = least significant bit), matching the exact noiseless value 3/3 on the free cloud
simulator where big-endian matched 0/3.
