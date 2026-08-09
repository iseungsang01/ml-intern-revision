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
