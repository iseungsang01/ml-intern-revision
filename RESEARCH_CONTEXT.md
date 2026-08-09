# RESEARCH_CONTEXT — 현재 연구의 전체 컨텍스트

이 문서는 **이 저장소의 연구 전체를 한 파일로 인수인계**하기 위한 다이제스트다. 처음 합류한 사람
(또는 새 세션의 에이전트)이 이 파일만 읽고 "무엇을 왜 하고 있고, 무엇이 확정되었으며, 무엇을
건드리면 안 되고, 다음에 뭘 해야 하는지"를 알 수 있도록 쓰였다.

**출처와 갱신 규칙.** 이 파일은 **파생 문서**이며 원본이 아니다. 충돌하면 항상 아래가 이긴다.

| 원본 | 역할 |
|---|---|
| `THESIS_RESULTS.md` | 결과 원장. §8a–§8u가 통제 실험 1건당 1절(설계·결과·판정) |
| `PROJECT_KNOWLEDGE.md` | 장기 기억. 지속되는 교훈, 폐기된 경로, 재현성 함정 |
| `CLAUDE.md` | 작업 규약. 명령, 데이터/모델 계약, 합의 사항 |
| `docs/paper/paper_numbers.json` | **논문에 인용되는 모든 수치의 단일 출처** (수기 전사 금지) |
| `ces_prediction/experiments/README.md` | 배치 ↔ §8 절 ↔ 판정 매핑, 새 배치의 비협상 조건 |

새 실험을 하면 **§8에 절을 추가**하고, 지속되는 교훈만 `PROJECT_KNOWLEDGE.md`로 올린 뒤,
이 파일의 해당 줄을 갱신한다. 이 파일에 새 사실을 **처음** 기록하지 않는다.

---

## 1. 한 문단 요약

KSTAR 토카막에서 **CES(Charge Exchange Spectroscopy)** 는 이온 온도 `CES_TI`와 토로이달 회전
`CES_VT`를 준다. 그런데 이 신호는 10 ms 격자 위에서 **성기고 결측이 많다**. 이 연구는 같은
시각의 **빠른 진단(BES 밀도요동 / ECEI 전자온도 / MC Mirnov 자기요동)** 과 **과거 CES 이력**만으로
그 시점의 CES 값을 채우는 **인과적 nowcasting 모델**을 만들고, 이를 **미래 값까지 쓰는 오프라인
보간(linear / PCHIP / local AR / GP)** 이라는 일부러 불리한 잣대에 붙였다. 결론: **`CES_TI`는
미래를 보는 보간을 4개 독립 split에서 모두 유의하게 이긴다**. **`CES_VT`는 전역적으로는 비긴다
(n.s.)** — 그리고 이 비대칭(`T_i` ↔ `V_rot`)이 물리적으로 예측 가능하고 ablation으로 확인된
**과학적 발견 자체**다. 인과 기준선(persistence/AR) 대비로는 두 타겟 모두 크게 이긴다.

---

## 2. 문제 정의와 주장

**질문.** 10 ms 타겟 시점에서, 동시각 빠른 진단 + 과거 CES 이력만으로 CES 정보를 복원할 수 있는가 —
**CES 자체의 시간 보간이 복원할 수 있는 것 이상으로**?

**잣대(deliberately-hard bar).** 기준선은 타겟 주변의 **과거+미래** CES를 모두 쓰는 오프라인 보간이다.
모델은 미래를 보지 못한다. 미래를 보는 상대를 이기면, 어떤 인과적 방법이든 a fortiori 이긴다.

**주장 (반드시 두 개로 분리해서 서술).**
1. **오프라인 주장** — "미래를 쓰는 보간을 이긴다": **관측된 점 집단**에 대한 진술. `CES_TI`만 성립.
2. **인과/배치 주장** — "배치 가능한 모든 인과적 방법을 이긴다": **실제로 결측인 점**으로 재가중해도
   살아남는 진술 (§8i: persistence 대비 4/4 +0.29). 온라인 가상센서의 경쟁자는 persistence이지
   미래를 읽는 보간이 아니다.

이 둘을 섞는 것이 이 결과를 과대판매하는 유일한 경로다. **절대 섞지 않는다.**

**프레이밍.** 이 연구는 "super-resolution"이 아니라 **10 ms gap-filling nowcasting**이다.

---

## 3. 데이터

### 3.1 원자료
- KSTAR 방전 **641 shot** CSV (`s30801.csv` … `s32751.csv`), 저장소 `data/` 에 로컬 존재.
  **git 추적 안 함**(`data/*` gitignored). 다른 위치를 쓰려면 `CES_DATA_DIR`.
- 컬럼 18개: `time`, `CES_TI`, `CES_VT`, `BES_*` ×9, `ECEI_*` ×4, `MC*` ×2.
  컬럼은 헤더 prefix로 **자동 추론**된다.
- 10 ms 격자 총 **247,207 행**.

### 3.2 결측 장부 (재생성: `python ces_prediction/analyze_data_evidence.py`)

| | `CES_TI` | `CES_VT` |
|---|---:|---:|
| ① NaN 결측 | 20,216 (8.2%) | 59,107 (23.9%) |
| ② held / forward-fill (직전 관측과 bit-identical) | 1 (0.0%) | 101,604 (41.1%) |
| **실질 결측 ①+②** | **20,217 (8.2%)** | **160,711 (65.0%)** |
| 관측값 중 held 비율 | 0.0% | **54.0%** |
| held run 포함 shot 파일 | 1 / 641 | 499 / 641 |
| held run 길이 (중앙/최대) | 2 / 2 행 | 10 / **1,214** 행 |

> **"`CES_VT`는 24% 결측"을 단독으로 인용하지 말 것.** 독립 정보가 없는 격자 비율은 **65.0%** 이다.

### 3.3 데이터에 박혀 있는 세 가지 인공물
1. **held(forward-fill) 값** — 계측기 padding. 학습 타겟과 `ces_history` 양쪽을 오염시켜
   "이력 복사가 최적"이라고 가르친다 (§8c에서 제거가 이득임을 4/4 seed로 확인).
2. **CES 스펙트럼 fit 실패** — `CES_TI` > 3 keV 행(0.4–0.6%, 최대 14,984 eV)은 물리가 아니라 피팅 실패.
   전역 p99 = 2,089 eV. 이 행들을 빼면 skill이 대략 2배가 된다 → **헤드라인은 보수적**(§8q).
3. **Mirnov(MC) 앨리어싱** — kHz `dB/dt`를 anti-aliasing 필터 없이 100 Hz로 데시메이트.
   블록 내 lag-1 자기상관: BES **+0.568** / ECEI **+0.572** / **MC −0.009** (|r|<0.1이 82%).
   정보가 모델에 오기 전에 이미 파괴되었으므로 **파생 MC 특징으로는 복구 불가**(재시도 금지).

### 3.4 데이터에 **없는** 것 — `CES_VT`의 진짜 레버
**NBI 토크 채널이 데이터셋에 아예 없다.** `T_e`(ECEI 평균)로 대리하려는 시도는 데이터가 반증한다:
shot 간 `T_e`~`CES_TI` r = **+0.353** (p=2.9e−17) vs `T_e`~`CES_VT` r = **+0.024** (p=0.58).
**power ≠ torque.** 회전은 빔 에너지·접선반경·주입 기하와 운동량 수송/edge braking이 지배한다.

---

## 4. 데이터/모델 계약 (불변식 — 승인 없이 깨지 말 것)

`model.py`, `train.py`, `dataset.py`가 모두 의존하고 `tests/test_architecture.py`가 일부를 검증한다.

- `model.forward(self, bes, ecei, mc, time_features=None, ces_history=None)` — 시그니처 고정.
- 출력은 정규화된 `[CES_TI, CES_VT]`, shape `(batch, 2)`. **`model.py` 안에서 역정규화 금지.**
- BES/ECEI/MC/타겟은 **train 파일에서만** 구한 통계로 채널별 z-score. 타겟 통계는 **NaN-aware**.
- `ces_history` = `(batch, window, 4)`: 이전 정규화 `CES_TI`, 이전 정규화 `CES_VT`,
  `CES_TI` 관측 flag, `CES_VT` 관측 flag. 두 타겟은 **독립적으로** 결측되므로 관측을 타겟별로 추적.
- **타겟 시점은 값·flag 모두 0으로 완전 마스킹** (누출 방지).
- 샘플마다 `target_mask` `(batch, 2)`. 학습/검증은 **타겟별 masked MSE** — 한쪽만 관측된 행도
  그쪽은 지도한다 (예전 코드는 양쪽 모두 요구해서 라벨 행의 ≈28%를 조용히 버렸다).
- time features 4채널: lookback 초, delta 초, `log1p` lookback, `log1p` delta.

---

## 5. 아키텍처와 배선

### 5.1 논문 모델 (`ces_prediction/model_iter009.py`, 201,258 params)
- 후기 융합(late fusion): 각 진단 스트림은 융합 전까지 분리.
- **관측 마스킹 multi-head attention pooling** — 각 타겟의 이력 readout이 **그 타겟이 실제로
  측정된** 시점에만 가중치를 줄 수 있다. 보간을 강하게 만드는 귀납편향을 파라미터 0으로 추가.
- **타겟 인지 라우팅** — `CES_TI`는 빠른 진단+이력+시간을, **`CES_VT`는 이력+시간만** 본다.
  물리적으로 0인 관측 함수는 **head가 아니라 encoder에서** 막아야 한다 (§8t).
- iter2 "before" 모델은 `ces_prediction/model_iter002.py` (GRU + 마지막 은닉상태 readout).
- **두 파일 모두 SHA-256으로 핀 고정되어 있다. 편집 금지.**

### 5.2 배선 규칙 (2026-08-09 정리, §8u)
- `ces_prediction/model.py`는 `model_iter009.py`를 **재수출**한다. 따라서 `train.py` /
  `evaluate.py` / `compare_baselines.py` / 모든 러너가 기본으로 논문 아키텍처를 쓰고,
  저장된 체크포인트가 그대로 로드된다(과거엔 0/45 → 지금 10/10).
- 아키텍처를 바꾸는 실험은 **`CES_MODEL_FILE`** 환경변수로 자기 `.py`를 가리킨다
  (`experiments/anchor/`가 참조 예시). **추적 소스를 덮어쓰는 러너를 다시 만들지 말 것.**

### 5.3 파이프라인 흐름
```
data/*.csv → KSTAR_CES_Dataset(윈도우·temporal subset·시간특징·이력 마스킹)
          → 파일(shot) 단위 고정 split → train 파일만으로 정규화 통계
          → MultimodalCESPredictor → train.py → metrics.json + weights/
```
비자명한 동작: split은 **행이 아니라 파일 단위**(인접 행 누출 방지)이고 `data/splits/*.csv`에 고정된다.
데이터셋 디스크 캐시는 `data/.ces_cache/*.npz`(윈도우·증강·컬럼·파일 mtime 해시로 자동 무효화).
temporal subset 증강은 조합 폭발이라 샘플 상한이 필수다.

---

## 6. 평가 프로토콜

- **3-way 파일 단위 split** (train/val/**test**). TEST는 아키텍처 탐색 전에 격리 — 선택은 val에서만.
  seed 42/1/7/123, seed 1/7/123은 모델 선택에 **한 번도 쓰이지 않았다**.
- **지표**: 물리 단위 타겟별 RMSE + Murphy skill `skill = 1 − MSE_model / MSE_baseline`.
- **사전등록(PR)**:
  - **PR1** 헤드라인 기준선 = **PCHIP** (ELM/sawtooth 강건성 이유로 사전 선택). 전체 사다리도 보고.
  - **PR2** 보간은 모델이 채점되는 모든 관측 타겟에서 예측. 미래 이웃이 없으면 persistence로 폴백
    (≥0.5 s 간극은 보간 거부) → 어느 arm도 표본이 얇아지지 않음.
  - **PR3** TEST 사전 격리, 하한 ≥15 shot & ≥3,000 관측 `CES_TI` 샘플.
  - **PR4** **shot-clustered paired bootstrap** (B=10,000, seed 12345). shot이 재표집 단위
    (한 방전 내 인접 행은 자기상관). 95% CI가 0을 모델 쪽으로 배제 = **PASS**.
- **두 가지 평가 처리(treatment)를 항상 함께 보고**:
  - `genuine` = held 제외 (**헤드라인**),
  - `stuck0` = held 포함 (역사적 관행, `CES_VT` RMSE를 35–55% 깎아내림).
  `paper_numbers.json`이 둘 다 담고 있는 이유가 이것이다 — 하나를 인용하며 다른 하나를 주장하는
  오류가 실제로 발견된 적이 있다(§8h).

---

## 7. 핵심 결과

### 7.1 헤드라인 (held-out TEST, `skill_vs_pchip`, genuine 평가)

| target | seed 42 | seed 1 | seed 7 | seed 123 | 평균 | PR4 |
|---|---:|---:|---:|---:|---:|---|
| **`CES_TI`** | **+0.179** | **+0.197** | **+0.280** | **+0.263** | **+0.230** | **4/4 PASS** |
| `CES_VT` | +0.203 | +0.162 | +0.100 | +0.183 | +0.162 | 1/4 (seed 1만) |

(`stuck0` 처리에서는 `CES_TI` +0.257/+0.194/+0.263/+0.280, 역시 4/4 PASS.)
기준선을 더 강한 `linear`로 바꿔도 결론 유지 (`stuck0` 4/4, `genuine` 3/4).

### 7.2 RMSE 사다리 (seed 42, 물리 단위, stuck0)

| arm | 접근 권한 | `CES_TI` RMSE | `CES_VT` RMSE |
|---|---|---:|---:|
| **모델 (nowcaster)** | 빠른 진단 + 과거 CES | **372.32** | **22.53** |
| linear 보간 | 과거+미래 | 422.66 | 24.01 |
| PCHIP 보간 *(헤드라인)* | 과거+미래 | 431.81 | 24.49 |
| persistence | 마지막 관측값 | 487.31 | 27.77 |
| AR (local) | 과거만 | 1005.66 | 57.23 |

모델이 **두 타겟 모두 사다리 최저 RMSE**. (genuine 평가에서 `CES_VT` RMSE는 ~35–47이 실제 값.)

### 7.3 비대칭의 메커니즘 — 입력 modality ablation (val, `skill_vs_persistence`)

| ablation | `CES_TI` | `CES_VT` |
|---|---:|---:|
| Full (이력+빠른진단+시간) | +0.428 | +0.296 |
| `no_fast` (이력만) | +0.458 | +0.295 |
| `no_history` (빠른진단만) | **+0.372** | **−0.642** |

빠른 진단만으로도 `CES_TI`는 persistence를 크게 이긴다(충돌 e–i 결합: ECEI가 `T_e`, BES가 `n_e`).
같은 모델이 `CES_VT`에서는 persistence보다 **나쁘다** — 회전 정보는 사실상 전부 과거 CES 이력에서 온다.

### 7.4 skill이 어디에 사는가
- **고변동("peak") 이웃에 집중** (입력만으로 정의된 비순환 프록시). val: `CES_TI` 전역 +0.272 →
  peak **+0.702** (PASS), `CES_VT` +0.131 → **+0.438** (PASS).
- **간극(Δt)별**: 4 split을 pooling하면 `CES_TI`는 Δt>15 ms 전체에서 **+0.191 [+0.10,+0.28]** 로
  미래를 쓰는 PCHIP도 이긴다. Δt>105 ms 극단 구간만 PCHIP이 유의하게 앞선다 —
  **넓은 간극은 양방향 보간의 영토**이고, 온라인 시스템에는 애초에 없는 능력이다.
- **인과 기준선 대비**: 모든 해상 가능한 간극에서 두 타겟 모두 persistence를 유의하게 이긴다
  (`CES_TI` +0.407 / +0.388, `CES_VT` +0.368 / +0.309).
- **`CES_VT`는 하나의 판정이 아니라 세 영역**(§8r): genuine·peak +0.55…+0.63 (persistence 대비
  +0.75…+0.82, **4/4 PASS**) / genuine·bulk ≈ 0 / held 행 −48…−411 (PCHIP이 정의상 이전 값을
  정확히 통과하므로 **어떤 인과 방법도 이길 수 없는 구조적 영역**).

### 7.5 가장 강한 상대 — GP
휴면 상태였던 GP arm을 실제로 구현(numpy Matern-3/2+white, 국소 16+16 최근접, 샘플별 grid-ML,
0.94 ms/fit)한 결과 **GP가 PCHIP을 +0.21…+0.28로 이기고**(`CES_TI` 4/4 PASS), **모델 vs GP는 무승부**
(1/4 PASS, 0/4 반대, 평균 ≈ −0.01). → **"모든 오프라인 보간을 이긴다"고 절대 쓰지 말 것.**
정직한 형태: "사전등록 보간(PR1)은 이긴다 / 미래를 쓰는 최강 ML-튜닝 smoother와는 비긴다 /
인과·배치 주장은 영향 없다(GP는 미래 앵커가 필요)."

---

## 8. 통제 실험 대장 (`ces_prediction/experiments/`, §8 절 대응)

| dir | §8 | 질문 | 판정 |
|---|---|---|---|
| `stuckfree/` | 8c | held 값이 *학습*까지 해치는가? | **KEEP** — `CES_VT` 4/4 개선, 3/4 유의. `CES_DROP_STUCK_TARGETS=1`이 새 기본값 |
| `seq/` | 8d, **8t** | 전체 격자 시퀀스 재프레이밍이 윈도우 방식을 이기는가? | **8t `seq_v2`: 4/4 split에서 최고 `CES_TI`, `V_rot` 결손 해소, 학습비용 1/10** |
| `window_sweep/` | 8f | 이력이 얼마나 필요한가? | 과거 관측 **1개면 전부**. W=4는 skill로 정당화 불가, coverage로만 정당화 |
| `largegap/` | 8g | 큰 간극에서 약한가? | 아니다 — 그건 보간의 영토. *인과* 기준선 대비로는 여전히 이김 |
| `mnar/` | 8i | 실제 결측점으로 재가중해도 살아남는가? | 인과 주장 4/4 (+0.29) 생존, 오프라인 주장 1/4 붕괴 |
| `anchor/` | 8k | 복잡도가 무엇을 사는가? | 1,258-param 완전설명 모델이 `CES_TI` 마진의 **31.5%**(`CES_VT` 7%)만 회수 |
| `latency/` | 8l | 실시간 가능한가, 어느 장치에서? | **CPU** batch-1 p99 = **6.4 ms**(10 ms 예산 내). **CUDA가 8× 느림** |
| `uq/` | 8m | 재학습 없이 교정된 구간을 얻을 수 있는가? | split conformal이 두 기준선을 Winkler로 **8/8** 압도. 단 coverage는 **marginal**이지 conditional 아님 |
| `campaign/` | 8n | 엄격한 시간(캠페인) split에서도 되는가? | 오프라인 우위 사망, 인과 우위 생존(+0.22). 원인 측정: BES가 1.22σ 드리프트 vs 타겟 0.115σ |
| `gp/` | 8p | 최강 오프라인 상대와 붙으면? | GP가 PCHIP을 이기고, **모델은 GP와 무승부** |
| `fitfail/` | 8q | fit 실패(>3 keV)가 헤드라인을 부풀리는가? | 반대로 **깎는다** — 제거하면 skill 약 2배. 헤드라인은 보수적 |
| `heldpeak/` | 8r | `CES_VT` peak 결과와 hold 패턴이 얽혔는가? | 얽혔고 **가설과 반대 방향**(peak가 hold-**부자**). `CES_VT`는 세 영역으로 분해 |
| `pershot/` | 8s | shot별 입력 표준화가 캠페인 전이를 고치는가? | **ADOPT** — 캠페인 `CES_TI` 평균 +0.155, 4/4 유의. 헤드라인에선 유의한 손실 0/4 |
| `quantum/` | — | (곁가지) VQC / IonQ QPU 추론 | 탐색용. Forte 전 기종 `unavailable`, 동일조건에서 VQC가 고전 MLP에 완패. 논문 주장 아님 |
| *(제거됨)* `ct/` | 8e | 연속시간 이력 encoder 4종 | **검증된 음성.** 코드는 2026-08-09 제거, **판정은 §8e에 보존** — W=4에서 재시도 금지 |

공유 파일 2개: `runner_common.py`(고정 split/control/env, `run_step`), `paired_model_compare.py`
(shot-clustered paired bootstrap — 두 arm이 같은 행을 같은 순서로 채점했는지 검증하기 전엔 계산 거부).

### 8.1 `seq_v2` — 구조에서 설계를 유도한 결과 (§8t)
이 문제는 **다중 속도 센서 융합 하의 잠재상태 추정**이다(성긴 타겟 회귀가 아니다). 이 동형성이
강제하는 4가지 설계 결정을, 이 저장소는 8개월에 걸쳐 각각 따로 발견했다:

| 동형성이 강제하는 것 | 우리가 발견한 방식 |
|---|---|
| 미관측 시각에도 상태는 존재 → **전체 격자 + 손실 측 마스킹** | §8d |
| hold는 관측이 아니다 | §8c |
| 각 방전은 독립 실현 → **shot별 표준화** | §8s |
| 물리적으로 0인 관측 함수는 **구조적으로 차단** | iter009의 `V_rot` 라우팅 |

넷을 모두 조립한 `seq_v2`(357,570 params, 인과 LSTM 2개)는 **4/4 split에서 최고 `CES_TI`**
(+0.255/+0.208/+0.305/+0.308, 4/4 PASS), §8d의 4/4 유의 `V_rot` 결손을 **0/4로 해소**,
학습은 **seed당 1.2–1.4분** (윈도우 파이프라인 12–22분). 분해 실험(`seq_v2_nops`)이 원인을
특정했다: **`V_rot` 복구는 라우팅이 하고**(4/4→1/4), shot별 표준화가 마지막 seed를 닫는다(1/4→0/4).
방어 가능한 표현은 **"1/10 비용으로 동급이거나 약간 낫다"** 이지 "이긴다"가 아니다
(paired `CES_TI` 유의는 1/4).

---

## 9. 방법론 규칙과 함정 (전부 대가를 치르고 배운 것)

1. **데이터 처리를 매 실행마다 명시적으로 고정하라.** `CES_DROP_STUCK_TARGETS`를 상속/`pop` 하지
   말 것. 조용한 기본값 폴백이 window sweep의 결론을 통째로 틀리게 만들었다(§8f), 그리고
   `anchor/` 러너도 같은 버그를 안고 출발했다.
2. **같은 처리로 학습된 control과 짝지어라.** `.sf_iter009_s*`는 held-free, `.vt_repro_*`는 held-kept.
   그리고 채점된 population이 **행 단위로 일치**하는지 검증하라.
3. **얼린 실행을 재채점할 땐 키를 가산적으로만 추가하고**, 기존 npz 키가 **bit-identical**로
   재현되는지 먼저 확인하라 (§8g, §8i, §8p 모두 그렇게 했다).
   - **단, `se_model`은 예외다** — 이 기계에서 float32 CUDA forward는 세션 간 bit-재현되지 않는다
     (상대 드리프트 중앙값 3e−4). 해법: population 키(shot, dt_ms, is_peak, 기준선 SE)는
     bit-identical을 요구하고, `se_model`은 **bounded-drift**(RMSE < 0.01 물리단위)로 검사하며,
     병합 산출물은 **참조 `se_model`을 유지**한다.
4. **숫자는 산출물에서 나온다. 산문에서 전사하지 않는다.**
   `collect_paper_numbers.py` → `paper_numbers.json` → 그림/표. 그림 스크립트에 리터럴 금지.
5. **체크포인트를 믿기 전에 자기 기록 지표와 대조하라.** `data/.improve_final_out/weights/`는
   기록된 `comparison_metrics.json`을 만든 체크포인트가 **아니다**(`CES_TI`는 재현, `CES_VT`는
   +0.056 vs 기록 +0.161). `.final_out`(iter2)은 멀쩡하다.
6. **평가 루프를 직접 짜지 마라.** `compare_baselines`는 기준선이 정의되지 않는 ~160개 `CES_TI`
   샘플을 버린다. 그 몇 개 fit-실패 행이 RMSE를 13% 흔든다.
7. **`train.py`는 `CES_SPLIT_DIR/split_manifest.json`을 덮어쓴다.** `CES_TEST_FRACTION=0`(기본)이면
   2-way manifest를 써서 3-way split의 `test_files`를 **파괴한다**. 백업하거나 사본을 가리켜라.
8. **Windows 장시간 배치 함정**: MKL의 Intel Fortran 런타임이 콘솔 close 이벤트에서 학습을 죽인다
   (`forrtl: error (200)`, exit 3221225786). `FOR_DISABLE_CONSOLE_CTRL_HANDLER=1`,
   `KMP_HANDLE_SIGNALS=0` 설정 + detached 실행.
9. **한 번에 한 변수만.** 단일 seed 결과는 증거가 아니다 — 4개 독립 split + shot-clustered paired
   bootstrap이 이 저장소의 최소 기준이다.
10. **음성 결과도 반드시 §8에 기록한다.** 기록이 코드보다 오래 남고, 재시도를 막는 것은 기록이다.

---

## 10. 서술 규칙 (결과 절을 쓰기 전에 읽을 것)

- **"정보가 부족하다"는 결론이 아니라 변명이다.** 음성 결과는 **그것을 뒤집을 측정을 함께 지목할 때만**
  자리를 얻는다. 이 저장소가 지목한 세 레버:
  1. 이력의 **깊이가 아니라 도달거리**(§8i: W=4에서 실제 결측 행의 54.1% `CES_TI` / 4.8% `CES_VT`만
     in-domain; §8f: W는 skill이 아니라 coverage를 산다),
  2. **Mirnov 정보는 플라즈마에 없는 게 아니라 전처리가 파괴했다**(§8b.2) → 원본 kHz 스트림에서
     윈도우 RMS / band power / mode number를 뽑는 것이 수리법,
  3. **NBI 토크 채널의 부재**(§8b.3) → `CES_VT`의 최상위 레버.
- **novelty는 부재-프레임 금지.** "선행 연구가 없다"로 시작하지 않는다. 올바른 순서:
  **계보 인정 → 3축 확장(전자→이온 타겟 / 동시각 memory-less→인과 이력 / 가정→사전등록 검정) →
  "우리가 아는 한 이 결합은 아직 다뤄지지 않았고, 계열의 자연스러운 다음 단계다"** 한 문장.
  부재-프레임은 반례 하나에 무너지지만 확장-프레임은 계보를 인용할수록 강해진다.
- **두 주장(§2) 분리 유지.** 그리고 GP 무승부(§7.5) 때문에 "모든 오프라인 방법을 이긴다"는 표현 금지.
- `CES_VT`에 대해 **"전역 n.s.인데 peak에서 PASS"만 쓰지 말 것** — 부호가 반대인 영역들의 평균이므로
  §8r의 3영역 분해와 함께 서술해야 한다.

---

## 11. 선행연구에서의 위치 (`docs/paper/NOVELTY.md`)

적대적 선행조사 2회(2026-07-03, 2026-08-05 재검증, 2026년 8월 발행분까지) 결과 **세 주장 모두 유지**.

- **계보**: NN-CES 피팅(JET, Bishop & Roach 1993; Svensson & von Hellermann 1999), EAST 고속 Ti
  (Chai 2019), 진단 간 추론 및 시간 조밀화 계열(Diag2Diag Nat.Commun. 2025, COMPASS PPCF 2026,
  EAST NF 2025, FusionMAE, RTCAKENN NF 2024, EAST XCS→Ti/rotation NF 2024).
- **가장 가까운 위협**: RTCAKENN(불순물 Ti+회전 출력, CER/TS 결측에 강건). 방어 3가지 — 입력이
  제어실 신호이지 BES/ECEI/Mirnov가 아니고, 과거-타겟-이력 채널이 없으며, 보간과 벤치마크된 적이 없다.
- **우리 기여 3축**: ① 성긴 **이온** 진단의 인과적 측정 간 채움(요동 진단 + 타겟 자신의 불규칙 과거),
  ② **미래를 쓰는 보간을 통계적 잣대로 세운 사전등록 프로토콜**(가장 강한 주장 — 우리 자신의
  `V_rot` 주장을 기각시킬 수 있는 프로토콜이다), ③ `T_i`↔`V_rot` 정보 비대칭의 **측정된 정량화**
  (발견이 아니라 정량화로 팔 것 — 물리는 사전에 예측 가능하고, Char et al. 2024가 토크 입력이
  있으면 회전이 학습 가능함을 보여 이 주장을 반증가능하게 만든다).

---

## 12. 실행 방법

```bash
python -m pip install -e ".[dev]"        # requires-python >= 3.11
python -m pytest -q                       # 34개 테스트
.\ces_prediction\run_smoke_test.ps1       # pytest + 1-epoch 소형 학습 (PowerShell)
python ces_prediction/train.py            # 전체 학습
python ces_prediction/evaluate.py         # persistence/mean 대비 클린 평가
python ces_prediction/compare_baselines.py # 모델 vs 보간 — 논문의 비교
python ces_prediction/analyze_data_evidence.py  # 결측 장부 + MC 앨리어싱 + Te/NBI 프로브
```

- **항상 저장소 루트에서 실행.** 러너/스크립트는 형제 모듈을 bare name으로 import한다
  (`from dataset import ...`). `ces_prediction.*` 패키지 import 경로는 런타임 코드에 없다.
- 로컬 인터프리터는 `py`(Python 3.14). GPU: RTX 5060(Blackwell), torch 2.11.0+cu128 (cu128 이상 필요).
- 테스트는 34개(4개 파일). `test_architecture.py`의 3개 중 **`test_dry_run`만 데이터 없이 돈다** —
  나머지 2개와 학습/평가/러너는 전부 실 CSV가 필요하다.
- 주요 환경변수: `CES_WINDOW_SIZE`, `CES_BATCH_SIZE`, `CES_EPOCHS`, `CES_LR`, `CES_SEED`,
  `CES_INIT_SEED`, `CES_MAX_TRAIN_SAMPLES`, `CES_MAX_VAL_SAMPLES`, `CES_MAX_SAMPLES_PER_FILE`,
  `CES_TEMPORAL_SUBSETS`, `CES_DROP_STUCK_TARGETS`(**항상 명시**), `CES_PER_SHOT_NORM`,
  `CES_ABLATE`, `CES_MODEL_FILE`, `CES_FILE_SPLIT_FROM`, `CES_TEST_FRACTION`,
  `CES_SPLIT_DIR`, `CES_OUTPUT_DIR`, `CES_DATA_DIR`.
- **일회성 실행은 별도 `CES_SPLIT_DIR`/`CES_OUTPUT_DIR`를 쓴다** (smoke는 `data/.smoke_*`).
- 코드 변경 후 `python -m pytest -q`; 학습/데이터/모델 동작이 바뀌었으면 smoke 학습도 돌리고
  통과 여부를 기록한다.

---

## 13. 산출물 지도

| 무엇 | 어디 |
|---|---|
| 영문 논문 | `docs/paper/main.tex` → `main.pdf` (24 pp, 6 figures) |
| 국문 논문 | `docs/paper/main_ko.tex` → `main_ko.pdf` (22 pp) |
| 논문 수치 단일 출처 | `docs/paper/paper_numbers.json` ← `ces_prediction/collect_paper_numbers.py` |
| 논문 그림 | `docs/paper/make_figures_en.py` → `docs/paper/figures/*.png` |
| 참고문헌 / novelty 판정 | `docs/paper/refs.bib`, `docs/paper/NOVELTY.md` |
| 발표자료 4종 | `docs/presentation/*.pptx` (일반 / 20분 / 연구흐름 / 종합방어) + `build_pptx*.py` |
| 1-pager | `docs/presentation/KSTAR_CES_1pager.pdf` |
| 보간 기준선 인용근거 | `docs/interpolation_baselines_references.md` |
| IonQ 곁가지 기록 | `docs/ionq_qpu_실험기록.md` |

**신뢰 가능한 체크포인트 계열** (전부 `data/` 아래, gitignored):

| 무엇 | 디렉터리 |
|---|---|
| 최종 모델 seed 42 / 1·7·123 | `.vt_repro_out` / `.vt_repro_ms_{1,7,123}` |
| ablation | `.vt_repro_ab_{no_fast,no_history}` |
| iter2 "before" 기준선 | `.final_out` (기록값 +0.0878 / RMSE 412.42 정확 재현) |
| held-free 계열 (§8c 이후 control) | `.sf_iter009_s{42,1,7,123}` |
| `seq_v2` / 분해 arm | `.seq_v2_lstm_s*` / `.seq_v2_nops_lstm_s*` |

논문 §구성: Introduction / Related work / Data and problem setup / Model / Evaluation methodology /
Results(인과 기준선 → 헤드라인 → skill의 위치와 큰 간극의 상대 → MNAR → 캠페인 → 비대칭 →
필요한 이력 → 복잡도 사다리 → 고변동 영역) / Architecture-selection protocol / Is it deployable? /
Where the remaining headroom is / Limitations / Conclusion / Reproducibility·Code·Data availability.

---

## 14. 명시된 한계 (논문 §Limitations와 동일)

- **통계적 검정력 (~96 test shot)** — shot이 올바른 독립 단위이고 그 수가 적다. 모든 유의성 판정의
  구속 조건이며 `CES_VT`가 해상되지 않는 이유다.
- **무거운 꼬리** — shot별 제곱오차 차이가 heavy-tailed. 소수 방전이 bootstrap 산포를 지배.
- **MNAR 낙관 상한** — skill은 관측된 CES 점에서만 측정된다. §8i가 재가중으로 정량화했지만
  weight 추정 변동성은 CI에 포함되지 않았다.
- **이웃 접근 비대칭** — 보간은 shot 전체 이웃을, 모델은 `window_size=4` 이력만 쓴다(의도된 설계).
- **단일 아키텍처·단일 윈도우** — 다만 §8f가 윈도우 민감도를, §8t가 대안 프레이밍을 다뤘다.
- **얇은 큰 간극 구간** — Δt > 25 ms 구간은 수십 샘플, per-bin CI 없음. §8g가 pooling으로 일부 해소.
- **CES fit 실패 인공물이 평가 집단에 남아 있다** — 헤드라인을 보수적으로 만든다(§8q).
- **conformal coverage는 marginal이지 conditional이 아니다** — shot별 coverage가 50–100%로 흔들린다.
- **shot별 표준화는 오프라인 상한** — 온라인 추정기는 shot의 미래를 볼 수 없다. 배치형(expanding
  window / EWMA) 버전과의 격차는 미측정.

---

## 15. 열린 항목 / 다음 단계

**측정으로 결판나는 것 (비용 순):**
1. **`seq_v2` seed 확대** — seed당 1.3분이므로 16 seed = 20분. 4/4 양의 부호(+0.045)가 실제인지
   seed 노이즈인지 정산할 수 있는, 기록 전체에서 가장 싼 미결 질문. **다음에 할 일 1순위.**
2. **`seq_v2` vs 윈도우 파이프라인의 학습 예산 균등화** (양쪽 다 고정 epoch 또는 양쪽 다 early
   stopping) — +0.045를 아키텍처 효과로 인용하기 전에 필요.
3. **인과(past-only) GP arm** — §8p의 무승부를 깨는 명시된 tie-breaker 중 하나(다른 하나는 shot 수 증가).
4. **shot별 표준화의 배치 가능 버전**(expanding-window/EWMA)과 오프라인 상한의 격차 측정.
5. **peak 탐지기를 genuine 이웃만으로 재계산** — §8r이 현재 탐지기가 hold 계단에 부분적으로
   반응함을 보였다.
6. **bracket-distance 층화** — Δt(마지막 관측 이후)가 아니라 가장 가까운 *미래* 앵커까지의 거리로
   층화하면 nowcaster의 한계 가치가 가장 큰 영역이 분리된다.

**데이터가 있어야 풀리는 것 (진짜 레버):**
7. **NBI 토크/파워 채널 확보** — `CES_VT`의 최상위 레버. 현재 데이터셋에 아예 없다.
8. **원본 kHz Mirnov 스트림** — 윈도우 RMS / band power / mode number. 파생 특징 재시도는 금지
   (§8b.2에서 정보가 이미 파괴됨이 확정).
9. **이력의 도달거리(reach) 확장** — 깊이가 아니라 도달거리. 전체 격자 프레이밍(§8t)이 이 제약의
   프레이밍 몫을 이미 제거했다.

**사용자(승상님) 결정 사항:**
10. **투고처(venue)**, **공저자/지도교수**, **아카이브 DOI**(Zenodo 발급은 본인 계정 필요).
    DOI가 두 `.tex`에 남은 마지막 `TODO(user)`다.

---

*작성 2026-08-10. 근거는 모두 `THESIS_RESULTS.md` §1–§9·§8a–§8u, `PROJECT_KNOWLEDGE.md`,
`docs/paper/paper_numbers.json`, `docs/paper/NOVELTY.md`, `ces_prediction/experiments/README.md`.
수치를 인용할 때는 이 문서가 아니라 원본을 재확인할 것.*
