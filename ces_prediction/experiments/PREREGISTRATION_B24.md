# B.24 — `V_rot` 측정 모형(σ_n, ν)의 TRAIN 우도 적합 (사전등록 v1, 2026-11-01)

**지위.** `PREREGISTRATION_B23.md` 전체를 상속한다(데이터 처리, 판독, 채점, 대조, 막대). **학습 없음.**
이 문서와 적합·판독·판정 코드는 B.24 적합 결과를 포함한 어떤 B.24 수치보다 먼저 커밋한다.

**출처.** B.23 규칙 3(§8bq): "λ는 레버가 아니다 → 같은 우도로 σ_n과 ν를 함께 TRAIN 적합. 그마저 동률이면 §8bl의
'10 ms `V_rot` 본류에서 persistence가 중앙값 최적'이 적응형 필터에도 성립한다고 기록한다." **이미 알고 있는 것:**
§8bs A1(광자 잡음 바닥, √2 비 8/8)과 §8bq(λ 적합은 M1 0.000)을 본 뒤에 쓰는 등록이며, 예측은 그 사실을 반영한다.

## 1. 통제 변수

`V_rot` 셀의 측정 모형 **(σ_n, ν) 하나**. (λ, σ_J)는 B.23 split별 적합값으로 고정한다(비교 기준 = `.b23_anchor`).
σ_d = 0.05, p0 = 1 고정. `T_i` 셀 불변.

## 2. 적합 (TRAIN만)

`b16_physics/fit_vrot_noise_b24.py` → `data/.b24_vrot_noise_fit.json`. 목적함수·재귀는 B.23과 같은 함수
(`fit_vrot_jump_b23.scan_loglik`, r과 ν를 격자 차원으로 넘김).
격자: σ_n ∈ {0.03, 0.04, 0.05, 0.06, 0.07, 0.085, 0.10, 0.13, 0.17} 표적 sd × ν ∈ {2.2, 2.5, 3, 4, 6, 10, 30}.
B.23 기준점(0.060, 3)이 격자 위에 있다. 가장자리 최적은 그대로 쓰고 기록한다(격자 확장 금지).

## 3. 판독과 채점

`anchor_readout_b16.py --arm v4kjt --vt-jump data/.b23_vrot_jump_fit.json --vt-noise data/.b24_vrot_noise_fit.json`
→ `.b24_anchor_{pop}_s{seed}`: B.19 체크포인트에서 `kalman_vt`의 lam_logit, j_raw(B.23 적합), r_raw, nu_raw(B.24 적합)만 바꾼다.
`rescore_b13.py --extra anchorf=.b24_anchor_{pop}_s{s} --out data/.b24_rescore.json`, `verdict_b24.py` → `data/.b24_verdict.json`.
**대조:** `T_i` TEST 오차가 B.19와 비트 동일(FATAL).

## 4. 가설과 판정 (수치 전에 고정)

- **H24-a.** 적합 앵커 `V_rot` M1 vs persistence > 0 ∧ CI > 0, ≥ 3/4 split.
- **H24-b.** B.24 − B.23 풀링 M1 `V_rot` ≥ +0.02 ∧ CI > 0.
- **H24-c.** 적합 앵커 `V_rot` M1 vs 백본 ≥ +0.05 ∧ CI > 0, ≥ 3/4 split.
- **H24-d (해석).** 적합 σ_n(km/s), ν, B.23 기준 대비 로그우도 이득, 가장자리 여부.

판정 규칙: (1) a ∧ c(두 모집단) → `V_rot` 물리 중앙값 추정량 무조건부, §8bp 표와 논문 갱신. (2) b만 → 부분.
(3) a, b 모두 실패 → **"10 ms `V_rot` 본류에서 persistence가 중앙값 최적"이 적응형 t 필터에도 성립**한다고 기록하고,
남은 레버는 §8bs의 노출별 맞춤 불확도(데이터 요청) 하나로 적는다.

## 5. 예측 (수치 전에)

σ_n 0.04–0.06 표적 sd(3–4.4 km/s), ν 2.5–4, `V_rot` 중앙값/persistence 0.97–0.98, H24-a 0–2/4, H24-b 실패, H24-c 4/4 → 규칙 3.

## 6. 비용

적합 CPU 수 분, 판독 GPU 8 × 약 3분, 재채점 CPU 약 1시간.
