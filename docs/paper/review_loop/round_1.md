# Round 1 판정 — 프로그램 위원장 (2026-11-13 01:50 기준)

대상: `docs/paper/short/main_short.tex` (영문 원본, 743행; 본문 8쪽, 9쪽은 참고문헌). 심사위원 6명
(physics, mlstats, hostile, integrity, venue, clarity)의 비평 70건과 저자 측 반박을 읽고, 다툼이 있는
사실은 저장소에서 직접 확인했다. 양쪽 어느 쪽 말도 그대로 믿지 않았다.

## 0. 한 단락 요약

T_i 결과는 실재한다. 다만 논문이 그 결과를 감싼 틀은 대부분 기록과 어긋난다. (1) "strictly causal"은
거짓이다. 15개 fast 채널을 그 방전 전체(미래 포함)의 평균·분산으로 표준화한다. 다만 무작위 분할의 T_i
헤드라인은 이 누설에 기대지 않는다. 학습 파일 통계만 쓰는 변형(이전 프로토콜)도 PCHIP 대비 4/4 분할에서
구간이 0을 제외한다. campaign-shift 결과는 이 누설에 전적으로 기댄다. (2) "인과 추정기가 미래 보간을
넘으면 CES 밖 정보가 있다"는 추론은 틀렸다. 파라미터 하나짜리 CES 이력 running mean도 PCHIP을 넘는다.
결정적인 배치형(deployable) 비교 대상은 causal GP다. seq_v2는 그 대비 +0.113 [+0.085, +0.132]다. (3) **V_rot의
PCHIP 대비 0 제외 결과는 반복당 약 71,700행 중 6–9행의 fit-failure anchor가 만든다.** 위원장이 재계산해
확인했다. 그 행을 빼면 구간이 0을 포함한다. persistence 대비 양수는 유지된다. (4) "the equation the network
computes"와 "physical filter"는 과장이다. 형태를 TEST 행 Jacobian에서 골랐으므로 retention은 post hoc이다.
방향: 이 논문은 **"같은 위치의 빠른 진단이 10 ms 희소 CES에 무엇을 더하는가"**라는 물리 질문으로 다시 쓴다.
T_i에는 CES 잡음 제거에 더해 BES/ECEI의 동시 과도 신호가 있다. V_rot에는 검출되는 fast 채널 정보가 없고
자기 이력만 남는다. T_i 경로는 20-parameter surrogate로 근사된다. 통계는 그 측정 도구로만 둔다.

## 1. 위원장이 직접 확인한 사실 (판정의 근거)

| # | 확인 내용 | 근거 |
|---|---|---|
| F1 | fast 채널은 파일 전체 행의 `fast.mean(axis=0)`, `fast.std(axis=0)`로 표준화된다. headline, B.5i, B.27c run.log 모두 `per_shot=1`이다. | `ces_prediction/experiments/seq/seq_data.py:207-210`, `data/.b1_seqv2_s42_i42/run.log` |
| F2 | 인과 정규화(학습 파일 통계, `per_shot=0`) seq_v2의 T_i vs PCHIP: +0.248 [+0.110, +0.334], +0.219 [+0.129, +0.288], +0.280 [+0.209, +0.333], +0.301 [+0.150, +0.385]로 4/4 모두 0을 제외한다. 단 이전 프로토콜이다(2026-08-09, held-free, 3 keV cut 없음, 다른 split 파일, train 446 blocks). | `data/.seq_v2_nops_lstm_s{42,1,7,123}/bootstrap_summary.json`, THESIS §8t Finding 5 |
| F3 | campaign split에서 per-shot 정규화는 그 split의 TEST 실패를 고치려고 채택됐다(window OFF 2/4 → ON 4/4). V_rot은 그 split에서 PCHIP 대비 0/4다(+0.19…+0.23, n.s.). | THESIS §8s Part 1, §8ab.6 |
| F4 | CES 이력만 쓰는 VAL-tuned EWMA의 PCHIP 대비 skill(cut): +0.050 [−0.069, +0.144], +0.136 [+0.072, +0.188], +0.139 [+0.062, +0.197], +0.170 [+0.100, +0.224]로 3/4가 0을 제외한다. ewma_inc의 causal GP 대비 skill은 −0.044…+0.026이다. | THESIS §8bz 표 |
| F5 | **V_rot fit-failure anchor (B.27c, repeats r2–r9, analyze_b27c와 같은 B=20,000·seed 2728).** 기록된 결과는 PCHIP 대비 +0.313 [+0.072, +0.701](cut)이다. anchor 오차가 1,000 km/s를 넘고 표적은 ≤1,000 km/s인 6–9행을 빼면 −0.011 [−0.066, +0.123]이 된다. 이 행들이 PCHIP SSE의 30–57 %를 차지한다. 대칭 규칙(표적 >1,000 km/s 4–6행도 제외)에서는 PCHIP 대비 +0.095 [−0.020, +0.179], persistence 대비 +0.257 [+0.174, +0.324], causal GP 대비 +0.176 [+0.015, +0.362], offline GP 대비 −0.073 [−0.189, +0.016]이다. inclusive에서는 각각 +0.083 [−0.050, +0.181], +0.241 [+0.148, +0.314], +0.144 [−0.010, +0.333], −0.098 [−0.224, −0.003]이다. 표적 >1,000 km/s인 4–6행은 모델 V_rot SSE의 66–68 %를 차지한다(8개 repeat 중 7개). | `docs/paper/review_loop/round1_chair_checks.py` (proxy 규칙, 저장된 오차로 anchor를 추정) |
| F6 | 20-parameter surrogate(b28h)를 단독 추정기로 보면, cut에서 causal GP 대비 +0.024 [−0.012, +0.057], +0.054 [+0.032, +0.074], +0.068 [+0.032, +0.101], +0.033 [−0.004, +0.069]로 2/4가 0을 제외한다. PCHIP 대비 +0.126…+0.205(4/4), persistence 대비 +0.322…+0.368(4/4), offline GP 대비 −0.050…−0.114(4/4 모두 0 아래)다. inclusive에서 causal GP 대비는 1/4만 0을 제외한다. | 같은 스크립트, `data/.b28_b28h_{cut,incl}_s*/comparison_errors_test.npz` |
| F7 | 등록 순서: B.27 NB primary (4d54928, 11-03 12:15) → B.27b post hoc pooled (f938f19, 16:22) → B.27c 등록 (e0ffd2f, 16:26)이다. 커밋 문구는 "r0–r1 excluded from the primary because their numbers shaped the rule"이다. §8bx reading 4는 "does not carry the TRAIN-overlap correlation"이라고 쓴다. | `git log -- ces_prediction/experiments/PREREGISTRATION_B27.md`, THESIS l.6484 |
| F8 | 전사 오류 세 건: ablation은 −0.2463 / −0.3753 / −0.4178 / −0.4246이므로 "−0.25 to −0.42"여야 한다(논문·§0은 −0.43). V_rot 필터의 backbone 대비 중앙값 skill은 cut +0.09…+0.23, inclusive +0.09…+0.26이다(논문 l.684는 +0.27). in-domain 비율은 cut 53.6–67.0 %, inclusive 54.1–67.7 %다. | `docs/paper/paper_numbers.json` |
| F9 | CES 표적은 R = 2.198 m 단일 chord다. BES 9채널과 ECEI 4채널은 모두 그로부터 약 1.5 cm 안에 있다. Mirnov 두 코일은 한 toroidal 위치에 있는 poloidal 배열이다. 논문에는 이 내용이 없다. | `PROJECT_KNOWLEDGE.md:1187-1196` |
| F10 | P02는 ΔV_rot–ΔBES/ECEI/MC = 0을 모든 lag와 10–200 ms window에서 보였다. 같은 통계로 T_i 양성 대조는 +0.15…+0.28이다. 비라우팅 seq v1은 V_rot에서 control 대비 −0.044 / −0.072 / −0.067 / −0.023으로 4/4 유의 손실이다. | THESIS l.4937 (P02), §8t Finding 5 |
| F11 | P07: 이웃 step 변화 5분위에 따라 결측률이 0.77 → 4.45 %로 오른다. BES 최하 5분위는 19.9 %, 최상은 3.7 %다. T_i/V_rot 마스크 일치율은 72.8 %다. | THESIS l.4942, §8at |
| F12 | 문맥 절단 시 T_i deficit은 20 ms −0.066, 30 ms −0.033, 40 ms −0.021이고 50 ms에서 포화한다. | THESIS §8al 표 |
| F13 | §8bs: A1(√2 비) 1.395–1.532는 통과했다. A2(√T_i 스케일링) slope 0.087–0.228은 0.5 ± 0.15 예측에 대해 0/4×2로 불통과다. | THESIS l.6230 |
| F14 | Fig. 2 SVG에 "(P02)", "(§8ab)", "pack_padded_sequence", "budget 1e6", "early stop 6"이 그대로 렌더링된다. | `docs/paper/figs/fig_arch_seq_v2_en.svg` |
| F15 | GPU(RTX 5060 Laptop)는 다른 프로젝트의 VLM probe가 메모리 7.9/8.1 GB, 100 %로 쓰고 있다. 이번 라운드에서 GPU 학습 분석은 불가하다(프로젝트 규칙상 CPU 대체 금지). | `nvidia-smi`, 01:40 |

## 2. 비평별 판정

판정 기준: STANDS = 핵심 주장이 기록과 일치하고 수정이 필요. PARTLY STANDS = 일부만 맞거나 이미 기록에 답이
있음. REFUTED = 기록이 반증. 70건 중 완전 반증은 없다. 다만 여러 건에서 핵심 하위 주장이 반박됐다.

### 2.1 physics

| id | 판정 | 한 줄 근거 |
|---|---|---|
| phys-1 | STANDS (심각도 fatal→major) | F1 확인. 무작위 분할 T_i는 인과 정규화 변형에서도 4/4 유지(F2, 이전 프로토콜). campaign-shift는 누설에 전적으로 기댐(F3). "strictly causal"·지연 시간의 배치 의미는 철회 대상. |
| phys-2 | STANDS | l.108-110 추론은 F4로 반증. causal GP도 Table 1 RMSE에서 PCHIP보다 낮음. CES-only 인과 smoother 위 여유(causal GP 대비 +0.113 [+0.085, +0.132])는 남음. |
| phys-3 | STANDS | 단일 chord R = 2.198 m 미기재(F9). "core to the pedestal top"(l.650, 731-732)은 근거 없는 범위. 3 keV cut은 spike 구조(85 % 단일 행, 이웃 대비 중앙 13×)로 일부 방어됨. |
| phys-4 | STANDS | 소비자 미명시. offline GP 대비 −0.010 [−0.043, +0.014]. 실시간 CES 가용성 근거 없음. |
| phys-5 | STANDS | §8bz-3 Jacobian은 동시 100 ms high-pass이고 적분형 equilibration 응답이 아님. per-shot z는 절대 T_e를 지움. beam-state 교란 미검. Q_ei 기전 문장(l.290-291)과 τ_eq "same order"(l.318-324)는 삭제. |
| phys-6 | PARTLY STANDS | 이름(physical), per-discharge σ 단위, inclusive 부호 반전은 맞음. Jacobian은 TEST 라벨이 아닌 입력·출력을 읽었으므로 "label leakage"는 과장. 그러나 stage-3 resid2가 TEST 오차를 읽었고 같은 TEST로 재채점했으므로 0.72–0.78은 post hoc 표기 필요. |
| phys-7 | PARTLY STANDS | "no route exists", NTV·turbulent-flux 표 행, Mirnov cross-phase는 과장. "lag-0만 검정"은 F10(P02 lag·window, seq v1 4/4 손실)로 반박. 잡음 감쇠 보정은 미보고 → 분석 A3. |
| phys-8 | PARTLY STANDS | "photon floor"는 F13으로 지지 안 됨. "holds가 공통 fit과 모순"은 약함(마스크 일치 72.8 %, 채널별 후처리 가능). "native cadence slower"(l.214-215)는 기록에 근거 없는 단정. |
| phys-9 | STANDS | F7 순서 확인. 신규 80 run이 새 fit이라는 완화는 인정. "conservative"는 기록과 모순. 꼬리 지배는 F5로 더 강하게 확인. |
| phys-10 | PARTLY STANDS | 적용 범위(54–68 %, 4–6 %)와 cut 2/4는 맞음. l.97 단정은 논문에 근거가 없으나 F11(P07)에 측정값이 있어 인용으로 해소. beam-off 가설은 §8at(동시 결측 2.5 %)로 대부분 반박. |
| phys-11 | PARTLY STANDS | raw-Mirnov 6/3 shot 학습 검정은 power 부족. NBI 0.34 %는 T_i 0.25 %와 같아 거의 동어반복. 100 Hz = raw 단일 샘플(24/24)은 모델 없는 측정이라 유지. |
| phys-12 | STANDS | Table 3은 W=2 windowed control의 평가 시점 zeroing이고 재학습 history-only seq_v2는 없음. |

### 2.2 mlstats

| id | 판정 | 한 줄 근거 |
|---|---|---|
| ml-1 | STANDS (fatal→major) | phys-1과 같음(F1–F3). |
| ml-2 | PARTLY STANDS | 문장 반증과 "AR (local)" = 2점 외삽(잡음 분산 5σ²라 −4.04 skill이 기대값)은 맞음. "남은 효과 +0.05"는 P06의 1-step 정보 경계이지 측정 격차가 아님. §8bz 표 산술로 EWMA 대비 ≈ +0.11…+0.14, causal GP 대비 +0.08…+0.17(8/8). |
| ml-3 | STANDS | 평가 시점 zeroing은 OOD 손상을 섞음. 다른 모델에서 측정됐음. −0.125 행은 F4와 모순. |
| ml-4 | STANDS | F7. "new/fresh partitions"는 같은 639 방전의 재분할. V_rot vs PCHIP는 F5로 어차피 재진술. |
| ml-5 | STANDS | B.1 gate(§8x), per-shot 채택(§8s), stage-3 Jacobian 모두 같은 4 TEST 분할 사용. l.457-458은 거짓. |
| ml-6 | PARTLY STANDS | fidelity 지표 부재, TEST 유래 형태는 맞음. "대부분 CES smoother"는 persistence 기준에서만 맞음. PCHIP 기준에서 EWMA 단독은 0.29/0.55/0.54/0.64, 식은 0.72–0.78. |
| ml-7 | PARTLY STANDS | tail statistic과 inclusive를 헤드라인에 쓰는 문제는 맞음. 그러나 cut T_i는 대칭 1 % trim +0.210*, Huber +0.156*(§8ax)로 소수 행의 산물이 아님. T_i 중앙값(−0.198 [−0.259, −0.135])은 초록에 들어가야 함. |
| ml-8 | PARTLY STANDS | T_i는 하한이 0에서 멀어 다중성 영향이 없음. V_rot은 맞으나 F5로 대체됨. BCa·coverage 점검은 minor. |
| ml-9 | STANDS | 소비자 부재, offline GP 구간이 0 포함. 크기 "+0.05"만 반박(ml-2). |
| ml-10 | STANDS (minor) | 학습형 raw-Mirnov 검정은 n = 6/3. aliasing 사실은 유지. |
| ml-11 | PARTLY STANDS (minor) | two-level bootstrap에 repeat(학습) 층이 있음. split×init×shot 분산 분해는 없음. |

### 2.3 hostile

| id | 판정 | 한 줄 근거 |
|---|---|---|
| sig-1 | STANDS | F5로 재현. 2026-08-16 규칙("every persistence-anchored V_rot comparison must report the spike-anchor rows' share of squared error", THESIS l.2280-2281)이 B.27c와 논문에 적용되지 않음. "without them it is zero"만 과장: 대칭 규칙에서 반복별 PCHIP 대비는 8/8 양수(+0.061…+0.111)이나 구간이 0을 포함. |
| sig-2 | STANDS (fatal→major) | phys-1과 같음. |
| sig-3 | STANDS | PCHIP는 잡음 anchor를 그대로 통과하는 보간. offline GP RMSE 153.8 < 157.8 eV. causal GP 대비 8/8 여유는 남음. |
| sig-4 | PARTLY STANDS | 소비자 부재는 맞음. "σ_meas 안쪽이라 무의미"는 범주 오류: skill은 같은 행 위 paired MSE 차이이고 RMSE² = σ² + b². |
| sig-5 | STANDS | ml-4와 같음. 등록 순서와 'conservative' 공개 필요. |
| sig-6 | PARTLY STANDS | 관측 행 masked 채점은 표준 프로토콜. MNAR cut 2/4는 초록·본문에 그대로 적어야 함. |
| sig-7 | STANDS | 단 "73–82 %가 running mean"은 persistence 기준. PCHIP 기준 EWMA 몫은 0.29–0.64. |
| sig-8 | PARTLY STANDS | inclusive의 여분은 spike 처리(논문도 l.577-579에서 인정). 결과(PCHIP−model 차)로 고른 0.1 % 절단은 증거로 인정하지 않음. 대칭 trim은 cut T_i를 지지. |
| sig-9 | STANDS | 초록에 약 40개 수치, claims는 실험 로그. |
| sig-10 | PARTLY STANDS | routing "사후 합리화"는 F10으로 반박. τ_eq 일치 문장은 반증 불가라 삭제. |
| sig-11 | PARTLY STANDS | memoryless MLP 기준선 없음은 맞음. ewma_inc(EWMA + 15 선형 fast 항)는 있고 backbone이 그 위에 있음. |
| sig-12 | PARTLY STANDS | 단일 장치·캠페인은 초록에 명시됨. Mirnov 3 shot은 탐색적으로 강등. |

### 2.4 integrity

| id | 판정 | 한 줄 근거 |
|---|---|---|
| causal-1 | STANDS (fatal→major) | F1. "title logic collapses"는 F2로 반박. windowed control은 per-shot OFF여서 gate +0.081은 정규화가 다른 비교라는 지적은 맞음(§8t: 둘 다 OFF일 때 +0.035로 부호 유지). |
| bar-2 | STANDS | F4. 귀속 증거는 다른 모델의 평가 시점 zeroing. |
| vrot-3 | STANDS | F7, 'conservative', campaign V_rot 0/4 생략(F3) 확인. "한 repeat(r9)이 주도"는 반박(r2–r8 +0.28…+0.35). 실제 동인은 F5의 소수 anchor 행. |
| title-4 | STANDS | 제목 동사, 중앙값 결과, offline GP 0 포함. "6.5 eV < σ_meas"는 범주 오류로 불인정. |
| test-5 | STANDS | ml-5와 같음. |
| floor-6 | STANDS | F13. "independent Gaussian line-fit errors와 일관"이 한계. |
| phys-7 | PARTLY STANDS | "ECEI supplies T_e", "physical", 기하 미기재, Mirnov 3 shot은 맞음. lag·비선형 미검은 F10으로 반박. |
| purpose-8 | PARTLY STANDS | ≥96 % 행이 Δt ≤ 15 ms(§0)이므로 헤드라인은 단일 step gap이 지배. >15 ms 층 +0.268/+0.206은 존재. held 행은 채점에서 빠지므로 "instrument가 주지 않는 양을 추정"은 과장. |
| num-9 | STANDS | F8로 세 건 확인. Fig. 3 캡션 문구(response on test inputs)도 고칠 것. |
| eq-10 | PARTLY STANDS | F6: causal GP 대비 2/4(cut). inclusive 한정은 초록에 있으나 결론(l.720-723)에 없음. |
| table-11 | STANDS | 등록 primary 표기 없음, 점추정 없음, AR 행 표기 오류. |

### 2.5 venue

| id | 판정 | 한 줄 근거 |
|---|---|---|
| causal-1 | STANDS (fatal→major) | F1·F3. per-shot을 campaign TEST 실패를 보고 채택한 뒤 같은 split 통과를 보고했으므로 독립 stress test가 아님. |
| purpose-1 | STANDS | σ_meas 비교만 범주 오류. |
| stats-1 | STANDS | ml-4와 같음. |
| stats-2 | STANDS | 80+ 실험 절이 같은 4 TEST 분할 사용. 등록은 자기 저장소. |
| phys-1 | PARTLY STANDS | 기하는 기록에 있음(F9) → 기재로 해소. ECEI 절대 T_e 불가, τ_eq 반증 불가는 맞음. event-detection 대안은 미검. |
| eq-1 | PARTLY STANDS | ml-6, phys-6과 같음. |
| eval-1 | PARTLY STANDS | 인공 masking 곡선 없음은 맞음. >15 ms 층과 §8al 문맥 절단(F12)은 있음. |
| base-1 | PARTLY STANDS | AR 표기 오류, 훈련형 CES-only 기준선(EWMA, ewma_inc, hold-or-jump, b28h)이 Table 1에 없음, GBM/MLP 없음은 맞음. 21k rung의 cut 동등(+0.002)은 설명 필요. |
| neg-1 | PARTLY STANDS | "closed"/"no route" 문구는 맞음. 비선형 held-out 검정(seq v1)은 존재. |
| fmt-1 | STANDS | CVPR형 형식, KFE/KSTAR 데이터 감사·사용 진술 없음, "released repository"(데이터 비공개), 30단어 제목. |
| minor-1 | STANDS (minor) | offline GP 셀 혼합, 60 vs 60–66 shot, 수치 반복. |

### 2.6 clarity

| id | 판정 | 한 줄 근거 |
|---|---|---|
| pres-1 | STANDS | 소비자 부재. causal vs offline 틀 자기모순. |
| pres-2 | STANDS | F7. |
| pres-3 | STANDS | offline GP RMSE가 더 낮음. 식은 독립 학습 surrogate이고 두 번째 학습 형태. |
| pres-4 | STANDS | 같은 수치가 4–5회 반복. |
| pres-5 | PARTLY STANDS | 내부 용어 문제와 Fig. 2 내부 태그(F14)는 맞음. "early stop 6 vs 30-epoch cap"은 patience와 상한이라 모순 아님. |
| pres-6 | PARTLY STANDS | Eq.(4) GNS, methods lineage, τ_eq, Table 1은 삭제 대상. Fig. 3도 삭제하고 1.3 step 기억은 본문 한 문장으로. |
| pres-7 | STANDS | 예측 trace 그림 없음. 기존 png는 확정 프로토콜 확인 전 재사용 금지. |
| pres-8 | STANDS | 세 판독 혼재. 등록 primary 표시와 점추정 필요. |
| pres-9 | STANDS | 귀속 증거가 다른 모델. V_rot 행은 구성상 동일. |
| pres-10 | PARTLY STANDS | "reach / whole segment"는 과장. 그러나 문맥 효과는 측정됨(F12: 20 ms −0.066, 50 ms 포화). "약 50 ms 문맥 + carried value"로 재진술. |
| pres-11 | STANDS (minor) | "registered" 21회. |
| pres-12 | STANDS (minor) | 본문 8쪽, 9쪽 참고문헌(허용). 41.1 %/54 % 분모, Mirnov 2 vs 24 연결 필요. |

## 3. 출판 가능성 (위원장 추정, 현재 → 이번 라운드 수정 반영 후)

| venue | 현재 판정 | 지금 | 이번 라운드 후 | 비고 |
|---|---|---|---|---|
| Nuclear Fusion | reject | ~3 % | ~6 % | 단일 chord·캠페인, 기전 미식별, 소비자 없음. 두 번째 캠페인 + 다채널 + 소비자가 필요. |
| PPCF | reject | ~3 % | ~6 % | 같은 이유. |
| Fusion Engineering and Design | major revision / reject | ~15 % | ~35 % (이후 작업 1–2 완료 시 ~50 %) | 가장 현실적인 archival 목표. NN-CES 게재지. elsarticle, KSTAR 데이터 진술 필수. |
| Nuclear Engineering and Technology (KNS) | major revision | ~20 % | ~40 % | 국내 SCIE, 핵융합 공학 수용. |
| Journal of the Korean Physical Society | major revision | ~25 % | ~45 % | KSTAR 결과 다수 게재. |
| Machine Learning: Science and Technology | reject | ~10 % | ~25 % | 방법론 신규성 요구. 단일 데이터셋이 약점. |
| NeurIPS ML4PS 2027 workshop (non-archival, 4쪽) | accept 가능권 | ~35 % | ~55 % | 2026 마감은 지남. 2027 CFP(대략 8–9월) 확인. |
| ICLR 2027 workshop (과학·시계열 계열) | 가능권 | ~30 % | ~45 % | 마감 대략 2027년 1–2월, CFP 확인. |
| KPS / KNS 2027 춘계 초록 | 거의 확실 | ~95 % | ~95 % | 출판 아님. |
| RSI / HTPD | 범위 불일치 | <5 % | <5 % | 계측기 기여 없음. |

## 4. 방향 (무엇을 주장하고 무엇을 버리는가)

**틀.** "같은 위치(1.5 cm 이내)의 빠른 진단이 10 ms 희소 CES에 무엇을 더하는가." 물리 기전이 주축이고
통계는 측정 도구다. 주장은 둘로 유지한다(2026-11-11 결정).

- **주장 1 (이온 온도).** 다음 CES 측정을 읽지 않고 T_i를 채우는 추정의 PCHIP 대비 여유는 두 부분으로 나뉜다.
  하나는 CES 이력만으로도 얻는 잡음 제거다(running mean +0.05…+0.17). 다른 하나는 BES/ECEI의 동시 100 ms
  high-pass 과도 신호다. 배치형 결정 비교는 causal GP이며 +0.113 [+0.085, +0.132]다. offline 참조는 offline
  GP이며 −0.010 [−0.043, +0.014]로 0을 포함한다. T_i 경로는 20-parameter **surrogate**(공백 의존 관측 이득 +
  3σ 차단 high-pass)로 근사되고 PCHIP-relative skill의 0.72–0.78을 유지한다(cut, 형태는 post hoc). 기전은
  "동시 공변(co-movement)"까지만 말한다. Q_ei 충돌 교환은 이 입력으로 식별되지 않는다.
- **주장 2 (회전).** 같은 fast 채널에서 V_rot 정보는 어떤 lag·window(10–200 ms)에서도 검출되지 않는다
  (T_i 양성 대조 +0.15…+0.28). fast 채널을 V_rot에 연결하면 4/4 분할에서 V_rot skill이 떨어진다. 그래서
  V_rot은 자기 이력으로 추정하고 persistence 대비 양수다(A1 후 수치). PCHIP 대비 0 제외는 fit-failure anchor
  소수 행이 만든 것이므로 등록 값과 그 SSE 비중을 함께 보고한다. 10 ms 산포는 Gaussian line-fit 잡음 바닥과
  일관된다("photon"은 쓰지 않는다). V_rot를 실패로 쓰지 않는다. "이력에서만 오는 이득과 그 이유"로 쓴다.

**버릴 것.** "strictly causal"(A3·이후 작업 1 전까지), "Lower Error than Acausal Interpolation" 제목,
"deliberately hard bar" 단락과 l.108-110 추론, "the equation the network computes / physical filter",
Q_ei·"ECEI supplies T_e" 기전, τ_eq "same order" 문장, Table 1(momentum term audit), "no route exists /
closed negative", "photon floor", "conservative", "test never read during selection", "core to the pedestal
top", NBI 0.34 % 분산 논증, raw-Mirnov 정량 주장(탐색적으로 강등), 경사 잡음 척도 식, LSTM 셀 그림,
methods lineage 단락, CPMP/ICH 인용, 초록·주장·결론의 수치 반복.

**유지할 것.** 등록된 두 모집단(주장은 둘 다에서 성립할 때만 무조건 진술 → T_i 헤드라인은 양쪽에서 성립,
식은 cut 한정). shot 군집 bootstrap과 two-level 판독(등록 순서 공개 조건). dual readout(중앙값 vs 평균,
압축). 100 Hz Mirnov aliasing(모델 없는 측정). MNAR 재가중(범위 한정).

**사용자 기존 결정과의 관계.** (1) V_rot uncut(2026-08-16)은 바꾸지 않는다. 같은 결정이 요구한 "spike-anchor
SSE 비중 보고"를 이번에 이행한다. (2) "V_rot는 양성 결과"는 persistence와 causal GP 대비로는 유지되지만,
PCHIP 대비 0 제외는 F5로 유지되지 않는다. memory 갱신이 필요하다(이후 작업 10).

## 5. 지금 돌릴 분석 (각 25분 이내, 모두 CPU, 기존 산출물만)

GPU는 다른 프로젝트가 점유 중이다(F15). 따라서 GPU 분석은 이후 작업으로 미룬다.

- **A1 — V_rot fit-failure anchor 민감도 (최종 수치).** 질문: 등록된 V_rot PCHIP 대비 결과가 사전 명시한 대칭
  물리 범위 규칙에서도 0을 제외하는가? 그 행들은 각 arm의 SSE를 얼마나 차지하는가(2026-08-16 규칙)? 방법: B.27c
  r2–r9 npz와 네 정규 분할(cut `.b1_seqv2_s{s}_i{s}`, inclusive `.b5i_seqv2_s{s}`)을 쓴다. CSV에서 held-free 관측
  계열(0.5 s 이상 gap에서 segment 분리)을 다시 만들고, 각 채점 행의 표적, persistence anchor, PCHIP 양쪽 anchor
  (이웃 2개씩)를 찾는다. 어느 하나라도 |V_rot| > T면 모든 arm에서 그 행을 뺀다. T는 1,000 km/s(1차)이고 500과
  2,000을 민감도로 둔다. 판독은 analyze_b27c.hier(B = 20,000, seed 2728)와 split별 shot 군집 CI(B = 10,000,
  seed 12345)이고 기준 4종을 모두 본다. 출력은 `docs/paper/vrot_failfit_sensitivity.json`과 THESIS §8 새 절이다.
  F5 proxy와 교차 확인한다. 결정 규칙: T = 1,000에서 PCHIP 대비 two-level 구간이 0을 포함하면 주장 2를
  persistence(그리고 구간이 0을 제외하면 causal GP) 기준으로 재진술한다. 이때 Table 2에는 등록 값 +0.313
  [+0.072, +0.701]과 fit-failure 행의 SSE 비중을 나란히 둔다. 세 T 모두에서 0을 제외하면 등록 진술을 유지하고
  공개 문장만 더한다.
- **A2 — T_i 여유 분해와 surrogate의 단독 성능.** 질문: PCHIP 대비 여유 가운데 CES만으로 얻는 몫과, 가장 강한
  CES-only 인과 smoother 위에 남는 몫은 얼마인가? 20-parameter surrogate는 단독 추정기로 어디에 서는가? 방법:
  (a) `experiments/b28_equation/stage1_score.py`의 frozen-row 경로로 8개 셀에서 backbone vs VAL-tuned EWMA의
  shot 군집 skill을 구한다(B = 10,000, seed 12345, 셀당 1.1–1.5분). (b) `data/.b28_b28h_{cut,incl}_s*`로
  surrogate vs causal GP, PCHIP, persistence, offline GP를 구한다(F6 미리보기). 출력은
  `docs/paper/b28_decomposition.json`과 THESIS §8이다. 결정 규칙: (a) cut에서 backbone vs EWMA가 3/4 이상
  0을 제외하면 "잡음 제거 몫(EWMA vs PCHIP) + 남는 증분(backbone vs EWMA)"을 둘 다 구간과 함께 쓴다. 2/4
  이하면 "여유의 대부분이 잡음 제거"로 쓴다. (b) surrogate vs causal GP가 cut 2/4 이하(미리보기 2/4)면
  surrogate는 "PCHIP 대비 4/4, causal GP 대비 2/4"로만 보고하고 CES-only 인과 smoother를 넘는다고 쓰지 않는다.
  3/4 이상이면 배치 가능한 20-parameter 인과 추정기로 주장한다.
- **A3 — 잡음 보정 다중 시간척도 결합 경계.** 질문: V_rot 라벨이 더 시끄럽다는 점을 보정한 뒤에도 fast 채널과
  V_rot의 결합이 T_i보다 확실히 작은가? 방법: 641 CSV(held-free, T_i는 3 keV cut)를 쓴다. window w ∈ {10,
  20, 50, 100, 200} ms와 lag 0–200 ms에서 BES 평균, ECEI 평균과 표적(T_i, V_rot)의 w-차분 사이 방전 내
  pooled Pearson r을 shot 군집 bootstrap CI와 함께 구한다. 표적 w-차분의 잡음 분산은 2σ²_meas(차분 추정기)로
  보고 r을 1/√(신호 비율)로 보정한다. T_i는 양성 대조다. level 상관도 같이 낸다. 출력은
  `docs/paper/fast_target_coupling_bound.json`과 THESIS §8이다. 결정 규칙: 보정된 V_rot |r|의 95 % 상한이
  모든 w에서 0.05 미만이고, 보정 T_i r의 하한이 w ≤ 50 ms에서 0.10을 넘으면 "no route exists"를 "보정
  |r| ≤ (최대 상한), T_i ≥ (최소 하한)"으로 바꾼다. 어떤 w에서 보정 V_rot 구간이 0을 제외하고 |r| ≥ 0.05면
  routing 논증을 "10 ms에서 미검출"로 약화하고, 그 window를 V_rot 이후 작업 1순위로 둔다.

## 6. main_short.tex 수정 목록 (영문 원본, 한글판은 이후 반영)

번호 규칙: 수치는 §0, `paper_numbers.json`, json 산출물, 또는 "pending A#"에서만 가져온다. 본문에 내부
표지(B.27c, §8ab)는 쓰지 않는다. 결과는 점추정 + 95 % 구간 + 0 제외 여부로 쓰고 "beats/ties"는 쓰지 않는다.
"first"는 쓰지 않는다.

| id | 종류 | 위치 | 지시 | 수치 출처 |
|---|---|---|---|---|
| E1 | rewrite | l.47, l.51-52 제목 | "Filling 10 ms Charge-Exchange Spectroscopy Gaps in KSTAR: Ion Temperature from Co-located Fast Diagnostics, Rotation from Its Own History, and a 20-Parameter Surrogate Filter". 이후 작업 1이 인과 정규화를 확인하면 앞에 "Causal"을 붙인다. | 없음 |
| E2 | rewrite | l.62-83 초록 | §6.1 초안으로 전면 교체. 수치는 약 9개로 줄인다. "strictly causal", "photon floor", "physics filter", inclusive 범위, raw-Mirnov 수치를 뺀다. 정규화가 방전별 통계(offline 단계)라는 점과 단일 chord를 명시한다. | §0(B.27c), §8bz, §8ax, §8bz-4, P02, b27c_hier.json, PROJECT_KNOWLEDGE.md:1187, pending A1 |
| E3 | rewrite | l.97-98 (MNAR 단정) + l.101-103 뒤 | MNAR 문장을 P07 측정값으로 바꾼다: "the missing rate rises from 0.77% to 4.45% across quintiles of neighbouring-step change and is 19.9% in the lowest BES-intensity quintile against 3.7% in the highest". 질문 뒤에 사용처 한 문장: "The use case is an estimate that cannot wait for the next CES exposure; offline, a noise-aware smoother that reads both neighbours reaches the same mean error (Sec. 6.2)." | THESIS l.4942 (P07) |
| E4 | delete+rewrite | l.105-112 "A deliberately hard bar" | 단락을 지우고 "Comparators" 단락으로 바꾼다: 배치형(persistence, running mean, past-16 causal GP)과 offline 참조(linear, PCHIP, offline GP)를 구분한다. "PCHIP passes exactly through noisy anchors, so a smoother of the CES past alone already gains on it (running mean +0.05 to +0.17 over PCHIP, cut, interval excluding zero on 3/4 splits); the deciding comparison for a fast-diagnostic contribution is the causal GP." l.108-110 추론 문장은 완전 삭제. | §8bz 표 |
| E5 | rewrite | l.118-155 Claims + 맺음 | 번호 목록 두 항목을 각 3줄 이내 산문으로 바꾼다(§4 주장 1·2, 구간 하나씩). Claim 1: causal GP 대비 +0.113 [+0.085, +0.132](cut) / +0.118 [+0.076, +0.157](inclusive), surrogate 0.72–0.78(cut, post hoc). Claim 2: lag·window 무결합(P02), 비라우팅 4/4 손실, persistence 대비(pending A1). l.150-155는 "Code and evaluation artifacts are released; the KSTAR shot files are not redistributable."로 바꾸고 THESIS_RESULTS.md 언급을 지운다. l.114-116 두 모집단 문장에서는 inclusive 여분이 spike 처리라는 점을 한 구절로 덧붙인다. | §0, b27c_hier.json (incl/CES_TI/gp_causal), P02, §8t Finding 5, pending A1 |
| E6 | add+rewrite | §3 l.206-215, l.221-226 | chord와 기하 문장을 넣는다: "Both CES targets are the 2022 channel at R = 2.198 m; the nine BES and four ECEI channels lie within about 1.5 cm of it, and the two Mirnov inputs are two positions of one poloidal array at one toroidal location." l.214-215 "because the native V_rot cadence is slower than the grid"는 "the files do not record why; the T_i and V_rot masks agree on 72.8% of rows, so the two channels are post-processed separately"로 바꾼다. l.224-225 CPMP/ICH 비임상 비유와 인용은 삭제한다. | PROJECT_KNOWLEDGE.md:1187-1196, §8at |
| E7 | rewrite+delete | §4.2 l.278-291, Table 1 l.293-310 | "no route exists"와 Q_ei/"ECEI supplies T_e"를 삭제하고 §6.2 초안 문단으로 교체한다(rate·level 상관 유지, P02 lag·window 무결합과 T_i 대조, 비라우팅 seq v1 V_rot paired −0.044/−0.072/−0.067/−0.023 4/4, T_i 결합은 "동시 공변, 기전 미식별"). Table 1은 지우고 "NBI and NTV torques have no channel here (dB/dt cannot see a static δB); BES intensity is not ⟨ṽ_r ṽ_φ⟩; residual stress needs ∇T_i" 한 문장으로 대체한다. 보정 경계 수치는 pending A3. | §8ar, THESIS l.4937 (P02), §8t Finding 5, pending A3 |
| E8 | delete | §4.3 l.318-324 | τ_eq 계산과 "the same order" 문장(“The electron–ion equilibration time … not a measurement of n_e, T_e)”)을 삭제한다. Mirnov aliasing과 자기상관 시간 문장은 유지. | 없음 |
| E9 | rewrite | l.355-369, l.378-380 | "22 strictly causal channels"를 "22 channels; every CES-derived input uses rows before t only, while the 15 fast channels are standardized with the discharge's own mean and variance, an offline step"로 바꾼다. "Under this protocol the cost of a causal (training-statistics) standardization is not measured; under an earlier protocol without the fit-failure cut, that variant kept a T_i skill over PCHIP of +0.22 to +0.30 with intervals excluding zero on 4/4 splits."를 넣는다. "The model's reach into the past is the whole segment"와 "What the framing buys is reach"는 "Truncating the backbone's own context costs −0.066 at 20 ms and saturates at 50 ms: the past reaches the T_i output through the carried CES value and about 50 ms of fast-channel context"로 바꾸고, control은 per-discharge 표준화가 꺼져 있다고 밝힌다. | `data/.seq_v2_nops_lstm_s*/bootstrap_summary.json`(THESIS §8t Finding 5), §8al 표 |
| E10 | delete | Fig. 3 l.343-353 및 l.357 참조, Eq.(bsimple) l.389-398, l.191-200 Methods lineage | Fig. 3를 삭제한다(1.3 step 기억은 §4.5 본문에 있음). 경사 잡음 척도 식과 문단을 지우고 지연 문장만 남기되 "on CPU, given causal standardization"으로 한정한다. "Size, family and cost"는 한 문장으로 줄이되, 21k rung의 cut 동등(+0.002)과 inclusive −0.194를 근거로 LSTM을 유지하는 이유를 남긴다. Methods lineage는 한 문장 + 인용 3개로 줄인다(Nadeau–Bengio, hierarchical bootstrap, Maheswaranathan). | 기존 본문 수치(§0 closed axes) |
| E11 | rewrite | §4.5 l.401-441, 그리고 l.52, 74-75, 122, 133, 720-722 | 소제목을 "A 20-parameter surrogate of the T_i branch"로 바꾼다. "physical filter / the equation the network computes"를 전부 "20-parameter surrogate filter"로 바꾼다. "Stage 3 read the backbone's Jacobian on the TEST rows of the same four splits on which stage 4 is scored, so the retention is post hoc"를 공개한다. "d and θ are in per-discharge standard deviations (offline normalization)"을 둔다. surrogate 단독 성능을 한 문장 덧붙인다: PCHIP 대비 +0.13 to +0.21(4/4), causal GP 대비 +0.02 to +0.07(2/4), offline GP 아래(4/4) — pending A2. | §8bz-3, §8bz-4, pending A2 (F6 미리보기) |
| E12 | rewrite | §5 l.449-461, l.468-475; Table 1 l.506-514 | "a local AR" → "two-point extrapolation" (Table 1 행 이름도). Table 1에 "Running mean of past CES (1 parameter)" T_i 169.2 eV, "20-parameter surrogate" T_i 162.4 eV(pending A2) 두 행을 더한다. l.457-458 "never read during selection"은 공개 문장으로 바꾼다: "Test files were reserved before the architecture search, but the adoption of the sequence backbone, of per-discharge standardization and the surrogate's form used results on the same four test splits; the repeated cross-validation re-partitions the same 639 discharges." l.470-475는 순서를 공개하고 "conservative"를 지운다: "The repeated cross-validation was registered with the fold-level interval as its deciding readout; for V_rot against PCHIP it included zero. A pooled readout was then computed post hoc, and eight new repeats were registered with the pooled two-level readout as deciding before any of them ran. That interval conditions on the trained models and does not carry the correlation between overlapping training sets; the fold-level interval does." | Table 1 RMSE는 §8bz s42 cut EWMA/persistence skill +0.264 → 197.2×√0.736; surrogate는 pending A2(F6 persistence 대비 +0.322 → 197.2×√0.678) |
| E13 | rewrite | Table 2 l.522-554 + l.568-575 | pooled 열에 점추정을 넣는다(T_i: PCHIP +0.221, persistence +0.391, causal GP +0.113, offline GP −0.010; V_rot: +0.313, +0.451, +0.361). 캡션에 "fold-level: deciding readout as first registered; pooled: deciding readout registered afterwards for eight new repeats"를 쓴다. "eight fresh partitions"는 "eight re-partitions of the same 639 discharges"로 바꾼다. V_rot 단락에 "6–9 of about 71,700 rows per repeat whose interpolation or persistence anchor is a fit failure (>1,000 km/s) carry 30–57% of PCHIP's squared error; excluding them and the fit-failure targets, the skill over PCHIP is [pending A1], over persistence [pending A1]"을 넣는다. | b27c_hier.json, §0, pending A1 (F5 미리보기 +0.095 [−0.020, +0.179], +0.257 [+0.174, +0.324]) |
| E14 | rewrite | l.582-594 stress tests | in-domain을 "53.6–67.0% (cut) and 54.1–67.7% (inclusive)"로 쓴다. PCHIP 대비 cut 2/4는 그대로 두고 문장 앞쪽으로 옮긴다. campaign: "per-discharge standardization was adopted after the windowed control failed this split, and it uses whole-discharge statistics, so this is not an independent test of a deployable estimator"를 공개한다. V_rot 행 "+0.19 to +0.23 over PCHIP (0/4), +0.33 over persistence (4/4, cut)"을 추가한다. | paper_numbers.json mnar, THESIS §8ab.6, §8s Part 1 |
| E15 | rewrite | §6.3 l.596-632, Table 3, l.131 | Table 3 제목을 "Evaluation-time input sensitivity of the windowed control (no retraining)"로 바꾸고 V_rot 행(l.627-629)을 지운다. "−0.25 to −0.43"을 "−0.25 to −0.42"로 고친다(l.131, 601, 624). "the margin over interpolation is fast-diagnostic information"은 분해 문장으로 바꾼다: "about half of the margin over PCHIP is denoising available from the CES past (running mean +0.05 to +0.17); the rest concentrates on event rows (top-10% \|ΔECEI\| rows, seeds 42/1: backbone +0.116/+0.269 over PCHIP against −0.011/+0.088 for the running mean plus a linear fast-channel term)" + backbone vs running mean 구간(pending A2). | paper_numbers.json ablation_window_eval, §8bz 표·strata, pending A2 |
| E16 | rewrite | l.634-651 | aliasing(24/24, lag 0)은 유지한다. raw-feature 검정은 "an exploratory test (six training and three test discharges, a 15-feature arm that is expected to overfit at this size) found no gain"으로 바꾼다. NBI 분산 비율 논증(l.645-649)은 "no torque channel exists, so NBI is untested here"로 줄인다. "from the core to the pedestal top"(l.650)은 "at the R = 2.198 m chord"로 바꾼다. | §0 |
| E17 | rewrite | §6.5 l.682-692 및 l.80, 147, 730 | "+0.09 to +0.27"을 "+0.09 to +0.23 (cut) and +0.09 to +0.26 (inclusive)"로 고친다. "The reason is measured … photon floor"는 "consistent with independent Gaussian line-fit errors: the ratio of relative T_i to V_rot scatter is √2 on 8/8 cells (1.395–1.532), while the √T_i scaling expected of photon statistics was not observed (slope 0.09–0.23 against 0.5)"로 바꾼다. "photon floor"는 전부 "fit-noise floor"로 바꾼다. | paper_numbers.json physfilter.dual_readout, THESIS §8bs (l.6230) |
| E18 | rewrite+add | l.698-736 결론, bibliography 앞, Fig. 2 | 한계·결론을 10줄 이내로 줄이고 구간을 반복하지 않는다. "strictly causal", "for ion temperature and for rotation", "the network computes", "photon floor", "core to the pedestal top"를 삭제한다. 사용처 한 문장, offline-normalized fast 채널, 단일 chord·캠페인을 넣고, surrogate의 inclusive 한정(0.43–0.81)을 명시한다. Acknowledgments와 데이터 진술 자리를 추가한다(제공자와 KFE 사용 허가는 저자가 채움, 지어내지 말 것). Fig. 2 원본 `docs/paper/figs/fig_arch_seq_v2_en.svg`에서 "(P02)", "(§8ab)", 코드 상세 상자를 지우고 다시 렌더링한다. 캡션 l.334 "per-shot z"는 "per-discharge z (whole-discharge statistics)"로 바꾼다. | §8bz-4 (inclusive 0.43–0.81) |

### 6.1 초록 초안 (E2, 영문)

> Charge-exchange spectroscopy (CES) gives KSTAR's ion temperature \Ti{} and toroidal rotation \Vrot{} at
> one chord ($R = 2.198$\,m); on the 10\,ms grid of 641 discharges of the 2022 campaign, 8.2\% of \Ti{} and
> 23.9\% of \Vrot{} samples are missing. We ask what beam-emission spectroscopy (BES) and
> electron-cyclotron-emission imaging (ECEI) channels within 1.5\,cm of that chord add to the CES past when a
> step must be filled without the next CES measurement. A two-branch LSTM reads the fast channels (standardized
> with each discharge's own statistics, an offline step) and the CES past for \Ti{}, and only the CES past for
> \Vrot{}. For \Ti{}, the mean-squared-error skill over a causal Gaussian-process smoother of the CES past is
> $+0.113$ $[+0.085, +0.132]$ (95\% interval, eight re-partitions of 639 discharges); over PCHIP
> interpolation it is $+0.221$ $[+0.188, +0.248]$, about half of which a one-parameter running mean of past
> CES already provides; against an offline Gaussian process it is $-0.010$ $[-0.043, +0.014]$. The fast
> channels enter as a same-instant 100\,ms high-pass response, and a 20-parameter surrogate filter built on it
> keeps 72--78\% of the PCHIP-relative skill. The gain sits in the tail: on the median row the error equals
> persistence's. For \Vrot{}, fast-channel increments show no association at any lag or window from 10 to
> 200\,ms (\Ti{}: $+0.15$ to $+0.28$); the history-only estimate has skill [pending A1] over persistence,
> while its registered skill over PCHIP is carried by a few rows anchored on fit failures. One device, one
> campaign, one chord.

### 6.2 §4.2 교체 초안 (E7, 영문)

> Both equations constrain a rate. Of the terms in~\eqref{eq:momentum}, NBI and NTV torque have no channel
> here (Mirnov $dB/dt$ cannot see a static $\delta B$), BES intensity is not the momentum flux
> $\langle\tilde v_r\tilde v_\varphi\rangle$, and residual stress needs $\nabla T_i$. Empirically the fast
> channels correlate with $d\Vrot/dt$ at $r=-0.006$ (BES) and $-0.003$ (ECEI) against $+0.070$ and $+0.078$
> with $d\Ti/dt$ (levels: \Ti{} $+0.341$, $+0.311$; \Vrot{} $+0.027$, $+0.005$), and \Vrot{} increments
> show no association at any lag or any window from 10 to 200\,ms while the same statistic gives $+0.15$ to
> $+0.28$ for \Ti{} [noise-corrected bound: pending A3]. A model whose shared encoder lets the fast channels
> reach \Vrot{} loses \Vrot{} skill on all four splits (paired $-0.044$, $-0.072$, $-0.067$, $-0.023$), which is
> why the branches are split at the encoder. For \Ti{} the fast-channel response is a same-instant
> co-movement of channels within 1.5\,cm of the chord (\S\ref{sec:model:eq}); its mechanism is not
> identified here, since per-discharge standardization removes the absolute $T_e$ and $n_e$ that collisional
> exchange would need and the files carry no beam-power or D$_\alpha$ signal.

**쪽수.** 삭제(Fig. 3 전폭 그림, Eq. bsimple, Table 1, τ_eq, methods lineage, claims 목록, 결론 압축)로 약
0.5쪽이 비고, 추가(공개 문장, 분해, V_rot 민감도, Table 1 두 행, 감사)는 그보다 작다. 본문 8쪽 안에 들어간다.

## 7. 이후 작업 (우선순위)

1. **[GPU, 최우선, 약 45–60분]** seq_v2를 `CES_PER_SHOT_NORM=0`(학습 파일 통계, 완전 인과)으로 GATE_ENV 아래
   4분할 × 2모집단 + temporal split에서 다시 돌리고 per-shot backbone과 paired로 비교한다. EWMA/expanding
   표준화(코드 추가 + test)는 그 다음이다. 결과가 유지되면 제목·초록에 "causal"을 복원한다. GPU가 비었을 때만
   실행하고 CPU 학습으로 대체하지 않는다.
2. **[GPU, 약 45분]** history-only seq_v2 재학습(T_i 분기에 7개 slow 채널만), 4분할 × 2모집단. 주 모델 위의
   fast-diagnostic 증분을 직접 재고, Table 3(windowed control zeroing)을 대체한다.
3. **[GPU]** surrogate(b28h)를 B.27c 재분할(r2–r9) 또는 보류 shot에서 다시 맞추고 채점한다(형태를 고르지 않은
   곳). post hoc 딱지를 떼기 위함이다. inference로 surrogate–backbone 예측 fidelity R²도 낸다.
4. **[CPU/GPU inference]** 실제 gap 길이 분포에서 뽑은 인공 masking 곡선(k = 1, 2, 5, 10, 20 step), 모든 arm.
   한 방전의 T_i/V_rot 예측 trace 그림(관측·결측·hold·fit failure 표시)으로 Fig. 1을 대체한다.
5. **[데이터 요청, KSTAR CES 팀/KFE]** 노출별 fit χ²·광자 수(3 keV와 V_rot 값 규칙을 대체하고 "photon"
   검증), CES_VT18 hold 원인, NBI 출력과 Dα(사건 태깅, beam-state 교란), EFIT 매핑(r/a), 실시간 CES 가용성.
   2023/2024 캠페인으로 동결 프로토콜 1회 외부 검정(OSF 등록).
6. **[프로토콜, 사용자 결정 필요]** A1 결과에 따라 V_rot fit-failure 규칙을 미리 등록할지 정한다(이력 입력
   포함 모든 arm에서 결측 처리 후 재학습). 2026-08-16 "V_rot uncut" 결정의 재검토 사안이다.
7. memoryless MLP/GBM(같은 22 입력, Diag2Diag형)과 TRAIN에서 맞춘 AR(p)·Kalman/OU 행을 Table 1에 넣는다.
8. V_rot을 30–50 ms 평균 라벨에서 다시 검정한다(잡음 바닥이 낮은 척도).
9. 투고 준비: FED(elsarticle) 1차 목표로 정한다. KSTAR 공동저자나 감사, 데이터 진술을 갖춘다.
   THESIS_RESULTS.md 대신 정리된 보충 PDF를 만든다. ML4PS 4쪽판과 KPS/KNS 초록도 준비한다.
10. 동기화: THESIS_RESULTS §0(주장 2, −0.42, +0.26, photon 문구)과 `collect_paper_numbers.py` 전사 점검
    3건을 갱신하고 `main_short_ko.tex`에 반영한다. memory 노트("V_rot is positive", "main theme")를 A1 결과에
    맞춰 갱신한다.
