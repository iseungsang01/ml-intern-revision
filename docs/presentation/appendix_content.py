# -*- coding: utf-8 -*-
"""Shared content for the two appendix blocks that `build_pptx.py` (1시간 덱) and
`build_pptx_jolnon.py` (졸논정리 형식 덱) both render.

부록 ①  시도한 모델의 계보와 닫은 이유 — THESIS_RESULTS.md §8의 판정을 한 표로 모은 것.
부록 ②  2026-09-05 문헌 조사 — 핵융합 / 일반 시계열·센서 예측 / 구조 동형 분야, 그리고
        그로부터 도출한 다음 실험 팔의 우선순위.

두 덱이 같은 문장을 쓰도록 여기 한 곳에만 둔다(README "Non-negotiables": 상수를 복사한
덱은 조용히 어긋난다). 모든 셀은 짧은 서술구이며 수치는 §8의 표에서 그대로 옮겼다.
글꼴에 없는 기호(U+2212 · ⟨⟩ · ⋯)는 쓰지 않는다.
"""

# --------------------------------------------------------------------------
# 부록 ① 시도한 모델과 닫은 이유
# --------------------------------------------------------------------------
TRIED_HEAD = ["모델 · 시도", "무엇을 바꿨나 (통제 변수)", "결과", "닫은 이유 · 근거"]

# (1/3) 윈도 계열 — W = 4 시대, 잠정 수치
TRIED_WINDOW = [
    ["AutoML 윈도 계열 iter2 → iter9 (GRU + 관측마스킹 multi-head attention)",
     "18회 자동 탐색. Pre-LN · attention pooling은 채택, 용량 확대 · skip 변형 · 국소 1D conv는 열화",
     "iter7의 val 0.483이 최선이었고 이후의 복잡도 증가는 열화하였다",
     "W = 2 윈도 대조군으로 강등 (§8u · §8x)"],
    ["연속시간 인코더 4종 (GRU-D형 감쇠 · ODE-RNN · Neural CDE · Δt 대각 SSM)",
     "이력 인코더만 교체, 4분할 짝지음 bootstrap",
     "KEEP 없음. NCDE · SSM은 유의 열세 seed가 생겼다",
     "규칙 10 ms 격자에서 Δt 진화는 정보를 더하지 않는다 (§8e, 코드 제거)"],
    ["파생 Mirnov 특징 (적분 · PCHIP 적분 · |MC| · 이동 RMS)",
     "입력 특징만 추가",
     "4 seed 짝지음에서 전부 무효",
     "앨리어싱으로 이미 잃은 정보는 하류에서 복원되지 않는다 (§8b.2)"],
    ["윈도 크기 스윕 W = 0 ... 6 (24 run × 2)",
     "과거 CES 관측 수",
     "Tᵢ는 W = 2에서 plateau, history 0은 붕괴",
     "W = 4는 정당화되지 않아 확정 프로토콜은 W = 2 (§8f)"],
    ["명명항 anchor + Δ (1,258 파라미터)",
     "완전히 설명 가능한 선형 항만 남김",
     "W = 4 여백의 31.5 %를 회복",
     "W = 2에서는 persistence로 붕괴 (§8k · §8z)"],
    ["W-SLIM (25,602 파라미터로 축소한 윈도 모델)",
     "CES_MODEL_FILE 하나",
     "Tᵢ -0.087, 4/4 유의 열세. V_rot는 무차이",
     "구조가 값을 하므로 비채택 (§8ad)"],
]

# (2/3) 시퀀스 계열 — 확정 프로토콜(W = 2 · held-free · 두 모집단)
TRIED_SEQ = [
    ["seq v1 공유 LSTM (전체 격자 + 마스킹 손실)",
     "프레이밍만 교체",
     "Tᵢ 4/4 개선, V_rot 4/4 유의 열세",
     "라우팅 부재. v2의 두 갈래 설계로 이어짐 (§8d)"],
    ["seq_v2 두 갈래 인과 LSTM + shot별 표준화 (357,570)",
     "V_rot 갈래를 인코더에서 차단 + 표준화",
     "B.1 관문 16/16 양수, 인과 GP 대비 4/4",
     "채택 백본 (§8t · §8x)"],
    ["seq v3 관측마스킹 attention 읽기 (396,930)",
     "읽기 경로만 추가, 0 초기화",
     "val +0.024* / +0.037*, TEST 4/4 양수 · 1/4 유의",
     "승격 규칙(유의 ≥ 3/4) 미달 (§8y)"],
    ["b3k8 해석 가능 칸 (21,498, K = 8 잠재)",
     "인코더를 8차원 잠재 + persistence 앵커로 압축",
     "컷 모집단에서 백본 +0.002, 포함에서 -0.16 ~ -0.21",
     "해석 칸으로만 채택하며 백본을 대체하지 않음 (§8z · §8ab)"],
    ["폭 스윕 hidden_ti 24 ... 260 (34k ... 879k)",
     "Tᵢ 인코더 폭",
     "평균 skill이 ±0.008 안에서 평평 (26배)",
     "크기 축 닫힘 (§8aa)"],
    ["최소 모델 v2m · b3m (1k ... 10k)",
     "모든 부분을 함께 축소",
     "v2m4k(3,898)가 4/4 바닥, v2m2k는 -0.036에 유의 열세 3",
     "3.9k 아래에서 skill이 무너짐 (§8ai)"],
]

# (3/3) 계열 · 문맥 · 기준선 · 가지
TRIED_MISC = [
    ["계열 비교 tcn15 · tcn63 · xfmr63 (확장 합성곱 · 밴드 어텐션)",
     "같은 reach에서 연산자만 교체",
     "±0.02 동률. 어텐션은 70 ms에서 -0.023 (3/4)",
     "계열은 skill이 아니라 비용을 정한다 (§8ag · §8ak)"],
    ["소형 합성곱 tcn8k · 3k · 2k",
     "10k 아래에서 계열",
     "크기 맞춘 LSTM 대비 +0.027 ~ +0.040 (3 ~ 4/4)",
     "10k 아래에선 conv 우세. tcn2k(1,808)도 인과 GP 4/4 (§8ai)"],
    ["대각 SSM (S4형) 사다리 6칸",
     "상태공간 연산자",
     "70 ms까지 문맥을 전환한 뒤 +0.105 천장. 같은 reach LSTM에 -0.022 / -0.044",
     "낮은 천장으로 비채택 (§8am 부록)"],
    ["reach 사다리 v2r2 ... r63 (20 ~ 630 ms)",
     "학습 시 문맥 길이",
     "50 ~ 70 ms 포화. 승리 방전 비율 0.52 → 0.66",
     "긴 문맥 축 닫힘 (§8af · §8al)"],
    ["오프라인 GP · 인과 GP 팔",
     "기준선 추가",
     "GP는 PCHIP 4/4 이기고 모델과 동률. 인과 GP는 백본이 4/4",
     "오프라인 주장의 상한 = GP 동률 (§8p · §8y)"],
    ["학습 분산 헤드 vs split conformal (Mondrian)",
     "불확실성 방식",
     "conformal이 Winkler 32/32 최선",
     "재학습이 헤드라인을 교란하므로 사후 보정 채택 (§8m)"],
    ["양자 VQC · 고정 특징맵 · 저장소 · 커널 (IonQ Forte)",
     "추정기 계열",
     "무잡음에서도 persistence에 짐(471.5 vs 449.6 eV). 22.9 s/예측, λ = 0.661",
     "하드웨어 검증 후 음성 종결 (§8ap)"],
    ["B.11 VQ 코드북 게이팅 (b3vqC, C = 4 · 8 · 16)",
     "읽기 가중치를 이산 코드로 선택",
     "사전등록 2026-09-03, val 선별 진행 중",
     "미판정 (PREREGISTRATION_B11.md)"],
]

TRIED_TAKEAWAY = ("데이터 처리(held 제거)와 프레이밍(전체 격자 마스킹 손실)이 skill을 움직였고, "
                  "아키텍처 미세 변형 · 크기 · 문맥 · 계열은 움직이지 않았다. "
                  "이 문장은 총합 MSE에 한정된다(§8aq).")

# --------------------------------------------------------------------------
# 부록 ② 문헌 조사 (2026-09-05)
# --------------------------------------------------------------------------
FUSION_HEAD = ["논문 (장치, 연도)", "문제 · 구조", "데이터 · 결과", "본 연구에 대한 의미"]
FUSION_ROWS = [
    ["Diag2Diag (DIII-D, Nat. Comm. 2025)",
     "Thomson 200 Hz → 500 kHz 시간 초해상. 무기억 MLP 236 → 952 → 80에 1 · 2차 미분 특징",
     "4,000 방전, R² 0.92, BNN 불확실성",
     "가장 가까운 문제가 가장 단순한 구조로 풀린다. seq_v2는 이미 인과 상태를 갖는다"],
    ["COMPASS 시간 초해상 (PPCF 2025)",
     "SXR · AXUV → Thomson Tₑ · nₑ. 4 × 256 MLP",
     "2,764 방전. 측정 불확실성 1/σ 가중 Huber 손실, 코어 MAPE 5 %",
     "불확실성 가중 손실은 §8aq의 꼬리 발견과 직결된다"],
    ["FusionMAE (HL-3, Comm. Phys. 2026)",
     "88채널 × 10 ms 마스크 오토인코더(MAE). 결측 채널 복원 = 가상 백업 진단",
     "복원 신뢰도 96.7 ~ 97.2 %",
     "본 과제를 MAE의 결측 채널 복원으로 보는 프레임. 사전학습 축은 미개척"],
    ["TokaMind (MAST, 2026)",
     "7M 미만 멀티모달 트랜스포머. DCT 임베딩, 5 ms 청크로 다중 샘플링률 처리",
     "11,573 방전. 미세조정 이득 전체 +0.017, 프로파일 -0.005, MHD +0.05",
     "프로파일류에는 사전학습이 거의 도움이 되지 않는다"],
    ["PanoMHD (KSTAR, 2026)",
     "Mirnov 스펙트로그램의 VQ-VAE 토큰 + 457M GPT-2",
     "978 방전, βN R² 0.987",
     "KSTAR에서 이산 코드북이 작동한 선례. 단 원시 2 MHz MC가 전제(B.6)"],
    ["TokaFormer · EAST 지식증류 (2025 ~ 26)",
     "붕괴 예측. 1D conv 임베딩 → 모달리티별 Transformer 후기 융합",
     "20 μs Mirnov와 1 ms 스칼라의 융합",
     "CNN + Transformer가 주류인 곳은 붕괴 예측이며 회귀 천장 문제가 아니다"],
    ["WEST (2026)",
     "사전 방전 설정 → 전역 파라미터. Transformer / LSTM / MLP 비교",
     "550 방전. MSE 0.0096 / 0.015 / 0.022",
     "작은 데이터에서 Transformer가 이긴 사례이나 10 ~ 70 s 장기 의존 과제이다"],
    ["EAST XCS ANN (NF 2024)",
     "X선 결정 분광 스펙트럼 → Tᵢ · 회전. 2 × 9 DNN, 31k CNN",
     "530 방전, R² 0.96, 시간 이력 없음",
     "회전을 실제로 추론한 선례이며 입력은 도플러 분광이다. 우리의 V_rot 경로가 어떤 입력을 확보해야 하는지를 가리킨다"],
    ["TCV 멀티모달 VAE (2025) · Abbate 물리 + 데이터 메타학습 (NF 2025)",
     "상태 감시용 순차 VAE · 물리 모델로 외삽",
     "1,600 방전 · 외삽 5 ~ 10 % 개선",
     "VAE의 용도는 해석 · 이상탐지이고, 물리 모델은 외삽을 맡는다"],
]

GENERAL_HEAD = ["방법군", "대표 (연도)", "문헌의 주장", "본 데이터에 대한 판정"]
GENERAL_ROWS = [
    ["결측 대체 (imputation)",
     "TSI-Bench 28개 (2024) · STDiff (2025)",
     "점 결측에선 선형보간이 딥러닝 전부를 이기는 데이터셋이 있고 블록 결측에서만 딥이 필요하다. 윈도 기반은 긴 공백에서 평탄화된다",
     "GP 동률(§8p)의 문헌 재확인. seq_v2는 이미 상태전이형이다"],
    ["불규칙 샘플링",
     "Neural CDE · mTAN · ContiFormer · iTimER (2025)",
     "연속시간 ODE와 어텐션이 주류",
     "격자는 규칙 10 ms이고 결측만 있다. staleness 채널로 처리 완료 (§8e)"],
    ["예측 백본 논쟁",
     "DLinear vs PatchTST · iTransformer · TimeXer (2024), Mamba 혼합, xLSTM",
     "계열 우열이 데이터마다 뒤집힌다",
     "B.9와 정합. 150 ms · 10k 파라미터 위에선 계열이 무관하다"],
    ["파운데이션 모델",
     "Chronos-2 (120M) · TiRex-2 (xLSTM, 스트리밍) · TabPFN-TS (11M) · Toto",
     "제로샷으로 강하다. 단 벤치마크 오염 경고(데이터셋 93 %가 어딘가의 사전학습에 쓰임)",
     "KSTAR는 오염이 불가능하다. 학습 없는 대조군으로 가치가 있다"],
    ["자기지도 사전학습",
     "TimeMAE (2026) · Ti-MAE · TokEye (핵융합, 2026)",
     "라벨이 희소할 때 이득",
     "Tᵢ 라벨은 92 % 관측이고 TokaMind의 프로파일 이득은 음수이다"],
    ["소프트센서 (공정 산업)",
     "반지도 VAE 회귀 · Soft-Sensing Transformer · ConFormer",
     "반지도가 핵심. MLP 백본이 Transformer와 동급인 보고도 있다",
     "Tᵢ에는 반지도 이득 조건이 없다. V_rot는 라벨이 아니라 입력을 늘리는 문제로 남아 있다"],
    ["MoE · 레짐 게이팅",
     "RG-ResMoE (2026) · Dynamic TMoE",
     "동결 기저 + 잔차 전문가 + soft 게이트. hard top-1 라우팅은 유의하게 열세",
     "B.11(hard argmin VQ)의 직접 선례. soft 게이트 자매 팔이 필요하다"],
    ["학습 상태추정",
     "KalmanNet · Bayesian KalmanNet (2025) · DLFM (2025)",
     "프로세스 모델은 두고 칼만 이득만 학습한다",
     "'persistence 신뢰 / 보정' 스위치의 원리적 형태이다"],
    ["이분산 · 강건 회귀",
     "가우시안 NLL 병리 (2023) · β-NLL · Student-t 헤드",
     "예측 분산이 평균의 그래디언트를 오염시켜 stop-grad가 필요하다",
     "§8aq가 지목한 유일하게 열리지 않은 축이다"],
]

ISO_HEAD = ["분야", "동형 관계", "주류 방법", "교훈"]
ISO_ROWS = [
    ["혼합주기 나우캐스팅 MIDAS (경제)",
     "고빈도 입력 → 저빈도 타깃",
     "U-MIDAS + 소형 NN 라그 가중 · GP-MIDAS (2024)",
     "파라메트릭 라그 구조로 충분하다. reach 사다리 · b3k8과 정합"],
    ["저가 센서 보정 (대기질, 2025 ~ 26)",
     "밀집 저가 센서 → 희소 기준 측정",
     "LSTM · GBM. 기준국에서 먼 센서는 도메인 적응",
     "캠페인 이동(§8n)과 shot별 표준화(§8s)가 같은 처방이다"],
    ["커프리스 혈압 PPG (2025 ~ 26)",
     "연속 파형 → 간헐 기준값",
     "CNN + Transformer 병렬 등. 보정 기반 > 보정 없음",
     "개체별 보정이 구조보다 크다 (§8s와 동일)"],
    ["구조 가상 센싱 SHM (2025 ~ 26)",
     "희소 센서 → 전체장",
     "칼만 · 물리유도 그래프 칼만 · PDE 제약 최적화",
     "상태추정 프레임이 주류이다 (§8t 재확인)"],
    ["멀티태스크 가상센서 (2026)",
     "수백 타깃 통합",
     "입력 선택을 학습하는 단일 모델",
     "타깃이 둘인 본 연구와 무관하다"],
    ["합성 진단 증강 (EAST → J-TEXT, 2026)",
     "데이터 없는 장치",
     "NIMROD 합성 진단 + 도메인 적응, 조기 경보 50 → 57 %",
     "NBI가 없으면 합성 토크 채널은 시뮬레이션에서만 온다"],
]

PRIORITY_HEAD = ["순위", "실험 팔 (통제 변수 하나)", "출처", "비용", "사전등록 판정 지표"]
PRIORITY_ROWS = [
    ["1", "seq_v2 손실을 σ_meas 가중 Huber 또는 Student-t NLL로 교체",
     "COMPASS · Toto · §8aq", "재학습 4 × 3 seed", "벌크 MAD skill 개선, 총합 MSE 비열세"],
    ["2", "B.11 자매 팔: soft 게이트 + persistence 축소 페널티, 기저 = 인과 GP",
     "RG-ResMoE", "b3k8 위 소형", "조용 3분위 승률 +0.05"],
    ["3", "KalmanNet 팔: 프로세스 = 인과 GP, 관측 = 고속 15채널, 이득 = 소형 GRU",
     "KalmanNet 계열", "신규 코드", "조용 3분위 승률 + 구간 보정"],
    ["4", "V_rot 입력 확장: 원시 kHz Mirnov 특징(B.6 A1) · NBI 토크 채널 · 도플러 분광 입력",
     "EAST XCS · PanoMHD · 합성 진단 증강", "데이터 획득", "변동 3분위 V_rot 승률이 먼저 오를 것"],
    ["5", "제로샷 대조군: TabPFN-v2 회귀(행 특징) · TiRex-2 / Chronos-2(과거 공변량)",
     "fev-bench · TSI-Bench", "학습 없음", "인과 GP 대비 4/4"],
    ["6", "FusionMAE식 사전학습 후 미세조정",
     "FusionMAE · TokaMind", "GPU 필요", "기대 이득 ≈ 0, 순위 최하"],
]

NOT_RECOMMENDED = [
    "CNN 뒤 Transformer와 VAE 잠재는 표현력 축이며 세 번 닫혔다(§8aa · §8ag · §8aq).",
    "Neural CDE 계열은 규칙 10 ms 격자에서 이미 음성으로 판정되었다(§8e).",
    "확산 대체는 추론이 느리고 점 결측 벤치마크에서 선형보간에 진다(TSI-Bench).",
]

VROT_NOTE = [
    "V_rot는 열린 과제이며 닫힌 결론이 아니다.",
    "§8ar의 항별 감사는 현재 입력이 닿는 항이 없음을 보였을 뿐이다.",
    "그것을 뒤집을 측정이 함께 지목되어 있다: 원시 kHz Mirnov(B.6, shot 집합 동결),",
    "NBI 토크 채널, CES 피팅 품질 메타데이터, 그리고 도플러 분광 입력의 확보이다.",
]

SURVEY_SOURCES = [
    ("Diag2Diag", "https://arxiv.org/html/2405.05908v4"),
    ("Diag2Diag Nat. Comm.", "https://www.nature.com/articles/s41467-025-63492-1"),
    ("COMPASS 시간 초해상", "https://iopscience.iop.org/article/10.1088/1361-6587/ae72c6"),
    ("FusionMAE", "https://arxiv.org/abs/2509.12945"),
    ("TokaMind", "https://arxiv.org/html/2602.15084v2"),
    ("TokaMark", "https://arxiv.org/html/2602.10132v2"),
    ("PanoMHD", "https://arxiv.org/html/2603.02672"),
    ("TCV 멀티모달 VAE", "https://arxiv.org/abs/2504.17710"),
    ("EAST 지식증류", "https://arxiv.org/html/2607.04241"),
    ("WEST", "https://arxiv.org/html/2602.19110"),
    ("EAST XCS ANN", "https://iopscience.iop.org/article/10.1088/1741-4326/ad73e8"),
    ("Abbate 2025", "https://iopscience.iop.org/article/10.1088/1741-4326/adc283"),
    ("합성 진단 증강", "https://arxiv.org/abs/2606.08462"),
    ("TSI-Bench", "https://arxiv.org/abs/2406.12747"),
    ("STDiff", "https://arxiv.org/html/2508.19011v1"),
    ("iTimER", "https://arxiv.org/html/2511.06854"),
    ("TimeXer", "https://github.com/thuml/TimeXer"),
    ("TiRex-2", "https://arxiv.org/html/2607.01204v1"),
    ("TabPFN-TS", "https://arxiv.org/abs/2501.02945"),
    ("TSFM 벤치마크 경고", "https://arxiv.org/html/2510.13654v1"),
    ("TimeMAE", "https://arxiv.org/html/2303.00320v4"),
    ("TokEye", "https://arxiv.org/pdf/2602.20317"),
    ("RG-ResMoE", "https://arxiv.org/html/2608.12251v1"),
    ("KalmanNet", "https://dl.acm.org/doi/abs/10.1109/tsp.2022.3158588"),
    ("DLFM", "https://link.springer.com/article/10.1007/s10994-025-06824-y"),
    ("이분산 회귀 병리", "https://arxiv.org/pdf/2306.16717"),
    ("반지도 VAE 회귀", "https://arxiv.org/pdf/2211.05979"),
    ("GP-MIDAS", "https://arxiv.org/html/2402.10574v2"),
    ("저가 센서 보정", "https://arxiv.org/html/2604.21527v1"),
    ("커프리스 혈압 벤치마크", "https://arxiv.org/html/2602.04725v1"),
    ("PiGGO", "https://arxiv.org/pdf/2604.26593"),
    ("멀티태스크 가상센서", "https://arxiv.org/abs/2601.20634"),
]


def sources_note(prefix=""):
    """One string for the speaker notes: every survey source, one per line."""
    lines = [prefix.strip()] if prefix else []
    lines.append("조사일 2026-09-05. 출처:")
    lines += ["- %s: %s" % (name, url) for name, url in SURVEY_SOURCES]
    return "\n".join(lines)
