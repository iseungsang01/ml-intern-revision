# 우리 결과의 관련성을 물리로 말하기 — 열 개의 증명과 하나의 시간영역 모형

작성 2026-10-13. 열 개의 독립 증명 에이전트(P01–P10)가 각각 연구의 한 결과를 제1원리에서 유도하고
641방전 CSV·얼린 TEST 산출물·몬테카를로로 검산했다. 이 문서는 그 증명들을 **하나의 시간영역 생성
모형** 아래 묶고, "이 연구의 결과가 왜 그런 값이며 무엇에 관련되는가"를 물리의 언어로 적는다.
앞선 두 문서(`theory_foundations.md`: 측정·추정·통계, `theory_plasma_physics.md`: 플라즈마 거동)가
도구를 만들었다면, 이 문서는 그 도구로 **결과를 설명**한다. 어떤 모델도 학습·채점·선택하지 않았다.
각 증명의 스크립트와 출력은 세션 scratchpad `pNN/`에 있고 핵심 수치는 본문 표에 전사했다.

**읽는 순서.** 0장(통일 모형과 결과 지도) → 1장(증명 P01–P10 전문) → 2장(종합: 관련성의 서사와
남은 측정). 0장만 읽어도 결론은 닫힌다.

---

## 0. 하나의 모형, 열 개의 결과

### 0.1 시간 $t$의 생성 모형

pedestal 한 자속면의 CES 타깃 $y\in\{T_i,V_\varphi\}$는 보존식을 선형화한 **1차 완화 + 확률 구동 + 사건
충격**이다(P09 식 (1), P01 §1.1, P08 §1′):
$$
\boxed{\;\frac{dy}{dt}=-\frac{y(t)-m}{\tau}+\sqrt{\frac{2\sigma_s^2}{\tau}}\,\xi(t)+\sum_k A_k\,\dot h(t-t_k),\qquad
h(s)=\Theta(s)\,e^{-s/\tau_{\rm rec}}\;}
\tag{M1}
$$
$$
y(t)=m+\!\int_{-\infty}^{t}\!e^{-(t-s)/\tau}\sqrt{\tfrac{2\sigma_s^2}{\tau}}\,\xi(s)\,ds+\sum_kA_k\,h(t-t_k),
\qquad
\tilde y(t_j)=y(t_j)+n_j,\quad t_j=j\Delta,\ \Delta=10\ \mathrm{ms}.
\tag{M2}
$$
두 타깃의 차이는 식의 **형태**가 아니라 **계수가 어디서 오는가**다.

| | $T_i$ (에너지 보존) | $V_\varphi$ (각운동량 보존) |
|---|---|---|
| 완화 $\tau$ | $\tau_{\rm eq}\parallel\tau_{E,i}$; 데이터 103–140 ms, IPB98 77 ms | $\tau_\varphi\approx\tau_E$; 데이터 89 ms ⇒ Pr ≈ 1 |
| 평형 $m$ | $T_e$·$n_e$·$P_{\rm NBI,i}$의 함수 — **ECEI·BES가 본다** | 토크 $T_{\rm NBI}+T_{\rm NTV}+T_{\rm int}$의 함수 — **데이터에 없다** |
| 충격 $A_k h$ | ELM 붕괴: ms 이하 계단, $\kappa_T\approx1.03$ (P08) | 토크 변화: 램프 $1-e^{-s/\tau_\varphi}$, $\kappa_V=\Delta/2\tau_\varphi=0.056$ (P08) |
| 사건 가시성 $p_{\rm obs}$ | $T_i$ 급변 중 ECEI 급변 동반 농축 1.27–1.34배 (P08) | 0.99배 = 우연 (P08) |
| 잡음 $\sigma_n$ | 광자 통계 39–46 eV, $N_{\rm ph}\approx300$ | 3–4 km/s, 같은 광자 예산 |

관측 모형은 세 층이다(P04, P07): 타깃은 절대 단위($T_i\propto(\sigma_\lambda/\lambda_0)^2$, $D$ 무관),
빠른 진단은 $x_f(t)=g(D_f)s_f(t)+c(D_f)$ (날짜 $D$의 계측 이득 × 플라즈마 상태), 관측 지시자는
$m_j=\mathbf 1[N(t_j)>N_c,\ B(t_j)<B_c]$로 광자 예산 $N(t)\propto n_b(t)n_C(t)$, 배경
$B(t)=B_0+B_1e^{-(t-t_0)/\tau_B}$($\tau_B\approx25$–35 ms 실측)의 함수다.

### 0.2 추정기의 오차를 시간으로 쓰면

사건 시각 $t_k$를 기준으로 $s=t-t_k$, $q=e^{-\Delta/\tau_{\rm rec}}$, 간극 $g=t-t_{\rm last}$,
$\rho(g)=e^{-g/\tau}$:
$$
\text{이력만(인과 필터)}:\quad e_b(t)=e_{\rm OU}(t)+\sum_kA_k\Big[\mathbf 1_{[0,\Delta)}(s)+(1-q)e^{-s/\tau_{\rm rec}}\mathbf 1_{s\ge\Delta}\Big],\qquad
E[e_{\rm OU}^2](g)=\rho(g)^2P+\sigma_s^2(1-\rho(g)^2)+\sigma_n^2 ;
\tag{M3}
$$
$$
\text{양측 보간(교량)}:\quad E[e_{\rm br}^2]\approx\tfrac32\sigma_n^2\ (\text{조용 구간}),\qquad e_{\rm br}\approx A_k/2\ (\text{사건 행});
\tag{M4}
$$
$$
\text{나우캐스터(이력+빠른 진단)}:\quad e_m(t)=e_{\rm OU}(t)+\varepsilon_b(t)+\sum_k(A_k-\xi_k\hat A_k)\,g(s),\qquad
\xi_k\sim\mathrm{Bern}(p_{\rm obs}),\ E[\varepsilon_b^2]=c_b .
\tag{M5}
$$
짝지은 차이를 시간 적분하면(P08 §1′)
$$
\boxed{\;D=\frac1T\!\int_0^T\!E[e_b^2-e_m^2]\,dt=p_{\rm obs}\,r\,\frac{N_e\kappa A^2}{n}-c_b\;}
\tag{M6}
$$
이고, 같은 것을 SSE 비중으로 쓰면 정확한 항등식(P01 §1.4)
$$
\boxed{\;\mathrm{skill}=f_I\,r_I-(1-f_I)\,\ell_B\;}
\tag{M7}
$$
— 천이 행의 기준선 SSE 비중 $f_I$ × 그 행의 회수율 $r_I$, 빼기 본류 손실. 이 두 식이 아래 결과 전부를 담는다.

### 0.3 결과 지도 — 무엇이 어느 항에서 나오는가

| 연구 결과 (원장 §) | 물리적 원인 (식) | 증명이 잰 수치 | 증명 |
|---|---|---|---|
| $T_i$ 헤드라인 +0.24 vs PCHIP, 4/4 (§8ab) | (M7)의 $f_Ir_I$: ELM 붕괴가 $T_e$·$n_e$와 **동시**에 오므로 사건 행에서 교량 $A/2$ 오차를 모델이 $(1-\kappa)A$로 줄임 | $f_Ir_I$ = 헤드라인의 90–114 %; 조용한 절반에서는 4/4 교량에 짐; $\Delta T_i$–$\Delta$ECEI 상관 0.14 → 천이 조건부 0.24, lag 0만 | P01 |
| $V_\varphi$: 보간과 동률, persistence 우위, 중앙값 −49 % (§8ab, §8ax) | 토크 항이 전부 미관측 ⇒ $\hat V(t)=E[V\mid\tilde V_{\le t_{\rm last}}]$ (OU 필터); 교량에 모든 간극에서 짐; 양의 집계는 꼬리 부등식 $G_T>D_B$ | 중앙값 하한 0.921×pers vs 모델 1.14; 상위 1 %(105행)가 skill 전부(+0.414 − 0.024); $\Delta V$–$\Delta$BES/ECEI/MC 상관 10–200 ms 전 척도 0 (±0.014), $T_i$ 대조 +0.15…+0.28; **PR2 폴백 43 %는 집계 인공물(실제 0.3–0.5 %)** | P02 |
| 문맥 포화 50 ms (§8al) | 냉시동 Kalman 수렴 $\propto(\rho(1-K))^{2k}$는 2–3 스텝에서 포화 ⇒ **OU만으로는 20 ms 결손의 26–43 %**; 나머지는 ELM 회복 $\tau_{\rm rec}$와 $\tau_{\rm eq}$(회복 형태 식별에 $t_{\rm ctx}\sim1.5\tau_{\rm rec}$ 필요) | 8al 곡선 대비 RMS 0.021–0.026(곡선 자체 0.032); 측정 감쇠 0.559/스텝(기억 3.4 스텝); $\tau_{\rm eq}$ 18–39 ms; 회복형 사건 n=2,783(1.3/s) $\tau_{\rm rec}$ **60 ms**(전체 정의 55–64 ms); 50 ms는 $\tau_{\rm eq}$와 $\tau_{\rm rec}$ 사이 — 자릿수 정합만 주장 | P03 |
| 캠페인 이동 1.22σ/0.53σ vs 타깃 0.115σ; shot별 표준화 무비용 (§8n, §8s) | $x=g(D)s+c(D)$: 계측 이득의 날짜 의존, 타깃은 파장비라 불변; shot 내 z-score가 $g,c$를 정확히 소거 | shot 간 분산 몫 BES 0.84 / ECEI 0.91 vs $T_i$ 0.49; §8n 재현 1.218/0.529/0.115; 절대 수준의 추가 $R^2$ 1.4–2.6 % | P04 |
| 3 keV 컷, 문턱 둔감 (§8ab) | PB 자기일관 상한 $p_{\rm ped,max}$ 4.3–9.6 kPa ⇒ $T_{i,\rm ped,max}$ 1.0–1.8 keV; 3 keV는 $\beta_{p,\rm ped}$ 1.5–4.5 필요(상한 0.89); 스파이크 중앙값 5.3 keV의 압력 = PB의 3.06배 | 런 통계 §8ab 전부 재현(1,197행/951런/85.5 % 단일/13.2배); 스파이크 ΔT 5,019 eV에 필요한 파워 5.6 MW = pedestal 몫의 39배, 싱크는 정상 수송의 46배, 3.6 µs; 폭 전용 지문: 같은 행 $V_\varphi$ 점프 0.05 vs $T_i$ 5.8배(p 0.75); 문턱 2.5–4 keV에서 0.15–0.19 % 행만 이동 | P05 |
| 빠른 진단의 1-step 이득 ≈ +0.05 (foundations §7.8) | $\Delta\mathrm{skill}=r^2_{yz\cdot h}\,\mathrm{MSE}_h/\mathrm{MSE}_{\rm pers}$; 올바른 조건부는 **혁신**, 경로는 잡음 앵커의 탈잡음 | 선형 한계 +0.047–0.056, 측정 +0.048; 함의 partial r 0.26; MC 0 | P06 |
| MNAR, 독립 결측, in-domain 54–68 %/4–6 % (§8i, §8ab 9) | $m_j$가 $N(t),B(t)$의 함수 ⇒ 플라즈마 상태의 함수; 중심·폭 적합의 채택 기준이 다름 | 결측률 점프 5분위 0.77→4.45 % (×5.8); BES 최하 5분위 19.9 % vs 3.7 %; $\tau_B$ 25–35 ms; 마스크 일치 72.77 %, φ 0.041 | P07 |
| 승률 비대칭 70 % vs 48 %, 삼분위 85 vs 55 % (§8al, §8an) | (M6): $D=p_{\rm obs}rE_{\rm imp}-c_b$; $T_i$는 $p\,r\approx1$, $V_\varphi$는 $p$ 우연·$r\le0$·$\kappa_V\ll1$ | 삼분위 43/85/86 vs 36/45/59; 조용 손실 −0.072 = −0.073(=$-c_b$); 충격행 승률 0.744 vs 0.425 | P08 |
| MAST 사전학습 = 무작위 초기화 (§8az) | (M1)의 형태만 장치 무관, 7개 계수 전부 장치·계기 고유; 한 shot(4 s)이면 $\rho$를 2.4 SE로 정함 | $\beta$ 7배, $\varepsilon$ 2.75배; 전이 가중치 98.2 %가 설명할 구조 = 식 한 줄; k 3→10 상승 네 모델군 퍼짐 0.006 | P09 |
| 실시간: 10 ms 비구속, 1 ms 보류 (§8ac, §8aj) | (M3)의 지연 $\ell$: $\mathrm{MSE}(\ell)-\mathrm{MSE}(0)\approx(\ell/\Delta)[\mathrm{MSE}(\Delta)-\mathrm{MSE}(0)]$; ELM 붕괴 0.1–1 ms는 라벨 적분이 배제 | 지연 비용 −0.023/ms(강건)·−0.009/ms(고전); 등급 A(수송 제어)만 해당; 회복형 사건 $\tau_{\rm rec}$ 25–37 ms | P10 |

### 0.4 한 문단으로

이 연구의 모든 수치는 (M1)의 두 계수 집합에서 나온다. $T_i$의 평형과 충격은 ECEI·BES가 같은 pedestal
사건을 **같은 시각에** 보기 때문에 관측 가능하고, 그래서 인과 나우캐스터가 미래를 보는 보간을 **사건
행에서만** 이기며(헤드라인의 90 % 이상이 거기서 옴), 조용한 본류에서는 잡음 평균화의 산수대로 교량에
진다. $V_\varphi$의 평형과 충격은 토크에서 오고 토크는 데이터에 없으므로, 최선은 이력의 OU 필터이고
그 결과가 "보간과 동률·persistence 우위·중앙값 열세"다. 두 타깃을 가르는 것은 아키텍처도 용량도 아니라
**보존식의 우변에 관측 가능한 항이 있는가**이며, 그 사실이 라우팅·W = 2·50 ms·캠페인 표준화·3 keV
컷·MNAR 가중·실시간 등급까지 전부를 결정한다. 남는 레버는 식이 지목하는 **없는 항**이다: 보정 $T_e$,
kHz Mirnov 모드 주파수(=회전의 Doppler), NBI·RMP 파형, ELM 지시자.


---

## 1. 열 개의 증명 (P01–P10)

각 증명은 독립 에이전트가 작성했고 형식은 같다: 명제 → 유도 → 시간 $t$의 식 → 데이터 검산 → 기록과의 연결 → 한계와 반증 조건. 스크립트 경로의 `scratchpad/`는 작성 세션의 임시 폴더이며, 수치는 본문에 전사되어 있다.


### P01 — $T_i$ 헤드라인(+0.24 vs PCHIP, 4/4)은 천이 물리다: 유도, 시간 영역의 식, 항등식, 측정

**한 줄 결론.** 타깃을 "OU 배경 + 희소 충격"으로 쓰면 MSE skill은 정확히 $\mathrm{skill}=f_I\,r_I-(1-f_I)\,\ell_B$ 로 분해되고, 얼린 TEST 4분할에서 천이 항 $f_I r_I$ 만으로 헤드라인의 90–114 %가 나온다($+0.199/+0.223/+0.244/+0.265$ 대 헤드라인 $+0.174/+0.248/+0.257/+0.264$). 조용한 절반의 행에서는 모델이 4/4 분할 모두 비인과 GP에 지고($-0.09\ldots-0.31$), 상위 5 % 천이 행에서는 4/4 모두 이긴다($+0.05\ldots+0.09$). $f_I$(천이 행의 PCHIP SSE 비중, 0.72–0.78)는 타깃 자신의 제곱 변동 중 천이 행 비중(0.71–0.77)과 같다. 원시 641파일에서 $\Delta T_i$와 $\Delta(\text{ECEI})$의 Pearson 상관은 전체 $+0.142$에서 $|\Delta T_i|\ge q_{95}$ 조건부 $+0.238$로 오르고(CI 분리), lag 0에서만 산다.

**스크립트.** `scratchpad/p01\p01_transient_proof.py`(npz `is_peak`·후방 점프 층화, shot별, 원시 증분 상관)와 같은 폴더의 `p01_part2.py`(양방향 점프 $J$ 층화, GP 교량 검정, lag 구조). 출력은 `p01_run.log`, `p01_part2_run.log`, `p01_results.json`, `p01_part2_results.json`. 추적 파일은 수정하지 않았고 모델은 학습·채점하지 않았다. 아래의 모든 수치는 이 두 로그 또는 인용한 파일(`docs/theory_foundations.md`, `docs/theory_plasma_physics.md`, `THESIS_RESULTS.md`)에서 왔다.

#### 1. 유도

**1.1 생성 모형.** 이온 에너지식(`docs/theory_plasma_physics.md` §8.1, `docs/theory_foundations.md` §3.1)
$$\tfrac32 n_i\frac{dT_i}{dt}=Q_{ei}(T_e-T_i)+P_{\rm NBI,i}-\tfrac32 n_i\frac{T_i}{\tau_{E,i}}-(\text{CX})+\underbrace{\sum_k \tfrac32 n_i J_k\,\delta(t-t_k)}_{\text{ELM/천이 충격}}$$
를 밀도 고정으로 선형화하면 충격 사이에서는 1차 완화계 $dT_i/dt=(T_e-T_i)/\tau_{\rm eq}-T_i/\tau_{E,i}+s$ 이고, 백색 구동 아래 이것은 정확히 잡음 있는 OU 과정이다(foundations §7.1; 강건 적합 $\sigma_n=39.0$ eV, $\sigma_s=68.7$ eV, $\tau=81$ ms, $\rho=e^{-\Delta/\tau}=0.884$, foundations §8 표). 충격은 pedestal 붕괴(plasma §6.3: 성장률 $\sim10^5\,{\rm s^{-1}}$, 0.1–1 ms 안에 붕괴, $\Delta W/W\sim5$–20 %)로, 10 ms 격자에서는 **한 점의 급락과 수 점의 회복**으로 보인다. 같은 붕괴가 $T_e$(ECEI, foundations §2.1)와 $n_e$(BES, §2.2)를 동시에 떨어뜨리므로 타깃과 빠른 채널은
$$y(t)=x(t)+\sum_k J_k\,H(t-t_k),\qquad z(t)=\xi(t)+\sum_k J^{z}_k\,H(t-t_k),\qquad \tilde y(t)=y(t)+n(t),\ \ \mathrm{corr}(J_k,J^z_k)>0$$
로 쓸 수 있다($x$: OU 배경, $\xi$: 빠른 채널의 자체 배경, $H$: 계단, $n$: CES 광자 잡음). 핵심 가정은 **충격 시각 $t_k$ 의 공유**(동시성)이다.

**1.2 (a) 충격 사이 — 양측 교량은 근최적이고 인과 추정기는 못 이긴다.** foundations §7.3–7.4: 1-step에서 양측 보간 $\mathrm{MSE}_{\rm interp}=\tfrac32\sigma_n^2+\sigma_s^2(\tfrac32+\tfrac12\rho^2-2\rho)$, 최적 인과 필터 $\mathrm{MSE}_{\rm filter}=P^-+\sigma_n^2$ 이며 persistence 대비 비는 0.813 대 0.875. §7.7(OU 교량 결과)을 그대로 인용하면 **"이력만으로는 보간에 항상 지고 간극이 길수록 더 진다": skill(인과 vs 보간) $=-0.21$ (10 ms) $\to-0.58$ (50 ms)**. 빠른 채널이 이 구간에서 더해 줄 수 있는 것은 §3.3의 $\Delta T_i\approx(\Delta/\tau_{\rm eq})(T_e-T_i)$ 뿐이고, §7.8은 동역학만으로 얻는 이득을 $\Delta/2\tau\approx0.06$으로 상한한다. 즉 충격 사이에서 $\ell_B\ge0$ 이며, 모델이 "앵커를 그대로 옮기면"(§8ax) $\ell_B=1/0.813-1=0.23$ 까지 진다.

**1.3 (b) 충격에서 — 교량 오차는 $O(J)$, 동시 관측은 그것을 추적한다.** 격자 사이 $t_k\in(t_{n-1},t_n)$에 크기 $J$의 계단이 오면 $\tilde y(t_{n-1})\approx y^-$, $\tilde y(t_n)\approx\tilde y(t_{n+1})\approx y^-+J$. 양측 보간은 행 $t_n$에서 $\tfrac12(\tilde y(t_{n-1})+\tilde y(t_{n+1}))=y^-+J/2$ 를 내어 오차 $J/2$, 행 $t_{n-1}$에서도 $\tfrac12(\tilde y(t_{n-2})+\tilde y(t_n))$ 로 오차 $J/2$: 충격당 SSE $\approx J^2/2$ (PCHIP는 §9.1의 Hermite 기울기 항이 더해져 더 크다). 이력만의 인과 추정기는 행 $t_n$에서 $J$, 행 $t_{n-1}$에서 0: SSE $\approx J^2$. 빠른 채널을 같은 시각에 보는 인과 모델은 학습된 전이계수 $\kappa=\mathrm{Cov}(J,J^z)/\mathrm{Var}(J^z)$ 로 행 $t_n$ 오차를 $(1-\kappa)J$ 로 줄인다. 일반형으로 쓰면 각 arm의 충격 오차가 $c\,J$ ($c_{\rm interp}^2=\tfrac12$, $c_{\rm hist}^2=1$, $c_{\rm model}^2=(1-\kappa)^2$)이고
$$r_I=1-\frac{c_{\rm model}^2}{c_{\rm interp}^2}=1-2(1-\kappa)^2>0\iff\kappa>1-\tfrac{1}{\sqrt2}=0.29 .$$
천이에서는 교량도 인과도 모두 $O(J)$ 를 틀리므로, **미래 앵커가 주던 우위가 사라지고 동시 관측이 그 자리를 대신한다.** 이것이 인과 모델이 양측 보간을 이길 수 있는 유일한 자리다.

**1.4 (c) 집계 항등식.** 행을 천이 집합 $I$와 나머지 $B$로 나누고 $f_I=\mathrm{SSE}_b^{(I)}/\mathrm{SSE}_b$, $r_I=1-\mathrm{SSE}_m^{(I)}/\mathrm{SSE}_b^{(I)}$, $\ell_B=\mathrm{SSE}_m^{(B)}/\mathrm{SSE}_b^{(B)}-1$ 로 두면 SSE의 가법성만으로
$$\boxed{\ \mathrm{skill}=1-\frac{\mathrm{SSE}_m}{\mathrm{SSE}_b}=f_I\,r_I-(1-f_I)\,\ell_B\ }\qquad(\text{정확한 항등식; 일반화 }\ \mathrm{skill}=\textstyle\sum_k f_k s_k).$$
1.1의 모형에서 $\mathrm{SSE}_b^{(I)}\approx c_b^2\sum_k J_k^2$, $\mathrm{SSE}_b^{(B)}\approx N_B\cdot\tfrac32\sigma_n^2$ 이므로 $f_I$ 는 **타깃의 제곱 변동 중 충격에 사는 비중**으로 결정되고, 1.2·1.3에 의해 $r_I>0$, $\ell_B\ge0$: 여유의 크기는 $f_I$ 가, 부호는 $\kappa$ 가 정한다.

#### 시간 t의 식 (1.1–1.4의 시간 영역 형태)

아래는 위 유도를 전부 시간 $t$의 명시적 함수로 다시 쓴 것이다. 격자 간격 $\Delta=10$ ms, 관측 시각 $t_n=n\Delta$, 타깃 시각 $t$, 마지막 과거 관측 $t_{\rm last}=t-g$(간극 $g$), 다음 관측 $t_{\rm next}=t+g'$, 빠른 채널의 지연 $\lambda$ (채널이 $t-\lambda$ 까지만 보임; 나우캐스팅은 $\lambda=0$).

**(T1) 충격 사이의 배경 — 선형 완화계의 해와 임펄스 응답.** 선형화한 에너지식 $\dfrac{dT_i}{dt}=-\dfrac{T_i}{\tau_r}+\dfrac{T_e(t)}{\tau_{\rm eq}}+\dfrac{S_i(t)}{\tfrac32 n_i}$, $\ \dfrac1{\tau_r}=\dfrac1{\tau_{\rm eq}}+\dfrac1{\tau_{E,i}}$ 의 해는
$$T_i(t)=T_i(t_0)\,e^{-(t-t_0)/\tau_r}+\int_{t_0}^{t}h(t-s)\,u(s)\,ds,\qquad h(t)=\frac{1}{\tau_{\rm eq}}\,e^{-t/\tau_r}\,\mathbf 1[t\ge0],\qquad u(s)=T_e(s)+\tau_{\rm eq}\,\frac{S_i(s)}{\tfrac32 n_i}.$$
$h$의 Fourier 변환이 foundations §3.3의 $H(\omega)=(1/\tau_{\rm eq})/(i\omega+1/\tau_r)$ 이다. 구동 $u$를 백색 잡음으로 두면 $x(t)$ 는 OU 과정
$$x(t)=\bar m+\bigl(x(t_0)-\bar m\bigr)e^{-(t-t_0)/\tau}+\sqrt{\tfrac{2\sigma_s^2}{\tau}}\int_{t_0}^{t}e^{-(t-s)/\tau}\,dW(s),\qquad
\mathrm{Cov}\bigl(x(t),x(t+h)\bigr)=\sigma_s^2\,e^{-|h|/\tau},\qquad \rho(g)\equiv e^{-g/\tau}$$
(측정: $\tau=81$ ms 강건 OU, $\rho(\Delta)=0.884$; 완화 시간 103–140 ms, plasma §8.4; 자기상관 1/e 159 ms, §8ar). 관측은 $\tilde y(t)=y(t)+n(t)$, $n$ 백색, $\sigma_n=39.0$ eV.

**(T2) 충격과 회복 — 사건 $t_k$ 뒤의 곡선.** 붕괴는 0.1–1 ms $\ll\Delta$ 이므로 격자에서는 계단이고, 회복은 pedestal 재건 시간 $\tau_{\rm rec}\sim\tau_{\rm build}=10$–50 ms(plasma §6.3)로 지수적이다:
$$y(t)=x(t)+\sum_k\delta_k(t),\qquad \delta_k(t)=-J_k\,e^{-(t-t_k)/\tau_{\rm rec}}\,\mathbf 1[t\ge t_k]\quad(\tau_{\rm rec}\to\infty\ \text{이면 1.1의 계단}\ -J_kH(t-t_k)),$$
$$z(t)=\xi(t)+\sum_k\delta^z_k(t),\qquad \delta^z_k(t)=-J^z_k\,e^{-(t-t_k)/\tau^z_{\rm rec}}\,\mathbf 1[t\ge t_k],\qquad J_k=\kappa J^z_k+\epsilon_k .$$
같은 $t_k$ 가 $y$와 $z$에 들어가는 것이 동시성 가정이고, (T2)의 1차 미분 $\dot\delta_k(t)=-J_k\delta(t-t_k)+(J_k/\tau_{\rm rec})e^{-(t-t_k)/\tau_{\rm rec}}\mathbf 1[t\ge t_k]$ 가 "한 점의 급락, 수 점의 상승"이다.

**(T3) 이력만 쓰는 인과 추정기의 오차 — 간극 $g$의 함수.** 충격 사이에서 과거 앵커 하나로 만드는 최적 예측과 그 MSE는
$$\hat y_{\rm hist}(t)=\bar m+e^{-g/\tau}\,\frac{\sigma_s^2}{\sigma_s^2+\sigma_n^2}\bigl(\tilde y(t-g)-\bar m\bigr),\qquad
\mathrm{MSE}_{\rm hist}(g)=\sigma_n^2+\sigma_s^2\Bigl(1-e^{-2g/\tau}\,\frac{\sigma_s^2}{\sigma_s^2+\sigma_n^2}\Bigr),$$
정상상태 Kalman 필터(모든 과거 사용)로는 $\mathrm{MSE}_{\rm filter}(g)=e^{-2g/\tau}P+\sigma_s^2\bigl(1-e^{-2g/\tau}\bigr)+\sigma_n^2$ (foundations §7.7의 $\rho^{2g}$ 를 $e^{-2g/\tau}$ 로 쓴 것). 과거 관측 $t-s$ 의 가중은 $w(s)\propto\bigl(\rho(1-K)\bigr)^{s/\Delta}=e^{-s/L}$, $L=\Delta/[-\ln(\rho(1-K))]=12$–18 ms(§7.6), 따라서 $t-50$ ms 이전 관측의 가중은 $e^{-50/L}\le e^{-2.8}$ — §8al의 50 ms 포화. persistence는 $\hat y(t)=\tilde y(t-g)$, $\mathrm{MSE}_{\rm pers}(g)=2\sigma_n^2+2\sigma_s^2(1-e^{-g/\tau})$.

**(T4) 양측 교량의 오차 — 간극 $g,g'$ 의 함수.** 선형 교량 $\hat y_b(t)=\dfrac{g'}{g+g'}\tilde y(t-g)+\dfrac{g}{g+g'}\tilde y(t+g')$ 는 충격 사이에서($g=g'$)
$$\mathrm{MSE}_b(g)=\tfrac32\sigma_n^2+\sigma_s^2\Bigl(\tfrac32+\tfrac12e^{-2g/\tau}-2e^{-g/\tau}\Bigr)\ \xrightarrow{g\ll\tau}\ \tfrac32\sigma_n^2 ,$$
이고 $\mathrm{MSE}_{\rm hist}(g)\to\sigma_n^2+\sigma_s^2\bigl(1-\tfrac{\sigma_s^2}{\sigma_s^2+\sigma_n^2}\bigr)>\tfrac32\sigma_n^2$ 가 되는 조건($\sigma_s>\sigma_n/\sqrt2$, 측정 68.7 > 27.6 eV)에서 교량이 이긴다. 인과 대 교량의 skill을 $g$ 의 함수로 적으면 $s_{\rm hist}(g)=1-\mathrm{MSE}_{\rm filter}(g)/\mathrm{MSE}_b(g)$: 강건 적합값으로 $s_{\rm hist}(10\,{\rm ms})=-0.21$, $(20)=-0.33$, $(30)=-0.44$, $(50)=-0.58$, $(200)=-0.80$ (§7.7). 교량 자체의 MSE는 $g$ 에 거의 무관(미래 앵커가 버팀)하고 인과 MSE는 $1-e^{-2g/\tau}$ 로 커지므로 차이는 $g$ 와 함께 단조 증가한다.

**(T5) 사건 둘레의 오차 — $t-t_k$ 의 함수.** 사건 $t_k$ 가 격자 $(t_{n-1},t_n]$ 안에 있고 회복이 $\tau_{\rm rec}$ 라 하자. 잡음·배경 항을 빼고 사건 성분만 쓰면

| 추정기 | $t<t_k-\Delta$ (사건 전, 앵커 둘 다 사건 전) | $t_k-\Delta\le t<t_k$ (직전 행) | $t_k\le t<t_k+\Delta$ (직후 행) | $t\ge t_k+\Delta$ (회복 구간) |
|---|---|---|---|---|
| 양측 교량 $e_b(t)$ | 0 | $-\tfrac{g}{g+g'}\,J\,e^{-(t+g'-t_k)/\tau_{\rm rec}}\ (\to-\tfrac J2)$ | $+\tfrac{g'}{g+g'}\,J\,e^{-(t-t_k)/\tau_{\rm rec}}\ (\to+\tfrac J2)$ | 곡률항 $\tfrac{g g'}{2}\,\dfrac{J}{\tau_{\rm rec}^2}e^{-(t-t_k)/\tau_{\rm rec}}$ (작음) |
| 이력만 $e_{\rm hist}(t)$ | 0 | 0 | $J\,e^{-(t-t_k)/\tau_{\rm rec}}$ (앵커가 사건을 못 봄) | $J\bigl(e^{-(t-g-t_k)/\tau_{\rm rec}}-e^{-(t-t_k)/\tau_{\rm rec}}\bigr)\approx \tfrac{g}{\tau_{\rm rec}}J e^{-(t-t_k)/\tau_{\rm rec}}$ |
| 동시 채널 모델 $e_m(t)$ | 0 | 0 | $(1-\kappa)\,J\,e^{-(t-t_k)/\tau_{\rm rec}}$ (단, $t_k\le t-\lambda$ 일 때; $t-\lambda<t_k$ 면 $e_{\rm hist}$ 와 같음) | 이력항과 같음(채널이 회복 기울기도 알려주면 $(1-\kappa')$ 배) |

읽기: 교량은 사건 **양쪽** 두 행에서 $O(J)$ 를 틀리고(사건을 가로질러 다리를 놓음), 이력만의 인과는 **직후** 한 행에서 $J$ 를 틀리며, 동시 채널 모델은 같은 행에서 $(1-\kappa)J$ 로 줄인다. 모든 사건 성분은 $e^{-(t-t_k)/\tau_{\rm rec}}$ 로 꺼지므로, 회복 구간에서는 세 추정기가 다시 (T3)·(T4)의 OU 체제로 돌아간다. 지연 $\lambda>0$ 이면 사건이 $(t-\lambda,t]$ 에 떨어진 행에서는 모델이 사건을 못 보므로, 사건 시각이 격자 안에 균등하다고 할 때 추적 확률은 $1-\lambda/\Delta$ 이고
$$r_I(\lambda)=1-2\Bigl[(1-\tfrac{\lambda}{\Delta})(1-\kappa)^2+\tfrac{\lambda}{\Delta}\Bigr],\qquad r_I(0)=1-2(1-\kappa)^2,\quad r_I(\Delta)=-1 ,$$
즉 천이 이득은 지연에 선형으로 깎이고 $\lambda=\Delta$ 에서 이력만의 값으로 떨어진다(실시간 사다리 §8ac–§8af의 "지연 손실"이 사는 자리; 여기서는 유도만 하고 측정하지 않았다).

**(T6) 집계 skill — 한 사건 주기에 대한 시간 적분.** 사건율 $\nu$ (단위 시간당 사건 수), 평균 제곱 점프 $\langle J^2\rangle$, 충격 사이의 교량·모델 MSE를 $\mathrm{MSE}^{\rm OU}_b,\ \mathrm{MSE}^{\rm OU}_m$ 라 하면 단위 시간당 SSE는
$$\dot S_b=\underbrace{\nu\,\frac{\langle J^2\rangle}{2}}_{\text{사건(두 행, 각 }J/2)}+\underbrace{\Bigl(\frac1\Delta-2\nu\Bigr)\mathrm{MSE}^{\rm OU}_b}_{\text{본류}},\qquad
\dot S_m=\nu\,(1-\kappa)^2\langle J^2\rangle+\Bigl(\frac1\Delta-2\nu\Bigr)\mathrm{MSE}^{\rm OU}_m ,$$
$$\mathrm{skill}=1-\frac{\int_0^{T}e_m(t)^2dt}{\int_0^{T}e_b(t)^2dt}=1-\frac{\dot S_m}{\dot S_b}
= f_I\,r_I-(1-f_I)\,\ell_B,\qquad
f_I=\frac{\nu\langle J^2\rangle/2}{\nu\langle J^2\rangle/2+(\Delta^{-1}-2\nu)\,\mathrm{MSE}^{\rm OU}_b},\quad
r_I=1-2(1-\kappa)^2,\quad \ell_B=\frac{\mathrm{MSE}^{\rm OU}_m}{\mathrm{MSE}^{\rm OU}_b}-1 .$$
이것이 1.4의 항등식의 시간 영역 형태다: $f_I$ 는 사건이 단위 시간에 만드는 제곱 변동 $\nu\langle J^2\rangle$ 과 본류 잡음 $\mathrm{MSE}^{\rm OU}_b\approx\tfrac32\sigma_n^2$ 의 비로 정해지고(측정 $f_I=0.72$–0.78 은 사건 SSE율이 본류 SSE율의 2.6–3.5배라는 뜻), $\ell_B$ 는 (T3)·(T4)의 차이로 $\ge0$ 이며(모델이 앵커를 옮기면 $\ell_B=\mathrm{MSE}_{\rm pers}/\mathrm{MSE}_b-1=1/0.813-1=0.23$), $r_I$ 의 부호는 $\kappa>0.29$ 가 정한다. 사건을 상위 5 % 행으로 잡은 측정값 $r_I=+0.27\ldots+0.34$ 는 $\kappa=0.40$–0.42 에 해당한다.

#### 2. 데이터 검산 (얼린 TEST npz `data/.b1_seqv2_s{42,1,7,123}_i*/comparison_errors_test.npz`, 컷 모집단)

**2.1 `is_peak` 층화 (§8ab 5절의 플래그 그대로).** 항등식 오차는 4/4 분할에서 $\le2\times10^{-16}$.

| 분할 | 헤드라인 | $f_{\rm peak}$ | $s_{\rm peak}$ | $s_{\rm nonpeak}$ | $f_{\rm peak}s_{\rm peak}$ |
|---|---:|---:|---:|---:|---:|
| s42 | +0.174 | 0.237 | +0.450 | +0.088 | +0.107 |
| s1 | +0.248 | 0.207 | +0.504 | +0.182 | +0.104 |
| s7 | +0.257 | 0.225 | +0.613 | +0.153 | +0.138 |
| s123 | +0.264 | 0.190 | +0.538 | +0.200 | +0.103 |

peak 행은 12.7–12.9 %인데 PCHIP SSE의 19–24 %를 차지하고, 층별 skill은 §8ab 5절의 기록(+0.45…+0.61 / +0.09…+0.20)을 그대로 재현한다. 다만 `is_peak`는 이웃 기울기·분산으로 정의된 **이웃 활동도** 플래그라 조용한 행이 섞여 있어 비peak 층도 양수다.

**2.2 양방향 점프 $J$ 층화 — 유도의 (a)·(b)를 직접 검정.** npz 행이 시간순임을 확인하고(같은 shot·10 ms 연속쌍에서 $|y_{t}-y_{t-1}|$ 와 $\sqrt{se_{\rm pers}}$ 일치율 99.2–99.5 %, 전방 이웃 보유 96–97 %), $J=\max(|y_t-y_{\rm prev}|,|y_{\rm next}-y_t|)$ 로 층화했다. (T5)대로 붕괴 **직전** 행도 교량이 $J/2$ 를 틀리는 행이므로, 후방 점프만 쓰면 그 행이 "조용한" 층에 들어간다. $I$ = 상위 5 %.

| 분할 | 헤드라인 | $f_I$ | $r_I$ | $r_B$ | $f_I r_I$ | 비중 | $J^2$ 상위 5 % 비중 | 모델 vs GP: 본류 / 천이 | 인과 GP vs GP: 본류 / 천이 |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| s42 | +0.174 | 0.734 | +0.271 | −0.094 | +0.199 | 1.14 | 0.718 | −0.308 / +0.049 | −0.344 / −0.061 |
| s1 | +0.248 | 0.723 | +0.309 | +0.091 | +0.223 | 0.90 | 0.713 | −0.086 / +0.091 | −0.283 / −0.036 |
| s7 | +0.257 | 0.734 | +0.333 | +0.046 | +0.244 | 0.95 | 0.728 | −0.145 / +0.086 | −0.341 / −0.055 |
| s123 | +0.264 | 0.779 | +0.340 | −0.003 | +0.265 | 1.00 | 0.770 | −0.163 / +0.049 | −0.312 / −0.057 |

읽기.
- $f_I r_I+(1-f_I)r_B$ 는 헤드라인과 $10^{-16}$ 안에서 일치하고, 천이 항만으로 90–114 %다. 본류 항 $(1-f_I)r_B$ 는 $-0.025\ldots+0.025$.
- $f_I$(0.72–0.78) $\approx$ $J^2$ 의 상위 5 % 비중(0.71–0.77). **여유의 크기 = 천이에 사는 제곱 변동의 비중**이라는 (T6)의 결론이 수치로 성립한다.
- (a)·(T3)·(T4)의 검정: 가장 조용한 절반($J<q_{50}$, PCHIP SSE의 0.9–1.3 %)에서 모델 RMSE 53–59 eV, PCHIP 28–36 eV, 비인과 GP 37–44 eV. 모델은 4/4 분할 모두 교량에 지고(vs GP $-0.63\ldots-1.28$, vs PCHIP $-1.76\ldots-2.79$), 이력만의 인과 GP도 그렇다(vs GP $-0.87\ldots-1.05$). 모델의 본류 RMSE는 측정 잡음 바닥(§8aq MAD 46.4 eV)에 앉아 있다 = "앵커를 옮긴다"(§8ax).
- (b)·(T5)의 검정: 상위 5 %에서 모델은 PCHIP에 +0.27…+0.34, **노이즈 인지 양측 교량인 GP에도 4/4 모두 +0.05…+0.09** 이기는 반면, 이력만의 인과 GP는 GP에 4/4 모두 진다(−0.04…−0.06). 동시 관측이 미래 앵커를 대체하는 자리가 천이뿐이라는 1.3의 명제 그대로다. 상위 1 %($f$ = PCHIP SSE의 44–51 %)에서는 모델 vs GP +0.08…+0.14, 중앙값 오차 495–649 eV 대 PCHIP 1009–1458 eV.
- $r_I=+0.27\ldots+0.34$ 를 (T5)의 계단 모형에 넣으면 $\kappa=1-\sqrt{(1-r_I)/2}=0.40$–0.42. 모델은 점프의 약 40 %를 같은 시각에 추적한다.
- 전체 모델 vs GP는 $-0.053/+0.039/+0.019/-0.006$(동률, §8p) = 천이 이득 − 본류 손실이고, 헤드라인 vs PCHIP 의 나머지(GP vs PCHIP +0.215…+0.269)는 PCHIP의 잡음 통과(foundations §9.1–9.2)다.

**2.3 shot별 — 여유의 크기가 천이 비중을 따른다.** shot별 skill(vs PCHIP)과 그 shot의 $f_I$ 의 Spearman $\rho$ = +0.531 / +0.274 / +0.319 / +0.538 (p = 1.5e-7 / 8.9e-3 / 1.8e-3 / 3.9e-8, 86–93 shot). 천이 비중 3분위 승률: 조용 0.24/0.40/0.29/0.17 → 중간 0.64/0.73/0.77/0.63 → 변동 0.76/0.77/0.65/0.81; 중앙 skill 조용 −0.20…−0.28, 변동 +0.20…+0.24. 반면 `is_peak` 비율과의 $\rho$ 는 −0.26…+0.09(3/4 n.s.). §8an 2절("peak fraction은 변수가 아니다")의 재현이다.

**2.4 원시 CSV 증분 상관** (641파일; 블록 = `time` 간격 ≥ 0.5 s; $T_i>3000$ eV → NaN 1,197행; held 제거 1행; shot별 표준화 ECEI·BES 평균; 같은 블록의 10 ms 연속 관측쌍 216,895개). $|\Delta T_i|$ 의 $q_{50}/q_{95}/q_{99}$ = 44.3 / 365.7 / 876.5 eV; $\sum\Delta T_i^2$ 의 상위 5 % 비중 0.782, 상위 1 % 0.538.

| 조건 | n | $r(\Delta T_i,\Delta{\rm ECEI})$ [95 % CI] | $r(\Delta T_i,\Delta{\rm BES})$ [95 % CI] | 부호 일치 E / B | lag −1 / 0 / +1 (ECEI) |
|---|---:|---|---|---|---|
| (i) 전체 | 216,895 | **+0.142** [+0.137, +0.148] | **+0.143** [+0.131, +0.154] | 0.57 / 0.59 | −0.029 / +0.142 / −0.099 |
| $|\Delta T_i|<q_{50}$ | 108,447 | +0.080 | +0.070 | 0.51 / 0.54 | −0.010 / +0.081 / −0.025 |
| (ii) $|\Delta T_i|\ge q_{95}$ | 10,845 | **+0.238** [+0.219, +0.257] | **+0.155** [+0.131, +0.178] | 0.67 / 0.67 | −0.076 / +0.237 / −0.178 |
| $|\Delta T_i|\ge q_{99}$ | 2,169 | +0.122 | +0.126 | 0.55 / 0.57 | — |

ECEI는 예측대로 조건부가 1.7배 크고 CI가 분리된다(Spearman 0.232 → 0.298). 상관은 lag 0에서만 살고 ±1에서는 0 또는 음수다. (T2)의 **같은 $t_k$** 가정, 즉 동시 붕괴다. BES는 조건부 상승이 작고(+0.143 → +0.155, CI 겹침) 부호 일치만 0.59 → 0.67로 오른다. 상위 1 %에서는 두 상관이 +0.12로 되돌아간다. 같은 파이프라인의 level-rate 상관은 ECEI +0.077, BES +0.063 으로 §8ar(+0.078 / +0.070)을 재현한다.

#### 3. 기록과의 연결
- **§8ab 헤드라인** $+0.174/+0.248/+0.257/+0.264$: 항등식 $f_Ir_I+(1-f_I)r_B$ 로 정확 재현, 천이 항 90–114 %. **§8ab 5절** peak/non-peak $+0.45\ldots+0.61$ / $+0.09\ldots+0.20$: 2.1에서 분할별로 재현. **§8ab 8절** `no_fast` 절제(컷 $T_i$ skill → −0.10…−0.18): $\kappa\to0$ 이면 (T5)의 $r_I\to1-2c^2<0$ 이 되어 천이 항이 음수로 뒤집힌다는 것과 정합.
- **§8ax** 꼬리 효과: 본류 행에서 모델 RMSE 53–59 eV ≈ 잡음 바닥, 중앙값 오차 26–31 eV 대 PCHIP 16–18 eV(모델이 앵커를 옮긴다, (T3)의 persistence 극한)이고 $J\ge q_{90}$ 부터 모델 중앙값이 PCHIP 아래로 내려간다. q75 교차의 자리다. foundations §10.3("상위 1 %가 skill의 62 %")은 $f_{\ge q99}=0.44$–0.51 에 대응한다.
- **§8an** "타깃이 움직이는 방전에서 이긴다"(조용 42 % → 변동 85 %): 2.3의 3분위(17–40 % → 65–81 %)와 $\rho$(skill, 천이 비중) 4/4 유의가 같은 현상이며, 이제 그 공변량이 **왜** 작동하는지((T6)의 $f_I=\nu\langle J^2\rangle$ 의 함수)가 있다.
- **§8ar** rate 상관 +0.078/+0.070 재현(+0.077/+0.063). 증분–증분 상관은 그 두 배(+0.14), 천이 조건부는 세 배(+0.24). **foundations §7.7** 인과 GP vs GP 전체 $-0.11\ldots-0.14$(유도 $s_{\rm hist}(10\,{\rm ms})=-0.21$, 부호 확인), 본류 $-0.28\ldots-0.34$. **§8al** 50 ms 포화는 (T3)의 $e^{-s/L}$, $L=12$–18 ms.

#### 4. 한계와 반증 조건
가정:
1. 충격이 격자 이하(0.1–1 ms)이며 $T_e$·$n_e$ 와 **같은 시각** $t_k$ 에 온다. ECEI에서는 확인(lag 0, 조건부 ×1.7), BES에서는 약하다(BES 채널 평균은 요동 위주, foundations §2.2).
2. 컷(3 keV) 아래 남은 1 % 극단 행에 비물리 적합 실패가 섞여 있을 수 있다. 거기서 증분 상관이 +0.12로 떨어지지만 모델 vs GP는 +0.08…+0.14 로 양수이므로 전부 인공물은 아니다. $D_\alpha$/ELM 지시자(plasma §12.4 항목 5)가 두 성분을 가른다.
3. (T5)의 계단 극한 $c_{\rm hist}^2/c_{\rm interp}^2=2$ 는 과대다. 측정된 천이 층의 인과 GP vs GP 는 −0.04…−0.06 뿐이다(10 ms 적분 CES가 계단을 뭉개고, 회복이 수 점에 걸치며, 단일점 스파이크에서는 모든 arm이 $J$ 를 틀림). $\tau_{\rm rec}$ 와 $\kappa$ 는 본 데이터에서 직접 적합하지 않았다.
4. "vs PCHIP" 헤드라인의 본류 양수는 PCHIP의 잡음 통과가 준 것이며 노이즈 인지 교량(GP) 대비 본류는 4/4 손실이다. 증명의 명제는 **최적 교량 대비**로 읽어야 한다.
5. (T5)의 지연 의존 $r_I(\lambda)$ 는 유도만 했고 측정하지 않았다.

반증 조건: (A) 어떤 분할에서 $f_Ir_I<\tfrac12\,$헤드라인; (B) 빠른 채널을 가진 인과 arm이 $J<q_{50}$ 층에서 비인과 GP를 이김((T3)·(T4) 교량 부등식과 §7.8 상한 위반); (C) $|\Delta T_i|\ge q_{95}$ 조건부 ECEI 증분 상관의 CI가 무조건부와 겹침, 또는 lag ±1 상관이 lag 0과 같음; (D) shot별 $\rho$(skill, $f_I$) $\le0$; (E) 평가 시 `no_fast` 절제 후에도 천이 층의 모델 vs GP 가 양수로 남음(그러면 이득은 동시 관측이 아니라 이력 처리 차이); (F) 지연 $\lambda$ 를 늘려도 천이 층의 $r_I$ 가 $\lambda/\Delta$ 에 비례해 줄지 않음.


### P02 — $V_\varphi$ 판정(양측 보간과 동률·persistence보다 우위·중앙값에서는 보간보다 열세·집계 우위는 꼬리에서만)은 "모든 토크 항이 비관측인 토로이달 운동량 균형"의 유일한 귀결이다

**스크립트** (읽기 전용, 학습·채점 없음):
`scratchpad/p02\p02_vrot_momentum.py` (Part A npz·Part B OU 닫힌 형식·Part C 원시 CSV 증분 교차상관) 및 같은 폴더의 `p02_part_d.py` (다중 척도 사건 검정·PCHIP 폴백 행), 출력 `p02_stdout.txt`, `p02_part_d_stdout.txt`, `p02_results.json`, `p02_part_d.json`, `p02_fallback.json`. 입력: `data/.b1_seqv2_s42_i42/comparison_errors_test.npz`(s42 컷), `data/.foundations.json`(D 강건 OU 적합, E 사다리), `data/*.csv` 641파일.

#### 0. 명제

입력 $z$(BES·ECEI·100 Hz Mirnov)가 운동량 균형의 어느 토크 항도 담지 않으면, $V_\varphi$의 Bayes 최적 인과 예측기는 **이력만 쓰는 잡음-OU 필터**와 같다. 그 필터의 오차는 간극 $g=t-t_{\rm last}$의 닫힌 함수이고, 거기서 네 가지가 따라 나온다. (a) 중앙값은 필터 하한 위, (b) 양측 보간에는 본류에서 진다(교량 분산), (c) 그럼에도 집계 MSE skill이 양수이려면 PCHIP의 꼬리 SSE 몫이 모델의 것을 "본류 결손"만큼 넘어야 한다는 부등식이 성립해야 하며, (d) 그 부등식은 s42 npz에서 수치로 성립한다. 원시 CSV에서 $V_\varphi$ 증분은 세 모달리티의 증분과 10–200 ms 어느 척도에서도 상관 0이고, 같은 통계가 $T_i$ 대조에서는 +0.15…+0.28을 낸다.

#### 1. 유도

##### 1.1 운동량 균형의 해 — ODE, 충격응답, 합성곱

0-D 토로이달 각운동량 $L=n m\langle R^2\rangle\omega_\varphi$, $V_\varphi=R\omega_\varphi$ (plasma_physics §7.1–7.2):

$$
\frac{dL}{dt}=T(t)-\frac{L}{\tau_\varphi},\qquad T(t)=T_{\rm NBI}(t)+T_{\rm NTV}(t)+T_{\rm int}(t).
$$

적분인자 $e^{t/\tau_\varphi}$를 곱하면

$$
L(t)=\int_{-\infty}^{t}e^{-(t-s)/\tau_\varphi}\,T(s)\,ds
\;\;\Longrightarrow\;\;
V_\varphi(t)=\int_{-\infty}^{t}h(t-s)\,T(s)\,ds,\qquad
h(u)=\frac{1}{n m R}\,e^{-u/\tau_\varphi}\,\mathbf 1[u\ge0].
$$

즉 $V_\varphi$는 토크의 **1차 저역통과**이고 $h$가 충격응답이다. 마지막 관측 시각 $t_{\rm last}$에서 적분을 자르면 (간극 $g=t-t_{\rm last}$)

$$
\boxed{V_\varphi(t)=e^{-g/\tau_\varphi}\,V_\varphi(t_{\rm last})+\underbrace{\int_{t_{\rm last}}^{t}h(t-s)\,T(s)\,ds}_{I(g):\ \text{간극 안의 토크만의 함수}}}
$$

첫 항은 (잡음까지) 관측된 양, 둘째 항 $I(g)$는 **간극 안의 토크**만 안다면 알 수 있는 양이다. 토크 요동이 백색이면 $V_\varphi$는 상관시간 $\tau_\varphi$, 정상 분산 $\sigma_s^2=\tau_\varphi S_T/(2n^2m^2R^2)$의 OU 과정이고, 관측 $\tilde V=V+n$, $n\sim\mathcal N(0,\sigma_n^2)$ (foundations §7.1).

##### 1.2 비관측 토크 ⇒ Bayes 최적 인과 예측기 = 이력 필터

MSE 최소자는 조건부 평균(foundations §6.2). 정보집합 $\mathcal F_t=\{z_{\le t},\ \tilde V_{\le t_{\rm last}}\}$에 대해

$$
\hat V(t)=E[V_\varphi(t)\mid\mathcal F_t]
=e^{-g/\tau_\varphi}\,E[V_\varphi(t_{\rm last})\mid\mathcal F_t]+E[I(g)\mid\mathcal F_t].
$$

§8ar의 항별 감사를 조건으로 쓰면: $T_{\rm NBI}$는 데이터에 채널이 없음, $T_{\rm NTV}\propto\delta B^2$은 100 Hz 데시메이션으로 파괴(§8b.2, 표본화 정리), $\nabla\!\cdot\!\Pi_\varphi$는 rate 상관 −0.006(null), $T_{\rm int}$는 $\nabla T_i$ 필요(스칼라만 보유). 따라서 간극 안의 $T(s)$는 $z$와 조건부 독립이고 ($z\perp T(s)\mid\tilde V$), 수준도 $z$에 담기지 않으므로(§8ar level 상관 +0.027/+0.005; 본 측정 +0.028/+0.005)

$$
E[I(g)\mid\mathcal F_t]=E[I(g)\mid\tilde V_{\le t_{\rm last}}],\qquad
E[V_\varphi(t_{\rm last})\mid\mathcal F_t]=E[V_\varphi(t_{\rm last})\mid\tilde V_{\le t_{\rm last}}].
$$

즉 $\hat V(t)$는 **이력만의 함수**이며 OU–Gauss에서는 정상상태 Kalman 필터(foundations §7.4)다. 10 ms 격자 $\rho=e^{-\Delta/\tau_\varphi}$로 사후분산 $P$를 Riccati로 풀고 ($P^-=\rho^2P+\sigma_s^2(1-\rho^2)$, $K=P^-/(P^-+\sigma_n^2)$, $P=(1-K)P^-$), 간극 $g$ 뒤의 오차분산은

$$
\boxed{\mathrm{Var}(g)=\rho(g)^2P+\sigma_s^2\bigl(1-\rho(g)^2\bigr)+\sigma_n^2,\qquad \rho(g)=e^{-g/\tau_\varphi}}
$$

이것이 $(z,\tilde V)$를 쓰는 **모든** 인과 예측기의 MSE 하한이다. persistence는 $\mathrm{Var}_{\rm pers}(g)=2\sigma_n^2+2\sigma_s^2(1-\rho(g))$.

**수치** (foundations D의 $V_\varphi$ 강건 적합 $\sigma_n=3.10$ km/s, $\sigma_s=8.43$ km/s, $\tau_\varphi=81.2$ ms, $\rho(10\,{\rm ms})=0.884$; 스크립트 Part B): $P=6.55$, $K=0.683$.

| $g$ [ms] | $\mathrm{Var}(g)$ 인과 필터 | $\mathrm{Var}_{\rm pers}(g)$ | $\mathrm{Var}_{\rm bridge}(g,10)$ | $\mathrm{Var}_{\rm bridge}(g,g)$ | skill 필터 vs pers | skill 필터 vs bridge$(g,10)$ |
|---:|---:|---:|---:|---:|---:|---:|
| 10 | 30.22 | 35.64 | 22.68 | 22.68 | +0.152 | −0.332 |
| 20 | 41.24 | 50.21 | 25.81 | 30.91 | +0.179 | −0.597 |
| 50 | 61.84 | 84.54 | 29.56 | 51.59 | +0.269 | −1.092 |
| 100 | 75.18 | 119.86 | 31.16 | 70.75 | +0.373 | −1.412 |
| 200 | 80.20 | 149.24 | 31.66 | 79.77 | +0.463 | −1.533 |

(단위 (km/s)². foundations D의 `by_gap` 값과 동일.)

**(a) 중앙값 하한.** OU–Gauss에서 최적 예측 오차는 Gauss이므로 $\mathrm{med}|e|=0.6745\sqrt{\mathrm{Var}(g)}$, 따라서 1-step에서 어떤 인과 예측기의 중앙값비(대 persistence)도 $\sqrt{\mathrm{Var}(10)/\mathrm{Var}_{\rm pers}(10)}$ 아래로 못 간다. $V_\varphi$ 적합은 **0.921**, 기록이 인용하는 $T_i$ 적합의 같은 식은 **0.875**(foundations §8 표, §8ba). 측정(1-step, Δt ≤ 15 ms, n = 10,372; `E.ladder_1step_CES_VT` 및 본 스크립트 재현): 모델 **1.140**, 인과 GP **1.009**, PCHIP **0.791**, linear 0.781, GP 0.815. 모델과 인과 GP는 하한 위(필연), PCHIP는 하한 아래 — PCHIP는 양측이므로 이 하한의 적용 대상이 아니고 (b)의 교량 하한 $\sqrt{22.68/35.64}=0.798$에 1 % 안으로 맞는다.

**(b) 교량 분산.** 과거 앵커가 $g_p$, 미래 앵커가 $g_f$ 떨어져 있을 때 두 잡음 앵커로부터의 최적 선형 예측 오차는 $c=\sigma_s^2(\rho_p,\rho_f)^\top$, $\Sigma=\sigma_s^2\begin{pmatrix}1&\rho_p\rho_f\\\rho_p\rho_f&1\end{pmatrix}$로

$$
\boxed{\mathrm{Var}_{\rm bridge}(g_p,g_f)=\sigma_s^2+\sigma_n^2-c^\top(\Sigma+\sigma_n^2I)^{-1}c,\qquad
\rho_{p,f}=e^{-g_{p,f}/\tau_\varphi}}
$$

잡음 없는 극한은 Brown 교량형 $\sigma_s^2\frac{(1-\rho_p^2)(1-\rho_f^2)}{1-\rho_p^2\rho_f^2}$ (예: $g_p=g_f=10$ ms에서 8.71), 잡음 극한($\rho\to1$)은 등가중 $\tfrac32\sigma_n^2$ 대 persistence $2\sigma_n^2$ (foundations §7.3). 표가 보이듯 이력만의 필터는 **모든 간극에서** 교량에 지고, 미래 앵커가 가까우면 간극이 길수록 더 진다(−0.33 → −1.53). 이것이 "본류에서 보간보다 열세"의 산수다.

**(c) 꼬리 부등식.** $\mathrm{MSE}=\int_0^1Q_{|e|}(p)^2dp$ (foundations §10.3). 행을 $se_m+se_p$ 순으로 상위 $q$ 꼬리 $\mathcal T$와 본류 $\mathcal B$로 나누고

$$
s_p=\frac{\mathrm{SSE}_p(\mathcal T)}{\mathrm{SSE}_p},\quad s_m=\frac{\mathrm{SSE}_m(\mathcal T)}{\mathrm{SSE}_m},\quad
r_T=\frac{\mathrm{SSE}_m(\mathcal T)}{\mathrm{SSE}_p(\mathcal T)},\quad r_B=\frac{\mathrm{SSE}_m(\mathcal B)}{\mathrm{SSE}_p(\mathcal B)}
$$

라 두면 항등식 $\mathrm{skill}=1-\bigl[s_p r_T+(1-s_p)r_B\bigr]$이 성립하고, 따라서

$$
\boxed{\mathrm{skill}>0\ \wedge\ r_B>1\ \Longleftrightarrow\ 
\underbrace{s_p-(1-\mathrm{skill})\,s_m}_{\text{꼬리 이득 }G_T=s_p(1-r_T)}
\;>\;\underbrace{(1-s_p)(r_B-1)}_{\text{본류 결손 }D_B}>0 }
$$

즉 **PCHIP의 상위 분위 SSE 몫이 (MSE비로 축척한) 모델의 몫을 본류 결손 $D_B$ 이상으로 넘어야** 중앙값비 > 1이면서 집계 skill > 0일 수 있다. 꼬리가 되는 행은 두 종류다: PCHIP가 비물리적 이웃(스파이크 앵커)을 **정확히 지나가는** 행 — 이력 필터는 $K<1$로 앵커를 수축하지만 보간은 앵커를 그대로 믿는다 — 과 간극 안의 큰 실제 점프 $A_k$ (1.1의 $I(g)$가 큰 행) 이다.

#### 2. 데이터 검정

**2.1 s42 컷 npz, `CES_VT` (n = 10,463)** — 스크립트 Part A. 기록과의 대조: `E.tail_CES_VT` skill 0.390·상위 1 % PCHIP 몫 0.828·q50비 1.442와 일치, §8ab s42 +0.390\*와 일치.

| 양 | 값 |
|---|---:|
| (i) 중앙값 $\lvert e\rvert$ 비 모델/PCHIP | **1.442** (4.505 / 3.124 km/s) |
| 중앙값비 모델/pers · 인과GP/pers · PCHIP/pers | 1.141 · 1.010 · 0.791 |
| (ii) MSE skill vs PCHIP · vs pers · vs 인과 GP | **+0.390 · +0.501 · +0.331** |
| 행 승률 모델<PCHIP · 모델<pers | 0.399 · 0.469 |
| 누적 skill 기여 q50 / q75 / q90 / q99 / q100 | −0.002 / −0.007 / −0.016 / **−0.024** / **+0.390** |
| 분위비 모델/PCHIP q50 / q75 / q90 / q95 / q99 | 1.442 / 1.214 / 1.070 / 1.011 / 0.878 |

(iii)·(iv) 꼬리 몫과 부등식:

| 꼬리 $q$ | 행 | $s_p$ | $s_m$ | $r_T$ | $r_B$ | $G_T$ | $D_B$ | $G_T>D_B$ | 본류 중앙값비 |
|---:|---:|---:|---:|---:|---:|---:|---:|:---:|---:|
| 1 % | 105 | **0.829** | 0.681 | 0.501 | **1.142** | +0.414 | +0.024 | 성립 | 1.448 |
| 5 % | 524 | 0.921 | 0.837 | 0.554 | 1.256 | +0.410 | +0.020 | 성립 | 1.474 |
| 10 % | 1,047 | 0.953 | 0.896 | 0.574 | 1.337 | +0.406 | +0.016 | 성립 | 1.472 |

$G_T-D_B=0.414-0.024=0.390=$ skill (항등식 재현). 본류는 PCHIP SSE의 17 %뿐이라 중앙값 44 % 열세가 SSE로는 0.024에 불과하고, 상위 1 %의 105행이 skill 전부를 준다. 그 105행의 해부: persistence 오차 중앙값 172.8 km/s(전체 3.95) — 즉 $\lvert A_k\rvert$가 거대한 점프/스파이크 행이고, 그중 38 %는 $se_{\rm PCHIP}>se_{\rm pers}$(미래 앵커가 해친 행, 꼬리 PCHIP SSE의 27.5 %), 나머지는 실제 점프. 꼬리 안에서 모델 skill vs pers +0.577, vs PCHIP +0.499. peak 층 skill +0.787(중앙값비 1.33) 대 non-peak +0.057(1.50).

$T_i$ 대조(같은 npz): 중앙값비 1.222, skill +0.174, 상위 1 % $s_p=0.529$, $r_B=0.859$(**본류에서도 우위**), 누적 기여가 q90에서 이미 양수 — $V_\varphi$와 부호 구조가 다르다.

**2.2 원시 CSV 641파일, held-free 증분 교차상관** — Part C·D. 블록은 `time` 간격 ≥ 0.5 s에서 분할, held는 블록 내 직전 관측과 같은 값(`CES_VT` 188,100 관측 중 101,590 = 54 % 제거, `CES_TI` 1행), $T_i>3$ keV 컷, 모달리티 평균은 shot별 채널 z-score 평균, 블록당 ≥ 40 유효쌍, 블록 중심화 후 풀링 Pearson. $se$는 블록 군집 표준오차.

lag $\ell$: $\mathrm{corr}(\Delta V_t,\ \Delta X_{t+\ell})$, 1-step 증분:

| 구동 → 타깃 | ℓ=−2 | ℓ=−1 | **ℓ=0** | ℓ=+1 | ℓ=+2 | 블록 수 |
|---|---:|---:|---:|---:|---:|---:|
| BES → $T_i$ | −0.004 | −0.029 | **+0.153** (se 0.009) | −0.068 | +0.015 | 608 |
| ECEI → $T_i$ | −0.007 | −0.031 | **+0.146** (se 0.008) | −0.099 | +0.027 | 608 |
| MC → $T_i$ | +0.000 | −0.000 | +0.001 | −0.001 | +0.002 | 608 |
| BES → $V_\varphi$ | −0.001 | +0.020 | **−0.004** (se 0.005) | +0.003 | −0.003 | 347 |
| ECEI → $V_\varphi$ | −0.003 | +0.020 | **−0.002** (se 0.005) | +0.002 | −0.001 | 347 |
| MC → $V_\varphi$ | −0.000 | −0.001 | +0.004 | −0.008 | +0.005 | 347 |

다중 척도(8az의 100 ms 질문): $k$-스텝 변화 $V_{t+k}-V_t$(양끝 관측) 대 창 안 구동 활동 $\sum_{(t,t+k]}\lvert\Delta X\rvert$ (사건 **시각** 정보) 및 순변화 $X_{t+k}-X_t$ (부호 정보), BES 기준:

| $k$ (ms) | $T_i$ 부호 | $T_i$ $\lvert\Delta\rvert$ vs 활동 | $V_\varphi$ 부호 | $V_\varphi$ $\lvert\Delta\rvert$ vs 활동 | $V_\varphi$ $\lvert\Delta\rvert$ vs $\lvert$순변화$\rvert$ |
|---:|---:|---:|---:|---:|---:|
| 1 (10) | +0.153 | +0.147 | −0.004 | +0.004 | +0.004 |
| 5 (50) | +0.245 | +0.128 | +0.008 | +0.006 | +0.003 |
| 10 (100) | **+0.281** | +0.114 | +0.012 (se 0.008) | **+0.003** (se 0.010) | +0.004 |
| 20 (200) | +0.277 | +0.097 | +0.014 (se 0.009) | +0.001 (se 0.011) | +0.005 |

(ECEI·MC도 $V_\varphi$에서 전부 $\lvert r\rvert\le0.014$; $T_i$ ECEI 부호 +0.146 → +0.248.) 방법은 $T_i$에서 모든 척도에 신호를 잡고, $V_\varphi$에서는 **부호도, 크기도, 사건 활동도** 어느 척도에서도 0이다. "구동이 비관측"은 lag 0의 한 수치가 아니라 ±20 ms·10–200 ms 창 전부에서 측정된 null이다.

#### 3. 기록과의 연결

- **§8ab, vs PCHIP 1/4.** $\mathrm{skill}=G_T-D_B$이고 $G_T$는 split당 ~100행(1 %)이 운반한다. shot 군집 bootstrap의 분산은 그 꼬리 행 수의 Poisson 요동으로 정해지므로 CI 폭 $\propto1/\sqrt{n_{\mathcal T}}$ — 꼬리가 큰 s42(+0.390)만 유의하고 +0.135…+0.305는 CI가 0을 품는다. 같은 식이 §8az의 "`V_rot` CI [+0.08, +0.53]"의 폭도 설명한다.
- **§8ab, vs persistence 3/4.** 이력 필터의 이론 skill은 $1-\mathrm{Var}(10)/\mathrm{Var}_{\rm pers}(10)=+0.152$(그중 동역학만의 몫 $(1-\rho)/2=0.058$, 나머지는 잡음 평균화). 측정 +0.501은 그보다 훨씬 크지만 본류 중앙값비는 1.141 > 1 — 즉 persistence 대비 우위도 필터링이 아니라 꼬리다: 점프 행에서 persistence 오차는 $\lvert A_k\rvert$ 전부, 모델은 헤지로 그 일부(꼬리 skill vs pers +0.577).
- **§8ax, 중앙값 −49 %.** 하한 0.921(V 적합)/0.875(T_i 적합) 대 측정 1.140: 모델은 도달 가능한 인과 중앙값보다 24–30 % 위에 있다. 입력에 정보가 없는 타깃에 MSE 손실을 걸면 손실의 83 %가 1 %의 행에서 오므로(상위 1 % $s_p=0.829$) 머리가 그 행들에 맞춰 헤지하고, 그 비용이 본류의 분산으로 나타난다(foundations §15.4의 기전). PCHIP 대비 −49 %($1/1.442$)는 여기에 교량 이득 0.798이 곱해진 값: $1.140/0.798=1.43\approx1.442$.
- **§8az, 오프라인 100 ms 척도 회복.** 1.1의 식을 사건 형태로 쓰면
  $$V(t)=V(t_{\rm last})+\sum_{t_{\rm last}<t_k\le t}A_k,$$
  오프라인에서는 두 앵커가 $\sum_k A_k$(총 진폭)를 잡음 한도 안에서 **준다**. 긴 간극 안에서 남는 미지수는 진폭의 시간 배치 $\{t_k\}$뿐이다. 힌트의 가설("빠른 채널이 $t_k$를 본다")은 본 측정에서 지지되지 않는다: 100–200 ms 창 안의 BES/ECEI 활동과 $\lvert\Delta V\rvert$의 상관은 +0.003/+0.001 (se 0.01). 게다가 §8az의 대조군 `bilstm`은 seq_v2의 라우팅을 그대로 가져 **$V_\varphi$ 분기가 빠른 채널을 아예 보지 않는다**(`experiments/b12/b12_models.py` 서두, `model_seq_v2.py` 라우팅). 따라서 >10스텝 간극의 +0.386은 빠른 채널이 아니라 **양방향 라벨 문맥**(진폭 정보, $\mathrm{Var}_{\rm bridge}$를 학습된 비선형 교량으로 구현)에서 온 것이고, 빠른 채널까지 보는 `tm_scratch`가 더한 것은 +0.049 [+0.001, +0.079]가 전부다. §8az의 문장 "diagnostics inside the gap … recoverable from the fast channels"는 "two-sided label context at the 100 ms scale"로 고쳐 읽어야 한다.
- **§8ab 블록 2의 PR2 폴백 43 %**는 npz와 맞지 않는다. s42 컷 npz에서 세 양측 팔이 모두 persistence와 일치하는 행(미래 이웃 없음)은 **58행 = 0.55 %**이고 그 행의 skill은 +0.174; `comparison_metrics.json`의 `future_neighbor_fraction` 0.566은 `eval_seq.py`가 `n_total`을 채점 여부와 무관하게 모든 격자 행에 대해 세기 때문으로 보인다(라벨 없는 행 포함). "vs PCHIP의 40 %는 vs persistence"라는 §8ab의 문장은 채점 행에 대해서는 성립하지 않는다 — 팀 리드 확인 요망.

#### 시간 $t$의 식

$$
\begin{aligned}
&\text{ODE:}\quad \frac{dL}{dt}=T(t)-\frac{L}{\tau_\varphi},\qquad L=nm\langle R^2\rangle\omega_\varphi,\quad V_\varphi=R\omega_\varphi\\[2pt]
&\text{충격응답:}\quad h(u)=\frac{1}{nmR}e^{-u/\tau_\varphi}\mathbf 1[u\ge0],\qquad
V_\varphi(t)=\int_{-\infty}^{t}h(t-s)\,T(s)\,ds\\[2pt]
&\text{감쇠·갱신 분해:}\quad V_\varphi(t)=e^{-(t-t_{\rm last})/\tau_\varphi}V_\varphi(t_{\rm last})+\int_{t_{\rm last}}^{t}h(t-s)T(s)\,ds
= V_\varphi(t_{\rm last})+\sum_{t_{\rm last}<t_k\le t}A_k\\[2pt]
&\text{OU 이산화:}\quad V_{t+\Delta}=\bar V+\rho\,(V_t-\bar V)+\varepsilon_t,\quad \rho=e^{-\Delta/\tau_\varphi},\ \mathrm{Var}\,\varepsilon=\sigma_s^2(1-\rho^2),\quad \tilde V=V+n\\[2pt]
&\text{추정기 오차(간극 }g\text{):}\quad \mathrm{Var}_{\rm causal}(g)=e^{-2g/\tau_\varphi}P+\sigma_s^2(1-e^{-2g/\tau_\varphi})+\sigma_n^2\\
&\qquad\qquad \mathrm{Var}_{\rm pers}(g)=2\sigma_n^2+2\sigma_s^2(1-e^{-g/\tau_\varphi}),\qquad
\mathrm{Var}_{\rm bridge}(g_p,g_f)=\sigma_s^2+\sigma_n^2-c^\top(\Sigma+\sigma_n^2I)^{-1}c\\[2pt]
&\text{정상 필터:}\quad P^-=\rho^2P+\sigma_s^2(1-\rho^2),\ K=\frac{P^-}{P^-+\sigma_n^2},\ P=(1-K)P^-;\quad \text{기억 }L=\frac{1}{-\ln(\rho(1-K))}=0.79\text{ 스텝}\\[2pt]
&\text{중앙값 하한:}\quad \frac{\mathrm{med}|e|_{\rm causal}(g)}{\mathrm{med}|e|_{\rm pers}(g)}\ \ge\ \sqrt{\frac{\mathrm{Var}_{\rm causal}(g)}{\mathrm{Var}_{\rm pers}(g)}}\quad(=0.921\ \text{at }g=10\text{ ms})\\[2pt]
&\text{skill 항등식:}\quad \mathrm{skill}=1-\bigl[s_pr_T+(1-s_p)r_B\bigr]=G_T-D_B,\qquad G_T=s_p-(1-\mathrm{skill})s_m
\end{aligned}
$$

#### 5. 한계

1. OU 모수는 held-free 쌍에서 적합되어 $\sigma_n$ 과대·$\tau_\varphi$ 하한 쪽 편향 가능(§8ar 2절). 하한 0.921은 그 적합에 조건부이며 고전 적합은 0.850을 준다; 어느 쪽이든 측정 1.140은 위다.
2. 중앙값 하한은 Gauss 오차(선형–Gauss OU의 Bayes 오차)에서만 등호형이다. MSE 학습 머리가 꼬리에 맞춰 분포를 비틀 수 있다는 것이 바로 §8ax의 기전이므로, 하한은 "최적 인과 예측기의 중앙값"에 대한 것이고 임의 추정기는 MSE를 희생해 중앙값을 그 아래로 내릴 수 있다.
3. 증분 상관은 풀링 선형 통계다. 비선형·shot 조건부 사건 정보(예: L–H 천이 직후 체제)는 이 검정이 못 볼 수 있다. 뒤집을 측정: $V_\varphi$ 분기에 빠른 채널을 라우팅한 bilstm 대 라우팅 대조군을 >10스텝 간극에서 짝지어 비교(`tm_scratch` 셀 +0.049가 그 상한의 첫 추정).
4. 꼬리 105행의 "스파이크 앵커 대 실제 점프" 분류는 $se_{\rm PCHIP}>se_{\rm pers}$ 대리지표이며 $V_\varphi$에는 스파이크 컷이 없어 $y_{\rm true}$ 자체가 비물리적인 행이 섞여 있다.
5. 간극 ≥ 25 ms 층은 $V_\varphi$에서 10행뿐이라 $\mathrm{Var}(g)$의 $g$-의존 곡선은 데이터로 검증되지 않았다(이론값만 표에 있음).
6. 43 % 폴백 불일치의 원인은 코드 읽기로 추정한 것이며, 해당 run의 `n_total` 집계를 직접 재실행해 확인하지는 않았다.


### P03. 50 ms 문맥 포화는 세 척도(필터 수렴·전자–이온 평형화·ELM 회복)의 일치인가 — 유도·물리·데이터 검산

**명제(검증 대상).** §8al이 잰 인과 문맥 포화 50 ms는 (A) 잡음 있는 Markov 타깃에 대한 최적 필터의 수렴 시간이고, (B) 전자–이온 평형화 시간 $\tau_{\rm eq}$·ELM 후 pedestal 재건 시간 $\tau_{\rm build}$와 일치하며, (C) 셋이 같은 pedestal 물리이기 때문이다.

**판정 요약.** (A)는 **수치로는 기각, 형태로는 지지**: 적합된 OU 모수로 cold-start Kalman 재귀를 풀면 2 % 문턱 도달은 **2 스텝(강건)·3 스텝(고전)**이지 5가 아니며, §8al 곡선과의 RMS 차는 0.021–0.026(측정 곡선 자체의 RMS 0.032)이다. 측정 곡선의 감쇠비 0.56/스텝(유효 기억 3.4 스텝)은 OU가 주는 0.18–0.33보다 느리고, 이를 OU 안에서 재현하려면 Kalman 이득 0.155($\sigma_n/\sigma_s=1.88$)가 필요한데 어느 적합도 그런 잡음비를 주지 않는다. 따라서 **CES 이력 채널의 잡음 평균화만으로는 50 ms의 절반도 설명되지 않고**, 나머지는 OU가 담지 않는 두 번째 척도 — 데이터에서 직접 잰 ELM형 회복 $\tau_{\rm rec}\approx55$–64 ms — 와 빠른 진단의 블록별 수준 보정이 운반한다. (B)는 자릿수로 성립: $\tau_{\rm eq}$ 18–39 ms(CES 구석), $\tau_{\rm build}$ 4–17 ms(껍질 체적) / 42–192 ms(플라즈마 체적), $\tau_{\rm rec}$ 55–64 ms가 모두 50 ms의 한 자릿수 안에 있다. (C)는 "같은 자릿수의 pedestal 척도"로만 주장하고, "동일한 물리"로는 주장하지 않는다.

스크립트: `scratchpad/p03\p03_proof.py` → 같은 폴더의 `p03_results.json`, `p03_stdout.txt`. 입력: `data/.foundations.json` 키 D(OU 적합)·641개 shot CSV(held-free, 3 keV 컷, ≥ 0.5 s 간극에서 블록 분할 — `analyze_foundations.py`와 동일 처리). 모델 학습·채점 없음.

---

#### 1. 유도 A — cold-start Kalman 재귀와 "문맥 k 스텝"의 MSE

**모형.** 잡음 있는 OU(foundations §7.1): $y_{t+1}=\bar m+\rho(y_t-\bar m)+\varepsilon_t$, $\rho=e^{-\Delta/\tau}$, $\mathrm{Var}\,\varepsilon=q\equiv\sigma_s^2(1-\rho^2)$, 관측 $\tilde y=y+n$, $n\sim\mathcal N(0,\sigma_n^2)$.

**재귀.** 관측이 하나도 없을 때의 사전은 정상분포이므로 $P_1^-=\sigma_s^2$("정보 없음"). $k$번째 관측 뒤
$$
K_k=\frac{P_k^-}{P_k^-+\sigma_n^2},\qquad P_k=(1-K_k)P_k^-,\qquad P_{k+1}^-=\rho^2P_k+q,
$$
그리고 $k$개 관측 뒤 다음 관측값의 1-step 예측 MSE는
$$
\mathrm{MSE}_k=P_{k+1}^-+\sigma_n^2 .
$$
Gauss 가정 아래 이 재귀는 과거 $k$관측을 쓰는 최적 선형 예측(foundations §7.4의 $\sigma_s^2+\sigma_n^2-c^\top(\Sigma_k+\sigma_n^2I)^{-1}c$)과 **정확히 같다** — 스크립트가 $k=1,2,3,5,10$에서 `.foundations.json`의 `mse_filter_K`와 소수 둘째 자리까지 일치함을 확인했다(강건 3450.66/3216.28/3178.05/3170.06/3169.80; 고전 32536.55/30638.37/30080.41/29845.21/29817.03).

**기하 수렴.** $f(x)=x\sigma_n^2/(x+\sigma_n^2)$라 두면 $P_{k+1}^-=\rho^2f(P_k^-)+q$이고 $f(x)-f(y)=\dfrac{\sigma_n^4(x-y)}{(x+\sigma_n^2)(y+\sigma_n^2)}=(1-K_x)(1-K_y)(x-y)$이므로, 정상상태 $P_\infty^-$에서의 편차 $\delta_k\equiv\mathrm{MSE}_k-\mathrm{MSE}_\infty$는 **정확히**
$$
\delta_{k+1}=\rho^2(1-K_k)(1-K_\infty)\,\delta_k\;\xrightarrow{k\to\infty}\;\bigl(\rho(1-K_\infty)\bigr)^2\delta_k\equiv r\,\delta_k .
$$
cold start에서는 $P_k^-\ge P_\infty^-$이므로 $K_k\ge K_\infty$, 즉 과도기의 수축은 점근비 $r$보다 **더 빠르다**(강건: 0.092, 0.166, 0.178, 0.180, …; 고전: 0.238, 0.302, 0.321, 0.327, …). 기억 길이 $L=1/[-\ln(\rho(1-K_\infty))]$와는 $r=e^{-2/L}$.

**닫힌 형식(점근비 상한).** 결손을 $\varepsilon$ 이내로 넣는 스텝 수는
$$
k_\varepsilon=\frac{\ln\bigl(\delta_0/(\varepsilon\,\mathrm{MSE}_{\rm ref})\bigr)}{-\ln r}
=\frac{L}{2}\ln\frac{\delta_0}{\varepsilon\,\mathrm{MSE}_{\rm ref}},\qquad
\delta_0=\sigma_s^2+\sigma_n^2-\mathrm{MSE}_\infty ,
$$
(관측 0개 = 평균 예측, $\mathrm{MSE}_0=\sigma_s^2+\sigma_n^2$). skill 단위 변환: 결손 $=(\mathrm{MSE}_k-\mathrm{MSE}_\infty)/\mathrm{MSE}_{\rm ref}$, 주 기준 $\mathrm{MSE}_{\rm ref}=\mathrm{MSE}_{\rm PCHIP}$ = 같은 OU의 양측 보간 최적값(키 `pred.mse_interp_opt`), 민감도로 $\mathrm{MSE}_\infty$(필터 자신의 정상상태 = "full-block 대비"의 유사물)와 persistence. $\varepsilon=0.02$ = `PRACTICAL_EPS`.

**평가(두 적합 모두).**

| 적합 | $\sigma_n$ | $\sigma_s$ | $\tau$ | $\rho$ | $K_\infty$ | $L$ | $r$ | $\mathrm{MSE}_0$ | $\mathrm{MSE}_\infty$ | $\delta_0$ | $\mathrm{MSE}_{\rm PCHIP}$ | $k_\varepsilon$ 닫힌형 | **포화 스텝(정확 재귀)** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 강건 | 39.0 eV | 68.7 | 81 ms | 0.884 | 0.520 | 1.17 | 0.180 | 6237 | 3170 | 3067 | 2734 | 2.35 | **2** |
| 고전 | 139.9 | 147.1 | 74 ms | 0.874 | 0.343 | 1.80 | 0.330 | 41226 | 29817 | 11409 | 28825 | 2.69 | **3** |

기준을 $\mathrm{MSE}_\infty$로 바꿔도 2 / 3, persistence로 바꾸면 2 / 2 — **어느 정규화에서도 5가 나오지 않는다.**

**예측 결손 곡선 대 §8al(모두 skill 단위, 음수 = 결손).**

| 문맥 | 20 ms | 30 | 40 | 50 | 60 | 70 | 100 |
|---|---:|---:|---:|---:|---:|---:|---:|
| §8al 측정(full-block 대비) | −0.066 | −0.033 | −0.021 | −0.017 | −0.005 | +0.002 | −0.009 |
| OU 강건 예측 ($/\mathrm{MSE}_{\rm PCHIP}$) | −0.017 | −0.003 | −0.0005 | −0.0001 | 0 | 0 | 0 |
| OU 고전 예측 | −0.029 | −0.009 | −0.003 | −0.001 | −0.0003 | −0.0001 | 0 |
| 측정 곡선의 기하 적합($0.214\cdot0.559^k$) | −0.067 | −0.037 | −0.021 | −0.012 | −0.007 | −0.004 | −0.0006 |

| 비교(문맥 2–7 스텝) | 강건 | 고전 |
|---|---:|---:|
| RMS(예측 − 측정) | **0.026** | **0.021** |
| RMS(측정 자체) | 0.032 | 0.032 |
| RMS 3–7 스텝만 | 0.018 | 0.015 |
| 예측 포화 스텝 / 측정 | 2 / 5 | 3 / 5 |

**읽기.** (i) 형태는 같다 — 둘 다 단조 기하 감쇠이고, 측정 곡선은 $k=2$–6에서 로그-선형($r_{\rm meas}=0.559$, $L_{\rm eff}=3.43$ 스텝)에 잘 맞는다. (ii) 그러나 **속도가 2–3배 다르다**: OU의 기억 1.2–1.8 스텝(12–18 ms)에 대해 측정은 3.4 스텝(34 ms). RMS 차 0.021–0.026은 측정 곡선의 RMS 0.032와 같은 자릿수 — 즉 OU는 20 ms 결손의 26 %(강건)–43 %(고전)만 설명하고 30 ms 이상의 결손은 거의 전부 설명하지 못한다. (iii) $\rho=0.884$를 고정하고 $r_{\rm meas}$를 만들려면 $K_\infty=0.155$, 즉 $\sigma_n/\sigma_s=1.88$이 필요한데 적합값은 0.57(강건)·0.95(고전)이다. **단일 척도 OU의 잡음 평균화 cold start는 50 ms 포화의 원인이 아니다.** foundations §7.5·§7.6이 "K≥5에서 8.1 % 포화 → 50 ms"라 쓴 것은 0.1 % 수준의 꼬리를 본 것이고, 사전등록 문턱 2 %로 재면 포화는 20–30 ms다 — 이 문장은 그렇게 정정되어야 한다.

**나머지를 운반하는 것(유도가 지목).** 측정 결손은 full-block `seq_v2`(빠른 진단 + 이력)와의 짝비교이므로, OU가 빠뜨리는 것이 그대로 결손에 들어 있다. 후보는 셋이고 각각 cold start의 **수렴 속도가 다르다**: (a) ELM형 급락 뒤의 결정론적 회복(§3에서 $\tau_{\rm rec}\approx60$ ms로 측정) — 백색 구동 OU에는 없는 두 번째 척도이고, 그 형태 식별은 문맥이 $\tau_{\rm rec}$의 상당 부분을 덮어야 가능하다(§2의 CRLB: 예측 분산이 $\sigma_n^2$ 아래로 내려오는 문맥이 7–10 스텝); (b) 빠른 진단의 블록별 수준 보정(ECEI·BES → $T_i$ level 상관 0.31·0.34, §8ar) — 오프셋 추정은 기하가 아니라 $1/k$로 수렴; (c) 학습된 LSTM이 최적 필터가 아니라는 근사 손실. (a)·(b)가 있는 한 포화는 OU의 2–3 스텝이 아니라 더 느린 척도가 정하고, 그 척도가 데이터에서 50–60 ms로 잰 $\tau_{\rm rec}$이다.

---

#### 2. 유도 B — pedestal 물리: $\tau_{\rm eq}$와 $\tau_{\rm build}$, 그리고 정보 논증

**평형화.** $\tau_{\rm eq}=(m_i/2m_e)\,\tau_e$, $\tau_e=3.44\times10^5\,T_e^{3/2}/(n_e[\mathrm{cm^{-3}}]\ln\Lambda)$ s(foundations §3.1, NRL 정확식과 3 % 이내). **재건.** pedestal 에너지 $W_{\rm ped}=\tfrac32 n(T_e+T_i)V_{\rm ped}$, $V_{\rm ped}=2\pi R\cdot2\pi a\sqrt\kappa\cdot\Delta_{\rm ped}$, $\tau_{\rm build}=W_{\rm ped}/P_{\rm loss}$. KSTAR형 $R=1.8$ m, $a=0.5$ m, $\kappa=1.8$, $\Delta_{\rm ped}=3$ cm → $V_{\rm ped}=1.43$ m³; 비교용으로 EPED식 정의(pedestal 압력 × 플라즈마 체적 $2\pi^2Ra^2\kappa=16.0$ m³)도 함께. $P_{\rm loss}=2$ MW.

| $n_e$ [m⁻³] | $T_e$ [eV] | $T_i$ [eV] | $\tau_{\rm eq}$ | $W_{\rm ped}$(껍질) | $\tau_{\rm build}$(껍질) | $W_{\rm ped}$(체적) | $\tau_{\rm build}$(체적) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| $2\times10^{19}$ | 500 | 593 | 20.8 ms | 7.5 kJ | 3.8 ms | 84 kJ | 42 ms |
| $3\times10^{19}$ | 600 | 593 | 18.2 | 12.3 | 6.2 | 138 | 69 |
| $3\times10^{19}$ | 1000 | 1000 | 39.2 | 20.6 | 10.3 | 231 | 115 |
| $5\times10^{19}$ | 1000 | 1000 | 23.5 | 34.4 | 17.2 | 384 | 192 |

($T_i$ 593 eV = 관측 중앙값, foundations §3.2.) 두 정의는 체적비 11.2배만큼 갈리지만 **둘 다 수 ms–백수십 ms, 즉 50 ms를 사이에 둔 한 자릿수**다. 껍질 정의는 ELM이 실제로 비우는 에너지($\Delta W_{\rm ELM}/W_{\rm ped}\sim5$–20 %, plasma_physics §6.3)의 재건에 가깝고, 체적 정의는 pedestal 구속 시간 $\tau_{E,\rm ped}$에 가깝다. $\tau_{\rm eq}$는 CES 구석($T_e\approx0.5$–1 keV, $n_e\approx2$–5$\times10^{19}$)에서 18–39 ms(§3.2 표 전체로는 8–59 ms).

**정보 논증(왜 필터가 $\tau_{\rm build}$급 문맥을 필요로 하나).** 급락 뒤 회복은 결정론적 완화 $y(t)=a-b\,e^{-(t-t_0)/\tau_{\rm rec}}$다. 문맥 $r$ 스텝($t_{\rm ctx}=r\Delta$)의 잡음 관측에서 $(a,b,\tau_{\rm rec})$를 추정해 다음 점을 외삽할 때의 Cramér–Rao 하한(delta 법, $b=180$ eV = 600 eV의 30 % 급락, $\sigma_n=39$ eV):

| $t_{\rm ctx}$ | 30 ms | 40 | 50 | 60 | 70 | 100 | 150 | 300 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| $\mathrm{Var}[\hat y]/\sigma_n^2$, $\tau_{\rm rec}=30$ ms | 9.97 | 3.89 | 2.19 | 1.44 | 1.03 | 0.48 | 0.21 | 0.06 |
| $\tau_{\rm rec}=50$ ms | 12.7 | 5.05 | 2.90 | 1.94 | 1.42 | 0.72 | 0.33 | 0.08 |
| $\tau_{\rm rec}=80$ ms | 14.7 | 5.90 | 3.43 | 2.33 | 1.72 | 0.91 | 0.46 | 0.13 |

문맥이 $\tau_{\rm rec}$의 절반 아래면 곡선이 직선과 구별되지 않아 $b$와 $\tau_{\rm rec}$가 퇴화하고(외삽 분산이 잡음의 10배 이상), 예측 분산이 잡음 바닥 아래로 내려오는 것은 $t_{\rm ctx}\approx\tau_{\rm rec}$–$1.5\tau_{\rm rec}$(70–100 ms)에서다. **따라서 회복 형태를 읽어 다음 점을 맞히려면 문맥은 $\tau_{\rm rec}$급이어야 하고**, 30–50 ms 구간에서 분산이 스텝당 1.6–2.5배씩 떨어지는 가파른 구간이 §8al의 40→60 ms 결손 감소(−0.021→−0.005)와 같은 자리에 있다. 이는 단일 척도 OU의 2–3 스텝 cold start와 달리 50 ms급 포화를 **내놓는** 메커니즘이다(학습된 모델은 매 블록 모수를 적합하지 않고 형태를 가중치에 담지만, 문맥 안에서 "어느 위상에 있는가"를 읽어야 하는 점은 같다).

---

#### 3. 데이터 검산 — ELM형 급락과 회복 시간 $\tau_{\rm rec}$

**검출.** 641파일, held-free(블록 내 같은 값 반복 제거), $T_i>3$ keV 컷, ≥ 0.5 s 간극 블록 분할: 관측행 225,793, 같은 블록의 연속 10 ms 쌍 216,895(격자 2,169 s). 이벤트 = 직전 관측값 대비 한 스텝에 25 % 넘게 하락한 행. 이벤트 뒤 30 스텝의 $T_i/T_{i,\rm pre}$ 곡선에 $y(k)=y_\infty-A\,e^{-k\Delta/\tau_{\rm rec}}$를 적합(τ 격자 3–2000 ms, 선형 모수는 최소제곱).

| 집단 | 정의 | 사건 수 | 격자상 빈도 | $\tau_{\rm rec}$(중앙값 곡선) | $y_\infty$ | 사건별 $\tau_{\rm rec}$ q25/50/75 | 사건별 < 50 ms / < 100 ms |
|---|---|---:|---:|---:|---:|---|---|
| 전체 | 25 % 급락 | **16,703** | 7.70 /s | 15.7 ms | 0.79 | 13.7 / **55.3** / 2000 ms (n 13,699) | 0.48 / 0.59 |
| 지속 | 다음 점도 < 0.9 | 10,843 | 5.00 /s | 28.8 ms | 0.75 | 16.7 / **59.0** / 2000 (n 9,117) | 0.47 / 0.58 |
| 단일점 | 다음 점 ≥ 0.9 | 4,303 | 1.98 /s | 3.0 ms | 0.96 | 3.0 / 39.0 / 2000 | 0.53 / 0.63 |
| 지속(전 3점 중앙값 기준) | 사전 수준 = 직전 3관측 중앙값 | 7,691 | 3.55 /s | 43.5 ms | 0.84 | 20.3 / **64.4** / 2000 (n 6,387) | 0.45 / 0.57 |
| **회복형**(위 중 10–30 스텝 중앙값 ≥ 0.9) | 실제로 되돌아온 사건 | **2,783** | 1.28 /s | **60.3 ms** | 1.16 | 23.2 / **59.0** / 802 (n 2,545) | 0.46 / 0.61 |

(사건이 있는 파일 594/641; 사전 수준 중앙값 554 eV, 급락 폭 중앙값 42 %; 다음 점 결측 1,557.)

회복형 집단의 중앙값 곡선($k=0,\dots,10$): 0.60, 0.67, 0.79, 0.82, 0.88, 0.92, 0.95, 0.98, 1.02, 1.03, 1.06; $k=15/20/30$: 1.11 / 1.12 / 1.18 — 급락 60 %에서 출발해 **6 스텝(60 ms)에 95 %, 8 스텝(80 ms)에 사전 수준**으로 돌아오고 그 뒤 계속 올라간다(ELM 뒤 pedestal이 다음 한계까지 서는 상승, §6.3의 "톱니"; $y_\infty=1.16$은 그 상승을 지수 하나로 적합한 결과이고 선택 편향도 섞여 있다). 지속 집단의 $y_\infty=0.75$–0.84는 사전 수준이 양의 스파이크였던 사건과 영구 천이(L-back, 가열 변화)가 섞인 탓이다.

**비교.** 어느 정의로 재도 사건별 $\tau_{\rm rec}$ 중앙값은 **55–64 ms**, 중앙값 곡선의 $\tau_{\rm rec}$는 가장 깨끗한 회복형에서 **60 ms**: §8al의 50 ms와 같은 자릿수이고, $\tau_{\rm eq}$ 18–39 ms보다 1.5–3배 길며, 껍질 $\tau_{\rm build}$ 4–17 ms보다 길고 체적 $\tau_{\rm build}$ 42–192 ms 안에 든다. 분포는 넓다(q25 14–23 ms, q75는 격자 상한 = 30 스텝 안에 회복이 없는 사건 25 % 이상).

**빈도의 뜻.** 25 % 급락은 격자 쌍의 7.7 %(13 스텝에 한 번)에서 일어나는데, 554 eV의 25 %는 139 eV = 고전 $\sigma_n$ 139.9 eV의 **1배**이므로 이 집단의 큰 몫은 ELM이 아니라 fit-failure 꼬리다 — 단일점 급락 26 %가 그 직접 증거(다음 점에서 1.08로 되튐). 실제 ELM/천이율의 추정으로 쓸 수 있는 것은 회복형 **1.3 /s**(또는 3점 기준 지속 3.6 /s)이고, 이는 RMP 억제 체제(plasma_physics §6.4: ELM 톱니 대신 간헐적 작은 이벤트)와 $f_{\rm ELM}\sim10$–100 Hz 자연 ELM의 하단 사이에 있다.

---

#### 4. 기록과의 연결, 한계

1. **§8al·§8af와의 관계.** 포화 50 ms(사전등록 §3.4 규칙)는 그대로 선다. 본 절이 바꾸는 것은 **해석**이다: foundations §7.6이 "기억 1.2–1.8 스텝 × 3–4 = 5 스텝"으로 50 ms를 설명한 것은 2 % 문턱으로 재면 성립하지 않는다(2–3 스텝). §7.5의 "K≥5 포화 8.1 %"는 0.1 % 꼬리를 본 수치다. 두 문장은 "OU cold start는 20–30 ms에 끝나고, 50 ms까지 남는 결손은 OU 밖의 척도가 정한다"로 정정해야 한다.
2. **§8ae와의 관계.** §8ae의 "84 %가 cold start"는 *학습되지 않은 절단*의 warm-up이고, 본 절의 cold start는 *최적 필터*의 정보 축적이다. 둘은 다른 양이며, 후자는 20 ms 결손 −0.066의 26–43 %만 설명한다.
3. **§8ar·plasma_physics §1.4와의 관계.** 50 ms 옆에 서는 척도는 이제 넷이다: OU 기억 12–18 ms(가장 짧음), $\tau_{\rm eq}$ 18–39 ms, 측정 $\tau_{\rm rec}$ 55–64 ms(가장 가까움), $\tau_{\rm build}$ 4–17 / 42–192 ms(정의 의존). "세 척도가 같은 pedestal 물리"는 **자릿수 일치**로만 주장한다 — $\tau_{\rm eq}$는 쿨롱 충돌, $\tau_{\rm build}$는 수송 파워 밸런스, 필터 기억은 잡음비의 함수로 서로 다른 물리량이고, 본 데이터는 $n_e,T_e$를 물리 단위로 갖지 않아 $\tau_{\rm eq}$를 shot별로 계산할 수 없다.
4. **한계.** (i) OU는 단일 척도·백색 구동이며 ELM형 급락–회복(두 번째 결정론적 척도)과 MNAR 결측을 담지 않는다. (ii) §8al의 결손은 full-block LSTM 대비이지 참 최적 필터 대비가 아니므로, 결손에는 근사 손실과 빠른 진단의 cold start가 함께 들어 있다. (iii) 이벤트 검출은 한 스텝 25 % 규칙이라 fit-failure 꼬리와 섞이며, 회복형 집단은 "되돌아온 사건"을 고른 것이므로 $y_\infty$에 선택 편향이 있다. (iv) $\tau_{\rm build}$의 두 정의는 11배 갈린다 — 어느 쪽이 KSTAR RMP 체제의 실제 재건을 재는지는 $D_\alpha$·pedestal 프로파일 없이 정할 수 없다.
5. **뒤집을 측정.** (a) `seq_v2` 도달 사다리에서 결손을 **사건 뒤 경과 시간으로 층화**해 재기: 회복형 사건 뒤 0–60 ms 행에서 결손이 60 ms까지 남고 그 밖의 행에서는 20–30 ms에 사라지면 본 절의 분해가 확인된다(학습 불필요, 동결 산출물로 가능). (b) 블록별 $\tau_{\rm rec}$와 블록별 포화 문맥의 상관. (c) $D_\alpha$ 한 채널 — ELM 시각을 외생적으로 주면 $\tau_{\rm rec}$ 검출의 fit-failure 오염이 사라진다.

---

#### 시간 t의 식

아래는 본 절의 모든 주장을 시간 $t$의 명시적 함수로 모은 것이다. $\Delta=10$ ms, $t_{\rm ctx}=k\Delta$(문맥), $t_{\rm gap}=g\Delta$(마지막 관측에서 예측 시각까지).

**(T1) 타깃의 생성 모형(OU SDE)과 해.**
$$
\frac{dy}{dt}=-\frac{y-\bar m}{\tau}+\sqrt{\frac{2\sigma_s^2}{\tau}}\,\xi(t),\qquad
y(t)=\bar m+(y_0-\bar m)e^{-t/\tau}+\sqrt{\frac{2\sigma_s^2}{\tau}}\int_0^t e^{-(t-s)/\tau}\,dW(s),
$$
$$
C(h)=\sigma_s^2e^{-|h|/\tau},\qquad \gamma(h)=\sigma_n^2+\sigma_s^2\bigl(1-e^{-h/\tau}\bigr),\qquad \tau=81\ \mathrm{ms}\ (\text{강건}),\ 74\ \mathrm{ms}\ (\text{고전}).
$$

**(T2) 이온 에너지 식의 선형 응답 — $T_i$는 $T_e$의 저역통과.**
$$
\frac{dT_i}{dt}=\frac{T_e(t)-T_i}{\tau_{\rm eq}}-\frac{T_i-T_i^{\rm tr}}{\tau_{E,i}}+\frac{S_i}{\tfrac32n_i},\qquad
\frac1{\tau_c}=\frac1{\tau_{\rm eq}}+\frac1{\tau_{E,i}},
$$
$$
h(t)=\frac{1}{\tau_{\rm eq}}e^{-t/\tau_c}\,\Theta(t),\qquad
T_i(t)=\int_{-\infty}^{t}h(t-s)\,T_e(s)\,ds+(\text{정상항}),\qquad
T_i(t)-T_e=\bigl(T_i(0)-T_e\bigr)e^{-t/\tau_{\rm eq}}\ (T_e\ \text{고정, 수송 무시}),
$$
$\tau_{\rm eq}=18$–39 ms(CES 구석), $\tau_{E,i}=77$–140 ms. 백색 구동을 받으면 (T2)는 정확히 (T1)이다.

**(T3) 최적 인과 필터의 가중과 예측 — 지연 $s$의 관측에 거는 무게.**
$$
\hat y(t)=\bar m+\sum_{j\ge0}K\rho\,(1-K)^{j}\rho^{j}\bigl(\tilde y(t-\Delta-j\Delta)-\bar m\bigr)
\;\approx\;\bar m+\int_{\Delta}^{\infty}h_K(s)\bigl(\tilde y(t-s)-\bar m\bigr)ds,\qquad
h_K(s)=\frac{K\rho}{\Delta}\,e^{-(s-\Delta)/\tau_K},
$$
$$
\tau_K=L\Delta=\frac{\Delta}{-\ln(\rho(1-K))}=11.7\ \mathrm{ms}\ (\text{강건}),\ 18.0\ \mathrm{ms}\ (\text{고전}).
$$

**(T4) cold-start MSE를 문맥 시간의 함수로.**
$$
\mathrm{MSE}(t_{\rm ctx})=\mathrm{MSE}_\infty+\delta_0\,r^{\,t_{\rm ctx}/\Delta}
=\mathrm{MSE}_\infty+\delta_0\,e^{-t_{\rm ctx}/\tau_{\rm cs}},\qquad
\tau_{\rm cs}=\frac{\tau_K}{2}=\frac{\Delta}{-2\ln(\rho(1-K))},
$$
$\tau_{\rm cs}=5.8$ ms(강건), 9.0 ms(고전); 과도기에는 $r$ 대신 더 작은 $\rho^2(1-K_k)(1-K_\infty)$가 들어가 더 빠르다. 2 % 도달 시각 $t_\varepsilon=\tau_{\rm cs}\ln\bigl(\delta_0/(\varepsilon\,\mathrm{MSE}_{\rm PCHIP})\bigr)=23.5$ ms(강건)·26.9 ms(고전). **측정 곡선**(§8al, $k=2$–6 기하 적합)은
$$
\mathrm{deficit}(t_{\rm ctx})=-0.214\,e^{-t_{\rm ctx}/\tau_{\rm cs,eff}},\qquad \tau_{\rm cs,eff}=\frac{\Delta}{-\ln0.559}=17.2\ \mathrm{ms},
$$
즉 OU의 $\tau_{\rm cs}$보다 1.9–3.0배 느리다.

**(T5) 간극 시간의 함수로서의 인과 MSE(교량 대 외삽, foundations §7.7).**
$$
\mathrm{MSE}_{\rm causal}(t_{\rm gap})=e^{-2t_{\rm gap}/\tau}P_\infty+\sigma_s^2\bigl(1-e^{-2t_{\rm gap}/\tau}\bigr)+\sigma_n^2,
$$
강건 적합값(`pred.by_gap`): 10/20/30/50/80/120/200 ms에서 3170 / 3839 / 4363 / 5092 / 5690 / 6033 / 6208 eV² — $t_{\rm gap}\to\infty$에서 $\sigma_s^2+\sigma_n^2=6237$으로 포화.

**(T6) 사건 뒤 회복 곡선(데이터 적합).** 급락 시각 $t_0$에서
$$
T_i(t)=T_{i,\rm pre}\Bigl[y_\infty-(y_\infty-y_0)\,e^{-(t-t_0)/\tau_{\rm rec}}\Bigr],\quad t\ge t_0,
$$
회복형(n 2,783): $y_0=0.60$, $y_\infty=1.16$, $\tau_{\rm rec}=60.3$ ms(중앙값 곡선), 사건별 중앙값 59.0 ms; 지속(3점 기준, n 7,691): $y_0=0.59$, $y_\infty=0.84$, $\tau_{\rm rec}=43.5$ ms. 급락 자체는 $0.1$–1 ms(이상 PB 성장)라 격자에서는 $t_0$의 한 점 계단이다.

**(T7) pedestal 재건(파워 밸런스).**
$$
\frac{dW_{\rm ped}}{dt}=P_{\rm loss}-\frac{W_{\rm ped}}{\tau_{E,\rm ped}}
\;\Rightarrow\;
W_{\rm ped}(t)=W_{\rm ped}^{\infty}-\Delta W_{\rm ELM}\,e^{-(t-t_0)/\tau_{\rm build}},\qquad
\tau_{\rm build}=\frac{W_{\rm ped}}{P_{\rm loss}}=\frac{\tfrac32n(T_e+T_i)V_{\rm ped}}{P_{\rm loss}},
$$
껍질 체적 4–17 ms, 플라즈마 체적 42–192 ms($P_{\rm loss}=2$ MW).

**(T8) 회복 형태의 식별 가능성을 문맥 시간의 함수로(CRLB).** $y(t)=a-b\,e^{-(t-t_0)/\tau_{\rm rec}}$를 $t_0\le t<t_0+t_{\rm ctx}$의 잡음 관측에서 적합해 $t_0+t_{\rm ctx}$를 외삽할 때
$$
\mathrm{Var}\bigl[\hat y(t_0+t_{\rm ctx})\bigr]=\mathbf g^\top\Bigl(\tfrac{1}{\sigma_n^2}\mathbf J^\top\mathbf J\Bigr)^{-1}\mathbf g,\qquad
\mathbf J_{k\cdot}=\Bigl(1,\,-e^{-k\Delta/\tau_{\rm rec}},\,-\tfrac{b\,k\Delta}{\tau_{\rm rec}^2}e^{-k\Delta/\tau_{\rm rec}}\Bigr),
$$
$\tau_{\rm rec}=50$ ms, $b=180$ eV, $\sigma_n=39$ eV에서 $\mathrm{Var}/\sigma_n^2$ = 12.7 (30 ms), 5.05 (40), 2.90 (50), 1.94 (60), 1.42 (70), 0.72 (100), 0.33 (150): 잡음 바닥을 넘는 시각은 $t_{\rm ctx}\approx1.5\,\tau_{\rm rec}$.

**(T9) 종합 — 결손의 두 성분(가설, §4.5(a)가 검정).**
$$
\mathrm{deficit}(t_{\rm ctx})\;\approx\;\underbrace{-\frac{\delta_0}{\mathrm{MSE}_{\rm PCHIP}}e^{-t_{\rm ctx}/\tau_{\rm cs}}}_{\text{OU 잡음 평균화, }\tau_{\rm cs}=6\text{–}9\ \mathrm{ms}}
\;+\;\underbrace{D_{\rm rec}(t_{\rm ctx};\tau_{\rm rec})}_{\text{회복 형태 식별, }\tau_{\rm rec}\approx60\ \mathrm{ms}}
\;+\;\underbrace{O(1/t_{\rm ctx})}_{\text{빠른 진단 수준 보정}},
$$
첫 항은 20–30 ms에 사라지고(측정 20 ms 결손의 26–43 %), 둘째 항이 50 ms 포화를 정한다.


### P04 캠페인 드리프트는 계측·기하 물리이고, shot별 표준화는 올바른 관측 모형이다

**스크립트**
- `scratchpad/p04\p04_campaign_drift.py` (per-file 통계·R²·η²·편상관·§8n 재현·EWMA 실측 → `p04_per_file_stats.csv`, `p04_results.json`)
- 같은 폴더의 `p04_extra.py` (§8n 정확 재현·run-day 묶음 R²·"공짜 손실" 검정 → `p04_extra_results.json`)
- 입력: `data/s*.csv` 641개, held-free(`STUCK_GAP` 0.5 s 블록 내 반복값 제거), `CES_TI > 3000 eV` 컷, 파이프라인과 같이 타깃이 하나라도 관측된 행만 사용. 모델 학습·채점 없음, 추적 파일 수정 없음.

#### 1. 유도: 세 진단의 관측 모형

**CES (타깃).** C VI 529.05 nm 선형의 Doppler 폭과 이동(foundations §1.1):
$$
T_i=\frac{m_C c^2}{k}\left(\frac{\sigma_\lambda}{\lambda_0}\right)^2,\qquad
V_\varphi=c\,\frac{\lambda_c-\lambda_0}{\lambda_0}.
$$
$T_i$는 **파장의 비** $\sigma_\lambda/\lambda_0$만으로 정해지므로 검출기 이득·광학 투과율·빔 전류가 어떻게 변해도 1차에서 불변이다(광자수 $N$은 §1.2의 분산 $\sigma_T/T=\sqrt{2/N}$에만 들어간다). $V_\varphi$는 기준선 $\lambda_0$에 대한 이동이므로 분광기 파장 교정 하나에만 의존한다. 두 타깃 모두 **절대 물리 단위(eV, km/s)** 이고, 캠페인 내내 같은 양을 같은 단위로 잰다.

**ECEI.** 광학적으로 두꺼운 2차 조화파 X-모드에서 복사 세기는 $I_\omega\propto T_e(R(\omega))$ (foundations §2.1)이지만 채널에 기록되는 것은 라디오미터 전압
$$
V_{\rm ECEI}=G(f)\,\eta\,T_e\!\big(R(f)\big)+V_0 .
$$
$G(f)$는 IF 체인 이득, $\eta$는 안테나·거울·창의 광학 결합(정렬, 코팅, 진공창 오염)이며, $R(f)=\frac{e}{2\pi m_e}\cdot\frac{B_0R_0}{f}$는 $B_T$ 설정이 바뀌면 다른 자속면으로 옮겨가고 Shafranov 이동(plasma physics §2.1, $\Delta\propto\beta_p$)은 같은 $R$을 다른 $\psi$에 대응시킨다. 즉 **shot 사이에 변하는 것은 $G\eta$(계측)와 $R(f)\mapsto\psi$(기하)** 이고, shot 안에서 변하는 것은 $T_e(\psi,t)$(플라즈마)다.

**BES.** 빔 발광(foundations §2.2)
$$
I_{\rm BES}=G_{\rm det}\,\Omega_{\rm view}\,n_b(E_b,I_b;\text{shine-through})\,n_e\,\langle\sigma v\rangle_{\rm exc}(E_b)+I_0 .
$$
빔 전류 $I_b$, 빔 에너지 $E_b$, 시선 입체각 $\Omega_{\rm view}$, 검출기 이득 $G_{\rm det}$, 배경 $I_0$는 **날마다 설정되고 캠페인 동안 노화**하며, shot 안에서는 $n_e(t)$만 움직인다($\delta I/I\approx\delta n_e/n_e$).

**공통 형태.** 세 진단 모두 shot $f$에 대해 $x_f=g_f\,s_f+c_f$ ($g_f,c_f$: shot 상수, $s_f$: 플라즈마 상태). shot 내부 통계 $\mu_f=g_f\bar s_f+c_f$, $\sigma_f=|g_f|\,\sigma_{s,f}$이므로 $z_f=\operatorname{sgn}(g_f)(s_f-\bar s_f)/\sigma_{s,f}$: **$g_f, c_f$가 정확히 소거되고** 플라즈마 상태의 shot 내부 형태만 남는다. 전역(학습 파일) 통계로 정규화하면 $z^{\rm glob}_f=(g_f s_f+c_f-\mu_{\rm tr})/\sigma_{\rm tr}$로 shot 상수가 입력에 그대로 남아, 모델은 $s\mapsto T_i$가 아니라 $(g_f s+c_f)\mapsto T_i$를 배워야 하고 $g_f$가 학습 범위를 벗어나면(foundations §14) 오차에 $(\nabla f^\top\delta)^2$이 더해진다.

**잃는 정보.** shot별 표준화는 $(\mu_f,\sigma_f)$ 두 숫자, 즉 **shot 사이 수준과 척도**를 버린다. 이 손실이 공짜인 조건은
$$
I\big(\mu_f,\sigma_f;\,T_i(t)\ \big|\ \text{CES 이력},\ z_f(\cdot)\big)=0 ,
$$
즉 CES 이력 앵커(절대 단위의 $T_i$ 수준을 이미 운반)와 shot 내부 형태 $z_f$가 주어졌을 때 빠른 진단의 절대 수준이 $T_i$에 대해 추가 정보를 갖지 않는 것. 모델 없이 검정 가능한 근사는 (i) shot별 이득 $\beta_f=\partial T_i/\partial z_f$(eV/σ)가 절대 수준에 의존하는가($T_i$ 수준 통제 편상관), (ii) $T_i$의 shot 내부 진폭이 절대 수준에 의존하는가.

#### 시간 t의 식

두 시간 척도를 구분한다. $t\in[0,T_f]$는 shot $f$ 안의 시각(10 ms 격자, $T_f\sim$ 수 초), $D_f$는 그 shot의 달력 시각(운전일, shot 번호가 단조 대리). 계측 상수는 $D$의 함수이되 shot 안에서는 상수다.
$$
\frac{d\ln g}{dD}\neq0,\qquad \left|\frac{\dot g}{g}\right|T_f\ll1 .
$$

**관측 모형 (shot $f$, 시각 $t$).**
$$
\begin{aligned}
T_i(t)&=\frac{m_Cc^2}{k}\left(\frac{\sigma_\lambda(t)}{\lambda_0}\right)^2, &
V_\varphi(t)&=c\,\frac{\lambda_c(t)-\lambda_0}{\lambda_0} &&\text{(}D\text{ 의존 없음)}\\[2pt]
V_{\rm ECEI}(t)&=G(f;D_f)\,\eta(D_f)\;T_e\!\big(\psi(R(f;B_T(D_f)),\,\Delta(t)),\,t\big)+V_0(D_f),\\[2pt]
I_{\rm BES}(t)&=G_{\rm det}(D_f)\,\Omega(D_f)\,n_b\!\big(E_b(D_f),I_b(t)\big)\,n_e(t)\,\langle\sigma v\rangle_{\rm exc}\!\big(E_b(D_f)\big)+I_0(D_f),\\[2pt]
V_{\rm MC}(t)&=-A\,\dot B(t)\quad(\bar V_{\rm MC}=0,\ \text{수준 없음}).
\end{aligned}
$$
요약 형태:
$$
x_f(t)=g(D_f)\,s_f(t)+c(D_f),\qquad 0\le t\le T_f .
$$
여기서 $s_f(t)$가 플라즈마($T_e(\psi,t)$, $n_e(t)$), $g(D),c(D)$가 계측·기하다. Shafranov 이동 $\Delta(t)$와 빔 전류 $I_b(t)$는 shot 안에서 변하는 항이므로 $s_f(t)$ 쪽에 속한다(한계 1).

**오프라인 shot별 표준화 (shot 전체를 본 뒤).**
$$
\mu_f=\frac1{T_f}\int_0^{T_f}x_f(s)\,ds,\qquad
\sigma_f^2=\frac1{T_f}\int_0^{T_f}\big(x_f(s)-\mu_f\big)^2ds,\qquad
z_f(t)=\frac{x_f(t)-\mu_f}{\sigma_f}=\operatorname{sgn}g(D_f)\;\frac{s_f(t)-\bar s_f}{\sigma_{s,f}} .
$$
$g(D_f),c(D_f)$는 $t$에 무관하므로 적분 밖으로 나와 소거된다. $z_f(t)$의 시간 구조(미분, 자기상관, $T_i(t)$와의 교차상관)는 $s_f(t)$의 것과 동일하다.

**전역 표준화 (학습 파일 집합 $\mathcal F_{\rm tr}$).**
$$
z^{\rm glob}_f(t)=\frac{g(D_f)\,s_f(t)+c(D_f)-\mu_{\rm tr}}{\sigma_{\rm tr}},\qquad
\mathbb E\big[z^{\rm glob}_f(t)\,\big|\,D_f\big]=\frac{g(D_f)\bar s+c(D_f)-\mu_{\rm tr}}{\sigma_{\rm tr}} .
$$
우변이 달력 시각 $D$의 함수로 남고, train 구간과 test 구간의 차이가 §8n의 $z=1.22$(BES)·$0.53$(ECEI)이다. 타깃에는 $g,c$가 없으므로 같은 식의 값이 $0.115$다.

**온라인(인과) 표준화.** 시정수 $\tau_w$의 EWMA 평균·분산과 확장 윈도 평균:
$$
m(t)=x(0)\,e^{-t/\tau_w}+\frac1{\tau_w}\int_0^{t}e^{-(t-s)/\tau_w}x(s)\,ds,\qquad
v(t)=\frac1{\tau_w}\int_0^{t}e^{-(t-s)/\tau_w}\big(x(s)-m(s)\big)^2ds,
$$
$$
z^{\rm on}(t)=\frac{x(t)-m(t)}{\sqrt{v(t)}},\qquad
m^{\rm exp}(t)=\frac1t\int_0^t x(s)\,ds .
$$
격자 $\Delta t=10$ ms에서 $\alpha=1-e^{-\Delta t/\tau_w}$, $m_k=(1-\alpha)m_{k-1}+\alpha x_k$ (스크립트 `ewma_z`가 이 식).

**추세 위의 편향.** $x(s)=a+bs$, $m(0)=a$를 넣으면
$$
m(t)=a+b\,(t-\tau_w)+b\,\tau_w e^{-t/\tau_w},\qquad
e(t)\equiv m(t)-x(t)=-\,b\,\tau_w\big(1-e^{-t/\tau_w}\big)\;\xrightarrow{t\gg\tau_w}\;-b\,\tau_w ,
$$
$$
v(t)\to(b\tau_w)^2+\sigma_n^2,\qquad
z^{\rm on}(t)\to\frac{b\tau_w}{\sqrt{\sigma_n^2+b^2\tau_w^2}}\quad(\text{추세가 상수 offset으로 변함}),
$$
$$
m^{\rm exp}(t)=a+\frac{bt}{2},\qquad e^{\rm exp}(t)=-\frac{bt}{2}\quad(\text{발산, 단 }t=T_f\text{에서 }\mu_f\text{와 일치}).
$$
동치 서술: EWMA는 전달함수 $1/(1+i\omega\tau_w)$의 저역 통과이고 $x-m$은 고역 통과이므로, 모서리 $1/\tau_w$ 아래의 shot 내부 추세(오프라인 $z_f$가 보존하는 성분)를 깎는다. 데이터의 $|b|$ 중앙값 0.309 σ/s(BES)·0.273 σ/s(ECEI)를 넣으면 $\tau_w=1$ s에서 $|e|\to0.31$·$0.27$ σ, 실측 2–8 s 구간 0.29–0.33 σ(§3 표).

#### 2. 데이터 검증 (641 파일)

**(a)·(b) 연대순이 설명하는 몫과 shot 사이 분산 몫**

| 양 (per-file 평균) | $R^2$ on shot 번호 (선형) | $R^2$ on 캠페인 1/3 | adj. $R^2$ on 5-shot 묶음 (run-day 대리) | $\eta^2$ = shot 사이 분산 / 총 분산 |
|---|---:|---:|---:|---:|
| BES 평균채널 | **0.291** | 0.243 | 0.663 | **0.838** |
| ECEI 평균채널 | 0.018 | 0.062 | 0.632 | **0.909** |
| MC 평균채널 | 0.000 | 0.001 | 0.056 | 0.003 |
| CES_TI | 0.021 | 0.057 | 0.538 | 0.494 |
| CES_VT | 0.000 | 0.002 | 0.011 | 0.282 |

캠페인 1/3별 per-file 평균: BES 0.192 → 0.158 → 0.111(단조 감소, corr(shot, BES) = −0.540), ECEI 0.465 → 0.833 → 0.670(비단조, 날 단위 계단), CES_TI 582 → 663 → 496 eV, CES_VT 57.0 → 64.4 → 59.1 km/s.

읽는 법. 빠른 진단은 **전체 분산의 84 % / 91 %가 shot 사이 수준**이고, 타깃은 49 % / 28 %다. 수준의 연대 구조는 두 종류인데, BES는 캠페인 전체에 걸친 단조 추세($g(D)$의 장기 노화·빔 조건 변화, 선형 $R^2$ 0.29)가 있고 ECEI는 선형 추세가 없는 대신($R^2$ 0.02) 5-shot 묶음이 63 %를 설명한다. 즉 **같은 날·같은 설정의 shot끼리는 수준을 공유하고 날이 바뀌면 뛴다** — 정렬·결합·$B_T$ 설정이 바뀔 때 $G\eta$와 $R(f)$가 바뀌는 $g(D),c(D)$ 모형의 서명이다. $T_i$도 5-shot 묶음에서 54 %(같은 날 같은 시나리오를 반복)이므로 날 단위 구조만으로는 계측과 플라즈마를 가르지 못한다. 가르는 것은 아래 (c)와 §8n 재현의 이득 서명이다.

**(c) 절대 수준과 $T_i$ 수준의 상관, shot 내부 상관과 대조**

| 쌍 | shot 사이 $r$ | shot 번호 통제 편상관 | 수준의 연대 성분 vs $T_i$ | 수준의 잔차 성분 vs $T_i$ | shot 내부 $r$ (Fisher 평균) |
|---|---:|---:|---:|---:|---:|
| BES – CES_TI | +0.373 | +0.355 | +0.143 | +0.351 | **+0.401** (85 % 양수, 625 shot) |
| ECEI – CES_TI | +0.153 | +0.176 | −0.143 | +0.174 | **+0.358** (85 % 양수, 624 shot) |
| MC – CES_TI | −0.007 | −0.004 | — | — | — |
| BES – CES_VT | −0.028 | −0.027 | — | — | — |
| ECEI – CES_VT | +0.037 | +0.036 | — | — | — |

빠른 진단 수준의 **연대 성분**(shot 번호에 회귀한 적합값)은 $T_i$와 $|r|=0.14$로만 이어지며 ECEI는 부호까지 반대다. 수준 가운데 $T_i$와 이어지는 부분은 연대와 무관한 잔차 성분(BES +0.35)이고, 가장 강한 결합은 shot 내부(+0.40/+0.36)다. 플라즈마 물리가 캠페인 추세를 만든다면 절대 단위로 재는 $T_i$가 같은 추세를 따라야 하는데, corr(shot, $T_i$ 수준) = −0.143, corr(shot, BES 수준) = −0.540이다.

**이득 서명(§8n의 train → test 채널별 비).** 순수 곱셈 이득 변화면 평균비 = 표준편차비다. `data/.campaign_shift_diagnosis.json`에서 ECEI 4채널은 평균비 0.492–0.609, 표준편차비 0.555–0.637로 **거의 같다**(곱셈 이득 $G\eta$, $c$ 거의 불변). BES 9채널은 평균비 0.545–0.591, 표준편차비 0.706–0.774로 평균이 더 많이 떨어졌다(곱셈 이득에 더해 배경 항 $c(D)$의 변화, 또는 $\delta n/n$ 증가). 두 경우 모두 $g s+c$ 2-모수 족 안에 있고, shot별 z-score가 소거하는 바로 그 두 모수다. 타깃은 CES_TI 평균 이동 0.147 σ, CES_VT 0.083 σ.

**"공짜 손실" 조건의 검정.** shot별 이득 $\beta_f=\mathrm{cov}(T_i,z_f)/\mathrm{var}(z_f)$ (eV per within-shot σ), $T_i$ 관측 8점 이상인 shot:

| 진단 | $\beta_f$ 중앙값 (IQR) | corr($\beta$, 절대수준) | 편상관 \| $T_i$ 수준 | corr($\beta$, $T_i$ 수준) | $R^2$($\beta$ ~ $T_i$ 수준) | $R^2$($\beta$ ~ 둘 다) | 수준이 더하는 $R^2$ | 편상관($T_i$ 내부 σ, 수준 \| $T_i$ 수준) | 원단위 기울기 $\partial T_i/\partial x$의 shot 간 CV |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BES | 66.9 eV/σ (136.5) | +0.049 | −0.134 | +0.447 | 0.200 | 0.214 | **0.014** | −0.284 | 1.30 |
| ECEI | 46.4 eV/σ (117.5) | +0.223 | +0.173 | +0.368 | 0.135 | 0.161 | **0.026** | +0.434 | 4.16 |

이득 $\beta_f$를 설명하는 것은 CES 이력이 이미 운반하는 $T_i$ 수준(20 % / 14 %)이고, 빠른 진단의 절대 수준이 그 위에 더하는 몫은 1.4 % / 2.6 %다. 반대로 전역 정규화가 학습해야 하는 원단위 기울기 $\partial T_i/\partial x$는 shot 사이 CV 1.3(BES)·4.2(ECEI)로 요동친다. 조건은 근사적으로 성립하되 **정확히 0은 아니다**. ECEI 절대 수준은 $T_i$의 shot 내부 진폭과 +0.43의 편상관을 남기며, 이것이 §8s가 측정한 작은 음의 점추정(평균 −0.036, 유의 0/4)의 물리적 정체다.

#### 3. 기록과의 연결, 그리고 온라인 판

**§8n 재현.** 같은 정의(포함 모집단, 입력 완전 행 전부, held 규칙만)로 train 30801–31991 → test 32312–32751의 $z=|\Delta\mu|/\sigma_{\rm train}$ 채널 중앙값을 다시 계산하면 BES **1.218**, ECEI **0.529**, MC 0.007, CES 그룹 **0.115**(TI 0.147, VT 0.083의 중앙값), 표준편차비 BES 0.750, ECEI 0.620으로 §8n 표와 소수 셋째 자리까지 일치한다. 컷 모집단·파이프라인 행으로 바꾸면 BES 1.308, ECEI 0.542, TI 0.267, VT 0.083. "시간 t의 식"의 전역 표준화 식에 대입하면 $z_{\rm BES}=1.22$는 $\mathbb E[z^{\rm glob}|D_{\rm test}]-\mathbb E[z^{\rm glob}|D_{\rm tr}]$, 즉 **$g(D),c(D)$의 이동**이고 $z_{\rm CES}=0.115$는 **$\bar s$ 자체의 이동**이다. 계측·기하가 플라즈마보다 5–11배 움직였다는 것은 이 두 항의 비다.

**§8s의 "측정 가능한 비용 없음".** shot별 z-score 뒤에는 $g(D),c(D)$가 사라져 캠페인 이동이 구성상 0이 되므로 temporal split에서 +0.155(4/4 유의)가 돌아오고, 랜덤 split에서 치르는 값은 위 표의 "수준이 더하는 $R^2$" 1.4–2.6 %에 해당하는 −0.036(0/4 유의)이다. §8ab 블록 6에서 shot별 정규화를 갖춘 `seq_v2`가 temporal TEST에서 PCHIP 4/4·인과 GP 4/4를 이기고 window OFF가 0/4–2/4로 무너지는 것도 같은 항등식의 결과다. OFF 팔은 BES 입력이 학습 분포에서 1.2 σ 바깥에 놓인 채 평가된다.

**온라인(배치 가능) 판과 그 편향.** 위 EWMA 식에서 정상 상태 편향은 **$-b\tau_w$로 포화**하고 분산 추정에 $(b\tau_w)^2$이 더해지며, 확장 윈도는 $-bt/2$로 발산하되 shot 끝에서 오프라인과 일치한다. 데이터의 shot 내부 추세는 연속 블록별 선형 기울기 $|b|$ 중앙값 **0.309 σ/s**(BES)·**0.273 σ/s**(ECEI), p90 1.39·1.28 σ/s(647 블록)이므로 $\tau_w=1$ s의 예측 편향은 전형 0.31·0.27 σ, 10 % 꼬리 1.3–1.4 σ다.

실측(온라인 $z^{\rm on}(t)$와 오프라인 $z_f(t)$의 평균 $|\Delta z|$, 블록 시작 후 경과 시간별):

| 진단 | 추정기 | 0–0.5 s | 0.5–1 s | 1–2 s | 2–4 s | 4–8 s | >8 s |
|---|---|---:|---:|---:|---:|---:|---:|
| BES | EWMA $\tau_w$=0.5 s | 2.10 | 0.85 | 0.46 | 0.33 | 0.41 | 0.52 |
| BES | EWMA $\tau_w$=1 s | 2.87 | 1.19 | 0.63 | **0.31** | **0.29** | 0.46 |
| BES | EWMA $\tau_w$=2 s | 3.98 | 1.72 | 0.98 | 0.51 | 0.30 | 0.29 |
| BES | 확장 윈도 | 1.11 | 0.76 | 0.53 | 0.34 | 0.16 | **0.07** |
| ECEI | EWMA $\tau_w$=0.5 s | 1.72 | 0.75 | 0.49 | 0.35 | 0.44 | 0.85 |
| ECEI | EWMA $\tau_w$=1 s | 2.37 | 1.00 | 0.57 | **0.29** | **0.33** | 0.71 |
| ECEI | EWMA $\tau_w$=2 s | 3.33 | 1.44 | 0.85 | 0.41 | 0.24 | 0.52 |
| ECEI | 확장 윈도 | 0.96 | 0.73 | 0.55 | 0.32 | 0.20 | 0.34 |

$\tau_w=1$ s의 정상 구간(2–8 s) 실측 0.29–0.33 σ는 유도값 $b\tau_w$ = 0.31/0.27 σ와 맞는다. 첫 0.5 s의 2–4 σ 오차는 평균 $m(t)$가 아니라 **표본 수 $\approx\tau_w/\Delta t$개로 추정한 분산 $v(t)$** 가 지배하며, 1–2 s가 지나야 0.5 σ 아래로 내려온다. 배치형은 추세를 깎지 않는 확장 윈도가 오프라인 추정기에 가장 가깝고, 어느 쪽이든 **학습과 추론에 같은 인과 추정기를 써야** 편향이 모형의 일부가 된다. 그 조건에서 남는 비용은 첫 1–2 s의 워밍업뿐이며, §8s가 "미측정"으로 남긴 간격이 이 표의 크기다.

#### 4. 한계

1. **곱셈 이득 + 덧셈 offset 가정.** $g(D)s_f(t)+c(D)$는 1차 모형이다. ECEI의 $\tau<1$ 영역(foundations §2.1), BES의 빔 에너지에 따른 $\langle\sigma v\rangle_{\rm exc}$ 변화, shot 안의 빔 전류 변조 $I_b(t)$와 Shafranov 이동 $\Delta(t)$는 $s_f(t)$의 **shot 내부 곱셈·비선형 항**이라 z-score가 소거하지 못한다. §8n의 ECEI 평균비≈표준편차비 일치는 1차 모형이 지배적임을 보이지만 잔차를 재지는 않는다.
2. **계측과 플라즈마의 완전한 분리는 데이터만으로 불가능하다.** BES 수준은 $g\,n_b$와 $n_e$의 곱이라 $n_e$ 캠페인 추세(연료·재순환, plasma physics §10)가 섞여 있을 수 있다. 본 증명은 "타깃이 따라가지 않는 추세"라는 간접 논거와 이득 서명에 의존하며, 빔 전류·$E_b$·ECEI 교정 로그가 있어야 직접 분해된다(세 양 모두 데이터에 없음, foundations 부록 A 성격).
3. **공짜 조건은 근사다.** 절대 수준이 이득 $\beta_f$에 더하는 $R^2$는 1.4–2.6 %지만 ECEI 수준과 $T_i$ 내부 진폭의 편상관 +0.43은 0이 아니다. 이는 이력 앵커가 진폭 정보도 일부 운반하므로 상한이고, 모델을 끼운 정확한 조건부 정보량은 §8s의 −0.036(0/4 유의) 이상으로는 측정되지 않았다.
4. **온라인 판은 미학습.** EWMA·확장 윈도 표는 오프라인 $z_f(t)$와의 거리이지 skill 비용이 아니다. 같은 추정기로 학습·평가하는 한 팔(§8s "뒤집을 측정" 1번)이 열려 있다.
5. **run-day 대리 변수.** 5-shot 묶음은 날짜의 대리일 뿐 실제 운전일 경계를 모른다. MC는 $\eta^2$ 0.003으로 수준이 없어(무작위 위상 $dB/dt$, §8b.2) 표준화의 대상이 아니라 척도만 맞추는 효과다.


### P05. 3 keV 컷은 pedestal 압력 한계이고, 단일 표본 스파이크는 스펙트럼 적합 실패의 통계다

**명제.** KSTAR pedestal에서 $T_i>3$ keV는 peeling–ballooning(PB) 압력 한계가 배제한다. 따라서 사전등록된 값 컷 `CES_TI_SPIKE_CUT_EV = 3000`(§8ab)은 사후 행 수술이 아니라 물리 제약이며, 컷이 제거하는 행의 통계(1,197행 · 951런 · 85 % 단일행 · 이웃의 13배)는 플라즈마 사건이 아니라 스펙트럼 적합 실패의 통계다.

**스크립트.**
- `scratchpad/p05\p05_pressure_limit.py` → `out.json` (A 압력 한계, B 파워, D 641파일 검열; `py <file>` 또는 `--skip-mc`)
- `…\scratchpad/p05\p05_fit_mc.py` → `mc.json` (C Gaussian 적합 오염 몬테카를로)

가정값(부록 A, `docs/theory_plasma_physics.md`): $I_p=0.6$ MA, $B_T=1.8$ T, $a=0.5$ m, $R=1.8$ m, $\kappa=1.8$, $q_{95}=4$, $n_e=3\times10^{19}$ m$^{-3}$, 흡수 가열 $P=3$ MW. 본 CSV에는 $n_e, T_e, P$가 물리 단위로 없으므로 이 절의 한계값은 "자릿수 경계"이지 shot별 수치가 아니다.

---

#### 1. 유도

##### 1(a) pedestal 압력 한계 → $T_{i,\rm ped}$의 상한

폴로이달 자기장과 그 압력:
$$
B_{\rm pol}=\frac{\mu_0 I_p}{2\pi a\sqrt{(1+\kappa^2)/2}}=0.165\ {\rm T},\qquad
p_{\rm pol}\equiv\frac{B_{\rm pol}^2}{2\mu_0}=10.8\ {\rm kPa},\qquad
\beta_{p,{\rm ped}}=\frac{p_{\rm ped}}{p_{\rm pol}} .
$$

EPED 그림(§6.2): 폭은 kinetic-ballooning 제약 $\Delta_\psi=0.076\sqrt{\beta_{p,{\rm ped}}}$, 높이는 PB 한계가 정한다. 가장 단순한 PB 대리는 ballooning 파라미터
$$
\alpha=-\frac{2\mu_0Rq^2}{B^2}\frac{dp}{dr}\le\alpha_c
\;\;\Longrightarrow\;\;
\left|\frac{dp}{dr}\right|_{\max}=\frac{\alpha_cB^2}{2\mu_0Rq^2},
$$
pedestal에서 $|dp/dr|\approx p_{\rm ped}/\Delta_r$, $\Delta_r=g\,a\,\Delta_\psi$($g$는 자속 팽창 계수, 1 또는 1/2). 두 식을 닫으면 $p_{\rm ped}$가 양변에 들어가 자기일관 상한이 나온다:
$$
p_{\rm ped}\le\left|\frac{dp}{dr}\right|_{\max} g\,a\,0.076\sqrt{\frac{p_{\rm ped}}{p_{\rm pol}}}
\;\;\Longrightarrow\;\;
\boxed{\;p_{\rm ped,max}=\left(\frac{\alpha_cB^2}{2\mu_0Rq^2}\cdot g\,a\cdot0.076\right)^2\frac{2\mu_0}{B_{\rm pol}^2}\;}
$$

| $\alpha_c$ (형상) | $|dp/dr|_{\max}$ [kPa/m] | $p_{\rm ped,max}$ [kPa], $g=1$ | $\beta_{p,{\rm ped}}$ | $\Delta_r$ [cm] | $g=\tfrac12$: $p_{\rm ped,max}$ |
|---|---:|---:|---:|---:|---:|
| 2 (원형) | 89.5 | 1.07 | 0.10 | 1.2 | 0.27 |
| 3 | 134 | 2.41 | 0.22 | 1.8 | 0.60 |
| 4 | 179 | 4.28 | 0.40 | 2.4 | 1.07 |
| 6 (강한 형상, 2차 안정 접근) | 269 | 9.64 | 0.89 | 3.6 | 2.41 |

독립 대조 두 가지. (i) EPED의 전형 $\beta_{N,{\rm ped}}=0.6$–$1.0$(가정)을 KSTAR형으로 환산하면 $p_{\rm ped}=5.2$–$8.6$ kPa($\beta_{p,{\rm ped}}$ 0.48–0.80) — 표의 $\alpha_c=4$–$6$ 사이. (ii) Troyon 전역 한계 $\beta_N\le2.8$–$3.5$는 **체적 평균** 압력을 $\langle p\rangle\le24$–$30$ kPa로 묶는다. 이하 상한은 가장 관대한 $p_{\rm ped,max}\in[4.3,\,9.6]$ kPa($\alpha_c=4$–$6$, $g=1$, 폭 2.4–3.6 cm = §6.2의 "폭 2–3 cm")로 잡는다.

이온 온도로 환산: $p_{\rm ped}=n_e e\,(T_i+T_e)=n_e e\,T_i(1+T_e/T_i)$. pedestal의 $\tau_{\rm eq}$는 8–59 ms(§3.2 표; 500 eV·$3\times10^{19}$에서 13.4 ms)로 $\tau_E=77$ ms(§11.1)보다 짧으므로 $T_i\approx T_e$가 기본이고, NBI 이온 가열 편중을 허용해 $T_i=1.5T_e$까지 본다.

| $n_{e,{\rm ped}}$ [m$^{-3}$] | $T_{i,\max}$ ($T_i=T_e$) [eV] | $T_{i,\max}$ ($T_i=1.5T_e$) [eV] | 3 keV에 필요한 $\beta_{p,{\rm ped}}$ | 3 keV가 허용되는 $n_e$ 상한 |
|---|---|---|---|---|
| $2\times10^{19}$ | 668–1,503 | 802–1,804 | 1.78 (1.48) | $1.0\times10^{19}$ ($1.2\times10^{19}$) |
| $3\times10^{19}$ | 446–1,002 | 535–1,203 | 2.67 (2.22) | 〃 |
| $5\times10^{19}$ | 267–601 | 321–722 | 4.45 (3.71) | 〃 |

**결론 1(a).** 가장 관대한 조합(저밀도 $2\times10^{19}$, $T_i=1.5T_e$, $\alpha_c=6$)에서도 $T_{i,{\rm ped}}\le1.8$ keV, 기본 조합($3\times10^{19}$, $T_i=T_e$)에서는 $\le1.0$ keV — **자릿수 1–2 keV**. 3 keV는 $\beta_{p,{\rm ped}}\ge1.5$–$4.5$(PB 상한 0.89의 1.7–5배, EPED 전형의 3–9배)를 요구하거나 $n_{e,{\rm ped}}\le1.0$–$1.2\times10^{19}$ m$^{-3}$(가정한 H-mode 범위 2–5 아래)를 요구한다. 관측 분포의 p99 2,089 eV·중앙값 592 eV(스크립트 D)는 이 상한 안에 있고, 1,197개의 >3 keV 행은 밖에 있다. 스파이크 중앙값 5,305 eV(이웃 835 eV)에서의 압력 $n_e e(T_i+T_e)=29.5$ kPa는 가장 관대한 PB 상한의 **3.06배**이자 Troyon **체적 평균** 상한(30.1 kPa)과 같다 — pedestal 한 겹이 플라즈마 전체의 $\beta$ 한계 압력을 가져야 한다.

##### 1(b) 시간 척도: 한 격자 안의 13배 상승·하강에 필요한 파워

pedestal 이온 에너지 수지(이온만, pedestal 체적만):
$$
\frac{3}{2}n_eV_{\rm ped}\frac{dT_i}{dt}=P_i(t)-\frac{3}{2}n_eV_{\rm ped}\frac{T_i-T_e}{\tau_{\rm eq}}-\frac{W_i}{\tau_{E,i}},
\qquad
V_{\rm ped}=4\pi^2Ra\sqrt{\tfrac{1+\kappa^2}{2}}\;\Delta_{\rm ped}=51.7\ {\rm m^2}\times0.03\ {\rm m}=1.55\ {\rm m^3}
$$
(플라즈마 체적 16.0 m³의 9.7 %). 손실항을 0으로 두고(가장 관대) 한 격자 $\Delta t=10$ ms 안에 $\Delta T$만큼 오르려면
$$
P_i\ \ge\ \frac{3}{2}\,\frac{n_e\,e\,\Delta T\,V_{\rm ped}}{\Delta t}.
$$

| 스파이크 − 이웃 평균 $\Delta T$ (데이터) | $\Delta W$ [kJ] | $P_i$ [MW] | / $P=3$ MW | / 설치 10 MW(가정 상한) | / pedestal 몫의 이온 가열 (3 MW × 0.5 × 9.7 % = 0.146 MW) | $T_e$가 따라올 때 |
|---|---:|---:|---:|---:|---:|---:|
| 중앙값 5,019 eV | 56.2 | 5.62 | 1.9× | 0.56× | **38.6×** | 11.2 MW |
| p90 11,720 eV | 131 | 13.1 | 4.4× | 1.3× | **90×** | 26.2 MW |
| 최대 14,630 eV | 164 | 16.4 | 5.5× | 1.6× | **112×** | 32.7 MW |

읽는 법. 요구 파워는 **설치 총 가열과 같은 자릿수**(0.6–1.6배)이지 "자릿수 초과"는 아니다 — 정직하게 그렇다. 배제는 세 겹으로 온다. (1) 그 파워가 전부, 손실 없이, 체적의 9.7 %인 pedestal의 **이온에만** 10 ms 안에 들어가야 한다: NBI는 중심부에 쌓이고 이온 몫은 ~½(Stix, §8.2)이므로 pedestal이 실제로 받을 수 있는 이온 가열(0.15 MW)의 **39–112배**다. (2) $\tau_{\rm eq}$ 13 ms $\approx\Delta t$이므로 $T_e$가 같은 프레임 안에 따라오고 요구 파워는 2배(11–33 MW, 설치 상한의 1.1–3.3배)가 된다 — 그리고 ECEI가 같은 프레임에 그 $T_e$ 상승을 봐야 하는데 보지 않는다(§2 (iii)). (3) **하강**: 70 %의 런은 양쪽 이웃이 < 2 keV인 고립 단일행이므로 56 kJ이 다음 10 ms 안에 다시 사라져야 한다. 이웃 온도(835 eV)에서 pedestal 이온 저장 에너지는 9.3 kJ, $\tau_E=77$ ms의 정상 수송 손실은 0.12 MW이므로 하강에는 정상 수송의 **46배**인 5.6 MW의 싱크가 필요하다. 그런 싱크는 ELM 붕괴뿐인데(§6.3, 0.1–1 ms에 $W_{\rm ped}$의 5–20 %), ELM은 PB 한계 **에서** 떨어지는 것이지 PB 한계의 3배에서 돌아오는 것이 아니다. 마지막으로 MHD: 스파이크 압력(PB의 3배)에서 이상 MHD 성장 시간 $1/\gamma\sim R/(0.1v_A)=3.6$ µs($v_A=5.1\times10^6$ m/s)는 프레임의 1/2,800이다 — CES가 보고하는 것은 10 ms **프레임 평균** 선폭이므로, 그 상태가 프레임의 대부분을 점유해야 13배가 나오고, 점유 시간이 짧을수록 필요한 첨두 압력은 더 커진다(§"시간 t의 식" 2). 어느 가지도 플라즈마가 아니다.

##### 1(c) 측정: 단일 프레임 오염은 선폭만 키우고 중심은 남긴다

프레임 $k$의 스펙트럼 모형 $I_k(\lambda)=A\,G(\lambda;\mu,\sigma)+b+c_k(\lambda)$, $G$는 §1.1의 Doppler Gaussian, $c_k$는 그 프레임에만 있는 오염. 4모수 최소제곱의 정규방정식에서 $\mu$ 성분은 $\partial G/\partial\mu\propto(\lambda-\mu)G$로 **홀함수**, $\sigma$ 성분은 $\partial G/\partial\sigma\propto(\lambda-\mu)^2G$로 **짝함수**다. 그러므로 $\mu$에 대해 대칭인 오염($c_k(2\mu-\lambda)=c_k(\lambda)$: 배경 단차, 연속광 버스트, 잘못 뺀 beam-off 배경)은 $\hat\mu$를 1차에서 움직이지 않고 $\hat\sigma$만 키운다 — $T\propto\sigma^2$이므로 온도는 **제곱으로** 부풀고, 이웃 프레임($c=0$)은 §1.2의 CRB 안에 남는다. 비대칭 오염(핫 픽셀, 우주선, 두 번째 선)은 $\hat\mu$를 끌고 간다. 두 부류가 데이터에서 다른 지문을 남긴다는 것이 검정의 요지다.

몬테카를로(`p05_fit_mc.py`, 48픽셀, $\sigma_{\rm true}=4$ px, 선 광자 $N=300$(§1.3·§1.5), 배경 2 ph/px, 비가중 4모수 LSQ, $\sigma$ 하한 0.3 px, 300회/조건; 1 px $\leftrightarrow$ 17 km/s, 1 $\sigma$ $\leftrightarrow$ $v_{th}=69$ km/s):

| 오염 | $T_{\rm fit}/T_{\rm true}$ 중앙값 (p90) | $|\Delta V|$ 중앙값 [km/s] | 지문 |
|---|---|---|---|
| 없음 | 1.00 (1.20) | 3.2 | 기준(상대 sd 0.145; CRB 0.082) |
| 배경 단차, 미적합(pinned), 첨두의 0.5 / 1 / 2 / 5배 | **11.0** (13.9) / **24.1** / **49.9** / **119** | 6.6 / 8.5 / 13.5 / 17.8 | **폭만 ×10–100, 중심 거의 보존** |
| 핫 픽셀·우주선, 첨두 3–10배, 중심에서 0–10 $\sigma$ | 0.006–0.02 (폭이 하한으로 붕괴) | 8 → 413 | 폭 **축소**(dip) + $V$ 스파이크; 수렴 실패 17–86 % |
| 저신호 프레임(빔 노치), $N$ = 30 / 10 / 3 / 0 | 0.69 / 0.045 / 0.009 / 0.014 (p90 ≤ 2.0; >13배는 0–1.8 %) | 19 / 70 / 234 / 241 | 폭 붕괴 + $V$ 난수 |
| 두 성분, 반분리 1 / 2 / 3 / 4 $\sigma$ | 2.6 / 0.44 / 0.41 / 0.43 | 5.9 / 137 / 207 / 278 | 한 성분에 잠김, $V$ 이동 |

**결론 1(c).** 13배의 **상향** 폭 편향을 단일 프레임에서 만들면서 중심을 남기는 메커니즘은 **대칭 오염(배경·연속광 단차)** 하나뿐이다. 핫 픽셀·저신호·두 성분은 모두 폭을 **줄이거나**(§8ab가 센 4,965개의 하향 dip의 후보) $V_{\rm rot}$를 수백 km/s 끌고 간다. 따라서 예측은 "스파이크 행에서 $V_{\rm rot}$는 관측되며 이웃과 다르지 않다"이고, 이는 §1.6($I_{\mu\sigma}=0$, 타깃별 독립 기각)과 같은 방향이다.

---

#### 시간 t의 식

**1. pedestal ELM 주기 — $p_{\rm ped}(t)$는 PB 한계를 넘지 못한다.** ELM 사이 구간 $t\in(t_k,t_{k+1})$에서 수송 시간 $\tau_{\rm build}\sim10$–50 ms(§6.3)로 쌓이고 PB 한계에서 붕괴한다:
$$
\frac{dp_{\rm ped}}{dt}=\frac{p_*-p_{\rm ped}}{\tau_{\rm build}}\quad(t_k<t<t_{k+1}),\qquad
t_{k+1}=\min\{t:\ p_{\rm ped}(t)=p_{\rm PB}\},\qquad
p_{\rm ped}(t_{k+1}^+)=(1-f_{\rm ELM})\,p_{\rm PB},\ f_{\rm ELM}\in[0.05,0.2],
$$
여기서 $p_{\rm PB}=p_{\rm ped,max}(\alpha_c,g)\le9.6$ kPa(표 1). 따라서 모든 $t$에 대해
$$
T_i(t)\ \le\ \frac{p_{\rm PB}}{n_e e\,(1+T_e/T_i)}\ \le\ 1.0\text{–}1.8\ {\rm keV},
\qquad
\bar T_{i,k}\equiv\frac{1}{\Delta t}\int_{t_k}^{t_k+\Delta t}T_i(t)\,dt\ \le\ \max_t T_i(t)
$$
— CES가 보고하는 프레임 평균도 같은 상한을 받는다. RMP 억제 체제(§6.4)에서는 $p_{\rm ped}$가 $p_{\rm PB}$ 아래에 머물므로 상한은 더 느슨해지지 않고 더 조여진다.

**2. 에너지 수지 ODE와 한 격자 안의 상승·하강에 필요한 $P(t)$.**
$$
\frac{3}{2}n_eV_{\rm ped}\,\dot T_i(t)=P_i(t)-\frac{3}{2}n_eV_{\rm ped}\frac{T_i(t)-T_e(t)}{\tau_{\rm eq}}-\frac{3}{2}n_eV_{\rm ped}\frac{T_i(t)}{\tau_{E,i}} .
$$
프레임 $k^*$에서만 $\bar T_{i,k^*}=\bar T_{\rm nbr}+\Delta T$, 양 이웃은 $\bar T_{\rm nbr}$라 하자. 손실항을 0으로 두고(관대) 상승 구간 $(t_{k^*}-\Delta t,\,t_{k^*})$과 하강 구간 $(t_{k^*}+\Delta t,\,t_{k^*}+2\Delta t)$에 상수 파워를 가정하면
$$
P_i(t)=\begin{cases}
+\dfrac{3}{2}\dfrac{n_e e\,\Delta T\,V_{\rm ped}}{\Delta t}=+5.6\ {\rm MW} & t_{k^*}-\Delta t<t<t_{k^*}\\[8pt]
0 & t_{k^*}<t<t_{k^*}+\Delta t\\[4pt]
-\dfrac{3}{2}\dfrac{n_e e\,\Delta T\,V_{\rm ped}}{\Delta t}=-5.6\ {\rm MW} & t_{k^*}+\Delta t<t<t_{k^*}+2\Delta t
\end{cases}
\qquad(\Delta T=5{,}019\ {\rm eV}),
$$
그리고 $\tau_{\rm eq}$(13 ms) $\approx\Delta t$이므로 결합항이 프레임 안에 $T_e$를 끌어올려 실제 요구는 $\approx2\times$가 된다. 음의 가지 $-5.6$ MW는 정상 수송 싱크 $W_i/\tau_{E,i}=0.12$ MW의 46배다. 점유 시간이 프레임보다 짧은 첨두($T_i(t)=T_{\rm nbr}+T_{\rm pk}\,\mathbb 1[t\in(t_{k^*}+\tau_0,\,t_{k^*}+\tau_0+\delta t)]$)로 같은 프레임 평균을 만들면
$$
\Delta T=\frac{\delta t}{\Delta t}T_{\rm pk}\;\Longrightarrow\;T_{\rm pk}=\frac{\Delta t}{\delta t}\Delta T,\qquad
p_{\rm pk}=n_e e\,(T_{\rm nbr}+T_{\rm pk}+T_e)\ \ge\ 3.06\,p_{\rm PB}\cdot\frac{\Delta t}{\delta t},
$$
즉 짧을수록 PB 초과가 커지고, $\delta t$는 이상 MHD 성장 시간 $1/\gamma=R/(0.1v_A)=3.6$ µs $=\Delta t/2{,}800$ 이상 유지될 수 없다. 두 극한($\delta t=\Delta t$, $\delta t\to1/\gamma$) 모두 배제된다.

**3. 프레임 $t$의 스펙트럼과 단일 프레임 오염.** 프레임 $k$가 $[t_k,t_k+\Delta t)$를 적분하면
$$
I_k(\lambda)=\int_{t_k}^{t_k+\Delta t}\Big[A(t)\,G\big(\lambda;\mu(t),\sigma(t)\big)+b(t)\Big]dt+\int c(\lambda,t)\,dt,
\qquad
c(\lambda,t)=C(\lambda)\,\Delta t\;\delta(t-t_{k^*})
$$
(연속시간의 임펄스; 프레임 지표로는 $c_k(\lambda)=\delta_{kk^*}C(\lambda)$). $\mu,\sigma,A,b$가 프레임 안에서 느리게 변하면 $I_k=\Delta t\,[\bar A\,G(\lambda;\bar\mu,\bar\sigma)+\bar b]+\delta_{kk^*}C(\lambda)$. 최소제곱 추정의 정규방정식
$$
\sum_\lambda\big[I_k(\lambda)-\hat A\,G-\hat b\big]\,\frac{\partial G}{\partial\theta}=0,\quad
\frac{\partial G}{\partial\mu}\propto(\lambda-\mu)G\ (\text{홀}),\quad
\frac{\partial G}{\partial\sigma}\propto(\lambda-\mu)^2G\ (\text{짝})
$$
에서 $C(2\mu-\lambda)=C(\lambda)$이면 $\mu$ 방정식의 오염 기여는 0이고 $\sigma$ 방정식의 기여는 양수이므로
$$
\hat T_k-T_k=\begin{cases}\mathcal O(\sigma_T^{\rm CRB})&k\ne k^*\\ \gg0\ (\times10\text{–}100,\ \text{MC 표})&k=k^*\end{cases},\qquad
\hat V_k-V_k=\mathcal O(\sigma_V^{\rm CRB})\ \ \forall k .
$$
잔차가 $k=k^*$ 한 프레임에만 서고 $V$ 채널에는 서지 않는 것이 "단일 프레임·폭 전용" 지문이며, 2항의 플라즈마 가지와 달리 에너지·압력을 요구하지 않는다.

---

#### 2. 데이터 검사 (641파일, `p05_pressure_limit.py` D 블록)

**(i) 런 통계 재현 — §8ab 메모와 일치.** 관측 `CES_TI` 226,991행, 블록은 관측행 시간순에서 ≥ 0.5 s 간격으로 분할.

| 항목 | 값 | §8ab |
|---|---:|---|
| > 3 keV 행 / 런 / shot | 1,197 / 951 / 274 | 1,197 / 951 / 274 |
| 단일행 런 | 813 (85.5 %) | 813 (85 %) |
| 2–4행 / ≥ 5행 (최대) | 121 / 17 (15) | 121 / 17 (15) |
| 고립 런(양 이웃 < 2 keV) | 70.5 % | 70 % |
| 런 최대 / 이웃 평균 (중앙값, IQR) | **13.2** (6.0–26.4) | 13 (6–26) |
| 스파이크 값 / 이웃 평균 중앙값 | 5,305 eV / 835 eV | — |
| 스파이크 − 이웃 중앙값 / p90 / 최대 | 5,019 / 11,720 / 14,630 eV | — |
| ≥ 5행 지속 런의 행 수 / 값 중앙값 | 115 / 5,406 eV | — |

지속 런(115행, 5.4 keV가 ≥ 50 ms)은 시간 척도 논증(1b)의 바깥이지만 압력 논증(1a)의 안이다: 5.4 keV를 50 ms 유지하는 pedestal은 PB의 3배를 $1/\gamma$의 10⁴배 동안 버텨야 하므로 역시 플라즈마가 아니다(빔 노치 블록 동안의 배경 불일치 같은 **지속형 적합 실패**).

**(ii) 폭 전용 예측 검정 — 지지.** 스파이크 행에서 `CES_VT`는 **더 자주** 관측된다(80.5 % vs 대조 76.6 %; held 제거 후 31.9 % vs 35.3 %) — 기각이 $V_{\rm rot}$로 번지지 않는다(§1.6의 독립 기각). 스파이크 행의 $|V_{\rm rot}-\text{이웃 평균}|$(held 제거, 양 이웃 관측, $n=371$ vs 78,645):

| | 스파이크 행 | 비스파이크 행 | 검정 |
|---|---:|---:|---|
| $|\Delta V|$ 중앙값 [km/s] | 3.55 | 3.38 | Mann–Whitney 양측 $p=0.75$ |
| $|\Delta V|$ p90 | 29.9 | 23.5 | |
| $|\Delta V|>100$ / $>300$ km/s | 1.08 % / 0.54 % | 1.13 % / 0.39 % | |
| 같은 행 상대 점프 중앙값 | $T_i$: **5.8** · $V$: 0.052 | $V$: 0.052 | |
| $V$가 수준의 ≥ 2배 점프 | 0.25 % | 0.26 % | |

(held 포함 원시 비교는 중앙값이 양쪽 0이고 $p=2\times10^{-4}$지만 p90이 11.1 vs 11.7로 **스파이크 쪽이 작다** — held 반복의 비율 차이지 $V$ 스파이크가 아니다.) 같은 프레임에서 $T_i$는 5.8배 뛰고 $V_{\rm rot}$는 0.05배(측정 바닥 4.3 km/s 수준) 움직인다: 1(c)의 대칭 오염 지문 그대로이며, 핫 픽셀·저신호·두 성분(수백 km/s 이동)은 배제된다.

**(iii) 빠른 진단의 활동 — 플라즈마 사건의 꼬리에 몰리지 않는다.** 인접 10 ms 격자행의 $|\Delta|$ 채널 평균:

| 채널 | 스파이크 행 중앙값 / 전체 중앙값 | 비 | $p$ | 스파이크 행 중 전체 p99 초과 |
|---|---|---:|---|---:|
| $|\Delta$ECEI$|$ | 0.0529 / 0.0366 | 1.44 | $4\times10^{-8}$ | 1.4 % (무작위 기대 1 %) |
| $|\Delta$ECEI$|$ 양측 max | 0.0709 / 0.0516 | 1.37 | $9\times10^{-7}$ | 1.8 % |
| $|\Delta$BES$|$ | 0.0098 / 0.0067 | 1.47 | $4\times10^{-56}$ | 7.0 % |
| $|\Delta$MC$|$ | 3.89 / 2.34 | 1.66 | $5\times10^{-36}$ | 3.9 % |

$|\Delta$ECEI$|$ 구간별 스파이크 발생률: 하위 10 % **0.70 %**, 중간 0.48 %, 상위 10–1 % 0.75 %, 상위 1 % 0.75 % — 평평(U자)하다. 하향 dip 행($\le\tfrac12\times$이웃)의 $|\Delta$ECEI$|$ 비도 **1.71**로 스파이크와 같다. 읽는 법: 스파이크 행의 플라즈마는 **약간**(×1.4–1.7) 더 활동적이지만, 13배의 $T_i$ 사건이라면 $\tau_{\rm eq}\approx\Delta t$로 $T_e$가 같은 프레임에 따라와 ECEI가 꼬리(p99)에 있어야 하는데 1.4 %뿐이다. 상향 스파이크와 하향 dip이 **같은 정도로** 활동과 동행한다는 것은 활동(밀도 펌프아웃·ELM 직후의 낮은 S/N, §1.3)이 적합 실패 확률을 **양방향으로** 올린다는 뜻이지, 스파이크 자체가 사건이라는 뜻이 아니다 — BES 꼬리 7 %(빔 광자 감소 = S/N 하락)가 그 경로를 가리킨다. 이것이 MNAR의 물리적 기원(§1.3, §12)과 같은 기전이다.

**(iv) 함의 파워.** 표 1(b): 중앙값 스파이크에 5.6 MW(이온·pedestal 한정, 설치 상한의 0.56배, pedestal 몫의 39배), 하강 싱크 46배, 첨두 압력 PB의 3.06배 = Troyon 체적 평균.

---

#### 3. 기록과의 연결

- **§8ab 두 모집단 규칙.** 컷 모집단(3 keV)은 "$> 3$ keV는 KSTAR 이온 온도가 아니다"를 사전등록 대리로 썼다. 본 절은 그 문장을 물리로 닫는다: 상한 1.0–1.8 keV(1a) 위의 값은 압력·시간·MHD 세 축에서 배제되므로 컷은 분포 꼬리 잘라내기가 아니라 **측정 가능 영역의 경계**다. 포함 모집단(p100)이 공동 1차로 남는 이유도 같은 곳에서 나온다: 컷은 상향 이상치의 19 %만 제거하고(§8ab) dip은 하나도 제거하지 않으므로, 값 컷은 적합 품질 메타데이터가 올 때까지의 **부분적** 대리다.
- **§8q "헤드라인은 보수적".** 스파이크 행은 어느 arm도 맞힐 수 없는(어떤 플라즈마 상태에도 대응하지 않는) 측정 잡음이므로 §10.2의 희석 항등식대로 skill을 깎는다(컷 후 +0.18…+0.28 → +0.36…+0.59, 4/4 PASS). 본 절은 "맞힐 수 없다"의 근거를 보탠다: 그 행의 값은 빠른 진단(ECEI·BES)이 담을 수 있는 어떤 플라즈마 상태의 함수도 아니다(2(iii)).
- **문턱 2.5–4 keV 불변(§8ab §7, 평균 ±0.006, PR4 전부 동일).** 물리 상한이 **2.5 keV 아래**(1.0–1.8 keV)이므로 2.5·3·4 keV는 모두 측정 가능 영역 밖에 놓인 문턱이고, 서로 다른 것은 새어 들어오는 적합 실패의 양뿐이다: 2,500–3,000 eV 430행, 3,000–4,000 eV 348행, > 4,000 eV 849행(> 2,500 합계 1,627) — 관측 226,991행의 0.19 %·0.15 %. 문턱을 2.5↔4 keV로 옮겨도 모집단이 0.3 % 안에서만 바뀌므로 수치가 안 움직이는 것이 당연하고, 관측 p99(2,089 eV)가 상한 바로 위에 놓인 것도 같은 그림이다.

---

#### 4. 한계

1. **플라즈마 파라미터는 가정값**(부록 A)이다. $n_e$, $T_e$, $P$, $q_{95}$, 형상이 CSV에 없으므로 1(a)·1(b)는 KSTAR 2022 H-mode 전형에 대한 자릿수 경계다. 상한은 $\alpha_c$(2–6)와 자속 팽창 $g$(½–1)에 각각 제곱·제곱으로 민감해 $p_{\rm ped,max}$가 0.3–9.6 kPa를 오간다; 본문은 가장 관대한 쪽(4.3–9.6 kPa)을 썼고, 그래도 3 keV는 1.7–5배 밖이다. CES 채널의 반경 위치도 데이터에 없다 — 채널이 pedestal 상단 안쪽이면 국소 $T_i$는 pedestal 상단보다 높을 수 있으나(경직 프로파일 $L_T\approx3$ cm), 그 경우에도 1(b)의 단일 프레임 13배 상승·하강은 위치와 무관하게 배제된다.
2. **파워 논증은 "자릿수 초과"가 아니라 "설치 총량과 동급 + 국소·순간·가역 불가능"이다.** 설치 상한 10 MW는 가정이며, 중앙값 스파이크의 5.6 MW(이온·pedestal 한정)는 그 0.56배다. 배제력은 pedestal 몫(39–112배)·$T_e$ 동반(2배)·하강 싱크(46배)·MHD 성장(1/2,800)에서 온다.
3. **적합 MC는 비가중 4모수 LSQ와 $\sigma$ 하한 0.3 px**를 가정했다. 실제 KSTAR CES 파이프라인(가중치, 다성분 모형, QC 규칙)은 비공개이므로 1(c)는 "지문 제공"이지 파이프라인 재현이 아니다. 핫 픽셀 조건의 높은 수렴 실패율(17–86 %)은 실제 QC에서 결측으로 나타날 수 있다. 2(ii)가 지지하는 것은 "대칭 오염 부류"이지 특정 하드웨어 원인이 아니다.
4. **2(iii)의 활동 비 1.4–1.7은 유의**($p<10^{-6}$)하므로 "스파이크 행과 플라즈마 활동이 무관하다"고 말할 수 없다. 본 절의 주장은 그보다 약하고 정확하다: 활동은 적합 실패 확률을 양방향으로 올리고(dip도 1.71), 사건 자체(ECEI 꼬리)는 스파이크 행에 몰리지 않는다.
5. 결정적 검정은 **CES 적합 품질 메타데이터(χ², 신호 수준, 배경 수준)**다(§8ab·§8v). 그것이 오면 값 컷은 품질 컷으로 대체되고, 본 절의 예측 — 스파이크 행은 배경/연속광 수준이 높고 χ²가 나쁘며 $V$ 불확실도는 정상 — 은 직접 검증된다. 그 전까지 본 절은 압력·시간·측정 세 독립 축의 합치다.


### P06 빠른 진단이 더하는 정보는 혁신과의 상호정보로 유계이고, 측정된 +0.05는 그 한계 안에 있다

스크립트: `scratchpad/p06\p06_mi_bound.py` (출력 `p06_output.txt`, `p06_results.json`) 및 시간 영역 보조 계산 `p06_timedomain.py` (출력 `p06_timedomain_output.txt`), 모두 같은 폴더. 모델 학습·채점 없음, 추적 파일 수정 없음. 인용 파일: `docs/theory_foundations.md` §3.2·§3.3·§7.4·§7.8·§8, `data/.foundations.json` D·E, `THESIS_RESULTS.md` §8ar·§8ab 8절.

#### 1. 유도

**(1) 가우시안 MMSE와 상호정보.** $(y,h,z)$가 결합 가우시안이면 조건부 분산이 곧 MMSE이고, 조건부 상호정보는 부분상관으로 닫힌다:
$$
I(y;z\mid h)=-\tfrac12\ln\!\bigl(1-r_{yz\cdot h}^2\bigr),\qquad
\mathrm{MSE}(y\mid h,z)=\mathrm{MSE}(y\mid h)\,\bigl(1-r_{yz\cdot h}^2\bigr)=\mathrm{MSE}(y\mid h)\,e^{-2I(y;z\mid h)} .
$$
따라서 최선의 이력 전용 예측기 위에 $z$를 더해 얻는 최대 이득은 이력 MSE 단위로 $1-e^{-2I}=r_{yz\cdot h}^2$. 비가우시안·비선형이면 $I\ge-\tfrac12\ln(1-R^2_{\rm lin})$이므로 선형 $R^2$는 상호정보 한계의 **하한**이고, 상호정보 자체가 모든 예측기(비선형 포함)의 한계다(데이터 처리 부등식).

**(2) persistence 기준 skill 단위로 변환.** $\mathrm{skill}_{h}=1-\mathrm{MSE}_h/\mathrm{MSE}_{\rm pers}$라 두면
$$
\Delta\mathrm{skill}=\mathrm{skill}_{hz}-\mathrm{skill}_h
= r_{yz\cdot h}^2\,\frac{\mathrm{MSE}_h}{\mathrm{MSE}_{\rm pers}}
= r_{yz\cdot h}^2\,(1-\mathrm{skill}_h).
$$
변환 계수 $\mathrm{MSE}_h/\mathrm{MSE}_{\rm pers}$: OU 강건 적합 $3169.8/4136.6=0.766$, 고전 $29816.9/44598.3=0.669$ (`.foundations.json` D), 측정 인과 GP $1-0.310=0.690$ (E.ladder_1step_CES_TI). 측정 이득 $+0.358-0.310=+0.048$은 $r^2_{yz\cdot h}=0.048/0.690=0.069$, $r=0.263$, $I=0.036$ nat $=0.052$ bit를 함의한다.

**(3) 올바른 조건부 변수는 필터 혁신이며, 이것은 level도 rate도 아니다.** 잡음 OU(§7.1)의 인과 필터 혁신은
$$
\nu_t=\tilde y_t-E[\tilde y_t\mid h]=\underbrace{\varepsilon_t}_{\text{과정 잡음}}+\underbrace{\rho\,(y_{t-1}-\hat y_{t-1\mid t-1})}_{\text{사전 상태 오차}}+\underbrace{n_t}_{\text{측정 잡음}},
$$
분산은 $\sigma_s^2(1-\rho^2)+\rho^2P+\sigma_n^2=P^-+\sigma_n^2=\mathrm{MSE}_{\rm filter}$(§7.4). level $y_{t-1}$은 $h$ 안에 있으므로 level 상관 $+0.34$(§8ar)는 그대로 쓸 수 없다. 그러나 **rate 상관 $+0.07$도 틀린 양이다**: 1-step 증분 $\Delta\tilde y=\varepsilon_t-(1-\rho)(y_{t-1}-\bar m)+n_t-n_{t-1}$의 분산은 $2\sigma_n^2$가 지배하고(고전 $\sigma_n=139.9$ eV), $z$가 혁신에 닿는 둘째 경로, 즉 **잡음 앵커의 사전 상태 오차 $\rho(y_{t-1}-\hat y_{t-1})$를 level 정보로 탈잡음하는 경로**를 담지 못한다. $z$가 $T_i$ level과 $r=0.34$로 상관하면 $\tilde y_{t-1}$이 주어졌을 때 $z$는 $n_{t-1}$에 관한 정보를 준다. 어느 경로가 큰지는 데이터가 정한다(2절).

**(4) 구조적 천장.** $n_t$는 어떤 외생 변수로도 예측되지 않으므로 $r_{\nu z\cdot h}^2\le P^-/(P^-+\sigma_n^2)=K$(Kalman 이득). 따라서
$$
\Delta\mathrm{skill}\le K\cdot\frac{\mathrm{MSE}_h}{\mathrm{MSE}_{\rm pers}}=
\begin{cases}0.520\times0.766=0.398 & \text{강건}\\ 0.343\times0.669=0.230 & \text{고전}\end{cases}
$$
측정 $+0.048$은 두 천장 모두 안이다. 어떤 정보원도 persistence 대비 skill을 $0.31+0.23\approx0.54$(고전) 너머로 못 올린다는 뜻이기도 하다.

**(5) 다채널 선형 한계.** $h$를 조건으로 한 혁신을 15개 빠른 채널(과 그 1-step 증분, 또는 50 ms 지연 묶음)에 회귀한 부분 $R^2$가 선형 한계: $\Delta\mathrm{skill}_{\rm lin}=R^2_{\nu\cdot z\mid h}\,\mathrm{MSE}_h/\mathrm{MSE}_{\rm pers}$. 단일 채널 $r^2$의 합이 아니라 결합 $R^2$다(채널 간 상관이 커서 합은 과대).

#### 시간 t의 식

이 증명의 모든 양을 시간 $t$의 식으로 모은다. 기호: $y(t)=T_i(t)$, $z(t)$ = 빠른 진단이 재는 구동자($T_e$ 대용: ECEI는 $T_e$ 자체, BES는 $n_e$를 통해 §2·§3.4), $\xi(t)$ = 백색 과정 잡음, $n(t)$ = CES 측정 잡음, $\Delta=10$ ms = 표본 간격, $\tau_{\rm eq}$ = 전자–이온 평형화 시간(§3.2: 8–59 ms).

**구동 방정식과 그 해.** §3.3의 선형화된 이온 에너지 방정식(밀도 고정, $\tau_{E,i}$ 항은 $\tau_{\rm eq}$에 흡수)
$$
\frac{dy}{dt}=-\frac{y(t)-z(t)}{\tau_{\rm eq}}+\xi(t)
\quad\Longrightarrow\quad
y(t)=\int_0^\infty h(s)\,z(t-s)\,ds+\int_0^\infty h(s)\,\tau_{\rm eq}\,\xi(t-s)\,ds,
\qquad h(s)=\frac{1}{\tau_{\rm eq}}e^{-s/\tau_{\rm eq}},\ \int_0^\infty h=1 .
$$
$T_i$는 $z$의 저역통과(차단 $1/\tau_{\rm eq}$)이고, 관측은 $\tilde y(t)=y(t)+n(t)$다. $\tau_{E,i}$를 분리하면 $h(s)=\frac{1}{\tau_{\rm eq}}e^{-s/\tau_*}$, $1/\tau_*=1/\tau_{\rm eq}+1/\tau_{E,i}$로 이득이 $\tau_*/\tau_{\rm eq}<1$이 될 뿐 형태는 같다.

**한 걸음의 식.** 두 표본 사이 $[t-\Delta,t]$에서 적분하면 정확히
$$
y(t)=\rho\,y(t-\Delta)+\underbrace{\int_0^{\Delta}h(s)\,z(t-s)\,ds}_{\text{구동 항}}+\underbrace{\int_0^{\Delta}e^{-s/\tau_{\rm eq}}\,\xi(t-s)\,ds}_{\varepsilon(t)},
\qquad \rho=e^{-\Delta/\tau_{\rm eq}},\quad \mathrm{Var}\,\varepsilon=\sigma_s^2(1-\rho^2).
$$
$z$가 한 걸음 안에서 매끈하면 구동 항 $\approx(1-\rho)\,\bar z(t)$, $\bar z$ = 그 걸음의 $h$-가중 평균. $\int_0^\Delta h=1-\rho$이므로 **한 걸음에 구동자가 $T_i$에 새기는 양은 $(1-\rho)\approx\Delta/\tau_{\rm eq}$ 배**다.

**혁신.** 정확한 과거를 조건으로 한 혁신과, 잡음 낀 과거를 조건으로 한 혁신:
$$
e(t)=y(t)-E\bigl[y(t)\mid y(s),\,s<t\bigr]=(1-\rho)\bigl[\bar z(t)-E[\bar z(t)\mid y(s),s<t]\bigr]+\varepsilon(t),
$$
$$
\nu(t)=\tilde y(t)-E\bigl[\tilde y(t)\mid \tilde y(s),\,s<t\bigr]=e(t)+\rho\bigl[y(t-\Delta)-\hat y(t-\Delta)\bigr]+n(t),
\qquad \mathrm{Var}\,\nu=\sigma_s^2(1-\rho^2)+\rho^2P+\sigma_n^2 .
$$
2절의 대리 $e_t=\tilde y_t-[m+\rho(\tilde y_{t-1}-m)]$는 $\nu(t)$를 1관측 이력으로 근사한 것이고, (b″)의 4관측 이력 회귀가 $E[\cdot\mid\tilde y(s),s<t]$를 50 ms 범위에서 선형으로 대신한다. $z(t)$는 $\nu(t)$에 두 경로로 닿는다: 구동 항(경로 A, "rate")과 사전 상태 오차 $\rho[y-\hat y]$(경로 B, "탈잡음"; $z$가 level과 상관하므로 $\tilde y(t-\Delta)$가 주어진 뒤에도 $n(t-\Delta)$에 관한 정보를 준다).

**표본 간격 $\Delta$ 대 $\tau_{\rm eq}$ — 한 걸음당 정보.** 경로 A에서, 구동자의 걸음당 혁신 분산을 $\mathrm{Var}(\bar z\mid h)=S_z\,\sigma_s^2$라 두면 정확한 과거 조건부 부분상관과 걸음당 상호정보는
$$
r^2_{e\bar z\cdot h}(\Delta)=\frac{(1-\rho)^2S_z}{(1-\rho)^2S_z+(1-\rho^2)}=\frac{\kappa S_z}{1+\kappa S_z},\qquad
\kappa(\Delta)=\frac{1-\rho}{1+\rho}=\tanh\!\frac{\Delta}{2\tau_{\rm eq}},\qquad
I_{\rm step}(\Delta)=\tfrac12\ln\bigl(1+\kappa S_z\bigr).
$$
두 극한:
- $\Delta\to0$: $\kappa\to\Delta/(2\tau_{\rm eq})$, $I_{\rm step}\to S_z\Delta/(4\tau_{\rm eq})\to0$. **걸음당 정보는 $\Delta$에 비례해 사라지고**, 정보율 $I_{\rm step}/\Delta\to S_z/(4\tau_{\rm eq})$ nat/s로 수렴한다 — 구동자의 정보는 시간당 일정하고, 표본을 촘촘히 할수록 한 걸음은 그 작은 몫만 받는다.
- $\Delta\gg\tau_{\rm eq}$: $\rho\to0$, $\kappa\to1$, $I_{\rm step}\to\tfrac12\ln(1+S_z)$. 걸음마다 상태가 통째로 새로워져 혁신 = level이 되고, 한계는 **level 상관**의 것으로 돌아간다. §8ar의 "level 상관 > rate 상관"은 이 두 극한 사이, $\Delta=10$ ms $<\tau_{\rm eq}$ 쪽에 있다는 진술이다.

$\Delta=10$ ms에서 $\kappa$와, 측정된 이력 조건부 부분 $R^2=0.066$을 경로 A만으로 내려면 필요한 구동자 걸음당 혁신 sd($\sqrt{R^2\,\mathrm{Var}\,\nu}/(1-\rho)$, 고전 적합 $\mathrm{Var}\,\nu=29{,}817$ eV², `p06_timedomain.py`):

| $\tau_{\rm eq}$ [ms] | $\rho$ | $\kappa=(1-\rho)/(1+\rho)$ | $\Delta/2\tau_{\rm eq}$ | $I_{\rm step}$ at $S_z=1$ [nat] | 필요 sd$(\bar z\mid h)$ [eV] (고전 / 강건) |
|---:|---:|---:|---:|---:|---:|
| 8 | 0.287 | 0.555 | 0.625 | 0.221 | 62 / 20 |
| 20 | 0.607 | 0.245 | 0.250 | 0.110 | 113 / 37 |
| 30 | 0.717 | 0.165 | 0.167 | 0.076 | 157 / 51 |
| 59 | 0.844 | 0.085 | 0.085 | 0.041 | 285 / 93 |
| 81.2 (OU 자기상관 $\tau$) | 0.884 | 0.062 | 0.062 | 0.030 | 384 / 125 |

읽기. $T_i$ 자체의 정상 sd는 고전 147 eV / 강건 69 eV다. $\tau_{\rm eq}$를 OU 자기상관 시간 81 ms로 두면 경로 A만으로 측정치를 내려면 구동자가 걸음마다 $T_i$ 정상 sd의 2.6배(384 eV)를 새로 움직여야 해 비현실적이고, §3.2의 8–59 ms면 62–285 eV로 $T_i$ 자신의 sd 안팎이다. 즉 **걸음당 측정된 정보량(0.034 nat)은 $\tau_{\rm eq}\lesssim30$–60 ms일 때 구동 경로와 양립**하며, 이것은 §3.2의 $\tau_{\rm eq}$ 범위와 §8al의 50 ms 포화에 세 번째 독립 척도를 나란히 세운다. 한 걸음당 정보가 $\Delta/\tau_{\rm eq}$에 비례하므로 **10 ms 격자는 구동자의 정보율을 걸음당 6–17 %씩 나눠 받는 영역**이고, 이것이 한 걸음 skill 이득이 +0.05 수준으로 작은 시간 영역의 이유다.

**정상 가우시안 두 경로 회계(§8ar의 풀링 상관을 입력으로).** 경로 B를 $z=a\,y+w$로 두고 level 상관 $r_L=0.341$을 쓰면 사전 오차에 대한 $r^2(\rho[y-\hat y],z\mid h)=P/(P+(1/r_L^2-1)\sigma_s^2)$, 혁신 부분 $R^2$ 기여는 정상 필터 0.007 / 1관측 이력 0.014(고전). 경로 A를 §8ar의 rate 상관 0.070–0.078로 재면 증분 분산 44,598 eV² 중 219–271 eV² = 혁신의 0.007–0.009. 두 경로 합 0.015–0.023은 측정 0.066의 1/3이다. 결론: **§8ar의 블록 전체 풀링 상관($r_L$, $r_{\rm rate}$)은 이 한계의 충분통계가 아니다** — 결합은 정상적이지 않고 천이에 몰려 있어(2절 (c): 이득의 81–83 %가 상위 10 % $\lvert e\rvert$ 행) 조용한 블록의 평균 상관으로는 1/3만 설명된다. (b″)의 이력 조건부 회귀는 이 비정상성을 우회해 부분 $R^2$를 직접 잰다. 50 ms 빠른 맥락이 $R^2$를 0.066→0.079로 올리는 것은 경로 B의 증거다: 정확한 과거 조건부 혁신 $e(t)$에는 $\int_0^\Delta$ 안의 $z$만 들어오지만, 잡음 낀 과거 조건부 혁신 $\nu(t)$의 사전 상태 오차는 더 오래된 $z$로도 줄어든다.

#### 2. 데이터 검산 (641파일, held-free, 3 keV 컷, 0.5 s 간극 블록, 연속 관측쌍만)

혁신 대리 $e_t=T_i(t)-[m+\rho(T_i(t-1)-m)]$, $\rho=e^{-10/81.19}=0.884$, $m$ = 블록 평균, 블록 내 중심화, 빠른 채널은 shot별 z-점수(§8ar·`CES_PER_SHOT_NORM=1`과 동일). 풀링 214,407행 / 618블록 / 595 shot. $\mathrm{Var}(e)=41{,}449$ eV² — 고전 OU 적합의 대리 분산 예측 $\sigma_s^2(1-\rho^2)+\sigma_n^2(1+\rho^2)=39{,}644$와 5 % 안. 2-fold는 정렬된 파일 index 홀/짝 shot 분할.

| 측정량 | 값 | $\Delta$skill vs pers (×0.690) |
|---|---:|---:|
| (a) $r(e,\,$BES 평균$)$ / ECEI 평균 / MC 평균 (level) | +0.112 / +0.119 / +0.001 | 0.009 / 0.010 / 0.000 |
| (a) 같은 것, 1-step 증분 | +0.139 / +0.139 / +0.001 | 0.013 / 0.013 / 0.000 |
| (a) 단일 raw 채널 $r$ 범위 (BES·ECEI / MC) | 0.099–0.112 / ≤0.004 | ≤0.009 |
| (b) 조건 없는 $R^2$: 15채널 / 15+증분 30 / 50 ms 지연 75 (OOS) | 0.0216 / 0.0424 / 0.0426 | 0.015 / 0.029 / 0.029 |
| (b′) **이전 level 1개 조건부** partial $R^2$: 3모달 / 15 / 30 / 75 (OOS) | 0.081 / 0.082 / 0.085 / 0.086 | 0.056 / 0.057 / 0.059 / 0.059 |
| (b″) **50 ms 이력(지연 1,2,3,5) 조건부**, 경험적 변환 (204,175행, pers MSE 35,506 eV²) | | |
| · 이력만 선형 (null) skill vs pers | 0.297 (OOS) | — |
| · + 15채널 (t) | partial $R^2$ 0.061 | **+0.043** |
| · + 15채널 + 증분 (t, t−1) | partial $R^2$ 0.066 ($I=0.034$ nat) | **+0.047** |
| · + 15채널 × 지연 0,1,2,3,5 (50 ms) | partial $R^2$ 0.079 ($I=0.041$ nat) | **+0.056** |
| (c) 상위 10 % $\lvert e\rvert$ (≥232 eV): SSE(e) 점유 / null 잔차 SSE 점유 | 0.869 / 0.734 | — |
| (c) 50 ms 이력 조건부 이득의 분해 (t, t−1 빠른 채널) | 꼬리 +0.039 + 본류 +0.008 | 꼬리 83 % |
| (c) 같은 분해, 50 ms 빠른 맥락 | 꼬리 +0.046 + 본류 +0.011 | 꼬리 81 % |
| (c) 꼬리 내부 / 본류 내부 partial $R^2$ (t, t−1) | 0.076 / 0.044 | — |
| **측정** (TEST s42 컷, Δt ≤ 15 ms): 모델 0.358 − 인과 GP 0.310 | **+0.048** → $r^2_{\rm partial}=0.069$, $I=0.036$ nat | — |
| §8ar rate 상관 단독 한계 (BES 0.070 / ECEI 0.078) | $r^2$ 0.0049 / 0.0061 | 0.003 / 0.004 |
| 구조적 천장 $K\cdot\mathrm{MSE}_h/\mathrm{MSE}_{\rm pers}$ (강건 / 고전) | — | 0.398 / 0.230 |

읽기.
- **측정 +0.048은 선형 한계 안에 있다.** 이력을 제대로 조건으로 두면(50 ms 이력) 빠른 진단 두 행(t, t−1)의 선형 한계가 +0.047, 50 ms 빠른 맥락까지 주면 +0.056. 측정 +0.048은 그 사이, 즉 **백본은 50 ms 이력 + 빠른 진단의 선형 판독기가 뽑을 수 있는 만큼을 뽑고 그 이상을 뽑지 못한다.** 함의 상호정보 0.036 nat 대 선형 0.034–0.041 nat. 비선형 초과분은 관측되지 않는다.
- **조건부가 결과를 결정한다.** 같은 30 regressor가 조건 없이는 $R^2$ 0.042(→0.029), 이전 level 하나를 조건으로 두면 0.085(→0.059), 50 ms 이력 조건부면 0.066(→0.047). 조건부가 $R^2$를 **올리는** 것은 억제변수 효과다: $z$는 level 정보로 잡음 앵커 $\tilde y_{t-1}$의 $n_{t-1}$을 식별한다(경로 B). 50 ms 이력이 그 탈잡음의 일부를 대신하므로 (b′)→(b″)에서 다시 줄어든다. 선형 이력 사다리 자체도 §8과 맞는다: 이력만 선형 0.297 < 인과 GP 0.310 < OU 고전 최적 필터 0.331.
- **§8ar의 rate 상관(+0.07)은 이 한계의 올바른 입력이 아니다.** 그것이 주는 한계 +0.003–0.004는 측정의 1/12. rate 상관은 $2\sigma_n^2$에 희석된 증분과의 상관이고, 혁신 대리와의 단일 모달 상관은 이미 0.11–0.14, 이력 조건부 결합 partial $r$은 $\sqrt{0.066}=0.26$ — 측정이 함의하는 0.263과 일치한다. §8ar의 결론("$V_{\rm rot}$는 rate 경로가 0")은 유지되나, $T_i$ 쪽 "+0.07이 결합의 크기"라는 읽기는 "+0.26(이력 조건부)"로 고쳐 써야 한다.
- **MC는 어느 조건부에서도 0이다** (r ≤ 0.007, 증분 포함). §5.4·§8b.2의 데시메이션 음성과 일치.
- **이득은 꼬리의 것이다.** 상위 10 % $\lvert e\rvert$ 행이 선형 이득의 81–83 %를 운반(꼬리 내부 $R^2$ 0.076 대 본류 0.044, 거기에 SSE 점유 0.73이 곱해짐). §8ax("헤드라인 skill은 꼬리 효과")·§8an("타깃이 움직이는 정도"가 승패 변수)의 선형 판본.
- **절제 결손 −0.25…−0.43(§8ab 8절)이 선형 한계를 넘는 이유.** (i) 단위: 그 수치는 PCHIP 대비 skill이고 컷 모집단 윈도 모델의 짝 비교다; 여기 한계는 persistence 대비 최적 예측기 둘의 차. (ii) 상호정보 한계는 "최적 이력 전용 예측기"와 "최적 결합 예측기"의 차를 제한하지, **학습된 모델의 입력을 0으로 둔 결과**를 제한하지 않는다. 0은 $E[z\mid h]$가 아니라 훈련 분포 밖의 점이고(주변화가 아니라 개입), 이력 가지의 가중치는 빠른 가지가 있는 채로 공적응되어 이력 전용 최적점이 아니다. 절제 결손 ≥ 정보 이득은 성립하되 상한은 없다. (iii) 비선형성. 포함 모집단의 결손 −0.03…−0.09가 오히려 여기 선형 한계와 같은 자릿수다.

#### 3. 기록과의 연결
- §7.8의 "빠른 진단은 1-step 집계 MSE에 ≈ +0.05를 더한다"에 **한계와 부분상관**이 붙는다: 선형 한계 +0.047–0.056, 함의 $r_{\rm partial}=0.26$, $I\approx0.04$ nat. 구조적 천장(고전 0.23)과 선형 한계 사이의 0.17은 어떤 선형 판독기도 못 닿는 영역이고, 백본도 거기 없다.
- §8 사다리: 이력만 선형 0.297 / GP 0.310 / OU 최적 0.331 / 이력+빠른 선형 0.344–0.353 / **백본 0.358**. 백본의 위치는 "선형 이력 + 선형 빠른 진단"의 끝과 같다. §8aa(크기 스케일링 평탄)·§8ag(연산자 군 무차별)의 정보론적 이유: 뽑을 선형 정보가 다 뽑혔다.
- §3.2·§8al·§8af(50 ms): 시간 영역 식은 걸음당 정보 $\propto\tanh(\Delta/2\tau_{\rm eq})$를 주고, 측정된 걸음당 0.034 nat은 $\tau_{\rm eq}\lesssim30$–60 ms에서만 구동 경로와 양립한다 — 50 ms에 세 번째 척도. 빠른 맥락을 10 ms→50 ms로 늘리면 선형 한계가 +0.047→+0.056(≈+0.01), 도달 사다리의 20 ms 결손 −0.065(§8af)와 같은 자릿수.
- §8ar: $T_i$의 결합 크기는 rate 상관 0.07이 아니라 이력 조건부 partial 0.26으로 인용해야 한다. 정상 두 경로 회계가 측정의 1/3만 내는 것은 결합이 천이에 몰린 비정상 결합이라는 뜻이다. $V_{\rm rot}$ 음성은 같은 방법으로 다시 재야 확정된다(한계 4).
- §8ab 8절: "컷 모집단의 $T_i$ 여유는 빠른 진단 정보로 되어 있다"는 서술은 유지; 그 **크기**는 절제 결손이 아니라 여기 한계로 인용해야 한다.

#### 4. 한계
- 풀링 모집단(641파일·595 shot)과 사다리 측정 모집단(TEST s42 컷 96 shot·Δt ≤ 15 ms)이 다르다. 분산 수준은 맞지만(41,449 대 OU 39,644 eV²; 같은 행 persistence RMSE 188 eV 대 TEST 184 eV) 같은 행에서 짝지은 비교는 아니다.
- 선형 한계는 상호정보 한계의 하한이다. 측정이 선형 한계와 같다는 것은 "비선형 초과분 없음"이지 "상호정보가 0.04 nat"이라는 증명은 아니다. 상호정보 자체의 비모수 추정은 하지 않았다.
- 시간 영역 식의 $S_z$(구동자 걸음당 혁신 분산)와 $\tau_{\rm eq}$는 본 데이터가 물리 단위로 갖지 않는다($T_e$, $n_e$ 없음, §3.2 주의). 표의 "필요 sd"는 측정 $R^2$를 역산한 요구 조건이지 측정값이 아니다.
- 혁신 대리는 블록 평균 $m$(블록 전체, 즉 미래 포함)으로 중심화된다; (b″)는 계수를 다시 적합하므로 $m$은 중심화에만 쓰이나 완전한 인과 필터 잔차는 아니다. 이력 전용 선형 0.297이 OU 최적 0.331보다 낮은 것은 지연 1,2,3,5 네 점만 쓴 탓이다.
- 2-fold 분할은 파일 index 홀/짝이지 캠페인(시간) 분할이 아니다. 다만 $p\le75$, $n\approx2\times10^5$라 in-sample과 OOS가 0.002 안에서 같다.
- 꼬리 분해는 종속변수 $\lvert e\rvert$로 조건화한 것이라 꼬리 내부 $R^2$는 선택 조건부 통계다; 이득의 분해 자체(한 적합의 잔차 SSE 분할)는 선택 편향이 없다.
- 인과 GP는 최적 필터가 아니므로(0.310 대 0.331) 측정 +0.048은 비최적 두 팔의 차다. 이력 전용 참조를 OU 최적 필터로 바꾸면 빠른 진단 귀속분은 +0.027까지 내려가고, 이것도 선형 한계 안이다.
- 이 결과를 뒤집을 측정: (i) TEST s42 컷의 같은 행에서 인과 GP 잔차를 빠른 채널에 회귀한 partial $R^2$ (한계와 측정을 같은 행에서 짝짓는 것), (ii) $V_{\rm rot}$에 같은 이력 조건부 회귀를 적용해 §8ar의 rate-null을 혁신-null로 격상, (iii) 비모수 상호정보 추정으로 선형 하한 위의 여지를 닫기, (iv) 천이 행만의 $\tau_{\rm eq}$ 적합(§8 Part F의 2SLS를 꼬리 행에 한정)으로 시간 영역 표의 어느 행이 실제인지 고정하기.


### P07 — CES 결측은 MNAR이다: 적합을 채택·기각하는 광자 예산이 플라즈마 상태의 함수이고, 그 상태가 $T_i$·$V_\varphi$ 자체를 움직인다

**한 줄 결론.** CES 프레임의 신호 광자수 $N(t)$는 빔 감쇠(선적분 $n_e$)·탄소 밀도·빔 전류의 함수이고, 배경 $B(t)$는 ELM 직후 재순환 중성자와 SOL 밀도로 솟았다가 $\tau_B$로 감쇠한다. Cramér–Rao 한계가 $\sigma^2_{\rm fit}\propto (1/N)(1+cB/N)$이므로 "$\sigma_{\rm fit}<\epsilon$" 또는 "$\chi^2<\chi^2_{\max}$" 꼴의 채택 규칙은 **$N$이 낮거나 $B$가 높은 프레임, 곧 pedestal이 변하는 순간에 기각**된다. 641 shot·247,207행에서 (i) 타깃 자체의 20 ms 점프 상위 5분위에서 결측률이 하위 5분위의 **5.8배**($T_i$·$V_\varphi$ 모두, BES 수준으로 층화해도 3.6–7.8배), (ii) BES(빔 발광 = 광자 예산의 대리) 최하 5분위 결측률 **19.9 %** 대 최상 5분위 **3.7 %**, (iii) BES 급락 사건 뒤 $T_i$ 결측 초과분이 **$\tau_B\approx$ 25–35 ms**로 지수 감쇠, (iv) 두 마스크의 $\phi$ = 0.041(일치 72.8 %, 독립 가정 71.8 %)로 "독립보다 약간 위, 1보다 한참 아래"가 확인된다. 실패한 예측도 있다: ECEI 증분은 결측을 예측하지 못하고(수준 교란, 층내 평탄), $V_\varphi$ 결측은 입력 활동과 무관하며 긴 블록 구조다.

#### 1. 유도

##### 1.1 신호 광자수 $N(t)$ — 상태의 함수

CES는 C VI $n=8\to7$ (529 nm) 능동 전하교환선을 적분 시간 $\tau$ 동안 센다. 시선 체적 $V$, 광학 효율 $\eta$:

$$
N(t)=\eta\,\tau\!\int_V n_b(\mathbf l,t)\,n_C(\mathbf l,t)\,\langle\sigma v\rangle_{\rm cx}\,dV
\;\approx\;\eta\,\tau\,V\,\langle\sigma v\rangle_{\rm cx}\;n_C(t)\;n_b(l_{\rm CES},t),
$$

$$
n_b(l,t)=n_{b0}(t)\exp\!\Big[-\!\int_0^{l}n_e(l',t)\,\sigma_s\!\big(E_b,T_e(l',t)\big)\,dl'\Big]
\quad\Longrightarrow\quad
\ln N(t)=\ln\!\big(\eta\tau V\langle\sigma v\rangle_{\rm cx}\big)+\ln n_{b0}(t)+\ln n_C(t)-\sigma_s\,\bar n_e(t)\,L .
$$

$\sigma_s$는 유효 빔 정지 단면적(이온화 + 전하교환), $L$은 빔이 CES 체적까지 지나는 경로다. 세 항이 각각 결측의 세 경로다. **(a) 빔 전류 $n_{b0}(t)$**: 빔이 꺼지거나 변조되면 $N\to0$(§8at가 본 블록 포락선 규모 결측). **(b) 선적분 밀도 $\bar n_e(t)$**: 밀도 램프·펌프아웃·pedestal 붕괴가 $\Delta\ln N=-\sigma_sL\,\Delta\bar n_e$로 $N$을 **지수적으로** 움직인다. **(c) 국소 탄소 밀도 $n_C(t)$**: ELM 붕괴는 pedestal의 $n_C$를 $n_e$와 함께 떨어뜨리고, 이것이 $T_i$ 급락과 같은 사건이다(theory_plasma_physics §6.3). 또 §8.3의 재순환 중성자 $n_0$는 $T_i$ 싱크이자 수동 CX 배경의 원천이다.

BES는 $I_{\rm BES}\propto n_b(l_{\rm BES},t)\,n_e(l_{\rm BES},t)\,\sigma_{\rm exc}$ (theory_foundations §2.2), 즉 **같은 빔의 같은 감쇠 인자를 재는 계측**이다. 그러므로 (a)·(b)는 입력 $x$ 안에 부분적으로 들어 있고, (c)와 $n_0$는 들어 있지 않다. 이 구분이 §1.6의 "MAR-given-$x$ 성분"과 "MNAR 성분"을 가른다.

##### 1.2 배경 $B(t)$ — ELM 뒤에 솟는다

$$
B(t)=\tau\Big[\underbrace{C_{ff}\,\overline{n_e^2 Z_{\rm eff}\,g_{ff}\,T_e^{-1/2}}}_{\text{제동복사(시선 적분)}}
+\underbrace{n_0(t)\,n_C(t)\,\langle\sigma v\rangle^{\rm pass}_{\rm cx}+n_e\,n_{C^{5+}}\langle\sigma v\rangle_{\rm exc}}_{\text{수동 C VI (가장자리·SOL)}}\Big]+B_{\rm dark}.
$$

ELM 붕괴($t_0$)는 pedestal 입자를 디버터·벽으로 쏟아 재순환 중성자 $n_0$와 SOL 밀도를 올린다. 두 항 모두 그 뒤 완화 시간 $\tau_B$(재순환·중성자 감쇠)로 되돌아온다:

$$
B(t)=B_0+B_1\,e^{-(t-t_0)/\tau_B},\qquad t\ge t_0 .
$$

같은 사건이 pedestal의 $n_C$, $\bar n_e$, $T_i$를 떨어뜨리고 수송 시간 $\tau_n$, $\tau_T$로 회복시킨다(§6.3의 톱니):

$$
T_i(t)=T_{i0}-\Delta T\,e^{-(t-t_0)/\tau_T},\qquad
n_C(t)=n_{C0}\big[1-a\,e^{-(t-t_0)/\tau_n}\big],\qquad
\bar n_e(t)=\bar n_{e0}-\Delta n\,e^{-(t-t_0)/\tau_n}.
$$

##### 1.3 배경이 있는 Cramér–Rao 한계 — 중심과 폭의 민감도가 다르다

theory_foundations §1.2의 $\lambda_k=Np_k(\mu,\sigma)$에 평탄 배경을 더한다: $\lambda(x)=N[\phi(x)+\beta]$ ($x$는 선폭 단위, $\beta$는 선폭당 배경/신호비). Fisher 정보 $I_{\theta\theta}=N\!\int(\partial_\theta\phi)^2/(\phi+\beta)\,dx$를 수치 적분하면 ($\pm6\sigma$ 창, $B/N=12\beta$; `p07_crb_background.py`):

| $B/N$ | 중심 분산 팽창 $1+c_\mu B/N$ | 폭 분산 팽창 $1+c_\sigma B/N$ | $c_\sigma/c_\mu$ |
|---:|---:|---:|---:|
| 0.012 | 1.030 | 1.100 | 3.4 |
| 0.12 | 1.17 | 1.44 | 2.6 |
| 0.36 | 1.39 | 1.88 | 2.2 |
| 1.2 | 2.01 | 2.93 | 1.9 |

즉

$$
\sigma^2_{\hat\mu}\simeq\frac{\sigma_\lambda^2}{N}\Big(1+c_\mu\frac{B}{N}\Big),\qquad
\sigma^2_{\hat\sigma}\simeq\frac{\sigma_\lambda^2}{2N}\Big(1+c_\sigma\frac{B}{N}\Big),\qquad c_\sigma\approx(2\text{–}3)\,c_\mu .
$$

폭 정보는 $\partial_\sigma\phi=(x^2-1)\phi$가 큰 **날개**($|x|\approx\sqrt3$)에 실리고 거기서 $\phi$가 작아 배경이 더 세게 경쟁한다. 배경 없는 극한에서 $I_{\mu\sigma}=0$(§1.2)은 배경이 있어도 홀함수 적분이라 유지된다. 또 $\sigma_\lambda\propto\sqrt{T_i}$이므로 선 아래 배경 $B_{\rm line}\propto\sigma_\lambda$: 뜨거울수록 유효 $B/N$이 커진다.

##### 1.4 채택 규칙 → 결측 확률

파이프라인의 실제 QC는 비공개이므로 일반형으로 쓴다. 폭(온도) 채택은 상대 정밀도, 이동(회전) 채택은 절대 정밀도와 기준 파장 정합:

$$
o_T(t)=\mathbf 1\!\Big[\tfrac{\sigma_{\hat\sigma}}{\hat\sigma}<\epsilon_T\Big]\;\mathbf 1\big[\chi^2_\nu<\chi^2_{\max}\big],\qquad
o_V(t)=\mathbf 1\big[\sigma_{\hat v}<\epsilon_V\big]\;\mathbf 1\big[|\hat\mu-\mu_{\rm ref}|<\Delta_{\max}\big].
$$

CRB를 넣으면 각각 $N$의 문턱이 된다:

$$
o_T=1\iff N>N^{\rm crit}_T(B)=\frac{1+\sqrt{1+8\epsilon_T^2c_\sigma B}}{4\epsilon_T^2},\qquad
o_V=1\iff N>N^{\rm crit}_V(B,T_i)\approx\frac{v_{th}^2(T_i)}{\epsilon_V^2}\Big(1+c_\mu\frac{B}{N}\Big),\; v_{th}\propto\sqrt{T_i}.
$$

**$o_V$의 문턱에는 $T_i$ 자체가 들어간다**(넓은 선은 중심을 못 잡는다); $o_T$의 문턱은 상대 규칙이라 $N,B$만의 함수이나 $B_{\rm line}\propto\sqrt{T_i}$를 통해 간접 의존한다. 프레임 간 광자수 요동을 로그정규($s$)로 두면

$$
P(o_c=1\mid s_t)=\Phi\!\left(\frac{\ln N(t)-\ln N^{\rm crit}_c\big(B(t),T_i(t)\big)}{s}\right),\qquad c\in\{T,V\}.
$$

#### 시간 $t$의 식

ELM(또는 pedestal 밀도 급락)이 $t_0$에 일어났다고 하자. 상태 $s_t=(n_{b0},\bar n_e,n_C,n_0,T_e,T_i,V_\varphi)(t)$에서 시간 영역의 식을 모아 쓰면:

**신호 광자수 — 플라즈마 상태의 함수**

$$
N(t)=\eta\,\tau\,V\,\langle\sigma v\rangle_{\rm cx}\;n_{b0}(t)\;n_C(t)\;\exp\!\big[-\sigma_s\,\bar n_e(t)\,L\big],
$$

$$
N(t)\big|_{t\ge t_0}=N_0\,\big[1-a\,e^{-(t-t_0)/\tau_n}\big]\;
\exp\!\big[\sigma_s\,\ell_{\rm ped}\,\Delta n\,e^{-(t-t_0)/\tau_n}\big],\qquad
N_0=\eta\tau V\langle\sigma v\rangle_{\rm cx}\,n_{b0}\,n_{C0}\,e^{-\sigma_sL\bar n_{e0}} .
$$

**배경 — ELM 뒤 지수 감쇠 (측정값 $\tau_B\approx$ 25–35 ms, §2.4)**

$$
B(t)=B_0+B_1\,e^{-(t-t_0)/\tau_B},\qquad \tau_B=2.6\text{–}3.5\ \text{행}=26\text{–}35\ \text{ms (BES 급락 정렬, }T_i\text{ 결측 초과분의 감쇠)}.
$$

**타깃 자체의 운동 — 같은 사건, 같은 지수**

$$
T_i(t)=T_{i0}-\Delta T\,e^{-(t-t_0)/\tau_T},\qquad
V_\varphi(t)=V_0-\Delta V\,e^{-(t-t_0)/\tau_\varphi}.
$$

**적합 정밀도 — $N(t)$, $B(t)$, $T_i(t)$의 함수**

$$
\sigma^2_{\hat\sigma}(t)=\frac{\sigma_\lambda^2\big(T_i(t)\big)}{2N(t)}\Big(1+c_\sigma\frac{B(t)}{N(t)}\Big),\qquad
\sigma^2_{\hat v}(t)=\frac{v_{th}^2\big(T_i(t)\big)}{N(t)}\Big(1+c_\mu\frac{B(t)}{N(t)}\Big).
$$

**채택 확률 — $N(t)$, $B(t)$를 거쳐 타깃 자체의 운동의 함수**

$$
P(\text{obs}_c\text{ at }t)=\Phi\!\left(\frac{\ln N(t)-\ln N_c^{\rm crit}\big(B(t),T_i(t)\big)}{s}\right),\qquad
P(\text{miss}_c\text{ at }t)=1-P(\text{obs}_c\text{ at }t),\qquad c\in\{T,V\}.
$$

$t\ge t_0$에서 전개하면, 1차로

$$
\ln\frac{N(t)}{N_c^{\rm crit}(t)}\approx \ln\frac{N_0}{N_c^{\rm crit,0}}
-\Big[a-\sigma_s\ell_{\rm ped}\Delta n\Big]e^{-(t-t_0)/\tau_n}
-c_c\frac{B_1}{N_0}\,e^{-(t-t_0)/\tau_B}
-\kappa_c\frac{\Delta T}{T_{i0}}\,e^{-(t-t_0)/\tau_T},
$$

여기서 $\kappa_V\approx\tfrac12$ ($v_{th}\propto\sqrt{T_i}$), $\kappa_T$는 $B_{\rm line}\propto\sqrt{T_i}$를 통한 약한 간접항이다. pedestal 채널에서는 국소 발광 손실($a$)이 경로 전체의 감쇠 완화($\sigma_s\ell_{\rm ped}\Delta n$, $\ell_{\rm ped}\ll L$)를 이기므로 $N$은 **내려가고** $B$는 **올라가** $B/N$이 $t_0$ 직후 최대, 이후 $\min(\tau_n,\tau_B)$로 완화한다. 그래서 결측 확률은 $t_0$에서 최대이고 $\tau_B$ 정도로 감쇠하며, **같은 $e^{-(t-t_0)/\tau}$가 $T_i(t)$의 급락·회복을 지배**한다. 결측은 타깃이 움직이는 바로 그 구간에 몰린다. 데이터(§2.4): $t_0$의 $T_i$ 결측률 20.6 % → 사전 기준 13.1 %, 초과분 7.5 → 5.4 → 3.5 → 1.2 pp (지연 0, 1, 2, 5행).

##### 1.6 왜 MNAR인가 — 형식적 진술과 학습·평가에 미치는 영향

$y_t=(T_i,V_\varphi)(t)$, $x_t$ = BES·ECEI·MC·과거 CES, $m_{t,c}=1-o_c(t)$. MAR은 $P(m\mid y,x)=P(m\mid x)$다. 위에서 $m$은 $N(t),B(t)$를 통해 $s_t$의 함수이고, $s_t$ 가운데 **$x_t$에 없는 성분**($n_C$, $n_0$, pedestal 국소 $\bar n_e$ 변화, 그리고 $N^{\rm crit}_V$ 속의 $T_i$ 자체)이 $y_t$와 같은 사건으로 움직인다. 따라서

$$
P(m_{t,c}\mid y_t,x_t)\neq P(m_{t,c}\mid x_t)\quad\text{(MNAR)},
$$

다만 $n_{b0}$·$\bar n_e$를 통한 부분은 BES가 재므로 **MAR-given-$x$**다. 데이터가 이 분해를 그대로 보여 준다(§2): BES 수준 효과(×5.3)는 $x$ 안에 있고, BES 수준으로 층화한 뒤에도 남는 타깃 점프 효과(×3.6–7.8)가 식별 불가능한 MNAR 성분이다.

**학습.** theory_foundations §6.3: 타깃별 마스킹 MSE의 모집단 최소자는 $E[y_c\mid x,m_c=0]$. 결측이 큰 $|\Delta y|$에 몰리면 관측 조건부 평균은 참 조건부 평균보다 **매끈한** 함수를 겨냥하므로 모델은 결측 지점의 천이를 과소 추정한다. **평가.** 관측 행 skill $=E[L\mid m=0]$, 배치 목표는 $E[L\mid m=1]$. §12의 사후층화

$$
E_{\rm miss}[L]=\sum_sP(s\mid{\rm miss})E[L\mid s,{\rm obs}]+\sum_sP(s\mid{\rm miss})\big(E[L\mid s,{\rm miss}]-E[L\mid s,{\rm obs}]\big)
$$

에서 둘째 항이 층내 MNAR 편향이다. §2의 J-검정은 입력만으로 정의된 활동 층 안에서도 결측 행의 타깃 점프가 더 크다는 뜻이므로 둘째 항은 0이 아니고 **부호가 양(더 어렵다)**이다. 즉 관측 행 skill은 낙관적 상한이며, §8i의 "optimism +0.06…+0.12"는 첫째 항만 교정한 값이다.

##### 1.7 두 타깃이 독립적으로 결측되는 이유 — 예측

$o_T=f_T(N,B)$와 $o_V=f_V(N,B,T_i,\text{calib})$는 **같은 프레임의 서로 다른 범함수**다. 공통 인자 $(N,B)$ 때문에 완전 독립일 수 없고, 다른 문턱($N_T^{\rm crit}\neq N_V^{\rm crit}$)·다른 민감도($c_\sigma>c_\mu$, $T_i$ 의존은 $o_V$만, 기준 파장 정합은 $o_V$만, 기기 함수 역합성은 $o_T$만) 때문에 한쪽 기각이 다른 쪽 기각을 함의하지 않는다(§1.6의 $I_{\mu\sigma}=0$). 단일 광자 예산 게이트라면 두 기각은 포개져($\phi\to1$) 일치도가 1에 가깝고, 완전히 다른 기준이면 $\phi\to0$. 예측: **$\phi$는 0보다 크지만 작고, 일치도는 독립 가정값보다 약간 높으며 1에는 한참 못 미친다.** 또 공통 인자가 지배하는 저신호 구간(빔 꺼짐·램프)에서만 결합이 보일 것이다.

#### 2. 데이터 검사 (641 shot, 247,207행, 블록 1,972개 — 0.5 s 이상 간극에서 분할; §8at는 4행 미만 블록을 버려 667개)

설계: 마스크는 두 가지. **raw** = CSV의 NaN, **genuine** = NaN 또는 held 반복(같은 블록 안에서 직전 관측값과 비트 동일, 로더의 `_stuck_repeat_mask` 규칙을 두 타깃에 적용). held는 $V_\varphi$에서만 나타난다(행의 41.1 %, 관측값의 54.0 %; $T_i$ held = 0행). 그러므로 $T_i$는 raw = genuine(결측 8.18 %), $V_\varphi$는 raw 23.91 % / genuine 65.01 %. 활동 대리변수: 타깃 자체의 이웃 점프 $J_t=|y(t+k_2)-y(t-k_1)|$(5스텝 이내 최근접 관측 이웃, 행 자신 제외), **스팬 정합 점프** $J_2=|y(t+1)-y(t-1)|$(양 이웃이 모두 관측될 때만 — 고립 결측 프레임과 관측 프레임을 같은 20 ms 스팬에서 비교해 "긴 스팬 → 큰 점프"의 기계적 편향 제거), 파일별 z-점수화한 ECEI·BES 평균의 중심 증분 $|dE|,|dB|$와 BES·ECEI 수준(모든 행에서 정의). Q5−Q1 차의 95 % CI는 shot 군집 bootstrap(B = 1,000).

##### 2.1 (a) $T_i$ 결측률 — 활동 5분위별

| 대리변수 (마스크) | Q1 | Q2 | Q3 | Q4 | Q5 | Q5−Q1 [95 % CI] | Q5/Q1 | 단조 |
|---|---:|---:|---:|---:|---:|---|---:|---|
| $J_{T_i}$, $k\le5$ (genuine) | 2.01 % | 2.00 % | 2.58 % | 4.28 % | 12.30 % | +10.3 pp [+9.0, +11.8] | 6.1 | Q2부터 |
| **$J_2$ 스팬 정합 (genuine)** | 0.77 % | 0.81 % | 1.07 % | 1.74 % | 4.45 % | +3.7 pp [+3.1, +4.3] | **5.8** | 예 |
| $J_2$ (3 keV 컷 모집단) | 0.83 % | 0.85 % | 1.14 % | 1.87 % | 4.34 % | +3.5 pp [+3.0, +4.2] | 5.3 | 예 |
| $|dB|$ BES 증분 | 8.34 % | 6.86 % | 6.06 % | 6.45 % | 10.54 % | +2.2 pp [+0.6, +4.0] | 1.26 | U자 |
| $|dE|$ ECEI 증분 | 10.75 % | 7.42 % | 6.61 % | 6.83 % | 6.63 % | −4.1 pp [−6.0, −2.3] | 0.62 | **아니오** |
| BES 수준 (낮음→높음) | **19.85 %** | 8.11 % | 5.39 % | 3.81 % | 3.73 % | −16.1 pp [−18.0, −14.1] | 0.19 | 감소 |
| ECEI 수준 | 14.63 % | 8.64 % | 6.63 % | 6.14 % | 4.84 % | −9.8 pp [−11.6, −8.2] | 0.33 | 감소 |

BES 수준(×ECEI 활동 교란 제거)으로 층화한 $J_2$와 $|dB|$ (genuine):

| BES 수준 층 | $J_2$ Q1 → Q5 | 배수 | $|dB|$ Q1 → Q5 | $|dE|$ Q1 → Q5 |
|---|---|---:|---|---|
| q1 (최저) | 1.20 % → 9.39 % | 7.8 | 23.3 % → 18.3 % | 24.5 % → 13.4 % |
| q2 | 0.88 % → 5.87 % | 6.7 | 6.61 % → 12.93 % | 8.73 % → 8.61 % |
| q3 | 0.64 % → 3.20 % | 5.0 | 4.01 % → 8.84 % | 5.45 % → 5.50 % |
| q4 | 0.50 % → 2.17 % | 4.3 | 2.61 % → 6.42 % | 3.66 % → 3.92 % |
| q5 (최고) | 0.56 % → 2.01 % | 3.6 | 3.04 % → 5.33 % | 3.14 % → 3.23 % |

읽기. **타깃 자체의 움직임**이 가장 강한 결측 예측자다: 스팬 정합·3 keV 컷·BES 층화 어느 조건에서도 단조 증가가 유지된다. 고립 결측 프레임 가운데 $J_2$ 상위 5분위에 있는 비율은 50.3 %(관측 프레임은 19.5 %). **BES 수준**(빔 발광 = $n_b n_e$, 곧 §1.1의 광자 예산 대리)은 ×5.3로 두 번째로 강하고, **BES 증분**(밀도 천이)은 q2–q5 층 안에서 ×2로 산다. **ECEI 증분은 예측에 실패**한다: 비층화에서는 오히려 감소하는데, 이는 ECEI가 조용한 행이 저 BES(빔 꺼짐·램프) 구간과 겹치는 수준 교란이고, 층 안에서는 평탄하다. 10 ms 격자의 ECEI 증분은 ELM 붕괴를 한 점으로만 보므로 §1.2의 메커니즘을 재는 데 둔하다. 보조 결과: $T_i$ 이웃 수준 5분위별 결측률 11.2 / 3.8 / 1.6 / 1.6 / 5.0 % — 오른팔의 상승이 $B_{\rm line}\propto\sqrt{T_i}$(§1.3)와 정합한다.

##### 2.2 (b) $V_\varphi$ 결측률

| 대리변수 (마스크) | Q1 | Q2 | Q3 | Q4 | Q5 | Q5−Q1 [95 % CI] | Q5/Q1 |
|---|---:|---:|---:|---:|---:|---|---:|
| $J_{V}$, $k\le5$ (raw) | 정의 불가 | | | | | held 반복으로 $J=0$ 동률이 40 %를 넘어 Q1·Q2가 비어 있음 | |
| **$J_{V}$, $k\le5$ (genuine, 덮임 36 %)** | 1.80 % | 2.71 % | 4.08 % | 5.54 % | 13.61 % | +11.8 pp [+9.5, +14.3] | **7.6** |
| **$J_2$ 스팬 정합 (genuine, 33 %)** | 0.91 % | 1.06 % | 1.88 % | 2.86 % | 5.27 % | +4.4 pp [+3.5, +5.3] | **5.8** |
| $|dE|$ (raw / genuine) | 27.4 / 69.2 % | 23.0 / 64.5 % | 22.8 / 64.1 % | 22.9 / 63.6 % | 21.4 / 63.2 % | −6.1 / −6.0 pp (둘 다 CI가 0 제외) | 0.78 / 0.91 |
| $|dB|$ (raw / genuine) | 26.5 / 65.1 % | 25.0 / 65.5 % | 23.6 / 65.3 % | 21.8 / 64.4 % | 20.6 / 64.2 % | −5.9 pp / −0.9 pp n.s. | 0.78 / 0.99 |
| BES 수준 (raw / genuine) | 29.8 / 68.8 % | 22.0 / 64.2 % | 23.2 / 65.0 % | 23.0 / 63.3 % | 21.7 / 63.8 % | −8.1 / −5.0 pp | 0.73 / 0.93 |
| $J_{T_i}$ (genuine V 마스크) | 67.0 % | 66.3 % | 64.8 % | 63.1 % | 62.9 % | −4.1 pp [−8.6, +0.6] n.s. | 0.94 |

읽기. $V_\varphi$ 결측은 **자기 자신의 움직임**에는 $T_i$와 같은 강도(×5.8)로 반응하지만 **입력 활동·BES 수준·$T_i$ 점프에는 반응하지 않는다**(평탄 또는 약한 감소). 고립 결측 프레임의 44.0 %가 $J_2$ 상위 5분위(관측은 19.4 %). 이것은 §1.7의 예측과 맞는다: $o_V$를 떨어뜨리는 기준(기준 파장 정합, 선 중심 S/N, 그리고 held 출력 정책)은 광자 예산보다 **계측·후처리 쪽**이고, 대부분의 $V_\varphi$ 결측은 2.4절처럼 긴 블록이다.

##### 2.3 (c) 마스크 일치와 결합

| 양 | raw | genuine |
|---|---:|---:|
| $P(T\text{ miss})$ / $P(V\text{ miss})$ | 8.18 % / 23.91 % | 8.18 % / 65.01 % |
| $P(\text{both miss})$ | 2.43 % | 5.61 % |
| $P(T)P(V)$ (독립) | 1.96 % | 5.32 % |
| 비 $P(\text{both})/P(T)P(V)$ | 1.24 | 1.05 |
| $P(V\text{ miss}\mid T\text{ miss})$ vs $P(V\text{ miss}\mid T\text{ obs})$ | 29.7 % vs 23.4 % | 68.6 % vs 64.7 % |
| **마스크 일치도** | **72.77 %** (§8at 72.8 % 재현) | 38.03 % |
| 독립 가정의 일치도 $pq+(1-p)(1-q)$ | 71.82 % | 37.45 % |
| $\phi$ 계수 | **0.041** | 0.022 |
| $\phi$, $|dE|$ 5분위 안 (Q1→Q5) | 0.052, 0.007, 0.003, −0.011, −0.030 | 0.003, 0.009, 0.006, 0.005, 0.020 |

읽기. 예측대로 **독립보다 약간 위(1.24배), 1보다 한참 아래**($\phi$ = 0.04). 일치도 72.8 %의 거의 전부(71.8 %)는 독립만으로 설명되고, 초과 1 pp가 공통 인자 $(N,B)$의 몫이다. 그 결합은 ECEI가 가장 조용한 저신호 층(Q1, $\phi$ 0.052)에 몰려 있고 활동 층에서는 사라지거나 뒤집힌다(Q5, −0.030): "게이트는 공유, 플라즈마 안의 기준은 다르다"는 §1.7의 그림 그대로다. held가 $V_\varphi$에만 있다는 사실 자체가 두 모수의 후처리가 분리되어 있다는 독립 증거다.

##### 2.4 (d) 결측 런 길이와 도메인 내 비율, 그리고 시간 영역 사건 정렬

| 통계 | $T_i$ (raw = genuine) | $V_\varphi$ raw | $V_\varphi$ genuine |
|---|---:|---:|---:|
| 결측 런 수 | 8,233 | 4,351 | 5,295 |
| 중앙값 / 평균 / p90 (행) | 1 / 2.46 / 4 | 1 / 13.58 / 6 | 1 / 30.35 / **91** |
| 길이 1 런의 비율 | 63.7 % | 66.9 % | 62.0 % |
| 길이 ≤ 2 런에 속한 결측 행 비율 | 39.0 % | 6.8 % | 2.8 % |
| 도메인 내(2행 이내 과거 관측) | **48.2 %** | 6.9 % | **3.5 %** |
| 도메인 내(4행 이내) | 61.4 % | 8.8 % | 4.9 % |

§8at(1 / 2.81 / 5, 1 / 18.83 / 11; 4행 미만 블록 제외)와 같은 그림이다. $T_i$ 결측은 고립 프레임(시간 $t$의 식의 $t_0$ 한 점)이고 $V_\varphi$ 결측은 블록이다. 전체 결측 행을 합쳐 센 도메인 내 비율 48 % / 3.5 %는 §8ab 9절의 분할·컷 모집단별 54–68 % / 4–6 %와 같은 체제다.

**사건 정렬(시간 $t$의 식의 직접 측정, `p07_event_study.py`).** 입력만으로 정의한 밀도 급락 사건 $t_0$ = 파일별 z-BES 평균의 1스텝 증분이 하위 1 %(−2.01)보다 작은 행(10행 불응기, 2,126건). $T_i$ 결측률은 사전 기준(지연 −5…−2) 13.1 %에서 **$t_0$에 20.6 %**로 뛰고 지연 +1, +2, +5, +8에서 초과분 7.5 → 5.4 → 3.5 → 1.2 → −0.3 pp로 감쇠한다. 지연 0–8의 로그-선형 적합 **$\tau_B$ = 3.5행 = 35 ms**(하위 0.5 % 사건 1,152건: 2.6행 = 26 ms). 직전 행(−1)은 6.7 %로 오히려 낮다 — 붕괴 직전 pedestal 밀도가 최고여서 $N$이 최대인 순간이다. $V_\varphi$ genuine 결측률은 전 지연에서 61.6–64.6 %로 평탄하다(사건 반응 없음).

#### 3. 기록과의 연결

- **theory_foundations §1.3·§12, plasma §6.3·§8.3**이 말로만 둔 "결측은 물리적으로 흥미로운 순간에 몰린다"를 식($N(t),B(t),P(\text{obs})$)과 수치(×5.8, $\tau_B$ 25–35 ms)로 바꿨다. §1.6의 $I_{\mu\sigma}=0$ 근거에 $c_\sigma/c_\mu\approx2$–3과 $\phi=0.041$이 붙는다.
- **§8at**의 72.8 %를 72.77 %로 재현하고, 그 가운데 71.8 %가 독립의 몫임을 보여 "독립적으로 결측된다"는 데이터 계약의 가정을 정량화했다. 듀티 사이클 부재 판정과도 정합(사건 정렬에서 주기 아닌 단발 감쇠).
- **§8i·§8ab 9절**의 사후층화는 첫째 항만 교정한다. 2.1의 층화 결과는 입력만의 활동 층 안에서도 결측 행이 더 크게 움직임을 뜻하므로 둘째 항(층내 MNAR 편향)은 양이고, 관측 행 skill은 상한이다. 도메인 내 비율 48 % / 3.5 %는 54–68 % / 4–6 %와 같은 체제이며, $V_\varphi$에서 재가중 추정이 결측 질량의 5 %만 답한다는 판정의 이유(긴 블록, 평균 30행)를 2.4가 준다.
- **§8r**: held가 $V_\varphi$에만 있고(관측의 54.0 %; §8r의 53.5–60.8 %), 그것이 $J_V$의 raw 5분위를 무너뜨린다(동률). held를 제거해야 $V_\varphi$의 자기 움직임 의존이 보인다 — genuine 처리가 분석의 전제라는 §8r의 결론을 결측 분석 쪽에서 다시 확인한다.
- **plasma §6.4(RMP 펌프아웃)·§1.1(b)**: BES 수준 최하 5분위의 19.9 %는 저밀도·빔 꺼짐 구간의 광자 예산 붕괴로, 이 성분은 BES가 재므로 MAR-given-$x$다. 모델이 이미 조건부로 삼는 부분과 삼지 못하는 부분(탄소·중성자·타깃 자체)을 데이터가 갈라 보여 준다.

#### 4. 한계

1. **적합 품질 메타데이터가 없다.** $\sigma_{\rm fit},\chi^2$, 프레임 광자수가 파일에 없으므로 §1.4의 채택 규칙은 추론이고, 문턱 $\epsilon_T,\epsilon_V$·$c_\mu,c_\sigma$의 실제 값은 모른다. **뒤집을 측정**: KFE의 per-frame 적합 품질(NN-CES 저자 W.H. Ko가 CES 책임자, §8at)과 원시 프레임 광자수. 그것이 있으면 $P(\text{obs}\mid N,B)$를 직접 적합해 시간 $t$의 식의 $\Phi$ 꼴과 $\tau_B$를 검증하고, 값 기반 3 keV 컷을 대체할 수 있다.
2. **ECEI 증분 예측은 실패했다.** 10 ms ECEI 증분은 결측을 예측하지 못한다(층내 평탄). 메커니즘이 틀렸다기보다 대리변수가 둔한 것으로 읽지만, 이 검사로는 구분할 수 없다. 구분할 측정: 원시 kHz ECEI/Mirnov로 ELM 붕괴 시각을 잡아 사건 정렬을 다시 하는 것(현재 정렬은 BES 급락으로만 했다).
3. **$J$-검정은 타깃 이웃이 필요**하므로 긴 결측 블록(특히 $V_\varphi$, 덮임 33–36 %)은 보지 못한다. 블록 결측의 원인(빔 변조·기준 파장·held 정책)은 이 데이터로 식별되지 않으며, §8at의 결론과 같이 빔 변조 로그가 필요하다.
4. 모든 효과는 shot 군집 bootstrap CI로 뒷받침되지만 **기술 통계**다. 결측 확률의 함수형($\Phi$, 지수 감쇠)은 적합이지 검정이 아니며, $\tau_B$ 26–35 ms는 사건 정의(하위 1 % vs 0.5 %)에 따라 움직인다.
5. 모델 학습·채점 없음(규칙 준수). 학습된 모델이 실제로 천이를 과소 추정하는지(§1.6의 함의)는 별도 측정이다: 관측 행 가운데 $J_2$ 5분위별 잔차 부호를 보면 된다.

#### 스크립트와 산출물

- `scratchpad/p07\p07_mnar_check.py` → `p07_results.json`, `p07_output.txt` (2.1–2.4 표 전체)
- 같은 폴더의 `p07_crb_background.py` (1.3 표), `p07_event_study.py` → `p07_event_p1.json`, `p07_event_p05.json` (2.4 사건 정렬)
- 인용 파일: `docs/theory_foundations.md` §1.2–1.6, 6.3–6.4, 12; `docs/theory_plasma_physics.md` §6.3–6.4, 8.3; `THESIS_RESULTS.md` §8i, §8ab 9절, §8at, §8r; held 규칙 `ces_prediction/dataset.py::_stuck_repeat_mask`


### P08. 방전별 승률 비대칭(T_i 70 % vs V_rot 48 %)은 두 생성 모형에서 따라 나온다

**명제.** 변동이 *관측된 입력*으로 구동되는 양은 변동하는 곳마다 이기고, 변동이 *관측되지 않은 입력*으로 구동되는 양은 그 변동이 우연히 사건 시점으로 빠른 채널에 비칠 때만 이긴다. 따라서 승률은 방전의 타깃 산포에 대해 증가하되 천장이 $p_{\rm obs}$로 정해지고, $T_i$의 천장은 1 근처, $V_\varphi$의 천장은 0.5–0.6이다.

#### 1. 유도

**설정.** 한 방전 $i$의 $n$개 행에서 타깃을 OU 배경 + 충격으로 쓴다(foundations §7.1 + plasma §12.1/§12.2의 "ELM/천이 충격" 항).
$$
y_t = s_t + \sum_{k=1}^{N_e} A_k\,h(t-t_k)+n_t,\qquad
s:\ \text{OU}(\sigma_s^2,\tau),\quad \mathrm{Var}\,n_t=\sigma_n^2,\quad E[A_k^2]=A^2 .
$$
충격 하나가 과거만 보는 추정량에 남기는 예측 불가 에너지를 $\kappa A^2$라 한다($\kappa$ = 기준선이 다음 관측으로 다시 닻을 내릴 때까지 새어 나가는 에너지의 행 단위 환산; §1′에서 $h$로부터 계산). 방전의 **충격 에너지 밀도**를 $E_{\rm imp}\equiv N_e\kappa A^2/n$으로 정의한다.

**기준선(인과 GP).** OU 부분에는 정상상태 Kalman 필터(foundations §7.4)로 닫힌 형식의 최적 MSE $M_{\rm OU}(\sigma_s^2,\sigma_n^2,\rho)$를 내고, 충격은 시점도 크기도 이력에 없으므로 그대로 오차로 남는다.
$$
\mathrm{MSE}_{\rm base}=M_{\rm OU}+E_{\rm imp}.
$$

**모델(이력 + 빠른 진단 $d$채널).** 두 항으로 분해한다.

(i) *본류 비용* $c_b$. 입력이 타깃에 대해 정보를 갖지 않는 행에서, 무관한 입력 $d$개의 계수를 $n_{\rm eff}$개의 유효 표본으로 적합한 추정량은 초과 위험 $\sigma_e^2 d/n_{\rm eff}$를 낸다(OLS 초과 위험의 표준 결과; LSTM은 상수가 다르지만 부호는 같다 — 영 신호 입력은 예측 분산만 더한다). 이것이 §8an 평결 1의 *"adds variance to a problem that had none"*의 유도이며, 타깃이 무엇이든 같은 크기라는 예측을 낳는다.
$$
c_b\;\approx\;\frac{\sigma_e^2\,d}{n_{\rm eff}}\;>\;0 .
$$

(ii) *충격 회수.* 충격 $k$의 발생이 같은 행의 빠른 채널에 보일 확률을 $p_{\rm obs}$, 보였을 때 회수되는 에너지 비율을 $r$이라 한다.
- $T_i$: ECEI/BES가 같은 pedestal 사건($T_e$·$n_e$의 급변)을 보고, $Q_{ei}\propto n_e(T_e-T_i)$ 결합(plasma §12.1)이 크기까지 전달하므로 $p_{\rm obs}$ 높고 $r\to1$.
- $V_\varphi$: 구동 토크(NBI·NTV·잔여응력)가 데이터에 없다(plasma §12.2, "입력 중 데이터에 있는 것: 없음"). 빠른 채널에 사건 *시점*만 비치고 *크기*는 영영 안 보이므로, $r=(E[A\mid\text{vis}])^2/E[A^2]\le1-\mathrm{CV}_A^2$, 부호가 무작위면 $r=0$, 반응이 틀리면 $r<0$.
$$
\mathrm{MSE}_{\rm model}=M_{\rm OU}+c_b+(1-p_{\rm obs}\,r)\,E_{\rm imp}.
$$

**방전별 기대 짝 MSE 차.**
$$
\boxed{\;D\equiv\mathrm{MSE}_{\rm base}-\mathrm{MSE}_{\rm model}=p_{\rm obs}\,r\,E_{\rm imp}-c_b\;}
$$

(a) **조용한 방전**($E_{\rm imp}\ll\sigma_s^2$): $D\approx-c_b<0$. OU 본류에서 GP는 이미 닫힌 형식의 최적이라 이길 것이 없고, 모델은 추정 분산 $\sigma_e^2 d/n_{\rm eff}$만큼 진다.

(b) **변동 방전**: $D>0\iff p_{\rm obs}\,r\,E_{\rm imp}>c_b$. 즉 **회수 가능한 충격 SSE가 본류 비용을 넘을 때만** 이긴다.

(c) **승률의 형태.** 산포는 $\mathrm{Var}(y)=\sigma_s^2+\sigma_n^2+(m_h/\kappa)E_{\rm imp}$로 $E_{\rm imp}$에 증가하므로, 방전마다 $(p_i,r_i,c_i)$가 분포 $F$를 따르면
$$
P(\mathrm{win}\mid E)=\int\mathbf 1\!\left[p\,r\,E>c\right]dF(p,r,c)\quad\text{는 }E\text{에 대해 증가하고,}\qquad
\lim_{E\to\infty}P(\mathrm{win}\mid E)=P(p_{\rm obs}\,r>0)\equiv\Pi .
$$
천장 $\Pi$는 "변동 가운데 보이면서 회수 가능한 성분이 하나라도 있는 방전의 비율"이다. 승리 조건의 최종 부등식은
$$
\boxed{\;\text{win}_i\iff p_{{\rm obs},i}\,r_i\,\frac{N_{e,i}\kappa A_i^2}{n_i}>\frac{\sigma_{e,i}^2 d}{n_{{\rm eff},i}},\qquad
\Pi_{T_i}=P(p\,r>0)\approx1,\quad \Pi_{V_\varphi}=P(p\,r>0)\approx0.5\text{–}0.6\;}
$$
$T_i$는 $p\,r\approx1$이라 "변동하면 곧 이긴다"($E_{\rm imp}>c_b$가 곧 변동의 정의). $V_\varphi$는 $p_{\rm obs}$가 우연 수준이고 $r$이 작아 변동해도 절반만 이긴다. 두 타깃의 **조용 삼분위 손실은 $-c_b$로 같아야 한다**(본류 비용은 구동원과 무관)는 것이 이 유도의 부수 예측이다.

#### 1′. 시간 t의 식

위 §1의 모든 양을 시간 $t$의 명시적 식으로 다시 쓴다. 격자 간격 $\Delta=10$ ms, 관측 시각 $t_j=j\Delta$, 방전 길이 $T=n\Delta$.

**생성 모형.** 사건 $k$가 $t_k$에 일어나고, 충격 응답은 "계단 뒤 지수 회복"이다.
$$
y(t)=y_{\rm OU}(t)+\sum_{k=1}^{N_e}A_k\,h(t-t_k)+n(t),\qquad
h(s)=\Theta(s)\,e^{-s/\tau_{\rm rec}},\qquad
dy_{\rm OU}=-\frac{y_{\rm OU}-\bar m}{\tau}dt+\sqrt{\tfrac{2\sigma_s^2}{\tau}}\,dW .
$$
$T_i$는 plasma §12.1의 식에서 ELM 충격 뒤 $\dot T_i=(T_e-T_i)/\tau_{\rm eq}-T_i/\tau_{E,i}+\dots$로 되돌아오므로 $\tau_{\rm rec}$는 데이터 완화 103–140 ms(§12.1)다. $V_\varphi$는 §12.2의 식 $\dot V_\varphi=\sum \mathcal T/(nm\langle R^2\rangle/R)-V_\varphi/\tau_\varphi$, $\tau_\varphi\approx89$ ms(데이터).

**기준선의 오차 $e_b(t)$ (사건 시각 기준 전·중·후).** 이력만 보는 최적 필터는 $t_k$를 미리 알 수 없고, $t_k$ 직후의 첫 관측으로 다시 닻을 내린 뒤에는 $h$의 회복을 한 걸음씩 뒤쫓는다. $q\equiv e^{-\Delta/\tau_{\rm rec}}$, $s\equiv t-t_k$라 하면
$$
e_b(t)=e_{\rm OU}(t)+\sum_k A_k\,g(t-t_k),\qquad
g(s)=\underbrace{\mathbf 1[0\le s<\Delta]}_{\text{사건 직후 행: 점프 전체}}+\underbrace{(1-q)\,e^{-s/\tau_{\rm rec}}\,\mathbf 1[s\ge\Delta]}_{\text{회복 중: 한 걸음씩 놓치는 양}},
$$
$$
\text{전 }(s<0):\ e_b=e_{\rm OU};\qquad
\text{중 }(0\le s<\Delta):\ e_b=e_{\rm OU}+A_k;\qquad
\text{후 }(s\ge\Delta):\ e_b=e_{\rm OU}+A_k(1-q)e^{-s/\tau_{\rm rec}} .
$$
$e_{\rm OU}$는 foundations §7.4의 필터 혁신, $E[e_{\rm OU}^2]=M_{\rm OU}$.

**모델의 오차 $e_m(t)$.** 모델은 같은 이력에 빠른 채널 $x(t)$를 더한다. 사건 $k$의 시점이 $x(t_k)$에 보이는 사건을 $\xi_k\sim\mathrm{Bernoulli}(p_{\rm obs})$, 보였을 때 추정한 점프를 $\hat A_k=E[A_k\mid x(t_k)]$라 하면
$$
e_m(t)=e_{\rm OU}(t)+\varepsilon_b(t)+\sum_k\bigl(A_k-\xi_k\hat A_k\bigr)\,g(t-t_k),\qquad
E[\varepsilon_b^2]=c_b\ \ \forall t,\qquad
r\equiv1-\frac{E[(A-\hat A)^2]}{E[A^2]} .
$$
전($s<0$): $e_m=e_{\rm OU}+\varepsilon_b$ — 기준선보다 $\varepsilon_b$만큼 나쁘다. 중($0\le s<\Delta$): $e_m=e_{\rm OU}+\varepsilon_b+(A_k-\xi_k\hat A_k)$. 후($s\ge\Delta$): 같은 잔차 $(A_k-\xi_k\hat A_k)$가 $g$를 따라 감쇠한다.

**시간 분해 짝 차와 그 적분.** $e_{\rm OU}$, $\varepsilon_b$, 사건항은 서로 독립·영평균이라 교차항이 사라져
$$
\delta(t)\equiv E\!\left[e_b(t)^2-e_m(t)^2\right]=-\,c_b+p_{\rm obs}\,r\,A^2\sum_k g(t-t_k)^2 ,
$$
$$
\boxed{\;D=\frac1T\int_0^T\delta(t)\,dt=p_{\rm obs}\,r\,\frac{A^2}{T}\sum_{k=1}^{N_e}\int_0^\infty g(s)^2ds-c_b
=p_{\rm obs}\,r\,\frac{N_e\,\kappa\,A^2}{n}-c_b,\qquad
\kappa\equiv\frac1\Delta\int_0^\infty g(s)^2ds=1+\frac{(1-q)^2q^2\,\tau_{\rm rec}}{2\Delta}\;}
$$
이것이 §1의 $D=p_{\rm obs}rE_{\rm imp}-c_b$다. 숫자: $T_i$의 $\tau_{\rm rec}=103$–$140$ ms, $\Delta=10$ ms에서 $\kappa=1.036$–$1.029$(이산 합 1.040–1.031) — **점프 에너지의 97 % 이상이 사건 직후 한 행에 있다.** 그래서 "회수"는 사건 직후 행에서 결정되고, 이력 필터의 50 ms 기억(foundations §7.6)은 회복 구간의 3 % 몫만 다룬다.

**$\delta(t)$의 부호가 시간에 따라 바뀌는 자리.** 사건 전에는 $\delta=-c_b<0$(모델이 항상 조금 진다); 사건 직후 행에서 $\delta=p_{\rm obs}rA^2-c_b$; 그 뒤 $\delta(s)=p_{\rm obs}rA^2(1-q)^2e^{-2s/\tau_{\rm rec}}-c_b$는
$$
s^\ast=\frac{\tau_{\rm rec}}{2}\ln\frac{p_{\rm obs}\,r\,A^2(1-q)^2}{c_b}
$$
에서 다시 음으로 떨어진다. 즉 모델의 이득은 각 사건 뒤 유한한 창 $[0,s^\ast]$에만 살고 그 밖의 모든 $t$에서 $-c_b$를 낸다. 방전 전체의 $D$ 부호는 창들의 합 $p_{\rm obs}rN_e\kappa A^2$가 창 밖의 누적 비용 $nc_b$를 넘는지로 정해진다 — §1(b)의 부등식을 시간 적분으로 읽은 것이고, §2의 "충격행 5 %가 $T_i$ 평균 $D$의 64 %를 운반한다"는 이 적분의 측정이다.

**$V_\varphi$의 시간 식이 다른 점(따름정리).** §12.2의 운동량 식은 토크 계단 $\Delta\mathcal T$에 대해 점프가 아니라 **램프** 응답을 준다:
$$
V_\varphi(t)=V_0+\frac{\Delta\mathcal T\,\tau_\varphi}{nm\langle R^2\rangle/R}\Bigl(1-e^{-(t-t_k)/\tau_\varphi}\Bigr),\qquad
h_V(s)=\Theta(s)\bigl(1-e^{-s/\tau_\varphi}\bigr).
$$
이력 필터는 램프를 한 걸음 뒤에서 따라가므로 놓치는 양은 걸음당 $g_V(s)\approx(\Delta/\tau_\varphi)e^{-s/\tau_\varphi}$이고
$$
\kappa_V=\frac1\Delta\int_0^\infty g_V(s)^2ds=\frac{\Delta}{2\tau_\varphi}=0.056\quad(\tau_\varphi=89\text{ ms; 이산 합 }0.056).
$$
같은 진폭 $A$에 대해 회수 가능한 에너지가 $T_i$ 계단의 $\kappa_T\approx1.03$ 대비 **18분의 1**이다. 즉 $V_\varphi$는 (i) $p_{\rm obs}$가 우연 수준이고 (ii) 크기 $A$가 안 보여 $r$이 작으며 (iii) 설령 둘 다 있어도 느린 램프라 $\kappa_V\ll1$ — 세 인자가 곱으로 $p\,r\,\kappa$를 누른다. $T_i$의 ELM 붕괴는 ms 이하의 계단이라 $\kappa_T\approx1$이다. (iii)은 유도이지 측정이 아니며, ELM에 동반하는 회전의 계단 성분은 이 따름정리 밖이다.

#### 2. 데이터 검산 (네 정준 run 풀링, 학습·채점 없음)

`data/.b1_seqv2_s{42,1,7,123}_i{같은 seed}/comparison_errors_test.npz`를 풀링하고, 방전 단위는 지시대로 **(split, shot) 쌍**으로 잡았다. 단 npz의 `shot` 키는 §8an 부록이 검증한 대로 `sorted(data/s*.csv)`의 전역 인덱스라 물리 shot과 1:1이며, 두 split의 test에 모두 든 shot은 이중 계수된다($T_i$ 81쌍, $V_\varphi$ 47쌍). 물리 shot 평균 승률도 함께 적는다. 행 20개 미만 방전 제외(§8an 규칙). 충격행 = 방전 내 $|\Delta y|$ 상위 5 % 행(§1′의 "사건 직후 행"의 데이터 대리). $p_{\rm obs}$ 대리 = 충격행 중 같은 두 CES 행 사이의 per-shot z-score ECEI 변화 $\max_c|\Delta z_c|>1$인 비율(CSV에서 직접; npz `y_true`→CSV 행 매칭 100 %, 샷 내 행 순서 단조 확인). 기준율 = 전체 행에서 같은 비율. 단위: $T_i$ eV, $V_\varphi$ km/s. 네 정준 run의 데이터 처리는 확정 프로토콜(W = 2, held-free, per-file cap 500, `CES_TI_SPIKE_CUT_EV=3000`)이다.

| 양 | $T_i$ | $V_\varphi$ |
|---|---:|---:|
| 방전 쌍 수 / 물리 shot 수 | 367 / 286 | 219 / 172 |
| 승률 vs 인과 GP (쌍 / 물리 평균) · §8al 값 | 0.714 / 0.696 · 0.695 | 0.466 / 0.455 · 0.481 |
| 승률 vs PCHIP · 중앙 skill vs GP | 0.567 · +0.093 | 0.306 · −0.012 |
| **(a) 산포 삼분위 승률** 조용/중간/변동 · §8an 값 | **0.431 / 0.852 / 0.861** · 42/83/85 | **0.356 / 0.452 / 0.589** · 34/48/55 |
| 삼분위 중앙 skill 조용 / 변동 | **−0.072** / +0.131 | **−0.073** / +0.032 |
| corr(산포, 승) · 순열 p | +0.335 · 0.0001 | +0.197 · 0.0013 |
| logit $\ln$(산포) 계수 (SE) | +1.75 (0.22) | +0.68 (0.18) |
| 천장: 산포 상위 10 % / 20 % 승률 | 0.811 / 0.811 | 0.682 / 0.659 |
| Theil–Sen $D$ vs $\mathrm{Var}(y)$: 절편 / 기울기 / Spearman | −684 eV² / +0.063 / +0.584 | −4.8 (km/s)² / +0.020 / +0.210 |
| 조용 삼분위 $D<0$ 비율 · 중앙 $D$ | 0.569 · −140 eV² | 0.644 · −1.8 (km/s)² |
| 변동 삼분위 $D<0$ 비율 | 0.139 | 0.411 |
| **(b)** 충격 비중 삼분위 승률 · corr(충격비중, 승) p | 0.683/0.721/0.738 · +0.079 (0.13) | 0.466/0.493/0.438 · +0.022 (0.75) |
| $D$ 분해 평균(중앙): 충격행 / 본류행 | +4367 (+1038) / +2428 (+411) | −1.5 (−0.7) / +194 (+0.5) |
| 충격행 승률 / 본류행 승률 | **0.744 / 0.594** | **0.425 / 0.530** |
| **(c)** peak 행 승률 / 비peak 행 승률 | 0.692 / 0.613 | 0.484 / 0.475 |
| peak 행 SSE 비중 model/GP 중앙 | 0.829 | 0.983 |
| $p_{\rm obs}$ 대리(ECEI): 충격행 / 전체행 / 농축 | **0.427 / 0.336 / 1.27** | **0.346 / 0.351 / 0.99** |
| 쌍 차(충격−전체) 평균 [부트스트랩 95 %] | +0.091 [+0.073, +0.110] | −0.004 [−0.023, +0.015] |
| ECEI∪BES: 충격행 / 전체행 · 쌍 차 · 방전 +/− | 0.625 / 0.467 · +0.158 [+0.140, +0.177] · 305/62 | 0.483 / 0.478 · +0.006 [−0.015, +0.026] · 105/106 |
| 방전 내 Spearman($|\Delta y|,|\Delta z_{\rm ECEI}|$) 평균 · >0 비율 | +0.117 · 0.790 | −0.016 · 0.534 |
| 변동 삼분위: $p_{\rm obs}$ 중앙 이상 / 이하 승률 · corr p | **0.934 / 0.787** · +0.250 (0.0044) | 0.531 / 0.634 · −0.154 (0.19) |
| logit win ~ $\ln$산포 + $p_{\rm obs}$: $p_{\rm obs}$ 계수 (SE) | **+1.92 (0.53)** | −0.55 (0.44) |

읽기.
1. **(a)가 재현된다.** 삼분위 승률 43/85/86 vs 36/45/59는 §8an의 42/83/85, 34/48/55와 1–4 %p 안(§8an은 v2r63 run). $D$의 Theil–Sen 절편은 두 타깃 모두 음, 기울기는 양 — 식 $D=p\,r\,E_{\rm imp}-c_b$의 부호 구조 그대로다. 조용 삼분위 중앙 skill이 **−0.072 vs −0.073으로 같다**: 본류 비용 $c_b$가 구동원과 무관하다는 부수 예측의 확인이다.
2. **(b) 충격 "비중"은 승패를 가르지 않는다**(두 타깃 모두 null). 가르는 것은 충격의 *회수*다: $T_i$는 행의 5 %인 충격행이 평균 $D$의 64 %, 중앙 기여의 72 %를 운반하고 충격행 승률(0.744)이 본류행(0.594)보다 높다 — §1′의 "$\kappa\approx1$, 이득은 사건 직후 창에 산다"의 측정이다. $V_\varphi$는 **충격행에서 진다**(0.425 < 0.530, 충격행 $D$ 중앙 −0.7): 사건 시점이 보여도 크기를 모르면 $r\le0$이라는 §1(ii)의 진술이다.
3. **(c) $p_{\rm obs}$의 직접 측정이 비대칭의 나머지 절반이다.** $T_i$의 큰 변화는 ECEI 급변과 **1.27배(ECEI∪BES 1.34배) 농축**되어 동반하고 방전 305/62로 한쪽이다. $V_\varphi$의 큰 변화는 빠른 채널 급변과 **우연 수준(0.99배, 105/106)**으로만 겹친다 — "회전의 구동이 데이터에 없다"는 §12.2의 문장을 사건 수준에서 수치화한 것이다. 이 변수는 $T_i$에서 산포를 통제한 뒤에도 승을 가른다(logit +1.92 ± 0.53; 변동 삼분위 0.934 vs 0.787)지만 $V_\varphi$에선 가르지 않는다(−0.55 ± 0.44). 천장은 $T_i$ 0.81–0.86(변동 삼분위 안에서 $p_{\rm obs}$ 상위 반은 0.934), $V_\varphi$ 0.59–0.68 — 유도의 $\Pi_{T_i}\approx1$, $\Pi_{V_\varphi}\approx0.5$–$0.6$ 범위다.

#### 3. 기록과의 연결

- §8al §4의 0.695 / 0.481과 §8an의 삼분위 표를 같은 아티팩트에서 재현했고, §8an 평결 3이 "절반의 이유"라고 남긴 잔여("변동해도 $V_\varphi$는 겨우 이긴다")를 $p_{\rm obs}\approx$ 우연 + 충격행 패배 + 램프 응답의 $\kappa_V\ll1$로 메웠다.
- §8am이 peak 분율에서 **부호가 거꾸로** 나온 이유: `is_peak`는 CES 이웃 기울기·분산으로 정의된 *타깃 활동* 플래그지 빠른 채널의 *가시성*이 아니다. 같은 자료에서 peak 행은 $T_i$ 승률을 0.692/0.613으로 약하게만 가르고($V_\varphi$ 0.484/0.475), 빠른 채널 가시성 대리는 산포 너머로 강하게 가른다 — 변수가 다르다.
- plasma §12.3의 "국소 결합 $Q_{ei}$ = 빠른 진단이 보는 양 / $V_\varphi$는 없음"과 foundations §7.7의 "보간을 이기려면 외생 정보가 차이 이상을 운반해야 한다"가 이 부등식의 두 변이다. §8ax의 "꼬리에서만 이긴다"는 $D$ 분해(충격행 64 %)와 같은 말이고, foundations §7.6의 50 ms 기억은 §1′에서 회복 구간의 3 % 몫에 해당한다.

#### 4. 한계

1. (split, shot) 쌍은 이중 계수한다(81/47쌍). 물리 shot 평균 승률(0.696 / 0.455)도 같은 결론이다.
2. $p_{\rm obs}$ 대리는 10 ms 격자에서 per-shot z-score 최대 채널 변화 > 1이라는 거친 문턱이고 사전등록이 아니다(탐색적). $T_i$의 농축 1.27–1.34는 유의하지만 완만하다 — $D_\alpha$/ELM 지시자(plasma §12.4 항목 5)가 있으면 날카로워질 양이다.
3. $c_b\approx\sigma_e^2d/n_{\rm eff}$는 선형 추정량의 형식이고 LSTM에는 부호만 유도된다; 크기는 측정값(조용 삼분위 −0.07)이다. $\kappa$의 수치는 $\tau_{\rm rec}$, $\tau_\varphi$의 데이터 적합값에 의존하며 계단/램프 형태는 가정이다.
4. $V_\varphi$에서 $p_{\rm obs}$, $r$, $\kappa_V$는 분리되지 않는다 — 측정되는 것은 곱이다. 충격행 패배(0.425)는 $r\le0$과 양립하지만 단독 증명은 아니다.
5. 상관이지 인과가 아니다: 가시성 대리는 "ELMy H-mode 체제"의 대리일 수 있다. **뒤집을 측정**: B.6의 kHz Mirnov 모드 주파수(또는 NBI/RMP 파형)를 입력으로 넣으면 (i) $V_\varphi$ 충격행의 $p_{\rm obs}$ 농축이 1을 넘고, (ii) $V_\varphi$ 충격행 승률이 본류행 승률(0.530)을 **역전**하며, (iii) §8an의 예측대로 변동 삼분위 승률이 먼저 오른다. 셋 중 하나라도 안 일어나면 이 기제는 틀렸다.

**스크립트·출력.** `scratchpad/p08\p08_winrate_asymmetry.py` → 같은 폴더의 `p08_results.json`(방전별 전체 표 포함). 입력: 위 네 npz + `data/s*.csv`(641). 쌍 차 CI는 같은 JSON에서 부트스트랩(5000)으로, §1′의 $\kappa$ 상수는 닫힌 식과 이산 합을 따로 계산해 대조했다. 추적 파일 수정 없음, 모델 학습·채점 없음.


### P09 — 장치 간 사전학습(MAST → KSTAR)은 CES 갭 메우기 지식을 옮길 수 없다: 무차원 유사성과 계기 단위 논증

**주장.** 두 장치는 수송을 지배하는 무차원 변수 $(\rho_*,\nu_*,\beta,q,\varepsilon,\kappa,T_e/T_i,M)$에서 다른 자리에 있고, 더 결정적으로 진단은 서로 다른 계기이며 모델의 입력은 계기 단위다. 따라서 옮겨질 수 있는 것은 "방정식의 구조"뿐인데, 그 구조는 파라미터 6개짜리 생성모델(§1.5의 식 (1))로 요약되고 KSTAR 라벨 몇 백~몇 천 행이면 10 % 정밀도로 추정된다. 1.8 M 가중치의 사전학습이 더할 것이 남지 않는다. §8az의 `tm_pre` = `tm_scratch`(모든 셀에서 0.004–0.013)는 이 유도의 예측이다.

#### 1. 유도

**(a) 무차원 유사성(Kadomtsev 1975, Connor–Taylor 1977).** Vlasov–Fokker–Planck–Maxwell 계를 $(n,T,B,L)$로 무차원화하면 독립 무차원수는 정확히 셋이다:
$$\rho_*=\frac{\rho_i}{a},\qquad \nu_*=\frac{\nu_{ii}\,qR}{\varepsilon^{3/2}v_{th,i}},\qquad \beta=\frac{2\mu_0 p}{B^2},$$
여기에 기하·형상 $(q,\varepsilon=a/R,\kappa)$와 비 $(T_e/T_i, M)$가 더해져 $\chi/\chi_B=F(\rho_*,\nu_*,\beta,q,\varepsilon,\kappa,T_e/T_i,M)$이다(이론 노트 §11.2). pedestal 가정값(KSTAR: $T_i=600$ eV, $n=3\times10^{19}$, $B=1.8$ T, $R=1.8$, $a=0.5$, $\kappa=1.8$, $q_{\rm ped}=4$ — 이론 노트 §1.1·§11.2의 $\rho_*=0.0055$, $\nu_*=0.24$, $\beta=0.45$ %를 재현; MAST: $T=200$–300 eV, $n=3$–$5\times10^{19}$, $B_T=0.5$ T, $R=0.85$ m, $a=0.65$ m, $\kappa=2$, $q_{\rm ped}=6$ 가정):

| 양 | KSTAR | MAST (T 250 eV, n 4e19) | MAST 범위(T 200–300, n 3–5e19) | MAST/KSTAR |
|---|---:|---:|---:|---:|
| $\varepsilon$ | 0.278 | 0.765 | — | 2.75 |
| $\kappa$ | 1.8 | 2 | — | 1.11 |
| $q_{\rm ped}$ | 4 | 6 | — | 1.5 |
| $\rho_i$ [mm] | 2.78 | 6.46 | 5.8–7.1 | 2.3 |
| $\rho_*$ | 0.0056 | 0.0099 | 0.0089–0.0109 | **1.6–2.0** |
| $\nu_*$ | 0.242 | 0.288 | 0.15–0.56 | 0.6–2.3 |
| $\beta$ [%] | 0.45 | 3.22 | 2.9–3.2 | **6.5–7.2** |
| 포획 분율 $\sqrt{2\varepsilon/(1+\varepsilon)}$ | 0.66 | 0.93 | — | 1.4 |
| $\chi_{gB}$ [m²/s] | 0.116 | 0.311 | 0.22–0.41 | 2.7 |
| IPB98 $\tau_E$ [ms] | 76.7 | 48.4 | — | 0.63 |
| 10 ms AR(1) $\rho=e^{-\Delta/\tau_E}$ | 0.878 | 0.813 | — | −0.064 |

읽기. (i) $\beta$가 7배, $\varepsilon$이 2.75배, 포획 분율이 0.93이라 MAST pedestal은 전자기 난류 영역이다: 문헌이 구형 토카막 pedestal에 지목하는 미세 찢김 모드(MTM)·ETG·KBM이 지배하고, 바나나 영역 식 $\varepsilon^{3/2}$ 자체가 $\varepsilon=0.76$에서 유효성 경계에 선다. KSTAR 쪽은 이론 노트 §5.2의 ITG/TEM 정전 난류와 §5.5의 $E\times B$ 억제 조건이 사는 자리다. (ii) $\rho_*$는 1.6–2.0배(기대 2–4배보다 작다: $B$가 3.6배 낮지만 $T$가 2–3배 낮고 $a$가 1.3배 커서 상쇄). (iii) $\nu_*$는 0.6–2.3배로 **판별력이 약하다** — 구형 토카막의 큰 $\varepsilon^{3/2}$가 낮은 pedestal 온도를 상쇄한다(EPS 2012 O4.111의 MAST pedestal $\nu^*_{e}=0.6$–1.4와 같은 자릿수). 유사성 논증의 결정 축은 $\nu_*$가 아니라 $\beta$·$\varepsilon$·$\kappa$·$q$다. (iv) pedestal 구조: EPED 폭 $\Delta_\psi\approx0.076\sqrt{\beta_{p,\rm ped}}$(이론 노트 §6.2)가 $\beta$에 묶이므로 폭·높이의 사다리가 다르고, 본 데이터셋의 RMP ELM 억제는 $n=1,2$(§6.4)인 반면 MAST의 RMP 실험은 $n=4,6$이다(arXiv 1305.3061). (v) 전역 구속: IPB98로 $\tau_E$가 77 vs 48 ms이므로 **10 ms 격자의 AR(1) 계수부터 다르다**(0.878 vs 0.813). "느린 완화 위의 점프"라는 구조는 같지만 그 숫자는 다르다.

**(b) 계기 논증.** 시퀀스 모델이 배우는 것은 사상
$$f_\theta:\ \mathbb R^{W\times 32}\to\mathbb R^2,\qquad x=\big(\text{BES}_{1..9},\text{ECEI}_{1..4},\text{MC}_{1..2},\Delta t,\ \text{CES 이력}\big)$$
이며 입력은 **학습 파일 통계로 z-score된 계기 단위**다. `b12_models.py`의 `TokaMindArm`은 토큰을 $t_g=P_g x_g+e^{\rm group}_g+e^{\rm pos}+e^{\rm mod}+e^{\rm role}$로 만들고 backbone(d 192, 4층, 6 헤드, FF 768)을 통과시킨다. MAST에서 오는 것은 backbone과 $e^{\rm pos},e^{\rm mod},e^{\rm role}$뿐이고, $P_g$와 $e^{\rm group}_g$는 **새로 무작위 초기화**된다(PREREGISTRATION_B12.md §4.5: "옮겨 오는 것은 토큰들 사이의 관계를 읽는 법뿐"). 사전학습 backbone이 본 분포는 MAST 신호의 DCT3D 토큰이고, KSTAR 토큰은 무작위 선형 사상 $P_g$의 상이므로 backbone의 입력 분포는 사전학습 분포와 무관하다. MAST에는 같은 BES/ECEI/Mirnov 채널 집합이 없으므로(TokaMind는 자기·프로파일·구동기 모달, arXiv 2602.15084), 옮겨질 수 있는 유일한 사전지식은 계기와 무관한 **시간축 구조**("매끄러운 완화 + 충격")뿐이고, 그것은 §1.5 식 (1)의 6개 수치다. 가중치 수 분해(구조 상수로 계산, §8az의 1,823,490과 정확히 일치):

| 부분 | 파라미터 | 출처 |
|---|---:|---|
| backbone(4층) | 1,779,456 | MAST 사전학습 |
| pos/mod/role 임베딩 | 10,944 | MAST 사전학습 |
| 신호별 투영 $P_g$ | 7,296 | **새로(KSTAR 계기 단위)** |
| group 임베딩 | 960 | 새로 |
| 두 헤드 | 24,834 | 새로 |
| 합계 | 1,823,490 | = §8az |

사전학습분은 98.2 %, 계기를 토큰으로 바꾸는 사상은 0.4 %다. 사전지식이 들어갈 수 있는 98.2 %가 설명할 구조는 6개 수치이고, 데이터로 정해야 하는 것은 계기 사상 쪽이다. **데이터가 제한하는 양은 prior가 아니라 계기 사상**이며, 그것은 MAST에서 올 수 없다.

**(c) 그 6개 수치는 KSTAR 라벨 몇 행이면 정해지는가.** `data/.foundations.json`의 robust OU 적합: $\rho_{1}=0.8841$, $\tau=81.2$ ms, $\sigma_s=68.7$ eV, $\sigma_n=39.0$ eV. AR(1) 계수의 점근 표준오차 $\mathrm{SE}(\hat\rho)=\sqrt{(1-\rho^2)/n}$에서 10 % 상대 정밀도에 필요한 행 수:
$$n_\rho=\frac{1-\rho^2}{(0.1\rho)^2}=28,\qquad n_\tau=\frac{1-\rho^2}{(0.1\,\rho|\ln\rho|)^2}=1{,}841,\qquad n_{\sigma_s^2}=\frac{2}{0.1^2}\frac{1+\rho}{1-\rho}=3{,}252.$$
Monte Carlo(AR(1), $\rho=0.884$, 2,000회)로 공식을 확인: $n=300$에서 MC 0.029 vs 공식 0.027, $n=1{,}800$에서 0.0113 vs 0.0110, $n=3{,}000$에서 0.0088 vs 0.0085($n=30$은 MC 0.108 vs 0.085, 소표본 편향 −0.048 — "28행"은 하한이고 실용 수치는 몇 백 행). 641 파일의 관측 `CES_TI` 행은 파일당 중앙값 295(합계 226,991), `CES_VT`는 265(합계 188,100)다. 따라서 $\rho$는 **한 shot의 10분의 1**, $\tau$는 6 shot, $\sigma_s^2$는 11 shot이면 10 %로 정해지고, 전체 데이터로는 $\mathrm{SE}(\hat\rho)=0.00098$이다. MAST에서 가져온 $\rho$ prior(IPB98 비로 0.813)는 KSTAR 값과 0.064 차이 = **KSTAR 추정의 66 SE**만큼 틀린 prior다. 이것이 §8az "context carries the shot"의 산술이다.

#### 1.5 시간 $t$의 식 — 이 증명의 시간 영역 형태

**(1) 옮겨질 수 있는 유일한 대상: 생성 방정식.** pedestal 한 자속면의 CES 타깃 $y(t)$(이온 온도 또는 회전)는 이온 에너지/각운동량 보존(이론 노트 §8·§7)을 선형화한 1차 완화 + 확률 구동 + ELM 충격이다:
$$\frac{dy}{dt}=-\frac{y(t)-m}{\tau}+\sqrt{\frac{2\sigma_s^2}{\tau}}\,\xi(t)-\sum_k A_k\,\big(h*\delta_{t_k}\big)(t),\qquad h(t)=\Theta(t)\Big[\delta(t)-\frac{1}{\tau_b}e^{-t/\tau_b}\Big],\tag{1}$$
$\langle\xi(t)\xi(t')\rangle=\delta(t-t')$, $t_k$는 비율 $\lambda$의 점과정. 해의 형태:
$$y(t)=m+\int_{-\infty}^{t}e^{-(t-s)/\tau}\sqrt{\tfrac{2\sigma_s^2}{\tau}}\,\xi(s)\,ds-\sum_k A_k\,\Theta(t-t_k)\,e^{-(t-t_k)/\tau_b},\tag{2}$$
즉 $t_k$에서 $A_k$만큼 급락한 뒤 $\tau_b$로 회복하는 톱니(이론 노트 §6.3: 붕괴 0.1–1 ms, 회복 $\tau_b\sim10$–50 ms)가 OU 완화 위에 얹힌다. 관측은 격자 $t_j=j\Delta$에서
$$z_j=y(t_j)+\epsilon_j,\qquad \epsilon_j\sim\mathcal N(0,\sigma_n^2),\qquad \sigma_n=T_i\sqrt{2/N_{\rm ph}}\ \text{(CRB, foundations §A)},\tag{3}$$
이고 (1)을 $\Delta$로 이산화하면 모델이 실제로 보는 AR(1)이다:
$$y_{j+1}-m=\rho\,(y_j-m)+\eta_j,\qquad \rho=e^{-\Delta/\tau},\qquad \mathrm{Var}\,\eta_j=\sigma_s^2(1-\rho^2).\tag{4}$$
식 (1)의 **형태**는 장치와 무관하다. 그 형태가 사전학습이 가져올 수 있는 전부이며, 파라미터는 $(m,\tau,\sigma_s,\sigma_n,\lambda,A,\tau_b)$ 7개(§1(b)의 6개 + 회복 시간 $\tau_b$)다.

**(2) 어느 파라미터가 장치 고유인가 — 무차원수의 함수로.**

| 파라미터 | 물리 | 무차원수 의존 | KSTAR vs MAST |
|---|---|---|---|
| $\tau$ | $\tau_E\parallel\tau_{\rm eq}$ (이론 노트 §11.1, §8) | IPB98 무차원형 $\Omega_i\tau_E\propto\rho_*^{-2.70}\beta^{-0.90}\nu_*^{-0.01}q^{-3.0}\varepsilon^{0.73}\kappa^{2.3}M^{0.96}$ (ITER Physics Basis 1999, 2장; 문헌) | 77 vs 48 ms ⇒ $\rho(\Delta=10\,\mathrm{ms})$ 0.878 vs 0.813 |
| $m$ | pedestal 높이 $T_{i,\rm ped}$, $V_{\varphi}$ | EPED: $\beta_{p,\rm ped}$·PB 한계·$\nu_*$ (§6.2); 회전은 NBI 토크/NTV(§7) | 600 vs 250 eV, 계기 단위 |
| $\sigma_s$ | 수송 요동 진폭 | $\chi_{gB}\propto\rho_*$ (§5.3) × $m$ | $\chi_{gB}$ 2.7배, $m$ 0.4배 |
| $A_k$ | ELM당 pedestal 에너지 손실 $\Delta W_{\rm ELM}/W_{\rm ped}=5$–20 % (§6.3) | $\nu_*$가 오르면 감소(Loarte 2003 스케일링; 문헌), RMP 억제 시 $A_k\to$ 작은 사건 | $A_k\approx(0.05$–$0.2)\,m\,g(\nu_*)$ |
| $\lambda$ | $f_{\rm ELM}\approx P_{\rm loss}/\Delta W_{\rm ELM}\sim10$–100 Hz (§6.3) | $P_{\rm loss}$, $A_k$의 함수 | 억제 체제($n=1,2$ RMP) vs MAST $n=4,6$ |
| $\tau_b$ | pedestal 재건 = 수송 시간 | $\propto a^2/\chi$, $\chi/\chi_B=F(\rho_*,\ldots)$ (§11.3) | 10–50 ms; 1–5 격자 단계, 95 % 회복 30–150 ms |
| $\sigma_n$ | 계기: 광자 수 $N_{\rm ph}$, CVI 529 nm 선 | **플라즈마 무관**, 분광기 고유 | $N_{\rm ph}=326\Rightarrow46$ eV(적합 39 eV); MAST CXRS는 다른 계기 |
| $\Delta$ | CES 격자 | **계기 고유** | 10 ms; MAST 주기는 미확인 |

형태를 제외한 **모든** 파라미터가 장치 고유다. 그중 $\tau$·$A_k$·$\tau_b$는 §1(a) 표의 무차원수가 다른 점에서 정해지고($\beta$ 7배, $\varepsilon$ 2.75배), $\sigma_n$·$\Delta$·$m$의 단위는 계기가 정한다. 사전학습이 (1)의 형태를 안다고 해도 KSTAR 값은 하나도 모른다.

**(3) 표본 수를 기록 길이 $T$의 함수로.** $n=T/\Delta$를 (c)에 넣으면
$$\mathrm{SE}(\hat\rho)=\sqrt{\frac{(1-\rho^2)\,\Delta}{T}},\qquad
\frac{\mathrm{SE}(\hat\tau)}{\tau}=\frac{\mathrm{SE}(\hat\rho)}{\rho\,|\ln\rho|},\qquad
\frac{\mathrm{SE}(\hat\sigma_s^2)}{\sigma_s^2}=\sqrt{\frac{2\Delta}{T}\,\frac{1+\rho}{1-\rho}}.\tag{5}$$
$\Delta\ll\tau$ 극한에서 $1-\rho^2\approx2\Delta/\tau$, $\rho|\ln\rho|\approx\Delta/\tau$이므로
$$\frac{\mathrm{SE}(\hat\tau)}{\tau}\approx\sqrt{\frac{2\tau}{T}},\qquad \frac{\mathrm{SE}(\hat\sigma_s^2)}{\sigma_s^2}\approx\sqrt{\frac{4\tau}{T}}:\tag{6}$$
정밀도는 격자가 아니라 **기록이 상관 시간 $\tau$를 몇 번 담는가**($T/\tau$)로 정해진다. 10 % 정밀도에 필요한 것은 $T\approx200\tau$다. 스크립트($\rho=0.884$, $\Delta=10$ ms, $\tau=81$ ms): $T_{10\%}(\rho)=0.28$ s, $T_{10\%}(\tau)=18.4$ s(=227 $\tau$, (6)의 $200\tau=16$ s와 일치), $T_{10\%}(\sigma_s^2)=32.5$ s. 641 shot의 `CES_TI` 라벨 구간은 중앙값 **3.98 s**(p10 1.31, p90 7.47; 라벨 행 × $\Delta$ = 2.97 s), 전체 2,270 s. 한 중앙 shot만으로 $\mathrm{SE}(\hat\rho)=0.027$(3.1 %), $\mathrm{SE}(\hat\tau)/\tau\approx\sqrt{2\cdot81/3980}=20$ %이고, 전체 기록으로 $\mathrm{SE}(\hat\rho)=0.00098$이다. MAST 유래 $\rho$ prior의 편향 0.064는 **한 shot 추정의 2.4 SE, 전체 기록의 66 SE**다 — 첫 shot을 지나면 KSTAR 우도가 이미 prior를 압도하고, 그 뒤 prior는 정보가 아니라 편향이다. 이것이 §8az k-곡선(같은 shot 라벨 3→10에서 네 모델군이 똑같이 +0.26 상승, k=0에서 어느 팔도 평균을 못 이김)의 시간 영역 설명이다.

#### 2. 검산

§8az 라벨 희소화 표(컷 모집단, skill vs PCHIP)와 스크립트 계산:

| k | `bilstm` | `np` | `tm_pre` | `tm_scratch` | GP |
|---:|---|---|---|---|---|
| 0 | −0.025 | −0.040 | −0.016 | −0.003 | 0 |
| 3 | +0.146 | +0.148 | +0.132 | +0.137 | +0.127 |
| 10 | +0.407 | +0.403 | +0.392 | +0.398 | +0.079 |
| 상승 3→10 | +0.261 | +0.255 | +0.260 | +0.261 | −0.048 |

3→10 상승의 네 모델군 퍼짐은 **0.006**, `tm_pre`−`tm_scratch`는 k=0/3/10에서 −0.013/−0.005/−0.006, 축 1의 $|\delta_{\rm pre}-\delta_{\rm scratch}|$는 컷/포함 `T_i` 0.004/0.007, `V_rot` 0.009/0.013 — §8az 본문의 "0.004–0.013"과 일치하며 모두 ±0.013 이내(`within_tol = True`). 0→3 상승은 퍼짐 0.048로 조금 더 벌어지지만 그 차이는 `np`가 k=0에서 최하(−0.040)인 데서 오고 사전학습 유무와는 무관하다(`tm_pre` +0.148 vs `tm_scratch` +0.140). 무차원 표는 이론 노트 §11.2의 KSTAR 값 셋($\rho_*$ 0.0055, $\nu_*$ 0.24, $\beta$ 0.45 %)과 §11.1의 $\tau_E$ 77 ms를 재현했고, 파라미터 합계는 §8az의 1,823,490과 정확히 일치했으며, SE 공식 (5)는 Monte Carlo와 $n\ge300$에서 7 % 이내로 맞았다.

#### 3. 기록과의 연결

- §8az H-tm: "사전학습과 무작위 backbone이 모든 셀에서 0.004–0.013으로 일치, TokaMind 자체의 프로파일 그룹 결과 −0.005와 같은 부호·크기." 본 유도는 그 null의 이유를 둘로 분해한다 — 무차원 점이 다르고(1a), 계기 사상 0.4 %가 어차피 KSTAR에서 새로 학습된다(1b) — 그리고 (1.5)는 그 사이에 남는 "형태"가 한 줄의 식이며 한 shot이면 그 계수를 정함을 보인다.
- 이론 노트 §11.2 마지막 문장("B.12의 MAST 전이 실패는 모델이 이 변수들로 조건화되지 않았다는 쪽에서 읽어야 한다")을 표로 뒷받침한다: 조건화했더라도 $\beta$ 7배·$\varepsilon$ 2.75배 떨어진 점에서 외삽이 된다.
- §8au: `V_rot` 분산의 28 %가 shot 간 오프셋이고 42 % shot이 라벨 ≤ 20개다. 그 오프셋은 식 (1)의 $m$이고, (c)·(1.5)(3)은 그것이 몇 행이면 정해지는 양임을 보이며, §8az k-곡선이 그것을 측정했다. 남는 것은 §8az가 지목한 운전 변수(NBI 토크, $I_p$, $B_T$)이지 다른 장치의 prior가 아니다.
- §8az 축 6·§8ag: 7배 파라미터의 Transformer가 `T_i`에서 BiLSTM보다 열세(컷 −0.09) — 가중치가 아니라 데이터 제한이라는 (b)의 읽기와 같다.
- §8ba의 OU 유도(한 스텝 RMSE 비 0.81/0.88/0.91, 50 ms 포화)는 식 (1)–(4)의 KSTAR 계수로 나온 결과이며, 그 계수는 본 데이터가 정했다.

#### 4. 한계

- MAST는 돌리지 않았다. MAST pedestal 값(T 200–300 eV, n 3–5e19, $q_{\rm ped}=6$, $\kappa=2$)은 가정이고, 기하는 MAST-U 공개 설계치 $R=0.85$ m, $a=0.65$ m(원 MAST는 0.9/0.6 m, $\varepsilon$ 0.67–0.76으로 결론 불변)이며 $B_T=0.5$ T는 원 MAST 값이다(MAST-U는 0.585–0.85 T). $\nu_*$ 비는 0.6–2.3으로 가정 범위에 민감하므로 판별 축으로 쓰지 않았다.
- 사전학습 가중치의 출처는 §8az·PREREGISTRATION_B12.md §9가 인용한 TokaMind(`tokamind-base-v2`, UKAEA·IBM·STFC, arXiv 2602.15084, `backbone.pt`·`token_encoder.pt` SHA-256 고정)이며, 그 토큰 시간 간격 $\Delta_{\rm MAST}$가 10 ms와 같다는 보장은 없다 — 전이된 위치 임베딩조차 다른 시간 단위일 수 있다는 뜻이고, 이것도 null 쪽 근거다.
- IPB98 $\tau_E$와 OU $\tau$의 동일시, 그리고 표 (2)의 IPB98 무차원 지수와 Loarte ELM 스케일링은 문헌 인용이며 자릿수 수준이다(이론 노트 §11.1: 77 vs 81–140 ms). "66 SE"는 그 비교의 크기를 보이는 수치이지 정밀한 편향 추정이 아니다.
- 식 (1)은 선형화이고 $A_k$를 충격으로 둔 근사다. ELM 붕괴 자체(0.1–1 ms)는 10 ms 격자 아래라 한 점으로만 보이므로(§1.4), 격자 위에서는 (2)의 톱니 형태가 전부다.
- 이 증명이 닫는 것은 "같은 구조의 다른 장치 가중치"다. 같은 장치의 라벨 없는 행을 쓰는 `mae` 팔(§8au가 연 경로)은 계기 사상을 KSTAR에서 배우므로 여기서 배제되지 않는다 — 그것이 다음 측정이다.

**스크립트.** `scratchpad/p09/p09_check.py` → `p09_out.json` (§1–§5 출력). 입력: `docs/theory_plasma_physics.md` §1·§5·§6·§11 가정값, `data/.foundations.json` D.CES_TI.fits.robust, `ces_prediction/experiments/b12/b12_models.py` 구조 상수, `THESIS_RESULTS.md` §8az 표, 641 shot CSV의 `time`·`CES_TI`·`CES_VT` 열.

**출처(웹).** MAST-U 기하·자장·전류: [arXiv 2407.07784](https://arxiv.org/pdf/2407.07784), [Wikipedia MAST](https://en.wikipedia.org/wiki/Mega_Ampere_Spherical_Tokamak); TokaMind: [arXiv 2602.15084](https://arxiv.org/abs/2602.15084v1); MAST pedestal 충돌도 0.6–1.4: [EPS 2012 O4.111](https://lac913.epfl.ch/epsppd3/2012/pdf/O4.111.pdf); MAST RMP n=4,6: [arXiv 1305.3061](https://arxiv.org/pdf/1305.3061).


### P10. 실시간 요건은 플라즈마가 정한다 — 세 지연 등급과 나우캐스터의 자리

**명제.** CES 가상 센서의 실시간 요건은 기기가 아니라 플라즈마가 정한다. 10 ms CES 격자, ~1 ms ELM 붕괴, ~100 µs 모드 주기, ~100 ms 구속 시간은 제어 관점에서 서로 다른 세 지연 등급을 만들고, 측정된 추론 비용 1.5–9 ms는 나우캐스터를 그중 정확히 하나(프로파일·수송 제어 등급)에 놓는다.

**결론 한 줄.** 등급 A(수송 제어, $\tau_E\sim$ 80–140 ms)에는 측정된 모든 인과 팔이 들어가고, 등급 B(ELM 트리거, < 1 ms)와 C(MHD 모드·회전 피드백, 0.1–1 ms)는 추론 지연이 아니라 **라벨의 10 ms 적분**이 배제한다. 따라서 §8aj가 1 ms 판정을 보류했어도 등급 배정은 흔들리지 않는다. 단, "지연 1–10 ms의 skill 비용은 $\tau\approx80$ ms라 미미하다"는 기대는 **eV 단위에서만** 맞고 skill 단위에서는 틀리다(§1(a)).

스크립트: `scratchpad/p10\p10_latency_classes.py` (출력 `p10_output.txt`, 수치 `p10_results.json`, 같은 폴더). 입력: `data/.foundations.json`(D.CES_TI.fits), `data/.b9_op_counts.json`, `data/.b9_latency_min.json`, `data/.latency_benchmark.json`, `data/*.csv` 641파일. 추적 파일 변경 없음, 학습·채점 없음.

#### 1. 유도

##### 1(a) OU 모형에서 지연 $\ell$의 가격

foundations §7.1의 잡음 있는 OU: $\rho_\ell=e^{-\ell/\tau}$, $y_t-\bar m=\rho_\ell\,(y_{t-\ell}-\bar m)+\varepsilon_\ell$, $\mathrm{Var}\,\varepsilon_\ell=\sigma_s^2(1-\rho_\ell^2)$, 관측 $\tilde y=y+n$, $n\sim\mathcal N(0,\sigma_n^2)$. 시각 $t-\ell$까지의 정보로 상태를 사후분산 $P$(§7.4 Riccati 정상상태: $P^-=\rho^2P+\sigma_s^2(1-\rho^2)$, $K=P^-/(P^-+\sigma_n^2)$, $P=(1-K)P^-$)로 알 때, $t$의 최적 추정은 $\hat y_t=\bar m+\rho_\ell(\hat y_{t-\ell}-\bar m)$이고 오차는 서로 독립인 세 항 $\rho_\ell e_{t-\ell}+\varepsilon_\ell+n_t$의 합이다. 그러므로

$$
\mathrm{MSE}(\ell)=\rho_\ell^{2}P+\sigma_s^2\bigl(1-\rho_\ell^{2}\bigr)+\sigma_n^2,\qquad
\mathrm{skill}(\ell)=1-\frac{\mathrm{MSE}(\ell)}{\mathrm{MSE}_{\rm ref}},\quad
\mathrm{MSE}_{\rm pers}=2\sigma_n^2+2\sigma_s^2(1-\rho_\Delta).
$$

검산: $\ell=\Delta=10$ ms에서 §7.4의 $P^-+\sigma_n^2$와 정확히 같아야 하며, 스크립트는 `.foundations.json`의 `mse_filter_inf`와의 일치를 assert로 확인한다(두 적합 모두 통과). 소지연 전개:

$$
\frac{d\,\mathrm{MSE}}{d\ell}\Big|_{0}=\frac{2}{\tau}\bigl(\sigma_s^2-P\bigr),\qquad
\mathrm{MSE}(\ell)-\mathrm{MSE}(0)\;\approx\;\frac{\ell}{\Delta}\bigl[\mathrm{MSE}(\Delta)-\mathrm{MSE}(0)\bigr]\quad(\ell\ll\tau).
$$

즉 **지연 $\ell$은 한 격자 스텝이 잃는 것의 $\ell/\Delta$ 만큼을 잃는다.** $T_i$ 강건 적합($\sigma_n=39.0$ eV, $\sigma_s=68.7$ eV, $\tau=81.2$ ms, $P=791$ eV², $K=0.520$)에서 기울기는 96.7 eV²/ms, skill로 −0.0234/ms; 고전 적합($139.9/147.1$ eV, 74.5 ms, $P=6723$, $K=0.343$)에서 400.9 eV²/ms, −0.0090/ms. 왜 $\tau\approx80$ ms인데 skill 손실이 작지 않은가: 기준인 persistence MSE(4137 eV²)가 $2\sigma_n^2$(3044)과 같은 10 ms 혁신 $2\sigma_s^2(1-\rho)$(1093)으로 이루어져 있어, 분자의 $\sigma_s^2-P=3924$ eV²가 분모와 같은 자릿수이기 때문이다. RMSE로 보면 1.5 ms가 48.1 → 49.6 eV(1.5 eV, $\sigma_n$ 39 eV의 4 %)이므로 "eV로는 미미"가 맞다.

해석 보조: 측정된 백본 skill +0.358(foundations §8, s42 컷 1-step, persistence 기준)은 이 표에서 $\ell=0$(+0.441)과 $\ell=10$ ms(+0.234) 사이에 있고, 기울기로 나누면 **CES급 상태를 3.5 ms(강건) / 5.8 ms(고전) 묵힌 것과 등가**다. 빠른 진단 한 행이 운반하는 정보의 크기를 "지연"의 단위로 읽은 것이며 해석일 뿐 측정은 아니다.

소비 방식에 따른 구분. 10 ms 틱에 동기해 소비하는 제어기는 추론이 $\ell<\Delta$이면 추정치의 나이가 항상 $\Delta$ 이하로 고정되므로 추론 지연의 가격은 0이다. 이것이 §8ac의 "10 ms는 결코 구속하지 않는다"의 정보론적 번역이다. 위 표가 값을 매기는 것은 **비동기·연속 소비**(틱 사이에 추정치를 쓰는 제어, 그리고 µs 프로그램이 지향하는 소비)다.

##### 1(b) 충격: 붕괴는 못 보고 회복은 본다

진폭 $A$의 계단이 $t_0$에서 $\tau_{\rm crash}$ 동안 일어나고 지연 $\ell$의 센서가 참값을 $\ell$만큼 늦게 출력한다고 하자. $\ell>\tau_{\rm crash}$이면 첫 지연 표본이 반영될 때 붕괴는 이미 끝나 있다: 센서는 **진행 중인 붕괴를 보고할 수 없고**(트리거 불가) $[t_0,t_0+\ell]$ 동안 오차 $\approx A$를 진다. 충격 발생률 $r$에서 연속 소비의 시간평균 MSE 부담은 $r\,\ell\,A^2$. 회복 구간에서 지연 센서의 오차 최댓값은 $(y_0-y_{\min})(1-e^{-\ell/\tau_{\rm rec}})\approx A\,\ell/\tau_{\rm rec}$, 즉 **회복은 상대오차 $\ell/\tau_{\rm rec}$로 추적된다**(식은 §"시간 t의 식"). 여기에 라벨 자체의 제약이 겹친다: CES 적분 시간 ~10 ms(§9.5)이므로 $T_i$·$V_\varphi$ 라벨은 10 ms 평균이고, 0.1–1 ms 붕괴(theory_plasma_physics §6.3)는 어떤 추론 지연에서도 한 점의 점프로만 나타난다.

세 등급(시간 척도는 theory_plasma_physics §1.4 표·§6.3·§7.2·§9):

| 등급 | 플라즈마 시간 척도 | 요구 지연 | 10 ms 라벨로 가능? | 나우캐스터 |
|---|---|---|---|---|
| **A** 프로파일·수송 제어 | $\tau_E$ 77–140 ms(§1.4), 적합 $\tau$ 81 ms(§7.2) | $\ell\ll\tau_E$, 격자 $\Delta=10$ ms | 예 | 상태유지 p99 **1.494 ms**(§8ac), CPU 윈도 p99 **6.36–8.74 ms**(§8l) → **모두 A** |
| **B** ELM 완화 트리거 | 붕괴 0.1–1 ms(§6.3), 전구 수백 µs(§9) | $\ell<\tau_{\rm crash}$, < 1 ms | 아니오(적분이 붕괴를 평균) | 추론만 보면 1.5 ms는 경계, 라벨이 배제 |
| **C** MHD 모드 제어·회전 피드백 | 모드 주기 0.1–1 ms(§1.4) | $\ell\lesssim$ 주기 | 아니오 | 배제; kHz Mirnov Doppler(§7.5)는 **입력**으로만 |

µs 프로그램(§9.5, B.6)이 더하는 것은 등급 C의 입력(모드 주파수·진폭 → $V_\varphi$ Doppler·NTV 토크)이지 출력 케이던스가 아니다. 출력은 등급 A에 남는다.

##### 1(c) 지연 대 정보: 재귀 한 스텝의 연산자 수와 바닥

도달은 50 ms에서 포화하고(§8al) 재귀 상태가 그것을 운반하므로 스텝당 격자 한 행만 처리한다(§8aj: 111 ops가 도달 2와 630에서 동일). `seq_v2`(입력 22, `hidden_ti` 160 × 2층, `hidden_vt` 64 × 1층(입력 7), 헤드 64)의 한 스텝 MAC:

$$
4h(n_{\rm in}+h)=4\cdot160\cdot182=116{,}480,\quad 4\cdot160\cdot320=204{,}800,\quad 4\cdot64\cdot71=18{,}176,\quad \text{헤드 }14{,}464\;\Rightarrow\;353{,}920
$$

(§8ac §3의 353,920과 정확히 일치). 5 GFLOP/s(2 FLOP/MAC)에서 0.142 ms, 20 GFLOP/s에서 0.035 ms. 융합 LSTM 스텝의 최소 디스패치는 층당 14(cat, addmm, 슬라이스 2, sigmoid, tanh, Hadamard 3, add, tanh(c), 슬라이스 2) × 스텝당 2층 실행(두 분기 공용 0층 + $T_i$ 분기 1층) + 꼬리 11(split, LayerNorm 2, 헤드 addmm 4, GELU 2, cat) = **39**. 출하된 `seq_v2_tight`는 111(`data/.b9_op_counts.json`; 뷰·상태 정리도 aten op로 집계), lean 161, stock 118. 2–3 µs/op 법칙으로 바닥 **0.22–0.33 ms**(111 ops), 측정 min median **0.352 ms = 3.17 µs/op**, min p99 0.751 ms(`data/.b9_latency_min.json`)로 법칙 위에 있다. 이론 최소 39 ops면 0.08–0.12 ms. 비교: 윈도 W=2는 두 행을 매 스텝 재계산해 min median 2.418 ms, min p99 4.370 ms.

#### 시간 t의 식

본 절의 물리를 시각 $t$의 명시적 식으로 모은다. 기호: $\Delta=10$ ms 격자, $\ell$ 전달 지연(추론 비용), $g$ 마지막 CES 관측으로부터의 경과 시간, $\hat y(t)=E[\,y(t)\mid \mathcal I_{t-\ell}\,]$ 전달된 추정($\mathcal I_{t-\ell}$은 $t-\ell$까지의 모든 정보).

**(T1) OU 상태의 시간 전개.** $t_{\rm obs}=t-g$에 마지막 CES 관측이 있고 그 시점의 필터 사후분산이 $P$일 때, $s\ge0$에 대해

$$
y(t_{\rm obs}+s)-\bar m=e^{-s/\tau}\bigl[y(t_{\rm obs})-\bar m\bigr]+\int_{0}^{s}e^{-(s-u)/\tau}\sqrt{\tfrac{2\sigma_s^2}{\tau}}\,dW_u,
\qquad
\mathrm{Var}\bigl[y(t_{\rm obs}+s)\mid\mathcal I_{t_{\rm obs}}\bigr]=e^{-2s/\tau}P+\sigma_s^2\bigl(1-e^{-2s/\tau}\bigr).
$$

**(T2) 두 추정기의 오차를 $\ell$과 $g$의 함수로.** CES만 쓰는 인과 필터는 $\mathcal I_{t-\ell}$에 $t-g$의 관측이 들어 있으면($\ell\le g$) 지연과 무관하고, 오차는 $g$에만 의존한다:

$$
\mathrm{MSE}_{\rm CES}(t)=e^{-2g/\tau}P+\sigma_s^2\bigl(1-e^{-2g/\tau}\bigr)+\sigma_n^2 .
$$

나우캐스터는 빠른 진단이 매 행 있으므로 $\mathcal I_{t-\ell}$에서 상태를 (CES급이라 가정하면) $P$로 알고, 오차는 $\ell$에만 의존한다:

$$
\mathrm{MSE}_{\rm now}(t)=e^{-2\ell/\tau}P+\sigma_s^2\bigl(1-e^{-2\ell/\tau}\bigr)+\sigma_n^2,
\qquad
\mathrm{MSE}_{\rm pers}(t)=2\sigma_n^2+2\sigma_s^2\bigl(1-e^{-g/\tau}\bigr).
$$

따라서 skill은 $\ell$과 $g$의 명시적 함수다:

$$
\mathrm{skill}(\ell,g)=1-\frac{e^{-2\ell/\tau}P+\sigma_s^2(1-e^{-2\ell/\tau})+\sigma_n^2}{2\sigma_n^2+2\sigma_s^2(1-e^{-g/\tau})},
\qquad
\frac{\partial\,\mathrm{skill}}{\partial\ell}\Big|_{\ell=0}=-\frac{2(\sigma_s^2-P)}{\tau\,\mathrm{MSE}_{\rm pers}(g)},\quad
\frac{\partial\,\mathrm{skill}}{\partial g}>0 .
$$

검산(스크립트 Part 4): $\mathrm{MSE}_{\rm CES}(g)$는 $g=10/20/50$ ms에서 3169.8 / 3839.5 / 5091.9 eV²(강건)로 `.foundations.json`의 `by_gap.mse_causal`과 자리수까지 일치. $\mathrm{skill}(\ell,g)$ ($T_i$ 강건; 고전은 괄호):

| $\ell$ \ $g$ | 10 ms | 20 ms | 50 ms | 100 ms |
|---:|---:|---:|---:|---:|
| 0 | +0.441 (+0.410) | +0.547 (+0.467) | +0.687 (+0.564) | +0.762 (+0.630) |
| 1.5 ms | +0.406 (+0.397) | +0.519 (+0.455) | +0.667 (+0.554) | +0.747 (+0.622) |
| 6 ms | +0.310 (+0.360) | +0.441 (+0.422) | +0.614 (+0.527) | +0.707 (+0.599) |
| 9 ms | +0.252 (+0.338) | +0.394 (+0.402) | +0.581 (+0.511) | +0.682 (+0.585) |

읽기: 간극 $g$가 길수록 persistence가 나빠져 나우캐스터의 skill은 오르고, 지연 $\ell$은 그와 독립으로 분자를 키운다. 간극이 긴 곳(50 ms 이상)에서는 9 ms 지연도 skill 0.58을 남긴다. 이것이 §7.7(교량 대 외삽)의 지연 버전이다.

**(T3) ELM 궤적.** 발생 $t_0$, 붕괴 지속 $t_c$, $t_1=t_0+t_c$:

$$
y(t)=\begin{cases}
y_0, & t\le t_0\\[2pt]
y_0-A\bigl[1-e^{-(t-t_0)/\tau_{\rm crash}}\bigr], & t_0<t\le t_1\\[2pt]
y_{\min}+(y_0-y_{\min})\bigl[1-e^{-(t-t_1)/\tau_{\rm rec}}\bigr], & t>t_1
\end{cases}
\qquad
y_{\min}=y_0-A\bigl[1-e^{-t_c/\tau_{\rm crash}}\bigr].
$$

**(T4) 전달된 추정과 그 오차.** 발생 시각은 $\mathcal I_{t-\ell}$에서 예측 불가(§6.3의 이상 MHD 성장은 10 ms 격자의 어느 정보로도 선행되지 않음)이고, 발생 뒤에는 센서가 참값을 $\ell$만큼 늦게 재생한다: $\hat y(t)=y(t-\ell)$. $\ell>t_c$(측정된 1.5–9 ms 대 $t_c\lesssim1$ ms)인 경우 오차 $e(t)=y(t)-\hat y(t)$는 네 구간이다:

$$
e(t)=\begin{cases}
-A\bigl[1-e^{-(t-t_0)/\tau_{\rm crash}}\bigr], & t_0<t\le t_1 \quad\text{(붕괴, 센서는 아직 }y_0)\\[2pt]
-(y_0-y_{\min})\,e^{-(t-t_1)/\tau_{\rm rec}}, & t_1<t\le t_0+\ell \quad\text{(회복 시작, 센서는 여전히 }y_0)\\[2pt]
y(t)-y(t-\ell)\ \text{(혼합)}, & t_0+\ell<t\le t_1+\ell \quad\text{(센서가 붕괴를 재생)}\\[2pt]
(y_0-y_{\min})\,e^{-(t-t_1)/\tau_{\rm rec}}\bigl(e^{\ell/\tau_{\rm rec}}-1\bigr), & t>t_1+\ell \quad\text{(둘 다 회복 중)}
\end{cases}
$$

마지막 구간의 최댓값은 $t=t_1+\ell$에서 $(y_0-y_{\min})(1-e^{-\ell/\tau_{\rm rec}})$, 그 뒤 $\tau_{\rm rec}$로 감쇠한다 — 회복은 추적된다. 첫 두 구간의 오차는 $\approx y_0-y_{\min}$로 $\ell$ 동안 지속한다 — 붕괴는 추적되지 않는다. 오차 에너지 $\int e^2dt\approx(y_0-y_{\min})^2\,\ell+\tfrac{\tau_{\rm rec}}{2}(y_0-y_{\min})^2(e^{\ell/\tau_{\rm rec}}-1)^2$이고 충격 발생률 $r$에서 시간평균 MSE 부담은 $r\int e^2dt$.

수치(스크립트 Part 4; $\tau_{\rm crash}=0.3$ ms, $t_c=1$ ms는 §6.3 가정값, $A$·$\tau_{\rm rec}$는 Part 2 census):

| 충격 | $\ell$ [ms] | peak $|e|$ [eV] | $\int e^2dt$ [eV²·s] | 충격 에너지 대비 | $e(t_1+\ell)$ [eV] |
|---|---:|---:|---:|---:|---:|
| 3 $\sigma_n$ census: $A=509$ eV, $\tau_{\rm rec}=25$ ms ($y_0-y_{\min}=490.8$) | 1.5 | 490.8 | 306.8 | 0.097 | 28.6 |
| | 6 | 490.8 | 1297.2 | 0.411 | 104.7 |
| | 9 | 490.8 | 1865.1 | 0.590 | 148.4 |
| 회복형 상위 1 %: $A=1198$ eV, $\tau_{\rm rec}=37$ ms ($y_0-y_{\min}=1155.3$) | 1.5 | 1155.3 | 1691.7 | 0.066 | 45.9 |
| | 9 | 1155.3 | 10742.6 | 0.421 | 249.4 |

읽기: peak 오차는 $\ell$과 무관하게 붕괴 전폭(센서가 붕괴 자체를 못 보므로), 오차 에너지는 $\ell$에 거의 선형(1.5 ms에서 충격 에너지의 7–10 %, 9 ms에서 42–59 %). 회복 구간 오차는 1.5 ms에서 29–46 eV로 측정 잡음 $\sigma_n$(39 eV 강건) 수준 — **회복은 상태유지 스텝 지연에서 잡음 바닥 안에서 추적된다.**

**(T5) 라벨의 시간 적분.** CES 라벨은 $\bar y(t_k)=\frac1\Delta\int_{t_k-\Delta}^{t_k}y(t')\,dt'$. 발생 위상에 따라 발생 직후 첫 격자 행이 보는 낙폭은 $(y_0-\bar y)/(y_0-y_{\min})=0.00\ldots0.82$(평균 0.56; 회복형에선 0.86, 평균 0.57). 즉 라벨 자체가 붕괴를 평균하므로, 나우캐스터의 $\hat y$가 $\ell\to0$이어도 등급 B·C의 양(붕괴 중 값, 모드 위상)은 라벨에 존재하지 않는다. 등급 배정은 $\ell$이 아니라 이 적분이 정한다.

**(T6) 연산 비용의 시간식.** 격자 행 $k$에 대한 추론은 $t_k$의 입력이 갖춰진 뒤 $\ell_{\rm inf}=N_{\rm ops}\times(2\text{–}3\ \mu\text{s})$ 뒤에 끝나고($N_{\rm ops}=111$ → 0.22–0.33 ms, 측정 0.352 ms), 동기 소비자는 $t_{k+1}$에 그것을 쓰므로 나이 $=\Delta$ 고정; 비동기 소비자는 나이 $=\ell_{\rm inf}$에서 (T2)의 $\mathrm{skill}(\ell_{\rm inf},g)$를 얻는다.

#### 2. 검산

**표 1. $T_i$ 지연 비용(스크립트 Part 1; skill은 1-step persistence 기준, retain = skill(ℓ)/skill(0))**

| $\ell$ [ms] | 강건 MSE [eV²] | 강건 RMSE [eV] | 강건 skill | Δskill | retain | 고전 RMSE [eV] | 고전 skill | Δskill | retain |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 2313.1 | 48.09 | +0.441 | 0 | 1.000 | 162.18 | +0.410 | 0 | 1.000 |
| 1 | 2408.6 | 49.08 | +0.418 | −0.023 | 0.948 | 163.39 | +0.401 | −0.009 | 0.978 |
| **1.5** | 2455.4 | 49.55 | +0.406 | −0.034 | 0.922 | 163.98 | +0.397 | −0.013 | 0.968 |
| 2 | 2501.7 | 50.02 | +0.395 | −0.046 | 0.897 | 164.57 | +0.393 | −0.018 | 0.957 |
| 5 | 2767.8 | 52.61 | +0.331 | −0.110 | 0.751 | 167.86 | +0.368 | −0.042 | 0.897 |
| **6** | 2852.2 | 53.41 | +0.311 | −0.130 | 0.704 | 168.89 | +0.361 | −0.050 | 0.879 |
| **9** | 3093.3 | 55.62 | +0.252 | −0.189 | 0.572 | 171.77 | +0.338 | −0.072 | 0.825 |
| 10 | 3169.8 | 56.30 | +0.234 | −0.207 | 0.530 | 172.68 | +0.331 | −0.079 | 0.808 |

persistence RMSE 64.32 eV(강건) / 211.18 eV(고전). skill 0.01을 잃는 지연: 0.4 ms(강건) / 1.1 ms(고전); 0.05: 2.2 / 6.0 ms. $V_\varphi$는 더 가파르다: 기울기 −0.0446/ms(강건) / −0.0132/ms(고전) — 회전은 운반값에 더 의존하므로 묵힌 상태의 가치가 더 빨리 준다.

**표 2. 충격 census(스크립트 Part 2; 641파일, held-free, 3 keV 컷, 같은 블록 연속 10 ms 쌍 216,895개, 음 증분 100,544개; 관측 $T_i$ 행 225,793 = 2258 s)**

| 모집단 | n | 비율 [/s] | 낙폭 중앙 [eV] | p90 [eV] | 낙폭/직전 값 중앙 (p90) | 80 % 회복 중앙 [스텝] | p90 | $\tau_{\rm rec}$ |
|---|---:|---:|---:|---:|---|---:|---:|---:|
| 상위 1 % 음 증분(문턱 ≥ 973 eV) | 1006 | 0.446 | 1453.9 | 2276.5 | 0.816 (0.950) | 12(683/1006 회복) | 60 | 75 ms |
| ↳ 스파이크 하강(직전 값이 과거 5점 중앙 대비 > 0.5·낙폭 급등) | 806 (80.1 %) | — | 1509 | 2356 | 0.82 (0.95) | 16 | — | — |
| ↳ 회복형(50스텝 내 80 % 회복) | 147 (14.6 %) | 0.065 | 1198 | 1767 | 0.79 (0.97) | 6 | — | 37 ms |
| ↳ 비회복 계단 | 53 (5.3 %) | — | 1607 | 2186 | 0.80 (0.95) | — | — | — |
| 물리 크기(10–50 %, ≥ 150 eV, 전-급등 없음, 50스텝 내 회복) | 4139 | 1.833 | 221 | 403 | 0.25 (0.43) | 3 | 12 | 19 ms |
| 물리 크기, ≥ 420 eV(= 3 $\sigma_n$ 고전) | 369 | 0.163 | 509 | 663 | 0.41 (0.48) | 4 | 15 | 25 ms |

읽기. (i) 상위 1 % 낙폭은 **80 %가 3 keV 컷 아래에 남은 1점 적합 실패의 하강**이다: 직전 값이 과거 대비 0.93·낙폭 만큼 솟아 있고(중앙값), 낙폭 후 값은 과거 수준으로 복귀한다(−0.07·낙폭). 이 모집단으로 "10 ms 지연 센서가 놓칠 충격"을 재면 $r\ell A^2$가 1.5 ms에서 1802 eV²(강건 MSE_pers의 44 %)로 나오지만 그것은 기기 인공물의 값이다. (ii) ELM형(회복형) 부분모집단은 §6.3이 말한 "붕괴는 한 점, 회복은 수 점"과 맞는다: 회복 6스텝, $\tau_{\rm rec}$ 37 ms. (iii) 3 $\sigma_n$(고전) 문턱의 보수적 물리 census에서 연속 소비 지연 부담 $r\ell A^2$는 **1.5 ms에서 73 eV²(MSE_pers의 1.8 %), 9 ms에서 441 eV²(10.7 %)**이고, 회복 추적 상대오차 $1-e^{-\ell/\tau_{\rm rec}}$는 $\tau_{\rm rec}$ 75 ms 기준 1.5 ms에서 2.0 %, 9 ms에서 11.4 %. (iv) 낙폭 다음 격자 행 결측 18.2 %(183/1006) — §6.3의 "붕괴 직후 저 S/N로 적합 기각" MNAR 자리.

#### 3. 기록과의 연결

- **§8ac**: 상태유지 스텝 0.966 / 1.494 ms(14.9 %), 10 ms는 구속하지 않고 논증은 skill-per-ms. §1(a)·(T2)는 그 사실을 정보 단위로 번역한다: 동기 소비에서 $\ell<\Delta$의 가격은 0, 비동기 소비에서 1.5 ms는 이상 필터 skill의 92 %(강건)/97 %(고전)를 보존. §8ac §3의 353,920 MAC·"연산은 3 %"는 §1(c)의 0.142 ms(5 GFLOP/s)와 같은 계산.
- **§8ah·§8aj**: 비용 = 디스패치 수 × 2–3 µs. 111 ops → 0.22–0.33 ms 바닥 대 측정 0.352 ms. 21.84× 세션 산포로 1 ms 판정 보류 — 그러나 등급 B·C는 라벨 적분(T5)이 배제하므로 **보류가 등급 배정을 바꾸지 않는다**. 1 ms가 복원되더라도 B·C는 열리지 않고, 열리는 것은 비동기 소비에서의 표 1 손실 축소뿐이다.
- **§8l**: CPU p99 6.36–8.74 ms(윈도 W=4/2; 아티팩트 `.latency_benchmark.json`은 2026-08-17 재측정 4.455/5.053 ms로 갱신됨, §8ac 정확도 주석), CUDA batch-1 21.70 ms 중앙값으로 GPU가 더 느림. 표 1 기준 9 ms 지연은 강건 적합에서 이상 skill의 57 %만 남긴다 — 등급 A 안이지만 비동기 소비에선 상태유지 스텝과의 차이가 skill 0.15에 달한다.
- **§8al**: 50 ms 포화 → 재귀 한 스텝이 격자 한 행만 처리해도 되는 근거(§1(c), T6).
- **§9.5**: µs 프로그램은 입력 확장. §1(b) 표의 등급 C 행이 그 정확한 자리이고, (T5)가 "µs 타깃은 만들 수 없다"의 식이다.

#### 4. 한계와 뒤집을 측정

1. **OU 추상화.** 표 1과 (T2)의 $\ell=0$은 "CES급 관측을 $t$에 동화한 필터"이며 나우캐스터는 그것을 갖지 않는다. 표는 지연의 **한계 가격**(기울기)이고, 3.5–5.8 ms 등가 나이는 해석이다. 뒤집을 측정: 평가 npz의 빠른 진단 행을 $k$행 뒤로 밀어 채점하는 지연 사다리(모델 재학습 없이 추론만) — 기울기 −0.023/ms가 실제 모델에서 재현되는지.
2. **지연은 추론만.** 취득·특징 조립 파이프라인은 §8l 범위 밖이라 미측정. 제어 시스템 하드웨어에서의 종단 측정이 명명된 측정(§8ah).
3. **상위 1 % census는 인공물이 지배.** 명명된 수정: 스파이크 컷에 **1점 급등 하강 규칙**(직전 값 − 과거 5점 중앙 > 0.5·낙폭)을 추가 — 데이터 처리 변경이므로 승인이 필요한 통제 실험이며, §8ax의 꼬리 회계가 바뀔 수 있다.
4. **150 eV 문턱은 고전 $\sigma_n$(140 eV)과 같은 크기**라 잡음 오염 가능. 그래서 420 eV(3 $\sigma_n$) 행을 보수적 값으로 병기했다; 두 행이 같은 결론($\tau_{\rm rec}$ 19–25 ms, 지연 부담 수 %)을 준다.
5. **등급 경계의 숫자**($\tau_{\rm crash}$ 0.1–1 ms, 모드 주기 0.1–1 ms, (T3)의 $\tau_{\rm crash}=0.3$ ms·$t_c=1$ ms)는 이론 문서의 pedestal 가정값이지 본 데이터의 측정이 아니다. 100 Hz 격자(Nyquist 50 Hz)는 그것을 볼 수 없고, B.6의 kHz Mirnov 스트림이 그것을 재는 측정이다. (T4)의 "발생은 예측 불가" 가정도 같은 스트림(ELM 전구, §9)으로만 검정된다.
6. **이 기계의 절대 ms는 증거가 아니다**(§8aj 21.84×). 본 절이 기대는 불변량은 연산자 수 111·MAC 353,920·OU 모수이고, ms는 min-of-sessions 하한과 §8ac·§8l의 순서만 인용했다.

---

## 2. 종합 — 결과의 관련성을 물리로 말하면

### 2.1 세 문장

1. **이 나우캐스터는 pedestal 사건 검출기다.** $T_i$ 헤드라인의 90 % 이상은 ELM·천이 행에서 오고(P01), 거기서
   이기는 이유는 에너지식의 우변 $Q_{ei}(T_e-T_i)$·$n_e$가 ECEI·BES에 **같은 시각**에 비치기 때문이다. 조용한
   본류에서는 잡음 평균화의 산수(교량 $\tfrac32\sigma_n^2$ 대 필터 $P^-+\sigma_n^2$)대로 미래를 보는 보간에
   진다 — 그래서 "천이에서 쓰고 평온 구간은 보간·인과 GP에 맡긴다"는 §8ax의 배치 서술이 물리적으로 옳다.
2. **$V_\varphi$의 모든 판정은 운동량식 우변이 비어 있다는 한 사실의 귀결이다.** 토크가 비관측이면 최적 인과 예측은
   이력의 OU 필터이고(P02), 그 중앙값 하한(0.921×persistence) 위에 모델(1.14)·인과 GP(1.01)가 앉으며, 양측 보간은
   교량 하한(0.798)에 정확히 앉는다. 집계 양수는 상위 1 % 행의 꼬리 부등식 $G_T>D_B$가 주고, 방전별 승률 천장
   0.5–0.6은 사건 가시성 $p_{\rm obs}$가 우연 수준(0.99배)이기 때문이다(P08). 빠른 진단이 더할 수 있는 정보의
   상호정보 한계(P06의 방법)를 $V_\varphi$에 적용하면 0이다 — 10–200 ms 모든 창에서 $|r|\le0.014$.
3. **나머지 결과는 계측·표본화·평가의 물리다.** 캠페인 드리프트는 $g(D)s(t)+c(D)$의 날짜 의존 이득(P04), 3 keV
   컷은 PB 압력 한계와 1격자 내 가역 불가 파워(P05), MNAR은 $P(\mathrm{obs}\mid N(t),B(t))$의 상태 의존(P07),
   MAST 전이 null은 식 (M1)의 계수가 전부 장치 고유라는 것(P09), 실시간 등급은 $\tau_E$·ELM 붕괴·모드 주기의 세
   척도(P10), 50 ms는 필터 수렴이 아니라 ELM 회복·결합 척도(P03, 아래 2.3)이다.

### 2.2 결과 사이의 정량적 일관성 — 서로 다른 증명이 같은 수를 낸다

| 양 | 증명 A | 증명 B | 일치 |
|---|---|---|---|
| $T_i$ 천이 행의 SSE 비중 | P01: $f_I$ 0.72–0.78 (상위 5 % $J$) | P06: 상위 10 % $|e|$가 이득의 81–83 %; foundations §10.3: 상위 1 %가 skill의 62 % | 같은 꼬리 |
| 사건 동시성 | P01: $\Delta T_i$–$\Delta$ECEI 조건부 0.238, lag 0만 | P08: $p_{\rm obs}$ 농축 1.27–1.34배, logit +1.92 | 같은 기전 |
| 빠른 진단의 한 걸음 정보 | P06: 선형 한계 +0.047–0.056 | foundations §7.8: 측정 +0.048; P10: "3.5–5.8 ms 지연과 등가" | 안에 있음 |
| 조용 방전의 손실 | P08: 삼분위 −0.072($T_i$) = −0.073($V_\varphi$) | P01: 조용 절반 vs GP −0.09…−0.31 | $-c_b$, 구동원 무관 |
| $V_\varphi$ 꼬리 | P02: 상위 1 % = skill 전부 | P08: 충격행 승률 0.425 < 본류 0.530 | 크기 비관측 ⇒ $r\le0$ |
| 50 ms | P03: 필터 수렴 2–3 스텝(결손의 26–43 %), $\tau_{\rm eq}$ 18–39 ms, 회복형 ELM $\tau_{\rm rec}$ 60 ms | P06: $I_{\rm step}$ 양립 조건 $\tau_{\rm eq}\lesssim30$–60 ms; P10: 물리 크기 충격 $\tau_{\rm rec}$ 25–37 ms; P07: $\tau_B$ 26–35 ms | 자릿수 일치(정의에 따라 25–64 ms) |
| 측정 잡음 | P07: CRB 폭 배경 민감도 2–3배 | foundations §1.5: $N_{\rm ph}\approx300$ | 같은 광자 예산 |
| ELM 회복 | P01/P08: $h(s)=e^{-s/\tau_{\rm rec}}$, $\kappa_T\approx1.03$ | P10: 회복 추적 오차 $A\ell/\tau_{\rm rec}$ | 같은 커널 |

### 2.3 50 ms의 세 척도 (P03) — 한 척도는 기각, 두 척도는 자릿수

P03은 세 후보를 따로 재었다. (i) **잡음 Markov 필터의 냉시동 수렴**: $P_0=\sigma_s^2$에서 출발한 Kalman 재귀는
foundations의 $\mathrm{MSE}_K$를 정확히 재현하고, 2 % 포화는 **2 스텝(강건)/3 스텝(고전)**에서 온다 — 측정된 5 스텝이
아니다. 8al 곡선과의 RMS 차 0.021–0.026은 곡선 자체의 RMS 0.032와 같은 자릿수이고, 측정 감쇠율 0.559/스텝(기억 3.4
스텝)을 내려면 $K=0.155$가 필요하다. **OU 냉시동은 20 ms 결손의 26–43 %만 설명한다** — 이 가설은 주된 원인으로는
기각된다. (ii) **전자–이온 결합** $\tau_{\rm eq}$ 18–39 ms(CES 구석). (iii) **ELM 회복**: 641파일에서 1-step 25 % 이상
하강 사건 16,703건(7.7/s, 26 %는 단일점 적합 실패 dip) 중 가장 깨끗한 "회복형" 2,783건(1.3/s)의 중앙 곡선이
0.60 → 0.95(60 ms) → 1.0(80 ms), $\tau_{\rm rec}=60$ ms(정의에 따라 55–64 ms). 3모수 회복 적합의 CRLB가 잡음 바닥을
넘는 문맥 길이는 $t_{\rm ctx}\sim1.5\tau_{\rm rec}$ — 회복 **형태**를 식별하려면 그만큼의 문맥이 필요하다는 정보
논증이다. 판정: 50 ms는 $\tau_{\rm eq}$와 $\tau_{\rm rec}$ 사이에 놓이며 "같은 pedestal 물리"는 자릿수 수준으로만
방어된다. P10이 다른 정의(물리 크기 충격)로 잰 $\tau_{\rm rec}$ 25–37 ms, P07의 배경 완화 $\tau_B$ 26–35 ms, P06의
걸음당 정보 양립 조건 $\tau_{\rm eq}\lesssim30$–60 ms가 모두 같은 자릿수에 모인다.

### 2.4 두 가지 기록 정정

- §8ab 블록 2 "PR2 폴백 43 %": `eval_seq.py`의 `n_total`이 비채점 행까지 세는 집계 인공물. 채점 행 기준 폴백은
  $V_\varphi$ 0.3–0.5 %, $T_i$ 0.2 %(4분할 전부). 원장에 정정 주석(P02).
- §8az "긴 간극 안에서 빠른 진단이 $V_\varphi$를 회복": `bilstm` 대조군의 $V_\varphi$ 분기는 빠른 채널을 보지 않으므로
  성립 불가. 라벨의 양측 문맥 효과이며 빠른 채널의 몫은 `tm_scratch` 초과분 +0.049가 상한. 원장에 정정 주석(P02).
- 덧붙여 §8ar의 "$T_i$ 결합 크기 = rate 상관 +0.07"은 이력 조건부 partial $r$ 0.26으로 읽어야 한다(P06).

### 2.5 증명들이 지목하는 측정 (식의 어느 항을 닫는가)

| 측정 | 닫는 항 | 지목한 증명 | 뒤집힐 결과 |
|---|---|---|---|
| $D_\alpha$ / ELM 지시자 | (M1)의 $t_k$ | P01, P07, P08, P03 | 사건 행 분류가 날카로워지고 $p_{\rm obs}$가 1에 가까워짐 |
| kHz Mirnov 모드 주파수·모드수 | $V_\varphi$의 Doppler 직접 측정 $v=2\pi Rf/n$ | P02, P08, P10 | $V_\varphi$ 충격행 승률이 본류를 역전, $p_{\rm obs}$ 농축 > 1 |
| Mirnov 대역 파워 | $T_{\rm NTV}\propto\delta B^2$ | P02 | 간극 안 $I(g)$의 일부 관측 |
| NBI·RMP 파형 | $T_{\rm NBI}$, $T_{\rm NTV}$ | P02, P08, P09 | 운동량식 우변이 채워짐 |
| CES 프레임별 적합 품질($\chi^2$, $N$) | 채택 지시자 $m_j$ | P05, P07 | 값 컷을 품질 컷으로 교체, 두 모집단 병합 |
| CES 자속면의 보정 $T_e$ | $Q_{ei}$ 항 | P06, P03 | 결합항 부호 무작위(0.46)가 일관 양수로 |
| OU/Kalman 인과 팔의 채점 | 최적 이력 기준 | P02, P06 | 빠른 진단 귀속분이 +0.027–0.048로 확정 |
| 같은 추정기로 학습·추론하는 EWMA 표준화 팔 | $g(D),c(D)$ 온라인 소거 | P04 | 첫 1–2 s 워밍업만 남는지 |
| bilstm의 $V_\varphi$ 분기에 빠른 채널 라우팅 | 긴 간극의 사건 시점 정보 | P02 | +0.049 상한의 확정 |

### 2.6 어떻게 말할 것인가 (논문·발표용 한 단락)

> KSTAR pedestal의 이온 온도와 토로이달 회전은 같은 형태의 1차 완화 방정식을 따르지만 우변이 다르다. 이온 에너지의
> 우변은 전자–이온 결합과 밀도 요동이고 ECEI·BES가 그것을 같은 시각에 보므로, 인과 나우캐스터는 pedestal 사건이
> 일어나는 행에서 미래를 보는 보간을 이기고(헤드라인 +0.24의 90 % 이상), 평온한 행에서는 잡음 평균화의 한계대로
> 보간에 진다. 토로이달 운동량의 우변은 토크이고 이 데이터셋은 그것을 담지 않으므로, 회전의 최선은 이력의 잡음
> 필터이며 측정된 모든 회전 판정(보간과 동률, persistence 우위, 중앙값 열세, 방전별 승률 0.48)이 그 필터의 닫힌
> 식에서 나온다. 두 타깃의 차이는 모델의 차이가 아니라 **보존식의 우변에 관측 가능한 항이 있는가**의 차이이고,
> 그 식이 다음에 재야 할 양 — ELM 지시자, kHz Mirnov의 모드 주파수, NBI·RMP 파형, 프레임별 적합 품질, 보정
> $T_e$ — 을 항별로 지목한다.
