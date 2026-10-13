# 연구를 뒷받침하는 물리·수학적 근거 — 유도와 검산

작성 2026-10-13. 이 문서는 KSTAR CES 나우캐스팅 연구(`THESIS_RESULTS.md`, `docs/paper/main_ko.tex`)의
모든 설계 결정과 핵심 결과를 **제1원리에서 유도**하고, 각 유도가 내놓는 **수치 예측을 실데이터·얼린
산출물·몬테카를로로 검산**한 기록이다. 검산은 `ces_prediction/analyze_foundations.py`가 수행하며
결과는 `data/.foundations.json`과 `data/.foundations.log`에 있다 (저장소 루트에서
`py ces_prediction/analyze_foundations.py`). 모델은 학습·채점·선택하지 않았고 TEST는 이미 원장에
인용된 얼린 npz를 통해서만 읽었다. 이 문서는 어떤 판정도 바꾸지 않는다.

**표기.** 격자 간격 $\Delta = 10$ ms. 타깃 $y\in\{T_i, V_\varphi\}$, 관측값 $\tilde y = y + n$
($n$은 측정 잡음, 분산 $\sigma_n^2$). 빠른 진단 $z$ (BES 9, ECEI 4, MC 2). 관측 지시자 $m_t\in\{0,1\}$.
skill $=1-\mathrm{MSE}_{\text{model}}/\mathrm{MSE}_{\text{base}}$ (Murphy). "컷 모집단" = $T_i>3$ keV
제외, "포함 모집단" = 전량. 수치 출처는 괄호에 §로 표시(`THESIS_RESULTS.md`의 절 번호).

**검산의 세 등급.** 각 명제 끝의 "검산" 줄은 다음 중 하나다.
- **항등식**: 얼린 산출물 위에서 대수적으로 반드시 성립해야 하는 관계를 확인(예: 앙상블 분산 항등식).
- **예측–측정 정합**: 유도가 내놓은 수치와 독립적으로 측정된 수치의 비교(예: OU 모형이 예측한 보간/persistence
  오차비 0.81 대 측정 0.83–0.85).
- **몬테카를로**: 닫힌 식과 시뮬레이션의 일치(예: Cramér–Rao 한계).

---

## 0. 한 장 요약 — 명제, 예측, 측정, 상태

| # | 명제 | 유도가 내놓는 수치 | 측정 | 상태 |
|---|---|---|---|---|
| 1.5 | $T_i$ 잡음 바닥과 $V_\varphi$ 잡음 바닥은 하나의 광자 예산으로 설명된다 | $N_T=326$, $N_V=258$ (비 1.27) | MAD 46.4 eV, 4.3 km/s (§8aq) | 정합(자릿수·1.3배) |
| 1.6 | 선 중심과 선폭의 ML 추정은 통계적으로 직교 | Fisher $I_{\mu\sigma}=0$ | 결측 독립(8.2% vs 23.9%, 마스크 일치 72.8%, §8at) | 정합(근거 제공) |
| 3.2 | 전자–이온 평형화 시간 | 8–59 ms (CES 영역) | 문맥 포화 50 ms (§8al) | 정합(자릿수) |
| 3.3 | 1차 완화계는 level 상관 > rate 상관 > 0 | 0.42 > 0.02 (시뮬) | 0.31 > 0.078 (§8ar) | 정합(순서) |
| 4 | 운동량식의 어떤 항도 보유 채널로 닿지 않는다 | $h_V(z)\equiv0$ | 빠른 채널 교란 시 $V_\varphi$ 비트 동일, rate 상관 −0.006 (§8ab, §8ar) | 확인 |
| 5.2 | 100 Hz 데시메이션은 kHz 모드의 lag-1 상관을 지운다 | $r_1\approx0$ ($\sigma_f\ge50$ Hz) | $r_1=-0.009$ (§8b.2); MC −0.03…+0.03 | 확인 |
| 5.3 | 한 표본이 담는 진폭 정보는 상대오차 $1/\sqrt2$ | 0.707 | MC 0.70–0.72 | 확인 |
| 7.2–7.4 | persistence / 보간 / 인과 필터의 1-step 오차비 | 보간 0.81, 필터 0.88, 1표본 0.91 | 중앙값비: PCHIP 0.85, linear 0.83, GP 0.84, 인과 GP 0.96, 모델 1.03 (s42) | 정합(10 % 이내) |
| 7.5 | 과거 1관측이 사실상 충분(Markov), 잔여 이득 ≤ 8 % MSE | K=2: 6.8 %, K≥5: 8.1 % | W 스윕 plateau W=2(§8f); 도달 사다리 −0.065 @20 ms (§8af) | 정합 |
| 7.6 | 필터 기억은 1–2 스텝, 포화 ≈ 5 스텝 | 50 ms | 포화 50 ms (§8al) | 정합 |
| 7.7 | 이력만으로는 보간에 항상 진다(간극이 길수록 더) | −0.21 (10 ms) → −0.58 (50 ms) | 인과 GP 중앙값비 1.12–1.25 (모든 간극에서 열세) | 부호 확인, 증가 미해상 |
| 7.8 | 이력 기반 인과 skill은 대부분 잡음 평균화 | 필터 0.33, 동역학 한계 0.067 | 인과 GP vs persistence +0.310 (1-step, s42) | 정합(0.02 이내) |
| 9.1 | PCHIP은 linear보다 잡음을 증폭 | 1.027 (MC) | 1.022 ($T_i$), 1.043 ($V_\varphi$) | 정합 |
| 10.2 | 공통 예측불가 행은 skill을 희석(헤드라인 보수성) | 정확식 +0.589 | 컷 후 +0.589 (항등식), 등비용 근사 +0.487 | 항등식 |
| 10.3 | MSE skill은 분위 적분; 상위 1 %가 skill의 62 %($T_i$)·100 %($V_\varphi$) | 누적 기여 곡선 | q75 교차(§8ax) | 항등식 |
| 11 | shot 군집의 설계효과 | $1+(m-1)\rho$ | 3.1 ($T_i$), 1.4 ($V_\varphi$) | 측정 |
| 15.4 | 유계 손실이 val-MSE에 보이는 몫 = 본류 SSE 비중 | 10 % (컷), 5.9 % (포함) | epoch 2–4 조기 종료 (§8ay) | 정합(기전) |
| 16 | 앙상블 이득 = 멤버 불일치 분산 / 기준 MSE | +0.0097 | +0.009 (s42), 4분할 평균 +0.02 (§8aw) | 항등식 |

---

# 제1부 측정 물리

## 1. CES: 전하교환 분광의 선형, 광자 통계, 잡음 바닥

### 1.1 Doppler 선형 — 선폭이 온도, 이동이 회전

시선 방향 속도 $v$로 움직이는 방출원의 파장은 $\lambda=\lambda_0(1+v/c)$다. 방출 이온(KSTAR CES는
C VI $n=8\to7$, $\lambda_0=529.05$ nm)이 온도 $T$의 Maxwell 분포 $f(v)\propto\exp(-m v^2/2kT)$를
따르면 선형은
$$
I(\lambda)\;\propto\;\exp\!\left[-\frac{(\lambda-\lambda_0-\lambda_0 v_\parallel/c)^2}{2\sigma_\lambda^2}\right],
\qquad
\sigma_\lambda=\frac{\lambda_0}{c}\sqrt{\frac{kT}{m}},
\qquad
\mathrm{FWHM}=\lambda_0\sqrt{\frac{8\ln2\,kT}{mc^2}} .
$$
따라서 **선폭 → $T$**, **중심 이동 $\Delta\lambda=\lambda_0 v_\parallel/c$ → $V_\varphi$** 이다. 탄소 온도를
$T_i$로 읽는 것은 불순물–주이온 열평형 시간이 측정 적분 시간보다 짧다는 표준 가정이다.

**검산.** $T_i=593$ eV(관측 중앙값)에서 $v_{th}=\sqrt{kT/m_C}=69.0$ km/s, $\sigma_\lambda=0.122$ nm,
FWHM $=0.287$ nm.

### 1.2 광자 통계와 Cramér–Rao 한계

픽셀 $k$의 기대 광자수 $\lambda_k=N p_k(\mu,\sigma)$ (총 $N$ 광자, $p_k$는 정규화된 Gaussian), 계수는
Poisson. Fisher 정보는
$$
I_{ij}=\sum_k\frac{1}{\lambda_k}\frac{\partial\lambda_k}{\partial\theta_i}\frac{\partial\lambda_k}{\partial\theta_j}
\;\xrightarrow{\text{연속극한}}\;
I_{\mu\mu}=\frac{N}{\sigma^2},\quad I_{\sigma\sigma}=\frac{2N}{\sigma^2},\quad I_{\mu\sigma}=0 .
$$
($I_{\mu\sigma}$의 피적분함수 $x\,(x^2-1)e^{-x^2/2}$가 홀함수라 0.) 그러므로 어떤 불편 추정량도
$$
\sigma_{\hat\mu}\ge\frac{\sigma_\lambda}{\sqrt N},\qquad
\sigma_{\hat\sigma}\ge\frac{\sigma_\lambda}{\sqrt{2N}}
\;\;\Longrightarrow\;\;
\boxed{\;\sigma_{v}=\frac{v_{th}}{\sqrt N},\qquad \frac{\sigma_T}{T}=\frac{2\sigma_{\hat\sigma}}{\sigma_\lambda}=\sqrt{\frac{2}{N}}\;}
$$
($T\propto\sigma_\lambda^2$이므로 상대오차가 2배).

**검산(몬테카를로).** 48픽셀 Poisson 스펙트럼, 4모수(진폭·중심·폭·배경) Poisson 최대우도 적합.
배경을 함께 적합하는 4모수 CRB는 2모수 한계 대비 중심 ×1.00, 폭 ×1.08. $N=3000$에서 표본 산포/CRB
= 중심 1.00, 폭 1.02(점근 효율 확인); $N=300$에서는 중심 1.04, 폭 1.25–1.43(유한표본 비효율, 폭
추정량의 치우친 분포).

### 1.3 왜 CES는 느린가

$N\propto n_b\,n_C\,\langle\sigma v\rangle_{\rm CX}\,V\,\eta\,\tau$ 이므로 $N\propto\tau$(적분 시간).
1.2에 의해 $\sigma_v\propto\tau^{-1/2}$, $\sigma_T/T\propto\tau^{-1/2}$. 10 ms 프레임의 유효 광자수가
~300(1.5)이라면 $V_\varphi$ 1 km/s 정밀도($N\approx4800$)에는 ~160 ms가 필요하다. **시간 분해능과
정밀도의 교환이 측정 원리에 박혀 있고**, 이것이 빠른 진단으로 CES 격자를 채우려는 연구의 출발점이다.
또 S/N이 떨어지는 구간(저밀도, 빔 변조, ELM 직후)에서 적합이 기각되므로 결측은 물리적으로 흥미로운
순간에 몰린다 — MNAR의 물리적 기원(§12).

### 1.4 측정 잡음 바닥은 추정기가 넘을 수 없는 하한

타깃이 $\tilde y=y+n$이면 참값 $y$를 완벽히 맞히는 예측자도 $\mathrm{RMSE}^2=\sigma_n^2+b^2$를 기록한다
(논문 식 floor). §8aq의 차분 추정기(Rice 1차 $\sigma^2=E[d_1^2]/2$, 4차 $E[d_4^2]/70$)는 매끄러운 신호
위의 백색 잡음에 대해 불편이며, 차수가 오를수록 신호 편향이 줄어 **상한**을 준다.

### 1.5 두 잡음 바닥은 하나의 광자 예산인가 — 예측–측정 정합

1.2를 뒤집으면 측정된 바닥에서 유효 광자수를 읽을 수 있다. §8aq의 본류(MAD) 바닥 $T_i$ 46.4 eV
(593 eV의 7.8 %), $V_\varphi$ 4.3 km/s:
$$
N_T=\frac{2}{(46.4/592.8)^2}=326,\qquad
N_V=\left(\frac{69.0}{4.3}\right)^2=258,\qquad \frac{N_T}{N_V}=1.27 .
$$
서로 다른 타깃, 서로 다른 추정 경로로 잰 두 바닥이 **같은 자릿수의 광자수(~300/프레임)로 모인다.**
1.2의 유한표본 폭 비효율(×1.25–1.4)을 넣으면 $N_T$는 500–650이 되어 비는 2 이내로 남는다. MAD 바닥은
10 ms 이내의 실제 물리도 잡음으로 세므로 두 $N$ 모두 하한이다. 이 정합은 "타깃 자체의 재현성 한계"가
계측 잡음(광자 통계)이라는 §8aq의 해석을 물리적으로 받쳐 준다.

### 1.6 두 타깃이 독립적으로 결측되는 통계적 근거

1.2에서 $I_{\mu\sigma}=0$: 대칭 선형에서는 **중심(회전)과 폭(온도)의 ML 추정이 점근적으로 비상관**이다
(MC: 상관 +0.04). 적합 품질 판정이 모수별 불확실도나 잔차에 기반하면 한 모수의 기각이 다른 모수의
기각을 함의하지 않는다. 이것이 데이터의 독립 결측(8.2 % vs 23.9 %, 마스크 일치 72.8 %, §8at)과
정합하며, 타깃별 마스크·타깃별 손실(§6.3)이 필요한 이유의 계측 측 근거다. (파이프라인의 실제 QC 규칙은
비공개이므로 "근거 제공"이지 "증명"은 아니다.)

## 2. 빠른 진단이 실제로 재는 양

### 2.1 ECEI — 광학적으로 두꺼운 전자 사이클로트론 복사는 $T_e$ 그 자체

전자 사이클로트론 주파수 $f_{ce}=eB/(2\pi m_e)$, $B\propto1/R$이므로 주파수가 위치를 고른다. 광학 깊이
$\tau\gg1$인 2차 조화파 X-모드에서 복사 세기는 흑체의 Rayleigh–Jeans 극한
$$
I_\omega=\frac{\omega^2}{8\pi^3c^2}\,T_e(R(\omega))
$$
이므로 라디오미터 출력 $\propto T_e$ (국소). ECEI의 4채널은 네 위치의 $T_e(t)$다. (주의: pedestal
바깥의 얇은 영역에서는 $\tau<1$이 되어 비례가 깨진다 — 채널은 계측기 단위이며 본 연구는 $T_e$를 물리
단위로 갖지 않는다.)

### 2.2 BES — 빔 발광은 국소 전자 밀도에 비례

중성빔 원자의 Balmer-α 발광 $I_{\rm BES}\propto n_b\,n_e\,\langle\sigma v\rangle_{\rm exc}(E_b)\cdot B_r$
(분기비 $B_r$, 여기율은 $T_e$에 약하게 의존). 작은 섭동에서 $\delta I/I\approx\delta n_e/n_e$. 즉 BES
9채널은 국소 $n_e(t)$와 그 요동이다.

### 2.3 Mirnov — 코일 전압은 $dB/dt$

$V=-d\Phi/dt=-A\,\dot B$. 회전하는 MHD 모드 $\tilde B=b\cos(m\theta-n\varphi-\omega t)$에 대해
$V\propto\omega b\sin(\cdot)$, $\omega/2\pi$는 kHz(본 데이터셋 방전의 EHO 조화파 ~4, 8 kHz, §8ao). 즉
원 신호는 kHz 협대역이고, 100 Hz 격자가 이것을 어떻게 파괴하는지가 §5다.

---

# 제2부 수송 물리 — 왜 $T_i$에는 닿고 $V_\varphi$에는 닿지 않는가

## 3. 이온 에너지 방정식과 전자–이온 결합

### 3.1 평형화율의 유도

이온 에너지 밀도의 보존식(논문 식 energy):
$$
\frac{\partial}{\partial t}\Bigl(\tfrac32 n_iT_i\Bigr)+\nabla\!\cdot\!\mathbf q_i
= Q_{ei}\,(T_e-T_i)+S_i .
$$
Maxwell 전자가 무거운 이온에 쿨롱 충돌로 넘기는 에너지율은 Spitzer의 결과로
$$
\frac{dT_i}{dt}\Big|_{ei}=\nu_\varepsilon^{i|e}\,(T_e-T_i),\qquad
\nu_\varepsilon^{i|e}=\frac{2m_e}{m_i}\frac{1}{\tau_e},\qquad
\tau_e=\frac{3(4\pi\varepsilon_0)^2\sqrt{m_e}\,(kT_e)^{3/2}}{4\sqrt{2\pi}\,n_e e^4\ln\Lambda}
$$
($T_e\gg(m_e/m_i)T_i$ 가정). 따라서 $\tau_{\rm eq}\equiv1/\nu_\varepsilon=(m_i/2m_e)\,\tau_e$ — 논문이
쓴 관례가 유도의 결과다. 실용형 $\tau_e=3.44\times10^5\,T_e^{3/2}/(n_e[\mathrm{cm^{-3}}]\ln\Lambda)$ s.

**검산.** NRL의 정확식 $\nu_\varepsilon^{i|e}=1.8\times10^{-19}\sqrt{m_em_i}\,Z^2n_e\ln\Lambda/(m_iT_e+m_eT_i)^{3/2}$
와 비교하면 전 범위에서 비 1.03 — 관례의 "order-unity 상수"는 3 %다.

### 3.2 $\tau_{\rm eq}$의 크기와 50 ms

| $T_e$ \ $n_e$ | $2\times10^{19}$ | $3\times10^{19}$ | $5\times10^{19}$ m$^{-3}$ |
|---|---:|---:|---:|
| 500 eV | 20.1 ms | 13.4 | 8.0 |
| 1000 eV | 56.9 | 37.9 | 22.8 |
| 2000 eV | 160.9 | 107.3 | 64.4 |

관측 $T_i$ 중앙값 593 eV에 해당하는 구석은 **8–59 ms**. 측정된 인과 문맥 포화 50 ms(§8al)와 같은
자릿수다. (플라즈마 파라미터는 인용값이며 본 데이터는 $n_e,T_e$를 물리 단위로 갖지 않는다 — 자릿수
정합 이상을 주장하지 않는다.)

### 3.3 선형 응답: $T_i$는 $T_e$의 저역통과 — 그리고 level 상관이 rate 상관보다 큰 이유

밀도를 고정하고 식을 선형화하면 1차 완화계
$$
\frac{dT_i}{dt}=\frac{T_e-T_i}{\tau_{\rm eq}}-\frac{T_i-T_i^{\rm tr}}{\tau_{E,i}}+\frac{S_i}{3n_i/2},
\qquad
H(\omega)=\frac{1/\tau_{\rm eq}}{i\omega+1/\tau_{\rm eq}+1/\tau_{E,i}} .
$$
10 ms 격자의 대역(≤ 50 Hz)에서 $\omega\tau_{\rm eq}\sim0.5$–3이므로 $T_i$는 $T_e$를 부분적으로
추종한다. 또 $\Delta T_i\approx(\Delta/\tau_{\rm eq})(T_e-T_i)$: **증분은 수준차에 비례**하되 측정 잡음은
차분에서 $\sqrt2$배 커지고 $\Delta/\tau_{\rm eq}\ll1$이 신호를 줄이므로 rate 상관은 level 상관보다 작다.

**검산(시뮬레이션).** $\tau_{\rm eq}=40$ ms, $T_e$의 OU 상관 160 ms, 양쪽에 단위 측정 잡음:
corr(level) 0.42, corr(rate) 0.02. 측정(§8ar, ECEI): 0.31 / 0.078. 순서와 부호가 같다. 이것이
"빠른 진단 → $T_i$" 경로의 물리이며, 모델이 history-0에서도 $T_i$는 persistence를 이기는(§6.3 옛 ablation)
이유다.

### 3.4 BES가 $T_i$에 닿는 경로

같은 식에서 $n_i\approx n_e$(준중성), $Q_{ei}\propto n_e$, $\nabla\cdot\mathbf q_i$는 밀도 요동이
구동하는 난류 열수송에 비례한다. 따라서 $n_e$와 그 요동(BES)은 $T_i$의 **변화율에 곱해지는 계수**로 들어온다.
측정: BES→$T_i$ level +0.341, rate +0.070(§8ar).

## 4. 각운동량 방정식 — 항별로 닫힌 음성

$$
\frac{\partial}{\partial t}\bigl(n_im_i\langle R^2\rangle\omega_\varphi\bigr)+\nabla\!\cdot\!\Pi_\varphi
= T_{\rm NBI}+T_{\rm NTV}+T_{\rm int}+\cdots
$$

### 4.1 NBI 토크 — 파워가 아니라 기하와 에너지

에너지 $E_b$의 빔 이온이 접선반경 $R_{\rm tan}$으로 들어오면 각운동량 $m_bv_bR_{\rm tan}$을 가져온다.
초당 $P/E_b$개가 들어오므로
$$
T_{\rm NBI}=\frac{P}{E_b}\,m_bv_bR_{\rm tan}=\frac{2PR_{\rm tan}}{v_b},\qquad v_b=\sqrt{2E_b/m_b}
\;\;\Longrightarrow\;\;
\frac{T_{\rm NBI}}{P}=\frac{2R_{\rm tan}}{v_b}\propto R_{\rm tan}E_b^{-1/2}.
$$
**검산.** $E_b=100$ keV D, $R_{\rm tan}=1.5$ m: $v_b=3.11\times10^6$ m/s, 토크 0.97 N·m/MW. 같은
파워라도 접선반경·에너지·공동/역방향에 따라 토크가 달라진다 — "power ≠ torque", 즉 $T_e$(가열 결과)로
토크를 대리할 수 없다는 §8b.3의 null의 물리.

### 4.2 반경 힘균형 — $V_\varphi$는 국소 스칼라의 함수가 아니다

이온 운동량 방정식의 반경 성분(정상, 관성 무시):
$$
E_r=\frac{1}{Zen_i}\frac{dp_i}{dr}-v_\theta B_\varphi+v_\varphi B_\theta .
$$
$v_\varphi=(E_r-\frac{1}{Zen_i}p_i'+v_\theta B_\varphi)/B_\theta$: 토로이달 회전은 **반경 전기장,
압력 구배, 폴로이달 회전**으로 정해지며, 그 셋 중 어느 것도 보유 채널(국소 $T_e$, $n_e$ 스칼라)이 담지 않는다.
신고전 폴로이달 회전 $v_\theta\approx k\,T_i'/(eB)$도 **구배**를 요구한다.

**검산(크기).** pedestal 수치($n_i=3\times10^{19}$, $T_i=600$ eV, 구배 길이 5 cm, $v_\varphi=100$ km/s,
$B_\theta=0.3$ T): $|(1/en)p_i'|\approx12$ kV/m, $|v_\varphi B_\theta|\approx30$ kV/m — 같은 자릿수.
어느 항도 무시할 수 없으므로 $T_i$ 하나로 $v_\varphi$를 역산할 수 없다.

### 4.3 NTV 제동은 $\delta B^2$에 비례

비축대칭 자기장 섭동의 신고전 토로이달 점성 토크는 $T_{\rm NTV}\propto\delta B^2\,\nu_*^{\,\alpha}\cdots$.
진폭의 **제곱**이 필요하고, §5가 보이듯 100 Hz 데시메이트된 Mirnov 표본은 진폭을 상대오차 71 %로만 담는다.

### 4.4 결론 — 관측 함수의 구조적 0과 인코더 분기

보유 채널 $z$에 대한 $V_\varphi$의 관측 함수는 $h_V(z)\equiv0$: $T_{\rm NBI}$ 부재, $T_{\rm NTV}$
(§5) 파괴, $\nabla\!\cdot\!\Pi_\varphi$ 측정 null(rate 상관 −0.006, §8ar), $T_{\rm int}$ 구배 필요.
추정 이론(§6.6)에 따르면 정보가 0인 입력을 추정기에 넣으면 분산만 늘므로 **인코더 수준에서 차단**하는
것이 Bayes 최적에 가장 가깝다. 검증: 15개 빠른 채널을 전부 교란해도 $V_\varphi$ 출력이 비트 단위로
동일(§8ab 블록 8).

### 4.5 운동량 확산 시간과 persistence

Prandtl 수 $\chi_\varphi/\chi_i\sim1$이므로 $\tau_\varphi\sim\tau_E\sim$ 수십–수백 ms. 10 ms 격자에서
$\Delta/\tau_\varphi\ll1$: §7.2에서 유도하듯 과거 값이 거의 최적 예측이고, 그 위에 더할 외생 정보가 없는
$V_\varphi$는 "이력 복사 + 잡음 평균화"가 도달 가능한 전부다. 이것이 $V_\varphi$가 보간과 비기고
persistence는 이기는 구조다.

## 5. Mirnov 데시메이션 — 표본화 정리로부터

### 5.1 Nyquist

$f_{\rm Nyq}=1/(2\Delta)=50$ Hz. kHz 모드는 표현 불가. 반앨리어싱 필터 없이 데시메이트하면 각 표본은
원 신호의 **순간값**이다.

### 5.2 lag-1 자기상관의 유도

협대역 신호의 자기상관은 Wiener–Khinchin으로 $R(\Delta)=\int S(f)\cos(2\pi f\Delta)\,df/\int S$.
중심 $f_0$, 폭 $\sigma_f$의 Gaussian 스펙트럼이면
$$
r_1=\cos(2\pi f_0\Delta)\,\exp\!\bigl(-2\pi^2\sigma_f^2\Delta^2\bigr)
=\cos(2\pi f_0\Delta)\,\exp\!\bigl[-(\sigma_f/22.5\,\mathrm{Hz})^2\bigr].
$$
스펙트럼 폭이 수십 Hz만 되어도 $r_1\to0$: 실제 모드는 chirp·진폭 변조·모드 교체로 폭이 수백 Hz이므로
**연속 표본의 상대 위상이 무작위**가 된다. (cos 항이 1이 되려면 $f_0$가 수 초 동안 1 Hz 이내로 안정해야
한다 — 비현실적.)

**검산(MC).** 4 kHz 모드, 200 kHz로 생성, 100 Hz로 데시메이트: 원 신호 $r_1=0.99$; 데시메이트 후
$\sigma_f=50/100/200/400$ Hz에서 $r_1=-0.005/0.014/0.027/-0.031$. 측정 $r_1=-0.009$(§8b.2).

### 5.3 한 표본이 담는 진폭 정보

$x=A\cos\phi$, $\phi$ 균일: $E[x^2]=A^2/2$, $\mathrm{Var}(x^2)=A^4/8$, 상대오차
$\sqrt{1/2}=0.707$. 10 ms 창(40주기)의 RMS는 상대오차 ~0.001(순음) 또는 $1/\sqrt{2BT}$(대역).
**검산(MC).** 0.70–0.72 / 0.001.

### 5.4 데이터 처리 부등식 — 파생 특징이 음성인 이유

$I(A;g(x_{1:W}))\le I(A;x_{1:W})$: 데시메이트된 $W$개 표본의 어떤 함수(적분, $|{\rm MC}|$, rolling RMS,
PCHIP 적분)도 그 표본이 가진 정보를 넘지 못한다. $W=2$면 $A^2$ 추정의 상대오차 ≥ 0.5, 즉 NTV 항
($\propto A^2$)의 S/N ≤ 2. 그래서 §8b.2·§7.14의 파생 특징 4종이 전부 음성이며, 레버는 **원본 kHz 스트림**
(5.3의 창 RMS)이지 하류 처리가 아니다.

---

# 제3부 추정 이론 — 문제의 수학적 구조

## 6. 다중 속도 관측 하의 잠재상태 추정

### 6.1 상태공간 형식과 "나우캐스팅 = 필터링"

$$
x_{t+1}=f(x_t)+w_t,\qquad
z_t=h_f(x_t)+v_t\ (\text{매 스텝}),\qquad
\tilde y_t=h_s(x_t)+n_t\ (m_t=1\text{일 때만}).
$$
나우캐스트는 $\hat y_t=E[y_t\mid z_{1:t},\{\tilde y_s:m_s=1,s<t\}]$. 선형–Gaussian이면 Kalman 필터이고
그 갱신은 **재귀** $\xi_t=F(\xi_{t-1},z_t,m_{t-1}\tilde y_{t-1},m_{t-1})$이다 — 관측 지시자를 입력으로
받는 순환망. 전체격자 인과 LSTM + 손실측 마스킹(`seq_v2`)은 이 재귀의 비선형 일반화이며, 라벨 없는 행이
입력 문맥을 제공한다는 §7.2의 발견은 "미관측 시각에도 상태는 존재한다"의 구현이다.

**특징 설계의 충분통계 근거.** 스칼라 OU 타깃(§7)에 마지막 관측 이후 $s$스텝이 지났으면 사후분포는
$$
\mu_t=\bar m+\rho^{s}(\hat\mu_{\rm last}-\bar m),\qquad P_t=\rho^{2s}P_{\rm last}+\sigma_s^2(1-\rho^{2s}),
$$
즉 (마지막 값, 경과 스텝) 두 수의 함수다. `seq_data`의 (carried, staleness, has) 삼중항은 정확히 이
충분통계이고 $\log(1+s)$는 단조 재매개화다. 윈도 계열의 시간 특징 4채널(lookback, delta, 각 log1p)도 같은 근거.

### 6.2 MSE의 최소자는 조건부 평균

$\arg\min_f E[(y-f(x))^2]=E[y\mid x]$ (제곱을 전개하면 교차항이 0). 보고되는 것은 조건부 **평균**이고
분포가 아니므로 불확실성은 별도(§13).

### 6.3 타깃별 마스킹 손실의 최소자

$L=\sum_t\sum_c m_{t,c}(\hat y_{t,c}-y_{t,c})^2/\sum m$의 모집단 최소자는 $E[y_c\mid x,m_c=1]$. 이것이
$E[y_c\mid x]$와 같으려면 $m_c\perp y_c\mid x$ (입력 조건부 MAR). MNAR이면 학습된 함수는 **관측 조건부
평균**을 겨냥하며, 이 편향은 모든 arm이 공유하고 평가 측에서 §12가 다룬다.

### 6.4 "둘 다 관측" 필터의 선택 편향

두 타깃 모두를 요구하면 $T_i$ 손실의 최소자는 $E[y_T\mid x,m_T=1,m_V=1]$. $m_V$가 $x$를 주고도 $y_T$의
정보를 담으면($V_\varphi$ 결측은 천이·저 S/N에 몰림) 이는 $E[y_T\mid x,m_T=1]$과 다르다 — 라벨 행 28 %의
손실에 더해 **분포 편향**. 타깃별 마스킹은 둘 다 제거한다.

### 6.5 held(forward-fill) 값이 학습을 오염시키는 기전

관측값이 확률 $p$로 직전 값을 복사한다고 하자: $\tilde y_t=y_{t-1}$ (확률 $p$), $y_t$ (확률 $1-p$).
MSE 최적 예측은
$$
E[\tilde y_t\mid\cdot]=p\,y_{t-1}+(1-p)\,E[y_t\mid\cdot]
$$
— 학습된 함수가 **복사 쪽으로 가중 $p$만큼 수축**한다. $V_\varphi$ 관측의 54 %가 held이므로 $p\approx0.54$:
"이력이 지배적"이라는 서사의 절반은 계측기 패딩이 가르친 것이다. 평가에서는 held 행에서 persistence
오차가 정확히 0이므로 어떤 인과 방법도 이길 수 없다(§8r의 −48…−411). 이력 입력도 오염된다: "마지막 관측값"이
미지의 시간만큼 오래된 값이 된다. 따라서 held는 타깃·이력·정규화 통계·보간 앵커 네 곳에서 함께 제거해야
한다(§7.1 KEEP의 유도).

### 6.6 라우팅의 구조적 0과 공유 인코더의 누출

$\hat y_V=g(h_V(x_{\rm slow}))$이면 $\partial\hat y_V/\partial x_{\rm fast}\equiv0$ (교란 불변, 비트
동일). 공유 인코더 $h(x_{\rm fast},x_{\rm slow})$에서는
$\partial\hat y_V/\partial x_{\rm fast}=\partial g/\partial h\cdot\partial h/\partial x_{\rm fast}\ne0$
이 일반적이다. 정보가 0인 $d$개 입력을 선형 추정기에 더하면 초과 위험이 $\approx\sigma^2d/n$만큼 늘어난다
(자유도 비용) — "분산만 더한다". 측정: 공유 LSTM(seq v1)은 held-free에서도 $V_\varphi$를 4/4 seed에서 잃고,
인코더 분기(v2)가 그것을 4/4→1/4→0/4로 회복(§7.2–7.3).

### 6.7 관측 마스킹 어텐션 = 보간의 귀납편향

윈도 대조군의 풀링 $\hat h=\sum_k w_kh_k$, $w=\mathrm{softmax}(s)$에 관측 안 된 행을 $-\infty$로
마스킹하면 $\hat h$는 **관측 행 표현의 볼록결합**이 된다. 선형 보간 $\hat y=\sum_k w_k\tilde y_k$
(역거리 가중)과 같은 형식이고, 가중만 학습된다 — Nadaraya–Watson 커널 회귀의 학습형. 마스크 없는
softmax는 0으로 채운 행에 질량을 흘리고 그 억제를 flag 채널로 암묵 학습해야 하므로 파라미터 비용이 든다.

## 7. 잡음 있는 Ornstein–Uhlenbeck 모형 — 모든 기준선의 닫힌 형식

### 7.1 모형

$$
dy=-\frac{y-\bar m}{\tau}dt+\sqrt{\frac{2\sigma_s^2}{\tau}}\,dW,\qquad
y_{t+1}=\bar m+\rho\,(y_t-\bar m)+\varepsilon_t,\quad \rho=e^{-\Delta/\tau},\ \mathrm{Var}\,\varepsilon=\sigma_s^2(1-\rho^2),
$$
관측 $\tilde y=y+n$, $n\sim\mathcal N(0,\sigma_n^2)$. 반변동도(semivariogram)
$$
\gamma(h)=\tfrac12E[(\tilde y_{t+h}-\tilde y_t)^2]=\sigma_n^2+\sigma_s^2(1-\rho^h).
$$
세 모수 $(\sigma_n,\sigma_s,\tau)$는 $\gamma(h)$ 곡선에서 식별된다(nugget = $\sigma_n^2$, sill = $\sigma_n^2+\sigma_s^2$,
range = $\tau$). 식 energy(§3.3)의 선형 완화계가 백색 구동을 받으면 정확히 이 과정이다.

### 7.2 persistence

1-step 오차 $\tilde y_t-\tilde y_{t-1}$:
$$
\mathrm{MSE}_{\rm pers}=2\gamma(1)=2\sigma_n^2+2\sigma_s^2(1-\rho).
$$
(항등식: persistence MSE는 lag-1 반변동도의 두 배. 641파일 전수의 고전 $\sqrt{\gamma(1)}=151.8$ eV(§8aq)는
$\sqrt2\times151.8=214.7$ eV의 persistence RMSE를 함의하고, TEST s42 컷의 1-step persistence RMSE는 183.8 eV다 —
TEST 96 shot이 전수보다 조용한 만큼의 차이이며, 같은 자릿수.)

### 7.3 양측 보간

인접 두 이웃의 등가중 평균: 오차 $=\tilde y_t-\frac12(\tilde y_{t-1}+\tilde y_{t+1})$,
$$
\mathrm{MSE}_{\rm interp}=\tfrac32\sigma_n^2+\sigma_s^2\bigl(\tfrac32+\tfrac12\rho^2-2\rho\bigr)
\;\xrightarrow{\rho\to1}\;\tfrac32\sigma_n^2 .
$$
최적 가중은 정규방정식 $w=(\Sigma+\sigma_n^2I)^{-1}c$로 조금 더 낮다. **잡음 극한에서 보간/persistence의
MSE비는 3/4, RMSE비 0.866**: 보간은 앵커 둘의 잡음을 평균하고 persistence는 하나를 그대로 옮기기 때문.
이것이 §8ax가 측정한 본류 순서(GP ≈ PCHIP < 인과 GP < persistence ≈ 백본)의 유도다.

### 7.4 최적 인과 필터 (Kalman 정상상태)

$$
P^-=\rho^2P+\sigma_s^2(1-\rho^2),\qquad K=\frac{P^-}{P^-+\sigma_n^2},\qquad P=(1-K)P^-,
\qquad \mathrm{MSE}_{\rm filter}=P^-+\sigma_n^2 .
$$
과거 $K$개 관측만 쓰는 최적 선형 예측은 $\mathrm{MSE}_K=\sigma_s^2+\sigma_n^2-c^\top(\Sigma_K+\sigma_n^2I)^{-1}c$,
$(\Sigma_K)_{ij}=\sigma_s^2\rho^{|i-j|}$, $c_k=\sigma_s^2\rho^k$.

### 7.5 Markov 성질 — 과거 관측 하나로 충분한 이유와 그 예외

$\sigma_n=0$이면 OU는 Markov: $y_{t-1}$이 충분통계이고 $K=1$이 최적 — **W 스윕의 plateau(§8f, W=2)의
수학**. $\sigma_n>0$이면 더 오래된 관측이 **잡음 평균화**로 이득을 주되, 그 가중은 $(\rho(1-K))^k$로
기하 감쇠한다. 유도가 주는 상한: 무한 이력이 1관측 대비 줄이는 MSE는
$1-\mathrm{MSE}_\infty/\mathrm{MSE}_1$.

**검산(실데이터 적합, §8).** $T_i$ 강건 적합: K=2에서 6.8 %, K=3 7.9 %, **K≥5에서 8.1 % 포화**;
고전 적합: 5.8 / 7.6 / 8.3 %(K=5), 8.4 %(K≥10). 즉 (i) 과거 1관측이 이력 가치의 90 %를 운반한다 —
W=2 plateau; (ii) 나머지 ~8 %는 skill 단위로 $0.08\times\mathrm{MSE}_1/\mathrm{MSE}_{\rm PCHIP}\approx0.10$
(강건·고전 모두) — 도달 사다리가 같은 행 위에서 잰 20 ms 결손 −0.065(§8af)와 같은 자릿수이고, 스윕의 seed
산포(0.07–0.16) 안이라 짝짓지 않은 스윕이 그것을 보지 못한 이유도 설명된다; (iii) 포화 K=5 = **50 ms**.

### 7.6 필터의 기억 길이와 50 ms 포화

정상상태 필터에서 $k$스텝 전 관측의 가중 $\propto(\rho(1-K))^k$, 기억 길이
$L=1/[-\ln(\rho(1-K))]$. **검산.** $T_i$ 1.2 스텝(강건) / 1.8 스텝(고전) = 12–18 ms; 3–4 기억 길이
≈ 5 스텝에서 포화 → §8al의 50 ms. §3.2의 $\tau_{\rm eq}$ 8–59 ms와 함께, 50 ms에는 이제 **두 개의**
독립적 척도가 나란히 선다(결합 물리, 잡음 평균화 기억).

### 7.7 간극 의존 — 교량 대 외삽

과거 앵커가 $g$스텝 뒤에 있고 미래 이웃이 1스텝 앞에 있을 때, 이력만 쓰는 최적 인과 예측의 MSE는
$\rho^{2g}P+\sigma_s^2(1-\rho^{2g})+\sigma_n^2$로 $g$와 함께 커지고, 양측 보간은 미래 이웃에 기대어 거의
일정하다. 따라서 **이력만으로는 보간에 항상 지고 간극이 길수록 더 진다.** $T_i$ 강건 적합: skill(인과 vs 보간)
= −0.21 (10 ms), −0.33 (20), −0.44 (30), −0.58 (50), −0.80 (200 ms). 보간을 이기려면 외생 정보(빠른 진단)가
그 차이 이상을 운반해야 한다 — 이것이 "일부러 불리한 잣대"(§1.4)의 정량.

**검산.** s42 컷 $T_i$, 인과 GP의 중앙값 오차 / PCHIP 중앙값 오차: (0,15] 1.13, (15,25] 1.25, (25,45] 1.19,
(45,105] 1.13, >105 1.12 — **모든 간극에서 인과 열세(부호 확인)**. 간극에 따른 단조 증가는 해상되지 않는다:
구간당 120–600행이고 미래 앵커까지의 거리를 npz가 담지 않는다(유도는 미래 이웃 인접을 가정). 그런데 같은
행에서 **모델**은 PCHIP 대비 MSE skill +0.17/−0.04/+0.25/+0.39/+0.26 — 이력만으로는 불가능한 값이므로
그 차이가 빠른 진단이 운반하는 정보의 하한이다.

### 7.8 인과 skill의 분해 — 잡음 평균화 대 동역학

$$
\mathrm{skill}_{\rm filter\,vs\,pers}=1-\frac{P^-+\sigma_n^2}{2\sigma_n^2+2\sigma_s^2(1-\rho)} .
$$
$\sigma_n\to0$: $1-\frac{1-\rho^2}{2(1-\rho)}=\frac{1-\rho}{2}\approx\frac{\Delta}{2\tau}$ — **동역학만으로
persistence를 이기는 양은 $a/2$**, $\tau=80$ ms면 0.06. 나머지는 전부 잡음 평균화.
**검산.** $T_i$: 고전 적합 0.331(동역학 한계 0.067), 강건 0.234(0.062). 측정(s42 컷, Δt ≤ 15 ms): 인과 GP vs
persistence **+0.310**, 모델 +0.358. 즉 (i) 이력 기반 인과 방법의 persistence 대비 이득은 80 % 이상이
잡음 평균화이고, (ii) 빠른 진단은 1-step 집계 MSE에 **≈ +0.05**를 더한다 — 꼬리(천이)에 집중된 정보라는
§8ax·§8an과 일관.

## 8. 실데이터 적합과 예측–측정 대조

`analyze_foundations.py` Part D: 641파일, held-free, 3 keV 컷, 같은 블록의 연속 격자쌍만, 지연 1–12.
강건 반변동도 $\gamma_{\rm rob}(h)=\tfrac12(1.4826\,\mathrm{MAD}\,d_h)^2$ (본류), 고전 $\tfrac12\overline{d_h^2}$ (꼬리 포함).

| 적합 | $\sigma_n$ | $\sigma_s$ | $\tau$ | $\rho$ | Kalman 이득 | 기억 |
|---|---:|---:|---:|---:|---:|---:|
| $T_i$ 강건 | 39.0 eV | 68.7 eV | 81 ms | 0.884 | 0.52 | 1.2 스텝 |
| $T_i$ 고전 | 139.9 eV | 147.1 eV | 74 ms | 0.874 | 0.34 | 1.8 스텝 |
| $V_\varphi$ 강건 | 3.10 km/s | 8.43 km/s | 81 ms | 0.884 | 0.68 | 0.8 스텝 |
| $V_\varphi$ 고전 | 28.4 km/s | 46.3 km/s | 144 ms | 0.933 | 0.41 | 1.7 스텝 |

고전 $\sigma_n$ 139.9 eV는 §8aq의 4차 차분 상한 129.9 eV와 일치(같은 양의 다른 추정). 강건 $\sigma_n$
39 eV는 MAD 46.4 eV와 같은 자릿수이되 OU가 10 ms 내 신호 변화를 분리해 낸 만큼 작다. $V_\varphi$ 강건 적합은
held 제거가 "값이 바뀐 쌍"만 남겨 $\sigma_n$을 과대평가할 수 있다(§8ar 2절의 주의).

**1-step 사다리 예측 대 측정** (s42 컷, Δt ≤ 15 ms, n = 31,986; 중앙값 비는 persistence 대비):

| arm | 유도(강건) | 측정 중앙값비 | 측정 RMSE skill vs pers |
|---|---:|---:|---:|
| 양측 보간 | 0.813 | PCHIP 0.848 · linear 0.827 · GP 0.841 | 0.226 / 0.256 / 0.390 |
| 최적 인과 필터 | 0.875 | 인과 GP 0.959 | 0.310 (유도 고전 0.331) |
| 과거 1관측 최적 | 0.913 | — | — |
| persistence | 1 | 1 | 0 |
| **모델(MSE 학습)** | — | **1.032** | 0.358 |

읽기. (i) 보간 예측 0.81 대 측정 0.83–0.85: 10 % 이내 정합 — 본류에서 보간의 우위는 **잡음 평균화의 산수**다.
(ii) 인과 GP(0.96)는 유도된 최적 필터(0.875)는 물론 과거 1관측만 쓰는 최적 수축(0.913)에도 못 미친다 —
16이웃 국소 적합과 거친 잡음 격자의 비용이며, 더 나은 배치형 인과 기준선이 존재한다는 뜻이다. (iii) MSE 학습 백본은 1.03: **본류에서 앵커를 그대로 옮긴다** — §8ax의 측정("백본은
persistence와 함께 앉아 있다")이 유도된 하한 0.875 대비 **12.5 %의 중앙값 여지**가 남아 있음을 뜻한다. Huber
팔(§15)은 풀링 47.2→43.4 eV(0.94)로 그 여지의 절반을 회수했다.

## 9. 보간 기준선의 성질

### 9.1 PCHIP이 linear보다 약한 이유 — 잡음 앵커의 미분 증폭

두 보간 모두 앵커를 정확히 지난다. PCHIP의 Hermite 항은 앵커의 유한차분 기울기 $d_k$를 쓰며, 인접 앵커 사이
중점에서 선형 보간과의 차이는 $\tfrac h8(d_0-d_1)$ 꼴이다. 앵커가 잡음 $\sigma_n$을 가지면 기울기 추정이
그 잡음을 $1/h$로 증폭해 추가 분산을 만들고, Δt ≤ 15 ms가 채점 행의 97.7 %라 3차항이 도울 곡률이 없다.
**검산.** 적합된 $T_i$ OU+잡음 위의 MC: RMSE(PCHIP)/RMSE(linear) = 1.027; 측정(s42) 1.022 ($T_i$), 1.043 ($V_\varphi$).

### 9.2 GP가 PCHIP보다 강한 이유 — 잡음 인지 최소제곱

Matérn-3/2 사전 + 백색 잡음의 사후평균 $\hat y=k_*^\top(K+\sigma_n^2I)^{-1}\tilde y$는 그 사전 아래
**MMSE 선형 smoother**이고 앵커를 정확히 지나지 않는다(수축). PCHIP은 $\sigma_n^2$을 그대로 통과시킨다.
따라서 GP < linear < PCHIP (§8p, §5.4 PR1)의 순서는 7.3의 잡음 항이 결정한다. 인과 GP는 같은 식에서 미래 행을
뺀 것이므로 7.4의 필터에 수렴해야 하나, 측정은 0.96(8절) — 구현의 거친 격자(ls 5–80 ms × noise 4점)가 비용.

### 9.3 PR2 폴백

미래 이웃이 없으면 보간이 persistence를 예측한다. 검산: s42 컷 $T_i$에서 폴백 비율은 Δt ≤ 45 ms에서 ≤ 1 %,
(45,105]에서 7 % — 7.7의 간극 비교를 폴백이 만들지 않는다. $V_\varphi$는 W=2에서 40–44 %(§8ab 2절)이므로
"$V_\varphi$ vs PCHIP"의 절반 가까이는 "vs persistence"다.

---

# 제4부 평가와 통계

## 10. skill 점수의 성질

### 10.1 Murphy 분해

$\mathrm{MSE}=\mathrm{Var}(e)+\overline{e}^2$, skill $=1-\mathrm{MSE}_m/\mathrm{MSE}_b$는 기준선의 MSE로 정규화된
상대 지표라 단위와 타깃 사이에서 비교 가능하고, 동일 행·동일 마스크로 채점할 때만 의미가 있다(C4).

### 10.2 공통 예측불가 성분의 희석 — 헤드라인이 보수적인 이유

제거 대상 행이 기준선 SSE의 비율 $f_b$, 모델 SSE의 $f_m$을 차지하면
$$
\mathrm{skill}'=1-\frac{(1-f_m)\,\mathrm{SSE}_m}{(1-f_b)\,\mathrm{SSE}_b}
\;\;\xrightarrow{f_m\mathrm{SSE}_m=f_b\mathrm{SSE}_b}\;\;
\frac{\mathrm{skill}}{1-f_b}.
$$
어떤 arm도 못 맞히는 스파이크가 양쪽에 같은 제곱오차를 주면 MSE비를 1로 끌어 skill을 **깎는다**.
**검산(항등식).** 포함 s42 $T_i$: > 3 keV 110행(0.34 %)을 빼면 +0.225 → **+0.589**(정확식 재현); 그 행들이
PCHIP SSE의 53.8 %, 모델 SSE의 75.5 % — 모델이 스파이크에서 절대 오차가 더 크므로 등비용 근사(+0.487)보다
실제 회복이 크다. §8q의 "컷 후 두 배"는 이 식이다.

### 10.3 MSE는 분위 적분 — 꼬리 효과의 항등식

$\mathrm{MSE}=\int_0^1Q_{|e|}(p)^2dp$. 따라서 $\mathrm{skill}=1-\int Q_m^2/\int Q_b^2$는 **큰 분위에 제곱 가중**이고,
중앙값 비가 1보다 커도(본류 열세) 상위 분위 비가 1보다 작으면 skill은 양수가 될 수 있다.
**검산(항등식, s42 컷).** $(se_m+se_b)$ 순으로 누적한 skill 기여: $T_i$ q50 −0.004, q70 −0.011, q90 +0.007,
q99 +0.066, q100 **+0.174** — **상위 1 % 행이 skill의 62 %**(0.108/0.174)를, 상위 10 %가 96 %를 준다;
$|e|$ 분위비 모델/PCHIP는 q50 1.22 → q75 0.99 → q99 0.85(§8ax의 q75 교차 재현). $V_\varphi$: q99까지 −0.023,
q100 +0.390 — skill 전부가 상위 1 %에서 오고, 그 1 %가 PCHIP SSE의 83 %다. 이것이 "$T_i$ 우위는 꼬리에서 온다"의
대수이며, 왜 용량 축 세 개(§8z, §8aa, §8ag)가 평평했는지의 이유다: 손실이 1 %의 행으로 결정된다.

## 11. shot 군집 bootstrap — 설계효과

행 $i$가 shot $s$에 속하고 shot 내 상관 $\rho_{\rm in}$, 평균 크기 $m$이면 평균의 분산은
$\sigma^2[1+(m-1)\rho_{\rm in}]/n$: 설계효과 $D=1+(m-1)\rho_{\rm in}$, 유효 표본 $n/D$. 행을 독립으로 보면
CI가 $\sqrt D$배 좁아진다. shot 단위 복원추출은 $\rho_{\rm in}$을 모델링하지 않고도 타당하다.
**검산(s42 컷, paired $se_{\rm PCHIP}-se_{\rm model}$).** $T_i$: 32,589행/96 shot, 순진 SE 826 vs 군집 SE 1,458 →
**D = 3.1**, 함의 $\rho_{\rm in}=0.006$; $V_\varphi$: D = 1.4. 순진 CI는 1.8배(1.2배) 과소. 주의: 짝지은 차이가
무거운 꼬리라 순진 SE 자체가 불안정하며, 원장의 "유효 표본 ≈ shot 수"는 이 통계량에 대해 문자 그대로는 아니다
(10,454). 그래도 **검정력의 구속 조건이 shot 수**인 것은 꼬리 행이 어느 shot에 몰리느냐가 재표본마다 바뀌기
때문이며, 군집 bootstrap이 그것을 바르게 반영한다.

## 12. MNAR 사후층화 재가중

관측 가능한 공변량 $S$(마지막 genuine 관측 이후 경과, 입력만의 활동 flag)로 층화하면
$$
E_{\rm miss}[L]=\sum_sP(s\mid{\rm miss})\,E[L\mid s,{\rm obs}]+\underbrace{\sum_sP(s\mid{\rm miss})\bigl(E[L\mid s,{\rm miss}]-E[L\mid s,{\rm obs}]\bigr)}_{\text{층내 MNAR 편향, 식별 불가}} .
$$
첫 항이 §7.6·§8ab 9절의 추정량이다. 둘째 항은 층 안에서 결측이 MAR일 때 0이며, 그 가정은 검증 불가하므로
(i) 모델 도메인(W−1 행 이내 과거 관측) 밖의 결측은 세지 않고, (ii) 30행 미만 층을 버리며 덮인 질량을 보고한다.
CI는 가중치를 고정한 shot 군집 bootstrap이므로 가중 추정 분산은 덮지 않는다(명시된 한계).

## 13. split conformal — marginal 보증과 conditional 실패

교정 점수 $s_i=|y_i-\hat y_i|$, $i=1..n$, 시험 점수 $s_{n+1}$. 교환가능성 아래 $s_{n+1}$의 순위는 $\{1..n+1\}$에서
균일이므로 $q=\lceil(n+1)(1-\alpha)\rceil$번째 작은 점수에 대해
$$
1-\alpha\le P(s_{n+1}\le q)\le1-\alpha+\frac1{n+1}.
$$
분포 무가정·유한표본·재학습 불필요. 그러나 교정 shot과 시험 shot의 점수 분포가 다르면(shot별 척도 이동)
보증은 혼합 전체에 대한 marginal로만 성립하고 shot별 coverage는 50–100 %로 흔들린다(§7.9). Mondrian(층별 분위)은
층이 그 이동을 설명하는 만큼만 회복한다.

## 14. shot별 입력 표준화 — 공변량 이동

캠페인 사이에 입력이 $\delta$만큼 이동하고($\delta=1.22\sigma$ BES, 0.53σ ECEI; 타깃은 0.115σ, §8n) 모델이
국소 선형이면 예측 오차에 $(\nabla f^\top\delta)^2$이 더해진다. 각 shot 안에서 평균·σ로 표준화하면 위치·척도
성분의 $\delta$가 구성상 0이 된다. 비용은 **절대 수준 정보**의 손실인데, 헤드라인 분할에서 측정 가능한 손실이
0/4(§8s)였으므로 앵커(이력)가 이미 수준을 운반하고 빠른 진단의 절대 수준은 추가 정보를 담지 않았다.
온라인에서는 shot의 미래 통계를 쓸 수 없으므로 이는 오프라인 상한이고, 배치형은 확장 윈도/EWMA 추정기다.

## 15. 손실의 영향 함수 — 잡음 바닥에 고정한 Huber

### 15.1 M-추정과 유계 영향

$\rho_\delta(r)=r^2\ (|r|\le\delta)$, $2\delta|r|-\delta^2$ (그 외); $\psi=\rho'$는 $|\psi|\le2\delta$로 유계.
MSE($\psi=2r$, 무계)는 한 행의 잔차가 커질수록 그 행이 그래디언트를 독점한다; Huber는 $\delta$ 밖에서
중앙값 추정의 영향($\mathrm{sign}$)으로 바뀐다. $\delta$ 안에서 $r^2$와 **정확히 일치**하도록 정규화했으므로
본류의 곡률과 $0.1\cdot\mathrm{ReLU}$ 페널티와의 비율이 세 팔에서 같다(숨은 두 번째 변수 제거, B.14 수정).

### 15.2 $\delta$를 데이터 상수로 — 측정 잡음 안은 평균, 밖은 중앙값

$\delta_t=\sigma_{\rm meas}/\mathrm{std}_t$(§8aq 4차 차분 상한 129.9 eV, 16.4 km/s). 1.4에 의해 $|r|\le\delta$인
잔차는 타깃 자체가 재현되지 않는 범위라 제곱(조건부 평균)이 옳고, 그 밖은 물리 또는 적합 실패라 유계 영향이
옳다. 학습하지 않는 상수이므로 하이퍼파라미터가 아니다.

### 15.3 Student-t($\nu=3$)의 재하강이 뒤처진 이유

$\psi_t=2r/(1+r^2/\nu\delta^2)\to0$ ($|r|\to\infty$): 큰 잔차의 **정보를 버린다**. 10.3에 의해 skill이
상위 1 %에서 오므로 그 행의 그래디언트를 0으로 보내면 집계 MSE가 손해 본다 — 측정: tnll이 huber보다 모든
지표에서 뒤짐(§8ay).

### 15.4 val-MSE 체크포인트 함정의 정량

유계 손실이 개선하는 것은 $|e|\le\delta$인 본류다. 그 본류가 val MSE에서 차지하는 몫은
**검산(s42 컷 백본)**: $|e|\le129.9$ eV인 행이 84 %이지만 모델 SSE의 **10.0 %**; $V_\varphi$ 85 %/6.0 %;
포함 모집단 93.5 %/5.9 %. 본류 오차를 10 % 줄여도 val MSE는 1 %(컷)/0.6 %(포함)만 움직이고, 이는
epoch 간 잡음 아래라 patience가 2–4 epoch에서 발화한다(§8ay 7/8 run). 목적함수와 선택 지표는 한 쌍이다.

## 16. 앙상블 — 불일치 분산 항등식

멤버 $\hat y_i$, 평균 $\bar y$. 임의의 $y$에 대해 정확히
$$
(\bar y-y)^2=\frac1M\sum_i(\hat y_i-y)^2-\frac1M\sum_i(\hat y_i-\bar y)^2 .
$$
행 평균을 취하면 $\mathrm{MSE}_{\rm ens}=\overline{\mathrm{MSE}_i}-\overline{\mathrm{Var}_i(\hat y)}$: 앙상블
이득은 **멤버 불일치의 평균 분산과 같고 그 이상일 수 없다**; skill 이득 $=\overline{\mathrm{Var}_i}/\mathrm{MSE}_b$.
**검산(항등식, B.15 s42 컷).** 행별 Jensen $se_{\rm ens}\le\overline{se_i}$ 100 % 성립; 불일치 분산 292 eV²
= PCHIP MSE의 **0.0097** → 예측 이득 +0.0097; 측정(canonical 대비) +0.0090 (B.15 보고 +0.011은 분모가
canonical MSE). 멤버가 1 %만 불일치하므로 +0.02 동률(§8aw)은 구조적 결과다 — 더 많은 멤버로도 상한은 불일치 분산.

## 17. 경사 잡음 척도 (인용)

$B_{\rm simple}=\mathrm{tr}\Sigma/\|G\|^2$, 블록 단위 정확 계산(§8as): 중앙값 33.6 블록, 배치 16은 그 절반.
그 아래에서는 배치와 스텝 수가 선형 교환되므로 16은 낭비도 명백한 열등도 아니다. 유도·측정 모두 §8as와 논문 §4.7에 있음.

---

# 제5부 종합 — 유도가 연구를 어떻게 받치는가

1. **측정 원리(§1)**가 문제를 만든다: 광자 통계가 시간 분해능과 정밀도를 교환시키고($\sigma\propto\tau^{-1/2}$),
   저 S/N에서 적합이 기각되어 결측이 MNAR이며, 중심과 폭의 직교성이 두 타깃을 독립적으로 결측시킨다.
   두 잡음 바닥은 하나의 광자 예산(~300/프레임)으로 모인다.
2. **수송 물리(§3–4)**가 라우팅을 결정한다: 에너지식은 $(T_e,n_e)$를 $T_i$의 변화율 계수로 끌어들이고
   ($\tau_{\rm eq}$ 8–59 ms, level>rate 순서 재현), 운동량식의 모든 항은 보유 채널 밖에 있다
   (토크 = 기하·에너지, $v_\varphi$ = $E_r$·$\nabla p$·$v_\theta$, NTV = $\delta B^2$). 따라서 $h_V(z)\equiv0$이고
   인코더 분기가 Bayes 최적에 가장 가깝다.
3. **표본화 정리(§5)**가 Mirnov를 닫는다: $r_1=\cos(2\pi f_0\Delta)e^{-(\sigma_f/22.5\,{\rm Hz})^2}$, 한 표본의
   진폭 정보 상대오차 0.71, 데이터 처리 부등식으로 파생 특징 무용.
4. **추정 이론(§6–7)**이 프레이밍과 특징을 정당화한다: 나우캐스팅 = 필터링 = 관측 지시자를 받는 재귀,
   (carried, staleness, has)는 OU 사후분포의 충분통계, 마스킹 손실의 최소자는 관측 조건부 평균, held는 복사
   쪽으로 $p=0.54$ 수축, Markov 성질이 W=2 plateau를, 잡음 평균화 기억(1–2 스텝)이 50 ms 포화를, 교량 대
   외삽이 "불리한 잣대"의 크기(−0.21…−0.80)를 준다.
5. **실데이터 OU 적합(§8)**이 기준선 사다리를 10 % 이내로 예측한다(보간 0.81 vs 0.83–0.85)하고, MSE 학습
   백본이 본류에서 앵커를 그대로 옮긴다는 측정(1.03)에 대해 **도달 가능한 하한 0.875**를 세운다.
6. **평가 통계(§10–16)**는 항등식 수준으로 닫힌다: 희석 공식(+0.589 재현), 분위 적분(상위 1 %가 skill의
   62 %/100 %), 설계효과 3.1, 앙상블 상한 = 불일치 분산(+0.0097 vs +0.009), 유계 손실이 val-MSE에 보이는
   몫 10 %.

**유도가 지목하는 남은 측정** (§8j 규칙 — 음성은 뒤집을 측정을 함께 적는다).
- §7.8의 분해를 모델에 직접 적용: 같은 행에서 "최적 인과 필터(OU, §7.4)"를 arm으로 채점하면 빠른 진단의
  순 정보 $=\mathrm{skill}_{\rm model}-\mathrm{skill}_{\rm filter}$가 간극·층별로 분리된다(재학습 없음).
- §8의 하한 0.875 대 Huber 0.94: 앵커 디노이징(과거 관측 2–5개의 Kalman 가중)을 입력 특징으로 주는 팔이
  남은 6 %를 회수하는지(§8ax가 지목한 "이력 디노이징").
- §7.7의 간극 단조성: npz에 미래 앵커 거리 키를 가산 추가하면 교량 대 외삽 곡선이 측정 가능해진다.
- §1.5의 광자 예산: CES 프레임별 적합 품질(χ², 신호 레벨)이 오면 $N$을 직접 읽어 값 기반 컷을 품질 컷으로 교체.

---

## 부록 A. 가정값 (본 데이터가 담지 않는 양)

| 양 | 값 | 쓰인 곳 |
|---|---|---|
| CES 선 | C VI 529.05 nm, 탄소-12 | §1 |
| $T_i$ 대표값 | 592.8 eV (관측 중앙값) | §1.5 |
| 플라즈마 | $n_e=2$–$5\times10^{19}$ m$^{-3}$, $T_e=0.5$–2 keV, $\ln\Lambda=17$, Z=1, 중수소 | §3.2 |
| NBI | $E_b=100$ keV D, $R_{\rm tan}=1.5$ m | §4.1 |
| pedestal | $n_i=3\times10^{19}$, $T_i=600$ eV, $L_p=5$ cm, $v_\varphi=100$ km/s, $B_\theta=0.3$ T | §4.2 |
| Mirnov 모드 | $f_0=4$ kHz, 스펙트럼 폭 50–400 Hz | §5 |

## 부록 B. 재현

```
py ces_prediction/analyze_foundations.py        # -> data/.foundations.json, 콘솔 로그
```
Part D는 641파일 전수(약 2분), Part E는 `data/.b1_seqv2_s42_i{42,1,7,123}`, `.b5i_seqv2_s42`,
`.b15_ens4_cut_s42`의 얼린 npz를 읽는다. 난수 seed 20261013 고정.
