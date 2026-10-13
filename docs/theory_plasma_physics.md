# 토카막 플라즈마의 거동 물리 — CES 타깃을 예측하는 물리는 무엇이고, 데이터는 그것을 어디까지 확인하는가

작성 2026-10-13. `docs/theory_foundations.md`가 "CES가 어떻게 나오고 추정기가 무엇을 할 수 있는가"를
유도했다면, 이 문서는 **플라즈마 자체가 어떻게 움직이는가**를 단일 입자 운동에서 전역 구속 스케일링까지
전개하고, 그 물리가 $T_i$와 $V_\varphi$의 시간 거동을 **예측**하는 경로를 KSTAR 641방전 데이터로 검증한다.
검산은 `ces_prediction/analyze_foundations.py` Part F(`data/.foundations.json`의 `F`, 로그
`data/.foundations.log`). 모델은 학습·채점·선택하지 않았다.

**질문.** "CES 자체를 예측하는 물리는 설계된 것이 없는가?" — 있다. 두 개의 보존식과 한 개의 Doppler 관계다.
1. $T_i$: 이온 에너지 보존 + 전자–이온 결합 + 프로파일 경직성 ⇒ $T_i(t)$는 $(T_e, n_e, P_{\rm NBI,i})$의
   **저역통과 함수**이며 시간상수는 $\tau_{\rm eq}\parallel\tau_{E,i}$(8–140 ms).
2. $V_\varphi$: 각운동량 보존 ⇒ $V_\varphi(t)$는 토크의 저역통과 함수(시간상수 $\tau_\varphi\approx\tau_E$),
   그리고 **MHD 모드의 실험실계 주파수가 회전의 Doppler 측정**이다: $f_{\rm lab}=n\,v_\varphi/(2\pi R)+\cdots$.
   100 Hz Mirnov는 이것을 잃었고 kHz 스트림은 가진다.

**이 문서가 데이터로 확인한 것 (요약표).**

| 물리 | 유도·스케일링 | 데이터 | 상태 |
|---|---|---|---|
| 전역 에너지 구속 | IPB98(y,2) $\tau_E=77$ ms (KSTAR형 파라미터) | OU 적합 $\tau=81$ ms, 2SLS $T_i$ 완화 103–140 ms, 자기상관 1/e 159 ms | 자릿수 정합 |
| 운동량 구속, Prandtl 수 | $\chi_\varphi/\chi_i\sim1\Rightarrow\tau_\varphi\approx\tau_E$ | 2SLS $V_\varphi$ 완화 **89 ms** 대 $T_i$ 103–140 ms ⇒ $\Pr\approx0.6$–1.4 | **데이터에서 Pr ~ 1** |
| 잡음 감쇠 편향 | OLS는 $c$를 $-\sigma_n^2/\mathrm{Var}$만큼 끌어내림 | OLS 36/32 ms vs 2SLS 140/89 ms | 예측대로 |
| 모드 Doppler | $v_\varphi=2\pi Rf/n$ | EHO 4 kHz($n=1$), 8 kHz($n=2$) ⇒ 53 km/s; CES 중앙값 69 km/s | 자릿수 정합 |
| E×B 전단 억제 | $\omega_{E\times B}\gtrsim\gamma_{\rm ITG}$ | $8\times10^5$ vs $7\times10^5$ s$^{-1}$ (pedestal 가정값) | H-mode 조건 충족 |
| NBI 이온 가열 비 | Stix $G_i(E_b/E_c)$ | $T_e=1$ keV, 100 keV D: $E_c=18.6$ keV, $G_i=0.29$ | 전자 71 % → e–i 결합이 $T_i$ 경로 |
| e–i 결합항의 식별 | $\Delta T_i=b\,T_e-c\,T_i$ 에서 $b>0$ | 비보정 ECEI 평균으로는 $b>0$ 비율 0.46(무작위) | **식별 불가** — 보정 $T_e$ 필요 |

---

## 1. 단일 입자 운동 — 모든 척도의 출발점

### 1.1 자이로 운동

Lorentz 힘 $m\dot{\mathbf v}=q(\mathbf E+\mathbf v\times\mathbf B)$. 균일 $\mathbf B$에서 수직 운동은 원운동:
$$
\Omega=\frac{qB}{m},\qquad \rho=\frac{v_\perp}{\Omega}=\frac{mv_\perp}{qB}.
$$
pedestal 값($T_i=600$ eV, $B=1.8$ T, 중수소): $v_{th,i}=\sqrt{2T/m}=240$ km/s, $\Omega_i=8.6\times10^7$ s$^{-1}$,
$\rho_i=2.8$ mm, $\rho_e=\rho_i\sqrt{m_e/m_i}\cdot\sqrt{T_e/T_i}\approx46$ µm. $\rho_*=\rho_i/a=0.0055$ —
난류 수송의 기본 소변수(§5).

### 1.2 표류

안내중심 근사($\rho\ll L$, $\Omega\gg\omega$)에서
$$
\mathbf v_E=\frac{\mathbf E\times\mathbf B}{B^2},\qquad
\mathbf v_{\nabla B}=\frac{mv_\perp^2}{2qB^3}\mathbf B\times\nabla B,\qquad
\mathbf v_c=\frac{mv_\parallel^2}{qB^2}\frac{\mathbf R_c\times\mathbf B}{R_c^2}.
$$
$\mathbf v_E$는 전하 무관(플라즈마 전체가 함께 돈다 — 회전의 반이 여기서 나온다, §7), $\nabla B$·곡률 표류는
전하 의존이라 토로이달 장만으로는 전하 분리→수직 전기장→바깥쪽 $E\times B$ 표류로 플라즈마가 빠져나간다.
이것이 폴로이달 장(플라즈마 전류)이 필요한 이유이고, 토카막이다.

### 1.3 자기 거울과 포획 입자

$\mu=mv_\perp^2/2B$ 단열 불변. 토러스에서 $B\propto1/R$, 안쪽이 강하므로 $v_\parallel/v<\sqrt{2\varepsilon/(1+\varepsilon)}$
인 입자는 저자장 쪽에 **포획**되어 바나나 궤도를 돈다. 포획 비율 $\sim\sqrt{2\varepsilon}\approx0.75$
($\varepsilon=0.28$). 바나나 폭 $\Delta_b\approx q\rho_i/\sqrt\varepsilon\approx2$ cm — pedestal 폭과 같은 자릿수.
신고전 수송(§4)과 부트스트랩 전류, 그리고 NTV(§7.4)는 전부 포획 입자의 물리다.

### 1.4 시간 척도의 사다리

| 과정 | 식 | 값(pedestal 가정) | 어느 진단이 보나 |
|---|---|---|---|
| 자이로 | $2\pi/\Omega_i$ | 73 ns | — |
| Alfvén | $R/v_A$, $v_A=B/\sqrt{\mu_0nm}$ | 0.35 µs | — |
| 음속 통과 | $qR/c_s$, $c_s=170$ km/s | 42 µs | — |
| MHD 모드 회전 | $2\pi/(n\omega_\varphi)$ | 0.1–1 ms (kHz) | **Mirnov(kHz)**, 100 Hz에서 상실 |
| ELM 붕괴 | 이상 PB 성장 | 0.1–1 ms | Mirnov·$D_\alpha$; CES엔 한 점 |
| e–i 평형화 | $\tau_{\rm eq}$ | 8–59 ms | $T_i$ ↔ ECEI |
| 난류 상관 | $L_c/v_E$ | 10–100 µs | BES(요동) |
| 에너지/운동량 구속 | $\tau_E,\tau_\varphi$ | 77–140 ms | CES 시계열의 완화 |
| 방전 포락선 | $I_p$, 가열 프로그램 | 1–10 s | 모든 채널의 추세 |

10 ms CES 격자는 **구속 시간의 1/10, MHD의 10배** 위치에 있다. 격자가 보는 것은 수송·가열 척도의 완화이고,
MHD·ELM은 한 점의 점프로만 나타난다. 이것이 §7.7(foundations)에서 "모델은 타깃이 움직이는 곳에서만 이긴다"는
결과의 시간 척도 쪽 설명이다.

---

## 2. 토카막 평형 — 왜 한 점의 CES가 자속면 전체를 대표하는가

### 2.1 MHD 힘균형

정상 이상 MHD: $\mathbf J\times\mathbf B=\nabla p$, $\nabla\times\mathbf B=\mu_0\mathbf J$, $\nabla\cdot\mathbf B=0$.
축대칭에서 $\mathbf B=\nabla\psi\times\nabla\varphi+F(\psi)\nabla\varphi$로 쓰면 Grad–Shafranov 방정식
$$
R\frac{\partial}{\partial R}\frac1R\frac{\partial\psi}{\partial R}+\frac{\partial^2\psi}{\partial Z^2}
=-\mu_0R^2p'(\psi)-FF'(\psi).
$$
$\mathbf B\cdot\nabla\psi=0$: 자기력선은 $\psi=$const 면(자속면) 위에 산다. $\mathbf B\cdot\nabla p=0$이므로 **압력은
자속면 함수**다.

### 2.2 안전인자, 전단, $\beta$

$q=\frac{1}{2\pi}\oint\frac{B_\varphi}{RB_\theta}dl\approx\frac{rB_\varphi}{RB_\theta}$, 자기전단 $s=r\,q'/q$,
$\beta=2\mu_0p/B^2$. pedestal 가정값으로 $\beta\approx0.45$ %, $q_{95}\approx4$. 유리수면 $q=m/n$은 tearing·
NTM·EHO가 사는 곳이고, Mirnov가 재는 모드의 $(m,n)$이 그 면을 지정한다(§9).

### 2.3 자속면 함수인 이유 — 평행 수송의 빠름

자기력선을 따라 열은 $\chi_\parallel\sim v_{th}^2/\nu$로, 수직으로는 $\chi_\perp\sim\rho^2\nu$로 흐른다. 비
$\chi_\parallel/\chi_\perp\sim(v_{th}/\rho\nu)^2=(\Omega/\nu)^2\sim10^{10}$. 자속면 위에서 평형화하는 시간
$\sim(qR)^2/\chi_\parallel\sim qR/v_{th}\cdot(\nu qR/v_{th})$는 µs 급. **따라서 $T_e$, $T_i$, $n$, 전기퍼텐셜
$\Phi$는 자속면 함수이며** CES 시선이 지나는 한 점의 $T_i$는 그 자속면 전체의 $T_i(\psi)$다. BES·ECEI 채널이 다른
$(R,Z)$에 있어도 같은 $\psi$ 근처면 같은 양을 본다 — 이것이 공간적으로 떨어진 진단이 CES를 제약하는 기하적 근거다.
반대로 토로이달 회전은 자속면 위에서 강체 회전 $\omega_\varphi(\psi)$이지만 $v_\varphi=R\omega_\varphi$라
$R$ 의존이 남는다(EHO Doppler 환산 §7.5의 $R$ 선택).

---

## 3. 운동론에서 유체로 — 충돌과 두 유체 방정식

### 3.1 Boltzmann–Fokker–Planck와 모멘트

$\partial_tf+\mathbf v\cdot\nabla f+\frac qm(\mathbf E+\mathbf v\times\mathbf B)\cdot\nabla_vf=C[f]$.
$1,\ m\mathbf v,\ \tfrac12mv^2$ 모멘트가 연속·운동량·에너지 방정식이고, 각 식은 다음 모멘트(압력텐서, 열속)를 부르므로
**닫힘**이 필요하다. Braginskii 닫힘($\nu\gg\omega$, $\lambda_{mfp}\ll L_\parallel$)이 두 유체 방정식이다:
$$
\partial_tn_s+\nabla\cdot(n_s\mathbf u_s)=S_s,\qquad
m_sn_s\frac{d\mathbf u_s}{dt}=q_sn_s(\mathbf E+\mathbf u_s\times\mathbf B)-\nabla p_s-\nabla\cdot\boldsymbol\pi_s+\mathbf R_s+\mathbf F_s,
$$
$$
\frac32n_s\frac{dT_s}{dt}+p_s\nabla\cdot\mathbf u_s=-\nabla\cdot\mathbf q_s-\boldsymbol\pi_s{:}\nabla\mathbf u_s+Q_s .
$$
논문의 식 energy·momentum은 이 식의 자속면 평균(1-D 수송 방정식)이다.

### 3.2 쿨롱 충돌 — 시간 척도의 유도

Rutherford 산란의 운동량 전달 단면적 $\sigma_m\approx4\pi b_0^2\ln\Lambda$, $b_0=e^2/(4\pi\varepsilon_0 mv^2)$.
전자–이온 운동량 완화율 $\nu_{ei}=n\sigma_mv\propto ne^4\ln\Lambda/(\varepsilon_0^2\sqrt{m_e}\,T_e^{3/2})$:
$$
\tau_e=\frac{3(4\pi\varepsilon_0)^2\sqrt{m_e}\,T_e^{3/2}}{4\sqrt{2\pi}\,ne^4\ln\Lambda}
=3.44\times10^5\frac{T_e[\mathrm{eV}]^{3/2}}{n[\mathrm{cm^{-3}}]\ln\Lambda}\ \mathrm s .
$$
질량 순서: $\tau_{ee}:\tau_{ii}:\tau_{\rm eq}=1:\sqrt{m_i/m_e}:m_i/m_e$ — 한 번의 e–i 충돌은 에너지를
$\sim m_e/m_i$만 넘기므로 평형화는 $m_i/2m_e\approx1836$배 느리다. 값($n=3\times10^{19}$, $T_e=1$ keV):
$\tau_e=21$ µs, $\tau_{ii}\approx1.3$ ms, $\tau_{\rm eq}=39$ ms. pedestal $\nu_{ii}=760$ s$^{-1}$.

### 3.3 충돌도 $\nu_*$

$\nu_*=\nu\,qR/(\varepsilon^{3/2}v_{th})$: 바나나 궤도 한 바퀴 동안 충돌 횟수. pedestal $\nu_{*i}=0.24<1$ —
**바나나 영역**. 신고전 계수($k$, 부트스트랩, NTV 영역)가 여기서 정해진다.

---

## 4. 충돌 수송 — 고전과 신고전

### 4.1 무작위 걸음

충돌마다 안내중심이 $\rho$만큼 뛰면 $\chi_{\rm cl}\sim\rho^2\nu$. 전자 $\chi_e^{\rm cl}\sim10^{-3}$ m²/s —
관측(1 m²/s)의 1000분의 1. 고전 수송은 토카막을 설명하지 못한다.

### 4.2 신고전: 포획 입자의 바나나 걸음

포획 입자(비율 $\sqrt\varepsilon$)는 바나나 폭 $\Delta_b=q\rho/\sqrt\varepsilon$만큼 뛰고, 유효 충돌률은
$\nu/\varepsilon$(포획–비포획 경계만 넘으면 됨):
$$
\chi^{\rm neo}_{\rm banana}\sim\sqrt\varepsilon\cdot\Delta_b^2\cdot\frac{\nu}{\varepsilon}=\frac{q^2\rho^2\nu}{\varepsilon^{3/2}} .
$$
pedestal: $\chi_i^{\rm neo}\approx1.0$ m²/s. 이온 열수송에서 신고전은 **무시할 수 없는** 몫이고(ELM 억제
pedestal에서는 이온 채널이 신고전 수준까지 내려간다는 보고가 많다), 전자는 $\sqrt{m_e/m_i}$배 작아 난류가 지배한다.
확산 시간 $a^2/\chi^{\rm neo}\approx260$ ms.

### 4.3 신고전 폴로이달 회전과 부트스트랩 전류

폴로이달 회전은 자기 펌핑(poloidal damping, $\nu_{ii}$ 척도)으로 신고전 값에 묶인다:
$$
v_\theta^{\rm neo}=\frac{k(\nu_*)}{eB}\frac{dT_i}{dr},\qquad k_{\rm banana}\approx1.17,\ k_{\rm PS}\approx-2.1 .
$$
**$v_\theta$는 온도 구배로 정해진다** — foundations §4.2의 힘균형에 들어가는 셋째 양도 결국 $\nabla T_i$다.
같은 물리가 부트스트랩 전류 $j_{bs}\approx-\sqrt\varepsilon\,\frac{1}{B_\theta}\frac{dp}{dr}$를 만들고, 그것이 pedestal
전류→peeling 모드→ELM 안정성(§6.3)으로 이어진다.

### 4.4 Ware 핀치와 신고전 입자 수송

$v_{\rm Ware}=-\sqrt\varepsilon E_\varphi/B_\theta$ 안쪽 핀치. 밀도 프로파일의 peaking에 기여. BES가 보는 $n_e$의
느린 변화의 한 성분.

---

## 5. 난류 수송 — 왜 프로파일이 경직되고 $T_i$가 $T_e$·$n_e$를 따르는가

### 5.1 표류파

밀도 구배 $L_n^{-1}=-\nabla\ln n$에서 전자 반자성 주파수 $\omega_{*e}=k_\theta\rho_sc_s/L_n$, $\rho_s=c_s/\Omega_i$
($T_e$로 정의된 이온 음속 Larmor 반경). 전자가 자기력선을 따라 Boltzmann 응답($\tilde n/n=e\tilde\phi/T_e$)을
못 따라가는 비단열성(충돌, 포획)이 불안정의 에너지원이다.

### 5.2 ITG, TEM, ETG

- **ITG**(이온 온도 구배): $\eta_i=L_n/L_{T_i}$가 임계 이상이면 불안정, 토로이달 분지의 임계 구배
  $R/L_{T_i,{\rm crit}}\approx\frac43\left(1+\frac{T_i}{T_e}\right)\left(1+2\frac sq\right)$ (Jenko–Dannert 류 근사).
  성장률 $\gamma\sim c_s/\sqrt{RL_{T_i}}$, $k_\theta\rho_s\sim0.1$–0.5 — **BES가 보는 밀도 요동의 척도**.
- **TEM**(포획 전자 모드): 포획 전자의 비단열성, $\nabla n$·$\nabla T_e$ 구동, 충돌도가 오르면 안정화.
- **ETG**: 전자 척도($k\rho_e\sim1$), 전자 열수송.

### 5.3 혼합 길이와 gyro-Bohm

$\chi\sim\gamma/k_\perp^2\sim\frac{c_s}{\sqrt{RL_T}}\rho_s^2\cdot\frac{R}{L_T}\Rightarrow
\chi_{gB}=\rho_*\,\chi_B,\quad\chi_B=\frac{T}{16eB}$.
pedestal: $\chi_B=21$ m²/s, $\chi_{gB}=0.12$ m²/s. 관측 $\chi\sim0.3$–1 m²/s는 둘 사이 — ITER 스케일링의
$\rho_*$ 의존이 이 사이 어디인지를 나타낸다.

### 5.4 임계 구배와 경직성 — 프로파일이 "묶이는" 이유

임계 이상에서 $\chi$가 $(R/L_T-R/L_{T,{\rm crit}})$에 급격히 커지면(경직), 어떤 가열에도 구배는 임계 근처에 머문다:
$$
T(r)\approx T_{\rm ped}\exp\!\left[\int_r^{r_{\rm ped}}\frac{R/L_{T,{\rm crit}}}{R}dr'\right].
$$
**pedestal 값이 코어 전체를 띄운다**("pedestal이 구속을 정한다"). 함의 둘. (i) 한 자속면의 $T_i$는 다른 면의
$T_i$와 강하게 묶여 있어, CES 채널 하나가 프로파일 전체의 정보를 담는다 — 이력(persistence)이 강한 또 하나의
이유. (ii) $R/L_{T_i,{\rm crit}}$가 $T_i/T_e$에 의존하므로 **$T_e$ 변화가 $T_i$ 구배를 즉시 바꾼다** — ECEI→$T_i$
경로의 난류 쪽 메커니즘(§8의 e–i 결합과 별개).

### 5.5 E×B 전단 억제와 대상류

전단 유동이 난류 와류를 찢는 비율 $\omega_{E\times B}=\frac{(RB_\theta)^2}{B}\frac{\partial}{\partial\psi}\frac{E_r}{RB_\theta}
\approx\frac{E_r}{BL_{E_r}}$. 억제 조건 $\omega_{E\times B}\gtrsim\gamma_{\rm lin}$ (Biglari–Diamond–Terry,
Hahm–Burrell). pedestal 가정값 $E_r=30$ kV/m, $L_{E_r}=2$ cm: $\omega_{E\times B}=8\times10^5$ s$^{-1}$ vs
$\gamma_{\rm ITG}\approx c_s/\sqrt{RL_T}=7\times10^5$ s$^{-1}$ — **H-mode pedestal의 조건이 충족되는 자리**다.
$E_r$은 힘균형(foundations §4.2)으로 $\nabla p_i$, $v_\theta$, $v_\varphi$와 묶이므로 **회전과 구속은 같은 변수의
두 얼굴**이다: $V_\varphi$가 바뀌면 $E_r$ 전단이 바뀌고 수송이 바뀌어 $T_i$가 바뀐다 — 두 타깃의 느린 공변의 물리.
난류가 스스로 만드는 대상류(zonal flow)는 포식자–피식자 진동으로 L–H 천이 직전의 limit-cycle을 만든다.

---

## 6. H-mode, pedestal, ELM — CES 시계열의 "천이"는 무엇인가

### 6.1 L–H 천이 문턱

Martin(2008) 스케일링 $P_{LH}=0.0488\,n_{20}^{0.717}B^{0.803}S^{0.941}$ MW. KSTAR형($n_{20}=0.3$, $B=1.8$ T,
$S=4\pi^2Ra\sqrt\kappa\approx48$ m²): $P_{LH}\approx1.2$ MW. NBI 2–4 MW로 여유 있게 넘는다. 천이는 ~ms에 일어나고
pedestal은 수십 ms에 걸쳐 선다 — CES 격자에서 몇 점의 급상승으로 보인다.

### 6.2 pedestal 구조

mtanh 프로파일, 폭 $\Delta$와 높이. EPED: 폭 $\Delta_\psi\approx0.076\sqrt{\beta_{p,{\rm ped}}}$(kinetic-ballooning
제약)와 peeling–ballooning 한계의 교점이 높이를 정한다. 폭 ~2–3 cm, 바나나 폭과 같은 자릿수(§1.3) — 이온 궤도
효과가 pedestal 물리에 들어오는 이유. CES 채널은 보통 이 pedestal 상단–중단에 놓인다.

### 6.3 ELM 주기

pedestal 구배가 $\tau_{\rm build}\sim10$–50 ms(수송 시간)에 걸쳐 PB 한계까지 서고, 이상 MHD 성장률
$\gamma\sim0.1\,v_A/R\sim10^5$ s$^{-1}$로 0.1–1 ms 안에 붕괴한다. 에너지 손실 $\Delta W_{\rm ELM}/W_{\rm ped}\sim5$–20 %,
주파수 $f_{\rm ELM}\approx P_{\rm loss}/\Delta W_{\rm ELM}\sim10$–100 Hz. **10 ms 격자에서 붕괴는 한 점의 급락, 회복은
수 점의 상승**이다. 이 "톱니"가 §8ab peak 층(전역 +0.24 대 peak +0.45…+0.72)이 사는 곳이고, 붕괴 직후 저 S/N로
CES 적합이 기각되는 MNAR의 물리적 자리다.

### 6.4 RMP ELM 억제 — 본 데이터셋의 체제

데이터셋은 RMP(공명 자기 섭동) ELM 억제 구간을 겨냥해 선별되었다(RESEARCH_CONTEXT §2.1). 물리: $n=1,2$ 외부 코일
장이 pedestal 유리수면에 침투·차폐되며 (i) 밀도 펌프아웃(BES 평균 하강), (ii) **NTV 제동으로 회전 감소**(§7.4),
(iii) $E\times B$ 전단 변화로 pedestal 구배가 PB 한계 아래에 머문다. 억제 상태의 pedestal은 ELM 톱니 대신
저주파 완화·간헐적 작은 이벤트를 보이고, $V_\varphi$는 RMP 전류 프로그램에 따라 수십 km/s 규모로 느리게 변한다 —
**RMP 전류 파형이 있었다면 $V_\varphi$의 최상위 외생 입력**이 된다(NBI 토크와 함께 "없는 채널").

---

## 7. 토로이달 회전의 물리

### 7.1 운동량 수송 방정식

$$
\frac{\partial}{\partial t}\bigl(nm\langle R^2\rangle\omega_\varphi\bigr)+\frac1V'\frac{\partial}{\partial r}\bigl(V'\Pi_\varphi\bigr)=T_{\rm NBI}+T_{\rm NTV}+T_{\rm int}+\cdots,
\qquad
\Pi_\varphi=-nm\chi_\varphi\frac{\partial(R\omega)}{\partial r}+nmV_{\rm pinch}R\omega+\Pi_{\rm res}.
$$
세 성분: 확산($\chi_\varphi$), 대류 핀치($V_{\rm pinch}$, 보통 안쪽), 잔여 응력($\Pi_{\rm res}$: 구배·전단에서
토크 없이 회전을 만드는 항).

### 7.2 Prandtl 수와 $\tau_\varphi$

ITG 난류는 운동량과 이온 열을 같은 와류로 나르므로 $\Pr=\chi_\varphi/\chi_i\sim0.5$–1(이론·실험). 따라서
$\tau_\varphi\approx\tau_E$. 0-D로 줄이면
$$
\frac{dL}{dt}=T_{\rm tot}-\frac{L}{\tau_\varphi}\quad\Rightarrow\quad
V_\varphi(t)=V_\infty+(V_0-V_\infty)e^{-t/\tau_\varphi},\qquad V_\infty=\frac{T_{\rm tot}\tau_\varphi}{nm\langle R^2\rangle}.
$$
**이것이 foundations §7의 OU 과정이다**: 토크 요동이 백색이면 $V_\varphi$는 상관시간 $\tau_\varphi$의 OU.

**검산(데이터).** 641파일, held-free, 블록별 2SLS($y_{t-1}$ 도구변수): $V_\varphi$ 완화 시간 중앙값 **89 ms**
(IQR 38–252), $T_i$ 103–140 ms. IPB98(y,2) $\tau_E=77$ ms. ⇒ $\Pr_{\rm eff}=\tau_{E}/\tau_\varphi\approx0.6$–1.4.
OLS는 36/32 ms를 주는데 이것은 측정 잡음의 감쇠 편향($c_{\rm OLS}\approx c-\sigma_n^2/\mathrm{Var}\,\tilde y$)이며,
도구변수가 그것을 제거한다는 foundations §7의 예측 그대로다. (held 제거가 "값이 바뀐 쌍"만 남기므로 $V_\varphi$의
89 ms는 하한 쪽으로 치우쳤을 수 있다.)

### 7.3 고유 회전

토크 없이도 회전이 생긴다: Rice 스케일링 $\Delta V_\varphi\propto\Delta W/I_p$(H-mode 천이 시 공동 방향 가속),
잔여 응력 $\Pi_{\rm res}\propto\partial_r(\cdots)\,\nabla T_i$. **잔여 응력은 $\nabla T_i$의 함수**이므로 "$T_i$ 스칼라만
보유"로는 닿지 않는다(foundations 표 torque의 $T_{\rm int}$).

### 7.4 NTV 제동

비축대칭 $\delta B$는 포획 입자의 바나나 궤도를 비틀어 토로이달 운동량을 벽으로 넘긴다. $1/\nu$ 영역에서
$T_{\rm NTV}\propto\frac{nT_i}{\nu_i}\Bigl(\frac{\delta B}{B}\Bigr)^2(\omega_\varphi-\omega_{\rm NC})$ — 회전을
신고전 오프셋 $\omega_{\rm NC}\sim-\frac{3.5}{eB}\frac{dT_i}{dr}$(역방향)으로 끌어당긴다. RMP가 켜지면 $\delta B/B\sim10^{-3}$에서
수십 km/s를 깎는다. 이것이 **본 데이터셋 체제의 주요 토크**이고, 필요한 입력은 $\delta B^2$(진폭 제곱) — 100 Hz Mirnov
표본 하나는 그것을 상대오차 71 %로만 담는다(foundations §5.3).

### 7.5 MHD 모드 주파수는 회전의 Doppler 측정 — "CES를 예측하는 물리"

유리수면 $q=m/n$의 모드는 플라즈마 틀에서 $\omega_0$(반자성 주파수 $\sim\omega_*$ 또는 0)로 돌고, 실험실 틀에서는
$$
\omega_{\rm lab}=n\,\omega_\varphi(\psi_{m/n})-m\,\omega_\theta+\omega_0\;\approx\;\frac{n\,v_\varphi}{R}+\omega_{*}
$$
($\omega_\theta$는 신고전 감쇠로 작다). 따라서 **Mirnov 스펙트럼의 모드 주파수 $f$와 모드수 $n$을 알면
$v_\varphi\approx2\pi Rf/n$**이다. 모드의 한 종류인 EHO(edge harmonic oscillation)는 pedestal에 살므로 그 주파수는
**pedestal 회전**을 준다 — CES가 보는 바로 그 자리.

**검산.** 본 데이터셋 방전에서 보고된 EHO 조화파 ~4, ~8 kHz(§8ao). $R=2.1$ m: 4 kHz를 $n=1$로, 8 kHz를 $n=2$로 읽으면
둘 다 $v_\varphi=53$ km/s; 관측 CES $V_\varphi$ 중앙값 69 km/s. $\omega_*$ 보정과 CES 시선 반경을 감안하면 자릿수
정합이고, 조화 구조(4 → 8 kHz가 $n$ 1 → 2)가 Doppler 해석과 자기일관적이다. **이것이 CES 없이 $V_\varphi$를 예측하는
물리 경로이며, kHz Mirnov 스트림(B.6)이 그 데이터다.** 100 Hz 격자에서는 표본화 정리로 상실(foundations §5).

### 7.6 회전과 구속의 결합

§5.5: $V_\varphi\to E_r\to\omega_{E\times B}\to\chi\to T_i$. 그러므로 느린 시간(수십 ms)에서 $T_i$와 $V_\varphi$는
약하게 공변하고, 그 공변은 $E_r$을 통해 일어나지 직접 결합이 아니다. 데이터: $T_i$ 이력으로 $V_\varphi$를 보정하는
교차 경로는 닫혀 있다(§8au).

---

## 8. 이온 온도의 물리 — $T_i$를 예측하는 보존식

### 8.1 이온 파워 밸런스

$$
\frac32n_i\frac{dT_i}{dt}=\underbrace{P_{\rm NBI,i}}_{\text{빔→이온}}+\underbrace{\frac32n_i\frac{T_e-T_i}{\tau_{\rm eq}}}_{Q_{ei}}
-\underbrace{\frac32n_i\frac{T_i}{\tau_{E,i}}}_{\text{수송}}-\underbrace{n_in_0\langle\sigma v\rangle_{\rm cx}\tfrac32(T_i-T_0)}_{\text{전하교환}} .
$$
정상 상태에서
$$
T_i-T_e=\tau_{\rm eq}\left[\frac{P_{\rm NBI,i}}{\tfrac32n_i}-\frac{T_i}{\tau_{E,i}}-\cdots\right]
\;\;\Rightarrow\;\;
\frac{T_i-T_e}{T_i}\sim\frac{\tau_{\rm eq}}{\tau_{E,i}}\Bigl(\frac{P_{\rm NBI,i}\tau_{E,i}}{W_i}-1\Bigr).
$$
$\tau_{\rm eq}/\tau_E\approx0.1$–0.5(8–59 ms 대 77–140 ms): **$T_i$는 $T_e$에 10–50 % 안에서 묶여 있고**, 그
차이의 부호는 빔이 이온을 직접 얼마나 데우는지로 정해진다.

### 8.2 NBI의 이온/전자 분배 — Stix 감속

빔 이온은 전자(마찰, $\propto v$)와 이온(쿨롱, $\propto v^{-2}$)에 에너지를 나눠 준다. 두 비율이 같아지는 임계 에너지
$$
E_c=14.8\,T_e\,A_b\Bigl(\sum_i\frac{n_iZ_i^2}{n_eA_i}\Bigr)^{2/3},\qquad
G_i\Bigl(\frac{E_b}{E_c}\Bigr)=\frac{E_c}{E_b}\int_0^{E_b/E_c}\frac{dy}{1+y^{3/2}} .
$$
**검산.** $T_e=1$ keV, 중수소 빔 100 keV in 중수소: $E_c=18.6$ keV, $E_b/E_c=5.4$, $G_i=0.29$. 빔 파워의 **71 %가
전자**로 간다 — 이온 가열은 대부분 $Q_{ei}$를 거쳐 오므로, pedestal $T_i$는 $T_e$의 저역통과 함수라는 §3.3(foundations)의
그림이 소스 쪽에서도 성립한다. (고온 코어, $T_e\gg1$ keV면 $E_c$가 올라 이온 몫이 커진다.)

### 8.3 전하교환 — 가장자리의 싱크

중성 밀도 $n_0$(재순환)가 있는 pedestal에서 $\nu_{cx}=n_0\langle\sigma v\rangle_{\rm cx}\sim10^{3}$ s$^{-1}$
($n_0\sim10^{16}$ m$^{-3}$): 이온 에너지의 추가 싱크이자 **CES의 신호원**(빔 중성자와의 전하교환)이다. $n_0$가 요동하면
$T_i$ 싱크와 CES S/N이 함께 흔들린다 — 결측과 $T_i$ 하강이 동시에 오는 MNAR 메커니즘의 또 한 조각.

### 8.4 데이터 검증 — 완화 시간은 잡히고, 결합항은 비보정 ECEI로 식별되지 않는다

블록별 2SLS $\Delta T_i=a+b\,E_t+c\,T_{i,t}$ ($E$ = shot별 표준화 ECEI 평균): 완화 시간 중앙값 103 ms(ECEI 포함)
/ 140 ms(미포함), IQR 44–263 ms — $\tau_E$ 스케일링 77 ms·OU 81 ms·자기상관 159 ms와 한 자릿수. 그러나 **$b>0$인
블록 비율 0.46** — 결합항의 부호가 블록마다 무작위다. §8ar의 pooled 상관(+0.31 level, +0.078 rate)은 방전 포락선
수준의 공변이고, 블록 안에서 "$T_i$가 $T_e$ 쪽으로 완화한다"는 식별에는 (i) ECEI가 계측기 단위이고 (ii) 네 채널이
CES 자속면에 있지 않으며 (iii) $T_e-T_i$ 차가 필요한데 두 양의 단위가 다르다는 세 장벽이 있다. **지목되는 측정**:
CES 시선의 자속면에 매핑된 **보정 $T_e(\psi_{\rm CES},t)$** 한 채널. 그것이 있으면 §8.1이 $T_i$의 물리 기반 예측기
(계수 $\tau_{\rm eq}$, $\tau_{E,i}$ 두 개)가 되고, 학습 모델은 그 잔차만 맡으면 된다.

---

## 9. MHD 모드와 Mirnov — kHz 스트림이 담는 물리

| 모드 | 자리 | 주파수(실험실) | CES와의 관계 |
|---|---|---|---|
| 톱니(sawtooth) | $q=1$ 코어 | 10–100 Hz 반복, 붕괴 ~100 µs | 코어 $T_e$ 붕괴→열펄스가 수 ms 뒤 가장자리 도달 |
| 피시본 | $q=1$, 빔 이온 구동 | 10–30 kHz | 빔 이온 손실→토크 변동 |
| Tearing/NTM | $q=3/2,2/1$ | $n\,v_\varphi/R\sim$ kHz; 잠기면 0 | 회전 제동(잠김 전 감속), $T_e$ 평탄화 |
| EHO / QH | pedestal | 수 kHz 조화파 | **pedestal 회전의 Doppler 측정**(§7.5) |
| ELM 전구(precursor) | pedestal | 수십 kHz, 수백 µs | 붕괴 예고 |
| TAE | $v_A/(2qR)$ | 100–200 kHz | 빔 이온 재분배 |

Mirnov 코일 배열이 주는 세 양 — 주파수 $f(t)$, 진폭 $\delta B(t)$, 모드수 $(m,n)$ — 은 각각 §7.5(회전), §7.4(NTV 토크),
§2.2(어느 자속면)에 대응한다. 100 Hz로 데시메이트된 두 채널은 셋 다 잃는다. 원본 스트림에서 10 ms 창마다 (피크 주파수,
대역 파워, 교차위상)을 뽑는 것이 B.6의 특징이며, 그중 **주파수가 $V_\varphi$의 외생 입력, 파워가 NTV 토크의 입력**이다.

---

## 10. 가열·연료·전류 소스

- **NBI**: 빔 정지 단면적 $\sigma_s(E_b,n_e,T_e)$로 $\exp(-\int n_e\sigma_sdl)$ 감쇠; 흡수 프로파일이 $n_e$에 의존 →
  밀도 펌프아웃(RMP)이 **가열·토크 프로파일을 바꾼다**. 분배 §8.2, 토크 foundations §4.1(0.97 N·m/MW).
- **ECH/ECCD**: 전자만 가열, 국소 전류 구동(NTM 제어). $T_i$에는 $Q_{ei}$를 통해서만.
- **오믹**: $P_\Omega=\eta j^2$, $\eta\propto T_e^{-3/2}$; H-mode에서 작다.
- **연료/재순환**: 가스 퍼핑·벽 재순환이 $n_0$, $n_e$ pedestal을 정하고(§8.3) BES 평균 레벨의 느린 변화가 된다.

---

## 11. 전역 구속 스케일링과 무차원 파라미터

### 11.1 IPB98(y,2)

$$
\tau_E=0.0562\,I_p^{0.93}B^{0.15}P^{-0.69}n_{19}^{0.41}M^{0.19}R^{1.97}\varepsilon^{0.58}\kappa^{0.78}\ \mathrm s .
$$
**검산.** KSTAR형($I_p=0.6$ MA, $B=1.8$ T, $P=3$ MW, $n=3\times10^{19}$, $M=2$, $R=1.8$ m, $a=0.5$ m, $\kappa=1.8$):
**77 ms**. 데이터가 준 $T_i$·$V_\varphi$ 완화 시간(81–140 ms, 89 ms)과 같은 자릿수 — 스케일링과 CES 시계열의 동역학이
서로를 확인한다. ($P^{-0.69}$: 가열을 올리면 구속이 나빠지는 "파워 열화", 난류 경직성 §5.4의 전역 표현.)

### 11.2 무차원 삼총사

| 양 | 정의 | pedestal 가정값 | 역할 |
|---|---|---|---|
| $\rho_*$ | $\rho_i/a$ | 0.0055 | gyro-Bohm 척도, 난류 수송의 크기 |
| $\nu_*$ | $\nu qR/(\varepsilon^{3/2}v_{th})$ | 0.24 | 신고전 영역(바나나), NTV·부트스트랩 계수 |
| $\beta$ | $2\mu_0p/B^2$ | 0.45 % | MHD 한계(PB, NTM), EPED 폭 |

같은 $(\rho_*,\nu_*,\beta,q,\varepsilon,\kappa)$면 장치가 달라도 물리가 같다(무차원 동일성) — B.12에서 MAST 사전학습이
KSTAR에 아무것도 옮기지 못한 결과(§8az)는 모델이 이 변수들로 조건화되지 않았다는 쪽에서 읽어야 한다.

### 11.3 확산 시간의 사다리

$a^2/\chi$: Bohm 12 ms, 신고전 260 ms, gyro-Bohm 2.2 s. 관측 구속 ~100 ms는 Bohm과 gyro-Bohm 사이, 신고전과 같은
자릿수 — 이온 채널에서 신고전이 바닥(floor)이라는 뜻이다.

---

## 12. 종합 — 플라즈마 물리가 CES 타깃에 대해 말하는 것

### 12.1 $T_i$의 생성 모형

$$
\boxed{\;\frac{dT_i}{dt}=\frac{T_e-T_i}{\tau_{\rm eq}(n_e,T_e)}+\frac{G_i(T_e)P_{\rm NBI}}{\tfrac32n_i}-\frac{T_i}{\tau_{E,i}}-\nu_{cx}(T_i-T_0)+\text{ELM/천이 충격}\;}
$$
입력 중 데이터에 있는 것: $T_e$(ECEI, 비보정, 다른 자속면), $n_e$(BES, 요동 위주). 없는 것: $P_{\rm NBI}$, $n_0$, 보정.
시간상수 8–140 ms. 데이터 확인: 완화 103–140 ms, 결합항은 비보정으로 식별 불가. **학습 모델이 하는 일**은 이 식의
계수와 단위 변환을 데이터에서 암묵적으로 얻는 것이며, 그것이 꼬리(ELM·천이, $dT_i/dt$가 큰 곳)에서만 보간을 이기는
이유다: 평온 구간에서는 $T_i\approx$ 이력(Markov)이고 식이 더해 줄 것이 없다.

### 12.2 $V_\varphi$의 생성 모형

$$
\boxed{\;\frac{dV_\varphi}{dt}=\frac{T_{\rm NBI}(E_b,R_{\rm tan},P)+T_{\rm NTV}(\delta B^2,\nu_*,V_\varphi-V_{\rm NC})+T_{\rm int}(\nabla T_i)}{nm\langle R^2\rangle/R}-\frac{V_\varphi}{\tau_\varphi}\;}
$$
입력 중 데이터에 있는 것: **없음**(100 Hz Mirnov는 $\delta B^2$을 71 % 오차로). 시간상수 ~89 ms(데이터). 그러므로
$V_\varphi$는 OU이고 최적 인과 예측은 이력의 잡음 평균화(foundations §7)다 — 측정된 결과 그대로. 바꿀 수 있는 것은
입력뿐: (a) kHz Mirnov의 모드 주파수 ⇒ $v_\varphi$ 직접(§7.5), (b) 대역 파워 ⇒ NTV, (c) NBI·RMP 파형 ⇒ 토크.

### 12.3 두 타깃의 비대칭은 물리의 비대칭

| | $T_i$ | $V_\varphi$ |
|---|---|---|
| 보존량 | 에너지 | 각운동량 |
| 국소 결합 | $Q_{ei}\propto n_e(T_e-T_i)$ — **빠른 진단이 보는 양** | 없음($E_r$, 구배, 토크) |
| 소스 | NBI(71 % 전자 경유), 전도 | NBI 토크, NTV, 잔여응력 — **데이터에 없음** |
| 난류 경로 | 임계 구배가 $T_i/T_e$ 의존 | Prandtl ~1로 열과 같은 $\chi$, 하지만 구동이 다름 |
| 비CES 측정 경로 | ECE(보정 시) | **MHD 모드 Doppler(kHz Mirnov)** |

### 12.4 유도가 지목하는 측정 (각각 어느 식의 어느 항인지)

1. **보정 $T_e(\psi_{\rm CES},t)$** — §8.1의 $Q_{ei}$ 항. 2SLS 결합항 부호 무작위(0.46)를 뒤집을 유일한 데이터.
2. **kHz Mirnov 모드 주파수·모드수** — §7.5의 $v_\varphi=2\pi Rf/n$; EHO 53 km/s vs CES 69 km/s의 정합을 shot별 시계열로.
3. **Mirnov 대역 파워** — §7.4 NTV의 $\delta B^2$.
4. **NBI 파워·빔 에너지·접선 반경, RMP 코일 전류 파형** — §7.1 토크의 외생 입력; §6.4 체제의 주 구동.
5. **$D_\alpha$ 또는 ELM 지시자** — §6.3의 충격 항을 입력으로 주면 MNAR 자리의 꼬리가 모델링 가능.

---

## 부록 A. 가정값과 출처 성격

Part F의 모든 플라즈마 파라미터는 KSTAR 2022 캠페인 H-mode의 전형값으로 **가정**한 것이며 본 CSV에는 없다
($I_p=0.6$ MA, $B_T=1.8$ T, $P=3$ MW, $n_e=3\times10^{19}$ m$^{-3}$, $R=1.8$ m, $a=0.5$ m, $\kappa=1.8$, $q_{95}=4$,
pedestal $T=600$ eV, $E_r=30$ kV/m, $L_{E_r}=2$ cm, $L_T=3$ cm, EHO $R=2.1$ m). 스케일링식(IPB98, Martin, Stix,
Braginskii/NRL)은 표준 교과서·ITER Physics Basis 형태다. 데이터 적합(2SLS, OU)은 641파일 전수·확정 데이터 처리에서
나왔고 shot별 이질성(IQR)을 함께 보고한다.

## 부록 B. 재현

```
py ces_prediction/analyze_foundations.py        # Part F = 이 문서의 검산 (data/.foundations.json "F")
```
