# CES가 보고하는 T_i는 실제 이온 온도와 얼마나 다른가: 문헌 조사 (2026-11-18)

표기: [본문] = 출판 본문(또는 저자 사본)을 직접 읽음, [초록] = 초록·메타데이터만, [스니펫] = 검색 결과 발췌만 보고 원문 위치 미확인, [계산] = 이 보고서에서 직접 계산(문헌 수치 아님). 서지는 Crossref 또는 OpenAlex 메타데이터로 제목·저자·연도·권·쪽을 대조한 것만 "확인"으로 적었다.

---

## 1. 한 문단 요약

CES(CXRS)가 보고하는 T_i는 **탄소 C⁶⁺(C VI 8→7, 529 nm) 불순물 이온의 도플러 폭 온도**이고, 주 이온(중수소) 온도가 아니다. 표준 분석이 다루는 계통 효과는 (i) 수동 방출·배경 분리(KSTAR는 가열 빔 변조 = blip으로 배경을 잼), (ii) 원자물리 보정(에너지 의존 전하 교환 단면적, 미세 구조, Zeeman, 기기 함수), (iii) 시선 적분, (iv) 불순물-주 이온 온도 차이다. 확인된 정량 근거로 보면 pedestal top 부근에서 **불순물 T와 주 이온 T의 차이가 가장 크다**: ASDEX Upgrade에서 보정 전 T_D가 T_z보다 pedestal top에서 약 100 eV 높고 확장 보정 후 50–60 eV로 줄며(Cano-Megias 2025, 본문), DIII-D에서는 기울기 영역에서 중수소 T가 탄소보다 "~50 % 낮을 수 있다"(Haskey 외 APS 2016 초록), 분리면에서는 D⁺가 C⁶⁺보다 낮다(Haskey 2018 초록). 다만 AUG의 같은 논문은 T_z를 이온 열수송에 쓰는 것이 "generally a valid assumption"이라 결론짓고, pedestal top 안쪽에서는 T_D가 T_e를 따른다. 시선 적분 보정은 pedestal top에서 약 10 eV(같은 논문), 단면적 효과는 낮은 T_i에서 작고 주로 겉보기 속도에 나타나며(von Hellermann 1995 초록, Bell 2010 본문), Zeeman·미세 구조·기기 함수는 **온도에 거의 무관한 가산 오프셋**으로 들어간다(이 보고서의 계산으로 KSTAR 가장자리 1.5–1.8 T에서 Zeeman만 약 7–22 eV, T_i ≈ 500 eV의 1.5–4.5 %). KSTAR에 고유한 위험은 **fit 실패 이상치**(Kim 2024: Levenberg–Marquardt fit 실패로 120개 프로파일 중 60개가 이상치 포함)와 **10 ms 노출 안의 ELM 평균**(KSTAR CES 시간 분해능 10 ms, Kim 2024 본문)이다. KSTAR CES 문헌은 행별 불확도의 크기를 숫자로 공개하지 않으며(Heo 2025는 표준편차를 오차 막대로 씀), 설계 요구 정확도는 "10 %"였다(Lee 외 2010 KNS 학회문).

---

## 2. 오차원별 표

| 오차원 | 크기(정량 문장) | 조건 | 출처 | 확인 수준 |
|---|---|---|---|---|
| 불순물 T vs 주 이온 T (pedestal top) | "T_D is 100 eV higher than T_z in the pedestal top" → 확장 보정 후 "reduced to 50–60 eV"; 안쪽에서는 "T_D closely follows T_e at the pedestal top" | AUG, B⁵⁺ 대 D, H-mode | Cano-Megias 외 2025 NF 65 096027, §3.2–3.3 | [본문] |
| 같은 차이의 해석 | T_z 사용은 "generally a valid assumption and has mild impact" | 이온 열수송 해석 | 같은 논문, 결론 (iii-a) | [본문] |
| 불순물 T vs 주 이온 T (기울기 영역) | 중수소 T와 그 척도 길이가 "~50% lower than the carbon measurement" | DIII-D, pedestal 기울기 영역 | Haskey 외, APS-DPP 2016 CO4.00014 초록 | [스니펫] (학회 초록, 원문 페이지에서 문장 직접 확인 못 함) |
| 같은 차이 (분리면) | "the D+ temperature is lower than the C6+ temperature at the separatrix" | DIII-D, L·H-mode | Haskey 외 2018 PPCF 60 105001 | [초록] |
| 시선 적분 | 보정이 "brings up T_z at the pedestal top by approximately 10 eV"; 주 이온·불순물 적분 차이는 "up to 100 eV in the pedestal top region" | AUG 가장자리 CXRS | Cano-Megias 2025, §3.3·부록 | [본문] |
| 주 이온 분석의 H 혼입 | "overestimation of the D temperature up to 8%" (H 15 %) | 주 이온 CXRS 한정(탄소에는 해당 없음) | 같은 논문 부록 | [본문] |
| 에너지 의존 CX 단면적 | 겉보기 효과가 "a lineshift (apparent velocity), usually ... reduction in linewidth (apparent temperature)" | 빔 에너지·T_i 의존, T_i가 낮을수록 작음 | von Hellermann 외 1995 PPCF 37 71 | [초록] |
| 같은 효과의 크기(보스턴 경계) | "over-estimate of the impurity ion temperature of no greater than 15%" | C-Mod 경계(조건 미확인) | Brunner 외 2013 PPCF 55 095010 (또는 PSFC 보고서판) | [스니펫] — 원문 위치 미확인, 인용 비권장 |
| 단면적 자료 자체의 불확도 | C⁶⁺ n=8 포획 단면적 불확도 "50% or larger for collision energies below 20 keV/amu" | 반·삼분의 일 에너지 빔 성분 | Bell 외 2010 PoP 17 (NSTX), §II.B (PPPL 보고서판) | [본문] |
| 미세 구조(l 분포)로 인한 선 중심 이동 | C VI 중심 파장 "0.054 Å" 이동, T 100 eV→3 keV, "apparent velocity shift of 3 km/s" | 통계 분포 가정 | Bell 외 2010, §II.A | [본문] (속도에 영향, T에는 정량 없음) |
| Plume(이차 CX) | 중심 시선에서 plume이 배경의 "about half" | NSTX 중심 시선 | Bell 외 2010 | [본문] (가장자리 시선에는 작다고 해석됨 — 내 판단) |
| Zeeman 넓어짐 | 무시하면 "error in measured ion temperature is greater at lower temperatures" | C-Mod 고자기장 | MIT PSFC 보고서 RR-03-9 (2003) | [스니펫], 서지 미확인 |
| Zeeman(이 보고서의 계산) | 정상 Zeeman 근사: Δλ = 4.67×10⁻¹³ λ²[Å] B[G] = 0.235 Å (1.8 T), 0.19 Å (1.47 T). 가산 분산 → T 오프셋 ≈ 7–11 eV(시선 ⊥B, π:σ=2:1:1)에서 15–22 eV(시선 ∥B, σ만). T에 무관한 상수 | KSTAR B_T 1.8 T, 가장자리 R ≈ 2.2 m | [계산] | 문헌 수치 아님 |
| 수동(배경) 방출 | 가장자리 C⁵⁺ 529 nm 선에 수동 방출이 "affecting the line considerably" | AUG 가장자리 | Viezzer 외 2011 PPCF 53 035002 | [초록] (T_i 편향 크기 없음) |
| 배경 분리 방법(KSTAR) | 빔 변조 없이 2-Gaussian fit로 "reasonable accuracy" | KSTAR 중심부 | Lee, Lee, Ko 2020 FST 76 787 | [초록] |
| 노출 시간 안의 변화(ELM) | AUG에서 ELM 주기를 "65 μs"로 분해; ∇T_i 회복은 n_e와 비슷한 시간 척도, pedestal top T는 다음 ELM까지 계속 상승 | AUG type-I ELM | Cavedon 외 2017 PPCF 59 105007 | [초록] |
| ELM 후 KSTAR T_i | "the reduced Ti pedestal remains almost the same after the ELM burst" (t_ELM + 6, 26, 46 ms) | KSTAR #5747 (2011), type-I | Ko 외 2012 KNS 춘계 학술대회 | [본문] (학회 발표문, 동료 심사 아님) |
| 통계 오차 산정 | 통계 오차는 활성 CX 선 fit에서 95 % 신뢰구간으로 산정 | 일반(리뷰) | Mazon 외 2025 NF 65 113001 (Diagnostics 장) | [스니펫] — 해당 절 본문은 못 읽음 |
| 신호 약한 가장자리 | HL-3: 25 ms 노출이 시간 분해능을 제한, 가장자리 채널 표본이 300–1000개라 불확도 큼 | HL-3 | Tian 외 2026 NF 67 016004 | [스니펫] (IOP 봇 차단) |
| Fit 실패 이상치(KSTAR) | 이상치 원인은 "poor results of the fitting to the charge exchange spectra"; 120개 프로파일 중 "60 profiles contain the outliers"; 이상치 43(학습)+19(시험) | KSTAR CES T_i, R = 1.790–2.180 m | Kim 외 2024 NF 64 106052, §4.1·부록 A | [본문] |
| 기기 함수·교정 | 파장·절대 세기 교정을 진공 용기 안에서 수행, 캠페인 중 반복 교정 계획 | KSTAR CES 2010 | Ko 외 2010 RSI 81 10D740 | [초록] (크기 없음) |
| 대기압에 따른 파장 이동 | 기압 2 % 변화 → "0.013 Å", 속도 0.74 km/s | 속도에만 영향 | Bell 외 2010 | [본문] |

---

## 3. KSTAR CES에 대해 문헌이 말하는 것과 말하지 않는 것

### 확인된 것
- **선과 반응**: C VI n=8→7, 529 nm, 가열 중성 빔과 탄소 불순물의 전하 교환 (Ko 외 2010 RSI [초록]; Lee 외 2010 KNS [본문]; Jang 외 2021 RSI [초록]).
- **배경 분리**: 설계는 "modulated neutral beam injection and background subtraction"(Ko, Oh, Kwon 2010 IEEE TPS 38 996 [초록]). 초기 구성은 빔을 보는 활성 카세트(M 포트)와 플라스마만 보는 배경 카세트(A 포트)를 두었다(Lee 외 2010 KNS [본문]). 2011년 운전 예: 5 Hz 변조, "190 msec on and 10 msec off"(Ko 외 2012 KNS [본문]). 즉 **빔이 꺼진 10 ms 프레임 = blip**이 배경 측정이다. 이것은 저장소 `docs/paper/file_rules.md`가 자료에서 찾은 "20–40 ms 간격 = 빔 없는 프레임 누락"과 맞는다.
- **NN-CES의 동기**: KSTAR는 별도 진단 빔이 아니라 주 가열 빔을 변조하므로 플라스마를 교란할 수 있고, NN-CES는 변조 없이 원 스펙트럼에서 T_i·V_φ를 낸다(Lee 외 2026 FED 222 115518 [스니펫: UNIST 저장소 초록]). Lee 외 2020 FST(2-Gaussian fit)도 변조 없는 분석을 제시.
- **시간 분해능**: "10 msec (time resolution of the CES)"(Kim 2024 [본문]); 2012년 고처리량 분광기로 200 ms → 약 10 ms로 개선(Ko 외 2012 KNS [본문]); 2024년 KSTAR 진단 개요 발표에서 CES(toroidal) 32채널, 100 Hz, "routine operation"(KSTAR 진단 개요, ENEA 2024 발표 슬라이드 13·15 [본문], 발표자 미상).
- **공간 분해능**: 설계 요구 가장자리 5 mm, 정확도 요구 "accuracy of 10%", 온도 범위 100 eV–20 keV (Lee 외 2010 KNS [본문]); 2012년 가장자리 시선 간격 5 mm 달성(Ko 외 2012 KNS [본문]).
- **이상치**: Levenberg–Marquardt fit 실패가 T_i 이상치를 만들고 분석 루틴을 "currently being improved"(Kim 2024 [본문]).
- **교차 검증**: 중심부 T_i·회전은 XICS와 "reasonable agreement"(Lee 외 2016 FST 69 555 [초록]; Lee, Ko 2020 FST 76 942 [초록]). **가장자리 교차 검증은 찾지 못함.**
- **불순물-주 이온 차이 인식**: KSTAR 그룹 스스로 가장자리에서 탄소 속도가 주 이온 속도와 "much different"일 것이라 적음(Ko 외 2012 KNS [본문]; 속도 이야기이고 온도 수치는 없음).
- **장래 W 벽 문제**: 2024 개요 슬라이드 23에 "Nuisance W lines in CES/CXRS", "Multi-Gaussian fit for CES data with W divertor". (2022 캠페인 자료가 W 디버터 이전인지는 이 조사에서 확인하지 못함.)

### 문헌으로 알 수 없는 것
- 10 ms 프레임마다 배경을 무엇으로 빼는가: blip 프레임 배경을 다음 blip까지 보간하는지, 배경 시선을 쓰는지, 2-Gaussian fit인지 — 2022 캠페인의 표준 분석 절차를 기술한 동료 심사 문헌을 찾지 못함.
- 행별(프레임별) fit 불확도의 전형적 크기(eV 또는 %). KSTAR 논문 어디에도 숫자 없음.
- Zeeman·미세 구조·단면적 보정이 KSTAR 표준 분석에 들어가 있는지. (Jang 2021 공저자에 von Hellermann이 있으나 보정 적용 여부는 초록에 없음.)
- 가장자리 측정선의 정확한 위치와 그 시선의 적분 길이.
- 같은 값이 반복되는(유지값) 행의 생성 규칙.

---

## 4. 출판된 값과 견줄 수 있는 것

- **Heo 외 2025 NF 65 036003** [본문, IOP 페이지]:
  - T_i·V_φ는 "measured by charge exchange spectroscopy (CES)"(§2).
  - 이온 온도는 "only available between two sequential beam blips"(§3). 그림 2(#31923, B_T = 1.8 T)는 3.5 s와 5.0 s blip 사이의 T_i,core·T_i,edge 시간 변화를 보이며, P_NBI가 4.0 s에 3.5 → 4.0 MW로 오를 때 T_i가 오른다.
  - 그림 3: "#31921 at 5.40 s"(FIRE mode)와 "#31921 at 8.05 s"(H-mode)의 재구성 프로파일, 사각 표지는 "the mean of the raw data points over the averaging time interval", 오차 막대는 "the standard deviation of these raw data points". 평균 구간 길이는 본문에 숫자로 없음.
  - 가장자리 대리값 T_i,edge는 ρ_N ≈ 0.94에서 취함(그림 5, 672점 자료). FIRE 중심 T_i 약 9 keV.
  - **가장자리 T_i의 숫자 값은 본문 텍스트에 없음.** 그림 축 눈금은 텍스트 추출로 읽지 못함(그림 이미지 확인 필요). 방전 31921의 H-mode 전환 시각·ELM 유무도 본문에 명시되지 않음.
- **비교 방법 제안**: 우리 10 ms 파일의 31921 5.40 s·8.05 s, 31923 3.5–5.0 s 구간 값과 그림 3·2를 이미지로 대조. 우리 측정선이 ρ_N ≈ 0.94인지(명목 ρ 0.8로 기록됨 — 저장소 메모)는 다르므로 값이 같을 필요는 없고, **시간 추이(4.0 s 전력 상승 뒤 증가)와 blip 위치**가 맞는지가 검증 대상이다.

---

## 5. 원고에 넣을 수 있는 문장 후보 (영어)

1. "The target is the temperature of C⁶⁺ impurity ions inferred from the Doppler width of the C VI (n = 8→7, 529 nm) charge-exchange line excited by a modulated heating beam, with the background taken from beam-off frames \citep{Ko2010KstarCesSystem,Ko2010ChargeExchange}; at KSTAR this analysis yields one value per 10 ms frame \citep{Kim2024KineticProfile}."
2. "At the pedestal top the impurity and main-ion temperatures are close but not identical: at ASDEX Upgrade the deuterium temperature exceeded the boron temperature by about 100 eV before and 50–60 eV after extended corrections, while treating the impurity temperature as the ion temperature was found to be generally valid for transport analysis \citep{CanoMegias2025MultiIon}; larger differences are reported in the steep-gradient region and at the separatrix \citep{Haskey2018MainIon}."
3. "Atomic-physics effects (energy-dependent charge-exchange cross-sections, fine structure, Zeeman splitting) mainly shift the apparent velocity and alter the apparent width by amounts that decrease with ion temperature \citep{vonHellermann1995CrossSection,Bell2010PoloidalNstx}; Zeeman and fine-structure broadening add a nearly temperature-independent width, so they enter as an approximately constant offset in the inferred temperature."
4. "Failures of the spectral fit are the dominant source of outliers in KSTAR CES ion-temperature profiles \citep{Kim2024KineticProfile}; we therefore treat values above 3 keV at the pedestal-side chord as fit failures (section~X)."
5. "Because every estimator and baseline is scored against the same reported values, a systematic offset that is constant in time cancels in the skill comparisons; what our results do not establish is the accuracy of the reported value as the main-ion temperature, which requires main-ion charge-exchange measurements \citep{Grierson2016MainIonCer,CanoMegias2025MultiIon}." (문장 5의 앞 절반은 수학적 사실이지 문헌 사실이 아님 — 오프셋이 시간에 따라 변하면 성립하지 않는다는 단서를 원고에 함께 둘 것.)

### BibTeX (Crossref/OpenAlex로 확인된 것만)

```bibtex
@article{Ko2010KstarCesSystem,
  title={{KSTAR} Charge Exchange Spectroscopy System},
  author={W. H. Ko and S. Oh and M. Kwon},
  journal={IEEE Transactions on Plasma Science},
  year={2010}, volume={38}, pages={996},
  doi={10.1109/TPS.2010.2042182}
}
@article{CanoMegias2025MultiIon,
  title={Investigation of multi-ion heat and neoclassical transport using new edge main ion measurements at {ASDEX} {Upgrade}},
  author={P. Cano-Megias and E. Viezzer and R. M. McDermott and C. Angioni and E. Fable and others},
  journal={Nuclear Fusion},
  year={2025}, volume={65}, pages={096027},
  doi={10.1088/1741-4326/adf3cb}
}
@article{Haskey2018MainIon,
  title={Main ion and impurity edge profile evolution across the {L}- to {H}-mode transition on {DIII-D}},
  author={S. R. Haskey and others},
  journal={Plasma Physics and Controlled Fusion},
  year={2018}, volume={60}, pages={105001},
  doi={10.1088/1361-6587/aad702}
}
@article{Grierson2016MainIonCer,
  title={High resolution main-ion charge exchange spectroscopy in the {DIII-D} {H}-mode pedestal},
  author={B. A. Grierson and others},
  journal={Review of Scientific Instruments},
  year={2016}, volume={87}, pages={11E545},
  doi={10.1063/1.4960604}
}
@article{vonHellermann1995CrossSection,
  title={Analytical approximation of cross-section effects on charge exchange spectra observed in hot fusion plasmas},
  author={M. von Hellermann and P. Breger and J. Frieling and R. K{\"o}nig and W. Mandl and A. Maas and H. P. Summers},
  journal={Plasma Physics and Controlled Fusion},
  year={1995}, volume={37}, number={2}, pages={71--94},
  doi={10.1088/0741-3335/37/2/002}
}
@article{Isler1994Overview,
  title={An overview of charge-exchange spectroscopy as a plasma diagnostic},
  author={R. C. Isler},
  journal={Plasma Physics and Controlled Fusion},
  year={1994}, volume={36}, pages={171--208},
  doi={10.1088/0741-3335/36/2/001}
}
@article{Bell2010PoloidalNstx,
  title={Comparison of poloidal velocity measurements to neoclassical theory on the {National Spherical Torus Experiment}},
  author={R. E. Bell and R. Andre and S. M. Kaye and R. A. Kolesnikov and B. P. LeBlanc and G. Rewoldt and W. X. Wang and S. A. Sabbagh},
  journal={Physics of Plasmas},
  year={2010}, volume={17}, number={8},
  doi={10.1063/1.3478571}
  % 논문 번호(쪽)는 메타데이터에 없어 미확인 — 넣기 전에 Crossref로 채울 것
}
@article{Viezzer2011PassiveEdge,
  title={Investigation of passive edge emission in charge exchange spectra at the {ASDEX} {Upgrade} tokamak},
  author={E. Viezzer and T. P{\"u}tterich and R. Dux and A. Kallenbach},
  journal={Plasma Physics and Controlled Fusion},
  year={2011}, volume={53}, pages={035002},
  doi={10.1088/0741-3335/53/3/035002}
}
@article{Viezzer2012HighResolution,
  title={High-resolution charge exchange measurements at {ASDEX} {Upgrade}},
  author={E. Viezzer and T. P{\"u}tterich and R. Dux and R. M. McDermott},
  journal={Review of Scientific Instruments},
  year={2012}, volume={83}, pages={103501},
  doi={10.1063/1.4755810}
}
@article{Lee2020TwoGaussian,
  title={Two-Gaussian Fitting Method for Charge Exchange Spectroscopy on Measurements of Ion Temperature and Toroidal Plasma Flow Velocity in the {KSTAR} Tokamak},
  author={H. H. Lee and J. K. Lee and W. H. Ko},
  journal={Fusion Science and Technology},
  year={2020}, volume={76}, pages={787},
  doi={10.1080/15361055.2020.1790712}
}
@article{Lee2011TwoGrating,
  title={Development of two-grating spectrometer for the charge exchange spectroscopy system on {KSTAR}},
  author={Hyungho Lee and Eun-ji Song and Young-dong Park and Soo-ghee Oh and W. H. Ko},
  journal={Review of Scientific Instruments},
  year={2011}, volume={82}, pages={063510},
  doi={10.1063/1.3606438}
}
@article{Brunner2013BoundaryTi,
  title={An assessment of ion temperature measurements in the boundary of the {Alcator C-Mod} tokamak and implications for ion fluid heat flux limiters},
  author={D. Brunner and B. LaBombard and R. M. Churchill and others},
  journal={Plasma Physics and Controlled Fusion},
  year={2013}, volume={55}, pages={095010},
  doi={10.1088/0741-3335/55/9/095010}
}
```
(이미 refs.bib에 있는 것: `Ko2010ChargeExchange`, `Lee2026NeuralNetworkbased`, `Heo2025ExperimentalIdentification`, `Kim2024KineticProfile`, `Jang2021DevelopmentUltrafast`, `Tian2026RealTimeCxrs`, `Cavedon2017Pedestal`, `Viezzer2020Dynamics`.)

**refs.bib 정정 후보**: `Jang2021DevelopmentUltrafast`의 셋째 저자가 refs.bib에는 "Kimin Li"인데 OpenAlex 메타데이터는 "Yingying Li"이고 공저자에 M. von Hellermann, Y. Shi, Y. S. Hwang이 있다. Crossref로 다시 대조 권장.

---

## 6. 이 저장소의 자료로 직접 재 볼 수 있는 계통 오차 점검

1. **가산 오프셋 불변성 점검(가장 값이 큼, CPU 몇 분)**: Zeeman·미세 구조·기기 함수·불순물-주 이온 차이의 상수 부분은 시간에 무관한 오프셋이다. 표적에서 상수 c(예: ±20, ±50, ±100 eV) 또는 비례 인자(0.9, 1.1)를 빼거나 곱한 뒤 고정된 예측·기준선 skill을 다시 계산한다. **가르는 것**: skill이 상수 오프셋에 불변(정의상 MAE 차이는 불변이어야 함)인지, 비례 인자에는 얼마나 민감한지 — "보고값 vs 실제값" 비판 중 어떤 부분이 결론을 바꿀 수 없는지를 원고에서 한 문장으로 닫을 수 있다.
2. **blip 경계의 단차 vs 물리 변화**: blip(빔 없는 프레임 누락, 20–40 ms 간격) 직후 약 90 ms 동안 T_i가 낮게 나오는 현상(file_rules.md)이 배경 분리의 산물인지 실제 가열 손실인지. 같은 구간 ECEI 수준(전자 온도 대리)의 하강 시간 척도·크기와 T_i 하강을 방전별로 비교; 원시 µs BES로 빔 꺼짐 시각을 정확히 짚는다. **가르는 것**: ECEI가 같은 시간 척도로 떨어지면 물리(가열 중단), ECEI 변화 없이 T_i만 계단처럼 바뀌면 배경·분석 산물. 산물이면 blip 뒤 프레임은 평가에서 따로 다뤄야 한다.
3. **신호 세기와 무작위 오차**(B.40 K2와 같은 축): 광자 잡음이면 T 분산 ∝ 1/(활성 CX 세기). 빔 방출(BES 직류 수준)을 세기 대리로 쓰고, B.40의 백색 잡음 상·하한을 세기 분위별로 계산. **가르는 것**: 기울기 −1(광자 잡음 지배) vs 평평(fit 모형·배경 오차 지배). 후자면 계통 성분이 크다는 근거.
4. **노출 안의 ELM 평균**: 12개 방전의 원시 µs BES·ECEI에서 ELM 후보 버스트를 찾아, 10 ms 프레임을 "버스트 포함/미포함"과 "버스트 뒤 경과 시간"으로 나누고 T_i 수준·이웃 프레임 대비 편차를 비교. **가르는 것**: 버스트 포함 프레임의 T_i가 체계적으로 낮은지(노출 평균 편향), 그리고 ELM 뒤 회복 시간(KSTAR 2012 발표문은 ELM 뒤 T_i pedestal이 46 ms까지 거의 그대로라고 보고)이 우리 자료에서 재현되는지. B.41 U1과 겹치므로 그 결과를 이 문헌과 같이 인용하면 된다.
5. **fit 실패 표지 규칙의 근거**: T_i > 3 keV 행과 유지값 행의 비율을 신호 세기(3번 대리)와 빔 상태에 대해 그린다. **가르는 것**: 이상치가 약한 신호에 몰리면 Kim 2024의 "fit 실패" 해석과 일치, 아니면 다른 원인(배경 과대 차감 등).
6. **MAST에서 보고 오차와의 대조**(B.40 M1 확장): 공개 MAST CXRS는 fit 오차를 함께 준다. 같은 반경에서 Thomson T_e와 CXRS T_i의 비를 pedestal top 근처에서 계산하고, 보고 오차 대비 차이를 본다. **가르는 것**: 고충돌 pedestal top에서 T_i ≈ T_e가 보고 오차 안에서 성립하면, "CXRS 보고값이 이온 온도에 가깝다"는 진술에 장치 하나 분량의 독립 근거가 생긴다(KSTAR에는 같은 시선 Thomson 비교가 자료상 없음).
7. **Heo 2025와의 대조**: 31921·31923의 blip 위치와 4.0 s 전력 상승 뒤 T_i 상승이 우리 10 ms 파일에서 같은 시각에 보이는지(4절). 값 자체가 아니라 시각·방향 대조.

---

## 7. 확인하지 못한 것과 확신이 낮은 판단

- **Lee 2026 NN-CES(FED) 본문**: 열지 못함(Elsevier). 정량 정확도·10 ms 값 산출 절차는 미확인. UNIST 저장소 초록 발췌만 봄.
- **Tian 2026 HL-3**: IOP 봇 차단으로 본문 못 봄. 25 ms 노출·가장자리 표본 수 문장은 검색 스니펫.
- **Brunner 2013의 "no greater than 15%"**: 검색 스니펫에서만 봄. PSFC 사본은 인증서 문제로 열지 못함(인증서 검사 우회는 하지 않음). 원고 인용 비권장.
- **DIII-D "~50 % lower"**: APS 학회 초록. 동료 심사 논문(Haskey 2018) 초록은 분리면에서 낮다는 정성 문장만 있음. 인용하려면 "conference abstract"로 표시하거나 쓰지 말 것.
- **Zeeman 수치**: 이 보고서의 계산(정상 Zeeman 근사, 시선-자기장 각도 가정). KSTAR toroidal CES 시선이 B에 대해 어떤 각인지 확인하지 못해 7–22 eV 범위로 남김. 미세 구조 폭의 수치는 구하지 못함(Bell 2010의 0.054 Å는 중심 이동).
- **C-Mod Zeeman 보고서(PSFC RR-03-9, 2003)**: 저자·제목 미확인.
- **Mazon 2025 Diagnostics 장**: CXRS 절 본문은 읽지 못함. "95 % 신뢰구간" 문장은 스니펫.
- **Heo 2025 그림 값**: 텍스트에 가장자리 T_i 숫자가 없어 그림 이미지 판독이 필요(미수행).
- **KSTAR 2022 캠페인의 W 디버터 여부**: 미확인.
- **Bell 2010 PoP 논문 번호**: 메타데이터에 없음(미확인).
- **판단(확신 낮음)**: pedestal top(명목 ρ 0.8, T_i 중앙값 약 500 eV)의 탄소 T_i는 AUG 결과에 비추어 주 이온 T와 수십–100 eV 안에서 다를 수 있으나, 그 차이는 대체로 느리게 변하므로 nowcasting skill 비교에는 1차적으로 상쇄된다는 것. 이것은 AUG 한 장치의 결과를 KSTAR로 옮긴 추론이다.
- 이 조사에서 IOP 본문(Heo 2025, Kim 2024, Cano-Megias 2025)은 열렸고, Tian 2026에서 봇 검증 페이지가 나타나 중단했다.

## 출처 URL (열람한 것)
- https://iopscience.iop.org/article/10.1088/1741-4326/adacfc (Heo 2025)
- https://iopscience.iop.org/article/10.1088/1741-4326/ad7304 (Kim 2024)
- https://iopscience.iop.org/article/10.1088/1741-4326/adf3cb (Cano-Megias 2025)
- https://iopscience.iop.org/article/10.1088/1741-4326/adfc7c (Mazon 2025, 앞부분만)
- https://www.osti.gov/servlets/purl/981659 (Bell 2010 PPPL 보고서판)
- https://kns.org/files/pre_paper/8/428이형호.pdf (Lee 외 2010 KNS)
- https://kns.org/files/pre_paper/4/663고원하.pdf (Ko 외 2012 KNS)
- https://agenda.enea.it/event/1065/contributions/3517/attachments/905/2118/202411ENEA_KSTARDiag.pdf (KSTAR 진단 개요 2024)
- https://meetings-archive.aps.org/dpp/2016/co4/14/ (Haskey 외 APS 2016)
- https://scholarworks.unist.ac.kr/handle/201301/88741 (Lee 2026 저장소 기록)
- OpenAlex·Crossref API 메타데이터(각 DOI)
