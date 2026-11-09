# 통계 방법론 문헌 점검 — KSTAR CES 나우캐스팅 논문 (2026-11-09)

작성: lit-statistics 에이전트. 조사 경로: OpenAlex API(DOI 필터·검색; 일일 무료 예산이 중간에 소진되어 이후는 불가), Crossref API, arXiv API(abstract 검색·id 조회), 공개 PDF 본문(PyMuPDF 추출), WebSearch. Semantic Scholar는 429로 전부 실패.

표기 규칙
- **[검증]** DOI 또는 arXiv id가 실제로 해석되고 서지(제목·저자·연도·학술지)가 일치함을 API로 확인.
- **[부분검증]** 초록·검색 스니펫만 확인, 본문 내용은 미확인.
- **[미발견]** 네 경로(OpenAlex·Crossref·arXiv·WebSearch)에서 해당 사례를 찾지 못함. "존재하지 않음"의 증명이 아님. OpenAlex 전문(fulltext) 검색은 예산 소진으로 이번에는 못 돌렸으므로, 핵융합 논문 본문에 묻힌 용례는 놓쳤을 수 있음.
- 판정 어휘: **표준(기존)** / **타 분야에서 차용·핵융합에서는 미발견** / **신규**.

---

## Q1. 사전등록(pre-registration)·등록 보고서(registered report)·블라인드 분석

### 발견 사항
- **기계학습 사전등록 워크숍은 실재하며 두 번 열렸다.**
  - *NeurIPS 2020 Workshop on Pre-registration in Machine Learning*, PMLR vol. 148 (편집 Bertinetto, Henriques, Albanie, Paganini, Varol; 2020-12-11 개최, 2021-07 출판). **[검증]** https://proceedings.mlr.press/v148/
  - *NeurIPS 2021 Workshop on Pre-registration in Machine Learning*, PMLR vol. 181 (편집 Albanie, Henriques, Bertinetto, Hernández-García, Doughty, Varol; 2021-12-13 개최, 2022-10 출판). **[검증]** https://proceedings.mlr.press/v181/
  - 형식: 제안서(가설·실험 설계)를 먼저 심사·채택하고 결과는 뒤에 붙임. 2022년 이후 후속 워크숍은 못 찾았음(**[미발견]**) — 즉 ML 주류 학회의 정규 트랙으로 정착하지는 않았다.
- **예측 모델링 사전등록 템플릿**: Hofman, Chatzimparmpas, Sharma, Watts, Hullman, "Pre-registration for Predictive Modeling", arXiv:2311.18807 (2023). 데이터 의존적 의사결정·테스트셋 재사용을 사전등록으로 막자는 제안 + 경량 템플릿 + ML 연구자 질적 연구. **[검증]** DOI 10.48550/arXiv.2311.18807 (OpenAlex 인용 27).
- **등록 보고서(Registered Reports) 원전**
  - Nosek & Lakens, "Registered Reports: A Method to Increase the Credibility of Published Results", *Social Psychology* 45(3), 137–141 (2014). **[검증]** DOI 10.1027/1864-9335/a000192 (인용 781).
  - Chambers & Tzavella, "The past, present and future of Registered Reports", *Nature Human Behaviour* 6, 29–42 (2022; 온라인 2021). **[검증]** DOI 10.1038/s41562-021-01193-7 (Crossref 인용 338).
- **물리학의 블라인드 분석**
  - Klein & Roodman, "Blind Analysis in Nuclear and Particle Physics", *Annual Review of Nuclear and Particle Science* 55, 141–163 (2005). **[검증]** DOI 10.1146/annurev.nucl.55.090704.151521 (인용 174).
  - MacCoun & Perlmutter, "Blind analysis: Hide results to seek the truth", *Nature* 526, 187–189 (2015). **[검증]** DOI 10.1038/526187a (인용 233).
  - Junk & Lyons, "Reproducibility and Replication of Experimental Particle Physics Results", *Harvard Data Science Review* (2020). **[검증]** DOI 10.1162/99608f92.250f995b.
  - 입자물리 ML 패턴 발견을 사전등록 대상으로 층화하자는 제안: Brilliant, "Stratification Criteria for Machine Learning Pattern Discovery in Particle Physics", arXiv:2609.06015, *London Journal of Physics* 3(1) (2026), DOI 10.69710/ljp.v3i1.17825. **[부분검증]** (arXiv journal-ref만 확인, DOI 해석은 미확인).
- **계산 모델링의 사전등록(인지과학)**: Crüwell & Evans, "Preregistration in diverse contexts: a preregistration template for the application of cognitive models", *Royal Society Open Science* 8, 210155 (2021). **[검증]** DOI 10.1098/rsos.210155. "계산신경과학" 고유의 등록보고서 지침은 따로 못 찾았음(**[미발견]**; 학술지 차원에서는 *eNeuro*·*Cortex* 등이 RR 트랙을 운영하나 이번에 확인하지 않음).
- **ML에서 사전등록 '실행' 사례(제안이 아니라 적용)**: arXiv에는 2026년 들어 "pre-registered decision rules" 아래 결과를 보고하는 논문이 여러 편 보임(예: arXiv:2606.15600, 2610.14112, 2608.13087, 2606.00524, 2608.18261 — 모두 CS/ML 응용). **[부분검증]** 즉 "규칙·예측을 먼저 써두고 판정을 공개한다"는 관행은 2025–26년에 ML 응용 쪽에서 확산 중.
- **핵융합·플라즈마 물리에서의 사전등록**: OpenAlex 검색, arXiv 검색, WebSearch 전부에서 **[미발견]**. 블라인드 분석조차 핵융합 ML 논문에서 언급된 사례를 찾지 못함. 핵융합 ML 평가의 체계화는 벤치마크 쪽(DisruptionBench, Spangher et al., *J. Fusion Energy* 2025, DOI 10.1007/s10894-025-00495-2 **[검증]**)으로 진행 중이며 사전등록과는 다른 축.

### 우리 사용법과의 대응
우리는 (i) 배치마다 규칙·신뢰도 달린 예측·공개사항을 **커밋으로 시점 고정**하고, (ii) 예측 장부를 held/missed로 채점하며, (iii) 통과 못 한 가설(H28-2, H28-3 등)을 그대로 기록한다. 이는 Nosek & Lakens의 등록보고서 논리와 Klein & Roodman의 "결과를 보기 전에 분석을 고정"하는 블라인드 분석 논리를 합친 형태이고, Hofman et al.(2023)이 예측 모델링에 제안한 바로 그 템플릿의 실행 사례다.

### 판정
**타 분야에서 차용·핵융합에서는 미발견.** 심리학/입자물리의 표준을 ML 예측 모델링(Hofman 2023, NeurIPS 2020·2021 워크숍)이 받아들였고, 우리는 그것을 핵융합 진단 ML에 적용한 첫 사례일 가능성이 높다(단, 전문 검색 미실시라 "첫"은 단정 금지 — "문헌에서 선례를 찾지 못했다"로 쓸 것).

---

## Q2. 핵융합 ML 논문의 성능 불확실성 보고 관행 — shot 분할, 신뢰구간, 방전 단위 클러스터 부트스트랩

### 발견 사항(대표 논문별, 본문 또는 초록 확인)
| 논문 | 분할 | 성능 불확실성 보고 | 검증 수준 |
|---|---|---|---|
| Kates-Harbeck, Svyatkovskiy, Tang, *Nature* 568, 526 (2019), DOI 10.1038/s41586-019-1116-4 | shot 단위 train/test | 초록·검색 스니펫에 AUC 단일값만; 분할 반복 CI 언급 없음 | [부분검증] |
| Rea, Montes, Erickson et al., *Nucl. Fusion* 59, 096016 (2019), DOI 10.1088/1741-4326/ab28bf | shot-by-shot 알람 시뮬레이션 | 성공률·경고시간 점추정; CI 없음(초록) | [부분검증] |
| Montes, Rea, Granetz et al., *Nucl. Fusion* 59, 096015 (2019), DOI 10.1088/1741-4326/ab1df4 | shot 단위 DB | 점추정(초록) | [부분검증] |
| Zhu, Rea, Montes et al., *Nucl. Fusion* 61, 026007 (2021), DOI 10.1088/1741-4326/abc664 (arXiv:2007.01401) | shot 단위 | **본문 확인**: 초기화 seed 12개 앙상블 평균을 "예측기"로 쓰고 AUC는 단일값; K-fold CV는 RF 하이퍼파라미터 탐색에만 사용; CI·유의성 검정 없음 | [검증] |
| Churchill, Tobias, Zhu, *Phys. Plasmas* 27, 062510 (2020), DOI 10.1063/1.5144458 | — | F1 ≈ 91 % 단일값(초록) | [부분검증] |
| Vega, Murari, Dormido-Canto et al., *Nature Physics* 18, 741 (2022), DOI 10.1038/s41567-022-01602-2 | shot 단위 적응학습 | 초록에 CI 없음 | [부분검증] |
| Abbate, Conlin, Kolemen, *Nucl. Fusion* 61, 046027 (2021), DOI 10.1088/1741-4326/abe08d | 캠페인(2013–18) | 초록에 불확실성 언급 없음 | [부분검증] |
| Boyer & Chadwick, *Nucl. Fusion* 61, 046024 (2021), DOI 10.1088/1741-4326/abe08b | — | 앙상블 예측 불확도(표본별), 성능 CI는 미확인 | [부분검증] |
| Jalalvand, Abbate, Conlin et al., *IEEE TNNLS* 33, 2630 (2022), DOI 10.1109/TNNLS.2021.3085504 | — | 미확인 | [검증: 서지만] |
| Char, Chung, Abbate, Kolemen, Schneider, "Full Shot Predictions for the DIII-D Tokamak via Deep Recurrent Networks", arXiv:2404.12416 (2024) | **shot 단위·시간순** 90/5/5 (최근 5 % shot이 test) | **본문 확인**: "mean over four different models with different random seeds, shaded area = standard error" — 불확실성 축은 **init seed**이지 shot 재표집이 아님; 단순 baseline(persistence) 없음; 보정은 coverage·miscalibration area | [검증] |
| Kim, Lee et al., "Enhancing Disruption Prediction through Bayesian Neural Network in KSTAR", arXiv:2312.12979 (2023) | 301 shot, 무작위 64/16/20 | **본문 확인**: 단일 분할 점추정, CI·검정 없음; 불확실성은 표본별 epistemic | [검증] |
| Poels, Venturini, Pau et al., *Nucl. Fusion* 65 (2025), DOI 10.1088/1741-4326/adf349 (arXiv:2502.17397) | **실험 주제(mission) 단위로 묶은 fold** — shot보다 큰 클러스터로 분할 | **본문 확인**: Cohen's κ + ECE; fold는 앙상블 다양성용이지 CI 추정용이 아님 | [검증] |
| Spangher et al., DisruptionBench, *J. Fusion Energy* 44 (2025), DOI 10.1007/s10894-025-00495-2 | 3 장치 × 3 체제 | 벤치마크; CI 방식은 미확인 | [부분검증] |
| Sorokin et al., arXiv:2610.24525 (2026) | — | 다중 seed 앙상블·Laplace·MH 비교를 **CRPS·coverage**로 채점 | [검증: arXiv] |
| Vakhshouri et al., arXiv:2611.06597 (2026) | — | conformal 오차막대의 coverage(표본별 보정) | [검증: arXiv] |

- **공통 패턴**: shot(또는 그보다 큰 mission/캠페인) 단위 분할은 2019년 이후 사실상 표준이다. 그러나 성능의 **표집 불확실성**은 (a) 보고하지 않거나, (b) init seed 몇 개의 표준오차로 대체한다. shot을 재표집 단위로 삼는 **클러스터 부트스트랩 CI**, 또는 모델 간 **짝지은(paired) 유의성 검정**을 쓴 핵융합 논문은 arXiv 초록 검색("tokamak" AND "bootstrap" AND "confidence interval" → 0건; "disruption prediction" AND "confidence intervals" → 0건)과 WebSearch에서 **[미발견]**. 검색 결과에서 "bootstrap"은 전부 bootstrap current(물리 용어)였다.
- 핵융합 외 플라즈마/천체 분야에서는 단위(shot) 재표집 부트스트랩 사례가 있음(Kepler 블레이저 광도곡선 "shot" 195개를 재표집, arXiv:1611.06311) — 방법은 동일하나 분야가 다름.

### 우리 사용법과의 대응
우리는 방전(shot)을 교환가능한 단위로 보고 skill = 1 − ΣSE_model/ΣSE_ref 를 **방전-클러스터 짝지은 부트스트랩(B = 10,000)**으로 구간추정한다. 이는 클러스터 부트스트랩의 교과서적 적용이다: Davison & Hinkley, *Bootstrap Methods and their Application* (CUP 1997, DOI 10.1017/CBO9780511802843 **[검증]**, §3.8 계층 자료); Field & Welsh, "Bootstrapping clustered data", *JRSS-B* 69, 369–390 (2007), DOI 10.1111/j.1467-9868.2007.00593.x **[검증]**; Cameron, Gelbach & Miller, "Bootstrap-based improvements for inference with clustered errors", *Rev. Econ. Stat.* 90, 414–427 (2008), DOI 10.1162/rest.90.3.414 **[검증]**. 비율형 통계량(두 합의 비)을 부트스트랩하는 것도 Davison & Hinkley의 표준 범위.

### 판정
**타 분야에서 차용·핵융합에서는 미발견.** shot 단위 분할 자체는 핵융합 표준. shot 단위 클러스터 부트스트랩 CI와 짝지은 검정은 통계학·계량경제·신경과학 표준을 들여온 것이고 핵융합 ML 선례를 찾지 못했다.

---

## Q3. 반복 k-fold의 분산 보정 — Nadeau–Bengio, Bouckaert–Frank, Dietterich 5×2cv, Bates–Hastie–Tibshirani

### 발견 사항
- Nadeau & Bengio, "Inference for the Generalization Error", *Machine Learning* 52, 239–281 (2003). **[검증]** DOI 10.1023/A:1024068626366, 인용 1,242(OpenAlex). 겹치는 학습집합 때문에 생기는 분산 과소추정을 (1/J + n₂/n₁) 인자로 보정.
- Bouckaert & Frank, "Evaluating the Replicability of Significance Tests for Comparing Learning Algorithms", *PAKDD 2004*, LNCS 3056, 3–12. **[검증]** DOI 10.1007/978-3-540-24775-3_3, 인용 470. 10×10 반복 k-fold + NB 보정 t 검정을 재현성 기준으로 추천.
- Dietterich, "Approximate Statistical Tests for Comparing Supervised Classification Learning Algorithms", *Neural Computation* 10, 1895–1923 (1998). **[검증]** DOI 10.1162/089976698300017197, 인용 4,042. 5×2cv 짝지은 t 검정. 확장: Alpaydın, "Combined 5×2 cv F Test", *Neural Computation* 11, 1885–1892 (1999), DOI 10.1162/089976699300016007 **[검증]**, 인용 439.
- Bates, Hastie & Tibshirani, "Cross-Validation: What Does It Estimate and How Well Does It Do It?", *JASA* 119, 1434–1445 (2024; 온라인 2023). **[검증]** DOI 10.1080/01621459.2023.2197686, 인용 612. CV 오차는 "이 학습집합의 모델"이 아니라 평균 모델의 오차를 추정하며, 단순 CV 표준오차 구간은 과소포함 → nested CV 구간 제안.
- Demšar, "Statistical Comparisons of Classifiers over Multiple Data Sets", *JMLR* 7, 1–30 (2006). **[검증]** (DOI 없음, JMLR 페이지 확인).
- **대규모 비교 벤치마크**: Schulz-Kümpel, Fischer, Hornung, Bischl, Hothorn, Nagler, "Constructing Confidence Intervals for 'the' Generalization Error — a Comprehensive Benchmark Study", arXiv:2409.18836 (2024). **[검증]** 13개 CI 방법 × 19 데이터셋. **결론 본문 확인**: 소표본(n ≤ 100)에서는 Nested CV(외부 반복 ≥ 25, K=5) 또는 Conservative-Z; 대표본에서는 **Corrected Resampled-T(= Nadeau–Bengio 보정, 분할비 0.9, 반복 ≥ 25)** 를 추천. 모든 방법이 어떤 시나리오에서는 실패한다고 명시. → NB 보정은 2024년 기준 "추천 집합" 안에 있음.
- 주의 보고(부분검증): WebSearch 스니펫 하나(arXiv:2604.04199, 2026)는 NB 보정의 실제 포함률이 명목 95 %에 대해 약 76 %였다고 하나 arXiv 초록에는 해당 문장이 없어 **본문 미확인**. 인용하려면 본문을 직접 봐야 함.
- 소프트웨어: `correctR`(CRAN, DOI 10.32614/CRAN.package.correctR), `mlr3inferr`(CRAN, DOI 10.32614/CRAN.package.mlr3inferr) — NB 보정이 패키지화된 표준임을 보여줌. **[검증]**
- **핵융합/플라즈마에서 보정된 반복 CV 구간**: OpenAlex·arXiv("tokamak" AND "corrected resampled" → 0건)·WebSearch 모두 **[미발견]**. 핵융합 ML에서 CV는 하이퍼파라미터 탐색(Zhu 2021) 또는 앙상블 다양성(Poels 2025) 용도이지 구간추정 용도로 쓰인 사례를 찾지 못함.

### 우리 사용법과의 대응
"4/4 분할 승리" 집계는 분할 간 상관을 무시하는 **횟수 세기**였다. 이를 (i) shot 단위 반복 5-fold + NB 보정 표준오차(= Bouckaert–Frank가 추천하고 Schulz-Kümpel 2024가 대표본에서 1순위로 꼽은 형태), (ii) 풀링된 out-of-fold 예측에 대한 2수준 부트스트랩(Q4)으로 대체한 것은 Bates et al.(2024)의 "단순 CV 구간은 과소포함" 지적에 대한 정합적 대응이다. 단, 우리 1차 통계량이 (ii)인 점은 Q4의 논거에 기댄다.

### 판정
**표준(기존) — 핵융합에서는 미발견.** ML 방법론 안에서는 20년 된 표준이자 2024년 벤치마크의 추천 집합. 핵융합 응용 선례는 못 찾았다.

---

## Q4. 계층(다수준) 부트스트랩 — "학습 반복 × 방전" 2수준 재표집의 선례

### 발견 사항
- Saravanan, Berman & Sober, "Application of the hierarchical bootstrap to multi-level data in neuroscience", *Neurons, Behavior, Data Analysis, and Theory* 3(5), 1–25 (2020); 선행 bioRxiv 10.1101/819334 (2019, 인용 161) 및 arXiv:2007.07797; PMID 33644783. **[검증]**(학술지판 DOI는 페이지에 없음 — 인용 시 bioRxiv DOI 또는 PMID 사용). 상위 단위(동물)→하위 단위(뉴런) 순차 재표집이 1종 오류를 지키고 요약통계 방식보다 검정력을 유지한다고 보임.
- Ren, Lai, Tong, Aminzadeh, Hou & Lai, "Nonparametric bootstrapping for hierarchical data", *Journal of Applied Statistics* 37, 1487–1498 (2010). **[검증]** DOI 10.1080/02664760903046102, 인용 155. 핵심 결론: **최상위 수준만 복원 재표집**하고 하위 수준은 비복원(또는 그대로)으로 두는 것이 분산 추정에 가장 정확.
- Davison & Hinkley (1997) §3.8; Field & Welsh (2007) — Q2 참조. **[검증]**
- 2026년 응용: Anglin, "Estimating Uncertainty in Classifier Performance with Applications to Large Language Models and Nested Data", arXiv:2606.26422 — 분류 성능 CI에서 **계층 부트스트랩 vs 클러스터 부트스트랩**을 비교, 개체당 관측이 적으면 계층 부트스트랩이 과보수적이라고 보고. **[검증: arXiv]**
- "학습 반복(seed·분할)"을 분산원으로 다룬 ML 선례: Bouthillier et al., "Accounting for Variance in Machine Learning Benchmarks", *MLSys 2021*, arXiv:2103.03098. **[검증: arXiv]** 데이터 표집·초기화·하이퍼파라미터 분산을 모두 무작위화하고 그 분포를 보고하라고 권고. 또한 Bates et al.(2024)의 nested CV와 Schulz-Kümpel(2024)의 "외부 반복 ≥ 25"가 **반복을 바깥 수준으로 두는 2수준 구조**의 통계학적 선례.
- **정확히 "학습 반복 × 클러스터" 두 축을 동시에 재표집**한 공개 선례: **[미발견]**. 신경과학 계층 부트스트랩은 두 수준이 모두 모집단 표집 단위(동물·뉴런)인 반면, 우리의 바깥 수준(학습 반복)은 알고리즘 잡음 수준이다. 이는 Bouthillier(2021)의 "seed 분산을 포함하라"와 Saravanan/Ren의 "클러스터를 최상위에서 재표집하라"를 결합한 것이고, 결합 자체의 선례는 찾지 못했다.
- 핵융합·플라즈마에서 계층 부트스트랩: arXiv "tokamak" AND "hierarchical bootstrap" → 0건. **[미발견]**

### 우리 사용법과의 대응
풀링된 OOF 예측 위에서 (반복 → shot) 순으로 재표집하는 우리 통계량은 Ren et al.(2010)의 "최상위 수준 복원 재표집" 처방과 Saravanan et al.(2020)의 순차 재표집을 따른다. 다만 Ren의 결과는 하위 수준을 복원 재표집하면 분산이 과대될 수 있음을 시사하므로, 논문에는 두 수준 모두 복원인지, shot 수준만 복원인지를 **명시**하고 Anglin(2026)이 지적한 과보수성 가능성을 한 문장으로 적는 것이 좋다. heavy-tail 타깃에 대해 중심극한 근사 대신 부트스트랩을 1차로 둔 것은 Davison & Hinkley의 표준 권고와 일치.

### 판정
**타 분야에서 차용·핵융합에서는 미발견 (+ 결합 형태는 신규에 가까움).** 각 구성요소는 표준(신경과학·응용통계·ML 벤치마크)이나 "학습 반복 × 방전" 2수준 결합의 명시적 선례는 못 찾았으므로, 논문에서는 "신경과학의 계층 부트스트랩(Saravanan 2020; Ren 2010)을 ML 벤치마크의 seed 분산 권고(Bouthillier 2021)와 결합" 으로 서술하고 신규성은 주장하지 말 것.

---

## Q5. persistence/보간 대비 MSE 스킬 점수와 CRPS — 기상 나우캐스팅 관행과 핵융합 채택 여부

### 발견 사항
- Murphy, "Skill Scores Based on the Mean Square Error and Their Relationships to the Correlation Coefficient", *Monthly Weather Review* 116, 2417–2424 (1988). **[검증]** DOI 10.1175/1520-0493(1988)116<2417:SSBOTM>2.0.CO;2, 인용 895. MSE 스킬 = 1 − MSE_f/MSE_ref, 기준은 climatology 또는 **persistence**.
- Wilks, *Statistical Methods in the Atmospheric Sciences*, 4판, Elsevier (2019). **[검증]** DOI 10.1016/C2017-0-03921-6, 인용 519. 스킬 점수·persistence 기준·CRPS 모두 교과서 표준.
- Hersbach, "Decomposition of the Continuous Ranked Probability Score for Ensemble Prediction Systems", *Weather and Forecasting* 15, 559–570 (2000). **[검증]** DOI 10.1175/1520-0434(2000)015<0559:DOTCRP>2.0.CO;2, 인용 1,572.
- Gneiting & Raftery, "Strictly Proper Scoring Rules, Prediction, and Estimation", *JASA* 102, 359–378 (2007). **[검증]** DOI 10.1198/016214506000001437, 인용 6,460. Gneiting, Balabdaoui & Raftery, "Probabilistic forecasts, calibration and sharpness", *JRSS-B* 69, 243–268 (2007). **[검증]** DOI 10.1111/j.1467-9868.2007.00587.x, 인용 1,456.
- 강수 나우캐스팅 관행: Shi et al., "Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nowcasting", *NeurIPS 2015*, arXiv:1506.04214 **[검증]** (ROVER 광류 외삽 대비); Ravuri et al., "Skilful precipitation nowcasting using deep generative models of radar", *Nature* 597, 672–677 (2021). **[검증]** DOI 10.1038/s41586-021-03854-z, 인용 1,191 — 제목부터 "skilful"이며 persistence·PySTEPS 외삽 대비 CRPS/CSI로 평가. 즉 "persistence와 외삽(보간) 두 기준을 함께 두고 스킬을 말한다"는 것이 나우캐스팅 표준.
- **핵융합에서 persistence 기준·스킬 점수**: arXiv "tokamak" AND "persistence" AND "baseline" → 플라즈마 예측 논문 0건(물리 용어 매치만). WebSearch도 0건. Char et al.(2024)도 persistence 비교 없음(본문 확인). **[미발견]**
- **핵융합에서 CRPS**: 2026년 arXiv 1건 — Sorokin et al., arXiv:2610.24525 (전자밀도 재구성 서로게이트의 UQ를 CRPS·coverage로 채점). **[검증: arXiv]** 저널 게재 핵융합 논문에서는 **[미발견]**. 핵융합 UQ 논문은 ECE(Poels 2025), coverage/miscalibration area(Char 2024), conformal coverage(Vakhshouri 2026)를 쓴다 — 즉 **보정(calibration) 지표**는 쓰되 **적절 채점규칙(proper scoring rule)** 은 거의 안 쓴다.

### 우리 사용법과의 대응
MSE 스킬 = 1 − ΣSE_model/ΣSE_ref(persistence·PCHIP·인과 GP)는 Murphy(1988)의 정의 그대로이고, 두 기준(persistence = "변화 없음", 보간 = "외삽/보간 기준")을 함께 두는 것은 Ravuri(2021)형 나우캐스팅 관행과 같다. 분포 예측의 CRPS(B.17 혼합 head)는 Gneiting & Raftery(2007)·Hersbach(2000) 표준.

### 판정
**타 분야(기상 나우캐스팅·예측통계)에서 차용·핵융합에서는 미발견.** 핵융합 ML은 AUC/F1/κ/MSE 점추정과 보정 지표를 쓰며, persistence 스킬과 CRPS를 같이 쓴 사례는 2026년 preprint 1건(CRPS만) 외에 찾지 못했다.

---

## Q6. 평균(MSE)과 중앙값(MedAE) — heavy tail 아래서 다른 목표량으로 함께 보고하기

### 발견 사항
- Huber, "Robust Estimation of a Location Parameter", *Annals of Mathematical Statistics* 35, 73–101 (1964). **[검증]** DOI 10.1214/aoms/1177703732, 인용 7,707. Huber & Ronchetti, *Robust Statistics*, 2판, Wiley (2009). **[검증]** DOI 10.1002/9780470434697, 인용 1,771.
- **핵심 참고**: Gneiting, "Making and Evaluating Point Forecasts", *JASA* 106, 746–762 (2011). **[검증]** DOI 10.1198/jasa.2011.r10138, 인용 1,237. 점예측 평가는 손실함수와 정합(consistent)해야 하며, **제곱오차는 조건부 평균, 절대오차는 조건부 중앙값**의 적절한 채점함수 → 두 지표는 "같은 것을 두 번 재는" 것이 아니라 **다른 범함수(functional)를 겨냥**한다. 우리 §8bp "이중 판독(평균 추정량 vs 중앙값 추정량)"의 이론적 근거가 바로 이것.
- Kolassa, "Why the 'best' point forecast depends on the error or accuracy measure", *International Journal of Forecasting* 36, 208–211 (2020; 온라인 2019). **[검증]** DOI 10.1016/j.ijforecast.2019.02.017, 인용 107. 동일 논지의 실무 해설.
- Hyndman & Koehler, "Another look at measures of forecast accuracy", *IJF* 22, 679–688 (2006). **[검증]** DOI 10.1016/j.ijforecast.2006.03.001, 인용 6,369. 척도 의존/독립 지표와 중앙값 기반 지표 논의.
- Taggart, "Point forecasting and forecast evaluation with generalized Huber loss", *Electronic Journal of Statistics* 16, 201–231 (2022). **[검증]** DOI 10.1214/21-EJS1957. Huber 손실이 겨냥하는 범함수("Huber functional")를 정의 — 우리 B.14 huber 민감도 행을 해석할 때 인용 가능.
- Lerch, Thorarinsdottir, Ravazzolo & Gneiting, "Forecaster's Dilemma: Extreme Events and Forecast Evaluation", *Statistical Science* 32, 106–127 (2017). **[검증]** DOI 10.1214/16-STS588, 인용 184. 극단 사건(우리의 fit-failure 스파이크)만 골라 평가하면 왜곡된다는 경고 → 두 모집단(컷/포함) 공동 1차 보고의 근거로 쓸 수 있음.
- **핵융합에서 MedAE 병기**: arXiv "tokamak" AND "median absolute error" → 0건. **[미발견]**

### 우리 사용법과의 대응
"평균 = v4kjt(컷)/seq_v2(포함), 중앙값 = 순수 t 앵커" 처럼 추정량을 범함수별로 나눠 선언한 것은 Gneiting(2011)의 정합성 원리와 정확히 일치한다. heavy tail(fit-failure 스파이크) 때문에 두 모집단을 공동 1차로 두는 것은 Lerch et al.(2017)의 경고에 대한 사전 대응이다.

### 판정
**표준(기존) — 예측통계·강건통계의 정립된 이론; 핵융합에서는 미발견.**

---

## 종합 표

| 항목 | 우리 방법 | 가장 가까운 원전 | 핵융합 선례 | 판정 |
|---|---|---|---|---|
| (a) 사전등록·예측 장부 | 규칙·예측·공개사항 커밋 후 실행, held/missed 채점 | Nosek & Lakens 2014; Klein & Roodman 2005; Hofman 2023; NeurIPS PMLR 148/181 | 미발견 | 타 분야 차용·핵융합 미발견 |
| (b) 방전 클러스터 짝지은 부트스트랩 | skill 비율의 shot 재표집 CI, B=10⁴ | Davison & Hinkley 1997; Field & Welsh 2007; Cameron et al. 2008 | 미발견(seed 표준오차가 관행) | 타 분야 차용·핵융합 미발견 |
| (c-i) 반복 k-fold + NB 보정 | shot 단위 반복 5-fold | Nadeau & Bengio 2003; Bouckaert & Frank 2004; Bates et al. 2024; Schulz-Kümpel 2024 추천 | 미발견 | 표준(ML)·핵융합 미발견 |
| (c-ii) 2수준(반복×shot) 부트스트랩 | 풀링 OOF 예측의 계층 재표집 | Saravanan 2020; Ren 2010; Bouthillier 2021 | 미발견 | 차용 + 결합은 선례 없음(신규 주장은 말 것) |
| (d) 두 모집단 공동 1차 | 컷/포함 동시 보고 | Lerch et al. 2017(극단 사건 평가 왜곡) | 미발견 | 신규에 가까움(근거는 표준) |
| (e) persistence/보간 MSE 스킬 + CRPS | Murphy 스킬, 두 기준, CRPS | Murphy 1988; Wilks 2019; Hersbach 2000; Gneiting & Raftery 2007; Ravuri 2021 | CRPS 2026 preprint 1건, persistence 스킬 미발견 | 타 분야 차용·핵융합 미발견 |
| (f) 동결 test 행 bit-identical 재검증 | 2차 분석 전 재유도 | 블라인드 분석/재현성 문헌(Junk & Lyons 2020) | 미발견 | 공학적 관행; 문헌 인용은 (a)와 묶어 |
| (g) MSE·MedAE 병기 | 범함수별 추정량 선언 | Gneiting 2011; Kolassa 2020; Huber 1964 | 미발견 | 표준·핵융합 미발견 |

## 논문 서술 권고(한 단락용)
"평가 설계는 세 분야의 표준을 들여왔다. 분석 계획의 사전 고정은 심리학의 등록보고서와 입자물리의 블라인드 분석에서, 방전 단위 클러스터 부트스트랩과 반복 교차검증의 분산 보정은 응용통계·기계학습 방법론에서, persistence/보간 기준 스킬 점수와 CRPS는 기상 나우캐스팅에서 가져왔다. 핵융합 진단 ML 문헌에서는 shot 단위 분할은 표준이나 성능의 표집 불확실성을 방전 단위로 구간추정한 선례를 찾지 못했다." — "최초"라는 단어는 쓰지 말고 "선례를 찾지 못했다"로.

## 이번 조사의 한계
- OpenAlex 전문(fulltext.search) 검색을 일일 예산 소진으로 못 돌림 → 핵융합 논문 **본문**에만 등장하는 bootstrap CI·NB 보정 용례를 놓쳤을 수 있다. 다음 세션에 `filter=fulltext.search:` 쿼리 9개(스크립트 `oa2.py`에 준비됨)를 재실행하면 보강 가능.
- Semantic Scholar 전면 429.
- IOP(Nuclear Fusion) 본문은 접근 불가; Zhu 2021은 arXiv 판, Poels 2025는 arXiv 판으로 확인.
