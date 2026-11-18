# Literature map

Date: 2026-11-19. Merged from four reader notes. The notes are not in the repository.

## 1. The corpus and how it was read

- **Files.** There are 253 PDF files in `data/literature/`. That folder is git-ignored, so the PDFs are not part of this public repository.
  - They were fetched on 2026-11-18 and 2026-11-19. Some came from open-access copies: publisher open access, OSTI accepted manuscripts and reports, arXiv, and institutional repositories. Others came through the browser with institutional access.
  - Each file was checked for a PDF header and for a first-page title that matches the work.
  - The files `data/literature/INDEX*.md` give the source, access route, size and SHA-256 of every file.
- **Distinct works.** The 253 files hold 249 distinct works.
  - Three files are second copies of a work already in the set: `Pastore2026ApplicationsNovel_arXiv` (preprint), `Shukla2021MultiTime` (byte-identical) and `Young2008PlasmaMeasurements_PPPL-4276` (report version).
  - One file is the conference version of a journal paper: `Ko2009KstarCharge` (of `Ko2010KstarCesSystem`).
  - Each of these is listed once below, with a note.
- **Against the committed manuscript and `refs.bib`** (HEAD 7d11c66; the citation audit of the four notes was then applied in commit 4aa1222, which corrected 20 bib entries and added four cited works):
  - 120 works have a `refs.bib` key. 63 of them are cited in the main text or the supplement, and 57 are in `refs.bib` but not cited.
  - 129 works are not in `refs.bib`.
  - Eight cited keys have no PDF here:
    - `Agarwal2021Deep`, `Joung2026PhysicsInformedPreprocessing`, `Little2002StatisticalAnalysis`, `Maheswaranathan2020HowRecurrent`, `Wesson2011Tokamaks` and `rasmussen2006` were not fetched.
    - `fluctana` and `syndia` are code (section 2.9).
- **How the files were read.** Four reader agents each took one share of 63, 63, 64 and 63 files. They wrote one structured note per PDF, with these fields:
  - the citation as printed in the PDF;
  - one theme tag;
  - a 2–4 sentence summary in the reader's own words, with page numbers;
  - relevance to this thesis;
  - an experiment idea where the paper suggests one;
  - a citation audit, where the key is cited;
  - a `refs.bib` check.

  The readers extracted the text with PyMuPDF and read figures from rendered pages where a note says so. Long reviews were read only in the parts that matter here. All 253 files have a note.
- **Paraphrase.** Every summary in this map paraphrases the notes. No more than a short phrase is taken from any paper.
- **Pages.** "p. N" is the page index of the PDF file; publisher cover pages count. "jp. N" is the page number printed on the paper.
- **Not repeated here.** Citation-audit verdicts and `refs.bib` mismatches are in lists (a) and (b) at the end of each reader's notes. They are being handled with the manuscript.

## 2. Map by theme

Tags are the readers' tags, except for five items that were moved so that one topic sits in one place (listed in 2.9). Counts below are distinct works.

The "cited" column uses these values:

| value | meaning |
|---|---|
| `main`, `supp`, `main+supp` | where the key is cited at HEAD |
| `bib` | in `refs.bib`, not cited |
| `—` | not in `refs.bib` |

### 2.1 CES-diag (32 works)

The KSTAR CES reports the C6+ temperature from the Doppler width of C VI 529 nm, with the background taken from beam-off frames of a modulated beam and interpolated between them; the values are fitted automatically and carry fit-only error bars. At a ~0.5 keV pedestal top the atomic corrections to the width (cross-section, halo) are small, and the Zeeman and fine-structure bias at KSTAR's field has not been quantified. Passive edge light and beam plume are not small at the edge and are the stated reason other fits lose accuracy there, while failed automatic fits are named as the source of KSTAR CES outliers. No paper in the corpus checks the edge CES value on KSTAR independently; the cross-checks against x-ray crystal spectroscopy are core-only.

**KSTAR CES: the system and its values**

| key | year | device | role for us | cited |
|---|---|---|---|---|
| Ko2010KstarCesSystem | 2010 | KSTAR | CES design (journal). Modulated beam plus background subtraction. Names the electron-impact, active CX and passive CX components. `Ko2009KstarCharge.pdf` (SOFE 2009) is its conference version, not listed again | main+supp |
| Ko2010ChargeExchange | 2010 | KSTAR | CES installation and calibration on C VI 529 nm. Primary source for what the CES measures | main+supp |
| Lee2011DevelopmentGrating | 2011 | KSTAR | Two-grating spectrometer for the CES. Hardware history only | — |
| Lee2016ValidationToroidal | 2016 | KSTAR | Core XICS–CES agreement. Assumes the background is steady over a few hundred ms. Its "blips" are beam-on pulses into ohmic plasma | supp |
| Lee2020CrossComparisons | 2020 | KSTAR | Core XICS–CES agreement over many discharges. XICS is core-only; CES is the edge candidate | main |
| Lee2020TwoGaussian | 2020 | KSTAR | Routine CES facts: 32 ch, ~10 ms, 5 mm edge spacing, 5 ms exposure, 1–5 Hz modulation with 10 ms off, background interpolated. Two-Gaussian fit needs no modulation but fails at the very edge. Conventional analysis waits ≥ ~1 s | main+supp |
| Lee2022MeasurementCarbon | 2022 | KSTAR | C6+ density. The T_i error bar is the fit's standard deviation only | — |
| Jang2021DevelopmentUltrafast | 2021 | KSTAR | Ultrafast CES design (100 kHz). States that standard CES is 100 Hz with a 10 ms integration | supp |
| Kim2024KineticProfile | 2024 | KSTAR | SVM outlier flags, then GP on CES profiles (F1 0.857). Outliers tend to come from failed automatic fits. CES at 10 ms | main+supp |
| Kim2026AutomatedOutlier | 2026 | KSTAR | mtanh fits with a good-and-bad mixture likelihood. Median T_i uncertainty 0.080 keV for NN-CES values (#32364) | supp |

**What the fitted width measures: corrections and contamination**

| key | year | device | role for us | cited |
|---|---|---|---|---|
| Fonck1983PlasmaTemperature | 1983 | PDX | Founding paper: T_i from the CX line width | — |
| Fonck1984DeterminationPlasma | 1984 | PDX | CX rates. Fine structure, l-mixing, Stark effect and plume change the line shape | — |
| Boileau1989DeductionTemperature | 1989 | JET | Cold passive edge components must be separated. He, C and O give the same central T_i | — |
| Isler1994OverviewCharge | 1994 | review | General CXRS review. Cold components from electron excitation and from CX with thermal neutrals | — |
| vonHellermann1995CrossSection | 1995 | JET | Cross-section effects shift and narrow the line. The width correction grows with T_i, so it is small at 0.5 keV | supp |
| Meister2001MeasurementPoloidal | 2001 | AUG | Corrections for the energy-dependent emission rate and gyro-motion. Small for toroidal views | — |
| Blom2002ParametrizationZeeman | 2002 | atomic data | Zeeman and fine structure of H-like lines, including C VI 8–7. Ignoring them biases T upward, more at low T | — |
| Hey2002OxygenImpurity | 2002 | TEXTOR | An O VI blend under C VI 529 nm can bias T_i | — |
| Bell2006CarbonPlume | 2006 | NSTX | Beam-induced C5+ plume emission. Beam-off frames do not remove it | — |
| Solomon2008CharacterizationCross | 2008 | DIII-D | Co/counter views test the cross-section rotation correction (up to 40 km/s). The calculated correction overcorrects | — |
| Viezzer2011PassiveEdge | 2011 | AUG | Passive C5+ edge line: mostly thermal CX. Scales with pedestal C6+ density and edge neutrals | supp |
| Viezzer2012HighResolution (file `doi_10_1063_1_4755810.pdf`) | 2012 | AUG | Edge CXRS at 2.2 ms and 3 mm. Cross-section T correction 0.01–0.05 eV at 200–500 eV (B5+) | — |
| Chen2019EffectEnergy | 2019 | LHD | The apparent velocity from the energy-dependent cross-section grows with T_i | — |
| Pavlova2021StudiesHalo | 2021 | ITER (sim.) | Beam halo: T_i error below 1 %, spatial blur 1.5–2.5 cm | — |
| Thorman2021VisibleSpectroscopy | 2021 | JET-ILW | Tungsten CX lines on C VI 529 nm bias fits | — |

**Other CXRS systems: main ion, fast systems, MAST**

| key | year | device | role for us | cited |
|---|---|---|---|---|
| Ogawa1988DeterminationTemperature | 1988 | JFT-2M | CXRS T_i agrees with the neutral particle analyser | — |
| KOBAYASHI2003DevelopmentFast | 2003 | JT-60U | Fit-free filter CXRS at 0.8 ms, against 16.7 ms for the CCD system | — |
| Conway2006HighThroughput | 2006 | MAST | MAST CXRS: 224 ch, 5 ms frames matched to 200 Hz Thomson. Our MAST source system | supp |
| Grierson2016HighResolution | 2016 | DIII-D | Main-ion CER at 8 mm spacing. D and C temperatures agree in the core and part near the separatrix | — |
| Haskey2017DeuteriumCharge | 2017 | DIII-D | Main-ion CER at 2.5 ms. D+ T can be half of C6+ at the separatrix | — |
| McDermott2017ExtensionsCharge | 2017 | AUG | Routine 10 ms, down to 2.5 ms. A constrained multi-Gaussian separates active and passive light | — |
| Haskey2018ActiveSpectroscopy | 2018 | DIII-D | Active–passive subtraction from the nearest ELM-free beam-off slice within 20 ms. Result insensitive to the slice chosen | — |

### 2.2 RT-CES (32 works)

Real-time charge-exchange analysis has been used in feedback at JT-60U (T_i every 10 ms, analysed within 10 ms), JET (every 50 ms), DIII-D (rotation in several loops; CERREAL ~1 ms after a 5 ms frame) and NSTX(-U) (velocity at kHz, T_i only after the discharge). All of these fit a spectrum that exists. Neural readers make the fit faster but keep the frame time, and at KSTAR they do not remove the wait for the next beam-off frame (up to ~1 s), which is the bottleneck. Model-based observers (RAPTOR, RAPDENS, the DIII-D EKF) merge slow and fast diagnostics to estimate current, T_e and density; the one T_e observer takes T_i as a multiple of T_e.

**Real-time CXRS in control**

| key | year | device | role for us | cited |
|---|---|---|---|---|
| Herrmann1992RealTime | 1992 | concept | Proposes real-time T_i from doping-beam neutrals. No T_i diagnostic was real-time capable then | — |
| Joffrin2003IntegratedScenario | 2003 | JET | Real-time CX T_i and rotation on 11 channels every 50 ms. Five-Gaussian fit under 50 ms | — |
| Scoville2007SimultaneousFeedback | 2007 | DIII-D | Real-time CER rotation (2 chords) in feedback. ~10 ms filter-plus-analysis delay | — |
| Kobayashi2007DevelopmentRealtime | 2007 | JT-60U | Filter CXRS: T_i at 4 core radii every 10 ms, processing under 5 ms, grad-T_i feedback | main |
| Yoshida2007FastMeasurement | 2007 | JT-60U | Filter CXRS down to 0.16 ms. Central T_i held within 0.5 keV of the reference | — |
| Yoshida2009RealTime | 2009 | JT-60U | Fast CCD at 2.5 ms. 4 spectra analysed within 10 ms, under 5 % from offline. T_i-gradient and rotation feedback | — |
| Litaudon2011RealTime | 2011 | review | Real-time profile-control review, including JT-60U fast CX | — |
| Podesta2012RealTime | 2012 | NSTX | Real-time velocity CXRS up to 5 kHz at 6 radii. T_i only above ~300 eV | — |
| Podesta2016InitialOperation | 2016 | NSTX-U | Velocity to the PCS at 5 kHz. T_i only post-discharge. Photon noise 15–20 % at 2 kHz | — |
| Lanctot2016ErrorField | 2016 | DIII-D | Real-time CER rotation drives extremum-seeking error-field correction | — |
| Boyer2019FeedbackControl | 2019 | DIII-D | Real-time CER rotation in feedback. Beam modulation perturbs the plasma and blinds beam diagnostics | — |
| Chrystal2026RealTime | 2026 | DIII-D | CERREAL: results ~1 ms after the data, every 5 ms. Edge less accurate (79 % against 88 % of T_i within 20 %). No beam-off subtraction | main |

**Neural and fast readers of spectra (they read a spectrum that exists)**

| key | year | device | role for us | cited |
|---|---|---|---|---|
| Bishop1993AutomaticAnalysis | 1993 | JET | First NN reading of CX spectra, ~100× faster than least squares. Least-squares fits fail or swap lines | bib |
| Svensson1999AnalysisJet | 1999 | JET | NN trained on synthetic C VI spectra: 3–4 % T_i, with a novelty detector. A real-time T_i profile was built | bib |
| Chai2019FastEstimation | 2019 | EAST | NN T_i in 0.1 ms against a ~100 ms fit. ~3 % on 95 test frames. Cites a ~10 ms T_i control cycle | bib |
| Tian2026RealTimeCxrs | 2027 (online 2026) | HL-3 | CNN at 0.56 ms per profile, R² 0.93 for T_i. The 25 ms exposure is longer than the ~10 ms control needs | main |
| Haskey2026MachineLearning | 2026 | DIII-D | ML fits of main-ion CER plus FIDASIM surrogates. Pedestal-top corrections of 150–300 eV | — |

**Model-based observers and other real-time kinetic systems**

| key | year | device | role for us | cited |
|---|---|---|---|---|
| Felici2011RealTime | 2011 | TCV | RAPTOR: real-time current diffusion in ~1 ms steps | — |
| Felici2014DynamicState | 2014 | sim. | EKF with random-walk disturbance states. Removes model–measurement offsets | — |
| Messmer2019OptimalPolarisation | 2019 | AUG | EKF on RAPTOR plus MSE for the q profile | — |
| Blanken2019ModelBased | 2019 | AUG, TCV | RAPDENS EKF for density, with fault handling, 1–1.5 ms cycle | — |
| Bosman2021ModelBased | 2021 | ITER (sim.) | Density observer with Thomson. Simulation only | — |
| Pastore2023ModelBased | 2023 | TCV | RAPDENS merges 60 Hz Thomson with 20 kHz FIR (1 kHz in the observer). Each Thomson profile corrects the FIR offset | main |
| Pastore2026ApplicationsNovel | 2026 | TCV | RAPDENS in control: FIR at 10 kHz, profile every 1 ms in under 1 ms. The arXiv preprint file is the same work | main |
| Morosohk2023RealTime | 2023 | DIII-D | EKF on T_e with NN surrogates. T_i is a multiple of T_e | main |
| Shousha2023MachineLearningbased | 2024 | DIII-D | RTCAKENN: kinetic equilibrium including carbon T_i, under 8 ms in a 20 ms cycle. Input dropout covers missing CER | main |
| Kim2026RealTimePlasma | 2026 | DIII-D | SHIELD streams 64 BES channels and ECE into the PCS | main |
| Kim2023LabviewProgramming | 2023 | KSTAR | XICS quasi-real-time analysis, 1.6 ms per image. Core only, outside the PCS | supp |
| Mazon2002RealTime | 2002 | JET | ITB control from the real-time ECE T_e gradient (0.5 ms). 10 ms control step | — |
| Santos2010RealTime | 2010 | AUG | Real-time reflectometry every 1 ms. Samples during ELMs are rejected (Kalman gain set to zero) | — |
| Santos2017RealTime | 2017 | AUG | Real-time reflectometry plug-in, control data every 1 ms | — |
| Mohammed2026CommissioningOperation | 2026 | W7-X | Real-time Thomson at 90 Hz: ~1 ms evaluation, ~11 ms end to end | — |

### 2.3 KSTAR-facts (38 works)

KSTAR has two beam lines of three deuterium sources (~2 MW each); the first source of NBI-1 could not be used in 2022, and a ~10 ms blank serves the CES and MSE background. The BES is a movable 16×4 APD array on NBI-1 whose signal follows the local density and the beam, and the ECEI's 24×8 arrays view a pedestal described as optically grey or thick. The plasma control system runs kHz cycles with ~0.1 ms network delay, and no published KSTAR controller reads CES or a pedestal T_i in real time (they read D-alpha, interferometer density, real-time equilibrium and Mirnov). Seven discharges of our set appear in these papers: 31357, 31359, 31747, 31873, 31921, 31923 and 32092.

**Device, campaigns, beams**

| key | year | device | role for us | cited |
|---|---|---|---|---|
| Kwak2020KstarStatus | 2020 | KSTAR | Ten-year status. Context | — |
| Lee2022OverviewRecent | 2022 | KSTAR | Diagnostics overview: CES 32 fibres, 10 ms, modulated-beam background, T_i used to classify L and H mode | supp |
| Ko2024OverviewKstar | 2024 | KSTAR | Overview of 2021–2023, including the 2022 goals | — |
| Nam2026OverviewKstar | 2026 | KSTAR | R0 1.8 m, a 0.5 m, B_T ≤ 3.5 T. A blip near 5.0 s helped form an ion ITB | supp |
| Ko2026RolesKstar | 2026 | KSTAR | Facility review: state estimation from sparse diagnostics (NN-CES included) is a strategic KSTAR contribution | — |
| Han2024CriterionLong | 2024 | KSTAR | NBI-1: three sources, ~5.5 MW at 100 keV. NB1A unusable in 2022 | supp |
| Kwak2025EstimationNeutral | 2025 | KSTAR | 2 × 3 D sources, ~2 MW each (NB1A 1.5 MW). A shared beam box couples the sources | supp |
| Kwak2019MeasurementFast | 2019 | KSTAR | ~10 ms NBI blank for CES and MSE. Central T_e and W fall by under 10 %. Slowing-down time from neutron decay | supp |
| Lee2021SystematicEvaluation | 2021 | KSTAR | MSE aligned with NB1-A. NB1-B is the most efficient heating source | — |
| DeBock2012RealTime | 2012 | KSTAR | Real-time MSE design on NBI-1 source 1 | — |
| Hong2024ImpactStray | 2024 | KSTAR (preprint) | Stray-field NBI loss. BES intensity falls ~20 % over 9 s at steady density (#27327) | — |
| Jeong2026OperationalStability | 2026 | KSTAR | Long-pulse NBI: delivered power lower at 90 s than at 20 s | — |
| Cha2024AnalysisNeoclassical | 2024 | KSTAR | #31747 (ours): 2/1 NTM, stabilised by ECCD after 7.3 s | — |
| Yang2021ParametricDependencies | 2021 | KSTAR | Locked-mode thresholds in L-mode. Outside our regime | — |
| Yang2024TailoringTokamak | 2024 | KSTAR | 31357 and 31359 (ours): H-mode avoided with an edge-localised n = 1 field | supp |
| Seo2026RevisedError | 2026 | KSTAR | Error-field penetration timed with ECE and CES. CES timing ±5 ms | — |

**Control system and the real-time path**

| key | year | device | role for us | cited |
|---|---|---|---|---|
| Park2011HighSpeed | 2011 | KSTAR | PCS reflective-memory transfer within 50 µs | — |
| Hahn2016ProgressPlan | 2016 | KSTAR | PCS. Mirnov amplitude used for NTM detection. ~20 ms delay to the ECH launcher | — |
| Hahn2018Achievements (file `doi_10_1016_j_fusengdes_2018_02_066.pdf`) | 2018 | KSTAR | Upgraded PCS: 400+ channels at 5 kHz, fastest cycle 20 kHz, ~100 µs network delay | — |
| Kwon2018DevelopmentReal | 2018 | KSTAR | Real-time network archive. PCS response budget under 50 µs | — |
| Hahn2020AdvancesChallenges | 2020 | KSTAR | List of kinetic controls. None uses CES or T_i | — |
| Shin2022PreemptiveDriven | 2022 | KSTAR | LSTM in the PCS on D-alpha and interferometer, 1 kHz cycle, ~0.8 ms. No direct real-time pedestal diagnostic | supp |
| Juhn2026FpgaBased | 2026 | KSTAR | FPGA interferometer density sent to the PCS | — |
| Tobin2026AvoidanceDisruptions | 2026 | KSTAR | Data-trained surrogate on magnetics, running in the PCS | — |

**BES, ECEI and ECE: systems and how to read them**

| key | year | device | role for us | cited |
|---|---|---|---|---|
| Nam2012AnalysisEdge | 2012 | KSTAR | Trial BES (4×8, 2 MHz, 2011). Not the 16×4 array of 2022 | main |
| Nam2014EdgeElectron | 2014 | KSTAR | 2D BES (8×4, 2 MHz) gives edge n_e profiles. Radial channels numbered from the inside | supp |
| Lampert2015CombinedHydrogen | 2015 | KSTAR | BES 4×16 APD array at ~1 cm, movable, row–column names. Part of the low-frequency signal is beam attenuation | supp |
| Lampert2018ScrapeLayer | 2018 | KSTAR | BES sees blobs and holes in the SOL | — |
| KFE2026DiagnosticsDataUserGuide | 2026 | KSTAR | Facility guide (slides) — see the note below this table | supp |
| Jeong2010ElectronCyclotron | 2010 | KSTAR | 48-channel ECE radiometer, not the ECEI | — |
| Yun2010DevelopmentKstar | 2010 | KSTAR | First ECEI system. Images of the optically thin edge may read density | main |
| Yun2011TwoDimensionalVisualization | 2011 | KSTAR | ECEI with 24×8 per array, ~1 µs, ~1 cm mapping error. KSTAR pedestal optically thick or grey | supp |
| Yun2014Quasi3d | 2014 | KSTAR | Second ECEI system: 2nd-harmonic X-mode, 24-element arrays | supp |
| Rathgeber2013EstimationEdge | 2013 | AUG | Edge ECE radiation transport: the shine-through peak is not T_e. Moved from `other` | — |
| Denk2018AnalysisElectron | 2018 | AUG | ECE forward model: at low optical depth the edge T_rad is not the local T_e | — |
| Vanovac2018EffectsDensity | 2018 | AUG | Edge ECEI is not local. Optical thickness scales with n_e T_e, and density fluctuations modulate the signal | supp |
| Zhu2019NewTrends | 2019 | review | Microwave imaging review. Pedestal optical depth varies rapidly (general, not KSTAR) | supp |
| Yu2022DiagnosingPedestal | 2022 | DIII-D | ECE shine-through near the separatrix. Pedestal radiation drops before ELMs | — |

`KFE2026DiagnosticsDataUserGuide` is the primary source for channel radii, naming, modulation and the status of CES values. From its slides:

- **CES:** 100 Hz. Radii from 2022 on: ch17 = 2192 mm, ch18 = 2198 mm. Modulation 1–2 Hz at 99 % duty. Values are fitted and uploaded automatically and are meant for preliminary analysis.
- **BES:** the signal follows the local n_e.
- **ECEI:** the guide gives the naming.

### 2.4 Pedestal physics (49 works)

Main-ion and carbon temperatures agree in the core and the steep gradient, differ by ~100 eV (50–60 eV after corrections, one discharge) at the AUG pedestal top and by up to a factor of two at the DIII-D separatrix, and lead to different pedestal transport conclusions. After an ELM the edge T_i gradient recovers in 2–4 ms, faster than a 10 ms frame, while the pedestal-top T_i rebuilds over the whole period between ELMs (40–70 ms on KSTAR). Beam ions slow down classically over tens of ms, so heating does not stop when the source stops. Published KSTAR blips perturb the plasma (a displacement and T_i fluctuations, mostly in the core), and one helped form an ITB.

**Impurity against main-ion temperature**

| key | year | device | role for us | cited |
|---|---|---|---|---|
| Haskey2018MainIon | 2018 | DIII-D | D+ temperature as low as half of C6+ at the separatrix. The large differences sit there | main+supp |
| CanoMegias2025MultiIon | 2025 | AUG | T_D ≈ T_z in the steep gradient. T_D above T_z at the pedestal top by ~100 eV, 50–60 eV corrected, in one discharge | main+supp |
| Haskey2022ThermalTransport | 2022 | DIII-D | With D+ the pedestal ion heat flux is above neoclassical. Ion-scale BES fluctuations are stronger at low collisionality | — |
| Fenstermacher2025ProgressPedestal | 2025 | review | Main-ion and carbon T_i lead to different pedestal transport conclusions | — |

**KSTAR pedestal, ELM and RMP studies (including our discharges)**

| key | year | device | role for us | cited |
|---|---|---|---|---|
| Ko2014RotationCharacteristics | 2014 | KSTAR | CES rotation under RMP. Signal-to-noise limited the time resolution | — |
| Ko2015IonTemperature | 2015 | KSTAR | T_i and rotation pedestals. NBI at 5 Hz with 10 ms off for background, 5.5 ms CCD. LCFS ~2.25 m in H-mode | supp |
| Ko2016CharacteristicsToroidal | 2016 | KSTAR | Pedestal recovery between ELMs, with 40–70 ms ELM periods. 10 ms frames with a 5 ms average. Knee at 2.22–2.23 m (2014) | supp |
| Kim2020PedestalElectron | 2020 | KSTAR | RMP database. The CES channels nearest the pedestal top are at R ≈ 2.19–2.20 m | supp |
| Heo2025ExperimentalIdentification | 2025 | KSTAR | 31921 and 31923 (ours). Edge CES T_i (ρ_N ≈ 0.94) used as a proxy for pedestal height. BES weakly coherent mode, bursts near H-mode, beam power drops at blips | main+supp |
| Na2026FireMode | 2026 | KSTAR | FIRE mode. 31921 has its pedestal top at r/a ≈ 0.97 in H-mode and ≈ 0.92 in FIRE. Edge error bars 0.09–0.14 keV | supp |
| Lee2026SpatiotemporalStructure | 2026 | KSTAR | EHO in 32092 (ours). Separatrix ~2.236 m, edge CES 2.18–2.25 m, T_i knee at ψ_N 0.95–0.96 | main+supp |
| Lee2023EffectCoherent | 2023 | KSTAR | A coherent edge mode seen in BES and Mirnov raises the pedestal | — |
| Kim2018EnhancedFast | 2018 | KSTAR | CES blips cause a displacement and T_i fluctuations, mostly core. NB1-B drops about once a second | main+supp |
| Kim2017ObservationResonant | 2017 | KSTAR | Magnetic braking by n = 1 fields (rotation) | bib |
| Park2013RotationalResonance | 2013 | KSTAR | NTV braking resonance (rotation) | bib |
| Park2018FieldPhase | 2018 | KSTAR | Operating windows for 3D fields. T_i from the two-Gaussian fit | — |
| Kim2022OptimizationControlled | 2022 | KSTAR | Adaptive RMP control from the D-alpha ELM frequency | — |
| Kim2024HighestFusion | 2024 | KSTAR | #31873 (ours): ML-chosen RMP. D-alpha and real-time equilibrium are in the loop | supp |
| Kim2026RealTimeFeedback | 2026 | KSTAR (preprint) | ELM-frequency feedback by gas puffing | — |
| Kim2017PhysicsBased | 2017 | KSTAR-type model | Model of confinement scaling | — |

**Time scales of the ELM cycle and pedestal transport**

| key | year | device | role for us | cited |
|---|---|---|---|---|
| Cavedon2017Pedestal | 2017 | AUG | ELM-synchronised CXRS at 65 µs. T_i gradient back in ~4 ms. Pedestal-top T_i rises until the next ELM | bib |
| Viezzer2013HighAccuracy | 2013 | AUG | E_r and the edge profiles recover within ~2 ms after an ELM | — |
| Viezzer2014EvidenceNeoclassical | 2014 | AUG | E_r in the pedestal is neoclassical | — |
| Viezzer2018IonHeat | 2018 | AUG | Ion heat diffusivity becomes anomalous at ELM onset. The T_i pedestal recovers in 3–4 ms | — |
| Viezzer2020Dynamics | 2020 | AUG | Ion gradient recovers in 3–4 ms, electron gradient in 7–8 ms. A 2.3 ms CXRS smears the rotation | supp |
| Burrell2009EdgePedestal | 2009 | DIII-D | QH-mode and EHO background | — |
| Chen2016RotationalShear | 2016 | DIII-D | EHO structure seen in BES, ECEI and magnetics | — |
| Snyder2009DevelopmentValidation | 2009 | DIII-D | EPED1 pedestal height | — |
| Snyder2011FirstPrinciples | 2011 | multi | EPED1.6 | — |
| PazSoldan2019EffectPlasma | 2019 | DIII-D | Rotation threshold for RMP suppression | — |
| PazSoldan2024PlasmaPerformance | 2024 | AUG, DIII-D, EAST, KSTAR | Operating space of RMP-suppressed edges | — |
| Ryter2014KeyRole (file `doi_10_1088_0029_5515_54_8_083003.pdf`) | 2014 | AUG | The edge ion heat flux sets the L-H transition | — |

**Core transport, MHD, rotation (background)**

| key | year | device | role for us | cited |
|---|---|---|---|---|
| ITER1999Chapter2 | 1999 | review | IPB98(y,2) scaling (eq. 20, jp. 2204). Source for the supplement's τ_E | — |
| Kinsey2011IterPredictions | 2011 | DIII-D, JET | TGLF validation | — |
| Mantica2009ExperimentalStudy | 2009 | JET | Rotation lowers ion stiffness | — |
| Ryter2011SimultaneousAnalysis | 2011 | JET | ICRH modulation. CXRS went from 50 ms to 10 ms after 2006 | — |
| Tresset2002DimensionlessCriterion | 2002 | JET | Criterion for an ITB | — |
| Yoshida2012CoreEdge | 2012 | JT-60U | Edge rotation changes within ms; the core follows over ~20 ms | — |
| Yoshida2025TransportConfinement | 2025 | review | Transport review. NN transport surrogates run in ~1 ms | — |
| Cole2011ObservationPeak | 2011 | DIII-D | Peak of the NTV force | — |
| Fitzpatrick1993InteractionTearing | 1993 | theory | Tearing-mode locking | bib |
| LaHaye2006NeoclassicalTearing | 2006 | review | NTM physics and control | bib |

**Beam and fast-ion physics (context for the blips)**

| key | year | device | role for us | cited |
|---|---|---|---|---|
| Stix1972HeatingToroidal | 1972 | theory | Critical energy. ~75 % of beam energy goes to ions at E_crit | supp |
| Heidbrink1988ComparisonExperimental | 1988 | DIII-D | After 2 ms beam pulses the slowing-down is classical within ~30 % | supp |
| Heidbrink1994BehaviourFast | 1994 | review | Formulas eqs. 7, 9, 10 give 9.3 keV, 33 ms and 39 ms at the supplement's inputs | main+supp |
| Zhang2012MeasurementsFast | 2012 | HL-2A | After 5 ms blips the neutron e-folding is ~13 ms (ohmic), classical. Moved from CES-diag | — |
| Zhao2023RefinedInterpretation | 2023 | DIII-D | Modulated NBI: prompt T_e cooling at turn-on, heating at turn-off. Moved from CES-diag | supp |
| Tala2009EvidenceInward | 2009 | JET | NBI modulation used on purpose to perturb rotation | — |
| Zimmermann2026DependenceMomentum | 2026 | DIII-D (preprint) | Momentum transport from modulated NBI | — |

### 2.5 ML-fusion (29 works)

NN readers of spectra (KSTAR NN-CES, HL-3, EAST, JET) read a frame whose spectrum exists. Reconstruction work (Diag2Diag, COMPASS, the EAST T_e and Z_eff gap-fillers, FusionMAE) predicts one diagnostic from others without the target's own history; most of it is memoryless or non-causal and reports R², MAPE or coverage rather than skill against hold or a causal reference. Far out of distribution, holding the initial profile beats ML and physics predictors (Abbate2025), which supports hold as a mandatory reference. BES fluctuations carry confinement-regime and ELM information at MHz rates, whereas our 10 ms rows hold single samples.

**Readers of data that exist (closest prior work)**

| key | year | device | role for us | cited |
|---|---|---|---|---|
| Lee2026NeuralNetworkbased | 2026 | KSTAR | NN-CES: MLP on raw C VI spectra, no beam-off background needed, ~210 ns per frame. Accuracy falls beyond R ≈ 2.2 m. Closest prior work | main+supp |
| Lin2024PredictionPlasma | 2024 | EAST | NN on XCS spectra gives T_i and rotation in tens of ms | bib |
| Pavone2019NeuralNetwork | 2019 | W7-X | NN surrogate of Bayesian XICS inference, ~10 µs | bib |
| Matos2016DeepLearning | 2017 | JET | NN bolometer tomography | bib |
| Wang2025PhysicsinformedDeep | 2025 | EAST, HL-2A | Physics-informed SXR tomography | bib |

**Reconstructing one diagnostic from others (second-closest)**

| key | year | device | role for us | cited |
|---|---|---|---|---|
| Jalalvand2025MultimodalSuperresolution | 2025 | DIII-D | Diag2Diag: Thomson T_e and n_e at 1 MHz from other diagnostics (CER is an input). MLP with derivatives. No target at inference | main |
| Imrek2026TemporalSuperresolution | 2026 | COMPASS | Memoryless MLP from SXR and AXUV to Thomson profiles at µs. Edge error ~11 % | main |
| Wang2025TimeSeries | 2025 | EAST | A missing ECE T_e channel from 15 signals. Offline inputs | bib |
| Jia2026TimeSeries | 2027 (online 2026) | EAST | Missing Z_eff from a BiGRU. Offline by design | bib |
| Yang2026FusionmaeSelfsupervised | 2026 | HL-3 | Masked autoencoder rebuilds missing signals ("virtual backup diagnostic") | bib |
| Chen2026FusionclipMultimodal | 2026 | EAST (preprint) | Contrastive pretraining of video with asynchronous signals | — |
| Kaldas2026DevelopingAutoencoder | 2026 | synthetic | Autoencoder fusion of Thomson and interferometry. Synthetic data only | bib |
| Wei2023ReconstructionTokamak | 2023 | DIII-D | q profile without MSE, under 1 ms | bib |
| Zheng2024RealtimeEquilibrium | 2024 | HL-3 | Multi-task EFIT surrogate | bib |
| Joung2023GsdeepnetMastering | 2023 | KSTAR | Unsupervised Grad–Shafranov equilibrium | bib |

**Profile prediction and NN observers**

| key | year | device | role for us | cited |
|---|---|---|---|---|
| Abbate2021DatadrivenProfile | 2021 | DIII-D | LSTM predicts profiles 200 ms ahead, with forward-fill on a 50 ms base. Warns about near-duplicate discharges | bib |
| Abbate2025Combining | 2025 | DIII-D, AUG, ITER | Far out of distribution, holding the initial profile beats ML and physics predictors | bib |
| Boyer2021PredictionElectron | 2021 | NSTX-U | NN profile shapes, corrected by an EKF with 17 ms real-time Thomson | bib |
| Jung2026MachineLearningbased | 2026 | KSTAR | ML profile parameters from Thomson, TCI and CES between shots | bib |
| Wakatsuki2021TemperatureGradient | 2021 | JT-60U (sim.) | RL controller fed a two-point CXRS T_i difference, NBI changed every 20 ms | — |
| Rothstein2026EnablingIntegrated | 2026 | DIII-D | PCS framework: CER T_i profiles every 20 ms, BES at 1 µs. Actuators no faster than ~10 ms | — |

**Fast fluctuation channels and real-time ML**

| key | year | device | role for us | cited |
|---|---|---|---|---|
| Gill2024RealTime | 2024 | DIII-D | CNN on 1 ms BES windows classifies L, H and QH mode (F1 0.94) | — |
| Joung2026ExplainableTokamakagnostic | 2026 | DIII-D → KSTAR | ELM forecasting from MHz BES transfers between devices without retraining. Data on request | supp |
| Gupta2026BenchmarkingSensor | 2026 | MAST (preprint) | Sensor-failure test: an LSTM suffers most when the last steps are corrupted | — |
| KatesHarbeck2019PredictingDisruptive | 2019 | DIII-D, JET | Recurrent disruption predictor (the LSTM lineage) | bib |
| Keith2024RiskAware | 2024 | C-Mod, DIII-D | Disruption prediction by survival analysis | bib |
| Mohapatra2019RealtimeSensor | 2020 | simulated | Classifiers that detect sensor faults | bib |
| Rath2022DataAugmentation | 2022 | DIII-D | Data augmentation with a Student-t state-space model | bib |
| Allen1992NeuralNetwork | 1992 | JET | Early NN regression of confinement time | bib |

### 2.6 TS-methods (35 works)

Our encoding of the sparse target (last value, time since it, a mask) is the GRU-Simple baseline of Che2018; GRU-D decay, T-LSTM, ODE/CDE encoders, Phased LSTM and mTAN are the alternatives, and four continuous-time encoders were tried in the window family without a keep (log §8e). Robust Kalman updates (Masreliez, Huber-based filters, Student-t filters, WoLF) are the established family of our filter's Huber update. The filter/smoother split (Shukla2020, VanMulders2025) gives standard names to our causal and offline references. PCHIP and GP have their primary references here.

**Missing data and irregular sampling in recurrent networks**

| key | year | device | role for us | cited |
|---|---|---|---|---|
| Che2018RecurrentNeural | 2018 | clinical | GRU-D. Its GRU-Simple baseline is our encoding (forward-filled value, mask, interval) | main |
| Lipton2016ModelingMissing | 2016 | clinical | Missingness indicators help LSTMs | — |
| Baytas2017PatientSubtyping | 2017 | clinical | T-LSTM: memory decays with elapsed time | — |
| Cao2018BritsBidirectional | 2018 | various | BRITS: bidirectional imputation (offline) | — |
| Shukla2019InterpolationPrediction | 2019 | clinical | Interpolation–prediction networks with an intensity channel | — |
| Shukla2020SurveyPrinciples | 2020 | survey | Separates filtering from smoothing: our causal against offline references | — |
| Shukla2021MultitimeAttention | 2021 | clinical | mTAN: attention over irregular times. Two identical files | bib |
| Weerakody2021ReviewIrregular | 2021 | review | Gated RNNs for irregular series. Evaluate under the real missingness pattern | — |
| Neil2016PhasedLstm | 2016 | event data | LSTM with a time gate | — |
| Rubanova2019LatentOrdinary | 2019 | various | ODE-RNN and latent ODE | bib |
| DeBrouwer2019GruOde | 2019 | various | GRU-ODE-Bayes | — |
| Kidger2020NeuralControlled | 2020 | various | Neural CDE | — |

**Architectures, multimodal and multitask**

| key | year | device | role for us | cited |
|---|---|---|---|---|
| Caruana1997MultitaskLearning | 1997 | — | Multitask learning. Features that exist only later can be extra training outputs | bib |
| Baltruaitis2018MultimodalMachine | 2018 | survey | Fusion taxonomy: early, late, hybrid | bib |
| Ilse2018AttentionbasedDeep | 2018 | — | Gated attention pooling. Origin of B.67's form | bib |
| Shi2015ConvolutionalLstm | 2015 | radar | ConvLSTM for nowcasting | bib |
| Ravuri2021SkilfulPrecipitation | 2021 | radar | Generative nowcasting | bib |
| Sussillo2013OpeningBlackBox | 2013 | — | Fixed-point analysis of trained RNNs | bib |

**State-space filters, robust updates, data assimilation**

| key | year | device | role for us | cited |
|---|---|---|---|---|
| Masreliez1977RobustBayesian | 1977 | — | Robust Kalman filter with influence functions. Rejection beyond 3σ is common practice | — |
| Agamennoni2012ApproximateInference | 2012 | — | Variational KF and RTS smoother with heavy-tailed noise | bib |
| Roth2013StudentT | 2013 | — | Student-t filter | bib |
| DuranMartin2024OutlierRobust | 2024 | — | WoLF: a robust KF update by weighted likelihood | — |
| eyheramendy2018 | 2018 | astronomy | IAR model for irregular series. Also a test for autocorrelated residuals | bib |
| elorrieta2019 | 2019 | astronomy | Complex IAR for irregular series | bib |
| vandenBrand2016ModelBased | 2016 | AUG | IMM bank of Kalman filters detects sawtooth crashes on ECE | bib |
| VanMulders2025ModelBased | 2025 | ITER (sim.) | RAPTOR EKF. Separates filter from smoother. On noisy data the EKF beats a spline | — |
| VanMulders2026ModelBased | 2026 | TCV | RAPTOR EKF on experimental data | — |
| Morishita2020DataAssimilation | 2020 | LHD | EnKF assimilates T_e, T_i and n_e at 80–160 ms cycles | bib |
| Morishita2024FirstApplication | 2024 | LHD | First control based on data assimilation | — |
| ICHIKAWA2025ApplicationData | 2025 | JT-60U | EnKF with tokamak transport models | — |
| Joung2018ImputationFaulty | 2018 | KSTAR | GP plus Bayesian imputation of faulty magnetic probes in under 1 ms | bib |

**Interpolation and GP references**

| key | year | device | role for us | cited |
|---|---|---|---|---|
| fritsch1980 | 1980 | — | Monotone piecewise cubic interpolation (PCHIP) | main |
| reinsch1967 | 1967 | — | Smoothing spline | bib |
| chilenski2015 | 2015 | C-Mod | GP proposed for profile fitting in place of splines | main |
| michoski2024 | 2024 | DIII-D | GP guide. Type-II maximum likelihood can understate uncertainty | bib |

### 2.7 Statistics (14 works)

Murphy1988 defines our skill and decomposes it, and Gneiting2007 requires comparison on identical cases and notes that skill scores are generally not proper. Goddard2013 is a published precedent for our two-level bootstrap. Cross-validation estimates the error of the procedure, not of one fitted model, and naive intervals undercover (Bates2024; Nadeau2003, whose correction is used in PK 4.32). Crossed bootstraps can instead be conservative (Gallas2009, Owen2007).

| key | year | device | role for us | cited |
|---|---|---|---|---|
| murphy1988 | 1988 | — | MSE skill score and its decomposition: correlation, conditional bias, mean bias | main |
| Gneiting2007StrictlyProper | 2007 | — | Compare on identical cases. Skill scores are not proper in general | bib |
| Goddard2013Verification | 2013 | climate | MSSS with a two-level bootstrap over start dates and members | bib |
| Gebetsberger2018Estimation | 2018 | weather | CRPS against maximum likelihood. Heavy-tailed predictive distributions | bib |
| Rasp2018NeuralNetworks | 2018 | weather | NN trained with CRPS gives a calibrated spread | bib |
| Huber1964RobustEstimation | 1964 | — | M-estimator. Source of the filter's Huber weight | main |
| Karlgaard2007HuberBased | 2007 | tracking | Huber measurement update in a nonlinear filter | — |
| Bates2024CrossValidation | 2024 | — | CV estimates the error of the procedure. Naive CV intervals undercover | bib |
| Nadeau2003Inference | 2003 | — | Corrected variance for resampled splits. Used for the fold interval in PK 4.32 | bib |
| Gallas2009Framework | 2009 | — | Crossed random effects. The bootstrap variance is biased upward | bib |
| Owen2007Pigeonhole | 2007 | — | Pigeonhole bootstrap for crossed data | bib |
| Ren2010Nonparametric | 2010 | — | Hierarchical bootstrap: resample the top level | bib |
| Saravanan2020Hierarchical | 2020 | neuroscience | A hierarchical bootstrap keeps the type I rate | bib |
| Weissgerber2015Beyond | 2015 | — | Show distributions, not bar graphs of means | bib |

### 2.8 Other (20 works)

ITER asks for the edge T_i every 100 ms at 10 %, beam modulation relaxes the ITER CXRS cadence to 0.2–1.5 s, and the FIRE design study specifies the edge T_i at 10 ms and 10 %. The 1999 ITER Physics Basis judged 100 ms enough for most profile work and ~1 ms necessary for transients. Control reviews call for state estimators that work from limited, noisy diagnostics. FAIR-MAST is our MAST data source.

**Measurement requirements**

| key | year | device | role for us | cited |
|---|---|---|---|---|
| Diagnostics1999ChapterMeasurement | 1999 | ITER | ~100 ms enough for most profile studies, ~1 ms for transients. T_i needs its own measurement | — |
| Donne2007ChapterDiagnostics | 2007 | ITER | Edge T_i: 100 ms, 10 % (Table 2, jp. S343) | — |
| Young2008PlasmaMeasurements | 2008 | FIRE design | Edge T_i: 10 ms, 5 mm, 10 % by CXRS (jp. 288–289, Table III). The PPPL-4276 report file is the same work. Moved from KSTAR-facts | main |
| Mazon2025Diagnostics | 2025 | ITER | Edge T_i: 100 ms, 10 %. Beam modulation relaxes CXRS to 200 ms (burst) and 1.5 s (control). Moved from CES-diag | main |

**Control context**

| key | year | device | role for us | cited |
|---|---|---|---|---|
| MHD1999ChapterPlasma | 1999 | ITER | Chapter on operation and control | — |
| Gribov2007ChapterPlasma | 2007 | ITER | Kinetic control is less developed than magnetic control | — |
| Humphreys2015NovelAspects | 2015 | ITER | State estimators from limited, noisy diagnostics are needed | — |
| Biel2019DiagnosticsPlasma | 2019 | DEMO | DEMO control diagnostics (no CXRS among them) | — |
| Bandyopadhyay2025DisruptionsControl | 2025 | review | Measurements must be real-time and fault tolerant. State observers | — |
| Na2025IntegratedOperation | 2025 | review | Real-time observers use simplified models | — |
| Laborde2005ModelBased | 2005 | JET | Model-based real-time profile control | — |
| Moreau2008TimeScale | 2008 | JET | Control model with two time scales | — |
| Goumiri2016ModelingControl | 2016 | NSTX | Model for rotation control | — |
| Goumiri2017SimultaneousFeedback | 2017 | NSTX-U | Rotation and β control (simulation) | — |
| Pajares2022Integrated (file `doi_10_1088_1741_4326_ac4868.pdf`) | 2022 | DIII-D | Integrated control of scalars, with NTM suppression | — |
| Volpe2012ObliqueElectron | 2012 | DIII-D | Oblique ECE inside a real-time NTM loop | — |
| Sabbagh2023DisruptionEvent | 2023 | KSTAR, MAST, NSTX | DECAF event chains, moving to real time on KSTAR | — |
| Reimerdes2007ReducedCritical | 2007 | DIII-D | Critical rotation for RWM stability | — |
| Rice2007IntermachineComparison | 2007 | multi | Scaling of intrinsic rotation | bib |

**Data source**

| key | year | device | role for us | cited |
|---|---|---|---|---|
| Jackson2024FairMast | 2024 | MAST | The FAIR-MAST archive, our MAST data source | supp |

### 2.9 Re-tagged items and non-paper files

**Re-tagged** (reader tag → tag here):

| item | from | to | reason |
|---|---|---|---|
| Rathgeber2013 | `other` | KSTAR-facts | Joins the other ECE/ECEI-interpretation papers |
| Zhang2012 | `CES-diag` | pedestal-physics | Joins the beam-physics block |
| Zhao2023 | `CES-diag` | pedestal-physics | Joins the beam-physics block |
| Mazon2025 | `CES-diag` | other | Requirements block |
| Young2008 | `KSTAR-facts` | other | Requirements block |

Counts in the readers' own tags:

| tag | files |
|---|---|
| pedestal-physics | 47 |
| KSTAR-facts | 39 |
| CES-diag | 36 |
| TS-methods | 36 |
| RT-CES | 33 |
| ML-fusion | 29 |
| other | 19 |
| stats | 14 |

**Not papers.**

- `KFE2026DiagnosticsDataUserGuide` is a facility slide deck, i.e. grey literature. It is kept in 2.3 because it is cited.
- `data/literature/` also holds files that are not PDFs and that the readers did not read:
  - seven text copies of public code (`code_delta_kstar_ecei`, `code_fluctana_kstarecei_kstarbes`, `code_prism_ecei_bes`, `code_stft_elmnet_shot_curation`, `code_syndia_ecei_forward_model`, `code_zenodo18491179_kstar_bes_geometry`, and one script of our own, `code_own_tau_x2_estimate.py`);
  - the indexes `INDEX*.md`, `TO_FETCH.md` and `ctx.txt`.

  The cited keys `fluctana` and `syndia` point to code of this kind.

## 3. What the literature says about our premises

### 3a. Is the pedestal-top CES T_i a physical quantity, and what are its known error sources?

**Short answer.** The CES value is the fitted Doppler temperature of C6+.

**Evidence that it follows the ion temperature:**

- It agrees with independent diagnostics in the core: XICS on KSTAR, and an NPA on JFT-2M.
- Physics papers use the edge value directly, for example as a proxy for pedestal height.
- At a ~0.5 keV pedestal top, the published corrections to the width are small: cross-section and halo.

**What can move it:**

- passive edge light and plume;
- line blends;
- the main-ion/impurity difference at the pedestal top (~50–100 eV at AUG, one discharge);
- failed automatic fits.

No paper in the corpus checks the edge CES value on KSTAR against another diagnostic.

| effect | what the literature reports | bearing on a ~0.5 keV pedestal-top chord | keys (pages) |
|---|---|---|---|
| What is measured | C6+ temperature from the Doppler width of the CX-excited line | The target is the impurity temperature as fitted | Fonck1983 p.2; Ko2010ChargeExchange p.2; Isler1994 |
| Independent cross-checks | XICS = CES in the core on KSTAR; CXRS = NPA on JFT-2M | Core only. XICS does not reach the edge | Lee2020CrossComparisons p.2; Lee2016ValidationToroidal p.3–4; Ogawa1988 jp.808 |
| Physics use of the edge value | Edge CES T_i at ρ_N ≈ 0.94 is a proxy for pedestal height (672 points, 8 discharges including two of ours) | Supports a physical reading of the edge value | Heo2025 jp.4–5 (Fig. 5b) |
| Main ion against carbon | AUG pedestal top: T_D − T_z ≈ 100 eV, 50–60 eV after extended corrections (#38540, raised ECRH). Equal in the steep gradient | Up to ~10–20 % of a 0.5 keV level, at the pedestal top | CanoMegias2025 p.9, p.19 |
| (same) | DIII-D separatrix: T_D+ as low as half of T_C6+. Equal away from the separatrix | Large only at the separatrix | Haskey2018MainIon p.1, p.6–7; Haskey2017 p.1; Grierson2016 p.1–2 |
| (same) | Pedestal transport conclusions change between carbon and main-ion T_i. Main-ion pedestal-top corrections are 150–300 eV (MICER, D-alpha, not C VI) | Supports claiming "the reported C VI value", not the main-ion T_i | Fenstermacher2025 p.20; Haskey2022 p.1; Haskey2026MachineLearning p.1 |
| Cross-section effect on the width | The width correction grows with T and reaches ~30 % at 30 keV. For B5+ at 200–500 eV it is 0.01–0.05 eV | Negligible at 0.5 keV. The supplement sentence has the direction reversed (audit list) | vonHellermann1995 p.3, p.9; Viezzer2012 p.8–9; Meister2001 p.2 |
| Cross-section effect on velocity | Apparent velocity of tens of km/s, growing with T_i. Calculated corrections overcorrect | Matters for V_rot only | Chen2019 p.3; Solomon2008 p.2–4 |
| Zeeman and fine structure | Ignoring them biases T. For C VI at 1 keV and 3 T a single Gaussian gives 1110 eV (> 10 %). The bias grows toward low T | Not read off for KSTAR's T and B; it varies slowly with T and B. Open | Blom2002 p.3–4 |
| Passive light (cold components, thermal CX, electron impact) | Must be separated from the active line. The passive C5+ line is mostly thermal CX and scales with pedestal C6+ density and edge neutrals | The main edge error source. Changes when the edge changes | Boileau1989 p.2–3; Isler1994 p.11; Ko2010KstarCesSystem p.1; Viezzer2011 p.17–18 |
| (same, in practice) | Edge real-time errors come mostly from unmodelled passive light. The two-Gaussian fit fails at the very edge. NN-CES accuracy falls beyond R ≈ 2.2 m (weaker active signal, ELMs, beam on/off differences) | Our chord (2.198 m) is where three methods report trouble | Chrystal2026 p.4; Lee2020TwoGaussian jp.788; Lee2026NN p.4 |
| Plume and halo | Plume emission is beam-induced and survives beam-off subtraction. Halo: T_i error below 1 %, location blurred 1.5–2.5 cm | Slow, beam-correlated bias, not frame noise | Bell2006 p.2–3; Pavlova2021 p.6 |
| Line blends | O VI under C VI 529 nm. Tungsten lines on C VI (JET-ILW) | Not established for 2022 KSTAR spectra | Hey2002 jp.1525–1527; Thorman2021 p.6 |
| Failed fits and outliers | KSTAR outliers tend to come from automatic Gaussian fits stuck in local minima, and are few. Least-squares CX fits fail or swap lines | The literature links outliers to failed fits. The link to low beam emission is our own (PK 4.21b, F1 3.18 [2.49, 4.04]) | Kim2024KineticProfile p.3, p.11; Bishop1993 p.2–5; Svensson1999 p.15–17 |
| Reported uncertainty | Error bar = fit standard deviation only. Median 0.080 keV (NN-CES values, #32364, 2–8 s at 50 ms). 0.09–0.14 keV at r/a 0.85–0.97 (31921). 15–25 eV inside 2.25 m and 70–350 eV beyond 2.26 m (2014 profile) | Our white component on typical rows is 39.1–42.8 eV, and the mean-square floor is 135.7 eV RMS (PK 4.21, 4.38) | Lee2022MeasurementCarbon p.6; Kim2026AutomatedOutlier p.7; Na2026 p.9; Ko2016 p.3 |
| Exposure and timing in the frame | 5 ms exposure; 5.5 ms CCD; 10 ms frames with a 5 ms average; "10 ms integration"; 100 Hz. One KSTAR paper allows ±5 ms on CES timing | Published values conflict. Our files place the exposure start at the stamp (PK §2) | Lee2020TwoGaussian jp.789; Ko2015 p.3; Ko2016 p.3; Jang2021 p.3; KFE2026 slides 60–63; Seo2026 p.4 |
| Background method | Interpolated between the two neighbouring beam-off frames, assuming a steady background over a few hundred ms. At DIII-D, the nearest ELM-free beam-off slice within 20 ms | Our test: no step at beam-off frames above 5.17 % of the level; the scatter does not grow with time since the last beam-off frame (PK 4.25) | Lee2020TwoGaussian jp.789; Lee2016 p.3; Haskey2018ActiveSpectroscopy p.4; KFE2026 slide 62 |
| Status of the stored values | Fitted and uploaded automatically, for preliminary analysis. `:err_bar` nodes exist | Fit errors exist in the database but are not in our files | KFE2026 slides 2, 62–63 |

**Where the chord sits**, as published for KSTAR:

| source | finding | key (page) |
|---|---|---|
| Kim2020 | The CES channels nearest the pedestal top are at R ≈ 2.19 m (before RMP) and ≈ 2.20 m (during RMP) | Kim2020PedestalElectron p.7–8 |
| Ko2016 | The knee of a 2014 inter-ELM profile is at 2.22–2.23 m | Ko2016 p.3, Fig. 2 |
| Ko2015 | LCFS ≈ 2.25 m in H-mode | Ko2015 p.3 |
| Lee2026Spat | In one 2022 equilibrium: separatrix ≈ 2.236 m; edge CES at 2.18–2.25 m with ~4 mm spacing; T_i knee at ψ_N 0.95–0.96. The figure does not name its discharge | Lee2026SpatiotemporalStructure p.4, p.6 |
| Na2026 | 31921: pedestal top at r/a ≈ 0.97 in H-mode and ≈ 0.92 in FIRE | Na2026 p.8–9 |
| Lee2026NN | CH16 at 2.192 m is called near the pedestal top (#28275, pre-2022) | Lee2026NN p.4–5 |
| Facility guide | From 2022 on: ch17 = 2192 mm, ch18 = 2198 mm | KFE2026 slide 63 |

Heo2025 and Na2026 use different radial coordinates: Heo's ρ_N is flux-based, while Na2026 uses r/a.

### 3b. Real-time CES/CXRS precedents: latency and resolution

**Precedents by device**

| device, year | system | cadence / resolution | latency | quantity | in a loop? | key (pages) |
|---|---|---|---|---|---|---|
| JET 1999 | NN on C VI spectra | per spectrum | real-time profile built | T_i (3–4 %) | — | Svensson1999 p.25 |
| JET 2003 | real-time CX | 50 ms, 11 channels | five-Gaussian fit < 50 ms | T_i(R), rotation | profile control | Joffrin2003 p.5–6 |
| JT-60U 2003 | filter CXRS | 0.8 ms (CCD 16.7 ms) | no fit | T_i, rotation | — | KOBAYASHI2003 p.1 |
| JT-60U 2007 | filter CXRS | down to 0.16 ms | — | central T_i | yes, within 0.5 keV | Yoshida2007 p.2, p.4 |
| JT-60U 2007 | real-time filter CXRS | 10 ms, 4 core radii | processing < 5 ms | T_i | grad-T_i feedback | Kobayashi2007 jp.S1049-1–3 |
| JT-60U 2009 | fast CCD | 2.5 ms | 4 spectra within 10 ms; < 5 % from offline | T_i, V_t | T_i gradient and rotation | Yoshida2009 p.4, p.8 |
| DIII-D 2007 | real-time CER, 2 chords | — | 10 ms filter + analysis (plus 12.5 ms beam averaging) | rotation | rotation + W | Scoville2007 p.5 |
| DIII-D 2016–2019 | real-time CER | — | — | rotation | error-field and rotation loops | Lanctot2016 p.5; Boyer2019 p.3 |
| DIII-D 2024 | RTCAKENN | 20 ms cycle | < 8 ms | kinetic equilibrium incl. carbon T_i | in the PCS | Shousha2023 p.17–18 |
| DIII-D 2026 | CERREAL, 24 views | 5 ms | ~1 ms after the data | T_i (core 88 %, edge 79 % within 20 %), rotation | sent to the PCS | Chrystal2026 p.1, p.4 |
| DIII-D 2026 | PCS inputs | CER profiles every 20 ms; BES at 1 µs | — | T_i, n, rotation | ML control framework | Rothstein2026 p.7 |
| NSTX 2012 / NSTX-U 2016 | real-time velocity | up to 5 kHz | — | v_φ (T_i only above 300 eV or after the discharge) | PCS at 5 kHz | Podesta2012 p.2–4; Podesta2016 p.4, p.9 |
| EAST 2019 | NN on spectra | per frame | 0.1 ms (fit ~100 ms) | T_i | — | Chai2019 p.2–4 |
| HL-3 2026 | CNN on spectra | 25 ms exposure | 0.56 ms per profile (fit ~90 ms) | T_i, v_t | — | Tian2026 p.2–3, p.15 |
| KSTAR 2020 | two-Gaussian fit | per frame | conventional analysis waits ≥ ~1 s for a beam-off frame | T_i, V_T | — | Lee2020TwoGaussian jp.794 |
| KSTAR 2026 | NN-CES | per frame | ~210 ns per frame batched (fit ~2 ms on CPU); conventional analysis waits up to 1 s | T_i, V_T | stated goal | Lee2026NN p.5 |
| KSTAR 2023 | XICS in LabVIEW | — | 1.6 ms per image after transfer | core T_i | outside the PCS | Kim2023LabviewProgramming p.4–5 |
| W7-X 2026 | real-time Thomson | 90 Hz | ~1 ms evaluation, ~11 ms end to end | T_e, n_e | steers a reflectometer | Mohammed2026 p.12, p.16, p.18 |

**KSTAR's real-time path**

| element | value | key (page) |
|---|---|---|
| PCS cycle | fastest 20 kHz; up to 9 real-time processes | Hahn2018 p.1–2; Hahn2020 p.1 |
| Network | reflective memory within 50 µs; ~100 µs pure delay | Park2011 p.1, p.4; Kwon2018 p.1–2; Hahn2018 p.2 |
| Real-time ML in the PCS | LSTM on D-alpha (20 kHz) and interferometer (10 kHz), 1 kHz cycle, ~0.8 ms per point. The authors say no direct pedestal diagnostic is available in real time | Shin2022 p.3–4, p.10 |
| Other loops | Edge control from D-alpha and real-time equilibrium. Density from an FPGA interferometer. Mirnov for NTM detection | Kim2022 p.2–4; Kim2024HighestFusion p.3; Juhn2026 p.1; Hahn2016 p.3 |
| Actuator delay | ~20 ms from the PCS to the ECH launcher | Hahn2016 p.3 |
| Our estimator | One step: median 0.49 ms on one CPU thread | PK 4.8 |

**Requirements and actuator time scales**

| requirement | value | key (page) |
|---|---|---|
| ITER edge T_i | 100 ms, 10 % (r/a > 0.9) | Donne2007 p.9 (jp.S343) |
| ITER edge T_i | 100 ms, 10–20 mm, 10 % (r/a > 0.85) | Mazon2025 p.27 (jp.26) |
| ITER CXRS | 100 ms and 10–30 %. Beam modulation relaxes it to 200 ms (burst) and 1.5 s (control) | Mazon2025 p.66 (jp.65) |
| FIRE design, edge T_i | 10 ms, 5 mm, 10 %, by CXRS | Young2008 p.9–10 (jp.288–289) |
| ITER Physics Basis 1999 | ~100 ms for most profile studies; ~1 ms for transients | Diagnostics1999 p.6 |
| T_i control cycle at JET and JT-60U | ~10 ms (secondary citation) | Chai2019 p.3 |
| Actuators | No faster than ~10 ms. Cycle target 1–50 ms. Rotation evolves over ~10 ms with actuators at 10–50 ms. NBI stepped every 20 ms in an RL T_i-gradient controller | Rothstein2026 p.6; Podesta2012 p.2; Wakatsuki2021 p.4 |

### 3c. Beam blips and beam-off interruptions

**Literature summary:**

- KSTAR blanks the beam for ~10 ms to measure the CES and MSE background. The core T_e and stored energy fall by under 10 %.
- Published KSTAR blips displace the plasma and perturb T_i, mostly in the core. One blip helped form an ITB.
- Beam ions keep heating for tens of ms after the source stops: classical slowing-down, ~13–40 ms.
- A beam step also perturbs the electrons promptly (DIII-D, core).

**Our files next to the literature** (PK 4.7, 4.17, 4.41):

- The reported T_i falls to 0.819 of its earlier level in the first row after a blip. ECEI falls to 0.926, BES to 0.656.
- T_i and ECEI recover over ~41 ms.
- The first frame carries an extra 0.030 of the level.

**What is known about the blips**

| fact | value | key (pages) |
|---|---|---|
| Blank length and purpose (KSTAR) | ~10 ms for the CES and MSE background; interlock retrigger 10–20 ms | Kwak2019 p.2 |
| Effect on the core | Central T_e and stored energy fall by under 10 %. The whole beam was blanked in the example (#18483) | Kwak2019 p.2–3 |
| Modulation rate | 1–5 Hz with 10 ms off; 1 Hz; 1 or 2 Hz at 99 % duty; 5 Hz at 95 % duty (2010-era) | Lee2020TwoGaussian jp.788–789; Lee2026NN p.2; KFE2026 slide 60; Ko2015 p.3 |
| Which source | NB1-B dropped about once a second (1.0 of 2.7 MW), 2017-era. NB1A unusable in 2022. In 31923, P_NBI drops by ≥ 1.1 MW from ~3.5 MW. MSE is aligned with NB1-A; NB1-B is the most efficient heater | Kim2018 p.3; Han2024 p.5; Heo2025 jp.4 (Fig. 2b); Lee2021SystematicEvaluation p.1 |
| Coupling of sources | Switching one source changes the power transmitted by the others (shared beam box) | Kwak2025 p.1 |
| Plasma response at KSTAR | Displacement and T_i fluctuations, mostly in the core. A blip near 5.0 s acted as a perturbation that helped form an ion ITB | Kim2018 p.3; Nam2026 p.10 |
| Plasma response elsewhere | Beam modulation perturbs the plasma and blinds beam diagnostics. At DIII-D, prompt T_e cooling at turn-on and heating at turn-off (core T_e −0.1 keV above 2 keV). NBI modulation is used on purpose to perturb rotation | Boyer2019 p.2, p.4; Zhao2023 p.1, p.10; Tala2009 p.1; Zimmermann2026 p.5 |
| Fast-ion slowing-down | Critical energy, slowing-down and thermalisation times (9.3 keV, 33 ms, 39 ms at the supplement's inputs). Classical within ~30 % after 2 ms pulses. ~13 ms e-folding after 5 ms blips (HL-2A, ohmic). ~75 % of beam energy to ions at E_crit | Heidbrink1994 p.17; Heidbrink1988 p.2–3; Zhang2012 p.4–6; Stix1972 p.9–10 |
| Diagnostic side | BES light is beam emission, so a blip removes it. Its background comes from NBI turn-off. In one edge analysis, CES T_i is usable only between two blips. Beam-on and beam-off spectra differ at the edge | Nam2014 p.2; KFE2026 slides 57–59; Heo2025 jp.3; Lee2026NN p.4 |
| Misattribution risk | Lee2016's "blips" are 10–20 ms beam-on pulses into ohmic plasma (the rotation jumps). They are not beam-off interruptions | Lee2016ValidationToroidal p.3 (Fig. 4) |

### 3d. What BES, ECEI and Mirnov carry about T_i

**Short answer.** The literature ties BES to the edge density and to fluctuation spectra. Its links to the pedestal state are spectral: band power, coherent modes, regime and ELM classification at 1 MHz. Our 10 ms rows hold one raw sample per channel (PK §2), so those spectral features are not in our inputs.

The ECEI level at the edge mixes density and temperature through the optical depth. No paper ties the Mirnov amplitude to the pedestal-top T_i level.

These match what we measure (PK 4.44):

- The present-frame BES level carries the increment: BES with the CES history gives +0.044 [+0.020, +0.066] over the twin.
- ECEI with the history carries none: −0.011 [−0.092, +0.022].
- The network without Mirnov loses nothing: +0.060 [+0.023, +0.085] over the twin.

| channel | what it measures at the edge | links to the pedestal or T_i in the literature | keys (pages) |
|---|---|---|---|
| BES | Beam emission that follows the local n_e (T_e effect < 5 %). Part of the low-frequency signal is beam attenuation by SOL density. The level drifts with delivered beam power (BES −20 % over 9 s at steady density) | The intermediate-frequency amplitude falls as the T_i pedestal height rises. A weakly coherent mode near 50 kHz. Ion-scale fluctuations are stronger where the ion heat flux exceeds neoclassical. 1 ms windows classify L, H and QH (F1 0.94). MHz BES forecasts ELMs across devices. A coherent edge mode in the recovery phase | KFE2026 slides 57–59; Lampert2015 p.8–9; Hong2024 p.12–13; Heo2025 jp.4–5; Haskey2022 p.1; Gill2024 p.2, p.8; Joung2026 p.1–2; Lee2023 p.8; Nam2014 p.2–3 |
| ECEI | The edge radiation temperature is not local. Optical thickness scales with n_e T_e; shine-through; density fluctuations modulate it. The KSTAR pedestal is grey or thick (τ ≳ 1 for 500–200 eV and 2–0.2·10¹⁹ m⁻³), with ~1 cm mapping error | EHO at ~4 kHz in 32092. Pedestal radiation drops before ELMs. Edge images may read as density | Vanovac2018 p.2–3; Denk2018 p.2–3; Rathgeber2013 p.3–4; Yu2022 p.4–8; Zhu2019 p.13; Yun2011 p.5; Yun2010 p.2; Lee2026Spat p.13 |
| Mirnov | MHD modes | NTM detection in the KSTAR PCS. A 2/1 NTM in #31747. Coherent edge mode. Tearing and locking theory. No link to the pedestal-top T_i level | Hahn2016 p.3; Cha2024 p.2–5; Lee2023 p.8; Fitzpatrick1993; LaHaye2006 |
| Time scales | After an ELM, the T_i gradient and E_r recover in 2–4 ms. The pedestal-top T_i rises through the period between ELMs. KSTAR ELM periods are 40–70 ms. The rotation pedestal forms before the T_i pedestal | A 10 ms frame mixes crash and recovery. Our reach saturates at 50 ms (PK §3) | Cavedon2017 p.1, p.10–11; Viezzer2013 p.10–11; Viezzer2018 p.6–7; Viezzer2020 p.3–4; Ko2016 p.3; Ko2015 p.2 |

### 3e. The closest prior ML work and how ours differs

**Our estimator** estimates the reported C VI T_i of the present frame at one pedestal-top chord. It is causal and reads two things: the fast channels of the present frame and the CES history. It is scored on withheld measured frames against hold, a causal GP, PCHIP and an offline GP, with a discharge-level bootstrap.

**The two nearest lines of work:**

- **NN-CES** (Lee2026NN) reads the spectrum of the frame. It needs the frame to exist and, at the edge, to be fitted well. Its lineage is Bishop1993 → Svensson1999 → Chai2019 → Lee2026NN and Tian2026.
- **Diag2Diag** (Jalalvand2025) reconstructs a target from other diagnostics without the target's own history. It is memoryless and smooths non-causally.

No paper in the corpus estimates a CXRS T_i frame without a usable value causally from fast channels plus the CES history.

| work | device | target | inputs at inference | own history? | causal? | time base | evaluation |
|---|---|---|---|---|---|---|---|
| Lee2026NN (NN-CES) | KSTAR | T_i, V_T of the frame | that frame's spectrum | no | yes (per frame) | 10 ms frames | against the conventional fit; 321 discharges split 80/10/10 by shot |
| Tian2026 | HL-3 | T_i, v_t | spectra (32 ch) | no | yes | 25 ms | R² 0.93 for T_i against the fit |
| Chai2019 | EAST | T_i | spectra (2 chords) | no | yes | ~5 ms edge | ~3 % on 95 frames of one shot |
| Svensson1999; Bishop1993 | JET | T_i, rotation | spectra | no | yes | per spectrum | 3–4 % (Svensson) |
| Lin2024; Pavone2019; Haskey2026 | EAST; W7-X; DIII-D | T_i (XCS, XICS, MICER) | spectra | no | yes | per frame | against fits or Bayesian inference |
| Jalalvand2025 (Diag2Diag) | DIII-D | Thomson T_e, n_e at 1 MHz | ECE, interferometer, magnetics, CER, MSE (+ derivatives) | no | no (smoothed) | 1 µs | R² 0.92 on Thomson times |
| Imrek2026 | COMPASS | Thomson T_e, n_e | SXR, AXUV, global signals | no | memoryless | µs | MAPE, edge ~11 % |
| Shousha2023 (RTCAKENN) | DIII-D | kinetic equilibrium incl. carbon T_i | real-time magnetics, MSE, Thomson, CER | CER as an input | yes | 20 ms | against offline CAKE; only qualitative without CER |
| Wang2025TimeSeries | EAST | one ECE T_e channel | 15 signals incl. offline EFIT | no | no | 1–20 kHz | > 95.9 % within 2σ |
| Jia2026 | EAST | Z_eff | exogenous signals (BiGRU) | no | no | — | similarity 90.3 %, coverage 91.4 % |
| Yang2026 (FusionMAE) | HL-3 | any of 88 masked signals | the other signals | no | no | — | reconstruction 97.2 % |
| Boyer2021 | NSTX-U | n_e and pressure shapes | actuators + EKF with real-time Thomson | via the EKF | yes | 17 ms Thomson | profile error |
| Abbate2021/2025 | DIII-D (+AUG) | profile change 200–300 ms ahead | profiles + actuators | yes | yes | 50 ms | against "constant" (hold wins far out of distribution) |
| Morosohk2023 | DIII-D | T_e profile (T_i ∝ T_e) | Thomson + NN-surrogate transport | yes (filter) | yes | — | consistency with Thomson |
| Pastore2023/2026 | TCV | n_e profile | Thomson + FIR + transport model | yes (filter) | yes | 1 ms | control use |
| Jung2026 | KSTAR | profile parameters | Thomson, TCI, CES | no | between shots | — | against MCMC fits |
| Kim2024KineticProfile | KSTAR | CES T_i profiles | CES points (SVM + GP) | yes | no | 10 ms | outlier F1 |

## 4. Experiment ideas from the literature

The table merges the EXPERIMENT IDEA entries of the four notes; 43 ideas remain after duplicates are removed. The last column was checked against `PROJECT_KNOWLEDGE.md` (PK) §3, §4.x and §7.

- "Log only" marks a record that sits in `THESIS_RESULTS.md` but has not been promoted to PK, so it is not a ground for a next step.
- Numbers are quoted only from PK.

### 4.1 All ideas

**A. Inputs**

| # | idea | source keys | what it decides | CPU/GPU | already measured? |
|---|---|---|---|---|---|
| A1 | Learned GRU-D decay of the carried CES value (optionally of the hidden state too) | Che2018, Weerakody2021, Baytas2017 | Does an explicit decay help after long gaps? | GPU | Partly. GRU-D-type decay was one of four continuous-time encoders in the W = 4 window family, with no keep (log §8e; PK §8 calls it reference-only). Never tried on `seq_v2` |
| A2 | Observation-intensity input (number of CES values in the last 50 and 150 ms) | Shukla2019 | Does the history branch lack staleness information? | GPU | Partly. Staleness and a has-observation flag are already inputs |
| A3 | Clock since the last absent frame, plus a 90 ms post-blip flag | Lipton2016 | Is the post-blip recovery a learnable residual? | GPU | No. The rows 1–3 after an absent frame carry the largest per-row increment (+0.151 [+0.093, +0.215], PK 4.33) |
| A4 | Burst-phase input: time since the last BES burst, plus a running burst rate | Cavedon2017, Shin2022 | Does the phase between ELMs carry T_i information that the LSTM does not extract? | GPU | No. Bursts were read only as intervals and classes (PK 4.6, 4.43) |
| A5 | Rule that masks CES history entries in burst rows and in the first row after an absent frame | Santos2010 (+ Lee2026NN p.4) | Do those CES values hurt as inputs? Gives a rule baseline for B.67's learned gate | GPU | No. B.67 (queued) is the learned counterpart |
| A6 | Causal per-discharge normalisation of the fast channels (running baseline) | Hong2024, Jeong2026, Joung2026, Pastore2023/2026, Felici2014 | Can the gain of whole-discharge normalisation be recovered causally? Is slow beam drift noise? | GPU | Partly. The filter reads changes and channel differences (PK 4.30, 4.37). Whole-discharge normalisation against causal: −0.026 [−0.051, +0.001] (PK 4.32). No causal running version trained |
| A7 | Short causal average in place of the raw snapshot | Imrek2026 | Does snapshot noise limit the increment? | CPU | Largely. 10 ms average −0.034, 1 ms average +0.008 (PK 4.19). Samples just before the stamp (PK 4.34). Older rows carry nothing (PK 4.44) |
| A8 | Causal first and second differences; an MLP on them against the LSTM | Jalalvand2025 | Level or rate of change? Is recurrence needed? | GPU | Partly. KFX moves with the changes of BES (PK 4.37). A memoryless tree regressor exists in the log only. No MLP trained |
| A9 | Separate attenuation from local emission (ratio of inner to outer columns) | Lampert2015 | Local pedestal signal, or path-integrated density? | — | Yes. Radial difference g (PK 4.30 B.51; 4.39 B.53; Kg in 4.37 B.56) |
| A10 | Optical-depth proxy as an input (BES × ECEI, or ECEI / BES) | Vanovac2018 | Why does ECEI carry no increment? | GPU | No |
| A11 | Band-limited BES power (10–30 kHz, ~50 kHz) from the raw streams | Heo2025, Gill2024 | Does turbulence carry pedestal height beyond the BES level? | CPU (12 discharges) | No. Raw streams exist for 12 discharges only (PK §2), so a model cannot be trained on them. Raw Mirnov features were negative (PK §8) |
| A12 | T_i history or estimate fed into the V_rot branch | Chen2019 | Use the cross-section coupling between velocity and T_i | GPU | No (PK §3). Rotation is supplementary |

**B. Training and loss**

| # | idea | source keys | what it decides | CPU/GPU | already measured? |
|---|---|---|---|---|---|
| B1 | Auxiliary training-only heads: the offline-GP estimate at t (with t withheld), and the T_i of the next frame | Caruana1997 | Does a denoised or privileged target raise causal skill? | GPU | No. Listed in PK §7 as unregistered ("denoised auxiliary target", "auxiliary tasks") |
| B2 | Robust training loss (Huber, MAE) | Tian2026 | Do outlier targets cost skill? | — | Yes. B.14: median −8 to −10 %, MSE unchanged, not adopted (PK §3; §8ay, §8bu) |
| B3 | Heavy-tailed or distributional head (Student-t, mixture, Gaussian CRPS); the predicted scale as the filter weight | Gebetsberger2018, Rasp2018, Roth2013, Kim2026AutomatedOutlier | Calibrated per-row uncertainty; replace the hard cut | GPU | Largely. B.17 and B.17b mixture and Student-t heads (log only, §8bf, §8bj); PK §3 lists Student-t among the losses compared. The scale as a filter weight is not measured |
| B4 | Down-weight unreliable labels in training only (post-blip rows, burst rows, rows flagged by the value rules) | Lee2026NN, Kim2024KineticProfile | Does edge label noise limit training? | GPU | No. Only scoring-side readouts exist (PK 4.5, 4.20, 4.33) |
| B5 | Modality dropout: zero the BES or ECEI block on 10 % of training rows, with a flag | Shousha2023 | Robustness to missing BES; does it regularise? | GPU | No |
| B6 | Self-supervised pretraining of the fast-channel encoder | Yang2026, Chen2026 | Does the encoder limit skill? | GPU | No. Encoder redesign is in PK §7, unregistered; the clustering route was negative (PK 4.15) |
| B7 | Train with a real-time CES delay of 1–2 frames | Scoville2007, Rothstein2026 | How much survives the pipeline delay? | GPU | Frozen replay measured (PK 4.23, 4.28). Training with a 2-frame delay was registered as B.47 L and cancelled before any run |
| B8 | Train with a thinned CES (100–200 ms cadence, as at ITER) | Mazon2025 | How much skill remains at an ITER-like cadence? | GPU | Frozen replay up to 100 ms (PK 4.4, 4.28). Training with thinning is a PK §7 candidate |

**C. Architecture**

| # | idea | source keys | what it decides | CPU/GPU | already measured? |
|---|---|---|---|---|---|
| C1 | Per-row source gating or gated attention pooling, with readable weights | Ilse2018 | Is BES used selectively (e.g. after absent frames)? | GPU | Queued: B.67, registered 2026-11-19, runs after B.65 |
| C2 | Late fusion: one head per source, combined by a gate | Baltrusaitis2018 | Does the BES increment need to interact with the history inside the LSTM? | GPU | No |
| C3 | Attention (mTAN-style) encoder for the CES history | Shukla2021 | Is the history branch limited on long gaps? | GPU | Partly. At ≥ 150 ms context, attention, LSTM and TCN end within 0.023 (PK §3, §8ag). The W = 2 control uses masked attention |
| C4 | Continuous-time encoders on the full grid (CDE, ODE-RNN, GRU-ODE, Phased LSTM) | Kidger2020, Rubanova2019, DeBrouwer2019, Neil2016 | Is the Δt structure on the full grid unused? | GPU | Partly. Window family, no keep (log §8e) |
| C5 | Bidirectional (offline) network as an upper bound | Cao2018, Jia2026 | How much of the error is the price of causality? | GPU | No network. Offline GP against the network: −0.037 [−0.068, −0.008] (PK 4.32) |

**D. Filters and references**

| # | idea | source keys | what it decides | CPU/GPU | already measured? |
|---|---|---|---|---|---|
| D1 | Robust filter update instead of the hard 3 keV cut (WoLF, variational Student-t, innovation gating, 3σ rejection, redescending) | DuranMartin2024, Agamennoni2012, Blanken2019, Masreliez1977, Roth2013, Karlgaard2007 | Can a bounded-influence update make the cut unnecessary? | CPU | Partly. KFX uses a robust update on the cut population (PK 4.37). A Student-t measurement model was tried in the physics filters (log only, §8bm) |
| D2 | Online per-discharge offset or gain state, updated at each CES arrival | Felici2014, Pastore2023/2026 | Can it remove the per-discharge drift of the fast-channel mapping? | CPU, then GPU | No |
| D3 | Linear state-space reference with BES and ECEI as exogenous inputs | VanMulders2025 | Does the increment need a nonlinear network? | — | Yes. KFX: the network over KFX is +0.036 [+0.012, +0.058] (PK 4.37). Linear filter (PK 4.31) |
| D4 | IAR/CAR(1) history-only reference; autocorrelation of the residuals | eyheramendy2018, elorrieta2019 | Is the causal GP a strong history reference? Do the residuals keep temporal structure? | CPU | Partly. KF0 is equivalent to the causal GP (PK 4.37). Residual autocorrelation not measured |
| D5 | Best-tuned causal GP (fixed or marginalised hyperparameters) | michoski2024 | Does the margin survive the strongest GP? | CPU | Partly. Stronger history-only references exist: twin over causal GP +0.043 [+0.023, +0.062] (PK 4.32); KF0 equals the causal GP (PK 4.37). GP variants not tried |
| D6 | Two-mode IMM filter (quiet / burst) | vandenBrand2016 | Does burst-conditioned relaxation explain the increment? | CPU | Partly. Event gate (B.16) and hold-or-jump (B.18), log only (§8bg, §8bk; PK §8 marks them reference-only) |
| D7 | Learned convex combination of the estimator with hold and the causal GP | Abbate2025 | Do the references hold complementary information? | — | Yes. +0.021 … +0.037, 4/4 (PK 4.9). Against the second stage (PK 4.18) |
| D8 | Novelty score of the inputs as a reliability flag, falling back to the twin | Svensson1999 | Are the large errors out of distribution? | CPU | Partly. Error-size model and switcher (PK 4.16, 4.18). A density-based score is not measured |

**E. Readouts and analyses (no training)**

| # | idea | source keys | what it decides | CPU/GPU | already measured? |
|---|---|---|---|---|---|
| E1 | At inference, forward-fill, zero or add noise to the present-row BES | Gupta2026 | Does the estimator degrade gracefully when the latest BES sample is missing? | CPU | Largely. BES one row old gives twin level, +0.000 [−0.024, +0.022]. All fast channels one row old fall below the twin, −0.027 [−0.050, −0.008] (PK 4.34). Zeroing and noise not tested |
| E2 | Skill by burst phase (0–10, 10–30, 30–70 ms since the last burst) | Ko2016, Viezzer2013, Viezzer2018 | Where in the ELM cycle does the increment live? | CPU | Partly. Burst intervals (PK 4.6), burst onsets (PK 4.20 U1), classes (PK 4.43: between bursts +0.050 [+0.027, +0.075]). Phase bins not read |
| E3 | Error and label jitter by position in the modulation cycle | Lee2020TwoGaussian | Is part of the residual an artefact of background interpolation? | CPU | Partly. The scatter does not grow with time since the last beam-off frame, 1.013 [0.982, 1.051] (PK 4.25). Estimator error by cycle position not read |
| E4 | Split 31921 and 31923 by phase (FIRE against H-mode) | Na2026 | Does the gain depend on where the chord sits on the profile? | CPU | No (two discharges) |
| E5 | Case study of #31747 (2/1 NTM, ECCD after 7.3 s): full, twin and no-Mirnov residuals | Cha2024 | Does "Mirnov can be dropped" hold with a strong mode? | CPU | No |
| E6 | Murphy decomposition of the skills into correlation, conditional bias and mean bias | murphy1988 | Is the BES increment better tracking, or less bias? | CPU | Partly. The correction is too large: slope 0.693 [0.589, 0.790] (PK 4.38). Full decomposition not done |
| E7 | Fixed-point and linear-response analysis of the T_i LSTM | Sussillo2013 | Time constants and input gains: a mechanism statement | CPU | Partly. Jacobians and lag profiles of the earlier whole-discharge backbone (log only, §8bz-3). Not done on the causal variant |
| E8 | Granger screening of the 15 channels and their lags | Wang2025TimeSeries | Which channels carry information, without a network? | CPU | Partly. Network ablations (PK 4.34, 4.44) |

**F. Statistics**

| # | idea | source keys | what it decides | CPU/GPU | already measured? |
|---|---|---|---|---|---|
| F1 | Nested CV; Nadeau–Bengio corrected variance | Bates2024, Nadeau2003 | Do the pooled intervals undercover? | CPU | Nadeau–Bengio done: fold interval in PK 4.32. Nested CV not done |
| F2 | Crossed variance over discharges × initialisations (pigeonhole bootstrap, unbiased U-statistic) | Owen2007, Gallas2009, Ren2010, Saravanan2020 | How wide should the pooled and split intervals be? | CPU | Partly. Two-level bootstraps (PK 4.26, 4.32) |

### 4.2 Ranking of what has not been measured

**Criteria.** (i) How directly the idea could raise the skill of the estimator or explain its mechanism; (ii) its cost. Ideas that are already measured, largely measured or queued are left out: A7, A9, B2, B3, B7, B8, C1, D3, D7, E1. Partly measured ideas are ranked on the part still open.

**GPU queue.** B.64 is running; B.65 and B.67 are queued (PK §7). Every GPU item below would be registered after B.67's readout, and its result would be read against B.67.

| rank | idea | (i) directness | (ii) cost | why here |
|---|---|---|---|---|
| 1 | B1 auxiliary denoised target | skill | 15 GPU runs; one head and one loss term | Three quarters of the MSE is scatter of the reported value (PK 4.38). The offline GP is lower than the network (PK 4.32) |
| 2 | E6 Murphy decomposition | mechanism | CPU, no training | Splits the increment into tracking and bias. The slope 0.693 already shows a conditional bias |
| 3 | A6 (+D2) causal per-discharge normalisation | skill | CPU filter check, then 15 GPU runs | Targets a measured gap (PK 4.32) and the beam drift seen in the BES level |
| 4 | A4 (+E2) burst-phase clock | mechanism, skill | CPU readout, then 15 GPU runs | The ELM period (40–70 ms) is longer than the 50 ms reach. The increment sits between bursts |
| 5 | A5 rule-based masking of suspect history entries | mechanism (target validity) | 15 GPU runs | Tests the premise directly. Baseline for B.67 G4 |
| 6 | E7 fixed points and linear response, causal variant | mechanism | CPU | Promotes a mechanism statement from the log to PK |
| 7 | B4 down-weight flagged labels in training | skill | 15 GPU runs | The increment lives on rows the value rules do not flag (PK 4.33) |
| 8 | B5 modality dropout | robustness, skill | 15 GPU runs | Real-time fallback; possibly a regulariser |
| 9 | C5 bidirectional upper bound | mechanism (cost of causality) | 15 GPU runs | Fixes the ceiling between the network and the offline GP |
| 10 | D5 best-tuned causal GP | claim robustness | CPU | A referee will ask; it does not raise skill |
| 11 | A3 post-blip clock | skill | 15 GPU runs | The LSTM already sees the absence flags, so the expected gain is small |
| 12 | B6 self-supervised encoder | skill, encoder | high: new pretraining code | T_i is measured on 92 % of grid rows, so few unlabelled rows exist. MAST rows would be the pool |
| 13 | A10 optical-depth proxy | mechanism (ECEI) | 15 GPU runs | Explains a null; unlikely to add skill |
| 14 | C2 late fusion | mechanism | 15 GPU runs | Run after B.67, depending on its gates |
| 15 | D1 robust filter without the cut | claim robustness | CPU | Moves the cut question to the filter |
| 16 | E8 Granger screening | mechanism (network-free) | CPU | Mostly answered by B.59 |
| 17 | D4 residual autocorrelation | mechanism | CPU | Shows whether temporal structure is left |
| 18 | A8 MLP on differences | simplicity | 15 GPU runs | The filter already covers most of it |
| 19 | E3 error by modulation-cycle position | mechanism (label artefact) | CPU | Partly answered by PK 4.25 |
| 20 | F1, F2 nested or crossed variance | statistics only | CPU | Interval width, not skill |
| 21 | E5 #31747 case study | anecdote | CPU | One discharge |
| 22 | E4 31921/31923 phase split | anecdote | CPU | Two discharges |
| 23 | A1 GRU-D decay on `seq_v2` | skill | 15 GPU runs | Tried without a keep in the window family (log §8e); staleness is already an input |
| 24 | A2 observation intensity | skill | 15 GPU runs | Staleness is already an input |
| 25 | C3 mTAN history encoder | skill | 15 GPU runs | The operator family did not decide skill (PK §3) |
| 26 | C4 continuous-time encoders, full grid | skill | 15+ GPU runs | Window-family negative |
| 27 | D6 IMM filter | mechanism | CPU | Close variants were negative in the log |
| 28 | D8 density-based novelty fallback | skill | CPU | The switcher variants are done |
| 29 | A12 T_i into the V_rot branch | rotation only | 15 GPU runs | Supplementary target |
| 30 | A11 raw band power | mechanism | CPU, 12 discharges | Cannot be trained here |

### 4.3 Top five: design sketches

**Common protocol** (as in B.59 and B.67):

- the 15 folds of re-partitions r2, r3 and r4 of the B.39 pooled cross-validation;
- `run_b39.pooled_env` for each fold: cut population, cap 80 with early stopping, `CES_PER_SHOT_NORM=0`, every `CES_*` variable pinned;
- one initialisation per fold;
- the stored B.39 network and twin of the same folds as references;
- readout by `analyze_b39.arm_p`, a two-level bootstrap with 20,000 draws, on the rows B.39 scored.

Each design changes one thing and is registered before any run.

**1. Auxiliary denoised target (B1).**

- *Source.* Caruana1997 (p.15–16): features available only after the prediction can serve as extra outputs during offline training.
- *Change.* The T_i branch of `seq_v2` gains one output head, used in training only. It predicts the offline-GP estimate of T_i at row t, computed from the measured values on both sides with row t withheld (the same construction as the offline-GP reference). It is computed from training-discharge data only.
  - The loss becomes: masked MSE on the reported value + λ × masked MSE on the auxiliary target. λ is fixed at registration (say 0.5) and not tuned.
  - Inference and scoring are unchanged; the head is discarded.
- *Statements.*
  - X1: the arm over the network, interval above zero.
  - X2: the arm over the twin, interval above zero.
  - X3: the arm's median absolute error over that of the causal GP, interval of the ratio below 1.
- *Why.* Three quarters of the network's MSE is scatter of the reported value (PK 4.38). The target is noisy at the frame level while its neighbours are informative; the offline GP is lower than the network (PK 4.32). An auxiliary target denoised by the neighbours should pull the network toward the part of the value it can predict.
- *If the outcome is null.* The frame scatter is not what limits the fit, and the remaining error is information the inputs lack.

**2. Murphy decomposition of the pooled skills (E6, CPU).**

- *Source.* murphy1988 (jp. 2418).
- *Change.* No training; the analysis compares two estimators on the same rows. On the r2–r4 rows of B.39:
  - Compute the predictions of the network, the twin, the causal GP and hold. Use stored predictions where they exist; otherwise re-run inference on the CPU, as B.55 did.
  - Within each discharge, decompose MSE into: mean bias, the conditional-bias (slope) term, and the correlation term.
  - Sum over discharges and pool with the same bootstrap.
- *Statements.*
  - M1: the MSE reduction of the network against the twin comes mostly from the correlation term (share above 0.5, interval above 0.5).
  - M2: the conditional-bias term is larger for the network than for the twin (consistent with the slope 0.693 of PK 4.38).
- *What it decides.* Whether the BES increment is better tracking of changes or a level correction. That is a mechanism sentence for the paper at no GPU cost, and it can run while the GPU is busy.

**3. Causal per-discharge normalisation of the fast channels (A6, with D2 as a CPU pre-check).**

- *Sources.*
  - Hong2024 (p.12–13): the BES level falls ~20 % at steady density as the delivered beam power drifts.
  - Joung2026: a per-channel slow-baseline normalisation lets a BES model transfer between devices.
  - Pastore2023 (p.3–4) and Felici2014 (p.4–6): online correction of offsets.
- *CPU pre-check (D2).* Add a random-walk offset state on the BES gain of the five-parameter filter KFX and read KFX′ against KFX on r3–r9, as in B.56.
- *GPU arm.* Replace the training-file standardisation of the 15 fast channels by a causal per-discharge standardisation: at row t, use the mean and standard deviation of that channel over the discharge's rows up to t (expanding window). Training statistics are used until a minimum count fixed at registration. The CES inputs are unchanged.
- *Statements.*
  - N1: the arm over the network, interval above zero.
  - N2: the arm over the twin, interval above zero.
- *What it decides.* Whether the gap to whole-discharge normalisation (−0.026 [−0.051, +0.001], PK 4.32) can be closed causally, or whether that variant's edge was information from the future.

**4. Burst-phase clock (A4, with E2 as its CPU readout first).**

- *Sources.*
  - Cavedon2017 (p.10–11): the pedestal-top T_i keeps rising until the next ELM.
  - Ko2016 (p.3): KSTAR ELM periods are 40–70 ms.
  - Shin2022: the KSTAR controller conditions on phase.
- *CPU readout first (E2).* On the stored B.39 r2–r4 errors, read the increment by time since the last burst (0–10, 10–30, 30–70, > 70 ms). Bursts come from the B.66 rule, made causal (trailing window).
- *GPU arm.* One added input to the T_i branch: log(1 + time since the last burst), clipped at 1 s.
- *Statements.*
  - C1: the arm over the network, interval above zero.
  - C2: on rows 10–70 ms after a burst, the arm over the network, interval above zero.
- *What it decides.* The ELM period is longer than the 50 ms reach at which the network saturates (PK §3). A gain means phase information beyond the network's memory. A null means the LSTM already extracts it from BES, or that phase carries no T_i information at 10 ms. The established increment between bursts (PK 4.43) is the context.

**5. Rule-based masking of suspect CES history entries (A5).**

- *Sources.*
  - Santos2010: real-time reflectometry rejects samples taken during ELMs and sets the Kalman gain to zero.
  - Lee2026NN (p.4): edge spectra are distorted by ELMs and by beam-on/off differences.
  - Lee2020TwoGaussian (jp.789): the background is interpolated across the modulation cycle.
- *Change.* In the CES-history inputs only, the value and the has-observation flag of two kinds of frame are set to "not observed": the first row after an absent frame, and rows flagged as bursts by the B.45 rule. The network then carries the previous value. Targets and scoring are unchanged.
- *Statement.* R1 is two-sided, for the arm against the network.
  - Above zero: those CES values hurt as inputs, i.e. they are less temperature-like than their neighbours. This bears on L3 and on the post-blip first-frame excess of PK 4.41.
  - Below zero: they carry information the network uses.
  - R2 is the same contrast on rows 1–3 after an absent frame.
- *What it decides.* It is the rule baseline for B.67's learned gate and statement G4, so it runs after B.67 has been read.

## 5. Where the notes or sources disagree

| # | disagreement | between | status |
|---|---|---|---|
| 1 | Theme tags of five items were inconsistent: ECE-interpretation papers sat under `other` or `KSTAR-facts`, slowing-down papers under `CES-diag` or `pedestal-physics`, requirement documents in three tags | readers | Re-tagged (2.9) |
| 2 | Published CES exposure: 5 ms (Lee2020TwoGaussian jp.789), 5.5 ms CCD (Ko2015 p.3), a 5 ms average in 10 ms frames (Ko2016 p.3), "10 ms integration" (Jang2021 p.3) | sources | Unresolved. Probably campaign-dependent; the supplement already says the values conflict |
| 3 | Beam modulation: 5 Hz at 95 % duty (Ko2015, 2010-era), 1–5 Hz (Lee2020), 1 Hz (Lee2026NN), 1 or 2 Hz at 99 % duty (KFE2026) | sources | Campaign-dependent; not a contradiction for 2022 |
| 4 | How much beam a blip removes: the whole beam (Kwak2019, #18483), one source of 2.7 MW (Kim2018), ≥ 1.1 of ~3.5 MW (Heo2025, 31923, 2022) | sources | Unresolved for 2022 |
| 5 | Channel index of R = 2.192 m: CH16 in Lee2026NN (checked in the PDF, p.5: discharge #28275, before 2022); ch17 in KFE2026 slide 63 (from 2022 on) | sources | Numbering or radius is campaign-dependent. Our chord is ch18 = 2198 mm (KFE2026) |
| 6 | Pedestal-top location: R ≈ 2.19–2.20 m (Kim2020); knee at 2.22–2.23 m (Ko2016, 2014); separatrix ≈ 2.236 m (Lee2026Spat, one 2022 equilibrium). ρ_N (Heo2025, flux-based) and r/a (Na2026) are different coordinates | sources | Campaign- and discharge-dependent; consistent with PK §2 (0.80 nominal, 0.95 from two discharges) |
| 7 | Viezzer2012's cross-section temperature correction of 0.01–0.05 eV looked implausibly small | note against plausibility | Checked in the PDF (p.8): printed as such |
| 8 | Statistics: fold-correlated CV intervals undercover (Bates2024, Nadeau2003), while crossed multi-level bootstraps can overcover (Gallas2009) | sources (opposite failure modes) | Both bear on our two-level intervals. Ideas F1 and F2 |
