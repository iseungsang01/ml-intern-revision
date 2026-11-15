# Nuclear Fusion acceptance bar for an ML-for-diagnostics paper, and the gap list for the CES gap-filling paper

Date: 2026-11-15. Inputs: `docs/paper/review_loop/round_1.md`, `round_2.md`, `docs/paper/NOVELTY.md`,
web search of this session. No experiment was run and no other file was edited.

Verification legend for citations: **[X]** = DOI and bibliographic data confirmed on Crossref this session;
**[A]** = abstract or full text read this session (arXiv HTML, IOP page, or publisher abstract through a
search snippet); **[S]** = secondary source only (conference abstract, repository record). Dataset sizes and
validation details are stated only where read; "not read" means the full text was not accessed.

## 0. Bottom line

1. Every surveyed Nuclear Fusion (NF) paper of this kind was carried by one of three things: a **control-room
   deployment** (RTCAKENN, the q-profile and Grad-Shafranov networks), a **physics result the diagnostic
   could not give before** (Diag2Diag's island flattening, BES-ELMnet's pre-ELM turbulence), or **breadth**
   (thousands of discharges over several campaigns, whole profiles or many signals). None was carried by a
   statistical margin over an interpolator on one channel.
2. The present paper has none of the three. Its strength (registered protocol, shot-clustered intervals,
   causal and offline comparators) exceeds the statistical practice of every surveyed paper, but that is
   not what NF's criterion ("significant advances to the field of controlled thermonuclear fusion") rewards.
3. The closest accepted template for a single-device temporal-densification paper is in **PPCF**, not NF:
   Imrisek et al. 2026 (COMPASS) pairs the network with a physics application (ELM stability analysis on
   the densified profiles). A physics application of that kind is the minimum addition for PPCF and the
   first of three additions needed for NF.
4. A 10 ms estimate cannot resolve an ELM crash, and the paper must not promise that. The reachable
   payoffs at 10 ms are: (a) a T_i value for a real-time consumer on rows where CES delivers nothing,
   (b) event-conditional tracking of T_i through L-H and H-L transitions, sawtooth ion heat pulses and
   RMP-suppression onsets, where future-reading interpolation smears the step, (c) more usable phases in
   NBI-modulation transport analysis. All three need signals the repository does not hold (D-alpha, NBI
   power, EFIT mapping, more chords).

## 1. Survey: what carried each paper

### 1.1 Nuclear Fusion

| # | Paper | Task | Data scale | Validation | What carried it |
|---|---|---|---|---|---|
| 1 | Shousha, Seo, Erickson, Xing, Kim, Abbate, Kolemen, "Machine learning-based real-time kinetic profile reconstruction in DIII-D", NF 64, 026006 (2024), doi:10.1088/1741-4326/ad142f **[A, full page]** | NN surrogate of offline CAKE: seven profiles (p, 1/q, j, T_e, n_e, carbon T_i, rotation) from real-time signals | 696 discharges, 18,959 time slices; L-mode, H-mode, ITB, hybrid | 1,334 held-out slices, mean Pearson R 0.972; ensemble of 10; input dropout for missing TS/CER; example discharges | **Deployment**: runs in the DIII-D PCS at about 7.7 ms inside a 20 ms cycle; positioned as input to profile control |
| 2 | Abbate, Conlin, Kolemen, "Data-driven profile prediction for DIII-D", NF 61, 046027 (2021), doi:10.1088/1741-4326/abe08d **[A, abstract]** | Forecast profiles about one confinement time ahead from actuators and state history | DIII-D 2013-2018 campaigns | Held-out discharges (details not read) | **Consumer**: fast enough for model-predictive control; a control experiment followed (APS DPP 2021 abstract) |
| 3 | Wan et al., "Experiment data-driven modeling of tokamak discharge in EAST", NF 61, 066015 (2021), doi:10.1088/1741-4326/abf419 **[A, abstract]** | Reproduce diagnostic signals (n_e, W_mhd, V_loop) from actuator signals | EAST 2016-2018 campaigns | Similarity 89 % (n_e), 97 % (W_mhd) on test discharges | **Breadth and use**: discharge-design tool over three campaigns |
| 4 | Wang, Wan, Lu, Yu et al., "Time series extrinsic regression for reconstructing missing electron temperature in tokamak", NF 65, 076008 (2025), doi:10.1088/1741-4326/addb5f **[X; abstract A]** | Reconstruct missing T_e from other signals | Size not read | "over 95.9 %" confidence that reconstructed T_e lies within two standard deviations over the discharge | **Operational use**: data integrity when a sensor fails or the diagnostic is absent |
| 5 | Lin, Zhang, Wang, Bae, Fu et al., "Prediction of plasma rotation velocity and ion temperature profiles in EAST Tokamak using artificial neural network models", NF 64, 106061 (2024), doi:10.1088/1741-4326/ad73e8 **[X; abstract A]** | NN reads x-ray crystal spectrometer data into T_i and rotation **profiles** | Size not read | Agreement with conventional analysis; input-range and error study | **Speed toward real time**: more than ten times faster than the standard analysis; whole profile |
| 6 | Joung, Smith, McKee, Yan et al., "Tokamak edge localized mode onset prediction with deep neural network and pedestal turbulence", NF 64, 066038 (2024), doi:10.1088/1741-4326/ad43fb **[X; abstract A]** | Forecast ELM onset from BES pedestal turbulence | Size not read | Onset probability rises through the inter-ELM phase; forecasts the first ELM after the L-H transition; later zero-shot transfer DIII-D to KSTAR (APS DPP 2025, IAEA FEC 2025 **[S]**) | **Physics and control**: a turbulence precursor of the ELM, aimed at proactive H-mode control |
| 7 | Wei, Sun, Tang, Lin et al., "Reconstruction of tokamak plasma safety factor profile using deep learning", NF 63, 086020 (2023), doi:10.1088/1741-4326/acdf00 **[X; S]** | q profile without MSE | DIII-D | Not read | **Deployment-ready**: sub-millisecond inference, replaces a diagnostic that is often unavailable |
| 8 | Joung et al., "Deep neural network Grad-Shafranov solver constrained with measured magnetic signals", NF 60, 016034 (2020), doi:10.1088/1741-4326/ab555f **[X for DOI and title; volume/number from memory]** | KSTAR equilibrium from magnetics | KSTAR | Not read this session | **Real-time equilibrium** |
| 9 | Kim et al., "Kinetic profile inference with outlier detection using support vector machine regression and Gaussian process regression", NF 64, 106052 (2024), doi:10.1088/1741-4326/ad7304 **[A, abstract]** | KSTAR CES T_i **profile** fitting robust to outliers | KSTAR CES | Profiles with and without the outlier; MAP against MCMC hyperparameters | **A named routine need**: automatic kinetic-profile analysis on KSTAR. This is a KSTAR CES paper in NF with no deployment: it is a profile, and it solves a problem the analysts name |

### 1.2 Contrast: PPCF, Nature Communications, Communications Physics, FED

| # | Paper | Task | Data scale | Validation | What carried it |
|---|---|---|---|---|---|
| 10 | Jalalvand, Kim, Seo, Hu, Curie, Steiner, Nelson, Na, Kolemen, "Multimodal super-resolution: discovering hidden physics and its application to fusion plasmas" (Diag2Diag), Nat. Commun. 16, 8506 (2025), doi:10.1038/s41467-025-63492-1; arXiv:2405.05908 **[X; A, arXiv v4 HTML]** | Thomson n_e, T_e at fluctuation-diagnostic rate from CER, interferometer, ECE, MSE, magnetics | 4,000 DIII-D discharges, 2017-2022; 135,233 / 22,084 / 18,721 train/val/test samples | Held-out samples (R^2 0.92); composite of more than 20 repeated ELM cycles; bunch-mode TS; Bayesian-NN uncertainty against measurement error; qualitative comparison with RMP simulation. **No quantitative interpolation baseline.** Memoryless MLP (with smoothed derivatives over about 4 ms) | **Physics**: simultaneous T_e and n_e flattening at pedestal top and foot under RMP, read as evidence for island-based ELM suppression |
| 11 | Imrisek et al., "Temporal super-resolution of kinetic plasma profiles at COMPASS using neural networks", PPCF (2026), doi:10.1088/1361-6587/ae72c6 **[X for DOI and title; content S]**. Volume and article number (68, 065049) are taken from `NOVELTY.md`, not re-confirmed | TS T_e, n_e **profiles** between laser pulses (about 8 ms) from fast signals | One device (COMPASS) | Reconstruction error about 10 % (conference abstracts of the same group, Odlozilik, Imrisek, Seidl, Sos) | **Physics application on one device**: densified profiles fed to peeling-ballooning stability analysis of the ELM cycle (MISHKA). The nearest accepted template to our paper |
| 12 | Yang et al., "FusionMAE, a self-supervised pretrained model to optimize and simplify diagnostic and control of fusion plasma", Commun. Phys. 9, 232 (2026), doi:10.1038/s42005-026-02626-3; arXiv:2509.12945 **[X; A, abstract]** | Masked autoencoder over 88 signals; reconstruct masked channels | 2,445 HL-3 discharges, four campaigns 2022-2025 | Pearson 96.7 % on masked signals; downstream disruption prediction and equilibrium tasks degrade slowly under masking | **Breadth and downstream control tasks** |
| 13 | Kim et al., "Highest fusion performance without harmful edge energy bursts in tokamak", Nat. Commun. 15 (2024), doi:10.1038/s41467-024-48415-w **[A, abstract]** | ML-optimized 3D field with real-time adaptive control | DIII-D and KSTAR | Experiments on two devices | **Two-device experimental result** |
| 14 | Lee, Ko, Lee, Kim et al., "Neural network-based analysis of charge exchange spectra in KSTAR" (NN-CES), Fusion Eng. Des. 222, 115518 (2026), doi:10.1016/j.fusengdes.2025.115518 **[X; A, abstract]** | T_i and V_rot from raw CES spectra without beam modulation | KSTAR; L-mode, H-mode, locked-mode, RMP conditions | Agreement with beam-modulation analysis | **A stated KSTAR problem**: beam modulation uses the heating beam and may disturb the plasma. Venue was FED, not NF |
| 15 | Pavone, Merlo, Kwak, Svensson, "Machine learning and Bayesian inference in nuclear fusion research: an overview", PPCF 65, 053001 (2023), doi:10.1088/1361-6587/acc60f **[X]** | Review | - | - | Context for the interpretability expectation in this community |

### 1.3 Patterns

- **Spatial scope.** Seven of the nine NF papers produce a profile, an equilibrium or a multi-signal
  state (rows 1, 2, 3, 5, 7, 8, 9). The other two (rows 4, 6) are an operational recovery tool and an
  event forecaster with a control consumer. None reports a single measurement chord as its result.
- **Data scope.** Where sizes were read: 696 discharges (RTCAKENN), 4,000 over six years (Diag2Diag),
  2,445 over four campaigns (FusionMAE), three campaigns (Wan), six campaigns (Abbate). Our 641 discharges
  are comparable to RTCAKENN in count; the weakness is one campaign, not the count.
- **Validation.** Held-out discharges plus one of: deployment in the PCS, a physics case study, a
  downstream task. Multi-campaign training is the norm; a separate held-out-campaign split is not
  reported in the texts read. Quantitative interpolation baselines are absent, including in Diag2Diag;
  our comparator ladder goes beyond the field's practice.
- **Mechanism.** No surveyed paper isolates by retraining which input carries the information. The output
  was instead checked against physics (island flattening, ELM-cycle recovery, stability boundaries). A
  physics check is accepted in place of an attribution; a margin alone is not.
- **Authorship.** The papers using a device's data carry authors from that device's team or its
  acknowledgement and data statement.

## 2. The bar, ranked

**Necessary (without all of these, an NF submission is most likely rejected at the editor or in the first round):**

1. A named consumer with a demonstration: either a physics analysis that the gap-filled series enables and
   the raw series does not, or a real-time replay showing the estimate feeding a controller or a profile
   reconstruction on rows where CES delivers nothing.
2. Spatial coverage beyond one chord: the same estimator on several CES channels (ideally the profile),
   with r/a mapped by EFIT.
3. A second campaign scored once under the frozen protocol.
4. A deployable pipeline: fast channels standardized causally, and the fast-channel share isolated by a
   retrained history-only backbone.
5. KSTAR team involvement: co-authors or a formal data statement, and per-exposure CES fit metadata.

**Strongly expected (each missing one is a major-revision item):**

6. Event-resolved evaluation (ELM, L-H, H-L, sawtooth, NBI step, RMP onset) using D-alpha, NBI power and
   coil currents, with one worked discharge figure.
7. The V_rot claim on registered footing: a fit-quality rule from instrument metadata, fixed before
   scoring, or V_rot reduced to the coupling bound.
8. Calibrated uncertainty on the estimate (coverage of a predictive interval), which a consumer needs.
9. A memoryless Diag2Diag-style baseline on the same inputs.

**Nice to have:** a second device or zero-shot transfer; validation against a faster instrument (the KSTAR
ultrafast CES when it yields data); a PCS implementation with measured latency; an open benchmark split.

## 3. Physics payoffs reachable with a 10 ms T_i / V_rot estimate on KSTAR

A limit first. The estimator fills missing samples on the 10 ms CES grid; it does not produce sub-10 ms
resolution. ELM-resolved pedestal work used 65 microseconds on ASDEX Upgrade (Cavedon et al., "Pedestal and
E_r profile evolution during an edge localized mode cycle at ASDEX Upgrade", PPCF 59, 105007 (2017),
doi:10.1088/1361-6587/aa7ad0 **[X; A, abstract]**; Cavedon et al., "A fast edge charge exchange
recombination spectroscopy system at the ASDEX Upgrade tokamak", Rev. Sci. Instrum. (2017),
doi:10.1063/1.4979801 **[X]**). A crash-resolved claim is out of reach unless the model is re-posed as a
true super-resolution model and validated the way Diag2Diag was (composite of many repeated ELM cycles),
and each CES exposure still integrates over its exposure time.

| Payoff | What the literature says | Reachable with our data? | What is needed |
|---|---|---|---|
| **P1. T_i for a real-time consumer when CES is missing** | RTCAKENN uses real-time CER and is trained with dropout so that it survives missing CER [1]. JT-60U ran real-time T_i-gradient control from a 10 ms CXRS (search snippet only: Plasma Fusion Res. 2, S1049 (2007); Kawamata et al., Fusion Eng. Des. 2009 **[S]**). NN-CES states "potential for real-time plasma control" on KSTAR [14]. No published KSTAR real-time CES feedback loop was found (two searches, negative); a KSTAR in-shot NBI control aimed at stabilizing T_i in FIRE mode appears in an IAEA FEC 2025 poster **[S]** | Partly. The estimator, the latency count and the missing-row statistics exist. The deployable variant (causal standardization) is not scored under the confirmed protocol | Round 1 follow-up 1 (GPU, about 1 h); a replay test that streams a discharge sample by sample; a statement from the KSTAR control group on whether a real-time T_i is wanted |
| **P2. Event-conditional tracking (L-H, H-L, sawtooth, RMP onset)** | JET: the H-L transition collapses edge T_i and rotation "within less than a sampling time of 12 ms", and the same cadence reveals ion heat pulses after sawteeth (JET CXRS paper, search snippet **[S]**). DIII-D measured the ion heat pulse after a sawtooth crash with 2 ms CER and found it not ballistic (PPPL-3443 **[S]**). KSTAR: T_i and V_rot pedestals form after the L-H transition and their tops drop under RMP ELM suppression (Ko et al., "Ion temperature and toroidal velocity edge transport barriers in KSTAR", NF 55, 083013 (2015), doi:10.1088/0029-5515/55/8/083013 **[X; A, abstract]**; Ko et al., Rev. Sci. Instrum. 85, 11E413 (2014), doi:10.1063/1.4890402 **[X]**). DIII-D L-H edge profile evolution: Haskey et al., PPCF 60, 105001 (2018), doi:10.1088/1361-6587/aad702 **[X]** | Not yet. The files hold no D-alpha, NBI power or coil current, so events cannot be labelled. The existing hint is statistical: the margin concentrates on the top-10 % rows by ECEI step | Data request: D-alpha, NBI power per source, RMP coil currents, a core ECE channel for sawteeth, EFIT. Then score every arm on rows within a fixed window of each event class, with shot-clustered intervals, registered before scoring |
| **P3. NBI-modulation transport analysis (chi_i, chi_phi, pinch)** | KSTAR: Yang, Na, Na, Park et al., "Perturbative studies of toroidal momentum transport in KSTAR H-mode and the effect of ion temperature perturbation", NF 58, 066008 (2018), doi:10.1088/1741-4326/aab90e **[X; A, abstract]** fits amplitude and phase of the modulated rotation and reports that the T_i perturbation from NBI changes the inferred coefficients. JET ion heat modulation was limited by a 50 ms T_i measurement to about 4 Hz (probably Ryter et al., "Simultaneous analysis of ion and electron heat transport by power modulation in JET", NF 51, 113016 (2011), doi:10.1088/0029-5515/51/11/113016 **[X]**; the 50 ms figure is from a search snippet **[S]**). Ion stiffness: Mantica et al., PRL 102, 175002 (2009), doi:10.1103/PhysRevLett.102.175002 **[X]** | Weak for V_rot (history only, coupling bound \|r\| <= 0.044, so the estimate adds nothing a smoother does not). Plausible for T_i: missing samples inside a modulation cycle cost phase accuracy, and a gap-filled series restores them | NBI waveforms for the 2022 discharges; identify modulation discharges; compare amplitude and phase of the modulated T_i from the raw series, PCHIP, causal GP and the estimator, with cycles that have no missing samples as reference. Multi-chord data turns this into a chi_i profile |
| **P4. Sub-10 ms T_i dynamics** | Recognized as a need at the turbulence scale: the KSTAR ultrafast CES is designed for 10 microseconds at two radial positions (Jang, Ko, Li, von Hellermann, Shi et al., "Development of an ultrafast charge exchange spectroscopy system on the KSTAR tokamak", Rev. Sci. Instrum. 92, 053525 (2021), doi:10.1063/5.0043824 **[X; A, abstract]**); the DIII-D counterpart is Truong et al., Rev. Sci. Instrum. 92, 053513 (2021), doi:10.1063/5.0043095 **[X]**. Conventional KSTAR CES is stated at 10 ms (100 Hz) and 5 mm (Ko 2015; IAEA FEC 2012 poster **[S]**) | No. Outside what the present model does, and no fast T_i ground truth exists in the files | Future work only: the 36 raw microsecond files (BES, Mirnov) could drive a super-resolution variant, validated by ELM-synchronized composites once D-alpha is available |

**Is fast T_i on KSTAR a recognized need?** Yes in two forms, neither of which is "fill 10 ms gaps":
(i) microsecond T_i and rotation fluctuations for turbulence (ultrafast CES), and (ii) continuous,
modulation-free CES with real-time potential (NN-CES, whose motivation is that KSTAR modulates the heating
beam). No paper was found that names missing samples on the 10 ms CES grid as a problem. The positioning
this supports is P1 (continuity of the T_i signal for a consumer) with P2 as the physics check, and no
resolution claim.

## 4. Nuclear Fusion scope and formats

Source: IOP "About Nuclear Fusion" page (https://iopscience.iop.org/journal/0029-5515/page/about), read
this session, and the IAEA journal page (https://www.iaea.org/publications/nuclear-fusion).

- Scope sentence: "Nuclear Fusion publishes articles making significant advances to the field of
  controlled thermonuclear fusion." Topics: plasma production, heating and confinement; plasma physical
  properties; experimental and theoretical methods; fusion reactor physics; reactor concepts; fusion
  technologies. Editorial office at the IAEA in Vienna; published with IOP Publishing.
- **Papers**: original research, no stated length limit; conclusions must be a significant advance.
- **Letters**: short communications of new, possibly preliminary results, "aiming for four printed pages
  including figures"; two reviewers, usually one revision; reviewers may reject as a letter with the option
  to resubmit as a full paper.
- Special Topic articles, Comments (with replies), and Reviews (commissioned only).
- Single-anonymous review. No ML-specific reporting guideline for NF was found (one search, negative).
- Implication: "experimental and theoretical methods" covers a diagnostic-analysis method, so the paper is
  in scope; the test is "significant advance". A Letter fits one sharp result (for example P2 on several
  chords), not the present eight-page statistical case.

## 5. Ranked gap list

Effort is working time for one person once data is in hand. "Request" items add the turnaround of the
KSTAR team, which is outside our control. GPU times follow the budgets already recorded in `round_1.md`.

| Rank | Gap | Why it blocks NF | Closing experiment or data request | Effort | Class |
|---|---|---|---|---|---|
| 1 | No consumer, no physics payoff | Every surveyed paper has one (Section 1.3) | **Request**: D-alpha, NBI power per source, RMP coil currents, core ECE, EFIT for the 641 discharges. **Experiment**: event-conditional scoring P2 (L-H, H-L, sawtooth, RMP onset, NBI step), registered before scoring, plus one worked-discharge figure. Addition or fallback: P3 on modulation discharges | Request: one message. Analysis: 1-2 weeks on existing predictions, CPU, no retraining | Necessary |
| 2 | One chord (R = 2.198 m) | No surveyed NF paper reports a single chord; the profile is the unit of the field | **Request**: all CES channels (T_i, V_rot, fit error) for the same discharges, the matching BES/ECEI channel geometry, EFIT r/a. **Experiment**: train the frozen architecture per chord, or one model with chord position as input; report skill against r/a | 1-2 weeks including GPU (about 1 h per chord for 4 splits) | Necessary |
| 3 | One campaign (2022) | Campaign shift is the standard doubt; the present campaign split depends on offline per-discharge standardization | **Request**: a 2023 or 2024 subset (about 100-200 discharges, same columns). **Experiment**: one frozen-protocol external test, registered publicly before the files are opened | 2-3 days once data arrives | Necessary |
| 4 | Not deployable: fast channels use whole-discharge statistics | A real-time claim is void without it | Round 1 follow-up 1: seq_v2 with `CES_PER_SHOT_NORM=0`, then an expanding or EWMA causal standardization, 4 splits x 2 populations, paired against the present backbone and scored against the causal GP | GPU 1-2 h plus about 1 day of code and tests | Necessary; no new data |
| 5 | Fast-channel share of the T_i margin not isolated | Without it the "what fast diagnostics add" title is unsupported | Round 1 follow-up 2: history-only seq_v2 retrain (7 slow channels on the T_i branch), paired with the full model. Cheaper precursor from round 2: the `a_c = 0` surrogate on frozen rows | GPU about 45 min; precursor CPU under 25 min | Necessary; no new data |
| 6 | No KSTAR authorship, data statement, or CES fit metadata | Required in practice; also the only way to replace the review-time 1,000 km/s rule and the 3 keV cut with instrument criteria | **Request** to the CES group: per-exposure chi-square, photon counts, fit status, cause of held V_rot values, whether a missing row had an exposure; co-authorship or acknowledgement terms | 3-5 days to re-score under the instrument rule; GPU about 1 h if history inputs are cleaned and V_rot retrained | Necessary |
| 7 | V_rot margin over PCHIP rests on fit-failure rows; rule set during review | Integrity item; a referee from a CES group will know these rows | With item 6 metadata: register the fit-quality rule, clean the history inputs, retrain, re-score. Without it: V_rot as the coupling bound plus the persistence margin only, as round 2 directs | 2-3 days after item 6 | Expected |
| 8 | No mechanism | Co-movement is a correlation, and NF asks for the physics | Follows from items 1 and 2: per-event response of the T_i branch to BES and ECEI, radial dependence of the coupling, and absolute T_e and n_e (ECE, Thomson) instead of per-discharge z-scores to test collisional exchange | 1-2 weeks after items 1-2 | Expected |
| 9 | No predictive uncertainty | A controller or analyst needs an error bar; Diag2Diag reports one | Coverage of a Gaussian or quantile head on the frozen backbone, per event class; the earlier Student-t and mixture heads can be re-scored for coverage | 1-2 days | Expected |
| 10 | No memoryless multimodal baseline | The closest prior work is memoryless; a referee will ask | Round 1 follow-up 7: MLP and gradient-boosted trees on the same 22 inputs, same splits | Under 1 day | Expected |
| 11 | No artificial-gap curve or trace figure | Every surveyed paper shows traces | Round 1 follow-up 4: masking curve at k = 1, 2, 5, 10, 20 steps for all arms and one discharge trace | 1 day, CPU | Expected |
| 12 | No real-time demonstration | Carried RTCAKENN; optional if item 1 delivers a physics result | Sample-by-sample replay with measured latency; PCS integration only with the KSTAR control group | Replay 2-3 days; PCS integration months | Nice to have |
| 13 | One device | Transfer is rare in the survey (BES-ELMnet zero-shot is the exception) | Not reachable now | - | Nice to have |

### Reading of the list

- Items 4 and 5 need only the GPU and already head the internal queue. They are necessary for any archival
  venue; alone they do not make the paper an NF paper.
- Items 1, 2, 3 and 6 are one data request to the KSTAR CES and operations groups. Send it as a single
  list: (a) all CES channels with fit metadata, (b) D-alpha, NBI power, RMP currents, core ECE, EFIT,
  (c) a 2023 or 2024 subset, (d) authorship and data-statement terms.
- Venue by what gets done:
  - Items 1-6 done, with a P2 result whose intervals exclude zero at several radii: NF becomes a
    defensible target. A four-page Letter on the event-resolved multi-chord result is a possible format.
  - Items 4-5 plus item 1 on the single chord: PPCF is the matching venue (Imrisek et al. as template).
  - Items 4-5 only: FED remains the realistic venue, as the review loop concluded.
- The paper must not claim ELM-crash or sub-10 ms resolution (Section 3, P4).

## 6. Limits of this survey

- Dataset sizes and validation details are confirmed for RTCAKENN, Diag2Diag and FusionMAE only. For the
  other papers the tables give what the abstract states. The full texts of Wang 2025, Lin 2024, Joung 2024,
  Wei 2023, Imrisek 2026 and NN-CES were not read.
- The JT-60U real-time control, JET 12 ms, JET 50 ms and DIII-D 2 ms sawtooth statements come from search
  snippets; read the originals before citing them in the paper.
- The KSTAR CES channel count and radial coverage in 2022 were not confirmed from a primary source; ask
  the CES group.
- No NF editorial guidance on ML submissions was found. The bar in Section 2 is inferred from published
  papers only, so it cannot show what was rejected.
- Web requests used generic topic queries and paper titles only.
