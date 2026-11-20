# Novelty critique: what in this project has prior work not already shown? (2026-11-20)

Role: hostile but fair PPCF referee and handling editor, asked one question. The author's verdict under test: "at this level there is nothing special compared with previous work; effectively unpublishable in PPCF; no novelty."

**Read.** `PROJECT_KNOWLEDGE.md` sections 1, 2, 4.6-4.7, 4.12, 4.19-4.21b, 4.48-4.54; `PAPER_DESIGN.md` section 1 (with the block "재지정(2026-11-20)"); `docs/paper/b74_exposure.md` (statements and part A-B); `docs/paper/review_loop/critique_20261120_cxrs.md` and `critique_20261120_pedestal.md` in full; `docs/paper/validation_methods.md` in full; `docs/paper/literature_map.md` by search. I did not reread `b73_targets.md` and `b71_scenarios.md` beyond what `PROJECT_KNOWLEDGE.md` 4.49 and 4.54 quote from them. I did not repeat the two earlier critiques. I used their verified readings and say so where I do.

**Searched.** I used OpenAlex (title and abstract search, `fulltext.search`, and `cites:` on Diag2Diag) through `critique_20261120_novelty_checks/oa.py`. Every call is logged in `log.jsonl` in that folder. I also used the WebSearch tool. No personal or project data went into any request. There was no mailto and the user agent was generic.

**Read levels.** These marks appear in section 8 and in the text.
- [A]: I read the abstract.
- [T]: I saw only the bibliographic record (the abstract was missing).
- [S]: I saw only a search snippet or a search-engine summary.
- [F-crit]: another critic read the full text today, and I did not reread it.
- [V-map]: taken from the authors' `literature_map.md` and not reread.
- [P]: a project file.
- [J]: my judgement.

**Checks.** N1-N3 are unregistered, after-the-fact, read-only checks. They are single-threaded numpy runs on `data/.b73/` and `data/mast/`. Scripts and printed outputs are in `docs/paper/review_loop/critique_20261120_novelty_checks/`. Nothing was trained and no tracked file was changed.

---

## 0. Verdict in one paragraph

The author is right for PPCF. Three of the five candidates are already known or follow from known results:
- C1 is a smoothing result whose size is set by a non-robust reference.
- C3 is the edge-CXRS ELM contamination that the KSTAR CES group itself documents.
- C4 is good practice assembled from published parts.

C5 consists of sensitivity checks and a small data-quality utility.

The one item I could not find in the fusion literature is C2. It is the fusion-specific demonstration that a cross-diagnostic gain scored against the same-time label is mostly co-variation with that label's frame-specific error. Its statistical principle is textbook (conditionally unbiased verification proxies, independent-noise self-supervision, error cross-correlation). It rests on one chord and has no fit diagnostics behind it. That supports a methods note, a thesis chapter, or a paper after a second demonstration. It does not support a PPCF physics paper. The question PPCF publishes, what the pedestal-top ion temperature does, is untouched by every candidate.

**Added after the session lead's second request (section 6b).** Judged as audits of an increment, C1-C5 do not make a paper. Judged as possibilities that the files could demonstrate, one candidate has the shape of the accepted papers in this area (RTCAKENN, NN-CES, Diag2Diag): a capability, proposed and then shown to work on withheld data. That candidate is bridging the beam-modulation latency of the edge CES with the fast diagnostics of the same radius. Most of its evidence already exists in today's B.75 output, which I read after writing sections 1-6. B.75 is checked by the lead in `docs/paper/b75_lead_checks.md` and is not yet recorded in `PROJECT_KNOWLEDGE.md`. It is not among C1-C5, so the verdict table above stands. Section 6b ranks six possibilities and names the one I would build the paper around.

---

## 1. C1: a learned causal history-only estimator has lower MSE than a causal GP (13-25 %, four designs)

The facts as given in [P] `PROJECT_KNOWLEDGE.md` 4.54:
- On the level target L5, T0 against the causal GP is +0.166 (V1), +0.150 (V2), +0.134 (V3) and +0.254 (V4).
- In V4 the RMSE values are: causal GP 166.6 eV, T0 143.9 eV, and K0 139.1 eV. K0 is the robust five-parameter state-space filter without fast channels.

**Why-chain.**

- **Why is this not new?** Causal estimation of a noisy plasma measurement from its own past is routine.
  - Model-based observers merge noisy Thomson data with a transport model in real time ([A] Morosohk et al., CTPP 2023). The RAPTOR EKF estimates T_e, n_e and j with disturbance estimation ([A] NF 2025, add16e).
  - Offline, profile tools fit in time to remove "jitter that has historically been smoothed ad hoc" ([A] Logan et al., OMFIT, FST 2018).
  - KSTAR profile fitting already replaces the Gaussian likelihood with an outlier-robust good-and-bad mixture ([A] Kim et al., NF 2026, ae93dd) or an SVM outlier filter ([A] Kim et al., NF 2024, ad7304).
  - That a model fitted across many series has lower error than per-series local models is a stated principle of forecasting ([T] Montero-Manso & Hyndman, IJF 2021). The same holds for learned filters against model-mismatched classical filters ([A] KalmanNet, IEEE TSP 2022).
  - **Why does the principle apply here with such force?** The reference is weak in two ways. The causal GP picks its hyperparameters per sample from 16 values under a Gaussian likelihood. The series has heavy-tailed one-frame departures: 5.9 % of scored rows rise more than 35 % and 5.0 % fall below 0.74 ([F-crit] CXRS critique, C3/C5). A population-fitted model learns the shrinkage and robustness that the per-sample GP cannot.
    - N1 (unregistered) rules out the most naive form of this. A causal median of the last five measured values has a higher MSE than the GP (skill -0.125 [-0.157, -0.090], L5, V1, r2-r4 identical). So the gain is not plain robustness.
    - N2 (unregistered) shows what the gain is. A four-coefficient linear blend (intercept, GP, hold, last-five median) was fitted on half of the discharges and scored on the other half. It gives +0.086 [+0.072, +0.101] over the GP, against +0.168 [+0.148, +0.187] for T0. That is half of C1.
    - **Why does that matter?** The size of C1 measures how mis-specified the reference is, not new information about T_i. A referee asks for the strongest causal reference, and the project's own registered table supplies it. Forward in time (V4), K0 has a lower RMSE than T0 (139.1 against 143.9 eV, [P] 4.54). The "learned" part of C1 is not established against a robust filter in the one design that mimics use.
      - **Why has nobody published "a learned smoother of a CXRS chord has lower error than a GP"?** Because there is no use case that needs it, not because nobody thought of it. The field smooths offline (OMFIT), filters ELMs, or fits robust likelihoods. In real time the history this estimator reads does not exist: the newest deliverable value is 0.65 s old at the median, and beyond 100 ms the causal GP has the lower error ([P] 4.49 and 4.52; [F-crit] both critiques).
        - **What would make it new?** All of the following together:
          - a causal estimator that keeps a margin over a robust reference (a Student-t or mixture-likelihood causal GP and K0);
          - the margin holds forward in time and under the real availability schedule (B.75 is registered for that);
          - the estimator is used for a purpose that needs it.

**ROOT.** C1 is the distance between a population-fitted smoother and a per-sample Gaussian-likelihood GP on a heavy-tailed series. Half of it is a four-coefficient linear correction. Forward in time, a five-parameter robust filter has an equal or lower error than the learned estimator. It is a property of the reference.

**Verdict:** obvious from known results.
- Closest prior work: Montero-Manso & Hyndman 2021 (globality; doi 10.1016/j.ijforecast.2021.03.004, [T]) and Kim et al. 2026 (robust likelihood for KSTAR profiles; doi 10.1088/1741-4326/ae93dd, [A]).
- How it differs: neither targets the per-frame time series of one CES chord, and that difference is not a contribution.

**Referee, one sentence:** "The gain over a Gaussian-likelihood GP on a heavy-tailed series is a statement about the baseline, and the authors' own robust five-parameter filter has a lower error than the learned estimator in the chronological split."

**Would settle it:**
- [existing data, CPU] A causal GP with a Student-t or good-and-bad likelihood and globally fitted hyperparameters, together with K0 as references on V1-V4/L5, with cluster intervals.
- [existing data, CPU] B.75 (registered; availability schedule).

---

## 2. C2: the "cross-diagnostic gain" is mostly co-variation with the frame's own reading

The facts in [P] 4.54:
- An estimator trained on F has, against L5, a fast-channel increment of +0.013 [-0.025, +0.045] (two-stage) and +0.016 [-0.036, +0.054] (network). On F it has +0.087.
- Refitted to L5, the increment is:
  - +0.075 [+0.039, +0.106] for V1 and +0.057 [+0.034, +0.079] for V2;
  - +0.031 [-0.006, +0.060] for V3;
  - +0.084 for V4 trees and -0.007 for V4 filters.
- The fitted correction scale λ is 0.864 for F and 0.529 for L5.

**Why-chain.**

- **Why is this not new as a principle?** Several fields state it already.
  - Ranking forecasts by MSE against an error-laden proxy is robust only if the proxy is conditionally unbiased given the forecasters' information ([S] Patton, J. Econometrics 2011; abstract seen through search summaries only).
  - Proper scores against observations with error "favour good forecasts of observations rather than of truth" ([A] Ferro, QJRMS 2017).
  - Self-supervised evaluation of a denoiser from noisy data alone requires the noise to be independent between the parts used as input and as target ([A] Noise2Self 2019; [A] Noise2Noise 2018).
  - Error cross-correlation between collocated products biases validation, and estimating it is an open problem there ([A] Gruber et al., JGR 2016).
  - Social science calls the same thing common-method variance ([A] Podsakoff et al., 2003).
  - C2 is a measured violation of that condition. The BES sample (an input) and the frame-specific part of the CES fit (the label error) share a nuisance: the light of the exposure.
  - **Why does that matter for fusion?** Every cross-diagnostic reconstruction paper I could check scores against the same-time label of the target diagnostic and treats that label as ground truth:
    - Diag2Diag: L1 against Thomson ([A]; [P] survey row 6);
    - RTCAKENN: Pearson R against CAKE fits ([P] survey row 11; full text read by the lead);
    - COMPASS temporal super-resolution, EAST T_e and Z_eff, FusionMAE ([P] survey rows 7-10);
    - HL-3 real-time CXRS and NN-CES (R² against the conventional fit; [A]).

    None reports a target that removes the label's frame-specific error. Here that choice moves the as-built increment from +0.087 to +0.013.
    - **Why has nobody tested it in fusion?** Each diagnostic in a cross-diagnostic model is assumed to have its own independent error. That is a reasonable default when input and target share no physical path. It fails exactly where they do: same beam, same edge light, same exposure window. BES and CES at KSTAR share the heating beam and the edge emission. In the Diag2Diag setting (Thomson target, ECE/CER/magnetics inputs) the shared nuisance is weaker. So whether C2 generalizes is an open, testable question, not a known answer. [J]
      - **Why can it not be closed with these files?** The nuisance cannot be identified. There is no fit residual, amplitude or error bar per frame, and no spectra. The CES level target (L5) is the same automatic analysis, so a bias lasting more than 50 ms passes it ([P] 4.54 limits). The refit residual (+0.06 to +0.08) is therefore "level information or slow artefact", undecided.
        - **What would make it new?** All of the following:
          - a second, independent demonstration where the label has fit diagnostics;
          - the same-time gain shrinks against a following-level reference in proportion to a measured shared nuisance (for example the reported fit error), and does not shrink where no nuisance is shared;
          - a test that detects the condition without a later target (a gain conditional on the label's reported error).

**ROOT.** C2 is a correct, transferable observation: same-time scoring of a cross-diagnostic estimator is biased when input and label share a nuisance. Its principle is textbook in forecast verification and denoising. What is missing from the retrieved fusion literature is only the demonstration. Here it rests on one chord, with a nuisance that cannot be named because the label's fit diagnostics are absent.

**Verdict:** partly new. The fusion-specific demonstration and its size are new; the principle is not.
- Closest prior: Patton 2011 (doi 10.1016/j.jeconom.2010.03.034, [S]), Ferro 2017 (doi 10.1002/qj.3115, [A]) and Noise2Self (doi 10.48550/arXiv.1901.11365, [A]) for the principle.
- Closest fusion work it differs from: Diag2Diag (doi 10.1038/s41467-025-63492-1, [A]), which scores against same-time Thomson and does not test this.

**Referee, one sentence:** "This is the known requirement that the verifying observation be conditionally unbiased given the forecaster's information; a useful caution for the fusion-ML community, but a single-chord case without the label's error bars is a note, not a PPCF paper."

**Would settle it:**
- [existing data, CPU] The same F-versus-following-level decomposition on public MAST CXRS. The inputs are history + mid-plane D-alpha (+ Thomson). Stratify by the reported T_i error, which MAST has and KSTAR lacks.
- [needs data that do not exist here] `ERR_BAR`, the fitted amplitude and the residual of the KSTAR chord.

---

## 3. C3: one-frame upward departures coincide with BES light bursts inside the exposure

The facts in [P] 4.53-4.54:
- J6: 9.48 [4.65, 18.68] times over-represented. J2: 1.105 against 1.006. J3: following level unchanged. J4: rotation unchanged.
- Log slope of width on BES: 1.285 [1.147, 1.422]. Rotation slope: 0.024 [-0.009, 0.058].
- 11 discharges. There are no spectra.

**Why-chain.**

- **Why is this not new?** Each part has a source.
  - The KSTAR CES group writes that at R ≳ 2.2 m the active signal is reduced, that spectra are "more easily distorted by transient events such as ELMs and by discrepancies between beam-on and beam-off phases", and that these "degrade the fitted parameters" ([F-crit] Lee et al., FED 222:115518, section 3.4, read in full by the CXRS and pedestal critics today; [A] abstract read by me).
  - The same institute defines CES outliers as abrupt changes in space and time that come from failed automatic fits ([F-crit] Kim et al., NF 2024, appendix A; [A] by me).
  - The KSTAR pedestal study keeps only slices without an ELM in the 5 ms CES average ([F-crit] Ko et al., PoP 23 (2016) 062502).
  - Standard profile tooling filters ELMs "that commonly scatter data" ([A] Logan et al., FST 2018).
  - ELM synchronisation is used "on numerous machines" ([A] JET HRTS, RSI 2016).
  - Cold passive edge components and edge blends contaminate CX lines ([A] Boileau et al., PPCF 1989; [V-map] Viezzer 2011, McDermott 2017, Haskey 2018).
  - Edge light bursts at ELMs contaminate beam-emission measurements ([S] JT-60U MSE/BES two-wavelength background study; DIII-D BES edge light pulses).
  - **Why does that leave only a little?** What remains is a proxy (BES bursts instead of D-alpha), numbers from 11 discharges, and the width/shift asymmetry. The asymmetry is what any additive, roughly symmetric contaminant does to a three-parameter Gaussian fit without a baseline term [J; the fit model is from F-crit, Lee 2026 section 2.2].
    - **Why can it not be more on KSTAR?** The files have no spectrum, amplitude, residual or error. So continuum, passive C VI, plume and a real hot-ion event cannot be separated (the CXRS critique, Objection 1, Why 5-6). And a device that does report fit errors flags these same frames itself. N3 (unregistered, feasibility only) uses public MAST data: 630 shots, edge chords within 6 cm of the outermost valid chord, and a mid-plane D-alpha burst in the 5 ms exposure (>1.5 times its preceding 50 ms):

      | MAST, exposure [t, t+5) ms (and [t-2.5, t+2.5)) | burst B | quiet Q | B - Q [95 % shot bootstrap] |
      |---|---|---|---|
      | share without a usable T_i | 0.159 (0.174) | 0.096 (0.097) | +0.063 [+0.050, +0.075] (+0.076 [+0.063, +0.090]) |
      | median reported T_i error / its recent level | 1.179 (1.215) | 0.960 (0.959) | +0.219 [+0.183, +0.257] (+0.256 [+0.212, +0.307]) |
      | median T_i / its recent level | 1.001 (0.989) | 1.032 (1.032) | -0.031 [-0.047, -0.015] (-0.043 [-0.061, -0.024]) |
      | share of one-frame down / up departures | 0.128 / 0.076 | 0.068 / 0.057 | |

      In the frames where transient edge light enters the exposure, the MAST analysis:
      - more often returns no value;
      - reports an error about 20 % larger;
      - scatters in both directions, more often downward.

      The KSTAR signature (more missing values, larger one-frame scatter, upward on KSTAR) is the generic degraded-fit signature. The direction depends on the analysis [J].
      - **Why has nobody published the KSTAR version?** Practitioners do not publish "ELM frames are bad"; they remove them, using D-alpha and the reported error. What is specific to these files is the absence of the error bar, not the phenomenon.
        - **What would make it new?** Either of the following:
          - an attribution from spectra, refitting the 12 raw discharges' spectra with and without a baseline or passive component;
          - a physics result, for example an ELM-phase-resolved pedestal-top T_i at 10 ms that the label can carry.

          Both need data that are not here, or are unlikely at 10 ms with 11 discharges.

**ROOT.** The diagnostic group documents that ELM-time edge spectra give degraded fits at this radius. C3 quantifies that through a proxy, without the quantities that would make it attributable. On a device whose analysis reports errors, the same frames are flagged by the analysis itself.

**Verdict:** obvious from known results. The numbers and the BES proxy are partly new.
- Closest prior: Lee et al. 2025/2026, FED 222:115518 (doi 10.1016/j.fusengdes.2025.115518, [F-crit]/[A]) and Kim et al. 2024 (doi 10.1088/1741-4326/ad7304, [F-crit]/[A]).

**Referee, one sentence:** "The authors rediscover, through a BES proxy, the ELM contamination of edge CXRS fits that the KSTAR CES group reports and that standard analyses remove; without spectra or fit errors it cannot be attributed."

**Would settle it:**
- [needs data that do not exist here] Spectra, fitted amplitude and `ERR_BAR`.
- [existing data, CPU] A registered version of N3 with a Thomson-defined MAST band and chi-square column checks.

---

## 4. C4: the validation protocol

Each element is published practice:
- Discharge-level split: 19 of 25 papers ([P] `validation_methods.md`).
- Chronological or forward evaluation in fusion: Wan et al. NF 2021 ([A], with the shot-order argument from the survey's full-text reading); Wang 2025, Jia 2026 and Zheng 2024 ([P] survey); predictors trained only on earlier shots ([T] Murari et al., NF 2018).
- Use-case baselines with hypothesis tests on a large database ([A] Abbate et al., PoP 31 (2024), doi 10.1063/5.0190908, which found no significant advantage of transport codes over a two-parameter baseline). A constant-profile baseline is in Abbate 2025 ([P] survey).
- Shot resampling: Keith 2024 ([P] survey).
- Rolling origin and blocked CV: Tashman 2000, Roberts 2017, Cerqueira 2020 ([P] survey).
- Corrected fold t and MCS: Nadeau-Bengio, Hansen-Lunde-Nason ([P] survey).
- Boosted trees as the network-free reference: [T] Grinsztajn et al. 2022; Kates-Harbeck 2019 ([P] survey).

**Why-chain.**

- **Why is this not new?** Every component exists, cited above.
  - **Why is it still unusual in fusion?** Practice is loose:
    - 4 of 25 papers give any interval;
    - 0 of 5 gap-filling papers compare with hold or interpolation;
    - only Wan 2021 and Abbate 2021 warn about adjacent shots ([P] survey);
    - I found no peer-reviewed fusion critique of random shot splits beyond these (web search; the only hit was a non-peer-reviewed package write-up).
    - **Why does that not make it a contribution?** A protocol's value is shown by how often it changes published conclusions. Here it changed one conclusion on one chord: the fast-channel increment is not established forward in time. PPCF does not publish protocols without a result.
      - **Why has nobody written the fusion validation paper?** It needs re-analysis of several tasks with public data and stored predictions. That is a benchmark effort (TokaMark-type), not a by-product of one thesis.
        - **What would make it new?** Apply the protocol to several public MAST tasks (TokaMark/FAIR-MAST) and show that gains or rankings reported under random shot splits, scored against same-time labels and without hold/interpolation references, shrink or reverse.

**ROOT.** C4 is rigorous practice assembled from published parts. As a contribution it depends on changing conclusions across tasks, and one chord gives one instance.

**Verdict:** already shown (by components). It is not a stand-alone contribution.
- Closest: Abbate et al. 2024 (doi 10.1063/5.0190908, [A]).

**Referee, one sentence:** "Commendable rigour; validation design is not a result."

**Would settle it:** [existing data, CPU, days] the protocol on public MAST tasks. That can make a methods paper (MLST or NF), not a PPCF one.

---

## 5. C5: replay, validity classifier, MAST

**Why-chain.**

- **Why is the replay (B.71) not new?** It is a standard sensitivity analysis. Its result is negative for the use case: the network's margin over the causal GP lasts to 50 ms, and from 100 ms the GP has the lower error ([P] 4.49).
  - **Why is the validity classifier not much?** Most of its AUC comes from the CES history (0.9285 to 0.9622 in V1; +0.023 [+0.012, +0.036] forward in time; [P] 4.54). The KSTAR literature already produces per-channel outlier probabilities as quality indicators ([A] Kim et al. 2026) and SVM outlier flags ([A] Kim et al. 2024). Spectrum-reading real-time networks report their own uncertainty ([A] HL-3, NF 2026, aeb037).
    - **Why is the part that is left small?** Predicting that the next frame will be unusable before its spectrum is analysed matters only while the analysis is slow. Spectrum readers remove that window (0.56 ms per 32 channels at HL-3, [A]).
      - **Why does MAST not help?** No network was trained on MAST. The CES-only causal filter against hold is +0.076 [-0.006, +0.148] on the gate set, and offline interpolation is far better ([P] 4.12). MAST also shows T_i increments associated negatively with mid-plane D-alpha increments, and rotation increments associated as well ([P] `docs/paper/mast_coupling.md`). The driver and the radii differ from KSTAR's BES, so it neither repeats nor tests the KSTAR "width follows the light, shift does not" pattern.
        - **What would make it new?** A demonstrated real-time use of the validity flag in a control or monitoring loop.

**ROOT.** C5 items are sensitivity checks or a minor data-quality utility. Their closest equivalents exist (outlier probabilities) or are superseded (spectrum readers with uncertainty).

**Verdict:** already shown or obvious. The validity classifier from fast scalars is partly new and minor.
- Closest: Kim et al. 2026 (doi 10.1088/1741-4326/ae93dd, [A]); HL-3 (doi 10.1088/1741-4326/aeb037, [A]).

**Referee, one sentence:** "These are robustness checks and a utility; none carries a claim."

**Would settle it:** [needs data that do not exist here] a real-time deployment.

---

## 6. Across candidates

**The strongest reason the author's verdict is right.** Only one result survives a level target and all four validation designs: the history-only learned estimator over the causal GP (C1). Its size is set by the reference:
- half of it is a four-coefficient linear blend (N2);
- forward in time, a five-parameter robust filter has a lower RMSE than the learned estimator (139.1 against 143.9 eV, V4);
- it has no real-time use under the analysis that makes the labels.

The original thesis, that fast diagnostics add information about the pedestal-top T_i, shrinks under a level target. As built it is +0.013 [-0.025, +0.045]. Refitted it is +0.075 and +0.057 under random and cluster designs, and it is not established forward in time. The one physics-facing observation (C3) is the documented ELM contamination of edge CXRS fits, and an analysis with error bars flags it itself. Nothing here is a statement about ion temperature physics at the pedestal top, which is what PPCF publishes.

**The strongest reason it might be wrong.** I found no fusion cross-diagnostic estimator paper that tests whether its same-time gain survives against a reference free of the label's frame-specific error. I checked Diag2Diag, RTCAKENN, the EAST attention network, COMPASS, the EAST T_e and Z_eff papers, FusionMAE, TokaMark and its robustness benchmark, ST40, HL-3, NN-CES, and 31 works citing Diag2Diag screened by title. In this project that test removes about 85 % of the as-built fast-channel gain. The finding is general in form and cheap to check elsewhere. It would apply to any virtual diagnostic whose inputs and label share a light path, a beam or an exposure. That is a real gap. But it is an application of a textbook condition, on one chord, so it is a methods result and not a plasma result.

**Recommended claim and venue.**
- **For PPCF: nothing.**
- The only claim I would build anything around is C2, stated like this: "Scored against the same frame's CES value, a beam-emission-driven nowcaster adds about 9 % in MSE over its history-only twin; scored against the level of the following 50 ms it adds +1.3 % [-2.5, +4.5] as built, because the frame-specific CES fit departure co-varies with the beam-emission light of its exposure; a correction refitted to the level keeps +7.5 % [+3.9, +10.6] (random folds) and is not established forward in time."
- Title-level object: same-time scoring of cross-diagnostic estimators when input and label share a nuisance (a KSTAR edge-CES case).
- Venue: a thesis chapter now. After a second demonstration on public MAST data with fit errors, a short methods paper in MLST, or a note in FED. PPCF only if that generalization comes with a physics result, which nothing here offers.

**What could change the verdict, from files that exist, ranked by how likely each is to work.**

1. **[existing data, CPU, about 2 days] A C2 demonstration on MAST.**
   - Method: estimators of edge CXRS T_i from history + D-alpha (+ Thomson), scored against the same frame and against the following level, stratified by the reported T_i error.
   - N3 already shows that D-alpha bursts raise MAST's reported error by about 20 % and the missing share by 6-8 points.
   - If the same-time D-alpha gain shrinks against the level where the error is high, C2 becomes a two-device methods result.
   - Risk: the MAST D-alpha increment is small (+0.023 [+0.012, +0.038], [P] 4.12).
2. **[existing data, CPU] Beam-notch characterization of the KSTAR edge CES product.**
   - Content: 1,394 notches; first-row T_i 0.819 [0.796, 0.838], ECEI 0.926, BES 0.656; recovery in 40-90 ms; wall-field change ([P] 4.7, 4.42).
   - It quantifies what the CES group only states qualitatively: modulation "may potentially cause plasma disturbances" ([A] NN-CES) and causes "large radial displacement" ([F-crit] Lee 2020).
   - Very likely to hold up. Attribution (temperature, displacement or fit) is impossible here.
   - Venue: FED or JKPS note, not PPCF.
3. **[existing data, CPU] Robust references for C1.** Use a Student-t or mixture-likelihood causal GP and K0 on V1-V4/L5 with cluster intervals. If T0 keeps a clear margin forward in time, C1 becomes "a population prior over a robust local filter". That would be a FED/thesis result. It is unlikely given V4.
4. **[existing data, CPU] B.75 (registered).** History availability under the beam-modulation schedule. If a learned history-only estimator keeps a margin there, C1 gains an operational meaning. It is unlikely given B.71. *Note added after B.75's output was read:* the expectation for C1 was right. R10 (T0 over the GP with a 650 ms old history) holds in V1, +0.336, and does not hold in V4, +0.012 c [-0.279, +0.164]. But the same batch shows a large fast-channel increment under a stale history. That is the basis of possibility B in section 6b.
5. **[existing data, CPU, 11-12 raw discharges] Reported T_i averaged against the time since the last BES burst.** This is a physics result only if a 10 ms pedestal-top chord carries the inter-ELM T_i rise known from 65 µs AUG measurements ([A] Cavedon et al., PPCF 2017). That is unlikely with 10 ms frames and 12-17 bursts per second.

What would change it but does not exist here: the chord's spectra, fitted amplitude and `ERR_BAR`; an equilibrium; D-alpha; the output of a real-time CES analysis.

---

## 7. Unregistered checks (N1-N3)

| id | script | question | result | limits |
|---|---|---|---|---|
| N1 | `n1_robust_median.py` → `.out` | Does a trivial robust causal filter (median of the last five measured values, stored by B.73) give C1? | No. On L5 and V1 (215,369 rows, 613 discharges, identical for r2-r4): m5 against GP -0.125 [-0.157, -0.090]; T0 against GP +0.168 [+0.148, +0.187] (r2), +0.164, +0.167 (r3, r4) | one-level discharge bootstrap, 1,000 draws |
| N2 | `n2_linear_blend.py` → `.out` | How much of C1 does a four-coefficient linear blend give? | Blend (intercept 39-43 eV, GP 0.57-0.68, hold -0.01 to -0.04, m5 0.28-0.38) fitted on half the discharges and scored on the other half: +0.086 [+0.072, +0.101] over GP; T0 over the blend +0.090 [+0.076, +0.104]. RMSE: GP 145.6, blend 139.2, T0 132.8 eV | r2 trees only; two-way cross-fit; a first version wrongly used a mean over `thist`, which holds five different history scalars and not five values; it was corrected and that output was discarded |
| N3 | `n3_mast_burst.py` → `.out` | Does a D-alpha burst in a MAST CXRS exposure move the fitted edge T_i, its reported error and the velocity? | Table in section 3. In addition, the median velocity change against its recent level is +0.43 (B) against +0.84 (Q) km/s, B-Q -0.41 [-0.77, -0.04] for the [t, t+5) window | feasibility only. The exposure convention is not in the metadata (two windows read). Edge chords are defined by radius, not by the Thomson band. D-alpha is a wide-angle mid-plane view. First 800 shots with both files (630 used). The chi-square and CX/background columns were not used because their column count does not match the chords |

---

## 8. Sources

Read levels are as defined in the header. "Lead" means the source was verified by the session lead before this task.

| source | DOI / id | read level | used for |
|---|---|---|---|
| Lee et al., NN-CES, FED 222:115518 | 10.1016/j.fusengdes.2025.115518 | [A]; [F-crit] (both critiques) | C3 (ELM distortion at R ≳ 2.2 m), notch |
| Lee, Lee, Ko, two-Gaussian CES, FST 76 (2020) | 10.1080/15361055.2020.1790712 | [A]; [F-crit] CXRS | C3, notch |
| Kim et al., SVMR+GPR outliers on KSTAR CES, NF 64 (2024) 106052 | 10.1088/1741-4326/ad7304 | [A]; [F-crit] CXRS (appendix A) | C3, C5 |
| Kim et al., good-and-bad mixture profile fitting, NF 2026 | 10.1088/1741-4326/ae93dd | [A] | C1, C5 |
| Ko et al., Phys. Plasmas 23 (2016) 062502 | DOI not checked | [F-crit] CXRS | C3 |
| Logan et al., OMFIT profiles, FST 2018 | 10.1080/15361055.2017.1386943 | [A] | C1, C3 |
| JET HRTS ELM synchronisation, RSI 2016 | 10.1063/1.4939855 | [A] | C3 |
| Boileau et al., PPCF 31 (1989) | 10.1088/0741-3335/31/5/006 | [A] | C3 |
| Cavedon et al., PPCF 59 (2017) | 10.1088/1361-6587/aa7ad0 | [A] | C3, rank 5 |
| McDermott et al., RSI 88 (2017) | 10.1063/1.4993131 | [A] | C3 |
| Haskey et al., RSI 89 (2018) | 10.1063/1.5038349 | [T]; content [V-map] | C3 |
| JT-60U MSE/BES ELM background (OSTI 22053976); DIII-D BES edge light; NSTX-U CHERS background slides | none | [S] | C3 |
| Ferro, QJRMS 143 (2017) | 10.1002/qj.3115 | [A] | C2 |
| Patton, J. Econometrics 160 (2011) 246 | 10.1016/j.jeconom.2010.03.034 | [T] + [S] | C2 |
| Batson & Royer, Noise2Self (2019) | 10.48550/arXiv.1901.11365 | [A] | C2 |
| Lehtinen et al., Noise2Noise (2018) | 10.48550/arXiv.1803.04189 | [A] | C2 |
| Gruber et al., JGR Atmos. 121 (2016) | 10.1002/2015jd024027 | [A] | C2 |
| Podsakoff et al., J. Appl. Psychol. 88 (2003) | 10.1037/0021-9010.88.5.879 | [A] | C2 |
| Jalalvand et al., Diag2Diag, Nat. Commun. 16 (2025) | 10.1038/s41467-025-63492-1 | [A]; full text by lead | C2 |
| works citing Diag2Diag (31 titles; abstracts of 4: TokaMark robustness 2607.11915, AI latency 2610.00845, KSTAR reactor diagnostics PST aeb854, GenToka 2611.08153) | OpenAlex `cites:W4414533721` | [T]/[A] | C2 (negative search) |
| ST40 cross-diagnostic inference (arXiv 2407.18741) | 10.48550/arXiv.2407.18741 | [A] | C2 (SXR → TS; not CXRS) |
| HL-3 real-time CXRS, NF 2026 | 10.1088/1741-4326/aeb037 | [A] | C5, C2 |
| Montero-Manso & Hyndman, IJF 2021 | 10.1016/j.ijforecast.2021.03.004 | [T] | C1 |
| Revach et al., KalmanNet, IEEE TSP 2022 | 10.1109/tsp.2022.3158588 | [A] | C1 |
| Morosohk et al., CTPP 2023 | 10.1002/ctpp.202200153 | [A] | C1 |
| RAPTOR EKF, NF 2025 | 10.1088/1741-4326/add16e | [A] | C1 |
| Chilenski et al., NF 55 (2015) | 10.1088/0029-5515/55/2/023012 | [T] | C1 |
| Abbate et al., PoP 31 (2024) | 10.1063/5.0190908 | [A] | C4 |
| Wan et al., NF 61 (2021) | 10.1088/1741-4326/abf419 | [A]; split rule from survey full text | C4 |
| Murari et al., NF 58 (2018) | 10.1088/1741-4326/aaaf9c | [T] | C4 |
| Grinsztajn et al. 2022 | 10.48550/arXiv.2207.08815 | [T] | C4 |
| Pavone et al., PPCF 65 (2023) review | 10.1088/1361-6587/acc60f | [A] | background only |
| Fusion-ML validation survey (25 papers), methodology references | `docs/paper/validation_methods.md` | [P] (the survey's own read levels apply) | C4 |
| RTCAKENN, EAST attention, COMPASS, EAST T_e/Z_eff, FusionMAE, TokaMark | as in the task brief | lead verified | C2, C4 |

**Negative search record.** Each item below was looked for and not found. Absence is limited to the sources searched: OpenAlex title/abstract and full-text index, `cites:` on Diag2Diag, and WebSearch. Conference proceedings, KSTAR reports and Korean-language journals were not searched systematically.
1. A fusion-ML paper that scores a cross-diagnostic estimator against a target without the label's frame-specific error, or that names input-label error correlation.
2. A peer-reviewed fusion critique of random shot splits beyond the warnings of Wan 2021 and Abbate 2021.
3. A quantification of the response of the reported edge CES T_i to KSTAR beam-modulation notches.
4. Use of BES, rather than D-alpha, to gate CXRS frames.
