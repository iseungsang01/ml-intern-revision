# Critique from the charge-exchange side (2026-11-20)

Role: a CXRS diagnostician asked to break the work before submission. No balance for its own sake.

**What I read.** `CLAUDE.md`, `PAPER_DESIGN.md`, `PROJECT_KNOWLEDGE.md` sections 1, 2, 3, 4 (without 4.13) and 7, `docs/paper/file_rules.md`, `b70_d1.md`, `b71_scenarios.md`, `b65_inclusive.md`, `b62_beam_return.md`, `b61_blip_dose.md`, `b57_yardsticks.md`, `b46_beam_off.md`, `b43_target_checks.md`, `b40_noise.md`, the main text `docs/paper/ppcf/main_ppcf.tex`, and the KSTAR papers in `data/literature/` named below. I did not read section 4.13, the other files of `docs/paper/review_loop/`, or the grading passages of `THESIS_RESULTS.md`.

**What I ran.** Read-only checks on the 641 files, the raw streams of the twelve discharges and the stored D1 estimates of B.70 (r2-r4). Scripts and the run order are in `docs/paper/review_loop/critique_20261120_cxrs_checks/` (ids C0 to C26). Nothing was trained; no tracked file was changed; nothing was committed.

**Marks.** [V: source] = verified, with the file and line or the check id and the number. [J] = my judgement as a diagnostician. Everything I computed today was chosen after looking at the data: it is after the fact and unregistered. Intervals on the stored estimates are two-level (a re-partition of r2-r4 drawn, then its discharges resampled, 2,000 to 4,000 draws); intervals on the files alone (the rotation tests, C8 and C18) resample discharges (400 to 500 draws).

**A definition used throughout.** *Following-frames reference*: for a scored frame t, the median of the reported T_i of frames t+1 to t+3 that lie within 35 ms (at least two of them). These are independent fits that no causal estimator read. Scoring an estimate of frame t against it instead of against the frame's own value removes whatever is specific to the fit of frame t and keeps whatever lasts.

---

## 0. The finding that decides the rest

I rebuilt the D1 table from the stored per-row estimates (222,672 rows per re-partition; two-stage estimator over the causal GP +0.177, over its twin +0.097, network over twin +0.052: identical to `b70_d1.md`) [V: C4]. Then I scored the same estimates against the following-frames reference, on the 96.4 % of D1 rows where it and a signed causal-GP estimate exist [V: C7b]:

| pair | skill against the frame's own value | skill against the following frames | share of the squared-error reduction that remains |
|---|---|---|---|
| two-stage estimator over its twin (the paper's fast-channel increment) | +0.114 [+0.093, +0.135] | +0.024 [+0.002, +0.044] | 0.156 [0.010, 0.278] |
| network over its twin | +0.064 [+0.039, +0.088] | +0.018 [-0.013, +0.045] | 0.207 [-0.219, 0.430] |
| filter with BES over filter without (KFX over KF0) | +0.065 [+0.055, +0.075] | +0.038 [+0.026, +0.050] | 0.455 [0.335, 0.579] |
| two-stage estimator over the causal GP | +0.190 [+0.169, +0.211] | +0.127 [+0.106, +0.146] | 0.500 [0.408, 0.597] |
| two-stage twin over the causal GP (no fast channel) | +0.086 [+0.072, +0.098] | +0.105 [+0.085, +0.124] | 0.915 [0.791, 1.063] |
| causal GP over hold | +0.318 [+0.301, +0.335] | +0.406 [+0.385, +0.427] | 1.097 [1.041, 1.159] |

RMSE against the following frames: hold 195.6, causal GP 150.7, two-stage twin 142.6, two-stage estimator 140.8 eV [V: C7b]. Three other definitions of the reference (mean of t+1..t+3, median of t+2..t+4, frame t+2 alone) give 0.10 to 0.21 for the two-stage row and 0.84 to 0.92 for the history-only row [V: C11].

Everything that reads only the CES history keeps its gain. Everything credited to the fast diagnostics loses 55 to 85 % of it. The rest of this document is about why, and about what the reported value is.

---

## 1. Objection 1 (the root of the scientific claim): the fast-channel increment is the width fit of one frame, not an ion temperature

**Statement.** The sentence the paper is built on, "the BES value of the present frame carries information about the reported T_i of that frame beyond the CES history", is true, and it is the problem. Most of what BES predicts is a component of the reported value that exists in that frame only, that scales one-to-one with the beam-emission sample, and that the rotation obtained from the same fit of the same line does not share.

- **Why 1. Why do I say the gain sits on something that is gone a frame later?**
  The table of section 0 [V: C7b]. The authors' own numbers give the same answer once they are put in the paper's currency: for the network minus its twin the regression slope on the own frame is beta_0 = 0.693 and the persistence is rho_1 = 0.810 [V: `b57_yardsticks.md`:49-50], so the slope on the next frame is 0.561, and a correction with slope 0.5 has no squared-error value: (2 x 0.561 - 1) / (2 x 0.693 - 1) = 0.32 of the reduction remains. My slopes, with the following-frames reference in place of the next frame: 0.713 and 0.545 for the network, 0.888 and 0.560 for the two-stage estimator [V: C14]. The manuscript reports rho_1 and writes "most of the network's correction is still present in the next, independently fitted frame" in the abstract and the conclusion [V: `main_ppcf.tex`:81-82, 803-804], concedes in one clause that rho_1 "is not the share of the error removed" [V: `main_ppcf.tex`:702-704], and never ran the test on the estimator of the paper [V: `main_ppcf.tex`:783-784]. A test that fails in mean squared error was reported in a unit in which it passes.
  - **Why 2. Why would an estimator chase a one-frame component?**
    Because the target has one and squared error on the frame's own value pays for chasing it. Among the D1 scored rows, 5.9 % are single-frame rises of more than 35 % above both 10 ms neighbours and 5.0 % are single-frame falls below 74 % of them (S beyond +-0.3, S = ln T_t minus the mean of ln T_(t-1) and ln T_(t+1)); the next frame is back at the level of the frame before (median ln T_(t+1)/T_(t-1) = -0.04 after a rise, -0.06 after a fall; median size of the excursion a factor 1.59 and 0.61) [V: C3, C5]. These 10.9 % of rows carry 48 % of the two-stage estimator's squared error and 61 % of its reduction over its twin; on the rises alone the two-stage estimator has +0.077 [+0.061, +0.094] over its twin [V: C5]. (Classes defined on the outcome favour whichever estimator moves, so I use them to locate the error, not to judge skill; the judgement rests on section 0 and on Why 4, which condition on nothing or on the input.) The paper's own "floor" is this component: minus the covariance of successive increments, "the frame-to-frame independent variance", of which the text says that "a present-frame signal correlated with the conditions of that frame's fit is not bounded by it, which the persistence test of [the section on fit or plasma] addresses" [V: `main_ppcf.tex`:407-416]. It addresses it, and the answer is that the fast channels are mostly reducing the floor.
    - **Why 3. Why is a one-frame component not a temperature?**
      Three independent reasons.
      (i) No source or sink at a pedestal top raises the C6+ temperature by a median 59 % inside one 5 ms exposure and removes it before the next. Plasma events at such a chord are one-sided and recover slowly (an ELM crash, a beam interruption); here rises and falls are equally frequent (ratio 1.14 to 1.19) [V: C3, C5]. [J]
      (ii) The KSTAR group's own working definition of a CES outlier is this object: a point is an outlier "if it shows abrupt changes both in space and time", judged on the profiles at t0 - 10 ms, t0 and t0 + 10 ms, and "such categorized outliers tend to have poor fitting results": the automatic Levenberg-Marquardt Gaussian fit "has failed to obtain a reasonable fitting result" [V: Kim et al., Nucl. Fusion 64 (2024) 106052, appendix A]. The KSTAR pedestal study with this diagnostic keeps only "time slices without any overlap with an ELM crash in the 5 ms time average of CES measurements" and drops edge points "with a large scatter" [V: Ko et al., Phys. Plasmas 23 (2016) 062502, p. 2].
      (iii) The rotation is a witness and it does not move. In the published procedure T_i and V_T are the width and the shift of one three-parameter Gaussian fitted to one background-subtracted spectrum [V: Lee et al., Fusion Eng. Des. 222 (2026) 115518, section 2.2, eq. 3-4]. On the 64,339 triples of 386 discharges where the file carries a fresh rotation value in all three rows, the single-frame excursion of T_i and that of V_rot have a rank correlation of +0.004 [-0.006, +0.015]; in frames where T_i rises by more than 35 % the median relative excursion of the rotation is +0.3 % [0.0, +0.5], in frames where it falls +0.1 % [-0.1, +0.3] [V: C8]. The witness is alive: fresh rotation values scatter by about +-7 % from frame to frame [V: C18] and fall to 0.973 [0.960, 0.989] in the first frame after a beam interruption, where T_i falls to 0.814 [V: `b62_beam_return.md`:38, `b61_blip_dose.md`:20]. By that calibration of the authors a plasma perturbation that moves T_i by 40 to 60 % should move the rotation by 6 to 9 %. It moves it by a few tenths of a per cent. The one physical single-frame rise I know of at an edge chord, hot pedestal ions thrown outward by an ELM past a chord that sits in the gradient, brings its rotation with it; the shift does not move, so that is not what the files show. [J]
      - **Why 4. Why does that component line up with BES?**
        Because the fitted width follows the beam-emission sample one-to-one, and only the width. For every grid row with ten regular rows before it (so no beam interruption in the last 100 ms), relative to the median of those ten rows [V: C18]:

        | BES sample at the stamp / level before | rows with T_i | median T_i ratio | rows with fresh V_rot | median V_rot ratio | median ECEI ratio |
        |---|---|---|---|---|---|
        | 0.40-0.60 | 296 | 0.500 | 209 | 1.006 | 0.867 |
        | 0.60-0.80 | 2,112 | 0.663 | 1,321 | 1.007 | 0.912 |
        | 0.80-0.90 | 7,891 | 0.844 | 3,278 | 1.003 | 0.953 |
        | 0.90-0.97 | 38,752 | 0.962 | 13,820 | 1.000 | 0.977 |
        | 0.97-1.03 | 95,396 | 1.003 | 33,524 | 1.000 | 1.000 |
        | 1.03-1.10 | 38,605 | 1.051 | 13,470 | 1.001 | 1.026 |
        | 1.10-1.20 | 9,355 | 1.156 | 3,609 | 1.004 | 1.046 |
        | 1.20-1.40 | 3,352 | 1.336 | 1,556 | 1.007 | 1.052 |
        | above 1.40 | 1,263 | 1.599 | 658 | 1.001 | 1.044 |

        On the rows that have both, the log-log slope of T_i on the BES sample is 1.28 [1.15, 1.44] and that of the rotation is 0.018 [-0.016, +0.050] [V: C18]. The ECEI level moves about a quarter as much, so something in the edge does change in those frames; I do not say the plasma is quiet. I say the reported width responds to it like no temperature does and the reported shift does not respond at all.
        Split by the sign of the BES departure, which conditions on the input only, the two halves of the story separate [V: C25, C26]:

        | frames with the BES sample | share of rows | two-stage over twin: own frame | following frames | network over twin: following frames | KFX over KF0: following frames | share of the own-frame reduction (two-stage) |
        |---|---|---|---|---|---|---|
        | below 0.9 of its recent level | 0.057 | +0.166 [+0.122, +0.208] | +0.114 [+0.074, +0.152] | +0.110 [+0.054, +0.158] | +0.149 [+0.120, +0.180] | 0.244 |
        | 0.9-0.97 | 0.189 | +0.107 [+0.075, +0.142] | +0.041 [+0.015, +0.067] | +0.039 [+0.003, +0.072] | +0.062 [+0.050, +0.075] | 0.150 |
        | 0.97-1.03 | 0.462 | +0.064 [+0.043, +0.083] | +0.007 [-0.018, +0.026] | -0.017 [-0.064, +0.014] | +0.004 [-0.002, +0.009] | 0.156 |
        | above 1.03 | 0.292 | +0.131 [+0.104, +0.158] | -0.013 [-0.049, +0.019] | -0.003 [-0.047, +0.033] | +0.002 [-0.016, +0.019] | 0.450 |

        A fall of the beam emission announces a fall of the reported value that lasts. A rise of the beam emission announces a rise of the reported value that is confined to that frame. The second kind holds 45 % of the two-stage increment, 55 % of the network's and 46 % of the filter's [V: C26], and for all three estimators it is worth nothing 10 to 30 ms later. "Two estimators of different kinds agree" (the network and the five-parameter filter) is no protection: they agree on the artefact as well.
        "Lasts" is not yet "plasma". In the dose table the rotation is flat on the low side too (1.003 to 1.007 where T_i reads 0.50 to 0.84), unlike after a programmed interruption (0.973). A reported width that stays low for as long as the beam emission stays low, with the shift unchanged, is what a weak active line on an imperfect background gives, and a bias shared by neighbouring frames passes the following-frames test, as the authors say of their own persistence test [V: `main_ppcf.tex`:708-710]. Objection 2 shows how far that can go. [J]
        - **Why 5. Why would a fitted width follow the beam emission of the frame?**
          Because of what the analysis is. One Gaussian with amplitude, shift and width, and in the published model no baseline term, is fitted by Levenberg-Marquardt to the beam-on spectrum minus a background interpolated between two beam-off frames [V: Lee 2026, section 2.2; Lee, Lee and Ko, Fusion Sci. Technol. 76 (2020) 787, p. 789; Kim 2024, appendix A]. The same group states the conditions under which this is valid and that KSTAR does not meet them: the beam-off spectrum "cannot be exactly the same" as the background during beam-on, so the modulation "is usually required to be a few tens hertz"; on KSTAR it is 1 to 5 Hz [V: Lee 2020, p. 788], and in these files the median spacing of beam-off frames is 1.48 s [V: C11]. At this radius the group reports that "the active CX signal is reduced due to the increased passive emission of thermal neutrals", that the spectra are "more easily distorted by transient events such as ELMs and by discrepancies between beam-on and beam-off phases", and that these "degrade the fitted parameters" at R above about 2.2 m [V: Lee 2026, section 3.4]. In such a fit, any light that is additive and roughly symmetric about the line and that changes faster than the interpolated background (continuum, which goes as density squared; passive C VI of similar shift; plume, which is proportional to density and cannot be subtracted by modulation) goes into the width and not into the shift. Which of these it is, I cannot say from the files. [J]
          - **Why 6. Why did nobody see it, and why can it not be settled here?**
            The authors had the witness and read it as architecture: fast-channel increments correlate with T_i increments (at least +0.127) and not with V_rot increments (at most 0.044), and from this the V_rot branch was denied the fast channels as an input that does not physically reach it (my translation of the record) [V: `PROJECT_KNOWLEDGE.md`:91, 104]. To a spectroscopist the same two numbers say that the photons of one line changed its width and not its position, which is what a fit artefact looks like. To go further one needs the spectrum of the frame, or at least the fitted amplitude, the fit residual and the background level. The files have none; the KSTAR tree has an error bar for every CES node (`:ERR_BAR`) and the facility guide says of these values that "it is necessary to be checked by the person in charge of CES before using it because Ti and Vt data is automatically calculated and uploaded" [V: KFE Diagnostics Data User Guide v.20260304, slide 62]. This ends at (c): only new data can say what the one-frame component is. What existing data can say, they have said above.

**THE ROOT.** The squared-error gain credited to the fast diagnostics is earned mostly on a component of the reported value that lasts one frame, follows the beam-emission sample with slope 1.3 and is not shared by the rotation from the same fit (slope 0.02): the behaviour of the automatic width fit at a chord where the group that runs the diagnostic says the fit is degraded. No quantity in the files can show that component to be ion temperature, and two that are in the files (the next frames, the rotation) say it is not.

**What would settle it.**
- [existing data, CPU] Register and repeat what I did today: every fast-channel statement (D1-2, D1-3, the six of B.59, KX1-KX6 of B.56, the thirds by temperature, the quiet rows) scored against the following-frames reference; the same split by the sign of the BES departure; the rotation dose table and excursion test beside it. Half a day.
- [existing data, CPU] Size the correction for what lasts: shrink the fast-channel correction by a factor fitted on training discharges against the following-frames reference. With a slope of 0.56 on that reference an optimal shrink would be worth about +0.06 over the twin instead of +0.024 [V: C14 for the slopes; the +0.06 is arithmetic on them, J].
- [needs training, GPU] Train the estimator toward a target that is not the frame's own fit (the following-frames median as an offline teacher; the estimator stays causal) and see what the fast channels are then worth.
- [needs data that do not exist here: `\CES_TI18:ERR_BAR`, the fitted amplitude and fit residual of this channel, the raw spectra of this channel for the twelve raw discharges, D-alpha for ELM timing.]

**What may honestly be claimed if it is not settled.** "The BES sample of a frame predicts the frame-specific departure of the automatic CES width at this chord (+0.11 over the twin on the frame's own value). Against the level reported over the following 10 to 30 ms the gain is +0.02 [0.00, +0.04] for the two-stage estimator and is not established for the network. It lasts where the beam emission falls (6 % of frames, +0.11 [+0.07, +0.15]) and does not where it rises (29 % of frames, -0.01 [-0.05, +0.02]). The rotation from the same fit does not share the departure. We do not claim that the departure is ion temperature."

---

## 2. Objection 2: the analysis writes a T_i with the beam off, and nobody here knows which light, which background or which method a value comes from

**Statement.** The target is described as "the temperature of C6+ ions from the Doppler width of the charge-exchange line excited by the heating beam" [V: `main_ppcf.tex`:239-245]. The files contain values written for frames exposed with the beam emission at its beam-off level. The pipeline has no gate on the presence of an active signal, and for more than half of the values the files do not even show a pair of beam-off frames around them.

- **Why 1. What is the evidence?**
  In the twelve discharges with raw streams, 15 rows are stamped with all nine BES channels below 0.4 of their level; in all 15 the raw BES stays below half its level for 100 % of the 5 ms after the stamp (the published exposure: "the exposure time of the CCD camera is set to be 5 ms for the 10-ms time resolution" [V: Lee 2020, p. 789]); 13 of the 15 carry a T_i [V: C15]:

  | discharge | stamps (s) | reported T_i (eV) | level before (eV) |
  |---|---|---|---|
  | 32027 | 5.132, 5.142, 5.362, 5.372 | 979, 557, 1,711, 1,133 | 1,063 to 1,202 |
  | 31873 | 13.002 to 13.042 | 561, 226, 343, 314, 225 | 542 to 578 |
  | 31747 | 8.002 to 8.052 | 1,117, 565, 516, none, 331, none | 871 to 1,015 |

  Discharge 31747 then goes on: from 8.001 to 8.491 s the raw nine-channel BES is at 0.208 of its earlier level, the same as inside the programmed beam-off frames of that discharge (0.08 to 0.23), with 100.00 % of the raw samples below half the earlier level; the file carries a T_i in 45 of those 50 frames, median 751 eV (quartiles 656 and 842; 1,015 eV before) [V: C19, C20]. All 45 are scored rows of D1 [V: C23]. Campaign-wide, 181 rows in 71 discharges are stamped with every BES channel below 0.4 of its level in the ten rows before; 133 of them (0.735) carry a T_i, at a median 0.86 of the level before (quartiles 0.52 and 1.12), 57 % within a factor 1.5 against 92 % for ordinary rows [V: C10]. With a looser rule (every channel below 0.3 of its discharge median) there are 1,727 such rows in 210 discharges and 700 carry a T_i [V: C16]. The authors know three of these frames (979, 1,711 and 1,133 eV in 32027), call them a counterexample, and write that "the files do not say from which beam the values of those rows come" [V: `PROJECT_KNOWLEDGE.md`:198; `file_rules.md`:54-61; `main_ppcf.tex`:219-221]. They did not look for the others.
  - **Why 2. Why would a value exist without the beam?**
    Three readings, none testable here. (a) The CES line of sight also crosses an NBI-1 source that the BES filter does not see and that stayed on (both systems collect through the M port and list the K port as measuring position [V: KFE guide, slides 57 and 60], so they look at the same place, but NBI-1 has three sources). (b) The fit ran on what was left of the passive C VI light after the subtraction. (c) Where no pair of beam-off frames exists, the value comes from another method: the two-Gaussian fit that needs no modulation, which the same group describes as having "larger uncertainty" and as failing "to reconstruct the edge profile reliably" in some cases [V: Lee 2020, p. 792-793]. [J]
    - **Why 3. Why does each reading damage the paper?**
      (a) Then BES is not "the beam the CES uses": a BES dropout is a partial loss of beam power, and a programmed beam-off frame is not free of active light, so the background contains charge exchange. (b) Then a width of 650 to 850 eV is available from passive light alone at this chord, for half a second, with the ordinary frame-to-frame scatter; with the beam on, a passive residual cannot be told from the active line by its width, and its weight in the fitted line changes with exactly what BES measures. This is Objection 1 seen from the other side. (c) Then the target is a mixture of two analyses with different errors, unlabelled.
      - **Why 4. Why were those frames not dropped, when the programmed beam-off frames are?**
        The analysis evidently takes its beam-off frames from the programmed modulation and not from the measured beam: all 45 absent-frame groups inside the raw windows coincide with a beam interruption [V: `file_rules.md`:49], yet 13 of 15 exposures without beam kept a value [V: C15]. And the standard analysis "requires two adjacent background frames" [V: Lee 2020, p. 793-794]; a published analysis of one of these discharges says ion temperatures "are only available between two sequential beam blips" [V: Heo et al., Nucl. Fusion 65 (2025) 036003, section 2]. In the files, of 226,989 values only 46.1 % lie between two absent frames of their segment; 31.5 % precede the first and 22.5 % follow the last; 132 of 641 discharges have no absent frame at all [V: C11, C12]. Values outside a pair have three times the missing rate (10.2 and 11.8 % against 3.6 %), two to four times the rate above 3 keV (0.54 and 0.93 % against 0.23 %) and, within the same discharge, twice the rate of excursions beyond +-0.3 (3.4 and 2.7 % against 1.7 and 1.6 %, medians over discharges) [V: C21]. Either the bracketing frames lie outside the delivered window or something else produced these values. The files do not say.
        - **Why 5. Why can the authors not find out?**
          No analysis code, no beam waveform per source, no spectrum, no method flag. The record says more data cannot be had [V: `PROJECT_KNOWLEDGE.md`:85]. This ends at (c).

**THE ROOT.** Nobody working on this knows the rule by which the KSTAR analysis decides that a frame has a T_i or which background and method it used, and the files show it writing ordinary-looking values for half a second with the beam emission off.

**What would settle it.**
- [existing data, CPU] List every exposure without beam emission from the twelve raw BES streams against the reported values (done: 13 of 15, plus 45 of 50 in 31747). Remove every row with the BES sample below 0.4 of its level from targets and history as a sensitivity, and report the headline table separately for values between two absent frames (46.5 % of D1 rows; there the two-stage estimator has +0.134 over its twin, against +0.088, +0.086 and +0.017 before the first, after the last and in discharges without any [V: C21, no intervals]).
- [needs data that do not exist here: the on/off waveform of each NBI-1 source, which sources the M-port CES sightline and the BES filter see in 2022, the frame-rejection rule of the analysis, and a per-value flag of the method (beam modulation or two-Gaussian).]

**What may honestly be claimed.** "The target is whatever the automatic analysis writes. In 13 of 15 frames exposed without the beam that the BES views, and in 45 of 50 frames after that beam ended in one discharge, it wrote a value. 54 % of the values are not between two beam-off frames within the file. Which light and which method those values come from is not known to us."

---

## 3. Objection 3: the population is set by a threshold on the answer, and the result changes with the threshold

**Statement.** "Measured frame" means "value at or below 3 keV". The threshold cuts through a continuous tail, takes nothing from the low side, and leaves the squared error on values that the KSTAR group would call outliers. The existence of the increment depends on where the threshold is put.

- **Why 1. Is 3 keV a boundary of anything?**
  No. Counts by band [V: C1]: 1.5-2 keV 4,207; 2-2.5 keV 859; 2.5-3 keV 430; 3-4 keV 348; 4-6 keV 326; 6-10 keV 321; above 10 keV 202 (largest 14,984 eV; smallest 0.002 eV). Share of the values of a band that exceed twice the median of their ten neighbouring rows: 2-2.5 keV 0.36, 2.5-3 keV 0.57, 3-4 keV 0.71, 4-6 keV 0.79 [V: C3]. At or below 3 keV there remain 3,128 values above twice and 7,112 values below half that local median, against the 1,197 values the rule removes [V: C3].
  - **Why 2. Why does it matter for the result?**
    Because the mean squared error is decided there. In the D1 scored rows, targets beyond twice or below half the local median are 4.85 % of the rows and carry 54.5 % of the two-stage estimator's squared error; the rows below half the local median (3.1 %) give 37 % of its reduction over the causal GP and 32 % of that over its twin [V: C5]. Rows above 1.5 keV are 2.4 % of the rows and carry 50 % of its squared error; rows above 2 keV are 0.54 % and carry 36 % [V: C13]. And the reading flips with the rule, in the authors' own batches: with the values above 3 keV kept the increment is -0.002 [-0.017, +0.010] [V: `b65_inclusive.md`:10]; with the 3 keV rule +0.052; with the rows flagged by the good-and-bad mixture set aside (16 % of the scored rows) the network over its twin is -0.015 [-0.110, +0.052] and over the causal GP +0.040 [-0.045, +0.106] [V: `b70_d1.md`:184, 210-211].
    - **Why 3. Why was the 3 keV rule kept after the mixture flagged 16 %?**
      Because the mixture only looks at T_i and cannot tell a fit failure from a fast real change (my translation of the record) [V: `PROJECT_KNOWLEDGE.md`:866]. Correct, and it applies word for word to the 3 keV rule, which is the same kind of rule with a looser threshold on one side only. The mixture was dismissed for a defect the adopted rule shares. [J]
      - **Why 4. Why is there no proper criterion?**
        Because the files carry no fit quality, and the value alone is all there is. What a group that uses this diagnostic does instead is in Why 3 of Objection 1 (ELM-free slices, scatter cuts, outlier classifiers). And the low side has an instrument limit that the rule ignores. The published instrumental width is 1.14 A [V: Ko et al., Rev. Sci. Instrum. 81 (2010) 10D740, wavelength-calibration section; the 2010 spectrometer, so the 2022 value may differ]; the Doppler FWHM of C VI 529.05 nm equals it at 94 eV [V: arithmetic, FWHM/lambda = 7.715e-5 sqrt(T[eV]/12)]. Below that the temperature is a small difference of two nearly equal widths. 3.8 % of the values are below 100 eV and 1.5 % below 50 eV [V: C1]; of the values below 25 eV, 88 % sit below half their local median, so they are downward spikes and not cold plasma [V: C3]. The analysis itself is stated to neglect Zeeman splitting on the ground that T_i is "typically well above 200 eV" [V: Lee 2026, section 2.2]; 11.5 % of these values are below 200 eV [V: C1]. On the scored rows below 50 eV the median error of the two-stage estimator is 6.1 times the value (hold: 2.7 times); from 50 to 100 eV it is 1.10 times (hold: 0.53) [V: C13].
        - **Why 5. What was never tested?**
          That the failures left in the training targets and in the history inputs do not shape what the network learns from BES. They are predictable from BES: the next value being missing or above 3 keV is predicted with an area under the curve of 0.96 to 0.97 once the fast channels are read [V: `PROJECT_KNOWLEDGE.md`:362]. A network trained with squared error on a target whose failures are announced by one of its inputs will learn to announce them. A cut applied at scoring cannot undo that. (a) An assumption nobody tested, and (c) the error bars exist in the KSTAR tree and not here.

**THE ROOT.** Without a per-frame quality measure the population is defined by a threshold on the quantity being predicted, and the paper's increment is +0.10, +0.05, 0.00 or -0.02 according to a threshold that nothing about the instrument justifies.

**What would settle it.**
- [existing data, CPU] The headline table under a ladder of causal, two-sided rules applied to scoring rows and to the history of every reference (value beyond 1.5, 2 and 3 times a causal running median; a floor at 100 and at 200 eV).
- [needs training, GPU] The same rule in the training targets and inputs, one rule per batch.
- [needs data that do not exist here: `\CES_TI18:ERR_BAR`.]

**What may honestly be claimed.** "Under the 3 keV rule the two-stage estimator has +0.177 over the causal GP and +0.097 over its twin. The increment is zero with the values above 3 keV kept and is not established for the network when rows unlike their neighbours are set aside. Half of every estimator's squared error is on the 2.4 % of rows above 1.5 keV."

---

## 4. Objection 4: the history the estimators read does not exist in real time, and where it could, the nowcast is worth one frame

**Statement.** Every causal estimator here reads the CES value of the previous frame. The analysis that produced these values cannot deliver that value until the next beam-off frame. Under that schedule the estimator has no advantage. Under a reader fast enough to supply the history, the frame's own value arrives by itself one frame later.

- **Why 1. Is the previous frame's value available at the present frame?**
  No. "The beam modulation method requires two adjacent background frames ... the spectrum analysis process cannot be started until the beam pulse is turned off again ... at least a 1-s time delay" [V: Lee 2020, p. 793-794]; "a delay of up to 1 second in KSTAR, thereby precluding real-time analysis" [V: Lee 2026, section 5]. The manuscript says so in one sentence: the estimators "are causal in their use of time stamps, not in the analysis that produced the values they read" [V: `main_ppcf.tex`:734-738].
  - **Why 2. How old would the newest value be?**
    For the valued rows that have an absent frame earlier in their segment, the newest value such an analysis could have delivered is at most 100 ms old in 7.1 % of the rows, 0.1 to 0.3 s old in 16.6 %, 0.3 to 1 s in 49.8 % and older than 1 s in 26.6 %; the median is 0.65 s. A further 31.5 % of all valued rows have no absent frame before them at all [V: C12].
    - **Why 3. What do the estimators do at that age?**
      The authors measured it. With the last value 50 to 100 ms old the network's reduction of the MSE over the causal GP is -5.9 % [-17.1, +2.5]; at 100 to 300 ms -6.9 % [-13.0, -1.8]; at 0.3 to 1 s -25.5 % [-42.9, -10.2]. Between 100 ms and 1 s the RMSE is 312 to 350 eV for the network, 302 to 313 eV for the causal GP and 343 to 361 eV for hold, against 176 to 180 eV for the network with updates [V: `b71_scenarios.md`:27-43, 82-87]. Weighting the MSE of each age bin of that table by the ages above gives about 120 (network), 122 (hold) and 92 (causal GP) x 10^3 eV^2: under the latency of the analysis that made the target, the network equals hold and the causal GP is a quarter lower. [J: arithmetic on two verified tables; it assumes the error depends on the age alone.]
      - **Why 4. So what would make the premise true?**
        A reader of the frame's own spectrum (the neural-network analysis or the two-Gaussian fit). But then, first, the history would be that reader's values and not the offline product, with errors that the neural-network paper says grow beyond 2.2 m; nothing here was run on such values. Second, such a reader delivers the value of frame t once the frame is exposed and read out, so a nowcast of frame t is ahead of the measurement by at most a frame. What one frame is worth is in the paper: the offline GP, which waits for the next value, has a lower error than the network (-0.031 [-0.061, -0.005]) and the two-stage estimator is +0.064 [+0.039, +0.086] above it [V: `b70_d1.md`:44, 50]. Third, the frames for which a nowcast would still be needed, those without a usable value, are the frames on which nothing can be scored; on rows weighted to resemble them the increment is +0.020 [-0.014, +0.049], and on the rows most like them the RMSE is 645 to 717 eV at a median value of 131 to 245 eV [V: `PROJECT_KNOWLEDGE.md`:531, 617].
        - **Why 5. What is assumed and untested?**
          That a CES history with 10 ms latency and the error properties of the offline product will exist at this chord. (a) An assumption nobody tested, and (c) testing it needs the output of a real-time reader for these discharges.

**THE ROOT.** The estimator's main input does not exist in real time for the analysis that made the target, and with a reader that could supply it, the quantity being nowcast is delivered by that reader one frame later.

**What would settle it.**
- [existing data, CPU] Replay with the true availability: at each frame the newest visible value is the last frame before the most recent absent frame. The stored replay machinery of B.71 does this.
- [needs training, GPU] Train under that schedule.
- [needs data that do not exist here: neural-network or two-Gaussian CES values for the same discharges.]

**What may honestly be claimed.** "If the CES value of the previous frame were available within a frame, the estimator would give the reported value of the present frame about one frame early, with an RMSE of 175 eV. With the present KSTAR analysis, whose newest value is 0.65 s old at the median, a causal GP on the history has the lower error. No real-time claim is made."

---

## 5. Objection 5: around beam interruptions the plasma probably does move, and it does not help the claim

**Statement.** The fall after a beam interruption is the one event here where I believe something physical happened at the chord. It still cannot be assigned to temperature, the first frame after it is exposed on a returning beam, and the event is scheduled.

- **Why 1. What happens?** First row after an absent frame: T_i 0.819 [0.796, 0.838], BES 0.656, ECEI 0.926, fresh rotation 0.973 to 0.975; T_i and ECEI recover with the same 40 ms [V: `file_rules.md`:75-91; `b62_beam_return.md`:38; `PROJECT_KNOWLEDGE.md`:334].
  - **Why 2. Why is it not the loss of heating?** A 10 ms loss of beam heating lowers the temperature by a few per cent at most by the slowing-down formulas, and the fall is not deeper after two absent frames than after one (-0.112 [-0.238, +0.054]) [V: `main_ppcf.tex`:745-747; `b61_blip_dose.md`:13]. The authors say this themselves.
    - **Why 3. Then what is it?** The group that runs the CES says it: "modulation of the heating beam can lead to large radial displacement of the plasma" [V: Lee 2020, p. 788]; the wall field changes during the interruption and for about 50 ms after it [V: `PROJECT_KNOWLEDGE.md`:771]. A chord fixed in major radius inside an edge gradient reads a lower value when the plasma moves inward by millimetres, with no change of the temperature of any flux surface. [J]
      - **Why 4. Why does this matter for the claim?** The largest increment per row is on rows 1 to 3 after an absent frame (+0.151 [+0.093, +0.215]) [V: `PROJECT_KNOWLEDGE.md`:618]. The exposure of the first such row falls 1 to 6 ms after the beam returns, when the raw BES is at 0.60 to 0.76 of its level [V: `file_rules.md`, section 6]. So in these rows BES tells the estimator how far the beam has come back; only the first-row excess of 0.030 of the level bounds the measurement part [V: `b62_beam_return.md`:37]. And a programmed interruption is known before it happens: a controller needs no BES to know that the beam was just off.
        - **Why 5. What would decide between displacement and temperature?** The position of the plasma boundary at 100 Hz or faster, and the voltage and current of each source. Neither is in the files. (c).

**THE ROOT.** At a chord fixed in space inside a gradient, a fall of the reported value after a beam interruption cannot be assigned to temperature without the position of the plasma, and that is not in the files.

**What would settle it.** [needs data that do not exist here: outer gap or separatrix radius from the real-time equilibrium, beam waveforms per source.] [existing data, CPU] Keep reporting the increment with the ten rows after every absent frame set aside (the authors do: +0.045 for the filter, +0.054 on quiet rows), and read those rows against the following-frames reference as in section 1.

**What may honestly be claimed.** What the manuscript now says: the value falls to 0.82 and recovers with the ECEI level in about 40 ms, and no mechanism is named. Nothing more, and not as support for "BES carries temperature information".

---

## 6. What I checked and did not find

A hostile reading is worth nothing if it hides the checks that came out for the authors.

1. The headline numbers reproduce exactly from the stored estimates (section 0) [V: C4].
2. The gain of the history-only estimators is real against the following frames: two-stage twin over causal GP +0.105 [+0.085, +0.124], two-stage estimator over hold +0.482 [+0.464, +0.499] [V: C7b]. A learned causal smoother of this noisy series is better than a causal GP. That result stands.
3. Where the beam emission falls, what the fast channels add does last (section 1, Why 4): a lasting signal in about 6 % of the frames. Lasting is not the same as plasma, for the reason given there.
4. I expected rows with a repeated rotation value to mark poor T_i fits. They do not: the relative one-frame scatter of T_i is 0.082 with a repeated and 0.091 with a fresh rotation value, and within a discharge the ratio is 1.02 (quartiles 0.80 and 1.26, 142 discharges) [V: C2]. Dropped. What remains odd is the column itself: 54 % of the rotation values repeat the previous one, in runs of up to 1,213 rows, in 175 discharges for more than nine values in ten [V: C1], while T_i from the same fit changes every frame. The rotation column is not the per-frame output of the fit in those discharges, and nobody knows why.
5. The increment is not concentrated in the badly bracketed values: it is largest between two absent frames (Objection 2, settling list) [V: C21].
6. The frame-to-frame scatter does not grow with the time since the last beam-off frame (the authors' B.46, 1.013 [0.982, 1.051]). I accept that slow drift of the background is not what makes the scatter. It says nothing about light that changes within a frame.

---

## 7. Closing

**The single strongest reason a CXRS expert would say "this result is not real".**
In the 29 % of frames where the beam-emission sample is above its recent level, the fast channels buy +0.13 of skill on the frame's own value and -0.01 [-0.05, +0.02] on the value of the next 10 to 30 ms; across all frames the fitted width follows the beam-emission sample with slope 1.3 while the fitted shift of the same line has slope 0.02 [-0.02, +0.05]. A Doppler temperature does not rise for one exposure in step with a microsecond of beam light and vanish before the next exposure, while the Doppler shift of the same line stays where it was. Half of the increment is that, for the network, the two-stage estimator and the filter alike.

**The single strongest reason they would say "it does not matter".**
The estimator needs the CES value of the previous frame, and the analysis that makes these values delivers it 0.65 s late at the median. At that age the authors' own replay gives the causal GP the lower error. With a spectral reader fast enough to supply the history, the value of the present frame arrives one frame later by itself, and the paper shows what one frame of waiting is worth: as much as the network.

**The one analysis I would demand first.**
[existing data, CPU] A registered re-read of every fast-channel statement against the following-frames reference, split by the sign of the BES departure, with the rotation dose table beside it. It costs half a day, it needs nothing new, and it decides whether the paper's central sentence is about the plasma or about the fit. If one thing can be asked of the provider, it is the error bar, the fitted amplitude and the method flag of this one channel.

**A defensible version of the claim.**
1. On withheld frames of the automatic KSTAR CES product at R = 2.198 m, a causal estimator that reads only the CES history (a recurrent network with a boosted second stage) has a lower squared error than a causal GP: +0.09 on the frame's own value and +0.105 [+0.085, +0.124] against the level of the following 10 to 30 ms.
2. A fall of the beam-emission sample below its recent level (6 % of frames below 0.9) announces a fall of the reported value that lasts into the following frames (+0.11 [+0.07, +0.15] over the same estimator without the fast channels); whether the plasma cools, the plasma moves or the fit reads low on a weak beam is not known, and outside the programmed interruptions the rotation does not fall with it.
3. A rise of the beam-emission sample goes with a rise of the fitted width confined to that frame (slope 1.3), which the fitted shift does not share and which is absent 10 to 30 ms later. We report it as a property of the automatic fit at this chord and as a candidate quality flag for it, not as ion temperature.
4. The target is what the automatic analysis writes, including values written with the beam off; the population rule is stated as a ladder, not as one threshold; no real-time claim is made, and the age of the newest deliverable value is stated.

Written that way it is a paper about a diagnostic product and how to smooth and vet it, which is true. Written as it stands it says that fast edge diagnostics see the ion temperature move from frame to frame, and the files say otherwise.

---

## Appendix: numbers of my checks not given above

- Grid: 641 files, 245,925 rows; T_i present on 226,991 (missing 7.70 %), V_rot missing 23.51 % [V: C0, C1]. These match `file_rules.md`.
- T_i quantiles (eV): 1 % 34.6, 5 % 120, 10 % 183, 50 % 592, 90 % 1,139, 99 % 2,089, 99.5 % 3,121, 99.9 % 9,601 [V: C1].
- Single-frame excursions over all triples within 100 eV to 3 keV (200,061): beyond +0.3 in 4.56 %, beyond -0.3 in 4.01 %; beyond +-0.5 in 2.02 % and 2.01 % [V: C3]. Probability of an excursion beyond +-0.3 by the single-frame excursion of the BES sample: central half 5.2 %, lowest hundredth 49.7 % (17.5 % up, 32.2 % down), highest hundredth 43.3 % (30.3 % up, 13.0 % down); rank correlation of the two excursions +0.228, with ECEI +0.202 [V: C3].
- Squared error by class of the target row (share of rows; share of the squared error of hold, causal GP, two-stage estimator): rise beyond +0.3: 0.059; 0.258, 0.321, 0.367. Fall beyond -0.3: 0.050; 0.259, 0.172, 0.113. Between 0.1 and 0.3: 0.249; 0.133, 0.134, 0.123. Within 0.1: 0.590; 0.048, 0.079, 0.086. Neighbours missing: 0.053; 0.302, 0.293, 0.311 [V: C5].
- Share of the squared-error reduction on rows that are an excursion beyond +-0.3 or directly follow one (15.7 % of rows): two-stage over causal GP 0.656 [0.607, 0.723]; two-stage over its twin 0.684 [0.606, 0.790] [V: C5].
- Rows by the BES sample relative to the ten rows before, share of the two-stage estimator's reduction over its twin on the own frame: below 0.4: -0.001; 0.4-0.8: 0.119; 0.8-0.9: 0.149; 0.9-1.1: 0.443; 1.1-1.2: 0.112; above 1.2: 0.181 [V: C23].
- Squared-error reduction per row, two-stage estimator over its twin: 3,179 eV^2 against the own frame, 496 eV^2 against the following frames; network over twin 1,911 and 422; KFX over KF0 1,950 and 874 [V: C26].
- Median spacing of consecutive absent frames 1.48 s (quartiles 0.98 and 1.48; 905 spacings) [V: C11].
- Replay causal GP against scorer causal GP: the squared errors differ by more than 1 % on 273 of 664,326 rows; those rows are left out of section 0 [V: C7b].
- Limits of these checks: thresholds (0.3 for an excursion, 0.9 and 1.03 for the BES bands, 0.4 for beam-off) were set today after seeing the data; r2-r4 only; no network was run; the following-frames reference is still the same automatic fit, so a bias shared by neighbouring frames passes it, exactly as the authors say of their own persistence test.
