# Critique from the pedestal and real-time control side

2026-11-20. Written by a reviewer in the role of a tokamak edge and pedestal physicist who also builds real-time profile control. Asked to find what is wrong at the root, not to list small flaws.

**Read.** `CLAUDE.md`, `PAPER_DESIGN.md`, `PROJECT_KNOWLEDGE.md` sections 1 to 4 (without 4.13) and 7, `docs/paper/file_rules.md`, the result documents b70_d1, b71_scenarios, b64, b65, b66, b41, b31, b61_blip_dose, b62, b59, the manuscript `main_ppcf.tex`, the target and rotation sections of `supp_ppcf.tex`, `literature_map.md` sections 3a to 3e, and one source PDF (Lee 2026, the neural-network CES paper). Not read: `docs/paper/review_loop/`, `PROJECT_KNOWLEDGE.md` 4.13, grading passages.

**Marks.** **[V: …]** is verified: a file and line, or a check I ran with its number. **[J]** is my judgement as a physicist. Literature facts taken from the authors' own map and not re-read in the source are marked **[V-map: …]**.

**Checks.** C1 to C9 are read-only scripts on the 10 ms files, the raw streams of the twelve discharges and the stored D1 estimates of B.70 (`data/.b70/stage2_r*_f*.npz`). Nothing was trained, no tracked file was changed, nothing was committed. Scripts and printed outputs: `docs/paper/review_loop/critique_20261120_pedestal_checks/`. They are a reviewer's readouts made after seeing the results, with a one-level bootstrap over discharges and the networks held fixed. Before any number below is used, the authors should register and repeat it.

## Verdict

1. The fast-channel increment is a gain on the reported value of the one frame that is scored. Against the level of the next five measured frames the paper's estimator gains +0.000, +0.019 and +0.021 over its own history-only twin (r2, r3, r4), where it gains +0.109, +0.111 and +0.112 on the frame itself.
2. What BES knows about a frame is the light collected during that frame's own exposure. Whether the reported width follows that light through the fit, through an edge gradient that moves past fixed sight lines, or through the ion temperature is not decided by anything in the files. The first two are not a statement about the pedestal-top ion temperature.
3. The estimator has a lower error than a Gaussian process on the CES history only while an offline, non-causal CES analysis keeps feeding it values at most 50 ms old. That is the condition under which nobody needs it.
4. "Pedestal top" is the provider's label. The events that define a pedestal are averaged out by the exposure, unlabelled, or in the frames without a value.
5. The increment is measured over the CES history alone, an information set that no control room has.

---

## Objection 1. The gain is on the frame, not on the temperature

**Statement.** The paper's increment exists against the reported value of the scored frame and nowhere after it. For where the reported temperature sits during the following 50 ms, the two-stage estimator has the same error as its twin without fast channels.

**What I measured (C7, C8).** I read the stored D1 estimates of r2, r3 and r4 and scored them against two targets: the reported value of the frame, as the paper does, and the median of the next five measured frames (at least three present, all on the 10 ms grid). The reading path reproduces the recorded skills to three decimals: network over twin +0.046 and +0.050, two-stage over two-stage twin +0.095 and +0.097, two-stage over network +0.092 and +0.095, network over hold +0.377 and +0.378 on r2 and r3 **[V: C7, against `b70_d1.md:10-11`, `:35`, `:49`]**. Rows with a following level: 211,906 of 222,672 in 613 discharges.

| target | network over twin (r2, r3, r4) | two-stage estimator over two-stage twin (r2, r3, r4) |
|---|---|---|
| reported value of the frame | +0.058 [+0.034, +0.081], +0.061 [+0.043, +0.077], +0.069 [+0.049, +0.090] | +0.109 [+0.088, +0.129], +0.111 [+0.093, +0.129], +0.112 [+0.092, +0.133] |
| median of the next five measured frames | +0.006 [-0.027, +0.033], +0.021 [-0.004, +0.043], +0.026 [-0.005, +0.051] | +0.000 [-0.021, +0.021], +0.019 [+0.001, +0.037], +0.021 [+0.003, +0.038] |
| the same, correction multiplied by a factor fitted on the other half of the discharges (0.48 to 0.58) | +0.062 [+0.050, +0.076], +0.064 [+0.052, +0.076], +0.070 [+0.055, +0.082] | +0.058 [+0.049, +0.068], +0.064 [+0.055, +0.073], +0.068 [+0.058, +0.076] |
| median of the five frames before and the five after (r2, r3 only) | -0.255 [-0.345, -0.176], -0.185 [-0.257, -0.123] | -0.283 [-0.347, -0.224], -0.212 [-0.261, -0.164] |

**[V: C8 for rows 1 to 3, C7 for row 4.]**

Error against the level of the next five frames, r2 (rows that also have five-frame neighbours on both sides, 203,227 rows) **[V: C7]**:

| estimate | RMSE (eV) | median absolute error (eV) |
|---|---|---|
| hold | 183.5 | 44.9 |
| median of the last five measured values | 146.0 | 38.6 |
| twin | 132.8 | 39.9 |
| network | 132.6 | 44.6 |
| two-stage twin | 128.3 | 38.4 |
| two-stage estimator | 128.3 | 39.9 |

By thirds of the discharges, two-stage over two-stage twin against the following level (r2, r3, r4): coldest -0.050 [-0.084, -0.020], -0.017 [-0.048, +0.011], -0.029 [-0.063, +0.002]; hottest +0.003 [-0.035, +0.035], +0.025 [-0.001, +0.052], +0.033 [+0.007, +0.058], where the same pair reads +0.166, +0.175 and +0.187 on the frame **[V: C8]**.

**Why-chain.**

- Why does a correction of which "most lasts into the next frame" give no gain for the following level?
  - Because a correction lowers a squared error only when its regression coefficient on the target is above one half. For the correction "network minus twin" the coefficient is 0.686 [0.606, 0.769] on the frame, 0.554 on the next frame and 0.506 [0.446, 0.564] on the level of the next five (r2; r3: 0.719, 0.578, 0.539). The paper's ratio 0.81 is 0.554 / 0.686. The gain goes as twice the coefficient minus one: 0.37 on the frame, 0.11 on the next frame, 0.01 on the following level **[V: C7]**. For the two-stage pair the coefficients are 0.893, 0.601 and 0.501 (r2): the second stage calibrates the correction to the frame and leaves it twice too large for what follows.
  - Why is the correction twice too large for what follows?
    - Because it was trained and scored on the frame's own reported value, and that value has a part that is tied to the fast-channel values of the same frame and is gone in the next frames. The departure of a scored frame from the median of its ten neighbours has a robust standard deviation of 50.3 eV and an r.m.s. of 157.4 eV; the r.m.s. error of the network on the same rows is 158.1 eV, so the sum of squared departures is 0.99 of the network's squared error and 1.10 of the two-stage estimator's **[V: C7]**. Frames more than 30 % below their neighbours (4.4 % of the rows away from blips) have a BES level more than 10 % below its neighbours in 16.5 % of cases, against 1.0 % for frames within 10 % **[V: C4-B]**.
    - Why is there a part that belongs to one frame only?
      - Three things can make it: an ELM-like event or a filament inside the exposure, a steep gradient that jitters past a fixed major radius, or an excursion of the fit in a frame with less light (Objection 2). An ion temperature on a flux surface is not among them: it changes on the local energy confinement time, tens of milliseconds, except at an ELM crash **[J]**. The paper's own readouts do not place the increment at bursts (Objection 4).
      - Why was this not seen?
        - The estimand "reported value of the present frame" was taken because it is the only label in the files, and its use was assumed. The persistence readout (`main_ppcf.tex:692-710`) was read as a ratio of coefficients and not as a squared error against a later target. In the typical frame every estimator already sits at the scatter of the label: median absolute error 41.2 eV for the two-stage estimator, 44.1 eV for the causal GP, 45.4 eV for hold **[V: `b70_d1.md:20-24`]**, against a frame-to-frame independent part of 39.1 to 42.8 eV **[V: `PROJECT_KNOWLEDGE.md:390`]**. So the whole gain had to come from the tail, and the tail is the single-frame departures.
        - The assumption nobody tested: that a lower squared error on the reported value of a frame means a better estimate of the ion temperature for anything that acts later than that frame.

- **THE ROOT.** The quantity that is estimated and scored, the reported value of the one frame, is not the quantity of use, the level of the ion temperature over the time in which anything can respond, and the gain does not carry from the first to the second.
- **What would settle it.** [existing data, CPU] Score every stored estimate (network, twin, both second stages, KFX, KF0, causal GP) against the level of the following 30, 50 and 100 ms, and fit the size of the fast-channel correction for that target on the training discharges of each fold. My read-only version gives +0.01 as built and +0.06 with the correction about halved. [needs training, GPU] Train the first stage on a level target (the mean or median of the next k measured frames, or a robust loss) and measure the increment there.
- **Honest claim if it stays as it is.** "The BES value of the present frame predicts part of what the fit of this one frame returns. As built, it does not lower the error of the estimate of where the reported temperature sits over the following 50 ms; a correction of about half the size lowers it by about 0.06 in skill."

---

## Objection 2. What the fast channels know about the frame is its light, and light is not ion temperature

**Statement.** The one physical sentence of the paper, that the BES value of the present frame carries information about the reported T_i, is correct as statistics and empty as pedestal physics. The reported C VI width follows the beam-emission light of its own exposure. Nothing in the files says whether that is the fit, the edge moving past fixed sight lines, or the temperature.

**What I measured.**

- The increment is one bivariate coupling. In 204,153 frame pairs away from blips (635 discharges) the change of ln T_i has a slope of +0.474 [+0.390, +0.563] on the change of ln BES of the same frame, rank correlation +0.239 **[V: C1]**. By thirds of the discharges the rank correlation is 0.105, 0.227 and 0.329; its square, 0.011, 0.052 and 0.108, tracks the pooled increments of the paper, +0.007, +0.042 and +0.092 **[V: C1b; `PROJECT_KNOWLEDGE.md:611`]**.
- It is the light of the frame's own exposure. On the raw streams (twelve discharges, 5,505 quiet frame pairs) the BES level averaged from 2 ms before the stamp to 8 ms after it has a rank correlation of +0.265 with the T_i change (positive in 12 of 12 discharges) and a slope of +0.90. In a joint regression the own-exposure mean takes the whole coefficient (+0.063 per standard deviation) and the snapshot none (-0.001) **[V: C2]**. The snapshot that the estimator reads is a noisy stand-in for the photon budget of the frame.
- The beam restart is a natural experiment. After 1,231 notches of one absent frame (479 discharges) the first frame has BES at 0.658 of its level, T_i at 0.814, ECEI at 0.929, rotation at 0.972 **[V: C4-A; the authors' values are 0.814, 0.658, 0.929 and 0.973, `b61_blip_dose.md:20-22`, `b62_beam_return.md:38`]**. The logarithmic slope from the two medians is 0.49, the same as in ordinary rows (0.47). The spread is wide: the T_i ratio is below 0.5 in 16.7 % of the events and below 0.3 in 6.8 % (control rows: 2.2 % below 0.5); its 5th, 25th and 75th percentiles are 0.225, 0.607 and 0.969 **[V: C4-A]**. In discharge 31921 at 5.012 s the reported value goes from about 420 eV to 74.8 eV in the first frame and returns over 80 ms while ECEI changes by 5 % **[V: `data/s31921.csv`, printed in the checks folder]**.
- A dip of the light takes the value away. Away from blips, T_i is missing or above 3 keV in 45.4 % of the frames whose BES level is more than 15 % below its ten neighbours, in 5.5 % of the frames within 5 %, and in 17.0 % of those more than 15 % above **[V: C4-B]**.
- The KSTAR CES group describes this radius as the place where the measurement degrades: "the active CX signal is reduced due to the increased passive emission of thermal neutrals", spectra are "more easily distorted by transient events such as ELMs and by discrepancies between beam-on and beam-off phases", "low active-to-passive signal ratio" at R of about 2.2 m and beyond **[V: Lee 2026, pp. 4-5, text search in the PDF]**. The background of every frame is interpolated between the preceding and the succeeding beam-off frame **[V: same PDF, p. 1]**, and in these files the absent frames are 1.470 s apart in the median **[V: `PROJECT_KNOWLEDGE.md:463`]**.
- Against the fit: the coupling is not steeper where there is less light. Within a discharge, the slope in the lowest fifth of the light level minus that in the highest is -0.111 [-0.234, +0.036], and the rank correlation is 0.20 to 0.27 in every fifth **[V: C9]**.
- For a moving profile: away from blips the BES change is tilted, the outer column changing more than the inner (slope of the radial difference on the common mode +0.233 [+0.210, +0.257], against +0.061 at beam restarts, where the change is the beam's) **[V: C5]**. On 50 to 200 ms the slope of T_i on BES is 1.2 to 1.3 **[V: C3]**. A displacement at blips is published for KSTAR **[V: `main_ppcf.tex:747-748`]**.
- ECEI, which uses neither beam nor spectrometer, follows too. The single-sample ECEI level of the frame has a rank correlation of +0.229 with the T_i change (192,907 pairs, 568 discharges, positive in 86 % of the discharges, each of the four channels alone +0.195 to +0.210), and +0.186 with BES partialled out **[V: C6]**. The paper states that a network with ECEI and the history has no increment (`b59_inputs.md:13`). That is a statement about one network's squared error, not about the information in the channel.
- Rotation does not follow, at any time scale. On the same 68,358 frame pairs the slope of ln T_i on ln BES is +0.520 [+0.378, +0.652] and that of ln V_rot is +0.006 [-0.012, +0.022]. On 100 ms blocks the two slopes are 1.31 and 0.14 [0.06, 0.24]. The changes of T_i and of V_rot are uncorrelated from 10 to 500 ms (rank correlation at most 0.035) **[V: C1b, C3]**.

**Why-chain.**

- Why does the BES sample of the same frame predict the reported T_i?
  - Because the reported width follows the beam-emission light collected during its own exposure, and the sample is a noisy measure of that light **[V: C2]**.
  - Why would a temperature follow the light of its own exposure?
    - Heating is excluded: the response is in the same frame, a network given BES one frame old has no increment **[V: `b59_inputs.md:9`]**, and the authors' own estimate of the thermal response to a 10 ms loss of beam is a few per cent **[V: `main_ppcf.tex:745-747`]**. Two candidates remain: (a) the fit of a weak active line on a passive background that is refreshed once in 1.5 s, and (b) an edge gradient that moves or breathes past sight lines fixed in major radius, which changes local density, ECE level and T_i together.
    - Why can (a) not be dismissed?
      - Because at a beam restart the light is known to be low for a reason that is not the plasma, ECEI says the plasma changed by 7 %, and the reported T_i is 19 % lower in the median and less than half in one event of six, with the logarithmic slope of the ordinary rows. The authors' answer, that the deficit of rows 1 to 8 has the shape of the ECEI deficit (`PROJECT_KNOWLEDGE.md:653`), describes the 80 ms recovery; it does not explain why the first-frame slope equals the ordinary-row slope, and their own bound leaves up to a third of the first-row drop proportional to BES (`PROJECT_KNOWLEDGE.md:654`).
      - Why can (b) not be dismissed either?
        - Because the BES change is tilted, ECEI follows, and the slope exceeds one on 50 to 200 ms, as a moving profile gives and a fit does not; and the coupling does not grow at low light. But then the rotation profile has to be flat at the chord, since the slope of rotation on BES is a tenth of that of T_i on 100 ms and at most a twentieth on 10 ms **[V: C1b, C3]**, and the "information" is the position of a gradient at a fixed major radius.
        - Why does it matter which of the two it is?
          - Because in neither case is the increment information about the ion temperature at the pedestal top. In (a) the estimator reproduces an error of the measurement. In (b) it reproduces a radial position that real-time magnetics give at kilohertz rates, and a controller wants the temperature on a flux surface or the pedestal height, not the reading at a fixed major radius in a gradient **[J]**.
          - The assumption nobody tested: that the frame-to-frame changes of the reported value that go with BES are changes of the ion temperature on a flux surface. The paper's persistence test cannot test it, as its own text says (`main_ppcf.tex:706-710`).

- **THE ROOT.** No measurement in the files separates a change of ion temperature on a flux surface from a dependence of the fit on the light of the exposure and from a gradient moving past a fixed chord, and the estimator's increment is made of whatever the mixture is.
- **What would settle it.** [needs data that do not exist here] Three things from the KSTAR database for the same frames: the fitted line amplitude and error bar of the chord (the error-bar nodes exist, `literature_map.md:536`); the two neighbouring chords at 2.192 and 2.203 m (`supp_ppcf.tex:173-175`); the equilibrium outer gap or the normalized flux of R = 2.198 m against time. With them: under (a) the coupling vanishes at fixed fitted amplitude; under (b) the three chords move as one shifted profile and the coupling vanishes at fixed normalized flux; what is left is temperature. Beam voltage and current per source would close the restart question. [existing data, CPU] A five-parameter filter with a properly normalized ECEI drive instead of the BES drive, to turn the beam caveat into a number: a beam-free increment.
- **Honest claim if it cannot be settled.** "At R = 2.198 m the reported C VI temperature co-varies, within the frame, with the beam-emission light of its own exposure and with the ECEI level; the reported rotation does not. Whether this is the fit, a displacement of the edge profile or the ion temperature is not determined."

---

## Objection 3. The estimator works only where nobody needs it

**Statement.** The stated motive is real-time use when the CES value is late or absent. The estimator has a lower error than a Gaussian process on the CES history only while it is fed a value at most 50 ms old, and the values it is fed come from an analysis that cannot run in real time.

**Facts.**

- With updates stopped, the MSE reduction of the network against the causal GP is +19.5 [+3.5, +36.8] % up to 10 ms, +13.5 [+5.1, +21.5] % from 10 to 50 ms, -5.9 [-17.1, +2.5] % from 50 to 100 ms, -6.9 [-13.0, -1.8] % from 100 to 300 ms and -25.5 [-42.9, -10.2] % from 300 to 1000 ms. Against its own twin it is +4.7 [-2.7, +9.2] % from 50 to 100 ms. The mean error of the free run is -60.7, -123.6 and -175.1 eV in the last three bins, and +36.3, +126.2 and +199.9 eV when the estimate is fed back **[V: `b71_scenarios.md:82-87`, `:52-54`]**.
- The five-parameter filter, run free on its BES drive, has an MSE of 433.90 against 100.05 (10^3 eV^2) for the filter without drive, beyond 1 s **[V: `b71_scenarios.md:32`]**.
- The T_i history of the training rows is 10 ms old in the median and 130 to 170 ms old at the 99th percentile **[V: `b71_scenarios.md:128-144`]**; 91.6 % of the gaps in the files are 50 ms or shorter **[V: `PROJECT_KNOWLEDGE.md:80`]**.
- The conventional KSTAR analysis "should wait for a beam-off phase to acquire a background spectrum, a delay of up to 1 second in KSTAR, thereby precluding real-time analysis", and the background is interpolated between the preceding and the succeeding beam-off frame **[V: Lee 2026, pp. 1 and 5]**. The manuscript says the same in one clause: the estimators are "causal in their use of time stamps, not in the analysis that produced the values they read" **[V: `main_ppcf.tex:735-738`]**.
- The one KSTAR analysis that could run in real time loses accuracy at R of 2.2 m and beyond **[V: Lee 2026, p. 4]**. No real-time use of CES or BES in the KSTAR control system is published **[V: `supp_ppcf.tex:200`]**.
- The estimate moves by 46.1 eV r.m.s. when the BES sample is taken 0.5 ms earlier **[V: `PROJECT_KNOWLEDGE.md:633`]**, the size of its median error.
- The ITER requirement the paper cites for the edge ion temperature is 100 ms and 10 % **[V-map: `literature_map.md:592-594`]**.

**Why-chain.**

- Why is the margin over the causal GP gone when the last CES value is 50 ms old?
  - Because the fast channels give a differential cue for the present frame and nothing absolute: without a CES value the mean error of the estimate is -61, -124 and -175 eV in the bins that start at 0.1, 0.3 and 1 s, and the filter's drive, integrated without updates, takes its squared error to four times that of the filter without drive **[V: `b71_scenarios.md:32`, `:52-54`]**.
  - Why does the cue not integrate?
    - Because the part of it that lowers the error belongs to the frame (Objection 1), and the slow changes of the BES level are not changes of T_i: the BES level drifts with the delivered beam power by 20 % in 9 s at steady density **[V-map: `literature_map.md:644`]**.
    - Why was the estimator never trained without a fresh value?
      - Because the files have almost no long gaps.
      - Why do the files have no long gaps, when the conventional analysis is up to a second late?
        - Because the files hold the offline result, in which every frame was filled afterwards with a background taken from a later beam-off frame. The gap that exists in real time, the wait for the background, is not in the data at all. The "CES history" of the estimator is therefore not available at the time the estimate is formed.
        - So there are two cases in operation. Either a modulation-free analysis delivers frames within two frames, and then the history has the error statistics of that analysis at the radius where it is weakest, not those the estimator was trained on, and a 100 ms requirement is served by the delivered values. Or the conventional analysis is used, the history is 0.1 to 1 s old, and the estimator has a higher error than the GP. Neither case was tested, and only new data can test the first.

- **THE ROOT.** The estimator is causal in time stamps only: it lives on a history produced by a non-causal analysis, and it loses to the CES-only reference in exactly the regime, more than 50 to 100 ms without a value, that defines the need.
- **What would settle it.** [needs training, GPU] Train the network and the twin with the CES history masked for random spans of 0.1 to 2 s and read B.71 again. [needs data that do not exist here] The values of a real-time capable analysis (the neural-network or two-Gaussian CES) for the same frames, to be used as the history; the latency of CES read-out and of BES streaming at KSTAR.
- **Honest claim if it cannot be settled.** "In an offline replay, a BES sample of the present frame lowers the error on the offline CES value of that frame when the offline values of the last five frames are given. No real-time use is shown or implied, and for gaps beyond 50 ms the CES history alone has the lower error."

---

## Objection 4. "Pedestal top" is a label; what makes a pedestal is averaged out, unlabelled or missing

**Statement.** Nothing verifies where the chord is relative to the pedestal, and the gain is not located at anything a pedestal does on 10 ms.

**Facts.**

- The chord was selected by the provider. Published positions come from other discharges and coordinates: separatrix near 2.24 m in one 2022 discharge, pedestal tops between 2.19 and 2.23 m, normalized radius 0.92 to 0.97 against a nominal 0.80; no equilibrium is in the files **[V: `supp_ppcf.tex:131-151`, `PROJECT_KNOWLEDGE.md:82`, `:85`]**. The BES array is movable from discharge to discharge and the ECEI column is uncertain by 2 to 3 cm between discharges **[V: `supp_ppcf.tex:162-168`]**.
- The discharge medians of T_i run from 213 to 1,020 eV between the 10th and the 90th percentile **[V: `PROJECT_KNOWLEDGE.md:81`]**. Of 639 discharges, 251 show no edge burst, 201 few and 159 frequent ones **[V: `b66_regimes.md:49-51`]**. There is no confinement-mode label.
- ELM periods at KSTAR are 40 to 70 ms and the T_i gradient recovers 2 to 4 ms after a crash at ASDEX Upgrade **[V-map: `literature_map.md:647`]**. A 10 ms exposure mixes crash and recovery. In the raw streams the bursts come at 12 to 17 per second over the delivered windows of four discharges **[V: `b31_bursts.md:12-17`]**.
- Where the edge bursts, the label fails. In rows within 10 ms after a burst, T_i is missing in 0.37, 0.24, 0.37 and 0.27 of the cases in four discharges, against 0.00 to 0.01 in quiet rows. In rows with at least three bursts in the 200 ms before, the median step of the reported value from frame to frame is 322.7 eV, against 28.4 eV in quiet rows, in 31359, and 231.5 eV against 32.7 eV in 31923 **[V: `b31_bursts.md:12-17`]**. No ion temperature steps by 230 to 320 eV every 10 ms on a level of several hundred eV **[J]**.
- The increment is not at the bursts. Frequent-burst discharges: +0.025 [-0.009, +0.052]. Burst phase: +0.029 [-0.027, +0.077] for the network and +0.014 [-0.013, +0.040] for the filter **[V: `b66_regimes.md:40`, `:53`]**. The reported T_i after a burst onset: 1.007 [0.947, 1.056] of its level before **[V: `b41_events.md:31`]**. Coldest third of the discharges: +0.007 [-0.023, +0.031] **[V: `PROJECT_KNOWLEDGE.md:611`]**.

**Why-chain.**

- Why is the gain not where the pedestal changes?
  - Because in those frames there is either no value or a value scattered by hundreds of eV, and a 10 ms exposure cannot resolve a 2 to 4 ms recovery.
  - Why does the value fail there?
    - Because ELM light distorts the edge spectrum and the passive background, taken once in about 1.5 s, is wrong for that frame; the KSTAR CES group says so for this radius **[V: Lee 2026, p. 4]**.
    - Why does that undo "pedestal-top nowcasting"?
      - Because between ELMs and transitions the pedestal-top ion temperature is flat within the noise on 10 ms. What is left to estimate in the labelled frames is the scatter of the fit and slow drift, for which the CES history suffices; hence an increment of a few per cent of a squared error that is three quarters scatter **[V: `PROJECT_KNOWLEDGE.md:709`]**.
      - Why is the chord called pedestal top at all?
        - Because the provider said so. Over 639 discharges with moving shape, gap and field, one major radius is the pedestal top in some, the steep gradient in others and the L-mode edge in the rest **[J]**. Only an equilibrium and a mode label per time can say which, and they do not exist here.

- **THE ROOT.** The scoring keeps only the frames in which the standard fit returned a value, and those exclude or blur the events for which a pedestal-top measurement is wanted; where the chord is, and in which regime, is not known for any frame.
- **What would settle it.** [needs data that do not exist here] Equilibrium mapping of R = 2.198 m per time, D-alpha, the fit status of each frame. [existing data, CPU, twelve discharges] Average the reported T_i and each estimate against the time since the last burst of the raw BES (0 to 10, 10 to 20, …, 60 ms) to see whether the label carries an ELM cycle at all.
- **Honest claim if it cannot be settled.** "An edge chord at R = 2.198 m whose normalized radius is not known per discharge (0.92 to 0.97 in published discharges). The statements hold for frames in which the standard analysis returned a value below 3 keV. Nothing is stated about ELM-resolved or transition dynamics." The title already says "edge chord"; the abstract, the keywords and the conclusion should follow it.

---

## Objection 5. The twin is not what a control room has

**Statement.** The increment is the gain over the CES history alone. Every candidate cause of it is a quantity the control system already holds.

**Facts.**

- The twin reads the time step and the CES history **[V: `main_ppcf.tex:284-296`]**. The files have no beam power, no equilibrium, no D-alpha **[V: `PROJECT_KNOWLEDGE.md:85`]**.
- The KSTAR control system has D-alpha at 20 kHz, interferometer density at 10 kHz and real-time equilibrium **[V: `supp_ppcf.tex:200`]**.
- The largest increment per row is in the three rows after an absent frame, +0.153 [+0.089, +0.220], against +0.049 [+0.025, +0.072] elsewhere **[V: `b59_inputs.md:40`]**. The notches are programmed, 1.470 s apart in the median, in 509 of 641 discharges **[V: `PROJECT_KNOWLEDGE.md:463`]**. For the following level the increment in the ten rows after an absent frame is +0.025 [-0.049, +0.088], +0.032 [-0.039, +0.092] and +0.025 [-0.048, +0.087] **[V: C8]**.

**Why-chain.**

- Why is the increment measured against the CES history only?
  - Because the files contain nothing else.
  - Why does that matter?
    - Because each candidate of Objection 2 is known to the plant: the notch times and the beam waveforms (restart), the outer gap from the real-time equilibrium (a moving profile), D-alpha (ELMs), line density (the BES level follows the local density, `literature_map.md:644`).
    - Why might these remove the increment?
      - If the coupling is the position of the gradient, the gap explains it; if it is the beam, the waveform does; if it is ELMs, D-alpha does. Which, is Objection 2.
      - Why was it not tested?
        - The signals cannot be obtained (`PROJECT_KNOWLEDGE.md:85`). Only new data settle it.

- **THE ROOT.** The increment is over an information set that no control room is limited to, so it is an upper bound on what BES adds in operation, and the bound cannot be tightened here.
- **What would settle it.** [needs data that do not exist here] Outer gap or separatrix radius, line-averaged density, D-alpha and the beam waveforms of the 639 discharges; then the twin reads them and the increment is measured again.
- **Honest claim if it cannot be settled.** "The increment is over the CES history alone."

---

## The four answers asked for

Every number in this section is one of the tagged numbers above (C4-A, C7, C8, `b71_scenarios.md`); the readings are my judgement.

**The strongest reason to say "this result is not real".** All of the claimed squared-error gain sits in the part of the reported value that belongs to one frame and is gone in the next five. An ion temperature at the pedestal top does not do that. A fit in a frame with less light does, and so does a gradient that jitters past a fixed chord; and at a beam restart, where the plasma is the same within 7 % by ECEI and the light is at two thirds, the reported value is 19 % lower with the same logarithmic slope that the "information" has everywhere else.

**The strongest reason to say "it does not matter".** For where the temperature sits over the next 50 ms the paper's estimator has the same error as its history-only twin (128.3 eV against 128.3 eV on r2; skill +0.000, +0.019, +0.021), a median error no lower than the median of the last five measured values (39.9 eV against 38.6 eV), and once the last CES value is more than 50 ms old a higher error than a Gaussian process on the history. There is no target later than the frame and no gap longer than 50 ms for which a controller is better off.

**The one analysis I would demand first.** Score every stored estimate against the level of the following 30, 50 and 100 ms, with the size of the fast-channel correction fitted for that target on the training discharges, for the network, the two-stage estimator, the filter pair and the causal GP, by row class and by thirds. It is CPU work of a day on files that exist, it decides whether there is anything here for a controller, and its number belongs in the abstract beside the present one. My read-only version: +0.01 as built, +0.06 with the correction halved.

**What a defensible and useful version looks like.**

1. The object is "the C VI temperature reported at R = 2.198 m", and the claim is about that reading. "Pedestal top", "nowcasting for control" and "real-time" leave the title, abstract and keywords.
2. It is written as a paper about a measurement and what it co-varies with: the frame-to-frame scatter and its floor; the missingness that follows the light (45 % against 5.5 %); the 80 ms low reading after each beam notch, with its wide spread, which anyone who uses edge CES values near a notch needs to know; the coupling of the width to the light of the own exposure and to ECEI, with rotation not taking part; the three explanations and the three data sets that would separate them.
3. The estimator is the smallest thing that does the job: a robust filter on the CES history, with a BES or ECEI drive whose size is fitted for the following level. The recurrent network and the second stage are a supplement, since for the following level the two-stage twin already has the error of the full estimator.
4. The product a control group can use today is the one the authors already measured and buried: the fast channels predict that the next CES value will be unusable with an area under the curve of 0.961 to 0.974 (`PROJECT_KNOWLEDGE.md:362`). A validity flag for an edge CES channel is a real-time need; a 3 eV change of a median error is not.
5. For a control claim the work needs what it does not have: training with outages of 0.1 to 2 s, a history from an analysis that can run in real time, references that read what the plant has, and a target defined on a flux surface.

---

## Suspicions I checked and withdrew

These would have strengthened my case and the data did not support them. They should not be pursued.

1. "The radial difference of BES leaks the beam through different background fractions." With the beam off the columns keep 0.110, 0.122 and 0.151 of their level **[V: C2-b, 41 notches]**, which predicts +0.024 for the change of the radial difference at a restart; the observed change is -0.025 **[V: C5]**.
2. "The CES history alone already meets 100 ms and 10 %." It does not: the mean of the previous 100 ms is within 10 % of the mean of the next 100 ms in 61.6 % of 18,267 block pairs **[V: C4-C]**. The reported value moves on 100 ms.
3. "Rotation is an internal control that separates fit from plasma." It is not: rotation follows BES at no time scale, including 50 to 200 ms where T_i follows with a slope above one **[V: C3]**. It only says that a displacement needs a flat rotation profile at the chord.
4. "After a restart the fit is captured by the cold passive line." Not typically: 3.7 % of the restarts with a level of at least 400 eV fall below 150 eV (control 0.66 %) **[V: C4-A]**.
5. "The coupling is steeper at low light, as a signal-dependent fit would make it." It is not: lowest minus highest fifth of the light level -0.111 [-0.234, +0.036] **[V: C9]**.

## Limits of these checks

- They were chosen after reading the results and are not registered.
- Intervals resample discharges within one re-partition with the trained networks fixed; the paper's two-level interval is wider.
- The "following level" is the median of reported values and carries their noise. That noise is independent of the estimates, so it dilutes a skill and does not create or remove one; the regression coefficients are not affected by it.
- C7 and C8 read r2 to r4, the re-partitions on which the second stage exists.
- "Quiet" rows are 10 ms steps more than 10 rows after any absent frame; the paper's row classes are not reproduced.

## Appendix: the checks

Folder `docs/paper/review_loop/critique_20261120_pedestal_checks/`. Run from the repository root; `load_all.py data all.npz` first.

| check | script | what it reads | what it gives |
|---|---|---|---|
| C1, C1b | `check1_width_only.py`, `check1b.py` | 10 ms files | slopes of T_i and V_rot on the same-frame BES level; thirds; lag structure |
| C2 | `check2_raw.py` | raw BES of twelve discharges | T_i against the BES level of its own exposure; beam-off residual per column |
| C3 | `check3_scales.py` | 10 ms files | the same slopes on block means of 10 ms to 1 s |
| C4 | `check4.py` | 10 ms files | A beam restart; B departures from neighbours and missingness; C the 100 ms scale; F rows with BES near its background |
| C5 | `check5.py` | 10 ms files | uniform or tilted BES change; regression of T_i on common mode and tilt |
| C6 | `check6.py` | 10 ms files | ECEI level against T_i change, with BES partialled out |
| C7 | `check7.py` | stored D1 estimates of B.70 | skills against the frame value, the two-sided level and the following level; regression coefficients of the correction |
| C8 | `check8.py` | stored D1 estimates of B.70 | the following-level readout by row class and thirds, and with a fitted factor on the correction |
| C9 | `check9.py` | 10 ms files | the coupling by fifths of the light level within a discharge |

Outputs: `outputs_checks_1_2_3.txt`, `check4.out` to `check9.out`.
