# Referee critique, 2026-11-20

Manuscript: "Causal nowcasting of the charge-exchange ion temperature on an edge chord of KSTAR with co-located fast diagnostics" (`docs/paper/ppcf/main_ppcf.tex`, 18 pages; `supp_ppcf.tex`, 49 pages), read together with the later records it has not yet absorbed (B.64, B.65, B.67, B.70, B.71).

Role taken: a referee for Plasma Physics and Controlled Fusion who rejects most "machine learning predicts diagnostic X" papers. No praise, no balance for its own sake.

**What I read.** Main text and generated tables (`b39_generated.tex`); the supplement sections on the target, timing, second stage, filter, split by time, stress tests and registration history; `docs/paper/b70_d1.md`, `b71_scenarios.md`, `b65_inclusive.md`, `b64_two_stage.md`, `b67_gated.md`; `PAPER_DESIGN.md`; `PROJECT_KNOWLEDGE.md` sections 0 to 3, 4 (without 4.13), 5 to 8; `docs/paper/dataset_ledger.md`; `docs/paper/literature_map.md` 2.2, 2.5, 3a to 3e; the PDFs of Lee 2026 (NN-CES), Lee 2020 (two-Gaussian fit), Chrystal 2026 (DIII-D real-time CER), Jalalvand 2025 (Diag2Diag) and Kim 2026 (outlier-robust profile fitting), by text extraction.

**What I did not read** (independence): `PROJECT_KNOWLEDGE.md` 4.13, the other files of `docs/paper/review_loop/`, `THESIS_RESULTS.md`.

**What I ran.** Six read-only checks, R1 to R6, on the 10 ms files and on the stored D1 row files (appendix A). Nothing was trained, no tracked file was edited, nothing was committed.

**Tags.** `[V: source]` = verified by me at that file and line, page, or check. `[J]` = my judgement. Abbreviations: MS = `main_ppcf.tex`, SUPP = `supp_ppcf.tex`, GEN = `b39_generated.tex`, PK = `PROJECT_KNOWLEDGE.md`, PD = `PAPER_DESIGN.md`, LEDGER = `dataset_ledger.md`, LIT = `literature_map.md`, B70/B71/B65/B67 = the files of `docs/paper/` named above.

---

## 1. What the paper claims, and what it would have to show

The paper claims that a causal estimator which reads the beam-emission (BES) sample of the present frame predicts the C VI temperature that the KSTAR CES analysis reports for that frame, at one edge chord, with a lower mean squared error than the same estimator without the fast channels (on one data definition: 17.7 % below a causal Gaussian process, 9.7 % below its history-only twin, values above 3 keV removed `[V: B70:9-10]`), and it reads this as BES carrying information about the ion temperature beyond the CES history.

To recommend it I would need three things: evidence that what is gained is temperature and not the reading of one frame (an independent truth, or at the least a gain that is still there in later frames); a mechanism or a use; and one frozen estimator confirmed once on discharges that had no part in its design.

It has none of the three.

---

## 2. Objections, most serious first

### Objection 1. The gain is about the reading of the frame, not about the temperature: three quarters of the fast-channel increment is gone one frame later

- **Statement.** The increment is scored against the reported value of the same frame whose BES sample the estimator reads. I scored the same stored estimates against the reported value of the *next* frame, an independent reading of the same slowly varying quantity that no estimator has seen. On the 200,740 D1 rows per re-partition (r2 to r4) that have the previous frame and three following frames 10 ms apart `[V: R3]`:

  | scored against | two-stage over two-stage twin | network over twin | two-stage twin over twin (CES history only) |
  |---|---|---|---|
  | reported value of the same frame | +0.136 [+0.113, +0.159] | +0.070 [+0.044, +0.094] | +0.046 [+0.037, +0.057] |
  | reported value of the next frame | +0.032 [+0.016, +0.048] | +0.017 [−0.004, +0.037] | +0.040 [+0.032, +0.049] |
  | mean of the next three frames | +0.038 [+0.017, +0.060] | +0.032 [+0.003, +0.058] | +0.068 [+0.058, +0.080] |

  What the estimator learns from the CES history keeps its whole gain in later frames and gains more against the less noisy three-frame mean, as an estimate of a state should. What it learns from the fast channels keeps a quarter (0.24 for both estimators against the next frame). Under a Huber loss the picture is the same: +0.132 → +0.029 [+0.015, +0.043] for the two-stage estimator, +0.066 → +0.010 [−0.011, +0.030] for the network `[V: R3]`. In the median, reading the fast channels does not help against later frames: 37.3 eV against 36.9 eV for the two-stage pair, 40.6 eV against 38.7 eV for the network and its twin (mean of the next three frames) `[V: R3]`.
  - **Why is that so?** Because the reported value is mostly scatter that belongs to one frame. The floor is 137 eV root mean square `[V: R1; GEN:139 gives 135.7]`; on consecutive triples twice the floor is 94 % of the squared error of hold `[V: R1]`; the lag-1 autocorrelation of the increments is −0.47 pooled and −0.416 at the median discharge, against −0.5 for white noise on a flat signal `[V: R1; PK:83]`. An estimator can lower its squared error against the value of frame t in two ways: by knowing the persistent level better, or by anticipating the part that belongs to frame t alone. The CES history can only do the first. A BES sample taken during the exposure can do both.
    - **Why did the authors not see it?** They ran the right test and read it in the wrong unit. Their persistence table gives a regression slope β₀ = 0.693 for the network's fast-channel correction on its own frame and a ratio ρ₁ = 0.810 one frame later `[V: GEN:388]`. The abstract and the conclusion turn this into "most of the network's correction is still present in the next, independently fitted frame" `[V: MS:81-82, 699-701, 803-804]`, with one clause in the body that ρ₁ "is not the share of the error removed" `[V: MS:702-703]`. The share of the error removed is what the paper is about. Adding a correction d with slope β removes (2β − 1)·E[d²] of squared error. With β₀ = 0.693 and β₁ = 0.810 × 0.693 = 0.561 the part of the gain that survives is (2 × 0.561 − 1)/(2 × 0.693 − 1) = 0.32 `[J: algebra on the verified table]`. R3 measures 0.24 directly `[V: R3]`. A fifth of the correction that does not persist costs two thirds or more of the gain, because a correction that does not persist does not merely stop helping one frame later, it hurts.
      - **Why does it matter that the gain belongs to one frame?** Because the part of the reported value that belongs to one frame is exactly the part that cannot be told from the fit. The KSTAR CES group writes that at R ≳ 2.2 m the active signal is weak against the passive emission, that spectra are "more easily distorted by transient events such as ELMs and by discrepancies between beam-on and beam-off phases", and that "these factors degrade the fitted parameters" `[V: Lee 2026, p.4]`. The chord is at R = 2.198 m `[V: MS:108]`. And no user of an ion temperature wants the scatter of one frame predicted (Objection 3).
        - **Why was the reported value of the same frame made the target at all?** Because the files hold nothing else: no fit error, no spectrum, no second measurement of the ion temperature at the edge `[V: PK:85; LIT:515]`. The paper notes that a constant offset changes no skill `[V: MS:246-248]` and moves on. The assumption underneath every table, that a lower error against the reading of frame t means a better knowledge of the temperature, was never tested `[J]`. The one test that could bear on it, persistence, was reported as a slope ratio.
- **THE ROOT.** The paper scores estimators of a noisy reading against that same reading, and presents as information about the ion temperature a gain of which three quarters is absent from the next independent reading.
- **What would settle it.** `[existing data, CPU]` Reproduce R3 for every estimator and reference (filters, causal GP, hold) and make the following frames the primary target. `[existing data, CPU]` Refit the filter gains and the boosted second stage to that target. `[needs training, GPU]` Retrain the network on it; the persistent part of the information may be larger than what an estimator trained on the same-frame value realises (an optimally rescaled correction would keep about ρ₁² = 0.66 of its same-frame best `[J]`).
- **What may honestly be claimed if it is not settled.** "Scored against the following frames, the fast-channel increment of the two-stage estimator is +0.03 to +0.04; for the network alone it is not separated from zero against the next frame; the median error against later frames is not lowered."

### Objection 2. What the fast channels co-vary with cannot be attributed: plasma, position, or fit

- **Statement.** The only sentence of the paper on physics is that "no mechanism is named" `[V: MS:749-750]`. Three mechanisms give the same sign and the same same-frame co-variation: (A) pedestal dynamics, with density and ion temperature moving together; (B) displacement of a steep edge profile across a fixed chord and fixed BES columns, which is real at fixed R and says nothing about the pedestal-top temperature; (C) the fit, through the ratio of active to passive light. The paper cannot separate them and does not try to state them.
  - **Why can it not?** The files have no beam power, no fit error or spectrum, no equilibrium, no D-alpha, no confinement mode and no neighbouring chord, and the authors record that none can be obtained `[V: PK:85; MS:191-193, 769-776]`.
    - **Why is the fit a live candidate and not a formality?** BES is beam density times plasma density at the chord; the active charge-exchange light is beam density times carbon density at the same place. KSTAR subtracts a background interpolated between beam-off frames about 1 s apart `[V: Lee 2020, p.4; Lee 2026, p.2-3]`. The DIII-D real-time paper explains why no automatic analysis there subtracts at all: the passive emission "can vary on prompt time scales due to changes in the edge plasma conditions" and "low quality subtraction is very detrimental to fit accuracy", and edge inaccuracy is "largely a result of" unmodelled passive emission `[V: Chrystal 2026, p.2, p.4]`. So at an edge chord every fast edge change between two beam-off frames leaves a wrong background in the spectrum, and BES sees that same change. The authors' own check shows the coupling at the extreme: values above 3 keV are 3.18 [2.49, 4.04] times as frequent in the lowest fifth of the BES level `[V: PK:405; SUPP:251-254]`. There is no reason for the coupling to stop at 3 keV.
      - **Why is position a live candidate?** The normalized radius of the chord is 0.80 nominally and 0.94 to 0.97 in two published discharges, not reconciled `[V: PK:82; SUPP:139-151]`. The increment is absent in the coldest third of the discharges and largest in the hottest `[V: GEN:473-474; R2]`, as it would be if a gradient at the chord were needed. After a beam blip the reported value falls by 19 % in one frame `[V: R1, median 0.814; MS table of blips]`; the authors size a heating loss at "a few per cent at most" `[V: MS:745-747]` and find the wall field moving at the blips `[V: PK:771]`.
        - **Why do the tests in the paper not decide?** Persistence cannot: plasma, position and a slowly varying fit bias all last tens of ms, and the paper says a bias common to consecutive frames "is not seen by this test" `[V: MS:708-710]`. The radial-difference filter (two thirds of the increment `[V: MS:564-566]`) excludes a pure beam change and nothing else. The one direct test of pedestal dynamics went against it: no change of the reported value after burst onsets, 1.007 [0.947, 1.056] `[V: PK:379]`. Two registered checks of the noise estimate failed `[V: SUPP:264-267]`. No paper checks the edge CES value of KSTAR against another diagnostic `[V: LIT:515]`.
          - **Why is even the timing uncertain?** The paper places "its 10 ms frame" as starting at the stamp `[V: MS:221-224]`. The CES group states a 5 ms exposure in the 10 ms cycle `[V: Lee 2020, p.4]`, and the supplement admits the published values conflict `[V: SUPP:223-227]`. The files side with a short exposure. Regressing the change of ln T_i of row t on the changes of the BES common mode at stamps t−2 to t+2 gives weights 0.08, 0.22, **0.67 [0.58, 0.77]**, **0.08 [0.02, 0.13]**, 0.15 `[V: R6]`. Under the paper's placement the sample at stamp t+1 closes the same exposure that the sample at stamp t opens, so the two should weigh alike; the difference is +0.59 [+0.49, +0.70] `[V: R6]`. Either the exposure is short and the BES "of the present frame" is a sample from inside it, or the reported value lags the beam emission by 5 ms or more. In the first case the estimate is simultaneous with the measurement, the lead of the "nowcast" is a few ms, and the fit reading gains weight `[J]`. The paper reports no lag structure at all.
- **THE ROOT.** The data delivery lacks the four routine signals that decide between plasma, position and fit (fit error or spectrum, beam waveform, equilibrium, neighbouring chords), and the one thing the files can decide, where the exposure sits relative to the fast sample, was assumed instead of measured.
- **What would settle it.** `[needs data that do not exist here: per-frame fit error bars (the facility database has nodes for them, SUPP:255-258, LIT:536) or the spectra; beam voltage and current waveforms; the equilibrium boundary; the CES chords at 2192 and 2203 mm; D-alpha]`, for the twelve raw-stream discharges at the least. `[existing data, CPU]` On the twelve raw-stream discharges, slide a 1 to 5 ms BES window from −10 to +15 ms around each stamp and find where the association with the reported value peaks and how wide it is; report R6 on all discharges.
- **What may honestly be claimed if it is not settled.** "The reported value co-varies with the BES sample of its own stamp (Δ ln T_i ≈ 0.6 to 0.7 Δ ln BES). Whether this is temperature, displacement or fit is not determined, and no statement about ion-temperature dynamics follows."

### Objection 3. No user: a guess, a few ms early, of a number that arrives anyway, from inputs that do not exist in real time, which fails when CES stops

- **Statement.** The estimator delivers the reported value of a frame before the frame is read out. Nobody is named who needs that.
  - **Why would a reader need the value before it is measured?** The paper's answer is requirement documents: edge ion temperature at 10 to 100 ms and 10 % `[V: MS:122-128]`. ITER asks 100 ms `[V: LIT:592-594]`. A value every 10 ms that arrives one frame late already meets it. Actuators are no faster than about 10 ms, and the KSTAR ECH launcher answers in about 20 ms `[V: LIT:585, 598]`. The DIII-D real-time system reads out 2.5 ms after the mean photon and fits within about 1 ms `[V: Chrystal 2026, p.1-2]`. The lead being offered is at most one frame, and a few ms if the exposure is 5 ms `[J]`.
    - **Why does the estimator need what does not exist?** "Use requires BES of the present frame and a CES value at most two frames late" `[V: MS:91-92]`. No real-time use of BES or CES on KSTAR is published `[V: MS:318-320; SUPP:200]`; the conventional analysis delivers values up to 1 s late `[V: Lee 2020, p.9; Lee 2026, p.5]`. So the estimator presupposes the real-time analysis that the CES group names as its goal, and its history inputs would then be that analysis's values, whose accuracy falls at this radius `[V: Lee 2026, p.4]`, not the offline values it was trained on. The paper says so in one clause: its estimators are "causal in their use of time stamps, not in the analysis that produced the values they read" `[V: MS:737-738]`.
      - **Why not use it where CES is absent, which would be a use?** Because it cannot. With updates stopped, the network is not separated from a Gaussian process on the stale history after 50 ms (−5.9 % [−17.1, +2.5]) and has the higher error after 100 ms (−6.9 % [−13.0, −1.8]; −25.5 % between 0.3 and 1 s), biased low by 61 to 175 eV `[V: B71:84-87, 52-54]`. Three of the five registered statements of that batch failed `[V: B71:13-17]`. In the frames that actually lack a value it is not scored at all, and on scored rows weighted to resemble them the increment is +0.020 [−0.014, +0.049] `[V: GEN:493]`. A filter that integrates the BES drive without updates has, after 1 s, four times the squared error of the same filter without it `[V: B71:32]`: the gain is not a calibration that holds.
        - **Why was this problem chosen?** The stated purpose is to nowcast the pedestal-top CES value without the next measurement, with a demonstration in use counted as a bonus `[V: PK:27]`. The problem was set by the file that was delivered: one chord and fifteen snapshot channels on the CES grid. The decision that the estimate would change was never written down `[J]`. For someone who wants the persistent level, the reading of the present frame itself, when it arrives, moves the uncertainty from about 89 to about 75 eV under a random-walk-plus-white-noise reading of R1 `[J: derived from verified numbers]`; the fast channels move the root-mean-square difference from the mean of the next three frames from 130.4 to 127.9 eV `[V: R3]`.
- **THE ROOT.** An assumption nobody tested: that an estimate of this number, a few ms early, is wanted. The paper has a method and no decision that it serves.
- **What would settle it.** `[needs rewriting only]` Drop the control framing and the requirement documents. Or `[existing data, CPU]` show one use: gap filling scored by hiding measured frames in the pattern of the real gaps (length, BES level) against a two-sided smoother, with the fast-channel part shown there; or flagging of frames whose reading is about to depart.
- **What may honestly be claimed if it is not settled.** "No use is demonstrated. Within 50 ms of the last value the estimator bridges a gap; it does not replace CES."

### Objection 4. The effect is small, sits in a few per cent of the frames, and exists only after a cut on the label

- **Statement.** The headline is a mean squared error on a target whose squared error is carried by a few rows, and the first stage shows nothing under any yardstick of the typical frame.
  - **Why is the score decided by few frames?** In the files, 1 % of the consecutive pairs carry 54 % of the squared step of the reported value and the root mean square is 4.8 times the median `[V: R1]`. On the D1 rows of r2, half of the gain of the two-stage estimator over the causal GP comes from 1.07 % of the rows and half of its fast-channel increment from 1.6 %; the two-stage estimator is closer than its twin in 53.7 % of the rows; the network is closer than its twin in 49.6 % and has the lower median error in 43 % of the discharges; a tenth of the discharges hold 70 % of the increment of the two-stage estimator and 114 % of that of the network, so the other nine tenths lose on balance `[V: R2]`.
    - **Why is the target that heavy-tailed?** Because of single-frame excursions. Upward spikes that revert in the next frame are 1.55 % of the rows (median 824 → 1,340 eV → back); with their returns they carry 48 % of the squared error of hold `[V: R5]`. Half of the network's fast-channel gain (49 %) lies in the 2.1 % of rows with a single-frame spike, a third in the 1.6 % with a step that stays, and 2 % in the 93.8 % of rows without a change above 300 eV `[V: R4]`. The facility's own profile fitting treats such points with a good-and-bad mixture likelihood `[V: Kim 2026, title and abstract]`. The example traces of the paper show reported values alternating between about 50 eV and 2.7 keV within tens of ms `[V: fig_trace_causal.png, panels a and b]`; 3.8 % of the values at or below 3 keV are under 100 eV `[V: R1]`.
      - **Why mean squared error and why 3 keV?** Because away from the large errors the first stage has no effect. Its increment does hold under a Huber loss at 200 eV and with 1 % of the rows trimmed (+0.056 [+0.032, +0.077], +0.073 [+0.034, +0.101]) `[V: B70:40]`, yardsticks that still weigh the jump rows. It does not hold where the typical frame is asked. Median error over that of hold 1.007 [0.968, 1.061], over that of the causal GP 1.037 [1.002, 1.088]; share within 20 % minus the causal GP −0.001 [−0.012, +0.006] `[V: B70:21, 55, 127]`. With the values above 3 keV kept: increment −0.002 [−0.017, +0.010], under Huber −0.012 [−0.038, +0.006], and a median error of 59.5 eV against 45.8 eV for hold `[V: B65:10, 12, 31, 34]`. Without the 15.9 % of rows that the mixture flags: −0.015 [−0.110, +0.052], and +0.065 [+0.010, +0.107] for the two-stage estimator `[V: B70:184, 208-211]`. Coldest third +0.007 [−0.023, +0.031]; without the tenth of the discharges with the largest gain +0.027 [−0.004, +0.051] `[V: GEN:473, 502]`. The size of the effect is a function of how many unusual rows are called truth.
        - **Why was a second stage added?** To repair the median and the 20 % share that the network had failed `[V: PD:182; PK:52, 356]`. It does, modestly: 41.2 eV against 43.0 eV for its twin, ratio 0.959 [0.944, 0.978] `[V: B70:20, 27, 55]`. It is a boosted tree on the network's disagreement with hold and with the causal GP, that is, a learned rule for when to believe which; its settings come from overlapping discharges `[V: MS:781-784]`; and three quarters of its fast-channel gain is frame-specific (Objection 1).
- **THE ROOT.** The headline number is the behaviour of 2 to 6 % of the frames, the spikes and steps of the reported value, under a squared-error yardstick and a 3 keV rule on the label; in the remaining frames the fast channels do nothing for the first stage.
- **What would settle it.** `[existing data, CPU]` Report every increment by class of frame as in R4, with the share of the change anticipated; adopt a robust primary yardstick. `[needs data that do not exist here: spectra or fit errors]` to say what the spikes are.
- **What may honestly be claimed if it is not settled.** "In about one frame in twenty the reported value changes by more than 300 eV; there the estimator anticipates 10 to 15 percentage points more of the change with BES than without `[V: R4]`; elsewhere no benefit of the first stage is established."

### Objection 5. The claim is a moving target and has never met a test it did not help design

- **Statement.** In four days (16 to 19 November) the estimator of the paper changed twice (whole-discharge variant → causal variant → two-stage estimator), with an average of three initialisations named in between as the one to use; the primary readout changed twice (four splits → pooled r2 to r9 → pooled r2 to r4); the population, the name of the target, the title and the axis of the claim each changed once `[V: PD:165-184]`.
  - **Why did it move?** Each registered statement that failed was followed by a readout that held. The split-level statements failed on two of four splits, so the pooled readout became primary; the single training failed, so the average of three initialisations was registered "after they had failed"; the median and the 20 % share failed, so a second stage was added and became "the estimator of this paper"; the increment was not established with the values above 3 keV kept (two of four splits then, −0.002 [−0.017, +0.010] now), so the text became the cut population; "ion temperature" became "the reported value"; "pedestal top" left the title `[V: MS:752-764; PD:168-170, 176, 182, 184; PK:34-36, 149, 593; B65:10]`.
    - **Why were there so many failures to route around?** The effect is of the size of the training noise: between initialisations the skill on one split differs by 0.006 to 0.067, against an increment of 0.05 `[V: PK:471-474, 920]`; and the yardstick is decided by a few rows (Objection 4).
      - **Why was nothing frozen beforehand?** Because no estimand and no decision fixed the design (Objections 1 and 3). The project began as a windowed regression of CES temperature and rotation on fast diagnostics and the question was rewritten to what held `[V: CLAUDE.md, "What this repo is"; PK:95-96]`. The traces of it: 45 registration documents first committed between 15 and 20 November, 19 on one day; 50 commits to the manuscript in five days; 20 dated design changes in four; 17 distinct data treatments among the results; of 15 numerical expectations of one batch 5 were met `[V: git log; PD:165-184; LEDGER:222; SUPP:2661]`. The manuscript describes a scoring rule (at most 500 rows per discharge) that the scorer does not contain `[V: MS:388-390; LEDGER:345-350; grep: CES_MAX_SAMPLES_PER_FILE is read only at train.py:479]`.
        - **Why does registration not protect here?** Registration protects when the data are new or the analyst is blind. Every one of these documents was written by the person who had seen all earlier results on the same 639 discharges; "no untouched hold-out exists" `[V: MS:788-789]`; and when the registered rule failed, the paper kept the estimator and departed from the rule `[V: MS:752-764]`. The main text has no displayed equation, 21 uses of "registered" and 18 of "re-partition", and the words ELM, H-mode and confinement do not occur in it `[V: grep on MS]`. It reports its own bookkeeping.
- **THE ROOT.** There has never been one frozen estimator and one frozen test; registration without fresh data is a diary, not a safeguard.
- **What would settle it.** `[needs training, GPU]` Finish B.69 (the split by time, paused at 25 of 40 runs, PK:941) as a single statement: the fast-channel increment of the frozen two-stage estimator against the following frames. It is still not clean, since those 97 discharges were in every design fold. Clean: `[needs data that do not exist here: discharges of another campaign, or shots never opened]`.
- **What may honestly be claimed if it is not settled.** "An exploratory result from repeated cross-validation on the discharges on which the design was made, with lower interval bounds of +0.02 to +0.08."

### Objection 6. The advance over NN-CES, Diag2Diag and a five-parameter filter is not shown where it counts, and nothing is shown beyond one campaign

- **Statement.** Relative to the prior work the paper cites, the contribution reduces to a two-gain co-variation law.
  - **Where does it count at KSTAR?** At the wait of up to 1 s for a background frame and at the failing edge fits `[V: Lee 2020, p.9; Lee 2026, p.3-5; LIT:100]`. NN-CES removes the wait. Neither NN-CES nor this estimator removes the failing fits: this one is not scored where the value is missing and shows no increment on the rows that resemble such frames (Objection 3).
    - **Why is there no result of the Diag2Diag kind?** Diag2Diag reconstructs Thomson data from other diagnostics with no target measurement at inference (R² 0.92), on 4,000 discharges, and uses the result for a physics observation `[V: Jalalvand 2025, p.1, p.3, p.4 and Methods]`. Here the fast channels without a recent CES value fall behind a stale Gaussian process within 100 ms `[V: B71:84-86]`, and one row of fast channels with the last CES value does not reach the causal GP `[V: SUPP:339]`. The reason is in the data: the 10 ms files keep one raw sample per frame of 0.5 to 2 MHz signals `[V: PK:74]`. The information on which that line of work lives was thrown away before the study began, and raw streams exist for twelve discharges.
      - **Why a network at all?** The five-parameter filter with a BES drive has the same increment as the network over its twin (+0.049), a median error of 41.8 eV against 45.8 eV for the network and 41.2 eV for the two-stage estimator, and is not separated from the network under a Huber loss (+0.013 [−0.009, +0.033]) `[V: GEN:240, 256-258, 274; PK:742]`. On the split by time the network has +0.024 [−0.004, +0.052] over that filter, and the filter's increment is +0.027 [+0.016, +0.038], half of the pooled value `[V: SUPP:488-492]`. The +0.130 of the two-stage estimator over the filter `[V: GEN:234]` compares a stacked estimator with a bare filter; the row "filter plus the same second stage" is missing. Weighing the sources per row before the recurrent layer added nothing `[V: B67:9-11]`.
        - **Why nothing beyond one chord and one campaign?** On MAST no network was trained and the inputs differ `[V: PK:236]`. The split by time lies inside the 2022 campaign and its two-stage version is unfinished `[V: PK:941; LEDGER:156]`. KSTAR has since changed to a tungsten divertor, which the CES group names as changing what the diagnostic must measure `[V: Lee 2026, p.5]`.
- **THE ROOT.** Stripped of machinery, the contribution is a five-parameter co-variation law on one chord in one campaign; the 358 k-parameter network and the boosted stage are not shown to add to it on later discharges.
- **What would settle it.** `[existing data, CPU]` The filter with the same second stage; filters and networks side by side against the following frames and on the split by time. `[needs training, GPU]` B.69. `[needs data that do not exist here: a later campaign or a second device with the same inputs]`.
- **What may honestly be claimed if it is not settled.** "A robust filter with a BES drive lowers the error of the same filter without it by 5 % in pooled cross-validation and by 3 % on the latest discharges."

---

## 3. The bottom of the chains

**The one question.** How much of the 10 ms variation of the reported edge C VI temperature is variation of the ion temperature, and does the estimate know the plasma better or only the reading better? Every chain ends there. Without an independent truth the paper cannot say whether the BES-driven correction is temperature (1, 2), whether anyone gains from it (3), whether the frames that carry the score are physics (4), what a fixed test would be (5), or what it adds to a filter (6). That is the question this thesis is really about. The co-located fast diagnostics are the only handle the files give on it. The thesis used the handle to build a predictor and did not answer the question.

**The single analysis I demand first.** Score every estimator and reference against the reported values of frames t+1 to t+3 and print every increment there beside the same-frame one. It uses only stored arrays and costs about an hour of CPU. My version (R3, appendix B) says three quarters of the fast-channel increment is not there. Until the authors have reproduced or refuted that, no further batch is worth running, and the sentence "most of the network's correction is still present in the next frame" must not stand in an abstract. The first thing to ask of the facility, in the same week, is the per-frame fit error or spectrum, the beam waveform and the equilibrium boundary for the twelve raw-stream discharges.

**The paper I would accept.**

- *Title-level claim.* "The beam-emission sample taken with a charge-exchange frame co-varies with the reported edge C VI temperature of that frame on KSTAR: size, persistence, and what it allows a causal estimate to gain."
- *Result 1.* The co-variation law, with equations: the lag weights of R6, the two gains of the filter, by temperature level, on one data definition, confirmed on the split by time.
- *Result 2.* The decomposition in the paper's own currency: what survives into the following frames (+0.03 to +0.04), what belongs to the frame (three quarters), by class of frame (spike, step, quiet); and the reach when CES stops (50 ms).
- *Result 3.* An identification of the frame-specific part: with fit errors, beam waveform and equilibrium on a subset, plasma against position against fit; or, with the raw streams of the twelve discharges, the location of the exposure and the events present at the stamp. Failing both, a plainly labelled non-identification.
- *Cut.* The two-stage estimator as "the estimator of the paper" (one short section at most); the network design ladder; the registration history and the departure narrative; the requirement documents and the real-time prospect; MAST; rotation; the 20-parameter reading; most of the 49 pages of supplement. Twelve to fourteen pages with equations.
- *Venue.* With results 1 and 2 only, this is a diagnostics note. It is a PPCF paper only if result 3 identifies plasma `[J]`.

---

## Appendix A. The checks I ran

All read-only, numpy only, one thread, each under five minutes. Scripts are in the session scratchpad (it can be cleared; the decisive one is reproduced in appendix B):

`C:\Users\lss\AppData\Local\Temp\claude\ceb9ea105373fe73\cdaee03d-1d09-4ad7-a2e8-81e16fd9fde3\scratchpad\ref_scripts\` (`stageA.py`, `check1b.py`, `check2.py`, `check3.py`, `check3b.py`, `check4.py`, `check5.py`, `check6.py`).

**R1. The 10 ms files, model-free** (`stageA.py`, `check1b.py`; 641 `data/s*.csv`, first and last row of each file dropped, values above 3 keV set missing).

- 245,925 grid rows; 226,991 with T_i; missing 7.70 %; 1,197 values above 3 keV. All four agree with the records.
- Of the values at or below 3 keV: 1.52 % under 50 eV, 3.79 % under 100 eV, 11.6 % under 200 eV.
- 216,897 pairs 10 ms apart: hold has RMSE 214.7 eV and median error 44.3 eV. The 1 % of pairs with the largest step carry 0.538 of the squared step, the 5 % carry 0.782. Pairs that change by more than a factor 2 in 10 ms: 6.29 % of pairs, 0.684 of the squared step; by more than a factor 3: 3.52 %, 0.567.
- 210,911 triples: floor (minus the covariance of successive increments) 18,758 eV², 137.0 eV RMS; hold on these rows 199.7 eV RMS, so twice the floor is 0.941 of its squared error. Lag-1 autocorrelation of increments −0.469 pooled, −0.416 median over 582 discharges.
- Blips with one absent frame, ratio to the median of the rows before, median over events: first row after, T_i 0.814, BES 0.660, ECEI 0.929 (the paper's numbers); last row before, T_i 1.004 against 1.002 for control rows, BES 1.002. The frame before a blip is unchanged, so the blips offer no frame with a partly missing beam except the first one after, in which everything changes at once `[J on the reading; V on the ratios]`.
- Derived `[J]`: with q = 39,880 − 2 × 18,758 = 2,364 eV² per frame, a random walk plus white noise has a steady prediction variance of 7,945 eV² (89 eV) and a posterior variance of 5,581 eV² (75 eV); its predicted error over the floor is 1.42, where the history-only twin sits (1.415, GEN:255).

**R2. Stored D1 rows of r2** (`check2.py`; `data/.b70/stage2_r2_f*.npz` joined to `data/.b70_rescore/b39p_v2_cut_r2_f*/comparison_errors_test.npz` on discharge and row; 222,672 rows, 635 discharges).

- Reproduces the records: network over causal GP +0.093, network over twin +0.046, two-stage over causal GP +0.176, two-stage over two-stage twin +0.095; for the two-stage estimator, the network and the causal GP, RMSE 175.0 / 183.6 / 192.8 eV and median error 41.5 / 47.0 / 44.1 eV.
- Share of rows in which the first estimator has the smaller absolute error: two-stage against causal GP 0.549; two-stage against its twin 0.537; network against its twin 0.496; network against causal GP 0.491.
- Rows needed for half of the net gain (ranked by the size of the difference): 1.07 % (two-stage over causal GP), 1.59 % (two-stage over its twin), 2.15 % (network over twin).
- Discharges with at least 50 rows (587): lower MSE in 0.818 / 0.721 / 0.605 of them; lower median error in 0.651 / 0.589 / 0.428; the tenth with the largest gain holds 0.54 / 0.70 / 1.14 of it (same three pairs).
- Thirds by discharge median: fast-channel increment of the two-stage estimator +0.039 / +0.091 / +0.155, of the network +0.009 / +0.043 / +0.084; the coldest third is 20 % of the rows and 38 % of the squared error of hold.
- Descriptive, the stratum depends on the outcome: in the 72.5 % of rows whose previous value is within 100 eV of the reported one, two-stage over its twin −0.059 and network over twin −0.246; in the 7.7 % of rows where it is off by more than 300 eV (73.7 % of the squared error of the causal GP), +0.105 and +0.080.

**R3. The same estimates scored against the following frames** (`check3b.py`, `check3.py`; r2, r3, r4; rows whose previous frame and three following frames are D1 rows 10 ms apart: 200,740 per re-partition, 92 % of D1 rows; skill = 1 − sum of squared differences of A / that of B; two-level interval, a re-partition drawn and then its discharges, 2,000 draws; the previous row reproduces the stored error of hold on every row).

| loss | scored against | network over twin | two-stage over two-stage twin | two-stage twin over twin | two-stage over hold | two-stage twin over hold |
|---|---|---|---|---|---|---|
| squared | same frame | +0.070 [+0.044, +0.094] | +0.136 [+0.113, +0.159] | +0.046 [+0.037, +0.057] | +0.454 [+0.432, +0.475] | +0.368 [+0.350, +0.384] |
| squared | next frame | +0.017 [−0.004, +0.037] | +0.032 [+0.016, +0.048] | +0.040 [+0.032, +0.049] | +0.353 [+0.335, +0.369] | +0.332 [+0.315, +0.348] |
| squared | mean of next three | +0.032 [+0.003, +0.058] | +0.038 [+0.017, +0.060] | +0.068 [+0.058, +0.080] | +0.477 [+0.458, +0.494] | +0.456 [+0.432, +0.477] |
| Huber 200 eV | same frame | +0.066 [+0.042, +0.091] | +0.132 [+0.115, +0.149] | +0.053 [+0.040, +0.064] | +0.375 [+0.357, +0.391] | +0.279 [+0.264, +0.293] |
| Huber 200 eV | next frame | +0.010 [−0.011, +0.030] | +0.029 [+0.015, +0.043] | +0.041 [+0.032, +0.050] | +0.258 [+0.241, +0.275] | +0.236 [+0.220, +0.251] |
| Huber 200 eV | mean of next three | +0.024 [−0.002, +0.047] | +0.030 [+0.014, +0.045] | +0.063 [+0.054, +0.072] | +0.351 [+0.336, +0.367] | +0.331 [+0.313, +0.348] |

Median absolute difference (eV; hold / twin / network / two-stage twin / two-stage): same frame 42.3 / 41.9 / 42.3 / 39.6 / 38.1; next frame 45.8 / 44.8 / 46.7 / 43.0 / 43.2; mean of next three 44.7 / 38.7 / 40.6 / 36.9 / 37.3. Without the requirement on the previous frame (204,733 rows): two-stage over its twin against frames t+2 and t+3, +0.021 [+0.007, +0.035] and +0.016 [+0.002, +0.027].

Limits: three re-partitions, one initialisation per network; rows just before a gap are excluded; a real excursion shorter than a frame would also fail to persist, so the non-persisting part is "of the frame", not proven "of the fit".

**R4. Where the fast-channel gain is, by kind of frame** (`check4.py`; r2 to r4, 204,733 rows with y(t−2) to y(t+1) 10 ms apart; a large change is |y(t) − y(t−1)| > 300 eV; classes use later values and are descriptive).

| class | rows | squared error of the two-stage twin | share of the fast-channel gain, two-stage | the same, network | share of the change anticipated, two-stage / its twin / network / twin |
|---|---|---|---|---|---|
| A spike at t that reverts at t+1 | 0.0211 | 0.378 | 0.309 | 0.493 | 0.218 / 0.119 / 0.179 / 0.092 |
| B return from a spike at t−1 | 0.0207 | 0.084 | 0.075 | 0.066 | 0.725 / 0.684 / 0.652 / 0.642 |
| C step at t that stays at t+1 | 0.0156 | 0.135 | 0.272 | 0.333 | 0.489 / 0.341 / 0.409 / 0.311 |
| D other large change | 0.0048 | 0.066 | 0.061 | 0.086 | 0.237 / 0.124 / 0.170 / 0.090 |
| E no large change | 0.9377 | 0.337 | 0.283 | 0.021 | 0.412 / 0.275 / 0.381 / 0.243 |

**R5. Spikes and steps in the files, and the BES sample of the same row** (`check5.py`; 206,216 rows).

- Upward spikes that revert: 3,198 rows (1.55 %), 0.230 of the squared step, median 824 → 1,340 eV; their returns 0.254. Downward dips that revert: 1,203 rows (0.58 %), 0.052; their returns 0.059. Steps that stay: 0.165. Rows without a large change (93.7 %): 0.208.
- The BES common mode rises into 65 % of the upward spikes (mean change of the logarithm +0.033) and 36.9 % of them occur in the highest fifth of the discharge's BES level; it falls into 69 % of the dips and 77 % of the downward steps. The upward spikes below 3 keV are therefore not the low-BES population of the values above 3 keV.
- Correlation of the change of ln T_i of row t with the change of the BES common mode of row t−1, t, t+1 (rows not after an absent frame): −0.023, +0.151, −0.058.

**R6. On which BES sample does the reported value of row t depend** (`check6.py`; 209,695 rows, 612 discharges; least squares of the change of ln T_i(t), winsorised at the 1st and 99th percentile, on the changes of C at stamps t−2 to t+2, C = mean logarithm of the nine BES channels; if ln T_i(t) = Σ a_j C(stamp t+j), the coefficient on the change at stamp t+j is a_j; interval over discharges, 300 draws).

- Common mode: a(−2 … +2) = +0.081 [+0.046, +0.117], +0.218 [+0.174, +0.265], +0.670 [+0.578, +0.768], +0.078 [+0.018, +0.132], +0.145 [+0.109, +0.185]; a(0) − a(+1) = +0.593 [+0.486, +0.698].
- Radial difference (column 07 minus column 05): +0.045, +0.185, +0.596 [+0.509, +0.673], +0.322 [+0.260, +0.394], +0.186; a(0) − a(+1) = +0.275 [+0.193, +0.355].
- The reading that the exposure is short is a judgement; a causal lag of the reported value behind the beam emission would give the same numbers.

**Other verifications.** `git log`: `main_ppcf.tex` has 50 commits between 2026-11-15 and 2026-11-19 (28 on 11-18); 45 of the 77 registration documents were first committed between 11-15 and 11-20. `grep` on MS: no displayed equation; "registered" or "registration" 21 times; "re-partition" 18 times; no occurrence of ELM, H-mode, L-mode or confinement. `grep` on `ces_prediction/`: `CES_MAX_SAMPLES_PER_FILE` is read only at `train.py:479`.

**Not verified by me.** The trainings themselves; the B.71 replay; the references' code; any spectrum. The numbers of B.65, B.67, B.71 and of the persistence table are quoted from the authors' files.

---

## Appendix B. The decisive check (R3)

A condensed copy of `check3b.py`. I ran this copy as well: the point values are identical to the table of R3, and the ends of the intervals move by up to 0.002 with the order of the bootstrap draws.

```python
# est(t) of the stored D1 rows compared with y(t), y(t+1) and the mean of y(t+1..t+3).
# py -I check3b.py data 2,3,4      (numpy only; reads data/.b70 and data/.b70_rescore)
import sys, os
import numpy as np
root = sys.argv[1]; reps = [int(x) for x in sys.argv[2].split(",")]
rng = np.random.default_rng(5)
def huber(e, k=200.0):
    a = np.abs(e); return np.where(a <= k, 0.5 * a * a, k * (a - 0.5 * k))
def load(r):
    out = []
    for k in range(5):
        s = np.load(os.path.join(root, ".b70", "stage2_r%d_f%d.npz" % (r, k)), allow_pickle=True)
        c = np.load(os.path.join(root, ".b70_rescore", "b39p_v2_cut_r%d_f%d" % (r, k), "comparison_errors_test.npz"), allow_pickle=True)
        ks = s["shot"].astype(np.int64) * 1000000 + s["row"].astype(np.int64)
        kc = c["CES_TI_shot"].astype(np.int64) * 1000000 + c["CES_TI_row"].astype(np.int64)
        os_, oc = np.argsort(ks), np.argsort(kc)
        assert np.array_equal(ks[os_], kc[oc])
        d = dict(y=s["y"][os_], c=s["est_c"][os_], c2=s["est_c2"][os_], t=s["est_t"][os_], t2=s["est_t2"][os_],
                 fidx=s["fidx"][os_].astype(np.int64), grow=s["grow"][os_].astype(np.int64),
                 dt=c["CES_TI_dt_ms"][oc], se_h=c["CES_TI_se_persistence"][oc])
        d["dis"] = d["fidx"] + 100000 * k
        out.append(d)
    return {k: np.concatenate([o[k] for o in out]) for k in out[0]}
def two_level(A, Bv, B=2000):
    pts = [1 - a.sum() / b.sum() for a, b in zip(A, Bv)]; draws = []
    for _ in range(B):
        j = rng.integers(0, len(A)); a, b = A[j], Bv[j]
        pick = rng.integers(0, len(a), len(a)); draws.append(1 - a[pick].sum() / b[pick].sum())
    return "%+.3f [%+.3f, %+.3f]" % (np.mean(pts), np.percentile(draws, 2.5), np.percentile(draws, 97.5))
R = {}; MED = {}
for r in reps:
    D = load(r); key = D["dis"] * 100000 + D["grow"]; o = np.argsort(key)
    for k in D: D[k] = D[k][o]
    key = key[o]; n = len(key); reg = np.abs(D["dt"] - 10) < 0.5   # the row's previous kept row is 10 ms earlier
    okp = np.zeros(n, bool); okp[1:] = (key[1:] == key[:-1] + 1) & reg[1:]
    ok3 = np.zeros(n, bool)
    ok3[:-3] = ((key[1:-2] == key[:-3] + 1) & reg[1:-2] & (key[2:-1] == key[:-3] + 2) & reg[2:-1]
                & (key[3:] == key[:-3] + 3) & reg[3:])
    idx = np.where(ok3 & okp)[0]
    y0 = D["y"][idx]; y1 = D["y"][idx + 1]; ym = (D["y"][idx + 1] + D["y"][idx + 2] + D["y"][idx + 3]) / 3.0
    est = dict(c=D["c"][idx], c2=D["c2"][idx], t=D["t"][idx], t2=D["t2"][idx], hold=D["y"][idx - 1])
    assert (np.abs((est["hold"] - y0) ** 2 - D["se_h"][idx]) < 1).all()     # alignment check
    us, inv = np.unique(D["dis"][idx], return_inverse=True)
    for tn, tg in (("same frame", y0), ("next frame", y1), ("mean of next three", ym)):
        for e in est:
            R.setdefault((tn, e, "mse"), []).append(np.bincount(inv, weights=(est[e] - tg) ** 2))
            R.setdefault((tn, e, "hub"), []).append(np.bincount(inv, weights=huber(est[e] - tg)))
            MED.setdefault((tn, e), []).append(np.median(np.abs(est[e] - tg)))
for loss in ("mse", "hub"):
    for tn in ("same frame", "next frame", "mean of next three"):
        print(loss, tn, "| net/twin", two_level(R[(tn, "c", loss)], R[(tn, "t", loss)]),
              "| two/twin2", two_level(R[(tn, "c2", loss)], R[(tn, "t2", loss)]),
              "| twin2/twin", two_level(R[(tn, "t2", loss)], R[(tn, "t", loss)]))
for tn in ("same frame", "next frame", "mean of next three"):
    print(tn, {e: round(float(np.mean(MED[(tn, e)])), 1) for e in ("hold", "t", "c", "t2", "c2")})
```
