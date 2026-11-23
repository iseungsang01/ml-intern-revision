# Referee report

**Manuscript:** "Causal nowcasting of the charge-exchange ion temperature on an edge chord of KSTAR with fast diagnostics at the pedestal top" (main text and supplement)

*Note on scope:* I could not run web searches in this setting. Statements about prior work therefore rest on the manuscript's own citations and my general knowledge. Several cited works carry 2026–2027 dates (e.g. [4], [5], [9], [10], [13], [16], [25]; supplement refs. [5], [6], [9], [14], [15]), and I could not verify them. The editor should check them.

## 1. Summary

The authors want a causal estimate, made each 10 ms frame, of the C VI charge-exchange (CES) ion temperature on a single KSTAR chord (R = 2.198 m) that the data provider labelled "pedestal top". The data are 635–639 discharges of the 2022 campaign.

The estimator has two stages:
- a Matérn-3/2 Gaussian process (GP) on the last 16 visible CES values;
- a default-setting gradient-boosted regression on the GP residual, using 10 history scalars plus 14 scalars from the BES and ECEI array means.

It is compared with hold, a causal GP, and two non-causal interpolants (PCHIP and a two-sided GP), under four validation designs, with shot-cluster bootstrap intervals. With the previous frame visible, the estimator beats PCHIP in MSE under all designs. It beats the two-sided GP only under random and cluster-held-out splits; in time order the difference is not established. The fast-channel increment is small (2–9 %).

When the CES history is artificially aged (0.05–1 s, a beam-modulation schedule, or thinned sampling), the increment grows to 20–35 %. A MAST replication with Thomson scattering shows the same direction.

Failures are reported openly: the gain is not radius-specific (Mirnov signals carry about a quarter of it), it vanishes for the last block of shots under the schedule, and the regression must be refitted for each availability condition. A budgeted TCN is reported as a non-adopted comparison.

## 2. Assessment against PPCF criteria

**(a) Significance.** A real-time pedestal-top Ti estimate is of genuine interest. However, the delivered estimator is not physics-based. It is tested only against the reported CES value, on one chord and one campaign, and its operational relevance is conditional: real-time fast data at KSTAR are not demonstrated, and the "beam-modulation schedule" is a hypothetical reading (§2.2). The physics content for PPCF readers is thin. The paper says no channel-specific mechanism is established (§4.2) and that ECEI is uncalibrated and density-dependent (§5).

**(b) Novelty.** The authors concede, correctly, that every component is standard and the contribution is "the measured comparison for this chord" (§1). Virtual-diagnostic, multimodal super-resolution and missing-channel reconstruction work ([12], [15]–[18]) is closely related. The new element is the careful causal/staleness framing, not a method.

**(c) Soundness.** The validation design is unusually careful:
- four splits;
- cluster bootstrap;
- permutation and anchor controls;
- pre-registration;
- independent recomputation.

Important weaknesses remain, listed under Major concerns: target contamination, small-cluster bootstrap, multiplicity, and ill-defined real-time relevance.

**(d) Conclusions vs evidence.** The headline claims are mostly hedged appropriately. Several summary sentences in §4.2, §4.7 and the abstract overgeneralise beyond their tables (see §8).

**(e) Clarity and length.** This is the weakest aspect. The prose is idiosyncratic and very dense ("the value of the frame", "the level of the following frames", "established", "recovered share"). Acronyms are overloaded: "TX" means both the estimator and the fast TCN (§3.2, Table 11). Row counts change from table to table (641/639/635/627/613/562/503 discharges). The 26-page supplement with 36 tables is excessive for the result. §4.7 reads like generated boilerplate, and its tables use a different style (fractions, unbolded intervals).

**(f) Reproducibility.**
- Good: code, registrations and MAST data are said to be public, and settings are given (scikit-learn defaults, seed).
- Limitation: the KSTAR data are available only on request to KFE.

## 3. Major concerns (ranked)

**1. Target F may reward predicting fit artefacts.**
- Table S5 shows one-frame upward Ti excursions are about 9.5× enriched in frames containing a BES burst.
- The fast BES/ECEI sample is taken at the stamp, after "most of that light has come in" (§S1).
- The D10 increment on F (Table 4, 8.7 % V1) could therefore partly be the regression learning to reproduce ELM-correlated fit errors rather than temperature.
- Similarly, the "advantage in larger errors" over interpolation (Table 4, lower block) is consistent with predicting excursions that the authors themselves suspect are not physical.

*Needed:* report F-target results with burst/ELM frames excluded or stratified (the 11 raw-stream discharges, plus a BES-derived burst proxy on the 10 ms data). Also state clearly which target is the physically meaningful one. *New analysis of existing data.*

**2. Real-time relevance is asserted but not established.**
- Fast samples delivered in real time are not published for KSTAR (§4.3).
- The CES analysis latency is unknown, and S is a constructed schedule (§2.2).
- The 2 ms timing (Table S13) excludes acquisition and preprocessing.

The "nowcast" framing therefore rests on hypotheticals. *Needed:* reframe as an offline feasibility study, or document the actual KSTAR data paths. *Rewriting (or new data).*

**3. Statistical inference is fragile in the time-ordered designs.**
- V4 scores 14–19 clusters with a percentile cluster bootstrap. Coverage is known to be poor at this size.
- There are 170 registered statements and no multiplicity control.
- Seed variance changes signs (Table S15, S rule).
- Fit variability is excluded from all intervals.

Several headline "established" verdicts sit at lower bounds near zero (e.g. V4 F increment 1.9 [0.1, 3.6]; V2 vs offline GP 3.7 [0.7, 7.0], which is in the abstract).

*Needed:*
- wild/studentised cluster bootstrap or permutation tests;
- a refit-inclusive uncertainty (seeds × folds);
- an explicit family-wise or FDR statement for the headline claims.

*New analysis of existing data.*

**4. No physics or model-based baseline.**
- The only causal competitors are hold, GP and the authors' own variants.
- A Kalman observer coupling Ti to an electron channel is absent. The K0/KX filters in Table S32 are only mentioned in the supplement.
- So is a simple ECEI-ratio scaling on KSTAR (RT is tried only on MAST, Table S23).

Without these, "the estimator with the lowest error that we found" is weak. *New analysis.*

**5. Interpretability of the fast inputs.**
- ECEI is uncalibrated, possibly optically grey (Table S1), constant over 3.5 % of rows, and fails in the last block.
- The chord's ρ is unknown (§S1 gives r/a ≈ 0.92 vs 0.94–0.97).
- Mirnov signals give about a quarter of the gain.

The paper therefore cannot say what physical information is being used. At minimum, the title's "at the pedestal top" and "fast diagnostics at the pedestal top" should be softened. *Rewriting.*

**6. The TCN section (§4.7, Tables 9–11) adds little and confuses.**
- It is scored on different rows, uses fraction rather than percent, and reuses the label TX.
- Its RMSE (~200 eV) cannot be compared with Table 3.

Either integrate it on common rows or move it to the supplement. *Rewriting / reanalysis.*

**7. Length and style.** Cut the main text by about 30 % and the supplement substantially. Use conventional terminology (bias/variance, ELM-filtered target, latency). *Rewriting.*

## 4. Minor concerns

- Table 1 caption (1,414 absent steps, 245,925 rows) and the table body (1,413; 245,785) differ. This is explained only implicitly.
- The supplement front matter says the batches are "B.73 to B.84", but Table S33 lists up to B.86.
- The GP length-scale grid (5–80 ms) has a minimum below the frame spacing (10 ms) and its maximum may be too short for D650/D1000 histories. Justify it.
- L5 requires ≥3 of 5 frames, which biases scored rows towards well-measured periods. Quantify this.
- The default HistGradientBoosting settings are untuned. A sensitivity check would help.
- Fig. 4 plots MAST's "O" line while MAST TX beats O. Discuss why the frame reading is so noisy there.
- The §5 sentence "within 10 % of the level in two frames out of three ... short of the 10 %" conflates accuracy with a coverage share. Rephrase.
- Define "skill" consistently: fraction in the supplement and §4.7, percent elsewhere.
- In Table 4, the PCHIP median ratio 1.04 is called "not established" in the text but is listed under "does not hold" in Table S36 I12. Harmonise the wording.

## 5. Recommendation

**Major revision**, bordering on reject-and-resubmit.

The validation work is careful and honest, and the reporting of negative results is exemplary. However:
- the physics contribution is limited;
- the main F-target gain may partly reflect ELM-correlated fit artefacts;
- the real-time framing rests on unverified assumptions;
- the statistics in the decisive time-ordered designs are fragile;
- the manuscript is far longer and harder to read than its single-chord empirical result warrants.

A shorter, reframed paper with the burst-stratified analysis and a physics/observer baseline could be acceptable. In its present form it reads more like an engineering/methods report, better suited to FED.

## 6. Probabilities

- **(i) Sent to referees by PPCF: 55 %.** The topic is in scope and the work is competent, but it is method-light and physics-light, and the idiosyncratic style invites a desk decision.
- **(ii) Eventually accepted by PPCF: 30 %.** The concerns are addressable with existing data, but PPCF referees will likely still find the physics content thin.
- **(iii) Eventually accepted by FED: 55 %.** Real-time estimation and diagnostic-engineering work fits that journal well.
- **(iv) Eventually accepted by RSI: 30 %.** No instrument development is involved, though a data-analysis paper is possible.

## 7. Changes that would most raise (ii)

1. Burst/ELM-stratified evaluation of the F and L5 gains, plus a physics/observer baseline (Kalman filter with Ti/Te coupling, ECEI-ratio scaling). *New analysis of existing data.*
2. Reframe as an offline feasibility study, soften "pedestal top"/"real-time", and cut length by at least 30 %, moving the TCN section and most batch bookkeeping to the supplement. *Rewriting only.*
3. Robust inference: small-cluster-appropriate bootstrap, refit/seed uncertainty, and multiplicity control for the headline claims. *New analysis of existing data.*

## 8. Errors, inconsistencies and overstatements

- **Table 7, "TX / T0" rows: columns are shifted.** The ratios 0.96/0.97/0.95 are the MAE/median/p90 ratios (cf. Table S14; 69.1/72.3 = 0.956), but they are printed under RMSE/MAE/median. The share differences sit under the wrong columns.
- **§4.2 "Controls" overstates the anchor-row result.** It says present-sample TX beats anchor-row samples "under every design". Table S17 shows this fails under rule S for both V4 fits (−1.1 [−40.8, 17.1]; −12.0 [−44.3, 13.2]). It holds only for D650.
- **§4.2 "Beam waveforms" overstates likewise.** It says fast diagnostics still help "under all four designs". This is true for D650 only; under S, V4 is not established (Table S16: 10.2 [−15.6, 24.2], −6.2; Table S20 N7).
- **The abstract's TCN sentence is inaccurate for one scope.** It says the gain "is established under the random design only in both of its evaluation scopes". In the fixed subset, V2 S/L5 is also established (Table 9: +0.194 [+0.007, +0.339]).
- **Label collision.** "TX" denotes both the GP+boosting estimator and the fast TCN (§3.2, Table 11). This invites misreading of the latency table.
- **The "nowcast" and "2 ms per frame" claims overstate real-time readiness.** Acquisition and transfer are excluded (Table S13, Table 11), and KSTAR real-time BES/ECEI delivery is unpublished (§4.3).
- **Increment values differ across tables for the same design without a pointer to the row sets.**
  - D10/L5 increment: V1 7.5 % in Table 4 vs 10.3 % in Tables 5/6; V4 8.4 vs 8.9.
  - The abstract quotes the F target, where V4 is 1.9 [0.1, 3.6], which is "only just" established.
- **Discharge and row counts are inconsistent.** §2.1 gives 639 discharges and the §3.5 data definition 639, while Table 1 and Table S6 give 635. Table 1 gives 1,413 absent steps vs 1,414 in its caption; Table 1 uses 245,785 grid rows vs 245,925.
- **The supplement header says registered batches run B.73–B.84; Table S33 lists through B.86.**
- **The title and §2.1 say "at the pedestal top", while §S1 places the chord at r/a ≈ 0.92.** That is below published pedestal-top positions, and the authors themselves "do not claim that it sits at the pedestal top".
- **§4.2 "Specificity": the Mirnov control undermines the "at the location" framing used throughout** (abstract, title). The framing should be dropped, not merely footnoted.

PPCF accept: 30 %; FED accept: 55 %; RSI accept: 30 %; sent to referees: 55 %