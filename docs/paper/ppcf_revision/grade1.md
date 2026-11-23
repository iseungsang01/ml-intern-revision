# Referee report

**Manuscript:** "Causal nowcasting of the charge-exchange ion temperature on an edge chord of KSTAR with fast diagnostics at the pedestal top" (PPCF submission, with supplementary material)

## 1. Summary

The manuscript asks how best to estimate the CES ion temperature on one KSTAR pedestal-top chord (R = 2.198 m) at each 10 ms frame, using only information available up to that frame. The data are 2022-campaign files on a 10 ms grid, holding the reported Ti, single samples of nine BES and four ECEI channels, and beam waveforms.

The proposed estimator (TX) has two stages. A causal Matérn-3/2 GP runs on the last 16 visible Ti values. A default-settings scikit-learn histogram gradient-boosting regression then corrects the GP residual, using 10 history scalars and 14 scalars from the BES and ECEI array means.

The estimator is scored against two targets: the frame's own reported value (F) and the median of the next five frames (L5). It is compared with hold, the causal GP, a history-only twin (T0), and two non-causal references (PCHIP and a two-sided GP). Four validation designs are used, with cluster-bootstrap intervals.

The main findings are:
- With the previous frame visible, TX beats PCHIP in all designs.
- TX beats the two-sided GP only under the random and cluster designs.
- The fast-channel increment is small (2–9 %).
- When the CES history is artificially aged (0.65 s, beam-modulation schedule, sparse sampling), the fast-channel increment grows to 20–35 %.
- A MAST analogue (Thomson as the "present" channel, 100 ms delay) shows the same direction.

The authors report the failures as well: no radial specificity (Mirnov signals carry about a quarter of the gain), the last shot block fails under schedule S, and the estimator needs refitting per condition. The authors themselves describe the contribution as empirical and application-specific.

## 2. Assessment against PPCF criteria

**(a) Significance.**
- The physics content is thin. No plasma-physics result is claimed, and the authors explicitly decline to infer ion–electron coupling.
- Practical value is conditional on things that do not exist at KSTAR:
  - real-time CES values (the "nowcast condition" D10 is hypothetical given the ~1 s analysis delay, §2.2);
  - real-time BES/ECEI delivery (§4.3).
- The realistic condition, S, is where the gain is not established for the latest block (Table 8).
- The interest to the PPCF readership is therefore limited. This reads as a diagnostics/data-processing study.

**(b) Novelty.**
- The authors concede (§1) that the components are standard and the general problem is not new.
- Virtual-diagnostic and multi-rate fusion work is cited (refs 12, 15–18, 24–26).
- The increment is a GP + GBM residual model on one chord. Methodological novelty is minimal, and empirical novelty is narrow (one chord, one campaign).

**(c) Soundness.** The validation design is unusually careful: four designs, cluster bootstrap, controls, registration, and independent recomputation. That deserves credit. The structural problems are:
1. The target is the same noisy analysis output, including fit artefacts.
2. The data were all read before registration, and estimator inputs were fixed on the same discharges (§5 Limits).
3. There are about 170 registered statements with no multiplicity correction.
4. Intervals exclude fit variance. Table S15 shows that this variance matters exactly where the conclusions are marginal.

**(d) Conclusions vs evidence.**
- The headline "lowest-error causal estimator" is supported only against the narrow set of competitors tried. The regression is untuned, there is one RNN, and the TCN results are absent (see below).
- The claim that the fast diagnostics "carry the difference" to the offline GP holds only under V1 and V2.

**(e) Clarity.**
- The prose is idiosyncratic and hard to follow: long hedged sentences, and many symbols (D10, Dd, S, Kk, V1–V4, F, L5, O, TE, TB, TN, TXN, TX8, TXall, TXa, TXrel, TM, TXs, TG…).
- The label TX is reused for a different model, the fast TCN (§3.2).
- Cross-references are unresolved ("§??", p.3, 6, 8, 10), and the supplement has dozens of "[? ]" citations.
- The total length (17 + 29 pages) is out of proportion to the result.

**(f) Reproducibility.**
- Code and registrations are promised (anonymised repository).
- The KSTAR data are not redistributable, so the main result cannot be reproduced externally.
- MAST is public, which is a plus.

## 3. Major concerns (ranked)

**M1. The TCN comparison is described but not reported.**
- The abstract makes specific claims about a "separately audited temporal convolutional network".
- §1, §3.2 and §3.7 refer to a section and a latency table ("§??") that do not exist in the manuscript or the supplement. No TCN number appears anywhere.
- *Fix:* either add the section and tables, or remove every TCN claim from the abstract and text. Rewriting only, if the results exist.

**M2. The target cannot separate temperature from measurement artefact, and the fast channels may be predicting the artefact.**
- Table S5 shows that one-frame upward Ti excursions are about 9.5 times enriched in frames with a BES burst inside the exposure.
- The BES sample is taken within ~2 ms of the frame start, after "most of that light has come in" (S1).
- A BES-informed estimator scored on F can therefore gain by predicting ELM-disturbed fits, not ion temperature. The paper itself says "the advantage lies in the larger errors" (§4.1), which is consistent with this.
- *Fix:* re-score TX vs T0 and vs the offline GP on F with burst/excursion frames excluded or stratified, and report the increment in quiet frames. New analysis of existing data (BES-derived burst flags for all shots, or a 10 ms proxy).

**M3. The practical relevance of the operating conditions is not demonstrated.**
- D10 presupposes a real-time per-frame CES reader that KSTAR does not have.
- S is "one condition of use, not a fact about an existing system" (§2.2), and assumes zero analysis time.
- Under S, the most recent block fails: V4 fit to 32310 gives −25.4 [−54.1, 3.9], and the random-design block value is 6.6 [−16.3, 24.8] (Table 8). The cause is unexplained.
- If the most recent hardware/settings state is where the method fails, the deployability claim is weak.
- *Fix:* frame the work explicitly as an offline feasibility study. Quantify a realistic analysis latency. Explain or bound the last-block failure, ideally with ECEI LO/field settings (new data/metadata).

**M4. Statistical inference is fragile where the claims are marginal.**
- Several headline "established" results have lower bounds near zero:
  - TX vs offline GP, V2: 0.7;
  - V4 increment on F: 0.1;
  - D650 V3: 3.3.
- These come with 14–22 clusters, no multiplicity correction across about 170 statements, and no fit-variance component.
- Table S15 shows seed spread of up to 5.8 pp under S.
- *Fix:* add fit variance via refits inside the bootstrap, or a nested scheme; apply a family-wise or FDR correction to the headline set; restate claims accordingly. New analysis of existing data.

**M5. Specificity and physical interpretation are absent.**
- Mirnov signals (not at the radius) reproduce about a quarter of the increment and are not beaten under V3 (Table S19).
- Discharge-level confounders (Ip, B, density) are not in the files.
- What the fast channels encode is therefore not identified: it may be a global confinement-state indicator.
- *Fix:* test against global signals (D-alpha, density, stored energy) as inputs. This likely needs new data from KSTAR.

**M6. The baseline set is limited and the estimator untuned.**
- The study claims the "lowest error that we found" with a default GBM, one LSTM, and no Kalman-type observer with an ECEI/BES measurement model. The K0/KX filters appear only in Table S31/S32, with five parameters.
- A tuned baseline is needed for the claim that "no neural network is needed".
- *Fix:* tune both stages under nested CV; include a proper state-space observer. New analysis.

**M7. Chord location and diagnostic co-location are uncertain.**
- The chord is likely about 4 cm inside the separatrix (r/a ≈ 0.92 per S1 arithmetic).
- The ECEI column is uncertain by 2–3 cm between discharges, and the BES column mapping is inferred.
- "Pedestal top" and "collocated" are therefore assertions, not facts.
- *Fix:* soften the title and text. Rewriting.

## 4. Minor concerns

- Discharge counts vary without reconciliation: 641 files, 639 (§2.1, §3.5), 635 (Table 1), 627 (abstract, §4.1). Grid rows are 245,925 (text) vs 245,785 (Table 1), and absent steps 1,414 vs 1,413.
- The supplement intro says batches B.73–B.84, but Table S33 runs to B.86.
- Many references carry 2026–2027 dates. Ref. [9] is "published online 22 October 2026" in vol. 67, 2027, and ref. [13] is arXiv:2610.00243. I could not verify these items and they should be checked. UFCES is cited inline by DOI rather than in the reference list.
- Table 3 mixes row sets across columns (222,005 / 215,369 / 160,343 rows), which hampers comparison.
- Fig. 4 mixes S results "drawn at median age" with fixed-delay curves on different rows. Plot them separately or label them more clearly.
- The phrase "a bias of the analysis that lasts longer than 50 ms is in them as well" should be quantified or moved to the limits.
- The "within 10 %" metric is quoted against a requirement of 10 % accuracy; the requirement's statistical definition is not stated.
- Equations (3)–(4) mix standardized and raw means. Justify the log ratio on raw means, given the uncalibrated ECEI.
- Removing values that repeat to the last digit could also remove genuine identical fits. Quantify how many are removed.

## 5. Recommendation

**Major revision**, bordering on reject-and-resubmit to a more suitable venue.

The empirical work is careful and unusually candid about failures. However:
- A claimed component (TCN) is missing from the manuscript.
- The primary target conflates the temperature with fit artefacts that the fast channels can see.
- The operating conditions where the method would be useful are hypothetical or fail on the latest data.
- The contribution is methodologically routine and limited to one chord.
- The presentation is very hard to read, with unresolved references throughout.

For PPCF the physics interest is marginal. The work fits better in a diagnostics/engineering journal once the issues above are fixed.

## 6. Probability estimates

- **(i) Sent to referees by PPCF: 40 %.** Diagnostics-ML papers do appear in PPCF, but the editor will see unresolved "§??" references, no physics result, and an engineering focus.
- **(ii) Accepted by PPCF: 20 %.** Even after revision, scope and novelty remain weak for PPCF; acceptance depends on a sympathetic editor.
- **(iii) Accepted by Fusion Engineering and Design: 50 %.** The scope is a good fit (real-time estimation, KSTAR), and the bar for methodological novelty is lower.
- **(iv) Accepted by Review of Scientific Instruments: 25 %.** RSI prefers instrument or measurement advances, so a pure post-processing estimator with no real-time implementation is a weak fit.

## 7. Changes that would raise (ii) most

1. Remove or fully report the TCN material, and cut the manuscript to a readable core with conventional terminology and resolved references. *Rewriting only* (or *new analysis of existing data*, if the TCN tables need regenerating).
2. Show that the fast-channel gain survives on frames without ELM/burst signatures and against global confinement signals. *New analysis of existing data*; global signals would need *new data*.
3. Explain the last-block failure using per-shot field and ECEI settings, and demonstrate the method on an independent campaign. *New data.*

## 8. Errors, inconsistencies, overstatements

- **Abstract, §1, §3.2, §3.7, §4.1:** TCN claims rest on a nonexistent section ("§??"), so these statements are unsupported in the submitted material.
- **Abstract "627 discharges"** vs Table 1 "635" vs §3.5 "639", and 245,925 vs 245,785 rows. Inconsistent.
- **Abstract "the fast diagnostics carry the difference"** to the two-sided GP: this is true only under V1 and V2 (Table 4). Under V3 and V4, TX is not distinguished from the offline GP either.
- **§4.2 "established under all four designs"** on F: the V4 interval is [0.1, 3.6], and S15 seed spread was not measured for D10. This overstates robustness.
- **§4.4 "the fast diagnostics are worth a CES that reports twice as often"**: this holds under V1 only. Under V4 at K10 the increment is not established (Table S27: 2.3 [−7.7, 12.0]).
- **§5 "Value for monitoring and control"**: real-time BES/ECEI and CES at KSTAR are unavailable (Table S1), so the "prospect" framing should be stronger.
- **Title and abstract, "at the pedestal top" and "at the same location"**: S1 places the chord about 4 cm inside one separatrix and the ECEI column with 2–3 cm uncertainty. Both are stronger than the evidence.
- **Supplement:** many "[? ]" citations, and "B.73 to B.84" contradicts Table S33 (B.86).

PPCF accept: 20 %; FED accept: 50 %; RSI accept: 25 %; sent to referees: 40 %