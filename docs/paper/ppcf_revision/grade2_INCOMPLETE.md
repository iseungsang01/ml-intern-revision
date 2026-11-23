Incomplete API response: output-token limit; not a completed grade.

# Referee report

**Manuscript:** "Causal nowcasting of the charge-exchange ion temperature on an edge chord of KSTAR with fast diagnostics at the pedestal top"

## 1. Summary

The manuscript studies one KSTAR CES chord (R = 2.198 m) labelled as the pedestal top, using 10 ms files from the 2022 campaign. It asks which causal estimator best predicts the reported C VI temperature of the current frame (F), or the median of the next five frames (L5), from earlier CES values and single-sample BES and ECEI array means.

The proposed estimator, TX, is a causal Matérn-3/2 GP on the last 16 values plus an untuned scikit-learn gradient-boosting correction on 24 hand-built scalars. With the previous frame visible, TX beats hold, the causal GP and PCHIP under four validation designs. It beats a two-sided GP only under the random and cluster-held-out designs; in time order no difference is established.

The fast-channel increment over the history-only estimator T0 is small with fresh history (about 2–9 % MSE). It grows to about 20–35 % when CES history is artificially delayed (D650), thinned (Kk), or restricted by a reconstructed beam-modulation schedule (S). A MAST replication with Thomson Te as the "present" input shows the same direction.

The paper reports many negative findings:
- no gain under S for the last block of shots;
- no radial specificity, since Mirnov signals carry about a quarter of the increment;
- the estimator must be refitted for each availability condition;
- an added TCN comparison that is not adopted.

## 2. Assessment against PPCF criteria

**(a) Significance.** The problem is real: edge Ti for control is scarce, and KSTAR CES is late. However, the deliverable is a statistical estimator of a *reported analysis output* on one chord, with no independent Ti reference. The best accuracy reached (two thirds of estimates within 10 % of L5 with fresh history; below half with stale history) falls short of the 10 % requirement the authors themselves quote. No control use is demonstrated. The physics content for PPCF readers is thin: §5 explicitly declines to test the electron–ion coupling reasoning.

**(b) Novelty.** The authors state that all components are standard and that the contribution is "application-specific and empirical". Virtual-diagnostic, missing-data and super-resolution work in fusion (refs [12], [15]–[18]) is close, and the novelty is incremental. The most relevant comparator is a real-time spectrum reader on the same discharges (ref [5], or UFCES). It was not run, so the practical value cannot be ranked.

**(c) Soundness.** The validation is careful:
- cluster bootstrap;
- four designs, including chronological;
- permutation and anchor-row controls;
- independent recomputation.

Its weaknesses:
- **Self-referential target.** The target is the same noisy CES output that forms the history, with no fit errors.
- **Weak registration.** "Registration" occurred on data already read, with inputs fixed on the same discharges (§5 Limits, "Selection").
- **No multiplicity correction** across 170 registered statements (Table S33).
- **Few independent units.** The chronological test has 14–19 clusters.
- **Intervals condition on the fits**, so seed variability (Table S15: a sign change under S) is excluded.
- **Simulated availability rules.** S and Dd are assumptions about an analysis schedule, not measured latencies. The 0.65 s age and the zero analysis time are hypothetical (§2.2).

**(d) Conclusions vs evidence.** The conclusions are mostly hedged, but several summary sentences overstate the supporting tables (see §8).

**(e) Clarity.** This is the weakest aspect.
- The prose is idiosyncratic and dense, with many "is not established" clauses.
- The main text runs to 17 pages, with 26 pages of supplement and 36 supplementary tables.
- The label TX means two different models (§3.2 vs §4.7, Tables 9–11).
- Row sets change from table to table (Table 1: 635/639/641; abstract: 627; the D10 increment is 10.3 in Table 5 but 7.5 in Table 4).
- The TCN section (§4.7) reads like auto-generated text, uses fractions where the rest of the paper uses per cent, and adds little.

**(f) Reproducibility.** Code and MAST data are public, but KSTAR data are not redistributable, so the KSTAR results cannot be reproduced by readers. Default hyperparameters and the fixed seed are documented, which helps.

## 3. Major concerns (ranked)

**1. The target may encode analysis artefacts that the fast channels help predict.**
- *Evidence:* Fig. 7b and Table S26 show the reported Ti dropping to about 0.76–0.84 of its level in the first row after a beam notch, while BES (beam-dependent light) also drops. This is likely a background-subtraction or beam-return artefact, not a plasma Ti change. Table S18 shows the largest D650 increments within 100 ms of a notch (45.2 %). BES and ECEI levels also co-vary with density and beam power, and part of the gain is non-local (Mirnov, Table S19).
- *Resolution:* (i) report all main results excluding rows within, say, 50–100 ms after notches; (ii) regress fast inputs against beam power and a density proxy and use the residuals; (iii) temper the language that "fast diagnostics at the location" inform Ti.
- *Effort:* new analysis of existing data.

**2. There is no independent reference, and the practical benchmark is missing.**
- *Problem:* Every skill is relative to the same CES analysis. The decisive comparison — a spectrum reader on the same frames (ref [5]), or the actual delivery latency of the KSTAR analysis — is absent. The S and D650 conditions are hypothetical schedules with zero analysis time.
- *Resolution:* obtain real delivery timestamps or reader outputs, or reframe the paper as a methods study of a hypothetical latency scenario.
- *Effort:* new data for the first option; rewriting for the second.

**3. The headline gains in time order are fragile.**
- *Evidence:* Against the offline GP, the result is not established under V3 or V4. The fresh-history increment on L5 is not established under V3. Under S the result fails for the last block, with the sign depending on the seed (Table S15), and the 32310 fit is −25.4 %. With 14 clusters and no multiplicity correction, the "established" chronological results (e.g. 1.9 [0.1, 3.6]) are marginal.
- *Resolution:* add fit-variance-inclusive intervals (refit over seeds and bootstrap jointly), apply a multiplicity correction or reduce the statement set, and give the V3 or V4 numbers equal prominence in the abstract.
- *Effort:* new analysis of existing data.

**4. The radial location of the fast channels is uncertain, and the paper does not claim specificity.**
- *Evidence:* ECEI position is uncertain by 2–3 cm and may be optically grey (Table S1). BES columns are inferred. The chord could be at r/a ≈ 0.92 (S1).
- *Implication:* The title and abstract stress "fast diagnostics at the pedestal top" and "at the same location", yet §4.2 concedes no radius specificity.
- *Resolution:* retitle and reframe.
- *Effort:* rewriting only.

**5. Length and presentation.**
- *Resolution:* cut §4.7 and Tables 9–11 to one paragraph, or move them to the supplement. Rename the TCN models. Unify the row sets. Reduce the main text to around five key tables. Rewrite in conventional prose.
- *Effort:* rewriting only.

**6. Weak MAST and KSTAR comparability.**
- *Evidence:* MAST Thomson is not faster than CXRS (§3.6). The pedestal band is defined by the same Thomson profile used as input, and the MAST data had already been read.
- *Implication:* This supports "direction" only weakly.
- *Resolution:* state this clearly, or define the pedestal band independently, for example from the EFIT separatrix.
- *Effort:* new analysis of existing public data.

## 4. Minor concerns

- **Row-set bookkeeping.** Table 3 TX F RMSE is 174.1, while Table S32 gives 174.8; T0 is 182.2 vs 182.9. Explain the different row sets.
- **Inline citation.** "Jang et al 2021, doi…" in the introduction should be a numbered reference; it appears as [22] only in the supplement.
- **References.** Several references are dated 2026–2027 or appear as an arXiv ID (2610.00243); verify the bibliographic details.
- **Value-repeat rule.** Clarify how often the "value repeats to last digit" rule removes genuine values when Ti is quantised.
- **Fig. 1b.** The histogram binning shows artefacts at 2 s; explain the pile-up.
- **Timing results.** Table S13 (1.95 ms median) comes from a shared machine and one unit only; give hardware details.
- **Table S11.** The anchor scalars hurt in V4 D10 (TX8 is better than TX); comment on this.
- **Regression tuning.** "Default settings, without tuning" for HistGradientBoosting — justify this, or show a sensitivity analysis.
- **Recovered share φ.** Define φ for the case where the D10 MSE is close to the ρ MSE; the intervals exceed 1 (Table S7, V4 D100: 0.77 [−0.02, 1.35]).

## 5. Recommendation

**Major revision.** The statistical care, transparent negative results and independent recomputation are commendable. However:
- the scientific payload for PPCF is small and partly confounded by possible analysis artefacts at beam notches;
- the target is self-referential;
- the operational scenarios are hypothetical;
- the key time-ordered results are marginal;
- the presentation is unusually hard to read.

A focused, much shorter paper with artefact-controlled analyses and clearer framing could be acceptable. Fusion Engineering and Design may be a more natural venue.

## 6. Probabilities

- **(i) Sent to referees by PPCF: 60 %.** The topic is in scope and the rigour is evident, but the editor may judge it engineering- or statistics-heavy and hard to read.
- **(ii) PPCF accept: 30 %.** Limited physics significance and no independent reference cap this even after revision.
- **(iii) FED accept: 55 %.** The real-time estimation focus fits FED well, and FED tolerates incremental applied work.
- **(iv) RSI accept: 30 %.** RSI focuses on instrumentation. A pure software estimator without new hardware or real-time demonstration fits less well, though RSI does publish diagnostic-analysis papers.

## 7. Changes that would raise (ii) most

1. Notch-excluded and beam- or density-residualised analyses showing that the fast-channel gain is not an analysis artefact or a beam-power proxy. *New analysis of existing data.*
2. Reframing around the actual delivery latency or a real-time reader comparison. Without new data, this means explicitly recasting as a scenario study, shortening the paper drastically and fixing the title and claims. *Rewriting only*, or *new data* for the stronger version.
3. Fit-variance-inclusive intervals with multiplicity control, and the time-ordered designs as primary. *New analysis of existing data.*

## 8. Errors, inconsistencies, overstatements

- **Abstract vs Table 9 (TCN).** The abstract says the stale-history TCN gain "is established under the random design only in both of its evaluation scopes". Table 9 shows V2 S/L5 = +0.194 [+0.007, +0.339], which is established under the cluster design in the fixed-subset scope.
- **§4.2, anchor-row control.** "The estimator with the present fast samples has a lower MSE than the one given the fast samples of the anchor row under every design" — Table S17, rule S, V4: −1.1 [−40.8, 17.1] and −12.0 [−44.3, 13.2], not established.
- **§4.2, beam waveforms.** "with 19 beam scalars … still reduce the MSE under all four designs" holds only for D650. Under S, V4 gives 10.2 [−15.6, 24.2] and −6.2 (Table S16; N7 does not hold).
- **§4.2, "Which diagnostic".** Table S16 shows ECEI-only also failing under V4 S, and BES-only failing under V4 at D650 (32310 fit). The statement is simplified.
- **Table S16 vs §5.** §5 says "ECEI half alone has an increment under three of four designs with history 0.65 s old". This is correct only for the V4 fit to 31991 and V1/V2; V3 is not established. That makes three of four, but omits that both V4 fits differ.
- **Title and abstract.** "Fast diagnostics at the pedestal top" and "at the same location" are stronger than the evidence: there is no specificity (§4.2), a Mirnov share exists, and the chord position is uncertain (S1: possibly r/a ≈ 0.92).
- **Abstract, "lowest error that we found needs no neural network".** The two-stage network ties TX on F (Table S32: −0.1 [−1.4, 1.5]).
- **Labelling.** TX is used for two different models (Tables 9–11 vs the rest). This is error-prone despite the disclaimer.
- **Discharge counts.** Counts are reported as 641/639/635/627/613 in different places without a single reconciliation table in the main text.
- **Supplement "How to read".** It says batches B.73–B.84, but the batches run to B.86 (Table S33).
- **Interpretation of Table S5.** The test establishes that one-frame upward excursions co-occur with BES bursts. It cannot exclude that TX learns burst-related fit artefacts; this is not discussed in the TX context.

PPCF accept: 30 %; FED