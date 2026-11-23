# Adversarial review: candidate programs for the pedestal-top CES nowcast

Nothing was run or fitted. Tags: [S] supplied source or packet fact, [H] my hypothesis, [U] unknown. Derivations are mine.

## 1. Verdict

No new learned model is justified yet. Three things block interpretation of any new fit:

- **The packet does not address the manuscript's question.** Every paired number is fast-enabled TCN (TX) versus history-only TCN (TH). There is no paired TX or TH contrast against hold, causal GP, or GP+boosted trees on identical rows [S: "TX labels are NOT the manuscript GP/tree estimator"; baseline supports may differ].
- **The sign convention and metric of the paired values are unverified [U].** If they are relative MSE skill, V3 D10/L5 = −0.81 is a very large degradation and deserves a row-level check before any mechanism is proposed.
- **Budget confounds everything.** 38 of 46 confirm fits were budget-limited under a wall-clock cap, and B104 owns that question and has not read out.

## 2. Actual errors in the candidate programs

**E1. The A2 delayed update double-counts.** The rule uses the stored pre-update decode, ν = y_s − ŷ_s, then adds α^{t−s}Kν.

- Under a D rule with several reports pending (my reading of the rules [U]), that stored decode ignores corrections applied since s.
- Counterexample: α=1, K=1, constant state error e, D100, ten pending reports. Each arrival applies +e. After the first the error is 0; after ten it is −9e.
- Correction: recompute the innovation against the corrected source-time state, or filter at source time and propagate open-loop to t. `model98.py::matrix` already does the latter in linear form [S].

**E2. The "matching variance reduction" is not diagonal.** A scalar report observed through a vector h gives a rank-one posterior. Diagonal P with K = P h/(hᵀPh + r) is an approximation. RKN avoids this only through its factorized observation structure [S: RKN §2.4]. Use one latent per observed scalar, or label the approximation.

**E3. A2's gain is a schedule, not adaptive.** If fast inputs enter only as drive and never touch P, then P and K are deterministic in age and flag. The mean is then a B98-type forced first-order predictor with a nonlinear drive. The stated distinction from B98 reduces to drive nonlinearity and latent dimension.

**E4. A1 does not cancel offsets.** φ(x_t) − φ(x_s) with nonlinear φ (13→32→16) is not offset-invariant.

- Counterexample: tanh(x_t+c) − tanh(x_s+c) depends on c.
- Cancellation requires φ(x_t − x_s) or a linear φ.
- Level removal was already exercised in B84 TXrel, and B98's comparator L is already a static anchor difference. A1 is a nonlinear L, not a new contrast.

**E5. γ(0)=0 is vacuous and biasing.** Anchors are strictly past, so age is never 0. A monotone γ imposes shrinkage toward the GP, which manufactures the short-age "repair". The single-switch test needs a γ≡1 arm.

**E6. The TS2Vec code locator is unsupported.** The candidate cites `ts2vec.py::encode` defaults as [S], but dossier 04 states no official TS2Vec code was supplied. Downgrade to [U].

- What is established: the PDF loss uses two-sided crop context and never states causal convolutions [S: pp.3–4].
- A causal-padded encoder is applicable [H].
- Any sequence-level normalization or pooling on the scored path is unknown, so the prefix-perturbation test is mandatory, not optional.

**E7. FULL15 and FOUR are not independent replications.** V4a FULL15 values equal FOUR seed 1 to about 1e−15 (D10/F −0.19300962…) [S]. "FOUR repeats the V3 sign" is a nested re-read, not confirmation.

**E8. Conditional-selection leakage in Program A's readout.** The short-age deficit that motivates A is strongest in FULL15 V2 (D10/F −0.37). In FOUR V2 it is −0.06 [−0.25, 0.06]. Reading out on FOUR, a subset of the same reused TEST, tests a pattern selected elsewhere on that TEST. Regression toward zero would look like a fix.

**E9. Program C restates the table.** D is a known constructed rule. A selector on age and flag is a lookup of the observed TEST sign pattern. Its "always-GP" arm also has no paired counterpart in the packet.

**E10. The B98 summary omitted harm.** "At most about 3%, often spanning zero" leaves out V3, the only unit with zero train/VAL cluster overlap. There M is worse than H at S (−0.38, interval excludes zero) and far worse in point estimate at D650 and D1000 [S]. With 27/28 and 23/23 cluster overlap, V1 and V2 are closer to in-cluster fits. Internal VAL only; no transfer to TEST.

**E11. Unfair pilot compute.** In roughly 600 s, T received 603–996 updates against 313–520 for A and C [S]. Selecting T was partly an updates-per-second selection. A and C are not shown inferior at matched updates.

## 3. Requested checks

**Source-time update versus B98 and B88.**
- B98 already supplies the source-time-consistent linear comparator: strictly earlier source index, exact exponential propagation, forcing referenced to source time [S].
- B88-A supplies the concatenation of current and anchor embeddings [S].
- B90 R/C supply the every-row learned-gain observer [S]. They are unrun, so there is no verdict on them.
- What no existing item isolates: a recomputed-innovation event update with one switch against its every-row twin, and fast-only versus NBI-only forcing. B98 M adds both together [S].

**Standard NLL versus faithful.**
- B90-N's joint NLL scales the mean gradient by 1/σ̂² and so changes the mean [S: Seitzer Eqs. 3–4]. Faithful Eq. 5 does not [S: Stirn Thm 1, p.4].
- The identity additionally needs update-count stopping, identical batch order, no cross-head clipping and MSE-based checkpointing [H]. The current 1200 s cap violates the first.
- Missing control: "faithful versus age×flag table" is beaten by any head that reads anchor value or recent report scatter. Counterexample: report scatter proportional to level. σ̂ wins with zero fast information.
- So a history-only variance head (B2) is required, and the fast-input question is head(fast) versus B2.
- TRAIN in-sample residuals understate variance for every arm. Apply the same internal-VAL quantile recalibration to all arms, or scale error dominates the interval score.

**Current-anchor teacher information.**
- A teacher that sees F_k, or an interpolant with a node at k, reproduces the label. T3's PCHIP+GP target must be leave-k-out, which is not specified.
- A teacher that sees k±1 is a leave-one-out smoother. That is an L5-like relabel already present, and it inherits serially correlated report error.
- A "fresh-anchor teacher for a stale-anchor student" is past-only but approximately hold, i.e. label plus noise.
- The only non-duplicated privileged content is future fast snapshots. B87 excluded these [S].
- Pseudo-labels at R=0 rows change the training population while scoring stays on R=1 rows. Under MNAR that can hurt.
- The E[t|P]=E[y|P] identity holds only for an exact conditional-mean teacher.

**Applicability versus novelty.**
- RKN and CRU are applicable without known f/h; original KalmanNet is not [S].
- Faithful loss, TS2Vec and GD are applicable.
- None is novel. Priority for this use is unknown: three fusion papers and no search settle nothing either way.
- A corrected source-age-aware update could still be a genuine algorithmic element. That is neither claimed nor excluded.

## 4. Inference rules to enforce

- **Tests.** Differences are tested as paired shot-cluster contrasts, never by overlap of marginal intervals. The B98 "L has lower MAE" statement is read from marginal numbers, not a paired contrast.
- **Nulls.** An interval spanning zero is "not resolved". FOUR V1 per-seed spread at D10/F (−0.15 to +0.04) is as large as the effects discussed.
- **Margins.** No margin has an external justification yet. Until one is fixed, report estimates and intervals only, with no "gain", "noninferior" or "equivalent" labels. Admissible anchors are a stated operational tolerance or a pre-registered fraction of the paired hold-versus-GP gap. "Noise level" is not admissible, because it is unidentified.
- **Supports.** F and L5 rows differ (27,566 versus 26,669 in FOUR V1). F_full, K10 and S differ in rows and clusters. Any F/L5 statement needs joint-eligible rows and its own estimand. No cross-design cluster intersection.
- **Noise.** No reading of σ̂, of F-minus-L5, or of F-only versus L5-only increments as fit noise, plasma variability or truth.
- **Metric.** B98 shows MSE/MAE discordance: at V1 D1000, M has far lower MSE than hold but higher MAE (211.7 versus 208.5 eV) [S]. Register both metrics. This is a reason to keep the deferred robust-loss families (B90 U/V) alive, not evidence about them.

## 5. Corrected minimum comparison (register before any fit or readout)

**Stage 0, no fits.** From stored predictions [U whether they exist with row identities]:
- Confirm sign and metric.
- Build paired TX, TH, hold, causal GP and GP+boosted contrasts on exact common rows, per design, rule and target.
- Report MSE and MAE, common-support and selection-population estimands, and eligibility rates.

**Stage 1, linear and CPU.** Extend the B98 family to outer TEST, for F and L5, in all four designs:
- Arms: H; M with fast-only forcing; M with NBI-only forcing; M with both; L.
- Controls: the existing permutation control applied to the fast forcing.
- Linear fits converge, so budget and seed confounds largely drop out.
- It is a new experiment on reused TEST and stays exploratory.

**Stage 2, only if warranted and after B104.** One nonlinear model with single-switch twins:
- Recomputed-innovation event update versus hold-filled every-row update.
- γ≡1 versus gated γ.
- φ(x_t − x_s) versus concatenation.
- Scalar latent, matched updates and a registered convergence criterion. Arms that do not meet it are labelled budget-limited and not compared.

## 6. Ranking

**1. Paired baseline crosswalk plus CES provenance audit (Stage 0; dossier 03 §5).**
- Value: it answers the framing question that the packet cannot, and it fixes E7 and E8 bookkeeping.
- Promote: it becomes the main results table if row identities exist.
- Abandon as a no-fit item: if stored predictions lack row identities. It then folds into Stage 1 as registered refits.
- The audit half is abandoned only if the diagnostic owners cannot supply timing and missingness records. In that case every claim stays "nominal-grid".

**2. Reduced Program A: B98 family on TEST with a fast/NBI split (Stage 1).**
- Value: it is the cheapest test of whether any fast increment beyond NBI exists under a source-time-consistent contract, before nonlinear machinery is paid for.
- Promote to Stage 2: if fast-only forcing exceeds both H and NBI-only on the paired contrast at the externally anchored scale in more than one design, permutation removes it, and B104 indicates the TCN short-age deficit is not a budget artifact.
- Abandon the nonlinear extension: if NBI-only matches fast-only, or the V3-type out-of-cluster harm reproduces. The latter would point to non-transfer of raw-voltage structure.
- An unresolved interval means "stop for lack of resolution", stated as such.

**3. Program B, corrected.**
- Value: it is the only program that adds a reliability statement at zero risk to the mean comparisons.
- Promote: if head(fast) beats the history-only variance head on the paired interval score after common recalibration. Promote the interpretation only if the audit supplies per-frame fit quality to associate with σ̂.
- Abandon the joint Eq. 5 form: if the identity gate cannot be met under update-count stopping. Fall back to a frozen-mean post-hoc head, labelled as the sequential variant.
- Abandon entirely: if B2 matches head(fast). Dispersion is then history bookkeeping.

**Not ranked.**
- Program C is a table restatement until Stage 0 defines its fallback arm.
- Program D is blocked by B104, and its teacher is near-degenerate. A single "future-fast-inputs" gate on the existing learner would be the only informative precursor.

Neither exclusion is a claim that the family cannot work. The unrun B90 families carry no negative evidence.
