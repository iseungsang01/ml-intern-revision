# Source audit: causal pretraining and a TRAIN-only future-information teacher for 10 ms CES nowcasting

**Tags.** [SF] fact read in a supplied source; [AC] source authors' claim; [AH] my application hypothesis; [UNK] not established.

**Limits.** The supplied code is unnumbered text, so I give repo@commit/path/function and no line numbers. No official TS2Vec or generalized-distillation (GD) code was supplied. Everything proposed is a design to register before any fit; outcomes stay TEST-informed exploratory. Neither idea is new: both are applications of published methods, and nothing here shows they work on these data.

## 1. Primary locators that matter

1. **TS2Vec PDF p.3 §2.3, p.4 Eqs. 1–3 [SF].** Positives are the same timestamp in two masked, cropped contexts. The loss uses context on both sides within a crop, and the PDF never says the convolutions are causal.
2. **TS2Vec PDF p.5 §3.2, p.11 App. C.2, p.13 C.4 [SF].** Forecasting uses the last-timestamp representation with a ridge head; the encoder is trained on the training split; ridge α is chosen on validation. Table 5 (p.6) reports that jitter, scaling and permutation augmentations hurt, for classification only.
3. **GD PDF p.3 Eq. 4, p.4 §4 steps 1–3, p.5 §4.1 [SF].** The teacher is fit on (x\*, y) and the student on hard plus teacher targets with weight λ. Benefit is argued under assumed rates, not proved. The "soft labels carry more than hard labels" argument [AC] is about classes and does not transfer to a scalar target [AH]. No regression loss is written (p.2 §2 only says the ideas apply).
4. **GD PDF p.7 Exp. 2 and p.8 Exp. 6 [SF].** With clean privileged features and independent noise, the teacher reached 90%, the student 68% and the distilled student 70%, reported as not significant. Distilling only on the 300 labelled samples gave no gain; the gain came from teacher labels on unlabelled data.
5. **CRU code, boschresearch/Continuous-Recurrent-Units@79723e7 [SF].** `lib/CRULayer.py::CRULayer.forward` is a forward-only loop, and `lib/CRUCell.py::RKNCell._update` reverts to the prior where `obs_valid` is false. This is an existing past-only filter pattern that needs no known f/h. Do not copy `lib/data_utils.py::load_data`, which uses the test file as `valid`.

## 2. TRAIN-time future context versus leakage

**Allowed, TRAIN shots only [AH]:**
- two-sided crop context inside the pretraining loss;
- teacher inputs from after index k;
- L5 labels, which are already future-derived.

**Leakage [AH]:**
- any scored forward pass at k that touches fast samples after k or CES at or after k;
- anchors built from L5 whose window reaches k;
- normalisation statistics from VAL/TEST or from the same shot's future;
- teacher outputs computed on VAL/TEST inputs;
- windows crossing shot boundaries;
- pretraining on VAL/TEST fast signals, which is label-free but transductive, so it could only be a separately labelled arm.

**Mechanical check [AH].** Register a perturbation test: altering all inputs after k must leave the output at k bit-identical.

**Caveat [UNK].** Real CES acquisition and transfer times are not established, so "strictly past on the nominal grid" is an assumption. Both hypotheses run entirely on the 10 ms grid; no full-rate input is needed.

## 3. What is actually privileged here

Unlabelled fast snapshots at grid points without a report are available online, so they are not privileged. They are the pretraining resource for H-R.

The genuinely privileged items are [AH]:
- future BES/ECEI snapshots x_{k+1:k+M};
- CES reports after k, outside the label's own window;
- the two-sided PCHIP+GP state.

L5 already injects future reports into the target. For an L5 student, the teacher's increment is therefore only future fast signals, reports beyond k+5 and the two-sided interpolant.

**Population identity (my derivation).** If t = E[y | P, x\*], then E[t | P] = E[y | P]. The infinite-sample student is unchanged. Any gain is finite-sample variance reduction from a less noisy target, or from extra pseudo-labelled points.

**The teacher is useless if any of these hold [AH]:**
- **Label leak.** It reproduces the label, which happens without an exclusion set ({k} for F; k…k+5 for L5).
- **No advantage.** Its out-of-fold error is no better than a past-only model's. The existing same-learner future-information latency curve bounds this, and I have not seen its values [UNK].
- **Unlearnable from the past.** Its advantage comes from future information that is unpredictable from past inputs (the Exp. 2 analogue).
- **Labelled-only.** It is applied only where labels already exist (the Exp. 6 analogue). The fraction of grid points lacking a valid F or L5 is [UNK].
- **Report noise dominates.** It inherits carbon-report noise or MNAR structure, so it denoises towards the report process, not towards bulk-ion truth.
- **Redundant with L5.** L5 or the two-sided GP already captures the available smoothing.

## 4. Experiment R: causal pretraining versus random initialisation

**Hypothesis [AH].** A causal-padded dilated-conv encoder pretrained with the TS2Vec objective on TRAIN fast sequences improves the fast-diagnostic increment over the causal GP, relative to the same encoder trained from scratch.

**Fixed in registration.**
- One receptive field. Varying it is B104's job and must not be duplicated.
- One objective, with no jitter or scaling augmentation.
- Existing past-only standardisation.
- The fixed FOUR3seed subset, kept distinct from FULL15.

**Arms.**
- R0: random initialisation, supervised.
- R1: pretrain, then fine-tune.
- R2: frozen pretrained encoder plus ridge head (the p.5 protocol).
- R3: frozen random encoder plus ridge head.
- R4: R0 given gradient steps equal to R1's pretraining plus fine-tuning.

**Readout.** Paired shot-cluster difference in error, in all four validation designs reported equally, for F and L5. State the multiplicity, and state that frozen-fit intervals omit fit and selection uncertainty.

**Registered negative result.**
- If R1 − R4 does not exclude zero in the registered direction, report "no detected pretraining benefit at this budget". This is not equivalence.
- If R2 ≈ R3, the learned representation carries no detectable label-relevant structure beyond random features.

**Known hazards.**
- Instance contrast (Eq. 2) uses other shots at the same timestamp as negatives, so it can encode shot identity or absolute level [AH]. Re-apply the existing absolute-level-removal control to R1/R2.
- TS2Vec's missingness evidence is random masking on classification sets (p.7 §4.2) [SF]. It says nothing about MNAR report targets.

## 5. Experiment T: TRAIN-only teacher versus controls

**Hypothesis [AH].** A cross-fitted future-information teacher supplies lower-variance targets that improve a past-only student.

**Design.**
- The teacher is K-fold by TRAIN shot. Inputs are past inputs plus x\* with the exclusion set applied.
- The student is past-only. Its loss is (1−λ)·ℓ(y, s) + λ·ℓ(t, s). This is my regression adaptation of Eq. 4, with no temperature.
- λ and M are fixed, or chosen on a VAL carved from TRAIN.
- Teacher targets are also used at TRAIN grid points lacking a label.

**Gate before any student fit.** The out-of-fold teacher must beat the out-of-fold past-only model on that internal VAL by a registered margin. If it does not, stop and report "privileged future information not exploitable by this teacher".

**Arms.**
- T0: no teacher, with matched steps.
- T1: privileged teacher.
- T2: self-distillation from a cross-fitted past-only teacher of the same architecture. This is the matched-compute control and isolates generic distillation and ensembling regularisation.
- T3: student trained directly on the existing two-sided PCHIP+GP target. This is a non-learned future target.
- T4: T1 with teacher outputs permuted within shot and age stratum.

**Readout.** As in Experiment R, with one TEST readout.

**Registered negative results.**
- T1 not better than T2: any gain over T0 is ordinary distillation, not privileged information.
- T1 not better than T3: a learned teacher adds nothing over the existing interpolant.
- T1 ≈ T4: the gain is not content-specific.
- T1 not better than T0: null at this budget, not noninferiority.

**Interpretation limit.** A win shows better agreement with future-smoothed carbon reports. It does not show better bulk-ion Ti, and it identifies no cause.

## 6. Unknowns that remain

- The share of unlabelled grid points and the values of the existing latency curve.
- TS2Vec's actual padding and normalisation implementation.
- How GD applied temperature in its regression example (p.8 Exp. 7).
- Whether either method has prior use on comparable fusion data. I have made no search, so absence of prior art cannot be inferred.
