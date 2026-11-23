Incomplete output; not a completed review.

# Dossier: causal representation, anchor-change and privileged-future teacher hypotheses for 10 ms BES/ECEI with sparse past CES

**Evidence tags.** [SF] fact read directly in a supplied source; [AC] source authors' claim; [AH] application hypothesis for this study; [UNK] not established.

**Citation limits.** The code inventory is unnumbered file text, so I cite repo@commit, path and function and give no line numbers. No official TS2Vec or generalized-distillation (GD) code was supplied, so their implementation details beyond the PDFs are [UNK]. No S4/Mamba source was supplied; nothing below is evidence for or against them.

**Status.** Everything proposed is a design for registration before any fit or readout. Because the campaign is reused, outcomes remain TEST-informed exploratory.

## 1. What the sources establish

### 1.1 TS2Vec (PDF, 20 pp.)

- **Architecture [SF].** Linear input projection, then latent timestamp masking (Bernoulli p=0.5), then ten residual dilated-conv blocks. Two overlapping random crops are drawn, and the positive pair is the same timestamp seen in two contexts (p.2 §2.2; p.3 §2.3). Hidden width 64, kernel 3, output 320, batch 8, lr 1e-3 (pp.11–12 App. C.2).
- **Loss [SF].** Temporal contrast uses other timestamps of the same series as negatives (Eq. 1). Instance contrast uses other series in the batch at the same timestamp as negatives (Eq. 2). The two are summed (Eq. 3) and applied hierarchically through max-pooling over time (Alg. 1, pp.3–4).
- **Causality [SF/UNK].** The PDF never states that the convolutions are causal, and the training objective is two-sided within a crop. Downstream use is end-of-window: forecasting regresses on r_t of the last timestamp (p.5 §3.2), and anomaly detection is streaming (p.5 §3.3).
- **Receptive field (derived).** With two convs per block, kernel 3 and dilations 2^0…2^9, RF = 1 + 4·1023 = 4093 samples. Indexing is ambiguous in the PDF, so it could be about 8.2×10³. At a nominal 10 ms grid this is at least 41 s, so the default TS2Vec encoder is already a long-context TCN.
- **Normalization and splits [SF/UNK].** Z-scoring is per dataset (p.11 C.1); whether the statistics are train-only is not stated. The representation model is fit on the training split (p.11 C.2), and the ridge α is chosen on validation (p.13 C.4).
- **Forecasting [AC, with SF qualifiers].** Univariate average MSE is 0.209 versus 0.338 for TCN (Table 2, p.5). The TCN is the authors' fixed-config reproduction (p.12 C.3), not budget-matched. In multivariate Table 8 (p.15) supervised models win at short horizons: ETTm1 H=24 gives TCN 0.324 and Informer 0.323 against TS2Vec 0.443; ETTh1 H=24 gives Informer 0.577 against 0.599.
- **Augmentation and backbone ablations [SF].** Adding jitter, scaling or permutation lowers UCR accuracy by 1.5, 1.5 and 3.3 points. LSTM and Transformer backbones lose 5.0 and 18.2 points (Table 5, p.6). These are classification-only results.
- **Missingness [AC].** Robustness is shown for random missingness on four UCR datasets (p.7 §4.2). This is not MNAR and not regression.
- **Position [SF].** Random cropping is designed to make representations position-agnostic (p.11 App. B.2).

### 1.2 Generalized distillation (PDF, 10 pp.)

- **Setup [SF].** Triplets (x, x\*, y) with x\* unavailable at test (p.1). The teacher is fit on (x\*, y); the student is fit on hard and teacher targets with imitation weight λ (Eq. 4, p.3; three steps, p.4 §4).
- **Regression [SF/UNK].** The exposition is classification, with the statement that the ideas apply to regression (p.2 §2). No regression loss is written. How softmax temperature entered the SARCOS regression (p.8, Exp. 7) is not specified.
- **Theory [SF].** §4.1 (p.5) is an argument under assumed rates, not a theorem. Benefit requires small teacher capacity, lower teacher approximation error and a fast student-from-teacher rate. The authors say they believe the fast rate is general because soft labels carry more than one-hot labels [AC]. That argument does not transfer to a scalar regression target [AH].
- **The authors' own nulls [SF].**
  - Exp. 2 (p.7), clean features as privileged with x = x\* + independent noise: teacher 90%, student 68%, distilled 70%, reported as not significant.
  - Exp. 4 (p.7), sample-dependent relevant features: 96 / 55 / 56.
  - Exp. 6 (p.8): distilling only on the 300 labelled samples gave no gain; the gain came from teacher labels on unlabelled data.
- **Positives [SF].** Exp. 1 (noisy labels, privileged margin): 96 / 88 / 95. Exp. 3 (globally relevant features): 98 / 89 / 97. Exp. 5: MNIST gains shrink with sample size (p.8).

### 1.3 Supplied official code

**CRU** (boschresearch/Continuous-Recurrent-Units@79723e7)

- `lib/CRUCell.py::RKNCell.get_transition_model` builds the transition as softmax-weighted learned basis matrices. `RKNCell._update` uses residual = latent observation − upper half of the prior, with gain q = cov_u/(cov_u + obs_var), and reverts to the prior where `obs_valid` is false. No user-supplied f or h appears [SF].
- `CRUCell._predict` uses `matrix_exp(A·Δt)`, with Δt taken from `time_points` in `lib/CRULayer.py::CRULayer.forward`. The loop is forward-only, so it is a filter and not a smoother [SF]. On a regular 10 ms grid Δt is constant, so the continuous-time machinery distinguishes CRU from RKN only if report arrival is treated as irregular [AH].
- `lib/CRU.py::CRU.forward` passes only `obs` to the encoder. `obs_valid` is per timestep, not per channel [SF]. `lib/models.py` and `lib/encoder.py` were not supplied, so actual encoder inputs are [UNK].
- Hazards that must not be copied [SF]:
  - `lib/physionet_preprocessing.py::normalize_data_and_save` and `lib/ushcn_preprocessing.py::cleaning_after_split` compute min/max and outlier thresholds separately per split file, so test is scaled by test statistics.
  - `lib/data_utils.py::load_data` uses the test file as `valid`.
  - `CRU.__init__` sets `_enc_out_normalization = "pre"`.

**RKN** (ALRhub/rkn_share@7cb29e1)

- `rkn_cell/RKNCell.py` has the same learned locally-linear transition, and `_masked_update` handles invalid observations [SF].
- `rkn/Encoder.py::Encoder.forward` L2-normalizes the last hidden vector ("pre") or the mean ("post") [SF].

**KalmanNet** (KalmanNet/KalmanNet_TSP@828a2cf)

- `KNet/KalmanNet_nn.py::InitSystemDynamics` and `step_prior` call `self.f` and `self.h` [SF].
- `step_KGain_est` L2-normalizes all four difference features [SF].
- `Pipelines/Pipeline_EKF.py::NNTrain` minimizes MSE against dense simulated state targets [SF]. The `CompositionLoss` branch calls `torch.unsqueeze` without a `dim` as supplied, so it is not a validated unsupervised path.
- This matches the stated constraint: the original KalmanNet needs f and h plus dense state targets, and this study has neither. CRU and RKN need neither.

**Diag2Diag** (Zenodo 16748363, `diag2diag_natcomm_model_train.py`)

- A per-sample MLP (236→512/256/128→80, L1 loss) with no temporal context [SF].
- `extract_features` aligns to sparse Thomson times with `reindex(..., method='ffill')` and stacks targets ± measurement error as augmentation [SF]. This study has no per-frame uncertainty, so that augmentation is unavailable.
- `stats` is hard-coded with duplicated placeholder entries; its provenance is [UNK]. `calculate_derivatives` uses `bfill`, which is non-causal at segment starts [SF].
- `generate_datasets` as supplied feeds `np.random.randn` dummies, with real loading commented out [SF].
- The file therefore documents a pattern (train a snapshot map at sparse label times, apply it densely) and contains no high-rate validation.

## 2. Three hypotheses

**Notation.** Shot s, 10 ms index k. x_k ∈ ℝ¹³ is the past-only-standardized BES9+ECEI4 snapshot. j(k)<k is the latest admissible CES anchor, with value a_k, age τ_k and flag f_k. The target is y_k ∈ {F_k, L5_k}. P_k is all strictly-past CES plus x_{≤k}. ĝ_k is the causal GP mean.

**H-R, causal representation.** An encoder pretrained without labels on TRAIN-shot fast sequences, including grid points with no CES report, yields z_k = g_θ(x_{k−W+1:k}) such that

ŷ_k = ĝ_k + h(z_k, a_k, τ_k, f_k)

beats the same architecture trained supervised from scratch at matched budget, and also beats the existing explicit-feature tree. TS2Vec's encoder is a dilated TCN, so H-R concerns the objective and the unlabelled data, not a new model.

**H-A, anchor/current change.** Fast signals inform the increment since the anchor, not the level:

ŷ_k = ĝ_k + h(z_k − z_{j(k)}, τ_k, f_k).

Both snapshots are standardized with statistics frozen at time j(k). Otherwise drift in the expanding standardizer masquerades as plasma change. H-A fails if replacing z_{j(k)} with z_{j′} from another time in the same age stratum leaves error unchanged.

**H-T, TRAIN-only privileged-future teacher.** Take x\*_k = (x_{k+1:k+M}, CES reports outside an exclusion set E_k, two-sided GP state). A cross-fitted teacher t_k = T^{(−fold(s))}(P_k, x\*_k) supplies targets to a past-only student s_θ(P_k). The exclusion set is E_k = {k} for F and {k,…,k+5} for L5. Without it the teacher reproduces the label.

## 3. Pretraining-time future context versus test-time leakage

**Allowed (TRAIN shots only).**

- Two-sided crop context inside the SSL loss.
- Teacher inputs x\*.
- L5 labels, which are already a future-derived smoothed label.
- Full-shot statistics for the teacher only.

**Leakage.** Any of the following counts:

1. An encoder or student forward at k that touches fast samples beyond k or CES at or after k.
2. An anchor value built from L5_j when j+5 ≥ k. Anchors should be raw reports.
3. Statistics from VAL/TEST or from the future of the same shot. Bad patterns in the supplied sources: CRU's per-split scaling, TS2Vec's unspecified per-dataset z-score, Diag2Diag's `bfill`.
4. Normalization or running statistics that update at evaluation.
5. Instance pooling over a whole sequence.
6. Windows that cross shot boundaries.
7. In any within-shot temporal design, teacher or L5 windows that reach across the split. This needs a purge of at least M+5 samples.
8. SSL pretraining on VAL/TEST fast signals. This uses no labels but is transductive, so it would be a separate labelled arm and never the headline.

**Residual mismatch that is not leakage.** A two-sided-pretrained encoder deployed on windows ending at k never sees right context at the scored position. A causal-padded variant removes this mismatch. The choice must be registered and not selected on TEST.

**Disjointness.** Encoder pretraining, teacher folds, student, heads, HPO and early stopping use TRAIN, with internal VAL carved from TRAIN shots. VAL is for selection. TEST gets one readout, in each of the four validation designs, reported equally. The student never receives a teacher output computed on VAL/TEST inputs.

**Normalization.** Use the existing past-only expanding standardization, or constants frozen from TRAIN. The registered mechanical check is a perturbation test: altering all fast inputs after k and all CES at or after k must leave the output at k bit-identical.

**Availability caveat.** Actual CES acquisition, analysis and transfer times are not established. "Strictly past on the nominal grid" is therefore an assumption, and D source-age rules are not forecast horizons.

## 4. Can sparse report supervision support a meaningful teacher?

**Population identity (my derivation, not in GD).** If t = E[y | P, x\*], then E[t | P] = E[y | P], so the infinite-sample student is unchanged. By orthogonality,

Var(y − E[y|P]) = Var(
