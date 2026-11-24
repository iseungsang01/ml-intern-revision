# B.106: train the LSTM-plus-correction comparator and compare MSE directly

2026-11-24, before preparation, smoke, new GPU/CPU fits or new comparative
readout. Author requests the missing LSTM+post comparison WITH TRAINING.
Claude API is now unavailable by the author's instruction: no API calls are
made in this batch. Existing reviews may inform planning; numerical/source
checks are independent. Existing B105 code, registrations and results stay
unchanged. This is exploratory reused-data work, not independent confirmation.

## Scope and identity

Four fixed outer units inherited from B105: V1_r2_k0,V2_s1_k0,V3_q3,V4a.
Use exactly the B105 internal TRAIN, VAL and outer TEST file manifests for
the full network; seed1701, one initialization, not all-fold stability.
No primary validation design. Compare current CES report F only: the original
LSTM temperature branch predicts F, not the later L5 auxiliary target.
Existing TCN F outputs remain usable but their joint F/L5/mixed-rule objective
and budgets differ; this is a practical estimator comparison, not pure
architecture attribution. Do not redefine labels or restore deleted results.

The first stage is the original retained SeqCESLSTMv2 T_i path: two160-unit
causal LSTM layers,LayerNorm,64-unit GELU output head,dropout.1. Extract the
temperature modules from the actual retained implementation; the independent
rotation branch is not trained/evaluated. Input22 is the existing causal fast,
held-Ti,age,flag,time layout with Mirnov/rotation zero. Cached35 tensors may be
used, but the model slices22 and does not use the extra anchor-fast13.
Verify initial active-branch outputs on real data against SeqCESLSTMv2.

This is a fresh retraining of the historical two-stage METHOD FAMILY with
explicit modern D1/manifests/calibration rules, NOT an exact old-checkpoint
reproduction. Differences from B70/B64 are disclosed: current internal TRAIN
only for correction fitting (not added earlystop VAL); no active rotation loss;
fixed100-iteration correction without automatic row-random early stopping.

## Allocation and calibration split

Twenty GPU fits fixed before readout: one full network and four file-OOF
networks per unit. For each inner network, exclude its held files from both
training AND earlystop validation; the shared outer innerVAL is disjoint from
every held group. Four held groups partition the outer internal TRAIN exactly.
V2 groups entire shot clusters; other units use seeded file folds. This is
cross-fit residual calibration within available historical TRAIN, not a claim
of prospective temporal inner evaluation. Outer temporal TEST remains unseen.
Each inner network fits its own statistics using only its own TRAIN subset.
OOF predictions are converted to physical eV before training the correction.
No second-stage model fits in-sample predictions or outer TEST labels.

Train D10/F only, matching the old model's deployment condition. Evaluate
the frozen model under D10,D50,D100,D300,D650,D1000,S,K10: source age is not a
forecast horizon. No state is transferred across different rule replays or
original segment/file boundaries. All target-row CES measurements excluded.
K10 is held out of BOTH network and second-stage training.

Two post-corrections, both preregistered and reported without selection:
- frozen_D10: fit only D10 OOF residuals, then apply frozen across all rules;
- mixed_post: fit OOF residuals pooled across seven rules with equal total
  weight per rule, then apply frozen to all8 rules. This is a declared scenario
  extension; the first-stage network is still D10-trained.

Fifteen historical post features: eight fast-window summaries, time-step,
CES age,hold,first-stage-minus-hold,first-stage-minus-GP,and their absolute
values. Predict residual y-first_stage and add it back. All feature means/
scales use outer internal TRAIN only. Same original feature formula; complete
features required to fit, missing-feature prediction falls back to first stage.
HGB random_state20261116,max_iter100,early_stoppingFalse,other defaults. Mixed
weights N_total/(7*n_rule),no rule ID. CPU fits=two per unit,eight total.

## Optimization, budget, controls

Import frozen train92 numerical loop using a new config/model/data namespace
adapter. Same AdamW lr.001,weight_decay.0001,clip1,batch16,dropout.1,negative-Ti
penalty,scheduler factor.5/patience2,earlystop6. F-only mask; original L5 mask
is explicitly false and its values0. Best checkpoint is pure D10/F VAL MSE.
Cap80epochs/3600s soft per fresh trajectory; hard4200s including preparation.
The stop string converged denotes patience only, not optimization proof.
No tuning search or performance-based reallocation. Use qualified native CUDA
dense path and real longest16 TRAIN-segment backward memory probes. Reserve
max(1200MiB,ceil(1.25*(peak_reservedMiB+768MiB))) per slot,at most4 slots.
No other GPU process is stopped. Whole suite cap8hours after registration
launch, not a completion-time estimate. No new fit unless >7800s remain,
reserving1hour for initial calibration/readout work. Insufficient time leaves
an explicit incomplete study; no silent cap extension or partial merit claim.
External/numerical failures stop new allocation; no blind scientific retries.

Commit registration AND all execution sources before preparation/runs. Order:
prepare manifests/caches/reference hashes -> real branch/causal/OOF checks ->
real prefix CUDA smoke exit0 -> no-update full-batch memory -> all20 fits ->
fit/exit/source/validation arithmetic audit -> held-file OOF inference ->
second-stage fits -> outer TEST inference -> frozen second-stage replay ->
MSE-first report -> separate independent raw-source/correction/interval audit.
Source/manifest/config/checkpoint hashes and normal parent-observed exits
required. Do not modify frozen code while jobs run. Explicit CES pins in all
children; D1 cut3000eV,stuck targets removed,no per-shot normalization,MC/VT0.
No synthetic observations. Missing data or empty support aborts loudly.

## Common-row comparisons and statements

Every model in each unit/rule is scored on the SAME B75 common_F or S_F mask;
K10 uses B78 common_F. Report row/file/cluster counts, no global cross-design
intersection. Read-only references: incumbent GP+fast boosting TX,history-only
boosting T0,GP alone,hold,internal-TRAIN fair specialized/mixture GP references
from B105,TCN22 andTCN35 F outputs. Specialized fair K10 is N/A,not imputed.
The GP+fast model is MULTIMODAL, not CES-only. Its increment over T0 separately
tests fast-diagnostic added value; GP-vs-TX alone does not isolate that value.
All reference files hash-pinned before training. No new reference fitting here.

First show physical MSE(eV^2) and RMSE(eV),95% cluster/discharge intervals,
then paired skill=1-MSE(A)/MSE(B) with95% interval and zero exclusion. One
fixed initialization;20,000 cluster/discharge multinomial resamples per cell,
seed[20261106,CRC32(unit+'/'+rule+'/F'+'/'+form)]. Chunk500,same weights across
all models of a cell. Grouped sums optimize computation without changing
estimand; row-weighted point MSE. Undefined draws abort,never silently redraw.
Pointwise fixed-fit intervals omit selection/refit/initialization uncertainty.
All32 F cells reported. No individual discovery cherry-picking/multiplicity
claim; global statements are conjunctions across preregistered cells.

S1: frozen_D10 LSTM+post has lower MSE than GP+fast boosting in D10/F of EVERY
fixed unit (all cluster skill lower endpoints>0). Expectation uncertain.
S2: mixed_post has lower MSE than GP+fast boosting in EVERY scenario/unit
(all32 cluster skill lower endpoints>0). Expectation uncertain.
S3: GP+fast boosting has a positive increment over GP+history boosting in
D10/F of EVERY fixed unit (all four lower endpoints>0). Expectation uncertain;
earlier full-scope B73 results remain separate,not independent replication.
Report LSTM-alone/post differences and TCN comparisons irrespective of S1/S2.

## Outputs, independent audit and interpretation

New folder b106_lstm contains common/model/train,predict,correction,checks,
readout,audit,pipeline,watch modules. Pipeline entry prepare -> register.
Ignored evidence data/.b106; public b106_lstm.json/md,b106_status.md.
Independent audit reconstructs correction features/predictions,checks OOF
exclusion,raw reference bindings,common masks,MSE,all intervals and statements
without producer statistical helpers. Independent local15s watcher sends phase/
failure and5min progress; actual CLI acceptance required before launch.
Notifier credentials use Windows-user DPAPI encrypted ignored configuration,
never printed/committed/sent externally. No Claude/API review or post call.

Supported S1 sentence: "Under the fixed-unit exploratory common-row comparison,
the retrained LSTM-plus-correction had a positive current-report MSE increment
over the GP-residual reference; initialization/full-design stability remains
unmeasured." Unsupported sentence: "The retrained comparator did not establish
an increment over the GP-residual reference in every fixed validation unit."
If S2 unsupported: "Frozen scenario replay did not establish a consistent
MSE increment under all registered availability rules." No model-class
inferiority,new physical discovery,method originality or acceptance claim.
A negative conclusion could change after an independently specified adequate
budget/common-information comparison on unused observations. No automatic
further model search follows this comparison. Novelty scouting remains a
separate task and does not make this baseline retraining a new algorithm.
