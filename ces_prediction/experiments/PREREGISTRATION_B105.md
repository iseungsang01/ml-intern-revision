# B.105: estimator merit before further TCN expansion

2026-11-24, committed before any new scientific fit (CPU included), CUDA smoke,
prediction or new comparative error/interval readout. Author explicitly requests
training and merit evaluation. This supersedes the planning-only execution state
of `docs/paper/tcn_merit_plan_20261124.md`, not the manuscript purpose. Existing
B94/B104 sources/results and all earlier registrations remain immutable.

## Question, facts, expectation

Does one source-time-anchor-enabled TCN approach the deployed GP+boosted-residual
estimator under causal current-report nowcasting and delayed/sparse history?
The working agent incorrectly reported that no direct comparison existed: B94
already has paired TCNfast versus TREE_TX/TREE_T0 comparisons. Fast/history
increment is not incumbent-relative merit. B104 budget extension did not recover
the fast increment. B84 already has a single rule-mixture GP+tree (L5 only, no
V2), so one model across scenarios is not by itself a demonstrated advantage.
Prior TEST was inspected: ALL of this batch is exploratory, no independent
confirmation, model-adoption or architecture-generalization claim. Expectation:
uncertain, no assumption that extra anchor channels rescue merit.

## Fixed allocation and unchanged data treatment

Exactly four fresh production fits, previously fixed units V1_r2_k0, V2_s1_k0,
V3_q3,V4a, seed1701 each. No primary validation design; one unit/one seed per
design is a screening subset, not the full design or initialization stability.
Reuse source-bound B94 prepared35 TRAIN/VAL tensors, manifests and train-only
normalization; no redefinition of D1 or scored masks. Real observations only;
absence or a failed provenance assertion aborts. Explicit CES treatment is
`common105.env` (3keV cut, stuck targets removed, no per-shot normalization,
no expanding normalization, MC/rotation zeroed). Training segment and batch
membership, seven-rule deterministic sampling, F/L5 supervision stay inherited.
F is the current report; L5 is the future-five-report median used only as an
offline label, not independent physical truth. D10...D1000 describe source-age
availability, not forecast horizons. S is the past observed-notch schedule.
K10 is excluded from fitting/VAL and only used in conditional TEST readout.
File-level disjoint splits prevent L5 crossing TRAIN/VAL/TEST file boundaries;
original cached label construction and same-segment invariants remain pinned.

## One controlled candidate

Same width128,kernel3,dilations1,2,4,8,16,32 causal TCN and original F/L5 heads.
Expose existing source-time13 fast inputs (cached columns22:35) instead of
slicing to22. Start from the seed1701 original TCN initialization; preserve
all original parameters, widen stem to35 with extra13 trainable columns zero,
preserve RNG state across widening. Check actual-batch initial function equality
at atol1e-5,rtol0 and nonzero finite real-data anchor-column gradients.
Current/history channel and source-time channel standardization both use the
same internal TRAIN files. No extra loss,NBI,residualization or backbone sweep.

Import original numerical train92 loop via new namespace/config/model adapter;
do not edit its source. Same batch16,AdamW lr.001,weight_decay.0001,clip1,
dropout.1,scheduler factor.5/patience2,earlystop6,40epoch cap. Only fully validated
epochs can supply best/latest. Soft3600s per production fit; older22 paths had
1200s, disclosed as a budget/information/parameter difference, not pure
architecture attribution. No added prefix checkpoint or second selection rule.
One prefix real-data one-epoch CUDA smoke precedes production. Native qualified
CUDA path, up to four global B105 workers. Before production, perform no-update
forward/backward memory probes on each unit's sixteen longest real TRAIN
segments. Reserve max(1200MiB,ceil(1.25*(maximum reserved MiB+768MiB))) per
worker, including validation/context allowance, and choose at most four slots
from free VRAM. Require total reserved slots fit free VRAM before dispatch;
this is virtual reservation, not ineffective immediate post-spawn sampling.
No other process is stopped. Inadequate memory/concurrency leaves the study
incomplete rather than extending its deadline or changing the model.

## Fair CPU references before GPU production

Original trees used all outer-fit files, including neural innerVAL; these
stored predictions MUST NOT select a new network. For each unit, train new
GP+HistGradientBoosting residual regressions on neural internal TRAIN files
only; fast-window and anchor standardization use exactly those files.
The reused GP input is optimized per row from at most16 visible CES values,
not hyperparameters estimated from outer-fit/VAL targets (source checked).
Specialized7rules x2heads x4units=56 CPU fits. Two rule-mixture regressions
per unit=8 further fits; total64. Fixed random_state20261116,max_iter100,
early_stopping=False for BOTH references, other sklearn defaults. This differs
from old automatic early stopping and is explicitly a new fair reference.
No row-random internal earlystop split of duplicated scenario-mixture rows.
No rule-ID input or new file weighting. Mixture observation weight is
N_total/(7*n_rule), equal total weight per rule and mean sample weight1.
Specialized trees use common target masks (S uses S mask); neural training
retains all original masked rows. Per-unit/rule/head fit and VAL row counts
are stored, not claimed identical training-row support. Models and hashes saved.
Only VAL predictions before gate; no reference TEST prediction before gate.
Specialized K10 is N/A; mixture, incumbent condition-trained K10,GP,hold may
be evaluated there, clearly distinguished. No post-hoc specialized remapping.

## Checks and execution order

Committed registration/source -> immutable preparation -> independent real
input/source/model checks -> CPU references (four workers,two OMP/MKL/OpenBLAS
threads each) -> real CUDA smoke with normal exit -> four production fits ->
all normal-exit,source/checkpoint/config and independent validation-arithmetic
checks -> common-row VAL predictions/gate -> conditional TEST -> readout ->
separate independent raw-prediction/interval/gate audit -> publish -> bounded
official Claude post review. Failed inputs/fits are not silently retried.
All artifacts are under data/.b105; no old clocks or outputs are overwritten.
One new suite clock150min from registration launch, CPU fits hard45min each,
GPU production hard4020s each including process startup/preparation. No extra
GPU dispatch unless at least5820s remain, reserving1800s for initial readout
and audit work. Stop NEW dispatch when this fails; active valid fits may finish.
Remaining-time checks and step deadlines bound the whole study. If these are
insufficient, report incomplete; no automatic broadening or extra fits.
Independent local15s monitor, phase/failure and5min progress notifications;
actual notifier CLI acceptance required before suite dispatch. Local UI
authentication config uses current-Windows-user DPAPI encrypted bytes, stays
ignored and is never printed/API-sent/committed. No plaintext UI credential file.

Input checks independently recompute TRAIN statistics and source-time gather,
strictly preceding source indices with same file/segment, missing-source zero
handling/finiteness, original22 slice from raw channels, stream/full/prefix
model parity and save/reload via smoke, labels-not-inputs. Existing frozen
source grid/rows/cache hashes bind original target and unavailable-source
mutation checks. Any commonVAL cell with <30 rows,<2 files,<2 clusters,NaN,
undefined ratio,missing CPU reference or failed fit is an invalid screening
study (no candidateTEST), not a scientific negative. CPU fit failure or
timeout likewise stops dispatch; partial outputs remain evidence.

## Fixed allocation gate (not inferential noninferiority)

Each model's best checkpoint still minimizes original equal-weight7rule x2head
VAL loss on its original support. Then use EXACT same target/rule commonVAL
rows as fair specialized tree. One shared TRAIN Ti scale per unit is inherited
for both heads. For model A define Q_A=mean over14cells(MSE_A/ti_sd^2).
R_unit=Q_TCN35/Q_fair_specialized. Gate passes iff R_unit<=1.05 for EVERY
fixed unit. Ratio of means, not mean of ratios. Also display TCN22 and fair
mixture scores, and F/L5 separated values. Threshold5% inherited from earlier
operational screening; not a measured physical tolerance. Checkpoint selection
on VAL gives the candidate an optimism advantage over non-VAL-selected trees.
Gate passage is a compute-allocation decision only. No new candidateTEST
prediction at all if any unit fails; report failed gate and all VAL comparisons.
All four production fits finish before gate/scoring; no performance-based fit
reallocation. No new seed or hyperparameter to rescue a failed gate.

## Registered readout and statements

Always report same-common-row TCN35 versus fair specialized and fair mixture
on seven VAL rules, F/L5, every unit. These are development/selection-affected
results. Also newly pair EXISTING22 TEST predictions with every available B84
TXall L5 prediction (all8rules,V1/V3/V4 subset); V2/F missing is reported, no
imputation and no claim of previously unseen endpoints. Specialized old trees
and full15/three-seed B94 evidence remain published separately, not pooled.

If and only if gate passes, freeze candidate, evaluate all8rules x2heads on
the four units' outerTEST. Compare with fair mixture, fair specialized except
K10, original incumbent outer-fit TREE_TX, original22TCN, hold and causal GP.
Distinguish old outer-fit versus internal-TRAIN references, budgets, source
information and support. Old D10-applied estimators are not chosen as the sole
scenario reference. Legacy LSTM+correction files exist but their fitted-stage
training manifests have not been independently bound to this contract: this
batch marks that comparison unavailable instead of restoring old scripts or
borrowing an unrelated GP/tree result as the LSTM comparison.

Per pair skill=1-MSE(candidate)/MSE(reference), paired on identical rows;
point,MSE/RMSE and conditional95% shot-cluster plus discharge percentile
intervals,20000 draws,one fixed initialization,no pooling across designs.
Generator seed sequence [20261105,CRC32(pair_name+'/'+resampling_form)].
Chunk500 multinomial group counts, all group labels in the actual pair cell;
the same weights apply to candidate and reference. No synthetic observations.
Undefined bootstrap draws abort publication; no redraw or silent omission.
Intervals omit refit,initialization,checkpoint/model-selection,multiplicity
and new-campaign uncertainty. No individual discovery claims; report all
registered cells, not a favorable subset. No latency advantage measured here.

S1: valid common-row allocation gate passes all four units. Expectation uncertain.
S2: if S1 passes, TCN35-vs-incumbent skill interval lower>0 in EVERY specified
TEST cell across all four fixed units (global accuracy increment); uncertain.
S3: conditional operational screening margin, skill interval lower>-.05 in
every corresponding TEST cell; uncertain. Even S3 passage does not establish
measured deployment benefit, which would require a separately registered
same-platform full-pipeline measurement. S2/S3 are conjunctions, no post-hoc
individual discovery selection or significance-change inference. Any wider
individual claims need a separate multiplicity procedure before new readout.

## Output, audit and paper sentences

New code b105_merit/{common105,model105,train105,trees105,checks105,evaluate105,
readout105,audit105,pipeline105,watch105,ui105}.py; review helper
research/review_b105.py (max4 official Fable attempts: design,initial incomplete
implementation review,focused completed review,post;
all attempts/usage preserved, no raw observations/identity/secrets). Advice
is not verified evidence. Public b105_merit.json/md,b105_status.md,b105_claude.md;
ignored provenance,fit checks,gate,predictions,pair specs and independent audit.

If gate fails: "In the exploratory fixed-unit screen, the anchor-enabled TCN
did not satisfy the prespecified common-validation allocation margin against
the internal-training GP-residual regressor; its outer-test evaluation was
not performed." If conditional TEST does not support S2: "The registered
screen did not establish an accuracy increment over the incumbent on all
specified availability rules and fixed validation units." If S2 supported:
"Conditional on the prespecified validation gate, the fixed-unit exploratory
comparison supported a positive reported-value MSE increment; initialization
and independent-campaign stability remain unmeasured." No architecture
inferiority,new physical mechanism,measurement replacement,novelty or journal
acceptance conclusion. A negative interpretation would change with an
independently specified adequate common-information/fit-budget comparison on
unused observations. No automatic15-unit or multi-seed expansion follows.
