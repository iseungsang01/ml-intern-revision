# B114 overnight: bounded VAL-selected scalar models

2026-11-26, author explicitly requests an overnight list and all listed local-GPU runs. Register/commit/push BEFORE smoke or new fits. This is a TEST-informed exploratory follow-up; existing B113 is completed and never retrained. Preserve paper framing; no claim that more runs must improve accuracy. The most recent user question is answered by keeping comparator and sign explicit: B113 CatBoost was favorable to some matched HGB controls but its D10 residual ensemble had negative GP+fast skill in all four designs, with cluster zero-exclusion in three.

## Fixed candidate listing and allocation

Use exactly the audited B112 TX numeric inputs, four original units (U0-U3; V1_r2_k0,V2_s1_k0,V3_q3,V4a), original internal TRAIN/VAL/TEST and same reported-current-frame F rows. D1 treatment unchanged. TRAIN and VAL contain D10,D50,D100,D300,D650,D1000,S equally weighted by rule. K10 is TEST-only. No new input, split, physical target, synthetic rows or TEST-driven candidate additions. Existing audited B113 depth6/1000/l2=3 is a saved reference only.

| Candidate | Change from fixed scalar baseline | Full fits |
|---|---|---:|
| Cat depth4 | depth4,1000trees,l2=3 | 24 |
| Cat depth8 | depth8,1000trees,l2=3 | 24 |
| Cat long4000 | depth6,4000trees,l2=3,final checkpoint only | 24 |
| Cat l2_10 | depth6,1000trees,l2=10 | 24 |
| ResNet128 | three residual MLP blocks | 24 |
| PLE128 | TRAIN-only piecewise linear scalar encoding, two-layer MLP | 24 |

Each row means two forms directF/residual(F-GP), seeds1701/1702/1703, four units. Total144full fits; no extension. Cat fits are fixed iterations,learning_rate.03,RMSE,border_count128,GPU0,thread1,gpu_ram_part.6,allow_writing_filesFalse,use_best_modelFalse. Other installed parameters/version captured. No VAL-based early stopping or snapshot choice in Cat; long4000 is final only.

Neural input normalization is rule-balanced pooled TRAIN mean/std floor1e-8, target normalization weighted TRAIN mean/std floor1eV per form. Float32, no AMP/TF32, AdamW lr.001,weight_decay.0001,batch8192,max80epochs,patience8; earliest strict min equal7rule physical VAL MSE. Each fit has180second measured epoch-loop budget, checked after complete epoch and VAL; budget can overrun by one epoch. At least one complete epoch required. Stop reason/actual time/epoch/best epoch all retained; budget truncation is disclosed, not a fully optimized architecture comparison. Same fixed procedure for both architectures.

ResNet: Linear(d,128),GELU then three blocks each Linear128,GELU,Linear128 followed by sum divided by sqrt2, then Linear128->1. PLE: each scalar has17 TRAIN rule-weighted midpoint-CDF quantile knots; scale knots using same TRAIN mean/std; concatenate16 clipped linear ramps per feature, denominator floor1e-6. Linear(d*16,128),GELU,Linear128,GELU,Linear128->1. Exact GELU. No future observations, feature-specific TEST quantiles, batch normalization or scenario ID.

## Smoke, queue and resource bounds

One GPU worker serially. TRAIN512prefix Cat10trees and one neural epoch per architecture are the only smoke fits; no VAL or TEST prediction in smoke. Run independent tree/functional replay on these real TRAIN prefixes before production. Smoke outputs are not scientific candidates. Full Cat96 fits then neural48 fits, global all144normal-exit barrier, VAL selection, selected-family TEST inference, independent audit, publication. No fit retry, auto-expansion or resuming partial fits. A failure blocks dependent stages and is preserved; never treat a partial family as selected. No completed older batch is rerun. Other processes are never stopped; idle GPU gate before launch.

Author's overnight execution is bounded at08:00 KST2026-11-26 for starting/continuing scientific GPU stages; owned child is stopped at this emergency boundary and entire run marked incomplete. CPU independent audit may finish by09:00. Controller/children are hidden Interactive/Limited scheduled task; no elevation, no OS security/power changes. Atomic receipts, exclusive launch marker, no scheduler restart-on-failure,15second live heartbeat; completed fits preserved. No guarantee that a sleeping/shut-down machine executes.

Explicit CES treatment pins: DROP_STUCK_TARGETS1,TI_SPIKE_CUT_EV3000,PER_SHOT_NORM0,SEQ_NORM_EXPANDING0,SEQ_ZERO_MC1,SEQ_ZERO_VT_HISTORY1,SEQ_ZERO_VT1,SEQ_DEVICEcuda; prefixes CES_ as in common114. F/MIXTURE7 recorded. inherited B112 provenance bound and B113 baseline archive/model/prediction audit checked. No old running code edited.

## Selection BEFORE TEST

For EACH outer unit and family choose ONE configuration plus form, using only that unit's own seven VAL rules. Across-design VAL cannot be pooled because outer TEST rows overlap other designs' VAL rows. For every candidate compute physical MSE of arithmetic3seed ensemble on each of7VALrules and divide by causal-GP MSE on those same rows. Score is equal mean of seven ratios. Select smallest score; exact ties broken lexicographically by configuration/architecture then form. This is standard within-outer-unit nested selection; configurations may differ by validation design. No perrule choice and no other design's VAL labels enter a unit's choice. No TEST score is read during selection. Independent selection validates VAL roles/idx/y/gp, same-unit disjoint TRAIN/VAL/TEST and seed averages, stores all scores/cell contributions and VAL hashes. Family selections and optional lower-VAL-score family are locked per unit in selection.json before any selected TEST inference. Existing B113 remains a reference, not retrospectively included in this search, because comparable all-VAL predictions were not previously exported. All144fits must complete normally first.


## Evaluation and judgement

Only each unit/family's selected configuration/form is inferred on all32unit/rule/F TEST supports; three seeds and fixed ensemble. Compare selectedCat/selectedNN against matched-form saved B113Cat ensemble, matched HGBdirect/residual,GPfast,GPfairmixture,GPonly. Original extra reference columns retained. Other candidates reported as VAL search table only. All four designs side by side, none primary. Whole selection reuses previously inspected TEST and is exploratory; percentile intervals condition on frozen selection/fits and omit search/refit uncertainty. GPU CatBoost is nondeterministic at fixed seeds. Keep initializations separate, never treat them as independent datasets.

Physical MSE/RMSE, skill=1-MSE(candidate)/MSE(reference), paired shot-cluster and separately discharge20000draw95percentile intervals; stable seed20261126 plus CRC32(cell/group). Row-weighted estimand retained on group resampling. Report every result with named comparator and zero-exclusion. R1 selectedCat vs matching B113Cat lowercluster skill endpoint>0 in EVERY D10design. R2 selectedCat vs GPfast same. R3 selectedNN vs GPfast same. All-scenario intervals descriptive/pointwise; no multiplicity-corrected universal claim or automated adoption. 10ms latency is not measured here.

Positive sentence: 'The VAL-selected candidate has a positive fit-conditional increment against [named comparator] under [specified designs]; unused-observation confirmation remains required.' Otherwise: 'A consistent increment for this registered search was not established; a matched adequate-budget comparison on unused observations could overturn this result.' No architecture impossibility, new physics, journal acceptance or algorithm originality claim.

## Independent verification, records and API status

All original manifests/source bytes/input and support hashes verified, TRAIN weights reconstructed. Independent selectedCat JSON-tree traversal (no CatBoost predict call), independent functional neural forward from stored weights without importing producer architecture, all48neural TRAIN normalization/PLE knots and checkpoint histories verified; selected models' seven-rule VAL predictions independently forward-replayed. Replay tolerances FIXED before fits: tree atol1e-5eV,rtol1e-7; neural CPU/GPU float32 differences atol.01eV,rtol1e-5; actual maxima reported. Replayed selected-neural VAL per-rule mean MSE versus best-checkpoint receipt uses rtol1e-4,atol1e-3eV^2, fixed before fitting. Smoke replay is mandatory. Independent per-unit VAL selection recomputation and all TEST predictions/ensemble/MSE/intervals verified before publishing docs/paper/b114_comparison.json/md. Independent code review before launch,89pytest plus records checks. No scientific output published on audit failure.

Official Claude preference remains, but the immediately preceding bounded official attempt failed HTTP401; no credential changes or blind repeated paid attempts. Preserve that provenance in docs/paper/b113_claude.md and ignored data/.b113_review; no completed Claude review is claimed for B114. Independent agent review and numerical checks proceed. Primary implementation references: https://catboost.ai/docs/en/references/training-parameters/common and official PyTorch AdamW/GELU documentation; APIs checked, installed versions captured.

Wiring: common114.py -> cat114.py/models114.py/neural114.py -> all_fits_complete -> selection114.py -> selected predictions -> audit114.py -> close114.py. queue114.py owns serial subprocesses, logs/status/deadlines and final closeout; docs/paper/b114_status.md and b114_overnight_plan.md are read by rootREADME/HANDOVER and experiment index. Raw arrays/models/attempts live ignored data/.b114; never commit data or secrets. Commit/push verified final records; do not modify original manuscript or estimator automatically.
