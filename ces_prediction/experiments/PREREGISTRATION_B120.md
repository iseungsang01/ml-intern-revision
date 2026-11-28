# B120: one sufficiently learned TCN35 optimization pilot

Registered 2026-11-28 before any scientific execution. Author explicitly requested execution of the focused TCN training experiment. This is TEST-informed exploratory planning, not unused-data confirmation.

## Question and fixed candidate

Does additional optimization of the existing B105 TCN35 improve its original TRAIN/VAL selection objective? One fixed first unit V1_r2_k0 and seed1701; not chosen by TEST or by greatest apparent training deficit. The other designs V2_s1_k0, V3_q3, V4a are NOT RUN; none is primary. No architecture search, model switching, GP input, boosting fit, new observations or TEST construction/predictions. The existing B119 job is neither resumed nor stopped by this workflow.

Use the exact frozen B105 AnchorTCN,35 inputs, original normalized F/L5 heads, cached real TRAIN/VAL splits and TRAIN-only statistics. Preserve batch16, segment-rule assignments/order, dropout, AdamW(.001,weight_decay=.0001), clip1, vectorized MSE plus original negative-temperature penalty, plateau scheduler(factor.5,patience2), equal-seven-rule/equal-two-head VAL mean, strict-best checkpoint selection and qualified B105 CUDA runtime. In particular this is not an F-only loss modification. Report F and L5 separately beside their original combined selection metric. K10 remains outside training/selection.

The old latest checkpoint lacks CPU/CUDA/NumPy/Python RNG state. Therefore execute one FRESH same-seed fit, not an exact continuation. First forty validated epochs must match historical B105 U0 train objective/VAL within rtol1e-5/atol1e-5, with exact update counts and scheduler learning rates. Any mismatch aborts without blind retry. Historical arrays/checkpoints/code remain immutable.

## Stop policy, bounds and recovery

Change only the stopping policy: minimum50 complete validated epochs; ten consecutive epochs actually trained with learning rate <=1e-5; at least twenty epochs since a relative0.1% improvement in the running significant-best VAL. All three required. This is an operational sufficient-learning criterion, not proof of a global optimum. Maximum200epochs and six hours of production loop time are safety caps, explicitly NOT_CONVERGED. Save strict-best weights and full validated model/optimizer/scheduler/RNG/history/significant-best/counters after every complete epoch, atomically. Do not select partial epochs. Preserve partial updates in total count and latest validated recovery state. No automatic retry or additional fit.

Qualification is real-TRAIN forward/stream/reset/padding/future-prefix mutation/save-load and finite backward with zero optimizer updates, separate process. Producer uses frozen loss/validator; independent CPU/NumPy audit recomputes every reported per-rule/head selected VAL MSE against source-bound cached labels and saved masked predictions, replays fitted forward on a fixed real prefix on CPU, and reconstructs stopping decisions/history/checkpoint hashes.

Queue waits at most twelve hours, requiring three consecutive resource checks with every GPU <1024MiB and <10% utilization AND no other experiment Python process. Memory alone is insufficient. Never terminate another process. Qualification maximum ten minutes; fit child maximum six hours plus thirty minutes preparation/export margin; audit maximum one hour. A hard child deadline may preserve only the preceding atomic epoch checkpoint. All owned launches hidden. Single-start receipt/locks prevent duplicate dispatch. No GPU deadline or fit retry reset.

## Judgment and interpretation

R1 sufficient-learning criterion met, distinguished from safety termination. R2 best combined VAL MSE decreases at least1% relative to original B105 U0 strict best, AND R1. Threshold is fixed before new execution. This is a descriptive optimization gate without a confidence interval: repeated VAL use is acknowledged. It does not establish lower TEST error or improvement over GP+fast. A successful gate motivates a separately registered fixed-checkpoint evaluation; this batch does not access TEST even on success. Report old/new full curves, epoch/update/LR/stop, physical head-wise MSE via TRAIN sd, original combined normalized selection metric, and all design statuses.

Positive paper sentence, only following later audited evaluation if supported: "The additional optimization reduced validation error; its effect on held-out prediction was assessed separately." Negative sentence: "Additional optimization did not establish a meaningful validation reduction under the registered policy; an independently specified evaluation on unused observations could overturn this conclusion." Neither sentence claims TCN impossibility, novel algorithm, physical truth or measured real-time latency. No manuscript performance numbers from this pilot.

Official Claude review was attempted once with redacted aggregate evidence before preparation; HTTP401, no response or advice obtained. Attempt/request/error retained in ignored data/.tcn_training_decision_20261128/. No paid retry. The working agent owns the design and independent source/numeric verification.

## Wiring and artifacts

common120.py -> frozen B105 provenance/prepared data/model; queue120.py -> train120.py --qualify -> train120.py -> audit120.py -> audited docs/paper/b120_training.json/md and status; ignored full state under data/.b120/. README.md and experiments/README.md list this wiring. Commit and push registration/source before dispatch; publish only after audit. Never stage data/ or unrelated work.
