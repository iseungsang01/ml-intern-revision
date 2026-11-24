# B109: continue the nine unfinished LSTM paths on Colab T4

2026-11-24, before export/new training. Author stopped local experiments and
authorized Colab execution. B108 now qualifies cuDNN on actual Tesla T4:
all15 registered cuDNN children exited0 with finite checks and prefix agreement.
Select that eligible backend for the same LSTM comparator; no TEST result or
predictive score informs this execution choice. No Claude/API calls.

Keep all eleven patience-stopped B106/B107 paths unchanged. Four author-stopped
B107 paths resume from preserved validated epochs20,12,6,2, respectively:
V4a__inner1,V1_r2_k0__inner2,V2_s1_k0__inner2,V3_q3__inner2. Five unstarted paths
start from the exact original seed1701 Ti weights: V4a__inner2 and all four
inner3 paths. Fixed order: these four resumes then the five fresh paths.
Same split manifests,TRAIN-only statistics,F-only D10,original causal160x2
LSTM/dropout.1,AdamW.001/weight_decay.0001,clip1,scheduler factor.5/patience2,
negative-Ti penalty and batch16. Normal stopping is six consecutive epochs
without lower fully observed VAL MSE; preserve accumulated patience/history.
Emergency400 total epochs/24h per child only; a guard hit is incomplete,not
evidence against merit. Colab can preempt earlier; keep original local states.
No local GPU work or restart. At most four T4 workers,with1630MiB free memory
reserved per new worker; no other process is stopped. Four is an admitted
practical schedule,not a proven optimal concurrency (B108 single/four workloads
have different padded lengths). Default float32,no AMP/TF32,cuDNN benchmark off.

Export only each path's own actual TRAIN/VAL arrays and weights,anonymous tags
C00-C19; no file/shot IDs,raw timestamps,credentials,.env or designated TEST.
Remove the13 unused tensor channels,retaining exact first22 active columns.
Retain segment ordering/lengths and epoch random.Random(epoch*10000+1701) batch
membership. Model output/loss is unchanged by dropping unused columns; CPU
reference checks use the frozen original branch on each actual VAL prefix.
Data remain private ignored archives,never committed. Record original source,
case-manifest/statistics/cache/checkpoint hashes locally and archive SHA on
both endpoints. Cross-design membership remains case-specific; no pooling.

Restore resumed latest validated model,AdamW moments,scheduler,current learning
rate,updates,visible CPU/CUDA/Python/NumPy RNG,full history and best checkpoint.
Discard incomplete epoch updates at the explicit stop. Before new updates,
check GPU prefix agreement and full-VAL best-checkpoint MSE against historical
best (absolute tolerance1e-4*max(1,abs(best))). Abort on disagreement; no silent
fresh restart. Fresh paths use original exported initial Ti weights,not a
remote-version initializer. Recheck that mask's second head is always false.

Platform changes Windows/PyTorch2.11/nativeCUDA to Linux/PyTorch2.14/cuDNN.
This preserves optimizer state and the mathematical task,not bitwise gradient
trajectories. Visible RNG state is restored; opaque cuDNN dropout descriptors
are not serialized. This mixed-platform continuation is a disclosed protocol
amendment,not a pure architecture or uninterrupted reproduction claim.

Write atomic best/latest checkpoints every fully validated epoch. New latest
includes history,best value,patience,optimizer/scheduler and visible RNG. The
parent observes exit0,checks saved hashes/VAL arithmetic/patience,and preserves
every nonzero/timeout; stop allocating new paths after a failure and let active
children finish. Do not retry scientific children automatically. On nine
normal completions,archive all states,metrics,configs and hashes for local
retrieval in MCP chunks. A disconnected/preempted VM is incomplete; never
substitute missing paths or claim merit. Remote files are ephemeral; original
local stop snapshots are preserved until final retrieval is verified.

This batch trains only the nine remaining paths. It does not fit post-correction
models or read TEST. Subsequent honest OOF inference/corrections/eight-scenario
MSE require their complete registered executable evaluation workflow. Do not
publish a final comparator or final MSE before all twenty paths and evaluation
audits are available. Supported sentence: nine remaining trajectories reached
the inherited validation patience rule under the disclosed Colab continuation.
If incomplete: say which process/check failed; no architectural inferiority,
algorithm originality,new physics or acceptance claim.

New b109_colab_train/export109.py makes the anonymous private archive and local
case map; train109.py runs child/controller and validates all exits; the frozen
B108 portable model/loss is copied and hash-bound. research/colab_b109_operator.py
uses the existing official MCP connection to check T4,deliver hash-verified
chunks,start/poll and retrieve aggregate checkpoints without raw-data outputs.
Commit/push all sources before export or fit. State b109_status.md; evidence
data/colab_b109 and data/.colab_connection/b109_remote.json. Local pause remains.
