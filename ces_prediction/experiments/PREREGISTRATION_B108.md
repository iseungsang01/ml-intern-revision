# B108: portable Colab T4 runtime qualification

2026-11-24, before export or execution. Author has a Colab T4 and asks how to
give it the local workload. This batch qualifies runtime speed/stability only;
it does not select a scientific model, estimate TEST error, or replace B107.
Local B107 jobs and frozen sources remain unchanged. No Claude/API calls.

Commit/push registration and all new sources before real-data export. Export
only registered B106 outer TRAIN/VAL arrays for the four fixed units, relabeled
A-D without file/shot IDs, timestamps, raw file names or credentials. Original
data remain local; the generated archive is private ignored data, never git.
Export the first batch16 from the original epoch0 shuffle (seed1701), first
VAL batch16, and longest16 TRAIN segments. Keep full real sequence lengths,
35-column tensors with the same22 active LSTM inputs and F-only target mask.
No TEST data or labels and no synthetic/mock observations. Missing data abort.
Use each immutable original B106 outer best.pt, not a running B107 artifact.

The portable class reproduces the original LSTM Ti modules exactly:22 inputs,
two160-unit layers,dropout.1,LayerNorm,64 GELU head,F and a masked zero head.
State names/shapes must load strictly; compare eval outputs with the frozen
original branch on actual64-row VAL prefixes on CPU (atol1e-6). Export those
reference outputs and actual TRAIN/VAL tensors in compressed NPZ. The portable
loss copies the frozen masked MSE plus negative-Ti penalty; independently check
its actual-input value and gradient against train92.vector_loss on CPU.

Run in a user-owned Colab T4 session. The notebook verifies archive SHA256,
safe flat member names, per-file hashes and actual T4 before running. No pip
upgrades, Drive mounting, git authentication, network API or background server
required. No automated data upload from the local workstation is performed.
The author uploads the generated zip to their private Colab runtime.

Compare float32 native CUDA and cuDNN enabled. Keep TF32/AMP off,cuDNN benchmark
off,threads2,AdamW lr.001/weight_decay.0001,clip1 and original loss unchanged.
For each backend: one worker(unitA) and four simultaneous workers(A-D),three
fixed fresh-process repetitions. All cases/attempts recorded, not retried or
filtered by speed. Each child starts from the bound original weights, checks
eval prefix agreement(atol1e-4,rtol1e-4),finite gradients,normal exit,then does
two warmup and five timed full optimizer steps on the real first TRAIN batch.
These are discarded timing fits, not model candidates or continuation of B107.
Use barriers before warmup and timing so four-worker windows are comparable.
Then measure one VAL forward and a longest-real-batch backward memory probe.
Require finite loss/gradients and normal parent-observed exit0 for every probe.
No probe checkpoint or row predictions are retained as scientific output.

Timeout600s per group and suite1800s are timing-test guards, not scientific
training stop rules. A timeout/crash makes that configuration unqualified;
preserve all stdout/stderr/exit records and partial results. No inference that
a model lacks merit. All registered repetition groups attempted unless the
whole timing suite guard is reached (then mark remaining unrun explicitly).

Report GPU/CPU/Python/PyTorch/CUDA/cuDNN,backend flags,batch shapes,source/data/
weight hashes,latency samples,median step time,peak reserved memory and normal
exit. Group throughput=sum(timed steps)/(max end-min start),with synchronized
timing gates; also report launch-to-exit wall time separately. Medians across
three fixed repetitions are descriptive,not an inferential speedup claim.
Backend eligible only if all its expected15 children exit0 and pass checks.
Small repeated-batch timing is not a full-epoch forecast or final MSE result.

Do not stop or benchmark concurrently on the busy local GPU. Local CPU export
and equality checks only. An exact local GPU comparison can run after B107
finishes with this same exported workload and protocol; until then, do not
assert a T4/local speedup based on GPU names or utilization percentages.
Changing execution backend for full scientific training needs separate
registration and whole-run numerical/reproducibility verification.

New b108_colab/portable108.py defines the portable model/loss; probe108.py is a
fresh CUDA child; run108.py controls barriers/attempts and aggregate outputs;
export108.py creates the hash-bound private zip, runnable notebook and local
provenance. Wiring/docs: README.md,experiments/README.md,b108_colab.md.
Outputs under ignored data/colab_b108: COLAB_T4.ipynb,ces_t4_bundle.zip,
export_verification.json. Colab downloads b108_results.zip with aggregate
timings/logs, no observations. No acceptance estimate or novelty claim.
