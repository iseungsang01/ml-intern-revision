# B.94 parallel GPU execution status

**Infrastructure revision before pilot:** the first task returned1 after WinError5 on atomic queue-status replacement. C smoke had ordinary exit0; A produced a checkpoint without an exit attestation. No pilot or TEST run started. All B.94 processes ended and the complete cache was archived at `data/.b94_failed_20261123_0648`; the old task is disabled. Addendum1 explicitly registers bounded metadata-persistence retries, fresh readiness/smokes and task `B93_GPU_B94_r2`, retaining the original deadline. This is not a retry of a scientific fit. The actual Windows handle-lock probe passed: save succeeded after a real sharing lock was released, and a held lock caused bounded failure while preserving the original target. `data/.b93/atomic_plumbing/checks.json` is the timestamp-only plumbing evidence. The specific holder of the original failing lock remains unidentified.

As of 2026-11-23 06:52 KST, the corrected isolated B.94 queue has been launched through the one-shot hidden Task Scheduler task `B93_GPU_B94_r2`. The existing CPU_r2 queue continues separately. The corrected independent preflight audit passed with no TEST construction; source pins and adapters were committed before launch. Code commits: `23a218d`, original-clock-continuation audit correction `614513b`, and metadata persistence correction `1ef4f49`. The current supervisor and queue identities are read from the live status command below rather than fixed in this document.

The runtime evidence is `data/.b93/runtime_decision93.json` and the independently recomputed `data/.b93/lead_audit93.json`; see [B.93 audit](b93_lead_checks.md). Both native CUDA paths completed three real full epochs, and two independent jobs on the selected dense native path also terminated normally. The registered aggregate-throughput criterion selected two workers. This evidence is a workload measurement under concurrent CPU training, not a whole-suite speed guarantee or predictive result.

The queue prepares and independently checks real TRAIN/VAL inputs, requires three sequential production-dropout normal-exit smokes, then dispatches the twelve pilot fits in two independent GPU processes. Only validation can choose the subsequent pair; confirmation and TEST readout remain gated. CPU and GPU fitted models are never pooled. The original suite deadline is retained through the verified clock continuation; launching B.94 does not reset it.

Read the current stage, exact process creation identities and heartbeat:

```powershell
py ces_prediction/experiments/b93_gpu/status93.py --cache data/.b94
```

`data/.b94/queue_status.json` reports active worker slots/PIDs and completed counts; `supervisor_status.json` reports durable process liveness. Consult both rather than interpreting an old stage string as ongoing computation. Logs are `durable.stdout.log`, `durable.stderr.log`, readiness/audit logs, and per-fit `process.log`. A production fit counts as complete only after its checkpoint/configuration hashes and ordinary child exit0 are validated. No automatic retry is enabled.

The current claims and their limits are reviewed in [the robustness and related-work report](claim_robustness_20261123.md). No B.94 TEST performance or physical discovery has been established by starting this queue.
