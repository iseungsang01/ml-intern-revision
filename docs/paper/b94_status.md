# B.94 parallel GPU execution status

As of 2026-11-23 KST, the isolated B.94 queue has been launched through the one-shot hidden Task Scheduler task `B93_GPU_B94`. The existing CPU_r2 queue continues separately. B.94's independent preflight audit passed with no TEST construction; source pins and adapters were committed before launch. Code commits: `23a218d`, with the original-clock-continuation audit correction in `614513b`.

The runtime evidence is `data/.b93/runtime_decision93.json` and the independently recomputed `data/.b93/lead_audit93.json`; see [B.93 audit](b93_lead_checks.md). Both native CUDA paths completed three real full epochs, and two independent jobs on the selected dense native path also terminated normally. The registered aggregate-throughput criterion selected two workers. This evidence is a workload measurement under concurrent CPU training, not a whole-suite speed guarantee or predictive result.

The queue prepares and independently checks real TRAIN/VAL inputs, requires three sequential production-dropout normal-exit smokes, then dispatches the twelve pilot fits in two independent GPU processes. Only validation can choose the subsequent pair; confirmation and TEST readout remain gated. CPU and GPU fitted models are never pooled. The original suite deadline is retained through the verified clock continuation; launching B.94 does not reset it.

Read the current stage, exact process creation identities and heartbeat:

```powershell
py ces_prediction/experiments/b93_gpu/status93.py --cache data/.b94
```

`data/.b94/queue_status.json` reports active worker slots/PIDs and completed counts; `supervisor_status.json` reports durable process liveness. Consult both rather than interpreting an old stage string as ongoing computation. Logs are `durable.stdout.log`, `durable.stderr.log`, readiness/audit logs, and per-fit `process.log`. A production fit counts as complete only after its checkpoint/configuration hashes and ordinary child exit0 are validated. No automatic retry is enabled.

The current claims and their limits are reviewed in [the robustness and related-work report](claim_robustness_20261123.md). No B.94 TEST performance or physical discovery has been established by starting this queue.
