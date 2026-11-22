# B.93 — production-epoch GPU runtime and concurrency qualification

Registered 2026-11-23 before execution. The author explicitly authorizes GPU work and asks for parallel execution. This is runtime qualification, not scientific model selection. Existing B.92 CPU_r2 training continues unchanged; unrelated processes are never stopped. B.92 sources, completed fits and original deadline remain unchanged.

## Inputs and scope

Use only the actual source-bound, SHA-verified B.92 prepared V1_r2_k0 data and C model, seed1701, actual dropout0.1, first full training epoch (all actual batches, unchanged ordering/objective/AdamW/clipping) plus all seven complete training-validation rules. Pin D1 through the existing prepared identity and immutable B.92 sources. No TEST inputs, accuracy selection, synthetic observations, prefix truncation or exported validation loss values. Bind actual batches, masks, initial weights, source hashes, raw data hashes and prepared/configuration hashes. No new CPU benchmark while the CPU queue is active.

## Statements and judgement

1. At least one cuDNN-disabled CUDA path can complete the actual workload reliably enough for a separate production qualification. Test packed native and dense native paths in three fresh processes each. Every repetition must have ordinary exit0, finite losses/gradients/parameters, identical actual data/order/initial weights and complete validation counts. Any nonzero exit or timeout disqualifies that path. Select the lowest median complete-epoch time among qualified GPU paths; choose packed when within5percent of the minimum. Expectation: native CUDA may be stable but the small-prefix benchmark may not predict full workload time.
2. Two independent GPU jobs improve aggregate throughput. On the selected path, run two additional fresh full-epoch children concurrently, repetitions30/31. Both must pass the same checks and ordinary exit0. Define conservative aggregate gain as twice the serial median epoch time divided by wall time for both simultaneous fresh jobs, including preparation. Authorize two production workers only when gain>=1.10; otherwise retain one. No promise of twice the speed. Expectation: contention can erase the gain.

Each child has a300second timeout; the entire qualification has a900second wall budget. Require at least2500MiB free VRAM before launching each GPU child. If budget runs out, record incomplete qualification rather than retrying. Record existing CPU queue/process state and resource contention. Because CPU training continues, report GPU timings under this contention and no CPU-versus-GPU acceleration factor. Never replace B.91 or B.92 runtime decisions or reuse these optimizer updates as scientific fits.

## Outputs and interpretation

`b93_gpu/profile93.py` launches `worker93.py`; artifacts and normal-exit attestations go to `data/.b93/probes/`, progress to `progress93.json`, and immutable selection to `runtime_decision93.json`. Independently audit sources, input membership, selected repetition artifacts, exit attestations, timing arithmetic and parallel threshold before a new scientific GPU revision. Any production dispatch requires its own registration and normal-exit production smokes; B.93 alone does not authorize changing an active fit.

Paper sentence if positive: "A separate full-workload runtime qualification selected a stable execution path and measured the aggregate throughput of concurrent fits; predictive conclusions use separately trained and validated models."

Paper sentence if negative: "The tested concurrent execution did not meet the registered stability or throughput criterion; runtime qualification does not establish a predictive limitation."

The measurements overturning a negative result are fresh ordinary-exit complete epochs meeting the same criteria under a separately registered runtime revision. No architecture originality, physical discovery, convergence, uncertainty interval for predictive skill or held-out performance is inferred from runtime probes.

## Supervision infrastructure

`durable93.py` and `status93.py` implement one-shot Task Scheduler supervision and process-birth-aware status. Existing CPU watch mode only observes; it never starts, stops or retries training. The prior timestamp-only plumbing test is infrastructure verification, not B.93 scientific/runtime workload evidence. See `docs/paper/b93_supervisor.md`.
