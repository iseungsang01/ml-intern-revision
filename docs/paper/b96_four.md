# B.96 author-selected four-worker production

Four global GPU slots are now the production capacity. This is an author-selected throughput/memory/per-fit-progress compromise; the previous10percent proof threshold did not establish that two was optimal. The original B.95 decisions remain unchanged.

The existing clean four-worker full-workload cohort completed in 127.02seconds, 1.889jobs/minute, with peak 3373MiB GPU memory. Its throughput was 3.82percent above the faster two-worker reference and 3.79percent below the slower six-worker reference. These are observed timings; no confidence interval or global optimum is established.

The two in-flight V3 C/A fits terminated normally and their original controller published normal-exit attestations before retirement. No trajectory was killed or restarted at this transition. Completed pilot fits now number eight; the remaining four pilot jobs are dispatched under the new capacity. Numeric sources, fixed budgets, seeds, data and validation/TEST gates are unchanged.

Live status: `py ces_prediction/experiments/b93_gpu/status93.py --cache data/.b94`. New task: `B93_GPU_B94_four96`. Policy: `data/.b94/dispatch_policy96.json`. New configurations explicitly distinguish global capacity from local groups of at most two; read `global_gpu_worker_limit` for the total. Historical95 configurations retain their own original policy.

Sources: `data/.b95/decision95.json`, `data/.b96/policy96.json`, `transition_complete.json` and the preserved fit exit/configuration/checkpoint hashes. The JSON companion contains the complete policy and transition source inventory.
