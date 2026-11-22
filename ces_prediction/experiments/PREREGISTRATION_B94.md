# B.94 — separately qualified GPU execution of the reduced candidate protocol

Registered 2026-11-23 before any B.94 model run. The author authorizes GPU execution and parallel fits. B.92 CPU_r2 continues independently. B.94 uses a new directory and cache, imports the frozen B.92 training loop and never edits active B.92 files or pools their fitted models with B.94. Preserve the author's five research questions and existing manuscript framing.

## Purpose, known facts and limits

Determine whether the already specified small candidate portfolio yields a validation-eligible model and a reproducible predictive increment under a stable GPU runtime. B.91 small-prefix timing did not measure production throughput; cuDNN production dropout failed ordinary process exit. B.93 tests native CUDA full epochs and concurrent throughput without predictive readout. No B.94 score is known. Expectation: GPU concurrency may increase allocation throughput; improved predictive skill and an eligible candidate are uncertain.

## Immutable scientific specification

Inherit B.92 and addendum1's real D1 data, complete segments, train-only normalization, shot manifests, per-target masks, causal availability, all seven equally weighted training-validation rules, objective, optimizer, batch16, dropout0.1, model sizes, seeds, scheduler, early stopping and candidate definitions. Explicitly set all CES data treatment and training environment variables in each child. Model C is B.88 with the measurement-time anchor embedding zeroed, A is the corresponding learned anchor model, T is the fixed B.90 TCN; C is not the original B.85 model.

Pilot: C/A/T on V1_r2_k0, V2_s1_k0, V3_q3 and V4a, seed1701, at most12epochs and600seconds per child. Use only complete validated epochs; a partial epoch is never the best. Eligibility for A/T requires mean of four best-validation-loss ratios to C<0.98 and both V3/V4 ratios<=1.05. If both eligible, prefer A when mean ratios differ by<=0.005; otherwise use the lower mean. If neither eligible, stop without TEST readout. Lock the validation decision and exact input artifact hashes immutably.

Confirmation: selected A/C pair or T-with-fast/T-history pair, fresh fits on all15units at seed1701 plus the four fixed pilot units at seeds1702/1703. At most46 additional fits,40epochs/1200seconds each. Report full15unit one-seed estimates separately from four-unit three-seed stability. Do not imply all15units were trained with three seeds. Maximum58 scientific fits, with at most three successful initial production smokes and two selected-pair smokes. Each smoke uses actual observations, dropout0.1 and ordinary exit0. No TEST readout before selection and all required confirmations and audits pass. No broad hyperparameter search, synthetic observations, test-based candidate selection or retries of failed fits.

## Runtime and parallel dispatch

Before any B.94 run, independently verify B.93 source/input/exit/result bindings and recompute the median ranking, packed-path5percent tie and concurrency threshold. Require a qualified native CUDA path with all three serial children clean. Use exactly its configuration and one or two workers according to B.93's rule; two workers require both concurrent children clean and aggregate gain>=1.10. Pin the B.93 decision SHA in B.94 configuration and freeze imported B.92/B.93 code plus the committed B.94 adapters and registrations. B.93 alone does not establish numerical parity; initial production checks must include the reused loop's model/data contracts before pilot.

Each fit is a separate process with its own RNG, optimizer, run directory and normal-exit attestation. Dispatch order follows the registered job list; completion order never determines candidate allocation. Prepare each unit under a cross-process lock and publish atomically; concurrent fits must see identical source-bound prepared inputs. No GPU data parallelism, altered batch membership, loss or hidden gradient accumulation. On any failed child, cease dispatching new jobs and let already-owned children terminate normally; no unrelated process is stopped. Runtime fallback requires another registration, never a mid-fit switch or fabricated CPU provenance.

Continue the original B.92 suite clock, started at the epoch stored in `data/.b92/suite_clock.json`, with its original24hour deadline; B.94 does not receive a fresh24hours. Use one-shot durable supervision with process creation identity and heartbeat. Incomplete allocation at the deadline is reported as incomplete. Do not combine completed B.92 and B.94 fits into a nominal complete allocation. Running both queues is resource contention, to be recorded; runtime-limited pilot/confirmation conclusions remain conditional on these budgets and reached epochs.

## Evaluation, output and falsification

Use the inherited B.92 readout: both F and L5 targets, all eight availability rules including held-out K10, all four validation designs, paired common rows and20,000shot-cluster bootstrap resamples with the registered seed construction. Report point estimates,95percent intervals and zero exclusion. Fixed-fit cluster intervals do not include refit, candidate-selection or new-campaign uncertainty. Separately measure CPU single-thread inference; neither training throughput nor inference timing demonstrates end-to-end diagnostic latency or new pedestal physics.

Code: `b94_parallel/`; cache and immutable records: `data/.b94/`; reports: `docs/paper/b94_*.{json,md}`. A separate checker revalidates source/configuration/checkpoint/input hashes, exits, selection arithmetic and final common-row/cluster readout before reporting predictive conclusions. Wire the adapters, checker and generated reports in README before launch. B.92 output files and existing manuscript numbers are not overwritten.

Paper sentence for a supported increment: "Under the registered reduced candidate allocation and fixed execution budget, the selected learned estimator's paired skill was [generated estimate and interval] on each validation design; the full-unit and multi-seed scopes are reported separately."

Paper sentence for an unsupported increment: "The registered candidate allocation did not establish an increment under its validation or confirmation criterion; this is conditional on the reached training budget and does not exclude a benefit after a separately registered longer fit."

A negative conclusion would be overturned by a new pre-registered sufficiently trained comparison whose paired interval excludes zero in the predicted direction on the specified validation designs and availability rules. No journal acceptance estimate, architecture originality or new physical mechanism is inferred from either outcome.
