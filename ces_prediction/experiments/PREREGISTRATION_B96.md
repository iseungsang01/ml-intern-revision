# B.96 — author-selected balanced four-worker production transition

2026-11-23, before any four-worker production launch. The author explicitly questions the conservative two-worker choice and requests the four-worker compromise. The previous10percent threshold was a conservative proof-of-improvement rule, not a measurement of the optimum. B.95's clean four-worker cohort establishes feasibility but only one timing cohort; do not describe four as a demonstrated global optimum or repeat that timing search unnecessarily. The author's preference supersedes that operational threshold. Preserve the original B.95 decisions and all completed fits unchanged.

## What changes and what does not

Set global production capacity to four independent GPU fit processes, using the already-tested dense native CUDA configuration. Retain frozen B.94 numeric training, batch membership, seeds, model portfolio, objective, optimizer, per-fit epoch/time caps, data treatment, original suite deadline and validation/TEST gates. No new CPU duplicate queue or model search. Preserve all existing complete normal-exit fits; do not restart them. The two reboot-interrupted restarts remain exactly as B.94 addendum2 registered.

New source-bound orchestration lives only in b96_four. Each new fit records dispatch_sources96, dispatch_policy96_sha256, global_gpu_worker_limit4, global_worker_slot and the explicit local group/slot mapping used in B.94 addendum3. Original runtime.parallel_workers and scheduled_concurrency describe groups of at most two; global fields describe total capacity. Exit records explicitly retain both local and global slots. Historical95 records keep their own original provenance and two-worker policy; the new verifier validates each provenance against its original source/policy rather than rewriting history.

## Preserve running trajectories at the transition

Temporarily suspend only the old owned dispatcher and its supervisor, with Windows creation identities recorded, while their already-running trainer children continue unchanged. Start no new production child until these in-flight fits have terminated normally. A separate one-shot transition controller observes exact process handles and normal exits, resumes the old dispatcher briefly to publish its real exit attestations, then quiesces it before more jobs launch. Verify configuration/checkpoint/exit evidence. If a new child races the boundary, restore the old controllers and abort this transition without killing or restarting any fit. If an old child fails, restore its original controller so the existing failure policy applies; do not turn an interrupted result into a normal completion.

Only after all current children have normal exit attestations and no live trainer remains, retire the two old controller processes, disable their owned task, preserve old status/lock evidence, and start a new one-shot durable four-worker queue. Never stop unrelated GPU holders. The transition is operational and does not pause an optimizer mid-update, reset a fit budget, lose a validated epoch or rewrite an existing result. The original per-child hard timeout and global deadline bound the wait; restore the old controllers if the boundary cannot be reached safely.

## Validation and interpretation

Recompute the feasibility evidence from the original four-worker result/exit hashes, actual full batch binding, sources and software. Verify each newly completed fit's four-worker metadata and each historical fit's original metadata. Keep mandatory pre/prepared/pilot/post independent checks and immutable validation selection. Four workers are an author-chosen balance between observed throughput, memory and per-fit progress; no confidence interval or global optimum is established by the runtime measurements. Publish actual reached epochs/updates under the fixed budgets.

Scientific statements, expectations, both-outcome paper sentences and falsification measurements remain B.94's. This change is a dispatch preference, not predictive evidence or a physical discovery. A runtime failure stops explicitly without automatic scientific retry.
