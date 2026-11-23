# B.101: repair a readout namespace collision, no new fits

2026-11-23, before repair/readout/audit execution. Continued evaluation is
authorized. B.100 readout completed at21:04 but the original independent post
audit stopped on a main-pair metric mismatch. Read-only source inspection shows
TCN code TX and the supplementary tree TX share a metrics dictionary key:
the latter overwrites the former in128 of136 cells. Failed output is not an
audited scientific result. Preserve all original outputs and failure evidence.

No training, inference, selection, support, target, bootstrap, RNG, frozen
source or deadline change. Preserve all368 prediction hashes. Prefix all
supplementary tree keys TREE_ and recompute only overwritten TCN marginal
metrics using the unchanged metrics implementation and original RNG names.
Reuse other completed statistics and measured latency exactly. Re-enter the
original readout renderer with a process-local cell adapter. Require equality
of all unaffected report content, paired skills, statements and fit budgets.
The original independent post audit is mandatory and unchanged. No automatic
retry after another failure. Keep a hash journal before modifying outputs.

Success: repaired output passes original independent point/paired-bootstrap
audit, reports are copied only afterwards, queue/supervisor exit0, and existing
Claude gate passes. Failure: preserve artifacts and stop without scientific
interpretation. Neither outcome adds a scientific paper claim; use original
B.94 statements only after verification. Expect less work than the preceding
approximately36-minute full readout because only overwritten marginal metrics
are recalculated; this is an expectation, not a measured completion estimate.

Use one bounded official Fable implementation review, adaptive mode, maximum
6000 output tokens. Persist request/response/error/usage. Send code and explicit
structural facts only, no partial TEST numbers. The existing post-evaluation
Claude workflow remains capped at16 substantive rounds and starts only after
the original audit/normal-exit/hash gates. TEST-informed planning is exploratory.
