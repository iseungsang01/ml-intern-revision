# B.85 implementation provenance and readiness history

The author authorized Claude API use and GPU work in the active session. Credentials were read at runtime from the named sibling repository and were never printed, copied into this repository, or transmitted as prompt content. Calls used the official Anthropic endpoint. Only methodological protocols and relevant source code were sent; no author identity, raw diagnostic arrays, or unpublished experimental error values were included.

## Claude authorship and local review

Claude authored the initial implementation. Seven complete source files were recovered from a response truncated during the eighth file. A targeted Claude request then replaced the actual input audits, training implementation and prediction implementation. The last request supplied the missing readout, but that response was also truncated and its proposed pooling/bootstrap/latency code contained substantive contract violations. It was not executed.

Local code review corrected import aliases, selected-unit interfaces, the exact validation RNG, serializable statistics, normalized target units, strictly prior availability anchors (including missing values at an S cutoff), actual project environment variable names, reuse of the inherited Ti modules, and removal of an unregistered extra dropout. Actual assertions replaced non-enforcing audit placeholders. The trainer's smoke uses real segments only. Completion/source metadata checks, atomic artifact writes, process-exit recording, and fail-fast queue supervision were added locally. These are implementation corrections, not new fitted variants or test-driven hyperparameter choices.

The final readout was locally corrected using the previously verified additive bootstrap mechanics: disjoint folds are pooled by row index, each model initialization is retained separately, an initialization is sampled first, and resampled cluster multiplicities weight per-unit sums. Median-error intervals use the existing weighted-quantile implementation. The original generated fold intersection, fold averaging, Boolean-mask set conversion, multiplicity-losing resampling, ensemble age summaries and synthetic latency inputs were rejected. This readout passed a separate parent audit using existing real B.87 predictions: pooled row handling, paired initialization-first bootstrap multiplicities, weighted median intervals, statement logic and the incomplete-experiment guard were checked before the autonomous queue was launched. No B.85 scientific errors have been read.

## API records

| Purpose | Returned/selected model | Input tokens | Output tokens | Outcome |
|---|---|---:|---:|---|
| Initial automatic newest-model review request | selected `claude-opus-5-5`; returned metadata not retained | unknown | unknown | no text; not used |
| Successful methodological review | `claude-opus-4-6` | 1238 | 3030 | complete; explicit thinking disabled |
| Initial implementation | `claude-opus-4-6` | 5358 | 24000 | output cap reached; seven complete files recovered |
| Targeted implementation repairs | `claude-opus-4-6` | 24582 | 16000 | output cap reached; three complete replacements recovered |
| Missing readout | `claude-opus-4-6` | 1633 | 7000 | output cap reached; unsafe generated readout locally corrected |

This is not an exhaustive usage total: the first unsuccessful request's usage was not retained. Exact safe prompts, returned code and usage are preserved locally in ignored `data/.b85/claude_review.json`, `claude_implementation.json`, `claude_repairs.json` and `claude_readout.json`; no credential appears in these records. The successful review prompt and response are also in `b85_claude_design_review.md`.

## Failed attempts preserved before training

- The first local label audit compared normalized labels directly with raw labels. It correctly blocked readiness; the checker was fixed to denormalize with frozen training statistics, while the independently verified normalized dataset stayed unchanged. The failed report is retained as `data/.b85/input_checks_failed_label.json`. All input audits then passed.
- The first real CUDA smoke failed the fixed step-vs-batch tolerance before an epoch. TF32 was disabled, with no tolerance relaxation, under preregistration addendum 2. Its log is `data/.b85/logs/smoke_failed_tf32_X.log`.
- With TF32 disabled, X passed finite forward/backward/gradient, recurrent-step equality, reset and save/load checks, and completed one real-data smoke epoch. The training child nevertheless returned nonzero after writing its completion artifact. The original launcher recorded only that it was nonzero, not the numerical child return code; the launcher itself exited 1. Root cause is unresolved. A later skip/metadata-check process exited 0, which does not establish a clean training-process exit. The artifact was therefore quarantined under `data/.b85/quarantine/smoke_X_unclean_20261122` and is excluded from the clean smoke gate. Future child return codes are recorded explicitly.
- An unrelated `llama-server.exe` subsequently occupied most GPU memory. It was not stopped or modified. The guarded queue waits for sufficient free memory, requires fresh X and H smoke processes to exit 0, and only then may enter the registered 90-run batch. A new nonzero child exit stops the queue without blind retry.

Current scientific status: registered and implemented; full experiment is not complete. Runtime truth is in `data/.b85/queue_status.json`, `docs/paper/b85_smoke.md`, `docs/paper/b85_status.md` and per-run logs. Input checks and computational smoke checks are distinct from successful full-run validation and from scientific conclusions.
