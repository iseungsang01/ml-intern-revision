# B.88 implementation provenance

Official Anthropic Messages API, `claude-opus-4-6`, thinking disabled. Three bounded code-generation requests used the existing read-only B.85 credential helper; the key remained runtime-only. Requests contained a redacted registered protocol and necessary source contracts, no author identity, raw observations, credentials or unpublished performance.

| Request | Input tokens | Output tokens | Stop |
|---|---:|---:|---|
| core | 1124 | 3152 | end_turn |
| runtime | 1843 | 9500 | max_tokens |
| readout | 2168 | 8500 | max_tokens |

Exact prompts, responses and usage are retained locally in ignored `data/.b88/claude/{core,runtime,readout}.json`. There was no paid retry. `claude_author88.py` records the requests and saves replay candidates under the ignored cache, never overwriting reviewed execution sources.

The core model originated in Claude's response. Review repaired repository path handling, source-pin validation, the statistics key/signature, A/C variant propagation, and the data gather. The response incorrectly treated the set of all anchor locations as a rolling history; the reviewed builder instead directly gathers the registered per-row `anchor[gi]`, checks strict prior/same file/segment and uses the identical current-channel standardization. It preserves the frozen builder's float64 arithmetic intermediates from list statistics before float32 storage, with an exact equality assertion against the already normalized source row. Masks retain their original boolean dtype.

The runtime and readout responses were truncated and proposed unwanted files and incompatible implementations. Their invented training loop, context cap, optimizer, model initialization, anchor search and completion logic were rejected. Minimal local adapter/CLI/supervisor code delegates directly to the frozen B.85 `train_unit`; it changes only local callback globals in a unique imported module. The new prediction and readout plumbing was locally rewritten using the verified data/model interfaces and frozen B.85 pure bootstrap/metric functions. Thus Claude was used for three authoring attempts, but the final implementation is a reviewed hybrid, not wholly Claude-authored. The rejected drafts were never used for training or scientific readout.

Windows process liveness uses query-only process handles, never `os.kill(pid,0)`. The dependency verifier uses a second unique, unpatched trainer import, keeps `common85.CACHE` at `.b85`, and requires the queue's complete state plus all registered B.85 clean-exit/config checks. Own runs, locks and artifacts stay in `.b88`. The source hash includes all B.88 execution modules, frozen dependency hashes and model metadata; the authoring helper and independent lead checks are excluded from execution hashing.

CPU real-input audits and 89 repository tests passed. No B.88 fitted-model scientific score or GPU training was run during implementation. GPU smokes and the registered batch remain conditional on B.85 completion and independent readiness approval.
