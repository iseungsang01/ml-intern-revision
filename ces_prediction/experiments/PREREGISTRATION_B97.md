# B.97 — Lean visibility selector assurance

2026-11-23. The author explicitly requested implementation after the Lean feasibility review. This is a deterministic assurance batch, not a predictive experiment. No model is trained, chosen, resumed or scored. GPU failure recovery remains a separate task. Only new files are used; frozen sources and prior registrations remain unchanged.

## Scope and statements

S1. Implement an executable Lean selector for the greatest eligible natural-number index strictly below a target index. Prove selected-index bounds, admissibility, maximality, absence iff no eligible source, prefix noninterference, and equality of selected readout under agreement on visible observations. The proofs must compile under pinned Lean4 v4.36.0, without `sorry`, `admit`, custom axioms or `native_decide`. Record `#print axioms` for every public theorem. Core/Std suffices; mathlib is deliberately not a dependency of this discrete contract. No result about OU/Bayes probability or PyTorch/CUDA arithmetic is claimed.

S2. Connect the compiled Lean function to the existing Python `data85.anchors` selector on actual TRAIN and VAL traces only. Select the fixed unit `V1_r2_k0`; select the lowest file ID of TRAIN and of VAL from its existing manifest. For each selected file, enumerate segments in recorded order and choose the first two segments having at least two rows. Within each segment choose unique local indices at floor fractions0,1/4,1/2,3/4 and the last row. Check all eight existing rules D10/D50/D100/D300/D650/D1000/S/K10. Use all preceding file rows as the query universe, masking different segments. This is a fixed finite bridge check, not proof of NumPy/searchsorted or all traces. No TEST segment or label is constructed. Fail loudly on missing data; no substitute or synthetic trace.

The independent Python reference uses a direct scan of actual eligible candidates, not searchsorted. Eligibility reproduces strict earlier index, finite CES, same file/segment, actual floating-point cutoff `time[target]-D/1000+0.003`, prefix-derived S gap cutoff, and the existing K10 frame cache. Check explicit source bounds, file/segment equality and latest eligibility separately. Preserve input/implementation/toolchain hashes and per-query results locally; publish only aggregate assurance metadata, not raw measurements or identifiers. No predictive errors or bootstrap interval apply to these deterministic assertions; no inference about the four validation designs is made.

## Judgement, expectation and stop conditions

S1 passes only if a clean Lake build and theorem/axiom audit succeed. S2 passes only if every registered query's compiled Lean result, independent scan and original Python result agree exactly and independent reconstruction verifies the artifact. Expect agreement but treat any mismatch as a failed assurance claim. Do not weaken a theorem or change sampling after readout to obtain success. A nontrivial repair or expanded scope needs an addendum before rerun; failed logs remain preserved. Compiler troubleshooting before the first real-data run is allowed and disclosed.

Do not equate Lean kernel checking with compiled executable correctness, actual instrument arrival times, model accuracy or independent physical truth. The compiled bridge relies on the Lean compiler/runtime, Python export, finite-value codec and correctly exported eligibility. Abstract noninterference assumes metadata and visible observations agree; it does not prove the existing full preprocessing or neural inference implementation.

## Both-outcome paper sentences and artifacts

If passed: "A discrete specification of latest-visible source selection was checked in Lean. A separately implemented finite comparison on registered training/validation traces agreed with the Python selector; this does not certify the complete inference implementation or physical acquisition latency."

If failed: "The proposed assurance bridge did not satisfy its registered checks and is not used as evidence of pipeline verification. A corrected specification/implementation bridge followed by the same registered comparison would overturn this result."

Deliverables: `formal/visibility/` Lean project and build/axiom logs; new `b97_lean/` runner and independent verifier; ignored `data/.b97/` artifacts; `docs/paper/b97_lean.{md,json}`, `b97_lean_checks.md`; wiring, handover and chronological record. No manuscript number or framing change.
