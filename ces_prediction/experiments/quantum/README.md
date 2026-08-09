# quantum — IonQ VQC vs a classical MLP (closed 2026-07-26, negative)

A **closed** controlled experiment, kept as the evidence behind the write-up in
`docs/ionq_qpu_실험기록.md`. It is not part of
the active pipeline; nothing in `ces_prediction/` imports it.

| file | what it is |
|---|---|
| `quantum_vqc.py` | trains a variational quantum circuit regressor (PennyLane) on PCA-reduced CES features; writes circuit params + PCA basis |
| `ionq_infer.py` | runs the trained circuit against IonQ (`qpu.forte-1`, falling back to the ideal simulator) |
| `quantum_vqc_result.json`, `quantum_vqc_weights.pt`, `ionq_simulator_result.json` | recorded outputs |

**Result: rejected.** Two findings, from the write-up:

1. **Hardware was never reached.** Credits were available but every Forte target reported
   `unavailable`, so QPU jobs were downgraded to the ideal simulator.
2. **Under a matched comparison the VQC lost to a classical MLP** — independent of the hardware
   problem, so it is the finding that actually settles the question.

Re-run only against the decision criteria in §5 of `docs/ionq_qpu_실험기록.md`.

Requires `pennylane` (not a project dependency — install ad hoc). Run from the repo root; both
scripts bootstrap `ces_prediction/` onto `sys.path` themselves, matching the other experiment dirs.
