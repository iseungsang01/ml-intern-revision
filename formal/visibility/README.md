# Latest-visible source selection

This project implements B.97's discrete contract in Lean4 v4.36.0, using only
the bundled Std library. It has no mathlib dependency. `Visibility.lean` contains
the executable selector and proofs; `Main.lean` calls that same selector on
local eligibility bit strings supplied by the registered real-trace bridge.

Run from the repository root on this Windows machine:

```powershell
py ces_prediction/experiments/b97_lean/toolchain97.py
py ces_prediction/experiments/b97_lean/run97.py
py ces_prediction/experiments/b97_lean/lead_checks/verify97_paths.py
```

`run97.py` refuses to replace a completed result. The registration precedes the
real-data run; do not change its selection to resolve a failed comparison.
Compiler troubleshooting before the first real-data run is permitted.
Addendum1 preserves the original verifier and fixes native/POSIX hash-dictionary
key separators in a separate adapter; the initial gate failure remains recorded.

Proofs cover any Boolean eligibility predicate on natural indices: bounds,
eligibility, greatest eligible index, absence, prefix stability, and selected
readout noninterference. Instrument visibility metadata and finite-value flags
must be exported correctly. The independent finite bridge checks those exports
and the existing Python selector on fixed actual TRAIN/VAL examples. This is
not a universal proof of NumPy, preprocessing, a neural network, CUDA execution,
acquisition timing, or physical temperature truth. The native executable also
depends on the compiler/runtime; Lean kernel checking of proofs and running
compiled code are distinct trust boundaries.

Expected evidence: `data/.b97/build.log`, `axioms.log`, `queries.json`,
`bridge.stdout`, `result.json`, `independent.json`, and the aggregate report
`docs/paper/b97_lean.json`. Raw trace information stays in ignored local data.
