# B107 pre-handoff orchestration lifetime clarification

2026-11-24, before handoff or any B107 fit. Initial source preparation only
has completed; its provenance is preserved before refreshed preparation.
Defer termination of the suspended original queue until all adopted original
GPU children have exited normally. Keep its process handle during adoption.
This keeps the original Windows scheduled-task process hierarchy alive while
its GPU children finish; new children belong to the new controller hierarchy.
The original queue remains suspended and cannot dispatch pending work or run
old inference. Once no adopted child remains, terminate only the old queue
and record its deliberate exit. No GPU child is suspended or terminated.
This changes orchestration lifetime only, not the registered numerical policy,
model, allocation, statistics, stopping rule or final comparison.
