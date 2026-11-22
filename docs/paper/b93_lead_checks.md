# B93 independent lead audit

The independent audit entry point is
`ces_prediction/experiments/b93_gpu/lead_checks/verify93.py`.
It never starts GPU work or a training/benchmark child. It does not import
`profile93.py` or reuse its selection implementation. Adding this file and the
verifier does not change the registered workload's source inventory.

Run from the repository root after the immutable decision appears:

```powershell
py ces_prediction/experiments/b93_gpu/lead_checks/verify93.py --output data/.b93/lead_audit93.json
```

An absent decision returns exit 2 with `state=pending`; a failed check returns
exit 1; a completed evidence audit returns exit 0. The audit checks every serial
slot, every recorded parallel attempt, failed and timed-out attempts, and slots
omitted for budget exhaustion. It verifies current frozen sources, B92 readiness
sources, prepared/configuration/raw hashes, result hashes, normal-exit attestation
hashes and artifact coverage. It rebuilds actual shuffled and padded input bytes
and label counts from the real prepared train/validation arrays, independently
of the production batch builder, and rebuilds initial weights on CPU without a
model forward or optimizer update. It checks the inherited source-bound rule
assignment, not a new rule schedule.

The verifier recomputes eligible serial medians, the packed-within-five-percent
tie rule, and the concurrency gain/worker threshold. It checks parallel wall
time against each child's reported preparation-plus-epoch duration, cross-checks
overall elapsed time against exported wall-clock records and reports the amount
by which recorded elapsed time exceeds the nominal wall budget. Runtime values
are not predictive scores and are not copied into the manuscript here.

## Evidence limits

The registered controller exports scalar elapsed and parallel-wall durations,
not launch/end monotonic timestamps per child. Therefore exact independent
reconstruction of the per-child timeout and total wall budget is unavailable;
the controller's deadline enforcement can be inspected in source and the
exported duration arithmetic can be audited. The concurrency wall's inclusion
of preparation is source-audited and bounded below by child durations, rather
than reconstructed from an independent external clock. Termination and JSON
finalization can consume time after the last deadline check.

Finite-gradient enforcement is checked in the worker source through
`clip_grad_norm_(..., error_if_nonfinite=True)`; gradient tensors are not exported.
The worker exports no scientific loss values. Failed artifacts are retained and
their presence cannot qualify a path with a nonzero or missing clean exit.

## Current state

The completed qualification passed the independent audit, including all serial
and concurrent records. The verifier process itself exited normally with code
zero. The rebuilt real input bytes and initial weights matched every successful
worker. No failed, timed-out, or budget-omitted slots were present in this run.
The independently recomputed choice is `cuda_dense_native` with two workers;
the packed tie rule does not change that selection. Recorded total elapsed time
is below the registered budget. No predictive result follows from this choice.

`data/.b93/lead_audit93.json` contains the decision hash, verifier hash, exact
recomputed medians, concurrency gain, budget accounting and evidence limits.
The corresponding raw results and exit attestations remain under
`data/.b93/probes/`; the immutable decision is
`data/.b93/runtime_decision93.json`. Any production GPU revision still requires
its own registration and normal-exit production smokes.
