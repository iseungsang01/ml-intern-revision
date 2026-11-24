# B107 pre-fit Windows console-host classification

2026-11-24, before any B107 scientific fit or successful handoff. Actual CIM
inspection showed conhost.exe as a direct child of the original queue, in
addition to its four known python training children. Exclude only conhost.exe
from the set of computational children that must match adopted training PIDs.
Do not terminate the console host or any GPU child. Unknown non-console
children still abort handoff. Original four training processes remain intact.

Stop/restart only the new B107 controller/readiness check before handoff, keep
its original suite clock and task registration, and preserve its initial
provenance/readiness attempt. Refresh source provenance after committing and
pushing this operational change, then restart the same registered controller.
No numerical training, source data, split, allocation or stopping rule change.
