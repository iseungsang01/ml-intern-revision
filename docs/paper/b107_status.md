# B107: LSTM continuation to validation patience

Live2026-11-24 12:10KST: registered controller is running. Sources a68c784,
4a77b6b and702acf6 were committed/pushed before successful handoff or new fits.
All89 existing tests passed. Actual cache/optimizer-moment/scheduler restoration
checks passed. At12:09:14 the new controller adopted four original GPU children
with matching PID/birth identities; zero GPU children were suspended or killed.
The old queue is suspended to prevent further old dispatch/readout, and will
be retired after its adopted children finish. The original suite clock is
preserved. New local watcher and supervisor heartbeat are active.

Four adopted paths continue unchanged; as slots become free, limited V1/V4a
outer models have continuation priority. New and continued paths stop by
six non-improving validation epochs rather than the arbitrary one-hour cap.
No B107 comparative MSE or merit has yet been published.

Pre-fit operational attempt: the first handoff safely aborted because an
observed conhost.exe console helper was classified as an unknown child; its
exception resumed the old queue without stopping any GPU child. Addendum2
documents excluding only that console helper. Failure/provenance attempts
remain under ignored data/.b107; no scientific fit was retried by this change.
B106 execution sources and original scientific artifacts remain frozen.
All twenty selected paths must meet six non-improving validation epochs before
final OOF correction fitting, scenario MSE comparison and independent audit.
No new Claude/API calls. See PREREGISTRATION_B107.md and b107_patience/.
Original RNG states were not saved; resumed dropout is a recorded new stream.
Emergency guards are not planned durations and cannot support a negative merit
conclusion if hit. Live evidence: ignored data/.b107 and pipeline107.py status.
