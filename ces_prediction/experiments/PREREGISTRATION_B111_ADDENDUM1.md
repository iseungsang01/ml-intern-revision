# B111 addendum1: transport response recovery before scientific launch

2026-11-24, during archive delivery and before any B111 scientific child.
The original384KiB transfer stopped acknowledging near72MB. Keep the original
request/response logs and state; wait for the pending operation to return or
time out before continuing. This does not change the149990972-byte archive,
its SHA256, any scientific source, data, checkpoint, fit or judgement.

research/recover_colab_b111.py is a transport-only continuation. Preserve a
copy of the original state, replace the owned transfer cell's large source
with short inspection code before any get_cells request, and inspect actual
remote file size and SHA of the complete prefix. Do not trust local progress
as an acknowledgement, duplicate bytes, or inspect large-source get_cells
whose serialized duplicate may exceed the MCP websocket limit.

Resume only a prefix exactly matching the original local archive. Use1MiB
chunks (the earlier B109 operational recovery route) with exact size guards.
On a missing run response, inspect the owned cell's small outputs only after
cleaning large source; on ambiguity verify remote prefix before writing more.
A short inspection may read finished cell outputs after an MCP response timeout;
it never relaunches scientific work. If the runtime workspace was lost, retain
that evidence and permit one fresh delivery of the exact same archive only
after confirming no launch marker, no started.json, and zero scientific fits.
A second workspace loss stops recovery. No browser-security bypass.

Full archive SHA must pass before extraction and launch. Assert no launch
marker/started scientific state; use the original frozen run111.py and CES
pins. Record the recovery helper/addendum hashes separately from the unchanged
export manifest. Register a poll cell and use the original B111 observer and
independent audit. No second start, changed scientific source or automatic
fit retry. The helper's launch log accurately identifies OOF/eight-post/all-rule
scope; original operator's inherited zero-fits text described its launch instant,
not this batch's total allocation. Commit/push this addendum/helper before use.
