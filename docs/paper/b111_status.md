# B111: full LSTM post-correction/scenario comparison

**AUTHOR PAUSED,2026-11-24.** The author requested temporary suspension after
actual T4/CUDA was successfully rechecked (torch2.14.1+cu130,Python3.13.16).
The selected T4 setting had to be applied with Save; the earlier CPU checks
were real old-backend observations,not evidence that the author had failed
to select T4. No new B111 OOF inference,post fit or TEST readout has launched.
No upload through the new native route has started. All20 selected checkpoints
and the exact private archive remain local. Colab_B111_watch is manual/Ready;
do not start it,upload or launch until the author explicitly resumes.
Native upload/launch addendum3 and helper were committed as e1a4275 before
use. The MCP bridge's registered task limit was extended to8h without a
restart to avoid losing results during this work; it is a connector,not an
automatic scientific queue. Previous failed transport/inspection evidence
remains preserved. Local GPU remains paused as previously requested.

**AWAITING COLAB KERNEL RESPONSE,2026-11-24 16:31KST.** No B111 scientific
inference or post fits have started. The original transfer's last acknowledged
size is72351744bytes; run_code_cell then timed out, while notebook read/edit
still works. A short inspection saw the prior chunk's output rather than a
fresh kernel response. Addenda1/2 preserve both attempts, register exact-prefix
1MiB recovery, and reject stale output fields. Sources000883c,4b1fda2 pushed
before use. Neither failed inspection accepted a prefix or wrote new bytes.
Current GPU/workspace state needs fresh acknowledgement; do not assume it
was lost or reset it. The author has been asked to inspect the runtime and
manually run the harmless B111_KERNEL_PROBE_ONLY cell. Observer task remains
Ready,not started; do not run another operator start or fit locally.

Numerical preflight also passed on512 real TRAIN rows of an already stored
HGB model: sklearn prediction, producer numeric-tree traversal and independent
tree traversal agree. No new fit or TEST score in that check. All89 tests pass.
After kernel execution is restored: run the registered recovery with explicit
--resume-transport; accept only valid T4/no-launch/prefix-SHA evidence, finish
the exact archive, then start Colab_B111_watch only once poll_cell exists.

**EXPORTED / UPLOADING,2026-11-24 16:22KST.** Complete executable code and
registration070896a were pushed before export. Real CPU checks passed:
23065 cached D10 segments and432 original-builder segments are bit-identical
to the shared representation; all20 original/portable model prefixes agree.
Private archive149990972bytes,SHA256
eeedc688f1e48222938072b741223f0c3703c880d27a6a8cdc7b159cfd0ab334.
T4 is verified; scientific inference/post fitting has not started at this
snapshot because the official MCP archive transfer is still in progress.
The prepared hidden Colab_B111_watch task must start only after the operator
has written poll_cell and launched the remote frozen controller.

Uses twenty selected B109-audited LSTM checkpoints without retraining them.
Sixteen held-file replays -> eight HGB15 OOF post fits -> four outer replays
over eight scenarios ->32 common-row F MSE cells -> independent audit.
Two post policies: frozen_D10 and mixed_post, both retained without selection.

Registration PREREGISTRATION_B111.md; code b111_full_compare/. Delivery and
observer: research/colab_b111_operator.py,watch_colab_b111.py. Private payload,
predictions,models and audit evidence data/colab_b111. Live state is in
data/.colab_connection/b111_progress.json and b111_watch.json after launch.
Local GPU stays paused; no local fit or Claude call. Commit/push before export.
