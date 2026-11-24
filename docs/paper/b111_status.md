# B111: full LSTM post-correction/scenario comparison

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
